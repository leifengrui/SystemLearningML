# dynamic shape

> **所属章节**: [[图捕获问题]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: dynamic shape / 动态 shape / symbolic shape / guard 失败 / 重编译 / shape specialization / dynamic=True
> **难度**: 高（需懂 [[TorchDynamo]] + [[torch.compile]] + [[graph break]]）


## 1. 一句话定义

**dynamic shape（动态 shape）** 是 `torch.compile` 中**输入 tensor 的 shape 在不同调用间变化**（如变长 batch、变长序列）的情景——[[TorchDynamo]] 的 **guard** 含 shape 断言（`x.shape == [4,4]`），shape 变 → guard 失败 → **重 trace + 重 codegen**（[[Inductor]] 重新生成 kernel），每个不同 shape 生成一份编译缓存。频繁变 shape → 反复编译开销大、缓存膨胀，可能负收益。**`dynamic=True`** 让 Dynamo 用 **symbolic shape**（shape 用变量 $s_0, s_1$ 而非常量），放宽 guard，生成支持 shape 范围的 kernel（一个 kernel 跨多 shape），减重编译与缓存。代价是 symbolic kernel 优化稍弱（需处理任意 shape，不能 hardcode tile）。dynamic shape 是 compile 在变长推理（continuous batching 变长 batch）与训练（变长序列 padding 后真长度变）场景的关键配置——理解何时用 `dynamic=True`、何时让 shape 静态化，是 compile 实战的核心抉择。

> [!note] 三句话定位
> - **是什么**：输入 shape 变化触发 guard 失败重编译；`dynamic=True` 用 symbolic shape 放宽 guard 减重编译。
> - **为什么**：变长 batch/序列场景 shape 频繁变，不处理则反复编译（开销 + 缓存膨胀）；symbolic 让一 kernel 跨 shape。
> - **代价**：symbolic kernel 需处理任意 shape（不能 hardcode tile/展开），优化稍弱于 static shape 专用 kernel。


## 2. 为什么需要它（动机与背景）

### 2.1 推理的变长 batch

continuous batching（[[continuous batching的调度实现]]）每步 batch 不同（请求来去），shape 变。若每个 shape 重编译，compile 开销爆炸。`dynamic=True` 让一 kernel 跨 shape，只编译一次（用 symbolic $s_0$）跑所有 batch size。

### 2.2 训练的变长序列

NLP/CV 训练的序列长度变（padding 后真长度不同，或 bucket 不同长度）。若每长度重编译，开销大。dynamic 让一 kernel 跨长度。但训练 batch size 常固定（static），只序列长度变。

### 2.3 guard 与重编译

Dynamo 的 guard 含 shape 断言（见 [[TorchDynamo]] 3.4）。shape 变 → guard 失败 → 重 trace + [[Inductor]] 重 codegen。每个不同 shape 一份缓存（`~/.cache/torchinductor/<hash>`，hash 含 shape）。频繁变 → 反复编译 + 缓存膨胀（GB 级）。

### 2.4 symbolic shape 的放宽

`dynamic=True` 让 guard 用 symbolic（`x.shape == [s0, s1]`，$s_0, s_1$ 是变量），guard 只断言"shape 是变量"而非具体值。shape 变（$s_0$ 从 4 到 8）guard 仍过（都是 valid symbolic）。只编译一次，生成处理任意 $s_0$ 的 kernel。

### 2.5 symbolic 的代价

symbolic kernel 需处理任意 shape：
- **不能 hardcode tile**：static kernel 可 hardcode `BLOCK=128`（shape 128 的倍数时最优）；symbolic 需 runtime 分块（`BLOCK` 配合 shape 余数处理）。
- **不能完全展开**：static 可展开已知 shape 的循环；symbolic 用 runtime 循环。
- **autotune 受限**：`max-autotune` 对 static shape 试 tile 选最优；symbolic 需选"跨 shape 都好"的 config（次优于 static 专用）。

故 symbolic kernel 性能稍弱于 static 专用 kernel，但避免反复编译。是 trade-off。

### 2.6 何时 static 何时 dynamic

- **shape 稳定**（训练固定 batch、推理固定 shape）→ static，拿最优 kernel。
- **shape 频繁变**（变长 batch/序列）→ `dynamic=True`，避免反复编译。
- **shape 有限几种**（bucket 几个长度）→ static 各编译一份（缓存几份），比 dynamic 优（每份是 static 专用 kernel）。


## 3. 核心概念详解

### 3.1 guard 的 shape 断言

```python
@torch.compile
def f(x): return x * 2
f(torch.randn(4, 4))   # 首次: guard x.shape==[4,4], 编译
f(torch.randn(4, 4))   # guard 过, 复用
f(torch.randn(8, 4))   # guard shape 失败 -> 重 trace + 重编译 (新 shape [8,4])
```

每 shape 一份缓存。shape 多 → 缓存膨胀。

### 3.2 dynamic=True 的 symbolic

```python
@torch.compile(dynamic=True)
def f(x): return x * 2
f(torch.randn(4, 4))   # 首次: guard x.shape==[s0, 4] (s0 symbolic), 编译 symbolic kernel
f(torch.randn(8, 4))   # guard 过 (s0=8 valid), 复用同一 kernel
f(torch.randn(16, 4))  # guard 过, 复用
# 只编译一次, 跨所有 s0
```

symbolic kernel 用 `tl.arange` 配合 runtime shape（`tl.cdiv(N, BLOCK)` 分块），处理任意 $s_0$。

### 3.3 symbolic kernel 的实现

```python
# Inductor 生成的 symbolic Triton (概念)
@triton.jit
def f_kernel(x_ptr, N, BLOCK: tl.constexpr):   # N 是 runtime 参数
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N                    # 处理 N 不是 BLOCK 倍数 (余数)
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, x * 2, mask=mask)
# vs static kernel (N 已知是 1024):
#   offs = pid*BLOCK + tl.arange(0, BLOCK)   # 不需 mask (1024 是 BLOCK 倍数)
```

symbolic 多 mask 与 runtime N 参数，稍慢于 static（mask 开销 + 不能 hardcode）。

### 3.4 symbolic 的 guard

```python
# dynamic=True 的 guard (概念)
# x.shape == [s0, 4]   s0 是 symbolic 变量
# guard 只断言: x.dim()==2, x.shape[1]==4, x.shape[0] 是任意 int
# 故 s0=4/8/16 都过
```

vs static guard `x.shape == [4, 4]`（精确值）。

### 3.5 mark_dynamic / 自动推断

```python
@torch.compile(dynamic=True)   # 全部输入的动态维自动 symbolic
# 或显式:
@torch.compile(dynamic_shapes={'x': {0: True}})   # 只 x 的第 0 维 symbolic
# Inductor 自动推断哪些维动态 (dynamic=True 全推断, 显式更精确)
```

显式 `dynamic_shapes` 指定哪些维 symbolic（其余 static），比 `dynamic=True` 全 symbolic 更精确（静态维仍可 hardcode）。

### 3.6 bucket 策略（有限 shape）

```python
# 若 batch 只有几种 (如 8/16/32/64), 不用 dynamic
@torch.compile   # static
def f(x): return x * 2
for bs in [8, 16, 32, 64]:
    f(torch.randn(bs, 4))   # 每 bs 编译一份 (4 份缓存), 每份 static 专用 kernel
# 比 dynamic 优 (每份 static kernel 性能更好, 只 4 份缓存不膨胀)
```

shape 种类少（<10）→ static 各编译；种类多（连续变化）→ dynamic。

### 3.7 dynamic 与 max-autotune 的冲突

`max-autotune`（[[Inductor]] 3.6）对 static shape 试 tile 选最优。dynamic shape 无法 autotune（shape 变，最优 tile 变）→ max-autotune 退化用默认 config。故 dynamic + max-autotune 收益打折。static shape 才 max-autotune 拿满收益。

### 3.8 dynamic 与 CUDA Graph 的冲突

[[CUDA Graph与graph capture]] 要求 static shape（graph 捕获的 kernel 序列 shape 固定）。dynamic shape 的 kernel shape 变 → graph 捕获失败或需多 graph（每 shape 一份）。故 `reduce-overhead`（含 CUDA Graph）与 dynamic 冲突，需 bucket（每 shape 一 graph）或放弃 CUDA Graph。

### 3.9 dynamic 与 graph break 的关系

dynamic shape 操作（如 `x.reshape(unknown_dim)`）某些场景触发 [[graph break]]（Dynamo 无法静态推 shape）。`dynamic=True` 让 symbolic 推 shape 减 break。两者协同减 compile 的兼容性问题。

### 3.10 torch.export 的 dynamic

`torch.export` 导出时可指定 `dynamic_shapes`，导出的 graph 带 symbolic shape（可跨 shape 部署）。是 export 在变长场景的配置。


## 4. 数学原理 / 公式

### 4.1 重编译次数

static guard：$S$ 个不同 shape → $S$ 次重 trace + 重 codegen。总编译 $\sim S \cdot T_{\text{compile}}$。

dynamic guard：1 次 trace + codegen（symbolic），跨所有 shape。总编译 $\sim T_{\text{compile}}$。

$S$ 大时 dynamic 省 $\sim S\times$ 编译。

### 4.2 symbolic kernel 的性能损失

static kernel：hardcode shape，tile 对齐，无 mask。throughput $\tau_{\text{static}}$。

symbolic kernel：runtime shape，mask，tile 配合余数。throughput $\tau_{\text{dynamic}} \approx 0.9 \tau_{\text{static}}$（mask 与分块开销）。

shape 种类 $S$ 少时 static 各份总收益 $S \cdot \tau_{\text{static}}$ 优于 dynamic 的 $\tau_{\text{dynamic}}$；$S$ 大时 dynamic 避免反复编译更优。

### 4.3 缓存膨胀

static：每 shape 一份缓存，总 $\sim S \cdot \text{cache size}$。$S$ 大 → GB 级。

dynamic：一份缓存（symbolic），总 $\sim 1 \cdot \text{cache size}$。

### 4.4 break-even shape 数

设 static 编译 $T_c$，每步省 $\Delta t$，dynamic 编译 $T_c$（同）但每步省 $\Delta t' < \Delta t$（symbolic 稍弱）。$S$ 个 shape 下：
- static 总开销 $S \cdot T_c + S \cdot N \cdot (t - \Delta t)$（$N$ 步，$t$ eager 时间）
- dynamic 总开销 $T_c + N \cdot (t - \Delta t')$

$S$ 大时 dynamic 省 $S \cdot T_c$ 编译，但每步少省 $\Delta t - \Delta t'$。break-even $S^* \approx T_c / (N(\Delta t - \Delta t'))$。$S > S^*$ 用 dynamic。


## 5. 代码示例（可选）

### 5.1 static vs dynamic

```python
import torch
# static (shape 变重编译)
@torch.compile
def f_static(x): return x * 2
for L in [4, 8, 12, 16]:
    f_static(torch.randn(L, 8))   # 每 L 重编译 (4 份缓存)

# dynamic (symbolic, 一份缓存)
@torch.compile(dynamic=True)
def f_dyn(x): return x * 2
for L in [4, 8, 12, 16]:
    f_dyn(torch.randn(L, 8))     # 只首次编译, 后续复用 (symbolic)
```

### 5.2 显式 dynamic_shapes

```python
@torch.compile(dynamic_shapes={'x': {0: True}})   # x 第 0 维 symbolic, 第 1 维 static
def f(x): return x @ x.T
f(torch.randn(4, 8))    # s0=4, 8 static
f(torch.randn(7, 8))    # s0=7 valid, 复用 (第 1 维仍 8 static)
```

### 5.3 bucket 策略

```python
# shape 种类少 -> static 各编译 (优于 dynamic)
@torch.compile
def f(x): return x * 2
for bs in [8, 16, 32, 64]:   # 4 种
    f(torch.randn(bs, 768))  # 4 份 static 专用 kernel, 每份最优
# 比 dynamic 优 (每份 static 性能好, 只 4 份缓存)
```

### 5.4 观察 guard 与缓存

```python
# TORCH_COMPILE_DEBUG=1 看 guard
# static: guard "x.shape == [4, 4]"
# dynamic: guard "x.shape[0] in [1, inf], x.shape[1] == 4"
# 缓存: ls ~/.cache/torchinductor/  static 多份, dynamic 一份
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[TorchDynamo]]（guard 是 Dynamo 的，dynamic 改 guard）、symbolic shape（形状用变量表示）、[[graph break]]（dynamic 减 shape 相关 break）。
- **下游（应用）**: [[torch.compile]]（dynamic=True 选项）、[[Inductor]]（codegen symbolic kernel）、[[CUDA Graph与graph capture]]（与 dynamic 冲突）、[[continuous batching的调度实现]]（变长 batch 用 dynamic）、[[dynamic shape|bucket 策略]]（有限 shape 用 static bucket）、`torch.export`（dynamic_shapes 导出）。
- **对比 / 易混**:
  - **dynamic shape vs [[graph break]]**：见 3.9，dynamic 是 shape 变的 guard 重编译；graph break 是结构不支持回退。dynamic 减 shape 相关 break，但非全部 break。
  - **dynamic=True vs dynamic_shapes**：前者全维 symbolic（粗）；后者显式指定哪些维（精）。显式更优（静态维仍 hardcode）。
  - **dynamic vs max-autotune**：见 3.7，dynamic 无法 autotune（shape 变），static 才 max-autotune 拿满。
  - **dynamic vs reduce-overhead**：见 3.8，dynamic 与 CUDA Graph 冲突，需 bucket 或放弃 graph。


## 7. 常见误区与易错点

> [!warning] 误区 1：dynamic 总是优于 static
> shape 种类少（<10）时 static 各编译一份（每份专用 kernel）优于 dynamic（symbolic kernel 稍弱）。只 shape 频繁连续变才 dynamic。盲目 dynamic 损失性能。

> [!warning] 误区 2：dynamic 不需 max-autotune
> dynamic 无法 autotune（shape 变最优 tile 变）→ max-autotune 退化默认 config。static shape 才 max-autotune 拿满。dynamic + max-autotune 收益打折。

> [!warning] 误区 3：dynamic + reduce-overhead 可用
> CUDA Graph 要求 static shape，dynamic 冲突。需 bucket（每 shape 一 graph）或放弃 CUDA Graph。reduce-overhead 模式与 dynamic 不兼容。

> [!warning] 误区 4：忽视缓存膨胀
> static 每 shape 一份缓存，shape 多 → GB 级膨胀 + 反复编译开销。dynamic 一份（symbolic）。监控 `~/.cache/torchinductor` 大小，定期清。

> [!warning] 误区 5：混淆 dynamic 与 graph break
> dynamic 是 shape 变的 guard 重编译（同代码不同 shape）；graph break 是结构不支持回退 eager。dynamic 减 shape 相关 break，但 data-dependent control flow 等结构 break 仍需改代码。

> [!warning] 误区 6：dynamic=True 全维最优
> `dynamic=True` 全维 symbolic 粗（可能不必要维也 symbolic 损失优化）。用 `dynamic_shapes` 显式指定哪些维（其余 static hardcode）更优。

> [!warning] 误区 7：symbolic kernel 与 static 性能一样
> symbolic 多 mask + runtime 分块 + 不能 hardcode，性能 $\sim 0.9$ static。shape 种类少时 static 各份更优。dynamic 是"避免反复编译"的 trade-off，非免费。


## 8. 延伸细节

### 8.1 symbolic shape 的推导

PyTorch 的 `torch.fx`/`torch._subclasses` 支持 symbolic shape（shape 是 `SymInt` 变量）。Dynamo `dynamic=True` 时把 shape 标 symbolic，guard 断言"是变量"而非值。Meta backend（[[ATen与c10]] 3.7）推 symbolic shape。是 dynamic 的底层支持。

### 8.2 自动 dynamic 推断

PyTorch 实验自动推断哪些维动态（不需手动 `dynamic_shapes`）：基于 trace 时见到的 shape 变化推断。但保守（可能误标 static）。手动 `dynamic_shapes` 更精确。

### 8.3 与 FSDP2/DTensor 的关系

DTensor（[[DTensor]]）的 sharding 维是 static（mesh 固定），local tensor 的非 sharding 维可能 dynamic（变长 batch）。FSDP2 与 dynamic shape 的交互是 compile + 分布式的复杂点。详见 [[FSDP2]]。

### 8.4 bucket 的 graph 池

推理 server 常用 bucket（固定几个 batch size）+ 每 bucket 一份编译图 + CUDA Graph（每 bucket 一 graph）。比 dynamic 性能好（每份 static 专用 + CUDA Graph），比全 static 缓存小（只几份）。是变长推理的主流方案。详见 [[推理 CUDA Graph]]。

### 8.5 内容来源

dynamic shape 整理自 PyTorch dev docs "Dynamic Shapes"、`torch/_dynamo/` 的 guard 与 symbolic shape 实现、`dynamic_shapes` 文档。bucket 策略见推理框架（vLLM/SGLang）的 graph 池设计。

---
相关: [[图捕获问题]] | [[TorchDynamo]] | [[torch.compile]] | [[Inductor]] | [[graph break]] | [[CUDA Graph与graph capture]] | [[continuous batching的调度实现]] | [[推理 CUDA Graph]] | [[torch profiler]] | [[FSDP2]] | [[DTensor]]
