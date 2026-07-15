# AOTAutograd

> **所属章节**: [[Compiler组件]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: AOTAutograd / AOT-Autograd / 前向分解 / joint graph / ahead-of-time autograd / decomposition / functionalization
> **难度**: 高（需懂 [[TorchDynamo]] + [[Autograd Engine]] + [[torch.compile]]）


## 1. 一句话定义

**AOTAutograd** 是 `torch.compile` 流水线的**第二段**——接 [[TorchDynamo]] 输出的**前向 FX Graph**（op 级，仅前向），用 PyTorch 的 autograd 公式 + **decomposition（把 high-level op 拆成 low-level ATen op）**生成**反向 FX Graph**，再把前向 + 反向合成一张 **joint graph**（前向+反向一起），交下游 [[Inductor]] 整体 codegen 融合。它的核心价值：让编译器看到**前向与反向一起**，从而跨前向/反向边界融合（如前向算的反向可复用、前向存的 activation 与反向的梯度计算融合），而非把前向、反向当两段独立编译。同时它做 **decomposition**（如把 `elu` 拆成 `exp`/`sub`，让 Inductor 更易融合）与 **functionalization**（消除 in-place op 的副作用，让图纯函数化便于分析）。AOTAutograd 取代了 eager 模式 [[Autograd Engine]] 逐节点调度的低效——编译路径把反向也融成 fused kernel。

> [!note] 三句话定位
> - **是什么**：compile 第二段，接 Dynamo 前向图，用 autograd 公式 + decomposition 生成反向图，合成 joint graph 交 Inductor。
> - **为什么**：让编译器看前向+反向一起，跨边界融合（eager 模式前向反向分开调度无法融合）；decomposition 把复杂 op 拆成简单 op 更易融合。
> - **与 [[Autograd Engine]] 关系**：Engine 是 eager 反向执行器（逐节点调度）；AOTAutograd 是编译路径把反向也融成 fused kernel，绕过逐节点调度。两条路径并存。


## 2. 为什么需要它（动机与背景）

### 2.1 eager 反向的低效

eager 模式反向走 [[Autograd Engine]] 逐 grad_fn 节点调度：每个 backward op 单独 kernel + HBM 往返 + dispatcher 路由。相邻 backward op 不融合（如 `MulBackward` 后接 `AddBackward` 各自 kernel）。且前向存的 activation 在反向逐节点取用，无法跨节点优化。

### 2.2 编译需看到 joint graph 才能跨边界融合

若编译器只看前向图（Dynamo 出的），融合仅限前向内部。反向仍走 eager Autograd Engine 逐节点，不融合。要让反向也融合，编译器需看到**前向 + 反向 joint graph**——AOTAutograd 用 autograd 公式从 Dynamo 的前向图生成反向图，合成 joint graph 交 Inductor。Inductor 看到的是一张含前向+反向的图，可跨边界融合（如前向算的中间值直接传给反向，不落 HBM）。

### 2.3 decomposition 让复杂 op 可融合

`elu`/`layer_norm`/`softmax` 等 high-level op 若整体当一节点，Inductor 难融合（不知内部结构）。AOTAutograd 做 **decomposition**：把 `elu` 拆成 `exp(x)-1` 等 low-level ATen op，让 Inductor 把拆开的 op 与相邻 op 融合。是 [[fused kernel]] 自动化的关键。

### 2.4 functionalization 消除副作用

in-place op（`x.add_(1`）改原 tensor，有副作用，图分析难（数据流不确定）。AOTAutograd 做 **functionalization**：把 in-place 转成 out-of-place（`x.add(1)` 返回新 tensor），让图纯函数化，Inductor 可自由融合/重排。是编译优化的前提。

### 2.5 取代 eager 反向的逐节点调度

编译路径（AOTAutograd + Inductor）把反向融成 fused kernel，绕过 Autograd Engine 的逐节点 Python/C++ 调度 + dispatcher 路由。反向也享 fusion + 去 launch 收益。是 compile 对反向的优化（eager 反向无此）。


## 3. 核心概念详解

### 3.1 三段流水中的位置

```
Dynamo (前向 op 级 FX Graph)
    ↓
AOTAutograd (加反向 + decomposition + functionalization → joint FX Graph)
    ↓
Inductor (融合 + codegen → Triton/C++ kernel)
```

AOTAutograd 输入 Dynamo 的前向图，输出 joint graph（前向+反向）。

### 3.2 joint graph 的构造

AOTAutograd：
1. 读 Dynamo 前向图（节点是 `aten.*` op）。
2. 对每前向 op，查其 **autograd 公式**（`aten.elu` → `EluBackward`，已注册在 dispatcher 的 Autograd 层）。
3. 反向遍历前向图，用公式生成反向图（与 Autograd Engine 的拓扑遍历同理，但这里是**编译期**生成静态反向图，非运行时调度）。
4. 合成 joint graph：前向节点 + 反向节点 + saved tensors 边（前向存的中间值连到反向用）。

joint graph 是静态的（不随输入变，只随 shape 变重 trace），Inductor 整体 codegen。

### 3.3 decomposition

```python
# elu decomposition (概念)
elu(x) = where(x>0, x, alpha*(exp(x)-1))
# 拆成: exp / sub / mul / where 等 low-level ATen op
# Inductor 可把 exp 与相邻 op 融合
```

high-level op（`elu`/`layer_norm`/`softmax`/`gelu`）拆成 low-level（`exp`/`mul`/`add`/`reduce`）。low-level 更易融合（elementwise 与 elementwise 融、reduction 单独留）。decomposition 是 Inductor 融合的前提。

### 3.4 functionalization

```python
# in-place (副作用)
x.add_(1)        # 改 x
# functionalization 转 out-of-place
x = x.add(1)    # 返回新 tensor, 不改原
```

in-place 致数据流不确定（哪步改了谁），图分析难。functionalization 把所有 in-place 转 out-of-place，图纯函数化（每个 op 输入不变、输出新 tensor），Inductor 可自由融合/重排/删冗余。

### 3.5 saved tensors 的显存

反向公式需前向的某些中间值（如 `MatMulBackward` 需前向 $Q,K$；`SoftmaxBackward` 需 softmax 输出）。eager 模式这些存在 grad_fn.saved_tensors（见 [[Autograd Engine]]），占 [[activation memory]]。AOTAutograd 编译期在 joint graph 里把这些 saved 标记为"前向算、反向用"，Inductor codegen 时决定是存（落 HBM，反向取）还是重算（不存，反向重算前向部分，省显存增 FLOP，即 [[gradient checkpointing]]）。编译路径的 saved tensors 管理更灵活。

### 3.6 与 Autograd Engine 的对比

| 维度 | eager [[Autograd Engine]] | 编译 AOTAutograd |
|---|---|---|
| 时机 | 运行时 `.backward()` 调度 | 编译期生成静态反向图 |
| 融合 | 逐节点，不融合 | joint graph 融合 |
| 调度 | 引用计数 + ready 队列 + 多线程 | 静态图 codegen，无运行时调度 |
| saved tensors | 存 grad_fn，固定 | 编译期决定存/重算 |
| hooks | 支持 | 编译图内 hook 难（绕过逐节点） |

编译路径快但失去逐节点 hook 灵活性。eager 与编译路径并存（compile 失败回 eager）。

### 3.7 decomposition 的边界

不是所有 op 都拆——某些 op 拆了反而差（如 `layer_norm` 的 reduction 拆开难融合）。AOTAutograd 有 decomposition 规则集，决定哪些拆哪些留。Inductor 对留的高层 op（如 `layer_norm`）有专门 codegen 路径。

### 3.8 与 torch.export 的关系

`torch.export` 也用 AOTAutograd 的 decomposition（保证导出的图是 low-level + functionalized，可移植）。AOT 序列化与运行时 compile 共用 AOTAutograd 的变换。

### 3.9 backend 的可替换

`torch.compile(backend='aot_eager')` 用 AOTAutograd 但不 Inductor codegen（joint graph 走 eager dispatcher）——用于调试 AOTAutograd 的输出，隔离 Dynamo/AOTAutograd/Inductor 的 bug。`backend='inductor'` 是默认（AOTAutograd + Inductor）。


## 4. 数学原理 / 公式

### 4.1 joint graph 的 FLOP

joint graph 含前向 FLOP $F_{\text{fwd}}$ + 反向 FLOP $F_{\text{bwd}} \approx 2 F_{\text{fwd}}$（autograd 链式）。总 $\approx 3 F_{\text{fwd}}$（与 [[FLOPs计算]] $6ND$ 一致：前向 $2$ + 反向 $4$，但 $2 F_{\text{fwd}}$ 是反向算力近似）。

### 4.2 融合的 byte 收益

eager 反向逐节点：每 backward op 读 saved + 读 grad + 写 grad，$\sim 3N$ byte/op。joint 融合：相邻 backward op 中间 grad 在 register 传递，$\sim N$ byte（只读 saved + 写最终 grad）。省 $\sim 3\times$ byte（见 [[fused kernel]]）。

### 4.3 saved tensors 的显存 vs 重算

存：activation memory $\propto L$（层.saved）。重算（gradient checkpointing）：显存 $\propto \sqrt{L}$（只存 checkpoint 点），但 FLOP $\times 2$（重算前向）。joint graph 让 Inductor 编译期权衡（部分存部分重算）。见 [[gradient checkpointing]]。

### 4.4 decomposition 的可融合性

high-level op 整体：Inductor 只能整体 codegen（或 fallback dispatcher）。拆成 low-level：相邻 low-level 融合（elementwise 串合一 kernel）。拆开后融合的 byte 收益见 [[fused kernel]] 4.1（$k$ op 合一省 $k\times$ byte）。


## 5. 代码示例（可选）

### 5.1 观察 AOTAutograd 输出

```python
import torch
@torch.compile(backend='aot_eager')   # 用 AOTAutograd 不 codegen, 调试
def f(x): return torch.relu(x * 2)
# aot_eager 打印 joint graph (前向+反向)
# TORCH_COMPILE_DEBUG=1 看分解前后
```

### 5.2 decomposition 示例

```python
# elu 拆成 low-level (概念)
# 原: y = elu(x, alpha=2)
# decomposition:
#   e = exp(x); m = 2*(e-1); y = where(x>0, x, m)
# Inductor 可把 exp/mul/sub/where 融合成一 kernel
# 而 eager 的 elu 是整体一 op (不拆), 不与相邻融合
```

### 5.3 functionalization

```python
@torch.compile
def f(x):
  x.add_(1)        # in-place
  return x * 2
# AOTAutograd functionalization: 转 x = x.add(1); return x*2
# 图纯函数化, Inductor 可融合 add+mul
# eager 模式 in-place 也对, 但图分析难 (数据流不确定)
```

### 5.4 saved tensors 的 codegen 决策

```python
# Inductor 对 joint graph 的 saved tensors:
# - 存: 前向算的中间值落 HBM, 反向读 (默认, 省 FLOP 费显存)
# - 重算: 反向重算前向部分 (gradient checkpointing, 省显存费 FLOP)
# torch.compile 的 activation_memory 受 Inductor 的 saved/recompute 策略影响
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[TorchDynamo]]（输入前向 FX Graph）、[[Autograd Engine]]（autograd 公式来源、eager 对应物）、[[Dispatcher]]（Autograd 层的公式注册）、[[ATen与c10]]（decomposition 到 ATen op）。
- **下游（应用）**: [[torch.compile]]（AOTAutograd 是第二段）、[[Inductor]]（接 joint graph codegen）、[[gradient checkpointing]]（编译路径的 saved/recompute 决策）、[[activation memory]]（saved tensors 显存）、[[FLOPs计算]]（joint graph 的 FLOP）、[[fused kernel]]（decomposition 让融合更易）、`torch.export`（共用 decomposition/functionalization）。
- **对比 / 易混**:
  - **AOTAutograd vs [[Autograd Engine]]**：见 3.6，编译路径（静态反向图 + 融合）vs eager（运行时逐节点调度）。
  - **AOTAutograd vs [[TorchDynamo]]**：Dynamo 出前向 op 级图；AOTAutograd 加反向 + decomposition + functionalization → joint graph。
  - **decomposition vs [[Custom C++ CUDA Operator]]**：decomposition 把 op 拆成 low-level 让 Inductor 融合；custom op 注册高层 op 让 Dynamo 识别。两者正交。
  - **functionalization vs in-place**：functionalization 把 in-place 转 out-of-place 让图纯函数化；用户仍可写 in-place，AOTAutograd 自动转。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 AOTAutograd 是运行时
> AOTAutograd 是**编译期**生成静态反向图（从 Dynamo 前向图 + autograd 公式），不是运行时调度。运行时跑的是 Inductor codegen 的 fused kernel。eager 才运行时调度（[[Autograd Engine]]）。

> [!warning] 误区 2：忽视 joint graph 的跨边界融合
> AOTAutograd 的核心价值是让 Inductor 看到前向+反向一起，跨边界融合。若分开编译前向反向，反向仍逐 op 不融合，与 eager 无异。

> [!warning] 误区 3：decomposition 总是好的
> 某些 op 拆开反而差（如 layer_norm 的 reduction 拆开难融合）。AOTAutograd 有规则决定拆哪些留哪些，Inductor 对留的 high-level 有专门 codegen。盲目全拆会劣化。

> [!warning] 误区 4：混淆 AOTAutograd 与 [[Autograd Engine]] 的反向
> Engine 是 eager 运行时逐节点调度；AOTAutograd 是编译期生成静态反向图。同一 autograd 公式，两种执行路径。compile 走 AOTAutograd，eager 走 Engine。

> [!warning] 误区 5：compile 后还能用 grad hook
> 编译路径把反向融成 fused kernel，绕过逐节点 [[Autograd Engine]]，故 `register_hook`/`full_backward_hook` 可能不触发或行为变。需 hook 的场景慎用 compile 或用 `backend='aot_eager'`（保留逐节点）。

> [!warning] 误区 6：忽视 saved tensors 在编译路径的灵活
> eager 的 saved 固定存 grad_fn；AOTAutograd 编译期让 Inductor 决定存/重算（gradient checkpointing）。compile 的 activation memory 受此影响，可能与 eager 不同。

> [!warning] 误区 7：functionalization 改变语义
> in-place 转 out-of-place 在纯函数语义下等价，但对依赖原 tensor id 的代码（如外部持有 x 引用期望被改）可能行为变。compile 假设用户代码 functional-friendly。


## 8. 延伸细节

### 8.1 AOTAutograd 的 decompositions 表

`torch._decomp` 模块维护 decomposition 规则（哪些 op 拆成哪些 low-level）。可自定义 decomposition（如把某 op 拆成更优的 low-level 序列）。是 Inductor 融合能力的基础。

### 8.2 与 compiled autograd 的关系

`torch._C._compiled_autograd` 实验：直接对 [[Autograd Engine]] 的 grad_fn DAG 编译（不经 AOTAutograd 的 joint graph 路径）。是另一条编译反向的路径，与 AOTAutograd 的 joint graph 路径并存。详见 [[Autograd Engine]] 8.3。

### 8.3 min-cut activation memory

AOTAutograd 有实验的 min-cut 算法：在 joint graph 上找"哪些 saved tensors 存、哪些重算"的最优割（最小显存 + 最小重算 FLOP）。是 compile 路径对 [[activation memory]] 的精细管理。eager 无此（固定全存）。

### 8.4 与 FSDP2/DTensor 的关系

DTensor op（[[DTensor]]）可被 AOTAutograd decomposition（如 `aten.slice` 的 DTensor 变体）。FSDP2 的 per-param sharding 在 AOTAutograd 看来是 DTensor op，可 compile。是 compile 与分布式结合的接口。详见 [[FSDP2]]。

### 8.5 内容来源

AOTAutograd 整理自 PyTorch dev docs "AOTAutograd"、`torch/_functorch/` 源码、`torch._decomp` decomposition 规则。joint graph/min-cut 见 functorch design doc。与 [[Autograd Engine]]/[[Inductor]] 的分工见 compile 流水文档。

---
相关: [[Compiler组件]] | [[torch.compile]] | [[TorchDynamo]] | [[Inductor]] | [[Autograd Engine]] | [[Dispatcher]] | [[ATen与c10]] | [[fused kernel]] | [[gradient checkpointing]] | [[activation memory]] | [[FLOPs计算]] | [[Custom C++ CUDA Operator]] | [[FSDP2]] | [[DTensor]]
