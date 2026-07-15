# torch.compile

> **所属章节**: [[torch.compile]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: torch.compile / JIT 编译入口 / compile / 编译模式 / reduce-overhead / max-autotune / 编译开销
> **难度**: 中（需懂 [[Dispatcher]] + eager 模式 + [[kernel launch overhead]]）


## 1. 一句话定义

**`torch.compile`** 是 PyTorch 2.0 引入的 **JIT（即时）编译入口**——`@torch.compile` 装饰函数或 `model.compile()` 后，PyTorch 在**首次调用时**把该函数的 Python 执行**追踪（trace）成静态计算图**，经 [[TorchDynamo]]（Python 字节码追踪 + guard）→ [[AOTAutograd]]（前向+反向 joint 分解）→ [[Inductor]]（codegen 成 [[Triton kernel开发]]/C++ fused kernel）三段编译流水线，生成优化的 fused kernel 跑后续调用。它提供**三种模式**：`default`（基础融合）、`reduce-overhead`（叠加 [[CUDA Graph与graph capture]] 压 launch）、`max-autotune`（叠加 autotune 选最优 tile/config）。收益是**减 [[kernel launch overhead]] + [[fused kernel]] 减 HBM 往返 + autotune 选最优 config**，代价是**首次编译开销（秒~分钟）+ 编译缓存 + 动态 shape 触发重编译**。它是 2024+ PyTorch 提速的主线 API，取代了 `torch.jit.script`/`torch.jit.trace`（遗留）。

> [!note] 三句话定位
> - **是什么**：JIT 编译入口，把 Python 函数 trace 成图，经 Dynamo→AOTAutograd→Inductor 生成 fused kernel。
> - **为什么**：eager 模式逐 op 走 dispatcher + Python 调用，launch 与 HBM 往返浪费；compile 融合 + 去 dispatcher + 去 Python 开销。
> - **三模式**：default（融合）、reduce-overhead（+CUDA Graph 压 launch）、max-autotune（+autotune 选 config）。开销与收益递增。


## 2. 为什么需要它（动机与背景）

### 2.1 eager 模式的浪费

eager 模式（默认）逐 op 执行：Python 调 `at::add` → [[Dispatcher]] 路由 → kernel launch → HBM 往返 → 下一个 op。问题：
- **[[kernel launch overhead]]**：每 op ~5 μs CPU launch，小 op 串时 launch 主导（见 [[fused kernel]] 2.1）。
- **[[fused kernel]] 缺失**：相邻 elementwise/norm op 各自 HBM 往返，中间结果落 HBM（$I$ 低，见 [[Roofline模型]]）。
- **Python/dispatcher 开销**：每 op Python 解释 + dispatcher 路由 ~几十 ns~μs。

大网络的 forward 有上千 op，这些开销累积可观。compile 一次性把整段融合 + 去 Python/dispatcher。

### 2.2 三段编译流水线

`torch.compile` 把"Python 函数 → 优化 kernel"分三段，每段可独立替换：
1. **[[TorchDynamo]]**：Python 字节码追踪，用 guard 检测"这段代码这次和上次一样吗"，一样则复用已编译的图，不一样则重 trace。处理 [[graph break]]（不支持的 Python 结构跳出）。
2. **[[AOTAutograd]]**：把前向 + 反向 joint 分解成 ATen op 级 graph，做 autograd 分解（把 high-level op 拆成 low-level + autograd 公式）。
3. **[[Inductor]]**：codegen，把 graph 的 op 融合成 [[Triton kernel开发]]（GPU）或 C++（CPU）fused kernel，可选 autotune。

三段解耦，每段可换后端（如 Inductor 换 `aot_eager` 或 `cudagraphs`）。

### 2.3 三模式

| 模式 | 叠加 | 开销 | 收益 |
|---|---|---|---|
| `default` | Dynamo + AOTAutograd + Inductor fusion | 中（首次编译秒~分钟） | 减 HBM 往返 + 去 Python/dispatcher |
| `reduce-overhead` | + [[CUDA Graph与graph capture]] | 高（需 static shape + graph 池） | 再压 launch（decode/小 batch 收益大） |
| `max-autotune` | + autotune（Inductor 试多 config 选最优） | 最高（autotune 耗时） | 再提 kernel 效率（GEMM/attention 选最优 tile） |

### 2.4 与 eager 的并存

compile 不是替代 eager，而是**叠加**：compile 失败处 graph break 回落 eager（见 [[graph break]]）。故 compile 是"尽量编译，不能则回退"的安全优化。首次调用编译，后续复用；shape 变 → 重编译（见 [[dynamic shape]]）。

### 2.5 取代 torch.jit

`torch.jit.script`/`torch.jit.trace`（TorchScript）是 1.x 的编译方案，需手写 `@torch.jit.script`、限制 Python 语法、不自动融合。`torch.compile` 装饰任意 Python 函数、自动 trace + 融合、graph break 安全回退，是 2.0+ 主线，TorchScript 遗留维护。


## 3. 核心概念详解

### 3.1 基本用法

```python
@torch.compile
def f(x, w): return torch.relu(x @ w)

@torch.compile(mode='reduce-overhead')
def g(x): return model(x)

model.compile()  # 等价 model.forward = torch.compile(model.forward)

@torch.compile(fullgraph=True)  # 强制全图, 有 graph break 报错
def h(x): ...
```

### 3.2 首次调用的编译

第一次调 `f(x)`：Dynamo trace 字节码 → AOTAutograd 分解 → Inductor codegen → 编译 Triton（`triton.compile`）→ 缓存。耗时秒~分钟（看复杂度）。后续调用复用缓存（μs 级 lookup）。

### 3.3 guard 与重编译

Dynamo 用 **guard**（条件断言）判断"上次编译的图这次还能用吗"。guard 是对输入 shape/dtype/值 的断言（如 `x.shape == [4,4]`、`x.dtype == float32`、`x.requires_grad == True`）。guard 失败（shape 变了）→ 重 trace + 重编译（见 [[dynamic shape]]）。故固定 shape 收益最大，变 shape 触发重编译开销。

### 3.4 graph break

Dynamo 遇不支持的 Python 结构（如 `print`、不支持的 builtin、副作用函数、data-dependent control flow）→ **graph break**：把当前累积的图编译，剩余回 eager 执行，下个可 trace 处再开新图。多处 graph break → 多段编译图 + 中间 eager。详见 [[graph break]]。

### 3.5 Inductor 的融合

Inductor 把 graph 里相邻的 elementwise/norm/reduction op 融合成一个 [[Triton kernel开发]] kernel（GPU）或 C++ kernel（CPU）。中间结果在 register/smem 传递不落 HBM（见 [[fused kernel]]）。GEMM/attention 走专门 kernel（cuBLAS/FlashAttention）+ epilogue 融合。是 compile 提速的核心。

### 3.6 reduce-overhead 与 CUDA Graph

`reduce-overhead` 模式在 Inductor fusion 基础上叠加 [[CUDA Graph与graph capture]]：把整段 forward 的 kernel 序列捕获成 graph，replay 时跳过 CPU launch。收益在小 batch/decode（launch 主导）最大。需 static shape（graph capture 要求）。

### 3.7 max-autotune 与 autotune

`max-autotune` 让 Inductor 对 GEMM/attention 等 op 试多个 tile/config（如不同 BLOCK_M/N/K、num_warps），实测 throughput 选最优（类似 [[Triton kernel开发]] 的 `@triton.autotune`）。autotune 耗时（每个 config 跑一遍），首次编译慢，但后续跑最优 config。

### 3.8 编译缓存

Inductor 缓存生成的 Triton/C++ code 到 `~/.cache/torchinductor`（hash = 源码 + shape + 模式）。重启后同 shape 复用（跳 codegen，秒级）。shape 变 → 新 hash → 新缓存。

### 3.9 与 distributed 的关系

DDP/FSDP 的 forward 也可 compile（compile 整个 forward 含 all_reduce）。但通信 op（NCCL）在 graph 里是特殊 node（不融合，单独执行）。FSDP2 的 DTensor op 可被 compile trace。见 [[FSDP2]]。

### 3.10 与 custom op 的关系

[[Custom C++ CUDA Operator]] 用 `TORCH_LIBRARY` 注册的 op 被 Dynamo trace（不 graph break）。未注册的 Python kernel → graph break。故要 compile-friendly 必须用 TORCH_LIBRARY。


## 4. 数学原理 / 公式

### 4.1 eager vs compile 的开销

eager 每 op：$t_{\text{eager}} = t_{\text{py}} + t_{\text{dispatch}} + t_{\text{launch}} + t_{\text{kernel}}$。

compile 后（融合 $k$ op 成 1）：$t_{\text{compiled}} = t_{\text{lookup}} + t_{\text{launch}} \cdot 1 + t_{\text{kernel,fused}}$。

省 $\approx k \cdot (t_{\text{py}} + t_{\text{dispatch}} + t_{\text{launch}}) + (k-1) \cdot t_{\text{hbm roundtrip}}$。小 op 串收益最大。

### 4.2 编译开销 vs 运行收益

设编译 $T_{\text{compile}}$（秒~分钟），每步省 $\Delta t$（μs~ms），跑 $N$ 步回本：

$$
N_{\text{break-even}} = T_{\text{compile}} / \Delta t
$$

训练跑几千步通常远超 break-even，compile 收益正。推理单次请求可能不回本（除非缓存跨请求复用）。

### 4.3 dynamic shape 的重编译

guard 失败（shape 变）→ 重 trace + 重 compile。设 $S$ 个不同 shape，总编译 $S \cdot T_{\text{compile}}$。shape 频繁变（如变长 batch）→ 反复编译，可能负收益。用 `dynamic=True` 让 Inductor 生成支持 dynamic shape 的 kernel（牺牲部分优化）减少重编译。见 [[dynamic shape]]。

### 4.4 reduce-overhead 的 launch 压缩

CUDA Graph replay 的 launch $\approx 1$ μs（vs 逐 op $k \cdot 5$ μs）。$k$ 大时省 $k$ 倍 launch。decode 的几十 op 串收益显著。


## 5. 代码示例（可选）

### 5.1 基本用法与模式

```python
import torch
model = MyModel().cuda()
model.compile()                              # default 模式
# 或
@torch.compile(mode='reduce-overhead')       # +CUDA Graph
def fwd(x): return model(x)
@torch.compile(mode='max-autotune')          # +autotune
def fwd2(x): return model(x)

# 首次调用编译 (慢)
x = torch.randn(64, 768, device='cuda')
out = model(x)   # 触发编译 ~秒级
# 后续复用 (快)
out2 = model(x)  # ~μs lookup + fused kernel
```

### 5.2 观察 graph break

```python
@torch.compile
def f(x):
  y = x * 2
  print(y.sum().item())   # 副作用 + data-dependent, graph break
  z = y + 1               # 新图
  return z
# Dynamo 报告: [WARNING] graph break at print, 2 graphs compiled
```

### 5.3 dynamic shape

```python
@torch.compile(dynamic=True)   # 允许变 shape, 减重编译
def f(x): return x * 2
for L in [10, 20, 30]:        # 不同 shape
  f(torch.randn(L, 8))        # dynamic: 不重编译 (生成通用 kernel)
# 不设 dynamic: 每个 L 重编译一次
```

### 5.4 编译缓存

```python
# ~/.cache/torchinductor/<hash>/  存生成的 Triton code
# 重启后同 shape 复用 (跳 codegen)
# 清缓存: rm -rf ~/.cache/torchinductor
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Dispatcher]]（compile 仍走 dispatcher，但融合后 op 数减）、[[kernel launch overhead]]（compile + reduce-overhead 压 launch）、[[fused kernel]]（Inductor 融合的原理）、[[Roofline模型]]（融合提 $I$）。
- **下游（应用）**: [[TorchDynamo]]（trace + guard）、[[AOTAutograd]]（joint 分解）、[[Inductor]]（codegen）、[[graph break]]（不支持的回退）、[[dynamic shape]]（shape 变的重编译）、[[CUDA Graph与graph capture]]（reduce-overhead 模式）、[[Custom C++ CUDA Operator]]（注册的 op 才 compile-friendly）、[[SM utilization]]/[[MFU与算术强度]]（compile 提 MFU 的手段）、[[训练显存估计]]（compile 改变 activation 与 kernel 数）。
- **对比 / 易混**:
  - **torch.compile vs eager**：eager 逐 op 走 dispatcher + Python；compile 融合 + 去 dispatcher + 去 Python。compile 失败回退 eager。
  - **torch.compile vs `torch.jit.script`**：compile 装饰任意 Python、自动 trace、graph break 安全；jit.script 需手写、限制语法、不自动融合。compile 是 2.0+ 主线。
  - **torch.compile vs [[CUDA Graph与graph capture]]**：compile 是编译（融合 + codegen）；CUDA Graph 是 kernel 序列捕获 replay（压 launch）。`reduce-overhead` 模式叠加两者。
  - **torch.compile vs [[fused kernel]]**：fused kernel 是原理（多 op 合一）；torch.compile 是自动融合的入口（Inductor 实现）。手写 fused kernel 仍在关键路径用。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 compile 一劳永逸
> 首次调用编译（慢），shape 变触发重编译（[[dynamic shape]]）。变长 batch 不设 `dynamic=True` 反复编译，可能负收益。

> [!warning] 误区 2：忽视 graph break
> Python 副作用、不支持的 builtin、data-dependent control flow 触发 graph break，分段编译 + 中间 eager，收益打折。用 `TORCH_LIBRARY` 注册 custom op、避免副作用、`fullgraph=True` 调试。

> [!warning] 误区 3：reduce-overhead 不需 static shape
> reduce-overhead 叠加 CUDA Graph，要求 static shape（graph capture 要求）。变长 batch 用 default 模式或 `dynamic=True`。

> [!warning] 误区 4：compile 改数值
> compile 融合改变浮点累加顺序，有 $O(\epsilon)$ 数值差异（与 [[训推不一致]] TIM 相关）。需严格复现用 `torch.use_deterministic_algorithms` 或固定编译。

> [!warning] 误区 5：推理单次请求用 compile 回不了本
> 编译开销秒~分钟，推理单次请求 $\Delta t$ 省的 μs 不回本。compile 收益在多次复用（训练几千步、推理 server 缓存跨请求）。单次推理用 AOT 编译的 onnx/tensorrt 更合适。

> [!warning] 误区 6：max-autotune 总是最优
> max-autotune 首次编译很慢（每 config 跑一遍），且 autotune 的最优未必稳定（不同输入可能不同）。先用 default，瓶颈 op 再 max-autotune 局部。

> [!warning] 误区 7：compile 万能提速
> compute-bound 的大 op（如大 GEMM）已接近 [[Roofline模型]] 峰值，compile 融合收益小（本就 1 kernel）。compile 主要提 memory-bound 小 op 串与 launch 主导场景。


## 8. 延伸细节

### 8.1 compile 的 backend 可替换

`torch.compile(backend='...')` 可换后端：`inductor`（默认）、`aot_eager`（AOTAutograd 不 codegen）、`cudagraphs`（只 CUDA Graph）、`eager`（禁用，调试）。自定义 backend 可注册。是 compile 的可扩展点。

### 8.2 与 torch.export 的关系

`torch.export`（AOT 序列化）把模型导成 graph（不依赖运行时 trace），可跨进程/跨后端/部署。与 `torch.compile` 互补：export 生成可移植 graph，compile 运行时优化。export 的 graph 可喂给 Inductor codegen。详见 torch.export dev docs。

### 8.3 compiled autograd

PyTorch 实验 `torch._C._compiled_autograd` 把反向 grad_fn 图也 compile（绕过 [[Autograd Engine]] 逐节点调度）。是 compile 对反向的扩展。详见 [[Autograd Engine]] 8.3。

### 8.4 与 [[MFU与算术强度]] 的关系

compile 减 byte 提 $I$ → 提 MFU，是 MFU 提升手段之首（见 [[MFU与算术强度]]）。但 compile 不改模型 FLOP（$6ND$ 不变），改的是 byte 与 launch。

### 8.5 内容来源

torch.compile 整理自 PyTorch 2.x dev docs "torch.compile"、PEP-style 设计文档、Inductor tutorial。三模式与 backend 见 `torch._inductor.config`。graph break/dynamic shape 见 [[TorchDynamo]]/[[graph break]]/[[dynamic shape]]。

---
相关: [[torch.compile]] | [[TorchDynamo]] | [[AOTAutograd]] | [[Inductor]] | [[graph break]] | [[dynamic shape]] | [[fused kernel]] | [[kernel launch overhead]] | [[CUDA Graph与graph capture]] | [[Roofline模型]] | [[SM utilization]] | [[MFU与算术强度]] | [[Custom C++ CUDA Operator]] | [[训推不一致]] | [[Dispatcher]] | [[计算图]]
