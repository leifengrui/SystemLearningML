# Inductor

> **所属章节**: [[Compiler组件]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: Inductor / torch._inductor / codegen backend / Triton codegen / C++ codegen / fusion / epilogue fusion / autotune
> **难度**: 高（需懂 [[torch.compile]] + [[AOTAutograd]] + [[Triton kernel开发]] + [[fused kernel]]）


## 1. 一句话定义

**Inductor** 是 `torch.compile` 流水线的**第三段（最终 codegen 后端）**——接 [[AOTAutograd]] 输出的 **joint FX Graph**（前向+反向，已 decomposition + functionalization），做**融合分析**（把相邻 elementwise/norm/reduction op 分组，同组融合成一个 kernel）、**epilogue 融合**（GEMM 后接 bias/activation 在寄存器接不落 HBM）、**codegen**（GPU 生成 [[Triton kernel开发]] 代码、CPU 生成 C++ 代码）、可选 **autotune**（试多 config 选最优），输出编译后的 fused kernel 供运行时调用。它是 compile 提速的**实际执行者**——Dynamo/AOTAutograd 出图，Inductor 把图变成优化 kernel。`max-autotune` 模式让它试多 tile/config，`reduce-overhead` 模式让它叠加 [[CUDA Graph与graph capture]]。是 PyTorch 2.0+ 自动 [[fused kernel]] 的核心引擎。

> [!note] 三句话定位
> - **是什么**：compile 第三段，接 joint graph 融合 + codegen 成 Triton/C++ kernel。
> - **为什么**：图的 op 要变成能跑的优化 kernel——融合减 HBM 往返（[[fused kernel]]）、Triton codegen 跨 GPU、autotune 选最优 config。
> - **与 [[Triton kernel开发]] 关系**：Inductor 是 Triton 的最大消费者——它自动生成 Triton 代码（elementwise/norm/reduction），用户无需手写；GEMM/attention 走专门 kernel（cuBLAS/FlashAttention）+ epilogue 融合。


## 2. 为什么需要它（动机与背景）

### 2.1 图要变成 kernel

Dynamo 出 FX Graph（op 级 IR），AOTAutograd 加反向 + decomposition + functionalization。但图本身不能跑——要变成 GPU/CPU 能执行的 kernel 代码。Inductor 是这个"图 → kernel"的 codegen 后端：分析图、融合 op、生成 Triton（GPU）或 C++（CPU）代码、编译成可加载 kernel。

### 2.2 融合是核心收益

eager 模式逐 op 走 dispatcher，相邻 elementwise op 各自 HBM 往返（[[fused kernel]] 2.1）。Inductor 把相邻 elementwise/norm 分组融合成一个 kernel——中间结果在 register/smem 传递不落 HBM，提 $I$（[[Roofline模型]]）。是 compile 提 [[MFU与算术强度]] 的实际手段。

### 2.3 Triton 作为 GPU codegen 目标

Inductor 不手写 CUDA，而是生成 [[Triton kernel开发]] 代码。原因：Triton 跨 GPU（生成 PTX/SPIRV）、autotune 原生支持、易扩展（Python DSL）。Inductor 把融合组的 op 翻译成 Triton 的 `@triton.jit` + `tl.load`/`tl.store`/`tl.dot`/`tl.reduce`。用户无需手写 Triton，Inductor 自动生成。

### 2.4 epilogue 融合

GEMM（$AB$）后接 bias/activation，若分开 $AB$ 落 HBM 再读。Inductor 借 [[CUTLASS与GEMM]] 的 epilogue 思想，把 GEMM 后的 elementwise 在寄存器接（$AB$ 在寄存器直接接 epilogue），不落 HBM。是 GEMM 路径的优化。

### 2.5 autotune 与 CUDA Graph 集成

`max-autotune` 模式让 Inductor 对 GEMM/attention 等试多 tile/config（BLOCK_M/N/K、num_warps、num_stages），实测 throughput 选最优（类似 Triton 的 `@triton.autotune`）。`reduce-overhead` 模式叠加 [[CUDA Graph与graph capture]]：把 Inductor 生成的 kernel 序列捕获成 graph，replay 压 launch。是 compile 三模式的后端实现。


## 3. 核心概念详解

### 3.1 三段流水中的位置

```
Dynamo (前向 FX Graph) → AOTAutograd (joint FX Graph) → Inductor (融合+codegen → kernel)
```

Inductor 输入 joint graph，输出编译的 Triton/C++ kernel（缓存到 `~/.cache/torchinductor`）。

### 3.2 融合分析

Inductor 遍历 joint graph，按融合规则分组：

| op 类型 | 融合性 | 处理 |
|---|---|---|
| elementwise（add/mul/relu） | 高 | 相邻 elementwise 融合一 kernel |
| reduction（sum/max） | 中 | reduction 后接 elementwise 融合（两遍 kernel） |
| GEMM（mm） | 低（专门 kernel） | 走 cuBLAS/手写 + epilogue 融合后接 elementwise |
| attention | 低（专门） | 走 FlashAttention 等 |
| broadcast | 中 | 与 elementwise 融合（带 broadcast） |

融合组的 op 在 register/smem 传递中间值，不落 HBM。是 [[fused kernel]] 的自动化。

### 3.3 Triton codegen

Inductor 把融合组翻译成 Triton：

```python
# Inductor 生成的 Triton (概念)
@triton.jit
def fused_kernel(x_ptr, w_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr+offs)         # 读 x
    w = tl.load(w_ptr+offs)         # 读 w
    y = tl.relu(x * w)              # 融合: mul + relu
    tl.store(y_ptr+offs, y)         # 写 y
```

原本 `x*w` 和 `relu` 两 op 两 kernel，融合成一 kernel（中间 `x*w` 在 register 不落 HBM）。Inductor 自动生成这种代码。

### 3.4 C++ codegen（CPU）

CPU 后端生成 C++（用 `<algorithm>`/`at::parallel_for`），不 Triton。性能不如 GPU Triton 但 CPU 也能融合减 HBM 往返（L1/L2 cache）。

### 3.5 epilogue 融合

```python
# 原: y = relu(x @ w + b)   三 op (mm, add, relu)
# Inductor: mm 走 cuBLAS, add+relu 在 mm 的 epilogue 寄存器接
# AB 在寄存器, 直接 add b + relu, 不落 HBM
```

借 [[CUTLASS与GEMM]] 的 epilogue 机制。GEMM 的 $I$ 提升（[[Roofline模型]]）。

### 3.6 autotune（max-autotune）

```python
# Inductor 对 mm 试多 config (概念)
configs = [
    {'BLOCK_M':128,'BLOCK_N':128,'BLOCK_K':32,'num_warps':8},
    {'BLOCK_M':64,'BLOCK_N':128,'BLOCK_K':64,'num_warps':4},
    ...
]
# 每个跑一遍, 实测 throughput 选最优
# 类似 triton.autotune, 但 Inductor 自动生成试的代码
```

`max-autotune` 模式启用。首次编译很慢（每 config 跑），后续跑最优 config。GEMM/attention 收益大，elementwise 小（tile 不敏感）。

### 3.7 CUDA Graph 集成（reduce-overhead）

`reduce-overhead` 模式：Inductor 生成的 kernel 序列被 [[CUDA Graph与graph capture]] 捕获成 graph，replay 时跳过 CPU launch。需 static shape（graph capture 要求）。decode/小 batch 收益大（launch 主导）。

### 3.8 缓存

Inductor 缓存生成的 Triton/C++ 代码到 `~/.cache/torchinductor/<hash>/`。hash = 源码 + shape + 模式 + config。重启后同 shape 复用（跳 codegen，秒级）。shape 变 → 新 hash → 新缓存 + 重编译。

### 3.9 fallback 到 dispatcher

Inductor 不认识的 op（如未注册 codegen 规则的 [[Custom C++ CUDA Operator]]）→ fallback 到 dispatcher 原 kernel（不融但能跑）。是 compile 的安全网——不融也能跑，只是收益打折。

### 3.10 与 cuBLAS/FlashAttention 的分工

- **GEMM**：Inductor 优先用 cuBLAS（最优 GEMM kernel），加 epilogue 融合。`max-autotune` 时可能试 Triton GEMM（[[CUTLASS与GEMM]] 风格）与 cuBLAS 比。
- **attention**：Inductor 调 [[FlashAttention]] 等 registered op，不自己 codegen attention（太复杂）。attention 的融合在 FlashAttention 内部完成。

Inductor 主导 elementwise/norm/reduction 融合，GEMM/attention 走专门 kernel + epilogue。


## 4. 数学原理 / 公式

### 4.1 融合的 byte 收益

$k$ 个相邻 elementwise op 不融合：HBM $\sim 2kN$ byte。融合：$\sim 2N$ byte。省 $k\times$（[[fused kernel]] 4.1）。$I$ 升 $k$ 倍，从 memory-bound 移向 compute-bound。

### 4.2 epilogue 的 $I$ 提升

GEMM 不融 epilogue：byte $2(MN+MN)$（$AB$ 落再读）。融：byte $\approx 2NK$（$AB$ 在寄存器接 epilogue）。$I = 2MNK/(2NK) = M$ 升。

### 4.3 autotune 的最优 config

设 $C$ 个 config，每 config throughput $\tau_c$（依赖 tile/num_warps/num_stages）。autotune 选 $\arg\max_c \tau_c$。首次编译 $\propto C$（每 config 跑），后续跑最优。最优 config 的 throughput 比默认高 10-50%（GEMM/attention）。

### 4.4 CUDA Graph 的 launch 压缩

Inductor 生成 $k$ 个 kernel，逐 op launch $k \cdot 5$ μs。CUDA Graph replay $\approx 1$ μs。省 $\sim k \cdot 5$ μs。decode 的几十 op 串省几百 μs。

### 4.5 fallback 的损失

未融合的 op 走 dispatcher：每 op $t_{\text{py}} + t_{\text{dispatch}} + t_{\text{launch}} + t_{\text{kernel}}$。若整图都 fallback（无融合），与 eager 无异。Inductor 融合覆盖率决定收益。


## 5. 代码示例（可选）

### 5.1 观察 Inductor 生成的 Triton

```python
import torch
@torch.compile
def f(x, w): return torch.relu(x * w)
x = torch.randn(1024, device='cuda'); w = torch.randn(1024, device='cuda')
out = f(x)   # 触发编译
# 生成的 Triton 在 ~/.cache/torchinductor/<hash>/
# 可 cat 看: fused kernel 含 mul + relu 一 kernel
```

### 5.2 epilogue 融合

```python
@torch.compile
def f(x, w, b): return torch.relu(torch.matmul(x, w) + b)
# Inductor: matmul 走 cuBLAS, +b 与 relu 在 matmul 的 epilogue 寄存器接
# AB 不落 HBM, 直接到 relu(x@w+b)
```

### 5.3 autotune 与 config

```python
@torch.compile(mode='max-autotune')
def f(x, w): return torch.matmul(x, w)
# Inductor 试多 BLOCK_M/N/K/num_warps, 选最优
# 首次编译很慢 (每 config 跑), 后续跑最优 config
# TORCHINDUCTOR_MAX_AUTOTUNE=1 控制范围
```

### 5.4 缓存与 fallback

```python
# 缓存: ~/.cache/torchinductor/<hash>/  含 .py (Triton) / .so (编译后)
# 清: rm -rf ~/.cache/torchinductor
# 不认识的 op (未注册 codegen 的 custom op) fallback dispatcher:
@torch.compile
def f(x): return my_custom_op(x)   # 若未在 Inductor 注册 codegen, 走 dispatcher 原 kernel
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[AOTAutograd]]（输入 joint FX Graph）、[[fused kernel]]（融合原理）、[[Roofline模型]]（融合提 $I$）、[[Triton kernel开发]]（GPU codegen 目标）、[[CUTLASS与GEMM]]（epilogue 融合借鉴）、[[kernel launch overhead]]（reduce-overhead 压 launch）。
- **下游（应用）**: [[torch.compile]]（Inductor 是其后端）、[[CUDA Graph与graph capture]]（reduce-overhead 集成）、[[SM utilization]]/[[MFU与算术强度]]（融合提 MFU）、[[activation memory]]（saved tensors 决策）、[[Custom C++ CUDA Operator]]（未注册的 fallback dispatcher）、[[dynamic shape]]（shape 变重 codegen）。
- **对比 / 易混**:
  - **Inductor vs [[Triton kernel开发]]**：Inductor 自动生成 Triton（elementwise/norm/reduction）；手写 Triton 用于 attention/GEMM 等复杂 kernel。Inductor 是 Triton 的最大消费者，两者互补。
  - **Inductor vs [[CUTLASS与GEMM]]**：Inductor 借 CUTLASS 的 epilogue 思想融 GEMM 后的 elementwise；CUTLASS 是手写 GEMM 库。Inductor 优先 cuBLAS，max-autotune 时试 Triton GEMM（CUTLASS 风格）。
  - **Inductor vs [[Dispatcher]]**：Inductor codegen 融合 kernel（绕过 dispatcher）；未融合的 op fallback dispatcher。compile 路径与 dispatcher 互补。
  - **Inductor vs [[AOTAutograd]]**：AOTAutograd 出 joint graph + decomposition；Inductor 接图融合 + codegen。AOTAutograd 是中段，Inductor 是终段。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 Inductor 手写 CUDA
> Inductor 生成 **Triton** 代码（GPU），不手写 CUDA。Triton 跨 GPU、autotune 原生、易扩展。CPU 生成 C++。用户不用碰 Triton，Inductor 自动生成。

> [!warning] 误区 2：以为所有 op 都融合
> GEMM/attention 走专门 kernel（cuBLAS/FlashAttention），不与 elementwise 融（只 epilogue 融）。reduction 单独留（难与 elementwise 完全融合）。Inductor 融合主要在 elementwise/norm 串。

> [!warning] 误区 3：autotune 总是值得
> max-autotune 首次编译很慢（每 config 跑）。elementwise 的 tile 不敏感（autotune 收益小）。autotune 主要给 GEMM/attention。先用 default，瓶颈 op 再局部 max-autotune。

> [!warning] 误区 4：忽视 fallback 的 op
> 未注册 codegen 规则的 custom op fallback dispatcher（不融但能跑）。若整图都 fallback，与 eager 无异。要 compile-friendly 用 `TORCH_LIBRARY` 注册 + 可能需 Inductor codegen 规则。

> [!warning] 误区 5：reduce-overhead 不需 static shape
> reduce-overhead 叠加 CUDA Graph，要求 static shape。变长 batch 用 default 或 `dynamic=True`。见 [[dynamic shape]]。

> [!warning] 误区 6：Inductor 改数值
> 融合改浮点累加顺序，$O(\epsilon)$ 差异（[[训推不一致]]）。需严格复现用确定性算法或固定编译。epilogue 融合的累加顺序与分开算不同。

> [!warning] 误区 7：忽视缓存膨胀
> 每个 shape 一份 Inductor 缓存（Triton code + 编译 .so）。shape 频繁变 → 缓存膨胀（GB 级）。`dynamic=True` 减缓存。定期清 `~/.cache/torchinductor`。


## 8. 延伸细节

### 8.1 Inductor 的 codegen 规则

`torch/_inductor/fx_passes/` 维护 codegen 规则：哪些 op pattern 融合成什么 Triton。可自定义规则（如把某 op pattern 融成更优 kernel）。是扩展 Inductor 融合能力的方式。

### 8.2 与 torch.export 的关系

`torch.export` 导出的 graph 可喂给 Inductor codegen（AOT 编译，不运行时 trace）。用于部署（导出 → Inductor 编译 → 跨环境跑）。与运行时 Inductor（compile 内）共用 codegen。

### 8.3 异构后端

Inductor 主要 GPU（Triton）+ CPU（C++）。异构后端（XLA/TPU）有专门 backend（非 Inductor）。Inductor 的 Triton 路径支持多种 GPU（NVIDIA/AMD via SPIRV）。

### 8.4 与 FSDP2/DTensor 的关系

FSDP2 的 DTensor op（[[DTensor]]）可被 Inductor codegen（生成 DTensor-aware kernel）。是 compile 与分布式结合的接口。但通信 op（NCCL all_reduce）在 graph 里是特殊 node，Inductor 不融合（单独执行）。详见 [[FSDP2]]。

### 8.5 内容来源

Inductor 整理自 PyTorch dev docs "TorchInductor"、`torch/_inductor/` 源码、Inductor tutorial。融合规则/epilogue/autotune 见 `torch/_inductor/fx_passes/` 与 `torch/_inductor/codegen/`。

---
相关: [[Compiler组件]] | [[torch.compile]] | [[TorchDynamo]] | [[AOTAutograd]] | [[Triton kernel开发]] | [[CUTLASS与GEMM]] | [[fused kernel]] | [[Roofline模型]] | [[kernel launch overhead]] | [[CUDA Graph与graph capture]] | [[SM utilization]] | [[MFU与算术强度]] | [[activation memory]] | [[Custom C++ CUDA Operator]] | [[dynamic shape]] | [[训推不一致]] | [[FSDP2]] | [[DTensor]]
