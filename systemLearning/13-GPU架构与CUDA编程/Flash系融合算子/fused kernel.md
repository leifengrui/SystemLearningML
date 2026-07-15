# fused kernel

> **所属章节**: [[Flash系融合算子]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: fused kernel / 算子融合 / kernel fusion / fused RMSNorm / fused Adam / fused AdamW
> **难度**: 中（需懂 [[Roofline模型]] + [[kernel launch overhead]]）


## 1. 一句话定义

**fused kernel（算子融合）** 是把原本**串行执行的多个小算子合成一个 kernel** 的优化——典型如 `RMSNorm + Residual + RoPE` 三个 op 融成一个、`Adam` 的 `m v update + weight update + bias correction` 融成一个、`GEMM + bias + activation` 在 [[CUTLASS与GEMM]] 的 epilogue 融合。融合的收益有两层：**减 HBM 往返**（中间结果不落 HBM，在 register/smem 直接传，提算术强度 $I$，[[Roofline模型]]）+ **减 launch 开销**（多 kernel launch 合一，省 $n$ 倍 [[kernel launch overhead]]）。它是 LLM 训练/推理提 [[MFU与算术强度]] 的基础手段，[[torch.compile]]/[[Inductor]] 的核心工作就是自动融合。

> [!note] 三句话定位
> - **是什么**：多 op 合一 kernel，中间结果不落 HBM，省 IO + 省 launch。
> - **为什么**：elementwise/norm/activation 等小 op 单跑是 memory-bound + launch 主导，串起来既喂不饱算力又卡 launch；融合后中间结果在 register 直传，$I$ 升、launch 降。
> - **自动 vs 手写**：torch.compile/Inductor 自动融合 elementwise；极致（CUTLASS epilogue、Triton 手写）需手写，性能更高。


## 2. 为什么需要它（动机与背景）

### 2.1 小算子的两重灾难

LLM 里大量小算子：RMSNorm、RoPE、activation（GELU/SiLU）、residual add、scale/bias、optimizer 的逐元素更新。每个单跑：

- **launch 主导**：算 1000 元素 GPU 跑 1 μs，launch 5 μs → 80% launch 浪费（[[kernel launch overhead]]）。
- **memory-bound**：elementwise 的 $I \approx 0.1$（[[Roofline模型]]），HBM 带宽喂不饱算力，SM 空转等数据。

串行跑 `RMSNorm → Residual → RoPE` 三 kernel：3 次 launch（15 μs）+ 3 次 HBM 往返（每 op 读输入写输出，中间 $N\times d$ 落 HBM 两次）。小且慢。

### 2.2 融合的两层收益

把三 op 合一 kernel：

1. **减 HBM 往返**：RMSNorm 的输出在 register/smem 直接传给 Residual，再传给 RoPE，**中间结果不落 HBM**。只 HBM 读一次输入、写一次最终输出。$I$ 升 → 趋 compute-bound。
2. **减 launch**：3 kernel launch 合 1，launch 开销 $15\to5$ μs。

LLM 的 decoder 层有大量这种"norm + add + activation + scale"的串，不融合的话 launch + IO 占 forward 时间可观。融合是提 MFU 的基础。

### 2.3 与 [[overlap strategy]] 的区别

- **fusion**：把**多 op 合一 kernel**，减 op 间 HBM 往返与 launch。是 kernel 内部优化。
- **overlap strategy**：把**计算与通信**（或计算与拷贝）放不同 stream 并发。是 kernel 间/引擎间调度。

两者正交，常并用：融合算子减 HBM，overlap 藏通信于计算。

### 2.4 自动 vs 手写

- **自动融合**：[[torch.compile]] 的 [[Inductor]] 后端自动把 elementwise/norm/reduction 融合成 [[Triton kernel开发]] 代码。覆盖大部分场景，零代码改动。
- **手写融合**：CUTLASS 的 epilogue（GEMM + bias + activation）、手写 Triton 的 RMSNorm+RoPE、fused Adam kernel。极致性能，需工程投入。

大框架两者并用：torch.compile 融合通用 op，关键路径（attention、GEMM epilogue、optimizer）手写。


## 3. 核心概念详解

### 3.1 融合的典型场景

| 融合 | 原多 op | 融合后 | 收益 |
|---|---|---|---|
| RMSNorm + Residual + RoPE | norm / add / rope 三 kernel | 一 kernel | 减 2 次 HBM 往返 + 2 launch |
| GEMM + bias + activation | gemm / bias / act 三 kernel | CUTLASS epilogue | 减 2 次 HBM 往返（$AB$ 在寄存器接 epilogue） |
| Adam fused | m_update / v_update / weight / bias_correct | 一 kernel | 减多次 HBM 往返（m/v/w 都在 HBM，融合减往返） |
| softmax + matmul | softmax / matmul | Flash 的 $P \to PV$ 融合 | [[FlashAttention]] 核心 |
| scale + mask + softmax | scale / mask / softmax | 一 kernel | 减 HBM 往返 |

### 3.2 epilogue 融合（CUTLASS）

GEMM 的 $AB$ 在寄存器，接 scale + bias + activation 融合写回，中间 $AB$ 不落 HBM。是 [[CUTLASS与GEMM]] 的 epilogue 机制，提 GEMM 的 $I$（[[Roofline模型]]）。

### 3.3 fused Adam（optimizer）

Adam 的更新：
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
\hat m = m / (1-\beta_1^t), \quad \hat v = v / (1-\beta_2^t), \quad w \mathrel{-}= \eta \hat m / (\sqrt{\hat v} + \epsilon)
$$

不融合：每步读 $m, v, w, g$（4 份）+ 多次中间写。融合：一 kernel 内读一遍 $g$、读写 $m, v, w$ 一次完成所有更新。fused Adam 是 ZeRO/FSDP 的标配，省 HBM 往返显著。

### 3.4 fused RMSNorm

RMSNorm：
$$
y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma
$$

单跑：算 mean square → 写 → 读回 → 除 → 写 → 读回 → 乘 $\gamma$。融合：一 kernel 内算 mean square（reduction）、就地除与乘。reduction 需跨 warp 同步（shared memory），是融合的难点之一。

### 3.5 fused RoPE

RoPE 的逐元素旋转（$\cos\theta, \sin\theta$ 乘加）是 elementwise，单跑 launch 主导。融合进 norm/add 的串里零成本。

### 3.6 reduction 的融合难点

含 reduction 的 op（norm 的 sum、softmax 的 max/sum）融合比纯 elementwise 难——reduction 需跨 warp 同步（shared memory + `__syncthreads`），不能纯 register 流水。Inductor 用 split-K + 两遍 reduction 处理，[[FlashAttention]] 用 online softmax 处理。手写常单独留 reduction kernel 或用 Triton 的 `tl.reduce`。

### 3.7 torch.compile 的自动融合

```python
@torch.compile
def f(x, w, b):
    return torch.relu(torch.matmul(x, w) + b)   # Inductor 融成 GEMM + epilogue 一 kernel
```

Inductor 把 PyTorch graph 的 elementwise/norm 融合成 Triton，GEMM 走 CUTLASS/cuBLAS + epilogue。见 [[torch.compile]]/[[Inductor]]。

### 3.8 graph capture 与 fusion 的关系

融合的 kernel 仍可被 [[CUDA Graph与graph capture]] 捕获（融合减 kernel 数，graph 压剩余 launch）。两者叠加：fusion 减 kernel + 减 HBM，graph 压 launch，decode 收益最大。


## 4. 数学原理 / 公式

### 4.1 IO 收益

设 $k$ 个串行 elementwise op，每 op 读 $N$ 写 $N$（fp16 $2N$ byte）：

$$
\text{不融合 HBM} = 2kN \text{ byte}, \quad \text{融合 HBM} = 2N \text{ byte（读一次写一次）}
$$

收益 $k\times$。$k=3$ 省 3× IO。

### 4.2 $I$ 的提升

不融合每 op $I \approx 1/(3s)$（elementwise）。融合后总 FLOP $\sum_i F_i$，byte $2N$，$I = \sum F_i / (2N)$ → 升 $k$ 倍 → 从 memory-bound 移向 compute-bound（[[Roofline模型]]）。

### 4.3 launch 收益

$$
\text{不融合 launch} = kL, \quad \text{融合} = L
$$

$k$ 大时省 $k$ 倍 launch。小 op 串收益大（如 decode 的几十个小 op）。

### 4.4 epilogue 的 $I$ 提升

GEMM 不融合 epilogue：FLOP $2MNK$，byte $2(MN+MN) = 4MN$（$AB$ 落 HBM 再读）。融合：byte $2(MK+NK+MN) \approx 2NK$，$I = 2MNK/(2NK) = M$ 升。

### 4.5 fused Adam 的 IO

不融合：每步 $m, v, w, g$ 多次读写 $\sim 8N$ byte。融合：读 $g$、读写 $m, v, w$ 各一次 $\sim 8N$？实际融合省的是中间版本（如 $\hat m, \hat v$ 不单独存），byte 省约 30-50%。

### 4.6 与 [[MFU与算术强度]] 的关系

fusion 减 byte → 提 $I$ → 提 MFU。是 MFU 提升手段之首（见 [[MFU与算术强度]] 8.1）。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟融合 IO 收益

```python
def fused_vs_unfused_io(N, k_ops, bytes_per_elem=2):
    """k 个串行 elementwise op: 不融合 vs 融合."""
    unfused = k_ops * 2 * N * bytes_per_elem   # 每 op 读+写
    fused = 2 * N * bytes_per_elem             # 读一次写一次
    return unfused, fused, unfused / fused

u, f, gain = fused_vs_unfused_io(4096*4096, 3)
print(f"不融合 {u/1e9:.2f} GB, 融合 {f/1e9:.2f} GB, 省 {gain}x")
```

### 5.2 PyTorch torch.compile 自动融合

```python
import torch
@torch.compile
def layer_op(x, gamma, beta):
    # norm + add + act 三 op 融合成一个 Triton kernel
    y = torch.relu(torch.layer_norm(x, x.shape[-1:], gamma, beta) + x)
    return y
# Inductor 生成 _fused kernel, nsys trace 只见 1 个
```

### 5.3 手写 fused Adam（Triton 风格）

```python
@triton.jit
def fused_adam_kernel(w_ptr, m_ptr, v_ptr, g_ptr, lr, beta1, beta2, eps, step, N,
                      BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(g_ptr+offs, mask=mask)
    m = tl.load(m_ptr+offs, mask=mask)
    v = tl.load(v_ptr+offs, mask=mask)
    m = beta1*m + (1-beta1)*g                    # 一遍读, 寄存器内更新
    v = beta2*v + (1-beta2)*g*g
    m_hat = m / (1-beta1**step)
    v_hat = v / (1-beta2**step)
    w = tl.load(w_ptr+offs, mask=mask) - lr*m_hat/(tl.sqrt(v_hat)+eps)
    tl.store(m_ptr+offs, m, mask=mask)           # 一遍写
    tl.store(v_ptr+offs, v, mask=mask)
    tl.store(w_ptr+offs, w, mask=mask)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GPU内存层级]]（register/smem 中间传递）、[[Roofline模型]]（减 byte 提 $I$）、[[kernel launch overhead]]（减 launch）。
- **下游（应用）**: [[torch.compile]]/[[Inductor]]（自动融合）、[[CUTLASS与GEMM]]（epilogue 融合）、[[Triton kernel开发]]（手写融合）、[[FlashAttention]]（$P\to PV$ 融合的典范）、[[MFU与算术强度]]（提 MFU 的首要手段）、[[Adam与AdamW]]（fused Adam）、[[layer norm]]/RMSNorm/[[RoPE]]（典型融合目标）、[[CUDA Graph与graph capture]]（叠加压 launch）。
- **对比 / 易混**:
  - **fused kernel vs [[overlap strategy]]**：fusion 减 op 间 HBM 往返（kernel 内）；overlap 是计算与通信并发（kernel 间/引擎间）。正交并用。
  - **fused kernel vs [[gradient checkpointing]]**：fusion 减 byte 提 $I$（性能）；checkpointing 重算换显存（省显存但增 FLOP）。前者提 MFU，后者降显存提 HFU。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为融合只是省 launch
> 不止。主要收益是减 HBM 往返（提 $I$），launch 只是附带。小 op 串时 launch 也重要，但大 op 融合（如 epilogue）的收益在 IO。

> [!warning] 误区 2：能融合的全融合就最优
> 含 reduction 的 op（norm/softmax）融合比纯 elementwise 难，强行融合可能引入同步开销降性能。有时保留 reduction 单独 kernel 更优。Inductor 自动权衡。

> [!warning] 误区 3：忽视 epilogue 融合
> GEMM 后的 bias/activation 不融合，$AB$ 落 HBM 再读，多一次往返，降 $I$。CUTLASS 的 epilogue 融合是 GEMM 性能关键。

> [!warning] 误区 4：fused Adam 当可选优化
> Adam 的 $m, v, w, g$ 都大（同 weight 尺寸），不融合每步多次往返 HBM，是 optimizer step 的瓶颈。fused Adam 是标配不是可选。

> [!warning] 误区 5：融合破坏数值一致性
> 融合改变浮点累加顺序，有 $O(\epsilon)$ 数值差异（与 [[训推不一致]] TIM 相关）。训练用融合、推理用不融合（或同融合）可能引入训推 logp 不一致。对齐用同 kernel。

> [!warning] 误区 6：torch.compile 融合万能
> Inductor 自动融合覆盖 elementwise/norm/reduction，但 attention/GEMM 仍走专门 kernel。复杂控制流、动态 shape 可能 graph break 不融合（见 [[graph break]]）。


## 8. 延伸细节

### 8.1 fusion 的极限：fused transformer block

极致融合把整个 decoder block（norm + attn + norm + mlp + residual）融成一 kernel——但太复杂、可维护性差、shape 变化难适配。工业上一般融合到 op 级（norm+add、epilogue），不做整 block。

### 8.2 Inductor 的融合策略

Inductor 用 fusion 规则：相邻 pointwise op 融合、reduction 后接 pointwise 融合（用两遍 kernel）、broadcast 处理。自动生成 Triton。复杂场景 fallback 到 eager。

### 8.3 fused kernel 与 [[推理 CUDA Graph]]

decode 的几十个小 op 经 fusion（减 kernel 数）+ graph capture（压 launch）双管齐下，把 decode 的 CPU launch 从几百 μs 压到几 μs。vLLM/SGLang 的 decode 路径核心。

### 8.4 fp8 训练的融合

fp8 训练需 amax 跟踪 + scale 更新 + 量化，这些若不融合每步多次 HBM 往返。fused fp8 kernel（amax + scale + 量化一体）是 [[FP8量化方案]] 训练的工程要点。

### 8.5 内容来源

fusion 的 IO 收益模型见 [[Roofline模型]] 与 CUDA Best Practices 的 Kernel Fusion 章节，fused Adam 见 PyTorch `torch.optim` 的 fused 实现，torch.compile 融合见 [[Inductor]] 文档。

---
相关: [[Flash系融合算子]] | [[GPU内存层级]] | [[Roofline模型]] | [[kernel launch overhead]] | [[torch.compile]] | [[Inductor]] | [[CUTLASS与GEMM]] | [[Triton kernel开发]] | [[FlashAttention]] | [[MFU与算术强度]] | [[overlap strategy]] | [[gradient checkpointing]] | [[Adam与AdamW]] | [[FP8量化方案]] | [[推理 CUDA Graph]]
