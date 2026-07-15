# FlashAttention

> **所属章节**: [[Flash系融合算子]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: FlashAttention / Flash-Attn / FA / IO-aware attention / online softmax attention
> **难度**: 高（需懂 [[GPU内存层级]] + [[Roofline模型]] + attention 数学）


## 1. 一句话定义

**FlashAttention** 是 Dao-AILab/Stanford 提出的 **IO-aware 精确（mathematically exact）注意力**算法——通过 **tiling**（把 $Q,K,V$ 切块分批载入 shared memory）+ **online softmax**（边累加边修正归一化因子，不需先算完整 $N\times N$ 的 $S=QK^T$）+ **kernel fusion**（$QK^T \to \text{softmax} \to PV$ 三步在单 kernel 内、中间结果不落 HBM），把 attention 的 HBM 读写从 $O(N^2)$ 降到 $O(N)$，**数值上与标准 attention 完全一致**（不是近似），但快 2-4×、省显存（不存 $N\times N$ 的 $S,P$）。它是 2022 年后 LLM 训练/推理 attention 的事实标准，几乎所有主流框架（PyTorch、Megatron、vLLM、SGLang）的 attention 路径都基于 FlashAttention-2/3，是 [[Roofline模型]]"减 HBM 访问提 $I$"的教科书级典范。

> [!note] 三句话定位
> - **是什么**：tiling + online softmax + 三步融合，attention 的 HBM IO 从 $O(N^2)$ 降到 $O(N)$，数值精确。
> - **为什么**：标准 attention 的 $S=QK^T$（$N\times N$）要写回 HBM 再读回算 softmax，HBM 访问主导（memory-bound）；Flash 把它全塞进 smem/register，跳过 HBM。
> - **版本**：FA1（2022）奠基；FA2（2023）减 non-matmul FLOP、重排 warp；FA3（2024，Hopper）用 TMA + warp specialization + FP8。与 [[FlashInfer]]（变长/稀疏 KV 的库）互补。


## 2. 为什么需要它（动机与背景）

### 2.1 标准 attention 是 memory-bound 的灾难

标准 attention：$S = QK^T \in \mathbb{R}^{N\times N}$，$P = \text{softmax}(S)$，$O = PV$。

- FLOP：$4N^2 d$（$QK^T$ $2N^2d$ + $PV$ $2N^2d$），随 $N$ 平方增长。
- HBM 访问：$S$ 写回 HBM（$N^2$）→ 读回算 softmax → 写 $P$（$N^2$）→ 读回算 $PV$。**多次 $N^2$ 级 HBM 往返**。

按 [[Roofline模型]]，attention 的 $I$ 低（FLOP/byte 比 ~$\sqrt{}$ 级，远 < ridge point），**强 memory-bound**——算单元等数据。$N$ 大（长上下文）时 HBM 读写爆炸，显存（$N^2$ 的 $S,P$）也爆炸。这是长序列训练/推理慢且爆显存的根因。

### 2.2 中间结果 $S,P$ 不该落 HBM

$S,P$ 是中间结果，最终只要 $O$。但标准实现把它们写 HBM 再读，因 softmax 需整行归一化（要 max/sum over $N$）。**online softmax** 打破这个依赖：分块累加，用 running max/sum 修正，不需先有完整 $S$ 行。

### 2.3 tiling 把热数据压进 smem

$Q,K,V$ 切块进 shared memory（小、快），一个 block 在 smem 内算完部分 attention，累加修正，最终只写 $O$ 一次。中间 $S,P$ 全在 register/smem，不落 HBM。这是 FlashAttention 的核心思想，与 [[GPU内存层级]] 的 tiling 哲学一脉相承。

### 2.4 数值精确，不是近似

FlashAttention 不是 sparse attention、不是 linear attention、不是低秩近似——它与标准 attention **数值一致**（除浮点累加顺序误差外）。快是因为减 IO，不是减计算。这点常被误读。

### 2.5 版本演进

- **FA1**（2022）：奠基 tiling + online softmax，HBM IO $O(N^2)\to O(N)$。
- **FA2**（2023）：减 non-matmul FLOP（softmax 的 exp/sum 用更少）、重排 warp 减少 sync、提 occupancy。比 FA1 又快 ~2×。
- **FA3**（2024，Hopper 专属）：用 TMA（硬件异步 global→smem）、warp specialization（producer/consumer warp）、FP8 support。H100 上比 FA2 快 1.5-2×。
- **FlashAttention-3** 与 [[FlashInfer]] 都面向新一代，但 FA3 重 Hopper 极致，FlashInfer 重可变长/稀疏的工程化。


## 3. 核心概念详解

### 3.1 标准 attention 的 IO 灾难

```python
# 标准 attention (naive, 不用 flash)
S = Q @ K.transpose(-1,-2) / sqrt(d)     # N×N, 写 HBM
P = softmax(S, dim=-1)                    # 读 S, 写 P, N×N HBM
O = P @ V                                 # 读 P, 写 O
```

HBM 读写：写 $S$（$N^2$）+ 读 $S$（$N^2$）+ 写 $P$（$N^2$）+ 读 $P$（$N^2$）+ 写 $O$（$N$）。$\sim 4N^2$ bytes。$N=8192$, fp16：$4 \times 8192^2 \times 2 \approx 512$ MB/序列。batch × head 多了 → TB 级 HBM 往返，慢且爆显存。

### 3.2 online softmax：分块归一化

标准 softmax：$P_{ij} = \exp(S_{ij} - m)/\ell$，$m = \max_j S_{ij}$，$\ell = \sum_j \exp(S_{ij}-m)$。需整行 $S$ 才能算 $m,\ell$。

online softmax 分块处理：

1. 处理 block 1：算 $S_1$，$m_1 = \max(S_1)$，$\ell_1 = \sum \exp(S_1-m_1)$，$O_1 = P_1 V_1$。
2. 处理 block 2：算 $S_2$，$m_{12} = \max(m_1, m_2)$，修正 $\ell_{12} = \ell_1 e^{m_1-m_{12}} + \ell_2 e^{m_2-m_{12}}$，$O_{12} = O_1 e^{m_1-m_{12}} + P_2 V_2$（用新 max 修正旧 $O$）。
3. 累加到全 $N$ 完成，得精确 $O$。

每块只在 smem/register 内有部分 $S,P$，全 $N$ 累加后输出 $O$。**中间 $S,P$ 从不落 HBM**。

### 3.3 tiling 与 block 结构

FlashAttention 的 block：

- 外层循环遍历 $K,V$ 的 block（block size $B_r \times B_c$）。
- 内层：$Q$ block 与 $K,V$ block 在 smem 算部分 $S,P$，online 修正 $O$。
- 一个 block 的 $S,P$ 在 register/smem，算完即丢，不写 HBM。

block size 选 occupancy 友好（如 $B_r=128$, $B_c=128$），见 [[occupancy分析]]。

### 3.4 HBM IO 从 $O(N^2)$ 到 $O(N)$

标准：$4N^2$ HBM。Flash：$Q,K,V,O$ 各读写一次 ≈ $4N \cdot d$ HBM（不随 $N^2$）。$N$ 大时降几个数量级。$I$ 升 → 从 memory-bound 趋 compute-bound（[[Roofline模型]]）。

### 3.5 显存收益

标准要存 $S,P$（$N^2$），反向时还要重算或存中间值。Flash 不存 $S,P$（kernel 内即抛），显存 $O(N)$（只 $Q,K,V,O$）。反向时用 $O$ 的 checkpoint 重算（recalculate，见 [[gradient checkpointing]] 思想），trade 计算换显存。

### 3.6 backward 的重算

标准 backward 需 $S,P$（或存下的中间值）算梯度。Flash 不存，backward 时重算 $S,P$（recalculation）。多 ~33-50% FLOP，但省 HBM 读写，整体仍快（IO 是主因）。这是 HFU > MFU 的来源之一（[[MFU与算术强度]]）。

### 3.7 FA2 的改进

- 减 non-matmul FLOP（softmax 的 rescale 用更少指令）。
- $Q$ 外循环、$K,V$ 内循环重排，减 shared memory 读写。
- warp 重排减 `__syncthreads`。
- 提 occupancy（更小 smem 占用）。

### 3.8 FA3（Hopper）

- TMA（硬件 global→smem 异步拷贝）替代 `cp.async`。
- warp specialization：producer warp 专 load，consumer warp 专 compute，流水。
- FP8 attention（E4M3，per-tensor scale）。
- in-register softmax（FA2 用 smem）。

### 3.9 与 FlashInfer / vLLM 的集成

[[FlashInfer]] 是变长/稀疏 KV 的 attention 库（基于 Flash 思想扩展），vLLM/SGLang 的 [[PagedAttention]] 也吸收 Flash 的 tiling。三者在 decode 的变长 batch、prefix cache、block-sparse 上有差异，但底层都是"减 HBM IO、tile 进 smem、online 归一化"。


## 4. 数学原理 / 公式

### 4.1 标准 attention

$$
S = QK^T/\sqrt{d} \in \mathbb{R}^{N\times N}, \quad P = \text{softmax}(S), \quad O = PV
$$

$m = \max_j S_{ij}$（行 max），$\ell = \sum_j \exp(S_{ij}-m)$（行 sum）。$P_{ij} = \exp(S_{ij}-m)/\ell$。

### 4.2 online softmax 的递推

分块处理，第 $k$ 块后维持 $(m_k, \ell_k, O_k)$：

$$
m_k = \max(m_{k-1}, \max(S_k)), \quad \ell_k = \ell_{k-1} e^{m_{k-1}-m_k} + \sum_j \exp(S_{k,j}-m_k)
$$

$$
O_k = O_{k-1} e^{m_{k-1}-m_k} + \sum_j \frac{\exp(S_{k,j}-m_k)}{\ell_k} V_{k,j} \cdot \ell_k'
$$

（$\ell_k'$ 为本块未归一的部分和，最终除以 $\ell_N$。）最终 $O_N/\ell_N$ 即精确 attention 输出。**关键：每步只用本块 $S_k$，不需全 $S$ 行**。

### 4.3 IO 复杂度对比

- 标准：HBM $\Theta(N^2 d + N^2)$（$S,P$ 多次往返）。
- Flash：HBM $\Theta(N d)$（$Q,K,V,O$ 各读写一次）。

降一个 $N$ 因子。$N$ 大（$8192+$）时降 100× 量级。

### 4.4 算术强度提升

标准 $I \approx \frac{4N^2 d}{4N^2} = d$（FP 一定，byte 随 $N^2$）。Flash $I \approx \frac{4N^2 d}{4Nd} = N$（byte 只 $Nd$）→ $I$ 随 $N$ 增 → 趋 compute-bound（[[Roofline模型]]）。

### 4.5 数值精确性

online softmax 与标准 softmax 在浮点累加顺序上不同，有 $O(\epsilon)$ 误差，但数学等价。不是近似算法。FP16/BF16 下与标准 attention 的差异在数值精度范围内。

### 4.6 backward 重算

反向需 $\frac{\partial O}{\partial Q}, \frac{\partial O}{\partial K}, \frac{\partial O}{\partial V}$，依赖 $S,P$。Flash 不存，反向时从 $Q,K,V$ 重算 $S,P$（forward 的 tiling 再跑一遍）。多 $O(N^2 d)$ FLOP，但省 HBM 读 $S,P$（$N^2$ 级）。trade FLOP for IO，整体快。

### 4.7 与 [[Roofline模型]] 的印证

Flash 不改 FLOP（仍 $4N^2d$），只改 HBM byte（$N^2\to N$）。$I$ 升 → kernel 从斜线段移向水平段 → throughput 升。是"减 byte 提 $I$"的典范。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟 online softmax

```python
import math
def flash_attention_ref(Q, K, V, block_c=4):
    """分块 online softmax 模拟 FlashAttention (数值与标准一致)."""
    N, d = Q.shape
    O = np.zeros_like(V); m = -np.inf*np.ones(N); l = np.zeros(N)
    for k0 in range(0, N, block_c):
        S = Q @ K[k0:k0+block_c].T / math.sqrt(d)   # 部分 S
        m_blk = S.max(-1)                            # 本块 max
        m_new = np.maximum(m, m_blk)                 # 全局 max 更新
        alpha = np.exp(m - m_new)                    # 旧 O/l 修正因子
        l = l * alpha + np.exp(S - m_new[:, None]).sum(-1)
        O = O * alpha[:, None] + (np.exp(S - m_new[:, None]) @ V[k0:k0+block_c])
    O = O / l[:, None]                              # 最终归一
    return O

# 对照标准
def standard_attn(Q,K,V):
    S = Q@K.T/math.sqrt(Q.shape[-1]); P = np.exp(S-S.max(-1,keepdims=True))/np.exp(S-S.max(-1,keepdims=True)).sum(-1,keepdims=True)
    return P@V
# 两者数值一致 (除浮点误差)
```

### 5.2 PyTorch FlashAttention 调用

```python
import torch
from flash_attn import flash_attn_func
q = torch.randn(B, N, H, d, device='cuda', dtype=torch.bfloat16)
k = v = torch.randn(B, N, H, d, device='cuda', dtype=torch.bfloat16)
o = flash_attn_func(q, k, v, causal=True)        # 数值精确, 比 naive 快 2-4x
# 对照: o_ref = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
```

### 5.3 真实命令对照

```python
# nsys trace 看 flash kernel vs 多个小 kernel
# nsys profile -t cuda,nvtx python attn.py
# 标准: matmul + softmax + matmul (3 kernel, 中间 N×N 落 HBM)
# flash: _flash_attn (1 kernel, 中间在 smem)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GPU内存层级]]（smem tiling/register）、[[Roofline模型]]（memory-bound → compute-bound）、[[多头注意力]]/[[自注意力]]（attention 数学）、[[softmax]] 的 online 归一化。
- **下游（应用）**: [[FlashInfer]]（变长/稀疏扩展）、[[PagedAttention]]（decode 的 KV block，吸收 Flash tiling）、[[KV cache]]（Flash 减 attention 读写也减 KV cache 压力）、[[gradient checkpointing]]（backward 重算思想同源）、[[长上下文]]（Flash 让长序列可训）、[[MFU与算术强度]]（提 $I$ 升 MFU）。
- **对比 / 易混**:
  - **FlashAttention vs sparse/linear attention**：Flash 是精确（数值一致），sparse/linear 是近似（减计算或改模型）。Flash 不改数学只改 IO。
  - **FlashAttention vs [[PagedAttention]]**：Flash 是 attention 算法（减 IO）；PagedAttention 是 KV 显存管理（块表映射）。vLLM 的 decode attention 常两者结合。
  - **FlashAttention vs [[FlashInfer]]**：Flash 是算法 + kernel；FlashInfer 是工程库（变长/稀疏/多级 cache），底层用 Flash 思想。
  - **FA1 vs FA2 vs FA3**：见 2.5，性能递进，FA3 Hopper 专属。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 FlashAttention 是近似
> 它是**精确**的，与标准 attention 数值一致（除浮点累加顺序误差）。不是 sparse/linear/低秩近似。快是因减 IO，不减计算。

> [!warning] 误区 2：以为 FlashAttention 减 FLOP
> forward FLOP 不减（仍 $4N^2d$）。backward 多重算（HFU > MFU）。它减的是 HBM byte（$N^2\to N$），提 $I$。

> [!warning] 误区 3：忽视显存收益
> 标准要存 $S,P$（$N^2$），长序列爆显存。Flash 不存，显存 $O(N)$。长上下文训练能跑全靠 Flash。

> [!warning] 误区 4：FA 版本不分
> FA1/2/3 性能差很多。H100 不用 FA3 浪费（FA3 用 TMA/warp specialization/FP8，比 FA2 快 1.5-2×）。A100 用 FA2。

> [!warning] 误区 5：忽视 backward 重算代价
> Flash backward 重算 $S,P$，HFU > MFU（多 ~33-50% FLOP）。报效率要分清 HFU 与 MFU（[[MFU与算术强度]]）。trade 计算换 IO/显存。

> [!warning] 误区 6：以为 causal mask 必须先算全 $S$
> causal 下 Flash 直接在 tiling 内跳过上三角 block（不载不算），进一步减 IO/FLOP ~50%。不需先算全 $S$ 再 mask。


## 8. 延伸细节

### 8.1 FA3 的 TMA 与 warp specialization

H100 的 TMA 让 global→smem 拷贝不经寄存器（硬件直接 DMA），warp specialization 让一组 warp 专 load（producer）、另一组专 compute（consumer），自然流水，比 FA2 的软件 double buffer 更高效。详见 [[CUTLASS与GEMM]] 的 8.1。

### 8.2 FlashAttention 的可变长扩展

[[FlashInfer]] 把 Flash 思想扩展到可变长 batch（每序列长度不同）、block-sparse KV（只算非零块）、prefix cache（复用前缀 attention）。是 vLLM/SGLang decode 的主力。

### 8.3 与 [[PagedAttention]] 的结合

vLLM 的 decode：KV 在 block table 映射的非连续物理块，attention kernel 要支持"逻辑连续、物理分块"的访问。Flash 的 tiling 天然适配——按 block table 索引 KV block 进 smem。两者结合是 vLLM 的 attention 路径核心。

### 8.4 sliding window / local attention

长上下文常用 sliding window（只算最近 $W$ 个 token），Flash 的 tiling 可跳过 window 外的 block，进一步减 IO。Mistral/Llama 长上下文用此。

### 8.5 FP8 attention

FA3 支持 FP8（E4M3）attention，per-tensor scale。但 attention 的 softmax 对数值敏感，FP8 需小心 scale。是 [[FP8量化方案]] 在 attention 的应用，待核实成熟度。

### 8.6 内容来源

FlashAttention 算法整理自 Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022) 与 FA2/3 论文，online softmax 的数值技巧见论文附录，与 [[Roofline模型]] 互补印证 IO 收益。

---
相关: [[Flash系融合算子]] | [[GPU内存层级]] | [[Roofline模型]] | [[自注意力]] | [[多头注意力]] | [[KV cache]] | [[PagedAttention]] | [[FlashInfer]] | [[gradient checkpointing]] | [[MFU与算术强度]] | [[长上下文]] | [[FP8量化方案]]
