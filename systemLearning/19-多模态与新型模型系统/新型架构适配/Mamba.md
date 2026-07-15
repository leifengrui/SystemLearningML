# Mamba

> **所属章节**: [[新型架构适配]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: Mamba / selective SSM / 选择性状态空间模型 / 线性时间序列模型 / S6
> **难度**: 高（需懂 [[自注意力]] + [[RoPE]] + [[位置编码]] + RNN + HiPPO + 卷积+递归对偶）


## 1. 一句话定义

**Mamba** 是 Albert Gu & Tri Dao（arXiv:2312.00752，2023-12）提出的基于 **State Space Model（SSM，状态空间模型）** 的序列架构，核心是 **selective scan（选择性扫描 / 数据相关的状态更新）**——把传统 SSM（如 S4）的**输入无关**参数变成**输入相关**的：状态转移矩阵 $A,B,C$ 是输入 $x_t$ 的函数，让模型**按当前 token 选择性地记忆或遗忘**历史（关键 token 多记、无关 token 多忘），从而获得类 attention 的内容感知能力，同时保持 SSM 的**线性时间复杂度 $O(N)$**（vs Transformer [[自注意力]] 的 $O(N^2)$）；推理时是**固定大小状态的递归**（像 RNN，无 [[KV cache]] 膨胀，显存与序列长度无关、常数 $O(1)$ per step），训练时用**硬件感知的并行扫描算法**（像卷积，可并行）；Mamba block 把 selective SSM 与线性投影、门控结合，**无 attention 无 MLP**（用 SSM 替代 attention、用门控投影替代 MLP），Mamba-3B 在语言建模上超同规模 Transformer、匹敌两倍大 Transformer。**Mamba-2**（arXiv:2405.21060，"Transformers are SSMs"）引入 **State Space Duality（SSD，状态空间对偶）**，证明 SSM 与 attention 是结构化半可分矩阵的不同分解，核心层**快 2-8×**。Mamba 因**长上下文高效、推理显存省、并行训练**受关注。是 [[新型架构适配]] 的代表——挑战 Transformer 的 SSM 路线。

> [!note] 三句话定位
> - **是什么**：selective SSM——参数 $A,B,C$ 是输入函数，按 token 选择记忆/遗忘；$O(N)$ 训练并行、$O(1)$ 推理递归、无 KV cache 膨胀。
> - **为什么**：Transformer $O(N^2)$ 长序列贵、推理 KV cache 膨胀；传统 SSM 线性但输入无关、不能内容检索。Mamba 要兼得两者。
> - **与 [[自注意力]] 关系**：attention 全局精确检索但 $O(N^2)$；Mamba 线性高效但状态容量有限、难精确检索。催生 [[Hybrid架构]]。


## 2. 为什么需要它（动机与背景）

### 2.1 Transformer 的两个痛

Transformer（[[自注意力]]）统治序列建模，但：
1. **$O(N^2)$ 复杂度**：attention 是所有 token 两两打分，长序列 $N$ 大时算力与显存 $\propto N^2$。长上下文（100k+）贵。
2. **推理 [[KV cache]] 膨胀**：生成时缓存所有历史 token 的 KV，显存 $\propto N$，长序列爆显存（见 [[KV cache]]）。

### 2.2 传统 SSM 的局限

SSM（S4 等）是**线性时间** $O(N)$ 的序列模型（连续状态方程离散化），训练可并行（卷积模式）、推理是固定状态递归（RNN 模式，无 KV 膨胀）。但 S4 的参数 $A,B,C$ **输入无关**（固定），即"不论输入什么，状态转移规则一样"——无法做**内容相关的选择性记忆**（如 attention 按内容检索相关 token）。故 S4 在语言（强内容依赖模态）上不如 attention。

### 2.3 Mamba 的关键洞察

论文核心论断：**subquadratic 架构（线性 attention、SSM 等）不如 attention 的关键弱点是无法做 content-based reasoning（基于内容的推理）**。Mamba 的解法：**让 SSM 参数成为输入的函数**——$A(x_t),B(x_t),C(x_t)$，即 selective（数据相关）。这样模型能"看输入决定记什么、忘什么"，获得类 attention 的内容感知，同时保 SSM 的线性高效。

### 2.4 为什么受关注

- **长上下文**：$O(N)$ 可处理百万级序列（论文到 million-length）。
- **推理显存省**：固定状态递归，无 KV cache 膨胀，显存与 $N$ 无关。
- **并行训练**：selective scan 有硬件感知并行算法，训练像卷积可并行。
- **性能**：Mamba-3B 超 Transformer 同规模、匹敌 2× 大 Transformer。

### 2.5 Mamba-2 的改进

Mamba-2（arXiv:2405.21060）发现 SSM 与 attention 都可视为**结构化半可分矩阵**（semiseparable matrix）的不同分解，建立 **State Space Duality（SSD）**。这让 Mamba 的 selective SSM 能用 attention 的优化技术（如 [[FlashAttention]] 的 tiling），核心层**快 2-8×**，且理论统一 SSM 与 attention。


## 3. 核心概念详解

### 3.1 SSM 基础：连续状态方程

SSM 把序列建模为一个隐状态 $h(t)$ 的连续动态：

$$h'(t) = A h(t) + B x(t),\quad y(t) = C h(t)$$

- $x(t)$：输入（token embedding）
- $h(t)\in\mathbb R^N$：隐状态（固定大小 $N$，即 state size）
- $A,B,C$：状态转移/输入/输出矩阵
- $y(t)$：输出

这是连续时间 RNN。**固定状态 $N$**，推理时 $h$ 不随序列长增长（与 KV cache 膨胀对比）。

### 3.2 离散化：连续→离散

神经网络处理离散 token 序列，需把连续方程离散化（零阶保持 ZOH 等）：

$$\bar A = \exp(\Delta A),\quad \bar B = (\Delta A)^{-1}(\exp(\Delta A)-I)\cdot \Delta B$$

离散递归：

$$h_t = \bar A h_{t-1} + \bar B x_t,\quad y_t = C h_t$$

- $\Delta$：步长（可学，控制"看多远"）
- $\bar A,\bar B$：离散化后的转移/输入矩阵

这是 RNN 形式。训练时可写成卷积形式并行（$A$ 不变时）。

### 3.3 S4 vs Mamba：输入无关 vs selective

**S4**（Gu, Goel, Ré, ICLR 2022，arXiv:2111.00396）：$A,B,C,\Delta$ **输入无关**（固定参数，不随 $x_t$ 变）。可写成卷积（核 = $CB\bar A^k$）并行训练。用 **HiPPO** 初始化 $A$（记忆函数历史的特殊矩阵）。

**Mamba (S6)**：$B(x_t), C(x_t), \Delta(x_t)$ **输入相关**（是 $x_t$ 的函数，通过线性投影从 $x_t$ 算出）。$A$ 可保持输入无关（结构化）。关键：$\Delta$ 输入相关让模型"按 token 决定时间步长"——重要 token $\Delta$ 大（多更新状态=多记）、不重要 token $\Delta$ 小（少更新=多忘）。这是 selective 的核心。

```
S4:   B,C,Δ 固定        -> 卷积可并行, 但不能内容检索
Mamba: B(x),C(x),Δ(x)  -> selective 记忆/遗忘, 类 attention 内容感知
                          但破坏卷积并行 -> 需 selective scan 算法
```

### 3.4 selective scan 算法

$B,C,\Delta$ 输入相关后，卷积形式失效（核随 $x$ 变）。Mamba 设计**硬件感知的并行 selective scan** 算法：在 GPU 上用并行扫描（前缀和式）算递归 $h_t=\bar A_t h_{t-1}+\bar B_t x_t$，融合到 kernel 避免 materialization $h_{1:N}$ 的 IO。是 Mamba 能高效训练的工程关键。

### 3.5 Mamba block

Mamba 用一个简化的 block 替代 Transformer 的 attention+MLP：

```
input x
  |-- linear (in_proj, expand) -> x_branch, z_branch
  |-- conv1d (x_branch)  -- 局部上下文
  |-- SiLU
  |-- selective SSM (Δ,B,C 输入相关) -> y
  |-- z_branch gate (SiLU) -> y*z
  |-- linear (out_proj) -> output
```

无 attention、无 MLP（用 SSM 替 attention、用门控投影替 MLP）。Mamba block 堆叠成网络。

### 3.6 与 Transformer 对比

| 维度 | Transformer | Mamba |
|---|---|---|
| **核心算子** | [[自注意力]]（$O(N^2)$） | selective SSM（$O(N)$） |
| **训练** | 并行（attention 矩阵） | 并行（selective scan） |
| **推理** | 缓存所有 KV，$O(N)$ 显存 | 固定状态递归，$O(1)$ 显存 |
| **长序列** | $O(N^2)$ 贵 | $O(N)$ 高效 |
| **内容检索** | 强（全局精确） | 弱（状态压缩，难精确检索） |
| **长程记忆** | 无上限（全 KV） | 容量上限（固定状态 $N$） |
| **位置信息** | 需 [[位置编码]]（[[RoPE]]） | 内含（递归顺序天然有序） |
| **KV cache** | 有（[[KV cache]]） | 无（固定 state） |
| **结构** | attention + MLP block | SSM + 门控 block |

### 3.7 为什么 SSM 难精确检索

Mamba 状态 $h_t$ 是**固定大小 $N$** 的压缩。要把所有历史塞进 $N$ 维向量，必有信息损失。**精确检索**（如"第 3 个 token 是什么"）需保留原始 KV（Transformer 有，Mamba 没有）。这是 Mamba 在需精确召回的任务（如 long-context retrieval、MQAR）弱的根因。催生 [[Hybrid架构]]（部分层用 attention 补检索）。


## 4. 数学原理 / 公式

### 4.1 SSM 连续方程与离散化

连续：

$$h'(t)=Ah(t)+Bx(t),\quad y(t)=Ch(t)$$

ZOH 离散化（步长 $\Delta$）：

$$\bar A=\exp(\Delta A),\quad \bar B=(\Delta A)^{-1}(\bar A-I)\Delta B$$

递归：

$$h_t=\bar A h_{t-1}+\bar B x_t,\quad y_t=C h_t$$

### 4.2 S4：卷积形式（输入无关）

当 $A,B,C,\Delta$ 固定，展开递归：

$$y_t = \sum_{k=0}^{t} C\bar A^{t-k}\bar B x_k = (K * x)_t$$

核 $K=(CB\bar A^0, CB\bar A^1, \dots)$，是卷积，可 FFT 并行训练。

### 4.3 Mamba：selective（输入相关）

$$B_t=B(x_t),\quad C_t=C(x_t),\quad \Delta_t=\Delta(x_t)$$

$$\bar A_t=\exp(\Delta_t A),\quad \bar B_t=\Delta_t B_t$$

$$h_t=\bar A_t h_{t-1}+\bar B_t x_t,\quad y_t=C_t h_t$$

参数随 $x_t$ 变 → 不是固定核卷积 → 需 selective scan。

### 4.4 复杂度：$O(N)$ vs $O(N^2)$

- **Transformer attention**：$y=\text{softmax}(QK^T/\sqrt d)V$，$Q,K,V\in\mathbb R^{N\times d}$。$QK^T$ 是 $N\times N$，算力 $O(N^2 d)$，显存 $O(N^2)$。
- **Mamba selective SSM**：递归 $h_t=\bar A_t h_{t-1}+\bar B_t x_t$，每步 $O(N\cdot d)$（状态大小 $N$），总 $O(N\cdot d)$（线性 $N$）。并行 scan 后训练 $O(N d)$。

$$\text{Transformer}: O(N^2 d),\quad \text{Mamba}: O(N d)$$

长序列 $N$ 大时 Mamba 远省。

### 4.5 推理显存：$O(N)$ vs $O(1)$

- **Transformer 推理**：缓存所有历史 KV，显存 $\propto N$（见 [[KV cache]]），长序列爆。
- **Mamba 推理**：固定状态 $h$（大小 $N$，常数），每步 $h_t=\bar A_t h_{t-1}+\bar B_t x_t$，显存 $O(1)$ 与 $N$ 无关。

### 4.6 HiPPO 初始化

S4 用 **HiPPO**（High-order Polynomial Projection Operators）矩阵初始化 $A$：$A$ 是特殊构造的矩阵，使状态 $h$ 能"记忆"输入函数历史的 Legendre 多项式投影，给 SSM 长程记忆先验。Mamba 用 HiPPO 的对角+低秩近似初始化 $A$。是 SSM 能处理长序列的初始化关键。

### 4.7 Mamba-2：SSD 对偶

Mamba-2 证明 selective SSM 的输出矩阵 $y=CSSM(x)$ 是**结构化半可分矩阵** $M$ 与 $x$ 的积：$y=Mx$。而 attention 也是某半可分矩阵的积。故 SSM 与 attention 是同族（半可分矩阵的不同分解），称 **State Space Duality (SSD)**。这让 Mamba 能借 attention 的 tiling 优化（[[FlashAttention]] 式），核心层快 2-8×。


## 5. 代码示例（可选）

### 5.1 最简 selective SSM scan（教学）

```python
import torch

def selective_scan_naive(x, A, B, C, delta):
    """
    最简 selective SSM (朴素串行, 教学). 真实用并行 scan kernel.
    x: [L, d]      输入
    A: [N,]        状态转移 (输入无关, 结构化)
    B: [L, N]      输入相关 (从 x 投影)
    C: [L, N]      输入相关
    delta: [L]     输入相关步长
    返回 y: [L, d]
    """
    L, d = x.shape
    N = A.shape[0]
    h = torch.zeros(N)                # 固定大小状态
    ys = []
    for t in range(L):
        dA = torch.exp(delta[t] * A)  # 离散化 \bar A_t (输入相关)
        dB = delta[t] * B[t]          # \bar B_t
        h = dA * h + dB * x[t]        # 递归更新 (selective: delta,B 随 x)
        y = torch.matmul(C[t], h)     # 输出
        ys.append(y)
    return torch.stack(ys)            # [L, d]
```

### 5.2 Mamba block 示意

```python
class MambaBlock(torch.nn.Module):
    def __init__(self, d, state_N=16, expand=2):
        super().__init__()
        self.in_proj = torch.nn.Linear(d, expand*d*2, bias=False)  # 分 x,z 两支
        self.conv = torch.nn.Conv1d(expand*d, expand*d, 3, padding=1, groups=expand*d)
        self.x_proj = torch.nn.Linear(expand*d, state_N*2 + d, bias=False)  # B,C,delta from x
        self.dt_proj = torch.nn.Linear(d, expand*d, bias=True)
        self.A_log = torch.nn.Parameter(torch.randn(state_N))  # A (log 参数化)
        self.out_proj = torch.nn.Linear(expand*d, d, bias=False)

    def forward(self, x):  # x: [B,L,d]
        xz = self.in_proj(x)                      # [B,L,2*E*d]
        x_b, z = xz.chunk(2, dim=-1)              # 两支
        x_b = self.conv(x_b.transpose(1,2)).transpose(1,2)
        x_b = torch.nn.functional.silu(x_b)
        # 从 x_b 投影出 B, C, delta (输入相关 = selective)
       xBCd = self.x_proj(x_b)
        B, C, delta = xBCd.split([state_N, state_N, ...], dim=-1)
        delta = self.dt_proj(delta)               # delta 输入相关
        A = -torch.exp(self.A_log)                # A 结构化
        y = selective_scan(x_b, A, B, C, delta)   # 真实用 mamba_ssm kernel
        y = y * torch.nn.functional.silu(z)       # 门控
        return self.out_proj(y)
```

### 5.3 复杂度验证

```python
# Transformer attention FLOPs ~ 2*N*N*d (QK^T + softmaxV)
# Mamba SSM FLOPs ~ 2*N*N_state*d (N_state 固定小)
import math
N, d, Ns = 100000, 4096, 16
attn_flops = 2*N*N*d          # ~ 3.3e12
mamba_flops = 2*N*Ns*d        # ~ 1.3e9  -> ~2500x 少
print(f"attention: {attn_flops:.1e}, mamba: {mamba_flops:.1e}, ratio {attn_flops/mamba_flops:.0f}x")
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[自注意力]]（对比基准）、[[位置编码]]/[[RoPE]]（Mamba 内含顺序不需外部位置编码）、HiPPO（初始化）、SS4/S5（前作）、卷积+递归对偶。
- **下游（应用）**: [[Hybrid架构]]（Mamba+Transformer 混合）、[[Transformer与新型架构适配]]（框架对 SSM 适配）、Jamba/[[MoE]]（Jamba = Transformer-Mamba MoE）、长上下文、推理省显存。
- **对比 / 易混**:
  - **Mamba vs [[自注意力]]**：$O(N)$ 线性省但状态压缩难精确检索 vs $O(N^2)$ 贵但全局精确无上限。见 3.6 表。
  - **Mamba (S6) vs S4**：selective（$B,C,\Delta$ 输入相关）vs 输入无关。Mamba 内容感知，S4 不能。
  - **Mamba vs RNN**：Mamba 是结构化 SSM + HiPPO + selective + 并行 scan；传统 RNN 无并行训练、梯度消失。
  - **Mamba vs [[FlashAttention]]**：后者是 attention 的 IO 优化（仍 $O(N^2)$ FLOPs）；Mamba 是架构替换（$O(N)$）。


## 7. 常见误区与易错点

> [!warning] 误区 1：Mamba 完全替代 attention
> 不完全。Mamba 状态压缩，难精确检索（如 long-context recall、MQAR）。需精确召回的任务仍需 attention，催生 [[Hybrid架构]]。见 3.7。

> [!warning] 误区 2：Mamba 推理也有 KV cache
> 没有。Mamba 推理是固定状态递归 $h_t=\bar A_t h_{t-1}+\bar B_t x_t$，状态 $h$ 大小固定，显存 $O(1)$ 与序列长无关。这是相对 Transformer 的优势。但固定状态也意味着信息压缩损失。

> [!warning] 误区 3：Mamba 是 RNN
> 部分对。推理是 RNN 形式（递归），但训练是并行的（selective scan，类卷积）。且 $A$ 结构化 + HiPPO 初始化 + selective，比朴素 RNN 强得多、可并行训练、无梯度消失。

> [!warning] 误区 4：Mamba 不需要位置编码
> Mamba 递归天然有序（$h_t$ 依赖 $h_{t-1}$），不需像 Transformer 加 [[RoPE]]。但 conv1d 层也提供局部位置信息。是相对 Transformer 的简化。但混合架构中 attention 层仍需 [[位置编码]]。

> [!warning] 误区 5：selective = attention
> 不同。selective 是"参数随输入变"（$B,C,\Delta$ 输入相关），让状态转移数据依赖。attention 是"全局两两打分"。前者线性压缩，后者全局精确。selective 让 SSM 有内容感知，但不等于 attention 的检索力。

> [!tip] 实践：用 mamba-ssm 库
> 落地用官方 `mamba_ssm` 包（含 selective_scan CUDA kernel）。朴素串行慢得不可用。Mamba-2 用 `mamba2_ssm`，更快。详见 [[Transformer与新型架构适配]]。


## 8. 延伸细节

### 8.1 关键论文（已核实 arXiv ID）

- **Mamba**: Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. **arXiv:2312.00752**, 2023-12。Mamba-3B 超 Transformer 同规模、匹敌 2× 大；5× 推理吞吐；million-length 序列。
- **Mamba-2**: Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. **arXiv:2405.21060**, ICML 2024。SSD 框架，核心层 2-8× 快。
- **S4**: Gu, Goel, Ré. *Efficiently Modeling Long Sequences with Structured State Spaces*. arXiv:2111.00396, ICLR 2022（HiPPO 初始化 + 结构化 $A$）。

### 8.2 HiPPO 矩阵

HiPPO（arXiv:2008.08665，待核实）构造 $A$ 使状态 $h$ 等于输入函数历史在 Legendre 基下的系数，给 SSM"记忆历史多项式投影"的先验。是 S4/Mamba 长程能力的初始化基础。Mamba 用 HiPPO 的对角+低秩近似（$A=\Lambda - P Q^T$）保结构化又可 selective。

### 8.3 SSD 的工程意义

SSD 让 Mamba-2 能用 attention 的 tiling/[[FlashAttention]] 式 IO 优化，把 selective scan 的 kernel 写得更高效。也统一了"Mamba 是 attention 的某分解"的理论视角，为 [[Hybrid架构]]（attention+SSM 互换）铺路。

### 8.4 Mamba 的失败模式

Mamba 在**精确检索**（ Needle-in-a-haystack 的精确位置、MQAR multi-query associative recall）上不如 attention——状态压缩丢信息。这是 Jamba/[[Hybrid架构]] 加 attention 层的直接动机。在自由文本生成/常识上 Mamba 接近或超 attention。

### 8.5 内容来源

整理自 Mamba (arXiv:2312.00752)、Mamba-2 (arXiv:2405.21060) 论文及 S4 (arXiv:2111.00396) 前作。复杂度与显存对比据 [[自注意力]]、[[KV cache]]。SSD 对偶据 Mamba-2。HiPPO 编号 2008.08665 待核实。

---
相关: [[新型架构适配]] | [[Hybrid架构]] | [[Transformer与新型架构适配]] | [[自注意力]] | [[位置编码]] | [[RoPE]] | [[KV cache]] | [[FlashAttention]] | [[MoE]] | [[autoregressive decoding]]
