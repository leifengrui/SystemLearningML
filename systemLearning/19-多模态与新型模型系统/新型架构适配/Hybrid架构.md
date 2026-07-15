# Hybrid架构

> **所属章节**: [[新型架构适配]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: Hybrid SSM-Attention / Transformer-Mamba 混合 / attention+SSM hybrid / Jamba-style 架构 / eidetic+fading memory
> **难度**: 高（需懂 [[Mamba]] + [[自注意力]] + [[MoE]] + [[位置编码]] + 训推系统适配）


## 1. 一句话定义

**Hybrid 架构** 是把 **Transformer（[[自注意力]]）** 与 **SSM（[[Mamba]]）** 在同网络内**混合**的架构范式——纯 Mamba 有局限（固定状态压缩→长程记忆容量有限、难学精确检索如 MQAR/needle-in-haystack），纯 Transformer 长序列 $O(N^2)$ 贵且推理 [[KV cache]] 膨胀；Hybrid **部分层用 attention**（提供全局精确检索/可编辑上下文，称 eidetic memory "全息记忆"）+ **部分层用 SSM**（提供 $O(N)$ 高效长序列建模/状态压缩，称 fading memory "衰减记忆"），两者互补，兼得 SSM 的线性复杂度与 attention 的全局精确性。典型 pattern：**"N 层 SSM + 1 层 attention 间隔"**（如每 5~8 层 Mamba 后插 1 层 attention，Jamba 用 8:1），或 **"前 SSM 后 attention"** / **"交错"**。代表作 **Jamba**（arXiv:2403.19887，AI21，Transformer-Mamba **+ [[MoE]]** 混合，256K 上下文，单 80GB GPU 可放）、**Jamba-1.5**（arXiv:2408.12570，94B/12B active）、Falcon-Mamba、Zamba 等。理论框架 B'MOJO（arXiv:2407.06324）把 Transformer/Mamba/Jamba 统一为 eidetic+fading memory 的可调混合。**为什么 Hybrid**：纯 SSM 召回弱、纯 attention 长序列贵，混合是当前长上下文 + 召回 + 效率的工程最优解。是 [[新型架构适配]] 中介于纯 Transformer 与纯 Mamba 之间的主流落地路线。混合后训练/推理系统需适配（见 [[Transformer与新型架构适配]]）。

> [!note] 三句话定位
> - **是什么**：部分层 attention（精确检索/全局）+ 部分层 SSM（线性高效/长序列）混合。典型 N:1 间隔（Jamba 8:1）+ [[MoE]] 增容。
> - **为什么**：纯 Mamba 召回弱（状态压缩丢信息）、纯 attention 长序列贵（$O(N^2)$+KV 膨胀）。Hybrid 兼得两者。
> - **与 [[Mamba]]/[[自注意力]] 关系**：是两者的混合，统一为 eidetic（全息，attention）+ fading（衰减，SSM）memory（B'MOJO 框架）。


## 2. 为什么需要它（动机与背景）

### 2.1 纯 Mamba 的两个局限

[[Mamba]] 线性高效、推理省显存，但：
1. **长程记忆容量有限**：状态 $h$ 固定大小 $N$，所有历史塞进 $N$ 维向量，必有信息损失。长序列后期无法保留早期细节。
2. **难学精确检索**：MQAR（multi-query associative recall）、needle-in-haystack 精确位置等任务需保留原始 token（attention 有 KV、Mamba 无），Mamba 弱。这是结构性的（状态压缩）。

### 2.2 纯 Transformer 的两个痛

[[自注意力]] 全局精确、召回强，但：
1. **$O(N^2)$ 复杂度**：长上下文（256K+）算力显存爆。
2. **推理 KV cache 膨胀**：缓存所有历史 KV，显存 $\propto N$，长序列爆显存（见 [[KV cache]]）。

### 2.3 Hybrid 的兼得思路

attention 与 SSM 互补：
- attention = **eidetic memory**（全息记忆，有限 span 内全保留，精确但贵）
- SSM = **fading memory**（衰减记忆，无限 span 内压缩衰减，高效但模糊）

混合：大部分层用 SSM（省、长序列高效），少量层用 attention（补精确召回）。既省钱又保召回。

### 2.4 pattern 探索的结论

社区实验发现：**纯 Mamba 召回差，但加少量 attention（如每 8 层 1 层）召回即恢复**且仍保 SSM 大部分效率。Jamba 的消融：N:1 间隔（8 Mamba : 1 attention）+ [[MoE]] 是大模型最优配置之一。这是 [[新型架构适配]] 当前长上下文工程最优解。


## 3. 核心概念详解

### 3.1 eidetic vs fading memory（B'MOJO 框架）

B'MOJO（arXiv:2407.06324）统一视角：
- **eidetic memory（全息记忆）**：有限 span 内全保留（精确），代表 attention 的 KV context。
- **fading memory（衰减记忆）**：无限 span 内压缩衰减（模糊），代表 SSM 的固定 state。
- Transformer = 纯 eidetic；Mamba = 纯 fading；Jamba = eidetic+fading 混合。
- B'MOJO 提出可无缝调制两者的模块，Transformer/Mamba/Jamba 是其特例。

### 3.2 典型 pattern

```
Pattern 1: N SSM + 1 attention 间隔 (Jamba 8:1)
  [SSM][SSM]...[SSM][ATTN] x 重复
  优点: 大部分层高效, 少量层补召回

Pattern 2: 前 SSM 后 attention
  [SSM]*k + [ATTN]*m
  优点: 前期压缩, 后期精确

Pattern 3: 交错
  [SSM][ATTN][SSM][ATTN]...
  优点: 召回密集, 但效率降

Pattern 4: block-level vs head-level
  block (Jamba): 块级交替层
  head (Hymba): attention head 内嵌 SSM 重要性信号
  score-level (SISA): attention score 内加 SSM 项
```

### 3.3 Jamba 架构

Jamba（arXiv:2403.19887）= Transformer-Mamba **+ [[MoE]]**：
- 交错 Transformer 与 Mamba block（约 8:1）。
- 部分层加 [[MoE]]（增容量、控 active 参数）。
- 256K 上下文，单 80GB GPU 可放（active 参数小）。
- 高吞吐、低显存、benchmark 接近/超 Transformer。

Jamba-1.5（arXiv:2408.12570）：Large 94B active / Mini 12B active，ExpertsInt8 量化，256K context。

### 3.4 为什么加 [[MoE]]

Hybrid 加 [[MoE]] 是为**增容量而不增 active 算力**——MoE 每次只激活部分专家，可在保持 active 参数小的同时扩总参数，弥补 SSM 层压缩的容量损失。Jamba 即 SSM+attention+MoE 三合一。

### 3.5 与纯 Transformer / 纯 Mamba 对比

| 维度 | 纯 Transformer | 纯 Mamba | Hybrid (Jamba 式) |
|---|---|---|---|
| **复杂度** | $O(N^2)$ | $O(N)$ | ~$O(N)$（大部分 SSM 层） |
| **推理显存** | $O(N)$（KV 膨胀） | $O(1)$（固定 state） | 混合（attention 层 KV + SSM state） |
| **精确召回** | 强 | 弱 | 强（attention 层补） |
| **长程记忆** | 无上限 | 容量上限 | 改善（attention 定期刷新） |
| **长上下文** | 贵 | 高效 | 高效 |
| **结构** | attention+MLP | SSM+门控 | SSM+attention(+MoE) |

### 3.6 混合后系统适配

Hybrid 改变了算子组合，训练/推理系统需适配：
- **推理**：SSM 层无 KV cache（固定 state），attention 层有 KV。引擎需同时管 state 与 KV 两类缓存。见 [[Transformer与新型架构适配]]。
- **训练**：SSM 层 selective scan 算子 + attention 层（[[FlashAttention]]）。CP（context parallel）需切 SSM 的 scan 与 attention 的 ring attention。见 [[Transformer与新型架构适配]]。
- **[[位置编码]]**：SSM 层不需（递归有序），attention 层需（[[RoPE]]）。混合模型只 attention 层加 RoPE。


## 4. 数学原理 / 公式

### 4.1 eidetic vs fading 的形式化

B'MOJO：记忆 $M$ 对历史 $\{x_1,\dots,x_t\}$ 的表示：
- **eidetic**：$M_t = [x_1,\dots,x_t]$（全保留，有限 span）。attention 的 KV context。
- **fading**：$M_t = h_t = \bar A_t h_{t-1}+\bar B_t x_t$（压缩，无限 span）。SSM 的 state。

Hybrid：$M_t = \text{concat}(\text{eidetic window}, h_t)$，即近期全保留 + 远期压缩。

### 4.2 复杂度：Hybrid 的 $O(N)$ 主体

设共 $L$ 层，attention 占比 $\rho$（如 1/8）。每层 token $N$：
- SSM 层：$O(N d)$
- attention 层：$O(N^2 d)$

总：$\approx L[(1-\rho)Nd + \rho N^2 d]$。$\rho$ 小（如 1/8）时主体 $O(Nd)$（线性），attention 层贡献 $\rho N^2 d$ 但 $\rho$ 小、且 attention 层可加 sliding window 进一步降。

### 4.3 召回的恢复

纯 SSM 的召回误差 $\propto$ 状态压缩损失。加 1 层 attention 提供有限 eidetic window $W$，在该 window 内全保留，召回误差在 $W$ 内降为 0、超出仍衰减。N:1 间隔使每 $N$ 层有一次 eidetic 刷新，召回沿序列周期性恢复。这是 Hybrid 召回优于纯 Mamba 的数学。

### 4.4 显存：Hybrid 的混合

- SSM 层：固定 state $O(1)$。
- attention 层：KV cache $O(N)$，但只 $\rho$ 层有，总 $\rho L N$（比纯 Transformer $L N$ 少 $\rho$ 倍）。

故 Hybrid 推理显存 $\approx \rho \cdot (\text{Transformer KV}) + L\cdot(\text{SSM state})$，远低于纯 Transformer。


## 5. 代码示例（可选）

### 5.1 Hybrid 层交替（示意）

```python
import torch, torch.nn as nn

class HybridBlock(nn.Module):
    def __init__(self, d, use_attn):
        super().__init__()
        self.use_attn = use_attn
        if use_attn:
            self.attn = AttentionLayer(d)        # attention 层 (带 RoPE + KV cache)
        else:
            self.ssm = MambaLayer(d)             # SSM 层 (固定 state, 无 KV)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, kv_cache=None):
        h = self.norm(x)
        if self.use_attn:
            y = self.attn(h, kv_cache=kv_cache)  # attention 用 RoPE
        else:
            y = self.ssm(h)                        # SSM 用 selective scan
        return x + y

class HybridStack(nn.Module):
    def __init__(self, d, n_layers, attn_every=8):
        super().__init__()
        self.layers = nn.ModuleList([
            HybridBlock(d, use_attn=(i % attn_every == attn_every-1))
            for i in range(n_layers)
        ])                                          # N SSM + 1 ATTN 间隔
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x
```

### 5.2 复杂度核算

```python
N, d, L, rho = 100000, 4096, 80, 1/8
flops_pure_attn = L * 2 * N * N * d
flops_hybrid   = L * ((1-rho)*2*N*16*d + rho*2*N*N*d)   # SSM state=16
print(f"pure Transformer: {flops_pure_attn:.2e}")
print(f"hybrid (1/8 attn): {flops_hybrid:.2e}, ratio {flops_pure_attn/flops_hybrid:.1f}x cheaper")
# hybrid 远省
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Mamba]]（SSM 层）、[[自注意力]]（attention 层）、[[MoE]]（增容）、[[位置编码]]/[[RoPE]]（attention 层）、B'MOJO（理论框架）。
- **下游（应用）**: [[Transformer与新型架构适配]]（系统适配）、Jamba/Jamba-1.5/Falcon-Mamba/Zamba 部署、长上下文 LLM、推理省显存。
- **对比 / 易混**:
  - **Hybrid vs 纯 [[Mamba]]**：前者加 attention 补召回，后者纯 SSM 召回弱。见 3.5 表。
  - **Hybrid vs 纯 Transformer**：前者 SSM 层省 + 长；后者全 attention 贵 + KV 膨胀。
  - **Hybrid (block-level) vs Hymba (head-level) vs SISA (score-level)**：混合粒度不同——块级层交替 / attention head 内嵌 SSM / attention score 加 SSM 项。
  - **Jamba (SSM+attn+[[MoE]]) vs Zamba (SSM+attn)**：前者加 MoE 增容。


## 7. 常见误区与易错点

> [!warning] 误区 1：Hybrid 总是优于纯 Transformer/Mamba
> 不总。小模型/短序列纯 Transformer 常更简单更优。Hybrid 的优势在**长上下文 + 召回 + 效率**三角，需长序列才显现。短序列 Hybrid 的调度/算子复杂度反而是负担。

> [!warning] 误区 2：Hybrid 的 attention 层仍需全序列 KV
> 可优化。attention 层可加 sliding window（只近期 KV）、或稀疏 attention，进一步降 $O(N^2)$。Jamba 等用 windowed attention。Hybrid 不必全 attention。

> [!warning] 误区 3：SSM 层也要 [[位置编码]]
> 不需。SSM 递归天然有序（$h_t$ 依 $h_{t-1}$）。只有 attention 层需 [[RoPE]]。混合模型只 attention 层加位置编码。

> [!warning] 误区 4：Hybrid 推理显存和纯 Transformer 一样
> 不同。SSM 层固定 state $O(1)$，只有 $\rho$ attention 层有 KV。Hybrid 推理显存 $\approx \rho \cdot (\text{Transformer KV})$，远低。见 4.4。

> [!warning] 误区 5：Jamba = SSM+attention
> 是 SSM+attention **+[[MoE]]** 三合一。MoE 增容量控 active 算力，是 Jamba 能单 GPU 放 256K 的关键。别漏 MoE。

> [!tip] 实践：从 Jamba 起步
> 落地 Hybrid 长上下文，优先 Jamba/Jamba-1.5（开源、256K、有 [[MoE]]、单 80GB GPU 可放）。需召回任务确保 attention 层占比够（1/8 是经验下限）。系统适配见 [[Transformer与新型架构适配]]。


## 8. 延伸细节

### 8.1 关键论文（已核实 arXiv ID）

- **Jamba**: Lieber et al. (AI21). *Jamba: A Hybrid Transformer-Mamba Language Model*. **arXiv:2403.19887**, 2024-03。交错 Transformer-Mamba + MoE，256K，单 80GB GPU。
- **Jamba-1.5**: Jamba Team. *Jamba-1.5: Hybrid Transformer-Mamba Models at Scale*. **arXiv:2408.12570**。Large 94B / Mini 12B active，ExpertsInt8，256K。
- **B'MOJO**: Zancato et al. *Hybrid State Space Realizations... with Eidetic and Fading Memory*. **arXiv:2407.06324**。统一 Transformer/Mamba/Jamba 为 eidetic+fading 调制。

### 8.2 Hymba / SISA 等其他粒度

- **Hymba**（NVIDIA）：head-level 混合，attention head 内嵌 SSM 重要性信号，非块级层交替。
- **SISA**（arXiv:2606.02332，编号待核实）：score-level，attention score 内加 SSM-derived 重要性项，单次 SDPA 调用，无需 custom kernel。
- **Zamba**：SSM+attention 混合（无 MoE）。

粒度越细融合越紧但工程越复杂。块级（Jamba）最易落地。

### 8.3 Falcon-Mamba

纯 Mamba 大模型（Falcon 系列），不含 attention，是纯 SSM 路线对照。与 Jamba 形成"纯 vs Hybrid"对比。

### 8.4 内容来源

整理自 Jamba (arXiv:2403.19887)、Jamba-1.5 (arXiv:2408.12570)、B'MOJO (arXiv:2407.06324) 论文。eidetic/fading 框架据 B'MOJO。与纯 Mamba/Transformer 对比据 [[Mamba]]、[[自注意力]]。SISA 编号 2606.02332 待核实。

---
相关: [[新型架构适配]] | [[Mamba]] | [[自注意力]] | [[Transformer与新型架构适配]] | [[MoE]] | [[位置编码]] | [[RoPE]] | [[KV cache]] | [[FlashAttention]] | [[autoregressive decoding]]
