# causal mask

> **所属章节**: [[LLM结构]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中级

## 1. 一句话定义

**因果掩码（causal mask，causal mask，下三角掩码，self-attention mask）** 是一个下三角矩阵 $M$：在 [[自注意力]] 的 $\text{softmax}(QK^T/\sqrt d)$ 前，把**上三角（未来位置）**填 $-\infty$，使位置 $i$ 只能 attend 到 $j\le i$（过去与自身），遮住未来 token；它让 [[decoder-only architecture]] 在**训练时一次前向并行算所有位置**（teacher forcing）的同时保持自回归性质——每个位置看不到"未来的答案"，与逐词推理一致。

## 2. 为什么需要它（动机与背景）

LLM 的训练目标是"看前文 $x_{<t}$ 预测 $x_t$"。训练时若 attention 全连接（双向），则：

- **信息泄露**：预测 $x_t$ 时模型直接看到了 $x_t$ 本身（及其后），等于偷看答案，loss 虚低、学不到东西。
- **训练-推理不一致**：推理是逐词生成（每步只看已生成的前文），训练若双向则会训练出"依赖未来"的模式，推理时未来不存在 → 崩。

**causal mask** 用一个 $-\infty$ 矩阵把未来注意力清零，既保证训练时各位置"只看过去"，又允许一次前向算完所有位置（并行高效），与逐词推理严格一致。这是 decoder-only 能高效训练的基石。

## 3. 核心概念详解

### 3.1 mask 矩阵

对序列长度 $n$，causal mask $M\in\mathbb{R}^{n\times n}$：

$$
M_{ij}=\begin{cases}0 & j\le i\text{（过去与自身，可见）}\\ -\infty & j>i\text{（未来，遮蔽）}\end{cases}
$$

即下三角为 0、上三角为 $-\infty$ 的矩阵。加到 attention scores 上：

$$
\text{scores}' = QK^T/\sqrt d + M
$$

softmax 后，未来列因 $e^{-\infty}=0$ 权重为 0。

### 3.2 训练并行与推理自洽

- **训练**：输入完整序列 $[x_1,\dots,x_n]$，一次前向得到所有位置的 logits，与 $[x_2,\dots,x_{n+1}]$ 算 CE。mask 保证位置 $t$ 的预测只依赖 $x_{\le t}$（不含 $x_t$ 的"答案"）。**并行、高效**。
- **推理**：逐词生成。第 $t$ 步输入 $[x_1,\dots,x_t]$，预测 $x_{t+1}$；新词 append 后继续。每步 attention 的 mask 自动适应新长度。与训练时位置 $t$ 的视野完全一致。

> [!note] teacher forcing
> 训练时用**真实**前文（而非模型自己生成的）作为输入，叫 teacher forcing。causal mask + teacher forcing 让训练稳定且并行。推理时换成自生成词，但每步的 mask 逻辑不变。

### 3.3 与 padding mask 组合

实际 batch 内序列长短不一，需 padding 到等长。除 causal mask 外还有 **padding mask**：把 padding 位置在 attention 里也置 $-\infty$（不让有效 token attend 到 pad）。两者相加得最终 mask：

$$
M_{\text{total}} = M_{\text{causal}} + M_{\text{padding}}
$$

### 3.4 attention sink 现象

研究发现 causal attention 下，模型常把**大量注意力放在第一个 token**（甚至无关的 sink token）。原因：因果 mask 下，所有位置都能看到首 token，它成了"无处可去"的注意力垃圾桶。这与 [[残差连接]] 提到的 attention sink 相关，工程上保留首 token 可稳推理。

## 4. 数学原理 / 公式

### softmax 的遮蔽效果

$$
\alpha_{ij}=\text{softmax}_j\!\left(\frac{q_i k_j^T}{\sqrt d}+M_{ij}\right)=\frac{\exp(s_{ij}+M_{ij})}{\sum_j\exp(s_{ij}+M_{ij})}
$$

当 $j>i$，$M_{ij}=-\infty$ → $\exp(-\infty)=0$ → $\alpha_{ij}=0$。故 $\sum_{j>i}\alpha_{ij}=0$，位置 $i$ 的输出只由 $j\le i$ 的 value 加权构成。

### 信息流的 DAG

causal mask 使 token 间信息流是**有向无环图**（DAG）：信息只从过去流向未来，无反向。这是自回归生成的数学保证——预测 $x_t$ 用的所有信息都来自 $x_{<t}$，无未来泄漏。

### 严格下三角的线性代数视角

把 attention 权重矩阵 $\Alpha$ 视为算子，causal mask 使 $\Alpha$ 严格下三角（对角含自身）→ 输出 $Y=\Alpha V$ 的第 $i$ 行只含 $V_{\le i}$ → 因果性。

## 5. 代码示例

```python
import torch
import torch.nn.functional as F

n = 6
scores = torch.randn(1, 1, n, n) * 5      # 模拟 QK^T/√d

# 1. 构造 causal mask（下三角为 0，上三角为 -inf）
mask = torch.full((n, n), float('-inf')).triu(1)   # 上三角(不含对角)=-inf
print(mask)
# tensor([[0., -inf, -inf, -inf, -inf, -inf],
#         [0., 0., -inf, -inf, -inf, -inf],
#         ...                                    ]) 下三角0

# 2. 加到 scores 后 softmax
attn = F.softmax(scores + mask, dim=-1)
print((attn.triu(1) == 0).all())           # True：上三角权重全0

# 3. 与 attention 结合（PyTorch 风格）
def causal_self_attn(q, k, v):
    n = q.size(-2)
    mask = torch.full((n, n), float('-inf'), device=q.device).triu(1)
    s = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5) + mask
    return F.softmax(s, dim=-1) @ v

q = k = v = torch.randn(2, 8, n, 64)
out = causal_self_attn(q, k, v)
print(out.shape)                           # (2,8,n,64)

# 4. 常用 API：torch.tril 的布尔版（做 attention 时用 attn_mask
bool_mask = torch.ones(n, n, dtype=torch.bool).tril()
# nn.MultiheadAttention(..., attn_mask=~bool_mask)  # False 处被遮
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[自注意力]]（mask 作用在 attention scores）、[[Tensor]]（布尔/浮点 mask）。
- **下游（应用）**: [[decoder-only architecture]]（每层 causal MHA 必用）、[[autoregressive decoding]]（推理每步的视野）、[[KV cache]]（缓存的是"过去"K/V，与因果性一致）。
- **对比 / 易混**:
  - **causal mask vs padding mask**：前者遮未来（结构），后者遮 pad（长度），二者叠加。
  - **causal vs bidirectional**：decoder-only 用 causal，encoder-only（BERT）用双向无 mask。
  - **causal mask vs [[位置编码]]**：mask 限制"看多远未来"，PE 注入"在哪个位置"，职责不同。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **mask 填 0 而非 $-\infty$**：softmax(0) 仍非 0，未来位置会有非零权重 → 信息泄露。必须 $-\infty$（或大负数如 -1e9）。
> 2. **上下三角搞反**：要遮的是**未来（上三角）**，下三角（过去）保持可见。`triu(1)` 得上三角置 $-\infty$，别用 `tril`。
> 3. **训练推理 mask 不一致**：训练用完整序列 + causal mask；推理每步也要用 causal mask（虽只看过去，但 mask 保证格式一致）。
> 4. **以为 mask 让模型"知道顺序"**：mask 只限制信息流（不看未来），**位置顺序信息靠 [[位置编码]]**。无 PE 的 causal attention 仍是无序的（只限制可见性，不注入位置）。
> 5. **忽略对角线**：$j=i$（自身）应可见（对角为 0），`triu(1)` 排除对角正确；若误用 `triu(0)` 会把对角也遮了。
> 6. **padding mask 忘加**：batch 变长序列不处理 padding，会让 token attend 到 pad，引入噪声。

## 8. 延伸细节

### 8.1 prefix-LM / UL2 的 mask 变体

并非所有 LLM 都用严格 causal。prefix-LM（如 PaLM 部分配置）对前缀段双向、对生成段因果，混合 mask。UL2 用多种 mask 模式混合训练。主流 decoder-only 仍严格 causal。

### 8.2 attention sink 与流匹配

causal mask 下首 token 是全局可见的"汇聚点"，研究（StreamingLLM）发现保留首几个 sink token 即可让注意力稳定、支持超长流式生成，不需重算全部 KV。

### 8.3 sliding window mask

长序列下 causal 全注意力 $O(n^2)$ 太贵，可用**滑动窗口 mask**（每 token 只看最近 $w$ 个）替代全 causal，Mistral 等用之降算力，配合 global token 兼顾长程。

### 8.4 mask 与 KV cache 的协同

推理时 [[KV cache]] 存的是"已生成位置的 K/V"。新词来时只与 cache 中的过去 K 做 attention——这正是 causal mask 的运行时实现：自然只看过去，cache 复用而非重算。

### 8.5 训练效率

causal mask 让训练从"逐词串行 $O(n)$ 步"变成"一次前向 $O(1)$ 步"，是 LLM 大规模预训练可行的前提。代价是 $O(n^2)$ attention 算力（[[flash attention]] 优化）。

---
相关: [[LLM结构]]、[[自注意力]]、[[decoder-only architecture]]、[[autoregressive decoding]]、[[KV cache]]、[[位置编码]]、[[flash attention]]
