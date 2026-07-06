# decoder-only architecture

> **所属章节**: [[LLM结构]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中级

## 1. 一句话定义

**decoder-only 架构（decoder-only architecture，纯解码器架构）** 是现代 LLM（GPT/Llama/Qwen 系）的统一骨架：把 $N$ 个 Transformer block 串叠，每个 block 内部是 **[[causal mask]] 的 [[多头注意力]] + [[FFN]]**（配 [[残差连接]] + Pre-[[layer norm]]），输入是 token 嵌入加 [[位置编码]]、输出经 final norm + [[logits generation]] 投到词表；它用"看前文预测下一词"的自回归目标统一了理解与生成，是当前主流 LLM 的事实架构。

## 2. 为什么需要它（动机与背景）

Transformer 原版是 **encoder-decoder**（为机器翻译设计：encoder 读源句、decoder 生成目标句，有 cross-attention）。但纯语言建模任务（预测下一词）没有"源句"：

- 只需"给前文，预测下一词" → decoder 的**因果 mask** 已足够表达"只看过去"。
- encoder-decoder 的 cross-attention 在纯生成里冗余。
- 单一 decoder 堆叠：**结构统一、scaling 友好、参数效率高**，且因果目标让模型在预训练中自然学会理解 + 生成（in-context learning 涌现）。

GPT 系率先走纯 decoder 路线，scaling law 实证 decoder-only 在同等参数下语言建模最优，遂成 LLM 标准。BERT 走 encoder-only（双向，理解强但不能自回归生成），适合理解任务但非通用 LLM 主流。

## 3. 核心概念详解

### 3.1 整体数据流

```
tokens → embedding(+PE) → N×[Block] → final LN → lm_head → logits
```

一个 Block 内部（Pre-LN 风格）：

```
h → LN → causal MHA ─┐ (+残差) → h'
h' → LN → FFN ───────┐ (+残差) → h''
```

- **输入端**：token id → embedding 查表 $E\in\mathbb{R}^{V\times d}$，加 [[位置编码]]（现代用 [[RoPE]]，作用在 Q/K 故此处不加绝对 PE）。
- **主干**：$N$ 个 Block 堆叠，每个 = causal MHA + FFN，配残差 + Pre-RMSNorm。
- **输出端**：final RMSNorm → lm_head（线性 $d\to V$）→ logits，详见 [[logits generation]]。

### 3.2 三种 Transformer 架构对比

| 架构 | 注意力方向 | 典型任务 | 代表 | 能否自回归生成 |
|---|---|---|---|---|
| **encoder-only** | 双向（无 mask） | 理解（分类/NER） | BERT | 否（需 MLM） |
| **encoder-decoder** | encoder 双向 + decoder 因果 + cross-attn | 序列到序列（翻译） | 原版 T5/BART | 是（decoder） |
| **decoder-only** | 因果（单向） | 语言建模→通用 | **GPT/Llama** | **是** |

> [!note] 三种架构的适用场景与优缺点（整合自扩展阅读）
> 
> 三种架构各有"擅长任务"，源于注意力方向与训练目标的差异。下表是对 3.2 的补充（侧重任务/训练目标/优劣）：
> 
> | 架构 | 训练目标 | 擅长任务 | 不擅长 | 代表实现 |
> |---|---|---|---|---|
> | **Decoder-Only** | next-token prediction（自回归） | 文本生成、对话、续写、通用 LLM | （理解任务也能做，靠 prompt 涌现） | GPT / Llama / Qwen |
> | **Encoder-Only** | Masked Language Model（MLM，遮盖预测） | 文本分类、情感分析、NER、句向量 | 不能自回归生成（无因果目标） | BERT / RoBERTa |
> | **Encoder-Decoder** | Seq2Seq（去噪/翻译） | 机器翻译、文本摘要、Seq2Seq 转换 | 通用 LLM（结构冗余、scaling 不如 decoder-only） | 原版 T5 / BART / Whisper |
> 
> **要点**：
> 
> - **Decoder-Only**：输入输出共享同一套嵌入，单向（因果）注意力只能看前文。优点是生成连贯、通用；早期局限被认为"理解弱"，但规模化后理解能力涌现，故成主流。
> - **Encoder-Only**：双向注意力，能同时关注前后文，上下文理解最充分；用 **MLM** 训练（随机遮盖约 15% 词再预测），泛化强。局限是无法直接自回归生成，故不适合对话/续写。
> - **Encoder-Decoder**：编码器双向编码输入为上下文表示，解码器单向生成但可 cross-attend 编码器输出。优点是 Seq2Seq 任务强；局限是参数/算力成本高、结构复杂、scaling law 下不如 decoder-only 高效。
> 
> **为何 LLM 选 Decoder-Only**：见 [[#3.3 为何 decoder-only 成主流]]——统一（理解+生成）、scaling 友好、in-context learning 涌现、无 cross-attn 参数更集中、训练并行（causal mask 一次算所有位置 loss）。

### 3.3 为何 decoder-only 成主流

1. **统一**：一个架构同时做理解 + 生成（next-token prediction 是万能接口）。
2. **scaling 友好**：堆深、堆宽都直接生效，scaling law 验证最优。
3. **in-context learning**：因果目标 + 大规模 → few-shot/CoT 涌现。
4. **效率**：无 cross-attention，参数更集中用于 self-attention + FFN。
5. **训练并行**：[[causal mask]] 让一次前向算所有位置 loss，无需逐词。

### 3.4 参数组成

一个 $L$ 层、宽 $d$、词表 $V$ 的 decoder-only：

- **embedding**：$V\times d$（常与 lm_head **tied**，共用）。
- **每 block**：attention $\approx4d^2$（Q/K/V/O 四投影）、FFN $\approx8d^2$（4× 升降维，SwiGLU 略不同）、norm $\approx O(d)$。
- **lm_head**：$d\times V$（tied 则 0 额外）。
- 总参数 $\approx L\times12d^2 + Vd$，**FFN 占约 2/3**，attention 约 1/3。

### 3.5 Pre-LN + RMSNorm 现代配置

现代 LLM 几乎全用 **Pre-LN**（残差主干恒等、深网稳）+ **RMSNorm**（省均值/β）。原版 Post-LN 已弃用。每 block 两个 norm（attn 前、FFN 前）+ 末端 final norm。

## 4. 数学原理 / 公式

### 前向（单层 Pre-LN block）

设输入 $h_{l-1}\in\mathbb{R}^{n\times d}$：

$$
a_l = h_{l-1} + \text{causalMHA}\!\big(\text{RMSNorm}(h_{l-1})\big)
$$
$$
h_l = a_l + \text{FFN}\!\big(\text{RMSNorm}(a_l)\big)
$$

堆叠 $L$ 层后：

$$
\text{logits} = \text{lm\_head}\!\big(\text{RMSNorm}(h_L)\big)\in\mathbb{R}^{n\times V}
$$

### 参数量

$$
P \approx L(4d^2 + 8d^2) + 2Vd = 12Ld^2 + 2Vd
$$

tied embedding 时 $2Vd\to Vd$。例 Llama-7B（$L=32,d=4096,V=32000$，tied）：$12\times32\times4096^2\approx6.4B$ + $0.5B\approx6.9B$ ✓。

### 因果目标的统一性

训练 loss 是每个位置的 next-token CE：

$$
\mathcal{L}=-\frac1n\sum_{t=1}^{n}\log p_\theta(x_t\mid x_{<t})
$$

一次前向 + [[causal mask]] 同时算所有 $t$ 的 loss，并行高效。

## 5. 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """Pre-LN + causal MHA + SwiGLU FFN，现代 LLM 单层。"""
    def __init__(self, d, h, d_ff, n):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)  # 实际用 RMSNorm
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.register_buffer('mask', torch.tril(torch.ones(n, n)).bool())
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask, need_weights=False)
        x = x + a                                   # 残差1
        h = self.n2(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))  # 残差2 + SwiGLU
        return x

class DecoderOnly(nn.Module):
    def __init__(self, V, d, L, h, d_ff, n):
        super().__init__()
        self.tok_emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, h, d_ff, n) for _ in range(L)])
        self.final_n = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # tied embedding
    def forward(self, ids):                          # ids: (B, n)
        x = self.tok_emb(ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_n(x)
        return self.lm_head(x)                       # logits: (B, n, V)

m = DecoderOnly(V=32000, d=512, L=6, h=8, d_ff=1376, n=128)
ids = torch.randint(0, 32000, (2, 128))
print(m(ids).shape)                                  # (2,128,32000)
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[自注意力]]、[[多头注意力]]（causal MHA）、[[FFN]]、[[残差连接]]、[[layer norm]]、[[位置编码]]/[[RoPE]]、[[causal mask]]、[[tokenization]]（输入）。
- **下游（应用）**: [[logits generation]]（输出端）、[[KV cache]]/[[autoregressive decoding]]（推理）、[[采样策略]]、预训练 next-token loss、MoE（替 FFN）。
- **对比 / 易混**:
  - **vs encoder-only**：见 3.2，双向 vs 单向，理解 vs 生成。
  - **vs encoder-decoder**：见 3.2，无 cross-attn，统一生成。
  - **decoder-only vs decoder block**：后者是组件，前者是整体架构。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 decoder-only 只能生成不能理解**：它也能做理解任务（分类/抽取），靠 next-token 目标 + prompt 涌现理解能力，BERT 式双向不是理解唯一路径。
> 2. **以为有 encoder**：decoder-only 无 encoder、无 cross-attention，只有 causal self-attention。
> 3. **以为 lm_head 一定 tied**：tied 是省参数的常见做法（Llama-7B tied，70B 不 tied），非强制。
> 4. **Pre/Post-LN 混用**：现代 LLM 必须 Pre-LN + RMSNorm，照抄原版 Post-LN 深网不稳。
> 5. **以为位置编码加在输入**：现代用 [[RoPE]] 作用 Q/K，输入端不加绝对 PE（[[sinusoidal PE]] 才加输入）。
> 6. **参数量算漏 FFN**：FFN 占 2/3，只算 attention 会严重低估。
> 7. **训练时不用 causal mask**：不加 mask 会信息泄露（看到未来词），训练-推理不一致。

## 8. 延伸细节

### 8.1 embedding tying

输入 embedding 与 lm_head 共享权重（$W_{\text{emb}}=W_{\text{lm\_head}}^T$），省 $Vd$ 参数且实证质量不降。大模型（如 70B）因容量充足常 untied 换稳定性。

### 8.2 GPT → Llama 演化

GPT-1/2：Post-LN + GELU + learned/sinusoidal PE。Llama：Pre-RMSNorm + SwiGLU + [[RoPE]] + GQA。核心组件全部升级，但"decoder-only 堆叠"骨架未变。

### 8.3 MoE 替 FFN

把每层 FFN 换成多个专家 FFN + 路由，参数总量大但每 token 激活稀疏（Mixtral/DeepSeek-MoE），是扩参数而不等比涨算力的方向，骨架仍是 decoder-only。

### 8.4 深度 vs 宽度

scaling law 指引下，给定算力偏好"深度与宽度协调增长"，过深过窄或过浅过宽都不优。decoder-only 的简洁使其易 scale 到百 B/千亿。

### 8.5 因果目标的统一性

next-token prediction 把理解、生成、推理都压缩进"预测下一词"，是 decoder-only 通用的根基。in-context learning、CoT 都源自该目标的涌现。

---
相关: [[LLM结构]]、[[causal mask]]、[[logits generation]]、[[多头注意力]]、[[FFN]]、[[残差连接]]、[[layer norm]]、[[位置编码]]、[[KV cache]]

> [!note] 解答：扩展阅读已整合到正文
> 原"扩展阅读"（三种架构定义/工作原理/优缺点）与正文 [[#3.2 三种 Transformer 架构对比]] 高度重复，已提炼整合为 3.2 表格后的 callout（补 BERT 的 MLM 训练目标、Seq2Seq 适用场景、各架构优缺点对比），原文删除以免冗余。