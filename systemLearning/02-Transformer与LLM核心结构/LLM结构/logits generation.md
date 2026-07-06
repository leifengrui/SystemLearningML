# logits generation

> **所属章节**: [[LLM结构]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中级

## 1. 一句话定义

**logits 生成（logits generation，lm_head，输出投影）** 是 [[decoder-only architecture]] 的"出口"：把主干最后一层输出的 $d$ 维隐藏表示 $h_L$，经一个线性层 **lm_head**（$W\in\mathbb{R}^{V\times d}$，投影 $d\to V$）映射成**词表大小 $V$ 的 logits 向量** $z\in\mathbb{R}^V$，再 $\text{softmax}$ 得下一词概率分布；训练时与真实下一词算交叉熵，推理时取 argmax 或采样。它是"语义向量→词表概率"的转换层。
> [!note] 解答：logits 是什么？
> 
> **定义**：logits 是 softmax **之前**的未归一化得分向量 $z\in\mathbb{R}^V$，每个分量 $z_v$ 对应词表里一个词 $v$ 的"该词作为下一词的原始得分"。它可以是任意实数（正/负/零），数值越大表示模型越倾向于选这个词，但**不是概率**（不归一、可负、和不为 1）。
> 
> **词源**：logit 来自 logistic regression，原义是 **log-odds（对数几率）** $\text{logit}(p)=\log\frac{p}{1-p}$，是 sigmoid 的反函数。在 softmax 多分类里，logits 是 softmax 的**输入**，softmax 的反函数 log-softmax 才回到"对数概率"。所以 LLM 语境里的"logits"是借词，泛指"softmax 前那层得分"，不再严格是 log-odds。
> 
> **怎么得到**：主干最后一层输出 hidden $h_L\in\mathbb{R}^d$，经 lm_head 线性投影 $z=h_L W^T$ 得到（见 [[#3.1 lm_head 的形式]]）。
> 
> **几何意义**：$z_v = h_L\cdot W_v = \|h_L\|\|W_v\|\cos\theta$，即 hidden 与词 $v$ 的"目标方向"的余弦相似度（乘模长）。越对齐，logit 越大。
> 
> **为什么不直接用概率 / 为什么需要 logits 这一步**：
> - 神经网络最后一层线性变换输出的是**无界实数**，自然就是 logits，softmax 才把它压成概率。
> - argmax 在 logits 和概率下**等价**（softmax 单调），所以 greedy 推理可直接取 `argmax(logits)` 省一次 softmax。
> - 但涉及**温度/采样/top-k/top-p** 时必须先 softmax 转概率（见 [[#8.5 logits 与采样]]、[[采样策略]]）。
> - 训练算交叉熵用 **log-softmax** 数值稳定（防 $e^z$ 溢出，见 [[#7. 常见误区与易错点]] 第4条）。
> 
> **辨析**：
> 
> | 概念 | 维度 | 含义 | 范围 |
> |---|---|---|---|
> | hidden $h_L$ | $d$（如 4096） | 语义表示 | 连续实数 |
> | logits $z$ | $V$（如 32000） | 词表上每个词的得分 | 任意实数 |
> | probability $p$ | $V$ | 词表上的概率分布 | $[0,1]$，和为 1 |
> 
> **一句话**：logits = "模型对每个候选词打的分"，softmax 前的原始分；hidden 是语义、logits 是得分、probability 是归一概率，三者层层递进。
## 2. 为什么需要它（动机与背景）

Transformer 主干输出的是 $d$ 维语义向量 $h_L\in\mathbb{R}^d$，但生成任务要回答"下一个词是词表中哪一个"。两者维度不匹配：

- $h_L$ 是连续语义表示，维度 $d$（如 4096）。
- 词表有 $V$ 个候选（如 32000/128000），要选一个。

需要一个把 $d$ 维"语义"映射到 $V$ 维"词表得分"的层 → **lm_head**：一个线性投影 $z=h_L W^T$，每个维度对应一个词的未归一化得分（logit）。再 softmax 归一成概率，训练用 CE、推理用 argmax/采样。

lm_head 与输入 embedding 常共享权重（**tied**），因为"词→语义"和"语义→词"是对偶的，共用投影省参数且质量不降。

## 3. 核心概念详解

### 3.1 lm_head 的形式

> [!note] 解答：lm_head 是什么？
> 
> **定义**：lm_head（language modeling head，语言建模头）是 LLM **最后一层的线性层** `nn.Linear(d, V, bias=False)`，把主干输出的 $d$ 维 hidden $h$ 投影成 $V$ 维 logits $z=hW^T$，即"语义向量 → 词表得分"的转换器。位置：final RMSNorm 之后、softmax/argmax 之前。
> 
> **命名**：lm = language modeling（语言建模），head = 任务头。源自多任务框架里"主干 + 任务头"的设计——预训练任务是"语言建模/预测下一词"，所以这个头叫 lm_head。微调到分类任务时可换一个 `cls_head`，但 LLM 通常保留 lm_head 对齐。
> 
> **形式**：
> 
> $$z_t = \text{lm\_head}(h_t) = h_t W^T,\quad W\in\mathbb{R}^{V\times d}$$
> 
> - $W$ 的每一行 $W_v$ 是"词 $v$ 的目标语义方向"，$h_t$ 与之点积 = 该词的 logit。
> - 通常**无 bias**（Llama 系无 bias）。
> 
> **与 embedding 的对偶 / tied**：
> - 输入 embedding $E$：词 id → 语义向量（$V\to d$）。
> - lm_head $W$：语义向量 → 词表得分（$d\to V$）。
> - 二者形状相同，常**共享权重**（tied，$W=E$），省 $Vd$ 参数。直觉：词的"输入语义"与"输出目标"应一致。大模型（70B+）有时 untied 换稳定性（见 [[#3.4 tied embedding]]）。
> 
> **类比**：embedding 是"词 → 语义字典"的正查，lm_head 是"语义 → 词"的反查——给定一个语义向量，问它最像词表里哪个词，点积最大的胜出。
> 
> **作用**：
> - **训练**：logits 与真实下一词算交叉熵（[[#3.3 训练：交叉熵 loss]]）。
> - **推理**：取最后位置 logits 的 argmax（greedy）或按概率采样（[[采样策略]]）。
> 
> **参数量**：$Vd$（untied 时，Llama-7B 约 1.3 亿占 ~2%；tied 则 0 额外）。大词表（128k+）时是显存/算力热点，需 parallel CE 或词表并行优化（[[#3.5 大词表的显存与算力]]）。
> 
> **代码**：
> 
> ```python
> import torch.nn as nn
> lm_head = nn.Linear(d, V, bias=False)       # 最简形式
> logits = lm_head(h)                          # h: (B, n, d) → logits: (B, n, V)
> # tied: lm_head.weight = embedding.weight
> ```

$$
z_t = \text{lm\_head}(h_t)=h_t W^T + b,\quad W\in\mathbb{R}^{V\times d},\ z_t\in\mathbb{R}^V
$$

- $W$ 的每一行 $W_v$ 是"词 $v$ 的目标语义方向"，$h_t$ 与之点积 = 该词的 logit。
- 直觉：$h_t$ 越接近词 $v$ 的语义方向，logit 越大 → 越可能预测 $v$。
- 常无 bias（Llama 系无 bias）。

### 3.2 logits → 概率 → 预测

$$
p = \text{softmax}(z),\quad \hat v = \arg\max_v z_v\ \text{（greedy）}
$$

- softmax 把任意 logits 归一为概率分布（和为 1）。
- greedy 取 argmax；采样按概率抽（见 [[采样策略]]）。

### 3.3 训练：交叉熵 loss

每个位置 $t$ 的 logits 与真实下一词 $y_t$（token id）算 CE：

$$
\mathcal{L}_t = -\log p_\theta(y_t\mid x_{<t})=-\log\frac{\exp(z_{t,y_t})}{\sum_v\exp(z_{t,v})}
$$

数值上用 **log-softmax** 稳定计算（防 exp 溢出），见 [[损失函数分类]]。

### 3.4 tied embedding

输入 embedding $E\in\mathbb{R}^{V\times d}$（词→语义）与 lm_head $W$（语义→词）形状相同，常共享：

$$
W = E\quad(\text{即 } z_t = h_t E^T)
$$

- 省 $Vd$ 参数（$V=32000,d=4096$ → 1.3 亿）。
- 直觉：词的"输入语义"与"输出目标"应一致，共享合理。
- 大模型（70B+）有时 untied 换稳定性。

### 3.5 大词表的显存与算力

$V$ 大（128k+）时 lm_head 是显存/算力热点：

- logits 张量 $(B,n,V)$ 巨大，CE 要对 $V$ 做 softmax。
- 优化：**parallel cross-entropy**（分片计算，不在物化全 $V$ logits 上做 softmax）、词表并行（tensor parallel 切分 $V$）。

## 4. 数学原理 / 公式

### logit 的几何意义

$$
z_{t,v} = h_t\cdot W_v^T = \|h_t\|\|W_v\|\cos\theta_{t,v}
$$

logit = $h_t$ 与词 $v$ 方向的余弦相似度（乘模长）。$h_t$ 越对齐词 $v$ 的方向，logit 越大 → 概率越高。

### softmax 的归一

$$
p_v = \frac{e^{z_v}}{\sum_k e^{z_k}}=\frac{e^{z_v-z_{\max}}}{\sum_k e^{z_k-z_{\max}}}
$$

减 $z_{\max}$（logitamax trick）防 $e^{z}$ 溢出，结果不变。

### tied 的等价

tied 下 $z_t=h_tE^T$，即"用输入 embedding 的转置当输出投影"。等价于：把 $h_t$ 与每个词的 embedding 算点积，最相似的词胜出。

### 参数量占比

$$
P_{\text{lm\_head}} = Vd
$$

untied 时是 $Vd$（Llama-7B：$32000\times4096\approx1.3$亿，占 ~2%），tied 则 0 额外。大词表（128k）时占比上升，优化价值凸显。

## 5. 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LMHead(nn.Module):
    def __init__(self, d, V, tied=None):
        super().__init__()
        self.lm = nn.Linear(d, V, bias=False)
        if tied is not None:                # 与输入 embedding 共享权重
            self.lm.weight = tied.weight
    def forward(self, h):                   # h: (B, n, d)
        return self.lm(h)                   # logits: (B, n, V)

d, V = 512, 32000
emb = nn.Embedding(V, d)
head = LMHead(d, V, tied=emb)               # tied embedding

h = torch.randn(2, 16, d)                    # 主干输出
logits = head(h)                             # (2,16,32000)
print(logits.shape)

# 训练：CE loss（用 log_softmax 稳定）
target = torch.randint(0, V, (2, 16))        # 真实下一词 id
logp = F.log_softmax(logits, dim=-1)         # (B,n,V)
loss = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean()
print(loss.item())

# 推理：greedy 取 argmax
next_tok = logits[:, -1].argmax(-1)          # 只取最后位置的预测
print(next_tok)                              # (B,) 下一词 id

# 数值稳定的 softmax（logitamax trick 手写）
z = torch.randn(V) * 100                     # 极端 logits
p = (z - z.max()).softmax(0)                 # 减最大值防溢出
print(p.sum(), p.max())
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[decoder-only architecture]]（主干输出 $h_L$）、[[tokenization]]（词表 $V$ 的来源）、[[损失函数分类]]（CE loss）。
- **下游（应用）**: [[采样策略]]（greedy/top-k/top-p/temperature 作用在 logits/概率上）、[[autoregressive decoding]]（每步生成下一词）。
- **对比 / 易混**:
  - **logits vs 概率**：logits 是未归一得分，需 softmax 才是概率。
  - **lm_head vs embedding**：方向相反（语义→词 vs 词→语义），tied 时共享权重。
  - **logits vs hidden**：hidden 是 $d$ 维语义，logits 是 $V$ 维词表得分。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **把 logits 当概率**：logits 可任意大小，必须 softmax 才是概率。直接比 logits 与比概率在 argmax 下等价，但涉及温度/采样时必须先 softmax。
> 2. **忘了 tie embedding**：untied 多 $Vd$ 参数；若想省参应显式 `lm_head.weight = emb.weight`。
> 3. **推理每步重算全序列 logits**：只需取**最后位置**的 logits 预测下一词（前面位置的 logits 推理时无用，仅训练算 loss 用）。
> 4. **CE 用裸 softmax 再 log**：`log(softmax(z))` 数值不稳（exp 溢出）。用 `F.log_softmax` 或 `F.cross_entropy`（内部 log-softmax）。
> 5. **大词表物化全 logits 爆显存**：$(B,n,V)$ 在 $V=128k$ 时巨大，用 parallel CE 或 TP 切分。
> 6. **logits 维度错**：LLM 的 CE 要 reshape 成 `(B*n, V)` 与 `(B*n,)`，否则广播错（见 [[损失函数分类]] 误区6）。

## 8. 延伸细节

### 8.1 logit lens

研究技巧：把中间层的 hidden $h_l$ 直接喂给 lm_head 得"中间层预测的词"。可观察模型在每层逐步确定输出的过程，是可解释性工具。

### 8.2 parallel cross entropy

大词表下，先物化 `(B,n,V)` 再 softmax 爆显存。parallel CE 把 vocab 分片，每片算部分 softmax 累积，避免物化全 $V$，省显存。Megatron-LM 的 `VocabParallelEmbedding` 实现。

### 8.3 untied 的取舍

tied 省参数但约束输入/输出空间一致；untied 让输出投影自由，大模型有富余参数时 untied 略提质量。Llama-7B tied、70B untied 是典型取舍。

### 8.4 final norm 的作用

lm_head 前必有一个 final RMSNorm（把 $h_L$ 归一到稳定范围），否则深层 $h_L$ 数值漂移会使 logits 量级失控、softmax 饱和。是 Pre-LN 架构的标配收尾。

### 8.5 logits 与采样

logits 经温度 $T$ 缩放 $z/T$ 再 softmax 控制分布锐度；top-k/top-p 在 logits 上截断后重归一化。这些都在 logits→概率 之后，详见 [[采样策略]]。

---
相关: [[LLM结构]]、[[decoder-only architecture]]、[[tokenization]]、[[损失函数分类]]、[[采样策略]]、[[autoregressive decoding]]
