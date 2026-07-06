# exposure bias

> **所属章节**: [[LLM特有RL问题]]
> **所属模块**: [[05-LLM-RL对齐]]
> **难度**: 易（概念清晰，是 LLM 训练经典问题）

## 1. 一句话定义

**exposure bias（暴露偏差 / 训练-推布分布偏移）** 是 LLM 训练的经典问题：**训练用 teacher forcing**（每步输入 ground-truth 前缀），**推布用自回归**（每步输入自己生成的前缀），导致训练与推布的输入分布不一致——模型从未"见过"自己的错误前缀，一旦推布时某步出错，后续在"错误上下文"上 snowball（滚雪球）放大错误。又称 **train-test mismatch**、scheduled sampling 问题。在 RLHF 语境中：[[SFT]] 阶段有 exposure bias（teacher forcing）；[[PPO optimization]] 的 rollout 用 actor **自生成**，某种程度上**部分缓解** exposure bias（让模型见过自己的前缀），但 RLHF 也有新变体（rollout prompt 分布与最终推布 prompt 分布偏移）。防法：scheduled sampling、RL 训练（自生成）、data augmentation、RLAIF/在线学习。

> [!note] 别名
> exposure bias = train-test mismatch = exposure bias problem = scheduled sampling problem。都指 teacher forcing 与自回归推布的分布偏移。

## 2. 为什么需要它（动机与背景）

LLM 训练的标准范式是 **teacher forcing**：
- 训练：$\mathcal{L}=-\sum_t\log\pi_\theta(y_t|y_{<t}^{\text{gt}})$，每步输入**真实前缀** $y_{<t}^{\text{gt}}$；
- 推布：$\hat y_t=\arg\max\pi_\theta(\cdot|\hat y_{<t})$，每步输入**自己生成的前缀** $\hat y_{<t}$。

问题：
1. **分布偏移**：训练时模型只见 ground-truth 前缀（干净），推布时见自己的前缀（可能有错）；
2. **从未见过错误**：模型没学过"在错误前缀上如何继续"，一旦推布某步错，后续在错误上下文上 snowball；
3. **错误累积**：自回归推布是马尔可夫链，一步错步步错（[[MDP]] §8.1）；
4. **长序列尤甚**：序列越长，前缀错误的累积概率越大，exposure bias 越显。

这是 LLM 训练的固有矛盾（teacher forcing 简单高效但引入偏移），是序列生成模型的通病（机器翻译/对话/摘要都有）。在 RLHF 中，SFT 阶段的 exposure bias 会被 PPO 的 rollout 部分缓解（actor 自生成 = 见过自己前缀），但需理解机制与残留问题。

## 3. 核心概念详解

### 3.1 teacher forcing vs 自回归（核心矛盾）

```
训练(teacher forcing):
  输入: y_gt = [a, b, c, d]
  step t: 输入 y_{<t}^gt(干净前缀), 预测 y_t^gt
  → 模型只见干净前缀

推布(自回归):
  step 1: 输入 [], 生成 ŷ_1
  step 2: 输入 [ŷ_1], 生成 ŷ_2        ← ŷ_1 可能有错!
  step 3: 输入 [ŷ_1, ŷ_2], 生成 ŷ_3   ← 错误累积
  → 模型见自己的(可能错误)前缀
```

### 3.2 snowball 效应（错误滚雪球）

- 推布某步 $\hat y_t$ 偏离 ground-truth；
- 后续在 $\hat y_{<t+1}$（含错）上预测，模型未训过此上下文；
- 错误被放大，序列越生成越离谱；
- 长序列尤甚（累积概率高）。

### 3.3 exposure bias 在 RLHF 各阶段

| 阶段 | exposure bias 情况 |
|---|---|
| **预训练** | 有（teacher forcing next-token） |
| **[[SFT]]** | 有（teacher forcing 示范） |
| **[[PPO optimization]] rollout** | **部分缓解**（actor 自生成，见过自己前缀） |
| **[[DPO]]** | 有（teacher forcing 算 logp，无自生成） |
| **最终推布** | 有（自回归） |

> [!tip] RLHF 的双刃
> PPO 的 rollout 用 actor 自生成，让模型在训练中"见过"自己的前缀，**部分缓解** exposure bias（这是 RL 训练相比 SFT 的优势）。但 PPO 的 rollout prompt 分布若与最终推布 prompt 分布偏移，又有新变体 exposure bias。

### 3.4 RLHF 的新变体：prompt 分布偏移

- PPO rollout 用某 prompt 集 $\mathcal{D}_{\text{rollout}}$；
- 最终推布用真实用户 prompt $\mathcal{D}_{\text{serve}}$；
- 若 $\mathcal{D}_{\text{rollout}}\ne\mathcal{D}_{\text{serve}}$（分布偏移），PPO 优化的 $\pi_\theta$ 在 $\mathcal{D}_{\text{serve}}$ 上可能不优；
- 这是 exposure bias 的 prompt 层变体（非前缀层）。

### 3.5 防法总览

| 防法 | 机制 | 适用 |
|---|---|---|
| **scheduled sampling** | 训练时混合 ground-truth 与模型生成的前缀 | SFT |
| **RL 训练（PPO）** | rollout 自生成，见过自己前缀 | RLHF PPO |
| **data augmentation** | 训练数据加噪/扰动前缀 | SFT/预训练 |
| **reinforcement learning** | 用 reward 优化自生成序列 | RLHF |
| **online learning** | 推布数据回训，对齐 $\mathcal{D}_{\text{serve}}$ | 持续对齐 |
| **Iterative DPO** | 多轮 DPO 用上轮 $\pi_\theta$ 生成数据 | DPO 变体 |

## 4. 数学原理 / 公式

### 4.1 teacher forcing 的损失

$$
\mathcal{L}_{\text{TF}}(\theta)=-\sum_{t=1}^T\log\pi_\theta(y_t^{\text{gt}}|y_{<t}^{\text{gt}},x)
$$

每步输入 ground-truth 前缀 $y_{<t}^{\text{gt}}$，与推布的 $\hat y_{<t}$ 分布不同。

### 4.2 自回归推布的分布

推布时 $\hat y_t\sim\pi_\theta(\cdot|\hat y_{<t},x)$，$\hat y_{<t}$ 是模型自己生成的随机变量。推布分布：
$$
P_{\text{serve}}(\hat y|x)=\prod_t\pi_\theta(\hat y_t|\hat y_{<t},x)
$$

与训练的 $P_{\text{train}}(y|x)=\prod_t\pi_\theta(y_t|y_{<t}^{\text{gt}},x)$ 在前缀分布上不一致（exposure bias）。

### 4.3 scheduled sampling

训练时以概率 $\epsilon$ 用模型生成的前缀替代 ground-truth：
$$
\tilde y_t=\begin{cases}y_t^{\text{gt}} & \text{prob }1-\epsilon\\\hat y_t\sim\pi_\theta(\cdot|\tilde y_{<t},x) & \text{prob }\epsilon\end{cases}
$$

$\epsilon$ 从 0（纯 teacher forcing）逐步升到 1（纯自生成），让模型渐进见过自己前缀。这是 Bengio et al. 2015 的经典防法。

### 4.4 RL 训练的缓解

PPO rollout 用 $\hat y\sim\pi_\theta(\cdot|x)$（自生成），reward $r_\phi(x,\hat y)$ 评估整条自生成序列。优化 $\max\mathbb{E}_{\hat y\sim\pi_\theta}[r_\phi]$ 时，模型在"自己前缀"的分布上训练，**对齐训练与推布的前缀分布**，缓解 exposure bias。这是 RLHF 相比 SFT 的理论优势之一。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F

class LM(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)

V, D = 50, 32
model = LM(V, D)
opt = torch.optim.Adam(model.parameters(), 1e-3)
x = torch.randint(0, V, (4, 8))   # ground-truth 序列

# (1) teacher forcing(标准 SFT)
def teacher_forcing_loss(model, x):
    ctx = x[:, :-1]; tgt = x[:, 1:]
    logits = model(ctx)
    return F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))

# (2) scheduled sampling(混合 gt 与自生成前缀)
def scheduled_sampling_loss(model, x, epsilon=0.3):
    B, T = x.shape
    ctx = x[:, :1]   # 起始 token
    total = 0
    for t in range(1, T):
        logits = model(ctx)
        tgt = x[:, t]
        total += F.cross_entropy(logits[:, -1], tgt, reduction='mean')
        # 以概率 ε 用模型生成替代 gt 前缀
        if torch.rand(1).item() < epsilon:
            nxt = logits[:, -1].argmax(-1)   # 自生成
        else:
            nxt = tgt                         # gt
        ctx = torch.cat([ctx, nxt.unsqueeze(1)], 1)
    return total / T

# (3) 自回归推布(演示 exposure bias:snowball)
@torch.no_grad()
def generate(model, x, max_len=10):
    out = x.clone()
    for _ in range(max_len):
        nxt = model(out)[:, -1].argmax(-1, keepdim=True)  # 贪心
        out = torch.cat([out, nxt], 1)
    return out

# 训练(teacher forcing)
for ep in range(10):
    loss = teacher_forcing_loss(model, x)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"teacher forcing 训练 loss={loss:.3f}")

# 推布:看是否 snowball(与 gt 偏离)
gen = generate(model, x[:, :2], max_len=6)
print(f"gt:    {x[0].tolist()}")
print(f"推布:  {gen[0].tolist()}")
print("差异 = exposure bias(snowball):模型未见自己前缀,推布可能偏离")

# scheduled sampling 防法
for ep in range(5):
    loss = scheduled_sampling_loss(model, x, epsilon=0.3)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"scheduled sampling 训练 loss={loss:.3f}(缓解 exposure bias)")
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[SFT]]（exposure bias 主场景）、[[PPO optimization]]（rollout 部分缓解）、[[MDP]]（自回归 = 马尔可夫链，错误累积）、teacher forcing（训练范式）。
- **下游（应用）**: LLM 训练策略、scheduled sampling、在线学习、Iterative DPO。
- **对比 / 易混**:
  - **exposure bias vs [[mode collapse]]**：前者是训练-推布分布偏移（前缀层），后者是输出单一（策略层）。不同问题，可并发。
  - **exposure bias vs [[policy collapse]]/[[KL explosion]]/[[reward hacking]]**：前三者是 RLHF PPO 的失败模式，exposure bias 是更基础的训练-推布偏移，贯穿 SFT/RLHF/推布。
  - **teacher forcing vs scheduled sampling vs RL**：teacher forcing 引入偏移，scheduled sampling 渐进缓解，RL（PPO）自生成训练彻底对齐。
  - **exposure bias vs covariate shift**：前者是序列生成特例（前缀分布偏移），后者是通用输入分布偏移。exposure bias 是 covariate shift 的序列版。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"exposure bias 是 RLHF 独有"**：错。是 teacher forcing 的通病，预训练/SFT/机器翻译都有。RLHF 的 rollout 反而部分缓解。
> 2. **"PPO 不缓解 exposure bias"**：错。PPO rollout 用 actor 自生成，让模型见过自己前缀，是缓解。
> 3. **"scheduled sampling 完全解决"**：部分。缓解但非根治，且训练复杂度高。RL 训练更彻底。
> 4. **"exposure bias = mode collapse"**：错。前者是分布偏移，后者是输出单一。不同。
> 5. **"推布出错就是 exposure bias"**：不一定。可能是模型能力不足/数据问题。exposure bias 特指训练-推布前缀分布偏移。
> 6. **"teacher forcing 应弃"**：错。teacher forcing 简单高效，是预训练/SFT 主流。exposure bias 是其代价，需防法补。
> 7. **"DPO 不受 exposure bias 影响"**：部分。DPO 用 teacher forcing 算 logp（无自生成），仍有 exposure bias；但 DPO 在偏好数据上前缀是固定的，偏移形式略不同。
> 8. **"长序列无影响"**：错。序列越长，exposure bias 越显（错误累积概率高）。

## 8. 延伸细节

### 8.1 exposure bias 的历史

Bengio et al. 2015 在 RNN seq2seq 时代提出 scheduled sampling 解决 exposure bias，是序列生成的经典问题。在 LLM 时代，预训练/SFT 仍有此问题，RLHF 的 rollout 是部分缓解，但完全解决需在线学习（推布数据回训）。

### 8.2 RLHF rollout 的缓解与残留

- **缓解**：PPO rollout 让 actor 在自生成前缀上训练，对齐训练与推布的前缀分布；
- **残留**：rollout prompt 分布 $\mathcal{D}_{\text{rollout}}$ 与最终推布 prompt 分布 $\mathcal{D}_{\text{serve}}$ 可能偏移（prompt 层 exposure bias）；
- **解法**：在线 RLHF（用真实用户 prompt 持续 rollout）+ RM 在线更新。

### 8.3 Iterative DPO 的缓解

Iterative DPO：用上轮 $\pi_\theta$ 生成 response → 标偏好对 → 下轮 DPO。生成用自回归，让 $\pi_\theta$ 见过自己前缀，部分缓解 exposure bias。是 DPO 的在线化变体。

### 8.4 与 LLM 幻觉的关系

exposure bias 与 LLM 幻觉（hallucination）部分相关：
- 推布 snowball 错误 → 生成不实内容（幻觉）；
- 但幻觉还有其他原因（知识缺失/训练数据错）；
- exposure bias 是幻觉的机制之一，非全部。

### 8.5 在线学习是终极解法

完全消除 exposure bias 需**在线学习**：用真实推布 prompt + 模型自生成 response + 用户反馈（reward）回训。这是 RLHF 在线 PPO + RM 在线更新的理论动力——让训练分布与推布分布完全对齐。但工程复杂（在线 rollout + 标注 + 稳定性），是现代对齐系统的前沿。

### 8.6 exposure bias 的检测

- 训练 loss 低但推布质量差 → exposure bias 信号；
- 对比 teacher forcing 与自生成的 perplexity 差距；
- 长序列推布质量比短序列差更多 → snowball 信号；
- 加噪前缀后模型输出崩 → 未见错误前缀的暴露。

---
相关: [[LLM特有RL问题]]、[[SFT]]、[[PPO optimization]]、[[RLHF (PPO)]]、[[DPO]]、[[MDP]]、[[mode collapse]]、[[policy collapse]]、[[reward hacking]]、[[KL explosion]]、[[RLAIF]]、[[teacher forcing]]
