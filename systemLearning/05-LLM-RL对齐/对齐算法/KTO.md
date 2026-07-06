# KTO

> **所属章节**: [[对齐算法]]
> **所属模块**: [[05-LLM-RL对齐]]
> **难度**: 中（DPO 的非成对变体，核心是前景理论的不对称加权；目录标注"了解即可"）

## 1. 一句话定义

**KTO（Kahneman-Tversky Optimization，前景理论优化）** 是 Ethayarajh et al. 2024 提出的 [[DPO]] 变体：**只需二元好/坏标签**（这个 response 好/坏，**非成对**），用 Kahneman-Tversky 前景理论的**不对称价值函数**——好样本当 gain、坏样本当 loss 且加权（loss aversion），并引入动态 baseline $z_{\text{ref}}$（数据集上 log-ratio 的 running mean）让好/坏相对 baseline 评估。KTO 损失对好样本最大化 $\beta\log\frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}-z_{\text{ref}}$、对坏样本压低且乘 loss-aversion 权重 $\lambda$。它**数据门槛最低**（任何带 👍/👎 的 response 都能用，无需构造 chosen/rejected 对），在标注成本高的场景比 DPO 实用。

> [!note] KTO = DPO 去"成对"约束 + 用前景理论
> 一句话记忆：**"DPO 要成对偏好对，KTO 只要 👍/👎 单标签，且用前景理论让坏样本加权"**。仍是同一 KL-constrained 目标，但损失形式从 BT（成对差）换成前景价值函数（单点 + baseline + 不对称）。

## 2. 为什么需要它（动机与背景）

[[DPO]]/[[IPO]] 需**成对偏好对** $(x, y_w, y_l)$——同一 prompt 必须有 chosen 和 rejected 两个 response。问题：
1. **成对标注贵**：要构造"同一 prompt 的两个候选 + 哪个好"的对比，标注者需看两个再判断，成本高；
2. **现成数据用不上**：大量已有 👍/👎 单标签数据（如 Reddit upvote、用户反馈、 thumbs up/down）是非成对的，DPO 用不上；
3. **成对偏好信息冗余**：BT 损失只用了 $r_w-r_l$ 的差，丢掉绝对水平信息；
4. **人类评估不对称**：Kahneman-Tversky 前景理论表明人类对 loss（坏）比 gain（好）更敏感（loss aversion），DPO 的对称 BT 损失未建模此。

KTO 的动机：**用单标签 + 前景理论**。任何带好/坏标签的 response 都能训，且用前景价值函数建模人类的不对称评估。数据门槛大降，且更贴近人类心理。代价：丢失成对对比的精确性（无 chosen/rejected 直接对照），但实践表明 KTO 效果接近 DPO，且数据多时更优。

## 3. 核心概念详解

### 3.1 成对 vs 非成对（KTO 的数据优势）

| 数据形式 | 示例 | 适用算法 |
|---|---|---|
| 成对偏好 | (prompt, chosen, rejected) | [[DPO]]、[[IPO]]、[[RLHF (PPO)]] |
| 二元标签 | (prompt, response, 👍/👎) | **KTO** |

- 成对：需同一 prompt 的两个候选 + 标注哪个好；
- 二元：任何 response 带 good/bad 标签即可，无需配对；
- 现成 👍/👎 数据（用户反馈、upvote/downvote）海量且免费，KTO 直接用，DPO 不能。

### 3.2 前景理论（KTO 的心理学基础）

Kahneman-Tversky 前景理论（诺贝尔经济学奖）刻画人类风险决策：
- **价值函数**：$v(x)=\begin{cases}x^\alpha & x\ge0\text{（gain）}\\-\lambda(-x)^\alpha & x<0\text{（loss）}\end{cases}$
- **loss aversion**：$\lambda>1$，人类对 loss 比 gain 更敏感（典型 $\lambda\approx2.25$）；
- **参考点依赖**：gain/loss 相对参考点 $z_{\text{ref}}$ 评估，非绝对。

KTO 把此搬到 RLHF：好 response 当 gain、坏 response 当 loss（加权 $\lambda$），相对 baseline $z_{\text{ref}}$ 评估。

### 3.3 KTO 的核心组件

- **隐式 reward** $\hat r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$（同 DPO 的重参数化）；
- **动态 baseline** $z_{\text{ref}}$：数据集上 $\hat r$ 的 running mean（detach，类似 [[reward normalization]] 的均值归一），让好/坏相对 baseline 评估；
- **loss aversion** $\lambda$：坏样本的 loss 加权（$\lambda>1$），建模人类对坏 response 更敏感；
- **不对称损失**：好样本用 $\sigma(\hat r-z_{\text{ref}})$（推高 $\hat r$），坏样本用 $\lambda\cdot\sigma(z_{\text{ref}}-\hat r)$（压低 $\hat r$ 且加权）。

### 3.4 KTO 损失的直观读法

$$
\mathcal{L}_{\text{KTO}}=\underbrace{\mathbb{E}_{y\sim D_+}[\sigma(-(\hat r-z_{\text{ref}}))]}_{\text{好样本:让 }\hat r\text{ 高于 baseline}}+\underbrace{\lambda\,\mathbb{E}_{y\sim D_-}[\sigma(\hat r-z_{\text{ref}})]}_{\text{坏样本:让 }\hat r\text{ 低于 baseline,加权 }\lambda}
$$

- 好样本（$D_+$）：最大化 $\hat r-z_{\text{ref}}$（推高好 response 的隐式 reward）；
- 坏样本（$D_-$）：最小化 $\hat r-z_{\text{ref}}$（压低坏 response 的隐式 reward），且乘 $\lambda$ 加权；
- $z_{\text{ref}}$：动态 baseline，让"好/坏"是相对而非绝对（防 reward 尺度漂移）；
- $\lambda$：loss aversion，坏样本权重更大（贴近人类心理）。

### 3.5 KTO vs DPO 的对比

| 维度 | [[DPO]] | KTO |
|---|---|---|
| 数据 | 成对 $(y_w,y_l)$ | 二元 good/bad |
| 损失基础 | BT（$r_w-r_l$ 差） | 前景价值函数（单点 + baseline） |
| 不对称 | 无（对称） | 有（loss aversion $\lambda$） |
| baseline | 无（$Z$ 消掉） | $z_{\text{ref}}$ 动态 |
| 数据门槛 | 高（需配对） | 低（任何 👍/👎） |
| 信息利用 | 用差，丢绝对水平 | 用绝对 + 相对 |
| 心理建模 | 无 | 前景理论 |

## 4. 数学原理 / 公式

### 4.1 从前景理论到 KTO 损失

前景理论价值函数（简化，取 $\alpha=1$ 线性）：
$$
v(x;\text{ref})=\begin{cases}x-\text{ref} & x\ge\text{ref（gain）}\\-\lambda(\text{ref}-x) & x<\text{ref（loss）}\end{cases}
$$

KTO 把隐式 reward $\hat r=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 当前景值，$z_{\text{ref}}$ 当参考点：
- 好样本（$y\sim D_+$）：$\hat r$ 应高于 $z_{\text{ref}}$（gain）；
- 坏样本（$y\sim D_-$）：$\hat r$ 应低于 $z_{\text{ref}}$（loss，加权 $\lambda$）。

用 logistic 损失（$\ell(u)=-\log\sigma(u)$ 的对偶 $\sigma(-u)$）回归到此价值，得 KTO 损失：
$$
\boxed{\mathcal{L}_{\text{KTO}}(\theta)=\mathbb{E}_{y\sim D_+}\bigl[\sigma\bigl(-(\hat r_\theta(x,y)-z_{\text{ref}})\bigr)\bigr]+\lambda\,\mathbb{E}_{y\sim D_-}\bigl[\sigma\bigl(\hat r_\theta(x,y)-z_{\text{ref}}\bigr)\bigr]} \tag{1}
$$

其中 $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，$z_{\text{ref}}=\mathbb{E}_{y\sim D}[\hat r_\theta(x,y)]$ 的 running estimate（detach）。

### 4.2 与 DPO 损失的对照

DPO（[[DPO]] §4.4）：$\mathcal{L}_{\text{DPO}}=-\mathbb{E}[\log\sigma(\hat r(y_w)-\hat r(y_l))]$，用成对差，无 baseline，对称。

KTO（式 1）：用单点 $\hat r(y)$ 相对 $z_{\text{ref}}$，非成对，不对称（$\lambda$）。

> [!tip] 等价性
> 理论上，若 KTO 的好/坏样本配对采样（同一 prompt 一好一坏），且 $\lambda=1$，则 KTO 退化为类似 DPO 的形式（差 $\hat r(y_w)-\hat r(y_l)$ 用 baseline 分解）。KTO 是 DPO 的**非成对推广**。

### 4.3 梯度

对好样本（$y\sim D_+$）：
$$
\nabla_\theta\mathcal{L}_+=-\sigma(-(\hat r-z_{\text{ref}}))\cdot\nabla_\theta\hat r \tag{2}
$$
推高 $\hat r$（提升好 response 概率）。

对坏样本（$y\sim D_-$）：
$$
\nabla_\theta\mathcal{L}_-=\lambda\,\sigma(\hat r-z_{\text{ref}})\cdot\nabla_\theta\hat r \tag{3}
$$
压低 $\hat r$（压低坏 response 概率），且 $\lambda$ 加权。

$z_{\text{ref}}$ 是 detach 的 running mean，不回传梯度（类似 BatchNorm 的 running stats）。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F, copy

class LM(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)
    def logp(s, x, y):
        ctx = torch.cat([x, y[:, :-1]], 1)
        return F.log_softmax(s(ctx),-1).gather(-1, y.unsqueeze(-1)).squeeze(-1).sum(-1)

V, D = 50, 32
pi_sft = LM(V, D)
pi_theta = copy.deepcopy(pi_sft)
pi_ref = copy.deepcopy(pi_sft)
for p in pi_ref.parameters(): p.requires_grad_(False)
opt = torch.optim.Adam(pi_theta.parameters(), 1e-6)
beta, lam = 0.1, 1.33       # λ ≈ 1.33 (论文用 1.0~1.5)
z_ref = 0.0                 # running baseline (EMA)

# 二元标签数据:(prompt, response, is_good)
data = [(torch.randint(0,V,(4,4)), torch.randint(0,V,(4,8)), g) for g in [True,False]*10 for _ in range(2)]

for ep in range(3):
    for x, y, is_good in data:
        logp = pi_theta.logp(x, y)
        with torch.no_grad():
            ref = pi_ref.logp(x, y)
        r_hat = beta * (logp - ref)                       # 隐式 reward (B,)
        # KTO 损失 (1)
        if is_good:
            loss = torch.sigmoid(-(r_hat - z_ref)).mean() # 好样本:推高 r_hat
        else:
            loss = lam * torch.sigmoid(r_hat - z_ref).mean()  # 坏样本:压低 r_hat,加权 λ
        opt.zero_grad(); loss.backward(); opt.step()
        # 更新 running baseline (EMA,detach)
        with torch.no_grad():
            z_ref = 0.9 * z_ref + 0.1 * r_hat.mean().item()
    print(f"ep{ep} z_ref={z_ref:.3f} loss={loss:.3f}")

print("KTO 训练完成:只用 👍/👎 标签,无需成对偏好对")
```

> [!tip] TRL 支持
> TRL 的 `KTOTrainer` 封装上述流程，支持 `beta`、`loss_weight`（$\lambda$）、`desirable_weight`/`undesirable_weight` 配置，大模型与 LoRA 兼容。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[DPO]]（KTO 是其非成对推广，同重参数化同目标）、[[SFT]]、[[KL penalty与KL control]]（β 角色）、前景理论（心理学基础，非本库笔记）。
- **下游（应用）**: 对齐后 LLM（KTO 版，适合 👍/👎 数据多的场景）、与 [[RLAIF]] 结合（AI 标 good/bad 更易）。
- **对比 / 易混**:
  - **KTO vs [[DPO]]/[[IPO]]**：成对 vs 非成对。KTO 数据门槛低，但无成对对比精确。
  - **KTO 的 $\lambda$ vs β**：$\lambda$ 是 loss aversion（坏样本加权），β 是偏离 ref 的温度；不同参数。
  - **KTO 的 $z_{\text{ref}}$ vs [[reward normalization]]**：两者都是动态 baseline 归一，KTO 用在 log-ratio 空间，reward normalization 用在 RM 输出空间。
  - **KTO vs [[RLHF (PPO)]]**：KTO 闭式无 rollout（轻），PPO 显式 RL（重但灵活）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"KTO 是排序"**：错。KTO 是**分类**（good/bad 二元），不是排序（DPO 是成对排序）。
> 2. **"KTO 不需要 baseline $z_{\text{ref}}$"**：错。$z_{\text{ref}}$ 是核心，让好/坏相对评估，防 reward 尺度漂移。漏 $z_{\text{ref}}$ 训练不稳。
> 3. **"$\lambda$ 是 KL 系数"**：错。$\lambda$ 是 loss aversion（坏样本加权），不是 β/KL。β 仍是偏离 ref 的温度。
> 4. **"KTO 比 DPO 一定好"**：不一定。成对偏好数据充足且干净时 DPO 更精确；KTO 优势在数据门槛低，数据多时更优。
> 5. **"KTO 可在线探索"**：错。和 DPO 一样离线固定数据，无 rollout。
> 6. **"$z_{\text{ref}}$ 回传梯度"**：错。$z_{\text{ref}}$ 是 running mean，detach，不回传（类似 BatchNorm running stats）。
> 7. **"KTO 不需 SFT"**：错。$\pi_{\text{ref}}$ 与 $\pi_\theta$ 初始化仍来自 SFT。
> 8. **"好/坏样本要 1:1"**：不强制。KTO 用 $\lambda$ 与采样平衡处理类别不平衡，但实践中常重采样平衡。

## 8. 延伸细节

### 8.1 KTO 的数据优势场景

- **现成 👍/👎 数据**：Reddit upvote、用户反馈、thumbs up/down，海量免费，KTO 直接用，DPO 不能；
- **AI 标签**：[[RLAIF]] 用 AI 标 good/bad 比 AI 构造成对偏好更易（单点判断 vs 配对比较），KTO+RLAIF 成本极低；
- **持续学习**：线上用户反馈实时标 good/bad，KTO 持续微调，比 DPO 重新配对轻。

### 8.2 KTO 的实践地位

KTO 在学术上重要（首次把前景理论引入对齐 + 解放非成对数据），实践中**适合 👍/👎 数据多的场景**（如对话系统的用户反馈对齐）。但成对偏好数据充足时 DPO 仍主流。KTO 与 DPO 的选择取决于数据形态：有成对用 DPO，只有单标签用 KTO。

### 8.3 前景理论的参数选择

- $\lambda$（loss aversion）：论文用 1.0~1.5，前景理论原始估计 ~2.25，对齐场景常取较小（数据已平衡）；
- $\alpha$（价值曲率）：KTO 论文取线性 $\alpha=1$ 简化；
- $\beta$：同 DPO，0.1~0.5。
这些参数对效果有影响，但比 DPO 的 β 不敏感（因 $z_{\text{ref}}$ 自适应）。

### 8.4 KTO 与 reward normalization 的关联

KTO 的 $z_{\text{ref}}$ 本质是 log-ratio 空间的 reward normalization（running mean 归一），与 [[reward normalization]]（PPO 中 RM 输出的 running std 归一）思路同源——都用动态统计量稳化训练。KTO 把此内建进损失，DPO 没有内置 normalization（故 DPO 对 reward 尺度更敏感，需外归一）。

### 8.5 DPO 家族总览

| 算法 | 数据 | 损失核心 | 特色 |
|---|---|---|---|
| [[DPO]] | 成对 | BT | 主流，闭式 |
| [[IPO]] | 成对 | squared（target margin） | 防 DPO 过推 |
| **KTO** | 二元 | 前景价值（不对称） | 非成对，数据门槛低 |
| SimPO | 成对 | 去 ref + 长度归一 | 更省 |
| rDPO | 成对 | 噪声鲁棒 | 抗偏好噪声 |

DPO 家族都是"闭式解 RLHF 目标 + 换损失/数据形式"的变体，KTO 是数据维度（成对→非成对）的推广。

---
相关: [[对齐算法]]、[[DPO]]、[[IPO]]、[[RLHF (PPO)]]、[[SFT]]、[[KL penalty与KL control]]、[[reward normalization]]、[[RLAIF]]、[[Bradley-Terry模型]]
