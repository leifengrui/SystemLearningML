# entropy regularization

> **所属章节**: [[常见trick]]
> **所属模块**: [[12-训练稳定性工程]]
> **别名**: entropy regularization / entropy bonus / entropy regularization / 熵正则
> **难度**: 中（需懂 policy gradient + PPO + 信息熵）


## 1. 一句话定义

**entropy regularization（熵正则）** 是在 policy gradient / PPO 的目标中加 **entropy bonus**（$+\beta_e H(\pi_\theta)$，鼓励高熵=多样探索），防 **mode collapse**（policy 塌到单一输出）和过早收敛到局部最优。熵 $H(\pi) = -\sum \pi \log \pi$，高熵 = policy 分布平（多样探索），低熵 = 分布尖（确定输出）。policy gradient 易 mode collapse（学到 reward 高的单一输出就塌），entropy reg 鼓励保持多样，逼 policy 探索更多解。与 [[KL divergence control]] 方向相反（entropy 鼓励偏离=探索，KL 约束偏离=防崩），需 balance $\beta_e$ 与 $\beta_{\text{KL}}$。是 RLHF/PPO 训练稳定性的核心之一。A3C（Asynchronous Advantage Actor-Critic）首次系统化引入。

> [!note] 三句话定位
> - **是什么**：加 entropy bonus 鼓励高熵（多样探索），防 mode collapse。
> - **为什么**：policy gradient 易塌到单一输出，entropy reg 逼探索更多解。
> - **怎么用**：$+\beta_e H(\pi)$，$\beta_e \approx 0.01$，与 KL penalty balance。


## 2. 为什么需要它（动机与背景）

### 2.1 mode collapse 的现象

policy gradient（REINFORCE/PPO）训练中，policy 一旦发现 reward 高的某输出 $y^*$，$\pi(y^*)$ 增大，reward 更集中在 $y^*$，正反馈致 $\pi(y^*) \to 1$，其他 $\to 0$——**mode collapse**。policy 塌到单一输出，停止探索，可能困在局部最优（$y^*$ 非全局最优）。RLHF 中 mode collapse 表现：模型总输出相同模板/重复短语。

### 2.2 entropy reg 的作用

加 entropy bonus：$L = \mathbb{E}[\text{reward}] + \beta_e H(\pi)$。高熵（分布平）→ bonus 大 → 鼓励多样。低熵（分布尖）→ bonus 小 → 惩罚塌缩。逼 policy 在 reward 高的同时保持多样，探索更多解，防过早 mode collapse。

### 2.3 与 KL control 的 balance

[[KL divergence control]] 约束 $\pi_\theta$ 不偏离 $\pi_{\text{ref}}$（防崩，方向=收敛）。entropy reg 鼓励 $\pi_\theta$ 高熵（防塌，方向=探索）。两者方向相反：

- entropy reg：防 mode collapse（探索）。
- KL control：防偏离过远（收敛）。

需 balance $\beta_e$（entropy）与 $\beta_{\text{KL}}$（KL）：太大 entropy 欠收敛（太散），太大 KL 过收敛（太尖）。典型 $\beta_e = 0.01$, $\beta_{\text{KL}} = 0.1$。

### 2.4 RLHF 中的 mode collapse

RLHF/PPO 的 mode collapse：

- **模板化输出**：policy 学到某高 reward 模板，所有输入都输出该模板。
- **重复短语**：reward model 偏好重复，policy 塌到重复。
- **单一风格**：policy 失去多样性，所有输出同风格。

entropy reg 鼓励 policy 保持多样输出，缓解。与 [[KL divergence control]]（防偏离 SFT）配合：KL 保语义质量，entropy 保多样性。


## 3. 核心概念详解

### 3.1 熵的定义

$$
H(\pi) = -\sum_y \pi(y) \log \pi(y)
$$

- 均匀分布（$\pi$ 平）：$H$ 最大（多样）。
- 确定分布（$\pi$ 尖，某 $y$ 概率 1）：$H = 0$（确定）。

高熵 = 探索（policy 不确定选哪个），低熵 = 利用（policy 确定选某 $y$）。

### 3.2 entropy bonus

policy gradient 目标加 entropy bonus：

$$
L = \mathbb{E}_{y \sim \pi}\left[\sum_t \nabla \log \pi(y_t) A_t\right] + \beta_e H(\pi)
$$

或 PPO：

$$
L = \mathbb{E}\left[\min(r_t A_t, \text{clip}(r_t) A_t)\right] + \beta_e H(\pi)
$$

$\beta_e > 0$ 鼓励高熵。梯度 $\nabla H = -\sum (\log \pi + 1) \nabla \pi$，增 $\pi$ 的平度。

### 3.3 $\beta_e$ 的调节

| $\beta_e$ | 效果 |
|---|---|
| 太大（>0.1） | policy 过散，欠收敛（reward 低） |
| 适中（~0.01） | balance 探索与利用 |
| 太小（~0） | 无探索，易 mode collapse |
| 0 | 纯 policy gradient，易塌 |

典型 $\beta_e = 0.001-0.01$（RLHF）。A3C 用 0.01。

### 3.4 entropy 与 temperature 的关系

temperature $\tau$ 控制 softmax 平度：$\pi(y) \propto \exp(\text{logit}/\tau)$。

- $\tau$ 大：分布平（高熵）。
- $\tau$ 小：分布尖（低熵）。

entropy bonus 间接鼓励"effective temperature 大"（分布平）。learned temperature（$\tau$ 可学）是 entropy reg 的变体。

### 3.5 与 KL control 的 balance

| 维度 | entropy reg | KL control |
|---|---|---|
| 方向 | 鼓励偏离（探索） | 约束偏离（收敛） |
| 防什么 | mode collapse | reward hacking / 崩 |
| 系数 | $\beta_e \approx 0.01$ | $\beta_{\text{KL}} \approx 0.1$ |
| 目标 | 多样 | 不偏离 SFT |

两者需 balance：entropy 防 policy 塌，KL 防 policy 飘。$\beta_e / \beta_{\text{KL}}$ 的比例是调参点。

### 3.6 DPO 的 entropy

DPO 无显式 entropy bonus（不像 PPO）。DPO 的 $\beta$（目标函数中的）隐式控制偏离容忍度，间接影响 entropy。DPO 的 mode collapse 风险低于 PPO（无需在线采样，offline 数据约束），但仍需调 $\beta$ 防过拟合偏好数据。


## 4. 数学原理 / 公式

### 4.1 熵

$$
H(\pi) = -\sum_y \pi(y) \log \pi(y), \quad H \in [0, \log |\mathcal{Y}|]
$$

均匀分布 $H = \log |\mathcal{Y}|$（最大），确定分布 $H = 0$。

### 4.2 entropy bonus

$$
L = L_{\text{policy}} + \beta_e H(\pi)
$$

$L_{\text{policy}}$ 是 policy gradient（REINFORCE/PPO）的原始目标。

### 4.3 entropy 的梯度

$$
\nabla_\theta H = -\sum_y \nabla_\theta \pi(y) (\log \pi(y) + 1)
$$

增 $\pi$ 的平度（使分布更均匀）。

### 4.4 与 KL 的对比

KL penalty 减 $D_{\text{KL}}(\pi \| \pi_{\text{ref}})$（缩到 reference），entropy bonus 增 $H(\pi)$（扩到均匀）。两者方向相反：

$$
L = \mathbb{E}[r] + \beta_e H(\pi) - \beta_{\text{KL}} D_{\text{KL}}(\pi \| \pi_{\text{ref}})
$$

balance $\beta_e, \beta_{\text{KL}}$。

### 4.5 mode collapse 的正反馈

若 $\pi(y^*)$ 增（reward 高），下步 $y^*$ 采样更多，reward 更集中 $y^*$，$\pi(y^*)$ 再增——正反馈致 collapse。entropy bonus 的负反馈（$\pi$ 尖则 $H$ 降，bonus 减）抑制此正反馈。


## 5. 代码示例（可选

### 5.1 熵计算

```python
import math
def entropy(probs):
    """H(π) = -Σ π log π. 高=多样, 低=确定."""
    return -sum(p * math.log(p + 1e-12) for p in probs if p > 0)

print(f"uniform  [0.25,0.25,0.25,0.25]: H={entropy([0.25]*4):.4f} (max={math.log(4):.4f})")
print(f"peaked   [0.97,0.01,0.01,0.01]: H={entropy([0.97,0.01,0.01,0.01]):.4f} (低)")
print(f"mid      [0.4,0.3,0.2,0.1]:     H={entropy([0.4,0.3,0.2,0.1]):.4f}")
```

### 5.2 entropy bonus 在 policy loss

```python
def policy_loss_with_entropy(logprobs, advantages, probs, beta_e=0.01):
    """L = -mean(logprob*A) - beta_e*H (maximize H = minimize -H)."""
    pg = sum(lp * a for lp, a in zip(logprobs, advantages)) / len(logprobs)
    H = -sum(p * math.log(p+1e-12) for p in probs if p > 0)   # H 不除 N
    return -pg - beta_e * H   # 负号: 最大化 reward+H = 最小化 -reward-H

lp = [-0.5, -0.3, -0.8, -0.2]
adv = [1.0, -0.5, 0.8, -0.3]
p = [0.4, 0.3, 0.2, 0.1]
print(f"loss (β_e=0.01): {policy_loss_with_entropy(lp, adv, p, 0.01):.4f}")
print(f"loss (β_e=0):    {policy_loss_with_entropy(lp, adv, p, 0.0):.4f}")
```


## 6. 与其他知识点的关系

- **上游（依赖）**: policy gradient（entropy bonus 的场景）、PPO（RLHF 的 entropy reg）、信息熵（数学基础）。
- **下游（应用）**: [[KL divergence control]]（与 entropy reg balance）、[[reward normalization]]（稳 reward 使 entropy bonus 有效）、[[clipping strategies]]（PPO clip + entropy 稳 PPO）、[[loss spike handling]]（mode collapse 致不稳）、RLHF/PPO 训练、A3C（entropy reg 首次系统化）。
- **对比 / 易混**:
  - **entropy reg vs [[KL divergence control\|KL penalty]]**：方向相反。entropy 鼓励偏离（探索，防 collapse），KL 约束偏离（收敛，防崩）。需 balance。详见 §3.5。
  - **entropy reg vs [[reward normalization]]**：前者改目标（加 bonus），后者改 reward 尺度（不改目标）。不同层级。
  - **entropy reg vs temperature**：entropy bonus 间接鼓励"effective temperature 大"（分布平）。learned temperature 是 entropy reg 的变体。
  - **entropy reg vs [[clipping strategies\|PPO clip]]**：clip 稳更新幅度（防单步偏离），entropy 防分布塌缩（防 collapse）。不同维度，PPO 同时用。


## 7. 常见误区与易错点

> [!warning] 误区 1：entropy 越大越好
> 不是。entropy 过大 policy 过散，欠收敛（reward 低，不学）。需 balance：适中 entropy 探索而不发散。$\beta_e$ 太大（>0.1）reward 降。

> [!warning] 误区 2：entropy reg = KL penalty
> 方向相反。entropy 鼓励偏离（探索，防 collapse），KL 约束偏离（收敛，防崩）。两者需 balance，非等价。详见 §3.5。

> [!warning] 误区 3：任意 $\beta_e$
> 需调。$\beta_e$ 太大欠收敛，太小无探索。典型 0.001-0.01。需按任务/reward 调。与 $\beta_{\text{KL}}$ 的比例是调参点。

> [!warning] 误区 4：entropy reg 解决 mode collapse
> 缓解非根治。entropy bonus 抑制 collapse 的正反馈，但强 reward signal 仍可能致 collapse。需配合 KL control / reward normalization / 数据多样。

> [!warning] 误区 5：DPO 也用 entropy reg
> DPO 无显式 entropy bonus（不像 PPO）。DPO 的 $\beta$ 隐式控制偏离，间接影响 entropy。DPO 的 mode collapse 风险低于 PPO（offline 数据约束），但仍需调 $\beta$。

> [!warning] 误区 6：entropy = randomness
> 不完全。entropy 是分布的不确定性（policy 选哪个不确定），非输出随机。高 entropy policy 仍可确定选某 $y$（如多项式分布的众数），只是分布平。entropy 鼓励分布平，非输出随机。

> [!warning] 误区 7：entropy reg 只 RLHF 用
> 不只。policy gradient / RL 普遍用（A3C/A2C/SAC 等都用 entropy reg 防过早收敛）。entropy reg 是 RL 通用技巧，非 RLHF 专属。


## 8. 延伸细节

### 8.1 A3C 的 entropy reg

A3C（Mnih 2016）首次系统化在 actor-critic 加 entropy bonus：$L = L_{\text{policy}} + L_{\text{value}} + \beta_e H$。$\beta_e = 0.01$（原论文）。防 Atari 等任务的 mode collapse，是 deep RL 标配。

### 8.2 RLHF 的 entropy reg

RLHF/PPO 的 entropy reg：

- $\beta_e \approx 0.001-0.01$（比 A3C 小，因 LLM output space 大，entropy 本身高）。
- 与 KL penalty（$\beta_{\text{KL}} \approx 0.1$）balance。
- 监测 entropy 曲线：持续降 = 正常收敛；突降 = mode collapse 预警。

### 8.3 learned temperature

learned temperature（$\tau$ 可学）是 entropy reg 的变体：

- $\tau$ 大：分布平（高熵）。
- $\tau$ 小：分布尖（低熵）。

学 $\tau$ 间接控制 entropy。某些模型（如 GPT-2 early RLHF）用 learned temperature 替代 entropy bonus。

### 8.4 entropy 与 reward 的 trade-off

$\beta_e$ 控制 reward vs entropy 的 balance：

- $\beta_e$ 大：偏 entropy（多样，reward 低）。
- $\beta_e$ 小：偏 reward（高 reward，但可能 collapse）。
- 最优 $\beta_e$：reward 高且不 collapse。

调 $\beta_e$ 是 RLHF 调参点。与 $\beta_{\text{KL}}$ 一起调。

### 8.5 entropy 监测

训练中监测 entropy：

- 持续降（收敛）：正常。
- 突降（<某阈值）：mode collapse 预警，增 $\beta_e$。
- 持续低：已 collapse，需 rollback / 增 $\beta_e$ 重训。

entropy 曲线与 KL 曲线、loss 曲线并列监测，是 RLHF 训练健康指标。

### 8.6 entropy reg 与 reward hacking

entropy reg 缓解 mode collapse 型 reward hacking（policy 塌到某 hacking 输出）。但非 mode collapse 型 hacking（如长度 exploit，分布不塌只是系统性偏长）entropy reg 不直接管。需配合 KL control / reward shaping。entropy reg 是防 collapse，非防所有 hacking。

### 8.7 SAC 的 entropy（最大熵 RL）

SAC（Soft Actor-Critic）用最大熵 RL：目标 $= \mathbb{E}[r] + \alpha H$，$\alpha$ 是 temperature（自动调）。是 entropy reg 的极端形式（entropy 与 reward 同等地位）。SAC 的 $\alpha$ auto-tune 是 entropy reg 的自适应版。RLHF 的 entropy reg 较温和（$\beta_e$ 小，辅助）。

---
相关: [[常见trick]] | [[KL divergence control]] | [[reward normalization]] | [[clipping strategies]] | [[loss spike handling]] | [[warmup]] | [[gradient explosion and vanishing]] | [[PPO]] | [[RLHF]] | [[DPO]] | [[reward model]] | [[A3C]] | [[SAC]] | [[temperature]] | [[policy gradient]] | [[advantage estimation]] | [[GAE]]
