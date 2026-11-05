# PPO clipped objective

> **所属章节**: [[PPO]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**PPO（Proximal Policy Optimization）** 的核心目标函数（**clipped surrogate objective**）是

$$
L^{CLIP}(\theta)=\mathbb{E}_t\left[\min\bigl(\rho_t(\theta)\,\hat A_t,\ \text{clip}(\rho_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat A_t\bigr)\right]
$$

其中 **importance ratio** $\rho_t(\theta)=\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$ 是新策略与采样策略的概率比，$\hat A_t$ 是 [[advantage function]] 的估计（通常用 [[GAE]]），$\epsilon$ 是 clipping 范围（常 0.1~0.3）。它用 **clip 把单步策略更新限制在信任域内**，让 on-policy 算法能对同一批 rollout 做多 epoch 复用而不跑飞——是 [[REINFORCE]]/[[Actor-Critic]] 之后工业界（含 RLHF）的事实标准策略梯度变体。

## 2. 为什么需要它

[[REINFORCE]] 严格 on-policy：梯度 $\nabla J=\mathbb{E}[\nabla\log\pi\,G_t]$ 必须在当前 $\pi_\theta$ 下采样才无偏，带来三个硬伤：

1. **样本用一次就丢**：rollout 算完梯度得扔，RL 采样极贵（LLM-RL 要生成上百万 token）。
2. **一步更新太大就崩**：无单步更新幅度限制，学习率稍大就跳出信任域 → 性能断崖式下跌。
3. **无法多 epoch 训练**：多 epoch 会让 $\pi_\theta$ 偏离 $\pi_{\theta_{old}}$ 越来越远，重要性权重 $\rho$ 爆炸。

**TRPO**（Schulman 2015）用 KL 硬约束 $\mathbb{E}[\text{KL}(\pi_{old}\|\pi_\theta)]\le\delta$ 保证策略单调改进，但需解二次规划（共轭梯度），实现复杂、慢。**PPO**（Schulman 2017）的洞见：**把 KL 硬约束换成 clip 软约束**，直接在 surrogate loss 上截断 $\rho$，效果接近 TRPO 但实现就是一个 `torch.clamp`，调一个 $\epsilon$ 即可。

> [!tip] PPO 的本质贡献
> PPO **没发明新的策略梯度定理**，梯度仍是 $\nabla\log\pi\cdot A$（[[REINFORCE]]）。它贡献的是一个让 on-policy 能多 epoch 复用样本而不崩的稳定目标函数——把"信任域"从约束变成 clip。

## 3. ratio 与 clip 怎么工作

### 3.1 importance ratio $\rho_t$

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

- $\pi_{\theta_{old}}$ 是**采样时**的策略快照，在本轮更新内冻结；
- $\rho_t$ 是 importance sampling 权重，修正"用 $\pi_{\theta_{old}}$ 采的样本估 $\pi_\theta$ 的期望"的分布偏差；
- $\rho_t=1$ 表示新旧策略一致；偏离 1 越多漂移越大；
- $\mathbb{E}_{a\sim\pi_{old}}[\rho_t]=1$，故 $\rho\hat A$ 理论无偏，但 $\rho$ 方差随漂移爆炸 → 需要 clip。

### 3.2 clipped surrogate 与四象限行为

$$
L^{CLIP}=\mathbb{E}_t\bigl[\min(\rho_t\hat A_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)\bigr]
$$

- **外层 min**：在原值 $\rho\hat A$ 与 clip 后值间取**较小**者，保证 $L^{CLIP}$ 是真实目标的**悲观下界**，不会高估改进。
- **关键**：不是直接 clip $\rho$ 再乘 $\hat A$，而是两路取 min——这样在"过度改进"方向截断梯度，在"倒退"方向仍保留修正梯度。

按 $\hat A_t$ 符号与 $\rho_t$ 大小分四种情况：

| $\hat A_t$ | 鼓励 $\rho$ 方向 | clip 边界 | 行为 |
|---|---|---|---|
| $>0$（动作好） | $\rho\uparrow(>1)$ | $1+\epsilon$ | $\rho<1+\epsilon$ 正常梯度上升；$\rho\ge1+\epsilon$ **梯度截断为 0** |
| $<0$（动作差） | $\rho\downarrow(<1)$ | $1-\epsilon$ | $\rho>1-\epsilon$ 正常；$\rho\le1-\epsilon$ **梯度截断为 0** |

> [!note] min 的双向保护
> 没 clip 时 $\hat A>0$ 且 $\rho$ 已大会继续推 $\rho$ 更大 → collapse。clip+min 在"已经改进到位"后关掉梯度。同时 min 提供**反向保护**：$\hat A>0$ 但 $\rho<1-\epsilon$（策略反而降低了好动作概率）时，min 选 $\rho\hat A$（更小），梯度仍鼓励修正回来——clip 只在"过度改进"方向截断。

### 3.3 梯度的分段性

$$
\frac{\partial L}{\partial\theta}=\hat A_t\cdot\frac{\partial\rho_t}{\partial\theta}\cdot\mathbb{1}[\text{clip 未触发}]
$$

clip 触发且在"过度改进"方向时梯度 = 0，该样本本 epoch 不再更新。一个 batch 内有效样本随 epoch 逐渐减少（clip_fraction 上升），是隐式的自适应早停。实践中监控 `clip_fraction`，0.1~0.3 健康。

### 3.4 多 epoch 复用

PPO 一个 rollout batch 做 $K$ epoch（典型 2~10，LLM-RL 常 2~4）的 mini-batch SGD。每个 epoch：

1. 重算 $\rho_t=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{old}}(a_t|s_t))$（$\pi_{\theta_{old}}$ 全程冻结）；
2. $\hat A_t$ 用 rollout 时的 critic 算好，**全程不变**（detach）；
3. 算 $L^{CLIP}$ 反传。

$K$ 太大 → clip 过多、样本失真；$K$ 太小 → 样本利用率低。

### 3.5 完整 PPO loss

$$
L^{PPO}(\theta)=\underbrace{-L^{CLIP}(\theta)}_{\text{actor}}+\underbrace{c_v\,L^{VF}(\theta)}_{\text{critic}}\underbrace{-c_e\,S[\pi_\theta]}_{\text{entropy bonus}}
$$

- $L^{VF}=(V_\phi(s_t)-\hat G_t)^2$，critic 回归（$\hat G_t=V_\phi+\hat A$，[[GAE]]）；
- $S[\pi_\theta]$ 是 [[entropy bonus]]，鼓励探索；
- $c_v\approx0.5$，$c_e\approx0.01$（LLM-RL 常更小或 0）；
- actor 项取负：因优化器做梯度下降，而要**最大化** $L^{CLIP}$。

### 3.6 LLM-RL 的 token 级 ratio

RLHF 里 $a_t$ 是 token，$\pi_\theta(a_t|s_t)$ 是 LLM 在 prompt+已生成前缀下对下一 token 的 softmax 概率。一条 response 长 $T$ → $T$ 个 $(s_t,a_t)$ 样本，每个算一个 $\rho_t$、$\hat A_t$。ratio 与 advantage 都在 token 级。

## 4. 关键推导

### 4.1 从 importance sampling 推 surrogate

目标 $J(\theta)=\mathbb{E}_{a\sim\pi_\theta}[A^{\pi_{old}}(s,a)]$，样本是 $\pi_{\theta_{old}}$ 采的，importance weighting：

$$
J(\theta)=\mathbb{E}_{\pi_{old}}\left[\frac{\pi_\theta}{\pi_{old}}A^{\pi_{old}}\right]=\mathbb{E}_{\pi_{old}}[\rho_t\,\hat A_t]
$$

这就是 $L^{IS}=\mathbb{E}[\rho\hat A]$。问题：$\rho$ 无界，漂移大时方差爆且高估改进 → PPO 用 clip 压住。

### 4.2 clip 与 KL 约束的近似关系

TRPO：$\max\mathbb{E}[\rho\hat A]$ s.t. $\mathbb{E}[\text{KL}(\pi_{old}\|\pi_\theta)]\le\delta$。PPO 用 clip 近似该约束：$\rho=1+\delta'$ 小扰动下 $\text{KL}\approx\mathbb{E}[\delta'^2/2]\approx\epsilon^2/2$。$\epsilon=0.2$ → KL 约 0.02，与 TRPO 常用 $\delta=0.01\sim0.05$ 同量级。这是"clip ≈ KL 约束"的近似依据，但 $\rho$ 远端时两者不等价，PPO 在大更新时单调性弱于 TRPO，靠 target_kl 早停补足。

## 5. 代码示例

```python
import torch, torch.nn as nn

class AC(nn.Module):
    def __init__(s,o,n): super().__init__(); s.a=nn.Sequential(nn.Linear(o,64),nn.Tanh(),nn.Linear(64,n)); s.c=nn.Sequential(nn.Linear(o,64),nn.Tanh(),nn.Linear(64,1))
    def pi(s,x): return torch.distributions.Categorical(logits=s.a(x))
    def v(s,x): return s.c(x).squeeze(-1)

ac=AC(4,2); opt=torch.optim.Adam(ac.parameters(),3e-3); gamma=0.99; lam=0.95; eps=0.2; K=4

# rollout(简化)
S=torch.randn(256,4); A=torch.randint(0,2,(256,)); R=torch.zeros(256); R[A==1]=1.0
SN=torch.randn(256,4)

# 1) GAE(用旧 critic 算,全程不变)
with torch.no_grad():
    Vs=ac.v(S); Vn=ac.v(SN); deltas=R+gamma*Vn-Vs
    adv=torch.zeros_like(deltas); adv[-1]=deltas[-1]
    for t in reversed(range(255)): adv[t]=deltas[t]+gamma*lam*adv[t+1]
    ret=Vs+adv
    adv=(adv-adv.mean())/(adv.std()+1e-8)

# 2) PPO 多 epoch
logp_old=ac.pi(S).log_prob(A).detach()           # ρ_old 快照,必须 detach
for _ in range(K):
    d=ac.pi(S); logp=d.log_prob(A)
    ratio=(logp-logp_old).exp()                    # ρ = π_θ/π_old
    surr1=ratio*adv; surr2=torch.clamp(ratio,1-eps,1+eps)*adv
    actor_loss=-torch.min(surr1,surr2).mean()      # 最大化 L^CLIP → 取负
    critic_loss=((ac.v(S)-ret.detach())**2).mean()
    loss=actor_loss+0.5*critic_loss-0.01*d.entropy().mean()
    opt.zero_grad(); loss.backward(); opt.step()
```

> [!tip] 工程要点
> - `logp_old` 必须 detach / 存标量，是 $\pi_{\theta_{old}}$ 快照，更新周期内不变。
> - `adv` 与 `ret` 都 detach，防 actor/critic 梯度互窜。
> - 数值稳定：`ratio = (logp-logp_old).exp()`，logp 差大时可 `clamp(logp-logp_old,-10,10).exp()`。

## 6. 与其他知识点的关系

- **上游**: [[REINFORCE]]（策略梯度骨架）、[[advantage function]]（$\hat A$）、[[GAE]]（$\hat A$ 估计）、[[Actor-Critic]]（critic 提供 $V$）、importance sampling（$\rho$）。
- **下游**: RLHF 的 PPO 内核、verl/TRL/DeepSpeed-Chat 实现、[[KL penalty与KL control]]（clip 外的第二道闸）、[[entropy bonus]]（loss 第三项）。
- **对比**:
  - **PPO vs TRPO**：TRPO 用 KL 硬约束+二阶求解，单调性严格但慢；PPO 用 clip 软约束+一阶 SGD，单调性近似但快、易调，工业主流。
  - **PPO vs REINFORCE**：REINFORCE 严格 on-policy 用一次、无 clip；PPO 多 epoch 复用+clip 防漂移。
  - **PPO vs [[DPO]]**：DPO 把 RLHF 重写成监督损失，隐式含 KL 与 reward；PPO 显式 rollout + clip + KL。
  - **clip vs KL penalty**：clip 是硬截断（梯度归零），KL penalty 是软惩罚；PPO 常二者并用。

## 7. 常见误区

1. **`logp_old` 没 detach**：$\pi_{\theta_{old}}$ 应冻结，若随 $\theta$ 变，$\rho$ 恒为 1，clip 失效，PPO 退化成 REINFORCE 且多 epoch 会爆。
2. **符号反**：$L^{CLIP}$ 要最大化，优化器做梯度下降 → actor loss 取负。漏负号越训越差。
3. **advantage 没 detach**：$\hat A$ 给 actor 时若带 critic 梯度，actor 会去最小化 $V$。
4. **min 方向搞反**：min 在原值与 clip 值间取较小，提供悲观下界。误写成 max 会高估改进、策略跑飞。
5. **$\hat A$ 每 epoch 用当前 critic 重算**：错。$\hat A$ 是 rollout 时用旧 critic 算好的，全程不变；每 epoch 只重算 $\rho$。
6. **clip 只截 $\rho$ 再乘 $\hat A$**：错。应是 $\min(\rho\hat A,\text{clip}(\rho)\hat A)$，否则丢"倒退方向仍修正"的保护。
7. **$K$ 过大**：LLM-RL $K>4$ 时 clip_fraction 飙升、样本失真，常取 2~4 并配合 target_kl 早停。
8. **LLM-RL 把 ratio 算成序列级**：应是 token 级 ratio，每个 token 一个 $\rho_t$；序列级丢细粒度信用分配。

---
相关: [[PPO]]、[[REINFORCE]]、[[GAE]]、[[KL penalty与KL control]]、[[entropy bonus]]、[[Actor-Critic]]、[[DPO]]
