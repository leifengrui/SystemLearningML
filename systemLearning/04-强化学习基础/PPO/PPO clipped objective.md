# PPO clipped objective

> **所属章节**: [[PPO]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 中等偏难（现代策略梯度的事实标准）

## 1. 一句话定义

**PPO（Proximal Policy Optimization，近端策略优化）** 的核心目标函数（**clipped surrogate objective**）是

$$
L^{CLIP}(\theta)=\mathbb{E}_t\left[\min\bigl(\rho_t(\theta)\,\hat A_t,\ \text{clip}(\rho_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat A_t\bigr)\right]
$$

其中 **importance ratio** $\rho_t(\theta)=\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$ 是新策略与采样策略的概率比，$\hat A_t$ 是 [[advantage function]] 的估计（通常用 [[GAE]]），$\epsilon$ 是 clipping 范围（常 0.1~0.3）。它用 **clip 把单步策略更新限制在信任域内**，使 on-policy 算法能像 SGD 那样**对同一批 rollout 做多 epoch 复用**而不会让策略一步跑飞——是 [[REINFORCE]]/[[Actor-Critic]] 之后工业界（含 RLHF）最常用的策略梯度变体。

> [!note] 一句话直觉
> 没有 clip 时，多 epoch 复用旧样本会让 $\rho$ 越走越远（旧策略采的样本对当前策略已严重失真），梯度方向失控。clip 像一个"保险阀"：当 $\rho$ 偏出 $[1-\epsilon,1+\epsilon]$ 且方向不利于稳定时，**截断梯度**，禁止策略继续朝那个方向剧烈漂移。代价是牺牲一点单调性保证，换来实现简单、调参容易——这是 PPO 相对 TRPO 的全部"卖点"。

## 2. 为什么需要它（动机与背景）

[[REINFORCE]] 是严格 on-policy：梯度 $\nabla J=\mathbb{E}[\nabla\log\pi_\theta\,G_t]$ 里的期望必须**在当前 $\pi_\theta$ 下采样**才无偏。这带来致命工程问题：

1. **样本用一次就丢**：一条 rollout 算完梯度就得扔，否则分布失配。RL 采样极贵（尤其 LLM-RL 要生成上百万 token），用一次浪费太大。
2. **一步更新太大就崩**：策略梯度没限制单步更新幅度，学习率稍大 $\pi_\theta$ 一步跳出信任域 → 性能断崖式下跌（[[policy collapse]]）。
3. **无法多 epoch 训练**：监督学习能对一个 batch 跑多个 epoch，REINFORCE 不行——多跑会让 $\pi_\theta$ 偏离采样分布 $\pi_{\theta_{old}}$ 越来越远，重要性权重 $\rho$ 爆炸。

**TRPO**（Trust Region Policy Optimization, Schulman 2015）先解决这问题：加 KL 约束 $\mathbb{E}[\text{KL}(\pi_{old}\|\pi_\theta)]\le\delta$ 保证策略单调改进，但解二次规划（CPO/共轭梯度）实现复杂、慢。**PPO**（Schulman 2017）的洞见：**把 KL 硬约束换成 clip 软约束**，直接在 surrogate loss 上截断 $\rho$，效果接近 TRPO 但实现就是一个 `torch.clamp`，调一个 $\epsilon$ 即可。这使 PPO 成为事实标准：OpenAI baselines、stable-baselines3、RLHF（TRL/verl/DeepSpeed-Chat）几乎全用 PPO。

> [!tip] PPO 的本质贡献
> PPO **没发明新的策略梯度定理**，它的梯度仍是 $\nabla\log\pi\cdot A$（[[REINFORCE]]）。它贡献的是**一个让 on-policy 能多 epoch 复用样本而不崩的稳定目标函数**——把"信任域"从约束变成 clip。理解这点就不至于把 PPO 神秘化。

## 3. 核心概念详解

### 3.1 importance ratio $\rho_t$

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

- $\pi_{\theta_{old}}$ 是**采样时**的策略（rollout 前的参数快照），在当前更新周期内冻结；
- $\rho_t$ 修正"用 $\pi_{\theta_{old}}$ 采的样本去估 $\pi_\theta$ 的期望"的分布偏差，这是 **importance sampling** 的权重（[[REINFORCE]] §8.2）；
- $\rho_t=1$ 表示新旧策略在该 $(s_t,a_t)$ 上完全一致；偏离 1 越多，策略漂移越大；
- 期望意义：$\mathbb{E}_{a\sim\pi_{old}}[\rho_t]=1$（归一化），故 $\rho\hat A$ 仍是 $A^\pi$ 的无偏估计（理论），但 $\rho$ 方差随策略漂移爆炸。

### 3.2 clipped surrogate objective

把朴素 importance-weighted surrogate $L^{IS}=\mathbb{E}[\rho_t\hat A_t]$ 改成：

$$
L^{CLIP}=\mathbb{E}_t\bigl[\min(\rho_t\hat A_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)\bigr]
$$

- **外层 min**：在"原值 $\rho\hat A$"与"clip 后值 $\text{clip}(\rho)\hat A$"之间取**较小**者。这保证 $L^{CLIP}$ 是真实目标的一个**下界**（悲观估计），从而不会高估改进、不会冒进；
- **clip**：把 $\rho$ 限制在 $[1-\epsilon,1+\epsilon]$，超出部分截断；
- **关键**：clip 不是对 $\rho$ 直接截断再乘 $\hat A$，而是**两路取 min**——这样梯度行为是分段 的（见 §3.4）。

### 3.3 四象限行为（理解 PPO 的核心图）

按 $\hat A_t$ 符号与 $\rho_t$ 大小分四种情况，决定梯度是否被截断：

| $\hat A_t$ 符号 | 含义 | 鼓励 $\rho$ 方向 | clip 边界 | 行为 |
|---|---|---|---|---|
| $\hat A>0$ | 该动作好于平均，应增大其概率 | $\rho\uparrow$（>1） | $1+\epsilon$ | $\rho<1+\epsilon$ 时正常梯度上升；$\rho\ge1+\epsilon$ 时梯度**截断为 0**，禁止继续增大 |
| $\hat A<0$ | 该动作差于平均，应减小其概率 | $\rho\downarrow$（<1） | $1-\epsilon$ | $\rho>1-\epsilon$ 时正常；$\rho\le1-\epsilon$ 时梯度**截断为 0**，禁止继续减小 |

> [!note] min 的作用
> 没有上限保护时，$\hat A>0$ 且 $\rho$ 已很大仍会继续推 $\rho$ 更大 → 策略过度偏向一个动作 → collapse。clip + min 在"已经改进到位"后**关掉梯度**。注意 min 还提供**反向保护**：$\hat A>0$ 但 $\rho<1-\epsilon$（即策略反而把这个好动作概率降低了）时，min 选 $\rho\hat A$（更小），梯度仍鼓励修正回来——clip 只在"过度改进"方向截断，不在"倒退"方向截断。这是 PPO 比朴素 clip 更精妙之处。

### 3.4 梯度的分段性

对 $L^{CLIP}$ 求 $\theta$ 的梯度（标量单样本版，忽略期望）：

$$
\frac{\partial L}{\partial\theta}=\hat A_t\cdot\frac{\partial\rho_t}{\partial\theta}\cdot\mathbb{1}[\text{clip 未触发}]
$$

- clip 未触发（$\rho\in(1-\epsilon,1+\epsilon)$ 或触发方向不利）→ 梯度 = $\hat A\cdot\nabla\rho$，正常策略梯度；
- clip 触发且"过度改进"方向 → 梯度 = 0，**该样本本 epoch 不再更新**。

这意味着 PPO 一个 batch 内每个 epoch 的有效样本数会**逐渐减少**（越更新越多样本被 clip 掉），是"自适应早停"的隐式机制。实践中常监控 `clip_fraction`，0.1~0.3 为健康。

### 3.5 多 epoch 复用（PPO 的工程价值）

PPO 一个 rollout batch 做 $K$ 个 epoch（典型 2~10，LLM-RL 常 2~4）的 mini-batch SGD。每个 epoch：

1. 重新算 $\rho_t=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{old}}(a_t|s_t))$（$\pi_{\theta_{old}}$ 全程冻结）；
2. $\hat A_t$ 用 rollout 时的 critic 算好，**全程不变**（detach）；
3. 算 $L^{CLIP}$ 反传。

随着 $\theta$ 更新，$\rho$ 偏离 1，clip 逐步生效，防止过度利用旧样本。$K$ 太大 → clip 过多、样本失真；$K$ 太小 → 样本利用率低。LLM-RL 因样本贵，倾向 $K=2\sim4$。

### 3.6 完整 PPO loss（actor + critic + entropy）

$$
L^{PPO}(\theta)=\underbrace{-L^{CLIP}(\theta)}_{\text{actor}}+\underbrace{c_v\,L^{VF}(\theta)}_{\text{critic}}\underbrace{-c_e\,S[\pi_\theta]}_{\text{entropy bonus}}
$$

- $L^{VF}=(V_\phi(s_t)-\hat G_t)^2$，critic 回归（$\hat G_t=V_\phi+\hat A$，[[GAE]] §4.3）；
- $S[\pi_\theta]=\mathbb{E}[H(\pi_\theta(\cdot|s_t))]$ 是 [[entropy bonus]]，鼓励探索；
- $c_v\approx0.5$，$c_e\approx0.01$（LLM-RL 常更小或 0）。
- actor 项取负：因优化器做梯度**下降**，而我们要**最大化** $L^{CLIP}$。

### 3.7 KL 监控：clip 之外的第二道闸

PPO 还常监控 $\text{KL}(\pi_{old}\|\pi_\theta)$（近似为 $\mathbb{E}[\rho\log\rho]$ 或直接算），若超过 `target_kl`（如 0.01~0.02）则**提前 break** 当前 batch 的 epoch 循环——这是 clip 之外的另一道防线，见 [[KL penalty与KL control]]。LLM-RL 中这道闸尤其重要（防 [[KL explosion]]）。

### 3.8 LLM-RL 中的 token 级 ratio

RLHF 里 $a_t$ 是 token，$\pi_\theta(a_t|s_t)$ 是 LLM 在 prompt+已生成前缀下对下一 token 的 softmax 概率。一条 response 长 $T$ → $T$ 个 $(s_t,a_t)$ 样本，每个算一个 $\rho_t$、$\hat A_t$。ratio 与 advantage 都在 token 级，是 PPO 在 LLM 上的直接应用（详见 §8）。

## 4. 数学原理 / 公式

### 4.1 从 importance sampling 推导 surrogate

目标 $J(\theta)=\mathbb{E}_{a\sim\pi_\theta}[A^{\pi_{old}}(s,a)]$，但样本是 $\pi_{\theta_{old}}$ 采的。用 importance weighting：

$$
J(\theta)=\mathbb{E}_{a\sim\pi_{\theta_{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}A^{\pi_{old}}(s,a)\right]=\mathbb{E}_{\pi_{old}}[\rho_t(\theta)\,\hat A_t]
$$

这就是 $L^{IS}=\mathbb{E}[\rho\hat A]$。**问题**：$\rho$ 无界，策略漂移大时方差爆炸（且高估改进）。PPO 用 clip 把它压住。

### 4.2 clip 下界保证（悲观 surrogate）

令 $L^{CLIP}=\mathbb{E}[\min(\rho\hat A,\,\text{clip}(\rho,1-\epsilon,1+\epsilon)\hat A)]$。对 $\hat A>0$：

$$
\min(\rho\hat A,\,\text{clip}(\rho,1-\epsilon,1+\epsilon)\hat A)\le\rho\hat A
$$

且 clip 段在 $\rho>1+\epsilon$ 时取 $1+\epsilon$，比 $\rho$ 小 → $L^{CLIP}\le L^{IS}$。对 $\hat A<0$ 同理（clip 在 $\rho<1-\epsilon$ 时取 $1-\epsilon$，因乘负数反而更大？需仔细：$\hat A<0$，$\rho<1-\epsilon$，$\text{clip}(\rho)=1-\epsilon>\rho$，$(1-\epsilon)\hat A<\rho\hat A$（更负），min 取 $\rho\hat A$? ——见误区 §7.4，需对 min 方向仔细）。

**结论**：$L^{CLIP}$ 是 $L^{IS}$ 的悲观下界（在"过度改进"方向），故最大化 $L^{CLIP}$ 不会高估真实 $J$，提供**保守策略改进**的数值保证（虽不严格单调，但实践中稳定）。

### 4.3 与 TRPO KL 约束的等价直觉

TRPO 解：$\max_\theta\mathbb{E}[\rho\hat A]$ s.t. $\mathbb{E}[\text{KL}(\pi_{old}\|\pi_\theta)]\le\delta$。

PPO 用 clip 近似该约束：$\rho\in[1-\epsilon,1+\epsilon]$ 大致对应 $\text{KL}\approx\epsilon^2/2$（小扰动展开）。故 $\epsilon=0.2$ 对应 KL 约 0.02，与 TRPO 常用 $\delta=0.01\sim0.05$ 同量级。这是"clip 约束 ≈ KL 约束"的近似依据。

### 4.4 ratio 期望为 1

$$
\mathbb{E}_{a\sim\pi_{old}}[\rho_t]=\sum_a\pi_{old}(a|s)\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}=\sum_a\pi_\theta(a|s)=1
$$

故 $\mathbb{E}[\rho\hat A]=\mathbb{E}_{\pi_\theta}[\hat A]$（无偏）。但方差 $\text{Var}(\rho)=\mathbb{E}[\rho^2]-1$ 随策略漂移增长，是 clip 存在的理由。

## 5. 代码示例

```python
import torch, torch.nn as nn, copy

class AC(nn.Module):
    def __init__(s,o,n): super().__init__(); s.a=nn.Sequential(nn.Linear(o,64),nn.Tanh(),nn.Linear(64,n)); s.c=nn.Sequential(nn.Linear(o,64),nn.Tanh(),nn.Linear(64,1))
    def pi(s,x): return torch.distributions.Categorical(logits=s.a(x))
    def v(s,x): return s.c(x).squeeze(-1)

ac=AC(4,2); opt=torch.optim.Adam(ac.parameters(),3e-3); gamma=0.99; lam=0.95; eps=0.2; K_epochs=4

def collect(n=256):
    buf=[]; s=torch.randn(4)
    for _ in range(n):
        d=ac.pi(s); a=d.sample(); logp_old=d.log_prob(a).item()
        sn=torch.randn(4); r=float(a==1)
        buf.append((s,a,r,sn,logp_old)); s=sn
    return buf

buf=collect()
S=torch.stack([t[0] for t in buf]); A=torch.tensor([t[1] for t in buf])
R=torch.tensor([t[2] for t in buf],dtype=torch.float32); SN=torch.stack([t[3] for t in buf])
logp_old=torch.tensor([t[4] for t in buf],dtype=torch.float32)

# 1) 用旧 critic 算 GAE（冻结，全程不变）
with torch.no_grad():
    Vs=ac.v(S); Vn=ac.v(SN); deltas=R+gamma*Vn-Vs
    adv=torch.zeros_like(deltas); adv[-1]=deltas[-1]
    for t in reversed(range(len(buf)-1)): adv[t]=deltas[t]+gamma*lam*adv[t+1]
    ret=Vs+adv                       # critic target = V + A
    adv=(adv-adv.mean())/(adv.std()+1e-8)   # advantage 归一化

# 2) PPO 多 epoch 复用
for _ in range(K_epochs):
    d=ac.pi(S); logp=d.log_prob(A)
    ratio=(logp-logp_old).exp()           # ρ = π_θ/π_old
    surr1=ratio*adv; surr2=torch.clamp(ratio,1-eps,1+eps)*adv
    actor_loss=-torch.min(surr1,surr2).mean()      # 最大化 L^CLIP → 取负
    critic_loss=((ac.v(S)-ret.detach())**2).mean()
    entropy=d.entropy().mean()
    loss=actor_loss+0.5*critic_loss-0.01*entropy
    opt.zero_grad(); loss.backward(); opt.step()

# 3) KL 监控（提前 break 用）
with torch.no_grad():
    kl=(ratio-1)-(logp_old+1e-8).log()  # 近似; 实际用 ρ*log ρ 的均值
    approx_kl=((ratio-1)-(logp_old+1e-8).log() if False else (ratio-1)-logp_old).mean()
print("clip_frac:", ((ratio-1-eps>0)|(1-eps-ratio>0)).float().mean().item())
```

> [!tip] 工程要点
> - `logp_old` 必须 **detach / 存成标量**，是采样时 $\pi_{\theta_{old}}$ 的快照，更新周期内不变。
> - `adv` 与 `ret` 都 detach，critic 与 actor 共享网络时尤其注意梯度不要串。
> - LLM-RL 里 `logp_old` 由 reference/actor 快照算，每条 response 的 token 级 logp 存 buffer。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[REINFORCE]]（策略梯度骨架）、[[advantage function]]（$\hat A$ 权重）、[[GAE]]（$\hat A$ 的工业估计法）、[[Actor-Critic]]（critic 提供 $V$）、importance sampling（$\rho$ 的来源）。
- **下游（应用）**: RLHF 的 PPO 内核（[[SFT]]→RM→PPO）、verl/TRL/DeepSpeed-Chat 的实现、[[KL penalty与KL control]]（clip 之外的第二道闸）、[[entropy bonus]]（PPO loss 第三项）。
- **对比 / 易混**:
  - **PPO vs TRPO**：TRPO 用 KL 硬约束+二阶求解，单调性严格但慢；PPO 用 clip 软约束+一阶 SGD，单调性近似但快、易调，工业主流。
  - **PPO vs REINFORCE**：REINFORCE 严格 on-policy 用一次、无 clip；PPO 多 epoch 复用+clip 防漂移。
  - **PPO vs DPO**：DPO 把 RLHF 重写成监督损失，**隐式**含 KL 与 reward；PPO 显式 rollout + clip + KL，样本效率低但可学 RM 未覆盖的行为。详见 [[DPO]]。
  - **clip vs KL penalty**：clip 是硬截断（梯度归零），KL penalty 是软惩罚（加到 reward/loss）；PPO 常二者并用，见 [[KL penalty与KL control]]。
  - **ratio clipping vs 梯度裁剪**：前者限制策略**概率比**（防分布漂移），后者限制梯度范数（防数值爆炸），是两回事（[[梯度裁剪]]）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **`logp_old` 没 detach**：采样时策略 $\pi_{\theta_{old}}$ 应冻结，若随 $\theta$ 变，$\rho$ 恒为 1，clip 失效，PPO 退化成 REINFORCE 且多 epoch 会爆。务必存成快照标量/detach。
> 2. **符号反**：$L^{CLIP}$ 要**最大化**，优化器做梯度下降 → actor loss 取负。漏负号越训越差。
> 3. **advantage 没 detach**：$\hat A$ 给 actor 时若带 critic 梯度，actor 会去最小化 $V$。`adv.detach()`。
> 4. **min 方向搞反**：min 是在"原值 $\rho\hat A$"与"clip 值"间取较小，提供**悲观下界**。误写成 max 会高估改进、策略跑飞。需配合 $\hat A$ 符号仔细推导四象限。
> 5. **$\hat A$ 用当前 critic 重算每个 epoch**：错。$\hat A$ 是 rollout 时用旧 critic 算好的，全程不变；每 epoch 只重算 $\rho$ 与 $L^{CLIP}$。
> 6. **clip 只截 $\rho$ 再乘 $\hat A$**：错。应是 $\min(\rho\hat A,\text{clip}(\rho)\hat A)$，否则会丢掉"倒退方向仍修正"的保护（§3.3）。
> 7. **$K$ epoch 过大**：LLM-RL $K>4$ 时 clip_fraction 飙升、样本失真，常取 2~4 并配合 target_kl 早停。
> 8. **不监控 clip_fraction / KL**：调参盲区。clip_fraction 0.1~0.3 健康；KL 超 target 提前 break。
> 9. **连续动作忘了重算 logp**：高斯策略每 epoch 要用当前 $\mu,\sigma$ 重算 $\log\pi(a|s)$，不能复用旧 logp。
> 10. **LLM-RL 把 ratio 算成序列级**：应是 **token 级** ratio，每个 token 一个 $\rho_t$；序列级会丢失细粒度信用分配。
> 11. **critic 与 actor 共享 backbone 时 critic loss 梯度污染 actor**：要么分离网络，要么 critic loss 系数小且 careful detach。

## 8. 延伸细节

### 8.1 PPO 的"信任域"近似质量

clip 是 TRPO KL 约束的**一阶近似**。$\rho=1+\delta$ 小扰动下 $\text{KL}\approx\mathbb{E}[\delta^2/2]$，$\epsilon=0.2$ → $\delta$ 上界 0.2 → KL 约 0.02。但 $\rho$ 远端时 clip 与 KL 约束**不等价**（clip 是硬截断、KL 是二次惩罚），故 PPO 在大更新时单调性弱于 TRPO。实践靠"$K$ 小 + target_kl 早停"补足。

### 8.2 target_kl 早停

每个 epoch 后算 `approx_kl = ((ratio-1)-log(ratio)).mean()` 或 `(ratio*log(ratio)).mean()`（前者更稳），超 `target_kl`（0.01~0.02）则 break。这把"KL 约束"从 penalty 形式转成"硬阈值早停"，与 clip 互补：clip 防单样本过度更新，target_kl 防整批漂移。

### 8.3 LLM-RL 的 PPO 全貌

RLHF-PPO 一条迭代：
1. **rollout**：当前 actor $\pi_\theta$ 生成 response，存 token 级 $(s_t,a_t,\log\pi_\theta(a_t|s_t))$；
2. **reward**：奖励模型 $r_\phi$ 对整段打分（仅末尾 reward），reference $\pi_{ref}$（冻结 SFT 模型）算 token 级 KL penalty 加到 reward：$r_t\leftarrow r_t-\beta\,\text{KL}(\pi_\theta(\cdot|s_t)\|\pi_{ref})$（见 [[KL penalty与KL control]]）；
3. **advantage**：用 GAE(γ≈1, λ≈0.95) 倒推 token 级 $\hat A_t$，critic 学 $V_\phi$；
4. **PPO update**：$K=2\sim4$ epoch，token 级 ratio + clip + value loss + (可选) entropy bonus；
5. **监控**：reward、KL、clip_fraction、response length、critic loss。

$\beta$（KL 系数）由 adaptive controller 调，target KL 约 6 nats/token 量级（见 [[KL penalty与KL control]] §8）。

### 8.4 PPO 样本效率的局限

PPO 仍 on-policy，rollout 用一两轮就丢，对 LLM 生成成本极高。这是 **DPO** 类离线对齐方法兴起的动机——DPO 不 rollout、不 clip，直接在偏好数据上监督训练。但 DPO 不能学 RM 未标注的分布外行为，且易过拟合偏好数据；PPO 在"持续探索+在线 reward"场景仍不可替代（如 RLEF/RLAIF with online RM）。

### 8.5 clip 的变体

- **P clipping（PPO-milestone / OpenAI 复现)**：部分实现对 advantage 正负分别 clip，或用 `clip(ratio,1-eps,1+eps)` 的下界与上界分写；
- **Quadratic penalty（Kakade 2002 style）**：在 $\rho$ 外加 $-c(\rho-1)^2$ 软惩罚，梯度连续不截断，但少用；
- **DaPO / GRPO**：近年 LLM-RL 变体调整 clip 策略（如 GRPO 去掉 critic、用组内相对 advantage），是 PPO 在 LLM 场景的演进，见 [[GRPO]]（待展开）。

### 8.6 代码级易错：ratio 的数值稳定

`ratio = (logp - logp_old).exp()` 在 $\log\pi$ 差大时会溢出。实践加 `torch.clamp(logp-logp_old, -10, 10).exp()` 或用 `log_ratio = logp - logp_old` 直接算 `surr = exp(log_ratio)*adv`。LLM-RL 中 logp 由 reference 与 actor logits 算，注意 float32 累加精度（[[数值类型与精度]]）。

---
相关: [[PPO]]、[[REINFORCE]]、[[advantage function]]、[[GAE]]、[[Actor-Critic]]、[[KL penalty与KL control]]、[[entropy bonus]]、[[baseline]]、[[variance reduction]]、[[梯度裁剪]]、[[policy collapse]]、[[KL explosion]]、[[reward hacking]]、[[DPO]]、[[SFT]]
