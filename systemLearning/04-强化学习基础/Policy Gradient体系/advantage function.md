# advantage function

> **所属章节**: [[Policy Gradient体系]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 中等（策略梯度的"正确权重"）

## 1. 一句话定义

**advantage function（优势函数）** $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ 衡量"在状态 $s$ 选动作 $a$ **比按策略 $\pi$ 的平均水平好多少**"——$A>0$ 表示这步 $a$ 优于平均（应增多）、$A<0$ 表示劣于平均（应减少）、$A=0$ 表示平平。策略梯度用 $A^\pi$ 作权重代替 [[return与reward]] 的 $G_t$：$\nabla_\theta J=\mathbb{E}[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)A^{\pi_\theta}(s_t,a_t)]$，期望不变方差大降（见 [[baseline]]），是 [[Actor-Critic]]/[[PPO clipped objective]] 的核心量。

> [!note] 一句话直觉
> $Q^\pi(s,a)$ = "做 $a$ 这步能拿多少"；$V^\pi(s)$ = "平均能拿多少"；$A^\pi=Q-V$ = "**比平均多/少拿多少**"。策略梯度只关心"动作选择带来的差异"，不关心"状态本身的底色"——advantage 正好把后者扣掉。

## 2. 为什么需要它（动机与背景）

[[REINFORCE]] 用 $G_t$ 作权重，问题：$G_t$ 含"状态本身好坏"的固定部分（优势棋面下任何动作 $G$ 都大），这部分与动作选择无关，是噪声。[[baseline]] 证明减 $V^\pi$ 不偏且降方差，减完的 $G_t-V^\pi$ 正是 advantage 的 MC 估计。所以 advantage 不是新发明，而是"baseline 减完后的自然产物"——但把它**单独命名、单独估**带来工程便利：

- advantage 量级稳定（约 0 附近），学习率好调；
- 可用 TD/GAE 等方法**比 MC 更准更省地估** advantage（见 [[GAE]]/[[advantage estimation]]）；
- PPO 的 ratio clipping 直接作用在"advantage 加权的 ratio"上。

advantage 把策略梯度从"绝对回报加权"升级为"相对优势加权"，是策略梯度方法可用的关键。

## 3. 核心概念详解

### 3.1 定义与三种关系

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

- $V^\pi(s)=\mathbb{E}_{a\sim\pi}[Q^\pi(s,a)]$ → $\mathbb{E}_{a\sim\pi}[A^\pi(s,a)]=0$（advantage 对策略动作的期望为 0，这是"比平均"的体现）。
- $Q^\pi(s,a)=V^\pi(s)+A^\pi(s,a)$（dueling 网络据此分头估 $V$ 与 $A$）。

### 3.2 advantage 在策略梯度里的角色

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A^{\pi_\theta}(s_t,a_t)\right]
$$

- $A>0$ → 沿 $\nabla\log\pi$ 方向更新 → 增大该动作概率；
- $A<0$ → 反向 → 减小；
- $|A|$ 大 → 更新幅度大。

相比 $G_t$，advantage 去掉状态底色，梯度更聚焦"动作选择"。

### 3.3 估 advantage 的三种途径

| 方法 | 公式 | 偏差/方差 |
|---|---|---|
| **MC** | $\hat A_t=G_t-V_\phi(s_t)$ | 无偏、高方差、需等回合 |
| **TD(0)** | $\hat A_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)=\delta_t$ | 有偏（依赖 $V_\phi$）、低方差、在线 |
| **GAE** | $\hat A_t^{GAE(\lambda)}=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}$ | 偏差/方差可调（λ 插值） |

其中 $\delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$ 是 **TD 误差**。GAE 是 MC 与 TD 的指数插值，是 PPO/A3C 的标配（见 [[GAE]]）。

### 3.4 advantage 的归一化

实践中常对一批 $\hat A_t$ 做 `(A - A.mean())/(A.std()+1e-8)`，进一步降方差稳训练。这不改变相对排序，只调尺度，是 [[variance reduction]] 的常用招。

### 3.5 advantage 与 $Q$ 的等价性

策略梯度用 $A$ 或 $Q$ 都行（差一个与 $a$ 无关的 $V$，[[baseline]] 不变）。但用 $A$ 数值更稳、量级更可控，是现代标准。

### 3.6 LLM-RL 里的 advantage

RLHF 中 reward 仅序列末尾一次，return 退化为单点；advantage = return - critic(s_t)。critic 学"当前策略下从 $s_t$（prompt+已生成）平均能拿多少分"，advantage = "这个 token 选择比平均分高/低"。这是 [[PPO clipped objective]] 在 token 级做信用的依据。

## 4. 数学原理 / 公式

### 4.1 advantage 形式的策略梯度推导

从 [[REINFORCE]] $\nabla J=\mathbb{E}[\sum_t\nabla\log\pi\,G_t]$，[[baseline]] 证明减 $V^\pi(s_t)$ 无偏：

$$
\nabla J=\mathbb{E}\left[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)(G_t-V^\pi(s_t))\right]
$$

而 $\mathbb{E}[G_t\mid s_t,a_t]=Q^\pi(s_t,a_t)$，故 $G_t-V^\pi(s_t)$ 是 $Q^\pi-V^\pi=A^\pi$ 的无偏 MC 估计：

$$
\nabla J=\mathbb{E}\left[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)\,A^\pi(s_t,a_t)\right]
$$

### 4.2 TD 误差与 advantage 的等价

$$
\delta_t=r_{t+1}+\gamma V^\pi(s_{t+1})-V^\pi(s_t)
$$

取期望（对 $s_{t+1}$）：

$$
\mathbb{E}_{s_{t+1}}[\delta_t]=R(s_t,a_t)+\gamma\mathbb{E}[V^\pi(s_{t+1})]-V^\pi(s_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)=A^\pi(s_t,a_t)
$$

故 **TD 误差是 advantage 的无偏估计**（在 $V^\pi$ 精确时）。这是用 $\delta_t$ 替代 $G_t-V$ 的理论依据，也是 [[GAE]] 的构造块。

### 4.3 GAE 的偏差-方差插值

$$
\hat A_t^{GAE(\lambda)}=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}
$$

- $\lambda=0$：$\hat A=\delta_t$（TD，低方差高偏差）；
- $\lambda=1$：$\hat A=G_t-V$（MC，高方差低偏差）；
- $0<\lambda<1$：插值。详见 [[GAE]]。

## 5. 代码示例

```python
import torch, torch.nn as nn

class AC(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,n_act))
        self.critic = nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,1))
    def pi(self, s): return torch.distributions.Categorical(logits=self.actor(s))
    def v(self, s):  return self.critic(s).squeeze(-1)

ac = AC(4,2); opt = torch.optim.Adam(ac.parameters(),1e-3); gamma=0.99

# 一段 rollout
trans = []
s = torch.randn(4)
for _ in range(64):
    d = ac.pi(s); a = d.sample()
    sn = torch.randn(4); r = float(a==1); done = False
    trans.append((s,a,r,sn,done)); s = sn

# 1) MC advantage: A = G - V
G = 0; Gs = [0]*len(trans)
for t in reversed(range(len(trans))):
    G = trans[t][2] + gamma*G*(not trans[t][4]); Gs[t]=G
Gs = torch.tensor(Gs, dtype=torch.float32)
Vs = torch.stack([ac.v(t[0]) for t in trans])
adv_mc = (Gs - Vs.detach())
print("MC adv mean/std:", adv_mc.mean().item(), adv_mc.std().item())

# 2) TD(0) advantage: A = delta = r + gamma V(s') - V(s)
deltas = []
for s,a,r,sn,done in trans:
    v_next = 0.0 if done else ac.v(sn).item()
    deltas.append(r + gamma*v_next - ac.v(s).item())
adv_td = torch.tensor(deltas)
print("TD adv mean/std:", adv_td.mean().item(), adv_td.std().item())

# 3) advantage 归一化（工程标配）
adv = (adv_mc - adv_mc.mean())/(adv_mc.std()+1e-8)
logps = torch.stack([ac.pi(t[0]).log_prob(t[1]) for t in trans])
actor_loss = -(logps * adv).mean()
critic_loss = ((Vs - Gs)**2).mean()
loss = actor_loss + 0.5*critic_loss
opt.zero_grad(); loss.backward(); opt.step()
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[policy与value function]]（$Q,V$ 定义）、[[return与reward]]（$G$）、[[baseline]]（$G-V=A$ 的来源）、[[REINFORCE]]（advantage 替换 $G$）。
- **下游（应用）**: [[Actor-Critic]]（critic 估 $V$，actor 用 $A$）、[[advantage estimation]]（如何估 $A$）、[[GAE]]（$\lambda$-return 版 advantage）、[[PPO clipped objective]]（ratio×advantage clipping）、[[value network]]（参数化 $V$）、dueling DQN（$Q=V+A$ 网络结构）。
- **对比 / 易混**:
  - **advantage vs $Q$ vs $V$**：$A=Q-V$；策略梯度用 $A$ 或 $Q$ 等价（差 baseline）。
  - **advantage vs return**：return 是绝对，advantage 是相对（扣状态底色）。
  - **MC advantage vs TD advantage vs GAE**：见 §3.3，偏差-方差权衡。
  - **advantage 归一化 vs reward 归一化**：前者归一化策略梯度权重，后者归一化 reward 尺度，互补。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **advantage 没 detach critic**：$V_\phi$ 的梯度混进 actor loss，actor 去最小化 $V$。`adv = (G - V).detach()`。
> 2. **混淆 advantage 与 return**：advantage 期望为 0（对策略动作），return 不为 0。用错会让梯度有偏/方差大。
> 3. **TD advantage 当 MC 用**：$\delta_t$ 依赖 $V_\phi$ 估准，未收敛时偏差大；早期用 MC，稳定后用 TD/GAE。
> 4. **GAE 的 $\lambda$ 调错**：$\lambda$ 大方差大偏差小，小则反；LLM 常用 0.95~1.0。
> 5. **advantage 不归一化**：量级随训练漂移，学习率难调；配合 batch 归一化。
> 6. **以为 advantage 改变最优策略**：advantage 是 $Q-V$，最优策略仍是 $\arg\max_a Q^*=\arg\max_a A^*$；只是梯度表达更稳。
> 7. **off-policy 用旧 advantage**：策略变了 advantage 也变，需重算或 importance reweight。
> 8. **LLM-RL 把序列级 reward 平均到每个 token 当 advantage**：太粗糙；要用 GAE 倒推 + critic 减 baseline。

## 8. 延伸细节

### 8.1 advantage 的"信度"含义

$A>0$ 的动作"值得多做"，但要多做多少？正比于 $A$——advantage 兼当"信度"。这与监督学习里"难样本权重更大"同构：advantage 大的样本对梯度贡献大。

### 8.2 GAE 与 n-step

n-step advantage $\hat A_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V(s_{t+n})-V(s_t)$，GAE 是对 n 做 $\lambda$ 加权平均。GAE 比 fixed n-step 自适应，是主流。

### 8.3 advantage 在 PPO 里的作用

PPO 的 clipped objective：$L=\mathbb{E}[\min(\rho\hat A,\text{clip}(\rho,1-\epsilon,1+\epsilon)\hat A)]$，$\rho=\pi_\theta/\pi_{\theta_{old}}$。advantage $\hat A$ 是权重，clip 限制更新幅度。advantage 估得准 PPO 才稳，故 GAE + critic 是 PPO 标配。

### 8.4 advantage 与 regret

bandit 里 advantage ≈ regret（选某臂比最优少多少）；MDP 里 advantage 是"比策略平均好多少"，方向相反但同构。两者都是"相对量"。

### 8.5 LLM-RL 的 advantage 实践

RLHF 通常：rollout 生成回答 → reward model 打分 → 用 GAE 算每个 token 的 advantage → critic 同步学 → PPO 更新 policy。critic 不准则 advantage 噪声大，故 critic 常用更大网络 + 多 step 预训练。详见 [[PPO clipped objective]] §8。

---
相关: [[Policy Gradient体系]]、[[baseline]]、[[REINFORCE]]、[[Actor-Critic]]、[[advantage estimation]]、[[GAE]]、[[PPO clipped objective]]、[[KL penalty与KL control]]、[[entropy bonus]]、[[value network]]、[[variance reduction]]
