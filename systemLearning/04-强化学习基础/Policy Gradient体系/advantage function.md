# advantage function

> **所属章节**: [[Policy Gradient体系]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**advantage（优势函数）** $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ 衡量"在状态 $s$ 选动作 $a$ **比按策略 $\pi$ 的平均水平好多少**"：$A>0$ 优于平均（应增多）、$A<0$ 劣于平均（应减少）。策略梯度用 $A^\pi$ 作权重代替 [[return与reward]] 的 $G_t$，期望不变、方差大降（见 [[baseline]]），是 [[Actor-Critic]]/[[PPO clipped objective]] 的核心量。

直觉：$Q^\pi$ = "做 $a$ 这步能拿多少"；$V^\pi$ = "平均能拿多少"；$A=Q-V$ = "比平均多/少拿多少"。策略梯度只关心**动作选择带来的差异**，不关心**状态本身的底色**——advantage 正好把后者扣掉。

## 2. 为什么需要它

[[REINFORCE]] 用 $G_t$ 作权重，问题：$G_t$ 含"状态本身好坏"的固定部分（优势棋面下任何动作 $G$ 都大），这部分与动作选择无关，是噪声。[[baseline]] 证明减 $V^\pi$ 不偏且降方差，减完的 $G_t-V^\pi$ 正是 advantage 的 MC 估计。把"减 baseline"单独命名、单独估带来工程便利：

- advantage 量级稳定（约 0 附近），学习率好调；
- 可用 TD/GAE 比 MC 更准更省地估（见 [[GAE]]）；
- PPO 的 ratio clipping 直接作用在"advantage 加权的 ratio"上。

## 3. 核心概念

### 3.1 定义与性质

$$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$$

- $V^\pi(s)=\mathbb{E}_{a\sim\pi}[Q^\pi(s,a)]$ → $\mathbb{E}_{a\sim\pi}[A^\pi(s,a)]=0$（advantage 对策略动作的期望为 0）。
- $Q^\pi=V^\pi+A^\pi$（dueling 网络据此分头估 $V$ 与 $A$）。

### 3.2 在策略梯度里的角色

$$\nabla_\theta J(\theta)=\mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A^{\pi_\theta}(s_t,a_t)\right]$$

$A>0$ 增大该动作概率，$A<0$ 减小，$|A|$ 大则更新幅度大。相比 $G_t$ 去掉状态底色，梯度更聚焦动作选择。

### 3.3 估计方法对比

| 方法 | 公式 | 偏差/方差 |
|---|---|---|
| **MC** | $\hat A_t=G_t-V_\phi(s_t)$ | 无偏、高方差、需等回合 |
| **TD(0)** | $\hat A_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)=\delta_t$ | 有偏（依赖 $V_\phi$）、低方差、在线 |
| **GAE** | $\hat A_t^{GAE(\lambda)}=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}$ | 偏差/方差可调（λ 插值） |

$\delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$ 是 **TD 误差**。GAE 是 MC 与 TD 的指数插值，PPO/A3C 标配。

### 3.4 归一化

实践中常对一批 $\hat A_t$ 做 `(A - A.mean())/(A.std()+1e-8)`，进一步降方差稳训练，不改变相对排序，是 [[variance reduction]] 常用招。

## 4. 数学原理

### 4.1 advantage 形式的策略梯度

从 [[REINFORCE]] $\nabla J=\mathbb{E}[\sum_t\nabla\log\pi\,G_t]$，[[baseline]] 证明减 $V^\pi(s_t)$ 无偏：

$$\nabla J=\mathbb{E}\left[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)(G_t-V^\pi(s_t))\right]$$

而 $\mathbb{E}[G_t\mid s_t,a_t]=Q^\pi(s_t,a_t)$，故 $G_t-V^\pi(s_t)$ 是 $A^\pi=Q^\pi-V^\pi$ 的无偏 MC 估计。

### 4.2 TD 误差是 advantage 的无偏估计

$$\delta_t=r_{t+1}+\gamma V^\pi(s_{t+1})-V^\pi(s_t)$$

$$\mathbb{E}_{s_{t+1}}[\delta_t]=Q^\pi(s_t,a_t)-V^\pi(s_t)=A^\pi(s_t,a_t)$$

（$V^\pi$ 精确时）。这是用 $\delta_t$ 替代 $G_t-V$ 的依据，也是 [[GAE]] 的构造块。

### 4.3 GAE 偏差-方差插值

$$\hat A_t^{GAE(\lambda)}=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}$$

$\lambda=0$ → TD（低方差高偏差）；$\lambda=1$ → MC（高方差低偏差）；中间插值。详见 [[GAE]]。

## 5. 代码示例

```python
# advantage 三种估计（伪 rollout 已采集 trans = [(s,a,r,sn,done), ...]）
# 1) MC advantage: A = G - V
G = 0; Gs = [0]*len(trans)
for t in reversed(range(len(trans))):
    G = trans[t][2] + gamma*G*(not trans[t][4]); Gs[t]=G
Gs = torch.tensor(Gs, dtype=torch.float32)
Vs = torch.stack([ac.v(t[0]) for t in trans])
adv_mc = (Gs - Vs.detach())          # detach 防 critic 梯度混进 actor

# 2) TD(0) advantage: A = delta = r + gamma V(s') - V(s)
deltas = torch.tensor([t[2] + gamma*(0.0 if t[4] else ac.v(t[3]).item()) - ac.v(t[0]).item() for t in trans])

# 3) advantage 归一化（工程标配）
adv = (adv_mc - adv_mc.mean())/(adv_mc.std()+1e-8)
logps = torch.stack([ac.pi(t[0]).log_prob(t[1]) for t in trans])
actor_loss = -(logps * adv).mean()
critic_loss = ((Vs - Gs)**2).mean()
```

## 6. 与其他知识点的关系

- **上游**: [[policy与value function]]（$Q,V$ 定义）、[[return与reward]]（$G$）、[[baseline]]（$G-V=A$ 的来源）、[[REINFORCE]]。
- **下游**: [[Actor-Critic]]、[[GAE]]、[[advantage estimation]]、[[PPO clipped objective]]（ratio×advantage clipping）、[[value network]]、dueling DQN（$Q=V+A$）。
- **对比**：
  - **advantage vs $Q$/$V$**：$A=Q-V$；策略梯度用 $A$ 或 $Q$ 等价（差 baseline），用 $A$ 数值更稳。
  - **MC/TD/GAE advantage**：见 §3.3 偏差-方差权衡。

## 7. 常见误区与易错点

1. **advantage 没 detach critic**：$V_\phi$ 梯度混进 actor loss，actor 去最小化 $V$。应 `adv = (G - V).detach()`。
2. **混淆 advantage 与 return**：advantage 期望为 0（对策略动作），return 不为 0；用错会让梯度有偏/方差大。
3. **TD advantage 当 MC 用**：$\delta_t$ 依赖 $V_\phi$ 估准，未收敛时偏差大；早期用 MC，稳定后用 TD/GAE。
4. **GAE 的 $\lambda$ 调错**：$\lambda$ 大方差大偏差小，小则反；LLM 常用 0.95~1.0。
5. **advantage 不归一化**：量级随训练漂移，学习率难调；配合 batch 归一化。
6. **off-policy 用旧 advantage**：策略变了 advantage 也变，需重算或 importance reweight。

## 8. 延伸细节

**LLM-RL 里的 advantage**：RLHF 中 reward 通常仅序列末尾一次，return 退化为单点；advantage = return - critic(s_t)。critic 学"当前策略下从 $s_t$（prompt+已生成）平均能拿多少分"，advantage = "这个 token 选择比平均分高/低"。PPO 在 token 级用 GAE 倒推做信用分配，critic 不准则 advantage 噪声大，故 critic 常用更大网络 + 多 step 预训练。

**advantage 与 regret 同构**：bandit 里 advantage ≈ regret（选某臂比最优少多少）；MDP 里 advantage 是"比策略平均好多少"。两者都是"相对量"。

---
相关: [[Policy Gradient体系]]、[[baseline]]、[[REINFORCE]]、[[Actor-Critic]]、[[GAE]]、[[variance reduction]]、[[PPO clipped objective]]
