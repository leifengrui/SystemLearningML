# policy与value function

> **所属章节**: [[RL基本概念]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 基础（RL 两大主角）

## 1. 一句话定义

**策略（policy）** $\pi$ 是智能体在状态 $s$ 下选动作 $a$ 的规则——确定性 $\pi(s)=a$ 或随机性 $\pi(a\mid s)$（概率分布），是 RL 里**唯一要学/要优化**的对象。**价值函数（value function）** 是在策略 $\pi$ 下，从某状态（或状态-动作）出发的**期望累积折扣回报**——$V^\pi(s)$（状态价值）与 $Q^\pi(s,a)$（动作价值）——它评估"这个状态/动作有多好"，是策略改进的依据。策略是"做什么"，价值是"这么做能拿多少"，二者在 [[Actor-Critic]] 里被两个网络联合学习。

> [!note] 谁是参数、谁是辅助
> 在**策略梯度方法**（[[REINFORCE]]/[[PPO clipped objective]]）里，策略 $\pi_\theta(a\mid s)$ 是被梯度直接优化的"主角"，价值函数 $V_\phi^\pi(s)$ 是估计梯度的"辅助 critic"（减方差）。在**基于价值的方法**（DQN）里没有显式策略，学 $Q$ 后用 $\pi(s)=\arg\max_a Q(s,a)$ 隐式派生。在 [[Actor-Critic]] 里两者都学。

## 2. 为什么需要它（动机与背景）

[[MDP]] 定义了环境 $(S,A,P,R,\gamma)$，但没说智能体怎么决策。要"求解"一个 MDP，需要两个量：

1. **策略 $\pi$**：给定 $s$ 怎么选 $a$。这是智能体的行为本身，是最终要交付的东西（部署时只需 $\pi$）。
2. **价值 $V^\pi/Q^\pi$**：评估当前策略有多好、某动作是否值得。这是改进策略的信号。

二者形成"评估→改进"循环（policy iteration）：先估当前 $\pi$ 的价值，再贪心改进 $\pi$（选价值大的动作），再估新 $\pi$ 价值……直到收敛到最优 $\pi^*$。深度 RL 把这循环做在线：神经网络同时逼近 $\pi$ 与 $V/Q$，用采样数据增量更新。

策略的**参数化**是关键创新：$\pi_\theta(a\mid s)=\text{softmax}(\text{NN}_\theta(s))\to$ 可用梯度上升优化，这就是 [[REINFORCE]] 的起点，也是 LLM-RL 微调的本质（LLM 本身就是策略网络 $\pi_\theta(\text{token}\mid\text{prompt})$）。

## 3. 核心概念详解

### 3.1 策略的两种形式

| 形式 | 表达 | 典型场景 |
|---|---|---|
| 确定性 | $a=\pi(s)$ | DQN 派生策略、连续控制 DDPG/TD3 |
| 随机性 | $\pi(a\mid s)$，$\sum_a\pi(a\mid s)=1$ | 离散动作分类、策略梯度方法、探索所需 |

策略梯度**要求随机性策略**（要 $\nabla_\theta\log\pi$ 有定义）。确定性策略梯度（DPG）是连续动作的特殊变体。

### 3.2 参数化策略

- 离散动作：$\pi_\theta(a\mid s)=\text{softmax}(\text{NN}_\theta(s))_a$，输出 categorical 分布。
- 连续动作（高斯）：$\pi_\theta(a\mid s)=\mathcal{N}(\mu_\theta(s),\sigma^2)$，NN 输出均值（方差可学或固定）。
- LLM：$\pi_\theta(\text{token}\mid\text{context})$ 就是语言模型的下一个 token 分布（softmax over vocab）。

### 3.3 三大价值函数

| 函数 | 定义 | 含义 |
|---|---|---|
| $V^\pi(s)$ | $\mathbb{E}_\pi[G_t\mid s_t=s]$ | "在 $s$ 按 $\pi$ 走，期望拿多少" |
| $Q^\pi(s,a)$ | $\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]$ | "在 $s$ 先做 $a$，之后按 $\pi$ 走，期望拿多少" |
| $A^\pi(s,a)$ | $Q^\pi(s,a)-V^\pi(s)$ | "**优势**：这步 $a$ 比平均水平好多少"（见 [[advantage function]]） |

三者关系：$V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$，$A^\pi=Q^\pi-V^\pi$。

### 3.4 贝尔曼方程（价值的递推）

$$
V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\bigl[R(s,a,s')+\gamma V^\pi(s')\bigr]
$$
$$
Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)\bigl[R(s,a,s')+\gamma V^\pi(s')\bigr]=R(s,a)+\gamma\mathbb{E}_{s'}[V^\pi(s')]
$$

这是价值的"自洽"定义（不动点），也是时序差分（TD）学习的依据：用 $r+\gamma V(s')$ 作为 $V(s)$ 的**带噪目标**去回归。

### 3.5 最优价值与最优策略

$$
V^*(s)=\max_\pi V^\pi(s),\quad Q^*(s,a)=\max_\pi Q^\pi(s,a),\quad \pi^*(s)=\arg\max_a Q^*(s,a)
$$

- **值迭代**：反复套贝尔曼最优方程求 $V^*$；
- **策略迭代**：估 $V^\pi$ → 贪心改进 $\pi$ → 再估，循环；
- **Q-learning**：model-free 估 $Q^*$（用 $\max_{a'}Q(s',a')$ 作目标）；
- **策略梯度**：直接对 $J(\pi_\theta)$ 求梯度上升，不显式求 $Q^*$（见 [[REINFORCE]]）。

### 3.6 on-policy vs off-policy

- **on-policy**：用**当前 $\pi$ 采的样本**估/改 $\pi$（[[REINFORCE]]、[[PPO clipped objective]]、A2C）。样本利用率低但稳。
- **off-policy**：用**旧策略采的样本**改当前 $\pi$（Q-learning、DQN、SAC、Importance Sampling 版 PG）。靠重采样/重加权，样本利用率高但方差/偏差需处理。

LLM-RL（PPO-RLHF）是 on-policy（每轮用当前 policy rollout），因为 off-policy 在 LLM 上重加权方差爆炸。

## 4. 数学原理 / 公式

### 4.1 策略梯度定理（核心）

目标 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0]$，则：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right]
$$

证明要点（[[REINFORCE]] §4 详）：轨迹概率 $P(\tau\mid\theta)=\rho_0\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$，求导时只有 $\pi_\theta$ 含 $\theta$，$\nabla P/P=\nabla\log P=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)$。这是"策略梯度=对数策略导数×回报"的来源。

### 4.2 价值作 baseline 减方差

$$
\nabla_\theta J=\mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A^{\pi_\theta}(s_t,a_t)\right]
$$

把 $G_t$ 换成**优势** $A^{\pi_\theta}=Q-V$（见 [[baseline]]/[[advantage function]]），方差大降。这就是 [[Actor-Critic]] 的形态。

### 4.3 贝尔曼最优的压缩性

算子 $T V(s)=\max_a\sum_{s'}P(s'\mid s,a)[R+\gamma V(s')]$ 是 $\gamma$-压缩：$\|TV_1-TV_2\|_\infty\le\gamma\|V_1-V_2\|_\infty$，故值迭代收敛。这是 MDP 求解的理论保证。

## 5. 代码示例

```python
import torch, torch.nn as nn

# 离散动作的策略+价值网络（Actor-Critic 共享主干）
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh())
        self.actor = nn.Linear(64, n_act)     # 策略 logits
        self.critic = nn.Linear(64, 1)        # 状态价值 V(s)
    def policy(self, s):
        logits = self.actor(self.trunk(s))    # (B, n_act)
        return torch.distributions.Categorical(logits=logits)  # 随机策略 pi(a|s)
    def value(self, s):
        return self.critic(self.trunk(s)).squeeze(-1)          # V(s)

ac = ActorCritic(obs_dim=4, n_act=3)
s = torch.randn(8, 4)
dist = ac.policy(s)
a = dist.sample()                 # 采样动作
logp = dist.log_prob(a)           # log pi(a|s)，策略梯度要用
v = ac.value(s)                   # 价值估计
print(a.shape, logp.shape, v.shape)
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[MDP]]（策略与价值定义其上）、[[return与reward]]（价值=期望回报）、概率论（条件分布、期望）。
- **下游（应用）**: [[REINFORCE]]（对 $\pi_\theta$ 求梯度）、[[baseline]]（用 $V$ 减方差）、[[advantage function]]（$Q-V$）、[[Actor-Critic]]（actor 学 $\pi$、critic 学 $V$）、[[value network]]（神经网络逼近 $V/Q$）、[[advantage estimation]]（GAE）、[[PPO clipped objective]]（限制 $\pi_\theta/\pi_{\theta_{old}}$ 比率）、RLHF（LLM 即策略 $\pi_\theta$）。
- **对比 / 易混**:
  - **policy-based vs value-based**：前者显式学 $\pi$（策略梯度），后者学 $Q$ 隐式派生 $\pi$（DQN）。
  - **$V$ vs $Q$ vs $A$**：见 §3.3。
  - **on-policy vs off-policy**：见 §3.6。
  - **确定性 vs 随机性策略**：策略梯度要随机性；DQN 派生确定性。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **把 $V^\pi$ 当成"状态本身的价值"**：它是"在策略 $\pi$ 下"的期望回报，换策略就变。说"状态价值"必须指明是哪个 $\pi$ 的。
> 2. **确定性策略直接用策略梯度**：$\nabla\log\pi$ 对确定性策略无定义；要用随机性策略或确定性策略梯度（DPG）。
> 3. **混淆 $V^*$ 与 $V^\pi$**：$V^*$ 是最优策略下的价值（上限），$V^\pi$ 是当前策略的。训练中估的是 $V^{\pi_\theta}$，随训练变化。
> 4. **off-policy 当 on-policy 用**：旧样本直接训练当前 $\pi$ 会有分布偏移；需重要性采样或修正。
> 5. **策略网络输出不归一化**：连续动作要输出分布参数（均值+方差），不是直接输出动作值；离散要 softmax 成概率。
> 6. **价值函数与策略不同步**：Actor-Critic 里 critic 落后 actor 太多 → 梯度方向错；通常 critic 更新频率≥actor。
> 7. **以为 $\pi^*(s)=\arg\max_a Q(s,a)$ 总能算**：连续动作下 $\arg\max$ 无解析解，需策略梯度或归一化流。
> 8. **LLM-RL 里忘了 $\pi_\theta$ 就是 LLM**：把 LLM 当"模型"而非"策略"，导致把 RL 目标和 SFT 损失混着写。$\pi_\theta(\text{token}\mid\text{ctx})$ 就是策略，KL 约束是限制它偏离参考策略。

## 8. 延伸细节

### 8.1 价值函数的"-bootstrap"与"蒙特卡洛"两极

- **MC**：用完整轨迹的实际回报 $G_t$ 作 $V(s_t)$ 目标，无偏但高方差、需等回合结束。
- **TD(0)**：用 $r+\gamma V(s')$ 作目标，有偏（依赖 $V$ 估计）但低方差、可在线。
- **TD(λ)/GAE**：二者插值（见 [[GAE]]），是 [[advantage estimation]] 的核心。

### 8.2 策略熵与探索

随机性策略的熵 $H(\pi(\cdot\mid s))$ 衡量探索度。熵高→探索多；熵低→确定性、易陷局部最优。[[entropy bonus]] 在目标里加 $+\beta H$ 鼓励探索，是 [[PPO clipped objective]] 的标配。

### 8.3 LLM 作为策略的细节

LLM 微调时 $\pi_\theta(a\mid s)$ 的 $s$ 是 prompt+已生成 token、$a$ 是下一 token。序列概率 $\pi_\theta(y\mid x)=\prod_t\pi_\theta(y_t\mid x,y_{<t})$。策略梯度对 $\theta$ 求导等于对序列对数概率求导——与语言建模的交叉熵梯度同构，这是 RLHF 能直接套 PPO 的数学原因。

### 8.4 价值函数的归一化

实践中 $V$ 的绝对值尺度因任务差异大，常对 TD 误差做**回报归一化**（reward normalization）或对 $V$ 目标做 running mean/std 标准化，稳定训练。见 [[reward normalization]]。

### 8.5 双 Q 与值过估

off-policy 估 $Q$ 时 $\max$ 操作会放大高估偏差（max of noisy estimate）。Double DQN/Double Q-learning 用独立网络定 target 动作减偏差。策略梯度方法不显式 max，无此问题。

---
相关: [[RL基本概念]]、[[MDP]]、[[return与reward]]、[[REINFORCE]]、[[baseline]]、[[advantage function]]、[[Actor-Critic]]、[[value network]]、[[GAE]]、[[PPO clipped objective]]
