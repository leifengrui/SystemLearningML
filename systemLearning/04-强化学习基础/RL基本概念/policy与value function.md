# policy与value function

> **所属章节**: [[RL基本概念]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**策略（policy）** $\pi$ 是智能体在状态 $s$ 下选动作 $a$ 的规则——确定性 $\pi(s)=a$ 或随机性 $\pi(a\mid s)$，是 RL 里唯一要优化的对象。**价值函数（value function）** 是在策略 $\pi$ 下从某状态（或状态-动作）出发的**期望累积折扣回报**：$V^\pi(s)$（状态价值）与 $Q^\pi(s,a)$（动作价值），评估"这个状态/动作有多好"。策略是"做什么"，价值是"这么做能拿多少"。

> [!note] 主角与辅助
> 策略梯度方法（[[REINFORCE]]/[[PPO clipped objective]]）里 $\pi_\theta$ 是被梯度直接优化的主角，$V_\phi$ 是减方差的辅助 critic。基于价值方法（DQN）没有显式策略，学 $Q$ 后用 $\pi(s)=\arg\max_a Q(s,a)$ 隐式派生。[[Actor-Critic]] 两者都学。

## 2. 为什么需要它

[[MDP]] 定义了环境 $(S,A,P,R,\gamma)$，但没说智能体怎么决策。要"求解"一个 MDP，需要两个量：策略 $\pi$（智能体的行为，部署时只需它）和价值 $V^\pi/Q^\pi$（改进策略的信号）。二者形成"评估→改进"循环（policy iteration）：估当前 $\pi$ 价值 → 贪心改进 → 再估，直到收敛到 $\pi^*$。深度 RL 把这循环做在线：神经网络同时逼近 $\pi$ 与 $V/Q$，用采样数据增量更新。

策略的**参数化**是关键创新：$\pi_\theta(a\mid s)$ 可用梯度上升优化，这是 [[REINFORCE]] 的起点，也是 LLM-RL 的本质——LLM 本身就是策略网络 $\pi_\theta(\text{token}\mid\text{prompt})$。

## 3. 核心概念

### 3.1 策略的两种形式

| 形式 | 表达 | 典型场景 |
|---|---|---|
| 确定性 | $a=\pi(s)$ | DQN 派生、连续控制 DDPG/TD3 |
| 随机性 | $\pi(a\mid s)$，$\sum_a\pi=1$ | 离散分类、策略梯度、探索 |

策略梯度**要求随机性策略**（$\nabla_\theta\log\pi$ 要有定义）。确定性策略梯度（DPG）是连续动作的特殊变体。

### 3.2 参数化

- 离散：$\pi_\theta(a\mid s)=\mathrm{softmax}(\mathrm{NN}_\theta(s))_a$。
- 连续（高斯）：$\pi_\theta(a\mid s)=\mathcal{N}(\mu_\theta(s),\sigma^2)$。
- LLM：$\pi_\theta(\text{token}\mid\text{context})$ 就是下一个 token 分布。

### 3.3 三大价值函数

| 函数 | 定义 | 含义 |
|---|---|---|
| $V^\pi(s)$ | $\mathbb{E}_\pi[G_t\mid s_t=s]$ | 在 $s$ 按 $\pi$ 走的期望回报 |
| $Q^\pi(s,a)$ | $\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]$ | 在 $s$ 先做 $a$ 后按 $\pi$ 走的期望回报 |
| $A^\pi(s,a)$ | $Q^\pi(s,a)-V^\pi(s)$ | 优势：这步比平均好多少 |

关系：$V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$。

### 3.4 贝尔曼方程（价值的递推）

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)[R(s,a,s')+\gamma V^\pi(s')]$$

$$Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)[R(s,a,s')+\gamma V^\pi(s')]$$

$V^\pi$ 是不动点。TD 学习用 $r+\gamma V(s')$ 作为 $V(s)$ 的带噪目标去回归。

### 3.5 on-policy vs off-policy

- **on-policy**：用当前 $\pi$ 采的样本改 $\pi$（[[REINFORCE]]、[[PPO clipped objective]]）。样本利用率低但稳。
- **off-policy**：用旧策略采的样本改当前 $\pi$（Q-learning、DQN、SAC）。靠重采样/重加权，样本利用率高但方差/偏差需处理。

LLM-RL（PPO-RLHF）是 on-policy：每轮用当前 policy rollout。

## 4. 数学原理

### 4.1 策略梯度定理

$$\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right]$$

证明要点：轨迹概率求导时只有 $\pi_\theta$ 含 $\theta$，$\nabla P/P=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)$。这是"对数策略导数×回报"的来源。

### 4.2 价值作 baseline 减方差

$$\nabla_\theta J=\mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A^{\pi_\theta}(s_t,a_t)\right]$$

把 $G_t$ 换成优势 $A^{\pi_\theta}=Q-V$，方差大降——这就是 [[Actor-Critic]] 的形态。

### 4.3 贝尔曼最优的压缩性

算子 $TV(s)=\max_a\sum_{s'}P(s'\mid s,a)[R+\gamma V(s')]$ 是 $\gamma$-压缩：$\|TV_1-TV_2\|_\infty\le\gamma\|V_1-V_2\|_\infty$，故值迭代收敛。

## 5. 代码示例

```python
# 离散动作的 Actor-Critic 共享主干
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim,64), nn.Tanh(),
                                   nn.Linear(64,64), nn.Tanh())
        self.actor = nn.Linear(64, n_act)   # 策略 logits
        self.critic = nn.Linear(64, 1)     # V(s)
    def policy(self, s):
        return torch.distributions.Categorical(logits=self.actor(self.trunk(s)))
    def value(self, s):
        return self.critic(self.trunk(s)).squeeze(-1)
```

## 6. 与其他知识点的关系

- **上游**: [[MDP]]、[[return与reward]]（价值=期望回报）、概率论。
- **下游**: [[REINFORCE]]、[[baseline]]、[[advantage function]]、[[Actor-Critic]]、[[PPO clipped objective]]、RLHF。
- **对比**: policy-based（显式学 $\pi$）vs value-based（学 $Q$ 隐式派生 $\pi$）；on-policy vs off-policy；确定性 vs 随机性策略。

## 7. 常见误区

1. 把 $V^\pi$ 当成"状态本身的价值"——它依赖策略 $\pi$，换策略就变。
2. 确定性策略直接用策略梯度——$\nabla\log\pi$ 无定义，要用随机性策略或 DPG。
3. 混淆 $V^*$（最优上限）与 $V^\pi$（当前策略的，训练中随训练变化）。
4. off-policy 旧样本直接训当前 $\pi$ 会分布偏移，需重要性采样。
5. 策略网络输出不归一化——连续要输出分布参数，离散要 softmax。
6. Actor-Critic 里 critic 落后 actor 太多 → 梯度方向错。
7. 连续动作下 $\arg\max_a Q(s,a)$ 无解析解。
8. LLM-RL 里忘了 $\pi_\theta$ 就是 LLM，导致把 RL 目标和 SFT 损失混着写。

## 8. 延伸

- **MC vs TD**：MC 用完整轨迹 $G_t$ 作目标，无偏高方差；TD(0) 用 $r+\gamma V(s')$，有偏低方差可在线；TD(λ)/GAE 是插值（见 [[GAE]]）。
- **策略熵与探索**：随机性策略的熵衡量探索度，[[PPO clipped objective]] 标配加 $+\beta H$ 鼓励探索。
- **LLM 作为策略**：$\pi_\theta(y\mid x)=\prod_t\pi_\theta(y_t\mid x,y_{<t})$，策略梯度对 $\theta$ 求导等于对序列对数概率求导——与语言建模交叉熵梯度同构，是 RLHF 能套 PPO 的数学原因。

---
相关: [[RL基本概念]]、[[MDP]]、[[return与reward]]
