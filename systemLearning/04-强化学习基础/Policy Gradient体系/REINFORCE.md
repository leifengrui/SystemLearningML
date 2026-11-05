# REINFORCE

> **所属章节**: [[Policy Gradient体系]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**REINFORCE**（vanilla policy gradient，最简策略梯度算法）直接对**期望回报** $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0]$ 求梯度上升：$\nabla_\theta J=\mathbb{E}[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t]$，用蒙特卡洛样本估期望——即"**每步对数策略导数 × 该步 return**"求平均作为梯度。无 critic、无偏、但方差极大，是 [[Actor-Critic]]/[[PPO clipped objective]] 的理论起点。别名 MC policy gradient，由 Williams 1992 提出。

## 2. 为什么需要它

[[MDP]] 定义了 RL 目标 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0]$，但没说怎么算梯度。难点在于轨迹分布 $P(\tau\mid\theta)$ 含环境转移 $P$（未知），似乎没法对 $\theta$ 求导。

REINFORCE 的核心突破是**策略梯度定理**：求导时 $P$ 不含 $\theta$ 会消失，只剩对 $\log\pi_\theta$ 的求导，于是**不需要知道环境模型 $P$** 也能算梯度——只要能采样轨迹。这把 RL 从"需要环境模型"解放到"只需交互采样"，是 model-free 深度 RL 的起点。

但 $G_t$ 作权重方差极大，实际几乎不直接用，所有现代策略梯度方法都是它的减方差+稳定化改进。理解 REINFORCE = 理解策略梯度的骨架。

## 3. 核心概念

### 3.1 目标与梯度

$$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0],\qquad \nabla_\theta J=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right]$$

### 3.2 为什么是 $\log\pi$（score function trick）

求导用对数恒等式 $\nabla P(\tau\mid\theta)=P(\tau\mid\theta)\nabla\log P(\tau\mid\theta)$。轨迹概率 $P(\tau\mid\theta)=\rho_0\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$，取 $\log$ 后连乘变连加：

$$\log P(\tau\mid\theta)=\log\rho_0+\sum_t[\log\pi_\theta(a_t\mid s_t)+\log P(s_{t+1}\mid s_t,a_t)]$$

对 $\theta$ 求导时 $\log P$ 与 $\log\rho_0$ 不含 $\theta$ 直接消失，只剩 $\sum_t\nabla\log\pi_\theta(a_t\mid s_t)$。取 $\log$ 的收益：

- **连乘变连加**：直接对乘积求导用乘积法则产生指数级交叉项，取 log 后线性 $O(T)$；
- **干净分离策略与环境**：环境项求导为 0 → model-free 的数学根基；
- **数值防下溢**：长序列 $\prod\pi$ 下溢到 0，$\sum\log\pi$ 是负数累加不下溢（LLM-RL 序列上千 token 必须用 log）；
- **梯度量级归一化**：$\nabla\log\pi=\nabla\pi/\pi$，分母把大小概率动作的梯度量级拉平；
- **与 SFT 交叉熵同构**：$\nabla\log\pi_\theta(a_t\mid s_t)$ 正是"让动作 $a_t$ 概率增大"的方向，RLHF 直接复用 LLM 训练框架，只是权重从 1 换成 $G_t$/$\hat A_t$。

### 3.3 直觉：增大好轨迹的动作概率

- $G_t>0$ → 沿 $\nabla\log\pi$ 方向更新 → 增大 $\pi_\theta(a_t\mid s_t)$；
- $G_t<0$ → 反向 → 减小该动作概率。

即"回报为正的动作多采样、为负的少采样"，正比于 return。

### 3.4 算法流程

1. 用当前 $\pi_\theta$ 采 $N$ 条轨迹；
2. 每条轨迹递推算每步 $G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}$；
3. 梯度 $\hat g=\frac1N\sum_\tau\sum_t\nabla\log\pi_\theta(a_t\mid s_t)G_t$；
4. $\theta\leftarrow\theta+\eta\hat g$（或丢给优化器）。

### 3.5 缺陷：高方差

$G_t$ 是单条轨迹的实际回报，随机性极大。同一状态不同轨迹 $G_t$ 差几个数量级 → 梯度方差爆炸 → 学习率必须极小 → 收敛慢。这正是 [[baseline]]、[[Actor-Critic]]、[[GAE]]、[[variance reduction]] 存在的理由。

## 4. 数学原理

### 4.1 策略梯度定理推导

$$\begin{aligned}
\nabla_\theta J(\theta)&=\int \nabla_\theta P(\tau\mid\theta)G_0\,d\tau
=\int P(\tau\mid\theta)\nabla_\theta\log P(\tau\mid\theta)G_0\,d\tau\\
&=\mathbb{E}_{\tau\sim\pi_\theta}\bigl[G_0\,\nabla_\theta\log P(\tau\mid\theta)\bigr]
=\mathbb{E}\left[\sum_t G_0\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]
\end{aligned}$$

（最后一步用 §3.2，环境项消失。）

### 4.2 因果性：$G_t$ 而非 $G_0$

$\nabla\log\pi_\theta(a_t\mid s_t)$ 与 $t'<t$ 的 reward 无关（$a_t$ 还没发生），故可从 $G_0$ 去掉 $a_t$ 之前的 reward 得 $G_t$，期望不变方差降：

$$\boxed{\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right]}$$

### 4.3 baseline 不变性

对任何只依赖 $s_t$ 的 $b(s_t)$（详见 [[baseline]] §4.1）：

$$\mathbb{E}[\nabla\log\pi_\theta(a_t\mid s_t)\,b(s_t)]=0$$

故减 baseline 不改期望梯度，但 $b\approx V^\pi$ 能大幅降方差，此时 $G_t-b\approx A^\pi$（[[advantage function]]）。

## 5. 代码示例

```python
import torch, torch.nn as nn

class Policy(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,n_act))
    def forward(self, s):
        return torch.distributions.Categorical(logits=self.net(s))

pi = Policy(4, 2); opt = torch.optim.Adam(pi.parameters(), lr=1e-2); gamma = 0.99

# rollout 已采集 traj = [(s, a, r, logp), ...]
# 1) 从后往前递推 G_t
G = 0; returns = [0]*len(traj)
for t in reversed(range(len(traj))):
    G = traj[t][2] + gamma*G; returns[t] = G
returns = torch.tensor(returns)
# 2) 策略梯度: -sum(logp * G)，负号因优化器做梯度下降实现梯度上升
logps = torch.stack([t[3] for t in traj])
loss = -(logps * returns).sum()
opt.zero_grad(); loss.backward(); opt.step()
```

> [!tip] 实际不会裸用
> 上面 `returns` 不减均值，方差大。生产代码至少 `returns = (returns - returns.mean())/(returns.std()+1e-8)` 再上 [[Actor-Critic]]。

## 6. 与其他知识点的关系

- **上游**: [[MDP]]、[[policy与value function]]、[[return与reward]]、概率论（score function estimator）。
- **下游**: [[baseline]]、[[advantage function]]、[[variance reduction]]、[[Actor-Critic]]、[[PPO clipped objective]]、RLHF（LLM-RL 的策略梯度内核）。
- **对比**：
  - **REINFORCE vs Actor-Critic**：前者用实际 $G_t$（MC、无偏、高方差、需等回合），后者用 critic 估的 $V$（TD、有偏、低方差、可在线）。
  - **REINFORCE vs PPO**：PPO 加 ratio clipping + 多 epoch 复用样本 + GAE，更稳更省样本。
  - **策略梯度 vs Q-learning**：前者直接学 $\pi_\theta$，后者学 $Q$ 隐式派生 $\pi$。
  - **$\log\pi$ vs SFT 交叉熵**：同构，权重从 1 变 $G_t$。

## 7. 常见误区与易错点

1. **以为需要知道环境 $P$**：策略梯度定理证明不需要；只需能采样。
2. **用 $G_0$ 而非 $G_t$ 作每步权重**：丢掉因果性，方差变大（期望不变但不偏）。
3. **不减 baseline 直接训**：高方差，学习率必须极小，常不收敛；至少 batch 内归一化 returns。
4. **loss 符号反**：策略梯度是**上升** $J$，优化器做**下降** → loss 取负。漏负号越训越差。
5. **log_prob 被误 detach**：采样后要 `dist.log_prob(a)` 保留计算图，否则梯度断。
6. **离线样本直接用**：REINFORCE 是 on-policy，旧策略采的样本有分布偏移，需 importance reweight（PPO 的 ratio）。
7. **连续动作忘了取 log**：高斯策略 $\log\pi=\log\mathcal{N}(a;\mu,\sigma)$ 含 $\log\sigma$ 与 $(a-\mu)^2/2\sigma^2$ 项。
8. **LLM-RL 用裸 REINFORCE**：方差太大且没限制 KL，policy 会跑飞；必须 PPO 的 clipping + KL 控制。

## 8. 延伸细节

**score function estimator**：$\nabla\mathbb{E}_{x\sim p_\theta}[f(x)]=\mathbb{E}[f(x)\nabla\log p_\theta(x)]$ 是通用估计器，黑盒随机函数 $f$ 都能用（变分推断、离散 VAE）。REINFORCE 是它在 RL 的特例（$f=G$）。

**与 importance sampling**：off-policy 用旧策略 $\beta$ 采的样本训当前 $\pi_\theta$，需加权重 $\frac{\pi_\theta}{\beta}$。REINFORCE 是 $\beta=\pi_\theta$（同策略）ratio=1 的特例；PPO 限制 ratio 在 $[1-\epsilon,1+\epsilon]$ 防 off-policy 偏差爆炸。

**LLM-RL 的地位**：RLHF 的 PPO 内核就是"带 ratio clipping 与 KL 约束的 REINFORCE"。理解 $\sum_t\nabla\log\pi_\theta\cdot A_t$ 就理解了 LLM-RL 的梯度从哪来。

---
相关: [[Policy Gradient体系]]、[[return与reward]]、[[baseline]]、[[advantage function]]、[[variance reduction]]、[[Actor-Critic]]、[[PPO clipped objective]]、[[GAE]]
