# REINFORCE

> **所属章节**: [[Policy Gradient体系]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 中等（策略梯度第一课）

## 1. 一句话定义

**REINFORCE**（也叫 vanilla policy gradient，最简策略梯度算法）是直接对**期望回报** $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0]$ 求梯度上升的最简策略梯度方法：梯度 $\nabla_\theta J=\mathbb{E}[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t]$，用蒙特卡洛样本估期望——即"**每步的对数策略导数 × 该步的 return**"求平均作为梯度，对 $\theta$ 走一步。它无 critic、无偏、但方差极大，是 [[Actor-Critic]]/[[PPO clipped objective]] 的理论起点。

> [!note] 名字来源
> "REINFORCE" 由 Williams 1992 提出（REward Increment = nonnegative Factor × Offset × Reinforcement × Characterization × Equity）。本质就是"用实际 return 作权重、对 log 策略求导"的蒙特卡洛策略梯度。它还有别名 MC policy gradient、vanilla PG。

## 2. 为什么需要它（动机与背景）

[[MDP]]/[[policy与value function]] 定义了 RL 目标 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0]$，但没说怎么算梯度。难点在于：轨迹分布 $P(\tau\mid\theta)$ 含环境转移 $P$（未知），似乎没法对 $\theta$ 求导。

REINFORCE 的核心突破是**策略梯度定理**：求导时 $P$ 不含 $\theta$ 会消失，只剩对 $\log\pi_\theta$ 的求导，于是**不需要知道环境模型 $P$** 也能算梯度——只要能采样轨迹。这把 RL 从"需要环境模型"解放到"只需交互采样"，是 model-free 深度 RL 的起点。

但 REINFORCE 用完整轨迹的 $G_t$ 作权重，方差极大（一条轨迹的随机性全进了梯度），实际几乎不直接用，但所有现代策略梯度方法（[[Actor-Critic]]、[[PPO clipped objective]]、A3C）都是它的减方差+稳定化改进。理解 REINFORCE = 理解策略梯度的骨架。

## 3. 核心概念详解

### 3.1 目标与梯度

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0]=\int P(\tau\mid\theta)G_0\,d\tau
$$

$$
\nabla_\theta J=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right]
$$

### 3.2 为什么是 $\log\pi$ 而不是 $\pi$

求导 $\nabla P(\tau\mid\theta)=P(\tau\mid\theta)\nabla\log P(\tau\mid\theta)$（对数恒等式 / score function trick）。$P(\tau\mid\theta)=\rho_0\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$，取 $\log$ 后 $\prod$ 变 $\sum$，而 $\log P(s_{t+1}\mid s_t,a_t)$ 不含 $\theta$ 求导为 0，只剩 $\sum_t\nabla\log\pi_\theta(a_t\mid s_t)$。这就是 $\log\pi$ 的来源——它是把"乘积分布"求导变成"可加导数"的数学技巧。
> [!note] 解答：为什么策略梯度用 $\log\pi$ 而不是直接用 $\pi$——五大好处
> 批注问的是 §3.2 取对数的好处。直接对轨迹概率 $P(\tau\mid\theta)=\rho_0\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$ 求导理论上也行，但取 $\log$ 后求导有五大工程与数学收益：
> 
> **好处 1：连乘变连加，求导项数从爆炸变线性。** $P(\tau\mid\theta)$ 是 $T$ 个因子的连乘，直接对 $\theta$ 求导要用乘积法则，产生 $O(2^T)$ 级别的交叉项。取 $\log$ 后 $\log P(\tau\mid\theta)=\log\rho_0+\sum_t[\log\pi_\theta(a_t\mid s_t)+\log P(s_{t+1}\mid s_t,a_t)]$，连乘变连加，求导只剩 $\sum_t\nabla\log\pi_\theta$，项数线性 $O(T)$，可算。
> 
> **好处 2：干净分离策略与环境 → model-free 的数学根基。** $\log P$ 拆成"策略项 $\sum\log\pi$"+"环境项 $\sum\log P$"+"初始 $\log\rho_0$"。对 $\theta$ 求导时后两项不含 $\theta$ 直接消失，只剩 $\sum_t\nabla\log\pi_\theta$。这就是 REINFORCE 不需要知道环境模型 $P$ 也能算梯度的根因——直接对 $P(\tau\mid\theta)$ 求导无法这么干净地分离 $P$，model-free 站不住。
> 
> **好处 3：数值防下溢，长序列必须。** 长轨迹下 $\prod_t\pi_\theta(a_t\mid s_t)$ 是很多小概率连乘，浮点下很快下溢到 0（尤其 fp16/bf16）。取 $\log$ 后变 $\sum_t\log\pi$，是负数累加（量级 $O(T)$），不下溢。LLM-RL 序列上千 token，不用 $\log$ 直接算 $\prod$ 会得到 0 梯度，根本训不动。
> 
> **好处 4：梯度量级归一化，训练稳。** $\nabla\log\pi_\theta=\nabla\pi_\theta/\pi_\theta$，分母 $\pi$ 把"小概率动作梯度极小、大概率动作梯度大"的不均衡归一化掉，所有动作的 score 量级一致。直接用 $\nabla\pi$ 会让小概率动作几乎学不到（梯度被 $\pi$ 压没），训练偏向大概率动作。
> 
> **好处 5：与监督学习交叉熵同构，工程可复用。** $\nabla\log\pi_\theta(a_t\mid s_t)$ 正是"让动作 $a_t$ 概率增大"的方向，与 SFT 的交叉熵梯度 $\nabla\log p(y_t\mid x,y_{<t})$ 同形。RLHF 把 LLM 当 $\pi_\theta$ 直接套策略梯度，框架代码可复用，只是权重从 1 换成 $G_t$/$\hat A_t$（见 [[Actor-Critic]] §3.2）。若用 $\nabla\pi$ 而非 $\nabla\log\pi$，与 SFT loss 不同构，复用困难。
> 
> **对比小结**：
> 
> | 维度 | 直接用 $\pi$ / $\nabla\pi$ | 用 $\log\pi$ / $\nabla\log\pi$ |
> |---|---|---|
> | 求导项数 | 乘积法则 $O(2^T)$ 爆炸 | 连加 $O(T)$ 线性 |
> | 环境分离 | 难，$P$ 混在乘积里 | 干净，$\log P$ 求导为 0 |
> | 数值稳定 | 长序列下溢 0 | 负数累加不下溢 |
> | 梯度量级 | 随 $\pi$ 大小变 | $\nabla\pi/\pi$ 归一化 |
> | 与 SFT 同构 | 否 | 是（交叉熵同形） |
> 
> **误区**：取 $\log$ 不是"让 $\pi$ 变大"的技巧，而是把"对乘积分布求导"转成"对可加对数求导"的数学等价变换（$\nabla P=P\nabla\log P$ 是恒等式，不改变数学，只改变可计算性）。$\log\pi$ 也不改变策略——$\pi$ 与 $\log\pi$ 单调对应，最大化 $\mathbb{E}[G]$ 的 $\theta^*$ 不变；变的只是梯度的**形式**与**数值可行性**。
> 
> 详见本篇 §4.1 策略梯度定理推导、[[policy与value function]] §4.1、[[Actor-Critic]] §4.2 advantage 形式、[[baseline]] §4.1 无偏性证明（都用 $\nabla\log\pi$ 这一套）。

### 3.3 直觉：增大好轨迹的动作概率

梯度每项 $\nabla\log\pi_\theta(a_t\mid s_t)\,G_t$：
- $G_t>0$（这步最终带来正回报）→ 沿 $\nabla\log\pi$ 方向更新 → 增大 $\pi_\theta(a_t\mid s_t)$；
- $G_t<0$ → 反向 → 减小该动作概率。

即"**回报为正的动作多采样、为负的少采样**"，正比于 return。这与监督学习的交叉熵梯度同构——$\nabla\log\pi$ 正是"让某动作更像目标"的方向，只是这里"目标"是按 return 加权的采样动作。

### 3.4 与监督学习/语言建模的同构

LLM 的 $\pi_\theta(\text{token}\mid\text{ctx})$ 与 SFT 的交叉熵 $\nabla\log\pi_\theta(y_t\mid x,y_{<t})$ 几乎同形。区别：SFT 的"权重"恒为 1（每条 demo 都学），REINFORCE 的权重是 $G_t$（按回报加权）。这是 RLHF 能直接复用 LLM 训练框架的根因。

### 3.5 蒙特卡洛估计（实际算法）

REINFORCE 实际流程：
1. 用当前 $\pi_\theta$ 采 $N$ 条轨迹；
2. 每条轨迹递推算每步 $G_t$；
3. 梯度 $\hat g=\frac1N\sum_\tau\sum_t\nabla\log\pi_\theta(a_t\mid s_t)G_t$；
4. $\theta\leftarrow\theta+\eta\hat g$（或丢给优化器）。

### 3.6 缺陷：高方差

$G_t$ 是单条轨迹的实际回报，随机性极大。同一状态不同轨迹 $G_t$ 差几个数量级 → 梯度方差爆炸 → 学习率必须极小 → 收敛慢。这正是 [[baseline]]、[[Actor-Critic]]、[[GAE]] 存在的理由。

## 4. 数学原理 / 公式

### 4.1 策略梯度定理推导

$$
\begin{aligned}
\nabla_\theta J(\theta)&=\nabla_\theta\int P(\tau\mid\theta)G_0\,d\tau=\int \nabla_\theta P(\tau\mid\theta)G_0\,d\tau\\
&=\int P(\tau\mid\theta)\nabla_\theta\log P(\tau\mid\theta)G_0\,d\tau\\
&=\mathbb{E}_{\tau\sim\pi_\theta}\bigl[G_0\,\nabla_\theta\log P(\tau\mid\theta)\bigr]
\end{aligned}
$$

而 $\log P(\tau\mid\theta)=\log\rho_0(s_0)+\sum_t[\log\pi_\theta(a_t\mid s_t)+\log P(s_{t+1}\mid s_t,a_t)]$，对 $\theta$ 求导：

$$
\nabla_\theta\log P(\tau\mid\theta)=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

（环境项 $\log P$ 与 $\log\rho_0$ 不含 $\theta$ 消失）。代入得：

$$
\boxed{\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right]}
$$

其中 $G_t$ 而非 $G_0$ 是因"因果性"——$a_t$ 只影响 $t$ 之后的回报，不影响之前（详见 [[return与reward]] §3.4 与 [[causal/错分代]]，去掉 $G_0$ 里 $a_t$ 之前的部分不改变期望但降方差）。

### 4.2 baseline 不变性

对任何只依赖 $s_t$（不依赖 $a_t$）的 $b(s_t)$：

$$
\mathbb{E}[\nabla\log\pi_\theta(a_t\mid s_t)\,b(s_t)]=b(s_t)\sum_a\pi_\theta(a\mid s_t)\nabla\log\pi_\theta(a\mid s_t)=b(s_t)\sum_a\nabla\pi_\theta(a\mid s_t)=b(s_t)\nabla 1=0
$$

故减 baseline 不改期望梯度，但若 $b\approx V^\pi$ 能大幅降方差（[[baseline]]）。$b=V^\pi$ 时 $G_t-b\approx A^\pi$（[[advantage function]]）。

### 4.3 因果性（去掉 $a_t$ 之前的 reward）

$\nabla\log\pi_\theta(a_t\mid s_t)$ 与 $r_{t'},t'<t$ 无关（$a_t$ 还没发生），故 $\mathbb{E}[\nabla\log\pi_\theta(a_t\mid s_t)\sum_{t'<t}\gamma^{t'}r_{t'}]=0$，可从 $G_0$ 去掉这些项得 $G_t$，期望不变方差降。

## 5. 代码示例

```python
import torch, torch.nn as nn

# 最小 REINFORCE：CartPole 风格（离散动作）
class Policy(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,n_act))
    def forward(self, s):
        return torch.distributions.Categorical(logits=self.net(s))  # pi(a|s)

pi = Policy(4, 2)
opt = torch.optim.Adam(pi.parameters(), lr=1e-2)
gamma = 0.99

def rollout(env_pi, max_steps=200):
    # 假环境：返回 (s, a, r, s', done)
    s, traj = torch.randn(4), []
    for _ in range(max_steps):
        d = env_pi(s); a = d.sample(); logp = d.log_prob(a)
        r = float(a.item()==1)  # 假奖励：选动作1给小奖
        traj.append((s, a, r, logp))
        s = torch.randn(4)
    return traj

for episode in range(100):
    traj = rollout(pi)
    # 1) 从后往前递推 G_t
    G = 0; returns = [0]*len(traj)
    for t in reversed(range(len(traj))):
        G = traj[t][2] + gamma*G; returns[t] = G
    returns = torch.tensor(returns)
    # 2) 组装策略梯度: sum_t logp_t * G_t  (取负因为优化器做梯度下降)
    logps = torch.stack([t[3] for t in traj])
    loss = -(logps * returns).sum()      # 负号：梯度下降实现梯度上升
    opt.zero_grad(); loss.backward(); opt.step()
    if episode % 20 == 0:
        print(f"ep{episode} loss={loss.item():.2f} G0={returns[0].item():.2f}")
```

> [!tip] 实际不会裸用
> 上面 `returns` 不减均值（baseline），方差大。生产代码至少做 `returns = (returns - returns.mean())/(returns.std()+1e-8)`（每 batch 归一化）减方差，再上 [[Actor-Critic]]。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[MDP]]、[[policy与value function]]、[[return与reward]]（$G_t$）、概率论（score function estimator）。
- **下游（应用）**: [[baseline]]（减 $V^\pi$）、[[advantage function]]（换 $G-V=A$）、[[variance reduction]]（系统减方差）、[[Actor-Critic]]（用 critic 估 $G$）、[[PPO clipped objective]]（限制更新幅度）、RLHF（LLM-RL 的策略梯度内核）。
- **对比 / 易混**:
  - **REINFORCE vs Actor-Critic**：前者用实际 $G_t$（MC、无偏、高方差、需等回合结束），后者用 critic 估的 $V$（TD、有偏、低方差、可在线）。
  - **REINFORCE vs PPO**：PPO 在 REINFORCE 基础上加 ratio clipping + 多 epoch 复用样本 + GAE，更稳更省样本。
  - **策略梯度 vs Q-learning**：前者直接学 $\pi_\theta$，后者学 $Q$ 隐式派生 $\pi$。
  - **REINFORCE 的 $\log\pi$ vs 监督学习交叉熵**：同构，权重从 1 变 $G_t$。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为需要知道环境 $P$**：策略梯度定理证明不需要；只需能采样。
> 2. **用 $G_0$ 而非 $G_t$ 作每步权重**：丢掉因果性，方差变大且无期望收益（虽然不偏）。
> 3. **不减 baseline 直接训**：高方差，学习率必须极小，常不收敛；至少 batch 内归一化 returns。
> 4. **loss 符号反**：策略梯度是**上升** $J$，优化器做**下降** → loss 取负。漏负号会越训越差。
> 5. **log_prob 没 retain 计算图**：采样后要 `dist.log_prob(a)` 而非 `dist.log_prob(a).detach()`，否则梯度断。
> 6. **离线样本直接用**：REINFORCE 是 on-policy，旧策略采的样本有分布偏移，需重要性重加权（PPO 的 ratio）。
> 7. **连续动作忘了取 log**：高斯策略要 $\log\pi=\log\mathcal{N}(a;\mu,\sigma)$，含 $\log\sigma$ 与 $(a-\mu)^2/2\sigma^2$ 项。
> 8. **混淆 $\pi(a\mid s)$ 与 $\pi(s)$**：策略梯度要条件分布 $\pi(a\mid s)$，log 是对其取。
> 9. **LLM-RL 用裸 REINFORCE**：方差太大、且没限制 KL，policy 会跑飞；必须 PPO 的 clipping + KL 控制。

## 8. 延伸细节

### 8.1 score function estimator 的更广意义

$\nabla\mathbb{E}_{x\sim p_\theta}[f(x)]=\mathbb{E}[f(x)\nabla\log p_\theta(x)]$ 是通用 **score function / REINFORCE estimator**，黑盒随机函数 $f$ 都能用它估梯度（变分推断、离散 VAE、可微采样）。REINFORCE 是它在 RL 的特例（$f=G$）。

### 8.2 与 importance sampling 的关系

off-policy 用旧策略 $\beta$ 采的样本训当前 $\pi_\theta$，需加权重 $\frac{\pi_\theta(a\mid s)}{\beta(a\mid s)}$（importance ratio）。REINFORCE 是 $\beta=\pi_\theta$（同策略）ratio=1 的特例。PPO 限制 ratio 在 $[1-\epsilon,1+\epsilon]$ 防 off-policy 偏差爆炸。

### 8.3 因果性与分时代理

更细的因果处理：$a_t$ 只影响 $t$ 及之后的 reward，梯度可写成 $\sum_t\nabla\log\pi_\theta(a_t\mid s_t)\sum_{t'\ge t}\gamma^{t'-t}r_{t'}$，这正是 $G_t$。再进一步的 future-aware 版本见 advantage estimation。

### 8.4 REINFORCE 的收敛性

带递减学习率 $\eta_t\to0,\sum\eta_t=\infty,\sum\eta_t^2<\infty$ 时，REINFORCE 几乎处处收敛到局部最优（随机逼近理论）。但实际用 Adam 固定 lr，需靠 baseline/归一化稳定。

### 8.5 在 LLM-RL 的地位

RLHF 的 PPO 内核就是"带 ratio clipping 与 KL 约束的 REINFORCE"。理解 REINFORCE 的 $\sum_t\nabla\log\pi_\theta\cdot A_t$ 公式，就理解了 LLM-RL 的梯度从哪来；PPO 的改进都围绕"如何让这个梯度稳定可控"。

---
相关: [[Policy Gradient体系]]、[[return与reward]]、[[baseline]]、[[advantage function]]、[[variance reduction]]、[[Actor-Critic]]、[[PPO clipped objective]]、[[GAE]]、[[KL penalty与KL control]]、[[entropy bonus]]、[[policy与value function]]
