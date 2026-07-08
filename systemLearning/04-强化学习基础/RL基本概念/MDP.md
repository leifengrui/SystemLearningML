# MDP

> **所属章节**: [[RL基本概念]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 基础（RL 第一定义）

## 1. 一句话定义

**MDP（Markov Decision Process，马尔可夫决策过程）** 是强化学习的**标准数学外壳**：用五元组 $(\mathcal{S},\mathcal{A},P,R,\gamma)$ 描述"智能体在环境里**按马尔可夫性**一步步决策"的过程——状态集 $\mathcal{S}$、动作集 $\mathcal{A}$、转移概率 $P(s'\mid s,a)$、奖励函数 $R(s,a)$（或 $R(s,a,s')$）、折扣因子 $\gamma\in[0,1)$。几乎所有 RL 算法（[[REINFORCE]]、[[Actor-Critic]]、[[PPO clipped objective]]、RLHF）都在 MDP 框架下定义和推导。
> [!note] 已展开为独立笔记
> "什么是 [[Actor-Critic]]" 已新建为专篇 [[Actor-Critic]]，归 [[04-强化学习基础]]/14. Actor-Critic 小节。要点速览：Actor-Critic = 策略梯度方法的两网络联合学习架构——actor（策略网络 $\pi_\theta$）决策、critic（价值网络 $V_\phi$）评估，critic 给 actor 提供低方差的 advantage 权重 $\hat A_t$（$G_t-V_\phi$ / TD 误差 / GAE），把 [[REINFORCE]] 的"用实际 return 当权重、无偏高方差"升级为"用 critic 估的 $V$ 减底色、有偏低方差、可在线"。它是 [[A2C]]/[[A3C]]/[[PPO clipped objective]]/RLHF-PPO 的统一骨架。完整推导（策略梯度定理 → advantage 形式 → 双时间尺度收敛 → GAE 标配）、可跑代码、与 REINFORCE/DQN/DDPG/SAC 对比、LLM-RL 映射表，见 [[Actor-Critic]]。
> [!note] 马尔可夫性是什么
> "未来只依赖当前状态，与历史无关"：$P(s_{t+1}\mid s_t,a_t,s_{t-1},a_{t-1},\dots)=P(s_{t+1}\mid s_t,a_t)$。它是 MDP 的"无记忆"前提——只要当前状态 $s_t$ 把所有历史信息压缩进去了，下一步转移就只看 $s_t,a_t$。如果环境有隐藏状态（如部分可观测 POMDP），马尔可夫性不成立，需要 belief state 或 RNN 编码历史（这正是 LLM 做 RL 时 prompt 上下文的作用）。

## 2. 为什么需要它（动机与背景）

监督学习有"数据集$(x,y)$+损失"的统一框架。RL 没有现成的 $(x,y)$，而是**序列决策**：智能体每步选动作、环境给奖励和新状态，目标是长期累积奖励最大。这比单步预测复杂得多——涉及时间、不确定性、延迟回报、探索。

MDP 把这套复杂性装进一个**精确定义的数学对象**：

- 用 $(S,A,P,R,\gamma)$ 五要素把"环境"形式化；
- 用**策略 $\pi$** 把"智能体"形式化；
- 用**回报/价值**把"目标"形式化；
- 用**贝尔曼方程**把"求解"形式化。

于是 RL 问题变成"在 MDP 中找最优策略 $\pi^*$ 最大化期望累积奖励"。所有算法都是对这一定义的不同求解途径（蒙特卡洛、时序差分、策略梯度）。没有 MDP，RL 就是一堆模糊的"试错"，无法严格推导、无法证明收敛、无法对比算法。

## 3. 核心概念详解

### 3.1 五要素逐个

| 要素 | 含义 | 离散/连续 | 例子 |
|---|---|---|---|
| $\mathcal{S}$ | 状态集（环境可能处的所有情况） | 离散或连续 | 棋盘局面、机器人关节角、LLM 已生成 token 序列 |
| $\mathcal{A}$ | 动作集（智能体可选手脚） | 离散或连续 | 上下左右、力矩、下一个 token |
| $P(s'\mid s,a)$ | 转移概率（在 $s$ 做 $a$，环境跳到 $s'$ 的概率） | - | 掷骰子移动、物理仿真 |
| $R(s,a)$（或 $R(s,a,s')$） | 奖励（这步动作的即时回报） | 标量 | 赢棋 +1、每步 -0.01、生成被偏好 +r |
| $\gamma\in[0,1)$ | 折扣因子（未来奖励的打折程度） | 标量 | 0.99：远期奖励权重每步 ×0.99 |

### 3.2 离散 vs 连续时间、有限 vs 无限回合

- **episodic**（有终止）：下棋到分胜负、一局游戏结束，长度 $T$ 有限。
- **continuing**（无终止）：机器人持续运行，$T\to\infty$，必须靠 $\gamma<1$ 让无穷级数收敛。
- $\gamma$ 越大越"远视"（接近 1 关注长期），越小越"短视"（接近 0 只看眼前）。

### 3.3 马尔可夫性带来的简化

因为未来只看 $(s_t,a_t)$，所以：
- 策略只需是 $\pi(a\mid s)$（只依赖当前状态），不必 $\pi(a\mid s_t,s_{t-1},\dots)$；
- 价值函数只需 $V^\pi(s)$（状态→期望累积奖励），不必依赖历史；
- 贝尔曼方程能递推（见 §4）。

这是 RL 整套动态规划可行性的根基。若不满足马尔可夫性，需升格到 POMDP（部分可观测），用 belief 或历史编码器近似。

### 3.4 期望与策略的引入

固定策略 $\pi(a\mid s)$ 后，MDP 变成**马尔可夫链**（转移被 $\pi$ 与 $P$ 联合决定），可求各种期望量。这引出 [[return与reward]] 的"期望累积奖励"目标和 [[policy与value function]] 的 $V^\pi$/$Q^\pi$。MDP 是壳，策略与价值是壳里要解的量。

### 3.5 与监督学习的对照

| 监督学习 | RL（MDP） |
|---|---|
| 数据 $(x,y)$ 静态 | 轨迹 $(s_0,a_0,r_0,s_1,\dots)$ 由交互产生 |
| 目标 $\min\mathbb{E}[\ell(f(x),y)]$ | 目标 $\max\mathbb{E}_\pi[\sum_t\gamma^t r_t]$ |
| 梯度对 $f$ 的参数 | 梯度对 $\pi$ 的参数（[[REINFORCE]]） |
| 数据分布固定 | 数据分布随策略变化（distribution shift） |

## 4. 数学原理 / 公式

### 4.1 轨迹与回报

一条轨迹 $\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\dots)$，折扣回报：

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}
$$

RL 目标：最大化期望回报 $J(\pi)=\mathbb{E}_{\tau\sim\pi}[G_0]$。

### 4.2 联合分布（轨迹概率）

在策略 $\pi$ 与转移 $P$ 下，一条轨迹的概率：

$$
P(\tau\mid\pi)=\rho_0(s_0)\prod_{t\ge0}\pi(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t)
$$

其中 $\rho_0$ 是初始状态分布。这是 [[REINFORCE]] 推导策略梯度的起点——对 $\theta$ 求导时只有 $\pi(a_t\mid s_t)$ 含 $\theta$，$P$ 不含 $\theta$，于是 $\nabla_\theta J$ 可绕过未知的 $P$。

### 4.3 贝尔曼方程（MDP 的核心递推）

状态价值：

$$
V^\pi(s)=\mathbb{E}_\pi\left[\sum_{k\ge0}\gamma^k r_{t+k+1}\mid s_t=s\right]=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\bigl[R(s,a,s')+\gamma V^\pi(s')\bigr]
$$

动作价值：

$$
Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)\bigl[R(s,a,s')+\gamma V^\pi(s')\bigr]=R(s,a)+\gamma\mathbb{E}_{s'}[V^\pi(s')]
$$

二者关系：$V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$，$Q^\pi(s,a)=R(s,a)+\gamma\mathbb{E}_{s'}[V^\pi(s')]$。这就是 [[policy与value function]] 要展开的核心。

### 4.4 贝尔曼最优方程与 $\pi^*$

$$
V^*(s)=\max_a\sum_{s'}P(s'\mid s,a)[R(s,a,s')+\gamma V^*(s)],\quad \pi^*(s)=\arg\max_a Q^*(s,a)
$$

这是"最优"的隐式定义；值迭代/策略迭代是求解它的动态规划算法。但 RL 中 $P,R$ 未知，只能靠采样估 $Q$/$V$，于是有 Q-learning、策略梯度等采样型算法。

## 5. 代码示例

```python
import numpy as np

# 一个最小 MDP：1D 网格 5 状态，左右移动，到右端 +1 终止
S = [0,1,2,3,4]                  # 状态集
A = ['L','R']                    # 动作集
gamma = 0.9
def P(s, a):                     # 转移：确定性，边界不动
    if a == 'L':  return max(0, s-1)
    else:         return min(4, s+1)
def R(s, a, s2):                 # 奖励：到 4 给 +1，其余 0
    return 1.0 if s2 == 4 and s != 4 else 0.0
def is_terminal(s): return s == 4

# 给定一个随机策略，估 V^pi（蒙特卡洛）
def pi(s): return np.random.choice(A)
def episode():
    s, traj = 0, []
    while not is_terminal(s):
        a = pi(s); s2 = P(s,a); r = R(s,a,s2)
        traj.append((s,a,r,s2)); s = s2
    return traj
# 计算每状态从出现到结束的折扣回报，平均
from collections import defaultdict
returns = defaultdict(list)
for _ in range(2000):
    tau = episode(); G = 0
    for s,a,r,s2 in reversed(tau):
        G = r + gamma*G; returns[s].append(G)
V = {s: np.mean(returns[s]) for s in S}
print("V^pi:", V)   # 越靠近 4 价值越高
```

## 6. 与其他知识点的关系

- **上游（依赖）**: 概率论（期望、马尔可夫链）、动态规划（贝尔曼递推）。
- **下游（应用）**: [[policy与value function]]（在 MDP 上定义策略/价值）、[[return与reward]]（目标量）、[[REINFORCE]]/[[Actor-Critic]]/[[PPO clipped objective]]（求解 MDP 的算法）、RLHF（把 LLM 微调建模成 MDP：state=prompt+已生成token，action=下一token，reward=奖励模型打分）。
- **对比 / 易混**:
  - **MDP vs POMDP**：前者状态全可观，后者只能看观测 $o_t$（部分可观测），需 belief/history 编码。LLM-RL 因 prompt 含完整生成历史，常近似为 MDP。
  - **MDP vs 多臂老虎机**：老虎机无状态转移（$|\mathcal{S}|=1$），是 MDP 的退化特例，只有探索-利用权衡。
  - **MDP vs 博弈（MDP→MG）**：多智能体时升格为马尔可夫博弈（Markov Game），引入对手策略。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **把奖励当目标本身**：RL 目标是**期望累积折扣回报** $J(\pi)$，不是单步奖励。贪心拿即時奖励的策略往往次优（如吃完眼前糖、错过更大后续回报）。
> 2. **$\gamma=1$ 用在 continuing 任务**：无穷级数可能发散，$V$ 无定义；continuing 必须 $\gamma<1$。episodic 可 $\gamma=1$。
> 3. **以为 $P,R$ 已知**：model-based 假设已知，model-free（绝大多数深度 RL、RLHF）不知道，靠采样估。算法推导里若用了 $P$ 说明是 model-based。
> 4. **忽视马尔可夫性**：状态设计丢信息（如只用当前帧而非帧差）→ 不马尔可夫 → 价值函数表达力不足、训练不稳。LLM-RL 里 state 必须含完整 prompt+已生成部分。
> 5. **混淆 $R(s,a)$ 与 $R(s,a,s')$**：前者只看动作，后者看结果；下棋"落子"与"落子后是否将军"奖励不同。推导时注意用的是哪种。
> 6. **以为 episodic 不需要 $\gamma$**：长 episode 不加 $\gamma$ 也会数值爆炸/信用分配难；实践中即使 episodic 也常用 $\gamma=0.99$。
> 7. **状态/动作空间混用连续离散**：连续动作下 $\arg\max_a$ 无解析解，要用策略梯度或归一化流；离散动作用 categorical 策略。

## 8. 延伸细节

### 8.1 LLM-RL 里的 MDP 映射

RLHF 把 LLM 微调建模为 MDP：
- $s_t$ = prompt + 已生成 token $[y_0,\dots,y_{t-1}]$；
- $a_t$ = 下一个 token $y_t$；
- $P$ = 确定性（生成 $a_t$ 后状态就是拼接，无随机转移）；
- $R$ = 仅在序列末尾由奖励模型给一次分（稀疏奖励）；
- $\gamma$ 常取 1（episodic）或 0.99。

这正是 [[PPO clipped objective]] 在 LLM 上应用的 MDP 背景。注意 $P$ 确定性使策略梯度推导简化（见 [[REINFORCE]]）。

### 8.2 折扣因子与有效视野

$\gamma$ 对应有效视野 $H\approx 1/(1-\gamma)$：$\gamma=0.99$ 时 $H\approx100$ 步。调 $\gamma$ 等于调"智能体考虑多远的未来"，是关键超参。

### 8.3 平均回报准则

continuing 任务除折扣回报外，还可优化**平均回报** $\lim_{T\to\infty}\frac1T\mathbb{E}[\sum r_t]$，适合无终止且不关心时间偏好的任务（如排队控制）。深度 RL 少用。

### 8.4 从 MDP 到部分可观测（POMDP）

现实多数是 POMDP（机器人传感器有噪、扑克看不见对手手牌）。理论需 belief state $b(s)=P(s\mid\text{history})$，实践用 RNN/Transformer 编码历史近似。LLM 自带的上下文窗口天然做了这件事。

### 8.5 贝尔曼方程的"自洽"

$V^\pi$ 同时出现在方程两边——它是**不动点**。值迭代反复套方程会收敛到不动点（压缩映射，模数 $\gamma$）。这是动态规划求解 MDP 的理论基础，也是 [[value network]] 时序差分学习的依据。

---
相关: [[RL基本概念]]、[[policy与value function]]、[[return与reward]]、[[REINFORCE]]、[[Actor-Critic]]、[[PPO clipped objective]]
