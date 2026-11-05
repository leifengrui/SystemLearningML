# MDP

> **所属章节**: [[RL基本概念]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**MDP（Markov Decision Process，马尔可夫决策过程）** 是强化学习的标准数学外壳：用五元组 $(\mathcal{S},\mathcal{A},P,R,\gamma)$ 描述"智能体在环境里按马尔可夫性一步步决策"的过程。几乎所有 RL 算法（[[REINFORCE]]、[[Actor-Critic]]、[[PPO clipped objective]]）都在 MDP 框架下定义。

> [!note] 马尔可夫性
> 未来只依赖当前状态：$P(s_{t+1}\mid s_t,a_t,\dots)=P(s_{t+1}\mid s_t,a_t)$。只要 $s_t$ 把历史信息压缩充分，下一步转移只看 $(s_t,a_t)$。不满足时升格为 POMDP（部分可观测），需 belief state 或 RNN 编码历史。

## 2. 为什么需要它

监督学习有"$(x,y)$+损失"的统一框架，RL 没有现成标签，而是序列决策：每步选动作、环境给奖励和新状态，目标是长期累积奖励。MDP 把这套复杂性装进精确定义的数学对象——五元组形式化环境、策略 $\pi$ 形式化智能体、回报/价值形式化目标、贝尔曼方程形式化求解。RL 问题变成"在 MDP 中找最优策略 $\pi^*$ 最大化期望累积奖励"。

## 3. 核心概念

### 3.1 五要素

| 要素 | 含义 | 例子 |
|---|---|---|
| $\mathcal{S}$ | 状态集 | 棋盘局面、机器人关节角、LLM 已生成 token 序列 |
| $\mathcal{A}$ | 动作集 | 上下左右、力矩、下一个 token |
| $P(s'\mid s,a)$ | 转移概率 | 掷骰子移动、物理仿真 |
| $R(s,a)$ 或 $R(s,a,s')$ | 即时奖励 | 赢棋 +1、每步 -0.01 |
| $\gamma\in[0,1)$ | 折扣因子 | 0.99：远期奖励每步 ×0.99 |

### 3.2 episodic vs continuing

- **episodic**（有终止）：长度 $T$ 有限，可取 $\gamma=1$。
- **continuing**（无终止）：$T\to\infty$，必须 $\gamma<1$ 让无穷级数收敛。
- $\gamma$ 越大越远视，越小越短视。有效视野 $H\approx 1/(1-\gamma)$。

### 3.3 与监督学习对照

| 监督学习 | RL（MDP） |
|---|---|
| 数据 $(x,y)$ 静态 | 轨迹 $(s_0,a_0,r_0,s_1,\dots)$ 由交互产生 |
| $\min\mathbb{E}[\ell(f(x),y)]$ | $\max\mathbb{E}_\pi[\sum_t\gamma^t r_t]$ |
| 数据分布固定 | 数据分布随策略变化（distribution shift） |

## 4. 数学原理

### 4.1 轨迹与回报

轨迹 $\tau=(s_0,a_0,r_0,s_1,\dots)$，折扣回报 $G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}$，目标 $J(\pi)=\mathbb{E}_{\tau\sim\pi}[G_0]$。

轨迹概率：$P(\tau\mid\pi)=\rho_0(s_0)\prod_t \pi(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$。求 $\nabla_\theta J$ 时只有 $\pi$ 含 $\theta$，可绕过未知的 $P$——这是策略梯度定理的起点。

### 4.2 贝尔曼方程（核心递推）

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)[R(s,a,s')+\gamma V^\pi(s')]$$

$$Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)[R(s,a,s')+\gamma V^\pi(s')]$$

关系：$V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$。

### 4.3 贝尔曼最优方程

$$V^*(s)=\max_a\sum_{s'}P(s'\mid s,a)[R(s,a,s')+\gamma V^*(s)],\quad \pi^*(s)=\arg\max_a Q^*(s,a)$$

值迭代/策略迭代是求解它的动态规划；RL 中 $P,R$ 未知，靠采样估 $Q/V$，于是有 Q-learning、策略梯度等。

## 5. 代码示例

```python
# 最小 MDP：5 状态 1D 网格，到右端 +1 终止
S = [0,1,2,3,4]; A = ['L','R']; gamma = 0.9
def P(s, a): return max(0, s-1) if a=='L' else min(4, s+1)
def R(s, a, s2): return 1.0 if s2==4 and s!=4 else 0.0
```

## 6. 与其他知识点的关系

- **上游**: 概率论（期望、马尔可夫链）、动态规划（贝尔曼递推）。
- **下游**: [[policy与value function]]、[[return与reward]]、[[REINFORCE]]/[[Actor-Critic]]/[[PPO clipped objective]]、RLHF（state=prompt+已生成token，action=下一token，reward=奖励模型打分）。
- **对比**: MDP（全可观）vs POMDP（部分可观测）；多臂老虎机是 $|\mathcal{S}|=1$ 的退化 MDP；多智能体升格为马尔可夫博弈。

## 7. 常见误区

1. 把奖励当目标本身——RL 目标是期望累积折扣回报，不是单步奖励。
2. continuing 任务用 $\gamma=1$ 会让级数发散。
3. 以为 $P,R$ 已知——model-free（深度 RL、RLHF）不知道，靠采样估。
4. 忽视马尔可夫性：状态设计丢信息 → 价值函数表达力不足、训练不稳。
5. 混淆 $R(s,a)$ 与 $R(s,a,s')$：前者只看动作，后者看结果。
6. 连续动作下 $\arg\max_a$ 无解析解，要用策略梯度或归一化流。

---
相关: [[RL基本概念]]、[[policy与value function]]、[[return与reward]]
