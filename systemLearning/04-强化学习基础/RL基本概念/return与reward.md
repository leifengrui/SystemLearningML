# return与reward

> **所属章节**: [[RL基本概念]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**reward（奖励）** $r_t$ 是环境在每步给智能体的即时标量反馈；**return（回报）** $G_t$ 是从时刻 $t$ 起到结束的累积折扣奖励 $G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}$，是 RL 真正要最大化的长期目标量。reward 是"每步的小分"，return 是"从现在起一路累加的总分"。RL 优化的永远是**期望 return**，不是单步 reward。

> [!note] 一句话区分
> - reward $r_t$：单步、即时、环境给的标量。
> - return $G_t$：长期、累积、带折扣、从 $t$ 往后求和。
> - value $V^\pi(s)=\mathbb{E}[G_t\mid s]$：return 的期望（见 [[policy与value function]]）。

## 2. 为什么需要它

监督学习每条样本有独立标签，损失是单步的。RL 不同：

- **延迟回报**：一局棋到结束才知道输赢，中间几十步没有 reward；要靠 return 把结束奖励"倒推"分给每步（信用分配）。
- **长期 vs 即时冲突**：贪心拿即时 reward 常次优，必须用 return 衡量长期影响。
- **无穷期收敛**：continuing 任务若不折扣 $\sum r_t$ 可能发散，必须 $\gamma<1$。

return 把"长期累积"形式化，reward 是它的原子。整个 RL 目标 $J(\pi)=\mathbb{E}_{\tau\sim\pi}[\sum_t\gamma^t r_t]$ 就是期望 return。所有梯度（[[REINFORCE]]）、所有价值（$V/Q$）、所有优势（$A=Q-V$）都围绕 return 展开。

## 3. 核心概念

### 3.1 reward 的几种约定

| 记号 | 含义 | 何时用 |
|---|---|---|
| $r_t$ | 第 $t$ 步收到的标量 | 通用 |
| $R(s,a)$ | 在 $s$ 做 $a$ 的期望即时 reward | [[MDP]] 五元组 |
| $R(s,a,s')$ | 转移到 $s'$ 时的 reward | 稀疏/结果相关奖励 |
| $R(s)$ | 只依赖状态 | 盘面评估类 |

### 3.2 return 的三种形式

| 名称 | 公式 | 适用 |
|---|---|---|
| 折扣回报（最常用） | $G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}$ | continuing + episodic |
| 有限视野 | $G_t^{(H)}=\sum_{k=0}^{H-1}\gamma^k r_{t+k+1}$ | MC 截断 |
| 平均回报 | $\bar G=\lim_{T\to\infty}\frac1T\sum r_t$ | continuing 无时间偏好 |

### 3.3 折扣因子 $\gamma$

- $\gamma\to1$：远视，收敛慢、方差大、数值易爆。
- $\gamma\to0$：近视，只看几步。
- 有效视野 $H\approx 1/(1-\gamma)$：$\gamma=0.99\Rightarrow H\approx100$ 步。

### 3.4 return 的递推（核心技巧）

$$\boxed{G_t=r_{t+1}+\gamma G_{t+1}}$$

当前 return = 即时 reward + 折扣×下一步 return。这是 RL 递归算法的基石——TD 学习（$V(s_t)\leftarrow r+\gamma V(s_{t+1})$）、GAE、策略梯度反向信用分配都用它。

### 3.5 信用分配

长轨迹里只有末尾有 reward（如下棋赢 +1），return 通过递推倒流给前面每步：

$$G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}\quad\Rightarrow\quad\text{越靠近末尾 }G_t\text{越大（}\gamma^k\text{打折少）}$$

[[GAE]] 进一步用偏差归一化做更精细的信用分配。

### 3.6 reward shaping

设计不当会诱发 reward hacking（钻空子拿高分但不达任务目标，见 [[reward hacking]]）。**势能函数法** $F(s,s')=\gamma\Phi(s')-\Phi(s)$ 是已证明不改变最优策略的 shaping 技巧。

## 4. 数学原理

### 4.1 期望 return = RL 目标

$$J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_{t\ge0}\gamma^t r_{t+1}\right]=\mathbb{E}_{\tau\sim\pi}[G_0]$$

策略梯度 $\nabla_\theta J=\mathbb{E}[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t]$ 就是"每步 log 策略导数×该步 return"。

### 4.2 方差与 baseline

$G_t$ 是随机变量，直接做策略梯度方差极大。减去与动作无关的 baseline $b(s_t)$（通常取 $V^\pi(s_t)$）不改变期望但降方差：

$$\nabla_\theta J=\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)(G_t-b(s_t))\right]$$

$b=V^\pi$ 时 $G_t-V^\pi\approx A^\pi$（优势）。

### 4.3 价值与 return

$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]$$

价值 = return 的期望，是 return 的"去噪版"。

### 4.4 收敛性

$\gamma<1$ 且 $|r_t|\le r_{\max}$ 时，$|G_t|\le r_{\max}/(1-\gamma)$，级数绝对收敛。continuing 任务 return 有定义的前提。

## 5. 代码示例

```python
# return 的递推计算：仅末尾 +1（如下棋赢）
rewards = [0.0, 0.0, 0.0, 0.0, 1.0]
gamma = 0.99
G = 0.0; returns = [0.0]*len(rewards)
for t in reversed(range(len(rewards))):
    G = rewards[t] + gamma*G
    returns[t] = G
# returns: [0.9606, 0.9703, 0.9801, 0.99, 1.0]  越靠后越大
```

## 6. 与其他知识点的关系

- **上游**: [[MDP]]（reward 是五元组之一）、概率论（期望、级数）。
- **下游**: [[policy与value function]]（$V^\pi=\mathbb{E}[G_t]$）、[[REINFORCE]]、[[baseline]]、[[advantage function]]、[[GAE]]、[[reward normalization]]、[[reward hacking]]。
- **对比**: reward（单步）vs return（累积）；return（单条轨迹实际量）vs value（其期望）；discounted return（有时间偏好）vs average reward（无）；extrinsic reward（环境给）vs intrinsic reward（自造，用于探索）。

## 7. 常见误区

1. 优化 reward 而非 return——贪心每步 reward 常次优。
2. $G_t$ 索引错位：常见约定 $G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}$（reward 在转移后），也有教材用 $r_t$；读公式先对齐索引。
3. continuing 用 $\gamma=1$ 会让 return 发散。
4. 把 reward 当 loss——它是标量信号不是可微损失，要靠策略梯度或 Q-learning 转成梯度。
5. reward 尺度不归一——过大梯度爆炸，过小学不动，常做 running std 归一化。
6. 稀疏 reward 不做信用分配——直接用 $G_t$ 方差巨大，用 [[GAE]]/advantage。
7. 随手加 shaping reward 可能让智能体钻空子；用势能函数法保证不改最优策略。
8. 终止状态 $G_T=0$（无后续），递推从 $G_T=0$ 倒推。

## 8. 延伸

### 8.1 LLM-RL 的 reward 形态

RLHF 中奖励模型 $r_\phi(\text{prompt},\text{response})$ 对整段回答打一个标量，仅在序列末尾给。等效 reward 序列：$r_1=\dots=r_{T-1}=0$，$r_T=r_\phi$。$\gamma=1$ 时所有 token 的 $G_t$ 相同。信用分配退化为"把一个序列级分数分到每个 token"——靠 advantage estimation 与 KL 约束完成，是 [[PPO clipped objective]] 在 LLM 上的核心难点。

### 8.2 generalized return（n-step 与 λ-return）

$G_t$ 与 $V^\pi$ 的插值：n-step return $G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V(s_{t+n})$，λ-return 在 n 上做指数加权。GAE 是其 advantage 版本（见 [[GAE]]）。

### 8.3 reward 有界性

实践中常 `reward = tanh(原始reward)` 或 clip 到 $[-c,c]$，防极端值引爆梯度，配合 [[reward normalization]] 的 running std。

---
相关: [[RL基本概念]]、[[MDP]]、[[policy与value function]]
