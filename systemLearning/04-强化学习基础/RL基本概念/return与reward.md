# return与reward

> **所属章节**: [[RL基本概念]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 基础（区分清楚才看得懂 RL 公式）

## 1. 一句话定义

**reward（奖励）** $r_t$ 是环境在每步给智能体的**即时标量反馈**（"这步动作好不好"）；**return（回报）** $G_t$ 是从时刻 $t$ 起到结束的**累积折扣奖励** $G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}$，是 RL 真正要最大化的长期目标量。reward 是"每步的小分"，return 是"从现在起一路累加的总分"，二者被 $\gamma$ 和时间累加区分——RL 优化的永远是**期望 return**，不是单步 reward。

> [!note] 一句话区分
> - reward $r_t$：单步、即时、环境给的标量。
> - return $G_t$：长期、累积、带折扣、从 $t$ 往后求和。
> - value $V^\pi(s)=\mathbb{E}[G_t\mid s]$：return 的期望（见 [[policy与value function]]）。
> 训练目标 $J(\pi)=\mathbb{E}[G_0]$，是 return 的期望，不是 reward 之和的期望那么简单——折扣与无穷级数收敛都要管。

## 2. 为什么需要它（动机与背景）

监督学习每条样本有独立标签，损失是单步的。RL 不同：

- **延迟回报**：一局棋到结束才知道输赢，中间几十步都没有 reward；要靠 return 把结束的奖励"倒推"分给每步动作（信用分配）。
- **长期 vs 即时冲突**：贪心拿即时 reward 常常次优（吃眼前糖错过后续大奖励），必须用 return 衡量"这步对长期的影响"。
- **无穷期收敛**：continuing 任务无穷步，若不折扣 $\sum r_t$ 可能发散，必须 $\gamma<1$ 让 return 有限。

return 把"长期累积"形式化，reward 是它的原子。整个 RL 的目标 $J(\pi)=\mathbb{E}_{\tau\sim\pi}[\sum_t\gamma^t r_t]$ 就是**期望 return**。所有梯度（[[REINFORCE]]）、所有价值（$V/Q$）、所有优势（$A=Q-V$）都围绕 return 展开。搞混 reward 与 return 是看不懂 RL 公式的头号障碍。

## 3. 核心概念详解

### 3.1 reward 的几种定义约定

| 记号 | 含义 | 何时用 |
|---|---|---|
| $r_t$ | 第 $t$ 步收到的 reward（标量） | 通用 |
| $R(s,a)$ | 在 $s$ 做 $a$ 的期望即时 reward | [[MDP]] 五元组 |
| $R(s,a,s')$ | 转移到 $s'$ 时的 reward（含结果） | 稀疏/结果相关奖励 |
| $R(s)$ | 只依赖状态的 reward | 盘面评估类 |

$r_t=R(s_t,a_t,s_{t+1})$（或期望形式）的样本实现。

### 3.2 return 的三种常见形式

| 名称 | 公式 | 适用 |
|---|---|---|
| **折扣回报**（最常用） | $G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}$ | continuing + episodic |
| **有限视野** | $G_t^{(H)}=\sum_{k=0}^{H-1}\gamma^k r_{t+k+1}$ | 蒙特卡洛截断、有限 horizon |
| **平均回报** | $\bar G=\lim_{T\to\infty}\frac1T\sum_{t=0}^{T-1}r_t$ | continuing 无时间偏好 |

### 3.3 折扣因子 $\gamma$ 的作用

$$
G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\dots
$$

- $\gamma\to1$：远视，远期 reward 几乎不打折，但收敛慢、方差大、数值易爆。
- $\gamma\to0$：近视，只看几步。
- 有效视野 $H\approx 1/(1-\gamma)$：$\gamma=0.99\Rightarrow H\approx100$ 步。

### 3.4 return 的递推（核心技巧）

$$
\boxed{G_t=r_{t+1}+\gamma G_{t+1}}
$$

即"当前 return = 即时 reward + 折扣×下一步 return"。这个递推是 RL 递归算法的基石——TD 学习用它（$V(s_t)\leftarrow r+\gamma V(s_{t+1})$）、GAE 用它（见 [[GAE]]）、策略梯度的反向信用分配用它（从轨迹末尾倒推累加）。

### 3.5 信用分配（credit assignment）

长轨迹里只有末尾有 reward（如下棋赢 +1），中间 reward=0。return 把末尾 reward 通过递推"倒流"给前面每步：

$$
G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}\quad\Rightarrow\quad \text{越靠近末尾的步，}G_t\text{越大（被 } \gamma^k \text{打折少）}
$$

靠前步的 $G_t$ 因 $\gamma^k$ 折扣较小，但仍非零——这就是"奖励倒推分配"。[[GAE]] 进一步用偏差归一化做更精细的信用分配。

### 3.6 reward shaping 与潜在陷阱

人工设计 reward 影响策略行为，设计不当会诱发"reward hacking"（智能体钻空子拿高分但不达任务目标，见 [[reward hacking]]）。加**潜在基础奖励** $F(s,s')=\gamma\Phi(s')-\Phi(s)$ 是已证明不改变最优策略的 shaping 技巧（势能函数法）。

## 4. 数学原理 / 公式

### 4.1 期望 return = RL 目标

$$
J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_{t=0}^{\infty}\gamma^t r_{t+1}\right]=\mathbb{E}_{\tau\sim\pi}[G_0]
$$

策略梯度 $\nabla_\theta J(\theta)=\mathbb{E}[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t]$（[[REINFORCE]]）就是"对每步 log 策略导数乘以该步 return"。

### 4.2 return 的方差与 baseline

$G_t$ 是随机变量（轨迹随机），直接用 $G_t$ 做策略梯度方差极大（见 [[variance reduction]]）。减去与动作无关的 baseline $b(s_t)$（通常取 $V^\pi(s_t)$）不改变期望但降方差：

$$
\nabla_\theta J=\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)(G_t-b(s_t))\right]
$$

$b=V^\pi$ 时 $G_t-V^\pi\approx A^\pi$（优势），见 [[advantage function]]/[[baseline]]。

### 4.3 折扣 return 的期望与 $V^\pi$

$$
V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]
$$

即"价值=return 的期望"。这是 value function 的定义，也说明 value 是 return 的"去噪版"。

### 4.4 收敛性

$\gamma<1$ 且 reward 有界 $|r_t|\le r_{\max}$ 时，$|G_t|\le r_{\max}/(1-\gamma)$，级数绝对收敛。这是 continuing 任务 return 有定义的前提。

## 5. 代码示例

```python
import numpy as np

# 一条轨迹的 reward 序列，演示 return 的递推计算
rewards = [0.0, 0.0, 0.0, 0.0, 1.0]   # 仅末尾 +1（如下棋赢）
gamma = 0.99

# 方法1：从后往前递推 G_t = r_{t+1} + gamma * G_{t+1}
G = 0.0
returns = [0.0]*len(rewards)
for t in reversed(range(len(rewards))):
    G = rewards[t] + gamma*G
    returns[t] = G
print("returns G_t:", returns)
# [0.96059601, 0.970299, 0.9801, 0.99, 1.0]  越靠后越大

# 方法2：直接按定义累加（演示用，O(n^2)）
def return_at(t):
    return sum(gamma**k * rewards[t+k] for k in range(len(rewards)-t))
print("check:", [return_at(t) for t in range(len(rewards))])

# 期望 return 的蒙特卡洛估计：跑很多条轨迹平均 G_0
def rollout():
    # 假装环境：随机走 5 步，末尾按到达 4 与否给 1/0
    s = 0; rs = []
    for _ in range(5):
        a = np.random.choice([0,1])  # 随机策略
        s2 = max(0, min(4, s + (1 if a else -1)))
        rs.append(1.0 if s2 == 4 else 0.0); s = s2
    return rs
Gs = []
for _ in range(2000):
    rs = rollout(); G = 0
    for r in reversed(rs): G = r + gamma*G
    Gs.append(G)
print("E[G_0] ≈", np.mean(Gs))
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[MDP]]（reward 是其五元组之一）、概率论（期望、级数）。
- **下游（应用）**: [[policy与value function]]（$V^\pi=\mathbb{E}[G_t]$）、[[REINFORCE]]（策略梯度用 $G_t$）、[[baseline]]（减 $V$ 降 return 方差）、[[advantage function]]（$A=G-V$）、[[GAE]]（带 λ 的 return 插值）、[[reward normalization]]（归一化 reward 稳定训练）、[[reward hacking]]（reward 设计陷阱）。
- **对比 / 易混**:
  - **reward vs return**：见 §1，单步 vs 累积。
  - **return vs value**：return 是单条轨迹的实际量，value 是其期望。
  - **discounted return vs average reward**：前者有时间偏好，后者无。
  - **extrinsic reward vs intrinsic reward**：前者环境给，后者智能体自造（好奇心等）用于探索。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **优化 reward 而非 return**：贪心每步 reward 常次优；RL 目标是期望 return。
> 2. **$G_t$ 索引错位**：常见约定 $G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}$（reward 在转移后），也有教材用 $r_t$；读公式先对齐索引约定。
> 3. **continuing 用 $\gamma=1$**：return 可能发散；continuing 必须 $\gamma<1$。
> 4. **把 reward 当 loss**：reward 是标量信号不是可微损失；要用策略梯度或 Q-learning 把它转化成梯度。
> 5. **reward 尺度不归一**：reward 绝对值过大 → 梯度爆炸；过小 → 学不动。常做 running std 归一化（见 [[reward normalization]]）。
> 6. **稀疏 reward 不做信用分配**：仅末尾有 reward 时直接用 $G_t$ 方差巨大；用 [[GAE]]/advantage 减方差。
> 7. **reward shaping 改变最优策略**：随手加 shaping reward 可能让智能体钻空子；用势能函数法保证不改最优。
> 8. **混淆 episodic 的 $G_T$**：终止状态 $G_T=0$（无后续），递推从 $G_T=0$ 开始倒推。
> 9. **LLM-RL 里 reward 只在末尾**：RLHF 的奖励模型对整段回答打一次分，return 退化为"序列末尾一个 reward"（$\gamma=1$ 时 $G_t=r$ 对所有 token 相同）→ 信用分配靠 advantage/GAE 与 KL 一起做。

## 8. 延伸细节

### 8.1 LLM-RL 的 reward 形态

RLHF 中奖励模型 $r_\phi(\text{prompt},\text{response})$ 对整段回答打一个标量，仅在序列末尾给。等效 reward 序列：$r_1=\dots=r_{T-1}=0$，$r_T=r_\phi$。return $G_t=\gamma^{T-t}r_\phi$（$\gamma=1$ 时所有 token 的 $G_t$ 相同）。这使信用分配退化为"把一个序列级分数分到每个 token"——靠 advantage estimation 与 KL 约束完成，是 [[PPO clipped objective]] 在 LLM 上的核心难点。

### 8.2 折扣 vs 无折扣的等价

对 episodic 固定长度 $T$，$\gamma=1$ 与 $\gamma<1$ 只是数值尺度不同，最优策略不变（证明：折扣是单调变换）。但 $\gamma<1$ 数值稳定、有偏置小，实践中即使 episodic 也常用 $\gamma=0.99$。

### 8.3 return 的方差随 $\gamma$ 增大

$\gamma$ 越接近 1，return 累积越多步的 reward 噪声，方差越大。这是"远视带来高方差"的根源，也是 [[baseline]]/[[GAE]] 存在的理由。

### 8.4 generalized return（n-step 与 λ-return）

$G_t$ 与 $V^\pi$ 的插值：n-step return $G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V(s_{t+n})$，λ-return 在 n 上做指数加权。GAE 是其 advantage 版本，见 [[GAE]]。

### 8.5 reward 的有界性与数值稳定

实践中常 `reward = tanh(原始reward)` 或 clip 到 $[-c,c]$，防极端值引爆梯度。配合 [[reward normalization]] 的 running std。

---
相关: [[RL基本概念]]、[[MDP]]、[[policy与value function]]、[[REINFORCE]]、[[baseline]]、[[advantage function]]、[[GAE]]、[[PPO clipped objective]]、[[KL penalty与KL control]]、[[reward normalization]]、[[reward hacking]]
