# baseline

> **所属章节**: [[Policy Gradient体系]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**baseline（基线）** 是策略梯度里**与动作 $a$ 无关、只依赖状态 $s_t$（或常量）的项 $b(s_t)$**，从 return $G_t$ 中减去后作为权重：$\nabla_\theta J=\mathbb{E}[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)(G_t-b(s_t))]$。它**不改变梯度的期望**（无偏），但若取 $b\approx V^\pi(s_t)$ 能**大幅降低梯度方差**——从 [[REINFORCE]] 演进到 [[Actor-Critic]]/[[advantage function]] 的关键一步。

本质：把"这步动作好不好"改成"这步动作**比平均水平**好不好"。$G_t$ 是绝对值（正负、量级都随状态变），$G_t-b(s_t)$ 是相对值，把"状态本身就好/坏"的固定部分扣掉，只留"动作选择带来的差异"——梯度信号更纯净、方差更小，但期望不变。

## 2. 为什么需要它

[[REINFORCE]] 用 $G_t$ 作权重，问题：

- 同一状态 $s$ 下不同轨迹 $G_t$ 量级差异巨大（一条赢 +100，一条输 -10），梯度方差爆炸；
- 学习率被迫极小 → 收敛慢；
- "状态本身就好"的固定奖励淹没了"动作选择"的信号。

baseline 的洞察：$G_t$ 里有一部分**与动作 $a_t$ 无关**（状态 $s_t$ 决定的"底色"），扣掉它不影响动作对回报的边际效应，但去掉了这部分噪声。最优 baseline 就是 $V^\pi(s_t)$，扣完剩 $G_t-V^\pi\approx A^\pi$（[[advantage function]]）。

## 3. 核心概念

### 3.1 合法条件

$b$ 必须**不依赖 $a_t$**（可依赖 $s_t$、$\theta$、可是常量）。若依赖 $a_t$，$\mathbb{E}[\nabla\log\pi\cdot b]\ne0$，引入偏差。

合法形式：$b(s_t)$、$b(s_t;\phi)$（参数化）、常数 $b$、$b(\tau_{<t})$（历史但不含 $a_t$ 之后）。

### 3.2 最优 baseline

最小化梯度方差 $\text{Var}=\mathbb{E}[(G_t-b)^2\|\nabla\log\pi\|^2]$ 对 $b$ 求导，严格最优 $b^*=\mathbb{E}[G_t\|\nabla\log\pi\|^2\mid s_t]/\mathbb{E}[\|\nabla\log\pi\|^2\mid s_t]$。若 $\|\nabla\log\pi\|$ 与 $G$ 无关，简化为 $b^*=V^\pi(s_t)=\mathbb{E}[G_t\mid s_t]$。实践中 $V^\pi$ 已够好。

### 3.3 参数化 baseline（value network）

$V^\pi$ 未知，用神经网络 $V_\phi(s)$ 逼近（见 [[value network]]）。此时策略梯度变成：

$$\nabla_\theta J=\mathbb{E}\left[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)(G_t-V_\phi(s_t))\right]$$

这就是 **Actor-Critic** 的雏形（actor 学 $\pi_\theta$、critic 学 $V_\phi$）。$G_t-V_\phi$ 是 advantage 的 MC 估计。

### 3.4 baseline 变体

| 形式 | 来源 | 效果 |
|---|---|---|
| 常数 $b$ | batch 内 returns 均值 | 简单，减部分方差 |
| $V_\phi(s_t)$ | critic 网络 | 最常用，接近最优 |
| batch mean | 每 minibatch 归一化 returns | 工程小技巧 |
| 同状态其他动作 return | 留一法 | 更优但贵 |

## 4. 数学原理

### 4.1 无偏性证明

对任何 $b(s_t)$（不含 $a_t$）：

$$\begin{aligned}
\mathbb{E}_{a\sim\pi_\theta}[b(s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)]
&= b(s_t)\sum_a\pi_\theta(a\mid s_t)\frac{\nabla_\theta\pi_\theta(a\mid s_t)}{\pi_\theta(a\mid s_t)}\\
&= b(s_t)\sum_a\nabla_\theta\pi_\theta(a\mid s_t)=b(s_t)\nabla_\theta 1=0
\end{aligned}$$

故 $\mathbb{E}[\nabla\log\pi\,(G_t-b)]=\mathbb{E}[\nabla\log\pi\,G_t]$，期望不变。

### 4.2 方差下降（降的是什么）

"方差"特指**梯度样本** $X=\nabla\log\pi\,(G_t-b)$ 的方差，不是 reward 或 $\theta$ 的方差。最终 $\hat g=\frac1N\sum X_i$ 的方差 $\text{Var}[\hat g]=\text{Var}[X]/N$。

$$\text{Var}[X]=\underbrace{\mathbb{E}[X^2]}_{\text{二阶矩}}-\underbrace{\mathbb{E}[X]^2}_{\text{真实梯度平方}}$$

- **期望项** $\mathbb{E}[X]^2$：§4.1 已证 baseline 不改期望，**减 baseline 前后不变**；
- **二阶矩** $\mathbb{E}[X^2]$：减 $b$ 后 $(G_t-b)^2$ 变小（尤其 $b=V^\pi$ 时 $G_t-V$ 是均值 0 的残差）→ 二阶矩降 → 方差降。

$b=V^\pi$ 时扣掉"状态底色"，方差下降量 $\approx\text{Var}[V^\pi(s)]$。**方差降的是二阶矩，期望不变故无偏**。

### 4.3 与 advantage 的关系

$$A^{\pi_\theta}(s_t,a_t)=Q^{\pi_\theta}(s_t,a_t)-V^{\pi_\theta}(s_t)$$

用 $G_t$ 作 $Q^\pi$ 的 MC 估计，则 $G_t-V^\pi(s_t)\approx A^\pi$。**减 baseline = 用 advantage**。

## 5. 代码示例

```python
# Actor-Critic 骨架：critic 即 baseline
# 采集后 trans = [(s,a,r,logp,sn,done), ...]，算 returns 与 values
G = 0; returns = [0]*len(trans)
for t in reversed(range(len(trans))):
    G = trans[t][2] + gamma*G*(not trans[t][5]); returns[t]=G
returns = torch.tensor(returns)
values = torch.stack([ac.v(t[0]) for t in trans])
logps = torch.stack([t[3] for t in trans])

# advantage = G - V（baseline 减掉，detach 防 critic 梯度混进 actor）
adv = (returns - values.detach())
# advantage 归一化（工程标配）
adv = (adv - adv.mean())/(adv.std()+1e-8)
actor_loss  = -(logps * adv).mean()
critic_loss = ((values - returns)**2).mean()
loss = actor_loss + 0.5*critic_loss
```

## 6. 与其他知识点的关系

- **上游**: [[REINFORCE]]（baseline 是它的改进）、[[return与reward]]、[[policy与value function]]。
- **下游**: [[advantage function]]（$G-V=A$）、[[Actor-Critic]]（critic=baseline）、[[value network]]、[[variance reduction]]、[[PPO clipped objective]]、[[GAE]]。
- **对比**：
  - **baseline vs bias**：合法 baseline 无偏；只有依赖 $a_t$ 的"baseline"才引入偏差。
  - **baseline vs reward shaping**：shaping 改 reward（可能改最优策略），baseline 改梯度权重（不改期望、不改最优）。
  - **$V$ 作 baseline vs $V$ 作 target**：前者 $V$ 从 $G$ 里减（actor 用），后者 $V$ 被 $r+\gamma V'$ 回归（critic 用）。同一网络两种用法。
  - **baseline vs importance ratio**：baseline 减方差（同策略），ratio 处理 off-policy，正交机制。

## 7. 常见误区与易错点

1. **baseline 依赖 $a_t$** → 引入偏差，梯度期望改变。必须不含 $a_t$。
2. **把 $V$ 当 baseline 却没 detach**：critic 梯度混进 actor loss。`adv = (G - V).detach()` 或分开优化器。
3. **以为 baseline 改变最优策略**：合法 baseline 不改期望，最优策略不变；只改方差。
4. **baseline 学得不好还硬用**：$V_\phi$ 估偏差大时减方差效果打折，需 critic 收敛后再信 advantage。
5. **常数 baseline 当万能**：常数只减全局均值偏移，不减状态相关方差；状态相关 baseline（$V_\phi$）才强。
6. **off-policy 直接套 baseline 公式**：需先 importance reweight，否则梯度期望错。

## 8. 延伸细节

**control variate 视角**：baseline 是 **control variate**（控制变量）方法的特例——用与 $a$ 无关的已知量减方差，是蒙特卡洛方差缩减的经典工具，RL 里专门化成 $V^\pi$。

**LLM-RL 的角色**：RLHF 的 PPO 里 critic（value model）就是 baseline，advantage = return - critic。KL 惩罚项也可视作"把参考策略当 baseline"的连续版（见 [[KL penalty与KL control]]）。

**没有 critic 时**：用 batch 内 returns 均值 $\bar G$ 当常数 baseline 也行：`adv = G - G.mean()`。简单但不状态自适应，仅比裸 REINFORCE 略好。

---
相关: [[Policy Gradient体系]]、[[REINFORCE]]、[[advantage function]]、[[Actor-Critic]]、[[value network]]、[[variance reduction]]、[[GAE]]、[[PPO clipped objective]]
