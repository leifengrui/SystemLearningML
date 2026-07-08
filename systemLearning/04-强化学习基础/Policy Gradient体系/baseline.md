# baseline

> **所属章节**: [[Policy Gradient体系]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 中等（策略梯度减方差第一招）

## 1. 一句话定义

**baseline（基线）** 是策略梯度里**与动作 $a$ 无关、只依赖状态 $s_t$（或常量）的项 $b(s_t)$**，从 return $G_t$ 中减去后作为权重：$\nabla_\theta J=\mathbb{E}[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)(G_t-b(s_t))]$。它在数学上**不改变梯度的期望**（无偏），但若取 $b\approx V^\pi(s_t)$ 能**大幅降低梯度方差**——这是从 [[REINFORCE]] 演进到 [[Actor-Critic]]/[[advantage function]] 的关键一步。

> [!note] 一句话本质
> baseline = "这步动作好不好" → 改成 "这步动作**比平均水平**好不好"。$G_t$ 是绝对值（正负、量级都随状态变），$G_t-b(s_t)$ 是相对值，把"状态本身就好/坏"的固定部分扣掉，只留"动作选择带来的差异"——梯度信号更纯净、方差更小，但期望不变。

## 2. 为什么需要它（动机与背景）

[[REINFORCE]] 用 $G_t$ 作权重，问题是：

- 同一状态 $s$ 下不同轨迹 $G_t$ 量级差异巨大（一条赢 +100，一条输 -10），梯度方差爆炸；
- 学习率被迫极小 → 收敛慢；
- "状态本身就好"（如棋面优势）的固定奖励也进了梯度，淹没了"动作选择"的信号。

baseline 的洞察：$G_t$ 里有一部分**与动作 $a_t$ 无关**（状态 $s_t$ 决定的"底色"），把它扣掉不影响动作对回报的边际效应，但去掉了这部分噪声。最优 baseline 就是 $V^\pi(s_t)$（"这个状态按当前策略平均能拿多少"），扣完剩 $G_t-V^\pi\approx A^\pi$（优势）。

baseline 的"不改变期望"是数学严格结论（见 §4），所以可以放心减；而"降方差"是实践刚需，是策略梯度从不可用变可用的关键工程。

## 3. 核心概念详解

### 3.1 baseline 的合法条件

$b$ 必须**不依赖 $a_t$**（可依赖 $s_t$、可依赖 $\theta$、可是常量）。若依赖 $a_t$，$\mathbb{E}[\nabla\log\pi\cdot b]\ne0$，会引入偏差。

合法形式：$b(s_t)$、$b(s_t;\phi)$（参数化）、常数 $b$、甚至 $b(\tau_{<t})$（历史，但不含 $a_t$ 之后）。

### 3.2 最优 baseline

最小化梯度方差 $\text{Var}=\mathbb{E}[(G_t-b)^2\|\nabla\log\pi\|^2]$ 对 $b$ 求导，最优 $b^*=\mathbb{E}[G_t\|\nabla\log\pi\|^2\mid s_t]/\mathbb{E}[\|\nabla\log\pi\|^2\mid s_t]$。若 $\|\nabla\log\pi\|$ 与 $G$ 无关，简化为 $b^*=V^\pi(s_t)=\mathbb{E}[G_t\mid s_t]$。故**用 $V^\pi$ 当 baseline 是近似最优**。

### 3.3 参数化 baseline（value network）

$V^\pi$ 未知，用神经网络 $V_\phi(s)$ 逼近，靠 TD 或 MC 回归学习（见 [[value network]]）。此时策略梯度变成：

$$
\nabla_\theta J=\mathbb{E}\left[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)(G_t-V_\phi(s_t))\right]
$$

这就是 **Actor-Critic** 的雏形（actor 学 $\pi_\theta$、critic 学 $V_\phi$）。$G_t-V_\phi$ 是 advantage 的 MC 估计（见 [[advantage function]]）。

### 3.4 baseline 的变体

| baseline 形式 | 来源 | 效果 |
|---|---|---|
| 常数 $b$ | batch 内 returns 均值 | 简单，减部分方差 |
| $V_\phi(s_t)$ | critic 网络 | 最常用，接近最优 |
| batch mean | 每 minibatch 归一化 returns | 工程小技巧 |
| 同状态其他动作 return | 留一法 | 更优但贵 |

### 3.5 baseline 与 off-policy

off-policy 里 baseline 仍可减（只要不含 $a_t$），但需配合 importance ratio。PPO 的 clipped ratio 与 baseline 是两套独立的减方差/稳机制。

### 3.6 与 reward 的尺度

baseline 还顺带把 reward 尺度归一化：$G_t-V$ 的量级大致在 $[-\text{几},\text{几}]$，学习率与 reward 绝对值解耦，训练更稳。

## 4. 数学原理 / 公式

### 4.1 无偏性证明

对任何 $b(s_t)$（不含 $a_t$）：

$$
\begin{aligned}
\mathbb{E}_{a\sim\pi_\theta}[b(s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)]
&= b(s_t)\sum_a\pi_\theta(a\mid s_t)\frac{\nabla_\theta\pi_\theta(a\mid s_t)}{\pi_\theta(a\mid s_t)}\\
&= b(s_t)\sum_a\nabla_\theta\pi_\theta(a\mid s_t)\\
&= b(s_t)\nabla_\theta\sum_a\pi_\theta(a\mid s_t)=b(s_t)\nabla_\theta 1=0
\end{aligned}
$$

故 $\mathbb{E}[\nabla\log\pi\,(G_t-b)]=\mathbb{E}[\nabla\log\pi\,G_t]-0$，期望不变。

### 4.2 方差下降

$$
\text{Var}[\nabla\log\pi\,(G_t-b)]=\mathbb{E}[(\nabla\log\pi)^2(G_t-b)^2]-\mathbb{E}[\nabla\log\pi(G_t-b)]^2
$$

第二项（期望平方）不变；第一项在 $b=V^\pi$ 时最小化（因 $G_t-V^\pi$ 是残差，方差最小）。整体方差降，降多少取决于 $V^\pi$ 估得准不准。
> [!note] 解答：这里"方差"特指梯度估计量 $\hat g$ 的方差，§4.2 公式降的就是它
> 这一行紧跟 §4.2 公式，问的就是公式里那个"方差"指什么。先定位再讲透。
>
> ### 这里"方差"指哪个随机变量的方差
> §4.2 的对象是 $\text{Var}[\nabla\log\pi\,(G_t-b)]$——也就是**梯度样本** $X=\nabla\log\pi\,(G_t-b)$ 的方差，不是 reward 的方差、不是 $G_t$ 的方差、不是参数 $\theta$ 的方差。最终 $\hat g=\frac1N\sum X_i$ 的方差 $\text{Var}[\hat g]=\text{Var}[X]/N$，所以降 $\text{Var}[X]$ 就是降梯度估计的噪声。
>
> ### 为什么 baseline 能降它（看公式两项）
> $$\text{Var}[X]=\underbrace{\mathbb{E}[X^2]}_{\text{二阶矩/能量}}-\underbrace{\mathbb{E}[X]^2}_{\text{真实梯度的平方}}$$
> - **第二项** $\mathbb{E}[X]^2$：§4.1 已证 baseline 不改期望（$\mathbb{E}[\nabla\log\pi\,b]=0$），所以这一项**减 baseline 前后不变**；
> - **第一项** $\mathbb{E}[X^2]$：减 $b$ 后 $G_t-b$ 的绝对值变小（尤其 $b=V^\pi$ 时 $G_t-V$ 是均值为 0 的残差），$(G_t-b)^2$ 变小 → 二阶矩降 → **方差降的就是这一项**。
>
> 直觉：$G_t$ 里混着"状态本身好坏"（$V^\pi$）和"动作好坏"（$A$）两部分，前者与动作无关却贡献了大量波动；扣掉 $V^\pi$ 后只剩 $A$，梯度样本的波动 $\downarrow$ 但方向（期望）$\to$ 不变。**方差下降量 $\approx\text{Var}[V^\pi(s)]$**，即"状态底色"那部分被扣掉了。
>
> ### 为什么"方差"在 PG 里重要到要专门证一节
> 梯度估计方差大 = 不同 batch 算出的 $\hat g$ 抖得厉害 = 学习率必须极小才不发散 = 训不动。baseline 是**无偏地**（不改期望）降方差的最佳手段，所以 PG 必上 baseline。详见 [[variance reduction]] §1 的"方差意味着什么"完整版（含方差来源拆解、偏差-方差 trade-off）。
>
> ### 一句话
> **§4.2 的"方差"= 梯度样本 $\nabla\log\pi\,(G_t-b)$ 的方差；baseline 通过压二阶矩 $\mathbb{E}[X^2]$ 降它（期望项不变故无偏）；$b=V^\pi$ 时扣掉状态底色、方差降最多。**
### 4.3 与 advantage 的关系

$$
A^{\pi_\theta}(s_t,a_t)=Q^{\pi_\theta}(s_t,a_t)-V^{\pi_\theta}(s_t)
$$

用 $G_t$ 作 $Q^\pi(s_t,a_t)$ 的 MC 估计，则 $G_t-V^\pi(s_t)\approx A^\pi(s_t,a_t)$。所以"减 baseline = 用 advantage"。详见 [[advantage function]]。

## 5. 代码示例

```python
import torch, torch.nn as nn

# Actor-Critic 骨架：critic 即 baseline
class AC(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,n_act))
        self.critic = nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,1))
    def pi(self, s): return torch.distributions.Categorical(logits=self.actor(s))
    def v(self, s):  return self.critic(s).squeeze(-1)

ac = AC(4,2); opt = torch.optim.Adam(ac.parameters(), lr=1e-3); gamma=0.99

def step_env(s, a): return torch.randn(4), float(a==1), False  # 假环境

s = torch.randn(4)
logp_sum, v_list, r_list = [], [], []
for t in range(200):
    d = ac.pi(s); a = d.sample(); logp_sum.append(d.log_prob(a))
    sn, r, done = step_env(s, a); r_list.append(r); v_list.append(ac.v(s)); s = sn

# 1) 算 G_t（从后往前）
G = 0; returns = []
for r in reversed(r_list):
    G = r + gamma*G; returns.insert(0, G)
returns = torch.tensor(returns)
values = torch.stack(v_list)
logps = torch.stack(logp_sum)

# 2) advantage = G - V（baseline 减掉）
adv = returns - values.detach()          # detach 防 critic 梯度混进 actor
# 3) actor loss: -logp * adv ; critic loss: (V - G)^2
actor_loss  = -(logps * adv).mean()
critic_loss = ((values - returns)**2).mean()
loss = actor_loss + 0.5*critic_loss
opt.zero_grad(); loss.backward(); opt.step()
print("actor_loss", float(actor_loss), "critic_loss", float(critic_loss))
```

> [!tip] advantage 也要归一化
> 工程上常 `adv = (adv - adv.mean())/(adv.std()+1e-8)` 进一步稳方差，见 [[variance reduction]]。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[REINFORCE]]（baseline 是它的改进）、[[return与reward]]（$G_t$）、[[policy与value function]]（$V^\pi$）。
- **下游（应用）**: [[advantage function]]（$G-V=A$）、[[Actor-Critic]]（critic=baseline）、[[value network]]（参数化 baseline）、[[variance reduction]]（系统降方差）、[[PPO clipped objective]]（用 advantage 做 ratio clipping）、[[GAE]]（更准的 advantage）。
- **对比 / 易混**:
  - **baseline vs bias**：合法 baseline 无偏；只有依赖 $a_t$ 的"baseline"才引入偏差。
  - **baseline vs reward shaping**：shaping 改 reward（可能改最优策略），baseline 改梯度权重（不改期望、不改最优）。
  - **$V$ 作 baseline vs $V$ 作 target**：前者 $V$ 从 $G$ 里减（actor 用），后者 $V$ 被 $r+\gamma V'$ 回归（critic 用）。同一网络两种用法。
  - **baseline vs importance ratio**：baseline 减方差（同策略），ratio 处理 off-policy（不同策略），正交机制。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **baseline 依赖 $a_t$** → 引入偏差，梯度期望改变，学错。必须不含 $a_t$。
> 2. **把 $V$ 当 baseline 却没 detach**：critic 梯度混进 actor loss，actor 去最小化 $V$ 而非最大化 $J$。`adv = (G - V).detach()` 或分开优化器。
> 3. **以为 baseline 改变最优策略**：合法 baseline 不改期望，最优策略不变；只改方差。
> 4. **baseline 学得不好还硬用**：$V_\phi$ 估偏差大时减方差效果打折，需 critic 收敛后再信 advantage。
> 5. **常数 baseline 当万能**：常数只减全局均值偏移，不减状态相关方差；状态相关 baseline（$V_\phi$）才强。
> 6. **减 baseline 后不归一化 advantage**：adv 量级仍随训练变，学习率难调；配合 adv 归一化。
> 7. **off-policy 直接套 baseline 公式**：需先 importance reweight，否则梯度期望错。
> 8. **混淆 baseline 与 value target**：baseline 是从 $G$ 里减的项，value target 是 $r+\gamma V'$ 回归 $V$。同一 $V_\phi$ 一身两职。

## 8. 延伸细节

### 8.1 最优 baseline 的精确形式

严格最优 $b^*=\mathbb{E}[G_t\|\nabla\log\pi\|^2\mid s_t]/\mathbb{E}[\|\nabla\log\pi\|^2\mid s_t]$，加权按 $\|\nabla\log\pi\|^2$（动作分布越平坦 $\|\nabla\log\pi\|$ 越大）。实践中 $V^\pi$ 已够好，不追精确。

### 8.2 多种 baseline 叠加

可同时用 $V_\phi$ 状态 baseline + batch mean + reward normalization，三者正交，组合降方差效果叠加。但要保证无偏（都不含 $a_t$）。

### 8.3 baseline 在 LLM-RL 的角色

RLHF 的 PPO 里 critic（value model）就是 baseline，advantage = return - critic。critic 常用 reward model 同骨架或独立网络。KL 惩罚项也可视作"把参考策略当 baseline"的连续版。详见 [[KL penalty与KL control]]。

### 8.4 baseline 与 control variate

baseline 是 **control variate**（控制变量）方法的特例：用与 $a$ 无关的已知量减方差。蒙特卡洛方差缩减的经典工具，RL 里专门化成 $V^\pi$。

### 8.5 没有 critic 时的退化

若不学 $V_\phi$，用 batch 内 returns 均值 $\bar G$ 当常数 baseline 也行：`adv = G - G.mean()`。简单但不状态自适应，仅比裸 REINFORCE 略好。这是最简基线。

---
相关: [[Policy Gradient体系]]、[[REINFORCE]]、[[advantage function]]、[[Actor-Critic]]、[[value network]]、[[variance reduction]]、[[GAE]]、[[PPO clipped objective]]
