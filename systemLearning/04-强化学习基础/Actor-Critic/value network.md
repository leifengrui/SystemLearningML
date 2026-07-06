# value network

> **所属章节**: [[Actor-Critic]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 中等（critic 的实现）

## 1. 一句话定义

**value network（价值网络 / critic）** 是用神经网络逼近 [[policy与value function]] 里 $V^\pi(s)$ 或 $Q^\pi(s,a)$ 的参数化模型 $V_\phi(s)$（或 $Q_\phi(s,a)$），在 [[Actor-Critic]] 中作为 **critic**：给 [[advantage function]] 提供 baseline、给 actor 提供低方差的梯度权重、自身靠 TD/MC 回归学习。它是策略梯度方法从 [[REINFORCE]]（无 critic、高方差）演进到 [[Actor-Critic]]/[[PPO clipped objective]]（有 critic、低方差）的工程载体。

> [!note] 三种叫法
> - **critic**：Actor-Critic 框架里给 actor 当"评委"的角色名。
> - **value network**：实现层面（神经网络逼近 $V$/$Q$）。
> - **baseline network**：强调它做策略梯度的 baseline（减方差）。
> 三者常指同一网络 $V_\phi$，视语境换用。LLM-RL 里也叫 value model / reward critic（有时与 reward model 共享骨干）。

## 2. 为什么需要它（动机与背景）

[[REINFORCE]] 用实际 return $G_t$ 作权重，方差爆。[[baseline]] 证明减 $V^\pi$ 不偏且降方差，但 $V^\pi$ 是期望量，未知，需估。三条估的路：

1. **表格法**：状态离散且少时可查表更新 —— 现实状态连续高维，不行。
2. **蒙特卡洛平均**：每个状态跑很多轨迹平均 —— 慢、每状态样本少。
3. **神经网络回归**：$V_\phi(s)\approx V^\pi(s)$，用 TD target $r+\gamma V_\phi(s')$ 或 MC return $G_t$ 作回归目标，泛化到未见过状态 —— **这就是 value network**。

价值网络的好处：函数逼近泛化（相似状态共享估计）、可与 actor 共享骨干（省算力）、可在线增量学习。代价：引入**函数逼近误差**（bias），且与 actor 互相依赖（critic 落后则 advantage 偏）。整个 Actor-Critic 训练动态就是 actor 与 critic 的"鸡生蛋"耦合，需谨慎设计学习率比与同步频率。

## 3. 核心概念详解

### 3.1 两类价值网络

| 网络 | 逼近 | 输出 | 用途 |
|---|---|---|---|
| $V_\phi(s)$ | 状态价值 | 标量 | 作 baseline，A3C/A2C/PPO 用 |
| $Q_\phi(s,a)$ | 动作价值 | 标量（输入含 $a$）或 |动作| 向量 | DQN 系、连续动作策略梯度 |

策略梯度方法（PG/PPO）几乎都用 $V_\phi$（只需 baseline，不需 per-action Q）；Q-learning 系用 $Q_\phi$。

### 3.2 训练目标：TD 与 MC

**TD(0) target**：

$$
\mathcal{L}_{\text{critic}}=\mathbb{E}\bigl[(V_\phi(s_t)-\underbrace{(r_{t+1}+\gamma V_{\phi_{\text{targ}}}(s_{t+1}))}_{\text{TD target}})^2\bigr]
$$

- 低方差、在线、有偏（依赖 $V_\phi$ 自身估准）。
- 常用**target network** $\phi_{\text{targ}}$（慢更新的旧副本）增加稳定。

**MC target**：

$$
\mathcal{L}_{\text{critic}}=\mathbb{E}\bigl[(V_\phi(s_t)-G_t)^2\bigr]
$$

- 无偏、高方差、需等回合。

**TD(λ)/GAE target**：插值，见 [[advantage estimation]]/[[GAE]]。

### 3.3 与 actor 的耦合

actor 用 $\hat A_t=G_t-V_\phi(s_t)$（或 GAE）作权重，故 critic 越准 advantage 越准 → actor 梯度越好 → policy 改进 → return 分布变 → critic 又要重学。这循环是 Actor-Critic 的本质动态。实务：

- critic 更新频率 $\ge$ actor（多步 TD 或更大 batch）；
- critic 学习率常大于 actor（让它跟得上）；
- 或 critic 用 target network 稳定。

### 3.4 共享骨干 vs 独立网络

- **共享**：actor 与 critic 共享 trunk（编码器），分头出 logits/value。省算力、特征复用，但 actor/critic loss 可能冲突干扰。
- **独立**：两个网络。互不干扰但算力翻倍。

PPO/CV 常共享；LLM-RL 里 critic 常独立（与 policy 同架构但不同参数，因 value 信号与 next-token 信号差异大）。

### 3.5 value 网络的输入

- CV/Atari：图像状态；
- 机器人：关节状态；
- LLM：prompt + 已生成 token（与 policy 同输入），输出该状态的未来期望 reward。

LLM 的 value network 通常与 policy 同 Transformer 架构，独立参数，head 输出标量 $V$。

### 3.6 value 网络的归一化

value 输出尺度随 reward 尺度变，常对 TD target 或 value 做归一化（pop-art、running mean/std），否则 loss 尺度漂移难调。见 [[reward normalization]]/[[variance reduction]]。

## 4. 数学原理 / 公式

### 4.1 TD 学习的不动点

$V_\phi$ 最小化 $\mathbb{E}[(V_\phi(s)-\mathbb{E}[r+\gamma V_\phi(s')])^2]$，不动点满足 $V^\pi(s)=\mathbb{E}[r+\gamma V^\pi(s')]$，即贝尔曼方程（[[policy与value function]] §3.4）。TD 是对 $V^\pi$ 的随机梯度下降（带 bootstrapping）。

### 4.2 半梯度与偏差

TD target 含 $V_\phi$ 自身，对 $\phi$ 求导时若对 target 也求导 → **半梯度**问题（target 不 detach 会有自举偏差）。实务 detach target：`target = (r + gamma*v_next).detach()`，loss 只对 $V_\phi(s_t)$ 求导。这是 TD 稳定训练的关键。

### 4.3 target network 的稳定作用

用慢更新 $\phi_{\text{targ}}\leftarrow\tau\phi+(1-\tau)\phi_{\text{targ}}$（Polyak）或周期性硬拷贝，使 target 缓慢变化，减小 TD 的"追移动目标"震荡，是 DQN 等能收敛的关键。

### 4.4 与 advantage 的关系

$A^\pi=Q^\pi-V^\pi$，$V_\phi$ 作 $V^\pi$ 的估计，advantage $\hat A=G_t-V_\phi$（MC）或 $\delta_t=r+\gamma V_\phi(s')-V_\phi(s)$（TD）。critic 越准 advantage 偏差越小、方差越低（见 [[variance reduction]]）。

## 5. 代码示例

```python
import torch, torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_act, shared=True):
        super().__init__()
        self.shared = shared
        if shared:
            self.trunk = nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
            self.actor_head = nn.Linear(64,n_act)
            self.critic_head = nn.Linear(64,1)
        else:
            self.actor = nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,n_act))
            self.critic = nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,1))
    def policy(self, s):
        h = self.trunk(s) if self.shared else self.actor(s)
        return torch.distributions.Categorical(logits=self.actor_head(h) if self.shared else h)
    def value(self, s):
        h = self.trunk(s) if self.shared else self.critic(s)
        return (self.critic_head(h) if self.shared else h).squeeze(-1)

ac = ActorCritic(4,2,shared=True)
opt = torch.optim.Adam(ac.parameters(), lr=3e-3)
gamma=0.99

# 一段 rollout + TD(0) critic 训练
s=torch.randn(4); buf=[]
for _ in range(64):
    d=ac.policy(s); a=d.sample(); sn=torch.randn(4); r=float(a==1)
    buf.append((s,a,r,sn)); s=sn

states=torch.stack([t[0] for t in buf]); nexts=torch.stack([t[3] for t in buf])
rs=torch.tensor([t[2] for t in buf],dtype=torch.float32)
v = ac.value(states)
with torch.no_grad():
    v_next = ac.value(nexts)
    target = rs + gamma*v_next        # TD target, detach
critic_loss = ((v - target)**2).mean()
# advantage = TD error, detach 给 actor
adv = (target - v).detach()
logps = torch.stack([ac.policy(t[0]).log_prob(t[1]) for t in buf])
actor_loss = -(logps*adv).mean()
loss = actor_loss + 0.5*critic_loss
opt.zero_grad(); loss.backward(); opt.step()
print("v_mean",v.mean().item(),"adv_std",adv.std().item())
```

> [!tip] target network 版本
> 加 `v_targ = AC(...); v_targ.load_state_dict(ac.state_dict())`，每 N 步 `v_targ.load_state_dict(ac.state_dict())` 或 Polyak 平均，target 用 `v_targ.value(nexts)`，更稳。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[policy与value function]]（$V^\pi$ 定义）、[[return与reward]]（target 来源）、[[baseline]]（critic=baseline）、[[advantage function]]（$A=G-V$）、[[forward与backward]]（神经网络训练）。
- **下游（应用）**: [[Actor-Critic]]（critic 是其一半）、[[advantage estimation]]（critic 估的准度决定 advantage 质量）、[[GAE]]（用 critic 算 $\delta$）、[[PPO clipped objective]]（critic + GAE 是标配）、[[value network]]（本条即它）、RLHF 的 value model。
- **对比 / 易混**:
  - **$V_\phi$ vs $Q_\phi$**：状态价值 vs 动作价值，PG 用前者、Q-learning 用后者。
  - **critic vs reward model**：reward model 给即时 reward $r$，critic 估未来累积 $V$。RLHF 里常是两个独立网络（reward model 训好后冻结，critic 在线学）。
  - **critic vs actor**：critic 评估、actor 决策；critic 可 detach 给 actor 当 baseline。
  - **TD vs MC critic**：偏差-方差权衡。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **TD target 没 detach** → 半梯度自举，critic 震荡甚至发散。`target=(r+gamma*v_next).detach()`。
> 2. **critic 学太慢** → advantage 用过时 $V$，偏差大；调大 critic lr 或增加更新频率。
> 3. **critic 学太快 / 无 target network** → 追移动目标震荡；off-policy 尤其要 target network。
> 4. **共享骨干但 actor/critic loss 冲突** → 梯度互相干扰；可调 loss 权重或独立网络。
> 5. **value 输出不归一化** → loss 尺度随 reward 漂移；归一化 target 或用 pop-art。
> 6. **LLM-RL 用 policy 骨干直接出 value** → value 信号与 next-token 信号差异大，常需独立 head 甚至独立网络。
> 7. **critic 与 actor 同步更新但 critic 落后** → advantage 偏；让 critic 多更新几步。
> 8. **value 网络估不准还信 advantage** → 早期 $V_\phi$ 偏差大，用 MC 或大 λ 的 GAE 过渡。
> 9. **混淆 value target 与 advantage**：target 是 $r+\gamma V'$ 回归 $V$，advantage 是 $G-V$ 给 actor。同一 $V_\phi$ 两用。
> 10. **reward model 当 critic 用**：reward model 输出即时 reward 不是累积 $V$，不能直接当 baseline。

## 8. 延伸细节

### 8.1 LLM-RL 的 value model 实践

RLHF 里 value model 常与 policy 同 Transformer 架构、独立参数、从 reward model 初始化或同预训练骨干。输入 prompt+已生成 token，输出该状态未来期望 reward。训练用 GAE target，rollout 时与 policy 并行前向。显存：另开一个同尺寸网络，开销大，催生 value/policy 共享或 value 蒸馏等优化。见 [[显存结构]]/[[FSDP]]。

### 8.2 Retrace 与 off-policy critic

off-policy 训 critic 时直接 TD 有分布偏移，Retrace(λ) 等方法用截断 importance ratio 修正，稳定估 $Q$。LLM-RL 几乎不用（on-policy 为主）。

### 8.3 self-criticism 与 self-reward

部分方法让 LLM 自己当 critic（self-critique、self-rewarding，如 SLiME、CRISP），省独立 value model，但稳定性存疑，是研究热点。

### 8.4 critic 的 over-estimation

$Q$ 网络的 max 操作放大高估偏差（[[policy与value function]] §8.5）。$V$ 网络无显式 max，但 TD bootstrapping 仍有轻微过估倾向，可用 double-critic（Clipped Double Q）缓解。

### 8.5 critic 与 reward normalization 协同

reward 归一化让 $V$ 输出尺度稳定，critic loss 易调；二者常配套。LLM-RL 里 reward model 输出可能漂移，running std 归一化 + critic 一起稳训练。

---
相关: [[Actor-Critic]]、[[advantage estimation]]、[[GAE]]、[[baseline]]、[[advantage function]]、[[policy与value function]]、[[PPO clipped objective]]、[[reward normalization]]、[[variance reduction]]
