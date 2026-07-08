# Actor-Critic

> **所属章节**: [[Actor-Critic]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 中等偏难（策略梯度方法的工程骨架）

## 1. 一句话定义

**Actor-Critic（演员-评论家，A-C）** 是策略梯度方法的**两网络联合学习架构**：**actor（演员）** = 策略网络 $\pi_\theta(a\mid s)$，负责决策、被策略梯度直接优化；**critic（评论家）** = 价值网络 $V_\phi(s)$（见 [[value network]]），负责评估当前策略、给 actor 提供**低方差的 advantage 权重** $\hat A_t$（见 [[advantage estimation]]）。它把 [[REINFORCE]] 的"用实际 return $G_t$ 当权重、无偏但方差爆炸"升级为"用 critic 估的 $V_\phi$ 减底色、有偏但方差低、可在线"，是 [[A2C]]/[[A3C]]/[[PPO clipped objective]]/RLHF-PPO 的统一骨架。

> [!note] 三个叫法
> - **Actor-Critic**：架构名（actor + critic 两半）。
> - **A2C / A3C**：Advantage Actor-Critic 的同步/异步工业实现（OpenAI/DeepMind 2016）。
> - **policy-critic / policy-value**：描述同一件事的非正式叫法。
> 本条讲架构本身；A2C/A3C 的工程细节见 §8.1，PPO 见 [[PPO clipped objective]]。

## 2. 为什么需要它（动机与背景）

[[REINFORCE]] 用实际 return $G_t$ 作策略梯度权重，数学无偏但工程不可用：

- 单条轨迹的 $G_t$ 量级从 -100 到 +100 乱跳，**梯度方差爆炸**，学习率被迫极小，收敛慢甚至不收敛；
- 必须等回合结束才能算 $G_t$（蒙特卡洛），**无法在线**，长回合或 continuing 任务不可行；
- "状态本身好坏"的固定底色（优势棋面下任何动作 $G$ 都大）淹没了"动作选择"的信号。

[[baseline]] 证明：减一个与动作 $a$ 无关的 $b(s_t)$ 不改变梯度期望但降方差，最优 $b=V^\pi(s_t)$。但 $V^\pi$ 是期望量、未知，需要**在线估**它——这正是 critic 的活儿。把 critic 估的 $V_\phi$ 当 baseline、把 $G_t-V_\phi$（或更稳的 GAE）当 advantage 喂给 actor，就是 Actor-Critic。

> [!tip] 一句话本质
> REINFORCE = 演员自己蒙眼试错、用"这次拿了多少分"当反馈（信号噪、慢）；Actor-Critic = 请一个评论家当场打分、用"比平均好/坏多少"当反馈（信号净、快）。评论家自己也要练，于是两个网络互相耦合、一起学。

Actor-Critic 的核心 trade 是：用 critic 的**偏差**换 return 的**方差**，并用 TD/GAE 在偏差-方差间连续调（见 [[advantage estimation]]）。它是现代深度 RL 与 LLM-RL（RLHF）的事实标准骨架——PPO 本质就是"带 ratio clipping + 多 epoch 复用的 Actor-Critic"。

## 3. 核心概念详解

### 3.1 两半：actor 与 critic

| 角色 | 网络 | 学什么 | 损失 | 角色 |
|---|---|---|---|---|
| **actor** | $\pi_\theta(a\mid s)$ | 策略（决策） | 策略梯度 $-\hat A_t\nabla\log\pi_\theta$ | 被 $\hat A$ 加权上升 |
| **critic** | $V_\phi(s)$ | 状态价值（评估） | TD/MC 回归 $\|V_\phi-y\|^2$ | 给 actor 提供 $\hat A$ |

actor 是"做什么"，critic 是"这么做平均能拿多少"。actor 直接被策略梯度优化，critic 靠回归价值目标学。两者通过 advantage $\hat A_t$ 耦合：critic 越准 → $\hat A$ 越准 → actor 梯度越好 → 策略改进 → return 分布变 → critic 又要重学。这循环是 A-C 的本质动态（见 §4.3 双时间尺度）。

### 3.2 advantage 是耦合点

A-C 的核心量是 advantage 估计 $\hat A_t$（见 [[advantage estimation]]）：

$$
\hat A_t =
\begin{cases}
G_t - V_\phi(s_t) & \text{MC advantage（无偏、高方差、需等回合）}\\
\delta_t = r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t) & \text{TD(0) advantage（有偏、低方差、在线）}\\
\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k} & \text{GAE(λ)（工业标配，偏差-方差可调）}
\end{cases}
$$

选哪种估计法决定了 A-C 的"风味"：MC 版接近 REINFORCE+baseline、TD 版接近 TD-actor-critic、GAE 版是 PPO/A3C 标配。

### 3.3 critic 的训练目标

critic 回归一个"价值目标" $y_t$，常见三种（见 [[value network]] §3.2）：

- **MC**：$y_t=G_t$（无偏、高方差）；
- **TD(0)**：$y_t=r_{t+1}+\gamma V_{\phi_{\text{targ}}}(s_{t+1})$（低方差、有偏、需 target network 稳定）；
- **GAE return**：$y_t=V_\phi(s_t)+\hat A_t^{GAE(\lambda)}$（最稳，PPO 标配）。

注意 **target 必须 detach**（`y = (r+gamma*v_next).detach()`），否则 TD 的"自举"（bootstrapping）会让 critic 追自己的移动目标、震荡发散（[[value network]] §4.2）。

### 3.4 共享骨干 vs 独立网络

| 方案 | 结构 | 优 | 劣 | 用 |
|---|---|---|---|---|
| **共享** | encoder 共享，分头出 logits / value | 省算力、特征复用 | actor/critic loss 可能冲突干扰 | CV/Atari/A2C 常用 |
| **独立** | 两个网络完全独立 | 互不干扰 | 算力翻倍、显存翻倍 | LLM-RL 几乎必独立 |

LLM-RL 里 value 信号（"未来累积 reward"）与 next-token 信号（"下一个词是什么"）差异极大，共享骨干会让梯度打架，故 PPO-RLHF 的 value model 几乎总是与 policy 同架构但**独立参数**、从 reward model 或同预训练骨干初始化（[[value network]] §8.1）。

### 3.5 同步 A2C vs 异步 A3C

- **A2C（Advantage Actor-Critic）**：同步版。多个 worker 各自 rollout，梯度汇总到中心参数服务器一起更新，参数始终一致。实现简单、GPU 利用率高，是工业主流。
- **A3C（Asynchronous Advantage Actor-Critic）**：异步版（Mnih 2016）。多个 worker 各自维护参数副本、异步 push 梯度，不等待彼此。增加探索多样性、不需锁，但参数有 staleness、实现复杂。
- 现代 GPU 集群上 A2C（同步、batch 大）几乎总比 A3C（异步、受通信限制）快，故 LLM-RL 全部走同步版（PPO 也是同步）。

### 3.6 on-policy 属性

A-C 是 **on-policy**：critic 估的是**当前** $\pi_\theta$ 的 $V^\pi$，actor 用**当前** $\pi_\theta$ 采的样本算梯度。旧策略采的样本不能直接用（分布偏移），要用 importance ratio 重加权（PPO 的 $\rho=\pi_\theta/\pi_{\theta_{old}}$，见 [[PPO clipped objective]]）。这是 A-C 样本利用率低的主因，也是 PPO 用 ratio clipping + 多 epoch 复用样本来缓解的痛点。

### 3.7 与 REINFORCE / baseline / advantage 的承接

```
REINFORCE (G_t, 无 critic, 无偏高方差)
   └─ + baseline b(s_t)  → 期望不变方差降 (但 b 要估)
        └─ b = V_φ(s_t) 参数化  → Actor-Critic (critic 在线学 V)
             └─ + GAE(λ) 估 A  → 现代 A-C / PPO 标配
                  └─ + ratio clipping + 多 epoch  → PPO
```

每一步都是"降方差 / 提样本效率 / 稳定更新"的工程改进，数学骨架始终是策略梯度定理（[[REINFORCE]] §4）。

## 4. 数学原理 / 公式

### 4.1 策略梯度定理回顾

目标 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0]$，由 [[REINFORCE]] §4.1：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right]
$$

环境转移 $P$ 不含 $\theta$ 求导消失，故不需要知道环境模型，只需采样。

### 4.2 引入 critic：advantage 形式

由 [[baseline]] §4.1，减 $V^\pi(s_t)$ 无偏：

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(G_t-V^\pi(s_t))\right]=\mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A^{\pi_\theta}(s_t,a_t)\right]
$$

（因 $\mathbb{E}[G_t\mid s_t,a_t]=Q^\pi$，故 $G_t-V^\pi$ 是 $A^\pi=Q-V$ 的无偏 MC 估计，见 [[advantage function]] §4.1。）

把 $V^\pi$ 换成在线学的 $V_\phi$、把 $G_t-V_\phi$ 换成更稳的 $\hat A_t$（GAE 等），就是 Actor-Critic 的梯度：

$$
\boxed{\nabla_\theta J(\theta)\approx \mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t\right]}
$$

$\hat A_t$ 的具体形式（MC/TD/GAE）决定 A-C 变体。

### 4.3 双时间尺度收敛（two-timescale）

actor 和 critic 互相依赖（critic 要 actor 的样本来估 $V^\pi$、actor 要 critic 的 $\hat A$ 来算梯度），理论上为什么能收敛？关键在**双时间尺度**：

- critic 用**更小的有效步长**（更大 lr 或更多更新步数）→ 在"快时间尺度"上近似收敛到当前 $\pi_\theta$ 的 $V^{\pi_\theta}$；
- actor 用**较慢步长** → 在"慢时间尺度"上更新 $\pi_\theta$，每一步 actor 更新时 critic 已"追上"当前策略。

形式上写成两个随机逼近过程 $\theta_{n+1}=\theta_n+\eta_n^\theta g_\theta(\theta_n,\phi_n)$、$\phi_{n+1}=\phi_n+\eta_n^\phi g_\phi(\theta_n,\phi_n)$，要求 $\eta^\phi/\eta^\theta\to\infty$（critic 步长相对快）。这是 Actor-Critic 算法收敛证明的标准框架（Konda & Tsitsiklis 2003）。

工程上对应做法：critic 更新频率 $\ge$ actor（每个 rollout step 多更新 critic 几次）、或 critic lr 大于 actor、或用 target network 给 critic 一个稳定锚点。

### 4.4 critic 的 TD 学习

critic 最小化（TD(0) 形式）：

$$
\mathcal{L}_{\text{critic}}(\phi)=\mathbb{E}\left[\bigl(V_\phi(s_t)-\underbrace{(r_{t+1}+\gamma V_{\phi_{\text{targ}}}(s_{t+1}))}_{\text{TD target }y_t,\ \text{detach}}\bigr)^2\right]
$$

target 用慢更新的 $\phi_{\text{targ}}$（Polyak 平均 $\phi_{\text{targ}}\leftarrow\tau\phi+(1-\tau)\phi_{\text{targ}}$ 或周期硬拷贝），避免"追自己的移动目标"（[[value network]] §4.3）。GAE 版 target 用 $y_t=V_\phi(s_t)+\hat A_t^{GAE}$ 更稳。

### 4.5 完整 Actor-Critic 更新公式

一条 rollout $\{(s_t,a_t,r_{t+1},s_{t+1})\}$：

1. 用 critic 算 $\hat A_t$（GAE 倒推）；
2. actor loss：$\mathcal{L}_{\text{actor}}(\theta)=-\frac1T\sum_t \log\pi_\theta(a_t\mid s_t)\,\hat A_t$（负号：梯度下降实现梯度上升）；
3. critic loss：$\mathcal{L}_{\text{critic}}(\phi)=\frac1T\sum_t(V_\phi(s_t)-y_t)^2$，$y_t$ detach；
4. 总 loss：$\mathcal{L}=\mathcal{L}_{\text{actor}}+c_v\mathcal{L}_{\text{critic}}-c_e H(\pi_\theta)$（最后一项是 [[entropy bonus]] 鼓励探索）；
5. 反向 + 优化器一步。

$\hat A_t$ 给 actor 时必须 **detach**（`adv = adv.detach()`），否则 critic 梯度混进 actor loss，actor 会去最小化 $V$ 而非最大化 $J$。

## 5. 代码示例

```python
import torch, torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_act, shared=True):
        super().__init__()
        self.shared = shared
        if shared:
            self.trunk = nn.Sequential(nn.Linear(obs_dim,64), nn.Tanh(),
                                       nn.Linear(64,64), nn.Tanh())
            self.actor_head = nn.Linear(64, n_act)
            self.critic_head = nn.Linear(64, 1)
        else:
            self.actor = nn.Sequential(nn.Linear(obs_dim,64), nn.Tanh(), nn.Linear(64,n_act))
            self.critic = nn.Sequential(nn.Linear(obs_dim,64), nn.Tanh(), nn.Linear(64,1))
    def policy(self, s):
        h = self.trunk(s) if self.shared else self.actor(s)
        return torch.distributions.Categorical(logits=self.actor_head(h) if self.shared else h)
    def value(self, s):
        h = self.trunk(s) if self.shared else self.critic(s)
        return (self.critic_head(h) if self.shared else h).squeeze(-1)

ac = ActorCritic(4, 2, shared=True)
opt = torch.optim.Adam(ac.parameters(), lr=3e-3)
gamma, lam = 0.99, 0.95          # GAE 参数
coef_v, coef_e = 0.5, 0.01       # critic / entropy 权重

def rollout(n=128):
    buf, s = [], torch.randn(4)
    for _ in range(n):
        d = ac.policy(s); a = d.sample(); logp = d.log_prob(a)
        sn = torch.randn(4); r = float(a == 1)        # 假环境：选动作1给小奖
        buf.append((s, a, r, sn, logp)); s = sn
    return buf

buf = rollout()
states  = torch.stack([t[0] for t in buf])
nexts   = torch.stack([t[3] for t in buf])
rs      = torch.tensor([t[2] for t in buf], dtype=torch.float32)
logps   = torch.stack([t[4] for t in buf])

with torch.no_grad():
    Vs = ac.value(states); Vn = ac.value(nexts)
    deltas = rs + gamma * Vn - Vs
    A = torch.zeros_like(deltas)
    A[-1] = deltas[-1]
    for t in reversed(range(len(buf)-1)):
        A[t] = deltas[t] + gamma * lam * A[t+1]      # GAE 倒推
    Gt = Vs + A                                       # critic target

A_norm = (A - A.mean()) / (A.std() + 1e-8)            # advantage 归一化
v = ac.value(states)
dist = ac.policy(states)
entropy = dist.entropy().mean()

actor_loss  = -(logps * A_norm.detach()).mean()       # detach 防 critic 梯度混入
critic_loss = ((v - Gt.detach())**2).mean()
loss = actor_loss + coef_v * critic_loss - coef_e * entropy

opt.zero_grad(); loss.backward(); opt.step()
print(f"actor={actor_loss.item():.3f} critic={critic_loss.item():.3f} "
      f"A_std={A.std().item():.3f} H={entropy.item():.3f}")
```

> [!tip] 这是 A2C 单步的最简形态
> 真 PPO 还要：①算 ratio $\rho=\pi_\theta/\pi_{\theta_{old}}$（用旧策略存的 logp）；②clip ratio 在 $[1-\epsilon,1+\epsilon]$；③对同批 rollout 多 epoch 复用；④KL 早停。把这些加上就是 [[PPO clipped objective]]。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[MDP]]（A-C 在 MDP 框架下定义）、[[policy与value function]]（actor 学 $\pi$、critic 学 $V$）、[[return与reward]]（$G_t$ / TD target）、[[REINFORCE]]（A-C 是它的减方差升级）、[[baseline]]（critic=参数化 baseline）、[[advantage function]]（$A=Q-V$ 是 actor 的权重）、[[forward与backward]]（两网络训练）。
- **下游（应用）**: [[value network]]（critic 的实现细节）、[[advantage estimation]]（如何估 $\hat A$，决定 A-C 风味）、[[GAE]]（工业标配 advantage）、[[variance reduction]]（A-C 整体即系统降方差）、[[PPO clipped objective]]（A-C + ratio clip + 多 epoch）、[[entropy bonus]]（A-C 标配加 $-\beta H$ 探索项）、RLHF-PPO（LLM-RL 的事实标准骨架）。
- **对比 / 易混**:
  - **Actor-Critic vs REINFORCE**：前者 critic 估 $V$ 给低方差 advantage、有偏但稳；后者用实际 $G_t$、无偏但方差爆。
  - **Actor-Critic vs DQN**：前者显式学 $\pi_\theta$（策略梯度）；后者学 $Q$ 隐式派生 $\pi=\arg\max_a Q$（无 actor）。
  - **Actor-Critic vs DDPG/TD3/SAC**：连续动作的 A-C 变体，critic 学 $Q_\phi(s,a)$ 而非 $V_\phi(s)$、actor 用确定性策略梯度或最大化 $Q$。
  - **A2C vs A3C**：同步 vs 异步（见 §3.5）。
  - **Actor-Critic vs PPO**：PPO = A-C + ratio clip + 多 epoch，更稳更省样本，是 A-C 的工业进化版。
  - **critic 的 $V_\phi$ vs reward model**：前者估未来累积 $V^\pi$（在线学、作 baseline）；后者给即时 reward $r$（预训练后冻结）。RLHF 里常是两个独立网络。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **$\hat A_t$ 没 detach critic**：critic 的 $V_\phi$ 梯度混进 actor loss，actor 会去最小化 $V$ 而非最大化 $J$。`adv = (G - V).detach()` 或 `A_norm = A.detach()`。
> 2. **TD target 没 detach**：critic 追自己的移动目标，自举偏差 → 震荡发散。`target = (r + gamma * v_next).detach()`。
> 3. **critic 学太慢**：落后 actor → advantage 用过时 $V$、偏差大。调大 critic lr 或增加每 rollout 的 critic 更新步数。
> 4. **critic 学太快但无 target network**：追移动目标震荡；off-policy 尤其要 target network（Polyak 平均或周期硬拷贝）。
> 5. **共享骨干但 actor/critic loss 冲突**：两路梯度打架；调 loss 权重或独立网络（LLM-RL 必独立）。
> 6. **把 GAE 的 $\hat A$ 当 MC 用而不归一化**：量级随训练漂移、学习率难调；`(A-A.mean())/(A.std()+1e-8)`。
> 7. **混淆 critic target 与 advantage**：target 回归 $V$（$r+\gamma V'$ 或 $V+A$），advantage 给 actor（$G-V$ 或 $\delta$）；同一 $V_\phi$ 两个用途。
> 8. **off-policy 直接用旧 advantage**：策略变了 advantage 也变，需重算或 importance reweight（PPO 的 ratio）。
> 9. **LLM-RL 用共享骨干直接出 value**：value 信号与 next-token 信号差异大，常需独立 head 甚至独立网络；从 reward model 初始化 critic。
> 10. **以为 A-C 比 REINFORCE 严格更优**：critic 不准时 A-C 的 advantage 偏差可能比 REINFORCE 的 $G_t$ 还差；早期 critic 未收敛时常先用大 λ 的 GAE 或 MC 过渡。
> 11. **loss 符号反**：策略梯度是上升 $J$，优化器做下降 → actor loss 取负。漏负号越训越差。
> 12. **A3C 当万能**：异步在 GPU 集群上常被同步 A2C 反超（通信瓶颈），现代 LLM-RL 全走同步版（PPO 也是同步）。

## 8. 延伸细节

### 8.1 A2C / A3C 工业实现

- **A3C（Mnih et al. 2016）**：每个 worker 一个 CPU actor，异步 rollout + 异步 push 梯度到全局参数。无需 GPU、强调探索多样性，是早期 Atari SOTA。
- **A2C**：同步版，多 worker 一起 rollout 后梯度汇总、同步更新。GPU batch 友好、实现简单。
- **PPO（OpenAI 2017）**：A2C 基础上加 ratio clip + 多 epoch 复用样本 + GAE + KL 早停，更稳更省样本，是当前 RLHF 事实标准（见 [[PPO clipped objective]]）。

演进脉络：REINFORCE → A3C/A2C → PPO → PPO-RLHF（LLM）。

### 8.2 LLM-RL（RLHF）的 Actor-Critic 映射

RLHF-PPO 是 A-C 的 LLM 特化：

| A-C 概念 | LLM-RL 对应 |
|---|---|
| actor $\pi_\theta(a\mid s)$ | 待微调的 LLM $\pi_\theta(\text{token}\mid\text{prompt+已生成})$ |
| critic $V_\phi(s)$ | value model，与 policy 同架构独立参数，从 reward model 或预训练骨干初始化 |
| reward $r$ | reward model 在序列末尾打一次分（稀疏） |
| $\hat A_t$ | GAE(λ=0.95~1.0, γ=1) 把末尾 reward 倒推分到每个 token |
| KL 约束 | 限制 $\pi_\theta$ 不偏离参考 $\pi_{\text{ref}}$（[[KL penalty与KL control]]） |
| 探索 | [[entropy bonus]] $-\beta H$ |

显存开销：另开一个与 policy 同尺寸的 value model，开销大，催生 value/policy 共享、value 蒸馏、[[offloading (CPU-NVMe)]] 等优化（[[value network]] §8.1、[[显存结构]]）。

### 8.3 连续动作的 A-C 变体

离散动作用 categorical 策略 + $V_\phi$；连续动作的 A-C 变体用 $Q_\phi(s,a)$ 当 critic：

- **DDPG**：确定性 actor $\mu_\theta(s)$ 最大化 $Q_\phi(s,\mu_\theta(s))$，critic 学 $Q$；
- **TD3**：DDPG + clipped double Q + delayed policy update，减过估；
- **SAC**：最大熵 A-C，actor 学随机策略 + 最大化 $Q+\alpha H$，样本效率与稳定性最强，是连续控制的事实标准。

它们都是 A-C 框架在连续动作 + off-policy 下的特化。

### 8.4 AlphaGo 的 Actor-Critic 血统

AlphaGo/AlphaZero 的 policy network（actor）+ value network（critic）双网络结构本质也是 Actor-Critic：actor 输出落子概率、critic 估局面胜率（$V$），用 MCTS 当 advantage 估计器。Self-play + MCTS 给 actor 提供"比平均好多少"的信号，是 A-C 在博弈搜索场景的特化。

### 8.5 与 GAE / 优势估计的依赖

A-C 的"好坏"极大程度上取决于 advantage 估得准不准：

- critic 准 + GAE(λ 中等) → advantage 准 → 稳定收敛；
- critic 不准 + 小 λ（信 TD）→ advantage 偏 → policy 跑飞（[[KL explosion]]/[[policy collapse]]）；
- critic 不准 + 大 λ（信 MC）→ advantage 方差大 → 收敛慢但不易跑飞。

实务：训练初期 critic 未收敛用大 λ（接近 MC），稳定后逐步降 λ（信 critic）—— 或直接固定 λ=0.95 全程用 GAE。详见 [[advantage estimation]] §8.4 的 critic 准度自检。

### 8.6 off-policy Actor-Critic

A-C 严格 on-policy，但 off-policy 版本存在：

- **ACER**：带 importance ratio 截断 + Retrace(λ) 的 off-policy A-C；
- **IMPALA / V-trace**：分布式 off-policy A-C，用截断 ratio 修正 staleness；
- **PPO**：介于 on/off 之间——on-policy 采样新 rollout，但 ratio clip + 多 epoch 让样本被复用多次，是"软 off-policy"的 A-C。

LLM-RL 因 ratio 在长序列上方差爆炸，几乎全走严格 on-policy + 少 epoch（PPO）。

### 8.7 双时间尺度的工程化

理论上 critic 应在 actor 的每步更新间"近似收敛"，但工程上 critic 不可能真收敛（数据有限）。实操用三个手段逼近：

1. **critic 更新频率更高**：每个 rollout 做 K 次 critic 更新、1 次 actor 更新；
2. **critic lr 更大**：让 critic 步长相对快（$\eta^\phi \gg \eta^\theta$）；
3. **target network**：给 critic 一个慢变的稳定锚点，避免追移动目标。

三者常组合使用。LLM-RL 里 critic 常用与 policy 相同的 lr 但每 rollout 多更新几步 + target network（Polyak τ=0.95~0.99）。

---
相关: [[04-强化学习基础]]、[[REINFORCE]]、[[baseline]]、[[advantage function]]、[[advantage estimation]]、[[value network]]、[[GAE]]、[[variance reduction]]、[[PPO clipped objective]]、[[entropy bonus]]、[[KL penalty与KL control]]、[[policy与value function]]、[[MDP]]、[[return与reward]]、[[reward normalization]]
