# Actor-Critic

> **所属章节**: [[Actor-Critic]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**Actor-Critic（A-C）** 是策略梯度的**两网络联合学习架构**：actor = 策略网络 $\pi_\theta(a\mid s)$ 负责决策、被策略梯度直接优化；critic = 价值网络 $V_\phi(s)$（见 [[value network]]）评估当前策略、给 actor 提供**低方差的 advantage 权重** $\hat A_t$（见 [[advantage estimation]]）。它把 [[REINFORCE]] 的"用实际 return $G_t$ 当权重、无偏但方差爆炸"升级为"用 critic 估的 $V_\phi$ 减底色、有偏但方差低、可在线"，是 [[A2C]]/[[A3C]]/[[PPO clipped objective]]/RLHF-PPO 的统一骨架。

> [!note] reward → return → value → advantage
> reward $r_t$ → 累加得 return $G_t$ → 取期望得 $V^\pi(s)=\mathbb E[G_t\mid s]$ → 相减得 advantage $A^\pi=Q^\pi-V^\pi$（动作比平均好/坏多少）。A-C 用 advantage 而非 reward 当 actor 权重：扣掉"状态底色"只留"动作选择带来的增量"，且对策略动作期望为 0。

## 2. 为什么需要它

[[REINFORCE]] 用 $G_t$ 作权重，无偏但工程不可用：单条轨迹 $G_t$ 量级乱跳方差爆炸；必须等回合结束才能算（MC）；"状态本身好坏"的底色淹没动作信号。[[baseline]] 证明减一个与 $a$ 无关的 $b(s_t)$ 不改变梯度期望但降方差，最优 $b=V^\pi(s_t)$。但 $V^\pi$ 未知需在线估——这正是 critic 的活儿。

核心 trade：用 critic 的**偏差**换 return 的**方差**，用 TD/GAE 在偏差-方差间连续调。PPO 本质就是"带 ratio clipping + 多 epoch 复用的 Actor-Critic"。

## 3. 怎么工作

### 3.1 两半：actor 与 critic

| 角色 | 网络 | 学什么 | 损失 |
|---|---|---|---|
| actor | $\pi_\theta(a\mid s)$ | 策略（决策） | $-\hat A_t\nabla\log\pi_\theta$ |
| critic | $V_\phi(s)$ | 状态价值（评估） | $\|V_\phi-y\|^2$ |

两者通过 advantage 耦合：critic 越准 → $\hat A$ 越准 → actor 梯度越好 → 策略改进 → return 分布变 → critic 又要重学（双时间尺度，见 §4.3）。

### 3.2 advantage 是耦合点

$$
\hat A_t =
\begin{cases}
G_t - V_\phi(s_t) & \text{MC（无偏、高方差、需等回合）}\\
\delta_t = r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t) & \text{TD(0)（有偏、低方差、在线）}\\
\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k} & \text{GAE(λ)（工业标配，可调）}
\end{cases}
$$

选哪种决定 A-C 的"风味"：MC 版≈REINFORCE+baseline、TD 版≈TD-actor-critic、GAE 版是 PPO/A3C 标配。详见 [[advantage estimation]]。

### 3.3 critic 的训练目标

critic 回归一个"价值目标" $y_t$（见 [[value network]]）：MC 取 $G_t$；TD(0) 取 $r_{t+1}+\gamma V_{\phi_{\text{targ}}}(s_{t+1})$；GAE return 取 $V_\phi(s_t)+\hat A_t^{GAE}$（最稳，PPO 标配）。**target 必须 detach**，否则 TD 自举让 critic 追自己的移动目标、震荡发散。

### 3.4 共享骨干 vs 独立网络

| 方案 | 优 | 劣 | 用 |
|---|---|---|---|
| 共享 encoder 分头 | 省算力、特征复用 | actor/critic loss 冲突 | CV/Atari/A2C |
| 独立两网络 | 互不干扰 | 算力/显存翻倍 | LLM-RL 几乎必独立 |

LLM-RL 里 value 信号与 next-token 信号差异大，共享骨干梯度打架，故 PPO-RLHF 的 value model 几乎总是同架构但**独立参数**、从 reward model 或预训练骨干初始化。

### 3.5 on-policy 属性

A-C 是 **on-policy**：critic 估**当前** $\pi_\theta$ 的 $V^\pi$，actor 用**当前** $\pi_\theta$ 采的样本算梯度。旧策略采的样本需 importance ratio 重加权（PPO 的 $\rho=\pi_\theta/\pi_{\theta_{old}}$）。这是 A-C 样本利用率低的主因，PPO 用 ratio clipping + 多 epoch 复用缓解。

### 3.6 承接脉络

```
REINFORCE (G_t, 无 critic, 无偏高方差)
   └─ + baseline b(s_t)  → 期望不变方差降
        └─ b = V_φ(s_t) 参数化  → Actor-Critic
             └─ + GAE(λ) 估 A  → 现代 A-C / PPO 标配
                  └─ + ratio clipping + 多 epoch  → PPO
```

## 4. 数学原理

### 4.1 策略梯度定理

$J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G_0]$，由 [[REINFORCE]]：

$$
\nabla_\theta J(\theta)=\mathbb E\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right]
$$

环境转移 $P$ 不含 $\theta$ 求导消失，故不需要环境模型，只需采样。

### 4.2 引入 critic：advantage 形式

由 [[baseline]]，减 $V^\pi(s_t)$ 无偏：

$$
\nabla_\theta J(\theta)=\mathbb E\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(G_t-V^\pi(s_t))\right]=\mathbb E\left[\sum_t\nabla_\theta\log\pi_\theta\,A^\pi\right]
$$

（$\mathbb E[G_t\mid s,a]=Q^\pi$，故 $G_t-V^\pi$ 是 $A^\pi$ 的无偏 MC 估计。）把 $V^\pi$ 换成在线学的 $V_\phi$、$G_t-V_\phi$ 换成更稳的 $\hat A_t$：

$$
\boxed{\nabla_\theta J(\theta)\approx \mathbb E\left[\sum_t\nabla_\theta\log\pi_\theta\,\hat A_t\right]}
$$

### 4.3 双时间尺度收敛

actor 与 critic 互相依赖，能收敛靠**双时间尺度**：critic 用更小有效步长（更大 lr 或更多更新步数）→ 在"快时间尺度"上近似收敛到当前 $\pi_\theta$ 的 $V^{\pi_\theta}$；actor 用较慢步长 → 每步更新时 critic 已"追上"。形式上要求 $\eta^\phi/\eta^\theta\to\infty$（Konda & Tsitsiklis 2003）[待核实]。工程对应：critic 更新频率 $\ge$ actor、或 critic lr 更大、或用 target network。

### 4.4 完整更新步骤

一条 rollout $\{(s_t,a_t,r_{t+1},s_{t+1})\}$：

1. 用 critic 算 $\hat A_t$（GAE 倒推）；
2. actor loss：$\mathcal L_{\text{actor}}=-\frac1T\sum_t \log\pi_\theta(a_t\mid s_t)\,\hat A_t$（负号：梯度下降实现梯度上升）；
3. critic loss：$\mathcal L_{\text{critic}}=\frac1T\sum_t(V_\phi(s_t)-y_t)^2$，$y_t$ detach；
4. 总 loss：$\mathcal L=\mathcal L_{\text{actor}}+c_v\mathcal L_{\text{critic}}-c_e H(\pi_\theta)$（最后一项 [[entropy bonus]] 鼓励探索）；
5. 反向 + 优化器一步。

$\hat A_t$ 给 actor 时必须 **detach**，否则 critic 梯度混进 actor loss，actor 会去最小化 $V$ 而非最大化 $J$。

## 5. 代码示例（关键片段）

```python
with torch.no_grad():
    Vs, Vn = ac.value(states), ac.value(nexts)
    deltas = rs + gamma * Vn - Vs
    A = torch.zeros_like(deltas); A[-1] = deltas[-1]
    for t in reversed(range(len(buf)-1)):
        A[t] = deltas[t] + gamma * lam * A[t+1]   # GAE(λ) 倒推
    Gt = Vs + A                                     # critic target

A_norm = (A - A.mean()) / (A.std() + 1e-8)
v, dist = ac.value(states), ac.policy(states)
actor_loss  = -(logps * A_norm.detach()).mean()    # detach 防 critic 梯度混入
critic_loss = ((v - Gt.detach())**2).mean()
loss = actor_loss + coef_v * critic_loss - coef_e * dist.entropy().mean()
```

> [!tip] 真 PPO 在此基础上还要：①算 ratio $\rho=\pi_\theta/\pi_{\theta_{old}}$；②clip ratio 在 $[1-\epsilon,1+\epsilon]$；③同批 rollout 多 epoch 复用；④KL 早停。见 [[PPO clipped objective]]。

## 6. 与其他知识点的关系

- **上游**: [[REINFORCE]]（A-C 是它的减方差升级）、[[baseline]]（critic=参数化 baseline）、[[advantage function]]（$A=Q-V$ 是 actor 权重）、[[policy与value function]]、[[return与reward]]。
- **下游**: [[value network]]（critic 实现）、[[advantage estimation]]/[[GAE]]（如何估 $\hat A$）、[[PPO clipped objective]]、[[entropy bonus]]、RLHF-PPO。
- **对比**:
  - **A-C vs REINFORCE**：前者 critic 给低方差 advantage、有偏但稳；后者用 $G_t$、无偏但方差爆。
  - **A-C vs DQN**：前者显式学 $\pi_\theta$；后者学 $Q$ 隐式派生 $\pi=\arg\max_a Q$（无 actor）。
  - **critic $V_\phi$ vs reward model**：前者估未来累积 $V^\pi$（在线学、作 baseline）；后者给即时 reward $r$（预训练后冻结）。RLHF 里常是两个独立网络。

## 7. 常见误区与易错点

1. **$\hat A_t$ 没 detach critic**：critic 的 $V_\phi$ 梯度混进 actor loss，actor 会去最小化 $V$。`adv = adv.detach()`。
2. **TD target 没 detach**：critic 追移动目标 → 震荡发散。`target = (r + gamma * v_next).detach()`。
3. **critic 学太慢**：落后 actor → advantage 用过时 $V$、偏差大。调大 critic lr 或增加 critic 更新步数。
4. **共享骨干但 actor/critic loss 冲突**：梯度打架；调 loss 权重或独立网络（LLM-RL 必独立）。
5. **GAE 的 $\hat A$ 不归一化**：量级漂移、学习率难调；`(A-A.mean())/(A.std()+1e-8)`。
6. **混淆 critic target 与 advantage**：target 回归 $V$，advantage 给 actor；同一 $V_\phi$ 两个用途。
7. **off-policy 直接用旧 advantage**：策略变了 advantage 也变，需重算或 importance reweight（PPO 的 ratio）。
8. **loss 符号反**：策略梯度是上升 $J$，优化器做下降 → actor loss 取负。漏负号越训越差。

## 8. 延伸细节

### 8.1 A2C / A3C / PPO 演进

- **A3C**（Mnih 2016）[待核实]：每 worker 一个 CPU actor，异步 rollout + 异步 push 梯度到全局参数。早期 Atari SOTA。
- **A2C**：同步版，多 worker 一起 rollout 后梯度汇总、同步更新。GPU batch 友好。
- **PPO**（OpenAI 2017）：A2C + ratio clip + 多 epoch 复用 + GAE + KL 早停，更稳更省样本，RLHF 事实标准。

### 8.2 LLM-RL（RLHF）的 Actor-Critic 映射

| A-C 概念 | LLM-RL 对应 |
|---|---|
| actor $\pi_\theta(a\mid s)$ | 待微调 LLM $\pi_\theta(\text{token}\mid\text{prompt+已生成})$ |
| critic $V_\phi(s)$ | value model，与 policy 同架构独立参数，从 reward model/预训练骨干初始化 |
| reward $r$ | reward model 在序列末尾打一次分（稀疏） |
| $\hat A_t$ | GAE(λ≈0.95~1.0, γ=1) 把末尾 reward 倒推分到每个 token |
| KL 约束 | 限制 $\pi_\theta$ 不偏离参考 $\pi_{\text{ref}}$（[[KL penalty与KL control]]） |

显存开销：另开一个与 policy 同尺寸的 value model，开销大，催生 value/policy 共享、value 蒸馏、[[offloading (CPU-NVMe)]] 等优化。

### 8.3 连续动作与 off-policy 变体

- **连续动作**（DDPG/TD3/SAC）：critic 学 $Q_\phi(s,a)$ 而非 $V_\phi(s)$，actor 用确定性策略梯度或最大化 $Q$；SAC 加最大熵，是连续控制事实标准。
- **off-policy A-C**：ACER（ratio 截断 + Retrace）、IMPALA/V-trace（分布式截断 ratio 修 staleness）；PPO 介于 on/off 之间——on-policy 采样但 ratio clip + 多 epoch 复用，是"软 off-policy"。LLM-RL 因 ratio 在长序列上方差爆炸，几乎全走严格 on-policy + 少 epoch。

### 8.4 advantage 准度自检

critic 准 + 中等 λ → 稳定收敛；critic 不准 + 小 λ（信 TD）→ advantage 偏 → policy 跑飞（[[KL explosion]]/[[policy collapse]]）；critic 不准 + 大 λ（信 MC）→ 方差大但不易跑飞。实务：训练初期 critic 未收敛用大 λ（接近 MC），稳定后降 λ，或固定 λ=0.95 全程 GAE。

---
相关: [[REINFORCE]]、[[baseline]]、[[advantage estimation]]、[[value network]]、[[GAE]]、[[PPO clipped objective]]、[[entropy bonus]]、[[KL penalty与KL control]]
