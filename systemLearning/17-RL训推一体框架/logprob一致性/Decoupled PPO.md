# Decoupled PPO

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §83
> **别名**: Decoupled Proximal Policy Optimization / 解耦 PPO / 三策略 PPO / batch-size-invariant PPO / ppo-ewma
> **难度**: 高（需懂 [[PPO clipped objective]]、[[importance sampling与off-policy correction]]、[[rollout correction与IS weight]]、[[rollout train reference logprob一致性]]、[[V1 async trainer]]）
> **来源**: ① Hilton, Cobbe, Schulman, *Batch size-invariance for policy optimization*, arXiv:[2110.00641](https://arxiv.org/abs/2110.00641), NeurIPS 2022（OpenAI，代码 `github.com/openai/ppo-ewma`）；② verl 实现 `docs/algo/rollout_corr_math.md` §1.3/§2，`verl/trainer/config/algorithm.py: RolloutCorrectionConfig`（decoupled_* 预设）；③ A-3PO, arXiv:2512.06547 对 Decoupled PPO 的引用。

> [!warning] 与 [[DPPO]]（Divergence Proximal Policy Optimization）不是同一个算法
> 两者缩写撞车但完全不同：
> - **本条 Decoupled PPO**（Hilton/Cobbe/Schulman 2021，OpenAI）——解耦"近端策略"与"行为策略"，目标是**batch size 不变性**，是 verl 三策略 rollout-correction 框架的理论基础。
> - [[DPPO]]（Qi et al. 2026，Sea AI Lab）——用**散度（TV/KL）替代 ratio clip**做信任域，目标是对 token 级 ratio 单样本估计的长尾偏差纠偏。
> 记忆：**Decoupled = 解耦两个策略角色（OpenAI）；Divergence = 散度替代 ratio（Sea AI）**。

## 1. 一句话定义

**Decoupled PPO** 把标准 PPO 里"既当 clip 锚点（近端策略）又当数据采集策略（行为策略）"的同一个 $\pi_{\text{old}}$ **拆成两个独立策略**——近端策略 $\pi_{\text{prox}}$（冻结，做 PPO ratio 的分母、控更新幅度）与行为策略 $\mu$（采集数据，做 IS 权重的分母、纠 off-policy 偏差）——从而让"策略更新控制"与"数据聚合方式"解耦，得到一个对 batch size 不敏感、能高效复用 stale data 的三策略 PPO。

## 2. 为什么需要它（动机与背景）

### 2.1 标准 PPO 的 batch size 敏感性

标准 PPO 的 ratio $r_t(\theta)=\pi_\theta/\pi_{\text{old}}$ 里，$\pi_{\text{old}}$ 身兼两职：既是 clip 锚点（控制 $\pi_\theta$ 更新幅度），又是行为策略（数据就是它采的）。问题：**改 batch size 等于改有效行为策略**——多个 worker 聚合、上 replay buffer、跨 step 复用旧轨迹，都会让"数据分布"偏离 $\pi_{\text{old}}$，而 clip 机制假设二者相等。结果是 batch size 一变，超参（lr / clip range / n_epochs）全得重调，算法**不具备 batch size 不变性**（Hilton et al. §1：SGD 在小 batch 下有此性质，PPO 没有）。

### 2.2 异步 RL 里的 stale data

异步训练（[[V1 async trainer]]、[[asynchronous training]]）里，rollout 用的是若干步前的旧权重，$\pi_{\text{rollout}}$ 与 $\pi_\theta$ 之间有显著 lag。标准 PPO 把 $\pi_{\text{old}}=\pi_{\text{rollout}}$ 绑死，要么强行 on-policy 丢 stale data（浪费），要么无视偏差硬训（不稳）。A-3PO 论文指出：Decoupled PPO 用一个**冻结的近端策略**把 off-policy 校正（IS 权重）与策略更新约束（trust region clip）解耦，是异步高 staleness 场景的成功范式。

### 2.3 训推不一致（TIM）的根因之一

[[rollout train reference logprob一致性]] 指出：rollout 引擎与训练引擎同权重但不同实现（精度 / backend），$\pi_{\text{rollout}}\neq\pi_{\text{train}}$ 形成 Drift 1。把 $\pi_{\text{old}}$ 单独算一次（decoupled mode）就能用 IS 权重显式校正这层漂移，而不是混在 ratio 里被 clip 掩盖。

## 3. 核心概念详解

### 3.1 三个策略（verl 三策略框架）

verl `docs/algo/rollout_corr_math.md` §2 把 Decoupled PPO 实现为三个独立策略：

| 策略 | 角色 | 何时产生 | 训练中状态 |
|---|---|---|---|
| $\pi_{\text{rollout}}$（=行为策略 $\mu$） | 采集数据 | rollout 阶段 | 一个 batch 上冻结 |
| $\pi_{\text{old}}$（=近端策略 $\pi_{\text{prox}}$） | PPO clip 锚点 | **每个训练 epoch 开头单独 `compute_log_prob()` 算一次** | 整个 batch 的所有 PPO update epoch 上冻结 |
| $\pi_\theta$（当前策略） | 被优化 | 每步梯度更新 | 持续变化 |

关键：$\pi_{\text{old}}$ 与 $\pi_{\text{rollout}}$ **分离**——$\pi_{\text{old}}$ 是训练 epoch 开始时重算的快照，而非 rollout 时的旧权重。这正是"decouple"的字面含义。

### 3.2 两段漂移（Two Drifts）

三策略框架显式拆开两段分布漂移，各自校正：

- **Drift 1：$\pi_{\text{rollout}}\to\pi_{\text{old}}$（off-policy gap）**——采集策略与训练参考策略的差。用 **IS 权重** $w_t=\pi_{\text{old}}(a_t\vert s_t)/\pi_{\text{rollout}}(a_t\vert s_t)$ 校正。$\pi_{\text{old}}$ 冻结 → $w_t$ 是常数，**无需 stopgrad**。
- **Drift 2：$\pi_{\text{old}}\to\pi_\theta$（policy update drift）**——训练中参数更新带来的漂移。用 **PPO clip** on $r_t(\theta)=\pi_\theta(a_t\vert s_t)/\pi_{\text{old}}(a_t\vert s_t)$ 校正。

标准 PPO 把 Drift 1 假装为 0（$\pi_{\text{old}}=\pi_{\text{rollout}}$），只处理 Drift 2；Decoupled PPO 两段都显式处理。

### 3.3 batch size 不变性怎么来的

"控制更新幅度"（靠 $\pi_{\text{prox}}$，与 batch 怎么聚合无关）和"off-policy 校正"（靠 $\mu$，随 batch 聚合方式变）解耦后：改 batch size 只改变 $\mu$ 的有效分布，由 $w_t$ 吸收；clip 锚点 $\pi_{\text{prox}}$ 不受影响，更新幅度的控制尺度不变 → 超参不必随 batch size 重调。Hilton et al. 的实验展示这使算法能更高效复用 stale data。

## 4. 数学原理 / 公式

### 4.1 Decoupled PPO 目标

$$
L_{\text{DecoupledPPO}}(\theta)=-\mathbb{E}_{(s,a)\sim\mu}\left[w_t\cdot\min\left(r_t(\theta)A_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]
$$

其中：
- $w_t=\dfrac{\pi_{\text{old}}(a_t\vert s_t)}{\pi_{\text{rollout}}(a_t\vert s_t)}=\dfrac{\pi_{\text{prox}}(a_t\vert s_t)}{\mu(a_t\vert s_t)}$：IS 权重，校正 Drift 1。$\pi_{\text{old}}$ 冻结 → $w_t$ 常数。
- $r_t(\theta)=\dfrac{\pi_\theta(a_t\vert s_t)}{\pi_{\text{old}}(a_t\vert s_t)}$：PPO ratio，Drift 2。
- $\epsilon$：clip range（典型 0.2）。

### 4.2 退化为标准 PPO

令 $\pi_{\text{old}}=\pi_{\text{rollout}}$（即 $w_t\equiv 1$），上式即标准 PPO clipped surrogate——标准 PPO 是 Decoupled PPO 的"耦合特例"。verl 的 **bypass mode** 就是这个退化（见 §5）。

### 4.3 与 token/sequence 级聚合正交

$w_t$ 的聚合粒度（token 级 / sequence 级 / 几何均值）与"是否 decouple"**正交**（verl §3.3）。Decoupled 模式下可配 token-TIS、seq-TIS、Geo-RS、K3-RS 等任意 IS/RS 估计器——IS 估计器选什么与三策略结构无关。

## 5. 代码：verl 的 Decoupled 实现

verl `RolloutCorrectionConfig`（`verl/trainer/config/algorithm.py`）用 `bypass_mode` 开关切换：

| `bypass_mode` | 策略数 | $\pi_{\text{old}}$ 来源 | 算法 |
|---|---|---|---|
| `False`（默认，**decoupled**） | 3 | epoch 开头 `actor.compute_log_prob()` 单独算 | Full Decoupled PPO，batch size 不变 |
| `True`（**bypass**） | 2 | $\pi_{\text{old}}=\pi_{\text{rollout}}$（复用 rollout log_prob） | 退化为标准 PPO 或 PG-IS，更快但不不变 |

Decoupled 模式预设（三策略，IS 权重校正 Drift 1）：

```python
from verl.trainer.config.algorithm import RolloutCorrectionConfig

# 3 策略：π_rollout / π_old / π_θ；IS 在不同聚合级
cfg = RolloutCorrectionConfig.decoupled_token_is()       # Token-TIS
cfg = RolloutCorrectionConfig.decoupled_seq_is()         # Seq-TIS
cfg = RolloutCorrectionConfig.decoupled_seq_is_rs()      # Seq-MIS（IS+RS）
cfg = RolloutCorrectionConfig.decoupled_geo_rs()         # Geo-RS（几何均值 ratio）
cfg = RolloutCorrectionConfig.decoupled_k3_rs()          # K3 KL 估计器，小 KL 更稳
cfg = RolloutCorrectionConfig.decoupled_token_icepop()   # token IcePop（IS 截断到 [0.5, 5]）
```

关键实现点（`algorithm.py:126` docstring）：`bypass_mode=False` 时走标准 PPO loss + 显式 IS 权重；`bypass_mode=True` 时跳过 `compute_log_prob()` 调用，省一次前向，用 `loss_type="ppo_clip"` 或 `"reinforce"`。完整预设表见 `algorithm.py:144-170` 注释。

> [!tip] 何时选 decoupled vs bypass
> - **decoupled**：异步 / 跨 worker 聚合 / replay / 训推不一致显著——Drift 1 不可忽略，需显式 IS 校正。代价是每 epoch 多一次 `compute_log_prob` 前向。
> - **bypass**：严格 on-policy、rollout 与训练同引擎同精度、Drift 1 可忽略——省前向，等价标准 PPO。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[PPO clipped objective]]（Decoupled 是其三策略推广）、[[importance sampling与off-policy correction]]（$w_t$ 的理论）、[[R3 rollout routing replay]]（另一种 stale data 处理思路，DPPO 散度法声称无需它）
- **下游（应用）**: [[V1 async trainer]]（verl 异步管线靠三策略处理 staleness）、[[rollout correction与IS weight]]（verl rollout-correction 框架的理论根基就是 Decoupled PPO）、[[rollout train reference logprob一致性]]（Drift 1 即 TIM）
- **对比 / 易混**: [[DPPO]]（Divergence Proximal，散度替代 ratio，见顶部 warning）；[[DAPO]]（decoupled **clip** 指上下 clip 范围不对称，不是解耦策略，同名易混）；标准 PPO（Decoupled 的耦合特例，$\pi_{\text{old}}=\pi_{\text{rollout}}$）

## 7. 常见误区与易错点

- ❌ 把 "Decoupled PPO" 与 [[DPPO]]（Divergence Proximal）混为一个——两者目标、方法、出处全不同（见顶部 warning）。
- ❌ 把 [[DAPO]] 的 "Decoupled Clip"（上下 clip 范围不对称）当成本条——DAPO 解耦的是 clip 上下界，不是策略角色。
- ❌ 以为 decoupled 总是更好——它每 epoch 多一次前向（`compute_log_prob`），严格 on-policy 时是浪费；Drift 1 可忽略时用 bypass 更快。
- ❌ 忘了 $w_t$ 在 decoupled 模式下是常数（$\pi_{\text{old}}$ 冻结），还去 stopgrad——标准 PPO 里 $\pi_{\text{old}}$ 是数据自带不需 stopgrad，decoupled 同理，IS 权重常数项不传梯度。
- ❌ 把 Drift 1 和 Drift 2 混为一谈——前者 IS 权重校正（冻结→常数），后者 PPO clip 校正（随 $\theta$ 变），机制不同。

## 8. 延伸细节

- **原论文实现**：OpenAI `ppo-ewma` repo 用 EWMA（指数加权移动平均）做近端策略 $\pi_{\text{prox}}$ 的更新——近端策略不是 epoch 快照而是跨 step 的滑动平均，进一步平滑 trust region。
- **A-3PO（arXiv:2512.06547）** 指出 decoupled 的代价是"额外前向"，并主张用插值近似 $\pi_{\text{prox}}$ 来消除该开销——是 Decoupled PPO 在异步 LLM 训练场景的加速变体。
- **与 [[R3 rollout routing replay]] 的关系**：DPPO（散度法）声称无需 R3；Decoupled PPO 不绕开 stale data，而是用 IS 权重校正它——两条路线（纠偏 vs 重放）。
- verl `docs/algo/rollout_corr_math.md` 的叙事线是 **REINFORCE → PPO → Decoupled PPO**，把 Decoupled PPO 定位为"处理一般 off-policy 问题的统一框架"的理论终点，bypass/decoupled × IS/RS 估计器构成正交组合空间。

---
相关: [[DPPO]]｜[[DAPO]]｜[[PPO clipped objective]]｜[[importance sampling与off-policy correction]]｜[[rollout correction与IS weight]]｜[[rollout train reference logprob一致性]]｜[[V1 async trainer]]｜[[synchronous与asynchronous rollout]]
