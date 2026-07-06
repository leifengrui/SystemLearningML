# stale policy problem

> **所属章节**: [[训推分离]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（需懂异步训练 + PPO ratio + 版本管理）

## 1. 一句话定义

**stale policy problem（策略过时问题）** 是异步 [[训推分离]] 的核心问题：**采样侧（[[rollout worker]] / [[policy deployment (PD)]]）用的 $\theta$ 与训练侧（learner / [[policy training (PT)]]）最新的 $\theta$ 版本不一致**——worker 用 $\theta_{\text{old}}$ 采的数据，learner 已更新到 $\theta_{\text{new}}$；或 PD 部署 $\theta_{\text{old}}$，PT 已训 $\theta_{\text{new}}$。导致 PPO ratio $\rho=\pi_{\theta_{\text{new}}}/\pi_{\theta_{\text{old}}}$ 偏离 1 **失真**（训练不稳）或用户接触旧版本（服务落后）。**staleness $k$ = 版本差**，$k$ 越大失真越重。防法：[[weight sync mechanism]] 定期同步、PPO ratio clip 截断、V-trace 修正、target_kl 早停、PD 版本管理。它是 [[asynchronous training]] 与 PT-PD 分离的固有代价。

> [!note] 两种 stale
> - **训练内部 stale**：[[rollout worker]] 用 $\theta_t$ 采，learner 已训到 $\theta_{t+k}$（[[asynchronous training]] §3.2）。
> - **PT-PD stale**：[[policy deployment (PD)]] 部署 $\theta_{\text{old}}$，[[policy training (PT)]] 已训 $\theta_{\text{new}}$（PD 版本滞后）。
> 两者机制同（版本不一致），层次不同。

## 2. 为什么需要它（动机与背景）

异步训推分离必然产生版本 lag：
1. **异步必然 lag**：worker/PD 与 learner/PT 时间解耦，lag 期间 $\theta$ 已更新；
2. **PPO 假设 ratio≈1**：importance sampling $\rho=\pi_{\theta_{\text{new}}}/\pi_{\theta_{\text{old}}}$ 假设新旧策略近，stale 时 $\rho$ 偏离 1，clip 截断丢梯度信号；
3. **训练不稳**：stale 重时 ratio 失真，PPO 更新方向错，[[KL explosion]]/[[policy collapse]] 风险升；
4. **PD 服务落后**：用户接触旧版本，能力/对齐落后，体验差；
5. **在线 RLHF 尤甚**：持续训练 + 部署，stale 是常态，需主动管理。

理解 stale policy problem 是异步训推分离稳定性的关键——不管理则训练崩/服务差，管理则需 weight sync + clip + 版本协议协同。

## 3. 核心概念详解

### 3.1 训练内部 stale（worker vs learner）

[[asynchronous training]] 的异步 split：
- worker 在时刻 $t$ 用 $\theta_t$ 采数据 $D_t$；
- learner 已训练到 $\theta_{t+k}$ 才消费 $D_t$；
- **staleness $=k$ 步**；
- PPO ratio $\rho=\pi_{\theta_{t+k}}(a|s)/\pi_{\theta_t}(a|s)$，$k$ 大时偏离 1。

### 3.2 PT-PD stale（部署 vs 训练）

[[训推分离]] 的 PT-PD：
- PD 部署 $\theta_{\text{PD}}$（某旧版本）；
- PT 已训到 $\theta_{\text{PT}}$（更新版本）；
- **版本差 $=\theta_{\text{PT}}-\theta_{\text{PD}}$**；
- 用户接触 $\theta_{\text{PD}}$（旧），线上反馈基于旧 $\theta$。

### 3.3 ratio 失真与 PPO clip

PPO 用 ratio $\rho$ 做 importance sampling（[[PPO clipped objective]]）：
$$
L^{CLIP}=-\min(\rho\hat A,\,\text{clip}(\rho,1-\epsilon,1+\epsilon)\hat A)
$$

- $k=0$（同步）：$\rho=1$，无失真；
- $k$ 小：$\rho$ 略偏离 1，clip 不触发，梯度有效；
- $k$ 大：$\rho$ 大幅偏离 1，clip 截断 → **有效梯度信号弱**（被 clip 的样本梯度被限）；
- $k$ 过大：clip frequency 飙升，训练信号丢失，[[KL explosion]]/[[policy collapse]] 风险。

### 3.4 stale 的防法

| 防法 | 机制 | 层次 |
|---|---|---|
| **[[weight sync mechanism]]** | 定期把 learner/PT 新 $\theta$ 同步给 worker/PD，减小 $k$ | 训练内部 + PT-PD |
| **PPO ratio clip** | $\rho$ clip 在 $[1-\epsilon,1+\epsilon]$，限失真 | 训练内部 |
| **V-trace** | IMPALA 的截断 ratio + 多步 return，容忍 stale | 训练内部 |
| **target_kl 早停** | 每 epoch 算 KL，超 target 则 break | 训练内部 |
| **buffer 容量限制** | 旧数据过期丢弃，限最大 $k$ | 训练内部 |
| **PD 版本管理** | 定期升级 PD 追 PT，blue-green 热切换 | PT-PD |
| **版本标记** | 线上数据标 $\theta$ 版本，回传 PT 时做 off-policy 修正 | PT-PD |

### 3.5 stale 的诊断

- **staleness 分布**：buffer 数据的 $k$ 均值/最大值（训练内部）；
- **clip frequency**：ratio 触 clip 比例，stale 重则高；
- **approx_kl**：每 epoch KL，stale 引起漂移；
- **PD 版本差**：$\theta_{\text{PT}}$ vs $\theta_{\text{PD}}$ 的版本距离；
- **线上反馈分布**：若与 PT 最新 $\theta$ 分布偏移，PD stale 信号。

## 4. 数学原理 / 公式

### 4.1 ratio 与 staleness

$$
\rho=\frac{\pi_{\theta_{\text{new}}}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}
$$

- $\theta_{\text{new}}=\theta_{t+k}$（learner 当前），$\theta_{\text{old}}=\theta_t$（worker 采样时）；
- $k=0$：$\rho=1$；
- $k$ 大：$\rho$ 偏离 1，方向取决于 $\theta$ 变化方向。

### 4.2 clip 的截断效应

$$
\text{有效梯度}\propto\begin{cases}\rho\hat A & \rho\in[1-\epsilon,1+\epsilon]\\(1\pm\epsilon)\hat A & \rho\text{ 超 clip}\end{cases}
$$

超 clip 的样本梯度被截断为常数，**丢失比例信息**。stale 重时大量样本超 clip，有效梯度信号弱。

### 4.3 V-trace 的截断修正

IMPALA V-trace 用截断 ratio（[[asynchronous training]] §4.4）：
$$
\rho_{\bar\theta}=\min(\bar\rho,\,\pi_\theta/\pi_{\theta_{\text{old}}})
$$

截断在 $\bar\rho$（如 1），容忍 stale 的多步 return 估计，是异步训练的理论保证。

### 4.4 weight sync 的 staleness 控制

[[weight sync mechanism]] 每 $S$ 步同步 $\theta$ 给 worker/PD。最大 staleness $\le S$（两次 sync 间）。$S$ 小 → stale 轻，但 sync 开销大（$\theta$ 大，带宽/延迟）；$S$ 大 → stale 重。tradeoff。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F, copy

class LM(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)
    def logp(s, x, a):
        return F.log_softmax(s(x),-1).gather(-1,a.unsqueeze(-1)).squeeze(-1)

V, D = 50, 32
theta_old = LM(V, D)                    # worker 采样时的 θ
theta_new = copy.deepcopy(theta_old)    # learner 训了 k 步后的 θ
# 模拟 k 步训练(θ_new 偏离 θ_old)
with torch.no_grad():
    for p_new, p_old in zip(theta_new.parameters(), theta_old.parameters()):
        p_new.add_(torch.randn_like(p_new) * 0.5)   # 模拟 k 步更新

x = torch.randint(0, V, (4, 8))
with torch.no_grad():
    logp_old = theta_old.logp(x, x)    # worker 采样时的 logp(π_θ_old)
    logp_new = theta_new.logp(x, x)    # learner 当前的 logp(π_θ_new)
    ratio = (logp_new - logp_old).exp()  # ρ = π_new/π_old
    print(f"ratio 均值={ratio.mean():.3f}, 偏离 1 程度={((ratio-1).abs().mean()):.3f}")
    # clip 截断
    eps = 0.2
    clipped = ratio.clamp(1-eps, 1+eps)
    clip_freq = (ratio != clipped).float().mean()
    print(f"clip frequency={clip_freq:.2%}(stale 重则高,有效梯度信号丢失)")
    print(f"staleness k 越大,ratio 偏离越大,clip freq 越高 → 训练不稳")
print("\n防法:weight sync 减小 k + ratio clip 限失真 + V-trace 修正")
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[asynchronous training]]（stale 的来源）、[[训推分离]]（PT-PD stale）、[[learner worker split]]、[[PPO clipped objective]]（ratio clip）、[[policy training (PT)]]/[[policy deployment (PD)]]。
- **下游（应用）**: [[weight sync mechanism]]（防法核心）、V-trace/IMPALA、在线 RLHF 的 stale 管理、PD 版本管理。
- **对比 / 易混**:
  - **stale policy problem vs [[exposure bias]]**：前者是参数版本不一致（θ 滞后），后者是训练-推布分布偏移（前缀分布）。不同问题，可并发。
  - **stale policy problem vs [[KL explosion]]**：stale 引起 ratio 失真，可导致 KL drift，但不等同。stale 是成因，KL explosion 是后果之一。
  - **训练内部 stale vs PT-PD stale**：见 §3.1/§3.2。层次不同，防法略异。
  - **stale data vs stale policy**：前者指数据滞后（现象），后者指策略版本不一致导致的问题（后果）。前者引出后者。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"stale 只在训练内部"**：错。PT-PD 也有（PD 部署旧版本）。
> 2. **"sync 免费解决 stale"**：错。$\theta$ 大（GB 级），sync 有带宽/延迟开销，需 tradeoff 频率。
> 3. **"clip 根治 stale"**：错。clip 只限失真（截断），不消除 stale，且截断丢梯度信号。
> 4. **"同步训练无 stale"**：对（同步 $k=0$），但吞吐低，是 tradeoff。
> 5. **"stale 一定坏事"**：轻度 stale 可接受（V-trace/clip 容忍），重度才崩。
> 6. **"PD stale 无影响"**：错。用户接触旧版，且线上反馈基于旧 $\theta$，回传 PT 有分布偏移。
> 7. **"staleness k 固定"**：异步中 $k$ 是分布（随 lag 波动），需监控均值/尾部。
> 8. **"weight sync 越频繁越好"**：错。频繁 sync 减 stale 但增开销，且打断 worker 生成。

## 8. 延伸细节

### 8.1 V-trace 的理论保证

IMPALA 的 V-trace（Espeholt et al. 2018）证明：截断 ratio $\rho_{\bar\theta}=\min(\bar\rho,\pi_\theta/\pi_{\theta_{\text{old}}})$ 的多步 return 估计，在 $\bar\rho\ge1$ 时收敛到某个 fixed point，容忍有限 stale。这是异步训练 + stale 的理论基础，使异步 PPO 稳定。

### 8.2 在线 RLHF 的 stale 管理

在线 RLHF（PT 持续训练 + PD 持续服务 + 线上反馈回标）stale 是常态：
- PT 持续训 $\theta$，PD 定期升级；
- 线上反馈基于 PD 的 $\theta_{\text{PD}}$（可能滞后 PT）；
- 回标时标 $\theta$ 版本，PT 用 off-policy 修正（ratio clip）；
- weight sync 流式推送减 lag。
这是现代持续对齐系统的核心挑战。

### 8.3 PT-PD 版本协议

为管理 PD stale：
- PT 每版本标 commit/版本号；
- PD 按版本拉 checkpoint，记录当前服务版本；
- 线上反馈数据标 $\theta_{\text{PD}}$ 版本；
- PT 消费时算 $\rho=\pi_{\theta_{\text{PT}}}/\pi_{\theta_{\text{PD}}}$ 做 off-policy 修正。
这是数据驱动的 stale 管理。

### 8.4 sync 频率的 tradeoff

- **$S$ 小（频繁 sync）**：stale 轻，ratio 准，但 sync 开销大（$\theta$ 大，带宽/延迟），打断 worker 生成；
- **$S$ 大（稀疏 sync）**：stale 重，ratio 失真，但 sync 开销小；
- **最优 $S$**：权衡 stale 修正收益与 sync 开销，常通过监控 clip frequency 调（维持 0.1~0.3）。
详见 [[weight sync mechanism]]。

### 8.5 stale 与训练稳定性的连锁

重度 stale 可引发连锁：
- ratio 失真 → PPO 更新方向错；
- → KL drift（[[KL explosion]]）；
- → policy 塌缩（[[policy collapse]]）；
- → 训练崩。
故 stale 管理是 RLHF 稳定性工程的一环（[[训练稳定性工程]]）。

---
相关: [[训推分离]]、[[asynchronous training]]、[[learner worker split]]、[[policy training (PT)]]、[[policy deployment (PD)]]、[[weight sync mechanism]]、[[PPO clipped objective]]、[[KL penalty与KL control]]、[[KL explosion]]、[[policy collapse]]、[[exposure bias]]、[[训练稳定性工程]]
