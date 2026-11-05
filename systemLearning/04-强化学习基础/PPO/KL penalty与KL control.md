# KL penalty与KL control

> **所属章节**: [[PPO]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**KL penalty / KL control** 是 [[PPO clipped objective]] 中除 ratio clipping 之外的**第二道策略约束机制**，用 KL 散度 $\text{KL}(\pi_\theta\|\pi_{\text{ref/old}})$ 限制当前策略 $\pi_\theta$ 偏离参考策略 $\pi_{\text{ref}}$（RLHF 中是冻结的 SFT 模型）或采样快照 $\pi_{\theta_{old}}$ 的程度，防 [[policy collapse]]/[[KL explosion]]/[[reward hacking]]。两种实现形态：

- **KL penalty（软惩罚）**：把 $-\beta\,\text{KL}$ 加到 reward（per-token）或 loss 上，$\beta$ 是系数；
- **KL control（自适应/早停）**：adaptive controller 动态调 $\beta$ 维持 `target_kl`，或监控 `approx_kl` 超 `target_kl` 直接 break epoch 循环。

二者常并用：reward 里加 fixed/adaptive KL penalty（约束到 $\pi_{ref}$ 保 SFT 人格），训练时用 target_kl 早停（约束到 $\pi_{\theta_{old}}$ 防单批漂移）。

> [!note] 名字辨析
> "KL penalty" 指把 KL 作为惩罚项加进 reward/loss（OpenAI PPO paper 的 Adaptive KL Penalty 版本）；"KL control" 指用 KL 做控制信号（自适应调系数或早停）。二者常混用，本笔记统一用"KL 约束"指代，具体形态见 §3。

## 2. 为什么需要它

[[PPO clipped objective]] 的 ratio clipping 已限制单步更新幅度，为什么还要 KL 约束？三重动机：

1. **clip 的局限**：clip 只在**单样本** $(s_t,a_t)$ 上限制 $\rho_t$，不约束**整批策略分布**的漂移。一批里每个样本都没触发 clip，但整体 KL 可能已很大——策略仍可能跑飞。KL 是**分布级**约束，补 clip 盲区。
2. **多 epoch 复用的累积漂移**：PPO 做 $K$ epoch，每 epoch $\rho$ 偏一点，累积起来 $\pi_\theta$ 可能严重偏离 $\pi_{\theta_{old}}$。target_kl 早停在"累积漂移超阈"时 break。
3. **RLHF 的 SFT 人格保护**：RM 只在它见过的偏好数据上可靠。$\pi_\theta$ 若漂离 $\pi_{ref}$ 太远会落到 RM 分布外，reward 不可信、模型跑飞输出乱码或钻空子。加 $-\beta\,\text{KL}(\pi_\theta\|\pi_{ref})$ 把 $\pi_\theta$ 软约束在 $\pi_{ref}$ 附近——这是 RLHF 区别于普通 PPO 的核心约束。

理论根基是 **KL-constrained Policy Optimization**（TRPO）：

$$
\max_\theta\,\mathbb{E}_{\pi_\theta}[r]\quad\text{s.t.}\quad \text{KL}(\pi_{\text{ref}}\|\pi_\theta)\le\delta
$$

带 KL 约束的策略改进有**单调性保证**（Kakade 2002, TRPO）。KL penalty 是把硬约束变软惩罚（拉格朗日），KL control 是变成自适应/早停。

## 3. 核心概念详解

### 3.1 三种 KL 约束形态对比

| 形态 | 加在哪 | 约束到谁 | 系数 | 何时用 |
|---|---|---|---|---|
| **reward KL penalty** | per-token reward $r_t$ | $\pi_{\text{ref}}$（SFT） | $\beta$（fixed/adaptive） | RLHF 标配（OpenAI/InstructGPT） |
| **loss KL penalty** | PPO loss $L$ | $\pi_{\theta_{old}}$ | $\beta$（adaptive） | 原始 PPO paper 的备选目标，实际少用 |
| **target_kl early stop** | 训练循环 | $\pi_{\theta_{old}}$（本批漂移） | target_kl 阈值 | 几乎所有 PPO 实现都加 |

> [!note] 两种"参考"的区别
> - $\pi_{\text{ref}}$：**冻结的 SFT/reference 模型**，整个 RLHF 全程不变，是"语言能力锚点"；
> - $\pi_{\theta_{old}}$：**当前 rollout 周期的采样快照**，每个 PPO 迭代重置，是"防本批漂移"。
>
> RLHF 的 reward KL penalty 用 $\pi_{\text{ref}}$（保人格），target_kl 早停用 $\pi_{\theta_{old}}$（防漂移）。两者作用不同，别混。

### 3.2 reward KL penalty（RLHF 主力）

把 KL 当 per-token 负 reward shaping：

$$
r_t\leftarrow r_t-\beta\,\text{KL}\bigl(\pi_\theta(\cdot|s_t)\big\|\pi_{\text{ref}}(\cdot|s_t)\bigr)
$$

- 实际用单样本估计：$-\beta\log\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}$（per-token log-ratio，是 reverse KL 的无偏估计，§4.3）；
- 这个 $r_t$ 进 [[GAE]] 与 [[advantage function]]，让 actor 在算 advantage 时就**看到** KL 代价——选偏离 $\pi_{ref}$ 的 token 会被扣 reward；
- $\beta$ 是"保 SFT 人格的力度"，InstructGPT 取 $\beta\approx0.01\sim0.1$，adaptive controller 调。

**等价理解**：reward shaping $r-\beta\,\text{KL}$ 对应优化 $\mathbb{E}_{\pi_\theta}[r_{\text{RM}}]-\beta\,\text{KL}(\pi_\theta\|\pi_{ref})$，这正是 RLHF 标准目标 **max reward - β·KL**。[[DPO]] 直接对此目标做闭式重写，故 DPO 隐式含此 KL penalty。

### 3.3 loss KL penalty（Adaptive KL Penalty 版）

原始 PPO paper 的备选目标（与 clipped surrogate 二选一）：

$$
L^{KPEN}=L^{IS}-\beta\,\mathbb{E}_t[\text{KL}(\pi_{\theta_{old}}\|\pi_\theta)]
$$

加在 surrogate loss 上，约束到本批采样快照 $\pi_{\theta_{old}}$。实际中**clipped 版（$L^{CLIP}$）效果更好且简单**，故 loss KL penalty 少用；但概念上它是"KL 软约束"的代表。

### 3.4 KL 方向：forward vs reverse

- **forward KL** $\text{KL}(\pi_{ref}\|\pi_\theta)$：mode-covering，$\pi_\theta$ 要覆盖 $\pi_{ref}$ 所有 mode → 倾向**分散**；
- **reverse KL** $\text{KL}(\pi_\theta\|\pi_{ref})$：mode-seeking，$\pi_\theta$ 集中在 $\pi_{ref}$ 高概率区 → 倾向**集中**（可能 mode collapse）。

RLHF 的 reward KL penalty 通常用 **reverse KL** $\text{KL}(\pi_\theta\|\pi_{ref})$：$\pi_\theta$ 只在 $\pi_{ref}$ 已支持的区探索，不乱跑到 ref 低概率区（保语言合法）。这是 per-token log-ratio 估计 $\log\frac{\pi_\theta(a_t)}{\pi_{ref}(a_t)}$ 对应的方向。注意 TRPO 原文用 forward $\text{KL}(\pi_{old}\|\pi_\theta)$，方向不同——别混。

### 3.5 Adaptive KL controller

$\beta$ 固定难调：太小 KL 爆，太大 reward 被压死训不动。adaptive controller 动态调 $\beta$ 维持 `target_kl`：

```
if kl > target_kl * 1.5:  β *= 1.5      # KL 太大,加大惩罚
if kl < target_kl / 1.5:  β /= 1.5      # KL 太小,放松惩罚
```

- target_kl 是"想要的策略漂移量"，LLM-RL per-token 累积约 6~10 nats，游戏 RL 约 0.01~0.02；
- InstructGPT/TRL 的 `AdaptiveKLCoeff` 实现；
- 应用**上一批**的 KL 调**本批**的 β，避免本批耦合。

### 3.6 target_kl early stopping（最常用）

每个 PPO epoch 后算 `approx_kl`，超 `target_kl` 则 break：

```
for epoch in range(K_epochs):
    update(...)
    approx_kl = ((ratio - 1) - log(ratio)).mean()   # Schulman 近似,比 ρ·log ρ 数值稳
    if approx_kl > target_kl: break
```

- target_kl 游戏约 0.01~0.02，LLM-RL per-token 约 0.001~0.01（更保守）；
- 这是 PPO 实现里几乎必加的"第二道闸"。

### 3.7 估计 KL 的几种方式

| 估计式 | 形式 | 对应 KL | 稳定性 |
|---|---|---|---|
| Schulman 近似 | $\mathbb{E}[(\rho-1)-\log\rho]$ | $\text{KL}(\pi_{old}\|\pi_\theta)$ | 数值稳（推荐） |
| 直接 | $\mathbb{E}[\rho\log\rho]$ | $\text{KL}(\pi_{old}\|\pi_\theta)$ | $\rho$ 大时数值差 |
| per-token log-ratio | $\mathbb{E}_{a\sim\pi_\theta}[\log\frac{\pi_\theta(a)}{\pi_{ref}(a)}]$ | $\text{KL}(\pi_\theta\|\pi_{ref})$ | reward penalty 用 |

$\rho=\pi_\theta/\pi_{\theta_{old}}$。Schulman 近似来自 $x-1-\log x\ge0$ 且小扰动下 $\approx(x-1)^2/2\approx\text{KL}$，对 $\rho$ 不敏感故稳。

## 4. 关键推导

### 4.1 KL-constrained PO 的拉格朗日

$$
\max_\theta\,\mathbb{E}_{\pi_\theta}[r]\quad\text{s.t.}\quad\text{KL}(\pi_{ref}\|\pi_\theta)\le\delta
$$

拉格朗日松弛：$\max_\theta\,\mathbb{E}_{\pi_\theta}[r]-\beta\,\text{KL}(\pi_{ref}\|\pi_\theta)$，$\beta\ge0$ 是拉格朗日乘子。这就是 reward/loss KL penalty 的来源——把硬约束转软惩罚，$\beta$ 是约束的"价格"。

### 4.2 reward KL penalty 的等价目标

reward shaping $r_t\leftarrow r_t-\beta\,\text{KL}(\pi_\theta(\cdot|s_t)\|\pi_{ref})$ 后，期望累积目标：

$$
J(\theta)=\mathbb{E}_{\pi_\theta}[R_{\text{RM}}]-\beta\,\mathbb{E}_{\pi_\theta}\left[\sum_t\gamma^t\,\text{KL}(\pi_\theta\|\pi_{ref})\right]
$$

这正是 RLHF 标准目标 **max reward - β·KL**，与 [[DPO]] 推导的出发点一致。

### 4.3 per-token log-ratio 是 reverse KL 的无偏估计

$$
\mathbb{E}_{a\sim\pi_\theta}\left[\log\frac{\pi_\theta(a|s)}{\pi_{ref}(a|s)}\right]=\text{KL}(\pi_\theta(\cdot|s)\|\pi_{ref}(\cdot|s))
$$

故 reward 里 $-\beta\log\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$ 是 $-\beta\,\text{KL}(\pi_\theta\|\pi_{ref})$ 的单样本无偏估计。这解释了 RLHF 实现里 reward penalty 为何是 per-token log-ratio。

### 4.4 与 ratio clipping 的层次互补

- **clip**：单样本 $(s_t,a_t)$ 上的硬截断，约束 $\rho_t\in[1-\epsilon,1+\epsilon]$；
- **KL penalty**：分布级软惩罚，约束 $\mathbb{E}[\text{KL}]$；
- **target_kl early stop**：整批漂移的硬阈值。

三者层次：clip（样本级）→ KL penalty（分布级软）→ target_kl（批级硬）。三者并用最稳，是 RLHF-PPO 的标准配置。

## 5. 代码示例

```python
import torch, copy

# adaptive KL controller(TRL/InstructGPT 风格)
class AdaptiveKLController:
    def __init__(self, init_beta=0.1, target_kl=6.0):
        self.beta=init_beta; self.target=target_kl
    def update(self, current_kl):
        if current_kl > self.target*1.5: self.beta*=1.5
        elif current_kl < self.target/1.5: self.beta/=1.5
        return self.beta

# reference model 全程冻结
ref = copy.deepcopy(actor)
for p in ref.parameters(): p.requires_grad_(False)

# 一条 response 的 token 级 logp
logp_actor = torch.log_softmax(actor.logits(S),-1).gather(-1,A.unsqueeze(-1)).squeeze(-1)
logp_ref   = torch.log_softmax(ref.logits(S),-1).gather(-1,A.unsqueeze(-1)).squeeze(-1)

# reward KL penalty: per-token -β·log(π_θ/π_ref),detach(是 reward 信号)
reward_kl_penalty = -beta * (logp_actor - logp_ref).detach()
# reward = r_rm + reward_kl_penalty  (然后进 GAE)

# 训练时监控 approx_kl 并 early stop
with torch.no_grad():
    ratio=(logp_actor-logp_ref).exp()
    approx_kl=((ratio-1)-torch.log(ratio+1e-8)).mean().item()
if approx_kl > target_kl: break
```

> [!tip] 实现要点
> - reference model 必须全程冻结（`requires_grad_(False)`），否则 KL 惩罚失效；
> - reward KL penalty 的 `kl_per_token` 要 detach（它是 reward 信号，不参与 actor 反传；actor 反传走 PPO 的 ratio clipping 路径）；
> - adaptive β 用上一批的 KL 调本批的 β，避免本批耦合。

## 6. 与其他知识点的关系

- **上游**: [[PPO clipped objective]]（clip 外的第二道闸）、信息论（KL 散度）、TRPO（KL 约束的来源）、[[policy与value function]]。
- **下游**: RLHF（保 SFT 人格的核心机制）、[[SFT]]（$\pi_{ref}$ 来源）、verl/TRL/DeepSpeed-Chat 的 PPO 实现、[[DPO]]（DPO 是对此 KL-constrained 目标的闭式重写，隐式含此 KL）。
- **对比**:
  - **KL penalty vs target_kl early stop**：前者软惩罚（加项），后者硬阈值（break）。常并用。
  - **reward KL penalty（→$\pi_{ref}$） vs loss KL penalty（→$\pi_{old}$）**：约束目标不同，前者保人格、后者防漂移。
  - **KL penalty vs ratio clipping**：clip 是样本级硬截断，KL 是分布级软/硬约束，层次不同，互补。
  - **forward KL vs reverse KL**：见 §3.4，方向不同导致 mode-seeking 行为不同。
  - **KL penalty vs [[entropy bonus]]**：KL 拉 $\pi_\theta$ **靠近** $\pi_{ref}$（收敛），entropy 推 $\pi_\theta$ **远离**确定性（探索），方向相反但可协同。

## 7. 常见误区

1. **reference model 没冻结**：$\pi_{ref}$ 随训练变，KL 惩罚失效甚至反向。`requires_grad_(False)` + 不进 optimizer。
2. **reward KL penalty 没 detach**：`kl_per_token` 是 reward 信号，应 detach；若带 actor 梯度，actor 会去直接最小化 KL（与 PPO ratio 路径冲突，梯度紊乱）。
3. **KL 方向搞反**：reward penalty 用 reverse $\text{KL}(\pi_\theta\|\pi_{ref})$（per-token log-ratio），TRPO 早停用 forward $\text{KL}(\pi_{old}\|\pi_\theta)$（Schulman 近似）。混用方向得错误估计。
4. **fixed β 调死**：太小 KL 爆（policy collapse），太大训不动；优先用 adaptive controller。
5. **target_kl 设错量级**：游戏 RL 的 0.01 与 LLM-RL 的 per-token 0.001~0.01 / 累积 6~10 nats 完全不同量级，照搬会失效。
6. **approx_kl 估计不稳**：用 $\rho\log\rho$ 在 $\rho$ 大时数值差；用 Schulman 近似 $(\rho-1)-\log\rho$ 更稳。
7. **reward KL penalty 与 loss KL penalty 重复算 KL**：reward 里已扣 KL（→ref），loss 里又加 KL penalty（→old）→ 双重惩罚。一般 reward penalty 与 loss clip 分工不同不重复，但若都用 loss KL penalty 要避免叠加。
8. **KL 只在序列级算**：应是 token 级（per-token log-ratio），序列级丢细粒度信用分配。
9. **以为 KL penalty 一定能防 reward hacking**：KL 只防"漂离 ref"，不防"在 ref 附近钻空子"；reward hacking 还要靠 RM 设计 + 多样性约束。

## 8. LLM-RL 的 KL 系数典型值

- **reward KL penalty β**：InstructGPT 取 0.01~0.1（adaptive）；LLaMA-2/3 RLHF 常用更小（0.001~0.05）配合更长训练；
- **target_kl（早停/自适应）**：per-token 累积约 6~10 nats；per-token 约 0.001~0.01；
- **β 过大**：$\pi_\theta$ 被钉死在 $\pi_{ref}$，reward 信号训不动，RLHF 退化成 SFT；
- **β 过小**：$\pi_\theta$ 漂离 ref，语言能力崩、输出乱码（[[KL explosion]]）。

监控 KL 曲线诊断：KL 持续涨超 target → β/学习率太大或 reward 过强，policy 跑飞；KL 始终接近 0 → β 太大或学习率太小，训不动；KL 在 target 附近振荡 → 健康。

---
相关: [[PPO]]、[[PPO clipped objective]]、[[entropy bonus]]、[[policy collapse]]、[[KL explosion]]、[[reward hacking]]、[[SFT]]、[[DPO]]、[[policy与value function]]
