# KL penalty与KL control

> **所属章节**: [[PPO]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 中等偏难（PPO/RLHF 的"防跑飞"闸门）

## 1. 一句话定义

**KL penalty / KL control** 是 [[PPO clipped objective]] 中、除 ratio clipping 之外的**第二道策略约束机制**，用 **KL 散度** $\text{KL}(\pi_\theta\|\pi_{\text{ref/old}})$ 限制当前策略 $\pi_\theta$ 偏离参考策略 $\pi_{\text{ref}}$（RLHF 中是冻结的 SFT 模型）或采样快照 $\pi_{\theta_{old}}$ 的程度，防 [[policy collapse]]/[[KL explosion]]/[[reward hacking]]。它有两种实现形态：

- **KL penalty（软惩罚）**：把 $-\beta\,\text{KL}$ 加到 reward（per-token）或 loss 上，$\beta$ 是系数——"软"地劝阻漂移；
- **KL control（自适应/早停）**：用 adaptive controller 动态调 $\beta$ 维持 `target_kl`，或监控 `approx_kl` 超 `target_kl` 直接 break epoch 循环——"硬"阈值控制。

二者常并用：reward 里加 fixed/adaptive KL penalty（约束到 $\pi_{ref}$ 保 SFT 人格），训练时再用 target_kl 早停（约束到 $\pi_{\theta_{old}}$ 防单批漂移）。是 LLM-RL 稳定性的关键工程。

> [!note] 名字辨析
> "KL penalty" 指把 KL 作为惩罚项加进 reward/loss（OpenAI PPO paper 的 Adaptive KL Penalty 版本）；"KL control" 指用 KL 做控制信号（自适应调系数或早停）。二者常混用，本笔记统一用"KL 约束"指代这一类机制，具体形态见 §3。

## 2. 为什么需要它（动机与背景）

[[PPO clipped objective]] 的 ratio clipping 已经限制单步策略更新幅度，为什么还要 KL 约束？三重动机：

1. **clip 的局限**：clip 只在**单样本** $(s_t,a_t)$ 上限制 $\rho_t$，不约束**整批策略分布**的漂移。一批里每个样本都没触发 clip，但整体 KL 可能已很大——策略仍可能跑飞。KL 是**分布级**约束，补 clip 的盲区。
2. **多 epoch 复用的累积漂移**：PPO 做 $K$ epoch，每 epoch $\rho$ 偏一点，累积起来 $\pi_\theta$ 可能严重偏离 $\pi_{\theta_{old}}$。target_kl 早停在"累积漂移超阈"时 break，是 clip 之外的刹车。
3. **RLHF 的"SFT 人格保护"**：RLHF 用 reward model 训 $\pi_\theta$，但 RM 只在它见过的偏好数据上可靠。$\pi_\theta$ 若漂离 SFT $\pi_{ref}$ 太远，会落到 RM 分布外（OOD），reward 不可信、模型"跑飞"输出乱码或钻空子（[[reward hacking]]）。加 $-\beta\,\text{KL}(\pi_\theta\|\pi_{ref})$ 把 $\pi_\theta$ 软约束在 $\pi_{ref}$ 附近，保住语言能力与 SFT 人格——这是 RLHF 区别于普通 PPO 的核心约束。

理论根基是 **KL-constrained Policy Optimization**（Trust Region Policy Optimization, [[PPO clipped objective]] §4.3）：

$$
\max_\theta\,\mathbb{E}_{\pi_\theta}[r]\quad\text{s.t.}\quad \text{KL}(\pi_{\text{ref}}\|\pi_\theta)\le\delta
$$

带 KL 约束的策略改进有**单调性保证**（Kakade 2002, TRPO），是 PPO/RLHF 稳定性的理论底座。KL penalty 是把它从硬约束变成软惩罚（拉格朗日），KL control 是把它变成自适应/早停。

## 3. 核心概念详解

### 3.1 KL 散度（复习）

对两个离散分布 $p,q$：

$$
\text{KL}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}=\mathbb{E}_{x\sim p}\left[\log\frac{p(x)}{q(x)}\right]
$$

- $\text{KL}\ge0$，$=0$ 当且仅当 $p=q$；
- **非对称**：$\text{KL}(p\|q)\ne\text{KL}(q\|p)$。方向选择见 §3.5；
- 在 PPO 里 $p=\pi_\theta(\cdot|s_t)$（当前），$q=\pi_{\text{ref}}$ 或 $\pi_{\theta_{old}}$。

### 3.2 三种 KL 约束形态对比

| 形态 | 加在哪 | 约束到谁 | 系数 | 何时用 |
|---|---|---|---|---|
| **reward KL penalty** | per-token reward $r_t$ | $\pi_{\text{ref}}$（SFT） | $\beta$（fixed/adaptive） | RLHF 标配（OpenAI/InstructGPT） |
| **loss KL penalty** | PPO loss $L$ | $\pi_{\theta_{old}}$ | $\beta$（adaptive, Adaptive KL Penalty 版） | 原始 PPO paper 的备选目标 |
| **target_kl early stop** | 训练循环 | $\pi_{\theta_{old}}$（本批漂移） | target_kl 阈值 | 几乎所有 PPO 实现都加 |

> [!note] 两种"参考"的区别
> - $\pi_{\text{ref}}$：**冻结的 SFT/reference 模型**，整个 RLHF 训练全程不变，是"语言能力锚点"；
> - $\pi_{\theta_{old}}$：**当前 rollout 周期的采样快照**，每个 PPO 迭代重置，是"防本批漂移"。
>
> RLHF 的 reward KL penalty 用 $\pi_{\text{ref}}$（保人格），target_kl 早停用 $\pi_{\theta_{old}}$（防漂移）。两者作用不同，别混。

### 3.3 reward KL penalty（RLHF 主力）

把 KL 当成 per-token 的**负 reward shaping**：

$$
r_t\leftarrow r_t-\beta\,\text{KL}\bigl(\pi_\theta(\cdot|s_t)\big\|\pi_{\text{ref}}(\cdot|s_t)\bigr)
$$

- 这步 reward 是**期望形式**（对 $\pi_\theta$ 整个分布的 KL），实际用样本估计：$-\beta\log\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}$（per-token 的 log-ratio，是 KL 的单样本无偏估计）；
- 然后这个 $r_t$ 进 [[GAE]] 与 [[advantage function]]，让 actor 在算 advantage 时就**看到** KL 代价——选一个偏离 $\pi_{ref}$ 的 token 会被 KL 惩罚扣 reward；
- $\beta$ 是"保 SFT 人格的力度"，InstructGPT 取 $\beta\approx0.01\sim0.1$，adaptive controller 调。

**等价理解**：reward shaping $r-\beta\,\text{KL}$ 对应优化 $\mathbb{E}_{\pi_\theta}[r_{\text{RM}}]-\beta\,\text{KL}(\pi_\theta\|\pi_{ref})$，这正是 RLHF 的标准目标（max reward - β·KL）。DPO 直接对这个目标做闭式重写（[[DPO]]），故 DPO 隐式含此 KL penalty。

### 3.4 loss KL penalty（Adaptive KL Penalty 版）

原始 PPO paper 的备选目标（与 clipped surrogate 二选一）：

$$
L^{KPEN}=L^{IS}-\beta\,\mathbb{E}_t[\text{KL}(\pi_{\theta_{old}}\|\pi_\theta)]
$$

- 加在 surrogate loss 上，约束到**本批采样快照** $\pi_{\theta_{old}}$；
- $\beta$ 由 adaptive controller 调（§3.6）；
- 实际中**clipped 版（$L^{CLIP}$）效果更好且简单**，故 loss KL penalty 少用；但概念上它是"KL 软约束"的代表。

### 3.5 KL 方向：$\text{KL}(p\|q)$ vs $\text{KL}(q\|p)$

- **forward KL** $\text{KL}(\pi_{ref}\|\pi_\theta)$（$p=ref,q=\theta$）：mode-seeking，$\pi_\theta$ 要覆盖 $\pi_{ref}$ 所有 mode → $\pi_\theta$ 倾向**分散**（保多样性）；
- **reverse KL** $\text{KL}(\pi_\theta\|\pi_{ref})$（$p=\theta,q=ref$）：mode-seeking，$\pi_\theta$ 集中在 $\pi_{ref}$ 高概率区 → $\pi_\theta$ 倾向**集中**（可能 mode collapse）。

RLHF 的 reward KL penalty 通常用 **reverse KL** $\text{KL}(\pi_\theta\|\pi_{ref})$：$\pi_\theta$ 只在 $\pi_{ref}$ 已支持的区探索，不乱跑到 ref 低概率区（保语言合法）。这是 per-token log-ratio 估计 $\log\frac{\pi_\theta(a_t)}{\pi_{ref}(a_t)}$ 对应的方向（$\mathbb{E}_{a\sim\pi_\theta}[\log\frac{\pi_\theta}{\pi_{ref}}]$）。注意 TRPO 原文用 forward $\text{KL}(\pi_{old}\|\pi_\theta)$，方向不同——别混。

### 3.6 Adaptive KL controller

$\beta$ 固定难调：太小 KL 爆（policy 跑飞），太大 reward 被压死训不动。**adaptive controller** 动态调 $\beta$ 维持 `target_kl`：

```
if kl > target_kl * 1.5:  β *= 1.5      # KL 太大,加大惩罚
if kl < target_kl / 1.5:  β /= 1.5      # KL 太小,放松惩罚
```

- 维持 KL 在 target 附近，自适应训练强度；
- target_kl 是"想要的策略漂移量"，LLM-RL 约 6~10 nats/token 量级（per-token KL 累积），游戏 RL 约 0.01~0.02；
- InstructGPT/trl 的 `AdaptiveKLCoeff` 实现。

### 3.7 target_kl early stopping（最常用的"KL control"）

每个 PPO epoch 后算 `approx_kl`，超 `target_kl` 则 break：

```
for epoch in range(K_epochs):
    update(...)
    approx_kl = ((ratio - 1) - log(ratio)).mean()   # 或 ratio*log(ratio).mean()
    if approx_kl > target_kl: break
```

- `approx_kl = ((ratio-1)-log(ratio)).mean()` 是 $\text{KL}(\pi_{old}\|\pi_\theta)$ 的近似（Schulman 提的稳定估计，比 $\rho\log\rho$ 数值稳）；
- target_kl 游戏约 0.01~0.02，LLM-RL 约 0.001~0.01（per-token，更保守）；
- 这是 PPO 实现里几乎必加的"第二道闸"，见 [[PPO clipped objective]] §3.7、§8.2。

### 3.8 估计 KL 的几种方式

| 估计式 | 形式 | 对应 KL | 稳定性 |
|---|---|---|---|
| Schulman 近似 | $\mathbb{E}[(\rho-1)-\log\rho]$ | $\text{KL}(\pi_{old}\|\pi_\theta)$ | 数值稳（推荐） |
| 直接 | $\mathbb{E}[\rho\log\rho]$ | $\text{KL}(\pi_{old}\|\pi_\theta)$ | $\rho$ 大时数值差 |
| per-token log-ratio | $\mathbb{E}_{a\sim\pi_\theta}[\log\frac{\pi_\theta(a)}{\pi_{ref}(a)}]$ | $\text{KL}(\pi_\theta\|\pi_{ref})$ | reward penalty 用 |

$\rho=\pi_\theta/\pi_{\theta_{old}}$。Schulman 近似来自 $x-1-\log x\ge0$ 且小扰动下 $x-1-\log x\approx(x-1)^2/2\approx\text{KL}$，对 $\rho$ 不敏感故稳。

## 4. 数学原理 / 公式

### 4.1 KL-constrained PO 的拉格朗日

$$
\max_\theta\,\mathbb{E}_{\pi_\theta}[r]\quad\text{s.t.}\quad\text{KL}(\pi_{ref}\|\pi_\theta)\le\delta
$$

拉格朗日松弛：$\max_\theta\,\mathbb{E}_{\pi_\theta}[r]-\beta\,\text{KL}(\pi_{ref}\|\pi_\theta)$，$\beta\ge0$ 是拉格朗日乘子。这就是 reward/loss KL penalty 的来源——把硬约束转成软惩罚，$\beta$ 是约束的"价格"。

### 4.2 reward KL penalty 的等价目标

reward shaping $r_t\leftarrow r_t-\beta\,\text{KL}(\pi_\theta(\cdot|s_t)\|\pi_{ref}(\cdot|s_t))$ 后，期望累积目标：

$$
J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_t\gamma^t\bigl(r_t^{\text{RM}}-\beta\,\text{KL}(\pi_\theta(\cdot|s_t)\|\pi_{ref})\bigr)\right]
=\mathbb{E}_{\pi_\theta}[R_{\text{RM}}]-\beta\,\mathbb{E}_{\pi_\theta}\left[\sum_t\gamma^t\,\text{KL}(\pi_\theta\|\pi_{ref})\right]
$$

（$\gamma$ 与 $\sum_t$ 的细节取决于 episodic/continuing 约定，常见简化为 $\gamma=1$ 累加到序列末尾）。这正是 RLHF 的标准目标 **max reward - β·KL**，与 DPO 推导的出发点一致（[[DPO]]）。

### 4.3 per-token log-ratio 是 reverse KL 的无偏估计

$$
\mathbb{E}_{a\sim\pi_\theta}\left[\log\frac{\pi_\theta(a|s)}{\pi_{ref}(a|s)}\right]=\text{KL}(\pi_\theta(\cdot|s)\|\pi_{ref}(\cdot|s))
$$

故 reward 里 $-\beta\log\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$ 是 $-\beta\,\text{KL}(\pi_\theta\|\pi_{ref})$ 的单样本无偏估计。这解释了 RLHF 实现里 reward penalty 为何是 per-token log-ratio。

### 4.4 Adaptive β 的不动点直觉

固定 $\beta$ 时，KL 随训练漂移：$\beta$ 小则 KL 大（漂移大），$\beta$ 大则 KL 小。adaptive controller 调 $\beta$ 让 KL≈target_kl，相当于在拉格朗日里找使约束紧（active）的乘子——$\beta$ 是"约束价格的均衡值"。

### 4.5 与 ratio clipping 的互补

- **clip**：单样本 $(s_t,a_t)$ 上的硬截断，约束 $\rho_t\in[1-\epsilon,1+\epsilon]$；
- **KL penalty**：分布级软惩罚，约束 $\mathbb{E}[\text{KL}]$；
- **target_kl early stop**：整批漂移的硬阈值。

三者层次：clip（样本级）→ KL penalty（分布级软）→ target_kl（批级硬）。三者并用最稳，是 RLHF-PPO 的标准配置。

## 5. 代码示例

```python
import torch, torch.nn as nn, copy

# adaptive KL controller(trl/InstructGPT 风格)
class AdaptiveKLController:
    def __init__(self, init_beta=0.1, target_kl=6.0, horizon=10000):
        self.beta=init_beta; self.target=target_kl
    def update(self, current_kl):
        # 简化版:KL 超目标则加大 β,低于则减小
        if current_kl > self.target*1.5: self.beta*=1.5
        elif current_kl < self.target/1.5: self.beta/=1.5
        return self.beta

kl_ctrl=AdaptiveKLController(init_beta=0.1, target_kl=6.0)

# 假设:actor 与 reference(SFT 冻结)都给出 token logits
class LM(nn.Module):
    def __init__(s,v,n): super().__init__(); s.l=nn.Linear(v,n)
    def logits(s,x): return s.l(x)
actor=LM(8,5); ref=copy.deepcopy(actor)
for p in ref.parameters(): p.requires_grad_(False)       # reference 冻结

# 一条 response 的 token 级 logp
S=torch.randn(10,8); A=torch.randint(0,5,(10,))
logp_actor= torch.log_softmax(actor.logits(S),-1).gather(-1,A.unsqueeze(-1)).squeeze(-1)
logp_ref   = torch.log_softmax(ref.logits(S),-1).gather(-1,A.unsqueeze(-1)).squeeze(-1)

# 1) reward KL penalty:per-token -β·log(π_θ/π_ref)
beta=kl_ctrl.beta
kl_per_token=(logp_actor-logp_ref).detach()
reward_kl_penalty=-beta*kl_per_token            # 加到原始 reward 上
# reward = r_rm + reward_kl_penalty  (然后进 GAE)

# 2) 训练时监控 approx_kl 并 early stop
with torch.no_grad():
    ratio=(logp_actor-logp_ref).exp()           # ρ = π_θ/π_old(此处用 ref 当 old 演示)
    approx_kl=((ratio-1)-torch.log(ratio+1e-8)).mean().item()
print("approx_kl:", approx_kl, "| beta:", beta)
kl_ctrl.update(approx_kl)                       # 自适应调 β

# 3) early stop 逻辑
target_kl=6.0
# for epoch in range(K):
#     ...update...
#     if approx_kl > target_kl: break
```

> [!tip] 实现要点
> - reference model 必须全程冻结（`requires_grad_(False)`），否则 KL 惩罚失效；
> - reward KL penalty 的 `kl_per_token` 要 detach（它是 reward 信号，不参与 actor 反传；actor 反传走 PPO 的 ratio clipping 那条路）；
> - adaptive β 用上一批的 KL 调本批的 β，避免本批耦合。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[PPO clipped objective]]（clip 之外的第二道闸）、信息论（KL 散度定义）、TRPO（KL 约束的来源）、[[policy与value function]]（策略分布）。
- **下游（应用）**: RLHF（保 SFT 人格的核心机制）、[[SFT]]（$\pi_{ref}$ 的来源）、verl/TRL/DeepSpeed-Chat 的 PPO 实现、[[DPO]]（DPO 是对此 KL-constrained 目标的闭式重写，隐式含此 KL）。
- **对比 / 易混**:
  - **KL penalty vs target_kl early stop**：前者软惩罚（加项），后者硬阈值（break）。常并用。
  - **reward KL penalty（→$\pi_{ref}$） vs loss KL penalty（→$\pi_{old}$）**：约束目标不同，前者保人格、后者防漂移。
  - **KL penalty vs ratio clipping**：clip 是样本级硬截断，KL 是分布级软/硬约束，层次不同，互补。
  - **forward KL vs reverse KL**：见 §3.5，方向不同导致 mode-seeking 行为不同。
  - **KL penalty vs [[entropy bonus]]**：KL 拉 $\pi_\theta$ **靠近** $\pi_{ref}$（收敛方向），entropy 推 $\pi_\theta$ **远离**确定性（探索方向），方向相反但可协同。
  - **adaptive β vs fixed β**：前者维持 target_kl 自适应，后者固定难调。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **reference model 没冻结**：$\pi_{ref}$ 随训练变，KL 惩罚失效甚至反向。`requires_grad_(False)` + 不进 optimizer。
> 2. **reward KL penalty 没 detach**：`kl_per_token` 是 reward 信号，应 detach；若带 actor 梯度，actor 会去直接最小化 KL（与 PPO ratio 路径冲突，梯度紊乱）。
> 3. **KL 方向搞反**：reward penalty 用 reverse $\text{KL}(\pi_\theta\|\pi_{ref})$（per-token log-ratio），TRPO 早停用 forward $\text{KL}(\pi_{old}\|\pi_\theta)$。混用方向会得错误估计。
> 4. **fixed β 调死**：太小 KL 爆（policy collapse），太大训不动；优先用 adaptive controller。
> 5. **target_kl 设错量级**：游戏 RL 的 0.01 与 LLM-RL 的 6 nats 完全不同量级，照搬会失效。LLM-RL 是 per-token 累积，量级大。
> 6. **approx_kl 估计不稳**：用 $\rho\log\rho$ 在 $\rho$ 大时数值差；用 Schulman 近似 $(\rho-1)-\log\rho$ 更稳。
> 7. **reward KL penalty 与 PPO ratio 路径重复算 KL**：reward 里已扣 KL，loss 里又加 KL penalty → 双重惩罚。一般 reward penalty（→ref）与 loss clip（→old）分工不同不重复，但若都用 loss KL penalty 要避免叠加。
> 8. **KL 只在序列级算**：应是 **token 级**（per-token log-ratio），序列级（整段 KL）丢细粒度信用分配。
> 9. **以为 KL penalty 一定能防 reward hacking**：KL 只防"漂离 ref"，不防"在 ref 附近钻空子"；reward hacking 还要靠 RM 设计 + 多样性约束。
> 10. **β 自适应耦合本批**：应用上一批 KL 调本批 β，避免本批 KL 与 β 互相耦合（数值不稳）。

## 8. 延伸细节

### 8.1 LLM-RL 的 KL 系数典型值

- **reward KL penalty β**：InstructGPT 取 0.01~0.1（adaptive）；LLaMA-2/3 RLHF 常用更小（0.001~0.05）配合更长训练；
- **target_kl（早停/自适应）**：per-token 累积约 6~10 nats；per-token 约 0.001~0.01；
- **β 过大**：$\pi_\theta$ 被钉死在 $\pi_{ref}$，reward 信号训不动，RLHF 退化成 SFT；
- **β 过小**：$\pi_\theta$ 漂离 ref，语言能力崩、输出乱码（[[KL explosion]]）。

### 8.2 KL constraint 与 DPO 的关系

DPO（Rafailov 2023）证明：RLHF 目标 $\max_\theta\mathbb{E}[r_{RM}]-\beta\,\text{KL}(\pi_\theta\|\pi_{ref})$ 的**最优解有闭式**：

$$
\pi^*(a|s)=\frac{1}{Z(s)}\pi_{ref}(a|s)\exp\!\left(\frac{1}{\beta}r(s,a)\right)
$$

反解出 $r(s,a)=\beta\log\frac{\pi^*(a|s)}{\pi_{ref}(a|s)}+\beta\log Z(s)$，代入偏好数据（Bradley-Terry），得 DPO loss（无 RL、无 rollout、无 clip、无 KL penalty 显式项——但**隐式含此 KL 约束**）。故 DPO 是对"reward - β·KL"目标的闭式重写，KL 系数 $\beta$ 直接对应 DPO 的 $\beta$。详见 [[DPO]]。

### 8.3 verl / TRL 的 KL 实现

- **verl**：rollout 时同时算 actor 与 ref 的 token logp，reward = r_rm - β·(logp_actor - logp_ref)，进 GAE；训练时另算 approx_kl 做 early stop；
- **TRL（PPOTrainer）**：`AdaptiveKLCoeff` 类自动调 β，每批用 KL 调整；reference model 用 `AutoModelForCausalLM` 冻结；
- **DeepSpeed-Chat**：类似，reward 里扣 per-token KL penalty。

### 8.4 KL constraint 与 trust region 的理论

TRPO 的核心定理（Kakake 2002 / Schulman 2015）：若新策略 $\tilde\pi$ 满足 $\text{KL}(\pi_{old}\|\tilde\pi)\le\delta$ 且 advantage 在某误差 $\epsilon$ 内估准，则真实性能改进下界 $\eta(\tilde\pi)\ge\eta(\pi_{old})-\frac{4\epsilon\gamma\delta}{(1-\gamma)^2}$——即**有 KL 约束则性能不会差太多**。这是 PPO/RLHF 敢做多 epoch 复用 + 多轮迭代的底气。clip 是它的近似实现，KL penalty/early stop 是它的直接实现。

### 8.5 KL 约束与 entropy 的协同

- **KL penalty**：把 $\pi_\theta$ 拉**向** $\pi_{ref}$（保人格、防 OOD）；
- **[[entropy bonus]]**：把 $\pi_\theta$ 推**离**确定性分布（保探索、防 mode collapse）。

二者方向相反但目标不同：KL 防"跑离 ref"，entropy 防"塌缩到一点"。LLM-RL 中常 KL penalty 开、entropy bonus 关或小系数（因已有 ref 锚点，探索需求弱；且 LLM 分布本就多 mode）。游戏 RL 反之：entropy bonus 大、无 ref（KL 用 $\pi_{old}$ 早停）。

### 8.6 KL 监控的诊断价值

训练中看 KL 曲线：
- KL 持续涨超 target：β/学习率太大或 reward 过强，policy 跑飞；
- KL 始终接近 0：β 太大或学习率太小，训不动（退化 SFT）；
- KL 在 target 附近振荡：健康，adaptive controller 工作。

配合 clip_fraction、reward、response length 一起看，是 RLHF 调参的核心诊断（见 [[PPO clipped objective]] §8.3）。

---
相关: [[PPO]]、[[PPO clipped objective]]、[[entropy bonus]]、[[policy collapse]]、[[KL explosion]]、[[reward hacking]]、[[SFT]]、[[DPO]]、[[policy与value function]]、[[reward normalization]]
