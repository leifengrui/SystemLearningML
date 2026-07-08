# RLHF 训练监控指标详解

> **所属章节**: [[常见指标]] / 性能分析与系统优化
> **所属模块**: [[10-性能分析与系统优化]]（兼跨 [[05-LLM-RL对齐]]、[[06-训推系统]]）
> **别名**: verl 训练日志指标、PPO 仪表盘、RLHF metrics dashboard


## 1. 一句话定义

这是 **RLHF/PPO 训练日志里那串带斜杠的 key**（如 `actor/pg_loss`、`critic/rewards/mean`），它们是 **verl / OpenRLHF 等 RLHF 框架在每一步训练里打出来的监控指标**，前缀（`actor/`、`critic/`、`training/`、`response_length/`）标明指标归属哪个子系统，后缀是具体量。看懂这串 key = 能读懂训练是否健康、有没有 reward hacking、策略有没有跑飞。

用户在 [[常见指标]] 里列出的清单（去重后）：

```
critic/rewards/mean
critic/advantages/mean
actor/pg_loss
actor/grad_norm
actor/kl_loss
actor/pg_clipfrac
actor/ppo_kl
response_length/mean
training/rollout_actor_probs_pearson_corr
```

> [!note] 命名约定
> verl 用 `子系统/量名` 的两级（有时三级）路径命名 metric，对应 TensorBoard / WandB 的 group + tag。前缀含义：
> - `actor/` — 策略模型（actor / policy $\pi_\theta$）相关的 loss / 梯度 / ratio；
> - `critic/` — 价值模型（critic / value $V_\phi$）相关的 reward / advantage；
> - `training/` — 训练全局统计（rollout 一致性、步数等）；
> - `response_length/` — 生成端（rollout）的响应长度统计。
> 下面逐条拆。

## 2. 为什么需要它（动机与背景）

RLHF 训练**极容易跑飞**，远比 SFT 难盯：
1. **多个模型同训**：actor + critic + reward model + reference，任一个出问题都连锁；
2. **reward 不可微且可能被钻空子**：[[reward hacking]]——actor 学会刷分而非真对齐；
3. **策略漂移失控**：PPO 一步更新太大，$\pi_\theta$ 瞬间远离 $\pi_{ref}$，生成崩成乱码（[[KL explosion]]、[[policy collapse]]）；
4. **异步/stale 数据**（[[asynchronous training]]）：ratio 失真，需监控修正质量；
5. **长序列生成**：response 变长 → 算力爆炸 / 退化成重复（[[mode collapse]]）。

SFT 时代看一个 `loss` 就够；RLHF 必须看**一整套仪表盘**才能定位"到底哪坏了"。这套 metric 就是仪表盘的刻度。**不会读它们 = 训练崩了也不知道为什么崩。**

## 3. 核心概念详解（逐条指标）

下面按"它是什么 / 怎么算 / 看什么 / 坏了说明什么"四段式逐条展开。

---

### 3.1 `critic/rewards/mean` —— 平均奖励

- **是什么**：一个 batch 内所有 response 由 **reward model**（或规则/代码评测器）打出的 reward 的**均值**。
- **怎么算**：$r = R_\omega(\text{prompt},\text{response})$（或 rule-based 0/1），对 batch 取 mean。
- **看什么**：
  - **整体趋势应单调上升**（RL 在学怎么拿更高分）；
  - 但**不是越高越好**——若突然飙高而 `actor/ppo_kl` 也飙高，多半是 [[reward hacking]]（钻 reward 漏洞而非真变好）；
  - 若**一直不动**：reward 信号太弱 / critic 评价无区分度 / 学习率太小 / advantage 全被归一化抹平。
- **坏了说明什么**：reward 爆炸 → reward hacking；reward 停滞 → 训练没学到东西或 reward shaping 失败。

> [!warning] 误区
> "`critic/rewards/mean` 上升 = 模型变好"**不一定**。必须和 `actor/ppo_kl`、`response_length/mean` 联看：reward 涨 + KL 涨 + 长度暴涨 = 典型 hacking（学会长篇大论骗分）。

---

### 3.2 `critic/advantages/mean` —— 平均优势

- **是什么**：一个 batch 内 **advantage $A(s,a)$** 的均值。advantage = "这个动作比平均好多少"。
- **怎么算**：典型用 [[GAE]]：$A_t=\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}$，其中 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ 是 TD 误差。RLHF 里常再做**组内 / batch 内归一化**（GRPO 风格：同 prompt 多采样后 z-score）。
- **看什么**：
  - **均值应≈0**（advantage 设计上就是"减 baseline 后的相对值"，期望 0）；
  - 偏离 0 很多 → baseline（critic）估偏了 / 归一化有 bug；
  - **方差**比均值更重要：方差太小 → 信号弱、学不动；方差大 → 高方差、训练抖。
- **坏了说明什么**：长期非零 → critic 跟不上 actor；方差塌缩 → 探索死亡（[[mode collapse]] 前兆）。

> [!note] 为什么 mean≈0 是健康的
> advantage 本质是 $A=Q-V$，而 $V$ 是 $Q$ 的条件期望（在最优 critic 下），故 $\mathbb{E}[A]=\mathbb{E}[Q]-\mathbb{E}[V]=0$。mean 显著非零说明 $V$ 估偏——critic 没训好。详见 [[advantage function]]、[[advantage estimation]]。

---

### 3.3 `actor/pg_loss` —— 策略梯度损失

- **是什么**：actor 的**主损失**，PPO clipped surrogate 的负值（要最小化它 = 最大化策略目标）。
- **怎么算**：
  $$
  \mathcal{L}_{\text{pg}}=-\mathbb{E}_t\!\Big[\min\big(\rho_t A_t,\;\text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\big)\Big],\quad \rho_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
  $$
  即 [[PPO clipped objective]] 取负号。
- **看什么**：
  - **应缓慢下降**（在 advantage 有界的前提下）；
  - **不能为负且持续暴跌**——若 $\mathcal{L}_{\text{pg}}$ 越来越负 = 目标越最大化越快 = 策略在"捷径式"刷分，常伴 KL 爆炸；
  - 突然跳变 → 梯度爆炸 / clip 大量触发 / 数据分布突变。
- **坏了说明什么**：单调暴跌 → reward hacking / KL 失控；剧烈抖动 → 学习率太大 / batch 太小 / 高方差。

> [!warning] 误区
> "`pg_loss` 下降 = 训练好"**错**。PPO 的 pg_loss 下降常意味着策略在沿 advantage 方向猛冲，**过快下降反而是危险信号**——必须配 KL 一起看。健康训练里 pg_loss 下降是**温和、伴随 KL 在 target 以内**的。

---

### 3.4 `actor/grad_norm` —— 梯度范数

- **是什么**：actor 反向传播后、**裁剪前**（或裁剪后，看框架）的梯度总范数 $\|\nabla_\theta\mathcal{L}\|$。
- **怎么算**：`torch.nn.utils.clip_grad_norm_` 返回的 total_norm，通常 $L_2$ 范数 $\sqrt{\sum_i\|g_i\|_2^2}$。
- **看什么**：
  - **稳定在一个量级**（健康）；
  - **突然飙高** → 梯度爆炸（[[gradient explosion and vanishing]]），多半是 loss spike / stale ratio 失真 / 数值溢出；
  - **趋近 0** → 梯度消失，学不动；
  - clip 阈值（max_grad_norm，典型 1.0）触发频率 = 不稳定程度。
- **坏了说明什么**：spike → 立刻查 KL、reward 是否异常；归零 → 学习率 / 梯度裁剪阈值 / advantage 归一化是否有问题。

> [!tip] 实践
> 同时记 **clip 前后两个 grad_norm**，二者比值反映"被裁掉了多少"。若几乎每步都裁到阈值 → 说明梯度常年爆炸，该降学习率或查数据。详见 [[梯度裁剪]]。

---

### 3.5 `actor/kl_loss` —— KL 损失项

- **是什么**：actor loss 里**显式**的 KL 正则项（注意区别于 §3.7 的 `actor/ppo_kl`）。
- **怎么算**：两种主流形式：
  1. **对参考模型的 KL**（OpenAI InstructGPT 式）：$\mathcal{L}_{\text{kl}}=\beta\,\mathbb{E}\big[D_{\mathrm{KL}}(\pi_\theta\|\pi_{ref})\big]$，$\beta$ 自适应（[[KL penalty与KL control]]）；
  2. **KL to reference logp（verl 常见，逐 token）**：$\mathcal{L}_{\text{kl}}=\beta\,\frac{1}{T}\sum_t\big(\log\pi_\theta(a_t)-\log\pi_{ref}(a_t)\big)\cdot\ldots$（具体形式见框架，本质都是限制 $\pi_\theta$ 别离 $\pi_{ref}$ 太远）。
  把它**加进总 loss**：$\mathcal{L}=\mathcal{L}_{\text{pg}}+\mathcal{L}_{\text{kl}}+\mathcal{L}_{\text{entropy}}+\ldots$
- **看什么**：
  - **应维持小而稳**——说明 $\pi_\theta$ 没跑离 $\pi_{ref}$；
  - **持续上涨** → 策略在远离参考模型，reward hacking / 漂移风险；
  - 配合自适应 $\beta$：KL 涨 → $\beta$ 自动加大 → 把策略拉回。
- **坏了说明什么**：KL_loss 失控增长 → 即将 [[KL explosion]]，需降学习率 / 加大 KL 系数 / 早停。

> [!note] KL 是什么
> KL = Kullback–Leibler divergence（相对熵），$D_{\mathrm{KL}}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}$，衡量两分布差异，非负、$=0\iff P=Q$、不对称。在 RLHF 里是"策略漂移量"的标尺。详见 [[KL penalty与KL control]]、[[KL divergence control]]。

---

### 3.6 `actor/pg_clipfrac` —— ratio 被裁比例

- **是什么**：batch 内 PPO ratio $\rho_t$ 触发 clip（即 $\rho_t$ 落到 $[1-\epsilon,1+\epsilon]$ 边界外、被截断）的**比例**。
- **怎么算**：$\text{clipfrac}=\frac{1}{N}\sum_t\mathbb{1}[\rho_t\notin[1-\epsilon,1+\epsilon]]$，典型 $\epsilon=0.2$。
- **看什么**：
  - **健康值很低**（< 5%–10%）：策略更新幅度在信赖域内；
  - **飙高**（> 30%）：策略一步走太远 / 数据 stale（[[asynchronous training]]）/ 学习率太大；
  - **长期高位** → PPO 的 clip 把大部分梯度截断了 → 有效学习信号弱 → 训练停滞。
- **坏了说明什么**：clipfrac 高 = 单步更新过大或 stale 严重。异步训练里这是 staleness 的**直接仪表**（详见 [[stale policy problem]]）。

> [!tip] 联动
> `pg_clipfrac` ↑ + `ppo_kl` ↑ → 策略漂移过快，降学习率 / 降 staleness_threshold（[[weight sync mechanism]]）；`pg_clipfrac` ↑ 但 `ppo_kl` 正常 → 多半是 stale data 致 ratio 失真，查异步配置。

---

### 3.7 `actor/ppo_kl` —— PPO 实际 KL

- **是什么**：当前策略 $\pi_\theta$ 与**采样时策略** $\pi_{\theta_{old}}$ 的**实际 KL**（注意：不是对 ref 的，是对"本 batch 采样的那个旧策略"的）。
- **怎么算**：$\mathrm{KL}=\mathbb{E}_t\big[\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{old}}(a_t|s_t)\big]$（近似，逐 token 平均）。
- **看什么**：
  - **应 < target_kl**（典型 0.001–0.1）。超 target → 触发**早停**（break 后续 epoch）；
  - 这是 PPO "信赖域"的**主开关**：超 target_kl 就停，防止单 batch 把策略改太狠；
  - 异步训练里 stale 让 `ppo_kl` 天然偏大，需更严的 target_kl 或更勤的 weight sync。
- **坏了说明什么**：长期超 target_kl → 学习率太大 / clip 失效 / stale 严重；训崩前兆。

> [!warning] 区分三个 KL
> RLHF 里有**三个 KL 容易混**：
> | 指标 | 对谁 | 用途 |
> |---|---|---|
> | `actor/ppo_kl` | $\pi_\theta$ vs $\pi_{\theta_{old}}$（采样策略） | PPO 单 batch 信赖域 / 早停 |
> | `actor/kl_loss` | $\pi_\theta$ vs $\pi_{ref}$（参考/SFT 模型） | 软约束别离原始对齐模型太远 |
> | （监控用）`kl_ref` | $\pi_\theta$ vs $\pi_{ref}$ | 整体漂移监控，决定自适应 $\beta$ |
> 三者数值量级不同、用途不同，看日志别搞混。

---

### 3.8 `response_length/mean` —— 平均响应长度

- **是什么**：rollout 阶段生成的 response token 数的均值。
- **怎么算**：$\bar{L}=\frac{1}{N}\sum_i \text{len}(\text{response}_i)$。
- **看什么**：
  - **应与任务匹配**：数学题该长则长，闲聊该短则短；
  - **暴涨** → reward 偏好长文 → actor 学会注水（reward hacking 的经典形态）；常伴 `rewards/mean`↑、`ppo_kl`↑；
  - **塌缩到极短** → actor 学会"躺平"输出 eos 骗 reward / 探索死亡；
  - **方差**也重要：方差爆炸 → 长度极不均，batch 算力浪费、padding 严重。
- **坏了说明什么**：长度异常是 reward hacking / mode collapse 的**头号表征**，比 reward 本身更早暴露。

> [!tip] 实践
> 配合 `response_length/max`、`response_length/std` 一起看。若 max 远超 mean（个别极长）→ 有 runaway 生成，需加长度上限 / 改 reward shaping（按长度归一化）。

---

### 3.9 `training/rollout_actor_probs_pearson_corr` —— rollout 与 actor 概率一致性

- **是什么**：这是 verl 里一个**关键健康度自检指标**——衡量"rollout 采样时用的策略"与"训练时 actor 算 logp 用的策略"是否**一致**，用 **Pearson 相关系数**度量。
- **怎么算**：
  - rollout 阶段记录每个 token 的 $\log\pi_{\text{rollout}}(a_t)$（采样策略的 logp）；
  - 训练阶段用当前 actor 重算 $\log\pi_\theta(a_t)$；
  - 对一批 token 算两者序列的 **Pearson 相关系数** $\rho_{\text{pearson}}$。
- **看什么**：
  - **严格 on-policy 同步训练**：理论上两策略同版本 → $\rho_{\text{pearson}}\approx 1.0$；
  - **异步 / stale 训练**：两策略版本不同 → $\rho$ 略降但仍应 $>0.9$；
  - **若 $\rho$ 明显下降**（< 0.8 或暴跌）→ 严重 stale / weight sync 没生效 / 甚至 rollout 与 train 用了不同模型 → **数据基本不可用**，PPO ratio 全失真。
- **坏了说明什么**：这是异步训练的"canary in the coal mine"（金丝雀预警）。$\rho$ 跌 = stale 失控，立刻查 [[weight sync mechanism]]、staleness_threshold、是否 partial_rollout 配错。

> [!note] 为什么用 Pearson
> KL 是分布级差异，Pearson 是"逐 token logp 序列"的线性相关。Pearson 计算便宜、对 scale 不敏感、对"整体趋势是否一致"敏感，适合做工程健康度快检。它 $\ne$ KL，但**两者高相关**：Pearson 跌通常 KL 涨。详见 [[asynchronous training]] §3.3、[[stale policy problem]]。

---

## 4. 数学原理 / 公式

### 4.1 PPO 总目标（这些 loss 的源头）

$$
\mathcal{L}_{\text{actor}}=\underbrace{-\mathbb{E}_t[\min(\rho_t A_t,\,\text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)]}_{\text{pg\_loss}}+\underbrace{\beta\,\mathbb{E}_t[D_{\mathrm{KL}}(\pi_\theta\|\pi_{ref})]}_{\text{kl\_loss}}-\underbrace{c_e\,\mathbb{E}[\mathcal{H}(\pi_\theta)]}_{\text{entropy bonus}}
$$

- `pg_loss` ← 第一项取负；
- `kl_loss` ← 第二项；
- `pg_clipfrac` ← 第一项里 clip 触发的比例；
- `ppo_kl` ← $\mathbb{E}[\log\pi_\theta-\log\pi_{\theta_{old}}]$（对 $\theta_{old}$ 不是 ref）。

### 4.2 advantage 与 reward 的关系

$$
A_t=\underbrace{r_t}_{\text{→ rewards/mean}}+\gamma V(s_{t+1})-V(s_t)+\text{GAE 多步累加}
$$

- `rewards/mean` 是 $r$ 的均值；
- `advantages/mean` 是 $A$ 的均值，最优 critic 下 $\mathbb{E}[A]=0$。

### 4.3 Pearson 相关（§3.9）

$$
\rho_{\text{pearson}}=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y},\quad X=\log\pi_{\text{rollout}}(a_t),\;Y=\log\pi_\theta(a_t)
$$

$\rho\in[-1,1]$，越接近 1 越一致。健康 RLHF 应 $\rho\to 1$。

### 4.4 stale 下 ratio 与 Pearson 的联动

异步训练（[[asynchronous training]]）：worker 用 $\theta_t$ 采，learner 在 $\theta_{t+k}$ 训。

$$
\rho_t=\frac{\pi_{\theta_{t+k}}(a_t)}{\pi_{\theta_t}(a_t)},\quad \text{stale 越重}\;\Rightarrow\;\rho_t\text{ 越偏离 1}\;\Rightarrow\;\text{clipfrac↑、ppo_kl↑、pearson↓}
$$

故 `pg_clipfrac`、`ppo_kl`、`pearson_corr` 三者是 **staleness 的三联仪表**。

## 5. 代码示例（verl 风格最小复现）

```python
import torch
import torch.nn.functional as F

# --- 假设已有: logp_old (采样时), logp_cur (当前actor), logp_ref (参考模型),
#              advantages, rewards, response_lengths, values ---
EPS = 0.2

def compute_rlhf_metrics(logp_old, logp_cur, logp_ref, advantages, rewards,
                         response_lengths, values, beta=0.04):
    """返回一个 dict, key 命名仿 verl。所有张量 shape [N, T] 或 [N]。"""
    metrics = {}

    # ---- reward / advantage ----
    metrics["critic/rewards/mean"] = rewards.mean().item()
    metrics["critic/advantages/mean"] = advantages.mean().item()

    # ---- PPO ratio & pg_loss ----
    ratio = (logp_cur - logp_old).exp()                       # ρ_t
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * advantages
    pg_obj = torch.min(surr1, surr2)                          # PPO clipped objective
    metrics["actor/pg_loss"] = (-pg_obj).mean().item()       # 取负 = 要最小化

    # ---- clipfrac ----
    clipped = (ratio < 1 - EPS) | (ratio > 1 + EPS)
    metrics["actor/pg_clipfrac"] = clipped.float().mean().item()

    # ---- ppo_kl: 对采样策略 θ_old ----
    metrics["actor/ppo_kl"] = (logp_cur - logp_old).mean().item()

    # ---- kl_loss: 对参考模型 ref ----
    kl_ref = (logp_cur - logp_ref).mean()                    # 近似 D_KL(π_θ||π_ref)
    metrics["actor/kl_loss"] = (beta * kl_ref).item()

    # ---- grad_norm: 反向后、裁剪前 ----
    # (假设 loss 已 backward, 这里读 .grad)
    # total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in params]), 2)
    # metrics["actor/grad_norm"] = total_norm.item()
    # (此处略, 需在训练循环里 backward 之后调 clip_grad_norm_ 拿返回值)

    # ---- response length ----
    metrics["response_length/mean"] = response_lengths.float().mean().item()

    # ---- rollout vs actor 一致性 (Pearson) ----
    x = (logp_old - logp_old.mean()).flatten()
    y = (logp_cur - logp_cur.mean()).flatten()
    pearson = (x * y).sum() / (x.norm() * y.norm() + 1e-8)
    metrics["training/rollout_actor_probs_pearson_corr"] = pearson.item()

    return metrics

# --- 使用 ---
# metrics = compute_rlhd_metrics(...)
# for k, v in metrics.items():
#     wandb.log({k: v})   # 或 tensorboard
```

> [!tip] 实际工程
> verl 在 `verl/trainer/ppo/` 下集中打这些 metric，每 step 自动写 TensorBoard / WandB。生产里**别只看单点值**，看**滑动平均 + 联动**：reward↑ 但 kl↑+length↑ = hacking；clipfrac↑+pearson↓ = stale 失控。

## 6. 与其他知识点的关系

- **上游（依赖）**:
  - [[PPO clipped objective]] — `pg_loss`/`pg_clipfrac`/`ppo_kl` 的算法源头；
  - [[GAE]] — `advantages/mean` 的计算；
  - [[KL penalty与KL control]] / [[KL divergence control]] — `kl_loss`/`ppo_kl` 的控制逻辑；
  - [[advantage function]] / [[advantage estimation]] — advantage 为何 mean≈0；
  - [[asynchronous training]] / [[stale policy problem]] / [[weight sync mechanism]] — `pg_clipfrac`/`pearson_corr` 的 stale 语义。
- **下游（应用）**:
  - RLHF 训练调参（看仪表盘决定调学习率 / KL 系数 / weight sync 频率）；
  - reward hacking / KL explosion / policy collapse 的**早期预警**（[[reward hacking]]、[[KL explosion]]、[[policy collapse]]、[[mode collapse]]）；
  - 异步系统 staleness 治理（AReaL / AsyncFlow / verl fully_async 的健康度监控，见 [[asynchronous training]] §3.3）。
- **对比 / 易混**:
  - **三个 KL**：`ppo_kl`(对 θ_old) vs `kl_loss`(对 ref，进 loss) vs 监控 `kl_ref`（决定自适应 β）——见 §3.7 表；
  - **`pg_loss` vs `pg_clipfrac`**：前者是损失值大小，后者是被截断比例，**方向相反的健康度**——loss 小未必好、clipfrac 高必坏；
  - **`rewards/mean` vs `advantages/mean`**：前者绝对奖励、后者相对优势，前者该升、后者该≈0。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"reward 上升 = 训练成功"**：最常见错觉。必须联看 KL + 长度，reward 涨 + KL 涨 + 长度涨 = hacking。
> 2. **"pg_loss 下降 = 好"**：PPO 的 pg_loss 下降过快反而是危险信号（策略猛冲），需配 KL 在 target 内。
> 3. **"advantages/mean 该上升"**：错。advantage 设计上 mean≈0，非零说明 critic 估偏。
> 4. **混淆三个 KL**：ppo_kl / kl_loss / kl_ref 三者对象不同、用途不同，看日志别张冠李戴。
> 5. **"clipfrac 高没事，反正 clip 兜底"**：clip 高 = 大量梯度被截 = 有效信号弱 + stale 严重，是病不是特性。
> 6. **"pearson_corr 略降无所谓"**：异步训练里 pearson 是 staleness canary，跌 = 数据快不可用了。
> 7. **"grad_norm 越小越好"**：趋近 0 = 梯度消失学不动。健康是**稳定在量级**，不是越小越好。
> 8. **只看单步不看趋势**：单点值噪音大，必须看滑动平均 + 多指标联动。
> 9. **"response_length 无所谓"**：长度异常是 hacking/collapse 最早信号，比 reward 更早暴露。
> 10. **指标全绿就放心**：还要看生成质量（人工/LLM-as-judge 抽样），指标健康≠对齐成功，只代表训练没崩。

## 8. 延伸细节

### 8.1 健康训练的"绿区"参考值（经验，非绝对）

| 指标 | 健康区间 | 危险信号 |
|---|---|---|
| `critic/rewards/mean` | 随任务，缓慢上升 | 暴涨（配 KL 涨）= hacking；停滞 = 没学 |
| `critic/advantages/mean` | ≈ 0 | 显著非零 = critic 偏 |
| `actor/pg_loss` | 温和下降 | 暴跌 = 猛冲；跳变 = 爆炸 |
| `actor/grad_norm` | 稳定量级 | spike = 爆炸；→0 = 消失 |
| `actor/kl_loss` | 小而稳 | 持续涨 = 漂移 |
| `actor/pg_clipfrac` | < 10% | > 30% = 更新过大/stale |
| `actor/ppo_kl` | < target_kl（0.001–0.1） | 超 target = 早停触发频繁 |
| `response_length/mean` | 与任务匹配 | 暴涨 = 注水；塌缩 = 躺平 |
| `pearson_corr` | > 0.95（同步）/ > 0.9（异步） | < 0.8 = stale 失控 |

> [!warning] 待核实
> 上表数值为社区/实践经验区间，**不同任务、模型规模、框架版本差异大**，仅作起步参考，最终阈值需在自己的 run 上统计确定。

### 8.2 调参联动决策表

| 现象 | 可能原因 | 调参方向 |
|---|---|---|
| reward↑ + kl↑ + length↑ | reward hacking | ↑β(kl_loss) / ↑target_kl 严 / 改 reward shaping |
| clipfrac↑ + pearson↓ | stale 失控 | ↓staleness_threshold / ↑weight sync 频率 / 降异步 |
| grad_norm spike | 梯度爆炸 | ↓lr / ↓max_grad_norm / 查 loss spike |
| advantages/mean ≠0 | critic 落后 | ↑critic lr / ↑critic 训练步数 |
| length 塌缩 | 探索死亡 | ↑entropy bonus / ↓kl 系数 / 查 reward |
| ppo_kl 常 > target | 单步更新太大 | ↓lr / ↓ppo_epochs / ↓batch 内 mini-step |

### 8.3 与异步训练系统的对应

verl 的 `fully_async` / AsyncFlow / AReaL 等[[asynchronous training]]系统里，这套 metric 是 staleness 治理的反馈环：
- `pearson_corr` → 决定 `staleness_threshold` 是否该收紧；
- `pg_clipfrac` → 决定 `partial_rollout` 是否该开；
- `ppo_kl` → 决定 `trigger_parameter_sync_step` 频率。
即**指标驱动异步超参自适应**，是 2025–2026 全异步 RL 系统的趋势（见 [[asynchronous training]] §3.3 的 AReaL/AsyncFlow/verl fully_async 段）。

### 8.4 verl 之外的其他框架命名

| 概念 | verl | OpenRLHF | trl (TRL/PPO) |
|---|---|---|---|
| pg_loss | `actor/pg_loss` | `actor/pg_loss` | `policy/loss` |
| clipfrac | `actor/pg_clipfrac` | `actor/pg_clipfrac` | `policy/clipfrac` |
| ppo_kl | `actor/ppo_kl` | `actor/ppo_kl` | `policy/approxkl` |
| kl to ref | `actor/kl_loss` | `actor/kl_loss` / `policy/kl_coef` | `policy/kl_div` |
| rewards | `critic/rewards/mean` | `rewards/mean` | `rewards/mean` |
| advantage | `critic/advantages/mean` | `advantages/mean` | `advantages/mean` |

概念一致、命名略异，读懂一套即可迁移。

---
相关: [[常见指标]]、[[PPO clipped objective]]、[[GAE]]、[[advantage function]]、[[advantage estimation]]、[[KL penalty与KL control]]、[[KL divergence control]]、[[梯度裁剪]]、[[gradient explosion and vanishing]]、[[reward hacking]]、[[KL explosion]]、[[policy collapse]]、[[mode collapse]]、[[asynchronous training]]、[[stale policy problem]]、[[weight sync mechanism]]、[[PPO optimization]]
