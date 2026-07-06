# KL divergence control

> **所属章节**: [[数值稳定]]
> **所属模块**: [[12-训练稳定性工程]]
> **别名**: KL divergence control / KL 控制 / KL penalty / KL controller / KL 约束
> **难度**: 中（需懂 PPO + [[reward model]] + KL 散度）


## 1. 一句话定义

**KL divergence control（KL 散度控制）** 是 RLHF/PPO 训练中**用 KL 散度约束 policy $\pi_\theta$ 不偏离 reference model $\pi_{\text{ref}}$ 太远**的稳定性机制——在 reward 目标中加入 KL 惩罚 $\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$，防止 policy 为追求 reward 而**reward hacking**（找到 reward 高但语义崩坏的解）或**崩坏**（偏离 SFT 分布过远）。它是 RLHF 训练稳定性的核心：reward model 不完美，KL 约束保 policy 在 SFT 附近"安全区"探索。实现含 **fixed KL penalty**（固定 $\beta$）和 **adaptive KL controller**（$\beta$ 随当前 KL vs target KL 动态调节）。target KL 还用于 early stop（KL 超阈值停训）。DPO 隐式含 KL（无需显式 controller）。

> [!note] 三句话定位
> - **是什么**：用 KL penalty 约束 policy 不偏离 reference 太远，防 reward hacking 和崩坏。
> - **为什么需要**：reward model 不完美，KL 保 policy 在 SFT 附近安全探索，是 RLHF 稳定性核心。
> - **怎么实现**：fixed/adaptive $\beta$ controller + target KL early stop。DPO 隐式含 KL。


## 2. 为什么需要它（动机与背景）

### 2.1 reward model 的不完美

RLHF 的 [[reward model]] 从人类偏好训练，但偏好数据有限、reward model 泛化有限。policy 可能找到 reward model 给高分但实际语义崩坏的解（**reward hacking**）：如重复输出、乱码、 exploiting reward model 的 blind spot。无约束时 policy 会追求这些"虚高 reward"，输出质量崩。

### 2.2 KL 约束的作用

加 KL 惩罚：$\text{objective} = \mathbb{E}[r(x,y)] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$。policy 偏离 $\pi_{\text{ref}}$（SFT model）越远，惩罚越大。这逼 policy 在 SFT 附近"安全区"探索——SFT model 已有合理语义，附近偏离不致崩坏。$\beta$ 平衡 reward 追求 vs 偏离约束。

### 2.3 reward hacking 的典型表现

无 KL 约束时 policy 常见的 hacking：

- **重复输出**：reward model 给重复短语高分（数据偏差），policy 学会重复。
- **格式 exploit**：reward 偏好某格式，policy 只输出格式忽略内容。
- **长度 exploit**：reward 偏长输出，policy 啰嗦。
- **乱码 exploit**：reward model 的 blind spot 给乱码高分。

KL 约束让 policy 不能偏离 SFT 太远去 exploit 这些。

### 2.4 RLHF 的目标重述

RLHF 的真实目标是：**在保持 SFT 语义质量的前提下，对齐人类偏好**。纯 reward 最大化会偏离语义（hacking），纯 SFT 不对齐。KL 约束让优化在两者间 balance：$\max r - \beta \cdot \text{KL}$。$\beta$ 大偏 SFT（稳但不对齐），$\beta$ 小偏 reward（对齐但可能 hacking）。

### 2.5 adaptive controller 的动机

固定 $\beta$ 的问题：训练不同阶段最优 $\beta$ 不同（早期需大 $\beta$ 稳，后期小 $\beta$ 对齐）。**adaptive KL controller**：监测当前 KL，KL > target 时增 $\beta$（拉回），< target 时减 $\beta$（放探索）。自动调节，比 fixed 稳。InstructGPT/PPO 标配。

### 2.6 target KL 与 early stop

训练中监测当前 KL，超 target KL（如 6-10 nats）时停训（early stop）——超阈值说明 policy 偏离过远，继续训会崩。是 RLHF 训练的"安全阀"。


## 3. 核心概念详解

### 3.1 KL 散度的定义

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right] = \mathbb{E}_{y \sim \pi_\theta}[\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)]
$$

采样估计：用 policy 采的样本 $y$，算 $\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)$ 的均值。$D_{\text{KL}}=0$ 时两分布同，$>0$ 时偏离。

### 3.2 PPO 的 KL penalty

PPO 的 objective：

$$
L = \mathbb{E}\left[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)\right] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

$r_t = \pi_\theta(y_t|x) / \pi_{\text{old}}(y_t|x)$ 是 importance ratio。KL 项约束 $\pi_\theta$ 不偏离 $\pi_{\text{ref}}$（SFT），与 PPO 的 clip（约束不偏离 $\pi_{\text{old}}$）正交。详见 [[PPO]]。

### 3.3 fixed vs adaptive $\beta$

| 方式 | $\beta$ | 特点 |
|---|---|---|
| **fixed** | 恒定 | 简单，但不同阶段不最优 |
| **adaptive controller** | $\beta \leftarrow \beta + \eta(\text{KL} - \text{target})$ | KL > target 增 $\beta$ 拉回，自动调节 |
| **heuristic** | 按 KL 比例调 | 介于两者 |

adaptive：KL 偏大（policy 偏离远）→ 增 $\beta$ 加强约束；KL 偏小（policy 太保守）→ 减 $\beta$ 放探索。

### 3.4 target KL 与 early stop

- **target KL**：期望的 KL 水平（如 6 nats）。adaptive controller 的目标。
- **early stop**：当前 KL > threshold（如 10 nats）时停训。防 policy 崩。

训练中 KL 持续监测，超阈值停。是 RLHF 的安全机制。

### 3.5 KL 的数值计算

- **per-token KL**：$D_{\text{KL}} = \sum_t \pi_\theta(y_t) \log(\pi_\theta(y_t)/\pi_{\text{ref}}(y_t))$，但需全词汇表 softmax（贵）。
- **sampled KL**（常用）：$\mathbb{E}_{y \sim \pi_\theta}[\log \pi_\theta(y) - \log \pi_{\text{ref}}(y)]$，用采样近似（便宜，RLHF 主流）。

### 3.6 DPO 的隐式 KL

[[DPO]] 不显式加 KL penalty，但目标函数隐式约束 $\pi_\theta$ 不偏离 $\pi_{\text{ref}}$：

$$
L_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)
$$

$\beta$ 控制偏离容忍度。DPO 的 $\beta$ 类似 RLHF 的 KL penalty 系数。故 DPO 不需 adaptive controller（隐式约束）。


## 4. 数学原理 / 公式

### 4.1 KL 散度

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right]
$$

### 4.2 RLHF 目标

$$
\max_\theta \mathbb{E}_{x, y \sim \pi_\theta}[r(x,y) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})]
$$

### 4.3 adaptive controller

$$
\beta \leftarrow \beta + \eta \cdot (D_{\text{KL}}^{\text{current}} - D_{\text{KL}}^{\text{target}})
$$

$D_{\text{KL}}^{\text{current}} > \text{target}$（偏离远）→ $\beta$ 增（强约束）；$<$ target → $\beta$ 减（放探索）。

### 4.4 sampled KL 估计

$$
\hat{D}_{\text{KL}} = \frac{1}{N}\sum_{i=1}^{N} [\log \pi_\theta(y_i|x) - \log \pi_{\text{ref}}(y_i|x)], \quad y_i \sim \pi_\theta
$$

### 4.5 target KL 的典型值

| 场景 | target KL | 备注 |
|---|---|---|
| InstructGPT | 6 nats | 经验值 |
| LLaMA-2 RLHF | ~10 nats | 较宽松 |
| early stop | > 2× target | 防崩 |


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 KL penalty + adaptive controller

```python
def adaptive_beta(beta, current_kl, target_kl=6.0, lr=0.1):
    """adaptive controller: KL > target 增 beta (强约束), < target 减."""
    return max(0.01, beta + lr * (current_kl - target_kl))

# 模拟训练: policy 逐渐偏离 (实测 KL 递增后回落)
beta = 0.1
kls = [2.0, 4.0, 7.0, 9.0, 5.0]   # 每步实测 KL (target=6)
for step, kl in enumerate(kls):
    penalty = beta * kl
    beta = adaptive_beta(beta, kl)
    print(f'step {step}: KL={kl:.1f}, penalty={penalty:.3f}, next beta={beta:.3f}')
```

### 5.2 target KL early stop

```python
def should_stop(current_kl, target_kl=6.0, threshold=2.0):
    """KL 超 threshold * target 停训."""
    return current_kl > threshold * target_kl

for kl in [3.0, 6.0, 8.0, 13.0, 15.0]:
    stop = should_stop(kl)
    print(f'KL={kl:.1f} -> {"early stop" if stop else "continue"}')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: PPO/RLHF（KL control 的场景）、[[reward model]]（被约束的 reward）、KL 散度（数学基础）、[[gradient explosion and vanishing]]（训练稳定性的数值维度）。
- **下游（应用）**: [[reward normalization]]（RLHF 稳定性组合）、[[entropy regularization]]（鼓励探索 vs KL 约束的 balance）、[[clipping strategies]]（PPO clip 是 KL 之外的约束）、[[loss spike handling]]（KL 失控→policy 崩→loss spike）、DPO（隐式 KL）、RLHF 训练 pipeline。
- **对比 / 易混**:
  - **KL penalty vs PPO clip**：KL 约束 $\pi_\theta$ 不偏离 $\pi_{\text{ref}}$（全局，SFT 参考），clip 约束不偏离 $\pi_{\text{old}}$（局部，上一迭代）。两者正交，PPO 同时用。
  - **KL control vs [[entropy regularization]]**：KL 拉回 reference（防偏离），entropy 鼓励探索（防 mode collapse）。方向相反，需 balance。
  - **explicit KL (PPO) vs implicit KL (DPO)**：PPO 显式加 penalty + adaptive controller，DPO 目标隐式约束（$\beta$ 参数）。DPO 更简单。


## 7. 常见误区与易错点

> [!warning] 误区 1：KL 越小越好
> 不是。KL=0 时 policy = reference（不学，不对齐）。需适当偏离才对齐。target KL 是平衡点（既对齐又不崩）。太小不学，太大崩。

> [!warning] 误区 2：$\beta$ 固定即可
> 固定 $\beta$ 在不同阶段不最优。早期 policy 不稳需大 $\beta$，后期对齐需小 $\beta$。adaptive controller 自动调节，比 fixed 稳。InstructGPT 用 adaptive。

> [!warning] 误区 3：KL 解决 reward hacking
> 缓解不根治。KL 约束 policy 不偏离 SFT 太远，减少 hacking 空间，但 reward model 仍有 blind spot。需配合更好的 reward model、reward normalization、early stop。KL 是第一道防线，非万能。

> [!warning] 误区 4：DPO 无 KL
> 有，隐式。DPO 的 $\beta$ 控制偏离容忍度，等价 RLHF 的 KL penalty 系数。DPO 不需显式 controller（隐式约束），但 $\beta$ 调节同样关键。

> [!warning] 误区 5：per-token KL = sampled KL
> 不同。per-token KL 需全词汇表 softmax（贵，精确），sampled KL 用采样近似（便宜，估计）。RLHF 主流用 sampled（PPO 的 rollout 已采样）。

> [!warning] 误区 6：KL target 任意设
> target KL 有经验范围（6-10 nats）。太小过保守，太大易崩。需按模型规模、reward model 质量调。太大模型需更小 target（更稳）。


## 8. 延伸细节

### 8.1 InstructGPT 的 KL controller

InstructGPT（GPT-3.5）用 adaptive controller：$\beta$ 按当前 KL vs target（6 nats）动态调。还用 **target KL 作为 early stop** 信号。是 RLHF 工程化的标杆。

### 8.2 KL 与 reward 的 trade-off

$\beta$ 控制 reward vs KL 的 balance：

- $\beta$ 大：偏 SFT（稳，reward 低，不对齐）。
- $\beta$ 小：偏 reward（对齐，但可能 hacking）。
- 最优 $\beta$：reward 提升明显但 KL 不超 target。

调 $\beta$ 是 RLHF 的核心超参。adaptive controller 自动寻优。

### 8.3 KL 的数值稳定性

sampled KL 的 $\log \pi_\theta(y) - \log \pi_{\text{ref}}(y)$：若 $\pi_\theta(y)$ 极小（$\log \to -\infty$），KL 不稳。对策：

- clip logprob（防极值）。
- bf16/fp32 精度（fp16 易下溢）。
- 大 batch 平均（减方差）。

[[gradient explosion and vanishing]] 的数值管理在此也适用。

### 8.4 reward normalization 的配合

[[reward normalization]]（reward 归一化）+ KL control 组合稳 RLHF：reward 归一化让 reward 尺度稳定，KL penalty 的 $\beta$ 更易调。两者配合是 RLHF 稳定性标配。

### 8.5 entropy regularization 的 balance

[[entropy regularization]]（加 $-\beta_e H(\pi)$ 鼓励高熵=探索）与 KL control（拉回 reference=防偏离）方向相反：

- entropy reg：防 mode collapse（policy 塌到单一输出）。
- KL control：防偏离过远。

两者需 balance：entropy 鼓励多样探索，KL 约束不偏太远。$\beta_e$ 和 $\beta_{\text{KL}}$ 的比例是调参点。

### 8.6 DPO 的 $\beta$ 调节

DPO 的 $\beta$（目标函数中的）控制偏离容忍度：

- $\beta$ 大：严格约束（偏 reference，稳但不对齐）。
- $\beta$ 小：宽松（对齐但可能过拟合偏好数据）。

DPO 无 adaptive controller（无显式 KL），但 $\beta$ 调节类似。典型 $\beta = 0.1-0.5$。

### 8.7 KL 失控与 loss spike

KL controller 失效时（$\beta$ 过小/调节不及），policy 快速偏离 → reward hacking → 梯度不稳 → [[loss spike handling|loss spike]]。KL control 是 RLHF 训练稳定性的第一道防线，失控直接致 spike。诊断 loss spike 时查 KL 曲线（是否突增）。

---
相关: [[数值稳定]] | [[gradient explosion and vanishing]] | [[loss spike handling]] | [[reward normalization]] | [[entropy regularization]] | [[clipping strategies]] | [[warmup]] | [[PPO]] | [[reward model]] | [[DPO]] | [[RLHF]] | [[混合精度]]
