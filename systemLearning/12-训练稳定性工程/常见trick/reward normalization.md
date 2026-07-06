# reward normalization

> **所属章节**: [[常见trick]]
> **所属模块**: [[12-训练稳定性工程]]
> **别名**: reward normalization / reward scaling / reward 归一化
> **难度**: 中（需懂 RLHF + [[reward model]] + PPO）


## 1. 一句话定义

**reward normalization（reward 归一化）** 是 RLHF/PPO 训练中对 [[reward model]] 输出做归一化（减均值、除标准差、或除 running scale），使 reward 尺度稳定，便于 [[KL divergence control]] 的 $\beta$ 调节和 PPO advantage 估计稳定。reward model 输出尺度不定（取决于偏好数据/训练），不同 batch 波动大；不归一化则 $\beta$ 难调、advantage 高方差、训练不稳。常见方式：**batch normalization**（用当前 batch 的 μ/σ）、**running mean/std**（EMA 全局统计）、**reward scaling**（除常数 scale）。与 advantage normalization 配合稳 PPO。InstructGPT 用 reward scaling（除 KL 系数相关 scale）。

> [!note] 三句话定位
> - **是什么**：对 reward model 输出归一化（减 μ / 除 σ），使 reward 尺度稳定。
> - **为什么**：reward 尺度不定 + 波动大，致 $\beta$ 难调、advantage 不稳。
> - **怎么用**：batch norm / running stats / scaling + advantage normalization。


## 2. 为什么需要它（动机与背景）

### 2.1 reward model 输出的尺度不定

[[reward model]] 输出是标量 $r(x,y)$，但尺度取决于：

- 偏好数据的标注分布（高分/低分习惯）。
- reward model 的训练（loss scale、init）。
- 输入长度（长输出 reward 可能偏高/低）。

不同训练、不同 batch 的 reward 尺度可能差 10-100×。无归一化时 $\beta$（KL penalty 系数）需按当前尺度调，但尺度波动使 $\beta$ 难定。

### 2.2 PPO 对 reward 尺度的敏感

PPO 的 advantage $A = r - V(s)$，更新方向由 advantage 决定。reward 尺度大 → advantage 大 → 更新大 → 不稳。尺度波动 → advantage 方差高 → PPO 震荡。归一化 reward 使 advantage 尺度稳定，PPO 稳。

### 2.3 KL penalty 的 β 调节

[[KL divergence control]] 的 penalty $\beta \cdot D_{\text{KL}}$ 与 reward 的尺度需匹配：reward 大则 $\beta$ 需大才有约束力。reward 尺度波动 → $\beta$ 难调。归一化 reward 到稳定尺度（如 std=1），$\beta$ 易定（如 0.1）。InstructGPT 的 reward scaling 即为此。

### 2.4 不归一化的后果

- $\beta$ 难调：尺度变 $\beta$ 失效，KL 失控。
- advantage 高方差：PPO 震荡/发散。
- loss spike：reward 极端值 → 梯度不稳 → [[loss spike handling|spike]]。

归一化是 RLHF 稳定性的基础，与 [[KL divergence control]] / [[warmup]] / gradient clip 并列。


## 3. 核心概念详解

### 3.1 三种归一化方式

| 方式 | 公式 | 特点 |
|---|---|---|
| **batch norm** | $r' = (r - \mu_{\text{batch}}) / \sigma_{\text{batch}}$ | 用当前 batch 统计，简单但 batch 小时不稳 |
| **running mean/std** | $r' = (r - \mu_{\text{EMA}}) / \sigma_{\text{EMA}}$ | EMA 全局统计，稳，主流 |
| **reward scaling** | $r' = r / \text{scale}$ | 除常数，最简单，InstructGPT 用 |

### 3.2 running mean/std（主流）

维护 reward 的 EMA 均值/标准差：

$$
\mu_{t+1} = \alpha \mu_t + (1-\alpha) \bar{r}_t, \quad \sigma_{t+1} = \alpha \sigma_t + (1-\alpha) \text{std}(r_t)
$$

$\bar{r}_t$, $\text{std}(r_t)$ 是 batch $t$ 的均值/标准差，$\alpha \approx 0.9$。归一化 $r' = (r - \mu_t) / \sigma_t$。比 batch norm 稳（用全局统计），比 scaling 灵（自适应）。

### 3.3 advantage normalization

PPO 常对 advantage 也归一化（batch 内）：

$$
A' = (A - \bar{A}) / \text{std}(A)
$$

使 advantage 均值 0、std 1，PPO 更新尺度稳定。与 reward normalization 互补：reward 归一化稳输入，advantage 归一化稳中间量。两者配合稳 PPO。

### 3.4 InstructGPT 的 reward scaling

InstructGPT 用简单 scaling：$r' = r / \eta$，$\eta$ 是与 KL 系数相关的 scale（如 $\eta = 1/\beta$ 或经验常数）。使 reward 与 KL penalty 尺度匹配。比 running stats 简单。

### 3.5 reward shaping 的区别

**reward shaping**：修改 reward 函数（如加 length penalty、format penalty）改变优化目标。**reward normalization**：只改尺度不改相对序（$r'$ 的排序同 $r$），不改目标。两者不同：shaping 改"学什么"，normalization 改"reward 的数值尺度"。

### 3.6 与 KL control 的配合

reward normalization（稳 reward 尺度）+ [[KL divergence control]]（稳 policy 偏离）+ [[warmup]]（稳前期 lr）+ gradient clip（稳梯度）= RLHF 稳定性栈。归一化让 $\beta$ 可调（KL 控制的前提），两者紧耦合。


## 4. 数学原理 / 公式

### 4.1 batch normalization

$$
r'_i = \frac{r_i - \mu_{\text{batch}}}{\sigma_{\text{batch}} + \epsilon}, \quad \mu_{\text{batch}} = \frac{1}{B}\sum_i r_i, \quad \sigma_{\text{batch}} = \sqrt{\frac{1}{B}\sum_i (r_i - \mu)^2}
$$

### 4.2 running mean/std（EMA）

$$
\mu_t = \alpha \mu_{t-1} + (1-\alpha) \bar{r}_t, \quad \sigma_t = \alpha \sigma_{t-1} + (1-\alpha) \text{std}(r_t)
$$

### 4.3 reward scaling

$$
r' = \frac{r}{\eta}, \quad \eta \text{ 是常数（如 } 1/\beta \text{ 或经验值）}
$$

### 4.4 advantage normalization

$$
A'_i = \frac{A_i - \bar{A}}{\text{std}(A) + \epsilon}
$$

### 4.5 与 KL penalty 的尺度匹配

KL penalty $\beta \cdot D_{\text{KL}}$ 应与 reward 同量级。reward std=1 时 $\beta \approx 0.1$ 合理。若 reward std=100，需 $\beta \approx 10$ 才同量级。归一化 reward 到 std=1 使 $\beta$ 跨训练可复用。


## 5. 代码示例（可选

### 5.1 running mean/std 归一化

```python
class RewardNormalizer:
    def __init__(self, alpha=0.9, eps=1e-8):
        self.mu, self.sigma, self.alpha, self.eps = 0.0, 1.0, alpha, eps
    def update(self, rewards):
        batch_mu = sum(rewards)/len(rewards)
        var = sum((r-batch_mu)**2 for r in rewards)/len(rewards)
        batch_sigma = var**0.5
        self.mu = self.alpha*self.mu + (1-self.alpha)*batch_mu
        self.sigma = self.alpha*self.sigma + (1-self.alpha)*batch_sigma
    def normalize(self, r):
        return (r - self.mu) / (self.sigma + self.eps)

norm = RewardNormalizer()
batches = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [1.5, 2.5, 3.5]]
for b in batches:
    norm.update(b)
    print(f"batch {b} -> normalized {[round(norm.normalize(r),3) for r in b]}")
```

### 5.2 advantage normalization

```python
def normalize_advantage(advantages):
    mu = sum(advantages)/len(advantages)
    var = sum((a-mu)**2 for a in advantages)/len(advantages)
    sigma = var**0.5
    return [(a-mu)/(sigma+1e-8) for a in advantages]

advs = [0.5, -0.3, 1.2, -0.8, 0.9]
print("normalized:", [round(a,3) for a in normalize_advantage(advs)])
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[reward model]]（被归一化的 reward 来源）、PPO/RLHF（归一化的场景）、统计学（归一化数学）。
- **下游（应用）**: [[KL divergence control]]（归一化是 $\beta$ 可调的前提）、[[loss spike handling]]（稳 reward 防 spike）、[[entropy regularization]]（稳 reward 稳探索）、[[clipping strategies]]（稳 advantage 使 PPO clip 有效）、RLHF 训练 pipeline、advantage estimation（GAE）。
- **对比 / 易混**:
  - **reward normalization vs reward shaping**：前者只改尺度不改序/目标，后者改 reward 函数改目标。详见 §3.5。
  - **reward normalization vs advantage normalization**：前者归一化 reward（输入），后者归一化 advantage（中间量）。互补，PPO 同时用。
  - **reward normalization vs [[clipping strategies\|gradient clip]]**：前者稳 reward 尺度（输入端），clip 稳梯度 norm（梯度端）。不同层级的稳定性手段。
  - **reward normalization vs [[KL divergence control]]**：归一化稳 reward（让 $\beta$ 可调），KL control 稳 policy 偏离（用 $\beta$）。归一化是 KL control 的前提。


## 7. 常见误区与易错点

> [!warning] 误区 1：reward normalization = reward shaping
> 不同。normalization 只改尺度不改相对序（$r'$ 排序同 $r$），不改优化目标；shaping 改 reward 函数（加 penalty）改目标。normalization 是数值稳定手段，shaping 是目标设计。

> [!warning] 误区 2：归一化丢信息
> 不丢相对序（谁高谁低不变），只改数值尺度。policy 仍按 reward 高低优化，信息保留。丢的只是绝对尺度（本来就不定，无意义）。

> [!warning] 误区 3：任意归一化
> 需稳定统计。batch norm 在小 batch 时不稳（统计噪声大）；running stats 更稳。归一化的统计不稳反而引入噪声致不稳。

> [!warning] 误区 4：归一化后 β 任意
> 归一化后 reward std≈1，$\beta$ 有经验范围（0.05-0.5）。仍需按模型/reward model 质量调，非任意。归一化让 $\beta$ 可调（跨训练可比），非消除调参。

> [!warning] 误区 5：只归一化 reward 即可
> 不足。reward normalization 稳输入，还需 advantage normalization（稳中间）+ KL control（稳偏离）+ clip（稳梯度）+ warmup（稳前期）。是稳定性栈之一，非全部。

> [!warning] 误区 6：scaling = running stats
> scaling 除常数（不自适应），running stats 自适应（随 reward 分布变）。running stats 更稳但复杂。InstructGPT 用 scaling（简单），现代 RLHF 倾向 running stats（稳）。


## 8. 延伸细节

### 8.1 InstructGPT 的 reward scaling

InstructGPT 用 $r' = r / \eta$，$\eta$ 与 KL 系数相关。使 reward 与 KL penalty 同量级，$\beta$ 可调。是工程化简化（比 running stats 早，简单可用）。

### 8.2 running stats 的冷启动

running mean/std 初始 $\mu=0, \sigma=1$，前几步统计不准（冷启动）。对策：

- 前几步用 batch norm（无 EMA），之后切 running stats。
- 或 warmup 阶段不归一化（reward 尺度先观察）。

类似 [[warmup]] 的冷启动问题。

### 8.3 per-batch vs global normalization

- **per-batch**：每个 batch 内归一化（batch norm）。简单但 batch 间不可比。
- **global**（running stats）：跨 batch 全局归一化。batch 间可比，稳。

RLHF 主流用 running stats（全局可比）。per-batch 仅小 batch 时退用。

### 8.4 reward normalization 与 length bias

长输出 reward 可能系统性偏高/低（reward model 的长度偏置）。归一化不减长度偏置（只改尺度）。需配合 length penalty（reward shaping）减偏置。normalization 与 shaping 正交。

### 8.5 与 advantage normalization 的配合

PPO 训练流程：

1. reward model 输出 $r$ → reward normalization（running stats）。
2. 算 advantage $A = r - V$（GAE）→ advantage normalization（batch）。
3. PPO clip 更新。

两层归一化稳 reward → advantage → 更新。是 PPO 标准稳定性 pipeline。

### 8.6 reward normalization 失败的诊断

归一化后仍不稳：

- 检查 running stats 是否稳（冷启动？切 global）。
- 检查 reward model 输出分布（极端值？需 clip reward）。
- 检查 advantage normalization 是否开。
- 检查 $\beta$ 是否匹配归一化后尺度。

归一化是稳定性栈之一，失败需查其他环节。

---
相关: [[常见trick]] | [[KL divergence control]] | [[loss spike handling]] | [[entropy regularization]] | [[clipping strategies]] | [[warmup]] | [[reward model]] | [[PPO]] | [[RLHF]] | [[DPO]] | [[advantage estimation]] | [[GAE]] | [[gradient explosion and vanishing]] | [[mixed precision]]
