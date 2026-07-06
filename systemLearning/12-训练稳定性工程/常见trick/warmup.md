# warmup

> **所属章节**: [[常见trick]]
> **所属模块**: [[12-训练稳定性工程]]
> **别名**: warmup / 学习率预热 / lr warmup / linear warmup
> **难度**: 入门（需懂 lr schedule + [[gradient explosion and vanishing]]）


## 1. 一句话定义

**warmup（学习率预热）** 是训练前期将学习率从 0（或小值）**线性（或余弦）增大到 peak lr** 的阶段，防止早期训练不稳定（[[loss spike handling|loss spike]] / [[gradient explosion and vanishing|梯度爆炸]] / 震荡）。早期权重随机、梯度方向混乱、Adam 的二阶矩 $v$ 冷启动（无历史统计），大 lr 易炸；warmup 让 lr 小步试探，先稳住 statistics 再加速。典型 warmup steps = 总 steps 的 **0.5%-2%**（大模型可达 2%）。与 cosine decay 配合构成 LLM 标准 lr schedule：warmup（升）+ cosine（降）。muP（maximal update parameterization）特别依赖 warmup 稳定超参迁移。RLHF/PPO 训练也用 warmup（短）。

> [!note] 三句话定位
> - **是什么**：前期 lr 从 0 线性升到 peak，防早期不稳。
> - **为什么**：早期权重随机 + Adam $v$ 冷启动 + 梯度高方差，大 lr 易炸/震荡。
> - **怎么用**：warmup 0.5-2% 总 steps + cosine decay；muP/RLHF 也用。


## 2. 为什么需要它（动机与背景）

### 2.1 早期训练的不稳定

训练初期（前几百步）：

- **权重随机**：初始权重未学到任何结构，梯度方向混乱（非最优下降方向）。
- **梯度高方差**：早期 loss 大、梯度大且方向乱，大 lr 过冲易炸。
- **Adam $v$ 冷启动**：Adam 的二阶矩 $v$（梯度平方的 EMA）无历史，初始 $v \approx 0$，归一化 $\hat{m}/\sqrt{\hat{v}}$ 不稳（分母小，更新大）。
- **batch norm / LayerNorm statistics 未稳**：running mean/var 未收敛。

任一因素 + 大 lr → 早期 [[loss spike handling|loss spike]] 或发散。GPT-3、LLaMA 等大模型早期 spike 集中在此阶段。

### 2.2 warmup 的作用

warmup 让 lr 从 0 线性增到 peak：

- 前几步 lr 小 → 权重小步更新 → 梯度方向逐步对齐 → Adam $v$ 积累统计 → LN statistics 稳。
- 到 peak lr 时，statistics 已稳，大 lr 安全。

防早期 spike，是 LLM 训练稳定性的第一道防线（与 gradient clip 并列）。

### 2.3 Adam 冷启动的细节

Adam 更新：$\theta \leftarrow \theta - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$。初始 $\hat{v}_t \approx 0$（无历史梯度平方），分母极小，更新极大（即使 $\eta$ 不大）。bias correction 缓解但不彻底。warmup 让前期 $\eta$ 小，抵消 $\hat{v}$ 小的影响，等 $v$ 积累后再大 $\eta$。这是 warmup 对 Adam 的特殊必要性（SGD 不需 warmup 这么迫切）。

### 2.4 muP 的依赖

[[muP]]（maximal update parameterization）通过调参让超参可从 small model 迁移到 large model。muP 的稳定性高度依赖 warmup——前期小 lr 防 init 不当致 layer 间 update 失衡。muP 的 warmup 比标准 init 更关键。

### 2.5 RLHF 的 warmup

RLHF/PPO 训练也用 warmup（短，如前 10-50 步），防 policy 早期大幅偏离 reference。与 [[KL divergence control]] 配合稳 RLHF 前期。


## 3. 核心概念详解

### 3.1 warmup 的 lr 曲线

```
lr
peak |       /‾‾‾\
     |      /     \         cosine decay
     |     /       \
     |    /         \
     |   /           \
     |  /             \
   0 |__/_______________\____
     0  warmup_end     total_steps
        (1-2%)
```

warmup 阶段 lr 线性增（0 → peak），之后 cosine decay（peak → 0）。

### 3.2 warmup steps 的选择

| 模型规模 | warmup 比例 | 典型 steps |
|---|---|---|
| 小模型 (<1B) | 0.5%-1% | 几百 |
| LLM (7-70B) | 1%-2% | 2000-8000 |
| 超大 (175B+) | 2% | ~10000 |
| RLHF | 0.5%-1% | 10-50 |

经验值：大模型需更长 warmup（statistics 更多需稳）。太短不稳，太长浪费算力（前期 lr 小学得慢）。

### 3.3 linear vs cosine warmup

| 方式 | 公式 | 特点 |
|---|---|---|
| **linear** | $\eta_t = \eta_{\text{peak}} \cdot \min(1, t/t_w)$ | 简单主流 |
| **cosine** | $\eta_t = \eta_{\text{peak}} \cdot 0.5(1 - \cos(\pi t/t_w))$ | 前期更缓 |

linear 最常用。cosine warmup 前期增更缓（更保守），适合极不稳场景。

### 3.4 与 cosine decay 的配合

LLM 标准 schedule：

1. **warmup**（0 → peak）：前 $t_w$ 步，线性增。
2. **cosine decay**（peak → 0）：$t_w$ 到 $T$，余弦降。

```python
def lr_schedule(step, peak_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return peak_lr * step / warmup_steps          # warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))  # cosine
```

### 3.5 warmup restart / resume

checkpoint resume 时：

- **从头 warmup**：resume 后重新 warmup（短），防 resume 后不稳。
- **继续 schedule**：按原 schedule 的 lr 续（step 计数累加）。

主流按原 schedule 续（不重 warmup），但若 resume 后 spike 可短 warmup。


## 4. 数学原理 / 公式

### 4.1 linear warmup

$$
\eta_t = \eta_{\text{peak}} \cdot \min\left(1, \frac{t}{t_w}\right)
$$

$t < t_w$ 时线性增，$t \geq t_w$ 时达 peak。

### 4.2 cosine warmup

$$
\eta_t = \eta_{\text{peak}} \cdot \frac{1}{2}\left(1 - \cos\left(\pi \cdot \min(1, t/t_w)\right)\right)
$$

### 4.3 Adam 冷启动的更新放大

Adam 更新：$\Delta \theta = \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$。初始 $\hat{v}_t \approx g_1^2$（第一步梯度平方），$\sqrt{\hat{v}_t} \approx |g_1|$。若 $g_1$ 小（早期随机），$\sqrt{\hat{v}_t}$ 极小，$\Delta \theta$ 放大。warmup 的小 $\eta$ 抵消此放大。

### 4.4 梯度方差早期大

早期 loss 大（权重随机），梯度 $\|g\|$ 大且方差高。$\text{Var}(g_t)$ 早期 > 稳态。大 $\eta$ × 高 Var → 更新震荡/爆炸。warmup 小 $\eta$ 降震荡。

### 4.5 warmup 与 spike 概率

经验：warmup 阶段 spike 少（$\eta$ 小）；peak 阶段（warmup 结束后）spike 多（$\eta$ 大）。warmup 把 spike 风险从早期推到 peak 阶段（此时 statistics 已稳，更可控）。


## 5. 代码示例（可选

### 5.1 lr schedule（warmup + cosine）

```python
import math
def lr_schedule(step, peak_lr=3e-4, warmup_steps=2000, total_steps=100000):
    if step < warmup_steps:
        return peak_lr * step / warmup_steps                       # linear warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))     # cosine decay

for s in [0, 500, 2000, 5000, 50000, 100000]:
    print(f'step {s:6d}: lr={lr_schedule(s):.6f}')
```

### 5.2 对比有/无 warmup 的早期 loss（模拟）

```python
import random
def simulate(warmup, steps=50, peak_lr=1.0):
    lr, loss, v = 0.0, 5.0, 0.0  # Adam v 冷启动
    history = []
    for t in range(steps):
        lr = peak_lr * min(1, t/warmup) if warmup else peak_lr
        g = random.gauss(-1, 2.0)        # 早期梯度大方差
        v = 0.9*v + 0.1*g*g              # Adam v EMA
        update = lr * g / (v**0.5 + 1e-8)
        loss += update
        history.append(loss)
    return history

no_warmup = simulate(warmup=0)
with_warmup = simulate(warmup=10)
print(f"无 warmup 前 5 步 loss: {[round(x,2) for x in no_warmup[:5]]}")
print(f"有 warmup 前 5 步 loss: {[round(x,2) for x in with_warmup[:5]]}")
```


## 6. 与其他知识点的关系

- **上游（依赖）**: lr schedule（warmup 的上下文）、Adam optimizer（$v$ 冷启动是 warmup 必要性之一）、[[gradient explosion and vanishing]]（warmup 防早期爆炸）、初始化策略（He/Xavier）。
- **下游（应用）**: [[loss spike handling]]（warmup 预防早期 spike）、[[clipping strategies]]（与 warmup 并列防 spike）、[[KL divergence control]]（RLHF warmup 稳前期）、[[reward normalization]]（RLHF 稳定性组合）、LLM 预训练（标准 schedule）、[[muP]]（依赖 warmup）、RLHF/PPO 训练。
- **对比 / 易混**:
  - **warmup vs cosine decay**：warmup 是前期升（0→peak），cosine decay 是后期降（peak→0）。方向相反，配合用。
  - **warmup vs constant lr**：constant lr 无升阶段，早期大 lr 易炸；warmup 缓解。
  - **warmup vs [[clipping strategies\|gradient clip]]**：warmup 调 lr（全局），clip 调梯度 norm（per-step）。两者正交，防 spike 维度不同。


## 7. 常见误区与易错点

> [!warning] 误区 1：warmup 越长越稳
> 不是。过长 warmup 前期 lr 小学得慢，浪费算力。最优 warmup 是"刚好稳 statistics"的长度（0.5-2%）。过长无额外收益，过短不稳。

> [!warning] 误区 2：warmup 只防 spike
> 不只。warmup 还稳 Adam $v$ 冷启动、LN running statistics、梯度方差。即使无 spike，warmup 也让前期收敛更稳（避免早期震荡浪费 step）。

> [!warning] 误区 3：warmup 只大模型需
> 小模型也受益（Adam $v$ 冷启动 + 梯度方差早期大是普遍问题）。只是小模型规模小、spike 风险低，warmup 可短。大模型 spike 风险高，warmup 更关键更长。

> [!warning] 误区 4：SGD 也需 warmup
> SGD 无 $v$ 冷启动问题（无二阶矩归一化），warmup 必要性低。Adam 系列（Adam/AdamW）才迫切需 warmup。SGD 用 constant lr 早期不稳风险低。

> [!warning] 误区 5：resume 必须重 warmup
> 不一定。主流按原 schedule 续 lr（step 累加，不重 warmup），因 statistics 已稳。重 warmup 是 resume 后 spike 的应急手段，非默认。

> [!warning] 误区 6：warmup lr 从 0 会卡住
> 不会。lr 从 0 线性增，几步后 lr > 0 即开始有效更新。从 0 是防第一步 lr 过大炸。linear warmup 前几步 lr 极小但不卡（Adam 的 $v$ 也在积累）。


## 8. 延伸细节

### 8.1 muP 与 warmup

[[muP]]（maximal update parameterization）通过调 init scale 和 lr scale 让 update 大小与宽度无关，实现超参从小模型迁移到大模型。muP 的稳定性对 warmup 敏感——前期小 lr 防 init 不当时 layer 间 update 失衡（某些层 update 过大）。muP 推荐更长 warmup。

### 8.2 LLM 的典型 schedule

LLaMA-2 7B：warmup 2000 steps（~0.8%），cosine decay 到 10% peak，total 2T tokens。GPT-3 175B：warmup 375M tokens（~2%）。大模型 warmup 比例相近（1-2%），总量大。

### 8.3 RLHF 的 warmup

RLHF/PPO 训练的 warmup 短（10-50 步），防 policy 早期大幅偏离 reference。与 [[KL divergence control]] 的 $\beta$ 配合：前期大 $\beta$ + warmup 小 lr 双保险。RLHF 前期最易 KL 失控，warmup 缓解。

### 8.4 warmup 与 batch size

大 batch 常配大 lr（linear scaling rule），但大 lr 更需 warmup（早期不稳风险更高）。大 batch 训练的 warmup 更关键。LLM 大 batch + 大 lr + 长 warmup 是标配。

### 8.5 wsd schedule（替代）

近期有 **wsd**（warmup-stable-decay）schedule：warmup 后保持恒定 lr，最后短 decay。比 cosine 更灵活（可续训），部分场景替代 cosine。但仍需 warmup（stable 前）。详见 lr schedule 文献。

### 8.6 warmup 失败的诊断

warmup 后仍 spike：

- 检查 warmup 是否够长（增至 2%）。
- 检查 init（He/Xavier 是否对）。
- 检查 gradient clip 是否开。
- 检查数据（异常样本）。
- 检查 bf16（是否 fp16 溢出）。

warmup 是预防之一，非万能，需与其他稳定性手段配合。

---
相关: [[常见trick]] | [[clipping strategies]] | [[loss spike handling]] | [[gradient explosion and vanishing]] | [[KL divergence control]] | [[reward normalization]] | [[entropy regularization]] | [[混合精度]] | [[AdamW]] | [[muP]] | [[cosine decay]] | [[lr schedule]] | [[attention]] | [[LayerNorm]]
