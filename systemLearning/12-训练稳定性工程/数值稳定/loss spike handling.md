# loss spike handling

> **所属章节**: [[数值稳定]]
> **所属模块**: [[12-训练稳定性工程]]
> **别名**: loss spike handling / loss spike / loss 飙升 / 训练发散 / spike recovery
> **难度**: 中（需懂 [[gradient explosion and vanishing]] + LLM 训练 + 数值稳定性）


## 1. 一句话定义

**loss spike（loss 飙升）** 是训练中 loss **突然大幅升高**（可能后恢复或致 NaN/发散）的现象，是训练不稳定的标志。LLM 预训练和 RLHF 常遇：某步 loss 突增 10-100×，轻则恢复（短暂 spike），重则 NaN 发散（训练崩）。原因复杂：[[gradient explosion and vanishing|梯度爆炸]]、数据异常（坏样本/极端长度）、lr 过大、bf16 精度损失、attention 数值不稳（大 logit）、[[KL divergence control]] 失控（RLHF policy 崩）。应对：**detect**（loss > 阈值×moving avg）、**skip step**（跳过该步不更新）、**rollback**（回滚到 checkpoint）、**lower lr** / **clip** / **warmup**（预防）。LLM 训练（如 GPT-3 175B、LLaMA）都有 spike 报告，工程上需 watchdog 自动处理。

> [!note] 三句话定位
> - **是什么**：loss 突然飙升（10-100×），可能恢复或致 NaN 发散，是训练不稳定标志。
> - **为什么**：梯度爆炸/数据异常/lr 过大/bf16 精度/attention 不稳/KL 失控。
> - **怎么应对**：detect + skip step + rollback + 预防（clip/lower lr/warmup/bf16）。


## 2. 为什么需要它（动机与背景）

### 2.1 LLM 训练的不稳定

LLM 预训练（大模型 + 大 batch + 大 lr + 长序列 + bf16）是数值不稳定的温床：

- 大模型：层数多，[[gradient explosion and vanishing|连乘]] 放大。
- 大 lr：早期/某步梯度方向不利时易炸。
- bf16：精度低（7 bit mantissa），累加误差 + 易溢出。
- attention 大 logit：QK^T 数值大，softmax 饱和，梯度消失/不稳。
- 数据：海量数据含异常样本（极端长度、乱码、重复）。

任一因素在某步触发 → loss spike。GPT-3、LLaMA、PaLM 等都报告过 spike。

### 2.2 spike 的两种结局

- **恢复（transient spike）**：loss 突增后几步恢复，训练继续。常因某步数据异常 + clip 限梯度，后续 step 修正。
- **发散（divergence）**：loss 突增后 NaN/inf 传播，权重变 NaN，训练崩。常因梯度爆炸未被 clip 限住，权重更新炸。

工程目标：让 spike 只 transient 不发散（detect + skip/rollback），并预防（clip/warmup/lower lr）。

### 2.3 RLHF 的 spike

RLHF/PPO 训练的 spike 常因：

- [[KL divergence control]] 失控：$\beta$ 过小/调节不及，policy 快速偏离，reward hacking，梯度不稳。
- reward model 输出异常（极端 reward 值）。
- advantage 估计高方差。

KL 曲线突增常预示 spike。详见 [[KL divergence control]] §8.7。

### 2.4 应对栈

| 阶段 | 手段 | 机制 |
|---|---|---|
| **预防** | gradient clip | 限梯度 norm，防爆炸传权重 |
| **预防** | [[warmup]] | 前期小 lr，防早期不稳 |
| **预防** | bf16（非 fp16） | 范围大，不易溢出 |
| **预防** | QK-norm | attention logit 归一化 |
| **检测** | loss watchdog | loss > 阈值×avg 报警 |
| **处理** | skip step | 跳过该步不更新权重 |
| **处理** | rollback | 回滚到上一 checkpoint |
| **处理** | lower lr | 降 lr 恢复稳 |


## 3. 核心概念详解

### 3.1 spike 的原因

| 原因 | 机制 | 典型 |
|---|---|---|
| **梯度爆炸** | 连乘 >1，梯度突大，权重更新炸 | 深层 + 不当 lr |
| **数据异常** | 坏样本（极端长度/乱码/重复）致 loss 大 | 大数据集难免 |
| **lr 过大** | 某步梯度方向不利，大 lr 过冲 | 调 lr 时 |
| **bf16 精度** | 累加误差/溢出 | 大模型 + bf16 |
| **attention 不稳** | QK^T 大 logit，softmax 饱和，梯度消失/炸 | 长序列/大 vocab |
| **KL 失控（RLHF）** | policy 偏离过远，reward hacking | $\beta$ 不当 |
| **optimizer state 异常** | Adam 的 $v$ 下溢/异常 | fp16 state |

### 3.2 detect：loss watchdog

监测 loss 突增：

$$
\text{spike if } \text{loss}_t > k \cdot \text{moving\_avg}(\text{loss}_{t-w:t-1})
$$

$k$ 阈值（如 3-10×），$w$ 窗口（如 10-50 步）。spike 时触发处理。也监测梯度 norm（突增 = 爆炸）和 NaN/inf。

### 3.3 skip step

spike 时**跳过该步的权重更新**（不 `optimizer.step()`），丢弃该 batch 的梯度。下步继续。若 spike 是数据异常 + transient，skip 几步即恢复。简单有效，是 LLM 训练 watchdog 的第一响应。

### 3.4 rollback

spike 致权重已坏（NaN 或持续高 loss）时，**回滚到上一 good checkpoint**，降 lr 重训。比 skip 更重，用于发散情况。需定期存 checkpoint（每 N 步）。

### 3.5 lower lr

spike 频发说明 lr 过大。**降 lr**（如 ×0.5）到稳定区间。LLM 训练的 lr schedule（cosine decay）后期 lr 低，spike 少；前期 warmup + 高 lr 阶段 spike 多。

### 3.6 预防：clip + warmup + bf16 + QK-norm

- **gradient clip**：限梯度 norm（$c \sim 1.0$），防爆传权重。详见 [[clipping strategies]]。
- **[[warmup]]**：前期 lr 从 0 线性增到 peak，防早期不稳（早期权重随机，梯度方向乱，大 lr 易炸）。
- **bf16**：范围大（max ~3.4e38 vs fp16 65504），不易溢出。LLM 训练主流。
- **QK-norm**：attention 的 Q, K 归一化后算 QK^T，控 logit 尺度，防 softmax 饱和。PaLM/Gemma 用。


## 4. 数学原理 / 公式

### 4.1 spike 检测

$$
\text{spike}_t = \mathbb{1}\left[\text{loss}_t > k \cdot \frac{1}{w}\sum_{i=t-w}^{t-1} \text{loss}_i\right]
$$

### 4.2 梯度 norm 监测

$$
\|g_t\|_2 = \sqrt{\sum_p g_{t,p}^2}
$$

$\|g_t\|$ 突增（> k× moving avg）= 爆炸预警。

### 4.3 NaN 传播

权重 $W$ 变 NaN 后，前向 $\text{NaN} \cdot x = \text{NaN}$，所有后续层 NaN，loss NaN。一旦 NaN 必须 rollback（skip 无用，权重已坏）。

### 4.4 skip step 的效果

skip 该步：$\theta_{t+1} = \theta_t$（不更新）。下步 $t+1$ 用新 batch 算梯度（可能正常），若正常则更新。transient spike 经 skip 几步后恢复。

### 4.5 lr 与 spike 的关系

lr 大 → 权重更新大 → 易过冲 → spike 概率高。warmup 让前期 lr 小：

$$
\eta_t = \eta_{\text{peak}} \cdot \min(1, t/t_{\text{warmup}})
$$

前期 $\eta$ 小，spike 少。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 spike 检测 + skip

```python
def detect_spike(loss_history, threshold=3.0, window=5):
    """loss > threshold * moving_avg = spike."""
    if len(loss_history) < window + 1:
        return False
    moving_avg = sum(loss_history[-window-1:-1]) / window
    return loss_history[-1] > threshold * moving_avg

# 模拟训练 loss (含一个 spike)
losses = [2.0, 1.8, 1.5, 1.3, 1.1, 1.0, 5.0, 1.2, 1.1, 1.0]
for i in range(5, len(losses)):
    h = losses[:i+1]
    if detect_spike(h):
        print(f'step {i}: loss={losses[i]:.1f} SPIKE -> skip step (不更新权重)')
    else:
        print(f'step {i}: loss={losses[i]:.1f} 正常更新')
```

### 5.2 NaN 检测 + rollback

```python
import math
def check_nan(weights):
    """检查权重是否 NaN (需 rollback)."""
    return any(math.isnan(w) for w in weights)

# 模拟
weights = [1.0, float('nan'), 2.0]
if check_nan(weights):
    print('NaN detected -> rollback 到上一 checkpoint, 降 lr')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[gradient explosion and vanishing]]（spike 的数值根因）、训练动力学、bf16/fp16 精度。
- **下游（应用）**: [[clipping strategies]]（预防 spike）、[[warmup]]（预防早期 spike）、[[KL divergence control]]（RLHF spike 的 KL 维度）、[[reward normalization]]（RLHF reward 异常致 spike）、[[entropy regularization]]（RLHF 探索与稳定）、LLM 预训练/RLHF 工程实践、checkpoint 管理（rollback 基础）。
- **对比 / 易混**:
  - **spike vs [[gradient explosion and vanishing\|explosion]]**：spike 是 loss 飙升的现象，explosion 是梯度大的原因。spike 常由 explosion 引起，但也可由数据异常引起（非 explosion）。
  - **transient spike vs divergence**：前者恢复（几步后 loss 降回），后者发散（NaN/持续高）。应对不同：skip vs rollback。
  - **spike vs plateau**：spike 是突增（不稳定），plateau 是 loss 不降（不收敛）。不同问题，对策不同。


## 7. 常见误区与易错点

> [!warning] 误区 1：spike 无害会自恢复
> 不一定。transient spike 可能自恢复，但发散 spike（NaN）不恢复，训练崩。需 detect + 处理，不能忽视。忽视 transient spike 也可能累积成发散。

> [!warning] 误区 2：spike 一定 NaN
> 不一定。transient spike 只是 loss 高（非 NaN），权重正常，skip 几步可恢复。NaN 是更重的发散（权重坏，需 rollback）。区分两者：NaN 检查 vs loss 高。

> [!warning] 误区 3：gradient clip 完全防 spike
> 缓解不根治。clip 限梯度 norm 防爆炸传权重，但 spike 也可能由数据异常（非大梯度）或 attention 不稳引起。clip 是预防之一，非万能。需配合 warmup/bf16/QK-norm。

> [!warning] 误区 4：lr 大才 spike
> lr 大增 spike 概率，但小 lr 也可能 spike（数据异常/attention 不稳）。lr 是主因之一非唯一。调 lr 是手段之一，非全部。

> [!warning] 误区 5：spike 是随机噪声
> 不是。spike 有具体原因（梯度爆炸/数据异常/数值不稳）。反复 spike 需诊断根因（看梯度 norm/数据/attention logit），非"运气不好"。忽视根因会反复 spike。

> [!warning] 误区 6：skip step 总是安全
> skip 丢弃该 batch 梯度，简单有效，但若 spike 是权重已坏（NaN），skip 无用（权重坏，下步仍 NaN）。需先查 NaN，NaN 时 rollback 非 skip。

> [!warning] 误区 7：bf16 不会 spike
> bf16 范围大不易溢出（比 fp16 好），但精度低（7 bit mantissa），累加误差 + 小数值下溢仍可能不稳。bf16 减少 spike 但不消除。仍需 clip/warmup。


## 8. 延伸细节

### 8.1 LLM 预训练的 spike 案例

- **GPT-3 (175B)**：OpenAI 报告训练中多次 spike，用 skip + lower lr 处理。
- **PaLM (540B)**：Google 报告 spike，用 QK-norm + skip 处理。
- **LLaMA**：Meta 报告 spike，主要前期（warmup 阶段），warmup 缓解。

大模型 + 大 lr 的 spike 几乎不可避免，工程上需 watchdog 自动 detect + skip/rollback。

### 8.2 watchdog 的实现

训练框架（DeepSpeed/Megatron）的 watchdog：

- 每 step 检查 loss/梯度 norm/NaN。
- spike 时 skip（不 optimizer.step）。
- NaN 时 rollback + 降 lr + 告警。
- 定期存 checkpoint（每 N 步）供 rollback。

是 LLM 训练稳定性的工程基础设施。

### 8.3 数据异常的 spike

大数据集含异常样本：极端长度（超长序列 attention 爆）、乱码（tokenization 异常）、重复（loss 异常）。spike 时检查该 batch 的数据（长度/内容），若异常则 skip + 标记坏样本。数据过滤/质量检查是预防。

### 8.4 attention 数值不稳

QK^T 的大 logit（大 vocab + 深层）致 softmax 饱和（某 token 概率 ~1，其余 ~0），梯度消失/不稳。QK-norm：Q, K 先 LayerNorm/RMSNorm 再算 QK^T，控 logit 尺度。PaLM、Gemma 2 等用。是 LLM 稳定性 trick。

### 8.5 RLHF 的 spike 诊断

RLHF spike 时查：

- **KL 曲线**：突增 = policy 偏离过远（[[KL divergence control]] 失控）。
- **reward 值**：极端 = reward model 异常。
- **advantage**：高方差 = PPO 不稳。

根因不同对策不同：KL 失控增 $\beta$，reward 异常 normalize，advantage 不稳降 lr/增 batch。

### 8.6 skip vs rollback 的决策

- **loss 高但权重正常（无 NaN）**：skip step（丢弃梯度，下步继续）。
- **权重 NaN/inf**：rollback（回滚 checkpoint，权重已坏 skip 无用）。
- **持续 spike（多步不恢复）**：rollback + 降 lr。

watchdog 自动判断 NaN 决定 skip/rollback。

### 8.7 checkpoint 策略

rollback 需近期 good checkpoint。策略：

- 每 N 步存 checkpoint（N=100-1000）。
- 保留最近 M 个（旧删除）。
- spike 时回滚到最近 good（非 spike 前）。

checkpoint 存储开销（7B ~88GB/checkpoint），M=3-5 合理。详见显存/存储管理。

### 8.8 lr schedule 与 spike

LLM 训练的 lr schedule（warmup + cosine decay）：

- warmup（0-2%）：lr 从 0 增到 peak，spike 少（lr 小）。
- peak（~2%）：lr 最大，spike 多。
- decay（cosine 降）：lr 降，spike 少。

前期 warmup 防早期 spike，后期 decay 防晚期 spike。peak 阶段最易 spike，需 clip + watchdog。

---
相关: [[数值稳定]] | [[gradient explosion and vanishing]] | [[KL divergence control]] | [[clipping strategies]] | [[warmup]] | [[reward normalization]] | [[entropy regularization]] | [[混合精度]] | [[optimizer state memory]] | [[attention]] | [[nsys]] | [[torch profiler]]
