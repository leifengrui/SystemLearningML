# loss scaling

> **所属章节**: [[训练工程基础]]
> **所属模块**: [[01-ML深度学习基础]]
> **难度**: 中级

## 1. 一句话定义

**loss scaling（损失缩放，损失缩放）** 是 **fp16 混合精度训练**的配套技术：前向前把 loss 乘一个大数 $S$（如 65536），让所有梯度同步放大 $S$ 倍、抬出 fp16 的下溢区，反向后、optimizer 前再把梯度除 $S$ 缩回原量级；**bf16 不需要**（动态范围同 fp32，不下溢）。

## 2. 为什么需要它（动机与背景）

fp16 的痛点（见 [[数值类型与精度]]、[[mixed precision training]]）：

- fp16 最小正规数 $\approx 6.1\times10^{-5}$，**小于它的梯度直接下溢成 0**。
- 深度学习很多梯度量级在 $10^{-7}\sim10^{-4}$，fp16 下大量被截成 0 → 梯度信息丢失、训练停滞或发散。

loss scaling 的思路极简：**既然小梯度会下溢，就先把 loss 放大，梯度跟着放大、不再下溢；用完再缩回去**。一个乘除就把 fp16 的下溢问题解决（但解决不了上溢，那是 bf16 的优势）。

> [!note] 只对 fp16 有效
> loss scaling 只解决"下溢"。bf16 范围同 fp32，小梯度不下溢，**不需要也不应该开 scaler**。bf16 开 scaler 不仅无益，GradScaler 的 overflow 检测还可能误判 skip step。判断：用 fp16 才开，用 bf16 不开。

## 3. 核心概念详解

### 3.1 工作流（fp16 AMP）

```
1. 前向（autocast fp16）算 loss
2. loss = loss * S                  ← scale：放大 loss
3. loss.backward()                  ← 梯度 = 真梯度 * S，抬出下溢区
4. scaler.unscale_(opt)             ← unscale：把 .grad 除以 S 缩回真量级
5. （可选）clip_grad_norm_          ← unscale 之后才能正确裁剪
6. opt.step()                       ← 用真梯度更新
7. scaler.update()                  ← 动态调整 S
```

关键：scale 在 backward **前**（放大梯度）、unscale 在 step **前**（缩回真值）。

### 3.2 dynamic scaling（动态调整 scale）

$S$ 太小 → 仍下溢；$S$ 太大 → 梯度超过 65504 上溢成 inf。PyTorch `GradScaler` 用**动态调整**：

- 反向后检测梯度是否含 inf/nan：
  - **无 inf/nan** → 正常 step，且若连续若干步没溢出，把 $S$ 翻倍（找更大可用 scale）。
  - **有 inf/nan** → **跳过本次 step**（不更新参数），把 $S$ 减半（降到一个不溢出的 scale）。
- 自动找平衡点，无需手调。

> [!tip] skip step 不是 bug
> 检测到 inf/nan 时 scaler **故意跳过这一步**（不更新参数），是为了防止用坏梯度更新炸模型。这是设计，不是故障。频繁 skip 说明 $S$ 太大或训练本身不稳。

### 3.3 overflow 检测原理

反向后，scaler 检查所有参数 `.grad` 是否含 inf/nan。fp16 下 inf 来自数值超过 65504。若 $S$ 太大使放大后的梯度越界 → 检测到 → 降 $S$。

### 3.4 static vs dynamic

- **static**：固定 $S$（如 128），简单但要手调、换模型/换 lr 可能要重调。
- **dynamic**（PyTorch 默认）：自动找 $S$，鲁棒，主流用这个。

## 4. 数学原理 / 公式

### 放大-缩放

设真梯度 $g$，scale factor $S$：

$$
\tilde g = S\cdot g \quad(\text{反向得到的"放大梯度"})
$$

若 $g < 6.1\times10^{-5}$（fp16 下溢），但 $S\cdot g > 6.1\times10^{-5}$，则 $\tilde g$ 不下溢、被正确表示。step 前缩回：

$$
g = \tilde g / S
$$

整个过程对优化器等价（乘除抵消），但中间表示避开了下溢。

### 与 clip 的顺序

clip 阈值针对**真梯度范数** $\|g\|$。若 unscale 前就 clip，clip 的是 $\|\tilde g\|=S\|g\|$，阈值被等效放大 $S$ 倍 → 形同虚设。故必须 **unscale 后再 clip**。

### 为什么 bf16 不需要

bf16 最小正规数 $\approx 1.2\times10^{-38}$（同 fp32），深度学习梯度远大于此，不下溢。放大无意义，反而可能推向上溢（bf16 max $3.4\times10^{38}$，放大后大梯度也可能越界）。故 bf16 路线直接 `loss.backward()`，无 scaler。

## 5. 代码示例

```python
import torch
import torch.nn as nn

model = nn.Linear(1000, 1000).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
x = torch.randn(64, 1000).cuda(); y = torch.randn(64, 1000).cuda()
lossf = nn.MSELoss()

# fp16 AMP + dynamic loss scaling（V100 场景）
scaler = torch.cuda.amp.GradScaler()          # 动态 scale，初始 65536
for step in range(20):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        out = model(x); loss = lossf(out, y)
    scaler.scale(loss).backward()             # 放大 loss 后反向
    scaler.unscale_(opt)                      # 缩回真梯度（clip 前必须）
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                          # 内部检测 inf/nan：有则 skip
    scaler.update()                           # 动态调 S
    print(step, "scale=", scaler.get_scale())

# bf16 路线：不要 scaler（错误示范）
# scaler = GradScaler()  # bf16 不需要，开了可能误 skip step

# 手动 static scaling（理解原理）
S = 1024.0
loss = lossf(model(x), y) * S
loss.backward()
for p in model.parameters():
    p.grad.div_(S)                            # 缩回
opt.step()
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[mixed precision training]]（scaling 是 fp16 路线的子组件）、[[数值类型与精度]]（下溢的根源）。
- **下游（应用）**: fp16 训练稳定性、与 [[梯度裁剪]] 的顺序协同、[[Adam与AdamW]]（unscale 后 step）。
- **对比 / 易混**:
  - **loss scaling vs gradient clipping**：scaling 是"放大防下溢"（fp16 训练，乘除抵消不改真值）；clipping 是"截断防上溢"（改真值，限步长）。一个防太小、一个防太大，可叠加，顺序：scale→backward→unscale→clip→step。
  - **loss scaling vs loss 归一化**：scaling 是数值技术（放大再缩回）；归一化是按 batch/序列长度除（改真 loss 量级，影响 lr）。
  - **dynamic vs static scaling**：见 3.4。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **bf16 也开 scaler**：bf16 不下溢，scaler 无益且误判 skip。只 fp16 开。
> 2. **clip 前忘 unscale**：clip 的是放大 $S$ 倍的梯度，阈值失效。
> 3. **把 skip step 当 bug**：检测 inf/nan 跳过 step 是设计，频繁 skip 才需排查（$S$ 太大或训练不稳）。
> 4. **scale 写在 backward 后**：必须在 backward 前 scale loss，否则梯度没被放大、照样下溢。
> 5. **手动 static 不随模型调**：换模型/lr/batch 后固定 $S$ 可能不合适，dynamic 更省心。
> 6. **以为 scaling 解决所有 fp16 问题**：只解决下溢，不解决上溢（大梯度超 65504 仍 inf），那是换 bf16 的理由。
> 7. **scaler.step 里 opt 已被 unscale 两次**：`scaler.step` 内部会 unscale，若你已手动 `unscale_` 又让 scaler 再 unscale 会报错；用 `unscale_` 后应直接 `opt.step()`，或让 `scaler.step` 全权处理。

## 8. 延伸细节

### 8.1 为什么初始 scale 是 65536

2^16，让 fp16 梯度范围从 $[6.1\times10^{-5}, 65504]$ 等效扩展到 $[9.3\times10^{-10}, 1.1\times10^9]$，覆盖典型梯度量级。dynamic 会从这开始自适应。

### 8.2 skip step 对训练的影响

skip 一步等于该步不更新参数，偶尔 skip 无害（就像丢一个坏样本），但频繁 skip 意味着 scale 一直没调好或训练数值不稳，要看 grad/loss 是否有异常。

### 8.3 fp8 的 scaling

fp8 范围更窄，必须配 **per-tensor scaling factor**：每个张量算一个 scale，把它的数值映射进 fp8 范围，运算后再反缩。比 fp16 的全局 loss scaling 复杂，是 fp8 训练的核心难点。

### 8.4 与分布式通信

fp16 训练时 all-reduce 的梯度也是放大后的（$S\cdot g$），通信后再 unscale；或在通信前 unscale 传真梯度。框架内部处理，注意精度。

### 8.5 bf16 取代 fp16 的趋势

A100/H100 普及后，bf16 硬件支持成熟，LLM 训练几乎全转 bf16，loss scaling 在新流程里逐渐少见。但理解它对读懂 fp16 时代代码、排查老模型仍有价值。

---
相关: [[训练工程基础]]、[[mixed precision training]]、[[数值类型与精度]]、[[梯度裁剪]]、[[Adam与AdamW]]
