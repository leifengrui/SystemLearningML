# epoch与step

> **所属章节**: [[训练工程基础]]
> **所属模块**: [[01-ML深度学习基础]]
> **难度**: 基础

## 1. 一句话定义

**epoch（轮）** 是把训练数据**完整过一遍**；**step / iteration（步/迭代）** 是**一次参数更新**（一次 `optimizer.step()`）；两者是衡量训练进度的基本单位，关系为 `steps_per_epoch = 数据量 / batch_size`，`total_steps = epochs × steps_per_epoch`。

> [!note] 解答：多次 epoch 就是数据集多跑几次吗？
> **对，本质上就是这样——多次 epoch = 把整个训练集从头到尾完整过多次。** 但有几个关键点要讲清，否则会理解偏：
>
> 1. **每个 epoch 内部还要分批**：一遍数据不是一次性喂进去，而是切成很多 batch，每个 batch 做一次前向+反向+参数更新（一个 step）。所以"过一遍数据" = 跑 `steps_per_epoch = ⌈N/B⌉` 个 step。$N$ 个 epoch = 数据过 $N$ 遍 = $N\times\text{steps\_per\_epoch}$ 个 step。
> 2. **每个 epoch 通常会 shuffle（打乱）**：如果不打乱，多轮只是按固定顺序重复，模型会记住顺序而非学规律。`DataLoader(shuffle=True)` 每个 epoch 重新洗牌，于是同一样本在不同 epoch 里出现在不同 batch 组合里。
> 3. **同样的数据多轮为何还有用**：每过一个 step 参数就变了，所以第二轮再遇到同一样本时，模型对它的预测/梯度已和第一轮不同——"同样的数据，不同的模型状态"，还能继续优化。这就是多 epoch 能继续降 loss 的原因。
> 4. **太多了会过拟合**：轮数过多 → 模型开始死记训练样本（memorization），训练 loss 还在降但验证 loss 反弹，即 [[overfitting/underfitting]]。小数据集靠多轮 + 正则；LLM 数据极大，常**只过 1 遍甚至不到 1 遍**，靠数据质量而非多轮（见 3.3）。
>
> 示意：
> ```
> 1 epoch = [batch1→step1][batch2→step2]...[batchK→stepK]   （K=⌈N/B⌉）
> 3 epoch = 上面那段重复 3 次（每轮 shuffle 顺序不同）= 3K 个 step
> ```
>
> 一句话：**N epoch = 数据集完整过 N 遍（每遍打乱、分批、跑若干 step）**；小数据多轮练，LLM 数据太大基本不谈 epoch 而谈 tokens。详见 [[batch与mini-batch]]、本节 3.3。

## 2. 为什么需要它（动机与背景）

训练要回答两个问题："训了多久了？"和"还要训多久？"。需要可量化、可比较的进度单位：

- **epoch**：以"数据过了几遍"计，CV/小数据集常用（如 ImageNet 90 epoch）。
- **step**：以"更新了几次参数"计，大模型/LLM 常用（因为数据太大、只过 1 遍甚至不到 1 遍，epoch 失去意义）。

没有它们就无法对齐 [[学习率调度]]（cosine 用 total_steps）、checkpoint 间隔、日志频率。

## 3. 核心概念详解

### 3.1 定义

| 单位 | 定义 | 典型场景 |
|---|---|---|
| **epoch** | 完整数据集过一遍 | CV（ImageNet 90ep）、小数据集多轮 |
| **step / iteration** | 一次 optimizer.step() | 通用，LLM 主力 |
| **micro-step** | 一次前向+反向（grad accum 下） | 显存受限时 |

注意 grad accumulation 下：**多个 micro-step = 一个 step**（累积够才 step）。step 指真正更新参数的次数。

### 3.2 换算

$$
\text{steps\_per\_epoch} = \left\lceil \frac{N}{B} \right\rceil,\qquad
\text{total\_steps} = \text{epochs}\times\text{steps\_per\_epoch}
$$

- $N$：数据集样本数，$B$：batch size。
- [[学习率调度]] 的 cosine/warmup 都以 total_steps 为分母，算错会导致 lr 提前归零或没降够。

### 3.3 LLM 为什么不按 epoch

LLM 训练数据是万亿 token，**一遍都跑不完**（计算预算决定训多少 token，不是数据遍数）。所以 LLM 用：

- **tokens seen / tokens trained**：累计训练过的 token 数（如 1.5T tokens）。
- **step**：仍用，但每 step 处理固定 token 数（如 0.5M tokens/step），step 数 × tokens/step = 总 token。

GPT-3 训了 ~300B token，Llama2 训 2T token —— 都是 token 计量，不谈 epoch。

### 3.4 checkpoint 与日志按 step

- checkpoint：每隔 N step 存一次（如每 2000 step），便于断点续训。
- 日志/eval：每隔 N step 记录 loss/eval（如每 500 step）。
- 这些"每隔"都用 step 而非 epoch，因为大模型训练不按 epoch。

### 3.5 warmup / decay 的步数

[[学习率调度]] 的 warmup_steps、total_steps 都是 step 数。改 batch 或 accum 会改变 steps_per_epoch → total_steps 变 → 要重算调度参数。

## 4. 数学原理 / 公式

### 训练量与计算预算

LLM 的训练预算常以 **FLOPs** 或 **tokens** 衡量。Chinchonchilla 定律：最优计算下，tokens $\approx 20\times$参数量。如 70B 模型应训 ~1.4T tokens。

$$
\text{total\_tokens} = \text{total\_steps}\times\text{tokens\_per\_step}
$$

tokens_per_step = batch_size × seq_len（× grad accum）。

### 一步的计算量

一个 Transformer 层前向 FLOPs ≈ $6\times\text{params\_in\_layer}\times\text{tokens}$（前向+反向约 6×）。这是估算训练时间的依据。

## 5. 代码示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

N = 1000; B = 32
X = torch.randn(N, 10); Y = torch.randn(N, 2)
loader = DataLoader(TensorDataset(X, Y), batch_size=B, drop_last=True)
model = nn.Linear(10, 2)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

steps_per_epoch = len(loader)                  # 1000/32 = 31
epochs = 5
total_steps = epochs * steps_per_epoch         # 155
print(f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")

global_step = 0
for epoch in range(epochs):                    # epoch 循环
    for x, y in loader:                        # step 循环
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        global_step += 1
        if global_step % 10 == 0:              # 日志按 step
            print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")
        if global_step % 50 == 0:              # checkpoint 按 step
            torch.save({'step': global_step, 'model': model.state_dict()},
                       f'ckpt_{global_step}.pt')

# LLM 风格：按 token 计量
tokens_per_step = B * 4096                      # 假设 seq_len=4096
print(f"trained {global_step * tokens_per_step} tokens")
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[batch与mini-batch]]（batch 决定 steps_per_epoch）。
- **下游（应用）**: [[学习率调度]]（warmup/cosine 用 step 数）、checkpoint、日志、tokens seen（LLM 计量）、[[学习率调度]] 的 total_steps。
- **对比 / 易混**:
  - **step vs micro-step**：grad accum 下多个 micro-step 凑一个 step；只有 step 才 optimizer.step()。
  - **epoch vs tokens**：小数据用 epoch（多轮），LLM 用 tokens（不足一轮）。
  - **iteration vs step**：常混用，严格说 iteration 偏向"一次前向"（含 micro-step），step 偏向"一次更新"。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **LLM 报 epoch 数**：大模型训练不按 epoch，报 tokens/step 才有意义。
> 2. **total_steps 算错**：忘乘 grad accum 或 drop_last，导致 cosine 调度失准。
> 3. **scheduler.step() 用 epoch 还是 step**：取决于调度器设计，PyTorch 的 `LambdaLR` 按 step 调，`MultiStepLR` 可按 epoch；用错 lr 乱跳。
> 4. **checkpoint 按 epoch 存**：大模型一个 epoch 极长，应按 step 存，否则难得存一次。
> 5. **断点续训不恢复 global_step**：lr 调度依赖内部 step 计数，要随 state_dict 恢复。
> 6. **把 tokens_per_step 算成 batch_size**：LLM 要乘 seq_len，少乘一个数量级。

## 8. 延伸细节

### 8.1 tokens seen 与 scaling law

LLM 训练进度看 tokens seen。Scaling law（Chinchonchilla）给出"给定计算预算，最优参数×tokens 配比"，是决定训多少 step 的理论依据。

### 8.2 有效 epoch

数据去重、过滤后实际参与训练的样本数决定有效 epoch。LLM 常见数据只过 1 遍甚至不到，靠数据质量而非多轮。

### 8.3 eval 频率

训练长时 eval 要够频繁捕捉最优 checkpoint（LLM loss 非单调，可能中途最优）。常按 step（如每 2000 step eval）。

### 8.4 step 与梯度累加的 lr

grad accum 凑大 batch 时，scheduler 的 step 应按**优化器 step**（累积后）推进，而非 micro-step，否则 warmup 提前结束。

---
相关: [[训练工程基础]]、[[batch与mini-batch]]、[[学习率调度]]、[[mixed precision training]]
