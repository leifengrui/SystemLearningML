# Interleaved schedule

> **所属章节**: [[Pipeline调度]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: Interleaved schedule / 1F1B-interleaved / virtual pipeline stage / interleaved 1F1B / 虚拟流水线 / chunk schedule
> **难度**: 高（需懂 [[1F1B调度]] + [[Pipeline Parallel]] + [[Megatron Core目录与执行链路]]）


## 1. 一句话定义

**Interleaved schedule（1F1B-interleaved，虚拟流水线）** 是 [[Megatron-LM]] 对 [[1F1B调度]] 的**减气泡演进**——把模型分 $L$ 层切成 $P \times V$ 个 **virtual stage**（chunk），每 device 持 $V$ 个**不连续**的 virtual stage（如 device $i$ 持 chunk $i, i+P, i+2P, \dots$，而非 1F1B 的连续 $L/P$ 层）；调度时 device 在自己的 $V$ 个 virtual stage 间**轮转** forward/backward，让相邻 device 的 micro-batch 流动更密集，气泡从 [[1F1B调度]] 的 $\frac{P-1}{M+P-1}$ 降到 $\approx \frac{P-1}{M+P-1} \cdot \frac{1}{V}$（**减 $V$ 倍**）。代价：跨 virtual stage 的 `send/recv` 通信增 $V$ 倍（device 内 $V$ 个 chunk 间需通信）+ 调度逻辑复杂（轮转 + 多 stage 跟踪）。是 Megatron PP 大 $P$（多 stage）减气泡的标准，$V=1$ 退化为 1F1B。适合 $P$ 大、气泡显著的场景（如 $P=8/16$ stage）。

> [!note] 三句话定位
> - **是什么**：模型切 $P \times V$ virtual stage，每 device 持 $V$ 个不连续 chunk，轮转 forward/backward，气泡减 $V$ 倍。
> - **为什么**：1F1B 气泡 $\frac{P-1}{M+P-1}$ 在 $P$ 大时显著；interleaved 用虚拟流水线让流动更密集，减 $V$ 倍气泡。
> - **与 [[1F1B调度]] 关系**：$V=1$ 退化为 1F1B。interleaved 是 1F1B 的演进（减气泡 + 增通信复杂度）。


## 2. 为什么需要它（动机与背景）

### 2.1 1F1B 的气泡在 P 大时显著

[[1F1B调度]] 气泡 $\frac{P-1}{M+P-1}$。$P$ 大（多 stage，如 8/16）时气泡显著（$P=8, M=64$，气泡 $\approx 10\%$；$P=16$ 更大）。大模型常需多 stage（单 device 装不下），气泡成瓶颈。

### 2.2 虚拟流水线减气泡

interleaved：每 device 持 $V$ 个不连续 virtual stage，轮转调度让 device 间 micro-batch 流动更密集（一个 device 算完 chunk $i$ 立刻算 chunk $i+P$，不等下流）。气泡减 $V$ 倍（$\frac{P-1}{M+P-1}/V$）。是 Megatron 减气泡的关键。

### 2.3 代价：通信增 V 倍

device 内 $V$ 个 virtual stage 不连续，需跨 chunk 通信（forward chunk $i$ → chunk $i+1$ 在不同 device）。通信增 $V$ 倍（vs 1F1B 的 device 内连续不需跨 device 通信）。是减气泡的 trade-off。

### 2.4 调度复杂度

轮转 $V$ 个 virtual stage，需跟踪每 micro-batch 在哪个 virtual stage、何时 forward/backward。调度逻辑复杂（Megatron `schedules.py` 的 interleaved 实现）。$V$ 大调度更难。

### 2.5 适合 P 大

$P$ 小（如 4）时气泡小，interleaved 的通信增量不值。$P$ 大（如 8/16）时气泡大，interleaved 减气泡赚。是大模型多 stage 的标准。


## 3. 核心概念详解

### 3.1 virtual stage 的切分

```
设 P=4 device, V=2 virtual stage per device, model 16 layer.
共 P*V=8 virtual stage (每 2 layer).

切分:
  device 0: virtual stage 0 (layer 0-1) + virtual stage 4 (layer 8-9)
  device 1: virtual stage 1 (layer 2-3) + virtual stage 5 (layer 10-11)
  device 2: virtual stage 2 (layer 4-5) + virtual stage 6 (layer 12-13)
  device 3: virtual stage 3 (layer 6-7) + virtual stage 7 (layer 14-15)

vs 1F1B (V=1):
  device 0: layer 0-3
  device 1: layer 4-7
  device 2: layer 8-11
  device 3: layer 12-15
```

每 device 持 $V$ 个不连续 chunk（间隔 $P$ 个 virtual stage）。device 内 forward 顺序：先 chunk 0，再 chunk 4（不连续，需跨 device 通信）。

### 3.2 轮转调度

```
device 0 的 forward 顺序 (1 micro-batch):
  chunk 0 (vs0) -> [send to device 1] -> chunk 4 (vs4) -> [send to device 1] -> done

vs 1F1B (V=1):
  chunk (layer 0-3) 全在 device 0, 无跨 device 通信 (连续)
```

interleaved 每 chunk 后需 send 到下一 virtual stage（不同 device）。$V$ 个 chunk $V$ 次 send。vs 1F1B 的 1 次（device 内连续）。

### 3.3 气泡减 V 倍

```
1F1B 气泡: (P-1)/(M+P-1)
interleaved 气泡: (P-1)/(M+P-1) / V  (近似, 因 V chunk 轮转让流动密集)

例: P=4, M=8, V=2
  1F1B: 3/11 ≈ 27%
  interleaved: 3/11 / 2 ≈ 13.5% (减半)
```

减 $V$ 倍气泡。$V$ 越大减越多（但通信增 $V$ 倍）。

### 3.4 通信增 V 倍

1F1B 每 micro-batch 每 device 1 次 send/recv（跨 stage，device 内连续不通信）。interleaved 每 micro-batch 每 device $V$ 次 send/recv（$V$ 个 chunk 各跨 device）。通信增 $V$ 倍。

### 3.5 调度的三阶段

interleaved 也有 warmup/steady/cooldown，但每 stage 处理 $V$ 个 virtual stage 的轮转。warmup 数 $\approx (P-1) \cdot V$（更多，因 $V$ chunk）。steady 1F1B 交错（每 chunk 各 1F1B）。cooldown 排空。

### 3.6 激活显存

interleaved 每 device 存 $V$ 个 virtual stage 的激活，每 virtual stage 存 $\approx P$ 份 micro-batch（warmup）。总激活 $\approx V \cdot P$ 份（vs 1F1B 的 $P$ 份）。增 $V$ 倍。是减气泡的显存代价。配 [[gradient checkpointing]] 减。

### 3.7 V 的选择

$V$ 大：气泡减多，但通信增、显存增、调度复杂。$V$ 小（如 2-4）：折中。$V=1$ 退化为 1F1B。典型 $V=2-8$（据 $P$ 与通信带宽）。

### 3.8 实现的 schedules.py

```python
# megatron/core/pipeline_parallel/schedules.py
def forward_backward_pipelining_with_interleaving(...):
    num_virtual_stages = V  # virtual stage 数 per device
    for vs in range(num_virtual_stages):  # 轮转 V 个 virtual stage
        forward_step(vs)  # chunk vs forward
        # send to next device (跨 device)
    # warmup -> steady 1F1B (V 轮转) -> cooldown
```

### 3.9 与 1F1B 的关系

$V=1$：interleaved = 1F1B（每 device 1 连续 chunk，无跨 device 通信）。$V>1$：interleaved 减气泡 + 增通信/显存。interleaved 是 1F1B 的超集（$V$ 参数化）。

### 3.10 与 3D 并行的配合

interleaved（PP）+ TP + DP 组成 3D 并行。PP 用 interleaved 减气泡，TP 切 weight，DP 切 batch。是 LLM 大模型训练的标准（如 $P=8$ interleaved $V=2$ + $T=8$ TP + $D=N/(8\cdot 8)$ DP）。


## 4. 数学原理 / 公式

### 4.1 气泡比例

$$\text{bubble}_{\text{interleaved}} \approx \frac{1}{V} \cdot \frac{P-1}{M+P-1}$$

vs 1F1B 的 $\frac{P-1}{M+P-1}$。减 $V$ 倍（近似，实际有边界效应）。

### 4.2 通信量

1F1B：每 micro-batch 每 device 1 次 send/recv，$\propto B \cdot S \cdot d_{\text{chunk}}$。interleaved：$V$ 次，$\propto V \cdot B \cdot S \cdot d_{\text{chunk}}$。增 $V$ 倍。但每次 chunk 更小（$d_{\text{chunk}} = d/V$），总通信 $\propto V \cdot B \cdot S \cdot (d/V) = B \cdot S \cdot d$（与 1F1B 同！因 chunk 切小）。实际通信量相近，但次数多（latency 增）。

### 4.3 激活显存

1F1B：$P$ 份激活/device。interleaved：$V \cdot P$ 份/device（$V$ 个 virtual stage 各 $P$ 份）。增 $V$ 倍。

### 4.4 V 的 trade-off

减气泡收益 $\propto 1/V$。通信次数 $\propto V$（latency $\alpha$ 项增）。显存 $\propto V$。最优 $V$ 据带宽/延迟/显存平衡。高带宽（NVLink）可大 $V$，低带宽（IB）小 $V$。见 [[alpha-beta性能模型]]。


## 5. 代码示例（可选）

### 5.1 Megatron interleaved 调度

```python
from megatron.core.pipeline_parallel import schedules
# 开 interleaving: num_model_chunks = V (virtual stage 数 per device)
loss = schedules.forward_backward_pipelining_with_interleaving(
    model, batch, forward_step, backward_step,
    pipeline_parallel_size=8, num_model_chunks=2,  # V=2
    micro_batch_size=...)
# 内部: 每 device 2 virtual stage 轮转, 气泡减 2 倍
```

### 5.2 virtual stage 切分

```python
# model 16 layer, P=4, V=2
# 总 virtual stage = P*V = 8 (每 2 layer 一 chunk)
# device 0: chunk 0 (layer 0-1) + chunk 4 (layer 8-9)
# device 1: chunk 1 (layer 2-3) + chunk 5 (layer 10-11)
# ...
# forward: chunk 0 -> send -> chunk 1 (device 1) -> ... -> chunk 7 -> send -> device 0 chunk 4 -> ...
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[1F1B调度]]（基础）、[[Pipeline Parallel]]、[[Megatron Core目录与执行链路]]、[[overlap strategy]]、[[activation memory]]、[[alpha-beta性能模型]]。
- **下游（应用）**: [[Megatron-LM]] 的 PP 高级实现、[[3D parallelism]]（与 TP/DP 组合）、大模型多 stage 训练、[[gradient checkpointing]]（互补省显存）。
- **对比 / 易混**:
  - **interleaved vs [[1F1B调度]]**：$V=1$ 退化为 1F1B。interleaved 减气泡 $V$ 倍，增通信次数 $V$ 倍 + 显存 $V$ 倍。是 trade-off。
  - **interleaved 的 virtual stage vs 1F1B 的 stage**：1F1B 每 device 1 连续 stage（layer 段）；interleaved 每 device $V$ 个不连续 virtual stage（chunk）。virtual stage 是更细切分。
  - **interleaved vs [[Sequence Parallel]]**：interleaved 切层维（PP，减气泡）；SP 切 sequence 维（TP 的显存优化）。正交，可共存。


## 7. 常见误区与易错点

> [!warning] 误区 1：interleaved 增通信量
> 通信**次数**增 $V$ 倍，但每次 chunk 更小（$d/V$），**总通信量**相近（$\propto B \cdot S \cdot d$）。增的是 latency（次数多 $\alpha$ 项）。误以为总通信增 $V$ 倍会错判。高带宽网络下 latency 可藏。

> [!warning] 误区 2：V 越大越好
> $V$ 大减气泡，但增 latency（次数）+ 显存 + 调度复杂。最优 $V$ 据带宽/延迟/显存平衡。$V$ 太大会 latency 主导。典型 $V=2-8$。

> [!warning] 误区 3：interleaved 省 activation
> 错。interleaved 增 activation（$V \cdot P$ 份 vs 1F1B 的 $P$）。减气泡的显存代价。需配 [[gradient checkpointing]] 减。

> [!warning] 误区 4：V=0 是 1F1B
> $V=1$ 是 1F1B（每 device 1 chunk）。$V=0$ 无意义。interleaved 的 $V \geq 1$。

> [!warning] 误区 5：virtual stage 是连续层
> virtual stage 是 chunk（不连续，间隔 $P$）。device 0 的 chunk 0 + chunk 4（非连续层）。误以为连续会错配 model 切分。

> [!warning] 误区 6：interleaved 与 TP 冲突
> 正交。interleaved（PP）+ TP + DP 可组合。3D 并行 + interleaved。误以为冲突会漏配。

> [!warning] 误区 7：interleaved 气泡精确除 V
> $\frac{P-1}{M+P-1}/V$ 是近似。实际有边界效应（warmup/cooldown 的 virtual stage 轮转不完全等价）。大 $M$ 时近似好，小 $M$ 时偏差。


## 8. 延伸细节

### 8.1 interleaved 的 ZB-H1 等演进

ZB-H1（Zero Bubble）等后续调度进一步减气泡（用异步 backward + grad 通信隐藏）。是 interleaved 之后的演进。详见 [[Pipeline Parallel]] 的调度演进。

### 8.2 与 [[overlap strategy]]

interleaved 的 send/recv async 与轮转计算 overlap。是 PP overlap 的复杂版（$V$ 轮转需精细 overlap 调度）。

### 8.3 内容来源

interleaved schedule 整理自 Megatron-LM 论文 v3（"Interleaved Pipeline"）、`megatron/core/pipeline_parallel/schedules.py` 源码。气泡/通信分析与 $V$ 的 trade-off 见论文。与 1F1B 的关系见 [[1F1B调度]]。

---
相关: [[Pipeline调度]] | [[1F1B调度]] | [[Pipeline Parallel]] | [[Megatron Core目录与执行链路]] | [[overlap strategy]] | [[activation memory]] | [[3D parallelism]] | [[gradient checkpointing]] | [[Megatron-LM]] | [[alpha-beta性能模型]] | [[compute vs communication bottleneck]]
