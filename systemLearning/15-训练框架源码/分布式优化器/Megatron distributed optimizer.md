# Megatron distributed optimizer

> **所属章节**: [[分布式优化器]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: Megatron distributed optimizer / D-optimizer / 分布式优化器 / Megatron ZeRO / flat grad buffer / gradient accumulation bucket / 分布式优化器状态分片
> **难度**: 高（需懂 [[ZeRO (DeepSpeed)]] + [[Fully Sharded Data Parallel|FSDP]] + [[all-reduce]]/[[reduce-scatter]] + [[optimizer state memory]]）


## 1. 一句话定义

**Megatron distributed optimizer（D-optimizer）** 是 [[Megatron-LM]] 的**ZeRO-1/2 原生实现**——把**优化器状态**（Adam 的 fp32 $m, v$ + fp32 master weight）沿 **DP group** 切成 $D$ 份（每 rank 持 $1/D$，对应 [[ZeRO (DeepSpeed)]] 的 stage-1），并把**梯度**也切分（每 rank 只 reduce-scatter 自己负责的那 $1/D$ 片 grad，对应 ZeRO-2），用**flat grad buffer（梯度累积桶）**：前向/反向攒 grad 到 fp32 桶，桶满触发 `reduce-scatter`（async，每 rank 只收自己片），与下一层 backward overlap；optimizer step 时每 rank 只更新自己那 $1/D$ 的 master weight + optimizer state，再 `all-gather` 把更新后的 fp16/bf16 param 广播给所有 rank（主参数分片、副本参数完整供前向）。它把 Megatron 原本 DDP 风格（grad all-reduce + 每 rank 完整 optimizer state）的 optimizer state 显存从 $\propto N$ 降到 $\propto N/D$、grad 显存从 $\propto N$ 降到 $\propto N/D$，是 Megatron 显存优化的核心组件，与 [[FSDP2]]/[[ZeRO (DeepSpeed)]] 理念同但实现是 Megatron 手写 flat 化风格。

> [!note] 三句话定位
> - **是什么**：ZeRO-1/2 原生实现——optimizer state + grad 沿 DP 切 $D$ 份，flat grad bucket 攒 + reduce-scatter async + step 后 all-gather 回 param。
> - **为什么**：DDP 每 rank 完整 optimizer state（$\propto N$）显存爆；D-optimizer 切 $D$ 份（$\propto N/D$）+ grad reduce-scatter overlap 省 + 藏。
> - **与 [[ZeRO (DeepSpeed)]]/[[FSDP2]] 关系**：理念同（切 optimizer state/grad/param），DeepSpeed 是独立框架，FSDP2 是 PyTorch 原生 DTensor，D-optimizer 是 Megatron flat 化手写。三者等价概念。


## 2. 为什么需要它（动机与背景）

### 2.1 optimizer state 显存爆

LLM 的 optimizer state（Adam 的 fp32 $m, v$ + master weight）$\approx 3 \times$ model size（fp32 的 3 份）。70B 模型 optimizer state $\approx 840$ GB（fp32）。单 rank 完整存爆。是显存大头（仅次于 activation）。

### 2.2 DDP 每 rank 完整 optimizer state

[[Distributed Data Parallel|DDP]] 每 rank 算完整 grad（all-reduce 合），每 rank 完整 optimizer state（更新完整 model）。optimizer state $\propto N$ 每 rank。大模型不行。

### 2.3 ZeRO 的切分思路

[[ZeRO (DeepSpeed)]] 的 stage-1（切 optimizer state）、stage-2（切 grad）、stage-3（切 param）。每 rank 只持 $1/D$，显存 $\propto N/D$。Megatron 原生实现 ZeRO-1/2（不切 param，param 仍完整供前向，即 stage-1/2 hybrid）。

### 2.4 grad reduce-scatter 省 + overlap

ZeRO-2 的 grad 切分用 `reduce-scatter`（每 rank 只收自己片）替代 `all-reduce`（每 rank 收完整）。通信量同（reduce-scatter $\approx$ all-reduce 的一半，但 all-reduce = all-gather + reduce-scatter），但显存省（grad 只持 $1/D$ 片）。且 reduce-scatter async 与 backward overlap（[[gradient bucket与通信重叠]]）。

### 2.5 flat 化的 Megatron 风格

Megatron 把所有 param/grad/optimizer state **flat 化**（拼成一维 buffer），便于按 offset 切分与 reduce-scatter。是 Megatron 的手写风格（vs [[FSDP2]] 的 DTensor 声明式）。两者等价。

### 2.6 与 TP/PP 的组合

D-optimizer 沿 **DP group** 切（TP/PP 不切 optimizer state，因 TP/PP 已切 param 本身）。与 TP/PP 正交。是 3D 并行 + ZeRO 的显存栈。


## 3. 核心概念详解

### 3.1 三层显存切分

| ZeRO stage | 切什么 | 每 rank 显存 | 通信 |
|---|---|---|---|
| stage-0 (DDP) | 无 | $N$（param+grad+opt） | all-reduce grad |
| stage-1 | optimizer state | $\text{param} + \text{grad} + \text{opt}/D$ | all-reduce grad |
| stage-2 | + grad | $\text{param} + \text{grad}/D + \text{opt}/D$ | reduce-scatter grad |
| stage-3 (FSDP) | + param | $\text{param}/D + \text{grad}/D + \text{opt}/D$ | all-gather param + reduce-scatter grad |

D-optimizer 是 stage-1/2 hybrid（切 opt + grad，不切 param）。FSDP 是 stage-3。D-optimizer 不切 param（param 仍完整供前向，省 all-gather 通信）。

### 3.2 flat grad buffer

```python
# megatron/core/optimizer/grad_buffer.py
class GradBuffer:
    # fp32 flat buffer, 攒 grad
    def __init__(self, data_parallel_size):
        self.data = torch.zeros(flat_grad_size // D, ...)  # 每 rank 持 1/D
    def add_grad(self, param):
        # 把 param.grad 写入 flat buffer 的对应 offset
    def all_reduce_or_reduce_scatter(self):
        reduce_scatter(self.data, dp_group)  # async, 每 rank 只收自己片
```

flat grad buffer 把所有 param 的 grad 拼成一维，按 offset 切 $D$ 份，每 rank 持 $1/D$。攒满触发 reduce-scatter。

### 3.3 reduce-scatter 的 async overlap

```
反向:
  layer L backward -> grad 写入 grad buffer (offset)
  layer L-1 backward (计算) || grad buffer reduce-scatter (async, 通信)
  -> overlap (通信藏到 L-1 backward 后)
  layer L-2 backward -> ...
```

grad buffer 满触发 `reduce-scatter`（async），与下一层 backward overlap。是 [[gradient bucket与通信重叠]] 的核心。详见该笔记。

### 3.4 optimizer step（分片更新）

```python
# 每 rank 只更新自己 1/D 的 master weight + optimizer state
def step(self):
    # self.master_param_flat: 1/D 片 (本 rank 负责)
    # self.optimizer_state (m, v): 1/D 片
    grad_local = self.grad_buffer_local  # 1/D 片 (reduce-scatter 后)
    # Adam 更新 (只本 rank 片)
    self.master_param_flat += -lr * m / (sqrt(v) + eps)  # 只 1/D
    # all-gather 把更新后的 fp16/bf16 param 广播给所有 rank
    param_flat = all_gather(self.master_param_flat.half())  # 完整 param
    # 写回各 param.data (供前向)
    for param, offset in zip(params, offsets):
        param.data = param_flat[offset]
```

每 rank 只算自己 $1/D$ 的 optimizer step，再 all-gather 把更新后的 param 广播。是 ZeRO-1 的 step。

### 3.5 master weight 与 param 副本

```
master_param_flat (fp32): 切 D 片, 每 rank 1/D (optimizer 用)
param (fp16/bf16): 完整 (前向用, 是 master 的副本)
step: master 更新 (1/D) -> cast fp16 -> all-gather -> 写回 param
```

master weight 是 fp32（精度），param 是 fp16/bf16（显存/算力）。master 切 D 片，param 完整。是混合精度 + ZeRO-1。

### 3.6 与 DDP 的对比

| 维度 | DDP | D-optimizer (ZeRO-1/2) |
|---|---|---|
| optimizer state | 每 rank 完整 | 切 $D$ 片，每 rank $1/D$ |
| grad | all-reduce（每 rank 完整） | reduce-scatter（每 rank $1/D$） |
| param | 完整 | 完整（不切，stage-1/2） |
| 通信 | all-reduce grad | reduce-scatter grad + step 后 all-gather param |
| optimizer state 显存 | $\propto N$ | $\propto N/D$ |
| grad 显存 | $\propto N$ | $\propto N/D$ |

D-optimizer 省 optimizer state + grad 显存，增 step 后 all-gather param 通信（vs DDP 无）。

### 3.7 与 FSDP 的对比

| 维度 | D-optimizer (ZeRO-1/2) | FSDP (ZeRO-3) |
|---|---|---|
| param | 完整（前向用） | 切 $D$ 片，前向 all-gather 合 |
| grad | reduce-scatter $1/D$ | reduce-scatter $1/D$ |
| optimizer state | 切 $1/D$ | 切 $1/D$ |
| param 显存 | $\propto N$ | $\propto N/D$ |
| 通信 | reduce-scatter grad + step all-gather | 前向 all-gather param + 反向 reduce-scatter grad |

D-optimizer 不切 param（省前向 all-gather 通信），FSDP 切 param（省 param 显存）。D-optimizer 通信少，FSDP 显存省。大模型（param 也大）用 FSDP，中模型用 D-optimizer。

### 3.8 与 TP/PP 的正交

D-optimizer 沿 **DP group** 切。TP/PP 切 param 本身（不是 optimizer state）。TP/PP 的 optimizer state 沿 DP 切（D-optimizer）。是 3D 并行 + ZeRO 的组合（TP+PP+DP+ZeRO-1/2）。详见 [[EP TP DP CP组合规则]]。

### 3.9 grad accumulation（微批）

micro-batching 的 grad 累积：多 micro-batch 的 grad 累加到 flat grad buffer（不 step）。攒够 $M$ 个 micro-batch（一个 data-parallel batch）才 step。是 [[micro-batching]] 与 D-optimizer 的配合。grad buffer 支持累加（`+=`）。

### 3.10 实现的 distrib_optimizer.py

```python
# megatron/core/optimizer/distrib_optimizer.py
class DistributedOptimizer(MixedPrecisionOptimizer):
    def __init__(self, ...):
        self.master_param_flat = ...  # 切 D 片
        self.grad_buffer = GradBuffer(...)  # flat, reduce-scatter
    def step(self):
        self.grad_buffer.reduce_scatter()  # async, overlap
        # 本 rank 1/D step
        self.master_param_flat += update  # Adam on 1/D
        param = all_gather(self.master_param_flat.half())  # 广播
```


## 4. 数学原理 / 公式

### 4.1 optimizer state 显存

设 model size $N$（param 数），DP size $D$。Adam optimizer state = fp32 $m, v$ + master weight = $3N$（fp32，12 bytes/param）。D-optimizer 切 $D$：每 rank $3N/D$。省 $D$ 倍。见 [[optimizer state memory]]。

### 4.2 grad 显存

grad = $N$（fp32，4 bytes/param，或 fp16/bf16）。DDP 每 rank 完整 $N$。D-optimizer（ZeRO-2）切 $D$：每 rank $N/D$。省 $D$ 倍。

### 4.3 param 显存

param = $N$（fp16/bf16，2 bytes/param）。D-optimizer 不切：每 rank $N$（完整供前向）。FSDP 切 $D$：$N/D$。D-optimizer 不省 param。

### 4.4 通信量

DDP：grad all-reduce $\approx 2N$（ring）。D-optimizer：grad reduce-scatter $\approx N$（每 rank 收 $N/D$）+ step 后 all-gather param $\approx N$（广播）。总 $\approx 2N$（与 DDP 同）。但 D-optimizer 的 reduce-scatter 与 backward overlap（藏），all-gather 与 step overlap。wall-clock 可能更优。

### 4.5 flat 化的 offset

所有 param 拼成一维 flat buffer（size $N$）。每 rank 的 $1/D$ 片是 flat buffer 的 $[r \cdot N/D, (r+1) \cdot N/D)$。reduce-scatter 后每 rank 持自己片。master weight flat 同 offset。是 Megatron flat 化风格。

### 4.6 算力

optimizer step 只算 $1/D$（每 rank）。算力切分 $1/D$。all-gather 不算（仅通信）。


## 5. 代码示例（可选）

### 5.1 Megatron D-optimizer 用法

```python
from megatron.core.optimizer import DistributedOptimizer, Adam
base_opt = Adam(lr=1e-4)  # 基础 Adam
opt = DistributedOptimizer(base_opt, model, ...)
# 自动: flat grad buffer + reduce-scatter + 1/D step + all-gather
for batch in dataloader:
    loss = forward_backward(model, batch)  # grad 攒到 flat buffer
    opt.step()  # reduce-scatter (async overlap) + 1/D step + all-gather
    opt.zero_grad()  # 清 flat buffer
```

### 5.2 flat grad buffer 概念

```python
# 所有 param.grad 拼成一维 flat buffer
grad_flat = torch.zeros(N // D)  # 每 rank 1/D
for param in model.parameters():
    offset = param_to_offset[param]
    grad_flat[offset:offset+param.numel()//D] = param.grad.flat[...]  # 本 rank 片
# reduce-scatter async
reduce_scatter(grad_flat, dp_group, async_op=True)  # 与下一层 backward overlap
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[ZeRO (DeepSpeed)]]（理念来源）、[[Fully Sharded Data Parallel|FSDP]]（对比）、[[Distributed Data Parallel|DDP]]（基础）、[[all-reduce]]/[[reduce-scatter]]/[[AllGather]]（通信原语）、[[optimizer state memory]]、[[Megatron Core目录与执行链路]]、[[混合精度训练]]（master weight）。
- **下游（应用）**: [[gradient bucket与通信重叠]]（overlap 实现）、[[Megatron-LM]] 的优化器、[[3D parallelism]]（与 TP/PP 组合）、[[Megatron distributed checkpoint]]（存分片 opt state）、[[训练显存估计]]。
- **对比 / 易混**:
  - **D-optimizer vs [[ZeRO (DeepSpeed)]]**：理念同（ZeRO-1/2），DeepSpeed 是独立框架（`DeepSpeedEngine`），D-optimizer 是 Megatron 手写 flat 化。等价。
  - **D-optimizer vs [[FSDP2]]**：D-optimizer 切 opt+grad（stage-1/2，param 完整），FSDP2 切 opt+grad+param（stage-3，param 切）。FSDP2 更省显存（param），D-optimizer 通信少（不 all-gather param 前向）。大模型用 FSDP2，中模型用 D-optimizer。
  - **D-optimizer vs DDP**：DDP 不切（全完整），D-optimizer 切 opt+grad。D-optimizer 省 $D$ 倍 opt/grad 显存，增 step all-gather 通信。
  - **flat grad buffer vs [[gradient bucket与通信重叠]]**：flat grad buffer 是攒 grad 的机制，[[gradient bucket与通信重叠]] 是其 async overlap 的策略。前者是结构，后者是时序。


## 7. 常见误区与易错点

> [!warning] 误区 1：D-optimizer 切 param
> D-optimizer 是 ZeRO-1/2（切 opt + grad），**不切 param**（param 完整供前向）。FSDP 是 ZeRO-3（切 param）。误以为切 param 会与 FSDP 混。

> [!warning] 误区 2：D-optimizer 通信比 DDP 多
> 通信量相近（reduce-scatter + all-gather $\approx$ all-reduce 的两部分）。但 reduce-scatter 与 backward overlap（藏），wall-clock 可能更优。误以为多通信会错判。

> [!warning] 误区 3：flat 化破坏 per-param
> flat 化是内部表示（按 offset 切分），per-param 的 name/shape 仍保留（param.data 按 offset 写回）。flat 是 buffer 组织，不破坏 per-param 边界。与 FSDP1 的 flat param（破坏边界）不同。

> [!warning] 误区 4：master weight 不切
> master weight（fp32）切 $D$ 片（与 optimizer state 一起，是 ZeRO-1 的 opt 的一部分）。每 rank $1/D$。param（fp16/bf16）不切（完整副本）。误以为 master 完整会显存爆。

> [!warning] 误区 5：reduce-scatter 同步阻塞
> reduce-scatter 用 async（`async_op=True`），与下一层 backward overlap。不阻塞。误以为同步会漏 overlap。

> [!warning] 误区 6：D-optimizer 与 TP 冲突
> 正交。D-optimizer 沿 DP 切，TP 沿 tp mesh 切 param。3D 并行（TP+PP+DP+ZeRO）可组合。误以为冲突会漏配。

> [!warning] 误区 7：step 后 all-gather 不算通信
> step 后 all-gather 把更新 param 广播，通信 $\propto N$。是 D-optimizer 的额外通信（vs DDP 无）。但与 step overlap（step 算 $1/D$，all-gather 通信藏）。误以为不算会低估通信。


## 8. 延伸细节

### 8.1 与 [[gradient bucket与通信重叠]] 的协同

D-optimizer 的 flat grad buffer 是 [[gradient bucket与通信重叠]] 的载体。bucket 满 → reduce-scatter async → 与 backward overlap。两者是结构与时序的关系。详见 [[gradient bucket与通信重叠]]。

### 8.2 CPU offload 的扩展

D-optimizer 可 CPU offload optimizer state（master weight + $m, v$ 放 CPU，用时 PCIe 传）。是 ZeRO-Offload 思路。见 [[offloading (CPU-NVMe)]]。

### 8.3 与 [[FSDP2]] 的趋同

Megatron 的 D-optimizer（手写 flat）与 [[FSDP2]]（DTensor 声明式）理念同。Megatron Core 在向 DTensor/DeviceMesh 重构。未来 D-optimizer 可能用 DTensor 实现。详见 [[DeepSpeed ZeRO vs FSDP vs Megatron]]。

### 8.4 内容来源

Megatron distributed optimizer 整理自 Megatron-LM 论文 v4（"ZeRO"）、`megatron/core/optimizer/distrib_optimizer.py` + `grad_buffer.py` 源码。与 ZeRO/FSDP 的对比见各自笔记与 [[DeepSpeed ZeRO vs FSDP vs Megatron]]。

---
相关: [[分布式优化器]] | [[ZeRO (DeepSpeed)]] | [[Fully Sharded Data Parallel]] | [[FSDP2]] | [[Distributed Data Parallel]] | [[gradient bucket与通信重叠]] | [[reduce-scatter]] | [[all-reduce]] | [[AllGather]] | [[optimizer state memory]] | [[混合精度训练]] | [[Megatron Core目录与执行链路]] | [[3D parallelism]] | [[Megatron distributed checkpoint]] | [[训练显存估计]] | [[offloading (CPU-NVMe)]] | [[DeepSpeed ZeRO vs FSDP vs Megatron]] | [[micro-batching]]
