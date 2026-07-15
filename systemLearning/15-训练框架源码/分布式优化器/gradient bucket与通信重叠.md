# gradient bucket与通信重叠

> **所属章节**: [[分布式优化器]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: gradient bucket / 梯度桶 / 通信与计算重叠 / reducer / NCCL async / all-reduce async overlap / bucket-based gradient sync
> **难度**: 中高（需懂 [[Megatron distributed optimizer]] + [[Distributed Data Parallel|DDP]] + [[all-reduce]]/[[reduce-scatter]] + [[overlap strategy]]）


## 1. 一句话定义

**gradient bucket 与通信重叠** 是 [[Megatron-LM]]（与 [[Distributed Data Parallel|DDP]] 共有）的**梯度通信与反向计算 overlap 机制**——反向时每算完一层就把该层 param 的 grad 写入 **bucket（梯度桶，预分配的连续 buffer）**，bucket 攒满（达阈值）就触发 **NCCL async** 集合通信（`all-reduce` 或 `reduce-scatter`，非阻塞，立即返回），让该通信与**下一层 backward 计算 overlap**（通信藏到计算后）；如此层间流水，整个 backward 过程的 grad 通信几乎全被计算掩盖，只在 backward 末尾 `wait` 确保所有 async 通信完成。**bucket 大小** 是 trade-off：大 bucket 攒慢、通信次数少（latency $\alpha$ 项小）但 overlap 机会少（攒满才发，前期计算无通信藏）；小 bucket 攒快、overlap 多但通信次数多（$\alpha$ 大）。**reducer** 是管理多 bucket + 调度 async 通信的引擎（DDP 的 `torch.nn.parallel.reducer`、Megatron 的 `GradBuffer`）。是 [[Megatron distributed optimizer]]（reduce-scarter）与 DDP（all-reduce）的底层 overlap 实现，是 [[compute vs communication bottleneck]] 优化的核心。

> [!note] 三句话定位
> - **是什么**：反向把 grad 攒 bucket，满触发 NCCL async 通信，与下一层 backward overlap，backward 末 wait 全部完成。
> - **为什么**：grad 通信（all-reduce/reduce-scatter）若同步阻塞，反向每层等通信，串行慢；bucket + async 藏通信到计算后，wall-clock $\approx$ 计算。
> - **与 [[Megatron distributed optimizer]] 关系**：D-optimizer 的 flat grad buffer 是 bucket 的载体，reduce-scatter async 是通信，overlap 是时序。bucket 是结构，async 是机制，overlap 是效果。


## 2. 为什么需要它（动机与背景）

### 2.1 grad 通信是反向瓶颈

LLM 训练反向的 grad all-reduce（DDP）或 reduce-scatter（D-optimizer）通信 $\propto N$（model size，每 param 一个 grad）。70B 模型 grad 通信几 GB，串行阻塞反向每层。是 wall-clock 通信大头。

### 2.2 同步阻塞的反向

无 overlap 的反向：层 L backward 算 grad → 同步 all-reduce → 等通信完 → 层 L-1 backward。每层等通信，串行。反向 wall-clock $\approx$ 计算 + 通信（通信不藏）。

### 2.3 bucket + async overlap

bucket 攒 grad，满触发 async 通信（不阻塞），与下一层 backward overlap。层 L backward → 触发 L 的 async 通信 → 层 L-1 backward（计算） || L 通信（藏）→ ... → 末尾 wait。通信藏到计算后，wall-clock $\approx$ 计算（若通信 < 计算）。

### 2.4 NCCL async 的支持

NCCL 的 `all_reduce`/`reduce_scatter` 支持 `async_op=True`（非阻塞，返回 work handle，后续 `wait`）。是 overlap 的底层。bucket 利用 async 实现藏通信。

### 2.5 bucket 大小的 trade-off

大 bucket：攒满慢（多 grad 合），通信次数少（$\alpha$ latency 小），但前期计算无通信藏（攒满才发，前期 backward 无 overlap）。小 bucket：攒满快，overlap 多，但通信次数多（$\alpha$ 大）。需调 bucket size 平衡 $\alpha$（次数）与 overlap 机会。典型 bucket = 几十 MB - 几百 MB。

### 2.6 与 DDP/D-optimizer 的共用

DDP（all-reduce）与 D-optimizer（reduce-scatter）都用 bucket + async overlap。机制同，通信原语不同。是分布式训练的通用 overlap 机制。


## 3. 核心概念详解

### 3.1 bucket 的结构

```python
# 简化: bucket 是预分配的连续 buffer
class Bucket:
    def __init__(self, params, size):
        self.grad_buffer = torch.zeros(size)  # flat, 预分配
        self.params = params  # 该 bucket 管的 param
        self.work = None  # async work handle
    def add_grad(self, param):
        # 把 param.grad 写入 flat buffer 的 offset
        self.grad_buffer[offset:offset+param.numel()] = param.grad.flat
    def maybe_trigger(self, threshold):
        if self.grad_buffer_filled >= threshold:
            self.work = all_reduce(self.grad_buffer, async_op=True)  # async, 不阻塞
    def wait(self):
        if self.work: self.work.wait()  # 确保通信完
```

bucket 是 flat buffer + param 列表 + async work handle。

### 3.2 反向的 overlap 时序

```
反向 (无 overlap, 同步):
  L backward | all-reduce L | L-1 backward | all-reduce L-1 | ...
  每层等通信, 串行

反向 (bucket + async overlap):
  L backward -> grad 写 bucket L -> 满 -> trigger async all-reduce L
  L-1 backward (计算) || all-reduce L (通信, 藏)
  -> grad 写 bucket L-1 -> 满 -> trigger async all-reduce L-1
  L-2 backward || all-reduce L-1
  ... -> 末尾 wait (确保全完成)
```

通信与下一层 backward overlap，wall-clock $\approx$ 计算（若通信 < 计算）。

### 3.3 reducer 的调度

```python
class Reducer:
    def __init__(self, params, bucket_size):
        self.buckets = split_into_buckets(params, bucket_size)
    def backward_hook(self, param):
        # autograd hook: param.grad 算完时触发
        bucket = find_bucket(param)
        bucket.add_grad(param)
        bucket.maybe_trigger(threshold)  # 满触发 async
    def finalize(self):
        for b in self.buckets: b.wait()  # 末尾 wait 全部
```

reducer 用 autograd hook（[[hooks机制]] 的 grad hook）在 grad 算完时触发 bucket 写入 + maybe trigger。末尾 finalize wait。

### 3.4 bucket size 的 trade-off

| bucket size | 通信次数 | latency $\alpha$ | overlap 机会 | 效果 |
|---|---|---|---|---|
| 大（如 1GB） | 少 | 小 | 少（攒满慢，前期无 overlap） | 通信集中，前期串行 |
| 小（如 10MB） | 多 | 大 | 多（攒满快，频繁 overlap） | 通信散，$\alpha$ 主导 |
| 中（如 100MB） | 中 | 中 | 中 | 平衡（典型） |

最优 bucket size 据网络 $\alpha/\beta$ 与计算量平衡。高带宽（NVLink）可小 bucket（$\alpha$ 小，overlap 多），低带宽（IB）大 bucket（$\beta$ 主导，减次数）。见 [[alpha-beta性能模型]]。

### 3.5 NCCL async 的机制

```python
# NCCL async: all_reduce 返回 work, 不阻塞
work = torch.distributed.all_reduce(tensor, async_op=True)  # 立即返回
# ... 这里可做其他计算 (overlap) ...
work.wait()  # 等通信完 (确保后续用结果)
```

NCCL 用 stream 分离（通信 stream 与计算 stream），async 时通信与计算并行（[[CUDA stream与event]]）。是 overlap 的底层。

### 3.6 与 DDP 的 all-reduce

DDP 的 reducer：bucket 攒 grad，满触发 `all_reduce` async，与 backward overlap。是 DDP 的 overlap 机制（PyTorch `torch.nn.parallel.DistributedDataParallel` 内置）。

### 3.7 与 D-optimizer 的 reduce-scatter

D-optimizer 的 flat grad buffer：bucket 攒 grad，满触发 `reduce_scatter` async（每 rank 只收 $1/D$ 片），与 backward overlap。是 ZeRO-2 的 overlap。机制同 DDP，原语不同（reduce-scatter vs all-reduce）。

### 3.8 末尾的 wait

backward 末尾，所有 bucket 的 async work 需 `wait` 确保通信完（否则 optimizer step 用未合的 grad 错）。是 overlap 的同步点。通常 wait 时间短（大部分通信已 overlap 完）。

### 3.9 与 TP all-reduce 的区别

TP 的 all-reduce（[[RowParallelLinear]] 后）是**同层依赖**（下一层 col 需完整 input），难 overlap（串行）。DP 的 grad all-reduce（bucket）是**跨层独立**（不同层的 grad 独立），能 overlap。两者 overlap 性质不同。

### 3.10 grad hook 的注册

```python
# 注册 grad hook: param.grad 算完时触发
for param in model.parameters():
    if param.requires_grad:
        param.register_post_accumulate_grad_hook(reducer.backward_hook)
# backward 时 autograd engine 算完 param.grad 后调 hook
```

用 PyTorch 的 `register_post_accumulate_grad_hook`（[[hooks机制]]）。是 bucket 触发的入口。


## 4. 数学原理 / 公式

### 4.1 通信量

每 param 一个 grad，总 $\propto N$（model size）。bucket 攒满触发一次 all-reduce/reduce-scatter。$B$ 个 bucket 则 $B$ 次通信，每次 $\propto N/B$。总 $\propto N$（不变，bucket 只改次数不改总量）。

### 4.2 latency 与 bandwidth

通信时间 $T = \alpha \cdot B + \beta \cdot N$（$B$ 次通信，$\alpha$ latency/次，$\beta$ bandwidth 倒数）。大 bucket（$B$ 小）：$\alpha B$ 小，但 overlap 少。小 bucket（$B$ 大）：$\alpha B$ 大，但 overlap 多。最优 $B$ 平衡。见 [[alpha-beta性能模型]]。

### 4.3 overlap 的收益

若通信时间 $T_c$ < 计算时间 $T_k$（per layer），overlap 后 wall-clock $\approx T_k$（通信藏）。否则 $\approx T_k + T_c$（部分串行）。大模型 $T_c \approx T_k$（通信与计算相当），overlap 关键。见 [[compute vs communication bottleneck]]。

### 4.4 bucket 攒满的时序

bucket 攒满需 $k$ 个 param 的 grad（$k$ 由 bucket size 与 param size 定）。攒满时间 $\approx k$ 层 backward 时间。触发后 async 通信 $\approx T_c$。若 $k$ 层 backward $> T_c$，通信全藏。需 bucket size 与层数匹配。

### 4.5 async 的 stream 分离

NCCL async 用通信 stream（与计算 stream 分离），GPU 并行执行（通信 SM 与计算 SM 独立，或 time-slice）。是硬件级 overlap。见 [[CUDA stream与event]]。


## 5. 代码示例（可选）

### 5.1 DDP 的 bucket overlap

```python
import torch.nn.parallel.DistributedDataParallel as DDP
model = DDP(model, device_ids=[0], bucket_cap_mb=100)  # bucket 100MB
# DDP 内置 reducer: grad hook -> bucket -> async all-reduce -> overlap
loss = model(x).sum()
loss.backward()  # 反向: grad 攒 bucket, async all-reduce overlap
# 末尾 DDP 自动 wait
optimizer.step()
```

### 5.2 Megatron D-optimizer 的 bucket

```python
from megatron.core.optimizer import DistributedOptimizer
opt = DistributedOptimizer(Adam(lr=1e-4), model, bucket_size=100*MB)
# 内部: flat grad buffer (bucket) -> reduce-scatter async -> overlap
loss = forward_backward(model, x)
loss.backward()  # grad 攒 buffer, async reduce-scatter overlap
opt.step()  # 末尾 wait + 1/D step + all-gather
```

### 5.3 手写 bucket + async

```python
bucket = torch.zeros(100*MB)  # 预分配
work = None
def grad_hook(param):
    bucket[offset:offset+param.numel()] = param.grad.flat
    if bucket_filled >= threshold:
        work = torch.distributed.all_reduce(bucket, async_op=True)  # async
# backward 后
if work: work.wait()  # 确保通信完
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Megatron distributed optimizer]]（载体）、[[Distributed Data Parallel|DDP]]（DDP 的 reducer）、[[all-reduce]]/[[reduce-scatter]]（通信原语）、[[NCCL通信拓扑]]、[[overlap strategy]]（overlap 概念）、[[alpha-beta性能模型]]、[[compute vs communication bottleneck]]、[[CUDA stream与event]]（stream 分离）、[[hooks机制]]（grad hook）、[[Autograd Engine]]（反向触发）。
- **下游（应用）**: [[Megatron-LM]] 的优化器、DDP/FSDP 的 grad sync、[[3D parallelism]]（DP 维的 overlap）、[[训练显存估计]]（grad buffer 显存）、[[overlap strategy]] 的具体实现。
- **对比 / 易混**:
  - **bucket + async vs 同步 all-reduce**：同步每层等通信（串行慢）；bucket + async 藏通信到计算后（overlap 快）。
  - **bucket（DP grad）vs TP all-reduce**：DP 的 grad all-reduce 跨层独立（能 overlap）；TP 的 all-reduce 同层依赖（难 overlap）。两者性质不同。
  - **bucket size vs [[Megatron distributed optimizer|D-optimizer]] 的 flat grad buffer**：bucket 是攒 grad 的单元（可多 bucket），flat grad buffer 是 D-optimizer 的整体 flat 化（一个大 bucket 切 D 片）。bucket 是通用机制，flat grad buffer 是 D-optimizer 的具体实现。
  - **grad hook vs forward hook**：grad hook 在反向 grad 算完触发（[[hooks机制]] 的 post_accumulate_grad_hook），forward hook 在前向触发。bucket 用 grad hook。


## 7. 常见误区与易错点

> [!warning] 误区 1：bucket 减少通信量
> bucket 只改通信**次数**（攒满发，少次大消息），不改**总量**（仍 $\propto N$）。减的是 latency $\alpha$（次数少），不是 bandwidth $\beta$（总量不变）。误以为减总量会错判。

> [!warning] 误区 2：bucket 越大越好
> 大 bucket 通信次数少（$\alpha$ 小），但 overlap 机会少（攒满慢，前期计算无通信藏）。小 bucket overlap 多但 $\alpha$ 大。需平衡。误以为越大越好会前期串行。

> [!warning] 误区 3：async 通信完全免费
> async 通信与计算 overlap，但占用通信带宽与 SM（stream 分离）。若通信 > 计算，部分串行。且末尾 wait（同步点）。非完全免费，是"藏"。

> [!warning] 误区 4：bucket 与 TP all-reduce 一样 overlap
> DP 的 grad all-reduce 跨层独立（能 overlap）；TP 的 all-reduce 同层依赖（下一层需完整 input，难 overlap）。bucket 是 DP 的，TP 不用 bucket（用同步 all-reduce）。

> [!warning] 误区 5：grad hook 在 backward 前触发
> grad hook 在 **grad 算完时**触发（[[Autograd Engine]] 算完该 param 的 grad 后）。不是 backward 前。是反向过程中逐 param 触发。

> [!warning] 误区 6：忘记末尾 wait
> async 通信需末尾 `wait` 确保完（否则 optimizer step 用未合 grad 错）。是 overlap 的同步点。漏 wait 会 grad 不全。

> [!warning] 误区 7：bucket 显存忽略
> bucket 是预分配 buffer（$\propto$ bucket size，如 100MB）。占显存。大 bucket 占多。需计入 [[训练显存估计]]。


## 8. 延伸细节

### 8.1 DDP 的 bucket_cap_mb

PyTorch DDP 的 `bucket_cap_mb` 参数（默认 25MB）控 bucket size。调大减通信次数（$\alpha$ 小），调小增 overlap。是 DDP 性能调参点。

### 8.2 NCCL async 与 stream

NCCL async 用独立 stream（与计算 stream 分离），GPU 并行。需 CUDA stream 同步（event）确保数据依赖。见 [[CUDA stream与event]] 与 [[overlap strategy]]。

### 8.3 与 [[overlap strategy]] 的关系

bucket + async 是 [[overlap strategy]] 在 grad sync 场景的具体实现。overlap strategy 是通用概念（通信藏计算），bucket 是 grad 维度的机制。

### 8.4 内容来源

gradient bucket 与通信重叠整理自 PyTorch DDP 文档（`reducer`）、Megatron-LM `grad_buffer.py` 源码、NCCL async 文档。bucket size trade-off 与 $\alpha/\beta$ 分析见 [[alpha-beta性能模型]]。与 D-optimizer 的关系见 [[Megatron distributed optimizer]]。

---
相关: [[分布式优化器]] | [[Megatron distributed optimizer]] | [[Distributed Data Parallel]] | [[all-reduce]] | [[reduce-scatter]] | [[NCCL通信拓扑]] | [[overlap strategy]] | [[alpha-beta性能模型]] | [[compute vs communication bottleneck]] | [[CUDA stream与event]] | [[hooks机制]] | [[Autograd Engine]] | [[3D parallelism]] | [[训练显存估计]] | [[ZeRO (DeepSpeed)]] | [[FSDP2]]
