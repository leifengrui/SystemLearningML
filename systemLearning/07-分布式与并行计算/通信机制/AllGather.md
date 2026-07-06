# AllGather

> **所属章节**: [[通信机制]]
> **所属模块**: [[07-分布式与并行计算]]
> **难度**: 中（需懂集合通信 + ring 拓扑）

## 1. 一句话定义

**AllGather（全收集）** 是集合通信原语：**每 rank 持有一片数据 $V_i$，经收集后所有 rank 得到完整的拼接全量 $[V_0, V_1, ..., V_{N-1}]$**。它与 [[all-reduce]]（归约）不同——AllGather 保留各片原值拼接，all-reduce 归约（sum）成单值。用途：[[weight sync mechanism]] 中 PT 向 PD/worker 广播全 $\theta$（push 模式）、[[Tensor Parallel|ColumnParallel]] 拼接输出、[[Fully Sharded Data Parallel|FSDP/ZeRO-3]] 算时 AllGather 权重分片拼全。ring AllGather 每卡通信量 $(N-1)/N\cdot V$（带宽最优）。[[NCCL通信拓扑]] 底层实现。它是 all-reduce 的"半边"——**all-reduce = reduce-scatter + AllGather**。

> [!note] AllGather 的语义
> - 输入：每 rank $i$ 有 $V_i$（片）；
> - 输出：所有 rank 得 $[V_0, V_1, ..., V_{N-1}]$（全量拼接）；
> - 不同于 [[all-reduce]]（归约成单值）、[[reduce-scatter]]（归约分片）。

## 2. 为什么需要它（动机与背景）

多场景需"收集各片到全 rank"：
1. **weight sync**：[[weight sync mechanism]] 中 PT/learner 把 $\theta$ 分片在多 rank，AllGather 拼全广播给 worker/PD（[[policy deployment (PD)]]）；
2. **ZeRO-3**：权重存时分片（省显存），算时 AllGather 拼全（前向/反向用全权重）；
3. **ColumnParallel**：[[Tensor Parallel]] 列切分，各卡算部分输出维，AllGather 拼完整输出；
4. **数据并行某些变种**：数据分片存，AllGather 拼全 batch；
5. **检查点同步**：多 rank 拼完整 checkpoint。

AllGather 是"分片存、用时拼"模式的核心，与 [[reduce-scatter]]（"归约分片"）互补，二者组合等价 all-reduce。

## 3. 核心概念详解

### 3.1 语义

$N$ 个 rank，各持片 $V_i$：
- **AllGather**：所有 rank 得 $[V_0, V_1, ..., V_{N-1}]$；
- 与 all-reduce 区别：AllGather 保留各片原值（拼接），all-reduce 归约（sum）成单值 $R$。

### 3.2 ring AllGather（带宽最优）

$N$ rank 排环，片分 $N$ 块：
1. 每 rank 发自己片给下一 rank，下一 rank 收并持有；
2. 重复 $N-1$ 步，每步转发累积的片；
3. $N-1$ 步后所有 rank 有所有片；
- **每卡通信量**：$\frac{N-1}{N}\cdot V$（$V$ 总全量，每卡发 $(N-1)/N V$）；
- **延迟**：$O(N)$（$N-1$ 步）。

### 3.3 与 reduce-scatter 的关系

- **reduce-scatter**：各 rank 持全 $V_i$，归约 + 分散，各 rank 得归约片 $R_i$；
- **AllGather**：各 rank 持片 $V_i$，收集，各 rank 得全量 $[V_i]$；
- **组合**：reduce-scatter（得归约片）+ AllGather（拼全）= all-reduce；
- 这是 ZeRO 通信分解的基础。

### 3.4 用途详解

| 场景 | 用法 |
|---|---|
| **weight sync** | PT/learner 各 rank 持 $\theta$ 片，AllGather 拼全广播给 worker/PD |
| **ZeRO-3** | 权重存分片，算时 AllGather 拼全（前向用） |
| **ColumnParallel** | 各卡算部分输出维，AllGather 拼完整输出 |
| **数据并行分片** | 数据分片存，AllGather 拼全 batch |
| **checkpoint 拼全** | 多 rank 持 checkpoint 片，AllGather 拼完整 |

### 3.5 通信量与延迟

- **通信量**（每卡）：$\frac{N-1}{N}\cdot V\approx V$（$N$ 大），带宽最优；
- **延迟**：$O(N)$（ring），$O(\log N)$（tree 变种）；
- **vs all-reduce**：AllGather 是 all-reduce 的一半（只 all-gather 阶段，无 scatter-reduce），通信量半。

## 4. 数学原理 / 公式

### 4.1 通信量

$$
\text{comm per rank}=\frac{N-1}{N}\cdot V\approx V\quad(N\text{ 大})
$$

- $V$：全量数据大小；
- 每卡发 $(N-1)/N V$，收 $(N-1)/N V$；
- 带宽最优（每卡 $\sim V$，均衡）。

### 4.2 vs all-reduce

$$
\text{AllGather}: \frac{N-1}{N}V;\quad \text{all-reduce}: 2\frac{N-1}{N}V
$$

- AllGather 是 all-reduce 通信量的一半（无 scatter-reduce 阶段）。

### 4.3 all-reduce = reduce-scatter + AllGather

$$
\text{all-reduce}(V_i)=\text{AllGather}(\text{reduce-scatter}(V_i))
$$

- reduce-scatter：各 rank 得归约片 $R_i$（通信 $\frac{N-1}{N}V$）；
- AllGather：拼全 $[R_i]$（通信 $\frac{N-1}{N}V$）；
- 总 $2\frac{N-1}{N}V$，等价 all-reduce。

### 4.4 ZeRO-3 的通信

- 存时：权重分片，每卡 $P/N$（省显存）；
- 算时：AllGather 拼全权重（通信 $V$），前向/反向；
- 通信量：$\sim V$ per layer（与 all-reduce 同量级，但是权重非梯度）；
- 详见 [[Fully Sharded Data Parallel]]。

## 5. 代码示例

```python
import torch

def ring_all_gather(simulated_ranks):
    """ring AllGather 概念模拟(N rank,各 V_i 片)"""
    N = len(simulated_ranks)
    gathered = [[r.clone()] for r in simulated_ranks]  # 各 rank 持自己片
    for step in range(N - 1):
        new_gathered = [g[:] for g in gathered]
        for i in range(N):
            send_rank = i
            recv_rank = (i + 1) % N
            # rank i 的最后收集片传给 rank recv
            new_gathered[recv_rank].append(gathered[send_rank][-1].clone())
        gathered = new_gathered
    return [torch.cat(g, 0) for g in gathered]

N, V_chunk = 4, 4
ranks = [torch.randn(V_chunk) for _ in range(N)]  # 各 rank 持一片
expected = torch.cat(ranks, 0)  # 全量拼接
result = ring_all_gather(ranks)
print("ring AllGather 结果(各 rank 应相同拼接):")
for i, r in enumerate(result):
    print(f"  rank {i}: {r.tolist()}, 一致={torch.allclose(r, expected)}")
print(f"每卡通信量: (N-1)/N*V = {(N-1)/N*(N*V_chunk):.1f} 元素(全量 V={N*V_chunk})")

# ===== 实际 PyTorch 用法 =====
print("\n实际 PyTorch 用法:")
print("  tensors = [torch.zeros(V) for _ in range(N)]  # 输出列表")
print("  dist.all_gather(tensors, local_tensor)  # 各 rank 的 local 拼全")
print("  # 或 NCCL 高级 API: dist.all_gather_into_tensor(单 tensor 输出)")

# ===== 用途示例 =====
print("\n用途:")
print("  weight sync: PT θ 分片各 rank -> AllGather 拼全 -> 广播 worker/PD")
print("  ZeRO-3: 权重存分片 -> 算时 AllGather 拼全前向")
print("  ColumnParallel: 各卡部分输出 -> AllGather 拼完整")
print("  all-reduce = reduce-scatter + AllGather(ZeRO 分解)")
```

> [!tip] 实际工程
> - **PyTorch**：`dist.all_gather(tensors, local)` 或 `dist.all_gather_into_tensor`（单 tensor，更高效）；
> - **NCCL**：底层 ring/tree 自动；
> - **ZeRO-3**：`--zero-stage=3`，算时 AllGather 权重；
> - **weight sync**：verl 用 NCCL AllGather broadcast $\theta$。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[all-reduce]]（组合关系）、[[reduce-scatter]]（互补）、[[NCCL通信拓扑]]、[[rank与world size]]、[[process group]]、[[torch.distributed]]。
- **下游（应用）**: [[weight sync mechanism]]（PT→PD/worker 广播 $\theta$）、[[Fully Sharded Data Parallel|FSDP/ZeRO-3]]（算时拼权重）、[[Tensor Parallel|ColumnParallel]]（拼输出）、[[policy deployment (PD)]]/[[rollout worker]]（接收 $\theta$）。
- **对比 / 易混**:
  - **AllGather vs [[all-reduce]]**：前者拼接保留各片，后者归约成单值。AllGather 通信量半。
  - **AllGather vs [[reduce-scatter]]**：前者片→全量，后者全量→归约片。方向反，互补。
  - **AllGather vs broadcast**：前者各 rank 都持片并收集（多源），后者一 rank 广播（单源）。AllGather 是"多源 broadcast"。
  - **AllGather vs all-to-all**：前者各 rank 持一片拼全，后者各 rank 给每 rank 不同片（全交换）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"AllGather = broadcast"**：错。broadcast 单源广播，AllGather 多源收集。通信量不同（broadcast $V$ from 1，AllGather 各 rank $\sim V$）。
> 2. **"AllGather 归约"**：错。AllGather 拼接保留各片原值，不归约（归约是 all-reduce）。
> 3. **"AllGather 通信量比 all-reduce 多"**：错。AllGather 是 all-reduce 的一半（无 scatter-reduce）。
> 4. **"AllGather 输出不同"**：错。输出相同拼接全量（所有 rank 一致）。
> 5. **"ZeRO-3 AllGather 权重免费"**：错。每层算时 AllGather 权重，通信量 $\sim V$（与 all-reduce 同量级，是 ZeRO-3 的开销）。
> 6. **"weight sync 用 all-reduce"**：错。weight sync 是推 $\theta$（拼全广播），用 AllGather/broadcast，非归约。
> 7. **"AllGather 不延迟"**：错。ring 延迟 $O(N)$，大 $N$ 显。

## 8. 延伸细节

### 8.1 AllGather 在 weight sync

[[weight sync mechanism]] 的 push 模式：
- PT/learner 各 rank 持 $\theta$ 片（TP/PP 切分）；
- AllGather 拼全 $\theta$；
- 广播给 worker/PD（或 worker pull）；
- 通信量 $\sim|\theta|$（全模型，大）；
- 增量/压缩减带宽（差分+量化）。

### 8.2 AllGather 在 ZeRO-3

- 存时：权重分片 $P/N$（省显存）；
- 算时：每层 AllGather 权重拼全（通信 $V_{\text{layer}}$），前向；
- 反向：AllGather 权重 + reduce-scatter 梯度；
- 通信量：$\sim2V$（与 all-reduce 同量级），但 ZeRO-3 省显存换通信；
- 详见 [[Fully Sharded Data Parallel]]。

### 8.3 AllGather 在 ColumnParallel

- [[Tensor Parallel]] 列切：各卡算 $XW_i$（部分输出维）；
- AllGather 拼完整 $[XW_0, ..., XW_{N-1}]$；
- 通信量 $Bd$（激活大小，每层）；
- Megatron 组合 RowParallel 用 all-reduce 替代（省通信）。

### 8.4 AllGather 的优化

- **ring/tree**：大消息 ring，小消息 tree；
- **多 ring 并行**：NCCL 多 ring 用多链路；
- **overlap**：与计算重叠（[[overlap strategy]]）；
- **bucket**：分桶 AllGather，小桶先发，overlap 反向。

### 8.5 all_gather_into_tensor

PyTorch 新 API（`dist.all_gather_into_tensor`）：
- 输出单 tensor（而非 tensor 列表）；
- 连续内存，更高效；
- NCCL 底层优化；
- 推荐用（vs 旧 `all_gather` 列表）。

### 8.6 AllGather 与 all-to-all 的区别

- **AllGather**：各 rank 持一片，拼全（所有 rank 得相同全量）；
- **all-to-all**：各 rank 给每 rank 不同片（全交换，每 rank 得不同片拼全）；
- all-to-all 用于 MoE 专家路由、注意力序列并行等；
- AllGather 通信量 $\sim V$，all-to-all $\sim V$（但模式不同）。

---
相关: [[通信机制]]、[[all-reduce]]、[[reduce-scatter]]、[[NCCL通信拓扑]]、[[rank与world size]]、[[process group]]、[[torch.distributed]]、[[weight sync mechanism]]、[[Fully Sharded Data Parallel]]、[[Tensor Parallel]]、[[policy deployment (PD)]]、[[rollout worker]]、[[overlap strategy]]
