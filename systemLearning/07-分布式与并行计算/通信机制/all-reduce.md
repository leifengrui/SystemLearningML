# all-reduce

> **所属章节**: [[通信机制]]
> **所属模块**: [[07-分布式与并行计算]]
> **难度**: 中（需懂集合通信 + ring 拓扑）

## 1. 一句话定义

**all-reduce（全归约）** 是集合通信原语：**所有 rank 各持有一份数据，经归约（如 sum/avg/max）后，所有 rank 得到相同的归约结果**。它是 [[Data Parallel|DP]] 梯度聚合（每卡算局部梯度，all-reduce 求和得全梯度）与 [[Tensor Parallel|TP]] 激活聚合（RowParallel 求和）的核心原语。主流实现 **ring all-reduce**（带宽最优，每卡通信量 $(N-1)/N\cdot V$，延迟 $O(N)$）与 **tree all-reduce**（延迟优 $O(\log N)$）。[[NCCL通信拓扑]] 底层实现，[[Distributed Data Parallel|DDP]] 默认用它聚合梯度。与 [[AllGather]]（收集不同数据拼全）不同：all-reduce 输出相同归约值，AllGather 输出拼接全量。

> [!note] all-reduce 的语义
> - 输入：每 rank $i$ 有 $V_i$；
> - 输出：所有 rank 得 $R=\text{reduce}(V_0,...,V_{N-1})$（如 sum）；
> - 不同于 broadcast（一传多）、[[AllGather]]（拼全）、[[reduce-scatter]]（分片归约）。

## 2. 为什么需要它（动机与背景）

数据并行训练需聚合梯度：
1. **DP 梯度聚合**：每卡算不同数据的局部梯度，需 all-reduce 求和得全梯度，各卡更新一致；
2. **TP 激活聚合**：RowParallel 各卡算部分结果，all-reduce 求和得完整输出；
3. **同步训练**：所有 rank 需相同结果（参数同步更新），all-reduce 保证一致性；
4. **带宽优化**：ring all-reduce 每卡通信 $(N-1)/N\cdot V$（vs 朴素 reduce+broadcast 的 $2V$），带宽最优；
5. **集合通信基础**：是 DP/TP 的通信核心，[[NCCL通信拓扑]] 优化对象。

all-reduce 是分布式训练最常用的集合通信原语，理解它（尤其 ring）是理解 DP/TP 通信开销的基础。

## 3. 核心概念详解

### 3.1 语义

$N$ 个 rank，各有向量 $V_i$：
- **reduce**：$R=\text{op}(V_0, V_1, ..., V_{N-1})$（op = sum/avg/max/min）；
- **all-reduce**：所有 rank 都得 $R$（reduce + 广播结果到所有 rank）；
- 典型 op：sum（梯度聚合）、avg（平均梯度）。

### 3.2 朴素实现：reduce + broadcast

1. **reduce**：所有数据传到 rank 0，rank 0 算归约；
2. **broadcast**：rank 0 把结果广播给所有 rank；
- **问题**：rank 0 是瓶颈（收 $N\cdot V$，发 $N\cdot V$），带宽不均，延迟 $O(N)$；
- **通信量**：rank 0 收发 $2NV$，其他 rank $V$，不均衡。

### 3.3 ring all-reduce（带宽最优）

$N$ 个 rank 排成环，数据分 $N$ 块：
1. **scatter-reduce 阶段**（$N-1$ 步）：每 rank 把一块传给下一 rank，累加收到的块，$N-1$ 步后每 rank 有一个块的完整归约；
2. **all-gather 阶段**（$N-1$ 步）：每 rank 把完整块传给下一 rank，$N-1$ 步后所有 rank 有所有块的归约；
- **每卡通信量**：$2\cdot\frac{N-1}{N}\cdot V$（$V$ 总数据量，scatter+gather 各 $(N-1)/N\cdot V$）；
- **带宽最优**：每卡通信量 $\sim V$（与 $N$ 无关），均衡；
- **延迟**：$O(N)$（$2(N-1)$ 步串行）。

### 3.4 tree all-reduce（延迟最优）

$N$ 个 rank 排成树：
- **reduce 阶段**：叶子向根归约（$O(\log N)$ 步）；
- **broadcast 阶段**：根向叶子广播（$O(\log N)$ 步）；
- **延迟**：$O(\log N)$（优，适合小消息）；
- **问题**：根节点带宽瓶颈（非均衡），大消息不如 ring。

### 3.5 NCCL 的混合实现

[[NCCL通信拓扑|NCCL]] 实际用 **ring + tree 混合**：
- **小消息**：tree（延迟优）；
- **大消息**：ring（带宽优）；
- **多 ring 并行**：用多环拓扑充分用带宽；
- 自动选最优算法。

### 3.6 与其他原语的关系

| 原语 | 语义 | 输出 |
|---|---|---|
| **all-reduce** | 归约到所有 | 各 rank 相同归约值 |
| **[[AllGather]]** | 收集拼全 | 各 rank 相同拼接全量 |
| **[[reduce-scatter]]** | 归约 + 分散 | 各 rank 不同归约片 |
| **broadcast** | 一传多 | 各 rank 相同源数据 |
| **reduce** | 归约到根 | 仅 rank 0 有归约值 |

- **all-reduce = reduce-scatter + all-gather**（分解）：reduce-scatter 得归约片，all-gather 拼全。这是 ZeRO-2/3 的通信分解基础。

## 4. 数学原理 / 公式

### 4.1 ring all-reduce 通信量

$$
\text{comm per rank}=2\cdot\frac{N-1}{N}\cdot V\approx2V\quad(N\text{ 大})
$$

- $V$：总数据量（如梯度大小）；
- 每卡发送 $\frac{N-1}{N}V$，接收 $\frac{N-1}{N}V$（scatter+gather 各一半）；
- 带宽最优：每卡通信量与 $N$ 无关（$\sim V$），均衡。

### 4.2 vs 朴素 reduce+broadcast

$$
\text{朴素 rank 0}: 2NV,\quad \text{其他}: V;\quad \text{ring}: 2V\text{（每卡均衡）}
$$

- 朴素 rank 0 瓶颈 $2NV$，ring 每卡 $2V$，均衡且省 $N$ 倍。

### 4.3 延迟

- **ring**：$2(N-1)$ 步串行，延迟 $O(N)$；
- **tree**：$2\log N$ 步，延迟 $O(\log N)$；
- **ring 适合大消息**（带宽瓶颈，延迟可容忍）；
- **tree 适合小消息**（延迟瓶颈）。

### 4.4 all-reduce = reduce-scatter + all-gather

$$
\text{all-reduce}(V_i)=\text{all-gather}(\text{reduce-scatter}(V_i))
$$

- reduce-scatter：各 rank 得归约的一片 $V/N$；
- all-gather：拼全 $V$；
- 通信量同 $2(N-1)/N\cdot V$（与 ring 等价）。

## 5. 代码示例

```python
import torch, torch.distributed as dist
# 概念模拟(单进程模拟 N rank)

def ring_all_reduce(simulated_ranks):
    """ring all-reduce 概念模拟(N rank,各 V 向量)"""
    N = len(simulated_ranks)
    V = simulated_ranks[0].shape[0]
    chunk = V // N
    # scatter-reduce 阶段
    blocks = [r.clone().chunk(N, 0) for r in simulated_ranks]  # 各 rank 数据分 N 块
    for step in range(N - 1):
        new_blocks = [b.clone() for b in blocks]
        for i in range(N):
            send = (i + step) % N
            recv = (i + step + 1) % N
            # rank i 的 block[send] 传给 rank recv,累加
            new_blocks[recv][send] = blocks[recv][send] + blocks[i][send]
        blocks = new_blocks
    # all-gather 阶段
    for step in range(N - 1):
        new_blocks = [b.clone() for b in blocks]
        for i in range(N):
            send = (i + step - (N-1)) % N
            recv = (i + step + 1 - (N-1)) % N
            new_blocks[recv][send] = blocks[i][send]
        blocks = new_blocks
    return [torch.cat(b, 0) for b in blocks]

N, V = 4, 8
ranks = [torch.randn(V) for _ in range(N)]
expected = sum(ranks)  # 全归约(sum)
result = ring_all_reduce(ranks)
print("ring all-reduce 结果(各 rank 应相同):")
for i, r in enumerate(result):
    print(f"  rank {i}: {r[:4].tolist()}... 一致={torch.allclose(r, expected)}")
print(f"每卡通信量: 2*(N-1)/N*V = {2*(N-1)/N*V:.1f} 元素(vs 朴素 rank0 {2*N*V})")

# ===== 实际 torch.distributed 用法(概念) =====
print("\n实际 PyTorch 用法:")
print("  tensor = torch.randn(1000)  # 各 rank 局部梯度")
print("  dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # 原地 all-reduce")
print("  # tensor 现为全 rank 梯度和(NCCL 自动 ring/tree)")

# ===== 用途 =====
print("\n用途:")
print("  DP 梯度聚合: 各卡算局部梯度 -> all-reduce sum -> 全梯度")
print("  TP 激活聚合: RowParallel 各卡部分结果 -> all-reduce sum -> 完整输出")
```

> [!tip] 实际工程
> - **PyTorch**：`dist.all_reduce(tensor, op=ReduceOp.SUM)`；
> - **NCCL**：底层 ring/tree 自动选，多 ring 并行；
> - **DP**：[[Distributed Data Parallel|DDP]] 反向后 all-reduce 梯度；
> - **TP**：Megatron RowParallel 输出 all-reduce。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[NCCL通信拓扑]]（底层实现）、[[rank与world size]]、[[process group]]、[[torch.distributed]]、集合通信理论。
- **下游（应用）**: [[Data Parallel]]（梯度聚合）、[[Tensor Parallel]]（激活聚合）、[[Distributed Data Parallel|DDP]]、[[Fully Sharded Data Parallel|FSDP/ZeRO]]（reduce-scatter+all-gather 分解）、[[3D parallelism]]。
- **对比 / 易混**:
  - **all-reduce vs [[AllGather]]**：前者归约（输出相同归约值），后者拼接（输出拼接全量）。DP 梯度用 all-reduce，weight sync/ColumnParallel 用 all-gather。
  - **all-reduce vs [[reduce-scatter]]**：前者输出全归约，后者输出归约片（分散）。all-reduce = reduce-scatter + all-gather。
  - **all-reduce vs broadcast**：前者所有 rank 都参与归约，后者一 rank 广播。
  - **ring vs tree**：ring 带宽优（大消息），tree 延迟优（小消息）。
  - **all-reduce vs reduce**：前者所有 rank 得结果，后者仅 rank 0。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"all-reduce = reduce + broadcast"**：语义等价，但朴素实现 rank 0 瓶颈，ring/tree 更优。别说"就是 reduce+broadcast"。
> 2. **"all-reduce 省带宽"**：错。每卡通信量 $\sim V$（vs 数据量 $V$），是均衡/带宽最优，非"省"。总带宽仍 $NV$。
> 3. **"ring 总优于 tree"**：错。ring 大消息优（带宽），tree 小消息优（延迟）。
> 4. **"all-reduce 不延迟"**：错。ring 延迟 $O(N)$，大 $N$ 时显。
> 5. **"all-reduce 输出不同"**：错。输出相同归约值（所有 rank 一致）。
> 6. **"all-reduce 等于 all-gather"**：错。前者归约（sum），后者拼接（保留所有数据）。
> 7. **"DP all-reduce 梯度是全模型"**：对（DP 每卡全模型，all-reduce 全梯度）。
> 8. **"all-reduce 死锁"**：错。集合通信需所有 rank 调用，若某 rank 不调则挂起（非死锁，是阻塞）。

## 8. 延伸细节

### 8.1 ring all-reduce 的两阶段详解

**阶段1 scatter-reduce**（$N-1$ 步）：
- 每 rank 把数据分 $N$ 块 $V_i=[b_{i,0},...,b_{i,N-1}]$；
- 每步 rank $i$ 发 $b_{i,(i+s)\%N}$ 给 rank $(i+1)\%N$，收后累加；
- $N-1$ 步后 rank $i$ 有 $b_{*,(i+N-1)\%N}$ 的完整归约；

**阶段2 all-gather**（$N-1$ 步）：
- 每步 rank $i$ 发完整块给下一 rank，下一 rank 复制；
- $N-1$ 步后所有 rank 有所有块的归约。

### 8.2 NCCL 的多 ring

- 单 ring 用一对收发链路；
- 多 ring（如 4 ring）并行用多链路，提带宽；
- NVLink 多通道支持多 ring；
- NCCL 自动拓扑发现，建最优 ring 数。

### 8.3 all-reduce 在 ZeRO 的分解

- **ZeRO-2**：reduce-scatter 梯度（各 rank 得一片归约梯度），省显存（只存片）；
- **ZeRO-3**：+ all-gather 权重（算时拼全），存时分片；
- 通信量同 all-reduce（分解为 reduce-scatter + all-gather）；
- 详见 [[Fully Sharded Data Parallel]]。

### 8.4 all-reduce 的 op

- **SUM**：梯度聚合（最常用）；
- **AVG**：平均梯度（等价 SUM 后除 $N$）；
- **MAX/MIN**：用于梯度裁剪同步、检查；
- **PRODUCT**：少见。

### 8.5 all-reduce 的延迟优化

- **tree/ring 混合**：小消息 tree，大消息 ring；
- **overlap**：计算与通信重叠（[[overlap strategy]]），减延迟感知；
- **梯度分桶**：分桶 all-reduce（小桶先发），overlap 通信与反向；
- 详见 [[overlap strategy]]/[[compute vs communication bottleneck]]。

### 8.6 all-reduce 的历史

- **MPI_Allreduce**：HPC 经典接口；
- **ring all-reduce**（Baidu 2017）：带宽最优算法，深度学习训练标准；
- **NCCL**：NVIDIA 实现，GPU 集合通信库；
- **Horovod**：基于 ring all-reduce 的分布式训练框架；
- all-reduce 是分布式训练的通信基石。

---
相关: [[通信机制]]、[[AllGather]]、[[reduce-scatter]]、[[NCCL通信拓扑]]、[[rank与world size]]、[[process group]]、[[torch.distributed]]、[[Data Parallel]]、[[Tensor Parallel]]、[[Distributed Data Parallel]]、[[Fully Sharded Data Parallel]]、[[3D parallelism]]、[[overlap strategy]]、[[compute vs communication bottleneck]]
