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
> [!note] 解答：all-reduce 不适合传"普通信号"——它是"大块张量归约"原语，不是"传消息"原语，乱用会又慢又同步死
> "普通信号"这里理解为**控制信号 / 小标量 / 状态标志 / 心跳 / 进度通知**这类**非训练张量的小消息**。结论是**不适合**，下面从"语义不对 + 性能浪费 + 同步过重"三层讲清，再给正确做法。
>
> ### 1. 语义就不对：all-reduce 是"归约"不是"传递"
> all-reduce 的定义是"所有 rank 各出一份 $V_i$，归约（sum/max/avg）后每 rank 拿到**相同的归约结果** $R$"。它**不保留任何单个 rank 的原始值**——你传进去的是"贡献"，出来的是"聚合"。而"传信号"通常要的是"**让某个 rank 的值原样被其他 rank 知道**"，这是 **[[broadcast]] / send-recv** 的语义，不是 all-reduce。
> - 例：rank 3 想告诉大家"我 rollout 跑完了"（一个 bool）。用 all-reduce(sum) → 所有 rank 得到"有几个人跑完了"（聚合数），**rank 3 那个 bool 本身被抹掉了**，别人不知道是不是 rank 3。要传"谁完了"得用 broadcast(src=3) 或 send-recv。
> - 只有当你要的是**聚合量**（总 loss、平均精度、已完成数）时，all-reduce 才语义对。
>
> ### 2. 性能浪费：all-reduce 对小消息有固定的 launch 开销，"大炮打蚊子"
> all-reduce 是为**大块张量**（MB~GB 级梯度）优化的，靠 ring/tree 把**带宽**吃满。但它有**固定的启动延迟 (launch latency)**，与数据量无关：
>
> | 消息大小 | all-reduce 合理性 | 实际延迟量级 | 瓶颈 |
> |---|---|---|---|
> | 标量/几字节（控制信号） | ❌ 极不划算 | ~10–100 μs（launch 主导） | kernel 启动 + 同步开销，远大于数据本身 |
> | KB 级（小 metadata） | ❌ 仍不划算 | ~几十 μs | launch 仍主导，带宽没发挥 |
> | MB 级（小梯度） | ⚠️ 勉强 | ~百 μs~ms | 带宽开始起作用 |
> | GB 级（大梯度） | ✅ 设计目标 | ms~秒 | 带宽主导，ring 最优 |
>
> 关键：**all-reduce 的延迟 = launch 开销 + 传输延迟**。launch 开销（NCCL kernel 启动 + 集合同步）对小消息是"地板价"，传 1 个 float 和传 1MB 在延迟上几乎没差。所以拿它传标量信号 = **每次都付固定几十 μs 的过路费，只为传 4 字节**。对比 [[broadcast]] 或 gloo send-recv 对小消息反而更轻。
>
> ### 3. 同步过重：all-reduce 是"集合通信"自带 barrier 语义
> all-reduce 要求**所有 rank 都到达调用点**才能进行——本质是个**隐式 barrier**。"传信号"却常是**点对点、异步、某 rank 先到先发**的：rank 3 跑完了想通知别人继续，结果 all-reduce 卡在等最慢的 rank 7 也到调用点——**信号没传成，反而被最慢者拖住**。控制流被训练同步绑架。这对在线/异步系统（[[asynchronous training]]、rollout 通知 learner）尤其致命。
>
> ### 4. 还有个底层坑：NCCL 是 GPU 张量库，CPU 信号根本不该走它
> [[NCCL通信拓扑|NCCL]] 只管 **GPU 上的 tensor**。如果你说的"普通信号"是 CPU 侧的（进程状态、调度消息、Python 对象），强行塞进 NCCL all-reduce 要先把信号搬到 GPU tensor、再搬回 CPU，**纯浪费 + 引入 GPU 占用**。CPU 侧控制信号该用 **gloo**（[[NCCL backend]] §gloo）/ **MPI** / **Ray actor 消息** / [[RPC]]。
>
> ### 5. 正确做法（什么信号用什么原语）
> | 信号类型 | 正确原语 | 理由 |
> |---|---|---|
> | 某 rank 的值要让所有人知道（状态/配置/进度） | **[[broadcast]](src=that)** | 语义=一传多，保原值 |
> | 两 rank 间点对点通知（完成/触发） | **send/recv**（gloo 或 NCCL point-to-point） | 异步、不绑全局 |
> | CPU 控制消息 / Python 对象 | **gloo send-recv / Ray actor / [[RPC]]** | NCCL 不处理 CPU 非张量 |
> | 标量指标聚合（总 loss / 平均精度 / 完成数） | **all-reduce(sum/avg)** ✅ | 语义=归约，这才是它本行 |
> | 进度同步（等所有人到齐） | **barrier** | 显式同步，别拿 all-reduce 兼任 |
> | 心跳/存活探测 | **Ray actor / HTTP/gRPC** | 应用层协议，别走集合通信 |
>
> ### 6. 性能影响量化小结
> - **误用代价**：每个信号多付 ~10–100 μs launch + 一次全局 barrier；高频信号（如每步通知）会**把异步流水平均拉成同步**，吞吐回到"等最慢者"。
> - **DDP 里 all-reduce 传 loss 是合理的**：因为 loss 本就是要聚合的标量指标（求平均看训练曲线），且只在 step 末尾调一次，频率低、语义对——这不算"普通信号"，算"指标归约"。
> - **判断口诀**：**"要原值用 broadcast，要归约用 all-reduce；CPU 信号走 gloo/Ray，小消息别上 ring。"**
>
> 详见 [[broadcast]]、[[NCCL backend]]（gloo vs NCCL）、[[send-recv]]（待展开）、[[asynchronous training]]、[[Ray基础]]。
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
