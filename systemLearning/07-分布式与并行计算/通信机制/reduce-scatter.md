# reduce-scatter

> **所属章节**: [[通信机制]]
> **所属模块**: [[07-分布式与并行计算]]
> **难度**: 中（需懂集合通信 + ring 拓扑）

## 1. 一句话定义

**reduce-scatter（归约分散）** 是集合通信原语：**每 rank 持有完整数据 $V_i$，经归约（sum）后按片分散，每 rank 得到归约结果的一片 $R_j$**（各 rank 片不同）。它是 [[all-reduce]] 的"前半"——all-reduce = reduce-scatter + [[AllGather]]。主要用途：[[Fully Sharded Data Parallel|ZeRO-2/3]] 梯度归约分片（各 rank 只存/更新自己负责的梯度片，省显存）。ring reduce-scatter 每卡通信量 $(N-1)/N\cdot V$（带宽最优）。[[NCCL通信拓扑]] 底层实现。与 [[AllGather]]（片→全量）方向相反：reduce-scatter 是全量→归约片。

> [!note] reduce-scatter 的语义
> - 输入：每 rank $i$ 有全量 $V_i=[v_{i,0},...,v_{i,N-1}]$；
> - 归约：$R_j=\sum_i v_{i,j}$（按片归约）；
> - 输出：rank $j$ 得 $R_j$（归约片，各 rank 不同）；
> - 不同于 [[all-reduce]]（全归约到所有）、[[AllGather]]（拼全到所有）。

## 2. 为什么需要它（动机与背景）

ZeRO 优化器需梯度归约分片：
1. **ZeRO-2 省显存**：DP 中每卡算全梯度，但只需存/更新自己负责的优化器状态片；reduce-scatter 把梯度归约 + 分片，每卡只持归约梯度片 $R_j$（vs 全梯度 $R$），省 $N$ 倍梯度显存；
2. **ZeRO-3 省更多**：权重也分片，reduce-scatter 梯度 + AllGather 权重，省权重+梯度+优化器显存；
3. **all-reduce 分解**：reduce-scatter + AllGather = all-reduce，通信量同但中间状态分片（省显存）；
4. **负载均衡**：每 rank 负责一片，归约+分散均衡（vs 朴素 reduce 到 rank 0 瓶颈）。

reduce-scatter 是"分片存、归约分散"模式的核心，ZeRO 用它把 DP 的显存从 $O(P)$ 降到 $O(P/N)$。

## 3. 核心概念详解

### 3.1 语义

$N$ rank，各持全量 $V_i=[v_{i,0},...,v_{i,N-1}]$（分 $N$ 片）：
- **归约**：$R_j=\text{op}_i(v_{i,j})$（如 sum）；
- **分散**：rank $j$ 得 $R_j$（归约片）；
- 各 rank 输出不同（片不同），非 all-reduce 的相同全量。

### 3.2 ring reduce-scatter（带宽最优）

$N$ rank 排环，数据分 $N$ 块：
1. 每 rank 发一块给下一 rank，累加收到的块；
2. $N-1$ 步后每 rank 有一个块的完整归约片 $R_j$；
3. 结束：每 rank 持归约片 $R_j$；
- **每卡通信量**：$\frac{N-1}{N}\cdot V$（发 $(N-1)/N V$）；
- **延迟**：$O(N)$（$N-1$ 步）；
- 这正是 all-reduce 的 scatter-reduce 阶段。

### 3.3 与 all-reduce / AllGather 的关系

- **all-reduce = reduce-scatter + AllGather**：
  1. reduce-scatter：各 rank 得归约片 $R_j$；
  2. AllGather：拼全 $[R_j]$ 到所有 rank；
  3. 总通信量 $2\frac{N-1}{N}V$（与 all-reduce 等价）；
- **ZeRO-2/3 的巧妙**：不做 AllGather（不拼全梯度），各 rank 只持归约片 $R_j$ + 优化器片，省显存；算时若需全权重再 AllGather（ZeRO-3）。

### 3.4 用途

| 场景 | 用法 |
|---|---|
| **ZeRO-2** | 梯度 reduce-scatter，各 rank 存归约梯度片 + 优化器片 |
| **ZeRO-3** | 梯度 reduce-scatter + 权重 AllGather（算时） |
| **all-reduce 分解** | reduce-scatter + AllGather = all-reduce |
| **分片更新** | 各 rank 只更新自己参数片（基于归约梯度片） |

### 3.5 通信量与延迟

- **通信量**（每卡）：$\frac{N-1}{N}\cdot V\approx V$（$N$ 大），带宽最优；
- **延迟**：$O(N)$（ring），$O(\log N)$（tree 变种）；
- **vs all-reduce**：reduce-scatter 是 all-reduce 的一半（无 all-gather 阶段）。

## 4. 数学原理 / 公式

### 4.1 通信量

$$
\text{comm per rank}=\frac{N-1}{N}\cdot V\approx V\quad(N\text{ 大})
$$

- $V$：全量数据；
- 每卡发 $(N-1)/N V$，收 $(N-1)/N V$；
- 带宽最优。

### 4.2 vs all-reduce / AllGather

$$
\text{reduce-scatter}: \frac{N-1}{N}V;\quad \text{AllGather}: \frac{N-1}{N}V;\quad \text{all-reduce}: 2\frac{N-1}{N}V
$$

三者各是 all-reduce 的一半，reduce-scatter + AllGather = all-reduce。

### 4.3 ZeRO-2 的显存

- 无 ZeRO：每卡存全梯度 $G$（$P$ 大小）+ 全优化器 $O$（$2P$ Adam）；
- ZeRO-1：优化器分片，每卡 $O/N$（梯度仍全 $G$，all-reduce 后）；
- ZeRO-2：梯度 reduce-scatter，每卡 $G/N$ + 优化器 $O/N$（省梯度+优化器）；
- ZeRO-3：+ 权重分片，每卡 $P/N$ + $G/N$ + $O/N$（全分片）。

### 4.4 all-reduce 分解

$$
\text{all-reduce}(V_i)=\text{AllGather}(\text{reduce-scatter}(V_i))
$$

- reduce-scatter：$V_i\to R_j$（归约片）；
- AllGather：$[R_j]\to$ 全量归约；
- 通信量同 all-reduce（$2(N-1)/N V$），但 ZeRO-2 中间停（不 AllGather），省显存。

## 5. 代码示例

```python
import torch

def ring_reduce_scatter(simulated_ranks):
    """ring reduce-scatter 概念模拟(N rank,各全量 V_i,输出归约片)"""
    N = len(simulated_ranks)
    V = simulated_ranks[0].shape[0]
    chunk = V // N
    blocks = [r.chunk(N, 0) for r in simulated_ranks]  # 各 rank 全量分 N 片
    # scatter-reduce 阶段(同 all-reduce 的前半)
    for step in range(N - 1):
        new_blocks = [[b.clone() for b in bl] for bl in blocks]
        for i in range(N):
            send_block = (i - step) % N
            recv_rank = (i + 1) % N
            # rank i 的 block[send_block] 累加到 rank recv 的同块
            new_blocks[recv_rank][send_block] = blocks[recv_rank][send_block] + blocks[i][send_block]
        blocks = new_blocks
    # N-1 步后: rank i 有 block[(i+1)%N] 的完整归约
    result = [blocks[i][(i + 1) % N] for i in range(N)]
    return result

N, V = 4, 8
ranks = [torch.randn(V) for _ in range(N)]  # 各 rank 全量
expected_full = sum(ranks)  # 全归约
expected_chunks = expected_full.chunk(N, 0)
result = ring_reduce_scatter(ranks)
print("ring reduce-scatter 结果(各 rank 得归约片,不同):")
for i, r in enumerate(result):
    print(f"  rank {i} 片: {r.tolist()}, = 全归约片{i} 一致={torch.allclose(r, expected_chunks[i])}")
print(f"各 rank 片不同(分散),拼全 = all-reduce 结果")
print(f"每卡通信量: (N-1)/N*V = {(N-1)/N*V:.1f} 元素")

# ===== all-reduce = reduce-scatter + AllGather =====
gathered = torch.cat([r for r in result], 0)  # AllGather 拼全
print(f"\nreduce-scatter + AllGather = all-reduce:")
print(f"  拼全: {gathered[:4].tolist()}... = 全归约? {torch.allclose(gathered, expected_full)}")

# ===== 实际 PyTorch 用法 =====
print("\n实际 PyTorch 用法:")
print("  input = torch.randn(V)  # 各 rank 全梯度")
print("  output = [torch.zeros(V//N) for _ in range(N)]")
print("  dist.reduce_scatter(output, input)  # 各 rank 得归约片")
print("  # 或 dist.reduce_scatter_tensor(单 tensor 输入输出)")

# ===== ZeRO-2 用途 =====
print("\nZeRO-2:")
print("  各 rank 算全梯度 -> reduce-scatter -> 各 rank 持归约梯度片 G/N")
print("  各 rank 只更新自己参数片(用 G/N + 优化器片 O/N)")
print("  显存: G+O -> (G+O)/N(省 N 倍)")
```

> [!tip] 实际工程
> - **PyTorch**：`dist.reduce_scatter(output_list, input)` 或 `dist.reduce_scatter_tensor`（单 tensor，高效）；
> - **NCCL**：底层 ring/tree；
> - **ZeRO-2/3**：DeepSpeed/FSDP 用 reduce-scatter 分片梯度；
> - **FSDP**：PyTorch 原生 `FullyShardedDataParallel`，reduce-scatter 梯度 + AllGather 权重。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[all-reduce]]（组合关系）、[[AllGather]]（互补）、[[NCCL通信拓扑]]、[[rank与world size]]、[[process group]]、[[torch.distributed]]。
- **下游（应用）**: [[Fully Sharded Data Parallel|FSDP/ZeRO-2/3]]（梯度分片）、all-reduce 分解、分片参数更新。
- **对比 / 易混**:
  - **reduce-scatter vs [[all-reduce]]**：前者输出归约片（各 rank 不同），后者输出全归约（相同）。reduce-scatter 是 all-reduce 的前半。
  - **reduce-scatter vs [[AllGather]]**：前者全量→归约片，后者片→全量。方向反，互补。
  - **reduce-scatter vs reduce**：前者归约+分散到所有 rank（各片不同），后者归约到 rank 0（仅 rank 0 有）。
  - **reduce-scatter vs all-to-all**：前者归约+分散，后者纯交换（不归约）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"reduce-scatter 输出相同"**：错。各 rank 得归约片（不同），非 all-reduce 的相同全量。
> 2. **"reduce-scatter 等于 all-reduce"**：错。reduce-scatter 只归约分片，all-reduce 还需 AllGather 拼全。
> 3. **"reduce-scatter 通信量比 all-reduce 少"**：对（一半），但不拼全（各 rank 片不同）。
> 4. **"reduce-scatter 归约到 rank 0"**：错。归约 + 分散到所有 rank（各片不同），非仅 rank 0。
> 5. **"ZeRO-2 用 all-reduce"**：错。ZeRO-2 用 reduce-scatter（分片梯度，省显存），非 all-reduce（全梯度）。
> 6. **"reduce-scatter 不延迟"**：错。ring 延迟 $O(N)$。
> 7. **"reduce-scatter + AllGather 通信更多"**：错。等价 all-reduce 通信量，但中间分片省显存。

## 8. 延伸细节

### 8.1 reduce-scatter 在 ZeRO-2/3

**ZeRO-2**：
- 反向后 reduce-scatter 梯度（各 rank 得归约梯度片 $G/N$）；
- 各 rank 用 $G/N$ + 优化器片 $O/N$ 更新自己参数片；
- 显存：$P$（权重全）+ $G/N$ + $O/N$（省梯度+优化器）；
- 通信：reduce-scatter（$\sim V$），同 all-reduce 一半。

**ZeRO-3**：
- + 权重分片，算时 AllGather 权重拼全；
- 反向：AllGather 权重（用时分片）→ reduce-scatter 梯度；
- 显存：$P/N + G/N + O/N$（全分片）；
- 通信：$\sim2V$（AllGather 权重 + reduce-scatter 梯度），同 all-reduce 量级；
- 详见 [[Fully Sharded Data Parallel]]。

### 8.2 reduce_scatter_tensor

PyTorch 新 API（`dist.reduce_scatter_tensor`）：
- 单 tensor 输入输出（连续内存，高效）；
- vs 旧 `reduce_scatter`（列表）；
- NCCL 底层优化；
- 推荐用。

### 8.3 ring reduce-scatter 的两阶段

reduce-scatter 实际是 all-reduce 的 scatter-reduce 阶段（无 all-gather）：
1. $N-1$ 步累加，每 rank 得一个块的完整归约；
2. 结束：各 rank 持归约片；
3. 不做 all-gather（故输出片不同）。

### 8.4 reduce-scatter 的负载均衡

- 每 rank 发/收 $(N-1)/N V$，均衡；
- vs 朴素 reduce（rank 0 收 $NV$ 瓶颈），reduce-scatter 均衡且带宽最优；
- 这是 ZeRO 选 reduce-scatter（非 reduce+broadcast）的原因。

### 8.5 reduce-scatter 与分片更新

- reduce-scatter 后各 rank 持归约梯度片 $R_j$（对应参数片 $j$）；
- 各 rank 用 $R_j$ 更新参数片 $j$（优化器状态也片 $j$）；
- 各 rank 间参数片不同，但合起来是完整更新；
- 这是 ZeRO 的"分片优化器"原理。

### 8.6 通信优化

- **ring/tree**：大消息 ring，小消息 tree；
- **多 ring**：NCCL 多 ring 并行；
- **overlap**：与反向计算重叠（[[overlap strategy]]），减延迟；
- **bucket**：分桶 reduce-scatter，小桶先发，overlap 反向；
- 详见 [[overlap strategy]]/[[compute vs communication bottleneck]]。

---
相关: [[通信机制]]、[[all-reduce]]、[[AllGather]]、[[NCCL通信拓扑]]、[[rank与world size]]、[[process group]]、[[torch.distributed]]、[[Fully Sharded Data Parallel]]、[[overlap strategy]]、[[compute vs communication bottleneck]]
