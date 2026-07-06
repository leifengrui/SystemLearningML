# process group

> **所属章节**: [[分布式基础]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 中等

## 1. 一句话定义

**Process Group（进程组）** 是 `torch.distributed` 里对**一部分 rank 的逻辑编组**——一个 ProcessGroup 对象代表"这 N 个 rank 之间约定好可以互相做 collective 通信（all-reduce/all-gather/broadcast…）"的子集。默认有一个包含全部 rank 的 `WORLD` 组；`new_group(ranks=[...])` 可创建任意子组，使**不同并行维度各自在子组内通信**成为可能（3D 并行的根基）。

> [!note] 名字辨析
> "Process Group" 在 PyTorch 语境有两层含义：(a) **逻辑子组**（`new_group` 返回的 `ProcessGroup` 对象，是一组 rank 的"频道"）；(b) **后端通信器**（每个 group 在 NCCL 里建一个 communicator，绑定一套通信拓扑）。二者一体两面：建一个 group = 在这些 rank 间建一个 NCCL communicator。

## 2. 为什么需要它（动机与背景）

单纯一个 `WORLD` 组够用于 [[DDP]]（所有 rank 一起 all-reduce 梯度）。但复杂并行需要**分组通信**：

- **3D 并行**（[[3D parallelism]]）：DP 维度在 DP 组里 all-reduce，TP 维度在 TP 组里 all-reduce，PP 维度在 PP 组里点对点——三个维度三个不同子组，互不干扰。
- **TP（[[Tensor Parallel]]）**：同一层的切片分布在几个 rank 上，它们之间要做 all-reduce 把部分和拼成完整输出 → 这几个 rank 组成 TP 组。
- **专家并行 / MoE**：同 token 路由到不同专家所在 rank，需在"专家 group"内 all-to-all。
- **评估时只让 DP 头汇总**：评估指标只在数据并行头之间 all-gather，避免打扰 TP/PP 组。

没有子组机制，就要手写"哪些 rank 该参与这次通信"的过滤逻辑，既易错又低效（NCCL 没法建专用拓扑）。`new_group` 把"子组通信"做成一等公民，让多维度并行能正交组合。

## 3. 核心概念详解

### 3.1 默认组：`dist.group.WORLD`

`init_process_group` 后自动建一个包含所有 rank 的组，叫 `WORLD`。`dist.all_reduce(t)` 不传 group 参数就是用 `WORLD`。对单维度并行（纯 DDP）这就够。

### 3.2 创建子组：`new_group`

```python
pg = dist.new_group(ranks=[0,1,2,3], backend="nccl")
# 所有 rank 都要调 new_group（即使不在列表里也要调，否则 rendezvous 死锁）
# 返回值 pg：在 [0,1,2,3] 内有效；在 [4,5,6,7] rank 上返回的 pg 用于其它组
```

> [!warning] `new_group` 必须全员调用
> 集体操作。即使某 rank 不属于新组，也要参与建组 rendezvous，否则其他 rank 会卡在 rendezvous。常见死锁来源。

### 3.3 子组上的 collective

```python
dist.all_reduce(t, group=pg)        # 只在 pg 内的 rank 间做
dist.broadcast(t, src=0, group=pg)
```

不在 `pg` 里的 rank 调它会被忽略/报错；通信只在组内 rank 间发生。

### 3.4 组的后端可不同

`new_group(..., backend="gloo")` 可以让某个子组用 CPU gloo（控制流、小张量），而 `WORLD` 用 nccl（GPU 大张量）。混合后端在"GPU 训练 + CPU 控制流同步"场景有用。

### 3.5 NCCL communicator 与拓扑

每个 group 在 NCCL 后端下对应一个独立 **communicator**，会探测组内 rank 间的物理拓扑（NVLink/PCIe/IB）建最优 ring/tree。建组有开销（拓扑探测 + 通道建立），不宜频繁新建。多组并存会占显存（每个 communicator 预留 buffer）。

### 3.6 rank 在组内的编号：`group.rank()`

- 全局 `dist.get_rank()` = 在 `WORLD` 里的编号。
- `dist.get_rank(group=pg)` = 在 `pg` 里的局部编号（0..len-1）。
- `dist.get_world_size(group=pg)` = 组大小。

3D 并行里一个全局 rank 同时属于 DP/TP/PP 三组，各自有局部编号，用于算"我在这个维度排第几"。

## 4. 数学原理 / 公式

无新数学，但子组 collective 的语义形式化：设组 $G\subseteq\{0,\dots,N-1\}$，张量 $t_r$ 在 $r\in G$ 上有定义，则

$$
\text{all\_reduce}^G_{\text{SUM}}:\quad t'_r=\sum_{i\in G} t_i,\ \ \forall r\in G
$$

组外 rank 不参与、不接收结果。3D 并行可看作在三个正交子组上分别做 collective 的组合。

### 3D 并行的组划分示例

设 8 卡，DP=2、TP=2、PP=2。一种划分：
- DP 组：`{0,4}`,`{1,5}`,`{2,6}`,`{3,7}`（4 个组，每组 2 rank）
- TP 组：`{0,1}`,`{2,3}`,`{4,5}`,`{6,7}`
- PP 组：`{0,2,4,6}×?`（具体看 stage 划分）

每个全局 rank 同时进三个组，三组 collective 互不影响。

## 5. 代码示例

```python
import os, torch
import torch.distributed as dist

def setup():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

setup()
rank, world = dist.get_rank(), dist.get_world_size()

# 1) 建一个偶数 rank 的子组
even_ranks = list(range(0, world, 2))
pg_even = dist.new_group(ranks=even_ranks)   # 全员都要调

# 2) 建一个奇数 rank 的子组
odd_ranks = list(range(1, world, 2))
pg_odd = dist.new_group(ranks=odd_ranks)

my_pg = pg_even if rank % 2 == 0 else pg_odd
my_group_rank = dist.get_rank(group=my_pg)

t = torch.tensor([rank], device="cuda")
# 只在同奇偶的 rank 间 all-reduce
if rank in even_ranks:
    dist.all_reduce(t, group=pg_even)
    print(f"global {rank} group_rank {my_group_rank} even_sum={t.item()}")

# 3) 模拟 3D 并行的 TP 组（相邻两卡一组）
tp_ranks = [rank - (rank % 2), rank - (rank % 2) + 1]
pg_tp = dist.new_group(ranks=tp_ranks)
dist.all_reduce(t, group=pg_tp)
print(f"global {rank} tp_sum={t.item()}")

dist.destroy_process_group()
```

启动：`torchrun --nproc_per_node=8 demo.py`。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[torch.distributed]]（group 是它的概念）、[[NCCL backend]]（每个 group 一个 communicator）、[[rank与world size]]（组的成员就是 rank）。
- **下游（应用）**: [[3D parallelism]]（DP/TP/PP 三组）、[[Tensor Parallel]]（TP 组内 all-reduce）、[[Pipeline Parallel]]（PP 相邻 stage 的点对点，也可视为 size=2 的组）、[[DDP]]（用 WORLD 组）、[[FSDP]]（用 WORLD 或自定义组）、MoE all-to-all（专家组）。
- **对比 / 易混**:
  - `WORLD` 组 vs 子组：前者全员，后者子集。
  - 全局 rank vs 组内 rank：`get_rank()` vs `get_rank(group=pg)`。
  - ProcessGroup（PyTorch 逻辑组）vs NCCL communicator（底层通信器）：一对一映射。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **`new_group` 只在被选中的 rank 调** → 其他 rank 卡在 rendezvous 死锁。**全员必须都调**，哪怕它不属于该组。
> 2. **混淆全局 rank 与组内 rank** → 在 TP 组里用 `dist.get_rank()` 做 src 参数会错；要用 `dist.get_rank(group=pg)`。
> 3. **在非组员的 rank 上调该组的 collective** → 行为未定义/报错；务必用 `if rank in group_ranks` 守卫。
> 4. **频繁 `new_group`** → 每次都建 NCCL communicator、占显存、慢；并行维度在训练开始一次性建好复用。
> 5. **以为子组 backend 必须和 WORLD 一致** → 可以不同，gloo 子组 + nccl WORLD 是合法组合。
> 6. **建组顺序不一致** → 所有 rank 必须以**相同顺序**调 `new_group`，否则 communicator 对不上。
> 7. **`group` 参数传错对象** → 传了别的 group 的句柄，跨组 collective 错位，难调试。

## 8. 延伸细节

### 8.1 3D 并行组划分的标准算法

给定 $N$ 卡和 $(dp, tp, pp)$ 满足 $dp\cdot tp\cdot pp=N$，标准划分（按行优先）：
- global rank $r$ 分解为 $(i_{pp}, i_{dp}, i_{tp})$；
- TP 组 = 固定 $(i_{pp},i_{dp})$、变 $i_{tp}$ 的所有 $r$；
- DP 组 = 固定 $(i_{pp},i_{tp})$、变 $i_{dp}$；
- PP 组 = 固定 $(i_{dp},i_{tp})$、变 $i_{pp}$。

Megatron-LM 的并行即按此建三套子组。

### 8.2 communicator 数量与显存

每个 NCCL communicator 预留通信 buffer（几 MB~几十 MB）。3D 并行建三套组，组数随并行度增长；百卡集群可能建几十个 communicator，显存与建组时间需权衡。

### 8.3 `ProcessGroup` 的惰性初始化

PyTorch 2.x 引入 lazy init：`new_group` 不立即建 NCCL communicator，首次 collective 时才建，减少启动时间与无用 communicator。

### 8.4 跨机组与网络拓扑

同机组的 rank 走 NVLink（快），跨机组走 IB/ROCE（慢）。建组时尽量把"高频通信的 rank"放同机（靠 `torchrun` 的 `--nproc_per_node` 与 placement group 安排）。Ray 的 [[placement group]] 就是干这个的。

### 8.5 `split` API（新）

`dist.distributed_c10d.new_group` 之外，PyTorch 实验 API `dist.split` 可从父组按颜色分裂子组，简化 3D 并行建组代码，但稳定性待观察。

---
相关: [[分布式基础]]、[[torch.distributed]]、[[rank与world size]]、[[NCCL backend]]、[[3D parallelism]]、[[Tensor Parallel]]、[[Pipeline Parallel]]、[[DDP]]
