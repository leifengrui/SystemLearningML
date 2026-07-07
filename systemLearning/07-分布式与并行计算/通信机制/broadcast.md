# broadcast

> **所属章节**: [[通信机制]]
> **所属模块**: [[07-分布式与并行计算]]
> **难度**: 易-中（最朴素的集合通信原语）

## 1. 一句话定义

**broadcast（广播）** 是集合通信原语：**root rank 持有一份数据 $V$，经广播后所有 rank 都得到相同的 $V$**。它是"一传多"——单一数据源复制到全组，与 [[all-reduce]]（多归约成单）、[[AllGather]]（多片拼全）、[[reduce-scatter]]（归约后分散）并列。用途：**权重同步**（rank 0 把初始化/更新后的 $\theta$ 发给所有 rank，见 [[weight sync mechanism]]、[[Fully Sharded Data Parallel|FSDP]] 启动时参数对齐）、[[Distributed Data Parallel|DDP]] 启动时参数与 buffer 同步、控制流元数据广播（`dist.broadcast_object_list` 传 config/dict）。[[NCCL通信拓扑|NCCL]] 底层实现，ring/tree 拓扑。它是**最朴素**的 collective——all-reduce/AllGather 在概念上都能拆成 broadcast/reduce 的组合。

> [!note] broadcast 的语义
> - 输入：root rank 有 $V$，其他 rank 有任意占位；
> - 输出：所有 rank 得相同 $V$（root 的拷贝）；
> - 不同于 [[all-reduce]]（多→单归约再分发）、[[AllGather]]（多片→拼接全量）、[[reduce-scatter]]（归约+分散）。

## 2. 为什么需要它（动机与背景）

多卡训练里有大量"一份东西要全组都有"的需求：

1. **参数初始化对齐**：各 rank 各自 `MyModel()` 实例化时随机种子不同，参数初始值不同。开训前必须由 rank 0 broadcast 一份初始 $\theta$ 给所有 rank，否则各 rank 从不同起点出发、梯度同步后也对不齐。这是 `DDP` 构造时自动做的第一件事。
2. **权重同步（训推分离 / rollout 刷新）**：[[weight sync mechanism]] 里 policy training（PT）端训完一段，新 $\theta$ 要发给 policy deployment（PD）/rollout worker。最简单实现：rank 0 broadcast 全 $\theta$，所有 worker 接住。
3. **Buffer 同步**：[[Distributed Data Parallel|DDP]] 默认每 forward 前 broadcast buffer（如 BatchNorm running stats），保证各 rank 非学习状态一致。
4. **元数据/配置广播**：小对象（config dict、超参、token 序列）用 `dist.broadcast_object_list` 从 rank 0 发给所有 rank，避免每 rank 各读一份文件。
5. **权重加载后分发**：rank 0 从磁盘 load checkpoint，broadcast 给其他 rank（比每 rank 各 load 省磁盘 IO）。

没有 broadcast，这些"一传多"就得手写 N 次 send/recv，且 root 要逐个发，带宽不均。broadcast 把它做成带宽最优的 collective。

## 3. 核心概念详解

### 3.1 语义

$N$ 个 rank，root 持张量 $V$：
- **broadcast**：所有 rank 得 $V$（root 的拷贝）；
- root 是谁由参数指定（`src=0`），其他 rank 是接收方；
- 输出所有 rank 相同（与 [[all-reduce]] 输出相同，但输入语义不同：broadcast 输入只有 root 有意义，all-reduce 所有 rank 的输入都参与归约）。

### 3.2 朴素实现：root 逐个 send

root 向其余 $N-1$ 个 rank 各发一份 $V$：
- root 通信量 $(N-1)V$（发送瓶颈）；
- 其他 rank $V$（接收）；
- 延迟 $O(N)$ 串行；
- 问题：root 带宽瓶颈，不均衡。

### 3.3 tree broadcast（延迟优、带宽次优）

rank 排成二叉树，root 向两子节点发，子节点再向其子节点发：
- 每层并行，延迟 $O(\log N)$（优，小消息）；
- root 只发 2 份，非瓶颈；
- 但中下层节点要转发，总通信量 $O(NV)$ 仍较高，大消息不如 ring。

### 3.4 ring broadcast（带宽优）

类似 ring all-reduce 的"半边"：root 的 $V$ 分 $N$ 块沿环传递，$N-1$ 步后所有 rank 都有所有块：
- 每卡通信量 $\frac{N-1}{N}V$（带宽最优，与 $N$ 几乎无关）；
- 延迟 $O(N)$（大 $N$ 显）；
- 适合大消息（如权重张量）。

### 3.5 NCCL 的混合实现

[[NCCL通信拓扑|NCCL]] 自动按消息大小选：
- **小消息**：tree（延迟优 $O(\log N)$）；
- **大消息**：ring（带宽优 $\sim V$）；
- 多 ring 并行充分利用 NVLink 多通道。

### 3.6 与其他原语的关系

| 原语 | 语义 | 输出 |
|---|---|---|
| **broadcast** | 一传多 | 各 rank 相同源数据 |
| **[[all-reduce]]** | 多归约到所有 | 各 rank 相同归约值 |
| **[[AllGather]]** | 多片拼全 | 各 rank 相同拼接全量 |
| **[[reduce-scatter]]** | 归约+分散 | 各 rank 不同归约片 |
| **reduce** | 多归约到根 | 仅 root 有归约值 |

- **all-reduce = reduce + broadcast**（概念分解）：先 reduce 到 root，再 broadcast 给所有。朴素实现即如此，但 ring all-reduce 更优。
- **broadcast 是 AllGather 的退化**：当只有一个 rank 的片非空时，AllGather 等价于 broadcast。

## 4. 数学原理 / 公式

### 4.1 ring broadcast 通信量

$$
\text{comm per rank}=\frac{N-1}{N}\cdot V\approx V\quad(N\text{ 大})
$$

- $V$：总数据量（root 的张量大小）；
- 每卡发送 $\frac{N-1}{N}V$，接收 $\frac{N-1}{N}V$；
- 带宽最优：每卡通信量与 $N$ 几乎无关、均衡。

### 4.2 vs 朴素 root 逐发

$$
\text{朴素 root}: (N-1)V,\quad \text{其他}: V;\quad \text{ring}: V\text{（每卡均衡）}
$$

- 朴素 root 瓶颈 $(N-1)V$，ring 每卡 $\sim V$，均衡且省 $N$ 倍。

### 4.3 延迟

- **ring**：$N-1$ 步串行，$O(N)$；
- **tree**：$\log N$ 步，$O(\log N)$；
- **小消息选 tree**（延迟瓶颈），**大消息选 ring**（带宽瓶颈）。

### 4.4 all-reduce = reduce + broadcast（朴素分解）

$$
\text{all-reduce}(V_i)=\text{broadcast}\big(\text{reduce}(V_i)\big)
$$

- 先 reduce 所有 rank 的 $V_i$ 到 root，再 broadcast 结果到所有 rank；
- 通信量同 ring all-reduce 的 $2(N-1)/N\cdot V$（reduce + broadcast 各一半），但朴素实现 root 瓶颈，ring all-reduce 更优。

## 5. 代码示例

```python
# broadcast_demo.py —— torchrun --nproc_per_node=4 broadcast_demo.py
import os, torch
import torch.distributed as dist

def setup():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    rank = dist.get_rank()

    # 1) 张量 broadcast：rank 0 的 V 发给所有 rank
    if rank == 0:
        V = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    else:
        V = torch.zeros(3, device="cuda")   # 占位，会被覆盖
    dist.broadcast(V, src=0)
    print(f"[rank {rank}] V={V.tolist()}")   # 所有 rank: [1,2,3]

    # 2) 元数据 broadcast：rank 0 的 dict 发给所有 rank
    obj = [None] * dist.get_world_size()      # 占位 list
    if rank == 0:
        obj[0] = {"lr": 1e-4, "epochs": 3, "note": "hello"}
    dist.broadcast_object_list(obj, src=0)
    cfg = obj[0]
    print(f"[rank {rank}] cfg={cfg}")

    # 3) 模型初始化对齐（概念）：rank 0 实例化后 broadcast 给所有 rank
    model = torch.nn.Linear(4, 4).cuda()      # 各 rank 随机初始化不同
    for p in model.parameters():
        dist.broadcast(p.data, src=0)         # 强制对齐到 rank 0 的初始化
    # 现在 all rank 的 model 参数一致

    cleanup()

if __name__ == "__main__":
    main()
```

> [!tip] 实际工程
> - **PyTorch**：`dist.broadcast(tensor, src=0)`（原地，接收方 tensor 被覆盖）；
> - **元数据**：`dist.broadcast_object_list(obj_list, src=0)` 传 dict/list/config；
> - **DDP 自动做**：`DistributedDataParallel(model)` 构造时自动 broadcast 模型参数与 buffer 到所有 rank，无需手动；
> - **weight sync**：rank 0 加载 ckpt 后 broadcast 全 $\theta$ 给 worker，比每 rank 各 load 快。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[NCCL通信拓扑]]（底层实现）、[[rank与world size]]（root 身份）、[[process group]]、[[torch.distributed]]、集合通信理论。
- **下游（应用）**: [[Distributed Data Parallel|DDP]]（启动时参数对齐、buffer 同步）、[[weight sync mechanism]]（训推权重广播）、[[Data Parallel]]（参数初始化对齐）、checkpoint 分发、配置广播。
- **对比 / 易混**:
  - **broadcast vs [[all-reduce]]**：前者一传多（输入只有 root 有意义），后者多归约（所有 rank 输入参与）。输出都是"所有 rank 相同"，但语义不同。
  - **broadcast vs [[AllGather]]**：前者单一源复制，后者多片拼接。当只有一个 rank 片非空时 AllGather 退化成 broadcast。
  - **broadcast vs reduce**：broadcast 一→多，reduce 多→一（仅 root 得结果）。二者方向相反，组合成 all-reduce。
  - **broadcast vs send/recv**：前者集体通信（全组参与、自动拓扑），后者点对点（手动指定 src/dst）。
  - **ring vs tree**：ring 带宽优（大消息），tree 延迟优（小消息）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"broadcast 输入所有 rank 都参与"** → 错。只有 root 的数据有意义，其他 rank 是占位（会被覆盖）。
> 2. **"broadcast 省带宽"** → 错。每卡通信量 $\sim V$（与数据量同），是均衡/带宽最优，非"省"。总带宽仍 $\sim NV$。
> 3. **"ring 总优于 tree"** → 错。ring 大消息优（带宽），tree 小消息优（延迟）。
> 4. **"DDP 训练时每步 broadcast 参数"** → 错。DDP 只在**构造时** broadcast 一次参数对齐，训练中靠梯度 all-reduce 同步，不再 broadcast 参数。
> 5. **"broadcast 小对象用 `dist.broadcast`"** → 不对。小对象/元数据用 `dist.broadcast_object_list`（自动 pickle）；`dist.broadcast` 只对张量。
> 6. **接收方 tensor shape/dtype 不匹配** → broadcast 要求所有 rank 的 tensor 同 shape/dtype，否则报错；接收方要先建同 shape 的占位 tensor。
> 7. **`src` 写成 rank 而非全局 rank** → 在子 [[process group]] 里 `src` 是组内相对 rank，易错；注意上下文。
> 8. **"broadcast 是点对点"** → 错。broadcast 是集体通信，所有 rank 必须同时调用（否则阻塞/死锁）。
> 9. **"root 不调 broadcast 也行"** → 错。root 也必须调用（既是发送方也是 collective 参与者）。
> 10. **大权重每步 broadcast** → 慢；weight sync 一般按"训一段刷一次"的节奏，不是每 step。

## 8. 延伸细节

### 8.1 broadcast 在 DDP 启动时的角色

`DistributedDataParallel(model)` 构造函数会：①把模型搬到正确 device；②**broadcast 所有参数与 buffer 到所有 rank**（以 rank 0 为 src），保证起点一致；③注册梯度 bucket all-reduce 钩子。所以**DDP 用户不用手动 broadcast 参数**——构造时已对齐。这也是"各 rank 随机种子不同也没事"的原因。

### 8.2 broadcast vs AllGather 在 weight sync 的选择

- **broadcast**：root 有完整 $\theta$，发给所有 rank——传统 push 模式；
- **AllGather**：每 rank 持 $\theta$ 的一片，拼全——FSDP/ZeRO-3 风格，省 root 内存；
- 现 LLM 训推分离常混合：PT 端 FSDP gather 全量 → broadcast 给 PD/worker。

### 8.3 元数据广播：`broadcast_object_list`

`dist.broadcast_object_list(obj_list, src=0)` 把 list 里的 Python 对象（dict/config/list）pickle 成字节串，broadcast 给所有 rank。常用于：rank 0 读 config/采样 prompt/dataloader 状态，广播给所有 rank 保证一致。注意对象要可 pickle，且 list 长度 ≥ world_size。

### 8.4 与 `scatter`/`gather` 的关系

- **scatter**：root 的一组数据分发（每 rank 拿一片，root 不同片）——broadcast 的"分片版"；
- **gather**：各 rank 的片收到 root（多→一）——broadcast 的反向；
- 二者都是"一与多"的非归约通信，broadcast 是 scatter 的"所有 rank 拿同一份"特例。

### 8.5 延迟优化

- **tree/ring 混合**：小消息 tree，大消息 ring；
- **多 ring 并行**：用多链路提带宽；
- **NCCL 自动选**：开发者一般不手动指定算法。

### 8.6 历史

- **MPI_Bcast**：HPC 经典接口；
- **tree/ring broadcast**：经典算法；
- **NCCL**：NVIDIA 实现，GPU 集合通信库标准；
- broadcast 是最古老的 collective，理解它是理解其他 collective（reduce/AllGather/all-reduce）的基础。

---
相关: [[通信机制]]、[[all-reduce]]、[[AllGather]]、[[reduce-scatter]]、[[NCCL通信拓扑]]、[[rank与world size]]、[[process group]]、[[torch.distributed]]、[[Distributed Data Parallel]]、[[weight sync mechanism]]、[[Data Parallel]]、[[overlap strategy]]
