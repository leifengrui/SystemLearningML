# rank与world size

> **所属章节**: [[分布式基础]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 基础（多卡术语第一课）

## 1. 一句话定义

**world_size** 是分布式训练里**总进程数（通常=总 GPU 数）**；**rank** 是**每个进程在全局中的唯一编号**（0 到 world_size-1）；**local_rank** 是**一个进程在所在节点（机器）内的 GPU 编号**（0 到 nproc_per_node-1）。三者是 `torch.distributed` 用来给每个进程定位身份的最基本坐标。

> [!note] 三个 rank 的区分
> - **global rank**（`dist.get_rank()`）：全局唯一身份，rank 0 通常当"主进程"做 IO/打印/保存。
> - **local rank**（`os.environ["LOCAL_RANK"]`）：本机内 GPU 编号，用来 `torch.cuda.set_device(local_rank)` 绑定本进程独占哪块卡。
> - **group rank**（`dist.get_rank(group=pg)`）：在某子 [[process group]] 内的局部编号，见 [[process group]] §3.6。

## 2. 为什么需要它（动机与背景）

多进程训练里，每个进程跑同一份脚本，**没有身份编号就分不清"我是谁、该干啥"**：

- **绑卡**：本进程该用哪块 GPU？靠 `local_rank` 决定 `set_device`。
- **数据分片**：DataLoader 要让每个 rank 拿不重叠的数据——用 `rank` 做 sampler 的 seed/offset，保证各 rank 见不同 batch。
- **主从分工**：保存 checkpoint、打印日志、下载数据只在 rank 0 做，避免重复；靠 `if rank==0` 守卫。
- **集体通信身份**：`broadcast(src=0)` 要知道谁是 0；`all_reduce` 不需显式身份但要求全员参与。
- **弹性训练**：节点增减时 world_size 变，rank 重排，需动态获取。

没有 rank/world_size，所有进程行为完全相同 → 重复保存、数据重复、抢卡崩溃。它们是分布式训练的"身份证 + 总人数"。

## 3. 核心概念详解

### 3.1 world_size

`dist.get_world_size()` = 总进程数。8 卡单机=8，4 机 8 卡=32。它决定 collective 的参与方数量、数据分片粒度、有效 batch 大小（`effective_batch = micro_batch × world_size × accum`）。

### 3.2 rank（global）

`dist.get_rank()` ∈ [0, world_size)。
- **rank 0 的约定职责**：保存 checkpoint、写日志、下载/预处理数据集、broadcast 权重给其他 rank、做主评估汇总。
- 其他 rank 通常只算+通信，不做 IO。
- 但**计算职责平等**：每个 rank 都前向+反向+all-reduce，不是 rank 0 指挥其他人（去中心化 collective）。

### 3.3 local_rank

`int(os.environ["LOCAL_RANK"])` ∈ [0, nproc_per_node)。
- 一台 8 卡机起 8 进程，local_rank ∈ [0,8)。
- 必须在 `init_process_group` 前后 `torch.cuda.set_device(local_rank)`，否则所有进程默认用 cuda:0 → 8 进程挤一块卡 OOM。
- `torchrun` 自动注入 `LOCAL_RANK` 环境变量。

### 3.4 node / nnodes

- **node**：一台机器 = 一个节点。`--nnodes=4` = 4 台机。
- **nproc_per_node**：每节点进程数（=每节点 GPU 数）。
- `world_size = nnodes × nproc_per_node`。
- 跨机时 rank 0 通常在 head 节点（`--rdzv_endpoint` 指定）。

### 3.5 rank 到 GPU 的映射

`torchrun --nproc_per_node=8` 在一台机上起 8 进程，给每个注入 `RANK=0..7`、`LOCAL_RANK=0..7`。多机时 `RANK` 全局连续，`LOCAL_RANK` 每机重置。映射关系：

| RANK（global） | LOCAL_RANK | 节点 |
|---|---|---|
| 0..7 | 0..7 | node 0 |
| 8..15 | 0..7 | node 1 |

### 3.6 用 rank 做数据分片

`DistributedSampler` 用 `rank` 和 `world_size` 把数据集切成不重叠的片：

```python
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, num_replicas=world_size, rank=rank, shuffle=True)
# rank 0 见第 0,8,16,... 样本；rank 1 见 1,9,17,...（示意）
```

每 epoch 开始 `sampler.set_epoch(epoch)` 改 shuffle 种子，否则每轮数据顺序相同。

### 3.7 主从约定清单

| 操作 | 谁做 | 原因 |
|---|---|---|
| 下载/预处理数据 | rank 0 | 避免重复下载 |
| 打印 loss/日志 | rank 0 | 避免刷屏 |
| 保存 checkpoint | rank 0 | 避免多进程写冲突 |
| 评估指标 all-gather 后打印 | rank 0 | 汇总后只打一份 |
| broadcast 权重 | rank 0 当 src | 训推分离同步 |
| 前向+反向+all-reduce 梯度 | 所有 rank | 集体通信需全员 |

## 4. 数学原理 / 公式

无复杂公式，但有几个恒等式：

$$
\text{world\_size}=\text{nnodes}\times\text{nproc\_per\_node}
$$

$$
\text{effective\_batch}=\text{micro\_batch}\times\text{world\_size}\times\text{accum}
$$

`DistributedSampler` 给每个 rank 分配约 `len(dataset)/world_size` 个样本，按 `(rank + i·world_size) % len` 索引（简化），保证不重叠（当 `len` 整除 `world_size` 时严格不重叠；不整除时补齐或丢弃，见 sampler 文档）。

## 5. 代码示例

```python
import os, torch
import torch.distributed as dist

def setup():
    dist.init_process_group("nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)        # 关键：绑本进程的卡
    return local_rank

def main():
    local_rank = setup()
    rank = dist.get_rank()
    world = dist.get_world_size()
    if rank == 0:
        print(f"hello from rank0, world_size={world}")

    # 只 rank0 保存
    if rank == 0:
        torch.save({"step": 0}, "ckpt.pt")
    dist.barrier()                            # 等 rank0 写完

    # 数据分片
    from torch.utils.data import TensorDataset
    from torch.utils.data.distributed import DistributedSampler
    ds = TensorDataset(torch.arange(32))
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, sampler=sampler)
    for epoch in range(2):
        sampler.set_epoch(epoch)              # 必须每 epoch 改种子
        for batch in loader:
            pass  # 各 rank 拿不重叠的样本

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

启动：`torchrun --nproc_per_node=8 --nnodes=1 demo.py`。
多机：`torchrun --nproc_per_node=8 --nnodes=2 --rdzv_backend=c10d --rdzv_endpoint=head:29500 demo.py`。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[torch.distributed]]（rank/world_size 是它的 API）、`torchrun`（注入 env vars）。
- **下游（应用）**: [[DDP]]（每个 rank 算梯度再 all-reduce）、[[process group]]（rank 是组的成员）、[[Data Parallel]]（数据按 rank 分片）、[[weight sync mechanism]]（rank 0 broadcast 权重）、[[batch与mini-batch]]（effective batch 含 world_size）、[[Ray与分布式调度]]（Ray actor 也类似 rank 概念）。
- **对比 / 易混**:
  - rank vs local_rank：全局 vs 本机，见 §3。
  - rank 0 vs master node：rank 0 是进程编号；head/master node 是 rendezvous 所在节点，二者可不同。
  - world_size vs nproc_per_node：前者全局总数，后者单节点数。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **不 `set_device(local_rank)`** → 8 进程都抢 cuda:0，OOM 崩。每个进程第一步绑卡。
> 2. **所有 rank 都保存 checkpoint** → 写冲突/重复占盘；只 rank 0 存。
> 3. **所有 rank 都 print** → 日志刷屏 8/32 倍；用 `if rank==0` 守卫或 `dist.all_gather` 后汇总。
> 4. **DistributedSampler 忘 `set_epoch`** → 每轮数据顺序相同，等于没 shuffle，泛化变差。
> 5. **用 `rank` 当随机种子** → 各 rank 种子不同看似分片，但模型初始化也变随机不一致；初始化应同种子（或 rank 0 broadcast），数据 sampler 用 `rank+epoch`。
> 6. **`effective_batch` 算错** → 忘乘 world_size 或 accum，LR 调参按错 batch 走。
> 7. **rank 0 崩了其他 rank 卡在 collective** → 单点故障；弹性训练（`--nnodes=min:max`）可重组。
> 8. **多机 LOCAL_RANK 没重置** → `torchrun` 已处理，但手写 spawn 时要自己管，否则跨机 local_rank 撞。

## 8. 延伸细节

### 8.1 rank 0 职责的"约定"本质

rank 0 当主是**社区约定**不是框架强制——`broadcast(src=k)` 可任选 src。但 IO/日志由 rank 0 做是工程习惯，因为它总是存在（弹性训练重排后也有 rank 0）。

### 8.2 弹性训练的 rank 重排

`torchrun --nnodes=2:4` 允许节点数在 2~4 间浮动。节点掉线后，剩余节点重新 rendezvous，`world_size` 与 `rank` 重新分配。训练逻辑要能容忍 rank 变化（用 `rank` 做数据分片时每步重读 `get_rank`，别缓存）。

### 8.3 local_rank 与 CUDA_VISIBLE_DEVICES

也可用 `CUDA_VISIBLE_DEVICES=0 python train.py`（每进程只见一块卡，local_rank 恒 0）。`torchrun` 不用这招，而是让所有进程见所有卡、靠 `set_device` 绑定。两种流派都行，`torchrun` 更主流。

### 8.4 rank 与梯度同步的等价性

DDP 下每个 rank 算的梯度经 all-reduce(MEAN) 后**完全一致** → 每个 rank 的参数演化相同 → 任何 rank 的 checkpoint 都代表"整个模型"。所以 rank 0 存的权重 = 任意 rank 的权重（在 DDP 下）。FSDP 不同（参数分片，存 checkpoint 要 gather）。

### 8.5 调试：只跑 rank 0

```python
if rank == 0:
    import pdb; pdb.set_trace()
```
但集体通信会让其他 rank 卡在 all-reduce 等待 → 调试单 rank 时要么用 `torchrun --nproc_per_node=1`，要么给其他 rank 跳过对应 collective（破坏一致性，仅限排查逻辑）。

---
相关: [[分布式基础]]、[[torch.distributed]]、[[process group]]、[[NCCL backend]]、[[DDP]]、[[Data Parallel]]、[[batch与mini-batch]]、[[weight sync mechanism]]
