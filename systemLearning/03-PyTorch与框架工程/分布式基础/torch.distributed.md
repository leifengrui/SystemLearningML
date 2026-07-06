# torch.distributed

> **所属章节**: [[分布式基础]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 中等（多卡训练入门）

## 1. 一句话定义

**`torch.distributed`（简称 `dist`）** 是 PyTorch 的**分布式通信后端框架**——它提供跨进程（通常一进程一 GPU）的**集合通信（collective ops：all-reduce / all-gather / broadcast / reduce-scatter…）、进程组管理、点对点通信、屏障同步**等原语，是 [[DDP]]、[[FSDP]]、[[RPC]]、[[Ray与分布式调度]] 集成 PyTorch 训练的底层基石。所有"多卡"训练，最终都落到 `torch.distributed` 的通信调用上。

> [!note] 别和 `nn.DataParallel` 混
> `nn.DataParallel`（[[Data Parallel]]）是单进程多线程、用 `torch.distributed` 不参与的旧方案，性能差已被弃用。现代多卡 = **多进程** + `torch.distributed` + `torchrun` 启动 → 即 [[DDP]]/[[FSDP]]。`torch.distributed` 是这条现代路线的地基。

## 2. 为什么需要它（动机与背景）

单卡装不下 LLM，必须多卡甚至多机。多卡训练的核心需求：

1. **进程间通信**：各 GPU 算完自己的梯度后，要把梯度**求和/平均**（all-reduce）得到一致梯度再各自更新，否则参数发散。
2. **进程协调**：N 个进程要一起开始、一起保存、一起评估，需要屏障（barrier）和 rank 协商。
3. **后端抽象**：GPU 间走 NCCL、CPU 间走 gloo、跨机走 MPI/gloo——上层算法不该关心底层传输细节。

`torch.distributed` 提供这一切：统一的 collective API（`dist.all_reduce` 等不关心后端）、`init_process_group` 初始化、`ProcessGroup` 抽象、点对点 send/recv。它把"多卡通信"从算法里解耦出来，使 [[DDP]] 只需调 `all_reduce`、[[FSDP]] 只需调 `all_gather`/`reduce_scatter` 即可表达算法。

## 3. 核心概念详解

### 3.1 初始化：`init_process_group`

```python
import torch.distributed as dist
dist.init_process_group(
    backend="nccl",       # GPU 用 nccl，CPU 用 gloo，跨机也可 mpi
    init_method="env://", # 约定由 torchrun 注入 env vars
    world_size=N,         # 总进程数
    rank=rank,            # 本进程编号
)
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

`init_method` 三种：
- `env://`：靠环境变量（`RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT`），最常用，`torchrun` 自动注入。
- `tcp://host:port`：指定 rendezvous 服务器。
- `file://`：用共享文件系统做 rendezvous（多机不便）。

### 3.2 后端（backend）对比

| 后端 | 设备 | 通信库 | 典型用途 |
|---|---|---|---|
| **nccl** | GPU | NVIDIA NCCL | GPU 多卡/多机，**首选** |
| **gloo** | CPU/GPU | 自研 | CPU 训练、控制流通信、非 NV 平台 |
| **mpi** | CPU/GPU | MPI | HPC 环境，需自编译 |
| **ucc** | 异构 | UCC | 新实验性，统一通信 |

GPU 训练几乎总是 `nccl`——它针对 NV 拓扑（NVLink/PCIe/NVSwitch）做了高度优化，是 GPU all-reduce 事实标准，见 [[NCCL backend]]。

### 3.3 集合通信原语一览

| 原语 | 语义 | 典型用途 |
|---|---|---|
| `broadcast` | 一→多：root 张量发给所有 | 权重同步、参数初始化 |
| `all_reduce` | 多→一一致：所有 rank 的张量求和（或 mean/max）后结果**每个 rank 都拿到** | **DDP 梯度同步** |
| `reduce` | 多→一：只有 root 拿到结果 | 偶尔省显存 |
| `all_gather` | 多→多：各 rank 的片段拼成完整张量，每个都拿到 | FSDP 参数收集、KV cache 拼接 |
| `reduce_scatter` | reduce 后再 scatter：各 rank 拿到结果的一段 | FSDP 梯度分片 |
| `scatter` / `gather` | 一→多分发 / 多→一收集 | 数据分片 |
| `barrier` | 同步点：所有 rank 到齐才继续 | 评估/保存的同步 |
| `send`/`recv`/`isend`/`irecv` | 点对点 | Pipeline 并行、rollout 通信 |

详见 [[all-reduce]]/[[all-gather]]/[[reduce-scatter]]（在并行章节）。

### 3.4 同步 vs 异步

默认 `op=` 同步阻塞直到完成；`async_op=True` 返回 `Work` 句柄，可与计算重叠——这是 [[overlap strategy]] 的基础。CPU 的 gloo 后端有些 op 不支持异步。

### 3.5 与 `torchrun` 的关系

`torchrun`（旧 `python -m torch.distributed.run`）负责**启动 N 个进程、注入 env vars、容错重启**。你只需写一个普通训练脚本，用 `torchrun --nproc_per_node=8 train.py` 启动，脚本里 `dist.init_process_group(backend="nccl", init_method="env://")` 即可。`torchrun` 是 `torch.distributed` 的标准启动器，取代了手写 `spawn`/`mpirun`。

### 3.6 销毁

`dist.destroy_process_group()` 在训练结束清理。进程退出前调用更干净。

## 4. 数学原理 / 公式

### all-reduce 语义

N 个 rank 各持张量 $G_r$（同形状），all-reduce(SUM) 后每个 rank 得：

$$
G'_r = \sum_{i=0}^{N-1} G_i,\quad \forall r
$$

若用 mean：$G'_r=\frac{1}{N}\sum_i G_i$。DDP 把各 rank 的梯度 all-reduce(MEAN) 后，所有 rank 拿到一致的平均梯度，各自更新 → 参数保持一致。这是数据并行的数学根基。

### ring all-reduce 带宽成本（NCCL 默认）

N 个 rank、张量大小 $M$ 字节的 ring all-reduce：通信量 $2(N-1)M/N$ per rank，总通信量 $\approx 2M$（与 N 无关，比 naive 的 $(N-1)M$ 树形更优）。这是大模型训练能用 all-reduce 的关键——详见 [[NCCL backend]]。

## 5. 代码示例

```python
# train.py —— 用 torchrun --nproc_per_node=8 train.py 启动
import os, torch
import torch.distributed as dist
import torch.nn as nn

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    local_rank = setup()
    rank, world = dist.get_rank(), dist.get_world_size()
    if rank == 0: print(f"world_size={world}")

    t = torch.ones(8, device=f"cuda:{local_rank}") * (rank + 1)
    # all-reduce SUM：每个 rank 拿到 1+2+...+world
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"rank {rank}: sum={t[0].item()}")   # 例 world=8 -> 36

    # all-gather：拼各 rank 的向量
    parts = [torch.zeros(8, device=t.device) for _ in range(world)]
    dist.all_gather(parts, t)
    print(f"rank {rank}: gathered[1][0]={parts[1][0].item()}")

    # barrier：同步
    dist.barrier()
    cleanup()

if __name__ == "__main__":
    main()
```

> [!tip] 单机多卡的最小启动命令
> `torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py`
> `--rdzv_backend=c10d`（旧 `static`）用于静态已知 world_size；弹性训练用 `--rdzv_backend=c10d`+`--nnodes`。多机用 `--rdzv_backend=c10d --rdzv_endpoint=<head_ip>:port --nnodes=N`。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[NCCL backend]]（实际传输库）、[[process group]]（通信的"频道"）、[[rank与world size]]（身份与规模）。
- **下游（应用）**: [[DDP]]（梯度 all-reduce）、[[FSDP]]（参数/梯度/优化器状态分片，靠 all-gather/reduce-scatter）、[[Pipeline Parallel]]（点对点 send/recv）、[[torchrun]]（启动器）、[[Ray与分布式调度]]（Ray 调 PyTorch actor 时也用 dist 通信）、[[weight sync mechanism]]（训推分离的权重广播靠 broadcast）。
- **对比 / 易混**:
  - `torch.distributed` vs `nn.DataParallel`：前者多进程、后者单进程多线程（弃用）。
  - `dist` vs Ray：dist 是 GPU 集合通信原语，Ray 是通用分布式调度框架（任务/actor），二者正交，常组合（Ray 调起多个 dist 进程组）。
  - `all_reduce` vs `reduce`：前者人人拿到结果，后者只有 root 拿到。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **不 `set_device(local_rank)` 就开 cuda 操作** → 默认 cuda:0 上多进程抢卡，OOM/错卡。每个进程必须先绑定自己的 GPU。
> 2. **`init_process_group` 前就用 cuda** → NCCL 后端要求先 set device 再 init；顺序反了会报错或挂起。
> 3. **`world_size`/`rank` 手写错** → 用 `torchrun` 注入 env vars，别自己 hardcode。
> 4. **CPU 用 nccl / GPU 用 gloo** → nccl 只支持 GPU；gloo GPU all-reduce 慢且功能有限。GPU 训练一律 nccl。
> 5. **rank 0 之外的进程做 IO（保存/打印）** → 多进程重复保存覆盖、重复打印刷屏；约定**只在 rank 0** 做主 IO，见 [[rank与world size]]。
> 6. **collective 不对称**：只有部分 rank 调 `all_reduce` → 死锁。集体通信要求**所有 rank 都参与同一 op**（或用 sub-ProcessGroup）。
> 7. **`dist.barrier()` 后不等 rank 0 写完文件就读** → barrier 不保证文件系统可见，要加 sleep 或用 `dist.broadcast_object_list` 传完成信号。
> 8. **不调 `destroy_process_group`** → NCCL daemon 残留，下次启动端口冲突；脚本末尾清理。

## 8. 延伸细节

### 8.1 rendezvous 与弹性训练

`torchrun --nnodes=min:max --rdzv_backend=c10d` 支持**节点动态增减**（弹性训练）：节点掉线自动重组 world_size。LLM 集群训练越来越依赖弹性，避免单节点故障整任务失败。实现依赖 `torch.distributed.elastic`。

### 8.2 `ProcessGroup` 与子组

`dist.new_group(ranks=[...])` 创建子 [[process group]]，可在子集内做 collective（如 Pipeline 的相邻 stage 通信、tensor parallel 的组内 all-reduce）。NCCL 会为每个 group 建独立 communicator，注意开销。

### 8.3 与 `torch.distributed.rpc`

`rpc` 框架在 `dist` 之上提供**远程函数调用/RRef**，用于 Pipeline 并行、模型并行参数托管。已被 `torch.distributed.pipelining` 等更新方案逐步替代，但概念仍在。

### 8.4 通信-计算 overlap

高级训练里把 all-reduce 拆成 `reduce_scatter` + 反向计算 + `all_gather`，让通信与反向计算重叠，把通信"藏"在计算时间里——这是 ZeRO-2/3、[[overlap strategy]] 的精髓。需要 `async_op=True` + 异步 Work 句柄管理。

### 8.5 调试：NCCL_DEBUG

`export NCCL_DEBUG=INFO` 打印 NCCL 拓扑探测、ring 建立、P2P 支持情况；`NCCL_DEBUG_SUBSYS=ALL` 更细。排查"卡死"和"慢"的第一手段，见 [[NCCL backend]] §8。

---
相关: [[分布式基础]]、[[NCCL backend]]、[[process group]]、[[rank与world size]]、[[DDP]]、[[FSDP]]、[[torchrun]]
