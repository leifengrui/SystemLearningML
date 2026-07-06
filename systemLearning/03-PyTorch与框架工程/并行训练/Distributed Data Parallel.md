# Distributed Data Parallel

> **所属章节**: [[并行训练]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 中等（LLM 训练默认起点）

## 1. 一句话定义

**Distributed Data Parallel（DDP）** 是 PyTorch 现代多卡数据并行的标准实现：**一进程一 GPU**，每个 rank 持有**完整模型副本**，各自跑不同 batch 的前向+反向，反向时用 **NCCL all-reduce 把各 rank 梯度求平均**，所有 rank 拿到一致梯度后各自 `optimizer.step()` 更新——从而保持参数始终同步。它是 [[Data Parallel]] 范式的高性能版，也是绝大多数多卡训练（CV、中小模型、LLM 单机）的默认起点。

> [!note] DDP vs DP vs FSDP 一句话
> - **DP**（[[Data Parallel]]）：单进程多线程、主卡聚合梯度，弃用。
> - **DDP**：多进程、all-reduce 梯度、各 rank 持全量参数，**主流**。
> - **FSDP**（[[Fully Sharded Data Parallel]]）：DDP 基础上把**参数/梯度/优化器状态也分片**，省显存，LLM 大模型用。

## 2. 为什么需要它（动机与背景）

[[Data Parallel]] 的主卡瓶颈、GIL、不支持跨机使其不可扩展。DDP 用三个改造解决：

1. **多进程**（一进程一 GPU）→ 绕开 GIL，每个 rank 独立 Python；
2. **all-reduce 梯度**（NCCL ring）→ 无主卡瓶颈，通信量与卡数无关（$\approx 2M$），线性扩展到数百卡；
3. **`torchrun` 启动** → 自动注入 env vars、支持跨机、弹性。

DDP 仍假设**每个 rank 装得下完整模型+优化器状态+梯度+激活**。当模型大到一个 rank 装不下（如 70B 全量训练），DDP 不够，要上 [[FSDP]]。所以 DDP 是"单卡能装下模型"场景的默认，FSDP 是"装不下"的扩展。

## 3. 核心概念详解

### 3.1 工作流（DDP）

```
各 rank: model = MyModel().cuda()       # 全量副本
        for batch in loader:            # 各 rank 见不同 batch
            fwd:  logits = model(batch)
            loss  = loss_fn(logits, y)
            bwd:  loss.backward()       # 反向时 DDP 拦截梯度, all-reduce
            opt:  optimizer.step()      # 各 rank 用一致梯度更新 -> 参数保持同步
```

DDP 在 `loss.backward()` 时**拦截每个参数的梯度**，分桶（bucket）异步 all-reduce，使通信与反向计算 overlap。这是它比朴素"反向完再 all-reduce"快的关键。

### 3.2 关键 API

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl", init_method="env://")
torch.cuda.set_device(local_rank)
model = MyModel().cuda()
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
# 之后 model(x) 内部自动处理跨 rank 同步
```

- `device_ids=[local_rank]`：单卡 DDP（每进程一卡）；CPU 训练不传或传 `None`。
- `find_unused_parameters`：若有参数本次前向没用上（条件计算），设 True 否则报错；但会拖慢通信，尽量让前向用全参数。
- `broadcast_buffers`：是否每 forward 前 broadcast buffer（如 BN running stats），默认 True。

### 3.3 梯度分桶与 overlap

DDP 把参数按类型/大小分到若干 **bucket**（默认 25MB）。反向时一个 bucket 内所有梯度算完就立即 all-reduce，与下一 bucket 的反向计算 overlap——把通信"藏"在反向时间里。这是 DDP 高性能的核心技巧，对应 [[overlap strategy]]。

### 3.4 梯度同步的正确性

DDP 在反向时对每个梯度做 all-reduce(MEAN)。因为每个 rank 的 batch 不同、梯度不同，平均后得到的是"全 batch 的平均梯度"（数学上等价于用整个 batch 算一次，见 §4）。更新后各 rank 参数一致。

> [!warning] 数据必须不重叠
> 各 rank 的 batch 必须用 `DistributedSampler` 切分不重叠。若两个 rank 拿到相同样本，all-reduce(MEAN) 会重复计数，梯度不对（等于 effective batch 里该样本权重翻倍）。

### 3.5 与 SyncBatchNorm

默认 DDP 下各 rank 的 BN 用自己 batch 统计（异步 BN）。若要"全 batch 的 BN 统计"，用 `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` 把 BN 换成跨 rank 同步版本。LLM 用 LayerNorm 不涉及，CV 大 batch 训练常用。

### 3.6 rank 0 职责与保存

DDP 下所有 rank 参数一致，**任意 rank 的 checkpoint 都代表整个模型**。约定 rank 0 存：`if dist.get_rank()==0: torch.save(model.module.state_dict(), ...)`。注意 `.module` 取原模型（DDP 包了一层，类似 DP）。

## 4. 数学原理 / 公式

### 数据并行梯度等价性

设 world_size=$N$，各 rank 持相同参数 $\theta$、不同 batch $B_i$（互不重叠且 $|B_i|=B$）。各 rank 算 $\ell_i=\frac1B\sum_{j\in B_i}\ell_j$，梯度 $g_i=\nabla_\theta\ell_i$。all-reduce(MEAN) 后：

$$
\bar g = \frac1N\sum_{i=0}^{N-1} g_i = \frac1{NB}\sum_{j\in \bigcup B_i}\nabla_\theta\ell_j = \nabla_\theta\left(\frac1{NB}\sum_j \ell_j\right)
$$

即 $\bar g$ 等于用整个大 batch $\bigcup B_i$（共 $NB$ 样本）算一次的平均梯度。各 rank 用 $\bar g$ 更新 → 参数保持一致。这是 DDP 数学正确性的根基。

### 通信量

all-reduce 梯度（共 $M$ 字节）每 rank 通信 $\approx 2M$（ring，见 [[NCCL backend]]）。与卡数 $N$ 无关，故线性扩展。对比 [[Data Parallel]] 的 $(N-1)M$ 且主卡瓶颈。

## 5. 代码示例

```python
# train_ddp.py —— torchrun --nproc_per_node=8 train_ddp.py
import os, torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    model = nn.Linear(1000, 10).cuda()
    model = DDP(model, device_ids=[local_rank])

    dataset = torch.utils.data.TensorDataset(torch.randn(1000,1000), torch.randn(1000,10))
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(3):
        sampler.set_epoch(epoch)            # 必须每 epoch 改种子
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()                # DDP 在此 all-reduce 梯度
            opt.step()
        if rank == 0:
            torch.save(model.module.state_dict(), f"ddp_ep{epoch}.pt")
            print(f"epoch {epoch} loss={loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    main()
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[torch.distributed]]、[[NCCL backend]]、[[rank与world size]]、[[nn.Module]]、[[forward与backward]]。
- **下游（应用）**: 单机/多机多卡训练默认方案；LLM 训练中模型装不下时升级到 [[FSDP]]；作为 [[3D parallelism]] 的 DP 维度。
- **对比 / 易混**:
  - **DDP vs DP**：见 [[Data Parallel]] §3.3。
  - **DDP vs FSDP**：DDP 全量复制参数（显存 $O(P)$ per rank），FSDP 分片（显存 $O(P/N)$ per rank）；FSDP 通信更多但显存友好。
  - **DDP vs `torch.distributed`**：DDP 是基于 `torch.distributed` 的高层封装，自动管理梯度同步。
  - **DDP 的 DP 维度 vs [[3D parallelism]] 的 DP**：3D 里的 DP 就是 DDP 在某个子组上的应用。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **模型单卡装不下还硬上 DDP** → OOM；大模型用 [[FSDP]] 或张量并行。
> 2. **`DistributedSampler` 忘 `set_epoch`** → 每轮数据顺序相同，等于没 shuffle。
> 3. **`find_unused_parameters=True` 默认开** → 其实默认 False；有条件计算（部分参数本次没用）时才开，否则报"expects all params in forward"。
> 4. **保存用 `model.state_dict()`** → 键带 `module.` 前缀；用 `model.module.state_dict()`。
> 5. **DDP 包裹顺序错** → 应在 `.cuda()`/`to(device)` **之后**包 DDP；先包后搬设备行为异常。
> 6. **各 rank batch 重叠** → 梯度重复计数；务必 DistributedSampler 切分。
> 7. **所有 rank 都打印 loss** → 刷屏；`if rank==0` 或 all-gather 后打一份。
> 8. **rank 0 保存时其他 rank 已进入下一步** → 加 `dist.barrier()` 等保存完；否则 checkpoint 文件半成品。
> 9. **LLM 用 DDP 训不动** → 因为 optimizer state 占 $8P$ per rank，70B 单卡装不下，必上 FSDP。
> 10. **DDP 下不同 rank 看到不同 loss** → 正常，因为各 rank 的 batch 不同；梯度才一致，loss 本就随 batch 变。

## 8. 延伸细节

### 8.1 梯度分桶大小

DDP 默认 `bucket_cap_mb=25`。大模型可调大（减少 all-reduce 次数）或调小（更细 overlap）。`static_graph=True` 假设每步参与梯度计算的参数集合固定，可优化调度。

### 8.2 DDP 在 LLM 训练的位置

LLM 预训练（如 7B）单机 8 卡能装下时用 DDP；70B+ 装不下用 FSDP。3D 并行里 DP 维度常是最后跨机铺开的维度（节点间带宽够 all-reduce）。见 [[3D parallelism]]。

### 8.3 与 `torch.distributed.algorithms` 的关系

PyTorch 在 DDP 之上提供 `torch.distributed.optim`、`_schedule` 等实验 API，支持 ZeRO-1/2 风格的优化器/梯度分片——是 FSDP 的前身。FSDP 是其集大成者。

### 8.4 DDP 的故障与弹性

单 rank 崩溃会让其他 rank 卡在 all-reduce。`torchrun --nnodes=min:max --rdzv_backend=c10d` 支持弹性重排 rank，但训练逻辑要能处理 world_size 变化（重算 effective batch、重置 sampler）。

### 8.5 调试 DDP

- `TORCH_DISTRIBUTED_DEBUG=DETAIL`：在 all-reduce 处校验各 rank 形状/数值，抓"某 rank 前向用了不同参数"。
- `NCCL_DEBUG=INFO`：看通信建环。
- 单卡等价测试：把 DDP 关掉单卡跑，对比 loss/梯度，排除数据/模型 bug 后再查分布式。

---
相关: [[并行训练]]、[[Data Parallel]]、[[Fully Sharded Data Parallel]]、[[torch.distributed]]、[[NCCL backend]]、[[rank与world size]]、[[overlap strategy]]、[[3D parallelism]]、[[ZeRO]]
