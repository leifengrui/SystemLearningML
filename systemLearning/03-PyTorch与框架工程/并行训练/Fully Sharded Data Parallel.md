---
aliases: [FSDP]
---

## 一句话

FSDP 将**参数、梯度、优化器状态**按 rank 分片；计算某个 FSDP 单元时才通过 `all-gather` 临时还原参数，反向后用 `reduce-scatter` 保留本 rank 的梯度分片。它用更多通信换取更低的常驻显存。

$$\text{常驻模型状态: }O(P)\ \rightarrow\ O(P/N)$$

其中 $P$ 为模型状态规模，$N$ 为数据并行 world size。峰值仍受**单个 FSDP 单元临时聚合的全量参数**和激活显存限制。

## 数据流：一层（或一个 Transformer block）发生什么

```text
每个 rank：param shard + grad shard + optimizer-state shard

前向：  all-gather 参数分片 -> 计算该单元 -> 释放临时全量参数
反向：  （需要时）all-gather 参数分片 -> 反向计算
        reduce-scatter 全量梯度 -> 仅保留本 rank 的梯度分片
更新：  每个 rank 只更新自己的参数及优化器状态分片
```

核心直觉：**存储是分片的，单元计算时参数是临时完整的。** 因此 FSDP 是数据并行维度的显存优化，不是 Tensor Parallel 那种“每张卡只计算权重的一部分”。

## 与 DDP / ZeRO / TP 的关系

| 方案 | 参数、梯度、状态 | 主要通信 | 适用判断 |
|---|---|---|---|
| DDP | 每卡完整副本 | 梯度 all-reduce | 模型及优化器状态能放进单卡 |
| ZeRO-2 / `SHARD_GRAD_OP` | 分片梯度、优化器状态 | 梯度通信 | 显存不够但不想频繁聚合参数 |
| FSDP / ZeRO-3 / `FULL_SHARD` | 三者均分片 | 参数 all-gather + 梯度 reduce-scatter | 大模型训练的常用选择 |
| TP | 权重和计算切到模型维度 | 层内激活通信 | 单层也过大；通常用在节点内 |

实际大规模训练常组合：**节点内 TP，跨节点 FSDP/ZeRO-3，必要时再加 PP**。

## 最小配置与实践重点

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                   reduce_dtype=torch.float32),
    device_id=local_rank,
)
```

- **包裹粒度**：通常以 Transformer block 为单元。整个模型一个单元会在聚合时 OOM；细到每个 Linear 会使通信调度开销过大。
- **激活显存**：FSDP 不会自动减少激活，长序列通常仍要配 [[gradient checkpointing]]。
- **性能**：通过 backward prefetch 和通信/计算 overlap 隐藏部分 all-gather；`limit_all_gathers` 用来限制聚合并发、控制峰值。
- **精度**：训练常用 bf16；数据类型、loss scaling 和通信精度要按模型稳定性验证，不能机械照搬配置。

## 检查点

FSDP 有两类常用状态字典：

- **全量状态字典（full state dict）**：聚合为普通模型可加载的完整权重，方便导出到推理或与非 FSDP 模型互通；保存时有额外显存/CPU 内存压力。
- **分片状态字典（sharded state dict）**：每个 rank 保存自己的分片，保存和恢复更省资源，但恢复环境需要按 FSDP 分片方式处理。

原则：**保存格式和加载方式必须成对设计**；不要假设任意 `state_dict()` 都能直接交给单卡模型或推理引擎。

## 高频误区

1. **“FSDP 让 70B 在 8 张卡上随便训练”**：错误。模型状态虽被分片，但 AdamW 状态、激活、临时聚合和通信 buffer 仍很大；所需卡数取决于精度、序列长度、batch、checkpointing 与 offload。
2. **“FSDP 自动解决 OOM”**：错误。它主要削减参数/梯度/优化器状态；激活峰值和包裹粒度仍可能 OOM。
3. **“分片越细越省且越快”**：只对峰值显存部分成立，过细会增加 collectives 与调度开销。
4. **“FSDP 等于 TP”**：错误。FSDP 临时聚合后每卡仍执行完整单元；TP 则将该单元的权重和计算真正切分。
5. **“CPU offload 是默认解法”**：它能继续省显存，但 PCIe/CPU 内存传输常显著降速，应先评估分片粒度、精度和 activation checkpointing。

## 面试速答

> FSDP 是 PyTorch 的全分片数据并行，接近 ZeRO-3：每卡只常驻参数、梯度、优化器状态的一个分片；执行某个 block 时 all-gather 参数，反向后 reduce-scatter 梯度。它将常驻模型状态从 $O(P)$ 降至 $O(P/N)$，代价是更频繁的通信。工程上最关键的是 block 级包裹、activation checkpointing、通信 overlap，以及与检查点格式匹配的保存/加载。

相关：[[Distributed Data Parallel]]、[[ZeRO]]、[[Tensor Parallel]]、[[gradient checkpointing]]、[[3D parallelism]]、[[显存结构]]
