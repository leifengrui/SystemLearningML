---
aliases: [FSDP]
---

# Fully Sharded Data Parallel

> **所属章节**: [[并行训练]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 高（LLM 大模型训练主力）

## 1. 一句话定义

**Fully Sharded Data Parallel（FSDP）** 是 PyTorch 原生的**全分片数据并行**：在 [[Distributed Data Parallel]] 的"数据并行 + 梯度 all-reduce"基础上，进一步把**模型参数、梯度、优化器状态都按 rank 分片**（每 rank 只存自己那 1/N），在需要时用 `all-gather` 临时聚合全量参数用于前向/反向，算完立即释放——从而把单 rank 显存从 $O(P)$ 降到 $O(P/N)$，使**单卡装不下的大模型（如 70B、175B）能在普通 8 卡/多机上训练**。它是 ZeRO-3 思想的 PyTorch 实现，是当前 LLM 预训练/微调的主力方案之一。

> [!note] FSDP vs ZeRO
> FSDP ≈ DeepSpeed **ZeRO-3**（参数+梯度+优化器状态全分片）。ZeRO-1 只分片优化器状态、ZeRO-2 再加分片梯度。FSDP 默认全分片（ZeRO-3），也可配 `ShardingStrategy.SHARD_GRAD_OP` 退化到 ZeRO-2、`NO_SHARD` 退化到 DDP。三者是一条显存-通信权衡的轴。

> [!note] 已展开为独立笔记：[[Megatron-LM]]
> FSDP 与 Megatron 是 LLM 大模型训练省显存的**两条主流路线**，对比着学才完整。FSDP/ZeRO-3 走"分片存储 + all-gather 临时聚合"，每层仍全量参数参与计算，通信量与参数同阶、可跨节点扩展；Megatron 走"真·切分权重，每卡只算一片"，每层只在边界 AllReduce 激活，通信量与激活同阶、不随模型变大但要求 NVLink。两者正交可叠加（TP 节点内 + ZeRO-3 跨节点）是万亿参数训练的标准配方。完整对比表、列切/行切数学推导、MLP 夹心组合、PP schedule、并行交叉熵、selective recombination 详见 [[Megatron-LM]]。

## 2. 为什么需要它（动机与背景）

[[Distributed Data Parallel]] 假设每 rank 装得下：参数 $4P$ + 梯度 $4P$ + AdamW 状态 $8P$ + 激活 ≈ $16P$ per rank。对 70B 模型（$P=70\times10^9$），fp32 下仅参数就 $280$ GB，单卡（80GB）根本装不下，DDP 直接 OOM。

FSDP 的洞察：**每 rank 其实不需要时刻持有全量参数**。前向算某一层时才需要那一层的全量参数，算完就能丢。于是把参数按 rank 分片：

- 静态：每 rank 只存**自己分片**的参数 + 自己分片的梯度 + 自己分片的优化器状态（各 $P/N$）。
- 前向到层 $l$：`all-gather` 把所有 rank 的层 $l$ 分片聚成全量 → 算前向 → 释放全量，只留自己分片。
- 反向到层 $l$：再 `all-gather` 聚全量 → 算梯度 → 各 rank 只保留自己分片的梯度 → `reduce-scatter` 把梯度分片 → 释放全量。
- `optimizer.step()`：每 rank 只更新自己的参数分片。

单 rank 峰值显存 ≈ $\frac{P+G+O}{N}$ + 单层全量参数（短暂）+ 激活 → 随 $N$ 线性下降，70B 在 8 卡可训。代价是**更多通信**（每层都要 all-gather + reduce-scatter），用通信换显存。

## 3. 核心概念详解

### 3.1 三级分片（ZeRO 的轴）

| 级别 | 分片对象 | 单 rank 显存（≈） | 通信量/step | 对应 FSDP 策略 |
|---|---|---|---|---|
| DDP（不分片） | 无 | $16P$ | $2M$（梯度 all-reduce） | `ShardingStrategy.NO_SHARD` |
| ZeRO-1 | 优化器状态 | $8P + P/N$ | $\approx 2M$ | （PyTorch 历史API） |
| ZeRO-2 | + 梯度 | $8P/N + \dots$ | $\approx 2M$ | `SHARD_GRAD_OP` |
| ZeRO-3（FSDP 默认） | + 参数 | $16P/N$ + 单层全量 | $\approx 4M$（参数 gather+梯度 reduce-scatter） | `FULL_SHARD` |

### 3.2 工作流（FSDP 默认 FULL_SHARD）

```
每 rank 持有: param_shard (1/N 参数), 同理 grad_shard, opt_state_shard

前向遍历层 l:
    all-gather(param_shard over ranks) -> 全量 param_l   (短暂)
    out = layer_l(x, param_l)
    free(全量 param_l)                          # 只留自己分片

反向遍历层 l:
    all-gather(param_shard) -> 全量 param_l      (再次聚合, 反向需全量)
    grad_l = backward(...)
    reduce-scatter(grad_l) -> grad_shard          # 各 rank 只留分片
    free(全量 param_l, 全量 grad_l)

optimizer.step():  各 rank 只更新自己的 param_shard/opt_state_shard
```

### 3.3 关键 API

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload

model = MyLLM().cuda()
model = FSDP(model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                    reduce_dtype=torch.float32,
                                    buffer_dtype=torch.bfloat16),
    cpu_offload=CPUOffload(offload_params=False),     # 可选：参数卸载到CPU
    device_id=local_rank)
```

- `ShardingStrategy`：`FULL_SHARD`(ZeRO-3)/`SHARD_GRAD_OP`(ZeRO-2)/`NO_SHARD`(DDP)。
- `MixedPrecision`：分片存 bf16、reduce 用 fp32 保数值。
- `CPUOffload`：参数/优化器状态卸到 CPU 内存，进一步省显存（换速度）。
- `backward_prefetch`：反向时预取下一层参数，overlap 通信。
- `forward_prefetch`：前向时预取下一层（实验）。
- `limit_all_gathers`：限制并发 all-gather 数防显存峰值。

### 3.4 分片粒度：FSDP 的"嵌套"

FSDP 按 Module 包裹：外层 FSDP 包整个模型、内层 FSDP 包每个 Transformer block。分片粒度由 FSDP 单元的范围决定：

- **大单元**（包整个模型）：一次 all-gather 全量参数，显存峰值高但通信次数少。
- **小单元**（每层一个 FSDP）：每层独立 all-gather，峰值低但通信次数多。
- 实践：**按 block 包 FSDP**（每 block 一个 FSDP 单元），平衡峰值与通信。

### 3.5 激活重算（activation checkpointing）

FSDP 常配 `activation_checkpointing`：前向不存激活，反向时重算，省激活显存（换一次前向算力）。LLM 长序列训练必开。详见 [[gradient checkpointing]]。

### 3.6 保存/加载：`FULL_STATE_DICT`

FSDP 下每 rank 只有分片参数，直接 `state_dict()` 得到的是分片 shape。要存成跨框架兼容的全量权重，用：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True)):
    sd = model.state_dict()   # rank 0 上 gather 成全量并卸到CPU
    if rank == 0: torch.save(sd, "fsdp_full.pt")
```

也可用 `SHARDED_STATE_DICT` 存分片（每 rank 存自己那片，省显存），加载时各自灌回。详见 [[state_dict与load_state_dict]] §8.1。

## 4. 数学原理 / 公式

### 显存账（fp32 + AdamW + ZeRO-3，忽略激活）

| 项 | DDP per rank | FSDP per rank |
|---|---|---|
| 参数 | $4P$ | $4P/N$ |
| 梯度 | $4P$ | $4P/N$ |
| Adam $m,v$ | $8P$ | $8P/N$ |
| **小计** | $16P$ | $16P/N$ |
| 前向 all-gather 全量（短暂） | — | $\le 4P$（一层或一单元） |
| 激活 | 见 [[显存结构]] | 同（可配 ckpt 降） |

FSDP 把"持续占用"从 $16P$ 降到 $16P/N$，峰值由"单 FSDP 单元的全量参数"决定（可调粒度控制）。70B、$N=8$：$16P/N = 16\times280\text{GB}/8 = 560$GB？不，$P$ 此处指参数量，70B 参数 fp32 = 280GB，$16P$ 含 states = $16\times(70\times4)$B... 用经验：70B bf16 参数 140GB + AdamW fp32 states 560GB + 梯度 bf16 140GB ≈ 840GB，分到 8 卡约 105GB/卡（仍超 80GB）→ 需更多卡或 CPU offload 或 ZeRO-3 + activation ckpt。所以 70B 训练通常 16~64 卡起步。

### 通信量

FSDP 每 step 每层：all-gather 参数（$\approx M_l$）+ reduce-scatter 梯度（$\approx M_l$），合计 $\approx 2M$（参数总量 $M$）。加上前向/反向各一次 all-gather 实际 $\approx 4M$。比 DDP 的 $2M$ 多，是"用通信换显存"的代价。

## 5. 代码示例

```python
# train_fsdp.py —— torchrun --nproc_per_node=8 train_fsdp.py
import os, torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

setup()
rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])

class Block(nn.Module):
    def __init__(self, d):
        super().__init__(); self.l = nn.Linear(d, d); self.act = nn.GELU()
    def forward(self, x): return self.act(self.l(x))

class LLM(nn.Module):
    def __init__(self, d, n=12):
        super().__init__(); self.blocks = nn.ModuleList([Block(d) for _ in range(n)])
        self.head = nn.Linear(d, d)
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return self.head(x)

model = LLM(d=2048).cuda()
# 每层一个 FSDP 单元（嵌套：外层包整模型，内层包每 block）
model = FSDP(model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                    reduce_dtype=torch.float32,
                                    buffer_dtype=torch.bfloat16),
    device_id=local_rank)

ds = torch.utils.data.TensorDataset(torch.randn(1024, 2048), torch.randn(1024, 2048))
sampler = DistributedSampler(ds); loader = DataLoader(ds, batch_size=8, sampler=sampler)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(2):
    sampler.set_epoch(epoch)
    for x, y in loader:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    if rank == 0: print(f"epoch {epoch} loss={loss.item():.4f}")

dist.destroy_process_group()
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[Distributed Data Parallel]]（FSDP 是其扩展）、[[torch.distributed]]、[[NCCL backend]]（all-gather/reduce-scatter）、[[optimizer state管理]]（分片的正是它）、[[显存结构]]（FSDP 优化的是它）。
- **下游（应用）**: LLM 预训练/微调主力；可作为 [[3D parallelism]] 的 DP 维度（与 TP/PP 组合）；与 [[gradient checkpointing]] 配省激活。
- **对比 / 易混**:
  - **FSDP vs DDP**：见 §3.1 表，显存 vs 通信的权衡。
  - **FSDP vs ZeRO**：FSDP ≈ ZeRO-3，是 ZeRO 思想的 PyTorch 原生实现；DeepSpeed 的 ZeRO 是独立实现，功能等价。
  - **FSDP vs [[Tensor Parallel]]**：FSDP 是数据并行的显存优化（参数全量参与每层计算，只是分片存储）；TP 是把单层权重切成 N 份各自算部分和再 all-reduce——TP 通信在层内、FSDP 通信在层间。LLM 超大模型常 FSDP+TP 混用。
  - **FSDP vs CPUOffload**：offload 把参数卸到 CPU 内存（进一步省但更慢），FSDP 的可选增强。
  - **FSDP vs [[Megatron-LM]]（TP）**：FSDP 是数据并行维的显存优化（参数全量参与计算，分片存储+all-gather 聚合，通信∝参数、可跨节点）；Megatron TP 是模型维的切分（每卡只算自己那片权重，边界 AllReduce 激活，通信∝激活、不随模型变大但需 NVLink）。两者正交可叠加，万亿参数训练用 TP(节点内)+ZeRO-3(跨节点)。详见 [[Megatron-LM]] §6 对比表。
  - **FULL_SHARD vs SHARD_GRAD_OP**：前者分片参数（ZeRO-3，省显存多通信多），后者只分片梯度（ZeRO-2，省显存少通信少）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **FSDP 后直接 `model.state_dict()` 存全量** → 得到分片 shape，灌回非 FSDP 模型报形状不符；用 `StateDictType.FULL_STATE_DICT` 上下文。
> 2. **分片粒度太粗**（整个模型一个 FSDP）→ all-gather 全量参数峰值高，OOM；按 block 嵌套包。
> 3. **分片粒度太细**（每个 Linear 一个 FSDP）→ 通信次数爆炸，慢；平衡到 block 级。
> 4. **不开 mixed precision** → fp32 全量 all-gather 慢且峰值高；bf16 参数 + fp32 reduce 是标配。
> 5. **不开 activation checkpointing 跑长序列** → 激活显存爆；LLM 长上下文必开。
> 6. **`backward_prefetch` 配错** → 通信不 overlap，性能差；用 `BACKWARD_PRE` 默认值。
> 7. **`limit_all_gathers` 不设** → 并发 all-gather 过多峰值超显存；设较小值（如 4）控峰值。
> 8. **加载旧 DDP ckpt 到 FSDP 模型** → 形状/键对不上；用 `FULL_STATE_DICT` 加载全量再分片，或 `SHARDED_STATE_DICT` 加载分片。
> 9. **以为 FSDP 自动省激活** → 不会，激活要单独配 `activation_checkpointing`。
> 10. **CPU offload 开了不开 pin memory / 不预取** → CPU↔GPU 搬运成瓶颈，慢到不可用。

## 8. 延伸细节

### 8.1 FSDP2（PyTorch 2.x 新版）

FSDP2 重构为基于 `DTensor` 的实现，分片是张量属性而非 Module 包装：`model.to_dtensor(...)` 直接把参数变分片张量，更灵活、与 TP/PP 组合更顺、支持 per-tensor 保存。是未来主流，逐步替代 FSDP1。

### 8.2 与 Tensor Parallel 组合（FSDP+TP）

超大规模（如 405B）单 FSDP 不够，常 FSDP（跨机 DP 维度）+ TP（机内 NVLink 高带宽）+ PP（层间流水）= 3D 并行。FSDP2 与 TP 的 DTensor 统一表示让组合更自然。见 [[3D parallelism]]。

### 8.3 通信-计算 overlap

FSDP 的 `backward_prefetch=BACKWARD_PRE` 在反向算当前层时预取下一层参数，把 all-gather 藏在当前层反向里；`forward_prefetch` 类似。配合 NCCL async stream，能逼近通信隐藏。这是 FSDP 性能调优的核心，见 [[overlap strategy]]。

### 8.4 与推理部署的衔接

训练完用 `FULL_STATE_DICT` 导出全量权重 → 转 safetensors → 给 vLLM 等推理引擎。FSDP 是训练态表示，推理态需要全量不分片权重（除非推理也分片，如 TP 推理）。

### 8.5 显存峰值控制清单

LLM 训练显存 = 参数+梯度+states+激活+通信 buffer+碎片。FSDP 控前三，`activation_checkpointing` 控激活，`limit_all_gathers` 控通信 buffer，`torch.cuda.empty_cache()` + `expandable_segments` 控碎片。组合调到不 OOM 且不浪费是工程经验活。

---
相关: [[并行训练]]、[[Distributed Data Parallel]]、[[Data Parallel]]、[[Megatron-LM]]、[[ZeRO]]、[[optimizer state管理]]、[[显存结构]]、[[gradient checkpointing]]、[[3D parallelism]]、[[Tensor Parallel]]、[[overlap strategy]]
