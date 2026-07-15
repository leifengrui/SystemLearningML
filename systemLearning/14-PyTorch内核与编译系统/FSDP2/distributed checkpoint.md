# distributed checkpoint

> **所属章节**: [[FSDP2]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: distributed checkpoint / DCP / DTensor checkpoint / 跨拓扑 checkpoint / resharding checkpoint / 分片检查点
> **难度**: 高（需懂 [[FSDP2]] + [[DTensor]] + [[DeviceMesh]]）


## 1. 一句话定义

**distributed checkpoint（DCP）** 是 PyTorch 2.x 的**分片检查点机制**——保存时每 rank 只写自己的 **local tensor**（DTensor 的物理片），文件名按 rank 与 placement 编码，并写入 **metadata**（DTensor 的 placement + [[DeviceMesh]] 拓扑 + 各 rank 的 offsets），**不聚合完整 tensor**（省显存/省通信）。加载时据 metadata 把 local 片重组为 DTensor，**支持跨拓扑 resharding**——加载的 mesh 与保存时不同（如保存 8 卡 shard、加载到 16 卡 shard，或 shard→replicate），DCP 自动用 DTensor 的 `redistribute` 插 all-gather/reduce-scatter/all-to-all 转换 placement 到目标拓扑。它取代 FSDP1 的"先 unflat 再全 rank gather 再存"流程（FSDP1 需 rank 0 聚合完整、显存爆、跨拓扑难）。是 FSDP2 的 per-param sharding 的 checkpoint 配套，也是 HF ↔ FSDP2 ↔ Megatron 互转的基础（见 [[checkpoint转换]]）。

> [!note] 三句话定位
> - **是什么**：每 rank 存 local tensor + metadata（placement+mesh），加载时跨拓扑 resharding。
> - **为什么**：FSDP1 的 unflat+gather 聚合显存爆、跨拓扑难；DCP per-rank 分散存、metadata 驱动 resharding。
> - **与 [[FSDP2]] 关系**：FSDP2 的 param 是 DTensor，DCP 是其 checkpoint 配套（存 DTensor 的 local + placement+mesh）。FSDP1 的 checkpoint 需 unflat，FSDP2 用 DCP 原生。


## 2. 为什么需要它（动机与背景）

### 2.1 FSDP1 checkpoint 的痛点

[[Fully Sharded Data Parallel|FSDP1]] 的 checkpoint：存时需 `unshard`（all-gather 合片完整 weight）再存，或存 flat param（加载需 reflat）。痛点：
- **显存爆**：unshard 合片完整 weight，显存峰值 = 完整模型大小（若 layer-wise unshard 缓解但仍大）。
- **跨拓扑难**：保存 8 卡，加载到 16 卡，flat param 的 shard 划分不同，需手动 resharding 逻辑。
- **per-param 边界丢**：flat param 无 per-param name/shape，加载需 unflat 映射回原 param（易错）。

### 2.2 DCP 的 per-rank 分散存

DCP：每 rank 只存自己的 local tensor（DTensor 的物理片），文件按 rank+offset 编码（`<rank>-<shard0>-<shard1>.pt`）。无聚合（不需 all-gather 完整），显存省（每 rank 只写自己片）。metadata 记 placement + mesh + 各 rank offsets。

### 2.3 metadata 驱动的 resharding

加载时据 metadata 把 local 片重组为 DTensor，再据目标 mesh（可不同）`redistribute` 到目标 placement。自动插 all-gather/reduce-scatter/all-to-all 转换。**跨拓扑**（8→16 卡、shard→replicate）无需手写。

### 2.4 per-param 边界保留

DCP 按 param name 存（per-param 文件或按 rank 分组），保留 per-param 边界（vs FSDP1 flat）。加载按 name 映射，清晰。

### 2.5 跨框架互转基础

DCP 的 metadata（placement+mesh）是框架无关的。HF checkpoint（replicate）→ DCP（加 placement）→ FSDP2 shard，或 → Megatron TP shard，转换在 placement 层（见 [[checkpoint转换]]）。DCP 是互转的中间格式。

### 2.6 与 torch.compile 的兼容

FSDP2 + compile 的模型，param 是 DTensor，DCP 直接存 DTensor（compile 不影响 checkpoint，因 compile 不改 param storage）。FSDP1 + compile 的 flat param checkpoint 复杂。


## 3. 核心概念详解

### 3.1 保存：per-rank local + metadata

```python
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint import FileSystemWriter

# FSDP2 模型, param 是 DTensor
writer = FileSystemWriter("/ckpt")
DCP.save(model.state_dict(), writer)  # 每 rank 存自己的 local 片
# 文件: /ckpt/0-0-0.pt (rank 0 的片), /ckpt/1-0-0.pt (rank 1), ...
# metadata: /ckpt/.metadata (placement + mesh + offsets)
```

每 rank 只写自己的 local tensor（DTensor 的 `.to_local()`），不聚合。metadata 记每 param 的 placement（`Shard(0)` 等）+ mesh 拓扑（`(8,)`）+ 各 rank 的 local shape/offsets。

### 3.2 加载：metadata 重组 + resharding

```python
# 目标 mesh 可与保存时不同 (如 8->16 卡)
mesh_new = init_device_mesh("cuda", (16,), mesh_dim_names=("fsdp",))
# DCP.load 据 metadata 重组 DTensor, 再 redistribute 到目标 mesh/placement
DCP.load(model.state_dict(), writer)  # 自动 resharding 到当前 mesh
```

加载流程：据 metadata 知每 param 的原 placement + 原 mesh + 各 rank local 片 → 各 rank 读自己对应的片 → 重组为 DTensor（原 placement）→ 若当前 mesh 不同，`redistribute` 到目标 placement（插 all-gather/reduce-scatter/all-to-all）。

### 3.3 跨拓扑 resharding

| 保存拓扑 | 加载拓扑 | resharding 通信 |
|---|---|---|
| 8 卡 shard(0) | 16 卡 shard(0) | all-gather 合（8 片）→ shard 切 16 片 |
| 8 卡 shard(0) | 8 卡 replicate | all-gather 合（完整） |
| 8 卡 replicate | 16 卡 shard(0) | shard 切（无合，每 rank 取自己片） |
| 8 卡 shard(0) | 8 卡 shard(1) | all-to-all 重排维 |

DCP 据 metadata + 目标 mesh 自动选通信。无需手写。

### 3.4 metadata 的内容

```
.metadata 文件 (JSON):
{
  "param_table": [
    {
      "name": "layer.0.weight",
      "placement": ["Shard(dim=0)"],
      "mesh": {"shape": [8], "dim_names": ["fsdp"]},
      "shards": [
        {"rank": 0, "offsets": [0], "shape": [96, 768]},
        {"rank": 1, "offsets": [96], "shape": [96, 768]},
        ...
      ]
    },
    ...
  ]
}
```

每 param：placement、mesh 拓扑、各 rank 的 offsets + local shape。加载据此定位片与重组。

### 3.5 与 FSDP1 checkpoint 的对比

| 维度 | FSDP1 checkpoint | DCP（FSDP2） |
|---|---|---|
| **存储** | unflat 后聚合 或 flat param | per-rank local 片 |
| **显存峰值** | 高（合片完整） | 低（per-rank 片） |
| **跨拓扑** | 手动 resharding | 自动（metadata+DTensor） |
| **per-param 边界** | flat 丢 | 保留（按 name） |
| **DTensor** | 不感知 | 原生 |

### 3.6 与 state_dict 分片的关系

[[state_dict分片]] 是 state_dict 级的 sharding 格式（`SHARDED_STATE_DICT` vs `STATE_DICT`）。DCP 是实现机制（per-rank 文件 + metadata）。`SHARDED_STATE_DICT` 用 DCP 存分片，`STATE_DICT` 用普通 torch.save 存完整。详见 [[state_dict分片]]。

### 3.7 与 checkpoint 转换的关系

[[checkpoint转换]] 把 HF（replicate）→ FSDP2（shard）或 Megatron（TP shard）互转。中间经 DCP：HF checkpoint 加 placement metadata 转 DCP，FSDP2 加载 DCP（resharding 到目标 shard）。DCP 是互转枢纽。详见 [[checkpoint转换]]。

### 3.8 异步保存（background checkpoint）

DCP 支持异步保存（`thread` mode）：训练流继续，后台线程写盘。减 checkpoint 对训练的阻塞。大模型 checkpoint 几分钟，异步让训练不卡。见 [[checkpointing策略]] 的 async。

### 3.9 增量 checkpoint

DCP 可只存变化的 param（如 LoRA 微调只存 adapter），未变的复用旧 checkpoint。减存储与时间。详见 [[checkpointing策略]]。

### 3.10 多 writer 后端

DCP 的 writer 可换：`FileSystemWriter`（本地 FS）、`TorchSaveWriter`（torch.save）、`S3Writer`（S3）、HDFSWriter（HDFS）。适配不同存储。见 [[并行文件系统]] 与 [[checkpointing策略]] 的存储层。


## 4. 数学原理 / 公式

### 4.1 存储量

每 rank 存自己的 local 片。若 param shard 沿某维 $d$ 个 rank，每 rank local size $= N/d$（$N$ = param 总 size）。总存储 $\sum_r N/d = N$（所有 rank 合 = 完整，与聚合存等）。但 per-rank 分散（不占单 rank 显存聚合）。

### 4.2 加载 resharding 通信

8 卡 shard → 16 卡 shard：先 all-gather 合（通信 $\propto N$）再 shard 切 16。通信 $\propto N$。8→8 replicate：all-gather 合 $\propto N$。replicate → shard：无合（每 rank 取片）。通信量取决于转换方向。

### 4.3 metadata 大小

metadata 是 JSON，记每 param 的 placement + mesh + 各 rank offsets/shape。大模型百万 param 时 metadata 可能 MB 级。但相对 tensor 数据（GB）可忽略。所有 rank 共享一份 metadata（broadcast 或每 rank 各存）。

### 4.4 写入带宽

per-rank 写自己的片，并行写（多 rank 多文件并发）。带宽 = 单 rank 写带宽 × rank 数（若 FS 支持并发，见 [[并行文件系统]]）。比 rank 0 聚合串行写快。


## 5. 代码示例（可选）

### 5.1 DCP 基本保存/加载

```python
import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("fsdp",))
model = MyModel().cuda()
for p in model.parameters():
    p.data = distribute_tensor(p.data, mesh, [Shard(0)])  # FSDP2 shard
model = FSDP(model, mesh=mesh)

# 保存 (每 rank 存 local 片)
writer = FileSystemWriter("/ckpt")
DCP.save(model.state_dict(), writer)
# 文件: /ckpt/0-0-0.pt, 1-0-0.pt, ..., /ckpt/.metadata

# 加载 (跨拓扑: 8 -> 16 卡)
mesh_new = init_device_mesh("cuda", (16,), mesh_dim_names=("fsdp",))
model_new = MyModel().cuda()
for p in model_new.parameters():
    p.data = distribute_tensor(p.data, mesh_new, [Shard(0)])  # 目标 16 shard
model_new = FSDP(model_new, mesh=mesh_new)
DCP.load(model_new.state_dict(), writer)  # 自动 resharding 8->16
```

### 5.2 异步保存

```python
# 异步: 训练继续, 后台写盘
DCP.save(state_dict, writer, thread=True)  # 后台线程
# 或用 dcp.async_save
```

### 5.3 不同 writer 后端

```python
from torch.distributed.checkpoint import FileSystemWriter
# writer 可换: S3Writer, HDFSWriter, TorchSaveWriter
writer = FileSystemWriter("/ckpt")  # 本地 FS
# writer = S3Writer("s3://bucket/ckpt")  # S3
DCP.save(state_dict, writer)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[FSDP2]]（per-param DTensor 的 checkpoint 配套）、[[DTensor]]（placement+mesh 作 metadata）、[[DeviceMesh]]（拓扑信息）、[[AllGather]]/[[reduce-scatter]]/[[all-to-all]]（resharding 通信）、[[state_dict与load_state_dict]]（state_dict 抽象）。
- **下游（应用）**: [[checkpoint转换]]（HF↔FSDP2↔Megatron 互转枢纽）、[[checkpointing策略]]（async/incremental 的实现）、[[并行文件系统]]（多 writer 并发写）、[[FSDP2]]（其 checkpoint 机制）。
- **对比 / 易混**:
  - **DCP vs `torch.save`/`state_dict与load_state_dict`**：`torch.save` 存完整 tensor（聚合，rank 0 存），无 placement/mesh metadata，跨拓扑需手写 resharding。DCP per-rank 分散存 + metadata 驱动 resharding。FSDP1 用 `torch.save` 存 flat/unflat，FSDP2 用 DCP。
  - **DCP vs FSDP1 checkpoint**：见 3.5，per-rank 片+metadata vs unflat 聚合。
  - **DCP vs [[gradient checkpointing]]**：DCP 是**权重 checkpoint**（存 model param 到盘，训练中断恢复），gradient checkpointing 是**激活重算**（反向重算前向激活省显存）。两者完全不同，勿混。gradient checkpointing 存的是中间激活（反向重算），DCP 存的是最终权重（恢复训练）。
  - **DCP `SHARDED_STATE_DICT` vs `STATE_DICT`**：见 [[state_dict分片]]，前者用 DCP 存分片，后者普通 torch.save 存完整。


## 7. 常见误区与易错点

> [!warning] 误区 1：DCP 是新存储格式
> DCP 存的还是 `.pt`（torch 序列化），只是 per-rank 分散存 + 加 metadata（placement+mesh+offsets）。存储格式不变，变的是组织方式（分散+metadata）与加载（resharding）。

> [!warning] 误区 2：DCP 必须同拓扑加载
> DCP 支持跨拓扑（不同 mesh/-placement）。据 metadata + 目标 mesh 自动 resharding。8→16 卡、shard→replicate 都行。是 DCP 的核心优势。

> [!warning] 误区 3：忽视 metadata 一致性
> metadata（placement+mesh+offsets）必须与 local 片对应。若手动改了 local 片没更新 metadata，加载 resharding 会错（offsets 对不上）。DCP 自动维护，勿手改。

> [!warning] 误区 4：DCP 与 gradient checkpointing 混
> DCP 是权重 checkpoint（存 param 到盘，恢复训练）。[[gradient checkpointing]] 是激活重算（反向重算激活省显存）。两者完全不同。勿因都叫"checkpoint"混。

> [!warning] 误区 5：DCP 慢因 per-rank 小文件
> per-rank 小文件多，FS metadata 开销大。但并行写（多 rank 并发）+ async 减阻塞。高性能 FS（[[并行文件系统]]）支持。大模型 checkpoint 几分钟，async 让训练不卡。

> [!warning] 误区 6：DCP 加载不通信
> 跨拓扑 resharding 需通信（all-gather/reduce-scatter/all-to-all）。同拓扑加载（mesh 相同）才无通信（每 rank 读自己片直接用）。8→16 卡需 all-gather 合再切。

> [!warning] 误区 7：DCP 是 stable API
> DCP 是 PyTorch 2.x 的演进 API（`torch.distributed.checkpoint`），API 可能变（如 `dcp` → `DCP`）。生产用前查最新文档。详见 PyTorch DCP 文档。


## 8. 延伸细节

### 8.1 DCP 的优化的写入

per-rank 小文件多，FS 元数据开销大。优化：合并多 param 到一文件（per-rank 一大文件，内部 offset）、用高并发 FS（[[并行文件系统]] 如 Lustre/GPFS）、async 写。详见 [[checkpointing策略]] 的存储优化。

### 8.2 DCP 与 [[checkpointing策略]]

DCP 是 checkpoint 策略的实现机制。策略层（何时存、存什么、异步/增量）由 [[checkpointing策略]] 定，DCP 是底层存取。如 async checkpoint 用 DCP 的 thread mode，incremental checkpoint 用 DCP 的 per-param 选择存。

### 8.3 DCP 与 [[overlap strategy]]

DCP 的异步保存可与训练 overlap（后台线程写盘，训练继续）。是 overlap 在 checkpoint 场景的应用。大模型 checkpoint 几分钟，overlap 减训练空闲。

### 8.4 DCP 的跨框架互转

HF checkpoint（replicate，safetensors）→ 加 placement metadata 转 DCP → FSDP2 加载（resharding 到 shard）。或 → Megatron TP shard。转换在 placement 层（见 [[checkpoint转换]]）。DCP 是互转枢纽。

### 8.5 内容来源

DCP 整理自 PyTorch dev docs "Distributed Checkpoint"、`torch/distributed/checkpoint/` 源码、RFC "DCP"。跨拓扑 resharding 见 DTensor 的 `redistribute`。与 FSDP1 checkpoint 的对比见 FSDP2 design doc。与 checkpoint 转换的关系见 [[checkpoint转换]]。

---
相关: [[FSDP2]] | [[DTensor]] | [[DeviceMesh]] | [[state_dict与load_state_dict]] | [[state_dict分片]] | [[checkpoint转换]] | [[checkpointing策略]] | [[并行文件系统]] | [[overlap strategy]] | [[AllGather]] | [[reduce-scatter]] | [[all-to-all]] | [[gradient checkpointing]] | [[Fully Sharded Data Parallel]]
