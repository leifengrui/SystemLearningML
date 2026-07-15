# FSDP2

> **所属章节**: [[FSDP2]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: FSDP2 / FSDP v2 / per-parameter sharding / DTensor-based FSDP / fully sharded data parallel v2 / per-param FSDP
> **难度**: 高（需懂 [[DTensor]] + [[Fully Sharded Data Parallel|FSDP1]] + [[ZeRO (DeepSpeed)]]）


## 1. 一句话定义

**FSDP2** 是 PyTorch 2.x 的**下一代 FSDP**——基于 [[DTensor]] 的 **per-parameter sharding**（每个参数独立声明 `Shard` placement 沿 fsdp mesh），前向时 DTensor 的 `redistribute` 自动 all-gather 合片用完整 weight 算、反向后 grad `Partial` 经 all-reduce 再 reduce-scatter 切回 `Shard` 存回。它**取代 FSDP1** 的"flat param"模型（FSDP1 把所有 param 拼成一个 flat param 统一 shard/unshard，需 module wrap、破坏 per-param 边界、难与 torch.compile/DTensor 集成）。FSDP2 的 per-param 设计：**保留 per-param 边界**（每 param 独立 shard，无需 flat 化）、**DTensor 原生**（与 [[DeviceMesh]]/[[distributed checkpoint]]/[[torch.compile]] 自然集成）、**eager-friendly**（无 module wrap，参数本身就是 DTensor）、**composable**（可与 TP/PP/EP 等组合）。是 PyTorch 2.x 分布式训练的主线，与 Megatron 的 TP/PP 趋同融合。

> [!note] 三句话定位
> - **是什么**：基于 DTensor 的 per-param sharding，前向 all-gather 合片、反向 reduce-scatter 切回，无需 flat param。
> - **为什么**：FSDP1 的 flat param 破坏 per-param 边界、难 compile/DTensor 集成；FSDP2 per-param + DTensor 原生，可组合多维并行。
> - **与 [[Fully Sharded Data Parallel|FSDP1]] 关系**：FSDP1 = flat param + module wrap；FSDP2 = per-param DTensor + 无 wrap。FSDP2 是继任者，理念同（ZeRO-3 shard weight）但实现重写。


## 2. 为什么需要它（动机与背景）

### 2.1 FSDP1 的痛点

[[Fully Sharded Data Parallel|FSDP1]] 用 **flat param**：`FullyShardedDataParallel(module)` wrap 后，把 module 所有 param 拼成一个 flat param（`_flat_param`），统一 shard/unshard（前向 all-gather flat、反向 reduce-scatter flat）。痛点：
- **破坏 per-param 边界**：flat 化后单 param 的 shape/name 丢失，checkpoint 需 unflat、custom op 难定位单 param。
- **module wrap 复杂**：要手动 `FullyShardedDataParallel(layer)` 嵌套 wrap，层级 wrap 策略（哪些层 wrap）需调。
- **难 torch.compile**：flat param 的 reshape/manip 在 compile 下易 graph break。
- **难 DTensor 集成**：flat param 不感知 mesh/placement，与 [[DTensor]]/[[DeviceMesh]] 抽象脱节。

### 2.2 FSDP2 的 per-param + DTensor

FSDP2 重写：**每个参数本身就是 DTensor**（placement = `[Shard(weight_dim)]` 沿 fsdp mesh），无需 flat 化、无需 module wrap。前向：DTensor op `redistribute` 到 `[Replicate]`（all-gather 合片，用完整 weight 算）。反向：grad `[Partial]` 经 all-reduce → `[Replicate]` → `redistribute` 到 `[Shard]`（reduce-scatter 切回存）。**per-param 边界保留**（每 param 独立 shard/unshard）、**DTensor 原生**（与 mesh/checkpoint/compile 自然集成）、**无 wrap**（参数本身 DTensor，eager 友好）。

### 2.3 per-param sharding 的灵活

FSDP2 的 per-param 让**不同 param 用不同 sharding 策略**：大 param shard（省显存）、小 param/bias replicate（不值得 shard）、embedding 可单独策略。vs FSDP1 的 flat（所有 param 统一 shard/unshard，一刀切）。灵活匹配 param 特性。

### 2.4 composable 多维并行

FSDP2 的 DTensor + mesh 让它与 TP/PP/EP 自然组合（一个 mesh 多维，placement 沿不同维）。HSDP（nested mesh）用 FSDP2 表达外 replicate + 内 shard。FSDP1 的 flat param 难组合 TP（flat 与 TP 的 sharding 维冲突）。

### 2.5 与 torch.compile 的协同

FSDP2 的 DTensor op 是 registered op，[[TorchDynamo]] 可 trace、[[Inductor]] 可 codegen（all-gather/reduce-scatter 是 graph 的特殊 node，不融但可 compile）。FSDP1 的 flat param reshape 易 graph break。FSDP2 是 compile-friendly 的分布式。

### 2.6 与 Megatron 的趋同

Megatron-LM 的 TP/PP 是手写 module，FSDP2 是 DTensor 声明式。两者趋同（Megatron 也向 DTensor/DeviceMesh 靠拢，见 [[Megatron-LM]]）。PyTorch 主线用 FSDP2，Megatron 极致性能场景仍可用。


## 3. 核心概念详解

### 3.1 per-param sharding 的设置

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.fsdp import FSDP, FullyShardedDataParallel as FSDP2  # v2 API

mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("fsdp",))
model = MyModel()
# per-param: 每个 param 声明 Shard placement
for name, param in model.named_parameters():
    dtensor = distribute_tensor(param.data, mesh, [Shard(0)])  # shard row 沿 fsdp
    param.data = dtensor  # param 本身是 DTensor
# FSDP2 应用 (注册 hook 做 all-gather/reduce-scatter)
model = FSDP(model, mesh=mesh)  # FSDP2 API (per-param)
```

vs FSDP1：

```python
# FSDP1: flat param + module wrap
model = FullyShardedDataParallel(model)  # wrap, 内部 flat 化所有 param
```

### 3.2 前向的 all-gather

FSDP2 前向：param 是 `[Shard(dim)]`，op 需完整 weight → DTensor `redistribute` 到 `[Replicate]`（all-gather 沿 fsdp 维合片）。用完整 weight 算 op。算完释放合片（free 完整副本，省显存，见 [[Fully Sharded Data Parallel]] 的 unshard/reshard）。

### 3.3 反向的 reduce-scatter

反向：grad 是 `[Partial]`（各 rank 算部分和）→ all-reduce 沿 fsdp 维 → `[Replicate]`（完整 grad）→ `redistribute` 到 `[Shard(dim)]`（reduce-scatter 切回 shard 存回 param）。每 param 独立做（per-param）。

### 3.4 与 FSDP1 的对比

| 维度 | FSDP1 | FSDP2 |
|---|---|---|
| **param 模型** | flat param（拼成一个） | per-param（各自 DTensor） |
| **wrap** | module wrap（`FullyShardedDataParallel`） | 无 wrap（param 本身 DTensor） |
| **shard 粒度** | flat（所有 param 统一） | per-param（各自策略） |
| **DTensor** | 不感知 | 原生（placement 声明） |
| **torch.compile** | flat reshape 易 graph break | DTensor op 可 trace |
| **多维组合** | flat 与 TP 冲突 | DTensor 沿不同 mesh 维自然组合 |
| **checkpoint** | 需 unflat | per-param 直接存（[[distributed checkpoint]]） |

### 3.5 per-param 策略的灵活

```python
# 大 param shard, 小 param/bias replicate
for name, p in model.named_parameters():
    if 'bias' in name or p.numel() < 1e6:
        p.data = distribute_tensor(p.data, mesh, [Replicate()])  # 不 shard
    else:
        p.data = distribute_tensor(p.data, mesh, [Shard(0)])   # shard
# FSDP1 的 flat 一刀切, 无此灵活
```

### 3.6 HSDP（Hybrid Sharded DP）

```python
# nested mesh: 外 dp_replicate (跨机), 内 dp_shard (机内)
mesh = init_device_mesh("cuda", (4, 8), mesh_dim_names=("dp_replicate","dp_shard"))
# weight: [Replicate(dp_replicate), Shard(dp_shard)]
for p in model.parameters():
    p.data = distribute_tensor(p.data, mesh, [Replicate("dp_replicate"), Shard("dp_shard")])
# 前向: all-gather 沿 dp_shard (机内合片, NVLink 快)
# 反向: reduce-scatter 沿 dp_shard + all-reduce 沿 dp_replicate (跨机)
# FSDP1 的 HSDP 需复杂 wrap, FSDP2 用 placement 声明
```

HSDP 让机内高带宽（NVLink）做频繁 shard/unshard，跨机低带宽（IB）只做 grad sync。是 [[alpha-beta性能模型]] 的优化。详见 [[DeviceMesh]] 3.4。

### 3.7 与 ZeRO 的关系

[[ZeRO (DeepSpeed)]] 的 ZeRO-3（shard weight+grad+optimizer）理念同 FSDP。FSDP1/FSDP2 是 PyTorch 原生的 ZeRO-3 实现。FSDP2 的 per-param + DTensor 比 DeepSpeed 的 ZeRO 更原生（与 PyTorch 主线集成）。optimizer state 仍需配套（shard 优化器状态，见 [[optimizer state memory]]）。

### 3.8 与 torch.compile 的协同

```python
model = FSDP(model, mesh=mesh)
model = model.compile()  # FSDP2 + compile
# DTensor op 被 Dynamo trace, all-gather/reduce-scatter 是 graph node (不融但可 compile)
# FSDP1 的 flat param reshape 易 graph break, FSDP2 友好
```

### 3.9 与 distributed checkpoint 的协同

FSDP2 的 param 是 DTensor，[[distributed checkpoint]] 直接存 DTensor（local tensor + placement + mesh metadata）。加载时若 mesh 不同，DTensor redistribute 自动 resharding。FSDP1 需 unflat 再存，加载需 reflat。FSDP2 的 checkpoint 更灵活（跨拓扑）。详见 [[distributed checkpoint]]。

### 3.10 unshard/reshard 的显存

FSDP2 前向 all-gather 合片临时占显存（完整 weight 副本）。用完 free（reshard）。显存峰值 = 单层完整 weight（not 全模型，因 layer-wise unshard）。与 FSDP1 同（layer-wise 限制合片显存）。详见 [[Fully Sharded Data Parallel]] 的 unshard 策略与 [[训练显存估计]]。


## 4. 数学原理 / 公式

### 4.1 显存（shard 后）

设模型总 param $N$，fsdp mesh $d$ 个 rank。shard 后每 rank param 显存 $= N/d$（FSDP2 与 FSDP1 同，ZeRO-3 理念）。grad 也 shard，$= N/d$。optimizer state（如 Adam 的 $m, v$）若 shard 也 $= 2N/d$。总每 rank $\approx 4N/d$（weight+grad+2 optimizer）。见 [[训练显存估计]]。

### 4.2 前向 all-gather 通信

每 param 前向 all-gather：通信量 $= \text{param size}$（合片）。layer-wise unshard 只 all-gather 当前层，总通信 $\propto N$（与 FSDP1 同）。all-gather 通信与计算可 overlap（[[overlap strategy]]）减延迟。

### 4.3 反向 reduce-scatter + all-reduce

反向 grad：reduce-scatter 切回 shard（通信 $\propto \text{grad size} = N/d$ per rank）+ HSDP 的 all-reduce 跨机（$\propto N/d$）。通信与 FSDP1 同。

### 4.4 per-param vs flat 的通信

per-param 每 param 独立 all-gather/reduce-scatter（小消息多次）vs flat 一次大消息。小消息多次的 latency 开销稍高（[[alpha-beta性能模型]] 的 $\alpha$ 项），但 per-param 的灵活（不同策略、checkpoint）与 compile 友好补偿。bucketing（合并多个 param 的通信）减 latency。

### 4.5 HSDP 的通信优化

HSDP：机内 NVLink all-gather（高带宽 $\beta_{\text{nvlink}}$）+ 跨机 IB all-reduce（低带宽 $\beta_{\text{ib}}$）。vs 单层 FSDP 全跨机 all-gather（全走 IB）。通信时间 $\propto N/\beta_{\text{nvlink}} + N/\beta_{\text{ib}}$ vs $N/\beta_{\text{ib}}$。$\beta_{\text{nvlink}} \gg \beta_{\text{ib}}$ 故 HSDP 快。见 [[alpha-beta性能模型]]。


## 5. 代码示例（可选）

### 5.1 FSDP2 基本用法

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate
from torch.distributed.fsdp import FSDP

mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("fsdp",))
model = MyModel().cuda()
# per-param sharding
for name, p in model.named_parameters():
    if p.numel() > 1e6:  # 大 param shard
        p.data = distribute_tensor(p.data, mesh, [Shard(0)])
    # 小 param 默认 replicate
model = FSDP(model, mesh=mesh)  # FSDP2 注册 hook

# 前向: DTensor op 自动 all-gather (redistribute to Replicate)
out = model(x)
loss = out.sum()
loss.backward()  # 反向: reduce-scatter + all-reduce
optimizer.step()
```

### 5.2 HSDP

```python
mesh = init_device_mesh("cuda", (4, 8), mesh_dim_names=("dp_replicate","dp_shard"))
for p in model.parameters():
    p.data = distribute_tensor(p.data, mesh, [Replicate("dp_replicate"), Shard("dp_shard")])
model = FSDP(model, mesh=mesh)  # HSDP
# 前向: all-gather 沿 dp_shard (NVLink 机内)
# 反向: reduce-scatter 沿 dp_shard + all-reduce 沿 dp_replicate (IB 跨机)
```

### 5.3 FSDP2 + compile

```python
model = FSDP(model, mesh=mesh)
model = model.compile()  # DTensor op 可 trace
# FSDP1 的 flat param reshape 易 graph break, FSDP2 友好
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[DTensor]]（per-param 的 placement 抽象）、[[DeviceMesh]]（sharding mesh）、[[AllGather]]/[[reduce-scatter]]/[[all-reduce]]（通信原语）、[[Fully Sharded Data Parallel|FSDP1]]（继任对象）、[[ZeRO (DeepSpeed)]]（理念同）。
- **下游（应用）**: [[distributed checkpoint]]（DTensor metadata + resharding）、[[torch.compile]]/[[TorchDynamo]]（DTensor op 可 trace）、HSDP（nested placement）、[[3D parallelism]]（与 TP/PP 组合）、[[训练显存估计]]（shard 后显存）、[[overlap strategy]]（all-gather 与计算 overlap）、[[optimizer state memory]]（shard 优化器状态）。
- **对比 / 易混**:
  - **FSDP2 vs [[Fully Sharded Data Parallel|FSDP1]]**：见 3.4，per-param DTensor vs flat param + wrap。FSDP2 是继任者。
  - **FSDP2 vs [[ZeRO (DeepSpeed)]]**：ZeRO-3 理念同（shard weight），DeepSpeed 是独立实现，FSDP2 是 PyTorch 原生（DTensor）。
  - **FSDP2 vs [[Megatron-LM]] TP**：TP shard weight 沿 tp mesh（all-reduce 后向），FSDP2 shard weight 沿 fsdp mesh（all-gather 前向 + reduce-scatter 后向）。两者可组合（不同 mesh 维）。TP 算力切分（每 rank 算部分），FSDP 显存切分（每 rank 存部分，前向合片算完整）。
  - **FSDP2 vs [[Distributed Data Parallel|DDP]]**：DDP 只 shard batch（grad all-reduce），weight replicate（每 rank 完整 weight，显存不省）。FSDP2 shard weight（显存省）。大模型用 FSDP2，小模型 DDP 够。


## 7. 常见误区与易错点

> [!warning] 误区 1：FSDP2 与 FSDP1 只是 API 变
> 不止。FSDP2 重写为 per-param + DTensor，解决 FSDP1 的 flat param 痛点（破坏边界、难 compile、难多维组合）。理念同（ZeRO-3 shard weight）但实现本质不同。

> [!warning] 误区 2：FSDP2 比 FSDP1 慢
> per-param 的小消息多次通信 latency 稍高，但 bucketing + compile 友好 + 多维组合补偿。总体性能相当或更优（compile 加持、HSDP 优化）。且灵活（per-param 策略、checkpoint）是 FSDP1 无的。

> [!warning] 误区 3：FSDP2 不需 unshard 显存管理
> 前向 all-gather 合片临时占显存（完整 weight 副本）。layer-wise unshard 限峰值，但仍需管理（与 FSDP1 同）。大层 all-gather 显存峰值大。见 [[Fully Sharded Data Parallel]] 的 unshard 策略。

> [!warning] 误区 4：忽视 HSDP 的通信优化
> 单层 FSDP 全 all-gather 跨机（IB 慢）。HSDP 机内 NVLink all-gather + 跨机 all-reduce，利用硬件层级。$\beta_{\text{nvlink}} \gg \beta_{\text{ib}}$ 故 HSDP 快。详见 [[alpha-beta性能模型]]。

> [!warning] 误区 5：FSDP2 的 optimizer 不 shard
> FSDP2 shard weight，但 optimizer state（Adam 的 $m,v$）若不配套 shard，仍每 rank 完整（显存不省）。需 shard 优化器状态（per-param 的 optimizer state 也 DTensor）。见 [[optimizer state memory]]。

> [!warning] 误区 6：FSDP2 + compile 全融合
> DTensor op 可 trace，但 NCCL 通信（all-gather/reduce-scatter）是 graph 特殊 node，Inductor 不融合（单独执行）。compile 融合的是计算 op，通信仍单独。overlap 通信与计算是性能关键（[[overlap strategy]]）。

> [!warning] 误区 7：FSDP2 是 stable API
> FSDP2 是 PyTorch 2.x 的实验性/演进 API（`torch.distributed.fsdp`），API 可能变。生产用前查最新文档与版本。DTensor 也是 experimental。详见 PyTorch FSDP2 RFC。


## 8. 延伸细节

### 8.1 FSDP2 的 bucketing

per-param 的小消息多次通信 latency 高。bucketing 把多个 param 的 all-gather/reduce-scarter 合并成大消息（类似 DDP 的 grad bucket），减 latency。是 per-param 模型补偿 $\alpha$ 项的手段。

### 8.2 FSDP2 与 [[overlap strategy]]

FSDP2 的 all-gather（前向下一层 weight）可与当前层计算 overlap（stream 分离）。reduce-scatter（反向）可与上一层计算 overlap。是 FSDP2 性能优化的关键。FSDP1 的 `forward_prefetch` 是此思路，FSDP2 原生支持。

### 8.3 FSDP2 的 CPU offload

shard weight 仍放 GPU，前向 all-gather 合片用。若显存仍紧，可 CPU offload shard weight（用时 all-gather 到 GPU）。FSDP2 的 per-param 让 offload 策略 per-param（大 param offload、小 param 留 GPU）。见 [[offloading (CPU-NVMe)]]。

### 8.4 与 [[Megatron-LM]] 的融合

Megatron 的 TP/PP 手写 module，FSDP2 的 DTensor 声明式。社区趋同：Megatron Core 向 DTensor/DeviceMesh 靠拢（`MegatronParallel` 用 mesh）。未来可能统一到 DTensor 抽象。详见 [[Megatron-LM]]。

### 8.5 内容来源

FSDP2 整理自 PyTorch dev docs "FSDP2"、`torch/distributed/fsdp/` 源码、RFC "FSDP2"。per-param vs flat 的对比见 FSDP2 design doc。HSDP 见 [[DeviceMesh]] nested mesh。与 FSDP1/ZeRO/Megatron 的关系见各自笔记。

---
相关: [[FSDP2]] | [[DTensor]] | [[DeviceMesh]] | [[Fully Sharded Data Parallel]] | [[ZeRO (DeepSpeed)]] | [[distributed checkpoint]] | [[torch.compile]] | [[TorchDynamo]] | [[3D parallelism]] | [[Megatron-LM]] | [[AllGather]] | [[reduce-scatter]] | [[all-reduce]] | [[overlap strategy]] | [[alpha-beta性能模型]] | [[训练显存估计]] | [[optimizer state memory]] | [[offloading (CPU-NVMe)]] | [[Distributed Data Parallel]]
