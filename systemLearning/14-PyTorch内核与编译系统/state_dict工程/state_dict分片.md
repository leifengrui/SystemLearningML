# state_dict分片

> **所属章节**: [[state_dict工程]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: state_dict 分片 / SHARDED_STATE_DICT / FULL_STATE_DICT / LOCAL_STATE_DICT / FSDP state_dict 类型 / 分片 state_dict
> **难度**: 中（需懂 [[state_dict与load_state_dict]] + [[FSDP2]] + [[distributed checkpoint]]）


## 1. 一句话定义

**state_dict 分片** 是 FSDP（[[Fully Sharded Data Parallel|FSDP1]] 与 [[FSDP2]]）对模型 `state_dict` 的**三种组织格式**选择——**`FULL_STATE_DICT`**（rank 0 先 all-gather 把所有 shard param unshard 成完整 model 再 `torch.save`，内存爆但跨框架通用、是 HF/megatron 互转的中间态）、**`SHARDED_STATE_DICT`**（每 rank 只暴露自己的 local 片 state_dict，用 [[distributed checkpoint|DCP]] 存分片 + metadata，省内存、支持跨拓扑 resharding，是 FSDP2 原生格式）、**`LOCAL_STATE_DICT`**（存 FSDP 内部 pre-shard 的 local param，历史遗留、不推荐）。用户通过 `FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT)` 上下文管理器切换，让 `model.state_dict()` / `model.load_state_dict()` 返回/接受对应格式。它是 checkpoint 流程的**格式选择层**（决定 state_dict 长什么样），底层存取由 DCP（`SHARDED`）或 `torch.save`（`FULL`）实现，上层策略（async/incremental）由 [[checkpointing策略]] 定。

> [!note] 三句话定位
> - **是什么**：FSDP state_dict 的三种格式（FULL=聚合完整/SHARDED=per-rank 分片/LOCAL=pre-shard 遗留），用上下文管理器切换。
> - **为什么**：FSDP 的 param 是 shard 的，直接 `state_dict()` 返回的是 shard 片（跨拓扑难、HF 不认）。FULL 聚合完整（通用但爆内存）、SHARDED 分片存（省内存+跨拓扑，用 DCP）。
> - **与 [[distributed checkpoint]] 关系**：`SHARDED_STATE_DICT` 用 DCP 存取（per-rank 片+metadata），`FULL_STATE_DICT` 用 `torch.save` 存完整。state_dict 分片是格式选择层，DCP 是底层实现。


## 2. 为什么需要它（动机与背景）

### 2.1 FSDP param 是 shard 的

FSDP（1/2）的 param 是 shard 的（每 rank 只存片）。直接 `model.state_dict()` 返回的 param 是 local 片（非完整 model）。`torch.save(model.state_dict(), ...)` 存的是片（加载到不同拓扑/框架会错）。需选格式决定 state_dict 长什么样。

### 2.2 三种格式的取舍

- **`FULL_STATE_DICT`**：rank 0 all-gather 把所有 shard param unshard 成完整 model，再 `torch.save` 存完整。跨框架通用（HF/megatron 认完整 model），但 rank 0 内存爆（聚合完整 model，大模型不行）。
- **`SHARDED_STATE_DICT`**：每 rank 只暴露自己的 local 片 state_dict，用 DCP 存（per-rank 片 + metadata）。省内存（不聚合）、支持跨拓扑 resharding（DCP 自动）。是 FSDP2 原生格式。
- **`LOCAL_STATE_DICT`**：存 FSDP 内部 pre-shard 的 local param（FSDP1 的原始 param，shard 前）。历史遗留，不推荐（与 SHARDED 类似但无 metadata，跨拓扑难）。

### 2.3 格式选择的解耦

state_dict 分片是**格式选择层**：决定 `state_dict()` 返回什么。底层存取（DCP 或 torch.save）与上层策略（async/incremental，见 [[checkpointing策略]]）解耦。用户只选格式，DCP/torch.save 透明。

### 2.4 跨框架互转的枢纽

`FULL_STATE_DICT` 是 HF/megatron 互转的中间态（HF 认完整 model）。FSDP2 训练用 `SHARDED`（省内存），导出给 HF 时切 `FULL`（聚合完整）→ 存 → HF 加载。[[checkpoint转换]] 的枢纽。详见 [[checkpoint转换]]。

### 2.5 跨拓扑 resharding 的支持

`SHARDED_STATE_DICT` + DCP 支持跨拓扑（8→16 卡、shard→replicate）。`FULL` 是完整 model，加载到任意拓扑都行（无 resharding 需，但内存爆）。`LOCAL` 无 metadata，跨拓扑难。


## 3. 核心概念详解

### 3.1 三种格式的对比

| 格式 | state_dict 内容 | 存储方式 | 内存 | 跨拓扑 | 跨框架 | 用途 |
|---|---|---|---|---|---|---|
| **`FULL_STATE_DICT`** | rank 0 聚合完整 model | `torch.save`（完整） | rank 0 爆 | ✅（完整 model） | ✅（HF/megatron 认） | 互转中间态、小模型 |
| **`SHARDED_STATE_DICT`** | 每 rank 自己 local 片 | DCP（per-rank 片+metadata） | 省 | ✅（DCP resharding） | ❌（HF 不认分片） | FSDP2 原生、大模型、训练中断恢复 |
| **`LOCAL_STATE_DICT`** | pre-shard local param | `torch.save`（local） | 省 | ❌（无 metadata） | ❌ | 历史遗留、不推荐 |

### 3.2 切换 state_dict 类型

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

# SHARDED (FSDP2 原生)
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    sd = model.state_dict()  # 返回每 rank local 片
    # 存: 用 DCP
    DCP.save(sd, writer)

# FULL (互转中间态)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    sd = model.state_dict()  # rank 0 聚合完整 (其他 rank 空)
    torch.save(sd, "full.pt")  # rank 0 存完整
```

### 3.3 FULL_STATE_DICT 的聚合

```python
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    sd = model.state_dict()
    # 内部: rank 0 all-gather 所有 shard param -> unshard 完整
    # sd 在 rank 0 是完整 model state_dict, 其他 rank 空
    # rank 0 torch.save 存完整
```

rank 0 内存峰值 = 完整 model 大小。大模型不行（如 70B = 140GB fp16，rank 0 装不下）。仅小模型或互转用。

### 3.4 SHARDED_STATE_DICT 的分片

```python
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    sd = model.state_dict()
    # 每 rank sd 只含自己 local 片 (DTensor 的 to_local)
    # 存用 DCP (per-rank 文件 + metadata)
    DCP.save(sd, writer)
```

每 rank 只存自己的片，内存省。加载时 DCP 据 metadata + 目标 mesh resharding。是 FSDP2 原生。

### 3.5 LOCAL_STATE_DICT 的历史

```python
with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
    sd = model.state_dict()
    # 存 FSDP1 pre-shard 的原始 local param (shard 前)
    torch.save(sd, "local.pt")
```

FSDP1 早期格式。pre-shard local param 无 placement metadata，跨拓扑/框架难。已被 SHARDED 取代，不推荐。

### 3.6 与 DCP 的关系

`SHARDED_STATE_DICT` 的存取用 DCP（per-rank 片 + metadata + resharding）。`FULL` 用 `torch.save`（完整聚合）。`LOCAL` 用 `torch.save`（local 片，无 metadata）。state_dict 分片是格式选择，DCP 是 SHARDED 的实现。详见 [[distributed checkpoint]]。

### 3.7 与 checkpoint 转换的关系

HF checkpoint（replicate，safetensors）↔ FSDP2（shard）：用 `FULL_STATE_DICT` 作中间态。FSDP2 → HF：切 FULL 聚合完整 → 存 safetensors → HF 加载。HF → FSDP2：HF 加载完整 → FSDP2 切 SHARDED 存分片。详见 [[checkpoint转换]]。

### 3.8 与 optimizer state 的分片

FSDP 的 optimizer state（如 Adam 的 $m, v$）也 shard（per-rank）。`SHARDED_STATE_DICT` 包含 optimizer state 的分片。`FULL` 聚合完整 optimizer state（更爆）。`SHARED` 存分片 optimizer state（省）。见 [[optimizer state memory]]。

### 3.9 上下文管理器的作用

`FSDP.state_dict_type(model, type)` 是上下文管理器，在 `with` 块内 `state_dict()`/`load_state_dict()` 行为变（返回/接受对应格式）。出 `with` 恢复默认（FSDP2 默认 SHARDED）。避免全局污染。

### 3.10 FSDP2 的默认

FSDP2 的 param 是 DTensor，默认 state_dict 返回 DTensor（含 placement）。`SHARDED_STATE_DICT` 是 FSDP2 的默认/推荐。`FULL` 仍可用（互转）。`LOCAL` 基本弃用。


## 4. 数学原理 / 公式

### 4.1 FULL 的内存峰值

rank 0 聚合完整 model。设 model size $N$，FULL 内存峰值 $= N$（rank 0 装完整）。大模型 $N$ 大（70B fp16 = 140GB），rank 0 装不下。仅小模型用。

### 4.2 SHARDED 的内存

每 rank 存自己片。shard 沿 $d$ 个 rank，每 rank local $= N/d$。SHARDED 内存 $= N/d$（per rank）。省 $d$ 倍。

### 4.3 FULL 的通信

FULL all-gather 所有 param 到 rank 0。通信 $\propto N$（聚合完整）。大模型通信大。SHARDED 无聚合通信（per-rank 各存各）。

### 4.4 SHARDED 跨拓扑的通信

加载跨拓扑（8→16 卡）需 DCP resharding 通信（all-gather 合再切）。$\propto N$。同拓扑无通信（每 rank 读自己片）。


## 5. 代码示例（可选）

### 5.1 SHARDED 保存/加载

```python
import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

model = FSDP(MyModel().cuda(), ...)  # FSDP2
writer = FileSystemWriter("/ckpt")

# 保存 SHARDED
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    sd = model.state_dict()  # 每 rank local 片
    DCP.save(sd, writer)  # DCP 存分片 + metadata

# 加载 SHARDED (跨拓扑: 8 -> 16)
model_new = FSDP(MyModel().cuda(), ...)
with FSDP.state_dict_type(model_new, StateDictType.SHARDED_STATE_DICT):
    sd = model_new.state_dict()
    DCP.load(sd, writer)  # DCP resharding
    model_new.load_state_dict(sd)
```

### 5.2 FULL 保存（互转中间态）

```python
# FSDP2 -> HF: 切 FULL 聚合完整
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    sd = model.state_dict()  # rank 0 完整, 其他空
    if dist.get_rank() == 0:
        torch.save(sd, "full.pt")  # rank 0 存完整
        # 或转 safetensors 给 HF
```

### 5.3 包含 optimizer state

```python
opt = torch.optim.Adam(model.parameters())
# SHARDED 含 optimizer state 分片
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    sd = {
        "model": model.state_dict(),
        "optimizer": FSDP.optim_state_dict(model, opt, ...)  # shard opt state
    }
    DCP.save(sd, writer)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[state_dict与load_state_dict]]（state_dict 抽象）、[[FSDP2]]/[[Fully Sharded Data Parallel|FSDP1]]（param 是 shard 的，需选格式）、[[distributed checkpoint]]（SHARDED 的实现）。
- **下游（应用）**: [[checkpoint转换]]（FULL 作互转中间态）、[[checkpointing策略]]（async/incremental 的格式基础）、[[optimizer state memory]]（optimizer state 分片）、HF/megatron 互转。
- **对比 / 易混**:
  - **SHARDED vs FULL vs LOCAL**：见 3.1，per-rank 分片+DCP vs rank 0 聚合+torch.save vs pre-shard 遗留。
  - **state_dict 分片 vs [[distributed checkpoint]]**：state_dict 分片是格式选择层（决定 state_dict 长什么样），DCP 是 SHARDED 的底层实现（per-rank 文件+metadata+resharding）。前者是 API 选择，后者是机制。
  - **state_dict 分片 vs [[gradient checkpointing]]**：前者是权重 state_dict 的格式（存/加载 model），后者是激活重算（省显存）。完全不同。勿混。
  - **SHARDED_STATE_DICT vs FSDP2 的 DTensor state_dict**：FSDP2 的 param 是 DTensor，默认 state_dict 返回 DTensor（含 placement）。SHARDED_STATE_DICT 是 FSDP1/2 通用的格式名（用 DCP 存）。FSDP2 下两者等价（DTensor state_dict = SHARDED）。


## 7. 常见误区与易错点

> [!warning] 误区 1：直接 `torch.save(model.state_dict())` 存 FSDP
> FSDP 的 `model.state_dict()` 返回 shard 片（默认 SHARDED）或需切格式。直接 `torch.save` 存片，加载到不同拓扑/框架会错（无 metadata/未聚合）。要用对应格式（SHARDED 用 DCP，FULL 聚合）。

> [!warning] 误区 2：FULL 省 memory
> FULL 是 rank 0 聚合完整 model，内存爆。仅小模型或互转用。大模型用 SHARDED（省）。误用 FULL 存 70B 会 OOM。

> [!warning] 误区 3：LOCAL 与 SHARDED 一样
> LOCAL 是 pre-shard local param（无 metadata），SHARDED 是 shard 片+metadata（DCP）。LOCAL 跨拓扑难（无 metadata），SHARDED 支持（DCP resharding）。LOCAL 是历史遗留，不推荐。

> [!warning] 误区 4：SHARDED 不需 resharding
> SHARDED 跨拓扑加载需 DCP resharding（8→16 卡 all-gather 合再切）。同拓扑（mesh 相同）才无通信。误以为 SHARDED 任意加载无通信会错。

> [!warning] 误区 5：忽视 optimizer state 的分片
> optimizer state（Adam 的 $m,v$）也 shard。`SHARDED_STATE_DICT` 含分片 optimizer state。若用 FULL，optimizer state 也聚合（更爆）。存/加载 optimizer state 要配套格式。

> [!warning] 误区 6：忘记切格式
> FSDP2 默认 SHARDED，但 FSDP1 默认 LOCAL。存/加载前显式切格式（`with FSDP.state_dict_type(...)`），勿依赖默认（版本变可能不同）。

> [!warning] 误区 7：state_dict 分片与 DCP 混
> state_dict 分片是格式选择（FULL/SHARDED/LOCAL），DCP 是 SHARDED 的底层实现。前者是 API 层，后者是机制层。勿混为一谈。


## 8. 延伸细节

### 8.1 FSDP.optim_state_dict

FSDP 提供 `FSDP.optim_state_dict(model, opt)` / `FSDP.optim_state_to_load` 把 optimizer state 也按格式分片/聚合。SHARDED 存分片 optimizer state（省），FULL 聚合（爆）。是 optimizer state memory 的配套。详见 [[optimizer state memory]]。

### 8.2 与 [[checkpointing策略]] 的协同

state_dict 分片是格式层，checkpointing 策略是策略层（何时存、async/incremental）。如 async checkpoint 用 SHARDED（DCP thread mode）+ 策略定何时触发。incremental 用 SHARDED 选 param 存。两者解耦。

### 8.3 FULL 的 rank 0 only

FULL 只 rank 0 存（其他 rank state_dict 空）。存时 `if rank == 0: torch.save`。加载时 rank 0 load 再 broadcast（或各 rank 从完整 unshard 到自己片）。是互转中间态的典型用法。

### 8.4 内容来源

state_dict 分片整理自 PyTorch dev docs "FSDP state_dict type"、`torch/distributed/fsdp/` 源码、FSDP state_dict RFC。三种格式的对比见 FSDP 文档。与 DCP/checkpoint 转换的关系见 [[distributed checkpoint]]/[[checkpoint转换]]。

---
相关: [[state_dict工程]] | [[state_dict与load_state_dict]] | [[FSDP2]] | [[Fully Sharded Data Parallel]] | [[distributed checkpoint]] | [[checkpoint转换]] | [[checkpointing策略]] | [[optimizer state memory]] | [[gradient checkpointing]]
