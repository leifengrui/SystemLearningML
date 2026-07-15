# Megatron distributed checkpoint

> **所属章节**: [[distributed checkpoint]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: Megatron distributed checkpoint / Megatron 分布式 ckpt / mp_rank tp_rank pp_rank 格式 / Megatron saver/loader / 框架原生 ckpt
> **难度**: 中高（需懂 [[EP TP DP CP组合规则]] + [[state_dict分片]] + [[distributed checkpoint|ch14 DCP]] + [[checkpoint转换]]）


## 1. 一句话定义

**Megatron distributed checkpoint** 是 [[Megatron-LM]] 框架原生的**分布式 checkpoint 格式与 saver/loader**——每个 GPU rank 把自己持有的那一份 sharded weight / optimizer state / scheduler state / RNG state **直接落盘**到 `iter_XXX/{mp_rank,tp_rank,pp_rank}_{..}/model.opt.ckpt`（文件名按 rank 的并行坐标 `mp_rank`（DP+EP 合并的模型并行 rank）、`tp_rank`（TP 维）、`pp_rank`（PP 维）命名），每 rank 一个文件（或一 rank 组一组文件），**不汇总到单进程**（避免单点显存爆）；目录里附 `latest_iteration.txt`（记录 iter）与 `ckpt_meta.json`（元数据：TP/PP/DP/EP 配置、layer 数、hidden、vocab、optimizer 类型）。loader 据 meta + 自己的 rank 坐标找到对应文件、加载对应 shard，支持**跨并行配置加载**（如训练时 TP=8 存、继续预训练 TP=4 加载，loader 据 meta 做跨 TP re-shard）。是 Megatron 训练断点续训、预训练→SFT 切换、checkpoint 归档的核心存储格式，与 [[distributed checkpoint|ch14 DCP]]（PyTorch 原生 DCP，per-rank + metadata）理念一致但格式不同（Megatron 用 `mp_rank` 命名 + 自己的 meta schema，是框架早期自研格式，现逐步对齐 DCP）。

> [!note] 三句话定位
> - **是什么**：每 rank 直接落盘（不汇总），文件按 `mp_rank/tp_rank/pp_rank` 坐标命名，附 meta（TP/PP/DP/EP 配置）；loader 据 rank 坐标加载对应 shard，支持跨配置 re-shard。
> - **为什么**：大模型单进程汇总会显存爆，且断点续训/换配置需灵活加载；分布式直存 + meta 让每 rank 独立、可跨配置。
> - **与 [[distributed checkpoint|ch14 DCP]] 关系**：理念一致（per-rank + metadata，跨拓扑 reshard），但 Megatron 用自研 `mp_rank` 命名 + meta schema（早期），ch14 DCP 是 PyTorch 原生统一标准（新）。两者在融合。


## 2. 为什么需要它（动机与背景）

### 2.1 单进程汇总的显存爆

朴素 ckpt：rank 0 汇总全模型（gather），存单文件。大模型（如 175B/671B）单进程持全量 weight + optimizer state 显存爆（数百 GB）。需每 rank 直接落盘（不汇总）。

### 2.2 并行配置的切换

训练中需切换并行配置：预训练 TP=8，继续预训练 TP=4；或 DP=64 存、DP=128 加载（增节点）。ckpt 需记录原配置（TP/PP/DP/EP），loader 据新配置 re-shard 加载。若只存单文件不记配置，无法切换。

### 2.3 optimizer state 与 scheduler

断点续训不只存 weight，还有 optimizer state（Adam 的 m/v，与 weight 同大）、scheduler state（lr step）、RNG state（可复现）、dataloader state（epoch/idx）。需一起存，否则续训发散。

### 2.4 每 rank 独立落盘

每 rank 直接写自己的 shard 到文件（不跨 rank 通信）。避免汇总通信开销 + 单点显存爆。文件按 rank 坐标命名，loader 按 rank 坐标找。

### 2.5 Megatron 自研格式

Megatron 早期（2019-2022）自研 `mp_rank` 格式（框架原生）。PyTorch 2023+ 出 DCP（[[distributed checkpoint|ch14 DCP]]）。两者理念一致（per-rank + metadata），Megatron 在对齐 DCP（`get_model_state_dict` 等 API 兼容）。是框架自研→标准化的演进。


## 3. 核心概念详解

### 3.1 目录结构

```
checkpoints/
├── iter_0001000/                    ← iter 1000
│   ├── mp_rank_00/
│   │   ├── model_optimizers.ckpt    ← 该 mp_rank 的 model+opt
│   │   └── rng.pth
│   ├── tp_rank_00/
│   │   └── model_optimizers.ckpt
│   ├── pp_rank_00/
│   │   └── ...
│   ├── ... (每 rank 一文件/一组)
│   └── latest_iteration.txt
├── iter_0002000/
├── latest_iteration.txt              ← 当前最新 iter (e.g., "2000")
└── ckpt_meta.json (或分散在各 rank 文件内)
```

每 rank 一文件（或按坐标命名的目录）。`mp_rank` 是 DP+EP 合并的 rank（model parallel rank），`tp_rank` 是 TP 维，`pp_rank` 是 PP 维。

### 3.2 文件命名

```
mp_rank_{mp:02d}/model_optimizers.ckpt          ← DP+EP 维
tp_rank_{tp:02d}/model_optimizers.ckpt          ← TP 维
pp_rank_{pp:02d}/model_optimizers.ckpt          ← PP 维
组合: mp_rank_{mp:02d}_tp_rank_{tp:02d}_pp_rank_{pp:02d}/...
```

rank 坐标命名。loader 据自己的 (mp, tp, pp) 坐标找文件。是 [[EP TP DP CP组合规则]] 的存储映射。

### 3.3 每 rank 存的内容

```python
# 每 rank 存的 state_dict (torch.save 序列化)
{
    'model': model_state_dict,       # 该 rank 持的 weight shard
    'optimizer': opt_state_dict,     # 该 rank 持的 opt state (Adam m/v 等)
    'scheduler': sched_state_dict,  # lr scheduler step
    'rng': rng_state,                # RNG (可复现)
    'dataloader': dl_state,         # epoch / idx
    'iteration': iter,              # 当前 iter
    # 'scaler': scaler_state (混合精度)
}
```

model + optimizer + scheduler + RNG + dataloader + iter。续训全恢复。是断点续训的完整状态。

### 3.4 saver 流程

```python
def save_checkpoint(iteration, model, optimizer, scheduler):
    # 每 rank 独立存 (不汇总)
    state_dict = {
        'model': get_model_state_dict(model),      # 该 rank shard
        'optimizer': get_optimizer_state_dict(optimizer),
        ...
    }
    path = f'iter_{iteration:07d}/{rank_coord}/model_optimizers.ckpt'
    torch.save(state_dict, path)  # 每 rank 独立写
    if is_rank_0:
        write_latest_iteration(iteration)
```

每 rank 独立写。无跨 rank 通信（除写 latest）。避免汇总开销。

### 3.5 loader 流程

```python
def load_checkpoint(model, optimizer, scheduler):
    iteration = read_latest_iteration()
    path = f'iter_{iteration:07d}/{rank_coord}/model_optimizers.ckpt'
    state_dict = torch.load(path, map_location='cpu')
    set_model_state_dict(model, state_dict['model'])       # 加载该 rank shard
    set_optimizer_state_dict(optimizer, state_dict['optimizer'])
    scheduler.load_state_dict(state_dict['scheduler'])
    set_rng(state_dict['rng'])
    ...
```

每 rank 据坐标找文件、加载对应 shard。无汇总。

### 3.6 跨配置 re-shard

```
训练: TP=8 存 -> 每 TP rank 一 shard (weight 切 8)
继续: TP=4 加载 -> 每 TP rank 需 2 个旧 shard 拼接 (8->4 re-shard)
loader 据 meta (原 TP=8) + 新配置 (TP=4) 自动拼接:
  - 读旧 TP rank 0,1 -> 拼成新 TP rank 0
  - 读旧 TP rank 2,3 -> 拼成新 TP rank 1
  ...
```

loader 据原配置（meta）+ 新配置做 re-shard（拼接/拆分）。支持切换并行配置。是核心灵活性。

### 3.7 meta 信息

```json
{
    "tp_size": 8, "pp_size": 4, "dp_size": 64, "ep_size": 2,
    "hidden_size": 8192, "num_layers": 80, "vocab_size": 128256,
    "num_experts": 256, "optimizer": "AdamW",
    "iteration": 1000, "lr": 1e-4
}
```

meta 记并行配置 + 模型结构 + optimizer 类型。loader 据此 re-shard + 校验结构。是跨配置加载的依据。

### 3.8 flatten param（Megatron-LM）与 per-param（FSDP2）

- **Megatron-LM flatten**：把该 rank 的所有 param flatten 成一维 buffer，opt state 也 flatten。存 flatten 的 buffer。高效但需 flatten/unflatten。
- **FSDP2 per-param**：每 param 独立 shard（DTensor），不 flatten。存 per-param shard。详见 [[FSDP2]] 与 [[state_dict分片]]。

Megatron 用 flatten（早期），FSDP2 用 per-param（新）。两者在融合（Megatron Core 支持 DTensor）。

### 3.9 与 DCP 的对齐

```python
# Megatron Core 新 API 对齐 DCP
from megatron.core.dist_checkpoint import save, load
save(model, optimizer, checkpoint_dir)   # 底层用 DCP 或兼容
load(model, optimizer, checkpoint_dir)  # 跨配置加载
```

Megatron Core 新 API 对齐 [[distributed checkpoint|ch14 DCP]]（`save`/`load` 接口 + per-rank + metadata）。旧 `mp_rank` 格式逐步迁移到 DCP。是自研→标准化的融合。


## 4. 数学原理 / 公式

### 4.1 每 rank 存储量

每 rank 存自己持有的 shard：
- weight: $\propto \frac{N_{\text{param}}}{T \cdot E}$（TP+EP 切 weight）
- optimizer state: $\propto \frac{N_{\text{param}}}{D}$（DP 切 opt，ZeRO-1）或 $\propto \frac{N_{\text{param}}}{D}$（ZeRO-2，含 grad）

每 rank 存 $\frac{N_{\text{param}}}{T \cdot E} + \frac{N_{\text{param}}}{D} \cdot k$（$k$ = opt state 系数，Adam $k=2$）。总存 $\sum$ 各 rank $= N_{\text{param}} + k \cdot N_{\text{param}}$（全模型 + 全 opt）。

### 4.2 跨配置 re-shard 的拼接

若旧 $T_{\text{old}}$ 存、新 $T_{\text{new}}$ 加载（$T_{\text{old}} > T_{\text{new}}$）：每新 rank 读 $\frac{T_{\text{old}}}{T_{\text{new}}}$ 个旧 shard 拼接。若 $T_{\text{old}} < T_{\text{new}}$：每新 rank 读 1 旧 shard 拆分。loader 自动。是 re-shard 的数学。

### 4.3 通信开销

saver/loader 每 rank 独立（无跨 rank 通信，除 re-shard 时读多文件）。re-shard 读多文件有 IO 开销（非 NCCL 通信）。是存储优化（避免汇总 NCCL）。

### 4.4 显存

saver/loader 每 rank 只持自己 shard（不持全模型）。显存 $\propto \frac{N_{\text{param}}}{T \cdot E} + \frac{N_{\text{param}}}{D}$（不爆）。是分布式存的核心收益。


## 5. 代码示例（可选）

### 5.1 Megatron saver

```python
from megatron.training.checkpointing import save_checkpoint
save_checkpoint(
    iteration=1000,
    model=model,            # 该 rank 的 model shard
    optimizer=optimizer,
    scheduler=lr_scheduler,
    # 每 rank 独立写, 文件名据 rank 坐标
)
# 写到 iter_0001000/{mp,tp,pp}_rank_XX/model_optimizers.ckpt
```

### 5.2 Megatron loader

```python
from megatron.training.checkpointing import load_checkpoint
iteration = load_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=lr_scheduler,
    # 据 rank 坐标找文件, 加载对应 shard
    # 支持跨配置 (meta re-shard)
)
```

### 5.3 Megatron Core DCP 接口

```python
from megatron.core.dist_checkpoint import save, load, load_sharded_state_dict
save({'model': model, 'optimizer': optimizer}, ckpt_dir)  # DCP 兼容
sd = load_sharded_state_dict(ckpt_dir)  # 跨配置
load(sd, model, optimizer)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[EP TP DP CP组合规则]]（rank 坐标映射）、[[state_dict分片]]（FULL/SHARDED/LOCAL）、[[distributed checkpoint|ch14 DCP]]（PyTorch 原生标准）、[[FSDP2]]（per-param 对比）、[[Megatron distributed optimizer]]（opt state 切分）、[[Megatron Core目录与执行链路]]。
- **下游（应用）**: Megatron 训练断点续训、预训练→SFT/RLHF 切换、[[HF与Megatron checkpoint转换]]（转 HF 格式）、checkpoint 归档/恢复。
- **对比 / 易混**:
  - **Megatron distributed ckpt vs [[distributed checkpoint|ch14 DCP]]**：前者是 Megatron 自研（`mp_rank` 命名 + 自家 meta），后者是 PyTorch 原生统一标准（per-rank + 标准 metadata）。理念一致（per-rank + 跨拓扑 reshard），格式/schema 不同。Megatron 在对齐 DCP。
  - **Megatron vs [[checkpoint转换|ch14 通用转换器]]**：Megatron 是框架原生格式（mp_rank），ch14 通用转换器是 HF↔FSDP2↔Megatron 三格式互转。后者是转换层，前者是存储格式。详见 [[HF与Megatron checkpoint转换]]。
  - **Megatron flatten vs FSDP2 per-param**：Megatron flatten param 成一维 buffer（早期，高效但需 unflatten），FSDP2 per-param（DTensor，新，易管理）。两者在融合。
  - **distributed ckpt vs [[gradient checkpointing]]**：前者是 weight/opt 的存储（断点续训），后者是 activation 的重算（省显存）。名字像但概念不同。


## 7. 常见误区与易错点

> [!warning] 误区 1：单进程汇总存
> 大模型单进程汇总显存爆。Megatron 每 rank 独立落盘（不汇总）。误以为汇总存会爆。

> [!warning] 误区 2：ckpt 只存 weight
> 断点续训需 weight + optimizer state + scheduler + RNG + dataloader。只存 weight 续训发散（Adam 的 m/v 丢失）。Megatron 存全状态。

> [!warning] 误区 3：不能跨配置加载
> loader 据 meta re-shard 支持跨配置（TP=8 存、TP=4 加载）。误以为不能会重训。

> [!warning] 误区 4：mp_rank 是单一 rank
> `mp_rank` 是 DP+EP 合并的 rank（model parallel rank），含 DP 与 EP 两个维度的信息。不是单一维。误以为单一会错找文件。

> [!warning] 误区 5：Megatron ckpt = DCP
> 理念一致但格式不同。Megatron 用 `mp_rank` 命名 + 自家 meta schema（早期自研），DCP 是 PyTorch 原生标准（per-rank + 标准 metadata）。Megatron 在对齐 DCP，但未完全等价。误以为等价会格式错。

> [!warning] 误区 6：flatten 与 per-param 一样
> Megatron flatten 成一维 buffer（高效但 unflatten 麻烦），FSDP2 per-param（DTensor，易管理）。两者格式不同，转换需 flatten/unflatten。详见 [[state_dict分片]]。

> [!warning] 误区 7：optimizer state 不切分
> Megatron distributed optimizer（ZeRO-1/2）把 opt state 切到 DP 维。ckpt 存切分后的 opt shard（不汇总）。误以为存全 opt 会显存爆。详见 [[Megatron distributed optimizer]]。


## 8. 延伸细节

### 8.1 flatten param 的细节

Megatron-LM 把该 rank 的所有 param flatten 成一维 buffer（`param.data` 指向 buffer 的 slice）。存的是 flatten buffer + metadata（每 param 的 offset/shape）。loader 据此 unflatten。是 [[state_dict分片]] 的 SHARDED 格式实现。

### 8.2 异步 IO 与 checkpoint 阠塞

大 ckpt 写盘慢（数百 GB）。Megatron 支持异步 IO（后台线程写，训练继续）。但下次存前需等上次写完（否则覆盖）。是 IO 优化。

### 8.3 与 [[HF与Megatron checkpoint转换]] 的衔接

Megatron ckpt 是训练格式，HF 格式是推理/发布格式（单文件 safetensors）。转换需 weight name mapping + TP/PP recombine。详见 [[HF与Megatron checkpoint转换]]。

### 8.4 内容来源

Megatron distributed checkpoint 整理自 Megatron-LM `megatron/training/checkpointing.py` 与 `megatron/core/dist_checkpoint/` 源码、Megatron 文档。与 DCP 的对照见 [[distributed checkpoint|ch14 DCP]] 与 PyTorch DCP 文档。flatten param 见 `megatron/core/tensor_parallel/mappings.py`。

---
相关: [[distributed checkpoint]] | [[state_dict分片]] | [[FSDP2]] | [[HF与Megatron checkpoint转换]] | [[checkpoint转换]] | [[EP TP DP CP组合规则]] | [[Megatron distributed optimizer]] | [[gradient checkpointing]] | [[Megatron Core目录与执行链路]] | [[Megatron-LM]]
