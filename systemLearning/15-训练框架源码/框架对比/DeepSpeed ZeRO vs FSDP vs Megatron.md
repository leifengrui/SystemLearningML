# DeepSpeed ZeRO vs FSDP vs Megatron

> **所属章节**: [[框架对比]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: DeepSpeed ZeRO vs FSDP vs Megatron / ZeRO vs FSDP vs Megatron-DO / 三大分布式训练框架对比 / 分布式优化器对比 / 框架选型
> **难度**: 中高（需懂 [[ZeRO]] + [[FSDP2]] + [[Megatron distributed optimizer]] + [[FSDP]] + [[3D parallelism]]）


## 1. 一句话定义

**DeepSpeed ZeRO vs FSDP vs Megatron** 是三大主流**分布式训练框架/优化器**的横向对比——[[DeepSpeed]] 的 **ZeRO**（Zero Redundancy Optimizer，ZeRO-1 切 optimizer state、ZeRO-2 切 grad、ZeRO-3 切 param，按 $D$ 维切，reduce-scatter + all-gather，配置驱动，易用，但框架侵入强、ZeRO-3 的 param all-gather 开销大）、[[FSDP]]（PyTorch 原生，FSDP1 用 flat param buffer（flatten），FSDP2 用 per-param DTensor（[[DTensor]] + [[DeviceMesh]]，torch.compile 友好，是 PyTorch 新主线）、[[Megatron-LM]] 的 **Megatron distributed optimizer**（[[Megatron distributed optimizer|Megatron-DO]]，原生 ZeRO-1/2，flat grad buffer + reduce-scatter async，与 TP/PP/EP/CP 紧耦合，框架原生、性能极致但配置复杂、封闭）。三者**切分粒度**（ZeRO-3 全切 param vs FSDP2 per-param vs Megatron-DO 只切 opt/grad）、**通信模式**（ZeRO-3/FSDP all-gather + reduce-scatter vs Megatron-DO reduce-scatter + all-gather，但 Megatron 与 TP all-reduce 耦合）、**易用性/可移植性**（DeepSpeed 配置易/FSDP2 PyTorch 原生最易/Megatron 封闭最难）、**与 torch.compile 集成**（FSDP2 最佳，Megatron 适配中，DeepSpeed 兼容弱）。是选型大模型分布式训练框架的核心决策表，决定显存、通信、易用、生态的权衡。

> [!note] 三句话定位
> - **是什么**：三大分布式训练框架的横向对比（DeepSpeed ZeRO / PyTorch FSDP / Megatron-DO），按切分粒度、通信模式、易用性、compile 集成四维对比。
> - **为什么**：不同框架在大模型训练的显存/性能/易用权衡不同，选型需对比表决策（极致性能选 Megatron，易用/原生选 FSDP2，存量生态选 DeepSpeed）。
> - **与 [[ZeRO]] 关系**：ZeRO 是切分理念（切 opt/grad/param），三者是实现（DeepSpeed 最早实现、Megatron 原生实现、FSDP2 原生+DTensor）。ZeRO 是共同理论。


## 2. 为什么需要它（动机与背景）

### 2.1 三框架并立

大模型分布式训练三大框架：
- **DeepSpeed**（微软，2020）：ZeRO 理论 + 实现，配置驱动，生态广（早期主流）。
- **FSDP**（PyTorch 原生，2022+）：FSDP1 flat param，FSDP2 per-param DTensor（新主线）。
- **Megatron-LM**（NVIDIA，2019+）：TP/PP/EP/CP + distributed optimizer，极致性能（训练大模型标杆）。

三者理念相关（ZeRO 切分）但实现与生态不同。选型需对比。

### 2.2 切分粒度的不同

- ZeRO-3：切 param（每 rank 持 param shard，前向 all-gather 用，用完释放）。
- FSDP2：per-param 切（每 param 独立 shard，DTensor）。
- Megatron-DO：只切 opt + grad（param 不切，靠 TP 切 param）。

切分粒度决定显存节省与通信开销的权衡。

### 2.3 通信模式的不同

- ZeRO-3/FSDP：前向 all-gather（param）+ 反向 reduce-scatter（grad）。
- Megatron-DO：反向 reduce-scatter（grad）+ step all-gather（param），与 TP all-reduce 耦合。

通信模式决定 overlap 策略与 wall-clock。

### 2.4 易用性/可移植性

- DeepSpeed：配置文件（`ds_config.json`），易上手，侵入框架（DeepSpeed engine）。
- FSDP2：PyTorch 原生，与 torch.compile 集成最佳，易移植。
- Megatron：框架封闭（自研训练 loop），配置复杂（TP/PP/EP/CP 全配），难移植。

易用性决定学习曲线与生态扩展。

### 2.5 compile 集成

torch.compile（[[torch.compile]]）是大模型训练的新主线（提算力）。FSDP2 与 compile 集成最佳（per-param DTensor 兼容图捕获），Megatron 适配中，DeepSpeed 兼容弱（flat param buffer 不利图捕获）。是新一代的选型因素。


## 3. 核心概念详解

### 3.1 三框架速览

| 维度 | DeepSpeed ZeRO | PyTorch FSDP | Megatron-DO |
|---|---|---|---|
| 切 opt state | ZeRO-1 ✅ | FSDP ✅（ZeRO-1） | ✅ |
| 切 grad | ZeRO-2 ✅ | FSDP ✅（ZeRO-2，部分） | ✅ |
| 切 param | ZeRO-3 ✅ | FSDP1 flat, FSDP2 per-param | ❌（靠 TP 切 param） |
| 通信 | all-gather + reduce-scatter | all-gather + reduce-scatter | reduce-scatter + all-gather（与 TP 耦合） |
| 并行维 | 主要 DP（+ PP via PipeDream） | DP（+ TP via DTensor） | TP+PP+DP+EP+CP 全维 |
| 易用 | 配置文件，易上手 | PyTorch 原生，最易 | 封闭，最难 |
| compile 集成 | 弱（flat buffer 不利图） | 最佳（per-param DTensor） | 适配中 |
| 生态 | 早期主流，存量广 | PyTorch 主线，新 | 训练大模型标杆 |

### 3.2 ZeRO 切分（共同理论）

[[ZeRO]] 是三者共同理论：
- ZeRO-1：切 optimizer state（每 rank 持 $\frac{1}{D}$ opt state）
- ZeRO-2：切 grad（每 rank 持 $\frac{1}{D}$ grad）
- ZeRO-3：切 param（每 rank 持 $\frac{1}{D}$ param，前向 all-gather）

三框架实现 ZeRO 的不同级别：
- DeepSpeed：ZeRO-1/2/3 全实现（最早）
- FSDP：FSDP1 ≈ ZeRO-3（flat param），FSDP2 ≈ ZeRO-3（per-param）
- Megatron-DO：ZeRO-1/2（不切 param，靠 TP 切）

### 3.3 DeepSpeed ZeRO 详解

```json
// ds_config.json
{
  "zero_optimization": {
    "stage": 3,           // ZeRO-3 (切 param)
    "overlap_comm": true,  // 通信与计算 overlap
    "reduce_scatter": true
  }
}
```

DeepSpeed 用配置文件驱动 ZeRO。ZeRO-3 前向 all-gather param（用完释放）+ 反向 reduce-scatter grad。优点：易上手（配置），切 param 省显存大。缺点：all-gather param 通信量大（$\propto N_{\text{param}}$ 每层），侵入框架（DeepSpeed engine 替换 optimizer）。

### 3.4 FSDP 详解

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# FSDP1 (flat param)
model = FSDP(model, sharding_strategy=ShardStrategy.FULL_SHARD)  # ZeRO-3
# FSDP2 (per-param DTensor)
from torch.distributed.tensor import DTensor
model = ...  # 用 DTensor 表达 sharded param
```

FSDP1 用 flat param buffer（flatten），FSDP2 用 per-param DTensor（每 param 独立 shard）。FSDP2 是新主线（torch.compile 友好）。详见 [[FSDP2]] 与 [[DTensor]]。

### 3.5 Megatron-DO 详解

```python
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
# Megatron-DO (ZeRO-1/2 native, flat grad buffer)
optimizer = DistributedOptimizer(optimizer, ...,
    clip_grad=..., overlap_comm=True)  # reduce-scatter async
# param 不切 (靠 TP 切), 只切 opt + grad
```

Megatron-DO 原生 ZeRO-1/2（不切 param，靠 TP 切 param）。flat grad buffer + reduce-scatter async（与 backward overlap）。与 TP all-reduce 耦合（TP 切 param + DO 切 opt/grad 配合）。详见 [[Megatron distributed optimizer]]。

### 3.6 切分粒度对比

```
DeepSpeed ZeRO-3: param 切 (每 rank 1/D param, 前向 all-gather)
FSDP2: per-param 切 (每 param 1/D, DTensor, all-gather per param)
Megatron-DO: param 不切 (靠 TP), opt+grad 切 (1/D)
```

ZeRO-3/FSDP 切 param（省显存大，但 all-gather 通信大）。Megatron-DO 不切 param（靠 TP 切），只切 opt/grad（通信少，但需 TP 配合）。是切分策略的核心差异。

### 3.7 通信量对比

- ZeRO-3/FSDP：前向 all-gather param（$\propto N_{\text{param}}$ 每层，大）+ 反向 reduce-scatter grad（$\propto N_{\text{param}}$，大）
- Megatron-DO：反向 reduce-scatter grad（$\propto N_{\text{param}}/D$，每 rank）+ step all-gather param（$\propto N_{\text{param}}/D$）+ TP all-reduce（$\propto L \cdot B \cdot S \cdot d$，每层）

ZeRO-3/FSDP 的 all-gather param 大（每层 $\propto N_{\text{param}}$）。Megatron-DO 的 opt all-gather 少（每 step 一次），但需 TP all-reduce（每层）。权衡：ZeRO-3 省显存但通信大，Megatron-DO 通信少但需 TP。

### 3.8 overlap 策略

- DeepSpeed：`overlap_comm=true`（reduce-scatter 与 backward overlap，bucket async）
- FSDP2：`reshard_after_forward`（用完释放）+ bucket async
- Megatron-DO：`overlap_comm=true`（reduce-scatter async，与 TP all-reduce 协同）

三者都支持 overlap。Megatron-DO 的 overlap 与 TP all-reduce 协同最深（原生集成）。详见 [[gradient bucket与通信重叠]] 与 [[overlap strategy]]。

### 3.9 compile 集成

| 框架 | compile 集成 | 原因 |
|---|---|---|
| FSDP2 | 最佳 | per-param DTensor 兼容图捕获，torch.compile 原生支持 |
| Megatron | 适配中 | 框架自研 loop，需适配 compile（Megatron-OSS 在做） |
| DeepSpeed | 弱 | flat param buffer 不利图捕获，ZeRO-3 的 all-gather 动态 |

FSDP2 是 torch.compile 的最佳搭档（PyTorch 原生 + DTensor）。是新一代选型的关键。

### 3.10 适用场景

| 场景 | 推荐 |
|---|---|
| 极致性能 + 大模型 + 多维并行 | Megatron-LM（TP+PP+EP+CP+DO） |
| PyTorch 原生 + torch.compile + 中大规模 | FSDP2 |
| 存量 DeepSpeed 生态 + 配置易用 | DeepSpeed ZeRO |
| 研究/中小模型 + 易上手 | FSDP2 或 DeepSpeed |

选型据场景。详见各框架笔记。


## 4. 数学原理 / 公式

### 4.1 显存（每 rank）

| 阶段 | weight | grad | opt state | activation |
|---|---|---|---|---|
| ZeRO-1 | $\psi$ | $\psi$ | $\frac{\psi k}{D}$ | $\propto L \cdot B \cdot S \cdot d$ |
| ZeRO-2 | $\psi$ | $\frac{\psi}{D}$ | $\frac{\psi k}{D}$ | $\propto L \cdot B \cdot S \cdot d$ |
| ZeRO-3 | $\frac{\psi}{D}$ | $\frac{\psi}{D}$ | $\frac{\psi k}{D}$ | $\propto L \cdot B \cdot S \cdot d$ |
| FSDP2 | $\frac{\psi}{D}$ | $\frac{\psi}{D}$ | $\frac{\psi k}{D}$ | $\propto L \cdot B \cdot S \cdot d$ |
| Megatron-DO | $\frac{\psi}{T \cdot E}$ | $\frac{\psi}{D}$ | $\frac{\psi k}{D}$ | $\propto \frac{L \cdot B \cdot S \cdot d}{P \cdot C}$ |

$\psi = N_{\text{param}}$，$k$ = opt state 系数（Adam $k=2$），$T$=TP，$D$=DP，$E$=EP，$P$=PP，$C$=CP。

ZeRO-3/FSDP 切 param（$\frac{\psi}{D}$），Megatron-DO 靠 TP 切（$\frac{\psi}{T \cdot E}$，TP 更省）。

### 4.2 通信量

- ZeRO-3/FSDP：$\propto 2 L \cdot \psi$（每层 all-gather + reduce-scatter，每层 $\psi$）
- Megatron-DO：$\propto \psi$（每 step 一次 reduce-scatter + all-gather）+ $L \cdot B \cdot S \cdot d$（TP all-reduce 每层）

ZeRO-3 通信 $\propto L \cdot \psi$（大，每层）。Megatron-DO 通信 $\propto \psi$（每 step）+ TP（每层）。Megatron-DO 用 TP 切 param 避免每层 all-gather param，通信更优（但需 TP 配合）。

### 4.3 overlap 的收益

三者通信可 overlap（reduce-scatter 与 backward）。overlap 后 wall-clock $\approx$ 计算。Megatron-DO 的 overlap 与 TP 协同最深。

### 4.4 显存 vs 通信的权衡

- ZeRO-3：显存最省（切 param）但通信大（每层 all-gather param）
- FSDP2：同 ZeRO-3 + per-param 灵活 + compile 友好
- Megatron-DO：显存靠 TP 切（更省）+ 通信少（不每层 all-gather param）+ 但需 TP 配置复杂

是显存 vs 通信 + 易用 vs 性能的权衡。


## 5. 代码示例（可选）

### 5.1 DeepSpeed

```python
import deepspeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model, optimizer=optimizer, config=ds_config  # ZeRO-3
)
# 配置驱动, 易上手
```

### 5.2 FSDP2

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor, Shard
# FSDP2 (per-param DTensor)
model = FSDP(model, sharding_strategy=ShardStrategy.FULL_SHARD)  # ZeRO-3
# 或用 DTensor 表达 sharded param
model = torch.compile(model)  # compile 集成
```

### 5.3 Megatron-DO

```python
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
optimizer = DistributedOptimizer(
    optimizer, clip_grad=1.0, overlap_comm=True  # ZeRO-1/2 native
)
# 需配合 TP/PP/EP/CP 全维配置
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[ZeRO]]（共同切分理论）、[[FSDP2]]、[[FSDP]]、[[Megatron distributed optimizer]]、[[DeepSpeed]]、[[DTensor]]、[[DeviceMesh]]、[[torch.compile]]、[[3D parallelism]]、[[gradient bucket与通信重叠]]、[[overlap strategy]]。
- **下游（应用）**: 大模型分布式训练框架选型、训练框架迁移（DeepSpeed→FSDP2）、torch.compile 训练栈选型。
- **对比 / 易混**:
  - **ZeRO-3 vs FSDP2 vs Megatron-DO**：ZeRO-3 切 param（DeepSpeed），FSDP2 per-param 切（PyTorch），Megatron-DO 不切 param（靠 TP）。切 param 策略不同。
  - **FSDP1 vs FSDP2**：FSDP1 flat param buffer（flatten），FSDP2 per-param DTensor（新，compile 友好）。详见 [[FSDP2]]。
  - **DeepSpeed vs Megatron**：DeepSpeed 配置易/生态广/ZeRO-3 强；Megatron 封闭/性能极致/全维并行。选型据易用 vs 性能。
  - **FSDP2 vs Megatron**：FSDP2 PyTorch 原生/compile 最佳/易移植；Megatron 性能极致/全维/封闭。选型据原生 vs 极致。


## 7. 常见误区与易错点

> [!warning] 误区 1：三者等价
> 理论同（ZeRO），但切分粒度（param 切 vs 不切）、通信（每层 all-gather vs 每 step）、易用（配置 vs 封闭）、compile（弱 vs 最佳）不同。误以为等价会误选型。

> [!warning] 误区 2：Megatron-DO = ZeRO-3
> Megatron-DO 只 ZeRO-1/2（切 opt/grad，不切 param）。param 靠 TP 切。误以为 ZeRO-3 会错判通信（Megatron 无每层 all-gather param）。

> [!warning] 误区 3：FSDP1 = FSDP2
> FSDP1 flat param buffer（flatten，不利 compile），FSDP2 per-param DTensor（新，compile 友好）。误以为一样会错判 compile 集成。

> [!warning] 误区 4：DeepSpeed 性能最优
> DeepSpeed 易用但 ZeRO-3 的 all-gather param 通信大（每层）。Megatron-DO 通信少（不切 param，靠 TP）+ 全维并行。极致性能选 Megatron。误以为 DeepSpeed 最优会误选。

> [!warning] 误区 5：compile 不重要
> torch.compile 是新一代训练主线（提算力 $\sim 1.5\times$）。FSDP2 与 compile 最佳（per-param DTensor 兼容图）。是新一代选型关键。误以为不重要会错选 DeepSpeed（compile 弱）。

> [!warning] 误区 6：三框架独立不能混
> 可部分混用：Megatron + FSDP2（DTensor 融合）、DeepSpeed + Megatron（PipeDream）。是生态融合趋势。误以为完全独立会限选型。

> [!warning] 误区 7：ZeRO-3 显存省但一定快
> ZeRO-3 显存最省（切 param）但通信大（每层 all-gather）。显存省不代表快（通信可能成瓶颈）。Megatron-DO 通信少但需 TP 配合。是显存 vs 通信的权衡。


## 8. 延伸细节

### 8.1 融合趋势

三者生态在融合：
- Megatron Core 向 DTensor/FSDP2 对齐（`get_model_state_dict` DCP 兼容）
- FSDP2 吸收 Megatron 的 TP/PP（DTensor 多维 mesh）
- DeepSpeed 与 Megatron 互转（checkpoint 转换）

是框架收敛趋势。详见 [[HF与Megatron checkpoint转换]] 与 [[checkpoint转换]]。

### 8.2 torch.compile 的选型影响

torch.compile（[[torch.compile]]）是大模型训练新主线（提算力）。FSDP2 与 compile 最佳（per-param DTensor 兼容图捕获），是新一代选型关键。Megatron 在适配（Megatron-OSS 的 compile 集成），DeepSpeed 兼容弱（flat buffer 不利图）。是新一代选型的决定因素。

### 8.3 大模型的实际选型

- DeepSeek-V3 671B：Megatron-like（自研 + TP+PP+EP+CP+DO 全维）
- Llama 3 405B：Megatron-LM（TP+PP+DP）
- 中小模型研究：FSDP2（PyTorch 原生 + compile）

是实际选型的参考。

### 8.4 内容来源

DeepSpeed ZeRO vs FSDP vs Megatron 对比整理自 DeepSpeed 文档、PyTorch FSDP 文档、Megatron-LM 文档、ZeRO 论文、社区对比文章（如 PyTorch blog "Introducing FSDP2"）。torch.compile 集成见 [[torch.compile]] 与 PyTorch compile 文档。融合趋势见 Megatron Core 与 FSDP2 的 DTensor 对齐。

---
相关: [[框架对比]] | [[ZeRO]] | [[FSDP2]] | [[FSDP]] | [[Megatron distributed optimizer]] | [[DeepSpeed]] | [[DTensor]] | [[DeviceMesh]] | [[torch.compile]] | [[3D parallelism]] | [[gradient bucket与通信重叠]] | [[overlap strategy]] | [[HF与Megatron checkpoint转换]] | [[checkpoint转换]] | [[Megatron-LM]]
