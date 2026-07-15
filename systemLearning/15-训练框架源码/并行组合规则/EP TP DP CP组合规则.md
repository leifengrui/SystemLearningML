# EP TP DP CP组合规则

> **所属章节**: [[并行组合规则]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: EP TP DP CP 组合规则 / 并行组合 / rank mapping / 通信组划分 / 重叠约束 / 多维并行组合 / 5D 并行
> **难度**: 高（需懂 [[EP]] + [[Tensor Parallel]] + [[Distributed Data Parallel|DP]] + [[Context Parallel|CP]] + [[Pipeline Parallel|PP]] + [[alpha-beta性能模型]]）


## 1. 一句话定义

**EP TP DP CP 组合规则** 是 [[Megatron-LM]] 把 **5 个并行维度**（EP expert 并行、TP 张量并行、DP 数据并行、CP context 并行、PP 流水线并行）**叠加到同一组 GPU rank** 上的**坐标映射与通信组划分规则**——总 $N$ 个 rank 分配给各维度（$N = T \cdot P \cdot D \cdot E \cdot C$，$T$=TP、$P$=PP、$D$=DP、$E$=EP、$C$=CP），每个 rank 有 5 维坐标 $(t, p, d, e, c)$，按**层次化 rank mapping**（TP 内层用 NVLink 高带宽、EP/CP 中层、DP/PP 外层跨节点 IB）建 5 组 process group（tp_group / ep_group / dp_group / cp_group / pp_group），各维度的通信只在对应 group 内发生（TP all-reduce 沿 tp_group、EP all-to-all 沿 ep_group、DP grad all-reduce 沿 dp_group、CP ring attention 沿 cp_group、PP send/recv 沿 pp_group）；**重叠约束**规定哪些维度的通信能 overlap（DP grad 的 bucket async 与 backward overlap、PP send/recv async 与计算 overlap）哪些不能（TP all-reduce 同层依赖难 overlap、EP all-to-all 与 expert 算有依赖）。是 [[3D parallelism]]（TP+PP+DP）扩展到 5D（加 EP+CP）的组合学，决定大 MoE 长序列模型的并行配置与性能。

> [!note] 三句话定位
> - **是什么**：5 维（TP/PP/DP/EP/CP）rank 坐标映射 + 5 组 process group + 各维通信在对应 group + overlap 约束（能/不能藏）。
> - **为什么**：大 MoE 长序列模型需多维并行叠加（单维不够），组合规则定如何分配 rank、通信组、overlap，避免冲突与低效。
> - **与 [[3D parallelism]] 关系**：3D 是 TP+PP+DP 的基础组合，本笔记扩展到 5D（加 EP 切 expert、CP 切 sequence）。3D 是子集。


## 2. 为什么需要它（动机与背景）

### 2.1 单维不够

大 MoE 模型（如 DeepSeek-V3 671B，37B 激活，256 expert）+ 长序列（128k context）：单维并行不够。TP 切 weight 但 expert 多需 EP；DP 切 batch 但单 rank 装不下 expert 需 EP；长序列 activation 大需 CP。需多维叠加。

### 2.2 rank 分配与通信组

$N$ 个 rank 分给 5 维。每 rank 有 5 维坐标。建 5 组 process group。各维通信在对应 group。若不规划，通信组乱（TP all-reduce 跑跨节点慢、EP all-to-all 跑错 group）。需规则定 mapping。

### 2.3 层次化匹配硬件

GPU 集群层次：节点内 NVLink（高带宽 $\beta_{\text{nvlink}}$）+ 节点间 IB（低带宽 $\beta_{\text{ib}}$）。TP 的 all-reduce 频繁（每层），放节点内 NVLink。EP 的 all-to-all 较大，可跨节点 IB。DP grad all-reduce 跨节点 IB（可 overlap）。PP 跨节点。CP ring 跨节点。层次化匹配硬件特性。见 [[alpha-beta性能模型]] 与 [[NCCL通信拓扑]]。

### 2.4 overlap 约束

各维通信能否 overlap 由依赖决定：DP grad（跨层独立，能 overlap）、PP send/recv（跨 stage，能 overlap）、TP all-reduce（同层依赖，难）、EP all-to-all（与 expert 算有依赖，部分）、CP ring（与 attention 有依赖，部分）。需规则定 overlap 时序，避免串行瓶颈。

### 2.5 EP 与 TP 的交互

EP 切 expert（每 rank 持部分 expert），TP 切每个 expert 的 weight（每 expert 的 weight 切 T）。EP+TP：每 rank 持部分 expert 的部分 weight（$E \cdot T$ 切分）。两者正交但需 mapping 协调（expert 在 EP 维、weight 在 TP 维）。

### 2.6 CP 与 DP 的区分

CP 切 sequence 维（长序列 activation 分片，ring attention 通信），DP 切 batch 维（不同 batch，grad all-reduce）。两者都"数据并行"但切不同维（sequence vs batch）。CP 的通信是 ring attention（前向），DP 是 grad all-reduce（反向）。需区分。


## 3. 核心概念详解

### 3.1 5 维 rank 坐标

```
N = T * P * D * E * C  (总 rank 数)
每 rank 坐标 (t, p, d, e, c), t∈[0,T), p∈[0,P), d∈[0,D), e∈[0,E), c∈[0,C)

例: T=8, P=4, D=4, E=2, C=2, N=8*4*4*2*2=512
每 rank 坐标 (t,p,d,e,c)
```

5 维坐标。global rank 映射到 5 维坐标。是组合的数学基础。

### 3.2 层次化 rank mapping

```
层次化 (内层用 NVLink, 外层 IB):
  最内: TP (节点内 NVLink, all-reduce 频繁)
  次: EP (节点内或跨节点 IB, all-to-all)
  次: CP (跨节点 IB, ring attention)
  外: DP (跨节点 IB, grad all-reduce)
  最外: PP (跨节点 IB, send/recv)

mapping: global_rank = ((((p * D + d) * E + e) * C + c) * T + t)
  或类似层次化公式, 让 tp 维最内 (同节点 NVLink)
```

层次化让 TP 的 group 在同节点（NVLink），PP/DP 跨节点（IB）。匹配硬件带宽。见 [[NCCL通信拓扑]]。

### 3.3 5 组 process group

```python
from megatron.core import parallel_state as ps
ps.initialize_model_parallel(
    tensor_model_parallel_size=T,        # TP
    expert_model_parallel_size=E,         # EP
    expert_tensor_parallel_size=...,      # EP 内 TP (expert 的 TP)
    pipeline_model_parallel_size=P,      # PP
    context_parallel_size=C,              # CP
)
# 建 5 组 group:
#   tp_group: 同 (p,d,e,c) 的 T 个 rank (节点内 NVLink)
#   ep_group: 同 (p,d,t,c) 的 E 个 rank (expert 维)
#   cp_group: 同 (p,d,e,t) 的 C 个 rank (sequence 维)
#   dp_group: 同 (p,e,t,c) 的 D 个 rank (batch 维, grad all-reduce)
#   pp_group: 同 (d,e,t,c) 的 P 个 rank (stage 维, send/recv)
```

每维一 group。通信在对应 group。`ps` 自动建（层次化 mapping）。

### 3.4 各维通信的归属

| 维 | 通信 | group | 时机 |
|---|---|---|---|
| TP | all-reduce（row 后） | tp_group | 前向 + 反向 |
| EP | all-to-all（token 派发） | ep_group | 前向 + 反向（MoE 层） |
| CP | ring attention | cp_group | 前向 + 反向 |
| DP | grad all-reduce | dp_group | 反向 |
| PP | send/recv（activation 跨 stage） | pp_group | 前向 + 反向 |

各维通信在对应 group，不串。

### 3.5 EP + TP 的交互

```
EP: E 个 rank 各持 N_expert/E 个 expert
TP: 每 expert 的 weight 切 T 片
EP+TP: 每 rank 持 N_expert/E 个 expert, 每 expert 持 1/T weight
  切分: N_expert * weight_size -> N_expert/E * weight_size/T per rank
  ep_group: E 个 rank (expert 维)
  tp_group: T 个 rank (每 expert 的 weight 维)
  expert_tensor_parallel: EP 内的 TP (expert weight 的 T 切)
```

EP 切 expert，TP 切每 expert weight。两者正交。expert_tensor_parallel_size 是 EP 内的 TP（每 expert 的 TP 维）。

### 3.6 CP + DP 的区分

```
CP: 切 sequence 维 (长序列 activation 分片)
  cp_group: C 个 rank, 持 S/C 的 sequence
  通信: ring attention (前向, Q/K/V 跨 CP rank)
DP: 切 batch 维 (不同 batch)
  dp_group: D 个 rank, 持 B/D 的 batch
  通信: grad all-reduce (反向)
两者正交: sequence vs batch
```

CP 切 sequence（前向 ring attention），DP 切 batch（反向 grad all-reduce）。两者都"数据"但切不同维。

### 3.7 PP 的正交

PP 切层维（stage）。与 EP/TP/DP/CP 正交。PP 的 send/recv 沿 pp_group。PP 可与 EP/TP/CP/DP 全组合（5D + PP = 实际 5 维含 PP）。

### 3.8 overlap 约束

| 维 | 通信 | overlap 性质 | 能否 overlap |
|---|---|---|---|
| DP | grad all-reduce | 跨层独立 | ✅（bucket async 与 backward） |
| PP | send/recv | 跨 stage | ✅（async 与计算） |
| TP | all-reduce | 同层依赖 | ❌（下一层需完整 input） |
| EP | all-to-all | 与 expert 算有依赖 | 部分（dispatch 与算可部分 overlap） |
| CP | ring attention | 与 attention 有依赖 | 部分（ring 与算可部分 overlap） |

DP/PP 能 overlap（独立）。TP 难（依赖）。EP/CP 部分。是 [[overlap strategy]] 的约束。

### 3.9 通信量的叠加

总通信 = TP all-reduce + EP all-to-all + CP ring + DP grad + PP P2P。各维独立（不同 group）。总 wall-clock 通信 $\approx \max$（各维，若并行）或 $\sum$（若串行）。优化目标：让各维通信 overlap（藏）。见 [[compute vs communication bottleneck]]。

### 3.10 配置的选择

- TP：节点内（NVLink，T = 8/GPU数）。TP 大减 weight 显存 + 提算力，但 all-reduce 多。
- EP：expert 多时用（E = expert 数 / 每 rank expert 数）。all-to-all 大。
- CP：长序列时用（C = S / 每 rank sequence）。ring attention 通信。
- DP：剩余 rank（D = N / (T*P*E*C)）。grad all-reduce。
- PP：单 stage 装不下时用（P = model / 单 stage 容量）。send/recv。

最优配置据模型大小、expert 数、序列长、集群规模调。是 [[3D parallelism]] 的扩展决策。


## 4. 数学原理 / 公式

### 4.1 rank 数恒等

$$N = T \cdot P \cdot D \cdot E \cdot C$$

5 维乘积 = 总 rank。配置需满足（如 $T=8, P=4, D=4, E=2, C=2$, $N=512$）。

### 4.2 显存（每 rank）

- weight: $\propto \frac{N_{\text{param}}}{T \cdot E}$（TP+EP 切 weight）
- activation: $\propto \frac{B \cdot S \cdot d}{C \cdot P}$（CP+PP 切 activation，约）
- optimizer state: $\propto \frac{N_{\text{param}}}{D}$（DP 切，ZeRO-1/2）
- KV cache: $\propto \frac{B \cdot S \cdot d}{C}$（CP 切 sequence）

各维切不同显存分量。是 [[训练显存估计]] 的多维版。

### 4.3 通信量（每维）

- TP all-reduce: $\propto 2 L \cdot B \cdot S \cdot d$（每层 row 后，attn+MLP）
- EP all-to-all: $\propto 2 L_{\text{MoE}} \cdot B \cdot S \cdot d$（每 MoE 层派发+派回）
- CP ring attention: $\propto 2 L \cdot B \cdot S/C \cdot d \cdot C = 2 L \cdot B \cdot S \cdot d$（ring 跨 C rank，近似）
- DP grad all-reduce: $\propto N_{\text{param}}$（每 param 一 grad）
- PP P2P: $\propto M \cdot B \cdot S \cdot d$（micro-batch 跨 stage）

各维通信量。总 $\sum$ 或 $\max$（据 overlap）。

### 4.4 overlap 的收益

若各维通信 < 对应计算，overlap 后 wall-clock $\approx$ 计算。否则部分串行。DP/PP 能 overlap（独立），TP 难（依赖）。是 wall-clock 优化关键。

### 4.5 层次化匹配

NVLink 带宽 $\beta_{\text{nvlink}} \gg$ IB $\beta_{\text{ib}}$。TP all-reduce 频繁，放 NVLink（通信时间 $N/\beta_{\text{nvlink}}$ 小）。EP/CP/DP 跨节点 IB（$N/\beta_{\text{ib}}$ 大，但可 overlap 或消息大 $\beta$ 主导）。层次化让高频通信走高带宽。见 [[alpha-beta性能模型]]。


## 5. 代码示例（可选）

### 5.1 Megatron 5D 并行初始化

```python
from megatron.core import parallel_state as ps
ps.initialize_model_parallel(
    tensor_model_parallel_size=8,        # T=8 (节点内 NVLink)
    expert_model_parallel_size=2,        # E=2
    expert_tensor_parallel_size=4,        # EP 内 TP=4
    pipeline_model_parallel_size=4,      # P=4
    context_parallel_size=2,             # C=2
    # D = N / (T*P*E*C) = 512 / (8*4*2*2) = 4 (隐式)
)
# 建 tp/ep/cp/dp/pp 5 组 group
# 各维通信在对应 group
```

### 5.2 rank 坐标查询

```python
tp_rank = ps.get_tensor_model_parallel_rank()      # t
ep_rank = ps.get_expert_model_parallel_rank()      # e
cp_rank = ps.get_context_parallel_rank()           # c
dp_rank = ps.get_data_parallel_rank()              # d
pp_rank = ps.get_pipeline_model_parallel_rank()    # p
# 每 rank 知自己 5 维坐标
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[EP]]、[[Tensor Parallel]]、[[Distributed Data Parallel|DP]]、[[Context Parallel|CP]]、[[Pipeline Parallel|PP]]、[[3D parallelism]]、[[NCCL通信拓扑]]、[[alpha-beta性能模型]]、[[overlap strategy]]、[[compute vs communication bottleneck]]、[[Megatron Core目录与执行链路]]、[[DeviceMesh]]（PyTorch 原生 mesh）。
- **下游（应用）**: 大 MoE 长序列模型训练（DeepSeek-V3 等）、[[Megatron-LM]] 配置、[[MoE token dispatcher]]（EP 维）、[[Sequence Parallel]]（TP 内）、[[Megatron distributed optimizer]]（DP 维）。
- **对比 / 易混**:
  - **EP TP DP CP vs [[3D parallelism]]**：3D 是 TP+PP+DP，本笔记是 5D（加 EP+CP）。3D 是子集。大 MoE + 长序列才需 5D。
  - **EP vs TP**：EP 切 expert（每 rank 不同 expert），TP 切每 expert weight（每 rank 同 expert 的 weight 片）。EP+TP：每 rank 部分 expert 的部分 weight。正交。
  - **CP vs DP**：CP 切 sequence（前向 ring attention），DP 切 batch（反向 grad all-reduce）。都"数据"但切不同维。
  - **rank mapping vs [[DeviceMesh]]**：rank mapping 是 Megatron 手写坐标，DeviceMesh 是 PyTorch 原生 mesh（多维）。概念等价，DeviceMesh 是新主线。


## 7. 常见误区与易错点

> [!warning] 误区 1：5 维随便分配 rank
> 需层次化（TP 内层 NVLink，PP/DP 外层 IB）。乱分配会让 TP all-reduce 跑跨节点（慢）。rank mapping 规则定层次。

> [!warning] 误区 2：EP 与 TP 冲突
> 正交。EP 切 expert，TP 切每 expert weight。EP+TP 可组合（每 rank 部分 expert 的部分 weight）。expert_tensor_parallel_size 协调。

> [!warning] 误区 3：CP 与 DP 一样
> CP 切 sequence（前向 ring attention），DP 切 batch（反向 grad all-reduce）。切不同维，通信不同。误以为一样会漏 CP 或 DP。

> [!warning] 误区 4：所有维通信能 overlap
> DP/PP 能（独立），TP 难（同层依赖），EP/CP 部分。需按 overlap 约束调时序。误以为都能会错判串行。

> [!warning] 误区 5：PP 与 EP/TP/CP/DP 冲突
> 正交。PP 切层维，可与 4 维全组合（5D 含 PP）。误以为冲突会漏配 PP。

> [!warning] 误区 6：expert_tensor_parallel 与 TP 重复
> expert_tensor_parallel 是 **EP 内的 TP**（每 expert 的 weight 切），TP 是非 expert 层的 TP。两者可能不同（expert 的 TP 维独立配置）。是 EP+TP 的细化。

> [!warning] 误区 7：CP 的 ring attention 不算通信
> CP 的 ring attention 通信 $\propto L \cdot B \cdot S \cdot d$（每层 ring 跨 C rank）。是 CP 的通信大头。误以为不算会低估通信。


## 8. 延伸细节

### 8.1 expert_tensor_parallel 的细化

EP 内的 TP（expert_tensor_parallel_size）可独立于非 expert 层的 TP。如非 expert 层 TP=8（NVLink），expert 层 EP=2 + expert TP=4（每 expert weight 切 4）。是 EP+TP 的灵活配置。

### 8.2 与 [[DeviceMesh]] 的对照

Megatron 的 rank mapping 是手写，[[DeviceMesh]] 是 PyTorch 原生多维 mesh（nested mesh 表达 5D）。两者等价，DeviceMesh 是新主线。Megatron Core 在向 DeviceMesh 迁移。详见 [[DeviceMesh]]。

### 8.3 配置的搜索

最优 5D 配置（T/P/D/E/C）是组合搜索（据模型/集群）。Megatron 提配置脚本，社区有经验值。是 [[3D parallelism]] 配置的扩展。见 [[compute vs communication bottleneck]]。

### 8.4 内容来源

EP TP DP CP 组合规则整理自 Megatron-LM `megatron/core/parallel_state.py` 源码、DeepSeek-V3 技术报告（5D 并行）、3D parallelism 论文/教程。层次化 mapping 与 overlap 约束见 Megatron 文档。与 DeviceMesh 的对照见 [[DeviceMesh]]。

---
相关: [[并行组合规则]] | [[3D parallelism]] | [[EP]] | [[Tensor Parallel]] | [[Distributed Data Parallel]] | [[Context Parallel]] | [[Pipeline Parallel]] | [[NCCL通信拓扑]] | [[alpha-beta性能模型]] | [[overlap strategy]] | [[compute vs communication bottleneck]] | [[Megatron Core目录与执行链路]] | [[DeviceMesh]] | [[MoE token dispatcher]] | [[Sequence Parallel]] | [[Megatron distributed optimizer]] | [[Megatron-LM]] | [[训练显存估计]]
