# 3D parallelism

> **所属章节**: [[并行范式]]
> **所属模块**: [[07-分布式与并行计算]]
> **难度**: 高（需懂 DP/TP/PP 三者 + 通信层次 + 拓扑）

## 1. 一句话定义

**3D parallelism（三维并行）** = [[Data Parallel|DP]] + [[Tensor Parallel|TP]] + [[Pipeline Parallel|PP]] 三种并行范式的**组合**：DP 切数据、TP 切层内、PP 切层间，三维正交，使任意大模型能跨任意多卡/节点训练。每维匹配不同通信层次——**TP 用节点内 NVLink（每层 AllReduce 激活）**、**PP 用节点间 IB（stage 间点对点）**、**DP 用节点间 IB（反向后 AllReduce 梯度）**。rank 组织为 3D 网格 $(D\times T\times P)$，每卡显存 $\sim1/(D\cdot T\cdot P)$ 模型。再加 ZeRO 切优化器状态，训万亿参数可行。它是大模型训练（Megatron-LM/DeepSpeed）的标准架构，[[policy training (PT)]] 的并行底座。

> [!note] 三维度的分工
> | 维度 | 切什么 | 通信 | 层次 | 适合 |
> |---|---|---|---|---|
> | **DP** | 数据 | AllReduce 梯度 | 节点间 IB | 提吞吐 |
> | **TP** | 层内权重 | AllReduce 激活 | 节点内 NVLink | 单层大 |
> | **PP** | 层段 | 点对点激活 | 节点间 IB | 层多 |
> 三维正交，通信不冲突（不同集合通信域）。

## 2. 为什么需要它（动机与背景）

单一并行不够训超大模型：
1. **TP 不够**：TP 受 NVLink 限（$\le8$），且节点数多时 TP 无法跨节点（慢）；
2. **PP 不够**：PP 切层，但单层仍可能单卡装不下（需 TP 层内再切）；
3. **DP 不够**：DP 每卡全模型，大模型单卡装不下（需 TP/PP 切模型）；
4. **三维互补**：TP 切层内（节点内）、PP 切层间（跨节点）、DP 切数据（跨节点），覆盖所有显存维度 + 提吞吐；
5. **互联匹配**：不同通信层次匹配不同并行维，避免慢互联拖累；
6. **ZeRO 加成**：3D 切权重/激活/梯度，ZeRO 进一步切优化器状态，训万亿参数。

3D parallelism 是大模型训练的"终极"并行架构，Megatron-LM/DeepSpeed 工业标准。

## 3. 核心概念详解

### 3.1 三维度的正交组合

模型参数 $P_{\text{total}}$、层 $L$、数据 batch $B$：
- **TP（$T$ 卡）**：每层权重切 $T$ 份，每卡 $1/T$ 权重/层；
- **PP（$P$ stage）**：层切 $P$ 段，每卡 $L/P$ 层；
- **DP（$D$ 副本）**：模型复制 $D$ 份（每份是 TP+PP 的一个完整模型），各算不同数据。

每卡显存：
$$
\text{mem}\approx\frac{P_{\text{total}}}{T\cdot P\cdot D}\text{（权重）}+\text{激活}+\frac{O_{\text{total}}}{T\cdot P\cdot D}\text{（优化器,无 ZeRO）}
$$

总并行度 $N=D\cdot T\cdot P$。

### 3.2 3D rank 网格

rank 组织为 3D 网格 $(D, T, P)$：
- 每卡有三个通信域：
  - **TP 域**：同 DP/PP 坐标的 $T$ 卡（节点内）；
  - **PP 域**：同 DP/TP 坐标的 $P$ 卡（跨节点）；
  - **DP 域**：同 TP/PP 坐标的 $D$ 卡（跨节点）；
- 三域集合通信互不干扰（不同 communicator）。

### 3.3 通信层次与互联映射

| 维度 | 通信 | 频率 | 互联 |
|---|---|---|---|
| **TP** | AllReduce 激活（前向+反向，每层） | 频繁 | 节点内 NVLink（~300GB/s） |
| **PP** | 点对点 send/recv（stage 间） | 稀疏 | 节点间 IB（~100GB/s） |
| **DP** | AllReduce 梯度（反向后） | 中（每步） | 节点间 IB |

映射原则：**TP 节点内**（需最快互联，因每层通信）、**PP/DP 跨节点**（通信稀疏/中频，容忍 IB）。这使 3D 充分利用集群拓扑。

### 3.4 典型配置

例：1024 卡训 175B 模型，8 卡/节点（128 节点）：
- **TP=8**（节点内，NVLink）；
- **PP=16**（跨 16 节点，IB）；
- **DP=8**（跨 8 组，IB）；
- 总 $8\times16\times8=1024$ 卡；
- 每卡显存 $\sim175\text{B}/(8\times16\times8)\approx1.7\text{B}$ 参数（FP16 ~3.4GB，加激活+优化器可装）。

### 3.5 ZeRO + 3D

ZeRO（[[Fully Sharded Data Parallel]]）在 DP 维进一步切优化器状态/梯度/权重：
- **ZeRO-1**：切优化器状态（DP 域分片）；
- **ZeRO-2**：+切梯度；
- **ZeRO-3**：+切权重（需 AllGather 算时）；
- 3D + ZeRO-3：每卡 $\sim1/(T\cdot P\cdot D^2)$ 权重（DP 维 ZeRO 再切），训万亿参数可行。

### 3.6 前向/反向的通信流

一次训练 step：
1. **DP 各副本**：不同数据；
2. **前向**：
   - 每 stage 内：TP 域 AllReduce（激活，每层）；
   - stage 间：PP 点对点 send/recv（激活，stage 边界）；
3. **反向**：对称（TP AllReduce 梯度 + PP 点对点梯度）；
4. **DP AllReduce**：各副本梯度聚合；
5. **优化器更新**：各卡更新自己负责的参数片。

## 4. 数学原理 / 公式

### 4.1 总并行度与显存

$$
N=D\cdot T\cdot P,\quad \text{mem}_{\text{weights per GPU}}=\frac{P_{\text{total}}}{N}
$$

加 ZeRO-3：$\frac{P_{\text{total}}}{T\cdot P\cdot D^2}$（DP 维再切权重）。

### 4.2 通信量

| 维度 | 每 step 通信量 |
|---|---|
| **TP** | $2L\cdot B\cdot d$（每层 AllReduce 激活，前向+反向） |
| **PP** | $2(P-1)\cdot B\cdot d$（stage 边界，前向+反向） |
| **DP** | $2\cdot P_{\text{total}}/N$（AllReduce 梯度，每卡） |

- TP 通信最大（每层），需 NVLink；
- PP/DP 较稀疏，IB 可承。

### 4.3 计算通信比

$$
\text{ratio}=\frac{\text{FLOPS}}{\text{comm}}\approx\frac{6P_{\text{total}}\cdot B/N}{\text{comm}}
$$

- 大模型（$P_{\text{total}}$ 大）比值高，并行高效；
- $N$ 大（过度切）则通信占比升，有最优 $N$。

### 4.4 气泡与吞吐

- TP 无气泡（每层算完即通信）；
- PP 气泡 $(P-1)/M$；
- DP 无气泡（数据并行）；
- 总吞吐受 PP 气泡限（最慢维度）：$\sim1-(P-1)/M$。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.distributed as dist

# ===== 3D rank 网格(概念,实际用 torch.distributed 3D communicator) =====
class Config3D:
    def __init__(self, D, T, P):
        self.D, self.T, self.P = D, T, P
        self.N = D * T * P
    def rank_to_coords(self, rank):
        """rank -> (dp, tp, pp) 坐标"""
        dp = rank // (self.T * self.P)
        rem = rank % (self.T * self.P)
        tp = rem // self.P
        pp = rem % self.P
        return dp, tp, pp
    def coords_to_rank(self, dp, tp, pp):
        return dp * (self.T * self.P) + tp * self.P + pp
    def tp_group(self, dp, pp):
        """同 dp,pp 的 T 卡(TP 域)"""
        return [self.coords_to_rank(dp, t, pp) for t in range(self.T)]
    def pp_group(self, dp, tp):
        """同 dp,tp 的 P 卡(PP 域)"""
        return [self.coords_to_rank(dp, tp, p) for p in range(self.P)]
    def dp_group(self, tp, pp):
        """同 tp,pp 的 D 卡(DP 域)"""
        return [self.coords_to_rank(d, tp, pp) for d in range(self.D)]

cfg = Config3D(D=4, T=8, P=16)
print(f"3D 配置: DP={cfg.D} x TP={cfg.T} x PP={cfg.P} = {cfg.N} 卡")
r = 100
dp, tp, pp = cfg.rank_to_coords(r)
print(f"rank {r} -> (dp={dp}, tp={tp}, pp={pp})")
print(f"  TP 域(NVLink,节点内): {cfg.tp_group(dp,pp)}")
print(f"  PP 域(IB,跨节点): {cfg.pp_group(dp,tp)}")
print(f"  DP 域(IB,跨节点): {cfg.dp_group(tp,pp)}")

# ===== 显存估算 =====
P_total = 175e9  # 175B 参数
mem_per_gpu = P_total / cfg.N * 2  # FP16 2 bytes
print(f"\n175B 模型, 3D={cfg.D}x{cfg.T}x{cfg.P}={cfg.N} 卡:")
print(f"  每卡权重显存: {mem_per_gpu/1e9:.2f} GB(FP16)")
print(f"  加激活+优化器(无 ZeRO): ~{mem_per_gpu*4/1e9:.1f} GB(需 80GB 卡)")
print(f"  加 ZeRO-3(DP 切权重+优化器): 每卡 ~{P_total/(cfg.T*cfg.P*cfg.D**2)*2*4/1e9:.2f} GB(可训更大)")

# ===== 通信量估算 =====
L, B, d = 96, 2, 12288  # GPT-3 175B
tp_comm = 2*L*B*d*4  # FP32 AllReduce 因子4
pp_comm = 2*(cfg.P-1)*B*d*4
dp_comm = 2*P_total/cfg.N*4
print(f"\n每 step 通信(概念):")
print(f"  TP(每层 AllReduce): {tp_comm/1e9:.1f} GB(需 NVLink)")
print(f"  PP(stage 间): {pp_comm/1e9:.2f} GB(IB 可承)")
print(f"  DP(梯度 AllReduce): {dp_comm/1e9:.2f} GB(IB 可承)")
print(f"3D 充分利用: TP 节点内 NVLink, PP/DP 节点间 IB")
```

> [!tip] 实际工程
> - **Megatron-LM/DeepSpeed**：3D 默认架构，配置 `pipeline_model_parallel_size`/`tensor_model_parallel_size`/`data_parallel_size`；
> - **rank 映射**：框架自动建 3D 通信域（TP/PP/DP 三个 process group）；
> - **ZeRO**：`--zero-stage=3` 配合 3D；
> - **调优**：TP=节点内 GPU 数（NVLink），PP 平衡层，DP 用剩余卡。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[Data Parallel]]、[[Tensor Parallel]]、[[Pipeline Parallel]]（三组件）、[[all-reduce]]/[[AllGather]]/点对点（通信原语）、[[NCCL通信拓扑]]、[[rank与world size]]/[[process group]]/[[torch.distributed]]。
- **下游（应用）**: [[policy training (PT)]]（PT 的并行底座）、Megatron-LM/DeepSpeed 训练框架、大模型预训练/RLHF、[[Fully Sharded Data Parallel|ZeRO]]（与 3D 组合）。
- **对比 / 易混**:
  - **3D vs 单一并行**：3D 组合三者，覆盖所有显存维度 + 互联层次；单一并行有局限（TP 受 NVLink、PP 无层内切、DP 单卡装不下）。
  - **3D vs ZeRO-3**：3D 是 DP+TP+PP（切权重/激活/数据），ZeRO-3 是 DP 维内切权重/梯度/优化器。可叠加（3D+ZeRO-3）。
  - **3D vs 2D（DP+TP）**：2D 适合单节点/小模型（无跨节点 PP），3D 适合多节点/大模型（加 PP 切层）。
  - **维度顺序**：DP/TP/PP 正交但映射到拓扑有讲究（TP 节点内、PP/DP 跨节点）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"3D 总是最优"**：错。小模型过度切增通信开销，2D/单 DP 更优。3D 适合超大模型 + 多节点。
> 2. **"维度顺序随意"**：错。TP 必须节点内（NVLink），PP/DP 跨节点（IB），乱映射则慢互联拖累。
> 3. **"TP/PP/DP 通信冲突"**：错。三者用不同通信域（process group），互不干扰。
> 4. **"3D 显存 1/N"**：对权重，但激活显存非简单 1/N（PP 存 P 个，TP 存部分激活）。
> 5. **"ZeRO 替代 3D"**：错。ZeRO 是 DP 维内分片，3D 是三维组合。叠加用。
> 6. **"PP 气泡在 3D 中消失"**：错。PP 气泡仍在，3D 不消除（需调度减）。
> 7. **"DP 域 AllReduce 梯度是全模型"**：错。3D 中每卡只算自己 TP+PP 片的梯度，DP AllReduce 是该片在 DP 域聚合（量 $P_{\text{total}}/N$）。
> 8. **"3D 配置固定"**：错。按模型大小/集群拓扑调（小模型 2D，大模型 3D，超大 +ZeRO）。
> 9. **"TP 可跨节点"**：低效。TP 每层通信，跨节点慢互联成瓶颈，应节点内。

## 8. 延伸细节

### 8.1 Megatron-LM 的 3D 实现

- 配置：`TP=p`（节点内）、`PP=pipeline_stage`（跨节点）、`DP=world_size/(TP*PP)`；
- 自动建三通信域：`tp_group`/`pp_group`/`dp_group`；
- 调度：1F1B/interleaved PP + TP 层内 + DP 数据；
- ZeRO：`--zero-stage` 配合；
- 是开源大模型训练的事实标准。

### 8.2 配置调优原则

1. **TP = 节点内 GPU 数**（NVLink，如 8）；
2. **PP = 跨节点层数**（按层均衡，如 16）；
3. **DP = 剩余卡**（提吞吐，如 8）；
4. **ZeRO**：若显存还紧，加 ZeRO-1/2/3；
5. **global batch** = DP × micro_batch × accumulation；
6. 调优目标：显存装下 + 通信开销小 + 吞吐高。

### 8.3 3D 的通信域隔离

- `torch.distributed.new_group()` 建三域；
- TP 域 AllReduce 不影响 PP/DP 域；
- 各域独立 NCCL communicator；
- 避免死锁：注意通信顺序（如 PP 先于 DP）。

### 8.4 3D + ZeRO-3 的显存

无 ZeRO：每卡 $\frac{P_{\text{total}}}{TP\cdot PP\cdot DP}$ 权重 + $\frac{O_{\text{total}}}{TP\cdot PP\cdot DP}$ 优化器（DP 域内复制优化器）；
ZeRO-1：优化器切 DP 域，每卡 $\frac{O_{\text{total}}}{TP\cdot PP\cdot DP}$（优化器分片）；
ZeRO-3：权重+梯度+优化器切 DP 域，每卡 $\frac{P_{\text{total}}}{TP\cdot PP\cdot DP^2}$（权重再切）。
训万亿参数：$TP=8,PP=64,DP=128$，ZeRO-3 → 每卡 $\sim10^9$ 参数级，可装。

### 8.5 3D 的气泡与重叠

- PP 气泡 $(P-1)/M$；
- 可在气泡时做 DP 通信（overlap）减空闲（[[overlap strategy]]）；
- interleaved PP 减气泡；
- 详见 [[overlap strategy]]/[[compute vs communication bottleneck]]。

### 8.6 3D 的历史与生态

- **Megatron-LM**（NVIDIA）：3D 标准实现；
- **DeepSpeed**：3D + ZeRO 组合；
- **Megatron-Deepspeed**：开源大模型栈；
- **Alpa/Unity**：自动并行搜索（选 DP/TP/PP 配比）；
- 3D 是当前大模型训练的主流架构，未来可能演进（如 auto-parallelism）。

### 8.7 3D 在 RLHF（PT）

- PT 训练用 3D（[[policy training (PT)]] §3.1）；
- rollout/PD 用 TP（推理，[[inference worker]]）；
- 两者 $\theta$ 同步用 [[weight sync mechanism]]；
- PT 的 3D 与 rollout 的 TP 配置可不同（训练重吞吐，推理重延迟）。

---
相关: [[并行范式]]、[[Data Parallel]]、[[Tensor Parallel]]、[[Pipeline Parallel]]、[[Fully Sharded Data Parallel]]、[[all-reduce]]、[[AllGather]]、[[NCCL通信拓扑]]、[[rank与world size]]、[[process group]]、[[torch.distributed]]、[[policy training (PT)]]、[[overlap strategy]]、[[compute vs communication bottleneck]]、[[activation memory]]
