---
aliases: [Megatron-LM, Megatron, Megatron LM]
---

# Megatron-LM

> **所属章节**: [[并行范式]]
> **所属模块**: [[07-分布式与并行计算]]
> **难度**: 高（大模型训练框架的"教科书"，TP+PP+DP 工业标准实现）

## 1. 一句话定义

**Megatron-LM** 是 NVIDIA 团队（Shoeybi 等，2019 起）开源的**大语言模型训练框架**，也是**张量并行（TP）/ 流水线并行（PP）/ 数据并行（DP）三维并行的参考实现**。它把"单层权重按维度切到多卡（ColumnParallel/RowParallel）""层段跨 stage 流水""数据跨副本复制"三件事整合成一套可在数千卡上训万亿参数模型的工程框架，论文级的 [[3D parallelism]] 架构几乎都直接叫"Megatron-style"。它和 [[Fully Sharded Data Parallel|FSDP]]/[[ZeRO (DeepSpeed)|ZeRO]] 是两条不同的省显存路线：**FSDP/ZeRO 用"分片存储 + all-gather 临时聚合"省显存，每层仍全量参数参与计算；Megatron 用"真·切分权重，每卡只算自己那块"省显存，每层只在边界做一次 AllReduce**。当前工业界大模型预训练常把两者结合（Megatron 的 TP/PP + FSDP 的 ZeRO-3 数据并行分片）。

> [!note] Megatron-LM 是什么层面的东西
> - **不是**一个像 PyTorch 那样的通用深度学习框架，而是**建立在 PyTorch 之上的大模型训练"参考实现/工程库"**：它提供 `ColumnParallelLinear`/`RowParallelLinear`/`RowParallelEmbedding` 等切分层、PP 的 `PipelineSchedule`、跨 rank 的 RNG/loss 聚合、checkpoint 格式等，你用它来"搭"一个 3D 并行的训练脚本。
> - **论文血统**：Megatron 1（2019，TP）/Megatron 2（2021，PP+selective recomputation）/Megatron 3（2022， interleaved schedule + 拓扑感知）逐代演进，是大模型并行训练的"标准教材"，DeepSpeed/Megatron-DeepSpeed/vLLM 的 TP 推理都源自它。
> - **和 [[FSDP]] 的关系**：FSDP 解决"数据并行维度上的显存"（参数全量参与计算但分片存），Megatron 解决"模型并行维度上的显存"（参数本身被切，每卡只算一片）。两者正交、可叠加。批注"提到了 FSDP 就应该讲一讲 Megatron"正是因为两者是 LLM 训练省显存的**两大主流路线**，对比着学才完整。

## 2. 为什么需要它（动机与背景）

[[Distributed Data Parallel|DDP]] 假设每卡装得下完整模型 + 优化器状态。一旦模型大到单卡装不下（70B FP16 权重就 140GB，单卡 80GB 根本放不下；加上 AdamW 状态/梯度训练态要 ~3×），DDP 直接 OOM。这时有三条路：

1. **[[FSDP]] / [[ZeRO (DeepSpeed)|ZeRO-3]]**：参数仍全量参与每层计算，只是**分片存储**，用到时 all-gather 聚合全量 → 算 → 丢。显存 $O(P/N)$，但每层都要 all-gather + reduce-scatter，**通信量与参数量同阶**。
2. **张量并行（[[Tensor Parallel|TP]]）**：参数**真的被切**，每卡只存只算自己那 $1/N$ 片，每层只在边界做一次 AllReduce 聚合激活。显存 $O(P/N)$，通信量与"激活"同阶（比参数小），但**要求 NVLink 级互联**（每层都通信）。
3. **流水线并行（[[Pipeline Parallel|PP]]）**：层段切到各 stage，stage 间点对点传激活。省显存但有"气泡"。

Megatron-LM 的贡献是把 2、3 和数据并行**整合成一套可训、可扩展、通信与拓扑匹配的框架**：

- **TP 解决"单层太大"**：一个 70B 模型的某个 FFN 权重就 2GB，TP=8 后每卡 256MB，单层装得下；
- **PP 解决"层数太多"**：80 层分到 8 stage，每 stage 10 层，显存进一步降；
- **DP 解决"提吞吐"**：上述 TP+PP 组成一个"模型副本"，再用 DDP 复制若干份切数据；
- **互联匹配**：TP 的每层 AllReduce 走节点内 NVLink（300GB/s），PP/DP 的跨 stage/跨副本通信走节点间 IB（200~400Gbps），**通信频率与互联带宽匹配**，不让慢互联拖垮频繁通信。

没有 Megatron 这套"切权重 + 边界 AllReduce + 拓扑匹配"的工程，千亿参数模型在通用框架（PyTorch DDP）下要么 OOM、要么通信慢到不可用。

## 3. 核心概念详解

### 3.1 张量并行的两种基本切法（Megatron 的"积木"）

Megatron 把线性层 $Y = XW$（$X\in\mathbb{R}^{B\times d_\text{in}}$，$W\in\mathbb{R}^{d_\text{in}\times d_\text{out}}$）切成两种基本算子：

| 算子 | 切哪一维 | 每卡算什么 | 输出聚合 | 典型用途 |
|---|---|---|---|---|
| **ColumnParallelLinear**（列切） | $W$ 按**输出维** $d_\text{out}$ 切 N | $Y_i = X W_i$，$W_i\in\mathbb{R}^{d_\text{in}\times d_\text{out}/N}$ | **AllGather** 拼完整 $Y=[Y_1,\dots,Y_N]$ | FFN 升维层、QKV 投影 |
| **RowParallelLinear**（行切） | $W$ 按**输入维** $d_\text{in}$ 切 N | $Y_i = X_i W_i$，$X_i\in\mathbb{R}^{B\times d_\text{in}/N}$，$W_i\in\mathbb{R}^{d_\text{in}/N\times d_\text{out}}$ | **AllReduce** $\sum_i Y_i$ | FFN 降维层、attention 输出投影 |

- **列切**：每卡算的是"完整输入 × 部分输出列"，结果沿输出维拼起来即可（AllGather），不需要相加；
- **行切**：每卡算的是"部分输入 × 部分权重 = 部分结果"，需要把各卡的部分结果**求和**（AllReduce）才得到完整 $Y$。

### 3.2 Megatron MLP：列切 + 行切的"夹心"组合

标准 Transformer MLP：$Y = \text{GELU}(X W_1)\, W_2$（$W_1$ 升维 $d\to 4d$，$W_2$ 降维 $4d\to d$）。朴素 TP 会在 $W_1$ 后 AllGather、$W_2$ 后再 AllReduce，**两次通信**。Megatron 的关键优化：

- $W_1$ 用 **ColumnParallel**：按输出维切 N，每卡算 $X W_{1i}$（$d \to 4d/N$）；
- **GELU 每卡独立算**：因为列切后每卡持有的是**输出的不同列**，非线性 GELU 是逐元素操作，各卡对自己的列独立做即可，**无需通信**；
- $W_2$ 用 **RowParallel**：按输入维切 N（与 $W_1$ 的列切天然对齐——$W_1$ 切出的 $4d/N$ 列正好喂给 $W_2$ 的 $4d/N$ 行），每卡算 $\text{GELU}(X W_{1i}) W_{2i}$（$4d/N \to d$）；
- **只在 $W_2$ 后做一次 AllReduce** $\sum_i$ 得完整 $Y$。

整个 MLP **前向只一次 AllReduce、反向只一次 AllReduce**。这是 Megatron 的精髓——**用列切+行切的几何对齐把通信压到层边界**，而不是每次矩阵乘都通信。详见 [[Tensor Parallel]] §3.2。

### 3.3 Megatron Attention：多头天然切

- 多头注意力（MHA）的"多头"本身就是一种天然分片：**头数按 N 切**，每卡算 $h/N$ 个头；
- QKV 投影用 ColumnParallel（按输出维切，正好按头切），输出投影用 RowParallel；
- attention 内部（softmax/dropout）各卡独立，无需通信；
- 输出投影后 **AllReduce** 聚合；
- **硬约束**：头数必须能被 TP size 整除（如 32 头 TP=8 每卡 4 头；GQA 还要满足 KV 头数也能整除）。

### 3.4 Embedding：RowParallel + 并行采样

- 词表嵌入 $V \times d$ 按 $V$ 切（RowParallelEmbedding），每卡持有 $V/N$ 行；
- 前向 lookup 时各卡只有自己那部分词表的 embedding，没命中的位置是 0，**AllReduce 求和**后得到完整 embedding；
- LM head（输出层）同理，但 Megatron 用"并行交叉熵"避免 AllGather 整个 logits（$V$ 太大）：每卡只算自己那 $V/N$ 个词的 logit，分段算 softmax 分母，再 AllReduce 聚合——**通信量从 $O(BV)$ 降到 $O(B)$**。

### 3.5 流水线并行（PP）：层段 + schedule

TP 解决单层，PP 解决层多：

- 模型按层切成 $P$ 个 **stage**，每 stage 在一张/一组卡上；
- 前向激活按 stage 顺序流（stage 0 → 1 → ... → P-1），反向梯度反向流；
- 朴素 schedule 有"气泡"（前面的 stage 等后面的算完才能反向）；
- Megatron 用 **1F1B（one forward one backward）schedule**：每个 stage 算完一个 forward 立刻算一个 backward，把气泡压到最小；
- **Interleaved schedule**（Megatron 3）：每个 stage 不再持有连续层，而是持有"交错"的层段（如 stage 0 持有层 0,8,16...），让 stage 间更紧凑、气泡更小，代价是更多点对点通信。

详见 [[Pipeline Parallel]]。

### 3.6 数据并行（DP）+ ZeRO

- TP+PP 组成一个"模型副本"，再用 DDP 复制 $D$ 份切数据提吞吐；
- Megatron 的 DP 维度上可叠 ZeRO-1/2（分片优化器状态/梯度），新版本也支持 ZeRO-3 风格的参数分片；
- 三维 $(D, T, P)$ rank 网格，每卡属于三个通信域：TP 域（节点内）、PP 域（跨节点）、DP 域（跨节点），互不干扰。详见 [[3D parallelism]]。

### 3.7 RNG 与 loss 聚合的"分布式正确性"

切到多卡后，很多"全局量"要小心：

- **Dropout RNG**：TP 切后每卡算不同头，dropout 必须每卡独立 RNG 才不重复；DP 维度上各副本要保证 dropout 不同（否则等于没正则）；Megatron 用 `get_cuda_rng_tracker` 按 (TP rank, DP rank, layer) 派生独立 RNG 状态，保证可复现且不重复。
- **Loss 聚合**：各 DP rank 算的是自己 batch 的 loss，要 AllReduce(MEAN) 得全局平均 loss 才能正确打印/调参；TP 维度上 loss 本就是分片算的，也要聚合。
- **梯度同步**：DP 维度各副本梯度 AllReduce(MEAN) 保持参数一致（同 [[Distributed Data Parallel|DDP]]）；TP/PP 维度各卡只更新自己那片权重，不需要同步（因为本来就不重叠）。

### 3.8 拓扑感知的 rank 排布

Megatron 启动时按硬件拓扑排 rank：**TP 组放同一节点内**（用 NVLink，因为每层 AllReduce 频繁，必须高带宽）、**PP/DP 组跨节点**（用 IB，通信稀疏）。这是把"通信频率"和"互联带宽"匹配的关键，否则 TP 跨节点会慢到不可用。详见 [[NCCL通信拓扑]]。

## 4. 数学原理 / 公式

### 4.1 列切的等价性

$Y = XW$，$W = [W_1, W_2, \dots, W_N]$ 按输出维切（$W_i \in \mathbb{R}^{d_\text{in}\times d_\text{out}/N}$）：

$$
Y = X[W_1,\dots,W_N] = [XW_1, \dots, XW_N] = [Y_1,\dots,Y_N]
$$

每卡算 $Y_i = XW_i$，AllGather 拼起来 $\Leftrightarrow$ 全量算一次。**注意**：列切要求每卡都有**完整输入 $X$**（输入不切），所以列切前如果 $X$ 是行切的结果，要先 AllGather。

### 4.2 行切的等价性

$Y = XW$，$X = [X_1, \dots, X_N]$ 按输入维切，$W = \begin{bmatrix} W_1 \\ \vdots \\ W_N \end{bmatrix}$ 对应切：

$$
Y = XW = \sum_{i=1}^{N} X_i W_i = \sum_{i=1}^{N} Y_i
$$

每卡算 $Y_i = X_i W_i$，AllReduce $\sum_i$ $\Leftrightarrow$ 全量算一次。

### 4.3 MLP 通信量

设 MLP 隐层 $H=4d$，TP size $N$，batch $B$，seq $S$：

- 输入 $X\in\mathbb{R}^{BS\times d}$，输出 $Y\in\mathbb{R}^{BS\times d}$；
- 列切+行切组合后，**前向通信 = 一次 AllReduce，量 $= BS\cdot d$（激活大小）**；
- 反向同样一次 AllReduce，量 $= BS\cdot d$；
- 对比朴素（每个矩阵乘都通信）$= 2 \cdot BS\cdot 4d$，Megatron 省 4×。

**关键**：TP 通信量与"激活"同阶（$BSd$），**与参数量 $d^2$ 无关**——这就是 TP 能 scale 到大模型的原因：模型越大参数越多，但激活通信不变大太多。而 FSDP 的通信量与参数量同阶（每层 all-gather 全量参数）。

### 4.4 显存账

设模型 $P$ 参数、TP=$T$、PP=$P_p$、DP=$D$、ZeRO 切优化器状态：

| 项 | 每卡显存 |
|---|---|
| 权重 | $\dfrac{P}{T \cdot P_p}$（TP+PP 切权重，DP 不切） |
| 梯度 | $\dfrac{P}{T \cdot P_p}$（同上，DP 可再 ZeRO-2 切） |
| 优化器状态 | $\dfrac{K \cdot P}{T \cdot P_p \cdot D}$（$K=8$ for AdamW fp32；ZeRO-1/3 切 DP 维） |
| 激活 | $\dfrac{\text{单 stage 激活}}{T}$（TP 切激活；配 [[gradient checkpointing|重算]] 进一步降） |

70B 模型、TP=8、PP=4、DP=8、bf16 权重+fp32 AdamW：权重 $140\text{GB}/(8\cdot4)\approx 4.4$GB/卡，优化器状态 $1120\text{GB}/(8\cdot4\cdot8)\approx 4.4$GB/卡——总显存可控，单卡 80GB 富余跑得起来。

## 5. 代码示例

```python
# megatron_style_mlp.py —— 用 Megatron 风格的列切+行切实现一个 TP MLP（最小示意）
# torchrun --nproc_per_node=2 megatron_style_mlp.py
import os, torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def init():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

init()
rank, world = dist.get_rank(), dist.get_world_size()

class ColumnParallelLinear(nn.Module):
    """按输出维切：每卡只持有输出维的 1/N 列权重"""
    def __init__(self, in_features, out_features):
        super().__init__()
        assert out_features % world == 0
        self.local_out = out_features // world
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))  # 注意 Linear 存 (out,in)
        nn.init.normal_(self.weight, std=0.02)
    def forward(self, x):
        # x: (B, in) —— 每卡需有完整输入
        return x @ self.weight.t()  # (B, local_out)

class RowParallelLinear(nn.Module):
    """按输入维切：每卡只持有输入维的 1/N 行权重"""
    def __init__(self, in_features, out_features):
        super().__init__()
        assert in_features % world == 0
        self.local_in = in_features // world
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        nn.init.normal_(self.weight, std=0.02)
    def forward(self, x_local):
        # x_local: (B, local_in) —— 每卡只算自己那片输入
        y_local = x_local @ self.weight.t()  # (B, out)
        # AllReduce 把各卡部分和加起来 = 完整结果
        dist.all_reduce(y_local, op=dist.ReduceOp.SUM)
        return y_local  # (B, out) —— 现在每卡都有完整结果

class MegatronMLP(nn.Module):
    """W1 列切 → GELU(本地) → W2 行切 → AllReduce"""
    def __init__(self, d, h=4*d):
        super().__init__()
        self.w1 = ColumnParallelLinear(d, h)        # 升维,列切
        self.act = nn.GELU()
        self.w2 = RowParallelLinear(h, d)           # 降维,行切
    def forward(self, x):
        # x: (B, d), 每卡都有完整输入
        mid = self.act(self.w1(x))   # (B, h/N) —— 列切后每卡持有输出的不同列,GELU 本地做
        # 注意: mid 已经是"按输出维切"的状态,正好喂给行切的 w2(它要按输入维切的输入)
        y = self.w2(mid)             # (B, d) —— AllReduce 后每卡有完整结果
        return y

d = 64
model = MegatronMLP(d, h=4*d).cuda()
# 注意:TP 下不要套 DDP(Megatron 自管 TP 通信),这里只是示意层结构
x = torch.randn(8, d).cuda()
y = model(x)
if rank == 0: print("output shape:", tuple(y.shape), " TP size =", world)

dist.destroy_process_group()
```

> [!warning] 上面是手写示意，生产用 Megatron 的 `tensor_parallel.layers.ColumnParallelLinear`
> 真正的 Megatron 库还处理了：输入是否需要先 AllGather（`input_is_parallel` 参数）、bias 的切分、初始化的跨 rank 一致性、backward 的梯度 AllReduce 等。自己手写容易踩坑，工程上直接用 Megatron-LM / Megatron-DeepSpeed 的现成层。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[Tensor Parallel]]（Megatron 的层内切分就是这个）、[[Pipeline Parallel]]（PP schedule）、[[3D parallelism]]（Megatron 是其参考实现）、[[NCCL backend]]（TP 的 AllReduce、PP 的 P2P 都走它）、[[nn.Module]]（切分层继承自它）。
- **下游（应用）**: 大模型预训练（GPT/Llama/Qwen 等千亿参数模型工业训练的事实标准框架之一）；衍生 Megatron-DeepSpeed（与 DeepSpeed ZeRO 结合）；vLLM 等推理引擎的 TP 推理也用 Megatron 的列切/行切思想。
- **对比 / 易混**:
  - **Megatron(TP) vs [[FSDP]](ZeRO-3)**：见 §1 与下表。两者正交可叠加。
  - **Megatron-LM vs Megatron-DeepSpeed**：前者是 NVIDIA 原版（TP/PP 强、ZeRO 弱），后者是微软 fork，把 DeepSpeed 的 ZeRO-3 与 Megatron 的 TP/PP 融合，万亿参数训练主力。
  - **Megatron(TP) vs [[Tensor Parallel|通用 TP]]**：Megatron 的 TP 是"列切+行切夹心减通信"的工程实现，是 TP 的工业标准版本。

### Megatron TP vs FSDP 对比表

| 维度 | Megatron TP | FSDP（ZeRO-3） |
|---|---|---|
| 切的对象 | **权重本身**（每卡只算一片） | 权重分片**存储**，计算时 all-gather 全量 |
| 每层每卡算的 | 自己那 $1/N$ 权重对应的输出 | 全量权重（聚合后） |
| 通信量/层 | AllReduce 激活 $\propto BSd$ | all-gather 参数 + reduce-scatter 梯度 $\propto d^2$ |
| 通信频率 | 每层 2 次（前向+反向） | 每层 4 次（前向 gather + 反向 gather + reduce-scatter） |
| 通信 vs 参数 | **与激活同阶，不随模型变大** | 与参数同阶，模型越大通信越重 |
| 互联要求 | **必须 NVLink**（每层都通信） | 可走 IB（通信没那么频繁） |
| 扩展上限 | TP size 通常 $\le 8$（受 NVLink 节点内） | 可跨多节点（ZeRO-3 是数据并行维，能扩展到数百卡） |
| 适合场景 | 单层大、节点内 | 模型大但单层不算特别大、跨节点 |

> [!tip] 工业实践：两者结合
> 超大模型（如 405B、万亿）训练用 **TP（节点内 8 卡 NVLink）+ PP（跨节点流水）+ DP/ZeRO-3（跨节点数据并行）**。TP 把单层切到节点内 8 卡（吃 NVLink 带宽），ZeRO-3 在 DP 维度再分片优化器状态/参数（吃 IB 带宽，跨多节点）。这是 Megatron-DeepSpeed / NeMo 的标准配方。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 Megatron = TP** → Megatron 是 TP+PP+DP 的**整套框架**，TP 只是它最出名的部分。PP 的 1F1B/interleaved schedule、分布式 RNG、并行交叉熵、checkpoint 格式都是它的核心贡献。
> 2. **以为 TP 和 FSDP 是竞争二选一** → 两者正交：TP 切模型维（节点内）、FSDP/ZeRO 切数据维（跨节点），大模型训练常叠加。批注"提到 FSDP 就该讲 Megatron"正是要补全这条对比线。
> 3. **TP size 任意设** → 受头数整除约束（MHA 头数 / GQA 的 KV 头数都要能被 TP 整除），且受 NVLink 限制通常 $\le 8$，跨节点 TP 性能崩塌。
> 4. **列切前忘了输入要对齐** → ColumnParallel 要求每卡有完整输入 $X$；若上游是 RowParallel 的输出（已是"部分和"），要先 AllReduce 成完整结果再喂下游，否则数值错。Megatron 的 `input_is_parallel` 参数控制这个。
> 5. **TP 下套 DDP** → 错。TP 维度各卡权重不重叠，不需要（也不能）梯度同步；只有 DP 维度才套 DDP。新手容易把 TP 当 DDP 用导致通信错乱。
> 6. **dropout RNG 不分卡** → 各卡用同一 RNG 会产生相同 dropout mask，等于没正则，且 TP 切后各头 dropout 重复；必须用 Megatron 的 `get_cuda_rng_tracker` 按维度派生独立 RNG。
> 7. **loss 直接打印本 rank 的** → TP/DP 下各 rank 的 loss 是分片的，要 AllReduce(MEAN) 才是全局 loss；直接打印会看到错的（偏小或偏大）数值。
> 8. **PP 用朴素 schedule 不开 1F1B** → 气泡占一半以上时间，吞吐腰斩；至少用 1F1B，大模型用 interleaved。
> 9. **checkpoint 存的是分片权重灌不回普通模型** → Megatron 存的是 TP/PP 分片格式，要转成"full checkpoint"（聚合所有分片）才能给单卡/非 Megatron 模型加载，有专门的转换脚本。
> 10. **以为 Megatron 能省计算** → TP 不省 FLOPS（总计算量不变，只是分摊到 N 卡），只省显存；PP 同样不省计算还有气泡。省显存≠省算力。

## 8. 延伸细节

### 8.1 并行交叉熵（Parallel Cross-Entropy）

LM head 输出 $V$ 维 logits（$V$ 可能 15 万+），算 softmax 交叉熵时若每卡 all-gather 全量 logits 会爆通信。Megatron 把 $V$ 切到各卡，每卡算自己那 $V/N$ 个词的 $\exp$ 并求和，再 AllReduce 求全局分母，最后算 log —— **通信量从 $O(BV)$ 降到 $O(B)$**。这是 Megatron 训千亿词表模型不爆通信的关键技巧。

### 8.2 Selective Recomputation（选择性重算）

[[gradient checkpointing|全重算]]省激活但代价是全部前向重算。Megatron 提出 **selective recomputation**：只重算那些"激活大、计算小"的子层（如 attention 的 softmax/dropout 中间激活），保留"计算大、激活小"的（如 GELU 后的矩阵）。这样用 1.04× 的前向算力换 5× 的激活显存，比全重算更划算。

### 8.3 Interleaved Pipeline Schedule

1F1B 还有气泡（$P-1$ 个 forward 的空窗）。Megatron 3 把每 stage 的连续层换成"交错多段"：stage $i$ 持有层 $i, i+P, i+2P, \dots$，让 stage 间接力更紧凑，气泡压到 $\frac{P-1}{m \cdot L}$（$m$ 为每 stage 的虚拟段数）。代价是 stage 间点对点通信次数变多，但每次传的激活更小，对 IB 友好。

### 8.4 Megatron 与 FSDP2 的融合趋势

PyTorch FSDP2 基于 [[DTensor]]，把"分片"变成张量属性，可统一表示 TP/PP/DP 的切分。未来 Megatron-style 的 TP 切分有望直接用 DTensor 表达，与 FSDP 自然组合，不再需要两套独立的层实现。这是 PyTorch 团队与 NVIDIA 合流的方向。

### 8.5 何时该用 Megatron 而非 FSDP

- **单层不大、模型不算特别大（如 7B~30B）**：FSDP 足够，简单（一行 `FSDP(model)`），跨节点扩展好；
- **单层大、模型数百亿到万亿（如 70B+、MoE）**：单层权重过大，FSDP 的 all-gather 全量参数通信太重，上 TP（Megatron）切层内更划算；
- **有 NVLink 机器**：TP 吃 NVLink 带宽，节点内 TP=8 是甜点；
- **只有 IB 跨节点**：DP/PP/ZeRO 走 IB，TP 别跨节点。

---
相关: [[并行范式]]、[[Tensor Parallel]]、[[Pipeline Parallel]]、[[3D parallelism]]、[[Fully Sharded Data Parallel]]、[[ZeRO (DeepSpeed)]]、[[Distributed Data Parallel]]、[[NCCL backend]]、[[NCCL通信拓扑]]、[[gradient checkpointing]]、[[overlap strategy]]
