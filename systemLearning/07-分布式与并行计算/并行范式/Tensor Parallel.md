# Tensor Parallel

> **所属章节**: [[并行范式]]
> **所属模块**: [[07-分布式与并行计算]]
> **难度**: 中高（需懂矩阵分块 + 集合通信）

## 1. 一句话定义

**Tensor Parallel（TP，张量并行）** 是把模型**单层的权重张量按维度切分到多张卡**，每卡算该层的一部分，通过 **AllReduce/AllGather** 聚合结果的并行范式。典型实现 **Megatron-LM**：线性层做**列切分**（ColumnParallel，输出维切）或**行切分**（RowParallel，输入维切），注意力多头切分到各卡。TP 使单卡装不下的大模型（70B+）能跨卡装下并训练/推理，每卡显存 ~$1/N$。与 [[Data Parallel]]（切数据、模型复制）正交：**TP 切模型、DP 切数据**。代价是**每层都需通信**（AllReduce），故需高速互联（NVLink/IB），TP 规模通常 2/4/8（受通信限）。

> [!note] TP 的核心特征
> - **切的是模型单层**（权重张量），不是数据或层段。
> - **每层都通信**（前向+反向各 AllReduce），通信频繁。
> - **显存 $1/N$**：每卡只存 $1/N$ 权重。
> - **不省计算**：总 FLOPS 不变，只是分摊到 N 卡（vs [[Pipeline Parallel]] 省显存但有气泡）。

## 2. 为什么需要它（动机与背景）

大模型训练/推理的核心障碍是**单卡显存不足**：
1. **显存墙**：70B 模型 FP16 约 140GB，单卡（80GB）装不下，更别说训练（+梯度+优化器状态 ~3×）；
2. **需切分模型**：[[Data Parallel]] 是模型复制（每卡全模型），不解决单卡装不下；TP 切模型本身；
3. **层内切分**：[[Pipeline Parallel]] 是层间切分（每卡若干层），但单层太大时仍需 TP 在层内再切；
4. **推理也需**：大模型部署（[[policy deployment (PD)]]）单卡装不下，用 TP 切多卡服务（[[inference worker]]）。

TP 通过矩阵分块把单层权重切到多卡，使任意大的单层都能跨卡装下。它是 3D parallelism（[[3D parallelism]]）中"层内"维度的并行，Megatron-LM 是标准实现。

## 3. 核心概念详解

### 3.1 矩阵切分：列切 vs 行切

线性层 $Y=XW$（$X\in\mathbb{R}^{B\times d_{\text{in}}}$，$W\in\mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}$）两种切法：

| 切法 | 切维度 | 每卡算 | 聚合 |
|---|---|---|---|
| **ColumnParallel（列切）** | $W$ 按输出维 $d_{\text{out}}$ 切 N 份 | $Y_i=XW_i$（$W_i\in\mathbb{R}^{d_{\text{in}}\times d_{\text{out}}/N}$） | AllGather $[Y_1,...,Y_N]$ |
| **RowParallel（行切）** | $W$ 按输入维 $d_{\text{in}}$ 切 N 份 | $Y_i=X_iW_i$（$X_i\in\mathbb{R}^{B\times d_{\text{in}}/N}$，$W_i\in\mathbb{R}^{d_{\text{in}}/N\times d_{\text{out}}}$） | AllReduce $\sum_i Y_i$ |

- **ColumnParallel**：每卡算部分输出维，AllGather 拼完整输出；
- **RowParallel**：每卡算部分输入维的部分结果，AllReduce 求和；
- **Megatron 组合**：MLP 用 ColumnParallel（升维）→ GELU → RowParallel（降维），中间只在 GELU 后一次 AllReduce（减通信）。

### 3.2 Megatron-LM 的 MLP 切分

标准 MLP：$Y=\text{GELU}(XW_1)W_2$（$W_1$ 升维 $d\to4d$，$W_2$ 降维 $4d\to d$）。
- $W_1$ **ColumnParallel**：按输出维切 N，每卡算 $XW_{1i}$（$d\to4d/N$）；
- GELU 每卡独立算（非线性，但各卡数据独立）；
- $W_2$ **RowParallel**：按输入维切 N（与 $W_1$ 列切对齐），每卡算 $\text{GELU}(XW_{1i})W_{2i}$（$4d/N\to d$）；
- **AllReduce** $\sum_i$ 得完整 $Y$。
- **通信**：整个 MLP 只一次 AllReduce（前向）+ 一次（反向），减通信。

### 3.3 注意力多头切分

- 多头注意力（MHA）天然适合 TP：**多头数按 N 切**，每卡算 $h/N$ 个头；
- 每卡独立算 QKV 投影 + attention + 输出投影；
- 输出投影用 RowParallel，AllReduce 聚合；
- **要求**：头数能被 N 整除（如 32 头 TP=8，每卡 4 头）。

### 3.4 通信模式

- **前向**：每层（MLP/Attention）结束 AllReduce 聚合（RowParallel 输出）；
- **反向**：对称 AllReduce（梯度）；
- **通信量**：每层 $O(B\cdot d)$（激活值大小），每卡；
- **频率**：每层都通信，故需低延迟高速互联（NVLink，~300GB/s）；
- **vs DP**：DP 通信是梯度（反向后一次 AllReduce），TP 通信是激活（每层），TP 频繁得多。

### 3.5 TP 的显存

| 项 | 单卡 |
|---|---|
| 权重 | $P/N$ |
| 激活 | $\sim A/N$（部分算） |
| 梯度 | $P/N$ |
| 优化器状态 | $O/N$（若 ZeRO 配合） |
| KV cache（推理） | $K/N$（按头切） |

每卡显存 ~$1/N$，使大模型能装下。

## 4. 数学原理 / 公式

### 4.1 ColumnParallel

$$
W=[W_1, W_2, ..., W_N]\text{（按输出维列切）},\quad Y=XW=[XW_1, XW_2, ..., XW_N]
$$

每卡算 $Y_i=XW_i$，AllGather 拼接 $Y=[Y_1,...,Y_N]$。

### 4.2 RowParallel

$$
X=[X_1, X_2, ..., X_N]\text{（按输入维切）},\quad W=\begin{bmatrix}W_1\\ \vdots\\ W_N\end{bmatrix},\quad Y=XW=\sum_{i=1}^{N}X_iW_i
$$

每卡算 $Y_i=X_iW_i$，AllReduce 求和 $Y=\sum_i Y_i$。

### 4.3 Megatron MLP 的通信

$$
Y=\text{AllReduce}\Big(\sum_{i=1}^{N}\text{GELU}(XW_{1i})W_{2i}\Big)
$$

- 前向一次 AllReduce；
- 反向一次 AllReduce（梯度对 $X$ 的偏导）；
- 通信量 $O(Bd)$ per layer。

### 4.4 计算通信比

$$
\text{compute}/\text{communicate}=\frac{2Bd^2/N}{2Bd}=\frac{d}{N}
$$

- $d$ 大（大模型隐维高）则比值高，TP 高效；
- $N$ 大（TP 规模大）则比值降，通信占比升；
- 故 TP 规模受 $d$ 与互联带宽限，典型 $N\le8$。

### 4.5 TP 显存

$$
\text{mem}_{\text{per GPU}}\approx\frac{P+A+G+O}{N}
$$

$P$ 权重、$A$ 激活、$G$ 梯度、$O$ 优化器状态。每卡 ~$1/N$。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.distributed as dist

# ===== Megatron 风格 TP 线性层(单进程模拟 N 卡) =====
class ColumnParallelLinear(nn.Module):
    """列切:按输出维切 N 份,每卡算部分输出"""
    def __init__(self, in_features, out_features, N=2):
        super().__init__()
        assert out_features % N == 0
        self.shards = nn.ModuleList([
            nn.Linear(in_features, out_features // N, bias=False) for _ in range(N)
        ])
    def forward(self, x):
        outs = [shard(x) for shard in self.shards]    # 各卡算 X*W_i
        return torch.cat(outs, -1)                    # AllGather 拼接(模拟)

class RowParallelLinear(nn.Module):
    """行切:按输入维切 N 份,每卡算部分输入的部分结果,AllReduce 求和"""
    def __init__(self, in_features, out_features, N=2):
        super().__init__()
        assert in_features % N == 0
        self.shards = nn.ModuleList([
            nn.Linear(in_features // N, out_features, bias=False) for _ in range(N)
        ])
    def forward(self, x):
        # x 按输入维切,各卡算 X_i*W_i,求和(AllReduce 模拟)
        chunk = x.shape[-1] // self.shards[0].in_features
        xs = torch.chunk(x, len(self.shards), dim=-1)
        outs = [shard(xi) for shard, xi in zip(self.shards, xs)]
        return sum(outs)                              # AllReduce sum

# ===== Megatron MLP: ColumnParallel -> GELU -> RowParallel(只一次 AllReduce) =====
class MegatronMLP(nn.Module):
    def __init__(self, d, d_ff, N=2):
        super().__init__()
        self.w1 = ColumnParallelLinear(d, d_ff, N)    # 升维,列切
        self.w2 = RowParallelLinear(d_ff, d, N)       # 降维,行切
        self.geluf = nn.GELU()
    def forward(self, x):
        h = self.w1(x)                                # 各卡算部分升维
        h = self.geluf(h)                             # 非线性,各卡独立
        return self.w2(h)                             # 行切 + AllReduce(只一次)

d, d_ff, B, N = 64, 256, 4, 2
mlp = MegatronMLP(d, d_ff, N)
x = torch.randn(B, d)
y = mlp(x)
print(f"Megatron MLP(TP={N}): in{x.shape} -> out{y.shape}")
print(f"每卡显存 ~1/{N}: w1 每卡 {mlp.w1.shards[0].weight.shape}, w2 每卡 {mlp.w2.shards[0].weight.shape}")
print(f"通信: 前向 1 次 AllReduce(模拟为 sum),反向 1 次")

# ===== 对比:普通 MLP(单卡) =====
class PlainMLP(nn.Module):
    def __init__(s, d, d_ff): super().__init__(); s.w1=nn.Linear(d,d_ff,bias=False); s.w2=nn.Linear(d_ff,d,bias=False); s.g=nn.GELU()
    def forward(s,x): return s.w2(s.g(s.w1(x)))
plain = PlainMLP(d, d_ff)
print(f"普通 MLP(单卡): w1 {plain.w1.weight.shape}, w2 {plain.w2.weight.shape}(TP 切分后每卡 1/{N})")
print("TP 不省计算(总 FLOPS 同),只切显存到多卡")
```

> [!tip] 实际工程
> - **Megatron-LM/DeepSpeed**：内置 ColumnParallelLinear/RowParallelLinear；
> - **TP 规模**：典型 2/4/8（受 NVLink 通信限）；
> - **头数对齐**：MHA 头数能被 TP 整除；
> - **推理 TP**：vLLM/TensorRT-LLM 支持，大模型部署用（[[inference worker]]）。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[Data Parallel]]（对照）、矩阵分块数学、[[all-reduce]]/[[AllGather]]（通信原语）、[[NCCL通信拓扑]]（底层）、[[nn.Module]]/[[state_dict与load_state_dict]]。
- **下游（应用）**: [[3D parallelism]]（TP 是其层内维度）、Megatron-LM/DeepSpeed 训练框架、[[inference worker]]/[[policy deployment (PD)]]（推理 TP）、[[sampling throughput]]（rollout 用 TP 切大模型）。
- **对比 / 易混**:
  - **TP vs [[Data Parallel]]**：TP 切模型单层（每卡部分权重，每层通信），DP 切数据（每卡全模型+不同数据，反向后通信）。正交，常组合。
  - **TP vs [[Pipeline Parallel]]**：TP 切层内（同层跨卡），PP 切层间（不同层不同卡）。TP 每层通信，PP 段间通信。
  - **TP vs [[Fully Sharded Data Parallel]]**：TP 是计算时分片（每层算部分），FSDP 是存储时分片（算时 AllGather 全层）。TP 通信是激活，FSDP 通信是权重。
  - **ColumnParallel vs RowParallel**：前者切输出维 AllGather，后者切输入维 AllReduce。Megatron 组合减通信。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"TP 省计算"**：错。总 FLOPS 不变，只分摊到 N 卡（提速度）。省的是显存。
> 2. **"TP 规模越大越好"**：错。通信随 N 增占比升，典型 $\le8$（NVLink 限）。
> 3. **"TP 不需高速互联"**：错。每层通信，普通以太网延迟扛不住，需 NVLink/IB。
> 4. **"TP 等于模型并行"**：部分对。TP 是模型并行的一种（层内），PP 也是模型并行（层间）。
> 5. **"MHA 切头随便"**：错。头数需被 N 整除，否则有的卡头多有的少，不均衡。
> 6. **"TP 通信只在反向"**：错。前向每层也 AllReduce（激活聚合）。
> 7. **"TP 与 DP 互斥"**：错。正交，常组合（DP×TP，数据切+模型切）。
> 8. **"推理不用 TP"**：错。大模型部署（PD）单卡装不下，必用 TP（[[inference worker]]）。
> 9. **"ColumnParallel 用 AllReduce"**：错。列切用 AllGather（拼输出），行切才 AllReduce（求和）。

## 8. 延伸细节

### 8.1 Megatron-LM 的通信优化

- **组合 Column+Row**：MLP 用 $W_1$列切+$W_2$行切，中间 GELU 不通信，只末端一次 AllReduce；
- **减通信次数**：朴素 TP 每线性层一次 AllReduce，Megatron 组合后每 MLP 一次；
- **注意力**：QKV 列切（多头）+ 输出行切，一次 AllReduce；
- **LayerNorm**：不切（每卡全算，输入已 AllReduce 完整）。

### 8.2 TP 在推理（PD/rollout）

- 大模型部署（70B+）单卡装不下，用 TP 切多卡服务；
- vLLM/TensorRT-LLM 支持 TP（`tensor_parallel_size`）；
- 推理 TP 与训练 TP 类似，但 KV cache 也按头切（每卡 $1/N$ KV）；
- 详见 [[inference worker]]/[[policy deployment (PD)]]/[[sampling throughput]]。

### 8.3 TP 与 ZeRO 的关系

- **TP**：计算时分片（每层算部分），通信是激活；
- **[[Fully Sharded Data Parallel|FSDP/ZeRO-3]]**：存储时分片（算时 AllGather 全层），通信是权重；
- **组合**：大模型训练常用 TP（层内）+ ZeRO（数据维），或纯 ZeRO（中小模型）；
- **选择**：TP 适合单层大（大隐维），ZeRO 适合层多（深网络）。

### 8.4 TP 的负载均衡

- 列切/行切默认均衡（均匀分维）；
- 但若某层维度不整除 N，需 padding 或非均匀切；
- MHA 头数不整除需调整（如 36 头 TP=8，不整除，改 TP=4 或 6）。

### 8.5 TP 的通信开销量化

- 每层 AllReduce 量 $2Bd$（前向+反向）；
- $L$ 层则 $2LBd$ per forward+backward；
- NVLink 带宽 ~300GB/s，IB ~100GB/s；
- 大模型（$d=8192,B=2,L=80$）每步通信 ~几 GB，需高速互联否则成瓶颈（[[compute vs communication bottleneck]]）。

### 8.6 TP 的历史与生态

- **Megatron-LM**（NVIDIA 2019）：TP 标准实现；
- **DeepSpeed**：TP + ZeRO 组合；
- **vLLM/TensorRT-LLM**：推理 TP；
- **Megatron-Deepspeed**：开源大模型训练栈；
- TP 是大模型训练/推理的基石，与 [[Pipeline Parallel]]/[[Data Parallel]] 组成 [[3D parallelism]]。

---
相关: [[并行范式]]、[[Data Parallel]]、[[Pipeline Parallel]]、[[3D parallelism]]、[[all-reduce]]、[[AllGather]]、[[NCCL通信拓扑]]、[[Fully Sharded Data Parallel]]、[[nn.Module]]、[[inference worker]]、[[policy deployment (PD)]]、[[sampling throughput]]、[[compute vs communication bottleneck]]、[[overlap strategy]]
