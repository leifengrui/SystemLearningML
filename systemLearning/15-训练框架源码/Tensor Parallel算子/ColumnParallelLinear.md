# ColumnParallelLinear

> **所属章节**: [[Tensor Parallel算子]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: ColumnParallelLinear / 列并行线性层 / column-parallel / 输出维切分 / shard output dim / Shard(1) 线性层
> **难度**: 中（需懂 [[Tensor Parallel]] + [[AllGather]]/[[all-reduce]] + [[Megatron Core目录与执行链路]]）


## 1. 一句话定义

**ColumnParallelLinear** 是 [[Megatron-LM]] 的 **TP 切分线性层**之一——把权重 $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ 沿**输出维（列维 dim=1）**切成 $T$ 片，每 rank 持 $W_i \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}/T}$，输入 $x$ **replicate**（每 rank 完整），前向每 rank 算自己的输出列 $y_i = x W_i$（**前向无通信**，因各 rank 产不同列），反向算 $\text{grad}_x = \sum_i \text{grad}_{y_i} W_i^\top$ 需 **all-reduce**（因 $x$ 是 replicate，grad 需跨 rank 求和）。用 [[DTensor]] 表达即 $W$ 的 placement `[Shard(1)]`、$x$ `[Replicate]`、$y$ `[Shard(1)]`（各 rank 持列片）。它是 attention 的 QKV 投影、MLP 的 gate/up 投影的标准算子——与 [[RowParallelLinear]]（输入维切分，前向 all-reduce、反向无通信）配对组成"col→计算→row"块（前向只在 row 末 all-reduce 一次，反向只在 col 末 all-reduce 一次），是 [[Tensor Parallel]] 的核心实现。Megatron 常把 QKV **合并**成一权重再列切（`query_key_value`），与 HuggingFace 的三分开权重（`q_proj`/`k_proj`/`v_proj`）需转换（见 [[HF与Megatron checkpoint转换]]）。

> [!note] 三句话定位
> - **是什么**：权重沿输出列维切分，输入 replicate，前向各 rank 算自己列片（无通信），反向 grad_input all-reduce。
> - **为什么**：attention/MLP 的 QKV/gate_up 需算力切分（每 rank 算部分 head/FFN），列切让每 rank 独立算自己片，前向不通信，通信推到配对的 row 末尾减半。
> - **与 [[RowParallelLinear]] 关系**：col（输出切，前向无通信）+ row（输入切，前向 all-reduce）配对。col 的反向 all-reduce = row 的前向 all-reduce 的对称，通信总量平衡。


## 2. 为什么需要它（动机与背景）

### 2.1 TP 需算力切分

[[Tensor Parallel]] 把大 matmul 切到多 GPU 算（算力切分）。attention 的 QKV 投影 $W_Q \in \mathbb{R}^{d \times d}$ 大（参数 $\propto d^2$），单卡算慢。切到 $T$ 卡，每卡算 $1/T$。

### 2.2 列切让前向不通信

QKV 投影输出 $y = xW$。若沿**列（输出）维**切 $W$，每 rank 算 $y_i = xW_i$（不同列），input $x$ replicate（每 rank 完整）。各 rank 产不同输出列，**无需通信**（各算各的）。vs 行切（输入切）需 all-reduce 合部分和。列切前向省通信。

### 2.3 与 row 配对减通信

attention 块：QKV（col，前向无通信）→ attention（各 rank 独立 head）→ O（row，前向 all-reduce）。整个块前向只在 O 末 all-reduce 一次。反向只在 QKV 末 all-reduce 一次。通信减半（vs 每算子都通信）。

### 2.4 head 独立性适配

attention 的多头特性：每 head 独立。列切 $W_Q$ 恰好让每 rank 持整数 head（如 $d=4096$，$T=8$，每 rank 4 head）。每 rank 算自己 head 的 attention，无需跨 rank。列切与 head 划分天然契合。

### 2.5 GQA 的列切

GQA（K/V head 少）的列切按 head group（每 rank 持整数 group）。$W_K$/$W_V$ 列切后每 rank 持 $d_K/T$ 列（head group）。是 [[Tensor Parallel]] 对 GQA 的适配。

### 2.6 取代手写 attention 切分

手写 attention 切分（各 rank 算部分 head，后 all-reduce）等价于 col QKV + row O。ColumnParallelLinear 把切分抽象成算子，复用于 MLP/任意线性层。是 Megatron TP 的算子化。


## 3. 核心概念详解

### 3.1 权重切分与 placement

```
W: [d_in, d_out]  (完整)
切 T 片沿列 (dim=1):
  rank 0: W_0 = W[:, 0 : d_out/T]
  rank 1: W_1 = W[:, d_out/T : 2*d_out/T]
  ...
  rank T-1: W_{T-1}
每 rank 持 [d_in, d_out/T]

DTensor 表达:
  W: [Shard(1)] 沿 tp mesh
  x: [Replicate] (每 rank 完整)
  y = x @ W: placement [Shard(1)] (各 rank 持列片)
```

### 3.2 前向（无通信）

```python
class ColumnParallelLinear:
    def forward(self, x):
        # x: [B, d_in] (replicate, 每 rank 完整)
        # self.weight: [d_in, d_out/T] (本 rank 列片)
        y = x @ self.weight  # [B, d_out/T] (本 rank 输出列片)
        return y  # 各 rank 不同列片, 无通信
```

每 rank 算自己输出列片，无需通信（各产不同列）。结果 $y$ 是 `[Shard(1)]`（各 rank 持列片）。下游若需完整 $y$ 需 all-gather（但通常下游算子直接消费分片，如 attention 各 rank 用自己 head）。

### 3.3 反向（all-reduce grad_input）

```python
    def backward(self, grad_y):
        # grad_y: [B, d_out/T] (本 rank 列片的 grad)
        # grad_weight = x^T @ grad_y  (本 rank 自己算, 无通信)
        grad_weight = x.T @ grad_y  # [d_in, d_out/T] (本 rank weight grad)
        # grad_x = grad_y @ weight^T  (部分和, 需 all-reduce)
        grad_x_partial = grad_y @ self.weight.T  # [B, d_in]
        grad_x = all_reduce(grad_x_partial, tp_group)  # 合 (因 x 是 replicate)
        return grad_x, grad_weight
```

反向：$\text{grad}_W$ 每 rank 算自己片（无通信，因 $W$ 是 shard）。$\text{grad}_x$ 是部分和（因 $x$ 是 replicate，各 rank 的 $\text{grad}_{y_i} W_i^\top$ 需求和）→ all-reduce。是 col 的反向通信。

### 3.4 与 RowParallelLinear 的配对

```
attention 块:
  QKV = ColumnParallelLinear(d, 3d)   # 前向无通信, 反向 all-reduce
  attn = attention(Q, K, V)            # 各 rank 独立 head
  O = RowParallelLinear(d, d)          # 前向 all-reduce, 反向无通信
  
通信时序:
  forward:  QKV(无) -> attn -> O(all-reduce)  # 只 O 末一次
  backward: grad_O(无) <- attn <- grad_QKV(all-reduce)  # 只 QKV 末一次
  
通信对称: col 反向 all-reduce = row 前向 all-reduce 的镜像
```

配对让整个块通信只在两端各一次。是 TP 通信优化的关键。详见 [[RowParallelLinear]]。

### 3.5 `gather_output` 参数

```python
ColumnParallelLinear(d, d_out, gather_output=True)
# gather_output=True: 前向后 all-gather 把列片合完整 y (replicate)
#   用途: 下游需完整 y (如非 TP 算子)
# gather_output=False (默认): 输出保持分片 [Shard(1)]
#   用途: 下游是 row parallel (消费分片)
```

`gather_output=True` 前向加 all-gather（合完整），用于下游非 TP 算子。默认 False（保持分片，下游 row 消费）。

### 3.6 QKV 合并

Megatron 把 Q/K/V 三权重合并成一 $W_{\text{qkv}} \in \mathbb{R}^{d \times 3d}$（按 `[Q; K; V]` concat 列），再列切。每 rank 持 $[d, 3d/T]$（含 Q/K/V 各 $d/T$ 列）。vs HuggingFace 三分开（`q_proj`/`k_proj`/`v_proj`）。转换需 concat/split（见 [[HF与Megatron checkpoint转换]]）。GQA 的 K/V head 少，合并后按 head group 切（不能等分）。

### 3.7 与 Sequence Parallel 的配合

[[Sequence Parallel]] 把 norm 区的 activation 沿 sequence 维分片。col 前向先 all-gather 激活（Shard→Replicate）再算。是 SP + TP 的组合。col 的反向 reduce-scatter 把 grad 激活切回。详见 [[Sequence Parallel]]。

### 3.8 类继承层次

```python
# megatron/core/tensor_parallel/layers.py
class ColumnParallelLinear(_ColumnParallelLinear):
    pass
class _ColumnParallelLinear(nn.Linear):
    def __init__(self, input_size, output_size, ...):
        # 切 weight 沿列, 每 rank 持片
        self.weight = Parameter(shard(weight, dim=1))
```

继承 `nn.Linear`（用 PyTorch autograd），但 `forward`/`backward` 加 TP 通信（hook 或自定义 `backward`）。

### 3.9 bias 的处理

bias $b \in \mathbb{R}^{d_{\text{out}}}$ 也列切（每 rank $b_i \in \mathbb{R}^{d_{\text{out}}/T}$）。$y_i = xW_i + b_i$。bias 小，通常 replicate 也可（但列切与 weight 对齐）。`skip_bias_add` 参数（融合 kernel 内部加 bias）。

### 3.10 与 DTensor 的对照

用 [[DTensor]] 表达：$W$ `[Shard(1)]` 沿 tp mesh、$x$ `[Replicate]`、$y$ `[Shard(1)]`。DTensor op `x @ W` 自动推导 placement 与通信（前向无，反向 all-reduce grad_x）。ColumnParallelLinear 是 DTensor 的手写等价（Megatron 历史先于 DTensor）。新代码可用 DTensor 替代。


## 4. 数学原理 / 公式

### 4.1 前向

$$y_i = x W_i + b_i, \quad W_i = W[:, i\cdot d/T:(i+1)\cdot d/T]$$

$x$ replicate，每 rank 算 $y_i$（不同列）。无通信。

### 4.2 反向

$$\text{grad}_{W_i} = x^\top \text{grad}_{y_i} \quad (\text{每 rank 独立, 无通信})$$

$$\text{grad}_x = \sum_{i=0}^{T-1} \text{grad}_{y_i} W_i^\top \quad (\text{all-reduce 合})$$

$\text{grad}_x$ 各 rank 算部分和 $\text{grad}_{y_i} W_i^\top$，all-reduce 合。

### 4.3 通信量

前向无通信。反向 all-reduce $\text{grad}_x$，通信 $\propto B \cdot d_{\text{in}}$（grad_x size）。$L$ 层 col 算子 $L$ 次反向 all-reduce。

### 4.4 与 RowParallel 的通信对称

| | 前向 | 反向 |
|---|---|---|
| ColumnParallel | 无通信 | all-reduce (grad_x) |
| RowParallel | all-reduce (合 y) | 无通信 |

配对块的通信对称：col 反向 = row 前向。总通信平衡（每块前向+反向各一次 all-reduce）。

### 4.5 算力切分

每 rank 算 $B \cdot d_{\text{in}} \cdot d_{\text{out}}/T$ FLOP（列切，算力 $1/T$）。总 $\sum = B \cdot d_{\text{in}} \cdot d_{\text{out}}$（= 全局，算力切分）。


## 5. 代码示例（可选）

### 5.1 Megatron ColumnParallelLinear 用法

```python
from megatron.core.tensor_parallel import ColumnParallelLinear
# QKV 投影 (合并 QKV, 列切)
qkv = ColumnParallelLinear(d_model, 3*d_model, bias=True, gather_output=False)
# 前向: x [B, d] (replicate) -> qkv_out [B, 3d/T] (本 rank 列片, 含 Q/K/V 各 d/T)
qkv_out = qkv(x)  # 无通信
# 拆 Q/K/V (本 rank)
q, k, v = split(qkv_out, ...)  # 各 [B, d/T]
```

### 5.2 DTensor 等价

```python
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor
W_dt = distribute_tensor(W, tp_mesh, [Shard(1)])  # 列切
x_dt = distribute_tensor(x, tp_mesh, [Replicate()])  # replicate
y_dt = x_dt @ W_dt  # DTensor op, 自动推导 [Shard(1)], 前向无通信, 反向 all-reduce grad_x
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Tensor Parallel]]（TP 概念）、[[Megatron Core目录与执行链路]]（执行时序）、[[AllGather]]/[[all-reduce]]（通信原语）、[[DTensor]]/[[Shard]]（placement 抽象）、[[nn.Linear]]/[[nn.Module]]（基础）、[[ATen与c10]]（底层 matmul）。
- **下游（应用）**: attention 的 QKV 投影、MLP 的 gate/up 投影、lm_head（列切 logits）、[[Sequence Parallel]]（配合）、[[HF与Megatron checkpoint转换]]（QKV 合并/拆）、[[RowParallelLinear]]（配对）。
- **对比 / 易混**:
  - **ColumnParallel vs [[RowParallelLinear]]**：col 列切（输出维，前向无通信反向 all-reduce）；row 行切（输入维，前向 all-reduce 反向无通信）。配对使用。详见 [[RowParallelLinear]]。
  - **ColumnParallelLinear vs [[DTensor]] `[Shard(1)]`**：DTensor `[Shard(1)]` 是抽象（声明 + op 自动通信），ColumnParallelLinear 是手写实现（Megatron 历史）。两者等价，DTensor 是新主线。
  - **ColumnParallelLinear 的 QKV 合并 vs HuggingFace 三分开**：Megatron `query_key_value`（合并列切）vs HF `q_proj`/`k_proj`/`v_proj`（分开 replicate）。转换需 concat/split。见 [[HF与Megatron checkpoint转换]]。


## 7. 常见误区与易错点

> [!warning] 误区 1：ColumnParallel 前向 all-reduce
> 错。ColumnParallel 前向**无通信**（各 rank 算自己输出列片，产不同列）。all-reduce 在**反向**（grad_x 求和）。误以为前向 all-reduce 会与 RowParallel 混。

> [!warning] 误区 2：列切与行切混淆
> "列"指输出维（dim=1，weight 的列方向），"行"指输入维（dim=0）。ColumnParallel 切输出维（每 rank 持不同输出列），RowParallel 切输入维（每 rank 持不同输入行）。勿按"weight 矩阵的行列"直觉反了。

> [!warning] 误区 3：GQA 的 QKV 合并能等分
> GQA 的 K/V head 少（如 8 个 vs Q 的 32）。合并 `query_key_value` 后 Q 部分列多、K/V 少。按 head group 切（每 rank 持整数 group），不能简单等分列。GQA 转换需 GQA-aware。

> [!warning] 误区 4：gather_output 默认 True
> 默认 False（保持分片，下游 row 消费）。若下游非 TP 算子需完整 y，设 True（加 all-gather）。误用默认会导致下游 shape 错。

> [!warning] 误区 5：ColumnParallel 反向无通信
> 错。col 反向需 all-reduce（grad_x）。因 x 是 replicate，各 rank 的 grad_x 部分和需合。误以为反向无通信会漏 all-reduce（grad 错）。

> [!warning] 误区 6：bias 不切
> bias $b$ 沿输出维，与 weight 一起列切（每 rank $b_i$）。若 bias replicate 而输出列切，$y_i = xW_i + b$（b 完整）会维度不匹配（$b \in \mathbb{R}^{d_{\text{out}}}$ vs $y_i \in \mathbb{R}^{d_{\text{out}}/T}$）。bias 必须列切。

> [!warning] 误区 7：ColumnParallel 比 DTensor 快
> ColumnParallel 与 DTensor `[Shard(1)]` 通信等价（同 all-reduce），性能相当。DTensor 是新主线（声明式 + compile 友好），ColumnParallel 是手写历史。极致场景手写可能微优（定制），但 DTensor 是趋势。


## 8. 延伸细节

### 8.1 融合 QKV 与 bias_add

Megatron 常用融合 kernel 算 QKV（`FusedQKV`）+ bias add（`skip_bias_add=True`，bias 在下游融合 kernel 内加）。是 [[fused kernel]] 的实战。TE 的 `LayerNormLinear` 融合 norm+QKV+bias。

### 8.2 与 transformer engine 集成

Megatron Core 可用 TE 的 `Linear` 层（内部 fused），配合 TP。TE 层封装 col/row 切分。详见 [[Megatron Core目录与执行链路]] 8.1。

### 8.3 padding 的等大要求

若 $d_{\text{out}} \bmod T \neq 0$（如 vocab 不整除 TP），需 padding 补到整除。每 rank shard 等大（NCCL all-reduce 需等量）。见 [[HF与Megatron checkpoint转换]] 的 padding 处理。

### 8.4 内容来源

ColumnParallelLinear 整理自 Megatron-LM 论文 v1（"Efficient Large-Scale Language Model Training on GPU Clusters"）、`megatron/core/tensor_parallel/layers.py` 源码、`mappings.py`。与 RowParallelLinear 的配对与通信对称见论文。与 DTensor 的对照见 [[DTensor]]。

---
相关: [[Tensor Parallel算子]] | [[Tensor Parallel]] | [[RowParallelLinear]] | [[Megatron Core目录与执行链路]] | [[Sequence Parallel]] | [[DTensor]] | [[AllGather]] | [[all-reduce]] | [[reduce-scatter]] | [[HF与Megatron checkpoint转换]] | [[Megatron-LM]] | [[nn.Module]] | [[ATen与c10]] | [[fused kernel]] | [[ncu (Nsight Compute)]]
