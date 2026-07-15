# RowParallelLinear

> **所属章节**: [[Tensor Parallel算子]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: RowParallelLinear / 行并行线性层 / row-parallel / 输入维切分 / shard input dim / Shard(0) 线性层
> **难度**: 中（需懂 [[Tensor Parallel]] + [[ColumnParallelLinear]] + [[all-reduce]]）


## 1. 一句话定义

**RowParallelLinear** 是 [[Megatron-LM]] 的 **TP 切分线性层**之一——把权重 $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ 沿**输入维（行维 dim=0）**切成 $T$ 片，每 rank 持 $W_i \in \mathbb{R}^{d_{\text{in}}/T \times d_{\text{out}}}$，输入 $x$ **沿输入维分片** $x_i \in \mathbb{R}^{B \times d_{\text{in}}/T}$（各 rank 持片），前向每 rank 算部分和 $y_i = x_i W_i$（$\sum_i y_i$ 才完整），需 **all-reduce** 合成完整 $y$；反向 $\text{grad}_{x_i} = \text{grad}_y W_i^\top$ 各 rank 独立算（**无通信**，因 $x$ 本就是分片）。用 [[DTensor]] 表达即 $W$ `[Shard(0)]`、$x$ `[Shard(1)]`（输入维分片）、$y$ `[Partial]`（all-reduce 后 `[Replicate]`）。它是 attention 的 O 投影、MLP 的 down 投影的标准算子——与 [[ColumnParallelLinear]]（输出维切分，前向无通信反向 all-reduce）配对组成"col→计算→row"块，整块前向只在 row 末 all-reduce 一次、反向只在 col 末 all-reduce 一次。RowParallel 的 `input_is_parallel` 参数控制输入是否已是分片（若 input 是 replicate 需先 `scatter`/切）。

> [!note] 三句话定位
> - **是什么**：权重沿输入行维切分，输入分片，前向各 rank 算部分和（all-reduce 合），反向 grad_input 各 rank 独立（无通信）。
> - **为什么**：attention/MLP 的 O/down 投影需把多 head/FFN 中间结果合回 $d_{\text{model}}$，行切让每 rank 算部分和后 all-reduce 合，通信推到块末减半。
> - **与 [[ColumnParallelLinear]] 关系**：col（输出切，前向无通信）+ row（输入切，前向 all-reduce）配对。row 的前向 all-reduce = col 的反向 all-reduce 的镜像，通信对称。


## 2. 为什么需要它（动机与背景）

### 2.1 TP 需算力切分且需合回

[[Tensor Parallel]] 的 attention 块：QKV（col 切，各 rank 算自己 head）→ attention → O（需把多 head 结果合回 $d_{\text{model}}$）。O 投影 $W_O \in \mathbb{R}^{d \times d}$ 大。行切让每 rank 算部分和（用自己 head 的输出片），后 all-reduce 合完整 $y$。

### 2.2 行切让前向 all-reduce 一次

O 投影：$y = \text{attn\_out} \cdot W_O$。若行切 $W_O$（输入维切），每 rank 持 $W_{O,i} \in \mathbb{R}^{d/T \times d}$，输入 $\text{attn\_out}_i \in \mathbb{R}^{B \times d/T}$（各 head 输出片）。$y_i = \text{attn\_out}_i W_{O,i}$（部分和），all-reduce 合 $y = \sum y_i$。前向只块末 all-reduce 一次。

### 2.3 与 col 配对减通信

col（前向无通信）+ row（前向 all-reduce）配对，整块前向只在 row 末一次 all-reduce。反向对称（row 无通信、col all-reduce）。通信减半。详见 [[ColumnParallelLinear]]。

### 2.4 反向无通信的便利

row 反向 $\text{grad}_{x_i} = \text{grad}_y W_i^\top$ 各 rank 独立（因 $x$ 是分片，grad_x 也分片，无需合）。无通信。与 col 反向 all-reduce 对称。配对让通信平衡。

### 2.5 输入已是分片的衔接

col 的输出是分片 `[Shard(1)]`（各 rank 持列片）。row 的输入需分片（沿输入维）。col 的输出列片恰好可作 row 的输入分片（转置视角：col 输出的列 = row 输入的行分片）。`input_is_parallel=True` 衔接。是 col→row 的天然组合。

### 2.6 取代手写合回

手写 attention 合回（各 head 输出 concat 再 $W_O$）等价于 row 切 $W_O$ 后 all-reduce。RowParallelLinear 把合回 + 切分抽象成算子。是 Megatron TP 的算子化。


## 3. 核心概念详解

### 3.1 权重切分与 placement

```
W: [d_in, d_out]  (完整)
切 T 片沿行 (dim=0):
  rank 0: W_0 = W[0 : d_in/T, :]
  rank 1: W_1 = W[d_in/T : 2*d_in/T, :]
  ...
每 rank 持 [d_in/T, d_out]

DTensor 表达:
  W: [Shard(0)] 沿 tp mesh
  x: [Shard(1)] (输入维分片, 每 rank 持 x_i [B, d_in/T])
  y = x @ W: [Partial] (各 rank 部分和) -> all-reduce -> [Replicate]
```

### 3.2 前向（all-reduce）

```python
class RowParallelLinear:
    def forward(self, x):
        # x: [B, d_in/T] (本 rank 输入片, 来自 col 输出)
        # self.weight: [d_in/T, d_out] (本 rank 行片)
        y_partial = x @ self.weight  # [B, d_out] (部分和)
        y = all_reduce(y_partial, tp_group)  # 合完整 y (replicate)
        return y
```

每 rank 算部分和 $y_i$，all-reduce 合 $y = \sum y_i$。前向通信 all-reduce。

### 3.3 反向（无通信）

```python
    def backward(self, grad_y):
        # grad_y: [B, d_out] (replicate, 因前向 all-reduce 后 y 是 replicate)
        # grad_weight = x_i^T @ grad_y  (本 rank 自己算, 无通信)
        grad_weight = x.T @ grad_y  # [d_in/T, d_out] (本 rank weight grad)
        # grad_x = grad_y @ weight^T  (本 rank 自己, 无通信)
        grad_x = grad_y @ self.weight.T  # [B, d_in/T] (本 rank 输入片 grad)
        return grad_x, grad_weight
```

反向：$\text{grad}_W$ 每 rank 算自己行片（无通信）。$\text{grad}_{x_i} = \text{grad}_y W_i^\top$ 各 rank 独立（因 $x$ 是分片，grad_x 也分片）。无通信。与 col 反向 all-reduce 对称。

### 3.4 与 ColumnParallelLinear 的配对

```
attention 块:
  QKV = ColumnParallelLinear(d, 3d)   # 前向无通信, 反向 all-reduce
  attn = attention(Q, K, V)            # 各 rank 独立 head
  O = RowParallelLinear(d, d)          # 前向 all-reduce, 反向无通信
  
通信时序:
  forward:  QKV(无) -> attn -> O(all-reduce)  # 只 O 末一次
  backward: grad_O(无) <- attn <- grad_QKV(all-reduce)  # 只 QKV 末一次
  
通信对称: row 前向 all-reduce = col 反向 all-reduce 的镜像
```

详见 [[ColumnParallelLinear]] 3.4。配对让整块通信减半。

### 3.5 `input_is_parallel` 参数

```python
RowParallelLinear(d, d_out, input_is_parallel=True)
# input_is_parallel=True (默认): 输入已是分片 (来自 col 输出), 直接用
# input_is_parallel=False: 输入是 replicate, 需先 scatter/切到本 rank 片
#   用途: row 单独用 (无前置 col), 输入完整
```

`input_is_parallel=True` 衔接 col 输出（已是分片）。`False` 需先切（额外通信或本地切片）。

### 3.6 MLP 的 down 投影

MLP 块：gate_up（col 切，前向无通信）→ activation（各 rank 独立）→ down（row 切，前向 all-reduce）。down 投影 $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$ 行切，输入 $h_i \in \mathbb{R}^{B \times d_{\text{ff}}/T}$（各 rank FFN 中间片）。$y_i = h_i W_{\text{down},i}$（部分和），all-reduce 合。与 attention O 同理。

### 3.7 与 Sequence Parallel 的配合

[[Sequence Parallel]]：row 的反向把 grad 激活 reduce-scatter 切回（Shard）。row 反向无 all-reduce（grad_x 分片），但 SP 配合 reduce-scatter。详见 [[Sequence Parallel]]。

### 3.8 类继承层次

```python
# megatron/core/tensor_parallel/layers.py
class RowParallelLinear(_RowParallelLinear):
    pass
class _RowParallelLinear(nn.Linear):
    def __init__(self, input_size, output_size, input_is_parallel=True, ...):
        # 切 weight 沿行, 每 rank 持片
        self.weight = Parameter(shard(weight, dim=0))
```

继承 `nn.Linear`，`forward`/`backward` 加 TP 通信。

### 3.9 bias 的处理

bias $b \in \mathbb{R}^{d_{\text{out}}}$ 不切（输出维完整，因 $y$ all-reduce 后是完整）。$y = \sum y_i + b$。bias 在 all-reduce 后加（或融合 kernel 内加）。`skip_bias_add` 参数。

### 3.10 与 DTensor 的对照

用 [[DTensor]]：$W$ `[Shard(0)]`、$x$ `[Shard(1)]`（输入维分片）、$y$ `[Partial]`（all-reduce 后 `[Replicate]`）。DTensor op `x @ W` 自动推导通信（前向 all-reduce、反向无）。RowParallelLinear 是 DTensor 的手写等价。


## 4. 数学原理 / 公式

### 4.1 前向

$$y = \sum_{i=0}^{T-1} x_i W_i + b, \quad W_i = W[i\cdot d/T:(i+1)\cdot d/T, :]$$

$x_i$ 各 rank 输入片。$y_i = x_i W_i$ 部分和，all-reduce 合 $y$。

### 4.2 反向

$$\text{grad}_{W_i} = x_i^\top \text{grad}_y \quad (\text{每 rank 独立, 无通信})$$

$$\text{grad}_{x_i} = \text{grad}_y W_i^\top \quad (\text{每 rank 独立, 无通信})$$

$\text{grad}_y$ 是 replicate（因前向 all-reduce 后 $y$ replicate）。各 rank 独立算自己片的 grad。无通信。

### 4.3 通信量

前向 all-reduce $\propto B \cdot d_{\text{out}}$（$y$ size）。反向无通信。$L$ 层 row 算子 $L$ 次前向 all-reduce。

### 4.4 与 ColumnParallel 的通信对称

| | 前向 | 反向 |
|---|---|---|
| ColumnParallel | 无通信 | all-reduce (grad_x) |
| RowParallel | all-reduce (合 y) | 无通信 |

配对块前向+反向各一次 all-reduce。通信平衡。

### 4.5 算力切分

每 rank 算 $B \cdot d_{\text{in}}/T \cdot d_{\text{out}}$ FLOP（行切，算力 $1/T$）。总 $\sum = B \cdot d_{\text{in}} \cdot d_{\text{out}}$（= 全局，算力切分）。


## 5. 代码示例（可选）

### 5.1 Megatron RowParallelLinear 用法

```python
from megatron.core.tensor_parallel import RowParallelLinear, ColumnParallelLinear
# attention 块
qkv = ColumnParallelLinear(d, 3*d, gather_output=False)  # 前向无通信
# attn: 各 rank 独立 head
attn_out = attention(qkv(x))  # [B, d/T]
o = RowParallelLinear(d, d, input_is_parallel=True, bias=True)  # 前向 all-reduce
y = o(attn_out)  # [B, d] (replicate, all-reduce 合)
```

### 5.2 DTensor 等价

```python
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor
W_dt = distribute_tensor(W, tp_mesh, [Shard(0)])  # 行切
x_dt = distribute_tensor(x, tp_mesh, [Shard(1)])  # 输入维分片
y_dt = x_dt @ W_dt  # [Partial] -> all-reduce -> [Replicate], 前向 all-reduce 反向无
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Tensor Parallel]]、[[ColumnParallelLinear]]（配对）、[[Megatron Core目录与执行链路]]、[[all-reduce]]/[[reduce-scatter]]、[[DTensor]]/[[Shard]]、[[nn.Linear]]/[[nn.Module]]、[[ATen与c10]]。
- **下游（应用）**: attention 的 O 投影、MLP 的 down 投影、[[Sequence Parallel]]（配合）、[[HF与Megatron checkpoint转换]]（权重映射）、[[ColumnParallelLinear]]（配对）。
- **对比 / 易混**:
  - **RowParallel vs [[ColumnParallelLinear]]**：row 行切（输入维，前向 all-reduce 反向无）；col 列切（输出维，前向无反向 all-reduce）。配对使用。
  - **RowParallelLinear vs [[DTensor]] `[Shard(0)]` + `[Partial]`**：DTensor 是抽象（声明 + op 自动通信），RowParallelLinear 是手写实现。等价，DTensor 是新主线。
  - **RowParallel 的 `input_is_parallel` vs col 的 `gather_output`**：`input_is_parallel=True` 衔接 col 分片输出；`gather_output=False` 保持 col 分片输出。两者衔接是 col→row 天然组合。


## 7. 常见误区与易错点

> [!warning] 误区 1：RowParallel 前向无通信
> 错。RowParallel 前向**all-reduce**（合部分和）。反向**无通信**。误以为前向无会与 col 混。

> [!warning] 误区 2：行切与列切混淆
> "行"指输入维（dim=0，weight 的行方向）。RowParallel 切输入维（每 rank 持不同输入行）。勿按直觉反。见 [[ColumnParallelLinear]] 误区 2。

> [!warning] 误区 3：input_is_parallel 默认 False
> 默认 True（衔接 col 分片输出）。若 row 单独用（输入 replicate），设 False（先切）。误用默认会导致输入维度错。

> [!warning] 误区 4：RowParallel 反向 all-reduce
> 错。row 反向无通信（grad_x 分片，各 rank 独立）。all-reduce 在前向。误以为反向 all-reduce 会漏（grad 错）或与 col 混。

> [!warning] 误区 5：bias 行切
> bias $b \in \mathbb{R}^{d_{\text{out}}}$ 不切（输出维完整）。在 all-reduce 后加。若 bias 行切（切输入维）无意义（bias 是输出维参数）。误切会维度错。

> [!warning] 误区 6：RowParallel 比 DTensor 快
> 与 [[DTensor]] `[Shard(0)]` + `[Partial]` 通信等价（同 all-reduce）。性能相当。DTensor 是新主线。

> [!warning] 误区 7：配对 col→row 的输入维度不匹配
> col 输出是分片 `[B, d_out/T]`（列片）。row 输入需 `[B, d_in/T]`。col 的 $d_{\text{out}}$ = row 的 $d_{\text{in}}$（如 attention 的 $3d$ → $d$，需 attention 内部降维）。衔接需 shape 对齐。


## 8. 延伸细节

### 8.1 融合 O + bias_add

Megatron 用融合 kernel 算 O（`FusedO`）+ bias add。TE 的 `Linear` 层封装。是 [[fused kernel]] 实战。

### 8.2 sequence parallel 的 reduce-scatter

SP 配合 row：row 反向后 grad 激活 reduce-scatter 切回 sequence 分片。是 SP 把 row 的"无通信反向"变成"reduce-scatter 反向"（换显存省）。详见 [[Sequence Parallel]]。

### 8.3 与 transformer engine

TE 的 `Linear` 层封装 row 切分，内部融合。详见 [[Megatron Core目录与执行链路]] 8.1。

### 8.4 内容来源

RowParallelLinear 整理自 Megatron-LM 论文 v1、`megatron/core/tensor_parallel/layers.py` 源码。与 ColumnParallelLinear 的配对与通信对称见论文。与 DTensor 的对照见 [[DTensor]]。

---
相关: [[Tensor Parallel算子]] | [[Tensor Parallel]] | [[ColumnParallelLinear]] | [[Megatron Core目录与执行链路]] | [[Sequence Parallel]] | [[DTensor]] | [[all-reduce]] | [[reduce-scatter]] | [[HF与Megatron checkpoint转换]] | [[Megatron-LM]] | [[nn.Module]] | [[ATen与c10]] | [[fused kernel]]
