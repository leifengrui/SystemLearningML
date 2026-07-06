# FLOPs计算

> **所属章节**: [[资源核算]]
> **所属模块**: [[11-显存与内存系统]]
> **别名**: FLOP estimation / 算力估算 / compute budget / C
> **难度**: 中（需懂矩阵乘 + 前向/反向 + Transformer 结构）


## 1. 一句话定义

**FLOPs 计算（FLOP estimation，注意小写 s = operations）** 是用纸笔估算训练或推理一个模型需要多少**浮点运算次数**的方法——核心三条规则：**矩阵乘 $A_{m\times n}B_{n\times p} \approx 2mnp$ FLOPs**、**前向每个 token 每个参数约 2 FLOPs**、**训练（前向+反向）合计 $C = 6ND$**（$N$ 参数、$D$ token 数）。它是 [[缩放定律]] 的基石（$C$ 是缩放曲线的横轴）、训练成本/时间估算的输入（见 [[MFU与算术强度]]）。

> [!note] 命名辨析
> - **FLOP** = floating point operation（一次浮点运算，单数）。
> - **FLOPs** = FLOP per second（每秒浮点运算次数，衡量硬件算力，如"A100 = 312 TFLOPs"）。**注意大小写 s**：`FLOPs`（小写 s，per second）指速率；文献里 `FLOPs`（大写 S 或就写 FLOPs）也常被滥用指"总运算次数"（operation count）。本笔记遵循 CS336 记法：**FLOP** = 次数（量），**FLOPs** = 速率（硬件峰值）。为避免混淆，文中"运算次数"用 **FLOP**，"硬件峰值"用 **FLOPs** 或"峰值算力"。

> [!note] 三句话定位
> - **是什么**：估算模型训练/推理要多少次浮点运算（FLOP）。
> - **核心公式**：训练 $C = 6ND$（前向 2 + 反向 4，每参数每 token 共 6 FLOP）；矩阵乘 $2mnp$。
> - **有什么用**：$C$ 是 [[缩放定律]] 横轴、训练时间/成本公式的分子、模型对比的客观度量（不依赖硬件）。

> [!note] 解答：前向 2、反向 4 是怎么得出来的？
> 
> 从**矩阵乘 $y = Wx$**（$W\in\mathbb{R}^{m\times n}$，$N_W=mn$ 个参数）一步步推：
> 
> **前向 2 FLOP/参数/token**：
> - $y = Wx$：每个输出元素 $y_i = \sum_{j=1}^n W_{ij} x_j$，是 $n$ 次乘法 + $n$ 次加法 $\approx 2n$ FLOP。
> - 共 $m$ 个输出元素，总 $2mn = 2N_W$ FLOP → **每参数每 token 2 FLOP**（1 乘 1 加）。
> - 整模型 $N$ 个参数，单 token 前向 $\approx 2N$；$D$ 个 token → $2ND$。
> 
> **反向 4 FLOP/参数/token**：
> 反向要对每个线性层算**两个梯度**（这是反向比前向多一倍的根本原因）：
> 1. **对输入的梯度** $\frac{\partial L}{\partial x} = W^\top \frac{\partial L}{\partial y}$：矩阵乘向量，FLOP $\approx 2N_W$（和前向同形）。
> 2. **对权重的梯度** $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y}\cdot x^\top$：外积，FLOP $\approx 2N_W$。
> - 合计 $4N_W$ → **每参数每 token 4 FLOP**。
> - 整模型 $D$ token → $4ND$。
> 
> **合计**：
> 
> $$C = \text{前向} + \text{反向} = 2ND + 4ND = 6ND$$
> 
> **直觉**：前向只算一次"乘加"（出 $y$），反向要算两次"乘加"（出 $\partial L/\partial x$ 和 $\partial L/\partial W$），所以反向是前向的 2 倍、合计 3 倍 = 6。
> 
> 详细推导见 [[#4.2 前向 2 FLOP/参数/token 推导]]、[[#4.3 反向 4 FLOP/参数/token 推导]]。
## 2. 为什么需要它（动机与背景）

### 2.1 算力是大模型训练的硬约束

LLM 训练的瓶颈不是"会不会写代码"，而是"算力够不够、要算多久、花多少钱"。要回答这些，必须能精确算出**这次训练总共要做多少次浮点运算**（$C$）。$C$ 算清了，配合硬件峰值算力和 MFU（见 [[MFU与算术强度]]），就能估出训练时间 $\approx C / (\text{GPU数} \times \text{峰值} \times \text{MFU})$ 和成本。

### 2.2 FLOP 是模型规模的客观度量

不同模型在不同硬件上训练，速度不可直接比（硬件不同、batch 不同）。但**总 FLOP $C$** 是与硬件无关的客观量，只依赖模型结构（$N$）和数据量（$D$）。故 [[缩放定律]] 用 $C$ 作横轴——$C$ 把"模型 + 数据"统一成一个可比较的数。

### 2.3 为什么是 6 而不是别的数

$C=6ND$ 里的 6 不是经验值，是**推导出来的**：前向每参数每 token 2 FLOP（一乘一加），反向约前向的 2 倍 = 4 FLOP，合计 6。这个推导是 [[参数量计算]] → [[FLOPs计算]] → [[缩放定律]] 的链条核心。


## 3. 核心概念详解

### 3.1 矩阵乘法的 FLOP

> **核心规则：矩阵乘 $A_{m\times n} \times B_{n\times p}$ 需要约 $2mnp$ FLOP。**

推导：结果矩阵 $C$ 是 $m \times p$，每个元素 $C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$，要 $n$ 次乘法 + $n$ 次加法 = $2n$ FLOP。共 $m \cdot p$ 个元素，总 $2mnp$ FLOP。

- 因子 2 = 乘 + 加。部分硬件把"乘加"合一（FMA, fused multiply-add）算 1 FLOP，但深度学习惯例按 2 算（与硬件峰值标注一致，如 A100 bf16 312 TFLOPs 是按 FMA=2 算的）。
- 线性层 $y = Wx$（$W \in \mathbb{R}^{m \times n}, x \in \mathbb{R}^n$）：FLOP $= 2mn$（一次矩阵乘向量）。

### 3.2 前向传播的 FLOP

> **关键近似：前向传播，每个 token、每个参数约 2 FLOP。**

$$
\text{Forward FLOP} \approx 2 \times N \times (\text{tokens})
$$

直觉：线性层 $y = Wx$，$W$ 有 $N_W$ 个参数，对单个 token（$x$ 是向量）做 $2N_W$ FLOP。整个模型 $N$ 个参数，单 token 前向 $\approx 2N$ FLOP。$D$ 个 token 则 $2ND$。

- 这忽略了 attention 的 $T^2$ 项（见 3.4）和归一化/激活等小项，对短序列是好的近似。
- 等价表述：前向 FLOP $\approx 2 \times$ "参数×token"。

### 3.3 训练总 FLOP：$C = 6ND$（贯穿全课）

```
前向:  2 FLOP / 参数 / token
反向:  4 FLOP / 参数 / token  (约前向的 2 倍: 算输入梯度 + 算权重梯度, 各 ~2)
合计:  6 FLOP / 参数 / token

C = 6 × N × D   (N=参数量, D=训练 token 数)
```

- **前向 2**：每个参数对每个 token 贡献 1 乘 1 加。
- **反向 4**：反向要算两个梯度——对输入的梯度 $\partial L / \partial x$（传给上层）和对权重的梯度 $\partial L / \partial W$（用于更新），各约 2 FLOP/参数/token，合计 4。
- 故训练（前向+反向）= $2 + 4 = 6$ FLOP/参数/token。

> [!tip] 这就是 [[缩放定律]] 的基石。$C=6ND$ 让"模型规模 $N$ + 数据 $D$"统一成一个算力预算 $C$，是 Lecture 9/11 缩放曲线的横轴。

### 3.4 注意力的 FLOP（容易忽略）

标准 self-attention 的 $QK^T$ 和 $\text{softmax}(QK^T) \cdot V$ 与**序列长度平方**相关：

$$
\text{Attention FLOP} \approx 2 \times L \times T^2 \times d \quad (\text{每序列，含 } QK^T \text{ 和 } PV)
$$

（$L$ 层、序列长度 $T$、隐藏 $d$；更精确是 $2 L n_h T^2 d_k = 2 L T^2 d$，因 $n_h d_k = d$。）

- **短序列**：$T^2$ 项 $\ll 6ND$ 里的 $ND$ 项（$N \sim 12 L d^2$，$T \ll d$ 时 $T^2 d \ll 12 d^2 \cdot T$），可忽略，$C=6ND$ 够用。
- **长序列**（$T$ 大，如 32k/128k）：$T^2$ 项主导，$C=6ND$ 严重低估。这也是 [[FlashAttention]] 的动机——长序列 attention 是算力+显存的双重瓶颈。

### 3.5 推理 FLOP

推理只有前向（无反向），故：

$$
\text{Inference FLOP} \approx 2 N \times (\text{生成 token 数})
$$

- prefill 阶段（处理 prompt）：约 $2 N \times T_{\text{prompt}}$ + attention 的 $T^2$ 项。
- decode 阶段（逐 token 生成）：每生成 1 token 约 $2N$ FLOP（attention 的 $T^2$ 项每步小，因 KV cache 只算新 token 对历史）。
- 推理 FLOP $\approx$ 训练 FLOP / 3（无反向，且 token 数通常远少于训练）。

### 3.6 矩阵乘 FLOP 的硬件对照

| 操作 | FLOP | 说明 |
|---|---|---|
| 矩阵乘 $A_{m\times n} B_{n\times p}$ | $2mnp$ | 基础规则 |
| 线性层 $Wx$ | $2 \cdot \text{params}$ | 单 token 前向 |
| 前向整模型 | $2ND$ | $D$ token |
| 训练整模型 | $6ND$ | 前向+反向 |
| 每层 attention（$T^2$ 项） | $2T^2 d$ | 长序列主导 |


## 4. 数学原理 / 公式

### 4.1 矩阵乘 FLOP 推导

$C = AB$，$A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}$，$C \in \mathbb{R}^{m \times p}$。

每个 $C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$：$n$ 次乘法 + $(n-1)$ 次加法 $\approx 2n$ FLOP（$n$ 大时 $-1$ 可忽略）。

共 $m \cdot p$ 个元素：

$$
\text{FLOP} = m \cdot p \cdot 2n = 2mnp
$$

### 4.2 前向 2 FLOP/参数/token 推导

线性层 $y = Wx + b$，$W \in \mathbb{R}^{m \times n}$ 有 $N_W = mn$ 个参数。对单个输入 $x \in \mathbb{R}^n$：

$$
\text{FLOP} = 2mn = 2 N_W
$$

（bias 的 $m$ 次加法 $\ll 2mn$，忽略。）

整个模型 $N$ 个参数，单 token 前向 $\approx 2N$ FLOP（所有线性层求和）。$D$ 个 token：

$$
\text{Forward FLOP} = 2 N D
$$

### 4.3 反向 4 FLOP/参数/token 推导

反向传播对每个线性层 $y = Wx$ 算两个梯度：

1. **对输入的梯度** $\frac{\partial L}{\partial x} = W^\top \frac{\partial L}{\partial y}$：矩阵乘向量，FLOP $\approx 2 N_W$。
2. **对权重的梯度** $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} x^\top$：外积，FLOP $\approx 2 N_W$。

合计 $4 N_W$ FLOP/参数/token。整个模型 $D$ token：

$$
\text{Backward FLOP} = 4 N D
$$

### 4.4 训练总 FLOP

$$
\boxed{C = \text{Forward} + \text{Backward} = 2ND + 4ND = 6ND}
$$

### 4.5 注意力 $T^2$ 项

每层 self-attention：
- $S = QK^\top$：$Q, K \in \mathbb{R}^{T \times d_k}$（单头），$S \in \mathbb{R}^{T \times T}$。FLOP $= 2 T^2 d_k$。
- $O = \text{softmax}(S) \cdot V$：$V \in \mathbb{R}^{T \times d_k}$，FLOP $= 2 T^2 d_k$。
- 单头合计 $4 T^2 d_k$，$n_h$ 头：$4 n_h T^2 d_k = 4 T^2 d$（因 $n_h d_k = d$）。

$L$ 层：

$$
\text{Attention FLOP} \approx 4 L T^2 d \quad (\text{每序列})
$$

（CS336 课件记 $2LT^2d$，是只算 $QK^T$ 或 PV 之一，看口径；量级一致。）

### 4.6 Llama-2 7B 训练 FLOP 算例

$N \approx 6.7$B，训 $D = 2 \times 10^{12}$（2T tokens）：

$$
C = 6 \times 6.7 \times 10^9 \times 2 \times 10^{12} = 8.04 \times 10^{22} \text{ FLOP}
$$

约 $8 \times 10^{22}$ FLOP。配合 2048 张 A100（312 TFLOPs bf16 峰值）和 40% MFU：

$$
\text{时间} \approx \frac{8 \times 10^{22}}{2048 \times 312 \times 10^{12} \times 0.4} \approx 3.13 \times 10^5 \text{ s} \approx 87 \text{ 小时}
$$

（见 [[MFU与算术强度]] 的训练时间公式。）


## 5. 代码示例

```python
def training_flops(N, D):
    """训练总 FLOP = 6 N D. N=参数量, D=token 数."""
    return 6 * N * D

def forward_flops(N, D):
    """前向 FLOP = 2 N D."""
    return 2 * N * D

def attention_flops(L, T, d):
    """每序列 attention 的 T^2 项 ≈ 4 L T^2 d."""
    return 4 * L * T * T * d

# Llama-2 7B, 2T tokens
N = 6.7e9
D = 2e12
print(f'7B 训练 2T tokens: {training_flops(N, D):.2e} FLOP')
print(f'  前向: {forward_flops(N, D):.2e}')
print(f'  反向: {3*forward_flops(N, D):.2e} (前向的2倍)')

# 注意力 T^2 项占比（短 vs 长序列）
L, d = 32, 4096
for T in [2048, 8192, 32768]:
    attn = attention_flops(L, T, d)
    linear = 2 * N * T   # 单序列前向线性部分
    print(f'T={T}: attn={attn:.2e}, linear={linear:.2e}, attn/linear={attn/linear:.2f}')
# 长序列时 attn 的 T^2 项反超线性项，6ND 失准

# 矩阵乘 FLOP 基础
def matmul_flops(m, n, p):
    return 2 * m * n * p
print(f'4096x4096 @ 4096x4096: {matmul_flops(4096,4096,4096):.2e} FLOP')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[参数量计算]]（$N$ 的来源）、矩阵乘规则（$2mnp$）、前向/反向传播（2 vs 4 的推导）、[[自注意力]]（$T^2$ 项的来源）。
- **下游（应用）**: [[缩放定律]]（$C$ 是横轴，$C=6ND$ 把 $N,D$ 统一）、[[MFU与算术强度]]（训练时间 $= C / (\text{GPU数}\times\text{峰值}\times\text{MFU})$）、训练成本估算、[[Roofline模型]]（算术强度 $= \text{FLOP} / \text{bytes}$）。
- **对比 / 易混**:
  - **FLOP（次数）vs FLOPs（速率）**：FLOP 是总运算次数（量，如 $8 \times 10^{22}$），FLOPs 是每秒运算次数（硬件峰值，如 A100 312 TFLOPs）。训练时间 = FLOP / (GPU数 × FLOPs × MFU)。
  - **训练 vs 推理 FLOP**：训练 $6ND$（含反向），推理 $2N \times \text{tokens}$（仅前向）。同模型同 token 数，训练 FLOP 是推理的 ~3 倍。
  - **$C=6ND$ vs attention $T^2$**：$6ND$ 是线性部分（短序列主导），$T^2$ 是 attention 平方部分（长序列主导）。长序列时 $6ND$ 严重低估，需加 $4LT^2d$。


## 7. 常见误区与易错点

> [!warning] 误区 1：混淆 FLOP 与 FLOPs
> **FLOP** = 浮点运算次数（量，如训练共 $8 \times 10^{22}$ FLOP）。**FLOPs** = 每秒浮点运算次数（硬件峰值速率，如 A100 = 312 TFLOPs = $3.12 \times 10^{14}$/s）。前者是"要做多少"，后者是"每秒能做多少"。训练时间 = FLOP / FLOPs。文献常滥用大小写，需看上下文。

> [!warning] 误区 2：长序列还用 $C=6ND$
> $6ND$ 只含线性层（attention 投影 + FFN）的 FLOP，**不含 attention 的 $QK^T/PV$ 的 $T^2$ 项**。短序列（$T \ll d$）时可忽略；长序列（$T=32k, 128k$）时 $T^2$ 项主导，$6ND$ 严重低估。报长上下文模型算力时必须加 $4LT^2d$。

> [!warning] 误区 3：把矩阵乘算成 $mnp$ 而非 $2mnp$
> 矩阵乘 $A_{m\times n} B_{n\times p}$ 是 $2mnp$ FLOP（因子 2 = 乘+加），不是 $mnp$。少算因子 2 会让所有后续 FLOP 估算偏差 2 倍。深度学习惯例按 FMA=2 算（与 GPU 峰值标注一致）。

> [!warning] 误区 4：推理用 $6ND$
> $6ND$ 是**训练**（前向+反向）。推理只有前向，是 $2N \times \text{tokens}$（约训练的 1/3）。把训练公式套推理会高估 3 倍。

> [!warning] 误区 5：MoE 用总参数算 FLOP
> [[MoE]] 模型每 token 只过 top-k 专家，FLOP 用**激活参数**算（不是总参数）。例：DeepSeek-V3 总 671B 但每 token 激活 37B，FLOP 按 37B 算。显存才按总参数 671B 算。

> [!warning] 误区 6：忽略 embedding/softmax 的 FLOP
> $6ND$ 主要是 transformer 层（attention 投影 + FFN）。embedding 查表（近乎 0 FLOP）和 lm_head 的 softmax（$O(V)$ 每 token）通常忽略，但大词表（$V=128$k）时 softmax 计算非平凡，可能占可观察比例。粗算忽略，精算需加。


## 8. 延伸细节

### 8.1 为什么是 6 而不是 3（FMA）

硬件有 FMA（fused multiply-add）指令，把 $a \times b + c$ 合成 1 个操作。若按 FMA=1 FLOP 算，矩阵乘是 $mnp$ 而非 $2mnp$，训练就是 $3ND$。但**业界惯例按 FMA=2 FLOP** 算（与 NVIDIA 标注 GPU 峰值一致：A100 bf16 312 TFLOPs 是按 FMA=2 算的）。故用 $6ND$ 和厂商峰值算时间才自洽。混用 FMA=1 的 $3ND$ 和 FMA=2 的峰值会偏差 2 倍。

### 8.2 attention $T^2$ 项与 [[FlashAttention]]

标准 attention 把 $QK^T$、softmax、PV 写到 HBM，既费 FLOP（$T^2$）又费显存（$T^2$，见 [[activation memory]]）。[[FlashAttention]] 不改 FLOP（该算的还算），但用 tiling 在 SRAM 内完成、不写回 HBM，减**显存读写**（memory-bound 时提速）+ 减**激活显存**。FLOP 不减，但墙钟时间减。

### 8.3 FLOP 与成本的换算

给定 $C$ 和硬件，成本估算：

$$
\text{GPU 时} = \frac{C}{\text{FLOPs} \times \text{MFU}} / 3600, \quad \text{成本} = \text{GPU 时} \times \text{每小时租金}
$$

例：$C = 8 \times 10^{22}$，2048 张 A100（312 TFLOPs），MFU 40%：

$$
\text{时间} = \frac{8 \times 10^{22}}{2048 \times 3.12 \times 10^{14} \times 0.4} \approx 87 \text{ 小时}
$$

A100 租金 ~$2/h → 约 $357k。这是 Llama-2 7B 训练成本的粗算（实际含实验/失败的总量更大）。

### 8.4 推理 prefill vs decode 的 FLOP

- **prefill**（处理 prompt，$T_p$ token）：并行算，FLOP $\approx 2N T_p + 4L T_p^2 d$。算力受用（大 batch prefill 接近 compute-bound）。
- **decode**（逐 token 生成）：每 token 读全部权重做一次前向，FLOP $\approx 2N$ per token，但每次只 1 token（batch=1 时算力极低，memory-bound）。这是推理 throughput 低的根因（见 [[batching tradeoff]]）。

### 8.5 内容来源

本笔记核心公式（矩阵乘 $2mnp$、前向 $2N$、训练 $C=6ND$、attention $T^2$、FMA=2 惯例）整理自 CS336 Lecture 2（PyTorch 与资源核算）。

---
相关: [[资源核算]] | [[参数量计算]] | [[MFU与算术强度]] | [[缩放定律]] | [[Roofline模型]] | [[FlashAttention]] | [[activation memory]] | [[自注意力]] | [[MoE]]
