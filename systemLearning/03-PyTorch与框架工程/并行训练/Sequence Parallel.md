# Sequence Parallel

> **所属章节**: [[并行训练]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **别名**: Sequence Parallel / SP / 序列并行 / 激活分片 / AllGather-ReduceScatter 协同 TP
> **难度**: 中高（需懂 [[Tensor Parallel]] + [[ColumnParallelLinear]]/[[RowParallelLinear]] + [[AllGather]]/[[reduce-scatter]]）


## 1. 一句话定义

**Sequence Parallel（SP）** 是 [[Megatron-LM]] 对 [[Tensor Parallel]] 的**激活显存优化扩展**——把 TP 块**非切分区**（norm、dropout、softmax 等"本应 replicate 的激活"）沿 **sequence 维**也切分成 $T$ 片，每 rank 只持 $1/T$ 的 sequence 激活，把 [[ColumnParallelLinear]] 前向的输入由"replicate 激活"改为"先 `all-gather` 合 sequence 再算"、把 [[RowParallelLinear]] 前向末尾的 `all-reduce`（合完整输出）换成 **`reduce-scatter`**（合部分和后沿 sequence 切回分片，输出仍 sharded）。这样整个 transformer 层的激活（norm 区 + attention/MLP 区）几乎全部分片，activation memory 降 $\approx T$ 倍（大序列长模型的显存瓶颈），通信量从一次 `all-reduce` 变 `all-gather`+`reduce-scatter`（总量相近，但 all-gather 可与上游 overlap）。SP 只切"norm 区"激活（attention/MLP 的 matmul 区仍 TP 切 weight，激活在那是 replicated 输入），是 TP 的显存补充，不独立使用（与 TP 强耦合）。Ulysses/DeepSpeed-Ulysses 是 SP 的一种变体（沿 head 维切 + all-to-all），思路类似。

> [!note] 论文出处（联网核实）
> SP 出自 NVIDIA 论文 **"Reducing Activation Recomputation in Large Transformer Models"**（arXiv:2205.05198，Korthikanti, Casper, Lym, McAfee, Andersch, Shoeybi, Catanzaro，MLSys 2023）。该论文**同时提出两项技术**：① **Sequence Parallelism**（本笔记）；② **Selective Activation Recomputation**（重算 attention scores，见 §8.5）。两者叠加使激活显存降 **5×**、recompute 计算开销从 30-40% 降到 ~3%，530B 模型 2240 张 A100 上 MFU 从 42.1% → 54.2%（+29%）。实现已并入 Megatron-LM 与 NeMo-Megatron。

> [!note] 三句话定位
> - **是什么**：把 TP 块的 norm/dropout/softmax 区激活沿 sequence 维分片，配 all-gather（前向前）/reduce-scatter（前向后）协同 TP，activation 降 $T$ 倍。
> - **为什么**：TP 只切 weight，norm 区激活仍 replicate（大序列长时显存瓶颈）。SP 把这些激活也切，省显存。
> - **与 [[Tensor Parallel]] 关系**：SP 是 TP 的延伸（不独立），把 TP 块的 all-reduce 换 reduce-scatter + 加 all-gather 协同，activation 显存从"部分 replicate"变"全分片"。


## 2. 为什么需要它（动机与背景）

### 2.1 TP 的激活显存盲区

[[Tensor Parallel]] 切 weight（[[ColumnParallelLinear]]/[[RowParallelLinear]]），但 norm/dropout/softmax 区的激活**仍 replicate**（每 rank 持完整 $[B, S, d]$）。大序列长（$S$ 大，如 32k/128k context）时，这些激活是显存瓶颈（$\propto B \cdot S \cdot d$）。TP 只省 weight 区，norm 区不减。

### 2.2 SP 切 sequence 维

SP 把 norm 区激活沿 sequence 维切 $T$ 片（每 rank $[B, S/T, d]$）。norm/dropout 是 element-wise（可逐元素切，每 rank 算自己 sequence 片）。activation memory 降 $\approx T$ 倍。

### 2.3 all-gather/reduce-scatter 协同 TP

norm 区激活切了，但 attention/MLP 的 matmul 区需 TP 切 weight（input 需完整 sequence 还是分片？）。SP 的设计：matmul 区的 input（从 norm 出来）是 sequence 分片，进 matmul 前先 `all-gather` 合 sequence（成完整 $[B, S, d]$），再 TP matmul。matmul 后 row 的 all-reduce 换 `reduce-scatter`（合部分和 + 沿 sequence 切回）。整层激活在 norm 区分片、matmul 区临时合（用完散）。

### 2.4 通信量相近，显存大省

TP 块原 all-reduce（合完整 $[B,S,d]$）→ SP 的 all-gather（合 sequence）+ reduce-scatter（合 + 切）。通信量：all-reduce $\approx 2 \cdot B \cdot S \cdot d$（ring all-reduce），all-gather + reduce-scatter $\approx 2 \cdot B \cdot S \cdot d$（合+散各 $\approx B \cdot S \cdot d$）。总量相近。但 activation memory 省 $T$ 倍（norm 区切）。长序列显存瓶颈时 SP 大赚。

### 2.5 all-gather 可 overlap

SP 的 all-gather（前向前合 sequence）可与上游计算 overlap（[[overlap strategy]]）。reduce-scatter（前向后切）也可 overlap。vs all-reduce 难 overlap（同层依赖）。通信可藏。

### 2.6 长序列/大 batch 的刚需

长 context（128k）或大 batch 训练，activation memory 极大（$\propto B \cdot S \cdot d$）。SP 减 $T$ 倍是刚需。是 LLM 长序列训练的标准（Megatron + SP）。


## 3. 核心概念详解

### 3.1 TP 块的激活分区

```
标准 TP 块 (无 SP):
  norm (replicate [B,S,d]) -> QKV col (replicate input) -> attn -> O row (all-reduce -> replicate) -> norm (replicate)
  
  激活: norm 区 replicate (B*S*d), matmul 区 input replicate, 全层激活 ~ 几个 B*S*d

SP 块:
  norm (shard sequence [B,S/T,d]) -> all-gather (合 [B,S,d]) -> QKV col -> attn -> O row (reduce-scatter -> shard sequence [B,S/T,d]) -> norm (shard)
  
  激活: norm 区 shard (B*S/T*d), matmul 区临时合 (用完散), 全层激活 ~ 几个 B*S/T*d
```

### 3.2 all-gather（前向前）

norm 区激活是 sequence 分片 $[B, S/T, d]$。进 matmul 前 `all-gather` 沿 sequence 合成完整 $[B, S, d]$（replicate）。matmul 用完整 input 算。合的临时 $[B,S,d]$ 用完释放。

### 3.3 reduce-scatter（前向后）

[[RowParallelLinear]] 前向原 `all-reduce`（合完整 $y$ replicate）→ SP 换 `reduce-scatter`（合部分和 + 沿 sequence 切回 $[B, S/T, d]$ 分片）。输出仍是 sequence 分片，给下一个 norm 区用。省一次完整 $[B,S,d]$ 的存。

### 3.4 整层激活全分片

```
norm1 (SP shard) -> all-gather -> QKV col -> attn -> O row reduce-scatter (SP shard)
-> norm2 (SP shard) -> all-gather -> gate_up col -> act -> down row reduce-scatter (SP shard)
-> ...
```

norm 区激活全 sequence 分片（$B \cdot S/T \cdot d$），matmul 区临时合（用完散）。整层激活 $\propto B \cdot S/T \cdot d$（vs TP 的 $\propto B \cdot S \cdot d$）。省 $T$ 倍。

### 3.5 反向的 all-gather/reduce-scatter

反向对称：row 反向原无通信（grad_x 分片）→ SP 加 `all-gather`（合 grad sequence）给上游。col 反向原 all-reduce（grad_input 合）→ SP 换 `reduce-scatter`（合 + 切回 sequence 分片）。反向通信也 all-gather/reduce-scatter 协同。

### 3.6 与 TP 的耦合

SP 不独立用（norm 区切了但 matmul 区仍需 TP 切 weight）。SP 是 TP 的扩展（TP 块加 SP 协同）。开 SP 需开 TP。无 TP 的 SP 无意义（matmul 区不切 weight，全 replicate，SP 只切 norm 区省得少）。

### 3.7 通信量的细算

TP 块（无 SP）每层：row 前向 all-reduce $\approx 2 B S d$（ring），col 反向 all-reduce $\approx 2 B S d$。总 $\approx 4 B S d$。

SP 块每层：前向 all-gather（norm 前，$\approx B S d$）+ reduce-scatter（row 后，$\approx B S d$）。反向 all-gather + reduce-scatter 各 $\approx B S d$。总 $\approx 4 B S d$。相近（all-gather+reduce-scatter 各 $\approx$ all-reduce 的一半，合起来等）。

### 3.8 DeepSpeed-Ulysses 变体

DeepSpeed-Ulysses 是 SP 变体：沿 **head 维**切（非 sequence 维），用 `all-to-all`（非 all-gather/reduce-scatter）在 head 与 sequence 维间转置。思路类似（激活分片 + 通信协同），但切法与通信原语不同。Ulysses 与 FlashAttention 配合更紧（head 切与 FA 的 head 循环契合）。

### 3.9 SP 的实现

```python
# megatron/core/tensor_parallel/mappings.py
def all_gather_last_dim(input_):  # SP -> TP (shard seq -> replicate)
    return all_gather(input_, dim=-2)  # gather sequence dim
def reduce_scatter_last_dim(input_):  # TP -> SP (replicate -> shard seq)
    return reduce_scatter(input_, dim=-2)  # scatter sequence dim
```

ColumnParallelLinear/RowParallelLinear 有 `sequence_parallel` 选项，开启后前向/反向插这些通信。

### 3.10 与 attention 的 SP 兼容

attention 内部（QK^T、softmax、PV）需完整 sequence（每 head 算所有 token）。SP 的 all-gather 在 attention 前合 sequence，attention 用完整 sequence 算。SP 不切 attention 内部（attention 需完整 sequence）。SP 只切 norm 区（element-wise 可切）。

### 3.11 论文的 g / ḡ 算子命名（联网核实）

论文给 SP 的两个"序列↔张量并行区转换器"取了名字，是理解 SP 数据流的关键：

- **$g$（进 TP 区）**：前向是 **all-gather**（SP shard $[B,S/T,d]$ → TP replicate $[B,S,d]$），反向是 **reduce-scatter**（TP grad → SP shard）。位于 [[ColumnParallelLinear]] 前。
- **$\bar{g}$（出 TP 区）**：前向是 **reduce-scatter**（TP partial sum → SP shard $[B,S/T,d]$），反向是 **all-gather**。位于 [[RowParallelLinear]] 后。
- **$g$ 与 $\bar{g}$ 共轭**（conjugate）：一个前向 AG/反向 RS，另一个前向 RS/反向 AG，对称。

> [!tip] 为什么不引入额外通信
> 朴素想法是 SP 在 TP 的 $f$/$\bar{f}$ 算子（all-reduce）前后各塞一个 all-gather/reduce-scatter，通信翻倍。论文的 $g$/$\bar{g}$ 把"额外 all-gather"**合并进 $f$/$\bar{f}$**：因为 **ring all-reduce = reduce-scatter + all-gather**（两步），SP 只是把这个两步拆开、中间插 matmul。所以**总通信字节不变**（4 次 all-reduce ↔ 4 次 all-gather + 4 次 reduce-scatter，每对 ≡ 一次 all-reduce），SP 是"等通信换显存"而非"多通信换显存"。

### 3.12 Y 张量的反向特例

论文指出：SP + TP 后，反向所需激活**几乎全部分片**，唯独第一个 linear 的输入 $Y$（norm 输出）需在反向时 all-gather 回完整。论文做法：**前向只存 $Y_i^s$（本 rank 片）**，反向时做一次额外 all-gather 合回 $Y$，并**把这次 all-gather 与 grad-Y 的计算 overlap** 掉延迟。这是 SP 唯一的"额外"通信，且被藏进计算。


## 4. 数学原理 / 公式

### 4.1 激活显存

设层激活约 $k \cdot B \cdot S \cdot d$（$k$ = 中间激活数，如 attention/MLP 的几份）。TP（无 SP）：norm 区 replicate，$\approx k \cdot B \cdot S \cdot d$。SP：norm 区切，$\approx k \cdot B \cdot S/T \cdot d$。省 $T$ 倍。

> [!note] 论文的精确公式（联网核实，Eq.2 → Eq.4）
> 论文用 $s$=序列长、$b$=batch、$h$=hidden、$a$=attention head 数、$t$=TP size，推出**单层激活显存**：
>
> **仅 TP（无 SP）**：$\text{mem} = sbh\!\left(10 + \frac{24}{t} + 5\frac{as}{h}\right)$（Eq.2）
>
> —— 注意 **$10sbh$ 项没除 $t$**：这 10 份是 LayerNorm/Dropout/residual 等 **replicated 区**（TP 不切它们），是 SP 要解决的"显存地板"。
>
> **TP + SP**：$\text{mem} = \frac{sbh}{t}\!\left(34 + 5\frac{as}{h}\right)$（Eq.4）
>
> —— 所有项都除 $t$ 了（10 那项也被 SP 切了）。34 = 10(norm/dropout) + 24(matmul 中间)，$5as/h$ 是 attention 的 $S^2$ 项（softmax/PV 的 score 张量，这部分 SP 仍无法切，要靠 selective recompute，见 §8.5）。
>
> 对 GPT-3（$a=96, s=2048, h=12288$）：$5as/h = 80$，vs 34 → attention 的 $S^2$ 项占大头。SP 把 34 那部分全切了（省 ~30%），但 $5as/h=80$ 那部分还在（占 70%）。所以**单 SP 不够，需配 selective recompute 才能到 5×**。

### 4.2 通信量

TP（无 SP）每层：$\approx 4 B S d$（前向+反向各 all-reduce $\approx 2BSd$）。SP 每层：$\approx 4 B S d$（前向+反向各 all-gather+reduce-scatter $\approx 2BSd$）。相近。SP 用"相近通信"换"显存 $T$ 倍省"。

### 4.3 all-gather/reduce-scatter 的关系

`all-reduce` = `all-gather`（合） + `reduce-scatter`（散）。即 all-reduce 可分解为 all-gather + reduce-scatter。SP 把 all-reduce 拆开，中间插入 matmul（合后算，散后存），省激活显存。

### 4.4 算力

SP 不切算力（matmul 区仍 TP 切 weight，算力 $1/T$）。norm 区 element-wise 各 rank 算自己 sequence 片（也算力 $1/T$）。算力与 TP 同。

### 4.5 overlap 收益

SP 的 all-gather（合 sequence）可与上游 norm/dropout overlap。reduce-scatter（散）可与下游 overlap。vs all-reduce 难 overlap（同层依赖）。SP 通信可藏，wall-clock 可能更优。


## 5. 代码示例（可选）

### 5.1 Megatron 开启 SP

```python
from megatron.core import parallel_state as ps
ps.initialize_model_parallel(tensor_model_parallel_size=8, sequence_parallel=True)
# ColumnParallelLinear/RowParallelLinear 自动开 SP 协同
# QKV col: 前向先 all-gather (norm 区 shard seq -> 合) 再算
# O row: 前向末尾 reduce-scatter (合 + 切回 shard seq)
```

### 5.2 SP 的 all-gather/reduce-scatter

```python
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,  # SP -> TP (all-gather seq)
    scatter_to_sequence_parallel_region,  # TP -> SP (reduce-scatter seq)
)
# norm 区激活 [B, S/T, d] (SP shard)
x_full = gather_from_sequence_parallel_region(x_shard)  # all-gather -> [B, S, d] (TP replicate)
# ... matmul (TP) ...
y_shard = scatter_to_sequence_parallel_region(y_partial)  # reduce-scatter -> [B, S/T, d] (SP shard)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Tensor Parallel]]、[[ColumnParallelLinear]]/[[RowParallelLinear]]、[[AllGather]]/[[reduce-scatter]]、[[all-reduce]]、[[Megatron Core目录与执行链路]]、[[activation memory]]、[[overlap strategy]]。
- **下游（应用）**: 长序列 LLM 训练（128k context）、Megatron Core 的 SP 实现、DeepSpeed-Ulysses（变体）、[[3D parallelism]]（与 TP/PP/DP 组合）、[[gradient checkpointing]]（互补省显存）。
- **对比 / 易混**:
  - **SP vs [[Context Parallel]]**：见 [[Context Parallel]] §1 易混表。SP 只在 norm/dropout 区切序列省激活显存（注意力前 all-gather 回完整），是 TP 的显存优化伴生；CP 连注意力也切（跨块用 ring/all-to-all 算），降算力+显存。两者正交可叠加（Megatron `--sequence-parallel` + `--context-parallel-size`）。
  - **SP vs [[Tensor Parallel]]**：TP 切 weight（算力切分），norm 区激活仍 replicate。SP 切 norm 区激活（显存切分），matmul 区仍 TP。SP 是 TP 的显存补充，不独立。
  - **SP vs [[gradient checkpointing]]**：SP 切激活分片（省 $T$ 倍），gradient checkpointing 重算激活（省 $\sqrt{L}$ 倍但费 FLOP）。互补（SP 省 sequence 维，recompute 省层维）。两者可叠加。
  - **SP 的 reduce-scatter vs [[RowParallelLinear]] all-reduce**：SP 把 row 的 all-reduce 换 reduce-scatter（合+切回 sequence）。通信量相近（all-reduce = all-gather + reduce-scatter），但输出从 replicate 变 shard sequence（省显存）。
  - **SP vs DeepSpeed-Ulysses**：SP 切 sequence 维用 all-gather/reduce-scatter；Ulysses 切 head 维用 all-to-all。思路类似（激活分片+协同），切法与原语不同。Ulysses 与 FlashAttention 配合更紧。


## 7. 常见误区与易错点

> [!warning] 误区 1：SP 是独立并行
> SP 不独立。是 TP 的扩展（norm 区激活切），matmul 区仍 TP。无 TP 的 SP 无意义（不切 weight，全 replicate，省得少）。SP 必须配 TP。

> [!warning] 误区 2：SP 切 attention 内部
> SP 不切 attention 内部（QK^T/softmax/PV 需完整 sequence，每 head 算所有 token）。SP 只切 norm 区（element-wise 可切）。attention 前 all-gather 合 sequence 用，attention 用完整 sequence 算。

> [!warning] 误区 3：SP 通信比 TP 多
> 通信量相近（all-gather+reduce-scatter $\approx$ all-reduce）。SP 用相近通信换显存 $T$ 倍省。不是"多通信换显存"，是"等通信换显存"。

> [!warning] 误区 4：SP 的 all-gather 不能 overlap
> SP 的 all-gather（norm 前）可与上游 overlap（独立层）。reduce-scatter（row 后）可与下游 overlap。vs all-reduce 难 overlap（同层依赖）。SP 通信可藏，wall-clock 可能更优。

> [!warning] 误区 5：SP 省 weight 显存
> SP 省**激活**显存（norm 区），不省 weight（weight 仍 TP 切，省 $T$ 倍是 TP 的功劳）。SP 的省是激活维度。

> [!warning] 误区 6：SP 与 PP 冲突
> SP 可与 PP 共存（PP 切层维，SP 切 sequence 维，正交）。3D 并行（TP+PP+DP）可加 SP。是 [[3D parallelism]] 的扩展。

> [!warning] 误区 7：SP 的 reduce-scatter 输出是 weight 分片
> SP 的 reduce-scatter 输出是 **sequence 维分片**（$[B, S/T, d]$），不是 weight 分片。weight 分片是 TP 的事。SP 切的是激活 sequence 维，勿混。


## 8. 延伸细节

### 8.1 SP 与 [[gradient checkpointing]] 叠加

SP 省 sequence 维激活（$T$ 倍），recompute 省层维激活（$\sqrt{L}$ 倍）。两者正交可叠加。长序列+深模型全开：SP + recompute + TP + PP + DP。是 LLM 长序列训练的显存栈。

### 8.2 Ulysses 的 all-to-all

DeepSpeed-Ulysses：沿 head 维切，用 all-to-all 在 head 与 sequence 维间转置。all-to-all 的通信与 all-gather/reduce-scatter 不同（all-to-all 是置换，需各 rank 互发）。Ulysses 与 FlashAttention 的 head 循环契合（每 rank 专责部分 head）。

### 8.3 SP 的 DropOut

SP 切 dropout 区（dropout 是 element-wise，可逐 sequence 片算）。但 dropout 的 mask 需各 rank 独立 RNG（[[Megatron-LM]] 的分布式 RNG，保证可复现）。SP 的 dropout 各 rank 用自己 sequence 片的 RNG。

### 8.4 内容来源

Sequence Parallel 出自 NVIDIA 论文 **"Reducing Activation Recomputation in Large Transformer Models"**（arXiv:2205.05198，Korthikanti et al., MLSys 2023），实现见 `megatron/core/tensor_parallel/mappings.py`（`gather_from_sequence_parallel_region` / `reduce_scatter_to_sequence_parallel_region`）与 `layers.py`（`ColumnParallelLinear`/`RowParallelLinear` 的 `sequence_parallel` 选项）。与 TP 的协同、g/ḡ 算子推导见论文 §4.2.2。DeepSpeed-Ulysses 见其论文。与 [[gradient checkpointing]] 的叠加见 [[gradient checkpointing]]。

### 8.5 Selective Activation Recomputation（论文的姊妹技术，联网核实）

SP 论文**同时提出** selective activation recompute——它是 SP 的必要补充，理解 SP 必须理解它：

- **问题**：SP 把 Eq.2 的 $10sbh$ 地板切掉了，但 $5as/h$ 项（attention 的 QK^T/softmax/score 张量，$\propto S^2$）**SP 切不掉**（attention 需完整 sequence）。长序列时这一项占大头（GPT-3 是 80/114 ≈ 70%）。
- **selective recompute 的做法**：前向**不存** attention 的中间 score 张量（QK^T/softmax/dropout output），只存 Q/K/V（属 34 那项，贵）；反向时**重算** score 部分。因为 score 张量**内存大但 FLOP 少**（几个 matmul），重算代价低。
- **收益**：对 GPT-3（$a=96,s=2048,h=12288$），selective recompute 省掉 $5as/h=80$ 那项 → 省内存 70%，额外 FLOP 仅 2.7%（$1+s/(6h)$）。对 MT-NLG 530B 省 65%、费 1.6%。
- **SP + selective 组合 = 5×**：SP 砍 34 项的 replicated 地板（~30%），selective 砍 $5as/h$ 项（~70%），合起来单层激活降到 TP-only 的 ~20%（5× 降幅），且几乎无需 full recompute 的 30-40% 算力税。530B/2240 A100 实测 MFU 42.1%→54.2%（+29%）。
- **何时开 selective**：$s/h$ 大（长序列、窄 hidden）时收益最大；$s$ 极长时重算成本 $s/(6h)$ 上升，需权衡。
- **与 [[gradient checkpointing]] 区别**：full recompute 省整层（$\sqrt{L}$ 倍但费 30-40% FLOP）；selective 只省 attention 的一部分（省 ~70% 但费 ~3%）；SP 省 norm 区（省 ~30% 费 0%）。三者正交可叠。

### 8.6 工程约束（联网核实 Megatron 源码）

从 `megatron/core/model_parallel_config.py` 与 `layers.py` 源码确认的硬约束：

1. **`CUDA_DEVICE_MAX_CONNECTIONS=1`**：SP 的 all-gather/reduce-scatter 要与 matmul overlap，必须强制 NCCL 流按调用顺序调度（否则调度器乱序，overlap 失效）。Megatron 在 `linear_with_grad_accumulation_and_async_allreduce` 里检测到 SP 且未设此环境变量会发 warning。这是 SP 性能调优的第一排查项。
2. **EP + TP 强制要求 SP**：源码 `if expert_model_parallel_size > 1 and tensor_model_parallel_size > 1: if sequence_parallel is False: warnings.warn("must be used")`。即 MoE 训练里只要同时开 EP 和 TP，SP 是**强制**的（否则 MoE token dispatcher 的激活分布会出问题）。
3. **TP=1 不能开 SP**：源码 `if sequence_parallel and tensor_model_parallel_size <= 1: raise ValueError`。SP 依赖 TP group 做通信，没 TP 就没 SP group。
4. **LayerNorm grad 的 all-reduce**：SP 把 LayerNorm 按 sequence 切了，但 LayerNorm 的 **weight/bias 是 replicate 的**（不切），所以反向时 LayerNorm 的 grad 需跨 TP group all-reduce（`allreduce_layernorm_grads` 在 optimizer finalize 阶段做）。这是 SP 引入的一个额外 grad 同步点，常被忽略。
5. **overlap 配套开关**：`tp_comm_overlap=True` + `tp_comm_overlap_ag`/`tp_comm_overlap_rs` 控制 AG/RS 与 GEMM 的 pipelining。需 TransformerEngine 支持。

---
相关: [[Sequence Parallel]] | [[Tensor Parallel]] | [[Context Parallel]] | [[ColumnParallelLinear]] | [[RowParallelLinear]] | [[AllGather]] | [[reduce-scatter]] | [[all-reduce]] | [[Megatron Core目录与执行链路]] | [[activation memory]] | [[overlap strategy]] | [[3D parallelism]] | [[gradient checkpointing]] | [[Megatron-LM]] | [[compute vs communication bottleneck]] | [[alpha-beta性能模型]]
