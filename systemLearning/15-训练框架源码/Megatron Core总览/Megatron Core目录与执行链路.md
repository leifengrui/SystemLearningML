# Megatron Core目录与执行链路

> **所属章节**: [[Megatron Core总览]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: Megatron Core / Megatron-LM 源码 / megatron.core 目录 / 执行链路 / 训练步通信时序 / forward→backward→all-reduce
> **难度**: 中高（需懂 [[Megatron-LM]] 框架概述 + [[Tensor Parallel]] + [[Pipeline Parallel]] + [[AllGather]]/[[all-reduce]]/[[reduce-scatter]]）


## 1. 一句话定义

**Megatron Core 目录与执行链路** 是 [[Megatron-LM]] 训练库的**源码模块拆分**（`megatron/core/` 下 `transformer/`（层数据流：embedding/attention/MLP/layernorm/rotary）、`tensor_parallel/`（[[ColumnParallelLinear]]/[[RowParallelLinear]]/[[Sequence Parallel]] 等切分算子）、`pipeline_parallel/`（[[1F1B调度]]/[[Interleaved schedule]] 的 stage manager + schedule）、`optimizer/`（[[Megatron distributed optimizer]] 的 D-optimier、grad bucket）、`moe/`（[[MoE token dispatcher]]/[[grouped GEMM]]）、`fusions/`（融合 kernel）、`distributed/`（[[DeviceMesh]]-like 的 `ParallelState`/rank group））与**一个训练步的通信时序**（forward：TP 行列并行 `all-gather`/`all-reduce`、PP `send/recv` micro-batch；backward：grad 反向 + DP `all-reduce` grad + TP `reduce-scatter`；optimizer step：D-optimier 分片更新）。它把 [[Megatron-LM]] 的"框架概述"落到**代码模块**与**时序**上——哪个目录管什么、一次 `forward` 内部按什么顺序触发哪些 NCCL 集合通信、通信与计算如何 overlap。是读懂 [[ColumnParallelLinear]]、[[1F1B调度]]、[[MoE token dispatcher]] 等具体源码的"地图"。

> [!note] 三句话定位
> - **是什么**：`megatron/core/` 的目录拆分（transformer/tensor_parallel/pipeline_parallel/optimizer/moe/...）+ 一次训练步的 forward→backward→all-reduce 通信时序。
> - **为什么**：[[Megatron-LM]] 概述只讲"有什么"，要看懂具体算子（[[ColumnParallelLinear]]）和调度（[[1F1B调度]]）得先有目录地图与时序骨架。
> - **与 [[Megatron-LM]] 关系**：[[Megatron-LM]] 是框架概述（TP/PP/DP 的"是什么/为什么"），本笔记是源码结构（目录+时序）。概述是入口，本笔记是导航到具体源码的地图。


## 2. 为什么需要它（动机与背景）

### 2.1 Megatron-LM 源码庞大难入门

[[Megatron-LM]] 的 `megatron/core/` 有数万行，涉及 transformer 层、并行算子、PP 调度、分布式优化器、MoE、融合 kernel 等多子系统。直接扎进具体文件（如 `tensor_parallel/mappings.py`）会迷失——不知它在全局扮演什么角色。需先有**目录地图**：哪个子目录管什么、彼此如何调用。

### 2.2 训练步的通信密集难理清

一次 LLM 训练步内部触发大量 NCCL 集合通信：TP 的 `all-gather`/`all-reduce`（行列并行算子）、PP 的点对点 `send/recv`（micro-batch 跨 stage）、DP 的 `all-reduce`（grad 同步）、SP 的 `reduce-scatter`/`all-gather`（激活分片）、MoE 的 `all-to-all`（token 派发）。这些通信何时触发、与计算如何交错（overlap）是性能关键。需**时序骨架**把它们排清。

### 2.3 通信与计算的 overlap 是性能核心

LLM 训练的 wall-clock 通信占比高（TP all-reduce、DP grad all-reduce）。Megatron 用 [[overlap strategy]] 把通信藏到计算后（如 DP grad all-reduce 与下一层 backward overlap）。读懂时序才知道哪里能 overlap、哪里是串行瓶颈。是 [[compute vs communication bottleneck]] 分析的基础。

### 2.4 源码演进的脉络

Megatron Core 持续演进：引入 [[Sequence Parallel]]（SP）、[[Megatron distributed optimizer]]（D-optimier，ZeRO-1/2）、[[Interleaved schedule]]（virtual pipeline）、[[MoE token dispatcher]]（aux-loss-free）。目录与时序反映这些演进。读懂地图才能跟最新版本。

### 2.5 与 PyTorch 原生（DTensor/FSDP2）的对照

Megatron 用手写 `ParallelState`（rank group）+ 手写算子，PyTorch 原生向 [[DeviceMesh]]/[[DTensor]]/[[FSDP2]] 演进。两者趋同（Megatron Core 也在用 DTensor/DeviceMesh 重构部分算子）。对照地图能理解两套实现的对应关系。


## 3. 核心概念详解

### 3.1 `megatron/core/` 目录拆分

```
megatron/core/
├── transformer/              # 层数据流（模型本体）
│   ├── transformer_layer.py       # TransformerBlock（多层堆叠）
│   ├── attention/                 # attention 子层
│   │   ├── self_attention.py      # SelfAttention（CoreAttention + linear）
│   │   ├── attention_layer.py    # 组合 QKV+o_proj
│   │   └── fused/                 # FlashAttention 融合
│   ├── mlp.py / mlp_layer.py      # MLP 子层（gate_up + down）
│   ├── embeddings.py              # word embedding
│   ├── layernorm.py / rms_norm.py # norm
│   └── positional_encoding/      # RoPE 等
├── tensor_parallel/          # TP 切分算子
│   ├── mappings.py                # all-gather/reduce-scatter/分片映射
│   ├── cross_entropy.py          # 并行交叉熵
│   └── layers.py                 # ColumnParallelLinear/RowParallelLinear
├── pipeline_parallel/        # PP 调度
│   ├── schedules.py             # 1F1B/interleaved schedule
│   ├── p2p_communication.py     # send/recv 通信
│   └── scheduler.py / stage_manager.py
├── optimizer/                # 分布式优化器
│   ├── optimizer.py              # 基础 optimizer
│   ├── distrib_optimizer.py     # D-optimier（分片 opt state）
│   └── grad_buffer.py           # grad bucket（overlap 通信）
├── moe/                      # MoE
│   ├── token_dispatcher.py      # aux-loss-free dispatcher
│   ├── grouped_gemm.py /专家 matmul
│   └── router.py
├── fusions/                  # 融合 kernel
│   ├── fused_softmax.py
│   ├── fused_layer_norm.py
│   └── fused_rope.py
├── distributed/             # 分布式 rank group
│   ├── parallel_state.py        # ParallelState（TP/PP/DP/EP group）
│   └── torch_batched_math.py
├── tensor_random/           # 分布式 RNG
├── models/                  # 顶层模型（GPTModel 等）
└── datasets/                # 数据
```

### 3.2 顶层 `models/`：GPTModel

```python
class GPTModel(nn.Module):
    def __init__(...):
        self.embedding = VocabParallelEmbedding(...)  # TP 切 vocab
        self.decoder = TransformerBlock(layers=[...])  # 多层堆叠
        self.output_layer = ColumnParallelLinear(...)  # lm_head (TP 切)
    def forward(self, tokens):
        x = self.embedding(tokens)
        x = self.decoder(x)  # 各层 forward
        return self.output_layer(x)  # logits (并行交叉熵)
```

模型本体在 `models/`，层与算子在 `transformer/`，并行算子在 `tensor_parallel/`。

### 3.3 `ParallelState`：分布式 rank group

```python
# megatron/core/distributed/parallel_state.py
class ParallelState:
    # 全局 rank -> 各并行维的 group
    tp_group, pp_group, dp_group, ep_group
    def initialize_model_parallel(...):
        # 建 tp/pp/dp/ep 的 process group
```

`ParallelState` 类似 [[DeviceMesh]]（PyTorch 原生）。各 rank 通过 group 通信。新版本 Megatron Core 在向 [[DeviceMesh]]/[[DTensor]] 靠拢（部分算子用 DTensor 重构）。

### 3.4 一个训练步的通信时序

```
[forward]
  for layer in transformer:
    # Sequence Parallel: all-gather 激活 (Shard -> Replicate)
    x = all_gather(x, sp_group)  # (若开 SP)
    # ColumnParallelLinear: input replicate, weight Shard(1), output Replicate (无通信)
    qkv = col_linear(x)  # 各 rank 算自己 col 片
    # attention (FlashAttention, 各 rank 独立 head)
    attn = flash_attn(qkv)
    # RowParallelLinear: input Shard(1), weight Shard(0), output Partial -> all-reduce
    out = row_linear(attn)  # 各 rank 算部分, all-reduce 合
    out = all_reduce(out, tp_group)  # TP all-reduce
    # MLP 同理 (gate_up col + down row + all-reduce)
  # PP: send/recv micro-batch 跨 stage
  if not last_stage: send_activation(next_stage)
  else: loss = parallel_cross_entropy(logits, labels)  # 不聚合 logits
[backward]
  for layer reversed:
    grad = backward(layer)  # 各算子反向
    # RowParallelLinear 反向: grad -> reduce-scatter (Shard)
    # ColumnParallelLinear 反向: 无通信 (col shard 各自算)
    # DP grad all-reduce (overlap 与下一层 backward)
    grad = all_reduce(grad, dp_group)  # DP grad 同步
[optimizer step]
  # D-optimier: shard 优化器状态
  optimizer.step()  # 各 rank 更新自己 shard
```

### 3.5 TP 通信时序

TP 的 `ColumnParallelLinear`（weight shard 列，input replicate，output 各 rank 持 col 片）→ `attention`（各 rank 独立 head）→ `RowParallelLinear`（weight shard 行，input 持片，output Partial）→ `all-reduce` 合。每层 attention/MLP 末尾一次 `all-reduce`（TP 通信）。$L$ 层则 $2L$ 次 all-reduce（attention+MLP）。是 TP 的通信开销。详见 [[ColumnParallelLinear]]/[[RowParallelLinear]]。

### 3.6 DP 通信时序

DP 的 `all-reduce` grad：各 rank 算不同 batch 的 grad，需 `all-reduce` 合。Megatron 用 [[gradient bucket与通信重叠]]：grad bucket 攒够触发 `all-reduce`（async），与下一层 backward overlap。$L$ 层 grad 全程 overlap，末尾等通信完成。详见 [[gradient bucket与通信重叠]]。

### 3.7 PP 通信时序

PP 的 `send/recv`：micro-batch 跨 stage 用点对点通信。[[1F1B调度]] 的 warmup/steady/cooldown 阶段控制 send/recv 时序。是 PP 的通信（非集合，是 P2P）。详见 [[1F1B调度]]。

### 3.8 SP 通信时序

[[Sequence Parallel]] 的 `all-gather`（前向，激活 Shard→Replicate）+ `reduce-scatter`（反向，Replicate→Shard）。与 TP 配合减 activation 显存。详见 [[Sequence Parallel]]。

### 3.9 MoE 通信时序

[[MoE token dispatcher]] 的 `all-to-all`：token 派发到目标 expert 所在 rank。每个 MoE 层一次 `all-to-all`（forward + backward 各一次）。是 MoE 的通信。详见 [[MoE token dispatcher]]。

### 3.10 通信与计算 overlap 的设计

Megatron 的时序设计核心是**通信藏到计算后**：
- TP `all-reduce`（row parallel 后）→ 下一层 `col linear` 前需等（依赖）。难 overlap（同层串行）。
- DP `all-reduce`（grad bucket）→ async，与下一层 backward overlap（独立）。能 overlap。
- PP `send/recv` → async，与计算 overlap。
- SP `all-gather`/`reduce-scatter` → 与 TP 配合 overlap。

[[overlap strategy]] 的具体实现。是 [[compute vs communication bottleneck]] 优化的基础。


## 4. 数学原理 / 公式

### 4.1 TP all-reduce 通信量

每层 row parallel 后一次 `all-reduce`，通信 $\propto B \cdot d_{\text{model}}$（output size）。$L$ 层 $2L$ 次（attn+MLP）。总 TP 通信 $\propto 2L \cdot B \cdot d$。详见 [[Tensor Parallel]] 与 [[alpha-beta性能模型]]。

### 4.2 DP all-reduce 通信量

每 rank 算不同 batch grad，`all-reduce` 合。通信 $\propto N$（model size，每 param 一个 grad）。bucket overlap 减延迟。详见 [[Megatron distributed optimizer]] 与 [[gradient bucket与通信重叠]]。

### 4.3 PP P2P 通信量

每 micro-batch 每 stage 一次 `send/recv`，$\propto B \cdot d$（activation size）。$M$ 个 micro-batch、$P$ stage。总 P2P 通信 $\propto M \cdot B \cdot d$（跨 stage）。详见 [[1F1B调度]] 与 [[Pipeline Parallel]]。

### 4.4 SP 通信量

`all-gather`/`reduce-scatter` 每 norm 节点一次，$\propto B \cdot d / T$（shard 后小）。减 activation 显存（activation $/T$）。详见 [[Sequence Parallel]]。

### 4.5 MoE all-to-all 通信量

每 MoE 层 `all-to-all`，$\propto \text{token} \cdot d$（token 派发）。$E_{\text{layer}}$ 个 MoE 层。详见 [[MoE token dispatcher]]。

### 4.6 overlap 的收益

通信 $\propto$ tensor size，计算 $\propto$ FLOP。若通信时间 $T_c$ < 计算时间 $T_k$，overlap 后 wall-clock $\approx T_k$（通信藏）。否则 $\approx T_k + T_c$（串行）。见 [[alpha-beta性能模型]] 与 [[overlap strategy]]。


## 5. 代码示例（可选）

### 5.1 Megatron 训练步骨架

```python
# 简化: Megatron Core 训练步
model = GPTModel(...)  # TP/PP/DP 切分
optimizer = DistributedOptimizer(...)  # D-optimier
for batch in dataloader:
    # forward (PP schedule 内部触发 send/recv)
    if pp_stage:
        loss = forward_backward_step(model, batch, pp_schedule="1f1b")
        # 内部: forward (TP all-gather/all-reduce, SP all-gather)
        #       backward (TP reduce-scatter, DP grad all-reduce overlap)
    optimizer.zero_grad()
    loss.backward()  # 触发 backward + grad bucket all-reduce (async)
    optimizer.step()  # D-optimier shard 更新
```

### 5.2 ParallelState 初始化

```python
from megatron.core import parallel_state as ps
ps.initialize_model_parallel(tensor_model_parallel_size=8,
                             pipeline_model_parallel_size=4,
                             expert_model_parallel_size=2)
# 建 tp/pp/dp/ep group
# 各 rank 的 (tp_rank, pp_rank, dp_rank, ep_rank) 坐标
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Megatron-LM]]（框架概述）、[[Tensor Parallel]]/[[Pipeline Parallel]]/[[Distributed Data Parallel|DDP]]（并行范式概念）、[[AllGather]]/[[all-reduce]]/[[reduce-scatter]]/[[all-to-all]]（通信原语）、[[NCCL通信拓扑]]、[[overlap strategy]]、[[compute vs communication bottleneck]]、[[alpha-beta性能模型]]。
- **下游（应用）**: [[ColumnParallelLinear]]/[[RowParallelLinear]]（TP 算子源码）、[[Sequence Parallel]]（SP 源码）、[[1F1B调度]]/[[Interleaved schedule]]（PP 调度源码）、[[Megatron distributed optimizer]]/[[gradient bucket与通信重叠]]（优化器源码）、[[MoE token dispatcher]]/[[grouped GEMM]]（MoE 源码）、[[EP TP DP CP组合规则]]（组合）、[[Megatron distributed checkpoint]]/[[HF与Megatron checkpoint转换]]（checkpoint）、[[DeepSpeed ZeRO vs FSDP vs Megatron]]（对比）。
- **对比 / 易混**:
  - **Megatron Core 源码 vs [[Megatron-LM]] 概述**：概述讲"是什么/为什么"（TP/PP/DP 概念），源码讲"怎么实现"（目录+时序）。概述是入口，源码是深入。
  - **Megatron ParallelState vs [[DeviceMesh]]**：ParallelState 是 Megatron 手写 rank group，DeviceMesh 是 PyTorch 原生 mesh。两者等价概念，Megatron Core 在向 DeviceMesh 靠拢。
  - **Megatron 时序 vs [[Fully Sharded Data Parallel|FSDP]] 时序**：Megatron TP 是每层 all-reduce（通信多但算力切分），FSDP 是前向 all-gather 完整 weight（通信省但算力不切）。时序不同，见 [[DeepSpeed ZeRO vs FSDP vs Megatron]]。


## 7. 常见误区与易错点

> [!warning] 误区 1：Megatron 是单一代码文件
> Megatron Core 是分目录的庞大库（transformer/tensor_parallel/pipeline_parallel/optimizer/moe/...）。读懂先看目录地图，再进具体算子。直接扎进单文件会迷失。

> [!warning] 误区 2：训练步只有 forward+backward
> 训练步内含大量 NCCL 通信（TP all-reduce、DP grad all-reduce、PP send/recv、SP all-gather/reduce-scatter、MoE all-to-all）。通信与计算交织 overlap。忽略通信时序会误判性能。

> [!warning] 误区 3：TP all-reduce 可随意 overlap
> TP 的 row parallel 后 all-reduce 是**同层依赖**（下一层 col linear 需完整 input），难 overlap（串行）。DP 的 grad all-reduce 是**跨层独立**，能 overlap。误以为都能 overlap 会错估性能。

> [!warning] 误区 4：忽视 SP 与 TP 的配合
> [[Sequence Parallel]] 的 all-gather/reduce-scatter 与 TP 的 all-reduce 配合（SP 把 norm 区的 activation 分片，TP 把 weight 区分片）。单独看 TP 或 SP 都不完整。SP 是 TP 的延伸。

> [!warning] 误区 5：ParallelState 与 DeviceMesh 混
> ParallelState 是 Megatron 手写（API 旧），DeviceMesh 是 PyTorch 原生（新）。两者概念等价但 API 不同。Megatron Core 新版在迁移到 DeviceMesh，旧代码仍 ParallelState。

> [!warning] 误区 6：Megatron 用 PyTorch autograd
> Megatron 算子继承 `nn.Module`，用 PyTorch autograd（[[Autograd Engine]]）。但并行通信（all-reduce/all-gather）通过 hook 或 `backward` 自定义插入，不是纯 autograd 自动。

> [!warning] 误区 7：Megatron 时序固定
> 时序随版本演进（如引入 SP、D-optimier、interleaved、aux-loss-free MoE）。读最新版时序，勿照旧版。核心骨架（TP all-reduce/DP grad/PP P2P）不变，细节变。


## 8. 延伸细节

### 8.1 TransformerEngine 集成

Megatron Core 常与 NVIDIA TransformerEngine（TE）集成：TE 提供 fused attention（FlashAttention）、fused layernorm、fused softmax 等 kernel。Megatron 的 `transformer/` 调 TE 的层，TE 内部用 cuDNN/cublas。是 [[fused kernel]] 的实战。

### 8.2 torch.compile 与 Megatron

Megatron 算子是 registered op（部分），[[torch.compile]] 可 trace。但 TP all-reduce 等 NCCL 通信是 graph 特殊 node，[[Inductor]] 不融合。Megatron + compile 是演进方向（部分算子 compile）。

### 8.3 Megatron Core 的开源演进

Megatron-LM（NVIDIA）→ Megatron Core（模块化重写）→ 社区分支（如 Megatron-DeepSpeed）。Core 是主线，向 DTensor/DeviceMesh/FSDP2 靠拢。详见 [[Megatron-LM]] 与 [[DeepSpeed ZeRO vs FSDP vs Megatron]]。

### 8.4 内容来源

Megatron Core 目录与执行链路整理自 `megatron/core/` 源码（GitHub `Megatron-LM/megatron/core/`）、Megatron-LM 论文（Megatron-LM v1-v5）、NVIDIA dev blog。目录结构据 2024-2025 版本。时序据 `forward_backward_step` 与 `schedules.py`。与 DeviceMesh/DTensor 的对照见 [[DeviceMesh]]/[[DTensor]]。

---
相关: [[Megatron Core总览]] | [[Megatron-LM]] | [[Tensor Parallel]] | [[Pipeline Parallel]] | [[Distributed Data Parallel]] | [[AllGather]] | [[all-reduce]] | [[reduce-scatter]] | [[all-to-all]] | [[NCCL通信拓扑]] | [[overlap strategy]] | [[compute vs communication bottleneck]] | [[alpha-beta性能模型]] | [[ColumnParallelLinear]] | [[RowParallelLinear]] | [[Sequence Parallel]] | [[1F1B调度]] | [[Interleaved schedule]] | [[Megatron distributed optimizer]] | [[gradient bucket与通信重叠]] | [[MoE token dispatcher]] | [[grouped GEMM]] | [[EP TP DP CP组合规则]] | [[Megatron distributed checkpoint]] | [[HF与Megatron checkpoint转换]] | [[DeepSpeed ZeRO vs FSDP vs Megatron]] | [[DeviceMesh]] | [[DTensor]] | [[FSDP2]] | [[torch.compile]] | [[Inductor]] | [[fused kernel]] | [[Autograd Engine]] | [[Fully Sharded Data Parallel]]
