# grouped GEMM

> **所属章节**: [[MoE源码]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: grouped GEMM / grouped matmul / GroupedMatmul / BlockSparse GEMM / 变长 batched GEMM / MoE expert kernel
> **难度**: 中高（需懂 [[MoE token dispatcher]] + [[CUTLASS与GEMM]] + [[Triton kernel开发]]）


## 1. 一句话定义

**grouped GEMM（GroupedMatmul / BlockSparse GEMM）** 是 [[Megatron-LM]] MoE 层的**多 expert 批量矩阵乘 kernel**——派发后每 rank 持多个 expert 各自被分配到的 token（每 expert 的 token 数不同，**变长**），若对每 expert 单独 launch 一个 GEMM（loop over experts）则 kernel launch 开销大、小 GEMM 算力利用率低；grouped GEMM 把所有 expert 的 GEMM **打包成一次 kernel 调用**（变长 batched GEMM，每 expert 一组 `[token_m, d] @ [d, d_ffn]`，组数 = expert 数，组内 token 数变长），底层用 **CUTLASS/Triton 的 grouped GEMM kernel**（一次 launch 处理所有组，按组 offset 索引，GPU 并行铺组内 thread block）。还支持 **BlockSparse 变体**（token-expert 赋值用稀疏 block 格式，跳过空 expert-block，零 token expert 不算）。是 MoE expert 计算的核心 kernel，把"派到各 expert 的变长 token"高效算出，是 [[MoE token dispatcher]] 派完后的"算"环节，与 dispatcher 的"派"配套。

> [!note] 三句话定位
> - **是什么**：多 expert 的变长 GEMM 打包一次 kernel 调用（CUTLASS/Triton grouped GEMM），省 launch 开销、提算力利用率，支持 BlockSparse 跳空。
> - **为什么**：MoE 每 rank 多 expert 各被分不同数量 token，loop over experts 单独 GEMM 开销大、小 GEMM 低效；grouped 一次处理省。
> - **与 [[MoE token dispatcher]] 关系**：dispatcher 派 token 到 expert（all-to-all），grouped GEMM 算派完的 expert 输出（matmul）。前者派，后者算。


## 2. 为什么需要它（动机与背景）

### 2.1 MoE expert 的变长 GEMM

[[Mixture of Experts|MoE]] 派发后，每 rank 的多个 expert 各被分到不同数量 token（变长，因负载不均/capacity padding）。每 expert 是一 FFN（`[token_m, d] @ [d, d_ffn]` + act + `[token_m, d_ffn] @ [d_ffn, d]`）。每 expert 的 GEMM 独立，token 数不同。

### 2.2 loop over experts 的低效

朴素：对每 expert launch 一个 GEMM。$E$ 个 expert 则 $E$ 次 launch。每次 launch 有开销（$\mu$s 级），小 GEMM（token 少）算力利用率低（SM 多 idle）。$E$ 大（如 64-256）时 launch 开销累积，小 GEMM 低效。是 MoE 的算力瓶颈。

### 2.3 grouped GEMM 一次打包

grouped GEMM：把 $E$ 个 expert 的 GEMM 打包成一次 kernel 调用。kernel 内部按组（每 expert 一组）offset 索引，GPU 并行铺组内 thread block。一次 launch 处理所有组。省 launch 开销，大组小组都满 SM。

### 2.4 变长支持

每 expert 的 token 数不同（变长）。grouped GEMM 支持变长（每组的 $M$ 维不同）。CUTLASS/Triton 的 grouped GEMM kernel 接受变长（offset + per-group shape）。

### 2.5 BlockSparse 跳空

若某 expert 无 token（空），grouped GEMM 跳过（不铺 thread block）。BlockSparse 格式：token-expert 赋值用稀疏 block（非零 block 才算）。省空 expert 的算力。

### 2.6 CUTLASS/Triton 实现

CUTLASS 的 `GroupedGemm`（C++ 模板，变长 batched）、Triton 的 grouped GEMM（Python 写，autotune）。Megatron 支持两者。是 [[CUTLASS与GEMM]] 与 [[Triton kernel开发]] 在 MoE 的实战。


## 3. 核心概念详解

### 3.1 loop over experts（朴素）

```python
# 朴素: 每 expert 单独 GEMM
for expert in experts:  # E 次
    tokens_e = dispatched_tokens[expert]  # [m_e, d] (变长 m_e)
    out_e = tokens_e @ expert.w1  # [m_e, d_ffn]
    out_e = act(out_e)
    out_e = out_e @ expert.w2  # [m_e, d]
    # E 次 kernel launch, 小 GEMM 低效
```

$E$ 次 launch，每 expert 独立 GEMM。launch 开销累积，小 GEMM 低效。

### 3.2 grouped GEMM（打包）

```python
# grouped: 一次 kernel 调用处理所有 expert
inputs = [dispatched_tokens[e] for e in experts]  # 变长 list
weights1 = [expert.w1 for expert in experts]
# grouped GEMM (变长 batched)
inter = grouped_gemm(inputs, weights1)  # 一次 kernel, 每组 GEMM
inter = act(inter)
out = grouped_gemm(inter, [expert.w2 for expert in experts])  # 一次 kernel
# 2 次 kernel launch (vs 2E), 满算力
```

一次 kernel 处理所有 expert。2 次 launch（gate_up + down，vs 2E）。省 launch，提算力。

### 3.3 变长的支持

```
组 0 (expert 0): [m_0, d] @ [d, d_ffn] -> [m_0, d_ffn]
组 1 (expert 1): [m_1, d] @ [d, d_ffn] -> [m_1, d_ffn]  (m_1 ≠ m_0)
...
每组的 M (token 数) 不同. kernel 据 per-group offset + shape 索引.
```

grouped GEMM 接受变长（per-group $M$）。kernel 内部按组 offset 跳到该组数据，铺组内 thread block 算该组 GEMM。

### 3.4 BlockSparse 跳空

```
若 expert 3 无 token (m_3 = 0):
  朴素: 仍 launch (空 GEMM, 浪费)
  grouped: 跳过组 3 (不铺 thread block)
  BlockSparse: token-expert 赋值用稀疏 block, 非零 block 才算
```

BlockSparse 格式：token-expert 赋值矩阵（`[token, expert]`）是稀疏的（每 token 选 k 个，非满），用 block 稀疏（如 16x16 block）存储非零 block。grouped GEMM 只算非零 block。省空 expert。

### 3.5 CUTLASS grouped GEMM

```cpp
// CUTLASS GroupedGemm (C++ 模板)
cutlass::gemm::device::GemmGrouped<...> gemm;
// 接受: per-group ptr (A, B, C), per-group M/N/K, group 数
// 内部: 一次 launch, 按 group offset 索引, 铺 thread block
```

CUTLASS 的 `GemmGrouped` 是变长 batched GEMM。Megatron 用 CUTLASS 的 grouped GEMM（C++ 扩展）。是 [[CUTLASS与GEMM]] 的应用。

### 3.6 Triton grouped GEMM

```python
# Triton grouped GEMM (Python, autotune)
@triton.autotune(...)
@triton.jit
def grouped_gemm_kernel(a_ptr, b_ptr, c_ptr, group_offsets, Ms, Ns, Ks, ...):
    group = tl.program_id(0)  # 哪组
    m = tl.program_id(1)  # 组内 block
    # 索引该组 offset, 算 GEMM block
    ...
```

Triton 的 grouped GEMM（Python 写，autotune）。Megatron 支持 Triton grouped GEMM（与 CUTLASS 二选一）。是 [[Triton kernel开发]] 的应用。

### 3.7 与 dispatcher 的配合

```
dispatcher.dispatch(tokens) -> 每 rank 持 [sum(m_e), d] (各 expert token 拼接)
grouped_gemm(dispatched, expert_weights) -> [sum(m_e), d_ffn] (各 expert 算)
dispatcher.combine(expert_output) -> 派回原 rank, 加权累加
```

dispatcher 派（变长 token 到 expert），grouped GEMM 算（变长 batched），dispatcher 合（派回 + 加权）。三者配套。

### 3.8 padding 的等大 vs 变长

- **等大（padding）**：每 expert padding 到 capacity（等大 $M$）。grouped GEMM 等大（每组 $M$ 同），简单（batched GEMM）。padding 浪费算力（空 token）。
- **变长（无 padding）**：每 expert 真实 $m_e$（变长）。grouped GEMM 变长，省 padding 算力，kernel 稍复杂。

CUTLASS/Triton 支持变长。Megatron 默认变长（省 padding）。

### 3.9 group-limited 的支持

DeepSeek 的 group-limited MoE（expert 分组）需 group-aware grouped GEMM（按 group 分批算）。grouped GEMM 支持 per-group 配置。是 DeepSeek-MoE 的算力支持。

### 3.10 实现的 grouped_gemm.py

```python
# megatron/core/moe/grouped_gemm.py
def grouped_gemm(inputs, weights, ...):
    # 选 CUTLASS 或 Triton backend
    if backend == "cutlass":
        return cutlass_grouped_gemm(inputs, weights, ...)  # C++ 扩展
    elif backend == "triton":
        return triton_grouped_gemm(inputs, weights, ...)  # Triton kernel
```


## 4. 数学原理 / 公式

### 4.1 expert 的 GEMM

每 expert $e$：$Y_e = X_e W_{1,e}, \quad X_e \in \mathbb{R}^{m_e \times d}, W_{1,e} \in \mathbb{R}^{d \times d_{\text{ffn}}}, Y_e \in \mathbb{R}^{m_e \times d_{\text{ffn}}}$。$m_e$ 变长（每 expert 被分 token 数）。$E$ 个 expert 共 $E$ 组 GEMM。

### 4.2 grouped 的总 FLOP

$$\text{FLOP} = \sum_{e=1}^{E} 2 m_e \cdot d \cdot d_{\text{ffn}} \cdot 2 \quad (\text{gate\_up + down, } \times 2)$$

变长，总 FLOP = 各 expert 之和。grouped 一次算完。

### 4.3 launch 开销的省

朴素：$2E$ 次 launch（gate_up + down 各 $E$）。grouped：2 次（gate_up + down 各 1）。省 $2E - 2$ 次 launch。每次 launch $\sim \mu$s。$E$ 大（如 64）时省显著。

### 4.4 算力利用率

朴素小 GEMM（$m_e$ 小）SM 不满（idle）。grouped 把所有组铺满 SM（组间并行）。算力利用率提。大 $E$ + 小 $m_e$ 时增益大。

### 4.5 BlockSparse 的省

空 expert（$m_e = 0$）朴素仍 launch（浪费）。grouped BlockSparse 跳过（不铺 block）。省空 expert 的 launch + 算力。

### 4.6 padding 的浪费

等大 padding 到 capacity：每 expert padding 补零到 $C$。FLOP $= E \cdot 2 C \cdot d \cdot d_{\text{ffn}} \cdot 2$。padding 部分（$C - m_e$）算零（浪费）。变长省 padding。$m_e \ll C$ 时 padding 浪费大。


## 5. 代码示例（可选）

### 5.1 Megatron grouped GEMM

```python
from megatron.core.moe.grouped_gemm import grouped_gemm
# dispatched: 每 rank 的各 expert token (变长, 拼接 [sum(m_e), d])
# expert_weights1: [E, d, d_ffn] (每 expert 一 weight)
inter = grouped_gemm(dispatched, expert_weights1, group_size=[m_0, m_1, ...])  # 一次 kernel
inter = act(inter)
out = grouped_gemm(inter, expert_weights2, group_size=[m_0, m_1, ...])  # 一次 kernel
```

### 5.2 CUTLASS grouped GEMM（C++ 概念）

```cpp
// CUTLASS GemmGrouped
cutlass::gemm::device::GemmGrouped<...>::Arguments args(
    {E, /*group_count*/},
    {per_group_A_ptr, per_group_LDA},  // 变长
    {per_group_B_ptr, per_group_LDB},
    {per_group_C_ptr, per_group_LDC}
);
gemm(args, stream);  // 一次 launch
```

### 5.3 Triton grouped GEMM（Python 概念）

```python
@triton.jit
def grouped_gemm_kernel(a_ptr, b_ptr, c_ptr, group_offsets, Ms, Ns, Ks, ...):
    group = tl.program_id(0)  # 组
    m_blk = tl.program_id(1)  # 组内 block
    # 索引该组
    a = a_ptr + group_offsets[group] + ...
    b = b_ptr + group * N * K + ...
    # GEMM block
    ...
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[MoE token dispatcher]]（派 token）、[[Mixture of Experts|MoE]]（MoE 概念）、[[CUTLASS与GEMM]]（CUTLASS grouped GEMM）、[[Triton kernel开发]]（Triton grouped GEMM）、[[GEMM]]（基础 matmul）、[[Megatron Core目录与执行链路]]。
- **下游（应用）**: [[Megatron-LM]] 的 MoE expert 计算、DeepSeek-MoE/MoE 模型训练、MoE 推理（[[TP PP DP EP推理]]）、[[fused kernel]]（grouped GEMM 是融合 kernel）。
- **对比 / 易混**:
  - **grouped GEMM vs loop over experts**：前者一次打包（省 launch、提算力），后者逐 expert（开销大、小 GEMM 低效）。grouped 是 MoE 标配。
  - **grouped GEMM vs batched GEMM**：batched GEMM 等大（每组 $M$ 同），grouped 变长（每组 $M$ 不同）。grouped 是变长 batched。MoE 用 grouped（变长）。
  - **grouped GEMM vs [[MoE token dispatcher]]**：dispatcher 派 token（all-to-all），grouped GEMM 算 expert（matmul）。前者派，后者算。
  - **CUTLASS grouped vs Triton grouped**：CUTLASS 是 C++ 模板（性能优，难改），Triton 是 Python（易改，autotune）。两者可选。是 [[CUTLASS与GEMM]] vs [[Triton kernel开发]] 的对照。


## 7. 常见误区与易错点

> [!warning] 误区 1：grouped GEMM 是新算子
> grouped GEMM 是**打包多个 GEMM 成一次 kernel**，底层仍是 GEMM（CUTLASS/Triton）。不是新算子，是组织方式（变长 batched）。算的还是 matmul。

> [!warning] 误区 2：等大 padding 必需
> 等大 padding 是简化（batched GEMM 等大），但浪费算力（空 token）。CUTLASS/Triton 支持变长（无 padding），省算力。Megatron 默认变长。误以为必需 padding 会浪费。

> [!warning] 误区 3：grouped GEMM 减 FLOP
> grouped 不减 FLOP（总 $\sum$ 各 expert 仍同）。减的是 launch 开销 + 提算力利用率（小 GEMM 满铺 SM）。误以为减 FLOP 会错判。

> [!warning] 误区 4：空 expert 仍算
> 空 expert（$m_e = 0$）grouped BlockSparse 跳过（不铺 block）。朴素 loop 仍 launch（浪费）。grouped 跳空是优化。

> [!warning] 误区 5：CUTLASS 与 Triton 等价
> 性能相近（都高效），但 CUTLASS 是 C++ 模板（难改，NVIDIA 优化），Triton 是 Python（易改，autotune）。场景不同：极致性能 CUTLASS，灵活可改 Triton。Megatron 支持两者。

> [!warning] 误区 6：grouped GEMM 与 dispatcher 独立
> 不独立。dispatcher 派 token 的格式（拼接 [sum(m_e), d] + group_size [m_0, m_1, ...]）是 grouped GEMM 的输入格式。两者配套。dispatcher 的输出格式定 grouped 的输入。

> [!warning] 误区 7：grouped GEMM 不支持变长
> 支持。CUTLASS `GemmGrouped` 与 Triton grouped 都支持变长（per-group $M$）。是 grouped 的核心特性（vs 等大 batched）。误以为不支持会用 padding 浪费。


## 8. 延伸细节

### 8.1 grouped GEMM 的 SM 铺排

grouped GEMM 的 thread block 铺排：每组的 $m_e$ 切 block，block 沿组内 $M$ 与 $N$ 铺。组间并行（不同组的 block 独立）。SM 分配组间+组内 block。是 [[occupancy分析]] 的应用。

### 8.2 与 [[ncu (Nsight Compute)]] 调优

grouped GEMM 是 MoE 性能关键。用 [[ncu (Nsight Compute)]] 看 SM utilization、memory throughput，调 block size / autotune。是 kernel 调优实战。

### 8.3 与 fused kernel 的关系

grouped GEMM 是融合 kernel（多 GEMM 一次）。可与 activation（act）融合（gate_up + act 一 kernel）。是 [[fused kernel]] 在 MoE 的应用。

### 8.4 内容来源

grouped GEMM 整理自 Megatron-LM `megatron/core/moe/grouped_gemm.py` 源码、CUTLASS `GemmGrouped` 文档、Triton `04-fused-attention` 等示例。变长与 BlockSparse 见 CUTLASS 文档。与 dispatcher 的配合见 [[MoE token dispatcher]]。

---
相关: [[MoE源码]] | [[MoE token dispatcher]] | [[Mixture of Experts]] | [[CUTLASS与GEMM]] | [[Triton kernel开发]] | [[GEMM]] | [[fused kernel]] | [[Megatron Core目录与执行链路]] | [[occupancy分析]] | [[ncu (Nsight Compute)]] | [[TP PP DP EP推理]] | [[alpha-beta性能模型]]
