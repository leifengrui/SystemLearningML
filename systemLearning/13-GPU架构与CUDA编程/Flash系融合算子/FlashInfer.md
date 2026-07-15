# FlashInfer

> **所属章节**: [[Flash系融合算子]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: FlashInfer / flash-infer / 变长 attention 库 / block-sparse KV attention / Ragged attention
> **难度**: 高（需懂 [[FlashAttention]] + [[KV cache]] + 推理场景）


## 1. 一句话定义

**FlashInfer** 是为 LLM 推理设计的**高性能 attention kernel 库**（基于 [[FlashAttention]] 的 tiling + online softmax 思想，但扩展到推理的真实场景）——支持**变长 batch**（每序列长度不同，ragged tensor）、**block-sparse / hierarchical KV cache**（只算非零块、复用前缀）、**多级 cache 复用**（prefix cache、跨请求复用）、**PagedAttention 兼容**（按 block table 索引非连续 KV）、**多种 attention 变体**（causal / sliding window / bidirectional / multi-query / grouped-query）。它是 vLLM/SGLang/[[model runner]] 的 attention 路径底层依赖之一，解决 [[FlashAttention]]（原为训练设计，假设规整 batch）在推理变长/稀疏/分页场景下的工程缺口。

> [!note] 三句话定位
> - **是什么**：Flash 思想（tiling + online softmax + 减 HBM IO）的推理工程化扩展，支持变长/稀疏/分页 KV。
> - **为什么**：推理的 batch 变长（每请求不同长）、KV 分页（[[PagedAttention]]）、前缀复用（[[Radix Tree prefix cache]]）——训练版的规整 Flash 不直接适用。
> - **与 [[FlashAttention]] 关系**：FA 是算法 + 训练场景 kernel；FlashInfer 是推理场景的工程库，底层思想同源，扩展变长/稀疏/分页。


## 2. 为什么需要它（动机与背景）

### 2.1 推理 batch 是变长的

训练的 batch 规整（同长度、padding 到 max）。推理的 continuous batching（[[continuous batching的调度实现]]）把**不同长度、不同 decode 阶段**的请求拼一 batch：A 在 prefill（5000 token）、B 在 decode（刚生成第 1 个）、C 在 decode（第 200 个）……KV 长度全不同。训练版的 Flash（假设规整 $B\times N\times d$）要 padding 到最长，浪费严重。

FlashInfer 用 **ragged tensor**（变长，不 padding）+ per-request 的 length array，每序列按真实长度算，零 padding 浪费。

### 2.2 KV cache 是分页的

vLLM 的 [[PagedAttention]] 把 KV 切成固定 block（如 16 token），用 block table 映射逻辑→物理（非连续）。attention kernel 要按 block table 索引 KV block 进 smem，不能假设 KV 连续。训练版 Flash 假设连续，不直接适用。

FlashInfer 支持 paged KV（输入 block table + 指针数组），与 [[block manager]] 的分配协同。

### 2.3 前缀复用（prefix cache）

多请求共享相同 system prompt / few-shot 前缀（如都带 "You are a helpful assistant..."）。[[Radix Tree prefix cache]] 把前缀 KV 复用，不重算。attention kernel 要支持"前缀 KV 已算好、本次只算增量"。FlashInfer 提供 hierarchical KV 接口，前缀 KV 从 cache 读、增量 KV 实算。

### 2.4 block-sparse 长上下文

长上下文（100K+ token）常用 sliding window / block-sparse（只算近 $W$ 个 + sparse 全局块）。FlashInfer 支持 block-sparse mask（只载入非零 KV block 进 smem），跳过零块，进一步减 IO。

### 2.5 多变体统一

causal（自回归 decode）、bidirectional（encoder 如 BERT/嵌入模型）、sliding window（Mistral/Llama 长）、MQA/GQA（多/分组 query 共享 KV）——FlashInfer 用统一 kernel + 模板参数支持，避免每变体写一 kernel。


## 3. 核心概念详解

### 3.1 ragged / 变长 batch

```python
# 不 padding, 用 length array
# q_len = [1,1,1,...] (decode), k_len = [200, 50, 5000, ...] (变长 KV)
# FlashInfer 的 BatchPrefillWithRaggedKVInput / BatchDecodeWithPagedKVCache
```

每序列按真实长度算，不 padding。save FLOP + HBM（padding 部分不读不算）。

### 3.2 paged KV 接口

```python
# block_table: [num_req, max_blocks], 每元素是物理 block id
# k_cache, v_cache: [num_blocks, block_size, num_kv_heads, head_dim] 物理
# FlashInfer 用 block_table 索引 KV block 进 smem
```

与 [[PagedAttention]] 的 block table 一致。FlashInfer 是其 attention kernel 实现。

### 3.3 hierarchical KV（prefix + 增量）

```python
# 前缀 KV: 从 L2/cache 复用 (不重算)
# 增量 KV: 本批新 token 实算
# FlashInfer 的 BatchDecodeWithSharedPrefixHNBackend 等
```

[[Radix Tree prefix cache]] 的底层。

### 3.4 block-sparse mask

mask 是 block 级的 bool 矩阵（哪些 KV block 参与）。FlashInfer 只载入 mask=True 的 block 进 smem，跳过 False。sliding window 是 mask 的特例（只最近 $W$）。

### 3.5 在线 + 离线 plan

FlashInfer 的 kernel 常有 "plan" 阶段（按 batch 的 length/block_table 预计算调度元数据）+ "run" 阶段（真算）。plan 一次（CPU 或 GPU），run 多次（每次 decode）。把变长处理的 CPU 开销移出 hot loop。

### 3.6 多变体统一

| 变体 | 场景 | mask |
|---|---|---|
| causal | 自回归 decode | 下三角 |
| bidirectional | encoder / 嵌入 | 全参与 |
| sliding window | 长上下文 | 最近 $W$ |
| block-sparse | 长上下文 + 全局 sparse | block bool |
| MQA/GQA | 共享 KV | query 分组共享 |

FlashInfer 用模板参数 + runtime flag 在统一 kernel 里支持，见 [[Triton kernel开发]] 的 autotune 思想。

### 3.7 与 vLLM/SGLang 集成

- vLLM：[[PagedAttention]] 的 v1/v2 用 FlashInfer 或自研 attention kernel，decode 路径依赖。vLLM 的 [[model runner]] 调 FlashInfer 的 BatchDecodeWithPagedKVCache。
- SGLang：[[Radix Tree prefix cache]] 的 hierarchical KV 用 FlashInfer 的 shared-prefix 接口。
- 两者把 FlashInfer 当 attention 后端，自己做 scheduler/block manager/routing。

### 3.8 与 FA3 的分工

- [[FlashAttention]]-3：训练/规整 batch 的极致（Hopper TMA/warp spec/FP8）。
- FlashInfer：推理变长/稀疏/分页的工程化，覆盖更广 GPU（不只 Hopper）。

二者底层思想同源（tiling + online softmax + 减 IO），场景互补。


## 4. 数学原理 / 公式

### 4.1 变长 batch 的 IO

规整 padding batch：每序列按 max_len $L_{\max}$ 算，HBM $\propto B \cdot L_{\max}^2$。变长不 padding：$\sum_i L_i^2 \le B L_{\max}^2$（Jensen）。长 batch 差异大时省几倍 IO。

### 4.2 paged 的 block 级访问

设 block size $b$，序列 $i$ 的 KV 占 $\lceil L_i/b \rceil$ 块。attention tiling 按 block 索引进 smem：

$$
S_{i,\text{block }j} = Q_i K_{\text{block}(\text{table}[i,j])}^T / \sqrt{d}
$$

物理 block 不连续，但 smem 内逻辑连续，online softmax 正常。IO $\propto L_i$（每 block 进一次 smem）。

### 4.3 prefix 复用的省算

前缀长度 $L_p$，增量 $L_d$。不复用：全 $L_p + L_d$ 重算。复用：前缀 KV 从 cache 读（不重算 $QK^T$），只算增量 $L_d$。省 $\sim L_p/(L_p+L_d)$ 的 FLOP（但 IO 仍读前缀 KV 用于 $PV$）。

### 4.4 block-sparse 的省

block-sparse 密度 $\rho$（非零块占比）。IO/FLOP $\propto \rho \cdot N$（vs 全参与 $\propto N$）。$\rho=0.1$ 省 10×。

### 4.5 与 [[Roofline模型]] 的契合

FlashInfer 的所有优化（变长去 padding、paged 减 IO、prefix 复用、block-sparse）都是减 HBM byte → 提 $I$ → 从 memory-bound（decode 天然低 $I$）趋 compute-bound。与 [[FlashAttention]] 同理。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟变长 batch

```python
import numpy as np
def batched_attn_ragged(Q_list, K_list, V_list):
    """变长 batch: 每序列按真实长度算, 不 padding."""
    outs = []
    for q, k, v in zip(Q_list, K_list, V_list):
        S = q @ k.T / np.sqrt(q.shape[-1])
        P = np.exp(S - S.max(-1, keepdims=True)) / np.exp(S-S.max(-1,keepdims=True)).sum(-1, keepdims=True)
        outs.append(P @ v)
    return outs

# padding 版对比: 全 batch padding 到 max_len, 浪费
def batched_attn_padded(Qpad, Kpad, Vpad, lengths):
    # 多数序列的 padding 部分也读不算 (但占 HBM/算)
    S = Qpad @ Kpad.transpose(0,2,1) / np.sqrt(Qpad.shape[-1])
    # 实际要 mask 掉 padding, 但 HBM 仍读全
    ...
# 变长省 padding 的 IO
```

### 5.2 真实 FlashInfer 调用（伪代码）

```python
import flashinfer
# paged decode: block_table + paged KV
plan = flashinfer.BatchDecodeWithPagedKVCachePlan()
plan.plan(block_table, seq_lens, num_q_heads, num_kv_heads, head_dim, ...)
o = plan.run(q, paged_kv_cache)        # 变长 batch decode
# shared prefix (SGLang 风格)
sp = flashinfer.BatchDecodeWithSharedPrefixHNBackend(...)
o = sp.run(q, paged_kv_cache, suffix_kv)
```

### 5.3 性能对照

```python
# nsys: FlashInfer 的 _flash_attn_varlen kernel vs padding 的 naive
# 变长 batch 长度差异大时 FlashInfer 省 2-5x IO
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[FlashAttention]]（算法基础）、[[GPU内存层级]]（smem tiling）、[[KV cache]]（paged/变长 KV 的语义）、[[多头注意力]]/[[自注意力]]。
- **下游（应用）**: [[PagedAttention]]（attention kernel 实现）、[[block manager]]（KV block 分配协同）、[[Radix Tree prefix cache]]（hierarchical KV）、[[continuous batching的调度实现]]（变长 batch）、[[model runner]]（调 FlashInfer）、vLLM/SGLang decode 路径。
- **对比 / 易混**:
  - **FlashInfer vs [[FlashAttention]]-3**：见 3.8，推理工程库 vs 训练极致。
  - **FlashInfer vs [[PagedAttention]]**：PagedAttention 是 KV 分页管理 + block table；FlashInfer 是支持这种分页的 attention kernel 库。常并用。
  - **FlashInfer vs vLLM 自研 attention**：vLLM 早期自研 PagedAttention kernel，后部分切到 FlashInfer（更通用、多后端）。两者在 vLLM/SGLang 里是可选后端。


## 7. 常见误区与易错点

> [!warning] 误区 1：把 FlashInfer 与 FlashAttention 混为一谈
> FA 是训练规整场景的算法 + kernel；FlashInfer 是推理变长/稀疏/分页的工程库。思想同源，场景与 API 不同。

> [!warning] 误区 2：忽视变长去 padding 的收益
> continuous batching 的 batch 长度差异大时，padding 到 max 浪费严重（HBM 读 padding、FLOP 算 padding）。变长（ragged）省几倍。是推理吞吐的关键。

> [!warning] 误区 3：paged KV 的 attention 当连续处理
> 物理非连续，必须按 block table 索引。直接当连续读会越界/错位。FlashInfer 的 plan 阶段处理 block table 元数据。

> [!warning] 误区 4：忽视 plan/run 两阶段
> 变长/paged 的元数据（length、block table）每 batch 变，plan 阶段算调度（CPU/GPU），run 阶段真算。把 plan 移出 hot loop 是性能关键。每次 decode 都 plan 反而劣化。

> [!warning] 误区 5：prefix 复用当成免费
> 前缀 KV 不重算 $QK^T$，但 $PV$ 仍要读前缀 KV 参与。省的是 FLOP，不是 IO 全免。prefix 太长时 IO 仍可观。

> [!warning] 误区 6：忽视 block-sparse 的 mask 准备
> block-sparse 的 mask 每 batch 变（如 sliding window 随 decode 移动），plan 阶段要更新。mask 准备错位会算错 attention。


## 8. 延伸细节

### 8.1 FlashInfer 的多后端

FlashInfer 提供 CUDA（手写，极致）与 [[Triton kernel开发]]（易改）两种后端。CUDA 后端极致性能，Triton 后端易扩展新变体。框架可按需选。

### 8.2 append-only KV 与 plan 缓存

decode 的 KV 每 step 加一 block（append-only），block table 几乎不变。FlashInfer 的 plan 可缓存（per batch_size bucket），decode 内复用，每新请求/完成才重 plan。

### 8.3 与 [[推理 CUDA Graph]] 的协同

decode 的 attention kernel 也在 CUDA Graph 里 replay（shape 固定 batch × 1）。FlashInfer 的 plan 元数据要随 batch_size 切 graph bucket。与 [[block manager]] 的 KV 分配、graph 池管理协同，是 vLLM/SGLang decode 的工程复杂点。

### 8.4 长上下文的 sliding + global sparse

Llama 3/Mistral 长上下文常 sliding window + sparse global block（如每 $k$ 个 token 取一个全局）。FlashInfer 的 block-sparse mask 支持这种混合。是 100K+ 上下文推理能跑的底层。

### 8.5 内容来源

FlashInfer 的变长/paged/sparse 接口整理自 FlashInfer 项目文档与 examples，与 [[PagedAttention]]/[[Radix Tree prefix cache]] 的协同见 vLLM/SGLang 源码注释。

---
相关: [[Flash系融合算子]] | [[FlashAttention]] | [[GPU内存层级]] | [[Roofline模型]] | [[KV cache]] | [[PagedAttention]] | [[block manager]] | [[Radix Tree prefix cache]] | [[continuous batching的调度实现]] | [[model runner]] | [[Triton kernel开发]] | [[推理 CUDA Graph]]
