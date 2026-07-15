# PagedAttention

> **所属章节**: [[PagedAttention]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: PagedAttention / 分页注意力 / block-level KV / 非连续 KV cache / 虚拟内存类比 KV / paged KV
> **难度**: 中（需懂 [[KV cache]] + [[KV cache management]] + OS 虚拟内存 + [[GPU内存层级]]）


## 1. 一句话定义

**PagedAttention** 是 vLLM 的**核心 KV cache 管理机制**——把 KV cache 按**固定大小的 block（页，如 16 token 一块）**分页存储，每个请求的 KV cache 由**一组逻辑上连续但物理上非连续的 block** 组成（block table 记录"逻辑页号→物理 block 地址"映射，类比 OS 虚拟内存的页表），attention kernel（PagedAttention V1/V2）直接在**非连续物理 block** 上做 attention（block-level 循环，每 block 算 partial attention 再累加），从而**消除 KV cache 的显存碎片**（传统连续 KV cache 因请求长度不一产生内部/外部碎片，浪费 60-80% 显存），让显存利用率从 ~20% 提到 ~90%，**并发请求数提升 2-4 倍**。是 vLLM 的奠基技术（SOSP 2023），也是 [[KV block manager]] 的计算底座——block manager 管 block 分配/回收/换出，PagedAttention 算非连续 block 上的 attention。与 [[KV cache management]]（ch08 概念/策略层）互为表里：ch08 讲为什么需要管 KV，本笔记讲 vLLM 怎么管（分页 + 非连续 kernel）。

> [!note] 三句话定位
> - **是什么**：KV cache 分页（block 16 token）+ block table 逻辑→物理映射 + attention kernel 直接在非连续 block 上算（block-level 循环累加），类比 OS 虚拟内存。消除碎片，显存利用率 20%→90%。
> - **为什么**：传统连续 KV cache 因请求长度不一碎片严重（浪费 60-80%），并发受限；分页让每 block 独立分配，无内部碎片（按需给 block），无外部碎片（block 可拼任意请求）。
> - **与 [[KV cache management]] 关系**：ch08 是概念/策略（为什么管、管什么），本笔记是 vLLM 源码实现（分页数据结构 + 非连续 attention kernel）。是 ch08 的落地。


## 2. 为什么需要它（动机与背景）

### 2.1 传统连续 KV cache 的碎片

传统推理：每请求预分配**连续** KV cache（max_seq_len 大小）。但请求实际长度不一（有的 100 token，有的 2000），预分配 max 导致：
- **内部碎片**：预分配 max 但实际用少（如 max=2048，实际 100，浪费 1948）。各请求碎片累加浪费 60-80% 显存。
- **外部碎片**：请求释放后留空洞，新请求放不下（虽总显存够）。碎片严重，并发受限。

### 2.2 OS 虚拟内存的类比

OS 用**分页**解决内存碎片：进程虚拟地址连续，物理页非连续（页表映射）。按需分配页（用到才给），无内部碎片（页固定小），无外部碎片（页可拼任意进程）。vLLM 把这思路搬到 KV cache。

### 2.3 分页 KV cache

PagedAttention：KV cache 按 block（页，如 16 token）分页。每请求的 KV 由一组 block 组成，逻辑连续（block table 记录顺序），物理非连续（block 散布在 KV pool）。按需给 block（生成到才给），无内部碎片（block 小），无外部碎片（block 可拼任意请求）。

### 2.4 非连续 attention kernel

传统 attention 假设 KV 连续（一指针 + offset 算）。PagedAttention 的 KV 非连续（多 block 散布）。需**改 attention kernel**：block-level 循环，每 block 取 K/V 算 partial attention（partial softmax），累加各 block。是 PagedAttention V1（block 循环 in CUDA）/V2（优化 block 调度）。

### 2.5 显存利用率的提升

传统连续：~20%（碎片浪费 80%）。PagedAttention：~90%（无碎片）。并发请求数 2-4×。是 vLLM 高吞吐的奠基。


## 3. 核心概念详解

### 3.1 block（页）

```
block_size = 16 (token/block, 典型)
KV pool: 预分配 N 个 block (物理 block, 散布)
每 block: [16, num_kv_heads, head_dim] 的 K + V (连续 within block)
```

block 是 KV 的最小分配单位。block 内连续，block 间非连续。类比 OS page。

### 3.2 block table

```
请求 A 的 block table:
  logical_page 0 -> physical_block 42
  logical_page 1 -> physical_block 7
  logical_page 2 -> physical_block 103
  ...
逻辑 token 0-15 -> block 42, token 16-31 -> block 7, ...
```

block table 记录逻辑页→物理 block 映射。类比 OS page table。每请求一 table。是 [[KV block manager]] 的核心数据结构。

### 3.3 非连续 attention（PagedAttention kernel）

```python
# 伪代码: PagedAttention (block-level)
Q = new_token.query  # [num_heads, head_dim]
for block_idx in block_table:  # 遍历物理 block
    K_block = KV_pool[block_idx].K  # [16, num_kv_heads, head_dim]
    V_block = KV_pool[block_idx].V
    scores = Q @ K_block  # partial attention (16 token)
    partial_softmax(scores)  # online softmax (max + sum)
    acc += scores @ V_block  # 累加
output = normalize(acc)  # final softmax
```

block-level 循环，每 block 算 partial attention，累加。用 online softmax（数值稳定，不需先算全局 max）。是 PagedAttention V1 的 CUDA kernel 逻辑。

### 3.4 PagedAttention V1 vs V2

| 版本 | 特点 | 性能 |
|---|---|---|
| V1 | block 循环在 CUDA kernel，每 thread block 算一 sequence | 快（vs 传统） |
| V2 | 优化 block 调度，split-K（长序列并行），减尾延迟 | 更快（长序列 2×） |

V2 优化长序列（split-K 把 block 并行度提），减 decode 的尾延迟。是 vLLM 默认。

### 3.5 在线 softmax（online softmax）

非连续 block 累加需**数值稳定**的 softmax。传统 softmax 先算全局 max（需先扫所有 block）。PagedAttention 用 online softmax：每 block 算时维护 running max + running sum，最后归一化。公式见 §4。是 block-level attention 的关键。

### 3.6 按需分配 block

```
请求 A prefill 100 token -> 需 ceil(100/16)=7 block
请求 A decode 到 101 -> 需第 8 个 block (第 7 block 满 16, 新 token 需新 block)
block manager 从 free list 给 block, 记入 A 的 block table
请求 A 结束 -> block 回收 free list
```

按需给 block（生成到才给）。无预分配 max 的内部碎片。block 回收后可给其他请求（无外部碎片）。

### 3.7 与连续 KV cache 的对比

| 维度 | 传统连续 KV | PagedAttention |
|---|---|---|
| 分配 | 预分配 max_seq_len | 按需 block（16 token） |
| 物理布局 | 连续 | 非连续（block 散布） |
| 内部碎片 | 严重（max-实际） | 无（block 小） |
| 外部碎片 | 严重（空洞） | 无（block 可拼） |
| 显存利用 | ~20% | ~90% |
| attention kernel | 标准（连续指针） | PagedAttention（block 循环） |
| 并发 | 受碎片限 | 2-4× |

### 3.8 copy-on-write（prefix sharing）

```
请求 A: "System: You are..." + "Question A"
请求 B: "System: You are..." + "Question B"
共享前缀 "System: You are..." 的 block (copy-on-write)
A 和 B 的 block table 指向同一物理 block
B 分叉时 (不同 token) 才 copy 新 block
```

共享前缀的 block 可被多请求指向（引用计数）。分叉时 copy-on-write（新 block）。是 prefix caching 的底层支持。详见 [[Radix Tree prefix cache]] 与 [[KV block manager]]。


## 4. 数学原理 / 公式

### 4.1 传统连续 KV 的碎片浪费

设 max_seq_len $L_{\max}$，请求实际长度 $L_i$。预分配 $L_{\max}$。内部碎片率：

$$\text{frag} = 1 - \frac{\sum_i L_i}{\sum_i L_{\max}} = 1 - \frac{\bar{L}}{L_{\max}}$$

若 $\bar{L} = 200$, $L_{\max} = 2048$，碎片率 $\approx 90\%$（显存利用仅 10%）。实测 vLLM 报告传统 ~20%（因 padding 到桶长略好）。

### 4.2 PagedAttention 的显存利用

PagedAttention 只给实际 token 的 block：

$$\text{util} = \frac{\sum_i \lceil L_i / B \rceil \cdot B}{\text{KV pool 总 block} \cdot B} \approx \frac{\sum_i L_i}{\text{pool}} \approx 90\%$$

$B$=block_size（16）。碎片仅 $\lceil \rceil$ 的尾部（$< B$ token，最多 15），可忽略。显存利用 ~90%。

### 4.3 在线 softmax（数值稳定）

传统 attention：$\text{attn}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) V$，需先算全局 max $\max_j (QK_j)$。

PagedAttention block 循环，用 online softmax：维护 running max $m$ + running sum $l$：

$$m_{\text{new}} = \max(m_{\text{old}}, \max_{j \in \text{block}} s_j), \quad l_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} l_{\text{old}} + \sum_{j \in \text{block}} e^{s_j - m_{\text{new}}}$$

$$\text{acc}_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \text{acc}_{\text{old}} + \sum_{j \in \text{block}} e^{s_j - m_{\text{new}}} V_j$$

最后 $o = \text{acc} / l$。每 block 局部算 max/sum/acc，用 running max 归一。数值稳定（不需先扫全局）。与 [[Flash Attention]] 的 online softmax 同源。

### 4.4 KV cache 显存

$$M_{\text{KV}} = 2 \cdot L \cdot n_{\text{kv}} \cdot d_h \cdot b \cdot \text{num\_block}$$

$2$=K+V，$L$=block_size，$n_{\text{kv}}$=KV head 数（GQA 少），$d_h$=head dim，$b$=byte（bf16=2）。block 数 = pool 容量。详见 [[KV cache]]。


## 5. 代码示例（可选）

### 5.1 vLLM PagedAttention 调用

```python
from vllm.attention.backends.flash_attn import FlashAttentionBackend
# vLLM 内部用 PagedAttention kernel (block-level)
# attention 输入: Q, K, V, block_table, seq_len
output = paged_attention(
    query=q,                    # [num_tokens, num_heads, head_dim]
    key_cache=KV_pool.key,      # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache=KV_pool.value,
    block_table=block_table,    # [num_requests, max_num_blocks] 逻辑->物理
    seq_lens=seq_lens,
)
```

### 5.2 block table 结构

```python
# 每 request 的 block table
block_table = torch.tensor([
    42, 7, 103, 5, 88,  # 逻辑页 0-4 -> 物理 block
    -1, -1, -1,          # 未分配 (按需扩)
])
# KV pool: KV_pool[42], KV_pool[7], ... 各 block 散布
# attention kernel 按 block_table 遍历物理 block
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[KV cache]]（KV 概念）、[[KV cache management]]（ch08 管理策略）、[[KV block manager]]（block 分配/回收，本笔记的计算底座）、[[Flash Attention]]（online softmax 同源）、[[GPU内存层级]]（block 在 HBM）、[[autoregressive decoding]]（decode 增量 KV）。
- **下游（应用）**: vLLM 推理引擎（奠基技术）、[[vLLM scheduler]]（调度 block）、[[cache eviction与swap offload]]（block 换出）、[[Radix Tree prefix cache]]（block sharing）、[[continuous batching的调度实现]]（batch 与 block 协同）。
- **对比 / 易混**:
  - **PagedAttention vs [[KV cache management]]**：前者是 vLLM 源码实现（分页 + 非连续 kernel），后者是 ch08 概念/策略（为什么管、管什么）。前者落地后者。
  - **PagedAttention vs [[Flash Attention]]**：前者是 KV cache 分页 + 非连续 attention（管显存碎片），后者是 attention 的 tiling + online softmax（管算力/带宽）。两者都用 online softmax，但目标不同（碎片 vs 算力）。vLLM 的 PagedAttention V2 借鉴 Flash Attention 的 split-K。
  - **PagedAttention vs [[Radix Tree prefix cache]]**：前者是 block 级分页（管物理显存），后者是 SGLang 的前缀树缓存（管逻辑前缀复用）。前者是底座，后者建其上（SGLang 也有 paged KV）。
  - **PagedAttention V1 vs V2**：V1 block 循环 in CUDA，V2 优化 split-K 调度（长序列并行）。V2 默认。


## 7. 常见误区与易错点

> [!warning] 误区 1：PagedAttention 是新 attention 算法
> 不是新算法。attention 的数学不变（$QK^T V$ softmax）。PagedAttention 改的是 **KV cache 的存储布局**（分页非连续）+ kernel 适配（block 循环）。是存储/工程优化，不是算法创新。

> [!warning] 误区 2：block 越小越好
> block 小则碎片少但 kernel 开销大（多 block 循环）。block 大则碎片多但 kernel 高效。vLLM 默认 16（经验最优）。太小会 kernel 慢。

> [!warning] 误区 3：PagedAttention = Flash Attention
> 两者都用 online softmax，但目标不同：PagedAttention 管 KV cache 碎片（分页非连续），Flash Attention 管 attention 算力/带宽（tiling）。vLLM PagedAttention V2 借鉴 Flash Attention 的 split-K，但核心是分页。

> [!warning] 误区 4：无碎片 = 无浪费
> 仍有 block 尾部碎片（$\lceil L/B \rceil$ 的尾部，最多 $B-1$ token 浪费）。但 $B$ 小（16）可忽略。比传统连续的 max 预分配碎片小得多。

> [!warning] 误区 5：prefix sharing 不需 copy-on-write
> 共享前缀的 block 被多请求指向。某请求分叉（不同 token）时需 copy 新 block（copy-on-write），否则覆盖共享 block。block manager 用引用计数管理。

> [!warning] 误区 6：PagedAttention 自动换出
> PagedAttention 本身只管非连续 block 的 attention。block 的分配/回收/换出由 [[KV block manager]] 管。PagedAttention 是计算底座，block manager 是管理底座。两者分工。


## 8. 延伸细节

### 8.1 block_size 的选择

vLLM 默认 block_size=16。选择权衡：
- 小（8）：碎片更少，但 kernel 循环多（开销）。
- 大（32-64）：kernel 高效，但尾部碎片大。
16 是经验最优（kernel 开销与碎片平衡）。

### 8.2 PagedAttention V2 的 split-K

V2 对长序列用 split-K：把 block 循环拆成多段并行（split-K），减 decode 的尾延迟（长 sequence 不再串行循环所有 block）。借鉴 Flash Attention 的 split-K 思路。

### 8.3 与 SGLang 的对照

SGLang 也用 paged KV cache（类似 PagedAttention），但加 RadixAttention（前缀树复用）。详见 [[Radix Tree prefix cache]]。两者底层 block 管理类似，前缀复用策略不同。

### 8.4 内容来源

PagedAttention 整理自 vLLM 论文（"Efficient Memory Management for LLM Serving with PagedAttention", SOSP 2023）、vLLM `vllm/attention/` 源码、社区教程。online softmax 见 [[Flash Attention]] 论文。与 ch08 [[KV cache management]] 的关系见该笔记。

---
相关: [[PagedAttention]] | [[KV cache]] | [[KV cache management]] | [[KV block manager]] | [[cache eviction与swap offload]] | [[Flash Attention]] | [[autoregressive decoding]] | [[Radix Tree prefix cache]] | [[continuous batching的调度实现]] | [[vLLM scheduler]] | [[GPU内存层级]]
