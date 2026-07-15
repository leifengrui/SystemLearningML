# Radix Tree prefix cache

> **所属章节**: [[Prefix Cache]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: Radix Tree prefix cache / RadixAttention / SGLang 前缀树缓存 / 前缀复用树 / 自动 prefix caching
> **难度**: 中高（需懂 [[PagedAttention]] + [[KV block manager]] + [[prefix caching]] + [[KV cache]] + 数据结构 Radix Tree）


## 1. 一句话定义

**Radix Tree prefix cache**（RadixAttention）是 SGLang 的**自动前缀 KV 复用**机制——用一个 **Radix Tree（基数树/压缩前缀树）** 做"**token 序列前缀 → KV block**"的索引：树的每条边是一段 token 子串，每节点存该前缀对应的 KV block（多请求共享同前缀则共享 block，引用计数管理），新请求来时**沿树匹配最长公共前缀**（如多请求都以 "System: You are a helpful assistant" 开头），命中则**直接复用该前缀已算的 KV**（不重算 prefill，省算力+降 TTFT），分叉处 copy-on-write（新 token 出现则 copy 新 block）；树带 **LRU 驱逐**（显存满时淘汰最近最少用的前缀节点，回收 block）。相比 vLLM 的**显式 prefix caching**（block-hash 匹配，需 block 对齐有粒度损失），RadixAttention 是**任意长度前缀自动复用**（树匹配，无对齐损失，复用率更高）。是 SGLang 的护城河技术（vs vLLM），prefill 快 3-5×、吞吐高 29%。与 [[prefix caching]]（ch08 概念层）互为表里：ch08 讲为什么复用前缀、复用什么，本笔记讲 SGLang 怎么用 Radix Tree 自动高效复用。

> [!note] 三句话定位
> - **是什么**：Radix Tree 索引 token 前缀→KV block，新请求沿树匹配最长公共前缀复用 KV（不重算），分叉 copy-on-write，LRU 驱逐。SGLang 默认开零配置。
> - **为什么**：多请求常共享前缀（system prompt/少样本/RAG context），重算 prefill 浪费算力；RadixAttention 自动复用，省算力+降 TTFT。vLLM block-hash 有粒度损失，RadixTree 无。
> - **与 [[prefix caching]] 关系**：ch08 是概念（为什么复用前缀），本笔记是 SGLang 源码实现（Radix Tree 自动复用）。是 ch08 的高效落地。


## 2. 为什么需要它（动机与背景）

### 2.1 前缀复用的普遍性

LLM serving 多请求常共享前缀：
- system prompt（"You are a helpful assistant..."）每请求都有
- few-shot examples（相同的几个示例）
- RAG context（相同的检索文档）
重算这些前缀的 prefill 浪费算力（相同输入算相同 KV）。

### 2.2 复用前缀 KV

前缀的 KV 已算过（在 cache）。新请求若前缀匹配，直接复用（不重算 prefill）。省算力 + 降 TTFT。是 prefix caching 的核心。

### 2.3 vLLM 的 block-hash 限制

vLLM 的 prefix caching 用 block-hash（每 block_size token 算 hash 匹配）。需 block 对齐（前缀必须是 block_size 倍数才有命中）。有粒度损失（前缀非 block 倍数则不命中）。

### 2.4 RadixTree 无对齐损失

RadixTree 沿树匹配任意长度前缀（边是任意子串，不要求 block 倍数）。无对齐损失，复用率更高。是 SGLang 的优势。

### 2.5 自动零配置

SGLang 的 RadixAttention 默认开（零配置）。vLLM 需 `--enable-prefix-caching` 显式开。是易用性优势。


## 3. 核心概念详解

### 3.1 Radix Tree 结构

```
Radix Tree (压缩前缀树):
  root
  ├── "System: You are" -> [KV block 42, refcount=3]  (3 请求共享)
  │   ├── "a helpful assistant" -> [block 7, refcount=2]
  │   │   ├── "Question A" -> [block 103, refcount=1]  (req A 独有)
  │   │   └── "Question B" -> [block 88, refcount=1]   (req B 独有)
  │   └── "an evil assistant" -> [block 9, refcount=1] (req C 分叉)
  ...
每边 = token 子串, 每节点 = 该前缀的 KV block
```

每边是 token 子串（任意长度，非 block 倍数）。每节点存该前缀的 KV block。共享前缀的请求共享节点（引用计数）。分叉处 copy-on-write。

### 3.2 最长公共前缀匹配

```
新请求 "System: You are a helpful assistant. Question C":
  沿树匹配:
    root -> "System: You are" (命中, block 42)
    -> "a helpful assistant" (命中, block 7)
    -> "Question C" (不命中, 新节点)
  复用 block 42 + 7 (前缀 KV 已算, 不重算)
  只算 "Question C" 的新 KV (新 block)
```

沿树匹配最长公共前缀。命中复用（不重算 prefill）。只算新部分。

### 3.3 KV block 复用

```
命中前缀的 KV block:
  直接指向 (block table 引用 block 42, 7)
  不重算 prefill (省算力)
  refcount 增 (共享)
```

复用 = block table 指向已有 block。不重算。省算力 + 降 TTFT。

### 3.4 copy-on-write（分叉）

```
共享 "System: You are a helpful assistant" (block 42, 7, refcount=2)
req A: "...Question A" (分叉, block 103)
req B: "...Question B" (分叉, block 88)
分叉处 (Question 出现):
  共享前缀不变 (block 42, 7 继续共享)
  分叉后的新 token 各自新 block (103, 88)
  若某请求修改共享前缀 (不常见, 如 in-place 编辑):
    copy-on-write (copy 新 block, 原块 refcount-1)
```

分叉后的新 token 各自新 block。共享前缀不变。copy-on-write 仅在修改共享前缀时（罕见）。详见 [[KV block manager]]。

### 3.5 LRU 驱逐

```
显存满:
  选最近最少用的前缀节点 (LRU)
  回收其 block (refcount 归零才回 free list)
  树删该节点
```

LRU 驱逐最近最少用的前缀。block 回 free list（refcount 归零）。是显存管理。详见 [[cache eviction与swap offload]]。

### 3.6 vs vLLM block-hash

| 维度 | vLLM block-hash | SGLang RadixTree |
|---|---|---|
| 索引 | block_size token 算 hash | RadixTree 子串 |
| 对齐 | 需 block 倍数（粒度损失） | 任意长度（无损失） |
| 复用率 | 中（对齐损失） | 高（无损失） |
| 配置 | `--enable-prefix-caching` | 默认开（零配置） |
| prefill 速度 | 基线 | 快 3-5× |
| 吞吐 | 基线 | 高 29% |

SGLang RadixTree 复用率更高（无对齐损失），更快。

### 3.7 与 PagedAttention 的关系

RadixAttention 建在 [[PagedAttention]] 之上（底层仍用 paged KV block）。RadixTree 是逻辑层（前缀索引），PagedAttention 是物理层（非连续 block）。两者叠加：物理非连续 + 逻辑前缀复用。

### 3.8 树的更新

```
新请求来:
  匹配最长前缀 (复用)
  新部分加新节点 (边=新子串, 节点=新 block)
请求走:
  refcount-1, 归零则节点删 (LRU 可能更早删)
显存满:
  LRU 选 victim 节点, 回收 block, 删节点
```

树动态更新（匹配/加节点/删节点）。是活的数据结构。


## 4. 数学原理 / 公式

### 4.1 前缀复用的省算力

设前缀长 $L_p$，新部分长 $L_n$。传统 prefill 算力 $2(L_p + L_n)d^2$。复用后只算新部分 $2 L_n d^2$。省：

$$\text{save} = 2 L_p d^2 \quad (\text{前缀部分不重算})$$

前缀长（system prompt/RAG 长）则省多。

### 4.2 TTFT 的降

传统 $TTFT = T_{\text{prefill}}(L_p + L_n)$。复用 $TTFT = T_{\text{prefill}}(L_n)$（只算新）。$L_p \gg L_n$ 则 TTFT 大降（SGLang 报告 3-5×）。

### 4.3 RadixTree 匹配复杂度

$$T_{\text{match}} = O(L) \quad (L = \text{请求 token 数, 沿树遍历})$$

线性匹配（沿树走）。高效。

### 4.4 LRU 的驱逐

$$\text{victim} = \arg\min_{\text{node}} \text{last\_access\_time}(\text{node})$$

选最近最少用的节点。block 回 free list（refcount 归零）。


## 5. 代码示例（可选）

### 5.1 SGLang RadixAttention（默认开）

```python
from sglang import Engine
engine = Engine(model="...")  # RadixAttention 默认开, 零配置
# 多请求共享前缀自动复用
out1 = engine.generate("System: You are... Question A")
out2 = engine.generate("System: You are... Question B")  # 前缀复用, 不重算
```

### 5.2 RadixTree 概念

```python
class RadixNode:
    tokens: list         # 该边的 token 子串
    kv_blocks: list      # 该前缀的 KV block
    refcount: int        # 引用数
    children: dict       # 子节点 (按下一 token)

def match_and_insert(root, tokens):
    # 沿树匹配最长前缀
    node, matched = longest_prefix_match(root, tokens)
    # 复用 matched 的 KV block
    # 新部分 (tokens[matched:]) 加新节点
    ...
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[prefix caching]]（ch08 概念）、[[PagedAttention]]（物理层非连续 block）、[[KV block manager]]（block 管理 + 引用计数）、[[KV cache]]、[[cache eviction与swap offload]]（LRU 驱逐）、[[chunked prefill]]（前缀复用减 prefill 量）。
- **下游（应用）**: SGLang serving 引擎、多请求共享前缀场景（system prompt/few-shot/RAG）、RLHF rollout（前缀复用提生成吞吐）、API 服务（前缀缓存降成本）。
- **对比 / 易混**:
  - **RadixTree vs vLLM block-hash**：前者 RadixTree 子串匹配（任意长度，无损失，SGLang），后者 block_size hash 匹配（需对齐，有粒度损失，vLLM）。前者复用率更高。
  - **RadixTree vs [[PagedAttention]]**：前者是逻辑层（前缀索引复用），后者是物理层（非连续 block）。两者叠加（SGLang 都用）。
  - **RadixTree vs [[prefix caching]]**：ch08 是概念（为什么复用前缀），本笔记是 SGLang 源码实现（RadixTree 自动复用）。是 ch08 落地。
  - **RadixTree vs [[continuous batching的调度实现]]**：前者是前缀复用（避重算），后者是 batch 动态进出（提利用率）。正交（可都开，SGLang 都开）。


## 7. 常见误区与易错点

> [!warning] 误区 1：RadixAttention 是新 attention 算法
> 不是。attention 的数学不变。RadixAttention 改的是 **KV cache 的复用索引**（RadixTree 前缀匹配）。是存储/工程优化，不是算法创新。

> [!warning] 误区 2：需 block 对齐
> 不需。RadixTree 沿子串匹配（任意长度）。是 vs vLLM block-hash（需 block 倍数）的优势。无对齐损失。

> [!warning] 误区 3：前缀复用免一切 prefill
> 只免**共享前缀**的 prefill。新部分（请求独有）仍需算。前缀长则省多，前缀短则省少。

> [!warning] 误区 4：共享 block 可修改
> 共享前缀 block 被多请求指向。修改（如 in-place 编辑）需 copy-on-write（copy 新 block），否则破坏其他请求 KV。引用计数管理。

> [!warning] 误区 5：RadixTree 不驱逐
> 显存满时 LRU 驱逐最近最少用的前缀节点。是显存管理。不驱逐会 OOM。

> [!warning] 误区 6：vLLM 不能 prefix caching
> vLLM 能（`--enable-prefix-caching`，block-hash）。只是 RadixTree 更高效（无对齐损失）。两者都支持 prefix caching，机制不同。

> [!warning] 误区 7：weight sync 后 prefix cache 仍有效
> RLHF 中权重更新后，旧前缀 KV 失效（不同权重算不同 KV）。需失效 prefix cache（否则 staleness + TIM）。详见 [[weight sync mechanism]] 与 [[训推不一致]]。


## 8. 延伸细节

### 8.1 SGLang 的优势来源

SGLang 的 prefill 快 3-5×、吞吐高 29% 主要来自 RadixAttention（无对齐损失 + 自动零配置）。是 SGLang 的护城河。RadixArk 已分拆融 $400M。

### 8.2 与 vLLM 的融合趋势

vLLM 也在改进 prefix caching（更细粒度 hash、自动开）。两者在趋同。但 RadixTree 的子串匹配仍更灵活。

### 8.3 RLHF 中的 staleness

RLHF 训练中权重更新，旧 prefix cache 的 KV 失效（新权重算不同 KV）。需失效 cache（weight sync 后）。否则 staleness + 训推不一致（TIM）。详见 [[weight sync mechanism]] 与 [[训推不一致]]。

### 8.4 内容来源

Radix Tree prefix cache 整理自 SGLang `sglang/srt/mem_cache/radix_cache.py` 源码、SGLang 论文（"SGLang: Efficient Execution of LLM Programs"）、社区对比（turion.ai/inference.net 2026）。与 ch08 [[prefix caching]] 的关系见该笔记。与 PagedAttention 的分层见 [[PagedAttention]]。

---
相关: [[Prefix Cache]] | [[prefix caching]] | [[PagedAttention]] | [[KV block manager]] | [[cache eviction与swap offload]] | [[KV cache]] | [[chunked prefill]] | [[continuous batching的调度实现]] | [[vLLM scheduler]] | [[weight sync mechanism]] | [[训推不一致]]
