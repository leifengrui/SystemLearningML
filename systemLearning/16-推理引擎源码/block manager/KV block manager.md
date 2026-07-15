# KV block manager

> **所属章节**: [[block manager]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: KV block manager / block allocator / block table / 引用计数 / free list / KV 内存分配器
> **难度**: 中（需懂 [[PagedAttention]] + [[KV cache]] + OS 内存管理）


## 1. 一句话定义

**KV block manager** 是 vLLM 中**管理 KV cache 物理块（block）分配/回收/共享/换出的数据结构**——它维护一个**物理 block pool**（预分配的 `num_blocks` 个 block，每 block 存 `block_size` 个 token 的 K+V）+ 一个 **free list**（空闲 block 链表，分配从 head 取、回收回 tail）+ 每请求的 **block table**（逻辑页号→物理 block 的映射表）+ **引用计数**（每个 block 被几个请求/table 项引用，归零才回 free list，支持 prefix sharing 的 copy-on-write）。当调度器要给请求分配 KV 时，manager 从 free list 取 block、填入该请求的 block table；请求结束或被 [[请求生命周期|preempt]] 换出时，manager 据 block table 回收 block（减引用计数，归零回 free list）。是 [[PagedAttention]]（非连续 attention kernel）的**管理底座**——PagedAttention 算 attention，block manager 管 block 的生命周期。与 [[cache eviction与swap offload]] 配套（后者决定何时换出/驱逐，前者执行 block 操作）。

> [!note] 三句话定位
> - **是什么**：物理 block pool + free list + block table（逻辑→物理映射）+ 引用计数（支持 prefix sharing copy-on-write）。分配/回收 block 的数据结构。
> - **为什么**：[[PagedAttention]] 的非连续 block 需有谁分配/回收/映射，否则 kernel 无 block 可用。block manager 是非连续 KV 的管理底座。
> - **与 [[PagedAttention]] 关系**：PagedAttention 是计算底座（非连续 attention kernel），block manager 是管理底座（block 生命周期）。两者分工配套。


## 2. 为什么需要它（动机与背景）

### 2.1 非连续 block 需管理

[[PagedAttention]] 把 KV 分页成非连续 block。需有谁分配 block（请求来）、回收 block（请求走）、映射 block（逻辑→物理）。无 manager 则 PagedAttention 无 block 可用。

### 2.2 OS 内存管理类比

OS 的分页内存需 page allocator（管页分配/回收）+ page table（映射）+ refcount（共享页）。vLLM 的 block manager 是 KV block 的 allocator + table + refcount。类比 OS 内存管理。

### 2.3 prefix sharing 的引用计数

多请求共享前缀 block（copy-on-write）。需引用计数：block 被几个请求/table 项引用。某请求分叉（copy）或结束时减 refcount，归零才回 free list。无引用计数会误回收共享 block。

### 2.4 调度器的配合

[[vLLM scheduler]] 调度请求时问 block manager：还有多少 free block？这请求需几 block？够则 admission（接收），不够则 preempt（换出）。block manager 是调度的显存预算依据。


## 3. 核心概念详解

### 3.1 物理 block pool

```
KV pool (预分配):
  block 0:  [16 token 的 K+V]
  block 1:  [16 token 的 K+V]
  ...
  block N-1: [16 token 的 K+V]
  共 num_blocks 个 block, block_size=16 token/block
```

预分配固定数 block（据显存预算）。block 内连续（16 token 的 K+V）。block 间非连续。

### 3.2 free list

```
free list (空闲 block 链表):
  [block 5] -> [block 42] -> [block 7] -> [block 103] -> ...
分配: 从 head 取 (pop block 5)
回收: 回 tail (append)
```

空闲 block 链表。分配取 head，回收回 tail。$O(1)$ 分配/回收。

### 3.3 block table

```
请求 A 的 block table:
  logical_page 0 -> physical_block 42
  logical_page 1 -> physical_block 7
  logical_page 2 -> physical_block 103
  ...
请求 B 的 block table:
  logical_page 0 -> physical_block 42  (与 A 共享前缀 block 42)
  logical_page 1 -> physical_block 88
  ...
```

每请求一 block table（逻辑页→物理 block）。是 [[PagedAttention]] kernel 的输入（kernel 据 table 遍历物理 block）。

### 3.4 引用计数

```
block 42 的 refcount:
  初始 0 (free)
  请求 A 分配 logical_page 0 -> block 42: refcount=1
  请求 B 共享前缀 logical_page 0 -> block 42: refcount=2
  请求 A 结束 (或分叉 copy): refcount=1
  请求 B 结束: refcount=0 -> 回 free list
```

每 block 维护 refcount。分配/共享 +1，释放/分叉 -1。归零回 free list。支持 prefix sharing 的 copy-on-write。

### 3.5 分配流程

```python
def allocate(request, num_tokens):
    blocks_needed = ceil(num_tokens / block_size)
    for _ in range(blocks_needed):
        block = free_list.pop()  # 取空闲 block
        block_table[request].append(block)
        refcount[block] += 1
    # block table 更新, PagedAttention kernel 可用
```

请求来：算需几 block，从 free list 取，填 block table，增 refcount。

### 3.6 回收流程

```python
def free(request):
    for block in block_table[request]:
        refcount[block] -= 1
        if refcount[block] == 0:
            free_list.append(block)  # 回 free list
    block_table[request].clear()
```

请求走：遍历 block table，减 refcount，归零回 free list。

### 3.7 copy-on-write（prefix 分叉）

```python
def fork(request):  # 请求分叉 (不同 token 出现)
    for page in range(len(block_table[request])):
        block = block_table[request][page]
        if refcount[block] > 1:  # 共享中
            new_block = free_list.pop()  # copy
            copy_data(new_block, block)
            refcount[block] -= 1  # 原块减
            refcount[new_block] = 1  # 新块
            block_table[request][page] = new_block
        # refcount=1 (独占) 则不 copy (已是自己的)
```

共享 block 分叉时 copy 新 block（copy-on-write）。避免覆盖共享 block。是 prefix sharing 的底层。

### 3.8 显存预算查询

```python
def can_allocate(request, num_tokens):
    blocks_needed = ceil(num_tokens / block_size)
    return len(free_list) >= blocks_needed  # 够不够

def get_num_free_blocks():
    return len(free_list)  # 调度器据此 admission/preempt
```

调度器问 free list 够不够。不够则 preempt 换出。详见 [[vLLM scheduler]] 与 [[cache eviction与swap offload]]。


## 4. 数学原理 / 公式

### 4.1 block 数预算

$$\text{num\_blocks} = \left\lfloor \frac{M_{\text{KV pool}}}{\text{block\_size} \cdot 2 \cdot n_{\text{kv}} \cdot d_h \cdot b} \right\rfloor$$

$M_{\text{KV pool}}$=预留给 KV 的显存，$2$=K+V，$n_{\text{kv}}$=KV head，$d_h$=head dim，$b$=byte。block 数 = 显存/每 block。

### 4.2 每 request 的 block 数

$$\text{blocks}(L) = \left\lceil \frac{L}{\text{block\_size}} \right\rceil$$

$L$=token 数。尾部碎片最多 $\text{block\_size} - 1$。

### 4.3 并发上限

$$B_{\max} = \left\lfloor \frac{\text{num\_blocks}}{\max_L \text{blocks}(L)} \right\rfloor \quad (\text{保守, 无 sharing})$$

实际因 prefix sharing（refcount > 1）可超 $B_{\max}$。是并发的显存约束。

### 4.4 引用计数的 amortized 开销

copy-on-write 的 copy 成本：仅分叉时 copy（refcount > 1）。大部分 block refcount=1（独占，不 copy）。amortized 开销低。


## 5. 代码示例（可选）

### 5.1 vLLM block manager

```python
from vllm.core.block_manager import BlockManager
manager = BlockManager(block_size=16, num_blocks=N)
# 分配
manager.allocate(request_id, num_tokens=100)  # 7 blocks
# 查询
manager.can_allocate(request_id, num_tokens=200)  # 够不够
# 回收
manager.free(request_id)  # block 回 free list
```

### 5.2 block table 传递

```python
# block_table 传给 PagedAttention kernel
block_table = manager.get_block_table(request_id)  # [num_pages]
# kernel 据此遍历物理 block 算 attention
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[PagedAttention]]（非连续 attention kernel，本笔记管其 block）、[[KV cache]]（KV 概念）、[[KV cache management]]（ch08 管理策略）、[[GPU内存层级]]（block 在 HBM）、[[autoregressive decoding]]（decode 增量需 block）。
- **下游（应用）**: [[vLLM scheduler]]（调度问 block 预算）、[[cache eviction与swap offload]]（block 换出执行）、[[Radix Tree prefix cache]]（prefix sharing 用引用计数）、[[continuous batching的调度实现]]（batch 与 block 协同）、[[请求生命周期]]（preempt 回收 block）。
- **对比 / 易混**:
  - **block manager vs [[PagedAttention]]**：前者管 block 生命周期（分配/回收/映射/引用计数），后者算非连续 block 的 attention。前者管理底座，后者计算底座。配套。
  - **block manager vs OS page allocator**：概念同（pool + free list + table + refcount），但 block manager 管 KV cache 的 block（GPU HBM），OS allocator 管进程内存页（DRAM）。
  - **block manager vs [[cache eviction与swap offload]]**：前者管 block 的分配/回收（数据结构），后者决定何时换出/驱逐（策略）。前者执行后者决策。


## 7. 常见误区与易错点

> [!warning] 误区 1：block manager 算 attention
> 不算。[[PagedAttention]] kernel 算 attention。block manager 只管 block 的分配/回收/映射/引用计数。两者分工。

> [!warning] 误区 2：block 释放即回 free list
> 共享 block（refcount > 1）不回。减 refcount，归零才回。误以为释放即回会误回收共享 block（其他请求的 KV 丢）。

> [!warning] 误区 3：prefix sharing 不需 copy-on-write
> 共享 block 分叉时（不同 token）需 copy 新 block，否则覆盖共享 block 破坏其他请求 KV。copy-on-write 必需。

> [!warning] 误区 4：block table 是物理连续
> block table 记录逻辑页→物理 block 映射。物理 block 非连续（散布 pool）。kernel 据 table 跳转。

> [!warning] 误区 5：free list 够就一定 admission
> 调度器还需考虑 decode 的 block 增长（每 step 可能需新 block）。预分配时留余量。详见 [[vLLM scheduler]]。


## 8. 延伸细节

### 8.1 block pool 的预分配

block pool 在引擎启动时预分配（固定 num_blocks，据显存预算）。运行时不扩（除非 GPU memory profiling 重新评估）。是静态预算。

### 8.2 与 SGLang 的对照

SGLang 也有 block manager（类似 pool + free list + table + refcount），但加 RadixAttention 的前缀树复用（逻辑层）。详见 [[Radix Tree prefix cache]]。底层 block 管理类似。

### 8.3 内容来源

KV block manager 整理自 vLLM `vllm/core/block_manager.py`（V1）与 `vllm/core/block/block_manager.py`（V2）源码、vLLM 论文（SOSP 2023）。OS 内存管理类比见 OS 教材。与 PagedAttention 的分工见 [[PagedAttention]]。

---
相关: [[block manager]] | [[PagedAttention]] | [[KV cache]] | [[KV cache management]] | [[cache eviction与swap offload]] | [[vLLM scheduler]] | [[Radix Tree prefix cache]] | [[continuous batching的调度实现]] | [[请求生命周期]] | [[GPU内存层级]] | [[autoregressive decoding]]
