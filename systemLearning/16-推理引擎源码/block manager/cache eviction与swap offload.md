# cache eviction与swap offload

> **所属章节**: [[block manager]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: cache eviction与swap offload / KV 驱逐与换出 / LRU KV / LRU-2 / CPU swap / KV offload / KV 溢出
> **难度**: 中（需懂 [[KV block manager]] + [[vLLM scheduler]] + OS swap + [[GPU内存层级]]）


## 1. 一句话定义

**cache eviction 与 swap offload** 是 vLLM 在 **GPU KV block 不够时**的两种应对策略——**(1) eviction（驱逐/重算）**：当 free block 耗尽，调度器选低优先级（如 waiting 队列、先来的）请求的 KV block **驱逐回收**（直接 free 其 block table，KV 丢弃），被驱逐请求回 waiting 队列，下次重新调度时**从头重算 prefill**（KV 重建）；**(2) swap offload（换出到 CPU）**：不丢 KV，把请求的 KV block **复制到 CPU 内存**（`swap_out`，block table 标记为 swapped），GPU block 回 free list 给高优先级请求；请求重新调度时 `swap_in` 把 KV 从 CPU 复制回 GPU（避重算 prefill）。驱逐用 **LRU/LRU-2**（优先驱逐最近最少用/最久未活跃的请求 KV）选 victim。两种策略权衡：**eviction 省显存但重算 prefill 耗算力**（适合 prefill 短/算力富余），**swap 保 KV 但 CPU 内存+拷贝带宽开销**（适合 prefill 长/算力紧）。是 [[KV block manager]] 的**满载应对**——block manager 分配/回收 block，eviction/swap 决定满载时丢谁/换谁，是 [[vLLM scheduler]] 的 preemption 机制底层。

> [!note] 三句话定位
> - **是什么**：GPU KV block 满时，eviction（丢 KV 重算 prefill）或 swap offload（KV 换 CPU 再换回）两种 preemption 策略，用 LRU/LRU-2 选 victim 请求。
> - **为什么**：高并发下 KV block 不够，需让位给高优先级（decode 中）请求，否则 OOM 或停服务；eviction/swap 是显存压力下的有序退让。
> - **与 [[KV block manager]] 关系**：block manager 管 block 分配/回收（数据结构），eviction/swap 是满载时的策略（丢/换）。block manager 执行 eviction/swap 的 block 操作。


## 2. 为什么需要它（动机与背景）

### 2.1 KV block 不够

高并发请求时 KV block 耗尽（free list 空）。新请求或 decode 增长需 block。若无应对则 OOM 或停服务。需有序退让（让位给高优先级）。

### 2.2 让位给 decode 中请求

decode 中的请求（已生成 token，用户等）优先级高于 waiting 请求（未开始）。KV 不够时，让 waiting 请求的 block 给 decode 请求。是 preemption（抢占）。

### 2.3 eviction vs swap 的权衡

- **eviction（重算）**：丢 KV，回 waiting，下次重算 prefill。省显存，耗算力。适合 prefill 短。
- **swap offload（换出）**：KV 换 CPU，下次换回。保 KV，耗内存+带宽。适合 prefill 长。

据算力/显存/延迟权衡选。

### 2.4 LRU 选 victim

满载时选谁让位？LRU（最近最少用）或 LRU-2（综合 recency + frequency）选低优先级 victim。避免驱逐刚活跃的请求（会损用户体验）。


## 3. 核心概念详解

### 3.1 eviction（驱逐重算）

```
KV block 满:
  选 victim 请求 (LRU, 如 waiting 队列尾)
  free victim 的 block table (KV 丢弃, block 回 free list)
  victim 回 waiting 队列 (状态保存: 已生成 token)
  高优先级请求拿 block
下次 victim 重新调度:
  从头重算 prefill (KV 重建)
  继续 decode (从上次断点)
```

eviction 丢 KV 重算。省显存（block 回 free），耗算力（重算 prefill）。

### 3.2 swap offload（换出/换入）

```
KV block 满:
  选 victim 请求
  swap_out: victim 的 KV block 复制到 CPU 内存 (block table 标 swapped)
  GPU block 回 free list
  victim 回 waiting (状态: swapped)
  高优先级请求拿 block
下次 victim 重新调度:
  swap_in: CPU 的 KV 复制回 GPU (新 block)
  继续 decode (KV 已恢复, 不重算)
```

swap 保 KV（换 CPU），耗 CPU 内存 + 拷贝带宽。避重算。

### 3.3 LRU / LRU-2

```
LRU: 选最近最少活跃的请求 (last_access_time 最小) 为 victim
LRU-2: 综合 recency (最近活跃) + frequency (活跃次数)
  更智能 (避免 LRU 误驱逐将活跃的)
```

LRU 简单。LRU-2 更智能（综合频率+recency）。vLLM 默认 LRU。

### 3.4 preemption（抢占）

```
调度 iteration:
  free block 不够新 decode/新请求
  preempt: 选 victim (LRU), eviction 或 swap
  victim 让位 (丢/换 KV)
  高优先级拿 block 继续
```

preemption 是 [[vLLM scheduler]] 的满载应对。eviction/swap 是 preemption 的两种策略。详见 [[vLLM scheduler]] 与 [[请求生命周期]]。

### 3.5 eviction vs swap 的选择

| 维度 | eviction（重算） | swap offload（换出） |
|---|---|---|
| KV | 丢 | 保（换 CPU） |
| 算力 | 耗（重算 prefill） | 省（不重算） |
| 显存 | 省（block 回 free） | 省（block 回 free） |
| CPU 内存 | 不耗 | 耗（存 KV） |
| 带宽 | 不耗 | 耗（拷贝） |
| 延迟 | 重算 prefill 时间 | 拷贝时间 |
| 适合 | prefill 短/算力富余 | prefill 长/算力紧 |

vLLM 默认可配（`--swap-space` 设 CPU swap 容量）。swap=0 则纯 eviction。

### 3.6 swap 的 CPU 内存

```
swap_space: 预留 CPU 内存 (如 16GB)
swapped block 存 CPU (与 GPU block 同结构)
swap_in/swap_out: NCCL/cudaMemcpy 拷贝
```

CPU swap 容量（`--swap-space`）。swapped block 存 CPU。拷贝用 cudaMemcpy（GPU↔CPU PCIe）。是 [[GPU内存层级]] 的 CPU offload。

### 3.7 与 block manager 的配合

```python
# block manager 执行 eviction/swap 的 block 操作
def preempt(manager, victim, strategy):
    if strategy == 'eviction':
        manager.free(victim)  # 丢 KV, block 回 free list
    elif strategy == 'swap':
        cpu_blocks = swap_out(manager.get_blocks(victim))  # 复制 CPU
        manager.free(victim)  # GPU block 回 free list
        victim.state = 'swapped'  # 标记
```

block manager 执行 block 操作（free/swap）。eviction/swap 是策略（决定丢/换）。两者配合。

### 3.8 recompute（重算）的细化

eviction 的重算不是从头重算全部——只重算 prefill 部分（prompt），decode 已生成的 token 不需重算（它们是 input，重算 prefill 后继续 decode）。但若 prompt 很长，重算 prefill 仍耗。详见 [[请求生命周期]]。


## 4. 数学原理 / 公式

### 4.1 eviction 的重算开销

$$T_{\text{evict}} = T_{\text{prefill}}(L_{\text{prompt}}) \quad (\text{重算 prefill})$$

$L_{\text{prompt}}$=prompt 长度。prefill 长则重算贵。prefill 短则便宜。

### 4.2 swap 的拷贝开销

$$T_{\text{swap}} = \frac{M_{\text{KV}}}{\beta_{\text{PCIe}}} \quad (\text{GPU↔CPU 拷贝})$$

$M_{\text{KV}}$=victim 的 KV 显存，$\beta_{\text{PCIe}}$=PCIe 带宽（~32 GB/s）。拷贝时间 = 显存/带宽。

### 4.3 eviction vs swap 的选择

重算 vs 拷贝：

$$\text{选 swap 若 } T_{\text{swap}} < T_{\text{evict}}, \quad \text{即 } \frac{M_{\text{KV}}}{\beta_{\text{PCIe}}} < T_{\text{prefill}}(L_{\text{prompt}})$$

prefill 长（$T_{\text{prefill}}$ 大）则 swap 划算。prefill 短则 eviction 划算。

### 4.4 CPU 内存预算

$$M_{\text{swap}} = \text{swap\_space} \quad (\text{预留 CPU 内存})$$

swapped block 总量受 swap_space 限。超则只能 eviction（CPU 也满）。

### 4.5 LRU 的 victim 选择

$$\text{victim} = \arg\min_{r} \text{last\_access\_time}(r) \quad (\text{LRU})$$

选最近最少活跃的。LRU-2 综合 frequency：

$$\text{score}(r) = \alpha \cdot \text{recency}(r) + \beta \cdot \text{frequency}(r)$$

选 score 最低。更智能。


## 5. 代码示例（可选）

### 5.1 vLLM swap 配置

```bash
# vLLM 启动配 swap space (CPU offload 容量)
python -m vllm.entrypoints.openai.api_server \
    --swap-space 16  # 16 GB CPU swap
# swap-space=0 则纯 eviction (不换出)
```

### 5.2 preemption 流程（概念）

```python
# scheduler 满 KV 时 preempt
if not manager.can_allocate(req, num_tokens):
    victim = lru.select_victim()  # LRU 选
    if swap_space > 0 and cpu_has_room(victim):
        swap_out(victim)  # 换 CPU
    else:
        manager.free(victim)  # eviction 丢
        victim.state = 'recompute'
    # victim 回 waiting, 高优先级拿 block
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[KV block manager]]（执行 block 操作）、[[vLLM scheduler]]（满载 preempt 决策）、[[请求生命周期]]（preempt 状态机）、[[KV cache]]（KV 概念）、[[GPU内存层级]]（CPU offload）、[[KV cache management]]（ch08 策略层）。
- **下游（应用）**: vLLM 高并发 serving、显存压力下的有序退让、prefill 长 vs 短的 preemption 策略选择。
- **对比 / 易混**:
  - **eviction vs swap**：前者丢 KV 重算（省显存耗算力），后者换 CPU 保 KV（耗内存带宽省算力）。据 prefill 长短选。
  - **eviction vs [[gradient checkpointing]]**：前者是推理 KV 的丢弃重算，后者是训练 activation 的重算。概念同（丢中间态重算），但场景不同（推理 vs 训练，KV vs activation）。
  - **swap vs OS swap**：概念同（内存满换出到次级存储），但 vLLM swap 是 KV block 换 CPU（GPU↔CPU PCIe），OS swap 是页换磁盘（DRAM↔disk）。
  - **LRU vs LRU-2**：LRU 只看 recency（最近最少用），LRU-2 综合 recency + frequency（更智能，避免误驱逐将活跃的）。


## 7. 常见误区与易错点

> [!warning] 误区 1：eviction 丢全部 token
> eviction 丢的是 KV cache（中间态），不是已生成的 token（输出）。已生成 token 是请求的输入（下次重算 prefill 后继续 decode 从断点）。重算的是 prefill 的 KV，不是生成结果。

> [!warning] 误区 2：swap 不耗资源
> swap 耗 CPU 内存（存 KV）+ PCIe 带宽（拷贝）。不是免费。prefill 短时 swap 的拷贝可能比重算还贵。

> [!warning] 误区 3：eviction 一定差
> eviction 省 CPU 内存（不耗），prefill 短时重算便宜。算力富余时 eviction 比 swap 划算。据场景选。

> [!warning] 误区 4：LRU 最优
> LRU 简单但可能误驱逐将活跃的（如刚 prefill 完待 decode 的）。LRU-2 综合频率更智能。但 vLLM 默认 LRU（足够，因请求模式可预测）。

> [!warning] 误区 5：swap 避免一切重算
> swap 避重算 prefill，但若 swapped 请求在 CPU 也满（swap_space 不够），仍需 eviction。swap 不是无限保 KV。

> [!warning] 误区 6：preemption 是错误
> preemption 是有序退让（让位高优先级），不是错误。高并发下 KV 不够是常态，preemption 保服务连续。是设计的容错。


## 8. 延伸细节

### 8.1 preempt 的频率

vLLM 的 preempt 频率据显存预算与并发。预算足（num_blocks 大）则少 preempt。预算紧则频繁。是 serving 配置的 trade-off（显存 vs 算力）。

### 8.2 与 SGLang 的对照

SGLang 也有类似的 block 管理与 preemption，但加 RadixAttention 的前缀复用（减少重算，因前缀 KV 可从树恢复）。详见 [[Radix Tree prefix cache]]。

### 8.3 recompute 的优化

vLLM 的 recompute 优化：只重算 prefill（不重算 decode 的已生成 token 的 KV，因它们是 input 的 KV）。但若 prompt 极长，重算仍贵。chunked prefill 可减轻（分块重算）。详见 [[chunked prefill]]。

### 8.4 内容来源

cache eviction 与 swap offload 整理自 vLLM `vllm/core/scheduler.py`（preempt 逻辑）与 `vllm/core/block_manager.py`（swap 操作）源码、vLLM 论文（SOSP 2023）。LRU/LRU-2 见 OS/缓存教材。与 block manager 的分工见 [[KV block manager]]。

---
相关: [[block manager]] | [[KV block manager]] | [[vLLM scheduler]] | [[请求生命周期]] | [[KV cache]] | [[KV cache management]] | [[GPU内存层级]] | [[chunked prefill]] | [[Radix Tree prefix cache]] | [[gradient checkpointing]]
