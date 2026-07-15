# vLLM scheduler

> **所属章节**: [[请求生命周期与scheduler]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: vLLM scheduler / iteration 调度 / running 与 waiting 队列 / 调度器 / vLLM 调度循环
> **难度**: 中高（需懂 [[请求生命周期]] + [[KV block manager]] + [[continuous batching]] + [[chunked prefill]] + [[cache eviction与swap offload]]）


## 1. 一句话定义

**vLLM scheduler** 是 vLLM 推理引擎的**调度核心**——它跑一个**iteration 级别的调度循环**：每 iteration（一次 forward）做三件事——**(1) 检查 running 队列**（正在 decode 的请求）的 KV 增长需求，若 [[KV block manager]] 的 free block 不够则 **preempt**（按 LRU 选 victim，recompute 丢 KV 或 swap 换 CPU，让位 decode 中请求）；**(2) 从 waiting 队列 admission 新请求**（FIFO，若 free block 够则分配 block、prefill 进 running）；**(3) 组 batch**（running 的 decode + 新 admission 的 prefill 拼成一批，[[continuous batching]] 动态进出，可能 [[chunked prefill]] 切长 prefill 与 decode 同 batch）。scheduler 维护 **running 队列**（decode 中，高优先级）与 **waiting 队列**（新/preempted，等 admission），每 iteration 重新评估 KV 预算、决定 preempt/admission/batch 组成，驱动 [[请求生命周期]] 的状态流转。是 vLLM 高吞吐的核心——dynamic 的 continuous batching + 按需 KV 分配 + 智能 preempt 让 GPU 显存与算力高利用。与 [[continuous batching的调度实现]] 配套（后者是 batching 细节，前者是调度循环）。

> [!note] 三句话定位
> - **是什么**：每 iteration 三步（检查 running/preempt、admission waiting、组 batch continuous），维护 running/waiting 队列，驱动请求状态流转。
> - **为什么**：LLM serving 需动态调度（请求进出、KV 增长、显存预算变），scheduler 每 iteration 重新决策，保高吞吐 + SLO。
> - **与 [[请求生命周期]] 关系**：scheduler 是驱动者（每 iteration 决定状态流转），请求生命周期是被驱动的状态机。前者主动，后者被动。


## 2. 为什么需要它（动机与背景）

### 2.1 LLM serving 的动态性

LLM serving 请求动态来/走、长度不一、KV 增长。需每 iteration 重新决策（谁进 running、谁 preempt、batch 怎么组）。静态调度（固定 batch）低效。需 dynamic scheduler。

### 2.2 KV 预算的约束

GPU 显存有限，KV block 数固定。每 iteration decode 增长需新 block。预算不够则 preempt。scheduler 需评估预算、决策 preempt。是显存约束下的调度。

### 2.3 continuous batching 的需求

[[continuous batching]] 需动态进出（完成出、新进）。scheduler 每 iteration 组 batch（running + 新 admission）。是 continuous batching 的调度实现。

### 2.4 SLO 保障

请求有延迟 SLO（TTFT/TPOT）。scheduler 优先 decode 中（用户等），preempt waiting（未开始）。保 SLO。


## 3. 核心概念详解

### 3.1 iteration 调度循环

```python
# vLLM scheduler 每 iteration (一次 forward)
def schedule():
    # 1. 检查 running, KV 不够则 preempt
    while not manager.can_allocate(running_growth):
        victim = lru.select_victim(running)
        preempt(victim)  # recompute 或 swap
    # 2. admission waiting (FIFO, 够 block 则进)
    for req in waiting_fifo:
        if manager.can_allocate(req.prefill_blocks):
            admission(req)  # 分 block, 进 running
        else:
            break  # 不够, 留 waiting
    # 3. 组 batch (running decode + 新 prefill, continuous batching)
    batch = running_decode + newly_admitted_prefill
    # (可能 chunked prefill 切长 prefill)
    return batch  -> forward
```

每 iteration 三步。驱动状态流转。

### 3.2 running 队列

```
running 队列 (高优先级, decode 中):
  [req_1 (decode step 50), req_2 (decode step 30), ...]
每 iteration: 一起 decode (batch forward)
KV 增长: 每步可能需新 block (block_size=16, 满 16 则新 block)
```

running 是 decode 中的请求。高优先级（用户等）。每 iteration 一起 decode。

### 3.3 waiting 队列

```
waiting 队列 (FIFO, 低优先级):
  [req_5 (新, 未 prefill), req_3 (preempted recompute), req_7 (preempted swapped)]
admission: 从头取, 够 block 则进 running (prefill)
```

waiting 是新/preempted 的请求。低优先级。FIFO admission。

### 3.4 preemption

```
KV 不够 running 增长:
  选 victim (LRU, 优先 preempt 刚 prefill 完待 decode 的)
  recompute: 丢 KV, 回 waiting (标 recompute)
  或 swap: 换 CPU, 回 waiting (标 swapped)
  高优先级 (decode 中) 拿 block
```

preempt 让位。详见 [[cache eviction与swap offload]] 与 [[请求生命周期]]。

### 3.5 admission

```
waiting FIFO admission:
  req = waiting.pop()
  if manager.can_allocate(req.prefill_blocks):
    manager.allocate(req)  # 分 block
    req.state = RUNNING  # 进 running
    # 下次 forward prefill
  else:
    break  # 不够, 留 waiting (下 iteration 再试)
```

admission 从 waiting 取，够 block 则进 running。

### 3.6 batch 组成

```
batch = running.decode (各 1 token) + newly_admitted.prefill (各 prompt)
  -> 一次 forward (continuous batching, 混 decode + prefill)
  (chunked prefill: 切长 prefill 与 decode 同 batch)
```

batch 是 running decode + 新 admission prefill。continuous batching 混。chunked prefill 切长 prefill。详见 [[continuous batching的调度实现]] 与 [[chunked prefill]]。

### 3.7 优先级

```
优先级 (谁拿 block):
  1. running decode (用户等, 最高)
  2. 新 admission prefill (用户等首 token, 次高)
  3. preempted 的恢复 (低, 已让位过)
preempt 优先选: preempted 的 (低优先级) 或刚 prefill 完待 decode 的
```

优先级保 SLO（用户等的先服务）。

### 3.8 scheduler 与 block manager

```
scheduler 问 block manager:
  can_allocate(req, blocks)?  # 够不够
  get_num_free_blocks()         # 预算
block manager 执行:
  allocate(req)  # 分 block
  free(victim)   # 回收
  swap_out/swap_in
```

scheduler 决策（preempt/admission），block manager 执行（block 操作）。分工。详见 [[KV block manager]]。

### 3.9 与 executor 的配合

```
scheduler 组 batch -> executor/worker forward -> 返回 logits
  -> sampler 采样 -> 更新 KV -> 检查完成
  -> 下一 iteration scheduler
```

scheduler 组 batch，executor forward。循环。详见 [[worker与executor]] 与 [[model runner]]。


## 4. 数学原理 / 公式

### 4.1 KV 预算约束

$$\sum_{r \in \text{running}} \text{blocks}(r) \leq \text{num\_blocks}$$

running 队列总 block $\leq$ 总 block。超则 preempt。

### 4.2 每 iteration 的 block 增长

decode 每 token 需 $\lceil 1 / \text{block\_size} \rceil$ 新 block（满 16 才需）。running $R$ 个请求，每 iteration 增长 $\leq R$ block（大部分不需，满 16 才需）。

### 4.3 admission 上限

$$\text{admission}(r) \iff \text{free\_blocks} \geq \text{prefill\_blocks}(r) + \text{margin}$$

需够 prefill block + 余量（decode 增长）。margin 是缓冲。

### 4.4 preempt 频率

$$f_{\text{preempt}} \propto \frac{1}{\text{num\_blocks} - \sum \text{blocks(running)}}$$

预算紧则频繁。预算足则少。是显存预算的 trade-off。

### 4.5 吞吐

$$\text{throughput} \propto \frac{|\text{batch}|}{T_{\text{forward}}}$$

batch 大（continuous batching 满）则吞吐高。scheduler 的 admission/preempt 决定 batch 大小。


## 5. 代码示例（可选）

### 5.1 vLLM scheduler 调用

```python
from vllm.core.scheduler import Scheduler
scheduler = Scheduler(block_manager, ...)
# 每 iteration
seq_group = scheduler.schedule()  # 返回 batch (running decode + 新 prefill)
# -> executor forward -> sampler -> 下一 iteration
```

### 5.2 调度循环（概念）

```python
while True:
    batch = scheduler.schedule()  # preempt + admission + 组 batch
    output = executor.forward(batch)  # GPU forward
    sampler.step(output)  # 采样, 更新 KV, 检查完成
    # 完成的 -> FINISHED, 释放 block
    # 下一 iteration
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[请求生命周期]]（被驱动的状态机）、[[KV block manager]]（block 操作执行）、[[continuous batching]]（batch 动态进出）、[[chunked prefill]]（长 prefill 切块）、[[cache eviction与swap offload]]（preempt 策略）、[[autoregressive decoding]]（prefill/decode）、[[worker与executor]]（forward 执行）。
- **下游（应用）**: vLLM serving 引擎核心、高并发 LLM 服务、SLO 保障调度。
- **对比 / 易混**:
  - **vLLM scheduler vs [[请求生命周期]]**：前者是调度器（每 iteration 决策，主动），后者是请求状态机（被驱动，被动）。前者驱动后者。
  - **vLLM scheduler vs [[continuous batching的调度实现]]**：前者是完整调度循环（preempt + admission + batch），后者是 batch 组成的 batching 细节。前者包含后者。
  - **vLLM scheduler vs [[dynamic batching]]**：前者是 iteration 级 continuous（每步进出），后者是请求级动态批（攒批）。前者更细粒度。
  - **vLLM scheduler vs [[Megatron Core目录与执行链路|Megatron 训练 step]]**：前者是推理调度（请求级），后者是训练调度（batch 级）。场景不同。


## 7. 常见误区与易错点

> [!warning] 误区 1：scheduler 一次调度完
> scheduler 是 **iteration 级循环**（每 forward 一次重新调度），不是一次调度完。每 iteration 重新评估 KV 预算、preempt/admission、组 batch。是动态的。

> [!warning] 误区 2：preempt 丢请求
> preempt 回 waiting 保留状态（不丢请求）。丢的是 KV（recompute）或换 CPU（swap）。下次继续。详见 [[请求生命周期]]。

> [!warning] 误区 3：admission 只看 FIFO
> FIFO 是默认，但可调优先级（如 prefill 短优先、SLO 紧优先）。不是纯 FIFO。是 FIFO + 优先级。

> [!warning] 误区 4：batch 固定
> batch 每 iteration 变（continuous batching，完成出/新进/preempt）。是动态的。不是固定 batch。

> [!warning] 误区 5：scheduler 算 forward
> scheduler 只组 batch + 调度。forward 由 executor/worker 执行。scheduler 决策，executor 算。分工。详见 [[worker与executor]]。

> [!warning] 误区 6：KV 预算静态
> KV 预算每 iteration 变（decode 增长、完成释放、preempt 回收）。scheduler 每 iteration 重新评估。是动态预算。

> [!warning] 误区 7：continuous batching 不需 scheduler
> continuous batching 需 scheduler 组 batch（谁进谁出）。无 scheduler 则无法动态进出。scheduler 是 continuous batching 的调度实现。


## 8. 延伸细节

### 8.1 调度策略的变体

vLLM 支持多种调度策略：FCFS（默认 FIFO）、priority（优先级）、SLO-aware（按 SLO）。是策略可配。详见 vLLM `SchedulerConfig`。

### 8.2 与 SGLang 的对照

SGLang 也有类似 scheduler。但 RadixAttention 的前缀复用影响 admission（前缀命中则少分 block）。详见 [[Radix Tree prefix cache]]。底层调度循环类似。

### 8.3 async scheduling

vLLM 支持 async scheduling（调度与 forward 异步，调度下一 batch 同时 forward 当前）。提吞吐。是优化。

### 8.4 内容来源

vLLM scheduler 整理自 vLLM `vllm/core/scheduler.py` 源码、vLLM 论文（SOSP 2023）。与请求生命周期的关系见 [[请求生命周期]]。与 continuous batching 的关系见 [[continuous batching的调度实现]]。

---
相关: [[请求生命周期与scheduler]] | [[请求生命周期]] | [[KV block manager]] | [[continuous batching]] | [[continuous batching的调度实现]] | [[chunked prefill]] | [[cache eviction与swap offload]] | [[autoregressive decoding]] | [[worker与executor]] | [[model runner]] | [[dynamic batching]] | [[Radix Tree prefix cache]]
