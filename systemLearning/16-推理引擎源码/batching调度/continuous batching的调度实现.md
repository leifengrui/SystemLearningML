# continuous batching的调度实现

> **所属章节**: [[batching调度]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: continuous batching 的调度实现 / iteration batching / vLLM/SGLang 调度实现 / 动态拼批实现 / 连续批调度源码
> **难度**: 中高（需懂 [[continuous batching]] + [[vLLM scheduler]] + [[PagedAttention]] + [[chunked prefill]] + [[请求生命周期]]）


## 1. 一句话定义

**continuous batching 的调度实现** 是 vLLM/SGLang 把 [[continuous batching]]（动态批，请求进出不等待）**落地为源码**的具体机制——核心是**每 iteration（一次 forward）重新组 batch**：scheduler 从 running 队列取所有 decode 中请求（各加 1 新 token），从 waiting 队列 admission 新请求（各加 prefill 的 prompt 或 chunk），用 [[PagedAttention]] 的**非连续 block 支持**把**变长请求无 padding 拼成一批**（每请求逻辑页独立，物理 block 散布，不需等长 padding 到桶长），一次 forward 同时算 decode（各 1 token）+ 新 prefill（各 prompt/chunk），forward 后 decode 请求各得 1 新 token（append KV）、prefill 完的进 running、生成 `<eos>` 的出 running 释放 block——下 iteration 再重新组 batch。这与**静态 batching**（攒批等齐 padding）对比：静态需等齐长度（短请求空转）、padding 浪费算力、新请求等攒批（延迟升）；continuous 每 iteration 动态进出（完成出、新进、preempt 回），无 padding（PagedAttention 非 contiguous）、无等待（即来即混）。vLLM 与 SGLang 实现细节略异（vLLM 用 block table + scheduler 显式组 batch；SGLang 用 RadixAttention + scheduler，prefix 复用影响 admission）。是 vLLM/SGLang 高吞吐的调度源码核心。

> [!note] 三句话定位
> - **是什么**：每 iteration 重新组 batch（running decode + 新 prefill），PagedAttention 非 contiguous 支持变长无 padding 拼批，完成出/新进/preempt 回动态进出。
> - **为什么**：静态 batching 等齐 padding（浪费/延迟），continuous 动态进出无 padding 无等待，吞吐 2-4×。需源码落地（scheduler + block manager + PagedAttention）。
> - **与 [[continuous batching]] 关系**：后者是概念（动态批，ch08），本笔记是 vLLM/SGLang 源码实现（iteration 组 batch + 非 contiguous + 进出机制）。是后者落地。


## 2. 为什么需要它（动机与背景）

### 2.1 静态 batching 的缺陷

静态 batching：攒一批，等齐长度（padding 到最长），一次 forward，全部完成才出。缺陷：
- 短请求 padding 浪费算力（补零 token）
- 新请求等攒批（延迟升）
- 完成的不出（等全部，空转）

LLM decode 每步 1 token，长度不一（有的早 `<eos>`），静态空转严重。

### 2.2 continuous 的动态进出

continuous batching：每 iteration 重新组 batch。完成的出（释放 block），新进（admission），preempt 回。无 padding（PagedAttention 非 contiguous 支持变长）。无等待（即来即混）。

### 2.3 PagedAttention 的非 contiguous 支持

变长请求（decode 1 token vs prefill 512 token）混 batch 需非连续（不能 padding 等长）。[[PagedAttention]] 的 block-level KV 支持非连续（各请求逻辑页独立，物理 block 散布）。是 continuous batching 的底层支持。

### 2.4 每 iteration 重新组 batch

batch 每 iteration 变（完成出/新进/preempt）。需 scheduler 每 iteration 重新组。是 dynamic 的核心。

### 2.5 vLLM vs SGLang 细节

两者都实现 continuous batching，但细节略异：
- vLLM：block table + scheduler 显式组 batch（running + waiting）
- SGLang：RadixAttention + scheduler，prefix 复用影响 admission（命中前缀则少分 block）

是源码层差异。


## 3. 核心概念详解

### 3.1 每 iteration 组 batch

```python
# 每 iteration (一次 forward)
def schedule():
    batch = []
    # running decode 中 (各 1 新 token)
    for r in running:
        batch.append(decode_step(r, 1))
    # 新 admission prefill (各 prompt 或 chunk)
    for r in waiting_admit:
        batch.append(prefill(r))  # 或 chunk
    # 完成的出 running, 新进的进 running
    update_lifecycle()
    return batch  # 变长混 (PagedAttention 支持)
```

每 iteration 重新组。running decode + 新 prefill 混。变长（PagedAttention 非 contiguous 支持）。

### 3.2 变长无 padding

```
batch:
  req_1 (decode 1 token): KV 逻辑页 5 个, 物理 block 散布
  req_2 (decode 1 token): KV 逻辑页 10 个
  req_3 (prefill 512 chunk): KV 逻辑页 32 个
  -> 一次 forward, 各自非连续 block, 无 padding (PagedAttention block-level)
```

变长请求无 padding。PagedAttention 的 block-level 支持（各请求独立 block table，物理散布）。是 vs 静态 batching（padding 等长）的核心差异。

### 3.3 动态进出

```
iteration N:
  running: [req_1, req_2, req_3]  (decode)
  waiting: [req_4, req_5]
forward -> req_3 生成 <eos> -> FINISHED -> 出 running, 释放 block
           req_4 admission -> 进 running (prefill)
iteration N+1:
  running: [req_1, req_2, req_4]  (req_3 出, req_4 进)
  -> 重新组 batch (continuous, 无需等齐)
```

完成出，新进。每 iteration 变。是 continuous（连续）的含义。

### 3.4 无等待即来即混

```
新请求 req_5 到 (waiting)
  下 iteration: scheduler admission (够 block 则进 running)
  与 running decode 混 batch
  不等攒批 (即来即混)
```

新请求即来即混（不攒批等齐）。降延迟（vs 静态等攒批）。

### 3.5 preempt 回 waiting

```
KV 不够:
  preempt victim (LRU) -> 回 waiting
  下 iteration: 可能重新 admission (或继续等)
  batch 动态调整 (preempt 的出 running)
```

preempt 也动态调整 batch。详见 [[请求生命周期]] 与 [[vLLM scheduler]]。

### 3.6 静态 vs continuous 对比

| 维度 | 静态 batching | continuous batching |
|---|---|---|
| 组 batch | 攒批一次，等齐 | 每 iteration 重新组 |
| padding | 等长 padding（浪费） | 无（PagedAttention 非 contiguous） |
| 进出 | 全部完成才出 | 完成即出，新即进 |
| 等待 | 等攒批 | 即来即混 |
| 短请求空转 | 严重（等长请求） | 无（完成即出） |
| 吞吐 | 低 | 2-4× |
| 延迟 | 高（等攒批） | 低（即来即混） |

### 3.7 vLLM 的实现

```python
# vLLM: block table + scheduler 显式组 batch
from vllm.core.scheduler import Scheduler
scheduler = Scheduler(block_manager, ...)
batch = scheduler.schedule()  # running decode + 新 prefill, 变长
output = executor.forward(batch, block_tables)  # PagedAttention 用 block_table
# 完成的 FINISHED, 新 admission, preempt 回
```

vLLM 用 block table（逻辑→物理）+ scheduler 显式组。PagedAttention kernel 据 block_table 算。

### 3.8 SGLang 的实现

```python
# SGLang: RadixAttention + scheduler, prefix 复用
# scheduler 组 batch 时查 RadixTree 前缀命中
# 命中则复用前缀 KV (少分 block), admission 更高效
batch = scheduler.schedule()  # + RadixTree 前缀复用
output = executor.forward(batch, ...)
```

SGLang 加 RadixAttention（前缀树复用），影响 admission（命中则少分 block）。详见 [[Radix Tree prefix cache]]。

### 3.9 与 chunked prefill 的配合

continuous batching 组 batch 时可加 chunked prefill（长 prefill 切 chunk 与 decode 混）。是 continuous 的变长混合细化。详见 [[chunked prefill]]。


## 4. 数学原理 / 公式

### 4.1 静态 padding 的浪费

静态 batch $B$ 请求，最长 $L_{\max}$，平均 $\bar{L}$。padding 浪费率：

$$\text{waste} = 1 - \frac{\bar{L}}{L_{\max}}$$

若 $\bar{L} = 100$, $L_{\max} = 2048$，浪费 95%。

### 4.2 continuous 无 padding

continuous 每 request 独立（PagedAttention 非 contiguous），无 padding。算力 100% 有效。吞吐 $\propto 1/\text{waste}$ 提升 2-4×。

### 4.3 每 iteration 的 batch 大小

$$|\text{batch}| = |\text{running\_decode}| + |\text{new\_prefill\_admit}|$$

每 iteration 变（running 变、admission 变、preempt 变）。受 max_num_batched_tokens 限。

### 4.4 吞吐

$$\text{throughput} \propto \frac{|\text{batch}|}{T_{\text{forward}}}$$

batch 大（continuous 满）则吞吐高。vs 静态（padding 浪费 + 等攒批），continuous 2-4×。

### 4.5 延迟

$$\text{latency} = T_{\text{wait}} + T_{\text{prefill}} + T_{\text{decode}} \cdot N$$

continuous 的 $T_{\text{wait}}$ 低（即来即混，不等攒批）。vs 静态 $T_{\text{wait}}$ 高（等攒批）。


## 5. 代码示例（可选）

### 5.1 vLLM continuous batching

```python
from vllm import LLM
llm = LLM(model="...", enable_chunked_prefill=True)  # continuous 默认开
# 底层: scheduler 每 iteration 组 batch (running + 新), PagedAttention 非 contiguous
outputs = llm.generate(prompts)  # 动态拼批, 完成/新进/preempt
```

### 5.2 调度循环（概念）

```python
while active_requests:
    batch = scheduler.schedule()  # running decode + 新 prefill, 变长无 padding
    output = executor.forward(batch, block_tables)  # PagedAttention
    sampler.step(output)  # 采样, KV append, 检查完成
    # FINISHED 出, 新 admission 进, preempt 回
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[continuous batching]]（ch08 概念）、[[vLLM scheduler]]（组 batch 的调度器）、[[PagedAttention]]（非 contiguous 支持变长）、[[chunked prefill]]（长 prefill 切块混）、[[请求生命周期]]（进出状态机）、[[KV block manager]]（block 分配）、[[autoregressive decoding]]、[[dynamic batching]]（ch08 请求级动态批）。
- **下游（应用）**: vLLM/SGLang serving 引擎核心、高并发 LLM 服务、RLHF rollout 生成（[[rollout worker]] 用 vLLM）。
- **对比 / 易混**:
  - **continuous batching 的调度实现 vs [[continuous batching]]**：前者是 vLLM/SGLang 源码实现（iteration 组 batch + 非 contiguous + 进出），后者是 ch08 概念（动态批，为什么/什么）。前者落地后者。
  - **continuous vs [[dynamic batching]]**：前者是 iteration 级（每 forward 进出，LLM 专属），后者是请求级（攒批等齐，通用）。前者更细粒度，后者更粗。
  - **continuous vs [[batching tradeoff]]**：前者是实现，后者是 trade-off 分析（延迟 vs 吞吐）。前者是后者的落地。
  - **vLLM vs SGLang**：两者都实现 continuous，但 SGLang 加 RadixAttention 前缀复用（影响 admission）。详见 [[Radix Tree prefix cache]]。


## 7. 常见误区与易错点

> [!warning] 误区 1：continuous batching = 动态批
> continuous 是 **iteration 级**（每 forward 进出），[[dynamic batching]] 是**请求级**（攒批等齐）。前者更细（LLM 专属），后者更粗（通用）。不能混。

> [!warning] 误区 2：continuous batching 需 padding
> 不需。[[PagedAttention]] 的非 contiguous block 支持变长（各请求独立 block table，物理散布）。无 padding 是 continuous 的核心优势。

> [!warning] 误区 3：batch 每 iteration 固定
> 每 iteration 变（完成出/新进/preempt 回）。是动态的。误以为固定会错判吞吐。

> [!warning] 误区 4：continuous batching 不需 scheduler
> 需 scheduler 每 iteration 组 batch（谁进谁出）。无 scheduler 则无法动态进出。scheduler 是 continuous 的调度实现。详见 [[vLLM scheduler]]。

> [!warning] 误区 5：vLLM 与 SGLang 实现完全一样
> 核心类似（iteration 组 batch + 非 contiguous），但 SGLang 加 RadixAttention 前缀复用（影响 admission）。细节略异。详见 [[Radix Tree prefix cache]]。

> [!warning] 误区 6：continuous batching 减 FLOPs
> 不减 FLOPs（每请求算力同）。提的是利用率（无 padding 浪费 + 无空转 + GPU 满）。是效率优化，非算力优化。

> [!warning] 误区 7：新请求立即 forward
> 新请求进 waiting，下 iteration admission（够 block 才进 running）。不是立即 forward。有 1 iteration 延迟。


## 8. 延伸细节

### 8.1 max_num_batched_tokens 的限制

vLLM 的 `--max-num-batched-tokens` 限每 iteration 总 token（含 decode + prefill chunk）。控制 batch 算力上限，防 OOM。是 continuous 的预算参数。

### 8.2 与 RLHF rollout 的关系

RLHF 的 [[rollout worker]] 用 vLLM 生成 trajectory。continuous batching 提生成吞吐（tokens/sec）。是 RLHF 工程的核心。详见 [[sampling throughput]]。

### 8.3 内容来源

continuous batching 的调度实现整理自 vLLM `vllm/core/scheduler.py` 与 `vllm/executor/` 源码、SGLang `sglang/srt/managers/` 源码、vLLM 论文（SOSP 2023）、Orca 论文（continuous batching 概念来源）。与 ch08 [[continuous batching]] 的关系见该笔记。与 chunked prefill 的配合见 [[chunked prefill]]。

---
相关: [[batching调度]] | [[continuous batching]] | [[vLLM scheduler]] | [[PagedAttention]] | [[chunked prefill]] | [[请求生命周期]] | [[KV block manager]] | [[autoregressive decoding]] | [[dynamic batching]] | [[batching tradeoff]] | [[Radix Tree prefix cache]] | [[sampling throughput]]
