# chunked prefill

> **所属章节**: [[batching调度]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: chunked prefill / 分块预填 / prefill 切块 / prefill chunk scheduling / 混合 prefill+decode batch
> **难度**: 中（需懂 [[vLLM scheduler]] + [[continuous batching]] + [[autoregressive decoding]] + [[PagedAttention]]）


## 1. 一句话定义

**chunked prefill** 是 vLLM/SGLang 的**长 prefill 切块调度**机制——当请求的 prompt 很长（如 8k token），传统做法是**一次 forward 算完整个 prefill**（compute-bound，占满 GPU，阻塞同 batch 的 decode 请求数秒），导致 decode 请求饿死（TPOT 飙升）；chunked prefill 把长 prompt **切成固定 chunk（如 512 token）**，每 iteration 只算一个 chunk，剩余 chunk 排队下 iteration，**让长 prefill 与 decode 在同 batch 混合**（decode 的 1 token + prefill 的 512 token 一起 forward），使 decode 不再被长 prefill 阻塞、TTFT 降（首 chunk 算完即进 decode）、KV 增量分配（每 chunk 只需该 chunk 的 block，免一次性大额申请触发 preempt 重算）、prefill 算力与 decode 带宽交替吃满更均衡。是 [[vLLM scheduler]] 的 batching 优化——scheduler 组 batch 时把长 prefill 切 chunk 与 decode 混，是 [[continuous batching的调度实现]] 的变长混合细化。注意 chunked prefill **几乎不减 FLOPs**（总 prefill 算力同），省的是**等待/闲置/重算**——核心是调度优化（避免长 prefill 独占 GPU 冻结 decode 产能）。

> [!note] 三句话定位
> - **是什么**：长 prefill 切固定 chunk（512 token），每 iteration 算一 chunk，与 decode 混合同 batch。避免长 prefill 阻塞 decode。
> - **为什么**：长 prefill 一次算占满 GPU 数秒，decode 请求饿死（TPOT 飙）；切块让 decode 不等，TTFT 降、KV 增量分配免 preempt、算力带宽交替均衡。
> - **与 [[continuous batching的调度实现]] 关系**：后者是 continuous batching 的 batch 组成，chunked prefill 是变长混合（prefill chunk + decode）的细化。是后者的一维优化。


## 2. 为什么需要它（动机与背景）

### 2.1 长 prefill 阻塞 decode

长 prompt（8k）一次 prefill 算数秒（compute-bound）。同 batch 的 decode 请求等（GPU 被 prefill 占）。decode 饿死，TPOT 飙升（用户等每 token）。

### 2.2 一次性 KV 申请触发 preempt

长 prefill 一次需大量 KV block（8k/16=500 block）。若 free list 不够，触发 preempt（驱逐其他请求重算）。长 prefill 反复触发 preempt，损吞吐。

### 2.3 切块混 decode

chunked prefill 切长 prompt 成 chunk（512）。每 iteration 算一 chunk + decode 混。decode 不等长 prefill，TPOT 稳。TTFT 降（首 chunk 算完即 decode）。

### 2.4 KV 增量分配

每 chunk 只需该 chunk 的 block（512/16=32 block）。增量申请，免一次性大额（500 block）触发 preempt。零存整取 vs 一次性大额。

### 2.5 算力带宽交替

prefill 是 compute-bound（算力），decode 是 memory-bound（带宽）。混合让两者交替吃满 SM + 带宽，更均衡。


## 3. 核心概念详解

### 3.1 传统 vs chunked

```
传统 (长 prefill 一次):
  iteration 1: prefill 8k (占 GPU 5s, decode 饿死)
  iteration 2: decode all (但 TPOT 已飙)
chunked (切 chunk 512):
  iteration 1: prefill chunk 0 (512) + decode all  (decode 不等)
  iteration 2: prefill chunk 1 (512) + decode all
  ...
  iteration 16: prefill chunk 15 (最后) + decode all  (prefill 完)
  -> decode 全程不饿死
```

传统长 prefill 独占 GPU 冻结 decode。chunked 切块混 decode。

### 3.2 chunk size

```
chunk_size = 512 (token/chunk, 典型 vLLM 默认)
  小: decode 不饿死, 但 prefill 慢 (多 iteration)
  大: prefill 快, 但 decode 等长 (接近传统)
512 是经验最优 (decode 延迟与 prefill 速度平衡)
```

chunk_size 是 trade-off。小则 decode 不饿但 prefill 慢。大则 prefill 快但 decode 等。vLLM 默认 512（可配）。

### 3.3 混合 batch

```
batch = decode 请求 (各 1 token) + prefill 请求 (1 chunk = 512 token)
  -> 一次 forward (continuous batching, 变长混合)
  decode: 各 1 新 token
  prefill: 该 chunk 的 KV 入 cache
  下 iteration: 下一 chunk
```

decode + prefill chunk 混。变长（1 vs 512）但 padding-free（[[PagedAttention]] 非 contiguous 支持）。

### 3.4 KV 增量分配

```
传统: 一次性申请 8k/16 = 500 block (可能不够 -> preempt)
chunked: 每 chunk 申请 512/16 = 32 block (增量, 够)
  -> 免 preempt, 0 重算
```

增量分配（每 chunk 32 block）vs 一次性（500 block）。免 preempt。详见 [[KV block manager]]。

### 3.5 TTFT 的降

```
传统: prefill 全部完 (5s) -> 首 token (TTFT=5s)
chunked: 首 chunk 完 (0.3s) -> 进 decode 首 token (TTFT=0.3s)
  -> TTFT 降 (用户更快看到首字)
```

TTFT 降（首 chunk 即 decode）。是用户体验提升。

### 3.6 不减 FLOPs

```
总 prefill FLOPs 同 (8k prompt 的算力):
  传统: 一次算完
  chunked: 16 chunk 分 16 iteration 算 (总同)
  不减 FLOPs, 只改调度 (避免阻塞)
```

chunked 不减总 prefill 算力。省的是等待/闲置/重算（调度优化）。是关键认知。

### 3.7 与 continuous batching 的关系

chunked prefill 是 [[continuous batching]] 的变长混合细化。continuous batching 动态进出，chunked 加 prefill chunk 与 decode 混。是 batching 的一维优化。详见 [[continuous batching的调度实现]]。

### 3.8 scheduler 的实现

```python
# scheduler 组 batch 时切长 prefill
def schedule():
    batch = []
    # decode 中请求 (各 1 token)
    for r in running:
        batch.append(decode_token(r))
    # 新 prefill 请求 (切 chunk)
    for r in waiting_admit:
        chunk = r.prompt[r.chunk_offset : r.chunk_offset + chunk_size]
        batch.append(chunk)  # prefill chunk
        r.chunk_offset += chunk_size
        if r.chunk_offset >= len(r.prompt):
            r.state = RUNNING  # prefill 完, 进 decode
    return batch
```


## 4. 数学原理 / 公式

### 4.1 传统 prefill 的 decode 冻结

长 prefill $L$ 一次算，耗时 $T_{\text{prefill}}(L) \propto L \cdot d^2$。此期间 decode $B_{\text{dec}}$ 请求冻结，产能损失：

$$\text{loss} \propto B_{\text{dec}} \cdot \frac{T_{\text{prefill}}(L)}{\text{TPOT}_{\text{target}}}$$

decode 请求等长 prefill，TPOT 飙。

### 4.2 chunked 的 decode 不冻

切 $K = \lceil L / C \rceil$ chunk（$C$=chunk_size）。每 iteration 算一 chunk $T_{\text{prefill}}(C) \propto C \cdot d^2$ + decode。decode 每步都跑（不被长 prefill 冻）。TPOT 稳。

### 4.3 FLOPs 不减

$$\text{FLOPs}_{\text{traditional}} = \text{FLOPs}_{\text{chunked}} = 2 L d^2$$

总同。chunked 只分 iteration。

### 4.4 TTFT 的降

传统 $TTFT = T_{\text{prefill}}(L)$。chunked $TTFT = T_{\text{prefill}}(C)$（首 chunk 完即 decode）。$C \ll L$ 则 TTFT 大降。

### 4.5 KV 增量的预算

传统一次需 $\lceil L/B \rceil$ block。chunked 每 chunk 需 $\lceil C/B \rceil$ block。$C \ll L$ 则增量小，免 preempt。


## 5. 代码示例（可选）

### 5.1 vLLM chunked prefill 配置

```bash
# vLLM 启动开 chunked prefill
python -m vllm.entrypoints.openai.api_server \
    --enable-chunked-prefill \  # 开
    --max-num-batched-tokens 8192  # 每 iteration 总 token 上限 (含 chunk + decode)
# scheduler 切长 prefill 与 decode 混
```

### 5.2 scheduler 切 chunk（概念）

```python
def schedule():
    batch_tokens = []
    for r in running_decode:
        batch_tokens.append(1)  # decode 1 token
    for r in waiting_prefill:
        chunk = min(chunk_size, len(r.prompt) - r.offset)
        batch_tokens.append(chunk)
        r.offset += chunk
    # 总 token <= max_num_batched_tokens
    return batch
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[vLLM scheduler]]（调度循环）、[[continuous batching]]（batch 动态进出）、[[continuous batching的调度实现]]（batch 组成）、[[autoregressive decoding]]（prefill/decode 区分）、[[PagedAttention]]（非 contiguous 支持变长混）、[[KV block manager]]（增量分配）。
- **下游（应用）**: vLLM/SGLang 长 prompt serving、TTFT 优化、decode 不饿死、高并发混合负载。
- **对比 / 易混**:
  - **chunked prefill vs [[continuous batching]]**：后者是 batch 动态进出（请求级），前者是 prefill 切块与 decode 混（token 级）。前者是后者的细化。
  - **chunked prefill vs [[PD分离]]**：前者是同 GPU 内 prefill 与 decode 混（切 chunk），后者是 prefill 与 decode 分到不同 GPU 池。前者同机混，后者分机。
  - **chunked prefill vs [[prefix caching]]**：前者是切 prefill 调度，后者是复用前缀 KV。正交（可都开）。前者不减 FLOPs，后者减（复用）。
  - **chunked prefill vs recompute**：前者是 preempt 后重算时也可分块（减重算冲击），后者是 preempt 策略。前者减重算冻结。


## 7. 常见误区与易错点

> [!warning] 误区 1：chunked prefill 减 FLOPs
> 不减。总 prefill 算力同（$2Ld^2$）。只分 iteration 改调度。省的是等待/闲置/重算，不是算力。

> [!warning] 误区 2：chunked prefill 总更快
> 不总是。短 prompt 不需切（切反慢，多 iteration overhead）。长 prompt 才受益。chunk 太小反慢（overhead）。

> [!warning] 误区 3：chunked prefill 与 PD 分离一样
> 不同。前者同 GPU 内 prefill+decode 混（切 chunk），后者分到不同 GPU 池。前者同机，后者分机。

> [!warning] 误区 4：chunk size 越小越好
> 小则 decode 不饿但 prefill 慢（多 iteration overhead）。大则 prefill 快但 decode 等。512 是经验最优。太小反慢。

> [!warning] 误区 5：chunked prefill 与 prefix caching 互斥
> 正交。可都开。前者切 prefill 调度，后者复用前缀 KV。两者叠加效果更好。

> [!warning] 误区 6：chunked prefill 免一切 preempt
> 免长 prefill 的一次性大额申请 preempt。但若总 KV 仍不够（高并发），仍需 preempt。是减触发，非消除。

> [!warning] 误区 7：chunked prefill 自动开
> vLLM 需 `--enable-chunked-prefill` 显式开。默认关（早期）。SGLang 默认开。是配置项。


## 8. 延伸细节

### 8.1 max_num_batched_tokens

vLLM 的 `--max-num-batched-tokens` 限每 iteration 总 token（含 prefill chunk + decode）。控制 batch 算力上限。是 chunked prefill 的预算参数。

### 8.2 与 SGLang 的对照

SGLang 默认开 chunked prefill（类似机制）。底层 block 管理类似。详见 [[continuous batching的调度实现]]。

### 8.3 内容来源

chunked prefill 整理自 vLLM `vllm/core/scheduler.py`（chunk 调度逻辑）源码、vLLM 论文与 blog（"How to serve LLMs with chunked prefill"）、社区教程。与 continuous batching 的关系见 [[continuous batching的调度实现]]。与 PD 分离的区别见 [[PD分离]]。

---
相关: [[batching调度]] | [[vLLM scheduler]] | [[continuous batching]] | [[continuous batching的调度实现]] | [[autoregressive decoding]] | [[PagedAttention]] | [[KV block manager]] | [[PD分离]] | [[prefix caching]] | [[TTFT与TPOT]]
