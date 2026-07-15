# model runner

> **所属章节**: [[runner worker executor]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: model runner / GPUModelRunner / ModelRunner / GPU 侧执行核心 / 推理执行器 / MRV2
> **难度**: 中高（需懂 [[vLLM scheduler]] + [[PagedAttention]] + [[KV block manager]] + [[推理 CUDA Graph]] + [[continuous batching的调度实现]]）


## 1. 一句话定义

**model runner** 是 vLLM/SGLang 推理引擎中 **GPU 侧的执行核心**——它接收 scheduler 已经决定好的"这一步算哪些请求"的抽象 batch，把它**翻译成 GPU 能吃的张量**（input_ids / positions / block_table / seq_lens / attn_metadata），跑一次 model forward 得到 logits，再做 **采样（sample）** 和 **logprob 收集（gather logprob）**，最后把采样出的 token / logprob 回给 scheduler。它和 scheduler 是"决策 vs 执行"的分工：**scheduler 决定 batch 组成与 KV 预算（谁 decode、谁 prefill、谁 preempt），runner 只管把这一步在 GPU 上算完**。它还负责 **CUDA Graph 的捕获与回放**（启动时 `_dummy_run` 触发 capture、运行时 execute 走 replay），以及显存 profiling（测最大 KV block 数）。vLLM v1 的实现是 `vllm/v1/worker/gpu_model_runner.py` 的 `GPUModelRunner`；SGLang 对应 `sglang/srt/model_executor/model_runner.py` 的 `ModelRunner`（配合 `ForwardBatch`）。两者都向 MRV2 / 模块化方向演化。

> [!note] 三句话定位
> - **是什么**：GPU 侧把 scheduler 的抽象 batch 翻译成张量→forward→采样→回结果，并管 CUDA Graph 捕获/回放与显存 profiling。
> - **与 scheduler 关系**：scheduler 决策（batch 组成、preempt、KV 分配），runner 执行（建张量、跑模型、采样）。一决策一执行。
> - **演化**：vLLM V1 的 `GPUModelRunner`（大单体类、`_dummy_run` 多职责）→ MRV2（模块化、async-first、`StagedWriteTensor`、Triton sampler、显式 `CUDAGraphManager`）。


## 2. 为什么需要它（动机与背景）

### 2.1 调度与执行必须解耦

LLM serving 的"决策"（谁进 batch、谁被 preempt、KV 怎么分配、要不要 chunked prefill）是 **CPU 侧的离散策略**，逻辑复杂、状态多；而"执行"（建张量、调 kernel、采样）是 **GPU 侧的高频流水**，要批处理、要低开销。把两者揉一起会让 CPU 决策阻塞 GPU、也让 GPU 代码混入策略难维护。故分离：**[[vLLM scheduler]] 管决策，model runner 管执行**。scheduler 每 iteration 产出一个"执行计划"（哪些请求、每请求算到第几 token、各自的 block_table），runner 把它落成 GPU 张量并执行。

### 2.2 GPU 输入准备是性能瓶颈

continuous batching 每步 batch 变（请求进出），要把 per-request 的 `input_ids / positions / block_table / seq_lens / temperature / top_p / …` 拼成**连续 GPU 张量**喂给模型。朴素做法（每步 Python 循环重建张量）在 batch=256 时慢到 CPU 主导。runner 用 **persistent batch**（V1 的预分配持久张量 + 增量 diff，或 MRV2 的 `StagedWriteTensor`）和 **GPU-native 输入准备**（Triton kernel 生成 positions/seq_lens）把这块压下去。

### 2.3 采样与 logprob 与 forward 必须同进程

采样要 logits（在 GPU），logprob 收集要 softmax 结果，它们和 forward 强耦合、要共享 GPU 上下文，不宜跨进程。故 runner 把"forward + sample + gather logprob"打包成一次 GPU 侧调用。

### 2.4 CUDA Graph 归 runner 管

捕获 graph 要真正跑一遍 forward（`_dummy_run`），warmup 也要，显存 profiling（跑最大 batch 看占多少显存以反推 KV block 上限）也要。这些"启动期跑一遍模型"的活天然归 runner，于是 runner 也成了 CUDA Graph 的捕获/回放管理者。详见 [[推理 CUDA Graph]]。


## 3. 核心概念详解

### 3.1 scheduler vs runner 的分工（核心）

```
┌─────────────── CPU 侧 ───────────────┐        ┌─────────────── GPU 侧 ───────────────┐
│  vLLM scheduler (每 iteration)        │        │  model runner (GPUModelRunner)        │
│                                       │        │                                       │
│  running/waiting 队列管理             │ ──计划─▶│  _prepare_inputs:                     │
│  KV 预算检查 / preempt                │        │    把请求计划 -> 张量                 │
│  admission / chunked prefill 切块     │        │  execute_model (forward, 含 graph     │
│  组 batch (decode + prefill chunk)    │        │    replay)                            │
│  -> 输出 SchedulerOutput (执行计划)   │        │  sample (采样)                        │
│                                       │        │  gather logprob                       │
│  <- 接收采样结果, 更新请求状态 ────────│◀─结果──│  -> 输出 token / logprob              │
└───────────────────────────────────────┘        └───────────────────────────────────────┘
```

| 维度 | scheduler | model runner |
|---|---|---|
| 侧 | CPU | **GPU** |
| 职责 | **决策**：batch 组成、preempt、KV 分配、chunked | **执行**：建张量、forward、采样、logprob |
| 频率 | 每 iteration 一次 | 每 iteration 一次（但内部含数百 kernel） |
| 状态 | running/waiting 队列、KV free list | persistent batch 张量、model 权重、graph 池 |
| 输入 | 请求队列、KV 状态 | scheduler 的执行计划（SchedulerOutput） |
| 输出 | 执行计划 | 采样 token / logprob |
| async | 决策 N+1 时 GPU 在跑 N（overlap） | 跑 N 时 CPU 在决策 N+1 |

### 3.2 runner 的五大职责

1. **`_prepare_inputs`（准备输入）**：把 scheduler 的执行计划翻译成 GPU 张量。
2. **synchronize KV cache**：确保 attention 用到正确的 KV block（block_table 已更新、KV 已写入对应物理 block）。详见 [[PagedAttention]]、[[KV block manager]]。
3. **`execute_model`（执行模型）**：跑 model forward 得 logits；命中则走 graph replay（[[推理 CUDA Graph]]），否则 eager。
4. **sample（采样）**：据 logits + temperature/top_p/top_k 等 → 选 token。
5. **gather logprob**：按需算选中 token / prompt token 的 logprob 回给 scheduler。

### 3.3 _prepare_inputs / _prepare_decode / _prepare_prefill

这三个方法（V0/V1 命名；V1 中合并为 `prepare_inputs` 返回 `ModelInput`，概念一致）把"执行计划"落成张量，按 prefill/decode 分路：

```
prepare_inputs(scheduler_output):
    # 1. 解析计划: 哪些请求 decode、哪些 prefill (含 chunked)
    # 2. 收集 per-request 的:
    #    input_ids     : 本步要算的 token (decode=1, prefill=chunk 长度)
    #    positions     : 每个 token 的绝对位置 (RoPE 用)
    #    block_table   : 每请求的 KV block 映射 (PagedAttention 寻址)
    #    seq_lens      : 每请求当前 KV 长度
    #    sampling_info : temperature/top_p/top_k/...
    # 3. 拼成连续 GPU 张量 (padding-free, PagedAttention 支持变长)
    # 4. 构 attn_metadata (query_start_loc, seq_lens, block_table, slot_mapping)
    return ModelInput(input_ids, positions, attn_metadata, ..., sampling_info)

_prepare_decode  : query_len=1 的路径 (走 graph replay)
_prepare_prefill : query_len>1 的路径 (chunked prefill, 多走 piecewise/eager)
```

> [!note] padding-free 拼接
> vLLM/SGLang 不把 batch padding 成等长（浪费算力），而是把所有请求的 token **首尾相接拼成一维** `[num_tokens]`，attention 用 `query_start_loc`（每请求 query 的起止）和 `seq_lens` 切分。这是 [[PagedAttention]] 的高效 batching 基础。

### 3.4 CUDA Graph 捕获与回放（runner 视角）

runner 是 graph 的"持有与触发者"：

- **捕获（启动期）**：runner 调 `_dummy_run`（V1）/ 显式 capture 路径（MRV2 的 `CUDAGraphManager`），用代表性 `attn_metadata` 跑一遍 forward，把 kernel 序列录成图。vLLM v1 用 `CudagraphDispatcher` + `BatchDescriptor` 决定捕获哪些 shape key（见 [[推理 CUDA Graph]]）。warmup 用 `NONE` mode eager 跑，再切目标 mode 捕获。
- **回放（运行期）**：`execute_model` 里若当前 batch 命中已捕获 key，则 `static_buf.copy_(real)` 注入数据后 `cudaGraphLaunch`；否则 eager。
- **显存 profiling**：runner 启动时用最大 batch 跑一遍 forward，测激活/权重/激活显存，反推剩余显存能放多少 KV block → 告诉 [[KV block manager]] 上限。

### 3.5 sampler

采样在 runner 内，输入 logits `[num_tokens, vocab]`，输出 token id + logprob。vLLM V1 的 sampler（`vllm/v1/sample/`）做：logits processor（temperature/top_p/top_k 裁剪）→ argmax / multinomial 采样 → top-k logprob 收集。MRV2 把采样重写为 **Triton kernel**（Gumbel-max 采样免显式 softmax materialization、top-k logprob 只算选中 token 而非整词表，降显存）。

> [!tip] 实践：采样是 decode 最后一步
> greedy（temperature=0）= argmax；随机采样 = softmax(logits/T) 后 multinomial。采样本身访存主导（读整词表 logits），MRV2 的 top-k-first 策略避免 materialize 全词表 logprob。

### 3.6 V1 → MRV2 演化（基于官方 design doc）

vLLM V1 的 `GPUModelRunner` 是个"大单体类"，`_dummy_run` 身兼数职（profiling + compile + graph capture + warmup + DP 空跑），技术债重。MRV2（Model Runner V2，`vllm/v1/worker/gpu/` 下模块化）从第一性原理重写：

| 维度 | V1 | MRV2 |
|---|---|---|
| 结构 | 大单体 `gpu_model_runner.py` | 模块化 `v1/worker/gpu/`（`model_states`、`sample`、`pool` 等） |
| persistent batch | 持久张量即模型输入（耦合，join/finish 要重排） | **解耦**：持久状态张量与每步输入张量分离，用 `idx` gather |
| async | 后加的 hack | **async-first**：核心是纯 CUDA stream，无 CPU sync |
| async 竞态 | 用 barrier 保护共享 CPU buffer | **消除竞态**：状态张量不 pin，临时 pin copy（`pin_memory()`） |
| block_table 更新 | 每步重建 | **`StagedWriteTensor`**：base 在 GPU、diff 在 CPU、打包 copy、一个 kernel apply |
| 输入准备 | Python 构张量 | **GPU-native**：Triton kernel 生成 positions/seq_lens/query_start_loc |
| 采样 | Python + 部分 kernel | **Triton-native**：Gumbel sampling、top-k-first logprob |
| dummy_run | 多职责（profiling/compile/capture/warmup/DP） | **专一化**：`execute_model` 支持空跑不改状态；capture 独立路径 |
| CUDA Graph | 隐式难推理 | **显式 `CUDAGraphManager`**（标准 PyTorch API 捕获/启动，可多 draft 合一图） |
| 请求状态 | `CachedRequestState` 冗余备份 | 用持久行槽（max_num_reqs 行），preempt 视为完成、恢复时新加 |


## 4. 数学原理 / 公式

### 4.1 persistent batch 的增量 diff

设持久 block_table 在 GPU 为 $T\in\mathbb{Z}^{R\times M}$（$R$=max_num_reqs，$M$=max_blocks/req）。每步只有少数请求的少量行变化。V1 整张 copy 开销 $\propto R\cdot M$；`StagedWriteTensor` 只 copy diff：

$$
\text{copy size} = \sum_{(r,s)\in \Delta} \text{width}(r,s) \ll R\cdot M
$$

其中 $\Delta$ 是本步变化的（行,起始）集合。打包成连续 buffer 后一次 H2D + 一个 scatter kernel apply，省带宽与 launch。

### 4.2 padding-free 拼接的 query 切分

batch 内请求 $i$ 的 query 长度 $q_i$、KV 长度 $s_i$。所有 token 拼成一维，第 $i$ 请求 query 起止：

$$
\text{start}_i = \sum_{j<i} q_j,\quad \text{end}_i = \text{start}_i + q_i
$$

attention 对请求 $i$：query $[\text{start}_i,\text{end}_i)$ 对 key/value $[0, s_i)$（由 block_table 映射到物理 block）。`query_start_loc` 即累积和 $\text{cumsum}(q)$。无 padding 时总 token $\sum q_i$，算力 $\propto \sum q_i$；若 padding 到 $\max q_i\cdot R$ 则浪费 $\propto R\max q_i - \sum q_i$。

### 4.3 采样的 softmax / Gumbel

标准随机采样：

$$
p_t = \frac{\exp(\text{logit}_t/T)}{\sum_k \exp(\text{logit}_k/T)},\quad t^* \sim \text{Categorical}(p)
$$

MRV2 的 Gumbel-max 避免 materialize softmax：抽 Gumbel 噪声 $g_t\sim\text{Gumbel}(0,1)$，选：

$$
t^* = \arg\max_t \big(\text{logit}_t/T + g_t\big)
$$

与 categorical 采样等价但无需归一化大词表，单 kernel 内完成、省显存。


## 5. 代码示例（可选）

### 5.1 V1 风格 runner 主循环（概念，对齐 vLLM 结构）

```python
import torch

class GPUModelRunner:
    def __init__(self, model, config, kv_cache_manager):
        self.model = model
        self.kv_manager = kv_cache_manager
        self.graph_pool = {}          # BatchDescriptor -> cuda graph
        self.static_buffers = {}      # 静态 input/output buffer (graph 用)

    def prepare_inputs(self, scheduler_output):
        """把执行计划 -> GPU 张量 (padding-free 拼接)"""
        reqs = scheduler_output.scheduled_reqs
        input_ids, positions, block_tables, seq_lens = [], [], [], []
        for r in reqs:
            input_ids.append(r.next_token_ids)      # decode: [1]; prefill: [chunk]
            positions.append(r.positions)
            block_tables.append(r.block_table)        # KV block 映射
            seq_lens.append(r.seq_len)
        # 首尾相接拼一维
        input_ids = torch.tensor(sum(input_ids, []), device="cuda")
        positions = torch.tensor(sum(positions, []), device="cuda")
        block_table = torch.tensor(sum(block_tables, []), device="cuda")
        seq_lens = torch.tensor(seq_lens, device="cuda")
        query_start_loc = torch.cumsum(torch.tensor([len(q) for q in reqs]), dim=0)
        attn_meta = AttnMetadata(query_start_loc, seq_lens, block_table, ...)
        sampling_info = SamplingInfo(temperatures=[r.t for r in reqs], ...)
        return ModelInput(input_ids, positions, attn_meta, sampling_info)

    def execute_model(self, model_input, kv_caches):
        """forward: 命中 graph 则 replay, 否则 eager"""
        key = self._dispatch_key(model_input)        # BatchDescriptor
        if key in self.graph_pool:
            self._copy_static(model_input, key)      # 数据注入静态 buffer
            self.graph_pool[key].replay()             # 一次 launch
            logits = self.static_buffers[key].logits
        else:
            logits = self.model(model_input, kv_caches)   # eager
        return logits

    def sample(self, logits, sampling_info):
        """采样 -> token + logprob"""
        logits = apply_logits_processors(logits, sampling_info)  # temperature/top_p/top_k
        if all(sampling_info.is_greedy):
            tokens = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits / sampling_info.temperatures, dim=-1)
            tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
        logprobs = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))
        return tokens, logprobs

    def __call__(self, scheduler_output, kv_caches):
        """一条龙: prepare -> execute -> sample"""
        mi = self.prepare_inputs(scheduler_output)
        logits = self.execute_model(mi, kv_caches)
        tokens, logprobs = self.sample(logits, mi.sampling_info)
        return tokens, logprobs
```

### 5.2 显存 profiling（启动期决定 KV 上限）

```python
@torch.inference_mode()
def profile_kv(self):
    """跑最大 batch forward, 测显存 -> 算 KV block 上限"""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    dummy = self._build_max_dummy_input()          # 最大 batch 的假输入
    self.model(dummy, self.kv_manager.kv_caches)  # 跑一遍
    peak = torch.cuda.max_memory_allocated()
    free = torch.cuda.get_device_properties(0).total_memory - peak - self._weights_mem
    num_blocks = free // self.kv_manager.block_bytes()
    self.kv_manager.set_num_blocks(num_blocks)     # 告诉 block manager 上限
```

### 5.3 MRV2 的 StagedWriteTensor（来自官方 doc 示例）

```python
from vllm.staged_write_tensor import StagedWriteTensor   # 路径待核实, 概念同官方示例

state = StagedWriteTensor(size=(1024, 1000), dtype=torch.int32, device="cuda")
state.stage_write(row=2, start=3, value=[3, 1, 2])     # 暂存 diff 到 CPU
state.stage_write(row=0, start=1, value=[-1, -2, -5])
state.apply_write()                                    # 打包 copy -> 一个 kernel apply, 无 sync
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[vLLM scheduler]]（给它执行计划）、[[continuous batching的调度实现]]（batch 组成规则）、[[PagedAttention]] + [[KV block manager]]（KV 寻址）、[[推理 CUDA Graph]]（捕获/回放）
- **下游（应用）**: [[worker与executor]]（worker 持 runner、跨进程调用它）、[[TTFT与TPOT]]（runner 的 forward/采样延迟直接决定 TPOT，prefill 延迟影响 TTFT）
- **对比 / 易混**:
  - vs scheduler：决策 vs 执行（见 3.1 表）。
  - vs worker：runner 是 GPU 执行核心，worker 是**进程级壳**（持 runner、管通信/权重加载/TP-PP 通信）。runner 被 worker 持有。见 [[worker与executor]]。
  - vLLM `GPUModelRunner` vs SGLang `ModelRunner`：见 3.7。
  - V1 vs MRV2：单体 vs 模块化（见 3.6 表）。


## 7. 常见误区与易错点

> [!warning] 误区1：以为 runner 做调度
> 不做。runner 不决定 batch 组成、不 preempt、不分配 KV——那是 scheduler。runner 只把已定计划落成张量并执行。混淆会导致读代码时找错"谁决定谁进 batch"。

> [!warning] 误区2：以为 prepare_inputs 每步全量重建张量就够
> batch=256 时 Python 全量重建会成 CPU 瓶颈。V1 用 persistent batch 增量、MRV2 用 StagedWriteTensor + GPU-native 生成，正是为此优化。

> [!warning] 误区3：把采样放在 scheduler/外进程
> 采样要 logits（GPU），跨进程拷 logits（[B,vocab] fp16，词表 12 万）极浪费。采样必须在 runner 内、与 forward 同上下文。

> [!warning] 误区4：以为 graph 捕获是 runner 之外的某组件触发
> 捕获要真跑 forward，runner 是唯一持模型+能跑的组件，故捕获归 runner（`_dummy_run` / `CUDAGraphManager`）。dispatcher 只决定"捕哪些 key"，真捕获动作在 runner。

> [!warning] 误区5：padding-free 拼接忘了 query_start_loc
> 一维拼接后 attention 必须靠 `query_start_loc`/`seq_lens` 切分每请求，否则 attention 越界/串请求。

> [!warning] 误区6：显存 profiling 漏算激活峰值
> profiling 必须跑"最大 batch + 最长 prefill"取峰值，少算会导致运行时 OOM。MRV2/long-prompt 还要 chunk 内分块避免 logits 峰值。


## 8. 延伸细节

- **MRV2 的 UVA**：用 Universal Virtual Addressing 让 GPU kernel 直接访问大 CPU 张量（如 `prefill_token_ids`）免复制，进一步降输入准备开销。
- **MRV2 的 top-k-first logprob**：先从 logits 选 top-k token，只对这些算 logprob，而非 materialize 整词表 softmax，省 `[B, vocab]` 中间显存。prompt logprob 还支持单 prompt 内分块避免峰值。
- **async overlap**：scheduler 决策 N+1 时 runner 在 GPU 跑 N，runner 的输入准备必须无 CPU sync（不调 `torch.cuda.synchronize`、不 unpinned `.to("cuda")`），否则 overlap 失效。MRV2 的"不 pin 状态张量 + 临时 pin copy"即为此。
- **MoE / EP**：MoE 的 token 路由使 attention 外的算子更复杂，runner 需处理 expert 路由的 gather/scatter；EP+DP 时空跑 forward 也走 runner（V1 滥用 `dummy_run`，MRV2 改用 `execute_model` 空跑）。
- **SGLang 的 ForwardBatch**：SGLang 用 `ForwardBatch`（`forward_batch_info.py`）封装"一步要算什么"，由 scheduler 构造、传给 ModelRunner；与 vLLM 的 `ModelInput` 角色对偶。SGLang 还把 cuda graph 逻辑拆到 `model_runner_components/cuda_graph_setup.py`，与 runner 解耦更彻底。
- **多模态 runner**：多模态输入（图像 token）的拼接、ViT 编码也在 runner 侧（或其 mixin），`mm`/`vit_cuda_graph_runner` 等处理视觉编码路径。

---
相关: [[runner worker executor]]、[[worker与executor]]、[[vLLM scheduler]]、[[continuous batching的调度实现]]、[[PagedAttention]]、[[KV block manager]]、[[推理 CUDA Graph]]、[[TTFT与TPOT]]
