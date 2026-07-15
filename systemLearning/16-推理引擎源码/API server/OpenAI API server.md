# OpenAI API server

> **所属章节**: [[API server]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: OpenAI API server / 推理 HTTP server / 前端 router / OpenAI 兼容协议 / API server / serving frontend / FastAPI server / streaming SSE / 负载均衡 router
> **难度**: 中（需懂 [[tokenization]] + [[采样策略]] + [[continuous batching的调度实现]] + [[model runner]] + [[TTFT与TPOT]] + [[goodput]]）


## 1. 一句话定义

**OpenAI API server** 是 vLLM / SGLang 的**HTTP 对外服务层**——它用 FastAPI + uvicorn 起一个 ASGI server，实现 **OpenAI 兼容协议**（`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings` 及 streaming SSE），前端 router 解析 HTTP 请求 → tokenize（`messages`/`prompt` → `input_ids`）→ 组 `SamplingParams` → 丢进异步引擎队列（vLLM V1 的 `AsyncLLM` / V0 的 `AsyncLLMEngine`；SGLang 的 `TokenizerManager`→`Scheduler`），引擎在 [[continuous batching的调度实现]] 里算完 → 结果 detokenize 回文本 → 流式（SSE chunked，token 逐个 yield）或非流式返回。它还管**负载均衡**（多实例 router：round-robin / KV-aware / prefix-aware routing）、**功能扩展**（LoRA 多模型路由、function calling 工具调用、structured output、logprob 返回、n 采样、best_of）、与 **OpenAI 的兼容边界**（支持哪些参数、哪些不支持）。性能要点：**asyncio event loop 不阻塞**（tokenize/detokenize 与引擎 forward 在不同协程/进程，HTTP IO 不卡 GPU）、**batch 在引擎层合并**（HTTP 层只入队，continuous batching 在引擎内合 batch）。SGLang 的同类是 `sglang.srt.entrypoints.http_server` 的 FastAPI + `TokenizerManager`/`DetokenizerManager` + `Scheduler`。

> [!note] 三句话定位
> - **是什么**：FastAPI/uvicorn 的 OpenAI 兼容 HTTP server：解析请求→tokenize→入队→引擎 continuous batching 算→detokenize→SSE 流式/非流式返回。管负载均衡与功能扩展。
> - **与引擎关系**：HTTP 层只管协议解析与入队，**不算模型**。模型 forward 在引擎内（[[model runner]] + [[continuous batching的调度实现]]）。两层用异步队列解耦，HTTP IO 不卡 GPU。
> - **与 OpenAI 关系**：尽力兼容 OpenAI Chat/Completions API（可用官方 OpenAI client 直连），但有边界（部分参数不支持、vLLM 扩展参数走 `extra_body`）。


## 2. 为什么需要它（动机与背景）

### 2.1 用户要 HTTP 接口，不要 Python API

生产 serving 面向异构客户端（Web 后端、SDK、curl、OpenAI client）。要一个**标准 HTTP 协议**让任意客户端能调。直接暴露 Python `llm.generate()` 不现实（跨进程/跨语言/并发难）。故起 HTTP server，把"调模型"封装成"发 HTTP 请求"。

### 2.2 OpenAI 协议是事实标准

OpenAI 的 Chat/Completions API 已成 LLM serving 的事实标准（client 库、评测脚本、应用框架都按它写）。vLLM/SGLang 实现 OpenAI 兼容协议，让用户**换后端不改 client 代码**（改 `base_url` 即可）。这是生态兼容的关键。

### 2.3 流式响应用户体验

LLM 生成慢（秒级），用户要**边生成边看**（流式）。HTTP server 用 **SSE（Server-Sent Events）chunked** 逐 token yield，首 token 即推（低 [[TTFT与TPOT]] 的 TTFT）。非流式要等全完才返，体验差。故流式是 serving 必备。

### 2.4 多实例要负载均衡

单实例吞吐有限，生产多实例部署。前端要 router 把请求分发到各实例——round-robin（最简）、KV-aware/prefix-aware（把相同前缀请求路由到已缓存该前缀的实例，命中 [[Prefix Cache]]，降 prefill）。router 是 server 层职责（或外置 Nginx/Envoy）。

### 2.5 功能扩展要在协议层落地

LoRA 多模型（一个 server serve 多个 LoRA adapter，按请求的 `model` 字段选）、function calling（模型输出工具调用，server 解析成 OpenAI tool_calls 格式）、structured output（JSON schema 约束输出）、logprob 返回、n 采样、best_of——这些都要在 HTTP 层解析参数 + 调引擎对应能力。


## 3. 核心概念详解

### 3.1 整体架构（vLLM V1）

```
┌─────────────── 前端 (FastAPI + uvicorn, asyncio) ───────────────┐
│  HTTP 请求 (OpenAI 协议)                                          │
│     |                                                            │
│  app.add_api_route("/v1/chat/completions", chat_endpoint)         │
│  OpenAIServingChat.create_chat_completion()                       │
│     |- tokenize (messages -> input_ids, 用 chat template 渲染)   │
│     |- 组 SamplingParams (temp/top_p/top_k/stop/max_tokens/...)   │
│     |- engine.add_request() -> 入 AsyncLLM 队列                  │
│  (异步) <- 引擎产出 token/logprob                                │
│     |- detokenize (token -> 文本, 增量)                          │
│     |- 流式: yield SSE chunk (delta) / 非流式: 一次性返          │
└──────────────────────────────────────────────────────────────────┘
                              |  异步队列
┌─────────────── 引擎 (AsyncLLM, asyncio + 线程) ────────────────┐
│  AsyncLLM.request_async_loop()                                   │
│     |- scheduler (continuous batching, chunked prefill)         │
│     |- model runner (forward + sample, 见 [[model runner]])      │
│     |- 产出 RequestOutput (token + logprob + finished)           │
└──────────────────────────────────────────────────────────────────┘
```

两层用**异步队列**解耦：HTTP 层 tokenize 后丢队列就返（不等算完），引擎在 event loop + worker 线程里 continuous batching 算，算出 token 经 detokenizer 协程流回 HTTP 层 yield。HTTP IO（解析/网络）与 GPU forward **不互相阻塞**。

### 3.2 OpenAI 兼容协议（端点）

vLLM 官方文档核实的端点：

| 端点 | 用途 | 任务 |
|---|---|---|
| `/v1/completions` | 文本补全（prompt → 续写） | `--task generate` |
| `/v1/chat/completions` | 对话（messages → 回复） | `--task generate` + chat template |
| `/v1/embeddings` | 向量嵌入 | `--task embed` |
| `/v1/audio/transcriptions` | 语音转写（Whisper） | `--task generate` |
| `/tokenize` `/detokenize` | 分词工具（vLLM 自有） | 任意 |
| `/pooling` | 池化输出（vLLM 自有） | pooling 模型 |
| `/score` | 打分（vLLM 自有） | `--task score` |
| `/rerank` `/v1/rerank` `/v2/rerank` | 重排（Jina/Cohere 兼容） | `--task score` |

`/v1/models` 列模型，`/health` 健康检查，`/version` 版本。

### 3.3 前端 router 解析请求

以 `/v1/chat/completions` 为例，`OpenAIServingChat.create_chat_completion(req)` 流程：
1. **解析 `messages`**：按 `--chat-template`（Jinja2）渲染成单条文本 prompt（`apply_chat_template`）。多模态消息含图像/音频时经多模态 processor 处理（见 [[新模型接入]]）。
2. **tokenize**：文本 → `input_ids`（用引擎 tokenizer，`--tokenizer-mode auto` 用 fast tokenizer）。
3. **组 `SamplingParams`**：从 req 抽 `temperature`/`top_p`/`top_k`/`max_tokens`/`stop`/`n`/`best_of`/`logprobs`/`presence_penalty`/`frequency_penalty`/`seed` 等 + vLLM 扩展（`extra_body` 里的 `guided_json`/`min_p`/`repetition_penalty`/...）。
4. **入队**：`engine.add_request(prompt_token_ids=input_ids, sampling_params=sp, request_id=...)`。
5. **异步等结果**：引擎产出 `RequestOutput`（含生成的 token/logprob/finished），detokenize 成文本，按流式/非流式返。

### 3.4 流式响应（SSE chunked）

流式（`stream=True`）时 server 用 **SSE**（`Content-Type: text/event-stream`）逐 token 推 `data: {chunk}\n\n`。每 chunk 含 `delta`（增量 token 文本）+ `finish_reason`（最后 chunk 为 `stop`/`length`）。格式仿 OpenAI streaming，client 可直接用官方 `client.chat.completions.create(stream=True)` 迭代。

```
流式 (SSE):
  data: {"choices":[{"delta":{"content":"H"},"index":0}]}
  data: {"choices":[{"delta":{"content":"i"},"index":0}]}
  ...
  data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}
  data: [DONE]

非流式:
  一次返 {"choices":[{"message":{"content":"Hello"},...}]}
```

增量 detokenize 是关键——不能等全完才 detokenize（那非流式），要 token 来一个 detokenize 一个（处理 BPE 跨字节边界，缓存前缀 bytes）。vLLM V1 的 `AsyncLLM` 配 detokenizer 协程增量吐文本。

### 3.5 负载均衡（多实例 router）

多实例部署时，前端 router（Nginx/Envoy/自研/或 vLLM 内置的 DP）分发请求：

| 策略 | 逻辑 | 优点 | 缺点 |
|---|---|---|---|
| **round-robin** | 轮询 | 简单 | 不考虑 KV 缓存状态，prefix 命中低 |
| **least-load** | 路由到当前负载最低的 | 均衡 | 需各实例报负载 |
| **prefix-aware / KV-aware** | 把相同前缀请求路由到**已缓存该前缀**的实例 | 命中 [[Prefix Cache]]，降 prefill，低 TTFT | 路由复杂，需查各实例缓存指纹 |
| **affinity** | 同会话粘同实例 | 上下文连续 | 不均衡风险 |

KV-aware routing 是 LLM serving 特有优化——把相同 system prompt / few-shot 前缀的请求聚到同实例复用 KV。vLLM 的 `--kv-transfer-config` 也支持跨实例 KV 迁移（PD 分离场景，见 [[PD分离]]）。生产常外置 router（Envoy + 自定义 Lua/Go 逻辑）做 prefix hashing 分发。

### 3.6 功能扩展

| 功能 | 实现 | 说明 |
|---|---|---|
| **LoRA 多模型** | `--enable-lora --lora-modules name=path`；req 的 `model` 字段选 LoRA | 一个 server serve 多 LoRA adapter，按 `model` 名动态加载 |
| **function calling / tool use** | `--enable-auto-tool-choice --tool-call-parser hermes/llama3_json/...` | 模型输出工具调用文本，server 按 parser 解析成 OpenAI `tool_calls` 格式 |
| **structured output** | `--guided-decoding-backend xgrammar/outlines/...`；req `guided_json`/`response_format` | 用 [[structured output]] 的 FSM/logit mask 约束输出 JSON schema |
| **reasoning content** | `--enable-reasoning --reasoning-parser deepseek_r1/granite` | 解析模型的 reasoning content（如 R1 的 `<think>`）成 `reasoning_content` 字段 |
| **logprob 返回** | req `logprobs=true`（+ `top_logprobs`） | 返回选中 token 的 logprob 及 top-k 候选 logprob |
| **n 采样** | req `n>1` | 一次生成 n 个候选（引擎内同 batch） |
| **best_of** | req `best_of` | 采样 best_of 个取 logprob 最高者返 |
| **prompt logprob** | req `prompt_logprobs` | 返回 prompt 各 token 的 logprob |

扩展参数（OpenAI 没有的）走 OpenAI client 的 `extra_body`（如 `extra_body={"top_k":50}`）或直接塞 JSON body。

### 3.7 与 OpenAI 的兼容边界

vLLM **尽力兼容**但非完全对等：

| OpenAI 参数 | vLLM 支持 |
|---|---|
| `temperature`/`top_p`/`max_tokens`/`stop`/`n`/`stream`/`logprobs`/`seed` | ✅ |
| `top_k`/`min_p`/`repetition_penalty`/`min_tokens` | ✅（OpenAI 无，走 `extra_body`） |
| `response_format`（json_object/json_schema） | ✅（guided decoding） |
| `tools`/`tool_choice`（function calling） | ✅（需 `--enable-auto-tool-choice` + parser） |
| `suffix`（completions 续写后缀） | ❌（不支持） |
| `parallel_tool_calls`/`user` | ❌（忽略） |
| `n`>1 与 `best_of` | ✅（引擎层多采样） |

默认 server 会读 HF 仓库的 `generation_config.json` 覆盖采样默认值（模型作者推荐的 temp/top_p 等）。`--generation-config vllm` 关掉用 vLLM 默认。

### 3.8 SGLang 的 HTTP server

SGLang 的 `sglang/srt/entrypoints/http_server.py` 同样是 FastAPI + uvicorn，OpenAI 兼容协议。架构组件：
- **`TokenizerManager`**：接 HTTP 请求，tokenize，发 token 给 `Scheduler`（引擎）。
- **`Scheduler`**：引擎核心，continuous batching + RadixAttention（前缀树 prefix cache，见 [[Prefix Cache]]），调 `ModelRunner` forward。
- **`DetokenizerManager`**：收引擎产出的 token，增量 detokenize，经 channel 流回 HTTP 层 SSE。
- 多实例 router：SGLang 提供 router 脚本（round-robin / prefix-aware）。

与 vLLM 同构（HTTP 层 tokenize 入队 + 引擎 continuous batching 算 + detokenize 流回），命名与内部组件略异。SGLang 的特色是 RadixAttention 前缀树（自动跨请求复用前缀 KV）+ PD 分离（见 [[PD分离]]）。

### 3.9 前端多进程（vLLM V1）

vLLM V1 默认**前端与引擎分进程**（前端 API server 一进程，引擎一进程，异步队列通信）。`--disable-frontend-multiprocessing` 让前端与引擎同进程（省一进程，调试方便，但 HTTP IO 与引擎共享 event loop 可能互相影响）。生产用默认分进程。

```
默认 (分进程):
  进程 A: API server (FastAPI + uvicorn, tokenize/detokenize)
     |  (异步队列 / ZMQ)
  进程 B: 引擎 (AsyncLLM: scheduler + model runner, GPU forward)

--disable-frontend-multiprocessing:
  单进程: API server + 引擎同 event loop (省进程, 但共享 GIL/event loop)
```


## 4. 数学原理 / 公式

### 4.1 吞吐与并发模型

设单实例 max concurrency `C`（同时 in-flight 请求数，受 `--max-num-seqs` 与 KV 上限）、单请求 decode 速率 `r_d` token/s（受 [[TTFT与TPOT]] 的 TPOT）、实例数 `K`。集群总吞吐：

$$
\text{throughput} = K \cdot \min(C, \sum_{\text{req}} r_d) \approx K \cdot \frac{C}{\text{TPOT}} \cdot \text{batch util}
$$

HTTP 层不直接影响吞吐（只入队），但 router 的 prefix-aware routing 提升 batch util（同前缀请求聚同实例，复用 KV，等效降 prefill 算力）。

### 4.2 流式 TTFT

流式首 chunk 在引擎产出首 token 后即可发（detokenize 首 token → SSE chunk）。TTFT ≈ prefill 时间 + 首 token decode + detokenize + 网络。非流式 TTFT ≈ 全部生成完才返（等于总延迟）。流式降的是**用户感知的首字延迟**（不是总延迟）。

### 4.3 增量 detokenize 的字节边界

BPE 分词可能跨字节边界（某 token 的文本是 `` 半字符，需后续 token 补全）。增量 detokenize 缓存未完成 bytes 前缀，待后续 token 来补全再 emit。公式上每步 detokenizer 维护 `pending_bytes`，新 token 的 bytes append 到 pending，emit 已确认的完整字符，留 incomplete 尾部。这避免流式输出乱码。

### 4.4 n 采样与 best_of 的算力

`n` 个候选同 prompt 同 batch 跑：prefill 共享（1 次），decode 各走各（n 路）。算力 ≈ prefill + n × decode。`best_of`（采 best_of 个取 logprob 最高者返）：实际生成 best_of 个再选，算力 = best_of × decode（除非用 speculative 优化）。HTTP 层只声明参数，引擎层做采样。


## 5. 代码示例（可选）

### 5.1 启动 server（vLLM）

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --api-key token-abc123 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser hermes \  # function calling
  --guided-decoding-backend xgrammar \                   # structured output
  --enable-lora --lora-modules my-lora=/path/to/lora \   # LoRA 多模型
  --served-model-name llama3 my-llama \                  # API model 字段别名
  --enable-chunked-prefill                                # 长 prompt 切块 (见 [[chunked prefill]])
```

### 5.2 流式调用（OpenAI client）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

stream = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "写一首关于秋天的诗"}],
    stream=True,            # 流式 SSE
    extra_body={"top_k": 50, "min_p": 0.05},   # vLLM 扩展参数走 extra_body
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)   # 逐 token 打印 (增量)
```

### 5.3 structured output（JSON schema）

```python
client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "提取: 张三, 25岁, 北京"}],
    extra_body={
        "guided_json": {                # 约束输出符合此 schema
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["name", "age", "city"]
        }
    },
)
```

### 5.4 function calling

```python
client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "北京今天天气?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }],
    # 模型输出 -> --tool-call-parser 解析成 choices[0].message.tool_calls
)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[tokenization]]（tokenize/detokenize）、[[采样策略]]（SamplingParams）、[[continuous batching的调度实现]]（引擎层合 batch）、[[model runner]]（forward）、[[chunked prefill]]（长 prompt 切块影响 TTFT）。
- **下游（应用）**: 生产 LLM serving、OpenAI 兼容部署、流式聊天、多 LoRA 服务、function calling agent、structured output、多实例负载均衡。
- **对比 / 易混**:
  - **API server vs [[model runner]]**：前者 HTTP 协议层（parse/tokenize/入队/detokenize/SSE），后者 GPU 执行层（forward/sample）。异步队列解耦。
  - **API server vs [[continuous batching的调度实现]]**：前者不入 batch 决策，只把请求丢引擎队列；batch 组成与进出由引擎 scheduler 定。HTTP 层不感知 batch。
  - **流式 vs 非流式**：流式 SSE 逐 token 推（降用户感知首字延迟），非流式等全完才返。两者总延迟同，体验差异。
  - **vLLM vs SGLang server**：同构（FastAPI+uvicorn+OpenAI 兼容+tokenize 入队+引擎 continuous batching+detokenize 流回）。命名异（vLLM `OpenAIServingChat`/`AsyncLLM`；SGLang `TokenizerManager`/`Scheduler`/`DetokenizerManager`）。SGLang 特色 RadixAttention 前缀树。
  - **prefix-aware routing vs [[Prefix Cache]]**：前者是 router 层把同前缀请求聚同实例（提升后者命中率）；后者是引擎内 KV 复用机制。一外一内，协同。


## 7. 常见误区与易错点

> [!warning] 误区 1：HTTP 层做 batch 合并
> 不。HTTP 层只把单个请求入队，**continuous batching 在引擎内合 batch**。HTTP 层不感知 batch 组成。若在 HTTP 层合 batch 会破坏引擎调度灵活性。

> [!warning] 误区 2：流式降低总延迟
> 流式降的是**用户感知首字延迟**（TTFT），不是总延迟。总生成时间同（都要算完所有 token）。非流式要等全完才返，故首字延迟=总延迟。

> [!warning] 误区 3：API server 阻塞 GPU
> 不会（设计上）。tokenize/detokenize 在前端 asyncio 协程，GPU forward 在引擎 worker 线程/进程，异步队列解耦。HTTP IO 不卡 GPU。但若前端做重 CPU 活（如复杂模板渲染）且不分进程，可能挤占——故默认分进程。

> [!warning] 误区 4：完全兼容 OpenAI
> 尽力兼容但有边界：`suffix` 不支持、`parallel_tool_calls`/`user` 忽略、部分参数语义略异。用 OpenAI client 直连要测，别假设全对等。

> [!warning] 误区 5：LoRA 多模型 = 多模型权重
> LoRA adapter 是**增量**权重（小），base 模型权重一份。server 按 `model` 字段选 adapter 动态加。不是每 LoRA 一份完整模型（那费显存）。

> [!warning] 误区 6：function calling 模型自动支持
> 需 `--enable-auto-tool-choice --tool-call-parser <parser>` 且模型要按某格式输出工具调用。parser 把模型文本解析成 OpenAI `tool_calls`。不配则不解析。

> [!warning] 误区 7：round-robin 足够
> round-robin 不考虑 KV 状态，prefix 命中低。LLM serving 的 prefill 贵，prefix-aware routing（同前缀聚同实例）能显著降 prefill、提吞吐。生产应上 prefix-aware。

> [!warning] 误区 8：generation_config 不重要
> 默认 server 读 HF 的 `generation_config.json` 覆盖采样默认（如推荐 temp=0.6）。不读可能与模型作者预期不符（输出质量差）。`--generation-config vllm` 用引擎默认。

> [!warning] 误区 9：n 采样很便宜
> n 个候选同 prompt 走，prefill 共享但 decode 各走 n 路，算力 ≈ n × decode。n 大时贵。best_of 同理（甚至更贵，除非投机优化）。


## 8. 延伸细节

### 8.1 asyncio event loop 与 worker 线程

vLLM V1 的 `AsyncLLM` 用 asyncio event loop 管请求生命周期，但 GPU forward 是阻塞的 CPU/GPU 调用，不能在 event loop 直接跑（会阻塞所有协程）。故 forward 放**worker 线程**（或引擎进程），event loop 只做调度协调。这样 HTTP IO（协程）与 GPU forward（线程/进程）真正并行。SGLang 用类似架构（asyncio + channel）。

### 8.2 batch 协议层合并的边界

"batch 在引擎层合并"——多个 HTTP 请求入队后，引擎 scheduler 在一个 iteration 里把它们组进一个 batch（continuous batching）。HTTP 层不显式 batch。但若 HTTP 层做请求聚合（如某些网关），需谨慎——可能破坏 per-request 的 streaming 与调度优先级。

### 8.3 中间件与可观测

vLLM `--middleware` 支持加 ASGI 中间件（日志/限流/tracing）。`--otlp-traces-endpoint` 发 OpenTelemetry trace。`--collect-detailed-traces` 细化 model/worker 级 trace。`/metrics`（Prometheus）暴露吞吐/延迟/缓存命中指标，接 [[goodput]] 监控。高 QPS 时 `--enable-request-id-headers` 有性能损耗，官方建议放 router 层（Istio）做。

### 8.4 内容来源

端点与参数整理自 vLLM 官方文档《OpenAI-Compatible Server》（核实 `/v1/completions`/`/v1/chat/completions`/`/v1/embeddings`/`/v1/audio/transcriptions`/`/tokenize`/`/detokenize`/`/pooling`/`/score`/`/rerank`/`/v1/rerank`/`/v2/rerank` 端点、`--api-key`/`--served-model-name`/`--lora-modules`/`--enable-auto-tool-choice`/`--tool-call-parser`/`--guided-decoding-backend`/`--reasoning-parser`/`--enable-reasoning`/`--disable-frontend-multiprocessing`/`--middleware`/`--generation-config`/`--kv-transfer-config` 等参数、`extra_body` 用法、generation_config 行为）、vLLM 源码 `vllm/entrypoints/openai/`（`api_server.py`/`serving_chat.py` 的 `OpenAIServingChat`/`serving_completions.py` 的 `OpenAIServingCompletions`/`serving_embedding.py`）、`vllm/v1/engine/async_llm.py`（`AsyncLLM`）、SGLang 源码 `sglang/srt/entrypoints/http_server.py` 与 `TokenizerManager`/`DetokenizerManager`/`Scheduler`。类名/参数若与最新版本有出入以仓库 main 与官方文档为准（待核实最新 V1 命名）。

---

相关: [[tokenization]] | [[采样策略]] | [[continuous batching的调度实现]] | [[model runner]] | [[worker与executor]] | [[chunked prefill]] | [[Prefix Cache]] | [[PD分离]] | [[TTFT与TPOT]] | [[goodput]] | [[structured output]] | [[TP PP DP EP推理]] | [[16-推理引擎源码]]
