# prefix caching

> **所属章节**: [[推理优化]]
> **所属模块**: [[08-推理系统]]
> **别名**: prefix caching / 前缀缓存 / Automatic Prefix Caching (APC) / prompt caching
> **难度**: 中（需懂 [[KV cache management]] + PagedAttention）


## 1. 一句话定义

**prefix caching（前缀缓存）** 是 LLM 推理引擎复用**多个请求公共前缀**的 [[KV cache management]] 的技术：当多个请求共享相同的开头（system prompt、few-shot 示例、文档前缀等），只对第一个请求算这部分前缀的 KV 并存入缓存，后续请求直接复用这部分 KV、只 prefill **各自不同的后缀**。vLLM 称 Automatic Prefix Caching (APC)，SGLang 用 radix tree（基数树）组织共享前缀，OpenAI API 称 prompt caching。
> [!note] 解答：这个技术现在是不是已经很容易实现了？直接调 API 就行？训推框架开发人员感知吗？
> 三个问题分别答，结论先行：**API 用户 = 几乎零成本白嫖；自建推理服务 = 一行参数即开；训推/推理框架开发人员 = 必须深度感知，是核心 KPI**。下面是 2026 年 7 月联网核实的情况。
>
> ### 一、"容易实现吗"——是的，已从研究特性沉淀为工业默认能力
> - **vLLM**：APC（Automatic Prefix Caching）基于 PagedAttention block 哈希，一行 `--enable-prefix-caching` 开关即启用，命中按 block 粒度复用物理 KV block（来源：docs.vllm.ai/en/latest/features/automatic_prefix_caching）。注意它要求**逐 token 完全相同**且按 block 边界对齐——共享前缀 3047 token、block_size=16 时，末尾 13 token 每个 block 都要重算，这是 PagedAttention 的固有粒度损失（来源：turion.ai 2026 对比）。
> - **SGLang**：RadixAttention 是**默认且零配置**的——用 radix tree 在 token 级自动发现任意公共前缀，没有 block 对齐损失，"系统从真实流量里自己学要缓存什么"（来源：inference.net 2026 完整指南；SGLang v0.5.8 2026-01 发布，已在 xAI/NVIDIA/AMD/LinkedIn 等 40 万+ GPU 上生产）。
> - **TensorRT-LLM**：NVIDIA 自家推理栈也内置 prefix caching，主打原始吞吐天花板（来源：turion.ai 2026）。
> - **学术界/工业共识**：2026 年 vLLM 与 SGLang 是"唯二重要的生产推理引擎"，prefix caching 已不是卖点而是入场券（来源：turion.ai "vLLM vs SGLang: Inference Engine Comparison 2026"、spheron 2026-06、techsy 2026-03 H100 benchmark）。Triton+TGI 已退场或被吸收。
>
> 所以"实现"本身已不构成门槛——开个开关就有。难点从"能不能做"转移到"**命中率够不够高**"和"**调度够不够聪明**"，这才是 2026 年的工程焦点。
>
> ### 二、"直接调 API 就行"——分两种用户
> **A. 闭源 API 用户（OpenAI / Anthropic / Gemini / DeepSeek / xAI）**：确实**几乎零代码改动**就能享受，但各家机制不同（2026-07 数据，来源：leanlm.ai、swfte.com、bytecosts.com、aicost.ai）：
>
> | Provider | 激活方式 | 写入费 | 读取折扣 | 最低 token | TTL | 备注 |
> |---|---|---|---|---|---|---|
> | **OpenAI** (GPT-5.x) | **全自动**，无 knob | 无 | **90% off**（GPT-5.5: $5→$0.5/M） | 1024 tok | ~5–10min，gpt-5.5+ 默认 24h 免费扩展 | 最省心，看 `prompt_tokens_details.cached_tokens` 字段 |
> | **Anthropic** (Claude) | 显式 `cache_control` 或 2026 新增自动模式 | **1.25× (5min) / 2× (1hr)** | **90% off** (0.1× input) | 512–4096 (model-dependent) | 5min 默认/1hr 加价/读刷新 | 写入有溢价，但一次写≈1.3 次读即回本 |
> | **Gemini** (2.5+) | 隐式自动 + 显式 `cached_content` API | 无写费，但有**存储租金** | **90% off** (uniform 10% of input) | 2048+ (2.5 系列) | 隐式不可控/显式默认 1h 可配 | 闲置也计费，低频流量慎用 |
> | **DeepSeek** (V4 Flash) | 自动磁盘级缓存 | 无 | **~98% off**（$0.14→$0.0028/M，业界最狠） | — | — | 把 hit/miss 直接分开计价 |
> | **xAI (Grok)** | 自动前缀缓存 | 无 | ~75% off | — | — | 无 API knob |
>
> **关键提醒**（这些是"调 API 也得懂"的点，否则钱白花）：
> 1. **必须把静态内容放最前、变量放最后**——所有 provider 都是 exact-prefix 匹配，一个 token 不同就 miss。在 cached 区域里塞时间戳/随机 ID 是最常见的"缓存失效自残"。
> 2. **低于最低 token 阈值静默不缓存**——Anthropic 低于 1024 时 `cache_creation_input_tokens` 返回 0、按原价计费，**不报错**，是新手最大坑。
> 3. **Gemini 显式缓存有存储租金**（$1/MTok/hr Flash，$4.5 Pro），闲置过夜可能比省的还贵。
> 4. **可与 Batch API 折扣叠加**——Anthropic 明确声明叠加：Sonnet 4.6 batch + cache read = $3×0.5×0.1 = **$0.15/M，95% off**。
> 5. 健康生产应用的命中率应到 **60–90%**，低于 30% 基本都是 prompt drift。
>
> **B. 自建推理服务用户（vLLM/SGLang 自部署）**：不是"调 API"，而是**自己当 provider**——开 `--enable-prefix-caching`（vLLM）或默认即有（SGLang），然后**自己设计 prompt 结构**让前缀命中率最大化，并监控命中率调 LRU/cache 显存预算。本质和你调闭源 API 面对的优化问题一样，只是 KV 在你显存里、省钱=省你自己的 GPU。
>
> ### 三、"训推框架开发人员感知吗"——必须深度感知，分三类角色
> 1. **纯推理引擎开发（vLLM/SGLang/TRT-LLM）**：这是**核心 KPI 与差异化卖点**。SGLang 靠 RadixAttention 在 prefix-heavy 场景比 vLLM 快 3–5× prefill、29% 吞吐，是它能在 2026 抢下 xAI/NVIDIA/LinkedIn 40 万 GPU 的根因；vLLM 的 APC 也是发布说明里的头牌特性。2026 年 RadixArk 从 SGLang 分拆独立融到 $400M，机构资本押注的就是"前缀缓存这块"。开发人员不光要实现，还要调 block_size、LRU、copy-on-write、与 chunked prefill/continuous batching 的调度协同。
> 2. **训练框架开发（Megatron-LM/DeepSpeed/FSDP/verl/OpenRLHF）**：RLHF/PPO 里 rollout worker 跑的就是推理栈，rollout 采样的 system prompt + prompt template 固定，prefix caching 直接拉高 [[sampling throughput]]，是 RLHF 训练加速的关键一环。verl/OpenRLHF 默认把 vLLM/SGLang 当 rollout 引擎，prefix caching 自动继承。开发人员要感知两件事：① **权重更新必须失效缓存**（[[weight sync mechanism]] 推新 $\theta$，旧 KV 与新 $\theta$ 不匹配，[[stale policy problem]] 在推理侧投影），生产系统要给缓存打版本号切换即清空；② 在 [[训推不一致]] TIM 治理里，KV 量化方案变化会让缓存与训练端 logp 不一致，需对齐。
> 3. **训推一体化/在线学习系统**：权重持续在变，prefix caching 的"复用"与"权重新鲜度"直接冲突——要么版本化缓存按 $\theta$ 隔离，要么干脆在热路径禁用、只在冷用户/冷版本上开。这是 [[policy deployment (PD)]] 服务降本与一致性的核心权衡点，开发人员必须显式处理。
>
> ### 四、一句话总结
> - **应用层调 API**：技术已"白嫖级"成熟，OpenAI 全自动无脑省 90%，Anthropic/Gemini 加一行 `cache_control`/建 cache object；但 prompt 结构设计（静态前→变量后）和命中率监控仍是用户侧责任。
> - **自建推理**：vLLM 开个 flag、SGLang 默认即有，但命中率、调度协同、版本失效是工程重点。
> - **框架开发人员**：推理引擎把它当核心卖点和性能护城河（SGLang RadixAttention 是 2026 头牌）；训练/RL 框架把它当 rollout 加速 + weight sync 失效治理 + TIM 对齐的必经关卡。**"容易实现"是对使用者的，对开发者它依然是性能与一致性的硬仗**。
>
> **来源**（2026-07-12 联网核实）：
> - [vLLM Automatic Prefix Caching 官方文档](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
> - [vLLM vs SGLang: Inference Engine Comparison 2026 — TURION.AI](https://turion.ai/blog/vllm-vs-sglang-inference-comparison-2026/)
> - [SGLang Complete Guide — Inference.net 2026-01](https://inference.net/content/sglang-complete-guide/)
> - [Prompt Caching in 2026: OpenAI vs Claude vs Gemini Pricing — leanlm.ai](https://leanlm.ai/blog/prompt-caching)
> - [Prompt caching explained 2026 — llmtest.io](https://llmtest.io/blog/prompt-caching-explained)
> - [How the Major LLM Providers Price Prompt Caching — ByteCosts](https://bytecosts.com/blog/how-llm-providers-price-prompt-caching/)
> - [vLLM vs SGLang 2026: H100 Benchmarks — TECHSY](https://techsy.io/en/blog/vllm-vs-sglang) / [Spheron 2026-06](https://www.spheron.network/blog/vllm-vs-sglang-2026/)
> [!note] 三句话定位
> - **是什么**：把"公共前缀的 KV"当成可跨请求复用的资源，避免重复 prefill。
> - **为什么**：system prompt 动辄 1K–8K token，每个请求都重算 prefill 是巨大浪费。
> - **怎么做**：按 token 序列哈希/前缀树定位已缓存的公共前缀 KV，复用其 block，只 prefill 增量后缀。


> [!note] 解答：项目一打开就把显存打爆——和 prefix caching 什么关系？
> 你这个体感很真实，但**多半不是 prefix caching 本身把显存打爆的**，而是 KV cache 显存预算 + 模型权重 + CUDA context 三件事叠在一起，prefix caching 只是让"已分配的 KV block 被多请求共享"，复用是指针级、零拷贝，几乎不额外占显存。下面把"启动即 OOM"拆开讲透。
>
> ### 一、先厘清：prefix caching 占不占额外显存？
> **几乎不占**。它的机制是让多个请求的 block table 前几项**指向同一物理 block**（见 §3.5），物理 block 还是住在 KV cache 的显存预算里，只是引用计数 +1。所以 prefix caching 本身**不是显存爆的元凶**——它反而通过共享**降低**单请求等效 KV 占用（同样显存能塞更多并发）。
>
> 真正吃显存的是下面这几块，**"一打开就爆"基本都中其中一条**：
>
> ### 二、显存爆的四大真凶（按 vLLM/SGLang 启动场景排）
>
> | # | 真凶 | 机制 | 怎么爆的 | 怎么治 |
> |---|---|---|---|---|
> | 1 | **KV cache 预分配过头** | vLLM `--gpu-memory-utilization`（默认 **0.9**）/SGLang `--mem-fraction-static`（默认 **0.9**）会在启动时把 90% 显存预算全划给 KV cache | 留给"权重+activation+临时张量+CUDA context"只剩 10%，稍微一跑就 OOM | 降到 **0.85~0.88**，给 PyTorch CUDA context（~1–2GB）和激活/临时张量留余量 |
> | 2 | **模型权重本身就大** | 权重显存 = 参数量 × bytes/dtype。70B bf16 = 140GB，单 80GB 卡装不下，必须 TP | 想单卡跑 70B → 直接 OOM；或 TP 切分后 KV cache 预算又挤压 | 大模型上 TP/PP，别想单卡；或上量化（[[FP8量化方案]]/AWQ/INT4）把权重压到 1/2~1/4 |
> | 3 | **多实例/多进程共存抢显存** | 同一张卡上既跑训练又跑 vLLM，或起多个 vLLM 进程 | 每个进程都按 `gpu_memory_utilization` 抢，叠加爆卡 | 用 `CUDA_VISIBLE_DEVICES` 隔离，一进程一卡；或显式设每个进程的 util 上限（如 0.4） |
> | 4 | **CUDA context + 临时张量被忽略** | PyTorch 初始化、cuBLAS/cuDNN workspace、forward 的 activation 都要显存，预算时没算进去 | KV cache 预算卡死 0.9，跑起来 activation 没地方放 → 严重 OOM crash | 留 10–15% headroom；监控 `nvidia-smi` 看实际占用 vs 预留 |
>
> ### 三、为什么"一打开"就爆（而不是跑一会才爆）
> 因为 vLLM/SGLang 启动时**一次性预分配 KV cache pool**（PagedAttention 的物理 block 池），按 `mem-fraction-static` 把显存吃满到预算上限。如果你设 0.9，启动瞬间 90% 显存就没了，紧接着 forward 一步要 activation → 直接 OOM。这是"一打开就爆"的典型时序。
>
> ### 四、prefix caching 在这里的间接作用
> 1. **命中率低时冷前缀堆积**：prefix cache 按 LRU 驱逐，但如果你流量很分散（每个请求前缀都不同），缓存里堆了一堆冷前缀 block 占着显存，挤压新请求可用 block → 等效显存变小。治法：监控命中率，低于 30% 就关掉 prefix caching 或缩小 cache 预算。
> 2. **版本未失效的旧权重 KV 残留**：[[weight sync mechanism]] 推新 $\theta$ 后如果没清缓存，旧版本 KV 还占着显存，新版又来分配 → 叠加爆。治法：权重切换时强制清 prefix cache。
> 3. **block_size 太小**：block 越小，哈希表 + 元数据开销越大，碎片多，等效可用显存缩水。vLLM 默认 16 是权衡点，别瞎调小。
>
> ### 五、一份"启动不爆"的 vLLM 配方
> ```bash
> # 关键参数（按 80GB A100/H100、70B 模型 TP=4 举例）
> python -m vllm.entrypoints.openai.api_server \
>   --model meta-llama/Llama-3-70B-Instruct \
>   --tensor-parallel-size 4 \                  # 大模型必上 TP
>   --gpu-memory-utilization 0.88 \             # 别顶 0.9，留 12% 给 context+activation
>   --max-model-len 8192 \                      # 限最大上下文，控 KV cache 上限
>   --enable-prefix-caching \                   # 开 prefix caching 省前缀 prefill
>   --swap-space 4                              # KV cache 换出 CPU 的 GiB，OOM 时兜底
> # 监控：nvidia-smi -l 1 看显存曲线；vLLM /metrics 看 cache_hit_rate、num_preemption
> ```
>
> **一句话**：你项目"一打开就爆"的真凶是 `gpu_memory_utilization` 顶太高 + 权重/激活没留余量，prefix caching 反而是帮你省显存的那个。把 util 降到 0.85–0.88、大模型上 TP、监控命中率，三条做完基本就不爆了。详见 [[KV cache management]] §1 显存公式与预算推导。

## 2. 为什么需要它（动机与背景）

### 2.1 公共前缀的重复 prefill 浪费

实际 LLM 服务中，请求常带**相同前缀**：

- **system prompt**：所有请求共享（如 "You are a helpful assistant..." + 工具说明），1K–4K token。
- **few-shot 示例**：同一任务的多个请求共享示例段。
- **RAG 上下文**：同一文档的不同提问共享检索到的文档段。
- **多轮对话**：历史轮次是后续请求的前缀。

每次请求都从 token 0 开始 prefill 整个前缀，算这段的 $K/V$。若前缀长 4K token，每个请求为此消耗 4K token 的 prefill 算力——而这段 KV 与其他请求**完全相同**。

> [!tip] 量化浪费
> Llama-7B prefill 4K token 前缀 ≈ $2 \times 7\text{B} \times 4096 \approx 5.7 \times 10^{13}$ FLOPs/请求。1000 req/s 的服务，每秒 $5.7 \times 10^{16}$ FLOPs 全在重算相同前缀——浪费。prefix caching 把它降到几乎 0（仅复用 + 小段后缀 prefill）。

### 2.2 没有它会怎样

- TTFT（首 token 延迟）被长前缀 prefill 拖高。
- prefill 算力被重复消耗，GPU 利用在"重算"而非"新请求"。
- 长 context 应用（RAG、agent）成本居高不下。

prefix caching 把"每个请求独立 prefill 全段"变成"首次 prefill + 后续复用"，是降本降延迟的关键。


## 3. 核心概念详解

### 3.1 前缀匹配：block 粒度 vs token 粒度

KV cache 在 PagedAttention 下按 **block**（如 16 token/block）组织。prefix caching 自然也按 block 粒度匹配：

- 把请求的 token 序列按 block 切分。
- 逐 block 计算哈希，查缓存是否已有该 block 的 KV。
- 命中的 block 直接复用物理 block（block table 指向同一物理 block），不重算。
- 从首个未命中 block 开始 prefill 增量部分。

> [!note] 为什么按 block 而非整段哈希
> 整段哈希要求前缀**完全相同**才命中，灵活性差。按 block 哈希可命中"部分公共前缀"——如两个请求前 10 个 block 相同、第 11 个 block 才分歧，仍能复用前 10 个 block 的 KV。SGLang 的 radix tree 进一步把这种"前缀共享"组织成树，自动发现任意长度的公共前缀。

### 3.2 radix tree（基数树，SGLang）

SGLang 用 **radix tree**（一种压缩前缀树）管理 KV cache 的共享：

- 每个节点代表一段 token 子序列及其对应 KV block。
- 从根到某节点的路径 = 一个请求的前缀。
- 多个请求共享根附近的节点（公共前缀），分歧处才分叉。
- 引用计数管理：某节点无引用时可回收。

```
root
 └─ [system_prompt 4K tok] (KV block #1..#256, ref=5)
     ├─ [few_shot_A] (KV #257..#260, ref=2)
     │   └─ 请求1 后缀
     └─ [few_shot_B] (KV #257..#259, ref=3)
         └─ 请求2 后缀
```

5 个请求共享 system_prompt 节点（ref=5），其 KV 只算一次。比朴素 block 哈希更高效（自动结构化发现共享）。

### 3.3 vLLM APC（Automatic Prefix Caching）

vLLM 的 APC 基于 PagedAttention 的 block 共享：

- 每个 block 计算哈希（基于 token 内容 + 位置）。
- 请求 prefill 时逐 block 查哈希表，命中则复用物理 block。
- **copy-on-write**：若某 block 需被修改（罕见，KV 不可变），才复制。
- 不命中的 block 正常 prefill 并写入缓存供后续复用。

### 3.4 版本与失效

KV cache 对应**特定模型权重版本 $\theta$**。若权重更新（在线学习、[[weight sync mechanism]] 推新 $\theta$），旧 KV 与新 $\theta$ 不匹配，前缀缓存**必须失效**——否则用旧 KV 算出的 attention 与新模型不一致（[[stale policy problem]] 在推理侧的投影）。所以缓存需带版本号，权重切换时清空或标记失效。

### 3.5 与 PagedAttention 的关系

prefix caching 几乎必然基于 PagedAttention：

- 朴素连续 KV cache 下，复用前缀要拷贝整段 KV（昂贵）。
- PagedAttention 下，复用 = 让多个请求的 block table 前几项指向**同一物理 block**（零拷贝，仅指针）。

所以 vLLM 的 APC 与 PagedAttention 是一体的。


## 4. 数学原理 / 公式

### 4.1 节省的 prefill 计算量

设请求前缀长 $P$ token，后缀长 $S$ token（$P \gg S$ 常见）。模型参数量 $N$，prefill 单 token 计算量 $\approx 2N$ FLOPs（前向）。

**无 prefix caching**：每请求 prefill $P + S$ token：

$$
C_{\text{no}} = (P + S) \cdot 2N
$$

**有 prefix caching**（前缀命中）：仅 prefill 后缀 $S$ token（前缀复用零计算）：

$$
C_{\text{with}} = S \cdot 2N
$$

**节省比**：

$$
\text{节省} = \frac{C_{\text{no}} - C_{\text{with}}}{C_{\text{no}}} = \frac{P}{P+S}
$$

> [!note] 推导要点
> 节省比例 = 前缀占比 $P/(P+S)$。前缀越长（相对总长），节省越多。RAG/agent 场景 $P \gg S$（如 $P=8000, S=100$），节省 $\approx 98.8\%$——这就是 prefix caching 在长 context 应用中收益巨大的根因。

### 4.2 数值实例

RAG 服务，文档前缀 $P=8000$ token，提问后缀 $S=100$ token，Llama-7B（$N=7\text{B}$）。

- 无缓存：$(8000+100) \times 2 \times 7\text{B} = 1.13 \times 10^{14}$ FLOPs/请求。
- 有缓存：$100 \times 2 \times 7\text{B} = 1.4 \times 10^{12}$ FLOPs/请求。
- 节省 $\approx 98.8\%$，TTFT 从 ~80ms 降到 ~1ms（量级）。

### 4.3 缓存命中率与 LRU

prefix cache 容量有限（显存预算）。命中率 $h$ 取决于请求分布与缓存大小。有效吞吐：

$$
\text{TPS}_{\text{eff}} \approx \frac{1}{(1-h) \cdot C_{\text{no}}/\text{throughput}_{\text{prefill}}}
$$

$h \to 1$ 时趋近"只 prefill 后缀"的极高吞吐。LRU 驱逐冷前缀以维持命中率。


## 5. 代码示例（可选

### 5.1 简化 prefix cache（block 哈希 + 最长前缀匹配）

```python
import hashlib

class PrefixCache:
    """简化 prefix caching: 按 token 块哈希, 复用最长公共前缀的 KV."""
    def __init__(self, block_size=4):
        self.block_size = block_size
        self.store = {}        # prefix_hash -> kv (示意, 实际是物理 block)
    def _hash(self, tokens):
        return hashlib.md5(str(tokens).encode()).hexdigest()[:8]
    def get_or_compute(self, tokens, compute_kv):
        """返回 (kv_list, reused_blocks). 复用最长已缓存前缀."""
        # 按 block 切分
        blocks = [tokens[i:i+self.block_size] for i in range(0, len(tokens), self.block_size)]
        reused = 0
        kv_list = []
        # 找最长连续命中前缀
        for blk in blocks:
            h = self._hash(blk)
            if h in self.store:
                kv_list.append(self.store[h]); reused += 1
            else:
                break   # 首个未命中即停 (前缀必须连续)
        # 剩余 block 需 compute, 并写入缓存
        for blk in blocks[reused:]:
            kv = compute_kv(blk)
            self.store[self._hash(blk)] = kv
            kv_list.append(kv)
        return kv_list, reused * self.block_size

# —— 演示: 三请求共享 system prompt 前缀 ——
cache = PrefixCache(block_size=4)
compute_kv = lambda blk: f"KV({blk})"   # 假装算 KV

sys_prompt = [10,11,12,13, 20,21,22,23]   # 8 token = 2 block
r1 = sys_prompt + [100,101,102,103]       # 前缀 + 后缀A
r2 = sys_prompt + [200,201,202,203]       # 前缀 + 后缀B
r3 = sys_prompt + [300,301]               # 前缀 + 短后缀

kv1, r1_used = cache.get_or_compute(r1, compute_kv)
print(f"r1: reused {r1_used} tokens")     # 0 (首次, 全算)
kv2, r2_used = cache.get_or_compute(r2, compute_kv)
print(f"r2: reused {r2_used} tokens")     # 8 (复用 system prompt)
kv3, r3_used = cache.get_or_compute(r3, compute_kv)
print(f"r3: reused {r3_used} tokens")     # 8 (同上)
```

预期：r1 无复用（首次），r2/r3 各复用 8 token 的 system prompt——只 prefill 自己的后缀。

### 5.2 radix tree 风格（示意）

```python
class RadixNode:
    def __init__(self, tokens):
        self.tokens = tokens; self.children = {}; self.kv = None; self.ref = 0
class RadixCache:
    def __init__(self): self.root = RadixNode([])
    def insert(self, tokens, compute_kv):
        """插入请求, 复用已有前缀, 返回复用长度."""
        node, i = self.root, 0
        # 沿树找最长公共前缀
        while i < len(tokens):
            head = tokens[i]
            if head not in node.children: break
            child = node.children[head]
            # 比较该 child 的 token 段
            j = 0
            while j < len(child.tokens) and i+j < len(tokens) and child.tokens[j] == tokens[i+j]:
                j += 1
            if j == len(child.tokens):
                node = child; i += j; node.ref += 1
            else:
                # 部分匹配: 需 split (略, 工程实现细节)
                break
        reused = i
        # 剩余 token 作为新子节点
        if i < len(tokens):
            new_child = RadixNode(tokens[i:]); new_child.kv = compute_kv(tokens[i:]); new_child.ref = 1
            node.children[tokens[i]] = new_child
        return reused
```

这是 SGLang radix tree 的极简示意（省略 split/copy-on-write）。


## 6. 与其他知识点的关系

- **上游（依赖）**: [[KV cache management]]（prefix caching 是 KV cache 复用的特化）、PagedAttention（block 共享是零拷贝复用的前提）、[[attention]]（前缀 KV 的来源）。
- **下游（应用）**: [[continuous batching]]（新请求 prefill 时先查 prefix cache）、[[batching tradeoff]]（减少 prefill 开销影响 batch 调度）、[[sampling throughput]]（RLHF 中 system prompt 固定时大幅提速采样）、[[policy deployment (PD)]]（PD 服务的降本关键）。
- **对比 / 易混**:
  - **prefix caching vs [[KV cache management]]**：后者是单请求内缓存历史 KV 供自回归复用；前者是**跨请求**复用公共前缀 KV。层次不同：单请求 vs 多请求。
  - **prefix caching vs checkpointing**：无关。后者是训练省激活（[[gradient checkpointing]]）。
  - **prefix caching vs [[量化|KV 量化]]**：前者复用计算（省 prefill FLOPs），后者压缩存储（省显存）。可叠加：量化 KV + 前缀复用。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为前缀"相似"就能复用
> 不能。prefix caching 要求**逐 token 完全相同**的前缀才命中（block 哈希严格匹配）。一个 token 不同，从该 block 起就不复用。"语义相似"不算数。

> [!warning] 误区 2：以为权重更新后缓存仍有效
> KV cache 是**对应特定 $\theta$** 的中间结果。权重更新（[[weight sync mechanism]]）后，旧 KV 与新 $\theta$ 不匹配，必须失效缓存，否则 attention 结果不一致（[[stale policy problem]]）。生产系统要给缓存打版本号，切换即清空。

> [!warning] 误区 3：以为 prefix caching 优化 decode
> 它优化的是 **prefill**（省前缀的 prefill 计算）。decode 阶段每步仍正常写自己的 KV cache，prefix caching 不直接干预 decode。

> [!warning] 误区 4：忽略缓存容量与命中率
> 显存有限，prefix cache 装不下所有可能前缀。冷前缀被 LRU 驱逐后，新请求若恰好用该前缀会 miss。设计时要预留 cache 显存预算，监控命中率。

> [!warning] 误区 5：以为 block 越小命中率越高越好
> 小 block 命中更细（部分前缀也能复用），但哈希表更大、查找开销高、碎片多。典型 block 16 token 是权衡点。


## 8. 延伸细节

### 8.1 OpenAI prompt caching 的实际形态

OpenAI API 的 prompt caching：对长 prompt（>1024 token）自动缓存，相同前缀的后续请求按缓存命中 token 数打折计费（如缓存命中部分 0.5× 费率）。这推动应用层把稳定 system prompt / few-shot 放最前面、变量放最后，以最大化命中。

### 8.2 SGLang 的结构化共享

SGLang 不仅做前缀缓存，还支持**结构化提示**的共享（如 few-shot 多分支、多轮对话树），通过 radix tree 自动发现任意共享结构，比朴素前缀匹配更激进。

### 8.3 与 RLHF 的关系

[[rollout worker]] 采样时，system prompt / prompt template 固定。prefix caching 让每个采样请求只 prefill 用户输入 + 生成部分，system prompt 段复用，大幅提升 [[sampling throughput]]。在 [[policy training (PT)]]-[[policy deployment (PD)]] 架构里，PD 侧的 prefix caching 是 RLHF 训练加速的关键一环。但注意 [[weight sync mechanism]] 推新 $\theta$ 时需失效缓存。

### 8.4 chunked prefill 与 prefix caching 的协同

长前缀即使缓存 miss 也要 chunked prefill（不阻塞 decode）；缓存 hit 时直接复用，连 chunked prefill 都省了。两者在 vLLM/SGLang 调度器里协同决定每个 iteration 做什么。

---
相关: [[推理优化]] | [[KV cache management]] | [[continuous batching]] | [[batching tradeoff]] | [[sampling throughput]]
