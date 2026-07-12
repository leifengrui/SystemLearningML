# KV cache management

> **所属章节**: [[推理优化]]
> **所属模块**: [[08-推理系统]]
> **别名**: KV cache / 键值缓存 / KV 缓存 / KV-Cache
> **难度**: 中高（需懂 [[attention]] + 显存系统）


## 1. 一句话定义

**KV cache（键值缓存）** 是 Transformer **自回归解码**时缓存的历史 token 的 **Key / Value 矩阵**：每生成一个新 token，只计算当前 token 的 $K, V$ 并 append 到 cache，再用当前 query 对整个 cache 做 attention，从而**避免每步重算历史 token 的 $K/V$ 投影**。它是 LLM 推理系统的"显存动脉"——既让解码从 $O(n^3)$ 计算降到 $O(n^2)$，又是推理时**最大的动态显存开销**（常超过模型权重本身）。

> [!note] 三句话定位
> - **是什么**：解码时把历史层的 $K, V$ 张量存显存里复用。
> - **为什么**：自回归每步只多 1 个 token，重算全部历史 $K/V$ 是浪费。
> - **代价**：显存 $\propto \text{layers} \times \text{seq} \times \text{batch}$，是推理吞吐的主要瓶颈。


## 2. 为什么需要它（动机与背景）

### 2.1 朴素自回归的浪费

Transformer 生成文本是**自回归**的：第 $t$ 步用已生成的 $t-1$ 个 token 预测第 $t$ 个。第 $t$ 步要计算 attention：

$$
\text{Attn}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\!\left(\frac{Q_t K_{1:t}^\top}{\sqrt{d}}\right) V_{1:t}
$$

这里 $K_{1:t}, V_{1:t}$ 是前 $t$ 个 token 经线性投影得到的 Key/Value。**问题**：第 $t+1$ 步又要算 $K_{1:t+1}, V_{1:t+1}$，其中前 $t$ 个 token 的 $K, V$ 与第 $t$ 步**完全相同**却被重算一遍。

| 项 | 朴素（每步重算全部） | KV cache（只算新 token） |
|---|---|---|
| 每步 $K/V$ 投影 | $O(t \cdot d)$（全部历史） | $O(1 \cdot d)$（仅当前） |
| $n$ 步总投影 | $\sum_{t=1}^{n} t = O(n^2)$ | $O(n)$ |
| 每步 attention scores | $O(t \cdot d)$（query 对全部 key） | 同左（无法省） |
| $n$ 步总 attention | $O(n^2)$ | $O(n^2)$ |
| **显存** | 低（不存历史） | **高**（存全部历史 $K/V$） |

> [!tip] KV cache 省的是"投影重算"，省不掉"attention 本身"
> attention scores $Q K^\top$ 每步仍要对全部历史 key 算（新 query 要看所有旧 key），这部分 $O(n^2)$ 是 attention 机制固有的，cache 无法消除。**KV cache 用显存换的是"重复的 $W_K, W_V$ 线性投影计算"**。所以长序列下 attention 的 $O(n^2)$ 计算（[[FlashAttention]] 优化）和 KV cache 的 $O(n)$ 显存是两个并行的瓶颈。

### 2.2 没有它会怎样

- 每生成一个 token 都把整段历史过一遍 $W_K/W_V$ 投影 → 序列越长越慢，实际不可用。
- 推理框架（vLLM / TGI / TensorRT-LLM / llama.cpp）的核心优化都围绕 KV cache 的**分配、复用、压缩**展开。

> [!note] 解答：KV cache 是通过什么来管理的？
> KV cache 的"管理"不是一个单一组件在做，而是**三层协同**——从底到高分别由**算子层、引擎 cache manager、调度器**各司其职：
>
> | 层次 | 管理者 | 管什么 | 典型机制 |
> |---|---|---|---|
> | **① 算子/层级别** | attention kernel 自身 | 单次前向内新 token 的 K/V 怎么 append 进已有 cache | 预分配 buffer + 填充指针 `pos`（`cache_k[:, :, pos:pos+1] = k_new`），见 §5.2；PagedAttention kernel 在 GPU 上按 block table 散落写入 |
> | **② 引擎 cache manager** | 推理引擎的 KV cache 管理模块 | block 池的分配/回收/映射/共享/换出 | vLLM `KVCacheManager`+`block pool`+`block table`；SGLang `RadixCache` 前缀树；TRT-LLM 连续/int8 KV |
> | **③ 调度器级别** | continuous batching scheduler | 决定**哪些请求**的 cache 该建、该留、该驱逐、该换出 | LRU 驱逐、recompute、CPU/SSD offload、prefix 共享，见 §3.5 |
>
> ### ① 算子层：append 语义的执行者
> 最底层是 attention 算子本身负责把每个新 token 的 $k_t, v_t$ 写进 cache。教学版用 `torch.cat`（每步复制整个 cache，$O(L)$ 拷贝，浪费）；工程版用**预分配 buffer + 填充指针**：
>
> ```python
> # 预分配 [B, H, max_len, hd] 的 buffer，用 pos 指针原地写
> self.k_buf[:, :, pos:pos+S] = k_new      # 不复制历史
> k_used = self.k_buf[:, :, :pos+S]        # attention 只看有效段
> ```
>
> PagedAttention kernel 则更进一步：buffer 被切成固定 block（如 16 token），kernel 按 block table 把逻辑位置映射到物理 block，写入可以在物理上不连续的 block 池里。这一层关心的是"一次前向内如何正确、无拷贝地把新 K/V 接进 cache"。
>
> ### ② 引擎 cache manager：显存池的分配者（核心）
> 这是"管理 KV cache"最关键的一层。以 **vLLM** 为例，它的 `KVCacheManager` 组件做这些事：
>
> - **block 池（BlockPool）**：启动时把 GPU 显存里预留给 KV cache 的总量切成固定大小 block（默认 16 token/block），组成物理 block 池。每个 block 是一段连续显存，但 block 之间无需连续。
> - **block table**：每个请求一张"页表"，把逻辑 block 序号 `[0,1,2,...]` 映射到物理 block 编号。请求增长时向 BlockPool 申请新物理 block，结束时归还。类比 OS 虚拟内存的页表（详见 §8.1）。
> - **引用计数 + copy-on-write**：公共前缀的 block 被多请求共享（prefix caching），引用计数管理；beam search 分支时只在写入分歧 block 时才复制（COW），不克隆整个 cache。
> - **prefix caching / radix tree**：vLLM 用 hash 索引前缀 block 复用；SGLang 用 RadixCache（前缀树）更激进地共享结构化 prompt 的 KV。
>
> 这一层关心的是"显存怎么切、怎么分配、怎么复用、怎么不浪费"——是 PagedAttention 把显存利用率从 ~30% 拉到 ~95%+ 的核心。
>
> ### ③ 调度器层：并发请求的准入与驱逐
> 最上层是 continuous batching 调度器（vLLM `Scheduler`、SGLang scheduler）决定**全局**层面哪些请求的 cache 该：
>
> - **分配/准入**（admission）：新请求来了，prefill 时建 cache，调度器看显存够不够、够就放进来；
> - **驱逐**（eviction）：显存满了，按 **LRU** 丢掉最近最少用请求的 cache；或按注意力重要性丢 token（H2O/StreamingLLM，见批注"很多论文可以有选择的保存kvcache的驻留"）；
> - **换出/重算**（offload/recompute）：把冷 cache 挪 CPU/SSD，或被驱逐请求回来时重算 prefill KV（compute < bandwidth 时更划算）；
> - **前缀共享**：多请求共享公共前缀 KV（system prompt），调度器识别后指向同一物理 block。
>
> 这一层关心的是"有限显存怎么在并发请求间分配、谁该留谁该走"——是 throughput 与显存预算的博弈点，与 [[continuous batching]]、[[batching tradeoff]] 直接挂钩。
>
> ### 一句话总结
> **KV cache 由"推理引擎（vLLM/SGLang/TRT-LLM）的 cache manager 组件"管理**——算子层负责单步 append，cache manager 负责显存分页/分配/复用，调度器负责并发请求的准入/驱逐/换出。三者合起来才能在大 batch + 长序列下把 KV cache 显存榨干（~95% 利用率）又支持 prefix 共享与 dynamic batching。模型本身（nn.Module）不管 cache 生命周期，它只在 forward 里"用"cache；cache 的**分配/回收/共享**完全是推理引擎的工程职责，这也是 vLLM/SGLang 相比朴素 HuggingFace `generate()` 的核心价值所在。

> [!note] 解答：调度器能自动管理 KV cache 吗？能——把请求路由到"命中更多前缀 cache 的实例"是当前推理服务化的核心能力（KV-cache-aware / prefix-aware routing）
> **可以，而且这正是 2025–2026 年生产级 LLM 推理调度的核心演进方向。** 上面 §3.5 和 §2 的"三层管理"讲的是**单实例内** cache 怎么管；本批注问的是**多实例（分布式部署）**层面：调度器怎么"把请求往命中率高的实例上赶"。答案是——**有专门的 KV-cache-aware router 做这件事**，单实例的 [[prefix caching]] 一旦 scale-out 到多副本就会被负载均衡器"打散"失效，必须靠这种感知式路由把 cache locality 找回来。
>
> ### 为什么需要它：单实例 prefix caching 一上集群就崩
> vLLM 单实例的 **Automatic Prefix Caching（APC）** 靠 hash 命中同一前缀 block，复用 KV，能把 TTFT 砍一个数量级（llm-d 实测：Qwen3-32B 重发 10k token prompt，TTFT 从 4.3s 降到 0.6s）。但**一旦多副本部署，标准负载均衡器（round-robin / least-connections）是 cache-blind 的**：它只看"哪台闲"，把同前缀的请求**均匀撒**到各 pod，导致每个 pod 都重新 prefill 一遍公共前缀 → **cache thrashing（颠簸）**：前缀被反复重建+驱逐、GPU 全耗在重复 prefill 上。对 agentic / 多轮对话这类"巨大静态前缀 + 小增量后缀"的负载（输入/输出比可达 100:1），这是灾难。
>
> ### 它怎么工作：给调度器一双"看得见 cache 的眼睛"
> 主流实现分两档（来源：vLLM Production Stack 文档、llm-d 2025-09 博客与 benchmark）：
>
> | 档位 | 机制 | 代表实现 | 精度 |
> |---|---|---|---|
> | **近似（approximate）** | 路由器维护"请求历史 → pod"的**外部索引**，按路由历史**预测** cache locality，不窥探 pod 内部真实 cache | vLLM production-stack 的 prefix-aware router、K8s Gateway API Inference Extension 的 prefix-aware 插件 | 中（高 QPS / 动态负载下预测失准） |
> | **精确（precise）** | 每个 pod 通过 **`KVEvents` 协议**实时上报"哪个 block 被创建/驱逐"，路由器维护**全局 block-hash → pod** 索引，直接算出"该请求前缀在哪个 pod 命中多少 %" | llm-d 的 `llm-d-kv-cache` 库 + `kvcache.Index` + **Precise Prefix-Cache Scorer** | 高（实测碾压近似） |
>
> **精确路由的工作流**（llm-d）：
> 1. 每个 vLLM pod 把 cache block 的**创建/驱逐事件**以 `KVEvents` 流上报；
> 2. `kvevents.Pool` 消费流，更新底层 **KV-Block Index**（block-hash → pod + 媒介 GPU/CPU）；
> 3. `kvcache.Index` 在其上把"逻辑 token 序列（前缀）"映射到"持有它的 pod 集合"；
> 4. **Precise Prefix-Cache Scorer** 对每个新请求，逐 pod 算"前缀命中百分比"作为 **cache affinity score**（=可省的 prefill 计算量）；
> 5. 最终路由 ≠ 只看 affinity（会把流量全压到同一 pod），而是把 affinity score 与**负载 score**（vLLM 队列长度、KV cache 利用率）加权平衡，给出"既粘 cache、又不压垮"的决策。
>
> 索引开销极小：llm-d 实测 DeepSeek-R1 on 8×H200、365GB KV pool，全局索引仅 ~339KB（data:metadata ≈ 1,000,000:1）。
>
> ### 实测收益（llm-d 2025-09 benchmark，8 pod × 2 H100，Qwen-32B，150 企业租户 × 6k token 前缀，3–60 QPS）
>
> | 调度策略 | 输出 tok/s | TTFT p90 | TTFT mean | vLLM 等待队列 |
> |---|---|---|---|---|
> | **precise（精确 cache 感知）** | **8730** | **0.542s** | **0.298s** | **0.1** |
> | approximate（近似 cache 感知） | 6944 | 31.08s | 13.32s | 8.1 |
> | load-only（只看负载） | 4429 | 94.87s | 46.99s | 28.9 |
> | random（盲调度） | 4429 | 92.55s | 45.28s | 27.3 |
>
> → 精确路由比近似路由 **TTFT 快 ~57×**、吞吐 +25%；比盲调度**快 ~170×**、吞吐翻倍。Anthropic / OpenAI 的 API 定价里**缓存 token 比未缓存便宜 10×**（$0.30 vs $3.00 / M），cache 命中率直接等于省钱。
>
> ### 选型梯度（llm-d 总结）
> 1. **random / round-robin**：对称负载、无前缀复用才用；
> 2. **load-aware**：非对称负载必备，但 cache-blind；
> 3. **approximate prefix-aware**：有前缀复用但规模/动态性不高；
> 4. **precise prefix-aware**：高规模 + 紧 SLO + 重前缀复用（agentic / 多租户 SaaS）的最优解。
>
> ### 与本笔记三层模型的对应
> 这是 §2 "三层管理"的**第 ③ 层（调度器层）在多实例维度的延伸**：单实例调度器管"同一 GPU 上多个请求的 cache 准入/驱逐"；多实例 router 管"集群里哪个实例的 cache 与新请求前缀最匹配"。两者正交叠加。**KV cache 的"自动管理"因此有两级**：实例内由 vLLM `Scheduler` + `KVCacheManager` 管；实例间由 KV-cache-aware router 管。未来方向（llm-d 路线图）：①KV offload 到 CPU 做分层池、router 做 latency-aware 决策；②**KV-Cache-Fusion** 实现 RAG 场景的"乱序文档复用"（位置无关的 KV 融合），打破简单前缀匹配的限制。
>
> 来源：[vLLM Production Stack – KV Cache Aware Routing](https://docs.vllm.ai/projects/production-stack/en/vllm-stack-0.1.4/tutorials/kvaware.html)；[llm-d – KV-Cache Wins You Can See (2025-09)](https://llm-d.ai/blog/kvcache-wins-you-can-see)；[vLLM Automatic Prefix Caching 文档](https://docs.vllm.ai/en/stable/design/prefix_caching/)。详见 [[prefix caching]]、[[continuous batching]]。

## 3. 核心概念详解

### 3.1 cache 的形状与生命周期

每个请求、每层、每个 head 都有一份 $K$ 和 $V$。以 Llama 风格多头注意力为例，单个请求的 cache 形状：

$$
K_{\text{cache}}, V_{\text{cache}} \in \mathbb{R}^{n_{\text{layers}} \times n_{\text{heads}} \times L \times d_{\text{head}}}
$$

其中 $L$ 是目前已生成的序列长度（随解码增长），$d_{\text{head}}$ 是每头维度，$n_{\text{heads}} \cdot d_{\text{head}} = d_{\text{model}}$。

> [!note] prefill 与 decode 两阶段
> - **Prefill（预填）**：处理输入 prompt。prompt 有多个 token，**一次性并行**计算它们的 $K/V$ 并填入 cache（这一步是 compute-bound，算力密集）。
> - **Decode（解码）**：逐 token 生成。每步喂 1 个 token，算其 $K/V$，**append** 到 cache 末尾，再对全 cache 做 attention（这一步是 memory-bound，每步只动一点点算力却要读整个 cache）。
>
> 两阶段的瓶颈不同：prefill 拼 [[GPU utilization]]（算力），decode 拼显存带宽与 [[batching tradeoff]]。vLLM 的 [[continuous batching]] 就是为了让 decode 阶段多个请求凑批、提升带宽利用率。

> [!note] 解答：prefill 和 decode 本质上都是算 KV，分两阶段不仅因为"token 拿到的时机不同"，更因为两者的**计算图形态、瓶颈、调度方式**根本不同
> 你的理解**大方向对**，但只说到了"token 什么时候拿到"这一层，缺了三个更关键的工程维度。完整答案如下。
>
> ### 一、你说对的部分（底层共性）
> 是的——prefill 和 decode 在"做什么"上是**同一件事**：把 token 过 $W_K, W_V$ 线性投影得到 $k_t, v_t$，append 进 KV cache，再用 query 对全 cache 做 attention。公式一模一样：
>
> $$
> \text{out}_t = \text{Attn}(q_t,\; K_{\text{cache}},\; V_{\text{cache}}), \quad K_{\text{cache}} \leftarrow [K_{\text{cache}};\, k_t]
> $$
>
> 两阶段只是**调用同一个 forward 的两种模式**：prefill 调 `layer(x[1..N], use_cache=True)` 一次，decode 在循环里反复调 `layer(x_t, kv_cache=cache, use_cache=True)`。代码层面就是 §5.1 那个 `CachedMHA` 里 `S>1` 与 `S=1` 两条分支。
>
> ### 二、你说漏的部分：为什么必须分两阶段（不止"拿到的时机"）
>
> | 维度 | Prefill | Decode |
> |---|---|---|
> | **输入 token 数** | $S = N$（整段 prompt，几十~几千） | $S = 1$（每步 1 个） |
> | **能否拿到全部输入** | 能（prompt 用户一次给全） | 不能（第 $t$ 个要等第 $t{-}1$ 步 argmax 出来才有）←你说的这点 |
> | **attention 因果性** | causal mask 允许 $N$ 个 query **互相并行**算（已知全部 token，第 $i$ 个只看 $\le i$ 的，但不阻塞并行） | 第 $t$ 步的 query **必须等**第 $t{-}1$ 步的 $k_{t-1},v_{t-1}$ 写进 cache 才能算（串行依赖） |
> | **计算量 vs 数据量** | 算 $O(N^2 d)$ 的 attention scores（矩阵-矩阵 GEMM） | 每步 $O(N d)$（矩阵-向量 GEMV） |
> | **瓶颈类型** | **compute-bound**（算力密集，GPU SM 打满） | **memory-bound**（每步读整个 KV cache，算力几乎闲置） |
> | **KV cache 动作** | 从空 → 一次性填满 $N$ 个 | 在末尾 append 1 个（指针 pos+1） |
> | **目标指标** | TTFT（首 token 延迟） | TPOT / 生成吞吐（每 token 延迟） |
>
> 关键洞察：**就算 token 能一次拿全，attention 的串行因果依赖仍然强制 decode 一步步走**——这是自回归生成的**算法本质**，与"prompt 是否提前给定"无关。prompt 给全了能并行，是因为 prompt 内部各 token 之间**没有循环依赖**（第 $i$ 个 prompt token 的 attention 只依赖 $\le i$ 的 prompt token，全部已知，可一次性并行算完）。而 decode 时第 $t$ 个生成 token 的 $q_t$ 要参与算出 $\text{logits}_t \to \arg\max \to x_{t+1}$，$x_{t+1}$ 才能算 $k_{t+1}$，**形成不可拆解的串行链**。这才是两阶段分野的**根本原因**，"prompt 提前给"只是表象。
>
> ### 三、两个被遗漏的工程后果（为什么推理引擎必须显式区分两阶段）
>
> 1. **kernel 完全不同**：prefill 用 **GEMM kernel**（批量大、算力密集，喂给 cuBLAS/Tensor Core 打满 SM）；decode 用 **GEMV kernel**（每步 1 个 token，算力极小，瓶颈在把 KV cache 从 HBM 读进 SRAM）。同一层、同一权重，两种 kernel 调用路径， Triton/cutlass 要各写一遍。[[FlashAttention]] 主要价值在 prefill 的 $O(N^2)$ attention；decode 阶段更多靠 **FlashDecoding / FlashAttention-2 的 split-kv** 路径优化带宽。
>
> 2. **调度策略完全不同**（这正是 vLLM / SGLang 的核心战场）：
>    - prefill 是 compute-bound → 应该**攒成大 batch 一次算**，但攒太久会拖高 TTFT；
>    - decode 是 memory-bound → 应该**多个请求凑批**共享一次 KV cache 读取（[[continuous batching]] 的本质：decode 阶段不同请求的 cache 各自独立，可同时算、共用一次 kernel launch）；
>    - 两阶段抢的资源冲突 → 产生 [[chunked prefill]]、[[prefill-decode disaggregation]]、SGLang 的 **chunked prefill + decode 融合**等调度策略：把长 prefill 切块穿插进 decode 流水，或干脆 prefill/decode 分到不同 GPU（PD 分离）各跑各的 kernel。
>
> ### 四、一句话总结
> **prefill 与 decode 本质是同一个 "算 KV + attention" 操作的两种调用模式**；之所以分两阶段，根本原因是**自回归生成的串行因果依赖**使 decode 无法并行（而非仅仅"prompt 拿到得早"），并由此衍生出 compute-bound vs memory-bound 两套截然不同的 kernel / 调度 / 指标体系。你的直觉抓住了表象，但工程上真正的"为什么"是 attention 的因果串行链 + 两阶段瓶颈切换。
>
> 关联：[[attention]]（因果 mask 决定 prefill 能并行）、[[continuous batching]]（decode 凑批）、[[chunked prefill]]（两阶段融合调度）、[[prefill-decode disaggregation]]（PD 分离）、[[GPU utilization]]（compute-bound vs memory-bound 判据）、[[batching tradeoff]]。
### 3.2 增量更新（append 语义）

解码第 $t$ 步：
1. 当前 token $x_t$ 过 $W_Q, W_K, W_V$ 得 $q_t, k_t, v_t$（各 $1$ 个 token）。
2. $K_{\text{cache}} \leftarrow [K_{\text{cache}};\, k_t]$, $V_{\text{cache}} \leftarrow [V_{\text{cache}};\, v_t]$（沿序列维 append）。
3. $\text{out}_t = \text{Attn}(q_t, K_{\text{cache}}, V_{\text{cache}})$。

> [!warning] 误区：append 不是原地写
> 直接 `torch.cat` 会每步复制整个 cache（$O(L)$ 拷贝），浪费且碎片化。生产系统用**预分配 buffer + 填充指针**（如 `cache_k[:, :, pos, :] = k_t`），避免反复分配。下文代码示例为清晰用 `cat`，工程实现见 [[#5. 代码示例（可选]] 的 buffer 版。

### 3.3 PagedAttention（vLLM 的分页 KV）

朴素 cache 要求**连续显存**且**预分配最大 seq_len**，导致：
- **碎片化**：请求长短不一，固定块浪费。
- **无法共享**：公共前缀（system prompt）的 KV 不能复用。

**PagedAttention**（vLLM）借鉴 OS 虚拟内存分页：把每个请求的逻辑 KV 序列切成固定大小 **block**（如 16 token/block），用 **block table** 把逻辑 block 映射到物理 block 池。物理 block 可不连续，空闲即用。

| 特性 | 朴素连续 cache | PagedAttention |
|---|---|---|
| 分配 | 一次分配 max_len | 按 block 动态增长 |
| 碎片 | 高（padding 到 max_len） | 低（仅末尾 block 可能不满 16） |
| 显存利用率 | ~30–50% | ~95%+ |
| 前缀共享 | 不支持 | 支持（指向同一物理 block）→ [[prefix caching]] |
| copy-on-write | 无 | 支持（分支生成/beam 不必复制全 cache） |

### 3.4 KV cache 量化

把 FP16 的 $K/V$ 存成 **FP8 / INT8 / INT4**，显存直接减半 / 减 4× / 减 8×。代价是精度损失（$K$ 对量化敏感因 attention score 放大误差，$V$ 相对鲁棒）。典型：FP8 KV 几乎无损，INT4 KV 需配 grouped-query attention + 细调。FP8 的不同格式（E4M3/E5M2）与缩放方案（per-tensor/per-token/delayed）详见 [[FP8量化方案]]；定点 INT8/INT4 的 [[量化]]（待展开，Chapter 1/2）。

### 3.5 cache 管理策略

当显存不够装所有并发请求的 cache 时，需要**驱逐/换出**：

| 策略 | 做法 | 场景 |
|---|---|---|
| **LRU eviction** | 按最近最少使用丢弃旧请求的 cache | 长尾请求多 |
| **recomputing** | 被驱逐的请求回来时重算其 prefill KV | 比换出便宜（compute < bandwidth）时 |
| **offload to CPU/SSD** | 把冷 cache 挪到 CPU 内存或 NVMe | [[activation memory]] 不足，DeepSpeed-MII / FlexGen |
| **prefix sharing** | 多请求共享公共前缀 KV | [[prefix caching]]、SGLang radix tree |


## 4. 数学原理 / 公式

### 4.1 显存占用公式

单个请求、单层、单个 head、单个 token 的 $K$ 或 $V$ 占 $d_{\text{head}}$ 个元素。$K$ 和 $V$ 共 2 份，全 $L$ 层、$n_{\text{heads}}$ 头、序列长 $S$、batch $B$、精度 $p$ 字节/元素：

$$
\boxed{M_{\text{KV}} = 2 \cdot n_{\text{layers}} \cdot n_{\text{heads}} \cdot d_{\text{head}} \cdot S \cdot B \cdot p}
$$

> [!note] 解答：为什么和头数 $n_{\text{heads}}$ 有关？——因为每个 head 独立存一份 K/V，但 MHA 下化简后又"消失"了，这正是 GQA/MQA 能省 cache 的关键
> **一句话**：每个 attention head 都有自己专属的 $W_K^{(h)}, W_V^{(h)}$ 投影矩阵，会算出**独立的一份** $k^{(h)}, v^{(h)} \in \mathbb{R}^{d_{\text{head}}}$；KV cache 要把**所有 head** 的 K/V 都存下来，所以 cache 形状里天然带 $n_{\text{heads}}$ 这一维。
>
> ### 1. 为什么每个 head 都要存一份——从 attention 结构倒推
> 多头注意力（MHA）对输入 $x \in \mathbb{R}^{d_{\text{model}}}$ 做投影：
>
> $$
> q^{(h)} = x W_Q^{(h)},\quad k^{(h)} = x W_K^{(h)},\quad v^{(h)} = x W_V^{(h)} \in \mathbb{R}^{d_{\text{head}}},\quad h=1,\dots,n_{\text{heads}}
> $$
>
> 每个 head 维度 $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$，**注意每个 head 是不同的投影矩阵、算出不同的 K/V**（不是把一个 K 切成 n 段）。所以单个 token 在单层的 cache 张量形状是：
>
> $$
> K_{\text{cache}}^{(\text{1 token, 1 layer})} = \underbrace{[k^{(1)}, k^{(2)}, \dots, k^{(n_{\text{heads}})}]}_{n_{\text{heads}} \times d_{\text{head}}} \in \mathbb{R}^{n_{\text{heads}} \times d_{\text{head}}}
> $$
>
> 全 $n_{\text{heads}}$ 个 head 拼起来正好 $n_{\text{heads}} \cdot d_{\text{head}} = d_{\text{model}}$ 个元素。再乘上层数 $n_{\text{layers}}$、序列 $S$、batch $B$、K+V 两份、精度 $p$ 字节，就得到原公式：
>
> $$
> M_{\text{KV}} = \underbrace{2}_{K+V} \cdot \underbrace{n_{\text{layers}}}_{层} \cdot \underbrace{n_{\text{heads}} \cdot d_{\text{head}}}_{\text{所有 head 的 K/V 拼起来} = d_{\text{model}}} \cdot S \cdot B \cdot p
> $$
>
> 所以"和头数有关"的**直接原因**：每个 head 独立产出 K/V，cache 是"所有 head 的 K/V 沿 head 维度堆叠"的张量。
>
> ### 2. 为什么化简后 $n_{\text{heads}}$ 又"消失"了
> 代入 $d_{\text{model}} = n_{\text{heads}} \cdot d_{\text{head}}$ 后：
>
> $$
> M_{\text{KV}} = 2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot S \cdot B \cdot p
> $$
>
> **头数 $n_{\text{heads}}$ 不见了**——这告诉你一个**反直觉事实**：在标准 MHA 下，KV cache 总显存**只取决于 $d_{\text{model}}$**，与"分成几个头"无关。8 头每头 512 维，和 32 头每头 128 维，cache 大小一样（因为拼起来都是 $d_{\text{model}}=4096$）。换个角度看：把 $d_{\text{model}}$ 固定下来，**多分头只是"把同样大的 K/V 切更多份让 attention 并行算"，不会让 cache 变大或变小**。
>
> ### 3. 那为什么 GQA/MQA 能省 cache？——它们打破了"$n_{\text{heads}} \cdot d_{\text{head}} = d_{\text{model}}$"这个等式
> MHA 的隐含约定：**KV 头数 = Query 头数 = $n_{\text{heads}}$**。GQA/MQA 动的就是这个约定：
>
> | 机制 | Query 头数 | KV 头数 $n_{\text{kv}}$ | 单 token 单层 K 元素数 | cache 倍率（vs MHA） |
> |---|---|---|---|---|
> | **MHA** | $n_h$ | $n_h$ | $n_h \cdot d_h = d_{\text{model}}$ | 1× |
> | **GQA** | $n_h$ | $g$（$1 < g < n_h$） | $g \cdot d_h = \frac{g}{n_h} d_{\text{model}}$ | $\frac{g}{n_h}$ |
> | **MQA** | $n_h$ | **1** | $1 \cdot d_h = \frac{d_{\text{model}}}{n_h}$ | $\frac{1}{n_h}$ |
>
> 关键：GQA/MQA 里 $d_{\text{head}}$ **不变**（仍 = $d_{\text{model}}/n_h$），只是 KV 头数从 $n_h$ 减到 $g$ 或 1。多个 Query head **共享**同一组 KV 头，所以 cache 里只需存 $n_{\text{kv}}$ 份而非 $n_h$ 份。以 Llama-3-70B（$n_h=64$, GQA $g=8$）为例，KV cache 直接省到 $\frac{8}{64} = 1/8$。
>
> 这就是为什么"GQA 让长上下文成本降一个量级"——它不是在 $d_{\text{model}}$ 里省，而是**承认所有 Query head 共用更少的 KV 头**，从根上把 KV 的总维度从 $d_{\text{model}}$ 砍到 $\frac{g}{n_h} d_{\text{model}}$。Llama-2-70B 用 GQA 正是为压 KV 显存以撑住长上下文窗口。
>
> ### 4. 常见误区
> - ❌ **"头数越多 cache 越大"**——MHA 下不成立，cache 只看 $d_{\text{model}}$；多分头省的是 attention 的并行度/表达能力，不增不减 cache。
> - ❌ **"GQA 是把每个 head 的维度缩小"**——错。GQA 的 $d_{\text{head}}$ 与 MHA 相同，减的是**KV 头的个数**（多 Q 头共用一个 KV 头）。
> - ❌ **"MQA cache 是 MHA 的 $1/n_h$ 是因为缩小了 $d_{\text{head}}$"**——错，是因为只剩 1 个 KV 头，元素数 $= 1 \cdot d_{\text{head}}$。
>
> 关联：[[attention]]（MHA/GQA/MQA 的定义与投影矩阵）、§4.3 GQA/MQA 对 cache 的影响。

用 $d_{\text{model}} = n_{\text{heads}} \cdot d_{\text{head}}$ 化简：

$$
M_{\text{KV}} = 2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot S \cdot B \cdot p
$$

> [!note] 推导关键
> - "2" = $K$ 和 $V$ 两份。
> - $n_{\text{heads}} \cdot d_{\text{head}}$ 合并为 $d_{\text{model}}$（总维度）。
> - 与 batch $B$、seq $S$、精度 $p$ **线性**相关——这就是长序列 / 大 batch 下 KV cache 显存爆炸的根因。

### 4.2 数值实例：KV cache 常超过权重

以 **Llama-2 7B**（$n_{\text{layers}}=32$, $d_{\text{model}}=4096$, FP16 $p=2$）为例。

**模型权重显存**：7B 参数 × 2 字节 ≈ **14 GB**。

**单条请求 KV cache**（$S=2048, B=1$）：

$$
M = 2 \times 32 \times 4096 \times 2048 \times 1 \times 2 = 1{,}073{,}741{,}824 \text{ B} \approx 1.07 \text{ GB}
$$

**batch=64 并发**：

$$
M = 1.07 \times 64 \approx 68.7 \text{ GB} \gg 14 \text{ GB} \text{（权重）}
$$

> [!warning] 工程直觉：推理显存主项是 KV cache，不是权重
> 大 batch + 长序列时，KV cache 显存可达权重的 4–10×。所以推理引擎的"能开多大 batch / 多少并发"几乎完全由 KV cache 显存决定，而非权重。这是 [[batching tradeoff]] 与 [[continuous batching]] 的出发点。

### 4.3 GQA / MQA 对 cache 的影响

- **MHA**：$n_{\text{heads}}$ 个 KV 头 → cache 全量。
- **MQA**（Multi-Query）：所有 query head 共享 **1** 个 KV 头 → cache 缩 $\times \frac{1}{n_{\text{heads}}}$。
- **GQA**（Grouped-Query）：分 $g$ 组，每组共享 1 KV 头（$g$ 个 KV 头）→ cache 缩 $\times \frac{g}{n_{\text{heads}}}$。

Llama-2 70B / Llama-3 全系用 GQA，正是为压 KV cache 显存、提升 decode 吞吐。详见 [[attention]]（Chapter 2，待展开 / 已有）。


## 5. 代码示例（可选

### 5.1 教学版：带 KV cache 的多头注意力（cat 语义，看清 append）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CachedMHA(nn.Module):
    """简化多头注意力 + KV cache（教学用，省略 RoPE/GQA/causal mask 细节）。"""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None, use_cache=False):
        # x: [B, S, D]  (prefill: S>1; decode: S=1)
        B, S, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)            # 各 [B, S, D]
        # 拆头: [B, S, D] -> [B, H, S, hd]
        split = lambda t: t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)

        # —— KV cache: 把新 k/v 拼到历史末尾 ——
        if kv_cache is not None:
            k_past, v_past = kv_cache                      # 各 [B, H, S_past, hd]
            k = torch.cat([k_past, k], dim=2)             # [B, H, S_past+S, hd]
            v = torch.cat([v_past, v], dim=2)
        new_cache = (k, v) if use_cache else None

        # attention: q 对更新后的 k/v
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)                  # [B, H, S_q, S_kv]
        ctx = torch.matmul(attn, v)                       # [B, H, S_q, hd]
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(ctx), new_cache


# 演示: prefill 一次建 cache, decode 逐步 append
torch.manual_seed(0)
layer = CachedMHA(d_model=64, n_heads=8)
prompt = torch.randn(1, 5, 64)                            # 假设 prompt 5 token

# prefill: 一次算 prompt 全部, 建 cache
out, cache = layer(prompt, use_cache=True)
print("prefill out:", tuple(out.shape), "cache K:", tuple(cache[0].shape))
# (1, 5, 64) (1, 8, 5, 8)

# decode: 每步喂 1 个新 token, cache 增长
for step in range(3):
    new_tok = torch.randn(1, 1, 64)
    out, cache = layer(new_tok, kv_cache=cache, use_cache=True)
    print(f"step {step} out:", tuple(out.shape), "cache len:", cache[0].shape[2])
# step 0 out: (1, 1, 64) cache len: 6
# step 1 out: (1, 1, 64) cache len: 7
# step 2 out: (1, 1, 64) cache len: 8
```

### 5.2 工程版要点：预分配 buffer（避免 cat 复制）

```python
class CachedMHA_Buffer(nn.Module):
    """工程风格: 预分配 max_len 的 KV buffer, 用指针 pos 填充, 避免 cat 拷贝。"""
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.max_len = max_len
        # 预分配 [B, H, max_len, hd] —— 实际由框架(vLLM PagedAttention)按 block 管理
        self.register_buffer("k_buf", torch.zeros(1, n_heads, max_len, self.head_dim), persistent=False)
        self.register_buffer("v_buf", torch.zeros(1, n_heads, max_len, self.head_dim), persistent=False)

    def forward(self, x, pos):
        # x: [B, S, D], pos: 当前要写入的起始位置
        B, S, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # 原地写入 buffer 的 [pos, pos+S) 段, 不复制历史
        self.k_buf[:, :, pos:pos+S] = k
        self.v_buf[:, :, pos:pos+S] = v
        # attention 只看 [0, pos+S)
        k_used = self.k_buf[:, :, :pos+S]
        v_used = self.v_buf[:, :, :pos+S]
        scores = torch.matmul(q, k_used.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v_used).transpose(1, 2).contiguous().view(B, S, D)
        return ctx   # 省略 out 投影
```

### 5.3 显存估算小工具

```python
def kv_cache_bytes(n_layers, d_model, seq_len, batch, bytes_per=2):
    """KV cache 显存(字节). K+V 共 2 份."""
    return 2 * n_layers * d_model * seq_len * batch * bytes_per

# Llama-2-7B, seq 2048, batch 64, FP16
print(f"{kv_cache_bytes(32, 4096, 2048, 64, 2)/1e9:.1f} GB")   # ~68.7 GB
# 对比权重 7B*2 = 14 GB
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[attention]]（自注意力机制，K/V 从哪来）、[[Transformer]]（整体结构）、[[混合精度训练|FP16/BF16 精度]]（决定 $p$）。
- **下游（应用）**: [[continuous batching]]（基于 KV cache 的迭代级调度）、[[prefix caching]]（复用公共前缀 KV）、[[speculative decoding]]（草稿模型也要写主模型 cache）、[[batching tradeoff]]（batch 大小受 KV 显存约束）、[[GPU utilization]]（decode 阶段带宽瓶颈在 cache 读取）、[[activation memory]]（KV 是激活显存的最大组成）。
- **对比 / 易混**:
  - **KV cache vs 模型权重**：权重是共享参数（固定，推理时不改）；KV cache 是**每个请求独立**的激活（动态，随序列增长）。推理显存 = 权重 + 所有并发请求的 KV cache 之和。
  - **KV cache vs activation checkpointing**：前者是推理时存历史 K/V 供复用；后者是训练时省前向激活的重算策略（[[gradient checkpointing]]，Chapter 11）。名字都含"cache/checkpoint"但场景完全不同。
  - **PagedAttention vs radix tree**：前者按固定 block 分页（vLLM）；后者按前缀树组织共享前缀（SGLang），见 [[prefix caching]]。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 KV cache 能消除 attention 的 $O(n^2)$ 计算
> 不能。$Q K^\top$ 每步新 query 仍要对**全部历史 key**算分。cache 省的是"$K/V$ 线性投影的重复计算"，attention scores 计算量不变。$O(n^2)$ 那部分靠 [[FlashAttention]] / 稀疏注意力减。

> [!warning] 误区 2：以为 KV cache 显存很小、可忽略
> 恰恰相反，大 batch + 长序列时 KV cache 显存 **远超权重**（见 §4.2，64×2048 → 64GB vs 14GB 权重）。推理吞吐的硬约束常是 KV cache 显存。

> [!warning] 误区 3：以为只有 decode 阶段才有 cache
> prefill 阶段也在**建** cache（一次性算 prompt 全部 K/V 填入）。prefill 是 compute-bound，decode 是 memory-bound，两者瓶颈不同。

> [!warning] 误区 4：以为请求结束 cache 立即无效
> cache 本身可被 [[prefix caching]] / SGLang radix tree **跨请求复用**（公共前缀）。只有权重更新（如在线学习）后 cache 才真正失效（版本不匹配）。

> [!warning] 误区 5：MHA/GQA/MQA 搞混 cache 倍率
> 记住：cache 大小正比于 **KV 头数**。MHA = $n_h$ 倍，GQA = $g$ 倍（$g$ 组），MQA = 1 倍。Llama-3 用 GQA 正是为此。


## 8. 延伸细节

### 8.1 PagedAttention 的 block table（类比 OS 页表）

逻辑上请求看到连续的 $[0, 1, \dots, L-1]$，物理上每个逻辑 block 映射到 block 池中某个空闲物理 block：

```
逻辑 block 序号:    [0]      [1]      [2]      [3]
block table:        → 物理block#5  → #2  → #9  → #1
物理 block 池:       #0(空) #1(用) #2(用) ... #5(用) #9(用)
```

请求增长时申请新物理 block；结束时归还。前缀相同时多个请求的 block table 前几项指向**同一物理 block**（[[prefix caching]]）。beam search 分支用 **copy-on-write**：只在写入分歧 block 时才复制，不必克隆整个 cache。

### 8.2 KV cache 与带宽瓶颈（decode 阶段）

decode 每步只算 1 个 token 的 forward（计算量极小），但要**读整个 KV cache**进 SRAM 算 attention。对 7B 模型 batch=1，每步读 ~1GB cache → 受 HBM 带宽限制（A100 1.5TB/s → 约 1.5ms/token → ~660 tok/s 上限，实际更低）。这是 decode 阶段 [[GPU utilization]] 低、必须靠 [[continuous batching]] 凑大 batch 提升带宽利用率的根因。

### 8.3 KV 量化与 offload

- **FP8 KV**（vLLM/TensorRT-LLM 支持）：显存减半，精度几乎无损，是当前主流。
- **INT4 KV**：省 4×，但 $K$ 量化误差被 attention score 放大，需配 GQA + 细调或 mixed-precision（$K$ 保高精度、$V$ 量化）。
- **KV offload**：DeepSpeed-MII / FlexGen 把冷 cache 换到 CPU 内存或 NVMe，用 PCIe/NVMe 带宽换 HBM 容量。适合离线批量推理（延迟不敏感）。

### 8.4 cache 与 RLHF / 训推分离

在 [[policy training (PT)]]-[[policy deployment (PD)]] 架构（[[训推分离]]）中，PD 侧的 KV cache 是**采样吞吐**的关键资源。[[weight sync mechanism]] 推新权重时，旧 cache 因对应旧 $\theta$ 而失效（logits 不一致），通常**清空 cache 或等当前请求完成再切换**。这是 [[stale policy problem]] 在推理侧的投影。

### 8.5 与 vLLM / SGLang / TRT-LLM 的对应

| 框架 | KV cache 管理 | 特色 |
|---|---|---|
| vLLM | PagedAttention（block 16） | 显存利用率高、prefix caching 内置 |
| SGLang | Radix tree 前缀复用 | 结构化 prompt 共享更激进 |
| TensorRT-LLM | 连续/int8 KV | 极致 kernel 融合、低延迟 |
| DeepSpeed-FastGen | Dynamic Splitfuse | prefill/decode 融合调度 |

---
相关: [[推理优化]] | [[continuous batching]] | [[prefix caching]] | [[attention]] | [[activation memory]]
