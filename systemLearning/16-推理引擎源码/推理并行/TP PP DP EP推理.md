# TP PP DP EP 推理

> **所属章节**: [[推理并行]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: 推理并行 / inference parallelism / TP 推理 / PP 推理 / DP 推理 / EP 推理 / vLLM TP / SGLang PD 分离 / expert parallel 推理 / 张量并行推理
> **难度**: 中高（需懂 [[Tensor Parallel]] + [[Pipeline Parallel]] + [[Data Parallel]] + [[专家并行]] + [[MoE]] + [[MoE路由]] + [[worker与executor]] + [[model runner]]）


## 1. 一句话定义

**推理并行** 是把一个 LLM 推理请求的**算子/权重/请求**分布到多 GPU 上执行的方式，主流有四种维度：**TP（张量并行，把每层权重列/行切分到多卡，每卡算一片，linear 后 all-reduce 聚合）**、**PP（流水线并行，按 layer 把模型切多段，跨节点，每段一卡/几卡，激活跨段传递）**、**DP（数据并行，多副本各自服务不同请求，forward 间零通信，最简单的纯扩吞吐）**、**EP（专家并行，MoE 的 expert 切到多卡，token 经 All-to-All 派发到目标 expert 所在卡算完再收回）**。与训练侧**"DP 为主流"**相反，推理侧**"TP 为主流"**——因为推理权重冻结、无梯度反向通信，且 TP 的 all-reduce 在 decode（seq=1）时通信量极小（仅 `batch×1×hidden`），而单请求往往放不下一卡需切分权重，TP 既能让单请求用多卡降延迟、又能切权重放得下，是"装得下 + 单请求加速"的最优解；DP 在推理是"装下之后纯堆副本扩吞吐"的叠加项，不能帮装下。vLLM 用 `--tensor-parallel-size`（`-tp`）/`--pipeline-parallel-size`（`-pp`）/`--data-parallel-size`（`-dp`）/`--enable-expert-parallel`（EP）控制这四维；SGLang 除 TP/EP 外，**PD 分离**（prefill/decode 池分离）是**部署架构维度**而非这里的"模型切分并行维度"，但常与 TP/EP 组合部署。

> [!note] 三句话定位
> - **是什么**：把推理的权重/算子/请求/专家分布到多卡。TP 切权重列行、PP 切层、DP 堆副本、EP 切专家。四维正交可组合。
> - **为什么 TP 主流**：推理权重冻结无梯度通信；TP all-reduce 在 decode 通信量极小（`B×1×H`）；单请求放不下需切权重，TP 既装得下又单请求加速。DP 是装下后扩吞吐的叠加。
> - **与训练侧区别**：训练侧 DP 主流（梯度 all-reduce 被大批 token 摊薄，compute:comm 比高），推理侧 TP 主流（DP 零通信但帮不了装下，TP 才能切权重装下 + 单请求多卡）。这是两边并行选型分叉的根因。


## 2. 为什么需要它（动机与背景）

### 2.1 单卡放不下 → 必须切分权重

大模型（70B / 405B / MoE）权重远超单卡显存（A100 80G）。推理时权重冻结（不训练），把权重**切分**到多卡是装下的唯一办法。TP（列/行切 linear 权重）、PP（按 layer 切段）、EP（按 expert 切）都是"切权重让它装得下"的维度。DP 不行——DP 是**复制**整个模型，每卡一份，更费显存，帮不了装下。

### 2.2 单请求要低延迟 → 单请求要多卡

推理是**用户感知延迟**的场景（TTFT/TPOT 见 [[TTFT与TPOT]]）。一个请求若只用一卡，算力不够 → 慢。TP 让**一个请求的 forward 跨多卡并行**（每卡算一片，all-reduce 聚合），单请求延迟降。DP 不行——DP 把不同请求分到不同副本，单请求仍在一卡，延迟不降。

### 2.3 装下后要高吞吐 → 堆副本

模型装下后（用 TP/PP 切到 K 卡），单实例吞吐有限。要扩吞吐就**多起几份副本**（DP），每副本独立服务不同请求，forward 间**零通信**。DP 是"装下之后纯堆吞吐"的最简扩容，无任何 all-reduce / all-to-all。

### 2.4 MoE 的 expert 放不下 → 切专家

MoE 模型（如 DeepSeek-V3，671B、每层 256 expert）的**全部 expert 权重极大**。把 expert 切到多卡（EP），token 按路由派发到目标 expert 所在卡算完再 All-to-All 收回，是 MoE 推理切权重的标准方式。EP 与 TP 正交（expert 维 vs head/hidden 维）。

### 2.5 训练侧 DP 主流、推理侧 TP 主流的根因

训练要**反向传播**：DP（DDP）每卡算不同数据，反向时 all-reduce **全模型梯度**（通信量 = 模型大小）。但训练 batch/seq 大，算力（`6×batch×seq×params`）远大于通信（`2×params`），compute:comm 比高，DP 最高效。TP 训练要每层 forward+backward 都 all-reduce（次数多、且 backward 翻倍），故训练只在装不下时才上 TP。

推理**无反向**：DP 零通信（forward 各算各的，不 sync），看似最优。但 DP 帮不了"装下"（要复制，更费显存），也帮不了"单请求低延迟"。所以推理**先用 TP 切权重装下 + 单请求多卡**，**再用 DP 堆副本扩吞吐**。TP 在 decode 的 all-reduce 通信量 `B×1×H` 极小（seq=1），开销可忽略，这是 TP 在推理"便宜"的关键。

> [!tip] 一句话记牢
> 训练侧 DP 主流（梯度通信被大批 token 摊薄）；推理侧 TP 主流（无梯度通信 + TP decode 通信极小 + TP 能装下能单请求加速；DP 是装下后的吞吐叠加项）。


## 3. 核心概念详解

### 3.1 四维并行总览

```
TP (tensor parallel)  : 每层权重 列/行 切到 N 卡, 每卡算一片, linear 后 all-reduce 聚合
PP (pipeline parallel): 按 layer 切多段, 每段一卡(或几卡), 激活跨段传递 (有 bubble)
DP (data parallel)    : 多副本, 每副本完整模型, 服务不同请求, forward 间零通信
EP (expert parallel)  : MoE expert 切到 N 卡, token 按路由 All-to-All 派发 -> 算 -> 收回

四维正交可组合 (vLLM 总卡数 = tp × pp × dp × ep_inner):
  常见: tp=8 (单机 8 卡), pp=1, dp=N (多副本), ep 可选 (MoE 时 --enable-expert-parallel)
```

| 维度 | 切什么 | 通信原语 | 通信频率 | 能帮"装下"? | 能降单请求延迟? | 主流场景 |
|---|---|---|---|---|---|---|
| **TP** | 每层权重列/行 | all-reduce | 每层 2 次（attn+mlp） | ✅（切权重） | ✅（单请求多卡） | 单机多卡、装不下 + 降延迟 |
| **PP** | 按 layer 切段 | send/recv（P2P） | 每段边界 1 次 | ✅（切层） | △（有 bubble） | 超大模型跨节点 |
| **DP** | 不切（复制） | 无（forward 独立） | 0 | ❌（复制更费） | ❌（单请求仍一卡） | 装下后扩吞吐 |
| **EP** | MoE expert | All-to-All | 每个 MoE 层 2 次 | ✅（切 expert） | △（派发开销） | MoE 推理 |

### 3.2 TP 推理（核心）

**TP（张量并行）** 是把每一层的 linear 权重**按列或按行切分**到 N 卡，每卡只算自己那一片，结果用 **all-reduce 聚合**。沿用 Megatron-LM 的列/行并行算法（vLLM 文档明确："we support Megatron-LM's tensor parallel algorithm"）。

**列并行（ColumnParallelLinear）**：权重 `W ∈ R^{in×out}` 按**输出维（列）**切，第 i 卡拿 `W_i ∈ R^{in×out/N}`，输入 `X` 全卡同（每卡都有完整 `X`），各卡算 `Y_i = X W_i`，输出 `Y_i` 是 `Y` 的**一列切片**（互补重叠，无需通信即可并行）。
**行并行（RowParallelLinear）**：权重 `W ∈ R^{out×in}` 按**输入维（行）**切，第 i 卡拿 `W_i ∈ R^{out/N×in}`，输入 `X` 也按维切（第 i 卡拿 `X_i`），各卡算 `Y_i = X_i W_i`，得到**部分和**，需 **all-reduce** 把各卡部分和加起来才是完整 `Y`。

**Transformer 一层的 TP 流水**（vLLM 实现的典型）：
```
Attention:
  QKV:  MergedColumnParallelLinear / QKVParallelLinear  (列并行, 无 input 通信, output 切片)
        -> 每卡算自己那 num_heads/tp 个 head 的 QKV
  attn: 各卡独立算自己的 head (PagedAttention, 见 [[PagedAttention]])
  o_proj: RowParallelLinear  (行并行)  -> all-reduce ①  聚合 attention 输出
MLP:
  gate_up: MergedColumnParallelLinear  (列并行, gate+up 融合切列)
        -> 激活 (SiLU)
  down:   RowParallelLinear  (行并行)  -> all-reduce ②  聚合 MLP 输出
Embedding:
  VocabParallelEmbedding  (vocab 切列)  -> all-reduce / all-gather 取 logits
LM Head:
  ParallelLMHead  (列并行)  -> all-gather 取 top-k
```

每层 **2 次 all-reduce**（attention o_proj 后、MLP down 后）。embedding/lm head 也有 gather，但只在首尾各一次。

**Attention 的 head 切分**：TP 把 `num_heads` 切给各卡，每卡算 `num_heads/tp` 个 query head。GQA（`num_kv_heads < num_heads`）时：
- 若 `num_kv_heads ≥ tp`：每卡拿 `num_kv_heads/tp` 个 KV head，正常切分。
- 若 `num_kv_heads < tp`：KV head 数少于卡数，**需复制 KV head**（部分卡共享同一 KV head），这是 TP 上限受 `num_kv_heads` 约束的来源（如 Llama-3 8B 的 8 KV head，TP 最大 8；再大就有冗余复制）。

> [!note] 补充：GQA 的 KV head 数是 TP 上限
> TP size 不能超过 `num_kv_heads`（否则 KV head 被迫复制，浪费显存且不增并行度）。设计模型时的 KV head 数直接决定单机能开的 TP 上限。

### 3.3 PP 推理（少见）

**PP（流水线并行）** 按 **layer 切段**，每段放一卡（或几卡）。请求的激活从 stage 0 流到 stage N-1，跨段用 **P2P send/recv** 传递激活（大小 `batch×seq×hidden`）。PP 有**流水线 bubble**：只有把 batch 切成 micro-batch 灌满流水线才能掩盖 bubble，micro-batch 数 M 越多 bubble 占比 `(P-1)/M` 越小。推理请求 batch 小、又对延迟敏感，**bubble 暴露更明显**，故 PP 在推理**少见**，只在**超大模型跨节点**（单机 TP 装不下，需多机）时用——vLLM 推荐：TP 切到单机卡数，PP 切到节点数（`tp=#GPU/node, pp=#nodes`）。

```
PP 推理 (4 stage, 1 micro-batch, bubble 大):
  t0: [stage0 算] ............................ 
  t1:            [stage1 算] ..................
  t2:                     [stage2 算] .........
  t3:                              [stage3 算]
  -> 每时刻只一卡在算, 其余空 (bubble)
PP 训练靠大批 micro-batch 掩流水线填满; 推理 batch 小, bubble 暴露, 故少用
```

vLLM 的 `--pipeline-parallel-size`（`-pp`）支持，配合 Ray 跨节点（多节点推理必须 Ray，单机可用 mp）。详见 [[Pipeline Parallel]]。

### 3.4 DP 推理（最简单）

**DP（数据并行）** 是**多副本**，每副本持有**完整模型**，服务**不同请求**。forward 之间**零通信**（不像训练 DP 要 all-reduce 梯度）。扩吞吐 = 线性加副本（近似线性，受总带宽/调度上限）。DP 是"模型已装下"之后的事——若单卡装不下，DP（复制）帮不上，必须先 TP/PP 切分装下，再 DP 堆副本。

```
DP 推理 (3 副本):
  replica 0 (完整模型, GPU0)  <- 请求 A, B, C...
  replica 1 (完整模型, GPU1)  <- 请求 D, E, F...
  replica 2 (完整模型, GPU2)  <- 请求 G, H, I...
  forward 间零通信, 各自独立调度
  吞吐 ≈ 3× (线性扩)
```

vLLM 的 `--data-parallel-size`（`-dp`）。实践中更常见的是**多实例部署 + 前端 router 负载均衡**（见 [[OpenAI API server]]），而非单进程内开 DP。两者效果同（多副本扩吞吐），单进程 DP 共享一份进程管理开销，多实例则各自独立。

> [!tip] 实践：DP 推理的两种形态
> ① 单进程内 `--data-parallel-size=N`（vLLM V1 支持，共享 engine 配置）；② 多进程/多容器各起一个 vLLM 实例 + 前端 router（Nginx/Envoy/自研 router）round-robin 或 prefix-aware 分发。后者更常见、更易水平扩展，但需自己写 router。

### 3.5 EP 推理（MoE）

**EP（专家并行）** 把 MoE 的 **expert 切到多卡**，每卡持有一组 expert（如 256 expert / 8 卡 = 32 expert/卡）。token 经 **[[MoE路由]]（gate）** 算出 top-k expert id 后，若目标 expert 不在本卡，需经 **All-to-All** 把 token 派发到目标 expert 所在卡，算完再 **All-to-All 收回**结果。

```
EP 推理 (MoE 一层, 8 卡, 每 32 expert):
  1. gate(token) -> top-2 expert id (本地算)
  2. All-to-All ①: 按 expert id 把 token 派发到对应卡 (send token, 收 token)
  3. 各卡对收到的 token 跑自己那 32 个 expert (grouped GEMM, 见 [[grouped GEMM]])
  4. All-to-All ②: 把结果按来源发回原卡 (send result, 收 result)
  5. 原卡把收回的 expert 输出加权合并 (gate 权重) -> 该层输出
```

EP 的通信是 **All-to-All**（每 token 可能去任意卡的任意 expert）。通信量 `batch×seq×hidden`（per MoE 层，双向 ×2）。EP 与 TP 正交：EP 切 expert 维，TP 切 head/hidden 维。vLLM 用 `--enable-expert-parallel`（默认关，TP 切 MoE；开则 EP 切 expert）。MoE 训练侧的 EP 见 [[专家并行]]、[[MoE token dispatcher]]；推理侧同理，但无反向 All-to-All（只 forward 双向）。

> [!note] 补充：EP vs TP 切 MoE
> 不开 EP 时，vLLM 用 **TP 切 MoE**——把每个 expert 的权重列/行切到多卡（每卡持有全部 expert 的 1/tp 片），token 不出卡，所有 expert 在所有卡上各算一片，all-reduce 聚合。这适合 expert 数不多、单 expert 不大的场景。开 EP 则**每卡只持部分 expert**，token 跨卡派发，适合 expert 数极多（如 DeepSeek 的 256 expert）——否则 TP 切会让每卡 expert 权重碎片化、kernel 效率低。

### 3.6 vLLM TP 实现（源码视角）

vLLM 的 TP 由两层支撑：

**① 分布式状态层**（`vllm/distributed/parallel_state.py`）：
- `init_distributed_environment(...)`：初始化 NCCL process group，设 `rank` / `world_size`。
- `initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, ...)`：划分 TP group / PP group（基于 world group 切）。每个 GPU 进程拿到自己的 `tp_rank`、`tp_world_size`、`pp_rank`。
- 查询 API：`get_tensor_model_parallel_rank()`、`get_tensor_model_parallel_world_size()`、`get_pipeline_model_parallel_rank()` 等。
- 执行后端：`--distributed-executor-backend`（`mp` 单机多进程 / `ray` 跨节点 / `uni`）。

**② 算子切分层**（`vllm/model_executor/layers/`）：
- `linear.py`：`ColumnParallelLinear`、`RowParallelLinear`、`MergedColumnParallelLinear`（融合 gate+up）、`QKVParallelLinear`（融合 QKV）。这些类在 `__init__` 时按 `tp_rank` 切权重，`forward` 时本地算 + 必要时 all-reduce。
- `embedding.py`：`VocabParallelEmbedding`（vocab 切列）、`ParallelLMHead`。
- all-reduce 后端：默认用 vLLM 的 **custom all-reduce kernel**（基于 NVLink/SHM 的低延迟实现），`--disable-custom-all-reduce` 回退到 **NCCL** all-reduce。

**启动参数**（vLLM 文档核实）：
```bash
vllm serve <model> \
  --tensor-parallel-size 4 \      # -tp, TP 大小 (单机 4 卡)
  --pipeline-parallel-size 2 \    # -pp, PP 大小 (跨 2 节点)
  --data-parallel-size 1 \        # -dp, DP 副本数
  --enable-expert-parallel        # MoE 用 EP (默认关, 用 TP 切 MoE)
# 单机: tp=#GPU/node; 多机: tp=#GPU/node, pp=#nodes, 用 Ray 后端
```

详见 [[worker与executor]]（TP worker 的封装）、[[model runner]]（每卡 runner 跑自己那片）。

### 3.7 SGLang PD 分离（部署架构，非并行维度）

**PD 分离（Prefill/Decode disaggregation）** 不是 TP/PP/DP/EP 那种"切模型权重"的并行维度，而是**部署架构维度**——把 prefill 池和 decode 池**物理分离**到不同 GPU/实例：
- prefill 是 **compute-bound**（算 prompt），吃 SM；
- decode 是 **memory-bound**（每步 1 token），吃 KV 带宽。
- 分离后 prefill 池可堆算力卡（少 KV、多 SM），decode 池可堆显存卡（大 KV、够带宽），各自按负载配比，**互不干扰**（长 prefill 不再阻塞 decode，见 [[chunked prefill]] 的动机同源）。
- 跨池传 **KV cache**：prefill 算完把 KV 发给 decode 池（用 `--kv-transfer-config` / 共享内存 / NCCL / RDMA）。

SGLang 的 PD 分离是其 serving 架构特性（配合 [[PD分离]] 思想），与 TP/EP 是**正交组合**（prefill 池内仍可 TP，decode 池内仍可 TP）。vLLM 也有 `--kv-transfer-config` 做 KV 跨实例迁移的同类机制。

> [!warning] 误区：PD 分离 = 一种并行维度
> 不是。PD 分离是**部署架构**（prefill/decode 池分离），不切模型权重。TP/PP/DP/EP 是**模型切分并行**（切权重/层/请求/expert）。两者正交，PD 池内仍可开 TP/EP。详见 [[PD分离]]。

### 3.8 SGLang 的 TP/EP

SGLang 的 TP 实现思路与 vLLM 类似（Megatron 式列/行并行 + all-reduce），代码在 `sglang/srt/layers/linear.py`（`ColumnParallelLinear` / `RowParallelLinear`）与 `sglang/srt/parallel_state.py`。SGLang 的特色是 **PD 分离 + RadixAttention prefix cache 共享**（见 [[PD分离]]），TP 是其底层切分手段之一。EP 同理（MoE expert 切 + All-to-All）。


## 4. 数学原理 / 公式

### 4.1 TP 的 all-reduce 通信量（核心推导）

设 batch `B`、当前序列长 `S`、hidden `H`、TP size `N`、环网带宽 `β`、启动延迟 `α`。每层 2 次 all-reduce（attn o_proj、MLP down），消息大小 `M = B·S·H`（元素数，每元素 fp16=2B）。

**Ring all-reduce** 的通信量：每个节点发送 + 接收各 `2·(N-1)/N · M` 数据（reduce-scatter + all-gather 两阶段，各传 `(N-1)/N·M`）。带宽成本（每节点）：

$$
\text{BW}_{\text{TP-per-layer}} = \frac{2 \cdot \frac{N-1}{N} \cdot M}{\beta} = \frac{2(N-1) \cdot B \cdot S \cdot H}{N \cdot \beta}
$$

每层 2 次 all-reduce，共 `L` 层，总：

$$
T_{\text{comm,TP}} \approx 2L \cdot \frac{2(N-1) \cdot B \cdot S \cdot H}{N \cdot \beta} + \text{latency}
$$

**关键：decode 时 `S=1`**：
$$
M_{\text{decode}} = B \cdot 1 \cdot H = B \cdot H
$$
如 `B=256, H=4096` → `M ≈ 1M` 元素 ≈ 2MB，all-reduce 在 NVLink（`β≈600GB/s`）上约 **微秒级**。这就是 TP 在推理 decode"几乎免费"的根因。

**对比 prefill 时 `S=L_p`**（如 `L_p=8192`）：
$$
M_{\text{prefill}} = B \cdot 8192 \cdot H
$$
比 decode 大 `L_p` 倍（8192 倍），prefill 阶段 TP 通信显著得多——这是 [[chunked prefill]] 把长 prefill 切块混 decode 的另一动机（减小单次 all-reduce 消息，配合 TP）。

### 4.2 训练侧 DP 的梯度通信量（对比）

训练 DP（DDP）：forward 各卡独立，**backward 后 all-reduce 全模型梯度**。通信量 `M_grad = 模型参数数 = P`（元素数）。

$$
T_{\text{comm,DP-train}} = \frac{2(N-1) \cdot P}{N \cdot \beta}
$$

而训练算力 `FLOPs ≈ 6 · B · S · P`（forward+backward）。compute:comm 比：

$$
\frac{\text{compute}}{\text{comm}} = \frac{6 B S P}{2(N-1)P/N \cdot \beta^{-1}} \propto \frac{B \cdot S}{\text{常数}}
$$

`B·S` 大时算力远大于通信 → DP 高效。这是训练侧 DP 主流的根因。

### 4.3 推理侧 DP 的通信量

推理无反向。DP 副本 forward 各算各的，**零 all-reduce**：

$$
T_{\text{comm,DP-infer}} = 0
$$

看似最优，但 DP 帮不了"装下"（要复制模型，每卡一份完整权重）也降不了单请求延迟。故推理 DP 是"装下后"的叠加。

### 4.4 PP 的通信与 bubble

PP 每段边界 P2P 传激活，消息 `M_pp = B·S·H`，`P` 段共 `P-1` 次传递。**bubble 占比**（`M` 个 micro-batch）：

$$
\text{bubble ratio} = \frac{P-1}{M + P - 1}
$$

`M` 越大 bubble 越小。训练靠大 micro-batch 数填满；推理 batch 小，`M` 小，bubble 暴露 → PP 推理少用。

### 4.5 EP 的 All-to-All 通信量

EP 每 MoE 层 2 次 All-to-All（派发 + 收回），消息 `M_ep = B·S·H`（每 token 派发 + 收回各一份）。All-to-All 每节点发送/接收 `(N-1)/N · M`：

$$
T_{\text{comm,EP-per-MoE-layer}} \approx 2 \cdot \frac{2(N-1) \cdot B \cdot S \cdot H}{N \cdot \beta}
$$

（2 次 All-to-All，每次 reduce-scatter-like 发收各 `(N-1)/N·M`）。EP 的 All-to-All 是**全互联**（每对节点都可能传），对网络拓扑敏感（需 All-to-All 拓扑优化的 NCCL / NVSwitch）。

### 4.6 四维通信量对比表

| 维度 | 通信原语 | 每层次数 | 消息大小 | 总通信量（每 forward） | decode 时（S=1） |
|---|---|---|---|---|---|
| **TP** | all-reduce | 2/层 | `B·S·H` | `4L·(N-1)/N·B·S·H` | **极小**（`B·H`） |
| **PP** | P2P send/recv | `P-1` | `B·S·H` | `(P-1)·B·S·H` + bubble | 中（有 bubble） |
| **DP** | 无 | 0 | — | 0 | **0** |
| **EP** | All-to-All | 2/MoE层 | `B·S·H` | `4L_moe·(N-1)/N·B·S·H` | 中（`B·H`，但全互联） |

> [!tip] 选型口诀
> **装不下用 TP（单机）/PP（跨机）切权重；MoE 用 EP 切 expert；装下后用 DP 堆副本扩吞吐；单请求降延迟用 TP。decode 时 TP/EP 通信极小，是推理 TP 主流的关键。**


## 5. 代码示例（可选）

### 5.1 vLLM 多维并行启动

```bash
# 单机 8 卡 TP (最常见)
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 8

# 跨 2 节点 (每节点 8 卡): TP=8 (单机内), PP=2 (跨节点), Ray 后端
# 节点0 (head):
ray start --head --port=6379
# 节点1 (worker):
ray start --address=<head_ip>:6379
# 任一节点起服务:
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray

# DP 扩吞吐 (2 副本, 单进程内):
vllm serve <model> --data-parallel-size 2 --tensor-parallel-size 4

# MoE 用 EP (DeepSeek-V3 类):
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel   # expert 切到 8 卡 (而非 TP 切 expert)
```

### 5.2 TP 切分层算子（概念，仿 vLLM `ColumnParallelLinear`）

```python
import torch
import torch.nn.functional as F
import torch.distributed as dist

class ColumnParallelLinear:
    """列并行: W 按 out 维切 N 份, 每卡一份, 输入 X 全卡同, 输出 Y_i 切片 (无需通信)"""
    def __init__(self, in_features, out_features, tp_size, tp_rank):
        assert out_features % tp_size == 0
        self.out_shard = out_features // tp_size
        # 本卡只持有 W 的第 tp_rank 片
        self.W = torch.randn(in_features, self.out_shard)  # 实际从权重文件按 rank 切加载
        self.tp_rank, self.tp_size = tp_rank, tp_size

    def forward(self, X):
        return X @ self.W  # 各卡算自己那 out_shard 列, 互补重叠

class RowParallelLinear:
    """行并行: W 按 in 维切, X 也按 in 维切, 各卡算部分和 -> all-reduce 聚合"""
    def __init__(self, in_features, out_features, tp_size, tp_rank):
        assert in_features % tp_size == 0
        self.in_shard = in_features // tp_size
        self.W = torch.randn(self.in_shard, out_features)  # 本卡第 tp_rank 行片
        self.tp_rank, self.tp_size = tp_rank, tp_size

    def forward(self, X_shard):  # X 已按 in 维切到本卡
        Y_partial = X_shard @ self.W  # 部分和
        dist.all_reduce(Y_partial)    # 聚合 -> 完整 Y
        return Y_partial

# 一个 transformer 层的 TP 流水 (概念)
# attn:  QKV = ColumnParallelLinear(H, 3H)  (无通信)
#        o   = RowParallelLinear(H, H)     (all-reduce ①)
# mlp:   gate_up = ColumnParallelLinear(H, 2*FFN) (无通信, 融合 gate+up)
#        down    = RowParallelLinear(FFN, H)     (all-reduce ②)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Tensor Parallel]]（Megatron 列/行并行算法）、[[Pipeline Parallel]]（按层切）、[[Data Parallel]]（副本）、[[专家并行]]（MoE expert 切）、[[MoE]] / [[MoE路由]]（EP 的 gate + 派发）、[[worker与executor]]（TP/PP worker 的封装）、[[model runner]]（每卡 runner 跑自己那片）、[[NCCL通信拓扑]]（all-reduce / All-to-All 后端）、[[alpha-beta性能模型]]（通信公式）。
- **下游（应用）**: vLLM/SGLang 多卡推理 serving、超大模型部署、MoE 推理、单请求低延迟、PD 分离池内 TP/EP 组合、[[goodput]]（并行选型影响吞吐与延迟）、[[TTFT与TPOT]]（TP 降单请求延迟）。
- **对比 / 易混**:
  - **推理并行 vs 训练并行**：训练侧 DP 主流（梯度通信被大批 token 摊薄）；推理侧 TP 主流（无梯度通信 + TP decode 通信极小 + TP 能装下能降单请求延迟）。见 4.2/4.3 推导。
  - **TP vs [[Pipeline Parallel]]**：TP 切每层权重（all-reduce），PP 切层（P2P + bubble）。TP 通信频繁但每步小、单机 NVLink 友好；PP 通信少但有 bubble、跨节点才用。
  - **TP vs [[专家并行]]**：TP 切 head/hidden 维（dense 层），EP 切 expert 维（MoE 层）。正交，可组合（expert 内仍 TP）。
  - **DP 推理 vs DP 训练**：训练 DP 要 all-reduce 梯度；推理 DP 零通信。同名不同本。
  - **PD 分离 vs 四维并行**：PD 分离是部署架构（池分离），不切权重；四维是模型切分。正交。
  - **EP 推理 vs EP 训练**：训练 EP 有 forward+backward 的 All-to-All（双向、含梯度）；推理 EP 只 forward 双向（派发+收回），无梯度。详见 [[MoE token dispatcher]]。


## 7. 常见误区与易错点

> [!warning] 误区 1：推理 DP 和训练 DP 一样要 all-reduce
> 推理无反向，DP 副本 forward 独立，**零通信**。训练 DP 才 all-reduce 梯度。同名不同本。

> [!warning] 误区 2：推理 TP 通信很贵
> decode 时 `S=1`，all-reduce 消息 `B·H` 极小（MB 级），在 NVLink 上微秒级，**几乎免费**。贵的是 prefill 时 `S=L_p`，但 chunked prefill 切块后也变小。TP 推理"贵"是相对训练的误解。

> [!warning] 误区 3：DP 能帮装下大模型
> DP 是**复制**模型，每卡一份完整权重，更费显存。装不下要用 TP/PP/EP **切分**。DP 只在装下后扩吞吐。

> [!warning] 误区 4：TP size 可以任意大
> 受 **GQA 的 `num_kv_heads`** 约束（KV head 不够会被迫复制）。也受单机 GPU 数、NVLink 拓扑约束。跨节点 TP 性能差（all-reduce 走 IB），跨节点应改 PP。

> [!warning] 误区 5：PP 推理和 PP 训练一样高效
> PP 训练靠大 micro-batch 数填满流水线掩盖 bubble；推理 batch 小、延迟敏感，bubble 暴露。故 PP 推理少用，只在超大模型跨节点时用。

> [!warning] 误区 6：PD 分离是一种并行维度
> PD 分离是**部署架构**（prefill/decode 池分离），不切模型权重，与 TP/PP/DP/EP 正交。PD 池内仍可开 TP/EP。

> [!warning] 误区 7：EP 的 All-to-All 和 TP 的 all-reduce 通信量同阶
> EP 是 **All-to-All（全互联）**，每对节点都可能传，对网络拓扑敏感（需 NVSwitch/All-to-All 优化）；TP 是 **all-reduce（ring/tree）**，对单机 NVLink 友好。EP 跨节点更挑网络。

> [!warning] 误区 8：开了 EP 就不用 TP
> EP 切 expert 维，TP 切 head/hidden 维，**正交可组合**。MoE 模型可同时 EP（切 expert）+ TP（切每个 expert 的权重列/行）。vLLM 的 `--enable-expert-parallel` 控制 MoE 层是否走 EP，dense 层仍 TP。


## 8. 延伸细节

### 8.1 vLLM 的 custom all-reduce vs NCCL

vLLM 默认用自研 **custom all-reduce kernel**（基于 NVLink/SHM 的低延迟实现，针对小消息优化，适合 decode 的 `B·H` 级 all-reduce）。`--disable-custom-all-reduce` 回退 NCCL（对大消息、跨节点更稳）。这是 vLLM 在 TP decode 性能优化的关键之一。

### 8.2 EP 与 grouped GEMM

EP 派发后，每卡对收到的 token 跑自己那组 expert，用 **grouped GEMM**（一次 kernel 调用算多个 expert 的 GEMM，避免 per-expert kernel launch 开销）。详见 [[grouped GEMM]]。DeepSeek-V3 类超大 MoE 推理性能高度依赖 EP + grouped GEMM 的协同。

### 8.3 PD 分离与 KV 传输

vLLM 的 `--kv-transfer-config` 与 SGLang 的 PD 分离都涉及跨实例 KV cache 传输（共享内存 / NCCL / RDMA GDR）。这是 PD 分离的核心难点（KV 大、传输延迟、与 decode 节奏对齐）。详见 [[PD分离]]、[[KV cache management]]。

### 8.4 内容来源

并行选型与通信公式整理自 vLLM 官方文档《Distributed Inference and Serving》（核实 `--tensor-parallel-size`/`-pp`/`-dp`/`--enable-expert-parallel`/`--distributed-executor-backend`/`--kv-transfer-config`）、vLLM 源码 `vllm/distributed/parallel_state.py` 与 `vllm/model_executor/layers/linear.py`、SGLang 源码 `sglang/srt/layers/linear.py`、Megatron-LM 张量并行论文、[[Tensor Parallel]] / [[专家并行]] 笔记。PD 分离细节见 [[PD分离]]。

---

相关: [[Tensor Parallel]] | [[Pipeline Parallel]] | [[Data Parallel]] | [[专家并行]] | [[PD分离]] | [[continuous batching的调度实现]] | [[model runner]] | [[worker与executor]] | [[KV cache management]] | [[MoE]] | [[MoE路由]] | [[grouped GEMM]] | [[PagedAttention]] | [[chunked prefill]] | [[TTFT与TPOT]] | [[goodput]] | [[NCCL通信拓扑]] | [[alpha-beta性能模型]] | [[16-推理引擎源码]]
