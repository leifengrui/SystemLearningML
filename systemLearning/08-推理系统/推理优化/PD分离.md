# PD分离

> **所属章节**: [[推理优化]]
> **所属模块**: [[08-推理系统]]
> **难度**: 中高级

## 1. 一句话定义

**PD分离（Prefill-Decode Disaggregation，预填充-解码分离，PD disaggregation，disaggregated serving）** 是 LLM 推理服务的一种**部署架构**：把一次推理的两阶段——**prefill（处理 prompt，compute-bound）** 与 **decode（逐词生成，memory-bound）**——物理分离到**不同的 GPU/实例**上，各自独立调度、独立批处理，避免两阶段在同一 batch 里互相拖累；prefill 实例算完 KV cache 后通过高速互联（NVLink/RDMA/IB）把 cache 迁移给 decode 实例持续生成，从而同时压低 **TTFT（首 token 延迟）**、稳定 **TBT（token 间延迟）**、提升总吞吐。
> [!note] 解答：为什么把 GPU 叫做"实例"？
> 因为这里"实例（instance）"指的**不是物理 GPU 这个硬件单位，而是一个独立运行的推理服务进程/部署单元**——是**逻辑调度单位**，不是硬件单位。一个实例背后可能挂 1 张卡，也可能挂多张卡（TP/PP 并行）。叫"实例"是云原生/服务部署语境的 borrow，强调的是"可被路由器调度、可独立扩缩容、可独立打 SLO"的那个东西。
>
> ### 为什么不直接说"GPU"
> 1. **一个实例 ≠ 一张 GPU**：vLLM 起 `--tensor-parallel-size 8` 时，一个实例就是 8 张卡的进程组（一个推理服务单元）；起 `--tensor-parallel-size 1` 时一个实例才 = 一张卡。如果话术写"把请求分到不同 GPU"，TP=8 的实例到底算 1 个还是 8 个？说不清。叫"实例"把"一个对外提供推理能力的单元"圈死，与硬件解耦。
> 2. **调度/路由/负载均衡的逻辑对象是实例**：PD 分离里 P 池/D 池由若干实例组成，路由器把请求分到"实例"（一个进程/一个 pod），不是分到"某张卡"。实例内部怎么用卡（TP/PP 切分）是它自己的事，路由器不关心。这与 Kubernetes pod、云上 EC2 instance 的"一个可调度服务单元"概念完全一致。
> 3. **可独立扩缩容**：D 池不够就"加实例"（再加一个 vLLM/SGLang 进程），而不是"加 GPU"——加实例可能复用现有卡（colocate）也可能新加卡。运维语言里"实例数"才是可操作的伸缩量。
> 4. **与 weight sync / 版本管理对齐**：[[policy deployment (PD)]] / [[训推分离]] 里推新 $\theta$ 是"按实例 sync"，一个实例对应一组权重版本，是版本管理的最小单位。说"给 GPU sync 权重"在 TP=8 时歧义（8 张卡是一组还是 8 组？）。
>
> ### 一张图厘清三层概念
> ```
> 物理层:   GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7   (8 张物理卡)
>              └─────────────┘ └──────────────────┘
> 实例层:        实例 A (TP=4)      实例 B (TP=4)        (2 个推理实例)
> 池层:          ┌──── P 池 ────┐   ┌──── D 池 ────┐
>                实例 A 属 P 池      实例 B 属 D 池
> 路由层:   调度器把 prompt 请求 → P 池里的某实例；KV cache 迁移 → D 池里的某实例
> ```
> 所以"实例"= 一个**对外提供推理能力的进程组**，背后挂 ≥1 张 GPU；P 池/D 池 = 实例的集合；路由器 = 在实例间分发请求。
>
> ### 术语对齐表（同义词）
> | 术语 | 同义/近义 | 出处 | 粒度 |
> |---|---|---|---|
> | **instance（实例）** | worker / serving unit / pod | vLLM/SGLang/K8s | 一个推理服务进程组 |
> | **GPU / device** | 物理 GPU | nvidia-smi | 一张物理卡 |
> | **worker（vLLM 语义）** | ≈ 实例（TP=1 时=1 GPU；TP>1 时=多 GPU 进程组） | vLLM/SGLang | 一个推理单元 |
> | **replica** | ≈ 实例（K8s/微服务语境） | K8s Deployment | 一个 pod 副本 |
>
> 注意 vLLM/SGLang 文档里 "worker" 和 "instance" 常混用，都指"一个推理服务单元"；K8s 语境叫 "pod/replica"；硬件语境才叫 "GPU/device"。PD 分离讲的是**实例级**调度，所以用"实例"而非"GPU"。
>
> **一句话**：叫"实例"是为了把"逻辑服务单元"和"物理 GPU"解耦——一个实例可挂多卡（TP），路由/扩缩容/版本管理都在实例这一层做，不直接操作物理卡。
> [!warning] 名字撞车：两个 "PD"
> 在 LLM 工程语境里 "PD" 有两个完全不同的含义，必须分清：
> - **本条 PD = Prefill-Decode 分离**（推理系统内部，把推理的两阶段分开部署）。
> - **训推分离 PD = Policy Deployment**（训练与推理部署分离，见 [[policy deployment (PD)]]、第二十章）。
> 二者缩写相同、领域不同。KV cache 笔记里读到 prefill/decode 两阶段后提的"pd分离"指的是**本条**。

## 2. 为什么需要它（动机与背景）

LLM 推理一次请求分两阶段（见 [[KV cache]] §3.1）：

| 阶段 | 算什么 | 算力特征 | 访存特征 |
|---|---|---|---|
| **prefill** | 一次处理整段 prompt，算所有位置的 K/V 存 cache | **compute-bound**（一次算 $n_0$ 个 token，FLOPs $\propto n_0^2$） | 算力打满，MFU 高 |
| **decode** | 逐词生成，每步算 1 个 token 的 Q/K/V，用 cache 做注意力 | **memory-bound**（每步 FLOPs 少，但要读全部权重 + cache） | 算力大量空闲，带宽是瓶颈 |

**混合 batching 的问题**（[[continuous batching]] 不分离时）：把 prefill 和 decode 请求塞进同一个 batch、同一张卡上跑，会**互相拖累**：

1. **prefill 阻塞 decode**：prefill 算力大、耗时长（长 prompt 可达几十~几百 ms），同 batch 的 decode 请求要等 prefill 算完才能继续出下一词 → decode 的 TBT 被拉高、输出卡顿。
2. **decode 拖慢 prefill**：decode 是 memory-bound，算力空闲，但它占着 SM/显存带宽，让同 batch 的 prefill 无法把算力打满 → prefill 的 TTFT 变长、吞吐降。
3. **batch 拼不齐**：prefill 长度差异大、decode 长度不可预测，混合 batch 的 padding 浪费严重，调度复杂。

根因：**两阶段的资源画像相反**——prefill 要算力、decode 要带宽。混在一起谁也跑不快。

**PD 分离的解法**：把 prefill 和 decode 分别放到**专门调优的实例**上：
- prefill 实例：高算力配置，专跑 prefill，MFU 打满、吞吐高。
- decode 实例：高带宽/大显存配置，专跑 decode，大 batch 高吞吐、TBT 稳。
- 两阶段间通过 **KV cache 迁移**衔接。

这样每阶段都在自己最优的工作点运行，互不干扰。
> [!note] 解答：这两个比例一样吗？1 个 node 8 卡 prefill + 8 卡 decode 会怎样？P 会阻塞 D 吗？
> 三个子问题分开答，结论先行：**§3.4 和 §4 提到的"P:D 比例"是同一个概念（都是 P 池与 D 池的资源配比），不是两套比例**；**1:1（8P:8D）合不合理完全看流量画像，偏生成就偏少、偏短输入就够用**；**分离架构的核心目的恰恰是消除 P 阻塞 D，分离后 P 不会直接阻塞 D，但会冒出"迁移延迟"和"池间失衡"两类新问题**。下面逐条拆。
>
> ### 一、"这两个比例"是不是同一个？
> 是同一个。§2 末尾的"prefill 实例高算力配置 / decode 实例高带宽配置"只是在描述**两池配置方向不同**（P 池堆算力、D 池堆带宽/显存），没给具体比例；§3.4 和 §4 的"P:D 算力比通常 1:2~1:5"才是同一条结论的两次表述。所以全文只有**一个 P:D 比例**，指"P 池分配多少资源 vs D 池分配多少资源"。这个比例既可指**卡数比**（粗），也可指**算力/带宽资源比**（细），实操中常粗按卡数分、细按算力调。
>
> ### 二、1 个 node 8 卡 prefill + 8 卡 decode（1:1）会怎样？
> 先说配置本身：**这是 PD 分离的甜点配置之一**——同 node 机内 NVLink（400–900 GB/s）做 KV cache 迁移，跨机带宽瓶颈不存在，迁移延迟最小（见 §4 实例：70B 8k context ~2.5GB cache，NVLink 400GB/s → ~6ms，可接受）。所以"1 node 8P8D"在拓扑上很理想。
>
> 但 1:1 这个**比例**合不合理，看流量画像：
>
> | 流量画像 | 算力需求 | 1:1 表现 | 建议 |
> |---|---|---|---|
> | **偏生成**（短 prompt 长输出，如 chatbot/写作） | D 池算力/带宽需求 >> P | P 池空转、D 池排队堆积 → 用户等 | 改 1:2~1:4，D 多 P 少 |
> | **偏 prompt**（长 prompt 短输出，如 RAG 摘要/分类） | P 池算力需求 >> D | D 池空转、P 池排队 → TTFT 拉高 | 改 2:1~3:1，P 多 D 少 |
> | **均衡**（prompt/生成长度相近） | 两池相当 | 1:1 合适，但仍是起点需动态调 | 1:1 起步，监控队列调 |
> | **长上下文高并发**（128k context） | 迁移开销主导 | 1:1 够算力，但单次 KV cache 迁移 40GB 跨机几百 ms | 上 [[KV cache management]] §池化 + 预取，比例次要 |
>
> 工业经验值 P:D 常见 **1:2~1:5**（因为多数服务是 chatbot/RAG 偏生成，D 占多）。1:1 是"中性起点"，**没有流量画像数据就上 1:1 是拍脑袋**，跑起来一定要看两池队列长度动态调（§3.4、§7 误区 5）。
>
> ### 三、会不会出现"P 阻塞 D"？
> **分离架构的核心目的就是消除 P 阻塞 D**——这正是 §2 讲的"混合 batching 的问题"（prefill 和 decode 同 batch 互相拖累）。分离后 P 池和 D 池是不同实例、不同 batch、不同调度，**P 在算 prefill 时不会卡住 D 的 decode 步**，TBT 稳定。这是分离相对混合批的根本收益。
>
> 但分离后**不会出现 P 阻塞 D，却会出现下面这些替代问题**（这才是要担心的）：
>
> 1. **KV cache 迁移延迟**（最常被误读成"P 阻塞 D"）：P 算完 cache 要传给 D，D 要等 cache 到了才能开始 decode。这段迁移是串行依赖，延迟 = $M_{KV}/\text{BW}$（§4 公式）。机内 NVLink 几 ms 可忽略；跨机 IB 几十~几百 ms 显著。**这不是 P 阻塞 D，是 D 在等 cache 到位**。治法：预取（prefill 算的同时提前推 cache）、部分迁移、池化就近取回。
> 2. **D 池饱和排队**：D 池并发满了，新请求的 cache 迁过来也没 D 实例接 → 在调度器里排队 → TBT 拉高。**这不是 P 阻塞 D，是 D 自己过载**。治法：加 D 实例、调高 D 池比例、D 池内 continuous batching 拼更大 batch。
> 3. **P 池饱和 → D 池空等**：P 算不过来，D 池没活干 → 整体吞吐被 P 卡住。**这是 P 拖整体，不是 P 阻塞 D**。治法：加 P 实例、调高 P 池比例。
> 4. **跨机带宽瓶颈**：P 和 D 跨 node 时，IB/RDMA 带宽（50–200 GB/s）远低于机内 NVLink，长 context 迁移成为瓶颈——这是规模化时分离架构的核心痛点，也是 Mooncake/DistServe 要做 KV cache 池化的根因（§8.2）。
>
> ### 四、所以 8P8D 在 1 个 node 上，最可能看到什么？
> - **拓扑好**：机内 NVLink 迁移快，迁移延迟小，PD 分离收益能兑现。
> - **比例要验**：跑起来看 `vLLM /metrics` 的 P 池队列长度、D 池队列长度、迁移延迟分布。P 队列长 → 加 P；D 队列长 → 加 D；两边都不长但延迟高 → 看是不是迁移/decode kernel 问题。
> - **不会 P 阻塞 D**：但会出现"D 等 cache 到位"和"D 池排队"两类现象，监控时别误判成"P 阻塞 D"而去做错优化（比如去给 P 加算力，其实 D 才是瓶颈）。
> - **1 node 装满 16 卡是重资产**：真要单 node 16 卡，多半是 H100/A100 8 卡 node × 2 做跨 node PD，那时跨机带宽才是主要矛盾，建议上池化架构。
>
> **一句话**：1:1 不是错，是起点；分离后 P 不阻塞 D，但你得监控两池队列动态调比例，并治迁移延迟/池化跨机带宽这两个真问题。详见 §3.4 调度与负载均衡、§4 资源配比、§7 误区 5/6。
## 3. 核心概念详解

### 3.1 整体架构

```
        ┌────────────┐   KV cache 迁移   ┌────────────┐
prompt →│  Prefill   │ ───────────────→ │   Decode   │ → 逐词输出
        │  Worker P  │  (NVLink/RDMA)   │  Worker D  │
        └────────────┘                   └────────────┘
         compute-bound                    memory-bound
         高算力、高 MFU                    大显存、大 batch、高带宽
```

- **Prefill Worker（P 池）**：接收 prompt，跑 prefill，产出该请求的 KV cache。
- **Decode Worker（D 池）**：接收迁移来的 KV cache，持续 decode 到生成结束。
- **KV cache 迁移**：P 把算好的 cache 通过互联传给 D，是分离架构的核心开销。

### 3.2 为什么能各自变快

- **P 池**：只跑 prefill，请求都是 compute-bound，可以拼大 batch 把算力打满（高 MFU）。不会被 decode 的空闲算力拖累。
- **D 池**：只跑 decode，请求都是 memory-bound，可以用**超大 batch**（几百路并发）摊销权重读取——读一次权重算几百个 token，带宽利用率拉满、吞吐飙升。不会被 prefill 的长计算阻塞。
- **TBT 稳定**：D 池里没有 prefill 抢资源，每步 decode 时间稳定，不会周期性卡顿。

### 3.3 KV cache 迁移

分离的关键代价：prefill 算出的 KV cache 要从 P 搬到 D，**迁移量 = 该请求的 KV cache 显存**（见 [[KV cache]] §4 显存公式）：

$$
M_{\text{KV}} = 2 \cdot L \cdot n_{\text{kv}} \cdot n_{\text{seq}} \cdot d_h \cdot b
$$

- 长上下文（32k/128k）下 $M_{\text{KV}}$ 可达几十 GB，迁移延迟显著（即使 NVLink 900 GB/s，几十 GB 也要几十 ms）。
- 优化：**KV cache 池化**（Mooncake 的 KVCache pool，P/D 共享分层存储：GPU HBM → CPU DRAM → SSD，按需取回）、**部分迁移**、**预取**、**压缩/量化传输**（fp8/int4 cache）。

### 3.4 调度与负载均衡

- **P/D 实例比例**：prefill 与 decode 算力需求不同，要按线上流量（prompt 长度、生成长度、QPS）动态调 P:D 比例（如 1:4、2:3）。流量偏生成 → D 多；偏短输入长输出 → D 多。
- **路由**：调度器把请求分到 P 实例，prefill 完后把 KV cache 路由到某 D 实例。要考虑 D 池负载、KV cache 已驻留位置（亲和性）。
- **跨节点通信**：迁移走 NVLink（同机跨卡）或 InfiniBand/RDMA（跨机），跨机带宽远低于机内，是分离架构规模化瓶颈。

### 3.5 与 continuous batching 的关系

PD 分离**不替代** [[continuous batching]]，而是**与之叠加**：
- 不分离时：一个池子里 prefill + decode 混合做 continuous batching。
- 分离后：P 池和 D 池**各自**做 continuous batching（P 池连续拼 prefill batch，D 池连续拼 decode batch）。
- 即"阶段分离 + 阶段内连续批"两层优化。

### 3.6 与 prefix caching 的关系

- [[prefix caching]]：不同请求**共享相同前缀**的 KV cache，省 prefill。
- PD 分离：阶段**物理分离**。
- 二者正交、可叠加：P 池里可以用 prefix caching 复用前缀，D 池里也可以。分离架构常与 prefix caching 配合（Mooncake 同时做池化 + prefix 复用）。

## 4. 数学原理 / 公式

### 两阶段算力特征

设层 $L$、宽 $d$、prompt 长度 $n_0$、生成长度 $n_g$：

- **prefill**：一次算 $n_0$ 个 token，注意力是 $n_0\times n_0$ 矩阵，FLOPs $\approx c\cdot n_0^2 L d$，**compute-bound**。
- **decode**：每步算 1 个 token，对 cache 的 $n$ 个 K 做注意力，每步 FLOPs $\approx c'\cdot n\cdot L d$，$n$ 从 $n_0$ 增到 $n_0+n_g$；总 FLOPs $\approx c'\cdot \bar n\cdot n_g\cdot L d$，但每步**读全部权重** $\Theta(P)$ → **memory-bound**（算力空闲、带宽是瓶颈）。

### 关键指标

| 指标 | 含义 | PD 分离影响 |
|---|---|---|
| **TTFT**（Time To First Token） | 首 token 延迟 | 降：prefill 不被 decode 拖 |
| **TBT**（Time Between Tokens） | token 间延迟 | 稳且低：decode 不被 prefill 阻塞 |
| **吞吐**（tokens/s 或 req/s） | 单位时间产出 | 升：两阶段各自最优工作点 |

### KV cache 迁移开销

迁移延迟 $\approx M_{\text{KV}} / \text{BW}$，BW 为互联带宽：

$$
T_{\text{migrate}} \approx \frac{2 L n_{\text{kv}} n_{\text{seq}} d_h b}{\text{BW}}
$$

- NVLink（机内）$\text{BW}\approx 300\text{–}900$ GB/s。
- IB/RDMA（跨机）$\text{BW}\approx 50\text{–}200$ GB/s。
- 例：Llama-3-70B，$n_{\text{seq}}=8\text{k}$，KV cache $\approx 2.5$ GB，NVLink 400 GB/s → $\approx 6$ ms；跨机 100 GB/s → $\approx 25$ ms。长上下文 128k（40 GB）跨机迁移要几百 ms，必须靠池化/预取/部分迁移缓解。

### 资源配比

给定 QPS、平均 prompt 长 $\bar n_0$、平均生成长 $\bar n_g$，P 池算力需求 $\propto \text{QPS}\cdot \bar n_0^2$，D 池带宽需求 $\propto \text{QPS}\cdot \bar n_g\cdot \bar n$。P:D 算力比通常 $\approx 1:2\sim 1:5$（偏生成流量 D 占多），需按线上画像动态调。

## 5. 代码示例（可选）

```python
# 伪代码：极简 PD 分离调度（示意，真实系统用 vLLM/SGLang/Mooncake）

class PrefillWorker:
    """只跑 prefill，产出 KV cache 并迁移给 DecodeWorker。"""
    def __init__(self, model):
        self.model = model
    def prefill(self, prompt_ids):
        # 一次前向算所有 prompt 位置的 K/V，存 cache
        out, kv_cache = self.model.forward(prompt_ids, use_cache=True)
        return kv_cache, out[:, -1]          # cache + 最后位置 logits

class DecodeWorker:
    """持有 KV cache，逐词 decode。"""
    def __init__(self, model):
        self.model = model
    def receive_cache(self, kv_cache):
        self.cache = kv_cache                # 从 P 迁移来的 cache
    def decode(self, first_tok, max_new):
        toks = [first_tok]
        cur = first_tok
        for _ in range(max_new):
            out, self.cache = self.model.forward(cur, cache=self.cache)  # 每步 append
            nxt = out.argmax(-1)
            toks.append(nxt); cur = nxt
        return toks

class PDDisaggregatedServing:
    """P 池 + D 池 分离调度。"""
    def __init__(self, P_pool, D_pool):
        self.P_pool, self.D_pool = P_pool, D_pool
    def generate(self, prompt_ids, max_new):
        p = self.P_pool.pick_least_loaded()           # 选负载低的 P
        kv, first_tok = p.prefill(prompt_ids)
        d = self.D_pool.pick_least_loaded()           # 选负载低的 D
        d.receive_cache(kv)                           # KV cache 迁移（实际走 NVLink/RDMA）
        return d.decode(first_tok, max_new)
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[KV cache]]（分离的衔接物就是 KV cache，要能在节点间迁移）、[[continuous batching]]（分离后各池内仍用）、prefill/decode 两阶段算力特征。
- **下游（应用）**: LLM 推理服务系统（vLLM、SGLang、TensorRT-LLM、Mooncake、DistServe 的核心架构）、长上下文高并发服务、TTFT/TBT SLO 保障。
- **对比 / 易混**:
  - **PD（Prefill-Decode）分离 vs 训推分离（[[policy deployment (PD)|Policy Deployment]]）**：前者是推理内部分阶段部署，后者是训练与推理部署分离。**两个 PD 缩写撞名，见 §1 warning。**
  - **PD 分离 vs [[continuous batching]]**：前者分离阶段，后者阶段内连续拼批，正交叠加。
  - **PD 分离 vs [[prefix caching]]**：前者分离阶段，后者共享前缀 cache，正交叠加。
  - **PD 分离 vs MoE 专家分离**：MoE 是模型内部分专家，PD 是推理流程分阶段，不同层。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **把 PD 分离当成训推分离**：LLM 工程里两个 PD 撞名，Prefill-Decode（本条）≠ Policy Deployment（训推分离），见 §1 warning。
> 2. **以为分离就没 batch 了**：分离后每个池**仍然做 continuous batching**，且 batch 可以比混合时更大（因为同质请求拼批更整齐）。
> 3. **以为 KV cache 迁移免费**：迁移量 $=M_{\text{KV}}$，长上下文几十 GB，跨机迁移几十~几百 ms，是分离架构的核心开销，必须靠池化/预取/部分迁移/量化缓解。
> 4. **以为一定比混合批优**：小规模、短上下文、低并发时，分离的迁移开销可能盖过收益，混合 continuous batching 更简单。分离的优势在**高并发 + 长上下文 + 长 generate**场景才显著。
> 5. **P:D 比例拍脑袋**：比例要按线上流量画像（prompt/生成长度、QPS）动态调，固定比例会在流量变化时失衡。
> 6. **忽略跨机带宽瓶颈**：机内 NVLink 快，但规模化后 P 与 D 常跨机，IB/RDMA 带宽骤降，迁移成为瓶颈——这也是为什么需要 KV cache 池化就近取回。
> 7. **以为 D 池不用 prefill**：D 池偶尔也要处理短 prefill（如 decode 中途的新请求、prefix cache miss 的兜底），完全分离不现实，多数系统是"分离为主 + 兜底混合"。

## 8. 延伸细节

### 8.1 代表系统

- **DistServe**（2024 OSDI）：学术上系统化提出 PD disaggregation，论证分离对 TTFT/TBT/吞吐的全局收益，给出 P:D 资源配比算法。
- **Mooncake**（Moonshot/月之暗面，2024）：KV-cache-centric 架构，把 KV cache 当一等公民，P/D 间用分层 KVCache pool（GPU→DRAM→SSD）池化共享，支持 prefix 复用与按需取回，是 kimi 长上下文高并发的底层。
- **Splitwise**（微软，2024）：把 prefill 与 decode 分到不同 GPU 池，研究资源配比与迁移调度。
- **TensorRT-LLM / vLLM / SGLang**：工业推理引擎陆续支持 disaggregated prefill/decode 模式，SGLang 的 PD 分离 + KV cache pool 是代表。

### 8.2 KV cache 池化

把 KV cache 当作可迁移、可缓存、可共享的资源，存在分层池子里（GPU HBM 热层 → CPU DRAM 温层 → SSD 冷层），按请求亲和性取回。好处：
- **解耦 P 与 D 的物理位置**：D 不必和 P 同机，cache 走池子就近取。
- **prefix 复用**：相同前缀的 cache 池里只存一份，多请求共享。
- **长上下文卸载**：超长 cache 不占 D 池显存，按需从池里取回。

### 8.3 部分迁移与预取

- **部分迁移**：只迁 decode 即将用到的 cache 片段（如最近窗口），其余按需拉取，降首迁延迟。
- **预取**：prefill 还在算时，提前把已算完的 cache 层往 D 推，与计算重叠。
- **量化传输**：cache 以 fp8/int4 传，省带宽，到 D 后反量化（或直接用低精度 cache decode）。

### 8.4 与 speculative decoding 的关系

[[speculative decoding]] 用小模型/草稿加速 decode，与 PD 分离正交：D 池里仍可挂 speculative decoding。但分离后 D 池 batch 大、带宽利用率高，speculative 的收益要重新评估（大 batch 下 draft 的省算力价值降低）。

### 8.5 趋势

- **KV-cache-centric 架构**：从"模型权重为中心"转向"KV cache 为中心"，cache 的存储/迁移/共享成为推理系统的一等设计对象（Mooncake 提出）。
- **disaggregated serving 普及**：从单机 continuous batching → 跨机 PD 分离 + 池化，是长上下文 LLM 服务规模化的事实方向。
- **与 MoE、并行解码组合**：PD 分离 + [[MoE]] 专家路由 + [[parallel decoding]]，构成新一代高吞吐推理栈。

---
相关: [[推理优化]]、[[KV cache]]、[[KV cache management]]、[[continuous batching]]、[[prefix caching]]、[[policy deployment (PD)]]、[[speculative decoding]]
