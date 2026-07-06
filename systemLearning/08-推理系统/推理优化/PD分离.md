# PD分离

> **所属章节**: [[推理优化]]
> **所属模块**: [[08-推理系统]]
> **难度**: 中高级

## 1. 一句话定义

**PD分离（Prefill-Decode Disaggregation，预填充-解码分离，PD disaggregation，disaggregated serving）** 是 LLM 推理服务的一种**部署架构**：把一次推理的两阶段——**prefill（处理 prompt，compute-bound）** 与 **decode（逐词生成，memory-bound）**——物理分离到**不同的 GPU/实例**上，各自独立调度、独立批处理，避免两阶段在同一 batch 里互相拖累；prefill 实例算完 KV cache 后通过高速互联（NVLink/RDMA/IB）把 cache 迁移给 decode 实例持续生成，从而同时压低 **TTFT（首 token 延迟）**、稳定 **TBT（token 间延迟）**、提升总吞吐。

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
