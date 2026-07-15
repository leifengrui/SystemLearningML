# TTFT与TPOT

> **所属章节**: [[推理性能指标]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: TTFT / TPOT / time to first token / time per output token / 首 token 延迟 / 每 token 生成时间 / 首 token 时延 / 单 token 时延
> **难度**: 中（需懂 [[autoregressive decoding]] 的 prefill/decode 两阶段、[[batching tradeoff]]、[[KV cache]]、[[prefix caching]]、[[PD分离]]）


## 1. 一句话定义

**TTFT（Time To First Token，首 token 延迟）** = 从请求提交到**第一个输出 token 产出**的墙钟时间，主要由 **prefill 阶段**（处理整个 prompt、算 attention、填 KV cache）决定，是 **compute-bound**（算力受限）的延迟；**TPOT（Time Per Output Token，每输出 token 时间）** = decode 阶段**每生成一个新 token 的平均耗时**，主要由 **KV cache 读取带宽**决定，是 **memory-bound**（带宽受限）的延迟。两者分别衡量 LLM 推理两个阶段——TTFT 衡量"用户等多久看到首字"，TPOT 衡量"用户看到首字后每多一个字等多久"——因为 prefill 与 decode 算力特征根本不同（前者吃 SM 算力，后者吃 HBM 带宽），必须分开度量、分开优化，是 [[autoregressive decoding]] 两阶段不对称的直观数字体现，也是 [[PD分离]] 把两类负载拆到不同 GPU 池分别调优的根本依据。

> [!note] 三句话定位
> - **是什么**：TTFT=首 token 延迟（prefill，compute-bound）；TPOT=每 token 时间（decode，memory-bound）。一对指标拆开 prefill 与 decode。
> - **为什么**：prefill 吃算力、decode 吃带宽，瓶颈不同→必须分开度量。TTFT 决定"等多久看到首字"，TPOT 决定"首字后逐字等多久"。
> - **与 [[PD分离]] 关系**：分离后 prefill 池专攻 TTFT（堆算力）、decode 池专攻 TPOT（堆带宽），两指标可独立优化。TTFT/TPOT 是 PD 分离的度量基础。


## 2. 为什么需要它（动机与背景）

### 2.1 prefill 与 decode 算力特征根本不同

LLM 自回归推理分两阶段（见 [[autoregressive decoding]]）：

```
prefill (处理 prompt, 一次性):
  输入 L 个 token, 一次 forward 算完所有 token 的 Q/K/V/attention
  FLOPs ∝ L · d²  (算力大, 算力受限 compute-bound)
  SM 占满, HBM 带宽闲 (算力是瓶颈)
  -> 决定 TTFT

decode (逐 token 生成, 自回归):
  每步输入 1 token, 算 1 层 Q/K/V, 读已有 KV cache
  FLOPs ∝ 1 · d² (算力小), 但要读 全部 L_kv 个历史 KV + 权重
  读量大, 带宽受限 memory-bound
  -> 决定 TPOT
```

prefill 是**算力受限**（FLOPs 大，SM 满载），decode 是**带宽受限**（FLOPs 小但要读大量 KV + 权重，HBM 带宽是瓶颈）。两阶段瓶颈不同→若用一个总延迟指标平均掉，看不出各自瓶颈→必须拆成 TTFT（prefill）与 TPOT（decode）两个指标，分别诊断。

### 2.2 用户体验的两个感知点

用户与 LLM 交互，感知延迟分两段：

```
用户发 prompt ───────────── 首 token 出 ──── 第2 token ──── 第3 token ── ... ── 末 token
                ↑──── TTFT ────↑
                               ↑── TPOT ──↑── TPOT ──↑ ... ── TPOT ──↑
```

- **TTFT**：用户发完 prompt 到"看到第一个字"的等待。这期间屏幕空白，用户最焦虑。TTFT 高→用户以为卡死。
- **TPOT**：首字之后每多一个字的间隔。流式输出时用户看到字一个个蹦，TPOT 决定"蹦得多快"。TPOT 高→字蹦得慢、卡顿。

两者体感完全不同，一个管"等多久开始"，一个管"开始后多快"。必须分开度量、分开定 SLO。

### 2.3 优化手段不同→指标必须拆

- **降 TTFT** 的手段：堆 GPU 算力（TP）/[[prefix caching]] 复用前缀 KV（免重算 prefill）/[[chunked prefill]] 切块让首 chunk 早出 token / 缩短 prompt / 用更短上下文模型。
- **降 TPOT** 的手段：堆 HBM 带宽 /[[continuous batching]] 加大 batch 摊销权重读取 /[[PD分离]] decode 池用带宽型卡 /[[量化]] 压 KV 与权重字节数 /[[PagedAttention]] 降 KV 读取量。

TTFT 优化打"算力"，TPOT 优化打"带宽"，手段几乎不重合。若只看一个总延迟，无法定位该优化谁。拆成 TTFT/TPOT 是为了**可定位、可分别优化**。

### 2.4 PD 分离的度量依据

[[PD分离]]（prefill/decode disaggregation）把 prefill 与 decode 拆到不同 GPU 池——prefill 池堆算力型卡（降 TTFT）、decode 池堆带宽型卡（降 TPOT）。这种拆分之所以合理，正因 TTFT/TPOT 瓶颈不同。TTFT/TPOT 是 PD 分离的**度量基础与动机依据**：没有这两个指标分拆，就无法量化"分离后 prefill 池 TTFT 降了多少、decode 池 TPOT 降了多少"。


## 3. 核心概念详解

### 3.1 TTFT 的构成

```
TTFT = T_queue + T_prefill(L_prompt, hit_prefix)

T_queue      = 请求在 scheduler waiting 队列等待被 admit 的时间
               (并发高 / batch 满 / KV 不够 -> 排队)
T_prefill    = 实际 prefill 计算 prompt 的时间
               (含 prefix cache 命中部分免算)
```

- **T_queue（排队时间）**：请求进 [[vLLM scheduler]] 的 waiting 队列，等被 admit 进 running batch。高并发时 batch 满 / KV cache 不够（free block 不足）→ 请求排队，TTFT 涨。是调度层延迟。
- **T_prefill（prefill 算力时间）**：算 prompt 的 forward 时间，与 prompt 长度 $L$、模型隐藏维度 $d$、层数 $n_l$、GPU 算力相关。是算力层延迟。[[prefix caching]] 命中时命中部分免算（只算差量），TTFT 大降。

### 3.2 TTFT 与 prompt 长度的关系

prefill 是 compute-bound，FLOPs ∝ $L \cdot d^2 \cdot n_l$，算力时间 ∝ FLOPs / GPU_FLOPS：

```
prompt 4k:   prefill ~0.2s (A100, 7B 量级)  -> TTFT ~0.2s (+queue)
prompt 8k:   prefill ~0.4s
prompt 32k:  prefill ~1.6s
prompt 128k: prefill ~6s+  (长 prompt TTFT 明显涨)
```

（数字为量级估计，具体随模型/卡/精度变，待核实。但趋势：TTFT 随 prompt 长度近线性涨，因 prefill compute-bound。）

### 3.3 TTFT 与 prefix caching 的关系

[[prefix caching]]（如 vLLM RadixTree、SGLang RadixAttention）复用已算过的前缀 KV。命中时只算"差量 prompt"（system prompt + few-shot 等公共前缀免算）：

```
无 prefix cache:  T_prefill(L_full)        ∝ L_full
有 prefix cache (命中 L_hit): T_prefill ∝ (L_full - L_hit)   只算差量
  -> TTFT 降 (公共前缀免算)
```

system prompt 长（2k）、few-shot 多、多轮对话（历史长）→ prefix cache 命中率高 → TTFT 大降。是 TTFT 优化的核心手段（不改算力，复用算过）。

### 3.4 TPOT 的构成

```
TPOT = T_decode_iter   (decode 阶段一个 iteration 的耗时, 每 iteration 产 1 token/请求)

T_decode_iter = max( T_compute(B), T_memory(B, L_kv) )

T_compute = B · d² · n_l / GPU_FLOPS          (B=batch size, 算 decode forward 的 FLOPs)
T_memory  = (weight_bytes + B · L_kv · d · kv_bytes) / HBM_BW   (读权重 + KV cache)
```

decode 每 iteration 给 batch 内每个请求产 1 token，所以 TPOT = T_decode_iter。decode 是 memory-bound，通常 T_memory > T_compute → TPOT 由 T_memory 决定。

### 3.5 TPOT 与 batch size 的关系（batching tradeoff）

```
B=1  (单请求):  读权重 + 1·L_kv·d·kv_bytes   -> 权重读独占, 带宽浪费, TPOT 高
B=64 (大 batch): 读权重 + 64·L_kv·d·kv_bytes -> 权重读被 64 摊销, 但 KV 读量涨
  -> TPOT 随 B 先降(权重摊销)后升(KV 读涨), 有最优 B
```

这是 [[batching tradeoff]] 的核心：batch 大→权重读取摊销→单 token 算力/带宽成本降→吞吐涨；但 batch 大→单 iteration KV 读量大→T_memory 涨→TPOT 涨→延迟涨。goodput（见 [[goodput]]）就是在这个 tradeoff 找平衡点。

### 3.6 TPOT 与 KV cache 读取带宽

decode 每步要读：① 模型权重（固定，一次）② 该请求已生成全部 KV（随生成长度涨）。decode 是 memory-bound，TPOT ≈ 读量 / HBM 带宽：

```
权重读: n_l · d² · bytes_per_param   (固定, ~模型参数量×字节)
KV 读:  B · n_l · 2 · L_kv · d · kv_bytes  (B 请求, 每请求 L_kv 历史, K+V 各一)
HBM_BW: A100 ~1.5TB/s, H100 ~3.35TB/s (待核实)

TPOT ≈ (权重读 + KV 读) / HBM_BW   (memory-bound 时)
```

所以 [[量化]]（压 KV 字节，如 FP8→1byte）降 KV 读量→降 TPOT；[[PD分离]] decode 池用大带宽卡→降 TPOT。都是打带宽。

### 3.7 两指标对比表

| 维度 | TTFT | TPOT |
|---|---|---|
| 全称 | Time To First Token | Time Per Output Token |
| 衡量阶段 | prefill（首 token） | decode（每后续 token） |
| 瓶颈类型 | **compute-bound**（算力/SM） | **memory-bound**（HBM 带宽） |
| 主决定因素 | prompt 长度 + GPU 算力 + [[prefix caching]] 命中 | batch size + KV cache 读量 + HBM 带宽 |
| 用户体感 | "等多久看到首字"（空白期焦虑） | "首字后逐字多快"（流式顺滑度） |
| 优化打 | 算力（TP）/prefix 复用/chunked prefill | 带宽/量化/continuous batching 摊销 |
| 典型量级 | 0.2s(4k prompt) ~ 6s(128k) | 20ms ~ 100ms+/token |
| SLO 常见 | P99 TTFT < 2s | TPOT < 100ms |
| PD 分离后 | prefill 池专攻 | decode 池专攻 |

### 3.8 测量方式（流式接口）

```
流式 API (SSE / chunked transfer):

client 发请求 ── T0
                 │
                 │  (prefill + queue)
                 ▼
              首 token 到达 ── T1   -> TTFT = T1 - T0
                 │
                 │  (decode)
                 ▼
              第2 token ── T2   -> ITL_1 = T2 - T1  (见 [[ITL]])
                 │
                 ▼
              第3 token ── T3   -> ITL_2 = T3 - T2
                 ...
              末 token ── Tn
                 -> TPOT = (Tn - T1) / (n - 1)   (平均 decode token 间隔)
```

- **TTFT**：流式接口首字节/首 token 到达时间 - 请求发出时间。
- **TPOT**：decode 阶段所有 token 间隔的**平均**（末 token 时间 - 首 token 时间）/ (token 数 - 1)。注意 TPOT 是**平均**，不看抖动；抖动看 [[ITL]]，长尾看 [[P99延迟]]。


## 4. 数学原理 / 公式

### 4.1 TTFT 公式

$$\text{TTFT} = T_{\text{queue}} + T_{\text{prefill}}(L_{\text{eff}})$$

其中 $L_{\text{eff}}$ 是**有效算力长度**（扣除 prefix cache 命中后实际需算的 prompt 长度）：

$$L_{\text{eff}} = L_{\text{prompt}} - L_{\text{hit}} \quad (\text{prefix cache 命中 } L_{\text{hit}})$$

prefill 是 compute-bound，算力时间 ∝ FLOPs / 算力：

$$T_{\text{prefill}}(L_{\text{eff}}) \approx \frac{2 \cdot L_{\text{eff}} \cdot n_l \cdot d^2}{\text{FLOPS}_{\text{GPU}} \cdot \text{util}}$$

- $2 \cdot L_{\text{eff}} \cdot n_l \cdot d^2$：prefill 的 FLOPs（每 token 每层约 $2d^2$ 算力，含 Q/K/V 投影 + attention + FFN，乘层数 $n_l$ 与 token 数 $L_{\text{eff}}$）。
- $\text{FLOPS}_{\text{GPU}}$：GPU 峰值算力（如 A100 FP16 ~312 TFLOPS）。
- $\text{util}$：算力利用率（[[SM utilization]] / [[GPU utilization]]，受 kernel 效率、batch 影响，典型 30%~60%）。

> [!note] 推导要点
> prefill 一次处理 $L_{\text{eff}}$ 个 token，每层每 token 算 Q/K/V 投影（$3 \cdot d \cdot d$）+ attention（$\sim d$）+ FFN（$\sim 2d^2$），量级 $\approx 2d^2$ FLOPs/token/layer。总 $\approx 2 L_{\text{eff}} n_l d^2$。除以算力得时间。compute-bound 时这步是瓶颈。

### 4.2 prefix caching 对 TTFT 的降

无 prefix cache：

$$\text{TTFT}_{\text{no\_cache}} = T_{\text{queue}} + \frac{2 L_{\text{prompt}} n_l d^2}{\text{FLOPS} \cdot \text{util}}$$

有 prefix cache 命中 $L_{\text{hit}}$：

$$\text{TTFT}_{\text{cache}} = T_{\text{queue}} + \frac{2 (L_{\text{prompt}} - L_{\text{hit}}) n_l d^2}{\text{FLOPS} \cdot \text{util}}$$

节省：

$$\Delta\text{TTFT} = \frac{2 L_{\text{hit}} n_l d^2}{\text{FLOPS} \cdot \text{util}} \propto L_{\text{hit}}$$

命中越多（公共 system prompt / few-shot / 多轮历史），TTFT 降越多。这是 [[prefix caching]] 降 TTFT 的数学依据。

### 4.3 TPOT 公式

decode 每 iteration 给 batch 内 $B$ 个请求各产 1 token，TPOT = 一个 iteration 耗时：

$$\text{TPOT} = T_{\text{decode\_iter}} = \max\left(T_{\text{compute}}(B),\ T_{\text{memory}}(B, L_{\text{kv}})\right)$$

算力时间：

$$T_{\text{compute}}(B) = \frac{2 B \cdot n_l \cdot d^2}{\text{FLOPS}_{\text{GPU}} \cdot \text{util}}$$

带宽时间（读权重 + KV cache）：

$$T_{\text{memory}}(B, L_{\text{kv}}) = \frac{\text{weight\_bytes} + B \cdot 2 n_l L_{\text{kv}} d \cdot \text{kv\_bytes}}{\text{HBM\_BW} \cdot \text{util}}$$

- $\text{weight\_bytes} = n_l \cdot d^2 \cdot \text{param\_bytes}$：模型权重字节数（固定，一次读）。
- $B \cdot 2 n_l L_{\text{kv}} d \cdot \text{kv\_bytes}$：$B$ 个请求，每请求 $L_{\text{kv}}$ 历史 KV，K+V 各一份，每层 $d$ 维。
- decode 通常 memory-bound：$T_{\text{memory}} > T_{\text{compute}}$ → $\text{TPOT} \approx T_{\text{memory}}$。

### 4.4 batching tradeoff 的数学

TPOT 随 batch $B$ 的变化（memory-bound 时）：

$$\text{TPOT}(B) \approx \frac{\text{weight\_bytes} + B \cdot \text{kv\_read\_per\_req}}{\text{HBM\_BW}}$$

- $B$ 小：权重读 $\text{weight\_bytes}$ 主导，摊不到 $B$ 上，单 token 成本高（带宽浪费在权重上）。
- $B$ 大：KV 读 $B \cdot \text{kv\_read\_per\_req}$ 主导，TPOT 随 $B$ 涨。

吞吐 throughput = $B / \text{TPOT}(B)$（每秒 token 数）：

- $B$ 小：throughput 低（单 token 慢，batch 又小）。
- $B$ 大：throughput 涨（摊销），但 TPOT 也涨→延迟 SLO 可能违反。

[[goodput]] = throughput × (1 - violation_rate) 在这个 tradeoff 找使 goodput 最大的 $B^*$。是 [[batching tradeoff]] 的核心数学。

### 4.5 PD 分离后的独立优化

分离前（同 GPU 混）：

$$\text{TTFT}_{\text{mixed}}, \text{TPOT}_{\text{mixed}} \text{ 受同一 GPU 算力+带宽争用}$$

分离后（prefill 池 P + decode 池 D）：

$$\text{TTFT}_{\text{P-pool}} = T_{\text{queue,P}} + \frac{2 L_{\text{eff}} n_l d^2}{\text{FLOPS}_\text{P} \cdot \text{util}} \quad (\text{prefill 池堆算力, 降 TTFT})$$

$$\text{TPOT}_{\text{D-pool}} \approx \frac{\text{weight\_bytes}_\text{D} + B \cdot \text{kv\_read}}{\text{HBM\_BW}_\text{D}} \quad (\text{decode 池堆带宽, 降 TPOT})$$

prefill 池用算力型卡（高 FLOPS）降 TTFT，decode 池用带宽型卡（高 HBM_BW）降 TPOT，两指标独立优化、互不争用。是 [[PD分离]] 的收益量化。

### 4.6 与 ITL、P99 的关系

- TPOT 是 decode **平均** token 间隔：$\text{TPOT} = \frac{1}{n-1}\sum_{i=1}^{n-1}(T_{i+1} - T_i)$。
- [[ITL]] 是**逐个**间隔 $T_{i+1} - T_i$，看抖动/长尾。
- [[P99延迟]] 是 TTFT 或 ITL 的 99 分位数（尾延迟）。
- TPOT 平均掉了抖动，ITL/P99 才看长尾。三者互补。


## 5. 代码示例（可选）

### 5.1 用流式接口测 TTFT 与 TPOT

```python
import time
import requests

def measure_ttft_tpot(url, prompt, max_tokens=64):
    """流式接口测 TTFT (首 token) 与 TPOT (平均 token 间隔)"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,          # 流式, 逐 token 返回
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()           # 请求发出
    ttft = None
    token_times = []                   # 各 token 到达时间

    resp = requests.post(url, json=payload, headers=headers, stream=True)
    for line in resp.iter_lines():
        if line and b"data:" in line:
            # SSE: data: {...}
            t = time.perf_counter()
            if ttft is None:
                ttft = t - t0          # 首 token -> TTFT
            token_times.append(t)

    # TPOT = decode token 间隔的平均 (去掉首 token)
    intervals = [token_times[i+1] - token_times[i] for i in range(len(token_times)-1)]
    tpot = sum(intervals) / len(intervals) if intervals else 0.0

    return {
        "ttft_s": ttft,                # 首 token 延迟 (prefill + queue)
        "tpot_s": tpot,                # 每 token 平均时间 (decode)
        "n_tokens": len(token_times),
        "intervals": intervals,        # 各间隔, 给 ITL/P99 用
    }

# result = measure_ttft_tpot("http://localhost:8000/v1/chat/completions", "讲讲 Transformer", 64)
# print(result)
```

### 5.2 模拟 prompt 长度对 TTFT 的影响

```python
import time

def fake_prefill_time(L_prompt, d=4096, n_l=32, flops=312e12, util=0.5, L_hit=0):
    """模拟 prefill 耗时 (compute-bound): 2*L_eff*n_l*d^2 / (FLOPS*util)"""
    L_eff = L_prompt - L_hit
    flops_needed = 2 * L_eff * n_l * d * d
    return flops_needed / (flops * util)

for L in [1024, 4096, 8192, 32768, 131072]:
    t = fake_prefill_time(L)
    t_cache = fake_prefill_time(L, L_hit=2048)   # 命中 2k 前缀
    print(f"prompt={L:>6}  TTFT_prefill={t*1e3:8.1f}ms  命中2k后={t_cache*1e3:8.1f}ms")
# 趋势: TTFT 随 prompt 近线性涨; prefix cache 命中降固定量
```

### 5.3 模拟 batch 对 TPOT 的影响（batching tradeoff）

```python
def fake_tpot(B, L_kv=2048, d=4096, n_l=32,
              param_bytes=14e9 * 2,   # 14B 参数 FP16
              kv_bytes=2,             # KV FP16
              hbm_bw=1.5e12, util=0.7):
    """模拟 decode TPOT (memory-bound): (权重 + B*KV) / HBM_BW"""
    weight = param_bytes
    kv = B * 2 * n_l * L_kv * d * kv_bytes
    return (weight + kv) / (hbm_bw * util)

for B in [1, 4, 16, 64, 128, 256]:
    tpot = fake_tpot(B)
    thr = B / tpot                    # throughput (token/s)
    print(f"B={B:>3}  TPOT={tpot*1e3:6.1f}ms  throughput={thr:8.1f} tok/s")
# 趋势: TPOT 先降(权重摊销)后升(KV 涨), throughput 涨但 TPOT 涨 -> tradeoff
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[autoregressive decoding]]（prefill/decode 两阶段是 TTFT/TPOT 的阶段基础）、[[vLLM scheduler]]（T_queue 由调度决定）、[[KV cache]]（decode 读 KV 决定 TPOT）、[[prefix caching]]（命中降 TTFT）、[[batching tradeoff]]（batch 影响 TPOT）。
- **下游（应用）**: [[PD分离]]（两指标拆分是 PD 分离的度量与动机）、[[goodput]]（SLO 用 TPOT/TTFT 定 violation）、[[ITL]]（TPOT 的逐 token 细化看抖动）、[[P99延迟]]（TTFT/ITL 的尾延迟）。
- **对比 / 易混**:
  - **TTFT vs TPOT**：见 3.7 对比表。前者 prefill compute-bound，后者 decode memory-bound。
  - **TPOT vs [[ITL]]**：TPOT 是 decode token 间隔**平均**，ITL 是**逐个**间隔（看抖动/长尾）。TPOT 平均掉尖峰，ITL 暴露尖峰。
  - **TPOT vs [[P99延迟]]**：TPOT 是均值，P99 是尾延迟分位数。P99 TPOT > 平均 TPOT，SLO 常用 P99。
  - **TTFT/TPOT vs [[goodput]]**：前者是延迟（单请求），后者是有效吞吐（系统级，含 SLO 违反率）。goodput 用 TTFT/TPOT 定 violation。


## 7. 常见误区与易错点

> [!warning] 误区 1：TTFT 就是 prefill 时间
> 不完全。TTFT = T_queue + T_prefill。高并发时 T_queue（排队等 admit）可占 TTFT 大头。调度拥塞也会涨 TTFT，不只 prefill 算力。

> [!warning] 误区 2：TPOT 越小越好，拼命降
> TPOT 降常靠加大 batch 摊销权重，但 batch 大→TPOT 自己也涨（KV 读涨）且 ITL 抖动/P99 涨。单看 TPOT 均值会漏长尾。要结合 [[ITL]]/[[P99延迟]]/[[goodput]]。

> [!warning] 误区 3：TTFT 和 TPOT 瓶颈一样
> 不同。TTFT 是 compute-bound（算力/SM），TPOT 是 memory-bound（HBM 带宽）。优化手段几乎不重合（见 2.3）。混为一谈会优化错方向。

> [!warning] 误区 4：prefill 短就 TTFT 低
> 还要看 prefix cache 命中与 queue。prompt 短但无 cache 命中 + 高并发排队，TTFT 仍高。TTFT 是三因素叠加。

> [!tip] 实践：定 SLO 分开定
> SLO 应分开定 TTFT 与 TPOT（如 P99 TTFT < 2s，TPOT < 100ms），而非一个总延迟。因优化手段不同。PD 分离后各池各定各 SLO。

> [!note] 补充：流式 vs 非流式
> TPOT 只在**流式**接口可测（逐 token 到达时间）。非流式（一次性返回全 token）只能测总延迟，拆不出 TTFT/TPOT/ITL。推理服务 benchmark 必用流式。


## 8. 延伸细节

### 8.1 per-request TTFT 与系统级

TTFT/TPOT 常按 per-request 测，但系统级看分布（P50/P99）。高并发下 P99 TTFT 远高于均值（排队长尾）。SLO 用 P99（见 [[P99延迟]]）。

### 8.2 与 continuous batching 的交互

[[continuous batching的调度实现]] 动态进出请求→batch size 实时变→TPOT 实时变（batch 大 TPOT 涨）。chunked prefill 混入 prefill chunk 也会让某次 decode iteration 变慢→ITL 尖峰（见 [[ITL]]）。

### 8.3 量化对 TPOT 的降

[[量化]]（FP16→FP8/INT4）降权重字节 + KV 字节→T_memory 降→TPOT 降。是 decode 优化核心。但量化有精度损失，需 tradeoff。

### 8.4 数字待核实

具体 TTFT/TPOT 数值随模型（7B/70B）、卡（A100/H100）、精度（FP16/FP8）、batch、prompt 长度大幅变化。文中量级仅作趋势参考，精确值需实测。A100 FP16 ~312 TFLOPS、HBM ~1.5TB/s；H100 ~990 TFLOPS(FP16稀疏待核实)、HBM ~3.35TB/s，均为公开规格待核实。

### 8.5 内容来源

整理自 vLLM/SGLang serving benchmark 文档、NVIDIA Triton inference metrics 定义、MLPerf Inference 规范、LLM serving 论文（vLLM/SARATHI-serve/DeepSpeed-FastGen）。两阶段算力特征见 [[autoregressive decoding]]。PD 分离见 [[PD分离]]。batching tradeoff 见 [[batching tradeoff]]。

---
相关: [[推理性能指标]] | [[autoregressive decoding]] | [[ITL]] | [[P99延迟]] | [[goodput]] | [[batching tradeoff]] | [[continuous batching的调度实现]] | [[PD分离]] | [[KV cache]] | [[prefix caching]] | [[vLLM scheduler]] | [[量化]] | [[GPU utilization]] | [[SM utilization]] | [[memory bandwidth]]
