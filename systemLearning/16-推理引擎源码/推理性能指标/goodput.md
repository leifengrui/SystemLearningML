# goodput

> **所属章节**: [[推理性能指标]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: goodput / 有效吞吐 / SLO 约束吞吐 / goodput throughput / 有效 QPS
> **难度**: 中（需懂 [[TTFT与TPOT]]、[[ITL]]、[[P99延迟]]、[[batching tradeoff]]、[[continuous batching的调度实现]]）


## 1. 一句话定义

**goodput（有效吞吐）** = 在 SLO 约束下**真正算数**的吞吐量，定义为 $\text{goodput} = \text{throughput} \times (1 - \text{violation\_rate})$——即原始吞吐扣掉那些违反延迟 SLO（如 [[P99延迟]] TTFT > 2s 或 [[ITL]] P99 > 阈值）的"无效"请求后剩余的有效部分。它之所以必要，是因为单看 throughput（原始 token/s 或 req/s）会**误把高延迟换来的吞吐当收益**：增大 batch 能拉高 throughput，但 batch 大→TPOT 涨、[[P99延迟]] 涨→SLO 违反率涨→用户体验崩，这部分违反 SLO 的吞吐**不算 goodput**。goodput 正是在这个 [[batching tradeoff]]（batch 大吞吐高但延迟也高）里找**使有效吞吐最大**的平衡点——既要吞吐又要满足 SLO。它是 LLM serving 的**终极优化目标**（不是裸 throughput），直接驱动 batch size 选择、KV 预留、GPU 数、[[PD分离]] 策略。与 Little's Law（$L = \lambda W$）配合可估算给定 SLO 下的最大并发与所需容量。

> [!note] 三句话定位
> - **是什么**：满足 SLO 的有效吞吐 = throughput × (1 - violation_rate)。扣掉违反 SLO 的"无效"吞吐。
> - **为什么**：单 throughput 会把高延迟换来的吞吐当收益（大 batch 吞吐高但 SLO 违反率高）。goodput 在 [[batching tradeoff]] 找吞吐与延迟的平衡点。
> - **与 [[batching tradeoff]] 关系**：batch 大→throughput↑ 但 violation↑，goodput 先升后降有最优 $B^*$。goodput 是 batching tradeoff 的优化目标。


## 2. 为什么需要它（动机与背景）

### 2.1 单 throughput 会骗人

只看 throughput（token/s 或 req/s）会误判：增大 batch→throughput 涨→看着"性能更好"。但 batch 大→TPOT 涨（见 [[TTFT与TPOT]]）、P99 涨（见 [[P99延迟]]）→SLO 违反率涨→那些超 SLO 的请求**用户体验崩**（卡顿/超时）。这部分"吐出来但违反 SLO"的吞吐是**无效**的——用户不满意、不算达标。单 throughput 把它们也算进去→虚高。

```
batch=16:  throughput=1000 tok/s, violation=1%   -> goodput=990
batch=64:  throughput=3000 tok/s, violation=30%  -> goodput=2100  (吞吐涨3倍但 goodput 只2倍)
batch=256: throughput=4000 tok/s, violation=60% -> goodput=1600  (吞吐涨但 goodput 反降!)
  -> 单 throughput 看不出 batch=256 已崩, goodput 暴露
```

### 2.2 batching tradeoff 的核心矛盾

[[batching tradeoff]]：batch 大→权重读取摊销→单 token 成本降→throughput↑；但 batch 大→KV 读量大→TPOT↑→P99↑→violation↑。throughput 与 violation 同向涨，goodput = throughput × (1 - violation) 在中间有**最大值**。goodput 是这个 tradeoff 的**优化目标**——不是 throughput 也不是延迟，而是"满足 SLO 的吞吐"。找使 goodput 最大的 batch $B^*$ 是推理引擎调度核心。

### 2.3 SLO 是硬约束，违反不算数

生产 LLM 服务有 SLO（如 P99 TTFT < 2s、P99 ITL < 200ms）。违反 SLO 的请求=体验不达标=不算有效产出（可能被客户端重试/超时/差评）。goodput 把这部分扣掉，反映"真正满足用户的有效吞吐"。是服务级 KPI，不是硬件级吞吐。

### 2.4 容量规划的依据

集群扩容决策不能按 throughput（会过估，含违反 SLO 的），要按 goodput（实际有效产出）。给定目标 goodput + SLO→反推所需 GPU 数、batch 上限、KV 预留。是 [[PD分离]] 池配比的依据（prefill 池保 TTFT SLO、decode 池保 TPOT/ITL SLO，合起来最大化 goodput）。


## 3. 核心概念详解

### 3.1 goodput 公式

$$\text{goodput} = \text{throughput} \times (1 - \text{violation\_rate})$$

- **throughput**：原始吞吐（req/s 或 token/s），不问 SLO。
- **violation_rate**：违反 SLO 的请求比例（如 TTFT > 2s 或 P99 ITL > 阈值的请求占比）。
- **(1 - violation_rate)**：满足 SLO 的比例（有效比例）。
- **goodput**：满足 SLO 的有效吞吐。

### 3.2 violation 的定义

violation 由 SLO 阈值定义，常见形式：

```
SLO (示例):
  TTFT:   P99 < 2s        (见 [[TTFT与TPOT]])
  ITL:    P99 < 200ms     (见 [[ITL]])
  TPOT:   < 100ms

violation 判定 (per-request):
  请求 r 违反 SLO  ⟺  r.TTFT > 2s  OR  max(r.ITLs) > 200ms  OR  r.TPOT > 100ms
violation_rate = (违反请求数) / (总请求数)
```

> [!note] P99 vs per-request violation
> 严格说 P99 是分布统计量（"99% 请求低于此"），不是 per-request 判定。但工程上常简化为"单请求 TTFT/ITL 超阈值即 violation"，violation_rate 即超阈值比例。若设阈值 = P99 SLO 值，则 violation_rate ≈ 1%。详见 [[P99延迟]]。

### 3.3 batching tradeoff 下的 goodput 曲线

```
goodput
  ^
  |           goodput = throughput × (1 - violation)
  |          /\
  |        /    \          <- goodput 先升 (throughput 主导) 后降 (violation 主导)
  |      /        \           顶点 = 最优 batch B*
  |    /            \
  |  /                \
  |/____________________\_______> batch size B
  0                    B*
     throughput (↗)  ^  violation (↗)
     (throughput 涨, violation 也涨, goodput 取平衡)
```

- $B$ 小：throughput 低（摊销差），violation 低（延迟低）→goodput 受 throughput 限。
- $B$ 大：throughput 高，violation 高（延迟高）→goodput 受 violation 限。
- $B^*$：goodput 最大点，是 [[batching tradeoff]] 的最优解。

### 3.4 throughput vs goodput 对比表

| 维度 | throughput | goodput |
|---|---|---|
| 定义 | 原始吞吐（req/s 或 token/s） | 满足 SLO 的有效吞吐 |
| 公式 | $B / \text{TPOT}$（token/s） | $\text{throughput} \times (1 - \text{violation})$ |
| 看 SLO | 不看 | 扣违反 SLO 的部分 |
| 受 batch 影响 | 随 $B$ 增单调升（摊销） | 先升后降（violation 后主导） |
| 优化目标 | 否（会过估） | **是**（终极目标） |
| 体感 | 不反映用户满意 | 反映满足 SLO 的有效产出 |
| 用途 | 硬件峰值 benchmark | 生产服务 KPI、容量规划 |

### 3.5 Little's Law 与 goodput

Little's Law：$L = \lambda W$（系统内平均并发 $L$ = 到达率 $\lambda$ × 平均驻留 $W$）。在 LLM serving：

- $L$：在飞请求数（并发）。
- $\lambda$：到达率（req/s）。
- $W$：平均端到端延迟（含 TTFT + decode 全程）。

给定 SLO（$W \le W_{\text{SLO}}$）与并发上限（$L \le L_{\max}$，受 KV cache 容量限）：

$$\lambda_{\max} = \frac{L_{\max}}{W_{\text{SLO}}}$$

即 SLO 约束下的最大到达率。若实际 $\lambda > \lambda_{\max}$→排队→$W$ 涨→SLO 违反→goodput 降。Little's Law 给 goodput 的容量上界。

### 3.6 violation 与 P99/ITL/TTFT 的关系

violation_rate 由 [[P99延迟]] SLO 阈值定义，作用于 [[TTFT与TPOT]]（TTFT）与 [[ITL]]（ITL）：

```
violation_rate = P(TTFT > SLO_TTFT) + P(ITL P99 > SLO_ITL) - 两者交集
  (粗略: 任一超即违反)
  SLO 越严 -> violation_rate 越高 -> goodput 越低
  preempt/swap/调度抖动 -> P99 涨 -> violation 涨 -> goodput 降
```


## 4. 数学原理 / 公式

### 4.1 goodput 定义

$$\text{goodput} = \text{throughput} \times (1 - \text{violation\_rate})$$

展开 throughput（decode，token/s）：

$$\text{throughput} = \frac{B}{\text{TPOT}(B)}$$

故：

$$\text{goodput}(B) = \frac{B}{\text{TPOT}(B)} \times (1 - \text{violation}(B))$$

### 4.2 batching tradeoff 的 goodput 极值

由 [[TTFT与TPOT]] 4.4，memory-bound 时：

$$\text{TPOT}(B) \approx \frac{\text{weight\_bytes} + B \cdot \text{kv\_read}}{\text{HBM\_BW}}$$

throughput：

$$\text{throughput}(B) = \frac{B}{\text{TPOT}(B)} = \frac{B \cdot \text{HBM\_BW}}{\text{weight\_bytes} + B \cdot \text{kv\_read}}$$

- $B$ 小：分母 weight 主导，throughput $\approx B \cdot \text{HBM\_BW}/\text{weight}$（线性涨）。
- $B$ 大：分母 KV 主导，throughput $\approx \text{HBM\_BW}/\text{kv\_read}$（饱和）。

violation 随 $B$ 升（TPOT 涨→P99 涨→violation 涨），设：

$$\text{violation}(B) = f(\text{TPOT}(B), \text{ITL\_P99}(B), \dots)$$

简化为 violation 随 TPOT 单调升。goodput = throughput × (1 - violation)：

- $B$ 小：throughput 涨快、violation 涨慢→goodput 升。
- $B$ 大：throughput 饱和、violation 涨快→goodput 降。
- 极值 $B^*$：$\frac{d}{dB}\text{goodput}(B) = 0$。

> [!note] 推导直觉
> goodput 是 throughput（随 $B$ 升）与 (1-violation)（随 $B$ 降）的乘积，一升一降→中间有峰。$B^*$ 是乘积最大点。这是 [[batching tradeoff]] 的数学本质：吞吐与延迟（SLO 违反）的平衡。

### 4.3 Little's Law 推导

稳态系统（到达率 = 离开率 $\lambda$），平均驻留 $W$，系统内并发 $L$。Little's Law：

$$L = \lambda W$$

> [!note] 推导直觉
> 想象系统是管道：每秒进 $\lambda$ 个请求，每个驻留 $W$ 秒，则管道内同时有 $\lambda W$ 个。是守恒律，对任何分布成立。

LLM serving 中 $L$ 受 KV cache 容量限（见 [[KV block manager]]）：

$$L_{\max} = \frac{\text{KV\_cache\_total\_blocks}}{\text{avg\_blocks\_per\_req}}$$

给定 SLO $W \le W_{\text{SLO}}$：

$$\lambda_{\max} = \frac{L_{\max}}{W_{\text{SLO}}}$$

若 $\lambda > \lambda_{\max}$→$L$ 超 $L_{\max}$（KV 不够）→preempt/排队→$W$ 涨→violation 涨→goodput 降。Little's Law 给 goodput 的容量上界。

### 4.4 实例计算

设 7B 模型，A100，decode 阶段：

- weight_bytes = 14GB（FP16）
- kv_read_per_req = $2 \times 32 \times 2048 \times 4096 \times 2$ bytes ≈ 2GB/req（FP16，L_kv=2048）
- HBM_BW = 1.5TB/s
- SLO: TPOT < 100ms，violation 当 TPOT > 100ms

```
B=1:   TPOT = (14+2)/1500 = 0.0107s = 10.7ms  (<100ms, violation=0)
       throughput = 1/0.0107 = 93 tok/s
       goodput = 93 × 1.0 = 93
B=16:  TPOT = (14+32)/1500 = 0.0307s = 30.7ms  (violation=0)
       throughput = 16/0.0307 = 521 tok/s
       goodput = 521 × 1.0 = 521
B=64:  TPOT = (14+128)/1500 = 0.0947s = 94.7ms (violation=0, 临界)
       throughput = 64/0.0947 = 676 tok/s
       goodput = 676 × 1.0 = 676
B=128: TPOT = (14+256)/1500 = 0.180s = 180ms (>100ms, violation=1.0 全违反)
       throughput = 128/0.180 = 711 tok/s
       goodput = 711 × 0.0 = 0   (SLO 全崩)
       (实际 violation 是渐变的, 此处简化为阶跃)
```

$B^*$ 约在 64 附近（goodput 峰）。超 $B^*$ throughput 仍涨但 violation 涨更快→goodput 崩。是 [[batching tradeoff]] 的实例。

（数字为简化模型量级，待核实。趋势：goodput 先升后降有峰。）


## 5. 代码示例（可选）

### 5.1 算 goodput（throughput + violation）

```python
def goodput(throughput, violation_rate):
    """goodput = throughput × (1 - violation_rate)"""
    return throughput * (1 - violation_rate)

# 模拟不同 batch 下的 throughput 与 violation
configs = [
    # (B, throughput_tok/s, violation_rate)
    (1,   93,  0.00),
    (16,  521, 0.00),
    (64,  676, 0.05),
    (128, 711, 0.30),
    (256, 740, 0.60),
]
print(f"{'B':>4} {'throughput':>12} {'violation':>10} {'goodput':>10}")
for B, thr, vio in configs:
    g = goodput(thr, vio)
    print(f"{B:>4} {thr:>12} {vio:>10.2f} {g:>10.1f}")
# goodput 先升后降, 找最大 = B*
```

### 5.2 批量测 violation（基于 ITL/TTFT）

```python
import numpy as np

def compute_goodput_from_latencies(ttft_list, itl_lists,
                                   slo_ttft=2.0, slo_itl_p99=0.200):
    """从测得的 TTFT 与各请求 ITL 序列算 goodput
    ttft_list: 每请求 TTFT (s)
    itl_lists: 每请求的 ITL 序列 (list of list, s)
    """
    n = len(ttft_list)
    violations = 0
    for ttft, itls in zip(ttft_list, itl_lists):
        p99_itl = np.percentile(itls, 99) if len(itls) > 0 else 0
        if ttft > slo_ttft or p99_itl > slo_itl_p99:
            violations += 1
    violation_rate = violations / n
    # throughput = 完成请求数 / 总时间 (这里用 n / max_end_time 近似)
    return {
        "violation_rate": violation_rate,
        "n": n,
        "n_violated": violations,
    }

# 模拟 100 请求
rng = np.random.default_rng(0)
ttfts = rng.normal(0.3, 0.1, 100)
itls = [rng.normal(0.030, 0.005, 64) for _ in range(100)]
# 注入 5 个违规 (TTFT 超大)
ttfts[:5] += 3.0
r = compute_goodput_from_latencies(ttfts, itls)
print(f"violation_rate = {r['violation_rate']:.2%}  ({r['n_violated']}/{r['n']})")
# throughput 需另测 (n / total_time), goodput = throughput × (1 - violation_rate)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[TTFT与TPOT]]（TTFT/TPOT 定 SLO 阈值）、[[ITL]]（P99 ITL 定 violation）、[[P99延迟]]（P99 SLO 是 violation 判据）、[[batching tradeoff]]（batch 影响 throughput 与 violation）、[[continuous batching的调度实现]]（batch 动态进出影响 goodput）。
- **下游（应用）**: 推理服务 KPI、容量规划（按 goodput 反推 GPU 数）、[[PD分离]] 池配比（prefill/decode 池合最大化 goodput）、调度优化（找 $B^*$）。
- **对比 / 易混**:
  - **goodput vs throughput**：见 3.4 对比表。前者扣 SLO 违反，后者不扣。goodput 是终极目标。
  - **goodput vs [[P99延迟]]**：P99 是延迟分位（定 violation 阈值），goodput 是有效吞吐（受 P99 约束）。前者是约束，后者是受约束的目标。
  - **goodput vs [[batching tradeoff]]**：batching tradeoff 是 batch 大小吞吐与延迟的矛盾，goodput 是该矛盾的优化目标（找 $B^*$）。
  - **goodput vs Little's Law**：Little's Law 给容量上界（$\lambda_{\max} = L_{\max}/W_{\text{SLO}}$），goodput 是实际有效吞吐（受 violation 扣）。前者是理论上限，后者是实际产出。


## 7. 常见误区与易错点

> [!warning] 误区 1：goodput = throughput
> 不等。goodput 扣违反 SLO 的部分。throughput 高但 violation 高→goodput 低。单 throughput 过估。

> [!warning] 误区 2：batch 越大 goodput 越高
> 不。batch 大→throughput 升但 violation 升更快→goodput 先升后降。有最优 $B^*$。超 $B^*$ goodput 崩。见 4.4 实例。

> [!warning] 误区 3：violation 只看 TTFT
> 不。violation 可由 TTFT、ITL P99、TPOT 任一超阈值定。流式场景 ITL P99 是关键（见 [[ITL]]）。

> [!warning] 误区 4：goodput 高就万事大吉
> goodput 是综合指标，但仍要看单项（TTFT/ITL/P99）分布。goodput 高但 P999 极差→极端用户体验差。需综合看。

> [!tip] 实践：调度器优化 goodput
> [[vLLM scheduler]] 调 batch 上限 / KV 预留的目标是最大化 goodput，不是 throughput。benchmark 时报 goodput（带 SLO），不只报 throughput。

> [!note] 补充：goodput 与 SLA
> SLA（服务级协议）常用 goodput 做承诺（如"goodput ≥ 1000 tok/s 且 P99 < 2s"）。违反 SLA 有赔付。goodput 是商业指标。


## 8. 延伸细节

### 8.1 goodput 在 PD 分离中的作用

[[PD分离]] 后 prefill 池与 decode 池各保各 SLO（TTFT / TPOT-ITL），合起来最大化总 goodput。池配比（prefill:decode GPU 数）按 goodput 反推。是 PD 分离的优化目标。

### 8.2 goodput 与抢占式调度

[[KV block manager]] 的 preempt 会拉爆 P99→violation↑→goodput↓。降 preempt（加 KV 预留、chunked prefill 增量分配）直接提 goodput。是 goodput 优化的关键杠杆。

### 8.3 数字待核实

文中 throughput/violation/goodput 数值为简化模型量级，待核实。实际值随模型/卡/SLO 场景变。SLO 阈值（P99 TTFT<2s、ITL<200ms）为常见经验值待核实。

### 8.4 内容来源

整理自 LLM serving 系统论文（vLLM、SARATHI-serve、DistServe、DeepSpeed-FastGen 关注 goodput/SLO-aware serving）、Google SRE 书（goodput 概念）、NVIDIA Triton inference metrics。batching tradeoff 见 [[batching tradeoff]]。P99 见 [[P99延迟]]。Little's Law 见排队论基础。

---
相关: [[推理性能指标]] | [[TTFT与TPOT]] | [[ITL]] | [[P99延迟]] | [[batching tradeoff]] | [[continuous batching的调度实现]] | [[vLLM scheduler]] | [[PD分离]] | [[KV block manager]] | [[chunked prefill]] | [[GPU utilization]] | [[memory bandwidth]]
