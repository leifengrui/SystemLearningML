# batching tradeoff

> **所属章节**: [[延迟与吞吐优化]]
> **所属模块**: [[08-推理系统]]
> **别名**: batching tradeoff / 批大小权衡 / throughput-latency tradeoff
> **难度**: 中（需懂 [[continuous batching]] + 排队论基础）


## 1. 一句话定义

**batching tradeoff（批大小权衡）** 是 LLM 推理服务中 **batch size 对吞吐（throughput）与延迟（latency）反向影响**的核心权衡：增大 batch 提升 GPU 利用率与**吞吐**（单位时间处理 token/请求多），但**凑批等待时间**与**每步 decode 的 KV cache 访存量**上升导致**延迟**增加，且 batch 受 [[KV cache management]] 显存预算硬约束。推理服务的本质就是在给定延迟 SLO（服务等级目标）下最大化吞吐，或反之。

> [!note] 三句话定位
> - **是什么**：batch 大了吞吐高但延迟高、显存大；小了延迟低但吞吐低。
> - **为什么**：GPU 是并行处理器，batch 小则算力闲置；batch 大则凑批等、KV 读量大。
> - **怎么做**：在 SLO（如 P99 延迟 < X）约束下找最大吞吐的 batch；用 [[continuous batching]] + chunked prefill 动态调节。


## 2. 为什么需要它（动机与背景）

### 2.1 推理服务的两个目标

推理服务被两端拉扯：

- **吞吐（throughput）**：QPS（请求/秒）或 tokens/s。运营成本视角——同样硬件吞吐越高，单请求成本越低。
- **延迟（latency）**：TTFT（首 token 延迟，影响首响体验）+ TPOT（每 token 延迟，影响流式速度）+ E2E（端到端）。用户体验视角——延迟超 SLO 用户感知卡顿。

batch size 同时影响两者，且**方向相反**：

| batch ↑ | 吞吐 | 延迟 |
|---|---|---|
| GPU 利用率 | ↑（算力被填满） | — |
| 凑批等待 | — | ↑（等够 batch） |
| 每步 KV 读量 | — | ↑（decode 读 batch×cache） |
| KV cache 显存 | — | ↑（可能 OOM） |

### 2.2 三个约束的拉扯

batch 选型被三力拉扯：

1. **算力利用率**：batch 小 → GPU 闲置 → 推增大 batch。
2. **延迟 SLO**：batch 大 → 凑批等 + KV 读多 → 推减小 batch。
3. **显存预算**：batch × 单请求 KV cache ≤ KV 显存预算 → 硬上限。

> [!tip] 核心矛盾
> 算力想大 batch，延迟与显存想小 batch。batching tradeoff 就是找这三者的平衡点，且这个点随负载、序列长度、SLO 动态变化。


## 3. 核心概念详解

### 3.1 prefill 与 decode 的不同 batch 特性

| 阶段 | 瓶颈 | batch 大小影响 |
|---|---|---|
| **prefill** | compute-bound（算 prompt 的 $K/V$） | 大 batch 充分用算力，吞吐↑明显，延迟主要为排队+计算 |
| **decode** | memory-bound（每步读 KV cache） | 大 batch 摊薄 cache 读的固定开销，吞吐↑；但单步耗时随 batch 缓增，TPOT↑ |

> [!note] 为什么 decode 也想大 batch
> decode 每步读整个 KV cache 进 SRAM，这个"读"是固定成本（不论 batch 1 还是 64 都要读 cache）。batch 1 时这次读只为 1 个 token 服务，算力利用率极低；batch 64 时同次读为 64 个 token 服务，利用率↑。所以 decode 也是大 batch 提吞吐——但单步耗时随 batch 缓增（64 个请求的 cache 一起读，访存量×64），TPOT 上升。

### 3.2 显存约束（KV cache 预算）

推理显存 = 权重 + 所有并发请求 KV cache 之和。权重固定，KV cache 随 batch×seq 线性涨：

$$
M_{\text{KV}}(B, S) = 2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot S \cdot B \cdot p \le M_{\text{KV budget}}
$$

（公式见 [[KV cache management]] §4.1）。给定显存预算 $M_{\text{budget}}$，最大并发 batch：

$$
B_{\max} = \frac{M_{\text{budget}}}{2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot S \cdot p}
$$

长序列（$S$ 大）直接压低 $B_{\max}$——这就是长 context 服务并发难的根因。

### 3.3 SLO（Service Level Objective）

典型 SLO：

- TTFT P99 < 500ms（首响要快）。
- TPOT P99 < 50ms（流式跟手）。
- E2E < 某值（端到端）。

batching tradeoff 的目标：**在满足 SLO 的前提下最大化吞吐**。即 throughput-latency 曲线的 Pareto 前沿上，选 SLO 约束允许的最大吞吐点。

### 3.4 Goodput（有效吞吐）

**Goodput** = 满足 SLO 的请求速率（不是裸吞吐）。裸 throughput 高但超 SLO 的请求不算 goodput。这是 SLO-aware 服务的核心指标——优化 goodput 而非裸 throughput。

### 3.5 动态调节

最优 batch 随负载变：低负载时小 batch 保延迟，高负载时大 batch 保吞吐。[[continuous batching]] 的 slot 动态进出 + chunked prefill 就是动态调节的实现。


## 4. 数学原理 / 公式

### 4.1 Little's Law

稳态排队系统 $L = \lambda W$：

- $L$ = 系统内平均请求数（并发）
- $\lambda$ = 到达率（req/s）
- $W$ = 平均逗留时间（延迟）

给定 SLO $W \le W_{\max}$，最大支持到达率 $\lambda \le L / W_{\max}$。$L$ 受显存约束 $\le B_{\max}$（§3.2）。所以：

$$
\lambda_{\max} = \frac{B_{\max}}{W_{\max}}
$$

> [!note] 推导
> 这是 batching tradeoff 的"上限公式"：吞吐上限 = 并发上限 / 延迟上限。增大并发（大 batch、PagedAttention 省 KV）或降低延迟（speculative、prefix caching）都能提吞吐上限。两者常冲突——大 batch 提并发但增延迟。

### 4.2 throughput-latency 曲线

设单请求 decode $n$ token，batch $B$ 时单步耗时 $T_{\text{step}}(B)$（随 $B$ 缓增，因 memory-bound）。每步产出 $B$ 个 token（batch 内每请求 1 个）。token 吞吐：

$$
\text{TPS}_{\text{tok}}(B) = \frac{B}{T_{\text{step}}(B)}
$$

单请求延迟：

$$
W(B) = T_{\text{prefill}} + n \cdot T_{\text{step}}(B) + T_{\text{queue}}(B)
$$

- $T_{\text{step}}(B)$ 随 $B$ 缓增 → 吞吐 $\frac{B}{T_{\text{step}}(B)}$ 随 $B$ 增（但增速递减，因 $T_{\text{step}}$ 也增）。
- $W(B)$ 随 $B$ 增（$T_{\text{step}}$ 增 + $T_{\text{queue}}$ 增）。

典型曲线：

```
吞吐 ↑
 |        ___________  (饱和)
 |      /
 |    /
 |  /
 |________________ batch →
            ↑ SLO 约束点

延迟 ↑                  /
 |                  ___ (超 SLO)
 |              /
 |          ___ (SLO 上限)
 |      /
 |__ __ __ __ __ __ batch →
```

最优 batch = SLO 延迟线与 throughput 饱和前的交点。

### 4.3 最优 batch 的解

在 $W(B) \le W_{\max}$ 约束下最大化 $\text{TPS}(B)$：

$$
B^* = \arg\max_B \text{TPS}(B) \quad \text{s.t.} \quad W(B) \le W_{\max}, \quad B \le B_{\max}
$$

通常 $B^*$ 取 $\min(B_{\text{SLO}}, B_{\max})$，其中 $B_{\text{SLO}}$ 是 $W(B) = W_{\max}$ 解出的 batch。


## 5. 代码示例（可选

### 5.1 throughput-latency 模拟

```python
import numpy as np

def step_time(B, base=0.01, growth=0.0003):
    """单步 decode 耗时: 随 batch 缓增 (memory-bound)."""
    return base + growth * B

def throughput(B):
    """token 吞吐 = B / step_time(B)."""
    return B / step_time(B)

def latency(B, n=200, prefill=0.1, queue_growth=0.002):
    """单请求延迟 = prefill + n*step + queue."""
    return prefill + n * step_time(B) + queue_growth * B * B

# 扫不同 batch, 找 SLO 约束下最优
slo = 3.0   # 延迟 SLO 3s
print(f"{'B':>4} {'throughput':>12} {'latency':>10} {'meets_SLO':>10}")
best, best_tp = None, 0
for B in [1,2,4,8,16,32,64,128]:
    tp, lat = throughput(B), latency(B)
    ok = lat <= slo
    print(f"{B:>4} {tp:>12.1f} {lat:>10.2f} {str(ok):>10}")
    if ok and tp > best_tp: best, best_tp = B, tp
print(f"\n最优 batch (SLO {slo}s): B={best}, throughput={best_tp:.1f} tok/s")
```

预期：throughput 随 B 增但饱和，latency 随 B 增；最优 B 是 SLO 约束下 throughput 最大的点（中间某值，非最大 B）。

### 5.2 KV cache 显存约束

```python
def max_batch(kv_budget_gb, n_layers, d_model, seq_len, bytes_per=2):
    """显存预算决定的最大并发 batch."""
    per_req = 2 * n_layers * d_model * seq_len * bytes_per / 1e9
    return int(kv_budget_gb / per_req)
# A100 40GB, 留 20GB 给 KV, Llama-7B(32层, d=4096), seq 2048
print('max batch:', max_batch(20, 32, 4096, 2048, 2))   # ~18
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[continuous batching]]（动态调节 batch 的实现）、[[dynamic batching]]（请求级凑批）、[[KV cache management]]（显存约束的来源）。
- **下游（应用）**: [[GPU utilization]]（batch 提利用率的目标）、[[sampling throughput]]（RLHF 采样受 batch 权衡影响）、[[policy deployment (PD)]]（PD 服务的 SLO 管理）。
- **对比 / 易混**:
  - **batching tradeoff vs [[continuous batching]]**：前者是"该用多大 batch"的权衡原则；后者是"如何动态调度 batch 成员"的实现机制。一个定策略，一个定机制。
  - **throughput vs goodput**：throughput 是裸速率，goodput 是满足 SLO 的有效速率。优化目标是 goodput。


## 7. 常见误区与易错点

> [!warning] 误区 1：batch 越大吞吐越高
> 增到一定程度后 throughput 饱和（GPU 算力/带宽到顶），再增大 batch 只增延迟不增吞吐，甚至因 preemption/显存压力下降。"越大越好"是错的。

> [!warning] 误区 2：忽略 SLO 只看裸吞吐
> 服务有 SLO，超 SLO 的请求算"失败"。裸 throughput 高但全超 SLO 等于没用。要优化 goodput（满足 SLO 的吞吐）。

> [!warning] 误区 3：混淆 prefill 与 decode 的 batch
> prefill 是 compute-bound，大 batch 充分用算力；decode 是 memory-bound，大 batch 摊薄 cache 读。两者 batch 的"甜点"不同，调度要分别对待（chunked prefill 与 decode 交错）。

> [!warning] 误区 4：忽略显存硬约束
> 算力允许的大 batch 可能被 KV cache 显存卡死（$B_{\max}$）。长序列下 $B_{\max}$ 骤降，"想开大 batch 也开不了"。

> [!warning] 误区 5：固定 batch 不调
> 负载变化时最优 batch 变。低负载小 batch 保延迟，高负载大 batch 保吞吐。固定 batch 要么低峰浪费要么高峰超 SLO。[[continuous batching]] 的动态调节是解法。


## 8. 延伸细节

### 8.1 prefill-decode 分离（Disaggregated Serving）

把 prefill（compute-bound）与 decode（memory-bound）分到不同 GPU 池：prefill 池用大算力 GPU、decode 池用大显存/带宽 GPU，各自独立调 batch。避免 prefill 的大计算阻塞 decode、decode 的大 batch 拖慢 prefill。Mooncake / DistServe 等采用。代价是 KV cache 跨池传输（[[weight sync mechanism]] 类似的传输问题）。

### 8.2 SLO-aware scheduling

调度器不只看吞吐，显式优化 goodput：预测每请求延迟、动态调节 slot/批大小、超 SLO 风险的请求降级或拒绝。vLLM/SGLang 的调度器朝此演进。

### 8.3 与 RLHF 的关系

[[rollout worker]] 采样是 RLHF 训练的时间大头。PD 服务的 batching tradeoff 直接决定 [[sampling throughput]]：SLO 宽松（离线训练）时可拉满 batch 保吞吐；SLO 严格（在线服务+训练）要权衡。[[训推分离]] 架构让 PD 侧能独立调 batch 不影响 PT 训练。

### 8.4 chunked prefill 的作用

长 prompt 的 prefill 切块与 decode 交错，避免大 prefill 长时间独占 GPU 拖高 decode 的 TPOT。这是 batching tradeoff 在 prefill-decode 混合调度上的解法，vLLM/SGLang 默认开启。

### 8.5 实测曲线的获取

生产中需实测 throughput-latency 曲线（不同 batch 跑 benchmark），找拐点。受模型大小、序列分布、GPU 型号影响，无通用最优 batch，需 per-deployment 调参。

---
相关: [[延迟与吞吐优化]] | [[continuous batching]] | [[KV cache management]] | [[GPU utilization]] | [[sampling throughput]]
