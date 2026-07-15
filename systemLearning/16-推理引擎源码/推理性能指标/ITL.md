# ITL

> **所属章节**: [[推理性能指标]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: ITL / inter-token latency / token 间延迟 / 逐 token 间隔 / inter-token delay / token-to-token latency
> **难度**: 中（需懂 [[TTFT与TPOT]]、[[continuous batching的调度实现]]、[[chunked prefill]]、[[KV block manager]]、[[vLLM scheduler]]）


## 1. 一句话定义

**ITL（Inter-Token Latency，token 间延迟）** = decode 阶段**相邻两个输出 token 之间的时间间隔**（第 $i$ 个 token 与第 $i+1$ 个 token 到达时间之差），是比 [[TPOT]] 更细粒度的指标——TPOT 是所有 decode token 间隔的**平均值**，而 ITL 看**逐次间隔本身**，专门暴露**抖动与长尾尖峰**（某次 decode 突然变慢）。在流式输出场景，ITL 尖峰就是用户看到的"字蹦着蹦着突然卡一下"，即使平均 TPOT 达标，单次 ITL 尖峰也会破坏体验，因此 SLO 常用 **P99 ITL** 而非平均 ITL/TPOT。ITL 尖峰主要来自调度层扰动——[[continuous batching的调度实现]] 的 batch 突变、[[chunked prefill]] 混入 prefill chunk 拖慢某次 decode、[[KV block manager]] 的 preempt 重算、KV swap 换出换入、RL 场景的 weight sync——这些都是"平均没事、偶发尖峰"的来源，必须用 ITL（而非 TPOT 均值）才能捕捉。

> [!note] 三句话定位
> - **是什么**：decode 阶段相邻两 token 的间隔（逐次，非平均）。暴露抖动与尖峰。
> - **为什么**：[[continuous batching的调度实现]]/[[chunked prefill]]/preempt/swap/weight sync 会让某次 decode 偶发变慢→ITL 尖峰。TPOT 均值平均掉尖峰，看不出→必须看 ITL（尤其 P99 ITL）。
> - **与 [[TTFT与TPOT]] 关系**：TPOT 是 ITL 的平均。ITL 是 TPOT 的"逐次展开"。ITL 看抖动，TPOT 看均值，P99 ITL 看长尾。


## 2. 为什么需要它（动机与背景）

### 2.1 TPOT 均值平均掉了尖峰

[[TPOT与TPOT]] 中 TPOT = decode token 间隔的**平均**：

$$\text{TPOT} = \frac{1}{n-1}\sum_{i=1}^{n-1}(T_{i+1} - T_i)$$

平均把"大多数快、偶尔慢"抹平。例：99 次 decode 30ms、1 次 600ms（preempt 重算），平均 TPOT ≈ 36ms 看着正常，但那次 600ms 尖峰用户明确感知卡顿。TPOT 达标≠体验好。

```
decode token 间隔 (ITL_i):
  30, 30, 30, 30, 600, 30, 30, 30, 30, 30  (ms)
  TPOT 平均 = 84ms  (看着还行?)
  但 600ms 那次用户明确卡顿 -> 必须 ITL 看逐次
```

### 2.2 流式输出用户对抖动敏感

流式输出（SSE/chunked）逐 token 推送，用户看着字一个个蹦。**稳定快**比"平均快但有尖峰"体验好得多。ITL 尖峰=卡顿=体验破坏，即使平均 TPOT 低。所以流式场景 SLO 用 **P99 ITL**（99% 间隔低于阈值），而非平均。

### 2.3 调度层扰动让 decode 偶发变慢

decode 每步本应稳定（同 batch、同 KV 长度），但推理引擎调度层会引入扰动：

- [[continuous batching的调度实现]] 动态进出请求→batch size 突变→该次 iteration KV 读量突变→ITL 突变。
- [[chunked prefill]] 把 prefill chunk 与 decode 混→含 prefill chunk 的 iteration 慢（算 chunk 多 token）→那次 decode 的 ITL 尖峰。
- [[KV block manager]] 的 preempt（KV 不够驱逐重算）→重算期间 decode 停→ITL 巨尖峰。
- KV swap（swap offload 到 CPU）→换出换入慢→ITL 尖峰。
- RL 场景 weight sync（policy 更新后同步权重）→同步期间 decode 停→ITL 尖峰。

这些扰动是"偶发"的——大多数 iteration 正常、个别尖峰。TPOT 均值看不出，ITL 逐次看才暴露。所以需要 ITL。


## 3. 核心概念详解

### 3.1 ITL 的定义

```
token 到达时间: T1, T2, T3, ..., Tn   (T1 = 首 token)

ITL_i = T_{i+1} - T_i   (第 i 个间隔, i = 1..n-1)
TPOT  = mean(ITL_i)       (平均)
P99 ITL = 99 分位的 ITL_i  (长尾, 见 [[P99延迟]])
```

ITL 是**逐次**间隔，TPOT 是其**均值**，P99 ITL 是其**99 分位**。三者关系：ITL 是原始数据，TPOT/P99 是其统计量。

### 3.2 ITL 尖峰的来源（核心）

```
正常 decode iteration:
  读 KV + 算 1 token forward -> 稳定 ~30ms (ITL 稳)

尖峰来源 (某次 iteration 突然变慢):
┌─────────────────────────────────────────────────────────────┐
│ 1. preempt 重算   │ KV cache 不够, 驱逐该请求, 下次重算     │
│                   │ prefill -> 该次 ITL = prefill 时间 (巨长)│
│ 2. KV swap        │ KV swap 到 CPU, 再 swap 回 -> PCIe 慢   │
│ 3. batch 突变     │ continuous batching 进新请求, B 跳变    │
│                   │ -> KV 读量跳变 -> 该次 ITL 跳            │
│ 4. chunked prefill│ 该 iteration 混了 prefill chunk         │
│                   │ -> 算 chunk 多 token -> decode 被拖慢    │
│ 5. weight sync    │ RL 场景 policy 更新后同步权重           │
│                   │ -> 同步期间 decode 停 -> ITL 尖峰        │
│ 6. HBM/PCIe 争用  │ 其他 kernel 抢带宽 -> 该次 decode 慢    │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 preempt 重算——最尖的尖峰

[[KV block manager]] 的 preempt：KV cache free block 不够时，scheduler 驱逐部分 running 请求（释放其 KV block 给高优/新请求），被驱逐请求下次重新 prefill（重算 prompt 填 KV）。重算期间该请求不产 token→那次 ITL = 整个 prefill 时间（数百 ms 到数秒），是最尖的 ITL 尖峰。

```
正常 ITL: 30ms
preempt 重算 ITL: 200ms ~ 数秒 (整个 prefill 重算)
  -> P99 ITL 被 preempt 尖峰拉爆
```

### 3.4 chunked prefill 混入——周期性小尖峰

[[chunked prefill]] 把长 prefill 切 chunk 与 decode 混。含 prefill chunk 的 iteration 要算 chunk（512 token）+ decode，比纯 decode iteration（1 token）慢→那次 decode 的 ITL 高。若 chunk 周期性出现，ITL 呈周期性小尖峰：

```
纯 decode iter:  ITL = 30ms
含 prefill chunk iter: ITL = 80ms  (chunk 拖慢)
  -> 周期性 80ms 尖峰 (每隔几步一次)
  -> P99 ITL 受影响, 但 TPOT 均值可能仍达标
```

### 3.5 batch 突变——continuous batching 的副作用

[[continuous batching的调度实现]] 动态进出请求。进新请求时 batch size $B$ 跳变（如 $B=16 \to 32$），该次 iteration KV 读量翻倍→TPOT 涨→该次 ITL 涨。出请求（完成）时 $B$ 降→ITL 降。ITL 随 batch 进出而上下跳。

### 3.6 weight sync——RL 场景特有

RLHF/GRPO 训练中，policy 模型更新后需把新权重同步到推理引擎（rollout worker）。同步期间 decode 停（或慢）→那次 ITL 尖峰。是 [[RL训推一体框架]] 的典型 ITL 来源。

### 3.7 ITL 与 TPOT 对比表

| 维度 | ITL | TPOT |
|---|---|---|
| 全称 | Inter-Token Latency | Time Per Output Token |
| 粒度 | **逐次**间隔（每对相邻 token） | **平均**间隔（全部 decode token） |
| 看什么 | 抖动、尖峰、长尾 | 平均每 token 时间 |
| 统计量 | 原始序列 → 看 P99/Pmax | 均值 |
| 捕捉能力 | 捕捉偶发尖峰（preempt/swap/batch 突变） | 平均掉尖峰 |
| SLO 形式 | P99 ITL < 阈值 | TPOT < 阈值 |
| 流式体感 | 直接对应"卡顿感" | 对应"平均蹦字速度" |
| 调度敏感度 | 高（调度扰动直接体现在 ITL） | 低（被平均） |

### 3.8 监控意义

ITL 是**调度层健康度**的敏感指标。TPOT 达标但 P99 ITL 高→调度有扰动（preempt/swap/batch 突变）→需查 [[vLLM scheduler]] 配置（KV 预留、batch 上限、chunk size）。监控 ITL 分布（直方图）能定位是哪种尖峰（周期性→chunked prefill；偶发巨尖峰→preempt；规律性→weight sync）。是推理引擎调优的核心观测。


## 4. 数学原理 / 公式

### 4.1 ITL 定义

设 decode 阶段产出 $n$ 个 token，到达时间 $T_1, T_2, \dots, T_n$（$T_1$ 为首 token）。第 $i$ 个 ITL：

$$\text{ITL}_i = T_{i+1} - T_i, \quad i = 1, 2, \dots, n-1$$

### 4.2 ITL 与 TPOT 的关系

TPOT 是 ITL 的算术平均：

$$\text{TPOT} = \frac{1}{n-1}\sum_{i=1}^{n-1} \text{ITL}_i = \frac{T_n - T_1}{n-1}$$

> [!note] 推导
> $\sum_{i=1}^{n-1}(T_{i+1}-T_i) = T_n - T_1$（望远镜求和）。故 TPOT = $(T_n - T_1)/(n-1)$，只看首末 token，中间抖动全丢。这就是 TPOT 平均掉尖峰的数学原因。

### 4.3 P99 ITL

把 $\{\text{ITL}_i\}$ 排序，99 分位数：

$$\text{P99 ITL} = \text{Percentile}_{0.99}(\{\text{ITL}_i\})$$

即 99% 的间隔低于此值。1% 的间隔高于此值（尖峰）。SLO 常用 P99 ITL < 阈值（如 200ms）。详见 [[P99延迟]]。

### 4.4 尖峰对统计量的影响

设 99 次 $\text{ITL}=30$ms，1 次 $=600$ms（preempt）：

$$\text{TPOT} = \frac{99 \times 30 + 600}{100} = 35.7\text{ms} \quad (\text{看着达标})$$

$$\text{P99 ITL} \approx 600\text{ms} \quad (\text{尖峰暴露, 远超 SLO})$$

TPOT 平均掉尖峰→达标假象；P99 ITL 暴露尖峰→真实体验。这是 ITL 单独存在的数学依据。

### 4.5 chunked prefill 的周期性 ITL 模型

设纯 decode iteration 耗时 $t_d$，含 prefill chunk 的 iteration 耗时 $t_d + t_c$（多算 chunk）。若每 $K$ 步混一次 chunk：

$$\text{ITL}_i = \begin{cases} t_d & \text{纯 decode 步} \\ t_d + t_c & \text{含 chunk 步} \end{cases}$$

$$\text{TPOT} = \frac{(K-1)t_d + (t_d + t_c)}{K} = t_d + \frac{t_c}{K} \quad (\text{均值几乎不涨})$$

$$\text{P99 ITL} \approx t_d + t_c \quad (\text{尖峰暴露, 若含 chunk 步占 >1%})$$

chunked prefill 让 TPOT 几乎不变（$t_c/K$ 小），但 P99 ITL 涨 $t_c$。这是 chunked prefill 拖长尾的数学。

### 4.6 preempt 尖峰的 ITL 模型

正常 ITL = $t_d$。preempt 那次该请求重算 prefill（耗时 $T_{\text{prefill}}(L)$）：

$$\text{ITL}_{\text{preempt}} = T_{\text{prefill}}(L) \gg t_d$$

若 preempt 率为 $p$（如 0.5%）：

$$\text{P99 ITL} \ge \text{preempt 尖峰值} \quad (\text{若 } p \ge 1\%)$$

preempt 率即使很低，只要 $\ge 1\%$，P99 ITL 就被 preempt 尖峰主导。所以降 P99 ITL 的关键是**消除 preempt**（加 KV 预留、用 [[chunked prefill]] 增量分配免 preempt）。


## 5. 代码示例（可选）

### 5.1 测 ITL 序列并算 TPOT/P99

```python
import time
import numpy as np

def measure_itl(token_times):
    """token_times: 各 token 到达时间列表 -> 返回 ITL 序列, TPOT, P99 ITL"""
    itls = [token_times[i+1] - token_times[i] for i in range(len(token_times)-1)]
    itls = np.array(itls)
    return {
        "itls": itls,
        "tpot": itls.mean(),               # 平均 = TPOT
        "p50_itl": np.percentile(itls, 50),
        "p99_itl": np.percentile(itls, 99),
        "pmax_itl": itls.max(),
    }

# 模拟: 99 次 30ms, 1 次 600ms (preempt 尖峰)
rng = np.random.default_rng(0)
normal = rng.normal(0.030, 0.003, size=99)   # 30ms ± 3
spike = [0.600]                               # 600ms preempt
token_times = []
t = 0.0
for d in np.concatenate([normal, spike]):
    t += d
    token_times.append(t)

r = measure_itl(token_times)
print(f"TPOT(均值)     = {r['tpot']*1e3:.1f}ms")
print(f"P50 ITL       = {r['p50_itl']*1e3:.1f}ms")
print(f"P99 ITL       = {r['p99_itl']*1e3:.1f}ms   <- 尖峰暴露")
print(f"Pmax ITL      = {r['pmax_itl']*1e3:.1f}ms")
# TPOT 看着达标, P99 暴露 preempt 尖峰
```

### 5.2 识别尖峰来源（周期性 vs 偶发）

```python
def classify_spikes(itls, threshold_ms=100):
    """按 ITL 尖峰分布猜来源"""
    itls = np.array(itls) * 1e3
    spikes = np.where(itls > threshold_ms)[0]
    if len(spikes) == 0:
        return "无尖峰"
    # 周期性? (等间隔 -> chunked prefill)
    diffs = np.diff(spikes)
    if diffs.std() / (diffs.mean()+1e-9) < 0.3:
        return f"周期性尖峰 (间隔~{diffs.mean():.0f}步) -> 疑 chunked prefill 混入"
    # 偶发巨尖峰? (值很大 -> preempt/swap)
    if itls[spikes].max() > 500:
        return "偶发巨尖峰 -> 疑 preempt 重算 / KV swap"
    return "不规则尖峰 -> 疑 batch 突变 / weight sync / 带宽争用"

# 示例
spike_pattern = [30]*10 + [80] + [30]*10 + [80]   # 周期 -> chunked
print(classify_spikes([x/1e3 for x in spike_pattern]))
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[TTFT与TPOT]]（TPOT 是 ITL 的平均，ITL 是 TPOT 的逐次展开）、[[continuous batching的调度实现]]（batch 突变→ITL 尖峰）、[[chunked prefill]]（混 prefill chunk→周期性 ITL 尖峰）、[[KV block manager]]（preempt→巨 ITL 尖峰）、[[vLLM scheduler]]（调度扰动是 ITL 尖峰主源）。
- **下游（应用）**: [[P99延迟]]（P99 ITL 是 ITL 的尾延迟统计，是流式 SLO 核心）、[[goodput]]（ITL P99 超阈值算 violation，扣 goodput）、流式服务 SLO、推理引擎调优（定位调度扰动）。
- **对比 / 易混**:
  - **ITL vs [[TPOT与TPOT|TPOT]]**：见 3.7 对比表。ITL 逐次看抖动，TPOT 平均看均值。
  - **ITL vs [[P99延迟]]**：ITL 是原始逐次序列，P99 ITL 是其 99 分位。P99 是 ITL 的统计量。
  - **ITL 尖峰 vs [[TTFT与TPOT|TTFT]]**：TTFT 是首 token 延迟（一次性，prefill），ITL 尖峰是 decode 中偶发卡顿（多次，调度扰动）。不同阶段不同源。


## 7. 常见误区与易错点

> [!warning] 误区 1：ITL 就是 TPOT
> 不是。TPOT 是 ITL 的**平均**。ITL 是逐次间隔。TPOT 平均掉尖峰，ITL 暴露尖峰。SLO 用 P99 ITL 不用 TPOT。

> [!warning] 误区 2：TPOT 达标体验就好
> 不一定。TPOT 达标但 P99 ITL 高（偶发尖峰）→流式卡顿。流式场景必须看 P99 ITL。

> [!warning] 误区 3：ITL 尖峰是模型/算力问题
> 多数是**调度层**扰动（preempt/swap/batch 突变/chunked prefill 混入/weight sync），不是模型慢。要查 [[vLLM scheduler]] 配置，不是堆算力。

> [!warning] 误区 4：首 token 到首 token 也算 ITL
> ITL 是 decode 阶段**相邻输出 token**间隔。首 token（T1）之前是 TTFT（prefill），不算 ITL。ITL 从 T2-T1 开始。

> [!tip] 实践：监控 ITL 直方图
> 生产环境监控 ITL 分布直方图，不只看均值。周期性尖峰→查 chunked prefill；偶发巨尖峰→查 preempt/swap；规律性→查 weight sync。直方图能定位来源。

> [!note] 补充：P99 ITL vs P99 TPOT
> "P99 TPOT" 说法不严谨——TPOT 本身是均值（单值），无分位。正确是 **P99 ITL**（对逐次间隔取 99 分位）。文档混用时要辨。


## 8. 延伸细节

### 8.1 ITL 与 goodput

[[goodput]] 的 violation 常用 P99 ITL 超阈值定义（如 P99 ITL > 200ms 算 violation）。ITL 是 goodput 扣分的依据。

### 8.2 chunked prefill 的 ITL 抖动是 tradeoff

[[chunked prefill]] 降 TTFT（首 chunk 早出 token）、免 preempt（增量 KV 分配），但引入周期性 ITL 小尖峰（含 chunk 的 iteration 慢）。是 TTFT↓ vs ITL 尖峰↑ 的 tradeoff。流式敏感场景需调 chunk size 平衡。

### 8.3 数字待核实

文中 30ms/600ms 等为示意量级。实际 ITL 随模型/卡/batch 变。preempt 尖峰可达数秒（重算长 prompt）。精确值需实测。SLO 阈值（如 P99 ITL < 200ms）为常见经验值待核实。

### 8.4 内容来源

整理自 vLLM/SGLang serving metrics（`vllm.engine.metrics` 有 ITL 直方图）、TGI observability 文档、LLM serving SLO 讨论（SARATHI-serve、DistServe 论文关注 ITL 抖动）。TPOT/ITL 区分见 [[TTFT与TPOT]]。preempt 见 [[KV block manager]]。P99 见 [[P99延迟]]。

---
相关: [[推理性能指标]] | [[TTFT与TPOT]] | [[P99延迟]] | [[goodput]] | [[continuous batching的调度实现]] | [[chunked prefill]] | [[KV block manager]] | [[cache eviction与swap offload]] | [[vLLM scheduler]] | [[PD分离]] | [[autoregressive decoding]]
