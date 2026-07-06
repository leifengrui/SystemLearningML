# dynamic batching

> **所属章节**: [[推理优化]]
> **所属模块**: [[08-推理系统]]
> **别名**: dynamic batching / 动态批处理 / 请求级动态凑批
> **难度**: 中（需懂调度 + 与 [[continuous batching]] 区分）


## 1. 一句话定义

**dynamic batching（动态批处理）** 是推理服务在**请求级**动态凑批的策略：请求随时间到达服务端，batcher 在"**凑够 max_queue_size 个**"或"**等够 max_queue_delay 时间**"任一条件满足时，把当前积压的请求聚成一个 batch 一起前向，而非逐个串行处理或用固定大小批。它是 NVIDIA Triton Inference Server 的招牌特性，**粒度仍是请求级**（一批跑完才发下一批），区别于 [[continuous batching]] 的 token 级。

> [!note] 三句话定位
> - **是什么**：到达时攒批，阈值或超时触发即发，整批跑完才换下一批。
> - **为什么**：单请求 GPU 利用率低，攒批提吞吐；但等满批又增延迟——dynamic 在两者间权衡。
> - **和 continuous 的区别**：dynamic 是"何时凑批"（请求级），continuous 是"批内如何调度"（token 级）。两者可叠加。


## 2. 为什么需要它（动机与背景）

### 2.1 三个极端都不好

推理服务的请求到达是**随机流**（Poisson 近似）。处理方式有三档：

| 策略 | 做法 | 问题 |
|---|---|---|
| 逐请求串行 | 来一个算一个 | GPU 利用率极低（batch=1，算力闲置） |
| 固定大小批 | 攒够固定 K 个才发 | 低峰时永远凑不齐，请求饿死 |
| 立即发 | 来一个发一个 | 等同串行，没攒到批 |

**dynamic batching** 用"**阈值或超时双触发**"破局：

- 凑够 `max_queue_size` 个 → **立即发批**（不等超时，吞吐优先）。
- 没凑够但等够 `max_queue_delay` → **超时发批**（用当前积压的请求数发，延迟优先）。

这样高峰时按阈值快速发批（高吞吐），低谷时按超时发批（低延迟），自适应负载。

### 2.2 与 static / continuous 的位置

| 维度 | static（固定批） | dynamic（请求级动态凑批） | continuous（token 级迭代调度） |
|---|---|---|---|
| 凑批时机 | 固定大小，满才发 | 阈值或超时双触发 | 不"凑批"，slot 边跑边换 |
| 调度粒度 | 请求级 | 请求级 | token/iteration 级 |
| 换批时机 | 整批跑完 | 整批跑完 | 每 iteration |
| 适合 | 离线批量 | 传统模型服务（CV/NLP 分类） | LLM 生成（变长输出） |

> [!tip] LLM 场景下 dynamic 为何不够
> 传统模型（ResNet/BERT 分类）输出定长，一批跑完时间相近，dynamic 够用。但 LLM 生成**输出长度差异巨大**（20 vs 800 token），dynamic 的"整批跑完才换"会让短请求空等长请求（[[continuous batching]] §2.1 的痛点）。所以 LLM 服务转向 continuous batching（token 级），dynamic 退居为"到达层凑批"的辅助。


## 3. 核心概念详解

### 3.1 双触发：max_queue_size + max_queue_delay

- **max_queue_size**（凑批阈值）：积压请求数达到此值立即发批。控制吞吐——高峰时快速凑齐大批。
- **max_queue_delay**（超时阈值）：最老请求在队列里等够此时间就发批（无论攒了多少）。控制延迟——低谷时不让请求干等。

两者任一满足即发。这构成一个**自适应**：负载高时 size 触发（大批高吞吐），负载低时 delay 触发（小批低延迟）。

### 3.2 Triton 的实现

NVIDIA Triton Inference Server 的 dynamic batching：

- 配置 `max_queue_delay_microseconds` 与 `max_batch_size`。
- 后台线程持续检查队列，双触发即发。
- 支持 `priority` 队列、`preserve_ordering`（保序）、`preferred_batch_size`（多个候选大小，如 [2,4,8]，攒到任一即发以利用 GPU 不同 batch 的吞吐曲线）。

### 3.3 preferred_batch_size（优选批大小）

GPU 对不同 batch 大小有吞吐"甜点区"（如 batch 8 和 16 都很高，batch 12 反而因 padding 低效）。`preferred_batch_size=[2,4,8]` 让 batcher 攒到 2/4/8 任一就优先发，否则超时兜底。这是对朴素双触发的细化。

### 3.4 与 continuous 的叠加

在 LLM 服务里，两层可叠加：

1. **外层 dynamic**：请求到达层，按阈值/超时把请求送进推理引擎的等待队列。
2. **内层 continuous**：引擎内 token 级迭代调度，slot 动态进出。

即 dynamic 决定"哪些请求一起进引擎"，continuous 决定"进引擎后怎么调度"。vLLM 主要实现内层 continuous，外层由前端（如 LitServe/Triton frontend）补。


## 4. 数学原理 / 公式

### 4.1 Little's Law（排队论基础）

稳态排队系统：$L = \lambda W$

- $L$ = 系统内平均请求数（队列+处理中）
- $\lambda$ = 请求到达率（req/s）
- $W$ = 平均逗留时间（等待+处理）

dynamic batching 的目标是在 $W$（延迟）与 $\lambda$（吞吐）间权衡。批大小 $B$ 越大，单次前向吞吐越高（GPU 利用率高），但凑批等待时间 $\uparrow$，$W \uparrow$。

### 4.2 凑批等待时间

设到达率 $\lambda$，凑批阈值 $B$。近似下，攒满 $B$ 个的期望等待：

$$
E[\text{凑批等待}] \approx \frac{B}{\lambda}
$$

超时阈值 $T_{\max}$ 兜底：实际凑批等待 $= \min(B/\lambda, T_{\max})$（粗略）。

- 高负载（$\lambda$ 大）：$B/\lambda < T_{\max}$，size 触发，等待短，批大。
- 低负载（$\lambda$ 小）：$B/\lambda > T_{\max}$，delay 触发，批小但延迟有上界。

### 4.3 吞吐-延迟权衡

单次前向耗时 $T_{\text{fwd}}(B)$（随 $B$ 缓增，因 GPU 并行）。吞吐 $\approx B / T_{\text{fwd}}(B)$（$B$ 大时趋于 GPU 峰值）。延迟 $W \approx \text{凑批等待} + T_{\text{fwd}}(B)$。

增大 $B$：吞吐 $\uparrow$，但凑批等待 $\uparrow$、延迟 $\uparrow$。这就是 [[batching tradeoff]] 的本质——dynamic batching 是这一权衡的请求级实现。


## 5. 代码示例（可选

### 5.1 简化 dynamic batcher（双触发）

```python
import time

class DynamicBatcher:
    """简化 dynamic batching: max_size 或 max_delay 任一满足即发批."""
    def __init__(self, max_size=4, max_delay=1.0):
        self.max_size, self.max_delay = max_size, max_delay
        self.queue, self.batch_id = [], 0
    def submit(self, req):
        self.queue.append((time.time(), req))
    def try_flush(self, now):
        """检查是否应发批, 返回 (batch_id, batch) 或 None."""
        if not self.queue: return None
        oldest = self.queue[0][0]
        if len(self.queue) >= self.max_size or (now - oldest) >= self.max_delay:
            batch = [r for _, r in self.queue]
            self.queue, self.batch_id = [], self.batch_id + 1
            return (self.batch_id, batch)
        return None

# 模拟请求随时间到达
b = DynamicBatcher(max_size=3, max_delay=1.0)
arrivals = [(0.0,'A'),(0.2,'B'),(0.4,'C'),(0.6,'D'),(1.2,'E'),(1.4,'F')]
t = 0.0
for at, r in arrivals:
    while t < at:                       # 推进时间, 期间检查超时触发
        flushed = b.try_flush(t)
        if flushed: print(f't={t:.1f}: flush #{flushed[0]} = {flushed[1]}')
        t += 0.2
    b.submit(r); print(f't={at:.1f}: submit {r}, queue={len(b.queue)}')
# 收尾: 把剩余的冲走
for _ in range(10):
    flushed = b.try_flush(t)
    if flushed: print(f't={t:.1f}: flush #{flushed[0]} = {flushed[1]}')
    t += 0.2
```

预期：A,B,C 攒满 3 → size 触发发批 #1；D 单独等到 t≈1.6 超时 → delay 触发发批 #2；E,F 类似。这演示了双触发的自适应。

### 5.2 对比：static 与串行

```python
def static_batch(requests, K):
    """固定大小 K, 攒满才发."""
    return [requests[i:i+K] for i in range(0, len(requests), K)]
# 低负载时永远凑不满 -> 请求饿死

def serial(requests):
    """逐个处理."""
    return [[r] for r in requests]
# GPU 利用率最低
```


## 6. 与其他知识点的关系

- **上游（依赖）**: 无强依赖（通用调度概念），但落地在推理服务框架（Triton）。
- **下游（应用）**: [[batching tradeoff]]（dynamic 是请求级权衡的实例）、[[continuous batching]]（LLM 场景的 token 级升级）、[[GPU utilization]]（dynamic 提利用率的目标）。
- **对比 / 易混**:
  - **dynamic batching vs [[continuous batching]]**：**最常被混淆的一对**。dynamic = 请求级凑批（何时攒批），整批跑完才换；continuous = token 级调度（批内怎么换人），每 iteration 可换。粒度差一个量级。
  - **dynamic batching vs static batching**：dynamic 用阈值/超时双触发自适应凑批，static 用固定大小满才发。
  - **dynamic batching vs in-flight batching**：in-flight ≈ continuous（token 级），不是 dynamic。


## 7. 常见误区与易错点

> [!warning] 误区 1：把 dynamic batching 等同于 continuous batching
> **这是最常见的混淆**。两者都叫"动态批"，但：
> - dynamic 的"动态"指**凑批时机动态**（阈值/超时），凑够后**整批一起跑完才换**。
> - continuous 的"动态"指**批内成员动态**（每 token 边界换人）。
> 一个是"何时凑批"，一个是"凑批后怎么调度"。LLM 生成场景因输出变长，dynamic 的整批跑完会拖累短请求，故转向 continuous。

> [!warning] 误区 2：以为 dynamic batching 是 LLM 推理的"主力"
> 在 LLM 生成服务里，主力是 [[continuous batching]]（token 级）。dynamic 更多用于传统模型（分类/检测，定长输出）服务，或作为 LLM 服务的外层到达层凑批。把它当 LLM 推理核心调度就搞错了层次。

> [!warning] 误区 3：忽略 max_queue_delay 的兜底作用
> 只设 max_queue_size 不设 delay，低负载时请求永远凑不齐 → 饿死。delay 是延迟上界保障，不可省。

> [!warning] 误区 4：以为 batch 越大吞吐一定越高
> 大 batch 提利用率，但凑批等待也增（延迟↑），且超过 GPU "甜点区"后 padding/显存开销反噬。`preferred_batch_size` 就是为此选甜点。


## 8. 延伸细节

### 8.1 Triton 的 dynamic batching 配置

```protobuf
dynamic_batching {
  preferred_batch_size: [2, 4, 8]
  max_queue_delay_microseconds: 100000   # 100ms
  preserve_order: true                   # 保序（同请求内先来先服务）
}
```

Triton 还支持 `priority` 队列（高优先请求先入批）、`queue_policy`（超时丢弃 vs 保留）。

### 8.2 LLM 场景：dynamic 为何退居辅助

LLM 生成的输出长度随机（20–2000 token 不等）。dynamic 凑批后整批跑，最短请求生成完仍要等最长请求，slot 大量空转（[[continuous batching]] §2.1）。所以 LLM 服务（vLLM/TGI/SGLang）用 continuous batching 做内核调度，dynamic 只在前端做请求准入/限流。

### 8.3 与 RLHF 的关系

[[rollout worker]] 采样时，若用第三方推理服务（Triton 部署的模型）做生成，dynamic batching 决定请求如何凑批进引擎；引擎内若是 LLM 专用（vLLM），则 continuous 接管。两层共同决定 [[sampling throughput]]。在 [[policy deployment (PD)]] 架构里，PD 服务侧的这两层调度直接影响 RLHF 训练的采样瓶颈。

---
相关: [[推理优化]] | [[continuous batching]] | [[batching tradeoff]] | [[GPU utilization]]
