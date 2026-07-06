# task queue backlog

> **所属章节**: [[工程问题]]
> **所属模块**: [[09-Ray与分布式调度]]
> **别名**: task queue backlog / 任务队列堆积 / backlog / 队列积压
> **难度**: 中（需懂 [[scheduling策略]] + 队列论）


## 1. 一句话定义

**task queue backlog（任务队列堆积）** 是 Ray 中待调度 task 在调度队列里**堆积**的状态——提交速率（submit rate）超过调度/执行速率（schedule/exec rate），队列长度持续增长，导致**调度延迟上升**（task 从提交到开始执行的等待时间变长）、调度器内存占用增长、最终可能 OOM 或超时。它是多种问题的**症状**：可能是 [[resource starvation]]（资源不足排队）、[[actor deadlock]]（actor 卡死不消费）、或单纯吞吐不足（调度器/worker 跟不上提交）。监控 backlog 是诊断 Ray 集群健康的核心指标。

> [!note] 三句话定位
> - **是什么**：待调度 task 在队列堆积，提交快于执行。
> - **为什么**：高并发提交（RLHF 批量采样）或资源不足/调度慢，task 排队等。
> - **定位**：backlog 是**症状**不是根因——需结合 starvation/deadlock/吞吐分析定位。


## 2. 为什么需要它（动机与背景）

### 2.1 提交与执行的速率差

Ray 的 task 提交是异步的：`f.remote(x)` 立即返回 ObjectRef，task 进入调度队列，调度器按资源可用性派发到 worker。若：

- **提交速率** $\lambda$（task/秒）> **执行速率** $\mu$（task/秒，受 worker 数和单 task 耗时限）。

队列长度 $L$ 按 $\lambda - \mu$ 增长。RLHF 批量采样场景：一次提交 1000 个采样请求 task，但集群只有 8 GPU、单采样 1 秒，执行速率 ~8 task/秒，提交瞬间 1000 task → backlog 1000，需 125 秒消化。

### 2.2 三种成因

backlog 增长的三大根因：

1. **资源不足（starvation 类）**：task 需 GPU 但集群 GPU 满，排队等释放。[[resource starvation]]。
2. **消费者卡死（deadlock 类）**：actor 死锁不消费 mailbox，task 在 actor 队列堆积。[[actor deadlock]]。
3. **吞吐不足（纯容量）**：调度器或 worker 处理速度跟不上提交。如调度器毫秒级、每秒只能调度 1000 task，提交 10000 task/秒 → backlog。

### 2.3 后果

- **延迟上升**：task 从提交到执行的等待时间 = backlog / $\mu$。backlog 1000、$\mu$ 8/s → 等 125s。
- **内存增长**：队列里的 task（含参数）占调度器/[[object store]] 内存。大参数（GB 级）堆积可 OOM。
- **超时雪崩**：backlog 长 → 等待超时 → 重试 → 更多 task → backlog 更长。雪崩。

### 2.4 RLHF 场景

- 采样请求 task 批量提交（一个 rollout 阶段提交上千请求）→ backlog。
- PolicyServer actor 死锁不消费请求 → 请求队列堆积。
- 训练 task 排队等 GPU（推理 task 占着）→ 训练 backlog。

监控 backlog 是 RLHF 系统健康的核心。


## 3. 核心概念详解

### 3.1 提交/调度/执行三阶段

```
[submit] → [schedule queue] → [schedule to worker] → [exec on worker] → [result]
```

- **submit**：`f.remote()` 把 task 加到调度队列（瞬时）。
- **schedule**：调度器选 worker 派发（毫秒）。
- **exec**：worker 执行 task（秒~分）。

backlog 可发生在任一阶段：调度队列（资源不足）、actor mailbox（消费者卡死）、worker 队列（吞吐不足）。

### 3.2 Little's Law（队列论）

稳定系统下：

$$
L = \lambda \cdot W
$$

$L$ 是平均队列长度，$\lambda$ 是到达率，$W$ 是平均等待时间。backlog $L$ 与等待时间 $W$ 成正比。若 $\lambda > \mu$（超容量），系统不稳定，$L \to \infty$。

### 3.3 背压（backpressure）

消费者跟不上时，**反向施压**给生产者，限流提交：

- 队列满时拒绝新 task（返回错误/阻塞）。
- 生产者感知消费者负载，降速提交。

背压防止 backlog 无限增长导致 OOM/雪崩。Ray 的 object store 满时会自然背压（写不进 → remote 阻塞），但 task 队列需应用层实现背压。

### 3.4 监控指标

| 指标 | 含义 | 健康阈值 |
|---|---|---|
| **pending task 数** | 调度队列长度 | 稳定不增长 |
| **调度延迟** | submit→exec 等待 | < 阈值（如 1s） |
| **object store 用量** | 待取结果占内存 | < 80% |
| **actor mailbox 深度** | actor 待处理方法数 | 稳定 |

Ray Dashboard 提供这些指标。backlog 持续增长 = 告警。

### 3.5 与 starvation/deadlock 的关系

backlog 是**症状**，starvation/deadlock 是**根因**之一：

- backlog + 资源满 + 公平分配正常 → starvation（某 job 被挤）。
- backlog + actor mailbox 增长 + actor 无进展 → deadlock。
- backlog + 资源有空 + 调度器慢 → 吞吐不足（调度器瓶颈）。

诊断 backlog 需定位是哪类根因。


## 4. 数学原理 / 公式

### 4.1 队列增长模型

到达率 $\lambda$（task/秒），服务率 $\mu$（task/秒）。队列长度 $L(t)$：

$$
\frac{dL}{dt} = \lambda - \mu \quad (L > 0)
$$

- $\lambda < \mu$：$L$ 衰减到 0（消化 backlog）。
- $\lambda = \mu$：$L$ 稳定。
- $\lambda > \mu$：$L$ 线性增长 $\to$ OOM/超时。

> [!note] 稳态条件
> 稳定需 $\lambda < \mu$（到达 < 服务）。RLHF 批量提交 1000 task 瞬时 $\lambda \to \infty$，但执行 $\mu$ 有限 → 暂态 backlog 增长，需等 $\lambda$ 降为 0 后 $\mu$ 消化。若持续高频提交 $\lambda > \mu$，backlog 无限增长 → 必须限流（背压）。

### 4.2 Little's Law

稳态（$\lambda \le \mu$）：

$$
L = \lambda \cdot W, \quad W = \frac{L}{\mu}
$$

backlog $L$ 越大，等待 $W$ 越长。如 $L=1000$、$\mu=8$/s → $W=125$s。

### 4.3 雪崩条件

若 task 有超时 $T_{\text{timeout}}$，超时后重试（新 task）：

$$
\lambda_{\text{effective}} = \lambda + \frac{L}{T_{\text{timeout}}}
$$

backlog $L$ 大 → 重试多 → $\lambda_{\text{effective}}$ 更大 → $L$ 更大 → 雪崩。需背压或长超时避免。

### 4.4 调度器吞吐瓶颈

调度器每秒处理 $S$ 个调度决策（毫秒级，$S \sim 1000$/s）。若提交 $\lambda > S$，调度器自身成瓶颈（task 还没派发就堆积）。分布调度（每 worker 本地调度）降低中心瓶颈。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 backlog + 背压

```python
import collections
class TaskQueue:
    """模拟 Ray 调度队列: 提交/调度/背压."""
    def __init__(self, capacity=10):
        self.queue=collections.deque(); self.capacity=capacity
        self.dropped=0; self.scheduled=0
    def submit(self, task):
        if len(self.queue) >= self.capacity:   # 背压: 满则拒绝
            self.dropped += 1
            return False
        self.queue.append(task)
        return True
    def schedule(self):
        if self.queue:
            self.queue.popleft(); self.scheduled += 1
            return True
        return False

q = TaskQueue(capacity=10)
submit_rate, schedule_rate = 5, 3   # 提交 5/轮, 调度 3/轮 (供不应求)
for t in range(20):
    for _ in range(submit_rate): q.submit(f'task_{t}')
    for _ in range(schedule_rate): q.schedule()
backlog = len(q.queue)
print(f'提交率 {submit_rate} > 调度率 {schedule_rate}:')
print(f'  最终 backlog={backlog}, 已调度={q.scheduled}, 背压丢弃={q.dropped}')
# backlog 持续增长 (5-3=2/轮 × 20=40, 但 capacity=10 背压丢弃多余)
```

### 5.2 真实 Ray API 对照

```python
# import ray
# # Ray Dashboard 看 pending_tasks / scheduling_latency
# # 背压: 应用层限流提交
# MAX_PENDING = 100
# refs = []
# for batch in data:
#     while sum(1 for r in refs if not r.ready()) > MAX_PENDING:
#         ray.wait(refs, num_returns=1)   # 等消化, 背压
#     refs.append(f.remote(batch))
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[scheduling策略]]（调度器吞吐决定消化速率）、[[resource request]]（资源不足导致排队）、[[actor model]]（actor mailbox 也是队列）。
- **下游（应用）**: [[resource starvation]]（starvation 导致 backlog）、[[actor deadlock]]（deadlock 导致 actor mailbox 堆积）、RLHF 采样批量提交设计、[[object store]] 容量管理。
- **对比 / 易混**:
  - **backlog vs [[resource starvation]]**：backlog 是**症状**（队列堆积），starvation 是**根因**（某 task 永远轮不到）。starvation 必导致 backlog，但 backlog 不一定是 starvation（可能是吞吐不足）。
  - **backlog vs [[actor deadlock]]**：deadlock 导致 actor 不消费 → mailbox backlog。deadlock 是根因，backlog 是症状。
  - **task 队列 backlog vs actor mailbox 堆积**：前者是调度器队列（task 等 worker），后者是 actor 方法队列（方法等 actor 线程）。不同阶段。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 backlog 就是 starvation
> backlog 是症状，starvation 是根因之一。backlog 也可能是吞吐不足（调度器/worker 慢）或 deadlock（消费者卡死）。诊断 backlog 需定位根因，不能一概归 starvation。

> [!warning] 误区 2：调度快就不 backlog
> 调度器快（毫秒）只解决"调度队列"backlog。若 worker 执行慢（资源不足/计算重），task 在 worker 队列堆积。backlog 可发生在 submit→schedule→exec 任一阶段。

> [!warning] 误区 3：忽略背压
> 无背压时，提交快于执行 → backlog 无限增长 → OOM/雪崩。生产必须有限流/背压（队列满拒绝、生产者等消化）。

> [!warning] 误区 4：超时设太短加剧雪崩
> task 超时短 + backlog 长 → 大量超时重试 → $\lambda_{\text{effective}}$ 暴增 → backlog 更长 → 雪崩。超时应大于 backlog 消化时间，或用背压而非重试。

> [!warning] 误区 5：只看 backlog 不看等待时间
> backlog 长度需结合消化速率看。backlog 1000 + $\mu$ 1000/s → 等 1s（可接受）；backlog 100 + $\mu$ 1/s → 等 100s（不可接受）。Little's Law：$W = L/\mu$。

> [!warning] 误区 6：忽略 object store 容量
> backlog 的 task 参数/结果存 [[object store]]，大对象堆积可 OOM object store（即使 CPU/GPU 不紧）。需监控 object store 用量。


## 8. 延伸细节

### 8.1 Little's Law 的应用

Little's Law（$L = \lambda W$）是排队论基础，适用于任何稳定排队系统。估算：若可接受 $W \le 1$s、$\lambda = 100$/s → 容忍 $L \le 100$。backlog 超 100 告警。

### 8.2 Ray Dashboard

Ray Dashboard 实时显示：pending task 数、调度延迟、object store 用量、actor 数。生产 RLHF 系统必监控，backlog 增长自动告警/扩容。

### 8.3 RLHF 批量采样 backlog

一次 rollout 提交上千采样请求：

- 瞬时 $\lambda \to \infty$，backlog 暴增。
- 消化 $\mu = N_{\text{gpu}} / T_{\text{sample}}$（8 GPU、1s 采样 → 8/s）。
- 1000 请求 / 8/s = 125s 消化。期间 backlog 1000 → 等 125s。

优化：分批提交（每批 8，等消化再提下一批）、PolicyServer 多副本提 $\mu$、背压限流。

### 8.4 背压的实现

- **应用层限流**：生产者检查 pending 数，超阈值 `ray.wait` 等消化。
- **object store 自然背压**：写满时 `remote` 阻塞。
- **队列容量**：设上限，满则拒绝（fail-fast 触发降级）。

### 8.5 调度器吞吐优化

- **分布式调度**：每 worker 本地调度本地 task，降低中心瓶颈。
- **批调度**：调度器批量派发（一次多 task），减少 per-task 开销。
- **资源预留**：[[placement group]] 避免每次调度找资源。

### 8.6 backlog 与 SLO

推理服务有 SLO（如 P99 延迟 < 1s）。backlog 增长直接抬高 $W$，破 SLO。需容量规划（$\mu \ge \lambda \cdot \text{SLO 系数}$）+ 自动扩容（backlog 告警 → 加 worker）。

---
相关: [[工程问题]] | [[scheduling策略]] | [[resource request]] | [[actor model]] | [[resource starvation]] | [[actor deadlock]] | [[object store]] | [[rollout worker]] | [[continuous batching]] | [[batching tradeoff]]
