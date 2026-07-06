# actor deadlock

> **所属章节**: [[工程问题]]
> **所属模块**: [[09-Ray与分布式调度]]
> **别名**: actor deadlock / actor 死锁 / 循环等待死锁
> **难度**: 中高（需懂 [[actor model]] + 并发）


## 1. 一句话定义

**actor deadlock** 是 Ray 中多个 [[actor model]] 互相等待对方响应（**循环依赖**）导致全部阻塞的死锁：actor 方法默认**串行执行**（单线程），A 执行中调 B、B 执行中调 A → A 等 B 返回、B 等 A 响应，两者 mailbox 都在排队对方的方法、执行线程都在等对方，**永久卡死**。它是 actor 有状态串行的副作用——[[task model]]（无状态并行）不会死锁，actor 因状态串行易在循环依赖下死锁。

> [!note] 三句话定位
> - **是什么**：多个 actor 循环等待对方响应，全部卡死。
> - **为什么**：actor 方法默认串行（单线程），A 调 B、B 调 A 时双方执行线程在等、mailbox 在排队，循环等待无解。
> - **典型场景**：PolicyServer 互调（A 推理时调 B 取数据、B 处理时调 A 推理）、actor 内 `ray.get` 阻塞另一 actor 调用。


## 2. 为什么需要它（动机与背景）

### 2.1 actor 串行执行的本质

[[actor model]] 的方法默认**串行**（单线程）：一个方法执行时，其他方法在 mailbox 排队等。这是保证状态一致的设计（避免并发改状态）。但串行意味着：方法执行中若调另一 actor 并 `ray.get` 阻塞，该 actor 的线程被占住，无法处理新方法（包括对方回调它的方法）。

### 2.2 循环依赖导致死锁

考虑两个 actor A、B：

- A 的 `method_a` 调 `b.method_b.remote(a)` 并 `ray.get` 等结果。
- B 的 `method_b` 调 `a.method_a2.remote()` 并 `ray.get` 等结果。

执行流程：

1. A 执行 `method_a`，调 B 的 `method_b`，A 线程**阻塞**等 B 返回。
2. B 执行 `method_b`，调 A 的 `method_a2`，B 线程**阻塞**等 A 返回。
3. A 的 `method_a2` 排在 A 的 mailbox，但 A 线程在 `method_a` 阻塞、**无法处理** mailbox。
4. A 等 B、B 等 A → **永久死锁**。

### 2.3 task 不死锁、actor 易死锁的原因

[[task model]] 无状态、算完即走、不保留执行线程状态 → 无循环等待。actor 有状态、方法串行、执行线程持有"状态锁" → 易循环等待。这是 actor 相对 task 的代价——灵活性换来了死锁风险。

### 2.4 RLHF 中的死锁风险

- PolicyServer 互调：A 推理时调 B 取 buffer 数据，B 处理时调 A 推理 → 环。
- Learner 调 PolicyServer 推理，PolicyServer 调 Learner 取最新权重 → 环。
- 多 actor 协作的 RLHF 系统若设计不慎，循环依赖普遍。


## 3. 核心概念详解

### 3.1 死锁四必要条件（经典 Coffman）

1. **互斥**：actor 方法串行（一次一个），状态独占。
2. **持有并等待**：A 执行 method_a（持有 A 线程）时调 B（等待 B）。
3. **不可剥夺**：不能强行打断 actor 方法（除非超时）。
4. **循环等待**：A 等 B、B 等 A 形成环。

四条件同时满足才死锁。打破任一即可避免（见 §3.3）。

### 3.2 actor 内 ray.get 的危险

在 actor 方法内 `ray.get(ref)` 阻塞会**占住 actor 线程**：

```python
@ray.remote
class A:
    def m(self, b_ref):
        x = ray.get(b_ref.m2.remote(self))   # 阻塞, 占住 A 线程
        return x
```

若 `b_ref.m2` 内回调 A 的方法，A 线程被占、无法响应 → 死锁。这是 actor 死锁最常见成因——**actor 内同步 ray.get 另一 actor**。

### 3.3 避免与解法

| 解法 | 机制 | 打破的条件 |
|---|---|---|
| **避免环** | 设计单向依赖（无循环调用） | 循环等待 |
| **async actor** | `max_concurrency > 1`，方法可并发 | 互斥 |
| **不在 actor 内 ray.get** | 用 future 传递、不阻塞 | 持有并等待 |
| **超时** | `ray.get(ref, timeout=...)` | 不可剥夺 |
| **回调/continuation** | 用 `.then()` 风格，不阻塞 | 持有并等待 |
| **watchdog** | 检测卡死 actor，重启 | 不可剥夺 |

### 3.4 async actor（max_concurrency）

```python
@ray.remote(max_concurrency=10)
class A: ...
```

`max_concurrency=N` 让 actor 用线程池（N 并发）处理方法，不串行。A 在 `method_a` 等 B 时，线程池其他线程可处理 B 回调的 `method_a2`，不死锁。但代价是**并发改状态**需自己加锁（asyncio/lock）——状态一致性责任从 Ray 转给开发者。


## 4. 数学原理 / 公式

### 4.1 资源分配图与环检测

actor 作为节点，调用关系作有向边：A→B（A 等 B）。死锁当且仅当图中有**环**：

$$
\text{deadlock} \iff \exists \text{ cycle } A_1 \to A_2 \to \cdots \to A_n \to A_1
$$

环检测（DFS / 拓扑排序）是死锁检测的算法基础。Ray 无内置环检测，需设计时避免或 watchdog 探测。

### 4.2 串行 actor 的吞吐退化

单 actor 串行：吞吐 $1/T$（$T$ 单方法耗时）。若 actor 内 `ray.get` 阻塞 $T_{\text{wait}}$，吞吐降到 $1/(T+T_{\text{wait}})$。高 $T_{\text{wait}}$ 时 actor 实际闲置——既吞吐低又易死锁。async actor 让等待时处理其他方法，吞吐回升。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 actor 死锁

```python
import threading
class ActorSim:
    """模拟 Ray actor: 串行 (busy 时新调用排队 -> 检测死锁)."""
    def __init__(self, name): self.name=name; self.busy=False
    def call(self, method, *args):
        if self.busy:                           # actor 串行, busy 时无法响应
            return f'DEADLOCK: {self.name} busy, cannot serve {method}'
        self.busy=True
        try: return getattr(self, method)(*args)
        finally: self.busy=False
    def method_a(self, peer):
        # A 调 B.method_b(A), B 回调 A.method_a2 -> A busy, 死锁
        return peer.call('method_b', self)
    def method_b(self, peer):
        return peer.call('method_a2')
    def method_a2(self): return 'ok'

a=ActorSim('A'); b=ActorSim('B')
# 触发死锁: A 正在 method_a (busy), B 回调 A.method_a2 失败
a.busy=True                                        # 模拟 A 在 method_a 等 B
print('串行 actor 死锁:', b.call('method_b', a))   # DEADLOCK
```

### 5.2 解法：async actor（max_concurrency）

```python
class AsyncActorSim(ActorSim):
    """模拟 max_concurrency>1: 不检查 busy, 允许并发."""
    def call(self, method, *args):
        return getattr(self, method)(*args)        # 不串行, 可响应回调

a2=AsyncActorSim('A'); b2=AsyncActorSim('B')
print('async actor 解法:', b2.call('method_b', a2))   # ok (A 可响应 method_a2)
```

### 5.3 真实 Ray API 对照

```python
# import ray
# @ray.remote
# class A:
#     def m(self, b): return ray.get(b.m2.remote(self))   # 危险: actor 内 ray.get
# @ray.remote(max_concurrency=10)  # 解法1: async actor
# class Aasync:
#     def m(self, b): return ray.get(b.m2.remote(self))
# # 解法2: 超时
# ray.get(ref, timeout=5)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[actor model]]（串行执行是死锁根源）、[[remote function]]（`.remote` + `ray.get` 阻塞）、[[scheduling策略]]（调度不当加剧）。
- **下游（应用）**: [[task queue backlog]]（死锁导致 mailbox 堆积）、[[resource starvation]]（死锁 actor 占资源不释放）、RLHF 多 actor 协作设计。
- **对比 / 易混**:
  - **actor 死锁 vs [[task model]] 不死锁**：task 无状态并行、算完即走，无循环等待；actor 有状态串行、持有线程，易环。死锁是 actor 特有问题。
  - **actor 死锁 vs [[resource starvation]]**：死锁是循环等待卡死（无进展），starvation 是资源不足排队（有进展但慢）。不同问题。
  - **actor 死锁 vs [[task queue backlog]]**：死锁导致 mailbox 堆积（backlog 的成因之一），但 backlog 也可能是吞吐不足。死锁是 backlog 的特殊成因。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 task 也会死锁
> 不会。[[task model]] 无状态、算完即走、不持有执行线程状态 → 无循环等待。死锁是 actor 有状态串行的特有问题。若"task 死锁"，通常是 actor 内 ray.get 阻塞导致（仍是 actor 死锁）。

> [!warning] 误区 2：在 actor 内随意 ray.get
> actor 内 `ray.get(ref)` 阻塞占住 actor 线程。若 ref 依赖本 actor 回调（环），死锁。actor 内应避免同步 ray.get 另一 actor，或用 async actor / future 传递。

> [!warning] 误区 3：以为 max_concurrency 是免费的
> async actor（max_concurrency>1）解死锁，但并发改状态需自己加锁（asyncio/lock），状态一致性责任转移。不是免费——换来死锁避免，付出并发复杂度。

> [!warning] 误区 4：忽略循环依赖设计
> 多 actor 协作时若设计成环（A 调 B、B 调 A），默认串行 actor 必死锁。设计时应单向依赖（如 Learner 调 PolicyServer，PolicyServer 不回调 Learner，而用 pull 权重）。

> [!warning] 误区 5：用 watchdog 重启会丢状态
> watchdog 检测卡死 actor 重启，但 actor 状态丢失（[[actor model]] 状态在内存）。重启 actor 需重建状态（如重加载模型权重）。watchdog 是兜底，不是首选。


## 8. 延伸细节

### 8.1 RLHF 中的死锁规避设计

verl/OpenRLHF 的设计：

- **单向依赖**：Learner 调 PolicyServer 推理，PolicyServer **不回调** Learner。PolicyServer 用 pull 模式取权重（Learner 不主动推）。
- **不在 actor 内 ray.get 跨 actor**：用 future 传递，或用 Ray Core 的 `ray.wait` 非阻塞。
- **PolicyServer 用 async actor**：`max_concurrency` 让推理并发，避免请求堆积。

### 8.2 asyncio actor

Ray 支持 asyncio actor（`@ray.remote` class 用 async def 方法）。asyncio 在单线程内并发（协程），避免真线程并发改状态的问题，同时不阻塞。是 async actor 的更优解（比 max_concurrency 线程池更安全）。

### 8.3 超时与 fail-fast

`ray.get(ref, timeout=5)` 设超时，超时抛 `GetTimeoutError`，避免永久卡死。生产环境必设超时，fail-fast 触发重试/告警。但超时治标不治本——根因仍是循环依赖设计。

### 8.4 分布式死锁检测

Ray 无内置分布式死锁检测。高级场景可用：周期性探活（actor 心跳）、mailbox 堆积监控（[[task queue backlog]]）、环检测算法分析调用图。生产 RLHF 系统常自建 watchdog。

### 8.5 与传统线程死锁的对比

actor 死锁是**分布式版本的线程死锁**：线程死锁是锁循环等待，actor 死锁是 actor 调用循环等待。Coffman 四条件相同。解法也类似（避免环、剥夺、检测恢复）。actor 死锁更难调试（跨进程跨节点）。

---
相关: [[工程问题]] | [[actor model]] | [[remote function]] | [[task model]] | [[scheduling策略]] | [[task queue backlog]] | [[resource starvation]] | [[policy training (PT)]] | [[rollout worker]] | [[policy deployment (PD)]]
