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

> [!note] 解答：六种规避方法细说
> 上表每种解法只给了一句话，这里逐条展开**机制 / 代码 / 适用场景 / 代价 / RLHF 落地**。
>
> ### 解法 1：避免环（单向依赖设计）——治本，设计层
> **机制**：actor 调用关系构 DAG，禁止 A→B 且 B→A 的回边。让调用方向单向流动（如 Learner→PolicyServer 单向），需要反向数据流时用**pull 模式**或**共享对象**而非回调。
> **核心模式：single-controller**（verl/OpenRLHF 的做法）——一个 driver 持有所有 actor handle，**driver 单向调 actor，actor 之间不互调**。这从结构上保证无环：driver→A、driver→B、driver→C，A/B/C 之间没有边，自然无环。
> ```python
> # ❌ 有环：A 调 B，B 回调 A
> class A:
>     def m(self, b): return ray.get(b.work.remote(self))  # B 拿到 A 的 ref 会回调
> # ✅ 无环：driver 编排，A/B 不互调
> out_a = ray.get(a.produce.remote(data))      # driver 调 A
> out_b = ray.get(b.consume.remote(out_a))     # driver 调 B，B 不知 A 存在
> ```
> **适用**：新系统设计时。**代价**：要把"actor 互调"重构成"driver 编排"，可能增加 driver 逻辑复杂度。**RLHF 落地**：verl `fit()` 主循环就是 single-controller，actor_rollout_wg/critic_wg 之间从不直接 RPC，全靠 driver `.remote().get()` 编排——天然无环，这是 verl 几乎不死锁的根因。
>
> ### 解法 2：async actor（`max_concurrency>1`）——线程池并发
> **机制**：`@ray.remote(max_concurrency=N)` 让 actor 用**线程池**（N 线程）处理方法，同一 actor 的 N 个方法可并发执行。A 在 `method_a` 阻塞等 B 时，线程池另一线程可处理 B 回调的 `method_a2`，不死锁。
> **数值选择**：推理服务（PolicyServer/vLLM rollout）设 10-100（高并发请求）；有状态训练 actor 设 1（默认串行保一致性）或小值。
> **代价**：并发改同一 actor 的状态 → **必须自己加锁**（`threading.Lock`/`asyncio.Lock`），状态一致性责任从 Ray 转给开发者。详见下面 max_concurrency 那条批注的展开。
> **适用**：actor 既要做阻塞 IO/RPC，又要响应回调；推理服务高并发。**RLHF 落地**：PolicyServer/rollout actor 常用 `max_concurrency>1` 让推理并发。
>
> ### 解法 3：不在 actor 内 `ray.get` 跨 actor——用 future 传递
> **机制**：actor 方法内不调 `ray.get(b.method.remote())`（阻塞占线程），而是**返回 ObjectRef（future）**让调用方（driver）去 get，或用 `ray.wait` 非阻塞轮询。把"同步等待"上提到 driver。
> ```python
> # ❌ actor 内同步 get 跨 actor：危险
> class A:
>     def m(self, b): return ray.get(b.work.remote())  # 占住 A 线程
> # ✅ actor 只返 future，driver get
> class A:
>     def m(self, b): return b.work.remote()           # 返 ObjectRef，不阻塞
> ref = a.m.remote(b)                                  # driver 拿到 future
> result = ray.get(ref)                                # driver get，A 线程早释放了
> ```
> **适用**：任何 actor 方法需要触发另一 actor 工作时。**代价**：调用链变长，driver 要管 future 编排。**RLHF 落地**：verl 的 `_compute_old_log_prob` 等都是 driver 调 `wg.method(batch).get()`，actor 方法内部不再嵌套跨 actor `ray.get`。
>
> ### 解法 4：超时——`ray.get(ref, timeout=...)`
> **机制**：`ray.get(ref, timeout=N)` 在 N 秒内不返回抛 `GetTimeoutError`，actor 线程释放，可走重试/告警。打破"不可剥夺"条件（超时强行剥夺等待）。
> **代价**：**治标不治本**——根因仍是循环依赖设计；超时只是让卡死变成异常而非永久挂。超时值难定（短了误杀长任务，长了仍烧钱）。
> **适用**：兜底防御，所有跨 actor `ray.get` 都该设。**RLHF 落地**：PPO 每步 RPC 设 300s 超时，一个 rollout actor hang 不会让整训练空转。
>
> ### 解法 5：回调 / continuation——非阻塞链式
> **机制**：用 `ray.wait(refs, timeout=0)` 非阻塞查 ready，或 Ray 的 task continuation（`.then()` 风格，老 API）让"完成后接着做"不阻塞当前线程。把"等结果再干下一步"变成"结果就绪时触发回调"。
> **适用**：流水线式异步编排。**代价**：代码从线性变回调式，可读性降。**RLHF 落地**：verl fully_async 模式用 `asyncio` + `ray.wait` 实现 rollout 与训练重叠，属此解法的进阶版（见 [[asynchronous training]]）。
>
> ### 解法 6：watchdog——检测 + 重启
> **机制**：周期性探活（actor 心跳/方法 ping），超时无响应判卡死，`ray.kill(actor)` 后重建。打破"不可剥夺"（强制杀）。
> **代价**：**重启丢 actor 内存状态**（模型权重/buffer），需重建（重新加载权重）。是兜底而非首选。误判（把慢但活的 actor 当死）会无谓重启。
> **适用**：生产容错兜底。**RLHF 落地**：训练框架常自建 watchdog 监控 actor 心跳 + mailbox 堆积（配合 [[Ray调试手段]] 的 dashboard/ray list）。
>
> ### 优先级口诀
> **设计层避免环（解法1）> 编码层不阻塞（解法3/5）> 并发层 async（解法2）> 兜底层超时+watchdog（解法4/6）**。从左到右，越左越治本越早介入，越右越治标越被动。verl 靠解法 1（single-controller 无环）+ 解法 3（actor 不嵌套 get）几乎免疫死锁，解法 2/4/6 是补充。
### 3.4 async actor（max_concurrency）

```python
@ray.remote(max_concurrency=10)
class A: ...
```

`max_concurrency=N` 让 actor 用线程池（N 并发）处理方法，不串行。A 在 `method_a` 等 B 时，线程池其他线程可处理 B 回调的 `method_a2`，不死锁。但代价是**并发改状态**需自己加锁（asyncio/lock）——状态一致性责任从 Ray 转给开发者。

> [!note] 解答：`max_concurrency` 是什么意思
> 这是 Ray actor 的一个**并发度参数**，决定"同一 actor 同时能跑几个方法调用"。很多人没听过，因为**默认值是 1（串行）**，不显式设就感受不到它的存在。下面逐层拆。
>
> ### 一、默认行为：actor 方法串行（`max_concurrency=1`）
> Ray actor 默认像一把单线程的"执行槽"：同一时刻**只有一个方法在执行**，其他方法调用进 actor 的 **mailbox 队列**排队，等当前方法返回才轮到下一个。这是 Ray 保证 **actor 状态一致性**的设计——串行执行就不会有两个方法并发改同一份状态。
> ```
> 调用 t1 ──┐
>           ├──> actor [单线程执行槽]── 执行中, t2/t3 在 mailbox 排队
> 调用 t2 ──┤
> 调用 t3 ──┘
> ```
> 这种串行**安全但低效**：一旦方法内 `ray.get` 阻塞等别人，整个 actor 卡住，mailbox 里排队的调用全等——既吞吐低（见 §4.2 吞吐退化公式 $1/(T+T_{\text{wait}})$）又易死锁（§2.2 循环依赖）。
>
> ### 二、`max_concurrency=N`：线程池，N 并发
> ```python
> @ray.remote(max_concurrency=10)
> class PolicyServer:
>     def infer(self, req): ...
> ```
> 设了 `max_concurrency=10` 后，Ray 给该 actor 配一个**10 线程的线程池**，同一 actor 的**最多 10 个方法可并发执行**。mailbox 不再串行排队——只要线程池有空槽就立即调度新方法。于是：A 在 `method_a` 里 `ray.get(b...)` 阻塞时，线程池另一线程能处理 B 回调的 `method_a2`，**循环等待被打破，不死锁**。
> ```
> 调用 t1 ──┐
> 调用 t2 ──┼──> actor [10 线程池]── t1/t2/.../t10 可并发
> 调用 t3 ──┤     t1 阻塞时, 其他线程仍可处理新调用
> ...        │
> 调用 t11 ──┘  超过 10 的进 mailbox 排队
> ```
>
> ### 三、数值怎么选
> | 场景 | 建议值 | 理由 |
> |---|---|---|
> | 有状态训练 actor（learner） | 1（默认） | 状态多、并发改风险高，保串行 |
> | 推理服务 actor（PolicyServer/vLLM rollout） | 10-100 | 高并发请求，阻塞 IO 多，需要并发吞吐 |
> | 调度/编排 actor | 1 | 逻辑串行即可 |
> | 既要阻塞等待又要响应回调 | >1（解死锁） | 否则回调进不去 |
>
> 经验：**能串行就串行**（默认 1 最安全），只有"方法会阻塞且需响应并发请求"时才调大。Ray 官方推荐从默认开始，遇到吞吐/死锁问题再调。
>
> ### 四、代价：状态一致性责任转移
> 串行（默认）时，Ray 替你保证"两个方法不会并发改状态"——你写 actor 像写单线程程序，不用考虑锁。一旦 `max_concurrency>1`，**并发方法可能同时读写 `self.xxx`**，状态一致性责任从 Ray 转给你：
> ```python
> @ray.remote(max_concurrency=10)
> class Counter:
>     def __init__(self): self.n = 0; self.lock = threading.Lock()
>     def inc(self):
>         with self.lock:           # 必须！否则并发 inc 丢更新
>             self.n += 1
>     def get(self): return self.n
> ```
> 忘加锁 → 数据竞争（丢更新/撕裂读）。这是 §7 误区 3"max_concurrency 不是免费的"的根因。
>
> ### 五、vs asyncio actor（更安全的并发解）
> Ray 还支持 **asyncio actor**（class 用 `async def` 方法）：
> ```python
> @ray.remote
> class A:
>     async def m(self, b):
>         x = await b.work.remote()    # await 不阻塞线程, 用协程
>         return x
> ```
> asyncio actor 是**单线程 + 协程并发**：`await` 时不占线程（让出给其他协程），能响应回调，又因为单线程**不会真并发改状态**（无数据竞争）。所以 asyncio actor 比 `max_concurrency` 线程池**更安全**——并发度够用又不用加锁。**优先用 asyncio actor**，只有方法不能用 `async def`（如调同步库）时才退回 `max_concurrency` 线程池。
>
> ### 六、一句话总结
> `max_concurrency` = actor 的"执行槽位数"。默认 1 串行安全但易死锁/低吞吐；调大用线程池并发破死锁提吞吐，但代价是自己加锁保状态；更优解是 asyncio actor（协程并发不阻塞又不数据竞争）。RLHF 里推理 actor 常调大、训练 actor 保默认，verl 的 single-controller 设计让大多数 actor 根本不需要调大（无环不死锁）。
>
> 相关：[[actor model]]、[[task model]]、[[Ray与分布式调度]] §3.4、Ray docs `max_concurrency`。
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
