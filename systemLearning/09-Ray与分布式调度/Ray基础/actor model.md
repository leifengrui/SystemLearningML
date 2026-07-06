# actor model

> **所属章节**: [[Ray基础]]
> **所属模块**: [[09-Ray与分布式调度]]
> **别名**: Ray actor / actor / 有状态远程对象
> **难度**: 中（需懂分布式编程 + [[remote function]]）


## 1. 一句话定义

**Ray actor（actor model）** 是 Ray 中**有状态的远程对象**：用 `@ray.remote` 装饰一个 class，实例化后在某个 worker 进程中**常驻**，外部通过 **actor handle** 异步调用其方法，方法在 actor 进程内**顺序（串行）执行**，状态（如模型权重、optimizer、replay buffer）跨方法保持。它是 Ray 区别于纯 [[task model]]（无状态函数）的核心抽象，RLHF 中 [[policy training (PT)]] 的 Learner、[[rollout worker]] 的 PolicyServer、[[replay buffer]] 都以 actor 实现。

> [!note] 三句话定位
> - **是什么**：远程常驻的有状态对象，方法异步调用、串行执行。
> - **为什么**：[[task model]] 无状态，每次调用要序列化传输状态，开销大且无法保持（如模型权重反复传输）。actor 让状态常驻 worker。
> - **典型用途**：模型推理服务、训练器、replay buffer、参数服务器——所有"有状态"的分布式组件。


## 2. 为什么需要它（动机与背景）

### 2.1 task model 的状态困境

Ray 的 [[task model]] 是无状态远程函数：每次 `ray.get(f.remote(x))` 起一个 task，算完返回，进程不保留状态。若一个组件**有状态**（如 LLM 推理服务要常驻模型权重、训练器要保 optimizer state、replay buffer 要累积数据），用 task model 会：

- 每次调用**重新加载/序列化状态**（模型权重 GB 级，传输成本极高）。
- 状态无法跨调用保持（每次 task 是新进程）。
- 并发写状态需外部协调（task 间无共享内存）。

### 2.2 actor 的解法

actor 让**有状态对象常驻 worker**：

- `@ray.remote` 装饰 class → 实例化为 actor，驻留某 worker 进程。
- actor 进程持有对象状态（权重、buffer 等），跨方法调用保持。
- 方法调用经 actor handle 异步提交到 actor 的**邮箱队列**，actor **串行处理**（保证状态一致，无并发写）。
- 状态不传输（数据 locality），只传方法参数与返回值（ObjectRef）。

> [!tip] 直觉
> task 是"无状态函数，算完即走"；actor 是"有状态对象，常驻服务"。前者适合纯计算（如一次前向），后者适合需保持状态的组件（如常驻模型的服务、累积数据的 buffer）。


## 3. 核心概念详解

### 3.1 定义与使用

```python
import ray
ray.init()

@ray.remote(num_gpus=1)           # 装饰为 actor, 声明用 1 GPU
class PolicyServer:
    def __init__(self, model):
        self.model = model         # 状态: 模型权重常驻
    def act(self, obs):
        return self.model(obs)    # 方法用常驻状态
    def update(self, new_model):
        self.model = new_model    # 更新状态

a = PolicyServer.remote(model_init)   # 创建 actor (返回 handle)
ref = a.act.remote(obs)               # 异步调方法, 返回 ObjectRef
out = ray.get(ref)                    # 阻塞取结果
```

### 3.2 actor handle 与 ObjectRef

- **actor handle**（`a`）：对远程 actor 的引用，可传给其他 task/actor，任何持有者都能调其方法。这是 Ray 分布式对象通信的基础。
- **ObjectRef**（`ref`）：方法调用的 future，`ray.get(ref)` 阻塞取结果，`ray.wait([ref])` 非阻塞查是否就绪。

### 3.3 顺序执行（单线程默认）

actor 默认**单线程顺序执行**方法：邮箱队列里的方法调用按提交顺序逐一执行，无并发。这保证状态一致（无需锁），但若方法慢（如 LLM forward 100ms），并发请求堆积、吞吐受限。

> [!warning] actor 默认串行是吞吐瓶颈
> 若 PolicyServer actor 单方法 100ms，10 个并发请求要 1s 才处理完（串行）。要并发需 `max_concurrency=N`（允许多线程处理，但需保证方法线程安全）或多 actor 副本（[[placement group]] + actor pool）。

### 3.4 资源声明

`@ray.remote(num_gpus=1, num_cpus=4)` 声明 actor 占用资源。Ray 调度器据此分配 worker（如 num_gpus=1 的 actor 调度到有 GPU 的节点，独占该 GPU）。资源在 actor 生命周期内占用，销毁后释放。详见 [[resource request]]。

### 3.5 actor 生命周期与故障

- actor 创建后常驻，直到 handle 被垃圾回收或显式 `ray.kill(actor)`。
- actor 所在 worker 崩溃 → actor 失效，持有其 handle 的调用会报错（**单点**）。容错需 actor 重建（Ray 支持 `max_restarts` 自动重启，但内存状态丢失需重建）。

### 3.6 async actor 与 max_concurrency

- `@ray.remote(max_concurrency=N)`：允许 N 个方法并发执行（需方法线程安全）。适合 IO 密集（如异步推理）。
- `async def` 方法：actor 内事件循环调度，进一步提升 IO 并发。


## 4. 数学原理 / 公式

### 4.1 actor 串行吞吐

设单方法耗时 $T$，actor 单线程顺序处理。并发 $N$ 个请求的总耗时：

$$
T_{\text{total}}(N) = N \cdot T
$$

吞吐 $\text{TPS} = 1/T$（与 $N$ 无关，串行限死）。若要并发提吞吐，需 $K$ 个 actor 副本（[[placement group]]）：

$$
\text{TPS}_K = \frac{K}{T}
$$

> [!note] 推导
> 串行 actor 是单服务台排队（M/D/1），吞吐上限 $1/T$。$K$ 个副本是 $K$ 服务台，吞吐 $K/T$。这是 actor pool（多副本）提吞吐的本质——但受总 GPU 资源约束（$K \le$ 可用 GPU 数）。

### 4.2 与 task model 的开销对比

| 项 | task model（无状态） | actor model（有状态） |
|---|---|---|
| 状态传输 | 每次序列化（GB 级权重贵） | 不传输（常驻） |
| 启动 | 每 task 起进程 | 一次创建常驻 |
| 并发 | 天然并行（task 间独立） | 串行（需 max_concurrency/副本） |
| 适合 | 无状态纯计算 | 有状态组件 |

actor 的"常驻状态"省的是**反复序列化传输**的开销，代价是**串行执行**的吞吐限制。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 actor（看清"常驻+顺序+异步"）

```python
import threading, queue, time

class ActorHandle:
    """模拟 Ray actor: 常驻对象 + 邮箱队列顺序执行 + 异步 future."""
    def __init__(self, cls, *args, **kwargs):
        self._obj = cls(*args, **kwargs)        # 状态常驻
        self._mailbox = queue.Queue()           # 方法调用队列
        self._results, self._rid = {}, 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    def _run(self):
        """actor 主循环: 顺序处理邮箱中的调用."""
        while True:
            call = self._mailbox.get()
            if call is None: break
            rid, method, args = call
            try:
                self._results[rid] = ('ok', getattr(self._obj, method)(*args))
            except Exception as e:
                self._results[rid] = ('err', e)
    def remote(self, method, *args):
        """异步提交方法调用, 返回 ref (ObjectRef)."""
        self._rid += 1
        self._mailbox.put((self._rid, method, args))
        return self._rid
    def get(self, ref):
        """阻塞取结果 (模拟 ray.get)."""
        while ref not in self._results: time.sleep(1e-4)
        status, res = self._results.pop(ref)
        if status == 'ok': return res
        raise res

def remote_actor(cls):
    """装饰器: 把 class 变成 actor 工厂 (模拟 @ray.remote)."""
    return lambda *a, **k: ActorHandle(cls, *a, **k)

# —— 使用 ——
@remote_actor
class Counter:
    def __init__(self): self.n = 0
    def inc(self): self.n += 1; return self.n
    def get(self): return self.n

a = Counter()              # 创建 actor (状态 n=0 常驻)
r1 = a.remote('inc')       # 异步: n->1
r2 = a.remote('inc')       # 异步: n->2 (顺序执行, 不会并发写)
r3 = a.remote('get')       # 异步: 取 n
print(a.get(r1), a.get(r2), a.get(r3))   # 1 2 2
# 关键: r1/r2/r3 顺序执行, 状态跨调用保持
```

### 5.2 真实 Ray API 对照

```python
# import ray; ray.init(num_gpus=2)
# @ray.remote(num_gpus=1)
# class PolicyServer:
#     def __init__(self, w): self.w = w
#     def infer(self, x): return self.w @ x
# a = PolicyServer.remote(w)        # 创建 actor (占 1 GPU)
# ref = a.infer.remote(x)           # ObjectRef
# out = ray.get(ref)                # 阻塞取
# 上面的纯 Python 版把 a.method.remote(x) 拆成 a.remote('method', x) 便于模拟
```

### 5.3 actor pool（多副本提吞吐）

```python
# K 个 actor 副本, 轮询分发请求, 提吞吐
# pool = [PolicyServer.remote(w) for _ in range(K)]
# refs = [pool[i % K].infer.remote(x[i]) for i in range(N)]
# outs = ray.get(refs)   # K 路并行, 总耗时 ~ N*T/K
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[remote function]]（task 的无状态基础，actor 是其有状态扩展）、[[task model]]（对比对象）、[[resource request]]（actor 的资源声明）。
- **下游（应用）**: [[placement group]]（多 actor 副本的放置）、[[GPU绑定]]（actor 与 GPU 的绑定）、[[scheduling策略]]（actor 调度）、[[actor deadlock]]（actor 互相等待的死锁）、[[policy deployment (PD)]]（PD 侧 PolicyServer 常用 actor）、[[replay buffer]]（RLHF 的 buffer 常用 actor）。
- **对比 / 易混**:
  - **actor vs [[task model]]**：actor 有状态常驻、方法串行；task 无状态、天然并行。有状态组件用 actor，无状态计算用 task。
  - **actor vs 普通对象**：普通对象在本地进程内同步调用；actor 在远程 worker、异步调用、状态跨网络保持。
  - **actor vs 微服务**：actor 类似微服务（常驻、远程调用），但 Ray 的 actor 与 task/object store 紧密集成，编程模型更统一。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 actor 方法能并发
> 默认不能。actor 单线程顺序执行方法（保状态一致）。并发需 `max_concurrency`（需方法线程安全）或多副本。把 actor 当并发服务会遇吞吐瓶颈。

> [!warning] 误区 2：忽略 actor 是单点
> actor 驻留单 worker，该 worker 崩溃则 actor 失效、状态丢失（除非 max_restarts 重建，但内存状态不保）。关键 actor 需考虑容错（checkpoint 状态、多副本）。

> [!warning] 误区 3：把大对象当方法参数传
> actor 方法参数经 object store 传输。若传 GB 级张量，序列化+传输贵。应传 ObjectRef（引用，Ray 自动 locality）或让 actor 持有该对象。RLHF 中权重更新常用 `a.update.remote(model_ref)` 传引用而非值。

> [!warning] 误区 4：混淆 actor handle 与 ObjectRef
> actor handle（`a`）是对远程对象的引用，可调方法；ObjectRef（`ref`）是对一次方法调用结果的 future，要 `ray.get` 取。两者不同层级。

> [!warning] 误区 5：忽略资源声明
> `@ray.remote` 不带 `num_gpus` 则 actor 调度到 CPU worker，跑不了 GPU 计算。GPU actor 必须声明 `num_gpus=1`，Ray 才调度到 GPU 节点。详见 [[resource request]]。


## 8. 延伸细节

### 8.1 RLHF 中的 actor 应用

verl/OpenRLHF 的 RLHF 架构大量用 actor：

- **PolicyServer actor**：常驻模型权重，响应 [[rollout worker]] 的采样请求（[[policy deployment (PD)]]）。
- **Learner actor**：持有 optimizer state + 模型，做 PPO 梯度更新（[[policy training (PT)]]）。
- **ReplayBuffer actor**：累积轨迹数据，供 learner 采样。

这些有状态组件用 actor 才能高效（权重不反复传输、状态保持）。verl 用 Ray 的 [[placement group]] 把这些 actor 协同放置（colocate）以减通信。

### 8.2 max_concurrency 的使用

`@ray.remote(max_concurrency=10)` 让 actor 内 10 个方法并发（线程池）。适合**方法内 IO 等待多**（如异步推理等 GPU）的场景。但方法须线程安全（无共享可变状态，或加锁）。纯计算方法用 max_concurrency 无益（受 GIL/算力限）。

### 8.3 async actor

`async def` 方法 + max_concurrency 让 actor 内事件循环调度，适合高并发 IO（如 HTTP 服务）。比线程版更轻量。

### 8.4 actor 与 RLHF 训推分离

在 [[训推分离]] 架构里，PT 的 Learner actor 与 PD 的 PolicyServer actor 分离部署，[[weight sync mechanism]] 通过 `a.update.remote(new_w_ref)` 在 actor 间推权重。actor 的常驻特性让权重更新只需传新权重引用，不需重建 actor。

### 8.5 Ray Core vs Ray Serve

- Ray Core 的 actor：通用分布式有状态对象（上述）。
- Ray Serve：基于 actor 的模型服务框架，封装 HTTP 入口、autoscaling、路由，底层仍是 actor。LLM 服务（vLLM on Serve）常用。

---
相关: [[Ray基础]] | [[task model]] | [[remote function]] | [[resource request]] | [[placement group]] | [[policy deployment (PD)]] | [[replay buffer]]
