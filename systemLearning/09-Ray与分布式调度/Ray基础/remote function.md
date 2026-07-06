# remote function

> **所属章节**: [[Ray基础]]
> **所属模块**: [[09-Ray与分布式调度]]
> **别名**: remote function / @ray.remote / 远程函数 / remote 装饰器
> **难度**: 中（需懂 [[task model]] + [[actor model]]）


## 1. 一句话定义

**remote function** 是 Ray 中用 `@ray.remote` 装饰器把**普通 Python 函数或类**转换为**远程可调用实体**的机制：装饰后，函数成为 [[task model]]（`f.remote(x)` 异步提交、返回 ObjectRef），类成为 [[actor model]]（`C.remote()` 创建常驻 actor）。它隐藏了序列化（cloudpickle）、跨进程/跨节点调度、object store 传输的细节，是 Ray 分布式编程的统一入口——所有 task 与 actor 都通过 `@ray.remote` 创建。

> [!note] 三句话定位
> - **是什么**：`@ray.remote` 装饰器，把函数→task、类→actor，调用变远程异步。
> - **为什么**：手动做序列化+调度+传输太繁；装饰器把"普通 Python"无缝升级为"分布式可调用"。
> - **与 task/actor 的关系**：remote function 是两者的**共同底层**，task 和 actor 是它的两种用法（装饰函数 vs 装饰类）。


## 2. 为什么需要它（动机与背景）

### 2.1 分布式调用的繁琐

若手动把一个函数远程化，需：

1. **序列化**函数代码 + 参数（cloudpickle）。
2. **调度**到某 worker（按资源需求选）。
3. worker **反序列化**、执行。
4. **序列化**结果、存 object store。
5. 调用方 `get` 时**反序列化**结果。

每步都易错（序列化失败、资源不匹配、超时）。`@ray.remote` 把这些封装成一行装饰 + `.remote()` 调用，用户写普通 Python，Ray 处理分布式细节。

### 2.2 统一 task 与 actor 的入口

Ray 的两个核心抽象（[[task model]] 无状态函数、[[actor model]] 有状态类）都用 `@ray.remote` 创建：

```python
@ray.remote               # 装饰函数 → task
def f(x): return x * 2

@ray.remote               # 装饰类 → actor
class C:
    def __init__(self): self.n = 0
    def inc(self): self.n += 1; return self.n
```

一个装饰器、两种产物，API 一致（都 `.remote(...)`），降低学习成本。

### 2.3 为什么不直接用 Python 函数

普通 Python 函数本地同步调用，无法跨进程/节点、无法并行、无法声明 GPU 资源。`@ray.remote` 是"本地→分布式"的升级开关，代码改动最小（加装饰器、调用加 `.remote`）。


## 3. 核心概念详解

### 3.1 装饰器语法与资源声明

```python
@ray.remote(num_gpus=1, num_cpus=4, max_retries=3, num_returns=2)
def f(x): return x, x*x      # num_returns=2 → 返回 2 个 ObjectRef
```

- `num_gpus`/`num_cpus`：资源声明（调度到满足的 worker）。
- `max_retries`：失败重试次数（task 默认 3，actor 默认 0）。
- `num_returns`：多返回值时返回多个 ObjectRef（减少序列化合并）。

### 3.2 调用语义：.remote() 提交，ray.get 取结果

`f.remote(x)` **不执行**函数，只**提交**到调度器、立即返回 ObjectRef。真正执行在远程 worker。`ray.get(ref)` 阻塞取结果。

```python
ref = f.remote(5)     # 提交, 立即返回 (函数此刻未必执行)
result = ray.get(ref) # 阻塞等执行完, 取结果
```

> [!warning] remote() 不等于执行
> 新手常误以为 `f.remote(x)` 就执行了。它只提交、返回 future。若不 `ray.get`，函数可能被调度延迟甚至（某些配置下）不执行。结果链必须以 `get` 收尾。

### 3.3 序列化（cloudpickle）

`@ray.remote` 的函数与参数经 **cloudpickle** 序列化传到 worker。cloudpickle 能序列化闭包、lambda、局部定义函数（比标准 pickle 强）。但仍有限制：

- 闭包引用的全局变量在 worker 侧需可导入。
- C 扩展/带硬件句柄的对象（如 CUDA tensor 的句柄）需特殊处理（Ray 对 tensor 有零拷贝优化）。
- 不可序列化的对象（如文件句柄、锁）会报错。

### 3.4 动态 remote（ray.remote(fn)）

除装饰器语法，也可动态包装：

```python
f = ray.remote(lambda x: x*2)   # 动态, 不用 @
ref = f.remote(5)
```

或 `ray.remote(普通函数)` 在运行时包装。适合从配置/反射动态生成远程函数。

### 3.5 num_returns 与多 ObjectRef

```python
@ray.remote(num_returns=2)
def split(x): return x, x*2
r1, r2 = split.remote(5)   # 返回 2 个 ObjectRef
print(ray.get(r1), ray.get(r2))   # 5 10
```

多返回值时各自独立 ObjectRef，消费方可只 get 需要的那个，避免全量序列化传输。


## 4. 数学原理 / 公式

### 4.1 远程调用开销

一次 `f.remote(x)` + `ray.get(ref)` 的总开销：

$$
T_{\text{remote}} = T_{\text{serialize}} + T_{\text{schedule}} + T_{\text{exec}} + T_{\text{deserialize}}
$$

- $T_{\text{serialize}}$：cloudpickle 序列化 fn + 参数（与参数大小成正比）。
- $T_{\text{schedule}}$：调度器选 worker、派发（毫秒级，受集群规模影响）。
- $T_{\text{exec}}$：函数实际执行（计算量）。
- $T_{\text{deserialize}}$：worker 侧反序列化、结果序列化、调用方反序列化。

> [!note] 开销与任务粒度
> 若 $T_{\text{exec}} \ll T_{\text{serialize}} + T_{\text{schedule}}$（任务太小），远程化反而慢——开销淹没计算。Ray 的经验法则：单 task 至少 ~1ms 计算，否则本地跑更划算。批量小任务（如用一个 task 处理一批数据）是优化。

### 4.2 batch 调用摊薄开销

```python
# 差: 1000 个小 task, 每 task 开销淹没
refs = [f.remote(x) for x in range(1000)]   # 1000 次序列化+调度
# 好: 1 个 batch task 处理 1000 个
@ray.remote
def batch_f(xs): return [x*2 for x in xs]
ref = batch_f.remote(list(range(1000)))     # 1 次开销
```

开销从 $1000 \cdot T_{\text{overhead}}$ 降到 $1 \cdot T_{\text{overhead}}$。这是 Ray 性能调优的关键——避免过细粒度 task。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 @ray.remote 装饰器

```python
import concurrent.futures, functools

def remote(num_gpus=0, num_cpus=1, num_returns=1):
    """模拟 @ray.remote: 把函数变远程可调用 (返回 future)."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    def decorator(fn):
        @functools.wraps(fn)
        def submitter(*args, **kwargs):
            # .remote() 即提交, 返回 future (ObjectRef)
            return pool.submit(fn, *args, **kwargs)
        submitter.remote = submitter   # f.remote(x) == submitter(x)
        return submitter
    return decorator

def get(future):                       # 模拟 ray.get
    return future.result()

# —— 装饰函数 → task ——
@remote(num_gpus=1)
def add(a, b): return a + b

ref = add.remote(3, 4)                  # 提交, 返回 future
print('task result:', get(ref))         # 7

# —— 动态 remote ——
mul = remote()(lambda a, b: a * b)
print('dynamic:', get(mul.remote(3, 4)))   # 12

# —— num_returns=2 (真实 Ray 返 2 个 ObjectRef; 模拟简化为单 future 取 tuple) ——
@remote(num_returns=2)
def split(x): return x, x * x
ref = split.remote(5)                   # 单 future (简化)
print('multi-return:', get(ref))        # (5, 25)
```

### 5.2 装饰类 → actor（对照）

```python
def remote_actor(num_cpus=1):
    def decorator(cls):
        # 简化: 返回工厂, 实例化时启动后台线程 (见 actor model 篇)
        def factory(*args, **kwargs):
            obj = cls(*args, **kwargs)
            return obj   # 真实 Ray 会包成 ActorHandle
        return factory
    return decorator

@remote_actor(num_cpus=1)
class Counter:
    def __init__(self): self.n = 0
    def inc(self): self.n += 1; return self.n
a = Counter()   # 创建 "actor"
print('actor inc:', a.inc(), a.inc())   # 1 2 (本地简化, 真实为远程)
```

### 5.3 真实 Ray API 对照

```python
# import ray; ray.init(num_gpus=2)
# @ray.remote(num_gpus=1, num_returns=2)
# def split(x): return x, x*x
# r1, r2 = split.remote(5)
# print(ray.get(r1), ray.get(r2))   # 5 25
```


## 6. 与其他知识点的关系

- **上游（依赖）**: cloudpickle（序列化）、object store（结果存储与传输）、Ray 调度器。
- **下游（应用）**: [[task model]]（装饰函数的产物）、[[actor model]]（装饰类的产物）、[[resource request]]（资源声明参数）。
- **对比 / 易混**:
  - **remote function vs 普通函数**：普通函数本地同步；remote function 远程异步、返回 ObjectRef、可分布式并行。代码几乎相同（加装饰器 + `.remote`），语义大不同。
  - **`@ray.remote` 装饰函数 vs 装饰类**：前者 → task（无状态），后者 → actor（有状态）。同一装饰器、两种产物。
  - **`.remote()` vs `ray.get()`**：`.remote()` 提交（返回 future，不执行）；`ray.get()` 取结果（阻塞）。新手常混。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 `f.remote(x)` 就执行了
> 只提交、返回 ObjectRef，函数在远程 worker 异步执行。要结果必须 `ray.get(ref)`。提交后不 get 直接用 ref 当结果会报错。

> [!warning] 误区 2：混淆装饰函数与装饰类
> `@ray.remote` 装饰**函数** → task（每次 `.remote()` 起一个无状态 task）；装饰**类** → actor（`.remote()` 创建一个常驻有状态对象）。两者调用 `.remote()` 的语义不同（task 提交执行、actor 创建实例）。

> [!warning] 误区 3：任务粒度太细
> 单 task 计算量 <1ms 时，序列化+调度开销淹没计算，远程化反而慢。应用批量（一个 task 处理一批数据）摊薄开销。

> [!warning] 误区 4：忽略序列化限制
> cloudpickle 虽强，但闭包引用的全局变量需 worker 可导入、C 扩展/硬件句柄需特殊处理、不可序列化对象（文件句柄/锁）会报错。遇到 "can't pickle" 报错先查序列化。

> [!warning] 误区 5：忘记资源声明
> `@ray.remote` 不带 `num_gpus` 的 task 调度到 CPU worker，跑不了 GPU 计算。GPU task 必须声明 `num_gpus=1`。详见 [[resource request]]。


## 8. 延伸细节

### 8.1 cloudpickle 的能力与限制

cloudpickle（Ray 默认序列化）能序列化 lambda、闭包、动态定义的函数/类（标准 pickle 不能）。但：

- 闭包捕获的全局变量：worker 侧需能 import 同名模块（或 Ray 传引用）。
- GPU tensor：Ray 有零拷贝优化（经共享内存/object store 直接传 buffer，不复制）。
- 不可序列化：打开的文件句柄、线程锁、网络连接——需在 worker 侧重建。

### 8.2 num_returns 的优化

多返回值时 `num_returns=k` 返回 $k$ 个独立 ObjectRef，消费方可只 get 需要的，避免全量传输。如一个 task 同时产出 logits + hidden state，下游只消费 logits 的 worker 不必取 hidden state。

### 8.3 ray.remote 的动态用法

- `ray.remote(fn)` 动态包装普通函数为 task。
- `ray.remote(cls)` 动态包装类为 actor。
- 适合从配置生成（如根据 YAML 定义动态创建远程函数）。

### 8.4 与 RLHF 的关系

verl/OpenRLHF 用 `@ray.remote` 定义 [[rollout worker]]（采样 task 或 PolicyServer actor）、[[policy training (PT)]] 的 Learner actor、[[replay buffer]] actor。remote function 是这些组件的统一创建入口。`num_gpus` 声明让 Ray 把 GPU task/actor 调度到 GPU 节点（[[placement group]] 精细化放置）。

### 8.5 Ray 的 object store 与 locality

`.remote()` 的参数与返回值经 object store。Ray 调度器优先把消费方 task 调度到生产方同 worker（locality 命中，零拷贝）。大对象（权重、大 batch）受益明显。[[placement group]] 的 colocate 策略强化这点。

---
相关: [[Ray基础]] | [[task model]] | [[actor model]] | [[resource request]] | [[placement group]] | [[policy training (PT)]] | [[rollout worker]]
