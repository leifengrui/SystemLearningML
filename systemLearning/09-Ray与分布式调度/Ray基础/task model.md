# task model

> **所属章节**: [[Ray基础]]
> **所属模块**: [[09-Ray与分布式调度]]
> **别名**: Ray task / task / 无状态远程函数 / remote task
> **难度**: 中（需懂并行 + [[remote function]]）


## 1. 一句话定义

**Ray task（task model）** 是 Ray 中**无状态的远程函数调用**：用 `@ray.remote` 装饰一个普通函数，每次 `f.remote(x)` 在某个 worker 上起一个**无状态 task** 执行、结果存 [[object store]]（返回 ObjectRef），task 间**天然并行**、执行完即销毁、不保留状态。它是 Ray 分布式并行的基本单元，与有状态的 [[actor model]] 互补——纯计算（数据并行前向、超参搜索、批处理）用 task，有状态组件（模型服务、buffer）用 actor。

> [!note] 三句话定位
> - **是什么**：无状态远程函数，调一次起一个 task，算完即走，天然并行。
> - **为什么**：分布式并行计算需把函数分发多 worker；task 无状态 → 可任意并行、可重试、无副作用。
> - **典型用途**：数据并行的批量前向、超参搜索、map-reduce、[[rollout worker]] 的并行采样。


## 2. 为什么需要它（动机与背景）

### 2.1 并行计算的需求

分布式计算的核心需求：把一个计算分成 $N$ 份，分到 $N$ 个 worker 并行跑。如：

- 超参搜索：100 组超参，每组训练一个模型，并行跑。
- 数据并行前向：8 个 batch 分到 8 GPU，并行算前向。
- [[rollout worker]]：8 个环境并行采样轨迹。

这些都需"把一个函数分发到多 worker 并行执行"。task model 提供这个原语。

### 2.2 为什么不用 actor

[[actor model]] 有状态、方法串行，适合常驻组件，但**并行纯计算**时：

- actor 串行执行 → 无法并行（单 actor 一次一个 task）。
- actor 有状态 → 并行多副本需协调状态，多余。

task model 的"无状态、算完即走"让并行天然：$N$ 个 `f.remote(x_i)` 立即分到 $N$ worker 并行，无需状态协调。

> [!tip] task vs actor 的分工
> - **task**：无状态纯计算，天然并行，算完即走。适合"分而治之"的并行计算。
> - **actor**：有状态常驻，方法串行。适合"保持状态"的服务/组件。
> 一个 Ray 程序常混用：actor 做有状态服务，task 做并行计算。


## 3. 核心概念详解

### 3.1 定义与使用

```python
import ray
ray.init()

@ray.remote(num_gpus=1)           # 装饰为 task, 声明用 1 GPU
def forward(batch, weights):
    return model(batch, weights)  # 纯函数, 无状态

refs = [forward.remote(b, w) for b in batches]  # 8 个 task 并行
outs = ray.get(refs)              # 阻塞等全部完成
```

### 3.2 无状态与可重试

task 无状态（不保留跨调用状态）→ Ray 可**自动重试**失败的 task（worker 崩溃时重新调度到其他 worker）。这是 task 相对 actor 的容错优势——actor 崩溃状态丢失，task 崩溃可重跑（因输入决定输出）。

> [!warning] 可重试要求无副作用
> task 重试假设"相同输入得相同输出"。若 task 有副作用（写文件、改全局变量、发网络请求），重试会重复副作用。需用 `max_retries=0` 禁用或保证幂等。

### 3.3 ObjectRef 与 ray.get/wait

- `f.remote(x)` 返回 **ObjectRef**（future），立即返回不阻塞。
- `ray.get(ref)` 阻塞取结果。
- `ray.wait([ref1, ref2], num_returns=1)` 非阻塞查哪些完成，支持"先到先处理"的流式。

### 3.4 嵌套 task（nested task）

一个 task 内部可以再提交 task：

```python
@ray.remote
def map_fn(x): return x*x
@ray.remote
def driver():
    refs = [map_fn.remote(i) for i in range(8)]  # 嵌套提交 task
    return ray.get(refs)
ray.get(driver.remote())  # [0,1,4,9,16,25,36,49]
```

driver task 在某 worker 执行时，内部提交的 map_fn task 被 Ray 调度到其他 worker 并行。这是 Ray 的"分布式 spawn"——任意 task 都能提交子 task，形成计算图。

### 3.5 资源声明与调度

`@ray.remote(num_gpus=1, num_cpus=4)` 声明 task 资源需求。Ray 调度器把 task 分配到满足资源的 worker（如 num_gpus=1 的 task 到 GPU 节点）。task 执行完资源立即释放（不同于 actor 的长期占用）。详见 [[resource request]]。


## 4. 数学原理 / 公式

### 4.1 并行加速比

$N$ 个独立 task，单 task 耗时 $T$，有 $W$ 个 worker 并行：

$$
T_{\text{total}}(N, W) = \left\lceil \frac{N}{W} \right\rceil \cdot T
$$

加速比（相对串行 $NT$）：

$$
S = \frac{N \cdot T}{\lceil N/W \rceil \cdot T} \approx W \quad (N \gg W)
$$

> [!note] 推导
> $N$ 个 task 分到 $W$ worker，每 worker 处理 $\lceil N/W \rceil$ 个，串行耗时 $\lceil N/W \rceil T$。$N \gg W$ 时趋近 $W \cdot T$，加速比 $\to W$（Amdahl 上限受 worker 数）。这是 task 并行的理想加速——前提是 task 间无依赖。

### 4.2 通信开销与 object store

task 结果存 object store，后续 task 取时若 locality 命中（同 worker）零拷贝，否则经 HBM/网络。大对象（GB 级）传输贵。优化：让消费方 task 与生产方 task colocate（[[placement group]]），或传小引用。

### 4.3 task vs actor 吞吐

| 维度 | task | actor |
|---|---|---|
| 并行度 | $N$ task 天然并行 | 单 actor 串行（需副本并行） |
| 吞吐上限 | $W/T$（$W$ worker） | $K/T$（$K$ 副本，受 GPU 限） |
| 状态 | 无 | 有 |
| 容错 | 自动重试 | 状态丢失需重建 |

对纯计算（无状态），task 的 $W/T$ 优于单 actor 的 $1/T$；但有状态组件只能用 actor。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 task（无状态+并行+future）

```python
import concurrent.futures

class TaskSystem:
    """模拟 Ray task: 无状态函数 + 线程池并行 + future."""
    def __init__(self, max_workers=4):
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers)
    def remote(self, fn, *args):
        """提交无状态 task, 返回 future (ObjectRef)."""
        return self.pool.submit(fn, *args)        # fn 必须无状态
    def get(self, future):
        """阻塞取结果 (模拟 ray.get)."""
        return future.result()
    def wait(self, futures, num=1):
        """非阻塞查完成 (模拟 ray.wait)."""
        done, not_done = concurrent.futures.wait(futures, timeout=0,
                                                  return_when='FIRST_COMPLETED')
        return list(done), list(not_done)

# 无状态纯函数
def square(x): return x * x     # 无副作用, 可安全重试

ts = TaskSystem(max_workers=4)
refs = [ts.remote(square, i) for i in range(8)]   # 8 task 并行
print('results:', [ts.get(r) for r in refs])      # [0,1,4,9,16,25,36,49]
done, not_done = ts.wait([ts.remote(square, 5)], num=1)
print('wait demo done count:', len(done))
```

### 5.2 嵌套 task（map-reduce 风格）

```python
def map_fn(x): return x * x
def driver(n):
    # 模拟 task 内提交子 task
    refs = [ts.remote(map_fn, i) for i in range(n)]   # 嵌套
    return sum(ts.get(r) for r in refs)               # reduce
print('map-reduce sum:', ts.get(ts.remote(driver, 5)))  # 0+1+4+9+16=30
```

### 5.3 真实 Ray API 对照

```python
# import ray; ray.init(num_gpus=4)
# @ray.remote(num_gpus=1)
# def forward(batch, w): return model(batch, w)
# refs = [forward.remote(b, w) for b in batches]   # 8 task 分到 4 GPU 并行
# outs = ray.get(refs)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[remote function]]（task 的实现载体，`@ray.remote` 装饰的函数）、[[resource request]]（task 的资源声明）。
- **下游（应用）**: [[placement group]]（task 的放置）、[[scheduling策略]]（task 调度）、[[rollout worker]]（并行采样常用 task）、[[Data Parallel]]（数据并行前向用 task）。
- **对比 / 易混**:
  - **task vs [[actor model]]**：**最核心对比**。task 无状态、天然并行、算完即走；actor 有状态、方法串行、常驻。无状态计算用 task，有状态组件用 actor。
  - **task vs 普通函数调用**：普通函数本地同步；task 远程异步、返回 future、可分布式并行。
  - **task vs Python multiprocessing**：multiprocessing 是单机进程池；Ray task 是分布式多机，带 object store、资源调度、容错重试。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 task 能保持状态
> 不能。task 无状态，每次调用是新执行，不保留跨调用状态。要状态必须用 [[actor model]]。把状态塞 task 的全局变量也无效（不同 worker 的全局变量独立）。

> [!warning] 误区 2：忘记 ray.get 阻塞
> `f.remote(x)` 立即返回 ObjectRef，不阻塞。要结果必须 `ray.get(ref)`，它会阻塞到 task 完成。提交后不 get 而直接用 ref 会报错（ref 不是结果值）。

> [!warning] 误区 3：task 有副作用时启用重试
> Ray 默认 `max_retries=3` 自动重试失败 task。若 task 有副作用（写文件、发请求），重试会重复副作用。需 `@ray.remote(max_retries=0)` 或保证幂等。

> [!warning] 误区 4：混淆 task 并行与 actor 并行
> $N$ 个 `f.remote()` 是 $N$ 个 task 天然并行（分到多 worker）；单 actor 的 $N$ 次方法调用是串行。要 actor 并行需 $K$ 副本（[[placement group]]），不是单 actor 多调用。

> [!warning] 误区 5：忽略 object store 传输开销
> task 结果存 object store，消费方取时可能跨 worker 传输。大对象（GB 级张量）传输贵，应让生产-消费 task colocate 或传小引用。


## 8. 延伸细节

### 8.1 RLHF 中的 task 应用

- [[rollout worker]] 的并行采样：$N$ 个环境/请求作为 $N$ 个 task 并行生成轨迹（无状态，每 task 独立采样）。
- 数据并行前向：batch 分 $N$ 份，$N$ 个 task 各算一份（[[Data Parallel]]）。
- 超参搜索：$N$ 组超参各一个 task 训练评估。

注意：若采样需常驻模型权重，用 actor（PolicyServer）更高效（权重不反复加载）；若每 task 独立加载权重可接受，用 task。verl 的采样常用 actor（PolicyServer）+ task（请求并行）混合。

### 8.2 map-reduce 模式

Ray task 天然支持 map-reduce：$N$ 个 map task 并行产出，$M$ 个 reduce task 聚合（嵌套 task 提交）。适合大规模数据并行处理。

### 8.3 task 与 object store 的流式

`ray.wait` 支持流式：提交 $N$ task，循环 `wait` 取先完成的处理，再提交新 task，形成流水线。适合长流式计算（如持续采样）。

### 8.4 与 RLHF 训推分离的关系

[[训推分离]] 中，PD 侧的 PolicyServer 是 actor（有状态常驻），但单个采样请求可作为 task 提交给 actor（`actor.infer.remote(obs)` 返回 ObjectRef）。actor 保证状态常驻，task 保证请求并行。两者协同：actor 是服务端，task 是客户端请求。

### 8.5 Ray Core 的 task vs actor 选择

经验法则：

- 函数无状态、可并行、算完即走 → **task**。
- 对象有状态、需常驻、方法串行 → **actor**。
- 不确定时优先 task（更简单、容错好），仅当必须保持状态才升级到 actor。

---
相关: [[Ray基础]] | [[actor model]] | [[remote function]] | [[resource request]] | [[placement group]] | [[rollout worker]] | [[Data Parallel]]
