# asyncio

> **所属章节**: [[并发与生命周期]]
> **所属模块**: [[20-可靠性与开源工程]]
> **别名**: asyncio / event loop / 事件循环 / coroutine / 协程 / async await / gather / async IO
> **难度**: 中（需懂 [[Python multiprocessing与GIL]]、IO 并发模型、推理 server 的请求处理）


## 1. 一句话定义

**asyncio** 是 Python 的**异步 IO 框架**——基于 **event loop（事件循环）** 调度 **coroutine（协程，`async`/`await` 标记的函数）**，在**单线程**内通过 **await 让出控制权**实现 **IO 并发**（一个进程同时处理成百上千个 IO 请求，等 IO 时不阻塞、切去处理别的请求）；**`asyncio.gather`** 并发执行多个协程并等全部完成；与多线程对比——协程单线程无 GIL 竞争、切换开销极小（用户态、无内核态切换），但 **CPU 密集任务仍受 GIL**（单线程无法多核并行）；在 ML 工程里它体现在——① **推理 API server 的高并发 IO**（一个进程处理多请求，`await` 等 GPU 推理/IO 时不阻塞其他请求，是 [[OpenAI API server]] 的核心并发模型），② **RL 框架的 async actor**（[[Ray与分布式调度]] 的 asyncio actor，一个 actor 内并发处理多 rollout/请求），③ **IO + 计算混合时与 multiprocessing 配合**（IO 用 asyncio、计算用进程）。最大陷阱：**阻塞调用会卡死 event loop**——在协程里调 `time.sleep` / 同步 `requests.get` / `torch.cuda.synchronize` 会把整个单线程 loop 冻结，所有请求都等它，必须用 `await asyncio.sleep` / `aiohttp` / 放线程池。注意 asyncio **不创造并行**（单线程，无多核加速），只创造 **IO 并发**（等待重叠）。

> [!note] 三句话定位
> - **是什么**：Python 单线程异步 IO 框架，event loop 调度 coroutine，`await` 让出控制权让别的协程跑，IO 不阻塞。
> - **为什么**：高并发 IO 场景（推理 server、多 rollout 采样）一个进程要同时处理上千请求，多线程有 GIL + 切换开销、多进程太重，协程最轻。
> - **与 multiprocessing 分工**：IO 用 asyncio（等的时候切走）、CPU 计算用 multiprocessing（绕 GIL 真并行）。RL 框架常两者叠：async actor 管 IO 调度、子进程/worker 管计算。


## 2. 为什么需要它（动机与背景）

### 2.1 高并发 IO 的痛点

推理 API server 收到请求 → 前处理 → GPU forward → 后处理 → 返回。GPU forward 期间（数十~数百 ms）**主线程若阻塞等**，其他请求全排队。并发上千请求时，要么起上千线程（内存爆 + GIL 切换开销 + 内核态调度重），要么起多进程（更重 + IPC）。需要**单线程内并发等 IO**——一个请求等 GPU 时切去处理别的请求。

### 2.2 多线程的局限

- **GIL 竞争**：上千线程抢一把 GIL，Python glue 串行，且 IO 放锁时频繁切换。
- **线程开销**：每线程栈 1-8MB，上千线程吃几 GB 内存；内核态调度，上下文切换开销。
- **同步原语复杂**：锁、条件变量易写错（死锁、race）。

### 2.3 协程的轻量

- **单线程**：无 GIL 竞争（只有一个线程），无锁。
- **用户态切换**：协程切换在用户空间（event loop 调度），无内核态 syscall，开销 ~纳秒级（vs 线程 ~微秒）。
- **栈极小**：协程无独立栈（generator-based），上千协程内存开销小。
- **await 显式让出**：协程在 `await` 处**主动**让出控制权（cooperative），不是抢占式——写代码时明确知道切换点，比线程抢占式更可控。

### 2.4 asyncio 在 ML 里的位置

| 场景 | 用 asyncio | 原因 |
|---|---|---|
| 推理 API server | ✅ 主流 | 高并发请求，forward 时 await 放行其他请求 |
| RL async actor | ✅ verl/OpenRLHF 用 Ray asyncio actor | 一个 actor 并发调度多 rollout/sample 请求 |
| 数据 pipeline（远端 fetch） | ✅ 部分 | 大量并发 IO 拉数据 |
| CPU 密集训练 | ❌ 不用 | GIL 限制，单线程无法多核——用 multiprocessing/DDP |
| GPU forward（C 扩展） | ⚠️ 部分 | kernel 期间放 GIL，但 await 才让出——需把 forward 放线程池或正确 await |


## 3. 核心概念详解

### 3.1 event loop（事件循环）

event loop 是 asyncio 的**调度核心**——一个单线程循环，不断：① 检查**就绪的 coroutine**（await 的 IO 完成了、timer 到了），② 取一个跑，跑到下一个 `await` 让出，③ 回到①。所有协程、IO callback、timer 都在这个**单线程**里串行执行。

```
event loop:
  while running:
    ready = select_ready_tasks()   # IO 完成的 / timer 到的
    for task in ready:
      run task until await          # 跑到下一个 await 让出
    poll IO (epoll/kqueue)          # 等 IO, 不占 CPU
```

> [!note] 单线程的"并发"
> asyncio 的"并发"不是"同时跑"，是**等待重叠**——A 协程 await 等 GPU（不占 CPU），loop 切到 B 协程跑（B 也可能 await 等），切到 C……当 A 的 GPU 完成通知 loop，loop 再调度 A 继续。CPU 同一时刻只跑一个协程，但 IO 等待期间不浪费。

### 3.2 coroutine（协程，async/await）

```python
async def fetch(url):
    data = await aiohttp.get(url)   # await 让出控制权, loop 跑别的
    return data
```

- `async def` 定义协程函数，调用它返回一个 **coroutine 对象**（不立即执行）。
- `await` **挂起当前协程**，把控制权交回 loop，等 await 的对象（another coroutine / future / awaitable）完成后恢复。
- `await` 是**唯一让出点**——没有 `await` 的协程会一口气跑完，不让出（卡 loop）。

### 3.3 gather（并发多协程）

```python
results = await asyncio.gather(coro1(), coro2(), coro3())
# 三个协程并发跑, gather 等全部完成, 返回 [r1, r2, r3]
# 一个 raise 异常 -> gather 默认取消其他并抛出 (return_exceptions=False)
```

`gather` 把多个协程**并发**调度（不是串行），等全部完成返回结果列表。是 asyncio 里"并发 N 个任务"的主力 API。还有 `asyncio.wait`（返回 done/pending 两组）、`asyncio.as_completed`（先完成的先 yield）、`TaskGroup`（3.11+，结构化并发，更安全）。

### 3.4 await 的本质：让出控制权

`await x` 等价于：① 把 `x`（一个 awaitable）注册到 loop，② 当前协程**挂起**，③ 控制权回 loop，loop 调度别的就绪协程，④ 当 `x` 完成（IO 返回/future resolve），loop 把当前协程重新放回就绪队列，⑤ 恢复执行。关键：**await 期间 CPU 不空转**（loop 在跑别的协程）。

### 3.5 阻塞调用卡 event loop（最大陷阱）

```python
async def bad():
    time.sleep(2)        # 阻塞! 整个 loop 冻结 2s, 其他请求全等
    # 必须用:
    await asyncio.sleep(2)  # 让出, loop 跑别的

async def bad2():
    resp = requests.get(url)   # 阻塞! 同步 IO
    # 必须用:
    resp = await aiohttp.get(url)  # 异步 IO

async def bad3():
    torch.cuda.synchronize()   # 阻塞等 GPU! loop 冻结
    # 必须放线程池:
    await asyncio.to_thread(torch.cuda.synchronize)
```

> [!warning] 误区：asyncio 自动让阻塞变异步
> 错。`async def` 里的**阻塞调用**（同步 IO、`time.sleep`、CPU 密集 for 循环、`torch.cuda.synchronize`）会**冻结整个 loop**。必须用异步 IO 库（aiohttp/aiofiles）、`await asyncio.sleep`、`await asyncio.to_thread(fn)` 把阻塞调用丢到线程池。这是 asyncio 最易踩的坑。

### 3.6 asyncio vs threading vs multiprocessing

| 维度 | asyncio | threading | multiprocessing |
|---|---|---|---|
| 并行度 | 单线程，无 CPU 并行 | GIL 限 CPU 并行 | 真并行（独立 GIL） |
| IO 并发 | ✅ 最佳 | 有效但重 | 有效但重 |
| 切换开销 | 极小（用户态） | 中（内核态） | 重 |
| 内存 | 极轻（上千协程） | 中（每线程栈） | 重（进程空间） |
| 让出方式 | 协作式（await 显式） | 抢占式（调度器） | 抢占式 |
| 阻塞调用 | ⚠️ 卡死 loop | 不卡其他线程 | 不卡其他进程 |
| ML 用途 | 推理 server / async actor | 少（被 GIL） | DDP / DataLoader |

### 3.7 推理 API server 的 asyncio 模型

```
client A ──┐
client B ──┼──> asyncio server (单进程, event loop)
client C ──┘
  收到请求 A -> await 前处理 (放线程池做 CPU 活) -> await GPU forward (submit kernel, 放锁)
  loop 切到 B -> 前处理 -> forward
  loop 切到 C ...
  A 的 GPU 完成 -> loop 恢复 A -> 后处理 -> 返回
  一个进程同时处理 A/B/C, GPU 流水, 前后处理重叠
```

[[OpenAI API server]] / vLLM 的 async endpoint 用 asyncio——一个 worker 进程 event loop 处理多请求，forward 时 await（kernel 异步 + C 扩展放 GIL，loop 切到别的请求的前处理）。是**单 worker 高并发**的关键。

### 3.8 RL 框架的 async actor

[[Ray与分布式调度]] 的 actor 可标 `asyncio`：

```python
@ray.remote
class AsyncActor:
    async def handle(self, req):
        data = await fetch(req.url)      # IO 等待时, loop 调度别的 handle 调用
        result = await self._compute(data)  # 若 _compute 是 CPU 密集, 应放线程池或子 actor
        return result
# 一个 AsyncActor 实例可并发处理多个 handle 调用 (asyncio 并发)
```

verl/OpenRLHF 的 rollout worker 用 async actor——一个 actor 并发调度多个 sample 请求（等 vLLM forward 时切别的请求），用单 actor 跑满 IO 并发，配合多 actor（多进程）拿计算并行。是 **asyncio（IO）+ multiprocessing（计算）** 的典型叠加。详见 [[actor model]]、[[Ray与分布式调度]]。


## 4. 数学原理 / 公式

### 4.1 IO 并发的吞吐模型

设每请求：CPU 处理时间 $t_{cpu}$，IO/forward 等待时间 $t_{io}$，单请求总耗时 $T=t_{cpu}+t_{io}$。

- **串行（单线程阻塞）**：$n$ 请求总耗时 $nT$，吞吐 $\text{TPS}=1/T$。
- **asyncio 并发**：CPU 处理串行（单线程），但 IO 等待重叠。总耗时 $\approx n\cdot t_{cpu} + t_{io}$（最后一批的 IO 等）。吞吐 $\approx \dfrac{1}{t_{cpu} + t_{io}/n}$，$n$ 大时 $\to 1/t_{cpu}$。
  - 若 $t_{io}\gg t_{cpu}$（IO 密集），asyncio 吞吐 $\to 1/t_{cpu}$，远高于串行 $1/T$。
  - 若 $t_{cpu}\gg t_{io}$（CPU 密集），asyncio 无优势（CPU 串行），退化到 $1/t_{cpu}$，与单线程一致。

### 4.2 协程切换开销

- **协程切换**：用户态，保存/恢复寄存器 + PC，约 $O(100ns)$。
- **线程切换**：内核态 syscall（schedule），上下文切换 + TLB 刷新，约 $O(1\mu s)$~$O(10\mu s)$。
- 上千协程 vs 上千线程：协程切换总开销 $\sim 100\mu s$，线程 $\sim 1$ms~10ms。高并发下协程开销低 1-2 数量级。

### 4.3 event loop 的 fairness

event loop 是**协作式 + 单线程**——一个协程若不 `await`，独占 CPU，其他协程饿死。设 loop 有 $n$ 协程，每协程两次 await 间跑 $t_i$ 时间（不让出段）。若一协程 $t_i\to\infty$（CPU 密集 for），其他全等。故 asyncio **不适合 CPU 密集**——CPU 密集协程会把 loop 占死。解法：CPU 密集段 `await asyncio.to_thread(fn)`（丢线程池，期间放 GIL 让 loop 跑别的——但线程池也受 GIL，真 CPU 并行仍需 multiprocessing）。

### 4.4 asyncio 的"伪并行"与 GIL

asyncio 单线程，**GIL 不影响它**（无竞争）。但若协程里 `await asyncio.to_thread(cpu_heavy)` 把 CPU 活丢线程池，**线程池多线程仍受 GIL**——CPU 密集无法多核并行，只是 IO 不阻塞。真 CPU 并行必须 multiprocessing。这是 asyncio + CPU 的根本局限。

### 4.5 gather 的并发度与背压

`gather(*[coro_i for i in range(N])}` 一次起 $N$ 协程。若 $N$ 太大（如 $10^6$ 请求同时 forward）：GPU 显存爆、fd 耗尽、内存爆。需**限流**——`asyncio.Semaphore(N_max)` 限制并发度：

```python
sem = asyncio.Semaphore(100)  # 最多 100 并发
async def handle(req):
    async with sem:
        return await forward(req)
await asyncio.gather(*[handle(r) for r in requests])  # 最多 100 同时跑
```

推理 server 的 batch scheduler（[[continuous batching的调度实现]]）本质就是这个——限制并发 batch 数防 GPU OOM。


## 5. 代码示例（可选）

### 5.1 基本 coroutine + gather

```python
import asyncio

async def fetch(i):
    await asyncio.sleep(0.5)   # 模拟 IO (非阻塞, 让出)
    return i * 2

async def main():
    results = await asyncio.gather(fetch(1), fetch(2), fetch(3))
    print(results)  # [2, 4, 6] (三个并发, 总 ~0.5s, 非 1.5s)

asyncio.run(main())
```

### 5.2 阻塞调用 vs 异步（陷阱演示）

```python
import asyncio, time

async def bad():
    time.sleep(1)   # 阻塞! loop 冻结, 别的协程等
    return "bad"

async def good():
    await asyncio.sleep(1)   # 让出, loop 跑别的
    return "good"

async def main():
    t = time.perf_counter()
    await asyncio.gather(bad(), bad())   # 串行, ~2s (bad 阻塞)
    print(f"bad: {time.perf_counter()-t:.2f}s")   # ~2.0s
    t = time.perf_counter()
    await asyncio.gather(good(), good())  # 并发, ~1s
    print(f"good: {time.perf_counter()-t:.2f}s")  # ~1.0s

asyncio.run(main())
```

### 5.3 阻塞调用丢线程池

```python
import asyncio, torch

async def forward(x):
    # torch 的 forward 是 C 扩展 (放 GIL), 但 host 端 launch + sync 若同步调用会卡 loop
    # 把它丢线程池, 期间 loop 跑别的协程
    return await asyncio.to_thread(lambda: torch.matmul(x, x).sum().item())

async def main():
    x = torch.randn(1024, 1024)
    results = await asyncio.gather(forward(x), forward(x), forward(x))
    print(results)

asyncio.run(main())
```

### 5.4 限流（Semaphore）防 OOM

```python
import asyncio

async def forward(req):
    await asyncio.sleep(0.1)  # 模拟 GPU forward
    return f"result_{req}"

sem = asyncio.Semaphore(50)  # 最多 50 并发 (防 GPU 显存爆)

async def handle(req):
    async with sem:
        return await forward(req)

async def main():
    requests = list(range(1000))
    results = await asyncio.gather(*[handle(r) for r in requests])
    print(len(results))  # 1000, 总耗时 ~2s (1000/50 * 0.1)

asyncio.run(main())
```

### 5.5 推理 server 的 asyncio 雏形

```python
import asyncio, aiohttp
from aiohttp import web

async def generate(request):
    prompt = (await request.json())["prompt"]
    # submit GPU forward, await 期间 loop 处理别的请求
    tokens = await asyncio.to_thread(run_model, prompt)
    return web.json_response({"tokens": tokens})

async def run_model(prompt):
    import torch
    # ... GPU forward ...
    return [1, 2, 3]

app = web.Application()
app.add_routes([web.post("/generate", generate)])
web.run_app(app, port=8000)
# 一个进程处理多并发 /generate 请求, forward 时 await 放行
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Python multiprocessing与GIL]]（asyncio 单线程绕开 GIL 竞争，但 CPU 密集仍受 GIL——两者分工）、Python 生成器（coroutine 的底层是 generator）
- **下游（应用）**: [[OpenAI API server]]（推理 server 的并发模型）、[[actor model]]（Ray asyncio actor）、[[Ray与分布式调度]]（async actor 调度多 rollout）、RL 框架的并发采样调度
- **对比 / 易混**:
  - **asyncio vs threading**：单线程协作式 vs 多线程抢占式；IO 密集 asyncio 优，CPU 密集都不行（GIL）。
  - **asyncio vs multiprocessing**：IO 并发 vs CPU 并行；常叠加（IO 用 asyncio、计算用进程）。
  - **asyncio vs threading 的 GIL**：asyncio 无 GIL 竞争（单线程），threading 有（但 IO 时放锁）。


## 7. 常见误区与易错点

> [!warning] 误区1：asyncio 让代码变快
> 错。asyncio **不提速单请求**（甚至略慢，await 开销）。它提升的是**高并发 IO 吞吐**（一个进程多请求重叠等待）。单请求延迟不变或略升。

> [!warning] 误区2：async def 自动异步
> 错。`async def` 里的**阻塞调用**（`time.sleep`、同步 `requests`、`torch.cuda.synchronize`、CPU for 循环）会**冻结整个 loop**。必须用异步 IO 库 + `await asyncio.to_thread` 把阻塞丢线程池。

> [!warning] 误区3：asyncio 能用多核
> 错。asyncio 单线程，无多核并行。CPU 密集仍受 GIL——`to_thread` 丢线程池也受 GIL。真并行用 multiprocessing。

> [!warning] 误区4：gather 没限流
> `gather(*[...])` 一次起全部协程，大并发爆资源（fd、显存、内存）。高并发必须 `Semaphore` 限流，推理 server 用 batch scheduler 限制并发。

> [!warning] 误区5：协程比线程快
> 不绝对。单请求串行下协程略慢（await 开销）。协程优势在**高并发 IO**（切换轻、内存小）。低并发或 CPU 密集场景协程无优势。

> [!tip] 实践：asyncio + 多进程的 ML 范式
> 推理 server：单进程 asyncio event loop 处理多请求 IO + 必要时 `to_thread` 跑 GPU forward（kernel 放 GIL，loop 切别的请求前处理）。RL 框架：Ray async actor 管 IO 调度 + 多 sync actor（多进程）管计算并行。两者分工：asyncio 管等待重叠，multiprocessing 管多核并行。

> [!tip] 实践：CPU 密集段必须让出
> 协程里跑 CPU 密集 for 循环会卡 loop。用 `await asyncio.sleep(0)`（让出一次）或 `await asyncio.to_thread(fn)` 丢线程池，让 loop 喘息。但真 CPU 并行仍需 multiprocessing。


## 8. 延伸细节

### 8.1 event loop 的实现（epoll/kqueue）

asyncio 的 event loop 底层用 OS 的高效 IO 多路复用：Linux `epoll`、macOS `kqueue`、Windows `IOCP`。一个 syscall 同时监听上千 fd，有 IO 就绪时返回，期间 loop 不占 CPU。这是单线程处理上千连接的根——`epoll` 是 $O(1)$（就绪事件数）而非 $O(n)$（连接数）。

### 8.2 TaskGroup（3.11+ 结构化并发）

```python
async with asyncio.TaskGroup() as tg:
    t1 = tg.create_task(coro1())
    t2 = tg.create_task(coro2())
# 退出 with 时等全部完成; 任一 raise -> 取消其他 + 抛 ExceptionGroup
```

比 `gather` 更安全（结构化、错误传播好），3.11+ 推荐替代 gather。

### 8.3 asyncio 与 CUDA stream 的配合

推理 server 里 `await forward()`——若 forward 是同步调用（`torch.matmul` 返回 tensor，需 `.cpu().item()` 才 sync），会让 loop 在 sync 处阻塞。正确姿势：submit kernel 后**不等 sync**，返回 stream 让 await 在"stream 完成"的 future 上——但 torch 没原生 await stream，需 `asyncio.Future` + `cuda.Stream` 的 callback 或 `to_thread`。vLLM/OpenAI server 的实现细节待核实，但原理是**把 GPU forward 放线程池或异步化**，loop 期间处理别的请求。

### 8.4 uvloop（更快 event loop）

[uvloop](https://github.com/MagicStack/uvloop) 是 asyncio event loop 的 C 加速实现（基于 libuv），比纯 Python loop 快 2-4x。生产推理 server 常用 `asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())` 提速。aiohttp/fastapi 都兼容 uvloop。

### 8.5 async actor 的并发度 vs 多 actor

Ray async actor 一个进程内并发多请求（IO 重叠），但 CPU 串行（GIL）。若要 CPU 并行，起多个 sync actor（多进程）。verl/OpenRLHF 的 rollout worker 拓扑常：**多 sync actor（每 actor 多卡或单卡）拿 GPU 并行 + 每 actor 内 asyncio 调度多 sample 请求**——两层并发。详见 [[actor model]]、[[Ray与分布式调度]]、[[RL角色拓扑]]。

### 8.6 asyncio 与 [[Python multiprocessing与GIL]] 的互补

| 任务类型 | 正解 |
|---|---|
| 高并发 IO（请求多、等 GPU/网络多） | asyncio（单进程单线程协程） |
| CPU 密集（数据增强、reward 计算） | multiprocessing（多进程绕 GIL） |
| 混合 | asyncio 管 IO 调度 + 子进程/线程池跑 CPU 活 |
| 多卡训练 | multiprocessing（DDP 每进程一卡）+ 不用 asyncio（计算为主） |

两者是 ML 并发体系的两大支柱，互补不互斥。详见 [[Python multiprocessing与GIL]] §3.8 对比表。


---
相关: [[并发与生命周期]] | [[Python multiprocessing与GIL]] | [[C++生命周期与并发]] | [[OpenAI API server]] | [[actor model]] | [[Ray与分布式调度]] | [[torch.distributed]] | [[20-可靠性与开源工程]]
