# Python multiprocessing 与 GIL

> **所属章节**: [[并发与生命周期]]
> **所属模块**: [[20-可靠性与开源工程]]
> **别名**: GIL / 全局解释器锁 / multiprocessing / fork vs spawn / CPython GIL / DataLoader num_workers
> **难度**: 中（需懂 [[torch.distributed]]、[[Distributed Data Parallel]]、Python 进程/线程模型、C 扩展与 GIL 交互）


## 1. 一句话定义

**GIL（Global Interpreter Lock，全局解释器锁）** 是 CPython 实现的一个**进程级互斥锁**——任一时刻**只有一个线程能执行 Python 字节码**（ bytecode），导致多线程在 **CPU 密集任务上无法真并行**（多核跑不满）；**multiprocessing** 则用**多进程**绕开 GIL（每个进程有独立解释器、独立 GIL，真并行），是 Python 里 CPU 密集并行的事实标准；在 ML 工程里它体现在两条主线——① **DataLoader 的 `num_workers`**（用多进程预取数据，绕 GIL 做数据增强/IO），② **DDP/[[Distributed Data Parallel|DDP]] 的 "每进程一卡"**（`torchrun --nproc_per_node=N` 起N个进程，每进程独立 GIL + 独立 CUDA context，互不抢锁）。GIL 只锁 **Python 字节码**，不锁 **C 扩展里跑的 native 代码**——所以 NumPy/Torch 的 C++/CUDA kernel 在执行时会**显式释放 GIL**（`Py_BEGIN_ALLOW_THREADS`），让其他 Python 线程同时跑，这也是为什么 GPU 训练用多线程/单进程多卡在计算阶段不卡 GIL（kernel 在 GPU 上跑、主线程已放锁）。注意 **PEP 703（GIL-free CPython，nogil）** 已合入 3.13+ 实验分支，但 3.13/3.14 默认仍带 GIL，nogil 是可选构建，生产 ML 栈（PyTorch/CUDA）短期仍以带 GIL 的 CPython 为准，待核实。

> [!note] 三句话定位
> - **是什么**：CPython 进程级互斥锁，任一时刻只一个线程跑 Python 字节码；多线程 CPU 密集任务被它锁成串行。
> - **为什么**：CPython 用引用计数管内存，多线程改 refcount 不安全，GIL 是最简方案——一把锁保住引用计数线程安全，代价是字节码串行。
> - **怎么绕**：CPU 密集 → multiprocessing（多进程，每进程独立 GIL）；IO 密集 → [[asyncio]] 或多线程（IO 时释放 GIL）；C 扩展/CUDA → `Py_BEGIN_ALLOW_THREADS` 显式释放，kernel 期间不持锁。


## 2. 为什么需要它（动机与背景）

### 2.1 CPython 的引用计数 vs 线程安全

CPython 内存管理的基石是**引用计数**（reference counting）：每个 PyObject 有一 `ob_refcnt`，`Py_INCREF`/`Py_DECREF` 改它，refcnt=0 立即释放。问题：若多线程同时改同一对象的 refcnt，**race condition**（数据竞争）会让 refcnt 错乱——漏减→内存泄漏，多减→double-free 崩溃。CPython 选了**最简单的方案**：一把全局锁（GIL）保证任一时刻只一个线程跑字节码（包括 refcnt 操作），就线程安全了。代价：**多核 CPU 密集任务无法并行**（被锁串行）。

> [!tip] 为什么不是细粒度锁
> 早期尝试过细粒度锁（free-threading，1999 前的 Greg Stein patch），但锁竞争开销大、复杂度高，被弃。GIL 简单粗暴但单线程性能好（无锁开销）。这是**单线程性能 vs 多线程并行**的历史 trade-off，nogil（PEP 703）正试图两者兼得。

### 2.2 多线程跑不满 CPU

```python
# 2 个 CPU 密集线程, 期望 ~2x 快, 实际几乎不快 (GIL 串行)
import threading
def burn():
    s = 0
    for i in range(50_000_000):
        s += i

t1 = threading.Thread(target=burn)
t2 = threading.Thread(target=burn)
t1.start(); t2.start()
t1.join(); t2.join()
# 双核 CPU 利用率 ~100% (单核), 不是 200% (双核) -> GIL 锁死
```

### 2.3 multiprocessing 绕 GIL

multiprocessing 起独立 OS 进程，每进程**独立 Python 解释器、独立 GIL、独立内存空间**——多进程 CPU 密集真并行（各占一核，无锁竞争）。

### 2.4 ML 里的两条主线

- **DataLoader `num_workers`**：数据增强（PIL/numpy 操作多在 Python 层）若单线程做，主训练循环等数据→GPU 闲置。`num_workers>0` 起多进程并行预取，每进程独立 GIL，数据增强真并行。
- **DDP `--nproc_per_node`**：[[Distributed Data Parallel]] 不用多线程而用**多进程**，每进程绑一卡（`set_device`），各自独立 GIL/CUDA context，靠 [[torch.distributed]] 的 NCCL 通信。若用单进程多线程，多卡 forward 时 GIL 会让 host 端 launch kernel 串行，且 CUDA context 共享有开销。


## 3. 核心概念详解

### 3.1 GIL 的持有与释放

GIL 不是"一直持有"，CPython 在两类点**主动释放**：

1. **IO 操作**：`socket.recv`、`file.read`、`time.sleep` 等 blocking IO 前**释放 GIL**，让其他线程跑；IO 完成再 reacquire。这是多线程 IO 并发能 work 的根。
2. **C 扩展显式释放**：C 扩展用 `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` 宏手动放锁，期间跑 native 代码（不碰 PyObject 就安全）。NumPy 的 ufunc、Torch 的 aten op、CUDA kernel launch 都这样——kernel 提交到 stream 后立即放锁，主线程去 launch 下一个，kernel 在 GPU/SM 上异步执行。
3. **check interval（`sys.setswitchinterval`）**：字节码解释器每执行 `check_interval`（默认 5ms，3.2+ 是时间间隔不是指令数）次**主动让出 GIL** 给其他线程（cooperative + preemptive 混合），防一个线程独占。

> [!warning] 误区：GIL 让多线程完全没用
> 错。IO 密集任务（网络/磁盘）多线程**有效**（IO 时放锁，其他线程跑）；CPU 密集但用 NumPy/Torch 的任务多线程**部分有效**（C kernel 期间放锁，但 Python glue 仍串行）。纯 Python 算 `for i in range: s+=i` 这种**才**完全串行。

### 3.2 fork vs spawn（multiprocessing start method）

`multiprocessing` 起子进程有三种 start method：`fork`、`spawn`、`forkserver`。

| 维度 | fork | spawn | forkserver |
|---|---|---|---|
| 机制 | `fork()` 系统调用，子进程**复制父进程内存**（COW） | `os.exec` 全新启动 Python，重新 import | fork 一个 server 进程，后续 fork 它 |
| 速度 | **快**（只复制页表，COW 按需拷） | **慢**（重新 init、重新 import、重新建对象） | 中（首启动慢，后续快） |
| 状态继承 | 继承父进程**全部内存**（fd、锁、线程状态……） | **干净**，只继承 pickle 过去的参数 | 干净（从 server fork） |
| 线程安全 | **不安全**：fork 只复制当前线程，其他线程状态丢失；持有锁的线程被 fork 走→死锁 | 安全（全新进程无继承） | 安全 |
| macOS/Windows | macOS 自 3.8 默认改 **spawn**（fork 不安全）；Win **只支持 spawn** | 是 | 是 |
| CUDA fork | **危险**：fork 不继承 CUDA context，子进程用 CUDA 易崩；Torch 在 fork 后用 CUDA 会报错 | 安全 | 安全 |
| 典型场景 | 简单 CPU 任务、无线程无 CUDA 的老 Linux 程序 | ML 首选（Torch/DataLoader macOS 默认） | 高频起停进程的服务 |

> [!warning] 误区：fork 总是更快更好
> 错。fork 快但有**COW 陷阱**（写时才拷，但 Python 对象 refcnt 改动会触发整页拷，大内存可能更慢）、**线程死锁**（fork 时别的线程持有 malloc 锁，子进程继承死锁状态）、**CUDA 崩溃**。Python 3.8+ macOS 把默认从 fork 改 spawn 就是为这些坑。生产 ML 用 spawn 更稳。

### 3.3 fork 的 COW 陷阱（copy-on-write）

fork 后子进程"共享"父进程内存页（只读），写时才拷（COW）。看似省内存，但 Python 每个 PyObject 有 `ob_refcnt`，**子进程哪怕只是"读"对象，解释器也会 `Py_INCREF` 改 refcnt**——触发该页 COW 拷贝。大数组（如模型权重、大 dataset）的 refcnt 散布各页，子进程一访问就逐页拷贝，**实际内存接近全拷**，省内存的预期落空。这是 DataLoader `num_workers` + fork 喂大 dataset 内存爆的常见根因。

### 3.4 spawn 的干净代价

spawn 子进程全新启动 Python：重新 import 所有模块、重新执行模块顶层代码、参数靠 pickle 传。慢但**状态干净**——没有继承的锁、没有继承的线程、没有继承的 CUDA context。Torch 在 macOS 默认 spawn，DataLoader `num_workers` 也倾向 spawn（避免 fork+CUDA 崩）。代价：起进程慢（import torch 可能要 1-2s），所以 DataLoader 用 **persistent workers**（worker 不销毁，跨 epoch 复用）摊销启动开销。

### 3.5 DataLoader 的 multiprocessing

```python
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, num_workers=4,
                    persistent_workers=True,  # 跨 epoch 复用, 免每 epoch 重启 worker
                    prefetch_factor=2)        # 每 worker 预取 batch 数
# 4 个子进程 (spawn) 并行: 各独立 GIL, 数据增强 (PIL/numpy) 真并行
# 主进程 forward 时, worker 已把下一 batch 准备好 (queue)
```

- `num_workers=0`：主进程单线程取数据，数据增强卡 GIL，GPU 等。
- `num_workers>0`：多进程并行预取，每 worker 独立 GIL，数据增强真并行；主进程 forward/反向时不阻塞取数据。
- `persistent_workers`：worker 跨 epoch 活着，免每 epoch 重新 spawn（spawn 慢，import torch 1s+）。
- `prefetch_factor`：每 worker 预取几个 batch，平滑 IO 抖动。

### 3.6 为什么 DDP 用多进程而非多线程

[[Distributed Data Parallel]] 用 `torchrun --nproc_per_node=N` 起 N 进程，每进程绑一卡。为什么不用单进程多线程？

1. **GIL 串行 launch kernel**：多线程共享 GIL，host 端 launch CUDA kernel 的 Python 调用串行，多卡无法重叠。
2. **CUDA context 共享有开销**：单进程多卡需多 context 或 MPS，管理复杂；多进程每进程独立 context，Torch `set_device` 后干净。
3. **崩溃隔离**：一卡 OOM/崩，只杀该进程，其他卡继续（配合 [[elastic training]] 重启该 rank）。
4. **NCCL 设计**：[[torch.distributed]] 的 NCCL backend 每进程一个 communicator，多进程模型自然。
5. **scale-out**：多进程模型天然扩展到多机（每机 N 进程），单进程多线程难跨机。

### 3.7 GIL 与 CUDA kernel 的关系（关键澄清）

> [!warning] 误区：GIL 阻碍 GPU 训练
> 错。GIL 只锁 Python 字节码，不锁 CUDA kernel。Torch 的 op（如 `torch.mm`）在 C++ 层 `Py_BEGIN_ALLOW_THREADS` 释放 GIL 后提交 kernel 到 stream，然后返回 Python（reacquire GIL）。期间：
> - GPU 上 kernel 异步跑（不持 GIL）
> - 主线程 reacquire GIL 跑下一段 Python（如下一个 op 的 Python glue）
> - 其他 Python 线程可在主线程放锁间隙跑
> 所以 GPU 训练的**计算阶段不被 GIL 卡**（kernel 在 GPU 上异步），卡的是**Python glue 的串行 launch**（多个 op 间 host 端串行）。这也是 [[torch profiler]] 能看到 host 端 gap 的来源。

### 3.8 多进程 vs 多线程 vs asyncio 对比

| 维度 | multiprocessing（多进程） | threading（多线程） | [[asyncio]]（协程） |
|---|---|---|---|
| 并行度 | **真并行**（独立 GIL，多核） | CPU 密集被 GIL 串行 | 单线程，无并行（并发 IO） |
| IO 密集 | 有效但重（进程开销） | 有效（IO 放 GIL） | **最佳**（无切换开销） |
| CPU 密集 | **最佳** | 差（GIL） | 差（单线程+GIL） |
| 内存 | 独立空间，IPC 靠 pickle/queue/共享内存 | 共享内存（需锁） | 共享内存（单线程无竞争） |
| 启动开销 | 重（fork 快/spawn 慢） | 中 | 轻 |
| 通信成本 | 高（pickle 跨进程） | 低（共享内存） | 极低（同线程） |
| ML 典型 | DDP、DataLoader workers | 极少（被 GIL 限制） | 推理 API server、async actor |
| 崩溃隔离 | 好（进程崩不连累父） | 差（线程崩进程崩） | 差（协程崩 event loop 崩） |


## 4. 数学原理 / 公式

### 4.1 GIL 的"有效并行度"模型

设一段计算：Python 字节码占比 $\alpha$（glue），C 扩展（放 GIL）占比 $1-\alpha$，单线程耗时 $T$。

- **多线程**（$n$ 线程，CPU 密集）：
  - Python 字节码段被 GIL 串行，总耗时 $\approx n \cdot \alpha T$（n 个线程的 Python 段串行）。
  - C 扩展段并行，总耗时 $\approx (1-\alpha)T$（重叠）。
  - 总耗时 $\approx n\alpha T + (1-\alpha)T$，加速比 $S_n = \dfrac{T}{n\alpha T + (1-\alpha)T} = \dfrac{1}{n\alpha + (1-\alpha)}$。
  - $\alpha\to1$（纯 Python）：$S_n\to 1/n$（n 倍慢，因调度开销）。
  - $\alpha\to0$（纯 C kernel）：$S_n\to1$（无加速，因 GIL 不卡 kernel 但 Python glue 仍串）。
  - **结论**：多线程 CPU 密集几乎无加速，甚至负加速（锁开销）。

- **多进程**（$n$ 进程，独立 GIL）：
  - Python 段也并行，总耗时 $\approx T/n$（理想），加速比 $S_n\to n$（受 IPC/通信开销打折扣）。
  - **结论**：多进程才是 CPU 密集并行正解。

### 4.2 Amdahl 定律与 multiprocessing 的加速上限

若任务串行部分占比 $s$（如 IPC pickle、通信），并行部分 $1-s$，$n$ 进程加速比：

$$S_n = \frac{1}{s + (1-s)/n}$$

$n\to\infty$ 时 $S_n\to 1/s$。故 multiprocessing 的瓶颈是**串行部分**（pickle IPC、GIL-free 段外的 Python glue、进程启动）。DataLoader 用 `persistent_workers` + 共享内存 tensor（`pin_memory`、`SHM`）压 $s$。

### 4.3 check interval 的公平性

GIL 的 `switch_interval`（默认 5ms）决定线程切换频率。设 $n$ 线程，每线程在 interval 内拿 GIL 跑 5ms 再让出。理想下各线程 CPU 公平（各 $1/n$）。但若一线程在做 IO（放 GIL 等待），其他线程立即抢占——IO 线程不占 interval，CPU 线程满跑。这是 IO + CPU 混合时多线程仍有效的根。

### 4.4 fork 的 COW 失效估算

父进程大数组 $M$ 字节（$P$ 页，页大小 $B$，$P=M/B$）。子进程"读"对象触发 `Py_INCREF` 改 refcnt——refcnt 在 PyObject 头部，每个对象一页。若对象分布密集，几乎每页被改→COW 全拷，子进程内存 $\approx M$。故 fork 省**初始**内存（fork 瞬间只拷页表），但**首次访问后**接近全拷。对 DataLoader 大 dataset，`num_workers=4` fork 可能让总内存 $\approx 5M$（父+4子各拷一份）。


## 5. 代码示例（可选）

### 5.1 多线程 vs 多进程 CPU 密集对比

```python
import threading, multiprocessing, time

def burn(n=20_000_000):
    s = 0
    for i in range(n):
        s += i
    return s

def time_it(fn, *args):
    t = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t

# 单线程基线
t0 = time_it(burn)

# 多线程 (GIL 串行, 几乎不快)
def thread_burn(nthreads):
    ts = [threading.Thread(target=burn) for _ in range(nthreads)]
    [t.start() for t in ts]; [t.join() for t in ts]
t_thread = time_it(thread_burn, 2)   # ~2*t0 (无加速, 甚至更慢)

# 多进程 (真并行, ~t0)
def proc_burn(nprocs):
    ps = [multiprocessing.get_context("spawn").Process(target=burn) for _ in range(nprocs)]
    [p.start() for p in ps]; [p.join() for p in ps]
t_proc = time_it(proc_burn, 2)       # ~t0 (理想 2x 加速)
print(f"single={t0:.2f}s thread={t_thread:.2f}s proc={t_proc:.2f}s")
# single=1.20s thread=2.40s proc=1.25s  (线程无加速, 进程近 2x)
```

### 5.2 fork vs spawn 演示

```python
import multiprocessing as mp
import os

# 父进程设一个全局, 看子进程是否继承
GLOBAL = [f"from pid {os.getpid()}"]

def worker(method):
    ctx = mp.get_context(method)
    p = ctx.Process(target=lambda: print(f"[{method}] pid={os.getpid()} sees GLOBAL={GLOBAL}"))
    p.start(); p.join()

# fork: 子继承父的 GLOBAL (同 pid 内容)
worker("fork")     # [fork] pid=... sees GLOBAL=['from pid <parent>']
# spawn: 子全新启动, GLOBAL 是模块默认值 (不继承父进程的修改)
worker("spawn")    # [spawn] pid=... sees GLOBAL=[...] (重新执行模块顶层)
```

### 5.3 C 扩展释放 GIL（torch op 期间其他线程跑）

```python
import threading, torch, time

def heavy_gpu():
    a = torch.randn(4096, 4096, device="cuda")
    for _ in range(100):
        a = a @ a           # C++ 层 Py_BEGIN_ALLOW_THREADS 放 GIL, kernel 异步跑
    torch.cuda.synchronize()

def light_cpu():
    s = 0
    for i in range(5_000_000):
        s += i             # Python 字节码, 需 GIL

t = time.perf_counter()
t1 = threading.Thread(target=heavy_gpu)
t2 = threading.Thread(target=light_cpu)
t1.start(); t2.start(); t1.join(); t2.join()
print(f"{time.perf_counter()-t:.2f}s")
# 若 GIL 不放: light_cpu 等 heavy_gpu 的 Python glue 串行, 总 ~T_gpu+T_cpu
# 实际: heavy_gpu 的 kernel 期间放 GIL, light_cpu 跑 -> 总 ~max(T_gpu, T_cpu)
```

### 5.4 DataLoader 多进程预取

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __len__(self): return 10000
    def __getitem__(self, i):
        # 数据增强 (Python 层多, GIL 卡) -> num_workers 并行
        import numpy as np
        x = np.random.randn(3, 224, 224).astype("float32")
        y = i % 10
        return x, y

loader = DataLoader(MyDataset(), batch_size=64,
                    num_workers=8,            # 8 进程, 独立 GIL, 数据增强真并行
                    persistent_workers=True,  # 跨 epoch 复用, 免每 epoch spawn
                    pin_memory=True,          # 预锁页内存, 加速 host->GPU
                    prefetch_factor=2)         # 每 worker 预取 2 batch
for x, y in loader:
    x = x.cuda(non_blocking=True)  # 异步拷贝 (pin_memory + non_blocking)
    ...  # forward
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[torch.distributed]]（DDP 的多进程通信底座）、[[rank与world size]]（多进程的身份标识）、Python 进程/线程/COW 操作系统概念
- **下游（应用）**: [[Distributed Data Parallel]]（每进程一卡的多进程模型）、[[Data Parallel]]（DP 是单进程多线程，受 GIL 限制，DDP 用多进程绕开）、DataLoader `num_workers`（数据预取并行）、[[asyncio]]（IO 密集的另一条路，与 multiprocessing 互补）、[[OpenAI API server]]（推理 server 的并发选型）
- **对比 / 易混**:
  - **[[Data Parallel]]（DP）vs [[Distributed Data Parallel]]（DDP）**：DP 单进程多线程（受 GIL，慢、卡），DDP 多进程（绕 GIL，快、scale）。详见 [[Data Parallel]] 与 [[Distributed Data Parallel]] 的对比小节。
  - **multiprocessing vs threading vs [[asyncio]]**：CPU 密集用进程，IO 密集用 asyncio，混用各取所长。
  - **GIL vs CUDA**：GIL 不锁 kernel（C 扩展放锁），别误以为 GPU 训练被 GIL 卡死。


## 7. 常见误区与易错点

> [!warning] 误区1：GIL 让 Python 多线程一无是处
> 错。IO 密集多线程有效（IO 放 GIL）；CPU 密集但用 NumPy/Torch 的多线程部分有效（C kernel 放锁）。只有纯 Python `for` 循环算术才完全串行。

> [!warning] 误区2：fork 总比 spawn 快
> fork 启动快，但 COW 陷阱（refcnt 改动触发全页拷）+ 线程死锁（fork 只复制当前线程，别的线程的锁继承成死锁）+ CUDA 崩（fork 不继承 CUDA context）。ML 生产用 spawn 更稳。

> [!warning] 误区3：DataLoader num_workers 越多越好
> 错。worker 进程有内存开销（spawn 后 dataset 拷贝）、CPU 核数上限、pickle IPC 开销。经验：`num_workers = min(CPU核数, 某个 4~16)`，配 `persistent_workers` 摊销启动。

> [!warning] 误区4：GIL 阻碍 GPU 训练
> 错。GPU 训练的 kernel 在 GPU 异步跑，C 扩展 `Py_BEGIN_ALLOW_THREADS` 放锁，主线程可同时 launch 下一个 op 或跑其他线程。卡的是 **Python glue 的串行 launch**（host 端 gap），不是 kernel 本身。

> [!warning] 误区5：multiprocessing 的子进程共享内存
> 错。fork 子进程"继承"父内存（COW），但写时拷（且 refcnt 改动触发拷）；spawn 子进程全新内存。要真共享得用 `multiprocessing.shared_memory` 或 `torch.Tensor` 放 SHM。

> [!warning] 误区6：fork + CUDA 没事
> 危险。fork 不继承 CUDA context，子进程用 CUDA 易崩或静默错。Torch 在 fork 子进程用 CUDA 会报错。macOS/Win 默认 spawn 规避。

> [!tip] 实践：ML 多进程默认 spawn
> DataLoader 默认用 OS 默认（Linux fork、macOS spawn）。生产 ML 建议显式 `multiprocessing.get_context("spawn")` 或 DataLoader 在 macOS 用 spawn，避开 fork+CUDA/fork+线程死锁。

> [!tip] 实践：persistent_workers + pin_memory
> DataLoader 用 `persistent_workers=True`（跨 epoch 复用 worker，免每 epoch 重启慢）+ `pin_memory=True`（预锁页内存，`non_blocking=True` 异步拷 GPU）。两者是 DataLoader 性能标配。


## 8. 延伸细节

### 8.1 PEP 703（nogil CPython）

PEP 703（2023 接受）提出**移除 GIL 的 CPython**（nogil），用 biased reference counting + 无锁分配器替代 GIL 保线程安全。已合入 3.13 作为**实验可选构建**（`--disable-gil`），3.14 继续推进。但：① 默认仍带 GIL（nogil 需特制构建）；② PyTorch/CUDA 生态对 nogil 的适配滞后；③ nogil 单线程有 ~10% 回退（biased refcnt 开销）。**生产 ML 短期仍以带 GIL CPython 为准**，nogil 成熟后多线程 CPU 密集才可能真并行。截至 2026-07，nogil 仍标"实验"，待核实。

### 8.2 free-threading（PEP 703）对 ML 的影响

若 nogil 成熟：多线程 DataLoader 可替代多进程（省 IPC pickle）；单进程多卡 DDP 理论可行（但 CUDA context 仍倾向多进程）；RL 框架的 async actor 可能简化。但**短期内不改 ML 主流多进程架构**——DDP/DataLoader 的多进程模型成熟、scale-out 友好、崩溃隔离好，nogil 不会快速颠覆。

### 8.3 fork 的线程死锁实例

```python
import threading, os
lock = threading.Lock()
lock.acquire()  # 主线程持有

# 起一个后台线程也抢锁
def bg():
    lock.acquire()   # 等
threading.Thread(target=bg).start()

# fork: 只复制当前线程, bg 不在子进程, 但子进程继承了"lock 被某线程持有"状态
pid = os.fork()
if pid == 0:
    lock.acquire()   # 子进程死锁 (lock 状态是"已持有", 但持有者不在子进程)
    print("child got lock")  # 永不执行
```

这是 fork + threading 的经典死锁。glibc 的 malloc 内部锁、logging 的锁都可能踩。spawn 全新进程无此问题。

### 8.4 multiprocessing 的 IPC 与共享内存

- **Queue/Pipe**：基于 pickle，跨进程传对象需序列化，大 tensor 慢。
- **`shared_memory`（3.8+）**：`SharedMemory` 共享内存段，tensor 零拷贝传（只传 handle）。DataLoader 内部用。
- **`Array`/`Value`**：共享的 C 数组/标量，带锁。
- **`torch.Tensor` 放 SHM**：`tensor.share_memory_()`，多进程直接访问同一 GPU/CPU 显存，免 pickle。

### 8.5 DDP 的多进程启动（torchrun）

```bash
# 2 机 8 卡 (每机 4 卡), 每卡一进程
# node0:
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_endpoint=node0:29500 train.py
# node1:
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_endpoint=node0:29500 train.py
# -> 8 个进程, 各 rank 独立 GIL/CUDA context, NCCL 通信
```

每进程 `train.py` 里 `torch.cuda.set_device(local_rank)`，独立 CUDA context，[[Distributed Data Parallel]] 包 model，[[torch.distributed]] init process group。这是 GIL 时代 ML 多卡训练的标准范式。

### 8.6 与 [[Ray与分布式调度]] 的关系

Ray 的 actor/task 默认**多进程**（worker 进程），每进程独立 GIL。Ray 的 **async actor**（`asyncio` actor）在一个进程内用协程并发 IO（[[asyncio]]），计算仍单线程受 GIL——所以 Ray actor 分两类：sync actor（一个进程一个请求，多 actor 并行）、async actor（一个进程多请求并发 IO）。两者组合：IO 用 async actor、计算用 sync actor 多进程，与 multiprocessing + asyncio 的分工同构。详见 [[actor model]]、[[Ray与分布式调度]]。


---
相关: [[并发与生命周期]] | [[asyncio]] | [[C++生命周期与并发]] | [[torch.distributed]] | [[Distributed Data Parallel]] | [[Data Parallel]] | [[rank与world size]] | [[Ray与分布式调度]] | [[actor model]] | [[OpenAI API server]] | [[elastic training]] | [[20-可靠性与开源工程]]
