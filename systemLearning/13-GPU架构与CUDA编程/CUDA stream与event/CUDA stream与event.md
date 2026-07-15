# CUDA stream与event

> **所属章节**: [[CUDA stream与event]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: CUDA stream / stream / event / 默认流 / 非默认流 / stream synchronization
> **难度**: 中（需懂异步编程 + GPU 执行模型）


## 1. 一句话定义

**CUDA stream** 是 GPU 上**按提交顺序串行执行**的命令队列——一个 stream 内的 kernel/memcpy 严格先后执行，**不同 stream 间可并发执行**（计算与计算 overlap、计算与拷贝 overlap）；**CUDA event** 是插在 stream 里的**同步标记点**，用于测量两 event 间耗时或跨 stream 等待（`cudaStreamWaitEvent` 让一个 stream 等另一 stream 的某 event 完成才继续）。stream/event 是把 GPU 的"异步 + 多引擎并发"能力暴露给程序员的接口，是 [[overlap strategy]] 在单机内的最底层实现手段，也是 [[CUDA Graph与graph capture]] 的执行载体。

> [!note] 三句话定位
> - **是什么**：stream = 命令串行队列，event = 同步标记；多 stream 并发，event 做跨流同步/计时。
> - **为什么**：GPU 有 compute engine + copy engine 多个执行单元，单 stream 串行用不满；多 stream 让它们并发。
> - **与 default stream 的坑**：default stream 对所有非默认流隐式同步（legacy 默认行为），会意外 serialize 多流并发，调优常用 `cudaStreamNonBlocking` 或 per-thread default stream。


## 2. 为什么需要它（动机与背景）

### 2.1 GPU 是异步机器

CPU 调 `kernel<<<...>>>()` 后**立即返回**（不阻塞），GPU 在后台执行。CPU 端的下一个语句可能马上又 launch 别的 kernel。这种异步让 CPU 不用等 GPU，但带来一个问题：**CPU 怎么知道 GPU 做完了？** 怎么安排多个 kernel 的顺序？答案就是 stream。

### 2.2 多引擎并发：计算 + 拷贝

GPU 物理上有**多个引擎**：compute engine（执行 kernel）、copy engine（执行 H2D/D2H memcpy）。A100 有 2 个 copy engine + 若干 compute engine。单 stream 内 memcpy 和 kernel 串行（不能并发）；但**不同 stream 的 memcpy 与 kernel 可并发**——A stream 在算，B stream 在拷。

这是单机内**计算-通信 overlap** 的基础。例如：训练时把下一 batch 的 H2D 拷贝（B stream）与本 batch 的前向（A stream）并发，藏数据传输于计算。详见 [[overlap strategy]]。

### 2.3 跨 stream 同步与计时

多 stream 并发后，有时需要"B stream 的 memcpy 完成后，A stream 才能用这批数据"——跨 stream 依赖。用 **event**：B stream 在 memcpy 后 `record(event)`，A stream `waitEvent(event)` 等待。event 也用于精确计时（`cudaEventElapsedTime` 记录 GPU 端两 event 间真实耗时，比 CPU 端 `time.time()` 准，因后者含异步 launch 的偏差）。


## 3. 核心概念详解

### 3.1 stream 的语义

| 属性 | 说明 |
|---|---|
| 命令顺序 | 同 stream 内**严格按提交顺序**执行（FIFO） |
| 跨 stream | 不保证顺序，可并发（视资源空闲） |
| 同步 | `cudaStreamSynchronize(s)` 阻塞 CPU 直到 s 全部完成；`cudaDeviceSynchronize()` 等所有 stream |
| 查询 | `cudaStreamQuery(s)` 不阻塞，返回是否完成 |

### 3.2 default stream vs 非默认流

- **default stream（流 0）**：不显式指定 stream 时用。legacy 行为下，**对设备上所有非默认流隐式同步**——任何 default stream 的操作会等所有非默认流完成，且阻塞后续非默认流直到它完成。这会意外 serialize 多流。
- **per-thread default stream**（编译时 `-cudart=shared` 或 `--default-stream per-thread`）：每线程有自己的 default stream，互不同步，是现代默认。
- **非默认流**：`cudaStreamCreate` 创建，互不隐式同步（除非用 event 显式等）。

> [!warning] legacy default stream 的坑
> 多流调优时若混用 default stream，会因隐式同步失去并发。要么全用非默认流，要么开 per-thread default stream。

### 3.3 event 的两个用途

**计时**：
```python
start, stop = cudaEvent_t(), cudaEvent_t()
cudaEventRecord(start, s)
kernel<<<...>>>(...args)          # 提交到 stream s
cudaEventRecord(stop, s)
cudaEventSynchronize(stop)         # 等 stop 完成
ms = cudaEventElapsedTime(start, stop)   # GPU 端真实耗时
```

**跨 stream 同步**：
```python
cudaEventRecord(e, streamA)       # A stream 录 event e
cudaStreamWaitEvent(streamB, e)    # B stream 等 e 完成（不阻塞 CPU）
kernel<<<...>>>(...args) on streamB  # 此时 B 才开始
```

### 3.4 并发的条件

多 stream 真能并发的条件：

1. **不同 stream**（非 default 或 per-thread）。
2. **资源够**：SM 空闲、copy engine 空闲。若一个 stream 已塞满 SM，另一 stream 的 kernel 仍要等。
3. **无隐式/显式同步**：default stream 隐式同步、`cudaDeviceSynchronize`、未配 `cudaStreamNonBlocking` flag 等会 serialize。
4. **数据无冲突**：理论上若两 kernel 写同地址，硬件不保证顺序（需 event 同步）。并发是调度层允许，不保证内存一致性。

### 3.5 PyTorch 中的 stream

```python
s = torch.cuda.Stream()
with torch.cuda.stream(s):            # 设当前 stream = s
    out = model(input)                # 提交到 s
# s.synchronize()                     # 可选等待
e = torch.cuda.Event()
e.record(s)                           # 录 event
other_stream.wait_event(e)            # 跨流等
```

PyTorch 内部大量用多 stream 做计算-通信 overlap（如 DDP 的 reducer 用单独 stream 发 allreduce，与前向/backward 计算并发，见 [[gradient bucket与通信重叠]]）。vLLM/SGLang 的 [[model runner]] 也用多 stream 分离 attention 与采样。

### 3.6 stream 与 graph 的关系

[[CUDA Graph与graph capture]] 把一段 stream 序列捕获成图，replay 时只付一次 launch 开销。stream 是 graph 的捕获源与执行载体。


## 4. 数学原理 / 公式

### 4.1 多 stream overlap 的收益

设单 stream 串行耗时 $T = T_{\text{copy}} + T_{\text{compute}}$。理想 2 stream overlap（copy 与 compute 无依赖且资源够）：

$$
T_{\text{overlap}} = \max(T_{\text{copy}}, T_{\text{compute}}) \quad \text{(稳态)}
$$

收益 $\frac{T}{T_{\text{overlap}}} = \frac{T_c + T_k}{\max(T_c,T_k)}$。当 $T_c \approx T_k$ 时接近 $2\times$；当一方远大，收益趋近 1。

### 4.2 event 计时 vs CPU 计时

CPU 端 `t0 = time.time(); kernel(); t1 = time.time()`：因 kernel 异步，`t1-t0` ≈ launch 开销（几 μs），不含真实 GPU 执行。必须 `cudaEventRecord + synchronize + elapsedTime` 才测到 GPU 真实耗时。

### 4.3 资源约束下的并发上界

设 SM 占用率 $U_k$（kernel 单独跑的占用）、copy engine 数 $C$。两 kernel 并发的实际 SM 利用 $\min(U_{k1}+U_{k2}, 1)$，若和 >1 则被 serialize 一部分。故多 stream 并发收益受空闲资源限。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟多 stream overlap

```python
def overlap_streams(copy_t, compute_t, n_stream=2):
    """模拟 n 个 stream 的 copy+compute overlap."""
    # 串行 (单 stream)
    serial = copy_t + compute_t
    # 理想 overlap (稳态): max(copy, compute)
    overlap = max(copy_t, compute_t)
    return serial, overlap

# copy 与 compute 相近 -> 2x 收益
s, o = overlap_streams(10, 10)
print(f"串行 {s}, overlap {o}, 收益 {s/o:.2f}x")
# 一方远大 -> 收益接近 1
s, o = overlap_streams(10, 100)
print(f"串行 {s}, overlap {o}, 收益 {s/o:.2f}x")
```

### 5.2 PyTorch 多 stream + event 计时

```python
import torch
s = torch.cuda.Stream()
e_start = torch.cuda.Event(enable_timing=True)
e_end = torch.cuda.Event(enable_timing=True)

with torch.cuda.stream(s):
    e_start.record()
    out = torch.matmul(torch.randn(8192, 8192, device='cuda'),
                       torch.randn(8192, 8192, device='cuda'))
    e_end.record()
torch.cuda.synchronize()                       # 等所有 stream 完成
ms = e_start.elapsed_time(e_end)               # GPU 真实耗时
print(f"matmul 8192^2 耗时: {ms:.2f} ms")
```

### 5.3 跨 stream 同步

```python
sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
with torch.cuda.stream(sa):
    data = torch.randn(1024, 1024, device='cuda')
    ev = torch.cuda.Event()
    ev.record(sa)                               # sa 录 event
with torch.cuda.stream(sb):
    sb.wait_event(ev)                           # sb 等 sa 的 ev
    out = data @ data                           # 此时 data 已就绪
torch.cuda.synchronize()
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GPU执行模型]]（kernel launch 异步、多引擎）、GPU 硬件（compute/copy engine）。
- **下游（应用）**: [[异步memcpy与pinned memory]]（copy engine + stream 的典型用法）、[[overlap strategy]]（stream 是单机 overlap 的载体）、[[gradient bucket与通信重叠]]（DDP/Megatron 用独立 stream 跑 NCCL）、[[CUDA Graph与graph capture]]（graph 捕获 stream 序列）、[[kernel launch overhead]]（多 launch 的开销与 stream 调度）、[[torch profiler]]/[[nsys (Nsight Systems)|nsys]]（trace 看 stream 时间轴）。
- **对比 / 易混**:
  - **stream vs event**：stream 是命令队列（载体），event 是同步标记（控制点）。
  - **stream vs thread**：stream 是 GPU 命令队列，thread 是 CPU 执行单元。一个 CPU 线程可往多 stream 提交。
  - **default stream vs 非默认流**：default 隐式同步（legacy），非默认/ per-thread 不隐式同步。


## 7. 常见误区与易错点

> [!warning] 误区 1：用 CPU time 计时 GPU kernel
> `time.time()` 包到 `kernel()` 前后只测到 launch 开销（异步立即返回）。必须用 event + synchronize + elapsedTime。

> [!warning] 误区 2：以为多个 stream 一定并发
> 不是。若都用 default stream（legacy）、或资源被占满、或隐式同步，仍 serialize。并发是允许条件，非保证。

> [!warning] 误区 3：跨 stream 数据竞争靠"提交顺序"保证
> 不同 stream 间**无内存顺序保证**。若 A stream 写、B stream 读同地址，必须用 event `cudaStreamWaitEvent` 显式同步，否则有 race。

> [!warning] 误区 4：忘了 synchronize 就读结果
> 异步 launch 后 CPU 立即继续，若马上 `tensor.cpu()`（隐含同步）或打印，可能拿到未完成结果。要么显式 `synchronize`，要么靠框架的 stream 序。

> [!warning] 误区 5：stream 太多反而劣化
> 每个 stream 有管理开销，过多 stream 会让调度器选择困难、内存占用增。常用 2-4 个 stream 做 copy/compute 分离，不要盲目开几十个。


## 8. 延伸细节

### 8.1 stream priority

`cudaStreamCreateWithPriority` 可设优先级，高优先流抢占低优（有限支持）。用于实时推理 vs 后台训练混合场景。

### 8.2 cudaStreamNonBlocking flag

非默认流创建时可加 `cudaStreamNonBlocking`，禁止它与 default stream 同步（更纯的并发）。legacy default stream 下推荐。

### 8.3 NCCL 与 stream

NCCL collective（all-reduce 等）在内部 stream 上跑，也可指定外部 stream。`work.wait()` 本质是 stream 同步。DDP reducer 把 NCCL 调用放独立 stream 与 backward 计算并发，是 [[gradient bucket与通信重叠]] 的实现。

### 8.4 内容来源

stream/event 语义整理自 NVIDIA CUDA C++ Programming Guide 的 Streams 与 Events 章节，PyTorch 包装见 `torch.cuda` 文档。

---
相关: [[CUDA stream与event]] | [[异步memcpy与pinned memory]] | [[overlap strategy]] | [[gradient bucket与通信重叠]] | [[CUDA Graph与graph capture]] | [[kernel launch overhead]] | [[nsys (Nsight Systems)]] | [[torch profiler]]
