# torch profiler

> **所属章节**: [[profiling工具]]
> **所属模块**: [[10-性能分析与优化]]
> **别名**: torch profiler / PyTorch Profiler / torch.profiler / Kineto
> **难度**: 中（需懂 PyTorch + 性能分析基础）


## 1. 一句话定义

**torch profiler** 是 PyTorch 内置的**性能分析工具**（`torch.profiler.profile`），基于底层 Kineto 框架，一站式采集 **CPU 算子**（Python/C++ op）、**CUDA kernel**（启动、执行、耗时）、**内存**（分配、释放、峰值显存）、**通信**（NCCL all-reduce 等）等事件，输出 **Chrome JSON trace** 供 `chrome://tracing` 或 Perfetto 可视化。它是 PyTorch 用户最常用的 profiler——无需换工具、与 PyTorch 无缝集成、可视化 trace 直观定位瓶颈（CPU 算子慢？kernel launch 间隙？显存峰值？NCCL 等待？）。与 [[nsys]]（系统级、更底层）、ncu（kernel 级）互补：torch profiler 是**应用层入口**，先定位到哪类瓶颈，再决定是否下钻 nsys/ncu。

> [!note] 三句话定位
> - **是什么**：PyTorch 内置 profiler，采集 CPU/GPU/内存/通信事件，输出 Chrome trace 可视化。
> - **为什么**：训练/推理慢需定位瓶颈类型（CPU/GPU/launch/内存/通信），torch profiler 一站式、与 PyTorch 无缝。
> - **与 [[nsys]]/ncu 的关系**：应用层入口（先定位瓶颈类型），下钻用 nsys（系统级）或 ncu（kernel 级）。


## 2. 为什么需要它（动机与背景）

### 2.1 性能瓶颈的多样性

PyTorch 训练/推理慢的可能瓶颈：

- **CPU 算子**：Python 开销、dispatcher、data loader 慢。
- **CUDA kernel**：kernel 本身慢（compute/memory-bound）。
- **launch overhead**：小 kernel 多、CPU 提交跟不上（[[kernel launch overhead]]）。
- **内存**：显存峰值高、分配开销、碎片。
- **通信**：NCCL all-reduce 等待、带宽不足。
- **CPU-GPU 流水线**：data loader 跟不上 GPU（[[CPU-GPU pipeline stall]]）。

不同瓶颈解法完全不同。需先**定位是哪类**，再对症下药。torch profiler 一站式采集所有维度。

### 2.2 trace 可视化的直观性

Chrome JSON trace 在 `chrome://tracing` 或 Perfetto 打开，显示**时间轴**：横轴时间、纵轴线程/stream，每事件一色块。直观看到：

- kernel 间空隙（launch overhead 或 CPU 慢）。
- CPU 算子与 GPU kernel 的对应（dispatch→launch→exec）。
- NCCL 通信的等待（all-reduce 占比）。
- 显存峰值时刻。

这种可视化是 torch profiler 的核心价值——比看数字指标更易定位。

### 2.3 与 PyTorch 的无缝集成

torch profiler 直接包 `model(input)`：

```python
with torch.profiler.profile(...) as prof:
    model(input)
prof.export_chrome_trace('trace.json')
```

无需换框架、不侵入代码。对 PyTorch 用户是最易上手的 profiler。

### 2.4 与 nsys/ncu 的分工

- **torch profiler**：应用层，采集 PyTorch op + CUDA + 内存 + 通信。易上手、可视化。但开销较大、精度不如底层。
- **[[nsys]]**：系统级，采 CUDA runtime/driver + NCCL + CPU 所有线程。更底层、更全、开销小。
- **ncu**：kernel 级，单 kernel 深入（寄存器、occupancy、bandwidth）。最细但慢。

典型流程：torch profiler 先定位瓶颈类型 → 下钻 nsys（系统级细节）或 ncu（单 kernel 优化）。


## 3. 核心概念详解

### 3.1 profile context

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(...),
    on_trace_ready=trace_handler,
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model(input)
```

- `activities`：采集哪类（CPU op、CUDA kernel）。
- `schedule`：何时采（wait、warmup、active、repeat）。
- `record_shapes`：记录 tensor shape（帮定位 shape 相关问题）。
- `profile_memory`：采内存分配/释放。
- `with_stack`：采 Python 调用栈（定位 op 来源）。

### 3.2 Chrome JSON trace

输出 `trace.json`，含事件列表（name、cat、ts、dur、tid）。用 `chrome://tracing` 或 Perfetto 打开可视化。事件类别：

- `cpu_op`：CPU 端 PyTorch op（如 `aten::mm`）。
- `cuda`：CUDA kernel（如 `volta_sgemm`）。
- `cuda_async`：内存拷贝（H2D/D2H）。
- `cpu_runtime`/`cuda_runtime`：runtime 调用。

### 3.3 profiler schedule

长训练不能全程 profile（开销大、trace 巨大）。schedule 控制：

- `wait`：不采（等热身）。
- `warmup`：采但丢弃（JIT 编译、cache 冷启动）。
- `active`：正式采。
- `repeat`：循环。

如 `wait=2, warmup=2, active=6, repeat=1` = 跳 2 步、热身 2 步、采 6 步。采到的 trace 代表稳态性能。

### 3.4 内存分析

`profile_memory=True` 采内存事件：

- `memory_allocated`：每次分配/释放。
- 峰值显存时刻、碎片、分配器开销。

[[GPU memory snapshot]] 是更结构化的内存分析（§31 另一篇）。

### 3.5 关键指标

torch profiler 的 summary 表：

- **Self CPU time**：op 自身耗时（不含子 op）。
- **CPU total**：含子 op。
- **Self CUDA**：kernel 自身。
- **CPU-CUDA 耗时差**：CPU dispatch 与 kernel 执行的对应。

看 Self CUDA 排序找最慢 kernel，看 CPU-CUDA 对应找 launch 间隙。


## 4. 数学原理 / 公式

> [!note] 本条不涉及
> torch profiler 是工具，无数学原理。其指标（耗时、显存）的数学在 [[SM utilization]]/[[memory bandwidth]] 等底层概念。

> [!note] profiler 开销
> profile 开启后吞吐降 10-30%（采事件、写 trace）。需 schedule 限制采集窗口，避免全程 profile 拖慢。生产 profile 只在调优时开。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 profiler + Chrome trace

```python
import json
class TorchProfilerSim:
    """模拟 torch.profiler: 采集 CPU/GPU 事件, 输出 Chrome trace."""
    def __init__(self): self.events=[]; self.t=0
    def start(self): self.events=[]; self.t=0
    def record(self, name, cat, dur, tid):
        self.events.append({'name':name, 'cat':cat, 'dur':dur, 'tid':tid, 'ph':'X', 'ts':self.t})
        self.t += dur
    def stop(self): return self.events
    def export_chrome_trace(self, path):
        with open(path,'w') as f: json.dump(self.events, f)
        return path

p = TorchProfilerSim()
p.start()
p.record('forward', 'cpu_op', 100, 0)         # CPU 端 forward
p.record('attn_kernel', 'cuda', 50, 1)        # CUDA kernel
p.record('launch', 'cpu_op', 5, 0)            # launch dispatch
p.record('mlp_kernel', 'cuda', 30, 1)
path = p.export_chrome_trace('/tmp/trace.json')
print(f'采集 {len(p.events)} 事件, trace: {path}')
for e in p.events:
    print(f"  {e['cat']:8s} {e['name']:15s} {e['dur']:4d}μs tid={e['tid']} ts={e['ts']}")
```

### 5.2 真实 PyTorch API 对照

```python
# import torch
# from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#              schedule=schedule(wait=2, warmup=2, active=6, repeat=1),
#              on_trace_ready=tensorboard_trace_handler('./log'),
#              record_shapes=True, profile_memory=True) as prof:
#     for i, batch in enumerate(loader):
#         model(batch)
#         prof.step()
# # 看 ./log 下的 trace (TensorBoard 或 chrome://tracing)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: PyTorch、Kineto（底层采集引擎）、CUDA runtime、NCCL。
- **下游（应用）**: [[nsys]]（系统级下钻）、[[GPU memory snapshot]]（内存深入）、[[SM utilization]]/[[memory bandwidth]]（kernel 指标解释）、[[kernel launch overhead]]（trace 看 launch 间隙）、[[compute bottleneck]]/[[memory bottleneck]]/[[CPU-GPU pipeline stall]]（瓶颈类型定位）。
- **对比 / 易混**:
  - **torch profiler vs [[nsys]]**：torch profiler 应用层（PyTorch op + CUDA），nsys 系统级（CUDA runtime + NCCL + 全 CPU 线程）。torch profiler 易上手，nsys 更全更底层。
  - **torch profiler vs ncu**：torch profiler 全 trace（多 kernel），ncu 单 kernel 深入（寄存器/occupancy）。
  - **torch profiler vs [[GPU memory snapshot]]**：前者宽（CPU/GPU/内存/通信），后者专内存结构化。


## 7. 常见误区与易错点

> [!warning] 误区 1：profiler 开着跑全程
> profiler 开销 10-30%，全程跑拖慢、trace 巨大难分析。用 schedule 限制采集窗口（wait/warmup/active），只采稳态几步。

> [!warning] 误区 2：只看 Self CUDA 不看 CPU-CUDA 对应
> Self CUDA 找最慢 kernel，但 launch 间隙（CPU 慢、GPU 等）需看 CPU op 与 CUDA kernel 的**时间对应**。trace 上 kernel 间空隙 = launch 瓶颈。

> [!warning] 误区 3：忽略 warmup
> 前几步有 JIT 编译、cache 冷启动，采到的不代表稳态。schedule 的 warmup 丢弃前几步，active 才正式采。

> [!warning] 误区 4：混淆 torch profiler 与 [[nsys]]
> torch profiler 是 PyTorch 应用层（PyTorch op 视角），nsys 是系统级（CUDA runtime/全线程视角）。torch profiler 看不到非 PyTorch 的 CUDA 调用（如自定义 C++ 扩展的底层），nsys 能看全。深度调优用 nsys。

> [!warning] 误区 5：trace 看不懂
> Chrome trace 的色块/类别需学：cpu_op（红）、cuda（蓝）、cuda_async（黄，内存拷贝）、cpu_runtime（灰）。先找最长色块（最慢 op/kernel），再看间隙（launch/等待），再看显存峰值。


## 8. 延伸细节

### 8.1 Kineto

torch profiler 底层是 **Kineto**（Facebook 开发的采集库），对接 CUDA activities、CPU stack sampling。Kineto 也可被 nsys 间接复用。理解 Kineto 有助于解释 torch profiler 的采集能力边界。

### 8.2 Perfetto

Chrome trace 格式（JSON）正被 **Perfetto**（Google 开源）替代/增强。Perfetto UI 更强（大 trace 不卡、SQL 查询）。torch profiler 新版支持 Perfetto 格式。

### 8.3 TensorBoard 集成

`tensorboard_trace_handler` 把 trace 写 TensorBoard，可在 TensorBoard UI 看trace + 内存曲线。生产调优常 TensorBoard + chrome trace 结合。

### 8.4 LLM 训练/推理的 profile

- 训练：profile 一个 step，看 forward/backward/all-reduce 占比，定位 [[compute bottleneck]]/[[communication bottleneck]]。
- 推理：profile 一个 token 生成，看 kernel launch 间隙（[[kernel launch overhead]]）、attention kernel 占比、显存峰值（KV cache）。

### 8.5 with_stack 与调用栈

`with_stack=True` 采 Python 调用栈，trace 里点击 op 能看是哪行代码调的。定位"哪个 op 慢"后查"哪行代码"必开。但 stack 采集有额外开销。

### 8.6 profiler 的局限

- **采样开销**：profile 时行为可能与非 profile 时不同（heisenberg）。
- **精度**：μs 级精度不如 nsys/ncu。
- **非 PyTorch 部分**：自定义 C++ 扩展、第三方库的 CUDA 调用可能采不全。

深度调优先 torch profiler 定位，再 nsys/ncu 下钻。

---
相关: [[profiling工具]] | [[nsys]] | [[GPU memory snapshot]] | [[SM utilization]] | [[memory bandwidth]] | [[kernel launch overhead]] | [[compute bottleneck]] | [[memory bottleneck]] | [[communication bottleneck]] | [[CPU-GPU pipeline stall]]
