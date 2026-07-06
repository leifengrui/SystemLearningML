# nsys (Nsight Systems)

> **所属章节**: [[profiling工具]]
> **所属模块**: [[10-性能分析与优化]]
> **别名**: nsys / Nsight Systems / nsight-sys / nsys profile
> **难度**: 中高（需懂系统级 profiling + CUDA）


## 1. 一句话定义

**nsys（Nsight Systems）** 是 NVIDIA 的**系统级 profiler**，采集**全系统**事件——CPU 所有线程的调用栈、CUDA runtime/driver 调用、kernel launch 与执行、NCCL 通信、内存拷贝（H2D/D2H）、OS 调度、GPU 引擎状态——生成 trace 供 Nsight Systems UI 或 Perfetto 可视化。与 [[torch profiler]]（应用层、PyTorch 视角）不同，nsys 是**系统层**：看非 PyTorch 的 CUDA 调用、OS 调度、NCCL 底层、CPU 多线程协同。与 ncu（Nsight Compute，kernel 级）互补：nsys 是**宏观系统 trace**（哪里慢、谁等谁），ncu 是**微观单 kernel**（寄存器/occupancy）。三者层次：torch profiler（应用）→ nsys（系统）→ ncu（kernel）。

> [!note] 三句话定位
> - **是什么**：NVIDIA 系统级 profiler，采全系统（CPU/CUDA/NCCL/OS），trace 可视化定位系统瓶颈。
> - **为什么**：torch profiler 看不到非 PyTorch 的 CUDA 调用、OS 调度、NCCL 底层；nsys 系统层补全。
> - **与 torch profiler/ncu 的关系**：torch profiler（应用层）→ nsys（系统层，定位谁慢谁等）→ ncu（kernel 层，单 kernel 深入）。


## 2. 为什么需要它（动机与背景）

### 2.1 系统级瓶颈的不可见性

[[torch profiler]] 是 PyTorch 应用层，能看到 PyTorch op + CUDA kernel。但看不到：

- **非 PyTorch 的 CUDA 调用**：自定义 C++ 扩展、第三方库（如 cuBLAS 内部）、CUDA driver 层。
- **OS 调度**：CPU 线程被 OS 调度切换、NUMA、锁等待。
- **NCCL 底层**：all-reduce 的 CUDA kernel 细节、拓扑、ring/tree 算法。
- **CPU 多线程协同**：data loader worker 与主线程、inference server 多请求线程的协同。

这些系统级瓶颈需系统级工具。nsys 采全系统，补 torch profiler 的盲区。

### 2.2 trace 的系统视角

nsys trace 显示**多进程/多线程/多 GPU 引擎**的时间轴：

- CPU 线程（Python 主线程、data loader workers、NCCL 线程）。
- CUDA streams（kernel、memcpy）。
- GPU 引擎（compute、copy engine 状态）。
- NCCL 通信。

直观看到**谁在等谁**：data loader 等 IO → CPU 主线程等 data → GPU 等 CPU 提交 → NCCL 等其他 rank。这种系统级因果链是 nsys 的核心价值。

### 2.3 分布式训练的 NCCL 分析

多 GPU 训练的 all-reduce 是常见瓶颈。nsys 采 NCCL 的底层 CUDA kernel（`ncclKernel*`）、ring/tree 拓扑、通信时间。能定位：

- all-reduce 占比（通信 vs 计算）。
- 是否与计算 overlap（[[overlap strategy]]）。
- 某个 rank 慢导致全局等（straggler）。

torch profiler 看 NCCL 粗粒度，nsys 看底层细节。

### 2.4 低开销

nsys 的开销比 torch profiler 小（~5% vs 10-30%），因采集系统事件（CUDA runtime hook）比 PyTorch op 采集轻。可较长时间 profile 稳态。


## 3. 核心概念详解

### 3.1 命令行采集

```
nsys profile -t cuda,osrt,nvtx,nccl -o trace \
    --stats=true \
    python train.py
```

- `-t`：采集哪类（cuda=runtime/kernel, osrt=OS 线程, nvtx=用户标记, nccl=通信）。
- `-o`：输出 qdrep（Nsight Systems UI 格式）或 sqlite。
- `--stats=true`：生成统计报告（top kernel、CPU 占用等）。

### 3.2 Nsight Systems UI

打开 `.qdrep`/`.nsys-rep`，UI 显示：

- **时间轴**：CPU 线程 + CUDA stream + GPU 引擎 + NCCL，色块表示事件。
- **事件表**：每事件 name、dur、start、tid。
- **统计**：top kernel by time、CPU thread 占用、NCCL 通信汇总。

### 3.3 关键视角

| 视角 | 内容 | 定位 |
|---|---|---|
| CPU 线程 | 各线程调用栈、占用 | data loader 慢、Python 开销、锁等待 |
| CUDA stream | kernel、memcpy、launch 间隙 | [[kernel launch overhead]]、H2D/D2H 拷贝 |
| GPU 引擎 | compute/copy engine 状态 | GPU 空闲时刻、计算-拷贝 overlap |
| NCCL | all-reduce kernel、通信时间 | [[communication bottleneck]]、overlap 效果 |

### 3.4 NVTX 标记

NVTX（NVIDIA Tools Extension）让用户在代码里**手动标记区间**：

```python
torch.cuda.nvtx.range_push('forward')
model(input)
torch.cuda.nvtx.range_pop()
```

nsys trace 里该区段显示为 'forward' 标记，便于在大量事件中定位"这段是 forward"。是定位 LLM 某阶段（prefill/decode/all-reduce）的常用手段。

### 3.5 与 ncu 的关系

nsys 定位"哪个 kernel 慢"后，用 **ncu**（Nsight Compute）深入该 kernel：

```
ncu --kernel-name regex:slow_kernel --set full python train.py
```

ncu 报寄存器、occupancy、bandwidth、roofline。nsys 是"找哪个慢"，ncu 是"为什么慢"。


## 4. 数学原理 / 公式

> [!note] 本条不涉及
> nsys 是工具，无数学原理。其 trace 的指标解释在 [[SM utilization]]/[[memory bandwidth]]/[[kernel launch overhead]] 等。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 nsys trace（多进程多引擎）

```python
import json
class NsysTraceSim:
    """模拟 nsys trace: 全系统事件 (CPU 线程 + CUDA stream + NCCL)."""
    def __init__(self): self.events=[]
    def add(self, name, cat, dur, tid, pid):
        self.events.append({'name':name,'cat':cat,'dur':dur,'tid':tid,'pid':pid})
    def export(self, path):
        with open(path,'w') as f: json.dump(self.events, f)
        return path

t = NsysTraceSim()
# pid=0 CPU 进程: dataloader worker + 主线程
t.add('DataLoader', 'cpu', 200, 0, 0)         # tid=0 dataloader
t.add('forward', 'cpu', 100, 1, 0)            # tid=1 主线程
# pid=1 GPU: compute stream + copy engine
t.add('mm_kernel', 'cuda', 50, 2, 1)          # tid=2 compute stream
t.add('memcpy_h2d', 'cuda_async', 10, 3, 1)   # tid=3 copy engine
# pid=2 NCCL
t.add('all_reduce', 'nccl', 30, 4, 2)         # tid=4 nccl
path = t.export('/tmp/nsys_trace.json')
print(f'nsys trace {len(t.events)} 事件 (跨 CPU/CUDA/NCCL):')
for e in t.events:
    print(f"  pid={e['pid']} tid={e['tid']} {e['cat']:11s} {e['name']:13s} {e['dur']}μs")
print(f'trace: {path}')
```

### 5.2 真实 nsys 命令对照

```bash
# 采集训练 step
nsys profile -t cuda,nvtx,osrt,nccl \
    --stats=true \
    -o train_trace \
    python train.py
# 用 Nsight Systems UI 打开 train_trace.qdrep 看 timeline
# 下钻慢 kernel:
# ncu --kernel-name regex:mm_kernel --set full python train.py
```


## 6. 与其他知识点的关系

- **上游（依赖）**: CUDA runtime/driver、NCCL、OS 调度、NVTX。
- **下游（应用）**: [[torch profiler]]（应用层互补）、[[compute bottleneck]]/[[memory bottleneck]]/[[communication bottleneck]]/[[CPU-GPU pipeline stall]]（系统瓶颈定位）、[[kernel launch overhead]]（trace 看 launch 间隙）、[[overlap strategy]]（看计算-通信 overlap）、[[NCCL通信拓扑]]（NCCL 底层）、[[SM utilization]]/[[memory bandwidth]]（kernel 指标，配合 ncu）。
- **对比 / 易混**:
  - **nsys vs [[torch profiler]]**：nsys 系统级（全 CPU/CUDA/NCCL/OS），torch profiler 应用层（PyTorch op）。nsys 更全更底层，torch profiler 易上手。
  - **nsys vs ncu**：nsys 系统宏观 trace（哪里慢、谁等谁），ncu 单 kernel 微观（为什么慢）。先 nsys 定位再 ncu 深入。
  - **nsys vs [[GPU memory snapshot]]**：nsys 采 trace（时间维事件），memory snapshot 采显存结构（空间维快照）。


## 7. 常见误区与易错点

> [!warning] 误区 1：混淆 nsys 与 ncu
> nsys（Nsight Systems）是系统级 trace（宏观、多 kernel、谁等谁）；ncu（Nsight Compute）是单 kernel 深入（微观、寄存器/occupancy）。名字像但层次不同。先 nsys 定位"哪个慢"，再 ncu 查"为什么慢"。

> [!warning] 误区 2：只用 nsys 不用 torch profiler
> nsys 看系统，但 PyTorch op 的语义（哪个 op 对应哪个代码行）需 torch profiler 的 with_stack。两者互补：torch profiler 定位 PyTorch op，nsys 定位系统层。深度调优两者都用。

> [!warning] 误区 3：不开 NVTX 标记
> nsys trace 事件密集，难定位"这段是 forward 还是 backward"。用 NVTX range 标记关键区段（forward、backward、all-reduce、data load），trace 里一目了然。

> [!warning] 误区 4：忽略 NCCL 视角
> 分布式训练的 all-reduce 常是瓶颈。nsys 的 NCCL 视角看 all-reduce kernel、通信时间、与计算的 overlap。不看 NCCL 等于漏掉分布式训练最大瓶颈点。

> [!warning] 误区 5：trace 太大难分析
> 长训练的 trace 巨大（GB 级），UI 卡。用 `-s cpu-process`/`-y` 限定采集范围、`--delay`/`--duration` 限窗口、NVTX 标记聚焦区段。只采有问题的几秒。


## 8. 延伸细节

### 8.1 Nsight Systems vs Compute

- **Nsight Systems（nsys）**：系统级 trace，宏观、多 kernel、低开销。
- **Nsight Compute（ncu）**：kernel 级 profile，微观、单 kernel、高开销（慢 10-100×）。

两者配合：nsys 找慢 kernel，ncu 深入。

### 8.2 分布式训练的 NCCL 分析

nsys 看 all-reduce：

- 通信时间占比（vs 计算时间）。
- 是否与计算 overlap（[[overlap strategy]] 是否生效）。
- 某 rank 慢导致其他 rank 等（straggler effect）。
- ring vs tree 拓扑的效率（[[NCCL通信拓扑]]）。

是分布式训练调优的核心。

### 8.3 CPU 瓶颈定位

LLM 推理的 Python 端开销（dispatcher、autograd、调度）可能占 ms 级/token。nsys 的 CPU 线程视角看 Python 主线程占用、是否被 OS 调度打断、是否等 data loader。[[kernel launch overhead]] 的 CPU-bound 根因在 nsys 直观可见。

### 8.4 GPU 引擎视角

GPU 有 compute engine + copy engine。nsys 看两者是否并行（计算时同时拷贝，overlap）。若 compute 和 copy 串行（不 overlap），带宽和算力不能复用，是优化点。

### 8.5 与 LLM 推理框架的集成

vLLM/SGLang 内部用 nsys profile 优化：

- prefill/decode 的 kernel 分布。
- attention kernel 占比、KV cache 拷贝。
- 多请求的调度（continuous batching 的请求交织）。

nsys 是这些框架的性能验证工具。

### 8.6 nsys 的报告与统计

`--stats=true` 生成文本统计（不需 UI）：

- Top kernel by time（最慢 kernel 排行）。
- CPU thread 占用（哪个线程忙）。
- NCCL 通信汇总（次数、时间）。

无 UI 环境也能用统计报告快速定位。

---
相关: [[profiling工具]] | [[torch profiler]] | [[GPU memory snapshot]] | [[compute bottleneck]] | [[memory bottleneck]] | [[communication bottleneck]] | [[CPU-GPU pipeline stall]] | [[kernel launch overhead]] | [[overlap strategy]] | [[NCCL通信拓扑]] | [[SM utilization]] | [[memory bandwidth]]
