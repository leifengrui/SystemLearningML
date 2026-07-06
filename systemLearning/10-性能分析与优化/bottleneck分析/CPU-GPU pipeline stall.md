# CPU-GPU pipeline stall

> **所属章节**: [[bottleneck分析]]
> **所属模块**: [[10-性能分析与优化]]
> **别名**: CPU-GPU pipeline stall / host bound / launch bound / CPU 瓶颈 / GPU 饥饿
> **难度**: 中（需懂 [[kernel launch overhead]] + CPU/GPU 异步执行模型）


## 1. 一句话定义

**CPU-GPU pipeline stall（主机端瓶颈）** 是 GPU **空闲等待 CPU 喂数据**的状态——CPU 端（kernel 启动、数据加载、Python 开销、host-device 同步）跟不上 GPU 消费速度，导致 GPU 在 kernel 间出现 **idle gap（空闲间隙）**，throughput 受 CPU 处理速度限而非 GPU 算力/带宽/通信限。它是四类瓶颈中唯一**瓶颈在 CPU 不在 GPU** 的类型。识别 host-bound 的意义：**优化方向是减 CPU 工作/异步化**（dataloader workers、CUDA Graph、消除同步点、降低 Python 开销），而非加 GPU 算力/带宽。典型场景：训练 dataloader 跟不上、LLM 推理调度/tokenization 占主导、小 kernel 高频启动的 launch bound。与 [[compute bottleneck]]/[[memory bottleneck]]/[[communication bottleneck]] 是四类互斥瓶颈。

> [!note] 三句话定位
> - **是什么**：CPU 喂不快，GPU idle 等待，throughput 受 CPU 限。
> - **为什么识别**：host-bound 优化是减 CPU 工作/异步化（dataloader/CUDA Graph/消除同步），加 GPU 无用。
> - **典型场景**：训练 dataloader 慢、LLM 推理调度主导、小 kernel 高频启动（launch bound）。


## 2. 为什么需要它（动机与背景）

### 2.1 CPU-GPU 异步执行模型

CPU 启动 kernel（enqueue 到 CUDA stream）后**立即返回**（不等 GPU 执行完），GPU 异步执行。理想情况下 CPU 启动 $k$ 个 kernel 的时间里 GPU 执行完 $k$ 个，流水线满。但若 CPU 启动一个 kernel 要 10μs 而 GPU 执行只要 5μs，GPU 每个 kernel 后 idle 5μs 等 CPU 启动下一个——**launch bound**（启动限）。同理 dataloader 慢、Python 慢、同步点都会让 CPU 跟不上，GPU 饥饿。

### 2.2 GPU 空闲 = 烧钱

GPU 贵（A100 ~$10k，H100 ~$30k），idle GPU = 浪费钱。host-bound 时 GPU 利用率（时间维）低（idle gap 多），但根因在 CPU，加 GPU 算力/带宽无用——GPU 没活干不是因为干不动，是 CPU 没喂饱。识别 host-bound 后优化 CPU 喂养速度，让 GPU 满载。

### 2.3 host-bound 的来源

| 来源 | 机制 | 典型 |
|---|---|---|
| **launch overhead** | 每 kernel 启动 5-10μs CPU 开销 | 小 kernel 高频（elementwise 链） |
| **dataloader 慢** | CPU 数据预处理跟不上 GPU 消费 | 图像 augmentation、tokenization |
| **Python 开销** | GIL 单线程 + 动态解释慢 | 训练循环、控制流 |
| **同步点** | `.item()`/`.cpu()`/`synchronize()` 强制 CPU 等 GPU | 训练 metric 记录、调试打印 |
| **host-device 传输** | CPU→GPU 拷贝阻塞 | 未 pinned memory、未异步 |

### 2.4 优化方向

host-bound 的 throughput 受 CPU 限，优化减 CPU 工作/异步化：

- **dataloader workers**：多进程数据加载（`num_workers>0`），并行预处理。
- **CUDA Graph**：捕获 kernel 序列，一次启动回放，消除 per-kernel CPU 开销（[[kernel launch overhead]] 的根治）。
- **kernel fusion**：多 kernel 合一，减启动次数（同时优化 launch + memory）。
- **消除同步点**：移除 `.item()`/`.cpu()`/`synchronize()`，用异步 metric 记录。
- **pinned memory**：`pin_memory=True` 加速 host-device 传输。
- **减 Python 开销**：把循环逻辑移到 C++/CUDA（自定义 op、`torch.compile`）。
- **预取（prefetch）**：`cudaStream` 异步传输，与计算重叠。

### 2.5 加 GPU 算力无用

host-bound 的 GPU 没活干（idle），加算力/带宽无用——GPU 不缺算力或带宽，缺的是 CPU 喂的活。优化 CPU 喂养速度，让 GPU 满载。

### 2.6 LLM 的 host-bound 场景

- **训练 dataloader**：tokenization、数据采样 CPU 慢，大 batch 时 dataloader 跟不上。`num_workers` + 预 tokenize。
- **推理调度**：LLM 推理的请求调度、tokenization、KV cache 管理在 CPU，小 batch + 短序列时 CPU 开销占比高（GPU 等 CPU 调度）。[[continuous batching]] 的调度器优化是关键。
- **小 kernel 高频**：LLM 的 layer norm、activation、residual 等小 kernel 链，每 kernel 5-10μs launch，累积成 host-bound。kernel Fusion + CUDA Graph 优化。


## 3. 核心概念详解

### 3.1 launch bound

每 kernel 启动有 CPU 开销（$t_{\text{launch}} \approx 5$-10μs）。若 GPU 执行时间 $t_{\text{exec}} < t_{\text{launch}}$，GPU idle 等 CPU 启动下一个——launch bound。一串 $N$ 个小 kernel：理想 $T = N \cdot t_{\text{exec}}$，实际 $T \approx N \cdot t_{\text{launch}}$（CPU 成主导）。详见 [[kernel launch overhead]]。

### 3.2 dataloader 瓶颈

训练每步 GPU 消费一个 batch 的数据。若 CPU dataloader 产出一个 batch 要 $T_{\text{load}}$，GPU 消费要 $T_{\text{compute}}$：

- $T_{\text{load}} < T_{\text{compute}}$：dataloader 跟得上，GPU 满载。
- $T_{\text{load}} > T_{\text{compute}}$：dataloader 跟不上，GPU 等 data，host-bound。

`num_workers>0` 多进程并行产 data，让 $T_{\text{load}}$ 降到 $< T_{\text{compute}}$。但 workers 太多又抢 CPU 资源，需调。

### 3.3 同步点

CPU-GPU 异步模型下，CPU 启动 kernel 后不等 GPU。但某些操作强制 CPU 等 GPU（同步）：

- `.item()` / `.cpu()`：取 GPU 值回 CPU，等 GPU 算完。
- `torch.cuda.synchronize()`：等所有 GPU kernel 完成。
- 条件控制流依赖 GPU 值：`if loss.item() < threshold: break`。

同步点打破异步流水线，CPU 阻塞等 GPU，GPU 后续 kernel 也延迟启动。消除同步点（用异步 metric、延迟记录）是 host-bound 优化。

### 3.4 GPU 利用率与 host-bound

[[GPU utilization]]（时间维）= GPU 忙时间 / 总时间。host-bound 时 GPU idle gap 多，利用率低（如 30%）。但低利用率不一定是 host-bound——也可能是 comm-bound（通信时 GPU 也 idle 等）或 memory-bound 的访问模式差。需 nsys 看时间轴：host-bound 的 idle gap 在 kernel 间且 CPU 线程忙；comm-bound 的 idle 伴随 NCCL kernel；memory-bound 的 GPU 满载但带宽利用率低。

### 3.5 与其他三类瓶颈的对比

| 维度 | CPU-GPU pipeline stall | [[compute bottleneck]] | [[memory bottleneck]] | [[communication bottleneck]] |
|---|---|---|---|---|
| 瓶颈源 | CPU 喂不快 | GPU 算力 | GPU HBM 带宽 | GPU 间带宽 |
| GPU 状态 | idle 等待 | 满载算 | 满载等数据 | idle/NCCL |
| 优化方向 | 减 CPU/异步化 | tensor core | 减数据量 | 减通信/overlap |
| 加 GPU 算力 | 无用 | 有效 | 无用 | 无用 |
| 典型 | dataloader 慢 | 训练 matmul | decode | 多卡 allreduce |


## 4. 数学原理 / 公式

### 4.1 launch bound 条件

$$
t_{\text{exec}} < t_{\text{launch}} \implies \text{launch bound}
$$

$N$ 个小 kernel 的总时间：

$$
T = N \cdot \max(t_{\text{exec}}, t_{\text{launch}}) \approx N \cdot t_{\text{launch}} \quad (\text{launch bound})
$$

### 4.2 dataloader 瓶颈

$$
T_{\text{step}} = \max(T_{\text{load}}, T_{\text{compute}})
$$

$T_{\text{load}} > T_{\text{compute}}$ 时 host-bound，step 受 dataloader 限。

### 4.3 GPU 利用率

$$
U_{\text{gpu}} = \frac{T_{\text{busy}}}{T_{\text{total}}} = \frac{T_{\text{compute}}}{T_{\text{step}}}
$$

host-bound 时 $T_{\text{step}} \approx T_{\text{load}}$（或 $N \cdot t_{\text{launch}}$），$U_{\text{gpu}} = T_{\text{compute}}/T_{\text{step}}$ 低。

### 4.4 CUDA Graph 的加速

无 CUDA Graph，$N$ 个 kernel：$T = N \cdot t_{\text{launch}}$（launch bound）。
有 CUDA Graph：捕获一次，回放一次启动 $N$ 个，$T \approx t_{\text{launch}}$（单次启动开销）。

$$
\text{加速比} = N \quad (\text{launch bound 时})
$$

### 4.5 overlap 吞吐

理想 CPU/GPU 流水线（CPU 启动 $k+1$ 时 GPU 执行 $k$）：

$$
T_{\text{step}} = \max(T_{\text{cpu}}, T_{\text{gpu}})
$$

$T_{\text{cpu}} < T_{\text{gpu}}$ 时 GPU 满载；$T_{\text{cpu}} > T_{\text{gpu}}$ 时 host-bound（CPU 限）。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 launch bound + CUDA Graph

```python
def launch_bound_time(n_kernels, t_exec_us=2, t_launch_us=7):
    """N 个小 kernel, launch bound 时 CPU 启动主导."""
    per_kernel = max(t_exec_us, t_launch_us)
    return n_kernels * per_kernel

def cuda_graph_time(n_kernels, t_exec_us=2, t_launch_us=7):
    """CUDA Graph: 一次启动回放 N kernel."""
    return n_kernels * t_exec_us + t_launch_us  # 单次启动 + N 次执行

# 100 个小 kernel (elementwise 链)
n = 100
t_lb = launch_bound_time(n)
t_cg = cuda_graph_time(n)
print(f'无 CUDA Graph: {t_lb}μs (launch bound, CPU 主导)')
print(f'有 CUDA Graph: {t_cg}μs (GPU 执行主导, {t_lb/t_cg:.1f}x 加速)')
```

### 5.2 dataloader 瓶颈模拟

```python
def step_time(t_load, t_compute):
    """step 时间 = max(load, compute), host-bound when load > compute."""
    return max(t_load, t_compute), 'host-bound' if t_load > t_compute else 'gpu-bound'

# dataloader 慢 (单进程 tokenization)
t, bound = step_time(0.050, 0.020)  # load 50ms, compute 20ms
print(f'单进程 dataloader: {t*1e3:.1f}ms/step, {bound}')

# 加 8 workers
t, bound = step_time(0.050/8, 0.020)  # load 6.25ms, compute 20ms
print(f'8 workers:        {t*1e3:.1f}ms/step, {bound}')
```

### 5.3 真实测量对照

```python
# # nsys 看时间轴: kernel 间 idle gap = host-bound 证据
# # nsys profile -t cuda,nvtx python train.py
# # nsys stats 报告: CUDA kernel 时间轴, gap 间 CPU 线程忙 = launch bound
# # GPU idle + CPU 忙 = host-bound; GPU idle + NCCL = comm-bound
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[kernel launch overhead]]（launch bound 的物理基础）、CPU-GPU 异步执行模型、Python GIL。
- **下游（应用）**: [[compute bottleneck]]/[[memory bottleneck]]/[[communication bottleneck]]（其他瓶颈）、[[torch profiler]]/[[nsys]]（测 idle gap）、CUDA Graph/[[torch.compile]]（减 CPU 开销）、dataloader workers/pinned memory（数据加载优化）、LLM 推理调度器（[[continuous batching]] 的 CPU 侧）、[[GPU utilization]]（host-bound 的指标）。
- **对比 / 易混**:
  - **host-bound vs [[compute bottleneck]]**：CPU 喂不快 GPU idle vs GPU 算力满。GPU 利用率低（idle）vs 高（满载）。加 GPU 算力对前者无用、对后者有效。
  - **host-bound vs [[communication bottleneck]]**：CPU 喂不快 vs 通信带宽满。都让 GPU idle，但前者 idle 时 CPU 忙、无 NCCL；后者 idle 时有 NCCL kernel 传输。nsys 时间轴区分。
  - **host-bound vs GPU idle（未调度）**：host-bound 是 CPU 喂不快（有活但启动慢）；未调度是没活（请求不足，LLM 推理空闲期）。前者优化 CPU，后者需增请求/调度。


## 7. 常见误区与易错点

> [!warning] 误区 1：GPU idle 是 GPU 问题
> GPU idle 的根因常在 CPU（喂不快），不是 GPU 本身。加 GPU 算力/带宽无用——GPU 没活干不是因为干不动。先 nsys 看 idle gap 时 CPU 忙不忙：CPU 忙 = host-bound（优化 CPU）；CPU 闲 = 无请求（增负载）。

> [!warning] 误区 2：dataloader 总是瓶颈
> dataloader 瓶颈取决于 $T_{\text{load}}$ vs $T_{\text{compute}}$。大模型（compute 长）时 dataloader 常不是瓶颈；小模型（compute 短）+ 重 augmentation 时才是。用 `num_workers` 让 $T_{\text{load}} < T_{\text{compute}}$ 即可，过多 workers 抢 CPU。

> [!warning] 误区 3：忽略同步点
> 一个 `.item()` 或 `print(loss)` 会让 CPU 阻塞等 GPU，打破异步流水线，GPU 后续 kernel 延迟启动。训练循环里的 metric 记录、early stopping 判断都可能是同步点。用异步记录（`loss.detach()` 累积，周期性同步）消除。

> [!warning] 误区 4：CUDA Graph 万能
> CUDA Graph 捕获静态 kernel 序列回放，适合 kernel 固定的场景（LLM 推理、训练 step）。动态控制流（if/while 依赖 tensor 值）不能进 Graph。且 Graph 捕获有一次性开销，短生命周期场景收益小。

> [!warning] 误区 5：混淆 host-bound 与 memory-bound
> 都让 GPU"看起来不满"，但 host-bound 是 GPU idle（没 kernel），memory-bound 是 GPU 满载（kernel 在执行等数据）。nsys 时间轴：host-bound 有 kernel 间 gap，memory-bound kernel 连续无 gap 但带宽利用率低。

> [!warning] 误区 6：LLM 推理无 host-bound
> LLM 推理的请求调度、tokenization、KV cache 管理在 CPU。小 batch + 短序列时 CPU 开销占比高（GPU 等 CPU 调度），是 host-bound。大 batch + 长序列时 compute/memory 主导。调度器优化（[[continuous batching]]）是推理 host-bound 的关键。


## 8. 延伸细节

### 8.1 CUDA Graph 的适用

CUDA Graph 适合：

- **LLM 推理**：每 token 的 kernel 序列固定（前向 N 层），Graph 捕获后回放，消除 per-kernel launch。vLLM/TGI 用 Graph 提小 batch 推理 throughput。
- **训练 step**：前向 + 反向 kernel 序列固定（无动态控制流时），Graph 捕获。但 optimizer step、dataloader 每步不同，需部分 Graph。

不适合：

- 动态 shape（每步 tensor shape 变，Graph 需重新捕获）。
- 动态控制流（if 依赖 tensor 值）。

### 8.2 torch.compile 的作用

[[torch.compile]] 把 Python 训练循环编译成优化图，减 Python 开销 + fusion kernel。是 host-bound 的现代优化（替代手写 CUDA Graph）。对动态控制流支持有限。

### 8.3 pinned memory

CPU→GPU 传输默认可分页内存，走 DMA 慢。`pin_memory=True` 锁页内存，DMA 直传，带宽提 2-3×。`DataLoader(pin_memory=True)` + `to(device, non_blocking=True)` 异步传输，与计算 overlap。是 dataloader 优化基础。

### 8.4 多进程 dataloader 的权衡

`num_workers=k` 开 k 进程并行产 data。k 太少跟不上（host-bound），太多抢 CPU（context switch 开销）+ 显存占用（每 worker 有 buffer）。经验值：4-8，按 CPU 核数和数据复杂度调。`persistent_workers=True` 避免每 epoch 重启 worker。

### 8.5 LLM 推理调度器的 host-bound

[[continuous batching]] 的调度器在 CPU：每步选请求、组 batch、分配 KV cache block。小 batch + 短序列时调度开销占比高（GPU 等 CPU 调度）。优化：调度算法高效（vLLM 的 PagedAttention 调度）、预测执行（提前调度下批）、CUDA Graph（捕获固定 kernel 序列减 CPU 启动）。

### 8.6 nsys 诊断 host-bound

nsys 时间轴看：

- CUDA kernel 时间段连续无 gap → GPU 满载（非 host-bound）。
- kernel 间有 idle gap，gap 时 CPU 线程忙 → launch bound（host-bound）。
- kernel 间 idle gap，gap 时 NCCL kernel → comm-bound。
- GPU 全程 idle → 无请求或严重 host-bound。

CPU 线程的函数栈（Python/C++）定位是 dataloader/sync/Python 哪个慢。

### 8.7 四类瓶颈的统一诊断

| 症状 | GPU 利用率 | 带宽利用率 | NCCL | idle gap 时 CPU | 判定 |
|---|---|---|---|---|---|
| 算力满 | 高 | - | - | - | [[compute bottleneck]] |
| 带宽满 | 高 | 高 | - | - | [[memory bottleneck]] |
| 通信满 | 低 | - | 高 | - | [[communication bottleneck]] |
| CPU 喂不快 | 低 | - | - | 忙 | **host-bound** |

nsys + ncu 综合诊断四类瓶颈。

---
相关: [[bottleneck分析]] | [[kernel launch overhead]] | [[compute bottleneck]] | [[memory bottleneck]] | [[communication bottleneck]] | [[torch profiler]] | [[nsys]] | [[GPU utilization]] | [[continuous batching]] | [[torch.compile]] | [[KV cache management]] | [[SM utilization]] | [[overlap strategy]]
