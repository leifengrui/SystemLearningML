# nvprof（旧）

> **所属章节**: [[profiling工具]]
> **所属模块**: [[10-性能分析与优化]]
> **别名**: nvprof / NVIDIA Visual Profiler / nvvp
> **难度**: 低（了解即可，已被 [[nsys]]/ncu 取代）


## 1. 一句话定义

**nvprof** 是 NVIDIA 的**旧版命令行 CUDA profiler**（2012-2020 主力），采集 CUDA kernel 时间、API 调用、部分指标，输出文本表格或 nvvp（GUI）可视化。它是 [[nsys]]（Nsight Systems）和 ncu（Nsight Compute）的**前身**——nsys 取代其系统级 trace 功能、ncu 取代其 kernel 指标功能。自 CUDA 11 起 nvprof **deprecated**（废弃），CUDA 12+ 不再支持新 GPU（如 H100）。新项目**不应使用 nvprof**，应迁移到 nsys/ncu。但旧代码、老 GPU（V100 及以前）、旧文档仍可能提到，需了解其历史地位与迁移路径。

> [!note] 三句话定位
> - **是什么**：NVIDIA 旧版 CUDA profiler，nsys/ncu 的前身，已 deprecated。
> - **为什么了解**：旧代码/老 GPU/旧文档仍用；新项目应迁移到 nsys/ncu。
> - **被取代的原因**：nsys 系统级更全、ncu kernel 级更深、nvprof 不支持新 GPU（Hopper+）。


## 2. 为什么需要它（动机与背景）

### 2.1 历史地位

2012-2020，nvprof 是 CUDA profiling 的主力工具：

- 命令行 `nvprof --print-gpu-trace` 看 kernel 时间。
- GUI（nvvp，NVIDIA Visual Profiler）可视化 timeline。
- 采集 kernel 指标（occupancy、bandwidth、缓存命中等）。

那时没有 nsys/ncu 的分离，nvprof 一个工具干所有。大量旧教程、论文、博客用 nvprof。

### 2.2 被 nsys/ncu 取代

NVIDIA 在 2018 左右推出 **Nsight Systems（nsys）** 和 **Nsight Compute（ncu）**，分工：

- **nsys** 专系统级 trace（更全、更准、更低开销）。
- **ncu** 专 kernel 级指标（更细、更多指标、roofline）。

nvprof 的功能被两者瓜分且超越。加上 nvprof 架构老、不支持新 GPU（Hopper 架构的 PMU/SMU 寄存器变化），CUDA 11 起 deprecated，CUDA 12+ 完全不支持 H100。

### 2.3 何时还会遇到

- **旧代码库**：老项目的脚本/文档用 nvprof。
- **老 GPU**：V100/T4/P100 等仍可用 nvprof（虽 deprecated）。
- **旧教程/论文**：2018 前的资料用 nvprof。
- **维护遗留系统**：不能升级 CUDA 的旧环境。

新项目不应使用，但需能读懂旧资料并迁移。

### 2.4 迁移路径

| nvprof 功能 | 迁移到 |
|---|---|
| `--print-gpu-trace`（kernel 时间） | `nsys profile --stats=true` |
| `--print-api-trace`（CUDA API） | `nsys profile -t cuda` |
| `--metrics`（kernel 指标） | `ncu --metrics` |
| GUI timeline（nvvp） | Nsight Systems UI（nsys） |
| roofline | `ncu --set roofline` |


## 3. 核心概念详解

### 3.1 命令行用法

```bash
# kernel 时间 (GPU trace)
nvprof --print-gpu-trace python train.py

# CUDA API 调用
nvprof --print-api-trace python train.py

# kernel 指标 (occupancy/bandwidth)
nvprof --metrics achieved_occupancy,dram_read_throughput python train.py

# GUI
nvprof python train.py   # 生成 .nvprof, 用 nvvp 打开
```

### 3.2 输出格式

nvprof 输出**文本表格**（区别于 nsys 的 trace timeline）：

```
==50596== GPU activities:
  50.12ms  100  50.1us  mm_kernel
  30.05ms  200  30.0us  attn_kernel
```

简洁但无时间轴可视化（GUI nvvp 才有 timeline）。

### 3.3 nvvp（NVIDIA Visual Profiler）

nvprof 的 GUI，打开 `.nvprof` 文件，显示 timeline + 指标。是 Nsight Systems UI 的前身，功能较弱。

### 3.4 与 nsys/ncu 的对比

| 维度 | nvprof | nsys | ncu |
|---|---|---|---|
| 层次 | 混合（系统+kernel） | 系统级 | kernel 级 |
| 输出 | 文本表格 + nvvp | qdrep timeline | 详细报告 |
| 新 GPU | ❌（H100 不支持） | ✓ | ✓ |
| 状态 | deprecated | 主力 | 主力 |
| 开销 | 中 | 低 | 高（慢 10-100×） |


## 4. 数学原理 / 公式

> [!note] 本条不涉及
> nvprof 是工具，无数学原理。其指标的解释在 [[SM utilization]]/[[memory bandwidth]]。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 nvprof 文本输出

```python
def nvprof_gpu_trace(kernels):
    """模拟 nvprof --print-gpu-trace 的文本表格输出."""
    print('== nvprof GPU trace (已废弃, 用 nsys) ==')
    print(f"{'Kernel':20s} {'Total(ms)':12s} {'Calls':6s} {'Avg(us)':10s}")
    for name, t, c in kernels:
        print(f"{name:20s} {t:12.2f} {c:6d} {t/c*1000:10.1f}")

# 模拟: kernel 名, 总时间 ms, 调用次数
nvprof_gpu_trace([
    ('mm_kernel', 50.0, 100),     # 100 次, 共 50ms
    ('attn_kernel', 30.0, 200),
    ('memcpy_h2d', 10.0, 50),
])
```

### 5.2 真实 nvprof 命令（已废弃，迁移到 nsys/ncu）

```bash
# 旧 (nvprof):
# nvprof --print-gpu-trace python train.py
# 新 (nsys):
# nsys profile --stats=true -o trace python train.py
# 旧 (nvprof metrics):
# nvprof --metrics achieved_occupancy python train.py
# 新 (ncu):
# ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active python train.py
```


## 6. 与其他知识点的关系

- **上游（依赖）**: CUDA runtime（旧版）、NVIDIA driver。
- **下游（应用）**: [[nsys]]（继任者，系统级）、ncu（继任者，kernel 级）、[[SM utilization]]/[[memory bandwidth]]（指标解释）。
- **对比 / 易混**:
  - **nvprof vs [[nsys]]**：nvprof 旧、混合层次、不支持新 GPU；nsys 新、系统级、主力。迁移：nvprof 的 trace 功能 → nsys。
  - **nvprof vs ncu**：nvprof 的 kernel 指标 → ncu（更细更多）。
  - **nvprof vs [[torch profiler]]**：nvprof 是 CUDA 层（独立于 PyTorch），torch profiler 是 PyTorch 应用层。


## 7. 常见误区与易错点

> [!warning] 误区 1：新项目还用 nvprof
> nvprof 自 CUDA 11 deprecated，CUDA 12+ 不支持 H100 等新 GPU。新项目必用 nsys/ncu。用 nvprof 会在新 GPU 报错或无数据。

> [!warning] 误区 2：混淆 nvprof 与 [[nsys]]/ncu
> nvprof 是旧统一工具；nsys（系统级）和 ncu（kernel 级）是新分立工具。nvprof 的功能被两者瓜分。读旧资料时把 nvprof 的用法映射到 nsys（trace）或 ncu（指标）。

> [!warning] 误区 3：以为 nvprof 能用 H100
> 不能。nvprof 不支持 Hopper 架构（PMU/SMU 寄存器变化）。H100 必须用 nsys/ncu。V100 及以前仍可用 nvprof（虽 deprecated）。

> [!warning] 误区 4：忽略迁移
> 旧代码用 nvprof 的脚本需迁移到 nsys/ncu，否则新环境失效。迁移对照表见 §3.4。

> [!warning] 误区 5：以为 nvprof 的指标和 ncu 一样
> nvprof 的指标较少且旧；ncu 指标更全更准（如 roofline chart、per-section report）。迁移后用 ncu 能看到 nvprof 看不到的细节。


## 8. 延伸细节

### 8.1 为什么 nvprof 被废弃

- **架构老化**：nvprof 采集依赖 GPU 的 PMU/SMU 寄存器，新 GPU（Hopper）的寄存器布局变，nvprof 不更新。
- **功能分立**：nsys（系统）+ ncu（kernel）分工更清晰、各自更强。
- **NVIDIA 战略**：推 Nsight 品牌统一（Systems + Compute），nvprof 是遗留。

### 8.2 迁移对照

完整迁移表见 §2.4。核心：

- 任何 trace/timeline 需求 → nsys。
- 任何 kernel 指标需求 → ncu。
- nvvp GUI → Nsight Systems UI。

### 8.3 老 GPU 的兼容

V100（Volta）、T4、P100 等老 GPU 上 nvprof 仍可运行（CUDA 11 deprecated 但未完全移除）。但 CUDA 12+ 对 H100 不支持。若维护老 GPU 环境，nvprof 可用但建议迁移。

### 8.4 旧资料的解读

读 2018 前的论文/博客看到 nvprof 命令，映射：

- `nvprof --print-gpu-trace` → `nsys --stats`。
- `nvprof --metrics X` → `ncu --metrics`（指标名可能变，查 ncu 文档对应）。

理解旧资料的结论（如"occupancy 50% 是瓶颈"），但用 nsys/ncu 重新验证。

### 8.5 nvvp GUI

nvvp（NVIDIA Visual Profiler）是 nvprof 的 GUI，Java 写，显示 timeline + 指标。Nsight Systems UI 是其继任，功能更强（大 trace 不卡、Perfetto 兼容）。旧 .nvprof 文件只能用 nvvp 打开，新工具不兼容。

---
相关: [[profiling工具]] | [[nsys (Nsight Systems)]] | [[torch profiler]] | [[SM utilization]] | [[memory bandwidth]]
