# GPU执行模型

> **所属章节**: [[GPU执行模型与内存层级]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: GPU execution model / SIMT / grid-block-thread / warp 调度模型
> **难度**: 中高（需懂硬件结构 + CUDA 编程模型）


## 1. 一句话定义

**GPU 执行模型**是 GPU 如何组织、调度、执行线程的抽象——程序员按 **grid → block → thread** 三级层次组织并行计算，硬件把同 block 的 32 个 thread 编成一个 **warp**，由 SM 的 warp 调度器在驻留的多个 warp 间快速切换以**隐藏延迟**，整体遵循 **SIMT**（Single Instruction, Multiple Threads）执行范式：同一 warp 的 32 线程在同一时刻执行同一指令，但各自处理不同数据。它与 CPU 的"低延迟单线程"哲学相反——GPU 靠**海量并行 + 切换**换吞吐，是理解所有后续 CUDA 知识（[[GPU内存层级]]、[[SM utilization]]、[[Roofline模型]]、[[FlashAttention]]）的根基。

> [!note] 三句话定位
> - **是什么**：grid/block/thread 三级 + warp（32 线程）调度 + SIMT 执行，靠多 warp 切换隐藏延迟。
> - **为什么**：GPU 算力远超单线程可消化的量，必须靠海量并行塞满算力；延迟靠"等一个 warp 时切到另一个"藏起来。
> - **与 CPU 执行模型区别**：CPU 重低延迟（深流水、分支预测、大缓存），GPU 重吞吐（多 warp、简单调度、靠并行度盖延迟）。


## 2. 为什么需要它（动机与背景）

### 2.1 GPU 的设计哲学：吞吐优先

CPU 的核心目标是**单线程低延迟**——深流水线（20+ 级）、分支预测、乱序执行、大 L1/L2/L3 cache、预取，都是为了把"一条指令"尽快做完。但单核算力有限。

GPU 反其道：放弃单线程低延迟，靠**成千上万线程并行**堆总算力。A100 有 108 个 SM × 64 warp × 32 线程 = **6912 个并发线程**，单卡 bf16 峰值 312 TFLOPs。这种算力规模只能用"海量并行"喂饱。

### 2.2 延迟靠并行度藏，不靠硬件预测

CPU 单线程遇到长延迟操作（cache miss 几百 cycle）只能等或切换进程（开销大）。GPU 的策略：**SM 上驻留很多 warp，一个 warp 等内存时，warp 调度器立刻切到另一个 ready 的 warp 继续算**。只要驻留的 warp 足够多，总能有可执行的，SM 不空转。

这就是 **latency hiding by parallelism**——靠并行度（驻留 warp 数）盖住延迟，不需要 CPU 那套复杂预测硬件。前提是**程序能提供足够的可并行线程**（occupancy 足，见 [[SM utilization]]）。

### 2.3 编程模型与硬件模型的桥

程序员写 CUDA kernel 时按 grid/block/thread 组织（编程模型）；硬件按 SM/warp 执行（硬件模型）。执行模型就是把两者对应起来：

- 程序员的 block → 调度到某个 SM 上驻留
- 程序员的 32 thread → 编成一个 warp → SIMT 同步执行
- 程序员的 grid → 跨所有 SM 分发 block

理解这个映射，才能写出 occupancy 高、访存合并、warp 不分化的 kernel。


## 3. 核心概念详解

### 3.1 三级线程层次：grid / block / thread

CUDA 编程模型把并行计算组织成三级：

| 层级 | 含义 | 硬件落点 | 规模 |
|---|---|---|---|
| **grid** | 一次 kernel launch 的全部线程 | 整个 GPU（所有 SM） | 数百万 thread |
| **block**（CTA） | 合作共享 smem/sync 的线程组 | 单个 SM（不跨 SM） | ≤1024 thread |
| **thread** | 最小执行单元 | warp 的 1/32 | 1 |

```python
# CUDA C 风格的 launch 配置
# <<<grid, block>>>：grid=若干 block，block=若干 thread
# 例：处理 N 个元素，每 block 256 thread
import math
N = 1_000_000
BLOCK = 256
GRID = math.ceil(N / BLOCK)
# kernel<<<GRID, BLOCK>>>()  # GRID 个 block，每 block 256 thread
```

- **block 内可同步**：`__syncthreads()` 屏障同步、共享 [[GPU内存层级|shared memory]]、可协作（如 tiling GEMM）。
- **block 间不可同步**（同一 kernel 内）、不共享 smem——这是 block 被调度到任意 SM、任意时刻执行的必然结果（硬件自由调度 block，无法保证 block 间时序）。
- **grid 间**（不同 kernel）串行或用 stream 控制（见 [[CUDA stream与event]]）。

### 3.2 warp：硬件的真正调度单位

程序员写的是 thread，但**硬件调度的最小单位是 warp = 32 个连续 thread**。一个 block 的 thread 按 `threadIdx` 顺序每 32 个编成一个 warp。

- **warp 的 32 线程同一时刻执行同一指令**（SIMT）。
- 若 block 有 256 thread → 8 个 warp。SM 调度的是这 8 个 warp，不是 256 个 thread。
- block size 必须是 32 的倍数才不浪费（否则最后一个 warp 不足 32，仍占一个 warp 的调度槽但部分线程 masked off）。

> [!tip] block size 经验
> 常用 128/256/512（32 倍数）。256 是很多 kernel 的甜点——够多 warp 隐藏延迟，又不过大压 occupancy。

### 3.3 SIMT vs SIMD

| | SIMD（CPU AVX） | SIMT（GPU） |
|---|---|---|
| 宽度 | 固定（AVX-512 = 16 × float32） | warp = 32，固定 |
| 编程模型 | 显式向量化（intrinsic） | 写标量代码，硬件自动束成 warp |
| 分支 | 全走同一指令，分支用 mask | warp divergence：分化的分支**串行**执行各路径 |
| 寄存器 | 一份向量寄存器 | 每线程独立寄存器，硬件分配 |

SIMT 让程序员写"看起来像标量"的代码，硬件按 warp 批量执行，比 SIMD 对程序员友好。但**warp divergence**是代价：同一 warp 内若 `if (cond)` 让一半线程走 A、一半走 B，硬件先让全 warp 走 A（B 的线程 masked idle），再走 B（A 的 idle），串行化 → 并行度折半。

### 3.4 warp 调度器与延迟隐藏

每个 SM 有若干 **warp 调度器**（A100 每 SM 4 个，可同时 issue 4 个 warp 的指令）。SM 上驻留的所有 warp 在调度器视角分为：

- **ready**：操作数就绪、可立即 issue。
- **stalled**：在等内存 / 等同步 / 等功能单元。

调度器每 cycle 从 ready 的 warp 里挑一个 issue。**一个 warp 等内存（200+ cycle）时，调度器不闲着，立刻切到另一个 ready warp**。只要驻留 warp 足够多、足够多 ready，SM 一直有指令可 issue → 不空转。

这就是延迟隐藏的核心。**occupancy（驻留 warp 数 / 最大 warp 数）是延迟隐藏能力的上界**——occupancy 太低，没足够 warp 切，SM 必空转。详见 [[SM utilization]] 与 [[occupancy分析]]。

### 3.5 kernel launch 到执行的链路

一次 `kernel<<<grid, block>>>(args)` 的全链路：

1. **CPU 端**：CUDA runtime 把 launch 参数打包，写入 GPU 可见的 launch buffer（异步，不阻塞 CPU）。见 [[kernel launch overhead]]。
2. **GPU 端前端**：GPU 的 front-end 从 launch buffer 取出 kernel 描述符 + grid/block 配置。
3. **block 分发**：GPU 的 block scheduler 把 grid 的 block 逐个分发到有空闲资源的 SM（一个 SM 可同时驻留多 block，只要资源够）。
4. **SM 执行**：每 block 的 thread 编成 warp，进入 SM 的 warp 调度池，调度器轮流 issue。
5. **完成**：所有 block 完成 → kernel 结束 → 若该 stream 后续有 kernel，继续；否则 stream 空闲。

整个 launch 有**固有开销**（几 μs），所以小 kernel（elementwise 少量元素）会被 launch 开销主导，见 [[kernel launch overhead]]——这也是 [[CUDA Graph与graph capture]] 的动机（把多次 launch 合成图，replay 只付一次开销）。


## 4. 数学原理 / 公式

### 4.1 延迟隐藏条件

设 SM 上驻留 $n$ 个 warp，每 warp 一次计算耗时 $T_c$ cycle、一次访存等待 $T_m$ cycle。SM 不空转的条件是"等内存时仍有足够 warp 可算"：

$$
n \cdot T_c \ge T_m \quad \Rightarrow \quad n \ge \frac{T_m}{T_c}
$$

> [!note] 示例
> $T_m = 400$ cycle（HBM 访问 ~400 cycle），$T_c = 10$ cycle（一次 FFMA）→ 需 $n \ge 40$ warp。A100 每 SM 最多 64 warp，occupancy $\ge 40/64 = 62.5\%$ 才能完全隐藏该访存延迟。低于此 → SM 出现空转气泡。

### 4.2 并行度规模

$$
\text{GPU 并发线程数} = N_{\text{SM}} \times \frac{W_{\max}}{\text{SM}} \times 32
$$

A100：$108 \times 64 \times 32 = 221{,}184$ 线程（理论驻留上限，实际受 occupancy 限）。

### 4.3 block 数量

$$
\text{grid block 数} = \left\lceil \frac{\text{元素数}}{\text{block\_size}} \right\rceil
$$

每个 SM 能驻留的 block 数受资源限（寄存器 / smem / 硬件上限），见 [[SM utilization]] 的 occupancy 公式。

### 4.4 warp divergence 代价

设 warp 内 fraction $f$ 的线程走分支 A（耗时 $t_A$）、$1-f$ 走 B（$t_B$）：

$$
\text{实际耗时} = t_A + t_B \quad (\text{串行})，\quad \text{理想并行耗时} = \max(t_A, t_B)
$$

浪费比 $= (t_A + t_B - \max(t_A,t_B)) / (t_A+t_B)$。极端 $f=0.5, t_A=t_B=t$ → 浪费 50%。故 kernel 应避免 warp 内分支分化（把同分支的 thread 排到一个 warp）。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟 grid/block/warp 映射

```python
def launch_config(N, block=256, warps_per_sm=64, sm=108):
    """模拟 CUDA launch: grid/block/warp 映射 + SM 分发."""
    grid = (N + block - 1) // block
    warps_per_block = (block + 31) // 32
    # 假设 block 均匀分发到 SM
    blocks_per_sm = (grid + sm - 1) // sm
    active_warps = blocks_per_sm * warps_per_block
    occupancy = min(active_warps / warps_per_sm, 1.0)
    return {
        'grid': grid, 'block': block, 'warps_per_block': warps_per_block,
        'blocks_per_sm': blocks_per_sm, 'active_warps_per_sm': active_warps,
        'occupancy': occupancy,
    }

# 处理 100 万元素，每 block 256 thread
cfg = launch_config(1_000_000, block=256)
print(f"grid={cfg['grid']}, warps/block={cfg['warps_per_block']}, "
      f"blocks/SM≈{cfg['blocks_per_sm']}, occupancy≤{cfg['occupancy']:.0%}")
# grid=3907, warps/block=8, blocks/SM≈37, occupancy=100% (受 cap)

# 小数据: 处理 100 元素 -> 1 个 block, 4 个 warp, occupancy 6%
cfg2 = launch_config(100, block=256)
print(f"小数据 occupancy≤{cfg2['occupancy']:.0%} (launch overhead 主导, 见 kernel launch overhead)")
```

### 5.2 warp divergence 演示

```python
def warp_divergence_cost(n_a, n_b, t_a, t_b):
    """同 warp 内 n_a 线程走 A, n_b 走 B (SIMT 串行)."""
    actual = t_a + t_b
    ideal = max(t_a, t_b)
    waste = (actual - ideal) / actual
    return actual, waste

# 半数走 A(10cyc) 半数走 B(10cyc) -> 串行 20, 浪费 50%
print(warp_divergence_cost(16, 16, 10, 10))   # (20, 0.5)
# 全走同一分支 -> 无分化
print(warp_divergence_cost(32, 0, 10, 0))    # (10, 0.0)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: GPU 硬件结构（SM / warp 调度器 / tensor core）。
- **下游（应用）**: [[SM utilization]]（occupancy 是延迟隐藏能力的上界）、[[GPU内存层级]]（warp 的访存合并、smem tiling）、[[Roofline模型]]（算力/带宽与执行模型共同决定 bound）、[[FlashAttention]]（tiling + warp 配合减 HBM 访问）、[[CUDA stream与event]]（grid 间的执行顺序）、[[kernel launch overhead]]（launch 链路的开销）、[[occupancy分析]]（测量与调优）。
- **对比 / 易混**:
  - **SIMT vs SIMD**：见 3.3，SIMT 写标量硬件束成 warp，SIMD 显式向量化。
  - **block vs warp**：block 是编程单位（可含 syncthreads/smem），warp 是硬件调度单位（32 thread SIMT）。一个 block 含多个 warp。
  - **grid vs block**：grid 是全 launch，block 是 SM 内合作单位。block 间不可同步。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 thread 是硬件调度单位
> 硬件调度的是 **warp（32 thread）**，不是单个 thread。你写的"每个 thread 干什么"会被硬件按 32 个束成 warp 一起执行。block size 不是 32 倍数会浪费末位 warp。

> [!warning] 误区 2：以为 SIMT = SIMD，分支免费
> SIMT 允许 warp 内分支，但**分化的分支串行执行**（warp divergence），并行度折半。应尽量让同 warp 走同分支（数据重排、warp-level voting）。

> [!warning] 误区 3：混淆 grid / block / thread 的同步边界
> `__syncthreads()` 只同步 **block 内**。block 间无屏障（同 kernel 内）、grid 间靠 stream 串行。跨 block 同步要用 separate kernel + stream 或 cooperative groups，不能在 kernel 内乱等。

> [!warning] 误区 4：block 越大 occupancy 越高
> 不一定。block 大 → 每 block 占的寄存器/smem 多 → SM 能驻留的 block 数少。有时小 block（128）反而能驻留更多 block，occupancy 更高。occupancy 由寄存器/smem/block-size 共同决定，见 [[SM utilization]]。

> [!warning] 误区 5：忽视 launch 开销
> 小 kernel（elementwise 少量元素）的耗时被 launch 开销（几 μs）主导，算力远用不满。要么用 [[CUDA Graph与graph capture]]，要么融合算子（[[fused kernel]]），要么提 batch。


## 8. 延伸细节

### 8.1 A100 / H100 的 SM 规格

| | A100 | H100 SXM |
|---|---|---|
| SM 数 | 108 | 132 |
| 每 SM 最大 warp | 64 | 64 |
| 每 SM 寄存器 | 65536 (256KB) | 65536 |
| 每 SM shared mem | 192KB（可配 164KB smem） | 228KB |
| warp 调度器/SM | 4 | 4 |

### 8.2 cooperative groups：超越 block 的协作

CUDA 9+ 的 cooperative groups 允许跨 block（甚至 grid）同步，但需 kernel launch 时保证所有 block 同时驻留（launch 限制更严）。用于全局 barrier 的算法，但常规 kernel 不用。

### 8.3 tensor core 与 warp

tensor core（矩阵乘加速单元）以 warp 为单位提交：一个 `wmma::mma_sync` 让整个 warp 协作完成一个 $16\times16\times16$ 的矩阵乘。这是 [[CUTLASS与GEMM]] 与 [[Triton kernel开发]] 在底层调用的硬件单元。

### 8.4 与 CPU 执行模型对比

| | CPU | GPU |
|---|---|---|
| 目标 | 低延迟 | 高吞吐 |
| 核心 | 少（8-64核） | 多（SM × warp × 32） |
| 延迟隐藏 | 深流水 + 分支预测 + 大 cache | 多 warp 切换 |
| 单线程性能 | 强 | 弱 |
| 适合 | 串行逻辑、分支密集 | 规整并行、数据并行 |

### 8.5 内容来源

执行模型核心（grid/block/thread、warp、SIMT、延迟隐藏）整理自 NVIDIA CUDA C++ Programming Guide 的 Hardware Implementation 章节，以及 《Programming Massively Parallel Processors》第 3-4 章。

---
相关: [[GPU执行模型与内存层级]] | [[GPU内存层级]] | [[SM utilization]] | [[occupancy分析]] | [[Roofline模型]] | [[kernel launch overhead]] | [[CUDA stream与event]] | [[CUDA Graph与graph capture]] | [[fused kernel]] | [[CUTLASS与GEMM]] | [[Triton kernel开发]] | [[FlashAttention]]
