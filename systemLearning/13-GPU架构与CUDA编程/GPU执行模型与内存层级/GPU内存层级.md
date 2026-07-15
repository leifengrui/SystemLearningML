# GPU内存层级

> **所属章节**: [[GPU执行模型与内存层级]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: GPU memory hierarchy / 显存层级 / register · shared · L1 · L2 · HBM
> **难度**: 中高（需懂硬件 + 访存模式）


## 1. 一句话定义

**GPU 内存层级**是 GPU 上从快到慢、从小到大的一组存储层级——**register（寄存器）→ shared memory（片上 SRAM，常与 L1 共用）→ L1/L2 cache（硬件管理）→ HBM（高带宽显存）→ host 内存（CPU DRAM，经 PCIe/NVLink 拿）**——每一级容量更大但带宽更低、延迟更高，CUDA 编程的核心优化就是**把热数据压到尽量靠近计算单元的层级**（register / shared memory），减少对 HBM 的访问。它是 [[Roofline模型]] 里"带宽"那一维的具体物理来源，也是 [[FlashAttention]] 等融合算子能减 HBM IO 的根基。

> [!note] 三句话定位
> - **是什么**：register → shared/L1 → L2 → HBM → host，从快小到慢大，越靠近算力越快。
> - **为什么**：HBM 虽快（TB/s 级）但相对算力仍是瓶颈（memory-bound），把热数据放 shared/register 才能喂饱算力。
> - **与 CPU 内存层级区别**：GPU 的 shared memory 可编程（程序员显式 tiling），CPU cache 完全硬件管理不可控。


## 2. 为什么需要它（动机与背景）

### 2.1 算力与带宽的不匹配

A100：算力 312 TFLOPs（bf16），HBM 带宽 2 TB/s。算一个 FMA 要 2 FLOP，每 byte 数据只能支撑 $\frac{2\text{ FLOP}}{1\text{ byte...}}$——若每读 1 byte 只算很少 FLOP（算术强度低），带宽喂不饱算力，算力闲置（memory-bound）。解决就是**减少对慢层级（HBM）的访问**，让数据尽量在快层级（shared/register）被复用。

### 2.2 物理层级的存在

GPU 物理上有多种存储介质，延迟/带宽/容量差异巨大：

- **register**：每线程私有，最快（0 cycle issue），容量极小。
- **shared memory**：每 SM 片上 SRAM，全 block 共享，可编程，延迟 ~20-30 cycle。
- **L1/L2 cache**：硬件自动管理，L1 与 shared 同一物理 SRAM（可配比），L2 全 GPU 共享。
- **HBM**：主显存，大但相对慢（~400-600 cycle），是 Roofline 里"带宽"的主来源。
- **host 内存**：CPU DRAM，经 PCIe/NVLink 传，最慢，需 pinned memory + async memcpy 提速（见 [[异步memcpy与pinned memory]]）。

程序员必须知道数据在哪个层级，才能写 tiling、合并访存、用 register 累加。

### 2.3 可编程性是 GPU 的关键差异

CPU 的 cache 对程序员透明不可控（除 prefetch/nt）。GPU 的 **shared memory 显式可编程**——程序员声明 `__shared__ T tile[...]`，显式决定哪些数据搬进 smem、怎么 tiling、何时同步。这是 CUDA 性能优化的核心抓手：**把 HBM 上的多次访问，通过 tiling 搬进 smem 一次，变成对 smem 的多次快访问**。[[FlashAttention]] 的核心就是这个思想。


## 3. 核心概念详解

### 3.1 层级总表

| 层级 | 物理位置 | 容量(A100/SM) | 延迟(cycle) | 带宽 | 可编程 | 范围 |
|---|---|---|---|---|---|---|
| **register** | 每 SM 的寄存器堆 | 65536 (256KB) | ~0（issue 即用） | 最高 | 间接（编译器分配） | 线程私有 |
| **shared memory** | SM 片上 SRAM | ≤164KB（可配） | ~20-30 | 高（~128B/cycle/SM） | **显式** `__shared__` | block 内共享 |
| **L1 cache** | 同 shared 的 SRAM | 余下部分 | ~20-30 | 高 | 硬件管理 | SM 内 |
| **L2 cache** | 全 GPU 共享 | 40MB(A100) | ~200 | 中 | 硬件管理 | 全 GPU |
| **HBM** | 显存颗粒 | 40/80GB | ~400-600 | 2 TB/s | 显式分配 | 全 GPU |
| **host (pinned)** | CPU DRAM | 系统内存 | PCIe/NVLink | PCIe 64GB/s, NVLink 600GB/s | pinned | 跨设备 |

> 容量/延迟为 A100 参考值，H100/B200 数值不同；延迟为粗略量级，随访问模式变化。

### 3.2 register：最快但最稀缺

每线程的局部变量、循环计数、累加器都在 register。A100 每 SM 65536 寄存器，若每线程用 32 个，则每 SM 可驻留 $65536/32 = 2048$ 线程 = 64 warp = 100% occupancy。若每线程用 128 寄存器，只能驻留 512 线程 = 16 warp = 25% occupancy。

**寄存器压力（register pressure）**：kernel 用寄存器过多 → occupancy 跌、甚至 spill 到 local memory（实为 HBM，慢）。调优用 `--maxrregcount` 或 `__launch_bounds__` 限寄存器。详见 [[occupancy分析]]。

### 3.3 shared memory：可编程的"手动 cache"

```python
# CUDA C 伪代码（说明 tiling 思想）
__shared__ float tile[BLOCK][TILE];      # block 内共享, 快
for (i in tile_load_range):
    tile[...] = global_mem[...];          # 一次慢的 HBM 读
__syncthreads();                          # 等 block 都载入
# 之后多次访问 tile[]（快）, 复用 HBM 读进来的数据
```

**shared memory 的 bank conflict**：smem 按 32 个 bank 组织（每 bank 4 byte 宽）。一个 warp 的 32 线程若同时访问同 bank 的不同地址 → 串行化（bank conflict），延迟翻倍。**无冲突**模式：32 线程访问 32 个不同 bank（如顺序读 float32 数组）。详见延伸 8.2。

### 3.4 L1/L2：硬件管理的 cache

- **L1**：与 shared 同一 SRAM，可配比（`cudaFuncSetAttribute`）。硬件自动 cache 全局访问（若不绕过）。透明加速重复访问。
- **L2**：全 GPU 共享（A100 40MB），跨 SM。HBM 访问先过 L2，命中则免 HBM。多 SM 间的数据复用靠 L2。

LLM 推理的 KV cache 命中 L2 能显著减 HBM 压力；vLLM/SGLang 的 [[Radix Tree prefix cache]] 部分依赖 L2 局部性。

### 3.5 HBM：主显存，主战场

HBM 是 GPU 主显存（A100 80GB），权重/激活/KV/optimizer state 都在这。带宽 2 TB/s 看似高，但相对 312 TFLOPs 算力仍吃紧（ridge point ~156 FLOP/byte，见 [[Roofline模型]]）。

**所有"减显存访问"的优化**（[[FlashAttention]] 的 tiling、[[fused kernel]] 的算子融合、[[gradient checkpointing]] 的重算换显存）本质都是减 HBM 读写次数。

### 3.6 host 内存与 pinned memory

CPU 内存经 PCIe/NVLink 传到 HBM。普通 pageable 内存要先拷到 pinned staging 再 DMA，慢。**pinned（page-locked）内存**是锁页的 host 内存，可直接 DMA，配合 `cudaMemcpyAsync` 走 copy engine 异步传输，与计算 overlap。详见 [[异步memcpy与pinned memory]]。

### 3.7 memory space 的可见性与一致性

| space | 同 block | 跨 block | 跨 SM |
|---|---|---|---|
| register | 线程私有 | 不可见 | 不可见 |
| shared | 可见（需 sync） | 不可见 | 不可见 |
| global (HBM) | 可见 | 可见（需 kernel 边界 / stream 序） | 可见 |
| L1/L2 | — | 透明 | 透明 |

global memory 的跨 block 一致性：同 kernel 内 block 间不保证顺序（除非 stream 序或 cooperative groups）；跨 kernel 由 stream 序保证。


## 4. 数学原理 / 公式

### 4.1 算术强度与层级

$$
I = \frac{\text{FLOP}}{\text{Bytes from HBM}}
$$

把数据从 HBM 搬到 shared/register 后多次复用，等效**减少 Bytes from HBM**（分子 FLOP 不变）→ $I$ 升 → 从 memory-bound 向 compute-bound 移。这是 tiling 的数学本质。

### 4.2 tiling 的复用收益

设一个 $N\times N$ 矩阵被访问 $k$ 次（如 $k$ 个 GEMM），无 tiling：HBM 读 $k\cdot N^2$ bytes。tiling 进 smem 一次读 $N^2$、复用 $k$ 次：HBM 读 $N^2$。收益 $k\times$。

### 4.3 smem 容量 vs occupancy

设每 block 用 $s$ bytes smem，SM smem 总 $S_{\max}$（164KB）：

$$
b_{\text{smem}} = \left\lfloor \frac{S_{\max}}{s} \right\rfloor, \quad \text{occupancy}_{\text{smem}} = \frac{b_{\text{smem}} \cdot w}{W_{\max}}
$$

$s$ 大 → 驻留 block 少 → occupancy 跌。tiling 尺寸与 occupancy 是 trade-off。详见 [[SM utilization]]。

### 4.4 bank conflict 延迟

无冲突：1 cycle 服务一个 warp 的 32 路访问。$n$-路 bank conflict：$n$ cycle。strided 访问若 stride 是 32 的倍数 → 全部撞同 bank → 32 路 conflict（最坏）。

### 4.5 HBM 带宽利用

$$
\text{有效 HBM 带宽} = \frac{\text{实际读写字节}}{\text{耗时}}, \quad \text{利用率} = \frac{\text{有效}}{\text{峰值 (2 TB/s)}}
$$

ncu 报 `dram__bytes.sum.per_second` 即此。低利用率说明 kernel 没"喂"满 HBM（可能 launch 太小或访问不合并），见 [[ncu (Nsight Compute)]]。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟 tiling 复用收益

```python
def tiling_benefit(N, k):
    """N*N 矩阵被访问 k 次: 无 tiling vs tiling 进 smem."""
    bytes_per_elem = 2  # fp16
    no_tile = k * N * N * bytes_per_elem          # 每次都从 HBM 读
    with_tile = N * N * bytes_per_elem            # 只读一次进 smem
    return no_tile, with_tile, no_tile / with_tile

no, yes, gain = tiling_benefit(4096, 8)
print(f"无 tiling: {no/1e9:.1f} GB HBM 读; tiling: {yes/1e9:.1f} GB; 复用收益 {gain}x")
# 无 tiling: 268.4 GB; tiling: 33.6 GB; 收益 8x
```

### 5.2 bank conflict 模拟

```python
def bank_access(stride, warp_threads=32, banks=32):
    """warp 32 线程按 stride 访问 fp32 数组, 返回 bank conflict 数."""
    banks_hit = [(i * stride) % banks for i in range(warp_threads)]
    # 同 bank 的最大冲突路数
    from collections import Counter
    c = Counter(banks_hit)
    return max(c.values())

print(f"stride=1 (顺序): {bank_access(1)}-way conflict")   # 1-way (无冲突)
print(f"stride=32 (撞 bank): {bank_access(32)}-way")        # 32-way (最坏)
print(f"stride=2: {bank_access(2)}-way")                     # 1-way (偶数/奇数分开, 无冲突)
```

### 5.3 真实 CUDA 对照（伪代码见 3.3）

```python
# PyTorch 视角: shared memory 对用户透明, 但 ncu 能看 smem 利用率
# ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
#       --metrics smem__throughput.avg.pct_of_peak_sustained_active \
#       python my_kernel.py
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GPU执行模型]]（warp / block 是访问的执行单位）、GPU 硬件物理结构。
- **下游（应用）**: [[Roofline模型]]（HBM 带宽是 memory-bound 的物理来源）、[[SM utilization]]/[[occupancy分析]]（register/smem 限 occupancy）、[[FlashAttention]]（tiling 减 HBM IO）、[[fused kernel]]（融合减中间结果落 HBM）、[[异步memcpy与pinned memory]]（host↔device 层级）、[[CUDA allocator与memory pool]]（HBM 的分配与碎片）、[[memory bandwidth]]（HBM 带宽利用率）。
- **对比 / 易混**:
  - **shared memory vs L1 cache**：物理同一 SRAM，但 shared 可编程（显式 `__shared__`），L1 硬件管理。可配比。
  - **shared memory vs register**：register 线程私有最快，shared block 共享稍慢但可协作。
  - **HBM vs host pinned**：HBM 是 GPU 本地显存（TB/s），host pinned 是 CPU DRAM（PCIe/NVLink，慢一个量级）。


## 7. 常见误区与易错点

> [!warning] 误区 1：把 shared memory 当 cache 用却忘了 tiling
> shared 是显式可编程的，不是自动 cache。你不主动 `__shared__` 声明 + 载入 + syncthreads，数据不会进 smem，每次访问都打 HBM。tiling 是手动的。

> [!warning] 误区 2：忽视 bank conflict
> warp 32 线程若访问 smem 的 pattern 撞同 bank（如 stride=32），串行化延迟翻几十倍。padding 一列（`tile[N][N+1]`）打破 stride 是常用 trick。

> [!warning] 误区 3：以为 L1/L2 能救所有重复访问
> L1/L2 容量有限（L1 几 KB/SM，L2 40MB），大矩阵的重复访问 cache 不住。显式 tiling 进 smem 才可靠。LLM 的 KV cache 大于 L2 时，prefix cache 命中率掉。

> [!warning] 误区 4：混淆 local memory 与 shared
> 寄存器 spill 出来的"local memory"物理在 HBM（慢），不是片上！叫"local"误导新手以为快。看 ncu 的 `l1tex__t_sectors_pipe_lsu_mem_global` 区分。

> [!warning] 误区 5：以为 HBM 带宽 = 可用带宽
> 峰值 2 TB/s 需要合并访问（coalesced：warp 32 线程访问连续 128B）才达得到。strided/不合并访问实际带宽可能只到峰值的 10-30%。访问合并是 kernel 优化基本盘。


## 8. 延伸细节

### 8.1 A100 / H100 / B200 显存规格

| | A100 80GB | H100 80GB | B200 |
|---|---|---|---|
| HBM | HBM2e | HBM3 | HBM3e |
| HBM 容量 | 80GB | 80GB | 192GB（待核实） |
| HBM 带宽 | 2.0 TB/s | 3.35 TB/s | ~8 TB/s |
| L2 | 40MB | 50MB | 增大（待核实） |
| 每 SM smem | ≤164KB | ≤228KB | 增大 |

### 8.2 smem bank 与 padding

smem 32 bank × 4 byte。访问 `tile[i][j]`（float32），warp 32 线程若 `i` 同、`j` 连续 → 无冲突。若 `j` stride=32 → 全撞 bank 0/32/64... → 32-way conflict。常见解法：`tile[N][N+1]` 加一列 padding，打破 stride=32 的对齐。

### 8.3 L2 persistent cache

CUDA 11+ 允许 L2 的 persistent 部分钉住某段 global memory（`cudaStreamSetAttribute`），用于高频访问的数据（如 LLM 的 attention bias）。是减 HBM 压力的手段。

### 8.4 unified memory

`cudaMallocManaged` 让 CPU/GPU 共享虚拟地址，运行时按需迁移页。方便但性能通常不如显式 memcpy + pinned，因页迁移开销与 thrash。生产训练一般不用，研究原型可。

### 8.5 内容来源

内存层级（register/shared/L1/L2/HBM、bank、tiling）整理自 NVIDIA CUDA C++ Programming Guide 的 Memory Hierarchy 章节，以及 《Programming Massively Parallel Processors》第 5 章。

---
相关: [[GPU执行模型与内存层级]] | [[GPU执行模型]] | [[Roofline模型]] | [[SM utilization]] | [[occupancy分析]] | [[memory bandwidth]] | [[FlashAttention]] | [[fused kernel]] | [[异步memcpy与pinned memory]] | [[CUDA allocator与memory pool]] | [[ncu (Nsight Compute)]] | [[Radix Tree prefix cache]]
