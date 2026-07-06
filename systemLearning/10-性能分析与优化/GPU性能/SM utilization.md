# SM utilization

> **所属章节**: [[GPU性能]]
> **所属模块**: [[10-性能分析与优化]]
> **别名**: SM utilization / SM 占用率 / SM occupancy / streaming multiprocessor utilization
> **难度**: 中高（需懂 GPU 硬件 + roofline）


## 1. 一句话定义

**SM utilization（SM 占用率）** 是 GPU **Streaming Multiprocessor（SM, 流式多处理器）** 的**空间维占用**——单个 SM 上**活跃 warp 数**占该 SM 最大可容纳 warp 数的比例，衡量 GPU 计算单元被多满地"塞满"。它关注**同一时刻有多少 warp 在 SM 上驻留**（空间），与 [[GPU utilization]]（时间维：GPU 在一段时间内有多少时间在算）是**两个不同维度**——SM utilization 高说明 SM 被塞满（并行度足），但若 kernel 是 memory-bound（等数据），SM 满了也在空转，throughput 仍低。SM utilization 是 GPU 性能调优的**空间维起点**，回答"GPU 计算单元塞满了没"。

> [!note] 三句话定位
> - **是什么**：单 SM 上活跃 warp 数 / 最大 warp 数，空间维占用。
> - **为什么**：occupancy 低 → SM 没塞满 → 算力浪费；但 occupancy 高 ≠ 性能好（memory-bound 时空转）。
> - **与 [[GPU utilization]] 的区别**：[[GPU utilization]] 是时间维（GPU 多忙），SM utilization 是空间维（SM 多满）。两者正交，需结合看。


## 2. 为什么需要它（动机与背景）

### 2.1 GPU 的 SM 结构

GPU 由多个 **SM**（如 A100 有 108 个 SM）组成，每个 SM 是独立的计算单元，含若干 CUDA core、tensor core、寄存器、shared memory。每个 SM 可同时驻留多个 **warp**（32 线程组，A100 每 SM 最多 64 warp = 2048 线程）。SM 通过 warp 调度器在驻留 warp 间快速切换，**隐藏延迟**（一个 warp 等内存时换另一个 warp 算）。

### 2.2 occupancy 与隐藏延迟

SM 的设计哲学：**靠多 warp 驻留 + 快速切换隐藏延迟**。若 SM 上只驻留 4 个 warp（occupancy 4/64=6%），一个 warp 等内存（200+ cycle）时，只有 3 个其他 warp 可切，可能都不够覆盖延迟 → SM 空闲。若驻留 64 warp（occupancy 100%），等内存时切到其他 63 个，足够覆盖 → SM 一直忙。

occupancy 低 → 延迟隐藏能力差 → SM 空闲 → 算力浪费。这是 occupancy 重要的根因。

### 2.3 occupancy 的限制因素

SM 能驻留多少 warp 受**资源**限制：

- **寄存器**：每线程用 N 寄存器，SM 总寄存器 65536（A100）。线程多用寄存器 → 每 block 线程数受限 → 驻留 block 数受限 → occupancy 降。
- **shared memory**：每 block 用 M 字节，SM 总 shared mem（A100 164KB）。block 多用 smem → 驻留 block 数受限。
- **block 结构**：每 block 线程数决定 warps/block。

这些资源是**零和**——多用寄存器换少驻留 block，trade occupancy 与单线程性能。

### 2.4 与 GPU utilization 的区分

- **SM utilization（空间）**：SM 上驻留多少 warp（塞满没）。
- **[[GPU utilization]]（时间）**：GPU 在一段时间内多少时间在执行 kernel（忙没）。

SM 满了（occupancy 高）但等内存（memory-bound）→ GPU utilization 时间维可能也高（一直在尝试执行），但有效 throughput 低。需结合 [[memory bandwidth]] 看。


## 3. 核心概念详解

### 3.1 warp 与 SM

- **warp**：32 线程组，SM 的调度单位。一个 warp 的 32 线程同步执行同一指令（SIMT）。
- **SM**：含多个 CUDA core/tensor core + 寄存器 + shared memory，可驻留多 warp。A100：108 SM × 64 warp = 6912 warp 并发。

### 3.2 occupancy 计算

$$
\text{occupancy} = \frac{\text{active warps per SM}}{\text{max warps per SM}}
$$

active warps 受资源限：

$$
\text{active warps} = \text{resident blocks} \times \text{warps per block}
$$

$$
\text{resident blocks} = \min(\text{blocks}_{\text{reg}}, \text{blocks}_{\text{smem}}, \text{blocks}_{\text{SM limit}})
$$

其中 $\text{blocks}_{\text{reg}} = \lfloor \text{regs per SM} / (\text{regs per thread} \times \text{threads per block}) \rfloor$，$\text{blocks}_{\text{smem}} = \lfloor \text{smem per SM} / \text{smem per block} \rfloor$。

### 3.3 occupancy 的限制维度

| 资源 | 限制机制 | 调优方向 |
|---|---|---|
| 寄存器 | 每线程多用寄存器 → 少驻留 block | 减寄存器（`--maxrregcount`）、`launch_bounds` |
| shared memory | 每 block 多用 smem → 少驻留 block | 减 smem、用 tiling 平衡 |
| block 大小 | warps/block 决定粒度 | 128/256 线程常用 |
| 硬件上限 | SM 最大 block/warp 数 | 不可调 |

### 3.4 occupancy 高 ≠ 性能好

> [!warning] occupancy 是必要非充分
> occupancy 高让延迟隐藏好（必要），但若 kernel 是 memory-bound（数据供不上），SM 满了也在等内存 → 有效 throughput 仍低。需结合 [[memory bandwidth]] 看是不是真算。有时**降低 occupancy**（每线程多用寄存器、减少内存访问）反而提性能（少 warp 但每 warp 算更多）。

### 3.5 occupancy 与 roofline

[[GPU utilization]] 的 roofline：算术强度 $I$（FLOP/Byte）vs ridge point $R = P_{\text{compute}}/P_{\text{bw}}$。

- $I > R$：compute-bound，occupancy 高帮提吞吐。
- $I < R$：memory-bound，occupancy 高也等内存，throughput 受带宽限。

occupancy 是 compute-bound 场景的关键；memory-bound 场景看 [[memory bandwidth]]。


## 4. 数学原理 / 公式

### 4.1 occupancy 完整公式

设 SM 最大 warp 数 $W_{\max}$（A100=64），每 block 线程数 $T$，warps/block $w = \lceil T/32 \rceil$，每线程寄存器 $r$，SM 寄存器 $R_{\max}$（A100=65536），每 block shared mem $s$，SM shared mem $S_{\max}$（A100=164KB）：

$$
b_{\text{reg}} = \left\lfloor \frac{R_{\max}}{r \cdot T} \right\rfloor, \quad b_{\text{smem}} = \left\lfloor \frac{S_{\max}}{s} \right\rfloor, \quad b = \min(b_{\text{reg}}, b_{\text{smem}}, b_{\text{hw}})
$$

$$
\text{active warps} = b \cdot w, \quad \text{occupancy} = \frac{b \cdot w}{W_{\max}}
$$

> [!note] 示例：256 threads, 32 regs, 0 smem
> $T=256, w=8, r=32, s=0$。$b_{\text{reg}} = \lfloor 65536/(32\cdot 256) \rfloor = \lfloor 65536/8192 \rfloor = 8$。$b_{\text{smem}}=\infty$（$s=0$）。$b=8$。active warps $= 8 \cdot 8 = 64 = W_{\max}$。occupancy $= 100\%$。

### 4.2 延迟隐藏

SM 上 warp 数 $n$，每 warp 算 $T_{\text{compute}}$ cycle、等内存 $T_{\text{mem}}$ cycle。SM 一直忙的条件（足够 warp 切换覆盖延迟）：

$$
n \cdot T_{\text{compute}} \ge T_{\text{mem}} \quad \Rightarrow \quad n \ge \frac{T_{\text{mem}}}{T_{\text{compute}}}
$$

如 $T_{\text{mem}}=400$ cycle、$T_{\text{compute}}=10$ cycle → 需 $n \ge 40$ warp。occupancy $\ge 40/64 = 62.5\%$ 才能隐藏延迟。低于此 → SM 空闲。

### 4.3 occupancy 与 throughput

compute-bound kernel 的 throughput 随 occupancy 升（直到算力饱和）：

$$
\text{throughput} \propto \min(\text{occupancy} \cdot P_{\text{warp}}, P_{\text{SM peak}})
$$

memory-bound kernel 的 throughput 受带宽限，与 occupancy 关系弱：

$$
\text{throughput} \propto P_{\text{bw}} \quad (\text{independent of occupancy once enough warps to saturate bw})
$$


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 occupancy 计算

```python
def compute_occupancy(regs_per_thread, threads_per_block, smem_per_block,
                      max_warps_per_sm=64, regs_per_sm=65536, smem_per_sm=164):
    """模拟 CUDA occupancy 计算."""
    warps_per_block = (threads_per_block + 31) // 32          # 向上取整到 warp
    regs_per_block = regs_per_thread * threads_per_block
    blocks_by_reg = regs_per_sm // regs_per_block if regs_per_block > 0 else 999
    blocks_by_smem = smem_per_sm // smem_per_block if smem_per_block > 0 else 999
    blocks = min(blocks_by_reg, blocks_by_smem, 32)           # 硬件 block 上限简化
    active_warps = blocks * warps_per_block
    occupancy = min(active_warps / max_warps_per_sm, 1.0)
    return occupancy, active_warps, blocks

# 场景1: 256 threads, 32 regs, 0 smem -> 100% occupancy
o1, w1, b1 = compute_occupancy(32, 256, 0)
print(f'256t/32reg/0smem: occupancy={o1:.0%}, warps={w1}, blocks={b1}')

# 场景2: 256 threads, 128 regs (寄存器多) -> occupancy 降
o2, w2, b2 = compute_occupancy(128, 256, 0)
print(f'256t/128reg/0smem: occupancy={o2:.0%}, warps={w2}, blocks={b2}')

# 场景3: 256 threads, 32 regs, 100 smem -> smem 限制
o3, w3, b3 = compute_occupancy(32, 256, 100)
print(f'256t/32reg/100smem: occupancy={o3:.0%}, warps={w3}, blocks={b3}')
```

### 5.2 真实 CUDA API 对照

```python
# from cuda import cudart
# # 或用 ncu (Nsight Compute) 看 achieved_occupancy
# # ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./my_kernel
# # 输出 achieved occupancy (运行时实测, 理论 occupancy 的上界)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: GPU 硬件结构（SM/warp/寄存器/smem）、CUDA 编程模型。
- **下游（应用）**: [[memory bandwidth]]（memory-bound 时 occupancy 不决定 throughput）、[[compute bottleneck]]/[[memory bottleneck]]（occupancy 帮助判断瓶颈类型）、[[kernel launch overhead]]（小 kernel 占不满 SM）、[[torch profiler]]/[[nsys]]（实测 occupancy）。
- **对比 / 易混**:
  - **SM utilization vs [[GPU utilization]]**：SM utilization 是空间维（SM 多满），[[GPU utilization]] 是时间维（GPU 多忙）。正交维度，需结合看。SM 满但等内存 → [[GPU utilization]] 高但有效 throughput 低。
  - **theoretical occupancy vs achieved occupancy**：理论 occupancy（资源算出的上界）vs 实测 occupancy（运行时 ncu 测的，≤ 理论）。调优看 achieved。
  - **occupancy vs throughput**：occupancy 高是必要非充分。compute-bound 时正相关，memory-bound 时弱相关。


## 7. 常见误区与易错点

> [!warning] 误区 1：occupancy 高就性能好
> 不一定。memory-bound kernel 的 occupancy 高也等内存，throughput 受 [[memory bandwidth]] 限。有时降 occupancy（多寄存器、少内存访问）反提性能。occupancy 是必要非充分。

> [!warning] 误区 2：混淆 SM utilization 与 [[GPU utilization]]
> SM utilization 是空间维（SM 驻留 warp 数），[[GPU utilization]] 是时间维（GPU 执行 kernel 的时间比）。SM 满了不代表 GPU 时间维满（可能等内存空转）。需结合看。

> [!warning] 误区 3：理论 occupancy 等于实测
> 理论 occupancy 是资源算出的上界，实测（achieved，ncu 测）可能更低（warp 调度不均、分支分化）。调优以 achieved 为准。

> [!warning] 误区 4：盲目追 100% occupancy
> 100% occupancy 不一定最优——可能为达 100% 压缩寄存器（增加内存访问）或减 smem，反而降性能。需 roofline + 实测 throughput 综合判断。

> [!warning] 误区 5：忽略寄存器压力
> 寄存器用多（每线程 >64）会显著降 occupancy。编译器可能 spill 到 local memory（慢）。用 `--maxrregcount` 或 `launch_bounds` 限寄存器，trade occupancy。


## 8. 延伸细节

### 8.1 A100 的 SM 规格

A100：
- 108 SM，每 SM 64 warp（max 2048 线程）
- 每 SM 65536 寄存器（256KB）
- 每 SM 192KB shared memory（可配 164KB smem + 余 L1）
- 4 warp 调度器/SM

### 8.2 launch_bounds 调优

`__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` 告诉编译器目标 occupancy，编译器据此分配寄存器（限制寄存器以保证 minBlocksPerSM 驻留）。是 CUDA 调优的核心 hint。

### 8.3 occupancy 与 LLM 推理

LLM 推理的 attention/MLP kernel：compute-bound（prefill 大 batch）时 occupancy 重要；memory-bound（decode 小 batch）时看 [[memory bandwidth]]。FlashAttention 的高 occupancy + 少内存访问是性能关键。

### 8.4 warp divergence

同一 warp 的 32 线程若走不同分支（if-else），串行执行各分支 → 有效并行度降。occupancy 数字不变但有效 throughput 降。需结合 divergence 看。

### 8.5 ncu 实测

`ncu`（Nsight Compute）是测 SM utilization 的工具：

```
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --kernel-name regex:attention ./my_program
```

输出 achieved occupancy。详见 [[nsys]]（系统级）vs ncu（kernel 级）。

---
相关: [[GPU性能]] | [[GPU utilization]] | [[memory bandwidth]] | [[compute bottleneck]] | [[memory bottleneck]] | [[kernel launch overhead]] | [[torch profiler]] | [[nsys]] | [[attention]] | [[FlashAttention]]
