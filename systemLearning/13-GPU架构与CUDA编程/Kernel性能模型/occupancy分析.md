# occupancy分析

> **所属章节**: [[Kernel性能模型]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: occupancy 调优 / achieved occupancy / theoretical occupancy / launch_bounds / 寄存器压力 / spill
> **难度**: 中高（需懂 [[SM utilization]] 原理 + ncu 工具）


## 1. 一句话定义

**occupancy 分析** 是把 [[SM utilization]] 里"occupancy = SM 上活跃 warp 数 / 最大 warp 数"的概念**从纸面公式落到实测与调优**——用 [[ncu (Nsight Compute)]] 测 **achieved occupancy**（运行时实测值，≤ 理论值）、诊断寄存器/shared memory 是否成瓶颈、用 `--maxrregcount`/`__launch_bounds__` 限寄存器以提 occupancy、识别 spill（寄存器溢出到 local memory=HBM，慢）与 warp stall 原因。它不重复 [[SM utilization]] 的原理推导，而是回答"我的 kernel 实测 occupancy 多少、为什么不到理论值、怎么调"。是 CUDA kernel 调优的核心闭环（理论 → 实测 → 调优 → 再实测），与 [[Roofline模型]] 共同构成 kernel 性能诊断的两把尺子。

> [!note] 三句话定位
> - **是什么**：用 ncu 测 achieved occupancy、定位寄存器/smem 瓶颈、调 launch_bounds 提 occupancy。
> - **为什么**：理论 occupancy 公式算出上界，但实测常更低（调度不均、分支分化、stall）；调优以实测为准。
> - **不是越高越好**：memory-bound kernel 提 occupancy 无益（仍等带宽），compute-bound 才受益。先 [[Roofline模型]] 判 bound 再调 occupancy。


## 2. 为什么需要它（动机与背景）

### 2.1 理论 ≠ 实测

[[SM utilization]] 的公式给出 occupancy 的**理论上界**（按寄存器/smem/block-size 算）。但实测 achieved 常低于理论：

- **warp 调度不均**：硬件调度器未必均匀，部分 warp 长期 stalled。
- **分支分化**：warp divergence 让有效并行度降。
- **资源竞争**：SM 上多 block 共享功能单元（如 tensor core 端口）。
- **stall 原因**：等内存、等同步、等功能单元。

故调优必须**实测**。`ncu` 的 `sm__warps_active.avg.pct_of_peak_sustained_active` 直接给 achieved 占比。

### 2.2 occupancy 不是终极指标

即使 achieved occupancy 100%，memory-bound kernel 仍受 [[Roofline模型]] 的带宽限（详见 [[SM utilization]] 误区 1）。**调 occupancy 之前先判 bound**：compute-bound 才值得提 occupancy，memory-bound 应减 HBM 访问（tiling/融合）而非硬塞 warp。

### 2.3 寄存器压力是常见杀手

kernel 用寄存器多 → SM 能驻留的 block 数少 → occupancy 跌。更糟的是寄存器不够时 **spill 到 local memory**（物理在 HBM，慢），既降 occupancy 又增访存。`launch_bounds` / `--maxrregcount` 限寄存器是常用调优手段。


## 3. 核心概念详解

### 3.1 theoretical vs achieved occupancy

| | 公式 | 来源 | 用途 |
|---|---|---|---|
| theoretical | [[SM utilization]] 4.1 | 寄存器/smem/block-size 算出 | 上界估计 |
| achieved | ncu metric | 运行时实测 | 调优基准 |

差距 = 调度损失。achieved 低于理论 80%+ 说明有调度/分支问题。

### 3.2 ncu 关键 metric

```bash
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active, \   # achieved occupancy %
  sm__warps_active.avg.per_cycle_active, \                # 平均活跃 warp
  launch__waves_per_multiprocessor, \                     # grid 占 SM 的波数 (越大越满)
  launch__occupancy_limit_blocks, \                       # 哪资源限制 occupancy (reg/smem/hw)
  ltsx__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld \  # 全局访问合并度
  --set full python train.py
```

| metric | 含义 | 解读 |
|---|---|---|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy % | 主要看这个，目标 compute-bound 60%+ |
| `launch__waves_per_multiprocessor` | grid 量 / SM 数 | <1 说明 grid 太小，SM 没填满 |
| `launch__occupancy_limit_blocks` | 限制资源 | 'Registers' / 'SharedMemory' / 'BlockSize' |
| `sm__sass_inst_executed_op_local_ld.sum` | local memory load | >0 说明有 spill（寄存器溢出） |
| `smsp__cycles_active.avg.pct_of_peak_sustained_active` | SM 活跃时间占比 | 实际忙没 |

### 3.3 寄存器与 occupancy 的 trade-off

A100：SM 寄存器 65536，每线程用 $r$ 个 → 每 block 256 thread 用 $256r$ 寄存器 → 每 SM 驻留 $\lfloor 65536/(256r) \rfloor$ block。

- $r=32$ → 8 block × 8 warp = 64 warp = 100% occupancy。
- $r=128$ → 2 block × 8 warp = 16 warp = 25% occupancy。
- $r=64$ → 4 block × 8 warp = 32 warp = 50% occupancy。

寄存器多用得起的算子（如复杂数学、大量中间值）可能换来单线程效率，但降 occupancy。**没有绝对最优**——要实测 throughput，见 [[Roofline模型]] 的 trade-off。

### 3.4 launch_bounds 调优

```c
__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
__global__ void my_kernel(float* out, ...) {
    // 编译器据此限寄存器, 保证 minBlocksPerSM 个 block 驻留
}
```

- `maxThreadsPerBlock`：告诉编译器 block 不会超过这个 size，可省寄存器。
- `minBlocksPerSM`：目标 occupancy，编译器压寄存器到能驻留这么多 block。

副作用：寄存器被压 → 可能 spill（若原本寄存器用得紧）。需 ncu 看 `local_ld` 是否增。

### 3.5 --maxrregcount

编译器选项 `--maxrregcount=N` 全局限每线程寄存器 ≤ N。粗暴但有效，trade occupancy 与 spill。`__launch_bounds__` 是 per-kernel 版本，更细。

### 3.6 warp stall 分析

ncu 的 "Warp State Stats" 看 warp 为何 stalled：

| stall 原因 | 含义 | 优化 |
|---|---|---|
| `stall_long_sb` | 等长延迟（HBM 访问） | 提 occupancy 隐藏 / tiling 进 smem |
| `stall_short_sb` | 等短延迟（smem/L1） | 减 bank conflict |
| `stall_imc_miss` | 等 instruction cache | 减 kernel 代码量 / 融合 |
| `stall_sync` | 等同步 | 减 __syncthreads |

stall 主因 + bound 类型共同决定优化方向。

### 3.7 与 grid 大小（waves）

`launch__waves_per_multiprocessor` = grid block 数 / (SM 数 × 每 SM 能驻留 block)。 <1 说明 grid 太小 SM 没填满（小 batch 的常见病）。compute-bound kernel 应让 waves ≥ 1（最好 2+ 让 SM 切换掩盖尾部效应）。memory-bound 则 waves 够即可（再多也等带宽）。


## 4. 数学原理 / 公式

### 4.1 achieved 与理论的差

$$
\eta_{\text{sched}} = \frac{\text{achieved}}{\text{theoretical}}
$$

$\eta_{\text{sched}}$ 反映调度/分支损失。>0.8 说明实现接近理论上界。

### 4.2 寄存器上限推导

$$
b_{\text{reg}} = \left\lfloor \frac{R_{\max}}{r \cdot T} \right\rfloor, \quad \text{occupancy} = \frac{b_{\text{reg}} \cdot w}{W_{\max}}
$$

（$r$ 每 thread 寄存器，$T$ 每 block thread，$w = \lceil T/32\rceil$，$W_{\max}=64$，$R_{\max}=65536$。）详见 [[SM utilization]] 4.1。

### 4.3 launch_bounds 的承诺

`__launch_bounds__(T, b_target)`：编译器调寄存器分配使 $b_{\text{reg}} \ge b_{\text{target}}$。若原 $r$ 使 $b_{\text{reg}} < b_{\text{target}}$，编译器压 $r$（可能 spill）以满足。约束：$b_{\text{target}} \cdot w \le W_{\max}$。

### 4.4 waves 与 occupancy

$$
\text{waves} = \frac{\text{grid blocks}}{N_{\text{SM}} \cdot b_{\text{achieved}}}
$$

waves < 1 → 部分 SM 空闲（小 kernel 通病）。waves ≥ 2 → 尾部 SM 切换掩盖，效果好。

### 4.5 memory-bound 时的 occupancy 无效

memory-bound kernel throughput $\propto B_{\text{hbm}}$（带宽），与 occupancy 弱相关（warp 够即可）。提 occupancy 不增 throughput，反可能因压寄存器致 spill 增访存而劣化。详见 [[Roofline模型]]。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟寄存器-occupancy trade-off

```python
def occupancy_vs_regs(regs_per_thread, threads_per_block=256,
                      regs_per_sm=65536, max_warps=64):
    """寄存器多 -> occupancy 跌. A100 参数."""
    warps_per_block = (threads_per_block + 31) // 32
    regs_per_block = regs_per_thread * threads_per_block
    blocks = regs_per_sm // regs_per_block if regs_per_block else 999
    blocks = min(blocks, 32)
    occ = (blocks * warps_per_block) / max_warps
    return occ

for r in [32, 64, 96, 128, 255]:
    print(f"r={r:3d} -> occupancy {occupancy_vs_regs(r):.0%}")
# r= 32 -> 100%, r=64 -> 50%, r=96 -> 33%, r=128 -> 25%, r=255 -> 12%
```

### 5.2 launch_bounds 效果模拟

```python
def launch_bounds_effect(orig_r, target_blocks, threads=256, regs_per_sm=65536):
    """编译器压 r 让 blocks>=target. 看 spill 风险."""
    max_r = regs_per_sm // (target_blocks * threads)
    if orig_r > max_r:
        return f"压 r {orig_r}->{max_r}, 可能 spill (差 {orig_r-max_r} 个)"
    return f"r={orig_r} 已满足 target_blocks={target_blocks}, 无需压"

print(launch_bounds_effect(80, 8))   # 100% occupancy 目标
print(launch_bounds_effect(96, 8))   # 要压
```

### 5.3 ncu 命令

```bash
# achieved occupancy + 限制资源 + spill 检测
ncu --kernel-name regex:attn \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_blocks,\
sm__sass_inst_executed_op_local_ld.sum \
    --set full python train.py
# 看 'Occupancy' 一栏: theoretical vs achieved
# 'Stall Reasons' 看 warp 为何等
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[SM utilization]]（occupancy 原理与公式）、[[GPU执行模型]]（warp 调度/延迟隐藏）、[[GPU内存层级]]（寄存器/smem 资源约束）。
- **下游（应用）**: [[ncu (Nsight Compute)]]（实测工具）、[[Roofline模型]]（先判 bound 再调 occupancy）、[[Triton kernel开发]]/[[CUTLASS与GEMM]]（用 launch_bounds/autotune 调 occupancy）、[[FlashAttention]]（高 occupancy + 少 HBM 访问的设计）、[[compute bottleneck]]/[[memory bottleneck]]（occupancy 帮判瓶颈类型）。
- **对比 / 易混**:
  - **theoretical vs achieved**：理论上界 vs 实测。调优看 achieved。
  - **occupancy vs throughput**：occupancy 高是必要非充分，compute-bound 时正相关，memory-bound 时弱相关。
  - **launch_bounds vs --maxrregcount**：per-kernel vs 全局限寄存器。前者细。


## 7. 常见误区与易错点

> [!warning] 误区 1：盲目追 100% occupancy
> memory-bound kernel 提 occupancy 无益（仍等带宽）。compute-bound 才受益。且为达 100% 压寄存器可能 spill 反劣化。要 [[Roofline模型]] 先判 bound。

> [!warning] 误区 2：用理论 occupancy 当真实性能
> 理论是上界，实测 achieved 常低 10-30%。调优以 ncu 实测为准。

> [!warning] 误区 3：忽视 spill
> 寄存器被 `launch_bounds` 压过头 → spill 到 local memory（HBM，慢），既降 occupancy 又增访存。ncu 看 `local_ld` 是否 >0。

> [!warning] 误区 4：只看 occupancy 不看 stall 原因
> occupancy 100% 但 throughput 低，可能是 warp 全在 stalled（等内存）。看 ncu 的 stall reasons 才知根因。

> [!warning] 误区 5：小 batch / 小 kernel 怪 occupancy
> 小 batch 的 grid 太小（waves <1），SM 本就填不满，不是 occupancy 公式的问题。提 batch 或融合算子才解决，不是调 launch_bounds。

> [!warning] 误区 6：忽视 block size 的 32 倍数
> block 不是 32 倍数 → 末位 warp 不足 32，仍占调度槽但部分线程 idle，等效浪费 occupancy。常用 128/256/512。


## 8. 延伸细节

### 8.1 occupancy 的"甜点"

经验上 compute-bound kernel 的 achieved occupancy 50-75% 常比 100% 更优——100% 往往压寄存器致 spill 或减单线程效率。需 ncu 实测 throughput 找甜点，不能只看 occupancy 数字。

### 8.2 register allocation granularity

CUDA 以 4 个寄存器为单位分配（register allocation unit = 4），故 $r$ 不是连续而是 4 倍数。$r=33$ 实际占 36。这让 occupancy 阶梯式变化。

### 8.3 occupancy calculator

CUDA Toolkit 附 `CUDA_Occupancy_Calculator.xlsx` 或 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API，输入 kernel 配置直接算 theoretical occupancy。是手工调优的辅助工具。

### 8.4 LLM 推理的 occupancy 现状

decode 小 batch 的 occupancy 天然低（batch 小 grid 小）——但 decode 是 memory-bound（每 token 读大量 weight/KV），occupancy 低不影响带宽饱和。vLLM/SGLang 靠 continuous batching 提 batch + [[推理 CUDA Graph]] 减 launch，不硬扣 occupancy。prefill 大 batch 才 compute-bound，occupancy 重要。

### 8.5 Triton 的 autotune

[[Triton kernel开发]] 的 `@triton.autotune` 自动试不同 block size/num_warps/num_stages，实测 throughput 选最优——相当于自动化的 occupancy 调优。比手调 launch_bounds 更省心。

### 8.6 内容来源

occupancy 调优流程整理自 NVIDIA CUDA C++ Programming Guide 的 Hardware Multithreading 章节与 CUDA C++ Best Practices Guide 的 Occupancy Calculator，ncu metric 见 [[ncu (Nsight Compute)]]。

---
相关: [[Kernel性能模型]] | [[SM utilization]] | [[Roofline模型]] | [[ncu (Nsight Compute)]] | [[GPU执行模型]] | [[GPU内存层级]] | [[Triton kernel开发]] | [[CUTLASS与GEMM]] | [[FlashAttention]] | [[compute bottleneck]] | [[memory bottleneck]]
