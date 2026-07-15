# ncu (Nsight Compute)

> **所属章节**: [[Nsight Compute]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: ncu / Nsight Compute / nsight-compute / kernel profiler
> **难度**: 中高（需懂 [[GPU执行模型]] + [[Roofline模型]] + [[occupancy分析]]）


## 1. 一句话定义

**ncu（Nsight Compute）** 是 NVIDIA 的 **kernel 级 profiler**——对一个（或多个）kernel 跑详细指标采集，输出**寄存器/occupancy、Memory Chart（roofline 可视化）、warp stall 原因、缓存命中率、指令吞吐、tensor core 利用率**等，定位**单 kernel 为什么慢、慢在哪**。与 [[nsys (Nsight Systems)]]（系统级 trace，宏观找哪个 kernel 慢、谁等谁）互补：**nsys 找"哪个慢"，ncu 查"为什么慢"**。2026 年 NVIDIA profiling 主线是 nsys + ncu，[[nvprof（旧）]] 已降级为历史（功能被 ncu 取代）。ncu 是 [[occupancy分析]]/[[Roofline模型]] 的实测工具，是 kernel 调优闭环的微观一环。

> [!note] 三句话定位
> - **是什么**：单 kernel 微观 profiler，报 occupancy/warp stall/Memory Chart/tensor core 利用率。
> - **为什么**：nsys 只说某 kernel 慢，不知为何；ncu 深入单 kernel 看瓶颈（算力？带宽？occupancy？stall？）。
> - **开销高**：ncu 重复跑 kernel 多次采多组 metric，慢 10-100×，只 profile 少量 kernel，不能全训练跑。先用 nsys 定位再 ncu 深入。


## 2. 为什么需要它（动机与背景）

### 2.1 nsys 找到慢 kernel 后的下一步

[[nsys (Nsight Systems)]] 的系统 trace 显示"attention kernel 占 forward 60% 时间"——知道**哪个**慢。但为什么慢？是 occupancy 低、是 bandwidth 没饱和、是 warp 全 stall 等内存、是 tensor core 没用上？这些**kernel 内部细节** nsys 看不到，需 ncu。

### 2.2 GPU 性能的多维瓶颈

kernel 慢的可能因：

- **算力不够**（compute-bound，tensor core 没满载）。
- **带宽不够**（memory-bound，HBM 没饱和）。
- **occupancy 低**（SM 没塞满 warp，延迟隐藏差）。
- **warp stall**（warp 在等内存/同步/功能单元）。
- **缓存未命中**（L2 miss 高，打 HBM）。
- **指令吞吐低**（pipeline 没满）。

这些需不同 metric 判断。ncu 一次采集几百个 metric，分门别类报。

### 2.3 roofline 的工具化

[[Roofline模型]] 的 $I$、达峰带宽/算力——ncu 的 **Memory Chart** 直接画 kernel 在 roofline 上的位置，一眼看 bound 类型。不用手算 $I$。

### 2.4 nvprof 退役

[[nvprof（旧）]] 是老工具（prof 命令），功能被 ncu 取代。nvprof 不支持新 GPU（SM80+）的新特性（TMA、warp specialization、tensor core 细 metric）。2026 年主线 nsys + ncu，nvprof 仅老代码遗留。

### 2.5 与 nsys 的开销差

nsys 低开销（~5%），可全训练跑。ncu 高开销（慢 10-100×，因重复跑 kernel 采多组 metric），只 profile 选定 kernel。故**先用 nsys 全局找慢 kernel，再 ncu 深入该 kernel**，是标准流程。


## 3. 核心概念详解

### 3.1 基本用法

```bash
# profile 指定 kernel, full metric set
ncu --kernel-name regex:attn --set full python train.py
# 只 profile 前 N 次调用
ncu --kernel-id ::regex:attn: --launch-skip 10 --launch-count 1 python train.py
# 输出 .ncu-rep, 用 Nsight Compute UI 看
```

`--set full` 采全 metric（慢）；`--set basic` 轻量；`--set roofline` 只采 roofline 相关。`--target-processes all` 多进程。

### 3.2 核心视图

| 视图 | 内容 | 定位 |
|---|---|---|
| **Occupancy** | theoretical vs achieved occupancy、限制资源（reg/smem/hw） | [[occupancy分析]] |
| **Memory Chart** | roofline 位置、各层级带宽利用、HBM/L2/smem 字节 | [[Roofline模型]] bound 判定 |
| **Warp State** | stall 原因分布（long_sb/short_sb/imc_miss/sync） | warp 为何等 |
| **Instruction Statistics** | 各指令占比、吞吐、tensor core 利用率 | 算力利用 |
| **Source** | per-line 源码耗时（需 -lineinfo 编译） | 定位代码热点 |
| **Scheduler** | warp 调度器 active/idle 占比 | 调度损失 |

### 3.3 Occupancy 视图（连 [[occupancy分析]]）

报 theoretical（资源算出）vs achieved（实测）、限制资源（Registers/SharedMemory/BlockSize/Hardware）。若 achieved 远低 theoretical → 调度/分支问题。若受 Registers 限 → 用 launch_bounds 调。详见 [[occupancy分析]]。

### 3.4 Memory Chart（连 [[Roofline模型]]）

画 kernel 的算术强度 $I$ 在 roofline 上的位置：

- 落斜线段（低 $I$）→ memory-bound → 减 HBM 访问（tiling/融合）。
- 落水平段（高 $I$）→ compute-bound → 提算力（tensor core/occupancy）。
- 报各级带宽利用（HBM、L2、smem）——哪级是瓶颈一目了然。

不用手算 $I$，直接看图。

### 3.5 Warp State Stats

warp 的 cycle 分布：

| stall 原因 | 含义 | 优化方向 |
|---|---|---|
| `stall_long_sb` | 等长延迟（HBM） | 提 occupancy 隐藏 / tiling / 融合 |
| `stall_short_sb` | 等短延迟（smem/L1） | 减 bank conflict |
| `stall_imc_miss` | 等 instruction cache | 减 kernel 代码量 / 融合 |
| `stall_sync` | 等同步 | 减 `__syncthreads` |
| `stall_math_pipe` | 等功能单元 | 换算力更足的指令 |

occupancy 100% 但 throughput 低 → 看 stall：若 `stall_long_sb` 高 → warp 全在等 HBM → memory-bound（提 $I$ 才有用，不是提 occupancy）。

### 3.6 tensor core 利用率

`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` 报 tensor core 的 active 占峰值比。GEMM 应 > 60%；低于 30% 说明没喂饱 tensor core（shape 不对齐、occupancy 低、或非 GEMM）。

### 3.7 source-level profiling

编译时加 `-lineinfo`（保留行号），ncu 报 per-line 耗时与 metric，定位到源码哪行是热点。是 CUTLASS/Triton 调优的依据。

### 3.8 多 metric 的 collection

ncu 用 "rule-based" 自动建议：根据 metric 值给"occupancy 受寄存器限 → 试 launch_bounds"、"memory-bound → 减 HBM 访问"等提示。新手友好。

### 3.9 与 nsys 的协作流程

1. `nsys profile` 全训练 → 找最慢 kernel（如 attn 占 60%）。
2. `ncu --kernel-name regex:attn` 深入 → 看 Memory Chart（memory-bound？）+ stall（等 HBM？）+ occupancy。
3. 据 ncu 改 kernel（tiling/融合/launch_bounds）→ 重测 nsys 看 forward 时间是否降。

闭环：nsys（宏观）→ ncu（微观）→ 改 → nsys 验证。


## 4. 数学原理 / 公式

### 4.1 roofline 的实测

$$
I_{\text{measured}} = \frac{\text{FLOP (ncu sm__inst_executed_pipe_* )}}{\text{HBM bytes (dram__bytes_*)}}
$$

ncu 直接给 $I$、给达峰比例。不用手算。

### 4.2 achieved occupancy

$$
\eta_{\text{achieved}} = \frac{\text{sm__warps_active.avg}}{W_{\max}}
$$

ncu metric `sm__warps_active.avg.pct_of_peak_sustained_active`。

### 4.3 stall 比例

$$
\text{stall ratio}_r = \frac{\text{warp cycles in stall}_r}{\text{total warp cycles}}
$$

各 stall 原因占比，主因 + bound 类型共同决定优化方向。

### 4.4 带宽利用率

$$
\text{HBM 利用率} = \frac{\text{dram__bytes.sum.per_second}}{B_{\text{peak}}}
$$

低 → 没喂饱 HBM（launch 太小、访问不合并）；高 → 带宽饱和（memory-bound，要减 byte 不是增带宽）。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟 ncu metric 解读

```python
def ncu_diagnose(achieved_occ, stall_long_sb, hbm_util, tensor_core_util):
    """据 ncu metric 判瓶颈."""
    if hbm_util > 0.7:
        return "memory-bound: 减 HBM 访问 (tiling/融合), 不是提 occupancy"
    if tensor_core_util < 0.3 and achieved_occ > 0.6:
        return "compute-bound 但 tensor core 没满: 检查 shape 对齐 / 算子选 tensor core 路径"
    if achieved_occ < 0.4:
        return "occupancy 低: 寄存器/smem 压, 调 launch_bounds"
    if stall_long_sb > 0.5:
        return "warp 全等 HBM: memory-bound, 提 I"
    return "无明显瓶颈, 查指令吞吐 / cache miss"

print(ncu_diagnose(0.3, 0.1, 0.9, 0.8))    # memory-bound
print(ncu_diagnose(0.95, 0.6, 0.3, 0.5))   # warp 等 HBM
print(ncu_diagnose(0.4, 0.1, 0.3, 0.2))    # occupancy 低
```

### 5.2 真实 ncu 命令

```bash
# 1. 先 nsys 找慢 kernel
nsys profile -t cuda,nvtx --stats=true -o train python train.py
# 假设发现 attn_kernel 最慢

# 2. ncu 深入 attn_kernel
ncu --kernel-name regex:attn_kernel \
    --set full \
    --launch-skip 10 --launch-count 1 \
    -o attn_profile \
    python train.py

# 3. Nsight Compute UI 打开 attn_profile.ncu-rep
#    - 看 Occupancy (theoretical vs achieved)
#    - 看 Memory Chart (roofline 位置)
#    - 看 Warp State (stall 原因)
```

### 5.3 关键 metric 速查

```bash
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active,\   # achieved occupancy
  dram__bytes.sum.per_second,\                            # HBM 带宽
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\  # tensor core 利用
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\       # 全局访问量 (算 I)
  launch__occupancy_limit_blocks \                        # 限制资源
  --kernel-name regex:attn python train.py
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GPU执行模型]]（warp/SM/tensor core）、[[Roofline模型]]（Memory Chart 的理论）、[[occupancy分析]]（Occupancy 视图）、[[GPU内存层级]]（各级带宽 metric）。
- **下游（应用）**: [[SM utilization]]（occupancy 实测）、[[Triton kernel开发]]（autotune 用 throughput，ncu 诊断为何慢）、[[CUTLASS与GEMM]]（GEMM 调优）、[[FlashAttention]]（attention IO 验证）、[[fused kernel]]（融合前后 IO 对比）、[[ncu (Nsight Compute)]]（本章总览）。
- **对比 / 易混**:
  - **ncu vs [[nsys (Nsight Systems)|nsys]]**：ncu 单 kernel 微观（为什么慢），nsys 系统宏观（哪个慢、谁等谁）。先 nsys 再 ncu。
  - **ncu vs [[nvprof（旧）]]**：ncu 是 nvprof 的继任者，支持新 GPU 新特性，nvprof 退役。
  - **ncu vs [[torch profiler]]**：torch profiler 是应用层（PyTorch op 视角），ncu 是 kernel 层（硬件 metric）。torch profiler 定位 op，ncu 深入 kernel。


## 7. 常见误区与易错点

> [!warning] 误区 1：用 ncu 跑全训练
> ncu 高开销（慢 10-100×），全训练跑不动。先 nsys 找慢 kernel，再 ncu 只 profile 该 kernel（`--launch-count 1`）。

> [!warning] 误区 2：只看 occupancy 不看 stall
> occupancy 100% 但慢 → 看 Warp State。warp 全 stall（等内存）说明 memory-bound，提 occupancy 无用。多维 metric 综合判。

> [!warning] 误区 3：忽视 Memory Chart
> Memory Chart 直接给 roofline 位置，一眼判 compute/memory bound。不用手算 $I$。是 ncu 最高价值视图之一。

> [!warning] 误区 4：编译不带 -lineinfo
> 不带 -lineinfo ncu 无法报 per-line 源码耗时，定位不到具体代码行。CUDA/Triton 编译要加。

> [!warning] 误区 5：混淆 ncu 与 nsys 的层次
> nsys 是系统 trace（CPU/CUDA/NCCL/OS 全采，找宏观瓶颈）；ncu 是单 kernel 深入（寄存器/occupancy/stall）。名字像但层次不同。详见 [[nsys (Nsight Systems)]] 的对比。

> [!warning] 误区 6：metric 一次全采
> `--set full` 采几百 metric，慢且报告巨大。调优时按需选 metric（如只看 roofline 用 `--set roofline`），或 `--section` 选视图。


## 8. 延伸细节

### 8.1 ncu 的 metric 命名

`sm__warps_active.avg.pct_of_peak_sustained_active` 这种 `domain__subdomain.metric.rollup.operation` 命名复杂，但 Nsight Compute UI 有 friendly name + rule-based 建议，新手可只看 UI 的高层视图。

### 8.2 与 [[torch profiler]] 的集成

torch profiler 的 `record_shapes`/`with_stack` 给 PyTorch op 与源码行映射，ncu 给 kernel metric。两者用同一 kernel name 对齐，可从"PyTorch op → kernel → ncu metric"全链路定位。详见 [[torch profiler]]。

### 8.3 多 GPU / 分布式的 ncu

ncu 主要单卡单 kernel。多卡分布式瓶颈（NCCL 通信）用 nsys 看（NCCL trace）。ncu 偶尔用于 NCCL kernel 的微观，但少。

### 8.4 ncu 的 replay 机制

ncu 对同一 kernel 跑多遍采不同 metric（因硬件 counter 有限，一次采不全）。故 kernel 必须可重复跑（确定性）。含随机性的 kernel 要固定 seed 或 ignore。

### 8.5 内容来源

ncu 用法整理自 NVIDIA Nsight Compute 文档与 `--help`，Memory Chart/roofline 集成见 Nsight Compute tutorial，与 [[nsys (Nsight Systems)]]/[[nvprof（旧）]] 的分工见 NVIDIA profiling guide。

---
相关: [[Nsight Compute]] | [[nsys (Nsight Systems)]] | [[nvprof（旧）]] | [[torch profiler]] | [[GPU执行模型]] | [[Roofline模型]] | [[occupancy分析]] | [[SM utilization]] | [[GPU内存层级]] | [[Triton kernel开发]] | [[CUTLASS与GEMM]] | [[FlashAttention]] | [[fused kernel]]
