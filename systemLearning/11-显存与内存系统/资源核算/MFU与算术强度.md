# MFU与算术强度

> **所属章节**: [[资源核算]]
> **所属模块**: [[11-显存与内存系统]]
> **别名**: MFU / HFU / arithmetic intensity / 算术强度 / 模型 FLOPs 利用率 / 训练效率
> **难度**: 中（需懂 [[FLOPs计算]] + GPU 算力/带宽特性）


## 1. 一句话定义

**MFU（Model FLOPs Utilization，模型 FLOPs 利用率）** 是衡量训练算力利用效率的核心指标——实际有用的训练 FLOP/秒 ÷ GPU 理论峰值 FLOP/秒；它依赖**算术强度**（arithmetic intensity = FLOP ÷ 内存访问字节数）判定任务落在 compute-bound 还是 memory-bound 区，再据 [[Roofline模型]] 推出可达利用率。MFU 工业界 **40–60% 算优秀**，是训练时间/成本估算的关键参数（训练时间 $\approx C / (\text{GPU数} \times \text{峰值} \times \text{MFU})$）。

> [!note] 三句话定位
> - **是什么**：MFU = 实际有用 FLOP/秒 ÷ GPU 峰值 FLOP/秒，衡量训练算力利用率（40-60% 优秀）。
> - **为什么有上限**：算术强度低 → memory-bound → 带宽不够喂饱算力 → MFU 上不去。
> - **有什么用**：训练时间/成本估算的分母、诊断训练瓶颈（算力 vs 带宽）、对比训练框架效率。


## 2. 为什么需要它（动机与背景）

### 2.1 峰值算力是"纸面"，实际达不到

A100 bf16 标称 312 TFLOPs，但训练时不可能跑满——有通信开销、kernel launch、memory bandwidth 瓶颈、overlap 不完全等。实际有用算力可能只有峰值的 40-50%。**MFU 就是把这个"折扣"量化**，让你用峰值估算时不至于过度乐观。

### 2.2 训练时间/成本估算的必备参数

给定训练总 FLOP $C$（见 [[FLOPs计算]]），训练时间：

$$
T = \frac{C}{\text{GPU数} \times \text{峰值 FLOPs} \times \text{MFU}}
$$

没有 MFU 就只能用峰值算（过度乐观，低估时间 2-3 倍）。MFU 把"理论"拉回"实际"。报训练预算时 MFU 是必填项。

### 2.3 MFU 的上限由算术强度决定

为什么 MFU 不能 100%？因为算力的"喂料"——数据从显存搬进计算单元——受**内存带宽**限制。如果一个任务每读 1 字节只做很少 FLOP（算术强度低），数据喂不过来，算力闲置，MFU 低。这是 [[Roofline模型]] 的核心：算术强度决定任务是 compute-bound 还是 memory-bound，从而决定 MFU 上限。


## 3. 核心概念详解

### 3.1 算术强度（arithmetic intensity）

$$
\text{算术强度} = \frac{\text{FLOP}}{\text{内存访问字节数}}
$$

- 单位：FLOP/byte。
- **高** → compute-bound（算力是瓶颈，数据够用）。
- **低** → memory-bound（带宽是瓶颈，算力等数据）。
- 分界点 = GPU 的**算力 ÷ 带宽**（见 [[Roofline模型]]）。例：A100 算力 312 TFLOPs / 带宽 2 TB/s ≈ 156 FLOP/byte。算术强度 > 156 才 compute-bound。

### 3.2 Roofline 模型（简述，详见 [[Roofline模型]]）

$$
\text{可达 FLOP/s} = \min(\text{峰值 FLOP/s},\ \text{带宽} \times \text{算术强度})
$$

- 算术强度低时，可达算力 = 带宽 × 算术强度（memory-bound 段，随算术强度线性增）。
- 算术强度高时，可达算力 = 峰值（compute-bound 段，封顶）。
- 转折点 = 峰值 ÷ 带宽（"ridge point"）。

MFU = 可达 FLOP/s ÷ 峰值 = min(1, 带宽 × 算术强度 / 峰值)。故 memory-bound 时 MFU < 1 且随算术强度增；compute-bound 时 MFU → 1（实际仍有其他开销，到不了 1）。

### 3.3 MFU 定义

$$
\text{MFU} = \frac{\text{实际有用的训练 FLOP/秒}}{\text{GPU 理论峰值 FLOP/秒}}
$$

- "有用 FLOP"指真正用于前向+反向的 FLOP（$6ND$ 那部分），**不含**通信、重算（gradient checkpointing 的额外前向）、kernel launch 等开销。
- 工业界优秀值 **40-60%**（大模型训练）。低于 30% 通常有优化空间；超过 60% 接近物理极限。
- 量纲：百分比，常以小数（0.45）或百分数（45%）表。

### 3.4 HFU（Hardware FLOP Utilization）

$$
\text{HFU} = \frac{\text{GPU 实际执行的总 FLOP/秒（含重算等)}}{\text{峰值}}
$$

- 与 MFU 区别：HFU 含**重算**（gradient checkpointing 反向时重做前向，多算一遍 FLOP）等"非有用但实际执行"的 FLOP；MFU 只计有用部分。
- 关系：用 gradient checkpointing 时 HFU > MFU（多算的 33% 重算 FLOP 进 HFU 不进 MFU）。不用 ckpt 时 HFU ≈ MFU。
- 报框架效率时注意区分：HFU 高可能是重算多（不一定好），MFU 才是"有用算力利用率"。

### 3.5 训练时间估算

$$
\boxed{T = \frac{C}{\text{GPU数} \times \text{峰值 FLOPs} \times \text{MFU}} = \frac{6ND}{\text{有效算力}}}
$$

- $C = 6ND$（见 [[FLOPs计算]]）。
- 有效算力 = GPU数 × 峰值 × MFU（"打折后的实际算力"）。
- 例：2048 张 A100（312 TFLOPs bf16），MFU 0.45 → 有效算力 $= 2048 \times 312 \times 0.45 \approx 2.88 \times 10^{14}$ FLOP/s。

### 3.6 各 GPU 的峰值与带宽（对照）

| GPU | 峰值 FLOPs（bf16/fp8） | HBM 带宽 | ridge point | 典型训练 MFU |
|---|---|---|---|---|
| A100 40/80GB | 312 / 624 TFLOPs | 1.5-2.0 TB/s | ~156-208 FLOP/byte | 40-50% |
| H100 80GB | 990 / 1979 TFLOPs | 3.35 TB/s | ~296 FLOP/byte | 45-55% |
| B200 | ~2.25 / ~4.5 PFLOPs | 8 TB/s | ~281 FLOP/byte | 50-60%（待核实） |

> 数值来自 NVIDIA 官方 spec，实际随精度/稀疏开关变化；MFU 是经验范围，非官方保证。

### 3.7 训练各阶段的算术强度

| 阶段 | 算术强度 | bound | MFU 表现 |
|---|---|---|---|
| 大 batch 前向/反向（FFN/attention 投影） | 高（大 GEMM） | compute-bound | 高，MFU 接近上限 |
| 小 batch decode / 长序列 attention | 低 | memory-bound | 低 |
| Optimizer step、归一化、激活函数 | 极低 | memory-bound | 低（小 kernel） |
| 通信（allreduce） | — | 通信 bound | 不算 FLOP，但占墙钟时间 |

训练 MFU 是这些阶段的加权平均，故提升 MFU 的关键是减少低算术强度段的占比（fuse kernel、提 batch、overlap 通信）。


## 4. 数学原理 / 公式

### 4.1 算术强度

$$
I = \frac{\text{FLOP}}{\text{Bytes}}
$$

例：矩阵乘 $A_{m\times n} B_{n\times p}$（fp16）：
- FLOP $= 2mnp$。
- Bytes 读 $A$ ($mn \cdot 2$) + $B$ ($np \cdot 2$) + 写 $C$ ($mp \cdot 2$) $\approx 2(mn + np + mp) \approx 2np$（$m$ 大时 $A$ 一次读多次复用）。
- 算术强度 $\approx 2mnp / (2np) = m$（大 $m$ 时高，compute-bound）。

故**大矩阵乘算术强度高**（compute-bound），**小矩阵乘/向量乘算术强度低**（memory-bound）。这是 batch 越大 MFU 越高的根因。

### 4.2 Roofline

$$
\text{Achieved FLOP/s} = \min\left(\text{Peak},\ \text{Bandwidth} \times I\right)
$$

### 4.3 MFU

$$
\text{MFU} = \frac{\text{Achieved useful FLOP/s}}{\text{Peak}} = \min\left(1,\ \frac{\text{Bandwidth} \times I}{\text{Peak}}\right) \times \text{效率系数}
$$

效率系数 < 1，含 kernel launch、overlap 不完全、小算子开销等。

### 4.4 训练时间

$$
T = \frac{C}{N_{\text{gpu}} \times \text{Peak} \times \text{MFU}} = \frac{6ND}{N_{\text{gpu}} \times \text{Peak} \times \text{MFU}}
$$

### 4.5 Llama-2 7B 训练时间算例

$N = 6.7$B, $D = 2$T tokens, $C = 8.04 \times 10^{22}$ FLOP。

2048 张 A100（Peak $= 312$ TFLOPs $= 3.12 \times 10^{14}$），MFU $= 0.45$：

$$
T = \frac{8.04 \times 10^{22}}{2048 \times 3.12 \times 10^{14} \times 0.45} = \frac{8.04 \times 10^{22}}{2.88 \times 10^{17}} \approx 2.79 \times 10^5 \text{ s} \approx 77.5 \text{ 小时}
$$

约 3.2 天。这是纯计算时间，不含 checkpoint 保存、数据加载停顿、故障恢复等（实际工程要乘 1.2-1.5 系数）。


## 5. 代码示例

```python
def training_time(N, D, n_gpu, peak_flops, mfu):
    """训练时间估算. C=6ND, 有效算力=GPU数*峰值*MFU."""
    C = 6 * N * D
    effective = n_gpu * peak_flops * mfu
    return C / effective, C

def mfu_from_time(N, D, n_gpu, peak_flops, measured_seconds):
    """从实测时间反推 MFU."""
    C = 6 * N * D
    achieved = C / measured_seconds           # 实际 FLOP/s
    return achieved / (n_gpu * peak_flops)

# Llama-2 7B, 2T tokens, 2048 A100
A100_PEAK = 312e12   # bf16
t, C = training_time(6.7e9, 2e12, 2048, A100_PEAK, 0.45)
print(f'7B 训练 2T tokens, 2048 A100, MFU=0.45:')
print(f'  总 FLOP C = {C:.2e}')
print(f'  估算时间 = {t/3600:.1f} 小时')

# MFU 敏感性：0.30 vs 0.50 vs 0.60
for mfu in [0.30, 0.45, 0.50, 0.60]:
    t, _ = training_time(6.7e9, 2e12, 2048, A100_PEAK, mfu)
    print(f'  MFU={mfu}: {t/3600:.1f} 小时')

# 算术强度分界判断
def is_compute_bound(arith_intensity, peak_flops, bandwidth):
    """ridge point = peak / bandwidth."""
    ridge = peak_flops / bandwidth
    return arith_intensity >= ridge, ridge

A100_BW = 2e12   # 2 TB/s
cb, ridge = is_compute_bound(200, A100_PEAK, A100_BW)
print(f'A100 ridge point = {ridge:.0f} FLOP/byte; I=200 -> compute_bound={cb}')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[FLOPs计算]]（$C=6ND$ 是分子）、[[参数量计算]]（$N$）、[[GPU硬件结构]]（峰值/带宽/SMEM 的来源）、[[Roofline模型]]（MFU 上限的推导工具）。
- **下游（应用）**: 训练时间/成本估算（本笔记核心）、训练框架效率对比（Megatron-LM vs DeepSpeed vs 自研）、[[SM utilization]]（GPU 微观利用率，MFU 是宏观对应）、[[overlap strategy]]（提 MFU 的手段）、[[compute bottleneck]]/[[memory bottleneck]]（MFU 低时的诊断目标）。
- **对比 / 易混**:
  - **MFU vs [[SM utilization\|SM utilization]]**：MFU 是"模型有用 FLOP / 峰值"（宏观，模型视角）；SM utilization 是"SM 在活跃的时间占比"（微观，GPU 视角）。SM 高不一定 MFU 高（SM 活跃但算的是低效 kernel），MFU 高一定 SM 高。
  - **MFU vs HFU**：MFU 只计有用 FLOP（$6ND$），HFU 计 GPU 实际执行的全部 FLOP（含 gradient checkpointing 重算）。用 ckpt 时 HFU > MFU。
  - **MFU vs 硬件利用率**：MFU 是 FLOP 维度，硬件利用率还可含带宽利用率、SM 占用率等多维。MFU 是训练场景最常报的一个数。


## 7. 常见误区与易错点

> [!warning] 误区 1：用峰值算训练时间
> 不乘 MFU 直接用峰值，会低估训练时间 2-3 倍（峰值 312 TFLOPs 实际只跑出 ~140 TFLOPs 有用算力）。报预算必须用 $T = C / (\text{GPU数} \times \text{峰值} \times \text{MFU})$，MFU 取 0.40-0.50 保守。

> [!warning] 误区 2：MFU 越高越好（忽视代价）
> MFU 高可能是靠堆大 batch（提算术强度），但大 batch 可能影响收敛（需调 lr/warmup）或放不下显存。也可是减少 gradient checkpointing（提 HFU 但显存爆）。MFU 是效率指标，不是唯一目标，要与收敛性、显存权衡。

> [!warning] 误区 3：混淆 MFU 与 HFU
> 用 gradient checkpointing 时，GPU 实际多算一遍前向（重算），HFU 含这 33% 额外 FLOP，HFU > MFU。报框架效率时若用 HFU 充 MFU，会高估有用算力。**训练时间公式用 MFU（有用 FLOP），不是 HFU**。

> [!warning] 误区 4：以为 MFU 能到 100%
> 不可能。即使 compute-bound，也有 kernel launch、归一化/激活等小算子（memory-bound）、通信（非 FLOP）、同步等开销。大模型训练 MFU 上限约 55-60%（H100 优秀值），A100 约 50%。报 70%+ 通常口径不对（可能用 HFU 或 fp8 稀疏峰值）。

> [!warning] 误区 5：忽略算术强度的 batch 依赖
> 同模型同 GPU，batch 小时算术强度低（矩阵小，每参数读的 byte 多），MFU 低；batch 大时算术强度高，MFU 高。故"训练 MFU = 0.45"必须注明 batch 配置，否则不可比。小 batch 训练 MFU 天然低，不是框架烂。

> [!warning] 误区 6：把通信开销算进 MFU
> MFU 的分子是"有用训练 FLOP"（$6ND$），**不含** allreduce 通信（通信不是 FLOP）。但通信占墙钟时间，会拉低"墙钟 MFU"（用墙钟算的 MFU）。区分：理想 MFU（无通信）vs 墙钟 MFU（含通信停顿）。多卡训练墙钟 MFU < 单卡 MFU，差值是通信代价。


## 8. 延伸细节

### 8.1 提升 MFU 的手段

1. **提 batch/序列并行度**：增大 GEMM 尺寸，提算术强度，向 compute-bound 靠。
2. **[[overlap strategy]]**：反向计算与 allreduce 通信 overlap，藏通信于计算。
3. **kernel fusion**（见 [[算子融合]]）：把归一化+激活+残差等小算子合成一个 kernel，减 kernel launch 和中间读写。
4. **[[FlashAttention]]**：减 attention 的 HBM 读写（不改 FLOP，但减 memory-bound 段时间）。
5. **降低非有用 FLOP**：选择性 gradient checkpointing（只重算大的，不全重算），平衡显存与 HFU。
6. **Tensor Core 友好的形状**：对齐到 16/8 的倍数，让 GEMM 走 Tensor Core 高效路径。

### 8.2 MFU 与 [[SM utilization]] 的关系

SM utilization 是 GPU 微观指标（SM 活跃时间占比），MFU 是模型宏观指标（有用 FLOP/峰值）。关系：

- MFU 高 → SM utilization 必高（算力在跑）。
- SM utilization 高 → MFU 不一定高（可能跑低效 kernel，或重算等非有用 FLOP）。
- 诊断顺序：先看 MFU（宏观够不够），低再看 SM utilization（微观忙不忙），SM 高但 MFU 低 → 在跑低效/非有用 FLOP，找 kernel 瓶颈。

### 8.3 单卡 vs 多卡 MFU

- 单卡 MFU：无通信，纯计算效率，大模型单卡 GEMM 能到 50-60%。
- 多卡 MFU（墙钟）：含 allreduce 通信停顿，低于单卡。差值 = 通信代价。
- 提多卡 MFU 靠 [[overlap strategy]] 和减少通信量（[[ZeRO (DeepSpeed)]]-3 的 all-gather 反而增通信，拉低墙钟 MFU，是显存/通信的 trade-off）。

### 8.4 推理的 MFU

推理（尤其 decode）batch 小、算术强度低，MFU 通常很低（个位数 %）。这是推理 throughput 低的根因，不是框架烂。提推理效率靠 continuous batching（见 [[continuous batching]]）、prefix caching、speculative decoding 等。推理报"吞吐"比报 MFU 更常用。

### 8.5 fp8 训练的 MFU

H100/B200 支持 fp8（峰值翻倍，如 H100 fp8 1979 TFLOPs），但 fp8 训练 MFU 通常低于 bf16（数值敏感，需更多保精度操作，且不是所有算子都能 fp8）。故"fp8 峰值 2×"不等于"训练时间减半"，实际收益约 1.3-1.7×（待核实，依赖实现）。

### 8.6 内容来源

本笔记核心（算术强度、MFU 公式、训练时间估算、40-60% 业界范围、ridge point）整理自 CS336 Lecture 2（PyTorch 与资源核算），与 Lecture 5（GPU）的 Roofline 内容互补。

---
相关: [[资源核算]] | [[参数量计算]] | [[FLOPs计算]] | [[Roofline模型]] | [[GPU硬件结构]] | [[SM utilization]] | [[算子融合]] | [[overlap strategy]] | [[compute bottleneck]] | [[memory bottleneck]] | [[ZeRO (DeepSpeed)]]
