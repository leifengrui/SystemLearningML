# Roofline模型

> **所属章节**: [[Kernel性能模型]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: Roofline / 屋顶线模型 / 算术强度 / arithmetic intensity / ridge point / compute-bound vs memory-bound
> **难度**: 中（需懂 [[GPU内存层级]] + 算力/带宽特性）


## 1. 一句话定义

**Roofline 模型** 是用一张图判定一个 kernel 在某 GPU 上能跑多快的性能模型——横轴是**算术强度** $I$（每读 1 byte 内存做多少 FLOP，单位 FLOP/byte），纵轴是**可达算力**（FLOP/s），曲线由两段"屋顶"构成：低 $I$ 时受**内存带宽**限（$\text{FLOP/s} = \text{带宽}\times I$，斜线段，memory-bound），高 $I$ 时受**峰值算力**限（水平段，compute-bound），转折点叫 **ridge point** $R = \text{峰值}/\text{带宽}$。它一句话回答"我的 kernel 是 compute-bound 还是 memory-bound、瓶颈在哪"，是 kernel 优化的第一步诊断，也是 [[MFU与算术强度]] 在 kernel 维度的物理依据。

> [!note] 三句话定位
> - **是什么**：算术强度 $I$ × 带宽 / 峰值 → 一张屋顶图，判 compute vs memory bound。
> - **为什么**：优化前必须知道瓶颈在算力还是带宽，否则白费劲——compute-bound 优化访存没用，memory-bound 加算力没用。
> - **ridge point**：$R = \text{峰值}/\text{带宽}$，A100 约 156 FLOP/byte，$I > R$ 才 compute-bound。


## 2. 为什么需要它（动机与背景）

### 2.1 优化前先判瓶颈

kernel 慢有两大类因：**算力不够**（compute-bound，GPU 算单元满载，加算力或换高效 kernel 才能快）或**带宽不够**（memory-bound，算单元等数据，减内存访问或提带宽才能快）。两类因优化方向**正交甚至相反**：

- compute-bound → 提 tensor core 利用、增 batch/算术强度、减 kernel 数（融合）。
- memory-bound → 减 HBM 访问（tiling/融合）、提访问合并、用 smem cache。

判错瓶颈 → 优化白做。Roofline 是**先判瓶颈的工具**。

### 2.2 算力与带宽的物理上限

GPU 有两个物理上限：峰值算力 $P$（FLOP/s，由 tensor core / CUDA core 数 × 频率）与峰值带宽 $B$（byte/s，由 HBM）。kernel 的可达性能不可能超两者。Roofline = $\min(P, B\cdot I)$。

### 2.3 与 [[MFU与算术强度]] 的分工

[[MFU与算术强度]] 是**模型宏观视角**——用 $I$ 推 MFU、估算训练时间，关心"这模型这 batch 在这 GPU 能跑多少 MFU"。Roofline 是 **kernel/GPU 视角**——判单 kernel 的瓶颈、指导 kernel 级优化（tile 尺寸、访存合并、算子融合）。两者公式同源（都 $\min(P, BI)$），视角不同。


## 3. 核心概念详解

### 3.1 算术强度 I

$$
I = \frac{\text{FLOP}}{\text{Bytes (HBM 访问)}}
$$

- 大 GEMM（$mnp$ FLOP / $2np$ bytes，$m$ 大时）→ $I \approx m$ → 高 → compute-bound。
- elementwise（少量 FLOP / 大量 byte）→ $I \approx 1$ → 低 → memory-bound。
- attention（naive $O(N^2)$，HBM 多次读写）→ $I$ 低；[[FlashAttention]] 通过 tiling 把 HBM 访问从 $O(N^2)$ 降到 $O(N)$ → $I$ 升 → 趋 compute-bound。

### 3.2 Roofline 公式

$$
\text{Achieved FLOP/s} = \min\left(P,\ B \cdot I\right)
$$

- $I < R$（ridge point）→ $B \cdot I < P$ → 受带宽限（斜线段），加算力无益、减内存访问有益。
- $I > R$ → $P < B \cdot I$ → 受算力限（水平段），减内存访问无益、加算力有益。

### 3.3 ridge point

$$
R = \frac{P}{B}
$$

是斜线与水平的交点。$I = R$ 时刚好同时打满算力与带宽。

| GPU | 峰值 $P$（bf16） | 带宽 $B$（HBM） | ridge $R$ |
|---|---|---|---|
| A100 80GB | 312 TFLOPs | 2.0 TB/s | ~156 FLOP/byte |
| H100 80GB | 990 TFLOPs | 3.35 TB/s | ~296 FLOP/byte |
| B200 | ~2.25 PFLOPs | ~8 TB/s | ~281 FLOP/byte（待核实） |

> H100 的 $R$ 比 A100 高 → 算力涨比带宽快 → 同 kernel 在 H100 上更易 memory-bound（更难喂饱算力）。这是新一代 GPU 越来越 memory-bound 的根因。

### 3.4 各算子的 $I$ 估算

| 算子 | FLOP | Bytes | $I$ | bound |
|---|---|---|---|---|
| GEMM $m\times n\times p$ (m 大) | $2mnp$ | $\approx 2np$ (A 复用) | $\approx m$ | compute (m> R) |
| GEMV (m×1) | $2mn$ | $2mn + 2n$ | ~1 | memory |
| elementwise add | 1 | 12 (读2写1) | 1/12 | memory |
| RMSNorm (向量) | ~O(N) | O(N) | 低 | memory |
| naive attention (seq $N$) | $O(N^2)$ FLOP, 但 HBM $O(N^2)$ 读 | — | 中低 | memory (naive) |
| FlashAttention | $O(N^2)$ FLOP, HBM $O(N)$ | — | 高 | 趋 compute |
| optimizer step | O(N) | O(N) × 多份 | 低 | memory |

### 3.5 与 occupancy 的关系

$I$ 决定 bound 类型，occupancy 决定延迟隐藏。memory-bound kernel 即使 occupancy 100% 也受带宽限（详见 [[SM utilization]] 的"occupancy 高 ≠ 性能好"）。Roofline 先判 bound，occupancy 再判是否塞满 SM——**两者正交**。

### 3.6 ncu 的 roofline 可视化

[[ncu (Nsight Compute)]] 的 Memory Chart 直接画 kernel 的 $I$ 与达峰带宽/算力的位置，一眼看瓶颈。是 roofline 的工具化。见 [[ncu (Nsight Compute)]] 的 3.x 节。

### 3.7 多 GPU 的 roofline

跨 GPU 通信不走 HBM，走 NVLink（带宽 $B_{\text{nvlink}}$）或 PCIe。通信的"算术强度"是 FLOP/byte 传输，受 NVLink 带宽限。这是 [[alpha-beta性能模型]] 的来源。


## 4. 数学原理 / 公式

### 4.1 GEMM 的 $I$ 推导

矩阵乘 $C = AB$，$A \in \mathbb{R}^{m\times n}, B \in \mathbb{R}^{n\times p}, C \in \mathbb{R}^{m\times p}$。

- FLOP $= 2mnp$（每元素 $n$ 次 mul-add = $2n$ FLOP，$mp$ 元素）。
- Bytes：读 $A$ ($mn$)，读 $B$ ($np$)，写 $C$ ($mp$)，每元素 $s$ 字节（fp16 $s=2$）：$\text{Bytes} = s(mn + np + mp)$。
- 大 $m$ 时（$m \gg n,p$）：$mn, mp$ 主，但 $A$ 每 element 被 $p$ 次复用 → 实际 HBM 读 $A$ 一次（若 tiling 好）：$\text{Bytes} \approx s(np + mp) \approx s \cdot np$（$p$ 大时）。$I \approx 2mnp / (s\cdot np) = 2m/s$。

$m$ 越大（batch/序列并行度高），$I$ 越高 → compute-bound。**这是 batch 越大 MFU 越高的根因**。

### 4.2 elementwise 的 $I$

加法 $C = A + B$，每元素 1 FLOP，读 2、写 1 → $s$ 字节 × 3。$I = 1/(3s)$，fp16 时 $1/6 \approx 0.17$，远 < $R$ → 强 memory-bound。

### 4.3 ridge point 的意义

$I = R$ 时 $B \cdot I = P$，带宽与算力同时打满。$I$ 升过 $R$ → 算力先饱和（compute-bound），带宽有冗余。$I$ 降过 $R$ → 带宽先饱和（memory-bound），算力有冗余。

### 4.4 优化的"屋顶移动"

- 减 HBM 访问（tiling/融合，如 [[FlashAttention]]）→ $I$ 右移 → 从斜线段移向水平段。
- 提峰值利用（tensor core / occupancy）→ $P$ 段上移（但只在 compute-bound 段有效）。
- 提带宽利用（访问合并、coalesced）→ $B$ 斜线段斜率升（但只在 memory-bound 段有效）。

### 4.5 实测性能 = roofline × 效率系数

实际 kernel 达不到 roofline 上限，因 occupancy 不足、访问不合并、warp divergence、launch 开销等。$\text{Achieved} = \text{Roofline} \times \eta$，$\eta < 1$。优化就是提 $\eta$ 向 1 靠。

### 4.6 训练/推理各阶段的 bound

| 阶段 | $I$ | bound | 优化方向 |
|---|---|---|---|
| prefill 大 batch attention/FFN | 高 | compute | tensor core、提 batch |
| decode 小 batch | 低 | memory | continuous batching、prefix cache |
| optimizer step | 低 | memory | 融合、CPU offload |
| all-reduce | — | 通信 bound | overlap、减通信量 |


## 5. 代码示例（可选）

### 5.1 纯 Python roofline 计算

```python
def roofline(P_peak, B_hbm, I):
    """P_peak 峰值 FLOP/s, B_hbm 带宽 byte/s, I 算术强度 FLOP/byte."""
    R = P_peak / B_hbm
    achieved = min(P_peak, B_hbm * I)
    bound = 'compute' if I >= R else 'memory'
    return achieved, bound, R

# A100 bf16, I 取典型值
A100_P, A100_B = 312e12, 2e12
for label, I in [('大GEMM(m=4096)', 4096), ('GEMV', 1), ('elementwise', 0.17),
                 ('naive attn(N=8192)', 2), ('FlashAttn', 200)]:
    a, b, R = roofline(A100_P, A100_B, I)
    print(f"{label:25s} I={I:6.1f} -> {a/1e12:6.1f} TFLOPs, {b}-bound (R={R:.0f})")
```

### 5.2 GEMM 算术强度推导

```python
def gemm_intensity(m, n, p, bytes_per_elem=2):
    flop = 2*m*n*p
    # 假设 A tiling 复用良好
    bytes_io = bytes_per_elem * (n*p + m*p)   # A 读一次, B/C 全读全写
    return flop / bytes_io, flop

for m in [128, 4096, 16384]:
    I, F = gemm_intensity(m, 4096, 4096)
    print(f"m={m:5d}: I={I:6.1f} FLOP/byte, FLOP={F:.2e}")
```

### 5.3 真实测量

```python
# ncu 直接给 kernel 的 I 与达峰比例
# ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
#       --metrics dram__bytes_read.sum,dram__bytes_write.sum \
#       --kernel-name regex:attn python train.py
# Memory Chart 自动画 roofline 位置
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GPU内存层级]]（带宽的物理来源）、GPU 硬件规格（峰值/带宽）、[[GPU执行模型]]（occupancy/访存模式影响 $I$）。
- **下游（应用）**: [[MFU与算术强度]]（宏观模型 MFU 的依据）、[[SM utilization]]/[[occupancy分析]]（occupancy 与 bound 的正交）、[[FlashAttention]]（tiling 提 $I$ 的典范）、[[fused kernel]]（融合减 byte 提 $I$）、[[compute bottleneck]]/[[memory bottleneck]]/[[communication bottleneck]]（瓶颈三分类）、[[memory bandwidth]]（带宽利用率）、[[ncu (Nsight Compute)]]（roofline 可视化工具）、[[alpha-beta性能模型]]（通信的 roofline 类比）。
- **对比 / 易混**:
  - **Roofline vs [[MFU与算术强度|MFU]]**：Roofline 是 kernel/GPU 视角判瓶颈；MFU 是模型视角算利用率。公式同源。
  - **compute-bound vs memory-bound vs communication-bound**：算力、HBM 带宽、NVLink 带宽三类瓶颈，roofline 主要判前两类，通信 bound 用 alpha-beta。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为加算力总能让 kernel 变快
> memory-bound kernel 的瓶颈是带宽，加 tensor core 利用率无益。必须先判 bound（看 $I$ vs $R$）。

> [!warning] 误区 2：用峰值算训练时间
> 峰值是 compute-bound 段的上限，memory-bound 段远低于峰值。不乘 MFU 直接用峰值会低估时间 2-3 倍。详见 [[MFU与算术强度]] 误区 1。

> [!warning] 误区 3：忽视新一代 GPU 更易 memory-bound
> H100/B200 的 $P$ 涨比 $B$ 快 → $R$ 升 → 同 kernel 在新 GPU 上更易落在斜线段（memory-bound）。"老 GPU 上 compute-bound 的 kernel，新 GPU 上变 memory-bound"是常事。

> [!warning] 误区 4：混淆 FLOP/byte 与 op/byte
> 一个 mul-add = 2 FLOP（mul + add），不是 1。算 $I$ 时按 FLOP（mul+add 各 1），否则低估 2 倍。

> [!warning] 误区 5：以为 roofline 能精确预测
> roofline 是上限模型，实际 kernel 有 occupancy、合并、divergence 等损失，$\eta < 1$。roofline 给方向（哪 bound），不给精确数值。精确看 ncu 实测。

> [!warning] 误区 6：忽略融合算子对 $I$ 的提升
> [[fused kernel]] 把多个小算子合成一个，中间结果不落 HBM → 总 byte 大降 → $I$ 升 → 从 memory-bound 移向 compute-bound。是提 MFU 的关键手段，不是只省 launch。


## 8. 延伸细节

### 8.1 multi-level roofline

精细 roofline 分 register/shared/L2/HBM 多级带宽。tiling 进 shared 后，热数据访问走 smem 带宽（远高于 HBM），第二层屋顶更高。FlashAttention 的精髓就是把 softmax 的中间读写从 HBM 搬到 shared。

### 8.2 L2/SM cache 的 roofline

若 kernel 有良好时间局部性（同数据多 warp 复用），L2 命中率高，有效带宽超 HBM 峰值。LLM 的 [[Radix Tree prefix cache]] 部分依赖此。

### 8.3 fp8 的 roofline

H100 fp8 峰值 1979 TFLOPs（bf16 的 2×），但带宽不变 → $R$ 翻倍 → 同 kernel 在 fp8 下更易 memory-bound。fp8 训练若 kernel 不提 $I$，算力增益打折。详见 [[MFU与算术强度]] 8.5。

### 8.4 communication roofline

跨 GPU all-reduce 的"算术强度"= FLOP/byte 传输。NVLink 带宽远低于 HBM，通信极易成瓶颈。用 [[alpha-beta性能模型]] 判 ring/tree 选型，用 [[overlap strategy]] 藏通信于计算。

### 8.5 内容来源

Roofline 模型整理自 Williams et al. "Roofline: An Insightful Visual Performance Model" (2009)，GPU 规格见 NVIDIA A100/H100/B200 whitepaper，与 [[MFU与算术强度]] 互补。

---
相关: [[Kernel性能模型]] | [[GPU内存层级]] | [[MFU与算术强度]] | [[SM utilization]] | [[occupancy分析]] | [[FlashAttention]] | [[fused kernel]] | [[compute bottleneck]] | [[memory bottleneck]] | [[memory bandwidth]] | [[ncu (Nsight Compute)]] | [[alpha-beta性能模型]]
