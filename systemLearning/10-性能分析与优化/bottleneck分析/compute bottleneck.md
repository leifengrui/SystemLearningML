# compute bottleneck

> **所属章节**: [[bottleneck分析]]
> **所属模块**: [[10-性能分析与优化]]
> **别名**: compute bottleneck / 计算瓶颈 / compute-bound / 算力瓶颈
> **难度**: 中（需懂 [[SM utilization]] + [[memory bandwidth]] + roofline）


## 1. 一句话定义

**compute bottleneck（计算瓶颈）** 是 kernel/工作负载的性能受**GPU 算力（FLOPS）限制**的状态——该 kernel 是 **compute-bound**（算术强度 $I$ 高于 ridge point），throughput 上限 = 峰值算力 $P_{\text{compute}}$，与带宽无关。识别 compute bottleneck 的意义：**优化方向是提算力利用**（用 tensor core、混合精度、减少冗余 FLOP），而非加带宽（带宽够用）。它与 [[memory bottleneck]]/[[communication bottleneck]] 是**三类互斥瓶颈**——同一 kernel 通常只有一类主导瓶颈，需先用 roofline 识别是哪类，再对症下药。LLM 训练（大 batch matmul）和推理 prefill（batch 复用权重）常是 compute bottleneck；推理 decode（小 batch 读权重）是 memory bottleneck。

> [!note] 三句话定位
> - **是什么**：kernel 受算力限制（compute-bound，$I >$ ridge point），throughput = 峰值算力。
> - **为什么识别**：compute-bound 的优化是提算力利用（tensor core/混合精度/减冗余 FLOP），加带宽无用。
> - **典型场景**：LLM 训练 matmul、推理 prefill（大 batch 复用权重，算术强度高）。


## 2. 为什么需要它（动机与背景）

### 2.1 三类瓶颈的互斥性

GPU 工作负载的 throughput 上限（roofline）：

$$
\text{throughput} = \min(\underbrace{I \cdot \text{BW}}_{\text{memory-bound}}, \underbrace{P_{\text{compute}}}_{\text{compute-bound}}, \underbrace{P_{\text{comm}}}_{\text{comm-bound}})
$$

- $I < \text{ridge point}$：**memory-bound**（带宽瓶颈，[[memory bottleneck]]）。
- $I > \text{ridge point}$ 且计算主导：**compute-bound**（算力瓶颈，本篇）。
- 多 GPU 通信占比大：**communication-bound**（通信瓶颈，[[communication bottleneck]]）。

同一 kernel 通常一类主导。识别是哪类决定优化方向：compute-bound 加算力（tensor core），memory-bound 加带宽/减数据，comm-bound 减通信/overlap。

### 2.2 compute-bound 的优化方向

识别为 compute-bound 后，优化提**有效算力**：

- **tensor core**：用 HMMA（fp16/bf16）/IMMA（int8）/TF32 等 tensor core 指令，算力远超 CUDA core（A100：CUDA core 19.5 TFLOPS vs tensor core fp16 312 TFLOPS，16×）。
- **混合精度**：fp16/bf16 计算比 fp32 快 2-16×，且减显存（顺带减带宽）。
- **减冗余 FLOP**：算法优化（如 FlashAttention 减 attention 的 FLOP？实际 FA 减的是内存访问，FLOP 同；但 kernel fusion 减中间计算）。
- **提 occupancy**：[[SM utilization]] 高让算力饱和。

### 2.3 加带宽无用

compute-bound kernel 的带宽够用（数据供得上算力），加带宽（更快的 HBM）不提 throughput——throughput 被 $P_{\text{compute}}$ 限，与带宽无关。这是识别 compute-bound 的关键——**别优化错方向**。

### 2.4 LLM 的 compute-bound 场景

- **训练**：大 batch matmul（如 batch 512 × seq 2048 的 forward），权重一次读、算很多次，$I \approx 2B \cdot \text{seq} \gg$ ridge point → compute-bound。throughput 受 tensor core 算力限。
- **推理 prefill**：大 batch prompt 一次前向，权重复用高，$I \approx 2 \cdot \text{batch} \cdot \text{seq}$ 高 → compute-bound。
- **推理 decode**：batch 1 每 token 读全部权重算少，$I \approx 2$ → memory-bound（非 compute）。


## 3. 核心概念详解

### 3.1 roofline 判定

$$
I = \frac{\text{FLOP}}{\text{Byte from HBM}}, \quad \text{ridge point} = \frac{P_{\text{compute}}}{\text{BW}}
$$

- $I > \text{ridge point}$：compute-bound，throughput $= P_{\text{compute}}$。
- $I < \text{ridge point}$：memory-bound。

A100 ridge point $\approx 153$ FLOP/Byte（312 TFLOPS / 2039 GB/s）。LLM prefill（$I=500$）> 153 → compute-bound；decode（$I=2$）< 153 → memory-bound。

### 3.2 tensor core 与有效算力

| 精度 | A100 算力 | 用途 |
|---|---|---|
| fp32 | 9.7 TFLOPS | 标量计算 |
| tf32 | 156 TFLOPS | 训练（自动用） |
| bf16/fp16 | 312 TFLOPS | 训练/推理主力 |
| int8 | 624 TOPS | 量化推理 |
| fp8 (H100) | 1980 TFLOPS | H100 推理 |

compute-bound kernel 用 tensor core（bf16/int8）能提 16-64× 算力。是 compute-bound 优化的首要手段。

### 3.3 混合精度训练

- **bf16/fp16 计算**：matmul 用 tensor core，快 16×。
- **fp32 累加**：reduction 用 fp32 保精度。
- **optimizer state fp32**：Adam 的 m/v 用 fp32，但计算少（非瓶颈）。

混合精度让训练的 compute-bound matmul 用 tensor core，throughput 大幅提升。

### 3.4 计算时间估算

matmul $M \times K \times N$：

$$
\text{FLOP} = 2MNK, \quad T_{\text{compute}} = \frac{2MNK}{P_{\text{compute}}}
$$

如 $4096^3$ matmul fp16 on A100：$T = 2 \cdot 4096^3 / (312 \times 10^{12}) \approx 0.44$ms。若实测 ~0.5ms，接近算力上限（compute-bound 且优化好）；若实测 2ms，说明算力没饱和（可能 occupancy 低或 memory 实际是瓶颈）。

### 3.5 与 memory bottleneck 的对比

| 维度 | compute bottleneck | [[memory bottleneck]] |
|---|---|---|
| 算术强度 | $I >$ ridge point | $I <$ ridge point |
| throughput 上限 | $P_{\text{compute}}$ | $I \cdot \text{BW}$ |
| 优化方向 | tensor core/混合精度/减 FLOP | 减数据量/tiling/coalesced |
| 加带宽 | 无用 | 有效 |
| 典型场景 | 大 batch matmul、prefill | decode、elementwise |


## 4. 数学原理 / 公式

### 4.1 roofline

$$
\text{throughput} = \min(I \cdot \text{BW}, P_{\text{compute}})
$$

compute-bound 当 $I \ge P_{\text{compute}}/\text{BW}$。

### 4.2 计算时间

$$
T_{\text{compute}} = \frac{\text{FLOP}}{P_{\text{compute}} \cdot U_{\text{compute}}}
$$

$U_{\text{compute}}$ 是算力利用率（实测/峰值，常 30-70%）。compute-bound kernel 的 $U_{\text{compute}}$ 高（接近峰值）；若低说明算力没饱和（可能 occupancy 或实际 memory-bound）。

### 4.3 tensor core 加速

标量 CUDA core 算力 $P_{\text{cuda}}$，tensor core $P_{\text{tc}} = k \cdot P_{\text{cuda}}$（$k \approx 16$ for bf16）。compute-bound kernel 用 tensor core：

$$
T_{\text{with tc}} = T_{\text{without}} / k
$$

但需数据是 tensor core 支持的精度（fp16/bf16/tf32/int8）。

### 4.4 LLM prefill 的 compute 估算

7B prefill，batch $B$、seq $S$：

$$
\text{FLOP} \approx 2 \cdot P_{\text{model}} \cdot B \cdot S = 2 \cdot 7 \times 10^9 \cdot B \cdot S
$$

$B=64, S=2048$：$\text{FLOP} = 1.8 \times 10^{15}$。A100 算力 312 TFLOPS → $T = 5.9$ms（compute-bound 理论下限）。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟瓶颈分类

```python
def classify_bottleneck(arithmetic_intensity, ridge_point=153,
                        peak_compute=312e12, peak_bw=2039e9):
    """roofline 判定 compute-bound vs memory-bound."""
    bw_limited = arithmetic_intensity * peak_bw
    if arithmetic_intensity >= ridge_point:
        return 'compute-bound', peak_compute
    else:
        return 'memory-bound', bw_limited

# LLM 场景
cases = [('train_matmul', 1000), ('prefill_b64', 500), ('small_gemm', 100),
         ('decode_b1', 2), ('elementwise', 1)]
for name, I in cases:
    bound, t = classify_bottleneck(I)
    print(f'{name:16s} I={I:5d} -> {bound:14s} throughput={t/1e12:6.1f} TFLOPS')
```

### 5.2 计算时间估算

```python
def matmul_time(M, K, N, peak_compute=312e12, precision='bf16'):
    FLOP = 2 * M * K * N
    return FLOP / peak_compute   # compute-bound 理论下限

# 4096^3 matmul on A100 bf16
t = matmul_time(4096, 4096, 4096)
print(f'4096^3 matmul: {t*1e3:.2f}ms (compute-bound 理论下限)')
```

### 5.3 真实测量对照

```python
# import torch
# # ncu 测单 kernel 算力利用率
# # ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
# #      --kernel-name regex:mm_kernel python train.py
# # > 50% tensor core 利用率 = compute-bound 且优化好
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[SM utilization]]（occupancy 决定算力饱和）、[[memory bandwidth]]（roofline 的另一维）、roofline 模型。
- **下游（应用）**: [[memory bottleneck]]/[[communication bottleneck]]/[[CPU-GPU pipeline stall]]（其他瓶颈类型，对比定位）、[[torch profiler]]/[[nsys]]（测瓶颈类型）、tensor core/混合精度（优化手段）、LLM 训练/prefill（compute-bound 场景）。
- **对比 / 易混**:
  - **compute vs [[memory bottleneck]]**：算力限 vs 带宽限。$I$ vs ridge point 判定。优化方向相反（提算力 vs 减数据）。**最核心对比**。
  - **compute vs [[communication bottleneck]]**：单 GPU 算力限 vs 多 GPU 通信限。单卡测 compute，多卡测 communication。
  - **compute-bound vs GPU idle**：compute-bound 时 GPU 满载算（[[GPU utilization]] 高）；GPU idle 是没 kernel 执行（launch/stall），不是 compute-bound。


## 7. 常见误区与易错点

> [!warning] 误区 1：compute-bound 就加 GPU 算力
> 加 GPU（更多 SM）能提 throughput，但更经济的优化是用 tensor core/混合精度（提有效算力，不换硬件）。先榨干 tensor core 再考虑加卡。

> [!warning] 误区 2：混淆 compute-bound 与 GPU 利用率高
> GPU 利用率高（时间维满载）不一定是 compute-bound——memory-bound kernel 的 GPU 也满载（一直在执行，只是等数据）。需看算力利用率（tensor core 占比）和 $I$ vs ridge point。

> [!warning] 误区 3：compute-bound 加带宽有用
> 无用。compute-bound 的 throughput 受算力限，带宽够用。加带宽（更快的 HBM）不提 throughput。优化方向是算力（tensor core），不是带宽。

> [!warning] 误区 4：忽略精度
> fp32 的 compute-bound kernel 用 bf16 能快 16×（tensor core）。先确认用的是 tensor core 友好的精度。fp32 matmul 在 A100 上算力仅 9.7 TFLOPS（CUDA core），bf16 是 312（tensor core）。

> [!warning] 误区 5：LLM 全程 compute-bound
> 不是。LLM 训练（大 batch）和 prefill（大 batch）compute-bound，但 decode（小 batch）memory-bound（$I=2$）。同模型不同阶段不同瓶颈，优化方向不同。


## 8. 延伸细节

### 8.1 tensor core 的使用条件

- 数据精度：fp16/bf16/tf32/int8/fp8（依 GPU 支持）。
- shape 对齐：M/N/K 需是 16/8 的倍数（HMMA 16×16×16 tile）。
- 内存布局：row/col major 符合 tensor core 要求。

不满足时 fall back 到 CUDA core（算力骤降）。框架（PyTorch cuBLAS）自动处理，但自定义 kernel 需注意。

### 8.2 混合精度训练的精度管理

bf16 计算快但有精度损失（7 bit mantissa）。reduction（如 softmax、layer norm）用 fp32 累加保精度。optimizer state fp32 防梯度下溢。这是混合精度的精度管理，不影响 matmul 的 compute-bound 优化。

### 8.3 LLM 训练的 compute 占比

7B 训练一个 step（batch 512, seq 2048）：

- forward + backward FLOP ≈ $6 \cdot P \cdot B \cdot S = 6 \cdot 7e9 \cdot 512 \cdot 2048 \approx 4.4 \times 10^{16}$
- A100 单卡 312 TFLOPS → 141ms（理论下限）
- 实际 ~300ms（含 communication、optimizer、launch）→ compute 占 ~50%

compute 是大头但非全部，通信和 launch 也占。需 [[nsys]] 分解。

### 8.4 FlashAttention 与 compute

[[FlashAttention]] 主要优化 memory（tiling 减 HBM 访问），不减少 FLOP（FLOP 与标准 attention 同）。它让 attention 从 memory-bound 转向更 compute-bound（数据供得上算力了）。是 memory bottleneck 优化的典型案例。

### 8.5 算力利用率测量

ncu 的 `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` 测 tensor core 利用率：

- >50%：compute-bound 且优化好。
- <20%：算力没饱和（可能 memory 实际是瓶颈，或 occupancy 低）。

结合 roofline 判定综合诊断。

### 8.6 与分布式训练的关系

多 GPU 训练的 compute-bound 体现在单卡算力饱和（每卡 tensor core 满）。若加卡后 throughput 不升（被 communication 限），则 [[communication bottleneck]] 主导。单卡 compute-bound + 多卡 comm-bound 是常见组合，需 [[overlap strategy]] 平衡。

---
相关: [[bottleneck分析]] | [[SM utilization]] | [[memory bandwidth]] | [[memory bottleneck]] | [[communication bottleneck]] | [[CPU-GPU pipeline stall]] | [[torch profiler]] | [[nsys]] | [[overlap strategy]] | [[FlashAttention]] | [[GPU utilization]]
