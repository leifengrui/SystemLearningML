# memory bandwidth

> **所属章节**: [[GPU性能]]
> **所属模块**: [[10-性能分析与优化]]
> **别名**: memory bandwidth / 显存带宽 / HBM bandwidth / 内存带宽
> **难度**: 中（需懂 roofline + GPU 内存层次）


## 1. 一句话定义

**memory bandwidth（显存带宽）** 是 GPU **HBM（高带宽显存）** 到计算单元的**数据传输速率**（GB/s），衡量 GPU 从显存取数据有多快。它是 GPU 性能的**另一条腿**——与算力（FLOPS）并列。多数 kernel（特别是 LLM decode、elementwise、attention 小 batch）是 **memory-bound**：算力够但数据供不上，throughput 被带宽限。roofline 模型把算力与带宽统一：kernel 的 throughput 上限 = min(算术强度 × 带宽, 峰值算力)。A100 HBM 带宽 2039 GB/s、H100 3350 GB/s——这些数字决定 memory-bound kernel 的性能天花板。

> [!note] 三句话定位
> - **是什么**：HBM 到计算单元的数据传输速率（GB/s），GPU 性能两条腿之一。
> - **为什么**：多数 kernel 是 memory-bound，throughput 被带宽限。带宽决定 memory-bound kernel 的天花板。
> - **与算力的关系**：roofline 统一——算术强度低 → memory-bound（带宽瓶颈），高 → compute-bound（算力瓶颈）。


## 2. 为什么需要它（动机与背景）

### 2.1 算力与带宽的剪刀差

GPU 算力增长快于带宽：A100 算力 312 TFLOPS（fp16）、带宽 2039 GB/s；H100 算力 990 TFLOPS、带宽 3350 GB/s。算力/带宽比（ridge point）：

- A100：$312 \times 10^{12} / 2039 \times 10^{9} \approx 153$ FLOP/Byte
- H100：$990 \times 10^{12} / 3350 \times 10^{9} \approx 296$ FLOP/Byte

算力涨 3.2×，带宽涨 1.6× → ridge point 升 → 越来越多 kernel 落在 memory-bound 区。这使**带宽成为越来越多场景的瓶颈**。

### 2.2 memory-bound 的普遍性

LLM 推理的 decode 阶段：每生成 1 token，需读全部权重（14GB for 7B）做一次前向，但只算少量 FLOP（每参数 ~2 FLOP）。算术强度 ~2 FLOP/Byte，远低于 ridge point 153 → 严重 memory-bound。throughput 被带宽限（2039 GB/s → ~145 tokens/s 理论上限，实际更低）。这是 LLM 推理慢的根因。

### 2.3 带宽利用率的测量

peak bandwidth 是规格上限，实际 kernel 能达到的 **measured bandwidth** 通常 60-85% peak（受访问模式、bank conflict、L2 命中等影响）。测 measured bandwidth 占 peak 的比例 = **带宽利用率**，是 memory-bound kernel 调优的核心指标。

### 2.4 GPU 内存层次

```
[HBM (大, 慢, ~2TB/s)] → [L2 cache (中, ~5TB/s)] → [L1/smem (小, 快, ~19TB/s)] → [registers (最快)]
```

- HBM：40-80GB，带宽 2-3TB/s，所有数据最初在此。
- L2：4-12MB，带宽更高，命中省 HBM 访问。
- L1/smem：每 SM 192KB，最快，需手动管理（shared memory）。

bandwidth 通常指 HBM 带宽（最瓶颈层）。优化方向：尽量让数据从 L2/smem 取（少访 HBM）。


## 3. 核心概念详解

### 3.1 HBM 规格

| GPU | HBM | 带宽 | 容量 |
|---|---|---|---|
| V100 | HBM2 | 900 GB/s | 32GB |
| A100 | HBM2e | 2039 GB/s | 40/80GB |
| H100 | HBM3 | 3350 GB/s | 80GB |
| H200 | HBM3e | 4800 GB/s | 141GB |

> [!note] 解答：联网核实 HBM 规格表正确性（2026-07 核实）
> 上表四行**全部正确**，来源 NVIDIA 官方 datasheet + 多家对比。补充与订正如下：
>
> | GPU | HBM | 带宽 | 容量 | 核实来源 |
> |---|---|---|---|---|
> | V100 | HBM2 | 900 GB/s | 16/32GB | ✓ V100 datasheet：32GB HBM2 900 GB/s（16GB 版同带宽） |
> | A100 40GB | HBM2e | **1555 GB/s** | 40GB | ⚠ 上表只写 2039 是 **80GB 版**，40GB 版带宽是 1555，需区分 |
> | A100 80GB | HBM2e | **2039 GB/s** | 80GB | ✓ A100 datasheet：80GB HBM2e "over 2TB/s" |
> | H100 | HBM3 | 3350 GB/s | 80GB | ✓ H100：3.35 TB/s HBM3 |
> | H200 | HBM3e | 4800 GB/s | 141GB | ✓ H200：4.8 TB/s HBM3e, 141GB |
>
> ### 2025 新增 Blackwell（建议补入表格）
> 上表停在 H200，2025 年 NVIDIA 发布 Blackwell / Blackwell Ultra，2026 已量产，LLM 推理新主流：
>
> | GPU | HBM | 带宽 | 容量 | 备注 |
> |---|---|---|---|---|
> | B200 | HBM3e | **8000 GB/s（8.0 TB/s）** | 192GB | Blackwell, 2025；FP4 推理主力 |
> | B300 | HBM3e | **8000 GB/s（8.0 TB/s）** | 288GB | Blackwell Ultra, 2025；大 KV cache / 100B+ 推理 |
>
> - B200/B300 带宽相比 H200 (4.8TB/s) **翻 1.67×**，容量大幅提升（192/288GB vs 141GB）。
> - NVLink 也换代：A100 600GB/s → H100/H200 900GB/s → B200/B300 **1.8 TB/s**（5th-gen NVLink）。
> - 注意 B200 有"per-GPU 8TB/s"与"DGX 系统 64TB/s 总带宽"两种表述，后者是 8 卡聚合；单卡约 8TB/s。
>
> ### 一句话结论
> 原表四行**正确**，但 **A100 应区分 40GB(1555)/80GB(2039)**，且建议补 **B200(192GB, 8TB/s)/B300(288GB, 8TB/s)** 两行——2026 年 LLM 推理 decode 的带宽天花板已从 A100 的 2039 GB/s 升到 B200 的 8000 GB/s，§4.2 算的 "7B decode ~145 tokens/s" 在 B200 上理论升到 ~571 tokens/s（带宽墙缓解但仍 memory-bound，因算术强度 I=2 没变）。来源：NVIDIA V100/A100/H100 datasheet、acecloud/exxact/cloud4y 2025 对比表（2026-07-12 核实）。
带宽增长靠 HBM 代际（HBM2→2e→3→3e）+ 堆叠层数 + 接口宽度。

### 3.2 roofline 模型

kernel 的 throughput 上限：

$$
\text{throughput} = \min(\underbrace{I \cdot \text{BW}}_{\text{memory-bound 上限}}, \underbrace{P_{\text{compute}}}_{\text{compute-bound 上限}})
$$

$I$ = 算术强度（FLOP/Byte），BW = HBM 带宽，$P_{\text{compute}}$ = 峰值算力。

- $I < \text{ridge point} = P_{\text{compute}}/\text{BW}$：memory-bound，throughput $= I \cdot \text{BW}$。
- $I > \text{ridge point}$：compute-bound，throughput $= P_{\text{compute}}$。

> [!note] ridge point
> ridge point = $P_{\text{compute}}/\text{BW}$（A100 ~153 FLOP/Byte）。算术强度低于此 → memory-bound；高于此 → compute-bound。这是判断瓶颈类型的分界线。

### 3.3 算术强度（arithmetic intensity）

$$
I = \frac{\text{FLOP}}{\text{Byte transferred from HBM}}
$$

- elementwise（如 activation、scale）：~1 FLOP/Byte（每元素算 1 次、读 1 个）。
- matmul（大 GEMM）：~$2N$ FLOP/Byte（$N$ 是边长，tile 复用）。
- LLM decode attention：~2-14 FLOP/Byte（读权重多、算少）。
- LLM prefill matmul（大 batch）：~500+ FLOP/Byte（权重复用高）。

### 3.4 measured vs peak bandwidth

- **peak**：规格上限（A100 2039 GB/s）。
- **measured**：实测（如 stream benchmark ~1900 GB/s，~93% peak）。
- **kernel measured**：单 kernel 实测，常 60-85% peak（受访问模式影响）。

调优目标：memory-bound kernel 的 measured bandwidth 接近 peak（如 >80%）。

### 3.5 带宽优化技术

- **coalesced access**：相邻线程访问相邻地址（合并成大事务），否则离散访问浪费带宽。
- **shared memory tiling**：数据从 HBM 读一次到 smem，多线程复用（[[FlashAttention]] 的核心）。
- **L2 cache 利用**：访问模式让数据留 L2，重复命中。
- **减少冗余读写**：kernel fusion（多操作合一，避免中间结果写回 HBM）。


## 4. 数学原理 / 公式

### 4.1 roofline 公式

$$
\text{throughput} = \min(I \cdot \text{BW}, P_{\text{compute}})
$$

ridge point：

$$
I^* = \frac{P_{\text{compute}}}{\text{BW}}
$$

### 4.2 LLM decode 的带宽瓶颈

7B 模型 decode 1 token：

- 读权重：$14 \times 10^9$ Byte（fp16）
- 算：~$28 \times 10^9$ FLOP（每参数 ~2 FLOP）
- $I = 28/14 = 2$ FLOP/Byte $\ll 153$ → memory-bound

throughput 上限（带宽限）：

$$
\text{tokens/s} = \frac{\text{BW}}{\text{weight size}} = \frac{2039 \times 10^9}{14 \times 10^9} \approx 145 \text{ tokens/s}
$$

实际 ~100 tokens/s（measured bandwidth ~70% peak）。这是 LLM decode 慢的数学根因——**带宽限，与算力无关**。

> [!note] decode vs prefill 的带宽差异
> decode（batch 1）：读 14GB 算 28 GFLOP，$I=2$，memory-bound。prefill（batch 512）：读 14GB 算 14 TFLOP（batch 复用权重），$I=500$，compute-bound。同模型不同阶段瓶颈不同——decode 看带宽，prefill 看算力。这是 [[batching tradeoff]] 的根因。

### 4.3 带宽利用率

$$
U_{\text{bw}} = \frac{\text{measured bandwidth}}{\text{peak bandwidth}}
$$

memory-bound kernel 的 $U_{\text{bw}}$ 是核心指标。>80% 算好（接近硬件极限），<50% 说明访问模式低效（bank conflict、不 coalesced）。

### 4.4 多层带宽

数据从 HBM→L2→smem→register，每层带宽递增：

$$
\text{effective BW} = \text{HBM BW} \cdot \text{L2 hit rate} + \text{L2 BW} \cdot (1 - \text{L2 hit rate})
$$

L2 命中率高 → effective BW 接近 L2 BW（快）。优化让热数据留 L2。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 roofline

```python
def roofline(arithmetic_intensity, peak_compute=312e12, peak_bw=2039e9):
    """roofline: throughput = min(intensity * bw, peak_compute)."""
    bw_limited = arithmetic_intensity * peak_bw       # memory-bound 上限
    return min(bw_limited, peak_compute), bw_limited >= peak_compute

# A100: 312 TFLOPS (fp16), 2039 GB/s HBM
kernels = [('decode_attn', 2), ('elementwise', 1), ('small_matmul', 50),
           ('prefill_matmul', 500), ('large_gemm', 2000)]
for name, intensity in kernels:
    t, compute_bound = roofline(intensity)
    bound = 'compute' if compute_bound else 'memory'
    print(f'{name:18s} I={intensity:5d} FLOP/Byte -> {t/1e12:6.1f} TFLOPS ({bound}-bound)')
```

### 5.2 LLM decode 带宽瓶颈计算

```python
weight_size_gb = 14                 # 7B fp16
bandwidth_gbs = 2039               # A100 HBM
tokens_per_sec = bandwidth_gbs / weight_size_gb
print(f'7B decode 理论上限: {tokens_per_sec:.0f} tokens/s (带宽限)')
# ~145 tokens/s (实际更低, measured bw < peak)
```

### 5.3 真实测量 API 对照

```python
# # ncu 测单 kernel 带宽:
# ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,sm__inst_executed.sum \
#     ./my_program
# # 或 stream benchmark 测峰值带宽:
# # cuda-samples/bandwidthTest
```


## 6. 与其他知识点的关系

- **上游（依赖）**: GPU 内存层次（HBM/L2/smem/register）、roofline 模型、[[SM utilization]]（occupancy 决定能否饱和带宽）。
- **下游（应用）**: [[memory bottleneck]]（带宽瓶颈诊断）、[[compute bottleneck]]（对比算力瓶颈）、[[GPU utilization]]（带宽利用率 vs 计算利用率）、[[KV cache management]]（KV cache 占带宽）、[[continuous batching]]（提算术强度饱和带宽）、[[FlashAttention]]（tiling 减 HBM 访问）。
- **对比 / 易混**:
  - **memory bandwidth vs [[SM utilization]]**：带宽是数据供给速率（数据维），SM utilization 是计算单元占用（计算维）。memory-bound 时带宽是瓶颈、SM utilization 高也空转。
  - **peak vs measured bandwidth**：peak 是规格上限，measured 是实测（60-93% peak）。调优看 measured。
  - **HBM 带宽 vs smem 带宽**：HBM 2-3TB/s（慢、大），smem 19TB/s（快、小）。优化让数据从 smem 取。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 peak bandwidth 可达
> peak 是规格上限，实测 measured 通常 60-93%（stream benchmark ~93%，单 kernel 60-85%）。调优目标是让 measured 接近 peak，但不可能 100%。

> [!warning] 误区 2：忽略算术强度直接说 memory-bound
> 判断 memory-bound 需算术强度 $I$ vs ridge point。$I < I^*$ 才 memory-bound。LLM decode $I=2$ memory-bound，但 prefill $I=500$ compute-bound。同模型不同阶段不同瓶颈。

> [!warning] 误区 3：bandwidth 利用率高就最优
> memory-bound kernel 的 bandwidth 利用率高（接近 peak）说明已接近硬件极限，进一步优化难。需从算法层减数据量（如 [[FlashAttention]] 减 HBM 访问），而非榨干带宽。

> [!warning] 误区 4：混淆 HBM 带宽与 smem 带宽
> HBM 带宽 2-3TB/s（主瓶颈），smem 带宽 19TB/s（快但小）。优化让数据从 smem 取（tiling），减少 HBM 访问。两者差 6-10×。

> [!warning] 误区 5：忽略 coalesced access
> 线程访问不连续地址（strided/random）→ 带宽浪费（一次事务只取有用部分）。coalesced（相邻线程相邻地址）才充分利用带宽。这是 memory-bound kernel 优基础。


## 8. 延伸细节

### 8.1 HBM 堆叠与带宽

HBM 通过 3D 堆叠 DRAM die + 硅孔（TSV）+ 宽接口（1024-bit）实现高带宽。HBM3e（H200）达 4800 GB/s。但 HBM 容量受堆叠层数限（难做大），是大模型显存墙的根因。

### 8.2 L2 cache 的作用

A100 L2 40MB、H100 50MB。热数据（如小 KV cache、频繁读的权重片）留 L2 可显著省 HBM 带宽。访问模式优化（让重复访问命中 L2）是调优点。但 L2 小（MB 级），大模型权重（GB 级）放不下。

### 8.3 LLM 推理的带宽优化

- **[[continuous batching]]**：多请求拼 batch，复用权重读取，提算术强度（从 $I=2$ 到 $I=2B$）。
- **[[prefix caching]]**：prefix KV 留 L2/内存，重复 prefix 不重算。
- **[[FlashAttention]]**：tiling 让 attention 中间结果留 smem，不写回 HBM。
- **量化**：权重 fp16→int8，减半数据量，带宽限下吞吐翻倍。

### 8.4 stream benchmark

`cuda-samples/bandwidthTest` 测 HBM peak bandwidth（顺序读写）。是 GPU 健康检查 + 带宽上限测量的标准工具。生产 GPU 若 stream bandwidth 远低于 spec，可能硬件故障或散热问题。

### 8.5 带宽墙与算力墙

- **带宽墙**：memory-bound kernel 受带宽限，加算力无用（如 LLM decode 单卡加 tensor core 无效）。
- **算力墙**：compute-bound kernel 受算力限，加带宽无用（如大 GEMM）。

LLM 训练（大 batch）多 compute-bound（算力墙），推理 decode（小 batch）memory-bound（带宽墙）。不同场景优化方向不同。

### 8.6 ncu 测带宽

```
ncu --metrics dram__bytes.sum,sm__cycles_active.avg \
    --kernel-name regex:attention ./my_program
```

dram__bytes.sum 测 HBM 传输字节数，除以 kernel 时长得 measured bandwidth。详见 [[nsys]]（系统级）vs ncu（kernel 级）。

---
相关: [[GPU性能]] | [[SM utilization]] | [[GPU utilization]] | [[compute bottleneck]] | [[memory bottleneck]] | [[KV cache management]] | [[continuous batching]] | [[prefix caching]] | [[FlashAttention]] | [[量化]] | [[nsys]] | [[batching tradeoff]]
