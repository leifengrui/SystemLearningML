# memory bottleneck

> **所属章节**: [[bottleneck分析]]
> **所属模块**: [[10-性能分析与优化]]
> **别名**: memory bottleneck / 内存瓶颈 / memory-bound / 带宽瓶颈
> **难度**: 中（需懂 [[memory bandwidth]] + [[SM utilization]] + roofline）


## 1. 一句话定义

**memory bottleneck（内存/带宽瓶颈）** 是 kernel/工作负载的性能受 **HBM 带宽限制**的状态——该 kernel 是 **memory-bound**（算术强度 $I$ 低于 ridge point），throughput 上限 = $I \cdot \text{BW}$，与算力无关。识别 memory bottleneck 的意义：**优化方向是减数据量/提带宽利用**（tiling、fusion、coalesced、cache、量化），而非加算力（算力够用）。多数 kernel 实际是 memory-bound——LLM 推理 decode（小 batch 读权重）、elementwise（scale/激活）、attention 小 batch、reduction。与 [[compute bottleneck]]/[[communication bottleneck]] 是三类互斥瓶颈，roofline 先识别是哪类。

> [!note] 三句话定位
> - **是什么**：kernel 受带宽限制（memory-bound，$I <$ ridge point），throughput = $I \cdot$ BW。
> - **为什么识别**：memory-bound 的优化是减数据量/提带宽利用（tiling/fusion/量化），加算力无用。
> - **典型场景**：LLM decode、elementwise、attention 小 batch、reduction——多数 kernel 是 memory-bound。


## 2. 为什么需要它（动机与背景）

### 2.1 memory-bound 的普遍性

算力增长快于带宽（§memory bandwidth 的剪刀差），ridge point 升（A100 ~153、H100 ~296 FLOP/Byte），越来越多 kernel 落在 memory-bound 区。LLM 推理 decode 是典型：每 token 读全部权重（14GB for 7B）算少（~28 GFLOP），$I=2 \ll 153$，严重 memory-bound。throughput 被带宽限（~145 tokens/s 理论上限）。这是 LLM 推理慢的根因。

### 2.2 优化方向：减数据量

memory-bound 的 throughput $= I \cdot \text{BW}$，提 throughput 需提 $I$（减 Byte）或提 BW（硬件，换 GPU）：

- **减 Byte（提 $I$）**：
  - **量化**：权重 fp16→int8/int4，数据量减半/1/4，$I$ 翻倍/4倍，throughput 同比升。LLM 推理最有效。
  - **tiling/shared memory**：数据从 HBM 读一次到 smem，多线程复用，减少重复 HBM 访问（[[FlashAttention]] 核心）。
  - **kernel fusion**：多操作合一，中间结果留 register/smem 不写回 HBM。
  - **coalesced access**：相邻线程访问相邻地址，一次事务取满带宽，不浪费。
- **提 BW 利用**：让 measured bandwidth 接近 peak（coalesced、避免 bank conflict、利用 L2）。

### 2.3 加算力无用

memory-bound kernel 的算力够用（数据供不上，算力闲置），加算力（tensor core、更多 SM）不提 throughput——throughput 被 BW 限。这是识别 memory-bound 的关键——**别优化错方向**（加算力浪费）。

### 2.4 LLM 的 memory-bound 场景

- **推理 decode**：batch 1，每 token 读权重算少，$I=2$，memory-bound。throughput = 2 × 2039 = 4 TFLOPS（远低于 312 算力）。优化：量化（提 $I$）、[[continuous batching]]（多请求复用权重，提 $I$ 到 $2B$）、[[prefix caching]]（省重复 prefix）。
- **elementwise**（activation、scale、residual add）：每元素算 1 次、读 1 个，$I=1$，memory-bound。优化：kernel fusion（多 elementwise 合一，减中间 HBM 读写）。
- **attention 小 batch**：QK^T、softmax、PV，中间结果大、读多算少，memory-bound。[[FlashAttention]] 的 tiling 优化。


## 3. 核心概念详解

### 3.1 roofline 判定

$$
I < \text{ridge point} = P_{\text{compute}}/\text{BW} \implies \text{memory-bound}
$$

throughput $= I \cdot \text{BW}$。

### 3.2 减 HBM 访问的手段

| 手段 | 机制 | 典型应用 |
|---|---|---|
| **tiling/shared memory** | 数据读一次到 smem，复用 | [[FlashAttention]]、GEMM |
| **kernel fusion** | 多操作合一，中间留 register | elementwise 链 |
| **coalesced access** | 相邻线程相邻地址，满带宽事务 | 所有 kernel 基础 |
| **L2 cache 利用** | 访问模式让热数据留 L2 | 小 KV cache、权重片 |
| **量化** | 减数据精度，减 Byte | LLM 推理 |
| **recomputation** | 重算 vs 存中间（trade CPU/带宽） | [[gradient checkpointing]] |

### 3.3 带宽利用率

$$
U_{\text{bw}} = \frac{\text{measured bandwidth}}{\text{peak bandwidth}}
$$

memory-bound kernel 的 $U_{\text{bw}}$ 是核心指标。>80% 说明接近硬件极限（优化好）；<50% 说明访问模式低效（bank conflict、不 coalesced、random access）。ncu 测 `dram__bytes.sum / kernel_time`。

### 3.4 与 compute bottleneck 的对比

| 维度 | memory bottleneck | [[compute bottleneck]] |
|---|---|---|
| 算术强度 | $I <$ ridge point | $I >$ ridge point |
| throughput | $I \cdot$ BW | $P_{\text{compute}}$ |
| 优化方向 | 减数据量/提带宽利用 | tensor core/混合精度/减 FLOP |
| 加算力 | 无用 | 有效 |
| 加带宽 | 有效（换硬件） | 无用 |
| 典型场景 | decode、elementwise | 训练 matmul、prefill |

### 3.5 L2 cache 的作用

HBM 带宽 2-3TB/s（主瓶颈），L2 带宽更高（5+TB/s）。热数据（如小 KV cache、频繁读的权重片）留 L2 可显著省 HBM 带宽。但 L2 小（MB 级），大模型权重（GB 级）放不下。访问模式优化让重复访问命中 L2 是 memory-bound 调优点。


## 4. 数学原理 / 公式

### 4.1 roofline

$$
\text{throughput} = I \cdot \text{BW} \quad (\text{memory-bound}, I < \text{ridge})
$$

### 4.2 带宽时间估算

读 $D$ Byte：

$$
T_{\text{memory}} = \frac{D}{\text{BW} \cdot U_{\text{bw}}}
$$

$U_{\text{bw}}$ 是带宽利用率（0.6-0.93）。

### 4.3 LLM decode 的带宽瓶颈

7B decode 1 token：

$$
D = 14 \times 10^9 \text{ Byte}, \quad T = \frac{14 \times 10^9}{2039 \times 10^9 \cdot 0.7} \approx 9.8 \text{ms}
$$

throughput $\approx 102$ tokens/s（$U_{\text{bw}}=0.7$）。这是 decode 慢的数学根因——**带宽限，与算力无关**。

### 4.4 量化的加速

权重 fp16（$D=14$GB）→ int8（$D=7$GB）：

$$
T_{\text{int8}} = T_{\text{fp16}} / 2, \quad \text{throughput} \times 2
$$

int4（$D=3.5$GB）→ 4×。量化是 memory-bound LLM 推理最有效的优化（不减 FLOP，减数据量）。

### 4.5 continuous batching 的 $I$ 提升

batch 1 decode：$I=2$（读 14GB 算 28 GFLOP）。batch $B$：读 14GB（权重复用）算 $28B$ GFLOP，$I = 2B$。

- $B=64$：$I=128$，仍 < 153 → memory-bound 但 throughput 升 64×。
- $B=128$：$I=256 > 153$ → 转向 compute-bound。

[[continuous batching]] 把 $I$ 从 2 提到 $2B$，是 memory-bound 优化的核心。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 memory-bound + 优化

```python
def mem_bound_time(bytes_transferred, peak_bw=2039e9, utilization=0.7):
    """memory-bound kernel 时间 = data / (bw * utilization)."""
    return bytes_transferred / (peak_bw * utilization)

# 7B decode: 读 14GB 权重
t_fp16 = mem_bound_time(14e9)
print(f'7B decode fp16: {t_fp16*1e3:.1f}ms, {1/t_fp16:.0f} tokens/s (带宽限)')

# 优化1: 量化 int8 (权重 7GB)
t_int8 = mem_bound_time(7e9)
print(f'7B decode int8:  {t_int8*1e3:.1f}ms, {1/t_int8:.0f} tokens/s (2x)')

# 优化2: continuous batching B=64 (权重复用, batch 共享一次权重读取)
t_b64 = mem_bound_time(14e9)   # batch 时间仍约 9.8ms (权重读一次算 64 份)
print(f'7B decode B=64:  {t_b64*1e3:.1f}ms/batch, {64/t_b64:.0f} tokens/s 总吞吐 (64x)')

# 对比: 加算力 (无效)
print(f'  (加算力不提 memory-bound throughput, 带宽是瓶颈)')
```

### 5.2 真实测量对照

```python
# # ncu 测带宽利用率
# # ncu --metrics dram__bytes.sum.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed
# #      --kernel-name regex:attn_kernel python infer.py
# # dram__throughput > 80% peak = memory-bound 且带宽利用好
# # < 50% = 访问模式低效 (bank conflict / random access)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[memory bandwidth]]（瓶颈的物理基础）、[[SM utilization]]（occupancy 决定能否饱和带宽）、roofline。
- **下游（应用）**: [[compute bottleneck]]/[[communication bottleneck]]/[[CPU-GPU pipeline stall]]（其他瓶颈类型）、[[torch profiler]]/[[nsys]]/ncu（测带宽利用率）、[[FlashAttention]]（tiling 减 HBM 访问）、[[量化]]（减数据量）、[[continuous batching]]/[[prefix caching]]（提 $I$）、[[kernel launch overhead]]（fusion 同时优化 launch 和带宽）、[[gradient checkpointing]]（trade 算力/带宽）。
- **对比 / 易混**:
  - **memory vs [[compute bottleneck]]**：带宽限 vs 算力限。$I$ vs ridge point 判定。优化方向相反（减数据 vs 提算力）。**最核心对比**。
  - **memory vs [[communication bottleneck]]**：单卡 HBM 带宽限 vs 多卡通信带宽限。单卡测 memory，多卡测 communication。
  - **memory-bound vs bandwidth 利用率低**：memory-bound 是 throughput 受带宽限（物理上限），bandwidth 利用率低是没榨干带宽（访问模式差）。前者换硬件，后者优化访问。


## 7. 常见误区与易错点

> [!warning] 误区 1：memory-bound 加算力有用
> 无用。memory-bound 的 throughput 受带宽限，算力够用（闲置）。加 tensor core/SM 不提 throughput。优化方向是减数据量/提带宽利用。

> [!warning] 误区 2：混淆 memory-bound 与 GPU 利用率高
> memory-bound kernel 的 GPU 也满载（一直在执行，等数据时不算 idle）。[[GPU utilization]] 高不一定是 compute-bound。需看 $I$ vs ridge point 和带宽利用率。

> [!warning] 误区 3：忽略带宽利用率
> memory-bound kernel 的 throughput $= I \cdot \text{BW} \cdot U_{\text{bw}}$。$U_{\text{bw}}$ 低（访问模式差）时 throughput 远低于 $I \cdot \text{BW}$。需 coalesced/避免 bank conflict 把 $U_{\text{bw}}$ 提到 >80%。

> [!warning] 误区 4：LLM 全程 memory-bound
> 不是。decode（小 batch）memory-bound，但 prefill（大 batch）和训练（大 batch）compute-bound。同模型不同阶段不同瓶颈。continuous batching 把 decode 的 $I$ 提到 $2B$，大 B 时转 compute-bound。

> [!warning] 误区 5：量化只减显存
> 量化减显存（存放），但 memory-bound 推理的更大收益是**减带宽**（读取快），throughput 升。显存是静态，带宽是动态性能。memory-bound 场景量化的带宽收益 > 显存收益。

> [!warning] 误区 6：忽略 L2 cache
> L2 命中可省 HBM 带宽（L2 带宽更高）。小数据（KV cache、权重片）留 L2 提 throughput。访问模式优化让热数据留 L2 是 memory-bound 调优点。


## 8. 延伸细节

### 8.1 FlashAttention 的 memory 优化

标准 attention：QK^T 写回 HBM、softmax 读回、PV 写回，多次 HBM 读写，$I$ 低（memory-bound）。[[FlashAttention]] 用 tiling：Q/K/V 分块到 smem，QK^T/softmax/PV 在 smem 内完成不写回 HBM，HBM 只读 K/V 一次。$I$ 大幅提升，从 memory-bound 转向更 compute-bound。是 memory bottleneck 优化的教科书案例。

### 8.2 LLM decode 的优化栈

- **量化**：fp16→int8/int4，减数据量 2-4×。
- **[[continuous batching]]**：多请求复用权重，$I$ 从 2 提到 $2B$。
- **[[prefix caching]]**：重复 prefix 的 KV 留 L2/内存，不重算。
- **[[FlashAttention]]**：attention 的 HBM 访问减。
- **speculative decoding**：draft 生成多 token 一次验证，摊权重读取。

这些都是在 memory-bound 前提下减数据量/提 $I$。

### 8.3 kernel fusion 的双重收益

elementwise 链（scale + activation + residual add）各自 kernel 时：每步读输入写输出（多次 HBM 读写），$I$ 极低。fusion 成一个 kernel：中间结果留 register/smem，只读初始输入写最终输出，HBM 访问减 3×。同时减 [[kernel launch overhead]]（launch 次数降）。fusion 是 memory + launch 的双重优化。

### 8.4 recomputation 的 trade

[[gradient checkpointing]]：前向不存中间 activation，反向重算。trade：省 activation 显存（静态）+ 增反向计算 FLOP（动态）。在 memory-bound（activation 读带宽是瓶颈）时，重算的 FLOP 可能不增 throughput（算力闲），但省显存。是 memory/显存的 trade。

### 8.5 带宽利用率测量

ncu 的 `dram__throughput.avg.pct_of_peak_sustained_elapsed` 测带宽利用率：

- >80%：memory-bound 且优化好（接近硬件极限）。
- <50%：访问模式低效，需 coalesced/改算法。

结合 roofline 判定综合诊断。

### 8.6 与显存的关系

memory bottleneck（动态带宽）vs 显存不足（静态容量）是不同问题：

- 带宽瓶颈：数据读得慢，throughput 低。优化减数据量/提带宽。
- 显存不足：数据放不下，OOM。优化减占用/分片。

LLM 推理常两者并存（权重 + KV cache 占满显存 + 读权重慢）。需 [[GPU memory snapshot]]（静态）+ roofline（动态）结合诊断。

---
相关: [[bottleneck分析]] | [[memory bandwidth]] | [[SM utilization]] | [[compute bottleneck]] | [[communication bottleneck]] | [[CPU-GPU pipeline stall]] | [[torch profiler]] | [[nsys]] | [[FlashAttention]] | [[量化]] | [[continuous batching]] | [[prefix caching]] | [[kernel launch overhead]] | [[gradient checkpointing]] | [[GPU memory snapshot]] | [[GPU utilization]]
