# GPU utilization

> **所属章节**: [[延迟与吞吐优化]]
> **所属模块**: [[08-推理系统]]
> **别名**: GPU utilization / GPU 利用率 / 算力利用率 / 带宽利用率
> **难度**: 中高（需懂 roofline + compute/memory bound）


## 1. 一句话定义

**GPU utilization（GPU 利用率）** 是 GPU 在推理/训练中**实际使用的算力（FLOPs/s）与显存带宽（Bytes/s）占峰值比例**的度量，分**compute utilization**（算力利用率）与**memory bandwidth utilization**（带宽利用率）两个维度。受 **compute-bound vs memory-bound** 决定，可用 **roofline 模型**统一分析。LLM **decode 阶段因算术强度极低**（每步读大 [[KV cache management]] 但算极少 FLOPs）利用率常 **<10%**，是 [[batching tradeoff]]、[[continuous batching]]、[[speculative decoding]] 的核心动因。

> [!note] 三句话定位
> - **是什么**：GPU 有两个峰值（算力 FLOPs/s、带宽 Bytes/s），实际用多少是利用率。
> - **为什么**：GPU 贵，闲置=烧钱；推理 decode 天然 memory-bound 算力闲置，需诊断优化。
> - **怎么分析**：roofline 模型——算术强度（FLOP/Byte）决定是 compute-bound 还是 memory-bound。


## 2. 为什么需要它（动机与背景）

### 2.1 GPU 的两个峰值

现代 GPU（如 A100）有两个独立性能上限：

| 资源 | A100 典型峰值 |
|---|---|
| 算力（FP16） | ~312 TFLOPS |
| 显存带宽（HBM） | ~1.5 TB/s |

实际任务能达到多少，取决于任务**每搬运 1 字节数据做多少运算**（算术强度）。算少→卡在带宽；算多→卡在算力。

### 2.2 LLM 推理的两阶段利用率差异

| 阶段 | 每步工作量 | 算术强度 | 瓶颈 | 利用率 |
|---|---|---|---|---|
| **prefill** | 算 prompt 全部 token 的 $K/V$（大计算） | 高（~100–500 FLOP/Byte） | compute-bound | 高（接近算力峰值） |
| **decode** | 算 1 个 token，但读整个 KV cache（大访存） | 极低（~1–10 FLOP/Byte） | memory-bound | 低（算力闲置，<10%） |

> [!warning] decode 利用率低不是 bug，是机制
> decode 每步生成 1 个 token：计算量 $\approx 2N$（$N$ 模型参数，前向一次），但要读**整个 KV cache**（$\propto n_{\text{layers}} \cdot d \cdot S \cdot B$ 字节）进 SRAM 算 attention。算少读多 → 算术强度极低 → 卡在带宽 → 算力利用率 <10%。这是自回归机制固有的，不是实现差。

### 2.3 提利用率的全部努力

推理系统的优化几乎都在提 decode 利用率：

- [[continuous batching]]：大 batch 让一次 cache 读为多 token 服务，摊薄访存。
- [[speculative decoding]] / [[parallel decoding]]：一次 forward 算多 token，提算力使用。
- [[KV cache management]] 量化：减小 cache 体积，降访存。
- kernel 融合：减少访存次数。

这些本质都是"让 GPU 算力别闲着"。


## 3. 核心概念详解

### 3.1 算术强度（arithmetic intensity）

$$
I = \frac{\text{FLOPs}}{\text{Bytes moved}} \quad [\text{FLOP/Byte}]
$$

每搬运 1 字节数据（HBM↔SRAM）做了多少次浮点运算。$I$ 高→算多读少→compute-bound；$I$ 低→算少读多→memory-bound。

### 3.2 roofline 模型

把 GPU 性能上限表示为算术强度 $I$ 的函数：

$$
\text{Attainable}(I) = \min\!\left(P_{\text{compute}},\ I \cdot P_{\text{bw}}\right)
$$

- $P_{\text{compute}}$ = 峰值算力（FLOPs/s）
- $P_{\text{bw}}$ = 峰值带宽（Bytes/s）

图形是"屋顶线"：$I$ 小时性能 $= I \cdot P_{\text{bw}}$（斜线，memory-bound）；$I$ 大时性能 $= P_{\text{compute}}$（平线，compute-bound）。两线交点为**脊点（ridge point）**：

$$
I_{\text{ridge}} = \frac{P_{\text{compute}}}{P_{\text{bw}}}
$$

A100：$I_{\text{ridge}} = 312\text{T}/1.5\text{T} = 208$ FLOP/Byte。$I < 208$ memory-bound，$I > 208$ compute-bound。

### 3.3 LLM 的算术强度

**decode（单 token）**：
- FLOPs $\approx 2N$（$N$ 参数前向一次，7B → $1.4\times 10^{10}$）。
- Bytes moved $\approx$ KV cache 大小（7B, seq 2048, batch 1 → ~1 GB）。
- $I \approx 1.4\times 10^{10} / 10^9 \approx 14$ FLOP/Byte $\ll 208$ → **强 memory-bound**。

**prefill（prompt 全部）**：
- FLOPs $\approx 2N \cdot S_{\text{prompt}}$（每 token $2N$，共 $S$ token）。
- Bytes moved $\approx$ 权重 $2N$（权重读一次，prompt 多 token 摊薄）。
- $I \approx \frac{2N \cdot S}{2N} = S$（每字节权重做 $S$ 次运算）→ 随 $S$ 增 → 接近或超 ridge → **compute-bound**。

> [!note] 直觉
> prefill 把"读一次权重"摊到 $S$ 个 token 的计算上，算术强度 $\propto S$，所以长 prompt 的 prefill 是 compute-bound、利用率高。decode 每步只 1 token，读权重 + 读整个 KV cache 但只算 1 token，强度极低。这是两阶段利用率天差地别的根因。

### 3.4 利用率的两维与混淆

- **compute utilization** = 实际 FLOPs/s ÷ 峰值 FLOPs/s。
- **memory bandwidth utilization** = 实际 Bytes/s ÷ 峰值 Bytes/s。

nvidia-smi 的 "GPU-Util" 是 **SM 占用率**（有多少 SM 在跑），不等于算力或带宽利用率！decode 时 SM-Util 可能 100%（所有 SM 都在跑 memory load）但算力利用率 <5%。诊断瓶颈要看 profiler（[[torch profiler]]/[[nsys]]，Chapter 10）的 FLOPs 与带宽计数，不能只看 nvidia-smi。

### 3.5 提升利用率的策略

| 策略 | 作用 | 知识点 |
|---|---|---|
| 大 batch（continuous） | 摊薄 KV cache 读，提 $I$ | [[continuous batching]] |
| speculative/parallel | 一次 forward 多 token，提算力使用 | [[speculative decoding]]/[[parallel decoding]] |
| KV 量化 | 减小 cache 体积，降访存 | [[KV cache management]]/[[量化]] |
| kernel 融合 | 减少访存次数 | FlashAttention 等 |
| GQA/MQA | 减 KV 头数，降 cache | [[attention]] |


## 4. 数学原理 / 公式

### 4.1 roofline 公式推导

GPU 每秒能做 $P_{\text{compute}}$ 次运算，每秒能搬 $P_{\text{bw}}$ 字节。对算术强度 $I$（FLOP/Byte）的任务：

- 若受算力限：每秒搬 $P_{\text{bw}}$ 字节能做 $I \cdot P_{\text{bw}}$ FLOPs；若 $I \cdot P_{\text{bw}} > P_{\text{compute}}$，则算力先到顶 → attainable $= P_{\text{compute}}$。
- 若受带宽限：$I \cdot P_{\text{bw}} < P_{\text{compute}}$，带宽先到顶 → attainable $= I \cdot P_{\text{bw}}$。

统一：$\text{Attainable} = \min(P_{\text{compute}}, I \cdot P_{\text{bw}})$。

交点 $I_{\text{ridge}} = P_{\text{compute}} / P_{\text{bw}}$。

### 4.2 decode 利用率估算

7B 模型，A100（$P_{\text{compute}}=312$T, $P_{\text{bw}}=1.5$T, $I_{\text{ridge}}=208$）：

- 单 token decode FLOPs $\approx 2 \times 7\text{B} = 1.4 \times 10^{10}$。
- KV cache 读 $\approx 1$ GB $= 10^9$ bytes（seq 2048）。
- $I = 14$ FLOP/Byte。
- Attainable $= 14 \times 1.5\text{T} = 2.1 \times 10^{13}$ FLOPs/s $= 21$ TFLOPS。
- Compute utilization $= 21 / 312 \approx 6.7\%$。

batch=64 时：FLOPs $\times 64$，KV 读 $\times 64$（每请求独立 cache），$I$ 不变？不对——batch 内每请求的 KV 独立读，但**权重读只一次**（所有 batch 共享权重）。所以：

- FLOPs $= 64 \times 2N$（每请求算）。
- Bytes $= 2N$（权重一次）$+ 64 \times \text{KV}$（每请求 cache）。
- 大 batch 时权重读被摊薄，$I \approx \frac{64 \times 2N}{64 \times \text{KV}} = \frac{2N}{\text{KV}}$... 仍由 KV 主导。

> [!note] 大 batch 提利用率的真正原因
> 不是改变 $I$，而是**让一次访存（读权重+cache）为多 token 的计算服务**——权重读固定，batch 越大权重读被摊薄，整体 $I$ 提升，向 ridge 靠近。这是 [[continuous batching]] 提利用率的本质。

### 4.3 prefill 利用率

7B prefill，prompt 2048 token：

- FLOPs $\approx 2 \times 7\text{B} \times 2048 = 2.87 \times 10^{13}$。
- Bytes $\approx 2 \times 7\text{B} = 1.4 \times 10^{10}$（权重一次）$+$ KV 写入。
- $I \approx 2048$ FLOP/Byte $\gg 208$ → compute-bound。
- Attainable $= 312$ TFLOPS，utilization 接近 100%。


## 5. 代码示例（可选

### 5.1 roofline 模型与 decode/prefill 利用率

```python
def roofline(intensity, peak_compute, peak_bw):
    """返回 (attainable_FLOPS, bound_type, ridge)."""
    ridge = peak_compute / peak_bw
    if intensity < ridge:
        return intensity * peak_bw, 'memory-bound', ridge
    return peak_compute, 'compute-bound', ridge

A100_compute = 312e12   # 312 TFLOPS FP16
A100_bw = 1.5e12        # 1.5 TB/s
ridge = A100_compute / A100_bw
print(f'ridge point: {ridge:.0f} FLOP/Byte')

# decode: 7B, 单 token, KV ~1GB
N = 7e9
decode_flops = 2 * N
decode_bytes = 1e9
I_decode = decode_flops / decode_bytes
att, bound, _ = roofline(I_decode, A100_compute, A100_bw)
print(f'decode  I={I_decode:.1f}  {bound}  attain={att/1e12:.1f}T  util={att/A100_compute*100:.2f}%')

# prefill: 7B, prompt 2048, 权重读一次
prefill_flops = 2 * N * 2048
prefill_bytes = 2 * N
I_prefill = prefill_flops / prefill_bytes
att2, bound2, _ = roofline(I_prefill, A100_compute, A100_bw)
print(f'prefill I={I_prefill:.0f}  {bound2}  attain={att2/1e12:.1f}T  util={att2/A100_compute*100:.1f}%')
```

预期：decode ~6.7% util（memory-bound），prefill ~100%（compute-bound）。

### 5.2 大 batch 摊薄权重读

```python
def util_with_batch(B, N=7e9, kv_bytes=1e9):
    """batch B 时, 权重读被摊薄, 算术强度提升."""
    flops = B * 2 * N
    bytes_moved = 2 * N + B * kv_bytes   # 权重一次 + 每请求 KV
    I = flops / bytes_moved
    att = min(A100_compute, I * A100_bw)
    return I, att / A100_compute * 100
for B in [1, 4, 16, 64, 256]:
    I, u = util_with_batch(B)
    print(f'B={B:>3}  I={I:.1f}  util={u:.1f}%')
# 大 batch util 提升 (但受显存约束, 见 batching tradeoff)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[KV cache management]]（decode 访存的主体）、[[attention]]（计算与访存的算子）、[[batching tradeoff]]（batch 影响利用率）。
- **下游（应用）**: [[continuous batching]]/[[speculative decoding]]/[[parallel decoding]]（提利用率的手段）、[[torch profiler]]/[[nsys]]（Chapter 10，实测利用率的工具）、[[activation memory]]/[[gradient checkpointing]]（Chapter 11，训练侧显存与重算影响利用率）。
- **对比 / 易混**:
  - **compute utilization vs memory bandwidth utilization**：两维独立，一个任务通常只卡一个。memory-bound 时算力利用率低但带宽利用率高（接近峰值）。
  - **SM-Util（nvidia-smi）vs 真利用率**：SM-Util 高只代表 SM 在忙（可能在等内存），不等于算力利用率高。诊断瓶颈要看 profiler 的 FLOPs/带宽计数。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 nvidia-smi 的 GPU-Util 高就是利用率高
> GPU-Util 是 SM 占用率（多少 SM 在跑），decode 时可能 100% 但算力利用率 <5%（SM 全在等 HBM 数据）。判断瓶颈要用 profiler（[[torch profiler]]/[[nsys]]）看 FLOPs/s 与带宽/s。

> [!warning] 误区 2：以为利用率高=速度快
> 不一定。compute-bound 任务利用率高但可能因算力到顶慢；memory-bound 任务带宽利用率高但算力闲置。优化方向取决于卡在哪个峰值。提利用率要提"卡的那个"。

> [!warning] 误区 3：以为 decode 利用率低是实现差
> decode 天然 memory-bound（自回归每步读大 cache 算少），利用率低是机制固有的。优化是"在低利用率里尽量提"（大 batch、speculative），不是"消除低利用率"。

> [!warning] 误区 4：混淆算力利用率与带宽利用率
> 一个任务通常卡其一。说"利用率低"要指明是算力低还是带宽低。decode 是算力低（带宽可能接近满）。

> [!warning] 误区 5：忽略 ridge point 因型号而异
> ridge $= P_{\text{compute}}/P_{\text{bw}}$ 因 GPU 型号不同（A100 vs H100 vs 消费卡）。A100 ridge ~208，H100 算力涨更快 ridge 更高。跨型号分析的 ridge 要重算。


## 8. 延伸细节

### 8.1 roofline 图的绘制

横轴算术强度（log），纵轴 attainable FLOPs/s（log）。GPU 的"屋顶"是斜线 $I \cdot P_{\text{bw}}$（左）接平线 $P_{\text{compute}}$（右），交点 ridge。把任务的 $(I, \text{attainable})$ 点画上去，落在斜线段=memory-bound，平线段=compute-bound。是性能分析的通用工具（Chapter 10 详）。

### 8.2 kernel 融合与 FlashAttention

[[attention]] 的 naive 实现把 $QK^\top$、softmax、$AV$ 分多个 kernel，中间结果写回 HBM → 多次访存、算术强度低。**FlashAttention** 把整个 attention 融合进一个 kernel，中间结果留 SRAM、用 tiling 算，大幅减访存 → 提算术强度 → 提利用率。这是 kernel 融合提利用率的典范。

### 8.3 MPS（Multi-Process Service）

NVIDIA MPS 让多进程共享 GPU 的 SM，把多个小 batch 的请求合并执行，提整体 SM 利用率。适合多用户小 batch 场景。

### 8.4 与训练侧的对比

训练时前向+反向，每步算 $6N$ FLOPs（前向 $2N$ × 反向 $\sim 2\times$），算术强度高 → 通常 compute-bound、利用率高。推理 decode 算少 → memory-bound、利用率低。这是推理优化比训练更难提利用率的根因。

### 8.5 与 RLHF 的关系

[[rollout worker]] 采样阶段利用率低（decode 为主），是 RLHF 训练的吞吐瓶颈。[[continuous batching]] + [[speculative decoding]] + [[prefix caching]] 组合提 decode 利用率是加速采样的关键。[[policy deployment (PD)]] 服务的成本由利用率直接决定。

### 8.6 profiler 工具（Chapter 10 详）

- [[torch profiler]]：PyTorch 内置，看算子级 FLOPs/访存/耗时。
- [[nsys]]（NVIDIA Nsight Systems）：系统级 timeline，看 kernel launch、内存传输、CPU/GPU 重叠。
- [[ncu]]（Nsight Compute）：kernel 级细粒度，看每个 kernel 的 roofline 位置。

这些工具在 Chapter 10（性能分析与优化）展开。

---
相关: [[延迟与吞吐优化]] | [[KV cache management]] | [[batching tradeoff]] | [[continuous batching]] | [[speculative decoding]] | [[torch profiler]]
