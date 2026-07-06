# gradient memory

> **所属章节**: [[显存结构]]
> **所属模块**: [[11-显存与内存系统]]
> **别名**: gradient memory / 梯度显存 / 梯度内存 / gradient buffer
> **难度**: 中（需懂反向传播 + optimizer + 分布式训练）


## 1. 一句话定义

**gradient memory（梯度显存）** 是反向传播产生的、**供 optimizer 更新参数用的梯度张量**——每个模型参数对应一个梯度，大小 = 参数大小（同精度）。它是训练显存三大组成之一（[[activation memory|activation]] + [[gradient memory|gradient]] + [[optimizer state memory|optimizer state]]），与 batch 无关（每参数固定 1 梯度），但与精度相关（fp16 减半 vs fp32）。在分布式训练中，gradient 还需 allreduce 同步（[[communication bottleneck]] 的来源），且可分片（[[ZeRO (DeepSpeed)]]-2）。混合精度训练时 gradient 常以 fp16 存（计算）+ fp32 master（精度保真），需区分。

> [!note] 三句话定位
> - **是什么**：反向产生的、供 optimizer 用的梯度，每参数 1 个，大小 = 模型大小。
> - **为什么需要**：optimizer（SGD/Adam）需梯度更新参数，必须存到 optimizer step 用。
> - **怎么优化**：ZeRO-2 分片、混合精度减半、gradient accumulation 省大 batch 显存。


## 2. 为什么需要它（动机与背景）

### 2.1 optimizer 需要梯度

参数更新 $\theta \leftarrow \theta - \eta \cdot \text{optimizer}(\nabla L)$。optimizer（SGD/Adam）需 $\nabla L$（梯度）算更新量。反向传播算出每参数梯度后，必须**保存**到 optimizer step（通常反向结束、梯度同步后）才用。这段保存期就是 gradient memory 的生命周期。

### 2.2 与 batch 无关

每参数 1 个梯度，梯度形状 = 参数形状，与 batch 无关（batch 影响算梯度的 compute，不影响梯度张量大小）。7B 模型 fp16 → 14 GB gradient，无论 batch=1 还是 512。这与 [[activation memory|activation]]（线性于 batch + $s^2$）形成对比。

### 2.3 三大组成中的位置

| 组成 | 7B fp16 | 与 batch | 生命周期 |
|---|---|---|---|
| 权重 | 14 GB | 无关 | 全程 |
| **gradient** | 14 GB | **无关** | 反向产生→optimizer 用完释放 |
| [[optimizer state memory\|optimizer state]] | 28-56 GB | 无关 | 全程 |
| [[activation memory\|activation]] | 4-253 GB | 线性 + $s^2$ | 前向→反向释放 |

gradient（14GB）小于 optimizer state（28-56GB），但大于权重的"静态"部分（gradient 是动态分配）。峰值显存含 gradient（反向时产生）。

### 2.4 分布式下的梯度同步

DP 训练时，各卡算各自 batch 的梯度，需 **allreduce** 聚合（求平均）才能更新。allreduce 通信量 = $2 \cdot V$（$V$=模型大小，ring allreduce 大 $N$）。7B fp16 → 28 GB 通信量，是 [[communication bottleneck]] 的主要来源。[[overlap strategy|overlap]]（反向分层 allreduce）和 [[ZeRO (DeepSpeed)]]-2（reduce-scatter 分片）是优化。

### 2.5 混合精度的梯度

混合精度训练的梯度精度层次：

- **fp16 gradient**：反向用 fp16 算（快，tensor core），存 fp16。
- **fp32 master gradient**：reduction/累加用 fp32 保精度（fp16 易下溢），存 fp32 copy。
- **gradient scaling**：反向前 scale up（防 fp16 下溢），反向后 unscale。

故混合精度的 gradient memory 可能是 fp16（14GB）+ fp32 copy（28GB）= 42GB，或仅 fp16（取决于实现）。[[Fully Sharded Data Parallel]]/ZeRO 管理 gradient 的精度和分片。


## 3. 核心概念详解

### 3.1 梯度的产生与消费

```
forward → 存 activation
backward → 算 gradient（用 activation），存 gradient
allreduce gradient（分布式）
optimizer step → 用 gradient 更新参数（用 optimizer state）
释放 gradient
```

gradient 生命周期：反向产生 → allreduce → optimizer 用 → 释放。比 activation 长（activation 反向用完即释放），比 optimizer state 短（optimizer 全程）。

### 3.2 梯度分桶（bucketing）

DP allreduce 时，不等全部梯度算完再 allreduce，而是**分层**：算完一层（或一桶）的梯度立即 allreduce 该桶，与下一层的反向计算 overlap。bucket 大小 trade-off：大桶通信次数少但 overlap 差；小桶 overlap 好但 launch 多。PyTorch DDP 默认 25MB bucket，自动调节。是 [[overlap strategy]] 的核心实现。

### 3.3 gradient accumulation

大 batch 训练但显存不够存大 batch activation 时，分多次小 batch 前向/反向，**梯度累加**不更新，累积 N 次后一次 optimizer step。等效 batch = N × 小 batch。gradient memory 不变（每参数 1 梯度，累加到同一 buffer），activation 减 N×（每次只存小 batch 的）。是显存不够时的常用技巧。

### 3.4 gradient clipping

反向后、optimizer 前，对梯度做 **clip**（$\|g\| \to \min(\|g\|, c)$）防梯度爆炸。需先 allreduce 全局梯度（算 $\|g\|$），再 clip，再 optimizer。引入额外同步点（allreduce norm）。[[训练稳定性工程]] 的手段。

### 3.5 ZeRO-2 的梯度分片

[[ZeRO (DeepSpeed)]]-2：allreduce → **reduce-scatter**，每卡只保留自己负责的那部分梯度分片（$V/N$），其余释放。gradient memory 从 $V$ 减到 $V/N$。通信量同 allreduce（reduce-scatter + 后续 all-gather 参数）。是 gradient memory 的分布式优化。


## 4. 数学原理 / 公式

### 4.1 梯度大小

$$
M_{\text{grad}} = N_{\text{params}} \cdot p
$$

$p$ = 每元素字节数（fp16=2, fp32=4）。7B fp16 → $7 \times 10^9 \cdot 2 = 14$ GB。

### 4.2 混合精度的梯度

$$
M_{\text{grad, mixed}} = N_{\text{params}} \cdot (p_{\text{fp16}} + p_{\text{fp32}}) = N_{\text{params}} \cdot 6 \quad (\text{fp16+fp32 copy})
$$

7B → 42 GB（若存 fp32 copy）。仅 fp16 → 14 GB。

### 4.3 ZeRO-2 分片后

$$
M_{\text{grad, ZeRO-2}} = \frac{N_{\text{params}} \cdot p}{N_{\text{gpu}}}
$$

7B fp16，8 卡 → 14/8 = 1.75 GB/卡。

### 4.4 allreduce 通信量

$$
V_{\text{comm}} = \frac{2(N-1)}{N} \cdot N_{\text{params}} \cdot p \approx 2 V \quad (\text{大 } N)
$$

7B fp16 → ~28 GB 通信量。详见 [[communication bottleneck]]。

### 4.5 与 activation 的对比

7B，batch=8, seq=2048：

| 组成 | 大小 |
|---|---|
| gradient (fp16) | 14 GB |
| [[activation memory\|activation]] (标准, 无 ckpt) | 253 GB |
| [[activation memory\|activation]] (FA) | 47 GB |
| [[activation memory\|activation]] (FA + ckpt) | 4.3 GB |

gradient（14GB）相对 activation（标准 253GB）小很多，但相对 FA+ckpt 后的 activation（4.3GB）大。故**优化 activation 后，gradient 反成显存大头**，此时 ZeRO-2 分片 gradient 才关键。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟梯度显存

```python
def gradient_memory(n_params, bytes_per_elem=2, mixed_precision=False):
    """每参数 1 梯度. 混合精度时存 fp16 + fp32 copy."""
    if mixed_precision:
        return n_params * (2 + 4)   # fp16 + fp32
    return n_params * bytes_per_elem

# 7B
g16 = gradient_memory(7e9, 2)
g32 = gradient_memory(7e9, 4)
g_mix = gradient_memory(7e9, mixed_precision=True)
print(f'7B gradient fp16:        {g16/1e9:.1f} GB')
print(f'7B gradient fp32:        {g32/1e9:.1f} GB')
print(f'7B gradient mixed (h+f): {g_mix/1e9:.1f} GB')

# ZeRO-2 分片 (8 GPU)
g_z2 = g16 / 8
print(f'7B gradient ZeRO-2 (8卡): {g_z2/1e9:.2f} GB/卡 ({g16/g_z2:.0f}x 减)')
```

### 5.2 gradient accumulation

```python
def grad_accum_effective_batch(micro_batch, n_accum):
    """累积 N 次 micro-batch = 等效大 batch."""
    return micro_batch * n_accum

# 显存只够 batch=4, 想要 batch=32
eff = grad_accum_effective_batch(4, 8)
print(f'micro_batch=4, accum=8 -> 等效 batch={eff} (gradient 显存不变, activation 减 8x)')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: 反向传播（gradient 的产生）、损失函数（梯度的源）、链式法则。
- **下游（应用）**: [[optimizer state memory]]（optimizer 用 gradient）、[[ZeRO (DeepSpeed)]]/[[Fully Sharded Data Parallel]]（gradient 分片）、[[overlap strategy]]/[[Data Parallel]]（gradient allreduce + 分桶）、[[communication bottleneck]]（gradient 同步是通信主源）、gradient clipping/accumulation（训练技巧）、混合精度（gradient 精度管理）、[[训练稳定性工程]]（梯度爆炸/消失）。
- **对比 / 易混**:
  - **gradient vs [[activation memory\|activation]]**：反向产生的梯度（供 optimizer）vs 前向存的中间（供反向）。前者与 batch 无关，后者线性于 batch + $s^2$。
  - **gradient vs [[optimizer state memory\|optimizer state]]**：optimizer 的输入（梯度）vs optimizer 的内部状态（Adam m/v）。前者反向产生用完释放，后者全程常驻。gradient 是"原料"，optimizer state 是"加工机器"。
  - **gradient vs 权重**：参数本身 vs 参数的梯度。同形状同大小，但用途不同（权重是模型，梯度是更新方向）。


## 7. 常见误区与易错点

> [!warning] 误区 1：gradient 与 batch 有关
> 无关。每参数 1 梯度，形状 = 参数形状，与 batch 无关。batch 影响算梯度的 compute（大 batch 算更久）和 activation（线性增长），但不影响 gradient 张量大小。7B fp16 gradient 恒 14GB。

> [!warning] 误区 2：混淆 gradient 与 activation
> activation 是前向存的中间（供反向算 gradient 用），gradient 是反向算出的结果（供 optimizer 用）。两者都"动态"但来源和用途不同：activation 是"输入材料"，gradient 是"产出"。activation 与 batch/seq 强相关，gradient 与 batch 无关。

> [!warning] 误区 3：混淆 gradient 与 optimizer state
> gradient 是 optimizer 的**输入**（$g$），optimizer state 是 optimizer 的**内部状态**（Adam 的 $m, v$）。gradient 反向产生用完释放，optimizer state 全程常驻。Adam 的 optimizer state 是 gradient 的 2-4× 大小（$m, v$ fp32）。

> [!warning] 误区 4：忽略混合精度的 gradient copy
> 混合精度训练可能存 fp16 gradient（计算）+ fp32 master gradient（精度保真），gradient memory 翻 3×（14→42GB for 7B）。看具体实现（PyTorch amp 默认 fp16，部分存 fp32 copy）。

> [!warning] 误区 5：gradient 不释放
> optimizer step 后**必须**释放 gradient（`optimizer.zero_grad()`），否则下次反向梯度累加到旧梯度（bug）。PyTorch 默认 accumulate，需手动 zero_grad。gradient memory 在 zero_grad 后释放。

> [!warning] 误区 6：ZeRO-2 减通信
> [[ZeRO (DeepSpeed)]]-2 把 allreduce 改 reduce-scatter，**通信量不减**（同 allreduce 量），只减 gradient **显存**（分片到 $V/N$）。减通信是 ZeRO-3 的 all-gather（但反而增通信）。区分显存优化 vs 通信优化。


## 8. 延伸细节

### 8.1 gradient 的生命周期管理

```
zero_grad()           # 清零 gradient buffer
forward               # 存 activation
backward              # 算 gradient（写入 buffer，默认累加）
allreduce (DP)        # 跨卡同步 gradient
clip (可选)           # 防 explosion
optimizer.step()      # 用 gradient 更新参数
# 下一 step: zero_grad 释放/清零
```

PyTorch 的 `optimizer.zero_grad()` 清零 gradient buffer（不释放内存，复用 buffer）。gradient memory 在第一次 backward 后分配，之后复用，不反复 alloc/free。

### 8.2 gradient bucketing 的实现

PyTorch DDP `gradient_as_bucket_view=True`：gradient 是 bucket 的 view，bucket 装满触发 allreduce。bucket 大小 `bucket_cap_mb`（默认 25MB）。大模型自动多 bucket，小模型单 bucket。调优：大 bucket 减通信次数（launch 开销），小 bucket 增 overlap。trade-off。

### 8.3 gradient checkpointing 与 gradient 的关系

[[gradient checkpointing]] 优化的是 **activation**（不存中间，重算），不优化 gradient。gradient 仍每参数 1 个。两者正交：checkpointing 减 activation，ZeRO-2 减 gradient，可叠加。

### 8.4 梯度爆炸/消失与 gradient

梯度数值（非显存）问题：梯度爆炸（$\|g\|$ 大）需 clip，梯度消失（$\|g\|$ 小）需 residual/normalization。是训练稳定性（[[训练稳定性工程]]）问题，与 gradient memory（显存容量）是不同维度。但 gradient clipping 需先 allreduce 算全局 norm，引入同步点（[[CPU-GPU pipeline stall]] 风险）。

### 8.5 LLM 训练的 gradient

7B 训练 gradient 14GB（fp16）。单卡 A100 80GB 放得下（权重 14 + gradient 14 + optimizer 56 + activation 47(FA+ckpt) ≈ 131GB > 80GB），故 7B 单卡放不下，需 ZeRO/TP。70B 更需分布式。gradient 分片（ZeRO-2/3）是大模型训练标配。

### 8.6 梯度压缩

减 gradient 通信量（非显存）：量化（fp16→int8）、稀疏化（只传 top-k 大梯度）。有精度风险，训练稳定性敏感，慎用。详见 [[communication bottleneck]] §8.6。

### 8.7 峰值显存中的 gradient

峰值显存（反向结束、optimizer step 前）= 权重 + activation 峰值 + gradient + optimizer state。gradient 在峰值时刻存在（反向产生未释放）。减峰值：ZeRO-2 分片 gradient、gradient 卸载（[[offloading (CPU-NVMe)]]）。

---
相关: [[显存结构]] | [[activation memory]] | [[optimizer state memory]] | [[ZeRO (DeepSpeed)]] | [[Fully Sharded Data Parallel]] | [[Data Parallel]] | [[overlap strategy]] | [[communication bottleneck]] | [[gradient checkpointing]] | [[offloading (CPU-NVMe)]] | [[混合精度]] | [[训练稳定性工程]] | [[GPU memory snapshot]]
