# activation memory

> **所属章节**: [[显存结构]]
> **所属模块**: [[11-显存与内存系统]]
> **别名**: activation memory / 激活显存 / 激活内存 / forward activations
> **难度**: 中（需懂前向/反向传播 + Transformer 结构）


## 1. 一句话定义

**activation memory（激活显存）** 是训练时前向传播**为反向计算梯度而保存的中间张量**——包括每层输入、QKV、attention 矩阵（QK^T/softmax/PV）、MLP 中间扩大层、residual 等。它是训练显存的三大组成之一（[[activation memory|activation]] + [[gradient memory|gradient]] + [[optimizer state memory|optimizer state]]），且对 **batch 和序列长度敏感**（长序列因 attention 矩阵 $b \cdot h \cdot s^2$ 平方增长）。优化激活显存是训练大模型/长序列的关键——[[gradient checkpointing]]（重算换显存）、[[FlashAttention]]（减 attention 中间）、混合精度（fp16/bf16 减半）。推理不存激活（无反向），故推理显存主要是权重 + KV cache。

> [!note] 三句话定位
> - **是什么**：前向存下供反向用的中间张量，训练显存三大组成之一。
> - **为什么大**：每层存多个中间 + attention 矩阵随 $s^2$ 平方增长，长序列爆炸。
> - **怎么优化**：gradient checkpointing（重算换显存）、FlashAttention（减 attention 中间）、混合精度（减半）。


## 2. 为什么需要它（动机与背景）

### 2.1 反向传播需要前向中间值

反向传播用链式法则算梯度，每层的梯度依赖**该层前向的输入和中间值**。如 $y = \text{relu}(Wx + b)$，反向时 $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x$（需 $x$）。故前向必须保存 $x$（及中间值）供反向用——这就是 activation memory 的来源。不存则反向无法算梯度（除非重算前向，即 [[gradient checkpointing]]）。

### 2.2 训练显存的三大组成

| 组成 | 大小（7B fp16） | 占比（训练） | 与 batch 关系 |
|---|---|---|---|
| **权重** | 14 GB | 静态 | 无关 |
| **[[gradient memory\|gradient]]** | 14 GB | 静态+动态 | 无关（每参数 1 梯度） |
| **[[optimizer state memory\|optimizer state]]** | 28-56 GB | 静态 | 无关（Adam m/v） |
| **[[activation memory\|activation]]** | 10-150 GB | 动态 | **线性于 batch，含 $s^2$ 项** |

小模型时 optimizer state 主导（Adam 2-4× 权重）；大 batch + 长序列时 activation 反超，成显存瓶颈。LLM 训练 activation 常是大头。

### 2.3 长序列的平方爆炸

attention 的 QK^T 和 softmax 矩阵是 $b \cdot n_{\text{head}} \cdot s \cdot s$——**随序列长度 $s$ 平方增长**。$s$ 从 512 → 8192（16×），attention 激活增 256×。这是长上下文训练显存爆炸的根因，也是 [[FlashAttention]]（减 attention 中间到 $O(s)$ 而非 $O(s^2)$）的革命性意义。

### 2.4 推理 vs 训练

推理无反向，不存激活（前向算完即弃），故推理显存 = 权重 + KV cache（[[KV cache management]]）。训练需存激活供反向，activation memory 是训练独有开销。这是训练显存 >> 推理显存的主因之一。


## 3. 核心概念详解

### 3.1 LLM 每层激活的构成

标准 Transformer 一层（self-attention + MLP）前向保存的中间（fp16，每元素 2 byte）：

| 中间张量 | 形状 | 大小 |
|---|---|---|
| input LN output | $b \cdot s \cdot h$ | $bsh$ |
| QKV | $3 \cdot b \cdot s \cdot h$ | $3bsh$ |
| QK^T 矩阵 | $b \cdot n_h \cdot s \cdot s$ | $bn_hs^2$ |
| softmax probs | $b \cdot n_h \cdot s \cdot s$ | $bn_hs^2$ |
| dropout mask | $b \cdot n_h \cdot s \cdot s$ | $bn_hs^2$ |
| attn context (PV) | $b \cdot s \cdot h$ | $bsh$ |
| attn output proj | $b \cdot s \cdot h$ | $bsh$ |
| residual add | $b \cdot s \cdot h$ | $bsh$ |
| post-attn LN | $b \cdot s \cdot h$ | $bsh$ |
| MLP up (扩大 4×) | $b \cdot s \cdot 4h$ | $4bsh$ |
| MLP activation | $b \cdot s \cdot 4h$ | $4bsh$ |
| MLP down | $b \cdot s \cdot h$ | $bsh$ |
| residual | $b \cdot s \cdot h$ | $bsh$ |

汇总：

$$
\text{per layer} \approx \underbrace{11 \cdot bsh}_{\text{非 attention 矩阵}} + \underbrace{3 \cdot bn_hs^2}_{\text{attention 矩阵}}
$$

对 $h = n_h \cdot d_k$（如 $h=4096$, $d_k=128$, $n_h=32$），attention 项 $= 3b \cdot (h/d_k) \cdot s^2$。

### 3.2 长短序列的激活特征

- **短序列**（$s$ 小）：$bsh$ 项主导，激活 $\propto b \cdot s$。
- **长序列**（$s$ 大）：$bn_hs^2$ 项主导，激活 $\propto s^2$，爆炸增长。

$s=2048, h=4096, n_h=32, d_k=128$：$bn_hs^2 = b \cdot 32 \cdot 2048^2 = b \cdot 1.34 \times 10^8$，而 $bsh = b \cdot 8.4 \times 10^6$，attention 项是 16×。长序列 attention 激活主导。

### 3.3 与 batch 的线性关系

激活（除 attention 矩阵的 $s^2$ 也线性于 $b$）全部线性于 batch $b$。batch 2× → 激活 2×。这是大 batch 训练显存压力大的原因（虽 compute 也线性，但显存是硬约束）。[[gradient checkpointing]] 和序列并行（sequence parallelism）是减激活的对策。

### 3.4 与 gradient / optimizer state 的区分

| 组成 | 存什么 | 何时分配 | 何时释放 | 与 batch 关系 |
|---|---|---|---|---|
| **activation** | 前向中间 | 前向时 | 反向用完即释放 | 线性于 batch + $s^2$ |
| **[[gradient memory\|gradient]]** | 参数梯度 | 反向时 | optimizer 用完释放 | 与 batch 无关（每参数 1 梯度） |
| **[[optimizer state memory\|optimizer state]]** | Adam m/v | 训练全程 | 训练结束 | 与 batch 无关 |

activation 是**动态**（前向分配、反向释放，生命周期短但峰值大），gradient 半动态，optimizer 静态。峰值显存 = 权重 + optimizer + 激活峰值（前向结束/反向开始时最大）。


## 4. 数学原理 / 公式

### 4.1 每层激活

$$
M_{\text{act/layer}} \approx (11 \cdot bsh + 3 \cdot bn_hs^2) \cdot p
$$

$p$ 是每元素字节数（fp16=2, fp32=4）。

### 4.2 总激活

$$
M_{\text{act}} = L \cdot (11 \cdot bsh + 3 \cdot bn_hs^2) \cdot p
$$

$L$ 是层数。

### 4.3 7B 估算（$h=4096, L=32, n_h=32, d_k=128$, fp16）

$b=8, s=2048$：

$$
M_{\text{act/layer}} = (11 \cdot 8 \cdot 2048 \cdot 4096 + 3 \cdot 8 \cdot 32 \cdot 2048^2) \cdot 2
$$

$$
= (7.4 \times 10^8 + 3.2 \times 10^9) \cdot 2 \approx 7.9 \text{ GB/layer}
$$

$$
M_{\text{act}} = 32 \cdot 7.9 \approx 253 \text{ GB}
$$

约 253 GB（无 checkpointing，标准 attention，attention 矩阵的 $s^2$ 项占 ~81% 主导）。这是 7B 训练 batch=8 seq=2048 的激活峰值——远超权重 14GB。**activation 是大头，且 attention 的 $s^2$ 项主导**。这也说明为何长序列训练**必须**用 [[FlashAttention]]（去 $s^2$ 项）+ [[gradient checkpointing]]（减峰值）。

### 4.4 gradient checkpointing 后

只存层边界（每层 1 个 $bsh$ 输入），中间重算：

$$
M_{\text{act/ckpt}} \approx L \cdot bsh \cdot p = 32 \cdot 8 \cdot 2048 \cdot 4096 \cdot 2 \approx 4.3 \text{ GB}
$$

减 59×（253 GB → 4.3 GB）。代价：反向多一次前向重算，FLOP 增 ~33%。详见 [[gradient checkpointing]]。

### 4.5 FlashAttention 后

[[FlashAttention]] 把 attention 中间（QK^T/softmax/PV，$3bn_hs^2$ 项）从 HBM 存储减到 $O(s)$ 流式计算（tiling 在 smem 内完成不写回 HBM）：

$$
M_{\text{act/FA}} = L \cdot (11 \cdot bsh + O(s)) \cdot p \approx L \cdot 11 \cdot bsh \cdot p
$$

减 attention 平方项。对 $s=2048$ 上面例子：$32 \cdot 11 \cdot 8 \cdot 2048 \cdot 4096 \cdot 2 \approx 47$ GB（vs 253 GB，减 5.4×）。长序列收益更大（去 $s^2$，$s=8192$ 时标准 3.5 TB vs FA 189 GB，减 18×）。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟激活显存估算

```python
def activation_memory(batch, seq, hidden, n_layers, n_heads, dk=128,
                      bytes_per_elem=2, factor=34, use_flash_attn=False):
    """每层激活 ≈ (11*bsh + [3*bn_h*s^2 if 标准 else 0]) * bytes.
    factor=34 是 Megatron 经验估算（含 attention 矩阵）."""
    bsh = batch * seq * hidden
    if use_flash_attn:
        per_layer = 11 * bsh * bytes_per_elem   # 无 attention 大矩阵
    else:
        attn_mat = 3 * batch * n_heads * seq * seq * bytes_per_elem
        per_layer = 11 * bsh * bytes_per_elem + attn_mat
    return per_layer * n_layers

# 7B: h=4096, L=32, n_h=32
full = activation_memory(8, 2048, 4096, 32, 32)
fa = activation_memory(8, 2048, 4096, 32, 32, use_flash_attn=True)
print(f'7B 标准 attention (b=8,s=2048): {full/1e9:.1f} GB')
print(f'7B FlashAttention:              {fa/1e9:.1f} GB ({full/fa:.1f}x 减)')

# 长序列 s=8192 (16x)
long_full = activation_memory(8, 8192, 4096, 32, 32)
long_fa = activation_memory(8, 8192, 4096, 32, 32, use_flash_attn=True)
print(f'7B 标准 s=8192: {long_full/1e9:.1f} GB (vs 2048: {long_full/full:.0f}x)')
print(f'7B FA s=8192:    {long_fa/1e9:.1f} GB (vs 2048: {long_fa/fa:.0f}x)')
```

### 5.2 gradient checkpointing 对照

```python
def activation_with_checkpoint(batch, seq, hidden, n_layers, bytes_per_elem=2):
    """checkpointing: 只存层边界输入, 每层 ~bsh."""
    per_layer = batch * seq * hidden * bytes_per_elem
    return per_layer * n_layers

ckpt = activation_with_checkpoint(8, 2048, 4096, 32)
print(f'7B checkpointing 后: {ckpt/1e9:.1f} GB (vs 全激活 {fa/1e9:.1f}, {fa/ckpt:.0f}x 减)')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: 前向传播 / 反向传播（activation 的存在理由）、[[Transformer]]/attention（激活的来源结构）。
- **下游（应用）**: [[gradient checkpointing]]（重算换激活显存）、[[FlashAttention]]（减 attention 激活的 $s^2$ 项）、[[ZeRO (DeepSpeed)]]/[[Fully Sharded Data Parallel]]（分片激活）、混合精度（减半激活）、sequence parallelism（切分激活）、[[offloading (CPU-NVMe)]]（激活卸载到 CPU）、[[memory bottleneck]]（激活读带宽是反向 memory-bound 主因）、[[GPU memory snapshot]]（诊断激活占用）、[[训练显存估计]]（激活是训练峰值显存公式 $M\approx 16N+M_{\text{act}}$ 的动态项，与 batch/seq 相关）。
- **对比 / 易混**:
  - **activation vs [[gradient memory\|gradient]]**：前向中间（供反向用）vs 反向梯度（供 optimizer 用）。前者线性于 batch+$s^2$，后者与 batch 无关。
  - **activation vs [[optimizer state memory\|optimizer state]]**：动态短生命周期 vs 静态全程。前者随 batch/seq 变，后者固定。
  - **activation vs KV cache**：训练存激活供反向 vs 推理存 KV 供自回归。两者都是"中间状态"但用途不同。


## 7. 常见误区与易错点

> [!warning] 误区 1：activation 只存每层输出
> 不止。每层存多个中间（QKV、QK^T、softmax、MLP 扩大层等），因子约 11×bsh（标准）或更多。只存层输出是 [[gradient checkpointing]] 后的简化，非默认。

> [!warning] 误区 2：激活与 batch 无关
> 有关。激活线性于 batch（所有项含 $b$）。大 batch 训练显存压力大，部分因激活。但 gradient/optimizer 与 batch 无关，区分清楚。

> [!warning] 误区 3：忽略 attention 矩阵的 $s^2$ 增长
> attention 的 QK^T/softmax 是 $b \cdot n_h \cdot s^2$，长序列平方爆炸。$s$ 16× → attention 激活 256×。这是长上下文训练显存瓶颈，[[FlashAttention]] 减此项是革命。

> [!warning] 误区 4：推理也存激活
> 推理无反向，不存激活（前向算完即弃）。推理显存 = 权重 + KV cache。把推理显存大头算成激活是错的。训练显存 >> 推理显存的部分主因是 activation + optimizer state。

> [!warning] 误区 5：activation 是静态的
> 动态。前向时分配、反向用完释放，生命周期短但峰值大（前向结束/反向开始时最大）。峰值显存 = 权重 + optimizer + 激活峰值。[[gradient checkpointing]] 减峰值。

> [!warning] 误区 6：FP16 和 BF16 激活一样大
> 大小一样（都 2 byte/元素），但 BF16 数值范围大（指数位多）不易溢出，训练更稳。激活显存两者同，但精度特性不同。详见混合精度。


## 8. 延伸细节

### 8.1 Megatron 的激活估算

Megatron-LM 论文给的标准估算：每层激活约 $34 \cdot b \cdot s \cdot h$ bytes（fp16，含 attention 矩阵，对 $s$ 不太大时近似）。本笔记的 11 项 + $3bn_hs^2$ 是更细拆解，两者量级一致。

### 8.2 sequence parallelism

Megatron 的 sequence parallelism：把非 attention 部分的激活（LN、dropout 等）沿序列维度切分到各卡，每卡只存 $s/N$。减激活 $N$×。与 tensor parallelism 配合（attention 仍 TP，非 attention SP）。

### 8.3 selective recomputation

[[gradient checkpointing]] 的精细版：不重算全部，只重算激活大的部分（attention 矩阵），保留小的（layer norm 输出等）。在显存和重算开销间取平衡。FlashAttention + selective recomputation 是长序列训练标配。

### 8.4 activation 卸载

[[offloading (CPU-NVMe)]] 把激活卸到 CPU 内存/NVMe，反向时取回。缓解 GPU 显存，但增 PCIe 传输（[[memory bottleneck]] 的带宽开销）。适合显存极紧但 CPU 内存富余的场景。

### 8.5 峰值显存管理

训练峰值显存出现在前向结束/反向开始时（所有激活已存、optimizer 在、gradient 开始产生）。管理峰值：

- [[gradient checkpointing]]：减激活峰值。
- 梯度分桶（bucketing）：分散 gradient 分配/释放。
- activation 卸载：前向存 CPU，反向取。

[[GPU memory snapshot]] 可捕捉峰值时刻的显存构成。

### 8.6 FlashAttention 的激活影响

[[FlashAttention]] 不改前向计算结果（attention 输出同），但**不存** QK^T/softmax/PV 中间到 HBM（在 smem tiling 内完成）。故激活的 $3bn_hs^2$ 项消失，减为 $O(s)$（只存必要的边界）。反向时重算 attention（在 smem 内，快）。是 activation memory 优化的核心手段，长序列尤其关键。

### 8.7 与 memory bottleneck 的关系

反向传播时读激活算梯度，是 memory-bound（激活读带宽是瓶颈，[[memory bottleneck]]）。减激活（checkpointing/FA）不仅省显存（静态容量），也减带宽（动态读取），提反向 throughput。是显存 + 带宽的双重优化。

---
相关: [[显存结构]] | [[gradient memory]] | [[optimizer state memory]] | [[训练显存估计]] | [[gradient checkpointing]] | [[FlashAttention]] | [[ZeRO (DeepSpeed)]] | [[Fully Sharded Data Parallel]] | [[offloading (CPU-NVMe)]] | [[memory bottleneck]] | [[GPU memory snapshot]] | [[KV cache management]] | [[Transformer]] | [[attention]] | [[混合精度]]
