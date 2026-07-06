# gradient checkpointing

> **所属章节**: [[优化技术]]
> **所属模块**: [[11-显存与内存系统]]
> **别名**: gradient checkpointing / 激活检查点 / activation checkpointing / rematerialization / 重计算
> **难度**: 中（需懂 [[activation memory]] + 前向/反向传播）


## 1. 一句话定义

**gradient checkpointing（梯度/激活检查点）** 是训练时**只保存部分层（检查点）的前向激活，反向时重算其余激活**的显存优化技术——以**反向多一次前向重算**（FLOP 增 ~33%）换**激活显存峰值减数十倍**。它针对 [[activation memory]]（训练显存的大头，长序列/大模型尤其爆炸）：全存激活需 $O(L \cdot \text{factor})$，checkpointing 后只存 $O(L)$ 个层边界（每层 1 个 $bsh$），减可达 10-60×。是长序列/大模型训练的**标配**，与 [[FlashAttention]]（减 attention 激活）叠加可让 7B 训练 activation 从 253 GB 降到 4.3 GB。代价是 FLOP 增 + 重算的带宽开销（反向 memory-bound 加重）。

> [!note] 三句话定位
> - **是什么**：前向只存检查点，反向重算中间激活，以算力换显存。
> - **为什么需要**：activation memory 是大头（253GB for 7B s=2048），checkpointing 减到 4.3GB（59×），让大模型训练可行。
> - **代价**：FLOP 增 ~33%（反向多一次前向），反向带宽开销增（重算读激活）。trade 算力/带宽换显存。


## 2. 为什么需要它（动机与背景）

### 2.1 activation memory 的爆炸

[[activation memory]] 是训练显存的大头。7B，batch=8, seq=2048：

| 配置 | activation |
|---|---|
| 标准 attention，无 ckpt | 253 GB |
| [[FlashAttention]]，无 ckpt | 47 GB |
| [[FlashAttention]] + ckpt | 4.3 GB |

标准 attention 的 253 GB 远超任何单卡（A100 80GB），故**必须优化**。FA 减 attention 的 $s^2$ 项到 47GB，仍超小卡。checkpointing 再减到 4.3GB（只存层边界）。两者叠加让 7B 训练可行。

### 2.2 重算换显存的思想

反向传播需前向的中间激活算梯度。全存 → 显存大。**只存部分（检查点）**，反向到某层时**重算**该层前向（从上一个检查点开始），得到中间激活。重算用算力（多一次前向）换显存（不存中间）。

- 显存：从 $O(L \cdot \text{factor})$ 减到 $O(L)$（检查点数 = 层数，每层 1 个边界）。
- 算力：反向多一次前向重算，FLOP 增 ~33%（原 fwd+bwd=3×fwd，加重算=4×fwd）。

### 2.3 长序列的刚需

长序列（$s$ 大）的 attention 激活 $\propto s^2$ 爆炸。即使 FA 减 $s^2$ 项，$s=8192$ 时 7B FA activation 仍 189GB（超卡）。checkpointing 必须叠加，减到 $\sim$L×bsh×2 = 8.6GB（7B s=8192）。**长上下文训练无 checkpointing 不可行**。

### 2.4 与 ZeRO/FA 的叠加

- [[FlashAttention]]：减 attention 内部激活（$s^2$ 项），不改变"存中间"逻辑。
- **gradient checkpointing**：减层间激活（存边界重算）。
- [[ZeRO (DeepSpeed)]]：分片权重/梯度/optimizer state。

三者正交，叠加使用：FA（减 attention）+ ckpt（减层间）+ ZeRO（减参数相关）= 大模型训练全套显存优化。


## 3. 核心概念详解

### 3.1 检查点策略

- **每层 checkpoint**（全 checkpointing）：每层都是检查点，只存每层输入（$L$ 个 $bsh$），反向每层重算。激活最省（$L \cdot bsh$），重算最多（全重算）。
- **每 $k$ 层 checkpoint**：每 $k$ 层存一次，反向重算 $k$ 层。trade 激活 vs 重算。
- **selective recomputation**：只 checkpoint 激活大的层（attention 矩阵），保留小的（layer norm 等）。最优 trade-off。

### 3.2 重算机制

```
forward:
  for layer i in 1..L:
    if i is checkpoint: save input_i   # 只存检查点输入
    compute layer i (不存中间)
backward:
  for layer i in L..1:
    if need activation of layer i:
      recompute forward from last checkpoint to i   # 重算获中间
    compute backward of layer i (用重算的中间)
```

重算在反向时按需进行，从最近的检查点重算到当前层。重算的中间用完即弃（不持久占显存）。

### 3.3 显存节省

全 checkpointing（每层 1 检查点）：

$$
M_{\text{act, ckpt}} = L \cdot b \cdot s \cdot h \cdot p
$$

vs 全存（FA 后）：

$$
M_{\text{act, FA}} = L \cdot 11 \cdot b \cdot s \cdot h \cdot p
$$

减 11×。vs 标准全存（253GB）减 59×（含 FA 的 $s^2$ 减除）。

### 3.4 FLOP 开销

训练 FLOP（无 ckpt）= forward + backward $\approx 3 \times \text{forward}$（反向约 2× 前向）。

全 checkpointing：+ 重算 forward $\to 4 \times \text{forward}$。

$$
\text{开销} = \frac{4-3}{3} = 33\%
$$

### 3.5 带宽开销（隐性代价）

重算时读激活算中间——这是 memory-bound 操作（[[memory bottleneck]]）。反向本就 memory-bound（读激活算梯度），重算再加一次前向的内存访问，带宽开销增。在 memory-bound 场景（长序列 decode-like）开销可能超 33%（算力闲但带宽紧）。故 checkpointing 在 compute-bound 训练（大 batch）代价小（算力闲可重算），memory-bound 场景代价大。

### 3.6 selective recomputation

不全 checkpoint，只重算激活大的部分。Megatron 的策略：

- **checkpoint attention 矩阵**（$s^2$ 项大）：重算 QK^T/softmax/PV，省 $3bn_hs^2$。
- **保留 layer norm / dropout mask**（小，$bsh$ 量级）：不重算（重算开销 > 省的显存）。

在显存和重算开销间取平衡，比全 checkpointing 更优（同显存省，重算少）。FlashAttention + selective recomputation 是长序列训练标配。


## 4. 数学原理 / 公式

### 4.1 激活显存

全存（FA 后）：

$$
M_{\text{full}} = L \cdot 11 \cdot bsh \cdot p
$$

全 checkpointing：

$$
M_{\text{ckpt}} = L \cdot bsh \cdot p
$$

减 $\frac{M_{\text{full}}}{M_{\text{ckpt}}} = 11$×。

### 4.2 FLOP

$$
\text{FLOP}_{\text{base}} = \text{fwd} + \text{bwd} = 3 \cdot \text{fwd}
$$

$$
\text{FLOP}_{\text{ckpt}} = \text{fwd} + \text{bwd} + \text{recompute} = 4 \cdot \text{fwd}
$$

$$
\text{overhead} = \frac{4-3}{3} \approx 33\%
$$

### 4.3 7B 实例（b=8, s=2048, fp16）

| 配置 | activation | FLOP 开销 |
|---|---|---|
| 标准 + 无 ckpt | 253 GB | 0% |
| FA + 无 ckpt | 47 GB | 0% |
| FA + 全 ckpt | 4.3 GB | +33% |
| FA + selective | ~10 GB | +10% |

selective 在显存和开销间最优。

### 4.4 长序列（s=8192）

| 配置 | activation |
|---|---|
| FA 无 ckpt | 189 GB |
| FA + ckpt | 8.6 GB |

减 22×。长序列 checkpointing 必需。

### 4.5 每 $k$ 层 checkpoint

$$
M_{\text{ckpt, k}} = \frac{L}{k} \cdot bsh \cdot p + k \cdot \text{(重算栈)}
$$

$k=1$（每层）：最省显存，重算最多。$k=L$（全存）：最费显存，无重算。中间 $k$ trade-off。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟显存 vs FLOP trade

```python
def activation_full(b, s, h, L, bpe=2, use_fa=True):
    """FA 后全存激活 = 11 * L * bsh * bpe."""
    factor = 11 if use_fa else 34
    return factor * b * s * h * bpe * L

def activation_ckpt(b, s, h, L, bpe=2):
    """全 checkpointing: 只存层边界, L * bsh * bpe."""
    return b * s * h * bpe * L

def train_flop(forward_flop, use_ckpt=False):
    """fwd + bwd + (recompute if ckpt). bwd ≈ 2*fwd."""
    fwd = forward_flop
    bwd = 2 * forward_flop
    recompute = forward_flop if use_ckpt else 0
    return fwd + bwd + recompute

# 7B FA
full = activation_full(8, 2048, 4096, 32, use_fa=True)
ckpt = activation_ckpt(8, 2048, 4096, 32)
print(f'7B FA 全激活:     {full/1e9:.1f} GB')
print(f'7B FA + ckpt:     {ckpt/1e9:.1f} GB ({full/ckpt:.1f}x 减)')

# FLOP 开销
base = train_flop(1)
ckpt_flop = train_flop(1, use_ckpt=True)
print(f'FLOP: 无 ckpt {base}x fwd, 有 ckpt {ckpt_flop}x fwd, 开销 +{(ckpt_flop-base)/base*100:.0f}%')

# 长序列
long_full = activation_full(8, 8192, 4096, 32, use_fa=True)
long_ckpt = activation_ckpt(8, 8192, 4096, 32)
print(f'7B s=8192 FA:     {long_full/1e9:.1f} GB, +ckpt {long_ckpt/1e9:.1f} GB ({long_full/long_ckpt:.1f}x 减)')
```

### 5.2 selective recomputation 模拟

```python
def selective_ckpt(b, s, h, L, n_h, bpe=2):
    """只重算 attention 矩阵(s^2 项), 保留 LN/dropout(bsh 项)."""
    # 保留: ~8 * bsh (LN, dropout mask 等小项)
    # 重算: attention 矩阵 (FA 已减, 但 selective 进一步省)
    saved = 8 * b * s * h * bpe * L   # 保留的小激活
    return saved

sel = selective_ckpt(8, 2048, 4096, 32, 32)
full = activation_full(8, 2048, 4096, 32, use_fa=True)
print(f'7B selective ckpt: {sel/1e9:.1f} GB (vs 全存 {full/1e9:.1f}, {full/sel:.1f}x 减, 重算开销少)')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[activation memory]]（优化的对象）、前向/反向传播（重算机制）、检查点算法。
- **下游（应用）**: [[FlashAttention]]（叠加减 attention 激活）、[[ZeRO (DeepSpeed)]]/[[Fully Sharded Data Parallel]]（叠加减参数相关）、[[offloading (CPU-NVMe)]]（叠加卸载激活）、长序列/大模型训练（标配）、[[memory bottleneck]]（重算的带宽代价）、[[GPU memory snapshot]]（诊断激活峰值）、sequence parallelism（切分激活）。
- **对比 / 易混**:
  - **gradient checkpointing vs [[ZeRO (DeepSpeed)]]**：前者减 activation（层间中间），后者减权重/gradient/optimizer state（参数相关）。正交，叠加用。
  - **gradient checkpointing vs [[FlashAttention]]**：前者减层间激活（存边界重算），后者减 attention 内部激活（tiling 不存中间）。正交，叠加用。
  - **gradient checkpointing vs [[offloading (CPU-NVMe)]]**：前者重算省显存（用算力），后者卸载省显存（用 CPU 内存/带宽）。trade 算力 vs 带宽。
  - **全 checkpointing vs selective**：全 checkpointing 最省显存但重算多，selective 只重算大激活更优 trade-off。


## 7. 常见误区与易错点

> [!warning] 误区 1：gradient checkpointing 减 gradient
> 不减。它减 **activation**（前向中间），不减 [[gradient memory|gradient]]（反向梯度）或 [[optimizer state memory|optimizer state]]。名字含"gradient"易误导，实际是 activation checkpointing（激活检查点）。

> [!warning] 误区 2：重算无开销
> 有。FLOP 增 ~33%（反向多一次前向），且重算是 memory-bound（读激活），带宽开销增。compute-bound 场景代价小（算力闲），memory-bound 场景代价大。不是免费午餐。

> [!warning] 误区 3：所有层都 checkpoint 最优
> 不一定。全 checkpointing 最省显存但重算最多。selective recomputation（只重算激活大的层）在显存和重算间更优。小激活（layer norm）重算的开销 > 省的显存，不值得 checkpoint。

> [!warning] 误区 4：checkpointing 与 FlashAttention 冲突
> 不冲突，正交。FA 减 attention 内部激活（$s^2$ 项），checkpointing 减层间激活（存边界）。叠加用：FA + ckpt = 7B 训练 activation 从 253GB → 4.3GB。是标配组合。

> [!warning] 误区 5：checkpointing 让训练慢 33%
> 不完全。FLOP 增 33%，但若训练是 memory-bound（反向读激活是瓶颈），重算的额外前向可能不增 wall-time（算力闲）。实际 wall-time 开销常 10-20%（小于 33% FLOP 开销）。compute-bound 训练才接近 33%。

> [!warning] 误区 6：checkpointing 改变计算结果
> 不改变。重算的前向与原前向数学等价（相同输入相同计算），梯度结果同。是**纯显存优化**，不影响模型/收敛。仅数值精度可能有微小差异（重算的浮点累加顺序，可忽略）。


## 8. 延伸细节

### 8.1 Megatron 的 selective recomputation

Megatron-LM 的 selective 策略：分析每层激活的显存占比和重算开销，选择性 checkpoint 大激活（attention 矩阵、MLP 扩大层），保留小激活（LN、dropout mask）。在保持显存省的同时减重算开销。论文 "Selecting Activation Checkpoints Efficiently" 系统化此方法。

### 8.2 PyTorch 的 checkpoint API

```python
# torch.utils.checkpoint
# from torch.utils.checkpoint import checkpoint
# def forward(x):
#     for layer in layers:
#         x = checkpoint(layer, x)   # 该层用 checkpoint, 反向重算
#     return x
```

`checkpoint(fn, *args)` 标记 fn 在反向时重算。PyTorch 自动处理。`use_reentrant` 参数控制重入策略（影响 autograd）。

### 8.3 unslo 优化

 Unsloth 等库优化 checkpointing 的重算：手写 kernel、fustion 重算的前向，减重算开销。让 4-bit 量化 + checkpointing 的微调更快。

### 8.4 与 sequence parallelism 叠加

Megatron 的 sequence parallelism：非 attention 部分的激活沿序列切分到各卡。与 checkpointing 叠加：每卡只存 $s/N$ 的检查点，进一步减。长序列训练组合：FA + selective ckpt + sequence parallelism + ZeRO。

### 8.5 checkpointing 与 offloading 的选择

显存不够时：

- **checkpointing**：用算力换显存（重算）。适合算力富余、显存紧。
- **[[offloading (CPU-NVMe)]]**：用 CPU 内存/带宽换显存（卸载）。适合算力也紧、CPU 内存富余。

两者可叠加（checkpoint 后的激活再卸载），但开销累加。按资源瓶颈选。

### 8.6 重算与 memory bottleneck

重算是 memory-bound（读激活算中间）。在反向本就 memory-bound 的训练中，重算加重带宽负担。故 checkpointing 在 compute-bound 训练（大 batch，算力饱和）代价小，memory-bound 训练（小 batch、长序列 decode-like）代价大。需用 [[nsys]] 测实际带宽利用率判断。

### 8.7 检查点位置的选择

检查点应在激活"边界"（层输入/输出），重算从上一个检查点到当前层。检查点间距 = 重算粒度：间距大 → 重算多、显存省；间距小 → 重算少、显存费。Megatron 自动选间距平衡。手动 checkpoint 需注意不要在 attention 内部切（破坏 FA 的 tiling）。

---
相关: [[优化技术]] | [[activation memory]] | [[FlashAttention]] | [[ZeRO (DeepSpeed)]] | [[Fully Sharded Data Parallel]] | [[offloading (CPU-NVMe)]] | [[memory bottleneck]] | [[GPU memory snapshot]] | [[gradient memory]] | [[Data Parallel]] | [[overlap strategy]] | [[nsys]]
