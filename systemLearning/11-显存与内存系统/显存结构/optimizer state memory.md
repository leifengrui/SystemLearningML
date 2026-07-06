# optimizer state memory

> **所属章节**: [[显存结构]]
> **所属模块**: [[11-显存与内存系统]]
> **别名**: optimizer state memory / 优化器状态显存 / optimizer states / Adam states
> **难度**: 中（需懂 optimizer 原理 + 训练显存构成）


## 1. 一句话定义

**optimizer state memory（优化器状态显存）** 是 optimizer（如 **Adam/AdamW**）为自适应更新参数而**全程保存的内部状态张量**——Adam 的一阶动量 $m$（梯度指数移动平均）和二阶动量 $v$（梯度平方指数移动平均），每个状态与参数同形状。它是训练显存三大组成之一（[[activation memory|activation]] + [[gradient memory|gradient]] + **optimizer state**），且常是**最大头**——Adam fp32 的 state 是权重的 4×（7B Adam = 56 GB > 权重 14 GB）。是 LLM 训练显存瓶颈的主因，催生 [[ZeRO (DeepSpeed)]]-1（state 分片）、8-bit optimizer、[[offloading (CPU-NVMe)]]（state 卸载）等优化。SGD 无 state（或仅 1 个 momentum），故 SGD 训练显存远小于 Adam。

> [!note] 三句话定位
> - **是什么**：optimizer 的内部状态（Adam 的 m, v），全程常驻，与参数同形状。
> - **为什么大**：Adam 有 2 个 state（m, v）且 fp32，是权重的 4×（7B = 56GB）。训练显存最大头。
> - **怎么优化**：ZeRO-1 分片（V/N）、8-bit optimizer（减 4×）、offloading（卸 CPU/NVMe）、用 SGD（无 state 但收敛差）。


## 2. 为什么需要它（动机与背景）

### 2.1 自适应优化器需历史统计

Adam 更新：

$$
\theta \leftarrow \theta - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

$m$（一阶动量，梯度的指数移动平均）和 $v$（二阶动量，梯度平方的指数移动平均）是**跨 step 累积的历史统计**，让更新方向更稳、自适应调步长（大梯度方向减速、小梯度方向加速）。这些 state 必须跨 step 保存（不像 [[gradient memory|gradient]] 用完释放），故全程常驻显存。

### 2.2 为什么是最大头

| optimizer | state 数 | 精度 | bytes/param | 7B 大小 |
|---|---|---|---|---|
| **SGD** | 0 | - | 0 | 0 GB |
| **SGD+momentum** | 1 | fp32 | 4 | 28 GB |
| **Adam/AdamW** | 2 (m,v) | fp32 | 8 | **56 GB** |
| **8-bit Adam** | 2 | int8 | 2 | 14 GB |

Adam（LLM 训练标配）的 state 是权重的 4×（fp16 权重 14GB vs state 56GB）。加上权重 + gradient + activation，Adam 训练显存远超推理。**optimizer state 是 LLM 训练显存最大组成**（常占 50%+）。

### 2.3 fp32 精度的必要性

Adam 的 $m, v$ 用 **fp32** 而非 fp16：

- $v$ 是梯度平方累加，数值易下溢（fp16 最小正数 ~6e-8，平方后更小）。
- $m$ 的指数移动平均需高精度累加（fp16 累加误差大）。
- fp32 保精度，防训练不稳定（[[训练稳定性工程]]）。

故即使混合精度训练（权重/梯度 fp16），optimizer state 仍 fp32，是显存大头。8-bit Adam 用 int8 量化 state 减显存，但有精度风险。

### 2.4 三大组成中的位置

| 组成 | 7B fp16 | 与 batch | 生命周期 | 占比 |
|---|---|---|---|---|
| 权重 | 14 GB | 无关 | 全程 | 14% |
| [[gradient memory\|gradient]] | 14 GB | 无关 | 反向→opt 释放 | 14% |
| **optimizer state (Adam)** | **56 GB** | **无关** | **全程** | **~55%** |
| [[activation memory\|activation]] (FA+ckpt) | 4.3 GB | 线性 | 前向→反向释放 | 4% |

optimizer state（56GB）占训练显存大头（权重 14 + grad 14 + opt 56 + act 4.3 ≈ 88GB，opt 占 64%）。减 optimizer state 是大模型训练显存优化的首要目标。

### 2.5 为什么 SGD 显存小但不用

SGD 无 state（或仅 1 个 momentum），显存小，但 LLM 训练收敛差（需精细 lr schedule、易卡 saddle point）。Adam 自适应、收敛稳，是 LLM 标配，代价是 state 显存大。trade-off：用 Adam + ZeRO-1 分片，而非 SGD。

### 2.6 ZeRO-1 的分片

[[ZeRO (DeepSpeed)]]-1：optimizer state 分片到 $N$ 卡，每卡只存 $1/N$。7B Adam 8 卡 → 每卡 7 GB（vs 56 GB）。通信不增（仍 allreduce gradient），纯显存优化。是 optimizer state 的核心优化，大模型训练标配。


## 3. 核心概念详解

### 3.1 Adam 的 m 和 v

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

- $m$：一阶动量（梯度均值），方向记忆。
- $v$：二阶动量（梯度方差），自适应步长（大方差方向减速）。
- $\beta_1=0.9, \beta_2=0.999$（默认）。

两者跨 step 累积，全程常驻，各与参数同形状。

### 3.2 各 optimizer 的 state 大小

| optimizer | state | bytes/param (fp32) | 7B 大小 |
|---|---|---|---|
| SGD | 无 | 0 | 0 |
| SGD+momentum | $u$ (momentum) | 4 | 28 GB |
| Adam | $m, v$ | 8 | 56 GB |
| AdamW | $m, v$（同 Adam，解耦 weight decay） | 8 | 56 GB |
| Adafactor | 因子化的 $v$（$R, C$ 而非全 $v$） | ~2 (因子化) | ~14 GB |
| 8-bit Adam | $m, v$ int8 | 2 | 14 GB |

### 3.3 与 gradient / 权重的区分

| 组成 | 是什么 | 生命周期 | 与 batch |
|---|---|---|---|
| 权重 | 模型参数 $\theta$ | 全程 | 无关 |
| [[gradient memory\|gradient]] | 当前 step 梯度 $g_t$ | 反向→opt 释放 | 无关 |
| **optimizer state** | 跨 step 历史统计 $m, v$ | **全程** | 无关 |

三者都与 batch 无关，但生命周期不同：权重全程、state 全程、gradient 用完释放。gradient 是 optimizer 的**输入**（$g_t$），state 是 optimizer 的**内部状态**（$m, v$ 跨 step 累积）。

### 3.4 8-bit optimizer

8-bit Adam（`bitsandbytes`）：$m, v$ 用 int8 量化存（+ 量化元数据 fp32），state 减 4×（56→14GB for 7B）。精度损失小（动态量化），训练稳定接近 fp32 Adam。是显存优化的高性价比手段。LLM 训练/微调常用（QLoRA 等）。

### 3.5 ZeRO 的分片层次

| ZeRO 阶段 | 分片对象 | 7B/8卡 显存 | 通信变化 |
|---|---|---|---|
| ZeRO-0 | 无 | opt 56 GB | - |
| **ZeRO-1** | **optimizer state** | **opt 7 GB** | 不变 |
| ZeRO-2 | + gradient | opt 7 + grad 1.75 = 8.75 GB | 不变 |
| ZeRO-3 | + 参数 | ~3 GB（全分片） | 增（all-gather 参数） |

ZeRO-1 是减 optimizer state 的纯显存优化，无通信代价，是首选。详见 [[ZeRO (DeepSpeed)]]。


## 4. 数学原理 / 公式

### 4.1 optimizer state 大小

$$
M_{\text{opt}} = N_{\text{params}} \cdot k \cdot p_{\text{state}}
$$

- $k$：state 数（Adam=2, SGD=0）。
- $p_{\text{state}}$：每元素字节数（fp32=4, int8=1）。

### 4.2 Adam（fp32）

$$
M_{\text{Adam}} = N_{\text{params}} \cdot 2 \cdot 4 = 8 \cdot N_{\text{params}} \text{ (bytes)}
$$

7B → $7 \times 10^9 \cdot 8 = 56$ GB。

### 4.3 与权重的比

fp16 权重 $M_W = 2 N_{\text{params}}$。Adam state $M_{\text{opt}} = 8 N_{\text{params}}$。

$$
\frac{M_{\text{opt}}}{M_W} = \frac{8}{2} = 4
$$

故 Adam state 是 fp16 权重的 4×。

### 4.4 ZeRO-1 分片后

$$
M_{\text{opt, ZeRO-1}} = \frac{N_{\text{params}} \cdot 8}{N_{\text{gpu}}}
$$

7B, 8 卡 → $56/8 = 7$ GB/卡。

### 4.5 8-bit Adam

$$
M_{\text{8bit Adam}} = N_{\text{params}} \cdot 2 \cdot 1 = 2 \cdot N_{\text{params}}
$$

7B → 14 GB（vs 56 GB，减 4×）。与 fp16 权重同大。

### 4.6 训练总显存（7B, 8卡, ZeRO-1, FA+ckpt）

$$
M_{\text{total/card}} = \underbrace{14}_{\text{权重}} + \underbrace{14}_{\text{gradient}} + \underbrace{7}_{\text{opt (ZeRO-1)}} + \underbrace{4.3}_{\text{act (FA+ckpt)}} \approx 39 \text{ GB}
$$

A100 40GB 刚好放得下（无 ZeRO-1 时 opt 56GB → 总 88GB 放不下）。**ZeRO-1 是 7B 单卡训练可行的关键**。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 optimizer state 显存

```python
def optimizer_state_memory(n_params, opt='adam', bytes_per_state=4):
    """k = state 数. SGD=0, momentum=1, Adam/AdamW=2."""
    k = {'sgd':0, 'momentum':1, 'adam':2, 'adamw':2}[opt]
    return n_params * k * bytes_per_state

# 7B
print(f'7B SGD:           {optimizer_state_memory(7e9,"sgd")/1e9:.0f} GB')
print(f'7B SGD+momentum:  {optimizer_state_memory(7e9,"momentum")/1e9:.0f} GB')
print(f'7B Adam fp32:     {optimizer_state_memory(7e9,"adam")/1e9:.0f} GB')
print(f'7B 8-bit Adam:    {optimizer_state_memory(7e9,"adam",1)/1e9:.0f} GB')

# ZeRO-1 分片 (8 GPU)
print(f'7B Adam ZeRO-1 (8卡): {optimizer_state_memory(7e9,"adam")/8/1e9:.1f} GB/卡')

# 训练总显存对比
def train_mem(n, opt='adam', bytes_per_state=4, n_gpu=1, use_ckpt=True):
    w = n*2; g = n*2
    o = optimizer_state_memory(n, opt, bytes_per_state)
    if n_gpu>1: o = o/n_gpu
    a = (4.3e9 if use_ckpt else 47e9)
    return w + g + o + a
print(f'7B 单卡 (无 ZeRO, FA+ckpt): {train_mem(7e9)/1e9:.1f} GB')
print(f'7B 8卡 ZeRO-1 (FA+ckpt):    {train_mem(7e9,n_gpu=8)/1e9:.1f} GB/卡')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: optimizer 算法（Adam/SGD，state 的来源）、[[gradient memory|gradient]]（state 的输入 $g_t$）、指数移动平均（$m, v$ 的数学）。
- **下游（应用）**: [[ZeRO (DeepSpeed)]]/[[Fully Sharded Data Parallel]]（state 分片，ZeRO-1/3）、[[offloading (CPU-NVMe)]]（state 卸载）、8-bit optimizer（量化 state）、混合精度（state 精度管理）、[[GPU memory snapshot]]（诊断 state 占用）、LLM 训练显存预算。
- **对比 / 易混**:
  - **optimizer state vs [[gradient memory\|gradient]]**：optimizer 内部状态（跨 step 的 $m,v$，全程）vs optimizer 输入（当前 step 的 $g$，用完释放）。state 是"机器记忆"，gradient 是"本次原料"。
  - **optimizer state vs 权重**：optimizer 的状态（$m,v$）vs 模型参数（$\theta$）。同形状但不同用途。state 更新参数，参数是模型。
  - **Adam vs SGD 显存**：Adam 2 个 state（56GB for 7B），SGD 0 个（0GB）。Adam 显存大但收敛好，SGD 显存小但 LLM 收敛差。


## 7. 常见误区与易错点

> [!warning] 误区 1：optimizer state = gradient
> 不同。gradient 是当前 step 的梯度 $g_t$（反向产生，用完释放）。optimizer state 是跨 step 累积的 $m, v$（全程常驻）。gradient 是输入，state 是内部累积。Adam 的 state 不是 gradient 本身，是 gradient 的移动平均。

> [!warning] 误区 2：Adam 和 SGD state 一样大
> 不同。SGD 无 state（0），SGD+momentum 1 个（28GB for 7B fp32），Adam 2 个（56GB）。Adam 的 $m, v$ 让它显存 2× SGD+momentum。LLM 用 Adam 故显存大。

> [!warning] 误区 3：混合精度减 optimizer state
> 不减。混合精度训练的 optimizer state 仍 **fp32**（$m, v$ 需高精度累加防下溢/误差）。只有权重和梯度减半（fp16），state 不减。8-bit Adam 才减 state（int8）。

> [!warning] 误区 4：optimizer state 随 batch 增长
> 不增长。state 与参数同形状，与 batch 无关。batch 影响算梯度的 compute 和 [[activation memory|activation]]（线性+$s^2$），不影响 state 大小。7B Adam state 恒 56GB。

> [!warning] 误区 5：忽略 ZeRO-1 的性价比
> ZeRO-1 分片 optimizer state，**无通信代价**（仍 allreduce gradient），纯显存减 $N$×。是大模型训练**首选**优化（先 ZeRO-1 再考虑 ZeRO-2/3）。常被忽略而直接上 ZeRO-3（增通信），得不偿失。

> [!warning] 误区 6：8-bit Adam 总是等效
> 8-bit Adam 减 state 4×，但量化有精度损失。对 lr 敏感或训练不稳定的场景（小数据微调、高 lr）可能收敛差。大模型预训练通常等效，微调需验证。

> [!warning] 误区 7：optimizer state 能释放
> 不能（训练全程）。state 跨 step 累积，必须全程常驻。只能分片（ZeRO-1）或卸载（offloading）减 GPU 显存，不能释放。训练结束才释放。


## 8. 延伸细节

### 8.1 AdamW 与 weight decay

AdamW 是 Adam 的变体，**解耦** weight decay（$L_2$ 正则）与梯度更新。state 同 Adam（$m, v$，56GB for 7B）。LLM 训练常用 AdamW（decoupled weight decay 更稳）。详见 optimizer 笔记。

### 8.2 Adafactor 的省显存

Adafactor 因子化二阶动量 $v$：不存全 $N \times M$ 的 $v$，而存行因子 $R$ 和列因子 $C$（$N+M$）。state 从 $2N_{\text{params}}$ 减到 $\sim N_{\text{params}}$（因子化 $v$）。T5/早期 LLM 训练用，但收敛常不如 Adam，现已少用（被 8-bit Adam + ZeRO 取代）。

### 8.3 8-bit optimizer 的实现

`bitsandbytes` 库：$m, v$ 量化 int8 存（+ 动态量化元数据 fp32）。读取时反量化 fp32 计算。state 减 4×，compute 增少量（量化/反量化）。QLoRA（4-bit 量化 + 8-bit Adam）是 4-bit 微调标配，让单卡微调 70B 可行。

### 8.4 optimizer state 的卸载

[[offloading (CPU-NVMe)]]：optimizer state 卸到 CPU 内存/NVMe，optimizer step 时取回。减 GPU 显存（state 大头），但增 PCIe 传输（[[CPU-GPU pipeline stall]] 风险）。适合显存极紧但 CPU 内存富余（如单卡训大模型）。ZeRO-Offload 是代表。

### 8.5 LLM 训练显存预算

7B 训练（Adam, fp16 权重/梯度, fp32 state, FA+ckpt activation）：

| 组成 | 单卡（无 ZeRO） | 8卡 ZeRO-1 |
|---|---|---|
| 权重 | 14 GB | 14 GB |
| gradient | 14 GB | 14 GB（或 ZeRO-2: 1.75） |
| optimizer state | 56 GB | 7 GB |
| activation | 4.3 GB | 4.3 GB |
| **合计** | **88 GB** | **39 GB** |

A100 40GB：无 ZeRO 放不下，ZeRO-1 刚好。H100 80GB：无 ZeRO 也能放（但浪费，仍用 ZeRO 提 batch）。**optimizer state 分片是 7B+ 训练的关键**。

### 8.6 optimizer state 与训练稳定性

Adam 的 $m, v$ 数值稳定性影响训练：$v$ 下溢（fp16）致除以 0、$m$ 累加误差致方向偏。fp32 state 保稳定。训练 instability（loss spike）有时与 optimizer state 数值有关，[[训练稳定性工程]] 关注。

### 8.7 推理无 optimizer state

推理无 optimizer（不更新参数），故无 optimizer state。推理显存 = 权重 + KV cache。optimizer state 是**训练独有**开销，是训练显存 >> 推理显存的主因之一（连同 activation）。

### 8.8 GPU memory snapshot 中的 optimizer state

[[GPU memory snapshot]] 的显存分类中，optimizer state 是大头（~55%）。snapshot 能定位 optimizer state 占比，确认是否需 ZeRO-1/8-bit/offload 优化。

---
相关: [[显存结构]] | [[activation memory]] | [[gradient memory]] | [[ZeRO (DeepSpeed)]] | [[Fully Sharded Data Parallel]] | [[offloading (CPU-NVMe)]] | [[混合精度]] | [[训练稳定性工程]] | [[GPU memory snapshot]] | [[Data Parallel]] | [[overlap strategy]]
