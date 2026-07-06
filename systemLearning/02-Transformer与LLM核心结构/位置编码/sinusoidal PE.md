# sinusoidal PE

> **所属章节**: [[位置编码]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中级

## 1. 一句话定义

**正弦位置编码（sinusoidal position encoding，sinusoidal PE，正弦位置编码/正余弦位置编码）** 是原版 Transformer（《Attention Is All You Need》）使用的**绝对位置编码**：用不同频率的 $\sin/\cos$ 函数为每个位置构造一个唯一向量 $p_t\in\mathbb{R}^d$，加到 token 嵌入上；其设计意图是**让相对位置可由线性投影解出**，从而兼具可学习性与理论外推性，是 [[位置编码]] 中"加法式绝对 PE"的代表。

## 2. 为什么需要它（动机与背景）

[[位置编码]] 总览已说明 attention 无序、需注入位置。具体到"用什么向量当 $p_t$"，有两条路：

- **learned PE**：直接学一个 $L\times d$ 查表。简单，但**超长度不能外推**（没见过的位置无向量），且无结构先验。
- **sinusoidal PE**：用公式生成，任意位置都有编码，理论可外推；且不同维度不同频率，自带"多尺度位置"先验。

原版作者选 sinusoidal 的两个理由：

1. **外推**：公式对任意 $t$ 成立，理论上可生成训练没见过的位置编码。
2. **相对位置可线性表达**：$\sin/\cos$ 的平移性质使 $p_{t+k}$ 是 $p_t$ 的线性变换，模型可用线性层（attention 的投影本质是线性）解出相对位置——不像 learned PE 那样位置间无此关系。

> [!note] 与 RoPE 的区别
> 同样是 sin/cos，[[RoPE]] 用它做**旋转**（作用 Q/K，相对位置内建于内积）；sinusoidal PE 用它做**加法**（叠加在输入，绝对位置）。前者耦合紧、外推好；后者耦合松、实测外推有限。但 RoPE 的频率设计直接继承自 sinusoidal，二者同源。

## 3. 核心概念详解

### 3.1 公式

对位置 $t=0,1,\dots,L-1$ 和维度 $i=0,\dots,d-1$，按奇偶配对：

$$
PE_{t,2i}=\sin\!\left(\frac{t}{10000^{2i/d}}\right),\quad
PE_{t,2i+1}=\cos\!\left(\frac{t}{10000^{2i/d}}\right)
$$

- $2i,2i+1$：相邻两维一对，共享频率。
- $10000^{2i/d}$：分母，频率随维度 $i$ 递减。
- 加到嵌入：$\tilde x_t = x_t + p_t$，$p_t=PE_{t,:}$。

### 3.2 多尺度频率

令 $\omega_i=1/10000^{2i/d}$，则 $PE_{t,2i}=\sin(t\omega_i)$。波长 $\lambda_i=2\pi/\omega_i=2\pi\cdot10000^{2i/d}$：

- $i$ 小（前几对）：$\omega$ 大、波长短 → 高频，编码**细粒度**位置（相邻位置差别大）。
- $i$ 大（后几对）：$\omega$ 小、波长长 → 低频，编码**粗粒度**位置（远处位置的宏观尺度）。
- 一句话：不同维度承担不同分辨率的"位置尺"，类似多尺度小波。

### 3.3 相对位置的线性可解性

二维一对 $(PE_{t,2i},PE_{t,2i+1})=(\sin t\omega_i,\cos t\omega_i)$ 是单位圆上角度 $t\omega_i$ 的点。由三角恒等式：

$$
\begin{aligned}
PE_{t+k,2i} &= \sin\big((t+k)\omega_i\big)=\sin(t\omega_i)\cos(k\omega_i)+\cos(t\omega_i)\sin(k\omega_i)\\
PE_{t+k,2i+1} &= \cos\big((t+k)\omega_i\big)=\cos(t\omega_i)\cos(k\omega_i)-\sin(t\omega_i)\sin(k\omega_i)
\end{aligned}
$$

即 $\begin{pmatrix}PE_{t+k,2i}\\PE_{t+k,2i+1}\end{pmatrix}=\begin{pmatrix}\cos k\omega_i&\sin k\omega_i\\-\sin k\omega_i&\cos k\omega_i\end{pmatrix}\begin{pmatrix}PE_{t,2i}\\PE_{t,2i+1}\end{pmatrix}$。

**关键**：$p_{t+k}$ 是 $p_t$ 的线性变换（旋转 $k\omega_i$），且变换只依赖相对偏移 $k$。所以一个线性层（$W$）能从 $p_t$ 解出 $p_{t+k}$——attention 的 Q/K 投影是线性，理论上可学会这种投影，从而感知相对位置。这是作者选 sin/cos 而非任意固定向量的核心理由。

## 4. 数学原理 / 公式

### 4.1 频率与波长的边界

- 最小波长（$i=0$）：$\lambda_0=2\pi\approx6.28$ → 每隔几步就转一圈，区分相邻位置。
- 最大波长（$i=d/2-1$）：$\lambda_{\max}=2\pi\cdot10000^{2(d/2-1)/d}\approx2\pi\cdot10000\approx6.3\times10^4$ → 远超常见序列长度，作"全局位置基准"。
- 覆盖从"相邻"到"超长"全尺度，是基数 10000 的设计目的。

### 4.2 外推性的理论 vs 实测

- **理论**：公式对任意 $t$ 有定义 → 任意长位置都有编码 → 可外推。
- **实测**：长序列质量仍衰减。原因：attention 的 $W_Q,W_K$ 在训练长度内学到的"如何用 PE 解位置"模式，外推到超长位置时**关系不再成立**（线性投影近似失效）。
- 结论：sinusoidal 的外推是**有界**的，远不如 [[RoPE]] + 插值。这也是现代 LLM 弃用的主因之一。

### 4.3 与 learned PE 的对比

| 维度 | sinusoidal PE | learned PE |
|---|---|---|
| 生成 | 公式，无参数 | 查表，$L\times d$ 参数 |
| 外推 | 理论可，实测有限 | **不可**（超位置无向量） |
| 先验 | 多尺度频率 | 无 |
| 实测质量 | 与 learned 接近 | 略好或接近 |
| 现代 LLM | 已弃用 | 已弃用 |

## 5. 代码示例

```python
import torch
import math

def sinusoidal_pe(d, L):
    """原版 Transformer 的正弦位置编码。
    返回 (L, d) 张量，加到 token 嵌入上。"""
    pe = torch.zeros(L, d)
    pos = torch.arange(L).float().unsqueeze(1)           # (L,1)
    # 频率：1/10000^(2i/d)，i 取 0,2,4,... 对应奇偶对
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)                   # 偶数维 sin
    pe[:, 1::2] = torch.cos(pos * div)                   # 奇数维 cos
    return pe

d, L = 512, 2048
P = sinusoidal_pe(d, L)                                  # (2048,512)
print(P.shape, P.mean(), P.std())                        # 方差≈0.5（sin/cos）

# 用法：加到嵌入
x_emb = torch.randn(2, 16, d)
x_with_pe = x_emb + P[:16].unsqueeze(0)                  # (2,16,512)

# 验证相对位置的线性变换性质：p_{t+k} = R_k p_t
t, k, i = 5, 3, 2
p_t = P[t, 2*i:2*i+2]; p_tk = P[t+k, 2*i:2*i+2]
wk = k * math.exp(-math.log(10000.0)*2*i/d)              # k*omega_i
R = torch.tensor([[math.cos(wk), math.sin(wk)],
                  [-math.sin(wk), math.cos(wk)]])
assert torch.allclose(R @ p_t, p_tk, atol=1e-5)          # ✓ 相对位置线性可解
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[位置编码]]（本节总览，它属绝对/加法式）、[[Tensor]]、token embedding。
- **下游（应用）**: [[自注意力]]（PE 加在输入后进 attention）、原版 Transformer / BERT。
- **对比 / 易混**:
  - **vs learned PE**：见 4.3，公式 vs 查表，外推性差异。
  - **vs [[RoPE]]**：见 2 节注，加法绝对 vs 旋转相对，同源 sin/cos 但用法不同。
  - **vs ALiBi**：绝对加法式 vs 相对偏置式。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 sinusoidal 能完美外推**：理论可生成任意位置，但 attention 学到的位置利用模式在超长位置失效，实测衰减明显。外推要靠 [[RoPE]] + 插值。
> 2. **奇偶配对搞反**：必须 $(\sin,\cos)$ 成对出现在相邻两维（$2i,2i+1$），不能全 sin 或全 cos，否则相对位置的旋转性质不成立。
> 3. **频率乱设**：基数 10000、指数 $2i/d$ 是精心选的多尺度覆盖，改成任意值会破坏频率范围与外推。
> 4. **以为加在输入=作用在 QK**：sinusoidal 是加法式，叠加在 token 嵌入；[[RoPE]] 才作用在 Q/K。混用会错。
> 5. **缩放忽略**：$p_t$ 的量级约 $\sqrt{d/2}$（每维方差 0.5），加到嵌入上若嵌入方差是 1，量级匹配；但实际常需注意嵌入/PE 的相对幅度。
> 6. **以为现代 LLM 还用它**：Llama/Qwen 等全用 [[RoPE]]，sinusoidal 只在原版/BERT 时代。

## 8. 延伸细节

### 8.1 为何现代 LLM 弃用

- 外推差（4.2），长序列需求下劣势明显。
- 加法式耦合松，attention 要"学会"解位置，效率低。
- [[RoPE]] 把相对位置内建于内积，耦合紧、外推好，全面替代。

### 8.2 NTK-aware 缩放

若硬要用 sinusoidal/RoPE 的频率框架做长度外推，NTK-aware scaling 按"高频不动、低频缩"改造频率，使远距离仍可分辨。这是 [[RoPE]] 外推工程的核心思想，源头是 sinusoidal 的多尺度设计。

### 8.3 相对位置线性可解的实证

3.3 的线性变换 $R_k$ 是理论可解性，但 attention 是否真学到取决于训练。实验显示在训练长度内学得很好，外推则退化——印证"外推瓶颈在 attention 的学习，不在 PE 公式"。

### 8.4 与 RoPE 的同源性

[[RoPE]] 的频率 $\theta_i=10000^{-2i/d}$ 正是 sinusoidal 的 $\omega_i$，旋转角度 $t\theta_i$ 正是 sinusoidal 的相位 $t\omega_i$。RoPE 可视为"把 sinusoidal 的相位用作旋转角，作用在 Q/K 而非叠加在输入"。同源不同用。

---
相关: [[位置编码]]、[[RoPE]]、[[自注意力]]、[[多头注意力]]、[[decoder-only architecture]]
