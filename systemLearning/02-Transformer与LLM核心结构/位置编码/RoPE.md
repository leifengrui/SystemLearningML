# RoPE

> **所属章节**: [[位置编码]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中高级

## 1. 一句话定义

**旋转位置编码（Rotary Position Embedding，RoPE，旋转位置编码）** 是一种**相对位置编码**：把每个位置 $t$ 的 Q/K 向量按位置旋转角度 $t\theta$（多频率分头旋转），利用旋转矩阵性质 $R_t^T R_j=R_{j-t}$ 让 $\langle q_t,k_j\rangle$ **只依赖相对距离 $j-t$**；它直接作用在 [[多头注意力]] 每头的 Q/K 上（V 不动），不增参数、外推可控，是 Llama/Qwen/Mistral 等现代 LLM 的事实标准位置编码。

## 2. 为什么需要它（动机与背景）

[[位置编码]] 总览已述 attention 无序。前代方案的痛点：

- **绝对 PE（[[sinusoidal PE]]/learned）**：加在输入，位置信息要靠 attention 自己"解"，耦合松散；learned 不能外推，sinusoidal 实测外推有限。
- **T5 相对偏置**：要额外存 $O(n^2)$ 偏置表，且与 attention 解耦度仍可改进。

RoPE 的设计目标（苏剑林 2021）：

> **能不能找到一种 PE，使 $q_t,k_j$ 的内积 $\langle f(q_t,t),f(k_j,j)\rangle$ 只依赖相对位置 $j-t$，且 $f$ 无参数？**

用**旋转**达成：$f(q_t,t)=R_t q_t$，则 $\langle R_t q_t,R_j k_j\rangle=q_t^T R_t^T R_j k_j=q_t^T R_{j-t}k_j$，天然相对。且 $R_t$ 是固定旋转矩阵（无参数）、作用在 Q/K（耦合紧）、可外推（相对位置范围有限 + 频率可缩放）。这套性质让它全面取代绝对 PE。

## 3. 核心概念详解

### 3.1 二维旋转的核心思想

先看 $d=2$ 的情形。设 $q=(q_0,q_1)$，位置 $t$ 的旋转矩阵

$$
R_t=\begin{pmatrix}\cos t\theta&-\sin t\theta\\\sin t\theta&\cos t\theta\end{pmatrix}
$$

则 $q_t'=R_t q$ 把 $q$ 在二维平面旋转角度 $t\theta$。两位置内积：

$$
\langle q_t',q_j'\rangle=\langle R_t q,R_j q\rangle=q^T R_t^T R_j q=q^T R_{j-t}q
$$

因 $R_t^T R_j=R_{j-t}$（旋转合成），内积只含 $j-t$。**这就是相对位置的来源**。

### 3.2 推广到高维：分头旋转

$d$ 维向量分成 $d/2$ 个二维子空间，每对 $(q_{2i},q_{2i+1})$ 用不同频率 $\theta_i$ 独立旋转：

$$
R_t=\text{diag}\!\left(R_t^{(0)},R_t^{(1)},\dots,R_t^{(d/2-1)}\right),\quad
R_t^{(i)}=\begin{pmatrix}\cos t\theta_i&-\sin t\theta_i\\\sin t\theta_i&\cos t\theta_i\end{pmatrix}
$$

频率 $\theta_i=10000^{-2i/d}$（继承自 [[sinusoidal PE]] 的多尺度设计）。每对二维独立旋转、各自一个频率，整个 $d$ 维由分块对角旋转矩阵作用。

### 3.3 复数视角（更简洁）

把每对 $(q_{2i},q_{2i+1})$ 看成复数 $q_i=q_{2i}+i\,q_{2i+1}$，旋转就是乘 $e^{it\theta_i}$：

$$
q_{t,i}=q_i\cdot e^{it\theta_i},\quad k_{j,i}=k_i\cdot e^{ij\theta_i}
$$

内积（复数实部）：

$$
\text{Re}(q_{t,i}\overline{k_{j,i}})=\text{Re}\!\left(q_i\overline{k_i}\,e^{i(t-j)\theta_i}\right)
$$

**只依赖 $t-j$**。复数视角是 RoPE 推导与实现的标准写法。

### 3.4 作用位置：Q/K，不是 V、不是输入

- 作用在 [[多头注意力]] **每头的 Q 和 K** 上（旋转后做点积）。
- **V 不旋转**（V 是被加权的"内容"，位置只通过 Q/K 的相对关系体现）。
- **不作用在输入嵌入**——这是与 [[sinusoidal PE]]（加在输入）的根本区别。
- 每头独立旋转，head_dim 通常是 $d/h$，旋转在该 head_dim 内进行。

### 3.5 频率与多尺度

$\theta_i=10000^{-2i/d}$，$i=0,\dots,d/2-1$：

- $i$ 小：$\theta$ 大、转得快 → 细粒度（相邻位置角度差大）。
- $i$ 大：$\theta$ 小、转得慢 → 粗粒度（远处位置才显差异）。
- 与 sinusoidal 同源的多尺度覆盖，但用作旋转角而非叠加值。

## 4. 数学原理 / 公式

### 4.1 内积的相对性（核心证明）

设 $R_t$ 为分块对角旋转矩阵。位置 $t$ 的 Q、位置 $j$ 的 K 旋转后内积：

$$
\langle R_t q,\,R_j k\rangle = q^T R_t^T R_j k
$$

对每个二维块 $i$，$R_t^{(i)T}R_j^{(i)}=R_{j-t}^{(i)}$（旋转群性质：先逆旋 $t$ 再正旋 $j$ = 净旋 $j-t$）。故：

$$
q^T R_t^T R_j k = q^T R_{j-t} k
$$

**仅依赖 $j-t$**。证毕。RoPE 因此是"相对位置编码"，尽管实现上每位置独立旋转。

### 4.2 旋转矩阵的无参数性

$R_t$ 由 $\cos t\theta_i,\sin t\theta_i$ 组成，$\theta_i$ 是预设频率，**无可学参数**。RoPE 不增加模型参数，只改变 Q/K 的几何变换。

### 4.3 与 sinusoidal 的同源

sinusoidal PE 的相位是 $t\omega_i=t\cdot10000^{-2i/d}=t\theta_i$，正是 RoPE 的旋转角。RoPE = 把 sinusoidal 的相位当旋转角、作用在 Q/K 而非叠加在输入。同源不同用。

### 4.4 外推的数学基础

相对位置 $j-t$ 的取值在 $[-L,L]$ 内，远小于绝对位置的可能范围。且因旋转以 $2\pi$ 为周期，远距离角度会"绕回"——这是 RoPE 远距离衰减的原因，也是外推需要缩放频率的依据（见 8.1）。

## 5. 代码示例

```python
import torch
import torch.nn.functional as F
import math

def build_freqs(d, base=10000.0):
    """构造 d/2 个频率 theta_i = 10000^(-2i/d)。"""
    i = torch.arange(0, d, 2).float()
    return 1.0 / (base ** (i / d))            # (d/2,)

def apply_rope(q, k, freqs):
    """对 Q/K 应用 RoPE。q,k: (B, h, n, d_h)，freqs: (n, d_h/2)。
    V 不动。返回旋转后的 q,k。"""
    n = q.size(-2)
    # 构造 t*theta_i：位置 t 乘频率
    t = torch.arange(n, device=q.device).float()
    angles = torch.outer(t, freqs)             # (n, d_h/2)
    cos = angles.cos(), angles.sin()
    # 把 q 按二维配对：(B,h,n,d_h/2,2) 旋转
    def rotate(x, cos, sin):
        x1, x2 = x[..., 0::2], x[..., 1::2]    # 偶/奇维
        # R=[[c,-s],[s,c]] 作用 (x1,x2)：新x1=c*x1-s*x2, 新x2=s*x1+c*x2
        return torch.stack([cos*x1 - sin*x2, sin*x1 + cos*x2], dim=-1).flatten(-2)
    cos_b = cos[0].unsqueeze(0).unsqueeze(0)  # (1,1,n,d_h/2) 广播
    sin_b = cos[1].unsqueeze(0).unsqueeze(0)
    return rotate(q, cos_b, sin_b), rotate(k, cos_b, sin_b)

# 用法
B, h, n, d_h = 2, 8, 16, 64
q = torch.randn(B, h, n, d_h)
k = torch.randn(B, h, n, d_h)
freqs = build_freqs(d_h)                        # (d_h/2,)
q2, k2 = apply_rope(q, k, freqs)
print(q2.shape)                                 # (2,8,16,64)

# 验证相对性：q_t·k_j 应只依赖 j-t
def dot_ij(q, k, i, j):
    return (q[0,0,i] * k[0,0,j]).sum().item()
# 旋转后
q2r, k2r = apply_rope(q, k, freqs)
# 对比 (i,j)=(2,5) 与 (i,j)=(7,10)，相对差都是 3，应相近
print(dot_ij(q2r, k2r, 2, 5), dot_ij(q2r, k2r, 7, 10))
```

> [!tip] 工程实现
> 实际框架（HuggingFace/llama.cpp）常用复数乘法或 `rotate_half` 算子高效实现，但逻辑等价于上面的二维配对旋转。关键是"偶奇配对、按频率旋转、作用 Q/K"。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[位置编码]]（本节总览，它属相对/旋转式）、[[多头注意力]]（作用在每头 Q/K）、[[自注意力]]、[[sinusoidal PE]]（频率同源）。
- **下游（应用）**: [[decoder-only architecture]]（每层用 RoPE）、长度外推工程（PI/YaRN/NTK）、长上下文 LLM。
- **对比 / 易混**:
  - **vs [[sinusoidal PE]]**：见 4.3，旋转作用 Q/K（相对） vs 加法叠加输入（绝对）。
  - **vs ALiBi**：旋转式 vs 偏置式，都相对；RoPE 用多频率旋转，ALiBi 用线性距离偏置。
  - **vs learned PE**：无参数 vs 有参数，相对 vs 绝对，外推性差异大。
  - **RoPE vs 绝对旋转**：RoPE 的"绝对旋转"实现给出"相对内积"结果，别被"每位置独立旋转"误导以为是绝对编码。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 RoPE 加在输入嵌入**：RoPE 作用在**每头 Q/K**，输入和 V 都不动。这是与 sinusoidal 最易混的点。
> 2. **以为 V 也旋转**：V 不旋转。位置只通过 Q/K 的相对关系进入 attention，V 是被加权的内容。
> 3. **以为 RoPE 是绝对位置编码**：实现上每位置独立旋转（看似绝对），但内积数学上**只依赖相对位置**，归类为相对编码。
> 4. **旋转方向/符号搞反**：$R_t=[[c,-s],[s,c]]$ 是标准逆时针旋转；配对时偶维=x1、奇维=x2，符号弄反会破坏相对性。照复数乘法 $e^{it\theta}$ 推最稳。
> 5. **以为频率随便定**：$\theta_i=10000^{-2i/d}$ 是多尺度设计，乱改会丢外推与分辨率。外推改造（见 8.1）也是分层改频率，不是乱改。
> 6. **忽视外推时的频率绕回**：远距离 $t\theta_i$ 超过 $2\pi$ 会绕回，导致远位置区分度下降——这是长上下文要缩放频率的原因。
> 7. **配对维度错位**：必须 $(q_{2i},q_{2i+1})$ 相邻配对，不能跨维乱配，否则每对的二维旋转不成立。

## 8. 延伸细节

### 8.1 长度外推工程（RoPE 的杀手锏）

训练 2k/4k、推理 32k+ 的需求催生一批改造 RoPE 频率的方法：

- **Position Interpolation（PI）**：把位置 $t$ 压缩为 $t\cdot L_{\text{train}}/L_{\text{test}}$，等价缩放所有频率。简单但损失高频细节。
- **NTK-aware scaling**：分层缩放——**高频不动、低频缩**，保留近距离分辨率、拉伸远距离尺度，质量优于 PI。
- **YaRN**：混合策略，对不同频率段用不同插值/外推，是当前长上下文 LLM 主流之一。
- 核心思想：RoPE 的频率是可调旋钮，外推问题 = 频率缩放问题。

### 8.2 为什么现代 LLM 选 RoPE

- 相对位置（内积天然相对）→ 符合语言"距离更重要"的先验。
- 无参数 → 不增显存/参数。
- 作用 Q/K 耦合紧 → attention 直接利用位置，不靠学习解。
- 外推可控 → 配合 PI/YaRN 可扩到 32k/128k/1M。
- 实测质量稳定，Llama 系带动生态全面采用。

### 8.3 远距离衰减

RoPE 的内积含 $\cos((j-t)\theta_i)$，随距离增大某些高频分量振荡、低频分量缓变，整体呈**远距离注意力衰减**。这与自然语言"局部依赖强、远依赖弱"吻合，是 RoPE 的隐含归纳偏置。

### 8.4 与 ALiBi 的对比

| 维度 | RoPE | ALiBi |
|---|---|---|
| 形式 | 旋转 Q/K | attention logits 加线性距离偏置 |
| 参数 | 无 | 无（斜率按头预设） |
| 外推 | 需缩放频率（PI/YaRN） | **天然外推**（线性偏置无周期） |
| 长序列 | 配合工程可达极长 | 原生好，但远距离衰减强 |
| 主流 | Llama/Qwen/Mistral | 部分（BLOOM/长序列特化） |

### 8.5 GPT-NeoX/LLaMA 的实现细节

不同框架的 RoPE 实现略有差异（旋转方向、频率排列），但数学等价。加载权重时要注意 `rope_theta`（base，Llama-2 为 10000，Llama-3 为 500000）和缩放配置，否则位置语义错乱。

---
相关: [[位置编码]]、[[sinusoidal PE]]、[[多头注意力]]、[[自注意力]]、[[decoder-only architecture]]、[[KV cache]]
