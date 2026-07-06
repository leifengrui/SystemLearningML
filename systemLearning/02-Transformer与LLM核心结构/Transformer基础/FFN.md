# FFN

> **所属章节**: [[Transformer基础]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中级（从零讲清"Transformer 的另一半"）

## 1. 一句话定义

**FFN（Feed-Forward Network，前馈网络 / MLP block）** 就是 Transformer 每一层里、紧跟在 [[多头注意力]] **之后**的一个**两层小神经网络（MLP）**：它对**每一个 token 单独、独立**地做一次"**升维 → 非线性激活 → 降维**"的加工。一句话定位——[[多头注意力]] 负责"token 之间互相交流信息"，**FFN 负责"每个 token 自己消化、加工、记忆"**，两者一外一内、缺一不可。FFN 占 Transformer 约 **2/3 的参数**，是模型"知识/记忆"的主要载体。

> [!note] 解答：已应"没看懂"的反馈重写本笔记
> 原版写得术语密集、缺直觉铺垫，初学者难懂。本次重写改进：(1) 加**白话类比**与 **ASCII 结构图**；(2) 从"Transformer 一个 block 长啥样"零起步讲动机；(3) 每个概念补"**为什么这么设计 / 直觉 / 数值例子**"；(4) 数学加**逐步推导**与 $d=512$ 的算参数例子；(5) 误区针对初学者常见困惑。核心一句话先记住：**FFN = 每个 token 独自过的两层 MLP，干"加工+记忆"的活，跟 attention 的"交流"互补**。读完 §2~§3 应能向别人讲清"为什么 Transformer 要有 FFN"。

> [!note] 一个类比，先建立直觉
> 把一个 Transformer 层想象成一节课：
> - **attention** = **课堂讨论**：学生们（token）互相听别人说什么、决定关注谁。这是"信息交流"，是**加权求和**（本质线性）。
> - **FFN** = **课后回工位自学**：每个学生**独自**把讨论来的信息消化、记进笔记本、加工成自己的理解。这是"加工与记忆"，需要**非线性**（人脑的思考不是简单加权）。
>
> 光讨论不消化 → 信息没沉淀；光自学不讨论 → 没有信息来源。两者必须交替。这就是 Transformer 一个 block = attention + FFN 的设计哲学。

## 2. 为什么需要它（动机与背景）

### 2.1 先看一个 Transformer block 长什么样

一个标准 Transformer block（layer）由**两个子层**组成，各包一层 [[残差连接]] 与 [[layer norm]]：

```
        输入 x (B, n, d)
          │
          ├──────────────────────┐
          ▼                      │
      [LayerNorm]                 │  残差
          │                      │
      [多头注意力]  ← token 间交互 │
          │                      │
          ▼                      │
          + ←────────────────────┘   x = x + Attn(LN(x))
          │
          ├──────────────────────┐
          ▼                      │
      [LayerNorm]                 │  残差
          │                      │
      [FFN]  ← 每个 token 独立加工 │
          │                      │
          ▼                      │
          + ←────────────────────┘   x = x + FFN(LN(x))
          │
          ▼
        输出 (B, n, d)
```

- **第一个子层是 attention**：让序列里 token 互相看，做信息路由；
- **第二个子层是 FFN**：对每个 token 单独做非线性变换，做信息加工；
- 两个子层都用 Pre-LN + 残差包起来（见 [[layer norm]]、[[残差连接]]）。

**FFN 就是上图第二个框**。少了它，Transformer 就只剩"线性加权路由"，没有非线性加工能力。

### 2.2 为什么光有 attention 不够（FFN 的三个不可替代作用）

1. **attention 本质是线性的（softmax 之外）**。attention 的输出是 $\sum_i \text{softmax}(q_i k_j^\top/\sqrt{d})\cdot v_i$，对 $v$ 是**加权求和**（线性组合）。一连串线性组合堆起来还是线性的 → 表达力弱，学不了复杂函数。FFN 的**非线性激活**（ReLU/GELU/SwiGLU）是 Transformer 获得非线性的**唯一来源**——没有 FFN，整个网络退化成线性变换嵌套，连 XOR 都学不了。

2. **模型需要"记忆/知识容量"**。研究表明（Geva et al.）FFN 的权重像 **key-value 记忆库**：第一层 $W_1$ 的行向量是"key"（匹配输入模式），第二层 $W_2$ 的列向量是"value"（存对应知识/事实）。attention 只是"路由"（决定把哪里的信息搬来），**真正存知识的是 FFN**。事实性知识（如"法国首都巴黎"）多可定位到 FFN 的某些神经元。

3. **FFN 是参数与计算的大头**。标准 FFN 因 4× 升维，参数约 $8d^2$，而 attention 约 $4d^2$ —— **FFN 占一个 block 约 2/3 参数**。所以"Transformer 有多少参数"很大程度由 FFN 决定；MoE、参数压缩等优化也主要打 FFN 的主意。

> [!tip] 一句话总结动机
> attention 做"信息交流"（线性、路由），FFN 做"信息加工与记忆"（非线性、容量）——FFN 给 Transformer 非线性与知识容量，是它不可替代的根本原因。

## 3. 核心概念详解

### 3.1 标准 FFN 结构（带数值例子）

公式：

$$
\text{FFN}(x)=W_2\,\sigma(W_1 x + b_1)+b_2
$$

各符号含义，配 $d=512$ 的数值例子一步步走：

| 符号 | 形状 | 作用 | $d=512$ 时的尺寸 |
|---|---|---|---|
| $x$ | $(d,)$ | 单个 token 的输入向量 | 512 维 |
| $W_1$ | $(d_{\text{ff}},\,d)$ | **升维**矩阵（$d\to d_{\text{ff}}$，常 $4d$） | $(2048,\,512)$ |
| $b_1$ | $(d_{\text{ff}},)$ | 升维后的偏置 | 2048 |
| $\sigma$ | - | **非线性激活**（ReLU/GELU/SwiGLU） | - |
| $W_2$ | $(d,\,d_{\text{ff}})$ | **降维**矩阵（$d_{\text{ff}}\to d$） | $(512,\,2048)$ |
| $b_2$ | $(d,)$ | 降维后的偏置 | 512 |

数据流（一个 token）：

```
x (512) --W1--> h (2048) --σ--> h' (2048) --W2--> y (512)
         升维        非线性         降维
```

- **升维**（$512\to2048$）：把 token 投影到更高维空间，给"更多抽屉"存不同模式；
- **非线性**：$\sigma$ 让网络能学非线性函数（这是关键，线性堆叠没用）；
- **降维**（$2048\to512$）：压回原维度，方便接残差与下一层。

> [!note] 为什么是"两层 MLP"不是更深
> FFN 只有两层（升维+降维），不像图像 CNN 的 MLP 那种好几层。原因：Transformer 的"深度"由**堆叠多个 block** 提供（每个 block 一个 FFN），单个 FFN 不需要很深。两层 + 巨宽（4× 升维）已足够容量，且宽比深更易并行、更稳。这是 Transformer 的设计选择。

### 3.2 为什么升维 4×

- **容量**：高维中间层像有更多"槽位"（2048 维 vs 512 维），可存更多模式 → 像 key-value 记忆库的 key 数量更多；
- **表达力**：升维 + 非线性 + 降维使 FFN 是**通用近似器**（MLP 理论），高维提升拟合能力；
- **经验默认**：原版 Transformer（Vaswani 2017）用 $d_{\text{ff}}=4d$，是经验甜点；Llama 系因用 SwiGLU（多一个投影）略调（如 Llama-7B 用 $11008\approx2.7d$）；
- **代价**：FFN 参数 $=2\cdot d\cdot d_{\text{ff}}\approx8d^2$，是 attention（$\approx4d^2$）的 **2 倍** → FFN 是参数大头，这也是为什么显存/算力优化重点在 FFN。

> [!note] $d=512$ 算参数量
> - $W_1$：$512\times2048=1,048,576$（约 100 万）
> - $W_2$：$2048\times512=1,048,576$（约 100 万）
> - FFN 总参数 ≈ 200 万（加偏置略多）。
> - 而同 $d$ 的 attention（QKV+O 四个投影）≈ $4\times512^2\approx100$ 万。
> - 故 FFN 是 attention 的 2 倍，占 block 的 2/3。一个 12 层、$d=512$ 的小模型，FFN 就有 $12\times200\text{万}=2400$ 万参数——大头在这。

### 3.3 位置独立（最关键、最易混的概念）

**FFN 对序列里每个 token 用同一组权重 $W_1,W_2$，token 之间完全不交互。**

具体说，输入 $x\in\mathbb{R}^{B\times n\times d}$（$B$ batch、$n$ 序列长、$d$ 维），FFN 对每个 $(B,n)$ 位置的 $d$ 维向量独立跑同一个两层 MLP：

```
token1 (d) ─┐
token2 (d) ─┤── 同一个 [W1, σ, W2] ──→ 各自输出 (d)
token3 (d) ─┘
```

- 三个 token 走**同一个** FFN，但**彼此看不到对方**；
- 这与 attention **相反**：attention 是"token 间加权求和"（看对方），FFN 是"token 各自加工"（不看对方）；
- 等价说法：FFN 是对最后一维 $d$ 的逐点操作，对序列维 $n$ 是"广播共享"。

> [!warning] 最常见误区
> 初学者常以为 FFN 也会让 token 交互。**不会**。FFN 是位置独立的，所有"信息交互"都在 attention 里完成。如果你在 FFN 里混进序列维操作（如对 $n$ 维卷积/注意力），就破坏了 Transformer 的分工结构。记住分工：**attention = 交流，FFN = 加工**。

### 3.4 激活函数演化（从 ReLU 到 SwiGLU）

| 激活 | 公式 | 特点 | 用例 |
|---|---|---|---|
| **ReLU** | $\max(0,x)$ | 简单、负区"死"（梯度 0） | 原版 Transformer（2017） |
| **GELU** | $x\cdot\Phi(x)$（$\Phi$ 标准正态 CDF） | 平滑可微、负区不彻底死 | GPT-2/BERT 时代 |
| **SwiGLU** | $\text{Swish}(xW_g)\odot(xW_u)$ | **带门控**，质量略好 | **Llama/Qwen/Mistral 等现代 LLM** |

通俗解释：
- **ReLU** 像硬开关：负数全砍掉。问题是负区神经元"死了"再也不更新，浪费容量；
- **GELU** 像软开关：负数不全砍，平滑过渡，训练更稳；
- **SwiGLU** 加了"门控"：让网络自己决定哪部分信息通过（不是无脑激活全部），表达力更强。

### 3.5 SwiGLU FFN（Llama 风格，现代 LLM 标配）

$$
\text{FFN}_{\text{SwiGLU}}(x)=\big(\text{Swish}(xW_{\text{gate}})\odot (xW_{\text{up}})\big)\,W_{\text{down}}
$$

- **三个投影**：gate、up、down（比标准 FFN 多一个）；
- $\text{Swish}(x)=x\cdot\sigma(\beta x)$（$\sigma$ 是 sigmoid，平滑可微）；
- $\odot$ 是逐元素相乘：gate 路算"开关"，up 路算"内容"，相乘 = 选择性通过 → 门控；
- **参数量调整**：标准 FFN 两投影 $2dd_{\text{ff}}$，SwiGLU 三投影 $3dd_{\text{ff}}$。为保持总参数不变，常把 $d_{\text{ff}}$ 从 $4d$ 降到 $\frac{8}{3}d\approx2.67d$。

> [!note] SwiGLU 为何流行
> 实验上 SwiGLU 比 GELU 质量略好（loss 更低），代价是多一个投影、实现稍复杂。Llama 系用它后，几乎所有现代开源 LLM 跟进。它是"用一点算力换一点质量"的工程选择，不是理论突破。

### 3.6 FFN 是"记忆库"（为什么说知识在 FFN 里）

研究（Geva et al. 2020）发现 FFN 可视为 **key-value 记忆**：

- $W_1$ 的每一行向量 = 一个 **key**（检测输入是否含某种模式）；
- $W_2$ 的对应列向量 = 一个 **value**（存该模式对应的知识/事实）；
- 输入 $x$ 与某些 key 匹配（内积大→激活）→ 取出对应 value → 加进输出。

实证：事实性知识（"法国首都巴黎""水沸点100度"）可定位到 FFN 的某些神经元，编辑这些神经元能改模型记忆（ROME 等知识编辑方法）。这就是"**Transformer 的知识存在 FFN 权重里**"说法的依据。attention 是"查表的路由"，FFN 是"被查的表"。

## 4. 数学原理 / 公式

### 4.1 标准 FFN

$$
\text{FFN}(x)=W_2\,\sigma(W_1 x + b_1)+b_2,\quad x\in\mathbb{R}^{d},\ d_{\text{ff}}=4d
$$

- $W_1\in\mathbb{R}^{d_{\text{ff}}\times d}$，$W_2\in\mathbb{R}^{d\times d_{\text{ff}}}$；
- 参数量 $=d\cdot d_{\text{ff}}+d_{\text{ff}}+d_{\text{ff}}\cdot d+d\approx2dd_{\text{ff}}$（偏置小）；
- $d=512,d_{\text{ff}}=2048$：$\approx2\times512\times2048\approx2.1\times10^6$（200 万）。

### 4.2 为什么升维+非线性+降维 = 通用近似器

两层 MLP（足够宽）能逼近任意连续函数（通用近似定理）。直觉：
- 升维把 $x$ 投到高维，使"不同模式落到不同坐标"（更可分）；
- 非线性把"线性不可分"变成"高维线性可分"（核技巧同构）；
- 降维把"高维可分表示"压回低维供下游用。

故 FFN 虽简单（两层），但够宽 + 非线性就能拟合复杂 token 变换，是 Transformer 非线性的来源。

### 4.3 位置独立 = 行级共享（形式化）

输入 $X\in\mathbb{R}^{n\times d}$（$n$ 个 token）。FFN 计算：

$$
H=\sigma(XW_1^\top+\mathbf{1}b_1^\top)\in\mathbb{R}^{n\times d_{\text{ff}}},\quad Y=HW_2^\top+\mathbf{1}b_2^\top\in\mathbb{R}^{n\times d}
$$

- 这是对 $X$ 的**逐行**操作：$Y_{i,:}=\text{FFN}(X_{i,:})$，第 $i$ 行（token）的输出**只依赖**第 $i$ 行输入；
- $W_1,W_2$ 在所有行（token）间**共享**；
- 故 FFN = "对每个 token 跑同一个 1D MLP"，无 token 间交互。交互由 attention 的 $\sum_i\text{softmax}(qk^\top)v_i$（跨 $i$ 求和）完成。

### 4.4 SwiGLU 参数量与 $d_{\text{ff}}$ 调整

标准 FFN 参数 $\approx2dd_{\text{ff}}^{std}$，SwiGLU 参数 $\approx3dd_{\text{ff}}^{swi}$。令二者相等：

$$
2d\cdot d_{\text{ff}}^{std}=3d\cdot d_{\text{ff}}^{swi}\ \Rightarrow\ d_{\text{ff}}^{swi}=\frac{2}{3}d_{\text{ff}}^{std}=\frac{2}{3}\cdot4d=\frac{8}{3}d
$$

故 SwiGLU 用 $d_{\text{ff}}=\frac{8}{3}d$ 保持总参数与标准 FFN 一致。这是 Llama 把 $d_{\text{ff}}$ 从 $4d$ 降到约 $2.67d$ 的数学原因。

### 4.5 FLOPs（每 token）

FFN 前向 FLOPs $\approx2\cdot(2d\cdot d_{\text{ff}})=4d\cdot d_{\text{ff}}$（两次矩阵乘，每次 $d\cdot d_{\text{ff}}$ 乘加，乘加算 2 FLOPs）。$d=512,d_{\text{ff}}=2048$：$\approx4\times512\times2048\approx4.2\times10^6$ FLOPs/token。是 attention（$\approx4d^2$ 量级）的数倍（因 $d_{\text{ff}}=4d$）。

## 5. 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= 1. 标准 FFN（GELU，GPT-2/BERT 风格）=========
class FFN(nn.Module):
    def __init__(self, d, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4*d                  # 默认 4× 升维
        self.fc1 = nn.Linear(d, d_ff)       # 升维: d → d_ff
        self.fc2 = nn.Linear(d_ff, d)       # 降维: d_ff → d
    def forward(self, x):                   # x: (B, n, d)
        h = self.fc1(x)                     # (B, n, d_ff) 升维
        h = F.gelu(h)                       # 非线性(关键!)
        return self.fc2(h)                  # (B, n, d) 降回原维

# ========= 2. SwiGLU FFN（Llama 风格，现代 LLM 标配）=========
class SwiGLUFFN(nn.Module):
    def __init__(self, d, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(8*d/3)           # 8/3·d 保持参数量与标准 FFN 相当
        self.w_gate = nn.Linear(d, d_ff, bias=False)  # gate 路(开关)
        self.w_up   = nn.Linear(d, d_ff, bias=False)  # up 路(内容)
        self.w_down = nn.Linear(d_ff, d, bias=False)  # 降维
    def forward(self, x):
        g = self.w_gate(x)                  # gate 投影
        u = self.w_up(x)                    # up 投影
        h = F.silu(g) * u                   # silu=Swish(β=1); 门控:逐元素相乘
        return self.w_down(h)               # 降回 d

# 验证形状与参数量
d = 512
ffn_std = FFN(d)                            # 标准 FFN
ffn_sw  = SwiGLUFFN(d)                      # SwiGLU FFN
x = torch.randn(2, 16, d)                   # (batch=2, seq=16, d=512)
print("标准 FFN 输出:", ffn_std(x).shape)   # (2, 16, 512)
print("SwiGLU 输出:", ffn_sw(x).shape)      # (2, 16, 512)
n_std = sum(p.numel() for p in ffn_std.parameters())
n_sw  = sum(p.numel() for p in ffn_sw.parameters())
print(f"标准 FFN 参数: {n_std:,}  SwiGLU 参数: {n_sw:,}")  # 两者接近

# ========= 3. 完整 Transformer block = Pre-LN 残差(Attn) + Pre-LN 残差(FFN) =========
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        Norm = nn.RMSNorm if hasattr(nn, 'RMSNorm') else nn.LayerNorm  # 新版 torch 才有 RMSNorm
        self.n1 = Norm(d)
        self.n2 = Norm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn = SwiGLUFFN(d)
    def forward(self, x):
        # 子层1: attention(信息交流) + 残差
        x = x + self.attn(self.n1(x), self.n1(x), self.n1(x), need_weights=False)[0]
        # 子层2: FFN(独立加工) + 残差
        x = x + self.ffn(self.n2(x))
        return x

blk = Block(d=512, h=8)
print("block 输出:", blk(x).shape)          # (2, 16, 512)
```

> [!tip] 关键观察
> 代码里 FFN 的 `forward` 对 `x: (B, n, d)` 只操作最后一维 $d$（`nn.Linear` 默认对最后一维），$n$ 维原样保留——这就是"位置独立"的代码体现。如果加任何对 $n$ 维的操作（如卷积、attention），就不再是标准 FFN。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[多头注意力]]（FFN 紧跟 attention 之后）、[[残差连接]]（FFN 包在残差里：$x+\text{FFN}(\text{LN}(x))$）、[[layer norm]]（Pre-LN 在 FFN 前）、MLP/通用近似定理。
- **下游（应用）**: [[decoder-only architecture]]（每层一个 FFN）、MoE（FFN 的稀疏化，多专家）、参数/显存大头（见 [[activation memory]]）、知识编辑（FFN 存知识）。
- **对比 / 易混**:
  - **FFN vs attention**：FFN 位置独立（加工/记忆），attention 位置交互（路由）。这是 Transformer 分工的核心，详见 §3.3。
  - **FFN vs MLP**：FFN 是两层 MLP 的特例，Transformer 语境下"FFN"与"MLP block"常混用，都指这个两层结构。
  - **标准 FFN vs SwiGLU FFN**：前者两投影+ReLU/GELU，后者三投影+门控，质量略好但稍复杂，现代 LLM 用 SwiGLU。
  - **FFN vs MoE**：MoE 把单个大 FFN 换成多个小 FFN（专家）+ 路由，每 token 只激活部分，参数总量大但算力省。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 attention 是参数大头**：实际 **FFN 占约 2/3 参数**（4× 升维使它比 attention 大），attention 只约 1/3。看模型参数量要先看 FFN。
> 2. **以为 FFN 也让 token 交互**：**不会**。FFN 位置独立，token 间交互全在 attention。在 FFN 里混序列维操作会破坏 Transformer 分工。这是初学者头号误区（§3.3）。
> 3. **SwiGLU 的 $d_{\text{ff}}$ 还用 $4d$**：会多 1/3 参数（三投影 vs 两投影），应降到 $\frac{8}{3}d$ 保持总量（§4.4）。
> 4. **激活用纯 ReLU**：负区"死神经元"浪费容量，现代 LLM 用 GELU/SwiGLU 平滑可微更稳。
> 5. **FFN 不加残差**：FFN 应包在残差里（$x+\text{FFN}(\text{LN}(x))$），裸 FFN 在深网里梯度衰减、难训（见 [[残差连接]]）。
> 6. **升维比例乱定**：$d_{\text{ff}}$ 影响参数/显存/质量，$4d$ 是经验默认，改要重算参数量与显存。
> 7. **以为 FFN 没用、attention 就够了**：缺 FFN 则无非线性，网络退化成线性变换嵌套，表达力崩溃（§2.2）。
> 8. **混淆 FFN 的"两层"与 Transformer 的"多层"**：FFN 自身只两层（升维+降维），Transformer 的深度由堆叠多个 block 提供，不是单个 FFN 深。
> 9. **门控（SwiGLU）与 attention 的 softmax 混淆**：SwiGLU 的门控是逐元素相乘（$\odot$），attention 的 softmax 是跨 token 归一化加权——机制不同，别混。

## 8. 延伸细节

### 8.1 MoE（Mixture of Experts）：FFN 的稀疏化

把单个大 FFN 换成 $N$ 个小 FFN（"专家"），每个 token 经一个门控网络（router）路由到 top-k（常 2）个专家，只算被选中的。效果：
- **参数总量**大（$N$ 个专家），但**每 token 激活参数**稀疏（只 top-k）；
- 等价于"扩参数而不等比涨算力"，是 LLM 扩规模的主流（Mixtral 8×7B、DeepSeek-MoE）；
- 门控路由是 MoE 的关键与难点（负载均衡、路由坍缩）。

### 8.2 FFN 是知识存储（可编辑）

事实性知识（"法国首都巴黎"）多存在 FFN 某些神经元/行向量，可定位与编辑：
- ROME/MEMIT 等方法直接改 FFN 权重，能修改模型记忆的事实；
- 这支撑"FFN = 记忆库"观点，也使知识编辑、模型可解释性研究聚焦 FFN。

### 8.3 FFN 的显存与算力

- **激活显存** $\propto B\times n\times d_{\text{ff}}$：4× 升维使 FFN 中间激活是 attention 的数倍，是 [[gradient-checkpointing]] 优化的重点（重算省显存）；
- **算力**：FFN 前向 FLOPs 是 attention 的数倍（$d_{\text{ff}}=4d$），故推理时 FFN 也是算力大头——这也是为什么量化/MoE 主要优化 FFN。

### 8.4 激活函数演化趋势

ReLU（2017）→ GELU（2018，GPT-2/BERT）→ SwiGLU（2020，Llama 系）→ 现代几乎全用 SwiGLU。核心演化方向："**平滑可微 + 门控**"，带来稳定与质量的边际提升。未来可能的新激活（如更高阶门控）仍在探索。

### 8.5 FFN 与 attention 分工的实证

消融实验：
- **移除 attention**（只留 FFN）：退化为"无交互 MLP"，token 间看不到对方，序列建模能力崩溃；
- **移除 FFN**（只留 attention）：退化为"纯线性路由"，无非线性，表达力崩溃。
两者都不可少。FFN 主要管"**是什么**"（内容加工/记忆），attention 主要管"**哪里来**"（依赖关系/路由）。

### 8.6 FFN 的宽度 vs 深度权衡

Transformer 选择"宽 FFN + 多层 block"而非"深 FFN + 少层 block"：
- 宽 FFN（4× 升维）容量足、易并行（矩阵乘 GPU 友好）；
- 多层 block 提供深度（每层一个 FFN + 一个 attention）；
- 深 FFN（单 block 内多层 MLP）会破坏"attention-FFN 交替"的分工，且难训。故 FFN 始终两层，深度靠堆 block。

---
相关: [[Transformer基础]]、[[多头注意力]]、[[残差连接]]、[[layer norm]]、[[decoder-only architecture]]、[[自注意力]]
