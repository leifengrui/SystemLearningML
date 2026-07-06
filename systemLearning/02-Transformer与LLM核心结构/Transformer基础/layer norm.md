# layer norm

> **所属章节**: [[Transformer基础]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中级

## 1. 一句话定义

**层归一化（layer normalization，layer norm，层归一化，LN）** 对**单个样本沿特征维**做归一化（减均值除标准差，再仿射），不依赖 batch 大小；在 Transformer 里它和 [[残差连接]] 配合，按位置分 **Pre-LN（先归一再子层）** 与 **Post-LN（先子层再归一）** 两种，现代 LLM 几乎全用 **Pre-LN**，且常用其简化版 **RMSNorm**。

## 2. 为什么需要它（动机与背景）

深度网络每层输入分布会随训练漂移（internal covariate shift），导致训练不稳、需小 lr。归一化把每层输入拉回稳定范围，允许更大 lr、加速收敛。

为什么不用 BatchNorm（BN）：

- BN 沿 **batch 维**归一化，对 batch size 敏感（小 batch 统计不稳）。
- BN 在序列/变长上不友好（不同长度样本混在一起统计无意义）。
- RNN/Transformer 的 batch 常小或变长，BN 不适用。

**LayerNorm 沿特征维归一化**，每个样本独立算，与 batch 无关 → 对 batch size / 序列长度鲁棒，成为序列模型标准。

## 3. 核心概念详解

### 3.1 LayerNorm 公式

对单个样本的特征向量 $x\in\mathbb{R}^d$：

$$
\mu=\frac1d\sum_i x_i,\quad \sigma=\sqrt{\frac1d\sum_i(x_i-\mu)^2+\epsilon}
$$
$$
\text{LN}(x)=\gamma\odot\frac{x-\mu}{\sigma}+\beta
$$

- $\gamma,\beta\in\mathbb{R}^d$：可学习仿射（恢复表达力）。
- $\epsilon$：防除零（1e-5）。
- 沿**特征维**（最后一维）归一化，每个 token 独立算。

### 3.2 LN vs BN（核心对比）

| 维度 | BatchNorm | LayerNorm |
|---|---|---|
| 归一化方向 | **batch 维**（跨样本） | **特征维**（单样本内） |
| 依赖 batch | 强（小 batch 不稳） | 无 |
| 序列/变长 | 不友好 | 友好 |
| 训练/推理差异 | 有（推理用 running stat） | 无 |
| 典型场景 | CV（图像） | NLP/Transformer |

### 3.3 Pre-LN vs Post-LN

残差块里 LN 的位置：

**Post-LN**（原版 Transformer）：

$$
x' = \text{LN}\!\left(x + \text{SubLayer}(x)\right)
$$

**Pre-LN**（现代 LLM 主流）：

$$
x' = x + \text{SubLayer}\!\left(\text{LN}(x)\right)
$$

| 项 | Post-LN | Pre-LN |
|---|---|---|
| 主干 $x$ | 每层被 LN 作用 | **不进 LN，直接流过残差链** |
| 深网稳定性 | 差（深易不稳，需 warmup） | **好**（恒等 highway 纯） |
| 初始梯度 | 大、易爆炸 | 小、温和 |
| 现代 LLM | 几乎不用 | **标准** |

> [!tip] 为何 Pre-LN 更稳
> Pre-LN 主干 $x$ 不被 LN 作用，残差链是一条纯恒等通路，梯度/信息沿主干直达任意层，深网不漂移。Post-LN 每层 LN 扰动主干，深层易发散。现代 LLM（GPT/Llama）全用 Pre-LN。

### 3.4 RMSNorm（LLM 新宠）

RMSNorm 去掉均值和 $\beta$，只用 RMS 归一化 + 缩放：

$$
\text{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}
$$

- 省去减均值和 $\beta$，**计算更少**（无均值）。
- 实测质量与 LN 相当，Llama 系全用 RMSNorm。
- 直觉：Transformer 激活常近似零均值，去均值收益小，省掉划算。

### 3.5 仿射 $\gamma,\beta$ 的作用

归一化把分布拉到 $\mu=0,\sigma=1$，但有时模型需要别的分布。$\gamma,\beta$ 让模型学回任意均值/方差，恢复表达力。若设 $\gamma=1,\beta=0$ 则是纯归一化。RMSNorm 只留 $\gamma$。

## 4. 数学原理 / 公式

### LN 的不变性

LN 对 $x$ 的**整体缩放和平移**（$x\to ax+b$）输出不变（均值方差同步变），故对层间量级漂移鲁棒，能稳住前向数值。

### RMSNorm 的等价性

当 $x$ 近似零均值时，$\sqrt{\frac1d\sum x_i^2}\approx\sigma$，故 RMSNorm ≈ LN 去 $\beta$。省一次均值计算。

### Pre-LN 的恒等主干

Pre-LN 下 $x_{l+1}=x_l+\text{Sub}(\text{LN}(x_l))$，主干 $x_l$ 直接出现在 $x_{l+1}$，递推得 $x_L=x_0+\sum\text{Sub}$ → 纯加法残差链，恒等 highway。Post-LN 主干每层被 $\text{LN}$ 包裹，无此性质。

## 5. 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# LayerNorm（PyTorch 内置）
ln = nn.LayerNorm(512)
x = torch.randn(2, 16, 512)
print(ln(x).shape)                  # (2,16,512)，沿最后一维归一化

# RMSNorm（Llama 风格，手写）
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.gamma * x * rms

rms = RMSNorm(512)
print(rms(x).shape)

# Pre-LN block（现代 LLM 标准）
class PreLNBlock(nn.Module):
    def __init__(self, d, attn, ffn):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn, self.ffn = attn, ffn
    def forward(self, x):
        x = x + self.attn(self.n1(x))     # Pre-LN 残差
        x = x + self.ffn(self.n2(x))
        return x

# Post-LN block（原版 Transformer，对照）
class PostLNBlock(nn.Module):
    def __init__(self, d, attn, ffn):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.ffn = attn, ffn
    def forward(self, x):
        x = self.n1(x + self.attn(x))    # Post-LN：先加再归一
        x = self.n2(x + self.ffn(x))
        return x
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[残差连接]]（LN 在残差块内）、[[Tensor]]。
- **下游（应用）**: [[decoder-only architecture]]（每 block 用 RMSNorm）、训练稳定性（防数值漂移）、[[mixed precision training]]（归一化算子保 fp32）、[[过拟合与欠拟合]]（LN 间接稳训练）。
- **对比 / 易混**:
  - **LN vs BN**：见 3.2，方向与 batch 依赖不同。
  - **Pre-LN vs Post-LN**：见 3.3，稳定性差异。
  - **LN vs RMSNorm**：RMSNorm 省均值/β，Llama 用。
  - **LN vs GroupNorm/InstanceNorm**：CV 的变体，方向不同，NLP 不用。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **LN 维度搞错**：LN 归一化最后一维（特征），不是序列维或 batch 维。
> 2. **小 batch 用 BN**：Transformer batch 常小或变长，BN 统计不稳，必须 LN。
> 3. **Pre/Post-LN 混用**：两者稳定性差大，深 LLM 必须 Pre-LN，别照抄原版 Post-LN。
> 4. **RMSNorm 当 LN 用错 API**：RMSNorm 无 $\beta$、不减均值，PyTorch 内置 LN 有，别混。
> 5. **LN 的 $\gamma$ 初始化为 0**：应初始化为 1（否则输出恒 0、梯度全 0、训练死）。Llama 有时初始化较小（如 0.1~1）。
> 6. **推理时 LN 用 running stat**：LN 无 running stat（与 BN 不同），训练推理一致。
> 7. **低精度下 LN 精度丢**：归一化涉及均值方差累加，bf16 下易丢精度，autocast 把 LN 保 fp32（见 [[mixed precision training]]）。

## 8. 延伸细节

### 8.1 DeepNorm

为训极深 Transformer（上千层），DeepNorm 修改残差缩放 $x_{l+1}=x_l+\alpha\,\text{Sub}(x_l)$ 并调初始化，让深网稳。是 Post-LN 系的深网改进。

### 8.2 LN 与训练动力学

LN 让每层输入数值稳定，允许更大 lr + 更快收敛。但它也改变损失曲面，与 [[学习率调度]] warmup 协同防前期 spike。

### 8.3 RMSNorm 的流行

Llama 系全用 RMSNorm，因其计算少、质量相当。后续多数开源 LLM（Qwen/Mistral）跟进。已成为事实标准之一。

### 8.4 LN 的位置变体

除 Pre/Post，还有 Sandwich-LN（LN-Sub-LN，子层前后都归一）等变体，少数模型用。主流仍是 Pre-RMSNorm。

### 8.5 与 attention/FFN 的数值协同

attention 的 softmax、FFN 的激活（GELU/SwiGLU）对输入量级敏感，LN 把它们拉到稳定范围，避免 softmax 饱和或激活死区。这是 LN 在每个子层前的原因。

---
相关: [[Transformer基础]]、[[残差连接]]、[[FFN]]、[[多头注意力]]、[[mixed precision training]]、[[过拟合与欠拟合]]
