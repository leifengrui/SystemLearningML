# gradient explosion and vanishing

> **所属章节**: [[数值稳定]]
> **所属模块**: [[12-训练稳定性工程]]
> **别名**: gradient explosion / vanishing / 梯度爆炸 / 梯度消失 / vanishing/exploding gradients
> **难度**: 中（需懂反向传播 + 链式法则 + 激活函数导数）


## 1. 一句话定义

**gradient explosion（梯度爆炸）** 是反向传播中梯度数值**指数放大至 inf/NaN**（连乘 $>1$），**gradient vanishing（梯度消失）** 是梯度**指数缩小至 0**（连乘 $<1$），两者都让训练失败（权重不更新或 NaN）。根因是**链式法则的连乘**：$\frac{\partial L}{\partial x} = \prod_i W_i \cdot \sigma'(\cdot)$，乘积 $>1$ 爆炸、$<1$ 消失。深层网络（层数多）、RNN（时间步连乘）、不当初始化（$W$ 过大/小）、不当激活函数（sigmoid 导数 $<1$）是主因。对策：残差连接（+1 让连乘不缩）、[[normalization]]（控尺度）、Xavier/He 初始化（保方差）、ReLU/GELU 激活（导数不过小）、gradient clipping（爆炸硬限）、[[warmup]]（前期稳）。LLM 训练常见 loss spike（爆炸的瞬时表现）。

> [!note] 三句话定位
> - **是什么**：连乘 $>1$ 爆炸至 inf，$<1$ 消失至 0，训练失败。
> - **为什么**：链式法则连乘，深层/RNN 连乘次数多，指数放大/缩小。
> - **怎么解决**：residual（+1 不缩）、normalization（控尺度）、He 初始化（保方差）、ReLU（导数不过小）、gradient clip（硬限爆炸）、warmup。


## 2. 为什么需要它（动机与背景）

### 2.1 链式法则的连乘

反向传播用链式法则：$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_L} \cdot \prod_{i=1}^{L} \frac{\partial x_i}{\partial x_{i-1}} = \prod_{i=1}^{L} W_i \cdot \sigma'(z_i)$。每层贡献一个 $W_i$（权重）和一个 $\sigma'$（激活导数）。若每层的 $|W_i \cdot \sigma'| > 1$，连乘指数增长（爆炸）；$<1$ 指数衰减（消失）。$L$ 层时连乘 $L$ 次，指数效应。

### 2.2 深层网络的指数效应

浅层（$L$ 小）连乘次数少，效应不显。深层（$L=100$）连乘 100 次，$1.1^{100} \approx 1.4 \times 10^4$（爆炸），$0.9^{100} \approx 2.7 \times 10^{-5}$（消失）。LLM（32-100 层）和 RNN（时间步 50-100）是重灾区。

### 2.3 RNN 的消失（经典场景）

RNN 在时间步上共享权重 $W$：$h_t = \sigma(W h_{t-1})$。反向时 $\frac{\partial L}{\partial h_0} = \prod_{t=1}^{T} W \cdot \sigma'$，连乘 $T$ 次。$T$ 大（长序列）+ $|W\sigma'|<1$ → 消失，早期时间步学不到。这是 RNN 难学长序列的根因，催生 LSTM（门控缓解）和 Transformer（attention 跨步直连）。

### 2.4 LLM 训练的 loss spike

LLM 训练（大模型 + 大数据 + 大 lr）常出现 **loss spike**：loss 突然飙升后恢复或 NaN。根因常是梯度爆炸（某层梯度突大，连乘放大，权重更新炸，后续 NaN）。对策：gradient clipping（硬限）、lower lr、warmup、bf16（比 fp16 不易溢出）。详见 [[loss spike handling]]。

### 2.5 对策概览

| 问题 | 对策 | 机制 |
|---|---|---|
| 消失 | residual 连接 | $x_{l+1} = x_l + F(x_l)$，导数含 $+1$，连乘不缩 |
| 消失 | ReLU/GELU 激活 | 导数 $\geq 1$（ReLU 正区），不过小 |
| 消失 | He 初始化 | $W$ 方差 $\sim 2/n$，保前向方差 |
| 爆炸 | Xavier 初始化 | $W$ 方差 $\sim 1/n$，控尺度 |
| 爆炸 | gradient clipping | $\|g\| \to \min(\|g\|, c)$，硬限 |
| 两者 | [[normalization]] | 控每层激活尺度，防漂移 |
| 两者 | [[warmup]] | 前期小 lr，防早期不稳 |


## 3. 核心概念详解

### 3.1 连乘的数学

$$
\frac{\partial L}{\partial x_0} = \prod_{i=1}^{L} \underbrace{W_i}_{\text{权重}} \cdot \underbrace{\sigma'(z_i)}_{\text{激活导数}}
$$

若 $|W_i \sigma'|$ 均值 $\mu$：

- $\mu > 1$：$\mu^L \to \infty$（爆炸）。
- $\mu < 1$：$\mu^L \to 0$（消失）。
- $\mu = 1$：稳定（理想）。

### 3.2 激活函数的导数

| 激活 | 导数 | 问题 |
|---|---|---|
| sigmoid | $\sigma(1-\sigma) \in (0, 0.25]$ | **$<1$，消失** |
| tanh | $1-\tanh^2 \in (0, 1)$ | $<1$，消失较轻 |
| ReLU | $1$（正区）或 $0$（负区） | 正区 $=1$ 不缩，但负区死神经元 |
| GELU | $\sim 1$（正区） | 近似 ReLU，平滑 |
| SwiGLU | LLM 常用 | 含门控，导数复杂 |

sigmoid 的导数最大 0.25，连乘快速消失——是早期深层网络难训的主因。ReLU/GELU 正区导数 1，不缩，故深层可训。

### 3.3 初始化的作用

权重 $W$ 的方差决定 $|W\sigma'|$ 的期望：

- **Xavier（Glorot）**：$\text{Var}(W) = 1/n$，保前向/反向方差稳定。适合 tanh/sigmoid。
- **He（Kaiming）**：$\text{Var}(W) = 2/n$，补偿 ReLU 负区零梯度。适合 ReLU。

初始化让 $|W\sigma'| \approx 1$，连乘不爆不消。不当初始化（$W$ 过大→爆炸，过小→消失）是训练失败的常见原因。

### 3.4 residual 连接

$$
x_{l+1} = x_l + F(x_l), \quad \frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F}{\partial x_l}
$$

导数含 $+I$（单位矩阵），即使 $\partial F/\partial x_l$ 小，总导数 $\geq 1$，连乘不缩。这是 ResNet 让 100+ 层可训的关键。Transformer 每层有 residual（attention + residual, MLP + residual），故深可训。

### 3.5 normalization 的作用

[[normalization]]（LayerNorm/BatchNorm）控每层激活尺度到 $\sim N(0,1)$，防激活漂移到大/小。间接稳梯度（激活不过大/小，导数不过大/小）。Transformer 用 LayerNorm（per-token），LLM 用 RMSNorm（LayerNorm 简化）。

### 3.6 gradient clipping

爆炸的硬限：

$$
g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|}\right)
$$

$\|g\| > c$ 时缩到 $c$。防梯度爆炸传到权重更新。LLM 训练标配（$c \sim 1.0$）。详见 [[clipping strategies]]。


## 4. 数学原理 / 公式

### 4.1 连乘

$$
\frac{\partial L}{\partial x_0} = \prod_{i=1}^{L} W_i \sigma'(z_i)
$$

### 4.2 指数增长/衰减

$$
\left|\frac{\partial L}{\partial x_0}\right| \approx \mu^L, \quad \mu = \mathbb{E}[|W\sigma'|]
$$

- $\mu > 1$：爆炸，$\mu^L \to \infty$。
- $\mu < 1$：消失，$\mu^L \to 0$。

### 4.3 RNN 的消失

$$
\frac{\partial L_T}{\partial h_0} = \prod_{t=1}^{T} W_{hh} \cdot \text{diag}(\sigma'(z_t))
$$

$T$ 大 + $|W\sigma'| < 1$ → $\to 0$。

### 4.4 He 初始化

$$
\text{Var}(W) = \frac{2}{n_{\text{in}}}
$$

让前向 $\text{Var}(x_L) \approx \text{Var}(x_0)$（ReLU 补偿 0.5 概率零）。

### 4.5 gradient clip

$$
\text{if } \|g\|_2 > c: \quad g \leftarrow g \cdot \frac{c}{\|g\|_2}
$$


## 5. 代码示例（可选

### 5.1 纯 Python 模拟连乘

```python
def gradient_chain(depth, mean_deriv=1.1):
    """链式法则连乘: 梯度 = prod(各层导数). >1 爆炸, <1 消失."""
    g = 1.0
    for _ in range(depth):
        g *= mean_deriv
    return g

print(f'导数 1.1, depth 100: {gradient_chain(100, 1.1):.2e} (爆炸)')
print(f'导数 0.9, depth 100: {gradient_chain(100, 0.9):.2e} (消失)')
print(f'导数 1.0, depth 100: {gradient_chain(100, 1.0):.2e} (稳定)')
print(f'导数 0.25 (sigmoid), depth 100: {gradient_chain(100, 0.25):.2e} (严重消失)')

# residual: 导数含 +1, 即使 F 导数小, 总 >= 1
def residual_chain(depth, f_deriv=0.5):
    g = 1.0
    for _ in range(depth):
        g *= (1 + f_deriv)   # residual: I + dF/dx
    return g
print(f'residual (F导数0.5), depth 100: {residual_chain(100, 0.5):.2e} (不消失, 1.5^100)')
```

### 5.2 gradient clip 模拟

```python
import math
def clip_gradient(g, c=1.0):
    norm = math.sqrt(sum(x*x for x in g))
    if norm > c:
        return [x * c/norm for x in g], norm
    return g, norm

g = [3.0, 4.0]  # norm = 5
clipped, orig_norm = clip_gradient(g, c=1.0)
clipped_norm = math.sqrt(sum(x*x for x in clipped))
print(f'原梯度 norm={orig_norm:.2f} (>1), clip 后 norm={clipped_norm:.2f}')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: 反向传播/链式法则（连乘的来源）、激活函数（导数大小）、权重初始化（Xavier/He）。
- **下游（应用）**: [[loss spike handling]]（爆炸的瞬时表现）、[[clipping strategies]]（爆炸硬限）、[[warmup]]（前期稳）、[[normalization]]（控尺度）、residual 连接（消失缓解）、[[混合精度]]（fp16 易溢出/下溢，bf16 缓解）、[[optimizer state memory]]/Adam（$v$ 下溢与数值稳定）、RNN/LSTM（消失的经典场景）、LLM 训练稳定性。
- **对比 / 易混**:
  - **explosion vs vanishing**：连乘 $>1$ 爆炸 vs $<1$ 消失。前者 NaN，后者不学。对策部分相反（clip 爆炸，residual/He 消失）。
  - **vanishing vs 死神经元**：消失是梯度连乘小（整体），死神经元是 ReLU 负区导数 0（个别）。不同问题，对策不同。
  - **explosion vs [[loss spike handling\|loss spike]]**：spike 是 loss 飙升的现象，爆炸是梯度大的原因。spike 常由爆炸引起。


## 7. 常见误区与易错点

> [!warning] 误区 1：只深层网络有梯度问题
> 浅层也有，但层数少连乘次数少，效应不显。深层（$L$ 大）指数效应放大，才明显。但浅层 + 不当初始化/激活也会爆/消。

> [!warning] 误区 2：爆炸和消失不会同时发生
> 会，不同层不同方向。如深层网络某层 $W$ 大（爆炸）、某层小（消失），梯度在不同层不同行为。诊断需看各层梯度 norm。

> [!warning] 误区 3：sigmoid 适合深层
> 不适合。sigmoid 导数最大 0.25，连乘快速消失。深层用 ReLU/GELU（正区导数 1）。sigmoid 只用于输出（二分类）或门控（LSTM）。

> [!warning] 误区 4：gradient clip 解决消失
> 不解决。clip 只限爆炸（$\|g\| > c$ 缩到 $c$），对消失（$\|g\|$ 小）无作用（不放大）。消失用 residual/He/ReLU 解决。

> [!warning] 误区 5：初始化不重要
> 很重要。不当初始化（$W$ 过大/小）直接导致爆炸/消失。Xavier/He 让 $|W\sigma'| \approx 1$ 是深层可训的前提。LLM 训练对初始化敏感。

> [!warning] 误区 6：fp16 的溢出 = 梯度爆炸
> 相关但不完全。fp16 数值范围小（max ~65504），梯度稍大就溢出 inf（数值溢出，非连乘爆炸）。bf16 范围大（指数位多）不易溢出。混合精度训练用 bf16 缓解。两者都表现为 NaN，但根因不同（数值范围 vs 连乘）。

> [!warning] 误区 7：residual 完全解决消失
> 大幅缓解但不完全。residual 让导数 $\geq 1$ 防消失，但 $F$ 的导数仍可能有问题（如死神经元）。且爆炸风险增（导数 $>1$ 可能）。需配合 normalization/clip。


## 8. 延伸细节

### 8.1 LLM 训练的稳定性挑战

LLM（大模型 + 大 batch + 大 lr + 长序列）训练易 loss spike。根因复杂：梯度爆炸（某层突大）、attention 数值不稳（softmax 大 logit）、bf16 精度损失、大 lr。对策栈：gradient clip + warmup + lower lr + bf16 + [[normalization]] + QK-norm（attention 的 QK 归一化）。[[loss spike handling]] 系统化此问题。

### 8.2 RNN/LSTM 的消失

RNN 时间步连乘是消失经典场景。LSTM 的门控（forget gate $f_t$）让导数可 $>1$（$f_t \approx 1$ 时导数接近 1，长程记忆）。GRU 类似。但 LSTM 仍不如 Transformer 的 attention 跨步直连（attention 的梯度直接传，不连乘）。故 Transformer 取代 RNN 做长序列。

### 8.3 QK-norm

LLM 的 attention 的 QK^T 数值可能很大（大 logit → softmax 饱和 → 梯度消失/不稳）。QK-norm：对 Q, K 做 LayerNorm/RMSNorm 后再算 QK^T，控 logit 尺度。是 LLM 训练稳定性 trick（PaLM/Gemma 等用）。

### 8.4 混合精度与数值稳定

fp16 易溢出（max 65504）和下溢（min ~6e-8）。梯度稍大溢出 inf，稍小下溢 0。对策：

- bf16（范围大，精度低）：训练主流，不易溢出。
- gradient scaling（反向前 scale up 防 fp16 下溢，反向后 unscale）。
- fp32 累加（reduction/optimizer 用 fp32）。

[[混合精度]] 的数值管理是稳定性的重要一环。

### 8.5 gradient clipping 的实现

PyTorch：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`。先算全局梯度 norm，超阈值缩。LLM 训练标配（max_norm=1.0）。clip 在反向后、optimizer 前需先 allreduce 全局梯度（算 $\|g\|$），引入同步点。详见 [[clipping strategies]]。

### 8.6 诊断梯度问题

- **看各层梯度 norm**：大层爆炸，小层消失。`for name, p in model.named_parameters(): print(name, p.grad.norm())`。
- **看权重 norm**：训练中权重持续大→爆炸，小→消失。
- **看 loss 曲线**：突飙升→爆炸，不降→消失。
- **梯度直方图**：多 inf/NaN→爆炸，多 0→消失。

[[torch profiler]]/tensorboard 的梯度直方图辅助诊断。

### 8.7 LLM 的层数与稳定性

LLM 层数多（32-100），连乘次数多，稳定性挑战大。但 residual + LayerNorm + 初始化让深可训。极深（100+）仍需精细调（如更多 residual、smaller lr）。Gemma 2 等用多层 + 残差 + QK-norm 稳定。

---
相关: [[数值稳定]] | [[loss spike handling]] | [[clipping strategies]] | [[warmup]] | [[normalization]] | [[混合精度]] | [[optimizer state memory]] | [[KL divergence control]] | [[reward normalization]] | [[entropy regularization]] | [[attention]] | [[Transformer]] | [[torch profiler]]
