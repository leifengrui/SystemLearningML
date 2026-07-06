# autograd机制

> **所属章节**: [[张量与自动微分]]
> **所属模块**: [[01-ML深度学习基础]]
> **难度**: 基础

## 1. 一句话定义

**autograd（自动微分引擎，Automatic Differentiation）** 是 PyTorch 内置的自动求导系统：只要张量设了 `requires_grad=True`，它记录前向运算形成 [[计算图]]，调用 `.backward()` 时自动沿图反推、把**每个参数的梯度累加到 `.grad`** 属性里，无需手写一行导数代码。

## 2. 为什么需要它（动机与背景）

LLM 有几百亿参数，每个都要算梯度。如果手写导数：

> [!note] 解答：梯度就是导数吗？"沿着该方向变化最快的量"对吗？
> **严格说，梯度（gradient）是导数（derivative）在多变量情形下的推广，两者相关但不完全等同。** 你的直觉方向是对的，下面拆清楚：
>
> | 概念 | 适用函数 | 形状 | 含义 |
> |---|---|---|---|
> | **导数 $f'(x)$** | 单变量 $f:\mathbb{R}\to\mathbb{R}$ | 标量 | 函数在该点的瞬时变化率（斜率） |
> | **梯度 $\nabla f$** | 多变量标量函数 $f:\mathbb{R}^n\to\mathbb{R}$ | **向量** $(\partial f/\partial x_1,\dots,\partial f/\partial x_n)$ | 各偏导数排成的向量 |
> | **Jacobian** | 向量值函数 $f:\mathbb{R}^n\to\mathbb{R}^m$ | $m\times n$ 矩阵 | 所有偏导组成的矩阵 |
>
> 三点关键：
> 1. **梯度是向量，导数是标量**。单变量时 $n=1$，梯度就退化成导数；多变量时梯度把每个方向的偏导打包成一个向量。所以"梯度 = 一堆偏导数组成的向量"。
> 2. **"沿着该方向变化最快"——对，但要说清是升还是降**。$\nabla f$ 指向的是 $f$ **上升最快**的方向，$|\nabla f|$ 是该最大变化率的大小；$-\nabla f$ 才是**下降最快**的方向。这就是**梯度下降法** $w \leftarrow w - \eta \nabla L$ 的数学根：沿负梯度走，loss 降得最快。
> 3. **深度学习里说的"梯度"**：loss $L$ 是标量，参数 $\theta$ 是几十亿维向量，我们要的 $\partial L/\partial \theta$ 就是一个和 $\theta$ 同形状的**梯度向量**。autograd 算的、落到 `.grad` 里的就是这个。日常大家把 $\partial L/\partial w$ 既叫"梯度"也叫"导数"，混用，但严格讲是梯度（多变量）。
>
> 一句话：**导数是单变量的斜率；梯度是多变量的偏导向量，指向函数上升最快的方向，负梯度就是下降最快的方向——这正是训练要走的路。** 详见 [[梯度与Jacobian]]。

- 模型结构一改，导数全得重推；
- 激活函数、layernorm、注意力这些复合算子的导数极容易写错；
- 工程上根本不可维护。

autograd 解决的正是"**任意可微函数的梯度自动化**"。它的哲学是 **define-by-run**：

> 你只负责写前向（就是普通 Python），我负责记录你做了什么、然后自动反推梯度。

这样模型代码 = 前向代码，梯度是"赠品"。这是 PyTorch 能在研究领域快速迭代的核心原因。

## 3. 核心概念详解

### 3.1 触发开关：`requires_grad`

- 默认 `Tensor` 的 `requires_grad=False`，不参与自动微分。
- 设 `True` 后，所有以它为输入的运算结果，都会被记录进图（带 `grad_fn`）。
- `nn.Parameter` 自动 `requires_grad=True`，所以模型参数天然可训练。

### 3.2 三类关键属性

| 属性 | 含义 |
|---|---|
| `requires_grad` | 是否需要对该张量求梯度 |
| `grad_fn` | 指向"产生它的算子"的反向函数（建图的证据） |
| `.grad` | 反向后梯度**累加**到这里（默认初始为 None） |

### 3.3 叶子节点（leaf tensor）

- 用户直接创建的、`requires_grad=True` 的张量是**叶子节点**（`is_leaf=True`）。
- 只有叶子节点的 `.grad` 会被**保留**；非叶子（中间结果）的 `.grad` 默认反向后被清掉，省内存。
- 想看中间梯度要 `.retain_grad()`。

> [!note] 参数都是叶子
> `nn.Linear` 里的 `weight`/`bias` 是用户创建的 `Parameter`，所以是叶子 —— 梯度最终落在这里，正好给优化器更新。

### 3.4 梯度累加（accumulate）

`.grad` 是**累加**而非覆盖：连续两次 `backward()`（且 `retain_graph=True`），梯度会相加。因此每个训练 step 前必须 `optimizer.zero_grad()`（或 `set_to_none=True`）清零，否则梯度越来越大。

### 3.5 关闭建图的三种方式

| 方式 | 语义 | 典型场景 |
|---|---|---|
| `with torch.no_grad():` | 该块内**不建图**，结果 `requires_grad=False` | 推理、评估、rollout 生成数据 |
| `tensor.detach()` | 返回一个**切断图连接**的同值张量 | 把生成数据喂回训练，避免梯度回流 |
| `requires_grad_(False)` | 就地把该张量改为不求导 | 冻结某些参数 |

> [!warning] rollout 数据必须 detach
> RLHF 里 rollout 产生的 token 序列要喂给 policy 算 loss。如果不 `detach()`，梯度会沿着采样过程回传到生成时，既错误又爆显存。

## 4. 数学原理 / 公式

autograd 实现的是 **反向模式自动微分（reverse-mode AD）**，本质是 [[计算图]] 上的链式法则。

设前向链 $u = f(x),\; v = g(u),\; L = h(v)$。autograd 反向时对每个节点计算并传递"上游传来的梯度 × 本节点局部导数"：

$$
\bar{u} = \bar{v}\cdot\frac{\partial v}{\partial u}, \qquad
\bar{x} = \bar{u}\cdot\frac{\partial u}{\partial x}
$$

其中 $\bar{\cdot}$ 表示"伴随导数（adjoint）"，即 $L$ 对该节点的梯度。对多元链式法则（多后继），是**求和**：

$$
\bar{x} = \sum_{s \in \text{后继}} \bar{s}\cdot\frac{\partial s}{\partial x}
$$

每个算子只需提供两样东西：

1. **forward**：算出输出；
2. **backward（vjp）**：给定 $\bar{y}$（输出端的伴随），算出 $\bar{x}$（输入端的伴随），即 $\bar{x} = \bar{y}\cdot J_{y\to x}$（向量-雅可比积，见 [[梯度与Jacobian]]）。

> [!tip] 每个算子只关心自己的局部导数
> autograd 把"算全图梯度"拆成"每个算子算自己那段 vjp"，然后让计算图把它们串起来 —— 这就是它能自动化的关键。

### 自定义算子导数

新算子通过继承 `torch.autograd.Function` 提供 `forward` 和 `backward`：

```python
class MyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)        # 存反向要用的前向值
        return x.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):     # grad_output = 上游传来的 \bar{y}
        x, = ctx.saved_tensors
        return grad_output * (x > 0).float()   # 返回每个输入的 \bar{x}
```

## 5. 代码示例

```python
import torch

# 1. 开启求导
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)   # 叶子节点
W = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

# 2. 前向 —— autograd 自动建图
y = W @ x + b              # grad_fn 指向 AddBackward
L = y.pow(2).sum()         # 标量根节点

# 查看图
print(y.requires_grad, y.grad_fn)         # True <AddBackward0...>
print(x.is_leaf, W.is_leaf)               # True True（用户创建）
print(y.is_leaf)                          # False（运算结果，非叶子）

# 3. 反向 —— 自动算梯度
L.backward()
print(x.grad, W.grad, b.grad)             # 梯度已填回叶子节点

# 4. 梯度累加 —— 必须手动清零
L2 = (W @ x + b).pow(2).sum()
L2.backward(retain_graph=True)
print(W.grad)              # 注意：这是 L 和 L2 梯度之和（累加！）—— 所以训练要 zero_grad

# 5. 关闭建图
with torch.no_grad():     # 推理/评估
    y_inf = W @ x + b
    print(y_inf.requires_grad)            # False，无 grad_fn，省内存

# 6. detach 切断图
x_no_grad = x.detach()    # 同值但 requires_grad=False
# 把 rollout/生成数据喂回训练时常用，防止梯度回流到采样过程

# 7. 看中间梯度
y2 = W @ x + b
y2.retain_grad()          # 非叶子默认不留 grad，显式 retain
y2.sum().backward()
print(y2.grad)            # 否则是 None
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[计算图]]（autograd 的数据结构）、[[Tensor]]（`requires_grad` 是张量属性）。
- **下游（应用）**: [[backward过程]]（autograd 的触发入口）、[[梯度与Jacobian]]（autograd 算的就是 vjp）、各类优化器（消费 `.grad` 更新参数）。
- **对比 / 易混**:
  - **autograd vs 手动求导**：autograd 自动、通用；手动快但不可维护。
  - **autograd vs 数值求导**：数值求导（差分）有截断误差且慢 $O(n)$；autograd 精确且只需一次反向。
  - **`detach` vs `no_grad`**：`no_grad` 是上下文管理器影响一段代码；`detach` 作用于单个张量。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **忘记 `zero_grad()`**：梯度累加导致越积越大，训练发散 —— 这是最常见 bug。
> 2. **对非标量直接 `backward()` 报错**：`loss` 必须是标量；若是向量，要传 `gradient=` 参数（即 $\bar{y}$）或先 `.sum()`。
> 3. **`detach()` 忘了导致梯度回流**：RL/数据处理把生成张量喂回网络时，不 detach 会沿采样路径回传，错误且爆显存。
> 4. **改了叶子节点 `.data` 但 autograd 不知道**：`.data` 直接改值绕过图，可能让反向用错值算梯度，应用 `with torch.no_grad()` 改。
> 5. **以为 `requires_grad=True` 就一定会更新参数**：还要优化器消费 `.grad`，且 `requires_grad` 只管"算不算梯度"，不管"怎么更新"。
> 6. **in-place 改动破坏前向保存的中间值**：autograd 反向时要用前向某些值，原地改后算错或报错。

## 8. 延伸细节

### 8.1 set_to_none 优化

PyTorch 2.x 默认 `zero_grad(set_to_none=True)`：不把 `.grad` 清成 0，而是设成 `None`，省内存且优化器跳过 None 的参数。比清零更快。

### 8.2 二阶导 / 高阶导

autograd 支持高阶导：在 `backward(create_graph=True)` 下，反向过程本身也建图，于是可以再 backward。用于元学习（MAML）、Hessian 相关分析，但开销大。

### 8.3 检测梯度正确性

`torch.autograd.gradcheck` 用**数值差分**和你写的算子解析导数对比，写自定义算子时必做，防止 backward 写错还跑得通。

### 8.4 混合精度下的 autograd

bf16/fp16 前向时，autograd 反向用的局部导数也用低精度；涉及精度的关键累加（如 loss scaling）由 loss scaler 协调，见 [[mixed precision training]] 与 [[loss scaling]]。

### 8.5 与 forward mode 的关系

autograd 主要做 reverse-mode（标量损失→全参数梯度）。若要显式 Jacobian，可用 `torch.autograd.functional.jacobian`（多次反向，贵）或 `torch.func.jacfwd`（正向模式，对输出多更高效）。

---
相关: [[张量与自动微分]]、[[计算图]]、[[backward过程]]、[[梯度与Jacobian]]
