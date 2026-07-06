# 梯度与Jacobian

> **所属章节**: [[张量与自动微分]]
> **所属模块**: [[01-ML深度学习基础]]
> **难度**: 进阶

## 1. 一句话定义

**梯度（gradient）** 是标量函数对向量输入的一阶导数（一个**向量**，指向最速上升方向）；**Jacobian（雅可比矩阵）** 是向量值函数对向量输入的一阶导数（一个**矩阵**，每行/列是一个输出的梯度）；它们是 [[autograd机制]] 与 [[backward过程]] 在数学上真正计算的对象，也是优化器更新参数的依据。

## 2. 为什么需要它（动机与背景）

训练就是"调参让 loss 降"。要降，就得知道**往哪个方向调**：

- 一维参数 $w$：看 $\frac{dL}{dw}$ 的正负，往反方向走一点。
- 多维参数 $\boldsymbol{\theta}$：要找一个方向 $\boldsymbol{d}$，使 $L(\boldsymbol{\theta}+\boldsymbol{d})$ 下降最快 —— 这个方向就是**负梯度**。

所以梯度是优化的"方向盘"。没有梯度，梯度下降、Adam、PPO 全都无从下手。

而 Jacobian 是梯度的推广：当**输出也是向量**时（如一个 batch 的 logits），导数从向量变成矩阵。深度学习里 loss 是标量、参数是向量，所以**主要用梯度**；但理解 Jacobian 能说清"为什么反向传播比正向高效"、"注意力梯度怎么传"，是进阶必备。

## 3. 核心概念详解

### 3.1 梯度

函数 $L: \mathbb{R}^n \to \mathbb{R}$（标量输出），其梯度：

$$
\nabla_{\boldsymbol{x}} L = \left[\frac{\partial L}{\partial x_1},\dots,\frac{\partial L}{\partial x_n}\right]^\top \in \mathbb{R}^n
$$

性质：$\nabla L$ 指向 $L$ **上升最快**的方向，所以梯度下降走 $-\nabla L$。

### 3.2 Jacobian 矩阵

函数 $\boldsymbol{f}: \mathbb{R}^n \to \mathbb{R}^m$（$m$ 个输出），其 Jacobian：

$$
J_{\boldsymbol{f}}(\boldsymbol{x}) =
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
\in \mathbb{R}^{m\times n}
$$

- 行 = 某个输出对所有输入的偏导（即 $\nabla f_i$）。
- 当 $m=1$（标量输出），Jacobian 退化为梯度（行向量的转置）。

### 3.3 Hessian（二阶）

二阶导数矩阵 $H_{ij}=\frac{\partial^2 L}{\partial x_i\partial x_j}$。描述梯度的变化率（曲率），用于牛顿法、收敛性分析。深度学习通常不算（参数太多，$H$ 是 $n\times n$ 太大），但训练稳定性、loss spike 分析里会定性讨论。

### 3.4 两种雅可比积（关键）

这是理解 autograd 效率的核心：

| 名称 | 形式 | 含义 | 对应模式 | 算一次得到 |
|---|---|---|---|---|
| **JVP**（Jacobian-Vector Product） | $J\,\boldsymbol{v}$ | 给输入方向 $\boldsymbol{v}$，算输出方向变化 | **正向模式** | Jacobian 的一**列** |
| **VJP**（Vector-Jacobian Product） | $\boldsymbol{u}^\top J$ | 给输出方向 $\boldsymbol{u}$，算输入端梯度 | **反向模式** | Jacobian 的一**行**（即一个输出的梯度） |

> [!note] autograd 用的是 VJP
> `backward(grad_output)` 里传的 `grad_output` 就是 $\boldsymbol{u}$，算出 $\boldsymbol{u}^\top J$ 给上游。loss 是标量（$m=1$），一次反向（一个 $\boldsymbol{u}=1$）就拿到**所有输入**的梯度 —— 这就是反向模式高效的原因。

### 3.5 链式法则的矩阵形式

设 $\boldsymbol{z}=g(\boldsymbol{x})$，$\boldsymbol{y}=f(\boldsymbol{z})$。复合 Jacobian：

$$
J_{\boldsymbol{y}\circ\boldsymbol{z}} = J_f\, J_g
$$

反向传播时，已知输出端伴随 $\bar{\boldsymbol{y}}=\boldsymbol{u}$，逐层往回算：

$$
\bar{\boldsymbol{z}} = J_f^\top \bar{\boldsymbol{y}}, \qquad
\bar{\boldsymbol{x}} = J_g^\top \bar{\boldsymbol{z}}
$$

即每层做一次 VJP（转置雅可比乘向量）。[[backward过程]] 就是反复套这个。

## 4. 数学原理 / 公式

### 梯度下降的数学依据

一阶泰勒展开：$L(\boldsymbol{x}+\boldsymbol{d}) \approx L(\boldsymbol{x}) + \nabla L^\top \boldsymbol{d}$。要让 $L$ 降最快，在 $\|\boldsymbol{d}\|\le 1$ 下最小化 $\nabla L^\top\boldsymbol{d}$，由柯西-施瓦茨不等式，$\boldsymbol{d}=-\nabla L/\|\nabla L\|$ 时最小。所以**负梯度方向 = 最速下降方向**。

### 反向模式复杂度

对 $n$ 输入、$m$ 输出：

- 正向模式算全 Jacobian：$O(n)$ 次遍历（每次拿一列）。
- 反向模式算"一个标量输出对全输入"的梯度：$O(1)$ 次遍历（约 2~3× 前向开销）。

loss 标量、参数 $n$ 很大 → 反向模式完胜。这也是 autograd 默认反向的原因。

### 注意力梯度里的 Jacobian 思维

attention 输出 $\text{softmax}(QK^\top/\sqrt d)V$ 对 $Q$ 的梯度涉及一个"逐位置"的 Jacobian。手推很繁，但 autograd 把它拆成 softmax、matmul 等基本算子的 VJP 自动串联。

## 5. 代码示例

```python
import torch

# 1. 标量 loss 对向量的梯度（autograd 主战场）
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
L = (x ** 2).sum()           # L = x1^2+x2^2+x3^2，梯度应为 2x
L.backward()
print(x.grad)                # tensor([2., 4., 6.])  == 2*x

# 2. 显式计算 Jacobian（向量输出 -> 多次反向，贵）
def f(t):
    return torch.stack([t.sum(), t.prod(), (t**2).sum()])  # 3 个输出

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
J = torch.autograd.functional.jacobian(f, x)
print(J.shape)              # torch.Size([3, 3]) —— 3 输出 × 3 输入
# 第 i 行 = 第 i 个输出对 x 的梯度

# 3. 手算验证：vjp = grad_output^T J
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.stack([x.sum(), (x**2).sum()])   # 2 输出
# 反向：传 u=[1, 1]，得到 x.grad = sum of each row of J
y.sum().backward()
print(x.grad)               # = [1,1,1] + [2,4,6] = [3,5,7]

# 4. 正向模式 jvp（对输出多更高效）
from torch.func import jvp, vjp
def f2(t):
    return (t ** 2).sum()
x = torch.tensor([1.0, 2.0, 3.0])
# vjp: 给输出伴随，返回 (输出, 输入梯度函数)
primals, cotangent_fn = vjp(f2, x)
print(primals)              # 14 = 1+4+9
grads = cotangent_fn(torch.tensor(1.0))   # 标量伴随 1
print(grads)                # tensor([2.,4.,6.]) —— 梯度

# jvp: 给输入方向，返回 (输出, Jv)
primals, tangents = jvp(f2, (x,), (torch.ones(3),))
print(tangents)             # J·[1,1,1] = [2,4,6]·[1,1,1] = 12
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[autograd机制]]（autograd 算的就是 VJP）、[[计算图]]（链式法则在图上的体现）。
- **下游（应用）**: [[backward过程]]（反向传播 = 反复 VJP）、各类优化器（用梯度更新）、[[advantage function]]（RL 里的梯度估计）、policy gradient（[[REINFORCE]] 用 $\nabla_\theta J$）。
- **对比 / 易混**:
  - **梯度 vs Jacobian**：标量输出用梯度（向量）；向量输出用 Jacobian（矩阵）。
  - **JVP vs VJP**：正向 vs 反向，分别拿 Jacobian 的列和行。
  - **梯度 vs subgradient**：不可微点（如 relu 在 0）用次梯度，autograd 直接取 0。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **梯度形状搞反**：$L$ 对 `x`（shape `(n,)`) 的梯度形状就是 `(n,)`，与 `x` 同形，不是 `(1,n)`。
> 2. **以为显式算 Jacobian 是常态**：实际训练**从不**显式算 Jacobian（太贵 $O(n)$ 次反向），都是用 VJP 一次反向拿标量 loss 对全参数梯度。
> 3. **把"梯度方向"当"最优方向"**：梯度只是一阶信息，大步长可能跨过最优；曲率（Hessian）信息才会更准（牛顿法）。
> 4. **不可微点的梯度**：relu 在 0、argmax 不可微，autograd 用约定值（relu@0 取 0）或 surrogate；离散采样要用 Gumbel-softmax/REINFORCE 绕过。
> 5. **batch 维度的 Jacobian 维度爆炸**：`(B, n)` 输出对 `(B, n)` 输入的 Jacobian 是 `(B,n,B,n)`，显式算会爆 —— 实际只在 batch 内求和后拿标量梯度。
> 6. **梯度 ≠ 参数更新量**：梯度只指方向，步长由学习率和优化器（动量等）决定。

## 8. 延伸细节

### 8.1 为什么不用正向模式

正向模式（JVP）一次遍历只拿 Jacobian 一列，要算"标量 loss 对全参数"（Jacobian 的一行，参数几十亿）得算几十亿次，不可行。反向模式一次反向就拿全行，所以深度学习锁死反向模式。

### 8.2 海森-向量积（HVP）

不算整个 Hessian（太大），只算 $H\boldsymbol{v}$：先正向算 $J\boldsymbol{v}$，再对其反向。用于影响函数、共轭梯度二阶优化、训练稳定性分析。

### 8.3 自然梯度 / Fisher 信息

不直接走 $-\nabla L$，而走 $-F^{-1}\nabla L$（$F$ 为 Fisher 信息矩阵），考虑了参数空间的"概率几何"。与 KL 散度控制、TRPO/KL penalty 相关，见 [[PPO]] 与 [[KL penalty]]。

### 8.4 梯度的数值估计

RL 里 reward 不可微，无法用 autograd 拿 $\nabla_\theta J$，于是用**梯度估计（gradient estimator）**：$\nabla_\theta \mathbb{E}[R] = \mathbb{E}[\nabla_\theta \log\pi \cdot R]$（REINFORCE/score function estimator）。这是 policy gradient 的数学根基。

---
相关: [[张量与自动微分]]、[[autograd机制]]、[[backward过程]]
