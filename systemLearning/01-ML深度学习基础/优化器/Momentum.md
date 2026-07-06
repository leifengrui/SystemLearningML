# Momentum

> **所属章节**: [[优化器]]
> **所属模块**: [[01-ML深度学习基础]]
> **难度**: 基础

## 1. 一句话定义

**动量（Momentum，动量法）** 在 [[SGD]] 基础上引入一个**历史梯度指数加权平均**作为"速度"，让更新方向不仅看当前梯度、还累积过去梯度，从而**加速收敛、抑制振荡、冲过小障碍**；PyTorch 里就是 `SGD(..., momentum=0.9)`。

## 2. 为什么需要它（动机与背景）

纯 SGD 有三个痛点：

1. **峡谷地形震荡**：loss 曲面在某个方向窄而陡、另一方向宽而缓，纯梯度会在陡壁间来回弹跳，沿缓方向进展极慢。
2. **小梯度区爬不动**：平坦区梯度小，步长小，收敛龟速。
3. **噪声抖动**：mini-batch 噪声让参数抖动，难以稳定下降。

动量借鉴物理直觉：**滚下一个有摩擦的球**，速度会累积历史受力（梯度），即使某一刻受力为 0，惯性还能往前冲。于是：

- 在一致方向上**加速**（动量同向叠加）；
- 在震荡方向上**抵消**（正负交替平均掉）；
- 在小梯度区**维持速度**继续前进。

## 3. 核心概念详解

### 3.1 经典动量更新

引入速度变量 $v$（与 $\theta$ 同形）：

$$
v_{t+1} = \mu\, v_t + g_t,\qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}
$$

- $\mu\in[0,1)$：动量系数，常用 **0.9**。
- $g_t$：当前 mini-batch 梯度。
- $v_t$：历史梯度的指数加权平均（"速度"）。

### 3.2 指数加权平均视角

把递推展开：

$$
v_t = g_t + \mu g_{t-1} + \mu^2 g_{t-2} + \dots = \sum_{k\ge0}\mu^k g_{t-k}
$$

即过去梯度按 $\mu^k$ 衰减加权求和。有效记忆窗口约 $\frac{1}{1-\mu}$ 步（$\mu=0.9$ 时约 10 步）。

### 3.3 Nesterov 动量（前瞻版）

普通动量先算梯度再叠速度；Nesterov 先按速度"前瞻"一步，在**前瞻点**算梯度：

$$
v_{t+1}=\mu v_t + \nabla L(\theta_t-\eta\mu v_t),\qquad \theta_{t+1}=\theta_t-\eta v_{t+1}
$$

直觉：先探一步再修正方向，对"急转弯"反应更快。凸问题理论收敛率更优；实践中差距不大，PyTorch 用 `SGD(..., momentum=0.9, nesterov=True)`。

### 3.4 动量 vs 自适应

动量是**方向上**的加速（所有参数共享一个速度方向感）；自适应（Adam）是**逐参数**的步长缩放。两者正交，Adam 把动量（一阶矩）和自适应（二阶矩）都纳入。

## 4. 数学原理 / 公式

### 收敛加速

对强凸二次问题，动量把条件数相关的收敛因子从 $\kappa$ 降到 $\sqrt{\kappa}$，迭代次数从 $O(\kappa)$ 降到 $O(\sqrt{\kappa})$。最优 $\mu$ 与条件数相关（重球法 heavy-ball）。

### 为何能穿峡谷

设两方向梯度交替正负 $+g,-g,+g,\dots$，则 $v_t\approx g(1-\mu+\mu^2-\dots)=\frac{g}{1+\mu}$，远小于瞬态 $g$ → 震荡方向被衰减；而一致方向 $g,g,g,\dots$ 则 $v_t\approx\frac{g}{1-\mu}$，被放大 $\frac{1}{1-\mu}$ 倍（$\mu=0.9$ 放大 10 倍）。这就是"震荡压、一致提"。

### 等效学习率

带动量后等效步长约 $\frac{\eta}{1-\mu}$，所以带动量时常**降低 lr**（经验：从 SGD 的 0.1 降到 0.02 左右，看任务）。

## 5. 代码示例

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 2)
# 动量法
opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
# Nesterov
opt_n = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, nesterov=True)

x, y = torch.randn(32, 10), torch.randn(32, 2)
loss_fn = nn.MSELoss()
for step in range(5):
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()          # 内部维护 param_state['momentum_buffer'] = mu*v + g
    print(step, loss.item())

# 手写动量看清状态
buf = None
mu, lr = 0.9, 0.02
for p in model.parameters():
    g = p.grad
    buf = mu * buf + g if buf is not None else g.clone()
    p.data -= lr * buf
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[SGD]]（动量是 SGD 的扩展）、[[backward过程]]（梯度来源）。
- **下游（应用）**: [[Adam与AdamW]]（一阶矩就是动量）、[[学习率调度]]（动量改变等效 lr）、训练稳定性。
- **对比 / 易横**:
  - **Momentum vs 纯 SGD**：动量加速 + 抑振，几乎总该开（CV 默认 0.9）。
  - **Momentum vs Nesterov**：Nesterov 前瞻，凸问题理论更优。
  - **Momentum vs Adam 自适应**：前者方向加速，后者逐参数步长，正交。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **带动量却不降 lr**：等效步长变 $\frac{1}{1-\mu}$ 倍，原 lr 会过大发散。
> 2. **动量系数乱设**：0.9 是经验值，0.99 会过度惯性冲过头，0.5 几乎没用。
> 3. **以为动量=Adam**：动量不分参数共享方向感；Adam 还按二阶矩逐参数缩放。
> 4. **动量 buffer 不清**：`zero_grad` 只清 `.grad`，不清动量 buffer（一般也不该清，它就是要累积）；但断点续训要确保 state_dict 含 momentum_buffer。
> 5. **Nesterov 在非凸没保证**：理论优势是凸问题的，深度学习里只是经验略好。

## 8. 延伸细节

### 8.1 重球法与共轭梯度

动量法是 heavy-ball 法的实例；共轭梯度（CG）可看作"自适应动量"，每步选最优共轭方向，凸二次问题 $n$ 步收敛。二阶优化（如 K-FAC）借鉴其思想。

### 8.2 动量与训练稳定性

大动量（0.99）在前期可能"冲下悬崖"导致 loss spike，故常配合 [[学习率调度]] 的 warmup：前期小 lr + 小动量，稳定后再放大。

### 8.3 Lookahead 等变体

Lookahead 在内循环跑 k 步动量后，外循环把"快权重"往"慢权重"方向拉一步，提升稳定性与泛化，是动量的廉价增强。

### 8.4 Adam 里动量的体现

Adam 的 $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$ 就是偏差校正后的动量（$\beta_1=0.9$）。所以理解动量是理解 Adam 的前置。

---
相关: [[优化器]]、[[SGD]]、[[Adam与AdamW]]、[[学习率调度]]
