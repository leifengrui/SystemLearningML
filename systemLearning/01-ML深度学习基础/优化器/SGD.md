# SGD

> **所属章节**: [[优化器]]
> **所属模块**: [[01-ML深度学习基础]]
> **难度**: 基础

## 1. 一句话定义

**SGD（Stochastic Gradient Descent，随机梯度下降）** 是最基础的优化器：每个 step 用一个 **mini-batch 估计的梯度** $\hat{g}$，按 $\theta \leftarrow \theta - \eta\,\hat{g}$ 更新参数；"随机"指用 mini-batch 梯度近似全数据梯度（有噪声但便宜），"梯度下降"指沿负梯度方向走一步。

> [!note] 解答：什么是 mini-batch？
> **mini-batch（小批）** 是从训练集里随机抽出的**一小撮样本**（如 32/64/256 条），用这批样本的平均梯度 $\hat g=\frac1{|B|}\sum_{i\in B}\nabla\ell_i$ 去近似"全数据梯度"。用小批而非全集的原因：全集一次太贵（百万样本算一次要几小时）、单条又噪声太大且吃不饱 GPU；小批是**速度/精度/算力**的甜点。三种梯度下降的对比见本节下文，完整展开见 [[batch与mini-batch]]。

## 2. 为什么需要它（动机与背景）

全量数据算一次梯度（BGD, batch gradient descent）虽然准，但：

- 数据集百万级时一次前向+反向就要几小时，根本训不动；
- 全量梯度过于"平滑"，容易陷在鞍点。

> [!note] 解答：怎么理解"有噪"？
> "有噪"指 mini-batch 梯度是全数据梯度的**随机估计**：每次抽的样本不同 → 估出的梯度在真值附近**抖动**，这就是噪声。数学上：若单样本梯度方差 $\sigma^2$，则小批平均 $\hat g$ 的方差 $\text{Var}(\hat g)=\sigma^2/|B|$ —— batch 越小方差越大、越抖；batch 越大方差越小、越接近真梯度（仅当 batch=全集时方差才为 0）。
> 噪声有两面性：**坏处**是收敛抖、要配 [[学习率调度]] 退火；**好处**是能把参数从尖锐极小值"震"出来、落到平坦解、泛化更好（见下文"逃离尖锐极小值"）。所以 SGD 的"随机"不是缺陷，而是被刻意保留的特性。

**SGD 的核心权衡**：用 mini-batch 的**有噪梯度**换取**高频更新**。每 step 只算几十~几千样本，秒级出梯度，一天能跑几万步。噪声反而有助于逃离尖锐极小值，得到泛化更好的解。

> [!note] BGD vs SGD vs mini-batch
> - **BGD**：用全部数据，梯度准、更新慢、易陷鞍点。
> - **SGD**（严格义）：用 1 个样本，噪声极大、无法用 GPU 并行。
> - **mini-batch SGD**：用几十~几千样本，兼顾速度/精度/GPU 利用率 —— 深度学习里说的"SGD"几乎都是这个。

## 3. 核心概念详解

### 3.1 更新公式

$$
\theta_{t+1} = \theta_t - \eta\, g_t,\qquad g_t = \frac{1}{|B|}\sum_{i\in B}\nabla_\theta \ell(f_\theta(x_i),y_i)
$$

- $\eta$：学习率（步长）。
- $g_t$：mini-batch $B$ 上的平均梯度。
- 一次 step = 一次 `zero_grad + forward + backward + step`。

### 3.2 三个超参数

| 参数 | 作用 | 典型 |
|---|---|---|
| `lr` ($\eta$) | 步长，最关键 | 0.01~0.1（分类）、1e-4~1e-5（LLM） |
| `batch_size` | 梯度噪声尺度 | 32~512（CV）、百万 token（LLM） |
| `momentum` | 见 [[Momentum]] | 0.9 |

### 3.3 噪声的作用与代价

- **好处**：梯度噪声让参数在平坦极小值附近"抖动"，逃离尖锐极小值 → 泛化好（SGD 在 CV 至今仍被泛化性能榜单青睐）。
- **代价**：方差大、收敛慢、对学习率敏感，需要 [[学习率调度]] 与 [[Momentum]] 配合。

### 3.4 权重衰减（weight decay, L2 正则）

SGD 的 L2 正则把惩罚加在**梯度**上：

$$
g_t \leftarrow g_t + \lambda\,\theta_t,\qquad \theta_{t+1}=\theta_t-\eta(g_t+\lambda\theta_t)
$$

等价于每步额外把参数往 0 拉 $\lambda\eta$ 比例。防过拟合，但与真正的"权重衰减"$\theta\leftarrow(1-\eta\lambda)\theta$ 在自适应优化器里不等价 —— 这是 [[Adam与AdamW]] 引入 AdamW 的原因。

## 4. 数学原理 / 公式

### 收敛性

凸优化下，SGD 期望收敛速率 $O(1/\sqrt{T})$（次线性），依赖步长递减 $\eta_t\propto 1/\sqrt{t}$ 与梯度方差有界。非凸（深度学习）只有到驻点的保证，但实践中泛化好。

### 为何负梯度是最速下降

由一阶泰勒：$L(\theta+d)\approx L(\theta)+g^\top d$，$\|d\|\le1$ 下最小化 $g^\top d$ 得 $d=-g/\|g\|$（柯西-施瓦茨）。所以负梯度方向单位步长下降最快。

### 学习率与 batch 的 scaling

线性缩放法则（linear scaling rule）：batch 放大 $k$ 倍，学习率也放大 $k$ 倍可保持单样本贡献不变（大 batch 训练经验法则，见 [[batch]]）。

## 5. 代码示例

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 2)
# SGD 标准用法：lr + 可选 momentum + weight_decay(L2)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

x = torch.randn(32, 10)
y = torch.randn(32, 2)
loss_fn = nn.MSELoss()

for step in range(5):
    opt.zero_grad(set_to_none=True)      # 清梯度（SGD 不自动清）
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()                      # 算梯度 -> .grad
    opt.step()                           # theta -= lr * grad
    print(step, loss.item())
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[backward过程]]（提供 `.grad`）、[[梯度与Jacobian]]（梯度即更新方向）。
- **下游（应用）**: [[Momentum]]（SGD + 动量）、[[Adam与AdamW]]（自适应版）、[[学习率调度]]（调控 $\eta$）、[[梯度裁剪]]（防爆炸）。
- **对比 / 易混**:
  - **SGD vs Adam**：SGD 噪声大泛化好但慢、对 lr 敏感；Adam 自适应快但泛化有时差，LLM 几乎只用 AdamW。
  - **SGD vs BGD**：见 2 节，权衡噪声与速度。
  - **weight decay vs L2**：在 SGD 下等价，在 Adam 下不等价（见 [[Adam与AdamW]]）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **忘 `zero_grad()`**：SGD 不会自动清梯度，累加导致参数飞掉。
> 2. **学习率不调度**：固定 lr 后期会震荡，需 [[学习率调度]] 退火。
> 3. **batch 改了不改 lr**：放大 batch 不放 lr，等效学习率太小收敛慢（违反线性缩放）。
> 4. **把 weight decay 当 Adam 的 L2**：AdamW 才是解耦权重衰减，普通 Adam+weight_decay 是 L2，不等价。
> 5. **以为 SGD 已过时**：CV 分类/检测 SOTA 仍常用 SGD+momentum，泛化更稳；LLM 才转向 AdamW。

## 8. 延伸细节

### 8.1 SGD 的"泛化优势"假说

SGD 噪声让解落在**平坦极小值**（flat minima），平坦解对各方向扰动不敏感 → 测试误差低。Adam 噪声小、解偏尖锐，泛化有时差。这是"SGD 调不出 Adam 速度，但泛化更稳"的根因。

### 8.2 Nesterov 动量

SGD 的 `momentum` 还可设 `nesterov=True`，先按动量"前瞻"一步再算梯度，理论上凸问题收敛更快。见 [[Momentum]]。

### 8.3 大 batch 训练

LLM 用超大 batch（百万 token/step），SGD 噪声不足、收敛差，且学习率要极大，故转向自适应的 AdamW + warmup + cosine。

---
相关: [[优化器]]、[[Momentum]]、[[Adam与AdamW]]、[[学习率调度]]、[[梯度裁剪]]
