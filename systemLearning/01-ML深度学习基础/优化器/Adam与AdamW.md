# Adam与AdamW

> **所属章节**: [[优化器]]
> **所属模块**: [[01-ML深度学习基础]]
> **难度**: 中级

## 1. 一句话定义

**Adam（Adaptive Moment Estimation，自适应矩估计）** 是把 [[Momentum]]（一阶矩）和逐参数自适应步长（二阶矩梯度平方）结合的优化器；**AdamW** 是其修正版——把**权重衰减从梯度项解耦**，直接作用在参数上，成为 LLM 训练的事实标准（`AdamW` + warmup + cosine）。

> [!note] 解答：优化器优化了什么？起到了什么作用？
> **优化的是模型的参数 $\theta$**（网络所有可学权重/偏置，不是数据、不是损失本身），目标是让损失 $L(\theta)$ 下降、使模型预测尽量贴近数据。优化器起的作用是把"梯度"变成"参数更新"，具体三件事：
> 1. **定方向**：根据梯度（+动量/自适应项）决定每个参数往哪走——负梯度方向下降。
> 2. **定步长**：给每个参数定一个合理步长（学习率），并按历史梯度自适应缩放（Adam 的 $v_t$ 让频繁更新的参数步长小、罕见的步长大）。
> 3. **定正则**：用权重衰减压住参数量级，防过拟合。
> 一句话：**优化器 = 参数更新的策略**，它读梯度 → 算更新量 → 写回参数。区别 SGD/Momentum/Adam/AdamW，本质就是"算更新量"的配方不同（见下文各节）。损失下降、模型收敛，靠的就是这套策略迭代调参。

## 2. 为什么需要它（动机与背景）

[[SGD]]+[[Momentum]] 有两个痛点：

1. **所有参数共享一个 lr**：稀疏特征（embedding）和稠密权重梯度量级差几个数量级，单一 lr 要么稀疏维度学不动、要么稠密维度爆炸。
2. **L2 正则在 Adam 里失效**：Adam 把 $\lambda\theta$ 加进梯度后，会被二阶矩 $v_t$ 归一化，导致大权重和小权重的衰减强度错配，正则效果弱且不稳。

Adam 解决 1：用**梯度平方的指数平均 $v_t$** 逐参数缩放步长，梯度大的参数步长自动小、梯度小的自动大 → 对每个参数"自适应"。

AdamW 解决 2：把权重衰减**从 $g_t$ 里拿出来**，直接 $\theta\leftarrow(1-\eta\lambda)\theta$，绕开二阶矩归一化，正则稳定可控。

> [!note] 为什么 LLM 几乎只用 AdamW
> LLM 参数海量、梯度稀疏且量级差异大（embedding vs attention），需要自适应；又必须正则防过拟合/防参数膨胀，但 Adam+L2 不可控。AdamW 同时满足两者，加上 [[mixed precision training]]、[[学习率调度]] 配合，成为 GPT/Llama 系列默认优化器。

## 3. 核心概念详解

### 3.1 Adam 更新（完整）

维护一阶矩 $m$（动量）和二阶矩 $v$（梯度平方的指数平均）：

$$
g_t = \nabla_\theta L
$$
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \qquad (\text{一阶矩, 动量})
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad (\text{二阶矩, 逐元素平方})
$$
$$
\hat m_t = \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t} \qquad (\text{偏差校正})
$$
$$
\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

- $\beta_1=0.9$（动量衰减）、$\beta_2=0.999$（二阶矩衰减，慢）。
- $\epsilon\approx 10^{-8}$ 防分母为 0。
- 偏差校正：前期 $m,v$ 从 0 起步被低估，除以 $1-\beta^t$ 补偿，让前期更新不偏小。

### 3.2 自适应的直观

$\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\approx$ **符号**（sign）。因为 $\sqrt{v_t}\approx|g|$，所以 $\frac{m}{\sqrt v}\approx\frac{g}{|g|}=\text{sign}(g)$。所以 Adam 近似"**按梯度符号等步长**更新"，每个参数步长量级一致 ≈ $\eta$，不随梯度绝对值波动 → 对量级差异鲁棒。

### 3.3 AdamW 的解耦权重衰减

**Adam + L2（错）**：把 $\lambda\theta$ 加进梯度 $g_t$，再进 $v_t$ 归一化，衰减强度被 $v_t$ 干扰。

**AdamW（对）**：

$$
\theta_t \leftarrow (1-\eta\lambda)\theta_t \quad (\text{权重衰减, 独立于梯度})
$$
$$
\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\right)
$$

权重衰减直接乘在参数上，与自适应更新解耦，等价于"每步把参数往 0 缩 $(1-\eta\lambda)$"。LLM 的 weight_decay 通常 0.1（Llama）。

### 3.4 超参数典型值

| 参数 | CV | LLM (Llama/GPT) |
|---|---|---|
| `lr` | 1e-3 | 1e-4 ~ 3e-4（峰值） |
| `betas` | (0.9, 0.999) | (0.9, 0.95) 或 (0.9, 0.999) |
| `eps` | 1e-8 | 1e-8（bf16 训练有时 1e-7） |
| `weight_decay` | 1e-2 | 0.1 |
| `fused` | - | True（ fused kernel 加速） |

LLM 常把 $\beta_2$ 降到 0.95，让二阶矩更快跟上变化（梯度分布随训练变化大）。

## 4. 数学原理 / 公式

### 偏差校正为何必要

$m_t=\beta_1^t m_0+(1-\beta_1)\sum_{k=0}^{t-1}\beta_1^k g_{t-k}$，初始 $m_0=0$ 时 $\mathbb{E}[m_t]\approx(1-\beta_1^t)\mathbb{E}[g]$，被低估 $1-\beta_1^t$ 倍；除以它校正。否则前期 lr 有效偏小，warmup 可部分缓解。

### 不变性

Adam 对梯度**整体缩放**不变（$g\to cg$ 时 $\frac{m}{\sqrt v}\to\text{sign}(g)$ 不变），所以对 loss 缩放、loss scaling 鲁棒 —— 这是 [[loss scaling]] 能用的前提。

### 收敛性

凸问题有 $O(\sqrt{T})$ 收敛保证；非凸无保证，但实践中极稳。early training 的偏差校正 + warmup 是稳定的关键。

## 5. 代码示例

```python
import torch
import torch.nn as nn

model = nn.Linear(100, 2)

# Adam + L2（不推荐，正则错配）
opt_l2 = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)   # 这里是 L2

# AdamW（推荐，解耦权重衰减）
opt = torch.optim.AdamW(model.parameters(),
                        lr=3e-4, betas=(0.9, 0.999), weight_decay=0.1,
                        eps=1e-8, fused=True)   # fused 用融合 kernel 省显存提速

x, y = torch.randn(32, 100), torch.randn(32, 2)
loss_fn = nn.MSELoss()
for step in range(5):
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()                # 内部：更新 exp_avg(m), exp_avg_sq(v), step, 再 decay
    print(step, loss.item())

# 查看优化器状态（m / v）
state = opt.state[list(model.parameters())[0]]
print(state.keys())          # dict_keys(['step', 'exp_avg', 'exp_avg_sq'])
# 这正是 Adam 的 m、v，断点续训要随 state_dict 一起保存
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[SGD]]、[[Momentum]]（一阶矩即动量）、[[backward过程]]。
- **下游（应用）**: [[学习率调度]]（warmup+cosine 配 AdamW）、[[loss scaling]]（fp16 训练）、[[mixed precision training]]、[[数值类型与精度]]（bf16 下 eps/grad_norm 注意）、LLM 预训练/微调全流程。
- **对比 / 易混**（核心对比表）:

| 优化器 | 自适应(逐参数) | 动量 | 权重衰减 | 典型场景 |
|---|---|---|---|---|
| SGD | ❌ | 可选 | L2(=WD) | CV 分类泛化好 |
| Momentum | ❌ | ✅ | L2 | CV 加速 |
| RMSProp | ✅ | ❌ | - | RNN/非平稳 |
| **Adam** | ✅ | ✅ | L2(错配) | 通用，但 WD 不纯 |
| **AdamW** | ✅ | ✅ | **解耦WD** | **LLM 事实标准** |
| Lion | ✅(符号) | ✅ | 解耦 | 部分 LLM 替代 |

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **把 Adam 的 `weight_decay` 当 AdamW**：Adam 里是 L2，正则错配，要换 AdamW。
> 2. **不 warmup 直接大 lr**：AdamW 前期 $v_t$ 还小、偏差校正后步长大，易 loss spike → 必须 [[学习率调度]] warmup。
> 3. **不存优化器 state_dict**：$m,v$ 要随断点一起存，否则续训从 0 重启，前几千步不稳。
> 4. **$\beta_2$ 死守 0.999**：LLM 梯度分布变化大，0.95 更稳；要据任务调。
> 5. **`eps` 太小遇 bf16**：低精度下 $v$ 可能很小，eps 太小导致数值不稳，适当放大到 1e-7。
> 6. **以为 Adam 不用 `zero_grad`**：`.grad` 仍累加，照样要清。
> 7. **Adam 泛化差不知为何**：自适应让解偏尖锐极小值，泛化有时不如 SGD+momentum（CV 经典观察）。

## 8. 延伸细节

### 8.1 fused / foreach 实现

PyTorch 2.x 的 `fused=True` 用单 kernel 跑所有参数的 AdamW，减少 kernel launch 与显存读写，LLM 训练必开（省显存 + 提速）。

### 8.2 显存占用：Adam 是 3× 模型

Adam state 要存 $m$ 和 $v$，各与参数同大（fp32 下各 4 字节/参数）。加上参数本身 + 梯度，**fp32 训练 ≈ 12 字节/参数 = 3× 模型权重**。这是 [[ZeRO]]、[[FSDP]] 要分片优化的根因 —— 详见 [[optimizer state memory]]。

### 8.3 8-bit Adam /bnb

把 $m,v$ 量化到 8 bit，state 从 8 字节/参数降到 2 字节，省 75% 显存，质量损失小，LLM 微调（QLoRA）常用。

### 8.4 Lion / Sophia 等替代

Lion 用符号更新省 $v$（显存减半）；Sophia 用对角 Hessian 估计做曲率感知。Llama 系仍以 AdamW 为主，但 Lion 在部分场景替代。

### 8.5 与 PPO 的关系

RLHF 的 PPO 也用 AdamW 更新 policy/value，但因 on-policy + clip，lr 和 clip 参数要协同调，见 [[PPO]]。

---
相关: [[优化器]]、[[SGD]]、[[Momentum]]、[[学习率调度]]、[[梯度裁剪]]、[[loss scaling]]
