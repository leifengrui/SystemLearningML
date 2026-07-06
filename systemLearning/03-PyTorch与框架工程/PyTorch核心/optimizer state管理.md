# optimizer state管理

> **所属章节**: [[PyTorch核心]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 中等（断点续训、显存账必懂）

## 1. 一句话定义

**优化器状态（optimizer state）** 是除了模型参数 $\theta$ 与梯度 $g$ 之外、优化器**自己存起来跨 step 复用的中间量**——SGD 无（或仅 momentum buffer），AdamW 为每个参数存**一阶矩 $m$ 与二阶矩 $v$** 两个同形状张量。这些状态随参数一起被 `optimizer.state` 字典管理，是**断点续训必须保存、显存账必须计入**的关键对象。

> [!note] 三类"模型相关张量"的归属
> 训练显存里有三大块张量：**参数 P**（`Parameter`，进 `state_dict`）、**梯度 P**（`Parameter.grad`，backward 填）、**优化器状态**（`optimizer.state[param]`，step 时维护）。三者往往被混称"模型状态"，但显存账和续训处理截然不同——见第 8 节的"4P 显存账"。

> [!note] 解答：什么是优化器？到底在优化什么？
> 
> **什么是优化器（Optimizer）**：训练中**把梯度变成参数更新**的算法组件。它接收梯度 $g$、维护跨 step 的状态（momentum / Adam 的 $m,v$）、按规则算出更新量 $\Delta\theta$ 并改进 $\theta$。PyTorch 里是 `torch.optim.Optimizer` 的子类（[[SGD]] / [[Adam与AdamW]]）。它不"定义目标"——目标（损失）由 loss function 给；优化器只管"怎么用梯度去挪参数"。
> 
> **在优化什么？** = 优化**模型参数 $\theta$**（权重 $W$、bias、LayerNorm 的 $\gamma/\beta$ 等 `nn.Parameter`），让损失 $\mathcal{L}(\theta)$ 下降。不是优化数据、不是优化 loss 函数本身（loss 函数是固定的），而是**沿 $\nabla_\theta\mathcal{L}$ 的反方向迭代调整 $\theta$**，找让 loss 最小的 $\theta^*$：
> 
> $$\theta^*=\arg\min_\theta \mathcal{L}(\theta),\quad \theta_{t+1}=\theta_t-\eta\cdot\text{update}_t\quad(\text{update}_t\text{ 由优化器算})$$
> 
> **三类对象的分工**（别混）：
> 
> | 对象 | 是什么 | 谁管 |
> |---|---|---|
> | 参数 $\theta$ | 被优化的对象（权重） | `nn.Module` 持有，进 `state_dict` |
> | 梯度 $g=\nabla_\theta\mathcal{L}$ | 告诉参数该往哪挪 | backward 填进 `theta.grad` |
> | 优化器状态 | 帮优化器用历史梯度算更好更新量 | `optimizer.state`，见本笔记 |
> 
> **一句话**：优化器 = 用梯度（+历史状态）算更新量、迭代改进参数 $\theta$ 以最小化 loss 的算法。本笔记讲的就是它**跨 step 持有的状态**（momentum / Adam 矩）。另见 [[SGD]]、[[Adam与AdamW]] 的"优化器优化了什么"。
> 
> 相关：[[SGD]]、[[Adam与AdamW]]、[[forward与backward]]（backward 产 grad，optimizer 消费）、[[state_dict与load_state_dict]]（续训要存 `optimizer.state_dict()`）。
## 2. 为什么需要它（动机与背景）

朴素 SGD 只需当前梯度 $g_t$ 就能更新 $\theta_{t+1}=\theta_t-\eta g_t$，没有跨 step 信息。但为了**收敛更快、更稳**，现代优化器要利用历史梯度：

- **Momentum**：存指数滑动平均 $\bar g_t=\beta\bar g_{t-1}+g_t$，跨 step 复用 → 需要 1 个 buffer。
- **Adam/AdamW**：同时存一阶矩 $m_t$（梯度的 EMA）和二阶矩 $v_t$（梯度平方的 EMA）→ 需要 2 个同形状 buffer。

这些 buffer 不能每步重建（依赖历史），必须由优化器**长期持有**。于是 `torch.optim.Optimizer` 用 `self.state`（`dict{Parameter: dict}`）来存。这也带来两大工程后果：

1. **显存**：AdamW 的优化器状态 = $2\times P$（fp32），是 LLM 训练显存大头，催生 [[FSDP]]/[[ZeRO]] 分片；
2. **续训**：只恢复 `model.state_dict()` 不够，Adam 的 $m$/$v$ 丢了 → 优化器"失忆"，损失曲线跳、要重新 warmup，必须同时存 `optimizer.state_dict()`。

## 3. 核心概念详解

### 3.1 `optimizer.state` 的结构

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
# 训练若干 step 后：
opt.state        # defaultdict, 键是 Parameter 对象，值是 dict
# opt.state[param] == {"step": tensor(100), "exp_avg": m, "exp_avg_sq": v}
```

- **键**：Parameter 对象本身（身份引用，不是名字）。
- **值**：dict，键名是优化器自定义（AdamW 用 `exp_avg`/`exp_avg_sq`，SGD 用 `momentum_buffer`）。
- **`step`**：全局步数计数器（用于 bias correction、LR schedule），通常存成 0-d tensor。

### 3.2 `optimizer.state_dict()` 的扁平化

因为 `state` 以 Parameter 对象为键，无法直接序列化，`state_dict()` 把它转成**按参数顺序索引**的形式：

```python
{
  "state":    {0: {"step":..., "exp_avg":...}, 1: {...}, ...},
  "param_groups": [{"lr":1e-3, "betas":(0.9,0.999), "weight_decay":0.01, "params":[0,1,...]}],
}
```

- `param_groups` 描述超参数（lr、betas、weight_decay、参数分组）。
- `state` 用**整数索引**对应 `param_groups[0]["params"]` 的顺序。
- 这意味着**参数顺序变了，加载就错位**——续训必须保证 `model.parameters()` 顺序与保存时一致（同一份代码、同一实例化顺序通常即可）。

### 3.3 不同优化器的状态量

| 优化器 | 每参数状态张量数 | 状态量（相对参数 P） | 说明 |
|---|---|---|---|
| SGD（无 momentum） | 0 | 0 | 无状态 |
| SGD+momentum | 1 | 1P | momentum buffer |
| Adam / AdamW | 2 | 2P | exp_avg + exp_avg_sq |
| Adafactor | ~0.5 | <1P | 因式分解的二阶矩，省显存 |
| Lion | 1 | 1P | 只存一阶的符号优化 |

> [!tip] LLM 训练显存账（"4P"经验公式）
> 混合精度训练（bf16 参数+bf16 梯度+fp32 master+fp32 AdamW 状态）大致：
> - 参数 bf16: $2P$（或 fp32 master: $4P$）
> - 梯度 bf16: $2P$
> - AdamW 状态 fp32: $8P$（$m$ $4P$ + $v$ $4P$）
> - 合计 ≈ $12P$ 左右（含 master 与激活则更多）。优化器状态占大头 → [[FSDP]]/[[ZeRO]] 把它分片到多卡。详见 [[显存结构]]/[[ZeRO]]。

### 3.4 `param_groups` 与分组

一个优化器可有多组参数，每组独立超参（如 backbone 用 lr=1e-5、head 用 lr=1e-3）：

```python
opt = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5, "weight_decay": 0.1},
    {"params": model.head.parameters(),     "lr": 1e-3, "weight_decay": 0.0},
], lr=1e-3)
```

`param_groups` 也进 `optimizer.state_dict()`，续训时一并恢复——所以**改了分组结构再加载旧 ckpt 会不兼容**。

### 3.5 LR 调度器与 step 的关系

`scheduler.step()` 修改 `optimizer.param_groups[i]["lr"]`；`optimizer.state[...]["step"]` 记录调用次数。续训要同时存调度器 state 与全局 step，否则 LR 曲线重置。

### 3.6 梯度累加与 zero_grad

`optimizer.zero_grad(set_to_none=True)` 把 `param.grad` 置 `None`（比置 0 省、且 autograd 在下次 backward 时按 None 重新分配）；`step()` 读 `param.grad` 更新参数与状态。梯度累积（见 [[batch与mini-batch]]）期间不 `step`，只累积 `.grad`，累积够再 `step` 后才 `zero_grad`。

## 4. 数学原理 / 公式

### AdamW 单步更新（带状态维护）

记 $g_t=\nabla_\theta\mathcal{L}$，$\beta_1,\beta_2$ 为衰减率，$\epsilon$ 防除零：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t && \text{(exp_avg, 一阶矩 EMA)}\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 && \text{(exp_avg_sq, 二阶矩 EMA)}\\
\hat m_t &= m_t/(1-\beta_1^t),\quad \hat v_t = v_t/(1-\beta_2^t) && \text{(bias correction，用 step } t)\\
\theta_t &\leftarrow \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} - \eta\lambda\,\theta_{t-1} && \text{(AdamW: 解耦权重衰减)}
\end{aligned}
$$

$m_t$、$v_t$ 就是 `optimizer.state[param]["exp_avg"]` 与 `["exp_avg_sq"]`，每个参数两份同形状 fp32 张量；$t$ 即 `["step"]`。续训丢掉 $m,v$ 等于把 $\hat m,\hat v$ 重置为 0，前几百步的"梯度趋势记忆"全没。

### SGD+momentum

$$
\bar g_t = \beta\bar g_{t-1}+g_t,\quad \theta_t\leftarrow\theta_{t-1}-\eta\bar g_t
$$

只需 1 个 buffer（`momentum_buffer`）。

## 5. 代码示例

```python
import torch, torch.nn as nn

model = nn.Linear(8, 8)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.999), weight_decay=0.1)

x = torch.randn(32, 8); y = torch.randn(32, 8)
loss_fn = nn.MSELoss()

# 1) 训练几步，观察 state 长出
for _ in range(3):
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()

p = next(model.parameters())
print("state keys:", list(opt.state[p].keys()))   # ['step','exp_avg','exp_avg_sq']
print("exp_avg shape:", opt.state[p]["exp_avg"].shape, "step:", opt.state[p]["step"])

# 2) 断点续训：模型+优化器一起存
ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict()}
torch.save(ckpt, "opt.pt")

# 恢复（注意顺序）
m2 = nn.Linear(8,8)
m2.load_state_dict(ckpt["model"])                 # 先模型，参数对象就位
opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
opt2.load_state_dict(ckpt["optimizer"])           # 再优化器，按参数顺序对齐
# 之后 m2,opt2 与 m,opt 状态完全一致，可继续 step

# 3) 分组参数（不同 lr/weight_decay）
backbone = nn.Linear(8,8); head = nn.Linear(8,8)
opt3 = torch.optim.AdamW([
    {"params": backbone.parameters(), "lr":1e-5, "weight_decay":0.1},
    {"params": head.parameters(),     "lr":1e-3, "weight_decay":0.0},
])
print("groups:", [(g["lr"], g["weight_decay"]) for g in opt3.param_groups])
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[nn.Module]]（参数来源）、[[forward与backward]]（`.grad` 由 backward 产生）、[[SGD]]/[[Adam与AdamW]]（具体优化器算法）。
- **下游（应用）**: [[state_dict与load_state_dict]]（续训要存 `optimizer.state_dict()`）、[[FSDP]]/[[ZeRO]]（把 AdamW 的 $m,v$ 分片省显存）、[[学习率调度]]（scheduler 写 `param_groups`）、[[显存结构]]（optimizer state 是显存大头）。
- **对比 / 易混**:
  - `optimizer.state`（运行时 dict，键=Parameter）vs `optimizer.state_dict()`（可序列化，键=整数索引）。
  - `model.state_dict()` vs `optimizer.state_dict()`：权重 vs 优化器动量，**续训都要存**。
  - 梯度 `param.grad`（每步重建）vs 优化器状态 $m,v$（跨 step 复用）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **续训只存 model 不存 optimizer** → Adam 的 $m,v$ 丢，相当于冷启动，loss 跳、要重新 warmup。
> 2. **加载顺序反了**（先 `optimizer.load_state_dict` 后 `model.load_state_dict`）→ 优化器 state 按"旧参数对象"注册，与新模型参数对象身份不一致，状态对不上。
> 3. **改了 `model.parameters()` 顺序再加载** → `state_dict` 整数索引错位，$m,v$ 灌到错参数。
> 4. **以为 `zero_grad` 会清优化器状态** → 不会，它只清 `param.grad`；$m,v$ 是长期复用的。
> 5. **新增层后用旧 optimizer ckpt** → 新参数没有对应 state，`load_state_dict` 会静默忽略（这些参数按"无状态"从 0 步开始，bias correction 退化）。
> 6. **显存估算漏算 optimizer state** → 以为 fp32 模型只要 $4P$，实际 AdamW 训练要 $4P+4P+8P+\cdots$，常被低估一个量级。
> 7. **`set_to_none=True` vs `False`**：True 省显存且下次 backward 重新分配；False 置 0 保留张量。现代默认 True。
> 8. **梯度累积后忘记 `zero_grad`** → 下一批梯度叠到旧的，等效 batch 翻倍、loss 翻倍。

## 8. 延伸细节

### 8.1 4P / 12P 显存账详表（bf16 混合精度 + AdamW）

| 项 | dtype | 量 |
|---|---|---|
| 参数（master） | fp32 | $4P$ |
| 参数（bf16 副本，前向用） | bf16 | $2P$ |
| 梯度（bf16） | bf16 | $2P$ |
| Adam $m$ | fp32 | $4P$ |
| Adam $v$ | fp32 | $4P$ |
| 激活 | bf16 | 与序列长度/层数相关 |

合计 ≈ $16P$ + 激活。优化器状态（$8P$）几乎占一半，这是 [[FSDP]] 优先分片 optimizer state 的原因（对应 ZeRO-1）。

### 8.2 ZeRO 三级分片

- **ZeRO-1**：只分片 optimizer state → 显存 $4P+4P+\frac{8P}{N}$。
- **ZeRO-2**：再分片梯度 → $4P+\frac{4P+8P}{N}$。
- **ZeRO-3**（≈FSDP）：参数也分片 → $\frac{4P+4P+8P}{N}$。

详见 [[ZeRO]]/[[FSDP]]。

### 8.3 `foreach` / `fused` 优化器

PyTorch 2.x 的 `foreach=True` 把多参数同操作合成一个大 kernel，`fused=True`（CUDA）写成一个融合 kernel，显著降低 optimizer step 的小 kernel launch 开销（LLM 训练能省可观时间）。`torch.optim.AdamW(..., foreach=True)`。

### 8.4 `optimizer.state` 与分布式同步

DDP 下梯度已 all-reduce 同步，每 rank 的 `param.grad` 相同，故每 rank 维护的 $m,v$ 也天然相同 → 无需额外同步优化器状态。FSDP 下参数分片，每 rank 只对自己分片参数维护 state，天然分片，无需同步。

---
相关: [[PyTorch核心]]、[[Adam与AdamW]]、[[state_dict与load_state_dict]]、[[FSDP]]、[[ZeRO]]、[[显存结构]]、[[forward与backward]]
