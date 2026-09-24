# forward与backward

> **所属章节**: [[PyTorch核心]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 基础（与 [[autograd机制]] 配对理解）

## 1. 一句话定义

- **forward（前向）**：输入 $x$ 经网络得到预测 $\hat y$ 与标量损失 $\mathcal{L}$，由用户在 `nn.Module.forward` 手写数据流。
- **backward（反向）**：从 $\mathcal{L}$ 出发，autograd 沿 [[计算图]] 反向链式求每个参数 $\theta$ 的梯度 $\partial\mathcal{L}/\partial\theta$ 填入 `theta.grad`，由 `loss.backward()` 触发。

前向建图、反向用图，靠张量的 `grad_fn` 反向指针串成闭环。数据流：

$$x \xrightarrow{f_\theta} \hat y \xrightarrow{\ell(\cdot,y)} \mathcal{L}\in\mathbb{R}$$

$\hat y$ 是张量（如 logits），不能直接 backward；$\mathcal{L}$ 是 $\hat y$ 经损失函数压成的标量，是 backward 的唯一入口（$\partial\mathcal{L}/\partial\mathcal{L}=1$）。

训练一步顺序固定：`zero_grad → forward → backward → step`。没 forward 就 backward 报 "element 0 does not require grad"；没 backward 就 step 是用 0 梯度更新（参数不动）。见 [[SGD]]/[[Adam与AdamW]]。

## 2. 为什么需要它

训练一句话：**用梯度下降最小化损失**。两步：
- forward：算出 $\mathcal{L}(\theta)$，知道"现在错多少"；
- backward：算 $\nabla_\theta\mathcal{L}$，知道"往哪挪、各挪多少"。

百亿参数下手动求偏导不可行。PyTorch 用**动态图 + autograd**：forward 自动记录算子与输入构造 DAG，backward 沿图反向链式、对每个算子用解析 backward 算局部梯度相乘累加。用户只写 forward，backward 免费。

## 3. 怎么工作

### 3.1 forward：建图

```python
x = torch.randn(8, requires_grad=True)
w = torch.randn(8, requires_grad=True)
y = x * w            # 节点 MulBackward0, 输入 x,w
loss = y.sum()       # 节点 SumBackward0, 输入 y
```

每个非叶子张量带 `grad_fn`；`requires_grad=True` 的叶子（用户建的 x/w 或 `nn.Parameter`）会累积 `.grad`。forward 同时算数值 + 建反向图。

### 3.2 backward：用图

```python
loss.backward()      # 反向遍历 grad_fn 链
# 之后 x.grad, w.grad 被填好
```

语义：从 $\partial\mathcal{L}/\partial\mathcal{L}=1$ 出发，沿 `loss.grad_fn` 反向链式，遇 `requires_grad=True` 的叶子就把梯度**累加**到 `leaf.grad`（累加≠覆盖，故先 `zero_grad`）。

> [!warning] backward 是"消耗图"
> 默认 `backward()` 调完释放反向图（动态图每步重建），一个 `loss` 不能 backward 两次（除非 `retain_graph=True`）。这是 PyTorch 与 TF 静态图的根本区别，见 [[计算图]]。

### 3.3 关键开关

| 项 | 作用 |
|---|---|
| `grad_fn` | 张量反向函数节点，反向图入口；叶子为 `None` |
| `requires_grad` | 是否求梯度；`nn.Parameter` 默认 True |
| `retain_graph=True` | 反向后保留中间图，可再次 backward（双 backward/Hessian） |
| `create_graph=True` | 反向本身建图，使梯度可求导（二阶，元学习/HVP） |
| `inputs=` | 只对指定叶子求梯度，省算力 |

### 3.4 backward 的入口维度

`loss.backward()` 要求 `loss` 标量。若 `y` 是向量要直接 `.backward()`，必须传 `gradient=torch.ones_like(y)`（JVP 的向量）。

### 3.5 模型的 forward / `__call__`

层重写 `forward`，调用写 `model(x)`：`nn.Module.__call__` 触发 [[hooks机制]] 后调 `forward`。backward 无需手写——只要 forward 用了可导算子，autograd 自动给所有 `Parameter` 填 `.grad`。所以**模型没有 `backward` 方法要重写**。

## 4. 数学原理

链式法则（标量对向量），设计算图 $x \xrightarrow{f_1} u \xrightarrow{f_2} v \xrightarrow{f_3} \mathcal{L}$：

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial v}\cdot\frac{\partial v}{\partial u}\cdot\frac{\partial u}{\partial x} = J_{f_3}^\top J_{f_2}^\top J_{f_1}^\top \cdot 1$$

autograd 本质是**反向模式自动微分（reverse-mode AD）**：从 $\partial\mathcal{L}/\partial\mathcal{L}=1$ 出发，对每算子用局部雅可比转置 $J_f^\top$ 左乘上游梯度逐节点传到叶子。一次反向得**全部参数**梯度，复杂度 $O(1)$ 倍 forward（输出少参数多时远优于前向模式或数值差分）。

局部梯度例：乘法 $y=xw$ → $\partial y/\partial x=w,\ \partial y/\partial w=x$；Sum → 全 1；ReLU → $\mathbb{1}[x>0]$（死区梯度 0 → ReLU 死神经元问题）。反向时 autograd 把"上游梯度×局部梯度"累加到下游 `.grad`。

## 5. 代码示例

最小片段（建图 → 填 grad → 标量入口）：

```python
x = torch.tensor([1.,2.,3.]); y = torch.tensor([2.,4.,6.])   # y=2x
w = torch.tensor(0.5, requires_grad=True)

y_hat = w * x                          # forward 建图 MulBackward0
loss = ((y_hat - y)**2).mean()         # forward 到标量 loss
loss.backward()                        # 填 w.grad
with torch.no_grad():                  # 更新关图，否则被记进图
    w -= 0.1 * w.grad

# y_hat.backward()  # 报错：向量非标量
y_hat.backward(gradient=torch.ones_like(y_hat))  # 等价 sum(y_hat).backward()
print(w.is_leaf, w.grad_fn)             # True, None
```

模型写法等价循环：`opt.zero_grad() → logits=model(batch) → loss=loss_fn(logits,labels) → loss.backward() → opt.step()`。这就是 [[nn.Module]]+[[forward与backward]]+[[optimizer state管理]] 标准三段式。

## 6. 与其他知识点的关系

- **上游**: [[nn.Module]]、[[autograd机制]]、[[计算图]]、[[梯度与Jacobian]]
- **下游**: [[optimizer state管理]]、[[hooks机制]]、[[SGD]]/[[Adam与AdamW]]、[[DDP]]（反向同步梯度）、[[FSDP]]（反向分片 all-gather）
- **易混**: forward 用户写、backward autograd 自动跑，不混、不需手写模型 backward

## 7. 常见误区与易错点

1. 用 `model.forward(x)` 而非 `model(x)` → 跳过 forward hooks。
2. 更新参数不开 `torch.no_grad()` → `w -= lr*w.grad` 被记进图，图变乱、显存涨。
3. 忘 `zero_grad` → `.grad` 累加，梯度越滚越大，参数飞掉。
4. 向量 loss 直接 `.backward()` → 报错；`.sum()`/`.mean()` 或传 `gradient=`。
5. `loss.backward()` 两次 → 默认图已释放报错，要 `retain_graph=True`。
6. `requires_grad=False` 的张量指望有 `.grad` → 没有；冻结不学。推理用 `torch.no_grad()`/`inference_mode()` 关图省显存。
7. `loss.backward()` 不更新参数，只填 `.grad`，要 `opt.step()`。
8. `detach()` 切断反向图，下游不传梯度给 x；用于固定 target、stop-gradient（DPO/SimCLR 常见）。

## 8. 延伸

- `torch.no_grad()`：上下文内不建图省显存，推理/参数更新用；`inference_mode()` 更快更省但产出张量不能进后续需 grad 图；`enable_grad()` 在 no_grad 区局部开梯度。
- 二阶导（HVP/MAML/gradient penalty）：`loss.backward(create_graph=True)` 让梯度本身可导，再 `second.backward()`，代价是显存显著上升。
- CUDA Graph：把前向+反向的 GPU kernel 序列捕获成静态图重放，跳过 Python 调度；与 autograd 计算图不是同一概念，见 [[计算图]]。
- PPO loss = policy loss + value loss + entropy，三段可导共用一次 `loss.backward()`；clip 处靠 mask 处理，`torch.where` 不会自动断梯度需 `detach`/`clamp`，见 [[PPO clipped objective]]。

> [!note] 解答（批注）：训推一致性为何只比 forward 的 logp
> 训推一致性 = 同模型同输入，训练框架（Megatron/DeepSpeed）与推理框架（vLLM/TRT-LLM）算出的 logp 数值一致（误差 < 阈值），RLHF/PPO 的 reward、advantage、importance sampling 都依赖它。
> - **比 logp 而非 logits/loss**：logits 词表大范围宽逐点噪声大且推理常 kernel 融合不暴露；loss 把 target 混进去更间接；logp = `log_softmax(logits)[token]` 单 token 一标量逐点可比，是 PPO 直接消费的 $\log\pi_\theta(a\mid s)$，log_softmax 数值稳定。
> - **为何不管 backward**：forward 一致 + 链式法则 + 局部雅可比数学确定 → backward 必然一致（充分条件）；且 backward 校验要构造 loss、跑反向、逐参数比 `.grad`，大模型上 grad 巨大昂贵。只有改了反向 kernel 才需额外验。
> - **指标**：逐 token 取训练 logp $a$ 与推理 logp $b$ 算 Pearson $r\in[-1,1]$（衡量共线，对 scale/常数偏移不敏感，贴合 advantage 只看相对序），但 Pearson 高绝对差也可能大，故**双指标**：`Pearson>0.99` 且 `max_abs_diff<1e-3`（[待核实] 阈值为常见工程经验值，非硬标准）。不一致来源：bf16/fp8 舍入路径、RMSNorm eps、RoPE/KV cache 量化、kernel 融合改累加顺序。见 [[数值类型与精度]]、[[mixed precision training]]、[[logits generation]]、[[KL散度]]、[[PPO]]。

---
相关: [[PyTorch核心]]、[[autograd机制]]、[[计算图]]、[[nn.Module]]、[[optimizer state管理]]
