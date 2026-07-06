# forward与backward

> **所属章节**: [[PyTorch核心]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 基础（与 [[autograd机制]] 配对理解）

## 1. 一句话定义

**forward（前向）** 是输入 $x$ 经网络得到输出 $\hat y$ 与损失 $\mathcal{L}$ 的过程，由用户在 `nn.Module.forward` 里手写数据流；**backward（反向）** 是从 $\mathcal{L}$ 出发、用 autograd 沿 [[计算图]] 反向链式求每个参数 $\theta$ 的梯度 $\partial\mathcal{L}/\partial\theta$、填入 `theta.grad` 的过程，由 `loss.backward()` 触发。前向建图、反向用图，二者通过 **`grad_fn`** 这条反向指针串成 [[autograd机制]] 的闭环。

> [!note] 解答：输出 $\hat y$ 与损失 $\mathcal{L}$ 是什么？
> 
> - **$\hat y$（读作 y-hat）** = 模型的**预测输出**（prediction）。帽子 `^` 在统计里约定表示"估计值/预测值"。例：线性回归 $\hat y = wx+b$ 是对真值 $y$ 的预测；LLM 里 $\hat y$ 通常是 logits（词表得分向量）或采样出的 token。它是 forward 算出来的"模型给出的答案"。
> - **$\mathcal{L}$（花体 L）= 损失**（loss），一个**标量**，衡量 $\hat y$ 与真值 $y$ 差多少。例：MSE 回归 $\mathcal{L}=\frac{1}{n}\sum(\hat y-y)^2$；分类交叉熵 $\mathcal{L}=-\log \hat p_{y_{\text{真}}}$；LLM 的 next-token loss $\mathcal{L}=-\frac{1}{T}\sum_t \log p_\theta(x_t\mid x_{<t})$。
> 
> **forward 的数据流**：
> 
> $$x \xrightarrow{f_\theta\ \text{(模型)}} \hat y \xrightarrow{\ell(\cdot,\,y)\ \text{(损失函数)}} \mathcal{L}\in\mathbb{R}$$
> 
> - $x$：输入（特征/上一 token）；
> - $f_\theta$：模型（参数 $\theta$）；
> - $\hat y$：预测；
> - $\ell$：损失函数，把"预测 vs 真值"压成一个标量；
> - $\mathcal{L}$：最终 loss，backward 的起点（$\partial\mathcal{L}/\partial\mathcal{L}=1$）。
> 
> **为什么要分 $\hat y$ 和 $\mathcal{L}$**：$\hat y$ 是**向量/张量**（如 logits），不能直接 backward（backward 要标量入口，见 [[#3.4 backward 的入口维度]]）；$\mathcal{L}$ 是 $\hat y$ 经损失函数压成的**标量**，是 backward 的唯一入口。所以 forward 必须走到 $\mathcal{L}$ 才能开始 backward。相关：[[损失函数分类]]、[[logits generation]]、[[autograd机制]]。

> [!note] 两阶段命名
> 训练一步 = `zero_grad → forward → backward → step`。其中 forward 对应"算预测+算 loss"，backward 对应"算梯度"。优化器 `step` 用梯度更新参数，见 [[SGD]]/[[Adam与AdamW]]。三者顺序不能乱：**没有 forward 建图就 backward 会报 "element 0 does not require grad"**；**没有 backward 就 step 会更新 0 梯度**（参数不动）。

## 2. 为什么需要它（动机与背景）

深度学习的全部训练可压缩成一句话：**用梯度下降最小化损失**。于是必须有两步：

- **forward**：把当前参数 $\theta$ 与样本 $x$ 算出 $\mathcal{L}(\theta)$，知道"现在错多少"；
- **backward**：算 $\nabla_\theta\mathcal{L}$，知道"该往哪挪 $\theta$、各挪多少"。

手动对每个参数求偏导在百亿参数的 LLM 上不可行。PyTorch 的解法是**动态图 + autograd**：forward 时自动记录每个算子及其输入，构造一张 DAG（有向无环图，节点=张量，边=算子）；backward 时沿这张图**反向链式**，对每个算子用其解析的 `backward` 公式算局部梯度并相乘累加。用户只写 forward，backward 免费。


> [!note] 解答：训推一致性——为什么比对 forward 的 logp？为什么不管 backward？Pearson 系数怎么用？
> 
> **训推一致性（training-inference parity）**：确保同一模型、同一输入，在**训练框架**（Megatron/DeepSpeed，bf16/fp8+张量并行）算出的 logp 与**推理框架**（vLLM/TensorRT-LLM，bf16/fp8+kernel 融合+KV cache）算出的 logp **数值一致**（误差 < 阈值）。RLHF/PPO 里 reward、advantage、importance sampling 都依赖 logp，训推不一致会让策略更新方向跑偏。
> 
> **为什么比 logp（对数概率）而不是 logits / loss**：
> 
> | 比什么 | 为什么 logp 最合适 |
> |---|---|
> | raw logits | 词表大（V~10^5）、数值范围宽，逐元素比对噪声大；且推理常做 kernel 融合（RMSNorm+QKV 融合），中间 logits 不一定暴露 |
> | loss | loss = -logp 的平均，已把 target/shift 混进去，比 logp 更间接 |
> | **logp = log_softmax(logits)[token]** | 单 token 一个标量，逐点可比；是 PPO 直接消费的量（$\log\pi_\theta(a\mid s)$）；log_softmax 数值稳定（减最大值），bf16 误差小 |
> 
> **为什么不管 backward**：
> 1. **forward 一致是 backward 一致的充分条件**：backward = autograd 沿前向图反向链式。如果前向每个算子的**数值**一致，且两端反向实现遵循同一套链式法则+局部雅可比（PyTorch autograd 与手写 kernel 的 backward 都是数学确定的），那梯度必然一致——不需要单独校验 backward。
> 2. **校验成本**：forward 跑一遍即可；校验 backward 要构造 loss、跑反向、再比对每个 Parameter 的 `.grad`，慢且大模型上 grad 张量巨大、逐参数比对昂贵。
> 3. **实务经验**：forward logp 一致基本就保证训练正确；只有改了反向 kernel（如自定义 backward、fp8 反向）才需额外验 backward。
> 
> **Pearson 相关系数怎么用**：逐 token 取训练 logp 序列 $a=(a_1,\dots,a_T)$ 与推理 logp 序列 $b=(b_1,\dots,b_T)$，算
> 
> $$r = \frac{\sum_t (a_t-\bar a)(b_t-\bar b)}{\sqrt{\sum_t(a_t-\bar a)^2}\sqrt{\sum_t(b_t-\bar b)^2}} \in [-1,1]$$
> 
> - $r\to 1$：两组数**线性共线**（形状一致），训推对齐。
> - 为什么用 Pearson：衡量"是否共线"，对整体 scale / 常数偏移不敏感——RLHF 里 advantage 只看相对序，Pearson 贴合需求。
> - 局限：Pearson 高但绝对差大也有问题（如推理整体偏移一个常数）。所以**通常同时看两个指标**：`Pearson > 0.99` **且** `max_abs_diff < 1e-3`（或 `mean_abs_diff < 1e-4`）。前者保形状，后者保绝对。
> 
> **常见不一致来源**：bf16/fp8 舍入路径不同、RMSNorm epsilon 不同、RoPE 实现差异、KV cache 量化、kernel 融合改累加顺序、位置编码 base 不同。排查时逐层 logp diff 定位。见 [[数值类型与精度]]、[[mixed precision training]]、[[logits generation]]、[[KL散度]]（KL 也是一致性度量）、[[PPO]]。
## 3. 核心概念详解

### 3.1 forward：建图

```python
x = torch.randn(8, requires_grad=True)
w = torch.randn(8, requires_grad=True)
y = x * w            # 节点: MulBackward0, 输入 x,w
loss = y.sum()       # 节点: SumBackward0, 输入 y
```

每个**非叶子张量**都带一个 `grad_fn`（指向产生它的算子的反向函数）；`requires_grad=True` 的**叶子张量**（用户直接建的 x、w，或 `nn.Parameter`）会积累 `.grad`。

forward 同时做了两件事：算数值（`y`、`loss` 的具体值）+ 建反向图（`grad_fn` 链）。

### 3.2 backward：用图

```python
loss.backward()      # 从 loss 出发反向遍历 grad_fn 链
# 之后: x.grad, w.grad 被填好
```

`loss.backward()` 的语义：

1. 初始 $\frac{\partial \mathcal{L}}{\partial \mathcal{L}}=1$；
2. 沿 `loss.grad_fn` 反向，按链式法则算每个上游节点的梯度；
3. 遇到 `requires_grad=True` 的叶子张量 → 把梯度累加到 `leaf.grad`（**累加不是覆盖**，所以要先 `zero_grad`）。

> [!warning] backward 是"消耗图"
> 默认 `backward()` 调完会**释放反向图**（PyTorch 动态图特性：每 step 重建）。所以一个 `loss` 不能 `backward` 两次（除非 `retain_graph=True`）。这是 PyTorch 与 TensorFlow 静态图的根本区别，见 [[计算图]]。

### 3.3 `grad_fn`、`requires_grad`、`retain_graph`、`create_graph`

| 项 | 作用 |
|---|---|
| `grad_fn` | 张量的反向函数节点，反向图入口。叶子张量为 `None`。 |
| `requires_grad` | 是否需要对该张量求梯度。`nn.Parameter` 默认 `True`。 |
| `retain_graph=True` | 反向后**保留中间图**，以便再次 backward（双 backward / Hessian 时用）。默认 False，省显存。 |
| `create_graph=True` | 反向时**本身建图**，使梯度也可被求导（二阶）。常用于元学习、Hessian-vector product。 |
| `inputs=` | 只对指定叶子求梯度，其它不填，省算力。 |

### 3.4 backward 的入口维度

`loss.backward()` 要求 `loss` 是**标量**（标量对向量求导才有定义）。若 `y` 是向量要直接 `.backward()`，必须传 `gradient=torch.ones_like(y)`（即雅可比-向量积 JVP 的向量）。

### 3.5 模型的 forward / backward 与 `__call__`

模型层重写的是 `forward`，但调用应写 `model(x)`：`nn.Module.__call__` 内部会触发 [[hooks机制]] 并最终调 `forward`。backward 不需要用户写——只要 forward 用了可导算子，autograd 自动给所有 `Parameter` 填 `.grad`。这也解释了为何**模型里没有 `backward` 方法要重写**（早期框架要，PyTorch 不用）。

## 4. 数学原理 / 公式

### 链式法则（标量对向量）

设计算图 $x \xrightarrow{f_1} u \xrightarrow{f_2} v \xrightarrow{f_3} \mathcal{L}$，则：

$$
\frac{\partial \mathcal{L}}{\partial x}
= \frac{\partial \mathcal{L}}{\partial v}\cdot\frac{\partial v}{\partial u}\cdot\frac{\partial u}{\partial x}
= J_{f_3}(v)^\top \cdot J_{f_2}(u)^\top \cdot J_{f_1}(x)^\top \cdot 1
$$

autograd 的本质：**反向模式自动微分（reverse-mode AD）**——从 $\partial\mathcal{L}/\partial\mathcal{L}=1$ 出发，对每个算子 $f$ 用其"局部雅可比转置"$J_f^\top$ 左乘当前上游梯度，逐节点传到叶子。一次反向即可得到**全部参数**的梯度，复杂度 $O(1)$ 倍 forward（对深度网络远优于数值差分或前向模式）。

### 局部梯度的例子

- 乘法 $y=x\cdot w$：$\partial y/\partial x = w$，$\partial y/\partial w = x$；
- 加法 $y=x+w$：两条都是 $1$；
- Sum $y=\sum_i x_i$：$\partial y/\partial x_i=1$；
- ReLU $y=\max(0,x)$：$\partial y/\partial x = \mathbb{1}[x>0]$（死区梯度为 0 → ReLU 死神经元问题）。

反向时 autograd 把"上游梯度 × 局部梯度"累加到下游节点的 `.grad`。

## 5. 代码示例

```python
import torch

# 1) 手动线性回归：演示 forward 建图 + backward 填 grad
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 4.0, 6.0])           # 真值 y = 2x
w = torch.tensor(0.5, requires_grad=True)   # 叶子参数

lr = 0.1
for step in range(50):
    if w.grad is not None:
        w.grad.zero_()                      # 等价于 optimizer.zero_grad()
    y_hat = w * x                           # forward：建图 MulBackward0
    loss = ((y_hat - y) ** 2).mean()        # forward：建图到 loss 标量
    loss.backward()                         # backward：填 w.grad
    with torch.no_grad():                   # 更新要关图，否则被记进图
        w -= lr * w.grad
    if step % 10 == 0:
        print(step, float(loss), float(w))  # w 逐步逼近 2.0

# 2) 标量检查：backward 只接受标量 loss
try:
    y_hat.backward()                        # y_hat 是向量 -> 报错
except RuntimeError as e:
    print(e)  # grad can be implicitly created only for scalar outputs

# 3) 向量 backward 需要 gradient 参数
g = torch.ones_like(y_hat)
y_hat.backward(gradient=g)                 # 等价于求 sum(y_hat) 的梯度
print("w.grad after vector backward:", w.grad)

# 4) 演示 grad_fn / 叶子节点
print("w.is_leaf:", w.is_leaf, " w.grad_fn:", w.grad_fn)          # True, None
print("y_hat.is_leaf:", y_hat.is_leaf, " y_hat.grad_fn:", y_hat.grad_fn)  # False, MulBackward0
```

> [!note] 模型写法的等价训练循环
> ```python
> opt.zero_grad(set_to_none=True)
> logits = model(batch)            # forward：内部 __call__ -> hooks -> forward
> loss = loss_fn(logits, labels)
> loss.backward()                  # backward：自动填所有 parameter.grad
> opt.step()                       # 用 .grad 更新 parameter
> ```
> 这就是 [[nn.Module]] + [[forward与backward]] + [[optimizer state管理]] 的标准三段式。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[nn.Module]]（forward 在 Module 里定义）、[[autograd机制]]（backward 由它实现）、[[计算图]]（forward 建图、backward 用图）、[[梯度与Jacobian]]（backward 输出即梯度）。
- **下游（应用）**: [[optimizer state管理]]（backward 产 `.grad`，优化器消费它）、[[hooks机制]]（forward/backward hook 挂在这两阶段）、[[SGD]]/[[Adam与AdamW]]（`step` 依赖 backward 的产物）、[[DDP]]（反向时同步梯度）、[[FSDP]]（反向时分片 all-gather）。
- **对比 / 易混**: `forward` 是用户写的、`backward` 是 autograd 自动跑的；二者不可混淆，也不需手写模型 backward。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **用 `model.forward(x)` 而非 `model(x)`** → 跳过 forward hooks；养成 `model(x)` 习惯。
> 2. **更新参数时不开 `torch.no_grad()`** → `w -= lr*w.grad` 会被记进图，图越来越乱、显存涨、报错。
> 3. **忘了 `zero_grad`** → `.grad` 累加，梯度越滚越大，参数飞掉。
> 4. **向量 loss 直接 `.backward()`** → 报错；要么 `.sum()`/`.mean()` 成标量，要么传 `gradient=`。
> 5. **以为 `loss.backward()` 之后再 `loss.backward()` 还能跑** → 默认图已释放，报 "Trying to backward through the graph a second time"；要 `retain_graph=True`。
> 6. **`requires_grad=False` 的张量指望有 `.grad`** → 没有；冻结参数不学。推理时用 `torch.no_grad()` 或 `torch.inference_mode()` 既关图又省显存。
> 7. **认为 `loss.backward()` 会更新参数** → 不会，它只填 `.grad`；要 `opt.step()` 才更新。
> 8. **detach 误用**：`y = x.detach()` 切断反向图，下游不传梯度给 x；用于固定输入（如 target、stop-gradient，DPO/SimCLR 常见）。

## 8. 延伸细节

### 8.1 `torch.no_grad` / `inference_mode` / `enable_grad`

- `torch.no_grad()`：上下文内 forward 不建图、不存 `grad_fn`，省显存加速，用于推理/参数更新。
- `torch.inference_mode()`（更新版）：比 `no_grad` 更快更省，但产出的张量不能用于后续需 grad 的图。
- `torch.enable_grad()`：在 `no_grad` 区内局部打开梯度（如想在推理中算某段梯度）。

### 8.2 `retain_graph` 与双 backward

求二阶导（HVP、元学习 MAML、gradient penalty）需要先 `loss.backward(create_graph=True)`——让梯度本身也是一张可导图，再对它 `second.backward()`。代价：显存显著上升。

### 8.3 反向模式 vs 前向模式 AD

- **反向模式**（PyTorch/TensorFlow 用的）：1 次 backward 得到**所有输入**的梯度，适合"输出少、参数多"的深度学习（loss 标量、参数亿级）。
- **前向模式**：1 次得一个方向的导数，适合"输入少、输出多"。JAX 同时支持前向模式（用 jvp）和反向（vjp）。

### 8.4 与 CUDA Graph 的关系

`loss.backward()` 每步都重建 Python 层图 → kernel launch 开销大。[[计算图]] 提到的 **CUDA Graph** 把整段前向+反向的 GPU kernel 序列捕获成一张静态图，重放时跳过 Python 调度，适合形状固定的训练/推理。但与 autograd 计算图不是一个概念。

### 8.5 在 RLHF / PPO 里 backward 的特殊性

PPO 的 loss = policy loss + value loss + entropy bonus，三段都要可导并共用一次 `loss.backward()`；clip 截断处的梯度要靠 mask 正确处理（`torch.where` 不会自动断梯度，需要 `detach` 或 `clamp`）。见 [[PPO clipped objective]]。

---
相关: [[PyTorch核心]]、[[autograd机制]]、[[计算图]]、[[nn.Module]]、[[optimizer state管理]]、[[hooks机制]]、[[backward过程]]
