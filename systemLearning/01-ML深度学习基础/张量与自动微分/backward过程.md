# backward过程

> **所属章节**: [[张量与自动微分]]
> **所属模块**: [[01-ML深度学习基础]]
> **难度**: 进阶

## 1. 一句话定义

**反向传播（backward / backpropagation，反向传播）** 是 [[autograd机制]] 的执行阶段：从标量 loss 出发，**沿 [[计算图]] 反向遍历**，每个算子执行一次向量-雅可比积（VJP，见 [[梯度与Jacobian]]），把 $L$ 对每个叶子参数的梯度**累加**进 `.grad`，整个过程由一次 `loss.backward()` 调用完成，开销约为前向的 2~3 倍。

## 2. 为什么需要它（动机与背景）

训练一步 = 前向算 loss + 反向算梯度 + 优化器更新参数。其中"算梯度"是最大难点：

- 几十亿参数，手算不可行；
- 数值差分慢且有误差；
- 正向模式自动微分要算几十亿次。

**反向传播**用链式法则 + 反向拓扑排序，**一次遍历**就拿到标量 loss 对全部参数的梯度，复杂度与前向同阶。这是深度学习能训练的根本算法，没有它就没有现代 LLM。

## 3. 核心概念详解

### 3.1 反向传播的完整流程

一次 `loss.backward()` 内部做的事：

1. **检查**：`loss` 是否标量（非标量需传 `gradient=`）。
2. **拓扑排序**：从 `loss` 出发对计算图做反向拓扑序，保证一个节点在其所有后继都处理完后才处理。
3. **逐节点 VJP**：对每个算子节点，调用其 `backward`：
   - 输入：后继传来的伴随 $\bar{y}$（即上游梯度）；
   - 输出：各输入端的伴随 $\bar{x}=\bar{y}\cdot J$（一次向量-雅可比积）。
4. **累加到叶子**：把算出的参数梯度**加**到 `param.grad`（不是覆盖）。
5. **释放图**（默认）：中间节点和激活被释放，省内存；`retain_graph=True` 则保留。

### 3.2 拓扑排序为什么必要

一个节点可能有多个后继（如 $W$ 同时影响两条路径）。必须等所有后继的伴随都算好并**累加**完，才往上游传 —— 否则梯度不全。反向拓扑序保证"后继先于自己"。

### 3.3 伴随变量（adjoint）的流动

图里每个节点 $u$ 维护一个伴随 $\bar{u}=\frac{\partial L}{\partial u}$。传播规则：

$$
\bar{u} = \sum_{\text{后继 }s} \bar{s}\cdot\frac{\partial s}{\partial u}
$$

从 $\bar{L}=1$ 出发，一层层往前算，直到叶子。这就是链式法则在图上的执行。

### 3.4 标量约束与非标量 backward

- `loss.backward()` 要求 `loss` 是标量（0 维）。
- 若是向量 $\boldsymbol{y}$，要 `y.backward(gradient=v)`，相当于以 $\bar{\boldsymbol{y}}=\boldsymbol{v}$ 为起点传播，最终拿到 $L=\boldsymbol{v}^\top\boldsymbol{y}$ 对各参数的梯度。
- 常见偷懒写法：`loss = output.sum(); loss.backward()`。

### 3.5 内存 vs 计算

反向传播需要前向的**中间激活**（保存的 `ctx.saved_tensors`）来算局部导数。所以：

- 前向越深，要存的激活越多 → 显存瓶颈。
- [[gradient-checkpointing]] 的取舍：不存激活、反向时重算前向，省显存换时间。

## 4. 数学原理 / 公式

### 链式法则的执行

前向链 $x\to u\to v\to L$。反向：

$$
\bar{L}=1,\quad
\bar{v}=\bar{L}\cdot\frac{\partial L}{\partial v},\quad
\bar{u}=\bar{v}\cdot\frac{\partial v}{\partial u},\quad
\bar{x}=\bar{u}\cdot\frac{\partial u}{\partial x}
$$

每个箭头是一次 VJP。对多后继，伴随**求和**（多元链式法则）。

### 复杂度

设前向算子数为 $N$：

- 正向模式拿全梯度：$O(N\cdot n)$（$n$ 个输入，每次拿一列）。
- 反向模式拿标量 loss 对全参数梯度：$O(N)$（约 2~3× 前向）。
- 因为 $n$ 极大（参数多），反向模式完胜。

### 为什么不爆炸（$O(N)$ 而非 $O(2^N)$）

朴素地对每个参数单独走链式法则是指数级。反向传播的精髓：**每个节点只算一次伴随**，结果被所有上游共享，所以是线性复杂度。这是 reverse-mode AD 相比"对每个输出独立求导"的根本优势。

## 5. 代码示例

```python
import torch

# 一个完整训练步的 backward 全流程
torch.manual_seed(0)
x = torch.randn(4, 3)                       # 输入
W = torch.randn(3, 2, requires_grad=True)   # 叶子参数
b = torch.randn(2, requires_grad=True)
y = torch.randn(4, 2)                       # 标签

# 1. 前向（autograd 自动建图）
logits = x @ W + b                          # grad_fn = AddBackward
loss = ((logits - y) ** 2).mean()           # 标量根节点

# 2. 反向：一次调用，全参数梯度自动算出
loss.backward()
print(W.grad.shape, b.grad.shape)           # torch.Size([3,2]) torch.Size([2])

# 3. 标量约束：非标量必须传 gradient
logits2 = x @ W + b                         # shape (4,2)，非标量
# logits2.backward()                        # 报错：grad can be implicitly created only for scalar outputs
logits2.backward(gradient=torch.ones_like(logits2))  # 等价于以 v=1 起步

# 4. 多次 backward：默认图会释放
loss = ((x @ W + b - y) ** 2).mean()
loss.backward()                             # 第一次，图释放
# loss.backward()                           # 报错：second time —— 要 retain_graph=True

loss = ((x @ W + b - y) ** 2).mean()
loss.backward(retain_graph=True)            # 保留图
loss.backward()                             # 第二次成功，但梯度是累加（要 zero_grad）

# 5. 训练循环标准写法（强调 zero_grad）
opt = torch.optim.SGD([W, b], lr=0.1)
for step in range(3):
    opt.zero_grad(set_to_none=True)         # 关键：清零，否则梯度累加
    logits = x @ W + b
    loss = ((logits - y) ** 2).mean()
    loss.backward()                         # 反向算梯度 -> W.grad, b.grad
    opt.step()                              # 用梯度更新参数
    print(step, loss.item())
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[autograd机制]]（backward 是其触发入口）、[[计算图]]（遍历的对象）、[[梯度与Jacobian]]（每个节点做的 VJP）。
- **下游（应用）**: 优化器（SGD/Adam/PPO 更新器消费 `.grad`）、[[gradient clipping]]（在 backward 后、step 前裁剪）、混合精度 [[loss scaling]]（缩放 loss 防梯度下溢）。
- **对比 / 易混**:
  - **backward vs forward**：前向算数值建图，反向算梯度。开销相近但反向要存激活。
  - **`backward()` vs `torch.autograd.grad`**：前者把梯度写进 `.grad`；后者返回梯度列表、不污染 `.grad`，适合多个 loss 共享参数。
  - **反向传播 vs 反向模式 AD**：反向传播专指神经网络里 loss→参数 的反向模式 AD，是后者的特例。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **忘了 `zero_grad()`**：梯度累加导致越来越大，训练发散 —— 头号 bug。
> 2. **非标量直接 `backward()` 报错**：`loss` 必须 0 维；向量要传 `gradient=` 或先 `.sum()/.mean()`。
> 3. **`retain_graph` 用错**：默认反向后图释放，想二次 backward（或边反向边再前向）要 `retain_graph=True`；但多次反向会让梯度累加，仍需 zero_grad。
> 4. **把反向当"免费"**：反向开销 ≈ 前向 2~3×，且要存激活吃显存；工程上显存瓶颈常来自反向而非前向。
> 5. **跨 step 重复用旧图**：动态图每次前向重建，旧的 `grad_fn` 指向已释放图，直接复用会报错，要重新前向。
> 6. **`create_graph=True` 想当然**：它让反向也建图（可二阶导），但显存翻倍，普通训练不要开。
> 7. **in-place 改了反向要用的值**：前向保存的中间值被原地改后，反向算错或报 `one of the variables needed for backward has been modified`。

## 8. 延伸细节

### 8.1 gradient checkpointing 与重算

反向需要前向激活。层数多时激活占显存大头。[[gradient-checkpointing]] 只保留少数检查点的激活，反向到该层时**重算前向**补回 —— 用时间换显存，LLM 训练几乎标配。

### 8.2 混合精度下的 loss scaling

fp16 梯度可能小到下溢成 0。loss scaling：前向前把 loss 乘大数 $S$，反向后梯度再除 $S$，保住小梯度。bf16 动态范围大，通常无需 scaling。见 [[loss scaling]]。

### 8.3 梯度裁剪的位置

`loss.backward()` 之后、`optimizer.step()` 之前做 `clip_grad_norm_`，对全局梯度范数裁剪，防梯度爆炸。RLHF/PPO 训练稳定性的标准动作。

### 8.4 并行/分布式下的反向

DDP/FSDP 在反向时**并行地**做梯度 all-reduce / reduce-scatter，让通信和反向计算重叠。反向因此不只是"算梯度"，还涉及跨卡梯度同步。见 [[DDP]]、[[FSDP]]、[[all-reduce]]。

### 8.5 stale policy 与异步训练

异步 RL（如 rollout 和 training 分离）中，backward 用的参数可能已不是采样时的参数 → [[stale policy problem]]。这是训推分离系统的核心难题之一。

---
相关: [[张量与自动微分]]、[[autograd机制]]、[[计算图]]、[[梯度与Jacobian]]
