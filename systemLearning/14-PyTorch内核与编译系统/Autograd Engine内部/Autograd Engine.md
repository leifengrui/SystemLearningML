# Autograd Engine

> **所属章节**: [[Autograd Engine内部]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: Autograd Engine / autograd 引擎 / Engine / grad_fn / Node / backward 执行器 / 拓扑排序反向遍历
> **难度**: 高（需懂 [[autograd机制]] + [[计算图]] + [[Dispatcher]]）


## 1. 一句话定义

**Autograd Engine** 是 PyTorch 的**反向传播执行器**——`.backward()` 触发时，它从**输出节点的 grad_fn** 出发，对前向构建的 **DAG（有向无环图，节点 = `Node`/`grad_fn`，边 = saved tensor 引用）** 做**反向拓扑排序**（先算叶子依赖的节点，再算上游），逐节点调用其 `apply`（即该 op 的 backward 公式）算局部梯度，沿边**链式法则累加**梯度到下游节点的 `grad`，直到所有 `requires_grad=True` 的叶子（weight）grad 都累积完毕。它是 [[autograd机制]]（用户视角"autograd 怎么记录/算梯度"）背后的 **C++ 执行核心**，与 [[Dispatcher]] 的 Autograd 层（前向插层建图）配套——dispatcher 在前向建图，Autograd Engine 在反向跑图。

> [!note] 三句话定位
> - **是什么**：`.backward()` 触发的 C++ 执行器，反向拓扑遍历 grad_fn DAG，逐节点算局部梯度并链式累加。
> - **为什么**：autograd 公式（如 `AddBackward`）是局部的，需引擎把数千节点串成正确顺序（拓扑序）跑完整个网络的反向。
> - **与 [[autograd机制]] 关系**：autograd机制 是用户视角（怎么记 grad_fn、怎么 backward）；Autograd Engine 是其 C++ 实现细节（拓扑排序、多线程、accumulate、saved tensors）。


## 2. 为什么需要它（动机与背景）

### 2.1 autograd 公式是局部的，需引擎串起来

每个 op 有自己的 backward 公式（`AddBackward`/`MulBackward`/`MatMulBackward`），只算"该 op 的局部梯度"。但一个网络有上千 op 串成 DAG，`.backward()` 要按**正确顺序**（下游先于上游）逐个调公式、沿边累加梯度。这个"排序 + 调度 + 累加"的执行器就是 Autograd Engine。没有它，autograd 公式只是一堆孤立函数。

### 2.2 反向拓扑排序保证正确性

DAG 里节点 $A$ 依赖 $B$（$A$ 的梯度需要 $B$ 的梯度先算）。反向传播要**先算叶子端（输出端），后算根端（输入/weight 端）**——即**反向拓扑序**。Autograd Engine 用**累计 ready 队列 + 引用计数**（每节点入度计数）调度，保证某节点所有下游梯度算完才调它的公式。是链式法则正确性的调度保障。

### 2.3 多线程并行反向

大网络反向有大量独立节点可并行（不同分支不依赖）。Autograd Engine 用**线程池**并行调度 ready 节点（引用计数归零的节点入 ready 队列，多线程取执行）。是 PyTorch 反向能利用多核的关键。

### 2.4 saved tensors 与显存

backward 公式需要前向的某些中间值（如 `MatMulBackward` 需前向的 $Q,K$；`SoftmaxBackward` 需前向 softmax 输出）。这些在 [[Dispatcher]] 的 Autograd 层建图时被 **save**（存 `grad_fn.saved_tensors_`），反向时取出用。是 [[activation memory]] 的来源（前向为反向存的中间值占显存）。详见 [[activation memory]]。

### 2.5 与 [[Dispatcher]] 的分工

- **Dispatcher Autograd 层**：前向每次 op 调用时插层，建 `grad_fn`、连边、save tensors，构 DAG。是"建图"。
- **Autograd Engine**：`.backward()` 时反向遍历 DAG、调公式、累加梯度。是"跑图"。

两者配合完成 autograd 全流程。


## 3. 核心概念详解

### 3.1 Node / grad_fn

```cpp
struct Node {
  std::shared_ptr<Node> next_edges_[N];   // 下游节点 (反向传播的下一跳)
  edge_list next_edges();
  virtual variable_list apply(variable_list&& grads);  // backward 公式
  // saved_tensors_ (前向存的中间值, 反向取用)
};
```

每 op 的 Autograd kernel（dispatcher 注册的 `AutogradCUDA` 等）创建一个 `Node` 子类（如 `AddBackward0`），其 `apply` 实现该 op 的 backward 公式。`next_edges_` 指向其输入 tensor 的 grad_fn（DAG 边）。`torch.Tensor.grad_fn`（Python）= 指向该 `Node` 的指针。

### 3.2 前向建图（Dispatcher Autograd 层）

前向 `at::add(x, y)`（`requires_grad=True`）经 dispatcher → `AutogradCUDA` kernel：
1. 建 `AddBackward0` node。
2. `node.next_edges_ = {x.grad_fn, y.grad_fn}`（连边到输入的 grad_fn）。
3. `node.save(x, y)`（按 backward 公式需 save 的）。
4. 输出 tensor 的 `grad_fn_ = node`，`requires_grad=True`。

如此串成 DAG。`.backward()` 跑这个 DAG。

### 3.3 反向拓扑排序（引用计数 + ready 队列）

```
Engine::execute(root_nodes, grads):
  1. 对每节点计算入度 (有多少上游依赖它) = next_edges 的引用计数
  2. root_nodes (输出端) 入 ready_queue, 配 grads
  3. while ready_queue 非空:
       node, grad = ready_queue.pop()
       new_grads = node.apply(grad)        # 调 backward 公式
       for (edge, ng) in zip(node.next_edges, new_grads):
           dst = edge.function
           dst.accumulate_grad(ng)         # 累加到 dst 的 grad
           dst.indegree--; if dst.indegree==0: ready_queue.push(dst)
```

引用计数归零 = 所有上游已算完 → 入 ready 队列。保证拓扑序。多线程版多 worker 从 `ready_queue`（thread-safe）取节点并行执行。

### 3.4 accumulate_grad

叶子节点（weight，`is_leaf=True`）的 grad 不是"算一次"，而是**累加**（同一 weight 在多路径被用，梯度从各路径累加）。`accumulate_grad` 把 `ng` 加到 `tensor.grad`（若已有则 `+=`）。是链式法则在分叉点的体现。

### 3.5 saved tensors 与显存

`node.saved_tensors_` 持有前向 save 的中间值（如 matmul 的 $Q,K$）。反向 `apply` 取出用。这些 saved tensors 占显存 = [[activation memory]]。`gradient checkpointing`（见 [[gradient checkpointing]]）通过"重算前向"减少 saved，省显存增 FLOP。

### 3.6 hooks

`register_hook`/`full_backward_hook` 在 node 上注册回调，`apply` 前后调用。用于 inspect/modify 梯度、debug。详见 [[hooks机制]]。

### 3.7 retain_graph

默认 `.backward()` 后**释放 DAG**（清 grad_fn、saved tensors）省显存。`retain_graph=True` 保留，可多次 backward（如 RNN 多步、GradNorm 需二阶）。释放后不能再 backward（报 "freed"）。

### 3.8 多 GPU / distributed 的反向

DDP/FSDP 的反向在 Autograd Engine 外叠加通信：grad accumulate 后 trigger `all_reduce`（DDP）或 `reduce_scatter`（FSDP）。见 [[Fully Sharded Data Parallel]]/[[FSDP2]]。Autograd Engine 只管本地节点的执行，通信是外挂 hook。

### 3.9 与 torch.compile 的关系

`torch.compile`（见 [[torch.compile]]）的 [[AOTAutograd]] 把前向+反向 joint 分解，Inductor codegen 成 fused kernel，**绕过 Autograd Engine 的逐节点调度**（直接一个 fused 反向 kernel）。但 eager 模式仍走 Autograd Engine。编译路径与 engine 路径并存。


## 4. 数学原理 / 公式

### 4.1 链式法则的引擎化

反向传播本质是链式法则：

$$
\frac{\partial L}{\partial x_i} = \sum_{j \in \text{children}(i)} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}
$$

引擎把每 $y_j$ 的 $\frac{\partial L}{\partial y_j}$（已算好的下游 grad）传给 $i$，$i$ 的 `apply` 算 $\frac{\partial y_j}{\partial x_i}$ 局部，累加所有 children 贡献得 $\frac{\partial L}{\partial x_i}$。

### 4.2 拓扑序的保证

设 DAG，节点 $i$ 的入度 $d_i$（有多少 children 依赖它）。reverse topological order 保证：$i$ 入 ready 队列当且仅当所有 children 已算完（$d_i$ 归零）。故 $i$ 的 `apply` 被调时，其所有 $\frac{\partial L}{\partial y_j}$ 已累积完整。

### 4.3 累加的必要性（分叉点）

若 $x$ 被 $y_1, y_2$ 两个 op 用，则：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y_1}\frac{\partial y_1}{\partial x} + \frac{\partial L}{\partial y_2}\frac{\partial y_2}{\partial x}
$$

两路径独立算到 $x$，`accumulate_grad` 把两次贡献 `+=` 到 `x.grad`。引用计数：$x$ 的入度 = 2，两次贡献归零才 complete。

### 4.4 时间复杂度

反向遍历每节点一次，$O(V+E)$（V=节点数，E=边数）。与 forward 的 $O(V)$ 同阶（每 op 一 backward node）。多线程并行把串行 $O(V)$ 压到 $O(V/P + \text{critical path})$（P 线程，critical path 是 DAG 最长依赖链）。


## 5. 代码示例（可选）

### 5.1 Python 层观察 grad_fn DAG

```python
import torch
x = torch.randn(4, requires_grad=True)
y = x * 2          # MulBackward0, y.grad_fn
z = y.sum()        # SumBackward0
print(z.grad_fn)                     # SumBackward0
print(z.grad_fn.next_functions)      # ((MulBackward0(obj), 0),)
print(z.grad_fn.next_functions[0][0].next_functions)  # ((AccumulateGrad{x}, 0),)
z.backward()
print(x.grad)       # tensor([2.,2.,2.,2.])  链式: dz/dy=1, dy/dx=2
```

### 5.2 模拟反向拓扑遍历

```python
from collections import deque
def backward_simulate(grad_fn_dag, root, root_grad):
    """模拟 Autograd Engine 的引用计数 + ready 队列."""
    indeg = {n: 0 for n in grad_fn_dag}
    for n in grad_fn_dag:
        for (child, _) in n.get('next', []):
            indeg[child] = indeg.get(child,0) + 1
    grads = {root: root_grad}
    ready = deque([root])
    while ready:
        node = ready.popleft()
        ng = grads[node]
        for (child, local_grad_fn) in node.get('next', []):
            contribution = local_grad_fn(ng)   # 局部 backward
            grads[child] = grads.get(child, 0) + contribution  # accumulate
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
    return grads
```

### 5.3 retain_graph 与多次 backward

```python
x = torch.randn(4, requires_grad=True)
y = (x**2).sum()
y.backward(retain_graph=True)   # 保留 DAG
y2 = (x**3).sum()
y2.backward()                   # 第二次, 不同图
# 若不 retain_graph, 第一次 backward 后 grad_fn 被释放, 再 backward 报错
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[autograd机制]]（用户视角的 autograd）、[[计算图]]（DAG 概念）、[[Dispatcher]]（前向 Autograd 层建图）、[[ATen与c10]]（`Node`/`TensorImpl`/`intrusive_ptr`）。
- **下游（应用）**: [[hooks机制]]（grad hook 在 engine 调用）、[[forward与backward]]（用户 API 背后）、[[gradient checkpointing]]（减 saved tensors 的 trade-off）、[[Fully Sharded Data Parallel]]/[[FSDP2]]（反向后触发通信 hook）、[[torch.compile]]/[[AOTAutograd]]（编译路径绕过 engine 调 fused 反向 kernel）、[[activation memory]]（saved tensors 占显存）。
- **对比 / 易混**:
  - **Autograd Engine vs [[Dispatcher]] Autograd 层**：前者反向跑图，后者前向建图。配合完成 autograd。
  - **Autograd Engine vs [[autograd机制]]**：autograd机制 是用户视角（怎么用 backward）；Engine 是 C++ 实现细节（拓扑排序/多线程/accumulate/saved）。
  - **Autograd Engine vs [[torch.compile]]**：eager 反向走 engine 逐节点；compile 路径把反向融成 fused kernel，绕过 engine 调度（快但失去逐节点 hook）。
  - **saved tensors vs [[activation memory]]**：saved tensors 是 engine 反向需要的中间值，它们占的显存即 activation memory 主体。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 backward 公式自己跑
> backward 公式（`AddBackward.apply`）是局部的，需 Autograd Engine 串成正确顺序（反向拓扑序）逐个调。没有 engine，公式是一堆孤立函数。

> [!warning] 误区 2：混淆前向建图与反向跑图
   建图在前向（dispatcher Autograd 层），跑图在反向（engine）。不是 backward 时才建图。`retain_graph=False` 释放的是建好的图。

> [!warning] 误区 3：忽视 accumulate 的累加语义
> 同一 weight 在多路径被用，grad 是各路径贡献**累加**，不是覆盖。`x.grad` 第二次 `+=` 不是 `=`。这也是 `.backward()` 前要 `zero_grad()` 的原因（上次的残留会累加进来）。

> [!warning] 误区 4：以为 saved tensors 免费
> backward 需要前向中间值，engine 存它们 = 占显存（activation memory）。长序列/大 batch 的 saved 很大，是显存瓶颈。gradient checkpointing 重算换省。

> [!warning] 误区 5：默认 backward 后还能再 backward
> 默认 `retain_graph=False`，backward 后 DAG 释放，再调报错。需多次 backward 要 `retain_graph=True`（多占显存）。

> [!warning] 误区 6：忽视多线程反向的非确定性
> 多线程 engine 并行调 ready 节点，累加顺序非确定 → 浮点累加顺序变 → grad 有 $O(\epsilon)$ 差异（与 [[训推不一致]] 相关）。需严格复现用单线程或固定调度。

> [!warning] 误区 7：混淆 grad_fn 与 leaf
> leaf（weight，`is_leaf=True`）的 grad 累加到 `.grad`（accumulate grad）；非 leaf 的中间 tensor 默认不保留 grad（`retain_grad()` 才留）。debug 时常见困惑。


## 8. 延伸细节

### 8.1 Node 的引用计数与生命周期

`Node` 用 `intrusive_ptr`，前向建图时输出 tensor 持有其 grad_fn（+1 ref），输入 grad_fn 被 next_edges 引用（+1）。backward 后释放 next_edges（-1），grad_fn 归零析构（释放 saved tensors）。`retain_graph` 阻止释放。

### 8.2 多线程反向的调度

Autograd Engine 的 `ready_queue` 是 thread-safe，多 worker 线程并发取节点。`Node` 的 `apply` 若 thread-unsafe（如某些自定义 op）需加锁或标 `not thread-safe`。默认 ATen op 的 backward thread-safe。

### 8.3 compiled autograd

PyTorch 有 `torch._C._compiled_autograd` 实验：把 grad_fn 图编译成 fused kernel，绕过 engine 的逐节点 Python/C++ 调度开销。是 torch.compile 路径对反向的优化。详见 [[AOTAutograd]]/[[Inductor]]。

### 8.4 二阶/高阶导

`create_graph=True` 的 backward 在算一阶时同时建二阶的 grad_fn 图，支持高阶导（如 GradNorm、meta-learning）。开销大（建额外图）。

### 8.5 内容来源

Autograd Engine 整理自 PyTorch `torch/csrc/autograd/engine.cpp`、`torch/csrc/autograd/function.h`（`Node`），以及 dev blog "Autograd Internals"。拓扑排序 + 引用计数调度见 `Engine::execute_with_graph_task`。

---
相关: [[Autograd Engine内部]] | [[autograd机制]] | [[计算图]] | [[forward与backward]] | [[Dispatcher]] | [[ATen与c10]] | [[hooks机制]] | [[gradient checkpointing]] | [[Fully Sharded Data Parallel]] | [[FSDP2]] | [[torch.compile]] | [[AOTAutograd]] | [[activation memory]] | [[训推不一致]]
