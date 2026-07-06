# MoE路由

> **所属章节**: [[MoE]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中级（从零讲清"路由器如何选专家、为什么不可导"）

## 1. 一句话定义

**MoE 路由（routing，路由，门控）** 是 [[MoE]] 层里那个**小线性网络（router / gate）**对每个 token 计算各专家的分数、选出 **top-k 个**、再用 softmax 权重把选中专家的输出加权合并的过程；它是 MoE 的核心机制——决定"每个 token 该由哪些专家服务"，但其**离散选择（top-k）不可导**，是 MoE 训练难的根源。

## 2. 为什么需要它（动机与背景）

### 2.1 稀疏激活必须做"选择"

[[MoE]] 的核心是稀疏激活：$E$ 个专家里每个 token 只用 $k$ 个。那么**谁来决定用哪 $k$ 个**？这就是路由器的任务。

如果没有路由器（如随机选专家），模型无法学到"哪个专家擅长哪类 token"，稀疏激活的容量浪费。路由器让模型**自适应地**把不同 token 分给不同专家，实现"专家分工"。

### 2.2 路由器要满足的条件

1. **数据相关**：每个 token 根据自身内容决定路由（不是固定分配）；
2. **可学习**：路由策略要随训练优化（让相似 token 走同一专家）；
3. **稀疏**：只选 $k$ 个，不能全选（否则退化成 dense + 加权平均）；
4. **梯度可传播**：路由决策要让梯度能回传到路由器权重（否则学不了）。

第 4 条最难——top-k 是离散操作，天然不可导，催生了一系列工程处理（§4.2）。

## 3. 核心概念详解

### 3.1 Top-k 路由的完整流程

一个 token $x \in \mathbb{R}^d$ 走 MoE 层的步骤：

```
1. 路由器算分数：  logits = x @ W_g          # (E,) 每个专家一个分数
2. 选 top-k：       topk_vals, topk_idx = TopK(logits, k)   # 选分数最高的 k 个
3. softmax 权重：   g = softmax(topk_vals)    # 对 k 个分数做 softmax，得 k 个权重
4. 专家计算：       y_i = Expert_{idx_i}(x)   # 把 x 送进选中的 k 个专家各算一遍
5. 加权合并：       out = Σ_i g_i · y_i       # 用 softmax 权重加权求和
```

关键点：

- **分数** $\text{logits} = x W_g$，$W_g \in \mathbb{R}^{d \times E}$，是一个线性层；
- **top-k** 只对分数最高的 $k$ 个专家做 softmax（不是对所有 $E$ 个 softmax 再选——那样会让未选中的也有非零权重，破坏稀疏）；
- **权重** $g_i$ 是 $k$ 个标量，和为 1（softmax 性质）；
- **加权合并**：每个选中专家独立算 $\text{Expert}_i(x)$，再用 $g_i$ 加权求和。

### 3.2 路由不可导问题（核心难点）

> [!warning] top-k 是离散操作，不可导
> "选分数最高的 k 个"是一个**离散选择**（argmax/TopK 的索引是整数，对 $W_g$ 不可微）。梯度无法从"是否被选中"回传到路由器权重 $W_g$。

工程上的处理：

- **只对选中的 k 个传梯度**：softmax 权重 $g_i$ 是可微的（对 $W_g$），梯度通过 $g_i$ 流到 $W_g$；
- **未选中的专家得 0 梯度**：它们没参与计算，路由器对它们的分数梯度为 0。

后果：

- **路由器只从被选中的专家学**：如果某专家一开始没被选，它的分数永远不会被更新（梯度 0）→ 永远不被选 → **专家坍缩**（见 [[负载均衡损失]]）；
- 训练初期路由器是"随机"的，一旦某几个专家偶然被多选，正反馈让它更强，其余饿死。

### 3.3 路由的粒度

两种主流路由粒度：

| 粒度 | 机制 | 优点 | 缺点 | 用例 |
|---|---|---|---|---|
| **token 级路由**（主流） | 每个 token 独立选 top-k 专家 | 简单、与序列解耦 | 易不均衡（热门 token 都抢同一专家） | Mixtral / DeepSeek / 多数 MoE |
| **expert choice 路由** | 反过来：每个专家选 top-k token | **天然均衡**（每专家固定 token 数） | token 间顺序破坏、难并行 | 部分研究模型 |

- **token 级**：token 选专家，可能 90% 的 token 都选专家 3，专家 3 过载、专家 7 饿死；
- **expert choice**（Zhou et al. 2022）：每个专家反过来选 $B \cdot k / E$ 个 token（固定配额），天然均衡，但破坏了 token 的序列顺序（不同 token 走不同专家，组合时顺序乱），且实现复杂。

主流仍是 token 级 + [[负载均衡损失]] 兜底。

### 3.4 路由器的实现细节

- 路由器通常**只有一个线性层** $W_g \in \mathbb{R}^{d \times E}$，**无激活**（softmax 在 top-k 之后才加）；
- 参数量 $d \cdot E$，相对专家参数很小（如 $d=4096, E=8$ → 32K 参数，可忽略）；
- 路由器分数的量级要和专家输入匹配，过大/过小都会让 softmax 饱和或退化。

## 4. 数学原理 / 公式

### 4.1 路由公式

对 token $x \in \mathbb{R}^d$：

$$
\text{logits} = x W_g \in \mathbb{R}^E, \quad W_g \in \mathbb{R}^{d \times E}
$$

$$
\mathcal{T}(x) = \text{TopK}(\text{logits}, k) \quad \text{（选 k 个最大分数的索引）}
$$

$$
g_i = \frac{e^{\text{logits}_i}}{\sum_{j \in \mathcal{T}(x)} e^{\text{logits}_j}}, \quad i \in \mathcal{T}(x) \quad \text{（对 k 个分数 softmax）}
$$

$$
\text{MoE}(x) = \sum_{i \in \mathcal{T}(x)} g_i \cdot \text{Expert}_i(x)
$$

注意：softmax 的分母**只对被选中的 k 个**求和，不是全部 $E$ 个。这保证未选中的专家权重为 0，稀疏性成立。

### 4.2 梯度分析（为什么不可导）

记 $s = x W_g$（logits），$\mathcal{T} = \text{TopK}(s)$。输出 $y = \sum_{i \in \mathcal{T}} g_i e_i(x)$，其中 $g_i = \text{softmax}_\mathcal{T}(s_i)$。

- **$g_i$ 对 $s_i$（$i \in \mathcal{T}$）可微**：标准 softmax 的雅可比，梯度能流；
- **$\mathcal{T}$ 对 $s$ 不可微**：TopK 返回的索引是整数，$\nabla_s \mathcal{T}$ 不存在；
- **未选中专家 $j \notin \mathcal{T}$**：$g_j = 0$，且 $g_j$ 不参与 $y$，梯度 $\frac{\partial y}{\partial s_j} = 0$。

故路由器权重 $W_g$ 只从被选中专家的 $g_i$ 路径得梯度。未被选中的专家分数永远不更新——这是专家坍缩的数学根源。

### 4.3 容量与负载

设 batch 共 $T$ 个 token，$E$ 个专家，每 token 选 $k$。理想均衡下每专家处理 $\frac{T k}{E}$ 个 token。定义**容量**：

$$
\text{cap}_e = \frac{T k}{E} \cdot \text{capacity\_factor}
$$

- $\text{capacity\_factor}$（容量因子）通常 1.0~1.5，给专家留余量；
- 超过 $\text{cap}_e$ 的 token 被**丢弃**（dropped）或走残差（直接传），影响质量；
- 详见 [[负载均衡损失]] §3.3。

## 5. 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    """Top-k 路由器：对每个 token 选 k 个专家，返回索引与 softmax 权重。"""
    def __init__(self, d, num_experts=8, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d, num_experts, bias=False)  # 路由线性层
        self.k = top_k
        self.E = num_experts

    def forward(self, x):                           # x: (B, n, d)
        B, n, d = x.shape
        x_flat = x.reshape(-1, d)                   # (T, d), T = B*n
        logits = self.gate(x_flat)                  # (T, E)
        # 1. 选 top-k
        topk_vals, topk_idx = logits.topk(self.k, dim=-1)   # (T, k)
        # 2. 对 k 个分数做 softmax(只对选中的,不对全部 E)
        topk_w = F.softmax(topk_vals, dim=-1)               # (T, k)
        return topk_idx, topk_w, logits                      # logits 留给负载均衡损失用

# 验证路由
router = TopKRouter(d=512, num_experts=8, top_k=2)
x = torch.randn(2, 16, 512)                                 # 32 个 token
idx, w, logits = router(x)
print("路由索引:", idx.shape)                                # (32, 2): 每个 token 选 2 个专家
print("路由权重:", w.shape, w[0].sum())                      # (32, 2), 每行和=1(softmax)

# 观察负载分布:统计每个专家被选了多少次
counts = torch.zeros(8)
for t in range(32):
    counts[idx[t]] += 1
print("每专家被选次数:", counts.tolist())  # 理想均衡应≈32*2/8=8; 实际会偏(需负载均衡损失)
```

> [!tip] 关键观察
> 不加干预时，"每专家被选次数"会严重不均——这就是专家坍缩的直观体现。运行上面代码多次会发现某些专家总是 0 次。[[负载均衡损失]] 就是来治这个的。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[MoE]]（路由是 MoE 层的核心组件）、[[FFN]]（专家本身就是 FFN，路由决定用哪个）、[[softmax]]（路由权重的归一化）。
- **下游（应用）**: [[负载均衡损失]]（治路由带来的专家坍缩）、[[专家并行]]（路由结果决定 token 要发去哪个 GPU）、MoE 训练稳定性。
- **对比 / 易混**:
  - **MoE 路由 vs attention**：attention 的 softmax 是"token 关注哪些 token"（跨 token），MoE 路由的 top-k 是"token 送给哪些专家"（跨专家）。都是加权选择，但对象不同。
  - **Top-k 路由 vs Top-k 采样**：[[采样策略]] 的 top-k 是推理时从 vocab 选 token，MoE 路由的 top-k 是训练时从专家选服务者。完全不同的场景。
  - **路由 vs 门控（gate）**：文献里 router/gate 混用，都指这个小网络。SwiGLU 里也有个"gate"，那是 FFN 内部的门控，和 MoE 路由器无关。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 top-k 可导**：错。TopK 返回索引是离散的，不可导。梯度只通过 softmax 权重流，未选中专家得 0 梯度（§4.2）。
> 2. **softmax 对全部 E 个专家做**：错。应对**选中的 k 个**做 softmax，否则未选中专家权重非 0，破坏稀疏。
> 3. **以为路由器很复杂**：实际就一个线性层 $W_g$，无激活。复杂度 $d \cdot E$，参数可忽略。
> 4. **以为所有专家都得梯度**：错。未被选中的专家分数梯度为 0，参数也不更新（除非被其他 token 选中）。这是专家坍缩的根源。
> 5. **路由器加非线性**：主流实现不加（就线性层）。加激活会让分数分布失真，且无必要。Llama/Mixtral 等都是裸线性。
> 6. **忽略负载均衡**：路由器裸跑必然坍缩，必须配 [[负载均衡损失]] 或无辅助损失均衡。
> 7. **以为 expert choice 路由更好就用它**：expert choice 虽天然均衡，但破坏 token 顺序、难并行，主流仍用 token 级 + 均衡损失。

## 8. 延伸细节

### 8.1 路由策略的演化

- **shazeer 2017 原版**：噪声 top-k 路由（给 logits 加高斯噪声再 top-k，鼓励探索）；
- **Switch Transformer**：简化为 top-1（$k=1$），路由更稀疏、通信更省，但质量略掉；
- **expert choice（2022）**：反过来专家选 token，天然均衡；
- **soft MoE（2024）**：用软权重（所有专家都算但加权不同），避免离散选择不可导，但失去稀疏性（算力涨），是质量/算力的折中探索；
- **DeepSeek 无辅助损失**：路由本身不变（token 级 top-k），但用偏置项动态调整均衡（见 [[负载均衡损失]] §8.2）。

### 8.2 路由的"专家分工"是如何涌现的

训练良好的 MoE 会涌现专家分工：不同专家专攻不同 token 类型（如代码、数学、不同语言）。这种分工**不是人为指定**的，而是路由器在负载均衡约束下，自然把相似 token 聚到同一专家（因为聚在一起路由器分数更高、梯度更一致）。分析专家分工是 MoE 可解释性的研究热点。

### 8.3 路由与批处理的交互

路由是**逐 token**的，但 GPU 喜欢规则批处理。MoE 实现的工程难点：把同专家的 token 打包成不规则 batch（每专家 token 数不同），用 scatter/gather 算。这是 [[专家并行]] 的核心工程问题。

---
相关: [[MoE]]、[[负载均衡损失]]、[[专家并行]]、[[FFN]]、[[softmax]]、[[采样策略]]
