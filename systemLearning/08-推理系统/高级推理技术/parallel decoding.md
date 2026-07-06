# parallel decoding

> **所属章节**: [[高级推理技术]]
> **所属模块**: [[08-推理系统]]
> **别名**: parallel decoding / 并行解码 / parallel sample generation
> **难度**: 中高（需懂自回归 + [[KV cache management]] + 不动点迭代）


## 1. 一句话定义

**parallel decoding（并行解码）** 泛指**不依赖独立 draft model**、通过模型自身结构或数学技巧**一次 forward 同时预测多个未来 token** 的解码加速技术家族，包括 **Medusa**（多头并行预测）、**Lookahead Decoding**（Jacobi 不动点迭代）、**blockwise parallel decoding**（分块并行）等。与 [[speculative decoding]] 的区别是：后者需一个独立的 draft model 串行猜，parallel decoding 用目标模型自身的并行预测头或迭代法，省去 draft model 的开销与 cache 管理。

> [!note] 三句话定位
> - **是什么**：用模型自身一次性预测/验证多个未来位置的 token，并行出多个。
> - **为什么**：自回归串行是 decode 瓶颈；speculative 需 draft model（额外参数/cache），parallel decoding 试图"单模型并行"。
> - **代表方法**：Medusa（多头）、Lookahead Decoding（Jacobi 迭代）、blockwise parallel decoding。


## 2. 为什么需要它（动机与背景）

### 2.1 自回归串行的瓶颈

标准 decode 每步 1 token，$n$ 步需 $n$ 次 forward。每次 forward 是 [[KV cache management]] 的 memory-bound 操作（读整个 cache）。长序列下 decode 慢、带宽利用率低。

### 2.2 speculative decoding 的局限

[[speculative decoding]] 用 draft model 加速，但：

- 需**独立 draft model**：额外参数、独立 KV cache、需训练/选型与 target 对齐。
- draft 串行生成 $k$ 个仍是 $k$ 次 forward（虽小模型快，但有开销）。
- 接受率依赖 draft-target 分布对齐，对齐差时退化。

### 2.3 parallel decoding 的思路

能否**单模型一次 forward 直接得多个 token**？挑战在于自回归的因果依赖（token $i$ 依赖 $i-1$）。parallel decoding 用两类思路破局：

1. **结构并行**（Medusa）：在模型最后隐藏态上加多个"未来头"，每个头预测未来第 $t$ 个位置，一次 forward 得 $k$ 个候选。
2. **迭代并行**（Lookahead/Jacobi）：把自回归视为不动点迭代，从初始猜测并行更新所有位置，若干轮收敛。

两者都**不需独立 draft model**，省去 draft 的选型与 cache 开销。


## 3. 核心概念详解

### 3.1 Medusa：多头并行预测

Medusa（Cai et al. 2024）在 target 模型的最后一层隐藏态 $h$ 上加 $k$ 个 **Medusa heads**（小型 MLP），第 $i$ 个 head 预测"从当前位置看，未来第 $i$ 个 token"：

$$
\hat{x}_{t+i} = \text{head}_i(h_t)
$$

一次 target forward（得到 $h_t$）即得 $k$ 个候选未来 token。配合 **tree attention** 验证这些候选（把候选组织成树，一次验证多个分支），接受符合分布的。

- 优点：无需独立 draft model，头小（参数少），与 target 共享 backbone。
- 代价：需训练 Medusa 头；预测精度低于独立 draft（因只看 $h_t$ 不看后续），接受率需靠 tree attention 补。

### 3.2 tree attention（树形验证）

Medusa 的 $k$ 个候选若各自有 top-$c$ 选择，组合成 $c^k$ 条路径（树）。tree attention 把这棵树压成一次 target forward 验证（用 attention mask 让每个节点只看其祖先路径），一次得所有路径的验证结果，取接受最长路径。

### 3.3 Lookahead Decoding：Jacobi 不动点迭代

Lookahead Decoding（Fu et al. 2024）基于 **Jacobi 迭代**：把生成 $n$ 个 token 视为求解不动点 $x^* = F(x^*)$（$F$ 是"给定当前 $n$ 个猜测，各位置重预测"的映射）。从随机初值 $x^{(0)}$ 出发，并行更新所有位置：

$$
x^{(t+1)}_i = \text{argmax/sample } p(\cdot | x^{(t)}_{<i} \cup \text{已知前缀})
$$

若干轮后，前面位置收敛（不再变），收敛部分作为已生成 token，后续位置继续迭代。每轮是一次 forward 并行更新 $n$ 位置。

- 优点：无需 draft model、无需训练额外头，纯算法。
- 代价：收敛轮数不确定，某些序列收敛慢；每轮 forward 处理 $n$ 位置（计算量比单 token 大）。

### 3.4 blockwise parallel decoding

Stern et al. 2018：把序列分块，每块 $k$ 个 token，训练模型一次预测整块的 $k$ 个 token（类似 Medusa 但更早）。生成时每块并行预测，再串行验证修正。Medusa 是其改进。

### 3.5 与 speculative decoding 的关系

parallel decoding 可视为 speculative decoding 的"无 draft model"变体——用模型自身的并行头或迭代替代 draft 的串行猜测。广义上 [[speculative decoding]] 包含 Medusa（vLLM 把 Medusa 作为 speculative 的一种实现），但严格区分时 parallel decoding 特指"不需独立 draft model"的技术。


## 4. 数学原理 / 公式

### 4.1 Jacobi 不动点（Lookahead 的基础）

自回归生成可写成：给定前缀 $x_{<t}$，求 $x_t = f(x_{<t})$（$f$ 是 argmax/采样）。把生成 $n$ 个新 token 视为联立方程组：

$$
x_t = f(x_{<t}), \quad x_{t+1} = f(x_{<t}, x_t), \quad \dots, \quad x_{t+n-1} = f(x_{<t}, \dots, x_{t+n-2})
$$

这是一个 $n$ 维不动点问题 $x^* = F(x^*)$。Jacobi 迭代从初值 $x^{(0)}$ 出发：

$$
x^{(t+1)} = F(x^{(t)})
$$

每轮并行更新全部 $n$ 个位置（一次 forward）。若收敛（某些位置连续两轮不变），收敛部分即为解。

> [!note] 收敛性
> Jacobi 迭代不保证收敛，但在 LLM 生成中经验上常在少数轮内部分收敛（因 argmax 路径稳定）。收敛的位置数 = 一轮产出的 token 数。

### 4.2 Medusa 多头预测的形式

target 骨干输出隐藏态 $h_t \in \mathbb{R}^d$。Medusa head $i$ 是线性层 $W_i \in \mathbb{R}^{V \times d}$：

$$
\hat{p}_i(\cdot) = \text{softmax}(W_i h_t)
$$

预测"未来第 $i$ 个位置"的分布。一次 forward 得 $\hat{p}_1, \dots, \hat{p}_k$。验证时用 target 在"假设接受前 $i$ 个"后的真实分布 $p(\cdot | x_{<t+i})$ 对比，接受-拒绝（类似 [[speculative decoding]]）。

### 4.3 加速比

设一次 target forward 耗时 $T_t$（处理 $k$ 位置仅略增于 1 位置，因 memory-bound）。每轮产出 $a$ 个收敛/接受的 token：

$$
S = \frac{a \cdot T_t}{T_t} = a
$$

即加速比 ≈ 每轮产出数 $a$。Medusa 典型 $a=2$–$3$，Lookahead 类似。比 [[speculative decoding]] 省去 draft 的 $k T_d$ 开销，但 $a$ 受预测精度限制。


## 5. 代码示例（可选

### 5.1 简化 Medusa 风格多头并行预测

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MedusaHead(nn.Module):
    """单个 Medusa head: 隐藏态 -> vocab logits."""
    def __init__(self, d_model, vocab):
        super().__init__()
        self.fc = nn.Linear(d_model, vocab)
    def forward(self, h):
        return self.fc(h)   # [B, vocab]

class MedusaLM(nn.Module):
    """假模型: 主 LM 头 + k 个 Medusa head, 一次 forward 得 k 个未来候选."""
    def __init__(self, d_model, vocab, k=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)
        self.medusa_heads = nn.ModuleList([MedusaHead(d_model, vocab) for _ in range(k)])
    def forward(self, tokens):
        h = self.embed(tokens)              # [B, S, d] (简化: 无 transformer 层)
        h_last = h[:, -1, :]                # [B, d] 最后位置隐藏态
        main = self.lm_head(h_last)         # [B, V] 主位置
        futures = [head(h_last) for head in self.medusa_heads]  # 各 [B, V]
        return main, futures

def medusa_decode(model, prompt, max_new=20, k=4, accept_all=True):
    """简化 Medusa 解码 (教学: 省略 tree attention 验证, 演示并行出多 token)."""
    tokens = list(prompt)
    produced = 0
    while produced < max_new:
        toks = torch.tensor([tokens])
        main, futures = model(toks)
        # main: 当前位置预测; futures[i]: 未来第 i+1 位置预测
        nxt = int(main.argmax(dim=-1)[0])   # 接受主头预测
        tokens.append(nxt); produced += 1
        # 多头并行候选 (实际需 tree attention 验证, 此处简化全接受)
        if accept_all:
            for fh in futures:
                cand = int(fh.argmax(dim=-1)[0])
                tokens.append(cand); produced += 1
                if produced >= max_new: break
    return tokens[len(prompt):]

torch.manual_seed(0)
model = MedusaLM(d_model=32, vocab=100, k=4)
out = medusa_decode(model, prompt=[5,6,7], max_new=12, k=4)
print('medusa out:', out, 'len:', len(out))
# 一次 forward 出 ~5 个 (1 主 + 4 头, 简化全接受)
```

### 5.2 Jacobi 迭代风格（Lookahead 简化）

```python
def jacobi_decode(model, prompt, n=4, max_rounds=20, max_new=16):
    """简化 Jacobi 迭代: 并行更新 n 个位置, 收敛部分作为产出."""
    tokens = list(prompt)
    produced = 0
    while produced < max_new:
        # 初始化 n 个猜测 (随便填)
        guess = list(tokens) + [0] * n
        for r in range(max_rounds):
            toks = torch.tensor([guess])
            main, _ = model(toks)
            logits = main[0]              # [V] 简化: 只看末位
            new_nxt = int(logits.argmax())
            if new_nxt == guess[-1]:
                break   # 收敛
            guess[-1] = new_nxt
        tokens.append(new_nxt); produced += 1
    return tokens[len(prompt):]
```

### 5.3 关键注释

- Medusa 版的"全接受"是简化，实际需 tree attention 用 target 真实分布验证每条候选路径（类似 [[speculative decoding]] 的接受-拒绝）。
- Jacobi 版高度简化，真实 Lookahead Decoding 维护一个滑动窗口并行更新多个位置、用历史猜测加速收敛。


## 6. 与其他知识点的关系

- **上游（依赖）**: [[KV cache management]]（并行验证仍需 cache）、[[attention]]（tree attention 的因果 mask）、autoregressive 生成。
- **下游（应用）**: [[batching tradeoff]]、[[GPU utilization]]（parallel decoding 提升算力利用率）、[[sampling throughput]]（RLHF 采样加速）、[[policy deployment (PD)]]。
- **对比 / 易混**:
  - **parallel decoding vs [[speculative decoding]]**：核心区别——speculative 需**独立 draft model**，parallel decoding **不需**（用模型自身多头或迭代）。vLLM 广义把 Medusa 归为 speculative 的一种，但严格区分时 parallel = 无 draft。
  - **parallel decoding vs [[beam search]]**：beam 是搜索（改分布、提质），parallel 是并行生成（保分布或加速），目标不同。
  - **parallel decoding vs naive 并行**：自回归无法朴素并行（因果依赖）；parallel decoding 用多头/迭代绕过依赖。


## 7. 常见误区与易错点

> [!warning] 误区 1：把 parallel decoding 等同于 beam search
> 不是。beam search 是**搜索**（保留 top-B 序列，改变输出分布倾向高概率）；parallel decoding 是**并行生成**（一次出多个 token，配合验证保持分布或加速）。目标与机制不同。

> [!warning] 误区 2：以为自回归能"朴素并行"
> 不能。token $i$ 依赖 $i-1$ 的具体值，朴素并行违反因果。parallel decoding 靠多头（基于 $h_t$ 预测未来）或 Jacobi 迭代（并行更新+收敛）绕过，不是"同时各走各的"。

> [!warning] 误区 3：以为 Medusa 无需训练
> Medusa 的多头需训练（用 target 输出做监督，让头学会预测未来位置）。不是插上即用。EAGLE 等改进也需训练 draft 结构。

> [!warning] 误区 4：忽略 tree attention 的验证开销
> Medusa 多头出 $k$ 个候选，组合成 $c^k$ 条路径，tree attention 验证需构造因果 mask、一次 forward 处理树。实现复杂，树过大时开销抵消并行收益。

> [!warning] 误区 5：以为 Jacobi 必收敛
> Jacobi 迭代不保证收敛，某些序列（高不确定性、分叉多）可能振荡。Lookahead 的工程实现靠历史窗口与接受策略缓解，但非万能。


## 8. 延伸细节

### 8.1 Medusa 的训练

Medusa heads 用 target 模型在语料上的"未来 token"做监督训练（教师强制），让头 $i$ 学会从 $h_t$ 预测 $x_{t+i}$。训练成本低（只训小头，backbone 冻结）。精度低于独立 draft（因 $h_t$ 信息有限），但靠 tree attention 多候选补回。

### 8.2 EAGLE：结构与 draft 的融合

EAGLE（Li et al. 2024）介于两者间：用一个**轻量 draft 模型**但输入是 target 的**隐藏态**（而非 token embedding），draft 与 target 对齐更紧，接受率高。可视为"用结构创新改善 draft"的混合路线。

### 8.3 Lookahead Decoding 的工程细节

Lookahead 维护一个大小 $n$ 的滑动窗口，每轮并行更新窗口内 $n$ 个位置，用前几轮的历史猜测加速收敛（类似 N-gram 提示）。收敛的位置出窗、新位置进窗，流水线推进。

### 8.4 与 RLHF 的关系

parallel decoding（尤其 Medusa/EAGLE）在 [[rollout worker]] 采样加速中渐成主流——不需独立 draft model、与 target 共享 backbone，适配 RLHF 中 $\theta$ 持续变化的场景（[[weight sync mechanism]] 后无需重训 draft，因头与 backbone 同步更新）。verl/vLLM 已集成 Medusa/EAGLE 选项。

### 8.5 局限与适用

- parallel decoding 在**算力受限、显存紧**的场景比 [[speculative decoding]] 优（省 draft）。
- 但并行预测精度通常低于独立 draft（Medusa 头信息少、Jacobi 收敛慢），加速比可能不如高接受率的 speculative。
- 实践常按场景选：有合适 draft 用 speculative，无 draft 用 Medusa/EAGLE，纯算法选 Lookahead。

---
相关: [[高级推理技术]] | [[speculative decoding]] | [[beam search]] | [[KV cache management]] | [[sampling throughput]]
