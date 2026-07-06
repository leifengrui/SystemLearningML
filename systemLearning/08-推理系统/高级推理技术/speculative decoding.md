# speculative decoding

> **所属章节**: [[高级推理技术]]
> **所属模块**: [[08-推理系统]]
> **别名**: speculative decoding / 推测解码 / 投机解码 / speculative sampling
> **难度**: 高（需懂 [[KV cache management]] + 拒绝采样 + 分布对齐）


## 1. 一句话定义

**speculative decoding（推测解码）** 是 LLM 推理的**无损加速**技术：用一个小的 **draft model** 快速串行生成 $k$ 个候选 token，再用大 **target model** **一次 forward 并行验证**这 $k$ 个候选（接受概率符合 target 分布的、拒绝并重采样不符合的），从而把"每步生成 1 token"加速到"每轮生成 $\approx k \cdot \text{接受率}$ 个 token"，且**最终输出分布与纯 target 模型完全相同**（lossless）。由 Leviathan et al. 2023（Google）与 Chen et al. 2023（DeepMind）独立提出。

> [!note] 三句话定位
> - **是什么**：小模型先猜、大模型并行验、按概率接受-拒绝。
> - **为什么**：大模型 decode 是 memory-bound，单步算力闲置；用小模型猜 + 大模型一次并行验，摊薄访存成本。
> - **关键性质**：lossless——输出分布严格等于 target 模型分布（不是"近似无损"，是数学上严格相等）。


## 2. 为什么需要它（动机与背景）

### 2.1 decode 阶段的 memory-bound 困境

LLM 自回归解码每步生成 1 个 token，需要：

- 读**整个 [[KV cache management]]**（所有历史层 K/V）进 SRAM 算 attention；
- 但只算 1 个 token 的 forward（计算量极小）。

这导致 decode 是 **memory-bound**：瓶颈在 HBM 带宽而非算力。大模型单步 forward 只用了 GPU 算力的极小一部分（可能 < 1%），算力大量闲置。

> [!tip] 关键观察
> 串行 decode $n$ 步 = $n$ 次 memory-bound forward，每次都为 1 个 token 读整个 cache。**若能让大模型一次 forward 处理多个 token**，就能把访存成本摊到多个 token 上，提升算力利用率。speculative decoding 正是利用这一点——让大模型一次 forward 验证 $k$ 个候选 token（并行计算 $k$ 个位置的 logits），相当于"用 1 次访存换 $k$ 个 token 的验证"。

### 2.2 为什么不直接并行生成

标准自回归依赖前文（token $t$ 依赖 $t-1$），无法天然并行。speculative 的巧思：用**小模型**串行猜 $k$ 个（小模型快、cache 小），再让**大模型**一次 forward 并行验证这些位置——大模型验证 $k$ 个位置只需 1 次 forward（而非 $k$ 次），因为验证是"给定前缀，看每个位置 target 的分布"。

### 2.3 与朴素并行的区别

- 朴素并行：无法做（自回归依赖）。
- speculative：小模型"预演"一条路径，大模型"批验"该路径上的若干位置。


## 3. 核心概念详解

### 3.1 两阶段：draft 生成 + target 验证

**Stage 1 — draft 生成（串行，小模型快）**：
draft model（小，如 1B）从当前序列串行生成 $k$ 个候选 token $x_1, \dots, x_k$，每步 $x_i \sim q(\cdot | x_{<i})$，并记录 $q(x_i | x_{<i})$。

**Stage 2 — target 验证（并行，大模型一次 forward）**：
target model（大，如 70B）对序列 $[\text{prefix}, x_1, \dots, x_k]$ 做一次 forward，得到每个位置的分布 $p(\cdot | x_{<i})$（$i=1..k+1$，最后一个位置是"全部接受后的 bonus"）。

**Stage 3 — 接受-拒绝**：
对每个 $i$，按接受概率决定是否接受 $x_i$：
- 接受 → 继续验证下一个。
- 拒绝 → 从修正分布重采样 1 个 token，终止本轮。

接受 $a$ 个后，本轮产出 $a$ 个（被接受的）+ 1 个（bonus 或重采样）= $a+1$ 个 token。

### 3.2 接受-拒绝规则（rejection sampling）

对 draft 采的 $x_i \sim q$，target 有 $p$：

$$
\text{accept } x_i \text{ 以概率 } \min\!\left(1, \frac{p(x_i)}{q(x_i)}\right)
$$

- 若 $p(x_i) \ge q(x_i)$：**必接受**（target 比 draft 更倾向这个 token）。
- 若 $p(x_i) < q(x_i)$：以 $p/q$ 概率接受，$1 - p/q$ 拒绝。

拒绝时，从修正分布重采样：

$$
x' \sim \frac{\max(p - q, 0)}{\sum_x \max(p(x) - q(x), 0)} = \frac{(p - q)_+}{\mathbb{E}[\max(1-p/q,0)] \cdot \dots}
$$

即归一化的 $(p - q)_+$（target 比 draft 多出的"余量"）。

### 3.3 bonus token（全接受奖励）

若 $k$ 个全部被接受，target 在第 $k+1$ 个位置的分布 $p(\cdot | x_{<k+1})$ 可直接采样一个 bonus token——这是"免费"的，因为 target forward 已经算了这个位置。所以全接受时产出 $k+1$ 个 token。

### 3.4 lossless 性质

通过接受-拒绝 + 重采样的修正，最终每个 token 的输出分布**严格等于** target 分布 $p$（证明见 §4.2）。所以 speculative decoding 不是"近似无损"，而是**数学上严格无损**——用 speculative 与纯 target 采样，得到的序列分布完全相同（只是具体随机数不同）。


## 4. 数学原理 / 公式

### 4.1 接受概率与期望接受长度

draft 采 $x \sim q$，接受概率：

$$
P(\text{accept}) = \mathbb{E}_{x \sim q}\!\left[\min\!\left(1, \frac{p(x)}{q(x)}\right)\right] = \sum_x \min(p(x), q(x))
$$

> [!note] 推导
> $\mathbb{E}_{x \sim q}[\min(1, p/q)] = \sum_x q(x) \min(1, p(x)/q(x)) = \sum_x \min(q(x), p(x))$。这正是 $p$ 与 $q$ 的**总变差**关系：$P(\text{accept}) = 1 - \text{TV}(p, q) = \sum_x \min(p, q)$。

单步接受概率 $\alpha = \sum_x \min(p, q) \in [0, 1]$。$k$ 个位置中接受 $a$ 个的期望：

$$
\mathbb{E}[a] = \sum_{i=1}^{k} P(\text{接受前 } i \text{ 个}) = \sum_{i=1}^{k} \alpha^{i} \cdot (1-\alpha) \cdot i + \alpha^{k} \cdot k
$$

简化（每步独立近似）：$\mathbb{E}[a] \approx \frac{\alpha(1-\alpha^k)}{1-\alpha}$，当 $\alpha \to 1$ 时 $\mathbb{E}[a] \to k$。

产出 token 数 $\approx \mathbb{E}[a] + 1$（bonus 或重采样）。

### 4.2 lossless 证明（重加权）

要证明：最终每个 token 的边缘分布 $\pi(x) = p(x)$。

考虑一轮中产出第一个 token（位置 1）的情况：

- 以概率 $\min(1, p(x_1)/q(x_1))$ 接受 $x_1 \sim q$。
- 否则（概率 $1 - \min(1, p/q)$）从 $(p-q)_+$ 重采样 $x'$。

接受 $x_1$ 的概率（$x_1$ 任意）：

$$
P(\text{产出 } x) = q(x) \cdot \min\!\left(1, \frac{p(x)}{q(x)}\right) + \underbrace{(1 - \alpha)}_{\text{拒绝总概率}} \cdot \frac{(p(x) - q(x))_+}{\sum_y (p(y) - q(y))_+}
$$

其中 $\alpha = \sum_y \min(p(y), q(y))$，且 $\sum_y (p(y) - q(y))_+ = 1 - \alpha$（因 $\sum p = \sum q = 1$，正部分之和 = 负部分绝对值之和 = $\frac{1 - \alpha}{1}$）。

代入：

$$
P(\text{产出 } x) = \min(q(x), p(x)) + (1 - \alpha) \cdot \frac{(p(x) - q(x))_+}{1 - \alpha}
$$

$$
= \min(q, p) + (p - q)_+ = \begin{cases} q + (p - q) = p & \text{若 } p \ge q \\ q + 0 = q \cdot \dots \end{cases}
$$

仔细分情况：
- 若 $p(x) \ge q(x)$：$\min(q, p) = q$, $(p-q)_+ = p - q$，和 $= q + (p-q) = p$。✓
- 若 $p(x) < q(x)$：$\min(q, p) = p$, $(p-q)_+ = 0$，和 $= p + 0 = p$。✓

两情况都 $= p(x)$。故 $\pi(x) = p(x)$，**严格无损**。后续位置同理（条件分布对齐）。

### 4.3 加速比

设 target 单次 forward 耗时 $T_t$（memory-bound，与位置数弱相关），draft 生成 $k$ 个串行耗时 $k \cdot T_d$（$T_d \ll T_t$ 因小模型）。一轮产出 $\mathbb{E}[a] + 1$ 个 token，耗时 $T_t + k T_d$。

加速比：

$$
S = \frac{(\mathbb{E}[a] + 1) \cdot T_t}{T_t + k T_d} = \frac{\mathbb{E}[a] + 1}{1 + k \cdot T_d / T_t}
$$

- 接受率高（$\alpha \to 1$）、draft 快（$T_d/T_t$ 小）→ 加速显著（可达 $2$–$3\times$）。
- draft 与 target 分布差异大（$\alpha$ 低）→ 接受少、draft 开销白花，可能负优化。

> [!warning] 加速取决于接受率
> speculative 不是免费午餐。若 draft 太弱（$\alpha$ 低），每轮接受 0–1 个，draft 的 $k T_d$ 开销纯浪费。实践要求 $\alpha$ 至少 0.5+ 才有正收益。draft 选型（与 target 同源、同 tokenizer、分布相近）是关键。


## 5. 代码示例（可选

### 5.1 简化 speculative decoding（接受-拒绝核心逻辑）

```python
import torch
import torch.nn.functional as F

class FakeLM:
    """假模型: logits 由 token 哈希生成, 仅教学."""
    def __init__(self, vocab, seed_base):
        self.vocab, self.seed_base = vocab, seed_base
    def logits(self, tokens):
        torch.manual_seed(self.seed_base + sum(tokens) * 7)
        return torch.randn(self.vocab)

def speculative_decode(target, draft, prompt, k=4, max_new=30):
    tokens = list(prompt)
    produced = 0
    while produced < max_new:
        # 1. draft 串行生成 k 个候选 + 记录 draft 概率
        draft_tok, draft_prob = [], []
        cur = tokens[:]
        for _ in range(k):
            p = F.softmax(draft.logits(cur), dim=-1)
            nxt = torch.multinomial(p, 1).item()
            draft_tok.append(nxt); draft_prob.append(p[nxt].item())
            cur = cur + [nxt]
        # 2. target 一次 forward 验证 (教学: 逐位置调, 实际是一次 forward)
        n_accept = 0
        for i in range(k):
            t_p = F.softmax(target.logits(tokens + draft_tok[:i+1]), dim=-1)
            t_prob = t_p[draft_tok[i]].item()
            d_prob = draft_prob[i]
            # 接受概率 min(1, p/q)
            if t_prob >= d_prob:
                accept = True
            else:
                accept = torch.rand(1).item() < (t_prob / max(d_prob, 1e-8))
            if accept:
                n_accept += 1
            else:
                break
        # 接受的 token 入序列
        tokens = tokens + draft_tok[:n_accept]
        produced += n_accept
        # 3. 拒绝处: 从 (target - draft)+ 重采样; 全接受: bonus token
        if n_accept < k:
            t_p = F.softmax(target.logits(tokens), dim=-1)
            d_p = F.softmax(draft.logits(tokens), dim=-1)
            adjusted = (t_p - d_p).clamp(min=0); adjusted = adjusted / (adjusted.sum() + 1e-8)
            tokens = tokens + [torch.multinomial(adjusted, 1).item()]; produced += 1
        else:
            t_p = F.softmax(target.logits(tokens), dim=-1)
            tokens = tokens + [torch.multinomial(t_p, 1).item()]; produced += 1
    return tokens[len(prompt):]

target = FakeLM(vocab=100, seed_base=42)
draft = FakeLM(vocab=100, seed_base=7)
out = speculative_decode(target, draft, prompt=[1,2,3], k=4, max_new=20)
print('generated:', out, 'len:', len(out))
```

### 5.2 关键点注释

- `draft.logits(cur)` 串行 $k$ 次（小模型快）；`target.logits(...)` 实际生产中是**一次 forward** 算 $k+1$ 个位置（教学版逐次调用是为清晰）。
- 接受规则 `min(1, p/q)`：$p \ge q$ 必接受，否则按比例。
- 拒绝重采样从 `(target - draft).clamp(min=0)` 归一化（即 $(p-q)_+$），保证 lossless。
- 全接受有 bonus token（target 已算末位）。

### 5.3 验证 lossless（分布对齐实验，概念）

```python
# 概念验证 (不跑): 采样 N 次, 统计首 token 频率
# speculative 采样 与 纯 target 采样 的频率分布应一致 (随机误差内)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[KV cache management]]（draft 和 target 各自维护自己的 KV cache，target 验证时要为 $k+1$ 个位置算 KV）、[[attention]]（target forward 的算子）、autoregressive 生成。
- **下游（应用）**: [[batching tradeoff]]（speculative 在 batch 调度中的位置）、[[GPU utilization]]（speculative 提升 decode 算力利用率）、[[sampling throughput]]（RLHF 中 [[rollout worker]] 用 speculative 加速采样）、[[policy deployment (PD)]]（PD 服务降延迟）。
- **对比 / 易混**:
  - **speculative vs [[beam search]]**：beam search 是**搜索**（保 top-k 候选序列，追求高概率输出）；speculative 是**采样加速**（保持采样分布，不改变输出质量，只是更快）。beam 改变输出分布（贪心倾向），speculative 不改变。
  - **speculative vs [[parallel decoding]]**：speculative 用 draft 串行猜 + target 并行验；parallel decoding 用多种并行策略（如 Medusa 多头同时预测多位置）直接并行生成，不依赖 draft model。
  - **speculative vs 朴素蒸馏**：蒸馏让小模型逼近大模型输出（永久替换）；speculative 让小模型辅助大模型加速（大模型仍是最终裁判，分布不变）。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 speculative 是"近似无损"
> 不是近似，是**严格无损**。通过接受-拒绝 + $(p-q)_+$ 重采样，输出分布数学上等于 target 分布 $p$（§4.2 证明）。同一 prompt 采样，speculative 与纯 target 得到的序列分布完全相同（仅随机数不同）。它加速但不改变输出质量。

> [!warning] 误区 2：以为任意小模型都能当 draft
> draft 必须与 target **分布相近**（高 $\alpha$）才有正收益。若 draft 太弱（$\alpha$ 低），接受率低，draft 的 $k$ 步开销纯浪费，甚至负优化。draft 选型要求：同 tokenizer、同训练数据/任务、规模适配（如 7B target 配 1B draft，70B 配 7B draft）。

> [!warning] 误区 3：以为 bonus token 是"送的"忽略其条件
> bonus token 来自 target 在"全部接受后"的下一位置预测 $p(\cdot | x_{<k+1})$，必须从 $p$ 采样（不能贪心 argmax，否则破坏 lossless）。它是 target forward 的副产品，但采样规则要遵守 $p$。

> [!warning] 误区 4：忽略 draft 的 KV cache 管理
> draft 串行生成 $k$ 个时也要维护自己的 KV cache。每轮结束后若 draft cache 失效要重算。工程上 draft 的 cache 管理也是开销来源。

> [!warning] 误区 5：以为加速比固定
> 加速比随 $\alpha$（分布对齐度）与 $k$（draft 长度）动态变化。$k$ 太大时若 $\alpha$ 不高，后面位置接受率衰减（$\alpha^i$），浪费 draft 算力。典型 $k=4$–$8$。


## 8. 延伸细节

### 8.1 Medusa（无 draft model 的并行验证）

Medusa（Cai et al. 2024）不另起 draft model，而是在 target 模型上加**多个预测头**（Medusa heads），每个头并行预测"未来第 $t$ 个位置"的 token。一次 target forward 即得多个候选位置，配合 tree attention 验证。省去独立 draft model 的开销与 cache 管理，但需训练 Medusa 头。

### 8.2 EAGLE

EAGLE（Li et al. 2024）让 draft 模型基于 target 的**中间表示（隐藏态）**而非 token 预测下一状态，draft 与 target 对齐更紧、接受率更高（$\alpha$ 可达 0.7+）。

### 8.3 self-speculative / 层跳过

同一模型用"跳过部分层"做 draft（轻量版本）、完整层做 target，省去独立 draft model。如 LayerSkip。

### 8.4 Lookahead Decoding

不依赖 draft model，用 Jacobi 迭代并行解出多个 token（把自回归看作不动点迭代）。无 draft cache 开销但实现复杂。

### 8.5 与 RLHF 的关系

[[rollout worker]] 采样是 RLHF 训练的主要时间开销。speculative decoding 在 PD 侧加速采样能显著提升 [[sampling throughput]]，缩短 RLHF 训练周期。但注意：RLHF 训练中 PD 侧模型 $\theta$ 在变（[[weight sync mechanism]]），draft 模型需与当前 $\theta$ 对齐，否则接受率下降。verl/OpenRLHF 等框架已集成 speculative 采样选项。

### 8.6 vLLM / TGI 的支持

vLLM 支持 speculative decoding（通过 `--speculative_model` 指定 draft），TGI 也内置。需 draft 与 target 同 tokenizer、同设备。tree attention（把 draft 候选组织成树并行验证）进一步提升效率。

---
相关: [[高级推理技术]] | [[KV cache management]] | [[beam search]] | [[parallel decoding]] | [[sampling throughput]]
