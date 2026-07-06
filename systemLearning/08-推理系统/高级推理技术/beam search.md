# beam search

> **所属章节**: [[高级推理技术]]
> **所属模块**: [[08-推理系统]]
> **别名**: beam search / 束搜索 / beam search decoding
> **难度**: 中（需懂自回归 + 搜索）


## 1. 一句话定义

**beam search（束搜索）** 是序列解码的**启发式搜索**算法：每步保留概率最高的 **top-$B$**（$B$ = beam width / 束宽）条候选序列，从这些序列各扩展下一步、再剪枝回 top-$B$，如此迭代直到结束，最终输出累积概率最高的那条。它是贪心解码（$B=1$）与穷举搜索（$B=\infty$）的折中——在计算量与输出质量间权衡。

> [!note] 三句话定位
> - **是什么**：每步保 $B$ 条最优候选，扩展-剪枝-重复，取最终最优。
> - **为什么**：贪心每步只取 argmax，可能错过全局更优序列；穷举指数爆炸。beam 折中。
> - **关键参数**：beam width $B$（越大越逼近穷举但越慢）、length normalization（防短序列偏置）。


## 2. 为什么需要它（动机与背景）

### 2.1 三种解码策略的谱系

| 策略 | 每步选择 | 计算量 | 输出 |
|---|---|---|---|
| 贪心 (greedy) | argmax（top-1） | $O(n \cdot V)$ | 局部最优，可能次优 |
| beam search | top-$B$ 保留 | $O(n \cdot B \cdot V)$ | 近似全局最优 |
| 穷举 (exhaustive) | 全部保留 | $O(V^n)$ 指数 | 全局最优 |
| sampling | 按分布采样 | $O(n \cdot V)$ | 多样性强，质量波动 |

$V$ = 词表大小，$n$ = 序列长度。穷举 $V^n$ 不可行（$V=50000, n=20 \to 10^{95}$），贪心易陷局部最优，beam search 是实践中"用可控计算换更好输出"的主流。

### 2.2 贪心为何次优

贪心每步选当前条件概率最高的 token，但局部最优不保证全局最优。经典反例：

- 真最优序列 $A$：第 1 步 token $a$ 概率 0.6，后续路径概率高，总 $0.6 \times 0.9 = 0.54$。
- 次优序列 $B$：第 1 步 token $b$ 概率 0.4，但贪心选了别的... 

实际：贪心在第 1 步可能选了一个 $p=0.7$ 的 token，但其后续路径概率骤降（如 $0.7 \times 0.1 = 0.07$），错过 $p=0.6$ 但后续 $0.9$ 的更优路径。beam search 保 $B$ 条，能"记住"那条 $0.6$ 路径继续扩展。


## 3. 核心概念详解

### 3.1 算法流程

1. 初始化：起始 beam = $[(\text{score}=0, \text{tokens}=\text{prompt})]$。
2. 对每个 beam 的每条候选，算下一步 logits，取 top-$B$ token，扩展成 $B \times B$ 条新候选，score = 旧 score + $\log p(x_i)$。
3. 从 $B \times B$ 候选剪枝回 top-$B$（按 score）。
4. 重复直到所有 beam 遇 EOS 或达 max_len。
5. 输出 score 最高的完整序列。

### 3.2 score：累积对数概率

序列 $x_{1:n}$ 的概率 $P = \prod_{i=1}^n p(x_i | x_{<i})$。连乘会下溢（$V=50000$，每项 $<10^{-4}$，长序列趋 0），所以用**对数域**：

$$
\text{score}(x_{1:n}) = \sum_{i=1}^n \log p(x_i | x_{<i})
$$

beam search 比较的是 $\text{score}$（越大越好）。$\log$ 把连乘变累加，数值稳定。

### 3.3 length normalization（长度归一化）

朴素 score 是 $\sum \log p$，**偏短序列**（累加项少，绝对值小，但更"负"少，反而 score 高）。为避免 beam 倾向短序列，加长度归一化：

$$
\text{score}_{\text{norm}} = \frac{1}{n^\alpha} \sum_{i=1}^n \log p(x_i | x_{<i})
$$

$\alpha \in [0.6, 1]$ 常用。$\alpha=1$ 完全平均，$\alpha=0$ 退化为朴素累加（偏短）。GNMT 取 $\alpha=0.7$。

### 3.4 beam width $B$ 的权衡

| $B$ | 计算 | 质量 | 倾向 |
|---|---|---|---|
| 1 | 最快 | 贪心，次优 | 低延迟 |
| 4–8 | 适中 | 较好 | 翻译/摘要常用 |
| 50+ | 慢 | 接近穷举 | 离线、质量优先 |

$B$ 越大越逼近穷举最优，但计算 $\propto B$，且**收益递减**（$B$ 从 4 到 8 提升明显，从 32 到 64 微弱）。

### 3.5 结束与 EOS 处理

某 beam 生成 EOS 即视为"完成"，移出活跃集存入完成集。当完成集有 $B$ 条或活跃集空，停止。输出完成集中 score 最高者。


## 4. 数学原理 / 公式

### 4.1 目标

求最大化序列概率的序列：

$$
x^* = \arg\max_{x_{1:n}} P(x_{1:n} | \text{prompt}) = \arg\max_{x_{1:n}} \prod_{i=1}^n p(x_i | x_{<i}, \text{prompt})
$$

对数域：

$$
x^* = \arg\max_{x_{1:n}} \sum_{i=1}^n \log p(x_i | x_{<i}, \text{prompt})
$$

### 4.2 beam search 的近似

精确求解需穷举 $V^n$，不可行。beam search 用**贪心剪枝近似**：每步只保 top-$B$，**不保证**找到全局最优 $x^*$（可能最优序列在某步排第 $B+1$ 被剪掉）。

> [!note] 不保证全局最优
> beam search 是启发式，非精确搜索。最优序列若在中间某步的概率排序中排第 $B+1$ 之后，会被剪掉、永远丢失。增大 $B$ 降低这种风险但不消除。它只在"每步 top-$B$ 路径包含真最优"的假设下有效。

### 4.3 计算复杂度

每步：$B$ 个 beam 各算 $V$ 维 logits + top-$B$ → $O(BV)$。$n$ 步共 $O(nBV)$。比贪心 $O(nV)$ 多 $B$ 倍，但远小于穷举 $O(V^n)$。

### 4.4 length-normalized score 的推导

朴素 $\sum \log p$ 偏短：短序列累加项少，绝对值"没那么负"（如长度 5 的 -10 比长度 10 的 -25 "高"），但实际质量可能差。归一化 $\frac{1}{n^\alpha}\sum \log p$ 把"每 token 平均对数概率"作为比较基准，消除长度偏置。$\alpha < 1$ 是在"完全平均"与"偏好长一点"间的调参。


## 5. 代码示例（可选

### 5.1 简化 beam search

```python
import torch
import torch.nn.functional as F

class FakeLM:
    """假模型: logits 由 token 哈希生成, 仅教学."""
    def __init__(self, vocab, seed_base):
        self.vocab, self.seed_base = vocab, seed_base
    def logits(self, tokens):
        torch.manual_seed(self.seed_base + sum(tokens) * 7 + len(tokens))
        return torch.randn(self.vocab)

def beam_search(model, prompt, beam_width=3, max_len=10, eos=0, alpha=0.7):
    """简化 beam search (教学)."""
    beams = [(0.0, list(prompt), False)]  # (score, tokens, finished)
    finished = []
    for step in range(max_len):
        candidates = []
        for score, tokens, done in beams:
            if done:
                candidates.append((score, tokens, True)); continue
            logp = F.log_softmax(model.logits(tokens), dim=-1)
            topk = torch.topk(logp, beam_width)
            for v, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                new_tokens = tokens + [idx]
                # length-normalized score (按当前生成长度归一)
                gen_len = len(new_tokens) - len(prompt)
                norm_score = (score + v) / (gen_len ** alpha)
                is_eos = (idx == eos)
                candidates.append((norm_score, new_tokens, is_eos))
        # 剪枝: 取 top-B, 已完成的单列
        candidates.sort(key=lambda x: x[0], reverse=True)
        active = [c for c in candidates if not c[2]][:beam_width]
        finished += [c for c in candidates if c[2]]
        beams = active if active else finished[:beam_width]
        if not active: break
    all_seqs = finished + [(s, t, True) for s, t, _ in beams]
    all_seqs.sort(key=lambda x: x[0], reverse=True)
    best = all_seqs[0][1]
    return best[len(prompt):]

model = FakeLM(vocab=20, seed_base=10)
out = beam_search(model, prompt=[5,6], beam_width=3, max_len=8, eos=0)
print('beam search out:', out)
```

### 5.2 对比贪心

```python
def greedy(model, prompt, max_len=10, eos=0):
    tokens = list(prompt)
    for _ in range(max_len):
        logp = F.log_softmax(model.logits(tokens), dim=-1)
        nxt = int(logp.argmax())
        tokens.append(nxt)
        if nxt == eos: break
    return tokens[len(prompt):]
print('greedy out:', greedy(model, [5,6], 8, 0))
```


## 6. 与其他知识点的关系

- **上游（依赖）**: autoregressive 生成、logits/log-softmax（数值稳定）。
- **下游（应用）**: 传统 NMT（神经机器翻译）解码、摘要生成；LLM 时代多被 sampling 取代（见 §8）。
- **对比 / 易混**:
  - **beam search vs 贪心**：贪心 $B=1$ 局部最优；beam $B>1$ 近似全局，但非保证。
  - **beam search vs [[speculative decoding]]**：**本质不同**！beam search 是**搜索**（改变输出分布，倾向高概率序列，质量优先）；speculative 是**采样加速**（保持 target 采样分布不变，仅加速）。两者正交：speculative 可配合 beam search（用 draft 加速 beam 扩展）。
  - **beam search vs sampling**：beam 取 top-B（确定性，质量高多样性低）；sampling 按分布采（随机，多样性高质量波动）。LLM 创作场景多用 sampling/temperature。
  - **beam search vs [[parallel decoding]]**：parallel decoding 是并行生成加速，beam search 是搜索提质，目标不同。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 beam search 保证全局最优
> 不保证。它是启发式剪枝，最优序列若在某步排第 $B+1$ 之后会被剪掉。增大 $B$ 降低风险但不消除。只有 $B=\infty$（穷举）才保证最优。

> [!warning] 误区 2：忽略 length normalization 导致偏短
> 朴素 $\sum \log p$ 让短序列 score 虚高（累加项少）。不归一化时 beam search 倾向输出过短序列。务必加 $\frac{1}{n^\alpha}$ 归一。

> [!warning] 误区 3：以为 LLM 生成都用 beam search
> 实际相反。LLM 时代（对话/创作）beam search **大量被 sampling（temperature/top-p/top-k）取代**——beam 输出"通用、平庸、多样性低"（high-probability 但 dull），而 sampling 更自然有创意。beam 现多用于翻译、摘要等"有标准答案"的任务，或代码生成等质量优先场景。

> [!warning] 误区 4：beam width 越大越好
> 收益递减。$B$ 从 1 到 4 提升明显，从 16 到 32 微弱。过大 $B$ 增计算且可能"过度优化"出 dull 序列。实践中 $B=4$–$8$ 是甜点。

> [!warning] 误区 5：beam search 与 speculative decoding 混淆
> 两者目标不同：beam 改变输出分布（搜索高概率），speculative 保持分布（采样加速）。把 speculative 当"并行 beam search"是错的。


## 8. 延伸细节

### 8.1 LLM 时代 beam search 的衰退

GPT 系列后，对话/创作多用 sampling（temperature + top-p）而非 beam。原因：

- beam 倾向 high-probability 但 dull/通用文本，缺乏创意与多样性。
- LLM 模型本身已很强，贪心/sampling 输出质量够用，beam 的搜索收益相对下降。
- beam 的 $B$ 倍计算在 LLM（大模型、长序列）下开销显著。

但 beam 在**结构化输出**（代码、SQL）、**有参考的任务**（翻译、摘要）、**确定性需求**（同 prompt 同输出）场景仍占优。

### 8.2 diverse beam search

为缓解 beam 输出多样性低，diverse beam search（Vijayakumar et al. 2016）把 $B$ 个 beam 分组，组间加**多样性惩罚**（Hamming 距离），让不同组探索不同子空间。用于多候选输出场景。

### 8.3 constrained beam search

加入硬约束（必须包含某 token / 满足正则），用于可控文本生成（如必须输出特定实体）。HuggingFace `ConstrainedBeamSearch` 支持。

### 8.4 与 RLHF 的关系

RLHF 训练中 [[rollout worker]] 采样多用 **sampling**（temperature > 0）以保证多样性（探索），beam search 因确定性（无随机性）不利于 RL 探索。但评估阶段可用 beam search 生成"参考最优"序列做对比。speculative decoding 更适合 RLHF 采样加速（保持采样分布）。

### 8.5 与 vLLM/TGI 的支持

vLLM 主要面向采样加速（continuous batching + speculative），beam search 支持有限（因 LLM 时代需求下降）。HuggingFace `generate` 默认支持 beam search。TGI 对 beam 支持也有限。这反映 LLM 时代 beam 的边缘化。

---
相关: [[高级推理技术]] | [[speculative decoding]] | [[parallel decoding]] | [[sampling throughput]]
