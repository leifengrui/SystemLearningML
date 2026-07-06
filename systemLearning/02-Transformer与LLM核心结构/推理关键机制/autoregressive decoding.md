# autoregressive decoding

> **所属章节**: [[推理关键机制]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中级

## 1. 一句话定义

**自回归解码（autoregressive decoding，自回归生成，自回归解码）** 是 LLM 生成的标准范式：每一步用**已生成的全部 token**（初始为 prompt）作为输入，前向得到下一词的 logits，经 [[采样策略]] 选出一个词，append 回输入序列，循环往复直到 `<eos>` 或达 max_len；因 [[causal mask]] 的因果性，模型只能"看前文预测下一词"，**无法并行输出整段**，必须逐词循环——这是 LLM 生成慢的根本原因，[[KV cache]] 把每步算力从重算全部历史降到只算新位置。

> [!note] 解答：为什么要"采样一个词 + append 回输入"循环？
> 
> 这是**因果性 + 概率生成**两条根因共同决定的，拆开看：
> 
> **1. 为什么必须逐词、不能一次出整段？——因果性**
> 模型训练学的是 $p(x_t\mid x_{<t})$（看前文预测下一词，见 [[#2. 为什么需要它（动机与背景）]]）。生成时第 $t+1$ 词的预测**必须以第 $t$ 词为前提**，而第 $t$ 词在算完第 $t$ 步之前根本不存在。所以无法预先并行算第 $t+1$ 词——**必须先算出第 $t$ 词，才能算第 $t+1$ 词**。这是自回归慢的根因，[[speculative decoding]] 用"草稿猜 + 验证"部分打破它。
> 
> **2. 为什么是"采样"出一个词，而不是 argmax / 直接取分布？**
> - 模型输出的是**下一词的概率分布** $p\in\Delta^V$（softmax 后，见 [[logits generation]]），不是一个确定的词。要把分布变成一个具体词 id，两种方式：
>   - **argmax（greedy）**：取概率最大的词，确定性强，但易重复、平庸、缺多样性。
>   - **采样**：按概率随机抽一个词（配 temperature/top-k/top-p 控制分布形态，见 [[采样策略]]）。
> - 采样的意义：**忠于模型给的概率分布**——模型说"下一词 70% 是'猫'、30% 是'狗'"，采样就有 30% 概率出"狗"，生成更自然、更多样、更有创造力；argmax 永远出"猫"，长程易陷入重复循环。
> - 数学上，采样是从模型分布 $p_\theta(x_{t+1}\mid x_{\le t})$ 里抽样本，这正是"从模型分布生成文本"的定义。
> 
> **3. 为什么要 append 回输入序列？**
> 下一步 forward 要算 $p(x_{t+2}\mid x_{\le t+1})$，输入必须是**含刚生成词 $x_{t+1}$ 的完整前文**。所以采样出的 $x_{t+1}$ 要拼回输入序列，形成闭环：
> 
> $$\text{forward}(x_{\le t}) \to p(x_{t+1}) \xrightarrow{\text{sample}} x_{t+1} \xrightarrow{\text{append}} x_{\le t+1} \to \text{forward}(x_{\le t+1}) \to \cdots$$
> 
> 配合 [[KV cache]]，append 实际只需把 $x_{t+1}$ 的 K/V 加进 cache，下一步只用新 Q 对 cache 算 attention，不必重算历史（见 [[KV cache]] §5 代码）。
> 
> **4. 何时停？**
> 循环到生成 `<eos>`（结束符）或达 max_len 才停。模型自己学会在内容结束时输出 `<eos>`，这是 [[tokenization]] 里定义的 special token。
> 
> **一句话**：因果性 → 必须逐词；概率分布 → 要采样而非 argmax；下一步要完整前文 → 要 append 回输入；三者串成 `前向→logits→采样→append→前向` 的自回归闭环，直到 `<eos>`。

## 2. 为什么需要它（动机与背景）

[[decoder-only architecture]] 的训练目标是"看 $x_{<t}$ 预测 $x_t$"，因果性决定生成时**第 $t$ 词必须依赖第 $1..t-1$ 词**：

- 不能一次性输出整段（第 2 词要等第 1 词确定才能算）。
- 只能**逐词循环**：算出第 $t$ 词 → 拼回输入 → 算第 $t+1$ 词。

这与训练时的 teacher forcing（用真实前文并行算所有位置 loss）不同：推理时前文是**模型自己刚生成的词**，必须串行。这个串行性是 LLM 生成延迟的根源，催生 KV cache、speculative decoding、continuous batching 等一系列优化。

## 3. 核心概念详解

### 3.1 两阶段：prefill + decode

- **prefill**：处理 prompt 一次，算所有 prompt 位置的 logits 与 KV cache。通常这一步就能输出第一个生成词。compute-bound（并行算多 token）。
- **decode**：逐词循环，每步 n_new=1，用 KV cache 只算新位置，取最后 logits → 采样 → append。memory-bound（每步算量小但读全部权重与 cache）。

### 3.2 单步 forward 的流程

decode 每步：

1. 上一步选出的 token id → embedding（+ [[位置编码]]，RoPE 在 Q/K 作用）。
2. 逐层：新 token 的 Q/K/V，K/V append 进 [[KV cache]]，新 Q 与 cache K 做 attention（causal 天然满足），FFN。
3. final norm → [[logits generation]] → 取**最后位置**的 logits。
4. [[采样策略]]（greedy/top-k/top-p+temperature）选下一词。
5. append 到序列；若 `<eos>` 或达 max_len 则停。

### 3.3 训练 vs 推理的区别

| 维度 | 训练 | 推理 |
|---|---|---|
| 前文来源 | 真实数据（teacher forcing） | 模型自生成 |
| 并行性 | 一次前向算所有位置（causal mask） | 逐词串行 |
| 目标 | 算所有位置 CE loss | 生成下一词 |
| KV cache | 不用（每步重算全序列，但一次前向） | 必用（避免重算历史） |

### 3.4 stopping criteria（停止条件）

- `<eos>`（end-of-sequence）token：模型主动输出结束符。
- max_len：达到最大长度上限。
- stop string：匹配特定文本（如 `\n\nuser:`）。
- EOS 概率阈值等自定义条件。

### 3.5 算力特征与瓶颈

- **prefill**：算量大（一次算 $n_{\text{prompt}}$ 个 token），GPU 算力打满 → **compute-bound**。吞吐由 FLOPS 决定。
- **decode**：每步算量小（1 个 token），但要读全部权重 + cache → **memory-bound**。吞吐由显存带宽决定，GPU 算力闲置。
- 这导致 decode 阶段 GPU 利用率低，是推理服务优化的重点（batch 凑数、speculative decoding 等）。

## 4. 数学原理 / 公式

### 自回归的联合概率

生成序列 $x_{1..n}$ 的概率按链式分解：

$$
P(x_{1..n}) = \prod_{t=1}^n P(x_t \mid x_{<t})
$$

每步模型输出 $P(\cdot\mid x_{<t})$（softmax of logits），采样得 $x_t$。因果性使分解成立——这正是 [[causal mask]] + next-token 训练目标的直接体现。

### 算力（有 KV cache）

- prefill：$O(n_0^2 L d)$（attention 是 $O(n_0^2)$）。
- decode 每步：$O(n\cdot L d)$（新 Q 对 cache K 的 attention 是 $O(n)$，FFN 是 $O(d^2)$ 但与 $n$ 无关；主要带宽读权重）。
- 总：$O((n_0^2 + n\cdot n_{\text{gen}}) L d)$，相比无 cache 的 $O(n^3)$ 大幅降低。

### decode 的 memory-bound 本质

每步算量 $\approx 2P_{\text{params}}$ FLOPs（前向一次模型），但要读 $P_{\text{params}}\cdot b$ 字节权重。算术强度 $\approx 2/b$（fp16 时 1 op/byte）远低于 GPU 的峰值比例（如 A100 312TFLOPS / 2TB/s ≈ 150）→ 严重 memory-bound，算力闲置。

## 5. 代码示例

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, tokenizer, prompt_ids, max_new=20, temperature=0.8, top_p=0.9):
    """极简自回归生成（贪心/采样，无 KV cache 简化版）。"""
    ids = prompt_ids.clone()
    for _ in range(max_new):
        logits = model(ids)                       # (1, n, V)  真实系统用 KV cache
        next_logits = logits[0, -1]               # 取最后位置
        # temperature + top-p 采样
        z = next_logits / temperature
        probs = F.softmax(z, dim=-1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = sorted_p.cumsum(-1)
        keep = cum - sorted_p < top_p             # 最小集合使累计>=p
        sorted_p[~keep] = 0
        sorted_p /= sorted_p.sum()
        nxt = sorted_i[torch.multinomial(sorted_p, 1)]
        if nxt.item() == tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, nxt[None]], dim=-1)
    return ids

# 真实系统用 KV cache：prefill 一次，decode 每步 past_key_values 增量
# from transformers import AutoModelForCausalLM
# out = model.generate(input_ids, use_cache=True, do_sample=True,
#                      temperature=0.8, top_p=0.9, max_new_tokens=20)
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[decoder-only architecture]]（自回归的主体）、[[causal mask]]（因果性使自回归成立）、[[logits generation]]（每步取 logits）、[[KV cache]]（避免重算历史）。
- **下游（应用）**: [[采样策略]]（每步如何选词）、推理服务系统（vLLM/TGI 的核心调度对象）、speculative decoding 等加速。
- **对比 / 易混**:
  - **自回归 vs 并行解码**：自回归串行，non-autoregressive（NAR）尝试并行输出（翻译领域有，LLM 生成少见）。
  - **autoregressive decoding vs teacher forcing**：见 3.3，推理自生成 vs 训练真实前文。
  - **prefill vs decode**：两阶段，算力特征相反。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为可以并行输出整段**：因果性决定逐词串行，第 $t$ 词依赖前 $t-1$ 词，无法并行（除非 speculative decoding 等近似）。
> 2. **逐词重算全序列**：不用 KV cache 每步重算历史，算力平方增长，完全不可行。必须用 cache。
> 3. **取错 logits 位置**：要取**最后位置**的 logits 预测下一词，不是取平均或首位。
> 4. **prefill/decode 不分**：两者算力特征相反（compute-bound vs memory-bound），服务调度要分别对待。
> 5. **忘 stopping**：不设 `<eos>`/max_len 会无限生成或到上下文上限才停。
> 6. **训练/推理前文不一致**：训练 teacher forcing 用真实前文，推理用自生成；若训练时不加 [[causal mask]] 让模型看到未来，推理会崩（exposure bias）。
> 7. **以为 decode 打满 GPU**：decode memory-bound，算力闲置，单请求吞吐低；要靠 batching 凑吞吐。

## 8. 延伸细节

### 8.1 speculative decoding（投机解码）

用小模型先**并行草拟**多个候选词，大模型一次验证多个，接受正确的前缀 → 把部分串行变并行，加速 decode。大幅降延迟，质量无损（拒绝采样保证等价）。是当前 LLM 推理加速的主流方向之一。

### 8.2 continuous batching

decode 阶段不同请求生成步数不同，静态 batch 会因短请求提前结束而空转。continuous batching 动态拼批（请求进出不阻塞），配合 PagedAttention 把 decode 阶段的 memory-bound 瓶颈转化为高吞吐。vLLM/SGLang 的核心。

### 8.3 Medusa / EAGLE

在模型头加多个并行预测头，一次 forward 预测多个未来词，配合树状验证 → 类似 speculative 但无需独立小模型。

### 8.4 exposure bias

训练用真实前文、推理用自生成，前文分布不同导致误差累积。缓解：scheduled sampling、RL 微调等，但大规模 LLM 实践中影响有限，主流靠规模与指令微调。

### 8.5 chunked prefill

超长 prompt 的 prefill 一次算会打爆显存，分块 prefill（chunk）边算边存 cache，与 decode 交错调度，是长 prompt 服务的关键。

### 8.6 上下文长度上限

自回归生成时 $n_0 + n_{\text{gen}}$ 不能超模型支持的最大长度（受 [[位置编码]] 外推与显存限制）。超长需 RoPE 外推工程或 cache 压缩。

---
相关: [[推理关键机制]]、[[KV cache]]、[[采样策略]]、[[causal mask]]、[[logits generation]]、[[decoder-only architecture]]
