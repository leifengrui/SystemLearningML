# iterative 与 parallel decoding

> **所属章节**: [[Diffusion LLM]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: iterative decoding / parallel decoding / coarse-to-fine decoding / block decoding / non-autoregressive decoding for dLLM / 并行 token 生成
> **难度**: 中高（需懂 [[autoregressive decoding]] + [[Diffusion LLM训练目标]] + [[speculative decoding]] + [[采样策略]]）


## 1. 一句话定义

**iterative 与 parallel decoding** 是 Diffusion LLM 的两种**去噪解码方式**——**iterative decoding**（迭代去噪）：从全噪声 token 序列（全 `[mask]` 或全随机向量）出发，多步（$T$ 步）迭代去噪，**每一步同时 refine 全部/部分未确定 token**，coarse-to-fine（从粗到细，早期步定大结构、后期步修细节），最后收敛到干净文本；**parallel decoding**（并行解码）：每个去噪步**一次并行生成/提交多个 token**（非 AR 的逐 token 串行），是 iterative 解码"每步并行提交"的体现，常配合 block-wise（块级）提交或 confidence-based（置信度门控）提交。**与 AR decoding 的根本区别**：AR 是**串行 $L$ 步、每步 1 token**（左到右、固定顺序、有 [[KV cache]]）；Diffusion 是**并行 $T$ 步、每步可提交 $k$ 个 token**（$T\ll L$、任意顺序、双向 attention）。**核心 trade-off**：步数 $T$ 多→质量高但慢；步数 $T$ 少→快但去噪不充分质量降（factorization error）。**与 [[speculative decoding]] 区别**：speculative 是 **AR 的加速**（draft 多 token + verify，仍保 AR 左到右顺序与最终分布），Diffusion 是**真并行 + 非左到右**（不保 AR 分布）。两者可混用（AR 当 draft + Diffusion refine）。是 [[Diffusion LLM]] 推理侧的核心——训练目标（[[Diffusion LLM训练目标]]）决定可用的解码方式。

> [!note] 三句话定位
> - **是什么**：iterative = 多步迭代去噪全序列 coarse-to-fine；parallel = 每步并行提交多 token。$T$ 步 $\ll L$ 即可出文。
> - **为什么**：AR 串行 $L$ 步慢、顺序死；Diffusion 并行 $T$ 步、任意顺序、可 infill。
> - **与 [[speculative decoding]] 区别**：spec 是 AR 加速保顺序保分布；Diffusion 是真并行非左到右。可混用（AR draft + Diffusion refine）。


## 2. 为什么需要它（动机与背景）

### 2.1 AR 串行瓶颈

[[autoregressive decoding]] 生成 $L$ token 需 $L$ 次 forward，串行（第 $i$ token 等 $i-1$）。长序列（$L=8k$）延迟线性增长。虽每步只 1 token forward（[[KV cache]] 复用、memory-bound），但**步数 $L$ 不可降**。

### 2.2 Diffusion 的并行承诺

Diffusion 训练时学的是"给定噪声态预测干净 token"（见 [[Diffusion LLM训练目标]]），**每个 token 的预测不依赖"上一个 token 已生成"**，而是依赖整段噪声上下文。所以推理时可**一次预测多个 token**。若 $T$ 步、每步提交 $k$ token，$T\cdot k \geq L$ 即可出文，$T\ll L$ 时比 AR 快。

### 2.3 但有 factorization error

并行一次提交多 token 时，模型预测每个 token 时**假设其他 token 也是噪声态**，但实际部分已确定——这种"联合分布被独立边缘分布近似"引入 **factorization error**（因子化误差）。步数太少、一次提交太多 → 误差大、质量崩。所以**步数是 trade-off**：多步逐步收敛、少步快但糙。

### 2.4 两种调度：iterative vs parallel

- **iterative**：强调"多步迭代 refine 全序列"，每步可能只改一小部分（confidence 门控），coarse-to-fine。
- **parallel**：强调"每步一次提交多个 token"，block-wise 或 confidence-based 批量提交。

实际两者交织：iterative 过程中每步并行提交 = parallel。社区常混称。


## 3. 核心概念详解

### 3.1 iterative decoding 流程

```
init: x_T = [mask]*L  (全 mask)  或  连续纯噪声
step 1: forward(x_T) -> 预测各 mask 位置的 token 分布
        commit 最 confident 的若干 (或一定比例)
step 2: forward(部分确定的 x_{T-1}) -> refine 剩余 mask
        commit 更多
...
step T: 全部 commit -> x_0 (生成文本)
```

每步 forward 全序列（双向 attention），coarse-to-fine：早期步定大结构（高频词、骨架），后期步修细节（低频、精确词）。

### 3.2 parallel decoding 的提交策略

每步"提交哪些 token"是关键调度：

- **uniform（均匀）**：每步提交 $L/T$ 个（按位置均匀选）。简单但可能提交不该提交的。
- **confidence-based（置信门控）**：每步预测后，提交模型最 confident 的 $k$ 个（top-$k$ 置信度）。最常见、质量最好。
- **block-wise（块级）**：把序列分块，每步提交一个块。介于 uniform 与 confidence 之间，利于 [[KV cache]] 复用（块边界内确定）。
- **any-order（任意顺序）**：训练时让模型学所有顺序，推理随机或 learned 顺序。

### 3.3 与 AR decoding 对比

| 维度 | AR decoding | iterative decoding | parallel decoding |
|---|---|---|---|
| **步数** | $L$ 步 | $T$ 步（$T\ll L$） | $T$ 步，每步多 token |
| **每步产出** | 1 token | refine 全/部分序列 | 并行提交 $k$ token |
| **顺序** | 固定左到右 | coarse-to-fine 任意 | 任意（confidence 选） |
| **上下文** | 单向 causal | 双向 full attention | 双向 |
| **KV cache** | 有（[[KV cache]]） | 难直接复用 | 部分可（block 内） |
| **步数-质量** | 步多必快不了但质量稳 | 步多质量高慢、步少糙 | 同 |
| **factorization error** | 无 | 有（步少时） | 有 |
| **能 fill-in-middle** | 需重排 | 天然 | 天然 |

### 3.4 步数 trade-off

```
T (步数):
  大 (如 256): 每步提交少, 去噪充分, 质量接近 AR, 但慢
  小 (如 8~32): 每步提交多, 快, 但 factorization error 大, 质量降
  典型: LLaDA 256 步, Dream ~ 混合, 蒸馏后可降到 10~30 步
```

社区共识：**步数是延迟-质量主旋钮**。蒸馏（如 dParallel、CDCD 式）可把步数从数百降到 10~30 而不掉点。

### 3.5 与 speculative decoding 的区别

| 维度 | speculative decoding | Diffusion parallel decoding |
|---|---|---|
| **本质** | AR 加速（draft+verify） | 非并行 AR（真并行去噪） |
| **保 AR 顺序** | 保（左到右） | 不保（任意/coarse-to-fine） |
| **保 AR 分布** | 保（verify 拒绝保证分布不变） | 不保（不同分布） |
| **改训练** | 不改（纯推理） | 改（[[Diffusion LLM训练目标]]） |
| **加速来源** | draft 模型快+verify 多 token | 并行提交多 token |
| **能否 infill** | 不能（仍左到右） | 能 |

> [!note] 可混用
> AR 当 draft 模型出草稿 + Diffusion refine（SpecRef 等混合策略）。也可 spec 在 dLLM 内部（如 SimSD 给 dLLM 加 temporally valid context 做 token-level spec）。两条加速路线正交。

### 3.6 混合 AR + Diffusion

- **block-diffusion**：序列分块，块内 Diffusion 并行去噪，块间 AR 串行（块边界用 [[KV cache]]）。兼顾 AR 的 cache 友好与 Diffusion 的并行。代表 Block Diffusion。
- **AR draft + Diffusion refine**：AR 快出草稿，Diffusion 局部 refine 修正。SpecRef 等。
- **causal-mask diffusion**：给 dLLM 加 causal mask 兼容 [[KV cache]]（如 R2LM 用 reverse Mamba 侧供右上下文，保 cacheable）。见 [[Hybrid架构]] 思路迁移。


## 4. 数学原理 / 公式

### 4.1 AR 的步数下界

AR 生成 $L$ token 必须至少 $L$ 步（每步 1 token，因果依赖）：

$$T_{\text{AR}} \geq L$$

### 4.2 Diffusion 的步数与每步提交

Diffusion 每步可提交 $k_t$ 个 token，总 $\sum_{t=1}^T k_t \geq L$ 即可。最简均匀：

$$k_t = \lceil L/T \rceil,\quad T_{\text{Diff}} = T \ll L$$

延迟 $\propto T$（每步全序列 forward），$T\ll L$ 时快。

### 4.3 factorization error

并行提交 $k$ 个 token 时，模型用独立边缘近似联合：

$$p_\theta(x_{i_1},\dots,x_{i_k} \mid x_{\text{noise}}) \approx \prod_{j=1}^k p_\theta(x_{i_j} \mid x_{\text{noise}})$$

误差 $\propto$ 一次提交数 $k$ 与噪声比例。步数 $T\uparrow$ → 每步 $k\downarrow$ → 误差 $\downarrow$。这是步数-质量 trade-off 的根源。

### 4.4 算力模型对比

- **AR**：$L$ 步 × 每步单 token forward（decode，memory-bound，算力 $\propto d^2$ 每步）+ [[KV cache]]。
- **Diffusion**：$T$ 步 × 每步全序列 forward（prefill 式，compute-bound，$\propto Ld^2$ 每步）。总 $\approx T\cdot Ld^2$。

> [!tip] 何时 Diffusion 赢
> 当 $T\cdot (\text{单步 compute-bound}) < L\cdot (\text{单步 memory-bound 实际延迟})$。单步 Diffusion 算力重但**并行**（一个 forward 出多 token），AR 单步轻但**串行**。batch 大时 Diffusion 并行优势放大。详见 [[Diffusion LLM缓存与调度]] §4 算力模型。


## 5. 代码示例（可选）

### 5.1 iterative + confidence-based parallel 一步

```python
import torch

def diffusion_decode_step(model, seq, mask_id, n_commit, temperature=1.0):
    """
    一步去噪 + confidence 门控并行提交.
    seq: [L] 当前序列 (部分 mask, 部分 token)
    n_commit: 本步提交多少个 mask 位置
    返回更新后的 seq
    """
    with torch.no_grad():
        logits = model(seq.unsqueeze(0)).squeeze(0)   # [L, V]
        # 只看 mask 位置
        mask_pos = (seq == mask_id).nonzero().squeeze(-1)
        if len(mask_pos) == 0:
            return seq
        sub_logits = logits[mask_pos] / temperature
        probs = sub_logits.softmax(-1)               # [n_mask, V]
        conf = probs.max(-1).values                  # [n_mask] 每位置置信
        # 选最 confident 的 n_commit 个
        k = min(n_commit, len(mask_pos))
        top = conf.topk(k).indices
        commit_pos = mask_pos[top]
        sampled = probs[top].multinomial(1).squeeze(-1)  # 采样而非 argmax, 保多样性
        seq[commit_pos] = sampled
    return seq

def iterative_decode(model, length, mask_id, steps=64, temperature=1.0):
    seq = torch.full((length,), mask_id, dtype=torch.long)
    per_step = max(1, length // steps)
    for t in range(steps):
        # 步数越往后, 提交越少 (coarse-to-fine) 或固定
        n_commit = per_step
        seq = diffusion_decode_step(model, seq, mask_id, n_commit, temperature)
    return seq
```

### 5.2 block-wise 并行（利于 KV cache）

```python
def block_decode(model, length, mask_id, block=64, steps_per_block=4):
    seq = torch.full((length,), mask_id, dtype=torch.long)
    for blk_start in range(0, length, block):
        blk = slice(blk_start, blk_start + block)
        for _ in range(steps_per_block):
            logits = model(seq.unsqueeze(0)).squeeze(0)
            pos = (seq[blk] == mask_id).nonzero().squeeze(-1) + blk_start
            seq[pos] = logits[pos].argmax(-1)
    return seq
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Diffusion LLM训练目标]]（训练目标决定可并行去噪）、[[autoregressive decoding]]（对比基准）、[[采样策略]]（去噪采样）、[[自注意力]]（双向 attention）。
- **下游（应用）**: [[Diffusion LLM缓存与调度]]（并行解码使 KV cache 调度变复杂）、LLaDA/Dream 推理、fill-in-the-middle、可控生成。
- **对比 / 易混**:
  - **iterative vs parallel**：前者强调多步迭代 coarse-to-fine；后者强调每步并行提交。实际交织。
  - **parallel vs [[speculative decoding]]**：前者真并行非左到右、改训练；后者 AR 加速保顺序保分布、不改训练。见 3.5 表。
  - **Diffusion decode vs AR decode**：步数-质量 trade-off vs 固定 $L$ 步质量稳。见 3.3 表。


## 7. 常见误区与易错点

> [!warning] 误区 1：Diffusion 一定比 AR 快
> 不一定。单步 Diffusion 是全序列 forward（compute-bound，$O(Ld^2)$），AR 单步单 token（memory-bound）。步数 $T$ 须 $\ll L$ 且能摊销才快。小 batch AR 常更快。见 4.4。

> [!warning] 误区 2：步数越少越好
> 错。步少→每步提交多→factorization error 大→质量崩。步数是延迟-质量旋钮，需调。蒸馏可降步数不掉点。

> [!warning] 误区 3：parallel decoding 等于 speculative decoding
> 不同。spec 是 AR 加速（draft+verify，保左到右保分布）；parallel 是真并行去噪（非左到右、不保 AR 分布、改训练）。见 3.5 表。

> [!warning] 误区 4：iterative 和 parallel 是二选一
> 实际交织。iterative 过程每步并行提交即 parallel。社区常混称"iterative/parallel decoding"。

> [!warning] 误区 5：Diffusion 解码不需 KV cache
> 仍需且更复杂。双向 attention 使 KV 难直接复用（提交一个 token 会改变其他 token 的 KV）。需特殊缓存策略。详见 [[Diffusion LLM缓存与调度]]。

> [!tip] 实践：步数蒸馏
> 落地 dLLM 推理时，优先做步数蒸馏（如 dParallel：LLaDA-8B GSM8K 256→30 步，8.5× 加速不掉点）。原始数百步太慢，蒸馏到 10~30 步才实用。


## 8. 延伸细节

### 8.1 confidence 门控的本质

confidence-based 提交本质是**自适应步数分配**：模型有把握的 token 早提交、没把握的多 refine。这比 uniform 均匀提交质量好，因 token 难度不均（低频/精确词难、高频骨架词易）。

### 8.2 any-order 与 learned order

训练时让模型学所有 unmask 顺序（any-order），推理随机选或学一个 order policy（如 SAS 用 GRPO 学 order）。order 对质量影响大——structured 任务（Sudoku）顺序敏感，自由文本次之。

### 8.3 关键论文/工作（已核实）

- **SEDD** (arXiv:2310.16834)：提出 score entropy + 并行采样，可比 AR 少 32× 网络评估。
- **DiffuLLaMA** (arXiv:2410.17891)：AR→DLM 适配，含 parallel decoding 评估。
- **MDLM** (arXiv:2406.07524)：absorbing 扩散，加权 CE 训练，多步 unmask 解码。
- **LLaDA / Dream**：开源大规模 dLLM，迭代去噪 256 步基线 + 蒸馏加速。
- **dParallel**（arXiv:2509.26488）：步数蒸馏，256→30 步。
- **Block Diffusion**：块级并行 + 块间 AR，兼顾 cache。

### 8.4 内容来源

整理自 SEDD/MDLM/DiffuLLaMA 论文、LLaDA/Dream 开源实践、dParallel/SimSD 等加速工作。步数-质量 trade-off 与 factorization error 是社区共识。

---
相关: [[Diffusion LLM]] | [[Diffusion LLM训练目标]] | [[Diffusion LLM缓存与调度]] | [[autoregressive decoding]] | [[speculative decoding]] | [[采样策略]] | [[KV cache]] | [[自注意力]]
