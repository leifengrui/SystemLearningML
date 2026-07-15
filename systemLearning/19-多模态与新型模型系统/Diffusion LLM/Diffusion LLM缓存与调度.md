# Diffusion LLM 缓存与调度

> **所属章节**: [[Diffusion LLM]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: dLLM 推理系统 / Diffusion KV cache 适配 / iteration 调度 / dLLM continuous batching / Fast-dLLM / dKV-Cache
> **难度**: 高（需懂 [[KV cache]] + [[continuous batching的调度实现]] + [[iterative与parallel decoding]] + [[PagedAttention]] + prefill/decode 区分）


## 1. 一句话定义

**Diffusion LLM 缓存与调度** 是把为 [[autoregressive decoding]] 设计的推理系统（vLLM/SGLang 的 [[PagedAttention]]、[[KV cache]]、[[continuous batching的调度实现]]、[[chunked prefill]]）适配到 Diffusion LLM 去噪推理的工程命题——核心三挑战：① **KV cache 适配**：Diffusion 每个去噪步**全序列 forward** 且**每个步都可能改动任意位置的 token**（双向 attention，提交一个 token 会改变其他位置的 KV 激活），AR 的 KV cache（增量、只增不减、token 确定即永久缓存）**不能直接复用**，需"近似缓存"（Fast-dLLM/dKV-Cache：刷新部分 KV、跨步复用）或"块级缓存"（block 内确定即缓存）；② **调度**：每个请求的**迭代步数固定但可早停**（confidence 收敛即停），多请求的 iteration 调度是 [[continuous batching]] 在 Diffusion 下的变体——每个请求处在不同去噪步 $t$，batch 混合不同 $t$ 的请求（而非 AR 的不同 decode 长度），连续 batching 需按"iteration 维度"而非"token 维度"管理；③ **算子**：Diffusion 用**双向全序列 attention**（非 AR 的 causal），attention 算子是 full（无 causal mask），且每步全序列 forward 是 compute-bound（像 prefill 而非 decode）。**与 AR 系统的关系**：[[PagedAttention]] 的非 contiguous 变长支持可复用，但 KV 增量假设失效；[[continuous batching的调度实现]] 的动态进出可复用，但调度单位从"token/decode step"变成"diffusion iteration"；[[chunked prefill]] 的 prefill+decode 混合在 Diffusion 下变形为"反复 prefill"（每步都像 prefill，无传统 decode 的 1 token 模式）。算力模型：Diffusion 总延迟 $\approx T_{\text{步数}} \times \text{单步 prefill 式 FLOPs}$（vs AR 的 $L \times \text{decode FLOPs}$）。是 [[Diffusion LLM]] 落地 serving 的系统瓶颈。

> [!note] 三句话定位
> - **是什么**：把 AR serving 的 KV cache / continuous batching / chunked prefill 改适配 Diffusion 去噪——双向 attention 使 KV 难增量复用、调度单位变 iteration、算子是 full attention 每步 prefill 式。
> - **为什么**：vLLM/SGLang 全为 AR 设计（causal、1 token decode、增量 KV）；dLLM 双向、多 token 步、全序列 forward，假设全失效。
> - **与 [[continuous batching的调度实现]] 关系**：后者按 token/decode step 调度；Diffusion 按 iteration 调度（每请求不同 $t$），是同一思想的变体重构。


## 2. 为什么需要它（动机与背景）

### 2.1 AR serving 全为 AR 假设设计

vLLM/SGLang 的三大支柱都依赖 AR 假设：
- **[[KV cache]]**：token 一旦生成即确定，KV 永久缓存、只增不减、增量计算新 token 的 KV。
- **[[continuous batching的调度实现]]**：请求按 decode step 进出，每 step 各请求 1 token forward。
- **[[chunked prefill]]**：prefill（compute-bound）+ decode（memory-bound）混，decode 1 token + prefill chunk。

### 2.2 Diffusion 打破每个假设

- **双向 attention**：第 $i$ 个 token 的 KV 依赖全序列（含未来 token）。提交一个 token → 其他位置 attention 上下文变 → **其他位置的 KV 激活变**。AR 的"token 确定 KV 即定"失效。
- **多 token 步**：每步可能提交多个 token，且 refine 已有 token。不是 AR 的 1 token decode。
- **全序列 forward**：每步是 prefill 式 compute-bound（$O(Ld^2)$），不是 decode 的 memory-bound 1 token。
- **步数可变**：固定 $T$ 步但可早停，每请求步数不同。

### 2.3 调度维度重构

AR 调度按"decode step"（第几个 token）。Diffusion 按"diffusion iteration"（第几步去噪）。每个请求在 batch 中处在不同 $t$（有人第 5 步、有人第 50 步），连续 batching 需按 iteration 维度管理。这是 [[continuous batching]] 的变体重构。

### 2.4 算子层：causal → full

AR 用 causal mask（下三角），算子可优化为"只看前"。Diffusion 双向，attention 是 full（无 mask，全序列互看）。[[FlashAttention]] 的 causal 版不能用，需 full 版。且每步全序列 forward，算力是 prefill 级。


## 3. 核心概念详解

### 3.1 KV cache 在 Diffusion 下的失效与近似

```
AR KV cache:
  token i 生成 -> KV_i 计算 -> 永久缓存 -> 后续 step 复用 (只增不减)
  新 token j: 只算 KV_j 增量

Diffusion 双向 attention:
  提交 token i -> 全序列 attention 重算 -> 其他位置 KV 激活变
  -> 上一步 cache 的 KV_i 已失效 (上下文变了)
  -> 难直接增量复用
```

三类近似/特殊缓存策略：

- **Fast-dLLM / dKV-Cache（近似刷新）**：缓存上一步的 KV，本步只**部分刷新**（被改动位置重算、其他复用旧 KV），跨步复用旧 KV 引入近似误差。换延迟换质量。
- **block-wise cache（块级）**：序列分块，**块边界内 token 确定→该块 KV 永久缓存**（块间 AR 式串行，块内 Diffusion 并行）。把 AR 的增量思路移植到块级。详见 [[iterative与parallel decoding]] §3.6 block-diffusion。
- **no cache（naive）**：每步全序列重算，无 cache。质量最保真但最慢。基线。

### 3.2 iteration 维度的 continuous batching

```
AR continuous batching (按 decode step):
  step k: req_A (第 50 个 token), req_B (第 3 个), req_C (第 200 个)
  -> 各 1 token forward, 动态进出

Diffusion continuous batching (按 iteration t):
  iter k: req_A (去噪 t=128/256), req_B (t=5/256), req_C (t=255/256 早停将出)
  -> 各全序列 forward (变长: 不同生成长度), 动态进出 (早停即出)
```

调度单位从"decode step"变"diffusion iteration"。每个请求有自己的步数预算 $T_r$ 和当前步 $t_r$。请求 $t_r$ 达 $T_r$ 或 confidence 收敛即出，新请求进。是 [[continuous batching的调度实现]] 的变体重构。

### 3.3 步数预算与早停

每请求给固定步数 $T$（如 256），但可**早停**：每步算全序列 confidence，若剩余 mask 数 < 阈值或平均 confidence > 阈值即停。早停让短/易请求少花步数，长/难请求多花。调度器按"剩余步数预算"排序，类似 AR 按"剩余生成长度"。

### 3.4 算子：full attention + 全序列 forward

```
AR forward (decode step):
  1 token in, causal attention (下三角), KV cache 增量
  memory-bound, O(d^2) 算力

Diffusion forward (去噪 step):
  全序列 L token in, full attention (无 causal mask), 
  compute-bound, O(L d^2) 算力 (像 prefill)
```

- [[FlashAttention]]：causal 版不能直接用，需 full attention 版（FlashAttention 本身支持 full，但少了 causal 的下三角剪枝，FLOPs 更高）。
- [[PagedAttention]]：非 contiguous 变长 KV 存储可复用，但增量假设失效（KV 会失效需刷新）。
- 算力是 prefill 级（compute-bound），batch 越大越摊销。

### 3.5 与 AR 系统组件对照

| AR 组件 | Diffusion 下状态 | 处理 |
|---|---|---|
| [[KV cache]] | 失效（双向，提交改其他 KV） | 近似刷新 / 块级缓存 |
| [[PagedAttention]] | 非 contiguous 存储可复用 | 增量假设失效，需刷新 |
| [[continuous batching的调度实现]] | 可复用思想 | 单位从 token→iteration |
| [[chunked prefill]] | prefill+decode 混 | 变"反复 prefill"，无 1 token decode |
| causal attention | 不能用 | full attention |
| [[speculative decoding]] | token-level 难（双向） | 需 temporally valid context（SimSD） |


## 4. 数学原理 / 公式

### 4.1 AR 单步 vs Diffusion 单步算力

- **AR decode step**：1 token forward，算力 $\propto d^2$（memory-bound，访 KV cache）。总 $L$ 步。
- **Diffusion denoise step**：全序列 $L$ token forward，full attention，算力 $\propto L d^2 + L^2 d$（compute-bound，像 prefill）。总 $T$ 步。

### 4.2 总算力对比

$$\text{FLOPs}_{\text{AR}} \approx L \cdot d^2 \quad (\text{串行 } L \text{ 步})$$

$$\text{FLOPs}_{\text{Diff}} \approx T \cdot (L d^2 + L^2 d) \quad (\text{并行 } T \text{ 步, 每步全序列})$$

> [!note] 注意 $L^2 d$
> Diffusion 每步 full attention 有 $O(L^2 d)$ 项（序列长度的平方）。长序列 + 多步，算力远超 AR 的 $O(Ld^2)$。但 **并行性**抵消：$T\ll L$ 且单步 GPU 满载，wall-clock 可更低。$L^2$ 项可被 [[FlashAttention]] 降到 $O(L^2)$ IO 而非 FLOPs。

### 4.3 KV cache 失效的根因

AR causal attention，token $i$ 的 attention 输出 $a_i = \sum_{j\leq i} \text{softmax}(q_i k_j) v_j$，只依赖 $j\leq i$，token $i$ 确定后 $a_i$ 定（$j>i$ 不影响）。故 KV 可增量缓存。

Diffusion full attention，$a_i = \sum_{j} \text{softmax}(q_i k_j) v_j$（含 $j>i$）。**提交一个新 token 改变所有 $a_j$**（因 softmax 分母变）。故旧 KV 失效：

$$\Delta a_j \neq 0 \quad \text{当任意 token } i \text{ 被提交/改动}$$

这是 KV 难直接复用的数学根因。

### 4.4 块级缓存的有效条件

把序列分块 $b_1,\dots,b_n$，块间 AR 串行（块 $b_k$ 依赖 $b_{<k}$），块内 Diffusion 并行。则 $b_k$ 内 token 的 attention 只依赖 $b_{\leq k}$，**$b_k$ 一旦全确定，其对后续块的贡献 KV 可永久缓存**（类似 AR）。这是 block-diffusion / R2LM 类方法保 [[KV cache]] 兼容的条件。

### 4.5 Sangam 式 token-budget 调度

每个请求去噪步是"prefill 式全序列 + 块级 decode 混"。Sangam（arXiv:2607.04206，待核实编号）提出 **deficit token-budget 调度**：每 iteration 有 token 预算，优先 admit in-flight decode（小步）、大 prefill 仅当预算够才 admit、预算可结转。实现 amortized stall-free（无饿死）。是 [[chunked prefill]] 思想在 dLLM 的迁移。


## 5. 代码示例（可选）

### 5.1 iteration 维度 continuous batching（概念）

```python
class DiffusionBatchScheduler:
    def __init__(self, max_steps=256, early_stop_conf=0.95):
        self.max_steps = max_steps
        self.early_stop_conf = early_stop_conf
        self.running = []   # 各请求: (seq, t, step_budget)

    def schedule(self):
        batch = []
        for r in list(self.running):
            if r['t'] >= r['step_budget'] or r['conf'] > self.early_stop_conf:
                self.running.remove(r); r['done_cb'](r['seq']); continue
            batch.append(r)
            r['t'] += 1
        # admit waiting (按剩余预算排序)
        for r in self.waiting:
            if len(batch) < self.max_batch: batch.append(r); self.running.append(r)
        return batch   # 各做一步全序列 forward (不同 t, 变长)

    def step(self, batch, kv_cache=None):
        # 全序列 full attention forward (causal=False)
        out = self.model([r['seq'] for r in batch], causal=False, kv_cache=kv_cache)
        for r, o in zip(batch, out):
            r['seq'] = commit_tokens(r['seq'], o, n=r['per_step'])
            r['conf'] = o['conf'].mean()
```

### 5.2 dKV-Cache 近似刷新（概念）

```python
def diffusion_forward_with_dkv_cache(model, seq, prev_kv, refresh_mask, step_t):
    """
    prev_kv: 上一步的 KV cache (近似)
    refresh_mask: 哪些位置 KV 本步要重算 (被改动位置)
    返回本步输出 + 更新后的 KV
    """
    # 只对 refresh 位置重算 KV, 其他复用 prev_kv (近似)
    new_kv = prev_kv.clone()
    new_kv[refresh_mask] = compute_kv(model, seq[refresh_mask])
    out = full_attention(seq, new_kv, causal=False)   # full, 非 causal
    return out, new_kv
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[KV cache]]（要改适配）、[[continuous batching的调度实现]]（思想复用、单位重构）、[[PagedAttention]]（非 contiguous 可复用、增量假设失效）、[[chunked prefill]]（prefill+decode 混→反复 prefill）、[[iterative与parallel decoding]]（去噪方式决定调度）、[[Diffusion LLM训练目标]]（双向 attention 来源）。
- **下游（应用）**: dLLM serving（LLaDA/Dream 部署）、Fast-dLLM/dKV-Cache 加速、Sangam 等系统、AR 栈复用（dLLM 用 AR serving 机制）。
- **对比 / 易混**:
  - **dLLM KV cache vs [[KV cache]]**：前者双向失效需近似刷新/块级；后者 causal 增量只增不减。
  - **dLLM continuous batching vs [[continuous batching的调度实现]]**：前者按 iteration 调度、每步全序列；后者按 decode step、每步 1 token。
  - **dLLM vs [[chunked prefill]]**：前者每步都 prefill 式（反复 prefill）、无 1 token decode；后者 prefill chunk+decode 1 token 混。
  - **dLLM 算子 vs AR causal**：full attention（无 causal mask） vs causal attention（下三角）。


## 7. 常见误区与易错点

> [!warning] 误区 1：Diffusion 能直接用 AR 的 KV cache
> 不能。双向 attention，提交一个 token 改变其他位置 KV 激活，旧 KV 失效。需近似刷新（Fast-dLLM/dKV-Cache）或块级缓存。根因见 4.3。

> [!warning] 误区 2：Diffusion 每步是 decode（memory-bound）
> 错。每步全序列 forward 是 **prefill 式 compute-bound**（$O(Ld^2+L^2d)$），不是 AR decode 的 memory-bound 1 token。算力模型完全不同，见 4.2。

> [!warning] 误区 3：Diffusion 调度按 token step
> 错。按 **diffusion iteration**（第几步去噪）。每请求不同 $t$，batch 混合不同 $t$ 请求。是 [[continuous batching]] 的变体重构，不是 token 维度。

> [!warning] 误区 4：Diffusion 步数固定不能优化
> 可早停（confidence 收敛即停）+ 蒸馏降步数。步数是调度与算力的主旋钮。见 [[iterative与parallel decoding]] §3.4。

> [!warning] 误区 5：full attention 等于 causal 去掉 mask
> 算子上对（full=无 causal mask），但 FLOPs 与 IO 更高（无下三角剪枝）。[[FlashAttention]] full 版仍可用但更贵。且 KV 失效问题来自 full。

> [!tip] 实践：块级缓存复用 AR 栈
> 落地 dLLM serving 时优先 block-diffusion（块内并行+块间 AR），可复用 [[KV cache]] 与 AR serving 栈（vLLM），改造最小。纯全序列去噪 cache 改造大。Sangam 等工作证明 dLLM serving 设计空间仍由 prefill-decode 干涉主导，AR 机制可迁移。


## 8. 延伸细节

### 8.1 Fast-dLLM / dKV-Cache

近似刷新类方法：缓存上一步 KV，本步只重算被改动位置 KV，跨步复用旧 KV（近似）。在质量-延迟间权衡。是当前 dLLM KV cache 加速主流。

### 8.2 Sangam：dLLM 用 AR 栈

Sangam（arXiv:2607.04206，编号待核实）核心洞察：dLLM 的近似缓存使推理呈"反复 prefill + 块 decode"结构，与 AR serving 机制同构。提出 deficit token-budget 调度：优先 admit in-flight decode、prefill 仅预算够才 admit、预算结转。colocated（同机混）适合 decode-heavy，hybrid（prefill/decode 分机）适合 prefill-heavy。证明 dLLM serving 设计空间仍由 prefill-decode 干涉与 partition 主导。

### 8.3 constrained decoding for dLLM

dLLM 并行提交多 token（mean-field 分布），AR 的 constrained decoding（按 token 顺序 mask 非法）失效。需在 mean-field 后验上做约束采样（有限自动机），如 arXiv:2607.07026 的方法，保 JSON schema 等约束。是 dLLM serving 的特有问题。

### 8.4 内容来源

整理自 Fast-dLLM、dKV-Cache、Sangam（arXiv:2607.04206，编号待核实）、SimSD、TACG 等 2025-2026 dLLM serving 工作及 LLaDA/Dream 部署实践。与 AR serving 组件对照据 [[continuous batching的调度实现]]、[[chunked prefill]]、[[KV cache]]、[[PagedAttention]]。

---
相关: [[Diffusion LLM]] | [[Diffusion LLM训练目标]] | [[iterative与parallel decoding]] | [[KV cache]] | [[continuous batching的调度实现]] | [[chunked prefill]] | [[PagedAttention]] | [[FlashAttention]] | [[speculative decoding]] | [[autoregressive decoding]]
