# continuous batching

> **所属章节**: [[推理优化]]
> **所属模块**: [[08-推理系统]]
> **别名**: iteration-level batching / continuous batching / 连续批处理 / 迭代级调度
> **难度**: 中（需懂 [[KV cache management]] + 调度）


## 1. 一句话定义

**continuous batching（连续批处理 / 迭代级调度）** 是 LLM 推理引擎的一种调度策略：在**每个 decode iteration（每生成一个 token）**的边界上动态地**加入新请求、驱逐已完成请求**，让执行槽（slot）始终被填满，而**不必等当前 batch 里所有请求都生成完才换批**。它是 vLLM/ORCA 的核心机制，相对 static batching（请求级、整批换）能显著提升 GPU 利用率与吞吐。

> [!note] 三句话定位
> - **是什么**：batch 的"成员"在 token 边界动态进出，slot 数固定但内容流动。
> - **为什么**：static batching 让短请求空等长请求，GPU 大段时间闲置。
> - **怎么做**：每步 iteration 检查哪些 slot 空了，立即塞入等待中的新请求（先 prefill 再 decode）。


## 2. 为什么需要它（动机与背景）

### 2.1 static batching 的痛点

传统推理服务（如早期 TGI / TRT-LLM static）用 **request-level batching**：攒一批请求 → 一起 prefill → 一起 decode → **等 batch 里所有请求都生成到 EOS 或 max_tokens** → 整批结束 → 才换下一批。

问题在"等最慢的那个"：

| 场景 | 请求 A | 请求 B | 请求 C | 请求 D |
|---|---|---|---|---|
| 生成长度 | 20 | 800 | 30 | 20 |

batch 周期 = max = 800。A/C/D 早早就生成完了，但它们的 slot 仍被占着、GPU 在为 B 一个请求空转，直到 B 结束才能换批。

- **slot 浪费**：短请求结束后 slot 闲置，不能接新请求。
- **吞吐塌陷**： batch 退化为单请求（只剩 B），GPU 利用率暴跌。
- **延迟劣化**：新到的请求必须等整批结束才能进，队尾延迟高。

### 2.2 continuous batching 的解法

把调度的**粒度从"请求级"降到"token 级"**：每生成一个 token（一个 iteration）就检查——

- 有没有请求刚生成完（hit EOS 或达到 max_tokens）？→ **立即释放它的 slot**。
- 等待队列里有没有新请求？→ **立即填进空 slot**（先做它的 prefill，再开始 decode）。
- 显存够吗？→ 不够就 **preempt**（驱逐）某些请求（见 §3.4）。

这样 slot 永远满，GPU 不为"等最慢请求"空转。

> [!tip] 直觉
> static batching 像公交车"满员发车，到终点才下客再发下一班"；continuous batching 像地铁"每站都能上下客，车厢始终保持满载"。地铁的吞吐远高于"到终点才换"的公交。


## 3. 核心概念详解

### 3.1 iteration-level vs request-level

| 维度 | static / request-level | continuous / iteration-level |
|---|---|---|
| 调度粒度 | 一次 batch（请求级） | 一个 token（迭代级） |
| 换批时机 | 整批结束 | 每个 iteration 边界 |
| slot 占用 | 整批周期内固定 | 请求结束即释放 |
| 短请求 | 空等长请求 | 立即让位给新请求 |
| GPU 利用率 | 随 batch 退化而跌 | 始终接近满槽 |
| 实现复杂度 | 简单 | 高（需 per-slot KV 管理 + 调度器） |

### 3.2 slot（执行槽）与固定 batch 上限

引擎维护 $N$ 个 **slot**（如 $N=64$），每个 slot 容纳一个活跃请求的解码状态（含它的 [[KV cache management]]）。slot 数上限由显存决定（KV cache 显存预算 / 单请求平均 cache 大小）。continuous batching 保证：

- 任意时刻活跃请求数 $\le N$；
- 某请求结束 → slot 立刻可被等待队列首个请求占用；
- 若 prefill 与 decode 用同一批 slot，需在 prefill（compute-heavy）与 decode（memory-heavy）间权衡（见 §3.5 chunked prefill）。

### 3.3 与 PagedAttention 的配合

continuous batching 的"动态进出"要求 KV cache 能**按请求独立分配/释放**且**不连续**。这正是 [[KV cache management]] 的 PagedAttention 擅长的：

- 请求加入 → 申请若干物理 block 建其 KV；
- 请求结束 → 归还 block 到池；
- 请求被 preempt → block 换出到 CPU 或丢弃（recompute）。

没有 PagedAttention 的连续显存模型下，动态进出会产生大量碎片，continuous batching 难以高效实现。所以 vLLM 把两者**捆绑**实现。

### 3.4 preemption（驱逐）

当新请求要 prefill 但显存不够时，continuous batching 引擎会 **preempt** 某些在跑请求——把它们的 KV cache **换出到 CPU 内存**（可恢复）或**直接丢弃**（之后用 prompt 重算 prefill，因 recompute 比 PCIe 换入更便宜）。被 preempt 的请求回到等待队列，稍后显存空出再 recompute + 续 decode。

vLLM 默认策略：按 FIFO（最早进来的先被驱逐）或 LRU。

### 3.5 chunked prefill（分块预填）

prefill 阶段算整个 prompt 的 $K/V$，对长 prompt（如 4K–32K token）是 **compute-bound** 的大块计算，会长时间独占 GPU、阻塞 decode。**chunked prefill** 把长 prompt 切成 chunk（如 512 token/chunk），与 decode iteration **交错**执行：

```
iter:  [prefill_chunk(A,512)] [decode(batch)] [prefill_chunk(A,512)] [decode(batch)] ...
```

好处：decode 不被长 prefill 饿死，首 token 延迟（TTFT）与每 token 延迟（TPOT）都更平滑。vLLM、SGLang、TGI 都支持。

> [!note] prefill vs decode 的混合调度
> - prefill：算力密集，吞吐高时拼 [[GPU utilization]]（compute bound）。
> - decode：访存密集，拼 KV cache 带宽（memory bound）。
> 混合调度要在两者间分配 iteration，避免单一阶段饿死另一阶段。这是 [[batching tradeoff]] 的核心议题。


## 4. 数学原理 / 公式

### 4.1 吞吐模型：static vs continuous

设 $N$ 个 slot，请求生成长度 $L_i$（随机变量，均值 $\bar L$，最大 $L_{\max}$），每步 iteration 生成 1 token 耗时 $T_{\text{iter}}$（与 batch 大小弱相关，因 decode 是 memory-bound，batch 越大单步略慢但吞吐高）。

**static batching**：一个周期 = 一个 batch 跑完 = $L_{\max} \cdot T_{\text{iter}}$，产出 $N$ 个请求。吞吐：

$$
\text{TPS}_{\text{static}} = \frac{N}{L_{\max} \cdot T_{\text{iter}}}
$$

**continuous batching**：slot 随到随走，稳态下每个 slot 的周转 = $\bar L \cdot T_{\text{iter}}$，吞吐：

$$
\text{TPS}_{\text{cont}} = \frac{N}{\bar L \cdot T_{\text{iter}}}
$$

**提升比**：

$$
\frac{\text{TPS}_{\text{cont}}}{\text{TPS}_{\text{static}}} = \frac{L_{\max}}{\bar L}
$$

> [!note] 推导要点
> 提升倍数 = **最大长度 / 平均长度**。长度方差越大（长短请求混跑），continuous 相对 static 优势越大。若所有请求等长（$L_{\max}=\bar L$），二者持平——continuous 的收益来自**长度不均**。

### 4.2 数值实例

服务 4 个请求，长度 $[20, 800, 30, 20]$，$\bar L = 217.5$, $L_{\max}=800$。$N=4$, $T_{\text{iter}}=20$ ms。

- static 周期 = $800 \times 20$ ms = 16 s，产出 4 → TPS = $4/16 = 0.25$ req/s。
- continuous 周转 = $\bar L \times 20$ ms = $217.5 \times 0.02 = 4.35$ s → TPS = $4/4.35 \approx 0.92$ req/s。

提升 ≈ $800/217.5 \approx 3.7\times$。

> [!warning] 这是简化模型
> 实际 $T_{\text{iter}}$ 随 batch 大小变化（大 batch 单步略慢因 KV cache 读量大），且有 prefill 中断、preemption 等开销。但量级关系成立：continuous 在长度不均时显著优于 static。


## 5. 代码示例（可选

### 5.1 简化 continuous batching 调度器（slot 视角）

```python
import random

class Request:
    def __init__(self, rid, gen_len):
        self.id = rid
        self.gen_len = gen_len        # 总要生成的 token 数
        self.generated = 0            # 已生成数
        self.kv_blocks = 0            # 占用的 KV cache block 数（简化）
    def step(self):
        """生成一个 token, 返回是否完成."""
        self.generated += 1
        self.kv_blocks += 1           # cache 增长 1
        return self.generated >= self.gen_len
    def __repr__(self):
        return f"R{self.id}({self.generated}/{self.gen_len})"

class ContinuousBatcher:
    def __init__(self, num_slots, queue):
        self.slots = [None] * num_slots
        self.queue = queue            # 等待中的请求
        self.iter = 0
    def _fill_slots(self):
        """把空 slot 填上等待队列里的请求 (即 prefill 后开始 decode)."""
        for i, s in enumerate(self.slots):
            if s is None and self.queue:
                self.slots[i] = self.queue.pop(0)
                print(f"  [iter {self.iter}] slot {i} <= {self.slots[i]} (prefill+start)")
    def run(self, max_iters=50):
        while max_iters > 0 and (any(s is not None for s in self.slots) or self.queue):
            self._fill_slots()
            self.iter += 1; max_iters -= 1
            # 每个 iteration: 所有 slot 各生成 1 token
            for i, req in enumerate(self.slots):
                if req is None: continue
                done = req.step()
                if done:
                    print(f"  [iter {self.iter}] slot {i}: {req} DONE, 释放")
                    self.slots[i] = None   # 释放 slot, 下个 iteration 立即被填
        print(f"结束, 共 {self.iter} iterations")

# —— 演示: 长短请求混跑, 短的结束立即让位 ——
random.seed(0)
queue = [Request(1, 3), Request(2, 10), Request(3, 2), Request(4, 5), Request(5, 4)]
ContinuousBatcher(num_slots=3, queue=queue).run()
```

输出（示意）：slot 1 的短请求先结束，立即被等待中的请求 4 填上，全程 3 个 slot 都在干活——这就是 continuous 的本质。

### 5.2 对比 static 的伪代码

```python
def static_batch(requests, batch_size):
    """请求级: 攒一批, 全跑完才换下一批."""
    out = []
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        Lmax = max(r.gen_len for r in batch)
        for _ in range(Lmax):                 # 必须等最长的
            for r in batch:
                if r.generated < r.gen_len:
                    r.step()
        out += batch
    return out
# 短请求会被 max(L) 拖累, slot 大量时间空转
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[KV cache management]]（按请求独立分配/释放 KV，PagedAttention 是 continuous 的前提）、[[attention]]（decode 每步的算子）。
- **下游（应用）**: [[batching tradeoff]]（batch 大小、prefill/decode 混合的权衡）、[[GPU utilization]]（continuous 提升利用率的目标）、[[sampling throughput]]（RLHF 中 [[rollout worker]] 的采样吞吐直接受 continuous batching 影响）、[[policy deployment (PD)]]（PD 侧服务依赖 continuous batching 提并发）。
- **对比 / 易混**:
  - **continuous batching vs [[dynamic batching]]**：粒度不同！continuous 是 **token/iteration 级**（每步换人），dynamic 是 **request 级**（攒请求凑批，但仍整批一起跑完）。两者常被混淆，详见 [[dynamic batching]]。
  - **continuous batching vs static batching**：前者 token 级动态进出，后者请求级整批换。
  - **continuous vs in-flight batching**：同义（in-flight 是 TGI/NVIDIA 用词，continuous 是 vLLM 用词）。


## 7. 常见误区与易错点

> [!warning] 误区 1：把 continuous batching 等同于 dynamic batching
> **不是一回事**。dynamic batching 在**请求到达时**凑批（不等满批就发，timeout 触发），但 batch 一旦发出仍是"整批跑完才换"——粒度仍是请求级。continuous batching 的粒度是 **token/iteration 级**，每步都能换人。两者可叠加（dynamic 决定何时攒批，continuous 决定批内如何调度）。

> [!warning] 误区 2：以为 continuous 是"无 batch"
> 不是。continuous 仍有固定 slot 上限 $N$（受显存约束），只是 slot 的**成员动态流动**。它仍是 batched 执行（一次 forward 多个请求），不是单请求串行。

> [!warning] 误区 3：以为 continuous 只优化 decode
> prefill 也受益（chunked prefill 让长 prompt 不阻塞 decode）。且新请求的 prefill 可与现有 decode 在同 iteration 内交错，提升整体吞吐。

> [!warning] 误区 4：忽略 preemption 的代价
> 显存不够时 preempt 会换出/重算被驱逐请求的 KV，造成**重复计算开销**与该请求的延迟尖峰。设计 batch 上限时要留余量，不能压满显存。

> [!warning] 误区 5：以为提升比总是 $L_{\max}/\bar L$
> 那是简化模型。实际 $T_{\text{iter}}$ 随 batch 大小增长（decode 是 memory-bound，cache 读量随 batch 涨），且 prefill 中断、preemption、调度开销都会拉低。但量级方向正确。


## 8. 延伸细节

### 8.1 vLLM 的实现要点

- **Scheduler**：每个 iteration 跑一次调度，决定哪些请求进/出、是否 preempt。
- **block manager**：配合 PagedAttention，按 block 分配/释放 KV。
- ** RunningList / WaitingList**：在跑的请求 vs 等 prefill 的请求，scheduler 在两者间分配显存预算。
- **preemption**：当 waiting 里有请求要 prefill 但显存不足，从 running 里按 FIFO 选若干换出（先到 CPU，实在不行 recompute）。

### 8.2 ORCA：continuous batching 的学术起源

ORCA（Yu et al., 2022, "Orca: A Distributed Serving System for Transformer-Based Generative Models"）首次提出 **iteration-level scheduling** 与 **选批（selective batch）**，是 vLLM/TGI 的理论前身。vLLM 在其上加了 PagedAttention 与 prefix caching。

### 8.3 与 RLHF 的关系

在 [[policy training (PT)]]-[[policy deployment (PD)]] 架构（[[训推分离]]）中，[[rollout worker]] 采样大量轨迹，其吞吐直接依赖 PD 侧的 continuous batching——RLHF 训练的瓶颈常在采样（生成）而非梯度更新，所以 continuous batching 是提升 RLHF 整体吞吐的关键。[[sampling throughput]] 与本篇强耦合。

### 8.4 首 token 延迟 vs 每 token 延迟

- **TTFT（Time To First Token）**：prefill 延迟，受 chunked prefill 调度影响。
- **TPOT（Time Per Output Token）**：decode 延迟，受 batch 大小与 KV 带宽影响。
continuous batching + chunked prefill 的目标是同时优化两者，详见 [[batching tradeoff]]。

---
相关: [[推理优化]] | [[KV cache management]] | [[dynamic batching]] | [[batching tradeoff]] | [[GPU utilization]]
