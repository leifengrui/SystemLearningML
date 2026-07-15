# dynamic batching

> **所属章节**: [[推理优化]]
> **所属模块**: [[08-推理系统]]
> **别名**: dynamic batching / 动态批处理 / 请求级动态凑批
> **难度**: 中（需懂调度 + 与 [[continuous batching]] 区分）


## 1. 一句话定义

**dynamic batching（动态批处理）** 是推理服务在**请求级**动态凑批的策略：请求随时间到达服务端，batcher 在"**凑够 max_queue_size 个**"或"**等够 max_queue_delay 时间**"任一条件满足时，把当前积压的请求聚成一个 batch 一起前向，而非逐个串行处理或用固定大小批。它是 NVIDIA Triton Inference Server 的招牌特性，**粒度仍是请求级**（一批跑完才发下一批），区别于 [[continuous batching]] 的 token 级。
> [!note] 解答：为什么攒成一个 batch 一起前向？一个请求不能前向吗？
> **能，而且完全可以。** 单请求前向就是 `batch=1`，把一条样本送进模型跑一次 forward，输出一条结果。这在代码上没有任何障碍，`model(x)` 里 `x.shape=(1, seq_len)` 就行。**问题不在"能不能"，而在"亏不亏"——单请求前向把 GPU 当单线程 CPU 用，绝大多数算力在空转。**
>
> ### 一、根本原因：GPU 是吞吐器件，不是延迟器件
>
> GPU 的设计哲学是"**成千上万个 CUDA 核心并行 + 高带宽显存喂饱它们**"。A100 有 108 个 SM、6912 个 CUDA core、312 TFLOPS 峰值算力、1.5 TB/s HBM 带宽。这些资源是**为"大批量并行"准备的**——只有当成百上千个独立的计算任务同时喂进来，GPU 才能把这些核心都占满。
>
> 一条请求（batch=1）送进 LLM 做 decode：每步只算 1 个 token 的 logits，计算量 $\approx 2N$（7B 模型 $1.4\times 10^{10}$ FLOPs），但要读完整权重 + 整个 [[KV cache management]]（GB 级）。算少读多，算术强度 $I \approx 14$ FLOP/Byte $\ll$ A100 的 ridge point 208 → **强 memory-bound，算力利用率 $<10\%$**（详见 [[GPU utilization]] §3.3、§4.2）。也就是说 90% 以上的 CUDA core 在等数据，GPU 在"空转烧钱"。
>
> 攒成 batch $B$ 一起前向时：**权重只读一次**（所有请求共享），$B$ 条请求的计算量 $\times B$ 但访存几乎不变 → 算术强度 $\uparrow$ → 利用率 $\uparrow$ → 吞吐 $\uparrow$。这才是 batch 的本质——**把一次访存摊到多个请求的计算上，把闲置的算力填满**。
>
> ### 二、数学：吞吐随 batch 几乎线性涨，单次耗时几乎不涨
>
> 单次前向耗时 $T_{\text{fwd}}(B)$ 随 $B$ **只缓慢增长**（因为 GPU 并行核多，小 batch 时根本没占满，加请求只是填空，几乎不增耗时）：
>
> $$
> T_{\text{fwd}}(B) \approx T_{\text{launch}} + \frac{2N \cdot B}{P_{\text{compute}}} \quad(\text{compute-bound 段})
> $$
>
> 吞吐 $\text{throughput}(B) = B / T_{\text{fwd}}(B)$ 在 $B$ 小时几乎随 $B$ **线性增长**（$T_{\text{fwd}}$ 被常数项 $T_{\text{launch}}$ + 访存项主导，$B$ 项很小）。直到 $B$ 大到占满算力或显存，吞吐才见顶。
>
> | batch | 单次耗时（相对） | 吞吐（相对） | 单请求延迟 |
> |---|---|---|---|
> | 1（串行） | 1× | 1× | 低但贵 |
> | 8 | ~1.1× | ~7× | 略增 |
> | 64 | ~1.5× | ~40× | 增（等凑批） |
>
> 串行跑 64 条请求耗时 $\approx 64 \times T_{\text{fwd}}(1)$；攒成 batch=64 一次跑 $\approx 1.5 \times T_{\text{fwd}}(1)$。**同样 64 条请求，batch 比串行快 ~40 倍**。这就是为什么要攒批——不是为了好玩，是 GPU 物理特性逼的。
>
> ### 三、那为什么不一直 batch=1？要回答两个反问
>
> 1. **单请求前向不是延迟最低吗？** 对，但**单位成本最高**。在线服务如果 QPS 高，串行根本跑不过来，请求堆积超时；离线批量则浪费 90% 算力。两种场景都被"单请求"的浪费拖垮。
> 2. **攒批不是要等凑齐、增加延迟吗？** 对，这正是 [[batching tradeoff]] 的核心矛盾：攒批提吞吐但增凑批等待。dynamic batching 用"**阈值或超时双触发**"在这两者间权衡——高峰按 size 触发（凑得快，延迟可控），低谷按 delay 触发（不干等，延迟有上界）。见本节 §2.1 的三档对比表。
>
> ### 四、一个请求前向的合法场景
>
> 不是说单请求永远不行。以下场景 batch=1 反而合理：
>
> - **调试 / 单样本推理**：本地跑一条 prompt 看输出，算力浪费无所谓。
> - **延迟极敏感的在线交互**：用户输入立即响应，不允许凑批等待（但即便如此，业界仍用 [[continuous batching]] 把多个单请求在引擎内 slot 级并行，而非真串行）。
> - **显存预算极小**：单卡只能装 batch=1 的大模型。
> - **流式生成**：单请求 decode 每步本质是 batch=1 的前向（但 vLLM 等会把多个请求的 decode 步合并成大 batch，见 [[continuous batching]]）。
>
> ### 五、LLM 场景的特殊性：batch=1 的 decode 是利用率洼地
>
> LLM 的 decode 阶段，每步生成 1 个 token，本质就是"batch=1 的小前向"——这是自回归机制固有的低利用率点。所以推理系统的几乎所有优化（[[continuous batching]] 把多请求 decode 合批、[[speculative decoding]] 一次 forward 多 token、[[KV cache management]] 量化减访存、kernel 融合减访存次数）**本质都是在对抗 batch=1 的浪费**，把利用率从 <10% 往上拉。见 [[GPU utilization]] §2.3 的"提利用率的全部努力"清单。
>
> ### 六、一句话总结
>
> **一个请求能前向，但那是把 312 TFLOPS 的 GPU 当 3 TFLOPS 用。攒批是为了把 GPU 的并行算力喂饱——一次访存服务多个请求的计算，吞吐近线性涨、单次耗时几乎不涨。这正是 batching（static/dynamic/continuous 三种）存在的物理理由，也是 [[batching tradeoff]] 权衡的出发点。**
>
> 相关：[[GPU utilization]]（batch=1 为何利用率低）、[[batching tradeoff]]（攒批的吞吐-延迟权衡）、[[continuous batching]]（LLM 里如何把多个"单请求 decode"合批）。

> [!note] 解答：你的矩阵乘法类比对吗？"m 越长一次算越多、显存固定"对吗？
>
> **方向抓得很准，但维度表述和"显存固定"这句要修正。** 你想用"一个固定矩阵 × 一个变长矩阵"来类比 batching，这个直觉是对的——batching 的数学本质就是 GEMM（矩阵乘）。但更准确的形式不是 $[1..n]\times[1..m]^T$，而是：
>
> $$
> \underbrace{X}_{B\times d}\ \times\ \underbrace{W}_{d\times d}\ =\ \underbrace{Y}_{B\times d}
> $$
>
> 其中 $B$ = **batch size（行数，就是你说的"m 越长"那个维度）**，$d$ = 隐藏维度，$W$ = **权重矩阵（固定不变的那个）**。每多一条请求，就是给 $X$ 多加一行，计算量随 $B$ 线性涨，但 $W$ 只读一次。
>
> ### 一、你抓对的部分：权重是"固定矩阵"，计算量随 batch 线性涨
>
> 一次前向的主要工作是若干层 GEMM：$X(B\times d) \times W(d\times d)$。
>
> | 量 | 公式 | 随 $B$ 变化 |
> |---|---|---|
> | FLOPs | $2 B d^2$（每层，所有层加和 $2BN$） | **线性 $\uparrow$** |
> | 权重访存 Bytes | $2d^2$（每层，$W$ 只读一次） | **不变（固定 ✓）** |
> | 算术强度 $I$ | $\frac{2Bd^2}{2d^2} = B$ | **随 $B$ 线性 $\uparrow$** |
>
> 所以你说"m 越长一次算的越多，因为[权重那部分]显存占用固定"——**这个直觉完全正确**，$B$ 越大算术强度 $I$ 越高，越靠近 ridge point，GPU 算力利用率越高（详见 [[GPU utilization]] §4.2 大 batch 摊薄权重读的推导）。**这就是 batching 提吞吐的数学根：固定访存（权重）被摊到更多计算（多行 $X$）上。**
>
> ### 二、必须修正的部分："显存占用固定"只对权重成立，KV cache 和激活随 batch 涨
>
> 你说"显存占用是固定的"——**这只对权重 $W$ 成立，对 KV cache 和中间激活不成立**。LLM 推理显存有三块：
>
> | 显存组成 | 是否随 batch $B$ 变 | 量级（7B, seq 2048） |
> |---|---|---|
> | 权重 $W$ | **固定** ✓（就是你说的那个） | ~14 GB（bf16） |
> | **KV cache** | **随 $B$ 线性涨** ✗ | batch 1 ≈ 0.5 GB，batch 64 ≈ 32 GB |
> | 中间激活 | 随 $B$ 线性涨 ✗ | 数 GB 级 |
>
> **KV cache 是每请求独立的**——每条请求有自己的 K/V 张量（因为每条请求的前文不同），不能共享。所以 batch 越大，KV cache 显存占用越大，这正是 **batch 不能无限大的硬约束**：显存被 KV cache 撑爆 → OOM。公式见 [[KV cache management]] §1：$M_{\text{KV}} = 2 B L n_{\text{kv}} S d_h b$，$B$ 在分子上。
>
> ### 三、所以更准确的类比
>
> 不是"两个矩阵相乘、显存全固定"，而是：
>
> > **权重矩阵 $W$ 固定（显存固定，只读一次）；输入矩阵 $X$ 的行数 $B$ 可变（计算量随 $B$ 线性涨）；同时每行 $X$ 还拖着一个随它走的 KV cache（显存也随 $B$ 涨）。**
>
> 增大 batch 的收益：权重那部分访存被摊薄 → 算术强度 $\uparrow$ → 利用率 $\uparrow$。
> 增大 batch 的代价：KV cache 显存 $\uparrow$（可能 OOM）+ 凑批等待时间 $\uparrow$（延迟 $\uparrow$）。
>
> 这正是 [[batching tradeoff]] 的两面：**收益来自权重固定（你抓对的），代价来自 KV cache 不固定（你要修正的）**。dynamic batching 的双触发就是在"凑大 batch 多摊薄权重"和"别让请求等太久/别撑爆显存"之间权衡。
>
> ### 四、一句话校准
>
> **你的矩阵乘法直觉对——权重 $W$ 是那个固定矩阵，batch $B$ 是变长维度，$B$ 越大一次算越多、权重访存不变。但"显存固定"只对权重成立，KV cache 随 batch 线性涨才是 batch 不能无限大的原因。** 把这个修正补进去，你对 batching 的理解就闭环了。
>
> 相关：[[GPU utilization]] §4.2（大 batch 摊薄权重读推导）、[[KV cache management]] §1（KV 随 batch 线性涨公式）、[[batching tradeoff]]（收益 vs 代价的两面）。

> [!note] 三句话定位
> - **是什么**：到达时攒批，阈值或超时触发即发，整批跑完才换下一批。
> - **为什么**：单请求 GPU 利用率低，攒批提吞吐；但等满批又增延迟——dynamic 在两者间权衡。
> - **和 continuous 的区别**：dynamic 是"何时凑批"（请求级），continuous 是"批内如何调度"（token 级）。两者可叠加。


## 2. 为什么需要它（动机与背景）

### 2.1 三个极端都不好

推理服务的请求到达是**随机流**（Poisson 近似）。处理方式有三档：

| 策略    | 做法         | 问题                      |
| ----- | ---------- | ----------------------- |
| 逐请求串行 | 来一个算一个     | GPU 利用率极低（batch=1，算力闲置） |
| 固定大小批 | 攒够固定 K 个才发 | 低峰时永远凑不齐，请求饿死           |
| 立即发   | 来一个发一个     | 等同串行，没攒到批               |

**dynamic batching** 用"**阈值或超时双触发**"破局：

- 凑够 `max_queue_size` 个 → **立即发批**（不等超时，吞吐优先）。
- 没凑够但等够 `max_queue_delay` → **超时发批**（用当前积压的请求数发，延迟优先）。

这样高峰时按阈值快速发批（高吞吐），低谷时按超时发批（低延迟），自适应负载。

### 2.2 与 static / continuous 的位置

| 维度 | static（固定批） | dynamic（请求级动态凑批） | continuous（token 级迭代调度） |
|---|---|---|---|
| 凑批时机 | 固定大小，满才发 | 阈值或超时双触发 | 不"凑批"，slot 边跑边换 |
| 调度粒度 | 请求级 | 请求级 | token/iteration 级 |
| 换批时机 | 整批跑完 | 整批跑完 | 每 iteration |
| 适合 | 离线批量 | 传统模型服务（CV/NLP 分类） | LLM 生成（变长输出） |

> [!tip] LLM 场景下 dynamic 为何不够
> 传统模型（ResNet/BERT 分类）输出定长，一批跑完时间相近，dynamic 够用。但 LLM 生成**输出长度差异巨大**（20 vs 800 token），dynamic 的"整批跑完才换"会让短请求空等长请求（[[continuous batching]] §2.1 的痛点）。所以 LLM 服务转向 continuous batching（token 级），dynamic 退居为"到达层凑批"的辅助。


## 3. 核心概念详解

### 3.1 双触发：max_queue_size + max_queue_delay

- **max_queue_size**（凑批阈值）：积压请求数达到此值立即发批。控制吞吐——高峰时快速凑齐大批。
- **max_queue_delay**（超时阈值）：最老请求在队列里等够此时间就发批（无论攒了多少）。控制延迟——低谷时不让请求干等。

两者任一满足即发。这构成一个**自适应**：负载高时 size 触发（大批高吞吐），负载低时 delay 触发（小批低延迟）。

### 3.2 Triton 的实现

NVIDIA Triton Inference Server 的 dynamic batching：

- 配置 `max_queue_delay_microseconds` 与 `max_batch_size`。
- 后台线程持续检查队列，双触发即发。
- 支持 `priority` 队列、`preserve_ordering`（保序）、`preferred_batch_size`（多个候选大小，如 [2,4,8]，攒到任一即发以利用 GPU 不同 batch 的吞吐曲线）。

### 3.3 preferred_batch_size（优选批大小）

GPU 对不同 batch 大小有吞吐"甜点区"（如 batch 8 和 16 都很高，batch 12 反而因 padding 低效）。`preferred_batch_size=[2,4,8]` 让 batcher 攒到 2/4/8 任一就优先发，否则超时兜底。这是对朴素双触发的细化。

### 3.4 与 continuous 的叠加

在 LLM 服务里，两层可叠加：

1. **外层 dynamic**：请求到达层，按阈值/超时把请求送进推理引擎的等待队列。
2. **内层 continuous**：引擎内 token 级迭代调度，slot 动态进出。

即 dynamic 决定"哪些请求一起进引擎"，continuous 决定"进引擎后怎么调度"。vLLM 主要实现内层 continuous，外层由前端（如 LitServe/Triton frontend）补。


## 4. 数学原理 / 公式

### 4.1 Little's Law（排队论基础）

稳态排队系统：$L = \lambda W$

- $L$ = 系统内平均请求数（队列+处理中）
- $\lambda$ = 请求到达率（req/s）
- $W$ = 平均逗留时间（等待+处理）

dynamic batching 的目标是在 $W$（延迟）与 $\lambda$（吞吐）间权衡。批大小 $B$ 越大，单次前向吞吐越高（GPU 利用率高），但凑批等待时间 $\uparrow$，$W \uparrow$。

### 4.2 凑批等待时间

设到达率 $\lambda$，凑批阈值 $B$。近似下，攒满 $B$ 个的期望等待：

$$
E[\text{凑批等待}] \approx \frac{B}{\lambda}
$$

超时阈值 $T_{\max}$ 兜底：实际凑批等待 $= \min(B/\lambda, T_{\max})$（粗略）。

- 高负载（$\lambda$ 大）：$B/\lambda < T_{\max}$，size 触发，等待短，批大。
- 低负载（$\lambda$ 小）：$B/\lambda > T_{\max}$，delay 触发，批小但延迟有上界。

### 4.3 吞吐-延迟权衡

单次前向耗时 $T_{\text{fwd}}(B)$（随 $B$ 缓增，因 GPU 并行）。吞吐 $\approx B / T_{\text{fwd}}(B)$（$B$ 大时趋于 GPU 峰值）。延迟 $W \approx \text{凑批等待} + T_{\text{fwd}}(B)$。

增大 $B$：吞吐 $\uparrow$，但凑批等待 $\uparrow$、延迟 $\uparrow$。这就是 [[batching tradeoff]] 的本质——dynamic batching 是这一权衡的请求级实现。


## 5. 代码示例（可选

### 5.1 简化 dynamic batcher（双触发）

```python
import time

class DynamicBatcher:
    """简化 dynamic batching: max_size 或 max_delay 任一满足即发批."""
    def __init__(self, max_size=4, max_delay=1.0):
        self.max_size, self.max_delay = max_size, max_delay
        self.queue, self.batch_id = [], 0
    def submit(self, req):
        self.queue.append((time.time(), req))
    def try_flush(self, now):
        """检查是否应发批, 返回 (batch_id, batch) 或 None."""
        if not self.queue: return None
        oldest = self.queue[0][0]
        if len(self.queue) >= self.max_size or (now - oldest) >= self.max_delay:
            batch = [r for _, r in self.queue]
            self.queue, self.batch_id = [], self.batch_id + 1
            return (self.batch_id, batch)
        return None

# 模拟请求随时间到达
b = DynamicBatcher(max_size=3, max_delay=1.0)
arrivals = [(0.0,'A'),(0.2,'B'),(0.4,'C'),(0.6,'D'),(1.2,'E'),(1.4,'F')]
t = 0.0
for at, r in arrivals:
    while t < at:                       # 推进时间, 期间检查超时触发
        flushed = b.try_flush(t)
        if flushed: print(f't={t:.1f}: flush #{flushed[0]} = {flushed[1]}')
        t += 0.2
    b.submit(r); print(f't={at:.1f}: submit {r}, queue={len(b.queue)}')
# 收尾: 把剩余的冲走
for _ in range(10):
    flushed = b.try_flush(t)
    if flushed: print(f't={t:.1f}: flush #{flushed[0]} = {flushed[1]}')
    t += 0.2
```

预期：A,B,C 攒满 3 → size 触发发批 #1；D 单独等到 t≈1.6 超时 → delay 触发发批 #2；E,F 类似。这演示了双触发的自适应。

### 5.2 对比：static 与串行

```python
def static_batch(requests, K):
    """固定大小 K, 攒满才发."""
    return [requests[i:i+K] for i in range(0, len(requests), K)]
# 低负载时永远凑不满 -> 请求饿死

def serial(requests):
    """逐个处理."""
    return [[r] for r in requests]
# GPU 利用率最低
```


## 6. 与其他知识点的关系

- **上游（依赖）**: 无强依赖（通用调度概念），但落地在推理服务框架（Triton）。
- **下游（应用）**: [[batching tradeoff]]（dynamic 是请求级权衡的实例）、[[continuous batching]]（LLM 场景的 token 级升级）、[[GPU utilization]]（dynamic 提利用率的目标）。
- **对比 / 易混**:
  - **dynamic batching vs [[continuous batching]]**：**最常被混淆的一对**。dynamic = 请求级凑批（何时攒批），整批跑完才换；continuous = token 级调度（批内怎么换人），每 iteration 可换。粒度差一个量级。
  - **dynamic batching vs static batching**：dynamic 用阈值/超时双触发自适应凑批，static 用固定大小满才发。
  - **dynamic batching vs in-flight batching**：in-flight ≈ continuous（token 级），不是 dynamic。


## 7. 常见误区与易错点

> [!warning] 误区 1：把 dynamic batching 等同于 continuous batching
> **这是最常见的混淆**。两者都叫"动态批"，但：
> - dynamic 的"动态"指**凑批时机动态**（阈值/超时），凑够后**整批一起跑完才换**。
> - continuous 的"动态"指**批内成员动态**（每 token 边界换人）。
> 一个是"何时凑批"，一个是"凑批后怎么调度"。LLM 生成场景因输出变长，dynamic 的整批跑完会拖累短请求，故转向 continuous。

> [!warning] 误区 2：以为 dynamic batching 是 LLM 推理的"主力"
> 在 LLM 生成服务里，主力是 [[continuous batching]]（token 级）。dynamic 更多用于传统模型（分类/检测，定长输出）服务，或作为 LLM 服务的外层到达层凑批。把它当 LLM 推理核心调度就搞错了层次。

> [!warning] 误区 3：忽略 max_queue_delay 的兜底作用
> 只设 max_queue_size 不设 delay，低负载时请求永远凑不齐 → 饿死。delay 是延迟上界保障，不可省。

> [!warning] 误区 4：以为 batch 越大吞吐一定越高
> 大 batch 提利用率，但凑批等待也增（延迟↑），且超过 GPU "甜点区"后 padding/显存开销反噬。`preferred_batch_size` 就是为此选甜点。


## 8. 延伸细节

### 8.1 Triton 的 dynamic batching 配置

```protobuf
dynamic_batching {
  preferred_batch_size: [2, 4, 8]
  max_queue_delay_microseconds: 100000   # 100ms
  preserve_order: true                   # 保序（同请求内先来先服务）
}
```

Triton 还支持 `priority` 队列（高优先请求先入批）、`queue_policy`（超时丢弃 vs 保留）。

### 8.2 LLM 场景：dynamic 为何退居辅助

LLM 生成的输出长度随机（20–2000 token 不等）。dynamic 凑批后整批跑，最短请求生成完仍要等最长请求，slot 大量空转（[[continuous batching]] §2.1）。所以 LLM 服务（vLLM/TGI/SGLang）用 continuous batching 做内核调度，dynamic 只在前端做请求准入/限流。

### 8.3 与 RLHF 的关系

[[rollout worker]] 采样时，若用第三方推理服务（Triton 部署的模型）做生成，dynamic batching 决定请求如何凑批进引擎；引擎内若是 LLM 专用（vLLM），则 continuous 接管。两层共同决定 [[sampling throughput]]。在 [[policy deployment (PD)]] 架构里，PD 服务侧的这两层调度直接影响 RLHF 训练的采样瓶颈。

---
相关: [[推理优化]] | [[continuous batching]] | [[batching tradeoff]] | [[GPU utilization]]
