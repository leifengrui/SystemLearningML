# MoE token dispatcher

> **所属章节**: [[MoE源码]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: MoE token dispatcher / token dispatcher / token 派发 / All-to-All dispatcher / 专家容量 / aux-loss-free / 无辅助损失负载均衡 / group-limited MoE
> **难度**: 高（需懂 [[Mixture of Experts|MoE]] + [[all-to-all]] + [[EP]] + [[grouped GEMM]]）


## 1. 一句话定义

**MoE token dispatcher** 是 [[Megatron-LM]] 的 **MoE 层 token 派发器**——根据 router（gate）给每个 token 算的**专家亲和度分数**，把 token 按其选中的专家**派发**到该 expert 所在 rank（用 `all-to-all` 通信，每 rank 同时是发送方与接收方，互发各自负责的 token），让目标 expert 在自己 rank 算；派发前对每 expert 设 **capacity（专家容量，每 expert 单步最多处理的 token 数）** 以防负载不均导致的内存爆/计算偏斜（超容量 token **drop 或溢出下一层**，空位 **padding**），传统做法加 **auxiliary loss（辅助负载均衡损失）** 惩罚负载不均驱动均衡。Megatron（与 DeepSeek-MoE）实现 **aux-loss-free（无辅助损失负载均衡）**：不用 aux loss，而用 router 的 **bias 项动态调整**——对负载低的 expert 加正 bias 增加被选概率、高的加负 bias 抑制，bias 是非学习统计量（每步据历史负载调），不影响梯度也不改变 top-k 选择之外的主 loss，从而**无 aux loss 也能均衡**且不拖训练稳定。dispatcher 还支持 **group-limited routing**（DeepSeek 细粒度，把 expert 分组、限制每组选择数）与 **shared expert**（公共 expert 处理共性 token，不参与 dispatch）。是 MoE 层的核心调度组件，决定 token 去哪个 expert、如何 all-to-all 通信、如何防负载不均。

> [!note] 三句话定位
> - **是什么**：router 算分数 → all-to-all 派发 token 到 expert rank → expert 算 → all-to-all 派回；capacity 防偏斜，aux-loss-free 用 bias 动态均衡（不加 aux loss）。
> - **为什么**：MoE 稀疏激活需把 token 送到对应 expert 算；负载不均会爆显存/偏斜，传统 aux loss 拖训练，aux-loss-free 用 bias 替代省 aux loss 不损稳定。
> - **与 [[grouped GEMM]] 关系**：dispatcher 派完 token 后，每 rank 的 expert 用 [[grouped GEMM]]（多 expert 的 batched matmul）算。dispatcher 负责"派"，grouped GEMM 负责"算"。


## 2. 为什么需要它（动机与背景）

### 2.1 MoE 的稀疏激活需派发

[[Mixture of Experts|MoE]] 每 token 只激活部分 expert（top-k）。expert 分布在不同 rank（[[EP]] expert parallel）。token 需派发到选中 expert 所在 rank 算。是 MoE 的核心调度。

### 2.2 all-to-all 通信

每 rank 同时有"该派给 expert 在别 rank 的 token"与"别 rank 派给本 rank expert 的 token"。用 `all-to-all`（每 rank 互发互收）一次完成派发。是 MoE 的通信原语。详见 [[all-to-all]]。

### 2.3 负载不均的显存爆/偏斜

若某 expert 被选多（热门 expert），该 rank 收到 token 超 capacity，显存爆或计算偏斜（其他 rank 空闲）。需 capacity 限制 + 负载均衡。

### 2.4 传统 aux loss 拖稳定

传统 MoE（如 GShard/Switch Transformer）加 auxiliary load balancing loss：惩罚 expert 被选比例不均（$\sum_i f_i \cdot P_i$，$f_i$ = expert $i$ 被选 token 比例，$P_i$ = 平均亲和度）。驱动 router 均衡选。但 aux loss 是额外项，系数难调，过强损主任务，过弱不均衡，且增训练不稳定（loss 多项）。

### 2.5 aux-loss-free 的 bias 替代

DeepSeek-MoE 的 aux-loss-free：不用 aux loss，router 输出加 **bias**（每 expert 一项）。bias 据历史负载动态调（负载低加正 bias 增被选，高加负 bias 抑制）。bias 非 learnable（不进梯度），不影响主 loss 与梯度。top-k 选择时用 `score + bias` 选，自然均衡。无 aux loss 也能均衡，不拖稳定。

### 2.6 group-limited 与 shared expert

DeepSeek 细粒度 MoE：expert 分 group（如 4 group），限制每 group 选数（防全选一组）。shared expert（公共，每 token 都过）处理共性。dispatcher 支持这些细化。是 DeepSeek-MoE 的设计。

### 2.7 capacity 与 padding/drop

capacity 防 token 过载。超 capacity 的 token drop（丢弃，不处理）或 overflow 到下一层（layer norm 后重派）。空位 padding（补零占位，保持 batch 等大便于 grouped GEMM）。是 MoE 的内存稳定机制。


## 3. 核心概念详解

### 3.1 router（gate）

```python
class MoERouter:
    def forward(self, x):  # x: [num_tokens, d]
        scores = x @ self.router_weight  # [num_tokens, num_experts]
        # 加 bias (aux-loss-free)
        scores = scores + self.bias  # bias 动态调
        # top-k 选 (每 token 选 k 个 expert)
        topk_scores, topk_indices = scores.topk(k, dim=-1)
        return topk_scores, topk_indices
```

router 算每 token 对每 expert 的亲和度（linear + softmax/sigmoid）。top-k 选 k 个。bias 项是 aux-loss-free 的关键。

### 3.2 all-to-all 派发

```
设 EP=E (expert 数 = rank 数, 每 rank 一 expert), 每 token 选 top-k.
派发前: 每 rank 持自己 batch 的 token (num_tokens/E 个)
  对每 token, 算它选的 expert 在哪个 rank
派发: all-to-all, 每 rank 把"派给 expert 在 rank r 的 token"发给 rank r,
       同时收"别 rank 派给本 rank expert 的 token"
派发后: 每 rank 收到所有选中自己 expert 的 token (num_assigned 个)
算: 每 rank 的 expert 算收到的 token (grouped GEMM)
派回: all-to-all 反向, 把算完的 token 派回原 rank
```

两次 all-to-all（派发 + 派回）。每 rank 既是发送方又是接收方。是 EP 的通信。

### 3.3 capacity（专家容量）

```python
capacity = 2 * num_tokens_per_rank * top_k / num_experts  # 预估容量, 2x 余量
# 每 expert 最多处理 capacity 个 token
# 超 capacity 的 token: drop (丢弃) 或 overflow
# 空位: padding (补零, 保持 batch 等大)
```

capacity 防 token 过载。超的 drop（信息丢，影响精度）或 overflow 到下一层（不丢，重派）。空位 padding（补零，便于 grouped GEMM 等大）。capacity factor（如 1.5-2）调余量。

### 3.4 aux-loss-free 的 bias

```python
class MoERouter:
    def __init__(self, num_experts):
        self.bias = torch.zeros(num_experts)  # 非 learnable
    def forward(self, x):
        scores = x @ self.weight
        scores = scores + self.bias  # bias 加
        return scores.topk(k)
    def update_bias(self, expert_load):
        # 每 step 据 expert 被选 token 数调 bias
        for e in range(num_experts):
            if expert_load[e] < avg: self.bias[e] += delta  # 负载低, 加正 bias 增被选
            else: self.bias[e] -= delta  # 负载高, 加负 bias 抑制
```

bias 非 learnable（不进梯度），每步据历史 expert_load 调。负载低的 expert 加正 bias（增被选概率），高的加负 bias（抑制）。自然均衡，无 aux loss。bias 调整幅度（delta）与频率调。

### 3.5 传统 aux loss

```python
# 传统 load balancing loss (GShard/Switch)
f = expert_selected_count / total_tokens  # 每 expert 被选比例
P = mean_expert_score  # 每 expert 平均亲和度 (softmax prob)
aux_loss = num_experts * sum(f * P)  # 均衡时大, 不均衡时小? 
# 实际: aux_loss = num_experts * sum(f_i * P_i), 鼓励 f 与 P 一致 (均衡)
loss = main_loss + alpha * aux_loss
```

aux loss 是额外项，系数 $\alpha$ 难调。aux-loss-free 用 bias 替代，省 aux loss。

### 3.6 aux-loss-free vs aux loss

| 维度 | aux loss | aux-loss-free (bias) |
|---|---|---|
| **机制** | 加辅助 loss 项惩罚不均 | router 加 bias 动态调 |
| **梯度** | bias 进主 loss 反传 | bias 非 learnable, 不进梯度 |
| **稳定性** | aux loss 系数难调, 拖训练 | 无 aux loss, 不拖 |
| **均衡** | 靠梯度驱动 | 靠 bias 动态调 |
| **效果** | 均衡但有精度损失 | 均衡且无损 |

aux-loss-free 是 DeepSeek 的改进，省 aux loss 不损稳定。Megatron 实现 aux-loss-free。

### 3.7 group-limited routing

DeepSeek 细粒度 MoE：expert 分 N 组（如 4 组），每 token 每组选 k/N（防全选一组导致偏）。dispatcher 按 group 分 top-k。是 DeepSeek 的负载均衡细化。

### 3.8 shared expert

DeepSeek 的 shared expert：每 token 都过 shared expert（公共，不参与 dispatch），处理共性 token（基础能力）。routed expert（派发的）处理特性。shared + routed 的输出相加。shared 不需 all-to-all（每 rank 都有完整 shared expert）。

### 3.9 padding 与 grouped GEMM 的等大

capacity 限制后，每 expert 收到的 token 数 $\leq$ capacity。空位 padding 补零到 capacity（等大）。便于 [[grouped GEMM]]（batched matmul 需等大或变长支持）。padding 补零不影响算（零 token 算出零）。

### 3.10 实现的 token_dispatcher.py

```python
# megatron/core/moe/token_dispatcher.py
class MoETokenDispatcher:
    def dispatch(self, tokens, scores, indices):
        # 1. 算每 token 派给哪个 rank (据 indices)
        # 2. all-to-all 派发 (token 到 expert rank)
        # 3. capacity 限制 + padding (超 drop, 空 pad)
        return dispatched_tokens  # 每 rank 收到的 token
    def combine(self, expert_output, indices):
        # 1. all-to-all 派回 (算完 token 到原 rank)
        # 2. 按 indices 累加到原 token (top-k 多 expert 输出加权求和)
        return combined_output
```


## 4. 数学原理 / 公式

### 4.1 router 的 score

$$\text{score}_i = \text{softmax}(x W_r)_i \quad \text{或} \quad \sigma(x W_r)_i$$

每 token 对 expert $i$ 的亲和度。top-k 选分数高的 k 个。

### 4.2 aux-loss-free 的 bias

$$\text{score}'_i = \text{score}_i + b_i, \quad \text{top-k 用 score}'$$

$b_i$ 是 bias（非 learnable）。$b_i$ 据 expert $i$ 的历史负载调：负载低 $b_i$ 增（正），高 $b_i$ 减（负）。

### 4.3 bias 的更新

$$b_i \leftarrow b_i + \eta \cdot \text{sign}(\bar{f} - f_i)$$

$\bar{f}$ = 平均被选比例，$f_i$ = expert $i$ 被选比例。$f_i < \bar{f}$（负载低）→ $b_i$ 增（正）。$\eta$ 是 bias 调整率。非 learnable，每步据统计调。

### 4.4 capacity

$$\text{capacity} = c \cdot \frac{N \cdot k}{E}, \quad c = \text{capacity factor (如 1.5-2)}$$

$N$ = token 数/rank，$k$ = top-k，$E$ = expert 数。$c$ 余量防突发。超 capacity drop/overflow。

### 4.5 aux loss（传统）

$$L_{\text{aux}} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i, \quad f_i = \frac{\text{count}_i}{N \cdot k}, \quad P_i = \frac{1}{N} \sum_t \text{score}_{t,i}$$

均衡时 $f_i \approx P_i \approx 1/E$，$L_{\text{aux}} \approx 1$。不均衡时小（$f$ 与 $P$ 错配）。驱动均衡。系数 $\alpha$ 调。

### 4.6 all-to-all 通信量

每 rank 派发 $\propto N \cdot d$（token size）给所有 rank，收 $\propto N \cdot d$（总接收）。单 all-to-all $\propto N \cdot d$（每 rank 发/收总量）。两次（派发+派回）$\propto 2 N d$。是 MoE 的通信大头。见 [[all-to-all]]。


## 5. 代码示例（可选）

### 5.1 Megatron MoE 层

```python
from megatron.core.moe import MoELayer, MoERouter, MoETokenDispatcher
router = MoERouter(d, num_experts, top_k=2, bias=True)  # aux-loss-free
dispatcher = MoETokenDispatcher(num_experts, capacity=..., ep_size=...)
moe = MoELayer(router, dispatcher, experts=[...])
# 前向
out = moe(x)  # router -> dispatch (all-to-all) -> experts (grouped GEMM) -> combine (all-to-all)
# bias 每 step 据 expert_load 更新 (aux-loss-free)
```

### 5.2 aux-loss-free 的 bias 更新

```python
# 每 step 后调 bias
expert_load = dispatcher.get_expert_load()  # 每 expert 被选 token 数
avg = expert_load.mean()
for e in range(num_experts):
    if expert_load[e] < avg: router.bias[e] += delta  # 负载低, 加正
    else: router.bias[e] -= delta  # 负载高, 加负
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Mixture of Experts|MoE]]（MoE 概念）、[[all-to-all]]（派发通信）、[[EP]]（expert parallel）、[[grouped GEMM]]（expert 算）、[[NCCL通信拓扑]]、[[Megatron Core目录与执行链路]]。
- **下游（应用）**: [[Megatron-LM]] 的 MoE 实现、DeepSeek-MoE/MoE 模型训练、[[EP TP DP CP组合规则]]（EP 维的组合）、MoE 推理（[[TP PP DP EP推理]]）。
- **对比 / 易混**:
  - **aux-loss-free vs aux loss**：见 3.6，bias 动态调 vs 加 loss 项。aux-loss-free 省 loss 不拖稳定。
  - **dispatcher vs [[grouped GEMM]]**：dispatcher 派 token（all-to-all + capacity + bias），grouped GEMM 算 expert（batched matmul）。前者派，后者算。
  - **capacity 的 drop vs overflow**：drop 丢超容量 token（信息丢），overflow 重派到下一层（不丢）。drop 简单损精度，overflow 复杂保精度。
  - **aux-loss-free 的 bias vs learnable router**：bias 非 learnable（不进梯度，统计调），router weight learnable（进梯度）。两者不同。bias 是 aux-loss-free 的额外项。


## 7. 常见误区与易错点

> [!warning] 误区 1：aux-loss-free 无均衡
> aux-loss-free 用 bias 动态调（负载低加正 bias），仍均衡，只是不用 aux loss。非"不均衡"。bias 是替代 aux loss 的均衡机制。

> [!warning] 误区 2：bias 进梯度
> bias **非 learnable**（不进梯度，不反传）。每步据 expert 被选统计调。误以为进梯度会与 router weight 混。

> [!warning] 误区 3：capacity 越大越好
> capacity 大防 drop，但占显存（每 expert buffer 大）+ padding 多（空位补零浪费算力）。capacity factor（1.5-2）平衡。误以为越大越好会显存爆。

> [!warning] 误区 4：all-to-all 与 all-reduce 混
> all-to-all 是**每 rank 互发互收不同 token**（置换），all-reduce 是**每 rank 收相同聚合**（求和）。MoE 派发用 all-to-all（token 去不同 expert），DP grad 用 all-reduce（grad 求和）。两者不同。

> [!warning] 误区 5：drop 的 token 信息不丢
> drop 的 token 不被 expert 算，其输出缺（或用残差直通）。是精度损失。drop 多损性能。需 capacity 足够或 overflow。

> [!warning] 误区 6：group-limited 与 aux-loss-free 冲突
> 两者正交。group-limited 限每 group 选数（防偏组），aux-loss-free 用 bias 调负载（防偏 expert）。可叠加。DeepSeek 两者都用。

> [!warning] 误区 7：shared expert 参与 dispatch
> shared expert 是**公共**（每 token 都过，不参与派发）。每 rank 都有完整 shared expert（不 all-to-all）。routed expert 才派发。误以为 shared 也派发会漏配。


## 8. 延伸细节

### 8.1 DeepSeek-MoE 的设计

DeepSeek-MoE 的细粒度（group-limited）+ shared expert + aux-loss-free 是其 MoE 设计三件套。详见 DeepSeek-MoE 论文。Megatron 实现兼容。

### 8.2 capacity 的 dynamic 调

动态 capacity（据负载实时调）比固定更优（防突发 drop），但实现复杂。Megatron 支持固定 capacity，动态在演进。

### 8.3 与 EP 的结合

dispatcher 的 all-to-all 沿 EP 维（expert 在不同 rank）。与 TP/DP/CP 的组合见 [[EP TP DP CP组合规则]]。

### 8.4 内容来源

MoE token dispatcher 整理自 Megatron-LM `megatron/core/moe/token_dispatcher.py` 源码、DeepSeek-MoE 论文（aux-loss-free）、GShard/Switch Transformer 论文（aux loss、capacity）。aux-loss-free vs aux loss 的对比见 DeepSeek-MoE 论文。与 grouped GEMM 的关系见 [[grouped GEMM]]。

---
相关: [[MoE源码]] | [[Mixture of Experts]] | [[all-to-all]] | [[EP]] | [[grouped GEMM]] | [[NCCL通信拓扑]] | [[Megatron Core目录与执行链路]] | [[EP TP DP CP组合规则]] | [[TP PP DP EP推理]] | [[Megatron-LM]] | [[alpha-beta性能模型]] | [[compute vs communication bottleneck]]
