# 多模态 sequence packing

> **所属章节**: [[多模态batching]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: 多模态 sequence packing / cross-sample packing / 跨样本 padding 消除 / 视觉+文本联合打包 / multimodal packing / cross-sample attention isolation / block-diagonal packing for VLM / NaViT packing 的图文联合版
> **难度**: 中高（需懂 [[sequence packing与动态采样]]、[[batching strategy]]、[[dynamic batching]]、[[continuous batching的调度实现]]、[[ViT与vision encoder]]、[[projector]]、[[自注意力]]、[[attention mask]]、[[位置编码]]）


## 1. 一句话定义

**多模态 sequence packing（多模态序列打包 / cross-sample packing / 跨样本 padding 消除）** 是 VLM 训练 / 推理 batching 的核心优化——把**多条变长的"视觉 token + 文本 token"样本**拼进**一个定长序列**（而非传统把每条样本 padding 到 batch 内最长那条），用 **block-diagonal attention mask**（块对角注意力掩码）**隔离不同样本**（让样本 A 的视觉 / 文本 token 不能 attend 样本 B 的任何 token），从而**消除 padding 浪费、提升 token 密度**（useful tokens / total → 接近 100%）、把长短不一的图文样本混 pack 做负载均衡；它与纯文本 packing 的根本区别是：多模态 packing **必须正确维护"视觉 token 与对应文本 token 的归属关系"**——哪段视觉 token 属于哪张图、哪段文本 token 与它配对、position id 怎么重置（视觉段与文本段各自从 0 起还是连续累加）、loss 只算文本生成部分（视觉 token 是输入条件不是预测目标），任何归属错位都会让图文对齐失效甚至发生 **cross-sample attention 污染**（不同样本的图被同序列里另一样本的文本"看见"，模型偷看别家的图）。

> [!note] 三句话定位
> - **是什么**：多条变长图文样本的视觉 token + 文本 token 拼进定长序列，block-diagonal mask 隔离各样本（消除 padding、提密度、长短混 pack 均衡），position id 重置 + label mask 只算文本生成。
> - **为什么**：VLM 的视觉 token 数随分辨率爆炸且样本间极度不均（一张高清图变上千 token、一张小图几十 token、文本又几百~几千 token），传统 padding 到最长可浪费 60–80% 算力 / 显存；packing 把这些"碎片"拼满。
> - **与 [[sequence packing与动态采样]] 关系**：后者是纯文本 RL 采样的 packing（每条 response 有自己的 reward / advantage / logp，block-diagonal 隔离）；本条是它的**多模态版**——多了视觉段、图文对齐、视觉段不计 loss。两者的 block-diagonal mask、position id 重置、padding 浪费率公式完全同构，本条在其上叠加"视觉 + 文本联合"维度。


## 2. 为什么需要它（动机与背景）

### 2.1 多模态样本长度极度不均——比纯文本更夸张

纯文本预训练 / RL 的 token 长度虽变长，但量级相对可控（数学题 200–4096 token）。VLM 的"视觉 token 数"由图像分辨率决定（详见 [[动态分辨率与视觉token]]、[[ViT与vision encoder]]）：

$$
N_{\text{visual}} = \frac{H}{P}\cdot\frac{W}{P}
$$

其中 $P$ 是 patch 大小。一张 224×224 图 $N=16^2=256$；一张 448×448 tile $N=32^2=1024$；一张高清文档经 InternVL **tile 切分**（如 6 tile）可达 $N=6\times 1024=6144$ 视觉 token；再经 [[projector]] 的 resampler 可能压到固定 256，但用 MLP projector 不压（LLaVA-NeXt）就原样进 LLM。

于是一个图文样本的总长 $L = N_{\text{visual}} + L_{\text{text}}$ 在 batch 内**剧烈变化**：

| 样本 | 图分辨率 | 视觉 token $N_v$ | 文本 token $L_t$ | 总长 $L$ |
|---|---|---|---|---|
| 小图 + 短问 | 224² | 256 | 32 | 288 |
| 高清文档 + 长答 | 6×448² | 6144 | 512 | 6656 |
| 中图 + VQA | 1×448² | 1024 | 80 | 1104 |

batch 内最长 6656、最短 288，**长度差 23 倍**。这是 VLM 比 NLP 预训练更严重的 batching 痛点——视觉段长度随分辨率离散且不可预测。

### 2.2 传统 padding 的浪费

传统 batching 把 batch 内 $B$ 条样本 padding 到最长 $L_{\max}$，堆叠成 $[B, L_{\max}]$ tensor。padding 浪费率（推导见 [[sequence packing与动态采样]] §4.1，本条复用）：

$$
\boxed{\text{waste}_{\text{pad}} = 1 - \frac{\sum_{i=1}^{B} L_i}{B \cdot L_{\max}} = 1 - \frac{\bar L}{L_{\max}}}
$$

代入上表 4 条样本（$L \in \{288, 6656, 1104, 1104\}$）：$\bar L = 9152/4 = 2288$，$L_{\max}=6656$，**浪费率 $= 1 - 2288/6656 = 65.6\%$**——近三分之二的算力 / 显存耗在 pad token 上。

> [!warning] 多模态 padding 浪费比纯文本更狠
> 纯文本 batch 变长系数通常 2–4×（浪费 ~50–75%）；VLM 因视觉 token 数随分辨率离散（256 → 6144，差 24×），**浪费常达 60–80%，极端 case 接近 95%**。每条高清样本会把整个 batch 拖到它的长度，短样本全在 pad。这是多模态 batching 的头号吞吐杀手。

### 2.3 padding 不只浪费算力——还占显存、撑爆 attention

padding 的危害不止 FLOPs 浪费：

1. **显存浪费**：pad token 要占 KV cache（推理时）与 activation（训练时）。VLM 高分辨率 batch 的 pad token 可达数千，白白吃掉 GB 级显存，挤掉能多塞样本的空间。
2. **attention $O(L^2)$ 膨胀**：padding 到 $L_{\max}$ 后 self-attention 是 $O(L_{\max}^2)$，即使 pad token 被 mask 掉（不参与输出），**矩阵乘法照样算**（FLOPs 照付），且 pad token 的 K/V 仍被生成、仍占显存。
3. **batch 吞吐崩**：batch 内一旦混入一条超长高清样本，整个 batch 被拖到它的长度，短样本全空转，GPU 利用率断崖式下跌。

### 2.4 cross-sample padding 的真正危害：attention 污染

如果把多条样本 padding 进同一序列而不加 block-diagonal 隔离（错误做法），更致命的问题出现——**cross-sample attention 污染**：

样本 A 的文本 token 去 attend 样本 B 的视觉 token（B 的图被 A "看见"）。模型不再靠自己的图回答，而是"偷看"邻居的图。这会让：

- **图文对齐失效**：A 的文本与 B 的图混 attention，[[projector]] 对齐的视觉特征被串味；
- **训练信号污染**：A 的文本生成 loss 受 B 的图影响，梯度方向错误；
- **推理泄漏**：A 的回答基于 B 的图（多用户 serving 场景，请求 A 偷看请求 B 的图，是**隐私泄漏**）。

> [!warning] 误区：pack 就是"拼一起"——错！
> 多模态 packing **不是简单把多条样本首尾相接拼成一个长序列**就完事。**必须同时构造 block-diagonal attention mask**，把每条样本圈在自己的 attention 块里，否则发生 cross-sample 污染（上图所示）。"拼一起但不隔离" = cross-sample padding，是 packing 最常见实现 bug，比传统 padding（各样本独立、pad 不泄漏）反而更危险——传统 padding 至少各样本 attention 互不交叉。

### 2.5 packing 的解：拼满 + block-diagonal 隔离

正确 packing：把多条变长样本**首尾相接**拼进定长 $S$（如 $S=8192$），同时构造**块对角 attention mask**——对角线上每个块是一条样本的 full attention（内部自由交互），块外全 0（跨样本不可见）：

```
样本A (L_A)     样本B (L_B)     样本C (L_C)
┌────────────┐                                  ← A 内部 full attention
│   block A  │ 0 0 0 ...                       
├────────────┤────────────┐                    
│ 0 0 0 ...  │   block B  │ 0 0 0 ...          ← B 内部 full, 跨样本 0
├────────────┤────────────┤────────────┐       
│ 0 0 0 ...  │ 0 0 0 ...  │   block C  │       ← C 内部 full
└────────────┴────────────┴────────────┘
```

结果：**总 token $\sum_i L_i$ 全是有效 token**（无 pad），密度 $\to 100\%$；各样本内部自由 attention，跨样本零污染。

### 2.6 多模态 packing 比 pure-text 多的两个约束

多模态 packing 在纯文本 packing（见 [[sequence packing与动态采样]]）基础上多两条：

1. **图文归属对齐**：一条样本内"视觉段 + 文本段"必须保持配对——视觉 token $v_{i,1..N}$ 与文本 token $t_{i,1..L}$ 属同一样本 $i$，拼序列时不能错位（视觉段不能被隔壁样本的文本"插队"）。block-diagonal 块要**覆盖整条样本的视觉+文本**，不能只盖文本。
2. **label mask 只算文本生成**：VLM 的训练目标是**生成文本**（视觉 token 是输入条件，不是预测目标）。packing 后 loss 只在**文本段的生成位置**算（视觉段、prompt 段的 label 设 `-100`/ignore）。若 label mask 没按样本边界正确切片，会把视觉 token 当生成目标算 loss，模型去"预测图"，崩。


## 3. 核心概念详解

### 3.1 纯文本 packing vs 多模态 packing

| 维度 | 纯文本 packing（[[sequence packing与动态采样]]） | 多模态 sequence packing（本条） |
|---|---|---|
| 序列组成 | 文本 token 拼接 | 视觉 token + 文本 token 拼接（每样本内图文同块） |
| 长度变差来源 | 文本长度（2–4×） | 视觉 token 数随分辨率（24×+）+ 文本长度 |
| padding 浪费 | ~50% | **60–80%**（更严重） |
| block-diagonal 边界 | 每条 response 一个块 | 每条样本（视觉+文本）一个块 |
| position id 重置 | 每条从 0 起 | 视觉段 / 文本段策略可分（见 §3.4） |
| label mask | 全 response 算 loss | **只文本生成段算，视觉段 ignore** |
| RL 额外约束 | per-trajectory reward/advantage/logp 隔离 | 视觉条件 + 文本 reward 分离 |
| 典型场景 | LLM-RL rollout、预训练 packing | VLM SFT / 预训练 / RL batching |

### 3.2 block-diagonal attention mask 构造

设 batch 内 $B$ 条样本，第 $i$ 条长 $L_i$，packing 后总长 $S = \sum_i L_i$。block-diagonal mask $M \in \{0,1\}^{S\times S}$：

$$
M_{p,q} = \begin{cases} 1 & \text{if } p,q \text{ 同属样本 } i \text{（即 } \sum_{j<i} L_j \le p,q < \sum_{j\le i} L_j\text{）} \\ 0 & \text{otherwise} \end{cases}
$$

即对角线上按 $L_1, L_2, \ldots, L_B$ 切 $B$ 个全 1 块，块外全 0。实现上用 `torch.block_diag` 或手动 scatter（见 §5.1）。

> [!note] 与 causal mask 的叠加
> 若是 decoder-only VLM（主流，见 [[decoder-only architecture]]），还要在 block 内叠加 **causal mask**（下三角，token 只 attend 自己及之前）：
> $$ M_{p,q}^{\text{final}} = M_{p,q}^{\text{block}} \cdot \mathbb{1}[q \le p] $$
> 即"块对角 × 因果"。视觉段通常是**双向 / prefix**（不被因果约束，整个视觉段可 full attention 当 KV 被 text query），文本段才因果。这就引出 **prefix-LM 风格 mask**（视觉段双向、文本段因果），详见 §3.5。

### 3.3 position id 重置

packing 后若 position id 不重置（直接用全局 0..S-1），第 $i$ 条样本的 token 会从 $\sum_{j<i} L_j$ 起编号——position id 被前面样本"顶"到很大的值。后果：

1. **位置编码超训练分布**：LLM 训练时没见过 $L > L_{\max}^{\text{train}}$ 的 position id，RoPE / ALiBi 外推崩；
2. **跨样本位置耦合**：样本 $i$ 的位置被前面样本长度影响，同样内容的样本在不同 batch 位置编号不同，语义不稳。

正确做法：**每条样本 position id 从 0 重置**（cyclic / per-sample reset）：

$$
\text{pos}_p = p - \sum_{j<i} L_j \quad \text{for token } p \in \text{样本 } i
$$

即每条样本内部从 0,1,2,... 重新编号。RoPE 据此重算旋转。这与 [[sequence packing与动态采样]] 的 per-trajectory position reset 完全同构。

### 3.4 视觉段 vs 文本段的 position 策略

多模态特有的纠结：一条样本内**视觉段**与**文本段**的 position id 怎么接？

| 策略 | 描述 | 代表 |
|---|---|---|
| **连续编号** | 视觉 0..$N_v$-1，文本 $N_v$..$N_v$+$L_t$-1（连续累加） | LLaVA 系（视觉 token 拼在文本前，position 连续） |
| **分段重置** | 视觉 0..$N_v$-1，文本 0..$L_t$-1（各自从 0） | 部分_prefix-LM VLM |
| **2D 视觉 + 1D 文本** | 视觉用 2D（行列）位置编码（NaViT factorized），文本用 1D | NaViT / Qwen2-VL |

主流 VLM（LLaVA）用**连续编号**——视觉 token 视为"特殊的文本 token 前缀"，position 紧接 system prompt 后。这是 [[early与late fusion]] 的 early fusion 自然产物（视觉 token 拼进 LLM 输入序列）。packing 时 block-diagonal 块覆盖视觉+文本连续段，position 在块内从 0 重置。

### 3.5 prefix-LM 风格 mask（视觉双向 / 文本因果）

主流 VLM 是 decoder-only LLM + early fusion，但视觉段常做**prefix-LM**：视觉段 token 之间双向 attention（图的全局信息自由流动），文本段 causal（生成方向）。block-diagonal 内部还要分：

```
样本 i 块内 (L_i = N_v + L_t):
              视觉段(N_v)      文本段(L_t)
视觉段      ┌────────────┐  
            │  full 2D    │ 0 (causal: 视觉不 attend 未来文本)
            ├────────────┤────────────┐
文本段      │  full       │  causal   │  ← 文本可 attend 视觉(全) + 自己(因果)
            └────────────┴────────────┘
```

即视觉段是双向 full（作 prefix），文本段 causal 且可 attend 整个视觉段。实现：`mask = block_diag(prefix_full_mask per sample)`。这与纯文本 packing 的纯 causal block 不同，是多模态独有。

### 3.6 label mask：只算文本生成

VLM 训练 loss：

$$
\mathcal{L} = -\frac{1}{|\mathcal{T}|}\sum_{p \in \mathcal{T}} \log p_\theta(x_p \mid x_{<p})
$$

其中 $\mathcal{T}$ 是**文本生成位置集合**（不含视觉段、不含 system prompt、不含 pad）。packing 后 label tensor 必须按样本边界正确切片：

- 视觉段位置 label = `-100`（ignore，**不预测视觉 token**）
- 文本 prompt 段 label = `-100`（不预测 prompt）
- 文本 response 段 label = 真实 token id（**算 loss**）

若 packing 时样本边界错位，label mask 会把样本 A 的视觉段算成样本 B 的生成目标，loss 崩。

### 3.7 长短混 pack 做负载均衡

多模态 packing 还能做**长短样本混 pack**——把一条超长高清样本 + 多条短样本拼进同序列，让每条"打包序列"总长接近 $S$（定长上限）。这是 [[batching strategy]] 的 length bucketing 在 token 级的极致版：

- 传统 bucketing：batch 内按长度分桶，桶内 padding 浪费小；
- packing：**跨桶**拼，一条长 + 多条短凑满 $S$，桶都不分了，浪费 $\to 0$。

对 VLM 尤其有效——高清文档（6k token）+ 小图 VQA（288 token）可拼进 8k 序列，密度 97%+。

### 3.8 与 NaViT packing 的关系

[[ViT与vision encoder]] 的 **NaViT**（Native Resolution ViT）本身就用 packing——变长视觉 token pack 进 batch，attention mask 隔离不同图（详见 [[动态分辨率与视觉token]] §"与多模态sequence packing 关系"）。本条是 NaViT packing 的**图文联合版**：NaViT 只 pack 视觉 token（vision encoder 内部），本条 pack 视觉+文本（LLM 输入端）。NaViT 是视觉侧 packing，本条是端到端 VLM packing。两者 mask 构造同构。


## 4. 数学原理 / 公式

### 4.1 padding 浪费率（多模态版）

batch $B$ 条样本，第 $i$ 条 $L_i = N_{v,i} + L_{t,i}$（视觉 + 文本），$L_{\max} = \max_i L_i$。

padding 后总 token $= B \cdot L_{\max}$，有效 token $= \sum_i L_i$。浪费率：

$$
\boxed{\text{waste}_{\text{pad}} = 1 - \frac{\sum_i (N_{v,i} + L_{t,i})}{B \cdot L_{\max}} = 1 - \frac{\bar L}{L_{\max}}}
$$

> [!tip] 多模态版为什么更糟
> 纯文本 $\bar L/L_{\max} \approx 0.3$–$0.5$（浪费 50–70%）；多模态因 $N_{v}$ 随分辨率离散，$\bar L/L_{\max}$ 常跌到 0.1–0.3（**浪费 70–90%**）。一条高清样本把整个 batch 拖垮。

### 4.2 packing 密度提升

packing 后总长 $S = \sum_i L_i$（无 pad，假设恰好填满定长槽）。token 密度：

$$
\boxed{\text{density}_{\text{pack}} = \frac{\text{useful tokens}}{\text{total tokens}} = \frac{\sum_i L_i}{S} = \frac{S}{S} = 1 \quad (\text{理想})}
$$

实际受"凑不满最后一条"影响（最后槽可能不满），密度 $< 1$ 但接近。设 batch 内样本长 $L_1 \le \ldots \le L_B$，packing 进 $K$ 个定长 $S$ 槽，期望密度：

$$
\text{density} \approx 1 - \frac{\mathbb{E}[\text{最后一个槽的余量}]}{K \cdot S} \approx 1 - \frac{L_B/2}{K \cdot S} \to 1 \quad (K \gg 1)
$$

$K$ 越大（batch 越大、样本越多），最后槽余量占比越小，密度 $\to 100\%$。

### 4.3 等效吞吐提升

固定 GPU 算力预算 $C$（FLOPs），padding 模式产 $B_{\text{pad}}$ 条 / iteration，packing 模式产 $B_{\text{pack}}$ 条：

$$
B_{\text{pad}} \cdot L_{\max} \cdot \text{cost/unit} = C \quad\Rightarrow\quad B_{\text{pad}} = \frac{C}{L_{\max} \cdot c}
$$
$$
B_{\text{pack}} \cdot \bar L \cdot c = C \quad\Rightarrow\quad B_{\text{pack}} = \frac{C}{\bar L \cdot c}
$$

等效吞吐比：

$$
\boxed{\frac{B_{\text{pack}}}{B_{\text{pad}}} = \frac{L_{\max}}{\bar L} = \frac{1}{1 - \text{waste}_{\text{pad}}}}
$$

即**吞吐反比于有效占比**。padding 浪费 75%（$\bar L/L_{\max}=0.25$）→ packing 吞吐 **4×**。浪费 90% → packing **10×**。这是多模态 packing 的高价值根源。

### 4.4 cross-sample attention 污染的形式化

错误 packing（拼一起不隔离）：样本 $i$ 的 token $p$ attend 全序列所有 token，attention 权重：

$$
\alpha_{p,q} = \frac{\exp(e_{p,q})}{\sum_{q'=1}^{S} \exp(e_{p,q'})}
$$

$q$ 遍历**全序列**（含别家样本）。设样本 $i$ 视觉段 token $p \in v_i$，它对样本 $j\neq i$ 的视觉 token $q \in v_j$ 的 attention 权重 $\alpha_{p,q}$ 非零——**$i$ 的视觉特征被 $j$ 的图污染**。

正确 packing（block-diagonal）：把 $q$ 求和范围限到同样本：

$$
\alpha_{p,q} = \frac{\mathbb{1}[q \in \text{sample}(p)] \cdot \exp(e_{p,q})}{\sum_{q' \in \text{sample}(p)} \exp(e_{p,q'}}
$$

$q \notin \text{sample}(p)$ 时 $\alpha = 0$，跨样本零污染。block-diagonal 的本质是把 softmax 归一化域**限制在样本内**。

### 4.5 attention FLOPs：padding vs packing

self-attention FLOPs $\propto S^2 \cdot d$（$S$ 序列长）。padding：$S = B \cdot L_{\max}$，FLOPs $\propto (B L_{\max})^2 d$。packing：$S = \sum_i L_i = B\bar L$，FLOPs $\propto (B\bar L)^2 d$。

$$
\frac{\text{FLOPs}_{\text{pack}}}{\text{FLOPs}_{\text{pad}}} = \left(\frac{\bar L}{L_{\max}}\right)^2 = (1 - \text{waste})^2
$$

padding 浪费 75%（$\bar L/L_{\max}=0.25$）→ packing attention FLOPs 仅为 padding 的 $0.25^2 = 6.25\%$。**packing 不止省了 pad token 的算力，连 attention 的 $O(S^2)$ 都因序列变短而骤降**——这是 padding（pad token 虽 mask 但仍参与 $S^2$ 矩阵乘）无法比的。

> [!note] 但 attention 不是唯一开销
> 注意 FFN / embedding 的 FLOPs $\propto S \cdot d^2$（线性于序列长），这部分 packing 与 padding 的"有效 token 数"相同（padding 多算的是 pad token 的 FFN，被 mask 但照算）。故**总 FLOPs 节省介于 $O(\bar L/L_{\max})$ 与 $O((\bar L/L_{\max})^2)$ 之间**，视 attention / FFN 占比而定（大模型 FFN 占比高，packing 省 FLOPs 接近线性；小模型 attention 占比高，接近平方）。但 packing **不省显存里的有效 activation**——它省的是 pad token 的 activation 与 KV（推理）/ activation（训练）。


## 5. 代码示例（可选）

### 5.1 block-diagonal attention mask 构造（含 causal / prefix-LM 叠加）

```python
import torch

def build_multimodal_packing_mask(sample_lens, n_visual_per_sample, device="cuda"):
    """
    多模态 sequence packing 的 attention mask 构造。
    - sample_lens: 每条样本总长 [L_1, ..., L_B] (视觉+文本)
    - n_visual_per_sample: 每条样本视觉段长度 [N_v1, ..., N_vB]
    返回:
      - block_attn: (S, S) block-diagonal + prefix-LM 风格 (视觉双向/文本因果)
      - position_ids: (S,) 每条样本从 0 重置
      - sample_ids: (S,) 每个 token 属于哪条样本 (用于 label 切片)
    """
    S = sum(sample_lens)
    block_attn = torch.zeros(S, S, device=device, dtype=torch.bool)
    position_ids = torch.zeros(S, device=device, dtype=torch.long)
    sample_ids = torch.zeros(S, device=device, dtype=torch.long)
    offset = 0
    for i, L in enumerate(sample_lens):
        Nv = n_visual_per_sample[i]              # 视觉段长
        Nt = L - Nv                              # 文本段长
        a, b = offset, offset + L                # [a, b)
        v_range = slice(a, a + Nv)              # 视觉段
        t_range = slice(a + Nv, b)              # 文本段
        # 视觉段: 双向 full (prefix)
        block_attn[v_range, v_range] = True
        # 文本段 -> 视觉段: 全可见 (prefix)
        block_attn[t_range, v_range] = True
        # 文本段内: causal (下三角)
        tri = torch.tril(torch.ones(Nt, Nt, device=device, dtype=torch.bool))
        block_attn[t_range, t_range] = tri
        # position id 重置 (每样本从 0 起; 视觉连续累加到文本)
        position_ids[a:b] = torch.arange(L, device=device)
        sample_ids[a:b] = i
        offset += L
    return block_attn, position_ids, sample_ids


# 示例: 3 条样本, 视觉段分别为 256/1024/256 token, 文本段 40/80/40
sample_lens = [256 + 40, 1024 + 80, 256 + 40]
n_vis = [256, 1024, 256]
mask, pos, sid = build_multimodal_packing_mask(sample_lens, n_vis)
print(mask.shape, pos.shape)      # torch.Size([1700, 1700]) torch.Size([1700])
# 验证跨样本零污染: 样本0 token 0 attend 样本1 token 应为 False
assert not mask[0, 296].item()    # 296 = 256+40 = 样本1 起点
```

### 5.2 多模态 packing：视觉+文本拼接 + label mask

```python
import torch

def pack_multimodal_batch(samples, S=8192):
    """
    把变长图文样本 pack 进定长 S。
    samples: list of dict, 每条 {visual_ids (Nv,), text_ids (Lt,), text_prompt_len}
      - text_prompt_len: 文本段里 prompt 部分(不预测), 之后是 response(预测)
    返回 input_ids(S,), labels(S,), attention_mask(同 §5.1 block mask)
    """
    input_ids = torch.zeros(S, dtype=torch.long)
    labels = torch.full((S,), -100, dtype=torch.long)   # 默认 ignore
    sample_lens, n_vis = [], []
    offset = 0
    for s in samples:
        v = s["visual_ids"]          # (Nv,)
        t = s["text_ids"]            # (Lt,), 含 prompt + response
        plen = s["text_prompt_len"]  # prompt 长度
        L = len(v) + len(t)
        if offset + L > S:
            break                    # 装不下, 停止 (剩余留下一批)
        # 拼接: 视觉段 + 文本段
        input_ids[offset:offset + len(v)] = v
        input_ids[offset + len(v):offset + L] = t
        # label: 视觉段 -100, prompt 段 -100, response 段 = 真实 id (右移一位预测下一步)
        # 标准 LM: labels = input_ids 右移, 这里简化: response 段直接标
        resp_start = offset + len(v) + plen
        labels[resp_start:offset + L] = t[plen:]   # response 部分算 loss
        sample_lens.append(L)
        n_vis.append(len(v))
        offset += L
    block_mask, pos, sid = build_multimodal_packing_mask(sample_lens, n_vis)
    return input_ids[:offset], labels[:offset], block_mask, pos, sid
```

### 5.3 验证 padding 浪费率与 packing 密度

```python
def padding_waste(sample_lens):
    Lmax = max(sample_lens)
    B = len(sample_lens)
    return 1 - sum(sample_lens) / (B * Lmax)

def packing_density(sample_lens, S):
    # pack 进定长 S 的密度 (含最后槽余量)
    packed, off = 0, 0
    for L in sample_lens:
        if off + L > S:
            break
        off += L
    return off / S

lens = [288, 6656, 1104, 1104]
print(f"padding 浪费: {padding_waste(lens):.1%}")        # ~65.6%
print(f"packing 密度: {packing_density(lens, S=8192):.1%}")  # ~91.8% (9152 装不进 8192, 装满前两条)
# 注意: 实际 packing 会把 288+6656+1104+1104 重新组合凑满多个 8192 槽
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[sequence packing与动态采样]]（纯文本 packing 的基础，block-diagonal mask / position reset / padding 浪费率公式的母本）、[[batching strategy]]（batch 组织策略，packing 是其 token 级优化）、[[dynamic batching]]（请求级动态凑批，packing 是其变长无 padding 拼批的延伸）、[[continuous batching的调度实现]]（推理服务 iteration 级组 batch，PagedAttention 非 contiguous 支持 packing）、[[ViT与vision encoder]]（视觉 token 的来源，决定 $N_v$）、[[projector]]（视觉特征对齐到 LLM 维，packing 在 projector 输出后）、[[自注意力]]（attention $O(L^2)$ 与 mask 机制）、[[attention mask]]（block-diagonal 的实现）、[[位置编码]]（position id 重置与 RoPE 外推）、[[decoder-only architecture]]（causal / prefix-LM mask 叠加）、[[动态分辨率与视觉token]]（视觉 token 数随分辨率离散，packing 的变长根源）。
- **下游（应用）**: [[多模态MoE路由]]（MoE 路由 per-token，packing 后样本边界要传给 router 做模态级负载均衡）、[[VLM RL与多模态reward]]（RL rollout 的多模态 batch packing，每条 trajectory 的 reward / logp 要按 packing 边界分离）、[[新模型接入]]（推理引擎要支持变长图文 packing 的 block mask + position reset）、LLaVA / Qwen2-VL / InternVL 训练与推理 batching、NaViT 视觉侧 packing。
- **对比 / 易混**:
  - **多模态 packing vs [[sequence packing与动态采样]]**：后者纯文本（response 拼接，per-trajectory reward/advantage/logp 隔离）；本条视觉+文本联合（图文归属对齐 + 视觉段不计 loss）。前者是后者的多模态推广，block-diagonal 与 position reset 同构。
  - **packing vs [[dynamic batching]] / [[continuous batching的调度实现]]**：后两者是**推理服务**的请求级 / iteration 级动态凑批（PagedAttention 非 contiguous，请求间天然隔离）；本条是**训练 / rollout**的 token 级拼批（同一序列内多样本，需显式 block mask 隔离）。前者请求级天然隔离，后者 token 级需 mask。
  - **packing vs length bucketing（[[batching strategy]]）**：bucketing 分桶减 padding（桶内仍 padding）；packing 跨桶拼消除 padding（桶都不分）。前者减浪费，后者近零浪费。
  - **多模态 packing vs NaViT packing**：NaViT 是 vision encoder 内部的视觉 token packing（只视觉）；本条是 LLM 输入端的图文联合 packing（视觉+文本）。前者视觉侧，后者端到端。
  - **多模态 packing vs cross-sample padding（错误做法）**：前者 block-diagonal 隔离（正确）；后者拼一起不隔离（错误，污染）。同名易混，前者是后者的"正确版"。


## 7. 常见误区与易错点

> [!warning] 误区 1：packing 就是把样本拼成一个长序列
> **错。** 必须同时构造 block-diagonal attention mask 隔离各样本。"拼一起不隔离" = cross-sample padding，会让样本 A 偷看样本 B 的图（attention 污染 / 隐私泄漏）。正确 packing = 拼满 + block-diagonal 隔离。

> [!warning] 误区 2：视觉 token 也要算生成 loss
> **错。** VLM 训练目标是生成**文本**，视觉 token 是输入条件（经 [[projector]] 对齐的视觉特征），**不是预测目标**。label mask 必须把视觉段标 `-100`（ignore）。若把视觉 token 当生成目标算 loss，模型去"预测图"，崩。多模态 packing 的 label 切片比纯文本 packing 多这条约束。

> [!warning] 误区 3：position id 用全局 0..S-1 不重置
> **错。** 不重置则第 $i$ 条样本 position 从 $\sum_{j<i} L_j$ 起，超 LLM 训练分布（RoPE 外推崩），且同样内容样本不同 batch 位置编号不同。必须**每条样本 position 从 0 重置**。

> [!warning] 误区 4：block-diagonal 等于 causal mask
> **不对。** block-diagonal 是**跨样本隔离**（块外 0），causal 是**块内时序**（下三角）。多模态 decoder-only VLM 要叠加：block-diagonal × (prefix-full on visual + causal on text)。两者正交，缺一不可。

> [!warning] 误区 5：packing 一定省 FLOPs
> **部分对。** attention $O(S^2)$ 因 $S$ 缩短而平方级省（显著）；FFN $O(S d^2)$ 省的是 pad token 的 FFN（被 mask 但照算，省这部分）。**有效 token 的 FFN 不省**（packing 与 padding 的有效 token 数同）。总省介于线性与平方之间。但 packing 必省 pad token 的显存（activation / KV），这是确定收益。

> [!warning] 误区 6：高分辨率样本 packing 就不爆显存了
> **不。** packing 消除 padding 浪费，但**单条超长样本（6k 视觉 token）本身的 activation / attention 仍要算**。packing 让 batch 能塞更多短样本，但长样本自己的 $O(L^2)$ attention 仍在。长样本还需 [[projector]] resampler 压 token + [[动态分辨率与视觉token]] tile 控制，与 packing 正交配合。

> [!warning] 误区 7：block-diagonal mask 全是 0/1，存 bool 就行
> 实现细节：bool mask 占 $S^2/8$ 字节，$S=8192$ 时 ~8MB，可接受；但 $S=32768$ 时 ~128MB，且 attention kernel 要支持。FlashAttention-2/3 原生不支持任意 block-diagonal（需 varlen API：把样本看作独立序列的 batch dim 而非拼接序列）。**生产实现常用 FlashAttention varlen**（`flash_attn_varlen_func`）把 packing 序列当变长 batch 处理，比显式 $S\times S$ mask 快且省显存。这是 packing 工程化的关键。

> [!tip] 实践：用 FlashAttention varlen 替代显式 block mask
> 显式构造 $S\times S$ block-diagonal mask 在长序列（$S>16$k）下显存与 kernel 效率差。**主流做法**：把 packing 序列按样本边界拆成 `cu_seqlens`（累计长度数组），调 FlashAttention 的 varlen API（`flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv)`），它在 kernel 内部按段做 attention、段间零交互，等价 block-diagonal 但无显式 mask。NaViT / LLaVA 训练栈都用此路。


## 8. 延伸细节

### 8.1 FlashAttention varlen 与 packing 的等价性

FlashAttention 的 varlen API 设计初衷就是 sequence packing——把 packing 进的 $S$ 长序列按 `cu_seqlens = [0, L_1, L_1+L_2, ..., S]` 切段，每段独立 attention（段间零交互），数学上**完全等价** block-diagonal mask，但：

- **无显式 $S\times S$ mask**：省 $S^2$ 显存；
- **kernel 内分段**：避免 pad token 参与 softmax（padding 模式 pad token 虽 mask 仍参与 kernel 计算）；
- **支持 causal / prefix-LM**：通过参数控制段内 mask 类型。

这是 packing 能大规模落地的工程基石。详见 [[attention mask]]、FlashAttention 文档。

### 8.2 多模态 packing 在 RL rollout 的额外约束

[[VLM RL与多模态reward]] 的 rollout 也要 packing，但多一层：

- **per-trajectory reward / advantage / logp 隔离**（同 [[sequence packing与动态采样]]）：每条打包的 trajectory 有自己的 outcome reward、GRPO 组内 advantage、`old_log_prob`，不能跨 trajectory 混；
- **视觉条件归属**：trajectory $i$ 的文本生成以**它自己的视觉 token** 为条件，packing 时视觉段与文本段同块、视觉段对文本段可见（prefix），但**不能让 trajectory B 的文本生成看到 A 的视觉 token**（否则 B 的回答基于 A 的图）。

故 VLM RL packing 的 mask 是：**block-diagonal（trajectory 隔离）× prefix-LM（trajectory 内视觉→文本 prefix）× causal（文本生成方向）**。三层叠加。比纯文本 RL packing 多 prefix-LM 层。

### 8.3 与 continuous batching 的关系（推理服务）

推理服务（vLLM / SGLang）的 [[continuous batching的调度实现]] 天然支持变长请求无 padding 拼批（PagedAttention 非 contiguous）。但**推理服务的"packing"是请求级**（每请求独立 KV page，请求间天然隔离，不需 block mask）——与本条的**训练 / rollout token 级 packing**（同序列多样本需 block mask）不同层次：

- **推理 continuous batching**：请求级，PagedAttention 隔离，无 block mask；
- **训练 packing**：token 级，同序列多样本，需 block mask（或 varlen cu_seqlens）。

两者都消 padding，但隔离机制不同（前者靠 page 隔离，后者靠 mask / varlen）。

### 8.4 模态混合 pack：视觉专家调度

[[多模态MoE路由]] 下，packing 后同序列有视觉 token + 文本 token，router 要按模态路由（视觉 token 走视觉专家、文本走文本专家）。packing 的 sample_ids / 模态标签要传给 router 做模态级负载均衡（见 [[负载均衡损失]] 的多模态版）。packing 的边界信息是 MoE 路由的输入。

### 8.5 内容来源

本条整理自 LLaVA / InternVL / Qwen2-VL 训练栈的 packing 实践、NaViT（arXiv:2307.06304）的 packing 机制（[[动态分辨率与视觉token]] 引用）、[[sequence packing与动态采样]] 的纯文本 packing 母本、FlashAttention varlen API 文档、Megatron / DeepSpeed 的 packing 实现。block-diagonal mask 与 padding 浪费率公式复用 [[sequence packing与动态采样]] §4.1–4.4。prefix-LM mask 参照 LLaVA / Qwen-VL 的 attention 设计（视觉双向 / 文本因果）。具体模型版本细节若有出入以最新论文为准。

---
相关: [[多模态batching]] | [[sequence packing与动态采样]] | [[batching strategy]] | [[dynamic batching]] | [[continuous batching的调度实现]] | [[ViT与vision encoder]] | [[projector]] | [[动态分辨率与视觉token]] | [[多模态MoE路由]] | [[VLM RL与多模态reward]] | [[attention mask]] | [[位置编码]] | [[自注意力]] | [[early与late fusion]] | [[新模型接入]]
