# speculative decoding

> **所属章节**: [[高级推理技术]]
> **所属模块**: [[08-推理系统]]
> **别名**: speculative decoding / 推测解码 / 投机解码 / speculative sampling
> **难度**: 高（需懂 [[KV cache management]] + 拒绝采样 + 分布对齐）
> **2026 生产默认**: EAGLE-3 / EAGLE-3.1

> [!note] 解答：已按批注重写——以 EAGLE-3 为主线重新整理整篇
> 原批注（重写整篇以 EAGLE-3 为主线、可联网）已处理。本次重写以 **EAGLE-3 为主线**重组全文：§2.3 给出演进脉络，§3 以 EAGLE 系列演进（feature 预测→动态树→token+多层 hidden+training-time test）为骨架讲核心概念，§8 整合 2024–2026 联网调研的工程实践、框架支持、RL rollout（verl-SpeCo 等）与训推一致性关系。经典 draft+verify 接受-拒绝原理作为 EAGLE-3 的理论基础在 §3.1/§4 保留。联网来源见 §8.7。

## 1. 一句话定义

**speculative decoding（推测解码）** 是 LLM 推理的**无损加速**技术：用一个轻量 **draft** 快速串行生成 $k$ 个候选 token，再用大 **target model** **一次 forward 并行验证**这 $k$ 个候选（按 target 分布接受-拒绝），把"每步生成 1 token"加速到"每轮 $\approx k \cdot \text{接受率}$ 个 token"，且**输出分布严格等于纯 target 采样**（lossless）。由 Leviathan et al. 2023（Google）与 Chen et al. 2023（DeepMind）独立提出。

> [!note] 三句话定位
> - **是什么**：小模型先猜、大模型并行验、按概率接受-拒绝。
> - **为什么**：大模型 decode 是 memory-bound，单步算力闲置；用小模型猜 + 大模型一次并行验，摊薄访存成本。
> - **关键性质**：lossless——输出分布严格等于 target 模型分布（不是"近似无损"，是数学上严格相等）。

> [!tip] 2026 年的生产形态：EAGLE-3
> 经典 speculative 用独立小 draft model，但 2026 年生产默认已收敛到 **EAGLE-3**：不再用独立小模型，而是在 target 上挂一个**轻量 draft head**，让它预测 target 的内部 hidden state 轨迹（而非离散 token），用**动态草稿树**一次 forward 验证多分支，接受率达 70–80%，加速 3.5–4×（13B 上 5.6× vs vanilla），几乎不占额外 VRAM，预训练 head 拿来即用。本文以 EAGLE-3 为主线，经典两阶段原理作为它的理论基础先讲清。

> [!note] 解答：细说 EAGLE-3 的四个概念——target / 轻量 draft head / hidden state / 一次验证多分支
> 你问的四个点正是理解 EAGLE-3 的"四块拼图"，结论先行：
> - **target** = 被加速的那个大模型本身（如 Llama-3.1-8B/70B），它永远是最终裁判。
> - **轻量 draft head** = 挂在 target 上的**1 层 transformer decoder + FC 投影 + 小 LM head**，约 1–2GB，不是第二个完整模型。
> - **hidden state** = target 每层 forward 后每个位置产出的连续向量（hidden_size 维）；EAGLE-3 从 target 的**3 个层（早/中/晚）**抓 hidden state 拼接投影后喂给 draft。
> - **一次验证多分支** = 靠 **tree attention mask**：把整棵候选树的 token 拼成一条序列一次 forward，用"只许看祖先链"的定制 attention mask 防止跨分支泄漏，于是 target 一次 forward 就并行算出所有候选位置的 logits。
> 下面逐个讲透（2026-07-12 联网核实，来源见末尾）。
>
> ### 一、什么是 target
> **target（目标模型）**就是 speculative decoding 要加速的那个**完整大模型**——它是"最终裁判"，决定输出分布、保证无损。在 EAGLE-3 的配置里，target 是你真正想跑的模型，例如 `meta-llama/Llama-3.1-8B-Instruct` 或 70B。它的角色：
> - **验证 draft 的猜测**：对 draft 提的 $k$ 个候选 token（或多分支树）做**一次 forward**，得到每个位置的 target 分布 $p$，按接受-拒绝规则决定接受哪些。
> - **产出 hidden state 喂给 draft**：EAGLE-3 的 draft head 不是独立模型，它的输入是 target 自己 forward 时中间层产出的 hidden state——所以 draft "寄生"在 target 上，target 既是裁判也是 draft 的信息来源。
> - **决定无损性**：输出分布严格 $=p$（target 分布），draft 只影响速度不影响正确性（§4.3 证明）。
>
> 对比：经典 vanilla speculative 里 target 是大模型、draft 是另一个**独立小模型**（如 70B 配 8B）；EAGLE-3 把 draft 换成 target 上的轻量 head，所以"target + draft head"是一体的，没有第二个完整模型。
>
> ### 二、什么是"轻量 draft head"——为什么叫"轻量"
> EAGLE-3 的 draft 不是独立模型，而是挂在 target 上的一个小模块，结构（来源：arXiv:2503.01840 §3.1、HuggingFace `thoughtworks/GLM-4.7-FP8-Eagle3` config、vLLM speculators docs）：
>
> ```
> target 多层 hidden state (l, m, h) ──concat──> [3k 维] ──FC──> [k 维] g
>                                                              │
>   上一 token embedding e ─────────────concat────────────────┘
>                                                              │
>                                                         [2k 维]
>                                                              │ FC
>                                                              ▼
>                                                       [1 层 Llama decoder]
>                                                              │
>                                                              ▼
>                                                     a ── LM head ──> draft logits
> ```
>
> 具体参数（以 Llama-3.1-8B 的 EAGLE-3 head 为例，来自 HF config）：
>
> | 组件 | 取值 | 说明 |
> |---|---|---|
> | `num_hidden_layers` | **1** | 只有 1 个 Llama-style decoder layer（不是 32 层） |
> | 输入维度 | `2 × hidden_size` | Q/K/V 吃 `[token_embedding, fused_feature]` 拼接 |
> | 辅助层 | `[2, N//2, N-3]` | 从 target 的早/中/晚 3 层抓 hidden state |
> | draft 词表 | 32000（target 可能 151552） | draft 用缩小词表，再映射回 target 词表 |
> | checkpoint 大小 | **~1.2GB** | vs target 8B 的 ~16GB，约 1/13 |
>
> **为什么"轻量"**：① 只有 1 层 decoder（target 是 32 层）；② 参数量是 target 的几十分之一；③ 复用 target 的 hidden state 作输入，不用自己从头算 embedding；④ draft 的 LM head 用缩小词表（32k vs 151k）；⑤ 预训练 head 已发布（RedHatAI speculator models collection，覆盖 Llama-3/Qwen3/gpt-oss/gemma-4/Llama-4-Maverick），拿来即用。它"轻"到几乎不占额外 VRAM（1–2GB），所以能几乎免费地挂上去。
>
> > [!warning] draft head 绑定单一 target
> > 这个 head 是针对**某个具体 target** 训的（学到该 target 的内部 feature 轨迹）。`EAGLE3-LLaMA3.1-Instruct-8B` 只能配 Llama-3.1-8B，换 target 必须换/重训 head。这也是为什么预训练 head 只覆盖主流模型。
>
> ### 三、什么是 hidden state——EAGLE-3 怎么用它
> **hidden state（隐藏状态 / 隐藏向量）**是 transformer 每一层 forward 后、每个 token 位置产出的**连续向量**（维度 = `hidden_size`，如 4096 / 5120）。它是模型内部"思考到一半"的中间表示，介于"原始 token embedding"和"最终 logits"之间。
>
> 关键直觉（来自 EAGLE-1 论文，EAGLE-3 继承并强化）：
> - **token 是离散的**（100k+ 词表里"抽签"），直接预测下一个 token 很难、对齐差。
> - **hidden state 是连续向量**，平滑、信息丰富，比离散 token **好预测得多**——这是 EAGLE 系列接受率（70–80%）远高于经典 draft（~50%）的根因。
>
> **EAGLE-3 相对 EAGLE-1 的关键改动**（arXiv:2503.01840 §3.1）：
> 1. **EAGLE-1 只用 target 顶层（紧挨 LM head 前）的 feature**。但顶层 feature 天然只承载"下一 token"的信息，用来预测"下下 token"信息不够。
> 2. **EAGLE-3 抓 target 的 3 个层**——**early（如第 2 层）/ mid（N//2）/ late（N-3）**——的 hidden state，concat 成 $3k$ 维，经一个 FC 投影回 $k$ 维得到 $g$，融合了低/中/高三级语义信息，给 draft 更丰富的 target 内部轨迹。
> 3. **training-time test**：训练 draft 时就模拟"多步自回归 + 把 draft 自己上一步的输出 $a_{t+1}$ 当下一步输入"（而不是只喂 target 的真实 feature），让 draft 学会处理自己生成时的误差累积，而非只在"喂真实 feature"的温室里训。
>
> 一句话：**hidden state = target 内部层的连续向量表示；EAGLE-3 从 3 个层抓它融合后喂给 draft，让 draft 预测 target 的内部轨迹而不是硬猜离散 token，接受率因此飙升。**
>
> ### 四、为什么一次 forward 能验证多分支——tree attention 的核心
> 这是 speculative decoding 最反直觉的一点。直觉上"分支 A 的 token 怎么能和分支 B 的 token 一起算，它们语境不同啊"。答案：**用定制的 attention mask（tree attention mask）让每个 token 只能看见自己的祖先链，看不见其他分支**，于是把整棵树塞进一次 forward 也不会信息泄漏。
>
> **机制**（来源：arXiv:2603.08088 EAGLE-Pangu §2.4、Red Hat Developer 2025-07、TensorRT-LLM PR #12062）：
> 1. draft 先用动态树展开多个候选分支（EAGLE-2/3 的 dynamic draft tree，按 confidence 决定每节点展开几条）。
> 2. 把树上所有候选 token **按固定顺序排成一条序列**（如深度优先或拓扑序），喂给 target 做一次 forward。
> 3. attention 的 mask 不是标准下三角因果，而是**祖先谓词 mask**：位置 $u$ 可以看位置 $v$ 当且仅当 $v$ 是 $u$ 的祖先（或 $u=v$）。
> 4. 这样每个候选 token 的 hidden state 只依赖"它那条分支的祖先链"，等价于"如果前文是这条分支"的正确上下文，target 一次 forward 就并行算出**所有分支所有位置**的 logits。
> 5. 再按接受-拒绝规则（§4.1）从树里挑最长接受前缀（+bonus token）。
>
> **为什么"一次"就够**：transformer 的 attention 本质是"每个 query 位置对所有 key 位置算权重"，权重由 mask 决定。标准自回归用下三角 mask 保证因果；tree decoding 把 mask 换成"祖先链 mask"，其余计算完全一样——一次矩阵乘就并行处理了所有位置。访存成本（读 KV cache）被摊到所有候选上，这就是 §2.1 说的"用 1 次访存换多个 token 的验证"。
>
> **可跑示意**（树 mask 怎么构造，8 个候选 token 的树）：
> ```python
> import torch
> # 树结构: 每个节点的父节点索引 (-1=根/prefix)
> # 节点 0..7, 父关系如下 (示意一棵 draft 树)
> parents = [-1, 0, 0, 1, 1, 2, 2, 4]   # 节点1,2 是根0的孩子; 3,4是1的孩子; ...
> # 祖先链: u 能看 v  当且仅当 v 是 u 的祖先(含自身)
> def ancestor(parents, u, v):
>     # v 是否为 u 的祖先 (含 u==v)
>     cur = u
>     while cur != -1:
>         if cur == v: return True
>         cur = parents[cur]
>     return False
> M = len(parents)
> mask = torch.full((M, M), float('-inf'))
> for u in range(M):
>     for v in range(M):
>         if ancestor(parents, u, v):
>             mask[u, v] = 0.0   # 只许看祖先链
> print(mask)  # 这就是 tree attention mask, 替代标准下三角因果 mask
> # 用法: target forward 时把这条序列 + 此 mask 喂进去, 一次出所有位置 logits
> ```
> 关键：`mask[u,v]=0` 当且仅当 $v$ 是 $u$ 的祖先；否则 $-\infty$。这样分支 1 的 token（如节点 3、4）看不到分支 2 的 token（节点 5、6），但都能看到公共前缀（节点 0、1、2 中的祖先）。一次 forward 就验完整棵树。
>
> > [!tip] 为什么不总开多分支
> > Red Hat 2025-07 实测：低并发（单请求延迟优先）时 tree decoding 最快；但高 QPS 时验证 64 个候选只为多接受 1–2 个 token，算力不划算，反而拖慢——所以 vLLM 默认 greedy（线性猜 K 个），TensorRT-LLM 用 dynamic tree 在吞吐和延迟间权衡。tree attention 机制能做，不代表所有场景都该开。
>
> ### 五、一句话回扣
> - **target** = 被加速的大模型，最终裁判 + 给 draft 喂 hidden state。
> - **draft head** = target 上挂的 1 层 decoder + FC + 小 LM head，~1GB 级，几乎免费。
> - **hidden state** = target 内部层的连续向量，3 层融合后喂 draft，让 draft 学轨迹而非猜 token。
> - **一次验多分支** = tree attention mask 让每 token 只看祖先链，整棵树塞进一次 forward 不泄漏。
> 四者合起来就是 EAGLE-3 的全部：**target 提供内部 hidden state → 轻量 draft head 自回归猜出候选树 → target 用 tree attention 一次 forward 验完 → 接受-拒绝挑最长前缀，无损且快 3.5–4×**。
>
> **来源**（2026-07-12 联网核实）：
> - [EAGLE-3 论文 arXiv:2503.01840](https://arxiv.org/pdf/2503.01840)（§3.1 推理管线、training-time test、多层 hidden 融合）
> - [vLLM speculators eagle3 docs](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/eagle3/)（draft head 结构：Llama-style 1 层 decoder + FC + LM head）
> - [HuggingFace thoughtworks/GLM-4.7-FP8-Eagle3 config](https://huggingface.co/thoughtworks/GLM-4.7-FP8-Eagle3)（num_hidden_layers=1、辅助层 [2,46,89]、checkpoint ~1.2GB）
> - [Red Hat Developer — Fly Eagle3 Fly Faster 2025-07](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding)（tree decoding 的取舍、greedy vs tree）
> - [EAGLE-Pangu arXiv:2603.08088 §2.4](https://arxiv.org/html/2603.08088)（tree attention mask 祖先谓词的形式化定义）
> - [TensorRT-LLM PR #12062](https://github.com/NVIDIA/TensorRT-LLM/pull/12062)（dynamic tree 实现：build_dynamic_tree_op、treeMask、retrieveIndex）
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

### 2.3 演进脉络：从独立 draft 到 EAGLE-3（2023–2026）

speculative decoding 在 2024–2026 快速演进，主线趋势：**独立 draft model → 挂在 target 上的轻量 head → 预测 feature 而非 token → 动态树 + 训练时对齐**。

| 方法 | 年份 | 核心创新 | 速度提升 | 状态 |
|---|---|---|---|---|
| **vanilla speculative** (Leviathan/Chen) | 2023 | 小 draft model 串行猜 + 大模型并行验 + 拒绝采样保证无损 | 2.0–2.8× | 经典基线，需独立 draft model |
| **Medusa** (Cai et al.) | 2024 | 在 target 上接 **K 个并行预测头**，每头预测 +1/+2/.../+K 位置 token，一次 forward 出 K 候选；tree attention 验证 | 1.8–2.5× | 接受率 50–65%，**正被 EAGLE 取代**，但架构简单、HF/vLLM 原生支持 |
| **EAGLE-1** (Li et al.) | 2024 | 关键洞察：draft 不预测离散 token，**预测 target 倒数第二层 hidden state（feature）**——feature 连续平滑远比 token 可预测；1 FC + 1 decoder layer 的轻量 head，复用 target 的 LM head 解码 draft token；保持无损 | 2.5–3× | 接受率 70–80%，显著高于 Medusa |
| **EAGLE-2** | 2024 | 在 EAGLE-1 上加 **dynamic draft tree**：不再固定线性猜 K 个，按 target confidence 动态展开候选树，一次 forward 验证多分支 | 3.0–3.5× | 主流生产方法之一 |
| **EAGLE-3** (arXiv:2503.01840) | 2025 | **放弃 feature 预测，改回直接 token 预测**，但用**多层 hidden state 聚合**（不只顶层）+ training-time test（训练时就对齐 target 内部轨迹），接受率再上一个台阶 | 3.5–4×（13B 上 5.6× vs vanilla） | **2026 生产默认**，预训练 head 已发布（Llama-3/Qwen3/gpt-oss/gemma-4 等） |
| **EAGLE-3.1** (vLLM blog 2026-05) | 2026 | EAGLE 团队 + vLLM + TorchSpec 协作：FC normalization、post-norm hidden-state feedback、TorchSpec 训练管线，提升鲁棒性与跨模型泛化 | — | vLLM 主推 |

> [!note] 为什么 EAGLE 系列赢了
> 2026 年所有生产指南（localaimaster / inference.net / vizuara）结论一致：**新部署直接上 EAGLE-3**，其他方法基本只剩历史/niche 价值。理由：①接受率最高（每轮验出最多 token）；②几乎不占额外 VRAM（小 head 而非第二模型）；③无 tokenizer 匹配问题（draft token 经 target 的 LM head 解码）；④预训练 head 已发布拿来即用；⑤vLLM 一行 `--speculative-config` 即开。EAGLE-3 的限制：draft head **绑定单一 target**（学到该 target 的内部 feature 轨迹），换 target 必须换 head；小 target + 快 GPU 是"难场景"（8B 在快卡上 decode 本身就快，memory-bandwidth 余量小，draft 自己的 forward 吃掉大部分收益），EAGLE-3 在 **70B+ 大 target** 上收益更大（~4×）。


## 3. 核心概念详解（以 EAGLE-3 为主线）

### 3.1 经典两阶段：draft 生成 + target 验证 + 接受-拒绝（原理基础）

EAGLE-3 仍是"draft 猜 + target 验"的骨架，只是 draft 从独立小模型换成了 target 上的轻量 head。先讲清经典两阶段：

**Stage 1 — draft 生成（串行，轻量快）**：
draft 从当前序列串行生成 $k$ 个候选 token $x_1, \dots, x_k$，每步 $x_i \sim q(\cdot | x_{<i})$，并记录 $q(x_i | x_{<i})$。

**Stage 2 — target 验证（并行，大模型一次 forward）**：
target 对序列 $[\text{prefix}, x_1, \dots, x_k]$ 做一次 forward，得到每个位置的分布 $p(\cdot | x_{<i})$（$i=1..k+1$，最后一个位置是"全部接受后的 bonus"）。

**Stage 3 — 接受-拒绝**：
对每个 $i$，按接受概率决定是否接受 $x_i$：
- 接受 → 继续验证下一个。
- 拒绝 → 从修正分布重采样 1 个 token，终止本轮。

接受 $a$ 个后，本轮产出 $a$ 个（被接受的）+ 1 个（bonus 或重采样）= $a+1$ 个 token。

### 3.2 EAGLE 的关键转向：预测 feature 而非 token

经典 speculative 的痛点：独立小 draft model 要从头预测离散 token，与 target 分布对齐难，接受率低（~50%）。

**EAGLE-1 的洞察**（Li et al. 2024）：让 draft 不要预测离散 token，**预测 target 倒数第二层的 hidden state（feature）**。

- **feature 连续平滑，远比离散 token 可预测**——token 是 100k+ 词表的"离散抽签"，feature 是连续向量，draft 学 target 的内部轨迹比学 target 的输出容易得多。
- draft head 极轻：1 个 FC layer + 1 个 decoder layer，输入是 [target 当前层 feature, 上一 token 的 embedding] 的拼接，输出预测的下一 feature。
- 预测的 feature **复用 target 自己的 LM head** 解码出 draft token——所以 draft token 仍经 target 的真实词表，无 tokenizer 匹配问题。
- 因为 draft 学的是 target 自己的内部轨迹，proposal 与 target 高度吻合，接受率飙升到 70–80%。
- **无损性不变**：draft token 仍经 target LM head 解码、经 target 验证接受-拒绝，EAGLE 的 feature trick 买的是速度，不是正确性，输出分布仍严格 = target。

> [!note] EAGLE-1 vs Medusa
> Medusa 在 target 上接 K 个并行头，每头直接预测某偏移位置的 token（离散），一次 forward 出 K 候选；EAGLE 让一个轻量 head 自回归地预测 feature 再解码 token。Medusa 简单但接受率低（50–65%），EAGLE 因 feature 平滑接受率高（70–80%）。两者都"无独立 draft model、挂在 target 上"，但 EAGLE 的 feature 预测思路赢了。

### 3.3 EAGLE-3：改回 token + 多层 hidden 聚合 + training-time test

EAGLE-3（arXiv:2503.01840, 2025）是当前生产默认，相对 EAGLE-1/2 的关键改动：

1. **放弃 feature 预测，改回直接 token 预测**：EAGLE-1 预测 feature 再经 target LM head 解码，EAGLE-3 直接让 draft head 输出 token logits（但仍用 target 的多层 hidden state 作输入），省去 feature→LM head 的环节，draft 更直接、训练更稳。
2. **多层 hidden state 聚合**：不只取 target 倒数第二层，而是**聚合 target 多个层的 hidden state**（用 FC 层投影拼接），给 draft 更丰富的 target 内部信息，提升对齐度。
3. **Training-time test（训练时就对齐 target 内部轨迹）**：训练 draft head 时不只看最终 token loss，还让 draft 在训练时就经历 target 的内部前向轨迹（"test-time"信息进训练），让 draft 学到的不是"表面模仿 token"而是"复现 target 的推理路径"，接受率再上一个台阶。

结果：13B 模型上 5.6× vs vanilla、1.8× vs EAGLE-1；接受率显著高于 EAGLE-1/2。预训练 EAGLE-3 head 已发布（Llama-3.1-8B、Qwen3-8B、gpt-oss-20b、gemma-4-31B、Llama-4-Maverick 等，RedHatAI speculator models collection），拿来即用。

### 3.4 动态草稿树（EAGLE-2/3）

EAGLE-2 引入、EAGLE-3 继承的 **dynamic draft tree**：不再固定线性猜 $k$ 个 token（一条链），而是按 target 的 confidence **动态展开候选树**——高 confidence 的位置多展开几个分支，一次 target forward 用 tree attention 验证整棵树的所有分支。

- 等效 $k$ 更大但显存预算不变（树共享前缀）。
- 高 confidence 路径接受率高，低 confidence 早停，减少浪费。
- TensorRT-LLM 的 `Eagle3DecodingConfig(use_dynamic_tree=True, dynamic_tree_max_topK, max_total_draft_tokens)` 即此机制。

### 3.5 bonus token（全接受奖励）

若 $k$ 个全部被接受，target 在第 $k+1$ 个位置的分布 $p(\cdot | x_{<k+1})$ 可直接采样一个 bonus token——这是"免费"的，因为 target forward 已经算了这个位置。所以全接受时产出 $k+1$ 个 token。

### 3.6 lossless 性质

通过接受-拒绝 + 重采样的修正，最终每个 token 的输出分布**严格等于** target 分布 $p$（证明见 §4.3）。所以 speculative decoding 不是"近似无损"，而是**数学上严格无损**——用 speculative 与纯 target 采样，得到的序列分布完全相同（只是具体随机数不同）。

> [!warning] 无损性的边界（RL 场景）
> 经典 spec decode 数学上无损（输出分布 = target），但**draft head 训练后冻结、target 在 RL 中演化**会破坏这个保证的工程意义——draft 提的候选经 target 验证仍无损，但 acceptance 掉 = 退化为接近 vanilla 速度。verl-SpeCo 的协同训练正是为维持高 acceptance（见 §8.4）。

### 3.7 方法家族对比（EAGLE-3 之外还有什么）

| 方法 | draft 来源 | 速度 | 训练 | 最佳场景 |
|---|---|---|---|---|
| **EAGLE-3** | target 上的轻量 head（预测 token+多层 hidden） | 3.5–4× | 需训 head（预训练版可即用） | **生产默认**，支持的主流模型 |
| **EAGLE-2** | target 上的轻量 head（预测 feature+动态树） | 3.0–3.5× | 需训 head | EAGLE-3 不可用时的备选 |
| **Medusa** | target 上 K 个并行头 | 1.8–2.5× | fine-tune 1–2 GPU-day | 想要简单架构、无 EAGLE head |
| **vanilla** | 独立同族小模型 | 2.0–2.8× | 无 | target 有强小兄弟（Llama-70B+Llama-8B） |
| **n-gram / Prompt Lookup** | prompt/历史的 n-gram 查找 | 1.5–3.0× | 无、零 VRAM | RAG/代码/重复文本，**零成本首试** |
| **Self-speculation** | target 自己跳层/早退 | 1.5–2.5× | 无 | 无第二模型、零额外 VRAM |
| **MTP** | 模型原生多 token 预测头（DeepSeek V3/Llama-4） | 2–3× | 训练时辅助 loss | 训推一体，原生支持 |
| **Lookahead/Jacobi** | 不动点迭代并行 | 1.5–2.5× | 无 | 无 draft cache 开销 |
| **Suffix+LSTM** | LSTM draft + suffix automaton | 1.5–2.5× | 需训 LSTM | 代码/模板化输出 |


## 4. 数学原理 / 公式

### 4.1 接受-拒绝规则（rejection sampling）

对 draft 采的 $x_i \sim q$，target 有 $p$：

$$
\text{accept } x_i \text{ 以概率 } \min\!\left(1, \frac{p(x_i)}{q(x_i)}\right)
$$

- 若 $p(x_i) \ge q(x_i)$：**必接受**（target 比 draft 更倾向这个 token）。
- 若 $p(x_i) < q(x_i)$：以 $p/q$ 概率接受，$1 - p/q$ 拒绝。

拒绝时，从修正分布重采样：

$$
x' \sim \frac{\max(p - q, 0)}{\sum_x \max(p(x) - q(x), 0)} = \frac{(p - q)_+}{1 - \alpha}
$$

即归一化的 $(p - q)_+$（target 比 draft 多出的"余量"），其中 $\alpha = \sum_x \min(p,q)$。

### 4.2 接受概率与期望接受长度

单步接受概率：

$$
P(\text{accept}) = \mathbb{E}_{x \sim q}\!\left[\min\!\left(1, \frac{p(x)}{q(x)}\right)\right] = \sum_x \min(p(x), q(x)) = 1 - \text{TV}(p, q)
$$

> [!note] 推导
> $\mathbb{E}_{x \sim q}[\min(1, p/q)] = \sum_x q(x) \min(1, p(x)/q(x)) = \sum_x \min(q(x), p(x))$。这正是 $p$ 与 $q$ 的**总变差**关系：$P(\text{accept}) = 1 - \text{TV}(p, q)$。draft 与 target 分布越接近（TV 越小），接受率越高——这就是 EAGLE 让 draft 学 target 内部轨迹的数学根因。

设单步接受概率 $\alpha = \sum_x \min(p, q) \in [0,1]$。$k$ 个位置中接受 $a$ 个的期望（每步独立近似）：

$$
\mathbb{E}[a] \approx \frac{\alpha(1-\alpha^k)}{1-\alpha}, \quad \alpha \to 1 \text{ 时 } \mathbb{E}[a] \to k
$$

产出 token 数 $\approx \mathbb{E}[a] + 1$（bonus 或重采样）。

> [!tip] EAGLE 为何接受率高
> 经典独立 draft 的 $\alpha \approx 0.5$（TV 大），EAGLE 的 draft 学 target 内部 feature 轨迹，$q$ 与 $p$ 高度吻合，$\alpha$ 可达 0.7–0.8，$\mathbb{E}[a]$ 显著提升。EAGLE-3 的多层 hidden 聚合 + training-time test 进一步压 TV。

### 4.3 lossless 证明（重加权）

要证明：最终每个 token 的边缘分布 $\pi(x) = p(x)$。

考虑一轮中产出第一个 token（位置 1）：

- 以概率 $\min(1, p(x_1)/q(x_1))$ 接受 $x_1 \sim q$；
- 否则（概率 $1-\alpha$）从 $(p-q)_+$ 重采样 $x'$。

$$
P(\text{产出 } x) = q(x) \cdot \min\!\left(1, \frac{p(x)}{q(x)}\right) + (1 - \alpha) \cdot \frac{(p(x) - q(x))_+}{1 - \alpha}
$$

其中 $\sum_y (p(y) - q(y))_+ = 1 - \alpha$（因 $\sum p = \sum q = 1$，正部分之和 = 负部分绝对值之和 = $1-\alpha$）。代入化简：

$$
P(\text{产出 } x) = \min(q, p) + (p - q)_+
$$

分情况：
- 若 $p(x) \ge q(x)$：$\min(q, p) = q$, $(p-q)_+ = p-q$，和 $= q + (p-q) = p$。✓
- 若 $p(x) < q(x)$：$\min(q, p) = p$, $(p-q)_+ = 0$，和 $= p + 0 = p$。✓

两情况都 $= p(x)$。故 $\pi(x) = p(x)$，**严格无损**。后续位置同理（条件分布对齐）。

### 4.4 加速比

设 target 单次 forward 耗时 $T_t$（memory-bound，与位置数弱相关），draft 生成 $k$ 个串行耗时 $k \cdot T_d$（$T_d \ll T_t$ 因轻量 head）。一轮产出 $\mathbb{E}[a] + 1$ 个 token，耗时 $T_t + k T_d$。

$$
S = \frac{(\mathbb{E}[a] + 1) \cdot T_t}{T_t + k T_d} = \frac{\mathbb{E}[a] + 1}{1 + k \cdot T_d / T_t}
$$

- 接受率高（$\alpha \to 1$）、draft 快（$T_d/T_t$ 小）→ 加速显著（EAGLE-3 可达 3.5–4×）。
- draft 与 target 分布差异大（$\alpha$ 低）→ 接受少、draft 开销白花，可能负优化。

> [!warning] 加速取决于接受率
> speculative 不是免费午餐。若 draft 太弱（$\alpha$ 低），每轮接受 0–1 个，draft 的 $k T_d$ 开销纯浪费。实践要求 $\alpha$ 至少 0.5+ 才有正收益。EAGLE-3 的高接受率正是其胜出根因。


## 5. 代码示例（可选）

### 5.1 经典 speculative decoding（接受-拒绝核心逻辑）

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

> [!note] 与 EAGLE-3 的对应
> 上面是经典独立 draft model 版。EAGLE-3 把 `draft` 换成"target 上的轻量 head + 多层 hidden 输入"，`draft.logits(cur)` 变成"draft head 基于 target 多层 hidden 预测下一 token logits"，接受-拒绝逻辑完全不变。EAGLE-2/3 还把线性 $k$ 个候选换成动态树（tree attention 一次验证多分支）。

### 5.2 EAGLE-3 生产配置（vLLM / TensorRT-LLM）

```python
# vLLM: EAGLE-3 一行开启 (Llama-3.1-8B 为例)
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    # 关键: 挂 EAGLE-3 draft head
    speculative_config={
        "method": "eagle3",
        "num_speculative_tokens": 3,   # lookahead K, 3 是 8B 甜点(2 时 ~45%, 4 时降到 ~28%)
        # draft head 路径 (RedHatAI 预训练版)
        "draft_model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    },
    gpu_memory_utilization=0.88,
)
out = llm.generate(["解释 speculative decoding"], SamplingParams(max_tokens=200))
```

```python
# TensorRT-LLM: EAGLE-3 + 动态树 + Suffix Automaton 增强
from tensorrt_llm.llmapi import Eagle3DecodingConfig, LLM

spec_cfg = Eagle3DecodingConfig(
    max_draft_len=6,
    speculative_model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    use_dynamic_tree=True,            # EAGLE-2/3 动态草稿树
    dynamic_tree_max_topK=10,        # 每节点最多展开 10 分支
    max_total_draft_tokens=60,       # 树总预算
    use_sa_spec=True,                # Suffix Automaton 增强(重复内容)
    sa_spec_threshold=4,
)
llm = LLM("meta-llama/Llama-3.1-8B-Instruct", speculative_config=spec_cfg)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[KV cache management]]（draft 和 target 各自维护自己的 KV cache，target 验证时要为 $k+1$ 个位置算 KV；EAGLE 的 draft head 复用 target 的 hidden state，KV 管理更省）、[[attention]]（target forward 的算子；tree attention 验证多分支）、autoregressive 生成。
- **下游（应用）**: [[batching tradeoff]]（speculative 在 batch 调度中的位置）、[[GPU utilization]]（speculative 提升 decode 算力利用率）、[[sampling throughput]]（RLHF 中 [[rollout worker]] 用 speculative 加速采样，是 RL rollout 加速关键）、[[policy deployment (PD)]]（PD 服务降延迟）。
- **对比 / 易混**:
  - **speculative vs [[beam search]]**：beam search 是**搜索**（保 top-k 候选序列，追求高概率输出，改变输出分布）；speculative 是**采样加速**（保持采样分布，不改变输出质量，只是更快）。两者正交：speculative 可配合 beam。
  - **speculative vs [[parallel decoding]]**：speculative 用 draft 串行猜 + target 并行验；parallel decoding（Medusa/Lookahead）用模型自身并行头或迭代法，不依赖独立 draft。广义上 Medusa 既算 parallel decoding 也算 EAGLE 系的前身，vLLM 把 Medusa 归为 speculative 的一种。
  - **speculative vs 蒸馏**：蒸馏让小模型永久替换大模型；speculative 让小模型辅助大模型加速，大模型仍是最终裁判，分布不变。
  - **EAGLE-3 vs vanilla**：EAGLE-3 的 draft 是 target 上的轻量 head（共享 target 内部表示），vanilla 是独立小模型；EAGLE-3 接受率更高、无 tokenizer 匹配问题、几乎零额外 VRAM。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 speculative 是"近似无损"
> 不是近似，是**严格无损**。通过接受-拒绝 + $(p-q)_+$ 重采样，输出分布数学上等于 target 分布 $p$（§4.3 证明）。同一 prompt 采样，speculative 与纯 target 得到的序列分布完全相同（仅随机数不同）。它加速但不改变输出质量。

> [!warning] 误区 2：以为任意小模型都能当 draft
> draft 必须与 target **分布相近**（高 $\alpha$）才有正收益。若 draft 太弱（$\alpha$ 低），接受率低，draft 的 $k$ 步开销纯浪费，甚至负优化。经典 vanilla 要求同 tokenizer、同训练数据、规模适配（7B target 配 1B draft，70B 配 7B）；EAGLE-3 用 target 自己的 hidden state 训 draft head，天然对齐，这也是它接受率高的根因。

> [!warning] 误区 3：以为 EAGLE-3 的 draft head 通用
> EAGLE-3 的 draft head **绑定单一 target**——它学到的是该 target 的内部 feature 轨迹。`EAGLE3-LLaMA3.1-Instruct-8B` 只能配 Llama-3.1-8B，换 target 必须换/重训 head。预训练 head 已覆盖主流模型（Llama-3/Qwen3/gpt-oss/gemma-4/Llama-4-Maverick），冷门模型要自己训。

> [!warning] 误区 4：忽略 draft 的 KV cache 管理
> draft 串行生成 $k$ 个时也要维护自己的 KV cache。每轮结束后若 draft cache 失效要重算。EAGLE 因为 draft head 复用 target 的 hidden state，KV 管理开销比独立 draft 小，但仍非零。

> [!warning] 误区 5：以为加速比固定 / $k$ 越大越好
> 加速比随 $\alpha$（分布对齐度）与 $k$ 动态变化。$k$ 太大时若 $\alpha$ 不高，后面位置接受率衰减（$\alpha^i$），浪费 draft 算力。Red Hat 实测 8B 上 $k=2$ 接受 ~45%，$k=4$ 降到 ~28%，**$k=3$ 是 8B 甜点**。大模型（70B）可更大。

> [!warning] 误区 6：小 target + 快 GPU 上盲开 EAGLE-3
> 8B 在快卡（H100）上 decode 本身就快（~45 tok/s），memory-bandwidth 余量小，draft head 自己的 forward 吃掉大部分收益，EAGLE-3 经济性远不如 70B（ Published ~2.4× for 8B vs ~4× for 70B）。小模型快卡是"难场景"，先量化再开。

> [!warning] 误区 7：RL 中 weight sync 后忘重载 draft
> [[weight sync mechanism]] 推新 $\theta$ 给 target 后，EAGLE draft head 若还指向旧 $\theta$ 的 feature 轨迹，acceptance 崩 + 输出偏差。verl PR #5925 的关键 fix：sync 后必须**重载 draft model 权重 + 重建 RoPE cos_sin cache**，否则 draft 跑飞（见 §8.4）。


## 8. 延伸细节

### 8.1 EAGLE-3.1 与 vLLM/TorchSpec 协作

EAGLE-3.1（vLLM blog 2026-05-26）是 EAGLE 团队 + vLLM + TorchSpec 三方协作的产物，改进：
- **FC normalization**：draft head 的 FC 层加 normalization，训练更稳、跨模型泛化更好。
- **post-norm hidden-state feedback**：调整 hidden state 反馈的位置（post-norm），提升 feature 质量。
- **TorchSpec 训练管线**：标准化的 draft head 训练流程，降低用户自训门槛。
- 是 vLLM 2026-05 起主推的 spec decode 实现。

### 8.2 框架支持现状（2026）

| 框架 | 支持方法 | 配置 | 备注 |
|---|---|---|---|
| **vLLM** | vanilla / Medusa / EAGLE-2 / **EAGLE-3（+3.1）** / n-gram / MTP | `--speculative-config {method, num_speculative_tokens, draft_model}` | 生产首选，EAGLE-3.1 是 2026-05 主推 |
| **SGLang** | vanilla / EAGLE | `--spec-draft-model` | 结构化输出场景友好 |
| **TensorRT-LLM** | vanilla / Medusa / **EAGLE-3** + **Dynamic Tree** + **Suffix Automaton** | `Eagle3DecodingConfig(max_draft_len, use_dynamic_tree, dynamic_tree_max_topK, use_sa_spec)` | 原始吞吐天花板，PyTorch backend 仅支持 Eagle3 |
| **HuggingFace Transformers** | vanilla / Prompt Lookup / EAGLE | `assistant_model=` in `generate()` | 最易用，adaptive K |
| **llama.cpp** | vanilla / prompt lookup | `--draft-max` 等 | 边缘部署 |

### 8.3 生产选型决策树（2026）

1. **先试 n-gram / Prompt Lookup**（零训练、零 VRAM）：接受率 >50% 就收工。
2. **<50% 加 EAGLE-2/3**（1–3GB VRAM，2.5–4×）：若有预训练 head 拿来即用。
3. **EAGLE 无预训练 head** → 用同族小 draft model（1–3B）。
4. **都没有** → fine-tune Medusa head（1–2 GPU-day）。

### 8.4 RL rollout 加速（verl-SpeCo 等）—— 2026 新热点

speculative decoding 在 **RL post-training** 里的应用是 2026 年新热点：RL rollout 阶段生成量巨大（PPO/GRPO 每 step 采成千上万条 response），加速 rollout 直接降 RL 训练墙钟时间。关键进展：

1. **verl + EAGLE/EAGLE-3 rollout 加速**（verl PR #5925）：在 vLLM rollout 里挂 EAGLE-3 draft model，实测 **rollout 提速 ~15%**（qwen3-8b + RedHatAI/Qwen3-8B-speculator.eagle3）。
   - **关键工程坑**：① actor weight sync 后必须**重载 draft model 权重**；② 重载后**重建 RoPE cos_sin cache**，否则 draft 跑飞；③ 加 rollout 侧 spec decode 指标（acceptance rate、mean acceptance length）便于训练时监控。
   - **已知问题**：policy drift 后 draft acceptance 会掉（step 100-120 可能 reward collapse），draft 需随训练更新而非静态。

2. **verl-SpeCo**（verl 主仓 2026-06 pre-release）：**co-training framework for speculative decoding across RL training and inference**——在 RL 训练过程中**协同训练 draft model**，让 draft 在 policy 演化时保持对齐、训练后直接复用于加速 serving。核心解决"静态 draft 在 policy drift 后 acceptance 掉"的痛点。仓库 verl-project/verl-SpeCo。

3. **Decoupled Speculation**（arXiv:2511.16193，verl issue #5559 / PR #5757）：把 draft 和 verify **解耦到异步流水线**——drafter 不等上一轮 verify 结果就开始 draft 下一窗口，两阶段在独立 GPU 资源上重叠执行，消除 bubble。verl 社区正在为 SGLang rollout 实现 decoupled scheduler（drafter/verifier 分 Ray actor、共享 weight sync 路径、复用 SGLang 的 EagleDraftInput/EagleVerifyInput 数据结构）。

4. **MTP（Multi-Token Prediction）在训练侧**（verl PR #6432）：把 MTP 既当训练辅助 loss（megatron 侧 loss mask 对齐 + 梯度隔离 functional call）又当 rollout draft source，统一 vLLM/SGLang 后端的 per-request acceptance 指标。DeepSeek V3 / Llama-4 原生 MTP 头让它成为"训练即 draft"的一体化路径。

5. **Speculator 训练框架**（verl PR #4947 RFC）：把 speculator（draft）训练流程与具体算法解耦——`verl/models/speculator/`（LSTM/MLP speculator）+ `verl/trainer/speculators/`（adapter 接口），checkpoint 单独存 `checkpoint/speculator/`，支持 FSDP/Megatron，便于在 verl 里训自己的 draft head。

### 8.5 与训推一致性 / weight sync 的关系（呼应本库主线）

speculative decoding 在 RL 场景引入新的"训推不一致"风险：

- **draft model 与 target 版本对齐**：[[weight sync mechanism]] 推新 $\theta$ 给 target 后，draft head 若还指向旧 $\theta$ 的 feature 轨迹，acceptance 崩 + 输出偏差。verl 的解法是 sync 时**同步重载 draft 权重 + 重建 RoPE cache**（PR #5925 的两个关键 fix）。verl-SpeCo 更进一步——训练时就让 draft 跟 $\theta$ 一起更新。
- **draft 自身的 logp 一致性**：若 draft 参与任何被采样进 buffer 的 token 生成，draft 与 target 的 logp 差异会污染 [[训推不一致]] 的 Pearson 指标，需要在 TIM 治理里把 draft-target 对齐单独算一档。
- **无损性边界**：经典 spec decode 数学上无损，但 **draft head 训练后冻结、target 在 RL 中演化**会破坏这个保证的工程意义——draft 提的候选经 target 验证仍无损，但 acceptance 掉 = 退化为接近 vanilla 速度，verl-SpeCo 的协同训练正是为维持高 acceptance。

### 8.6 前沿：NanoSpec 等系统-算法协同

**NanoSpec**（arXiv:2605.26444, 2026）：训练-free 的**动态上下文感知词表裁剪**。draft 的 lm_head 是 100k+ 词表的大矩阵，是 draft forward 的计算瓶颈。NanoSpec 利用语言生成的时间局部性，为每步动态构造 <3k active vocab（40× 压缩），通过异步 gather + GPU-resident state 的系统-算法协同实现硬件收益。作为 EAGLE-2/3 的**即插即用补充**，平均省 draft time 51.6%，端到端再提速 1.17–1.29×。

其他方向：
- **Self-speculation（layer-skipping / early-exit）**：target 自己当 draft，跳过中间层做粗略前向，零额外 VRAM。
- **Streaming / overlapped execution**：draft 和 verify 流水化重叠，消除两阶段锁步 bubble。
- **Lookahead / Jacobi decoding**：不动点迭代并行解多 token，无 draft cache 开销。
- **Suffix Decoding + LSTM speculator**（Snowflake Arctic / verl RFC #4791）：LSTM 超轻量 draft + suffix automaton 复用重复片段，适合代码/模板化输出。

### 8.7 联网来源（2026-07-12 核实）

- [EAGLE-3 论文 arXiv:2503.01840](https://arxiv.org/html/2503.01840v1) / [SafeAILab/EAGLE GitHub](https://github.com/SafeAILab/EAGLE)
- [EAGLE 3.1 — vLLM Blog 2026-05-26](https://vllm.ai/blog/2026-05-26-eagle-3-1)
- [verl-SpeCo pre-release — verl GitHub 2026-06](https://github.com/verl-project/verl-SpeCo)
- [verl PR #5925 EAGLE/EAGLE3 rollout 加速](https://github.com/verl-project/verl/pull/5925)
- [verl issue #5559 Decoupled Speculation](https://github.com/verl-project/verl/issues/5559) / [arXiv:2511.16193](https://arxiv.org/pdf/2511.16193)
- [verl PR #4947 Speculator 训练框架 RFC](https://github.com/verl-project/verl/pull/4947) / [verl issue #4791 Suffix+LSTM RFC](https://github.com/verl-project/verl/issues/4791)
- [NanoSpec arXiv:2605.26444](https://arxiv.org/html/2605.26444)
- [Speculative Decoding Guide 2026 — localaimaster](https://localaimaster.com/blog/speculative-decoding-guide)
- [Speculative Decoding — inference.net 2026-03](https://inference.net/content/speculative-decoding/)
- [Speculative Decoding Theory & vLLM Implementation — vizuara 2026-07](https://vizuara.substack.com/p/speculative-decoding-theory-and-implementation)
- [TensorRT-LLM Speculative Decoding docs](https://nvidia.github.io/TensorRT-LLM/features/speculative-decoding.html)
- [vLLM speculators eagle3 docs](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/eagle3.md)

---
相关: [[高级推理技术]] | [[KV cache management]] | [[beam search]] | [[parallel decoding]] | [[sampling throughput]] | [[weight sync mechanism]] | [[训推不一致]] | [[stale policy problem]]
