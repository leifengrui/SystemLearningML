# MoE

> **所属章节**: [[MoE]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中级（从零讲清"稀疏激活"如何解耦参数量与计算量）

## 1. 一句话定义

**MoE（Mixture of Experts，专家混合，混合专家）** 是把 Transformer 的 [[FFN]] 换成**多个并行的"专家"FFN + 一个路由器**的稀疏架构：每个 token 只被路由到 **top-k 个专家**计算，其余专家不激活——从而**解耦"模型总参数量"与"每 token 计算量"**，用同样的算力训练容量大得多的模型。一句话定位：**MoE = 用稀疏激活换更大容量**，总参数 $\gg$ 激活参数。
激活-》参与计算
## 2. 为什么需要它（动机与背景）

### 2.1 Dense 模型的瓶颈：参数量 = 计算量

回顾 [[FLOPs计算]]：训练计算量 $C \approx 6ND$，其中 $N$ 是**参与计算的参数量**。对 dense 模型，**每个 token 都用上全部参数**，所以 $N_{\text{总}} = N_{\text{激活}}$。

这意味着：想让模型容量更大（更多参数存更多知识），**算力必须等比上涨**。一个 70B 的 dense 模型，每 token 要算 70B 参数的量；扩到 700B，算力涨 10 倍。算力是钱的函数，这条路走不远。

> [!note] 解答：这里的"算力"指显存吗？
> 
> **不是，算力和显存是两回事**（虽然 dense 模型里都随参数涨）：
> 
> | 概念 | 是什么 | 单位 | Dense 扩参数时 |
> |---|---|---|---|
> | **算力（compute）** | 计算量 = 要做多少次浮点运算 | FLOP（如 $6ND$） | 等比涨（$C\propto N$） |
> | **显存（memory）** | 存参数/激活/梯度/优化器状态的物理内存 | bytes（如 $2N$ 存 fp16 权重） | 也等比涨（$\propto N$） |
> 
> - 这句"算力涨 10 倍"指 **FLOP 涨 10 倍** → 训练时间/成本涨 10 倍（钱）。
> - 显存也涨 10 倍（要装 10 倍参数），但那是"装得下装不下"的问题，不是"算多久"。
> - 二者常被混说，因为 dense 模型里它们都 $\propto N$，但**物理本质不同**：算力 = GPU 算多久（FLOPs/s），显存 = GPU 装多大（bytes）。见 [[FLOPs计算]]、[[参数量计算]]、[[activation memory]]。
> 
> **MoE 的关键**：解耦二者——算力按**激活参数**算（省），显存按**总参数**算（费）。所以 MoE 是"省算力、费显存"，见 [[#7. 常见误区与易错点]] 误区 1、[[FLOPs计算]] 误区 5。
### 2.2 MoE 的破局：解耦参数量与计算量

MoE 的洞察：**不是所有参数都需要对每个 token 都参与计算**。一个 token 可能只需要某些"专家知识"，没必要动用全部参数。

做法：把一个大 FFN 换成 $E$ 个小 FFN（专家），每个 token 只挑 $k$ 个（$k \ll E$，常 $k=2$）。这样：

- **总参数量** $N_{\text{总}}$：所有专家加起来，可以非常大（如 Mixtral 8×7B 总参数 ~47B）；
- **每 token 激活参数** $N_{\text{激活}}$：只算被选中的 $k$ 个专家，很小（Mixtral 每 token 激活 ~13B）；
- **计算量** $C \approx 6 N_{\text{激活}} D$：只取决于激活参数，**不随总参数等比涨**。

> [!tip] 一句话总结动机
> Dense 模型"扩参数 = 扩算力"，成本线性涨；MoE 让"扩参数 ≠ 扩算力"，同样算力下训出容量大几倍的模型。这是 LLM 继续扩规模的主流方向（GPT-4 传闻、Mixtral、DeepSeek-V3 都是 MoE）。

## 3. 核心概念详解

### 3.1 Dense vs MoE 对比

| 维度 | Dense | MoE |
|---|---|---|
| 总参数 | $N$ | $\gg N_{\text{激活}}$（所有专家之和） |
| 每 token 激活参数 | $N$ | $N_{\text{激活}} = \frac{k}{E}\cdot N_{\text{总}}$（只 top-k） |
| 计算成本（每 token） | 高（全部参数） | 低（只算 k 个专家） |
| 显存占用 | $N$ | 大（要存所有专家，即使不激活） |
| 训练难度 | 标准 | 更难（路由、均衡、稳定性） |
| 典型代表 | Llama-7B/13B/70B | Mixtral 8×7B、DeepSeek-V3 |

### 3.2 MoE 层的结构

MoE 通常**只替换 FFN 子层**，[[多头注意力]]保持 dense（attention 是 token 间交互，稀疏化代价大）：

```
                ┌──────────────────────────────────┐
                │            MoE 层                │
  token x ──→  Router(门控网络) ──→ 选 top-k 专家   │
                │              ↓                   │
                │   E1   E2   E3  ...  EE          │
                │  (每个专家是一个独立 FFN)          │
                │   ↓    ↓    ↓          ↓         │
                │   (只激活选中的 k 个)              │
                │              ↓                   │
                │      按路由权重加权求和 → 输出      │
                └──────────────────────────────────┘
```

- **专家（Expert）**：每个专家就是一个独立的 FFN（标准或 [[SwiGLU]]），参数不共享；
- **路由器（Router / Gate）**：一个小线性层 $W_g \in \mathbb{R}^{d \times E}$，对每个 token 算 $E$ 个专家的分数，选 top-k。

> [!note] 解答：路由器怎么选出 top-k 个专家
> 
> 选 top-k 是一个**"打分 → 取最大 k 个 → 归一化作权重 → 加权求和"**的四步过程，全程对每个 token 独立做：
> 
> **第 1 步：打分（router / gate）**
> 每个 token 的 hidden state $x\in\mathbb{R}^d$ 经一个小线性层 $W_g\in\mathbb{R}^{d\times E}$，得到 $E$ 个专家各自的"得分"logits：
> 
> $$\text{logits} = x\,W_g \in \mathbb{R}^{E}$$
> 
> 普通 $d\to E$ 矩阵乘，每个专家一个标量分数。$W_g$ **可学习**（路由器本身也是模型参数，靠梯度训练学会"哪种 token 该找哪个专家"）。
> 
> **第 2 步：取 top-k（离散选择，不可导）**
> 从 $E$ 个分数里挑**分数最大的 $k$ 个**专家索引 $\{i_1,\dots,i_k\}$。例 $E=8,k=2$：8 个分数取最大两个下标。未被选中的 $E-k$ 个专家**不参与前向**（省算力）、**也不参与反向**（梯度为 0，§3.5 路由不可导的根源）。
> 可选变体：**Noisy top-k gating**（Shazeer 2017 原版）打分时加 Gumbel 噪声 $\text{logits} = xW_g + \epsilon\cdot\text{Gumbel}(0,1)$，逼路由器探索、防一开始就锁死少数专家（防专家坍缩）。
> 
> **第 3 步：归一化作加权权重**
> 对选中的 $k$ 个分数做 **softmax**（只在 k 个里归一，非选中置 0）：
> 
> $$g_i = \frac{\exp(\text{logits}_i)}{\sum_{j\in\text{top-k}}\exp(\text{logits}_j)},\ i\in\text{top-k};\quad g_i=0\text{ 否则}$$
> 
> **只对 top-k 内做 softmax**（不对全部 $E$ 个做），未选中专家权重严格为 0、不会偷偷贡献计算。Mixtral/DeepSeek 都用这套。
> 
> **第 4 步：分发与加权求和**
> 把 token $x$ 送入选中的 $k$ 个专家各算一遍，按 $g_i$ 加权求和：
> 
> $$y = \sum_{i\in\text{top-k}} g_i \cdot \text{Expert}_i(x)$$
> 
> Mixtral 8×7B（$E=8,k=2$）：每 token 算 2 个专家按 softmax 权重相加；DeepSeek-V3（$E=256,k=8$）：256 个选 8 个，专家更细。
> 
> **梯度怎么流**：$W_g$ 通过 softmax 后的 $g_i$ 可导，梯度能流回路由器；但"选哪 k 个"这个**离散 argmax 选择本身不可导**，未选中专家拿不到梯度——这正是 MoE 训练难的根源（§3.5）。负载均衡损失 / DeepSeek 偏置均衡就是从外部打破这个不可导死结。详见 [[MoE路由]]、[[负载均衡损失]]。
- **典型配置**：$E=8/64/256$ 个专家，每 token 选 $k=1/2$ 个。

### 3.3 现代 MoE 模型实例

| 模型 | 专家数 $E$ | top-k | 总参数 | 激活参数 | 特点 |
|---|---|---|---|---|---|
| Mixtral 8×7B | 8 | 2 | ~47B | ~13B | 开源 MoE 标杆 |
| DeepSeek-V3 | 256（细粒度） | 8 | ~671B | ~37B | 细粒度 + 共享专家 + 无辅助损失 |
| Qwen-MoE | 细粒度 | - | - | - | 细粒度专家设计 |
| GPT-4（传闻） | - | - | - | - | 据传是 MoE |

### 3.4 现代趋势

1. **细粒度专家（fine-grained experts）**：用更多更小的专家（如 $E=256$ 而非 8），提高组合灵活性。同样激活参数下，组合数 $\binom{E}{k}$ 更大，表达力更强。DeepSeek-V3 走这条路。
2. **共享专家（shared experts）**：一部分专家**总是激活**（不参与路由），捕捉通用知识；其余专家稀疏激活，捕捉专业化知识。这样通用知识不用每个专家重复存，省参数。
3. **无辅助损失均衡（auxiliary-loss-free balancing）**：DeepSeek 提出，用可学习偏置项动态调整路由概率，而非加辅助损失（辅助损失会干扰主目标）。详见 [[负载均衡损失]]。

### 3.5 训练挑战速览

MoE 比 dense 更难训，主要痛点（详见 [[MoE路由]]、[[负载均衡损失]]）：

1. **专家坍缩**：路由器倾向总选少数专家，其余饿死 → 需 [[负载均衡损失]]；
2. **路由不可导**：top-k 是离散选择，梯度无法流过未选中专家；
> [!note] 解答：路由不可导"那又怎样"？
> 
> 后果很严重，正是 MoE 一系列难题的根源：
> 
> 1. **未选中的专家得不到梯度 → 永远不更新**：top-k 是离散选择，未被选中的专家不参与前向，反向时梯度为 0，权重不更新。久而久之这些专家"冻死"，表达力永远停留在初始水平，参数浪费。
> 2. **路由器也感受不到未选专家 → 无法自我纠正**：路由器的梯度只来自被选专家的路径，它"看不见"还有别的专家可用，无法学会"试试别的"。一旦初始偶然偏某几个专家，正反馈把它锁死（见 [[负载均衡损失]] §2.1 专家坍缩恶性循环）。
> 3. **结果：专家坍缩**：$E$ 个专家里只有少数几个在工作，MoE 退化成小模型，总参数浪费、容量没兑现——这是 MoE 训练的头号失败模式。
> 4. **必须外部干预**：正因为路由不可导、自我纠正不了，才需要 [[负载均衡损失]]（辅助损失）或 DeepSeek 偏置均衡从外部打破正反馈，逼路由器把流量分匀。
> 
> **一句话**：路由不可导 → 未选专家无梯度 → 专家坍缩 → 必须靠负载均衡损失/偏置兜底。这是 MoE 训练比 dense 难的核心原因之一。
3. **token dropping**：容量因子限制下部分 token 被丢弃，影响质量；
4. **训练不稳定**：MoE 比 dense 更易发散，需小心调超参；
> [!note] 解答：什么是超参（hyperparameter）？
> 
> **超参数（hyperparameter，超参）** 是训练前**人为设定**、不通过梯度学习的配置量；对比**参数（parameter）** 是模型自己通过梯度学习更新的权重。
> 
> | | 参数 | 超参 |
> |---|---|---|
> | 谁定 | 模型自己学（梯度下降） | 人设定 |
> | 例子 | $W, b$（权重/偏置） | 学习率、batch size、层数 $L$、宽度 $d$ |
> | 怎么调 | 反向传播 | 手动 / 网格搜索 / 贝叶斯优化 |
> | 训练中变 | 每个 step 更新 | 固定（或按 schedule） |
> 
> **MoE 里的常见超参**：
> - 结构：专家数 $E$、top-k $k$、专家宽度 $d_{\text{ff}}$、层数 $L$；
> - 均衡：辅助损失权重 $\alpha$、容量因子 $c$、噪声方差 $\sigma$；
> - 优化：学习率、warmup 步数、weight decay；
> - 训练：batch size、序列长度、dropout。
> 
> **"小心调超参"**指 MoE 比 dense 更敏感（路由不稳、易发散），这些超参的取值对训练成败影响大，要反复实验找甜点（如 $\alpha$ 太大伤质量、太小坍缩；$c$ 太小丢 token、太大不均衡）。见 [[负载均衡损失]] §7 误区 1、5。
> 
> 相关：[[学习率调度]]、[[batch与mini-batch]]、[[参数量计算]]（参数 vs 超参）。
5. **微调困难**：MoE 在下游微调时容易过拟合（每个专家见的数据少）；
6. **显存大**：虽然计算省，但要存所有专家参数，显存反而是瓶颈。

## 4. 数学原理 / 公式

### 4.1 参数量与激活参数

设单个专家 FFN 参数 $N_{\text{expert}}$，共 $E$ 个专家，每 token 选 $k$ 个：

$$
N_{\text{总}} = E \cdot N_{\text{expert}} + N_{\text{router}} + N_{\text{attn}}\quad(\text{含注意力等 dense 部分})
$$

$$
N_{\text{激活}} = k \cdot N_{\text{expert}} + N_{\text{router}} + N_{\text{attn}}
$$

- 注意力层保持 dense，每个 token 都算，所以 $N_{\text{attn}}$ 在两者里都全额计入；
- 路由器 $N_{\text{router}} = d \cdot E$（小，通常忽略）；
- 稀疏只发生在 FFN 专家部分：$\frac{N_{\text{激活}}}{N_{\text{总}}}\big|_{\text{FFN}} = \frac{k}{E}$。
> [!note] 解答：MoE 典型结构是"一层 dense + 一层专家"交替吗？
> 
> **不完全是。主流有三种布局，"dense + MoE 交替"只是其一，且不是 Mixtral 的做法。** 以下数字均经联网核实（Mixtral 论文 arXiv:2401.04088 / DeepSeek-V3 技术报告 arXiv:2412.19437 / Switch Transformer arXiv:2101.03961）。
> 
> **布局 A：全 MoE（每层 FFN 都换成 MoE）**
> - 每层都是 attention + MoE-FFN，无 dense FFN。
> - 代表：**Mixtral 8×7B**（32 层全是 MoE）。Mixtral 论文明确写 "we replace **all** FFN sub-blocks by MoE layers while GShard replaces every other block"——直接把 Mistral 7B 每层 FFN 换成 8 专家 MoE。
> - 优点：每层都享稀疏激活的容量红利；结构最简单、实现统一。
> - 缺点：每层都路由 + All-to-All 通信，工程与通信开销最大；浅层用 MoE 性价比低。
> 
> **布局 B：dense + MoE 交替（隔层 / 每 N 层一个 MoE）**
> - 一部分层保留 dense FFN，一部分换 MoE，典型每隔 1 层插一个 MoE。
> - 代表：**GShard**（"replaces **every other** block"，隔层一个 MoE）、**Switch Transformer**（"experts at **every other** feed-forward layer"，隔层一个 Switch MoE，且用 top-1 路由）。
> - 优点：路由通信开销减半甚至更少；dense 层提供稳定信息加工路径，训练更稳；浅层 dense 学通用特征、深层 MoE 学专业知识。
> - 缺点：容量红利不如全 MoE。
> 
> **布局 C：前几层 dense + 其余 MoE + shared expert（现代细粒度 MoE 主流）**
> - **浅层（前 N 层）用 dense FFN**，**深层用 MoE**，且常配 **shared expert（共享专家，总是激活）** 捕捉通用知识。
> - 代表：**DeepSeek-V3**——技术报告原话 "We substitute all FFNs **except for the first three layers** with MoE layers. Each MoE layer consists of **1 shared expert** and **256 routed experts**"。即 61 层中前 3 层 dense（layers 0-2）、后 58 层 MoE（layers 3-60），每 MoE 层 1 共享 + 256 路由专家、每 token 激活 8 个路由专家。Qwen-MoE 等细粒度 MoE 同理。
> - 优点：浅层 dense 稳定提取底层语言特征（语法、词法），不需专家化；深层 MoE + shared expert 兼顾专业化与通用性，是当前细粒度 MoE 的工程最优解。
> 
> **为什么不全用 MoE？三个理由：**
> 1. **浅层不需要专家化**：浅层学通用底层特征（token embedding、语法），所有 token 都要相似加工，专家化收益小、反增路由不稳。
> 2. **dense 层是稳定锚点**：MoE 路由有不可导问题（§3.5），全 MoE 训练更易发散；穿插 dense 层提供稳定密集计算路径，梯度流通更顺。
> 3. **通信开销**：每层 MoE 都要 All-to-All 把 token 发到对应专家所在卡（见 [[专家并行]]），层数越多通信越重；隔层或浅层 dense 可减半通信。
> 
> **一句话**：Mixtral 是"全 MoE"特例（简单直接，来自 dense 模型直接改造），DeepSeek-V3 是"前 dense + 后 MoE + shared expert"现代主流（工程更优）。你说的"一层 dense + 专家"接近布局 B/C，但更准确是"**部分层 dense + 部分层 MoE**"或"**前几层 dense + 后面 MoE**"，不是严格的"1 dense : 1 MoE"交替（GShard/Switch 才是隔层交替）。详见 §3.2 结构图、§8.1 历史脉络。
### 4.2 计算量

训练计算量公式 $C \approx 6 N_{\text{激活}} D$ 仍然成立，但 **$N$ 是激活参数**（见 [[FLOPs计算]]）：

$$
C_{\text{MoE}} \approx 6\, N_{\text{激活}}\, D
$$

对比同总参数的 dense 模型 $C_{\text{dense}} \approx 6\, N_{\text{总}}\, D$，MoE 省算力 $\frac{N_{\text{激活}}}{N_{\text{总}}} = \frac{k}{E}$（FFN 部分）。Mixtral 8×7B：$k=2, E=8$，FFN 部分算力省到 1/4。

### 4.3 一个具体例子（Mixtral 8×7B 量级）

设 $d=4096$，$d_{\text{ff}}=14336$（SwiGLU，$\frac{8}{3}d$），$E=8$，$k=2$，$L=32$ 层：

- 单个 SwiGLU 专家参数 $\approx 3 \cdot d \cdot d_{\text{ff}} = 3 \times 4096 \times 14336 \approx 176\text{M}$；
- 8 个专家 FFN 总参数 $\approx 8 \times 176\text{M} \approx 1.4\text{B}$/层；
- 激活 FFN 参数（2 个专家）$\approx 2 \times 176\text{M} \approx 352\text{M}$/层；
- 32 层 FFN 总参数 $\approx 45\text{B}$，激活 $\approx 11\text{B}$（加注意力等约 47B 总 / 13B 激活，与公开数据吻合）。
## 6. 与其他知识点的关系

- **上游（依赖）**: [[FFN]]（MoE 是 FFN 的稀疏化，把一个大 FFN 换成多个小 FFN）、[[decoder-only architecture]]（MoE 层替换其中的 FFN 子层）、[[SwiGLU]]（现代 MoE 专家多用 SwiGLU FFN）、[[FLOPs计算]]（$C \approx 6ND$ 中 $N$ 是激活参数）、[[参数量计算]]。
- **下游（应用）**: [[MoE路由]]（路由器是 MoE 的关键组件）、[[负载均衡损失]]（防专家坍缩）、[[专家并行]]（MoE 的分布式训练）、LLM 扩规模（Mixtral/DeepSeek-V3 等大模型标配）。
- **对比 / 易混**:
  - **MoE vs Dense**：Dense 每 token 用全部参数（参数=计算），MoE 每 token 只用 top-k 专家（总参数≫激活参数）。MoE 省算力、费显存、难训练。
  - **MoE vs Dropout**：Dropout 是训练时随机丢神经元（推理全用），MoE 是路由选专家（训练推理都稀疏）。机制不同，别混。
  - **MoE vs Ensemble**：Ensemble 是多模型平均（都算），MoE 是单模型内路由（只算部分）。MoE 是"模型内集成"。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 MoE 省显存**：错。MoE **省算力**（每 token 只算 k 个专家），但**费显存**（要存所有 E 个专家参数）。MoE 适合"算力受限但显存充足"的场景。推理时显存往往是 MoE 的瓶颈。
> 2. **以为所有专家都得梯度**：错。top-k 是离散选择，**未被选中的专家得不到梯度**（详见 [[MoE路由]] §4.2），这正是专家坍缩的根源。
> 3. **以为 $C \approx 6ND$ 的 $N$ 是总参数**：错。MoE 下 $N$ 是**激活参数**（只算被选中的 k 个专家）。用总参数算 FLOPs 会严重高估。
> 4. **以为 MoE 一定比 dense 好**：不一定。MoE 训练更难、微调易过拟合、显存大。小规模或微调密集场景 dense 可能更优。MoE 的优势主要在"大算力预算 + 大规模预训练"。
> 5. **以为注意力层也稀疏化**：主流 MoE 只稀疏化 FFN，attention 保持 dense。attention 稀疏化（如 sparse attention）是另一条线，别混。
> 6. **以为 k 越大越好**：k 大→激活参数多→算力涨、质量边际提升递减。$k=2$ 是经验甜点（Mixtral）。
> 7. **忽略负载均衡**：不加 [[负载均衡损失]]，路由器必然专家坍缩，大部分专家饿死，MoE 退化成小模型。

## 8. 延伸细节

### 8.1 MoE 的历史脉络

- **Shazeer et al. 2017（Outrageously Large NNs）**：稀疏门控 MoE 奠基，在 LSTM 上用；
- **Switch Transformer（Fedus 2021）**：简化为 top-1 路由，首次大规模 MoE 预训练；
- **GShard（Lepikhin 2020）**：MoE 分布式训练工程，专家并行；
- **Mixtral（Jiang 2024）**：开源 8×7B，MoE 进入开源主流；
- **DeepSeek-V3（2024）**：细粒度 + 共享专家 + 无辅助损失均衡，MoE 工程新标杆。

### 8.2 MoE 的"性价比"从哪来

MoE 用同样的**训练算力**（FLOPs）训出更大容量的模型，但这个"赚"只在**训练阶段**。推理时：

- 每个 token 仍只算 k 个专家（省算力）；
- 但要**把所有专家参数加载到显存**（费显存）；
- 且路由是**数据相关**的（每个 token 路由不同），导致 batch 内 token 要分发到不同专家，工程复杂、[[专家并行]]通信开销大。

所以 MoE 的推理吞吐优化（如专家并行、expert grouping）是单独的研究方向。

### 8.3 MoE 与 scaling laws

[[缩放定律]]（Kaplan/Chinchilla）原是针对 dense 模型的。MoE 的 scaling law 略有不同：

- 最优 expert 数随算力增长；
- 最优 $k$ 相对稳定（2 左右）；
- 同算力下，MoE 的 loss 略低于 dense（容量更大），但差距随规模收窄。

### 8.4 MoE 微调的特殊性

MoE 微调比 dense 易过拟合，原因：

- 每个专家只被部分 token 激活，见的数据少；
- 路由器可能把微调数据全路由到少数专家，加剧不均；
- 常需更低 lr、更强正则、或冻结部分专家。

---
相关: [[FFN]]、[[MoE路由]]、[[负载均衡损失]]、[[专家并行]]、[[decoder-only architecture]]、[[FLOPs计算]]、[[参数量计算]]、[[SwiGLU]]
