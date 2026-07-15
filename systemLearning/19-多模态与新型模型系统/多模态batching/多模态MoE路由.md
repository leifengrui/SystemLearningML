# 多模态MoE路由

> **所属章节**: [[多模态batching]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: 多模态 MoE 路由 / 模态专家 / modality-specific experts / 跨模态路由 / cross-modal routing / 多模态 Mixture-of-Experts / modality-aware routing / 视觉专家 / visual expert / MoE for VLM
> **难度**: 中高（需懂 [[MoE]]、[[MoE路由]]、[[负载均衡损失]]、[[ViT与vision encoder]]、[[projector]]、[[early与late fusion]]、[[FFN]]、[[自注意力]]、[[多模态sequence packing]]）


## 1. 一句话定义

**多模态 MoE 路由（modality-specific experts / 跨模态路由 / 模态专家）** 是把 [[MoE]] 用在多模态大模型（VLM / MLLM）上的路由范式——把 FFN 的专家**按模态分工**：**视觉专家**专吃视觉 token、**文本专家**专吃文本 token、（多模态扩展）**音频专家**专吃音频 token；路由器（router）不再只看 token 语义，还看 token **所属模态**（视觉 token 倾向走视觉专家、文本 token 走文本专家），也可做**跨模态路由**（让视觉 token 故意路由到文本专家，借文本专家的语言知识做跨模态对齐 / grounding）；其动机是**不同模态的算力需求与特征分布差异巨大**（视觉 token 稠密空间相关、文本 token 离散语义、音频 token 时序相关），单个 dense FFN 难兼顾，**专家分工**让模型"用对算力、各司其职"，提升容量同时控激活算力；核心风险是**模态专家坍缩**（某模态 token 全挤进一个专家、其余饿死），故多模态 MoE 在 [[负载均衡损失]] 基础上还要加**模态级负载均衡**（per-modality load balancing），逼路由器在每模态内均衡分配。代表：LIMoE（Google，模态专家**自发涌现**）、CogVLM（显式视觉专家 + 文本专家并联门控）、Uni-MoE（显式训练模态专家）、MoE-LLaVA（标准 MoE 路由用于 VLM）。

> [!note] 三句话定位
> - **是什么**：MoE 的专家按模态分工（视觉专家 / 文本专家 / 音频专家），router 按 token 模态 + 语义选专家；可跨模态路由（视觉 token 走文本专家做对齐）。比 dense FFN 用对算力、提容量。
> - **为什么**：视觉 / 文本 / 音频 token 的算力需求与特征分布差异大，dense FFN 一套权重难兼顾；MoE 专家分工让模型"各司其职"，稀疏激活控算力。LIMoE 实证模态专家会**自发涌现**（MoE 天然适合多模态）。
> - **与 [[MoE]]/[[MoE路由]]/[[负载均衡损失]] 关系**：路由机制、top-k、不可导处理完全复用；本条加"模态"维度——router 输入含模态信号、专家按模态初始化 / 分工、负载均衡要**模态级**（防某模态全挤一专家）。

> [!note] 别名辨析
> - **modality-specific experts（模态专家）**：强调"专家按模态分工"，是设计目标。
> - **cross-modal routing（跨模态路由）**：强调"视觉 token 可路由到文本专家"的跨模态对齐行为，是路由策略。
> - **visual expert（视觉专家）**：CogVLM 的叫法，指 LLM 每层并联的"处理视觉 token 的 FFN 分支"。
> 三者都属"多模态 MoE 路由"，侧重不同（分工 vs 跨模态 vs 视觉分支）。本条统一讲。


## 2. 为什么需要它（动机与背景）

### 2.1 不同模态算力 / 特征分布差异巨大

[[MoE]] 在纯 LLM 的动机是"稀疏激活提容量"（参数多但每 token 只用 $k$ 个专家，控算力）。多模态 MoE 多一层动机：**不同模态的 token 特征分布与算力需求根本不同**，dense FFN 一套权重难兼顾。

| 模态 | token 特征 | 算力需求 | 稠密度 |
|---|---|---|---|
| 视觉 token（[[ViT与vision encoder]] 输出） | 椭空间稠密、空间相关（相邻 patch 相关）、连续 | 高（patch 间交互复杂） | 稠密、相关 |
| 文本 token | 离散语义、强上下文、稀疏语义场 | 中（语言知识检索） | 稀疏、离散 |
| 音频 token（Whisper encoder） | 时序相关、频谱特征 | 中高 | 时序相关 |

dense FFN 用一套权重同时处理这三种 token：要么视觉专家化不足（被文本分布主导）、要么文本能力弱（被视觉稠密特征带偏）。**MoE 让专家分工**——视觉专家学视觉相关的 FFN 变换、文本专家学语言变换，各模态用对算力。

### 2.2 dense FFN 的"模态冲突"

设 dense FFN 权重 $W$，输入是视觉 token $x_v$ 与文本 token $x_t$ 混合。$W$ 要同时满足：

$$
\text{视觉}: \quad x_v W_2 \cdot \sigma(x_v W_1) \approx f_v^*(x_v) \quad \text{(视觉变换目标)}
$$
$$
\text{文本}: \quad x_t W_2 \cdot \sigma(x_t W_1) \approx f_t^*(x_t) \quad \text{(语言变换目标)}
$$

$f_v^*$ 与 $f_t^*$ 是两个**不同函数**（视觉变换 ≠ 语言变换）。一套 $W_1, W_2$ 是两者的**折中拟合**，不是任何一者的最优。这就是 dense FFN 的"模态冲突"——视觉与文本在 FFN 权重空间竞争，互相掣肘。

> [!note] MoE 的天然解：专家分工
> MoE 把 $W$ 拆成 $E$ 个专家 $W^{(1)}, \ldots, W^{(E)}$，router 让视觉 token 走 $W^{(v)}$（视觉专家）、文本 token 走 $W^{(t)}$（文本专家）。各专家学各模态的最优变换，**消解模态冲突**。LIMoE（Mustafa et al., arXiv:2206.02770, 2022）实证：**模态专家会自发涌现**——即使 router 不显式喂模态信号，训练后某些专家天然专攻视觉、某些专攻文本。这是 MoE 天然适合多模态的根本证据。

### 2.3 容量提升同时控激活算力

多模态模型要处理多模态需更大容量（视觉细节多、语言知识广）。dense scaling 是全参数全激活，贵。MoE 稀疏激活：总参数 $E \cdot d_{\text{expert}}$ 大（容量大），但每 token 只激活 $k$ 个专家（算力 $k \cdot d_{\text{expert}}$，恒定）。MoE-LLaVA（Lin et al., arXiv:2401.15947, 2024）用 ~3B 激活参数（总参数更大）达到 LLaVA-1.5-7B 性能，是稀疏激活控算力的代表。

### 2.4 跨模态路由：视觉 token 借文本专家做对齐

更深的设计：**不让视觉 token 只走视觉专家**。视觉 token 路由到文本专家，借文本专家的**语言知识**做 grounding（视觉特征与语言概念对齐）。这是跨模态路由——模态专家不必严格按模态分，router 可让 token 跨模态路由，专家学跨模态知识。这正是 LIMoE "modality-specific experts emerge" 的另一面：部分专家是**跨模态混合专家**（既处理视觉也处理文本），承担对齐职责。

### 2.5 模态专家坍缩：多模态 MoE 的头号风险

[[负载均衡损失]] 治的是单模态 MoE 的"专家坍缩"（router 总选少数专家）。多模态 MoE 多一层"**模态专家坍缩**"：

- 某模态（如视觉）的 token **全挤进一个专家**（如 expert #3），其余视觉专家饿死；
- 或某专家**只被一个模态选**（如 expert #3 全是视觉 token，文本 token 从不走它），专家模态化过强，失去跨模态能力；
- 或 router 对某模态 token 输出**退化路由**（所有视觉 token 同样分数 → 同走 top-1），模态内无分工。

故多模态 MoE 在 [[负载均衡损失]] 的"全局均衡"基础上，要加**模态级均衡**——每模态内各专家被选频次均匀，防模态专家坍缩。

### 2.6 与 early fusion 的协同

[[early与late fusion]] 的 early fusion 把视觉 token 拼进 LLM 输入序列，LLM 的 FFN 要同时处理视觉 + 文本 token。若 FFN 是 dense，模态冲突（§2.2）；若 FFN 是 MoE，专家分工消解冲突。**多模态 MoE 是 early fusion 的"FFN 升级版"**——保留 early fusion 的简单（拼接 + projector），用 MoE 治 FFN 的模态冲突。CogVLM 走这条路（early fusion + 每层视觉专家 / 文本专家并联）。


## 3. 核心概念详解

### 3.1 模态专家 vs 通用专家

| 维度 | 通用专家（单模态 MoE，[[MoE路由]]） | 模态专家（多模态 MoE，本条） |
|---|---|---|
| 专家分工依据 | token 语义（"哪个 token 该走哪专家"） | token **模态 + 语义**（视觉 / 文本 / 音频 + 内容） |
| router 输入 | token 隐状态 $x$ | token 隐状态 $x$ + 模态信号（可选显式） |
| 专家类型 | 无模态语义（专家分工自发涌现） | 显式或隐式按模态分工（视觉专家 / 文本专家） |
| 坍缩风险 | 全局专家坍缩（少数专家被多选） | **模态专家坍缩**（某模态全挤一专家） |
| 负载均衡 | 全局 [[负载均衡损失]] | 全局 + **模态级**负载均衡 |
| 跨模态能力 | 无（单模态） | 跨模态路由（视觉 token 走文本专家做对齐） |
| 代表 | Mixtral / DeepSeek-MoE / GShard | LIMoE / CogVLM / Uni-MoE / MoE-LLaVA |

### 3.2 模态专家的三种实现路径

| 路径 | 描述 | 模态信号 | 代表 |
|---|---|---|---|
| **自发涌现** | router 不喂模态信号，训练后专家自发专攻某模态 | 无（隐式） | LIMoE |
| **显式分工训练** | 用各模态数据**单独预训练**对应专家（"激活专家偏好"），再合并 router | 训练阶段隐式 | Uni-MoE |
| **显式并联分支** | LLM 每层并联视觉专家 FFN + 文本专家 FFN，门控按 token 路由 | 显式（架构层） | CogVLM |

LIMoE 论文（arXiv:2206.02770）关键发现："the organic emergence of modality-specific experts"——**模态专家自发涌现**。即使 router 设计上不区分模态，训练后某些专家天然主要处理视觉 token、某些主要处理文本。这证明 MoE 架构**天然适合多模态**——专家分工是多模态数据的自然产物。

### 3.3 路由器：按模态 + 语义选专家

多模态 MoE 路由器在 [[MoE路由]] 基础上加模态维度。设 token $x$，模态标签 $m \in \{\text{vis}, \text{txt}, \text{aud}\}$（可选显式输入或从 token 隐状态推断）。router：

$$
\text{logits} = x W_g + b(m) \quad \text{(可选加模态偏置 } b(m)\text{)}
$$

top-k 选 $k$ 个专家，softmax 加权（与 [[MoE路由]] §3.1 完全同构）。模态偏置 $b(m)$ 是可选的"模态先验"——视觉 token 给视觉专家加正偏置、文本 token 给文本专家加正偏置，引导模态分工。也可不加（LIMoE 涌涌现）。

### 3.4 跨模态路由（视觉 token 走文本专家）

关键设计：**不强制视觉 token 只走视觉专家**。router 可让视觉 token 路由到文本专家：

- **动机**：视觉 token 经文本专家变换，借文本专家的**语言知识**做 grounding——视觉特征与语言概念对齐。如视觉 token "猫的图"经文本专家变换后更接近"cat"的语言表示，便于 LLM 后续生成"cat"。
- **实现**：不加模态硬约束（不强制 $b(m)$ 把视觉 token 钉死视觉专家），让 router 自由学习跨模态路由。
- **风险**：跨模态路由过度则视觉专家饿死（视觉 token 都跑去文本专家），需模态级负载均衡约束。

### 3.5 模态专家坍缩的形式化

设 $E$ 个专家，模态集 $\mathcal{M}$，token 总数 $T$，模态 $m$ 的 token 数 $T_m$。定义：

- **专家 $e$ 被选频次**：$f_e = \frac{1}{T}\sum_{t=1}^{T} \mathbb{1}[e \in \text{topk}(x_t)]$
- **专家 $e$ 在模态 $m$ 的被选频次**：$f_{e,m} = \frac{1}{T_m}\sum_{t \in m} \mathbb{1}[e \in \text{topk}(x_t)]$
- **专家 $e$ 的模态分布**：$p(e \mid m) = f_{e,m}$

**模态专家坍缩**的三种表现：

1. **模态内单专家坍缩**：某模态 $m$ 内某专家 $e^*$ 被选频次 $f_{e^*,m} \to 1$，其余 $\to 0$——该模态全挤一专家；
2. **专家模态钉死**：某专家 $e$ 只被模态 $m$ 选（$p(e\mid m) \to 1, p(e\mid m') \to 0 \,\forall m'\ne m$），失去跨模态能力；
3. **模态路由退化**：router 对某模态 token 输出几乎相同分数（所有该模态 token 走同一 top-k），模态内无分工。

### 3.6 模态级负载均衡

[[负载均衡损失]] 的经典形式 $\mathcal{L}_{\text{aux}} = \alpha E \sum_e f_e P_e$（$P_e$ = router 给专家 $e$ 的平均概率）治全局均衡。多模态版加**模态级**项：

$$
\boxed{\mathcal{L}_{\text{aux}}^{\text{MM}} = \alpha \cdot E \sum_e f_e P_e + \beta \sum_{m \in \mathcal{M}} \sum_e f_{e,m} P_{e,m}}
$$

其中 $P_{e,m} = \frac{1}{T_m}\sum_{t \in m} \text{softmax}(\text{logits}_t)_e$（模态 $m$ 内 router 给专家 $e$ 的平均概率）。第一项全局均衡，第二项模态内均衡。$\beta$ 控模态级强度。这逼 router 在每模态内把 token 均匀分到各专家，防模态专家坍缩。

> [!note] LIMoE 的 entropy-based 正则
> LIMoE（arXiv:2206.02770）提出**基于熵的正则**替代经典 load balancing loss——最大化 router 输出分布的熵 $H = -\sum_e P_e \log P_e$（熵大 = 分布均匀 = 均衡），等价 push router 不坍缩。多模态版可对每模态内 router 分布加熵正则：$\mathcal{L}_{\text{ent}} = -\sum_m \sum_e P_{e,m} \log P_{e,m}$。与 [[负载均衡损失]] 的 $f_e P_e$ 形式不同但目的一致（防坍缩）。

### 3.7 CogVLM 的视觉专家：每层并联分支

CogVLM（Wang et al., arXiv:2311.03079, 2023）走**显式并联分支**路线——LLM 每层的 attention 与 FFN 都加"视觉专家"分支：

```
x -> [文本 attention / FFN]  ->  x_text  (冻结或主路径)
  \-> [视觉 attention / FFN] ->  x_vis   (可训练视觉专家)
  门控 g(x) 加权合并:  out = (1-g) x_text + g x_vis
```

视觉专家是**可训练的并联 FFN / attention 分支**，门控 $g(x)$ 决定多少视觉信息走视觉专家分支。与文本专家（冻结的 LLM 原分支）并联。这是"模态专家的架构层显式实现"——视觉与文本在 LLM 内部分支处理，是 [[early与late fusion]] 的 cross-attention fusion 的 MoE 变体（见 [[early与late fusion]] §8.2）。CogVLM-17B SOTA 10 个跨模态 benchmark。CogVLM2（arXiv:2408.16500）继承视觉专家架构 + 1344×1344 高分辨率。

### 3.8 Uni-MoE：显式训练模态专家

Uni-MoE（Li et al., arXiv:2405.11273, 2024）走**显式分工训练**路线——三阶段：

1. **Cross-modality alignment**：各模态 encoder + connector 对齐到统一表示；
2. **Training modality-specific experts**：用各模态的 cross-modality instruction 数据**单独训练对应专家**（"activate experts' preferences"——让每专家先专长一模态）；
3. **Tuning Uni-MoE**：LoRA 微调整体，router 学习跨模态协作。

关键是阶段 2——**先用单模态数据让每专家专长化**，再合并 router 让专家"记住"模态偏好，避免训练初期 router 随机分配导致专家没分工。是"显式分工训练"路径的代表。

### 3.9 MoE-LLaVA：标准 MoE 路由用于 VLM

MoE-LLaVA（Lin et al., arXiv:2401.15947, 2024, PKU）走**标准 MoE 路由**路线——LLM 的 FFN 换成标准 top-k MoE（router 按语义选专家，不显式按模态）。提出 **MoE-Tuning**：解决多模态稀疏学习的性能退化（直接把 MoE 套 VLM 会掉点，需特定训练策略）。~3B 激活参数 ≈ LLaVA-1.5-7B，超 LLaVA-1.5-13B 在 object hallucination。是"不显式模态分工，靠 MoE 涌现"的简化路径（介于 LIMoE 涌现与 Uni-MoE 显式之间）。


## 4. 数学原理 / 公式

### 4.1 单 token 路由概率（多模态版）

token $x$ 模态 $m$，router 权重 $W_g \in \mathbb{R}^{d \times E}$，可选模态偏置 $b(m) \in \mathbb{R}^E$：

$$
\text{logits} = x W_g + b(m) \in \mathbb{R}^E
$$

选 top-k：$\mathcal{S}_k = \text{TopK}(\text{logits}, k)$，softmax 权重：

$$
g_i = \frac{\exp(\text{logits}_i)}{\sum_{j \in \mathcal{S}_k} \exp(\text{logits}_j)}, \quad i \in \mathcal{S}_k
$$

输出（与 [[MoE路由]] 同）：

$$
\text{MoE}(x) = \sum_{i \in \mathcal{S}_k} g_i \cdot \text{FFN}_i(x)
$$

与单模态 MoE 唯一区别：$b(m)$ 给模态先验（可选），且专家初始化 / 训练可能按模态分工（显式路径）。数学形式 router 不变，变化在**训练策略与负载均衡**。

### 4.2 模态专家坍缩的概率刻画

设 router 对模态 $m$ 的 token 输出专家选择分布 $P_m = (P_{1,m}, \ldots, P_{E,m})$（router 给专家的平均概率）。理想均衡：$P_{e,m} = 1/E$（均匀）。

**模态专家坍缩**：某 $e^*$ 使 $P_{e^*,m} \to 1$，其余 $\to 0$。即 $P_m$ 退化为 one-hot。坍缩程度用熵衡量：

$$
H(P_m) = -\sum_e P_{e,m} \log P_{e,m}
$$

$H = \log E$（最大熵，均匀）→ 无坍缩；$H = 0$（one-hot）→ 完全坍缩。模态级负载均衡的目标是**最大化每模态 $H(P_m)$**（或等价最小 $E\sum_e f_{e,m} P_{e,m}$）。

### 4.3 模态级负载均衡损失推导

经典 [[负载均衡损失]]（[[负载均衡损失]] §）的动机：理想负载下每专家被选频次 $f_e = 1/E$、router 平均概率 $P_e = 1/E$。辅助损失 $\alpha E \sum_e f_e P_e$ 在均衡时取最小值 $\alpha$（因 $f_e=P_e=1/E \Rightarrow E\sum_e (1/E)(1/E) = 1$）。

多模态版推导：对每模态 $m$，理想 $f_{e,m} = P_{e,m} = 1/E$。模态级损失：

$$
\mathcal{L}_{\text{bal},m} = E \sum_e f_{e,m} P_{e,m}
$$

均衡时 $\mathcal{L}_{\text{bal},m} = 1$（最小）。对所有模态求和：

$$
\boxed{\mathcal{L}_{\text{aux}}^{\text{MM}} = \alpha \underbrace{E \sum_e f_e P_e}_{\text{全局均衡}} + \beta \sum_m \underbrace{E \sum_e f_{e,m} P_{e,m}}_{\text{模态级均衡}}}
$$

两项各自在均衡时为 1（系数归一）。$\alpha, \beta$ 调强度。$\beta$ 大则强逼模态内均衡（防模态专家坍缩），但过大则 router 过度均匀、失去语义分工能力（trade-off，见 §7）。

### 4.4 跨模态路由的对齐增益

设视觉 token $x_v$，若路由到文本专家 $W^{(t)}$，变换后 $x_v W^{(t)}$ 接近文本空间。对齐损失（如对比学习）：

$$
\mathcal{L}_{\text{align}} = -\log \frac{\exp(\text{sim}(x_v W^{(t)}, y_{\text{text}})/\tau)}{\sum_{y'} \exp(\text{sim}(x_v W^{(t)}, y')/\tau)}
$$

其中 $y_{\text{text}}$ 是对应文本概念 embedding。跨模态路由让 $x_v$ 经文本专家变换后**与语言概念对齐**，是 grounding 的机制。这是 LIMoE 对比学习目标（CLIP-style）在 MoE 上的延伸。

### 4.5 激活算力与容量

设专家 FFN 维 $d_{\text{exp}}$，$E$ 个专家，每 token 激活 $k$ 个。dense FFN 算力 $\propto d \cdot d_{\text{exp,dense}}$；MoE 激活算力 $\propto k \cdot d \cdot d_{\text{exp}}$。控激活算力：

$$
k \cdot d_{\text{exp}} \approx d_{\text{exp,dense}} \quad \text{(激活算力持平)}
$$

但总参数 $E \cdot d_{\text{exp}} \gg d_{\text{exp,dense}}$（容量大）。MoE-LLaVA: $E$ 大、$k$ 小，激活 ~3B 但总参数更大。这是稀疏激活的算力 / 容量解耦——容量提（多模态需大容量）但激活算力控（省推理成本）。多模态 MoE 把这个解耦用在"视觉 + 文本"混合 token 上。

### 4.6 期望路由分布与模态混合

设 batch 内视觉 token 占比 $\rho_v$、文本占比 $\rho_t = 1-\rho_v$。若 router 无模态偏好（均匀），每专家被选频次 $f_e \approx k/E$（均匀）。若 router 有模态偏好（视觉 token 偏视觉专家），视觉专家 $e_v$ 被选频次：

$$
f_{e_v} \approx \rho_v \cdot \frac{k}{|\mathcal{E}_v|} + \rho_t \cdot \frac{k \cdot \text{cross-rate}}{|\mathcal{E}_v|}
$$

其中 $|\mathcal{E}_v|$ 是视觉专家数，cross-rate 是文本 token 走视觉专家的比例（跨模态路由）。模态级负载均衡要约束 $f_{e_v}$ 不超过上限（防视觉专家被视觉 token 淹没）。


## 5. 代码示例（可选）

### 5.1 多模态 MoE 路由示意（含模态偏置 + 模态级负载均衡）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalMoE(nn.Module):
    """
    多模态 MoE: 模态偏置 + top-k 路由 + 模态级负载均衡损失.
    - d: 隐维, E: 专家数, k: 激活专家数, n_modalities: 模态数(视觉/文本/音频)
    """
    def __init__(self, d, E=8, k=2, n_modalities=3, d_exp=1024):
        super().__init__()
        self.E, self.k, self.d = E, k, d
        self.router = nn.Linear(d, E, bias=False)
        self.modal_bias = nn.Parameter(torch.zeros(n_modalities, E))  # 模态偏置 b(m)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d_exp), nn.GELU(), nn.Linear(d_exp, d))
            for _ in range(E)
        ])

    def forward(self, x, modality_ids):
        # x: (N, d), modality_ids: (N,) in {0:vis, 1:txt, 2:aud}
        B, d = x.shape
        logits = self.router(x) + self.modal_bias[modality_ids]  # (N, E) + 模态偏置
        topk_vals, topk_idx = logits.topk(self.k, dim=-1)        # (N, k)
        weights = F.softmax(topk_vals, dim=-1)                   # (N, k)
        # 稀疏 dispatch: 只算选中专家
        out = torch.zeros_like(x)
        for i in range(self.k):
            e_idx = topk_idx[:, i]                               # (N,) 每token第i选专家
            w = weights[:, i:i+1]                                # (N,1)
            # 按专家分组算 (实际用 grouped GEMM / scatter)
            for e in range(self.E):
                mask = (e_idx == e)
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])
        # 负载统计 (用于 aux loss)
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)                    # (N, E) 全 softmax 概率
            # 全局
            f = torch.zeros(self.E, device=x.device)             # 被选频次
            for e in range(self.E):
                f[e] = (topk_idx == e).float().mean()
            P = probs.mean(0)                                    # (E,) 平均概率
            aux_global = self.E * (f * P).sum()
            # 模态级
            aux_modal = 0.0
            for m in range(self.modal_bias.shape[0]):
                m_mask = (modality_ids == m)
                if m_mask.sum() == 0:
                    continue
                f_m = torch.zeros(self.E, device=x.device)
                for e in range(self.E):
                    f_m[e] = (topk_idx[m_mask] == e).float().mean()
                P_m = probs[m_mask].mean(0)
                aux_modal += self.E * (f_m * P_m).sum()
        aux = aux_global + aux_modal                              # 总 aux loss (待加系数)
        return out, aux

# 示例: 5 个 token, 模态 [vis, txt, vis, aud, txt]
moe = MultimodalMoE(d=64, E=8, k=2, n_modalities=3)
x = torch.randn(5, 64)
mod = torch.tensor([0, 1, 0, 2, 1])
out, aux = moe(x, mod)
print(out.shape, aux.item())   # torch.Size([5, 64]), aux 标量
```

### 5.2 模态专家坍缩检测

```python
def detect_modal_collapse(router_probs_per_modal, E):
    """
    router_probs_per_modal: dict {模态m: (T_m, E) router 全 softmax 概率}
    返回每模态的熵 H(P_m), 越低越坍缩 (0=完全坍缩, log E=均衡)
    """
    import math
    results = {}
    for m, probs in router_probs_per_modal.items():
        P_m = probs.mean(0)                       # (E,) 模态m内专家平均概率
        H = -(P_m * (P_m.clamp_min(1e-9).log())).sum().item()
        results[m] = {"entropy": H, "max_entropy": math.log(E),
                      "collapse_ratio": 1 - H / math.log(E)}
    return results
# collapse_ratio 0=均衡, 1=完全坍缩
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[MoE]]（稀疏激活 / 专家分工的母概念）、[[MoE路由]]（top-k 路由 / 不可导处理 / softmax 加权的路由机制，本条完全复用其 router 设计）、[[负载均衡损失]]（全局负载均衡损失 $E\sum f_e P_e$，本条加模态级项）、[[ViT与vision encoder]]（视觉 token 的来源，决定模态）、[[projector]]（视觉特征对齐到 LLM 维，是模态进入 MoE 前的对齐）、[[early与late fusion]]（early fusion 把视觉 token 拼进 LLM 触发 FFN 模态冲突，MoE 是其 FFN 升级；CogVLM 是 cross-attn fusion 的 MoE 变体）、[[FFN]]（MoE 替换的就是 FFN）、[[自注意力]]（attention 层与 FFN 层分离，MoE 通常只替 FFN）。
- **下游（应用）**: [[多模态sequence packing]]（packing 后同序列有视觉+文本 token，router 按模态路由，packing 的 sample_ids / 模态标签传给 router 做模态级均衡）、[[VLM RL与多模态reward]]（VLM RL 的 policy 可用 MoE，rollout 时模态专家要稳定）、[[新模型接入]]（推理引擎要支持多模态 MoE 的 router + 专家并行）、LIMoE / CogVLM / CogVLM2 / Uni-MoE / MoE-LLaVA。
- **对比 / 易混**:
  - **多模态 MoE vs [[MoE]]/[[MoE路由]]**：前者专家按模态分工（视觉专家 / 文本专家），router 看模态 + 语义，负载均衡要模态级；后者专家无模态语义（语义分工），router 看语义，全局负载均衡。前者是后者的多模态推广。
  - **多模态 MoE vs [[early与late fusion]]**：fusion 是"模态怎么合"（拼 / cross-attn），MoE 路由是"LLM 内专家怎么路由"。两者正交——early fusion + MoE（CogVLM）是组合。CogVLM 是 fusion + MoE 的混合（见 [[early与late fusion]] §8.2）。
  - **模态专家 vs 跨模态路由**：前者是专家按模态分工（设计目标），后者是 token 跨模态走专家（路由行为）。前者是结构，后者是行为，两者可共存（LIMoE：涌现模态专家 + 跨模态路由）。
  - **CogVLM 视觉专家 vs [[early与late fusion]] cross-attn**：CogVLM 视觉专家是**每层并联 FFN 分支 + 门控**（MoE 式），cross-attn fusion 是**每层插 cross-attn 层**。前者 FFN 层 MoE 化，后者 attention 层 cross 化。CogVLM 是 cross-attn fusion 的 MoE 变体。
  - **多模态 MoE vs 单模态 dense FFN**：前者稀疏激活 + 模态分工（容量大、算力控、消模态冲突），后者全激活 + 单套权重（容量小、模态冲突）。前者用对算力、提容量。


## 7. 常见误区与易错点

> [!warning] 误区 1：多模态 MoE 就是把 MoE 套在 VLM 上
> **片面。** 直接套标准 MoE（如 MoE-LLaVA 早期实验）会**性能退化**——多模态稀疏学习有特殊性（视觉 token 稠密、文本离散，router 初期易乱路由）。需 **MoE-Tuning**（MoE-LLaVA 提出）或**显式分工训练**（Uni-MoE 三阶段）或**显式并联分支**（CogVLM）解决。直接套用会掉点。

> [!warning] 误区 2：模态专家必须显式指定
> **不必。** LIMoE 实证**模态专家自发涌现**——router 不喂模态信号，训练后专家天然专攻某模态。显式分工（Uni-MoE）是加速涌现的手段，不是必需。但显式分工训练收敛更快、更可控。

> [!warning] 误区 3：视觉 token 只能走视觉专家
> **错。** 跨模态路由允许视觉 token 走文本专家（借语言知识做 grounding）。强制视觉 token 只走视觉专家（硬约束 $b(m)$）会失去跨模态对齐能力。LIMoE 的涌现专家里部分是**跨模态混合专家**（既视觉也文本），承担对齐。

> [!warning] 误区 4：模态级负载均衡越强越好
> **不对。** $\beta$ 过大则 router 在每模态内过度均匀，失去**语义分工**能力（同模态内不同 token 该走不同专家）。模态级均衡防"模态专家坍缩"，但要有度——目标是模态内**有分工但不坍缩**，不是模态内完全均匀。$\beta$ 是 trade-off，需调。

> [!warning] 误区 5：多模态 MoE 的 router 与单模态完全不同
> **router 机制相同。** top-k、softmax 加权、不可导处理（[[MoE路由]] §3.2）完全复用。多模态 MoE 的差异在：① router 输入可加模态偏置 $b(m)$（可选）；② 专家初始化 / 训练按模态分工（训练策略）；③ 负载均衡加模态级项。router 的数学形式不变。理解 [[MoE路由]] 是本条前提。

> [!warning] 误区 6：CogVLM 的视觉专家就是标准 MoE
> **不完全。** CogVLM 视觉专家是**每层并联 FFN 分支 + 门控加权**（视觉专家分支 vs 文本主分支），门控 $g(x)$ 是连续加权（不是 top-k 离散选）。与标准 top-k MoE 的离散路由不同——CogVLM 是"软 MoE"（所有 token 都走两分支，加权合并），不是稀疏激活。CogVLM 容量大但**不省激活算力**（两分支都算）。是 MoE 思想的"稠密变体"。

> [!tip] 实践：模态专家分工的检验
> 训练后检验模态专家是否分工：统计每专家被各模态 token 选择的频次矩阵 $M_{e,m}$（专家 $e$ × 模态 $m$）。理想：每专家对某模态有偏好（$M_{e,m}$ 不均匀）但不过度钉死（不 one-hot）。若 $M$ 接近均匀（无分工）→ router 没学到模态分工（调 $\beta$ 或显式分工训练）；若 $M$ 极端 one-hot（专家钉死单模态）→ 模态专家坍缩（调 $\beta$ 或加跨模态路由）。


## 8. 延伸细节

### 8.1 LIMoE：模态专家涌现的开创性证据

LIMoE（Mustafa et al., arXiv:2206.02770, 2022, Google）是首个把 sparse MoE 用在多模态对比学习（CLIP-style）的工作。关键贡献：

1. **模态专家自发涌现**：router 不区分模态，训练后某些专家主要处理视觉、某些主要处理文本。证明 MoE 天然适合多模态——专家分工是多模态数据的自然产物。
2. **entropy-based 正则**：用 router 输出分布的熵替代经典 load balancing loss，防坍缩且训练更稳。
3. **性能**：LIMoE-L/16 零样本 ImageNet 78.6%（vs CLIP-L/14 76.2%）；H/14 84.1%，可比 SOTA（用更大单模态 backbone + 复杂预训练）。dense 等算力下显著优。

LIMoE 是"多模态 MoE 路由"概念的事实奠基，后续 CogVLM / Uni-MoE / MoE-LLaVA 都在其启发下。

### 8.2 CogVLM 视觉专家：架构层显式分工

CogVLM（Wang et al., arXiv:2311.03079, 2023, 清华/智谱）的"Visual Expert for Pretrained Language Models"——LLM 每层 attention 与 FFN 都加可训练视觉专家分支（与冻结 / 主分支文本路径并联），门控加权。是"显式并联分支"路径的代表。CogVLM-17B 在 10 个跨模态 benchmark SOTA（NoCaps / Flickr30k captioning / RefCOCO / GQA / ScienceQA / VizWiz VQA / TDIUC 等），超或匹配 PaLI-X 55B。CogVLM2（arXiv:2408.16500）继承视觉专家 + 1344×1344 高分辨率 + 视频理解（CogVLM2-Video）。

### 8.3 Uni-MoE 与 MoE-LLaVA：训练策略对比

| 维度 | Uni-MoE（arXiv:2405.11273） | MoE-LLaVA（arXiv:2401.15947） |
|---|---|---|
| 模态分工 | 显式（三阶段：对齐→单模态训专家→合并） | 隐式（标准 MoE + MoE-Tuning） |
| 模态信号 | 阶段 2 用单模态数据激活专家偏好 | 不显式 |
| 训练稳定性 | 三阶段渐进，稳 | MoE-Tuning 治多模态稀疏退化 |
| 并行 | 模态级数据并行 + 专家级模型并行 | 标准 MoE 并行 |
| 主打 | 统一多模态（多模态混合减 bias） | 稀疏 VLM 基线 |

两者都解决"直接套 MoE 会退化"，但路径不同（Uni-MoE 显式分工 vs MoE-LLaVA 隐式 + 训练策略）。

### 8.4 与专家并行的关系

多模态 MoE 的专家常做 [[专家并行]]（不同专家放不同 GPU）。多模态版加一层：**模态级专家并行**——视觉专家放一组 GPU、文本专家放另一组，按模态路由 dispatch。模态级负载均衡在这里直接影响 All-to-All 通信效率（某模态专家过载 → 该组 GPU 瓶颈）。是 [[负载均衡损失]] §8.3"均衡与系统效率"的多模态延伸。

### 8.5 多模态 MoE 与 packing 的协同

[[多模态sequence packing]] 把视觉+文本 token 拼进同序列。多模态 MoE 的 router 对这序列按 token 路由——视觉 token 走视觉专家、文本走文本专家。packing 的 sample_ids / 模态标签是 router 的输入（决定模态偏置 $b(m)$ 与模态级负载统计）。两者协同：packing 提 token 密度，MoE 提 FFN 容量，共同优化多模态训练 / 推理效率。

### 8.6 内容来源

本条整理自 LIMoE（Mustafa et al., arXiv:2206.02770, 2022）、CogVLM（Wang et al., arXiv:2311.03079, 2023）、CogVLM2（arXiv:2408.16500, 2024）、Uni-MoE（Li et al., arXiv:2405.11273, 2024）、MoE-LLaVA（Lin et al., arXiv:2401.15947, 2024）等论文（arxiv 元数据已核实）。模态级负载均衡损失形式由 [[负载均衡损失]] 的全局形式推广而来。CogVLM 视觉专家描述参照 [[early与late fusion]] §8.2。具体实验数字以原论文为准，本条引用的均为论文摘要 / abstract 已确认值。

---
相关: [[多模态batching]] | [[MoE]] | [[MoE路由]] | [[负载均衡损失]] | [[ViT与vision encoder]] | [[projector]] | [[early与late fusion]] | [[多模态sequence packing]] | [[VLM RL与多模态reward]] | [[FFN]] | [[自注意力]] | [[新模型接入]]
