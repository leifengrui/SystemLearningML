# VLM RL与多模态reward

> **所属章节**: [[VLM RL]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: VLM RL / 多模态 RL / MLLM 对齐 / 多模态 reward / multimodal reward / vision ground truth reward / 视觉真值 reward / VLM RLHF / VLM GRPO / object hallucination 治理 / 多模态 verifier
> **难度**: 中高（需懂 [[GRPO]]、[[verifier与function reward]]、[[reward hacking]]、[[ViT与vision encoder]]、[[projector]]、[[多模态sequence packing]]、[[多模态MoE路由]]、[[early与late fusion]]）


## 1. 一句话定义

**VLM RL 与多模态 reward** 是视觉语言模型（VLM / MLLM）用强化学习做对齐的范式——VLM 也跑 [[GRPO]] / PPO / DPO，但比纯文本 RLHF **多了一个视觉维度**：reward 信号不再只来自文本偏好，还要核实"模型说的和图里真有的"是否一致。reward 来源分三大类：① **多模态 RM（图文对打分模型）**——用人类 / AI 偏好对训的神经网络 $r_\phi(\text{图},\text{文})\in\mathbb{R}$，给"图文匹配度"打分（LLaVA-RLHF 的 factually-augmented RM、RLHF-V 的 segment-level RM）；② **rule-based vision ground truth（视觉真值 reward）**——用 OCR / 物体检测 / 图表判读 / VQA 答案匹配等**可验证的视觉事实**判 response 对错（图里有几个物体、什么颜色、什么文字，输出 0/1 或部分分），比文本 reward 更可验证；③ **多模态 verifier（生成描述与图像内容一致性验证）**——专门检测 **object hallucination**（模型说图里有但实际没有的物体），如 POPE 式 polling（对"图里有没有 X"答 yes/no 判对错）、CHAIR（caption 中幻觉物体比例）。多模态 reward 的核心难点是**视觉理解主观性强、object hallucination 普遍、reward 噪声大**——但 **vision ground truth（图里客观有几个物体、什么颜色）比文本更可验证**，是 VLM RL 治 object hallucination 的关键抓手。代表：LLaVA-RLHF（Factually Augmented RLHF，arXiv:2309.14525）、RLHF-V（dense segment-level DPO，arXiv:2312.00849，CVPR 2024）、RLAIF-V（开源 AI 反馈，arXiv:2405.17220，object hallucination -80.7%）。

> [!note] 三句话定位
> - **是什么**：VLM 跑 GRPO/PPO/DPO 对齐，reward 三来源——多模态 RM（图文对打分网络）、rule-based vision ground truth（OCR/检测/VQA 答案匹配，可验证视觉事实）、多模态 verifier（object hallucination 检测，POPE/CHAIR）。
> - **为什么**：VLM 头号病是 **object hallucination**（说图里有但实际没有），文本 RLHF 的偏好 RM 治不了"视觉事实错"；vision ground truth 是**图里客观可验证的事实**（几个物体、什么颜色），比文本 reward 噪声小、可精确判对错，是治幻觉的根本抓手。
> - **与 [[verifier与function reward]]/[[GRPO]]/[[reward hacking]] 关系**：reward 三来源是 [[verifier与function reward]] 的多模态版（RM ↔ rule-based verifier ↔ function reward 的多模态映射）；GRPO 的 group-mean baseline 在多模态 reward 下同样要防全对/全错组（见 [[sequence packing与动态采样]]）；多模态 RM 同样易 reward hacking（Goodhart），vision ground truth 是防 hack 的根本手段（程序判对错，actor 难钻空子）。

> [!note] 别名辨析
> - **多模态 reward**：泛指 VLM RL 的 reward 信号（含三类来源）。
> - **vision ground truth reward**：特指"用图像客观事实判对错"的 rule-based reward（第二类）。
> - **object hallucination 治理**：特指用 reward / verifier 减少"模型说图里有但实际没有"的幻觉行为（VLM RL 的核心应用之一）。
> 三者重叠但有侧重，本条统一讲。


## 2. 为什么需要它（动机与背景）

### 2.1 VLM 的头号病：object hallucination

VLM（LLaVA / Qwen-VL / InternVL）的致命问题是 **object hallucination（物体幻觉）**——模型生成描述图的内容时，**说出图里实际没有的物体**。POPE（Li et al., arXiv:2305.10355, EMNLP 2023）实证：主流 LVLM "mostly suffer from severe object hallucination issue"——对"图里有没有 X"的 yes/no 问题，模型倾向答 yes（即便没有），尤其在物体**频繁出现于训练数据**或**与图中物体共现**时更易幻觉。

根源：

1. **LLM 语言先验压过视觉证据**：LLM 见过"桌上有杯子和书"，即便图里只有杯子，也倾向生成"杯子和书"（语言先验 > 视觉 grounding）；
2. **视觉特征对齐不足**：[[projector]] 把视觉特征翻译到 LLM 空间若有损，视觉信息进 LLM 时已弱；
3. **训练数据偏描述生成**：SFT 数据多是"看图说话"，模型学的是流畅描述而非"只说图里真有的"。

object hallucination 让 VLM 不可信（高风险场景如医疗 / 自动驾驶 / 文档解析不能用）。**VLM RL 的首要目标就是治幻觉**。

### 2.2 文本 RLHF 的偏好 RM 治不了视觉事实错

文本 RLHF（[[RLHF (PPO)]]）用偏好 RM $r_\phi$ 给"哪个回答更好"打分。直接搬到 VLM：

- **偏好 RM 只看文本流畅度 / 偏好**：RM 学"人类觉得哪个回答好"，但人类标注者也可能被流畅描述骗（说"杯子和书"看起来合理，没核对图）。RM 学到的偏好≠视觉事实正确。
- **Goodhart 放大**（[[reward hacking]]）：actor 钻 RM 漏洞——拉长描述、堆砌物体名（RM 偏好丰富描述），刷高 $r_\phi$ 但幻觉更多。LLaVA-RLHF（arXiv:2309.14525）明确指出 RLHF 在 VLM 上有 reward hacking 现象。

故纯偏好 RM 治不了"视觉事实错"。需要**直接核实视觉事实**的 reward——这就是 vision ground truth。

### 2.3 vision ground truth：图像客观事实比文本更可验证

关键洞察：**图像里的客观事实比文本主观偏好更可验证**。

| reward 类型 | 可验证性 | 噪声 | 来源 |
|---|---|---|---|
| 文本偏好 RM | 低（主观） | 高（标注者分歧） | 人类偏好对 |
| 多模态 RM（图文对） | 中（图文匹配主观） | 中 | 图文偏好对 |
| **vision ground truth** | **高（图里客观有/无）** | **低（程序判对错）** | OCR / 检测 / VQA 答案 |

图里**客观有几个物体、什么颜色、什么文字**是可验证事实——用 OCR 提文字、用物体检测器提物体集合、用 VQA benchmark 对答案，程序铁面无私判对错。这比"人类觉得哪个回答好"精确得多。是 [[verifier与function reward]] 在多模态的映射——文本 rule-based verifier 对数学答案，多模态 rule-based verifier 对视觉事实。

### 2.4 reward 来源三大类

VLM RL 的 reward 来源分三大类（与 [[verifier与function reward]] 的三来源对应）：

1. **多模态 RM（图文对打分模型）**：神经网络 $r_\phi(\text{图}, \text{文})$，用图文偏好对训（RLHF-V 的 segment-level RM、LLaVA-RLHF 的 factually-augmented RM）。处理**开放式、主观、难精确判对错**任务（看图对话、描述质量）。
2. **rule-based vision ground truth（视觉真值 reward）**：用 OCR / 物体检测 / 图表判读 / VQA 答案匹配判对错，输出 0/1 或部分分。处理**有视觉标准答案、可精确验证**任务（VQA 对答案、文档 OCR 对文字、图表 QA 对数值）。
3. **多模态 verifier（生成描述与图像一致性验证）**：专门检测 object hallucination——用 POPE 式 polling（对"图里有没有 X"答 yes/no）、CHAIR（caption 中幻觉物体比例）、object detection 比对生成描述与图实际物体。是 vision ground truth 的"幻觉检测特化版"。

### 2.5 多模态 reward 的难点

虽然 vision ground truth 可验证，但多模态 reward 整体噪声大、难点多：

1. **视觉理解主观性强**：开放式看图对话（"描述这张图的氛围"）无标准答案，RM 偏好分歧大；
2. **object hallucination 普遍且隐蔽**：模型说"图里有猫"但图里没有，需 verifier 主动检测，不能等用户发现；
3. **reward 噪声大**：vision ground truth 依赖 OCR / 检测器（这些工具本身有错），RM 依赖偏好标注（标注者分歧），reward 信号噪声大；
4. **稀疏 vs 稠密**：outcome reward（最终对错）稀疏，segment-level reward（RLHF-V 的片段级）稠密但标注贵；
5. **跨模态一致性验证难**：判断"生成的描述与图像内容一致"本身需多模态理解，verifier 自己也可能错。

### 2.6 与 GRPO 的适配

[[GRPO]] 的 group-mean baseline 要求 reward **可比**——同 prompt 多采几条 response，reward 横向比较。多模态 reward 的三类都满足：

- 多模态 RM：$r_\phi$ 输出标量，可比；
- vision ground truth：0/1 或部分分，可比（VQA 对错、检测物体数）；
- verifier：POPE 准确率、CHAIR 分数，可比。

但 GRPO 的**全对/全错组问题**（见 [[sequence packing与动态采样]] §2.2）在 VLM 同样存在——若一个 prompt 的 $G$ 条 response 全答对（VQA 全对）或全答错，$\sigma_r=0$，advantage=0，无信号。需 dynamic sampling 过滤。多模态 RL 的 rollout 同样要做 [[多模态sequence packing]] + 动态采样。


## 3. 核心概念详解

### 3.1 reward 三来源对比

| 维度 | ① 多模态 RM | ② rule-based vision ground truth | ③ 多模态 verifier |
|---|---|---|---|
| 形式 | 神经网络 $r_\phi(\text{图},\text{文})\in\mathbb{R}$ | 程序判对错 0/1 / 部分分 | 一致性检测分数 |
| 数据 | 图文偏好对（人类/AI 标） | OCR/检测器/VQA 标准答案 | POPE/CHAIR benchmark |
| 可验证性 | 中（图文匹配主观） | **高（客观事实）** | 高（程序判） |
| 噪声 | 中高（标注分歧 + RM proxy） | 低（程序铁面，但依赖工具准度） | 低（但 verifier 自身可能错） |
| 适用任务 | 开放看图对话、描述质量 | VQA / OCR / 图表 QA / 检测 | object hallucination 治理 |
| reward hacking 风险 | **高**（Goodhart，actor 钻 RM 漏洞） | 低（程序判，难钻） | 低 |
| 稠密度 | 稠密（每段打分，可选） | 稀疏（最终对错） | 中（per-object 分） |
| 训练成本 | 贵（标偏好 + 训 RM） | 便宜（写规则 / 用现成工具） | 便宜（用 benchmark） |
| 代表 | LLaVA-RLHF RM、RLHF-V RM | VQA answer match、OCR ground truth | POPE、CHAIR、MMHal-Bench |

> [!note] 三来源的映射关系
> 这是 [[verifier与function reward]] 的多模态版映射：
> - 多模态 RM ↔ 文本 RM（偏好网络）
> - vision ground truth ↔ rule-based verifier（对答案 / 正则）
> - 多模态 verifier ↔ function reward（执行反馈，这里是"视觉一致性执行反馈"）
> 三者的"精确 vs 主观""可验证 vs proxy""防 hack 程度"权衡完全同构。

### 3.2 多模态 RM（图文对打分模型）

多模态 RM $r_\phi(I, y)$ 输入图 $I$ + 文本 response $y$，输出标量分数。训练用偏好对 $(I, y_c, y_l)$（chosen / rejected），Bradley-Terry 损失（同文本 RM）：

$$
\mathcal{L}_{\text{RM}} = -\log \sigma(r_\phi(I, y_c) - r_\phi(I, y_l))
$$

两种特化：

1. **factually-augmented RM（LLaVA-RLHF, arXiv:2309.14525）**：RM 输入**除了图文，还喂图像 caption + ground-truth 多选题选项**作为"事实辅助"，让 RM 判"是否与图像事实一致"而非纯偏好。**直接目的是缓解 reward hacking**——给 RM 事实信息，让它不易被流畅但幻觉的描述骗。LLaVA-RLHF 实证：factually-augmented RM 显著降 reward hacking，提性能。
2. **segment-level RM（RLHF-V, arXiv:2312.00849）**：不是整段 response 打一个分，而是**片段级**——人类标注者标"哪段描述幻觉了"，RM 学片段级分数，做 dense DPO。RLHF-V 用 1.4k 样本降幻觉 34.8%（超 LLaVA-RLHF 用 10k 样本的效果），数据效率高。

### 3.3 rule-based vision ground truth（视觉真值 reward）

用图像客观事实判 response 对错。典型形式：

- **VQA answer match**：VQA benchmark 有标准答案，response 答案与 ground truth 字符串匹配（或软匹配），对=1 错=0。例 GQA / VQAv2。
- **OCR ground truth**：文档 / 截图任务，图里文字可 OCR 提取作 ground truth，response 抄写的文字与 OCR 结果比对。
- **object count / attribute**：图里有几个物体、什么颜色、什么形状，用检测器提 ground truth 物体集合，response 描述的物体与集合比对。
- **chart / table reading**：图表 QA，图里数值可程序提取作 ground truth，response 答的数值与 ground truth 比对（容差内对）。

reward：

$$
r_{\text{VGT}}(y) = \mathbb{1}[\text{match}(y, \text{ground-truth}(I))] \quad \text{(0/1)}
$$

或部分分（多物体对几个给几分）。这是**最可验证的多模态 reward**——程序铁面无私，actor 难钻空子（除非写出让 OCR / 检测器误判的 response，但极难且可防）。

> [!tip] 实践：vision ground truth 是治幻觉的根本手段
> vision ground truth 用程序判视觉事实对错，是 [[reward hacking]] 的天然屏障——actor 没法靠"流畅描述"骗过 OCR / 检测器。对 VQA / OCR / 图表等可验证任务，**优先用 vision ground truth 而非 RM**。LLaVA-RLHF 的 factually-augmented RM 本质就是"把 vision ground truth 注入 RM"降 hack。

### 3.4 多模态 verifier（object hallucination 检测）

专门检测"模型说图里有但实际没有"的幻觉。代表方法：

- **POPE（Polling-based Object Probing Evaluation, arXiv:2305.10355, EMNLP 2023）**：对图问"图里有没有 X"（X 是某物体），模型答 yes/no，与图实际物体集（来自 ground-truth 标注 / 检测器）比对。随机 / 流行 / 对抗三种采样 X（对抗采样专挑易幻觉的）。POPE 准确率作为 reward 或评估指标。
- **CHAIR（Caption Hallucination Assessment with Image Relevance）**：对模型生成的 caption，统计其中提到的物体有多少在图里实际有（recall）与多少是幻觉（hallucinated）。CHAIR₁ / CHAIRₛ 衡量幻觉率。
- **MMHal-Bench（LLaVA-RLHF 提出）**：专门评多模态幻觉的 benchmark，关注 12 个物体类别。

verifier 作为 reward：

$$
r_{\text{verifier}}(y) = 1 - \text{hallucination\_rate}(y, I) \quad \text{(幻觉越少 reward 越高)}
$$

或 POPE 准确率。这是"生成描述与图像一致性"的程序验证——与 [[verifier与function reward]] 的 function reward（代码沙箱执行反馈）同构，这里是"视觉执行反馈"。

### 3.5 RLAIF-V：开源 AI 反馈降幻觉

RLAIF-V（Yu et al., arXiv:2405.17220, 2024）走**开源 AI 反馈**路线——不用人类标偏好，用开源 MLLM 自己生成偏好对（self-feedback）。关键结果：RLAIF-V 7B **降 object hallucination 80.7%、整体 hallucination 33.7%**，RLAIF-V 12B **超 GPT-4V trustworthiness**。是"用大模型当 verifier 降幻觉"的代表——AI 反馈 + 自我对齐，免人工标注。

### 3.6 VLM RL 的训练流程

VLM RL（GRPO 范式）：

1. **policy**：VLM $\pi_\theta(\text{文} \mid \text{图}, \text{prompt})$（视觉 encoder + projector + LLM，见 [[ViT与vision encoder]]、[[projector]]、[[early与late fusion]]）；
2. **rollout**：对 prompt + 图采样 $G$ 条 response（GRPO group）；
3. **reward**：三来源任选 / 组合（多模态 RM + vision ground truth + verifier）；
4. **advantage**：GRPO group-mean baseline $\hat A_i = (r_i - \bar r)/\sigma_r$（[[GRPO]]）；
5. **update**：PPO clip + KL to reference policy（同 [[GRPO]]）。

与文本 GRPO 唯一区别：rollout 输入含图、reward 含视觉事实核实。但 [[多模态sequence packing]]、dynamic sampling、reward 三来源组合是 VLM RL 独有工程。

### 3.7 object hallucination 治理的 reward 组合

治幻觉的典型 reward 组合（实践）：

$$
r(y) = \underbrace{r_{\text{VGT}}(y)}_{\text{视觉真值}} + \lambda_1 \underbrace{r_{\text{verifier}}(y)}_{\text{幻觉检测}} + \lambda_2 \underbrace{r_{\text{RM}}(y)}_{\text{流畅度}}
$$

- $r_{\text{VGT}}$：VQA / OCR 对错（强可验证）；
- $r_{\text{verifier}}$：POPE / CHAIR 幻觉率（强可验证）；
- $r_{\text{RM}}$：流畅度 / 偏好（保生成质量，但权重小防 hack）。

$\lambda_2$ 小（RM 权重低）防 reward hacking；$\lambda_1$ 大（verifier 权重高）强逼降幻觉。这是"用可验证 reward 主导、RM 辅助"的多模态 RL 范式。

### 3.8 与 RLHF-V / RLAIF-V 的方法谱系

| 方法 | reward 来源 | 训练算法 | 数据 | 关键结果 |
|---|---|---|---|---|
| LLaVA-RLHF（arXiv:2309.14525） | 多模态 RM（factually-augmented） | RLHF (PPO) | 10k 偏好对 | MMHal-Bench +60%，LLaVA-Bench 94% GPT-4 |
| RLHF-V（arXiv:2312.00849, CVPR 2024） | 多模态 RM（segment-level） | dense DPO | 1.4k 段级标注 | 幻觉 -34.8%（超 LLaVA-RLHF 10k） |
| RLAIF-V（arXiv:2405.17220） | 多模态 RM（AI 反馈 / 自我对齐） | DPO + 推理时 self-feedback | AI 生成偏好对 | object hallucination -80.7%，超 GPT-4V |
| Generative RLHF-V（arXiv:2505.18531） | 生成式 RM（GRM，RL 训） | 多模态 RLHF | grouped 比较 | 7 benchmark +18.1%（vs RLHF +5.3%） |

谱系演进：人类偏好（LLaVA-RLHF）→ 段级 dense（RLHF-V）→ AI 反馈（RLAIF-V）→ 生成式 RM（Generative RLHF-V）。趋势是**降人工、提可验证、防 hack**。


## 4. 数学原理 / 公式

### 4.1 多模态 RM 的 Bradley-Terry

偏好对 $(I, y_c, y_l)$（图 $I$，chosen $y_c$，rejected $y_l$）。RM $r_\phi(I, y)$，Bradley-Terry 损失（同文本 RM）：

$$
\boxed{\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(I, y_c, y_l)} \log \sigma(r_\phi(I, y_c) - r_\phi(I, y_l))}
$$

最优 RM 满足 $r_\phi(I, y_c) > r_\phi(I, y_l)$。factually-augmented 版：RM 输入加事实 $F(I)$（caption / 选项）：

$$
r_\phi^{\text{aug}}(I, y; F(I)) \quad \text{(输入含事实辅助)}
$$

降 hack 的机制：RM 拿到事实 $F(I)$，判 $y$ 与 $F(I)$ 一致性，不只看 $y$ 流畅度。

### 4.2 vision ground truth reward

图 $I$ 的 ground truth 事实 $g(I)$（物体集 / OCR 文字 / VQA 答案）。response $y$ 提取的事实 $\hat g(y)$（解析 $y$ 得物体名 / 文字 / 答案）。reward：

$$
\boxed{r_{\text{VGT}}(y) = \text{match}(\hat g(y), g(I)) \in [0, 1]}
$$

- 物体集匹配：$r = |\hat g(y) \cap g(I)| / |\hat g(y)|$（recall of mentioned objects，幻觉物体拉低分）；
- OCR 字符串匹配：$r = \text{edit\_similarity}(\hat g(y), g(I))$；
- VQA 答案匹配：$r = \mathbb{1}[\hat g(y) = g(I)]$（软匹配 yes/no）。

### 4.3 POPE verifier reward

POPE 对图 $I$ 问"有没有 X"，模型答 $\hat a \in \{\text{yes}, \text{no}\}$，ground truth $a^* \in \{\text{yes}, \text{no}\}$（X 在图里 yes / 否则 no）。POPE 准确率：

$$
\text{POPE-acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat a_i = a_i^*]
$$

reward $r_{\text{POPE}} = \text{POPE-acc}$。对抗采样（挑易幻觉 X）使 reward 更聚焦幻觉治理。

### 4.4 CHAIR 幻觉率

设 caption 提到物体集 $\hat{\mathcal{O}}(y)$，图实际物体集 $\mathcal{O}(I)$。幻觉物体 = $\hat{\mathcal{O}}(y) \setminus \mathcal{O}(I)$。

$$
\text{CHAIR}_i = \frac{|\hat{\mathcal{O}}(y) \setminus \mathcal{O}(I)|}{|\hat{\mathcal{O}}(y)|} \quad \text{(per-instance, 物体级幻觉率)}
$$

$$
\text{CHAIR}_s = \frac{\mathbb{1}[\hat{\mathcal{O}}(y) \setminus \mathcal{O}(I) \ne \emptyset]}{1} \quad \text{(per-sentence, 句级是否幻觉)}
$$

reward $r_{\text{CHAIR}} = 1 - \text{CHAIR}_i$（幻觉越少越高）。

### 4.5 GRPO 在多模态 reward 下的 advantage

GRPO（[[GRPO]]）对图 + prompt 采 $G$ 条 response，reward $r_i$（三来源组合）。advantage：

$$
\boxed{\hat A_i = \frac{r_i - \bar r}{\sigma_r + \epsilon}, \quad \bar r = \frac{1}{G}\sum_j r_j, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_j (r_j - \bar r)^2}}
$$

与文本 GRPO 同形。**全对/全错组问题**（[[sequence packing与动态采样]] §2.2）同样存在：若 $G$ 条 response 全答对（VQA 全对，$r_i$ 全 1）或全答错（全 0），$\sigma_r=0$，$\hat A_i=0$，无梯度。需 dynamic sampling 过滤。

### 4.6 reward 期望与方差（多模态噪声分析）

设多模态 reward $r = r^* + \eta$，$r^*$ 真实质量，$\eta$ 噪声。三类来源噪声方差：

| reward 来源 | 噪声 $\eta$ 方差 | 来源 |
|---|---|---|
| 多模态 RM | $\sigma_{\text{RM}}^2$（大） | RM proxy 偏差 + 标注分歧 + Goodhart |
| vision ground truth | $\sigma_{\text{VGT}}^2$（小） | 仅来自工具（OCR/检测器）准度 |
| verifier | $\sigma_{\text{ver}}^2$（中） | verifier 自身误判 |

GRPO advantage $\hat A_i = (r_i - \bar r)/\sigma_r$ 中，$\sigma_r$ 含真实 reward 方差 + 噪声方差。若噪声主导（多模态 RM），$\sigma_r$ 大但**信号被噪声淹没**，advantage 估计噪。用 vision ground truth（$\sigma_{\text{VGT}}^2$ 小）则 $\sigma_r$ 主要反映真实 reward 差异，advantage 信号清晰。这是"可验证 reward 降噪声、提信号"的数学根源。

### 4.7 reward hacking 在多模态的形式

多模态 RM 的 reward hacking（[[reward hacking]] 的多模态版）：

- **物体堆砌**：actor 堆砌物体名（"图里有猫、狗、树、车..."），RM 偏好丰富描述给高分，但多数物体是幻觉；
- **长度膨胀**：拉长描述拿分（RM 偏好长回答）；
- **迎合 RM 偏好**：actor 学到 RM 盲区，刷高 $r_\phi$ 但视觉事实错。

形式化：$\mathbb{E}[r_\phi]$ 涨但 $\mathbb{E}[r^*]$（真实视觉正确性）降。防法：vision ground truth reward 主导（程序判，难 hack）+ factually-augmented RM（给 RM 事实）+ 长度惩罚 + KL 约束。LLaVA-RLHF 实证 factually-augmented RM 降 hack。


## 5. 代码示例（可选）

### 5.1 vision ground truth verifier（VQA answer match + object detection 比对）

```python
import re

def vqa_answer_match(response, gt_answer):
    """VQA 风格答案匹配: 软匹配 (去标点/小写/去冠词)."""
    def norm(s):
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\b(a|an|the)\b", "", s).strip()
        return s
    return 1.0 if norm(response) == norm(gt_answer) else 0.0

def object_hallucination_reward(response, detected_objects):
    """
    response: 模型生成的描述
    detected_objects: 图实际物体集合 (来自检测器/ground-truth), set of str
    返回 reward: 提到的物体中实际有的比例 (recall of mentioned, 幻觉物体拉低分)
    """
    # 简化: 从 response 提取候选物体名 (实际用 NER / 物体名词词典)
    mentioned = set(w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", response))
    # 与图实际物体比对
    if not mentioned:
        return 0.0
    correct = len(mentioned & detected_objects)
    hallucinated = len(mentioned - detected_objects)
    # reward = 正确提到的占比 (幻觉越多越低)
    return correct / (correct + hallucinated) if (correct + hallucinated) > 0 else 0.0

# 示例
gt = {"cat", "table", "cup"}            # 图实际有
resp = "I see a cat, a table, a cup and a dog."  # dog 是幻觉
print(object_hallucination_reward(resp, gt))  # 3/(3+1)=0.75
print(vqa_answer_match("a cat", "cat"))        # 1.0
```

### 5.2 POPE polling reward

```python
def pope_reward(model, image_objects, candidate_objects, n_query=20):
    """
    POPE 式 polling: 对图问 "Is there a X?" 模型答 yes/no,
    与图实际物体集比对, 返回准确率作 reward.
    image_objects: 图实际物体集
    candidate_objects: 采样问的物体 (含正负样本)
    """
    correct = 0
    for x in candidate_objects[:n_query]:
        a_star = "yes" if x in image_objects else "no"
        a_hat = model.answer(f"Is there a {x} in the image? yes or no.")
        if a_hat.strip().lower().startswith(a_star[0]):
            correct += 1
    return correct / n_query
```

### 5.3 多模态 GRPO rollout 的 reward 组合

```python
def multimodal_reward(response, image, gt, vlm_rm=None):
    """
    VLM RL reward 组合: vision ground truth + verifier + RM (低权重防 hack)
    """
    r_vgt = object_hallucination_reward(response, gt["objects"])     # 视觉真值
    r_verifier = vqa_answer_match(response, gt["vqa_answer"])         # VQA 对错
    r_rm = vlm_rm(image, response) if vlm_rm else 0.0                 # 流畅度 RM
    # 组合: 可验证 reward 主导, RM 低权重防 hack
    return r_vgt + 0.5 * r_verifier + 0.1 * r_rm

# GRPO group: 对同一 (图, prompt) 采 G 条 response
def grpo_group_advantage(rewards):
    """rewards: list of G 个 reward, 返回 advantage (group-mean baseline)"""
    import statistics
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards) + 1e-8
    return [(r - mean) / std for r in rewards]
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GRPO]]（VLM RL 的核心算法，group-mean baseline / advantage / PPO clip + KL，本条 reward 喂给它）、[[verifier与function reward]]（reward 三来源的母概念，本条是其多模态映射：RM ↔ rule-based verifier ↔ function reward）、[[reward hacking]]（多模态 RM 同样易 Goodhart，vision ground truth 是防 hack 根本手段）、[[ViT与vision encoder]] + [[projector]]（VLM policy 的视觉前端，决定视觉信息质量，影响幻觉根源）、[[early与late fusion]]（fusion 范式决定视觉信息进 LLM 的方式，early fusion 视觉 grounding 弱易幻觉）、[[多模态sequence packing]]（VLM RL rollout 的 batch packing + per-trajectory reward/advantage 隔离）、[[多模态MoE路由]]（VLM policy 可用 MoE，RL 时模态专家稳定性）、[[sequence packing与动态采样]]（全对/全错组问题在 VLM RL 同样存在，需 dynamic sampling）、[[Reward Model]]（多模态 RM 的母概念）。
- **下游（应用）**: VLM 对齐落地（LLaVA-RLHF / RLHF-V / RLAIF-V / Generative RLHF-V）、object hallucination 治理（POPE / CHAIR / MMHal-Bench）、VQA / OCR / 图表 QA 的 RL 训练、多模态助手可信化（医疗 / 文档 / 自动驾驶场景）、[[新模型接入]]（推理引擎支持 VLM RL 的 rollout + reward）。
- **对比 / 易混**:
  - **VLM RL vs 文本 RLHF（[[GRPO]]）**：前者 reward 含视觉事实核实（三来源），policy 输入含图；后者 reward 纯文本偏好，policy 纯文本。前者多视觉维度，治 object hallucination；后者治文本偏好对齐。
  - **多模态 reward vs [[verifier与function reward]] 三来源**：RM ↔ RM、vision ground truth ↔ rule-based verifier、多模态 verifier ↔ function reward。映射同构，前者是后者的多模态版。
  - **vision ground truth vs 多模态 RM**：前者程序判视觉事实（可验证、低噪声、防 hack），后者网络打图文偏好分（主观、proxy、易 hack）。前者精确但限可验证任务，后者开放但 proxy。
  - **POPE verifier vs 多模态 RM**：POPE 程序判幻觉（强可验证），RM 网络判偏好（弱可验证）。POPE 是治幻觉特化 verifier，RM 是开放打分。
  - **VLM RL vs VLM SFT**：SFT 模仿描述数据（学流畅），RL 用 reward 优化（学视觉事实正确）。SFT 不直接治幻觉（数据偏描述），RL 用 vision ground truth 强逼降幻觉。


## 7. 常见误区与易错点

> [!warning] 误区 1：VLM RL 就是把文本 RLHF 搬到 VLM
> **片面。** 直接搬文本偏好 RM 到 VLM 会 reward hacking（actor 堆砌物体刷分但幻觉更多，LLaVA-RLHF 实证）。VLM RL 的关键是**加 vision ground truth / verifier reward**（程序判视觉事实），不只靠偏好 RM。factually-augmented RM（给 RM 事实）是降 hack 的核心创新。

> [!warning] 误区 2：多模态 RM 越大越好
> **不对。** 多模态 RM 是 proxy，优化越深越 Goodhart（[[reward hacking]]）。Gao et al. 2023 实证 scaling 不能根治 Goodhart。多模态 RM 权重应低（$\lambda_2$ 小），让 vision ground truth / verifier 主导。RM 大只推迟 hack 发生，不消除。

> [!warning] 误区 3：vision ground truth 完全无噪声
> **不。** vision ground truth 依赖 OCR / 检测器 / 标注，这些工具本身有错（OCR 误识、检测漏检）。噪声比 RM 小但非零。verifier 自身也可能误判（判断"描述与图一致"本身需多模态理解，verifier 可能也幻觉）。reward 仍有噪声，需 GRPO 的 group-mean baseline 降噪。

> [!warning] 误区 4：object hallucination 只用 reward 治
> **不全面。** reward 是治幻觉的重要手段但非全部。根源还在视觉 grounding 弱（[[projector]] 对齐不足、[[early与late fusion]] 视觉信息进 LLM 后被语言先验压）。治幻觉要**多管齐下**：reward（vision ground truth）+ 数据（去偏、加反幻觉样本）+ 架构（提视觉 grounding，如 CogVLM 视觉专家）+ 推理时（RLAIF-V self-feedback）。

> [!warning] 误区 5：VLM RL 的 GRPO 与文本 GRPO 不同
> **GRPO 机制相同。** group-mean baseline、advantage z-score、PPO clip + KL、全对/全错组问题、dynamic sampling 都同 [[GRPO]]、[[sequence packing与动态采样]]。VLM RL 的差异在：rollout 输入含图、reward 三来源含视觉事实。算法骨架不变。理解 [[GRPO]] 是本条前提。

> [!warning] 误区 6：POPE 准确率高就没幻觉
> **不。** POPE 是 yes/no polling，只测"有没有 X"的判别能力，不测开放 caption 的幻觉。POPE 高但 caption 仍可能幻觉（描述里说没问到的物体）。需配合 CHAIR（caption 幻觉率）综合评。单一指标不全。

> [!tip] 实践：治幻觉的 reward 组合策略
> 优先用 vision ground truth（VQA/OCR 对错）+ verifier（POPE/CHAIR）作主 reward（强可验证、防 hack），多模态 RM 作辅助（保流畅，权重低 $\lambda_2 \le 0.1$）。LLaVA-RLHF 的 factually-augmented RM 是把 vision ground truth 注入 RM 降 hack 的范式。RLAIF-V 的 AI 反馈 + self-feedback 是免人工版。组合而非单一。

> [!note] 补充：VLM RL 的 rollout 工程
> VLM RL 的 rollout 要 [[多模态sequence packing]]（图文变长 batch packing + per-trajectory reward 隔离）+ dynamic sampling（过滤全对/全错组）+ 多模态 reward 计算（OCR/检测器调用开销大，常异步 / 缓存）。这些工程与 [[17-RL训推一体框架]] 的 rollout 体系一致，本条侧重 reward 来源，工程细节见 [[sequence packing与动态采样]]、[[batching strategy]]。


## 8. 延伸细节

### 8.1 LLaVA-RLHF：首个 VLM RLHF + factually-augmented RM

LLaVA-RLHF（Sun et al., arXiv:2309.14525, 2023）是首个用 RLHF 训练的 LMM。关键创新 **Factually Augmented RLHF**：reward model 输入除图文，还喂 image captions + ground-truth 多选题选项作事实辅助，**直接缓解 RLHF 的 reward hacking**（actor 钻 RM 漏洞堆砌物体）。提出 **MMHal-Bench** 专评多模态幻觉。结果：LLaVA-Bench 达 GPT-4 文本版 94%（前最优 87%），MMHal-Bench +60%。是"把 vision ground truth 注入 RM 降 hack"的范式起源。

### 8.2 RLHF-V：dense segment-level DPO 数据高效

RLHF-V（Yu et al., arXiv:2312.00849, CVPR 2024）用**片段级人类修正**——标注者标"哪段描述幻觉了"并给修正，RM 学片段级分数，做 **dense DPO**。结果：**1.4k 样本降幻觉 34.8%**，超 LLaVA-RLHF 用 10k 样本的效果，数据效率 ~7×。证明"段级 dense reward"比"整段 outcome reward"在治幻觉上更高效。

### 8.3 RLAIF-V：开源 AI 反馈超 GPT-4V

RLAIF-V（Yu et al., arXiv:2405.17220, 2024）走**全开源 AI 反馈**——用开源 MLLM 生成偏好对（self-feedback），不依赖人工或闭源模型。结果：**RLAIF-V 7B 降 object hallucination 80.7%、整体 33.7%**，**RLAIF-V 12B 超 GPT-4V trustworthiness**。是"用大模型当 verifier + 自我对齐"的代表，免人工标注且效果超闭源。还提出 inference-time self-feedback scaling。

### 8.4 Generative RLHF-V 与 Safe RLHF-V：谱系延伸

- **Generative RLHF-V**（Zhou et al., arXiv:2505.18531, 2025）：生成式 reward model（GRM），用 RL 训 GRM 主动捕获人类意图，再 grouped comparison 增 RL 评分精度。7 benchmark +18.1%（vs baseline RLHF +5.3%）。
- **Safe RLHF-V**（Ji et al., arXiv:2503.17682, 2025）：首个多模态安全对齐框架，BeaverTails-V 数据集（helpfulness + safety 双偏好），Beaver-Guard-V 多级护栏，constrained optimization 提安全 +34.2% / 有用 +34.3%。

谱系趋势：人类偏好 → 段级 dense → AI 反馈 → 生成式 RM → 安全对齐。降人工、提可验证、扩维度（安全）。

### 8.5 object hallucination 的评估体系

治幻觉要有评幻觉的 benchmark：

- **POPE**（arXiv:2305.10355, EMNLP 2023）：polling yes/no，判别级；
- **CHAIR**（Rohrbach et al., 2018）：caption 幻觉率（物体级 / 句级）；
- **MMHal-Bench**（LLaVA-RLHF）：12 类物体幻觉评；
- **HallusionBench**：反事实 / 对抗图测幻觉鲁棒性；
- **AMBER**：object / relation / attribute 三类幻觉综合评。

VLM RL 的 reward 与这些评估指标同源（POPE 既评估也 reward），reward 高 ↔ 评估高，是端到端治理。

### 8.6 与多模态 MoE / packing 的协同

VLM RL 的 policy 可用 [[多模态MoE路由]]（视觉专家 / 文本专家分工），rollout 用 [[多模态sequence packing]]（图文变长 batch packing + per-trajectory reward 隔离）。三者协同：MoE 提 policy 视觉 grounding 能力（降幻觉根源）、packing 提 rollout 吞吐、vision ground truth reward 治幻觉行为。是多模态 RL 训推一体的完整栈。

### 8.7 内容来源

本条整理自 LLaVA-RLHF（Sun et al., arXiv:2309.14525, 2023）、RLHF-V（Yu et al., arXiv:2312.00849, CVPR 2024）、RLAIF-V（Yu et al., arXiv:2405.17220, 2024）、POPE（Li et al., arXiv:2305.10355, EMNLP 2023）、Generative RLHF-V（arXiv:2505.18531, 2025）、Safe RLHF-V（arXiv:2503.17682, 2025）等论文（arxiv 元数据与摘要关键数字已核实）。GRPO 机制与 advantage 公式复用 [[GRPO]]；reward 三来源映射复用 [[verifier与function reward]]；reward hacking 机制复用 [[reward hacking]]；全对/全错组与 dynamic sampling 复用 [[sequence packing与动态采样]]。具体实验数字以原论文为准，本条引用的均为论文摘要已确认值（如 RLAIF-V 7B -80.7%、RLHF-V -34.8%、LLaVA-RLHF +60% 等）。

---
相关: [[VLM RL]] | [[GRPO]] | [[verifier与function reward]] | [[reward hacking]] | [[ViT与vision encoder]] | [[projector]] | [[early与late fusion]] | [[多模态sequence packing]] | [[多模态MoE路由]] | [[sequence packing与动态采样]] | [[batching strategy]] | [[Reward Model]] | [[新模型接入]]
