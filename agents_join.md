# agents_join.md — CS336 查漏补缺式合并到「系统学习ML」规则

> **任务**：把 `/Users/lei/Documents/Obsidian_database/CS336/`（源库）作为**素材源**，
> 查漏补缺地合并到 `/Users/lei/Documents/Obsidian_database/系统学习ML/systemLearning/`（目标库）。
> **不整篇迁入** CS336 笔记，不建 CS336 课程专章；只把目标库**还没有**或**不够详实**的知识点，按 [[AGENTS]] 模板补到对应主题章/小节。
> 本文件是合并任务的**唯一规则文档**，执行前必须完整读完。合并完成后保留作存档。
> 日常笔记维护仍以 [[AGENTS]] 为准；本文件只管"查漏补缺合并"这一动作。

---

## 一、合并原则

1. **查漏补缺，不整篇搬**：CS336 笔记不进库、不建 `13-CS336课程笔记/` 章。读每篇 CS336 笔记 → 提取其中知识点 → 对照目标库现有 12 章 → 只补"缺的"和"薄的"。
2. **不重复造笔记**：目标库已有对应知识点笔记的，**增补**该笔记（补内容/误区/代码/双链），不新建同名文件。
3. **CS336 是外部素材，不是库内文件**：CS336 笔记不进库，因此**不用 `[[双链]]` 指向 CS336**（库内无对应文件，会成红链）。引用时用普通文本，如"内容整理自 CS336 Lecture 5"。
4. **格式合规**：新建笔记严格按 [[AGENTS]] 第三节模板（frontmatter 双链 + 8 小节 + `相关: [[]]` 尾部）；增补笔记在原模板内补内容，不改原结构。
5. **内容质量**：遵守 [[AGENTS]] 第六节——公式必推导、代码必可跑、对比必列表、误区必列、不杜撰。
6. **源库不动**：`CS336/` 原文件只读不改不删。所有产出落在 `systemLearning/` 内。

---

## 二、源库与目标库结构对比

### 2.1 源库（CS336/）

扁平结构，24 个 .md，按 Lecture 1–19 排列，存在多版本重复。命名三套混杂：
- `CS336_LectureXX_主题详细笔记.md`（主体，中文详细版）
- `cs336_lecture_01/02/03_..._notes.md`（Lecture 1–3 早期英文标题版，旧）
- `CS336 Lecture 6 · Kernels, Triton, XLA.md` / `CS336 Lecture 8.md`（英文标题版）

特征：frontmatter 是课程信息引用块（课程/讲师/主题/日期/关联作业），不含目标库双链归属；正文是课程笔记体裁（目录 + 多节 + 速查表 + 面试问题），不套 8 小节知识点模板；内容质量高，有公式/表格/代码/误区。

### 2.2 目标库（systemLearning/）已有覆盖

12 章，强结构化。逐章已有笔记（截断列出关键）：

| 章 | 已有知识点（节选） |
|---|---|
| 01-ML深度学习基础 | Tensor、数值类型与精度、计算图、autograd机制、SGD/Momentum/Adam与AdamW、学习率调度、梯度裁剪、batch与mini-batch、epoch与step、损失函数分类、mixed precision training、loss scaling |
| 02-Transformer与LLM核心结构 | 自注意力/多头注意力/残差/layer norm/FFN、sinusoidal PE/RoPE/位置编码、decoder-only/causal mask/logits/tokenization、KV cache/autoregressive decoding/采样策略 |
| 03-PyTorch与框架工程 | nn.Module、forward与backward、state_dict、optimizer state管理、hooks机制、torch.distributed、process group、rank与world size、NCCL backend、DP/DDP/FSDP |
| 04-强化学习基础 | MDP、policy与value function、return与reward、REINFORCE/baseline/advantage/variance reduction、PPO clipped objective/KL penalty/GAE/entropy bonus |
| 05-LLM-RL对齐 | SFT/Reward Model/PPO optimization、RLHF(PPO)/DPO/IPO/KTO、reward hacking/KL explosion/policy collapse/mode collapse/exposure bias |
| 06-训推系统 | learner worker split/rollout worker/inference worker/asynchronous training、PT/PD/stale policy/weight sync、trajectory generation/sampling throughput/batching strategy |
| 07-分布式与并行计算 | DP/TP/PP/3D parallelism、all-reduce/AllGather/reduce-scatter/NCCL通信拓扑、overlap strategy/compute vs communication bottleneck |
| 08-推理系统 | KV cache management/continuous batching/dynamic batching/prefix caching、speculative decoding/beam search/parallel decoding、GPU utilization/batching tradeoff |
| 09-Ray与分布式调度 | actor model/task model/remote function/resource request、placement group（目录有/文件未见）、GPU绑定/scheduling策略、actor deadlock/resource starvation/task queue backlog |
| 10-性能分析与优化 | SM utilization/memory bandwidth/kernel launch overhead、torch profiler/nsys/nvprof/GPU memory snapshot、compute/memory/communication bottleneck、CPU-GPU pipeline stall |
| 11-显存与内存系统 | activation/gradient/optimizer state memory、gradient checkpointing、ZeRO、offloading |
| 12-训练稳定性工程 | gradient explosion and vanishing、KL divergence control、loss spike handling、warmup/reward normalization/clipping strategies/entropy regularization |

### 2.3 缺口识别（CS336 能补什么）

对照 CS336 各 Lecture 与上表，预判缺口如下（执行时按实际 Read 内容校正）：

| CS336 Lecture | 目标库现状 | 预判缺口（待补知识点） | 落点章/小节 |
|---|---|---|---|
| 1 概览与分词 | 已有 [[tokenization]] | BPE 训练/编解码、预分词、压缩率、分词陷阱（数字/多语言/glitch token）、字节级 BPE | 02/LLM结构（增补 tokenization 或新建 [[BPE]]、[[分词陷阱]]） |
| 2 PyTorch与资源核算 | 已有 03 PyTorch 核心 + 11 显存结构 | **资源核算**（参数量/激活量/显存 4P 公式）、**MFU/HFU** | 11/显存结构（新建 [[资源核算]]、[[MFU]]）或 10/GPU性能 |
| 3 架构与超参数 | 已有 02 Transformer 基础 | **LLM 架构超参**（d_model/n_heads/n_layers/d_ff 关系）、配置选择经验 | 02/Transformer基础（新建 [[LLM架构超参数]]） |
| 4 MoE | **无** | **MoE 整体**、专家路由、load balancing loss、top-k routing、capacity factor | 02（新建 MoE 小节：[[MoE]]、[[专家路由]]、[[load balancing loss]]） |
| 5 GPU | 已有 10/GPU性能（SM/bandwidth/kernel launch） | **GPU 硬件结构**（SM/CUDA core/Tensor Core）、**内存层级**（SRAM/L1/L2/HBM）、**warp/SIMT**、**Roofline 模型**、**算子融合** | 10/GPU性能 + 11/显存结构（新建 [[GPU硬件结构]]、[[GPU内存层级]]、[[warp与SIMT]]、[[Roofline模型]]、[[算子融合]]） |
| 6 Kernel与Triton | 基本无 | **FlashAttention**、**Triton 编程**、kernel 编写、kernel fusion 实操 | 10（新建 [[FlashAttention]]、[[Triton]]、[[kernel编写]] 小节） |
| 7 并行训练基础 | 已有 07/并行范式 + 03 DP/DDP/FSDP | 并行范式总览梳理（可能已覆盖，看是否要增补 [[并行范式总览]]） | 07/并行范式（增补，按需） |
| 8 通信/collectives | 已有 07/通信机制 | collective 通信总览、通信带宽/时延模型（按需增补） | 07/通信机制（增补，按需） |
| 9 缩放定律 | **无** | **Scaling Laws**、**Chinchilla**、计算最优缩放、IsoFLOP | **新建 13-缩放定律与评估**（[[缩放定律]]、[[Chinchilla scaling laws]]、[[计算最优缩放]]） |
| 10 推理 | 已有 08 推理系统 | 推理整体流程、prefill vs decode 两阶段分析（按需增补） | 08/推理优化（增补 [[prefill与decode]]） |
| 11 缩放定律2 | **无** | 同 9，深化（训练 compute 公式、token/param 配比） | 13-缩放定律与评估 |
| 12 评估 | **无** | **LLM 评估方法**、perplexity、benchmark（MMLU/HellaSwag 等）、eval 陷阱 | 13-缩放定律与评估（[[LLM评估方法]]、[[perplexity]]、[[benchmark综述]]） |
| 13 数据来源 | **无** | **数据来源**、Common Crawl 清洗、数据去重（MinHash）、质量过滤 | **新建 14-数据工程**（[[数据来源]]、[[Common Crawl清洗]]、[[数据去重]]、[[数据质量过滤]]） |
| 14 数据混合 | **无** | **数据混合**、domain mixing、配比实验 | 14-数据工程（[[数据混合]]） |
| 15 对齐SFT/RLHF | 已有 05/RLHF流程 + 对齐算法 | 对齐三阶段整体流程、RLHF 全景（按需增补 [[对齐流程总览]]） | 05/RLHF流程（增补） |
| 16 对齐RL/策略梯度 | 已有 04 Policy Gradient + 05 对齐 | **GRPO**、**RLVR**（reinforcement learning from verifiable rewards） | 05/对齐算法（新建 [[GRPO]]、[[RLVR]]） |
| 17 对齐RL2 | 已有 05 对齐 | **RLAIF**、更深层 reward hacking 防御 | 05/对齐算法（新建 [[RLAIF]]） |
| 18 客座Qwen | 部分（架构相关散落） | **Qwen 架构**（如有独到设计） | 02/LLM结构（新建 [[Qwen架构]]，仅当 CS336 笔记有独到干货） |
| 19 客座Llama | 同上 | **Llama 架构**（RoPE/SwiGLU/RMSNorm 等已部分有，整合） | 02/LLM结构（新建 [[Llama架构]]，整合已有知识点） |

> 上表是**预判蓝图**，不是定论。执行时每篇 CS336 笔记先完整 Read，再对照目标库实际内容校正缺口，避免重复造笔记。

---

## 三、缺口识别与处理流程（逐 Lecture 执行）

对 CS336 的每个 Lecture（按 1→19 顺序，旧版/英文标题版按第四节去重选最终版 Read）：

### 3.1 Read + 提取

1. Read 该 Lecture 保留版全文。
2. 提取其中的**知识点清单**（每个独立概念/公式/方法 = 一个知识点）。
3. grep 该 CS336 笔记里的 `【user】` 批注，记录待处理（见第八节）。

### 3.2 对照目标库

对每个提取出的知识点：

1. 在目标库用 Grep / Glob 搜同名/近义笔记。
2. 判定归属：
   - **目标库无对应笔记** → 标记为「新建」。
   - **目标库有对应笔记但内容薄/缺 CS336 的干货** → 标记为「增补」。
   - **目标库已有且足够详实** → 标记为「跳过」，不动。

### 3.3 处理

- **新建**：按 [[AGENTS]] 第三节模板，在对应主题章/小节文件夹下建新笔记。路径 `systemLearning/{章}/{小节}/{知识点}.md`。文件夹不存在则先建。
- **增补**：在已有笔记内补内容——优先补到「3. 核心概念详解」「4. 数学原理」「7. 常见误区」「8. 延伸细节」等对应小节；CS336 提供的独有内容在「8. 延伸细节」末尾加一段，注明"本节内容整理自 CS336 Lecture X"（**纯文本，不用 `[[]]`**）。
- **跳过**：不动文件，仅在执行日志里记一笔"XX 已存在且详实，跳过"。

### 3.4 索引同步

每新建一个笔记：
1. 在 `整体目录.md` 对应小节条目处，把纯文本名替换为 `[[笔记名]]` 双链（若该小节尚无对应条目，新增小节标题 + 双链条目）。
2. 若新建笔记属于目标库**尚无的主题章**（如 13-缩放定律与评估、14-数据工程），在 `整体目录.md` 末尾新增该章标题与小节结构。
3. 在新建/增补笔记的「6. 与其他知识点的关系」小节，与相关已有笔记互补双链。

---

## 四、去重规则（选 Read 哪个版本）

源库同一 Lecture 多版本时，只 Read **保留版**作为素材，其余版本忽略（不 Read、不迁、不动源库）：

| Lecture | 候选 | Read 保留版 | 忽略 | 备注 |
|---|---|---|---|---|
| 1 | 详细笔记版(6/14) / 旧版(5/10) | 详细笔记版 | 旧版 | — |
| 2 | 详细笔记版(6/14) / 旧版(5/11) | 详细笔记版 | 旧版 | — |
| 3 | 详细笔记版(6/14) / 旧版(5/12) | 详细笔记版 | 旧版 | — |
| 6 | 详细笔记版(6/14) / 英文标题版(5/23) | 详细笔记版 | 英文标题版 | 执行前 Read 两者对比，若英文标题版有独有干货，把该段并入详细笔记版素材再处理 |
| 8 | 英文标题版(5/27，唯一) | 此版 | — | 无重复 |

其余 Lecture（4、5、7、9–19）单一详细笔记版，直接 Read。

---

## 五、新建主题章

缺口对照发现，CS336 的 scaling laws/评估（L9/11/12）和数据工程（L13/14）在目标库**整章缺失**，需真正新建主题章（不是占位）：

### 5.1 `13-缩放定律与评估/`

```
13-缩放定律与评估/
├── 缩放定律/        ← [[缩放定律]]、[[Chinchilla scaling laws]]、[[计算最优缩放]]、[[IsoFLOP]]
└── 评估方法/        ← [[LLM评估方法]]、[[perplexity]]、[[benchmark综述]]
```

### 5.2 `14-数据工程/`

```
14-数据工程/
├── 数据来源与清洗/  ← [[数据来源]]、[[Common Crawl清洗]]、[[数据质量过滤]]
├── 数据去重/        ← [[数据去重]]、[[MinHash]]
└── 数据混合/        ← [[数据混合]]、[[domain mixing]]
```

> 文件夹在首次建该章某笔记时一并创建，不预建空文件夹。
> 建后更新 `整体目录.md`：在十二章之后追加"十三、缩放定律与评估""十四、数据工程"两章及小节双链。
> 现有目标库的"十二、训练稳定性工程"之后顺序追加，章号顺延。

### 5.3 MoE 小节（不新建章，并入 02 章）

L4 MoE 在 `02-Transformer与LLM核心结构/` 下新建小节文件夹 `MoE/`，放 [[MoE]]、[[专家路由]]、[[load balancing loss]]。`整体目录.md` 第二章下新增"8. MoE"小节条目（原小节号顺延）。

---

## 六、格式规范

### 6.1 新建笔记

严格按 [[AGENTS]] 第三节模板：frontmatter 双链（所属章节/所属模块）+ 8 小节（一句话定义/为什么需要/核心概念/数学原理/代码示例/与其他知识点关系/常见误区/延伸细节）+ `---` + `相关: [[]]` 尾部。内容自包含、详实、不外引。

### 6.2 增补已有笔记

- 不改原文件首行标题与 frontmatter。
- 在原 8 小节框架内补内容到对应小节；CS336 独有干货放「8. 延伸细节」末尾，注明 `本节内容整理自 CS336 Lecture X`（纯文本）。
- 补双链：在「6. 与其他知识点的关系」补相关新笔记的双链。

### 6.3 引用 CS336 的方式

- **不用 `[[CS336...]]` 双链**（库内无 CS336 文件，会红链）。
- 用纯文本：`内容整理自 CS336 Lecture 5（GPU）` 或 `> 参考：CS336 Lecture 9 — Scaling Laws`。
- 若该知识点在目标库已有笔记，优先 `[[已有笔记名]]` 双链，CS336 仅作素材来源不显式标注。

---

## 七、索引与互链

### 7.1 `整体目录.md`

- 新建笔记对应小节条目：把纯文本名改 `[[笔记名]]` 双链；小节不存在则新增小节标题。
- 新建主题章（13/14）：在十二章后追加章标题 + 小节 + 双链。
- MoE 小节：第二章下新增小节条目。
- 完成后 `整体目录.md` 仍是全库唯一主索引，无孤岛。

### 7.2 双链互链

- 新建笔记之间、新建笔记与已有笔记之间，按「6. 与其他知识点的关系」互补双链。
- 例：[[MoE]] 的下游指向 [[专家路由]]、[[load balancing loss]]；[[FlashAttention]] 上游指向 [[KV cache]]、[[自注意力]]，下游指向 [[算子融合]]。
- 不造红链：双链目标必须是已存在或将同批建成的笔记。

### 7.3 `QA.md`

- 处理 CS336 笔记里 `【user】` 批注的答案，若落点是新建/增补的目标库笔记，在 QA.md 对应章分组下登记。
- 新建主题章 13/14 在 QA.md 新增对应分组标题。

---

## 八、`【user】` 批注处理

CS336 源笔记不进库，但其 `【user】` 批注仍要回答（用户在源笔记里提的问题，合并时一并消化）：

1. **扫描**：对每个 Lecture 保留版，grep `【user】`，记录所有批注。
2. **分类**：
   - 批注问题对应的知识点**正在被补进目标库** → 答案写进目标库对应笔记（行内型用 `> [!note] 解答：…` callout 落在该知识点相关小节），QA.md 登记。
   - 批注问题对应的知识点**目标库已有且不增补** → 在目标库已有笔记里补一条 `> [!note] 解答：…` callout 回答，QA.md 登记。
   - 批注问题**不在本次合并范围**（如纯课程作业细节）→ 在 QA.md 的"十三、CS336 课程笔记"分组下记一条"待处理/暂悬"，标注来源 Lecture，不强行落盘。
3. **源笔记不改**：CS336 源文件保持原样，`【user】` 行不删不改（因为源库不进版本管理，改了也没意义；答案只落目标库）。
4. **登记**：每答一条，在 `QA.md` 新增分组"十三、CS336 批注"下追加 `- [[答案所在笔记名]] — {问题精炼}（源自 CS336 Lecture X）`。
5. **校验**：合并完成后，grep `【user】` 范围 = `systemLearning/`，应**无命中**（因 CS336 源不在该范围）；CS336 源里的批注以 QA.md 登记清单为准报告。

---

## 九、执行顺序

1. **去重选定 Read 版本**：按第四节表，确定每 Lecture 的保留版。
2. **逐 Lecture 处理**（按 1→19 顺序）：
   - Read 保留版全文。
   - 提取知识点清单 + grep `【user】`。
   - 按 3.2 对照目标库，标记新建/增补/跳过。
   - 执行新建/增补（3.3），每建一个同步索引（3.4）。
   - 处理该 Lecture 的 `【user】` 批注（第八节）。
3. **新建主题章**：首次遇到缩放定律/评估/数据工程类笔记时，建 `13-缩放定律与评估/` 或 `14-数据工程/` 文件夹与 `整体目录.md` 章条目。
4. **MoE 小节**：L4 时在 02 章建 `MoE/` 小节。
5. **互联回填**：所有新建笔记完成后，统一检查「6. 与其他知识点的关系」双链是否齐全，补漏。
6. **QA.md 登记**：所有 CS336 批注处理完毕后，确认 QA.md「十三、CS336 批注」分组齐全。
7. **完成校验**（第十节）。

> 建议每处理完一个 Lecture，向用户报告该 Lecture 补了哪些笔记（新建/增补/跳过清单），再进下一个，便于用户即时校正缺口判断。

---

## 十、完成校验

1. **新建笔记清单**：列出本次新建的所有笔记路径。
2. **增补笔记清单**：列出本次增补的所有已有笔记路径 + 增补小节。
3. **索引核对**：`整体目录.md` 新增条目/小节/章均可点击且不红链；13/14 章结构齐全。
4. **双链核对**：随机抽 3 篇新建笔记，确认 frontmatter 双链 + 「与其他知识点关系」双向链接齐全 + 无红链。
5. **批注核对**：QA.md「十三、CS336 批注」分组条目数 = 本次处理的 CS336 批注数；`grep '【user】' systemLearning/` 无命中。
6. **源库未动**：确认 `CS336/` 原文件未被修改（mtime 不变）。
7. **无重复**：新建笔记名在目标库无同名重复。

把校验结果以清单报告用户。

---

## 十一、禁止事项

- ❌ 整篇复制 CS336 笔记进库 / 建 CS336 课程专章。
- ❌ 用 `[[CS336...]]` 双链指向库内不存在的 CS336 文件（造红链）。
- ❌ 目标库已有对应笔记还新建同名/同主题文件（重复造）。
- ❌ 不对照目标库实际内容就凭预判表新建（预判表只是蓝图，必须 Read 后校正）。
- ❌ 新建笔记不套 [[AGENTS]] 第三节模板 / 内容空洞。
- ❌ 新建主题章不更新 `整体目录.md` / 不补双链，成孤岛。
- ❌ 修改或删除 `CS336/` 源文件。
- ❌ 跳过 `【user】` 批注不登记到 QA.md。
- ❌ 强行补已足够详实的知识点（该跳过就跳过，不过度膨胀）。

---

## 十二、备注

- **Lecture 18/19（客座 Qwen/Llama）**：仅当 CS336 笔记里有目标库没有的独到架构干货时才新建 [[Qwen架构]]/[[Llama架构]]；若只是已有知识点（RoPE/SwiGLU/RMSNorm）的复述，跳过，不强行造笔记。
- **Lecture 7/8（并行/通信）**：目标库 07 章已较完整，大概率多数知识点标"跳过"或轻量增补，执行时据实判定。
- **本规则文档与 [[AGENTS]] 冲突时**：合并任务以本文件为准（如 CS336 引用方式、批注源处理），其余维护规则仍以 [[AGENTS]] 为准。

---

相关: [[AGENTS]]、[[整体目录]]、[[QA]]
