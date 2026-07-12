# R3 rollout routing replay

> **所属章节**: [[Rollout系统]]
> **所属模块**: [[06-训推系统]]
> **别名**: R3 / Rollout Routing Replay / Router Replay R3 / 训推路由对齐回放
> **难度**: 中高级（需懂 [[MoE路由|Mixture-of-Experts 路由]] + RLHF rollout/训练双引擎架构 + PPO 重要性采样）
> **来源**: Ma et al., *Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers*, arXiv:2510.11370 (2025-10); 工程实现见 NeMo-RL / verl + vLLM·SGLang + Megatron。

## 1. 一句话定义

**R3（Rollout Routing Replay，rollout 路由回放）** 是一种**保证 MoE 模型 RL 训练"训推一致性"的方法**：在 **rollout（推理引擎，如 vLLM/SGLang）** 生成轨迹时，记录 MoE router 对每个 token 选了哪些专家（routing mask）；在 **训练引擎（如 Megatron）** 前向时，**不重新跑 router 选专家，而是直接回放（replay）推理时记录的那组专家选择**，从而让训练和推理走完全相同的专家子网络，从根因上消除 MoE 的"训推不一致（Training-Inference Mismatch, TIM）"，稳定 RL 训练、防止崩溃。

> [!note] 纠正之前对 "R3" 的错误解释
> "把 **rollout** 阶段的 **routing**（专家路由）决定 **replay**（回放）到训练阶段"。命名里的 "routing" 指 **MoE 的 expert routing**，不是"请求路由分发到 worker"。用户的批注"这是保证训推一致性的手段"是对的，本笔记据此全文重写。

> [!note] 关于命名 R1/R2/R3
> 文献/工程里把"路由一致性"分三档：
> - **R1**：朴素做法，训练和推理各跑各的 router，不做对齐（baseline，TIM 最严重）。
> - **R2（Router Replay, training-side）**：只在**训练引擎内部**记录路由并在后续 micro-batch 回放，确定性强但**不解决推理-训练的 gap**，TIM 仍存。
> - **R3（Rollout Routing Replay）**：在**推理引擎**记录路由、跨引擎回放到训练引擎，**直击训推 gap 的根因**，是本笔记主题。
> 实测中 R2 对 TIM 几乎无改善，R3（修好 bug 后）能把 TIM 压到接近 dense 模型水平。


> [!note] 解答：项目里怎么落地 R3——记录什么、在哪记、怎么记
> 先纠正一个易混点：**R3 的"记录"发生在推理端（rollout/vLLM·SGLang），不是训练端**。训练端（Megatron）是"回放/使用"那方——它消费推理记下来的路由，不自己记。只有 **R2** 才是在训练端记录（且 R2 对 TIM 无效，见上文）。所以"记录训练端的什么"这个问法本身把方向反了：要记的是**推理端**的路由，训练端只是读。
>
> **1. 记录什么（记录的对象）**
> 记的是 **MoE router 对每个 token、每个 MoE 层，TopK 选了哪几个专家的"编号（expert indices）"**，不是 logits、不是 gating 权重、不是激活值。就是离散的整数索引——"这个 token 在第 3 层选了专家 {2,7}"。
> - 不记 router logits $s$：那是连续浮点，体积大且训练端要重算（保梯度）。
> - 不记 gating 权重 $g$：训练端用自己算的 `softmax(s_train)`（保梯度流），不能固化推理的 $g$。
> - 只记**离散选择掩码 $I_{\text{infer}}$**（等价于 TopK 专家索引列表），体积小、是常数、不回传梯度。
>
> **2. 张量契约（记成什么形状）**
> ```
> rollout_routed_experts.shape == (len(tokens) - 1, num_layers, moe_router_topk)
> dtype: int32
> ```
> - 第一维 `len(tokens)-1`：生成序列长度（去掉最后一个 token，它没有"被路由到下一步"的语义）。
> - 第二维 `num_layers`：**全部 transformer 层**（含 dense 层，dense 层对应位用全 `-1` sentinel 占位，因为 dense 层没有 router）。
> - 第三维 `moe_router_topk`：每 token 每层选 $K$ 个专家的编号。例如 Qwen3-30B-A3B 是 `(seq-1, 48, 8)`（48 层、topk=8）。
> 这个 int32 张量就是"路由记录"，随轨迹一起从推理端流到训练端。
>
> **3. 怎么记录（推理端，vLLM/SGLang）**
> - **SGLang**（端到端原生支持）：启动加 `--enable-return-routed-experts`，引擎生成时把路由张量序列化成 base64 int32 buffer，放在响应的 `meta_info.routed_experts` 字段。rollout worker 直接取用。
> - **vLLM**（需 patch）：启动加 `--enable-return-routed-experts` 开引擎内部捕获（`RoutedExpertsCapturer`，在 `FusedMoE` 首次 forward 时绑定）。但 HTTP `/inference/v1/generate` 默认不回显路由字段，要 patch 暴露两段：`prompt_routed_experts`（prompt 阶段路由，response 顶层）+ `choices[].routed_experts`（decode 阶段路由，每个 choice 内）。client 端把两段 `np.concatenate` 成完整张量。
> - **关键坑**（见 §7 误区 5）：①捕获时机——vLLM 里 capturer 在 KV-cache init 时创建，但 `FusedMoE` 在 init 时 bind capture 会错过，要在首次 forward 延迟绑定；②EPLB——开专家并行负载均衡会把逻辑 ID 映成物理 ID，必须在 EPLB 映射**之前**捕获逻辑 ID；③dense 层偏移——用 router 的全局 `layer_number` 映射，别用局部 `i+offset`。
>
> **4. 怎么传递（rollout→learner 的数据通路）**
> 路由张量随轨迹走标准 RLHF 数据通路，但每一步都要**保 token 身份不变**：
> ```
> vLLM/SGLang 生成 → sample.rollout_routed_experts（rollout worker 塞入）
>   → TransferQueue（跨 worker 传输，token 对齐不能错位）
>   → packing（多条轨迹拼 batch，路由张量要按 token 顺序拼）
>   → context-parallel slicing（CP 切序列，路由张量要同步切）
>   → Megatron actor 前向
> ```
> 任一步把 token 和它的路由记录错位，R3 就失效（甚至比不开更糟）。NeMo-RL 有专门 checker 验证这条链路。
>
> **5. 怎么使用（训练端，Megatron）**
> Megatron 的 `TopKRouter` 旁挂一个 `RouterReplay` 模块。前向到某 MoE 层时：
> - 不调 `TopKRouter` 算 `I_train`；
> - 从 `rollout_routed_experts` 张量取**当前 token、当前层**对应的那组专家索引，构造掩码 $I_{\text{infer}}$；
> - 用 $I_{\text{infer}}$ 选专家，但 gating 仍用训练侧 logits 算：$g_{\text{replay}}=\text{softmax}(s_{\text{train}})\odot I_{\text{infer}}$（保梯度流，见 §3.3）。
> - 缺失的路由条目（vLLM 偶尔返回不全）用全 `-1` sentinel，对这些 token 回退到 Megatron 自己的 router（局部 fallback，不整批禁用 R3），并记 `r3/routed_experts_fallback_token_route_fraction`（应≈0）。
> - `RouterReplay` 在 **prev-logprob 阶段和 train 阶段都要装**（两处前向都用同一组路由）。
>
> **6. 落地步骤清单**
> 1. 确认模型是 MoE（dense 不需要 R3）。
> 2. 选推理引擎：SGLang 优先（原生支持），vLLM 需打 patch。
> 3. 推理引擎启动加 `--enable-return-routed-experts`；vLLM 还要 `--no-async-scheduling`（R3 与 async 调度不兼容）。
> 4. rollout worker 从响应解析路由张量，按 `(seq-1, num_layers, topk)` int32 存进 `sample.rollout_routed_experts`（vLLM 要 concat prompt 段 + decode 段）。
> 5. 数据通路（TransferQueue/packing/CP slicing）加 token 对齐校验。
> 6. Megatron 侧接 `RouterReplay`，前向时安装 $I_{\text{infer}}$，gating 用 `softmax(s_train)`。
> 7. 加监控：`r3/routed_experts_fallback_token_route_fraction`（应≈0）、训推 `pearson`/`max_abs_diff`/`KL`（应接近 dense baseline）。
> 8. 配对跑 R3-on/R3-off，验证 TIM 下降、训练稳定。
>
> **7. 一句话总结**
> 在**推理端**记"每 token 每 MoE 层 TopK 选了哪几个专家"（int32 张量 `(seq-1, n_layers, topk)`），随轨迹传到**训练端**，训练前向时**跳过自己的 TopK、直接用记下来的那组专家**，但 gating 仍用训练 logits 保梯度。开 `--enable-return-routed-experts`，SGLang 原生、vLLM 需 patch + 关 async。
>
> 下面 §3.5 的代码示例有最小 MoE 层的 `_replay_mask` 注入演示，§8.1 列了 verl/NeMo-RL/SGLang/vLLM 的具体 PR 与 flag。

## 2. 为什么需要它（动机与背景）

### 2.1 MoE 的 router 是训推不一致的放大器

MoE（[[MoE路由|Mixture-of-Experts]]）模型里，每个 token 经 **router** 算出一组 logits $s$，取 **TopK** 选 $K$ 个专家激活。问题在于：**同一个 token、同一个权重，训练引擎和推理引擎可能选出不相同的专家**。原因是：

1. **两套实现不同**：训练用 Megatron/DeepSpeed 的 router，推理用 vLLM/SGLang 的 router，浮点精度、累积顺序、归一化细节有差异，router logits $s$ 的微小数值差经过 TopK 这种**不连续的 argmax-like 操作**会被放大成"选了不同专家"。
2. **即便同一套实现、多次前向也可能选不同专家**：浮点非结合性 + 并行归约顺序非确定 → 同输入重复跑 router 结果不一致。
3. **MoE 比 dense 严重得多**：dense 模型训推只有 logits 的数值小扰动；MoE 多了 router 这道"硬选择"开关，一次选错专家 → 该 token 的激活完全不同 → logits 差距被放大。论文实测 **Qwen3-30B-A3B（MoE）的训推 KL ≈ 1.5×10⁻³，而 Qwen3-8B（dense）≈ 6.4×10⁻⁴**，MoE 的训推不一致是 dense 的 ~2.4 倍。约 **10% 的 router 在训练 vs 推理选了不同专家**。

### 2.2 训推不一致在 RL 里是致命的

> [!note] 已展开为独立笔记
> "什么时候会有训推不一致 / 怎么衡量（pearson）/ 会有什么影响"已展开为 [[训推不一致]]，涵盖 TIM 来源分层（精度/算子/融合/MoE 路由）、指标体系（Pearson+max_abs_diff+KL+$F(\tau)$）、影响四级（噪声→有偏→KL 爆炸→崩溃）、与 staleness 区分、消除手段分层。本节给出 R3 视角的摘要。

在 PPO/GRPO 等 RL 训练里，rollout 用推理引擎生成轨迹并记下 `logprob_old`；learner 用训练引擎重算 `logprob_new`，重要性采样比 $\rho = \pi_{\theta_{\text{new}}}/\pi_{\theta_{\text{old}}}$ 依赖两套 logprob **来自同一分布**。一旦训推 router 选不同专家，`logprob_old`（推理算的）和 `logprob_new`（训练算的）就对不上，$\rho$ 失真 → 梯度有偏 → 训练发散/崩溃。

论文观察到：**不带 R3 的 MoE RL 训练经常崩溃**，崩溃前伴随训推 KL 与"极端 token 比例"$F(\tau{=}2)$ 持续攀升——例如 SFT+GRPO 单 mini-step 设定下，60 步后 $F(\tau{=}2)>0.1$（10% 的 token 训推概率差超 2 倍），随后崩。而带 R3 的训练 $F(\tau{=}2)$ 始终 $<10^{-4}$。

### 2.3 现有缓解手段的局限

| 手段 | 思路 | 局限 |
|---|---|---|
| **rollout correction / clip**（如 GSPO、FSPO） | 检测训推概率差超阈的样本，裁断/丢弃 | 治标不治本；TIM 涨得比丢弃快时**有效样本被丢光、训练停摆** |
| **TIS**（Training-Inference Sync） | 算法层强制对齐 | 仍受数值噪声限制，效果不及 R3 |
| **R2**（训练侧回放） | 训练引擎内部记录并回放路由 | 不跨引擎，**不解决推理→训练的 gap** |
| **R3**（本篇） | 推理记录→训练回放，**根因对齐** | 需推理引擎支持导出路由；工程实现有坑（见 §7） |

R3 的定位：**不丢弃数据、不在算法层打补丁，而是从工程上让训练和推理走同一条专家子网络**，把 TIM 压到接近 dense 模型水平。

## 3. 核心概念详解

### 3.1 MoE router 的标准前向

设某 MoE 层输入为 $x$，router 权重 $W_r$：

$$
s = x W_r \in \mathbb{R}^{n_{\text{experts}}}, \qquad I = \text{TopKMask}(s, K)
$$

- $s$：router logits；
- $I$：TopK 的 0/1 掩码，标记选中的 $K$ 个专家；
- 门控权重 $g = \text{softmax}(s) \odot I$（只对选中专家做 softmax）；
- 输出 $y = \sum_{i \in I} g_i \cdot E_i(x)$，$E_i$ 是第 $i$ 个专家网络。

**关键**：$I$ 是离散选择，一旦训练和推理算出的 $I$ 不同，激活的专家子网络就不同 → 输出 $y$ 差距大。

### 3.2 训推不一致（TIM）的来源

$$
I_{\text{infer}} = \text{TopKMask}(s_{\text{infer}}), \quad I_{\text{train}} = \text{TopKMask}(s_{\text{train}})
$$

即便 $x$、$W_r$ 完全相同，$s_{\text{infer}} \neq s_{\text{train}}$（数值实现差异），且 TopK 对相近的 logit 极敏感（第 K 名和第 K+1 名差一点就换专家），导致 $I_{\text{infer}} \neq I_{\text{train}}$。这就是 TIM 的根因。

### 3.3 R3 的核心思想：replay 推理的 mask

R3 的做法一句话：**训练前向时，用 $I_{\text{infer}}$ 替代重新计算 $I_{\text{train}}$，但 softmax 仍用训练侧的 logits 算（保留梯度）**。

设推理阶段 router 输入 $x_{\text{infer}}$，算出 $s_{\text{infer}} = x_{\text{infer}} W_r$，记录 mask：

$$
I_{\text{infer}} = \text{TopKMask}(s_{\text{infer}}, K)
$$

训练阶段（用同样的 token，但走训练引擎），router 输入 $x_{\text{train}}$，算训练 logits $s_{\text{train}} = x_{\text{train}} W_r$。**R3 不用 $\text{TopKMask}(s_{\text{train}})$，而是直接复用 $I_{\text{infer}}$**，只把 softmax 套在 $s_{\text{train}}$ 上以保梯度：

$$
g_{\text{replay}} = \text{softmax}(s_{\text{train}}) \odot I_{\text{infer}}
$$

输出：

$$
y_{\text{replay}} = \sum_{i \in I_{\text{infer}}} g_{\text{replay},i} \cdot E_i(x_{\text{train}})
$$

两个目的（论文原文）：

- **(a) 对齐训推**：训练激活的专家集合 = 推理激活的专家集合，消除专家选择 mismatch。
- **(b) 保留梯度流**：softmax 仍在 $s_{\text{train}}$ 上算，梯度能回传到 router 权重 $W_r$，且**只优化推理时真正用到的那批专家**，避免优化"推理没走、训练却走了"的专家引入梯度噪声。

> [!tip] 为什么不直接把推理的 $g$ 也回放（连 softmax 一起固定）？
> 那样 router 权重 $W_r$ 就拿不到梯度，无法学习路由策略。R3 只固定"选谁"（mask，离散、无梯度），不固定"给多少权重"（softmax 值，连续、有梯度），在一致性和可学习性之间取平衡。

### 3.4 数据通路：路由张量怎么从推理流到训练

R3 要求推理引擎在生成时**额外导出每个 token、每个 MoE 层选了哪些专家**，作为一条"侧带数据"随轨迹一起送进训练。契约（NeMo-RL / slime-vllm 实现）：

```
rollout_routed_experts.shape == (len(tokens) - 1, num_layers, moe_router_topk)
dtype: int32
```

- 第一维 `len(tokens)-1`：去掉最后一个 token（它没有"下一步被路由"的语义），对齐生成序列长度。
- 第二维 `num_layers`：**全部 transformer 层**（含 dense 层，dense 层对应位用 sentinel 占位，见 §7.2）。
- 第三维 `moe_router_topk`：每 token 每层选 $K$ 个专家的编号。

流水：

```
vLLM/SGLang 生成 ──enable_return_routed_experts──→ 输出 routed_experts 张量
        │
        ▼
rollout worker 把 routed_experts 塞进 sample.rollout_routed_experts
        │  (经 TransferQueue / packing / CP slicing，token 身份须保持)
        ▼
Megatron actor 前向：每层 MoE 不调 TopKRouter，改用 RouterReplay 安装 I_infer
        │
        ▼
训练 forward 用 I_infer 选专家 + softmax(s_train) 算门控 → 保梯度
```

- SGLang：`--enable-return-routed-experts`，路由以 base64 int32 buffer 放在 `meta_info.routed_experts`，端到端原生支持。
- vLLM：`--enable-return-routed-experts` 开引擎内部捕获，但 HTTP `/inference/v1/generate` 默认不回显路由字段，需 patch（`prompt_routed_experts` + `choices[].routed_experts` 两段，client 端 concatenate）。
- Megatron：`TopKRouter` 旁挂一个 `RouterReplay`，前向时把 `I_infer` 安装进去，供 prev-logprob 阶段和 train 阶段共用。

### 3.5 与 PPO/RLHF 训练循环的关系

R3 不改 PPO/GRPO 算法本身，只改"训推前向怎么算 logprob"。在标准 RLHF 一轮里：

1. **rollout**：推理引擎生成 response，记 `logprob_old`、reward，**并额外记 `routed_experts`**。
2. **learner**：训练引擎对同条轨迹重算 `logprob_new`，**前向时 replay `routed_experts`**，保证 `logprob_new` 和 `logprob_old` 走同一专家子网络。
3. **PPO loss**：$\rho = \exp(\text{logp}_{\text{new}} - \text{logp}_{\text{old}})$，因训推对齐，$\rho$ 不再被路由 mismatch 污染，clip 频率降、训练稳。

> [!note] R3 解决的是"训推前向不一致"，不是"off-policy staleness"
> 注意区分两类不一致：
> - **训推实现不一致（TIM）**：同一权重、同一 token，训练引擎 vs 推理引擎算出不同 logprob → R3 解决这个。
> - **off-policy staleness**：rollout 用旧权重 $\theta_{\text{old}}$、learner 用新权重 $\theta_{\text{new}}$ → 由重要性采样 $\rho$ 修正（见 [[stale policy problem]]、[[PPO clipped objective]]），**R3 不解决也不替代它**。
> 两者正交：R3 保证"两套引擎在同权重下算出一样的 logprob"，$\rho$ 修正"不同权重下的分布偏移"。R3 让 $\rho$ 只反映权重更新、不被路由噪声污染。

## 4. 数学原理 / 公式

### 4.1 标准 MoE 前向（无 R3）

单层、单 token，router 输入 $x$：

$$
s = x W_r,\quad I = \text{TopKMask}(s, K),\quad g = \text{softmax}(s)\odot I,\quad y = \sum_{i\in I} g_i E_i(x)
$$

### 4.2 训推 mask 不一致

$$
I_{\text{infer}} = \text{TopKMask}(s_{\text{infer}}),\quad I_{\text{train}} = \text{TopKMask}(s_{\text{train}}),\quad \Pr[I_{\text{infer}} \neq I_{\text{train}}] \approx 10\%
$$

（论文实测 Qwen3-30B-A3B，约 10% 的 router 训推选不同专家。）

### 4.3 R3 回放

$$
g_{\text{replay}} = \text{softmax}(s_{\text{train}}) \odot I_{\text{infer}},\quad y_{\text{replay}} = \sum_{i\in I_{\text{infer}}} g_{\text{replay},i}\, E_i(x_{\text{train}})
$$

梯度流：$\partial y_{\text{replay}}/\partial W_r$ 经 $\text{softmax}(s_{\text{train}})$ 回传（$I_{\text{infer}}$ 是常数掩码，不回传）；只对 $i\in I_{\text{infer}}$ 的专家 $E_i$ 有梯度。

### 4.4 训推 KL 的降低

用 logits $p$ 的估计 KL（论文用 token 级概率比的 KL 估计）：

$$
\text{KL}_{\text{train-infer}} = \mathbb{E}\left[\text{KL}\big(p_{\text{train}}(\cdot\mid x)\,\|\,p_{\text{infer}}(\cdot\mid x)\big)\right]
$$

实测（Qwen3-30B-A3B，SGLang 推理 + Megatron 训练）：

| 设置 | KL (×10⁻³) | 说明 |
|---|---|---|
| dense baseline (Qwen3-8B) | 0.64 | 无 router，TIM 下界 |
| MoE，无 R3 | 1.535 | router 放大 TIM |
| **MoE + R3** | **0.75** | 降到接近 dense 水平 |

且"极端 token 比例" $F(\tau{=}2)$（训推概率比超 $\tau$ 的 token 占比）从 $>0.1$ 降到 $<10^{-4}$，**降约一个数量级**。

### 4.5 与 PPO 重要性采样的衔接

PPO 目标（见 [[PPO clipped objective]]）：

$$
\mathcal{L}^{\text{PPO}} = \mathbb{E}\big[\min(\rho A,\, \text{clip}(\rho,1{-}\epsilon,1{+}\epsilon)A)\big],\quad \rho = \frac{\pi_{\theta_{\text{new}}}(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}
$$

无 R3 时 $\pi_{\theta_{\text{old}}}$（推理算）与 $\pi_{\theta_{\text{new}}}$（训练算）即使 $\theta$ 未变也因路由 mismatch 不等 → $\rho$ 基线漂移、方差大、clip 频发。R3 把这个"假性 $\rho$ 偏移"消掉，让 $\rho$ 只反映真正的权重更新。

## 5. 代码示例

下面用一个最小 MoE 层演示 R3 的 mask 回放逻辑（教学用，非生产代码）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """最小 MoE：1 个 router + n 个专家，TopK 选专家。"""
    def __init__(self, d, n_experts, topk):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_experts)])
        self.topk = topk
        self._replay_mask = None   # R3 注入的 I_infer，None 时正常跑 router

    def set_replay_mask(self, mask):
        """R3：注入推理阶段记录的 TopK 掩码 I_infer (B, n_experts) 0/1。"""
        self._replay_mask = mask

    def forward(self, x):                       # x: (B, d)
        s = self.router(x)                      # router logits (B, n_experts)
        if self._replay_mask is None:
            # 正常：训练自己跑 TopK
            topk_val, topk_idx = s.topk(self.topk, dim=-1)
            mask = torch.zeros_like(s).scatter_(-1, topk_idx, 1.0)
        else:
            # R3：直接用推理记录的 mask，不重新选专家
            mask = self._replay_mask.float()
        gating = F.softmax(s, dim=-1) * mask    # softmax 仍用训练 logits，保梯度
        # 专家输出加权求和
        out = torch.zeros_like(x)
        for i, e in enumerate(self.experts):
            out = out + gating[..., i:i+1] * e(x)
        return out, mask.detach().long()        # mask 供回放记录用

# ---- 模拟 rollout（推理引擎）----
torch.manual_seed(0)
moe = MoELayer(d=8, n_experts=4, topk=2)
x = torch.randn(3, 8)
with torch.no_grad():
    _, infer_mask = moe(x)                       # 推理时记录 I_infer (3,4)
print("推理选的专家 mask:\n", infer_mask)

# ---- 模拟训练引擎前向（带 R3）----
moe.set_replay_mask(infer_mask)                  # 注入 I_infer
out_train, _ = moe(x)                            # 训练用 I_infer，不重选专家
loss = out_train.pow(2).sum()
loss.backward()
print("router 梯度范数（有梯度，因 softmax(s_train) 保流）:", moe.router.weight.grad.norm().item())

# 对比：无 R3 时训练自己跑 router，可能选不同专家
moe.set_replay_mask(None)
out_nor3, train_mask = moe(x)
print("无 R3 训练自己选的专家 mask:\n", train_mask)
print("与推理 mask 是否一致:", torch.equal(train_mask, infer_mask))
```

> [!note] 教学要点
> - `_replay_mask` 就是 $I_{\text{infer}}$，注入后训练不再调 `topk`。
> - `gating = softmax(s) * mask`：mask 来自推理（常数），softmax 用训练 logits（有梯度），正是 R3 的 (a)+(b) 双目的。
> - 无 R3 时训练自己跑 router，因数值差异 `train_mask` 可能 $\neq$ `infer_mask`，这就是 TIM。
> - 生产实现里 mask 不是逐层 Python 注入，而是整条序列、所有 MoE 层一次性以 `(seq-1, n_layers, topk)` 的 int32 张量从 rollout 流到 Megatron 的 `RouterReplay`（见 §3.4）。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[MoE路由]]（router/TopK/专家门控，R3 操作的核心对象）、[[trajectory generation]]（rollout 产轨迹，R3 在此记录路由）、[[rollout worker]]（推理引擎承载 rollout）、推理引擎 vLLM/SGLang（需支持 `enable_return_routed_experts`）、训练引擎 Megatron（需 `RouterReplay` 安装 mask）。
- **下游（应用）**: MoE 模型的 RLHF/RL 训练稳定性（防 [[policy collapse]]、[[KL explosion]]）、[[训推不一致]]（R3 是其 MoE 结构型 TIM 的专门解法）、PPO/GRPO 的 $\rho$ 精度（[[PPO clipped objective]]、[[stale policy problem]]）、训推一致性工程（[[训推分离]] 里的 weight sync 是权重一致，R3 是前向路由一致，两者正交）。
- **对比 / 易混**:
  - **R3（跨引擎回放）vs R2（训练引擎内回放）**：R2 只在 Megatron 内部记录并回放路由，不解决推理→训练 gap；R3 跨引擎，直击根因。实测 R2 对 TIM 几乎无改善。
  - **R3 vs GSPO/TIS（算法层裁断）**：算法层靠 clip/丢弃超阈样本治标，TIM 涨快时样本丢光训练停摆；R3 不丢数据，从工程根因消除 mismatch。论文里 R3 在稳定性上优于 GSPO、TIS。
  - **R3 的"路由回放" vs experience replay 的"经验回放"**：都叫 replay 但完全不同——R3 回放的是**MoE 专家选择掩码**（前向路由一致性），experience replay 回放的是**RL 轨迹**（样本复用，见 [[R3 误解纠正前内容|旧版误记]] 已废弃）。
  - **R3（训推前向一致）vs weight sync（训推权重一致）**：weight sync 保证两边权重 $\theta$ 相同（[[weight sync mechanism]]），R3 保证**即便权重相同、前向算出的专家选择也相同**。前者是参数一致，后者是计算路径一致，正交但都需做。
  - **R3 vs bitwise consistency**：逐 bit 一致（同实现、定 determinism）最彻底但性能开销大；R3 只对齐离散的专家选择，保留 softmax 数值自由度，是"够用且轻"的折中。

## 7. 常见误区与易错点

> [!warning] 误区 1：把 R3 当成"rollout/routing/replay 三段流水线"
> 这正是本笔记之前的错误。R3 是一个**具体的路由回放方法**（论文 arXiv:2510.11370），三个 R 都指向"把 rollout 阶段的 routing 回放到训练"，不是"生成→分发→经验回放"的流水线缩写。看到 "R3" 应直接联想到"MoE 训推路由对齐"。

> [!warning] 误区 2：以为 R3 是 dense 模型也要用的
> R3 **只对 MoE 模型有意义**（dense 模型没有 router，不存在专家选择不一致）。NeMo-RL 文档明说 "It is not needed for dense models"。dense 模型的训推数值小扰动由别的手段（fp8/精度对齐/算子对齐）处理。

> [!warning] 误区 3：以为 R3 连 softmax/门控权重也固定
> R3 只固定**离散的专家选择掩码 $I$**，softmax 仍用训练侧 logits 算以保梯度。若连 $g$ 也固定，router 权重 $W_r$ 拿不到梯度，路由策略无法学习。

> [!warning] 误区 4：以为 R3 能替代重要性采样/stale 修正
> R3 只解决"同权重下训推前向不一致"，不解决"rollout 用旧权重、learner 用新权重"的 off-policy staleness。$\rho$ 修正仍要做，R3 只是让 $\rho$ 不被路由噪声污染。

> [!warning] 误区 5：路由张量维度/映射搞错
> 工程上几个典型坑（macaron.im 博客记录的 verl/vLLM 实现 bug）：
> - **dense 层偏移错位**：vLLM 按"全部 transformer 层"（含 dense）报路由，Megatron 只有 MoE 层有 router。若用 `i + offset` 局部映射，dense 层后的每个 MoE 层都会被错位。修法：用 router 的全局 `layer_number` 映射，而非局部 offset。
> - **EPLB 物理/逻辑专家 ID 混淆**：vLLM 开 EPLB（专家并行负载均衡）会把逻辑专家 ID 映成物理 ID，Megatron 回放期望逻辑 ID。必须在 EPLB 映射**之前**捕获逻辑 ID。
> - **捕获时机错误**：vLLM 里 `RoutedExpertsCapturer` 在 KV-cache init 时创建，但 `FusedMoE` 在 `init` 时 bind capture，导致捕获器还没建好就错过绑定 → 路由张量全 0。修法：在首次 forward 时延迟绑定。
> - **alltoall split size 不匹配**：router replay 下 `routing_map.sum()` 可能 $<$ `num_tokens * topk`（重复 index），导致 alltoall split size 算错。修法：从 routing_map 推导而非直接用。
> - **缺 token 的 fallback**：vLLM 偶尔返回的路由条数少于 token 数，缺的位用全 `-1` sentinel，Megatron 对这些位回退到自己的 router（局部 fallback，不整批禁用 R3），并记 `r3/routed_experts_fallback_token_route_fraction`（正常应 ≈0）。

> [!warning] 误区 6：以为 R3 开了就稳，忽略异步调度冲突
> vLLM 0.21.x 上 R3 与 async scheduling 不兼容，必须 `--no-async-scheduling`。开 R3 时要同步关掉 async 调度，否则路由捕获/回放错乱。

> [!warning] 误区 7：以为 R2 ≈ R3
> R2（训练侧回放）确定性好但**不跨引擎**，对推理→训练的 TIM 几乎无效。只有 R3（跨引擎）才直击根因。选型时别拿 R2 充数。

## 8. 延伸细节

### 8.1 代表实现

- **NeMo-RL**（NVIDIA）：`Router Replay`，文档 `docs/guides/router-replay.html`。配 vLLM rollout + Megatron MoE 训练，`enable_return_routed_experts=True`。dense 模型默认关。
- **verl**（火山引擎）：`--use-rollout-routing-replay`，配 SGLang（原生支持）或 vLLM（需 patch）。Megatron actor 侧 `RouterReplay` 安装 mask。修复 PR：verl #5037（全局层号对齐）、#4986（alltoall dispatcher）、#4973（R2/R3 数据隔离）。
- **vLLM**：`--enable-return-routed-experts` 引擎内捕获，HTTP 层需 patch 暴露 `prompt_routed_experts` + `choices[].routed_experts`。修复 PR：vllm #33013（捕获逻辑专家 ID）。
- **SGLang**：`--enable-return-routed-experts`，base64 int32 buffer 放 `meta_info.routed_experts`，端到端原生支持，是 R3 的参考实现。

### 8.2 跨引擎一致性的验证

R3 的正确性靠两类端到端检查（NeMo-RL 文档）：

1. **路由携带不变性**：rollout payload 的 `input_ids` + `routed_experts` 经 TransferQueue → packing → context-parallel slicing → Megatron replay，**token 身份与专家编号一一对应不变**。
2. **TIM 降低有效性**：配对跑 R3-on / R3-off，比 `train/token_mult_prob_error` 和 `train/js_divergence_error`，R3-on 应显著更低。

### 8.3 不同 MoE 架构的敏感度

macaron.im 博客指出：**不是所有 MoE 对 TIM 同样敏感**。Qwen3 的均匀路由逻辑较鲁棒；DeepSeekV3 风格的混合路由（shared expert + routed expert，复杂语义）对路由发散极敏感，Moonlight-16B-A3B 上 TIM 比 Qwen3-30B-A3B 严重得多，R3 收益更大。DeepSeekV3 风格模型上 R3 修好前训练直接崩（路由张量全 0 / 层号错位），修好后 TIM 大幅下降、PPO KL 近零（step 46 时 2.6×10⁻⁵）。

### 8.4 与其他一致性手段的层次

| 层次 | 手段 | 解决什么 |
|---|---|---|
| 权重一致 | [[weight sync mechanism]] | rollout worker 和 learner 的 $\theta$ 相同 |
| 前向路由一致 | **R3** | 同 $\theta$ 下 MoE 选同一组专家 |
| 数值精度一致 | fp16/bf16/fp8 对齐、bitwise determinism | 同 $\theta$、同路由下 logits 数值一致 |
| 算子实现一致 | 训推同框架 / 算子对齐 | 同输入下算子输出一致 |

R3 是"前向路由一致"这一层，**不替代**上下的权重同步或数值对齐，而是补上 MoE 特有的"离散选择"这一环。

### 8.5 趋势

- **MoE 成为 RLHF 主力**（DeepSeek-V3/R1、Qwen3、Kimi K2 等都是 MoE），TIM 问题普遍化，R3 从"可选优化"变成"大规模 MoE RL 训练的必备稳定性工程"。
- **推理引擎原生支持路由导出**：SGLang 已原生支持，vLLM 在跟进（RFC: vllm-project/vime#32），未来 R3 的工程门槛会降低。
- **与更激进的 off-policy 算法协同**：GRPO/DAPO 等 off-policy 化方法 $\rho$ 更发散，R3 把"假性 $\rho$ 偏移"消掉，让 off-policy 的 $\rho$ 只反映真实权重更新，两者正交叠加。
- **从 MoE 路由推广到更一般的"训推离散选择"对齐**：任何训推两侧存在不连续选择（如量化区间选择、采样温度下的具体 token）的地方，都有类似的"记录-回放"思路。

---
相关: [[Rollout系统]]、[[MoE路由]]、[[trajectory generation]]、[[rollout worker]]、[[weight sync mechanism]]、[[stale policy problem]]、[[PPO clipped objective]]、[[policy collapse]]、[[KL explosion]]、[[训推分离]]、[[asynchronous training]]
