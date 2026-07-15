# rollout train reference logprob一致性

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: 三端 logprob 一致性 / rollout-train-ref logp parity / TIM 在 RL 的后果与治理 / training-inference mismatch in RLHF / 训推 logp 一致性
> **难度**: 高（需懂 [[训推不一致]]、[[importance sampling与off-policy correction]]、[[RL权重同步]]、[[trainer与rollout引擎组合]]、[[PPO clipped objective]]、[[KL penalty与KL control]]、[[FP8量化方案]]、[[MoE]]、[[R3 rollout routing replay]]）

> **所属小节**: §83


## 1. 一句话定义

**rollout / train / reference logprob 一致性** 是 [[17-RL训推一体框架]] 中三个角色算出的逐 token 对数概率 $\log\pi$ 必须**数值一致**的要求——**rollout 端**（vLLM/SGLang 采样时记的 `old_log_prob`=$\log\pi_{\theta_{old}}$）、**train 端**（trainer 用更新后 $\theta_{new}$ 重算的 `new_log_prob`=$\log\pi_{\theta_{new}}$）、**reference 端**（冻结 ref 模型算的 `ref_log_prob`=$\log\pi_{ref}$）三者虽各自权重不同导致 $\log\pi$ 本应不同（rollout 用旧 $\theta$、train 用新 $\theta$、ref 用冻结 ref，这是 importance sampling 的本意），但**前提是"同权重下算出的 $\log\pi$ 必须数值一致"**——即 $\theta_{rollout}=\theta_{trainer}$ 时 $\log\pi_{rollout}\overset{!}{=}\log\pi_{trainer}$、$\theta_{trainer}=\theta_{ref}$ 时 $\log\pi_{trainer}\overset{!}{=}\log\pi_{ref}$。一旦同权重下 $\log\pi$ 不一致（即 **TIM, training-inference mismatch**：同一 $\theta$、同一输入，rollout 的 vLLM 与 trainer 的训练框架算出不同 logp），PPO/GRPO 的 ratio $\rho=\exp(\log\pi_{\theta_{new}}-\log\pi_{\theta_{old}})$ 在 $\theta_{new}=\theta_{old}$ 时本应 $=1$ 却偏离 $1$，importance sampling 修正被污染 → 梯度有偏 → KL 爆炸（[[KL explosion]]）/ policy 崩（[[policy collapse]]）。TIM 的来源分层——**精度路径**（rollout 用 FP8/INT8 量化、train 用 BF16/FP32 master）、**算子实现差异**（vLLM 的 PagedAttention/FlashAttention 与训练框架的实现细节微差）、**kernel 融合**（推理 fused kernel 改浮点累加顺序）、**归约顺序**（DP/TP 并行归约 vs 单卡）、**RoPE base** 不同、**MoE 路由选择不同**（推理与训练选不同专家，详见 [[R3 rollout routing replay]]、[[训推不一致]]）；衡量指标是 **Pearson 相关系数 $>0.99$ 且逐 token max_abs_diff $<10^{-3}$**；后果四级——噪声 → 有偏 → KL 爆炸 → 崩溃（MoE 尤甚）；消除手段分层——**统一精度**（训推同 dtype，或 MXFP8 统一）、**bitwise 对齐算子**（用同一 kernel 实现）、**recompute 策略**（用 train 端 logp 重算 `old_log_prob` 做锚点，而非用 rollout 算的）、**MoE 的 R3 回放**（推理记路由训练回放，保同一专家子网络）。本条与 [[训推不一致]] 互为专章——[[训推不一致]] 讲 TIM 的**机理**（来源分层、衡量、对训练的污染链），本条讲 TIM **在 RL 训练中的后果与治理**（ratio 失真→梯度有偏→KL 爆→崩，及 recompute/R3 等工程手段）。

> [!note] 三句话定位
> - **是什么**：三端（rollout/train/ref）算的 $\log\pi$ 要求"同权重下数值一致"，否则 TIM 让 ratio $\rho$ 在 $\theta_{new}=\theta_{old}$ 时偏离 1 → IS 修正失真 → 梯度有偏 → KL 爆/崩。
> - **为什么**：RL 的 ratio/KL/advantage 全依赖 $\log\pi$，logp 算错一点，ratio 指数放大（$\rho=\exp(\Delta\log p)$）→ clip 失效/KL 爆/policy 崩。TIM 来源是训推两套引擎（精度/算子/融合/MoE 路由）的数值差。
> - **与 [[训推不一致]] 关系**：后者讲 TIM 的**机理**（来源分层、衡量、污染链），本条讲 TIM 在 **RL 中的后果与治理**（ratio 失真→梯度有偏→KL 爆→崩，recompute/R3 手段）。互为专章。


## 2. 为什么需要它（动机与背景）

### 2.1 三端 logp 各算各的

RLHF/GRPO 一个 batch 的数据流里，**三个角色各算一遍 $\log\pi$**：

| 角色 | 权重 | 算什么 | 引擎 | 用途 |
|---|---|---|---|---|
| **rollout** | $\theta_{old}$（采样时权重） | `old_log_prob`=$\log\pi_{\theta_{old}}(o_t\|q,o_{<t})$ | vLLM/SGLang | IS 的分母 $\rho=\pi_\theta/\pi_{old}$ |
| **train** | $\theta_{new}$（trainer 更新后权重） | `new_log_prob`=$\log\pi_{\theta_{new}}(o_t\|q,o_{<t})$ | FSDP/Megatron | IS 的分子 + 梯度源 |
| **reference** | $\theta_{ref}$（冻结 ref） | `ref_log_prob`=$\log\pi_{ref}(o_t\|q,o_{<t})$ | 常同 trainer 引擎或独立 | KL to ref：$\mathbb{D}_{KL}[\pi_\theta\|\pi_{ref}]$ |

三端权重本不同（$\theta_{old}$/$\theta_{new}$/$\theta_{ref}$），$\log\pi$ 本应不同——这是 importance sampling 与 KL 约束的**本意**，不是 bug。

### 2.2 真正的问题：同权重下 logp 不一致（TIM）

**前提假设**：三端算 $\log\pi$ 的引擎若**数值实现一致**，则"同权重同输入同 $\log\pi$"。即若 $\theta_{rollout}=\theta_{trainer}$，应 $\log\pi_{rollout}=\log\pi_{trainer}$（同一 $\theta$、同一输入，两套引擎算出同一 logp）。

**TIM 打破这个假设**：rollout 用 vLLM（BF16/FP8 + 优化 kernel + PagedAttention + CUDA Graph），train 用 FSDP/Megatron（FP32 master + BF16 forward + Megatron 自己的 attention kernel），**同一 $\theta$、同一输入算出的 logp 有数值差**。这就是 training-inference mismatch（[[训推不一致]]）。

### 2.3 ratio 失真：TIM 在 RL 的直接后果

PPO/GRPO 的 ratio：

$$
\rho_t = \frac{\pi_{\theta_{new}}(o_t)}{\pi_{\theta_{old}}(o_t)} = \exp\!\big(\log\pi_{\theta_{new}}(o_t) - \log\pi_{\theta_{old}}(o_t)\big)
$$

考虑第一步更新前 $\theta_{new}=\theta_{old}$（刚 sync 完，rollout 与 trainer 同权重），**理论上 $\rho$ 应 $=1$**。但若 TIM 让 rollout 算的 $\log\pi_{\theta_{old}}^{vLLM}$ 与 trainer 算的 $\log\pi_{\theta_{new}}^{trainer}$ 差 $\epsilon$（即便 $\theta$ 相同），则：

$$
\rho_t = \exp(\epsilon) \approx 1 + \epsilon \ne 1
$$

即 **$\theta$ 完全没动，ratio 就偏离 1**。这污染 IS 修正——trainer 以为"rollout 用的旧策略与新策略有偏差需修正"，实际是两套引擎的数值差。后续：
- $\rho$ 偏离 1 → PPO clip 误触发（`pg_clipfrac` 虚高）→ 有效梯度被错误截断；
- $\rho$ 指数放大偏离（$\exp$）→ ratio 方差爆 → 梯度噪声大；
- KL to ref 同理失真 → KL 约束误判 → 爆 KL（[[KL explosion]]）或误松弛；
- 累积 → policy 朝错误方向漂移 → reward hacking / policy collapse（[[policy collapse]]）。

> [!warning] 误区：$\theta_{new}=\theta_{old}$ 时 ratio 就该是 1
> 理论上是，但 TIM 让 rollout 的 vLLM 与 trainer 算的 logp 有数值差，$\theta$ 相同时 $\rho=\exp(\epsilon)\ne1$。这个"本应为 1 却偏离"的 $\epsilon$ 是 TIM 的直接度量，也是 RL 训练不稳的隐藏杀手。

### 2.4 为什么 LLM-RL 的 TIM 格外严重

- **模型大**：70B 模型每层 attention/FFN 的浮点累加次数极多，$\epsilon$ 累积放大。
- **序列长**：response 几百~几千 token，每 token 都算 logp，TIM 逐 token 偏差累加。
- **ratio 指数放大**：$\rho=\exp(\Delta\log p)$，logp 的线性偏差经 $\exp$ 指数放大成 ratio 的剧烈偏离。
- **MoE 路由**：MoE 模型推理与训练选不同专家 → 完全不同的子网络 → logp 差异是结构性的（非仅数值），尤严重（详见 §3.3、[[R3 rollout routing replay]]）。
- **FP8 量化推理**：vLLM 推理常开 FP8/INT8 KV cache 或权重量化，训练用 BF16/FP32，精度路径不同是 TIM 主因（详见 [[FP8量化方案]]）。

故 LLM-RL 对 TIM 的容忍度极低，必须治理。


## 3. 核心概念详解

### 3.1 三端 logp 的数据流

一个 RLHF/GRPO batch 的完整 logp 流：

```
rollout (vLLM, θ_old):
  采 o ~ π_θ_old(·|q), 逐 token 记 old_log_prob = log π_θ_old(o_t|q,o_<t)  [vLLM forward]
                                  │
                                  ▼
train (FSDP/Megatron, θ_new):
  重算 new_log_prob = log π_θ_new(o_t|q,o_<t)  [trainer forward, θ 已更新]
  ratio = exp(new_log_prob - old_log_prob)     ★ IS 修正
  pg_loss = -min(ratio*A, clip(ratio,1±ε)*A)   [PPO clipped objective]
  backward 更新 θ

reference (冻结 ref, θ_ref):
  ref_log_prob = log π_ref(o_t|q,o_<t)         [ref forward]
  KL = new_log_prob - ref_log_prob  (近似)     ★ KL to ref 约束
```

三端各 forward 一遍算 logp。**一致性要求**：若三端引擎数值实现一致，"同权重同输入同 logp"。TIM 打破此。

### 3.2 TIM 来源分层（数值型）

来自 [[训推不一致]] §3.1 已核实的来源表，本条按"对 RL 的污染方式"重组：

| 来源 | rollout（vLLM） | train（FSDP/Megatron） | 差异性质 | 严重度 |
|---|---|---|---|---|
| **精度路径** | BF16 forward + 可能 FP8/INT8 KV cache/权重量化 | FP32 master weight + BF16 forward（或 FP8 forward via TE） | 数值精度不同 → 累加误差 | 高（FP8 KV 尤甚） |
| **算子实现** | PagedAttention（非 contiguous KV block）/ FlashAttention 推理版 | Megatron 的 attention / FlashAttention 训练版 | kernel 实现细节微差 | 中 |
| **kernel 融合** | fused kernel（attention+RoPE+norm 融合）改浮点累加顺序 | 较少融合或不同融合策略 | 浮点累加顺序不同 → $O(\epsilon)$ 差 | 中 |
| **归约顺序** | TP 并行归约（vLLM TP） | DP/TP 并行归约（Megatron 3D） | 并行归约顺序不同 → 浮点差 | 低~中 |
| **RoPE base** | vLLM 默认或配置 base | Megatron 默认或配置 base | 若 base 不同 → 位置编码完全不同 → logp 差大 | 高（若配错）/ 0（若对齐） |
| **KV cache 量化** | vLLM 可开 `kv_cache_dtype="fp8"` | 训练无 KV cache | 历史 KV 被量化引入误差 → attention 累积放大 | 高（若开） |
| **tokenizer/特殊 token** | vLLM 的 tokenize 与 trainer 微差 | Megatron 的 tokenize | 特殊 token 处理不同 → token id 微差 → logp 差 | 中（需对齐） |

前六类是**数值型** TIM——同 $\theta$ 同 token id，算出不同 logp（浮点差）。可靠统一精度/bitwise 对齐/recompute 治理。

### 3.3 MoE 路由不一致：结构性 TIM（最严重）

[[MoE]] 模型的 expert routing 是**离散选择**：每个 token 经 router（gating）选 TopK 专家。推理（vLLM）与训练（Megatron）若选**不同专家**，等于走了**完全不同的子网络** → logp 差异是**结构性的**，非仅浮点差，量级远大于数值型 TIM。

来源：
- **路由实现差异**：vLLM 的 MoE router 实现与 Megatron 的 gating 实现细节微差（如 tie_break 规则、噪声处理）→ TopK 选择不同。
- **精度路径**：router 用 FP8/BF16 算 logits，TopK 在边界 token（第 K 与第 K+1 专家分数接近）易选不同。
- **量化**：专家权重若量化，路由 logits 受影响。

后果：同 $\theta$ 同输入，vLLM 走专家 [3,7]，Megatron 走 [3,5] → 两个不同子网络 → logp 差异巨大（不是 $O(\epsilon)$ 而是 $O(1)$）。ratio 严重失真，训练直接崩。这是 MoE-RL 最坑的 TIM，靠 [[R3 rollout routing replay]] 治理——推理时记录每 token 每层的 TopK 专家**索引**，训练时回放（强制走同一专家子网络），gating 仍用训练态 $s_{train}$ 算保梯度。dense 模型无此问题（无路由选择）。详见 [[R3 rollout routing replay]]、[[训推不一致]]。

### 3.4 衡量指标

TIM 的衡量（来自 [[训推不一致]] 已核实）：

| 指标 | 计算 | 阈值 | 含义 |
|---|---|---|---|
| **Pearson 相关系数** | rollout 与 trainer 的逐 token logp 序列的 Pearson | $>0.99$ | 形状一致性（对 scale 不敏感） |
| **max_abs_diff** | $\max_t\|\log\pi_{rollout,t}-\log\pi_{trainer,t}\|$ | $<10^{-3}$ | 绝对差上界 |
| **mean_abs_diff** | 平均绝对差 | $<10^{-4}$ | 平均偏离 |
| **KL** | $\mathbb{D}_{KL}[\pi_{rollout}\|\pi_{trainer}]$ | 极小 | 分布级差异 |
| **ratio 偏移** | $\theta_{new}=\theta_{old}$ 时 $\rho$ 的均值/方差 | 均值 $\approx1$，方差极小 | TIM 对 IS 的直接污染 |

**工程健康度快检**：Pearson 跌通常 KL 涨。colocate 同步训练下 Pearson 应稳定 $>0.99$；若跌 $<0.99$ 查 TIM（精度/算子/MoE 路由）。详见 [[RLHF训练监控指标详解]] 的 `pearson_corr` 指标。

> [!note] 补充：Pearson 与 KL 的关系
> KL 是分布级差异，Pearson 是"逐 token logp 序列"的线性相关。Pearson 计算便宜、对 scale 不敏感、对"整体趋势是否一致"敏感，适合做工程健康度快检。它 $\ne$ KL，但两者高相关：Pearson 跌通常 KL 涨。详见 [[asynchronous training]] §3.3、[[stale policy problem]]。

### 3.5 后果四级：噪声 → 有偏 → KL 爆 → 崩

TIM 对 RL 训练的污染是渐进的：

| 级别 | 现象 | 原因 | 可恢复性 |
|---|---|---|---|
| **① 噪声** | ratio 在 $\theta_{new}=\theta_{old}$ 时小偏离 1（$\rho=1\pm\epsilon$），clip 偶发触发 | 数值型 TIM（精度/算子/融合微差） | 可（clip 兜底，训练仍能进行） |
| **② 有偏** | ratio 系统性偏离 1（某方向），梯度方向持续偏 | 精度路径系统性差（FP8 vs BF16） | 需治理（统一精度） |
| **③ KL 爆炸** | KL to ref 失控增长，`ppo_kl` 飙 | ratio 指数放大 + KL 算错累积 | 需立即干预（降 lr/加 KL 系数） |
| **④ 崩溃** | policy 漂成乱码 / reward hacking / MoE 完全失配 | MoE 路由结构性失配 / 长期 TIM 累积 | 难恢复（需回滚 checkpoint） |

MoE 模型常直接从①跳到④（结构性失配，非渐进）。dense 模型多 ①→②→③ 渐进。

### 3.6 消除手段分层

| 层次 | 手段 | 治理的 TIM 类型 | 效果 | 成本 |
|---|---|---|---|---|
| **精度对齐** | 训推同 dtype（都 BF16），或 MXFP8 统一方案 | 精度路径 | 根治精度差 | 低（配置） |
| **bitwise 算子对齐** | rollout 与 train 用同一 kernel 实现（如都用 FlashAttention 同版） | 算子实现/融合/归约 | 根治算子差 | 中（需框架适配） |
| **recompute 策略** | 用 train 端 logp 重算 `old_log_prob`（trainer 用 $\theta_{old}$ forward 算），而非用 rollout 算的 | 全部数值型（绕过 vLLM logp） | 根治数值型 TIM | 高（多一次 forward） |
| **MoE R3 回放** | 推理记路由、训练回放，保同一专家子网络 | MoE 路由（结构性） | 根治 MoE TIM | 中（需 R3 支持） |
| **KV 量化对齐** | rollout 不开 FP8 KV cache，或训推统一 KV 精度 | KV cache 量化 | 根治 KV 路径 | 低（配置，但损显存） |
| **RoPE base 对齐** | 训推配置同一 RoPE base | RoPE base 配错 | 根治（若配错） | 零（配置） |

**主流实践**：verl/OpenRLHF 常用 **recompute 策略**做兜底——不信任 rollout 算的 `old_log_prob`，trainer 拿到 $(q,o)$ 后用 $\theta_{old}$（sync 时的权重）重新 forward 算 `old_log_prob`，保证 old/new/ref 三端都出自同一引擎（trainer），彻底消除数值型 TIM。代价是多一次 forward（用 $\theta_{old}$ 重算 old）。MoE 模型额外加 R3。详见 §5.2。


## 4. 数学原理 / 公式

### 4.1 推导：ratio 在 TIM 下的偏离

设真实权重 $\theta$，rollout 用 vLLM 算 $\log\pi_\theta^{v}$，trainer 算 $\log\pi_\theta^{t}$，TIM 引入偏差 $\epsilon_t$：

$$
\log\pi_\theta^{v}(o_t) = \log\pi_\theta^{t}(o_t) + \epsilon_t,\quad \mathbb{E}[\epsilon_t]\ne 0\text{（可能有偏）},\ \mathrm{Var}[\epsilon_t]>0
$$

第一步更新前 $\theta_{new}=\theta_{old}=\theta$，理论 ratio $=1$。实际：

$$
\rho_t = \exp\!\big(\log\pi_\theta^{t}(o_t) - \log\pi_\theta^{v}(o_t)\big) = \exp(-\epsilon_t) \approx 1 - \epsilon_t + \frac{\epsilon_t^2}{2}
$$

- $\mathbb{E}[\rho_t]\approx 1 - \mathbb{E}[\epsilon_t] + \frac{1}{2}\mathbb{E}[\epsilon_t^2]$：若 $\epsilon_t$ 有非零均值（系统性偏），$\rho$ 系统性偏离 1 → 梯度有偏。
- $\mathrm{Var}[\rho_t]\approx \mathrm{Var}[\epsilon_t]$：TIM 方差直接传给 ratio 方差 → 梯度噪声大。

**关键**：即便 $\theta$ 完全没动，TIM 让 ratio $\ne1$，且偏离由 $\epsilon$ 的均值与方差决定。$\exp$ 放大让小 $\epsilon$ 也显著。

### 4.2 推导：ratio 经 PPO clip 后的梯度失真

PPO clipped objective（[[PPO clipped objective]]）：

$$
L^{CLIP}(\theta) = \mathbb{E}_t\big[\min(\rho_t \hat A_t,\ \text{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\hat A_t)\big]
$$

无 TIM 时 $\theta_{new}=\theta_{old}$ $\Rightarrow$ $\rho_t=1$，$L^{CLIP}=\hat A_t$（advantage 直接传），clip 不触发。梯度 $\nabla L = \nabla\log\pi\cdot\hat A$ 正常。

有 TIM 时 $\rho_t=1-\epsilon_t$，若 $|\epsilon_t|>\varepsilon$（TIM 超过 clip 范围，如 $\varepsilon=0.2$ 但 $\epsilon_t=0.3$），clip 触发：

$$
\text{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\hat A_t = (1-\varepsilon)\hat A_t\quad(\epsilon_t>\varepsilon)
$$

此时 $\min$ 取 clip 项，梯度被截断在 $(1-\varepsilon)\hat A$，**真实梯度信号丢失**。若 TIM 是系统性的（$\epsilon_t$ 同向），大量 token 被错误 clip，梯度方向被扭曲。

**clip 频率** `pg_clipfrac` 是 TIM 的直接仪表：colocate 同步训练下 `pg_clipfrac` 应低（$<0.1$）；若 $\theta$ 没大动却 `pg_clipfrac` 高，查 TIM。详见 [[RLHF训练监控指标详解]]。

### 4.3 推导：KL to ref 的 TIM 失真

KL 约束（[[KL penalty与KL control]]）逐 token 近似：

$$
\mathbb{D}_{KL}^{(t)}[\pi_\theta\|\pi_{ref}] \approx \exp(\log\pi_{ref}^{?}(o_t) - \log\pi_\theta^{?}(o_t)) - (\log\pi_{ref}^{?}(o_t) - \log\pi_\theta^{?}(o_t)) - 1
$$

（Schulman k3 近似，$?$ 表示哪端引擎算。）若 ref 与 trainer 不同引擎，$\log\pi_{ref}$ 与 $\log\pi_\theta$ 都带各自的 TIM，KL 估计被双重污染。若 ref 与 trainer 同引擎（ref 常用 trainer 引擎算），则 ref 与 trainer 间无 TIM，但 ref 与 rollout 间有（不过 KL to ref 用的是 trainer 的 $\log\pi_\theta$ 与 ref 的 $\log\pi_{ref}$，若同引擎则一致）。**故 ref 常用 trainer 引擎算**，避免 ref 端引入额外 TIM。

### 4.4 推导：MoE 路由失配的 ratio 量级

设 MoE 第 $l$ 层 token $t$，vLLM 选专家 $S_v\subset\{1..E\}$（$|S_v|=K$），trainer 选 $S_t$。若 $S_v\ne S_t$：

$$
\log\pi^{v} = \log\text{softmax}(\text{router}_v) + \sum_{e\in S_v}\log\text{gate}_e + \log f_{S_v}(x)
$$

$$
\log\pi^{t} = \log\text{softmax}(\text{router}_t) + \sum_{e\in S_t}\log\text{gate}_e + \log f_{S_t}(x)
$$

$f_{S}$ 是专家子网络的输出。$S_v\ne S_t$ $\Rightarrow$ $f_{S_v}\ne f_{S_t}$ $\Rightarrow$ $\log\pi^v - \log\pi^t = O(1)$（非 $O(\epsilon)$）。ratio $\rho=\exp(O(1))$ 偏离 1 剧烈（如 $\exp(1)\approx2.7$），远超 PPO clip $\varepsilon=0.2$ → 几乎所有 token 被 clip → 梯度全失真 → 训练直接崩。这就是 MoE-RL 不治理 R3 会崩的数学根。详见 [[R3 rollout routing replay]]、[[importance sampling与off-policy correction]]。

### 4.5 recompute 策略的代价

recompute（trainer 用 $\theta_{old}$ 重算 `old_log_prob`）消除数值型 TIM，但多一次 forward：

$$
T_{\text{recompute}} = T_{\text{forward}}(\theta_{old})\quad\text{（全 response 一次 forward）}
$$

vs 用 rollout 算的 `old_log_prob`（零额外 forward，但带 TIM）。trade-off：recompute 多一次 forward 算力换 TIM 根治。对大模型这是不小开销，但保训练稳定性，主流框架（verl/OpenRLHF）默认开 recompute。MoE 还需 R3（记录+回放路由）叠加。


## 5. 代码示例（可选）

### 5.1 衡量 TIM：Pearson + max_abs_diff

```python
import numpy as np
from scipy.stats import pearsonr

def measure_tim(logprob_rollout, logprob_trainer):
    """
    logprob_rollout: (T,) vLLM 算的逐 token logp (同 θ)
    logprob_trainer: (T,) trainer 算的逐 token logp (同 θ)
    返回 TIM 指标
    """
    diff = logprob_rollout - logprob_trainer
    pearson, _ = pearsonr(logprob_rollout, logprob_trainer)
    return {
        "pearson": pearson,                  # > 0.99 才合格
        "max_abs_diff": np.max(np.abs(diff)),  # < 1e-3 才合格
        "mean_abs_diff": np.mean(np.abs(diff)),
        "ratio_drift_mean": np.mean(np.exp(diff)),  # θ_new=θ_old 时 rho 均值, 应≈1
        "ratio_drift_std": np.std(np.exp(diff)),    # 应极小
    }

# demo: 假设 vLLM 与 trainer 同 θ 算的 logp (理想应一致, 实际有 TIM)
np.random.seed(0)
logp_true = np.random.randn(1000) * 2 - 5       # 模拟真实 logp
tim_noise = np.random.randn(1000) * 1e-2       # TIM 噪声 (BF16 级)
logp_rollout = logp_true + tim_noise
logp_trainer = logp_true                       # trainer 用 FP32 更精确

metrics = measure_tim(logp_rollout, logp_trainer)
for k, v in metrics.items():
    print(f"{k}: {v:.6f}")
# pearson: 0.9999... (合格), max_abs_diff ~0.03 (可能超 1e-3), ratio_drift_mean ~1.0
# -> 若 max_abs_diff > 1e-3 需治理 (统一精度/recompute)

# 极端: MoE 路由失配 (结构性 TIM, O(1) 差)
logp_rollout_moe = logp_true + np.random.randn(1000) * 0.5  # 大偏差
metrics_moe = measure_tim(logp_rollout_moe, logp_trainer)
print(f"\nMoE 路由失配: pearson={metrics_moe['pearson']:.4f}, "
      f"max_abs_diff={metrics_moe['max_abs_diff']:.3f}, "
      f"ratio_drift_mean={metrics_moe['ratio_drift_mean']:.3f}")
# pearson 可能仍高 (形状对), 但 max_abs_diff 大, ratio 偏离 1 剧烈 -> 需 R3
```

### 5.2 verl recompute 策略（trainer 重算 old_log_prob）

```python
import torch

def rl_step_with_recompute(actor, ref, rollout, batch, eps=0.2, beta=0.001):
    """
    recompute 策略: 不信任 rollout 算的 old_log_prob, trainer 用 θ_old 重算
    消除数值型 TIM (old/new/ref 三端都出自 trainer 引擎)
    """
    q, o = batch["prompt"], batch["response"]
    theta_old = actor.state_dict()  # sync 时的权重 (rollout 用的)

    # ① rollout 采 (vLLM, 用 θ_old), 只拿 response, 不用其 logp
    #    gen = rollout.generate(q)  # vLLM 采, 记的 old_log_prob 被丢弃

    # ② trainer 用 θ_old 重算 old_log_prob (★ recompute, 消 TIM)
    with torch.no_grad():
        old_log_prob = actor.forward(q, o, weights=theta_old).logp  # trainer 引擎算

    # ③ trainer 更新 θ: old -> new
    actor.backward_step(batch)  # backward 更新 θ
    theta_new = actor.state_dict()

    # ④ trainer 用 θ_new 算 new_log_prob (同引擎, 与 old 无 TIM)
    new_log_prob = actor.forward(q, o, weights=theta_new).logp  # 注意要 grad

    # ⑤ ref 算 ref_log_prob (用 trainer 引擎或独立 ref, 但同引擎免 TIM)
    with torch.no_grad():
        ref_log_prob = ref.forward(q, o).logp

    # ⑥ ratio (三端同引擎, 无 TIM)
    ratio = (new_log_prob - old_log_prob).exp()
    adv = batch["advantage"]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    pg_loss = -torch.minimum(surr1, surr2)

    kl = (new_log_prob - ref_log_prob)  # k3 近似
    loss = pg_loss + beta * kl
    return loss
# 代价: ②多一次 forward (用 θ_old), 但保 old/new/ref 三端同引擎无 TIM
```

### 5.3 MoE R3 回放配置（概念，以 SGLang/verl 文档为准）

```bash
# SGLang 推理时记录路由 (R3 前提)
python -m sglang.launch_server ... \
  --enable-return-routed-experts  # ★ 记录每 token 每层 TopK 专家索引

# verl/OpenRLHF 训练时回放 (R3: 推理记的路由训练回放)
# trainer 的 Megatron MoE 安装 RouterReplay mask, 强制走推理选的专家
# gating 仍用 s_train 算 (保梯度), 但 expert 选择 mask 成推理的路由
# 详见 [[R3 rollout routing replay]] §5
```

详见 [[R3 rollout routing replay]] 已批注解答的工程坑（SGLang `--enable-return-routed-experts` 原生 / vLLM 需 patch + `--no-async-scheduling`；张量契约 `(seq-1, n_layers, topk)`；TransferQueue/packing/CP 切片传到 Megatron 的 `RouterReplay`）。


## 6. 与其他知识点的关系

- **上游（依赖）**: [[训推不一致]]（TIM 的机理，本条是其 RL 后果专章）、[[trainer与rollout引擎组合]]（两套引擎是 TIM 的根）、[[RL权重同步]]（sync 保证 θ 对齐，TIM 是同 θ 下 logp 不一致，两者正交）、[[FP8量化方案]]（精度路径 TIM 的具体来源）、[[MoE]]（MoE 路由结构性 TIM）、[[R3 rollout routing replay]]（MoE TIM 的治理手段）、[[PPO clipped objective]]（ratio clip 是 TIM 的兜底与仪表）、[[importance sampling与off-policy correction]]（ratio 的数学）。
- **下游（应用）**: [[17-RL训推一体框架]] 全章、[[RLHF训练监控指标详解]]（`pearson_corr`/`pg_clipfrac`/`ppo_kl` 是 TIM 的监控仪表）、[[KL penalty与KL control]]（KL 失真由 TIM 污染）、[[policy collapse]]/[[KL explosion]]（TIM 不治的终局）。
- **对比 / 易混**:
  - **本条（TIM 在 RL 的后果）vs [[训推不一致]]（TIM 的机理）**：后者讲来源分层、衡量、污染链的机理；本条讲在 RL 中 ratio 失真→梯度有偏→KL 爆→崩的后果与 recompute/R3 治理。互为专章。
  - **TIM（同 θ 下 logp 不一致）vs staleness（不同 θ 的 ratio 偏离）**：前者是两套引擎同权重的数值差（$\theta$ 相同但 logp 不同）；后者是 rollout 用旧 θ 与 trainer 新 θ 的权重差（$\theta$ 不同导致 logp 不同，是 IS 本意）。两者正交——staleness 靠 weight sync 降，TIM 靠精度/recompute/R3 治。详见 [[RL权重同步]] §3.8、[[stale policy problem]]。
  - **三端 logp 一致性 vs 三端 θ 一致性**：θ 一致靠 [[RL权重同步]]（rollout 与 trainer 同 θ）；logp 一致靠 TIM 治理（同 θ 下同 logp）。两层正交。


## 7. 常见误区与易错点

> [!warning] 误区 1：θ 对齐了 logp 就一致
> 错。[[RL权重同步]] 保证 rollout 与 trainer 的 $\theta$ 相同，但同 $\theta$ 下 vLLM 与 trainer 算的 logp 因精度/算子/MoE 路由不一致仍有数值差（TIM）。θ 对齐与 logp 对齐是两层，正交。

> [!warning] 误区 2：$\theta_{new}=\theta_{old}$ 时 ratio 就该是 1
> 理论上是，但 TIM 让 rollout 的 vLLM 与 trainer 算的 logp 有数值差，$\theta$ 相同时 $\rho=\exp(\epsilon)\ne1$。这个"本应为 1 却偏离"是 TIM 的直接度量，也是 RL 不稳的隐藏杀手。

> [!warning] 误区 3：Pearson 高就没事
> 不一定。Pearson 衡量形状一致性，对 scale 不敏感。MoE 路由失配可能 Pearson 仍高（形状对）但 max_abs_diff 大（绝对差大），ratio 剧烈偏离。需 Pearson + max_abs_diff 双指标。详见 [[训推不一致]]。

> [!warning] 误区 4：TIM 是 staleness 的一种
> 不是。TIM 是**同 θ** 下两套引擎的数值差；staleness 是**不同 θ**（rollout 旧 vs trainer 新）导致的 logp 差。两者正交——staleness 靠 weight sync 降，TIM 靠精度/recompute/R3 治。详见 [[RL权重同步]] §3.8、[[stale policy problem]]。

> [!warning] 误区 5：MoE 的 TIM 与 dense 一样是数值型
> 不是。MoE 路由失配是**结构性**的——选不同专家 = 走不同子网络，logp 差 $O(1)$（非 $O(\epsilon)$），ratio 剧烈偏离，训练直接崩。dense 是数值型（浮点差），渐进恶化。MoE 必须 R3，dense 可 recompute。详见 [[R3 rollout routing replay]]。

> [!warning] 误区 6：recompute 总是值得
> recompute 消数值型 TIM 但多一次 forward。大模型下这是不小算力开销。若 TIM 本身小（同 dtype + bitwise 对齐算子），可不用 recompute 直接用 rollout 的 logp。需权衡 TIM 严重度与 recompute 成本。verl/OpenRLHF 默认开（保稳）。

> [!warning] 误区 7：FP8 推理与 BF16 训练的 logp 差是 $O(\epsilon)$ 可忽略
> 不可。FP8 KV cache 量化让历史 KV 带误差，attention score 累积放大，logp 差可能远超 $10^{-3}$。FP8 路径是 TIM 严重度"高"档。若开 FP8 KV 必须 recompute 或对齐。详见 [[FP8量化方案]] §3.4。

> [!tip] 实践：TIM 治理优先级
> 1. **统一精度**：训推同 dtype（都 BF16，不开 FP8 KV），或 MXFP8 统一方案——治精度路径，成本最低。
> 2. **bitwise 算子对齐**：训推用同版 FlashAttention/同 RoPE base——治算子/融合，中成本。
> 3. **recompute**：trainer 用 $\theta_{old}$ 重算 old_log_prob——治全部数值型，高成本（多 forward），verl/OpenRLHF 默认开。
> 4. **MoE R3**：推理记路由训练回放——治 MoE 结构性，MoE-RL 必须开。
> 5. **监控**：盯 `pearson_corr`/`pg_clipfrac`/`ppo_kl`，colocate 同步下 Pearson $>0.99$，clipfrac $<0.1$，KL 稳。


## 8. 延伸细节

### 8.1 recompute 策略的工程实现

verl/OpenRLHF 的 recompute：trainer 拿到 rollout 采的 $(q,o)$ 后，不直接用 vLLM 记的 `old_log_prob`，而是**用 sync 时的 $\theta_{old}$ 在 trainer 引擎重新 forward 一遍**算 `old_log_prob`。这样 old/new/ref 三端都出自 trainer 引擎，彻底消除数值型 TIM。代价是多一次 forward（用 $\theta_{old}$），但保训练稳定性。这是"用算力换一致性"的工程决策，主流框架默认开。

### 8.2 MoE R3 的张量契约与传递

R3（Rollout-Routing-Replay，Ma et al. 2025 arXiv:2510.11370）的工程细节（来自 [[R3 rollout routing replay]] 已批注解答）：
- **记录对象**：每 token 每层 TopK 专家**索引**（int32），不记 logits/gating。
- **张量契约**：`(seq-1, n_layers, topk)`（seq-1 因首 token 无路由）。
- **SGLang**：`--enable-return-routed-experts` 原生支持记录。
- **vLLM**：需 patch + `--no-async-scheduling`（异步调度会乱序路由记录）。
- **传递**：随轨迹经 TransferQueue/packing/CP 切片传到 Megatron 的 `RouterReplay` 安装 mask。
- **回放**：gating 仍用训练态 $s_{train}$ 算（保梯度），但 expert 选择 mask 成推理选的路由。
- **R1/R2/R3 分档**：R3 是最严格档（强制回放），R2 训练端记录，R1 宽松。dense 模型不需要。

详见 [[R3 rollout routing replay]]。

### 8.3 与 speculative decoding 的 TIM 叠加

若 rollout 用 speculative decoding（[[speculative decoding]]）加速采样，draft model 与 target model 的 logp 差异会**叠加进 TIM**——draft 参与生成的 token 其 logp 受 draft-target 一致性影响。weight sync 推新 $\theta$ 给 target 后，draft head 若还指向旧 $\theta$ 的 feature 轨迹，acceptance 崩 + logp 偏差。verl PR #5925 的 fix：sync 后必须重载 draft model 权重 + 重建 RoPE cos_sin cache。verl-SpeLo 更进一步——训练时就让 draft 跟 $\theta$ 一起更新。详见 [[speculative decoding]] §8.4。

### 8.4 与监控指标的耦合

[[RLHF训练监控指标详解]] 的几个指标是 TIM 的直接仪表：
- `pearson_corr`：rollout 与 trainer 逐 token logp 的 Pearson，TIM 的核心指标。colocate 同步下应 $>0.99$。
- `pg_clipfrac`：PPO clip 触发率。$\theta$ 没大动却 clipfrac 高 → TIM 让 ratio 偏离 1 触发 clip。
- `ppo_kl`：KL to ref。TIM 污染 KL 估计 → KL 失真。

排查 TIM：`pearson_corr` 跌 + `pg_clipfrac` 涨 + `ppo_kl` 异常 → 查精度路径/算子/MoE 路由。详见 [[RLHF训练监控指标详解]] §3.3。

### 8.5 内容来源

TIM 来源分层与衡量指标来自 [[训推不一致]] 已联网核实的 QA 解答（精度路径/算子实现/kernel融合/归约顺序/KV量化/RoPE base/MoE路由，Pearson>0.99 + max_abs_diff<1e-3）。MoE R3 来自 [[R3 rollout routing replay]] 已批注解答（arXiv:2510.11370，SGLang `--enable-return-routed-experts`/vLLM patch+`--no-async-scheduling`/张量契约/`RouterReplay`）。recompute 策略与 verl/OpenRLHF 实现来自 [[trainer与rollout引擎组合]] §4.4 与 [[RL权重同步]] §5.4。ratio 经 $\exp$ 指数放大的推导基于 [[importance sampling与off-policy correction]] §4。监控指标耦合来自 [[RLHF训练监控指标详解]]。speculative decoding TIM 叠加来自 [[speculative decoding]] §8.4。FP8 路径严重度来自 [[FP8量化方案]] §3.4。截至 2026-07。

---
相关: [[logprob一致性]] | [[训推不一致]] | [[importance sampling与off-policy correction]] | [[RL权重同步]] | [[trainer与rollout引擎组合]] | [[R3 rollout routing replay]] | [[FP8量化方案]] | [[MoE]] | [[PPO clipped objective]] | [[KL penalty与KL control]] | [[policy collapse]] | [[KL explosion]] | [[RLHF训练监控指标详解]] | [[stale policy problem]] | [[speculative decoding]] | [[rollout worker]] | [[17-RL训推一体框架]]
