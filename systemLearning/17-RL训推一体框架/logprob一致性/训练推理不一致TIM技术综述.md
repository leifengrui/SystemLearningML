# 训练推理不一致TIM技术综述

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §83
> **别名**: TIM Survey / Training-Inference Mismatch 技术综述 / 训推不一致方法图谱
> **难度**: 高（需懂 [[训推不一致]]、[[rollout train reference logprob一致性]]、[[importance sampling与off-policy correction]]、[[R3 rollout routing replay]]、[[GRPO体系|GRPO]]、[[MoE路由]]）
> **定位**: 本综述是 TIM 问题域的**方法图谱与选型指南**，聚焦 2025–2026 年密集发表的新方法及其传承演化。TIM 的**机理**（来源分层/指标/影响四级）见 [[训推不一致]]，**在 RL 的后果与 recompute/R3 治理**见 [[rollout train reference logprob一致性]]，**IS/clip/staleness 数学**见 [[importance sampling与off-policy correction]]——本综述与四篇互链、不重写它们。

## 1. 一句话定义与三层根因框架

**训练-推理不一致（Training-Inference Mismatch, TIM）** 指 LLM RL 中 rollout（推理引擎 vLLM/SGLang）与 trainer（训练引擎 FSDP/Megatron）即便加载同一权重，对同一输入算出的 $\log\pi$ 数值不等，使 PPO/GRPO 的 $\rho=\exp(\log\pi_\theta-\log\pi_{old})$ 在 $\theta$ 未动时偏离 1，污染 IS 修正、污染梯度，轻则效率降、重则崩溃（MoE 尤甚）。其根因分**三层**，每层对应一族治理方法：

| 层 | 根因 | 典型现象 | 代表方法（本综述覆盖） |
|---|---|---|---|
| **① 引擎层** | kernel 实现/精度/归约顺序/MoE 路由/采样截断的数值差 | 同 $\theta$ 两套引擎 logp 不等 | [[FP16-fix]]、[[VeXact]]、[[Bitwise-consistent-RL]]、[[KSM与Keep-Routing]]、[[R3 rollout routing replay]]、[[IcePop]] |
| **② 版本/staleness 层** | rollout 用旧 $\theta$、trainer 用新 $\theta$；异步 partial rollout；bypass 跳过重算 | $\rho$ 偏离 1（本应偏离但需正确修正） | [[TIS]]、RS/MIS、SIS、[[TRM]]、AReaL/Prime-RL |
| **③ 优化层** | 多 epoch 复用制造 off-policy；ratio clip 在长尾词表结构性有偏 | 低概率 token 过罚、高概率 token 欠罚 | [[GSPO]]、[[DPPO]]、[[DRPO]] |

> [!note] TIM vs staleness 的正交（关键）
> **TIM（引擎层）**：同 $\theta$ 下两套引擎算出不同 logp，$\rho$ **错误偏离** 1（应消除）。
> **staleness（版本层）**：rollout 用旧 $\theta$、trainer 新 $\theta$，$\rho$ **正确偏离** 1（IS 该修正）。
> 两者都表现为 $\rho\ne1$ 但根因解法不同。详见 [[rollout train reference logprob一致性]] §3.5、[[importance sampling与off-policy correction]] §3.7。本综述的三层划分刻意把"引擎差"与"权重差"分开。

## 2. 引擎层根因（为什么 rollout ≠ train 的分布）

> 机理细节见 [[训推不一致]] §3.1。本节给方法图谱视角的要点。

### 2.1 kernel 实现与归约顺序差异
vLLM 用 PagedAttention（非 contiguous KV block）+ FlashInfer 优化 kernel；SGLang 用 RadixAttention；Megatron 用 FlashAttention + TransformerEngine。算子融合（SiLU MLP、带残差 RMSNorm）改浮点累加顺序。**Thinking Machines（Horace He, 2025-09）** 与 **VeXact（arXiv:2605.14220）** 独立诊断到同一结论：单 kernel run-to-run 是确定的，真正非确定性来自**批量变化触发不同归约策略**（matmul Split-K、RMSNorm 分卡归约、attention split-size）——缺 **batch invariance**。浮点非结合性 $(a+b)+c\ne a+(b+c)$ 在自回归多步中放大。

### 2.2 量化 rollout 的偏差量级
FP8/INT8 KV cache 或权重量化的 rollout 与 BF16/FP32 训练精度路径不同。FlashRL 实测 DAPO-32B rollout 占 ~70% RL 训练时间，量化加速但引入 off-policy 偏差；TIS 修正后 INT8 rollout 可匹配 BF16 性能（Qwen2.5-32B AIME2024 = 50）。

### 2.3 MoE 路由离散跳变（与 dense 的本质区别）
dense 模型每个 token 走全部参数，数值扰动产生**连续、成比例**的输出差（$\mathcal{O}(\epsilon)$）。MoE 有 router TopK **离散选择**：边界 token（第 K 与 K+1 专家分数接近）一点数值扰动就**跳专家** → 走完全不同子网络 → 输出差 $\mathcal{O}(1)$。R3 论文（arXiv:2510.11370）实测 Qwen3-30B-A3B 训推 KL $\approx1.5\times10^{-3}$，dense Qwen3-8B $\approx6.4\times10^{-4}$，MoE 是 dense 的 ~2.4 倍，~10% router 训推选不同专家。

### 2.4 top-p/top-k 截断的概率质量去向
top-k 选 k 个最高概率 token，其余置零，**归一化**幸存者使和为 1。产生条件分布 $q(y|x)\ne\pi(y|x)$。rollout 从 $q$ 采样，训练却用 $\pi$ 算梯度，截断掉的概率质量被重分配给幸存 token，训练端看不到 → 系统性偏差（违反 IS 假设）。DeepSeek-V3.2 的 [[KSM与Keep-Routing|KSM]] 直接复用截断 mask 让训推动作空间一致。

## 3. 引擎层方法

### 3.1 [[FP16-fix]] — 精度根治（轻量）
- **机制**：BF16（7 位 mantissa）→ FP16（10 位，多 3 位 = 精度 8×），把序列级 logp 比值 mismatch 降 **~24×**（arXiv:2510.26788 Fig.2）。
- **结果**：sanity test（可完美 MATH）BF16 GRPO 崩 → FP16 全算法 ~99%；AIME24 34%→39%。
- **局限**：FP16 动态范围小（$\sim\pm65504$）极端模型可能溢出；MoE 路由结构型 TIM 仍需 R3；不归零（VeRL 偶现尖峰）。
- **关键纪律**：**训推必须同改 FP16**，单改一边无效（mismatch 是两引擎的差）。

### 3.2 [[VeXact]] — 零 mismatch 诊断标尺
- **机制**：verl 上构建 rollout 引擎复用训练引擎 HuggingFace kernel + batch-invariant 确定性 kernel，实现 bit 级一致，作 ground truth 量化 vLLM 的 TIM 量级。
- **诊断结论**：TIM 来源两类 = ① kernel 实现差异 ② 归约顺序/tiling 差异（与 Thinking Machines 一致）。recompute 下前 700 步 $K_1/K_3$ 平但 reward 已降（KL 估计器不灵敏）。
- **局限**：吞吐低，定位诊断/校准工具而非生产。

### 3.3 [[Bitwise-consistent-RL]] — kernel 根治（最严最贵）
- **机制**：vLLM batch-invariant kernel 注入 TorchTitan 训练 forward + 自定义 backward，训推共享 kernel → KL **恒为 0**。
- **定位**：**替代非互补**——bit 一致后算法层补丁（IS/KL/clip）无对象。
- **代价**：2.4× 慢；当前需 eager 模式、双份模型代码、仅 Qwen3-1.7B demo。
- **与算法层关系**：从"打补丁"转向"修根因"的哲学转向。

### 3.4 [[KSM与Keep-Routing]] — DeepSeek 的"复用"路线
- **KSM（Keep Sampling Mask）**：rollout 的 top-p/k 截断 mask 带入训练，训推动作子空间一致（arXiv:2512.02556）。
- **Keep Routing**：rollout 记 MoE 路由，训练强制走同组专家（≈[[R3 rollout routing replay]]，DeepSeek-V3-0324 起采用，对 MoE RL 稳定性"至关重要"）。
- **off-policy 序列 masking**：负 advantage 且序列级几何均值 IS 超阈 $\delta$ 的序列丢弃。
- **无偏 KL 估计**：用 IS ratio 修正 Schulman K3，原 K3 在 $\pi_\theta\ll\pi_{ref}$ 时无界权重。
- **局限**：技术报告（非独立方法论文），精确阈值未公开，无独立 ablation。

### 3.5 [[IcePop]] — 算法层 token masking（引擎层兜底）
- **机制**：双边 mask 训推概率比 $k=\pi_{\text{train}}/\pi_{\text{infer}}$，$k\in[\alpha,\beta]=[0.5,5.0]$ 保留，超阈 token 梯度置零（arXiv:2510.18855, Ring-1T）。裁剪仅 ~0.2%。Ring-mini-2.0 AIME +14%。
- **局限**：固定阈值假设误差均匀（为 KPop 铺垫）；丢弃数据不修根因；MoE 建议叠加 R3。

> [!note] 引擎层方法关系
> - **根治派**：FP16（精度）、Bitwise（kernel）、KSM/Keep-Routing + R3（复用离散决策）——消根因。
> - **兜底派**：IcePop（mask 分歧 token）、TIS（截 IS 上限）——算法层补偿。
> - **诊断派**：VeXact——测真值校准阈值。
> 实践常**根治 + 兜底叠加**：FP16/R3 降 baseline，IcePop/TIS 兜底残余。

## 4. 版本/staleness 层根因与方法

### 4.1 异步与 partial rollout
异步 RL 把生成与训练解耦，rollout 用旧 $\theta$、trainer 用新 $\theta$，staleness $k$ 让 $\rho=\exp(k\cdot g_t)$ 指数偏离。partial rollout 指一个 batch 跨多 $\theta$ 版本（前半用 $\theta_{old-k}$、sync 后后半用 $\theta_{old}$）。
- **AReaL**（arXiv:2505.24298）：可中断 rollout worker（中途加载新权重续采）、staleness-enhanced PPO，2.77× 加速；A-3PO（arXiv:2512.06547）近似 proximal policy 省 forward，1.8× 加速。
- **Prime-RL**（PrimeIntellect GitHub）：全分布式纯异步，训 INTELLECT-3（106B MoE，512 H200）。
- **LLA-MARL**：**未找到可靠来源**（最接近 LlamaRL arXiv:2505.24034，非同一物）。

### 4.2 bypass 后无法回放 old_log_prob
verl `bypass_mode=true` 跳过 trainer 用 $\theta_{old}$ 重算 `old_log_prob`，直接用 rollout 引擎算的（省一次 forward），但 IS 权重分母带 TIM，无法准确算重要性权重。`bypass_mode=false` + recompute 时挂 TIS/MIS 修正残余。详见 [[rollout train reference logprob一致性]] §8.1、[[TIS]] §8.2。

### 4.3 IS baseline 失效
朴素 IS 无偏但**高方差**：$\rho$ 在 $\pi_{old}(x)\to0$ 而 $\pi_\theta(x)>0$ 时 $\to\infty$。长训练中方差累积让 IS 估计噪声大到不可用。**"Beyond Precision"（arXiv:2602.01826）未找到可靠来源**——IS 失效的优化视角分析散见于 GIPO（2603.03955）、ESTR（2607.22186）、DPPO（2602.04879）。

### 4.4 [[TIS]] — 截断 IS 上限（baseline）
$\tilde w_t=\min(\rho_t,C)$，verl 默认 $C=2.0$，只截上限、不丢 token。R3 论文实测 TIS **延缓但不治** MoE 崩溃（崩得晚但仍崩）；DPPO 论文实测 TIS **恶化** DPPO 稳定性（低概率 token 方差最大，最常被截反而放大偏差）。来源是 Yao Feng Notion 博客（非 arXiv）+ verl PR #2953。

### 4.5 RS / MIS — 双边丢弃
verl MIS：$\rho\in[1/C,C]$ 内保留，超阈**整 token 置零**（丢弃），"本质是拒绝采样"。**RS 作为"Li et al. 2026"独立论文未找到可靠来源**——概念散落在 verl MIS / IcePop / DeepSeek-V3.2 各自实现里。
- **IcePop 的 MIS**：token 级，基于训推概率比分歧。
- **DeepSeek-V3.2 的序列 masking**：序列级，负 advantage + 几何均值 IS 超阈。
- 两者**不同类**（粒度、判据、advantage 条件都不同）。

### 4.6 SIS — 序列级 IS
**SIS 作为独立论文（"Li et al. 2025"）未找到可靠来源**。最接近的实质内容：
- [[GSPO]]（arXiv:2507.18071）的序列级几何均值 ratio $s_i$。
- **VESPO**（arXiv:2602.10693）的 bias-variance 理论：naive IS 无偏高方差（序列级方差随长指数爆）；token 级 IS 有偏低方差；GSPO 长归一引偏差换方差可控——这就是"Qwen team 视角的 token 级 IS 是有偏低方差估计器"的出处（实由 VESPO 系统化，非单独 SIS 论文）。

### 4.7 [[TRM]] — 序列级 trust region masking（长程）
- **动机**：经典 $O(T^2)$ trust region 界在 $T=4096$ 时 $\approx1677$（reward $\in[0,1]$，vacuous）。
- **机制**：KL 路（Pinsker，子线性 $\sqrt t$）+ TV 路（耦合界，对集中偏差紧）按位置取更紧者，**合一严格更优**（$T=4096$：仅 KL $\le8.2$，KL+TV $\le4.1$）。序列级 binary mask $M(x,y)=\mathbf{1}[\max_t D_{KL}(c_t)\le\delta]$，超阈整序列零梯度。
- **局限**：长度偏差（拒绝概率随 $T$ 增，惩罚长 reasoning）；高拒绝率降样本效率。

## 5. 优化层根因与方法

### 5.1 多 epoch 复用 = 人为制造的 off-policy
同一 rollout batch 跑 `ppo_epochs>1`，第一个 mini-batch 后 $\theta$ 已变，后续 mini-batch/epoch 全是**陈旧策略**采的——制造 off-policy。$\rho=\pi_\theta/\pi_{\theta_{old}}$ 修正，但方差随 staleness 增。多源佐证：DPPO §5（即便 lr=$10^{-6}$ 仍累积）、MiniRL（arXiv:2512.01374，mini-batch staleness+引擎 mismatch 分类）、VESPO、verl（`train_batch_size=ppo_mini_batch_size`+`ppo_epochs=1` 才 on-policy）、BAPO。

### 5.2 PPO ratio clip 的长尾结构性偏差
PPO 的 $|r_t-1|\le\varepsilon$ 是真实 TV 散度 $\frac12\mathbb{E}[|r_t-1|]$ 的**单样本 MC 估计**。DPPO 论文 Fig.2 bound 论证：
- 低概率 token：$\mu=10^{-4},\pi=10^{-2}\Rightarrow r=100$ → 被裁，**过罚**，仅移 0.0099 质量
- 高概率 token：$\mu=0.99,\pi=0.80\Rightarrow r\approx0.808$ → 在区间内，**欠罚**，移 0.19 质量

### 5.3 clip 族演化：PPO-clip → GSPO → DPPO → DRPO
- **PPO-clip**（Schulman 2017）：token 级 ratio 双裁 $[1-\varepsilon,1+\varepsilon]$。局限：(a) $A<0$ 无下界→dual-clip；(b) 抑制低概率 token 探索→entropy collapse；(c) token 级 MC 噪声；(d) 长序列方差爆炸。
- **Dual-Clip PPO**：$A<0$ 时加下界 $cA$。TRL/MOSS-RLHF 标准。仍 symmetric。
- **DAPO**（arXiv:2503.14476）：Clip-Higher 解耦 $\varepsilon_{low}/\varepsilon_{high}$ + dynamic sampling + token-level loss + overlong shaping，去 KL。仍 token 级 ratio。
- **[[GSPO]]**（2507.18071）：序列级几何均值 ratio $s_i$，clip 作用整 response。解方差爆炸/MoE 专家跳变。**遗留**：仍 ratio 代理，长尾偏差未解，$\varepsilon$ 无原则选择。
- **[[DPPO]]**（2602.04879）：散度（Binary-KL/Binary-TV/Top-K）替代 ratio，自适应界 $|r-1|\le\delta/\mu$ 抵消长尾偏差；无需 R3 也稳 MoE。**遗留**：硬 mask 边界处丢弃而非修正，$\varepsilon$ 需 per-setting 调。
- **[[DRPO]]**（2606.09821）：硬 mask→平滑 advantage 加权二次正则 $\frac{|\hat A|}{2\delta}\mu(r-1)^2$，$w_t\in[1-1/\delta,1+1/\delta]$ 连续不归零；$\ell_2^2$ 替代 $\chi^2$（低概率 token 不过敏）；universal $\delta=12.5$。暴露 DPPO 硬 mask 局限。

## 6. 技术演化时间线（2025 中 ~ 2026 中）

> 每个方法标注**解决的上代局限**。三条传承线：引擎层（IcePop→KPop）、版本层（TIS→SIS→TRM）、优化层（PPO-clip→GSPO→DPPO→DRPO）。

```
2025-05  AReaL (2505.24298) ── 异步 RL 系统代表，partial rollout / staleness-enhanced PPO
2025-07  GSPO  (2507.18071) ── 解 GRPO token 级 ratio 长序列方差爆炸 (MoE ~10% 专家跳变)
2025-09  Thinking Machines batch-invariance blog ── 澄清 TIM 根因=缺 batch invariance (非 GPU 并发)
2025-10  R3    (2510.11370) ── MoE 路由离散跳变 (IcePop/R3 同期, R3 治根因 vs IcePop 治症状)
2025-10  IcePop(2510.18855) ── 双边 token mask 训推概率比, Ring-1T 稳定; 局限=固定阈值
2025-10  FP16-fix(2510.26788)── 精度根治 (BF16→FP16, mismatch↓24×), 引擎层轻量根治
2025-11  Bitwise-consistent vLLM blog ── kernel 根治 (KL=0), 替代算法层, 2.4× 慢
2025-12  DeepSeek-V3.2 (2512.02556) ── KSM/Keep Routing "复用"路线 + 序列几何均值 masking
2025-12  TRM   (2512.23075) ── 解 O(T²) trust region 界长程 vacuous; KL+TV 合一
2025-12  MiniRL(2512.01374) ── Qwen: mini-batch staleness + 引擎 mismatch 系统分类
2026-02  DPPO (2602.04879) ── 解 GSPO 遗留的 ratio-as-proxy; 散度替代 ratio, 自适应界抵消长尾偏差
2026-02  VESPO(2602.10693) ── IS bias-variance 理论 (token 有偏低方差, sequence 无偏高方差)
2026-05  VeXact(2605.14220) ── 零 mismatch 诊断标尺, TIM 来源两类系统化
2026-06  DRPO (2606.09821) ── 解 DPPO 硬 mask 边界丢弃; 平滑二次正则, universal δ=12.5
```

**传承脉络**：
- **引擎层**：FP16（精度降级）→ Bitwise（kernel 归零）+ KSM/R3（复用离散决策）；IcePop（算法兜底）→ KPop（自适应，**未找到来源**）
- **版本层**：TIS（截 IS 上限，延缓不治）→ SIS/GSPO（序列级，**SIS 无独立论文**）→ TRM（序列级 trust region mask，长程界）
- **优化层**：PPO-clip（token ratio）→ GSPO（序列 ratio，改粒度）→ DPPO（散度，改代理）→ DRPO（平滑正则，改连续性）

## 7. 对比表

| 方法 | 所属层 | 核心机制（一句话） | 解决的问题 | 计算开销 | 改引擎/框架 | 出处 | 已知局限 |
|---|---|---|---|---|---|---|---|
| **FP16-fix** | 引擎 | BF16→FP16（mantissa 7→10），训推同改 | 精度路径 TIM | 近零 | dtype 配置 | 2510.26788 | FP16 可能溢出；不归零；MoE 仍需 R3 |
| **VeXact** | 引擎(诊断) | rollout 复用训练 kernel，bit 一致做标尺 | 无 ground truth 难测 TIM | 低(诊断) | verl 改 | 2605.14220 | 吞吐低，诊断非生产 |
| **Bitwise-consistent** | 引擎 | vLLM batch-invariant kernel 注入 TorchTitan | kernel 差异（根治） | 2.4× 慢 | 训推都改 | vLLM blog 2025-11 | 慢；双份代码；模型支持窄 |
| **KSM/Keep-Routing** | 引擎 | 复用截断 mask + MoE 路由 | 采样截断 + 路由跳变 | 低 | 推理导出+训练回放 | 2512.02556 | 技术报告无阈值/ablation |
| **R3** | 引擎 | rollout 记 TopK mask，训练回放（softmax 保梯度） | MoE 路由结构型 TIM | <3% | 推理导出+训练回放 | 2510.11370 | 仅 MoE；工程坑多 |
| **IcePop** | 引擎(算) | 双边 mask 训推概率比 $k\in[0.5,5]$ | 训推分歧 token 噪声梯度 | 近零 | 否 | 2510.18855 | 固定阈值假设均匀；丢数据 |
| **TIS** | 版本 | $\min(\rho,C)$ 只截 IS 上限 | IS 高方差 | 近零 | 否 | blog+verl PR#2953 | 延缓不治 MoE 崩；恶化 DPPO |
| **MIS/RS** | 版本 | 双边 $\rho\in[1/C,C]$ 外丢 token | 极端 IS 权重 | 近零 | 否 | verl/IcePop/DeepSeek | 丢数据；无独立论文来源 |
| **SIS** | 版本 | 序列级几何均值 IS | token 级 IS 有偏 | 近零 | 否 | (无独立论文) | 序列级方差爆；来源=GSPO+VESPO |
| **TRM** | 版本 | 序列级 trust region mask（KL+TV 合一） | 长 horizon 界 vacuous | 中(logits 存) | 否 | 2512.23075 | 长度偏差；高拒绝降样本效率 |
| **GSPO** | 优化 | 序列级几何均值 ratio clip | token 级方差爆炸(MoE) | 近零 | 否 | 2507.18071 | 仍 ratio 代理；长尾偏差未解 |
| **DPPO** | 优化 | Binary-KL/TV/Top-K 散度 mask | ratio clip 长尾偏差 | 低 | 否 | 2602.04879 | 硬 mask 边界丢弃；$\varepsilon$ 需调 |
| **DRPO** | 优化 | 平滑二次正则 $w_t\in[1-1/\delta,1+1/\delta]$ | DPPO 硬 mask 不修正 | 低 | 否 | 2606.09821 | (当前最新,局限待观察) |
| **DAPO** | 优化 | Clip-Higher 解耦+dynamic sampling | PPO 抑制探索 | 近零 | 否 | 2503.14476 | 仍 token 级 ratio |
| **AReaL** | 版本(系统) | 异步解耦+staleness PPO | 同步 rollout 太慢 | 省(异步) | 系统级 | 2505.24298 | staleness 引入 off-policy |

## 8. 场景选型综合判断

### 场景①：dense 模型 + 同构引擎（如 Qwen3-8B dense，训推同框架）
**推荐**：**FP16-fix（训推同改）+ recompute 兜底**，算法层用 **GSPO 或 DPPO**，无需 R3。
- **理由**：dense 无路由跳变，TIM 主要是精度/算子数值型。FP16 把 mismatch 降 24× 已足够；recompute（trainer 用 $\theta_{old}$ 重算 old_log_prob）彻底消数值型 TIM（[[rollout train reference logprob一致性]] §3.6）。算法层 DPPO 的散度界对 dense 长尾词表也优于 ratio clip。R3 对 dense 无意义（无路由）。
- **若追求极致一致**：上 Bitwise-consistent（2.4× 慢可接受时）。

### 场景②：MoE 模型 + 异步训练（如 Qwen3-30B-A3B / DeepSeek-V3 风格 + AReaL）
**推荐**：**R3（路由复用）+ FP16 + DPPO/DRPO**，**不要单独用 IcePop**，TIS 慎用。
- **理由**：MoE 的结构型 TIM 是崩溃主因，必须 R3 治根因（KL 降一个数量级，<3% 开销）。FP16 降数值型 baseline。算法层 DPPO 论文实测**不加 R3 也稳 MoE**，但叠加 R3 正交增益；DRPO 更平滑、universal $\delta=12.5$ 免调。异步 staleness 让 $\rho$ 偏离大，DPPO 的散度界比 ratio clip 更鲁棒。
- **不推荐单独 IcePop**：固定阈值丢数据不修根因，R3 论文定位其为"治症状"。若用 IcePop 需叠加 R3。
- **不推荐 TIS**：DPPO 论文实测 TIS 恶化稳定性（截低概率高方差 token 放大偏差）；R3 后 TIS 无对象。
- **DeepSeek 风格混合路由**对 TIM 极敏感，R3 收益更大（[[R3 rollout routing replay]] §8.3）。
- **若启用 KSM/Keep-Routing**：DeepSeek-V3.2 路线与 R3 同源，可替代 R3 的路由部分，但需推理引擎支持导出。

### 场景③：极致吞吐量化 rollout（FP8/INT8 KV cache + 权重量化）
**推荐**：**FP16 对齐训练端 + R3（若 MoE）+ TIS/MIS 修正量化 off-policy + DPPO**；若 MoE 量化严重考虑 **Bitwise** 或退回 BF16。
- **理由**：量化 rollout 与 BF16/FP32 训练精度路径不同是 TIM 重灾区（[[训推不一致]] §3.1 精度路径行）。FlashRL 用 TIS 修正后 INT8 rollout 匹配 BF16 性能。但 TIS 在 MoE 上延缓不治，需叠加 R3。DPPO 的散度界对量化引入的分布偏移更鲁棒。
- **DeepSeek-V3 用 FP8 混合精度训练**（arXiv:2412.19437）保 $5.576M 成本，是量化+一致性协同的范例。
- **若量化致 MoE 路由严重失配**：FP16 压不住离散跳变，考虑 Bitwise-consistent 或退回 BF16 不量化。

> [!tip] 通用选型原则
> 1. **先治根因再兜底**：引擎层（FP16/R3/Bitwise/KSM）优先于算法层（TIS/IcePop/DPPO）。根因消了，算法层无对象。
> 2. **MoE 必须 R3 或 Keep-Routing**：路由离散跳变是 MoE 专属致命 TIM，算法层兜底（IcePop/TIS）延缓不治。
> 3. **散度优于 ratio**：长尾词表上 DPPO/DRPO 的散度界比 PPO/GSPO 的 ratio clip 结构性更优。
> 4. **异步需 staleness 修正**：AReaL/Prime-RL 的 staleness 让 $\rho$ 偏离大，散度界或 V-trace 截断兜底。
> 5. **诊断先行**：用 VeXact 或 verl 的 `pearson`/`max_abs_diff` 量化 TIM 量级，再选手段。

## 9. 引用清单

### 已联网核实
| 方法 | 出处 | arXiv/链接 |
|---|---|---|
| TIM 诊断/VeXact | Zhong et al., *Diagnosing TIM in LLM RL* | arXiv:2605.14220 |
| FP16-fix | Qi et al., *Defeating TIM via FP16* | arXiv:2510.26788 |
| Bitwise-consistent | vLLM Blog | vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference |
| batch invariance 根因 | Horace He, Thinking Machines Blog | 2025-09-10 |
| DeepSeek-V3.2 KSM/Keep-Routing | DeepSeek-AI | arXiv:2512.02556 |
| IcePop/Ring-1T | Ling Team, *Every Step Evolves* | arXiv:2510.18855 |
| R3 | Ma et al., *Stabilizing MoE RL by Aligning Routers* | arXiv:2510.11370 |
| GSPO | Zheng et al., *Group Sequence Policy Optimization* | arXiv:2507.18071 |
| DPPO | Qi et al., *Rethinking the Trust Region in LLM RL* | arXiv:2602.04879 |
| DRPO | Yao et al., *Rethinking the Divergence Regularization in LLM RL* | arXiv:2606.09821 |
| TRM | Li et al., *Trust Region Masking for Long-Horizon LLM RL* | arXiv:2512.23075 |
| DAPO | ByteDance+THU AIR | arXiv:2503.14476 |
| AReaL | Wei Fu et al., *AREAL* | arXiv:2505.24298 |
| A-3PO | Li et al., *Staleness-aware Proximal Policy Approximation* | arXiv:2512.06547 |
| VESPO | Shen et al., *VESPO* | arXiv:2602.10693 |
| MiniRL | Qwen team, *Stabilizing RL with LLMs* | arXiv:2512.01374 |
| DeepSeek-V3 FP8 训练 | DeepSeek-AI | arXiv:2412.19437 |
| TIS | Yao Feng, Notion blog + verl PR #2953 | （非 arXiv，部分来源）|
| Prime-RL | PrimeIntellect GitHub | github.com/PrimeIntellect-ai/prime-rl |

### 未找到可靠来源（明示）
| 项 | 状态 |
|---|---|
| **KPop** | arXiv 无记录；唯一科普文（知乎 zhuanlan.zhihu.com/p/2044073395222999883）403 不可读；CSDN 候选文未提 KPop。**未找到可靠来源**。二手描述（仅搜索摘要，未核 primary）：IcePop 后继，Ant Group/NUS，binary KL 自适应区域，masking 0.2%→10–30%。标 caution。 |
| **"Beyond Precision"** | arXiv:2602.01826 未找到匹配论文。IS 失效的优化视角散见 GIPO(2603.03955)/ESTR(2607.22186)/DPPO。 |
| **LLA-MARL** | 无精确匹配。最接近 LlamaRL(2505.24034)，非同一物。 |
| **SIS（独立论文）** | 无 "Li et al. 2025" 的 SIS 独立论文。实质内容由 GSPO(2507.18071) + VESPO(2602.10693) 承载。 |
| **RS/MIS（独立论文）** | 无 "Li et al. 2026" 的 RS 独立论文。概念散落 verl MIS / IcePop / DeepSeek-V3.2 各实现。 |

### 大纲错误前提修正
- R3 论文用 **Qwen3-30B-A3B**（非大纲写的 ZAYA1-8B）；R3 **未引用** DeepSeek Keep Routing；R3 **无** KSM。
- DPPO 正确 ID = **2602.04879**（非 2502.04879，后者是无关统计论文）。
- DRPO 正确 ID = **2606.09821**（非 2506.09821）；DRPO ≠ UniRL（同团队另一框架）。

---
相关: [[训推不一致]] | [[rollout train reference logprob一致性]] | [[importance sampling与off-policy correction]] | [[R3 rollout routing replay]] | [[IcePop]] | [[FP16-fix]] | [[VeXact]] | [[Bitwise-consistent-RL]] | [[KSM与Keep-Routing]] | [[TIS]] | [[TRM]] | [[GSPO]] | [[DPPO]] | [[DRPO]] | [[GRPO体系]] | [[token-level与sequence-level objective]] | [[MoE路由]] | [[asynchronous training]] | [[synchronous与asynchronous rollout]] | [[17-RL训推一体框架]]
