# KSM与Keep-Routing

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §83
> **别名**: KSM / Keep Sampling Mask / Keep Routing / 截断 mask 复用 / 路由复用
> **难度**: 高（需懂 [[训推不一致]]、[[R3 rollout routing replay]]、[[MoE路由]]、top-p/top-k 采样、[[GRPO体系|GRPO]]）
> **来源**: DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, arXiv:2512.02556 (2025-12). 这是技术报告（非独立方法论文），机制描述为高级别，精确阈值未完全公开。

## 1. 一句话定义

**KSM（Keep Sampling Mask）/ Keep Routing** 是 DeepSeek-V3.2 技术报告里的一组工程手段：在 rollout（推理引擎）阶段记录 **top-p/top-k 截断 mask** 与 **MoE 路由选择**，在训练阶段**复用同一份 mask/路由**，使训练与采样的"动作子空间"完全一致——既消除采样截断引入的概率质量重分配（KSM），又消除 MoE 路由训推跳变（Keep Routing，沿用自 DeepSeek-V3-0324，对 MoE RL 稳定性"至关重要"）。

## 2. 为什么需要它（动机与背景）

### 2.1 top-p/top-k 截断破坏 IS 假设

top-k 选 k 个最高概率 token，其余置零，再**归一化**幸存 token 使其和为 1。这产生**条件分布** $q(y|x)$，与模型原始 softmax $\pi(y|x)$ 不同：

$$q(y|x)=\frac{\pi(y|x)\cdot\mathbf{1}[y\in\text{TopK}]}{\sum_{y'\in\text{TopK}}\pi(y'|x)}\ne\pi(y|x)$$

rollout 实际从 $q$ 采样，训练却用 $\pi$ 算梯度。IS 要求 $\pi(y|x)/q(y|x)$，但多数框架不显式算截断修正，把 $q$ 当 $\pi$ → **系统性偏差**。截断掉的概率质量被重分配给幸存 token，训练端看不到这个重分配。

### 2.2 MoE 路由训推跳变

见 [[R3 rollout routing replay]] §2.1、[[训推不一致]] §3.3：同权重同输入，vLLM 与训练引擎的 MoE TopK 可能选**不同专家** → 结构性 TIM → $\rho$ 失真 → 训练崩。

### 2.3 DeepSeek 的"复用"思路

不重算、不修正，直接把 rollout 的离散决策（截断 mask + 路由索引）**带进训练**，让训练走同一动作子网络。与 [[R3 rollout routing replay]] 的路由复用同源（DeepSeek-V3-0324 起就这么做，R3 论文是独立学术化表述）。

## 3. 核心概念详解

### 3.1 KSM（Keep Sampling Mask）

> "We preserve the truncation masks during sampling from $\pi_{old}$ and apply them to $\pi_\theta$ during training" — DeepSeek-V3.2 报告

机制：rollout 时 top-p/top-k 产生 0/1 mask $M$（哪些 token 在采样动作空间内）；训练算 $\pi_\theta$ 时套同一 $M$，使 $\pi_\theta$ 也只在同一动作子空间内归一化。两策略共享"identical action subspaces"。实测 top-p + KSM "effectively preserves language consistency during RL training"。

### 3.2 Keep Routing

> "We preserve the expert routing paths used during sampling and enforce identical routing during training" — DeepSeek-V3.2 报告

机制：rollout 记每 token 每 MoE 层的专家选择，训练 forward **强制走同一组专家**（与 [[R3 rollout routing replay]] 的 $I_{\text{infer}}$ 回放一致）。保证"identical expert parameters are optimized"。报告称自 V3-0324 起采用，对 MoE RL 稳定性至关重要。

### 3.3 Off-Policy 序列 masking（相关，但对象不同）

报告另一项：对**负 advantage 且 KL 超阈**的序列 mask 掉：

$$M_{i,t}=0\quad\text{when}\quad \hat A_{i,t}<0\ \wedge\ \frac1{|o_i|}\sum_t\log\frac{\pi_{old}(o_{i,t})}{\pi_\theta(o_{i,t})}>\delta$$

注意求和项是**逐 token log-ratio 的算术平均** = 逐 token importance ratio **几何均值**的 log。即"对负 advantage + 序列级几何均值 IS 偏离超阈的序列丢弃"。动机：模型从自己的错误学得最多，但高度 off-policy 的负样本有害。

### 3.4 无偏 KL 估计

报告用 importance ratio 修正 Schulman K3 估计器，得**无偏 KL**：原 K3 在 $\pi_\theta\ll\pi_{\text{ref}}$ 时给"不成比例的无界权重"，修正后无偏。不同域偏好不同 KL 强度；数学域甚至完全去掉 KL 有时更好。

## 4. 数学原理 / 公式

### 4.1 KSM 动作空间一致

设 top-p/top-k 截断 mask $M(\cdot|x)$（动作子空间 $\mathcal{A}_M\subset$ vocab）：

$$q(y|x)=\frac{\pi_{old}(y|x)M(y|x)}{Z_{old}},\quad \tilde\pi_\theta(y|x)=\frac{\pi_\theta(y|x)M(y|x)}{Z_\theta}$$

KSM 让训练用 $\tilde\pi_\theta$（同 mask 归一化）而非 $\pi_\theta$，使 IS ratio $\tilde\pi_\theta/q$ 的动作空间对齐（$Z$ 不同但样本支撑集相同）。

### 4.2 Keep Routing 回放

$$I_{\text{infer}}=\text{TopKMask}(s_{\text{infer}},K),\quad g_{\text{replay}}=\text{softmax}(s_{\text{train}})\odot I_{\text{infer}}$$

与 [[R3 rollout routing replay]] §4.3 一致：mask 来自推理（常数），softmax 用训练 logits（保梯度）。

### 4.3 序列级几何均值 IS

$$\bar\rho_i^{\text{geo}}=\exp\!\left(\frac1{|o_i|}\sum_t\log\frac{\pi_\theta(o_{i,t})}{\pi_{old}(o_{i,t})}\right)=\left(\prod_t\rho_{i,t}\right)^{1/|o_i|}$$

负 advantage 且 $\bar\rho$ 偏离大 → mask 该序列。这与 [[GSPO]] 的序列级几何均值 ratio 同构——KSM 把它用于 off-policy 检测而非 PPO ratio 本身。

## 5. 代码示例

```python
import torch
# KSM: rollout 的 top-k mask 带入训练
logits_old = torch.randn(1, 50)        # π_old 的 logits
probs_old = torch.softmax(logits_old, -1)
topk_val, topk_idx = probs_old.topk(10, -1)
mask = torch.zeros_like(probs_old).scatter_(-1, topk_idx, 1.0)  # 动作子空间

# rollout 从截断归一化分布 q 采样
q_old = (probs_old * mask) / (probs_old * mask).sum(-1, keepdim=True)
# 训练时: π_θ 也套同一 mask 归一化 (KSM)
logits_new = torch.randn(1, 50)        # π_θ
probs_new = torch.softmax(logits_new, -1)
q_new = (probs_new * mask) / (probs_new * mask).sum(-1, keepdim=True)
# IS ratio 在同动作子空间内: q_new / q_old (而非 π_new/π_old)
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[训推不一致]]（KSM/Keep Routing 治采样截断与路由两类 TIM）、[[R3 rollout routing replay]]（Keep Routing 与 R3 同源，R3 是学术化独立表述）、[[MoE路由]]、[[importance sampling与off-policy correction]]（截断破坏 IS 假设）
- **下游（应用）**: [[训练推理不一致TIM技术综述]]（DeepSeek 的"复用"路线代表）、DeepSeek-V3.2 RL 训练稳定性
- **对比 / 易混**:
  - **KSM vs [[R3 rollout routing replay]]**：R3 复用 MoE 路由 mask；KSM 复用采样截断 mask。Keep Routing ≈ R3 的路由部分。两者在 DeepSeek 体系里是同一"复用"哲学的两个对象。
  - **KSM vs [[IcePop]]**：IcePop mask 的是**训推概率比偏离的 token**（梯度可信度 gate）；KSM mask 的是**采样动作子空间**（让训推动作空间一致）。不同 masking 对象。
  - **序列级几何均值 masking vs [[GSPO]] 序列级 ratio**：KSM 用几何均值做 off-policy 检测丢弃；GSPO 用几何均值做 PPO ratio clip。同构不同用途。

## 7. 常见误区与易错点

> [!warning] 误区 1：以为 KSM = R3
> 不等价。KSM 复用**采样截断 mask**（动作子空间），Keep Routing 才复用 MoE 路由（≈R3）。报告把两者并列，但对象不同。

> [!warning] 误区 2：截断 mask 复用就消了 IS 偏差
> KSM 让训推动作空间一致，但 $Z_{old}\ne Z_\theta$（归一化常数不同），IS ratio 仍需正确算 $\tilde\pi_\theta/q_{old}$。KSM 是"动作空间对齐"，不替代 IS 本身。

> [!warning] 误区 3：把技术报告描述当精确算法
> DeepSeek-V3.2 是技术报告，KSM/Keep Routing 机制是高级别描述，**精确阈值与算法细节未完全公开**。工程复现需参考 R3 的实现（[[R3 rollout routing replay]] §8）。

## 8. 延伸细节

### 8.1 与 R3 论文的关系
R3（arXiv:2510.11370）**未引用** DeepSeek "Keep Routing"——两者是独立/并行表述同一思路。Keep Routing 是 DeepSeek 自 V3-0324 起的工程实践，R3 是学术化命名与系统实验。详见综述 §3、[[R3 rollout routing replay]]。

### 8.2 局限
- 无独立 ablation 隔离 KSM/Keep Routing 与其他 RL 稳定手段
- 阈值未公开
- 作为技术报告而非方法论文，可复现性依赖 R3/verl 实现

### 8.3 内容来源
KSM/Keep Routing 机制、序列级几何均值 masking、无偏 KL 修正均来自 arXiv:2512.02556（2025-12-02 提交）全文已联网核实。R3 与 Keep Routing 的关系澄清来自 arXiv:2510.11370 全文。截至 2026-09。

---
相关: [[训练推理不一致TIM技术综述]] | [[训推不一致]] | [[R3 rollout routing replay]] | [[MoE路由]] | [[IcePop]] | [[GSPO]] | [[importance sampling与off-policy correction]] | [[rollout train reference logprob一致性]] | [[17-RL训推一体框架]]
