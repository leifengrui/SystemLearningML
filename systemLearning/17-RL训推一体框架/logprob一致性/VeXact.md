# VeXact

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §83
> **别名**: VeXact / Diagnosing TIM / 零 mismatch 诊断引擎
> **难度**: 高（需懂 [[训推不一致]]、[[rollout train reference logprob一致性]]、[[trainer与rollout引擎组合]]、verl）
> **来源**: Zhong et al., *Diagnosing Training Inference Mismatch in LLM Reinforcement Learning*, arXiv:2605.14220 (2026-05). 代码: https://github.com/verl-project/vexact

## 1. 一句话定义

**VeXact** 是一个基于 verl 构建的**零训推 mismatch 诊断引擎**：让 rollout 端复用训练引擎的 HuggingFace kernel 代码路径 + batch-invariant 确定性 kernel，实现"同权重同输入 → bit 级一致 logp"，用作**诊断标尺**——把它当 ground truth，量化 vLLM/SGLang 等推理引擎引入的 TIM 量级与训练崩溃关系，并校准 TIS/序列拒绝等算法层方法的阈值。

## 2. 为什么需要它（动机与背景）

### 2.1 没有 ground truth 就测不准 TIM

[[训推不一致]] §3.2 定义了 Pearson/max_abs_diff/KL/$F(\tau)$ 等指标，但它们都是"训 vs 推"的**相对差**。若想精确归因"训练崩溃里有多少来自 TIM、多少来自 staleness/算子"，需要一个**本身零 mismatch** 的 rollout 基线作 ground truth。vLLM/SGLang 为吞吐做了 kernel 优化（FlashInfer 等），与训练引擎不同源，无法当基线。

### 2.2 VeXact 的设计取舍

VeXact 不追求吞吐，追求**与 FSDP 训练引擎 bit 级一致**，同时保留可用吞吐（chunked prefill + CUDA Graph + pipeline parallelism + optimistic KV allocation）：
- 用训练引擎同款 HuggingFace kernel（不用 FlashInfer）
- batch-invariant kernel：固定 tiling 与归约顺序（RMSNorm、batched matmul、batch-invariant fused MoE）
- attention 确定性：禁用 KV splitting

## 3. 核心概念详解

### 3.1 TIM 来源的两类（论文诊断结论）

| 类 | 描述 | 例 |
|---|---|---|
| **① kernel 实现差异** | 推理用推理优化 kernel 库（FlashInfer），训练用另一套 | attention/RMSNorm/MoE 各自不同实现 |
| **② 归约顺序/tiling 差异** | 即便同实现，性能优化（atomic add、autotuning 按批量换 launch grid）引入非确定性；批量变 → grid 变 → 浮点非结合性 → 不同结果 | batched matmul、softmax over vocab reduce |

> [!note] 与 Thinking Machines 的"batch invariance"一致
> VeXact 的第二类根因与 Horace He（Thinking Machines, 2025-09 博客）的结论吻合：单 kernel run-to-run 是确定的，真正的非确定性来自**批量变化触发不同归约策略**——即缺 batch invariance。见 [[Bitwise-consistent-RL]]、综述 §2。

### 3.2 诊断量：recompute vs bypass

veRL 有 `bypass_mode`（见 [[rollout train reference logprob一致性]] §8.1、综述 §4）：bypass 时直接用 rollout 算的 `old_log_prob`（省一次 forward）但引入 TIM；recompute 时 trainer 用 $\theta_{old}$ 重算（消数值 TIM 但贵）。VeXact 定义修正比：

$$r_{\text{corr}}=\frac{\pi_{old}^{\text{train}}(a_t|s_t)}{\pi_{old}^{\text{rollout}}(a_t|s_t)}$$

- $r_{\text{corr}}\approx1$：两引擎一致；偏离即 TIM
- bypass 下 PPO ratio $r_{\text{ppo}}^{\text{rollout}}=\pi_\theta/\pi_{old}^{\text{rollout}}$ 把 TIM 混进分母
- recompute 下 $r_{\text{ppo}}^{\text{train}}=\pi_\theta/\pi_{old}^{\text{train}}$ 分母出自训练引擎，消 TIM

KL 估计器：$K_1(r)=-\log r$；$K_3(r)=(r-1)-\log r$（Schulman）。诊断用零中心损失贡献 $C(r_{\text{ppo}})=-(r_{\text{ppo}}-1)A_t$。

### 3.3 论文建议的补救组合
1. VeXact（源头零 mismatch）
2. token 级 TIS
3. 序列级拒绝（RS）
4. token + 序列联合
5. VeXact 当阈值校准工具

## 4. 数学原理 / 公式

### 4.1 token 级 mismatch

$$\delta_t=\log\pi_{old}^{\text{train}}(a_t|s_t)-\log\pi_{old}^{\text{rollout}}(a_t|s_t)$$

### 4.2 PPO ratio 的两种算法

$$r_{\text{ppo}}^{\text{train}}=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}^{\text{train}}(a_t|s_t)},\qquad r_{\text{ppo}}^{\text{rollout}}=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}^{\text{rollout}}(a_t|s_t)}$$

$$r_{\text{corr}}=\pi_{old}^{\text{train}}/\pi_{old}^{\text{rollout}}$$

### 4.3 联合 token + 序列拒绝阈值

token 级 TIS 阈 $\tau_{\text{tok}}=2$（$r_{\text{corr}}$ 上限），序列级拒 $\tau_{\text{seq}}=0.001$（序列 log-ratio 累积下限）。联合方案最接近 VeXact 基线。

## 4.4 实验崩溃点

| 模型 | 算法 | 引擎 | 行为 |
|---|---|---|---|
| Qwen3-30B-A3B (MoE) | REINFORCE | vLLM | ~step 280 退化（0.574→0.255） |
| 同上 | REINFORCE | **VeXact** | 训练 reward 0.753 / 验证 0.534 |
| Qwen3-1.7B (dense) | REINFORCE | vLLM | 失稳 |
| 同上 | REINFORCE | VeXact | 稳定 |
| Qwen3-30B-A3B | GRPO | vLLM recompute | 650 步内 0.87→0.40，后崩近 0（~step 1665） |
| 同上 | GRPO | VeXact | 维持 ~0.93 |

- recompute 下前 700 步 $K_1/K_3$ 仍平但 reward 已降（说明 KL 估计器不灵敏）
- bypass 下 $K_1/K_3$ 明显上升

## 5. 代码示例

```python
import numpy as np
# VeXact 思路: 用训练引擎同款 kernel 做 rollout, 当 ground truth
# 诊断 vLLM 引入的 TIM
logp_vexact = np.random.randn(1000)*2 - 5      # ground truth (零 mismatch)
logp_vllm   = logp_vexact + np.random.randn(1000)*0.01  # vLLM 引擎差
r_corr = np.exp(logp_vexact - logp_vllm)       # 修正比
delta_t = logp_vexact - logp_vllm
print(f"r_corr mean={r_corr.mean():.4f} std={r_corr.std():.4f}")
print(f"超出 τ_tok=2 的 token 占比: {((r_corr>2)|(r_corr<0.5)).mean():.4f}")
# TIS: w_t = min(r_corr, 2.0); 序列拒绝: 若 sum(log r_corr) < log(0.001) 则丢整条
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[训推不一致]]（TIM 机理与指标，本条是其诊断标尺）、[[rollout train reference logprob一致性]]（recompute/bypass、$r_{\text{corr}}$）、[[trainer与rollout引擎组合]]（双引擎是 TIM 根）、verl 框架
- **下游（应用）**: [[训练推理不一致TIM技术综述]]（VeXact 是诊断层代表）、TIS/RS 阈值校准
- **对比 / 易混**:
  - **VeXact（诊断）vs [[FP16-fix]]（生产）**：VeXact 追求 bit 级一致做标尺，吞吐不优先；FP16-fix 降 mismatch 量级保生产吞吐。前者诊断、后者生产。
  - **VeXact vs [[Bitwise-consistent-RL]]**：都追求 bit 一致，但 VeXact 是 verl 上的 rollout 诊断引擎，Bitwise-consistent 是 vLLM+TorchTitan 的生产训练链路。
  - **VeXact vs [[R3 rollout routing replay]]**：R3 治 MoE 路由 TIM；VeXact 测所有 TIM（含路由）的总量级。

## 7. 常见误区与易错点

> [!warning] 误区 1：把 VeXact 当生产 rollout 引擎
> 它吞吐低（为确定性牺牲），定位是**诊断标尺/校准工具**，非大规模生产 rollout。

> [!warning] 误区 2：$K_1/K_3$ 平就说明没 TIM
> 论文实测 recompute 下前 700 步 $K_1/K_3$ 平但 reward 已降——KL 估计器对早期 TIM 不灵敏。要直接看 reward 与 $r_{\text{corr}}$ 分布。

> [!warning] 误区 3：recompute 就等于 VeXact
> recompute 让 old/new 同出自训练引擎，消数值型 TIM，但若 rollout 仍用 vLLM 生成，rollout 期的路由/采样动作空间仍可能与训练不一致（MoE）。VeXact 连 rollout 都用训练 kernel，更彻底。

## 8. 延伸细节

### 8.1 局限
- 规模/覆盖有限，泛化待验证
- 算法层修正都是 post-hoc（只能丢弃已生成样本，不能修根因）
- 无 VeXact 时阈值标定靠经验
- 固定 tiling 换稳定性但损性能

### 8.2 内容来源
两类 TIM 来源、VeXact 设计、$r_{\text{corr}}$、$K_1/K_3$、崩溃点（Qwen3-30B-A3B REINFORCE/GRPO、dense 1.7B）、TIS+RS 联合均来自 arXiv:2605.14220 全文已联网核实。代码 verl-project/vexact。截至 2026-09。

---
相关: [[训练推理不一致TIM技术综述]] | [[训推不一致]] | [[rollout train reference logprob一致性]] | [[Bitwise-consistent-RL]] | [[FP16-fix]] | [[R3 rollout routing replay]] | [[TIS]] | [[trainer与rollout引擎组合]] | [[17-RL训推一体框架]]
