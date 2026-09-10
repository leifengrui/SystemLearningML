# TIS

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §83
> **别名**: Truncated Importance Sampling / token 级 IS 截断 / verl rollout IS 修正
> **难度**: 中（需懂 [[importance sampling与off-policy correction]]、[[PPO clipped objective]]、[[rollout train reference logprob一致性]]、verl）
> **来源**: Yao Feng (姚峰), *Your Efficient RL Framework Secretly Brings You Off-Policy RL Training* (Notion blog, 2025；**非 arXiv 论文**)。工程实现: verl PR #2953（及 #3694/#3915/#3955/#3984）、FlashRL https://github.com/yaof20/Flash-RL。

> [!warning] 来源性质
> TIS **没有 arXiv 论文**，原始出处是作者 Notion 博客（WebFetch 未渲染，内容据二级描述）。但其工程实现在 verl 主线 PR 里，公式可从代码核实。本条公式与配置以 verl 代码为准标注。

## 1. 一句话定义

**TIS（Truncated Importance Sampling，截断重要性采样）** 是 verl 等框架默认的 token 级 IS 修正：把重要性权重 $\rho_t=\pi_\theta/\pi_{\text{sampler}}$ **只截上限** $\tilde w_t=\min(\rho_t,C)$（不双裁），所有 token 仍贡献梯度但权重有界；用于缓解 [[训推不一致]]（TIM）与 8-bit 量化 rollout 引入的 off-policy 偏移，是 PPO ratio clip 之外的轻量 IS 兜底。

## 2. 为什么需要它（动机与背景）

### 2.1 双引擎 + 量化 rollout 的 IS 污染

rollout 用 vLLM/SGLang（可能 FP8/INT8 量化 KV cache 或权重），trainer 用 FSDP/Megatron BF16/FP32。同 $\theta_{old}$ 下两套 logp 有差（TIM，见 [[rollout train reference logprob一致性]]），IS 权重 $\rho=\exp(\log\pi_\theta-\log\pi_{old}^{\text{rollout}})$ 在 $\theta$ 未动时偏离 1。极端 token $\rho$ 很大 → IS 高方差。

### 2.2 PPO clip vs TIS：两套不同的"截断"

- **PPO ratio clip**（[[PPO clipped objective]]）：截的是**策略更新 ratio** $r=\pi_\theta/\pi_{\theta_{old}}$，双裁 $[1-\varepsilon,1+\varepsilon]$，超阈时**梯度被截断**（min 取 clip 项）
- **TIS**：截的是**训推引擎 IS 权重** $\rho=\pi_\theta/\pi_{\text{sampler}}$，**只截上限** $C$，超阈 token **仍贡献梯度**（只是权重压到 $C$）

TIS 的对象是"引擎数值差引入的 IS 权重"，与 PPO 的"权重更新 ratio"正交，可叠加。

### 2.3 为什么只截上限

TIM 让 $\rho$ 的分布右尾长（少数训推分歧大的 token $\rho$ 极大）。只截上限压住右尾方差，保留所有 token 的梯度信号（比 MIS 双边丢弃更温和）。Yao 的博客核心论点：很多"高效 RL 框架"因 bypass/reuse 默认就是 off-policy 的，TIS 是显式承认并修正。

## 3. 核心概念详解

### 3.1 数学定义

$$\tilde w_t(\theta)=\min\!\left(\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\text{sampler}}(y_t|x,y_{<t})},\ C\right)$$

- $\pi_{\text{sampler}}$：rollout 引擎（vLLM/SGLang）算的采样分布概率
- $C$：截断上界（verl 默认 $C=2.0$，部分场景提到 $C=5$）
- 超过 $C$ 的 token 权重压到 $C$，但仍进梯度（非零）

### 3.2 与 MIS（双边 masking）对比

verl 支持两种 IS 控制策略：
- **TIS**：$\tilde w=\min(\rho,C)$，只截上限，token 不丢
- **MIS（Masked IS）**：双边 $\rho\in[1/C,C]$ 内保留，超出**整 token 权重置零**（丢弃），"本质是一种拒绝采样"

verl 三级 IS 粒度 × 两种控制策略：
| 粒度 | 公式 |
|---|---|
| Token-level | $\rho_t=\exp(\text{old\_log\_prob}_t-\text{rollout\_log\_prob}_t)$ |
| Sequence-level | $\prod_t\rho_t=\exp(\sum_t\log\rho_t)$ |
| Geometric-level | $(\prod_t\rho_t)^{1/T}$（长归一） |

### 3.3 TIS 在 verl 的配置

```yaml
actor_rollout_ref:
  rollout:
    calculate_log_probs: true   # recompute: trainer 用 θ_old 重算 old_log_prob
    rollout_is_threshold: 2.0    # TIS 上限 C
    rollout_rs_threshold: null   # MIS 上限 (双边)
    rollout_rs_threshold_lower: null  # MIS 下限 (auto=1/upper)
```

## 4. 数学原理 / 公式

### 4.1 TIS 估计量

$$\hat\mu_{\text{TIS}}=\frac1N\sum_i\min(\rho_i,C)\,f_i,\qquad \rho_i=\frac{\pi_\theta}{\pi_{\text{sampler}}}$$

- **有偏**（截断后期望 $\ne$ 真 IS 期望）
- **方差有界**：$\text{Var}\le C^2\mathbb{E}[f^2]$，上限被 $C$ 锁死
- bias-variance trade：$C$ 小→偏大但方差小；$C$ 大→偏小但方差大

### 4.2 与 PPO clip 的叠加

PPO 目标里 TIS 权重乘在 ratio 之外（概念）：

$$L\approx\mathbb{E}\big[\min(\rho_t,C)\cdot\min(r_t\hat A_t,\text{clip}(r_t,1\pm\varepsilon)\hat A_t)\big]$$

$\rho_t$（训推 IS）管引擎差，$r_t$（PPO ratio）管权重更新，两者独立。

### 4.3 实测定位（来自 R3 论文对照）
- TIS 能**延缓但治不好** MoE RL 崩溃（R3 论文：SFT+GRPO+TIS 在 step 90/170 崩，比无 TIS 的 60/100 晚，但仍崩）
- IcePop 在 AIME-2025 上比 TIS 高 ~6%
- R3 论文里 TIS+R3 无明显增益甚至略降（R3 已把分歧压到接近 dense，TIS 无对象）

## 5. 代码示例

```python
import torch
def tis_weight(logp_train, logp_sampler, C=2.0):
    """TIS: 只截上限的 token 级 IS 权重.
    logp_train : (T,) trainer 算的 logp
    logp_sampler: (T,) rollout 引擎算的 logp
    """
    rho = torch.exp(logp_train - logp_sampler)   # 训推 IS 权重
    return torch.clamp(rho, max=C)               # 只截上限, 不丢 token

# 对比 MIS (双边 masking)
def mis_weight(logp_train, logp_sampler, C=2.0):
    rho = torch.exp(logp_train - logp_sampler)
    mask = ((rho >= 1/C) & (rho <= C)).float()   # 超出 [1/C, C] 整 token 置零
    return rho * mask

# 模拟: TIM 让少数 token ρ 极大
logp_old = torch.randn(1000)*2 - 5
tim = torch.randn(1000)*0.01
logp_roll = logp_old - tim
w_tis = tis_weight(logp_old, logp_roll, C=2.0)
w_mis = mis_weight(logp_old, logp_roll, C=2.0)
print(f"TIS: max={w_tis.max():.2f} 非零=100%  (压上限不丢)")
print(f"MIS: max={w_mis.max():.2f} 非零={w_mis.gt(0).float().mean():.2%} (双边丢)")
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[importance sampling与off-policy correction]]（IS 理论、方差爆炸）、[[PPO clipped objective]]（ratio clip 是另一套截断）、[[rollout train reference logprob一致性]]（bypass/recompute、TIM 污染 IS）、[[训推不一致]]（TIM 是 TIS 修正对象）
- **下游（应用）**: [[训练推理不一致TIM技术综述]]（版本/staleness 层 baseline）、verl/OpenRLHF 默认 IS 修正、FlashRL 8-bit 量化 rollout
- **对比 / 易混**:
  - **TIS（截 IS 上限）vs PPO clip（截 ratio 双裁）**：对象不同（训推 IS vs 策略 ratio）、裁法不同（单上限 vs 双裁）、梯度处理不同（压权重 vs 截梯度）。可叠加。
  - **TIS（单上限）vs MIS（双边丢弃）**：TIS 温和保留所有梯度；MIS 激进丢弃超阈 token。
  - **TIS vs [[IcePop]]**：IcePop 双边 mask 训推概率比（基于分歧）；TIS 单边截 IS 权重上限（基于 off-policy 程度）。IcePop 实测优于 TIS。
  - **TIS vs [[R3 rollout routing replay]]**：TIS 算法层延缓崩溃；R3 工程层治根因。R3 后 TIS 无对象。

## 7. 常见误区与易错点

> [!warning] 误区 1：把 TIS 当 PPO clip
> 对象不同。TIS 截训推 IS 权重 $\rho=\pi_\theta/\pi_{\text{sampler}}$（引擎差），PPO clip 截策略 ratio $r=\pi_\theta/\pi_{\theta_{old}}$（权重更新）。两者正交可叠加。

> [!warning] 误区 2：TIS 能治好 MoE 崩溃
> R3 论文实测 TIS 只是**延缓**（崩得晚），治不好结构性 MoE 路由 TIM。MoE 需 R3 治根因。

> [!warning] 误区 3：以为 TIS 有正式论文
> 没有。原始出处是 Notion 博客（非 arXiv）。公式以 verl 代码实现为准。

## 8. 延伸细节

### 8.1 FlashRL 的 TIS 用途
FlashRL 用 TIS 桥接 INT8/FP8 量化 rollout 与 BF16 训练的 gap：Qwen2.5-32B 在 AIME2024 用 INT8 rollout 达 50，匹配 BF16 rollout 性能——靠 TIS 修正量化引入的 off-policy。

### 8.2 bypass_mode 的关系
verl `bypass_mode=true` 跳过 trainer 用 $\theta_{old}$ 重算 `old_log_prob`（直接用 rollout 算的），省一次 forward 但 IS 权重分母带 TIM。`bypass_mode=false` + `calculate_log_probs=true` 时 trainer 重算并可挂 TIS/MIS 修正残余。详见 [[rollout train reference logprob一致性]] §8.1。

### 8.3 内容来源
公式 $\min(\rho,C)$、verl 默认 $C=2.0$、三级 IS × 两种策略、TIS/MIS 对照、FlashRL INT8 实测来自 verl PR #2953/#3694 与 FlashRL repo 已联网核实。TIS"延缓不治"MoE 崩溃来自 R3 论文 arXiv:2510.11370 对照实验。原始 Notion 博客未渲染（WebFetch 失败），故标注为部分来源。截至 2026-09。

---
相关: [[训练推理不一致TIM技术综述]] | [[importance sampling与off-policy correction]] | [[PPO clipped objective]] | [[rollout train reference logprob一致性]] | [[训推不一致]] | [[IcePop]] | [[R3 rollout routing replay]] | [[17-RL训推一体框架]]
