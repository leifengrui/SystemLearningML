# DPPO

> **所属章节**: [[GRPO体系]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §81
> **别名**: Divergence Proximal Policy Optimization / Binary-KL PPO / Binary-TV PPO / 散度替代 ratio clip
> **难度**: 高（需懂 [[GSPO]]、[[importance sampling与off-policy correction]]、[[PPO clipped objective]]、[[R3 rollout routing replay]]、trust region / 散度）
> **来源**: Qi et al. (Sea AI Lab / NUS), *Rethinking the Trust Region in LLM Reinforcement Learning*, arXiv:2602.04879 (2026-02). 代码: https://github.com/sail-sg/Stable-RL；verl 实现 https://verl.readthedocs.io/en/latest/algo/dppo.html

> [!warning] arXiv 号澄清
> 正确 ID 是 **2602.04879**（2026 年 2 月）。曾被误传为 2502.04879——后者实为无关统计论文《Statistical Collusion by Collectives on Learning Platforms》。本条按已核实 2602.04879。

## 1. 一句话定义

**DPPO（Divergence Proximal Policy Optimization）** 主张用**策略散度（TV/KL）直接估计**替代 PPO 的**概率 ratio 截断**作信任域——因 PPO 的 token 级 ratio 是真实散度的**单样本 MC 估计**，在长尾词表上结构性有偏（低概率 token 被过度惩罚、高概率 token 惩罚不足）；DPPO 用 **Binary-KL / Binary-TV / Top-K** 三种散度近似，给出自适应 ratio 界 $|r-1|\le\delta/\mu$（界与行为概率 $\mu$ 成反比，恰好抵消长尾偏差），在 MoE 上**无需 [[R3 rollout routing replay]] 也能稳定训练**。

## 2. 为什么需要它（动机与背景）

### 2.1 PPO ratio clip 的结构性偏差

PPO 的 $|r_t-1|\le\varepsilon$ 是对真实 TV 散度 $D_{TV}(\mu\|\pi)=\frac12\mathbb{E}_{y\sim\mu}[|r_t-1|]$ 的**单样本噪声估计**，不是真信任域。论文 Fig.2（Qwen3-30B-A3B）显示低概率 token 的 ratio 极波动而 TV 稳定。bound 论证：

- **低概率 token**：$\mu(a_{\text{low}})=10^{-4},\ \pi=10^{-2}\Rightarrow r_{\text{low}}=100$ → 被裁，**过度惩罚**，尽管只移了 0.0099 质量
- **高概率 token**：$\mu(a_{\text{high}})=0.99,\ \pi=0.80\Rightarrow r\approx0.808$ → 在裁剪区间内，**惩罚不足**，尽管移了 0.19 质量（更大的 TV 贡献）

→ 低概率 token 被过罚、高概率 token 欠约束。这是 ratio 代理的根本缺陷。

### 2.2 散度才是正确的信任域

直接估散度 $D(\mu\|\pi)$（TV 或 KL），不再用单 token ratio 噪声估计。

### 2.3 MoE 稳定不依赖 R3

论文 Fig.8：**DPPO 不加 R3** 持续**优于**加 R3 的基线；R3 与 DPPO 收益"largely orthogonal"，DPPO+R3 进一步增益。MoE-Base-no-R3 设定下 CISPO 崩、DPPO 稳。

## 3. 核心概念详解

### 3.1 Binary-KL（Bernoulli 折叠）

把 vocab 折成 Bernoulli（采样 token vs 其余）：

$$D_{KL}^{\text{Bin}}(t)=\mu(a_t|s_t)\log\frac{\mu(a_t|s_t)}{\pi(a_t|s_t)}+(1-\mu(a_t|s_t))\log\frac{1-\mu(a_t|s_t)}{1-\pi(a_t|s_t)}$$

证明为真 KL 的下界（附录 B）。**动机**：精确散度需两策略的完整 softmax 分布（显存重）；Bernoulli 折叠只需采样 token 的概率，开销可忽略。

### 3.2 Binary-TV

$$D_{TV}^{\text{Bin}}(t)=|\mu(a_t|s_t)-\pi(a_t|s_t)|=\mu(a_t|s_t)\,|r_t-1|$$

→ 自适应 ratio 信任域：

$$|r_t-1|\le\frac{\delta}{\mu(a_t|s_t)}$$

**允许的 ratio 偏离与行为概率 $\mu$ 成反比**——低概率 token 容许大 ratio 偏离（不被过罚），高概率 token 容许小偏离。恰好抵消 §2.1 的长尾偏差。

### 3.3 Top-K 散度近似

取 $\mathcal{A}'_t=\text{TopK}(\mu(\cdot|s_t),K)\cup\{a_t\}$，余者并入"other"桶，得扩展词表 $\mathcal{A}''_t$：

$$D_{TV}^{\text{TopK}}(t)=\frac12\sum_{a\in\mathcal{A}''_t}|p_t^\mu(a)-p_t^\pi(a)|,\quad D_{KL}^{\text{TopK}}(t)=\sum_{a\in\mathcal{A}''_t}p_t^\mu(a)\log\frac{p_t^\mu(a)}{p_t^\pi(a)}$$

实验 $K=20$。消融（Fig.11）：Binary 与 Top-K 相近，均显著超基线 → **Binary 足够**。

### 3.4 DPPO 目标与 mask

$$L_\mu^{DPPO}(\pi)=\mathbb{E}_{y\sim\mu}\!\left[\sum_{t=1}^{|y|}M_t^{DPPO}\cdot r_t\hat A_t\right]$$

$$M_t^{DPPO}=\begin{cases}0,&(\hat A_t>0\wedge r_t>1\wedge D>\delta)\ \text{或}\ (\hat A_t<0\wedge r_t<1\wedge D>\delta)\\1,&\text{otherwise}\end{cases}$$

mask 只阻**朝离开信任域方向**的更新，永不阻**朝 $r=1$ 修正**的更新。把 $D$ 换成 $|r_t-1|$ 即退化 vanilla PPO。

## 4. 数学原理 / 公式

### 4.1 TV 与 ratio 的关系

$$D_{TV}(\mu(\cdot|s_t)\|\pi(\cdot|s_t))=\frac12\mathbb{E}_{y\sim\mu}[|r_t-1|]$$

PPO 截的是样本 $|r_t-1|$，是上式期望的**噪声单样本估计**——不是真信任域。

### 4.2 Binary-TV 的自适应界（前述）

$|r_t-1|\le\delta/\mu$，界 $\propto1/\mu$。

### 4.3 实验结果

- 模型：Qwen3-30B-A3B-Base/-A3B、Qwen3-8B-Base、+LoRA、Gemma-2-9B-It、Qwen3-4B-Instruct-2507、Llama family
- 基准：AIME24/25 (Avg@32)、AlpacaEval 2.0、MATH
- DPPO 超 GRPO-ClipHigher 与 CISPO（5 个大配置）
- RLHF：AlpacaEval 2.0 **80.90** 长度控制 / **79.93** raw win rate，声称 SOTA
- 稳定性（§5）：信任域须锚定 **rollout 策略 $\mu_{\theta'}$**（非 recompute 的 $\pi_{\theta'}$，recompute 导致崩 Fig.4，浪费 ~25% 算力）；不稳源是 ≤0.5% 负样本更新把策略推离信任域；TIS **恶化**稳定性（低概率 token 方差最大，最常被截）

### 4.4 与 R3 的关系
DPPO 不加 R3 优于加 R3 基线；DPPO+R3 进一步增益（正交）。MoE-Base-no-R3 下 CISPO 崩、DPPO 稳。

## 5. 代码示例

```python
import torch
def dppo_mask(mu_a, pi_a, A, r, delta=0.15):
    """Binary-TV 散度 mask. mu_a/pi_a: (T,) 采样 token 的行为/目标概率.
    只阻朝离开 trust region 方向的更新."""
    D_bin_tv = (mu_a - pi_a).abs()        # = mu_a * |r-1|
    leave = ((A > 0) & (r > 1)) | ((A < 0) & (r < 1))
    mask = torch.where(leave & (D_bin_tv > delta), 0.0, 1.0)
    return mask
# verl: LOSS_MODE=dppo_kl (Binary-KL) / dppo_tv (Binary-TV); Top-K 在 Stable-RL repo
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[GSPO]]（序列级 ratio，DPPO 是其后继）、[[importance sampling与off-policy correction]]（ratio vs 散度）、[[PPO clipped objective]]（ratio clip 的偏差是 DPPO 的动机）、[[R3 rollout routing replay]]（DPPO 与 R3 正交）
- **下游（应用）**: [[训练推理不一致TIM技术综述]]（优化层散度路线代表）、verl `dppo_kl`/`dppo_tv`、Stable-RL
- **对比 / 易混**:
  - **DPPO（散度 mask）vs [[GSPO]]（序列 ratio clip）**：DPPO 改代理（ratio→散度）；GSPO 改粒度（token→序列）。DPPO 解决 GSPO 遗留的 ratio-as-proxy。
  - **DPPO（硬 mask）vs [[DRPO]]（平滑正则）**：DPPO 边界处梯度直接置零（丢弃）；DRPO 平滑阻尼（修正）。DRPO 暴露 DPPO 硬 mask 的局限。
  - **DPPO vs [[TRM]]**：DPPO token 级散度 mask；TRM 序列级散度界 mask。粒度与界推导不同。

## 7. 常见误区与易错点

> [!warning] 误区 1：DPPO 的 Binary 近似是为了精度
> 不是。Binary 是为**省显存**——精确散度需两策略完整 softmax，Bernoulli 折叠只需采样 token 概率。消融显示 Binary 与 Top-K 相近，足够。

> [!warning] 误区 2：DPPO 必须配 R3 才能稳 MoE
> 不必。论文 Fig.8 明示 DPPO 不加 R3 优于加 R3 基线。R3 是正交增益，非必需。

> [!warning] 误区 3：信任域锚定 recompute 策略
> 错。论文 §5：须锚定 **rollout 策略 $\mu_{\theta'}$**，recompute 导致崩（Fig.4）、浪费 25% 算力。

> [!warning] 误区 4：TIS 与 DPPO 可叠加增效
> 论文实测 TIS **恶化** DPPO 稳定性——低概率 token 方差最大，TIS 截上限最常截这些，反而放大偏差。

## 8. 延伸细节

### 8.1 verl / Stable-RL 实现
verl 仅实现 Binary-KL（`LOSS_MODE=dppo_kl`）与 Binary-TV（`LOSS_MODE=dppo_tv`）；Top-K 在原 Stable-RL repo。env var `LOSS_MODE` 控制，例 `examples/dppo_trainer/run_qwen3_30b_a3b_megatron.sh`。$\varepsilon=0.15$（BF16）/ $0.6$（FP8-E2E）。

### 8.2 内容来源
Binary-KL/TV/Top-K 公式、自适应界 $\propto1/\mu$、长尾偏差 bound、mask 定义、Fig.8 DPPO>R3、Fig.4 recompute 崩、TIS 恶化、AlpacaEval 80.90 均来自 arXiv:2602.04879 全文已联网核实。verl/Stable-RL 实现已核实。截至 2026-09。

---
相关: [[训练推理不一致TIM技术综述]] | [[GSPO]] | [[DRPO]] | [[importance sampling与off-policy correction]] | [[PPO clipped objective]] | [[R3 rollout routing replay]] | [[TRM]] | [[17-RL训推一体框架]]
