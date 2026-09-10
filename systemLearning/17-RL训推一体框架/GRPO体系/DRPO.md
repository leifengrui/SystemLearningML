# DRPO

> **所属章节**: [[GRPO体系]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §81
> **别名**: Divergence Regularized Policy Optimization / 平滑散度正则 PPO
> **难度**: 高（需懂 [[DPPO]]、[[GSPO]]、[[importance sampling与off-policy correction]]、$\ell_2^2$ vs $\chi^2$ 正则）
> **来源**: Yao et al. (Tencent Hunyuan / NUS), *Rethinking the Divergence Regularization in LLM RL*, arXiv:2606.09821 (2026-06). 作者: Jiarui Yao, Xiangxin Zhou, Penghui Qi, Wee Sun Lee, Liefeng Bo, Tianyu Pang.

> [!warning] 两个澄清
> 1. arXiv 正确 ID 是 **2606.09821**（2026-06）。曾被误传为 2506.09821——按已核实 2606.09821。
> 2. **DRPO ≠ UniRL**。DRPO 是独立方法论文；UniRL 是同团队（Tencent Hunyuan）的另一个统一多模态 RL 框架，二者不要混。

## 1. 一句话定义

**DRPO（Divergence Regularized Policy Optimization）** 把 [[DPPO]] 的**硬 binary mask**（边界处梯度直接置零=丢弃）换成**平滑、advantage 加权的二次正则**，给出**有界连续**的逐 token 梯度权重 $w_t\in[1-1/\delta,1+1/\delta]$——既保持 DPPO 的散度信任域边界，又给边界处的"有害偏离"以**修正性**（而非归零）梯度；理论核心是 advantage 加权 **$\ell_2^2$ 正则**替代 SPO/PPO 的 **$\chi^2$ 正则**（后者对低概率 token 过敏，7.8% 采样 token $\mu\le0.01$），用 universal $\delta=12.5$ 跨设置稳定，直接暴露 DPPO 硬 mask 的局限。

## 2. 为什么需要它（动机与背景）

### 2.1 DPPO 硬 mask 的局限

[[DPPO]] 的 $M_t\in\{0,1\}$：token 朝离开信任域方向越界时梯度**直接丢弃**，不是**修正**——不连续、全或无。边界处的有害更新被扔掉而非拉回，收敛慢、上限低。论文实测：硬 mask 方法（GRPO/DPPO）"often underperform their counterparts with smooth regularization"，DPPO 收敛比 DRPO 慢且低。

### 2.2 $\ell_2^2$ vs $\chi^2$：正则的敏感度

- SPO/PPO 的 ratio 二次正则对应 **$\chi^2$ 惩罚**，分母含 $1/\mu(a|s_t)$ → **对低概率 token 偏离极度敏感**
- DRPO 的正则对应 advantage 加权 **$\ell_2^2$ 惩罚** $\sum_a|\hat A_t(a)|(\pi(a|s_t)-\mu(a|s_t))^2$ → **对低概率 token 不过敏**

实测 **7.8% 采样 token $\mu(y_t|s_t)\le0.01$**，$\chi^2$ 的过敏是实打实的痛点。

### 2.3 好正则的三标准

论文给：(a) 边界与分布偏移对齐；(b) 长尾词表逐 token 梯度权重有界；(c) 平滑修正信号。DPPO 满足 (a) 不满足 (c)；DRPO 三者皆足。

## 3. 核心概念详解

### 3.1 DRPO 目标

$$\mathcal{L}_{\text{DRPO}}(x,\pi)=\mathbb{E}_{y\sim\mu(\cdot|x)}\!\left[\sum_{t=1}^{|y|}r_t\hat A_t-\frac{|\hat A_t|}{2\delta}\,\mu(y_t|s_t)\,(r_t-1)^2\right]$$

- 第一项 $r_t\hat A_t$：标准 PPO 策略梯度
- 第二项 $\frac{|\hat A_t|}{2\delta}\mu(y_t|s_t)(r_t-1)^2$：平滑二次正则，advantage 加权（$|\hat A_t|$ 大的 token 强约束），行为概率加权（$\mu$ 大的 token 强约束）
- $r_t=\pi(y_t|s_t)/\mu(y_t|s_t)$

### 3.2 诱导的梯度权重

$$w_t=1-\operatorname{sign}\!\big(\hat A_t(r_t-1)\big)\,\frac{D_t^{\text{Bin-TV}}}{\delta},\qquad D_t^{\text{Bin-TV}}=|\pi(y_t|s_t)-\mu(y_t|s_t)|$$

- $w_t\in[1-1/\delta,\ 1+1/\delta]$：**连续、永不归零**
- 朝离开信任域方向：$w_t$ 降低（阻尼），但不为 0
- 朝 $r=1$ 修正方向：$w_t$ 增大（鼓励）

### 3.3 与 DPPO 的 mask-DRPO 对照
论文做 "Mask-DRPO" 消融：只在 DPPO 信任域**外**施加正则，其余不变 → 与完整 DRPO 等价。证明增益**来自边界外的修正性正则**，而非内部。

### 3.4 稳定点
不动点 $\pi^\star=\mu+\operatorname{sign}(\hat A)\delta$——与 DPPO 保持**同一信任域边界** $|\pi-\mu|\le\delta$，但路径平滑。

## 4. 数学原理 / 公式

### 4.1 $\ell_2^2$ vs $\chi^2$

DRPO 正则 $\Leftrightarrow \sum_a|\hat A_t(a)|(\pi(a|s_t)-\mu(a|s_t))^2$（$\ell_2^2$）。
SPO/PPO ratio 二次 $\Leftrightarrow \chi^2$（分母 $1/\mu(a|s_t)$）。

### 4.2 universal δ

- **$\delta=12.5$ 跨所有设置稳定**（Qwen3-4B-Base、Qwen3-30B-A3B-Base BF16/FP8 rollout/FP8-E2E、Qwen3.5-35B-A3B-Base、DeepSeek-R1-Distill-Qwen-1.5B）
- DPPO 需 per-setting 调 $\varepsilon$（0.15 BF16 / 0.6 FP8-E2E）
- $\delta$ 从 12.5 降到 2.5 仅小幅下降 → 不敏感

### 4.3 实验结果
- DRPO 在 6 个设置上匹配/超越最佳基线（Fig.3）
- ratio 方法（GRPO、SPO）崩溃，尤其低精度
- 去掉 $|\hat A_t|$ 加权 → 失稳
- 替代散度（KL、TV、K3）均逊于 DRPO 公式

## 5. 代码示例

```python
import torch
def drpo_weight(A, r, mu_a, pi_a, delta=12.5):
    """DRPO 平滑连续梯度权重 (永不归零).
    A: (T,) advantage; r: (T,) ratio; mu_a/pi_a: (T,) 行为/目标概率.
    """
    D_bin_tv = (pi_a - mu_a).abs()
    w = 1 - torch.sign(A * (r - 1)) * D_bin_tv / delta
    return w.clamp(1 - 1/delta, 1 + 1/delta)  # 连续有界, 不归零
# 对比 DPPO 硬 mask: 越界 token w=0 (丢弃); DRPO 越界 token w 降低但>0 (修正)
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[DPPO]]（DRPO 是其平滑化后继，暴露 DPPO 硬 mask 局限）、[[GSPO]]（clip 族演化链）、[[importance sampling与off-policy correction]]（$\ell_2^2$ vs $\chi^2$）
- **下游（应用）**: [[训练推理不一致TIM技术综述]]（优化层散度正则路线终点）、Tencent Hunyuan 后训练
- **对比 / 易混**:
  - **DRPO（平滑正则）vs [[DPPO]]（硬 mask）**：边界处修正 vs 归零。DRPO 收敛快、上限高。
  - **DRPO vs [[TRM]]**：DRPO token 级连续权重；TRM 序列级 binary mask。粒度与连续性不同。
  - **DRPO $\ell_2^2$ vs PPO/SPO $\chi^2$**：低概率 token 敏感度不同，$\ell_2^2$ 不过敏。
  - **DRPO vs UniRL**：DRPO 是方法；UniRL 是同团队另一框架。勿混。

## 7. 常见误区与易错点

> [!warning] 误区 1：DRPO = UniRL
> 不是。DRPO 是独立方法论文《Rethinking the Divergence Regularization in LLM RL》；UniRL 是同团队 Tencent Hunyuan 的统一多模态 RL 框架，另一篇。

> [!warning] 误区 2：DRPO 抛弃了 DPPO 的信任域
> 没有。DRPO 与 DPPO 共用**同一边界** $|\pi-\mu|\le\delta$，不动点 $\pi^\star=\mu+\text{sign}(\hat A)\delta$ 一致。只把边界处从硬丢改平滑阻尼。

> [!warning] 误区 3：DRPO 的 $\delta$ 像 DPPO $\varepsilon$ 要精调
> 不用。universal $\delta=12.5$ 跨设置稳定，且 12.5→2.5 只小幅下降。这是 DRPO 相对 DPPO 的工程优势之一。

## 8. 延伸细节

### 8.1 Mask-DRPO 消融
只在信任域外施加正则、内部不变 → 与完整 DRPO 等价。证明增益来自边界外的修正性正则，是 DPPO 硬 mask 的直接改进证据。

### 8.2 内容来源
DRPO 目标公式、$w_t\in[1-1/\delta,1+1/\delta]$、$\ell_2^2$ vs $\chi^2$、universal $\delta=12.5$、7.8% 低概率 token、Fig.3 六设置、Mask-DRPO 消融均来自 arXiv:2606.09821 全文已联网核实。截至 2026-09。

---
相关: [[训练推理不一致TIM技术综述]] | [[DPPO]] | [[GSPO]] | [[importance sampling与off-policy correction]] | [[TRM]] | [[PPO clipped objective]] | [[17-RL训推一体框架]]
