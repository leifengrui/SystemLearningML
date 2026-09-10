# TRM

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §83
> **别名**: Trust Region Masking / 序列级信任域 masking / 长程 trust region
> **难度**: 高（需懂 [[importance sampling与off-policy correction]]、[[PPO clipped objective]]、[[rollout train reference logprob一致性]]、trust region / Kakade-Langford 性能差恒等式）
> **来源**: Li et al., *Trust Region Masking for Long-Horizon LLM Reinforcement Learning*, arXiv:2512.23075 (2025-12, v5 2026-06). 作者: Yingru Li, Jiacai Liu, Jiawei Xu, Yuxuan Tong, Ziniu Li, Qian Liu, Baoxiang Wang.

## 1. 一句话定义

**TRM（Trust Region Masking）** 针对**长程 LLM RL** 任务指出：经典 $O(T^2)$ trust-region 界在长序列上**完全空洞（vacuous）**——$T=4096$ 时界 $\approx1677$（reward 在 $[0,1]$，毫无信息量）；TRM 同时维护 **KL 路线**（Pinsker，子线性 $\sqrt t$）与 **TV 路线**（耦合界，对集中偏差紧），按位置取更紧者，证明两者**严格互补、合一更优**，并据此对序列做 binary mask（超阈即丢梯度），稳定长程训练。

## 2. 为什么需要它（动机与背景）

### 2.1 长 horizon 让经典界失效

trust region 方法（TRPO/PPO）的理论根是 Kakade-Langford 性能差恒等式。朴素最坏界用 $\|g_t\|_\infty\le2\varepsilon$ + 无界耦合 $\|d_t^{\pi_\theta}-d_t^{\pi_{\text{roll}}}\|_{TV}\le(t-1)\varepsilon$：

$$|\text{Error}|\le4\varepsilon^2\sum_{t=1}^T(t-1)=2T(T-1)\varepsilon^2\le T(T-1)\delta$$

$T=4096,\ \delta=10^{-4}$ → 界 $\approx1677$。reward $\in[0,1]$，这个上界**无法证明优化在改进性能**——vacuous。二次 $O(T^2)$ 让微小逐 token 分歧复合到无用。

### 2.2 PPO token 级 clip 反而加剧失稳

论文实测（Qwen3-8B-Base, DAPO-MATH-17k）：token 级 PPO clip 让 PPL gap 增大、score 退化——clip 在长程上不是稳，是伤。

### 2.3 思路：用更紧的散度界 + 序列级 mask

不再用 ratio clip，而是直接估序列级 trust region（用 rollout 时存的 $\pi_{\text{roll}}$ logits 与训练时 $\pi_\theta$ logits），对超阈序列整条 mask（拒绝采样式，梯度按总 batch N 归一）。

## 3. 核心概念详解

### 3.1 KL 路线（Pinsker）

用 Pinsker 不等式把 KL 转 TV，得**子线性** $\sqrt t$：

- Pinsker-Marginal 界：$B_{PM}^{KL}=\frac43 T^{3/2}\delta$（小分歧区）
- 混合界（KL）：$B_{Mix}^{KL}=2T\sqrt{\delta\cdot D_{KL}^{\text{seq}}}$（$O(T)$）

**代价**：Pinsker 在分歧**集中在少数 token**（如 MoE 路由跳变：$\pi_{\text{roll}}(v)=0.9$ 但 $\pi_\theta(v)=0.001$）时**松**——它假设分歧均匀分布。

### 3.2 TV 路线（直接全 TV）

不做 KL→TV 转换，直接用 TV：

- 耦合界：$B_{Coup}=4\min(1,\varepsilon)\sum_{t=1}^T\min(1,(t-1)\varepsilon)$，$T\varepsilon>1$ 时 $O(T)$

**代价**：context-shift 缩放是线性（非子线性），分歧均匀时不如 KL 路线紧。

### 3.3 合一严格更优

KL 与 TV **互补**：
- 分歧**均匀分布**：Pinsker 紧（$\varepsilon=\sqrt{\delta/2}$），KL 路线子线性赢
- 分歧**集中**（MoE 路由跳变）：Pinsker 松（$\varepsilon\ll\sqrt{\delta/2}$），TV 路线的紧 advantage 因子赢

Adaptive 界按位置取更紧者：

$$B_{Adap^*}=4\sum_{t=1}^T\bar D_t\cdot\min\!\big(1,\ (T-t)\varepsilon,\ \sqrt{(T-t)\delta/2}\big)$$

统一界 $B^*=\min\{B_{PM}^{KL},B_{PM}^{TV},B_{Mix}^{KL},B_{Mix}^{TV},B_{Coup},B_{Adap^*}\}$。数值（$T=4096$）：
- 仅 KL 统一界 $\le8.2$
- KL+TV 统一界 $\le4.1$（**2× 改进**）

TV 界在 Pinsker 紧时退化为 KL 界，故同时维护两者**只会更好**。

### 3.4 TRM 的序列 mask

$$M(x,y)=\mathbf{1}\!\left[\max_t D_{KL}(c_t)\le\delta\right]$$

- $D_{KL}(c_t)$ 用整个 vocab 在位置 $t$ 的 KL（rollout logits 与训练 logits 都已存，**无额外推理开销**）
- 可选 $D_{TV}^{\text{tok}}$ 并行算
- 梯度按**总 batch N**（非接受数）归一 → 被拒序列零梯度（拒绝采样式）

## 4. 数学原理 / 公式

### 4.1 经典界 vacuous（前述）

$2T(T-1)\varepsilon^2\le T(T-1)\delta$；$T=4096,\delta=10^{-4}$ → $1677$（vacuous）。

### 4.2 KL vs TV 互补（前述）

$B^*=\min(\text{KL 路},\text{TV 路},\text{Adaptive})$；KL+TV $\le4.1$ vs 仅 KL $\le8.2$。

### 4.3 实验结果

- 模型：Qwen3-8B-Base，DAPO-MATH-17k，AIME25
- token 级 PPO clip **加剧**失稳（PPL gap 大、score 降）
- TRM-Max（$\delta=0.05$）与 TRM-Avg（$\delta=0.001$）各自稳定
- 单独松的 TRM-Max（$\delta=0.1$）或 TRM-Avg（$\delta=0.002$）**单独失败**，但**两者组合成功**——"max 抓异常点，avg 限累积漂移"

## 5. 代码示例

```python
import torch
def trm_mask(kl_per_pos, delta=0.05):
    """序列级 binary mask: max_t KL(c_t) <= delta 才保留.
    kl_per_pos: (T,) 每位置的全 vocab KL (rollout vs train logits)
    """
    return (kl_per_pos.max() <= delta)  # 标量, 整条序列 mask
# 生产: 梯度按总 batch N 归一 (非接受数), 被拒序列零梯度
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[importance sampling与off-policy correction]]（IS/信任域理论根）、[[PPO clipped objective]]（token 级 clip 在长程失效）、[[rollout train reference logprob一致性]]（rollout/train logits 存储）、trust region / Kakake-Langford
- **下游（应用）**: [[训练推理不一致TIM技术综述]]（版本/staleness 层 masking 代表）、长程 reasoning RL
- **对比 / 易混**:
  - **TRM（序列级 trust region mask）vs [[PPO clipped objective]]（token 级 ratio clip）**：TRM 用散度界对整序列判定，clip 用 ratio 对单 token 截断。论文实测 clip 在长程加剧失稳，TRM 稳。
  - **TRM vs [[IcePop]]**：IcePop mask 训推概率比 token（引擎层 TIM）；TRM mask 序列级 trust region（staleness/off-policy 层）。不同层。
  - **TRM vs [[DPPO]]**：DPPO 用散度替代 ratio 做 token 级 mask；TRM 用散度界做序列级 mask。粒度与界推导不同。

## 7. 常见误区与易错点

> [!warning] 误区 1：经典 $O(T^2)$ 界够用
> 长程（$T$=几千）下界远超 reward 范围，vacuous。不能用它证优化改进。

> [!warning] 误区 2：KL 与 TV 二选一
> 两者互补：均匀分歧 KL 紧，集中分歧 TV 紧。合一严格更优，TV 退化进 KL 所以不会变差。

> [!warning] 误区 3：max 或 avg 单独够
> 实测单独松的 max/avg 会失败，**组合**才稳——max 抓异常点（路由跳变），avg 限累积漂移。

## 8. 延伸细节

### 8.1 局限
- **长度偏差**：拒绝概率随 $T$ 增，惩罚长 reasoning 链（附录 F 的 LN-TRM/SER 部分缓解）
- **全局前提**：最紧界要求所有可达 context 的 $D_{KL}^{\text{tok,max}}\le\delta$，不可验证
- $L_{\text{masked}}\ne L$（前提不满足时被拒序列可能带不成比例 IS 质量）
- 被拒序列无梯度，高拒绝率降样本效率

### 8.2 内容来源
$O(T^2)$ vacuous 推导、$B_{PM}^{KL}/B_{Mix}^{KL}/B_{Coup}/B_{Adap^*}$、$T=4096$ 数值（8.2 vs 4.1）、序列 mask $M(x,y)$、Qwen3-8B-Base 实验均来自 arXiv:2512.23075 全文已联网核实。截至 2026-09。

---
相关: [[训练推理不一致TIM技术综述]] | [[importance sampling与off-policy correction]] | [[PPO clipped objective]] | [[rollout train reference logprob一致性]] | [[DPPO]] | [[IcePop]] | [[17-RL训推一体框架]]
