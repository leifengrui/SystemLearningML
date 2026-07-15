# DrGRPO

> **所属章节**: [[GRPO体系]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: Dr. GRPO / GRPO Done Right / 去 std 与长度偏差的 GRPO
> **难度**: 高（需懂 [[GRPO]]、[[token-level与sequence-level objective]]、[[advantage function]]、[[baseline]]、[[policy gradient]]）


## 1. 一句话定义

**Dr.GRPO（Dr. GRPO，"Group Relative Policy Optimization Done Right"）** 是 Sea AI Lab 在 **arXiv:2503.20783**《Understanding R1-Zero-Like Training: A Critical Perspective》（Liu et al. 2025）提出的 GRPO 修正版——它指出标准 [[GRPO]] 目标里**两个归一化项**会引入**优化偏差**：① advantage 除以组内 reward 标准差 $\sigma_r$（"question-level difficulty bias"，让简单/难问题权重虚高）② loss 除以响应长度 $1/|o_i|$（"response-level length bias"，对错样本非对称地偏好短/长）；Dr.GRPO **同时去掉这两项**（advantage 只减均值不除 std、loss 用常数 `MAX_TOKENS` 归一化而非 `mask.sum`），恢复成"MC return + 无偏 baseline"的**严格无偏策略梯度**（等价于 Eq.2 的 PPO 目标但 advantage 换成 group-relative），实测能**消除 GRPO 训练后期"错误回答越写越长"的 length-inflation 病**，提升 token 效率而不损推理性能。作者用 Dr.GRPO 在 Qwen2.5-Math-7B 上跑极简 recipe 拿到 AIME 2024 43.3%（7B SOTA）。

> [!note] 三句话定位
> - **是什么**：GRPO 目标里删掉 $\frac1{|o_i|}$ 和 $\frac1{\sigma_r}$ 两项，advantage $=R_i-\bar r$（只减均值），loss 用常数 `MAX_TOKENS` 归一化。
> - **为什么**：这两项让 GRPO 偏离无偏策略梯度——std 归一化按"组内离散度"重加权不同 prompt（难度偏差），长度归一化对正确/错误样本非对称（正确偏短、错误偏长，净效果错误回答越长）。删掉即恢复无偏。
> - **与 [[GRPO]] 关系**：GRPO 的"Done Right"版。advantage 估计同源（group-relative），只改归一化。是 [[GRPO体系]] 里"去偏差"那条线，与 [[DAPO]] 的"加 trick"路线互补。


## 2. 为什么需要它（动机与背景）

### 2.1 R1-Zero 训练后期"回答越来越长"是 bug 还是 feature？

DeepSeek-R1-Zero 训练时 response 长度持续增长，常被解读为"长思维链涌现 / self-reflection 发展"的标志。社区照搬 GRPO 复现 R1 时也观察到这个现象。但 Sea AI Lab 质疑：**这个"变长"可能不是推理能力涌现，而是 GRPO 目标函数本身引入的优化偏差**——尤其错误回答越写越长（overthinking）。若如此，"长 CoT 涌现"就是个被算法偏差污染的假象，必须修正才能看清真实信号。

### 2.2 GRPO 目标里两个"看似无害"的归一化项

回看 [[GRPO]] §4.1 目标：

$$
\mathcal{J}_{GRPO}=\mathbb{E}\Big[\frac1G\sum_i\underbrace{\frac1{|o_i|}}_{\text{① 长度归一}}\sum_t\min\big(\rho_{i,t}\,\underbrace{\hat A_i}_{\text{②}\,=\,(r_i-\bar r)/\sigma_r},\,\text{clip}\,\hat A_i\big)\Big]
$$

- 项① $\frac1{|o_i|}$：每条 response 内 token 平均（token-level mean）。看似只是把 loss 归一到可比尺度。
- 项② $\hat A_i=(r_i-\bar r)/\sigma_r$：除组内 std（z-score）。看似只是把 advantage 归一到 $[-1,1]$。

两项都被当作"无害的尺度调整"。但 Dr.GRPO 指出它们**不是全局标量**（不均匀缩放整个 batch），而是**逐 response / 逐 group 变化的重加权**，从而扭曲梯度方向 → 引入偏差。

### 2.3 修法：恢复"无偏策略梯度"

策略梯度的无偏形式（[[REINFORCE]] + 不依赖 action 的 baseline）应是：

$$
\hat g_{unbiased}=\frac1G\sum_i \nabla\log\pi(o_i|q)\,(R_i-\bar R)
$$

其中 advantage $=R_i-\bar R$（**只减组均值，不除 std**），且 token 求和**不除 $|o_i|$**（或除常数）。Dr.GRPO 就是把这个"最朴素的无偏形式"忠实实现出来——所以名字叫"Done Right"。


## 3. 核心概念详解

### 3.1 偏差一：Response-level length bias（$1/|o_i|$ 项）

GRPO 对每条 response 内 token 取平均 $\frac1{|o_i|}\sum_t$，于是 response $i$ 的每个 token 权重 $\propto \hat A_i/|o_i|$。关键在于 advantage 有正负（对=正、错=负），$1/|o_i|$ 对正负 advantage 作用方向相反：

| advantage 符号 | 含义 | $1/|o_i|$ 的效果 | 偏好 |
|---|---|---|---|
| $\hat A_i>0$（答对） | 要被强化 | 越短 $\Rightarrow$ 每 token 梯度越大 $\Rightarrow$ 短回答被强化得更猛 | **正确回答偏短** |
| $\hat A_i<0$（答错） | 要被抑制 | 越长 $\Rightarrow$ 每 token 惩罚越小（$|\hat A_i|/|o_i|$ 小） $\Rightarrow$ 长回答被惩罚得轻 | **错误回答偏长** |

净效果：**错误回答越来越长**（这正是 arXiv:2503.20783 摘要说的"artificially increases response length, especially for incorrect outputs"）。这跟"涌现长 CoT"无关，是优化偏差把错误样本往长了推。Dr.GRPO 删掉 $1/|o_i|$ → 每 token 权重 $\propto\hat A_i$（与长度无关）→ 无长度偏差。

> [!warning] 误区："GRPO 偏好短回答"
> 不对称。正确回答偏好短、错误回答偏好长。净的可见效果是**错误回答长度膨胀**（错误样本数量随训练变多/少不同，但单条错误回答越长）。不能简单说成"偏好短"。

### 3.2 偏差二：Question-level difficulty bias（$1/\sigma_r$ 项）

GRPO advantage $\hat A_i=(r_i-\bar r)/\sigma_r$。整个 group 被同一个 $1/\sigma_r$ 缩放。于是：

| group 的 $\sigma_r$ | 含义 | $1/\sigma_r$ | 在目标里权重 |
|---|---|---|---|
| 小（$\to0$） | 全对 / 全错（问题太易/太难） | $\to\infty$ | **虚高** |
| 大 | 一半对一半错（问题难度适中、信号最强） | 小 | 被压低 |

**这颠倒了该有的权重**：信息量最小的组（全对/全错，组内无相对信号，advantage 本该 $\approx0$）反而因为 $1/\sigma_r$ 被放大占主导；信息量最大的组（混合组）被压低。虽然全对/全错时 advantage 本身趋 0（[[GRPO]] §5.1），但**接近**全对/全错时 advantage 小、$1/\sigma_r$ 大，两者相乘的尺度不稳，且不同 prompt 的相对权重被 $\sigma_r$ 扭曲。

对比标准 RL 的"advantage normalization"技巧：通常是对**整个 batch** 归一化（$\hat A\leftarrow(\hat A-\bar A_{batch})/\sigma_{batch}$），这只是全局标量缩放、不改变方向，无害。GRPO 的 std 归一化是**逐 group（逐 prompt）**的——不同 prompt 缩放不同 → 跨 prompt 重加权 → 偏差。

### 3.3 Dr.GRPO 的修正

```
GRPO:    \hat A_i = (r_i - \bar r) / \sigma_r       loss = (1/G) Σ_i (1/|o_i|) Σ_t ...
Dr.GRPO: \hat A_i = (r_i - \bar r)                  loss = (1/(G·MAX_TOKENS)) Σ_i Σ_t ...
         (去 std)                                     (去 1/|o_i|, 改用常数 MAX_TOKENS)
```

- **advantage**：只减组均值 $\bar r$（baseline），不除 std。即 $\hat A_i=R_i-\bar R$。这正是 §2.3 的无偏形式。
- **loss 归一化**：不除每条的 $|o_i|$，改用常数 `MAX_TOKENS`（生成预算，训练全程固定）作分母（`masked_mean = (tensor*mask).sum(-1) / MAX_TOKENS`）。等价于"token-level sum 除以常数"——每 token 权重 $\propto\hat A_i/\text{MAX\_TOKENS}$ 与长度无关。

> [!note] 补充：常数分母不影响梯度方向
> `MAX_TOKENS` 是全局常数，对所有 step / 所有 batch 一样，只是把整个 loss 缩放一个固定标量——不改变梯度方向，只影响 learning rate 的等价尺度。故可严格还原无偏策略梯度。

### 3.4 Dr.GRPO = 无偏 PPO（Eq.2）+ group-relative MC baseline

作者证明：上述修正后，GRPO 目标**严格回到 PPO 原始目标（arXiv:2503.20783 Eq.2）**——即标准 PPO 的 surrogate objective，只是 advantage 不用 GAE+critic，而用"MC return + 组均值 baseline"（$\hat A_{i,t}=R_i-\bar R$，sequence-level 广播）。故名"Done Right"：GRPO 本想做的事（去 critic 的 PPO），其原实现被两个归一化项带偏了，Dr.GRPO 把它做对。

### 3.5 开源 PPO 实现也普遍带 length bias

Dr.GRPO 论文还指出一个"反直觉"发现：**主流开源 PPO for LLM 实现（trl / OpenRLHF / verl / SimpleRL-Zero / Open-Reasoner-Zero）几乎都按 response 长度归一化 loss**（`masked_mean = sum / mask.sum(dim)`），即都中了偏差一。作者推测这源自预训练阶段"pack 进定长 context、`loss.mean(-1)` 提数值稳定"的习惯被无脑搬到 RL 阶段——但预训练里 context 长度是常数，RL 里 response 长度是变量，搬过来就引入了长度偏差。所以 length bias 不是 GRPO 独有，是整个 LLM-RL 社区的实现惯例病。

```python
# arXiv:2503.20783 Listing 1 简化
def masked_mean(tensor, mask, dim):
    # 偏差版 (红): 按 response 长度归一 -> 长度偏差
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)
    # 无偏版 (绿): 除常数 MAX_TOKENS
    # return (tensor * mask).sum(axis=-1) / MAX_TOKENS
```


## 4. 数学原理 / 公式

### 4.1 GRPO vs Dr.GRPO 目标对比

设 $R_i=R(q,o_i)$，$\bar R=\frac1G\sum_j R_j$，$\sigma_R=\sqrt{\frac1G\sum_j(R_j-\bar R)^2}$，importance ratio $\rho_{i,t}=\pi_\theta(o_{i,t})/\pi_{\theta_{old}}(o_{i,t})$。

$$
\boxed{\text{GRPO:}\quad
\mathcal{J}=\mathbb{E}\Big[\frac1G\sum_i\frac1{|o_i|}\sum_t\min\!\big(\rho_{i,t}\,\underbrace{\tfrac{R_i-\bar R}{\sigma_R}}_{\hat A_i},\,\text{clip}(\rho_{i,t})\,\hat A_i\big)\Big]}
$$

$$
\boxed{\text{Dr.GRPO:}\quad
\mathcal{J}=\mathbb{E}\Big[\frac1{G\cdot\text{MAX\_TOKENS}}\sum_i\sum_t\min\!\big(\rho_{i,t}\,\underbrace{(R_i-\bar R)}_{\hat A_i},\,\text{clip}(\rho_{i,t})\,\hat A_i\big)\Big]}
$$

差异仅在两处归一化：$1/|o_i|\to1/\text{MAX\_TOKENS}$（常数），$\hat A_i$ 去掉 $1/\sigma_R$。

### 4.2 推导一：std 归一化如何扭曲梯度（难度偏差的数学根源）

设无偏 advantage $\tilde A_i=R_i-\bar R$（[[GRPO]] §4.3 已证 $\frac1G\sum_i\nabla\log\pi(o_i)\tilde A_i$ 渐近无偏，期望 $(1-1/G)\nabla J$）。GRPO 用 $\hat A_i=\tilde A_i/\sigma_R$。则 GRPO 的梯度估计：

$$
\hat g_{GRPO}=\frac1G\sum_i\nabla\log\pi(o_i)\,\frac{\tilde A_i}{\sigma_R}
=\frac{1}{\sigma_R}\cdot\underbrace{\frac1G\sum_i\nabla\log\pi(o_i)\,\tilde A_i}_{\hat g_{unbiased}\,\propto\,\nabla J}
$$

**关键**：$1/\sigma_R$ 是**逐 group（逐 prompt）**的标量，不同 prompt 的 $\sigma_R$ 不同。于是目标里不同 prompt 被乘了不同尺度：

$$
\mathcal{J}_{GRPO}=\mathbb{E}_{q\sim p_Q}\Big[\frac1{\sigma_R(q)}\cdot(\text{该 prompt 的无偏梯度贡献})\Big]
$$

而真实目标 $\mathcal{J}=\mathbb{E}_{q\sim p_Q}[\,\cdot\,]$ 按 $p_Q(q)$ 等权加权 prompt。GRPO 把权重改成了 $\frac{p_Q(q)}{\sigma_R(q)}$ —— **$\sigma_R$ 小的 prompt 权重被放大**。这就是"difficulty bias"：太易（全对 $\sigma_R\to0$）/太难（全错 $\sigma_R\to0$）的 prompt 权重虚高，挤占了难度适中、信号最强的 prompt 的权重。Dr.GRPO 去掉 $1/\sigma_R$ → 恢复 $p_Q(q)$ 等权 → 无此偏差。

> [!note] 补充：与 batch-level normalization 的区别
> 标准"advantage normalization"是 $\hat A\leftarrow(\hat A-\bar A_{batch})/\sigma_{batch}$——分母是**整个 batch 的 std**，对所有样本同一个标量 → 只是全局缩放，不改变样本间相对权重，**无害**（仅调梯度尺度，lr 吸收）。GRPO 的 std 是**逐 group** 的，才引入 per-prompt 重加权。Dr.GRPO 批判的是后者，不是前者。

### 4.3 推导二：$1/|o_i|$ 的非对称长度偏差

GRPO 目标对 response $i$ 的贡献 $\propto\frac1{|o_i|}\sum_t\rho_{i,t}\hat A_i$。设 $\rho\approx1$（on-policy 单步），则 $\approx\hat A_i$（与长度无关的总量），但**每个 token** 的梯度 $\propto\hat A_i/|o_i|$。

考虑梯度方向（参数 $\theta$）：对 response $i$，梯度 $g_i\propto\frac1{|o_i|}\sum_t\nabla\log\pi(o_{i,t})\cdot\hat A_i=\frac{\hat A_i}{|o_i|}\sum_t\nabla\log\pi(o_{i,t})$。记 $S_i=\sum_t\nabla\log\pi(o_{i,t})$（response 的总 logprob 梯度）。

- **正确回答**（$\hat A_i>0$）：梯度 $\propto+\frac{\hat A_i}{|o_i|}S_i$。$|o_i|$ 越小，系数越大 → 强化得越猛 → 模型学到"答对时短回答收益更大" → **偏好短的正确回答**。
- **错误回答**（$\hat A_i<0$）：梯度 $\propto\frac{\hat A_i}{|o_i|}S_i$（负）。$|o_i|$ 越大，$|\hat A_i|/|o_i|$ 越小 → 惩罚得越轻 → 模型学到"答错时长回答挨骂少" → **偏好长的错误回答**。

净：错误回答越长。删掉 $1/|o_i|$（Dr.GRPO 用常数分母）→ 系数 $\propto\hat A_i/\text{const}$，与 $|o_i|$ 无关 → 无长度偏差。

### 4.4 推导三：Dr.GRPO 的无偏性

Dr.GRPO 的 advantage $\hat A_i=R_i-\bar R$，正是 [[GRPO]] §4.3 的 $\tilde A_i$。其梯度估计 $\frac1G\sum_i\nabla\log\pi(o_i)(R_i-\bar R)$ 的期望 $=(1-1/G)\nabla J$（因 $\bar R$ 含自身，见 [[GRPO]] §4.3 / [[RLOO]] §4.2）——**渐近无偏**（$G$ 大时 $\to\nabla J$），且无 std/length 引入的额外重加权。若要**精确无偏**，把 $\bar R$ 换成 leave-one-out 均值（[[RLOO]]）即可。Dr.GRPO 论文的"unbiased"主要指**去掉了 std 与 length 两个额外偏差**，恢复成标准 MC policy gradient（baseline 的 $(1-1/G)$ 是已知小尺度偏差，作者按惯例当无偏处理；附录 A 给了推导）。

> [!tip] 实践
> Dr.GRPO 的"无偏"=去 std/length 偏差。baseline 自带的 $(1-1/G)$ 它没专门处理（沿用 GRPO 的 group-mean baseline）。若要连这点也消，组均值换成 leave-one-out（[[RLOO]]）。两者可叠加（Dr.GRPO + RLOO baseline）但开源实现少见。


## 5. 代码示例（可选）

### 5.1 GRPO vs Dr.GRPO advantage 与 loss 归一化对比

```python
import numpy as np

def grpo_advantage(r):
    """GRPO: (r - mean) / std  (带 std 归一化, 被批判)"""
    return (r - r.mean()) / (r.std() + 1e-8)

def drgrpo_advantage(r):
    """Dr.GRPO: r - mean  (只减均值, 不除 std)"""
    return r - r.mean()

# demo: 两个 prompt 各采 8 条
r_easy = np.array([1,1,1,1,1,1,0,1], dtype=float)   # 接近全对, std 小 -> GRPO 放大
r_hard = np.array([0,0,0,1,0,0,0,0], dtype=float)   # 接近全错, std 小 -> GRPO 放大
r_mix  = np.array([1,0,1,0,1,0,1,0], dtype=float)   # 一半一半, std 大 -> GRPO 压低

for name, r in [("easy(near all-correct)", r_easy),
                ("hard(near all-wrong)", r_hard),
                ("mixed(half-half)", r_mix)]:
    g = grpo_advantage(r)
    d = drgrpo_advantage(r)
    print(f"{name:24s} std={r.std():.3f}  GRPO_adv norm={np.linalg.norm(g):.3f}  DrGRPO_adv norm={np.linalg.norm(d):.3f}")
# 观察: GRPO 下 easy/hard(小std)的 advantage 范数反而最大 -> 难度偏差 (信息少的组占主导)
```

输出（示意）：
```
easy(near all-correct)    std=0.354  GRPO_adv norm=4.243  DrGRPO_adv norm=1.000
hard(near all-wrong)      std=0.354  GRPO_adv norm=4.243  DrGRPO_adv norm=1.000
mixed(half-half)          std=0.500  GRPO_adv norm=2.828  DrGRPO_adv norm=1.414
```
GRPO 下信息最少的 easy/hard 组（std 小）advantage 范数反而最大（4.24 > 2.83），主导梯度——难度偏差。Dr.GRPO 把它扳回（easy/hard=1.0 < mixed=1.41，信号强的组主导）。

### 5.2 masked_mean 的偏差版 vs 无偏版（arXiv:2503.20783 Listing 1）

```python
MAX_TOKENS = 2048  # 生成预算, 全程常数

def masked_mean_biased(tensor, mask, dim=-1):
    # 偏差版 (trl/OpenRLHF/verl 惯例): 按 response 长度归一 -> 长度偏差
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)

def masked_mean_unbiased(tensor, mask):
    # 无偏版 (Dr.GRPO): 除常数 MAX_TOKENS -> 与长度无关
    return (tensor * mask).sum(-1) / MAX_TOKENS

# 模拟: 两条 response, 一短(对)一长(错)
import torch
loss = torch.tensor([[1.0, 0.5, 0.8, 0.0, 0.0],     # 短(4 token), 答对 -> advantage>0
                     [0.9, 0.7, 0.6, 0.5, 0.4]])    # 长(5 token), 答错 -> advantage<0
mask = torch.tensor([[1,1,1,1,0],
                     [1,1,1,1,1]], dtype=torch.float)
print("biased   :", masked_mean_biased(loss, mask, dim=-1))     # 短回答被放大
print("unbiased :", masked_mean_unbiased(loss, mask))           # 与长度无关
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GRPO]]（Dr.GRPO 是其修正）、[[REINFORCE]] / [[policy gradient]]（恢复无偏 MC PG）、[[baseline]]（保留组均值作 baseline）、[[advantage function]]、[[token-level与sequence-level objective]]（length bias 即归一化粒度问题）、[[PPO clipped objective]]（Dr.GRPO 仍保留 clip）。
- **下游（应用）**: R1-Zero 类训练的去偏差 baseline（sail-sg/Oat-Zero-7B 43.3% AIME）、long-CoT RL 的 token 效率优化（防 overthinking）。
- **对比 / 易混**:
  - **Dr.GRPO vs [[GRPO]]**：去 std 归一化 + 去 $1/|o_i|$ 长度归一化；advantage 从 $(r-\bar r)/\sigma$ 改成 $r-\bar r$，loss 从 per-response mean 改成常数分母。详见 §3.3、§4.1。
  - **Dr.GRPO vs [[DAPO]]**：两者都修 GRPO 的病，路线相反——Dr.GRPO 做**减法**（删归一化、恢复无偏），DAPO 做**加法**（加 4 个 trick：Clip-Higher / Dynamic Sampling / Token-Level Loss / Overlong Reward Shaping）。Dr.GRPO 的"Token-Level Loss"与 DAPO 的 Trick 3 同源（都改归一化粒度），但 Dr.GRPO 用常数分母、DAPO 用 token-mean，细节不同。详见 [[DAPO]]。
  - **Dr.GRPO vs [[RLOO]]**：Dr.GRPO 修归一化偏差（std/length），保留 group-mean baseline（含自身，$(1-1/G)$ 偏差）；RLOO 修 baseline 偏差（用 leave-one-out 精确无偏），保留 std/length。两者正交，可叠加。详见 [[RLOO]]。
  - **Dr.GRPO vs [[PPO optimization]]**：Dr.GRPO 声称恢复成"无偏 PPO（Eq.2）+ group-relative MC advantage"，即 PPO 目标但去 critic、advantage 不用 GAE。


## 7. 常见误区与易错点

> [!warning] 误区 1：Dr.GRPO 是新算法
> 严格说不是。它是 GRPO 的"去两个归一化项"修正，advantage 估计同源（group-relative）。两个改动都是工程化修正非理论突破。作者自称"Done Right"即此意。

> [!warning] 误区 2：std 归一化无害（只是缩放）
> 若是**batch-level** 全局归一化则无害（全局标量）。但 GRPO 是**group-level（逐 prompt）** 归一化，不同 prompt 缩放不同 → 跨 prompt 重加权 → 偏差。Dr.GRPO 批判的是后者。别混。

> [!warning] 误区 3：GRPO 偏好短回答
> 不对称：正确回答偏短、错误回答偏长。净可见效果是**错误回答长度膨胀**。见 §3.1。

> [!warning] 误区 4：length bias 是 GRPO 独有
> 不是。trl/OpenRLHF/verl/SimpleRL-Zero/Open-Reasoner-Zero 的 PPO 实现也都按 response 长度归一化（§3.5）。是社区实现惯例病，源自预训练的 `loss.mean(-1)` 习惯被搬到 RL。

> [!warning] 误区 5：Dr.GRPO 完全精确无偏
> 它去掉了 std/length 两个额外偏差，恢复成标准 MC PG。但 baseline 用 group-mean（含自身），仍有 $(1-1/G)$ 的渐近偏差（[[GRPO]] §4.3）。要精确无偏需配 [[RLOO]] 的 leave-one-out baseline。作者按惯例当无偏。

> [!warning] 误区 6：删 std 后训练不稳（advantage 尺度乱）
> 删 std 后 advantage 不再 z-score 到 $[-1,1]$，尺度随 reward 量级变。但作者实测仍稳（可用 batch-level normalization 或调 lr 补偿）。删 std 的收益（去难度偏差）> 尺度不统一的代价。


## 8. 延伸细节

### 8.1 Dr.GRPO 的实测收益（arXiv:2503.20783 Fig.5）

- **response 长度**：GRPO 训练后期即使 reward 改善放缓，response 长度仍持续涨；Dr.GRPO 长度增长与 reward 同步，reward 饱和后长度也饱和。**错误回答长度大幅下降**（缓解 overthinking）。
- **正确率**：与 GRPO 相当或略好，但 token 效率（单位 token 的正确率提升）更高。
- **Oat-Zero-7B**：Dr.GRPO + 极简 recipe（Qwen2.5-Math-7B + MATH level 3-5 + R1 template），8×A100 仅 27 小时，AIME 2024 43.3%（7B SOTA）。

### 8.2 与 DAPO 的关系（去偏差 vs 加 trick）

Dr.GRPO 和 [[DAPO]] 都治 GRPO 的病，但路线互补：

| | Dr.GRPO | DAPO |
|---|---|---|
| 路线 | 减法（删归一化） | 加法（加 4 trick） |
| 治的病 | std 难度偏差 + length 偏差 | 熵坍塌 + 梯度消失 + 长度漂移 + 奖励噪声 |
| length 处理 | 删 $1/|o_i|$ 用常数分母 | Overlong Reward Shaping（reward 端 shaping） |
| std 处理 | 删 std | Dynamic Sampling（剔除 std=0 组） |
| 复杂度 | 极简（改 2 行） | 中等（4 个 trick） |

两者可组合：Dr.GRPO 的无偏目标 + DAPO 的 Clip-Higher / Dynamic Sampling / Overlong Shaping。社区实现常各取所需。

### 8.3 代码实现（sail-sg/Oat）

Dr.GRPO 实现在 [sail-sg/understand-r1-zero](https://github.com/sail-sg/understand-r1-zero)（基于 Oat 框架）。核心改动就是 advantage 不除 std、masked_mean 用 MAX_TOKENS。verl 框架的 `loss_agg_func` 配置可切换归一化方式（`token-mean` vs `token-sum` / 常数），与 Dr.GRPO 思路一致。

### 8.4 内容来源

arXiv:2503.20783《Understanding R1-Zero-Like Training: A Critical Perspective》（Liu et al., Sea AI Lab, 2025-03，v2 2025-10）。GitHub: sail-sg/understand-r1-zero。两偏差定义（§3.1/§3.2）、Listing 1（masked_mean）、Fig.4/5（偏差示意与实测）均来自原文 §3。公式形式以原文 v2 为准。

---
相关: [[GRPO体系]] | [[GRPO]] | [[token-level与sequence-level objective]] | [[DAPO]] | [[RLOO]] | [[REINFORCE]] | [[baseline]] | [[advantage function]] | [[variance reduction]] | [[PPO clipped objective]] | [[policy gradient]] | [[importance sampling与off-policy correction]] | [[KL penalty与KL control]]
