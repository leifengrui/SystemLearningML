# token-level与sequence-level objective

> **所属章节**: [[GRPO体系]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: loss 归一化粒度 / token-mean vs token-sum / sample-level vs token-level / 长度偏差归一化
> **难度**: 中高（需懂 [[GRPO]]、[[DrGRPO]]、[[policy gradient]]、[[advantage function]]、[[REINFORCE]]）


## 1. 一句话定义

**token-level 与 sequence-level objective** 指的是 LLM-RL 里**策略梯度 loss 的归一化粒度**——按 token 求和/求均值，还是按 sequence 求均值，决定了**每个 token / 每条 response 在梯度里的权重**，进而决定模型是否产生**长度偏差（length bias）**。核心结论（arXiv:2503.20783 [[DrGRPO]] 严格推导）：**token 求和不归一（或除以全 batch 总 token 数这一常数）= 无偏策略梯度**；而 **GRPO 默认的"每条 response 内 token 求均值再 batch 平均"（$1/|o_i|$，常被叫 token-level mean / sample-level）会引入长度偏差**——正确回答偏好短、错误回答偏好长，净效果错误回答越写越长。这是 [[DrGRPO]]、[[DAPO]]（Trick 3 Token-Level Loss）共同要修的病。本篇澄清三个易混方案、推导各自的梯度缩放与长度偏差方向，并指出社区术语混用。

> [!note] 三句话定位
> - **是什么**：loss 归一化有三种粒度——token-sum（不除长度，无偏）、token-mean/sample-level（除 $|o_i|$，GRPO 默认，有长度偏差）、batch-token-mean（除全 batch 总 token 数，无偏的等价缩放，DAPO/DrGRPO 用）。
> - **为什么**：归一化粒度决定每 token 权重。$1/|o_i|$ 让每条 response 等权、token 权重反比长度 → 对正/负 advantage 非对称（正确偏短、错误偏长）。删掉它恢复无偏。
> - **与 [[GRPO]] / [[DrGRPO]] 关系**：GRPO 目标里的 $\frac1{|o_i|}$ 就是 token-mean（被批判项）；Dr.GRPO 删它改用常数 `MAX_TOKENS`、DAPO 用 $\frac1{\sum|o|}$，两者都把它扳回无偏。是 [[GRPO体系]] 的"归一化工程"子题。


## 2. 为什么需要它（动机与背景）

### 2.1 两个易混的"粒度轴"——先分清

社区说"token-level / sequence-level"时常混两个轴，必须先拆开：

| 轴 | 含义 | 选项 |
|---|---|---|
| **A. reward 粒度** | reward 在哪一层给出 | sequence-level（outcome，整条一个标量）/ token-level（process/PRM，每 token 一个分） |
| **B. loss 归一化粒度** | loss 怎么对 token / sequence 取均值 | token-sum / token-mean / batch-token-mean |

本篇讲 **轴 B**（loss 归一化）。轴 A 是 reward 设计（outcome vs PRM），见 [[reward工程]]。两者正交但常被一锅炖，是误解之源。

### 2.2 为什么归一化粒度会引入长度偏差

outcome reward 下 advantage 对一条 response 的所有 token 共享（$\hat A_{i,t}=\hat A_i$）。记 response $i$ 的"原始梯度贡献" $S_i=\sum_t\nabla_\theta\log\pi(o_{i,t})$（该 response 全 token logprob 梯度之和），则序列 $i$ 的梯度 $\propto \hat A_i S_i$。**归一化粒度决定这个 $\hat A_i S_i$ 前面乘什么系数**：

- 不除（token-sum）：系数 $1$，长序列贡献更多（$\propto|o_i|$）→ 无偏（这就是真实策略梯度）。
- 除 $|o_i|$（token-mean）：系数 $1/|o_i|$，每序列等权 → **偏**（见 §4.2）。
- 除 $\sum_j|o_j|$（batch-token-mean）：系数 $1/\sum|o_j|$（全 batch 常数）→ 无偏（只是全局缩放）。

"除什么"决定长度是否耦合进梯度。LLM 里 response 长度是变量（数学题短则几十、长则几千 token），故 $|o_i|$ 不能当常数——除它就把长度耦合进去了。

### 2.3 长度偏差在 R1-Zero 训练里的可见症状

GRPO 默认 token-mean（$1/|o_i|$）→ 训练后期 response 越来越长，**尤其错误回答**。常被误读为"长 CoT 涌现"，但 [[DrGRPO]]（arXiv:2503.20783）证明这是优化偏差：删 $1/|o_i|$ 后长度增长与 reward 同步、reward 饱和后长度也饱和，错误回答长度大幅下降。这是 Dr.GRPO / DAPO 要修的核心动机。


## 3. 核心概念详解

### 3.1 三个方案的精确定义

设 batch 含 $N$ 个 prompt、每 prompt 一组 $G$ 条 response（GRPO 设定；单条也适用，令 $G{=}1$）。记响应 $i$ 长度 $L_i=|o_i|$，逐 token 的 surrogate loss $\ell_{i,t}=\min(\rho_{i,t}\hat A_i,\text{clip}(\rho_{i,t})\hat A_i)$（advantage 对 token 共享）。

**方案 A：token-sum（无偏，= 标准 PG / Dr.GRPO Eq.2）**
$$
\mathcal{J}_A=\frac1{N G}\sum_{i}\sum_{t=1}^{L_i}\ell_{i,t}
$$
每个 token 权重 $\frac1{NG}$（常数），长序列贡献更多总权重（$\propto L_i$）。**不除 $L_i$**。这就是 [[REINFORCE]] / 标准 PPO surrogate（arXiv:2503.20783 Eq.2）的原始形式——策略梯度本就是 $\sum_t\nabla\log\pi\cdot\hat A$，求和不归一。

**方案 B：token-mean / sample-level（GRPO 默认，有偏）**
$$
\mathcal{J}_B=\frac1{N G}\sum_{i}\frac1{L_i}\sum_{t=1}^{L_i}\ell_{i,t}
$$
每条 response 内先 token 平均（$\frac1{L_i}$），再 batch 平均。每序列总权重 $\frac1{NG}$（与长度无关），每 token 权重 $\frac1{NG\cdot L_i}$（反比长度）。**这是 [[GRPO]] §4.1 目标里的 $\frac1{|o_i|}$**，被 [[DrGRPO]] 批判。

**方案 C：batch-token-mean（无偏的等价缩放，DAPO Trick 3 / Dr.GRPO `MAX_TOKENS`）**
$$
\mathcal{J}_C=\frac1{\sum_j L_j}\sum_{i}\sum_{t=1}^{L_i}\ell_{i,t}
$$
每个 token 权重 $\frac1{\sum_j L_j}$（全 batch 常数），每序列总权重 $\propto\frac{L_i}{\sum_j L_j}$。**方向上等价于方案 A**（只差一个全 batch 的常数缩放，不改变梯度方向）。DAPO 叫它 "Token-Level Loss"，Dr.GRPO 用常数 `MAX_TOKENS` 当分母（更极端的常数，同样不改变方向）。

> [!note] 补充：A 和 C 为什么都无偏
> 策略梯度的无偏形式是 $\mathbb{E}[\sum_t\nabla\log\pi(o_t)\cdot\hat A]$——**逐 token 求和**。方案 A 直接实现这个求和；方案 C 在求和后除一个**全 batch 常数** $\sum_j L_j$（不依赖单条 $o_i$），常数缩放不改变期望方向 → 也无偏。两者等价（仅 lr 尺度差异）。方案 B 除的 $L_i$ **依赖单条 $o_i$** → 把长度耦合进单条权重 → 偏。

### 3.2 各方案的 token 权重与序列权重

| 方案 | 每 token 权重 | 每序列总权重 | 长序列是否吃重 | 无偏? |
|---|---|---|---|---|
| A (token-sum) | $\frac1{NG}$（常数） | $\propto L_i$（长则多） | 是（$\propto L_i$） | ✅ |
| B (token-mean, GRPO) | $\frac1{NG\cdot L_i}$（反比长度） | $\frac1{NG}$（等权） | 否 | ❌ 长度偏差 |
| C (batch-token-mean, DAPO/DrGRPO) | $\frac1{\sum L_j}$（全 batch 常数） | $\propto L_i$（长则多） | 是（$\propto L_i$） | ✅ |

### 3.3 reward 粒度（轴 A）的补充

- **sequence-level reward（outcome）**：整条一个标量（数学答对=1/错=0、RM 打分）。advantage $\hat A_{i,t}=\hat A_i$ 对 token 共享。GRPO/RLOO/DrGRPO 默认此模式。
- **token-level reward（process / PRM）**：每 token 一个分（步骤对错）。advantage 逐 token 不同。需更细的 credit assignment，但常配 token-level loss（方案 A/C）。

本篇 loss 归一化（轴 B）与 reward 粒度（轴 A）正交：outcome reward 配 token-sum loss 是无偏的，配 token-mean loss 才有长度偏差。


## 4. 数学原理 / 公式

### 4.1 三方案的统一形式

令 $w_i$ 为序列 $i$ 的归一化系数（$\sum_t\ell_{i,t}$ 前的因子）：

$$
\mathcal{J}=\sum_i w_i\sum_{t=1}^{L_i}\ell_{i,t},\qquad
w_i^A=\frac1{NG},\quad w_i^B=\frac1{NG\,L_i},\quad w_i^C=\frac1{\sum_j L_j}
$$

序列 $i$ 的梯度贡献 $\propto w_i\,\hat A_i\,S_i$（$S_i=\sum_t\nabla\log\pi(o_{i,t})$）。**$w_i$ 依赖 $L_i$ 与否决定是否引入长度耦合**：$w_i^A,w_i^C$ 不依赖 $L_i$（A 是常数、C 依赖全 batch 总和但非单条），$w_i^B\propto1/L_i$ 依赖单条长度 → 耦合。

### 4.2 推导一：方案 B（token-mean）的长度偏差——正负 advantage 非对称

方案 B 下序列 $i$ 梯度 $g_i\propto\frac{\hat A_i}{L_i}S_i$。对**正负 advantage 分别看**：

| advantage | 含义 | 系数 $\frac{\hat A_i}{L_i}$ 随 $L_i$ 变化 | 模型学到的"偏好" |
|---|---|---|---|
| $\hat A_i>0$（答对） | 要强化 | $L_i$ 越小，$\frac{|\hat A_i|}{L_i}$ 越大 → 强化越猛 | **正确回答偏短** |
| $\hat A_i<0$（答错） | 要抑制 | $L_i$ 越大，$\frac{|\hat A_i|}{L_i}$ 越小 → 惩罚越轻 | **错误回答偏长** |

**净可见效果：错误回答越来越长**（[[DrGRPO]] §3.1 原文："artificially increases response length, especially for incorrect outputs"）。这就是 GRPO 默认归一化引入的 length bias。

> [!warning] 误区："token-level mean 偏好短回答"
> 不对称。只对**正确回答**（$\hat A>0$）偏好短；对**错误回答**（$\hat A<0$）偏好长。净效果是错误回答长度膨胀。常见"偏好短/偏好长"的一句话论断都丢了 advantage 符号这个关键变量，是误导。严谨说法见上表。

### 4.3 推导二：方案 A/C 为何无偏

方案 A：$g_i\propto\hat A_i S_i$，系数与 $L_i$ 无关 → 长度不耦合进单条权重。梯度期望 $=\mathbb{E}[\sum_t\nabla\log\pi\,\hat A]=\nabla J$（标准 REINFORCE，无偏）。

方案 C：$g_i\propto\frac{L_i}{\sum_j L_j}\hat A_i S_i$... 等等，仔细——方案 C 是 $\frac1{\sum_j L_j}\sum_i\sum_t\ell_{i,t}$，序列 $i$ 贡献 $\propto\frac1{\sum_j L_j}\hat A_i S_i$（求和后除全 batch 常数）。$\frac1{\sum_j L_j}$ 对单条 $o_i$ 是常数（虽依赖全 batch，但求期望时是全局标量）→ $E[g_i]\propto\nabla J$ → **无偏**（仅全局缩放，lr 吸收）。

故 A、C 都还原无偏 PG。Dr.GRPO 的 `MAX_TOKENS` 常数分母是 C 的特例（分母换成更固定的常数），DAPO 的 $\frac1{\sum L_j}$ 是 C 的标准形式。两者都把方案 B 扳回无偏。

### 4.4 推导三：梯度尺度的等价缩放关系

三方案的梯度 $g\propto$（忽略 advantage 共享、clip 等细节）：

$$
g_A\propto\sum_i\hat A_i S_i,\quad
g_B\propto\sum_i\frac{\hat A_i}{L_i}S_i,\quad
g_C\propto\frac1{\sum_j L_j}\sum_i\hat A_i S_i=\frac{g_A}{\sum_j L_j}
$$

故 $g_C=g_A/(\sum_j L_j)$ —— **C 与 A 只差一个全 batch 常数**，方向完全一致。$g_B$ 则把每项额外乘了 $1/L_i$（依赖单条）→ **改变方向**（不同序列按长度重加权）→ 偏。

> [!tip] 实践
> 要无偏：用方案 A（token-sum）或方案 C（batch-token-mean / DAPO Token-Level Loss / Dr.GRPO MAX_TOKENS）。要复现 GRPO 默认（有偏但常稳）：用方案 B（token-mean）。verl 的 `loss_agg_func` 可切：`token-mean`（B，GRPO 默认）/ `token-sum`（A）/ `sample-mean` 等。

### 4.5 与"标准 PG 是 token-level"的澄清

社区常说"标准 PG 是 token-level（每 token $\log\pi\cdot\hat A$）"——这指**方案 A**（token-sum，逐 token 贡献，无 $1/L_i$）。但又常把 GRPO 的"token-level mean"（方案 B，有 $1/L_i$）也叫"token-level"——**两个"token-level"指不同方案**，是术语撞车。混淆点：

- "标准 PG token-level" = 方案 A（sum，无偏）
- "GRPO token-level mean" = 方案 B（mean，有偏）

[[DrGRPO]] 批判的是后者（B），不是前者（A）。看代码里 `masked_mean` 的分母：`mask.sum(dim)`（按响应长度，B，偏）vs `MAX_TOKENS`/`sum_all`（常数/A/C，无偏）——一眼区分。


## 5. 代码示例（可选）

### 5.1 三方案 loss 归一化对比（最小可跑）

```python
import numpy as np

# 模拟: batch=1 prompt, G=2 条 response
# response 1: 短(3 token), 答对 -> advantage = +1
# response 2: 长(6 token), 答错 -> advantage = -1
# 逐 token 的 surrogate loss \ell_{i,t} 假设都是 1 (简化, 实际是 min(rho*A, clip*A))
L    = np.array([3, 6])                          # 长度
A    = np.array([1.0, -1.0])                     # advantage (对=+1, 错=-1)
ell  = [np.ones(L[i]) * A[i] for i in range(2)] # 每条逐 token loss (token 共享 A)

def loss_token_sum(ell):          # 方案 A: 不除长度 (无偏)
    return sum(e.sum() for e in ell) / len(ell)
def loss_token_mean(ell):         # 方案 B: 每条 token 平均 (GRPO 默认, 有偏)
    return sum(e.mean() for e in ell) / len(ell)
def loss_batch_token_mean(ell):  # 方案 C: 除全 batch 总 token (DAPO/DrGRPO, 无偏)
    total = sum(len(e) for e in ell)
    return sum(e.sum() for e in ell) / total

print("A token-sum        :", loss_token_sum(ell))        # (3*1 + 6*(-1))/2 = -1.5
print("B token-mean(GRPO) :", loss_token_mean(ell))       # (1 + (-1))/2 = 0.0
print("C batch-token-mean :", loss_batch_token_mean(ell)) # (3-6)/9 = -0.333
```

观察：
- 方案 B（GRPO）下两条 response 的 loss 贡献被 $1/L_i$ 抹平：短对（$+1/3\cdot3=+1$）和长错（$-1/6\cdot6=-1$）等量抵消 → **长错回答的惩罚被长度稀释了**（这正是 length bias：长错挨骂轻）。
- 方案 A/C 下长序列贡献按长度计：长错回答惩罚 $\propto -6$（更重）→ 无偏，不被长度稀释。

### 5.2 验证长度偏差方向（对 vs 错的梯度幅度）

```python
import numpy as np
rng = np.random.default_rng(0)
# 固定 advantage, 变长度, 看每条 response 的梯度幅度 ∝ w_i * |A| * L_i (因 S_i ∝ L_i 粗略)
def grad_mag_scheme(L, A, scheme):
    if scheme == 'A':   w = 1.0                       # 不除
    elif scheme == 'B': w = 1.0 / L                   # 除 L_i
    elif scheme == 'C': w = 1.0 / L.sum()             # 除全 batch
    return w * A * L   # |S_i| ∝ L_i 粗略假设

# 对(+1)的短 vs 长; 错(-1)的短 vs 长
for A, tag in [(1.0,"correct"), (-1.0,"wrong")]:
    for L in [3, 9]:
        gB = grad_mag_scheme(np.array([L]), np.array([A]), 'B')
        print(f"{tag:8s} L={L}  GRPO(B) 梯度幅度: {gB[0]:+.3f}")
# correct: L=3 -> +1.0, L=9 -> +1.0  (等权: 短回答强化更猛 per-token)
# wrong  : L=3 -> -1.0, L=9 -> -1.0  (等权: 长回答惩罚更轻 per-token)
# -> correct 偏短(短回答每token强化大), wrong 偏长(长回答每token惩罚小)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GRPO]]（其目标 $\frac1{|o_i|}$ 即方案 B）、[[DrGRPO]]（删 $1/|o_i|$ 改常数 `MAX_TOKENS`，方案 C）、[[DAPO]]（Trick 3 Token-Level Loss 即方案 C）、[[REINFORCE]] / [[policy gradient]]（方案 A 是标准 PG 原形）、[[advantage function]]、[[reward工程]]（轴 A reward 粒度，outcome vs PRM）。
- **下游（应用）**: LLM-RL 框架（verl `loss_agg_func`、TRL、OpenRLHF）的归一化选型；R1-Zero 复现的 length-inflation 诊断与修复；long-CoT 训练的 token 效率。
- **对比 / 易混**:
  - **token-sum(A) vs token-mean(B, GRPO) vs batch-token-mean(C)**：见 §3.2 表。A/C 无偏，B 有长度偏差。
  - **loss 归一化粒度(轴B) vs reward 粒度(轴A)**：正交。本篇是轴 B。轴 A 见 [[reward工程]]。
  - **"标准 PG token-level"(A) vs "GRPO token-level mean"(B)**：术语撞车，前者无偏后者有偏，看 `masked_mean` 分母区分。§4.5。
  - **[[DrGRPO]] MAX_TOKENS vs [[DAPO]] Token-Level Loss**：都实现方案 C（无偏），前者用固定常数 `MAX_TOKENS` 当分母、后者用全 batch 总 token $\sum L_j$ 当分母。差别仅在分母是"固定预算"还是"当前 batch 实际总长"，方向一致。


## 7. 常见误区与易错点

> [!warning] 误区 1："token-level 一定偏好短 / sequence-level 一定偏好长"
> 这是丢掉 advantage 符号的一句话论断。严谨结论（方案 B）：**正确回答偏短、错误回答偏长**，净效果错误回答膨胀。对方案 A/C（无偏）则无偏好。看方案与 advantage 符号再下结论。§4.2。

> [!warning] 误区 2："GRPO 的 token-level = 标准 PG 的 token-level"
> 术语撞车。GRPO 默认是方案 B（token-mean，有 $1/L_i$，偏）；标准 PG 是方案 A（token-sum，无 $1/L_i$，无偏）。看 `masked_mean` 分母：`mask.sum(dim)`（B 偏）vs `MAX_TOKENS`/`sum_all`（A/C 无偏）。

> [!warning] 误区 3："除以长度只是无害缩放"
> 若除**全 batch 总长**（方案 C）是无害全局缩放。但除**单条长度** $L_i$（方案 B）是逐条重加权 → 改变梯度方向 → 偏。是否"无害"看分母依赖不依赖单条 $o_i$。

> [!warning] 误区 4：length bias 只在 GRPO
> 不是。[[DrGRPO]] §3.5 指出 trl/OpenRLHF/verl/SimpleRL-Zero/Open-Reasoner-Zero 的 PPO 实现也都按 response 长度归一化（方案 B）。是社区实现惯例病，源自预训练 `loss.mean(-1)` 习惯被搬到 RL。

> [!warning] 误区 5："sequence-level reward 就一定配 sequence-level loss"
> 正交。outcome reward（轴 A sequence-level）可配 token-sum loss（轴 B 方案 A，无偏）或 token-mean（方案 B，有偏）。reward 在哪层给 与 loss 怎么归一化是两件事。

> [!warning] 误区 6：方案 A 会让长回答主导（不平衡）
> 方案 A 下长序列贡献 $\propto L_i$ 确实更多，但这是**真实策略梯度**（$\sum_t\nabla\log\pi\cdot\hat A$ 本就逐 token 求和），不是 bug。若担心长回答主导，那是 reward/advantage 设计问题（如加 length penalty），不是归一化该解决的——强行用 $1/L_i$ 平衡会引入 length bias。

> [!warning] 误区 7：Dr.GRPO 和 DAPO 的 Token-Level Loss 是不同算法
> 同一思路（方案 C，去 $1/L_i$ 恢复无偏），实现细节略异（Dr.GRPO 用固定 `MAX_TOKENS`、DAPO 用 batch 总长 $\sum L_j$）。方向一致，可视为同构。


## 8. 延伸细节

### 8.1 verl 的 `loss_agg_func` 配置

verl 框架统一抽象 advantage 估计与 loss 归一化：
- `adv_estimator: grpo / rloo / ...`（advantage 公式，见 [[GRPO]]/[[RLOO]]）
- `loss_agg_func: token-mean / token-sum / sample-mean / ...`（归一化粒度，本篇轴 B）
- `use_valid_token_scale: True`（DAPO 用 valid token 数当分母，方案 C 变体）

切换 `loss_agg_func` 即在方案 A/B/C 间切。GRPO recipe 默认 `token-mean`（B），DAPO recipe 用 `token-mean + use_valid_token_scale`（C 变体），Dr.GRPO 用 `token-sum` 或常数分母（A/C）。

### 8.2 MAX_TOKENS 常数分母的细节

[[DrGRPO]] 用 `MAX_TOKENS`（生成预算，全程固定）当分母：`masked_mean = (tensor*mask).sum(-1) / MAX_TOKENS`。等价于把每条 response 的 loss 求和后除一个**全程常数**。比方案 C 的 $\sum L_j$（随 batch 变）更"固定"，梯度尺度更稳。两者方向一致（都不依赖单条 $o_i$）。

### 8.3 与 reward length penalty 的区别

length bias（本篇）是**算法/归一化引入的偏差**，应从归一化层修（删 $1/L_i$）。另一种"response 太长"是**模型行为问题**，可用 reward 端 length penalty（如 Overlong Reward Shaping，[[DAPO]] Trick 4）从外部约束。两者正交：归一化层去 bias + reward 层加 penalty 可叠加。别把"用归一化修"和"用 reward 罚"混为一谈。

### 8.4 内容来源

[[DrGRPO]] arXiv:2503.20783 §3.1（length bias 定义）、§3.2（Dr.GRPO 删 $1/|o_i|$ 用 MAX_TOKENS）、Listing 1（masked_mean 偏差 vs 无偏）。[[DAPO]] arXiv:2503.14476 Trick 3（Token-Level Loss）。三方案形式化与梯度缩放推导基于标准 [[policy gradient]] / [[REINFORCE]]。verl `loss_agg_func` 见 verl 源码 `verl/trainer/ppo/core_algos.py`。开源 PPO 实现的 length bias 见 arXiv:2503.20783 Table 2。

---
相关: [[GRPO体系]] | [[GRPO]] | [[DrGRPO]] | [[DAPO]] | [[REINFORCE]] | [[policy gradient]] | [[advantage function]] | [[reward工程]] | [[value network]] | [[baseline]] | [[variance reduction]] | [[PPO clipped objective]] | [[RLHF (PPO)]] | [[importance sampling与off-policy correction]] | [[rollout train reference logprob一致性]]
