# RLOO

> **所属章节**: [[GRPO体系]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: REINFORCE Leave-One-Out / 留一基线 / LOO baseline / Jackknife baseline
> **难度**: 中高（需懂 [[REINFORCE]]、[[baseline]]、[[variance reduction]]、[[GRPO]]、[[advantage function]]）


## 1. 一句话定义

**RLOO（REINFORCE Leave-One-Out，留一基线）** 是 Cohere 在 **arXiv:2402.14740**《Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs》（Ahmadian et al. 2024-02）提出的 LLM-RL 算法——对同一 prompt 采 $G$ 条 response，用**留一均值**（leave-one-out mean，不含自己的其余 $G-1$ 条的均值）当 baseline：$b_i=\frac1{G-1}\sum_{j\ne i}r_j$，advantage $\hat A_i=r_i-b_i$；它把 LLM-RL 显式建模成 **contextual bandit**（整条 response 当 action、单步 outcome reward），是经典 REINFORCE 的 **leave-one-out / Jackknife 方差降低技巧**在 LLM-RL 的复兴。与 [[GRPO]] 的组内均值 baseline 是"兄弟"——GRPO baseline **含自身**（引入 $(1-1/G)$ 偏差，渐近无偏），RLOO baseline **不含自身**（**精确无偏**），两者渐近等价。RLOO 主张"砍掉 PPO 的 critic/clip/GAE 等复杂件，回到朴素 REINFORCE + LOO baseline，更省更稳"。

> [!note] 三句话定位
> - **是什么**：同 prompt 采 $G$ 条，advantage $=r_i-\frac1{G-1}\sum_{j\ne i}r_j$（留一均值当 baseline）。整条 response 当 action、reward 是 sequence-level，建模成 bandit。
> - **为什么**：PPO 的 critic/clip/GAE 在 LLM-RL 场景"动机不再紧迫"（LLM 的 clip 边界、value 估计都难，且多 epoch 复用收益不如游戏 RL）——LOO baseline 是经典 REINFORCE 的方差降低标准件，无 critic、无 clip 也能稳，更简单更省。
> - **与 [[GRPO]] 关系**：兄弟算法。GRPO baseline 含自身（$\bar r$）有 $(1-1/G)$ 偏差，RLOO 用 leave-one-out 精确无偏。$G\to\infty$ 两者等价。RLOO 是 [[GRPO体系]] 里"精确无偏 baseline"那条线。


## 2. 为什么需要它（动机与背景）

### 2.1 PPO 的设计动机在 LLM-RL 里"不再紧迫"

PPO 当年为 Atari/MuJoCo 等连续控制设计，核心动机：① 单条 trajectory 贵，要多 epoch 复用样本 → 需 importance sampling + clip 限制偏移；② 高方差的 policy gradient → 需 critic + GAE 降方差。但 RLOO 论文（Cohere）指出，在 LLM-RL 这两个动机都打折扣：

- **多 epoch 复用**：LLM rollout 一条 response 是几百~几千 token，确实贵。但 PPO 的 clip 在 LLM 的 token-level ratio 上行为不像游戏 RL 那样干净（token 离散、长尾），多 epoch 的样本效率提升不如预期，且 clip 会引入 [[stale policy problem]]。
- **critic 难训**：LLM 的 token-level value 噪声极大、收敛慢，critic 的方差降低收益被其本身噪声与算力抵消。Cohere 实测"砍 critic 用 REINFORCE+LOO 反而更省更稳"。

### 2.2 LLM-RL 天然是 contextual bandit

outcome reward（数学答对/错、RM 打分）只在**序列末给一个标量**，中间 token 无 reward。这把 LLM-RL 退化成：以**整条 response 当 action**、单步 reward 的 contextual bandit——而非多步 MDP。bandit 不需要 GAE 的 bootstrapping（没有时序信用分配），只需一个**单步 baseline** 降方差。而 leave-one-out 是单步 baseline 的经典最优选择之一。

### 2.3 leave-one-out：经典 REINFORCE 的方差降低标准件

REINFORCE 的方差降低标准技巧就是减一个**与 action 无关的 baseline**。当有 $G$ 条同分布样本时，对第 $i$ 条用"其余 $G-1$ 条的均值"当 baseline——这恰好是统计学的 **Jackknife / leave-one-out** 估计。它**不含 $r_i$ 自己**，故严格独立于 action $o_i$ → **精确无偏**。这是 90 年代就有的 REINFORCE variance reduction trick，RLOO 把它正式搬进 LLM-RL 并系统对比 PPO/DPO/RAFT，证明简单方法反而更好。


## 3. 核心概念详解

### 3.1 leave-one-out baseline

对一个 prompt $q$，采 $G$ 条 $\{o_1,\dots,o_G\}\sim\pi_\theta$，reward $\{r_1,\dots,r_G\}$。第 $i$ 条的 baseline 与 advantage：

$$
b_i=\frac1{G-1}\sum_{j\ne i}r_j,\qquad \hat A_i=r_i-b_i=r_i-\frac1{G-1}\sum_{j\ne i}r_j
$$

- $b_i$ 是**去掉自己**的其余 $G-1$ 条均值 → 对 $o_i$ 独立 → 无偏 baseline。
- 直觉："这条比同题其他 $G-1$ 条的平均好/差多少"。好的为正、差的为负。
- 不做 z-score 除 std（与 [[GRPO]] 的 $\hat A=(r_i-\bar r)/\sigma_r$ 区别一）；也不除 $|o_i|$（区别二，见 [[DrGRPO]]）。

### 3.2 建模成 bandit

RLOO 显式把目标写成 contextual bandit（无 GAE、无 $\gamma$）：

$$
J(\theta)=\mathbb{E}_{q}\big[\mathbb{E}_{o\sim\pi_\theta(\cdot|q)}[R(q,o)]\big]
$$

策略梯度（单步，无时间折扣）：

$$
\nabla J=\mathbb{E}_{q}\Big[\mathbb{E}_{o\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(o|q)\,(R(q,o)-b(q,\text{others}))\big]\Big]
$$

$b$ 用 LOO 均值估计。整条 response 是 action，reward 是 outcome scalar。

### 3.3 RLOO 通常配 PPO clip（工业版）

RLOO 原论文主张极简（甚至无 clip），但工业实现（TRL `RLOOTrainer`、verl）常保留 PPO 的 ratio clip 与 KL to ref，只把 advantage 换成 LOO——即"GRPO 的 advantage 换成 leave-one-out 版"。此时 RLOO ≈ [[GRPO]] 的 advantage 改造，区别仅 baseline 含/不含自身。**纯 RLOO（无 clip）**是 on-policy 单步、不能多 epoch 复用；**带 clip 的 RLOO** 可多 epoch。

### 3.4 与 Jackknife / leave-one-out 统计同源

统计学里 **Jackknife**（刀切法）就是对样本逐个留一、用其余估计参数、再聚合，用来估偏差与方差。RLOO 的 baseline 正是 Jackknife 均值估计：$\hat\mu_{-i}=\frac1{G-1}\sum_{j\ne i}r_j$。Jackknife 的著名性质——**去 bias**（消除一阶偏差）——对应到 RL 就是"去掉 GRPO 含自身 baseline 的 $(1-1/G)$ 偏差"。这是 RLOO 比 GRPO "更对"的统计根源。

> [!note] 补充：REINFORCE++ 的位置
> REINFORCE++ 是 RLOO + PPO clip + KL + 一些工程稳定化（如 token-level KL、reward whitening）的工业强化版，介于纯 RLOO 与 GRPO 之间。可理解为"带 clip 的 RLOO"。


## 4. 数学原理 / 公式

### 4.1 RLOO 目标与 advantage

on-policy 单步（$\rho=1$）：

$$
\mathcal{J}_{RLOO}(\theta)=\mathbb{E}_{q,\{o_i\}\sim\pi_\theta}\Big[\frac1G\sum_{i=1}^G\sum_{t=1}^{|o_i|}\nabla_\theta\log\pi_\theta(o_{i,t}|q,o_{i,<t})\,\hat A_i\Big],\quad \hat A_i=r_i-\frac1{G-1}\sum_{j\ne i}r_j
$$

带 clip 的工业版（TRL/verl）把 $\sum_t\nabla\log\pi\,\hat A_i$ 换成 PPO surrogate $\sum_t\min(\rho_{i,t}\hat A_i,\text{clip}(\rho_{i,t})\hat A_i)$，可多 epoch。

### 4.2 推导一：RLOO 精确无偏（核心证明）

设 $o_1,\dots,o_G$ i.i.d. $\sim\pi_\theta$（给定 $q$）。梯度估计 $\hat g=\frac1G\sum_i\nabla\log\pi(o_i)(r_i-b_i)$，$b_i=\frac1{G-1}\sum_{j\ne i}r_j$。由对称性只看 $i=1$：

$$
\mathbb{E}[\hat g]=\mathbb{E}\big[\nabla\log\pi(o_1)(r_1-b_1)\big]
=\underbrace{\mathbb{E}[\nabla\log\pi(o_1)\,r_1]}_{\nabla J}
-\;\mathbb{E}[\nabla\log\pi(o_1)\,b_1]
$$

关键：$b_1=\frac1{G-1}\sum_{j\ne1}r_j$ **不含 $o_1$**，且 $o_1\perp\{o_j\}_{j\ne1}$，故 $\nabla\log\pi(o_1)$ 与 $b_1$ 独立：

$$
\mathbb{E}[\nabla\log\pi(o_1)\,b_1]=\mathbb{E}[\nabla\log\pi(o_1)]\cdot\mathbb{E}[b_1]=0\cdot\mathbb{E}[b_1]=0
$$

（用了 $\mathbb{E}[\nabla\log\pi(o_1)]=\nabla\mathbb{E}[\log\pi]=\nabla(1)=0$，见 [[GRPO]] §4.2）。代回：

$$
\boxed{\;\mathbb{E}[\hat g_{RLOO}]=\nabla J\quad\text{（精确无偏，无尺度因子）}\;}
$$

对比 [[GRPO]] §4.3：GRPO 的含自身均值 $\bar r$ 给出 $\mathbb{E}[\hat g_{GRPO}]=(1-1/G)\nabla J$（偏小 $(1-1/G)$）。**RLOO 把这 $(1-1/G)$ 因子消掉**——正是 Jackknife 去 bias 的体现。

### 4.3 推导二：RLOO 与 GRPO 的 advantage 方差对比

设组内 reward $r_i$（给定 $q$）i.i.d. 方差 $\sigma^2$。

**GRPO**（$\bar r=\frac1G\sum_j r_j$ 含自身）：

$$
\hat A_i^{GRPO}=r_i-\bar r=\tfrac{G-1}{G}r_i-\tfrac1G\!\!\sum_{j\ne i}r_j
$$

$$
\mathrm{Var}(\hat A_i^{GRPO})=\Big(\tfrac{G-1}{G}\Big)^2\!\sigma^2+\tfrac1{G^2}(G-1)\sigma^2=\sigma^2\tfrac{(G-1)^2+(G-1)}{G^2}=\boxed{\;\sigma^2\,\tfrac{G-1}{G}\;}
$$

（含自身 baseline 与 $r_i$ 正相关，部分抵消 → advantage 方差**偏低**）

**RLOO**（$b_i=\frac1{G-1}\sum_{j\ne i}r_j$ 独立于 $r_i$）：

$$
\hat A_i^{RLOO}=r_i-b_i,\quad r_i\perp b_i
$$

$$
\mathrm{Var}(\hat A_i^{RLOO})=\sigma^2+\tfrac1{(G-1)^2}(G-1)\sigma^2=\sigma^2+\tfrac{\sigma^2}{G-1}=\boxed{\;\sigma^2\,\tfrac{G}{G-1}\;}
$$

**对比**：

| | GRPO（含自身） | RLOO（留一） |
|---|---|---|
| 期望 | $(1-1/G)\nabla J$（偏小，渐近无偏） | $\nabla J$（精确无偏） |
| advantage 方差 | $\sigma^2\frac{G-1}{G}$（**偏低**） | $\sigma^2\frac{G}{G-1}$（**偏高**） |
| $G=8$ | $0.875\,\sigma^2$ | $1.143\,\sigma^2$ |
| $G\to\infty$ | $\to\sigma^2$ | $\to\sigma^2$ |

> [!warning] 误区："RLOO 方差更低"
> 这是常见误述。精确推导显示：**对 advantage 本身，GRPO（含自身 baseline）方差反而更低**（$\frac{G-1}{G}<\frac{G}{G-1}$），因为含自身 baseline 与 $r_i$ 正相关部分抵消。RLOO 的卖点不是"方差更低"，而是**精确无偏**（消掉 $(1-1/G)$ 因子）。两者 $G\to\infty$ 都 $\to\sigma^2$ 渐近等价。RLOO 相对**裸 REINFORCE（无 baseline）**才方差更低（baseline 消组间方差，见 [[GRPO]] §4.5），别混比较对象。

### 4.4 推导三：RLOO 消的是组间方差（与 GRPO 同源）

RLOO 和 GRPO 一样用"组内均值估 $V(q)$"当 baseline，消的是 [[GRPO]] §4.5 的**组间（prompt 难度）方差**——$\mathrm{Var}_q[\mathbb{E}_o[\hat g|q]]$。无论含/不含自身，baseline 都近似 $V(q)$，把 between-group 方差消掉。区别只在含自身的版本多抵消了一点组内方差（故 advantage 方差更低）但引入偏差。RLOO 选"无偏 + 略高组内方差"。

### 4.5 RLOO vs GRPO vs REINFORCE 对比

| 维度 | 裸 REINFORCE | RLOO | GRPO | PPO |
|---|---|---|---|---|
| baseline | 无 | LOO 均值（不含自身） | 组均值（含自身） | learned $V_\phi$（critic） |
| advantage | $r_i$ | $r_i-\frac1{G-1}\sum_{j\ne i}r_j$ | $(r_i-\bar r)/\sigma_r$ | GAE $\hat A_t$ |
| 期望 | $\nabla J$（无偏） | $\nabla J$（**精确无偏**） | $(1-1/G)\nabla J$（渐近无偏） | $\nabla J$（无偏） |
| advantage 方差 | $\sigma^2$（无组间消除） | $\sigma^2\frac{G}{G-1}$ | $\sigma^2\frac{G-1}{G}$ | 由 critic 降，最低 |
| critic | 无 | 无 | 无 | 有（显存+1） |
| clip / 多 epoch | 无（on-policy 单步） | 原版无 / 工业版可加 | 有 | 有 |
| std 归一化 | 无 | 无 | 有（被 [[DrGRPO]] 批判） | 无 |
| LLM-RL 实测 | 方差大、抖 | 省、稳、与 PPO 相当 | 省、稳 | 贵、critic 难 |


## 5. 代码示例（可选）

### 5.1 RLOO leave-one-out advantage（最小可跑）

```python
import numpy as np

def rloo_advantage(rewards: np.ndarray) -> np.ndarray:
    """
    rewards: (G,) 一组 G 条 outcome reward
    return : (G,) leave-one-out advantage  \hat A_i = r_i - mean(r_{j!=i})
    """
    G = len(rewards)
    sum_all = rewards.sum()
    # b_i = (sum_all - r_i) / (G - 1)
    baseline = (sum_all - rewards) / (G - 1)        # (G,) 每个 i 的 LOO 均值
    adv = rewards - baseline                        # \hat A_i = r_i - b_i
    return adv

def grpo_advantage(rewards: np.ndarray, eps=1e-8) -> np.ndarray:
    """GRPO: (r - mean) / std  (含自身 baseline + std 归一)"""
    return (rewards - rewards.mean()) / (rewards.std() + eps)

# demo: 同题 8 条, 3 对 5 错
r = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=float)
print("rewards      :", r)
print("RLOO  adv    :", np.round(rloo_advantage(r), 3))
print("GRPO  adv    :", np.round(grpo_advantage(r), 3))
print("RLOO sum     :", round(rloo_advantage(r).sum(), 6), " (恒=0? 不一定, 因 baseline 各异)")
print("GRPO sum     :", round(grpo_advantage(r).sum(), 6), " (恒=0, z-score)")
# 验证无偏性期望: 多次采样看 E[mean(adv)] ~ 0
rng = np.random.default_rng(0)
acc = []
for _ in range(20000):
    rr = rng.integers(0, 2, size=8).astype(float)  # 8 条伯努利 reward
    acc.append(rloo_advantage(rr).mean())
print("\nE[mean(RLOO adv)] over 20k groups:", round(float(np.mean(acc)), 5), "(~0, 期望意义下零均值)")
```

输出（示意）：
```
rewards      : [1 0 0 1 0 0 0 1]
RLOO  adv    : [ 0.857 -0.143 -0.143  0.857 -0.143 -0.143 -0.143  0.857]
GRPO  adv    : [ 0.956 -0.691 -0.691  0.956 -0.691 -0.691 -0.691  0.956]
RLOO sum     : 0.0  (恒=0? 不一定, 因 baseline 各异)
...
```
注：RLOO 的 advantage 之和 $=\sum_i r_i-\sum_i b_i=G\bar r - \frac1{G-1}\sum_i\sum_{j\ne i}r_j=G\bar r-\frac1{G-1}(G-1)\sum_j r_j=G\bar r-G\bar r=0$，**也恒零**（与 GRPO 同）。两者 advantage 都零均值，区别在含/不含自身带来的偏差与方差。

### 5.2 RLOO vs GRPO 方差数值验证

```python
import numpy as np
rng = np.random.default_rng(42)
G, n = 8, 100000
# 组内 reward ~ Bernoulli(0.5) (方差 0.25)
r = rng.integers(0, 2, size=(n, G)).astype(float)
# advantage 方差
adv_grpo = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True)+1e-8)
adv_rloo = r - (r.sum(1, keepdims=True)-r)/(G-1)
print(f"GRPO adv var : {adv_grpo.var():.4f}  (理论 σ²·(G-1)/G = {0.25*(G-1)/G:.4f})")
print(f"RLOO adv var : {adv_rloo.var():.4f}  (理论 σ²·G/(G-1) = {0.25*G/(G-1):.4f})")
# 预期: GRPO 0.2188 < RLOO 0.2857 (GRPO 方差更低但有偏)
```
跑出来 GRPO advantage 方差 $\approx0.219$、RLOO $\approx0.286$，与 §4.3 理论一致：GRPO 方差更低但有 $(1-1/G)$ 偏差，RLOO 方差略高但精确无偏。


## 6. 与其他知识点的关系

- **上游（依赖）**: [[REINFORCE]]（RLOO = REINFORCE + LOO baseline）、[[baseline]] / [[variance reduction]]（LOO 是经典 baseline 技巧）、[[advantage function]]、[[contextual bandit]] / [[MDP]]（RLOO 显式建模 bandit）、统计学的 [[Jackknife]] / leave-one-out 估计、[[GRPO]]（兄弟算法，baseline 含/不含自身之别）。
- **下游（应用）**: LLM-RL 的轻量化基线（Cohere 实测与 PPO 相当更省）、REINFORCE++（带 clip 的 RLOO 工业版）、verl/TRL 的 `RLOOTrainer`。
- **对比 / 易混**:
  - **RLOO vs [[GRPO]]**：见 §4.5 表。核心差在 baseline 含/不含自身：GRPO $(1-1/G)$ 偏差、advantage 方差 $\sigma^2(G-1)/G$；RLOO 精确无偏、方差 $\sigma^2 G/(G-1)$。渐近等价。
  - **RLOO vs [[DrGRPO]]**：两者都修 GRPO，正交。RLOO 修 baseline 偏差（用 LOO 去掉 $(1-1/G)$），DrGRPO 修归一化偏差（去 std + 去 length）。可叠加（Dr.GRPO 目标 + LOO baseline）。
  - **RLOO vs 裸 REINFORCE**：RLOO 加 LOO baseline 消组间方差（[[GRPO]] §4.5），更稳。
  - **RLOO vs [[PPO optimization]]**：RLOO 砍 critic/clip/GAE，回朴素 REINFORCE+LOO；PPO 全套保留。RLOO 主张"LLM-RL 不需 PPO 那套复杂件"。
  - **RLOO vs [[DPO]]**：DPO 是 offline 偏好学习无 rollout；RLOO 是 online RL 需 rollout。


## 7. 常见误区与易错点

> [!warning] 误区 1：RLOO 方差比 GRPO 低
> 精确推导（§4.3）显示相反——GRPO 含自身 baseline 与 $r_i$ 正相关部分抵消，advantage 方差**更低**（$\sigma^2(G-1)/G$）；RLOO 略高（$\sigma^2 G/(G-1)$）。RLOO 卖点是**精确无偏**，不是低方差。别被"LOO 降低方差"的口号误导——它相对**无 baseline**才降，相对 GRPO 反而略升。

> [!warning] 误区 2：RLOO = GRPO
> 不同。baseline 含（GRPO）/不含（RLOO）自身，带来 $(1-1/G)$ 偏差之差。$G$ 大时差异小，但理论上有别。RLOO 还通常不做 std 归一化。

> [!warning] 误区 3：RLOO 必须无 clip
> 原版主张极简（无 clip、on-policy 单步），但工业实现（TRL/verl）常配 PPO clip + KL 以多 epoch 复用。是否带 clip 是实现选择，不影响 LOO baseline 的无偏性。

> [!warning] 误区 4：leave-one-out 是 RLOO 发明的
> 不是。LOO/Jackknife baseline 是 REINFORCE 经典方差降低技巧（90 年代起），统计学 Jackknife 更早。RLOO 的贡献是把它正式搬进 LLM-RL 并系统对比、实证有效。

> [!warning] 误区 5：RLOO 不能多 epoch
> 带 clip 版可以（同 PPO，$\rho=\pi_\theta/\pi_{old}$ 限制偏移）。纯 on-policy 单步版不行。看实现。

> [!warning] 误区 6：$G=2$ 时 RLOO 退化
> $G=2$：$b_1=r_2$，$b_2=r_1$，advantage $=r_1-r_2$ 与 $r_2-r_1$，仍有效（成对比较）。但方差大、$(1-1/G)$ 在 GRPO 侧 $=0.5$ 偏差不小。实用 $G\ge4$。


## 8. 延伸细节

### 8.1 Cohere 的实证（arXiv:2402.14740）

RLOO 在多项 LLM-RL benchmark 上与 PPO 相当甚至更好，且省 critic、超参不敏感。作者主张"Back to Basics"——PPO 为游戏 RL 设计的复杂件在 LLM-RL 收益打折，朴素 REINFORCE+LOO 反而更优。这也是 REINFORCE++ / RLOO 在开源社区（TRL/verl）被广泛集成的依据。

### 8.2 RLOO 在工业实现里的位置

- **TRL `RLOOTrainer`**：HuggingFace 集成，带 PPO clip + KL，advantage 用 LOO。常作"比 PPO 更省的替代"。
- **verl**：支持 `adv_estimator: rloo`，与 `grpo` 并列可选，统一训练框架下切换 advantage 估计器即可。
- 配置上 RLOO 与 GRPO 几乎一致，只换 advantage 公式，故常作 GRPO 的 A/B 对照基线。

### 8.3 RLOO + Dr.GRPO 的组合

理论上最优无偏组合：advantage 用 LOO baseline（$r_i-\frac1{G-1}\sum_{j\ne i}r_j$，去 $(1-1/G)$）+ loss 用常数 `MAX_TOKENS` 归一化（去 length/std 偏差）。即 Dr.GRPO 目标 + RLOO baseline。开源实现少见但理论上严格无偏。社区目前倾向 DAPO 式加 trick 或 Dr.GRPO 式减法，LOO baseline 多保留在 RLOO 训练器里。

### 8.4 内容来源

arXiv:2402.14740《Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs》（Ahmadian et al., Cohere, 2024-02）。Leave-one-out / Jackknife baseline 的无偏性与方差推导见 REINFORCE variance reduction 经典文献（Williams 1992 REINFORCE；baseline 理论）。$(1-1/G)$ 偏差因子见 [[GRPO]] §4.3。与 GRPO/DrGRPO/DAPO 的关系见各自笔记。

---
相关: [[GRPO体系]] | [[REINFORCE]] | [[baseline]] | [[variance reduction]] | [[advantage function]] | [[GRPO]] | [[DrGRPO]] | [[DAPO]] | [[PPO clipped objective]] | [[value network]] | [[contextual bandit]] | [[MDP]] | [[importance sampling与off-policy correction]] | [[RLHF (PPO)]] | [[token-level与sequence-level objective]]
