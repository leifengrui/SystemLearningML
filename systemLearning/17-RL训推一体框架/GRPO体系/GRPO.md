# GRPO

> **所属章节**: [[GRPO体系]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: Group Relative Policy Optimization / 组内相对策略优化 / 组相对优势 / 去 critic 的 PPO
> **难度**: 高（需懂 [[REINFORCE]]、[[PPO clipped objective]]、[[advantage function]]、[[baseline]]、[[variance reduction]]、[[value network]]）


## 1. 一句话定义

**GRPO（Group Relative Policy Optimization，组内相对策略优化）** 是 DeepSeek 在 **DeepSeekMath**（arXiv:2402.03300，Shao et al. 2024）提出的 LLM 强化学习算法——对**同一个 prompt 采样 G 条 response 组成一个 group**，用 group 内 reward 的**均值当 baseline**、组内相对差当 advantage（$\hat A_i=(r_i-\bar r)/\sigma_r$），从而**去掉 PPO 里的 critic / value network**，省一半算力与显存；它本质是 **REINFORCE + group-mean baseline** 的方差降低变体，再套上 [[PPO clipped objective]] 的 clip 与到 reference policy 的 KL 约束。DeepSeekMath / DeepSeek-V3 / DeepSeek-R1 均用 GRPO 做 RL 后训练，是当前 LLM-RL（尤其数学/代码等可验证任务）的事实基线之一。

> [!note] 三句话定位
> - **是什么**：同 prompt 采 G 条 response，用组内 reward 均值当 baseline，advantage = 组内相对差（z-score）；去 critic，套 PPO clip + KL to ref。
> - **为什么**：PPO 的 critic（value network）在 LLM 场景**难训、贵、显存翻倍**；GRPO 用"组内均值"这个**不依赖 action 的 Monte-Carlo baseline** 代替 $V_\phi(s)$，省掉一整个网络，且数学/代码任务 reward 可 rule-based 验证、天然适合多采样比较。
> - **与 [[RLHF (PPO)]] 关系**：GRPO 是 PPO 在 LLM 的**精简变体**——保留 clip 与 KL，但把 advantage 估计从"GAE+critic"换成"group-mean baseline + MC return"。是 [[17-RL训推一体框架]] 的核心算法。


## 2. 为什么需要它（动机与背景）

### 2.1 PPO 的 critic 在 LLM 场景很贵也很难训

标准 [[RLHF (PPO)]] 用 [[PPO optimization]]：policy（actor）+ value network（critic）+ reference policy + reward model，**四个模型同屏**。其中 critic $V_\phi(s)$ 用来估 baseline、算 [[GAE]] advantage $A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$（$\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$）。在 LLM-RL 里这带来三个痛点：

1. **显存翻倍**：critic 与 actor 同规模（都跑完整 forward/backward），训练时 actor + critic + ref + rm 四份权重，显存吃紧。大模型 RLHF 卡显存的主要是 critic。
2. **critic 难训**：LLM 的 state 是"已生成 token 序列"，$V_\phi(s_t)$ 要估"当前这串 token 之后还能拿多少 reward"。token 级 value 噪声极大、收敛慢，比游戏 RL 的像素 state 难估得多。
3. **rollout 量巨大**：LLM 生成一条 response 要几百~几千 token forward，PPO 每 step 采成千上万条，rollout 是墙钟时间大头；critic 还要多跑一遍 forward 算每个 token 的 value，进一步拖慢。

### 2.2 数学/代码任务的 reward 可 rule-based 验证 → 多采样可比较

数学题答案对错可用 verifier 判（最终答案 `\boxed{}` 是否正确）、代码题可用单测判（pass/fail）。这类**稀疏但可验证**的 reward 有个关键性质：**同一 prompt 多采几条 response，它们的好坏可直接横向比较**（对=1，错=0，或部分分）。这正好给"组内相对 advantage"提供了土壤——PPO 在通用对话场景训 RM+critic，是因为 reward 不可直接比较；而数学场景 reward 本身就是可比的标量。

> [!tip] 实践
> GRPO 的甜区是 **reward 可 rule-based 验证** 的任务（数学/代码/逻辑题/格式约束）。若 reward 只能靠 RM 打分且噪声大、prompt 间无可比性，GRPO 的 group-mean baseline 不如 PPO 的 learned critic 稳。

### 2.3 思路：用"组内均值"代替 learned value 当 baseline

REINFORCE 的策略梯度：$\nabla J(\theta)=\mathbb{E}_{o\sim\pi_\theta}[\nabla_\theta\log\pi_\theta(o|q)\cdot R(q,o)]$。减一个**不依赖 action 的 baseline** $b$ 不改变期望（无偏），但能降方差：$\nabla J=\mathbb{E}[\nabla\log\pi\cdot(R-b)]$。PPO 的 $b=V_\phi(s)$ 是学出来的；GRPO 的洞察是——**对同一 prompt，$G$ 条 response 的 reward 均值 $\bar r$ 就是 $V(q)=\mathbb{E}_{o\sim\pi}[R(q,o)]$ 的 Monte-Carlo 估计**，可直接当 baseline，不必训 critic。于是 advantage = 组内相对差，critic 整个网络被砍掉。

```
PPO:   advantage_t = GAE(return_t, V_φ)        ← 需训 V_φ (critic), 显存+1
GRPO:  advantage_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)   ← 组内 MC 估计, 无 critic
```


## 3. 核心概念详解

### 3.1 group 采样

对一个 prompt $q$，用当前 policy $\pi_{\theta_{old}}$（off-policy 的行为策略，rollout 时的权重）采 $G$ 条 response $\{o_1,\dots,o_G\}$（如 $G=8/16/64$）。每条算一个 outcome reward $r_i=R(q,o_i)$（数学题：答对=1 否则=0；也可加格式/过程 reward）。这 $G$ 条组成一个 **group**。GRPO 一次训练 batch 含 $N$ 个 prompt，故总 rollout = $N\times G$ 条 response。

> [!note] 补充：group size 的工程意义
> $G$ 越大，$\bar r$ 对 $V(q)$ 估计越准、advantage 方差越低，但 rollout 成本线性涨。DeepSeekMath 用 $G=64$，R1 报告更大。verl/DAPO recipe 常用 $G=16$。$G$ 太小（如 4）advantage 噪声大、训练不稳。

### 3.2 组内相对 advantage

$$
\hat A_i=\frac{r_i-\bar r}{\sigma_r},\quad \bar r=\frac{1}{G}\sum_{j=1}^G r_j,\quad \sigma_r=\sqrt{\frac{1}{G}\sum_j(r_j-\bar r)^2}
$$

- $\bar r$：组内均值，是 $V(q)$ 的 MC 估计 → **baseline**。
- $r_i-\bar r$：组内相对差 → 比"绝对 reward"更能反映"这条比同 prompt 其他条好/差多少"。
- 除 $\sigma_r$（z-score 归一化）：把 advantage 缩放到大致 $[-1,1]$ 量级，跨 prompt 统一尺度。**注意这步会引入"难度偏差"——见 [[DrGRPO]] 的批判。**
- advantage **对一条 response 的所有 token 共享同一个值**（outcome reward 是 sequence-level），即 $\hat A_{i,t}=\hat A_i$ 对 $t=1..|o_i|$。

### 3.3 套 PPO clip + KL to reference

GRPO 不只做 REINFORCE，还套上 PPO 的 importance-sampling clip（用 [[importance sampling与off-policy correction]]，训推权重不同时 $\rho=\pi_\theta/\pi_{old}$）和对 reference policy 的 KL 约束（[[KL penalty与KL control]]，防 policy 漂太远）。完整目标见 §4.1。

> [!warning] 误区：GRPO = REINFORCE
> 不完全等价。GRPO 的 advantage 估计确实是 REINFORCE 风格（MC return + baseline），但**目标函数套了 PPO 的 ratio clip**（$\min(\rho\hat A,\text{clip}(\rho,1-\varepsilon,1+\varepsilon)\hat A)$），允许多 epoch 复用同一批 rollout（off-policy）。纯 REINFORCE 是 on-policy 单步、无 clip。所以 GRPO = **PPO 的 advantage 换成 group-relative + 去 critic**，不是裸 REINFORCE。

### 3.4 去 critic 带来的省

| 项目 | PPO（RLHF） | GRPO |
|---|---|---|
| 训练时同屏模型 | actor + **critic** + ref + rm | actor + ref + rm（critic 砍掉） |
| critic 前向/反向 | 每条 response 全序列跑一遍算 token value | 无 |
| critic 显存 | ~actor 量级（同 size，optim states 同样大） | 0 |
| value 训练 step | 有（critic loss + GAE） | 无 |
| advantage 估计 | GAE（bootstrapped，需 $V_\phi$） | group-mean baseline（MC，无 bootstrap） |

大模型 RLHF 里 critic 显存常是 actor 的 1×（权重+优化器状态同量级），砍掉 critic ≈ **省一半训练侧显存**，且 rollout 不再需 critic forward。这是 GRPO 在工程上最大的吸引力。

### 3.5 DeepSeekMath / V3 / R1 用 GRPO 的实证

- **DeepSeekMath**（2024.02）：首提 GRPO，在 7B/67B 数学模型上 RL，验证去 critic 后仍能涨 MATH/GSM8K 分；引入 unbiased KL estimator（Schulman k3 近似的无偏版，见 §8.2）。
- **DeepSeek-V3**（2024.12）：通用大模型，RL 阶段用 GRPO（reward = rule-based + RM 混合）。
- **DeepSeek-R1 / R1-Zero**（2025.01）：R1-Zero 直接对 base model 跑 GRPO（纯 rule-based reward，无 SFT），涌现"Aha moment"与长思维链；R1 = cold-start SFT + GRPO 多阶段。**GRPO 是 R1 的 RL 引擎**。


## 4. 数学原理 / 公式

### 4.1 GRPO 完整目标函数

对 prompt 集 $\mathcal{Q}$，每个 $q$ 采 $G$ 条 $\{o_i\}_{i=1}^G\sim\pi_{\theta_{old}}(\cdot|q)$，reward $r_i=R(q,o_i)$，advantage $\hat A_i=(r_i-\bar r)/\sigma_r$（对 token $t$ 共享）。目标（DeepSeekMath 原式，token-level mean 归一化 $1/|o_i|$，见 [[token-level与sequence-level objective]]）：

$$
\mathcal{J}_{GRPO}(\theta)=\mathbb{E}_{q\sim\mathcal{Q},\,\{o_i\}\sim\pi_{\theta_{old}}}
\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\Bigg(
\min\!\Big(\rho_{i,t}(\theta)\,\hat A_i,\;
\text{clip}\big(\rho_{i,t}(\theta),\,1-\varepsilon,\,1+\varepsilon\big)\,\hat A_i\Big)
-\beta\,\mathbb{D}_{KL}^{(t)}\!\big[\pi_\theta\,\|\,\pi_{ref}\big]
\Bigg)\Bigg]
$$

- $\rho_{i,t}(\theta)=\dfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})}$：importance sampling 比（[[importance sampling与off-policy correction]]）。rollout 用 $\theta_{old}$，训练用 $\theta$，clip 限制每 epoch 偏移。详见 [[rollout train reference logprob一致性]]——训推两套 logprob 必须同分布否则 $\rho$ 失真。
- $\varepsilon$：clip ratio（典型 0.2，DAPO 解耦成 $\varepsilon_{low}/\varepsilon_{high}$，见 [[DAPO]]）。
- $\beta\,\mathbb{D}_{KL}$：到 reference policy 的 KL 约束（[[KL penalty与KL control]]）。DeepSeekMath 用**无偏 KL estimator**（k3 近似的无偏版，§8.2），逐 token 算后取均值。R1-Zero 场景因 reward 是 rule-based（无分布偏移顾虑）常设 $\beta=0$ 或极小。
- $\frac{1}{G}\frac{1}{|o_i|}$：先组内 token 平均、再 batch 平均——**token-level mean 归一化**，被 [[DrGRPO]] 批判引入长度偏差（见 §4.5 与 [[token-level与sequence-level objective]]）。

> [!note] 补充：advantage 是 sequence-level 的
> 数学/代码 outcome reward 只在序列末给一个标量 $r_i$，故 $\hat A_{i,t}=\hat A_i$ 对所有 $t$ 相同。这把 LLM-RL 退化成 **contextual bandit**（以整条 response 为 action，单步 reward）——这也是 [[RLOO]] 把它显式建模成 bandit 的依据。若用 process reward（PRM，每步给分）则 $\hat A_{i,t}$ 可不同，但 GRPO 原版是 outcome-only。

### 4.2 推导一：baseline 不改变期望（无偏性的基础）

设 action $a$（这里是一条 response $o$）从 $\pi_\theta(\cdot|q)$ 采样，return $R$。REINFORCE 梯度：

$$
\nabla J(\theta)=\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a|q)\cdot R\big]
$$

减一个**与 action $a$ 无关**的 baseline $b$（可以是 $q$ 的函数）：

$$
\mathbb{E}\big[\nabla\log\pi(a)\,(R-b)\big]
=\underbrace{\mathbb{E}[\nabla\log\pi(a)\,R]}_{\nabla J}
-\;b\,\underbrace{\mathbb{E}[\nabla\log\pi(a)]}_{=\,0}
=\nabla J
$$

其中 $\mathbb{E}[\nabla\log\pi(a)]=\int\pi(a)\nabla\log\pi(a)da=\int\nabla\pi(a)da=\nabla\int\pi(a)da=\nabla 1=0$。**故只要 $b$ 不依赖 $a$，梯度期望不变（无偏）。** 这是"baseline 可降方差但不引入 bias"的根源。

### 4.3 推导二：GRPO 的 group-mean baseline 的（近似）无偏性 + 精确偏差因子 $(1-1/G)$

GRPO 的 baseline $\bar r=\frac1G\sum_j r_j$ **包含了 $r_i$ 自己**，故严格说**不独立于 $o_i$**。下面精确推导它带来的偏差。

梯度估计量（忽略 clip，on-policy $\rho=1$，单步）：

$$
\hat g=\frac1G\sum_{i=1}^G \nabla\log\pi(o_i|q)\,(r_i-\bar r),\qquad \bar r=\frac1G\sum_j r_j
$$

各 $o_i$ i.i.d. $\sim\pi_\theta$。由对称性只看 $i=1$：

$$
\mathbb{E}[\hat g]=\mathbb{E}\big[\nabla\log\pi(o_1)\,(r_1-\bar r)\big]
=\underbrace{\mathbb{E}[\nabla\log\pi(o_1)\,r_1]}_{=\nabla J}
-\;\mathbb{E}\big[\nabla\log\pi(o_1)\,\bar r\big]
$$

把 $\bar r=\frac1G r_1+\frac1G\sum_{j\ne1}r_j$ 拆开：

$$
\mathbb{E}[\nabla\log\pi(o_1)\,\bar r]
=\frac1G\underbrace{\mathbb{E}[\nabla\log\pi(o_1)\,r_1]}_{\nabla J}
+\frac1G\sum_{j\ne1}\underbrace{\mathbb{E}[\nabla\log\pi(o_1)\,r_j]}_{\text{$o_1\perp o_j$ $\Rightarrow$ $=\mathbb{E}[\nabla\log\pi(o_1)]\cdot\mathbb{E}[r_j]=0$}}
=\frac1G\nabla J
$$

代回：

$$
\boxed{\;\mathbb{E}[\hat g]=\Big(1-\frac1G\Big)\nabla J\;}
$$

**结论**：GRPO 的 group-mean baseline 使梯度估计**偏小一个常数因子 $(1-1/G)$**（$G{=}8$ 时为 $0.875$，$G{=}64$ 时为 $0.984$）。

- $G\to\infty$：因子 $\to1$，**渐近无偏**。实践中 $G\ge8$ 偏差 $<12\%$，且是均匀缩放（不改变梯度方向），学习率可吸收 → 工程上当作"无偏"。
- $G=1$：因子 $=0$，退化（baseline=reward，梯度消失）——故 GRPO 要求 $G\ge2$。
- 这是 [[RLOO]] 用 **leave-one-out**（不含自身的均值）实现**精确无偏**的动机——见 [[RLOO]] §4.2。

> [!tip] 实践
> "GRPO 无偏"是**渐近说法**。严格讲它有 $(1-1/G)$ 的尺度偏差。但因偏差是常数标量、不偏方向，且 $G$ 通常 $\ge8$，工程上可忽略。若要数学上严格无偏，用 [[RLOO]]。

### 4.4 推导三：为什么 $E[\hat A_i]=0$（advantage 零均值）

注意"$\hat A$ 零均值"和"梯度无偏"是两件事。$\hat A_i=(r_i-\bar r)/\sigma_r$：

$$
\mathbb{E}_{i}[\hat A_i]=\frac{\mathbb{E}[r_i]-\mathbb{E}[\bar r]}{\sigma_r}=\frac{\mu-\mu}{\sigma_r}=0
$$

即组内 advantage 均值恒为 0（正负抵消）。这是"组内相对"的直观：好的为正、差的为负、平均归零。但它不保证梯度无偏（无偏性取决于 baseline 是否独立于 action，见 §4.3）。

### 4.5 推导四：方差降低——baseline 消的是"组间（prompt 难度）方差"

方差分解（law of total variance），把 prompt $q$ 当条件：

$$
\mathrm{Var}_{q,o}[\hat g]=\underbrace{\mathrm{Var}_q\!\big[\mathbb{E}_o[\hat g\mid q]\big]}_{\text{组间：prompt 难度方差}}+\underbrace{\mathbb{E}_q\!\big[\mathrm{Var}_o[\hat g\mid q]\big]}_{\text{组内：采样方差}}
$$

- **无 baseline（裸 REINFORCE）**：$\hat g\propto\nabla\log\pi\cdot r$。不同难度 prompt 的 $r$ 量级差异巨大（简单题几乎全对 $r\approx1$、难题全错 $r\approx0$），组间项 $\mathrm{Var}_q[\cdot]$ 很大 → 总方差大、训练抖。
- **GRPO（减组内均值 $\bar r$）**：$\bar r$ 是 $V(q)=\mathbb{E}[r|q]$ 的 MC 估计，$\mathbb{E}_o[r-\bar r\mid q]\approx0$，于是 $\mathbb{E}_o[\hat g|q]\approx0$ → **组间项被消掉**，只剩组内采样方差（$\propto1/G$）。这是 GRPO 降方差的本质——**消的是 prompt 难度带来的 between-group 方差**，这正是 LLM-RL 主导方差源。

方差降低量近似：设组间 reward 方差 $\sigma_b^2$、组内方差 $\sigma_w^2$。裸 REINFORCE 方差 $\approx\sigma_b^2+\sigma_w^2$；GRPO $\approx\sigma_w^2/G+\sigma_w^2\approx\sigma_w^2$（$\bar r$ 估计噪声 $\sim\sigma_w/\sqrt G$）。**净降 $\approx\sigma_b^2$**（prompt 难度方差，通常远大于组内）。

> [!note] 补充：与最优 baseline 的差距
> 最优 baseline $b^*=\frac{\mathbb{E}[(\nabla\log\pi)^2 r]}{\mathbb{E}[(\nabla\log\pi)^2]}$（带梯度加权的 reward 均值）。GRPO 用未加权的组内均值 $\bar r$ 近似 $V(q)$，不是最优但简单、无需训练。PPO 的 learned $V_\phi$ 逼近 $b^*$ 更精细 → 方差更低，代价是训 critic。GRPO 用"采样换训练"：多采 $G$ 条把 $V(q)$ 估出来，省 critic。

### 4.6 与 GAE 的关系

[[GAE]] 的 advantage：$\hat A_t^{GAE}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$，$\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$（TD 误差，bootstrapped）。

GRPO 可看作 GAE 的两个极端简化：

| 维度 | GAE（PPO） | GRPO |
|---|---|---|
| $\lambda$（bootstrapping 程度） | $\lambda\in[0,1]$ 可调 | $\lambda=1$（纯 MC，不 bootstrap） |
| baseline $V$ | learned $V_\phi(s_t)$（token 级） | 组内均值 $\bar r$（prompt 级 MC 估计） |
| token-level advantage | $\hat A_t$ 逐 token 不同 | $\hat A_{i,t}=\hat A_i$ 全序列共享 |
| 需 critic | 是 | 否 |

即 **GRPO = GAE with $\lambda=1$（MC return）+ $V$ 用 group-mean 代替 + advantage sequence-level 广播**。它放弃了 bootstrapping 的方差降低（$\lambda<1$ 让 $\hat A$ 依赖 $V$ 缩短回报 horizon 降方差），换得不训 critic。代价：outcome-only 时 advantage 在整条序列上恒定，token 级 credit assignment 粗（全序列共享一个信号）——这是 process reward（PRM）想补的点。


## 5. 代码示例（可选）

### 5.1 手写 GRPO group advantage（最小可跑）

```python
import numpy as np

def grpo_advantage(rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    rewards: shape (G,) 一组 G 条 response 的 outcome reward
    return : shape (G,) 组内相对 advantage (z-score)
    """
    mean = rewards.mean()
    std  = rewards.std()                       # 注: std 归一化会被 Dr.GRPO 去掉
    adv  = (rewards - mean) / (std + eps)      # \hat A_i = (r_i - \bar r) / \sigma_r
    return adv

# demo: 同一数学题采 8 条, 3 条答对=1, 5 条答错=0
r = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=float)
print("rewards   :", r)
print("advantage :", np.round(grpo_advantage(r), 3))
# 全对/全错组: std=0 -> advantage 全 0 (零梯度), 需 DAPO Dynamic Sampling 剔除

print("\n全对组 (std=0):")
print("advantage :", grpo_advantage(np.array([1,1,1,1])))   # -> [0,0,0,0] 零梯度
print("\n全错组 (std=0):")
print("advantage :", grpo_advantage(np.array([0,0,0,0])))   # -> [0,0,0,0] 零梯度
```

输出（示意）：
```
rewards   : [1. 0. 0. 1. 0. 0. 0. 1.]
advantage : [ 0.956 -0.691 -0.691  0.956 -0.691 -0.691 -0.691  0.956]
全对组 (std=0):
advantage : [0. 0. 0. 0.]
全错组 (std=0):
advantage : [0. 0. 0. 0.]
```

> [!warning] 误区：全对/全错组零梯度
> 全对（acc=1）或全错（acc=0）时 $\sigma_r=0$，$\hat A$ 全 0 → **整组无梯度信号**。训练越往后模型越强、全对组越多，有效样本持续掉。这是 [[DAPO]] Dynamic Sampling（剔除全对/全错组、不够重采）和 [[DrGRPO]]（去 std 归一化）要修的病。

### 5.2 GRPO loss（带 clip + token-level mean，概念版）

```python
import torch
import torch.nn.functional as F

def grpo_loss(logprob_new,   # (N, G, T) 训练权重下的逐 token logprob
              logprob_old,    # (N, G, T) rollout 时的逐 token logprob
              ref_logprob,    # (N, G, T) reference 权重下逐 token logprob (KL 用)
              adv,            # (N, G)   组内相对 advantage (sequence-level, 广播到 token)
              mask,           # (N, G, T) response token mask (1=生成 token, 0=pad)
              eps=0.2, beta=0.001):
    """
    logprob_new/old/ref: 已 gather 好 target token 的 logprob
    adv: (N,G) -> 广播成 (N,G,T)
    """
    rho = (logprob_new - logprob_old).exp()                      # \pi_theta / \pi_old
    adv_t = adv.unsqueeze(-1)                                     # (N,G,1) 广播
    surr1 = rho * adv_t
    surr2 = torch.clamp(rho, 1 - eps, 1 + eps) * adv_t
    pg = -torch.minimum(surr1, surr2)                            # 负号: 要最大化 J -> 最小化 -J

    # token-level mean 归一化 (被 Dr.GRPO 批判): 先 token 内 mean, 再 batch mean
    token_mean = (pg * mask).sum(-1) / mask.sum(-1).clamp(min=1) # (N,G) 每条 response 内 token 平均
    loss_pg = token_mean.mean()                                  # batch 平均

    # KL to reference (逐 token, 无偏 k3 估计此处用简化版)
    kl = (logprob_new - ref_logprob)                            # 近似 (ref - new) 的 token KL
    loss_kl = (kl * mask).sum(-1).sum(-1) / mask.sum().clamp(min=1)
    return loss_pg + beta * loss_kl.sum()
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[REINFORCE]]（GRPO = REINFORCE + group baseline + PPO clip）、[[PPO clipped objective]]（ratio clip 复用）、[[advantage function]] / [[baseline]]（组内均值当 baseline）、[[variance reduction]]（baseline 降组间方差）、[[importance sampling与off-policy correction]]（$\rho=\pi_\theta/\pi_{old}$ 多 epoch 复用）、[[MDP]] / [[contextual bandit]]（outcome reward 退化为 bandit）、[[value network]] / [[GAE]]（GRPO 是 $\lambda=1$ 的 MC 版，$V$ 换成组均值）、[[KL penalty与KL control]]、[[rollout train reference logprob一致性]]（$\rho$ 精度依赖训推 logprob 同分布）。
- **下游（应用）**: [[RLHF (PPO)]]（GRPO 是 PPO 在 LLM 的精简变体，数学场景崛起）、[[DAPO]]（GRPO + 4 trick 治 long-CoT 的熵坍塌/梯度消失/长度漂移/奖励噪声）、[[DrGRPO]]（去 std + 去长度偏差的无偏修正）、[[RLOO]]（leave-one-out 精确无偏版）、DeepSeekMath/V3/R1 的 RL 后训练。
- **对比 / 易混**:
  - **GRPO vs [[PPO optimization]]**：见 §3.4 表。PPO 训 critic 用 GAE，GRPO 去 critic 用组内均值。省一半显存，代价是 reward 方差大时不如 critic 稳。
  - **GRPO vs [[RLOO]]**：GRPO baseline 含自身（$\bar r$），偏差因子 $(1-1/G)$；RLOO baseline 不含自身（leave-one-out），精确无偏。渐近等价，$G$ 大时几乎一致。详见 [[RLOO]]。
  - **GRPO vs [[DrGRPO]]**：GRPO advantage 除 $\sigma_r$ + loss 除 $|o_i|$（token mean），引入难度偏差与长度偏差；Dr.GRPO 两者都去掉。详见 [[DrGRPO]]。
  - **GRPO vs [[DPO]]**：DPO 绕过 RL 直接用偏好对学（offline，无 rollout）；GRPO 是 online RL 需大量 rollout。DPO 快但不学新行为，GRPO 能涌现新能力（R1 长思维链）。
  - **GRPO vs [[REINFORCE]] / REINFORCE++**：纯 REINFORCE 无 clip 无 KL、on-policy 单步；GRPO 加 clip + KL + group baseline，可多 epoch。REINFORCE++ 是 RLOO + clip 的工业版。


## 7. 常见误区与易错点

> [!warning] 误区 1：GRPO 完全无偏
> 严格说有 $(1-1/G)$ 的尺度偏差（§4.3）。渐近无偏，工程上当无偏用没问题，但理论上 [[RLOO]] 才精确无偏。

> [!warning] 误区 2：GRPO 不需要 reference policy
> 看 $\beta$。DeepSeekMath 原 GRPO 有 KL to ref（$\beta>0$）。R1-Zero 场景因 rule-based reward 无分布偏移顾虑常设 $\beta=0$（免 ref），但这是配置，不是算法本质。仍需训推 logprob 一致（[[rollout train reference logprob一致性]]）算 $\rho$。

> [!warning] 误区 3：GRPO 的 advantage 是 token 级
> outcome reward 是 sequence-level，$\hat A_{i,t}=\hat A_i$ 对所有 token 相同（§4.1 note）。credit assignment 粗（全序列共享信号）。要 token 级不同需 process reward（PRM）。

> [!warning] 误区 4：GRPO 一定比 PPO 省显存
> 只在 outcome-reward + 多采样可比的场景成立。若 reward 靠 RM 且噪声大、prompt 间无可比性，group-mean baseline 方差大、训练抖，反而不如 learned critic；此时要么上 critic，要么换 RLOO/REINFORCE++ 加 batch-level baseline。

> [!warning] 误区 5：全对/全错组也有梯度
> $\sigma_r=0$ 时 $\hat A=0$，整组零梯度（§5.1）。训练后期 acc 高的 prompt 越多，有效样本持续掉，需 [[DAPO]] Dynamic Sampling 剔除。

> [!warning] 误区 6：std 归一化只是无害的缩放
> 不是。除 $\sigma_r$ 让 advantage 与组内 reward 离散度耦合：std 小的组（全对/全错边缘）advantage 被放大，引入"难度偏差"（简单/难问题权重虚高）。[[DrGRPO]] 据此去掉 std。见 [[DrGRPO]] §3。

> [!warning] 误区 7：GRPO = PPO 去 critic
> 不只去 critic。还把 advantage 从 GAE 换成 group-relative MC（$\lambda=1$），并保持 PPO clip。是 PPO 的**精简变体**，保留了 clip 与 KL。


## 8. 延伸细节

### 8.1 group size $G$ 的影响

- $G$↑：$\bar r$ 估 $V(q)$ 更准、组内方差 $\propto1/G$ 更低、$(1-1/G)$ 偏差更小 → 更稳。但 rollout 量 $\propto G$，墙钟与算力线性涨。
- $G$↓（如 4）：advantage 噪声大、偏差 $1-1/G=0.75$ 不小 → 易抖。
- DeepSeekMath $G=64$，verl/DAPO recipe 常用 $G=16$，R1 报告更大。需在 rollout 成本与稳定性间权衡。

### 8.2 DeepSeekMath 的无偏 KL estimator

标准 KL $\mathbb{D}_{KL}[\pi_\theta\|\pi_{ref}]=\mathbb{E}_{o\sim\pi_\theta}[\log\frac{\pi_\theta(o)}{\pi_{ref}(o)}]$ 需从 $\pi_\theta$ 采样（on-policy），但 rollout 用 $\pi_{old}$。Schulman 给的 k3 近似 $k\sapprox\exp(\log\pi_{ref}-\log\pi_\theta)-( \log\pi_{ref}-\log\pi_\theta)-1$（负的 lower bound），是**有偏的 lower bound**。DeepSeekMath 改用其**无偏版** $k_3=(\frac{\pi_{ref}}{\pi_\theta}-1)-\log\frac{\pi_{ref}}{\pi_\theta}$，逐 token 算后取均值作为 KL penalty。这是 GRPO 在工程上的细节贡献（与 [[KL penalty与KL control]] 相关）。待核实：具体公式形式以 DeepSeekMath 论文附录为准。

### 8.3 GRPO 的 off-policy 程度

GRPO 用 $\rho=\pi_\theta/\pi_{old}$ clip，允许在同一个 rollout batch 上跑多 epoch（PPO 风格），属轻度 off-policy。但比 DAPO 等更激进 off-policy 方法温和。clip $\varepsilon$ 控制 stale 程度。MoE 模型训推路由不一致会让 $\rho$ 失真，需 [[R3 rollout routing replay]] 修。

### 8.4 与 long-CoT / R1 的关系

R1-Zero 用 GRPO 在 base model 上直接 RL（无 SFT），reward 纯 rule-based（答对/格式）。涌现"Aha moment"与长思维链。但社区照搬 GRPO 复现 R1 时集体翻车（熵坍塌/梯度消失/长度漂移/奖励噪声），催生 [[DAPO]]（4 trick 包）、[[DrGRPO]]（去偏差）。GRPO 是 R1 范式的算法基线，但"裸跑"易病，需 trick 加固。

### 8.5 内容来源

DeepSeekMath GRPO：arXiv:2402.03300（Shao et al. 2024）。DeepSeek-V3 / R1 / R1-Zero 技术报告。verl / DAPO recipe。Schulman KL estimator blog。$(1-1/G)$ 偏差推导见 [[RLOO]] arXiv:2402.14740 附录与 baseline 理论。与 [[DrGRPO]] / [[RLOO]] / [[DAPO]] 的差异见各自笔记。

---
相关: [[GRPO体系]] | [[REINFORCE]] | [[PPO clipped objective]] | [[advantage function]] | [[baseline]] | [[variance reduction]] | [[value network]] | [[GAE]] | [[RLHF (PPO)]] | [[DAPO]] | [[DrGRPO]] | [[RLOO]] | [[token-level与sequence-level objective]] | [[importance sampling与off-policy correction]] | [[KL penalty与KL control]] | [[rollout train reference logprob一致性]] | [[Actor-Critic]]
