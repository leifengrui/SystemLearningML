# importance sampling与off-policy correction

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: IS / 重要性采样 / off-policy correction / 离策略修正 / importance weighting / π_θ/π_old ratio / importance ratio
> **难度**: 高（需懂 [[REINFORCE]]、[[PPO clipped objective]]、[[advantage function]]、[[rollout train reference logprob一致性]]、[[RL权重同步]]、[[stale policy problem]]、[[baseline]]、[[variance reduction]]）

> **所属小节**: §83


## 1. 一句话定义

**importance sampling（IS，重要性采样）与 off-policy correction（离策略修正）** 是 RL 中用**行为策略 $\pi_{old}$（behavior policy，采数据的策略）采的样本**去估计**目标策略 $\pi_\theta$（target policy，要优化的策略）的期望**的数学技术——核心恒等式 $\mathbb{E}_{x\sim\pi_\theta}[f(x)]=\mathbb{E}_{x\sim\pi_{old}}[\frac{\pi_\theta(x)}{\pi_{old}(x)}f(x)]$，用 ratio（重要性权重）$\rho=\pi_\theta/\pi_{old}$ 把旧策略采的样本"重加权"给新策略，从而**复用旧策略的数据训新策略**（不必每更新一步就重新采——这在 LLM-RL 因 rollout 极贵而救命）；在 LLM-RL 里 ratio 取 log 形式 $\rho=\exp(\log\pi_\theta-\log\pi_{old})$（$\log\pi$ 来自 forward 输出，避免连乘下溢）；PPO 的 clipped objective $\min(\rho\hat A,\text{clip}(\rho,1-\varepsilon,1+\varepsilon)\hat A)$ 截断 ratio 防 IS 的高方差爆炸；on-policy（rollout 用最新 $\theta$ 采、$\rho\approx1$）vs off-policy（rollout 用旧 $\theta$、$\rho$ 偏离 1，staleness 越大偏离越大）的频谱由 [[RL权重同步]] 频率控制；V-trace（IMPALA）进一步截断 ratio 防异步 staleness 爆炸。本条与 [[rollout train reference logprob一致性]] 紧耦合——IS 的 ratio $\rho=\exp(\log\pi_\theta-\log\pi_{old})$ 直接依赖两端的 $\log\pi$，若 TIM（训推不一致）让同 $\theta$ 下 logp 算错，ratio 在 $\theta_{new}=\theta_{old}$ 时本应 $=1$ 却偏离 $1$，IS 修正被污染 → 梯度有偏 → KL 爆/policy 崩。是 [[REINFORCE]]→[[PPO clipped objective]]→[[GRPO体系|GRPO]] 算法链的数学基石，也是 [[17-RL训推一体框架]] 训推分离/异步训练能成立的理论根（没有 IS 就不能复用旧数据，rollout 开销扛不住）。

> [!note] 三句话定位
> - **是什么**：用 $\pi_{old}$ 采的样本估 $\pi_\theta$ 的期望，靠 ratio $\rho=\pi_\theta/\pi_{old}$ 重加权；LLM-RL 取 log 形式 $\rho=\exp(\log\pi_\theta-\log\pi_{old})$ 防下溢；PPO clip 截断防 IS 高方差爆。
> - **为什么**：LLM rollout 极贵，不能每更新一步重采（on-policy 不可行）；IS 让旧策略采的数据可复用训新策略，是 RL 训推分离/异步/多 epoch 复用的数学根。
> - **与 [[rollout train reference logprob一致性]] 关系**：ratio 直接依赖两端 $\log\pi$，TIM 让同 $\theta$ 下 logp 算错→ratio 在 $\theta$ 未动时偏离 1→IS 修正失真→梯度有偏。logp 一致性是 IS 正确的前提。


## 2. 为什么需要它（动机与背景）

### 2.1 on-policy 的不可行：rollout 太贵

纯 on-policy 的策略梯度（[[REINFORCE]]）：每更新一步 $\theta$，立刻用**新 $\pi_\theta$** 重新采一批样本，再算梯度。数学上无偏、方差低，但 LLM-RL 里**致命**——生成一条 response 要几百~几千 token forward，PPO/GRPO 每 step 采成千上万条，rollout 是墙钟时间大头（常占 50~80%）。若每更新一步就重采，训练几乎全在采，算力扛不住。

**解法**：用**旧策略 $\pi_{old}$ 采一次**，在它上面跑多步更新或多 epoch（off-policy 复用），靠 importance sampling 的 ratio 修正"样本是 $\pi_{old}$ 采的但梯度对 $\pi_\theta$ 算"的偏差。这样一次 rollout 的数据可训多步，rollout 开销摊薄。

### 2.2 期望恒等式：IS 的数学根

核心恒等式（推导见 §4.1）：对任意 $f$，

$$
\mathbb{E}_{x\sim\pi_\theta}[f(x)] = \mathbb{E}_{x\sim\pi_{old}}\!\left[\frac{\pi_\theta(x)}{\pi_{old}(x)}f(x)\right]
$$

即"用 $\pi_\theta$ 采的期望" = "用 $\pi_{old}$ 采的样本乘 ratio $\rho=\pi_\theta/\pi_{old}$ 再求期望"。ratio 把旧策略采的样本重加权成新策略的等价样本。**这是 off-policy 数据复用的数学合法性来源**——只要 ratio 算对，旧策略数据可无偏估新策略期望。

### 2.3 高方差问题：ratio 爆炸与 clip 兜底

IS 无偏但**高方差**：ratio $\rho=\pi_\theta/\pi_{old}$ 当两策略差距大时，$\rho$ 可能极大（$\pi_\theta$ 远大于 $\pi_{old}$ 处）或极小，方差爆炸 → 梯度噪声大、训练不稳。极端情况 $\pi_{old}(x)\to0$ 而 $\pi_\theta(x)>0$ 时 $\rho\to\infty$。

**PPO 的解法**（[[PPO clipped objective]]）：截断 ratio 在 $[1-\varepsilon,1+\varepsilon]$（$\varepsilon$ 典型 0.2），$\min(\rho\hat A,\text{clip}(\rho,1\pm\varepsilon)\hat A)$——牺牲少量有偏换方差可控。这是 PPO 相对朴素 IS 的关键改进。V-trace（IMPALA）在异步场景进一步截断 $\bar\rho=\min(\rho,\bar c)$ 防 staleness 爆炸。

### 2.4 LLM-RL 的 log 形式：防下溢

LLM 生成一条 response $o=(o_1,\dots,o_T)$，其概率 $\pi(o|q)=\prod_t\pi(o_t|q,o_{<t})$ 是 $T$ 个条件概率连乘。长序列连乘易下溢（浮点 underflow）。取 log：

$$
\log\pi(o|q) = \sum_{t=1}^T \log\pi(o_t|q,o_{<t})
$$

ratio 也取 log 形式：

$$
\rho = \frac{\pi_\theta(o)}{\pi_{old}(o)} = \exp\!\big(\log\pi_\theta(o) - \log\pi_{old}(o)\big) = \exp\!\Big(\sum_t(\log\pi_\theta(o_t) - \log\pi_{old}(o_t))\Big)
$$

但 PPO/GRPO 实际是 **token 级 ratio**（每 token 一个 $\rho_t$，非整条一个），因为 advantage 是 token 级（或 sequence-level 广播）。详见 §4.3、[[token-level与sequence-level objective]]。$\log\pi$ 直接来自模型 forward 的 log_softmax 输出（gather target token 的 logp），无需连乘。

### 2.5 与 staleness/sync 的频谱

ratio 偏离 1 的程度由 rollout 用的 $\theta_{old}$ 与 trainer 的 $\theta_{new}$ 的差距决定，即 staleness：
- **on-policy**（每步重采，$\theta_{old}=\theta_{new}$）：$\rho\approx1$，无 IS 修正需要（但 LLM 不可行）。
- **轻度 off-policy**（colocate 每 step sync，$\theta_{old}$ 落后一步）：$\rho$ 小偏离 1，clip 基本不触发。
- **重度 off-policy**（disaggregate 异步，$\theta_{old}$ 落后 $k$ 步）：$\rho$ 指数偏离 1（$\exp(k\cdot g)$），clip 频繁触发，需更频繁 sync 降 $k$。

[[RL权重同步]] 频率控制这个频谱——sync 越频繁越接近 on-policy（$\rho\approx1$），越不频繁越 off-policy（$\rho$ 偏离大）。详见 §4.4、[[stale policy problem]]。


## 3. 核心概念详解

### 3.1 行为策略 vs 目标策略

| 概念 | 符号 | 角色 | LLM-RL 中 |
|---|---|---|---|
| **行为策略（behavior policy）** | $\pi_{old}$ / $b$ | 采数据的策略 | rollout 用 $\theta_{old}$ 跑 vLLM 采 response |
| **目标策略（target policy）** | $\pi_\theta$ | 要优化的策略 | trainer 用 $\theta_{new}$ 算梯度更新 |

两者可以相同（on-policy，$\theta_{old}=\theta$）或不同（off-policy，$\theta_{old}\ne\theta$）。IS 处理后者——用 $\pi_{old}$ 的数据估 $\pi_\theta$ 的期望。

### 3.2 ratio（重要性权重）

$$
\rho(x) = \frac{\pi_\theta(x)}{\pi_{old}(x)}
$$

- $\rho>1$：$\pi_\theta$ 比 $\pi_{old}$ 更可能采 $x$ → 该样本在新策略下被"高估"，加权放大。
- $\rho<1$：$\pi_\theta$ 比 $\pi_{old}$ 更不可能采 $x$ → 该样本在新策略下被"低估"，加权缩小。
- $\rho=1$：两策略对该样本同等 → 无需修正（on-policy 极限）。

IS 估计量：$\hat\mu_{IS} = \frac1N\sum_i \rho(x_i) f(x_i)$，$x_i\sim\pi_{old}$。无偏（$\mathbb{E}[\hat\mu_{IS}]=\mathbb{E}_{\pi_\theta}[f]$），但方差 $\mathrm{Var}[\rho f]$ 可能大。

### 3.3 LLM-RL 的 token 级 ratio

LLM-RL 的 ratio 是**逐 token** 的：

$$
\rho_t = \frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{old}(o_t\mid q,o_{<t})} = \exp\!\big(\log\pi_\theta(o_t) - \log\pi_{old}(o_t)\big)
$$

- $\log\pi_\theta(o_t)$：trainer 用 $\theta_{new}$ forward 算的 target token logp（`new_log_prob`）。
- $\log\pi_{old}(o_t)$：rollout 用 $\theta_{old}$ 采时记的 target token logp（`old_log_prob`）。
- $\exp$ 防 token 概率连乘下溢（每 token 独立算 ratio，不连乘）。

整条 response 的 ratio（若需 sequence-level）= $\prod_t \rho_t$，但 PPO/GRPO 用 token 级（每 token 一个 $\rho_t$ 配 token 级 advantage $\hat A_t$）。

### 3.4 PPO clip：防 IS 方差爆

PPO clipped objective（[[PPO clipped objective]]）：

$$
L^{CLIP}(\theta) = \mathbb{E}_t\big[\min(\rho_t \hat A_t,\ \text{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\hat A_t)\big]
$$

- $\varepsilon$：clip ratio（典型 0.2，DAPO 解耦 $\varepsilon_{low}/\varepsilon_{high}$）。
- $\min$：取两项较小者——若 $\rho_t\hat A_t$ 与 clip 项同向取 clip（截断过大 ratio），若反向取原值（保梯度符号）。
- 效果：截断过大 ratio 防 IS 高方差爆，牺牲少量有偏换稳定。

> [!warning] 误区：PPO clip 让 IS 无偏
> 不。clip 引入有偏（截断后的期望 $\ne$ 真 IS 期望）。但方差大幅降，工程上 trade-off 划算。纯 IS 无偏但方差爆不可用。PPO 的"无偏"是相对 TRPO 的 KL 约束近似，非 IS 层。

### 3.5 V-trace：异步 staleness 的截断

IMPALA 的 V-trace 在异步场景（rollout 用 $\theta_{old-k}$ 落后 $k$ 步）进一步截断：

$$
\bar\rho_t = \min(\rho_t,\ \bar\rho),\quad \bar c_t = \min(\rho_t,\ \bar c)
$$

- $\bar\rho$：retue 截断阈值（控制目标策略与行为策略的偏离上界）。
- $\bar c$：value 截断阈值（控 bootstrap）。
- 效果：异步 staleness 下 $\rho$ 可能极大，V-trace 截断防爆，保异步训练稳定。

LLM-RL 的 disaggregate 异步常用类似截断思路（PPO clip $\varepsilon$ 兜底 + 频繁 sync 降 $k$）。详见 [[asynchronous training]]。

### 3.6 on-policy vs off-policy 频谱

| 维度 | on-policy | off-policy |
|---|---|---|
| 行为=目标策略 | $\pi_{old}=\pi_\theta$ | $\pi_{old}\ne\pi_\theta$ |
| ratio | $\rho\approx1$ | $\rho$ 偏离 1 |
| 数据复用 | 单步用完弃（每步重采） | 可多 epoch/多步复用 |
| 方差 | 低 | 高（需 clip/V-trace） |
| LLM-RL 可行性 | 不可行（rollout 太贵） | 主流（IS + clip） |
| staleness $k$ | 0 | $>0$（sync 频率控制） |
| 代表 | 纯 REINFORCE | PPO/GRPO（轻度 off-policy + clip） |

LLM-RL 全是 off-policy（程度不同）：colocate 每 step sync 是轻度（$k=1$，$\rho$ 小偏离）；disaggregate 异步是重度（$k>1$，$\rho$ 大偏离）。[[RL权重同步]] 频率决定落在频谱哪端。

### 3.7 与 logp 一致性的耦合

IS 的 ratio $\rho=\exp(\log\pi_\theta-\log\pi_{old})$ 直接依赖两端 $\log\pi$。若 TIM（[[rollout train reference logprob一致性]]、[[训推不一致]]）让同 $\theta$ 下 rollout 的 vLLM 与 trainer 算的 logp 不一致，则 $\theta_{new}=\theta_{old}$ 时 $\rho=\exp(\epsilon)\ne1$——IS 修正被污染，trainer 误以为有策略偏差需修正，实际是数值差。后果：ratio 失真 → clip 误触发 → 梯度有偏 → KL 爆/崩。**logp 一致性是 IS 正确的前提**，治 TIM 是治 IS 的根。详见 [[rollout train reference logprob一致性]]。

> [!note] 补充：IS 修正的"正确偏离"与 TIM 的"错误偏离"
> ratio 偏离 1 有两种：①**正确偏离**（$\theta_{new}\ne\theta_{old}$，staleness 本意，IS 该修）；②**错误偏离**（$\theta_{new}=\theta_{old}$ 但 TIM 让 logp 不同，不该修却被修）。前者是 IS 的功能，后者是污染。sync 频率控前者程度，TIM 治理消后者。两者正交。详见 [[RL权重同步]] §4.5。


## 4. 数学原理 / 公式

### 4.1 推导：IS 期望恒等式

**目标**：用 $\pi_{old}$ 采的样本估 $\mathbb{E}_{x\sim\pi_\theta}[f(x)]$。

从定义出发：

$$
\mathbb{E}_{x\sim\pi_\theta}[f(x)] = \int \pi_\theta(x) f(x)\, dx
$$

凑 $\pi_{old}$（假设 $\pi_{old}(x)>0$ 当 $\pi_\theta(x)>0$，即 $\pi_{old}$ 覆盖 $\pi_\theta$）：

$$
= \int \pi_{old}(x)\,\frac{\pi_\theta(x)}{\pi_{old}(x)}\, f(x)\, dx = \int \pi_{old}(x)\,\rho(x)\, f(x)\, dx
$$

这正是对 $\pi_{old}$ 的期望：

$$
\boxed{\;\mathbb{E}_{x\sim\pi_\theta}[f(x)] = \mathbb{E}_{x\sim\pi_{old}}\!\left[\frac{\pi_\theta(x)}{\pi_{old}(x)}f(x)\right] = \mathbb{E}_{x\sim\pi_{old}}[\rho(x)\,f(x)]\;}
$$

**关键假设**：$\pi_{old}(x)>0$ 当 $\pi_\theta(x)>0$（覆盖性，support）。若 $\pi_{old}$ 采不到的 $x$，$\pi_\theta$ 也采不到（零概率），IS 失效。LLM-RL 中 $\pi_{old}$ 与 $\pi_\theta$ 同模型架构、权重相近，覆盖性基本满足（除非 $\pi_{old}$ 把某 token 采样概率压到 0，如 temperature=0 贪心，则 IS 退化）。

**无偏性**：$\mathbb{E}_{x\sim\pi_{old}}[\rho f] = \mathbb{E}_{\pi_\theta}[f]$，IS 估计量无偏。

### 4.2 推导：IS 估计量的方差与 clip 的动机

IS 估计量 $\hat\mu = \frac1N\sum_i \rho_i f_i$ 的方差：

$$
\mathrm{Var}[\hat\mu] = \frac{1}{N}\mathrm{Var}_{\pi_{old}}[\rho f]
$$

展开 $\mathrm{Var}[\rho f] = \mathbb{E}_{\pi_{old}}[\rho^2 f^2] - (\mathbb{E}_{\pi_{old}}[\rho f])^2 = \mathbb{E}_{\pi_\theta}[\rho f^2] - (\mathbb{E}_{\pi_\theta}[f])^2$。

当 $\pi_\theta$ 与 $\pi_{old}$ 差距大时，$\rho$ 在某些 $x$ 处极大（$\pi_\theta\gg\pi_{old}$），$\mathbb{E}[\rho^2 f^2]$ 被这些点主导，方差爆。极端 $\pi_{old}(x)\to0$ 而 $\pi_\theta(x)>0$ 时 $\rho\to\infty$。

**PPO clip 的动机**：截断 $\rho$ 在 $[1-\varepsilon,1+\varepsilon]$，把极大 $\rho$ 压回上界，直接限制 $\rho^2$ 的上界 → 限制方差。代价：截断后 $\mathbb{E}[\text{clip}(\rho)\hat A]\ne\mathbb{E}[\rho\hat A]=\nabla J$，引入有偏。但方差从 $O(\rho_{\max}^2)$ 降到 $O((1+\varepsilon)^2)$，trade-off 划算。$\varepsilon=0.2$ 是经验最优（偏差小、方差可控）。详见 [[PPO clipped objective]]、[[variance reduction]]。

### 4.3 推导：ratio 的 log 形式（防下溢）

LLM 的 response $o=(o_1,\dots,o_T)$，sequence 概率：

$$
\pi(o|q) = \prod_{t=1}^T \pi(o_t|q,o_{<t})
$$

$T$ 个 $<1$ 的概率连乘，长序列（$T$=几百~几千）下溢成 0（浮点 underflow）。取 log 化连乘为连加：

$$
\log\pi(o|q) = \sum_{t=1}^T \log\pi(o_t|q,o_{<t})
$$

sequence-level ratio：

$$
\rho_{seq} = \frac{\pi_\theta(o)}{\pi_{old}(o)} = \exp\!\big(\log\pi_\theta(o) - \log\pi_{old}(o)\big) = \exp\!\Big(\sum_t \big(\log\pi_\theta(o_t) - \log\pi_{old}(o_t)\big)\Big)
$$

但 PPO/GRPO 用 **token 级 ratio**（每 token 一个 $\rho_t$，配 token 级 advantage）：

$$
\rho_t = \frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{old}(o_t|q,o_{<t})} = \exp\!\big(\log\pi_\theta(o_t) - \log\pi_{old}(o_t)\big)
$$

- $\log\pi_\theta(o_t)$、$\log\pi_{old}(o_t)$ 直接来自 forward 的 log_softmax gather（无需连乘）。
- $\exp$ 把 log 差转回 ratio，单 token 不会下溢（差是 $O(1)$）。
- token 级 ratio 配 token 级 advantage $\hat A_t$（或 sequence-level 广播 $\hat A_{i,t}=\hat A_i$，GRPO）。

**关键**：用 $\log\pi$ 而非 $\pi$ 是工程必需（防下溢），用 $\exp(\Delta\log\pi)$ 而非 $\pi_\theta/\pi_{old}$ 直接除是数值稳定（$\log\pi$ 是 forward 直接输出，除法可能 $0/0$）。

### 4.4 推导：staleness 如何放大 ratio 偏离（sync 频率的数学根）

设 rollout 用 $\theta_{old-k}$（落后 $k$ 步），trainer 当前 $\theta_{new}$。一阶 Taylor：

$$
\log\pi_{\theta_{new}} - \log\pi_{\theta_{old-k}} \approx k\cdot\nabla_\theta\log\pi_{\theta_{old}}\cdot\Delta\theta = k\cdot g_t
$$

故：

$$
\rho_t = \exp(k\cdot g_t)
$$

- $k=1$（每 step sync）：$\rho_t=\exp(g_t)$，$g_t$ 小（单步更新），$\rho$ 小偏离 1，clip 基本不触发。
- $k$ 大（异步 staleness）：$\rho_t=\exp(k g_t)$，偏离指数放大，大量 token 超 clip 范围被截断。

**sync 频率 $S$（=两次 sync 间隔步数）= $k_{\max}$**（staleness 上界）。$S$ 小则 $\rho$ 偏离小（稳但 sync 重）；$S$ 大则偏离大（快但 stale）。PPO clip $\varepsilon$ 是兜底：当 $\rho$ 超出 $[1-\varepsilon,1+\varepsilon]$ 截断。**sync 频率 + clip $\varepsilon$ 共同控制 off-policy 程度**。colocate $S=1$ 最稳；disaggregate 权衡 $S\in[1,8]$。详见 [[RL权重同步]] §4.2、[[stale policy problem]]。

### 4.5 V-trace 截断的数学

IMPALA V-trace 的截断 ratio（异步 staleness 兜底）：

$$
\bar\rho_t = \min(\rho_t,\ \bar\rho),\quad \bar c_t = \min(\rho_t,\ \bar c)
$$

- $\bar\rho$：retue（target value）截断阈值，典型 $\bar\rho=1$（等价完全截断，目标策略=行为策略的 bootstrap）。
- $\bar c$：bootstrap 截断阈值，典型 $\bar c=1$。
- 效果：$\rho_t$ 被 $\bar\rho$ 截顶，防 $k$ 大时 $\exp(k g)$ 爆炸。

V-trace 的目标 value 用截断 ratio 加权 bootstrap：

$$
v_t = V(x_t) + \sum_{s\ge t}\gamma^{s-t}\Big(\prod_{i=t}^{s-1}\gamma\bar c_i\Big)\bar\rho_s\big(r_s + \gamma V(x_{s+1}) - V(x_s)\big)
$$

截断 $\bar\rho,\bar c$ 让 $v_t$ 的方差有界，异步训练稳定。LLM-RL disaggregate 异步借鉴此思路（PPO $\varepsilon$ + 频繁 sync）。详见 [[asynchronous training]]。

### 4.6 PPO 目标的全貌（连接 IS + clip + KL）

GRPO/PPO 的完整 token 级目标（来自 [[GRPO]] §4.1）：

$$
\mathcal{J}(\theta) = \mathbb{E}\Big[\frac{1}{|o|}\sum_t\big(\min(\rho_t\hat A_t,\ \text{clip}(\rho_t,1\pm\varepsilon)\hat A_t) - \beta\,D_{KL}^{(t)}[\pi_\theta\|\pi_{ref}]\big)\Big]
$$

- $\rho_t\hat A_t$：IS 重加权的 advantage（off-policy 修正）。
- $\text{clip}(\rho_t,1\pm\varepsilon)\hat A_t$：截断防 IS 方差爆。
- $\min$：取较小者（保守）。
- $\beta D_{KL}$：到 ref 的 KL 约束（防 policy 漂太远，[[KL penalty与KL control]]）。

IS 是第一项的数学根，clip 是 IS 的方差治理，KL 是 IS 之外的分布约束。三者共同构成 PPO/GRPO 的目标。


## 5. 代码示例（可选）

### 5.1 手写 importance sampling 加权估计（最小可跑）

```python
import numpy as np

def importance_sampling_estimate(samples, f, pi_old, pi_theta):
    """
    用 pi_old 采的 samples 估 E_{pi_theta}[f]
    samples: (N,) 从 pi_old 采的 x
    f      : 函数, f(x) -> 标量
    pi_old : 行为策略 pmf (或可计算 pi_old(x))
    pi_theta: 目标策略 pmf
    返回 IS 估计量 (无偏但可能高方差)
    """
    rho = np.array([pi_theta(x) / pi_old(x) for x in samples])  # ratio
    f_vals = np.array([f(x) for x in samples])
    is_est = np.mean(rho * f_vals)          # IS 估计量 E_{pi_old}[rho*f] = E_{pi_theta}[f]
    return is_est, rho

# demo: pi_old=N(0,1), pi_theta=N(1,1), f(x)=x (估 E_{pi_theta}[x]=1)
np.random.seed(0)
samples = np.random.randn(10000)            # 从 pi_old=N(0,1) 采
from scipy.stats import norm
pi_old = lambda x: norm.pdf(x, 0, 1)
pi_theta = lambda x: norm.pdf(x, 1, 1)
f = lambda x: x

est, rho = importance_sampling_estimate(samples, f, pi_old, pi_theta)
print(f"IS 估计: {est:.4f} (真值 E_pi_theta[x]=1.0)")
print(f"ratio 均值: {rho.mean():.4f} (应≈1, 因 E[rho]=1)")
print(f"ratio 方差: {rho.var():.4f} (高方差 -> 需 clip)")
# IS 估计接近 1, 但 ratio 方差大 -> 大 sample 才稳, 这就是 PPO clip 的动机
```

### 5.2 PPO ratio 计算与 clip（LLM-RL token 级）

```python
import torch

def ppo_ratio_and_clip(logprob_new, logprob_old, adv, eps=0.2):
    """
    logprob_new: (N, G, T) trainer 用 θ_new 算的逐 token logp
    logprob_old: (N, G, T) rollout 用 θ_old 记的逐 token logp
    adv        : (N, G)   advantage (sequence-level 广播到 token)
    eps        : PPO clip ratio
    返回 PPO clipped objective (要最大化, 训练取负最小化)
    """
    # ratio = exp(logp_new - logp_old), token 级 (防下溢的 log 形式)
    ratio = (logprob_new - logprob_old).exp()              # (N,G,T)

    adv_t = adv.unsqueeze(-1)                              # (N,G,1) 广播
    surr1 = ratio * adv_t
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv_t   # clip 防 IS 方差爆
    pg = torch.minimum(surr1, surr2)                       # PPO clipped objective

    # 监控指标
    clipfrac = ((ratio - 1).abs() > eps).float().mean()    # clip 触发率 (TIM/stale 仪表)
    return pg, ratio, clipfrac

# demo: θ_new=θ_old 时 ratio 应=1 (无 stale), TIM 让它偏离
logp_old = torch.randn(4, 8, 100) * 2 - 5       # 模拟 rollout logp
tim_noise = torch.randn(4, 8, 100) * 0.01       # TIM 噪声
logp_new = logp_old + tim_noise                  # θ_new=θ_old 但 TIM 让 logp 差
adv = torch.randn(4, 8)                          # 随机 advantage
pg, ratio, clipfrac = ppo_ratio_and_clip(logp_new, logp_old, adv, eps=0.2)
print(f"ratio 均值: {ratio.mean():.4f} (θ 未动应=1, TIM 让它偏离)")
print(f"clipfrac: {clipfrac:.4f} (TIM 让 clip 误触发, 应≈0)")
# -> TIM 让 θ 未动时 ratio≠1, clip 误触发, 这就是 [[rollout train reference logprob一致性]] 的后果
```

### 5.3 V-trace 截断（异步 staleness 兜底，概念）

```python
import torch

def vtrace_truncate_ratio(ratio, rho_bar=1.0, c_bar=1.0):
    """
    V-trace 截断 ratio 防异步 staleness 爆炸 (IMPALA)
    ratio   : (N,G,T) 原始 IS ratio (异步 k 大时可能极大)
    rho_bar : retue 截断阈值 (典型 1.0)
    c_bar   : bootstrap 截断阈值 (典型 1.0)
    返回截断后的 ratio_bar
    """
    rho_bar_t = torch.clamp(ratio, max=rho_bar)   # 截顶防爆炸
    c_bar_t = torch.clamp(ratio, max=c_bar)
    return rho_bar_t, c_bar_t

# demo: 异步 staleness k=8 让 ratio 剧烈偏离
g_t = 0.3  # 单步影响 (大)
k = 8
ratio = torch.tensor([np.exp(k * g_t)])   # exp(2.4)=11, 远超 clip 0.2
print(f"staleness k={k}: ratio={ratio.item():.2f} (剧烈偏离 1, PPO clip 全截断)")
rho_bar, c_bar = vtrace_truncate_ratio(ratio, rho_bar=1.0)
print(f"V-trace 截断后: rho_bar={rho_bar.item():.2f} (截顶 1.0 防爆炸)")
# -> 异步 staleness 让 ratio 爆, V-trace 截断 + 频繁 sync 降 k 是异步稳定的关键
```

### 5.4 verl/OpenRLHF 的 ratio 计算配置（概念）

```bash
# OpenRLHF: PPO ratio + clip (importance sampling 修正)
python -m openrlhf.cli.train.ppo_ray \
  --eps_clip 0.2 \          # ★ PPO clip ratio ε (IS 方差治理)
  --kl_coef 0.001 \         # KL to ref β (分布约束, 非 IS)
  --vllm_sync_backend nccl \ # weight sync 频率控制 staleness -> ratio 偏离程度
  --colocate                 # colocate 每 step sync -> k=1, ratio≈1 (轻度 off-policy)
# 去 --colocate 开 disaggregate 异步 -> k>1, ratio 偏离大, 需 clip 兜底 + 频繁 sync
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[REINFORCE]]（IS 是 REINFORCE off-policy 扩展的数学根）、[[advantage function]]（ratio 配 advantage 构成 PPO 目标）、[[baseline]]/[[variance reduction]]（IS 高方差需降，PPO clip 是手段）、[[PPO clipped objective]]（clip 是 IS 的方差治理）、[[RL权重同步]]（sync 频率控制 ratio 偏离程度，即 off-policy 程度）、[[stale policy problem]]（staleness 是 ratio 偏离的根）。
- **下游（应用）**: [[GRPO体系|GRPO]]（GRPO 的 ratio = IS + clip + KL）、[[PPO optimization]]/[[RLHF (PPO)]]（PPO 全靠 IS 修 off-policy）、[[17-RL训推一体框架]] 全章（训推分离/异步/多 epoch 复用都靠 IS）、[[asynchronous training]]（异步靠 IS + V-trace 截断 + 频繁 sync）、[[rollout train reference logprob一致性]]（logp 一致是 ratio 正确的前提，TIM 污染 IS）。
- **对比 / 易混**:
  - **IS（本条）vs [[REINFORCE]]**：REINFORCE 是 on-policy（每步重采，$\rho=1$ 无需 IS）；IS 是 off-policy 扩展（复用旧数据，$\rho$ 修正）。PPO/GRPO = REINFORCE + IS + clip + KL。
  - **IS 无偏 vs PPO clip 有偏**：纯 IS 无偏但方差爆；PPO clip 截断引入有偏换方差可控。工程用后者。
  - **staleness 偏离（IS 本意）vs TIM 偏离（污染）**：前者是 $\theta_{new}\ne\theta_{old}$ 的正确偏离（IS 该修）；后者是 $\theta_{new}=\theta_{old}$ 但 logp 算错的错误偏离（不该修却被修）。sync 频率控前者，TIM 治理消后者。详见 [[RL权重同步]] §4.5、[[rollout train reference logprob一致性]] §3.7。
  - **token-level ratio（本条）vs sequence-level ratio**：前者每 token 一个 $\rho_t$ 配 token advantage；后者整条一个 $\rho_{seq}=\prod\rho_t$。PPO/GRPO 用 token 级。详见 [[token-level与sequence-level objective]]。


## 7. 常见误区与易错点

> [!warning] 误区 1：IS 估计量无偏所以总能用
> 无偏但高方差。当 $\pi_\theta$ 与 $\pi_{old}$ 差距大时 ratio $\rho$ 极大，方差爆炸，估计噪声大到不可用。PPO clip 截断牺牲少量有偏换方差可控，是工程必需。纯 IS 在 LLM-RL 异步 staleness 下不可行。

> [!warning] 误区 2：ratio 偏离 1 就是 staleness
> 不一定。偏离有两种：①正确偏离（$\theta_{new}\ne\theta_{old}$，staleness，IS 本意）；②错误偏离（$\theta_{new}=\theta_{old}$ 但 TIM 让 logp 不同，污染）。前者靠 sync 降，后者靠 TIM 治理。两者正交。详见 §3.7、[[rollout train reference logprob一致性]]。

> [!warning] 误区 3：PPO clip 让 IS 无偏
> 不。clip 截断引入有偏（$\mathbb{E}[\text{clip}(\rho)\hat A]\ne\mathbb{E}[\rho\hat A]$）。但方差大幅降，trade-off 划算。PPO 的"无偏"是相对 TRPO 的 KL 约束近似，非 IS 层。

> [!warning] 误区 4：用 $\pi_\theta/\pi_{old}$ 直接除比 $\exp(\Delta\log\pi)$ 好
> 不好。直接除面临 $0/0$（两概率都下溢成 0）的数值问题。$\exp(\Delta\log\pi)$ 用 forward 的 log_softmax 输出（稳定，不连乘），是工程标准做法。

> [!warning] 误区 5：sequence-level ratio 与 token-level 等价
> 不等价。sequence-level $\rho_{seq}=\prod_t\rho_t$ 是整条一个；token-level 每 token 一个 $\rho_t$ 配 token advantage。PPO/GRPO 用 token 级（advantage 逐 token）。sequence-level 会把方差累乘更爆。详见 [[token-level与sequence-level objective]]。

> [!warning] 误区 6：on-policy 不需要 IS 所以 LLM-RL 可去掉 IS
> 不可。LLM-RL 全是 off-policy（rollout 太贵不能每步重采）。即使 colocate 每 step sync，rollout 采那一刻是 $\theta_{old}$、trainer 训完变 $\theta_{new}$，仍 off-policy（差一步），需 IS 修。纯 on-policy 在 LLM-RL 不可行。

> [!warning] 误区 7：覆盖性假设自动满足
> 不自动。若 $\pi_{old}$ 用 temperature=0 贪心采（某些 token 概率 0），而 $\pi_\theta$ 对这些 token 有概率，IS 失效（$\rho=\infty$）。需 $\pi_{old}$ 用 temperature>0 采样保覆盖。LLM-RL 默认 temperature>0，基本满足，但极端配置需注意。

> [!tip] 实践：控制 off-policy 程度的三手段
> 1. **sync 频率**（[[RL权重同步]]）：colocate 每 step sync（$k=1$，$\rho\approx1$，轻度 off-policy）；disaggregate 权衡 $S\in[1,8]$。
> 2. **PPO clip $\varepsilon$**：截断 ratio 防方差爆，典型 0.2。$\varepsilon$ 小则保守（稳但学慢），大则激进（快但易爆）。
> 3. **V-trace 截断**（异步）：$\bar\rho$ 截顶防 staleness 爆，disaggregate 异步用。
> 三者配合：sync 控 staleness 程度 → clip/V-trace 兜底方差 → 共同保 off-policy 训练稳定。


## 8. 延伸细节

### 8.1 IS 在经典 RL 与 LLM-RL 的差异

经典 RL（Atari/MuJoCo）：state-action 空间小，rollout 便宜，on-policy 可行（A3C/PPO 经典版每步重采）。IS 主要用于 off-policy 复用（DQN 的 experience replay 用 IS 修正，PER 用 IS 权重采样）。

LLM-RL：rollout 极贵（生成几百~几千 token），on-policy 不可行，IS 是救命——一次 rollout 复用多 epoch/多步。ratio 用 log 形式防下溢。clip $\varepsilon$ 调小（0.2，因 LLM logp 量大方差大）。V-trace 在 disaggregate 异步借鉴。详见 [[REINFORCE]]、[[PPO clipped objective]]。

### 8.2 多 epoch 复用与 ratio 累积

PPO/GRPO 允许在同一个 rollout batch 上跑多 epoch（PPO 典型 4 epoch，GRPO 可调）。每 epoch $\theta$ 更新，ratio $\rho=\pi_\theta/\pi_{old}$ 随 epoch 偏离增大（$\theta$ 越来越远离 $\theta_{old}$）。靠 clip $\varepsilon$ 兜底——当某 epoch ratio 超 clip 范围，该 token 梯度被截断（early stop 思想）。这是 PPO "多 epoch 复用 + clip" 的 IS 实践。epoch 数与 $\varepsilon$ 联动：epoch 多则 $\varepsilon$ 需小（防偏离过大）。详见 [[PPO clipped objective]]、[[GRPO]] §8.3。

### 8.3 与 GRPO 的 group 采样交互

GRPO 对同一 prompt 采 $G$ 条 response 组成 group，每条单独算 ratio（token 级）。group 内 $G$ 条共享同一 $\theta_{old}$（同时采），但 $\theta$ 更新后 ratio 各自偏离。group-mean baseline（advantage）与 ratio 独立——baseline 降组间方差，ratio 修 off-policy。两者正交但协同。详见 [[GRPO]] §4.1、§4.5。

### 8.4 与异步训练的 partial rollout 交互

disaggregate 异步下，rollout 采的 batch 可能跨多个 $\theta$ 版本（partial rollout）——batch 前半用 $\theta_{old-k}$、sync 后后半用 $\theta_{old}$。每条 response 记自己采时的 $\theta$ 版本号，trainer 按版本号算对应 ratio。这是异步 IS 的复杂化——ratio 不再统一 $\pi_\theta/\pi_{old}$，而是 $\pi_\theta/\pi_{old-k}$（混合 staleness）。需版本号机制 + 每条独立 ratio。详见 [[asynchronous training]]、[[synchronous与asynchronous rollout]]。

### 8.5 内容来源

IS 期望恒等式推导基于标准概率论（覆盖性假设 + 重加权）。PPO clip 的方差治理动机来自 [[PPO clipped objective]] 与 Schulman PPO 原论文（arXiv:1707.06347）。V-trace 来自 IMPALA 论文（arXiv:1802.01561）。log 形式防下溢是 LLM-RL 工程标准（verl/OpenRLHF 实现）。staleness→ratio 指数放大推导与 [[RL权重同步]] §4.2 一致。与 TIM 的耦合来自 [[rollout train reference logprob一致性]] 与 [[训推不一致]]。多 epoch 复用与 clip 联动来自 [[GRPO]] §8.3。截至 2026-07。

---
相关: [[logprob一致性]] | [[rollout train reference logprob一致性]] | [[REINFORCE]] | [[PPO clipped objective]] | [[advantage function]] | [[baseline]] | [[variance reduction]] | [[RL权重同步]] | [[stale policy problem]] | [[asynchronous training]] | [[synchronous与asynchronous rollout]] | [[训推不一致]] | [[KL penalty与KL control]] | [[GRPO体系]] | [[token-level与sequence-level objective]] | [[17-RL训推一体框架]]
