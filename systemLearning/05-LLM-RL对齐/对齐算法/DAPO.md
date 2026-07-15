# DAPO

> **所属章节**: [[整体目录#五、LLM RL（RLHF / RLAIF）|对齐算法]]
> **所属模块**: [[整体目录#五、LLM RL（RLHF / RLAIF）|05-LLM-RL对齐]]

## 1. 一句话定义

**DAPO**（**D**ecoupled clip and **D**ynamic s**A**mpling **P**olicy **O**ptimization，解耦裁剪与动态采样策略优化）是字节跳动 Seed 团队 2025-03 提出的开源 LLM 强化学习算法（arXiv:2503.14476），在 GRPO 基础上叠加 **4 个关键 trick**（Clip-Higher / Dynamic Sampling / Token-Level Loss / Overlong Reward Shaping），专治 long-CoT（长思维链）RL 场景下 GRPO 的 **熵坍塌、梯度消失、奖励噪声、长度漂移**，用 Qwen2.5-32B 在 AIME 2024 拿到 50 分（DeepSeek-R1-Zero-Qwen-32B 同规模 47 分，DAPO 用其一半 step）。实现基于 [[verl]] 框架，代码+数据全开源。

## 2. 为什么需要它（动机与背景）

DeepSeek-R1 用 RL 点燃了 reasoning LLM 的范式，但 R1 报告**隐瞒了关键训练细节**。社区照搬 GRPO 复现时集体翻车，主要栽在 4 个坑上：

| 病灶 | 现象 | 根因 |
|---|---|---|
| **熵坍塌 (entropy collapse)** | 训练几百步后策略熵快速掉到 0，生成样本趋同、失去探索能力 | PPO/GRPO 对称 clip (±0.2) 把低概率 token 的上界锁死，exploitation token 容易涨、exploration token 涨不动 |
| **梯度消失 (gradient diminishing)** | batch 里 acc=1 的 prompt 越来越多，advantage=0 → policy gradient=0，有效样本数持续下降 | GRPO 用 group reward 标准化做 advantage，全对/全错组 advantage 恒 0，无梯度信号 |
| **长度漂移 (length bias)** | 样本级 loss 归约让长样本中每个 token 权重被稀释，模型学成"越长越好"+ 出现 gibberish/repetition | sample-level loss reduction 长短样本等权 |
| **奖励噪声 (reward noise)** | 超长截断样本被简单赋负奖励，把"质量差"和"只是太长"混为一谈，误导策略 | truncation 默认惩罚过于粗暴 |

DAPO 针对这 4 个病一一对症下药，每个 trick 修一个，合起来把 GRPO 在 long-CoT 场景的几个塌方点全堵住。

## 3. 核心概念详解

### 3.1 总目标

DAPO 仍是 group-relative 的 policy gradient，每条 prompt $q$ 采样 $G$ 条响应 $\{o_i\}_{i=1}^G$，目标：

$$
J_{\text{DAPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal{D},\,\{o_i\}\sim\pi_{\theta_{\text{old}}}(\cdot|q)}
\frac{1}{\sum_i |o_i|}\sum_i\sum_t \min\!\Big(r_{i,t}(\theta)\hat A_{i,t},\;\text{clip}\big(r_{i,t}(\theta),1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}}\big)\hat A_{i,t}\Big)
$$

其中 $r_{i,t}(\theta)=\pi_\theta(o_{i,t}|q,o_{i,<t})/\pi_{\theta_{\text{old}}}(o_{i,t}|\cdots)$ 是重要性采样比，$\hat A_{i,t}$ 是 group-relative advantage（同 prompt 内 $G$ 条 reward 归一化），注意**约束 $0<|\{o_i:\text{is\_equivalent}(a,o_i)\}|<G$** —— 这是 Dynamic Sampling 的过滤条件（见 §3.3）。

### 3.2 Trick 1 — Clip-Higher：解耦上下裁剪，治熵坍塌

**问题**：标准 PPO/GRPO 用对称 clip $[1-\varepsilon,1+\varepsilon]$（$\varepsilon=0.2$）。看似对称公平，实则严重偏向 exploitation：

- 一个**已大概率** token（$\pi_{\text{old}}=0.9$）正梯度下能涨到 $1.08$（相对 $+20\%$）；
- 一个**小概率探索 token**（$\pi_{\text{old}}=0.01$）正梯度下最多涨到 $0.012$（绝对只 $+0.002$）。

低概率 token 是探索的种子，却被上界锁死，模型快速收敛到少数高概率 token 上 → **熵坍塌**。

**做法**：解耦 $\varepsilon$ 为 $\varepsilon_{\text{low}},\varepsilon_{\text{high}}$，**抬高上界**放宽探索空间，下界保持小避免直接抹杀 token：

$$
\varepsilon_{\text{low}}=0.20,\qquad \varepsilon_{\text{high}}=0.28
$$

> [!tip] 直觉
> "鼓励往新方向迈大步，但禁止把已有方向砍得太狠"。上界松=奖励探索，下界紧=保命不坍塌。

### 3.3 Trick 2 — Dynamic Sampling：prompt-group 级过滤，治梯度消失

> [!warning] 这是用户批注的核心问题
> 用户批注"DAPO 在 Timely rollout 上提供 **prompt-group 级** dynamic sampling、pre-filter、post-filter、replay 和 refill 语义"——这一串术语其实是**对 DAPO Dynamic Sampling 在 rollout 系统实现层面的语义拆解**，不是论文原文用词，下面逐词对译。

**问题**：GRPO 用 group-relative advantage $\hat A_i=(r_i-\bar r)/\text{std}$。如果一个 prompt 的 $G$ 条全对（acc=1，reward 都=1）或全错（acc=0，reward 都=0），$\text{std}=0$，归一化后 $\hat A_i=0$，**整组零梯度**。训练越往后 acc=1 的 prompt 越多（模型在变强），batch 里有效样本数持续掉，梯度信噪比变差、方差变大。

**做法**：**过采样 + 过滤**。采样时**丢弃**那些 reward 全相同（全 0 或全 1）的 prompt-group，只保留"既有对又有错"的混合组（$0<\text{correct\_count}<G$），**反复采样直到 batch 凑满** N 个有效 prompt-group。

> [!note] 解答：批注里那 5 个词逐个翻译
> 把论文的 "keep sampling until the batch is fully filled with samples whose accuracy is neither 0 nor 1" 这一句话，按 rollout 系统的工程语义拆开看：
>
> | 批注用词 | 工程语义 | 对应 verl 代码动作 |
> |---|---|---|
> | **Timely rollout** | "及时采样"——采样/过滤发生在 rollout 生成**当场**（rollout stage），而不是事后再筛；同 step 内边采边判 | `gen_batch_size` 一次性采 $G\times$ `gen_batch_size` 条，立即按 metric 判 |
> | **prompt-group 级** | 过滤粒度是 **一个 prompt + 它的 $G$ 条响应** 整组判，而非单条 response；全组 reward 标准差=0 才整组扔 | `filter_groups.metric: acc`，按 group 维度算 std/all-same |
> | **dynamic sampling** | 采多少是**动态**的——单轮采不够就再采，采到够为止；采样量随训练进度（acc=1 越多→越难凑）自动涨 | `num_gen_batches += 1; continue` 循环 |
> | **pre-filter** | 在**进训练 batch 之前**过滤（rollout 端就把 trivial 组剔除），不浪费 forward+reward+ref 的算力 | `if num_prompt_in_batch < prompt_bsz: continue` |
> | **post-filter** | 在**累积阶段**再判一次——已采到的合格组累积，没满就继续采；满了对齐截断 | `batch = batch[:traj_bsz]` 切到精确 `train_batch_size * n` |
> | **replay（重采）** | 不够就 **continue 回到采样循环重采一轮**，这一轮采到的新组合并进累积 buffer | 外层 `while num_prompt_in_batch < prompt_bsz` 循环 |
> | **refill（补满）** | 把被剔除的空位**用新合格组补满**到目标 `train_batch_size`；最终 batch 恒定大小 | 达到 `prompt_bsz` 后 break，对齐切片 |
>
> **一句话**：DAPO 在 rollout 环节"当场判定、整组过滤、不够重采、补满为止"，保证每个 training batch 都是 N 个**带有效梯度信号的混合 prompt-group**，batch 大小恒定、梯度信号稳定。论文叫 Dynamic Sampling，rollout 系统视角就是这 5 个语义。

**约束写进目标函数**：公式里的 `s.t. $0 < |\{o_i:\text{is\_equiv}\}| < G$` 就是这个过滤条件的数学表达——保留组必须既不全对也不全错。

> [!tip] 不亏效率
> 看似多采很多浪费，实则不然：同步 RL 系统 rollout 耗时被 **长尾样本**（最长的那条）主导，多采几条短样本几乎不增加 wall-clock；论文实测加 Dynamic Sampling 反而**收敛更快**（同性能更少 step）。

### 3.4 Trick 3 — Token-Level Policy Gradient Loss：治长度漂移

**问题**：GRPO 默认 **sample-level loss**：先在每条 response 内 token 平均，再在 batch 内 sample 平均。后果：每条样本等权，**长样本里每个 token 的权重被稀释**。模型学成"长样本的 token 贡献小，多生成点无害"，于是长度疯涨、gibberish/repetition 涌现。

**做法**：改成 **token-level loss**——直接对 batch 内**所有 token** 做平均，不分 sample 边界：

$$
\text{loss} = \frac{1}{\sum_i |o_i|}\sum_i\sum_t \big[\cdots\big]_{i,t}
$$

这样**长样本对总梯度贡献更大**（token 多权重大），长样本里的好模式被强化、坏模式（repetition）被惩罚——"每个 token 都为自己造成的 reward 变化负责"。

> [!note] 数学等价性
> 当所有样本等长时两种归约数学等价；差异只在变长场景才显化，long-CoT 恰是变长场景，故 token-level 在此关键。

### 3.5 Trick 4 — Overlong Reward Shaping：治奖励噪声

截断样本的奖励处理两步走：

1. **Overlong Filtering**：被强制截断的样本（length ≥ hard cap），**loss 里 mask 掉**（不参与梯度）。这些样本 reward 噪声大，强行学反而混淆"质量"与"长度"。
2. **Soft Overlong Punishment**：对落在惩罚区间 $[L_{\max}-L_{\text{cache}},\;L_{\max}]$ 的样本，**按长度线性追加负奖励**：

$$
R_{\text{len}}(L) = \begin{cases}
0 & L \le L_{\max}-L_{\text{cache}} \\
-\dfrac{L-(L_{\max}-L_{\text{cache}})}{L_{\text{cache}}} & L_{\max}-L_{\text{cache}} < L \le L_{\max} \\
-1 & L > L_{\max}
\end{cases}
$$

加到原 rule-based correctness reward 上。越接近硬截断惩罚越重，平滑引导模型自己学会"该停就停"。

> [!tip] 论文实测
> 官方实验里**最优配置其实没用 Overlong Filtering**（和 Soft Punishment 功能重叠），只开 Soft Punishment；verl 默认实现也如此。Filtering 是 ablation 用的对照组。

## 4. 数学原理 / 公式

### 4.1 完整目标（含全部 4 trick）

$$
\boxed{\;
J_{\text{DAPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal{D},\,\{o_i\}_{i=1}^G\sim\pi_{\theta_{\text{old}}}(\cdot|q)}
\frac{1}{\sum_i |o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\min\!\Big(
r_{i,t}(\theta)\,\hat A_{i,t},\;
\text{clip}\big(r_{i,t}(\theta),\,1-\varepsilon_{\text{low}},\,1+\varepsilon_{\text{high}}\big)\,\hat A_{i,t}
\Big)
\;}
$$

约束：$0 < |\{o_i:\text{is\_equivalent}(a,o_i)\}| < G$（Dynamic Sampling 过滤条件）

### 4.2 group-relative advantage（继承自 GRPO）

$$
\hat A_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}
$$

- 全对组：$\text{std}=0, \hat A=0$ → **Dynamic Sampling 剔除**
- 全错组：同上 → 剔除
- 混合组：$\hat A\in[-1,1]$ 附近，**保留**

### 4.3 推导：为什么对称 clip 偏向 exploitation

重要性采样比 $r=\pi_\theta/\pi_{\text{old}}$，clip 后等价更新概率上界 $\pi_\theta \le (1+\varepsilon)\pi_{\text{old}}$。

- $\pi_{\text{old}}=0.9,\ \varepsilon=0.2$：上限 $1.08$，**绝对增量** $+0.18$
- $\pi_{\text{old}}=0.01,\ \varepsilon=0.2$：上限 $0.012$，**绝对增量** $+0.002$

低概率 token 的绝对可涨空间比高概率小 **90 倍**，探索被结构性压制。Clip-Higher 把 $\varepsilon_{\text{high}}$ 抬到 $0.28$，低概率 token 上限变 $0.0128$，相对增量 $+28\%$（vs $+20\%$），缓解但未根除（故 DAPO 是**缓解**而非消除熵坍塌）。

## 5. 代码示例

### 5.1 verl 配置（生产配方，Qwen2.5-32B 复现 50% AIME）

```yaml
# verl recipe/dapo/run_dapo_qwen2.5_32b.sh 关键片段
data:
  gen_batch_size: 1536        # 每轮过采样 prompt 数（>train 才有过滤余量）
  train_batch_size: 512       # 目标训练 prompt 数（凑满有效组）

actor_rollout_ref:
  rollout:
    n: 16                     # 每个 prompt 采 16 条响应 = group size G
    response_length: 20480    # 硬截断 max_length
  model:
    path: Qwen/Qwen2.5-32B

algorithm:
  adv_estimator: grpo         # group-relative advantage
  loss_agg_func: "token-mean" # ← Trick 3: token-level loss
  use_valid_token_scale: True
  filter_groups:              # ← Trick 2: Dynamic Sampling
    enable: True
    metric: acc
    max_num_gen_batches: 10   # 最多重采 10 轮，防死循环
  clip_ratio_low: 0.20        # ← Trick 1: ε_low
  clip_ratio_high: 0.28       # ← Trick 1: ε_high
  kl_penalty_type: kl         # DAPO 实测弱化 KL（long-CoT 允许大幅漂移）
  kl_beta: 0.0

reward:
  overlong_buffer:            # ← Trick 4: Soft Overlong Punishment
    enable: True
    soft_max_length: 16384    # L_max
    soft_cache_length: 4096   # L_cache
```

### 5.2 Dynamic Sampling 核心循环（verl `dapo_ray_trainer.py` 简化版）

```python
prompt_bsz = config.data.train_batch_size      # 目标 N
buffer = []                                    # 累积合格 prompt-group
num_gen_batches = 0

while len(buffer) < prompt_bsz:                # ── refill：没满就继续
    # ── timely rollout：本轮一次性过采
    batch = rollout.gen(gen_batch_size, n=group_size)   # gen_batch_size * G 条
    batch = reward_model.score(batch)                   # 给每条打 reward
    num_gen_batches += 1

    # ── pre-filter：prompt-group 级过滤（acc 全 0 或全 1 的整组扔）
    valid = [g for g in batch.group_by_prompt()
             if 0 < g.acc_count < g.size]                # 保留混合组
    buffer.extend(valid)                                # ── replay：不够继续

    if num_gen_batches >= max_num_gen_batches:         # 防死循环熔断
        raise RuntimeError("Generated too many, check data")

    # ── post-filter：满则切片对齐
traj_bsz = prompt_bsz * group_size
buffer = buffer[:traj_bsz]                              # 对齐到精确大小
return buffer
```

## 6. 与其他知识点的关系

- **上游（依赖）**:
  - [[RLHF (PPO)]] —— DAPO 是 PPO 系在 LLM long-CoT 场景的特化
  - [[PPO clipped objective]] —— Clip-Higher 直接改的就是 PPO clip 范围
  - [[verl]] —— 官方实现框架
- **下游（应用）**:
  - long-CoT reasoning RL（AIME/MATH/GSM8K 类数学推理训练）
  - R1-Zero 范式复现（DAPO 是开源复现 R1 风格 RL 的标杆）
- **对比 / 易混**:
  - **vs [[GRPO]]**：DAPO = GRPO + 4 trick，group-relative advantage 同源；区别在 clip 解耦、token-level loss、dynamic sampling、length shaping
  - **vs vanilla PPO**：PPO 用 critic 估 value baseline，DAPO/GRPO 用 group-relative 不需 critic，省一个 value 网络显存
  - **vs [[DPO]]/[[IPO]]/[[KTO]]**：DPO 系是 offline 偏好对齐，不需在线 rollout；DAPO 是 online RL，需 rollout worker 实时采

## 7. 常见误区与易错点

1. **"Dynamic Sampling = 在线学习/数据重放"** ❌ —— 只是**同一 step 内的过采样+过滤**，不跨 step；不是 replay buffer 那种 off-policy 历史回放。
2. **"Dynamic Sampling 一定增加 wall-clock"** ❌ —— 同步 RL rollout 时间被最长样本主导，多采短样本几乎免费；实测反而**加速收敛**（有效 step 多了）。
3. **"Clip-Higher 越大越好"** ❌ —— $\varepsilon_{\text{high}}$ 太大 → ratio 飞出 trust region → 训练不稳；论文用 0.28 是甜点，别瞎调。
4. **"token-level loss = 不归一化"** ❌ —— 仍归一化，只是按 token 数而非 sample 数归一；长短样本贡献按 token 比例分配。
5. **"DAPO 去掉了 KL penalty"** —— 论文实验里弱化（`kl_beta=0`），因为 long-CoT 策略**合理地**大幅偏离初始 SFT 模型；但配置仍提供 KL 项，场景不同可开。
6. **"Overlong Filtering 和 Soft Punishment 都要开"** ❌ —— 两者功能重叠，官方最优配置只开 Soft Punishment；Filtering 仅 ablation 用。
7. **"DAPO 是新算法不是 GRPO"** —— 严格说 DAPO **是 GRPO 的 trick 包**，advantage 估计同 GRPO（group-relative），4 个 trick 都是工程化改进非理论突破。

## 8. 延伸细节

### 8.1 复现基准

| Setup | AIME 2024 Acc |
|---|---|
| DAPO w/o Token-level & Dynamic Sampling | 44% |
| DAPO w/o Dynamic Sampling | 50% |
| **DAPO (full)** | **52%** |
| DeepSeek-R1-Zero-Qwen-32B | 47% |

### 8.2 训练超参（Qwen2.5-32B 官方）

| 项 | 值 |
|---|---|
| optimizer | AdamW |
| lr | $1\times10^{-6}$ 常数 + 20 rollout step linear warmup |
| prompt batch | 512 |
| group size $G$ | 16 |
| mini-batch | 512（每 rollout 16 次梯度更新） |
| max length | 20480（$L_{\max}=16384 + L_{\text{cache}}=4096$） |
| $\varepsilon_{\text{low}}/\varepsilon_{\text{high}}$ | $0.20 / 0.28$ |

### 8.3 与训推一致性 / 异步训练的交叉

- DAPO 的 Dynamic Sampling 假设**同步 RL**（同 step 内 rollout 用同一 $\theta_{\text{old}}$）；在 [[asynchronous training]] 异步系统里 replay/refill 跨 step 采样会引入 staleness，需配合 staleness PPO 或 V-trace 修正。
- rollout 端的 vLLM/SGLang 推理引擎算出的 $\log\pi_{\theta_{\text{old}}}$ 必须与训练端对齐，否则 advantage 偏；详见 [[训推不一致]]。

### 8.4 后续工作（2025-2026）

- **band-pass filter**（verl PR #6155）：把硬过滤 $0<\text{acc}<G$ 改成软阈值 $[0.2, 0.8]$ 区间保留，聚焦中等难度样本。
- **DUPO / GSPO**：同方向后续算法，分别从 clip 策略和 group 归一化再改进。
- **StreamRL / AReaL 异步化**：把 DAPO 的同步过采样改造成流式异步，replay/refill 跨 step 不阻塞 learner。

---
相关: [[RLHF (PPO)]]、[[GRPO]]、[[verl]]、[[训推不一致]]、[[asynchronous training]]、[[rollout worker]]、[[整体目录]]
