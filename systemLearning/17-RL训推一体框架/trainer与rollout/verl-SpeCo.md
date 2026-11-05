# verl-SpeCo

> **所属章节**: [[trainer与rollout]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §在整体目录 trainer与rollout 条目下
> **别名**: verl-SpeCo / SpeCo / speculative co-training / 投机协同训练
> **难度**: 高（需懂 [[speculative decoding]]、[[weight sync mechanism]]、[[stale policy problem]]、[[训推不一致]]、[[trainer与rollout引擎组合]]）
> **来源**: [verl-project/verl-SpeCo](https://github.com/verl-project/verl-SpeCo)、[verl blog 2026-08-13](https://verl-project.github.io/posts/2026-08-13-verl-speco-release/)、[Vincent Zhong — Keeping Speculative Decoding Fast During RL Training](https://vincentzed.com/writing/verl-speco)、[verl PR #5925](https://github.com/verl-project/verl/pull/5925)


## 1. 一句话定义

**verl-SpeCo** 是 verl 上的一个**轻量 drafter 协同训练 overlay**：它把 [[speculative decoding|投机解码]]的 draft model 训练**嵌入 RL 训练循环内部**——在 rollout 阶段收集 target 的 hidden states、每若干步训练一次 drafter、随策略权重一起热更新到 rollout 引擎，从而让 drafter 在 policy 持续演化的 RL 训练中**始终保持高接受率**，避免"静态 drafter 因 policy drift 而失效"。名字 **SpeCo = Speculative Co-training**（投机协同训练）。

> [!note] 定位
> verl-SpeCo **不是**新的投机解码算法，而是**让现有 drafter（EAGLE-1/2/3、DFlash、DFlash2、DSpark、Domino、P-EAGLE）在 RL 训练中不掉队**的训练-发布框架。它把上游 verl 当作 import-only 依赖，通过 `python -m verl_speco.main` 启动，**不修改已安装的 verl 代码树**；drafter 关掉时跑的就是 plain verl。


## 2. 为什么需要它（动机与背景）

### 2.1 RL rollout 是训练墙钟时间的大头

reasoning 模型的 RL 训练（GRPO/PPO）每 step 要采成百上千条 response，每条数千 token。以 GRPO 为例：512 条 response × 均长 4000 token ≈ **每 step 生成 200 万 token**。即便 8 卡节点跑到 20000 tok/s，单 step 生成也要 **100+ 秒**，几百 step 的训练生成占比极大——[Together AI 报告 rollout 占总训练时间约 70%](https://www.together.ai/blog/distribution-aware-speculative-decoding)。

一个同步 RL step 的四阶段时间分解：

$$
T_{\text{step}} = T_{\text{gen}} + T_{\text{old}} + T_{\text{ref}} + T_{\text{update}}
$$

- $T_{\text{gen}}$：rollout 生成（最大头）
- $T_{\text{old}}$：old-logprob pass（用训练引擎重算生成 token 的 logp，[[rollout train reference logprob一致性]]）
- $T_{\text{ref}}$：reference model forward（KL penalty，无 KL 时跳过）
- $T_{\text{update}}$：optimizer 更新

[[speculative decoding]] 攻击的就是最大的 $T_{\text{gen}}$——让每次 target forward 多产出几个 token。

### 2.2 核心痛点：policy drift 让 drafter 失效

[[speculative decoding]] 的加速取决于 **acceptance length $\tau$**（每次 verify 产出多少 token），而 $\tau$ 取决于 drafter 分布 $q$ 与 target 分布 $p$ 的吻合度（$\alpha = \sum_x \min(p,q) = 1 - \text{TV}(p,q)$）。

drafter 之所以猜得准，是因为它**被训练来模仿某一个特定 target checkpoint**。在 serving 场景这没问题——发布模型几个月不变。但 **RL 训练每步都在更新 policy**（这正是 RL 的目的），policy 一动，drafter 模仿的那个旧 checkpoint 就过时了：

> **RL 每步更新 → policy 离 drafter 训练时的 checkpoint 越来越远 → $\alpha$ 持续下降 → $\tau$ 下降 → 加速比萎缩。**

[verl PR #5925](https://github.com/verl-project/verl/pull/5925) 的实验观察到：静态 EAGLE-3 drafter 在 RL 训练到 step 100–120 时 acceptance 明显掉，甚至可能 reward collapse。越长训练越严重——run 越长 policy 漂移越远，且长 run 生成时间占比也最大。

### 2.3 朴素的三个解法及其代价

| 解法 | 代价 |
|---|---|
| **每 N 步离线重训 drafter** | 暂停训练 + 用当前模型生成新数据 + 跑独立训练 job → 推理算两次 + 多一条 pipeline 要维护 |
| **关掉 speculative decoding** | 放弃加速，恰好在 rollout 最长的中后期 |
| **保留旧 drafter** | 长跑后可能比不开 spec 还慢——$\tau \to 1$ 时既要付 draft 开销又要付 verify 开销，却只拿回本就免费的那 1 个 token |

### 2.4 SpeCo 的洞察：训练数据本就在 RL 循环里

verl-SpeCo 的关键发现：**drafter 需要的训练数据，RL 循环本来就在算，然后扔掉了。**

所有 drafter（EAGLE-3/DFlash/DSpark）的训练样本形式都一样：**target 在某位置的 hidden states + target 真正生成的下一个 token（或其完整分布）**。而 RL 训练每个 step 都对每个生成 token 计算了这两样：

1. **生成阶段**：rollout 引擎算 target 的 hidden states；
2. **old-logprob pass（$T_{\text{old}}$）**：训练引擎对所有生成 token 跑一遍 forward 重算 logp——这次 forward 同时产出 hidden states 和 target 的完整下一 token 分布。verl 算完就**丢弃**了它们。

verl-SpeCo 就是把这份"本要丢弃的免费教师信号"收集起来训 drafter。且因为 [[speculative decoding]] **不改变输出分布**（见 [[投机解码原论文精读]] §4 的无损证明），用这些数据训 drafter **不 bias RL 训练本身**——drafter 只影响 rollout 到达训练 buffer 的速度，不影响内容。

> [!tip] 一句话动机
> **RL 循环已经免费算出了 drafter 的训练数据再扔掉；speculative decoding 又不改变输出分布——那就用这些数据在循环内持续把 drafter 拉回当前 policy，零 bias、近乎零额外推理成本。**


## 3. 核心概念详解

### 3.1 三个 Hook：Collect / Train / Publish

verl-SpeCo **从外部包裹** verl 的训练循环（不改动 verl 代码），在 RL step 的三个点插 hook：

| Hook | 时机 | 动作 |
|---|---|---|
| **Collect** | 生成阶段或 old-logprob pass | 保存 target 的 hidden states（drafter 的输入）+ 教师信号 |
| **Train** | 策略 update 之后、同一组 GPU 上 | 用收集的数据跑一轮小型 drafter 训练 |
| **Publish** | 随策略权重同步一起 | 把新 drafter 权重热加载进 rollout 引擎 |

drafter 关闭时，三个 hook 全空，跑的就是 plain verl。

```
一个 RL step（verl-SpeCo 视角）:
  [verl 原生] 生成 rollout ──(Collect: 存 hidden states)
  [verl 原生] old-logprob pass ──(Collect: 存 hidden states, 备选源)
  [verl 原生] ref forward (KL)
  [verl 原生] policy update
  [SpeCo]     Train: 训 drafter（10 optimizer steps）
  [SpeCo]     Publish: drafter 权重 + policy 权重一起送 rollout 引擎
                 → rollout 引擎: 中断生成、清 KV cache、写新 drafter、再清 KV cache、恢复生成
```

### 3.2 两个 hidden states 来源

drafter 的输入是 target 的 hidden states，有两个采集源，**选择主要取决于 rollout 引擎**：

| 来源 | 谁产出 | 适用 | 说明 |
|---|---|---|---|
| **rollout 引擎** | 生成时引擎本就算 hidden states | **只有 SGLang 能返回**（DFlash worker除外） | 这些就是 drafter serving 时看到的张量——同引擎同 kernel，最对齐 |
| **old-logprob pass** | 训练引擎重算 logp 时顺便存 | **始终可用**（vLLM 路径全用它） | 不额外 forward；只依赖训练引擎 |

> [!warning] 两个来源的张量不完全相同
> 引擎用不同 kernel；SGLang 返回的是**最后一层 final norm 之前**的 hidden states，训练 pass 则**应用了 norm**。对读最后一层的 drafter，verl-SpeCo 因此**拒绝 SGLang 采集**，除非 run 显式声明训练与 serving 都用 pre-norm 值。这是 [[训推不一致]] 在 drafter 维度的一个具体体现。

### 3.3 收集什么、保留多少

默认策略（每 5 步触发一次）：

- 每个 rollout replica 最多保留 **16 个样本**；
- 每个样本只取 response 的**前 512 token**，跳过更短的；
- 按 batch 顺序取样 → **重跑会选同样的样本**（可复现）；
- 每个样本记录**哪个模型、哪些层、哪一步**产出 → 训练时能识别并**拒绝来自旧模型的数据**（防 staleness 污染）。

### 3.4 训练一轮 drafter

训练轮也每 5 步跑一次（攒够新数据后）。**故意小而频繁**——常把 drafter 推向当前 policy，每次成本很低：

| 超参 | 默认值 |
|---|---|
| optimizer steps / 轮 | 10 |
| batch / GPU | 4 样本 |
| learning rate | $10^{-5}$ |
| 数据 | 默认仅当前步收集的（不混历史） |

> [!note] drafter 总是滞后几步，有问题吗？
> RL 用小 lr 且常有 KL penalty，5 步内 policy 移动很小。若 acceptance 仍掉得比 schedule 快，可缩短间隔（代价是更多训练时间）。

### 3.5 各 drafter 的 loss

loss 跟随各 drafter 原论文，**对决定 acceptance 的位置加权**：

**DFlash**（块扩散，一次填一整块 blank）——交叉熵，块内位置 $k$ 权重指数衰减：

$$
\mathcal{L}_{\text{DF}} = -\sum_{k=1}^{K} w_k \log q(x_{a+k} \mid x_{\le a}, c), \quad w_k = \exp\!\left(-\frac{k-1}{\gamma}\right),\ \gamma=7
$$

其中 $a$ 是 anchor 位置，$c$ 是 target hidden states，$q$ 是 drafter 分布。块第 1 个 blank 权重 1，第 8 个约 0.37——因为块首猜错整块作废，块尾错只丢 1 token。

**EAGLE-3**——默认用复制来的 target output layer（见 §4.1）算 target 完整下一 token 分布（限制到 drafter 的小词表），对它做交叉熵。

**DSpark**——**直接为 acceptance 训练**的典型：混合 CE（权重 0.1）+ L1（权重 0.9），L1 项是 drafter 与 target 分布的距离、按同样的 $w_k$ 加权。由 §4.2，acceptance 概率 $= 1 - \tfrac{1}{2}\|q-p\|_1$，所以**缩小 L1 距离 = 直接提升 acceptance**。（verl-SpeCo 关掉论文的 confidence-head loss，因为它的 vLLM 集成始终 verify 固定数量 token。）

### 3.6 Publish：热更新 drafter 到 rollout 引擎

训练轮成功后，**一个 worker** 导出新 drafter 权重并加载进 rollout 引擎。关键设计：

- **与策略自身的 weight update 同步发生** → 每个引擎在两次生成阶段的间隙切换 drafter；
- 默认**同步 publish** → 所有引擎始终用同一版 drafter（不会有的新有的旧）；
- 引擎内 swap 流程：**中断生成 → 清 KV cache → 写新 drafter 权重 → 再清 KV cache → 恢复生成**。

这呼应 [[verl PR #5925]] 发现的两个工程坑：actor weight sync 后必须**重载 draft 权重**、重载后**重建 RoPE cos_sin cache**，否则 draft 跑飞。verl-SpeCo 把这个 sync 做成了自动化、版本化的 publish。

### 3.7 版本检查（防训错数据）

drafter 在多个 worker 上同时训。若一轮里部分 worker 开训、部分跳过，或某 worker 读的数据在轮规划后又被改，就会训出"基于错误数据"的 drafter。verl-SpeCo 给每轮带**版本号**（plan / collected data / 每个 worker process 三层），训练前所有 worker 校验，任一不符 → **全员跳过该轮、不 publish、rollout 引擎保留旧 drafter**。

### 3.8 支持矩阵

| Draft model | vLLM | vLLM-Ascend(NPU) | SGLang | 训练引擎 |
|---|---|---|---|---|
| EAGLE-1 | ✓ | runtime-specific | — | FSDP |
| EAGLE-2 | ✓ | runtime-specific | — | FSDP |
| EAGLE-3 | ≥0.18.0 | ≥0.18.0 | ≥0.5.10 | FSDP |
| DFlash | ≥0.20.2 | ≥0.20.2 | ≥0.5.12 | FSDP |
| DFlash2 | ≥0.28.0 | — | main | FSDP |
| DSpark | GPU:main / NPU:`58d3918` | NPU:`6af9257`/MRV2:`27a9476` | — | FSDP |
| Domino | DFlash-compatible | runtime-specific | runtime-specific | FSDP |
| P-EAGLE | 未接入 | 未接入 | 未接入 | 仅训练 |

drafter trainer 始终在 **FSDP2** 上；主 actor 可用 verl 的 FSDP/FSDP2 或 **VeOmni** 模型引擎（dense/MoE/多模态 actor 都行，SpeCo adapter 补 old-logprob hidden-state 与 lm-head 同步路径）。


## 4. 数学原理 / 公式

### 4.1 教师信号近乎免费

drafter 训练的"标签"是 target 的下一 token 分布 $p$，它就是 **target 的 output layer（lm_head）作用于 target 最后 hidden state**。所以**不必存 target 的完整分布**，只要把当前 output layer 中覆盖 drafter 词表的那几行复制给 drafter trainer，trainer 自己就能算教师分布。

以 Qwen3-8B 为例：

| 项 | 大小 |
|---|---|
| 完整 output layer（151936 × 4096，FP16） | ~1.24 GB |
| 存完整分布（151936 × 2B × 8192 位置） | ~2.5 GB |

存 lm_head 行远比存分布省——且 lm_head 本就是 policy 的一部分，weight sync 时自然带着。

### 4.2 drift 可量化

由 [[投机解码原论文精读]] 的接受规则，target 接受 drafter 提的 token $x$ 的概率：

$$
A(x) = \min\!\left(1, \frac{p(x)}{q(x)}\right)
$$

单位置期望接受概率：

$$
\alpha = \mathbb{E}_{x \sim q}[A(x)] = \sum_x \min(p(x), q(x)) = 1 - \tfrac{1}{2}\|q - p\|_1
$$

所以**只要有 target 的 hidden states 跑一遍 drafter 对比两分布，不用解码就能读出 $\alpha$**——drift 实时可测。verl-SpeCo 实际记录的是 vLLM 在 rollout 中观测到的 **mean acceptance length**，直接显示 drift 发生过程（指标 `drafter/spec_decode/mean_acceptance_length`）。

### 4.3 加速比与 drift 的敏感度

设每轮 draft $\gamma$ 个 token，单步接受概率 $\alpha$，则每次 verify pass 产出（含 bonus）：

$$
\tau = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

| $\alpha$ | $\tau$（$\gamma=4$） |
|---|---|
| 0.5 | 1.94 |
| 0.7 | 2.77 |
| 0.8 | 3.36 |
| 0.9 | 4.10 |

**drift 的杀伤力**：$\alpha$ 从 0.9 掉到 0.85，$\tau$ 从 4.10 掉到 3.71——**约 10% 的速度直接没了**。这就是为什么"drafter 滞后几步"虽小却必须治理。

考虑 drafter 自身成本 $\beta$（draft 一块 = $\beta$ 个 target pass），净加速：

$$
S = \frac{\tau}{1 + \beta} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(1+\beta)}
$$


## 5. 代码示例（配置开关）

verl-SpeCo 用同一个 entrypoint `python -m verl_speco.main`，drafter 开关在 `actor_rollout_ref.rollout.drafter.*`：

```bash
python -m verl_speco.main \
  +actor_rollout_ref.rollout.drafter.enable=True \
  +actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
  +actor_rollout_ref.rollout.drafter.model_path=/path/to/drafter \
  +actor_rollout_ref.rollout.drafter.speculative_algorithm=EAGLE3
```

主要配置组：

```yaml
actor_rollout_ref:
  rollout:
    drafter:
      enable: True                       # rollout 时开 speculative decoding
      enable_drafter_training: True      # 开在线 drafter trainer worker
      model_path: /path/to/drafter
      speculative_algorithm: EAGLE3      # EAGLE1/2/3, DFLASH, DSPARK, ...
      rollout:
        num_speculative_tokens: 3        # lookahead K
        top_k: ...                       # draft tree 宽度
      training:
        collect_interval: 5              # 每 5 步收集一次 hidden states
        train_interval: 5                # 每 5 步训一轮
        publish_interval: 5              # 每 5 步 publish 一次
        max_samples_per_replica: 16
        max_tokens_per_sample: 512
        optimizer_steps: 10
        batch_size_per_gpu: 4
        lr: 1.0e-5
        update_mode: sync                # 同步 publish
      vllm:
        # vLLM 专属 drafter 覆盖
```

> 对比：**不开 SpeCo、只用 verl 原生静态 speculative**（[verl PR #5925](https://github.com/verl-project/verl/pull/5925)）的配置在 `rollout.speculative_decoding.*`：
> ```yaml
> rollout.speculative_decoding.method: eagle3
> rollout.speculative_decoding.num_draft_tokens: 4
> rollout.speculative_decoding.draft_tensor_parallel_size: 1
> ```
> 静态路径的工程坑（sync 后重载 draft + 重建 RoPE cache）在 SpeCo 里由 publish 自动处理。


## 6. 与其他知识点的关系

- **上游（依赖）**:
  - [[speculative decoding]]（SpeCo 加速的对象；drafter 架构 EAGLE-3/DFlash/DSpark 的原理）
  - [[投机解码原论文精读]]（无损性证明——是 SpeCo 敢在 RL 循环内训 drafter 而不 bias 训练的数学根）
  - [[weight sync mechanism]]（publish 复用策略权重同步路径）
  - [[stale policy problem]]（drafter 滞后是 stale policy 在 drafter 维度的体现）
  - [[trainer与rollout引擎组合]]（FSDP2 drafter trainer + vLLM/SGLang rollout 的两端组合）
  - [[V1 async trainer]]（异步 RL 下 SpeCo 的收益位置从 $T_{\text{gen}}$ 变为 rollout 到达率，见 §8）
- **下游（应用）**:
  - [[sampling throughput]]（SpeCo 直接提 RL rollout 采样吞吐）
  - [[训推不一致]]（hidden states 两来源的 norm/kernel 差异是一类新不一致）
- **对比 / 易混**:
  - **vs [[speculative decoding]]（静态）**：静态 drafter 训一次冻结；SpeCo 在 RL 循环内持续重训。静态在长 run 后失效，SpeCo 维持。
  - **vs slime / Miles（joint MTP）**：DeepSeek-V3/GLM-4.7 等模型**自带 MTP 层**，slime/Miles 把 MTP 层放进策略自身的 backward 一起训——无独立 trainer、无 hidden-state pipeline、无独立 publish，最简；但**只支持自带 MTP 的模型**。SpeCo 支持独立 drafter（EAGLE/DFlash/DSpark），适用面广但多一套 publish 流程。
  - **vs SpecForge（disaggregated）**：SGLang 项目的 SpecForge 把 drafter 训练**搬到独立 GPU**，从 live SGLang server 抓 hidden states 写入 Mooncake 分布式对象存储（RDMA）——训练完全不占 RL GPU；但 SpecForge 自己不跑 RL，要跟 RL run 需框架不断把最新模型灌进它的 server。drafter 滞后程度取决于灌模型的频率。
  - **vs Decoupled Speculation**（arXiv:2511.16193）：把 draft 与 verify 解耦成异步流水线消除 bubble，是**调度层**优化；SpeCo 是**训练层**让 drafter 不过时。两者正交可叠加。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 SpeCo 改变了 RL 训练的输出分布
> 没有。[[speculative decoding]] 数学上严格无损（输出分布 = target），drafter 好坏只影响速度不影响分布。所以**在循环内训 drafter 不 bias RL**——verl-SpeCo 甚至默认拒绝有损 acceptance 方法，除非 run 显式 opt-in。

> [!warning] 误区 2：把 RL benchmark 数字当 serving 数字
> verl-SpeCo 报告 Qwen3-8B + EAGLE-3（Ascend NPU）：**rollout 快 ~20%、端到端训练快 ~11%**，远小于 serving 的 EAGLE-3 4.4× / DFlash 6.1×。四个原因：①baseline 可能已含静态 drafter；②serving 数字是单请求 + draft tree，RL 是大 batch + 单链 3 token；③RL 数字含 collect/train/publish 开销；④生成只是四阶段之一（20% 快的 rollout → 11% 快的 step，反推生成至少占 step 一半）。别拿 serving 加速比套 RL。

> [!warning] 误区 3：hidden states 两个来源混用
> vLLM 路径**只能**用 old-logprob pass 的 hidden states（vLLM 不返回生成时 hidden states）；SGLang 可用生成时 hidden states 但**是 pre-norm**，与训练 pass 的 post-norm 不一致。混用会让 drafter 训练数据与 serving 数据错位，acceptance 反而掉。

> [!warning] 误区 4：忘 weight sync 后重载 drafter + 重建 RoPE cache
> 这是 [verl PR #5925](https://github.com/verl-project/verl/pull/5925) 踩的两个关键坑。SpeCo 的 publish 自动化了它，但若自己手搓静态 spec + RL，actor weight sync 后 draft 还指向旧权重 → acceptance 崩 + 输出偏差。

> [!warning] 误区 5：以为 drafter 滞后 5 步无所谓
> $\alpha$ 从 0.9 掉到 0.85 就丢 10% 速度（§4.3）。5 步滞后若 policy 移动快（高 lr / 无 KL），acceptance 掉得比 train interval 快，需要缩短间隔。


## 8. 延伸细节

### 8.1 异步 RL 下 SpeCo 的收益位置

同步 RL 里 speculative 缩短 $T_{\text{gen}}$ 直接缩短 step。异步 RL（[[V1 async trainer]]）移除了生成与训练的 barrier——trainer 不等最慢 rollout。此时 spec 的收益从"$T_{\text{gen}}$ 缩短"变为**"finished rollout 到达训练 buffer 的速率提高"**。但引擎仍要逐 token 解码每条 rollout，异步不加速解码本身；且更快 rollout 让权重滞后步数更短（每条 rollout 解码期间跨越的 trainer step 更少）→ **off-policy 程度更低**。所以异步 RL + speculative 是天然搭配（Miles 框架两者都用：GLM-5.2 744B，64 GB300，rollout 权重平均滞后 1.7 步）。

### 8.2 benchmark（2026-10 核实）

| 配置 | 结果 |
|---|---|
| Qwen3-8B + EAGLE-3，vLLM-Ascend/NPU，100 步 GRPO | co-training 的 mean acceptance length **高于**固定 drafter；vs baseline **rollout 快 ~20%、端到端快 ~11%**，accuracy 无回归 |
| verl PR #5925 静态 EAGLE-3（qwen3-8b + RedHatAI speculator） | rollout 提速 ~15%（无 co-training，仅静态挂载） |

> GPU 上的 verl-SpeCo 数字仓库称后续补（截至 2026-10 benchmark 跑在 Ascend NPU）。

### 8.3 三种 drafter 训练放置的对比（设计空间）

| 维度 | slime / Miles（joint MTP） | **verl-SpeCo**（循环内独立 trainer） | SpecForge（disaggregated） |
|---|---|---|---|
| drafter 训练在哪 | 策略自身 backward | RL GPU 上、策略 update 之后 | 独立 GPU |
| 支持的 drafter | 仅模型自带 MTP 层 | EAGLE/DFlash/DSpark/Domino/P-EAGLE | 多种 |
| 训练数据来源 | 策略 forward 的 hidden states | old-logprob pass 或 rollout 引擎 | live SGLang server 的 prefill hidden states |
| 权重如何到 rollout | 随策略 weight sync 一起 | 随策略 weight sync 一起（publish） | RL 框架需把最新模型灌进 SpecForge server |
| drafter 滞后 | 0 步（与模型同版） | ≤ train_interval（默认 5 步） | 取决于灌模型频率 |
| 额外部件 | 无 | collect/train/publish 三 hook + 版本检查 | Mooncake 对象存储 + 独立 trainer 集群 |

### 8.4 0.1.0 版本（2026-08）新增

1. **DSpark 全流程**：解决"逐 token 训练"与"块预测"的语义错位；GPU 用 Flex Attention 的 block mask，NPU 用等价 dense mask。
2. **独立 drafter 训练工作流**：online 训练 / 仅 feature 收集 / offline 训练三种模式——drafter 可独立于完整 RL job 训练。
3. 算法覆盖扩到 7 个 drafter 家族（DSpark/EAGLE-1/2/3/DFlash/Domino/P-EAGLE），vLLM + SGLang，GPU + Ascend NPU。

### 8.5 联网来源（2026-11-04 核实）

- [verl-SpeCo GitHub](https://github.com/verl-project/verl-SpeCo) / [verl blog 2026-08-13 v0.1.0](https://verl-project.github.io/posts/2026-08-13-verl-speco-release/) / [PyPI verl-speco](https://pypi.org/project/verl-speco/)
- [Vincent Zhong — Keeping Speculative Decoding Fast During RL Training (2026-10-02)](https://vincentzed.com/writing/verl-speco)（最清晰的 SpeCo 原理综述）
- [verl PR #5925 EAGLE/EAGLE3 rollout 加速](https://github.com/verl-project/verl/pull/5925)（静态 spec 在 verl 的工程接入与两个关键 fix）
- [Together AI — distribution-aware speculative decoding](https://www.together.ai/blog/distribution-aware-speculative-decoding)（rollout 占 70% 训练时间）
- [slime](https://github.com/THUDM/slime) / [Miles](https://github.com/radixark/miles)（joint MTP 对比） / [SpecForge](https://github.com/sgl-project/SpecForge)（disaggregated 对比）

---
相关: [[trainer与rollout]] | [[speculative decoding]] | [[投机解码原论文精读]] | [[weight sync mechanism]] | [[stale policy problem]] | [[训推不一致]] | [[sampling throughput]] | [[trainer与rollout引擎组合]] | [[V1 async trainer]] | [[17-RL训推一体框架]]
