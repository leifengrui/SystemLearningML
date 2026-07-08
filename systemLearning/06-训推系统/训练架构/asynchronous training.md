# asynchronous training

> **所属章节**: [[训练架构]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（需懂同步/异步 tradeoff + stale data 修正）

## 1. 一句话定义

**asynchronous training（异步训练）** 是 [[learner worker split]] 的**运行模式**：worker 采样与 learner 训练**并行流水线**，不互相等待——worker 持续采数据进 buffer，learner 持续从 buffer 取训，两者时间重叠。对比**同步训练**（worker 采完 batch 等 learner 训完再采，互相等、利用率低），异步**吞吐趋近 $\max(T_w, T_l)$**（重叠采训时间）。代价是 **stale data**：worker 用 $\theta_t$ 采的数据，learner 已更新到 $\theta_{t+k}$ 才消费，**staleness = $k$ 步**，导致 PPO ratio $\rho=\pi_{\theta_{t+k}}/\pi_{\theta_t}$ 失真（[[stale policy problem]]）。防法：[[weight sync mechanism]] 定期同步新 $\theta$ 给 worker + ratio clip / V-trace 修正。IMPALA（DeepMind 2018）是经典异步架构，RLHF 中 verl 支持同步/异步两种模式。

> [!note] 异步 vs 同步
> - **同步**：worker 采 → 等 learner 训 → 同步新 θ → worker 再采。利用率 < 1（快的等慢的），但无 stale data。
> - **异步**：worker 持续采进 buffer，learner 持续取训。吞吐高（重叠），但 stale data。
> 异步是 throughput 与 staleness 的 tradeoff。

## 2. 为什么需要它（动机与背景）

[[learner worker split]] 的同步模式有利用率问题：
1. **互相等**：worker 采完 batch 等 learner 训完，learner 训完等 worker 采完，GPU 互相闲置；
2. **利用率低**：快的等慢的，吞吐 = $\min(T_w, T_l)$，低于理论上限；
3. **大模型 RL 尤甚**：生成（worker）与训练（learner）耗时都长（秒级），互相等的浪费大；
4. **在线 RL 需持续采**：PPO 每轮需新 rollout，采训并行流水线才能匹配 RL 的数据需求。

异步训练解此：worker 与 learner 各自持续工作，通过 buffer 解耦时序，吞吐趋近 $\max(T_w, T_l)$。代价是 stale data，需修正。这是大模型 RL 提吞吐的关键模式。

## 3. 核心概念详解

### 3.1 同步 vs 异步的时间线

```
同步:
  worker: [采 batch1][等......][采 batch2][等......]
  learner:        [等][训 batch1][等......][训 batch2]
  ↑ 互相等,利用率低

异步:
  worker:  [采 b1][采 b2][采 b3][采 b4][采 b5]...   ← 持续采
  buffer:     ↓    ↓    ↓    ↓    ↓
  learner:    [训 b1][训 b2][训 b3][训 b4]...       ← 持续训
  ↑ 采训重叠,吞吐高;但 b2 用 θ_0 采,learner 训 b2 时已是 θ_1(stale)
```

### 3.2 stale data 与 staleness

- worker 在时刻 $t$ 用 $\theta_t$ 采数据 $D_t$；
- learner 已训练到 $\theta_{t+k}$ 才消费 $D_t$；
- **staleness = $k$ 步**（数据比训练滞后 $k$ 步）；
- $k$ 越大越 stale，ratio $\rho=\pi_{\theta_{t+k}}(a)/\pi_{\theta_t}(a)$ 偏离 1 越多；
- $k$ 太大 → PPO ratio 失真 → 训练不稳（[[stale policy problem]]）。

### 3.3 异步的防 stale 机制

| 机制 | 作用 | 详见 |
|---|---|---|
| **[[weight sync mechanism]]** | 定期把 learner 新 $\theta$ 同步给 worker，减小 $k$ | [[weight sync mechanism]] |
| **PPO ratio clip** | $\rho$ clip 在 $[1-\epsilon,1+\epsilon]$，限失真 | [[PPO clipped objective]] |
| **V-trace** | IMPALA 的多步 return + 截断 ratio，容忍 stale | §3.5 |
| **target_kl 早停** | 每 epoch 算 KL，超 target 则 break | [[KL penalty与KL control]] |
| **buffer 容量限制** | 旧数据过期丢弃，限制最大 $k$ | replay buffer 设计 |
> [!note] 解答：KL 是什么
> **KL = Kullback–Leibler divergence（KL 散度 / 相对熵）**，衡量两个概率分布 $P$ 与 $Q$ 的差异程度，定义：
> $$
> D_{\mathrm{KL}}(P\,\|\,Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
> $$
> （连续情形把求和换成积分。）它**非负**且 $D_{\mathrm{KL}}=0 \iff P=Q$，但**不对称**（$D_{\mathrm{KL}}(P\|Q)\ne D_{\mathrm{KL}}(Q\|P)$），不是真正的"距离"。
>
> ### 在 RL/PPO 里它干什么
> 在 RLHF 的 PPO 中，KL 几乎处处出现，扮演三种角色：
>
> | 角色 | 公式 / 用法 | 作用 |
> |---|---|---|
> | **1. 策略漂移监控** | $\mathrm{KL}=\mathbb{E}_{a\sim\pi_\theta}[\log\pi_\theta(a)-\log\pi_{\theta_{old}}(a)]$ | 衡量当前策略 $\pi_\theta$ 偏离开局策略 $\pi_{\theta_{old}}$ 多少。越大 → 越偏离 → 越容易 reward hacking / 训崩 |
> | **2. target_kl 早停** | 每 epoch 算一次 KL，若 $\mathrm{KL}>\text{target\_kl}$（典型 0.001–0.1）就 `break` 该 batch 的后续 epoch | 防止单次更新把策略改太多——PPO"信赖域"思想的工程兜底 |
> | **3. KL penalty / KL control** | 把 $-\beta\cdot\mathrm{KL}(\pi_\theta\|\pi_{ref})$ 加进 reward 或 loss，自适应调 $\beta$ | 硬约束太死，改用软惩罚让策略不远离参考模型 $\pi_{ref}$（OpenAI InstructGPT 的两段式 KL 控制） |
>
> ### 为什么 PPO 这么在意 KL
> - PPO 的 clip 机制 $\rho=\pi_\theta/\pi_{\theta_{old}}$ 被截断在 $[1-\epsilon,1+\epsilon]$，本质就是**隐式地限制 KL**（信赖域 Trust Region 的近似）。clip 只限单步 ratio，KL 是**全局漂移量**的更稳度量。
> - **异步训练**（本笔记主题）里 worker 用 $\theta_t$ 采、learner 在 $\theta_{t+k}$ 消费，stale data 让 ratio $\rho=\pi_{\theta_{t+k}}/\pi_{\theta_t}$ 偏离 1 → KL 变大。所以异步系统**必须盯 KL**：超 target_kl 立刻停，避免 stale 把策略带飞。这正是 §3.3 表格里 "target_kl 早停" 这一行的来由。
>
> ### 一句话记忆
> **ratio clip 是"单步"的防飞阀，KL 是"累计"的漂移仪表**——PPO 用前者限单次更新、用后者（target_kl 早停 / KL penalty）限整体漂移，异步训练因 stale 让漂移更易发生，故 KL 监控更关键。
>
> 详见 [[KL penalty与KL control]]、[[PPO clipped objective]]、[[KL divergence control]]。
### 3.4 异步的吞吐优势

- **同步**：吞吐 $=\min(T_w, T_l)$，利用率 $<1$；
- **异步**：吞吐 $\to\max(T_w, T_l)$（流水线重叠），利用率高；
- 当 $T_w\approx T_l$（生成与训练耗时近），异步约 2× 同步吞吐；
- 当 $T_w\ll T_l$ 或反之，异步优势小（瓶颈侧决定）。
RLHF 中生成（$T_w$）常是瓶颈且慢，异步让训练与生成重叠，吞吐提升显著。

### 3.5 IMPALA 与 V-trace

IMPALA（Importance Weighted Actor-Learner Architecture）是异步训练的经典：
- 多 actor（worker）并行采 trajectory，单 learner 异步训练；
- **V-trace**：用截断 ratio $\rho_{\bar\theta}=\min(\bar\rho, \pi_\theta/\pi_{\theta_{old}})$ 对 $V$ 做多步 return 估计，容忍一定 staleness；
- V-trace 是异步 stale data 的理论修正，使异步训练稳定。
RLHF 的异步 PPO 借鉴此思想（ratio clip + weight sync）。


> [!note] 解答：业内异步 RL 训练系统最新动态（AReaL / AsyncFlow / Fully-Async，2025–2026）
> 用户提名的三个关键词 `areal` / `asyncflow` / `fullyasync` 正对应当下 LLM-RL 异步训练的三大主流系统/模式。以下逐一拆解，附来源与实测数据。
>
> ## 一、AReaL —— 全异步 RL 系统（蚂蚁 + 清华 IIIS）
> - **论文**：[arXiv:2505.24298](https://arxiv.org/abs/2505.24298)（Wei Fu 等，2025-05），代码 [github.com/inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)。
> - **定位**：完全解耦 generation 与 training 的**全异步** RL 系统，基于 ReaLHF（MLSys 2025）+ SGLang v0.4.6 推理 + Megatron-Core v0.11.0 训练，SLURM 调度。面向大规模**推理模型（LRM）/ agentic** 训练。
> - **核心架构**：4 组件——① **Interruptible Rollout Worker**（可中断生成 worker，`update_weight` 请求会打断进行中的生成，引入"一条 trajectory 由多版本 θ 段拼接"的新算法挑战）；② **Reward Service**（数学题字符串匹配 / 代码题跑 unit test，独立线程与生成重叠）；③ **Trainer Workers**（持续从 replay buffer 采样，凑齐 batch 就 PPO 更新，数据**只用一次**保新鲜）；④ **Rollout Controller**（桥接 rollout/reward/trainer，训完调 `update_weight`）。
> - **算法创新**：
>   1. **Staleness-Aware Training**：超参 $\eta$ 限最大 staleness，给定最新版本 $i$、总轨迹数 $N_r$、batch size $B$，强制约束防过 stale；
>   2. **Decoupled PPO Objective**：解耦 **behavior policy $\pi_{\text{behav}}$**（采样策略）与 **proximal policy $\pi_{\text{prox}}$**（近期正则目标），重要性采样后可容忍**最多 8 步旧**的样本而不掉点。
> - **性能**：v0.3（boba²，2025-06）在数学/代码推理上对同步系统 **2.77× 加速**且精度持平甚至更优；线性扩展到 512 GPU；模型规模到 32B。
> - **生态**：2025-07 发布 **AReaL-lite**（80% 更少代码、90% 性能，算法优先 API，原生支持全异步 agentic RL，集成 CAMEL-AI / OpenAI Agents SDK via HTTP proxy + token-level tracking + reward 回传）。
>
> ## 二、AsyncFlow —— verl-recipe 的异步流式 RL 框架
> - **论文**：[arXiv:2507.01663](https://arxiv.org/abs/2507.01663)（Zhian Han 等，2025-07），代码 [github.com/verl-project/verl-recipe/async_flow](https://github.com/verl-project/verl-recipe/tree/main/async_flow)。
> - **定位**：verl 的 **训推异步流水线 recipe**，用 `TransferQueue(TQ)` + `CheckpointEngine(CE)` 把 Rollout 与 Trainer 解耦，staleness 容忍 + 权重异步同步，提 RLHF/GRPO 吞吐。核心能力与引擎解耦，服务化接口。
> - **5 大核心组件**：
>   1. **AgentLoop Layer**（`AsyncFlowAgentLoopManager`+`AgentLoopWorker`，异步生成层，带 `model_version` 追踪）；
>   2. **TransferQueue (TQ_AsyncFlow)**：专用高吞吐数据中枢，控制面/数据面分离、`topic` 抽象、引用计数去重、动态扩容、列式元数据、端到端零拷贝、多分片并行传输；
>   3. **四个独立 Worker**：`ActorForwardWorker`(old_log_probs/entropys)、`ReferenceForwardWorker`(ref_logprobs)、`RewardAdvWorker`(reward/advantage, GRPO 归一化)、`ActorTrainWorker`(PPO loss + 参数更新)，各跑独立 asyncio 事件循环，`DataDispatchStrategy` 抽象 FSDP/NONE 等并行后端；
>   4. **CheckpointEngine**：HCCL 异步权重广播，ZMQ PUB/SUB 传 tensor 元数据，双缓冲 send/recv 重叠通信，bucketed 传输批量小 tensor；
>   5. **FlowControlQueue**：staleness 控制队列，`max_capacity=(staleness+1)×batch_size` 限 in-flight 请求数，`staleness=0` 退化严格 on-policy。
> - **实测（GRPO, prompt 2k→response 16k, Ascend NPU）**：
>
>   | 指标 | verl On-Policy | AsyncFlow | Fully Async |
>   |---|---|---|---|
>   | NPU 切分 | 64 | 50-2-2-8 | 48-16 |
>   | throughput | 59.3 | **226.8** | 149.66 |
>   | 加速 | / | **3.81×** | 2.52× |
>
>   精度：`staleness ≤ 2` 时与同步对齐甚至微幅提升。注：CheckpointEngine 当前需 Ascend NPU + HCCL。
>
> ## 三、Fully-Async（verl `fully_async_policy`）—— verl 主仓的全异步 PPO 模式
> - **文档**：[verl docs/advance/fully_async.md](https://verl.readthedocs.io/en/latest/advance/fully_async.html)（2026-05 更新），[github volcengine/verl](https://github.com/volcengine/verl/blob/main/docs/advance/fully_async.md)。
> - **定位**：verl 主仓自己实现的**完全解耦 Trainer 与 Rollouter** 的全异步 PPO，借鉴 AReaL / Magistral / StreamRL / AsyncFlow 思路。4 部分：Rollouter / MessageQueue / Trainer / ParameterSynchronizer。
> - **4 种工作模式**（调参组合）：
>   1. **on-policy pipeline**：严格同步，小规模稳定优先；
>   2. **stream off-policy**（`trigger_parameter_sync_step>1, staleness_threshold=0`）：流式同步，一次产更多样本降空闲；
>   3. **async stream with stale samples**（`staleness_threshold>0, partial_rollout=False`）：容忍 stale，除首批外训练不等首批 rollout；
>   4. **async stream with partial rollout**（`partial_rollout=True`）：sync 时打断进行中生成，sync 后续生成，进一步减等待。
> - **关键参数**：`async_training.require_batches`、`trigger_parameter_sync_step`、`staleness_threshold`（新鲜度控制）、`partial_rollout`。
> - **实测**：Qwen2.5-Math-7B 长候选多资源，`async stream with stale samples` 在 32/64/128 卡约 **2× 提升**且不掉点；流式≈1.6×，叠加 staleness+partial_rollout 达 **2.35×**；Qwen3-30B-A3B-Base 上 `staleness samples` 模式 **1.7×**。另支持 `AsyncPartialToolAgentLoop`（多轮 tool calling + partial_rollout）。
>
> ## 四、其他同期系统（背景，便于横向定位）
> - **StreamRL**（[arXiv:2504.15930](https://arxiv.org/abs/2504.15930)）： disaggregated stream generation，异构弹性 RL；
> - **Magistral**（[arXiv:2506.10910](https://arxiv.org/abs/2506.10910)）：流式训练；
> - **StaleFlow**（[arXiv:2601.12784](https://arxiv.org/abs/2601.12784)，2026-01）：统一治理 **data staleness + data skewness**（轨迹长度变异致负载不均），全局一致性协议 + 数据服务器，比 SOTA 高 1.42–2.68× 吞吐；
> - **Laminar**（EuroSys 2026，[doi:10.1145/3767295.3803580](https://dl.acm.org/doi/10.1145/3767295.3803580)）：全解耦架构 + 自适应放置；
> - **Applied Compute**（2026-07）：预测与控制全异步 RL 训练的 staleness。
>
> ## 五、趋势小结
> 1. **全异步解耦成主流**：AReaL/AsyncFlow/verl fully_async/StaleFlow/Laminar 都走"rollout/reward/train 分离到独立资源 + 异步执行"，彻底消除 reshard 开销与互相等待；
> 2. **staleness 从"被动容忍"到"主动治理"**：AReaL 的 decoupled PPO（8 步旧不掉点）、StaleFlow 的全局一致性协议、verl 的 `staleness_threshold` + `partial_rollout`，都在把 stale 从代价变成可调旋钮；
> 3. **数据层新组件**：TransferQueue（topic/零拷贝/引用计数）、MessageQueue、数据服务器等"异步数据中枢"成为系统核心；
> 4. **权重同步工程化**：HCCL/ZMQ 双缓冲/bucketed/打断式 partial rollout，把 weight sync 开销压到临界路径外；
> 5. **agentic RL 友好**：AReaL-lite 的 HTTP proxy + token-level tracking + reward 回传、verl 的 `AsyncPartialToolAgentLoop`，让多轮工具调用与异步训练原生结合。
>
> **与本笔记的对应**：§3.3 的防 stale 机制（weight sync / ratio clip / V-trace / target_kl）在上述系统里都已工业化落地——AReaL 的 decoupled PPO 是 ratio 修正的进化版，verl `staleness_threshold` 是 §4.3 的工程化，`partial_rollout` 是 §8.3 "按需/流式同步"的打断式实现。详见 [[stale policy problem]]、[[weight sync mechanism]]。
## 4. 数学原理 / 公式

### 4.1 吞吐对比

记 worker 采样吞吐 $T_w$（trajectory/秒），learner 训练吞吐 $T_l$（batch/秒）。

- **同步**：每轮时间 $= 1/T_w + 1/T_l$，吞吐 $=\frac{1}{1/T_w+1/T_l}=\frac{T_w T_l}{T_w+T_l}$（调和平均，$<\min$）；
- **异步**：吞吐 $\to\max(T_w, T_l)$（瓶颈侧决定，另一侧重叠）。

例：$T_w=T_l=1$，同步吞吐 $=0.5$，异步 $\to 1$（2× 提升）。

### 4.2 staleness 与 ratio 失真

worker 用 $\theta_t$ 采，learner 在 $\theta_{t+k}$ 消费。PPO ratio：
$$
\rho=\frac{\pi_{\theta_{t+k}}(a|s)}{\pi_{\theta_t}(a|s)}
$$

$k=0$（同步）：$\rho=1$。$k$ 大：$\rho$ 偏离 1，clip $[1-\epsilon,1+\epsilon]$ 截断。$k$ 过大 → clip 频率高 → 有效梯度信号弱 → 训练慢/不稳。

### 4.3 weight sync 的 staleness 控制

[[weight sync mechanism]] 每 $S$ 步同步 $\theta$ 给 worker。worker 的最大 staleness $\le S$（在两次 sync 间采的数据）。$S$ 小 → stale 轻，但 sync 开销大（$\theta$ 大）；$S$ 大 → stale 重。tradeoff。

### 4.4 V-trace 的截断

IMPALA V-trace target：
$$
v_s=V_\theta(s_s)+\sum_{t=s}^{s+n-1}\gamma^{t-s}\rho_{\bar\theta}^{(t)}\bigl(r_t+\gamma V_\theta(s_{t+1})-V_\theta(s_t)\bigr)
$$

其中 $\rho_{\bar\theta}^{(t)}=\min(\bar\rho, \pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t))$ 截断 ratio，$\bar\rho\approx1$。容忍 stale 的多步 return。

## 5. 代码示例

```python
import torch, threading, queue, time

class Policy:
    def __init__(self):
        self.theta = torch.zeros(4)
        self.theta_version = 0           # θ 版本号(追踪 staleness)
    def generate(self, n):
        return torch.randn(n, 4) * self.theta
    def train_step(self, batch):
        self.theta += 0.01 * batch.mean(0)
        self.theta_version += 1

policy = Policy()
buffer = queue.Queue(maxsize=20)

def worker():
    """异步 worker:持续采,记录采时的 θ 版本"""
    while True:
        traj = policy.generate(8)
        data_version = policy.theta_version   # 采时的 θ 版本
        buffer.put((traj, data_version))
        time.sleep(0.1)

def learner():
    """异步 learner:持续训,算 staleness"""
    while True:
        traj, data_version = buffer.get()
        staleness = policy.theta_version - data_version   # k 步
        policy.train_step(traj)
        if staleness > 0:
            print(f"  stale data: k={staleness}(用 θ_{data_version} 采,θ_{policy.theta_version} 训)")
        time.sleep(0.15)

# 启动异步训练
threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=learner, daemon=True).start()
time.sleep(2)
print(f"异步训练后 θ = {policy.theta}, version = {policy.theta_version}")
print("异步:采训并行,吞吐高,但有 stale data(需 weight sync + ratio clip 修正)")
```

> [!tip] 实际工程（verl）
> verl 支持同步/异步两种模式：
> - **同步**：rollout worker 采完 → learner 训 → weight sync → 再采。简单稳定，吞吐低。
> - **异步**：rollout worker 持续采进 Ray object store，learner 持续取训，weight sync 定期广播新 θ。吞吐高，需 ratio clip + target_kl 修 stale。
> 大模型 RLHF 常用同步（稳定优先），异步在高吞吐场景用。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[learner worker split]]（异步是 split 的运行模式）、[[PPO optimization]]（PPO 的 ratio clip 修正 stale）、[[replay buffer]]（异步的数据管道）。
- **下游（应用）**: [[stale policy problem]]（异步的代价）、[[weight sync mechanism]]（防 stale）、IMPALA/V-trace、verl/OpenRLHF 的异步模式。
- **对比 / 易混**:
  - **异步 vs 同步训练**：见 §3.1。吞吐 vs 稳定 tradeoff。
  - **asynchronous training vs [[learner worker split]]**：前者是**运行模式**（并行流水线），后者是**架构**（采训分离）。split 可同步可异步。
  - **异步 vs 分布式数据并行（DDP）**：DDP 是同步数据并行（梯度 all-reduce 后同步更新），异步训练是采训流水线。不同层次。
  - **stale data vs [[stale policy problem]]**：前者是现象（数据滞后），后者是后果（ratio 失真训练不稳）。前者引出后者。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"异步一定比同步快"**：当 $T_w\approx T_l$ 时约 2×，但若 stale 修正开销大（clip 频繁/重采），净收益可能减。且瓶颈侧决定上限。
> 2. **"异步无代价"**：代价是 stale data → [[stale policy problem]]，需 weight sync + ratio clip 修正。
> 3. **"staleness k 越小越好"**：k 小 stale 轻，但需频繁 weight sync（$\theta$ 大，sync 慢），tradeoff。
> 4. **"异步 = 并发多线程"**：不只是线程，是多进程/多设备的采训流水线，用 Ray/分布式。
> 5. **"PPO 不能异步"**：可以，靠 ratio clip 修正 stale（off-policy 修正）。IMPALA 的 V-trace 是理论保证。
> 6. **"异步训练 = 异步梯度更新"**：不同。前者是采训流水线（数据层异步），后者是参数服务器异步梯度（PS 架构）。不同概念。
> 7. **"buffer 越大异步越好"**：buffer 大平滑波峰，但旧数据堆积 stale 重。需容量限制 + 过期丢弃。
> 8. **"RLHF 用异步"**：多数 RLHF 实践用同步（稳定优先），异步在高吞吐场景（如大规模在线 RLHF）才用。

## 8. 延伸细节

### 8.1 同步 vs 异步 PPO 的选择

| 场景 | 推荐 |
|---|---|
| 小规模、稳定优先 | 同步 PPO |
| 大规模在线 RLHF、吞吐瓶颈 | 异步 PPO + weight sync |
| 生成与训练耗时近 | 异步收益大（2×） |
| 生成远慢于训练 | 异步让训练与生成重叠，收益大 |
| 生成远快于训练 | 异步收益小（训练瓶颈） |

RLHF 中生成（rollout）常远慢于训练，异步收益大，但工程复杂度高，故开源框架默认同步，异步是高级选项。

### 8.2 IMPALA 的遗产

IMPALA 是异步训练的里程碑（DeepMind 2018）：
- 多 actor 并行采 + 单 learner 异步训；
- V-trace 修正 stale；
- throughput 大超同步 A3C。
其思想（采训分离 + 异步 + ratio 修正）被 RLHF 框架（verl/OpenRLHF）继承，只是 actor 换成 LLM rollout worker。

### 8.3 异步的 weight sync 策略

- **周期同步**：每 $S$ 步广播新 $\theta$，简单，$S$ 需调；
- **按需同步**：staleness 超 threshold 才 sync，自适应；
- **增量同步**：只传 $\theta$ 的差分（$\Delta\theta$），减带宽（但大模型 $\Delta$ 仍大）；
- **流式同步**：后台流式推 $\theta$，worker 渐进更新。
详见 [[weight sync mechanism]]。

### 8.4 异步与 stale policy problem 的深度关联

异步是 stale policy problem 的**主要成因**（同步无 stale）。stale 的修正：
- ratio clip（PPO）：限 ratio 失真；
- V-trace（IMPALA）：多步 return + 截断；
- weight sync：减小 $k$；
- target_kl 早停：监控 stale 引起的 KL 漂移。
四者协同，详见 [[stale policy problem]]。

### 8.5 异步训练的监控

- **staleness 分布**：buffer 中数据的 $k$ 分布，均值/最大值；
- **clip frequency**：ratio 触 clip 的比例，stale 重则高；
- **weight sync 延迟**：同步耗时，影响 $k$；
- **buffer 积压**：worker 产速度 vs learner 消费速度，积压说明 learner 慢。
这些是异步训练的诊断仪表盘。

---
相关: [[训练架构]]、[[learner worker split]]、[[rollout worker]]、[[inference worker]]、[[stale policy problem]]、[[weight sync mechanism]]、[[PPO optimization]]、[[PPO clipped objective]]、[[KL penalty与KL control]]、[[replay buffer]]、[[Ray基础]]
