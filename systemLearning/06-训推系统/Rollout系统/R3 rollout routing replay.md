# R3 rollout routing replay

> **所属章节**: [[Rollout系统]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中高级

## 1. 一句话定义

**R3（Rollout-Routing-Replay，rollout 路由与回放）** 是 RL/RLHF 训练系统中 **rollout 子系统的三段数据通路**：**Rollout**（用当前 policy 对 prompt 生成 response 并算 reward，本质是推理）→ **Routing**（把 prompt 集合分发到多个分布式 rollout worker，做负载均衡与权重版本一致）→ **Replay**（把生成的经验存入 buffer，供 learner 做 off-policy 多次复用）。三者串成"**生成 → 分发 → 回放 → 训练 → 同步权重回 rollout**"的闭环，是决定 RL 训练**吞吐**与**样本效率**的核心机制。

> [!note] 关于 "R3" 命名
> "R3" = 三个 R（Rollout / Routing / Replay）的助记缩写，并非某个广为人知的标准术语，本笔记按这三段机制组织。不同框架（verl / SGLang / Ray RLlib）实现里可能用不同名字称呼同一组机制。

## 2. 为什么需要它（动机与背景）

RL 训练（尤其 PPO/RLHF）的算力瓶颈在 **rollout**：每一轮 policy 更新前，都要**在线**用当前权重对大量 prompt 跑推理生成 response、算 reward，得到一批经验。这比单纯的训练 forward 慢得多（生成是逐词自回归，且要跑完整序列）。瓶颈表现为：

1. **生成慢**：rollout 是逐词 decode，memory-bound，单卡吞吐低 → 要**分布式**铺开多个 rollout worker 并发生成。
2. **分发难**：prompt 长度/生成长度差异大，worker 负载不均；且异步训练下 worker 的权重版本可能与 learner 不同步（stale）→ 需要**路由**策略做负载均衡 + 权重版本管理。
3. **样本贵**：每条经验都要花一次完整 rollout 才能得到，丢掉太可惜；off-policy 方法允许把经验存起来多次复用 → 需要 **replay** 机制提样本效率、降 rollout 频次。

R3 就是把这三件事拧成一根流水线：**rollout 产出经验、routing 高效分发、replay 复用经验**，让 rollout 子系统的吞吐与样本效率都跟上 learner。

## 3. 核心概念详解

### 3.1 Rollout（轨迹生成）

用当前 policy $\pi_\theta$ 对一批 prompt $x$ 生成 response $y$、算 reward $r$，得到一条轨迹 $(x, y, r)$ 及每步的 logprob、value 等。本质是**推理**，用 vLLM/SGLang/TensorRT-LLM 等推理引擎跑（见 [[trajectory generation]]、[[rollout worker]]）。

- 关键：rollout 用的权重 $\theta_{\text{rollout}}$ 可能落后于 learner 的 $\theta_{\text{current}}$（异步），差多少决定 staleness。
- rollout 是 RL 训练里**最贵**的一步，通常占训练总时长 50%–80%。

### 3.2 Routing（请求/数据路由）

把 prompt 集合分发到 $K$ 个 rollout worker 的策略。核心目标：**负载均衡** + **权重版本一致** + **亲和性**。

| 策略 | 思路 | 优点 | 缺点 |
|---|---|---|---|
| **round-robin** | 轮询发 | 实现极简 | 不考虑 worker 实际负载与请求长度 |
| **least-loaded** | 发给当前队列最短/算力最闲的 worker | 负载均衡好 | 要实时感知 worker 状态 |
| **affinity-based** | 同一对话/同 prefix 粘同一 worker | 复用 [[prefix caching]]/KV cache | 可能负载倾斜 |
| **weight-version-aware** | 只把请求发给权重版本最新的 worker | 防 stale | 升级时旧 worker 闲置 |

要点：

- **负载均衡**：长 prompt + 长 generate 的请求会占住 worker 很久，routing 要动态感知（类似 [[continuous batching]] 的 backpressure），否则慢 worker 积压、快 worker 空转。
- **权重版本一致**：异步训练下，rollout worker 的权重在两次 [[weight sync mechanism|weight sync]] 之间是旧版 $\theta_{\text{old}}$，用它生成的经验是 off-policy，送给 learner 时要做**重要性采样修正**（见 §4）。routing 记录每条经验对应的权重版本号，供 learner 校正。
- **权重同步**：learner 更新后，新权重 $\theta_{\text{new}}$ 要同步到所有 rollout worker（见 [[weight sync mechanism]]、[[stale policy problem]]）。同步方式：整权重广播（同步训练）或增量/分块拉取（异步训练）。

### 3.3 Replay（经验回放/轨迹重放）

把 rollout 产出的经验 $(x, y, r, \text{logprob}, \text{value}, \theta_{\text{version}})$ 存入 **replay buffer**，供 learner 多次采样复用。

| 方法 | 复用粒度 | 代表 | staleness 处理 |
|---|---|---|---|
| **PPO（准 on-policy）** | 同批经验做 K 个 epoch，不跨 rollout 步 | PPO/RLHF 主流 | K 小（2–4），staleness 轻 |
| **off-policy + 重要性采样** | 大 buffer，跨多步复用 | IMPALA/V-trace、A3C | 用 $\rho=\pi_\theta/\pi_{\theta_{\text{old}}}$ 修正 |
| **prioritized experience replay (PER)** | 按 TD 误差优先采样 | PER、SAC | 优先回放"意外"样本 |

要点：

- **PPO 的 buffer**：不是"无 replay"，而是**短期 buffer**——存当前 rollout 一批经验，做 K 个 epoch 的 minibatch SGD，然后丢弃换下一批。本质是"批内复用"，不是跨步 replay。
- **严格 off-policy replay**：大 buffer 存历史经验，跨多轮 rollout 复用，rollout 频次降、样本效率升，但要靠**重要性采样比** $\rho=\pi_\theta(a)/\pi_{\theta_{\text{old}}}(a)$ 修正分布偏移；$\rho$ 过大时裁断（PPO 的 clip 就是这个思想）。
- **buffer 大小权衡**：太大 → 经验太旧、$\rho$ 失控、训练不稳；太小 → 复用不够、rollout 成本降不下来。RLHF 里通常偏小（近 on-policy）。
- **prioritized**：按 TD 误差/reward 惊喜度优先采样，提升样本价值，但引入采样偏倚需修正。

### 3.4 三者闭环

```
        ┌─────────────────────────────────────────────────────────┐
        │                                                           │
        ▼                                                           │
  ┌───────────┐  routing   ┌──────────────┐  experience  ┌─────────┴───┐
  │  prompt   │ ─────────→ │ rollout pool │ ───────────→ │ replay buf  │
  │  queue    │            │ (K workers)  │              │             │
  └───────────┘            └──────────────┘              └──────┬──────┘
        ▲                          ▲                             │ sample
        │                          │ weight sync                 ▼
        │                          │                      ┌─────────────┐
        │                          └────────────────────  │   learner   │
        │                                                 │ (PPO/SAC…)  │
        └────────────────── new prompt ──────────────────┤  θ update   │
                                                          └─────────────┘
```

- prompt 经 routing 分发到 rollout worker；
- rollout 生成经验，存 replay buffer；
- learner 从 buffer 采样、算 loss、更新 θ；
- 新 θ 经 weight sync 推到 rollout worker，进入下一轮。

## 4. 数学原理 / 公式

### rollout 吞吐

设 $K$ 个 worker、单 worker 生成吞吐 $T_{\text{w}}$（tokens/s），总吞吐：

$$
T_{\text{rollout}} = K \cdot T_{\text{w}}
$$

训练一轮需要 $N$ 条经验、每条平均长 $L$，rollout 耗时 $\approx N L / T_{\text{rollout}}$。R3 的目标就是把这个时间压低（routing 提并行度、replay 降 $N$ 的复用需求）。

### 样本效率（replay 复用）

设每条经验被复用 $C$ 次（PPO 的 K epoch 即 $C=K$），等效样本量：

$$
N_{\text{eff}} = N \cdot C
$$

rollout 成本 $\propto N$，训练有效样本 $\propto N_{\text{eff}}$，replay 把"1 次生成"摊成"$C$ 次训练样本"，是 rollout 成本与样本量的解耦。

### 重要性采样修正（off-policy）

rollout 用 $\theta_{\text{old}}$ 生成、learner 用 $\theta$ 训练时，分布偏移用重要性采样比修正：

$$
\rho = \frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}, \qquad \mathcal{L} = \mathbb{E}_{\text{buffer}}\left[\rho \cdot A\right]
$$

PPO 裁断 $\rho\in[1-\epsilon, 1+\epsilon]$（见 [[PPO clipped objective]]）防方差爆炸。staleness 越大 $\rho$ 越发散 → 训练不稳，这是 replay 与异步训练的核心代价。

### staleness

$$
\Delta\theta = \|\theta_{\text{current}} - \theta_{\text{rollout}}\|
$$

$\Delta\theta$ 越大，经验越 off-policy、$\rho$ 越失控。weight sync 频率与 replay buffer 大小共同决定 $\Delta\theta$ 的上界。

## 5. 代码示例（可选）

```python
import collections
import numpy as np

class ReplayBuffer:
    """最简 FIFO replay buffer（存经验，供 learner 采样）。"""
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def push(self, exp):
        self.buf.append(exp)                # exp: dict(prompt, response, reward, logprob, theta_ver)
    def sample(self, batch):
        idx = np.random.randint(0, len(self.buf), batch)
        return [self.buf[i] for i in idx]
    def __len__(self):
        return len(self.buf)

class Router:
    """least-loaded 路由：把 prompt 发给当前队列最短的 worker。"""
    def __init__(self, workers):
        self.workers = workers              # list of RolloutWorker
    def dispatch(self, prompt):
        w = min(self.workers, key=lambda w: w.queue_len())   # 最闲的
        w.submit(prompt)
        return w.id

class RolloutWorker:
    """持有 policy 权重，对 prompt 跑推理生成 response。"""
    def __init__(self, wid, engine, theta_ver):
        self.id, self.engine, self.theta_ver = wid, engine, theta_ver
        self.q = []
    def submit(self, prompt):
        self.q.append(prompt)
    def queue_len(self):
        return len(self.q)
    def step(self):
        if not self.q: return None
        p = self.q.pop(0)
        resp, logp = self.engine.generate(p)            # 用 θ_rollout 生成
        return dict(prompt=p, response=resp, logprob=logp, theta_ver=self.theta_ver)

# 训练循环（伪代码）
buffer = ReplayBuffer(capacity=10000)
router = Router([RolloutWorker(i, engine, theta_ver=0) for i in range(8)])

for train_iter in range(1000):
    for p in fetch_prompts(N=256):
        router.dispatch(p)                               # routing
    exps = [w.step() for w in router.workers]            # rollout
    for e in exps: buffer.push(e)                        # replay 入 buffer
    batch = buffer.sample(64)                           # replay 出 buffer
    loss = ppo_loss(batch, current_theta)               # 含重要性采样修正
    learner_step(loss)
    if train_iter % sync_every == 0:
        sync_weights_to(router.workers)                  # weight sync
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[trajectory generation]]（rollout 本身）、[[rollout worker]]（worker 抽象）、[[sampling throughput]]（单 worker 吞吐）、[[batching strategy]]（worker 内批处理）、推理引擎（vLLM/SGLang，rollout 用推理跑）。
- **下游（应用）**: PPO/RLHF 训练效率（[[PPO clipped objective]]、[[GAE]]）、异步训练（[[asynchronous training]]、[[stale policy problem]]、[[weight sync mechanism]]）、大规模 RL 训练系统（verl/OpenRLHF/Ray RLlib）。
- **对比 / 易混**:
  - **experience replay vs [[gradient checkpointing]]**：前者存"经验"（轨迹）供训练复用，后者存"前向激活"供反向重算省显存——名字都带"replay/checkpoint"但领域不同（RL vs 训练显存优化）。
  - **routing（rollout 分发）vs [[continuous batching]]**：前者是 worker 间分发请求，后者是单 worker 内拼批，两层正交。
  - **replay buffer vs [[prefix caching]]**：前者存 RL 经验轨迹，后者存推理 KV cache 前缀，不同层。
  - **R3 的 routing vs [[PD分离]] 的 routing**：前者路由 prompt 到 rollout worker（训练侧），后者路由请求到 prefill/decode 实例（推理服务侧），思路相似但场景不同。
  - **R3 的 weight sync vs [[policy deployment (PD)|训推分离 weight sync]]**：前者训练中 learner→rollout worker 的训练权重同步，后者训练→推理部署的权重同步，机制类似、周期与精度要求不同。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 PPO 是纯 on-policy 不用 replay**：PPO 仍有 buffer——存当前 rollout 一批，做 K（2–4）个 epoch minibatch 复用，是"批内 replay"。真正无 replay 的是 REINFORCE（每条只用一次）。
> 2. **routing 只看负载不看权重版本**：异步训练下，负载最闲的 worker 可能权重最旧，发给它生成的是最 off-policy 的经验，staleness 失控。要**weight-version-aware routing**。
> 3. **replay buffer 越大越好**：越大复用越多，但经验越旧、$\rho$ 越发散、训练不稳；RLHF 偏 on-policy 通常用小 buffer。
> 4. **把 experience replay 和 gradient checkpointing 混了**：都带"replay/checkpoint"，前者复用经验轨迹，后者重算前向激活，完全不同层（见 §6）。
> 5. **rollout worker 用旧权重生成的经验直接当 on-policy 训**：异步下必须做重要性采样修正（$\rho$），否则梯度有偏、训练发散（见 [[stale policy problem]]）。
> 6. **weight sync 整权重广播太重**：大模型整权重广播带宽爆，要分块/增量同步，或用参数服务器/ZeRO 分片拉取。
> 7. **routing 不复用 prefix caching**：同一 prompt 前缀粘同一 worker 能复用 KV cache（见 [[prefix caching]]），routing 策略应考虑亲和性。

## 8. 延伸细节

### 8.1 代表实现

- **verl**（字面源 Volcano Engine RL）：Rollout-Routing-Replay 一体化引擎，rollout 用 vLLM/SGLang，支持异步 + 重要性采样修正，是开源 RLHF 训练系统的代表。
- **OpenRLHF / ColossalAI-Chat**：类似分布式 rollout + buffer 设计。
- **Ray RLlib**：通用 RL 框架，rollout worker（actor）+ replay buffer（可选）+ learner 分离，routing 由 Ray 的 actor 调度承担。
- **IMPALA/V-trace**：严格的 off-policy 异步架构，replay + V-trace 修正 $\rho$，是 R3 思想的早期系统化实现。

### 8.2 prioritized experience replay (PER)

按 TD 误差/reward 惊喜度给经验打优先级，优先回放"意外"样本，提升样本价值。代价：引入采样偏倚，需用重要性权重修正。在 RLHF 里 reward 稀疏，PER 价值有限，多用于 dense reward 的 RL。

### 8.3 异步训练下的 R3

异步训练（[[asynchronous training]]）里 rollout 与 learner 解耦：learner 持续更新 θ，rollout worker 周期性 sync。R3 三段都在异步流水线里：
- routing 要 weight-version-aware；
- replay buffer 是异步的数据桥梁（learner 不等 rollout，从 buffer 取最近的）；
- weight sync 增量化、分块化降开销。
代价：staleness 引入的 $\rho$ 方差，靠 clip/V-trace 控制。

### 8.4 与推理系统的复用

rollout 本质是推理，R3 的 routing/replay 思想与推理服务系统的 continuous batching、prefix caching、PD 分离高度同构——很多 RL 训练系统（verl）直接复用 vLLM/SGLang 作为 rollout engine，把推理优化成果反哺到训练。这也是为什么 rollout 子系统是"训推融合"的关键结合部（见 [[06-训推系统]] 概述）。

### 8.5 趋势

- **rollout engine 推理化**：rollout 全面用推理引擎（vLLM/SGLang）跑，复用 continuous batching、prefix caching、PD 分离等推理优化。
- **off-policy 化**：为降 rollout 成本，方法向更 off-policy 演进（更激进的 replay + 更强的 $\rho$ 修正，如 GRPO、DAPO 等 RLHF 算法），R3 的 replay 段越来越重要。
- **rollout 与 learner 的资源 disaggregation**：类似推理的 PD 分离，把 rollout pool 与 learner pool 物理分离、独立扩缩容，是大规模 RL 训练系统的方向。

---
相关: [[Rollout系统]]、[[trajectory generation]]、[[rollout worker]]、[[sampling throughput]]、[[batching strategy]]、[[asynchronous training]]、[[stale policy problem]]、[[weight sync mechanism]]、[[PPO clipped objective]]、[[prefix caching]]、[[PD分离]]
