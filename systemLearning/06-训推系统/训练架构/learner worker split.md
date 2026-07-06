# learner worker split

> **所属章节**: [[训练架构]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（架构概念，需懂 RL 训练流程与大模型工程）

## 1. 一句话定义

**learner / worker split（训练器/采样器分离）** 是训推系统的基础架构：把**采样数据**（worker / actor / rollout worker）与**训练更新**（learner / trainer）解耦到**不同进程/设备**，通过 **buffer / replay** 传递数据。worker 负责与环境交互或自回归生成采数据，learner 跑 forward/backward/optimizer step 做梯度更新。对比传统 DL 的"worker 既采又训"同步模式，split 让**生成与训练可异步并行**、**资源异构**（生成用推理引擎、训练用训推框架）、**吞吐提升**，是大模型 RL（尤其 [[RLHF (PPO)]] 的 [[PPO optimization]] §8.3）与在线学习的工程基础。经典实现是 IMPALA（Espeholt et al. 2018），RLHF 中 verl/OpenRLHF 沿用此架构。

> [!note] 别名
> learner/worker split = actor-learner architecture = sampler-trainer split = producer-consumer 训练架构。都指"采样的不训，训的不采样"。

## 2. 为什么需要它（动机与背景）

传统 DL 训练（监督学习）的数据是**静态数据集**，worker（DDP 数据并行）只做 forward/backward，无需"采样"。但 RL 训练的数据是**动态采的**（rollout 生成），引入新矛盾：
1. **生成与训练耗时不对称**：大模型 RL 的 rollout 自回归生成慢（秒~分钟级），训练 forward/backward 也重（秒级），若**同进程同 GPU 交替**做生成+训练，GPU 互相闲置（生成时训练等，训练时生成等）→ 利用率低；
2. **框架异构**：生成用推理引擎（vLLM/SGLang，优化 [[KV cache management]]/连续批处理），训练用训推框架（DeepSpeed/FSDP/ZeRO，优化 [[activation memory]]/通信），两者 API 与优化点不同，同进程难兼容；
3. **资源异构**：生成吃显存（KV cache）少但延迟敏感，训练吃显存（activation+optimizer state）大但可切分，两者 GPU 配置最优不同；
4. **在线 RL 需持续采新数据**：PPO 每轮需新 rollout（$\pi_\theta$ 更新后旧数据 stale），采与训需并行流水线提吞吐；
5. **分布式扩展**：多 worker 并行采样（提数据吞吐），单/多 learner 聚合训练（提算力），split 让两者独立扩展。

故 learner/worker split 是大模型 RL 的工程必然——把不同性质的工作负载解耦到专责进程，各自最优配置，通过 buffer 流水线协作。

## 3. 核心概念详解

### 3.1 learner 与 worker 的角色分工

| 角色 | 职责 | 框架/优化点 | 典型实现 |
|---|---|---|---|
| **worker（采样器）** | 用当前 policy $\pi_\theta$ 与环境交互/自回归生成，采 trajectory 进 buffer | 推理引擎（vLLM/SGLang），优化生成吞吐/KV cache | rollout worker, inference worker |
| **learner（训练器）** | 从 buffer 取数据，跑 forward/backward/optimizer step 更新 $\theta$ | 训推框架（DeepSpeed/FSDP），优化显存/通信 | trainer, PPO learner |
| **buffer/replay** | worker→learner 的数据管道，存 trajectory（state/action/reward/logp 等） | 共享内存/Redis/Ray object store | replay buffer, trajectory buffer |

```
worker(采样)  ──trajectory──>  buffer  ──batch──>  learner(训练)
   ↑                                                      │
   └────────────── weight sync (新 θ) ────────────────────┘
```

### 3.2 与传统 DL worker 的对比

| 维度 | 传统 DL worker（DDP） | learner/worker split |
|---|---|---|
| 数据来源 | 静态数据集 | 动态采样（rollout） |
| worker 职责 | forward/backward（既采又训） | 只采样，不训 |
| learner 职责 | = worker（同进程） | 独立进程，只训 |
| 同步模式 | 同步（all-reduce 梯度） | 可同步可异步 |
| 框架 | 单一（PyTorch DDP） | 异构（推理引擎 + 训推框架） |
| 适用 | 监督学习 | RL / 在线学习 |

### 3.3 同步 vs 异步 split

| 模式 | 流程 | 优劣 |
|---|---|---|
| **同步 split** | worker 采完 batch → 等 learner 训完 → learner 同步新 θ → worker 再采 | 简单、无 stale data，但 GPU 互相等、利用率低 |
| **异步 split** | worker 持续采进 buffer，learner 持续从 buffer 取训，两者并行 | 吞吐高，但有 stale data（worker 用旧 θ 采的数据，learner 已更新 N 步） |

异步的 stale data 是 [[stale policy problem]] 的根源，需 [[weight sync mechanism]] 定期同步新 θ 给 worker。详见 [[asynchronous training]]。

### 3.4 buffer 的作用

- **解耦**：worker 与 learner 不直接交互，通过 buffer 传递，时序解耦；
- **批量化**：worker 逐条采，learner 取 batch 训，buffer 做汇集；
- **复用**：同一 trajectory 可被 learner 多次采（off-policy，如 PPO 的 K epoch 复用）；
- **异步缓冲**：worker 产速度快于 learner 消费时 buffer 积压，反之空等，buffer 平滑波峰。

### 3.5 split 在 RLHF 中的具体形态

[[PPO optimization]] §8.3 的 RLHF-PPO 用 split：
- **rollout worker**：跑 actor $\pi_\theta$ 自回归生成 response（用 vLLM/SGLang）；
- **reward worker**（可选）：跑冻结 RM/reference 算 reward 与 KL（推理引擎）；
- **learner**：跑 actor + critic 的 forward/backward/optimizer step（FSDP/DeepSpeed）；
- **buffer**：存 (prompt, response, logp_old, logp_ref, reward) 等；
- **weight sync**：learner 更新后把新 actor θ 同步给 rollout worker（[[weight sync mechanism]]）。
verl/OpenRLHF 是此架构的代表，用 Ray 调度多 worker 与 learner。

## 4. 数学原理 / 公式

### 4.1 吞吐模型

记 worker 采样吞吐 $T_w$（trajectory/秒），learner 训练吞吐 $T_l$（batch/秒）。

- **同步 split**：每轮时间 $= \max(1/T_w, 1/T_l)$（互相等），吞吐 $=\min(T_w, T_l)$，**利用率 < 1**（快的等慢的）；
- **异步 split**：吞吐趋近 $\max(T_w, T_l)$（流水线），利用率高，但需 buffer 容量够大平滑波峰。

异步的吞吐提升来自**重叠**采样与训练的时间。

### 4.2 stale data 的 lag

异步 split 中，worker 在时刻 $t$ 用 $\theta_t$ 采数据 $D_t$，learner 已训练到 $\theta_{t+k}$ 才消费 $D_t$。**staleness = $k$ 步**。

PPO 的 importance sampling ratio $\rho=\pi_{\theta_{t+k}}(a)/\pi_{\theta_t}(a)$ 随 $k$ 增大偏离 1，clip 限制此偏离。$k$ 太大 → ratio 失真 → 训练不稳（[[stale policy problem]]）。故需 [[weight sync mechanism]] 定期同步，限制 $k$。

### 4.3 IMPALA 的 V-trace（异步 split 的代表算法）

IMPALA 用 V-trace 修正异步 split 的 stale data：用 $\rho_{\bar\theta}=\min(\bar\rho, \pi_{\theta}/\pi_{\theta_{old}})$ 截断 ratio，对 $V$ 做多步 return 估计，容忍一定 staleness。这是异步 learner/worker 架构的理论基础。

## 5. 代码示例

```python
import torch, threading, queue, time

class Policy:
    def __init__(self):
        self.theta = torch.zeros(4)          # 简化参数
    def generate(self, n):
        return torch.randn(n, 4) * self.theta # 模拟 rollout
    def train_step(self, batch):
        self.theta += 0.01 * batch.mean(0)    # 模拟梯度更新

policy = Policy()
buffer = queue.Queue(maxsize=10)              # worker→learner 管道

# ===== worker(采样器,独立线程) =====
def worker():
    while True:
        traj = policy.generate(8)             # 用当前 θ 生成
        buffer.put(traj)                      # 进 buffer
        time.sleep(0.1)                       # 模拟生成耗时

# ===== learner(训练器,独立线程) =====
def learner():
    while True:
        batch = buffer.get()                  # 从 buffer 取
        policy.train_step(batch)              # 训练更新 θ
        time.sleep(0.2)                       # 模拟训练耗时

# 启动(异步 split:采样与训练并行)
threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=learner, daemon=True).start()
time.sleep(2)
print(f"异步 split 训练后 θ = {policy.theta}")
print("worker 持续采进 buffer,learner 持续取训,两者并行提吞吐")
print("实际:worker=vLLM 生成,learner=FSDP 训练,buffer=Ray object store")
```

> [!tip] 实际工程（verl）
> verl 用 Ray 调度：rollout worker（vLLM 生成）→ Ray object store（buffer）→ learner（FSDP 训练），weight sync 用 Ray actor 的参数同步。这是 RLHF 大规模 split 的标准实现。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[RLHF (PPO)]]、[[PPO optimization]]（split 的需求来源）、[[MDP]]（rollout = MDP 交互）、[[replay buffer]]（数据管道，待展开）。
- **下游（应用）**: [[rollout worker]]、[[inference worker]]、[[asynchronous training]]（split 的具体形态）、[[训推分离]]、[[weight sync mechanism]]、[[stale policy problem]]（异步的代价）、verl/OpenRLHF/IMPALA 实现。
- **对比 / 易混**:
  - **learner/worker split vs actor-critic**：前者是**系统角色**（采样的不训），后者是**算法角色**（actor 出动作、critic 估价值）。不同维度，可叠加（worker 跑 actor 生成，learner 跑 actor+critic 训练）。
  - **learner/worker split vs [[训推分离]]**：前者是**训练内部**的采训分离（worker 采数据给 learner 训），后者是**训练与部署**的分离（PT 训练 vs PD 部署）。不同层次，split 是 PT 内部的架构。
  - **同步 split vs 异步 split**：见 §3.3。异步提吞吐但引入 stale data。
  - **worker vs [[rollout worker]] vs [[inference worker]]**：worker 是泛称，rollout worker 专跑 RL 生成，inference worker 专跑部署服务。详见各自专篇。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"learner/worker = actor/critic"**：错。前者是系统角色（采训分离），后者是算法角色（动作/估值）。不同维度。
> 2. **"split 是大模型独有"**：不完全。RL 早有（IMPALA/A3X），大模型因生成贵+框架异构更需。
> 3. **"split 一定提吞吐"**：异步 split 才提（重叠采训），同步 split 仍互相等（利用率低）。
> 4. **"buffer 越大越好"**：异步 buffer 大平滑波峰，但 stale data 也多（旧数据堆积），需 tradeoff。
> 5. **"worker 和 learner 用同框架"**：不必。生成用 vLLM（推理优化），训练用 FSDP（显存优化），异构框架是 split 的优势。
> 6. **"split 无代价"**：异步 split 引入 stale data（[[stale policy problem]]），需 weight sync + ratio clip 修正。
> 7. **"小模型也该 split"**：不必。小模型生成快、同进程交替够用，split 的复杂度不值得。
> 8. **"worker 只采不训"**：对。但 worker 需前向（生成需 $\pi_\theta$），只是不 backward/step。

## 8. 延伸细节

### 8.1 IMPALA：经典 learner/worker 架构

IMPALA（Importance Weighted Actor-Learner Architecture，DeepMind 2018）是 learner/worker split 的里程碑：
- 多 actor（worker）并行采 trajectory，单 learner 训练；
- V-trace 修正异步 stale data；
- throughput 大超同步 A3C。
RLHF 的 verl/OpenRLHF 架构是其在大模型时代的延续。

### 8.2 split 的分布式扩展

- **多 worker**：N 个 rollout worker 并行采样，吞吐 N×；buffer 汇集；
- **多 learner**：M 个 learner 数据并行训练（梯度 all-reduce）；或模型并行（TP/PP）；
- **混合**：多 worker + 多 learner，用 Ray/RLlib 调度。
扩展瓶颈在 buffer 带宽与 weight sync 延迟。

### 8.3 split 与训推分离的关系

- **learner/worker split**：训练内部，worker 采数据给 learner 训（都在 PT 侧）；
- **[[训推分离]]**：训练（PT）与部署（PD）的分离（PT 的 learner 训出的 θ 给 PD 的 inference worker 部署）。
两者层次不同：split 是 PT 内部架构，训推分离是 PT-PD 之间。但都涉及 weight sync（参数同步）。

### 8.4 split 的现代实现：Ray

Ray 是 split 架构的主流调度框架：
- worker 与 learner 是 Ray actor（独立进程）；
- buffer 用 Ray object store（共享内存，零拷贝）；
- weight sync 用 Ray actor 的参数传递；
- verl/OpenRLHF/RLlib 都基于 Ray。
详见 [[Ray基础]]（待展开）。

### 8.5 split 的瓶颈与优化

- **weight sync 延迟**：learner 更新后同步 θ 给 worker，大模型 θ 大（GB 级），同步慢 → worker 用旧 θ 久（stale 重）。优化：增量同步/参数分片/异步流；
- **buffer 带宽**：trajectory 数据大（长序列 × 多样本），buffer 传输出入瓶颈。优化：共享内存/压缩/就地存储；
- **负载均衡**：worker 与 learner 速度不匹配时 buffer 积压或空等。优化：动态调 worker/learner 数量。
这些是 split 工程的调优点。

---
相关: [[训练架构]]、[[RLHF (PPO)]]、[[PPO optimization]]、[[rollout worker]]、[[inference worker]]、[[asynchronous training]]、[[训推分离]]、[[weight sync mechanism]]、[[stale policy problem]]、[[replay buffer]]、[[MDP]]、[[Ray基础]]
