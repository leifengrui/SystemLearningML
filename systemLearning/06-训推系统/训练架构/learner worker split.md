# learner worker split

> **所属章节**: [[训练架构]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（架构概念，需懂 RL 训练流程与大模型工程）

## 1. 一句话定义

**learner / worker split（训练器/采样器分离）** 是训推系统的基础架构：把**采样数据**（worker / actor / rollout worker）与**训练更新**（learner / trainer）解耦到**不同进程/设备**，通过 **buffer / replay** 传递数据。worker 负责与环境交互或自回归生成采数据，learner 跑 forward/backward/optimizer step 做梯度更新。对比传统 DL 的"worker 既采又训"同步模式，split 让**生成与训练可异步并行**、**资源异构**（生成用推理引擎、训练用训推框架）、**吞吐提升**，是大模型 RL（尤其 [[RLHF (PPO)]] 的 [[PPO optimization]] §8.3）与在线学习的工程基础。经典实现是 IMPALA（Espeholt et al. 2018），RLHF 中 verl/OpenRLHF 沿用此架构。

> [!note] 别名
> learner/worker split = actor-learner architecture = sampler-trainer split = producer-consumer 训练架构。都指"采样的不训，训的不采样"。

> [!note] 解答：训练=learner/trainer，但 worker≠"推理/部署"意义上的 inference——别把 split 和训推分离搞混
> **先直接回答三连问：**
> 1. **"训练就是 trainer？"** —— 是。learner = trainer = 训练器，跑 forward/backward/optimizer.step 更新参数 $\theta$。文中"learner（训练器）"和"trainer"是同义词，只是社区叫法不同（RL 圈习惯叫 learner，DL 圈习惯叫 trainer）。
> 2. **"推理就是 worker？"** —— **不准确，容易踩坑**。worker 确实在做"前向推理"（用 $\pi_\theta$ 生成 trajectory），但它做的是**为训练采数据的前向**，不是**为用户服务的前向**。两件事都叫"inference"，却属不同层次：
>
> | 维度 | worker 的"推理" | 部署侧的"推理"（inference worker） |
> |---|---|---|
> | 目的 | 采 trajectory 喂 learner 训练 | 给用户返回回答 |
> | 数据去向 | 进 buffer → learner 消费 | 直接出给用户 |
> | 权重是否变 | 变（learner 更新后 weight sync 回来） | 冻结（部署快照） |
> | 框架 | vLLM/SGLang（重生成吞吐） | vLLM/TGI/TRT-LLM（重服务延迟/吞吐） |
> | 所属阶段 | PT（训练）内部 | PD（部署） |
>
> 所以 worker 是"训练流程里负责采样的进程"，它的推理只是**手段**（采样手段），不是**目的**（服务用户）。文中 §6 已把 [[rollout worker]]（RL 采样用）与 [[inference worker]]（部署服务用）分成两个专篇，正是为了堵这个混用。
> 3. **"这是训推分离？"** —— **不是**。这是最容易混的点，本文 §6/§8.3 已专门辨析：
>    - **learner/worker split** = **训练（PT）内部**的"采样 vs 训练"分离，两端都在训练侧，产物是更新后的 $\theta$。
>    - **[[训推分离]]** = **训练（PT）与部署（PD）之间**的分离，把训好的 $\theta$ 拿去给用户服务。
>
> 两者都涉及 weight sync（参数同步），但层次不同：split 是 PT 内部架构，训推分离是 PT↔PD 之间。一句话口诀：**"split 拆的是采与训，训推分离拆的是训与用"**。
>
> **正名小结**：learner=trainer=训练器（更新 $\theta$）；worker=采样器/rollout worker（用 $\pi_\theta$ 采数据）；inference worker=部署推理（服务用户）。三者别混。详见 §3.1 角色表与 §6 关系小节。
## 2. 为什么需要它（动机与背景）

传统 DL 训练（监督学习）的数据是**静态数据集**，worker（DDP 数据并行）只做 forward/backward，无需"采样"。但 RL 训练的数据是**动态采的**（rollout 生成），引入新矛盾：
1. **生成与训练耗时不对称**：大模型 RL 的 rollout 自回归生成慢（秒~分钟级），训练 forward/backward 也重（秒级），若**同进程同 GPU 交替**做生成+训练，GPU 互相闲置（生成时训练等，训练时生成等）→ 利用率低；
2. **框架异构**：生成用推理引擎（vLLM/SGLang，优化 [[KV cache management]]/连续批处理），训练用训推框架（DeepSpeed/FSDP/ZeRO，优化 [[activation memory]]/通信），两者 API 与优化点不同，同进程难兼容；
> [!note] 解答：Megatron 属于这里；但"四个"其实只有三个框架，ZeRO 是 DeepSpeed 里的算法
> **先纠一个常见误会**：把 DeepSpeed / FSDP / ZeRO / Megatron 并列成"四个框架"是错的。真实关系是：
> - **DeepSpeed**：微软开源的**训练库/优化引擎**，里面实现了一套叫 **ZeRO** 的显存分片算法（Zero Redundancy Optimizer）。所以 **ZeRO 不是独立框架，是 DeepSpeed 的核心算法**。
> - **FSDP**（Fully Sharded Data Parallel）：PyTorch **原生**实现的、和 ZeRO 同思路的分片数据并行——可理解为"PyTorch 版的 ZeRO"。PyTorch 2.x 起官方主推，逐步替代 DeepSpeed ZeRO 的角色。
> - **Megatron-LM**：NVIDIA 开源的**模型并行训练框架**，主推 **TP（Tensor Parallel 张量并行）+ PP（Pipeline Parallel 流水线并行）+ DP** 三维并行，是另一条省显存路线。
>
> 所以严格讲是 **三个框架（DeepSpeed / FSDP / Megatron-LM）+ 一个算法（ZeRO）**，且 FSDP 与 DeepSpeed-ZeRO 同源思路。
>
> **Megatron 属于这里吗？** —— 属于。learner 跑 forward/backward 用的"训推框架"，Megatron 是其中之一（与 DeepSpeed/FSDP 并列的另一条路线）。文中只举了 DeepSpeed/FSDP 是举例，Megatron 同属 learner 侧框架。
>
> **四者差别对照表**（把 ZeRO 也列进去便于对照，但标注它是算法不是框架）：
>
> | 维度 | DeepSpeed | FSDP | ZeRO（算法） | Megatron-LM |
> |---|---|---|---|---|
> | 性质 | 训练库/优化引擎 | PyTorch 原生 DP 升级版 | 显存分片**算法**（非框架） | 训练框架 |
> | 出身 | 微软 DeepSpeed | PyTorch 官方 | DeepSpeed 论文（Rajbhandari 2019/2020） | NVIDIA |
> | 主并行策略 | ZeRO-1/2/3 数据并行分片 | ZeRO-3 同思路分片 | 把 optimizer state/grad/param 按 stage 切分到各 rank | TP（切权重矩阵）+ PP（切层到 stage）+ DP |
> | 切分对象 | optimizer state → +grad → +param 逐级切 | optimizer state+grad+param 全切（=ZeRO-3） | Stage1/2/3 三档 | 权重矩阵按列/行切（TP）、层段切（PP） |
> | 通信量 ∝ | 参数量（all-gather/reduce-scatter） | 参数量 | 参数量 | **激活值**（不随模型变大而变大） |
> | 跨节点扩展 | ZeRO-3 可跨节点，但通信∝参数，超大模型受带宽限 | 同 ZeRO-3 | 同 | TP 需 NVLink（机内），PP 可跨机 |
> | 强项 | 配置成熟、3D parallel、ZeRO-Infinity 可 offload 到 CPU/NVMe | 官方原生、与 PyTorch 生态无缝、torch compile 友好 | 理论根基，定义了"切什么"的分层 | **超大模型**（百 B/千亿）必选，TP+PP 把单层和层间都切，通信不随模型涨 |
> | 弱项 | 超大模型 ZeRO-3 通信瓶颈、配置复杂 | 超大模型仍受 all-gather 带宽限、不如 Megatron TP 省通信 | 只是算法，要靠 DeepSpeed/FSDP 实现 | TP 强耦合 NVLink 拓扑、PP 有 bubble、工程复杂 |
> | 典型规模 | 7B~70B 友好 | 7B~70B 友好 | — | 100B+、千亿训练主力 |
> | 与本文 learner 关系 | learner 框架 ✓ | learner 框架 ✓ | learner 内部算法 ✓ | learner 框架 ✓ |
>
> **直觉记忆**：
> - **DeepSpeed = ZeRO 算法的"原厂实现库"**；**FSDP = "PyTorch 原生版 ZeRO-3"**；两者同思路（数据并行 + 把训练态显存分片），二选一即可，主流正从 DeepSpeed 迁向 FSDP（PyTorch 官方加持）。
> - **Megatron = 另一条路**：不靠分片存全量，而是**真切分权重**让每卡只算一片，通信∝激活而非参数，超大模型才扛得住，代价是强依赖 NVLink 拓扑。
> - **ZeRO 三 stage 速记**：Stage1 切 optimizer state（省 4×）、Stage2 再切 grad（省更多）、Stage3 再切 param（全切，省最多但通信最重）。FSDP ≈ Stage3。
>
> **何时用哪个**（learner 选型经验）：
> - 7B~70B、单机多卡或少量节点 → **FSDP**（官方、省心、torch compile 友好）。
> - 老项目/特定配置依赖 DeepSpeed → **DeepSpeed ZeRO-2/3**。
> - 100B+、千亿、跨机大规模 → **Megatron-LM TP+PP**（必要时还可与 FSDP/ZeRO 混用，叫 3D parallel）。
> - RLHF 的 learner 一般模型不太大（actor 7B/70B 居多），**FSDP/DeepSpeed 是 verl/OpenRLHF 的默认选择**；Megatron 主要用于预训练阶段。
>
> 详见 [[Fully Sharded Data Parallel]]、[[Megatron-LM]]、[[ZeRO]]（待展开）、[[DeepSpeed]]（待展开）。
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
> [!note] 解答：吞吐里的 "trajectory" 是什么？——一条完整 rollout = 一条轨迹 = 一个训练样本单位
> §4.1 公式里 $T_w$ 的单位是 **trajectory/秒**（每秒采几条轨迹），所以得先搞清"一条 trajectory"是什么。
>
> **trajectory（轨迹）** = 一次完整 rollout 采到的**全部数据**，在 LLM-RL 里就是**一个 prompt 对应的一次完整生成 + 它的全部训练副产物**，打包成一个样本单元：
> $$\text{trajectory} = \big(\,x,\ y,\ \log\pi_\theta(y|x),\ v_\phi(x,y),\ r\,\big)$$
> - $x$ = prompt；
> - $y$ = rollout worker 自回归生成的 response（一串 token）；
> - $\log\pi_\theta(y|x)$ = 生成时每个 token 的对数概率（PPO 算 ratio 用，见 [[forward与backward]] §logp）；
> - $v_\phi(x,y)$ = critic 估的价值（可选，PPO 用）；
> - $r$ = reward（RM 前向或规则算）。
>
> **"一条 trajectory" 的粒度**：在 LLM-RL 中**默认 = 一次 prompt→response**（不是每个 token，也不是整个 batch）。RLHF 里的一个 batch = $B$ 条 trajectory（$B$ 个 prompt 各采一条）。在传统 RL（Atari/MuJoCo）里一条 trajectory = 一个 episode（一局游戏从开始到结束的 (s,a,r) 序列），LLM 把"生成一条 response"类比为"跑一局 episode"。
>
> **为什么 $T_w$ 用 trajectory/秒 而不用 token/秒**：因为 learner 消费的单位是 trajectory（一条喂一次 PPO update 的样本），不是 token——token 是生成内部的开销，trajectory 才是"产出训练数据"的单位。$T_w$ 衡量"每秒产几条可用样本"。若要衡量生成效率本身，用 [[sampling throughput]] 的 tokens/sec。
>
> **trajectory/秒 ↔ tokens/秒 的换算**：$T_w = \frac{\text{tokens/sec}}{\bar L}$，$\bar L$ = 平均每条 response 长度。例：vLLM 生成 10000 tokens/sec、每条 response 平均 500 token → $T_w = 20$ trajectory/秒。
>
> **延伸**：trajectory 的生成、采样策略、打包回传详见已展开的专章 [[trajectory generation]]（§21 Rollout系统），它与本条 §4.1 的 $T_w$ 是"产物 vs 产速"的关系。另见 [[batching strategy]]（trajectory 如何组 batch 喂 learner）。

### 4.2 stale data 的 lag

异步 split 中，worker 在时刻 $t$ 用 $\theta_t$ 采数据 $D_t$，learner 已训练到 $\theta_{t+k}$ 才消费 $D_t$。**staleness = $k$ 步**。

PPO 的 importance sampling ratio $\rho=\pi_{\theta_{t+k}}(a)/\pi_{\theta_t}(a)$ 随 $k$ 增大偏离 1，clip 限制此偏离。$k$ 太大 → ratio 失真 → 训练不稳（[[stale policy problem]]）。故需 [[weight sync mechanism]] 定期同步，限制 $k$。

### 4.3 IMPALA 的 V-trace（异步 split 的代表算法）

IMPALA 用 V-trace 修正异步 split 的 stale data：用 $\rho_{\bar\theta}=\min(\bar\rho, \pi_{\theta}/\pi_{\theta_{old}})$ 截断 ratio，对 $V$ 做多步 return 估计，容忍一定 staleness。这是异步 learner/worker 架构的理论基础。
> [!note] 解答：IMPALA 的 V-trace 细说——异步 split 的"截断重要性采样 + 多步价值回溯"
> IMPALA（Espeholt 2018）是异步 learner/worker split 的里程碑，难点是 **worker 用 $\theta_{\text{old}}$ 采、learner 已到 $\theta_{\text{new}}$**，直接用旧数据训会偏。V-trace 是它的修正手段。下面拆三步。
>
> **第 1 步：问题——朴素 n-step return 在异步下有偏**
> 朴素多步价值估计：$V_{\text{target}} = V_\theta(s_t) + \sum_{i=t}^{t+n-1}\gamma^{i-t}\big(r_i+\gamma V_\theta(s_{i+1})-V_\theta(s_i)\big)$。这要求**采样策略 = 当前策略**（on-policy）。但异步 split 里 worker 用 $\theta_{\text{old}}$ 采，$\pi_{\theta_{\text{old}}}\ne\pi_\theta$，直接用 → 偏。
>
> **第 2 步：截断重要性采样系数 $\rho_s$**
> 经典重要性采样用 $\rho_s=\frac{\pi_\theta(a_s|s_s)}{\pi_{\theta_{\text{old}}}(a_s|s_s)}$ 修正分布差异。但 $\rho_s$ 方差大、可能爆炸。**V-trace 截断成 $\bar\rho$ 和 $\rho$ 两个上界**：
> $$\rho_s = \min\big(\bar\rho,\ \frac{\pi_\theta(a_s|s_s)}{\pi_{\theta_{\text{old}}}(a_s|s_s)}\big),\quad c_s = \min\big(\bar c,\ \frac{\pi_\theta(a_s|s_s)}{\pi_{\theta_{\text{old}}}(a_s|s_s)}\big)$$
> - $\rho_s$（截断到 $\bar\rho$，常取 1）控制 **value target 的偏差**——截断让修正不爆炸，代价是引入轻微偏差；
> - $c_s$（截断到 $\bar c$，常取 1）控制 **trace 系数**（多步回溯的衰减权重），$\bar c$ 决定回溯多远。
> 这与 PPO 的 ratio clip 是**同一思想的不同实现**：PPO 在 loss 端 clip ratio（$\text{clip}(\rho,1-\epsilon,1+\epsilon)$），V-trace 在 value 估计端 clip。都为压方差防爆炸。
>
> **第 3 步：V-trace target 公式**
> $$v_s = V_\theta(s_t) + \sum_{i=t}^{s-1}\gamma^{i-t}\Big(\prod_{j=t}^{i-1} c_j\Big)\Big(\rho_i\,\delta_i\Big),\quad \delta_i = r_i + \gamma V_\theta(s_{i+1}) - V_\theta(s_i)$$
> - $\delta_i$ = TD 误差；
> - $\rho_i$ 截断修正 $\delta_i$ 的偏差（on-policy 时 $\rho_i=1$）；
> - $\prod c_j$ = trace 衰减，控制回溯距离（$\bar c=1$ 时退化为 n-step，$\bar c<1$ 时指数衰减）。
> learner 最小化 $(V_\theta(s)-v_s)^2$。
>
> **直觉记忆**：V-trace = "**截断 IS 修正 × 多步 TD 回溯**"。$\rho$ 截断防 IS 爆炸，$c$ 截断控回溯深度。两者分离，比 PPO 单一 clip 更细粒度（value 与 policy 不同力度），适合异步的轻度 stale。
>
> **在 LLM-RL 里的现状**：RLHF-PPO 多用 **PPO ratio clip + target_kl 早停**（比 V-trace 简单、社区熟），V-trace 在传统 RL（IMPALA/A3C 系）更主流。但思想一致——异步 split 必须修正 $\pi_{\theta_{\text{old}}}\to\pi_\theta$ 的分布偏移。AReaL（蚂蚁 2025，[[#9.3 业界进展调研]]）用 "staleness-enhanced PPO" 也是同一脉络。详见 [[stale policy problem]]、[[PPO clipped objective]]、[[variance reduction]]。

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
  - **learner/worker split vs [[训推分离]]**：前者是**训练内部**的采训分离（worker 采数据给 learner 训），后者是**训练与部署**的分离（PT 训练 vs PD 部署）。不同层次，split 是 PT 内部的架构。详见 [[训推分离]] §3.2——已展开为独立专章。
  > [!note] 已展开为独立笔记
  > 训推分离已单独成章，见 [[训推分离]]（§20）。要点速览：
  > - **层次嵌套**：训推分离 ⊃ split。split 是 PT 内部（采 vs 训）的分离，训推分离是 PT↔PD（训 vs 用）的分离。
  > - **weight sync 节奏不同**：split 的 sync 是 learner→rollout worker，step 级高频；训推分离的 sync 是 PT→PD，版本级低频。
  > - **撞名警告**：[[PD分离]]（Prefill-Decode，推理内部两阶段）也简称"PD 分离"，但 PD 在这里是 Policy Deployment，不是 Prefill-Decode。
  > - **口诀**：split 拆采与训，训推分离拆训与用。
  > 完整动机/角色表/weight sync 形态/版本管理/误区见 [[训推分离]]。
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
> [!note] 解答：吞吐 ≠ "吐字的量"——它是"单位时间产出量"，且分三个层次，别只盯 token
> "吞吐"字面像"吐出多少字"，但工程上 **throughput = 单位时间完成的单位工作量**，"吐字"只是其中一种。在本章 split 语境里"吞吐"分**三个层次**，单位不同、对应不同瓶颈，必须分清：
>
> | 层次 | 吞吐单位 | 衡量谁 | 公式 | 优化对象 |
> |---|---|---|---|---|
> | **① 生成吞吐**（[[sampling throughput]]） | **tokens/sec** | rollout worker 生成 | $\frac{B\cdot T}{T_{\text{gen}}}$ | KV cache / continuous batching / speculative decoding |
> | **② 采样吞吐**（$T_w$，§4.1） | **trajectory/秒** | worker 产训练样本 | $\frac{\text{tokens/sec}}{\bar L}$ | 多 worker 并行、采样策略 |
> | **③ 训练吞吐**（$T_l$，§4.1） | **batch/秒** 或 **samples/秒** | learner 训练 | $\frac{1}{t_{\text{fwd}}+t_{\text{bwd}}+t_{\text{step}}}$ | FSDP/DeepSpeed 显存通信、批大小 |
>
> **为什么"吐字量"只是 ①**：token 是生成的最小单位，"每秒吐几个 token"=①生成吞吐。但 learner 关心的不是 token，是"每秒能消费几条 trajectory / 几个 batch"=③。中间 ② 把 token 换算成 trajectory（除以平均长度 $\bar L$）。三者通过 $\bar L$ 和 batch size 串起来。
>
> **split 的总吞吐 = min($T_w$, $T_l$)（同步）或趋近 max（异步）**——瓶颈在慢的那侧。RLHF 里 ①生成慢（占 60~80% 时间），故 **rollout 侧（①→②）是瓶颈**，优化重心在生成侧；若 learner 慢（大模型 + 大 batch），瓶颈转到 ③。
>
> **与延迟的区别**（常混）：吞吐 = "总产出/总时间"（重批量），延迟 = "单请求从进到出多久"（重单体）。rollout worker 重吞吐（大批并行），[[inference worker]] 重延迟（单请求快）。详见 [[sampling throughput]]、本文件 §4.1、[[rollout worker]] §4.2。
### 8.3 split 与训推分离的关系

- **learner/worker split**：训练内部，worker 采数据给 learner 训（都在 PT 侧）；
- **[[训推分离]]**：训练（PT）与部署（PD）的分离（PT 的 learner 训出的 θ 给 PD 的 inference worker 部署）。
两者层次不同：split 是 PT 内部架构，训推分离是 PT-PD 之间。但都涉及 weight sync（参数同步）。
> [!note] 解答：split 与训推分离的"展开细说"——同是分离、同是 weight sync，但层次/节奏/产物全不同
> 这两句话最容易让人觉得"split 和训推分离是一回事，都分离、都 sync 参数"。其实它们是**两层嵌套**的分离，下面把"同与不同"一次讲透（完整版见专章 [[训推分离]] §3.2）。
>
> ### 同在哪（为什么易混）
> - 都把"采/生成"与"训/用"拆成两个独立进程/集群；
> - 都有一条 **weight sync 通道**把 $\theta$ 从"更新方"单向推给"使用方"；
> - 都因异步产生 **[[stale policy problem]]**（使用方的 $\theta$ 落后更新方）。
>
> ### 不同在哪（关键）
> | 维度 | learner/worker split | 训推分离 |
> |---|---|---|
> | **分离两端** | rollout worker ↔ learner | PT ↔ PD |
> | **两端所属** | **都在 PT 内部** | PT 与 PD，跨阶段 |
> | **使用方干什么** | learner 拿 $\theta$ 继续**训练**（更新它） | PD 拿 $\theta$ **服务用户**（不更新，只前向） |
> | **weight sync 频率** | step 级，高频（每 N 步） | 版本级，低频（训出一版才推） |
> | **sync 触发** | learner 每 N 步主动推 | PT 训完一个 checkpoint/版本才推 |
> | **stale 后果** | ratio 失真→训练不稳 | 用户接触旧版本→服务落后 |
> | **框架** | rollout=vLLM, learner=FSDP（都训练侧） | PT=FSDP, PD=vLLM server（训 vs 部署） |
> | **产物** | 更新后的 $\theta$ | 用户回答 |
>
> ### 层次嵌套（最重要的一张图）
> ```
> 训推分离 ────────────────────────────────────────
>   PT(训练) ─────────────  weight sync(版本级) ───→  PD(部署)
>     └─ learner/worker split(训练内部)              inference worker 服务用户
>          rollout worker ↔[sync step级]↔ learner
>   ↑ 两条 weight sync 叠加：内层 step级(learner→rollout)，外层版本级(PT→PD)
> ```
> 即 **训推分离 ⊃ split**：split 是 PT 内部的子分离，训推分离是包在外面的训练-部署分离。一个 RLHF 系统里**两条 weight sync 同时存在**——内层让 rollout 用新 $\theta$ 采数据，外层让 PD 用新 $\theta$ 服务用户。内层高频、外层低频。
>
> ### 两条 weight sync 的对比
> | | split 的 weight sync | 训推分离的 weight sync |
> |---|---|---|
> | 方向 | learner → rollout worker | PT → PD |
> | 频率 | step 级（高频） | 版本级（低频） |
> | 实现 | Ray actor 参数注入 / NCCL AllGather | checkpoint 加载 / 流式 broadcast + 热切换 |
> | 失败影响 | rollout 用旧 $\theta$ 采→stale data 训不稳 | PD 用旧版本→服务落后 |
> | KV cache | sync 后 rollout 侧 cache 可清（采新数据） | sync 后 PD 侧 cache 失效（用户请求需重算） |
>
> **口诀再强化**：**"split 拆采与训，训推分离拆训与用；内 sync 高频喂采样，外 sync 低频喂部署。"** 详见 [[训推分离]]、[[weight sync mechanism]]、[[stale policy problem]]、本文件 §3.5、§8.3。

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
> [!note] 解答：联网调研——worker↔learner 负载均衡的业界进展（2024–2026）
> 本节 §8.5 说"动态调 worker/learner 数量"是负载均衡的朴素思路。调研当前业界，已从"手动配比"进化到**系统级自动均衡**，以下是四条主线（截至 2026-07 查询）：
>
> ### 1. 全异步解耦 + 主动均衡（AReaL，蚂蚁/清华/港科大 2025）
> 论文 [arXiv:2505.24298](https://arxiv.org/abs/2505.24298)。核心贡献正是解决本节痛点——**同步 RL 里"生成必须等本 batch 最长 response 跑完才更新"导致 GPU 闲置**。AReaL 是**完全异步**系统：rollout worker 不等待、持续生成，training worker 攒够一批就更新。它**显式把"均衡 rollout 与 training workload 以控制 data staleness"作为设计目标**——通过调节两侧 worker 数/速度把 staleness $k$ 压在可控范围，并配 **staleness-enhanced PPO**（在 ratio clip 基础上按 $k$ 自适应）容忍过时样本。实测比同 GPU 数同步系统 **2.77× 加速**，性能持平或更优。2026 年 AReaL 2.0 扩展到自演进 Agent 在线 RL（来源：arXiv / AReaL 官网）。
>
> ### 2. 3D-HybridEngine + 自动设备映射（HybridFlow / verl，字节 EuroSys 2025）
> 论文 [arXiv:2409.19256](https://arxiv.org/abs/2409.19256)。不是"动态加减 worker"，而是**让同一批 GPU 在训练态/生成态之间高效 reshard**——训练用 FSDP/TP 切分，生成切到 vLLM 的 TP/PP，**零显存冗余**切换 + 通信最小化。配**自动设备映射算法**：枚举各模型（actor/critic/reward/ref）的并行策略与设备分配，按负载**自动算最优配比**最小化单次迭代端到端延迟。这是"用 reshard 替代加减 worker"的均衡思路，比同步 RLHF 基线 **1.53×~20.57× 吞吐**（来源：arXiv / verl 文档 verl.org.cn）。
>
> ### 3. 生成侧变长不均 → continuous batching + 长度分桶
> rollout 生成长度天然不均（数学题短则几十 token、长则几千），同 batch 等最长→短板效应。业界用 **vLLM continuous batching**（动态插批、谁完成谁退出）+ **length bucketing / packing**（[[batching strategy]]）把相近长度组批，减 padding 浪费。SGLang 的 **RadixAttention** 对 GRPO"同 prompt 采 G 条"场景复用前缀 KV，进一步均衡多候选生成的吞吐（来源：vLLM/SGLang 文档）。
>
> ### 4. 在线/离线混合 + 弹性调度
> - **AsyncActor / OpenRLHF async mode**：rollout 与 trainer 解耦成独立 Ray actor group，用 queue/buffer 平滑两侧速度差，buffer 积压时动态起更多 rollout worker（Ray autoscaler）；
> - **colocate vs 分离**（[[placement group]]）：weight sync 频繁时把 rollout 与 learner colocate 同节点走 NVLink；显存紧张时分离部署各占资源——verl 支持 `rollout_sleep_level`（`0/1/2`）控制 rollout 模型在训练期的显存释放级别，是显存-切换延迟的均衡旋钮（来源：verl 文档）。
>
> ### 小结表
> | 进展 | 解决的均衡痛点 | 手法 | 来源 |
> |---|---|---|---|
> | AReaL 全异步 | 生成等最长→GPU 闲置 | 解耦+主动均衡 workload+staleness PPO | arXiv:2505.24298 |
> | HybridFlow 3D-HybridEngine | 训练/生成切分切换慢 | reshard 零冗余+自动设备映射 | arXiv:2409.19256 |
> | continuous batching+bucketing | 变长生成短板效应 | 动态插批+长度分桶 | vLLM/SGLang |
> | Ray autoscaler+colocate | 速度不匹配/weight sync 慢 | 弹性扩缩 worker+同节点共置 | verl/OpenRLHF |
>
> **对本节的更新**：原文"动态调 worker/learner 数量"是早期朴素思路；当前主流已升级为——**全异步解耦（AReaL）+ reshard 均衡（HybridFlow）+ 生成侧 dynamic batching（vLLM）+ 弹性调度（Ray）** 四件套协同。详见 [[asynchronous training]]、[[sampling throughput]]、[[batching strategy]]、[[weight sync mechanism]]、[[placement group]]。
这些是 split 工程的调优点。

---
相关: [[训练架构]]、[[RLHF (PPO)]]、[[PPO optimization]]、[[rollout worker]]、[[inference worker]]、[[asynchronous training]]、[[训推分离]]、[[weight sync mechanism]]、[[stale policy problem]]、[[replay buffer]]、[[MDP]]、[[Ray基础]]
