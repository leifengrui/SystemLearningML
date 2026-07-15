# RL worker故障恢复

> **所属章节**: [[容错]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: RL worker recovery / RL 故障恢复 / RL checkpoint / replay buffer / 断点续训 / RL 容错 / multi-role checkpoint / rollout state recovery / 弹性 RL 训练 / RL training resilience
> **难度**: 高（需懂 [[RL角色拓扑]]、[[colocated与disaggregated部署]]、[[synchronous与asynchronous rollout]]、[[asynchronous training]]、[[distributed checkpoint]]、[[elastic training]]、[[Ray与分布式调度]]、[[rollout worker]]、[[RL权重同步]]、[[多轮Agent rollout与tool calling]]、[[sandbox与轨迹回放]]）

> **所属小节**: §88


## 1. 一句话定义

**RL worker 故障恢复** 是 [[17-RL训推一体框架]] 中应对"多角色（actor/rollout/ref/critic/reward model）、跨多机、长时间运行的 RL 训练**任一 worker 故障**后如何恢复训练"的体系——RL 训练比单纯监督学习**更易故障且更难恢复**：因为它除了要存 model checkpoint，还要恢复 **rollout 状态（采到哪了）、replay buffer（已采的 trajectory）、optimizer state、critic/value state、scheduler 队列（哪些样本待训）、staleness 计数（异步训练的版本号）、sandbox/env 状态（多轮 Agent rollout）** 这一整套**有状态的训练上下文**，任一项丢都可能导致恢复后训练**不一致**（重复训已训样本、漏训、staleness 错乱、gradient 统计偏差）。恢复手段分四层——① **checkpoint**：定期把 model+optimizer+scheduler+RL 专属状态（rollout 进度/buffer/critic）落盘，故障后从 ckpt resume；② **replay buffer**：把采过的 trajectory 存进 buffer，故障后从 buffer 重放重训，**不需重采**（尤其异步训练 buffer 是数据复用与容错的核心，trajectory 贵不能白采）；③ **断点续训**：从 ckpt resume 时恢复 **RNG seed** 保证可复现、恢复 scheduler 队列与 staleness 计数保证进度一致；④ **弹性**：节点故障剔除重排，`torchrun --rdzv` / Ray 自动重启 actor，`elastic training` 动态伸缩 worker 数。staleness 与容错有权衡——**异步训练容忍旧数据某种程度上也是容错**（某 worker 慢/挂，其旧数据仍可用）。verl/OpenRLHF 的 ckpt 机制在 model ckpt 基础上扩展了 RL 状态。本条与 [[distributed checkpoint]] 分工——后者是底层 DTensor/FSDP 的**权重保存**机制，本条讲 **RL 层面的恢复语义**（要恢复哪些 RL 特有状态、buffer 如何重放、staleness 如何续）。

> [!note] 三句话定位
> - **是什么**：RL 训练多角色跨多机，任一 worker 挂全停；恢复要存的不止 model ckpt，还有 rollout 状态、replay buffer、optimizer、critic、scheduler 队列、staleness 计数、sandbox 状态；手段四层——checkpoint、replay buffer 重放、断点续训（resume+RNG）、弹性（剔除重启）。
> - **为什么**：RL 比 SFT 难恢复——SFT 只存 model+optimizer 就能续，RL 还要续 rollout 进度（不然重采贵）、buffer（不然丢数据）、staleness（不然异步版本错乱）。trajectory 采样占墙钟 50-80%，故障重采极浪费。
> - **与 [[distributed checkpoint]] 关系**：后者是底层权重保存（DTensor/FSDP shard 存盘），本条是 RL 层面的恢复语义（恢复哪些 RL 状态、buffer 怎么重放、staleness 怎么续）。底层 + 上层语义。


## 2. 为什么需要它（动机与背景）

### 2.1 RL 训练比 SFT 更易故障

RLHF/GRPO 训练的故障面比监督学习（SFT）大得多：

| 维度 | SFT | RL（PPO/GRPO） |
|---|---|---|
| **角色数** | 1（只 actor） | 3-5（actor + rollout + ref + critic + RM/verifier） |
| **跨机规模** | 单组 trainer | actor 池 + rollout 池 + ref 池（disaggregate 常跨数十~上百卡） |
| **运行时长** | hours-days | days-weeks（RL step 数多、每步含 rollout+train） |
| **外部依赖** | 无 | rollout 依赖 vLLM/SGLang、verifier 依赖 sandbox/API、多轮依赖 tool env |
| **故障面** | trainer 挂 | 任一角色挂 + 引擎挂 + env 挂 + 网络挂 |

角色多、跨机广、跑得久、外部依赖多——任一环节故障概率叠加，RL 训练**故障几乎必然**（长训练不挂是例外）。故容错不是可选项，是 RL 框架的**必需能力**。

### 2.2 RL 恢复比 SFT 难：要恢复的不止权重

SFT 故障恢复简单：从 ckpt 读 model + optimizer + scheduler state，续训下一 batch。因为 SFT 数据是**静态数据集**（预存好），故障不影响数据——重读 batch 即可。

RL 的数据是**在线采的**（rollout 实时采 trajectory），故障时要恢复**一整套有状态上下文**：

- **rollout 进度**：采到哪个 prompt、采了几条、哪条 trajectory 采到哪步（多轮）——丢了不知从哪续采；
- **replay buffer**：已采未训 / 已训可复用的 trajectory——丢了要重采（极贵）；
- **optimizer state**：Adam 的 $m,v$ 动量——丢了重启 optimizer 损失收敛性；
- **critic state**（PPO）：value 网络权重 + 其 optimizer——丢了 value 估计重头来；
- **scheduler 队列**：哪些 prompt 待采、哪些 trajectory 待训、batch 怎么组——丢了调度乱；
- **staleness 计数**（异步）：每条 trajectory 的 θ 版本号 $k$——丢了 IS ratio 算错；
- **sandbox/env 状态**（多轮）：trajectory 的 env 状态——丢了 observation 回放断（[[sandbox与轨迹回放]]）。

任一项丢都可能导致恢复后训练**不一致**：重复训已训 trajectory（浪费 + gradient 偏）、漏训（数据亏）、staleness 错（ratio 失真）、gradient 统计偏差（optimizer 重启）。故 RL ckpt 要存**远多于** SFT 的状态。

### 2.3 trajectory 太贵，不能故障重采

LLM-RL 的 rollout 占墙钟 50-80%（[[synchronous与asynchronous rollout]] §2.1），一条 trajectory 采几百~几千 token forward、多轮还要调 env 等待。若故障丢弃已采 trajectory 重采，浪费巨大。故 **replay buffer 的持久化**是 RL 容错的核心——已采 trajectory 存盘，故障后从 buffer 重放重训，不重采。这在异步训练下尤关键（buffer 是异步数据复用的根，详见 [[asynchronous training]]）。

### 2.4 异步训练的容错张力

异步训练（[[synchronous与asynchronous rollout]]、[[asynchronous training]]）下 rollout 与 trainer 解耦并行，某 rollout worker 慢/挂时，trainer 可用其他 worker 的（稍旧的）数据继续训——**异步容忍旧数据某种程度上就是容错**（某 worker 失效不阻塞全局）。但代价是 staleness $k>1$ 需 IS 修正。故 staleness 与容错有权衡：容忍旧数据 → 既能异步提速又能容错，但引入 ratio 偏离风险。设计 RL 容错要利用这层关系——别一挂就全停，让能用的旧数据先训。


## 3. 核心概念详解

### 3.1 RL checkpoint：要存什么

RL ckpt 比 SFT 多存一整套 RL 状态。完整 RL ckpt 清单：

| 状态 | SFT 存? | RL 存? | 用途 |
|---|---|---|---|
| **actor 权重** $\theta$ | ✓ | ✓ | 策略网络 |
| **optimizer state**（$m,v$） | ✓ | ✓ | Adam 动量 |
| **lr scheduler state** | ✓ | ✓ | 学习率调度 |
| **step / global iteration** | ✓ | ✓ | 进度 |
| **rng seed / state** | 建议 | ✓ 必须 | 可复现（数据采样、dropout、采样顺序） |
| **ref 权重** $\theta_{ref}$ | ✗ | ✓ | 冻结参考，KL 约束 |
| **critic 权重** + optimizer（PPO） | ✗ | ✓ | value 网络 |
| **reward model 权重**（若在线训 RM） | ✗ | 视情况 | RM 更新 |
| **replay buffer**（采的 trajectory） | ✗ | ✓ | 故障后重放，免重采 |
| **rollout 进度**（采到哪个 prompt） | ✗ | ✓ | 续采 |
| **scheduler 队列**（待采/待训/已训） | ✗ | ✓ | 调度续 |
| **staleness 计数**（异步版本号） | ✗ | ✓（异步） | IS ratio 续 |
| **sandbox/env 状态**（多轮） | ✗ | ✓（多轮） | env 状态续 + observation 回放 |

**原则**：RL ckpt = SFT ckpt + RL 特有状态（ref/critic/buffer/scheduler/staleness/env）。漏存任一项，恢复后训练不一致。

### 3.2 SFT ckpt vs RL ckpt 对比

| 维度 | SFT checkpoint | RL checkpoint |
|---|---|---|
| **model** | actor $\theta$ | actor $\theta$ + ref $\theta_{ref}$ + critic $V_\phi$ + (RM) |
| **optimizer** | 1 套（actor） | 2-3 套（actor + critic + RM 若训） |
| **数据** | 静态数据集（不存，重读） | replay buffer（trajectory，在线采的，要存） |
| **进度** | batch offset | rollout 进度 + scheduler 队列 + staleness |
| **外部状态** | 无 | sandbox/env（多轮）、verifier 状态 |
| **恢复后** | 续读下一 batch | 续 rollout + 续训 + 续 buffer + 续 staleness |
| **大小** | model+optimizer（如 140GB for 70B） | model×3-5 + buffer + 队列（更大） |
| **故障代价** | 重读数据（便宜） | 重采 trajectory（极贵）→ buffer 持久化是关键 |

### 3.3 replay buffer：数据复用与容错

replay buffer 存**已采的 trajectory**（含 prompt、response/多轮 steps、reward、`old_log_prob`、θ 版本号）。在 RL 容错中三重作用：

1. **故障后重放重训**：故障恢复后，buffer 里的 trajectory **不需重采**，直接送 trainer 重训——省 rollout 开销（最核心的容错价值）；
2. **多 epoch 复用**（PPO 风格）：同一批 trajectory 训多个 epoch（[[GRPO]] 是轻度 off-policy，靠 ratio clip 容忍）——buffer 让"采一次训多轮"成为可能，提数据利用率；
3. **异步数据池**（异步训练）：rollout 持续采入 buffer、trainer 持续从 buffer 取训——buffer 是异步解耦的中间件，故障时 buffer 持久化让两端独立恢复。

buffer 的内容（每条 trajectory 记录）：

```python
buffer_entry = {
    "prompt": "...",
    "trajectory": {"steps": [...]},   # 单轮: response; 多轮: 多步 (见 [[多轮Agent rollout与tool calling]])
    "reward": 1.0,
    "old_log_prob": [...],             # rollout 时 θ_old 算的逐 token logp (mask 后)
    "ref_log_prob": [...],             # ref 算的 (若 KL)
    "theta_version": 5,                # 采样时的 θ 版本号 (staleness k)
    "advantage": 0.3,                  # group-relative (GRPO) 或 critic (PPO)
}
```

故障后恢复：从 ckpt 读 buffer → trainer 对 buffer 内 trajectory 用新 $\theta$ 重算 `new_log_prob` + ratio + advantage → 续训。

### 3.4 断点续训：resume + RNG + 队列 + staleness

从 ckpt resume 不是简单读权重续训，要恢复**完整可复现场景**：

1. **model/optimizer/scheduler**：读 ckpt 恢复 $\theta$、Adam $m,v$、lr；
2. **RNG seed/state**：恢复采样时的随机数生成器状态，保证续训的数据采样顺序、dropout、采样温度序列**与不挂时一致**（否则 gradient 不可复现、调试难）；
3. **rollout 进度**：恢复"采到哪个 prompt、哪条 trajectory 采到哪步"，续采从断点；
4. **scheduler 队列**：恢复"待采/待训/已训"队列，避免重复采/训或漏；
5. **staleness 计数**（异步）：恢复每条 trajectory 的 θ 版本号，ratio 续算；
6. **sandbox/env 状态**（多轮）：恢复 trajectory 的 env 状态或 observation 记录（[[sandbox与轨迹回放]]），多轮 rollout 续采需 env 状态。

任一项不恢复，续训可能：重复训（gradient 重复）、漏训（数据亏）、staleness 错（ratio 失真）、env 断（observation 回放失败）。

### 3.5 弹性：节点故障剔除重启

节点故障（GPU 挂、OOM、网络断）时，弹性训练动态伸缩：

- **torchrun `--rdzv`（rendezvous）**：PyTorch 原生弹性，节点故障后剩余节点重新 rendezvous、重排 world_size、续训。详见 [[elastic training]]。
- **Ray actor 重启**：Ray（[[Ray与分布式调度]]）的 actor 故障可自动重启（`max_restarts`），verl/OpenRLHF 的 single-controller 用 Ray 管 worker，worker 挂 Ray 重启该 actor、重新加载 ckpt 续。
- **剔除故障节点**：故障节点从集群剔除，剩余节点重排 DP/TP rank 续训（world_size 缩）；
- **新增节点**：训练中加节点（world_size 涨）需重 shard 数据/模型，更复杂，常不支持动态加，只支持动态缩。

弹性与 checkpoint 配合：故障 → 剔除 → 重排 → 从最近 ckpt resume（因重排后部分 worker 状态丢，需 ckpt 兜底）。

### 3.6 staleness 与容错的权衡

异步训练容忍旧数据 = 容错的一种形式：

| 场景 | 行为 | 与容错关系 |
|---|---|---|
| **worker 慢**（未挂） | trainer 用其他 worker 的旧数据训 | 容忍旧数据 = 容忍慢 worker（容错） |
| **worker 挂** | trainer 继续用其他 worker 数据训（若 buffer 有） | 不阻塞全局，等同容错 |
| **worker 重启** | 重启后重新采，数据版本号续 | staleness 计数续，ratio 续算 |

故异步 + buffer 持久化 = 天然容错（某 worker 挂，trainer 用 buffer 的旧数据继续）。代价是 staleness $k$ 大 → ratio 偏离需 IS 修正（[[importance sampling与off-policy correction]]）。同步训练则一挂全停（强一致但无容错）。设计时权衡：要强一致（同步、一挂全停）还是容错（异步、容忍旧数据）。

> [!tip] 实践：异步 + buffer 持久化是 RL 容错的甜区
> 同步训练一挂全停、故障代价高；纯同步弹性恢复要全量 ckpt + 重排。异步 + 持久化 buffer 让"某 worker 挂，trainer 用 buffer 旧数据继续"成为可能，天然容错 + 不浪费已采数据。代价是 staleness 需 IS 修正，但 clip 兜底。这是大规模 RL（AReaL/verl fully_async）选异步的容错动机之一。


## 4. 数学原理 / 公式

### 4.1 checkpoint 恢复时间公式

设 ckpt 大小 $D$（bytes）、存储带宽 $B$（bytes/s）、保存频率每 $S$ step 一次。故障在距上次 ckpt $d$ step 处发生（$0 \le d < S$）。

**恢复时间**（从 ckpt 重启 + 重做 $d$ step 的 rollout+train）：

$$
T_{\text{recover}} = \underbrace{\frac{D}{B}}_{\text{读 ckpt}} + \underbrace{d \cdot T_{\text{step}}}_{\text{重做未 ckpt 的 step}}
$$

其中 $T_{\text{step}}$ 是单 RL step 墙钟（rollout + train）。若 $d$ 大（故障刚好在下一次 ckpt 前），重做开销大。故 $S$ 小（频繁 ckpt）降重做开销但增 ckpt 开销（每 $S$ step 存一次 $D/B$）。

**ckpt 开销占比**：

$$
\eta_{\text{ckpt}} = \frac{T_{\text{save}}}{T_{\text{step}}} = \frac{D/B}{T_{\text{step}}} \quad (\text{每 } S \text{ step 一次, 平均 } \eta/S)
$$

权衡：$S$ 小 → 重做少但 ckpt 开销占比大；$S$ 大 → ckpt 开销小但重做多。典型 $S$ 取 10-100 step（视 $T_{\text{step}}$ 与故障率）。异步训练 ckpt 可在 trainer 闲时异步存（不阻塞 rollout），降 $\eta$。

### 4.2 replay buffer 的数据复用率

设 buffer 容量 $C$（trajectory 条数）、每 step 采 $N$ 条新 trajectory、每 trajectory 训 $E$ epoch（PPO 风格多 epoch 复用）。数据复用率：

$$
\text{reuse} = \frac{E \cdot N}{N} = E \quad (\text{每条训 } E \text{ 次})
$$

即每采 1 条 trajectory 训 $E$ 次，rollout 开销摊薄 $E$ 倍。故障后从 buffer 重放，若 buffer 有 $C$ 条已采 trajectory，可重训 $\lfloor C/N \rfloor$ 个 step **不需重采**——省 $\lfloor C/N \rfloor \cdot T_{\text{rollout}}$ 墙钟。

**buffer 容量 vs 故障恢复价值**：

$$
T_{\text{saved}} = \lfloor C/N \rfloor \cdot T_{\text{rollout}} \quad (\text{故障后免重采的时间})
$$

$C$ 越大省越多，但 buffer 占存储（每条 trajectory 含 response/多轮 steps/logp，几百~几千 token × bytes）。权衡 $C$：大 → 容错强但存储贵；小 → 存储省但容错弱。

### 4.3 staleness 续算的一致性

异步训练每条 trajectory 记 $\theta$ 版本号 $k$（采样时 trainer 已更新 $k$ 步）。ratio：

$$
\rho = \frac{\pi_{\theta_{\text{new}}}}{\pi_{\theta_{\text{old}-k}}} = \exp(\log\pi_{\theta_{\text{new}}} - \log\pi_{\theta_{\text{old}-k}})
$$

故障恢复后续算 ratio，必须保留每条 trajectory 的 $k$（版本号）——否则不知该条数据用哪个 $\theta_{\text{old}}$ 算的、ratio 分母错。若恢复时 staleness 计数丢，trainer 可能用错版本号算 ratio（如把 $k=3$ 当 $k=1$），ratio 失真 → gradient 有偏（[[rollout train reference logprob一致性]] 的 IS 失真在异步 + 故障的叠加）。

**一致性要求**：ckpt 存每条 buffer entry 的 $\theta\_version$（$k$），恢复后 trainer 读 $k$ 续算 ratio。这是异步 RL ckpt 的特有项（SFT 无 staleness）。

### 4.4 弹性重排的通信开销

$n$ 节点故障后重排为 $n'$ 节点（$n' < n$）。DP/TP rank 重排，model 需 reshard（FSDP shard 重新切分到 $n'$ 卡）。reshard 开销：

$$
T_{\text{reshard}} \approx \frac{D_{\text{model}}}{B_{\text{all2all}}} \cdot \log_2 n'
$$

（all-to-all reshard，$D_{\text{model}}$ model 大小，$B_{\text{all2all}}$ 聚合带宽）。重排后从 ckpt resume。弹性适合故障偶发场景；若频繁故障，重排开销累积。详见 [[elastic training]]、[[distributed checkpoint]]。

### 4.5 期望故障损失

设故障率 $\lambda$（次/step）、每次故障损失 $L = T_{\text{recover}} + T_{\text{saved\_lost}}$（恢复时间 + 未 ckpt 数据的损失）。期望每 step 故障损失：

$$
\mathbb{E}[\text{loss/step}] = \lambda \cdot \frac{d_{\text{avg}}}{S} \cdot L \quad (d_{\text{avg}} = S/2 \text{ 平均})
$$

降 $\lambda$（更稳的硬件/网络）、降 $d_{\text{avg}}$（更频繁 ckpt）、降 $L$（buffer 免重采）三者降期望损失。RL 因 $L$ 大（trajectory 贵）故对 $\lambda$ 敏感——同样的故障率，RL 损失比 SFT 大，更需 buffer 容错。


## 5. 代码示例（可选）

### 5.1 RL checkpoint save/load（含 RL 特有状态）

```python
import torch, os, json, pickle

def save_rl_ckpt(path, step, actor, actor_opt, ref, critic, critic_opt,
                 lr_sched, replay_buffer, rollout_progress, staleness_counter,
                 rng_state):
    """存完整 RL ckpt: model + optimizer + RL 特有状态"""
    os.makedirs(path, exist_ok=True)
    # 1. 权重 (用 distributed checkpoint / FSDP 的原生保存, 见 [[distributed checkpoint]])
    torch.save({"step": step,
                "actor": actor.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "ref": ref.state_dict(),
                "critic": critic.state_dict() if critic else None,
                "critic_opt": critic_opt.state_dict() if critic_opt else None,
                "lr_sched": lr_sched.state_dict()},
               os.path.join(path, "models.pt"))
    # 2. replay buffer (trajectory 列表, 含 theta_version/old_logp)
    with open(os.path.join(path, "buffer.pkl"), "wb") as f:
        pickle.dump(replay_buffer, f)
    # 3. rollout 进度 + staleness + scheduler 队列
    with open(os.path.join(path, "rl_state.json"), "w") as f:
        json.dump({"rollout_progress": rollout_progress,
                   "staleness_counter": staleness_counter}, f)
    # 4. RNG state (可复现)
    torch.save({"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "np": __import__("numpy").random.get_state()},
               os.path.join(path, "rng.pt"))

def load_rl_ckpt(path, actor, actor_opt, ref, critic, critic_opt, lr_sched):
    """从 ckpt 恢复完整 RL 状态"""
    m = torch.load(os.path.join(path, "models.pt"), map_location="cpu")
    actor.load_state_dict(m["actor"]); actor_opt.load_state_dict(m["actor_opt"])
    ref.load_state_dict(m["ref"])
    if critic: critic.load_state_dict(m["critic"]); critic_opt.load_state_dict(m["critic_opt"])
    lr_sched.load_state_dict(m["lr_sched"])
    with open(os.path.join(path, "buffer.pkl"), "rb") as f:
        buffer = pickle.load(f)
    with open(os.path.join(path, "rl_state.json")) as f:
        rl_state = json.load(f)
    rng = torch.load(os.path.join(path, "rng.pt"))
    torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state_all(rng["cuda"])
    __import__("numpy").random.set_state(rng["np"])
    return m["step"], buffer, rl_state["rollout_progress"], rl_state["staleness_counter"]
```

### 5.2 replay buffer 重放重训（故障后免重采）

```python
def replay_and_retrain(buffer, actor, actor_opt, steps_to_redo):
    """故障后从 buffer 重放 trajectory 重训, 不重采"""
    for _ in range(steps_to_redo):
        batch = sample_from_buffer(buffer, batch_size=32)   # 取已采 trajectory
        for entry in batch:
            # 用当前 θ 重算 new_log_prob (observation 回放, 见 [[sandbox与轨迹回放]] §5.2)
            new_logp = recompute_logp(actor, entry["trajectory"])
            # ratio 用 entry 的 theta_version (staleness k) 算
            ratio = torch.exp(new_logp - entry["old_log_prob"])  # π_θ / π_{θ_old-k}
            advantage = entry["advantage"]   # 已算好的 (group-relative)
            loss = -(torch.clamp(ratio, 1-0.2, 1+0.2) * advantage).mean()  # PPO clip
            loss.backward(); actor_opt.step()
```

### 5.3 Ray actor 故障重启（verl/OpenRLHF 式）

```python
import ray

@ray.remote(num_gpus=1, max_restarts=3)   # 挂了自动重启 3 次
class RolloutWorker:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.load_ckpt()   # 重启时从 ckpt 恢复 (actor 重启 = 重新 init)
    def load_ckpt(self):
        if os.path.exists(self.ckpt_path):
            self.state = load_rl_ckpt(self.ckpt_path, ...)
        else:
            self.state = init_fresh()
    def rollout(self, prompts):
        # 从 self.state["rollout_progress"] 续采
        return do_rollout(prompts, self.state)

# driver: worker 挂了 Ray 自动重启 + 从 ckpt 恢复, 不需手动处理
worker = RolloutWorker.remote("/ckpt/step_1000")
trajs = ray.get(worker.rollout.remote(prompts))   # 若中途挂, Ray 重启 worker 续
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[RL角色拓扑]]（多角色架构决定 ckpt 要存哪些角色）、[[colocated与disaggregated部署]]（disaggregate 多池故障面大）、[[distributed checkpoint]]（底层 DTensor/FSDP 权重保存机制）、[[elastic training]]（弹性重排）、[[Ray与分布式调度]]（Ray actor 重启）、[[rollout worker]]（rollout 状态恢复）、[[asynchronous training]]（异步 + buffer = 天然容错）、[[synchronous与asynchronous rollout]]（异步容忍旧数据 = 容错）。
- **下游（应用）**: [[17-RL训推一体框架]] 容错能力、[[多轮Agent rollout与tool calling]]（多轮 trajectory 更长更贵，故障恢复要续 sandbox 状态）、[[sandbox与轨迹回放]]（多轮 observation 回放支持 ckpt 恢复后的 context 一致）、[[RL权重同步]]（sync 权重保证 trainer 与 rollout 一致，与 ckpt 配合）。
- **对比 / 易混**:
  - **[[distributed checkpoint]] vs 本条**：前者底层权重保存（DTensor shard 存盘），本条 RL 层面恢复语义（恢复哪些 RL 特有状态、buffer 怎么重放、staleness 怎么续）。底层 + 上层语义。见 §1。
  - **SFT ckpt vs RL ckpt**：SFT 只 model+optimizer；RL 多 ref/critic/buffer/scheduler/staleness/env。见 §3.2。
  - **[[elastic training]] vs 本条**：弹性是节点故障剔除重排（world_size 动态），本条是 RL 状态恢复（含弹性作为手段之一）。弹性是本条的"节点层"手段。
  - **replay buffer（本条容错）vs importance sampling buffer（[[importance sampling与off-policy correction]]）**：同一 buffer 两面——容错视角是"故障后重放免重采"，off-policy 视角是"旧数据训新 θ 靠 ratio 修正"。一个讲容错、一个讲数据复用的数学合法。
- **三篇互链**: [[多轮Agent rollout与tool calling]]（多轮 trajectory 故障恢复要续 env 状态）、[[sandbox与轨迹回放]]（observation 回放支持 ckpt 恢复后 context 一致）。


## 7. 常见误区与易错点

> [!warning] 误区 1：RL ckpt 只存 model+optimizer 就够
> 错。RL 还要存 ref/critic（若 PPO）、replay buffer、rollout 进度、scheduler 队列、staleness 计数、RNG、sandbox 状态（多轮）。只存 model+optimizer 恢复后：ref 丢了 KL 重来、buffer 丢了重采（极贵）、staleness 丢了 ratio 算错、RNG 丢了不可复现。RL ckpt = SFT ckpt + 一堆 RL 特有状态。见 §3.1。

> [!warning] 误区 2：故障后丢弃已采 trajectory 重采
> 浪费。trajectory 采占墙钟 50-80%，重采极贵。应存 replay buffer，故障后从 buffer 重放重训，不重采。buffer 持久化是 RL 容错的核心价值。见 §3.3、§4.2。

> [!warning] 误区 3：异步训练 staleness 计数不用存
> 错。异步每条 trajectory 有 θ 版本号 $k$，ratio $\rho=\pi_{\theta_{new}}/\pi_{\theta_{old-k}}$ 依赖 $k$。故障恢复若 $k$ 丢，trainer 用错版本号算 ratio → IS 失真 → gradient 有偏（[[rollout train reference logprob一致性]] 在异步+故障的叠加）。staleness 是异步 RL ckpt 的特有必存项。见 §4.3。

> [!warning] 误区 4：ckpt 太频繁浪费算力 / 太稀疏故障损失大
> 两难。ckpt 频繁（$S$ 小）降重做开销但 ckpt 本身占开销（每 $S$ step 存一次）；稀疏（$S$ 大）ckpt 开销小但故障重做多。典型 $S$=10-100 step，异步可在 trainer 闲时异步存降开销。见 §4.1。

> [!tip] 实践：异步 + 持久化 buffer 是大规模 RL 容错甜区
> 同步训练一挂全停、损失大；异步 + buffer 持久化让"某 worker 挂，trainer 用 buffer 旧数据继续"成为可能，天然容错 + 不浪费已采数据。代价是 staleness 需 IS 修正，clip 兜底。AReaL/verl fully_async 选异步的容错动机之一。见 §3.6、§4.5。

> [!tip] 实践：ckpt 用分布式异步存，不阻塞 rollout
> ckpt 保存 $D/B$ 若同步阻塞 rollout，浪费 rollout 算力。应异步存（trainer 存时 rollout 继续采）、或用 [[distributed checkpoint]] 的异步保存。verl/OpenRLHF 的 ckpt 多异步存。见 §4.1。

> [!note] 补充：RNG 恢复保证可复现
> RNG state 存 ckpt 是为保证故障恢复后**续训与不挂时一致**——数据采样顺序、dropout 模式、采样温度序列一致，gradient 可复现。这对调试（对比挂/不挂的 gradient）与严格复现重要。SFT 也建议存，RL 因数据在线采更必须。见 §3.4。


## 8. 延伸细节

### 8.1 verl/OpenRLHF 的 ckpt 机制

verl、OpenRLHF 的 ckpt 在 model ckpt 基础上扩展 RL 状态（以最新官方文档/代码为准，部分待核实）：

- **verl**：基于 FSDP 的 model ckpt（[[distributed checkpoint]]），扩展存 ref/critic、replay buffer、rollout 进度。Ray 驱动的 single-controller，worker 故障由 Ray `max_restarts` 自动重启 + 从 ckpt 恢复。多轮 rollout 的 sandbox/env 状态随 trajectory 存 buffer；
- **OpenRLHF**：Ray + DeepSpeed/FSDP，ckpt 存多角色权重 + buffer。`--save_freq` 控制 ckpt 频率。异步/partial 模式下 staleness 计数随 buffer entry 存；
- **AReaL/AsyncFlow**：fully async，buffer 是异步解耦核心，ckpt 异步存（不阻塞 rollout），容错靠 buffer 持久化 + Ray actor 重启。

具体 API（`save_checkpoint`、`load_checkpoint`、buffer 类名、配置项）随版本变动，建议查最新官方文档。

### 8.2 多轮 Agent rollout 的故障恢复

多轮 trajectory 更长更贵，故障恢复要续 **sandbox/env 状态** 与 **observation 回放**（[[sandbox与轨迹回放]]）：

- **env 状态**：多轮 trajectory 采到一半，env 状态（代码沙箱的全局变量、文件、会话）要能续。若 sandbox 崩了，状态丢——要么 graceful 降级（崩步返 ERROR observation 续）、要么该 trajectory 丢弃（从 buffer 删，重采）。续采需 sandbox 状态恢复，复杂；常不如重采该条；
- **observation 回放**：buffer 里多轮 trajectory 的 observation 文本要存全（[[sandbox与轨迹回放]] §5.2），故障恢复后 trainer 重算 logp 时能回放 observation 保 context 一致；
- **trajectory 部分失败**：某 trajectory 崩了一步，若 graceful 降级继续（observation=ERROR），trajectory 完整可训；若崩断（sandbox 死、不可续），trajectory 不全，丢弃（类似 Dynamic Sampling 过滤无信号样本）。

详见 [[多轮Agent rollout与tool calling]] §8、[[sandbox与轨迹回放]] §8.5。

### 8.3 异步训练的容错优势

异步训练（[[asynchronous training]]、[[synchronous与asynchronous rollout]]）的容错优势：

- **不阻塞全局**：某 rollout worker 慢/挂，trainer 用其他 worker / buffer 的数据继续训，不 全停；
- **buffer 持久化兜底**：异步下 buffer 是数据池，持久化后 trainer 端故障也能从 buffer 重放；
- **staleness 即容错**：容忍旧数据 = 容忍慢/挂的 worker，staleness $k$ 大需 IS 修正但 clip 兜底。

代价：staleness $k$ 大 → ratio 偏离需 [[importance sampling与off-policy correction]]；异步调试更难（多版本数据混）。大规模长训练（weeks）选异步的容错动机强。详见 [[asynchronous training]]、[[synchronous与asynchronous rollout]]。

### 8.4 ckpt 的存储后端

ckpt 存到哪影响恢复速度与可靠性：

| 后端 | 速度 | 可靠性 | 适用 |
|---|---|---|---|
| **本地 NVMe** | 快（GB/s） | 单机故障丢 | 临时/快 ckpt |
| **并行文件系统**（Lustre/GPFS） | 中（多机共享） | 高（冗余） | 生产级共享 ckpt |
| **对象存储**（S3/OSS） | 慢（网络） | 极高 | 异地备份/长期 |
| **内存/分布式内存** | 极快 | 易失（重启丢） | 高频增量 ckpt |

常见：高频增量 ckpt 存本地 NVMe/内存（快恢复），低频全量 ckpt 存并行 FS/S3（可靠备份）。详见 [[distributed checkpoint]] 的存储后端。

### 8.5 故障检测与自动恢复

故障检测手段：

- **heartbeat**：worker 周期发心跳，driver 超时未收判挂；
- **NCCL 超时**：集合通信超时（`NCCL_TIMEOUT`）判某 rank 挂；
- **Ray actor health**：Ray 监控 actor 状态，挂了自动重启（`max_restarts`）；
- **process exit**：worker 进程退出，driver 收 SIGCHLD。

自动恢复：检测故障 → 剔除故障节点 → 重排 world_size → 从最近 ckpt resume → 续训。verl/OpenRLHF 的 single-controller + Ray 让这流程相对自动化（Ray 管 actor 生命周期）。详见 [[Ray与分布式调度]]、[[elastic training]]。

### 8.6 与监督学习容错的统一视角

RL 容错 = SFT 容错 + RL 特有状态恢复。SFT 的 model/optimizer/scheduler/RNG ckpt 是基础，RL 在此上加：ref/critic/buffer/scheduler队列/staleness/env。把 SFT 容错框架（[[distributed checkpoint]]、[[elastic training]]）作为底层，RL 层面补"恢复哪些 RL 状态、buffer 怎么重放、staleness 怎么续"的语义。统一视角：RL 容错是 SFT 容错的多角色、有状态数据、异步版本号的扩展。详见 [[distributed checkpoint]]、[[elastic training]]。

RL ckpt 内容与恢复语义综合自 verl/OpenRLHF/AReaL 的容错设计（多角色 ckpt、buffer 持久化、Ray actor 重启）与 [[RL角色拓扑]] §3（多角色架构）、[[synchronous与asynchronous rollout]] §3（异步 + staleness）。replay buffer 重放与 IS ratio 的关系基于 [[importance sampling与off-policy correction]] 的 off-policy 数据复用。staleness 续算的 ratio 一致性基于 [[rollout train reference logprob一致性]] §4 的 IS 体系。弹性重排与 torchrun --rdzv / Ray 基于 [[elastic training]]、[[Ray与分布式调度]]。多轮 env 状态恢复基于 [[sandbox与轨迹回放]] §8.5。ckpt 存储后端与分布式保存基于 [[distributed checkpoint]]。具体框架 API 随版本变动，已标注待核实项建议查最新官方文档。截至 2026-07。

---
相关: [[容错]] | [[多轮Agent rollout与tool calling]] | [[sandbox与轨迹回放]] | [[RL角色拓扑]] | [[colocated与disaggregated部署]] | [[synchronous与asynchronous rollout]] | [[asynchronous training]] | [[distributed checkpoint]] | [[elastic training]] | [[Ray与分布式调度]] | [[rollout worker]] | [[RL权重同步]] | [[importance sampling与off-policy correction]] | [[rollout train reference logprob一致性]] | [[17-RL训推一体框架]]
