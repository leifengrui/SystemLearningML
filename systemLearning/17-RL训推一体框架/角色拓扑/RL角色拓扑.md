# RL角色拓扑

> **所属章节**: [[角色拓扑]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: RLHF 角色拓扑 / PPO 多角色 / RL worker 拓扑 / actor-rollout-reference-critic-reward / RLHF 的五个模型
> **难度**: 高（需懂 [[RLHF (PPO)]]、[[GRPO]]、[[Actor-Critic]]、[[Reward Model]]、[[value network]]、[[训推分离]]、[[weight sync mechanism]]）


## 1. 一句话定义

**RL 角色拓扑** 是 LLM 强化学习（[[RLHF (PPO)]] / [[GRPO]]）中**多个神经网络"角色"如何分工、如何互联、如何部署**的体系——一次 RLHF 训练不是"一个模型自练自"，而是 **actor（被训练的策略网络）、rollout（采样生成 response 的引擎）、reference（冻结的 $\pi_{ref}$，算 KL 防 reward hacking）、critic（value 网络，估 baseline）、reward model（偏好数据训的 $r_\varphi$ 或 rule-based verifier）** 这 5 类角色协同：rollout 用 actor 的当前权重采出 trajectory → reward model 给 outcome/过程 reward → reference 重算 $\log\pi_{ref}$ 供 KL 约束 → critic 估 $V$ 供 advantage（GRPO 去掉 critic，用 group-mean 当 baseline）→ actor 用 advantage + PPO clip + KL 做策略更新。各角色可**同进程同卡 colocate**（verl 的 `actor_rollout_ref`）、可**同机分进程**、可**跨机独立 GPU 池 disaggregate**——拓扑选型直接决定显存/通信/吞吐/稳定性。是 [[17-RL训推一体框架]] 的"骨架"，[[colocated与disaggregated部署]] 与 [[trainer与rollout引擎组合]] 是它在"部署维度"与"引擎维度"的两个正交展开。

> [!note] 三句话定位
> - **是什么**：RLHF/GRPO 不是单模型，而是 actor / rollout / reference / critic / reward model 五角色的"多模型编排"，每角色有明确输入输出数据流与时序。
> - **为什么**：策略梯度需要"采样的行为策略"与"更新的目标策略"分离（off-policy + ratio 修正），需要 reference 防 drift hacking，需要 baseline/critic 降方差，需要 RM 把人类偏好转标量——拆成多角色各司其职、可独立并行/部署/扩缩。
> - **与 [[trainer与rollout引擎组合]] / [[colocated与disaggregated部署]] 关系**：本篇讲"有哪些角色、数据怎么流"；后两篇讲"这些角色用什么引擎跑（训练框架 vs 推理框架）"和"这些角色放同机还是跨机"——三个维度正交，组合出 verl/OpenRLHF/HybridFlow 的具体部署。


## 2. 为什么需要它（动机与背景）

### 2.1 一个模型干不了 RLHF

朴素的"自练自"想象：拿一个 LLM，让它自己生成、自己打分、自己更新。这立刻撞墙：

1. **打分要另一个网络**：人类偏好是排序数据（A 比 B 好），不能直接当 loss，需训一个 [[Reward Model]] $r_\varphi$ 把"偏好"映射成标量 reward。actor 自己既当策略又当裁判等于无监督。→ 需要 **reward model 角色**。
2. **采样要推理优化**：生成 response 是几百~几千 token 的 autoregressive forward，用训练框架（带梯度/优化器状态）裸跑采样吞吐极低（无 KV cache/continuous batching/CUDA Graph）。→ 需要 **rollout 角色**（用推理引擎，见 [[trainer与rollout引擎组合]]）。
3. **更新要 reference 防漂**：actor 只追 reward 最大化会 reward hacking（钻 RM 漏洞输出高分废话）。必须约束 $\pi_\theta$ 不离初始 $\pi_{ref}$ 太远。→ 需要 **reference 角色**（冻结的 SFT 模型，算 $\log\pi_{ref}$）。
4. **降方差要 baseline**：策略梯度方差大，需减一个不依赖 action 的 baseline。PPO 用学出来的 $V_\varphi$（[[value network]]）；GRPO 用 group-mean 代替。→ PPO 需要 **critic 角色**，GRPO 不需要。
5. **行为策略 ≠ 目标策略**：rollout 用的权重 $\theta_{old}$ 与更新时的 $\theta$ 不同步（off-policy），需 importance ratio $\rho=\pi_\theta/\pi_{old}$ 修正（[[importance sampling与off-policy correction]]）。→ rollout 与 actor 必须是"不同时刻的权重"，是时间分离的两角色。

### 2.2 五角色各有不可替代的职责

| 角色 | 网络 | 是否训练 | 核心职责 | 不可替代原因 |
|---|---|---|---|---|
| **actor** | $\pi_\theta$（policy 网络） | ✅ 唯一被梯度更新 | 被训练的策略，forward 算 logp、backward 更新 $\theta$ | RL 的目标就是优化它，它是唯一"终点" |
| **rollout** | actor 的权重副本（推理态） | ❌（只采样） | 用 $\theta_{old}$ 采 response，产 trajectory | 需推理优化（KV cache/batching），训练框架做不来 |
| **reference** | $\pi_{ref}$（冻结的 SFT 权重） | ❌（冻结） | 算 $\log\pi_{ref}(o\|q)$ 供 KL 约束 | 没 ref 则 actor 无约束漂移、reward hacking |
| **critic** | $V_\varphi$（value 网络） | ✅（TD/GAE 更新） | 估 $V(s)$，算 GAE advantage baseline | PPO 降方差必需；GRPO 用 group-mean 替代故无 |
| **reward model** | $r_\varphi$（偏好数据训） | 通常冻结（训练时另训） | 给 response 打分 $r=R(q,o)$ | 把人类偏好转标量；或 rule-based verifier |

### 2.3 拓扑选型即工程命脉

五角色怎么放——同进程同卡？同机分进程？跨机独立池？——直接决定：

- **显存**：colocate 多角色同卡可复用权重/分时复用显存省卡，但挤；disaggregate 各自独立池不挤但贵。
- **通信**：colocate 的 weight sync 走 in-process/同机 NCCL 极快；disaggregate 走跨机 NCCL/IPC/共享 FS 有开销与 staleness 风险（[[stale policy problem]]）。
- **吞吐**：rollout 与 training 能否 overlap 决定墙钟时间；同步拓扑下串行慢，异步拓扑 overlap 快但 stale。
- **稳定性**：colocate 资源争用致 GPU 利用率波动、OOM 风险；disaggregate 隔离稳定但版本管理复杂。

这正是 [[colocated与disaggregated部署]] 与 [[trainer与rollout引擎组合]] 要展开的，本篇先定"角色是什么、数据怎么流"。


## 3. 核心概念详解

### 3.1 五角色的定义与职责

#### actor（策略网络 $\pi_\theta$，被训练者）
RL 的优化目标。它持有**可训练权重 $\theta$**，承担两个动作：
- **forward 算 logp**：给定 batch 的 $(q,o)$，算 $\log\pi_\theta(o\|q)$（逐 token logp），供 PPO ratio $\rho=\exp(\log\pi_\theta-\log\pi_{old})$ 与 KL 用。
- **backward 更新 $\theta$**：用 PPO clipped objective（或 GRPO 目标）+ KL penalty 算梯度，optimizer step 更新 $\theta$。

训练态用训练框架（[[Fully Sharded Data Parallel]] / [[Megatron-LM]]，管梯度/优化器状态/并行）。是**唯一被梯度更新的角色**。actor 的权重 $\theta$ 训练后要同步给 rollout（见 §3.3 数据流）。

#### rollout（采样引擎，用 $\theta_{old}$ 生成）
用 actor 的**当前权重副本**（推理态）对 prompt $q$ 采 response $o\sim\pi_{\theta_{old}}(\cdot\|q)$。是**墙钟时间大头**（LLM 生成几百~几千 token forward × 成千上万条）。所以 rollout 用**推理引擎**（vLLM/SGLang，有 [[continuous batching的调度实现]]/KV cache/CUDA Graph/prefix caching），而不是训练框架裸跑——详见 [[trainer与rollout引擎组合]]。

> [!note] rollout 与 actor 是"同一权重不同时刻"
> rollout 用的 $\theta_{old}$ 是 actor 上一次同步过来的权重。一次 batch 里 rollout 用 $\theta_{old}$ 采、actor 用 $\theta$ 算 ratio，两者权重不同步正是 off-policy 修正的由来。训练完一个 batch 后 actor 把新 $\theta$ 同步回 rollout，下一 batch rollout 用更新的 $\theta_{old}$。

#### reference（冻结 $\pi_{ref}$，算 KL）
SFT 后的初始模型，**冻结不训练**。对 batch 的 $(q,o)$ 算 $\log\pi_{ref}(o\|q)$，供 KL penalty：

$$
\text{KL}_t \approx \log\frac{\pi_{ref}(o_t\|q,o_{<t})}{\pi_{\theta_{old}}(o_t\|q,o_{<t})}
$$

（用 $\theta_{old}$ 的 logp，是 off-policy 近似；详见 [[KL penalty与KL control]]、[[rollout train reference logprob一致性]]）。reference 存在的唯一目的：**防 actor 漂离人类意图太远、防 reward hacking**。ref 与 actor 同架构同初始权重，只是不再更新。

#### critic（value 网络 $V_\varphi$，PPO 专属）
学一个 token 级 value 函数 $V_\varphi(s_t)$（$s_t$="已生成 token 序列"），估"当前这串 token 之后还能拿多少 reward"。用 TD/GAE loss 训练（[[value network]]、[[GAE]]）。算 advantage：

$$
A_t = \sum_{l\ge 0}(\gamma\lambda)^l\delta_{t+l},\quad \delta_t = r_t + \gamma V_\varphi(s_{t+1}) - V_\varphi(s_t)
$$

是 PPO 降方差的 baseline。**GRPO 用 group-mean 代替它**（见 §3.5），故 GRPO 无 critic 角色——省一整个网络的显存与算力，是 GRPO 相对 PPO 的核心工程优势。

#### reward model（$r_\varphi$ 或 rule-based verifier）
给一条 response 打标量 reward $r=R(q,o)$。两条来源：
- **学习式 RM**：用人类偏好数据（chosen/rejected pair）按 Bradley-Terry / DPO-style loss 训一个 $r_\varphi$，输入 $(q,o)$ 输出标量。适用于通用对话/开放任务（reward 无法直接验证）。
- **rule-based verifier**：数学题判 `\boxed{}` 对错、代码题跑单测 pass/fail、格式约束正则。reward 稀疏但可验证。GRPO/R1 范式甜区。

RM 一般**训练时另训**，RL 训练中冻结当 reward 函数用。也可两者混合（rule + RM 打分）。

### 3.2 各角色的输入输出数据流

一次 RL batch 的数据流（PPO 版，GRPO 去 critic 分支）：

```
                  ┌─────────────────────────────────────────────────────┐
prompts q ───────►│ rollout (θ_old)                                     │──► responses o, logp_old
                  │   o ~ π_θ_old(·|q), 算 logπ_old(o|q)                │
                  └─────────────────────────────────────────────────────┘
                                                      │
              ┌───────────────────────────────────────┴──────────────────────────┐
              ▼                                                                    ▼
  ┌─────────────────────────┐                                      ┌─────────────────────────┐
  │ reward model (frozen)   │──► r = R(q,o)  (outcome/过程 reward) │ reference (π_ref 冻结)  │──► logπ_ref(o|q)
  └─────────────────────────┘                                      └─────────────────────────┘
              │                                                                    │
              │              (PPO only) ┌──────────────────┐                       │
              └────────────────────────►│ critic (V_φ)     │──► V(s_t)  per token  │
                                        │ forward + train │                       │
                                        └──────────────────┘                       │
              │                          │                  │                       │
              ▼                          ▼                  ▼                       ▼
         reward r                   V(s_t)            ───────────────────────────────
              │                          │            │  在 driver 本地组装  │
              └──────────────────────────┴────────────►│  advantage A_t      │ (GAE 或 group-mean)
                                                       │  KL_t (用 ref logp) │
                                                       │  ratio ρ (用 θ logp)│
                                                       └─────────┬───────────┘
                                                                 │
                                                                 ▼
                                          ┌──────────────────────────────────────┐
                                          │ actor (π_θ)  被训练                    │
                                          │  forward: logπ_θ(o|q)                  │
                                          │  loss = PPO-clip(ρ·A) + β·KL          │
                                          │  backward: 更新 θ                      │
                                          └──────────────────────────────────────┘
                                                                 │
                                                                 ▼
                                                  weight sync: θ → rollout (下一 batch 用 θ_old=θ)
```

**逐角色输入输出表**：

| 角色 | 输入 | 输出 | 给谁用 |
|---|---|---|---|
| rollout | prompts $q$，权重 $\theta_{old}$ | responses $o$，$\log\pi_{old}(o\|q)$ | RM 打分；actor/ref/critic 算 logp/value；driver 算 ratio/adv |
| reward model | $(q,o)$ | scalar $r$ | driver 组 reward（含 in-reward KL）；critic 算 return；actor 间接用 advantage |
| reference | $(q,o)$ | $\log\pi_{ref}(o\|q)$ | driver 算 KL penalty |
| critic | $(q,o)$ token 序列 | $V_\varphi(s_t)$ 逐 token | driver 算 GAE advantage；critic 自身 TD loss |
| actor | $(q,o,\hat A,\rho,\text{KL})$ | 梯度 → 更新 $\theta$ | 唯一终点；新 $\theta$ weight sync 给 rollout |

> [!note] old_log_prob 的角色
> rollout 采样时算的 $\log\pi_{old}$ 是 **PPO/GRPO 的 off-policy 锚点**——ratio $\rho=\exp(\log\pi_\theta-\log\pi_{old})$ 全靠它。但训练框架重算 logp 与 rollout（推理框架）算的 logp **可能不一致**（TIM 训推不一致，见 [[rollout train reference logprob一致性]]），不一致会让 $\rho$ 失真、训练崩。这是 rollout 与 actor 分离引擎引入的核心风险，[[trainer与rollout引擎组合]] §3 详述。

### 3.3 weight sync：actor → rollout 的动脉

训练完一个 batch，actor 的新 $\theta$ 要同步给 rollout（下 batch 用更新权重采样）。这是 RL 训推一体里**最频繁的跨角色通信**：

- **colocate 场景**：actor 与 rollout 同卡，weight sync 走 in-process（verl 的 `update_weights`）/同机 NCCL/CUDA IPC，毫秒~微秒级。verl 还会 `sleep_replicas()` 先把 rollout 副本显存让给训练 forward，训完再 `update_weights` 推回——详见 [[colocated与disaggregated部署]]、[[weight sync mechanism]]。
- **disaggregate 场景**：rollout 独立 GPU 池，weight sync 走跨机 NCCL broadcast / 共享 FS delta pull / checkpoint engine，开销大且引入 staleness。详见 [[weight sync mechanism]]、[[RL权重同步]]。

### 3.4 角色拓扑图：哪些同进程 / 同卡 / 跨机

RLHF 部署的本质是"哪些角色物理共置省显存、哪些分开求隔离"。三种典型拓扑：

```
拓扑 A: 全 colocate（verl actor_rollout_ref 模式，省卡）
┌──────────────── 同一台机/同一组 GPU ────────────────┐
│  [actor + rollout + reference]  共一进程/同卡组分时   │   ← actor_rollout_ref
│  [critic]                        独立 group 或也 colocate│   ← critic_wg
│  [reward model]                  独立 group 或 colocate  │   ← reward_loop
└────────────────────────────────────────────────────────┘
  优点: weight sync in-process 极快、显存分时复用省卡
  缺点: 资源争用、显存挤、role 切换 GPU 利用率波动

拓扑 B: 全 disaggregate（各角色独立 GPU 池）
┌─actor 池─┐  ┌─rollout 池─┐  ┌─ref 池─┐  ┌─critic 池─┐  ┌─RM 池─┐
│  跨机     │  │  跨机      │  │ 跨机   │  │  跨机     │  │ 跨机  │
└─────────┘  └───────────┘  └────────┘  └───────────┘  └───────┘
  weight sync 走跨机 NCCL/FS，开销大、staleness 风险；但资源独立扩缩、不争用

拓扑 C: hybrid（混合，主流）
  trainer+ref colocate 同卡组（ref 只算 forward 可分时）；rollout 独立池（按推理算力配比）；critic/RM 按需 colocate 或独立
```

verl 用 `resource_pool_manager` + `role_worker_mapping` + `create_colocated_worker_cls` 灵活配——把 `Role.ActorRolloutRef` 三角色塞同一 Ray actor 进程（物理共置省显存），`spawn` 拆出逻辑 worker group（见 [[Ray与分布式调度]] §3.5）。这正是 [[colocated与disaggregated部署]] 的核心，本篇只点拓扑图，细节在那篇。

### 3.5 GRPO vs PPO 的角色差异：去 critic

GRPO（[[GRPO]]）相对 PPO（[[RLHF (PPO)]]）在拓扑上的唯一结构性差异：**砍掉 critic 角色**。

| 维度 | PPO（RLHF） | GRPO |
|---|---|---|
| 训练时同屏角色 | actor + rollout + ref + **critic** + rm | actor + rollout + ref + rm（**无 critic**） |
| baseline 来源 | 学出来的 $V_\varphi(s)$（critic forward） | group-mean $\bar r=\frac1G\sum r_j$（同 prompt 多采比较） |
| advantage | GAE：$A_t=\sum(\gamma\lambda)^l\delta_{t+l}$，token 级 | z-score：$\hat A_i=(r_i-\bar r)/\sigma_r$，sequence 级 |
| critic 显存 | ~actor 量级（同 size + optim states） | 0 |
| critic 算力 | 每条 response 全序列 forward 算 token value | 0 |
| critic 训练 step | 有（critic loss + TD/GAE） | 无 |
| 代价 | group 采 $G$ 条 response（rollout 量 ×$G$） | 单采即可但需训 critic |

```
PPO  拓扑: actor ─ rollout ─ ref ─ critic ─ rm   (5 角色)
GRPO 拓扑: actor ─ rollout ─ ref ─ ─ ─ ─ ─ ─ rm   (4 角色, critic 砍掉, rollout 量×G 补)
```

> [!warning] 误区：GRPO 不需要 ref
> 错。GRPO 砍的是 **critic**，不是 reference。GRPO 仍有 KL to $\pi_{ref}$ 约束（DeepSeekMath 用无偏 k3 estimator，见 [[GRPO]] §8.2）。ref 角色在 PPO/GRPO 都在，是"防漂移"的必需。只在 DPO 等"用 ref 一次性离线"的范式里 ref 不在在线训练拓扑，但那不是 PPO/GRPO。

> [!tip] 实践：选 PPO 还是 GRPO 看资源与 reward 性质
> - reward 可 rule-based 验证（数学/代码）→ GRPO 甜区，省 critic 一整网络，多采 $G$ 条换。
> - reward 只能 RM 打分且噪声大、prompt 间不可比 → PPO 的 learned critic 更稳（group-mean 噪声大）。
> - 显存极紧（大模型 RLHF 卡 critic）→ GRPO 省。这也是 DeepSeek-R1 选 GRPO 的工程原因之一。

### 3.6 verl / OpenRLHF / HybridFlow 的角色映射

三框架都实现上述五角色拓扑，抽象层与默认 colocate 策略略不同：

| 框架 | 角色抽象 | 默认 colocate | 备注 |
|---|---|---|---|
| **verl**（ByteDance，HybridFlow 论文 arXiv:2409.19256 的开源实现） | `Role.ActorRolloutRef`（actor+rollout+ref 塞同进程）/ `Role.Critic` / `Role.RefPolicy` / `Role.RewardModel`；driver=`RayPPOTrainer` 持各 `RayWorkerGroup` | actor_rollout_ref 共置，critic/ref/RM 可独立池 | `create_colocated_worker_cls`+`spawn`：物理同进程省显存、逻辑多 wg；GRPO 时 `use_critic=False` 跳过 critic 分支。见 [[Ray与分布式调度]] §3.5 |
| **OpenRLHF**（OpenLLMAI） | `RayPPOActor`/`RayPPOCritic`/`RayRefModel`/`RayRewardModel`/`RayvLLMRollout`；各为独立 Ray actor 类 | 支持 colocate 模式（actor+ref+critic 同卡分时 sleep/wake）与 hybrid 模式（rollout 独立 vLLM 副本） | `--vllm.sync_backend nccl/IPC`；colocate 时 sleep/wake 切换显存，weight sync 走 NCCL/IPC |
| **HybridFlow**（论文） | 统一抽象：control flow（RL 算法）与 distributed execution（role 的并行）解耦；**3D-HybridEngine** 处理 actor 训练态（TP/PP/DP）↔ rollout 推理态（TP）的并行 reshard | actor 同时承担训练+生成（3D-HybridEngine 内 colocate 训推），critic/ref/rm 按 placement 配 | HybridFlow 的核心贡献是"统一抽象 + 3D-HybridEngine"，verl 是其落地。3D-HybridEngine 解决训练态 3D 并行与推理态 TP 的 reshard，详见 [[trainer与rollout引擎组合]] §3.5 |

> [!note] 补充：verl 的 `use_critic` 开关
> verl 的 `RayPPOTrainer.fit()` 里 `if self.use_critic:` 守卫所有 critic 分支（`_compute_values` / `_update_critic`）。GRPO 配置 `algorithm.adv_estimator=grpo`（或 actor_rollout_ref.use_critic=False，具体配置名待核实）时这些分支跳过——同一套拓扑代码同时支持 PPO 与 GRPO，差异只在 critic 分支开关 + advantage 估计器选择。这是 verl/HybridFlow "统一抽象"的体现：换算法不需换拓扑骨架。


## 4. 数学原理 / 公式

### 4.1 各角色的数学定义

- **actor**：策略 $\pi_\theta(o\|q)$，参数 $\theta$。目标 $\max_\theta J(\theta)=\mathbb{E}_{q,o\sim\pi_\theta}[\ldots]$。
- **rollout**：行为策略 $\pi_{\theta_{old}}$（$\theta_{old}$ 是上次同步的权重）。采样 $o\sim\pi_{\theta_{old}}(\cdot\|q)$。
- **reference**：$\pi_{ref}$（冻结）。KL 约束 $\mathbb{D}_{KL}[\pi_\theta\|\pi_{ref}]$。
- **critic（PPO）**：$V_\varphi(s)$，TD target $y_t=r_t+\gamma V_\varphi(s_{t+1})$，loss $\|V_\varphi(s_t)-y_t\|^2$。
- **reward model**：$r_\varphi(q,o)$；偏好训练用 Bradley-Terry $P(o_w\succ o_l)=\sigma(r_\varphi(q,o_w)-r_\varphi(q,o_l))$。

### 4.2 ratio（rollout↔actor 的 off-policy 修正）

rollout 用 $\theta_{old}$ 采、actor 用 $\theta$ 更新，importance ratio：

$$
\rho_t=\frac{\pi_\theta(o_t\|q,o_{<t})}{\pi_{\theta_{old}}(o_t\|q,o_{<t})}=\exp\big(\log\pi_\theta(o_t\|\cdot)-\log\pi_{\theta_{old}}(o_t\|\cdot)\big)
$$

PPO clipped objective：$\mathcal{L}_\theta=\mathbb{E}[\min(\rho_t\hat A_t,\,\text{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\hat A_t)]$。$\rho$ 全靠 rollout 算的 $\log\pi_{old}$ 当锚点——若 rollout（推理框架）与 actor（训练框架）算的 logp 不一致，$\rho$ 失真（[[rollout train reference logprob一致性]]）。

### 4.3 KL（reference 的职责）

in-reward KL penalty（OpenAI/TRL 风格）：$r_t \leftarrow r_t - \beta\cdot\text{KL}_t$，其中

$$
\text{KL}_t \approx \log\frac{\pi_{ref}(o_t\|\cdot)}{\pi_{\theta_{old}}(o_t\|\cdot)}=\log\pi_{ref}(o_t\|\cdot)-\log\pi_{old}(o_t\|\cdot)
$$

（off-policy 近似用 $\theta_{old}$ 而非 $\theta$）。或 explicit KL loss（GRPO 风格）：$\mathcal{L}=\ldots-\beta\cdot\mathbb{D}_{KL}[\pi_\theta\|\pi_{ref}]$，用 k3 无偏估计。$\beta$ 可固定或自适应（verl `AdaptiveKLController`）。无 ref 则无 KL，actor 失约束 → reward hacking。

### 4.4 advantage：critic（PPO）vs group-mean（GRPO）

PPO（GAE，需 critic）：

$$
\hat A_t^{\text{GAE}}=\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l},\quad \delta_t=r_t+\gamma V_\varphi(s_{t+1})-V_\varphi(s_t)
$$

GRPO（group-mean baseline，无 critic）：同 prompt $q$ 采 $G$ 条，

$$
\hat A_i=\frac{r_i-\bar r}{\sigma_r},\quad \bar r=\frac1G\sum_{j=1}^G r_j,\quad \sigma_r=\sqrt{\frac1G\sum_j(r_j-\bar r)^2},\quad \hat A_{i,t}=\hat A_i\ (\forall t)
$$

数学等价性：$\bar r$ 是 $V(q)=\mathbb{E}_{o\sim\pi}[R(q,o)]$ 的 Monte-Carlo 估计 → 不依赖 action 的 baseline → 无偏降方差。GRPO 用"多采样"换"不训 critic"，是 PPO 在可验证 reward 场景的精简。详见 [[GRPO]] §2.3、[[baseline]]、[[variance reduction]]。

### 4.5 weight sync 通信量（actor → rollout）

一次 batch 后 actor 把新 $\theta$ 推给 rollout，通信量：

$$
\text{Vol} = P\cdot\text{size}\cdot(\text{全量}=1\text{ 或增量}=\|\Delta\theta\|_0/P)
$$

- $P$：参数量（70B ≈ $70\times10^9$），size：FP16=2 字节 → 全量 ~140GB。
- colocate 同机：走 NVLink/CUDA IPC，带宽数百 GB/s，毫秒~微秒级。
- disaggregate 跨机：走 IB NCCL broadcast，带宽数十~数百 GB/s，十毫秒级 + staleness。

详见 [[weight sync mechanism]] §4.2。这是"五角色拓扑选型"在通信维度的硬约束——colocate 省这步开销是它吞吐优势的主因。

### 4.6 staleness（disaggregate 的代价）

disaggregate 下 rollout 与 actor 时间解耦，rollout 用 $\theta_{old-k}$（落后 $k$ 步）。ratio 真实值 $\rho=\pi_\theta/\pi_{old-k}$，偏离 1 越多训练越不稳。最大 staleness $k_{\max}\le S$（两次 sync 间隔步数）。需 clip $\varepsilon$ 兜底（维持 clip frequency 0.1~0.3）。详见 [[stale policy problem]]、[[asynchronous training]]、[[weight sync mechanism]] §4.1。


## 5. 代码示例（可选）

### 5.1 verl 角色拓扑配置（概念，配置名以官方文档为准）

```python
# verl 的 ResourcePool + role mapping 决定五角色拓扑（colocate / disaggregate）
# 来自 verl RayPPOTrainer 初始化逻辑（见 Ray与分布式调度 §3.5）
from verl.single_controller.base import Role
from verl.single_controller.ray import RayResourcePool, RayClassWithInitArgs, create_colocated_worker_cls

# 角色→worker 类映射（决定每组 actor 跑什么）
role_worker_mapping = {
    Role.ActorRolloutRef: ActorRolloutRefWorker,   # actor + rollout + ref 塞同进程（colocate 核心）
    Role.Critic:         CriticWorker,              # critic（GRPO 时不用）
    Role.RefPolicy:      RefPolicyWorker,          # 也可独立 ref（若不塞 ActorRolloutRef）
    Role.RewardModel:    RewardModelWorker,        # RM（colocate 或独立）
}

# 资源池：决定各 role 放哪些卡、是否 colocate
resource_pool_spec = {
    "actor_rollout_ref_pool": {"num_gpus": 8, "colocate": True},   # actor+rollout+ref 同 8 卡共置
    "critic_pool":            {"num_gpus": 8, "colocate": False},  # critic 独立池（PPO）
    "reward_pool":            {"num_gpus": 2, "colocate": True},   # RM 与 actor 共置
}
# create_colocated_worker_cls 把同池的多个 role 塞进同一 Ray actor 进程（省显存）
# spawn(prefix_set) 再按 role 拆出逻辑 worker group（各自 RPC）
```

### 5.2 五角色数据流时序（verl fit() 的骨架，来自 [[Ray与分布式调度]] §3.5 解答）

```python
# RayPPOTrainer.fit() 主循环（driver 是"脑子"，actor 是"手脚"）
for batch in train_dataloader:
    # 1) rollout 采样（θ_old，推理引擎 vLLM/SGLang）
    gen = self.async_rollout_manager.generate_sequences(batch)   # o ~ π_θ_old
    self.checkpoint_manager.sleep_replicas()      # 释放 rollout 副本显存给训练 forward
    batch = batch.union(gen)                       # 含 response + logp_old

    # 2) reward model 打分
    if self.use_rm:
        batch = batch.union(self._compute_reward_colocate(batch))   # r = R(q,o)

    # 3) actor 重算 old_log_prob（PPO 锚点，训练框架 forward）
    old_log_prob = self._compute_old_log_prob(batch)               # logπ_θ(o|q)
    batch = batch.union(old_log_prob)

    # 4) reference 算 logπ_ref（供 KL）
    if self.use_reference_policy:
        ref_log_prob = self._compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)

    # 5) critic 估 V（PPO only，GRPO 跳过）
    if self.use_critic:                                   # ← GRPO 时 False，整段跳过
        values = self._compute_values(batch)
        batch = batch.union(values)

    # 6) driver 本地组装 advantage + KL（轻量，不 RPC）
    batch = compute_advantage(batch, adv_estimator="gae" or "grpo",
                             gamma=..., lam=..., num_repeat=rollout_n)

    # 7) 更新 critic（PPO only）+ actor（PPO clipped / GRPO 目标）
    if self.use_critic:
        self._update_critic(batch)                       # critic TD/GAE loss
    self._update_actor(batch)                            # actor PPO-clip + KL, backward 更新 θ

    # 8) weight sync: 新 θ → rollout（colocate 走 in-process，disaggregate 走跨机 NCCL）
    self.checkpoint_manager.update_weights(self.global_steps)
```

> [!note] `use_critic` 开关 = PPO/GRPO 拓扑统一
> 注意步骤 5、7 的 `if self.use_critic:` 守卫——GRPO 配置时整段 critic 分支跳过，**同一套五角色拓扑骨架既跑 PPO 也跑 GRPO**，差异只在 critic 开关 + advantage 估计器。这就是 verl/HybridFlow "统一抽象"的价值：换算法不换拓扑工程。

### 5.3 OpenRLHF colocate 模式启动（配置项以官方文档为准）

```bash
# OpenRLHF colocate 模式：actor+ref+critic+rm 同卡分时，rollout 用 vLLM 副本
ray job submit --address=... \
  -- python3 -m openrlhf.cli.train.ppo_ray \
    --pretrain {BASE_MODEL} \
    --save_model {OUTPUT} \
    --vllm_num_engines 4 --vllm_sync_backend nccl \  # rollout 用 vLLM, weight sync 走 NCCL
    --colocate --reward_model_type ... \              # colocate 模式（actor/ref/critic 同卡 sleep/wake）
    --zero_stage 3 --bf16 --packing_samples            # 训练用 DeepSpeed Zero-3 (≈FSDP)
# disaggregate 模式：去 --colocate，rollout 独立 GPU 池，weight sync 走跨机 NCCL/IPC
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[RLHF (PPO)]]（PPO 五角色来源）、[[GRPO]]（去 critic 的变体）、[[Actor-Critic]]（actor/critic 拆分原理）、[[Reward Model]]（rm 角色）、[[value network]]（critic 角色）、[[importance sampling与off-policy correction]]（rollout↔actor 的 ratio）、[[KL penalty与KL control]]（reference 角色）、[[Ray与分布式调度]]（多角色编排落地）。
- **下游（应用）**: [[colocated与disaggregated部署]]（拓扑在部署维度的展开）、[[trainer与rollout引擎组合]]（拓扑在引擎维度的展开）、[[RL权重同步]]/[[weight sync mechanism]]（actor→rollout 的动脉）、[[rollout train reference logprob一致性]]（rollout 与 actor 分离引擎引入的风险）、[[synchronous与asynchronous rollout]]（拓扑同步/异步时序）、[[17-RL训推一体框架]] 全章。
- **对比 / 易混**:
  - **RL 角色拓扑 vs [[训推分离]] PT/PD**：前者是 RLHF 训练内部的五角色（actor/rollout/ref/critic/rm 都为训练服务）；后者是部署侧训练↔服务分离（PT 训新版本推 PD 服务）。前者是"训练内多模型"，后者是"训练↔部署"。但 weight sync 机制同构（[[weight sync mechanism]]）。
  - **RL 角色拓扑 vs [[Actor-Critic]]**：后者是 RL 算法层面的"策略+价值"二分；前者是工程层面的"五角色部署"，把 Actor-Critic 的 actor/critic 各自配 rollout/ref/rm 陪跑并定部署。前者是后者的工程化展开。
  - **PPO 拓扑 vs GRPO 拓扑**：见 §3.5 表。唯一差异是 critic 有无。
  - **colocate vs disaggregate 拓扑**：见 [[colocated与disaggregated部署]]，是本篇 §3.4 拓扑图的专篇展开。


## 7. 常见误区与易错点

> [!warning] 误区 1：RLHF 是"一个模型自练自"
> 错。一次 RLHF 是 **5 个角色**（actor/rollout/ref/critic/rm）协同，各有不可替代职责。actor 是唯一被训的，但 rollout（采样优化）、ref（防漂移）、critic（降方差）、rm（打分）都是独立角色/独立权重副本。把 RLHF 当单模型 RL 是初学者最大误解。

> [!warning] 误区 2：rollout 和 actor 是同一个模型
> 是同一架构同一初始权重，但**是不同时刻的权重**（rollout 用 $\theta_{old}$、actor 用 $\theta$），且常是**不同引擎**（rollout 用 vLLM/SGLang 推理优化、actor 用 FSDP/Megatron 训练）。这正是 off-policy ratio 修正与 weight sync 的由来。详见 [[trainer与rollout引擎组合]]。

> [!warning] 误区 3：GRPO 不需要 reference
> 错。GRPO 砍的是 **critic**，不是 reference。GRPO 仍有 KL to $\pi_{ref}$。ref 在 PPO/GRPO 都在，防 reward hacking。

> [!warning] 误区 4：reference 就是 reward model
> 不同。reference 是**冻结的 SFT 策略** $\pi_{ref}$，算 logp 供 KL；reward model 是**偏好数据训的打分函数** $r_\varphi$。两者网络来源、用途、数学含义都不同。ref 防漂移、rm 给 reward，正交职责。初学者易混因都"另一个模型"。

> [!warning] 误区 5：critic 与 actor 同权重
> 不同。critic 是**独立的 value 网络** $V_\varphi$，与 actor $\pi_\theta$ 是不同网络、不同目标（actor 最大化 reward、critic 拟合 value）。只是同规模、同初始（常从 actor 拷贝初始化）但分开训。GRPO 直接不要它。

> [!warning] 误区 6：weight sync 是一次性
> 错。是**每个 batch 后都做一次**的高频通信（colocate in-process、disaggregate 跨机）。是 RL 训推一体最频繁的跨角色通信，开销决定吞吐。

> [!warning] 误区 7：五角色必须全 disaggregate 才"正确"
> 错。主流恰恰相反——verl/OpenRLHF 默认 **colocate**（actor_rollout_ref 共置省显存）为主、disaggregate 为辅（按需独立 rollout 池）。纯全 disaggregate 通信开销与 staleness 太重。选型看模型大小/资源/reward 性质，见 [[colocated与disaggregated部署]]。


## 8. 延伸细节

### 8.1 HybridFlow 的"统一抽象"

HybridFlow 论文（arXiv:2409.19256，ByteDance）核心贡献是把 RLHF 拆成两层：**control flow**（RL 算法本身，PPO/GRPO 的步骤序列，与分布式无关）+ **distributed execution**（各 role 怎么在多机多卡上跑、怎么并行）。上层算法写一次，下层执行可换不同 trainer（FSDP/Megatron）与不同 rollout（vLLM/SGLang）。这就是 verl 能同时支持 PPO/GRPO + FSDP/Megatron trainer + vLLM/SGLang rollout 的根——五角色拓扑是 control flow 的"角色声明"，引擎组合是 execution 的"实现绑定"。

### 8.2 3D-HybridEngine：actor 训练态↔rollout 推理态的 reshard

HybridFlow 的另一贡献。actor 角色既训练又（其副本）采样。训练态用 Megatron 3D 并行（TP/PP/DP，权重按 TP 切+PP 层切+DP 复制），rollout 推理态用 vLLM TP 并行（权重只按 TP 切）。两者**切分方式不同**，同卡组内切换需 reshard（把训练态分片还原/重切到推理态布局）。3D-HybridEngine 封装这个 reshard。这是 [[trainer与rollout引擎组合]] §3.5 的核心，本篇只点其作为"actor 角色训推一体"的拓扑含义。

### 8.3 reward model 的位置变体

- **colocate RM**：RM 与 actor 同池，forward 打分完释放，省独立卡（小 RM 常用）。
- **独立 RM 池**：RM 大或 reward 计算重（含多 RM ensemble / process reward），独立池按需扩缩。
- **rule-based verifier**：常不需 GPU（CPU 跑单测/正则），driver 本地算，不占角色 GPU。GRPO/R1 数学题常如此。
- **process reward model (PRM)**：逐 token/step 打分（不只 outcome），算力重，常独立池。待核实各框架对 PRM 的拓扑支持现状。

### 8.4 reference 的两种 KL 用法

- **in-reward KL**（OpenAI/TRL/verl 默认）：KL 作为 reward 的减项 $r_t\leftarrow r_t-\beta\text{KL}_t$，actor loss 不显式含 KL。$\beta$ 可自适应。
- **explicit KL loss**（GRPO 风格）：$\mathcal{L}=\ldots-\beta\mathbb{D}_{KL}[\pi_\theta\|\pi_{ref}]$，actor loss 显式含 KL。
两者都依赖 reference 角色，只是 KL 进入目标的位置不同。详见 [[KL penalty与KL control]]。

### 8.5 多机多卡的进程组拓扑

五角色各自在自己的 worker group 内有 torch.distributed 进程组（trainer 用 NCCL 做梯度 AllReduce/权重 AllGather；rollout 内部 vLLM TP 通信；critic 独立进程组）。跨角色的张量同步（weight sync、logp 回传）走：colocate 时 in-process 共享显存/同机 NCCL，disaggregate 时跨机 NCCL broadcast。driver（RayPPOTrainer）只持 handle，靠 `wg.method(batch).get()` RPC 编排，非张量控制走 ray.get。见 [[Ray与分布式调度]] §3.4/§3.5。

### 8.6 内容来源

五角色定义综合自 [[RLHF (PPO)]]（InstructGPT / TRL PPO trainer）、[[GRPO]]（DeepSeekMath arXiv:2402.03300）。verl 角色映射与 fit 时序来自 [[Ray与分布式调度]] §3.5 已核实的代码解答。HybridFlow 统一抽象与 3D-HybridEngine 来自 arXiv:2409.19256。OpenRLHF colocate/sync_backend 来自 OpenRLHF 官方文档（`architecture.html`/`hybrid_engine.rst`/`async_training.html`），具体配置名待以最新文档为准。weight sync 路线来自 [[weight sync mechanism]] 已联网核实的四路线总结。截至 2026-07。

---
相关: [[角色拓扑]] | [[colocated与disaggregated部署]] | [[trainer与rollout引擎组合]] | [[RLHF (PPO)]] | [[GRPO]] | [[Actor-Critic]] | [[Reward Model]] | [[value network]] | [[rollout worker]] | [[inference worker]] | [[learner worker split]] | [[训推分离]] | [[RL权重同步]] | [[weight sync mechanism]] | [[rollout train reference logprob一致性]] | [[stale policy problem]] | [[asynchronous training]] | [[Ray与分布式调度]] | [[importance sampling与off-policy correction]] | [[KL penalty与KL control]] | [[GAE]] | [[baseline]] | [[variance reduction]] | [[新模型接入]] | [[17-RL训推一体框架]]
