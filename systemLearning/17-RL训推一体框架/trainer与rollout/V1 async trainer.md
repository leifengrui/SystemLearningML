# V1 async trainer

> **所属章节**: [[trainer与rollout]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §81
> **别名**: V1 异步 Trainer / verl async trainer / colocate_async / separate_async / HybridFlow async pipeline
> **难度**: 高（需懂 [[synchronous与asynchronous rollout]]、[[RL权重同步]]、[[asynchronous training]]、[[trainer与rollout引擎组合]]、[[importance sampling与off-policy correction]]、[[colocated与disaggregated部署]]）
> **来源**: ① verl 官方文档 `docs/advance/v1_async_trainer.md`（2026-08-20）；② 社区考古：知乎/CSDN verl 0.7 架构解析、TransferQueue 接入梳理（PR #5401）、训推分进程方案、异步异卡方案考古、异步执行实测。Zhihu 多数源 WebFetch 返回 403，其内容据搜索摘要标注。

## 1. 一句话定义

**verl V1 async trainer** 是 verl `verl.trainer.main_ppo` 入口下的异步 PPO 训练后端，提供两种异步模式——`colocate_async`（生成与训练共享同一 GPU 池，靠 warmup batch + partial rollout 加速）和 `separate_async`（生成跑在独立 rollout 卡池、训练跑在混合卡池，可选把空闲混合卡借给生成）——两者都建立在 V1 **TransferQueue**（控制流/数据流解耦）+ **ReplayBufferAsync** + **FullyAsyncLLMServerClient**（partial rollout 客户端）之上，用模型版本号单位的陈旧度控制（`max_off_policy_threshold=8`）+ drop/wait 策略保收敛，是 verl v0.7+ HybridFlow 双控制器架构里"one-step-off / fully-async"两条管线的生产实现。

## 2. 为什么需要它（动机与背景）

### 2.1 同步 PPO 的两段式空闲
同步 PPO 里 rollout 与训练串行：训练等全部样本生成完才能开始 update。多轮 tool-calling / 长 CoT 任务里，少数样本要跑 3 分钟、多数 20 秒，同步要等最慢样本（[知乎 agent rl fully async 解析](https://zhuanlan.zhihu.com/p/2031409957392880688)）。rollout GPU 在训练时空闲、训练 GPU 在 rollout 时空闲；colocate 则同批卡在 generate/train 间互相等。

### 2.2 控制器带宽瓶颈
旧 `RLTrainer` 单进程同时扛控制流与数据流，每步要把整 batch 经控制器序列化/反序列化 5–6 次（rollout→reward→old_log_prob→ref→adv→update）。多模态海量张量下控制器成瓶颈。TransferQueue（PR #5401，2026-04 合入）把数据流从控制流剥离：控制器只移动**元数据**，worker 直接从分布式存储取张量（[知乎 TransferQueue 接入梳理](https://zhuanlan.zhihu.com/p/2053046423659325008)）。

### 2.3 colocate 共进程的 VRAM 冲突
verl v0.7.0 的 colocate（共卡共进程）让每张卡一个 Ray worker 同时演推理与训练两角色，靠内存"relay"切换。但训练要 `expandable_segments`、vLLM 的 KV cache 估计却被它伤；且缺全局负载均衡。v0.7.1 改 **分进程**（`separate_async`）各管各 VRAM（[知乎 训推分进程方案](https://zhuanlan.zhihu.com/p/2026697817284965439)，403 据摘要）。这是官方文档 `separate_async` 模式的工程来由。

### 2.4 partial rollout：不丢长轨迹
模式切换时若直接 abort 生成，长 CoT 轨迹全丢。partial rollout 让已生成 token + logp 保留、只重试未完成部分，重采请求跑在新权重上、KV cache 按保留前缀重建续采。代价是额外 prefill + 轨迹内策略版本切换（官方 §Partial Rollout and Staleness）。

## 3. 核心概念详解

### 3.1 三种 trainer_mode（官方）

| Mode | Rollout 资源 | 典型用途 |
|---|---|---|
| `sync` | 混合 rollout 副本与 trainer colocate | 基线 PPO，要求严格同步 |
| `colocate_async` | 混合 rollout 副本与 trainer colocate | warmup batch + partial rollout 加速 |
| `separate_async` | 独立 rollout 副本 + 混合 trainer 副本 | 无切换/offload 成本的资源隔离，更紧凑高效 |
| `separate_async` + switch | 同上 | trainer 空闲时把混合卡借给 rollout |

`colocate_async` 与 `separate_async` 都通过 `FullyAsyncLLMServerClient` 开 partial rollout：生成在模式切换中被 abort 时，已完成 token 保留、剩余重试，**一条恢复轨迹可跨多个模型版本**（官方）。

### 3.2 三条管线（社区/概念层）
verl 0.7+ 把 RL 管线分三层（[CSDN verl 0.7 架构解析](https://blog.csdn.net/gitblog_00166/article/details/154463814)）：

| 管线 | 执行 | 资源 | 场景 |
|---|---|---|---|
| **on-policy（同步）** | rollout 与训练串行 | 通常共享 GPU | 基线，优先算法正确性 |
| **one-step-off-policy** | 当前步训练与下批生成重叠，用上步参数 rollout | 资源隔离 | +20–40% 效率，稳定性近 on-policy |
| **fully-async** | Trainer 与 Rollouter 完全解耦到不同节点，流式数据+陈旧度控制+partial rollout | 解耦分离式 | 128+ GPU 或长 CoT；v0.9 计划合入 main |

V1 文档的 `colocate_async`/`separate_async` 是 one-step-off 与 fully-async 在生产 V1 入口下的具名实现。

### 3.3 HybridFlow 双控制器 + 四组件
verl 0.7 统一两种编程模型（[CSDN verl 0.7 解析](https://blog.csdn.net/gitblog_00166/article/details/154463814)）：
- **高层单控制器（MPMD）**：单进程 `RLTrainer` 管全局计算图，调度 rollout/reward/分布式训练等宏任务，支持"灵活数据依赖、多样资源分配、细粒度异步陈旧度控制"。
- **内部多控制器（SPMD）**：ModelEngine 内部 Workers 跑标准分布式训练，FSDP/Megatron/VeOmni 后端，集合通信同步。

四核心可插拔组件：
1. **ModelEngine**：训练引擎，接口 `initialize`/`forward_backward_batch`/`optimizer_step`/`lr_scheduler_step`/`get_per_tensor_param`/`to`/`save_checkpoint`/`load_checkpoint`。后端 FSDP（dense 中等/MoE 低）、MCore（DP+TP+PP+EP+CP，高性能）、VeOmni（FSDP+SP+EP，alpha）。
2. **Rollout Engine + AgentLoop**：v0.7 移除 SPMD rollout，默认 rollout **server 模式**，在线服务方式 + 动态批处理。集成 vLLM 0.12.0 / SGLang 0.5.6 / TensorRT-LLM。`AgentLoop` 抽象（`SingleTurnAgentLoop`/`ToolAgentLoop`）管理"向 LLM server 请求响应、与环境交互取反馈"的循环；`AgentLoopOutput` 含 prompt_ids/response_ids/response_mask(1=LLM,0=tool)/response_logprobs/reward_score/num_turns/metrics。
3. **TransferQueue**：控制流/数据流解耦，张量引用传递 + 零拷贝 + RDMA，后端 ZeroMQ/NIXL/Ray RDT，v0.8 计划默认。依赖外部 `transfer_queue` 包 0.1.8。三个胶水抽象（[知乎 PR #5401 梳理](https://zhuanlan.zhihu.com/p/2053046423659325008)，403 据摘要）：
   - **KVBatchMeta**：thin 数据契约替代 `DataProto`，只带元数据
   - **tqbridge**：装饰器，自动把 dispatch 层调用桥接到 TransferQueue
   - **ReplayBuffer**：控制器侧元数据镜像，`replay_buffer.sample()` 让控制器等数据
4. **Checkpoint Engine**：跨节点权重同步统一抽象，接口 `send_weights`/`receive_weights`。后端 `naive`（同 GPU 零开销）/`nccl`（broadcast 集合通信）/`nixl`（P2P 点对点）/`hccl`。分桶流式 `bucket_size=2048MB`，每 chunk 带 `TensorMeta`。`CheckpointEngineWithCache` 支持 shm/disk 本地缓存，使权重同步不打断进行中请求——**partial rollout 的前提**。`CheckpointEngineManager` 支持 `add_replicas`/`remove_replicas` 弹性扩缩容。

### 3.4 演化时间线（社区考古）
verl 异步异卡方案演化（[CSDN 异步异卡方案考古](https://blog.csdn.net/h_2025/article/details/156595424)，521 据摘要；[知乎考古](https://zhuanlan.zhihu.com/p/1988208406008464858)）：

```
MultiTurn Rollout → AgentLoop → RecatAgentLoop → AsyncFlow TransferQueue → OneStepOff → FullyAsyncPolicyTrainer
```
- `AgentLoop` 原无 cancel/resume 接口，全异步必须补（[知乎 Partial Rollout 工程实现](https://zhuanlan.zhihu.com/p/2019147267240653513)）。
- `FullyAsyncRollouter` 自持数据迭代器、控制生产节奏，靠 `MessageQueue` 与 Trainer 解耦。
- 开关：`config.actor_rollout_ref.rollout.mode = "async"`。

## 4. 数学原理 / 公式

### 4.1 mini-batch 不变量
所有 trainer_mode 每全局步处理的 PPO mini-batch 数相同（官方）：

$$\text{mini-batches per PPO epoch}=\frac{\text{train\_batch\_size}}{\text{ppo\_mini\_batch\_size}}$$

`separate_async` 的流式约束（让上式等于其 mini-batch 数）：

$$\text{train\_batch\_size}=\text{parameter\_sync\_step}\times\text{ppo\_mini\_batch\_size}$$

例：train_batch=64、ppo_mini_batch=16 → `parameter_sync_step=4`。

### 4.2 陈旧度（模型版本号单位）
partial rollout 让一条轨迹跨多版本，V1 sampler 以**模型版本号差**度量陈旧度（官方）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `trainer.v1.sampler.max_off_policy_threshold` | `8` | 首次生成到被训练之间最大模型版本差，超则触发陈旧度处理 |
| `trainer.v1.sampler.max_off_policy_strategy` | `drop` | `drop` 丢弃陈旧 prompt 组（GRPO 下一整组丢）；`wait` 阻塞等在途组完成 |

监控指标：
- `training/off_policy/trajectory_spans/*`：一轨迹用过的版本数，`1`=单版本全程
- `training/off_policy/trajectory_staleness/*`：轨迹用过的最新版本 vs 当前训练版本差
- `training/off_policy/trajectory_staleness_worst/*`：轨迹用过的最旧版本 vs 当前训练版本差

> [!note] 与 staleness 数学的关系
> 版本号差 $k$ 让 IS ratio $\rho=\exp(k\cdot g_t)$ 指数偏离 1——这是 [[importance sampling与off-policy correction]] 里"staleness→ratio 指数放大"的工程具象。`max_off_policy_threshold=8` 是把 $k$ 钉死在 8 以内。

### 4.3 separate_async 的解耦 PPO（pi_old 跨 mini-batch 恒定）
`separate_async` 以 mini-batch 粒度训练，recompute old logp 时 actor 权重会在 controller 级 mini-batch 间变化。为整 `parameter_sync_step` 周期保持同一 $\pi_{old}$（官方）：
1. 首 mini-batch：`pi_old` 拷到 CPU，GPU 上用当前权重算
2. 后续每个 mini-batch 前：当前更新权重拷到 CPU → `pi_old` 恢复到 GPU 算 old logp → 当前权重恢复到 GPU，清 CPU 临时副本

$N$ 个 mini-batch → $N$ 次 `save_model_to_cpu` + $2(N-1)$ 次 `restore_model_from_cpu`。`pi_old` 本身存 1 次、恢复 $N-1$ 次；其余调用保当前权重。例：4 mini-batch = 4 save + 6 restore（其中 1 save + 3 restore 传 `pi_old`）。

`algorithm.rollout_correction.bypass_mode=True` 时跳过这些传输与 old logp 计算——但 IS 权重分母带 [[训推不一致]]（TIM），详见 [[rollout train reference logprob一致性]] §8.1。

### 4.4 step switching 阈值
借卡阈值（官方）：

$$\text{target}=\text{round}(\text{switch\_threshold\_ratio}\times\text{train\_batch\_size})$$
$$\text{threshold}=\text{clamp}(\text{target},\ \text{one\_mini\_batch},\ \text{train\_batch\_size})$$

`adaptive_switch_threshold=True` 时按观测 trainer 空闲自适应：连续 `switch_threshold_release_steps` 个空闲步 → ratio +`switch_threshold_step_up`；连续非空闲 → ratio -`switch_threshold_step_down`。release 区间双向防噪。

## 5. 代码示例

### 5.1 colocate_async 启动
```bash
trainer.use_v1=True \
trainer.v1.trainer_mode=colocate_async \
trainer.v1.colocate_async.num_warmup_batches=1
```
warmup batch 在首个训练步前先开始生成，减初始空 buffer 等待。

### 5.2 separate_async 启动（2 训练节点 + 2 独立 rollout 节点）
```bash
trainer.use_v1=True \
trainer.v1.trainer_mode=separate_async \
trainer.nnodes=2 \
trainer.n_gpus_per_node=8 \
actor_rollout_ref.rollout.nnodes=2 \
actor_rollout_ref.rollout.n_gpus_per_node=8 \
actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
data.train_batch_size=64 \
actor_rollout_ref.actor.ppo_mini_batch_size=16 \
trainer.v1.separate_async.parameter_sync_step=4 \
trainer.v1.separate_async.num_warmup_batches=1
```
`separate_async` 要非 `naive` 的 checkpoint-engine backend（`nccl`/`nixl`/`mooncake`）做独立 rollout 权重同步。

### 5.3 开 step switching
```bash
trainer.v1.separate_async.hybrid_rollout.enable_switch=True \
trainer.v1.separate_async.hybrid_rollout.adaptive_switch_threshold=True
```

### 5.4 partial rollout 的 off-policy 控制（Python 伪码）
```python
# V1 sampler 的 drop/wait 决策（概念）
def should_drop_group(group, current_version, threshold=8, strategy="drop"):
    staleness = current_version - group.oldest_version
    if staleness > threshold:
        return strategy == "drop"        # wait 模式不丢，阻塞等在途组完成
    return False

# replay buffer refill：丢 k 组就补 k 条新 prompt
def on_evict(replay_buffer, k, dataloader, transfer_queue):
    new_prompts = dataloader.fetch(k)          # 从训练 dataloader 取 k 条
    transfer_queue.mark_pending(new_prompts)   # 在 TransferQueue 标 pending
    agent_loop.dispatch(new_prompts)           # 派给 AgentLoop 生成
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[synchronous与asynchronous rollout]]（同步/异步/partial/off-policy 概念）、[[RL权重同步]]（Checkpoint Engine 的 nccl/nixl 后端）、[[asynchronous training]]（异步训练一般）、[[trainer与rollout引擎组合]]（FSDP/Megatron + vLLM/SGLang 两端组合）、[[colocated与disaggregated部署]]（colocate vs separate 的部署视角）、[[importance sampling与off-policy correction]]（staleness→ratio 指数偏离）
- **下游（应用）**: [[rollout train reference logprob一致性]]（bypass_mode 与 recompute 的 TIM 治理）、[[训练推理不一致TIM技术综述]]（版本/staleness 层的工程代表，AReaL 同类）、[[TIS]]（异步 staleness 的 IS 兜底）、verl V1 生产训练
- **对比 / 易混**:
  - **colocate_async vs separate_async**：前者同 GPU 池 colocate + warmup/partial 加速；后者独立 rollout 卡池 + 混合 trainer 卡池，无切换/offload 成本。后者需非 naive checkpoint backend。
  - **one-step-off vs fully-async**：前者当前步训练与下批生成重叠（+20–40%，稳定性近同步）；后者 Trainer/Rollouter 完全解耦到不同节点（128+ GPU 或长 CoT）。
  - **V1 async trainer vs FullyAsyncPolicyTrainer**：前者是 `verl.trainer.main_ppo` 入口的生产 V1 模式（colocate/separate_async）；后者是 `recipe/fully_async_policy/` 的高级全异步训练器，`ray.wait()` 驱动独立 FullyAsyncRollouter/Trainer，`asyncio.Queue` 管待办/在途/取消。
  - **partial rollout vs bypass_mode**：partial rollout 是生成中断保前缀续采（V1 async client 行为）；bypass_mode 是跳过 trainer 用 $\theta_{old}$ 重算 old_log_prob（省 forward 但 IS 带 TIM）。正交可叠加。

## 7. 常见误区与易错点

> [!warning] 误区 1：partial rollout 不会引入 off-policy
> 会。一条轨迹跨多模型版本，前缀用旧 $\theta$、后缀用新 $\theta$，轨迹内策略版本变。代价是额外 prefill + 轨迹内 off-policy。`max_off_policy_threshold` 与 drop/wait 就是为此。

> [!warning] 误区 2：separate_async 用 naive checkpoint backend
> 不行。`separate_async` 要 `nccl`/`nixl`/`mooncake` 做独立 rollout 权重同步。`naive` 仅适合同 GPU 零开销传递，用于 colocate。

> [!warning] 误区 3：step switching 与 rollout PD disaggregation 可同开
> 官方明示：暂时不能与 rollout PD disaggregation 组合。

> [!warning] 误区 4：把 V1 async trainer 等同 fully-async
> V1 文档的 `colocate_async`/`separate_async` 对应 one-step-off / fully-async 谱系的生产 V1 模式；`FullyAsyncPolicyTrainer`（`recipe/fully_async_policy/`）是更激进的高级全异步训练器，不是同一入口。

> [!warning] 误区 5：validation 能干净测推理吞吐
> validation 与未完成训练轨迹共享同一 AgentLoop + rollout server 池，partial 轨迹继续跑。`timing_s/testing` 含竞争与 rollout 容量消耗，非隔离测量。

## 8. 延伸细节

### 8.1 benchmark（官方，Qwen3.5-35B-A3B，150 步，4 节点）
| Mode | 资源 | 150 步时长 | 聚合 tokens/s | 平均响应长 |
|---|---|---|---|---|
| `sync` | 4 hybrid | 22.79 h | 12,053 | 12,720 |
| `colocate_async` | 4 hybrid | 14.72 h（-35.4%） | 18,852（+56.4%） | 12,854 |
| `separate_async` | 2 hybrid + 2 standalone | 14.10 h（-38.1%） | 18,829（+56.2%） | 12,288 |

step switching（100 步，3×8 H100，2 hybrid + 1 standalone）：
| Mode | 100 步时长 | tokens/s | 平均响应长 | 平均 reward |
|---|---|---|---|---|
| no-switch | 13.15 h | 14,604 | 13,343 | 0.7755 |
| switch | 11.79 h（-10.3%） | 16,445（+12.6%） | 13,478 | 0.7762 |

switch 在 reward 不降前提下 -10.3% 时长、+12.6% 吞吐。

### 8.2 社区实测（补充，非官方）
- 8×A100 80G、Qwen2-7B、batch=32：异步让 rollout 端到端 -42.6%、整体吞吐 2.3×、GPU 利用率 58%→89%（[CSDN 异步执行实测](https://blog.csdn.net/weixin_31776191/article/details/157565609)）。
- TransferQueue（PR #5401）多模态后训练 128 H100 端到端 +49.1%（[知乎 PR #5401 梳理](https://zhuanlan.zhihu.com/p/2053046423659325008)，403 据摘要）。
- AsyncFlow on Ascend Atlas 900 A3 SuperPod 长序列（prompt 2k→response 16k）3.81×（59.3→226.8），资源 52-2-2-8（[百度百家号 昇腾 Async](https://baijiahao.baidu.com/s?id=1869976883861023133)）。

### 8.3 调参顺序（官方）
1. 先平衡 trainer 与 standalone rollout 资源，再开 switching
2. 由 batch-size 不变量定 `parameter_sync_step`
3. 按 policy-lag 容忍度选 `max_off_policy_threshold` 与 drop/wait
4. `timing_s/gen` 显示持续 trainer 空闲时才开 switching

关键 timing 指标：`timing_s/gen`（trainer 等下个可训练 batch 的空闲）、`timing_s/update_actor`、`timing_s/update_weights`（separate_async 独立权重同步）、`timing_s/switch_wait`（借卡填阈值，**非空闲**）、`timing_s/switch_to_rollout`、`timing_s/switch_to_trainer`。

### 8.4 来源与核实
- 模式表、partial rollout 四步、off-policy 参数表、replay buffer refill、解耦 PPO 的 pi_old CPU save/restore 次数公式、step switching 阈值与自适应、benchmark 表、调参顺序：均来自官方 `docs/advance/v1_async_trainer.md`（2026-08-20）已读全文。
- HybridFlow 双控制器/四组件/三管线、TransferQueue PR #5401 三抽象、colocate→分进程演化、演化时间线、社区实测：来自 CSDN verl 0.7 架构解析（已 WebFetch 取实）+ 知乎多条（WebFetch 403，内容据搜索摘要，已标注）。Zhihu 源未直接渲染核验，标 caution。
- 与 AReaL（arXiv:2505.24298）同属异步 RL 系统族，但 V1 async trainer 是 verl 框架内生产实现，AReaL 是独立学术系统。截至 2026-09。

---
相关: [[trainer与rollout引擎组合]] | [[synchronous与asynchronous rollout]] | [[RL权重同步]] | [[asynchronous training]] | [[colocated与disaggregated部署]] | [[importance sampling与off-policy correction]] | [[rollout train reference logprob一致性]] | [[训练推理不一致TIM技术综述]] | [[TIS]] | [[17-RL训推一体框架]]
