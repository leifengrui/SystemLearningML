# Ray与分布式调度

> **所属章节**: [[09-Ray与分布式调度]]（章索引）
> **所属模块**: [[09-Ray与分布式调度]]
> **难度**: 中（通用分布式调度框架，RL 训推系统基石）
> **别名**: Ray、分布式调度

## 1. 一句话定义

**Ray** 是一个**通用分布式计算框架**——它把"远程函数（task）"和"远程对象/actor"做成 Python 一等公民，让你用 `@ray.remote` 装饰器把普通函数/类变成**可跨进程、跨机调度执行**的分布式单元，由 Ray 的**调度器 + 对象存储（object store）+ GCS** 统一管资源、放任务、传数据。它是 **RLHF/LLM 训推系统**（如 verl、OpenRLHF、vLLM Ray 版）的主流调度底座，也是 [[PPO optimization]]、[[rollout worker]]、[[policy deployment (PD)]] 多机分布式的实现基础。本笔记是第九章总览，把 Ray 的三大原语、调度机制、工程坑串成一张图，细节进各子笔记。

> [!note] Ray 在 LLM/RL 栈的位置
> Ray 不训模型、不通信——训练靠 PyTorch + NCCL（[[torch.distributed]]），推理靠 vLLM 等引擎。Ray 干的是**把这些组件编排到多机**：起若干 actor（policy trainer、reward model、rollout worker、vLLM engine），按资源放卡，在它们之间传权重/数据/控制信号。所以 Ray ≈ "RL/LLM 训推系统的操作系统"，PyTorch/vLLM 是它上面的应用。

## 2. 为什么需要它（动机与背景）

LLM/RL 训推系统天然多角色、多机：

- **多角色**：policy trainer（PT）、rollout worker、reward model（RM）、reference model、value model、policy deployment（PD）各自独立进程/容器；
- **多机**：单机 8 卡装不下 70B 训练 + 几十路 rollout + RM 推理，必须跨机；
- **异构资源**：trainer 要 NVLink 全互联，rollout 要高吞吐 batch，vLLM 要连续批处理——资源画像不同；
- **动态调度**：rollout 产数据快慢不一、trainer 要等 rollout、PD 要从 PT 拿新权重——需要任务队列、资源预约、actor 生命周期管理。

PyTorch 的 [[torch.distributed]] 只管"一组 rank 一起做 collective"，不管"哪个进程跑哪台机、谁先起谁后起、actor 之间怎么传消息"。Ray 补这一层：

1. **声明式远程对象**：`@ray.remote` 一行把函数/类变分布式，不用手写 socket/RPC；
2. **统一对象存储**：跨进程传张量走共享内存/ Plasma 对象存储，零拷贝或近零拷贝；
3. **资源调度**：actor/task 声明要几张卡、几个 CPU，调度器按拓扑放置；
4. **容错与弹性**：actor 挂了能重启、节点能动态加入（与 `torchrun --rdzv` 互补）；
5. **生态**：vLLM、verl、OpenRLHF、Ray Serve、RLlib 都建在 Ray 上，复用同一调度层。

## 3. 核心概念详解

### 3.1 三大原语

| 原语 | API | 语义 | 对应笔记 |
|---|---|---|---|
| **Task（远程函数）** | `@ray.remote def f(...)` → `f.remote(...)` | 无状态的远程函数调用，返 `ObjectRef` Future | [[task model]]、[[remote function]] |
| **Actor（远程对象）** | `@ray.remote class C` → `a = C.remote()` | 有状态的远程对象，方法调用异步返 Future | [[actor model]] |
| **Object（对象存储）** | `ObjectRef`、`ray.put`/`ray.get` | 不可变值对象，跨进程共享，底层走 Plasma | （隐含于 task/actor） |

三者关系：task 产 object，actor 持状态并暴露方法（方法本身也是 task），`ray.get(ref)` 取回 object 阻塞等结果。

### 3.2 调度模型

Ray 的调度器看每个 task/actor 的 **resource request**（CPU/GPU/内存/自定义）和 **placement group** 约束，在集群里找满足的节点放置：

- **resource request**：`@ray.remote(num_gpus=1, num_cpus=8)` 声明需求；
- **placement group**：`pg = ray.util.placement_group([...], strategy="STRICT_SPREAD")` 预占一组资源包，保证 actor 共置/散布；
- **scheduling策略**：PACK（装箱省资源）/SPREAD（散布避干扰）/STRICT_*（强制）；

详见 [[resource request]]、[[placement group]]、[[scheduling策略]]、[[GPU绑定]]。

### 3.3 数据流：Object Store + GCS

- **Object Store（Plasma）**：每个节点一个共享内存对象存储，跨进程传大张量走共享内存（同机零拷贝）、跨机走 TCP/NCCL（大张量可用 `ray.train` 的 NCCL 集成）；
- **GCS（Global Control Store）**：存 actor/task 元数据、ObjectRef→节点映射，去中心化查询；
- 传小对象（config/dict）直接 pickle 走 GCS；传大张量走 object store 零拷贝。

### 3.4 与 PyTorch/NCCL 的集成

Ray 不替代 NCCL。典型集成：
- Ray 起若干 actor，每个 actor 内部跑一个 PyTorch `torch.distributed` 进程组（`init_process_group`）；

> [!note] 解答：actor 之间怎么传递信息？需要传吗？主控怎么控制？
> **结论先行**：actor 之间**需要**传递信息，但分两条正交通道——**张量数据（大）走 NCCL 集体通信**，**控制信号/小对象走 Ray Object Store**。主控用 **single-controller 模式**：一个 driver 进程持有所有 actor 的 handle，靠 `actor.method.remote(args)` + `ray.get()` 编排全局时序，actor 内部再各自起 `torch.distributed` 进程组做张量同步。
>
> ### 一、为什么 actor 之间需要传信息
> RLHF/LLM-RL 系统是多角色流水线，每一步都产生数据要交给下一步：
>
> | 阶段 | 产出 | 谁传给谁 | 数据类型 |
> |---|---|---|---|
> | rollout | 轨迹 `(prompt, response, logp_rollout)` | rollout actor → driver → PT | 张量+文本 |
> | reward | `rm_scores` | RM actor → driver → PT | 张量 |
> | ref logp | `ref_log_prob` | ref actor → driver → PT | 张量 |
> | value | `values` | critic actor → driver → PT | 张量 |
> | weight sync | 新 $\theta$ | PT actor → rollout/PD actor | 大张量(权重) |
> | 控制 | "开始 rollout""权重就绪" | driver → 各 actor | 小对象/flag |
>
> 所以"需要传"是肯定的——不传就没人喂下一步。
>
> ### 二、两条通道：Ray Object Store vs NCCL
>
> | 维度 | Ray Object Store（`ray.get`/`ray.put`） | NCCL 集体通信 |
> |---|---|---|
> | 传什么 | 任意可 pickle 的 Python 对象（张量、dict、DataProto、控制 flag） | GPU 张量 |
> | 底层 | Plasma 共享内存(同机零拷贝)/TCP(跨机) | NVLink/RDMA/IB，GPU 直达 |
> | 语义 | 点对点 Future（caller→callee）+ 阻塞取回 | 集体归约/广播（all-reduce/broadcast/all-gather） |
> | 带宽量级 | 跨机 ~10-25 Gbps（受 object store + pickle） | NVLink 300-900 GB/s，IB 100-400 Gbps |
> | 适合 | 控制流、小对象(<100KB 走 GCS pickle)、中等张量 | 大张量（权重 $\theta$、梯度、激活） |
> | 典型用法 | driver 调 `rollout_wg.generate_sequences(batch).get()` 拿轨迹 | PT 内 all-reduce 梯度、broadcast 新 $\theta$ 给 rollout |
>
> **关键工程取舍**：大 $\theta$（70B bf16 = 140GB）**绝对不能**走 Ray object 传输（慢+占带宽+阻塞），必须走 NCCL broadcast（见 [[weight sync mechanism]]）。控制信号"开始 rollout"这种小对象走 Ray 即可。
>
> ### 三、主控怎么控制各 actor：single-controller 模式
> verl/OpenRLHF 都采用 **single-controller** 架构（区别于 RLlib 的 multi-controller）：
>
> ```
> ┌──── driver 进程（单 CPU，不持卡）────┐
> │  持有: actor_rollout_wg (RayWorkerGroup)  │
> │       critic_wg, ref_policy_wg            │
> │  fit() 主循环:                            │
> │   1. rollout_out = async_rollout_mgr      │
> │        .generate_sequences(batch)  # RPC  │
> │   2. reward = reward_loop_mgr.compute(batch)│
> │   3. old_logp = actor_rollout_wg          │
> │        .compute_log_prob(batch).get() #RPC│
> │   4. adv = compute_advantage(batch) #本地 │
> │   5. actor_rollout_wg.update_actor(batch) │
> │        .get()  # RPC 触发 PPO 更新        │
> │   6. ckpt_mgr.update_weights(step) #权重sync│
> └──────────────┬───────────────────────────┘
>                │ ray.get(.remote()) 编排时序
>   ┌────────────┴────────────┐
>   ▼                         ▼
> Ray actor 0...N          Ray actor 0...M
> (rollout/vLLM)          (critic/Megatron)
> 内部各跑 torch.distributed 进程组（NCCL）
> ```
>
> **driver 不持卡，只编排**：它通过 `RayWorkerGroup` 持有每个 actor 的 `ray.actor.ActorHandle`，调 `worker.method.remote(*args)` 发 RPC（异步返 `ObjectRef`），`ray.get(ref)` 阻塞取结果。所有"谁先谁后"的时序逻辑写在 driver 的 `fit()` 里，actor 自己不知道全局流程，只响应方法调用——这就是 single-controller：**一个脑子，多个手脚**。
>
> ### 四、verl 代码实证（`verl/single_controller/ray/base.py`）
>
> ```python
> # RayWorkerGroup.execute_all_async —— driver 一次性给所有 worker 发同一个方法
> def execute_all_async(self, method_name, *args, **kwargs):
>     length = len(self._workers)
>     # 如果 args 都是 list 且长度==worker 数，自动按位置切片分发（dispatch）
>     if all(isinstance(a, list) for a in args) and all(len(a)==length for a in args):
>         return [self._execute_remote_single_worker(self._workers[i],
>                 method_name, *(a[i] for a in args)) for i in range(length)]
>     # 否则同一个 args 广播给所有 worker
>     return [self._execute_remote_single_worker(w, method_name, *args, **kwargs)
>             for w in self._workers]
>
> # _execute_remote_single_worker —— 对单个 actor handle 调 .remote()
> def _execute_remote_single_worker(self, worker, method_name, *args):
>     remote_call = getattr(worker, method_name)   # actor 方法 = Ray remote function
>     return remote_call.remote(*args, **kwargs)   # 立即返 ObjectRef，不阻塞
> ```
>
> **数据怎么到 actor**：`args`（含张量/DataProto）随 `.remote()` 调用被 Ray 序列化进 object store，actor 进程从本地 plasma 取出反序列化。`ray.get(ref)` 让 driver 阻塞等 actor 返回的结果（也走 object store 回传）。这就是"Ray 传对象"的通道。
>
> **worker 内部怎么起 NCCL 组**：driver 在 `_create_worker` 里把 `MASTER_ADDR/MASTER_PORT/WORLD_SIZE/RANK` 通过 `runtime_env.env_vars` 注入每个 actor 进程（见 `base.py` L630-638），actor 启动时读这些 env 调 `init_process_group(backend="nccl")` 自组织进程组。之后 actor 之间做 all-reduce/broadcast 走 NCCL，driver 完全不参与——这就是"NCCL 传张量"的通道，与 Ray RPC 正交。
>
> ### 五、误区
> - ❌ "actor 之间直接互相 `ray.get` 对方的方法" → 默认不这样。single-controller 下 actor 不持有彼此 handle，只 driver 持有；actor 间若需张量同步走 NCCL，不走 Ray RPC。
> - ❌ "driver 也要参与 NCCL 通信" → 不参与。driver 是纯 CPU 编排者，NCCL 组只在 actor 之间。
> - ❌ "大权重走 `ray.put`/`ray.get` 传给 rollout" → 慢且占带宽，必须走 NCCL broadcast 或 weight sync engine（见 [[weight sync mechanism]]）。
> - ❌ "actor 之间不传信息，各跑各的" → 错。PPO 数据流是强耦合流水线，rollout→reward→advantage→update 每步都跨 actor 传数据。
>
> 相关：[[actor model]]、[[weight sync mechanism]]、[[Ray与分布式调度]] §3.4/§3.5、verl `RayWorkerGroup`。

- actor 间的集体通信走 NCCL（Ray 用 `ray.train.torch` / `TorchTrainer` 帮你起 process group）；
- actor 间的非张量控制（"开始 rollout""权重已就绪"）走 Ray 的 `ray.get`/Future。

所以 **Ray = 调度+编排+对象存储，NCCL = 张量集体通信**，二者正交，常组合。见 [[torch.distributed]] §6、[[Ray与分布式调度]] §3.4。

### 3.5 Ray 在 RLHF 系统里的典型拓扑

```
┌────────────── Ray Cluster ──────────────┐
│  PT actor (trainer)    ──┐               │
│  RM actor (reward)       │  ray.get传权重│
│  Ref actor (KL ref)      │ ← /数据/控制  │
│  Rollout actors (vLLM)   │              │
│  PD actor (policy deploy)┘              │
│  各持 placement_group 的资源包          │
│  内部各自 torch.distributed 进程组      │
└──────────────────────────────────────────┘
```

PT 训完一段 → broadcast 新 $\theta$ 给 PD/rollout（[[weight sync mechanism]]）→ rollout 产轨迹 → RM 打分 → PT 用 PPO 更新。Ray 编排这一切的时序与资源。

## 4. 数学原理 / 公式

本条不涉及新公式，但调度成本可形式化：

- **一个 task 的端到端延迟** = 排队 + 调度 + 数据搬运 + 执行：
$$T_{\text{task}} = T_{\text{queue}} + T_{\text{sched}} + T_{\text{xfer}}(D) + T_{\text{exec}}$$
- $D$ 为输入数据量，$T_{\text{xfer}}(D) \approx D/B$（同机走共享内存 $B$ 极大、跨机走网络 $B$ 较小）；
- 小对象（<100KB）走 GCS pickle，大对象走 object store 零拷贝/大带宽。

## 5. 代码示例

```python
# ray_demo.py —— ray start --head; python ray_demo.py
import ray
ray.init(address="auto")  # 连已有集群，或 ray.init() 起单机

# 1) Task（无状态远程函数）
@ray.remote(num_cpus=2)
def gen_rollout(seed):
    import torch
    return torch.randn(8)  # 假装生成一条轨迹

fut = gen_rollout.remote(42)       # 立即返回 ObjectRef，不阻塞
data = ray.get(fut)                # 阻塞取回结果
print("rollout:", data[:3])

# 2) Actor（有状态远程对象）
@ray.remote(num_gpus=1)
class PolicyDeploy:
    def __init__(self):
        import torch
        self.theta = torch.zeros(8)
    def set_weights(self, w):
        self.theta = w
    def act(self, obs):
        return self.theta * obs

pd = PolicyDeploy.remote()
pd.set_weights.remote(torch.ones(8))     # 异步方法调用
out = ray.get(pd.act.remote(torch.tensor([1.,2.,3.])))
print("act:", out)

# 3) Placement Group（共置）
from ray.util.placement_group import placement_group, placement_group_table
pg = placement_group([{"GPU": 1, "CPU": 4}]*2, strategy="PACK")
ray.get(pg.ready())
pd2 = PolicyDeploy.options(num_gpus=1, placement_group=pg).remote()

ray.shutdown()
```

> [!tip] 实际工程
> - **PyTorch 集成**：用 `ray.train.torch.TorchTrainer` 自动起 process group，比手写 `init_process_group` 省；
> - **vLLM on Ray**：`VLLMWorker` 用 Ray actor 包 vLLM engine，按 rollout batch 调推理；
> - **权重刷新**：PT 端 `ray.get(weight_ref)` 或 NCCL broadcast 把 $\theta$ 推给 PD actor；
> - **资源画像**：trainer actor `num_gpus=8`、rollout actor `num_gpus=1`、RM actor `num_gpus=2`，按画像放卡。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[torch.distributed]]（Ray actor 内部跑的进程组）、[[NCCL backend]]（张量通信）、[[rank与world size]]（actor 类比 rank）、操作系统资源管理。
- **下游（应用）**: [[rollout worker]]、[[policy deployment (PD)]]、[[policy training (PT)]]、[[weight sync mechanism]]、[[asynchronous training]]、verl/OpenRLHF 等系统。
- **对比 / 易混**:
  - **Ray `ray.remote` vs [[RPC|torch.distributed.rpc]]**：概念同（远程调用+远程对象），实现独立。Ray 通用、生态广；PyTorch RPC 绑 NCCL、为模型并行定制。新项目倾向 Ray。
  - **Ray actor vs PyTorch rank**：actor 是 Ray 调度的有状态单元（可单进程可多），rank 是 `torch.distributed` 进程组里的身份；一个 Ray actor 内可跑多个 rank（如 `TorchTrainer`）。
  - **Ray 调度 vs K8s**：K8s 调度容器级粗粒度，Ray 调度 actor/task 级细粒度，常组合（K8s 起节点、Ray 在节点内调度）。
  - **Ray Object Store vs NCCL**：前者传通用对象（含张量，跨机走 TCP），后者专做 GPU 集体通信（NVLink/RDMA）。大张量同步走 NCCL，控制流走 Ray。
  - **Ray vs `torchrun`**：`torchrun` 起一组对等 rank 做集体训练，Ray 起异构 actor 做编排；RL 系统常 Ray 起 `torchrun` 在 actor 内跑。

### 本章子笔记索引

- **Ray基础**: [[actor model]]、[[task model]]、[[remote function]]、[[resource request]]
- **调度机制**: [[placement group]]、[[GPU绑定]]、[[scheduling策略]]
- **工程问题**: [[actor deadlock]]、[[resource starvation]]、[[task queue backlog]]

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **Ray 替代 NCCL** → 错。Ray 不管 GPU 集体通信，张量同步仍走 NCCL；Ray 管调度与编排。
> 2. **`@ray.remote` 函数闭包能 pickle** → 远程函数靠 pickle，闭包/lambda 不行；用顶层函数。
> 3. **大张量用 `ray.get` 同步传** → 跨机大张量走 object store 慢，能走 NCCL broadcast/all-reduce 就别走 Ray 对象传输。
> 4. **不设 placement group 导致 actor 散布** → trainer 和它要通信的 actor 被分到不同机，NCCL 跨机慢；用 PG 共置。
> 5. **`num_gpus` 超额** → 声明 1 GPU 占 1 张卡，超额会被调度器卡住；按真实需求声明。
> 6. **actor 死锁**：A actor 等 B actor 方法结果，B 等 A → [[actor deadlock]]；用 `asyncio` actor 或超时重试。
> 7. **rollout 快慢不一导致 task queue 积压** → [[task queue backlog]]；用动态 batching/超时丢弃/反压。
> 8. **资源饿死**：高优 actor 占满资源，低优永远排不上 → [[resource starvation]]；用优先级队列/资源预留。
> 9. **权重刷新靠 Ray 对象传** → 大 $\theta$ 走 Ray 慢且占带宽，应走 NCCL broadcast（`ray.train` 集成）。
> 10. **不 `ray.shutdown()`** → 集群资源残留；脚本末尾清理。

## 8. 延伸细节

### 8.1 Ray 的架构

- **Head node**：跑 GCS、调度器、dashboard；
- **Worker node**：跑 actor/task 进程、本地 object store；
- **GCS**：存 actor/task 元数据、placement group、object 位置；
- **Object Store (Plasma)**：每节点共享内存，跨进程零拷贝；
- **Raylet**：每节点本地调度 agent，管资源与进程起停。

### 8.2 与 vLLM 的集成

vLLM 可作 Ray actor：`Worker` class 用 `@ray.remote(num_gpus=1)` 包，主进程通过 `ray.get` 调 `step`。多卡张量并行时 vLLM 内部再起 `torch.distributed` 组。LLM-RL 系统里 rollout actor 就是这种 vLLM-on-Ray。

### 8.3 与 verl/OpenRLHF 的关系

verl、OpenRLHF 等 RL 训推系统把 PT/RM/Rollout/PD 各做成 Ray actor group（每组一个 placement group），用 Ray 编排 PPO 循环的时序：rollout→reward→advantage→PT update→weight sync。Ray 是它们的"操作系统"，PyTorch+vLLM 是"应用"。

> [!note] 解答：用 verl `ray_trainer.py` 逐段展开 single-controller PPO
> 用户指定文件 `verl/verl/trainer/ppo/ray_trainer.py`（1784 行）。这里是 verl 的 **`RayPPOTrainer`**，它就是上面 §3.5 那张拓扑图的"driver 进程"代码实现——一个跑在单 CPU 节点、不持卡的编排者，靠 Ray RPC 驱动多组 GPU actor 跑完整 PPO 数据流。下面按"建 actor → 跑循环"两段拆。
>
> > [!warning] 版本说明
> > `RayPPOTrainer` 标了 `@deprecated`（L283），verl v0.8.0 起改用 `main_ppo_sync.py`，但新版的 `RayPPOTrainer`（`verl/trainer/ppo/core_ppo.py`/`sync_ppo_trainer.py`）骨架一致：仍是 driver + RayWorkerGroup + fit() 主循环。这里以用户指定的 `ray_trainer.py` 为准讲解，原理不变。
>
> ### 一、`__init__`：driver 持有什么（L294-371）
>
> driver 进程**自己不加载模型、不持卡**，它只准备"编排所需的元数据"：
>
> | 字段 | 类型 | 作用 |
> |---|---|---|
> | `role_worker_mapping` | `dict[Role, WorkerType]` | Role→worker 类映射（`Role.ActorRolloutRef`/`Role.Critic`/`Role.RefPolicy`），决定每组 actor 跑什么 |
> | `resource_pool_manager` | `ResourcePoolManager` | 管各 role 的 placement group（几张卡、几节点、共置策略） |
> | `ray_worker_group_cls` | `type[RayWorkerGroup]` | worker group 实现类，可按 role 不同配不同 |
> | `kl_ctrl_in_reward` | `AdaptiveKLController` | 自适应 KL 系数 β，用于 in-reward KL penalty |
> | `train_dataloader/val_dataloader` | `StatefulDataLoader` | 续训可恢复的 dataloader（state 存 checkpoint） |
> | `_dump_executor` | `ThreadPoolExecutor` | 异步把 rollout 样本 dump 成 jsonl，不阻塞主循环 |
>
> 注意 `__init__` **不创建 actor**，只存配置。actor 在 `init_workers()` 里建。
>
> ### 二、`init_workers()`：把 Ray actor 起起来（L788-986）
>
> 这是"Ray 起若干 actor"那句话的落地代码，分四步：
>
> **步骤 1 — 建资源池**：`self.resource_pool_manager.create_resource_pool()` 按 config 给每个 Role 预占 placement group（`STRICT_PACK` 共置同机）。
>
> **步骤 2 — 包装 worker 类**：对每个 role 用 `RayClassWithInitArgs(cls=..., config=...)` 把"类+初始化参数"打包成可远程实例化的胶囊。
>
> **步骤 3 — colocate + spawn**（核心，L874-885）：
> ```python
> for resource_pool, class_dict in self.resource_pool_to_cls.items():
>     worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
>     ray_worker_group_cls = self._get_ray_worker_group_cls(set(class_dict.keys()))
>     wg_dict = ray_worker_group_cls(resource_pool=resource_pool,
>                                    ray_cls_with_init=worker_dict_cls, **wg_kwargs)
>     spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())  # 拆出各 role 的 wg
>     all_wg.update(spawn_wg)
> ```
> `create_colocated_worker_cls` 把多个 role 的 worker 类**塞进同一个 Ray actor 进程**（colocate，省显存：rollout 和 actor 共享同一份权重），`spawn(prefix_set)` 再按 role 前缀拆出多个 `RayWorkerGroup` 视图——**物理同一组 actor，逻辑多组 wg**，这是 verl 省卡的关键技巧。
>
> **步骤 4 — 赋值各 wg + 起子管理器**（L887-986）：
> ```
> self.critic_wg        = all_wg[str(Role.Critic)]      # critic actor group
> self.ref_policy_wg    = all_wg[str(Role.RefPolicy)]  # ref policy actor group
> self.actor_rollout_wg = all_wg[str(actor_role)]      # actor+rollout 混合 group
> self.reward_loop_manager = RewardLoopManager(...)   # RM（colocate 或独立）
> self.async_rollout_manager = AgentLoopManager.create(...)  # rollout 编排
> self.checkpoint_manager   = CheckpointEngineManager(...)   # 权重sync/ckpt
> self.llm_server_manager   = LLMServerManager.create(...)    # vLLM 引擎封装
> ```
> 到此 Ray 集群里所有 actor 已起好、进程组已建好（driver 通过 env_vars 注入 `MASTER_ADDR/PORT` 让 actor 内部 `init_process_group`，见 [[Ray与分布式调度]] §3.4 解答），driver 持有它们的 handle。
>
> ### 三、`fit()`：PPO 主循环（L1376-1784）
>
> 这就是 §3.5 拓扑图里"PT 训完 → broadcast 新 $\theta$ → rollout → RM → PPO 更新"的代码。driver 的 `fit()` 用 `marked_timer` 把每段标出来，时序一目了然：
>
> ```
> for epoch in range(...):
>   for batch_dict in self.train_dataloader:
>     batch = DataProto.from_single_dict(batch_dict)
>     # ── gen：rollout 采样 ──
>     combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
>     self.checkpoint_manager.sleep_replicas()          # 释放 rollout 显存给训练用
>     batch = batch.union(gen_batch_output)             # 把 response/logp 合进 batch
>     # ── reward：RM 打分 ──
>     if self.use_rm and "rm_scores" not in batch.batch.keys():
>         batch = batch.union(self._compute_reward_colocate(batch))
>     reward_tensor, reward_extra_infos_dict = extract_reward(batch)
>     # ── old_log_prob：重算 π_old 的 logp（PPO 锚点） ──
>     old_log_prob, mfu = self._compute_old_log_prob(batch)   # RPC actor_rollout_wg
>     batch = batch.union(old_log_prob)
>     # ── ref_log_prob：参考策略 logp（算 KL） ──
>     if self.use_reference_policy:
>         ref_log_prob = self._compute_ref_log_prob(batch)    # RPC ref_policy_wg
>         batch = batch.union(ref_log_prob)
>     # ── values：critic 估值 ──
>     if self.use_critic:
>         values = self._compute_values(batch)                # RPC critic_wg
>         batch = batch.union(values)
>     # ── adv：driver 本地算 advantage（轻量，不 RPC） ──
>     batch = compute_advantage(batch, adv_estimator=...,
>                               gamma=..., lam=..., num_repeat=rollout_n)
>     # ── update_critic：critic PPO 更新 ──
>     if self.use_critic:
>         critic_output = self._update_critic(batch)          # RPC critic_wg.train_mini_batch
>     # ── update_actor：actor PPO 更新 ──
>     actor_output = self._update_actor(batch)                # RPC actor_rollout_wg.update_actor
>     # ── save_checkpoint + weight sync ──
>     if 该存ckpt: self._save_checkpoint()
>     self.checkpoint_manager.update_weights(self.global_steps) # 新θ→rollout
>     # ── validate ──
>     if 到测试频率: val_metrics = self._validate()
> ```
>
> ### 四、关键观察：driver 是"脑子"，actor 是"手脚"
>
> 1. **driver 本地只做轻量计算**：`compute_advantage`、`apply_kl_penalty`、`extract_reward`、metric 聚合都在 driver CPU 上跑（L1634-1647、L76-115），不占卡。所有重活（forward/backward/rollout）都靠 `self.xxx_wg.method(batch).get()` RPC 发给 actor。
> 2. **`.get()` 是同步点**：每段 `with marked_timer("xxx", ...)` 里调 `wg.method(...)` 返 `ObjectRef`，紧跟 `.get()` 阻塞等结果——这就是 [[Ray与分布式调度]] §3.4 说的"非张量控制走 `ray.get`/Future"。时序的同步靠 `.get()`，actor 之间的张量同步靠各自进程组内的 NCCL。
> 3. **`sleep_replicas` / `update_weights` = 训推分离的 weight sync**：rollout 用的 vLLM 副本和训练用的模型在同一卡上 colocate，`sleep_replicas()` 把 vLLM 副本卸显存给训练 forward，训练完 `checkpoint_manager.update_weights(step)` 把新 $\theta$ 推回 vLLM 副本——这就是 [[训推分离]] §weight sync 的版本级同步在 colocate 场景的实现，详见 [[weight sync mechanism]]。
> 4. **`create_colocated_worker_cls` + `spawn`**：多个 role 物理同进程（省显存），逻辑分多个 wg（各自 RPC 方法）。这是 verl 区别于 OpenRLHF 的核心工程优化之一。
> 5. **async rollout**：`async_rollout_manager`（`AgentLoopManager`）封装 vLLM 的连续批处理 + 异步采样，driver 调 `generate_sequences` 是非阻塞的，可重叠 rollout 与上一步训练——这是 verl fully_async 模式的基础（见 [[asynchronous training]]）。
>
> ### 五、一句话总结
> `RayPPOTrainer` = **driver 进程持 RayWorkerGroup handle × N**，在 `fit()` 里按 `rollout→reward→old_logp→ref_logp→values→adv→update_critic→update_actor→weight_sync` 时序依次 `.remote().get()` 各 wg 方法，actor 内部各自跑 torch.distributed/NCCL 做张量同步。Ray 编排控制流与对象传输，PyTorch+vLLM 干计算——这正是 §3.5 拓扑图的代码化身。
>
> 相关：[[actor model]]、[[Ray与分布式调度]] §3.4/§3.5、[[weight sync mechanism]]、[[训推分离]]、[[asynchronous training]]、verl `single_controller/`。
### 8.4 Ray 的容错

- actor 默认挂了不重启（`max_restarts=0`）；设 `max_restarts=-1` 自动重启；
- 节点掉线，GCS 标记其上 actor 死亡，依赖它的 Future 抛错；
- 与 `torchrun --rdzv` 弹性不同：Ray 重启 actor，`torchrun` 重排 rank。

### 8.5 调试 Ray

> [!note] 已展开为独立笔记
> Ray 的调试与可观测性手段已单独成篇，见 [[Ray调试手段]]。该笔记含五层工具链（Ray Dashboard / Ray Distributed Debugger / ray timeline / ray list state APIs / `ray.get` 超时 + 异常）拆解、调度延迟公式各项对应工具、verl RLHF 调试流程、与 PyTorch/NCCL/nsys 的分工表、8 条误区。已联网核实 Ray Distributed Debugger 2026-04 GA（VSCode 扩展）、Anyscale 持久化 Dashboard 2026-05 GA。

- **dashboard**：`http://<head>:8265` 看 actor/task/资源/耗时；
- **`ray timeline`**：可视化 task 调度时序，找排队/搬运瓶颈；
- **`ray.get` 超时**：加 `ray.get(ref, timeout=...)` 防死等。
---
相关: [[09-Ray与分布式调度]]、[[torch.distributed]]、[[NCCL backend]]、[[rollout worker]]、[[policy deployment (PD)]]、[[policy training (PT)]]、[[weight sync mechanism]]、[[asynchronous training]]、[[RPC]]、[[3D parallelism]]
