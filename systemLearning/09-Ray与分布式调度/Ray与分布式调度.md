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

### 8.4 Ray 的容错

- actor 默认挂了不重启（`max_restarts=0`）；设 `max_restarts=-1` 自动重启；
- 节点掉线，GCS 标记其上 actor 死亡，依赖它的 Future 抛错；
- 与 `torchrun --rdzv` 弹性不同：Ray 重启 actor，`torchrun` 重排 rank。

### 8.5 调试 Ray

- **dashboard**：`http://<head>:8265` 看 actor/task/资源/耗时；
- **`ray timeline`**：可视化 task 调度时序，找排队/搬运瓶颈；
- **`ray.get` 超时**：加 `ray.get(ref, timeout=...)` 防死等。

---
相关: [[09-Ray与分布式调度]]、[[torch.distributed]]、[[NCCL backend]]、[[rollout worker]]、[[policy deployment (PD)]]、[[policy training (PT)]]、[[weight sync mechanism]]、[[asynchronous training]]、[[RPC]]、[[3D parallelism]]
