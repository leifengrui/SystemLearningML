# Ray调试手段

> **所属章节**: [[09-Ray与分布式调度]] §30 工程问题
> **所属模块**: [[09-Ray与分布式调度]]
> **难度**: 中
> **别名**: Ray debugging、Ray observability、Ray Dashboard、ray debug、Ray Distributed Debugger

## 1. 一句话定义

**Ray 调试手段**是一套从"Web 可视化（Ray Dashboard）"到"交互式断点（Ray Distributed Debugger）"再到"CLI 状态查询（ray list）+ 时序可视化（ray timeline）+ 代码级超时（`ray.get(timeout)`）"的多层工具链，用来在分布式、多 actor、多机的 Ray 应用里定位**挂起（hang）、死锁、资源饿死、OOM、调度延迟、任务排队**等单机调试器抓不到的系统性问题；它是 [[Ray与分布式调度]] §7 误区清单里那些坑的"破案工具"。

## 2. 为什么需要它（动机与背景）

单机 Python 程序挂了，你 `print` / `pdb` / IDE 断点就够。但 Ray 应用有三重"看不见"：

1. **多进程**：driver + N 个 actor 各自独立进程、各自 stdout/stderr，`print` 散落在几十个日志文件里，拼不成时间线；
2. **多机**：actor 跑在 worker node 上，你 SSH 上去翻 `/tmp/ray/session_latest/logs/` 才能看日志，跨机关联靠手工；
3. **异步 + Future**：`ray.get(ref)` 阻塞了——是 actor 在算？是排队没调度上？是死锁互相等？还是 OOM 被 raylet 杀了？从 driver 的 traceback 看不出来，只看到"卡住"。

所以 Ray 必须提供**集群级、跨进程、带时序**的观测与交互工具，否则分布式 bug 几乎无法复现定位。Ray 官方把这套能力叫 **Observability（可观测性）**，调试是它的核心用途之一。

## 3. 核心概念详解

Ray 调试手段可分五层，从"看全局"到"单步断点"层层下钻：

### 3.1 Ray Dashboard（Web UI，最常用）

- **入口**：`ray start --head` 启动后打印的 URL，默认 `http://<head>:8265`；或 `ray.init()` 返回的 context 对象里也有。Docker 里要加 `--dashboard-host=0.0.0.0`。
- **依赖**：`pip install "ray[default]"`（含 dashboard 组件）；Metrics view 需额外配 Prometheus + Grafana。
- **核心 Views**（tab）：

| View | 看什么 | 调试什么 |
|---|---|---|
| **Overview** | 集群级硬件利用率、autoscaling 状态（pending/active/failed 节点数） | 节点掉线、autoscaler 不扩容 |
| **Cluster** | 节点↔worker 进程↔actor 的层级树 + GPU 资源分配 | actor 放错机、GPU 没用满、CPU slot 被占 |
| **Jobs** | Job 进度条、失败 task/actor 列表、`ray status` 输出、pending 资源需求 | task 失败、资源不够调度不上（[[resource starvation]]） |
| **Logs** | 按节点/文件组织的日志，支持搜索过滤 | actor stderr 报错、OOM kill 记录 |
| **Metrics** | 每 task/actor 的 CPU/内存/GPU 时序曲线（需 Prometheus+Grafana） | 内存泄漏、某个 actor 吃满 CPU |
| **Events** | autoscaler/job 事件时间线 | 节点为何被踢、job 为何被杀 |
| **Serve** | Serve 应用状态与副本指标 | Ray Serve 部署调试 |

> [!tip] Cluster view 是定位"actor 放错机"的杀手锏
> 点开节点 → 看 worker → 看 actor 占了哪几张卡。verl 里如果发现 trainer actor 和 rollout actor 被分到不同机导致 NCCL 跨机慢，一眼就能看出来——治法是加 [[placement group]] 共置。

### 3.2 Ray Distributed Debugger（交互式断点，VSCode 扩展）

这是 2026 年 Ray 主推的新调试能力——让分布式 actor/task 里的 `breakpoint()` 能在 VSCode 里交互式单步调试，不再只能看日志猜。

- **组成**：debugger backend（跑在 Ray 集群里）+ VSCode 扩展前端（[Visual Studio Marketplace: ray-distributed-debugger](https://marketplace.visualstudio.com/items?itemName=anyscalecompute.ray-distributed-debugger)）。
- **原理**：在 task/actor 代码里插 `breakpoint()`（或 `ray.util.pdb.set_trace()`），运行时 actor 进程挂起，VSCode 扩展连上 backend，列出所有挂起的 task/actor，你选一个 attach 进去，像本地 pdb 一样看变量、单步、evaluate。
- **适用**：actor 内部逻辑 bug（reward 算错、logp 形状不对、rollout 死循环）——这种 bug 在 driver 端只看到结果错，得进 actor 进程看中间张量。
- **限制**：多 actor 同时命中断点时要手动选哪个 attach；大规模 RLHF 训练里慎用（会卡住整个 PPO 流水线），通常只在 debug 模式小规模复现时用。

> [!note] 2026 状态
> Ray Distributed Debugger 由 Anyscale 主导，2026-04 正式 GA（[Anyscale blog](https://www.anyscale.com/blog/ray-distributed-debugger)），VSCode 扩展已上架 marketplace。开源 Ray 2.55+ 内置 backend（`ray debug` CLI 也可触发）。Anyscale 工作区原生集成。

### 3.3 ray timeline（时序可视化）

- **产出**：Chrome tracing 格式文件，用 [Perfetto UI](https://ui.perfetto.dev/) 或 `chrome://tracing` 打开。
- **内容**：每个 task/actor 方法的调度时序——排队等待、数据搬运（object xfer）、执行时长，按 worker 分 lane铺开。
- **用途**：找**瓶颈段**。verl `fit()` 里每段都包了 `marked_timer("gen"/"reward"/"update_actor"/"update_weights", ...)`，结合 ray timeline 能看出是 rollout 慢、还是 weight sync 卡、还是 update_actor 在 all-reduce 拖时间——这正是 [[Ray与分布式调度]] §4 调度延迟公式 $T_{\text{task}}=T_{\text{queue}}+T_{\text{sched}}+T_{\text{xfer}}+T_{\text{exec}}$ 各项的可视化拆解。
- **开启**：`ray.init(runtime_env_config={"enable_ray_logging": True})` + `ray timeline` CLI，或代码里 `ray.timeline` API。

### 3.4 ray state APIs / `ray list`（CLI 查状态）

Dashboard 是 Web，`ray list` 是 CLI 等价物，适合 SSH/脚本/CI：

```bash
ray list actors          # 所有 actor 状态（ALIVE/DEAD/DEPENDENCY_FAILED）
ray list tasks           # 所有 task
ray list nodes           # 节点状态
ray list cluster-events  # 集群事件（节点掉线、autoscale）
ray list jobs            # job 状态
```

- **调试死锁**：`ray list actors` 看是否有大量 `PENDING` actor（调度不上=资源不够或 PG 没就绪）、`DEPENDENCY_FAILED`（依赖的 actor 挂了）。
- **调试 hang**：`ray list tasks` 看 task 是不是卡在 `RUNNING` 不结束——配合 actor 日志看是不是在 `ray.get` 互相等（[[actor deadlock]]）。

### 3.5 代码级防御：`ray.get` 超时 + 异常

```python
import ray
from ray.exceptions import GetTimeoutError, RayActorError, RayTaskError

try:
    result = ray.get(ref, timeout=30)   # 30 秒不返回就抛 GetTimeoutError
except GetTimeoutError:
    print("actor 卡住，去 dashboard 看 actor 状态")
except RayActorError:
    print("actor 进程挂了，看 ray list actors 是否 DEAD")
except RayTaskError as e:    # Ray 把远端异常重新抛到 driver
    print("task 内部抛了异常:", e)   # e 里含远端 traceback
```

- **`timeout`**：防 `ray.get` 永远阻塞的标配。verl 里 PPO 每步 RPC 都该设超时，否则一个 rollout actor hang 住整个训练。
- **`RayTaskError`**：Ray 自动把 actor/task 内部未捕获异常包装后重抛给 driver，traceback 完整——这是"分布式 try/except"的基础。

## 4. 数学原理 / 公式

本条不涉及新公式，但调试的本质是**定位调度延迟公式各项**（见 [[Ray与分布式调度]] §4）：

$$T_{\text{task}} = \underbrace{T_{\text{queue}}}_{\text{ray list tasks 看 PENDING}} + \underbrace{T_{\text{sched}}}_{\text{timeline 看 sched lane}} + \underbrace{T_{\text{xfer}}(D)}_{\text{timeline 看 object xfer}} + \underbrace{T_{\text{exec}}}_{\text{timeline 看 exec}}$$

每层工具对应公式的一项——这是把"为什么慢"拆成可定位的四个数。

## 5. 代码示例（可选）

### 5.1 最小调试：断点 + 超时

```python
# ray_debug_demo.py
import ray
from ray.exceptions import GetTimeoutError

ray.init(address="auto")

@ray.remote(num_cpus=1)
class BuggyActor:
    def work(self, x):
        breakpoint()            # 命中后挂起，VSCode Distributed Debugger attach 进来
        return x * 2

a = BuggyActor.remote()
try:
    out = ray.get(a.work.remote(21), timeout=10)   # 防卡死
except GetTimeoutError:
    print("超时——actor 在 breakpoint 等你 attach，或真卡了")

ray.shutdown()
```

### 5.2 程序化查 actor 状态（无 GUI 环境）

```python
import ray
ray.init(address="auto")

# 列所有 actor 及状态，无需打开浏览器
for a in ray.util.list_actors():
    print(a["name"], a["state"], a.get("death_cause"))
# 状态: PENDING(排队) / RUNNING / DEAD / DEPENDENCY_FAILED
```

### 5.3 verl 风格：每段 RPC 加超时 + 日志

```python
# 仿 verl RayPPOTrainer.fit() 的防御式写法
def safe_rpc(wg, method, batch, timeout=300, tag="step"):
    try:
        ref = getattr(wg, method).remote(batch)
        return ray.get(ref, timeout=timeout)
    except GetTimeoutError:
        print(f"[{tag}] {method} 超时 {timeout}s，查 dashboard actor 状态 + ray list tasks")
        raise
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[Ray与分布式调度]]（章总览，调试对象就是它的 actor/task/object）、[[actor model]]（调试单元）、[[placement group]]（调试"放错机"要懂 PG）、[[resource request]]（调试资源不够要懂 request 语义）。
- **下游（应用）**: [[actor deadlock]]（用 dashboard + ray list 定位）、[[resource starvation]]（用 Metrics view 的 CPU 曲线定位）、[[task queue backlog]]（用 Jobs view 的 pending 队列定位）、verl/OpenRLHF 的 PPO hang 排查。
- **对比 / 易混**:
  - **Ray Dashboard vs `torchrun` 日志**：`torchrun` 是每个 rank 一个 stdout，手工拼；Ray Dashboard 集中 Web UI 跨进程聚合。RL 系统里 Ray 编排层用 dashboard，rank 内部训练用 torch profiler/nsys，两者互补。
  - **Ray Distributed Debugger vs `pdb`**：pdb 只能调单进程；Ray Distributed Debugger 能跨 actor attach，但大规模训练慎用（会卡流水线）。
  - **Ray timeline vs [[torch profiler]] / [[nsys (Nsight Systems)]]**：ray timeline 看的是**调度/编排层**（task 排队、object 搬运、actor 方法时序），nsys/torch profiler 看的是**单进程 GPU kernel 层**（kernel 耗时、显存、NCCL 调用）。前者管"谁先谁后"，后者管"算得多快"，互补。
  - **Ray state APIs vs Kubernetes logs**：K8s 管 pod/容器级粗粒度，Ray state APIs 管 actor/task 级细粒度。KubeRay 部署时两层都要。

### 本章子笔记索引

- **Ray基础**: [[actor model]]、[[task model]]、[[remote function]]、[[resource request]]
- **调度机制**: [[placement group]]、[[GPU绑定]]、[[scheduling策略]]
- **工程问题**: [[actor deadlock]]、[[resource starvation]]、[[task queue backlog]]、[[Ray调试手段]]

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **Dashboard 打不开** → 忘了 `pip install "ray[default]"`（只装了 `ray` 裸包不含 dashboard 组件）；或 Docker 里没加 `--dashboard-host=0.0.0.0`。
> 2. **Metrics view 空白** → 没配 Prometheus + Grafana。Dashboard 的 CPU/内存时序曲线全靠 Prometheus 抓指标，不配就只有静态状态没有时序。
> 3. **`ray.get` 不设 timeout 然后整训练 hang 死** → 一个 actor 死锁，driver 永远等，8 卡空转烧钱。生产代码**必须**加 `timeout`。
> 4. **Ray Distributed Debugger 在大规模训练里乱插 breakpoint** → actor 挂起等 attach，整条 PPO 流水线停摆，几十分钟没人 attach 就烧完一个 plan。只在 debug 模式小规模复现时用。
> 5. **只看 driver traceback 不看 `RayTaskError`** → actor 内部异常会被 Ray 包装重抛，driver 的 traceback 第一眼是 `RayTaskError`，要点开 `.cause` 看真正的远端错误，否则误判成"Ray 出 bug"。
> 6. **把 ray timeline 当 nsys 用** → ray timeline 没有 GPU kernel 级粒度，看的是调度层；要看 NCCL kernel 用 nsys，两者互补不是替代。
> 7. **`ray list actors` 看到一堆 DEAD 就以为 bug** → 正常结束的 actor 也会进 DEAD，要看 `death_cause` 字段区分正常退出还是异常死亡。
> 8. **Dashboard 数据集群关了就没了** → 开源 Ray dashboard 是内存态，cluster 一关全丢。Anyscale 的 Cluster/Actor Dashboard 做了持久化（2026-05 GA），可事后复盘；自建集群要做 post-mortem 得自己导出 `ray list` + 日志到外部存储。

## 8. 延伸细节

### 8.1 日志在哪

- **路径**：每节点 `/tmp/ray/session_latest/logs/`，文件名含 `worker-*.out`（stdout）、`worker-*.err`（stderr）、`gcs_server.out`、`raylet.out`、`dashboard.log`。
- **Dashboard Logs view** 直接 Web 搜，不用 SSH；也支持按 task/actor 过滤。
- **driver 日志**：driver 进程自己的 stdout/stderr 不在 Ray 日志目录，是你跑脚本的那个终端/重定向文件。

### 8.2 调试 RLHF 系统的典型流程（verl）

1. 训练 hang → `ray list tasks` 看哪个 `RUNNING` 久 → Dashboard Jobs view 看进度条停在哪段；
2. 停在 `gen` 段 → 进 Cluster view 看 rollout actor 状态：`PENDING` 说明资源不够/PG 没就绪；`RUNNING` 但不出结果 → actor 日志看 vLLM 是否 OOM 或死循环；
3. 停在 `update_weights` 段 → 看 NCCL 是否跨机（Cluster view 看 actor 是否共置），timeline 看 object xfer 是否爆；
4. 数值错（reward/logp 异常）→ 小规模 debug 模式跑，在 actor 的 `compute_log_prob`/`compute_rm_score` 里插 breakpoint，VSCode Distributed Debugger attach 看中间张量；
5. 复盘已结束的 job → 自建集群得事先导出日志+`ray list`；Anyscale 用户直接看持久化 Dashboard。

### 8.3 与 PyTorch/NCCL 工具的分工

| 层次 | 工具 | 看什么 |
|---|---|---|
| 编排/调度层 | Ray Dashboard / ray list / ray timeline | actor 状态、task 排队、资源放置、object 搬运 |
| 进程组/通信层 | [[NCCL backend]] 日志、`ray list` 看 actor | NCCL 拓扑、集合通信报错、rank 死亡 |
| GPU kernel 层 | [[nsys (Nsight Systems)]]、[[torch profiler]] | kernel 耗时、显存、NCCL kernel 调用 |
| 训练逻辑层 | wandb/swanlab 指标（见 [[RLHF训练监控指标详解]]） | loss/kl/reward 曲线 |

四层从上到下，先看 Ray（编排）定位哪段卡，再看 PyTorch（通信+kernel）定位段内为什么慢，最后看训练指标定位数值对不对——**Ray 调试是这条链的最上层入口**。

### 8.4 版本与生态

- Ray 2.55+ 内置 Distributed Debugger backend；2.54 之前主要是 Dashboard + `ray.get` timeout。
- Anyscale（Ray 背后公司）2026-05 GA 了持久化 Cluster/Actor Dashboard，解决开源 dashboard"集群关了数据没了"的痛点；自建集群可参考其 Event Export Framework 思路自己接外部存储。
- verl 的 `global_profiler` 配置（`ray_trainer.py` L861-871）支持 `nsys` 工具按 step 采样，是 Ray 调度层 + nsys kernel 层的联动调试入口。

---
相关: [[09-Ray与分布式调度]]、[[Ray与分布式调度]]、[[actor deadlock]]、[[resource starvation]]、[[task queue backlog]]、[[placement group]]、[[nsys (Nsight Systems)]]、[[torch profiler]]、[[RLHF训练监控指标详解]]
