# worker 与 executor

> **所属章节**: [[runner worker executor]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: worker 与 executor / Worker / WorkerWrapperBase / Executor / 进程级执行壳 / TP/PP worker / model executor
> **难度**: 中高（需懂 [[model runner]] + [[Tensor Parallel]] + [[Pipeline Parallel]] + [[vLLM scheduler]] + [[推理 CUDA Graph]]）


## 1. 一句话定义

**worker 与 executor** 是 vLLM/SGLang 推理引擎的**进程级执行层**：**executor** 是"如何拉起一组 GPU 进程"的抽象（单进程 / 多进程 / Ray 分布式），它负责创建 worker、初始化分布式环境、加载权重、并把 engine 的每次 `execute_model` 调用**广播**到所有 worker；**worker** 是"每个 GPU 进程里的执行壳"，持有一份模型副本 + 一个 [[model runner]]，负责 `init_device`、`load_weights`、分配 KV cache、在 forward 中做 **TP all-reduce** 与 **PP send/recv**、执行采样并回结果。executor 管"拓扑与调度调用"，worker 管"单卡执行与跨卡通信"。vLLM v1 的实现：executor 基类 `vllm/v1/executor/abstract.py:Executor`，单进程 `UniProcExecutor`、多进程 `MultiprocExecutor`、Ray 分布式 `RayGPUExecutor`；worker 基类 `vllm/v1/worker/worker_base.py:WorkerBase`，RPC/本地派发壳 `WorkerWrapperBase`，GPU 实现 `vllm/v1/worker/gpu_worker.py:Worker`。SGLang 对应 `Scheduler`（含进程通信）+ `tp_worker.py:TpWorker`。

> [!note] 三句话定位
> - **是什么**：executor 拉起 N 个 GPU 进程（worker）、把调用广播过去；worker 持模型+runner、做 TP all-reduce / PP send-recv、执行 forward+采样。
> - **为什么**：多 GPU 推理要把模型切到多卡（TP/PP），需一个"进程级壳"管通信与执行；engine 只跟 executor 对话，不直接碰 worker。
> - **与 [[model runner]] 关系**：runner 是 GPU 执行核心（建张量/forward/采样），worker 是持 runner 的进程壳（加通信/权重加载/KV 分配）。executor 是 worker 的管理者。


## 2. 为什么需要它（动机与背景）

### 2.1 大模型单卡装不下 / 单卡太慢

70B 模型 fp16 要 ~140GB，单 GPU（80GB）放不下，必须 **[[Tensor Parallel]]** 切到多卡（每卡持 1/N 权重）。或用 **[[Pipeline Parallel]]** 把层分到多卡。多卡 = 多进程（每进程绑一 GPU），需要一个"执行层"管理这些进程、协调通信。

### 2.2 engine 不应直接碰 worker

engine（[[vLLM scheduler]] 所在的 LLMEngine）跑在主进程，逻辑是"调度决策"。若它直接操作每个 worker（spawn 进程、建 NCCL group、发 RPC、收结果），会把分布式细节混进调度逻辑，难维护、难异步。故引入 **executor**：engine 只调 `executor.execute_model(plan)`，executor 负责把调用送到 worker(s)、聚合结果——**executor 屏蔽了"单卡 / 多卡 / Ray"的差异**，engine 无感。

### 2.3 worker 要管 runner 之外的"跨卡通信与生命周期"

[[model runner]] 只管"单卡 forward+采样"，但 TP 要在 linear 后 all-reduce、PP 要在 stage 间 send/recv activation、KV cache 要按 rank 分配、权重要按 TP 切片加载、还要在线更新权重（RL 场景）。这些"跨卡 + 生命周期"的活不属于 runner（runner 不该知道有几个 rank），故由 **worker** 承担——worker 持 runner，在 forward 前后包通信。

### 2.4 异步调度需要进程隔离

async scheduling 下，scheduler 决策 N+1 时 GPU 在跑 N。要重叠 CPU 决策与 GPU 执行，worker 常跑在**独立进程**（`VLLM_ENABLE_V1_MULTIPROCESSING`），engine 与 worker 经 RPC/IPC 通信，避免 GIL 互相阻塞。executor 负责这套进程拓扑的拉起与 RPC。


## 3. 核心概念详解

### 3.1 三层架构：engine → executor → worker → runner

```
┌─────────────────────── 主进程 (engine) ────────────────────────┐
│  LLMEngine / AsyncLLMEngine                                     │
│    ├─ vLLM scheduler (决策: batch 组成, preempt, KV 分配)      │
│    └─ Executor (抽象: 怎么拉起/调用 worker)                      │
│         │                                                       │
│         │  execute_model(scheduler_output)  ← 广播到所有 worker  │
│         │  collective_rpc(method, args, non_block=...)          │
│         ▼                                                       │
└─────────┼───────────────────────────────────────────────────────┘
          │  (本地调用 或 RPC/IPC 到子进程)
          ▼
┌──── worker 进程 0 (GPU0) ────┐  ┌──── worker 进程 1 (GPU1) ────┐
│  WorkerWrapperBase (RPC 派发)  │  │  WorkerWrapperBase            │
│    └─ Worker (gpu_worker)      │  │    └─ Worker                  │
│          ├─ init_device         │  │          ├─ init_device       │
│          ├─ load_weights (TP 切片)│ │          ├─ load_weights     │
│          ├─ initialize_from_config (KV 分配)│ ├─ initialize_from_config │
│          ├─ GPUModelRunner       │  │          ├─ GPUModelRunner   │
│          │    (forward + sample) │  │          │  (forward+sample) │
│          ├─ TP: all-reduce (NCCL)│  │          ├─ TP all-reduce    │
│          └─ PP: send/recv       │  │          └─ PP send/recv     │
└────────────────────────────────┘  └──────────────────────────────┘
          ▲  TP all-reduce / PP send-recv 经 NCCL (跨进程)  ▲
          └──────────────────────────────────────────────────┘
```

- **engine**：跑 scheduler，每 iteration 调 `executor.execute_model(plan)`。
- **executor**：决定"worker 在哪（本进程 / 子进程 / Ray actor）、怎么调它们"。`collective_rpc` 把同一方法名广播到所有 worker，聚合返回。
- **worker**：每 GPU 一个，持 runner + 模型副本，做通信与执行。
- **runner**：worker 内的 GPU 执行核心（[[model runner]]）。

### 3.2 Executor 抽象（拉起与调用）

vLLM v1 executor 基类 `Executor`（`vllm/v1/executor/abstract.py`），按部署形态分：

| Executor | 文件 | 适用 | worker 形态 |
|---|---|---|---|
| **`UniProcExecutor`** | `uniproc_executor.py` | 单 GPU | worker 在**本进程**（`driver_worker`，无 RPC） |
| **`ExecutorWithExternalLauncher`** | 同上（子类） | torchrun 兼容，每 engine 一个 worker，多 engine 协同 | 本进程，靠 torchrun 拉多 engine |
| **`MultiprocExecutor`** | `multiproc_executor.py` | 单机多 GPU | spawn 子进程，每进程一 worker，IPC 通信 |
| **`RayGPUExecutor`** | `ray_distributed_executor.py` | 多机多 GPU | Ray actor 持 worker，跨机 RPC |

executor 的核心方法（源自 `UniProcExecutor` 源码）：

```python
class UniProcExecutor(Executor):
    def _init_executor(self):
        # 1. 建 worker 壳 (RPC rank 0 = 本进程)
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        # 2. 分布式参数 (init_method, rank, local_rank)
        method, rank, local_rank = self._distributed_args()
        self.driver_worker.init_worker(all_kwargs=[dict(vllm_config=..., rank=rank, ...)])
        self.driver_worker.init_device()        # set CUDA device
        self.driver_worker.load_model()         # 加载权重 (TP 切片)
        current_platform.update_block_size_for_backend(...)

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None,
                       non_block=False, single_value=False):
        # 广播方法到 worker; non_block=True 返回 Future (async)
        result = run_method(self.driver_worker, method, args, kwargs)
        if non_block: return AsyncOutputFuture(result, single_value)
        return [result] if not single_value else result

    def execute_model(self, scheduler_output, non_block=False):
        return self.collective_rpc("execute_model", args=(scheduler_output,),
                                   non_block=non_block, single_value=True)

    def supports_async_scheduling(self): return True   # UniProc 支持 async
```

> [!note] 补充：collective_rpc 是 executor 的统一调用入口
> engine 不直接调 worker 方法，全经 `collective_rpc(method_name, ...)`。单进程 = 直接调；多进程/Ray = RPC 广播到所有 worker 并 gather。`non_block=True` 让 engine 不等结果即返回 Future，实现 async scheduling（scheduler 决策 N+1 时 worker 在跑 N）。

### 3.3 Worker（进程级执行壳 + 通信）

vLLM v1 worker 三层：

| 类 | 文件 | 职责 |
|---|---|---|
| **`WorkerBase`** | `worker_base.py` | 抽象基类：定义 `execute_model`/`load_model`/`init_device` 等接口 |
| **`WorkerWrapperBase`** | `worker_base.py` | RPC/本地**派发壳**：`init_worker`、`init_device`、`load_model`；把 RPC 调用转给真正的 `Worker`。这就是"local-or-distributed worker"概念——同一壳在单进程是本地直调、多进程是 RPC 派发 |
| **`Worker`** | `gpu_worker.py` | **GPU 实现**：持 `GPUModelRunner`、分配 KV、TP all-reduce、PP send/recv、在线更新权重 |

`Worker` 的关键方法（源自 `gpu_worker.py` API）：

| 方法 | 作用 |
|---|---|
| `initialize_from_config(kv_cache_config)` | 按 rank 分配 GPU KV cache（[[KV block manager]] 各 rank 一份） |
| `determine_available_memory()` | profile 模型峰值显存 → 算可用 KV 显存（与 [[model runner]] 的 profiling 协同） |
| `execute_model(scheduler_output)` | 调 runner forward+采样；含 TP all-reduce、PP send/recv |
| `start_weight_update()` / `update_weights()` / `finish_weight_update()` | 在线权重更新（RL/PD 场景热加载） |
| `start_draft_weight_update()` | 重定向到 spec decode draft model |
| `init_weight_transfer_engine()` | 初始化权重传输引擎（如 NCCL/RDMA） |
| `update_max_model_len()` | 自适应显存后更新最大模型长度 |
| `get_kv_connector_handshake_metadata()` | 跨节点 KV 传输（disaggregated prefill）握手 |

### 3.4 TP 通信：linear all-reduce

[[Tensor Parallel]] 把每层 linear 的权重按列（output dim）切到 N 卡，每卡算自己那份得部分和，需 **all-reduce** 求和得完整激活。vLLM 用 NCCL process group，`init_worker_distributed_environment(backend='nccl')` 建组，linear 层后调 `tensor_model_parallel_all_reduce`。

```
TP=2, 某层 linear (X @ W):
  GPU0: y0 = X @ W0   (部分和, [B, hidden/N])
  GPU1: y1 = X @ W1   (部分和)
  all-reduce: y = y0 + y1  (每卡都拿到完整 [B, hidden])   <-- NCCL all-reduce
  -> 下层用完整 y
```

attention 的 QKV 也按 head 切，attention 内各卡算自己的 head（不通信），O proj 后再 all-reduce。故 TP 每 transformer 层有 **2 次 all-reduce**（attention O proj 后、MLP down 后，实际 fused）。all-reduce 是 TP 的主要通信开销，走 NVLink/IB。

### 3.5 PP 通信：stage 间 send/recv tensor

[[Pipeline Parallel]] 把层切到多个 stage（rank），第 i stage 算完一段把**中间激活（IntermediateTensors）**发给 i+1 stage。vLLM 用 `send_and_recv_tensor` / `batch_send_recv`，基于 NCCL point-to-point。

```
PP=2:
  stage0: forward(L0..Lk) -> intermediate_tensors (hidden states)
          send(intermediate_tensors) ──▶ stage1
  stage1: ◀── recv(intermediate_tensors)
          forward(Lk+1..Lend) -> logits -> sample
```

**异步 PP 通信**（`AsyncIntermediateTensors`，源自 `gpu_worker.py`）：send/recv 返回 `comm_handles`（NCCL 句柄），下游**懒等待**（`wait_for_comm()`）才真正同步——让通信与计算 overlap。executor 调 driver worker（最后一 stage）拿最终 logits/采样结果，中间 stage 只 send 不返回。

### 3.6 SGLang 对应组件对比

SGLang 的进程级执行层与 vLLM 形似但命名/分工略不同（基于 GitHub main 分支 2026 年结构，类名随版本演化，待核实细节）：

| vLLM | SGLang | 说明 |
|---|---|---|
| `Executor`（单/多进程/Ray） | `Scheduler` 内的进程通信 + `TpWorker` | SGLang 把"调度"与"worker 通信"更紧耦合在 `Scheduler` 里 |
| `Worker`（gpu_worker） | `TpWorker`（`managers/tp_worker.py`） | 每 GPU 进程一 worker，持 ModelRunner |
| `WorkerWrapperBase`（RPC 派发） | `ModelRpcClient` + `TpWorker` | SGLang scheduler 经 RPC client 调 worker |
| `GPUModelRunner`（runner） | `ModelRunner`（`model_executor/model_runner.py`） | GPU 执行核心 |
| TP all-reduce（NCCL） | 同（`get_dp_local_local_world_size` 等 NCCL 组） | 机制一致 |
| PP send/recv | `scheduler_pp_mixin.py` | SGLang 用 mixin 给 scheduler 加 PP 能力 |

SGLang 倾向把 scheduler 与 worker 通信做成 **producer-consumer**（scheduler 线程产生 batch、worker 进程消费），而 vLLM 把 executor 做成显式的"调用广播层"。两者目标一致：把决策（CPU）与执行（GPU）解耦、支持 TP/PP/多机。

### 3.7 colocate（共置）

**colocate** 指"worker 与 engine 是否同进程"：

- **单 GPU / debug**：worker 与 engine **同进程**（`UniProcExecutor`，`driver_worker` 即本进程对象，无 RPC 开销）——colocate。
- **多 GPU**：worker 在**独立进程**（`MultiprocExecutor` spawn / `RayGPUExecutor` actor），engine 经 RPC 调——non-colocate。
- **async scheduling**：通常 non-colocate（独立进程避 GIL 互锁），但 `UniProcExecutor` 也支持 `supports_async_scheduling=True`（靠 `non_block` Future 而非进程隔离）。

> [!tip] 实践：colocate 的取舍
> colocate 省 RPC/IPC 开销、调试简单，但 engine 与 worker 共享 GIL，CPU 密集的输入准备会与 GPU 执行互相阻塞；non-colocate 隔离好、能真 overlap，但有 IPC 序列化开销。vLLM 默认多 GPU 走 non-colocate（`VLLM_ENABLE_V1_MULTIPROCESSING=1`），单 GPU 可 colocate。


## 4. 数学原理 / 公式

### 4.1 TP all-reduce 的通信量

TP=N，每层 2 次 all-reduce（attention O proj + MLP down，或 fused 为若干次）。每次 all-reduce 通信量（每卡收发）：

$$
V_{\text{AR}} = 2 \cdot \frac{B \cdot H \cdot \text{dtype\_bytes}}{1} \quad\text{(ring all-reduce 每卡发收约 }2\tfrac{M(N-1)}{N}\text{)}
$$

精确地，ring all-reduce 每卡发送量 $\frac{N-1}{N}M$、接收量 $\frac{N-1}{N}M$（$M$=张量大小），总点对点量 $2\frac{N-1}{N}M$。TP 通信量正比于 batch×hidden，与层数成正比。详见 [[Tensor Parallel]] 的 α-β 模型。

### 4.2 PP send/recv 的通信量

PP=M stage，stage 间传 intermediate_tensors（hidden states，大小 $B\times L_{\text{stage}}\times H$）。每步通信量：

$$
V_{\text{PP}} = B \cdot L_{\text{last\_layer}} \cdot H \cdot \text{dtype\_bytes}
$$

（只传一次激活，量小于 TP 的 per-layer all-reduce，故 PP 通信效率高但流水线有 bubble）。详见 [[Pipeline Parallel]] 的 bubble 分析。

### 4.3 async overlap 的收益

设单步 CPU 决策 $t_c$、GPU 执行 $t_g$、IPC/RPC 开销 $t_{\text{rpc}}$。

- **串行**（colocate 且同步）：$T = t_c + t_g$。
- **async overlap**（non-colocate + non_block）：决策 N+1 与执行 N 重叠，$T \approx \max(t_c + t_{\text{rpc}},\, t_g)$。当 $t_g \gg t_c + t_{\text{rpc}}$（decode GPU 主导），吞吐接近 $\frac{1}{t_g}$，CPU 决策"免费"。这正是 worker 独立进程 + executor `non_block` 的价值。


## 5. 代码示例（可选）

### 5.1 executor→worker 调用链（对齐 vLLM V1 源码结构）

```python
# === 主进程: engine 持 executor ===
class LLMEngine:
    def __init__(self, config):
        # 按 parallel_config 选 executor
        if config.parallel_config.world_size == 1:
            self.executor = UniProcExecutor(config)      # 单进程
        elif config.parallel_config.use_ray:
            self.executor = RayGPUExecutor(config)       # Ray 多机
        else:
            self.executor = MultiprocExecutor(config)    # 单机多进程

    def step(self, scheduler_output):
        # async: non_block=True, 返回 Future; 同步: non_block=False
        fut = self.executor.execute_model(scheduler_output, non_block=True)
        return fut            # engine 下轮再 .result() 取

# === executor: 广播到 worker ===
class UniProcExecutor(Executor):
    def execute_model(self, scheduler_output, non_block=False):
        return self.collective_rpc("execute_model",
                                   args=(scheduler_output,),
                                   non_block=non_block, single_value=True)

# === worker: 通信 + 调 runner ===
class Worker(WorkerBase):
    def __init__(self, vllm_config, rank, ...):
        self.model_runner = GPUModelRunner(...)    # 持 runner
        self.tp_group, self.pp_group = None, None

    def execute_model(self, scheduler_output):
        # PP 第一 stage 外, 先 recv 上一 stage 的 intermediate_tensors
        if not self.is_first_pp_stage:
            inter = recv_tensor(self.pp_prev_rank, self.pp_group)
        # runner forward + sample (内部 TP all-reduce 由 linear 层触发)
        out = self.model_runner(scheduler_output, inter)
        # PP 非末 stage: send 给下一 stage, 不返回最终结果
        if not self.is_last_pp_stage:
            send_tensor(out.hidden, self.pp_next_rank, self.pp_group)
            return None
        return out              # 末 stage 返回 token/logprob 给 executor
```

### 5.2 异步 PP 通信（AsyncIntermediateTensors，源自 V1）

```python
class AsyncIntermediateTensors(IntermediateTensors):
    """带 lazy comm 句柄的中间张量, 收方按需 wait"""
    def __init__(self, tensors, comm_handles=None, comm_postprocess=None):
        super().__init__(tensors)
        self._comm_handles = comm_handles      # NCCL send/recv 句柄
        self._comm_waited = False
    def wait_for_comm(self):
        if self._comm_waited: return
        for h in self._comm_handles: h.wait()  # 真正同步
        for fn in (self._comm_postprocess or []): fn()
        self._comm_waited = True
    def __getattribute__(self, name):
        # 访问 .tensors 前自动 wait -> 通信与计算 overlap
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)
```

### 5.3 TP all-reduce（线性层后）

```python
import torch.distributed as dist

def tensor_model_parallel_all_reduce(x):
    """TP: 各卡部分和 -> all-reduce 得完整激活"""
    dist.all_reduce(x, group=tp_group)     # NCCL, 走 NVLink/IB
    return x

# 模型内 linear (TP 列切):
class ColumnParallelLinear:
    def forward(self, x):
        y = x @ self.weight_this_rank     # [B, hidden/N], 部分和
        return tensor_model_parallel_all_reduce(y)  # -> [B, hidden] 完整
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[model runner]]（worker 持 runner 调 forward/采样）、[[Tensor Parallel]]（TP all-reduce 机制）、[[Pipeline Parallel]]（PP send/recv 与 bubble）、[[vLLM scheduler]]（executor 接收其执行计划）
- **下游（应用）**: [[推理 CUDA Graph]]（graph 捕获/回放在 worker 的 runner 内）、[[PagedAttention]] + [[KV block manager]]（worker 按 rank 分配 KV）、[[TTFT与TPOT]]（worker 的通信/执行延迟影响 TPOT；async overlap 降 TPOT）
- **对比 / 易混**:
  - worker vs runner：进程壳（通信/权重/KV 生命周期）vs GPU 执行核心（建张量/forward/采样）。worker 持 runner。详见 [[model runner]]。
  - executor vs scheduler：executor 管"拉起与调用广播"（拓扑），scheduler 管"决策 batch 组成"。engine 同时持两者。
  - vLLM `Executor` vs SGLang `Scheduler`：vLLM 把通信抽到 executor 层，SGLang 把通信耦合进 scheduler（见 3.6 表）。
  - colocate vs non-colocate：同进程省开销但 GIL 互锁 vs 独立进程能 overlap（见 3.7）。


## 7. 常见误区与易错点

> [!warning] 误区1：以为 executor 做 batch 调度
> 不做。executor 不管"谁进 batch"（那是 [[vLLM scheduler]]），只管"把已定计划的 execute_model 调用送到 worker"。混淆会误把调度 bug 找到 executor。

> [!warning] 误区2：以为 worker 只跑 forward
> worker 还做 TP all-reduce、PP send/recv、KV 分配、权重加载/在线更新、分布式环境初始化。runner 才是"纯 forward+采样"。

> [!warning] 误区3：TP=all-reduce、PP=不通信
> TP 每 all-reduce 通信量正比 batch×hidden 且每层多次；PP 只 stage 间传一次激活，量小但有流水线 bubble。两者通信模式完全不同，别混。

> [!warning] 误区4：单 GPU 也要开多进程
> 单 GPU 用 `UniProcExecutor`，worker 本进程直调（colocate），无 RPC 开销。多进程是为多 GPU + async 隔离 GIL，单 GPU 开多进程反而浪费。

> [!warning] 误区5：async scheduling 必须多进程
> 不必须。`UniProcExecutor.supports_async_scheduling=True`，靠 `non_block=True` 返回 Future 在单进程内也能 async（但 GIL 仍可能限制 overlap）。真 overlap 最好 non-colocate。

> [!warning] 误区6：PP 中间 stage 也要返回采样结果
> 只有最后一 stage 算 logits+采样。中间 stage 只 send intermediate_tensors 不返回结果；executor 只从 driver（末 stage）worker 取结果。误让中间 stage 采样会重复计算。

> [!warning] 误区7：KV cache 各 rank 共享
> 不共享。每 worker(rank) 独立分配自己的 KV 份（TP 下 attention 按 head 切，各卡存自己 head 的 KV；PP 下各 stage 存自己层的 KV）。`initialize_from_config` 按 rank 分配。


## 8. 延伸细节

- **在线权重更新（RL/PD）**：`start_weight_update`/`update_weights`/`finish_weight_update` 支持训练侧把新权重流式送推理 worker，免重启换权重——是 [[17-RL训推一体框架]] 的权重同步基础。`init_weight_transfer_engine` 可用 NCCL/RDMA 传权重。
- **disaggregated prefill**：`get_kv_connector_handshake_metadata` + KV transfer connector（NIXL/mooncake/lmcache 等，见 `vllm/distributed/kv_transfer/`）让 prefill 与 decode 分到不同 worker 集群、跨节点传 KV，是 PD 分离架构的执行层支撑。
- **elastic EP / 专家并行**：`VLLM_ELASTIC_EP_SCALE_UP_LAUNCH`、`elastic_ep_execute("load_model")` 支持 MoE 专家并行的弹性扩缩，worker 持部分专家。
- **driver worker 概念**：executor 的 `driver_worker` 是 rank 0（末 stage）worker，executor 直接持有它（单进程时即本进程对象），最终结果从它取；其他 rank worker 经 RPC/IPC。这是 vLLM "driver + workers" 模型。
- **check_health / shutdown**：executor 的 `check_health` 探活（UniProc 总健康）、`shutdown` 优雅停 worker 进程。Ray executor 还支持节点故障检测。
- **SGLang 的 breakable cuda graph 与 worker**：SGLang worker 的 `ModelRunner` 配 `runner_backend`（full/breakable/piecewise cuda graph backend），worker 进程内执行含 graph 的 forward，通信与 graph 回放在同进程内协调。

---
相关: [[runner worker executor]]、[[model runner]]、[[vLLM scheduler]]、[[Tensor Parallel]]、[[Pipeline Parallel]]、[[PagedAttention]]、[[KV block manager]]、[[推理 CUDA Graph]]、[[TTFT与TPOT]]
