# NCCL 异步错误定位

> **所属章节**: [[NCCL故障定位]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: NCCL 异步错误 / NCCL async error / NCCL 延迟错误上报 / collective 异步 abort / NCCL watchdog / 异步通信错误滞后 / ECONNRESET / NCCL INTERNAL ERROR
> **难度**: 高（需懂 [[NCCL核心机制]] / [[NCCL backend]] / [[NCCL通信拓扑]] / [[InfiniBand]] / [[RDMA与GPUDirect]] / [[PCIe]] / [[NCCL hang与timeout]] / [[actor deadlock]]）


## 1. 一句话定义

**NCCL 异步错误定位** 是 GPU 集合通信中**"错误已经发生，但因为 NCCL 默认异步执行、错误异步上报、滞后且混乱"** 这一类故障的定位方法。NCCL 的 collective（all-reduce/all-gather/send-recv）默认是**异步**的——调用 `dist.all_reduce()` 只是往 CUDA stream 提交一个 kernel，**立即返回**，真正的通信和计算在 stream 异步执行，错误也**异步上报**（可能延迟到下一次 `wait()`、下一次 collective、甚至几秒后才 surfacing）。结果是：**rank A 的网络/硬件已经出错（如 `ECONNRESET` 连接重置、IB `INTERNAL ERROR`、GPU ECC 不可纠正错误、NCCL `ERROR_IO_ERROR`），但报错滞后、且可能在另一个 rank 上先 surface，导致日志因果错乱、真凶难辨**。定位要靠 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`（异步错误转同步 abort，让 watchdog 在出错时立刻 abort collective、抛异常，而不是傻等或滞后报）+ `TORCH_NCCL_BLOCKING_WAIT=1`（让 `wait()` 阻塞到完成或超时，错误更早 surface）+ 全 rank `NCCL_DEBUG=INFO` 日志拼接因果。本篇讲**有错误的"卡住/崩溃"**（async error），与 [[NCCL hang与timeout]]（**无错误**卡死，hang）互补——**hang 是"没报错就卡住"，async error 是"报错了但滞后/混乱"**，两者现象都是"卡住/崩溃"但根因与定位路径不同。

> [!note] 三句话定位
> - **是什么**：NCCL collective 默认异步执行，错误异步上报、滞后且可能在别的 rank 先 surface，定位要靠"同步 abort + 阻塞 wait + 全 rank log 拼因果"。
> - **为什么难**：异步 = 错误时机错位 + 跨 rank 因果错乱 = 真凶（出错的 rank/硬件）被表象（先报错的 rank）掩盖，容易误判。
> - **与 [[NCCL hang与timeout]] 关系**：hang 是**无错误**卡死（rank 没到/数据没到，NCCL 在 wait 死等，无 error log，直到 timeout）；async error 是**有错误**（ECONNRESET/INTERNAL/ECC）但异步滞后上报。两者现象都是"卡住/崩溃"，但根因（无错 vs 有错）与定位手段不同。本篇讲后者（async error），那篇讲前者（hang/timeout）。


## 2. 为什么需要它（动机与背景）

### 2.1 NCCL 默认异步执行

[[NCCL核心机制]] 的 collective 是异步的：`dist.all_reduce(t)` 往 NCCL communicator stream 提交一个 kernel 后**立即返回 Python**，通信在 GPU stream 异步执行，`wait()` 时才真正阻塞等待完成。这种设计是为了**让通信与计算 overlap**（通信在跑的同时 Python 能继续发下一个或做别的），但副作用是**错误也异步**——rank A 的 IB 连接断了，错误可能在 stream 里延迟，直到下一次 poll CQ / 下一次 `wait()` 才 surface。

### 2.2 错误滞后导致因果错乱

异步 = **出错时机与报错时机错位**：

- rank 3 的 IB NIC 在 $t=10s$ 出错（连接重置）；
- 但错误在 NCCL 的 completion poll 里延迟，直到 $t=12s$ rank 3 的下一次 `wait()` 才 surface；
- meanwhile rank 7 因为 rank 3 没回数据，在 $t=11s$ 先报了自己的 `wait timeout`；
- 日志看：**rank 7 先报错，rank 3 后报**，但**真凶是 rank 3 的 NIC**。

这种因果错乱是 async error 定位的核心难点——**先报错的不一定是真凶**。

### 2.3 默认行为是"异步上报，不 abort"

PyTorch NCCL 后端默认（不开 `TORCH_NCCL_ASYNC_ERROR_HANDLING`）：
- 异步错误**不立即 abort** collective，错误信息进队列；
- 其他 rank 的 collective **继续在 stream 里跑**（在已损坏的连接/数据上），可能跑出错误结果或继续 hang；
- 错误可能在**下一次 collective** 才被检查到，滞后严重。

这导致两个问题：
1. **滞后**：错误不立即 surface，调试窗口长；
2. **损坏数据继续跑**：不 abort 意味着可能在坏数据上继续训练，结果错误（静默 corruption），比 hang 更危险。

### 2.4 解决思路：同步化 + 阻塞化 + 全量日志

定位 async error 的三件套：
1. **`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`**：让 watchdog 在检测到异步错误时**立即 abort collective**、抛 `DistributedTimeoutError`/`NCCLError`，错误早 surface、不继续跑坏数据；
2. **`TORCH_NCCL_BLOCKING_WAIT=1`**：让 `wait()` **阻塞到完成或超时**，错误在 wait 时直接报（而非滞后到下次 collective）；
3. **全 rank `NCCL_DEBUG=INFO NCCL_DEBUG_FILE=...`**：每个 rank 单独日志，**拼接因果**——找最早的 error 时间戳、找真凶 rank/硬件。

这三者配合，把"异步、滞后、混乱"的错误转成"同步、即时、可拼因果"的错误，才能定位。


## 3. 核心概念详解

### 3.1 异步执行的代价

| 阶段 | 同步世界 | 异步世界（NCCL 默认） |
|---|---|---|
| 调用 `all_reduce()` | 阻塞到完成才返回 | 立即返回（提交 kernel 到 stream） |
| 出错时机 | 当场报 | stream 里延迟，下次 poll/wait 才 surface |
| 报错 rank | 出错的 rank | 可能是依赖它的另一个 rank 先报 |
| 错误传播 | 局部 | 跨 rank 蔓延（坏数据继续跑） |
| 定位 | 直接看出错点 | 要拼全 rank log 找最早 error |

### 3.2 典型异步错误

| 错误 | 含义 | 常见根因 |
|---|---|---|
| `NCCL ECONNRESET` / `Connection reset by peer` | IB/TCP 连接被重置 | NIC 故障、对端 crash、SM 路径失效 |
| `NCCL INTERNAL ERROR` / `internal error -2` | NCCL 内部状态错 | 通信器损坏、上一次错误未清理 |
| `NCCL ERROR_IO_ERROR` / `system error` | IO 系统调用错 | NIC 驱动、IB verbs 错、ECC |
| `NCCL uncorrectable ECC` / GPU XID | GPU 硬件 ECC 不可纠正 | GPU 显存故障，需隔离/换卡 |
| `Watchdog caught collective operation timeout` | PyTorch watchdog 超时 | hang（rank 没到）或 async error 拖到 timeout |
| `ibv_poll_cq` error 12 / `transport error` | IB CQ 轮询错 | IB RC 连接断、链路降速/抖动 |
| `CUDART_ERROR` / `cudaError` | CUDA runtime 错 | stream/ECC/驱动 |

### 3.3 两个核心 PyTorch 环境变量

| 变量 | 作用 | 默认 | 值 |
|---|---|---|---|
| `TORCH_NCCL_BLOCKING_WAIT` | 让 `wait()` 阻塞到完成或超时（错误在 wait 时直接报，不滞后到下次 collective） | 0（非阻塞，但 watchdog 后台计时） | `1` 开 |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | watchdog 检测到异步错误时**立即 abort collective**、抛异常（不继续跑坏数据、错误早 surface） | 0（不 abort，错误进队列滞后报） | `1` 开 |

> [!warning] 误区：用旧名 `NCCL_BLOCKING_WAIT` / `NCCL_ASYNC_ERROR_HANDLING`
> 这是**旧名**（`NCCL_` 前缀）。PyTorch 2.x 推荐 **`TORCH_` 前缀**（`TORCH_NCCL_BLOCKING_WAIT` / `TORCH_NCCL_ASYNC_ERROR_HANDLING`），语义更明确（是 PyTorch ProcessGroupNCCL 的行为，不是 NCCL 库自己的）。旧名仍兼容但优先用新名。**别混用**，统一用 `TORCH_` 前缀。

> [!warning] 误区：以为这俩是 NCCL 库的变量
> 不是。它们是 **PyTorch `torch.distributed` 的环境变量**，控制 ProcessGroupNCCL 的 Python 层行为（wait 阻塞、watchdog abort）。NCCL 库自己有 `NCCL_DEBUG` 等变量（见 [[NCCL hang与timeout]] §3.4 表），但 `BLOCKING_WAIT`/`ASYNC_ERROR_HANDLING` 是 PyTorch 层。这是两套变量，别混。

### 3.4 定位流程（async error 怎么查）

**第一步：确认是 async error 而非 hang**
- 看 NCCL log：**有 error/warning**（ECONNRESET/INTERNAL/IO_ERROR/ECC）→ async error；
- 无 error、纯卡死 → hang（转 [[NCCL hang与timeout]]）。

**第二步：开同步化重跑**
```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # 错误立即 abort, 早 surface
export TORCH_NCCL_BLOCKING_WAIT=1           # wait 阻塞, 错误在 wait 报
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=nccl.%h.%p           # 每机一文件
```
重跑，让错误**早 surface、在出错点附近报**，缩短调试窗口。

**第三步：全 rank log 拼因果**
- 比对所有 rank 的 `nccl.<host>.<pid>` 日志；
- 找**最早的 error 时间戳**（不是先报错的 rank，是时间最早的 error）；
- 早 error 的 rank 多半是真凶或离真凶最近；
- 看 error 类型：
  - `ECONNRESET`/`ibv_poll_cq error` → 网络/NIC，查 `ibstat`/`dmesg`；
  - `ECC`/`XID` → GPU，查 `nvidia-smi -q ECC`；
  - `INTERNAL ERROR` → 通信器损坏，常是上一个错误没清理的级联。

**第四步：硬件定位**
```bash
nvidia-smi -q | grep -A5 "ECC Errors"      # GPU ECC
ibstat                                      # IB 端口/速率
ibsysstat / ibping                          # IB 链路质量
dmesg | grep -iE "mellanox|mlx|ib|nvidia|xid|ecc" | tail -30  # 内核错误
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total --format=csv
```

**第五步：隔离试验（复现/确认）**
- 怀疑 IB：`NCCL_IB_DISABLE=1` 退 TCP，看是否不报错（确认 IB）；
- 怀疑 GPUDirect：`NCCL_NET_GDR_LEVEL=0` 关 GDR；
- 怀疑 P2P：`NCCL_P2P_DISABLE=1`；
- 怀疑某节点：剔除该节点重跑，看是否稳定（定位坏节点）。

### 3.5 真凶 vs 表象

async error 定位的核心心法：**先报错的不一定是真凶**。

- rank A 的 NIC 在 $t_0$ 出错；
- rank A 的错误异步延迟到 $t_0+\Delta$ 才 surface；
- rank B 因依赖 A 的数据，在 $t_0+\delta$（$\delta<\Delta$）先报 wait timeout；
- 日志看 B 先报、A 后报，但**真凶是 A**。

定位靠**全 rank 时间戳拼接 + 找最早 error**，不是看谁先 crash。这也是为什么必须每 rank 单独 `NCCL_DEBUG_FILE`——合并日志会丢时间戳因果。

### 3.6 与 hang 的边界

| 维度 | hang（[[NCCL hang与timeout]]） | async error（本篇） |
|---|---|---|
| 有无错误 | 无 error log | 有 error（ECONNRESET/INTERNAL/ECC） |
| rank 状态 | 卡在 wait，没出错 | 出错了，错误异步滞后 |
| 现象 | 卡死不动 | 卡住或崩溃，有 error log |
| 典型根因 | rank 没到/数据没到/逻辑死锁 | NIC/IB/GPU 硬件错、连接重置 |
| 定位手段 | 找"谁没到调用点"+硬件 | 找"最早 error 时间戳"+硬件 |
| 核心变量 | `NCCL_DEBUG`/`NCCL_IB_*`/timeout | `TORCH_NCCL_ASYNC_ERROR_HANDLING`/`TORCH_NCCL_BLOCKING_WAIT`/`NCCL_DEBUG` |
| 关系 | 无错卡死，等 timeout | 有错滞后，可被同步化提前 surface |

> [!note] 两者交织
> 现实中 async error 未被同步化时，可能表现为 hang（错误滞后、其他 rank 在 wait 死等直到 timeout）。开了 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 后，async error 会早 surface 成 error 而非 hang。故**生产必开同步化**，把潜在的 hang 转 成明确的 async error，便于定位。


## 4. 数学原理 / 公式

### 4.1 异步错误延迟模型

设 rank $A$ 在时刻 $t_0$ 出错（NIC/GPU/IB），错误在 NCCL stream 的 completion poll 里延迟 $\Delta$ 秒后 surface：

$$
t_{\text{surface},A} = t_0 + \Delta
$$

依赖 $A$ 的 rank $B$ 因收不到数据，在 $t_0 + \delta$（$\delta < \Delta$，B 的 wait 先超时）报错：

$$
t_{\text{surface},B} = t_0 + \delta \quad (\delta < \Delta)
$$

故 $B$ 先报、$A$ 后报，但真凶是 $A$。定位须找 $\min(t_{\text{surface}})$ 对应的**最早 error**，而非先 crash 的 rank。

### 4.2 同步化缩短调试窗口

开 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 后，watchdog 在检测到异步错误时立即 abort：

$$
\Delta_{\text{同步化}} \approx 0 \quad \Rightarrow \quad t_{\text{surface},A} \approx t_0
$$

错误即时 surface，调试窗口从 $\Delta$（可能数秒）缩到 $\approx 0$，因果不再错乱。这是同步化的核心价值。

### 4.3 阻塞 wait 的错误时机

`TORCH_NCCL_BLOCKING_WAIT=1` 让 `wait()` 阻塞到完成或超时。错误在 wait 时直接报（而非滞后到下次 collective）：

$$
t_{\text{surface,wait}} = t_{\text{wait}} \quad (\text{而非 } t_{\text{next collective}})
$$

若不开，错误可能滞后到下一次 collective 调用时才被检查，延迟一个 step 周期（$T_{\text{step}}$），调试窗口扩大。

### 4.4 IB 错误的时间线

IB RC 连接出错（如 `ECONNRESET`）的典型时间线：
1. $t_0$：NIC/链路故障，IB RC 连接重置；
2. $t_0 + \epsilon$：NCCL 的 IB transport 在 `ibv_poll_cq` 拿到 error CQE；
3. $t_0 + \Delta_{\text{NCCL}}$：NCCL 上报到 PyTorch（异步，$\Delta$ 可能秒级）；
4. $t_0 + \Delta_{\text{NCCL}} + \Delta_{\text{watchdog}}$：watchdog abort collective（若开了 ASYNC_ERROR_HANDLING）。

总延迟 $\Delta_{\text{NCCL}} + \Delta_{\text{watchdog}}$ 是 async error 滞后的量化。同步化把 $\Delta_{\text{watchdog}}$ 缩到最小。


## 5. 代码示例

### 5.1 模拟 async error 并定位

```python
# async_error_repro.py —— 模拟 rank 0 的 collective 出错, 看异步滞后与同步化
# torchrun --nproc_per_node=4 async_error_repro.py
import os, time, torch
import torch.distributed as dist

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_FILE"] = "nccl.%h.%p"
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")   # 同步化: 错误立即 abort
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")          # 阻塞 wait: 错误在 wait 报
os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "10000")  # flight recorder

dist.init_process_group("nccl", init_method="env://",
                        timeout=__import__("datetime").timedelta(seconds=30))
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
rank = dist.get_rank()

for step in range(100):
    t = torch.ones(1_000_000, device="cuda") * rank
    dist.all_reduce(t, op=dist.ReduceOp.SUM)   # rank 0 若 NIC 坏, 这里异步出错
    torch.cuda.synchronize()
    if rank == 0 and step == 5:
        # 模拟: 让 rank 0 等一会, 制造异步滞后的观察窗口
        time.sleep(2)
    print(f"[rank {rank}] step {step} done", flush=True)
```

```bash
# 运行后查各 rank 的 nccl.<host>.<pid> 日志:
# - 找最早的 error 时间戳 (不是先 crash 的 rank)
# - ECONNRESET/INTERNAL/IO_ERROR -> 网络/通信器, 查 ibstat/dmesg
# - ECC/XID -> GPU, 查 nvidia-smi -q ECC
# - 开了 ASYNC_ERROR_HANDLING: error 附近就有 "watchdog aborting collective" -> 早 surface
```

### 5.2 生产同步化配置

```bash
#!/bin/bash
# nccl_async_safe.sh —— 生产环境 async error 同步化套件
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1     # 核心: 异步错误立即 abort, 不跑坏数据
export TORCH_NCCL_BLOCKING_WAIT=1            # wait 阻塞, 错误在 wait 报 (非滞后到下次 collective)
export NCCL_DEBUG=WARN                        # 生产用 WARN (INFO 太 verbose)
export NCCL_DEBUG_FILE=nccl_logs/nccl.%h.%p   # 每机一文件, 拼因果
export NCCL_IB_RETURN_ASYNC_EVENTS=1          # IB 致命事件停止通信 (默认 1, 保持)
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000     # flight recorder, 排 hang/async error

# 注意: ASYNC_ERROR_HANDLING=1 时, 任何 rank 出错都会 abort 整个 communicator,
#       所有 rank crash —— 这是期望行为 (早暴露, 不静默跑坏数据).
#       若要单 rank 错不影响其他 (容错训练), 需框架层 elastic/restart, 不是 NCCL 层.
```

### 5.3 全 rank log 拼因果脚本

```bash
#!/bin/bash
# nccl_causality.sh —— 拼接所有 rank 的 NCCL 日志, 找最早 error
LOGDIR=nccl_logs
echo "===== 所有 rank 的 error/warning (按时间排序) ====="
grep -Hn -iE "error|warn|ECONNRESET|INTERNAL|IO_ERROR|ECC|XID|abort|timeout" $LOGDIR/nccl.* \
  | sort -t: -k1 -k2 \
  | head -50
echo ""
echo "===== 每个文件的最早 error (找真凶 rank) ====="
for f in $LOGDIR/nccl.*; do
  first_err=$(grep -in -E "error|ECONNRESET|INTERNAL|IO_ERROR|ECC|XID" "$f" | head -1)
  [ -n "$first_err" ] && echo "$f : $first_err"
done
echo ""
echo "===== 硬件体检 ====="
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total --format=csv
ibstat 2>/dev/null | grep -E "mlx|State|Rate" || echo "no IB"
dmesg --time-format iso 2>/dev/null | grep -iE "mellanox|mlx|ib|nvidia|xid|ecc" | tail -20
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[NCCL核心机制]]（collective 异步执行、`async_op`/`wait`、stream）、[[NCCL backend]]（ProcessGroupNCCL、`TORCH_NCCL_*` 环境变量、watchdog）、[[NCCL通信拓扑]]（channel/transport、IB transport 的 `ibv_poll_cq` 错误源）、[[InfiniBand]]（IB RC 连接重置、SM、`NCCL_IB_*`）、[[RDMA与GPUDirect]]（GDR 失败退 CPU、`NCCL_NET_GDR_LEVEL`）、[[PCIe]]（ACS 阻断 P2P 的 IO 错误）。
- **下游（应用）**: [[NCCL hang与timeout]]（async error 未同步化时常表现为 hang，两者互补）、[[actor deadlock]]（RL 框架的 rank crash 导致 collective 出错）、[[集群拓扑感知placement]]（拓扑错导致 IO 错误）、[[3D parallelism]]（多 communicator 的错误级联）、elastic/restart 工程（async error abort 后的恢复）、[[all-reduce]]（最常出 async error 的 collective）。
- **对比 / 易混**:
  - **async error（本篇）vs hang（[[NCCL hang与timeout]]）**：async error 是**有错误**（ECONNRESET/INTERNAL/ECC）但异步滞后上报；hang 是**无错误**卡死（rank 没到/数据没到）。现象都是"卡住/崩溃"，但定位手段不同（async error 找最早 error 时间戳+硬件；hang 找谁没到调用点+硬件）。生产开 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 可把潜在 hang 转成明确 async error，便于定位。
  - **`TORCH_NCCL_BLOCKING_WAIT` vs `TORCH_NCCL_ASYNC_ERROR_HANDLING`**：前者控制 `wait()` 是否阻塞（错误在 wait 时报 vs 滞后到下次 collective）；后者控制检测到错误是否立即 abort（早 surface/不跑坏数据 vs 错误进队列滞后）。两者互补，生产都开。
  - **`TORCH_NCCL_*`（PyTorch 层）vs `NCCL_*`（NCCL 库层）**：前者是 PyTorch ProcessGroupNCCL 的行为（wait 阻塞、watchdog abort）；后者是 NCCL 库自己的（DEBUG/IB_TIMEOUT/P2P_DISABLE 等）。两套变量，别混。
  - **async error vs [[actor deadlock]]**：async error 是 NCCL 通信层的错误滞后；actor deadlock 是 Ray actor 间的死锁（互相 await）。RL 系统里 actor deadlock 会导致 rank 不调 collective，进而 hang 或 async error，两者因果链相连。


## 7. 常见误区与易错点

> [!warning] 误区1：用旧名 `NCCL_BLOCKING_WAIT` / `NCCL_ASYNC_ERROR_HANDLING`
> 这是**旧名**（`NCCL_` 前缀）。PyTorch 2.x 推荐 `TORCH_NCCL_BLOCKING_WAIT` / `TORCH_NCCL_ASYNC_ERROR_HANDLING`（`TORCH_` 前缀），语义更明确。旧名兼容但优先用新名，别混用。

> [!warning] 误区2：把这俩当 NCCL 库的变量
> 不是。它们是 **PyTorch `torch.distributed` 的环境变量**，控制 ProcessGroupNCCL 的 Python 层行为。NCCL 库自己的变量是 `NCCL_DEBUG`/`NCCL_IB_*` 等（见 [[NCCL hang与timeout]] §3.4）。两套变量，别混。

> [!warning] 误区3：以为先报错的是真凶
> 异步 = 因果错乱。rank A 出错但延迟 surface，rank B 因依赖 A 先报 timeout。日志看 B 先报、A 后报，但真凶是 A。定位靠**全 rank 时间戳拼接 + 找最早 error**，不是看谁先 crash。**必须每 rank 单独 `NCCL_DEBUG_FILE`**，合并日志丢因果。

> [!warning] 误区4：不开 ASYNC_ERROR_HANDLING，让坏数据继续跑
> 默认不 abort，错误进队列，其他 rank 的 collective 继续在坏连接/坏数据上跑，可能产出**静默 corruption**（训练结果错误但不报错），比 hang 更危险。生产必开 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`，早 abort 早暴露。

> [!warning] 误区5：把 async error 当 hang 处理
> 现象都是"卡住"，但 async error 有 error log（ECONNRESET/INTERNAL/ECC），hang 无。先看 log 区分：有 error = async error（本篇），无 error = hang（[[NCCL hang与timeout]]）。定位手段不同。

> [!tip] 实践：生产 async error 三件套
> 1. `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`（错误立即 abort，不跑坏数据）；
> 2. `TORCH_NCCL_BLOCKING_WAIT=1`（wait 阻塞，错误在 wait 报，不滞后）；
> 3. 每机 `NCCL_DEBUG=WARN NCCL_DEBUG_FILE=nccl.%h.%p`（拼因果）+ `TORCH_NCCL_TRACE_BUFFER_SIZE=10000`（flight recorder）。
> 三者配合把"异步滞后混乱"转成"同步即时可拼因果"。

> [!tip] 实践：定位 async error 的"三找"
> 1. **找最早 error 时间戳**（不是先 crash 的 rank）——真凶在最早 error 附近；
> 2. **找 error 类型**——ECONNRESET/ibv 错→网络/NIC，ECC/XID→GPU，INTERNAL→通信器损坏级联；
> 3. **找硬件佐证**——`ibstat`/`dmesg`/`nvidia-smi -q ECC` 证实软件 error 的硬件根因。

> [!note] 补充：RL 系统的 async error 更难
> RLHF/Agent 系统里，rollout actor 与 learner 解耦，rollout 侧的 rank crash/NIC 错会异步滞后到 learner 的权重同步 collective 才 surface，真凶在 rollout、表象在 learner。须跨角色拼因果。见 [[actor deadlock]]。


## 8. 延伸细节

### 8.1 Flight Recorder（黑匣子）

PyTorch 2.x 的 `TORCH_NCCL_TRACE_BUFFER_SIZE` 开启 NCCL 通信的 flight recorder——环形缓冲记录最近的 collective 调用（参数、时间戳、stream 状态）。hang/crash 时 dump 出来，看最后卡在哪条 collective、错误何时开始。是排 async error（错误滞后）的新利器，弥补"事后才看 log"的滞后。可配合 `FlightRecorderHook` 在 crash 时自动 dump。

### 8.2 IB `NCCL_IB_RETURN_ASYNC_EVENTS`

IB NIC 有"致命异步事件"机制——NIC 硬件检测到链路/端口故障时，异步通知 verbs 层。`NCCL_IB_RETURN_ASYNC_EVENTS=1`（NCCL 2.23 起默认 1）让 NCCL 在收到此类事件时**停止该 transport 的通信**，避免在坏链路上继续跑。排 async error 时保持开（默认即开），它能让 IB 硬件错早 surface 成 NCCL 错误。见 [[NCCL hang与timeout]] §3.4。

### 8.3 GPU ECC / XID 的 async 特性

GPU 显存 ECC 不可纠正错误或 XID 是**异步**的——GPU 硬件在出错时标记，但 NCCL kernel 在 stream 里继续跑，直到下一次同步点（`cudaDeviceSynchronize`/`wait`）才 surface。表现为 collective 出 async error，真凶是 GPU 硬件。`nvidia-smi -q | grep ECC` 查错误计数，`dmesg` 看内核报的 XID。这类错误**不可恢复**，需隔离 GPU 或重启节点。部分 ECC 错误也可能表现为 hang（kernel 卡住），故与 [[NCCL hang与timeout]] §8.3 交集。

### 8.4 容错训练与 elastic

开了 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 后，任何 rank 出错都会 abort 整个 communicator，所有 rank crash——这是"早暴露"的代价。若要单 rank 错不影响其他（容错训练），须**框架层** elastic/restart（如 `torchrun --rdzv`、Ray train 的 fault tolerance），不是 NCCL 层。NCCL 层只负责"早暴露"，恢复交给上层。见 [[20-可靠性与开源工程]] 的 elastic 训练。

### 8.5 内容来源

整理自 PyTorch distributed 文档（`TORCH_NCCL_BLOCKING_WAIT`/`TORCH_NCCL_ASYNC_ERROR_HANDLING`/`TORCH_NCCL_TRACE_BUFFER_SIZE` FlightRecorder，已联网核实 2.13 文档）、NCCL 官方文档（`NCCL_IB_RETURN_ASYNC_EVENTS` 默认 1 自 2.23，已联网核实 2.30 文档）、[[NCCL核心机制]]/[[NCCL backend]]/[[NCCL通信拓扑]]/[[InfiniBand]]/[[NCCL hang与timeout]]/[[actor deadlock]] 笔记。`NCCL_BLOCKING_WAIT`/`NCCL_ASYNC_ERROR_HANDLING` 为旧名（`NCCL_` 前缀），PyTorch 2.x 推荐 `TORCH_NCCL_` 前缀新名，旧名兼容但优先新名。

---
相关: [[NCCL故障定位]] | [[NCCL核心机制]] | [[NCCL backend]] | [[NCCL通信拓扑]] | [[NCCL hang与timeout]] | [[InfiniBand]] | [[RDMA与GPUDirect]] | [[PCIe]] | [[NVLink与NVSwitch]] | [[all-reduce]] | [[actor deadlock]] | [[集群拓扑感知placement]] | [[18-集群网络存储与数据系统]]
