# NCCL hang 与 timeout

> **所属章节**: [[NCCL故障定位]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: NCCL 卡住 / NCCL 挂起 / NCCL 死等 / collective hang / NCCL timeout / watchdog timeout / 集合通信死锁
> **难度**: 中高（需懂 [[NCCL核心机制]] / [[NCCL backend]] / [[NCCL通信拓扑]] / [[InfiniBand]] / [[RDMA与GPUDirect]] / [[PCIe]] / [[actor deadlock]]）


## 1. 一句话定义

**NCCL hang 与 timeout** 是 GPU 集合通信最常见的**"卡住不动"类故障**的总称——**hang** 指某个集合通信（all-reduce/all-gather/send-recv）**有 rank 没有参与**（crash/未到调用点/网络断了），导致**已到达的 rank 在 NCCL 的 wait 里无限阻塞**，进程不报错也不前进；**timeout** 指这个阻塞**超过阈值后被打破**（PyTorch NCCL 后端默认 **10 分钟**超时，由 `init_process_group(timeout=...)` 控制；超时后 collective 被异步 abort、进程 crash），或 IB 层的 `NCCL_IB_TIMEOUT`（默认 20，约 4.3s/重试 ×7 ≈ 30s）先报网络错误。原因五花八门——**rank crash/OOM、collective 顺序不一致死锁、网络中断、IB 子网 SM 故障、PFC 暂停风暴、防火墙挡端口、GPU ECC 错误、某 rank 落后/挂起**。定位靠一套 **NCCL debug 环境变量**（`NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS`、`NCCL_DEBUG_FILE`、`NCCL_TOPO_DUMP_FILE`、`NCCL_P2P_DISABLE`、`NCCL_IB_DISABLE`、`NCCL_SHM_DISABLE`）+ 硬件体检（`nvidia-smi`/`ibstat`）+ 看哪个 rank 卡在哪步。本篇讲 **hang/timeout（无错误卡死型）**，与 [[NCCL异步错误定位]]（有错误但异步上报滞后型）互补——**hang 是"没报错就卡住"，async error 是"报错了但滞后/混乱"**，两者现象都是"卡住"但根因与定位路径不同。

> [!note] 三句话定位
> - **是什么**：集合通信某 rank 没参与，其他 rank 在 NCCL wait 里无限阻塞（hang）；阻塞超 10 分钟被 PyTorch watchdog abort（timeout），或 IB 层 30s 报网络错误。
> - **为什么**：集合通信要求所有 rank 到齐，一个没到全体卡死；原因从 rank crash/OOM 到网络/SM/PFC/防火墙/ECC 不一而足。
> - **与 [[NCCL异步错误定位]] 关系**：hang 是**无错误**卡住（rank 没到/没参与，NCCL 在 wait 里死等，直到 timeout）；async error 是**有错误**但 NCCL 默认异步执行，错误异步上报、滞后/混乱。本篇讲前者（hang/timeout），那篇讲后者（async error）。两者现象都是"卡住"，但根因（无错 vs 有错）与定位路径不同。


## 2. 为什么需要它（动机与背景）

### 2.1 集合通信是"全员到齐"语义

[[all-reduce]]/[[all-gather]] 等集合通信要求**所有参与 rank 都到达调用点**，否则无法完成（reduce 要凑齐所有输入）。这是隐式 barrier——一个 rank 没到，其他 rank 都在 NCCL 的 wait 里阻塞等待。这与点对点 send/recv 不同（send/recv 一方没到另一方可阻塞或报错，但不牵连全局）。

### 2.2 一个 rank 出问题，全体卡死

集合通信的"全员到齐"导致**单点故障全局化**：

- rank 5 因 OOM crash → 它不会调 all-reduce → 其他 rank 在 all-reduce 的 wait 里卡死；
- rank 3 的 collective 调用顺序与其他 rank 不一致（先 all-gather 再 all-reduce，其他 rank 反过来）→ 双方在各自的 wait 里死锁（见 [[actor deadlock]]）；
- rank 7 与 rank 2 之间的 IB 链路断了 → ring 上 rank 2 发给 rank 3 的数据永远不到 → 整个 ring 卡死；
- 某 rank 的 DataLoader 慢/卡 → 它还没到 all-reduce 调用点 → 全体等。

### 2.3 默认阻塞 + 长超时，故障被"放大"

PyTorch NCCL 后端默认 `wait()` 是**非阻塞**的（不阻塞 Python 线程，但 CUDA stream 阻塞），watchdog 在后台轮询。**默认超时 10 分钟**（NCCL 后端；其他后端如 gloo 默认 30 分钟）。意思是：一个 rank 挂了，其他 rank 要**傻等 10 分钟**才报 timeout crash——这期间集群资源全闲置、用户以为"训练在跑"实则全卡。这就是 hang 类故障"难发现、难定位"的根因：**现象是慢/卡，不是报错**。

### 2.4 IB 层的超时更早

跨机通信走 IB 时，IB RC 连接有**本地 ACK 超时**（`NCCL_IB_TIMEOUT`，默认 20 → 4.096μs × 2^20 ≈ 4.3s）× 重试次数（`NCCL_IB_RETRY_CNT` 默认 7）≈ **30 秒**后报 IB 网络错误（`ibv_poll_cq` error 12 等）。所以跨机 hang 通常 **30 秒左右报 IB 错误**，而不是等满 10 分钟。但若是机内 NVLink/PCIe hang 或 collective 顺序死锁（不涉及 IB RC 超时），就要等满 10 分钟 PyTorch timeout。


## 3. 核心概念详解

### 3.1 hang 的两种本质

**hang = 集合通信无法完成，rank 在 wait 里阻塞，无错误上报**。分两种：

1. **rank 没到调用点（控制流 hang）**：
   - rank 因 OOM/crash/异常退出 → 永远不调下一个 collective → 其他 rank 死等；
   - rank 卡在 collective 之前的代码（DataLoader 慢、CPU 计算慢、checkpoint 加载慢、某个 actor 死锁）→ 还没到 all-reduce → 其他 rank 等；
   - collective 顺序不一致（rank A 先 all-gather 后 all-reduce，rank B 反过来）→ 互相等对方先调 → 死锁。这是最常见的"逻辑死锁"，见 [[actor deadlock]]。

2. **rank 到了调用点但数据不到（数据流 hang）**：
   - 网络中断（IB 链路断、NIC 故障、SM 路径失效）→ ring 上某段数据永远不到 → 其他 rank 在 wait 里等数据；
   - PFC 暂停风暴（RoCE 无损以太网的 PFC 反压卡死）→ 数据流被流控暂停、不前进；
   - GPU ECC 错误 / XID 错误 → GPU 卡住，NCCL kernel 不推进；
   - 某段 NVLink/PCIe 故障 → 数据传输停滞。

**两种 hang 的区别**：控制流 hang 的 rank "根本没进 NCCL wait"；数据流 hang 的 rank "进了 wait 但等的数据不到"。定位手段不同（前者看 stack trace / 进程状态，后者看 NCCL log + 硬件）。

### 3.2 timeout 的层次

timeout 不是单一来源，有**三层**：

| 层 | 触发者 | 默认值 | 现象 | 控制方式 |
|---|---|---|---|---|
| **IB RC 超时** | IB 硬件/NIC | `NCCL_IB_TIMEOUT=20` → ~4.3s × 重试 7 ≈ **30s** | 跨机 hang 30s 后报 `ibv_poll_cq` error / `NCCL ECONNRESET` | `NCCL_IB_TIMEOUT`/`NCCL_IB_RETRY_CNT` |
| **PyTorch NCCL watchdog** | torch.distributed | **10 分钟**（NCCL 后端）/ 30 分钟（其他） | 全体卡 10min 后 collective 被异步 abort、进程 crash | `init_process_group(timeout=timedelta(...))` |
| **用户/框架层** | 训练循环 | 无默认（看实现） | 训练 step 超过某阈值告警/重启 | 框架 checkpoint/restart 逻辑 |

> [!warning] 误区：以为"NCCL timeout"是个环境变量
> **没有 `NCCL_TIMEOUT` 这个环境变量**。PyTorch NCCL 后端的超时是 `init_process_group(timeout=datetime.timedelta(minutes=N))` 的 Python 参数，默认 10 分钟（NCCL）/ 30 分钟（gloo）。NCCL 自己的环境变量里管超时的是 `NCCL_IB_TIMEOUT`（IB RC ACK 超时，默认 20）和 `NCCL_IB_RETRY_CNT`（重试次数，默认 7），这是网络层超时，不是集合通信整体超时。**别把两者混为一谈**。

### 3.3 hang 的典型原因清单

| 类别 | 原因 | 现象 | 定位线索 |
|---|---|---|---|
| **rank 异常** | OOM / 段错误 / Python 异常退出 | 某 rank 进程不在了 | `nvidia-smi` 看哪卡空了、`ps`/slurm 看 rank 进程 |
| **逻辑死锁** | collective 顺序不一致 / 条件分支导致部分 rank 不调 | 全体卡、无错误 | 代码 review collective 调用顺序；`NCCL_DEBUG=INFO` 看谁没进 |
| **网络中断** | IB 链路断 / NIC 故障 / 线缆坏 | 跨机 hang，30s 报 IB 错 | `ibstat`/`ibping`/`ibnetdiscover` |
| **SM 故障** | IB 子网管理器挂 / 路径失效 | 跨机通信全断、新连接建不上 | `sminfo` 看主备 SM |
| **PFC 风暴** | RoCE 无损以太网 PFC 反压死锁 | RoCE 环境下卡、无 IB 错误 | 抓包看 PFC 帧、看交换机 PFC 统计 |
| **防火墙** | 端口被挡、IB RC 握手失败 | 建连阶段卡 | `NCCL_DEBUG=INFO` 看 bootstrap/connect 卡住 |
| **GPU ECC/XID** | GPU 硬件错误、NCCL kernel 不推进 | 某 GPU 卡、`nvidia-smi` 报 ECC | `nvidia-smi -q` 看 ECC/XID、`dmesg` |
| **rank 落后** | 某 rank CPU/IO 慢、还没到调用点 | 全体等慢者 | 各 rank 加时间戳日志、看谁最后到 |
| **GPUDirect 失败** | `nvidia-peermem` 没装、ACS 阻断 P2P | 跨机退到 CPU 中转、巨慢或卡 | `NCCL_DEBUG=INFO` 看有无 GDR、是否走 CPU |
| **进程/GPU 错配** | 一卡多进程、进程数≠GPU 数 | 拓扑利用率异常、建环错 | `NCCL_DEBUG` 看 rank-GPU 映射 |

### 3.4 debug 环境变量速查表

| 变量 | 作用 | 值 | 何时用 |
|---|---|---|---|
| `NCCL_DEBUG` | 日志级别 | `WARN`(默认)/`VERSION`/`INFO`/`TRACE`/`ABORT` | 排障必开 `INFO`；`TRACE` 极详细 |
| `NCCL_DEBUG_FILE` | 日志写文件（支持 `%h`主机 `%p`PID） | 路径 | 多机时每 rank 单独日志，便于比对 |
| `NCCL_DEBUG_SUBSYS` | 按子系统过滤日志 | `ALL`/`COLL`/`GRAPH`/`init`/`NET`... | hang 时 `COLL` 看集体调用、`GRAPH` 看拓扑建图 |
| `NCCL_TOPO_DUMP_FILE` | dump 探测到的拓扑 XML | 路径 | 拓扑探测不对时看 |
| `NCCL_P2P_DISABLE` | 禁 P2P（NVLink/PCIe 直连）退 SHM/NET | `1` | 怀疑 P2P 坏（ACS/PCIe 拓扑问题）时试 |
| `NCCL_SHM_DISABLE` | 禁共享内存传输，退 NET | `1` | 跨 NUMA SHM 有问题时试 |
| `NCCL_IB_DISABLE` | 禁 IB/RoCE，退 TCP socket | `1` | 怀疑 IB 问题时试（隔离用） |
| `NCCL_IB_TIMEOUT` | IB RC ACK 超时指数 | `20`(默认)/0-31 | 超大规模网络 hang 时增大（如 22/23） |
| `NCCL_IB_RETRY_CNT` | IB 重试次数 | `7`(默认)/0-7 | 配合 timeout 调网络层超时 |
| `NCCL_IB_RETURN_ASYNC_EVENTS` | IB 致命异步事件时停止通信 | `1`(默认,自2.23) | IB 故障上报，保持开 |
| `NCCL_SOCKET_IFNAME` | 指定跨机网卡 | `eth0`/`^docker` | 网卡选错导致 bootstrap 卡时设 |
| `NCCL_IB_HCA` | 指定 IB 网卡端口 | `=mlx5_0:1` | 多 NIC 时选对 rail |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA 级别 | `5`(PHY 直连最佳) | 跨机慢/卡时确认 GDR 级别 |
| `NCCL_BLOCKING_WAIT` | (PyTorch,见下) | — | 见 [[NCCL异步错误定位]] |

> [!note] PyTorch 层的两个变量（TORCH_ 前缀）
> `TORCH_NCCL_BLOCKING_WAIT=1`（阻塞等待，让 `wait()` 阻塞到完成或超时）与 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`（异步错误转异常 abort）是 **PyTorch 的环境变量**（不是 NCCL 自己的），控制的是 `torch.distributed` 的 ProcessGroupNCCL 行为。详见 [[NCCL异步错误定位]]。本篇的 hang/timeout 主要靠上表的 NCCL 变量 + PyTorch 的 `timeout` 参数。

### 3.5 定位流程（hang 怎么查）

**第一步：确认是 hang 还是 async error**
- 看进程状态：全体 `D` 状态（不可中断睡眠，卡在 wait）→ 多半 hang；
- 看 NCCL log：有无 error/warning → 有 = async error（转 [[NCCL异步错误定位]]）；无 = 纯 hang。

**第二步：找哪个 rank 卡在哪步**
- `NCCL_DEBUG=INFO NCCL_DEBUG_FILE=nccl.%h.%p` 每机一文件，比对所有 rank 的 log：
  - 哪个 rank 的 log 停在某个 collective 调用（`NCCL DEBUG: ... collective ...`）之后没下文 → 它卡在那；
  - 哪个 rank 根本没到该 collective（log 里没这条）→ 它卡在 collective 之前（控制流）。

**第三步：硬件体检**
```bash
nvidia-smi                       # 哪卡空了(rank 退出)/ECC 错误/XID
nvidia-smi -q | grep -A5 ECC     # GPU ECC 错误
ibstat                           # IB 端口 State=Active? 降速?
sminfo                           # 主备 SM 在?
ibnetdiscover | head             # 拓扑有无单点/断链
dmesg | tail -50                 # 内核有无 GPU/NIC 错误
```

**第四步：隔离试验**
- 怀疑 IB：`NCCL_IB_DISABLE=1` 退 TCP，看是否不卡（确认是 IB 问题）；
- 怀疑 P2P：`NCCL_P2P_DISABLE=1` 禁 P2P 退 SHM/NET；
- 怀疑 GPUDirect：`NCCL_NET_GDR_LEVEL=0` 关 GDR 看是否退到 CPU 中转；
- 怀疑拓扑：`NCCL_TOPO_DUMP_FILE=topo.xml` dump 出来比对 `nvidia-smi topo -m`。

**第五步：代码层（逻辑死锁）**
- 若硬件全对、NCCL log 显示 rank 没到调用点 → 查代码 collective 顺序是否所有 rank 一致；
- 条件分支（if rank==0 才调某 collective）会导致其他 rank 在等 → 死锁。NCCL 要求**所有 rank 按相同顺序调相同 collective**（见 [[NCCL核心机制]] §7 误区8）。

### 3.6 NCCL_DESC_ORDER / channel 重排（待核实）

任务提及"`NCCL_DESC_ORDER` 重排"——**该环境变量名待核实**，NCCL 官方文档环境变量列表中未见明确的 `NCCL_DESC_ORDER`。与之相关的、可核实的是：
- `NCCL_MAX_NCHANNELS` / `NCCL_MIN_NCHANNELS`：控制 channel（逻辑环）数，调多 ring 带宽；
- `NCCL_ALGO=Ring/Tree/Collnet`：强制算法；
- `NCCL_TOPO_FILE`：注入外部拓扑 XML 改变建图（间接改变 channel/rank 排布）。
若遇到"channel 顺序"类问题，实际调的是上述几个。**`NCCL_DESC_ORDER` 若存在于某 NCCL 版本，待核实其语义**。


## 4. 数学原理 / 公式

### 4.1 IB RC 超时计算

IB 本地 ACK 超时（Local Ack Timeout）= $4.096\,\mu s \times 2^{\text{timeout}}$。总等待时间 = timeout × retry_cnt：

$$
T_{\text{IB,timeout}} = \text{NCCL\_IB\_RETRY\_CNT} \times 4.096\,\mu s \times 2^{\text{NCCL\_IB\_TIMEOUT}}
$$

默认 `NCCL_IB_TIMEOUT=20`、`NCCL_IB_RETRY_CNT=7`：

$$
T \approx 7 \times 4.096\mu s \times 2^{20} = 7 \times 4.3\text{s} \approx 30\text{s}
$$

故跨机 hang 约 **30 秒**报 IB 错误。超大规模网络（延迟大）可调 `NCCL_IB_TIMEOUT=22/23`（34s/68s 单次）避免误报。

### 4.2 PyTorch NCCL 超时

PyTorch NCCL 后端 watchdog 默认 **10 分钟**（`init_process_group` 的 `timeout` 默认 NCCL=10min、其他=30min）。超时后：

$$
T_{\text{torch,timeout}} = \text{timeout 参数} \quad (\text{默认 } 600\text{s, NCCL})
$$

collective 被异步 abort、进程 crash（因 CUDA 异步，继续执行可能在损坏数据上）。设 `TORCH_NCCL_BLOCKING_WAIT=1` 时 `wait()` 阻塞到超时；否则非阻塞但 watchdog 仍在后台计时。

### 4.3 hang 的"代价"建模

设单 step 训练时间 $T_{\text{step}}$（正常），hang 后到 timeout 的闲置时间 $T_{\text{timeout}}$（10min）。一次 hang 事件的 GPU 小时损失：

$$
\text{cost} \approx \frac{T_{\text{timeout}}}{T_{\text{step}}} \times \text{单 step 算力} \times N_{\text{GPU}}
$$

万卡集群一次 10 分钟 hang ≈ 万卡 × 10min 闲置 ≈ 1600 GPU·小时浪费。故生产必开**快速 timeout + watchdog + checkpoint restart**，别傻等 10 分钟。


## 5. 代码示例

### 5.1 模拟 hang 并定位

```python
# hang_repro.py —— 故意让 rank 0 不参与 all-reduce, 制造 hang
# torchrun --nproc_per_node=4 hang_repro.py
import os, torch
import torch.distributed as dist

os.environ["NCCL_DEBUG"] = "INFO"                          # 必开
os.environ["NCCL_DEBUG_FILE"] = "nccl.%h.%p"               # 每机一文件, 便于比对
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")  # 异步错误转 abort, 别傻等

dist.init_process_group("nccl", init_method="env://",
                        timeout=__import__("datetime").timedelta(seconds=30))  # 调短 timeout 加速复现
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
rank = dist.get_rank()

t = torch.ones(1_000_000, device="cuda") * rank
if rank != 0:                          # rank 0 故意跳过 -> 其他 rank 在 all_reduce wait 里卡
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
# rank 0 退出, rank 1/2/3 卡 ~30s 后 timeout crash
print(f"[rank {rank}] done")
dist.destroy_process_group()
```

```bash
# 运行后看各 rank 的 nccl.<host>.<pid> 日志:
# rank 1/2/3 的日志停在 "NCCL INFO ... allreduce" 之后无下文 -> 它们卡在 wait
# rank 0 的日志没有 allreduce 这条 -> 它根本没到调用点 (控制流 hang)
# timeout 30s 后: "Watchdog caught collective operation timeout" -> 确认 PyTorch timeout
```

### 5.2 排障启动模板

```bash
#!/bin/bash
# nccl_debug_launch.sh —— 排障时给训练加的 NCCL debug 套件
export NCCL_DEBUG=INFO                       # 主日志
export NCCL_DEBUG_FILE=nccl_logs/nccl.%h.%p  # 每机每进程一文件
export NCCL_DEBUG_SUBSYS=COLL,GRAPH,init     # 关心集合调用 + 拓扑建图 + 初始化
export NCCL_TOPO_DUMP_FILE=nccl_topo.xml     # dump 拓扑
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1     # 异步错误转 abort, 不傻等 10min
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000    # 通信 trace buffer (排 hang)

# 可选隔离试验 (怀疑时逐个试):
# export NCCL_IB_DISABLE=1          # 怀疑 IB: 退 TCP 看是否不卡
# export NCCL_P2P_DISABLE=1         # 怀疑 P2P: 退 SHM/NET
# export NCCL_SHM_DISABLE=1          # 怀疑跨 NUMA SHM
# export NCCL_IB_TIMEOUT=22         # 大网络 hang 误报: 增大 IB 超时

torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE train.py
```

### 5.3 硬件体检一键脚本

```bash
#!/bin/bash
# hw_health.sh —— hang 时硬件体检
echo "===== GPU ====="
nvidia-smi --query-gpu=index,name,utilization.gpu,ecc.errors.uncorrected.volatile.total --format=csv
nvidia-smi -q | grep -A3 "ECC Errors" | head -20
echo "===== GPU 拓扑 ====="
nvidia-smi topo -m | head -12
echo "===== IB ====="
ibstat 2>/dev/null | grep -E "mlx|State|Rate" || echo "no IB"
sminfo 2>/dev/null | head -5
echo "===== 内核日志 (最近 GPU/NIC 错误) ====="
dmesg --time-format iso 2>/dev/null | grep -iE "nvidia|mellanox|mlx|ib|ecc|xid" | tail -20
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[NCCL核心机制]]（集合通信"全员到齐"语义、async_op/wait）、[[NCCL backend]]（PyTorch NCCL 后端、`timeout` 参数、环境变量）、[[NCCL通信拓扑]]（channel/transport、拓扑探测）、[[InfiniBand]]（IB RC 超时、SM、PFC）、[[RDMA与GPUDirect]]（GPUDirect 失败退 CPU）、[[PCIe]]（ACS 阻断 P2P）。
- **下游（应用）**: [[NCCL异步错误定位]]（hang 的"有错版"互补）、[[actor deadlock]]（RL 框架的 collective 顺序死锁）、[[集群拓扑感知placement]]（拓扑错导致 hang）、[[3D parallelism]]（多 communicator 顺序问题）、checkpoint/restart 工程（timeout 后恢复）。
- **对比 / 易混**:
  - **hang vs timeout**：hang 是"卡住"现象，timeout 是"超时打破"机制。hang 必然导致 timeout（若没先被 IB 错误打破）。先 hang 后 timeout。
  - **hang（本篇）vs async error（[[NCCL异步错误定位]]）**：hang 是**无错误**卡住（rank 没到/数据没到，NCCL 在 wait 死等）；async error 是**有错误**但异步上报滞后（某 rank 报 ECONNRESET/INTERNAL，其他 rank 还在 wait，错误信息滞后混乱）。现象都是"卡住"，但定位手段不同（hang 看谁没到调用点/硬件；async error 看全部 rank log 拼因果）。
  - **PyTorch timeout vs NCCL_IB_TIMEOUT**：前者是集合通信整体超时（10min，torch 层），后者是 IB RC ACK 超时（30s，网络层）。跨机 hang 通常后者先报。
  - **hang vs [[actor deadlock]]**：actor deadlock 是 Ray actor 间死锁（互相 await 对方消息），NCCL hang 是集合通信死锁（rank 没到调用点）。RL 系统两者交织——actor 死锁会导致 rank 不调 collective，进而 NCCL hang。


## 7. 常见误区与易错点

> [!warning] 误区1：以为有 `NCCL_TIMEOUT` 环境变量
> 没有。PyTorch NCCL 超时是 `init_process_group(timeout=...)` 参数（默认 10min NCCL / 30min gloo）。NCCL 自己管的是 `NCCL_IB_TIMEOUT`（IB RC，默认 20）。别把 PyTorch timeout 和 IB timeout 混为一谈。

> [!warning] 误区2：以为 hang 会很快报错
> 不会。纯 hang（机内/逻辑死锁）要等满 10 分钟 PyTorch timeout。跨机 hang 可能 30s 报 IB 错（先于 timeout）。故生产**必开 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` + 调短 timeout + watchdog 监控**，别傻等。

> [!warning] 误区3：把 async error 当 hang 处理
> 现象都是"卡住"，但 async error 是有错误（IB ECONNRESET/INTERNAL）但异步上报滞后，定位要拼全部 rank log 找因果；hang 是无错误卡死，定位找"谁没到调用点"。先看 NCCL log 有无 error 区分。见 [[NCCL异步错误定位]]。

> [!warning] 误区4：条件分支里调 collective
> `if rank==0: dist.all_reduce(...)` 会让其他 rank 永远等 → 死锁。NCCL 要求所有 rank 按相同顺序调相同 collective（见 [[NCCL核心机制]] §7 误区8）。

> [!warning] 误区5：忘设 `NCCL_SOCKET_IFNAME`
> 多机不设 → NCCL 选到 docker0/lo，bootstrap 卡在连接阶段，表现为 hang（建不上 communicator）。多机必设 `NCCL_SOCKET_IFNAME=eth0`（或对应网卡）。

> [!tip] 实践：生产必做防 hang 配置
> 1. `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`（异步错误转 abort，不傻等）；
> 2. `init_process_group(timeout=timedelta(minutes=5))` 调短（别 10min）；
> 3. `NCCL_DEBUG=WARN NCCL_DEBUG_FILE=...` 持久日志；
> 4. 外部 watchdog（监控 GPU 利用率，全员 0 利用超 N 秒告警重启）；
> 5. 定期 checkpoint，hang 重启可恢复。

> [!tip] 实践：定位 hang 的"三看"
> 1. **看进程**：`ps`/`nvidia-smi` 哪个 rank/GPU 不在了（crash）或全 `D` 状态（wait）；
> 2. **看 log**：`NCCL_DEBUG=INFO` 各 rank log，谁停在 collective 后无下文（卡 wait）、谁没到 collective（卡控制流）；
> 3. **看硬件**：`ibstat`/`sminfo`/`nvidia-smi -q ECC`/`dmesg` 排除网络/GPU/SM 故障。

> [!note] 补充：RL 系统的 hang 更隐蔽
> RLHF/Agent 系统里，rollout actor（推理）与 learner（训练）解耦，rollout 卡住会导致 learner 的权重同步/样本收集 collective 缺 rank → NCCL hang。这种 hang 根因在 rollout 侧（推理 OOM/环境卡/actor 死锁），不在 NCCL。见 [[actor deadlock]]。


## 8. 延伸细节

### 8.1 PFC 暂停风暴（RoCE 特有）

RoCE 无损以太网靠 PFC（Priority Flow Control）防丢包，但 PFC 反压过度会形成"暂停风暴"——某端口被暂停，上游级联暂停，整 fabric 数据流停滞，表现为 NCCL hang（无错误、无 IB timeout，因为 RoCE 走以太网不经 IB RC 超时）。抓包看 PFC 帧、看交换机 PFC 统计，调 ECN/PFC 阈值缓解。见 [[RoCE]] §3。

### 8.2 IB SM 主备切换瞬间的 hang

IB 子网管理器（SM）主备切换时，几秒内新路径未收敛，跨机通信短暂 hang/报错。生产必配主备 SM（`sminfo` ≥2 条），但切换瞬间的秒级感知仍可能让 NCCL 报错重连。见 [[InfiniBand]] §3.3 / §7 误区2。

### 8.3 GPU ECC / XID 错误

GPU 显存 ECC 不可纠正错误或 XID（GPU 错误码）会让 GPU 卡住，NCCL kernel 不推进，表现为 hang。`nvidia-smi -q | grep ECC` 查错误计数，`dmesg` 看内核报的 XID。硬件故障需隔离该 GPU 或重启节点。见 [[NCCL异步错误定位]] §3（部分 ECC 错误是 async error）。

### 8.4 NCCL Flight Recorder

PyTorch 2.x 起，`TORCH_NCCL_TRACE_BUFFER_SIZE` 开启 NCCL 通信的 flight recorder——环形缓冲记录最近的 collective 调用，hang/crash 时 dump 出来，看最后卡在哪条 collective。是排 hang 的新利器（类似飞机黑匣子）。见 PyTorch distributed 文档的 `FlightRecorderHook`。

### 8.5 内容来源

整理自 NCCL 官方文档（`NCCL_DEBUG`/`NCCL_DEBUG_FILE`/`NCCL_DEBUG_SUBSYS`/`NCCL_TOPO_DUMP_FILE`/`NCCL_P2P_DISABLE`/`NCCL_SHM_DISABLE`/`NCCL_IB_DISABLE`/`NCCL_IB_TIMEOUT`/`NCCL_IB_RETRY_CNT`/`NCCL_IB_RETURN_ASYNC_EVENTS`/`NCCL_TOPO_FILE`，已联网核实 2.30 文档）、PyTorch distributed 文档（`init_process_group` timeout 默认 10min NCCL/30min 其他，`TORCH_NCCL_BLOCKING_WAIT`/`TORCH_NCCL_ASYNC_ERROR_HANDLING`/`TORCH_NCCL_TRACE_BUFFER_SIZE` FlightRecorder，已联网核实 2.13 文档）、[[NCCL核心机制]]/[[NCCL backend]]/[[InfiniBand]]/[[actor deadlock]] 笔记。`NCCL_DESC_ORDER` 环境变量未在官方文档找到，语义待核实；channel/算法调整用 `NCCL_MAX/MIN_NCHANNELS`/`NCCL_ALGO` 替代。

---
相关: [[NCCL故障定位]] | [[NCCL核心机制]] | [[NCCL backend]] | [[NCCL通信拓扑]] | [[NCCL异步错误定位]] | [[InfiniBand]] | [[RoCE]] | [[RDMA与GPUDirect]] | [[PCIe]] | [[NVLink与NVSwitch]] | [[actor deadlock]] | [[集群拓扑感知placement]] | [[18-集群网络存储与数据系统]]
