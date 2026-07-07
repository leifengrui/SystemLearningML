# NCCL backend

> **所属章节**: [[分布式基础]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 中等偏难（性能调优必备）

## 1. 一句话定义

**NCCL（NVIDIA Collective Communication Library）** 是 NVIDIA 提供的**GPU 集合通信库**，针对 NV GPU 间拓扑（NVLink / NVSwitch / PCIe / InfiniBand + GPUDirect RDMA）做了深度优化的 all-reduce / all-gather / reduce-scatter / broadcast 等原语实现。它是 `torch.distributed` 在 GPU 训练下的**事实标准后端**（`backend="nccl"`），也是 [[DDP]]/[[FSDP]]/[[Tensor Parallel]] 通信性能的地基。

> [!note] NCCL 的地位
> 几乎所有 NV GPU 多卡训练都走 NCCL：PyTorch `nccl` 后端、Megatron-LM、DeepSpeed、vLLM 推理的多卡 all-reduce，底层都是 NCCL。非 NV 平台（AMD ROCm）用对应库 RCCL；Intel GPU 用 oneCCL。CPU 训练用 gloo。

> [!note] 解答：gloo 是什么？为什么会有 CPU 训练？
> 
> **gloo** 是 Facebook（Meta）开源的一个**进程间集合通信库**（[github.com/facebookincubator/gloo](https://github.com/facebookincubator/gloo)），用 C++ 写、提供 Python 绑定，能在**纯 CPU 环境**（机器有若干 CPU 核心、可能跨机走 TCP）上做 all-reduce / all-gather / broadcast 等集合通信原语。PyTorch 把它集成成 `torch.distributed` 的一个后端：`dist.init_process_group(backend="gloo")`。它和 [[NCCL backend|NCCL]] 是**并列的两个后端**——NCCL 管 GPU 集合通信，gloo 管 CPU 集合通信（也能传 GPU 张量但慢，不推荐）。
> 
> ### 为什么"第一次听说 CPU 训练"很正常
> 
> 因为 **LLM / 深度学习时代默认全是 GPU 训练**，CPU 训练只出现在一些边缘场景，所以教程里很少提 gloo。CPU 训练的真实场景：
> 
> 1. **CPU 推理 / 小模型调试**：几百万参数的 MLP、传统 ML 模型、CPU 上跑得通的 GNN/推荐小塔，不需要 GPU；
> 2. **模型太小 GPU 不划算**：比如 embedding 训练的前期原型、轻量 finetune，CPU 多机反而比单 GPU 便宜；
> 3. **没有 GPU 的集群**：纯 CPU 服务器做参数服务器（PS）式训练、强化学习里的非神经部分（如环境 rollout 的部分计算）；
> 4. **CPU 上做分布式逻辑的单元测试**：写 DDP 代码时不想占 GPU，用 gloo 在 CPU 上验证流程对不对（数据切分、rank 同步、checkpoint），跑通了再换 nccl 上 GPU。这是工程里 gloo 最常见的用法——**"CPU 上调分布式逻辑，GPU 上调性能"**。
> 
> 所以 gloo 不是"落后"，而是**给没有 GPU 或不想用 GPU 的场景兜底**。
> 
> ### gloo vs NCCL 对比
> 
> | 维度 | **gloo** | **NCCL** |
> |---|---|---|
> | 厂商 | Meta 开源 | NVIDIA |
> | 加速硬件 | **CPU**（多核 + TCP/共享内存） | **GPU**（NVLink / NVSwitch / PCIe / IB+RDMA） |
> | 张量位置 | CPU 张量（`tensor.cpu()`） | GPU 张量（`tensor.cuda()`） |
> | 通信路径 | TCP socket / POSIX 共享内存 | NVLink / GPUDirect RDMA |
> | 带宽量级 | 数 GB/s（受网卡/内存带宽限制） | 数百 GB/s（NVLink） |
> | 是否 GPU kernel | 否（CPU 线程算 reduce） | 是（CUDA kernel，可与计算 overlap） |
> | 典型用途 | CPU 训练、分布式逻辑单元测试、CPU 推理编排 | 一切 GPU 多卡训练（DDP/FSDP/TP） |
> | 跨机 | TCP（慢） | IB+RDMA（接近线速） |
> 
> 同样是 all-reduce 梯度：NCCL 在 8 卡 NVLink 机器上几 GB 梯度只要几百微秒；gloo 走 TCP 跨机要慢一两个量级。所以**有 GPU 就用 nccl，没 GPU 才用 gloo**，这是铁律。
> 
> ### 代码：gloo 后端长什么样
> 
> ```python
> import torch
> import torch.distributed as dist
> import os
> 
> # torchrun --nproc_per_node=4 demo_gloo.py
> dist.init_process_group(backend="gloo")   # 注意这里是 gloo，不是 nccl
> rank = dist.get_rank()
> 
> # 注意：CPU 张量！不调 .cuda()
> t = torch.ones(8) * rank                   # 各 rank 不同值
> dist.all_reduce(t, op=dist.ReduceOp.SUM)   # 各 rank 求和
> print(f"[rank {rank}] result={t.tolist()}")  # 应为 [6,6,...,6] (0+1+2+3)
> dist.barrier()
> ```
> 
> 把 `gloo` 改成 `nccl`、把 `torch.ones(8)` 改成 `torch.ones(8).cuda()`，就是 GPU 版。**backend 必须和张量所在设备匹配**——gloo 后端传 GPU 张量会报错或极慢，nccl 后端传 CPU 张量直接报错。
> 
> ### 一个常被忽略的点：gloo 的 GPU 支持
> 
> gloo 其实也能传 GPU 张量（旧版曾支持），但 reduce 计算仍走 CPU（要把数据拷到 CPU 算再拷回 GPU），比 NCCL 慢得多，**没有任何理由在 GPU 场景用 gloo**。所以记住：**GPU 张量 → nccl；CPU 张量 → gloo**，一一对应，不要混。
> 
> ### 误区清单
> 
> 1. **"gloo 是 NCCL 的替代"** → 不是平替。gloo 是 CPU 后端，NCCL 是 GPU 后端，**按设备选**。
> 2. **CPU 训练 = 落后** → 取决于模型规模。小模型 / 调试 / 无 GPU 集群，CPU+gloo 是合理选择；LLM 这种百亿参数级别 CPU 根本跑不动，与 gloo 无关，是算力问题。
> 3. **在 GPU 机器上还用 gloo** → 错误配置，慢且不发挥 NVLink；GPU 机器一律 `nccl`。
> 4. **写 DDP 时忘了 backend** → 默认在 GPU 机器 PyTorch 会提示用 nccl；若你显式传 `gloo` 又放 GPU 张量，会踩坑。`init_process_group(backend="nccl")` 是 GPU 训练标配。
> 5. **以为 gloo 不能跨机** → 能，走 TCP；只是带宽小、延迟高，不适合大模型梯度同步。
> 
> ### 关联
> - gloo 是 [[torch.distributed]] 的后端之一，与 [[NCCL backend]] 并列；选哪个看张量在 CPU 还是 GPU。
> - [[DDP]]/[[FSDP]] 在 GPU 训练时强制用 nccl；若你硬要用 gloo 跑 DDP（CPU 模型 / 单元测试），可以，但别指望性能。
> - 强化学习系统里 rollout 环境（非神经网络部分）有时用 gloo 做 CPU 侧编排，神经网络部分仍用 nccl，见 [[Ray与分布式调度]] 的多 actor 集成。
## 2. 为什么需要它（动机与背景）

GPU 多卡训练需要频繁 all-reduce 梯度（每 step 一次，百亿参数模型梯度几十 GB）。朴素实现（rank 0 收集所有再广播）在 8 卡上要传 $(N-1)M\approx 7M$ 且 rank 0 是瓶颈，不可扩展。

NCCL 的核心价值：

1. **ring all-reduce** 算法：把通信量降到 $\approx 2M$（与 N 无关），且无单点瓶颈；
2. **拓扑感知**：自动探测 NVLink/PCIe/IB，建最优 ring/tree，能用 NVLink（带宽数百 GB/s）就绝不用 PCIe（数十 GB/s）；
3. **GPUDirect RDMA**：跨机 GPU 直接到 IB 网卡，不绕 CPU 内存，跨机 all-reduce 接近线速；
4. **CUDA kernel 融合**：通信本身是 GPU kernel，可与计算 stream 异步 overlap；
5. **算子丰富**：all-reduce/all-gather/reduce-scatter/broadcast/all-to-all 全套，且支持 async。
【user】这几点可以展开讲一讲 甚至新建文件
没有 NCCL，多卡 LLM 训练要么慢一个量级，要么跨机根本跑不动。

## 3. 核心概念详解

### 3.1 ring all-reduce（核心算法）

N 个 rank 排成逻辑环，张量分成 N 段。两阶段：
- **reduce-scatter 阶段**（N-1 步）：每步每 rank 把一段发送给下游并累加，N-1 步后每 rank 持有"一段的完整和"。
- **all-gather 阶段**（N-1 步）：把完整段沿环传播，N-1 步后每 rank 持有完整结果。

每 rank 通信量 $\frac{2(N-1)}{N}M\approx 2M$，总通信量 $\approx 2M$（与 N 无关）。对比朴素树形 $(N-1)M$ 且 root 瓶颈，ring 适合大 N。

### 3.2 拓扑与 channel

NCCL 启动时探测每个 GPU 对之间的连接类型（NVLink/PCIe/IB），建**channel**：每 channel 是一对发送/接收的 GPU 对，多个 channel 并行用满带宽。NVSwitch 全连接拓扑下 channel 数多、带宽高；普通 PCIe 拓扑下受限。

### 3.3 单机 vs 跨机

- **单机 NVLink**：8 卡通过 NVLink/NVSwitch 全连接，all-reduce 带宽数百 GB/s，延迟低。
- **跨机 InfiniBand**：通过 GPUDirect RDMA，GPU 直接 DMA 到网卡，不经 CPU 内存；带宽受 IB（200/400 Gbps）限制，是跨机瓶颈所在。
- **跨机无 IB（以太网）**：NCCL 也能走 TCP socket（`NCCL_SOCKET_IFNAME`），但带宽低、延迟高，不推荐大模型。

### 3.4 communicator 与 stream

每个 [[process group]] 在 NCCL 下对应一个 **communicator**，绑定到一个 CUDA stream。通信默认在**默认 stream** 上同步执行，`async_op=True` 时放到独立 stream 可与计算 overlap。NCCL 通信本身是 GPU kernel，在 stream 上调度。

### 3.5 环境变量调优

| 变量 | 作用 |
|---|---|
| `NCCL_DEBUG=INFO` | 打印拓扑探测/建环过程，排障第一手段 |
| `NCCL_DEBUG_SUBSYS=ALL` | 更细子模块日志 |
| `NCCL_SOCKET_IFNAME=eth0` | 指定跨机通信网卡（否则可能选错） |
| `NCCL_IB_HCA=mlx5_0` | 指定 IB 网卡端口 |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA 级别（跨机性能） |
| `NCCL_P2P_DISABLE=1` | 禁用 P2P（某些 PCIe 拓扑下 P2P 坏） |
| `NCCL_IB_DISABLE=1` | 禁用 IB，退化到 TCP |
| `NCCL_BUFFSIZE` / `NCCL_NCHANNELS` | 调通信 buffer/通道数 |
| `NCCL_ALGO=Ring`/`Tree`/`Collnet` | 强制算法 |
| `NCCL_NET_FLEX_LEVEL` | 动态选网卡 |

### 3.6 与 PyTorch 的接口

`dist.init_process_group(backend="nccl")` → 后续 `dist.all_reduce` 等走 NCCL。`torch.distributed` 把 NCCL 的 communicator 抽象成 `ProcessGroupNCCL`。`async_op=True` 返回 `Work` 句柄，`.wait()` 同步、`.is_done()` 查询。

### 3.7 NCCL 版本与 GPU 架构

NCCL 跟随 CUDA 版本演进，新版针对新架构（Hopper/Blackwell）优化、支持新原语（如 `ncclAllGather` 张量分组、FP8 reduce）。PyTorch 自带 NCCL，但可 `pip install --upgrade nvidia-nccl-cu12` 替换新版。

## 4. 数学原理 / 公式

### ring all-reduce 通信量

设 N 个 rank、张量 $M$ 字节、链路单向带宽 $B$。ring 两阶段各 N-1 步，每步每 rank 发送 $M/N$ 字节。故：

$$
\text{每 rank 通信量}=\frac{2(N-1)}{N}M\approx 2M\ (N\gg1),\qquad
\text{时延}\approx \frac{2(N-1)}{N}\cdot\frac{M}{B}
$$

对比朴素 all-reduce（root 收集再广播）每 rank $(N-1)M$、root 瓶颈。ring 把量降到与 N 无关的 $2M$，且每 rank 同时收发、无瓶颈。

### 树形与 ring 的取舍

- **ring**：带宽利用率高，但 N 很大时延迟随 N 增长（N-1 步串行）。
- **tree**（双二叉树）：延迟 $\log N$，适合小消息或超多 rank。
- NCCL 自动选算法，大消息用 ring、小消息用 tree/protocol。

### all-gather / reduce-scatter 通信量

- all-gather：每 rank 发 $M/N$、收 $M$，总 $M$ per rank。
- reduce-scatter：每 rank 发 $M$、收 $M/N$，总 $M$ per rank。

FSDP 把参数 all-gather（$M$）+ 梯度 reduce-scatter（$M$）= $2M$ per step，与 DDP 的 all-reduce $2M$ 相当，但分片后峰值显存大降。

## 5. 代码示例

```python
import os, torch
import torch.distributed as dist

# 调试时打开
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")  # 跨机指定网卡

dist.init_process_group("nccl", init_method="env://")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
rank, world = dist.get_rank(), dist.get_world_size()

t = torch.randn(1024, 1024, device="cuda")

# 1) 同步 all-reduce
dist.all_reduce(t, op=dist.ReduceOp.SUM)

# 2) 异步 all-reduce（可与计算 overlap）
work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
# 这里可插计算 ...
work.wait()   # 等通信完成

# 3) 查 NCCL 版本
import torch
print("NCCL version:", torch.cuda.nccl.version())

dist.destroy_process_group()
```

> [!tip] 排障三板斧
> 1. `NCCL_DEBUG=INFO` 看建环日志，找 "P2P disabled"、"IB not found"。
> 2. `nccl-tests` 套件测裸带宽（`all_reduce_perf -b 1M -e 1G`），隔离是 NCCL 还是 PyTorch 问题。
> 3. `nvidia-smi nvlink`/`ibstat` 看硬件链路状态。

## 6. 与其他知识点的关系

- **上游（依赖）**: CUDA（NCCL 是 CUDA 库）、GPU 拓扑（NVLink/IB）、`torch.distributed`（封装层）。
- **下游（应用）**: [[DDP]]（梯度 all-reduce）、[[FSDP]]（参数 all-gather + 梯度 reduce-scatter）、[[Tensor Parallel]]（TP 组内 all-reduce）、[[Pipeline Parallel]]（点对点 send/recv，NCCL 也支持）、[[3D parallelism]]（多 group 多 communicator）、[[overlap strategy]]（async_op + 计算 stream 重叠）、[[NCCL通信拓扑]]。
- **对比 / 易混**:
  - **NCCL vs gloo**：NCCL 是 GPU 专用、性能极高；gloo 是 CPU/通用、性能低但功能全（CPU 训练、控制流）。
  - **NCCL vs MPI**：MPI 是 HPC 通用通信库，NCCL 专注 GPU collective；PyTorch GPU 训练用 NCCL，少数 HPC 用 MPI。
  - **NCCL vs RCCL/oneCCL**：跨厂商对应物（ROCm/Intel GPU）。
  - **ring vs tree**：NCCL 内部算法选择，见 §4。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **跨机不设 `NCCL_SOCKET_IFNAME`** → NCCL 选错网卡（选到 docker0/lo），通信走慢路径甚至挂起。多机必设。
> 2. **以为 all-reduce 慢就是 NCCL 问题** → 可能是拓扑差（PCIe 无 NVLink）、IB 带宽不够、或梯度太小 latency bound。用 `nccl-tests` 量化。
> 3. **禁用 P2P/IB 后变慢却不知** → 某些环境默认禁了，性能掉一半；看 `NCCL_DEBUG` 日志确认 P2P/IB 启用。
> 4. **跨机不走 GPUDirect RDMA** → GPU 数据先经 CPU 内存再到网卡，带宽腰斩。需要 IB + `nvidia-peermem` 模块 + 合适 `NCCL_NET_GDR_LEVEL`。
> 5. **把 NCCL 通信与计算放同一 stream** → 无法 overlap，通信白等。要用独立 stream + `async_op`。
> 6. **NCCL 版本与 CUDA 不匹配** → 启动报错或性能退化；保持 NCCL 与 CUDA 版本对齐。
> 7. **进程数 ≠ GPU 数** → 一卡多进程或一进程多卡 collective，拓扑利用率下降，性能异常。
> 8. **死锁：collective 顺序不一致** → NCCL 要求所有 rank 按相同顺序调同一组 collective；顺序乱会死锁。

## 8. 延伸细节

### 8.1 ring all-reduce 的"分代"

经典 ring（2015 Baidu）后，NCCL 引入 **double tree**（双二叉树，适合小消息/超多 rank）、**collnet**（InfiniBand collective offload，硬件加速）、**nvls**（NVLink SHARP，NVSwitch 硬件 reduce）。NCCL 按消息大小和拓扑自动选最优算法。

### 8.2 NVLS / SHARP

Hopper + NVSwitch 支持 **NVLink SHARP**：reduce 在交换机硬件完成，不占 GPU，all-reduce 延迟与带宽双降。IB SHARP 类似。需 NCCL 新版 + `NCCL_NET_GDR_LEVEL`/`NCCL_ALGO` 配置。

### 8.3 通信-计算 overlap 的工程化

[[overlap strategy]] 把反向梯度 all-reduce 拆成"逐层 reduce-scatter + 反向 + all-gather"，让每层通信藏在下一层反向计算里。ZeRO-2/3、PyTorch 的 `torch.distributed.algorithms._schedule` 实现。NCCL 的 async stream 是前提。

### 8.4 NCCL 调优经验值

- 小消息（<1MB）latency bound，调 `NCCL_NTHREADS`/算法；
- 大消息 bandwidth bound，调 `NCCL_NCHANNELS`/`NCCL_BUFFSIZE`；
- 跨机加 `NCCL_NET_GDR_LEVEL=5`、`NCCL_IB_TIMEOUT`；
- 用 `nccl-tests` 先测裸性能，再比对训练实际，找 PyTorch 层开销。

### 8.5 NCCL 在推理系统的角色

多卡推理（TP=2+）也用 NCCL 做层间 all-reduce；vLLM/Ray Serve 多卡部署时 NCCL communicator 建立成本要摊到长生命周期。[[KV cache]] 的多卡分片有时也靠 NCCL all-gather 拼装。见 [[推理系统]]。

---
相关: [[分布式基础]]、[[torch.distributed]]、[[process group]]、[[rank与world size]]、[[DDP]]、[[FSDP]]、[[Tensor Parallel]]、[[overlap strategy]]、[[NCCL通信拓扑]]
