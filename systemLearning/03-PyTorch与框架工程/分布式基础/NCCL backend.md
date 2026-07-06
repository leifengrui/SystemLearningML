# NCCL backend

> **所属章节**: [[分布式基础]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 中等偏难（性能调优必备）

## 1. 一句话定义

**NCCL（NVIDIA Collective Communication Library）** 是 NVIDIA 提供的**GPU 集合通信库**，针对 NV GPU 间拓扑（NVLink / NVSwitch / PCIe / InfiniBand + GPUDirect RDMA）做了深度优化的 all-reduce / all-gather / reduce-scatter / broadcast 等原语实现。它是 `torch.distributed` 在 GPU 训练下的**事实标准后端**（`backend="nccl"`），也是 [[DDP]]/[[FSDP]]/[[Tensor Parallel]] 通信性能的地基。

> [!note] NCCL 的地位
> 几乎所有 NV GPU 多卡训练都走 NCCL：PyTorch `nccl` 后端、Megatron-LM、DeepSpeed、vLLM 推理的多卡 all-reduce，底层都是 NCCL。非 NV 平台（AMD ROCm）用对应库 RCCL；Intel GPU 用 oneCCL。CPU 训练用 gloo。

## 2. 为什么需要它（动机与背景）

GPU 多卡训练需要频繁 all-reduce 梯度（每 step 一次，百亿参数模型梯度几十 GB）。朴素实现（rank 0 收集所有再广播）在 8 卡上要传 $(N-1)M\approx 7M$ 且 rank 0 是瓶颈，不可扩展。

NCCL 的核心价值：

1. **ring all-reduce** 算法：把通信量降到 $\approx 2M$（与 N 无关），且无单点瓶颈；
2. **拓扑感知**：自动探测 NVLink/PCIe/IB，建最优 ring/tree，能用 NVLink（带宽数百 GB/s）就绝不用 PCIe（数十 GB/s）；
3. **GPUDirect RDMA**：跨机 GPU 直接到 IB 网卡，不绕 CPU 内存，跨机 all-reduce 接近线速；
4. **CUDA kernel 融合**：通信本身是 GPU kernel，可与计算 stream 异步 overlap；
5. **算子丰富**：all-reduce/all-gather/reduce-scatter/broadcast/all-to-all 全套，且支持 async。

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
