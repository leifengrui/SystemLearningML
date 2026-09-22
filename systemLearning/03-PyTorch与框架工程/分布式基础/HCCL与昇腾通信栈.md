# HCCL与昇腾通信栈

> **所属章节**: [[分布式基础]]
> **所属模块**: [[03-PyTorch与框架工程]]


## 1. 一句话定义

**HCCL**（Huawei Collective Communication Library，华为集合通信库）是华为昇腾 NPU 上对标 NVIDIA NCCL 的高性能集合通信库；**UB**（Unified Bus / 灵衢）是华为从物理层到事务层自研的统一互联协议，用一套协议栈替代 PCIe + NVLink + RoCE 的多协议拼接方案。两者合称"昇腾通信栈"，是华为 AI 集群的通信底座。

## 2. 为什么需要它（动机与背景）

**问题一：协议割裂。** 传统 GPU 集群用 PCIe（CPU↔GPU）、NVLink（GPU↔GPU）、InfiniBand/RoCE（跨机）三套协议拼接，每层各自优化，但跨层协议转换引入不可控延迟。华为受制程限制，单芯片算力约为 H100 的 60–70%，需用更多芯片组网弥补，因此互联规模和延迟控制比 NVIDIA 更关键——多协议拼接的代价更大。

**问题二：master-slave 模型不可扩展。** 传统 PCIe 架构中 CPU 是主设备，GPU/NPU 是从设备，设备间横向通信须经过 CPU 中转，成为瓶颈。

**问题三：RDMA 的连接状态爆炸。** RoCE 的 Queue Pair（QP）状态按 $O(N \times M)$（本地应用数 × 远端主机数）增长，大规模集群（数千节点）下 QP 表溢出片上 SRAM，每次操作都需从 DRAM 重新加载上下文（~1μs 级惩罚）。

**UB 的解法**：一套协议覆盖从片内到集群的所有互联场景；所有组件（CPU、NPU、DPU、SSU、交换机）作为平等的 UBPU（UB Processing Unit），任何设备可直接 Load/Store 访问其他设备内存（URMA 语义），绕过 CPU/OS。连接状态按 $O(N+M)$ 增长，常驻片上 SRAM。

**HCCL 的解法**：作为 CANN 的核心组件，向上对接 PyTorch/MindSpore 框架，向下利用 HCCS/UB/RoCE/PCIe 链路提供 AllReduce/AllGather 等原语，使昇腾集群的分布式训练 API 与 GPU 侧 `torch.distributed` 几乎对齐。

## 3. 核心概念详解

### 3.1 HCCL 软件架构

HCCL 由两个子库组成：

| 子库 | 职责 |
|---|---|
| **HCCL 集合通信库** | 内置算子（AllReduce/AllGather/ReduceScatter/Broadcast/AlltoAll/Send/Recv）+ 扩展算子（用户用 HCOMM 接口自定义） |
| **HCOMM 通信基础库** | 分层解耦：**控制面**（拓扑查询、通信资源管理）+ **数据面**（数据搬运、算子间同步、通信计算） |

控制面提供资源，数据面提供操作方法。这种分层使算子开发者无需关心底层芯片实现细节。

### 3.2 支持的通信链路

HCCL 文档明确列出四种高速链路：

| 链路 | 定位 | 对标 NVIDIA 侧 |
|---|---|---|
| **HCCS**（Huawei Cache Coherent System） | 昇腾 NPU 片间互联，Ascend 910/920 系列 | NVLink |
| **UB**（Unified Bus / 灵衢） | 统一全栈互联，Ascend 950 系列起 | NVLink + PCIe + RoCE 的统一替代 |
| **RoCE**（RDMA over Converged Ethernet） | 跨机互联 | InfiniBand / RoCEv2 |
| **PCIe** | CPU↔NPU 通用 I/O | PCIe |

> [!note] HCCS → UB 演进路线
> - **HCCS 1.0**：Ascend 910 系列，第一代高速缓存一致性总线
> - **HCCS 2.0**：Ascend 920，片间带宽 480 GB/s，支持 4 芯片互联
> - **UB 1.0**：Atlas A3 产品（基于 UnifiedBus 1.0）
> - **UB 2.0**：Atlas 950 SuperPoD（8192 NPU，灵衢 2.0）
>
> HCCL 文档同时列出 HCCS 和 UB，是因为它同时支持旧款（910/920 用 HCCS）和新款（950 用 UB）硬件。

### 3.3 UB 协议核心设计

| 设计要素 | 说明 |
|---|---|
| **全对等架构** | 所有组件都是 UBPU（UB Processing Unit），地位平等，通过 EID（Entity ID）寻址，CPU 不再是通信中枢 |
| **URMA 语义** | Unified Remote Memory Access——任何设备可 Load/Store 直接访问远端设备内存，零拷贝、绕过 OS，类似 RDMA 但统一 |
| **Jetty 抽象** | 替代 RDMA 的 Queue Pair。Jetty 是应用的"泊位"（~20 B，持有完成队列句柄+访问令牌），TP Channel 是每远端主机的传输状态（~56 B）。状态按 $O(N+M)$ 增长 |
| **无连接** | 不维护端到端连接状态，目的地由包头携带，任何 Jetty 通过共享 TP Channel 池访问任意远端 |
| **弱序** | 默认允许乱序执行和乱序完成（多路并行），应用可按需配置排序保证 |
| **双数据路径** | 短同步操作走 Load/Store 路径（~500 ns/64B），长吞吐操作走 work-queue 路径 |

### 3.4 通信原语对照

| 原语 | NCCL | HCCL | 说明 |
|---|---|---|---|
| AllReduce | ✅ | ✅ | 全局规约 |
| AllGather | ✅ | ✅ | 全收集 |
| ReduceScatter | ✅ | ✅ | 规约分散 |
| Broadcast | ✅ | ✅ | 广播 |
| AlltoAll | ✅ | ✅ | 全交换 |
| Send/Recv | ✅ | ✅ | 点对点 |

### 3.5 通信算法对照

| 算法 | NCCL | HCCL |
|---|---|---|
| Ring | ✅ | ✅ |
| Tree | ✅ | — |
| Mesh | — | ✅ |
| Recursive Halving-Doubling (RHD) | ✅ | ✅ |

### 3.6 UB-Mesh 拓扑

Atlas 950 SuperPoD 采用 **nD-FullMesh** 分层拓扑：

- **1D**：单板上相邻 NPU 全连接
- **2D**：机架内板间全连接（每架 64 NPU，8 板×8 芯片）
- **3D+**：跨架通过 UB Switch 层互联
- **NPU IO**：x72 UB lanes（2 个 UB IO 控制器，每个 x36）
- **CPU IO**：x32 UB
- **跨架默认**：UB x16（长序列 64K+ 需 x32）

> [!tip] UBoE（UB over Ethernet）
> UB 也支持运行在以太网上（UBoE），客户可复用现有以太网交换机。UBoE 相比 RoCE 集群有更低静态延迟、更高可靠性、更少交换机和光模块需求。

## 4. 数学原理 / 公式

### 4.1 状态复杂度：UB $O(N+M)$ vs RoCE $O(N \cdot M)$

传统 RoCE 的 Queue Pair 状态与"本地应用数 $N$ × 远端端点数 $M$"成正比：

$$\text{QP 状态量} \propto N \times M$$

在 $N = M = 1024$ 时，QP 表约 537 MB，溢出片上 SRAM（~256 KB），每次操作需从 DRAM 重新加载上下文（~1 μs 级惩罚）。

UB 将连接状态拆分为两层：

$$\text{UB 状态量} = \underbrace{N \times 20\text{ B}}_{\text{Jetty（应用层）}} + \underbrace{M \times 56\text{ B}}_{\text{TP Channel（传输层）}} + \text{内存区域表} \approx 110\text{ KB}$$

在 $N = M = 1024$ 时仅 ~110 KB，常驻片上 SRAM。控制器因此可放在 CPU 片上总线而非 PCIe 背后，CPU 的 Load/Store 直接到达——省去 PCIe DMA（250–500 ns/次）。

### 4.2 64B 远程读取延迟对比

| 路径 | 延迟 | 倍率 |
|---|---|---|
| UB Load/Store（URMA bypass） | ~500 ns | 1.0× |
| UB work-request（READ） | ~757 ns | 1.51× |
| RoCE BF（BF NIC） | ~1686 ns | 3.37× |
| RoCE DMA（标准 NIC） | ~2186 ns | 4.37× |

差距主要由 target 侧 NIC↔DRAM 的 PCIe DMA 贡献：RoCE 每次操作付 250–500 ns PCIe DMA，UB 仅付 ~30 ns 片上总线穿越。

### 4.3 带宽对比

| 互联 | 单 lane 带宽 | 最大规模 |
|---|---|---|
| PCIe 6.0 | 64 GT/s | 1 设备/槽 |
| NVLink 5/6 | 224 Gbps | 72 GPU（NVL72） |
| UALink 1.0 | 200–224 Gbps | 1024 加速器 |
| **UB（灵衢）** | **118 Gbps** | **8192 UBPU** |
| Ethernet 802.3ck | 106.25 Gbps | 数万节点 |

> [!note] UB 的设计权衡
> UB 单 lane 118 Gbps **低于** NVLink 6（224 Gbps），但用大规模 FullMesh（8192 卡）的多路聚合弥补单 lane 带宽不足。这是"以规模补单 lane"的工程权衡——降低 SerDes 设计难度和功耗，11% 快于以太网即可。

### 4.4 Ring All-Reduce 通信量（NCCL/HCCL 共享）

Ring all-reduce 分 reduce-scatter + all-gather 两阶段，每阶段 $N-1$ 步：

$$\text{每 rank 通信量} = \frac{2(N-1)}{N} M \approx 2M$$

其中 $N$ 为 rank 数，$M$ 为张量大小。总通信量 $\approx 2M$，与 $N$ 无关——这是 ring 方案的核心优势，NCCL 和 HCCL 都用此算法。详见 [[NCCL核心机制]]。

## 5. 代码示例

> [!warning] 前置条件
> 以下代码需在昇腾 NPU + CANN 环境运行（需安装 `torch_npu` 适配包）。普通 GPU 环境无法运行 HCCL。

```python
# HCCL 基本使用（昇腾 NPU 环境）
# 前置：pip install torch torch_npu; 安装 CANN 工具包并 source environment
import os
import torch
import torch.distributed as dist
import torch_npu  # 导入后自动注册 "hccl" backend

# 1. 环境变量（昇腾侧与 NCCL 侧对照）
# NCCL: NCCL_DEBUG=INFO          HCCL: 无直接等价（用 CANN 日志）
# NCCL: NCCL_SOCKET_IFNAME=eth0  HCCL: 由 rank_table 配置
# HCCL: HCCL_OP_RETRY_ENABLE="L0:0, L1:1, L2:1"  通信算子级重执行

# 2. 初始化进程组（与 torch.distributed API 完全一致）
#    唯一区别：backend="hccl" 而非 "nccl"
dist.init_process_group(
    backend="hccl",          # ← 关键：HCCL 替代 NCCL
    init_method="env://",
    rank=int(os.environ["RANK"]),
    world_size=int(os.environ["WORLD_SIZE"]),
)

# 3. 通信原语用法与 NCCL 完全一致
tensor = torch.randn(1000, device="npu")  # device="npu" 替代 "cuda"
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # HCCL 底层走 HCCS/UB 链路

# 4. 异步通信
tensor2 = torch.randn(1000, device="npu")
work = dist.all_reduce(tensor2, op=dist.ReduceOp.SUM, async_op=True)
# ... 可在此做计算 ...
work.wait()

dist.destroy_process_group()
```

> [!tip] 框架迁移成本极低
> 从 GPU+NCCL 迁移到 NPU+HCCL，PyTorch 侧代码只需改两处：`backend="nccl"` → `"hccl"`、`device="cuda"` → `"npu"`。这是因为 `torch_npu` 适配层将 `torch.distributed` API 映射到 HCCL 调用。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[torch.distributed]]（HCCL 作为 backend 插入 torch.distributed 抽象层）；CANN（HCCL 是 CANN 组件）
- **下游（应用）**: [[Distributed Data Parallel]]（DDP 在昇腾上用 HCCL 做 gradient all-reduce）；[[Fully Sharded Data Parallel]]（FSDP 的 gather/scatter 走 HCCL）；[[专家并行]]（MoE All-to-All 通信走 HCCL）
- **对比 / 易混**: [[NCCL backend]]（NVIDIA GPU 侧对应物）；[[NVLink与NVSwitch]]（GPU 片间互联，对应 HCCS/UB）；[[InfiniBand]]（跨机互联，对应 RoCE/UBoE）

## 7. 常见误区与易错点

> [!warning] 两个 UB，不要混淆
> 华为语境有两个 **UB**：
> 1. **UB (Unified Bus / 灵衢)** — 互联协议层面，NPU 间高速全栈互联（本文主题）
> 2. **UB (Unified Buffer)** — AI Core（达芬奇架构）内部片上存储单元，是 Vector 核的私有缓存（类似寄存器文件/L1），CANN 文档"图融合和 **UB** 融合规则"指的是这个 Buffer 层面的算子融合
>
> 两者完全不同。讨论通信链路时指前者；讨论算子融合/存储层级时指后者。

> [!warning] UB ≠ HCCS
> HCCS 是 Ascend 910/920 的片间互联（类似 NVLink），UB 是 950 起的统一全栈协议。HCCL 文档同时列出两者是因为支持不同代硬件。不要把 UB 当 HCCS 的别名。

> [!warning] UB 单 lane 不等于 NVLink
> UB 单 lane 118 Gbps 低于 NVLink 6 的 224 Gbps。UB 的优势在大规模 FullMesh（8192 卡）而非单 lane 带宽。在同等互联域规模下，UB 需要更多 lane 数来补偿——增加 pin 和线缆复杂度。

## 8. 延伸细节

### 8.1 C-AQM 拥塞控制

UB 的 **C-AQM**（Congestion Active Queue Management）采用"请求–精确反馈"闭环：交换机主动向发送方反馈可用速率，使发送速率精确匹配网络服务能力，交换机队列维持在近零水平。对比 RoCE 的 DCQCN（被动"先拥塞后减速"），C-AQM 消除 Bufferbloat 造成的高延迟，是 UB 微秒级端到端延迟的关键基石。

### 8.2 可靠性机制

| 层级 | 技术 | 作用 |
|---|---|---|
| 物理层 | Lane decrease | 单 lane 故障时降级运行，零丢包 |
| 链路层 | LLR（Link Layer Retry） | 比特错误/间歇断连重传 |
| 模块级 | 2+2 光模块备份 | 单光模块故障时业务无感知恢复 |
| 算子级 | HCCL_OP_RETRY_ENABLE | 通信算子级重执行（L0 机内/L1 机间/L2 超节点间），~95% 成功率 |
| 网络层 | 多路交换 + 端到端重试 | 死锁规避 + 带宽利用率 |

HCCL 重执行环境变量示例：
```bash
export HCCL_OP_RETRY_ENABLE="L0:0, L1:1, L2:1"
export HCCL_OP_PARAMS="MaxCnt:3, HoldTime:5000, IntervalTime:1000"
```

### 8.3 工程效果

- Atlas 950 SuperPoD 在 DeepSeek V3 PP8/DP64/EP64 配置下，unmasked 通信比传统服务器集群降低 80%
- UB-Mesh 相比传统 Clos 架构：2.04× 成本效率、7.2% 可用性提升、95%+ 训练线性度
- 长 64K–10M 序列下，跨架 UB x32 相比 x16 带来 1.85% 性能提升（TP/SP 跨架通信敏感）

### 8.4 UB 硬件实现

UB 控制器集成在 Ascend 950 芯片内部片上总线上（不是独立 NIC），与 AI 加速器并列。用户态 verb 库和内核驱动开源，协议实现闭源。开源实现参考 OpenURMA（arXiv:2605.28717），在 Alveo U50 上做了 RTL/SystemC/gem5 三级建模验证。

---
相关: [[NCCL backend]] [[NCCL核心机制]] [[NVLink与NVSwitch]] [[InfiniBand]] [[torch.distributed]] [[Distributed Data Parallel]] [[专家并行]]
