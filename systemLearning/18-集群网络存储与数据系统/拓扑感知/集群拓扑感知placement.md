# 集群拓扑感知 placement

> **所属章节**: [[拓扑感知]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: topology-aware placement / 拓扑感知调度 / NUMA-aware placement / NVSwitch-domain placement / 跨交换机感知 / affinity-aware scheduling
> **难度**: 中高（需懂 [[NVLink与NVSwitch]] / [[PCIe]] / [[InfiniBand]] / [[NCCL核心机制]] / [[Tensor Parallel]] / [[Pipeline Parallel]] / [[placement group]]）


## 1. 一句话定义

**集群拓扑感知 placement（topology-aware placement）** 是调度器在给分布式训练任务分配 GPU/CPU/网卡资源时，**感知物理硬件拓扑层次**——从近到远依次是 **NUMA 节点（CPU 内存节点，同 NUMA 访存快、跨 NUMA 访存慢）→ PCIe switch（同 switch 下 GPU P2P 快、跨 switch 走 root 慢）→ NVSwitch domain（同 domain 内 GPU NVLink 全连接、跨 domain 走 IB 慢）→ ToR 交换机（同机柜/同 ToR 节点间带宽高、延迟低）→ spine 交换机（跨 spine 走胖树上行、带宽竞争）**——并把**通信最频繁的并行组放在最快的链路上**（TP 组放同 NVSwitch domain 吃 NVLink、PP 组可跨 domain 因 PP 通信量小、DP 组跨节点走 IB），**避免把高带宽需求的集合通信铺到慢链路上导致延迟暴增、训练吞吐塌陷**。它是 [[Megatron-LM]] 3D 并行调度、Ray [[placement group]] 资源打包、Kubernetes GPU 拓扑调度背后的**通用原则**；NCCL 自身也有拓扑探测（`nvidia-smi topo`、`NCCL_TOPO_DUMP_FILE`、自动选 channel/transport），但**框架层先做好 placement 才能让 NCCL 建出最优通信图**——placement 是"把对的 rank 放到对的硬件上"，NCCL 是"在给定硬件上建最优环"，两者互补缺一不可。

> [!note] 三句话定位
> - **是什么**：调度器按硬件拓扑层次（NUMA → PCIe switch → NVSwitch domain → ToR → spine）把并行组 rank 放到"通信最近"的硬件上，让高带宽集合通信走快链路。
> - **为什么**：TP/SP 每层 all-reduce/all-gather 通信量 ∝ 激活、频率极高，若跨慢链路（如跨 NVSwitch domain 走 IB、或跨 spine），通信延迟暴涨一个量级，GPU 算力闲置、吞吐塌陷；placement 把它们压回同 NVSwitch domain 内走 NVLink（300~900 GB/s vs IB 25~100 GB/s）。
> - **与 [[NVLink与NVSwitch]] 关系**：NVLink/NVSwitch 提供"机内全连接"的物理能力，拓扑感知 placement 决定"哪些 rank 共享这片全连接域"——TP 组必须落在同 NVSwitch domain 才能吃满 NVLink，跨 domain 就退化成 IB，TP 就不可行。placement 是 NVLink 价值兑现的调度前提。


## 2. 为什么需要它（动机与背景）

### 2.1 硬件拓扑是异构分层、带宽差几个量级

一台 8 卡 DGX A100/H100 内部就不是"扁平 8 卡"，而是分层的：

```
CPU socket0 (NUMA 0) ── root port ── PCIe switch ── GPU0,1,2,3 + NIC0
CPU socket1 (NUMA 1) ── root port ── PCIe switch ── GPU4,5,6,7 + NIC1
       │
       └── 所有 8 GPU 通过 baseboard 上的 NVSwitch 全连接 (NVSwitch domain = 8 卡)
```

跨节点又有机柜/ToR/spine 层次。**从近到远，每跳一层带宽降、延迟升**：

| 层次 | 典型带宽（单向） | 典型延迟 | 通信代价 |
|---|---|---|---|
| 同 GPU（HBM） | 1.5~3 TB/s | ~ns | 免费（计算即访存） |
| 同 NVSwitch domain（NVLink） | 300~900 GB/s | 1~2 μs | 极低，TP 可行 |
| 跨 PCIe switch（同 NUMA） | 32~64 GB/s | 2~4 μs | 中，PP/小通信可接受 |
| 跨 NUMA（CPU 内存节点） | 64~128 GB/s（QPI/UPI） | 5~10 μs | 中高 |
| 同 ToR（跨节点 IB/RoCE） | 25~100 GB/s | 1~5 μs | 高，DP/PP 可接受 |
| 跨 spine（跨机柜 IB） | 25~100 GB/s（但竞争） | 5~20 μs | 很高，需 AR/ECMP 缓解 |

带宽差 **NVLink : PCIe : IB : 跨 spine ≈ 900 : 64 : 50 : (竞争)**，量级跨度极大。把一次 all-reduce 铺到跨 spine 上，通信时间可能比同 NVSwitch domain 慢 **几十倍**。

### 2.2 集合通信的通信量随并行方式而异

不同并行范式每步通信量天差地别：

- **TP（Tensor Parallel）**：每层前向 + 反向都 all-reduce/all-gather **激活**，通信量 ∝ batch×seq×hidden × 层数，频率极高、带宽需求最大 → **必须 NVLink**。
- **SP（Sequence Parallel）**：借机内 NVLink，通信量比 TP 小但仍频繁 → 同 NVSwitch domain。
- **PP（Pipeline Parallel）**：stage 间点对点 send/recv **激活/bubble 填充**，通信量 ∝ batch×seq×hidden（单向单次），频率低（每 stage 一次）→ **可跨 NVSwitch domain / 跨节点 IB**。
- **DP（Data Parallel）**：每步 all-reduce **梯度**，通信量 ∝ 参数量，一步一次 → **跨节点 IB**，靠 [[alpha-beta性能模型]] 把延迟摊到反向计算里 overlap。
- **EP（Expert Parallel）**：MoE all-to-all token 路由，通信量 ∝ token×hidden，跨专家分布 → **机内 NVLink + 跨机 IB 混合**。

### 2.3 不感知拓扑，吞吐直接塌陷

假设有 4 台 8 卡 H100（4×8=32 卡），要跑 TP=8 + DP=4。**错误放法**：把 TP 组的 8 个 rank 随机分散到 4 台机器（每台 2 个 rank）——TP 的每层 all-reduce 要跨 4 个节点走 IB，带宽从 NVLink 450 GB/s 退化到 IB 50 GB/s（**慢 9 倍**），且每层都要跨机同步，GPU 大量时间在等通信，MFU（model FLOPs utilization）从 50% 掉到个位数。**正确放法**：每个 TP 组的 8 个 rank 全放同一台机器（同 NVSwitch domain），4 个 DP 组各占一台——TP 通信全在机内 NVLink，DP 跨机走 IB。这就是拓扑感知 placement 的核心价值。

> [!warning] 误区：以为"NCCL 自动选最优链路就万事大吉"
> NCCL 的拓扑感知是**在给定 rank 分布下**建最优通信图——它能把 ring 铺到 NVLink 上，但**无法把跨了节点的 TP 组"拉回"机内**。如果 placement 把 TP 组拆到 4 台机器，NCCL 只能在 4 台机器间建 IB ring，带宽上限就是 IB。**placement 决定上限，NCCL 决定逼近上限的程度**。框架层 placement 不对，NCCL 再优化也救不回来。


## 3. 核心概念详解

### 3.1 拓扑层次全图（从近到远）

```
┌─────────────────────────────────────────────────────────────────┐
│  spine 交换机层 (胖树上行, 跨机柜, 带宽竞争 + AR/ECMP)            │  最远
│   ├── ToR0 (机柜0)         ├── ToR1 (机柜1)                      │
│   │   ├── node0 (8 GPU)    │   ├── node2 (8 GPU)                  │
│   │   │   ├─ NVSwitch domain (8 卡全连接, NVLink)                 │
│   │   │   ├─ NUMA0: GPU0-3 + NIC0   NUMA1: GPU4-7 + NIC1          │
│   │   │   └─ 每 GPU 12/18 NVLink 全打到 NVSwitch                   │
│   │   └── node1 (8 GPU)   └── node3 (8 GPU)                       │
│   └── 跨 ToR 走 spine 上行 (慢, 需 AR 缓解)                        │
└─────────────────────────────────────────────────────────────────┘
```

- **NVSwitch domain**：同一组 NVSwitch 芯片下的 GPU 池，**域内 NVLink 全连接**。DGX A100/H100 单机 = 1 个 domain（8 卡）。Blackwell NVL72 = 1 个 domain（72 卡，跨机柜用 NVLink Switch）。
- **跨 domain**：NVLink 不出 domain，跨 domain 必须走 IB/RoCE（除非用 NVLink-Network 这种特殊产品）。
- **NUMA**：CPU 侧的内存节点划分。GPU0-3 挂 NUMA0 的 CPU socket，GPU4-7 挂 NUMA1。CPU 侧算子（数据加载、tokenizer、控制流）跨 NUMA 访存慢；GPU 侧经 GPUDirect 时若 NIC 与 GPU 跨 NUMA，PCIe P2P 也要绕路。
- **PCIe switch**：机内 PCIe crossbar，同 switch 下 GPU↔GPU P2P / GPU↔NIC 直连快；跨 switch 走 root port 上行再下行，慢。
- **ToR（Top-of-Rack）**：机柜顶交换机，同 ToR 下节点间一跳、带宽高；跨 ToR 走 spine 上行，带宽竞争、延迟增。

### 3.2 NUMA 感知（CPU 侧）

NUMA（Non-Uniform Memory Access）是多 socket 服务器的 CPU 内存节点划分：

- **NUMA node**：一个 CPU socket + 它本地直连的内存（DRAM）+ 挂在它 root port 下的 GPU/NIC。
- **本地访存**：CPU 访问自己 NUMA 的内存快（DRAM 带宽 ~100-200 GB/s，延迟 ~100ns）。
- **跨 NUMA 访存**：CPU 访问另一 socket 的内存要走 QPI/UPI 互连，带宽降、延迟升（~200-300ns）。
- **对 GPU 训练的影响**：
  1. **数据加载 worker**：若 DataLoader worker 进程跑在 NUMA1 的 CPU 上，却读 NUMA0 挂的 NVMe，I/O 跨 NUMA；应 `numactl --cpunodebind` 把 worker 绑到数据所在 NUMA。
  2. **GPUDirect RDMA**：若 NIC0 挂 NUMA0、GPU4 挂 NUMA1，NIC↔GPU 的 PCIe P2P 要跨 NUMA root，延迟增；理想是 NIC 与 GPU 同 NUMA（同 PCIe switch）。
  3. **CPU 侧算子**（tokenizer、reward 计算）：跨 NUMA 访存慢，应绑核。

- **查看 NUMA 拓扑**：

```bash
# 看 CPU 的 NUMA 节点与各 GPU/NIC 的归属
numactl --hardware        # 列出 NUMA node 与各 node 的 CPU/内存
nvidia-smi topo -m        # 矩阵里 PIX/PHB/NV#/SYS 等标记含 NUMA 归属
lspci | grep -i mellanox  # 看 NIC 挂哪个 root port (推 NUMA)
cat /sys/bus/pci/devices/<pci_addr>/numa_node  # 直接查某 PCI 设备的 NUMA
```

### 3.3 NVSwitch domain 感知（GPU 侧核心）

见 [[NVLink与NVSwitch]] §3.3。核心：

- **同 domain = NVLink 全连接**，任意两 GPU 间聚合带宽打满（A100 600、H100 900 GB/s 双向）。
- **跨 domain = 退到 IB/RoCE**，带宽骤降到 25~100 GB/s。
- **TP 组必须同 domain**：TP 每层 all-reduce 通信量 ∝ 激活、频率极高，只有 NVLink 带宽 + 微秒延迟能压住；跨 domain TP 不可行。

```bash
# 确认是否同 NVSwitch domain (全连接)
nvidia-smi topo -m
# 期望: 所有 GPU 对都是 NV12 (A100) / NV18 (H100), 无 PIX/PHB/SYS 空洞
# 若出现 PIX/PHB -> 该对走 PCIe (非 NVLink 全连接), 不能放同一 TP 组
```

### 3.4 PCIe switch 层级

见 [[PCIe]] §3.2。同 PCIe switch 下的 GPU 间 P2P 走 switch 片上，延迟 2~4 μs、带宽 Gen4 x16 双向 64 GB/s；跨 switch 走 root port 上行再下行，延迟增、带宽竞争 root 端口。无 NVLink 的卡（消费卡、推理卡）只能靠 PCIe P2P。

### 3.5 跨交换机（ToR / spine）

跨节点走 [[InfiniBand]]/[[RoCE]] 的胖树 fabric：

- **同 ToR**：节点间一跳，带宽高、延迟低（~1-5 μs）。
- **跨 ToR（走 spine）**：上行到 spine 再下行，跳数增、延迟升、且多路径 ECMP/AR 可能 hash 冲突导致单链路拥塞。
- **rail-optimized（轨道优化）**：每节点的 NIC0 连交换机 rail0、NIC1 连 rail1……同一 ring/tree 尽量走同一 rail，避免跨 rail 拥塞。NCCL 用 `NCCL_CROSS_NIC` 控制（见 [[NCCL核心机制]] / [[InfiniBand]] §8.3）。

> [!tip] 实践：跨 spine 的性能塌陷很隐蔽
> 跨 spine 的延迟增加可能只有几 μs，但 all-reduce ring 跨 spine 跳数多、且胖树上行带宽被多节点共享，**聚合带宽上不去**。表现是 nccl-tests 测单机 all-reduce 几百 GB/s、跨机几十 GB/s、跨机柜个位数 GB/s——排查时先 `ibnetdiscover` 看拓扑是不是预期胖树、有没有单点链路，再 `NCCL_DEBUG=INFO` 看 ring 拓扑走了几跳。

### 3.6 NCCL 的拓扑探测

NCCL 启动时自动探测 GPU 互联拓扑，建 channel 选 transport：

- **探测数据源**：`nvidia-smi topo`（CUDA 的 `cuDeviceGetP2PAttribute`）、IB 设备（`ibv_devinfo`）、PCI 拓扑（`lspci`）。
- **建图**：把每对 GPU 的连接类型（NV#/PIX/PHB/SYS/NET）与带宽建成加权图，在每个 communicator 上求最优 ring/tree。
- **选 transport**：NVLink (`NV#`) > PCIe P2P (`PIX`) > SHM（跨 NUMA 共享内存）> NET（IB/RoCE）。
- **channel**：一对收发 GPU 对，多 channel 并行用满带宽。NVSwitch 全连接下 channel 数多。
- **手动干预**：
  - `NCCL_TOPO_FILE`：用外部 XML 覆盖探测到的拓扑（调试/虚拟机/容器无真实拓扑时用）。
  - `NCCL_TOPO_DUMP_FILE`：把探测到的拓扑 dump 出来供排查。
  - `NCCL_P2P_DISABLE`：禁用 P2P（某些 PCIe 拓扑 P2P 坏时用，性能掉一半）。
  - `NCCL_IB_DISABLE`：禁 IB 退化到 TCP socket（排查 IB 问题时用）。
  - `NCCL_MAX_NCHANNELS` / `NCCL_MIN_NCHANNELS`：调 channel 数。

```bash
# NCCL 启动时 dump 拓扑
NCCL_DEBUG=INFO NCCL_TOPO_DUMP_FILE=topo.xml ./train.py
# 看 topo.xml 里每对 GPU 的 link 类型, 对照 nvidia-smi topo -m
```

### 3.7 调度策略：并行组 vs 拓扑层次

核心原则——**通信越频繁/带宽需求越大的组，放越近的层次**：

| 并行组 | 每步通信 | 频率 | 带宽需求 | 放置层次 |
|---|---|---|---|---|
| **TP 组** | all-reduce/all-gather 激活 | 每层 ×3（前向+input grad+weight grad） | 极高 | **同 NVSwitch domain（机内 NVLink）** |
| **SP 组** | all-gather/reduce-scatter（部分激活） | 每层 | 高 | 同 NVSwitch domain |
| **EP 组** | all-to-all token 路由 | 每层（MoE） | 高 | 机内 NVLink + 跨机 IB 混合 |
| **PP 组** | send/recv 激活（点对点） | 每 stage 一次 | 中 | 可跨 domain / 跨节点 IB |
| **DP 组** | all-reduce 梯度 | 每步一次 | 中（靠 overlap 藏） | 跨节点 IB |

推论：
1. **TP 先占机内**：3D 并行里 TP 维度优先填满单机 NVSwitch domain（TP=8 一台机）。
2. **PP 跨机/跨 domain**：PP 通信量小且可 overlap（1F1B bubble），可跨 NVSwitch domain 甚至跨节点。
3. **DP 填剩余**：DP 维度跨节点，每台机一个 DP rank。

例：32 卡 H100（4 机 ×8 卡），3D 并行 TP=8 / PP=4 / DP=1：
- TP=8 → 一台机的 8 卡（同 NVSwitch domain，NVLink）。
- PP=4 → 4 台机（stage 0 在 node0、stage 1 在 node1……跨机 IB send/recv）。
- DP=1 → 单组。

### 3.8 Ray placement group 与拓扑感知结合

见 [[placement group]]。Ray 的 placement group（PG）是资源预留 + 放置策略（PACK/SPREAD/STRICT_PACK/STRICT_SPREAD），**本身不直接感知 GPU NVLink 拓扑**——它管的是"哪些 bundle 同节点"。要实现拓扑感知需：

1. **bundle = 一台机的全部 GPU**：建含 N 个 `{GPU:8}` bundle 的 PG，策略 `STRICT_PACK` 强制每 bundle 同节点 → 天然让一个 TP 组落同 NVSwitch domain。

```python
import ray
from ray.util.placement_group import placement_group

# 一个 PG 含若干 8-GPU bundle, 每个填满一台 8 卡机 -> 每 bundle 即一个 NVSwitch domain
pg = placement_group(
    [{"GPU": 8, "CPU": 16} for _ in range(4)],   # 4 个 TP 组 = 4 台机
    strategy="STRICT_PACK"                        # 每 bundle 必须同节点
)
ray.get(pg.ready())

# 每个 actor 绑一个 bundle -> 8 个 rank 同节点同 NVSwitch domain -> TP 全 NVLink
for i in range(4):
    workers = [Worker.options(placement_group=pg,
                              placement_group_bundle_index=i).remote() for _ in range(8)]
```

2. **bundle 内再排 rank**：Ray 把 8 个 actor 放同节点后，**框架层（Megatron/verl）再按 NVSwitch domain 内的 GPU 拓扑排 rank**——把 TP 组的相邻 rank 放 NVLink 直连的 GPU 对上（`nvidia-smi topo -m` 里 NV 数最大的对）。Ray 不做这层，框架做。

3. **NCCL 拓扑兜底**：即便框架层没精细排 rank，只要 8 个 rank 同节点，NCCL 探测到 NVSwitch 全连接后会自动建最优 ring。所以"PG 保证同节点 + NCCL 探测"组合通常够用。

> [!note] Ray PG 不感知 NVLink，但够用
> Ray PG 的策略粒度到"节点"（同/不同节点），不到"同 NVSwitch domain 内哪两卡 NVLink 直连"。但只要一个 TP 组全在同节点，NCCL 会接管把 ring 铺到 NVLink 上。所以实践中 **PG 做"同节点装箱"、NCCL 做"节点内 NVLink 建环"** 两层分工即可。极端拓扑（如 NVL72 跨机柜 NVLink-Network）才需框架层更精细的 rank 排布，**具体 NVLink-Network 跨机 rank 排布策略待核实**。

### 3.9 Kubernetes 与 Slurm 的拓扑调度

- **Kubernetes**：`DevicePlugin` + `Topology Manager`（`single-numa-node`/`best-effort`/`restricted` 策略）可做 NUMA 感知；NVIDIA 的 `gpu-feature-discovery` + `nvidia.com/gpu` 资源可配 `topology.nvidia.com/gpu`。K8s 对 GPU NVLink 拓扑的调度支持弱于 Slurm，复杂训练多用 Slurm 或 Ray。
- **Slurm**：`--gpus-per-node` + `--gpu-bind`（`map_gpu`/`single`）做 NUMA/PCIe 绑定；`gres.conf` 配 GPU 拓扑；Slurm 对 HPC 拓扑调度成熟。


## 4. 数学原理 / 公式

### 4.1 跨链路延迟累积

一次集合通信若要跨 $H$ 跳异构链路，每跳带宽 $B_i$、延迟 $\alpha_i$，则总延迟（粗略，串行跳）：

$$
T \approx \sum_{i=1}^{H} \left(\alpha_i + \frac{M_i}{B_i}\right)
$$

其中 $M_i$ 是第 $i$ 跳上传输的消息量。**瓶颈是带宽最小的那一跳**（类似木桶效应）——若 ring 有一段跨了 spine（$B_i=10$ GB/s），整环被这段拖慢，NVLink 段再快也没用。这正是 placement 要把 ring 压在快链路上的数学依据。

### 4.2 TP 跨 domain 的代价

设 TP all-reduce 消息 $M$、NVLink 单向带宽 $B_{\text{NV}}$（H100 450 GB/s）、IB 单向 $B_{\text{IB}}$（NDR 50 GB/s）。同 NVSwitch domain（ring 全 NVLink）：

$$
T_{\text{TP, intra}} \approx \alpha_{\text{NV}} + \frac{2(N-1)}{N}\cdot\frac{M}{B_{\text{NV}}}
$$

跨 domain（ring 跨节点走 IB，$k$ 跳跨机）：

$$
T_{\text{TP, cross}} \approx k\left(\alpha_{\text{IB}} + \frac{2(N-1)}{N}\cdot\frac{M}{B_{\text{IB}}}\right) \approx \frac{B_{\text{NV}}}{B_{\text{IB}}}\cdot T_{\text{TP, intra}} \approx 9\times T_{\text{TP, intra}}
$$

跨 domain TP 通信慢约 **9 倍**（H100 NVLink 450 / NDR 50）。若 TP 通信占训练时间 20%（同 domain），跨 domain 就占 ~67%，MFU 从 50% 掉到 ~16%。详见 [[alpha-beta性能模型]] §4 与 [[NVLink与NVSwitch]] §4.4。

### 4.3 PP 跨 domain 的可接受性

PP stage 间点对点 send/recv，单向单次消息 $M_{\text{act}}$（激活大小），带宽 $B$：

$$
T_{\text{PP}} \approx \alpha + \frac{M_{\text{act}}}{B}
$$

无 ring 的 $\frac{2(N-1)}{N}$ 放大，单次单跳。即便 $B=B_{\text{IB}}=50$ GB/s、$M_{\text{act}}=96$ MB，$T_{\text{PP}}\approx 1\mu s + 1.9\text{ms} \approx 1.9$ ms，远小于 stage 计算 time（几十 ms）。且 PP 通信可与上一 stage 反向 overlap。故 **PP 跨 domain/跨节点可接受**，TP 不行。

### 4.4 DP 跨节点的 overlap 可行性

DP all-reduce 梯度 $M_{\text{grad}}$（参数量，GPT-3 175B 约 700 MB ×2 ≈ 1.4 GB），跨机 ring：

$$
T_{\text{DP}} \approx \alpha_{\text{IB}} + \frac{2(N-1)}{N}\cdot\frac{M_{\text{grad}}}{B_{\text{IB}}} \approx 1\mu s + \frac{1.4\text{GB}}{50\text{GB/s}} \approx 28\text{ms}
$$

若反向计算 time > 28 ms（大模型通常如此），DP 通信可藏在反向里 overlap，故 **DP 跨节点可行**（靠 [[overlap strategy]] / 梯度分桶）。

### 4.5 利用率与碎片

设集群 $N$ 节点每节点 $G$ GPU，要放 $P$ 个 TP=$G$ 的组（每组占一节点）。若调度器把组分散（用了 $P$ 节点各占 $G$ 卡，利用率 $G/G=100\%$）vs 装箱（挤到 $\lceil P \cdot G / G \rceil = P$ 节点，同 100%）——TP 维度下两者节点数相同，但 **PP/DP 混合时装箱能腾出整节点给其他组**，分散则碎片化。碎片率公式见 [[placement group]] §一。


## 5. 代码示例

### 5.1 全套拓扑探测（一键体检）

```bash
#!/bin/bash
# topo_audit.sh —— 训练前一键体检集群拓扑
set -e
echo "===== 1. GPU 与 NVLink 拓扑 ====="
nvidia-smi topo -m                       # 矩阵: NV12/NV18=全连接, PIX/PHB=PCIe, SYS=跨NUMA
echo "===== 2. NVLink 链路状态 ====="
nvidia-smi nvlink -s 0 | head -20        # GPU0 各 NVLink Active/Inactive
echo "===== 3. NUMA 归属 ====="
numactl --hardware | head -30            # CPU NUMA 节点
for g in /sys/bus/pci/devices/*/numa_node; do  # 每设备的 NUMA
  echo "$g: $(cat $g)"
done 2>/dev/null | head -20
echo "===== 4. IB 网卡 ====="
ibstat 2>/dev/null | grep -E "mlx|State|Rate" || echo "no IB"
echo "===== 5. NCCL 探测的拓扑 dump ====="
NCCL_DEBUG=INFO NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml python -c "import torch.distributed" 2>/dev/null || true
[ -f /tmp/nccl_topo.xml ] && head -40 /tmp/nccl_topo.xml
```

### 5.2 用 nvidia-smi topo 判断能否放同 TP 组

```python
import subprocess, re

def nvlink_pairs():
    """返回 {(g0,g1): nvlink数}, 用于判断哪些 GPU 对可放同一 TP 组"""
    out = subprocess.check_output(["nvidia-smi", "topo", "-m"]).decode()
    # 解析矩阵, NV12/NV18 = NVLink 全连接, PIX/PHB = PCIe (不适合 TP), SYS = 跨 NUMA
    rows = [l for l in out.splitlines() if re.match(r"^\s*\d+", l)]
    pairs = {}
    for r in rows:
        cells = r.split()
        g0 = int(cells[0])
        for j, c in enumerate(cells[2:], start=0):
            m = re.match(r"NV(\d+)", c)
            if m:
                pairs[(g0, j)] = int(m.group(1))
    return pairs

pairs = nvlink_pairs()
full = [(a,b,n) for (a,b),n in pairs.items() if n >= 12]   # NV12+ = 全连接
print("可放同一 TP 组的 GPU 对 (NVLink 全连接):", full)
# 若某对是 PIX/PHB -> 不能放同一 TP 组 (要走 PCIe, 通信塌陷)
```

### 5.3 Ray PG 实现拓扑感知的 3D 并行

```python
import ray
from ray.util.placement_group import placement_group

ray.init()

# 4 台 8 卡 H100, 3D 并行 TP=8 / PP=4 / DP=1
# 关键: 一个 PG bundle = 一台机的 8 GPU (同 NVSwitch domain), STRICT_PACK 强制同节点
tp_world = 8        # TP 组 = 一台机 8 卡
num_stages = 4      # PP = 4 台机
pg = placement_group(
    [{"GPU": tp_world, "CPU": tp_world * 2} for _ in range(num_stages)],
    strategy="STRICT_PACK",     # 每 bundle 必须同节点 -> 每 TP 组同 NVSwitch domain
)
ray.get(pg.ready())

# 每 stage 一个 worker, 绑一个 bundle -> 8 rank 同节点, NCCL 自动建 NVLink ring
workers = []
for stage in range(num_stages):
    w = Worker.options(
        placement_group=pg,
        placement_group_bundle_index=stage,
    ).remote(tp_world=tp_world, pp_rank=stage, num_stages=num_stages)
    workers.append(w)
# PP stage 间跨节点 IB send/recv (通信量小可接受), TP 组内 NVLink all-reduce (全速)
```

### 5.4 Megatron-LM 的拓扑感知 rank 排布

Megatron-LM 的 `RuntimeArgs`/`DistributedDataParallelConfig` 在初始化时按 3D mesh 排 rank：TP 维度优先填同机 NVSwitch domain、PP 跨机、DP 填剩余。概念伪码：

```python
# Megatron 风格的 rank 排布 (TP, PP, DP)
def ranks_3d(tp_size, pp_size, dp_size):
    # global_rank = tp + pp*tp_size + dp*tp_size*pp_size
    # 同一 (pp,dp) 的 tp 个 rank 连续 -> 同机同 NVSwitch domain
    for dp in range(dp_size):
        for pp in range(pp_size):
            for tp in range(tp_size):
                yield tp + pp * tp_size + dp * tp_size * pp_size, (tp, pp, dp)
# torchrun --nproc_per_node=8 ... 时 LOCAL_RANK 0-7 对应同机, global_rank 跨机
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[NVLink与NVSwitch]]（NVSwitch domain 是 TP placement 的物理基础）、[[PCIe]]（PCIe switch 层级 / NUMA）、[[InfiniBand]]（跨节点 fabric / ToR-spine / SM）、[[RoCE]]（以太网 fabric 替代）、[[RDMA与GPUDirect]]（NIC↔GPU 同 NUMA 直连）、[[alpha-beta性能模型]]（量化跨链路代价）。
- **下游（应用）**: [[NCCL核心机制]]（NCCL 在 placement 给定拓扑下建最优 ring）、[[NCCL通信拓扑]]（NCCL 拓扑探测与 channel）、[[Megatron-LM]]（3D 并行按 NVSwitch domain 排 TP/PP/DP）、[[Tensor Parallel]]（TP 强依赖同 domain）、[[Pipeline Parallel]]（PP 可跨 domain）、[[placement group]]（Ray 的资源装箱实现 placement 约束）、[[3D parallelism]]。
- **对比 / 易混**:
  - **placement vs [[NCCL核心机制]] 拓扑感知**：placement 是框架/调度层把"对的 rank 放对的硬件"（决定上限），NCCL 拓扑感知是库层"在给定硬件建最优环"（逼近上限）。先 placement 再 NCCL。
  - **NVSwitch domain vs NUMA node**：NVSwitch domain 是 GPU 侧 NVLink 全连接域；NUMA node 是 CPU 侧内存节点域。两者维度不同，同 domain 的 GPU 仍可能跨 NUMA（A100 8 卡分属 2 个 NUMA）。
  - **placement group vs 拓扑感知**：PG 是 Ray 的资源装箱机制（同/不同节点），拓扑感知是更细的硬件层次感知（NUMA/NVSwitch/ToR）。PG 做"节点级装箱"，拓扑感知做"节点内+跨节点链路级排布"。
  - **PACK vs SPREAD**：PACK 共置（省通信，适合 TP colocate），SPREAD 分散（提容错，适合无通信依赖的独立任务）。训练 colocate 用 PACK/STRICT_PACK。


## 7. 常见误区与易错点

> [!warning] 误区1：以为 NCCL 拓扑感知能补救 placement 错误
> NCCL 只能在给定 rank 分布下优化。TP 组跨了节点，NCCL 只能建 IB ring，带宽上限就是 IB。**placement 决定上限，先 placement 再 NCCL**。

> [!warning] 误区2：把 TP 组跨 NVSwitch domain
> TP 每层 all-reduce 走 NVLink 才能 <20% 通信开销；跨 domain 走 IB 慢 9 倍，MFU 塌陷。**TP 组必须同 NVSwitch domain**（DGX 单机 = 8 卡 domain）。

> [!warning] 误区3：NUMA 只影响 CPU，与 GPU 无关
> 错。GPUDirect RDMA 时 NIC 与 GPU 跨 NUMA 会绕路、PCIe P2P 跨 NUMA root 延迟增；DataLoader worker 跨 NUMA 读 NVMe 慢。**NUMA 影响数据加载与跨机通信**，需 `numactl` 绑核。

> [!warning] 误区4：以为跨 ToR 和同 ToR 差不多
> 跨 spine 跳数增、带宽被多节点共享竞争，聚合带宽可能差几倍。nccl-tests 测跨机柜 all-reduce 常见个位数 GB/s 即此因。**生产尽量让 DP ring 走同 ToR，或开 IB Adaptive Routing**。

> [!tip] 实践：3D 并行排布口诀
> "TP 填机、PP 跨机、DP 填余"——TP 维度优先占满单机 NVSwitch domain（NVLink），PP 维度跨机（IB，通信小可 overlap），DP 维度填剩余机。`nvidia-smi topo -m` 全 NV# 才能放同 TP 组。

> [!tip] 实践：训练前必查拓扑
> 1. `nvidia-smi topo -m` 全 NV#（无 PIX/PHB 空洞）= NVSwitch domain 全连接；
> 2. `numactl --hardware` + 每 GPU/NIC 的 `numa_node` 同 NUMA（GPUDirect 不绕路）；
> 3. `ibstat` 全 Active、`ibnetdiscover` 是预期胖树无单点；
> 4. `NCCL_DEBUG=INFO` 启动日志出现 `via NV18` 而非 `via PIX/NET`。

> [!note] 补充：NVL72 / NVLink-Network 的 placement
> Blackwell GB200 NVL72 把 NVSwitch domain 扩到 72 卡（跨机柜 NVLink Switch）。此时 TP 可跨机柜（NVLink-Network 带宽仍远高于 IB），但跨 NVL72（>72 卡）仍退到 IB。**NVL72 内的精细 rank 排布与 NVLink Switch tray 的轨道划分，具体策略以 NVIDIA 官方 NVL72 部署指南为准，待核实**。


## 8. 延伸细节

### 8.1 Slurm 的 GPU 绑定与拓扑

Slurm `--gpu-bind=map_gpu:0,1,2,3,4,5,6,7` 把 task 绑定到指定 GPU index；`TopologyPlugin=topology/tree` + `gres.conf` 的 `Name=gpu File=... Type=...` 可配 GPU 拓扑。Slurm 的 `select/cons_tres` 调度器可按 NVLink 拓扑优先共置。HPC 集群多用 Slurm 做拓扑感知。

### 8.2 verl / Ray 的 RL 训推一体 placement

[[actor deadlock]] 与 RL 框架里，Learner（GPU 训练）与 rollout worker（GPU 推理）的 placement 要兼顾：Learner 内部 TP colocate（同 NVSwitch domain），rollout 与 Learner 间若频繁权重同步则 colocate 同节点省 IB，若异步则 SPREAD 提容错。verl 用 `RayResourcePool` + STRICT_PACK 做 worker 装箱，见 [[placement group]] §四。

### 8.3 拓扑探测的边界情况

- **容器/虚拟机**：容器内 `nvidia-smi topo` 可能拿不到真实 NVLink 拓扑（需 `--device` + `--ipc=host` + `--gpus all` 正确透传）。此时用 `NCCL_TOPO_FILE` 注入宿主机真实拓扑 XML。
- **PCIe ACS 阻断 P2P**：Access Control Service 出于 IOMMU 安全会阻断 GPU↔NIC/GPU↔GPU 的 PCIe P2P。训练前要确认 ACS 关闭或支持 P2P 转发（BIOS + `setpci`），否则 GPUDirect 失败、退到 CPU 内存中转。见 [[PCIe]] §3 / [[RDMA与GPUDirect]]。
- **NVLink 链路降速**：`nvidia-smi nvlink -s` 若有 Inactive 链路（硬件故障/未协商），domain 不是全连接，TP 带宽打折。排查物理连接与 `dmesg`。

### 8.4 与 NVLink SHARP / 集合硬件 offload

Hopper + NVSwitch 支持 NVLink SHARP（reduce 在交换机硬件完成），Blackwell 进一步。SHARP 对 placement 要求更高——需所有参与 rank 在同一 NVSwitch（SHARP 域）内，跨 domain 退回 ring。见 [[NCCL核心机制]] §8.1。

### 8.5 内容来源

整理自 NVIDIA DGX/HGX 硬件白皮书、[[NVLink与NVSwitch]]/[[PCIe]]/[[InfiniBand]]/[[RDMA与GPUDirect]] 笔记、NCCL 官方文档（`NCCL_TOPO_FILE`/`NCCL_TOPO_DUMP_FILE`/`NCCL_CROSS_NIC`）、Ray [[placement group]] 文档、Megatron-LM 3D 并行 rank 排布实践。NVL72/NVLink-Network 细节以官方最新文档为准。

---
相关: [[拓扑感知]] | [[NVLink与NVSwitch]] | [[PCIe]] | [[InfiniBand]] | [[RoCE]] | [[RDMA与GPUDirect]] | [[NCCL核心机制]] | [[NCCL通信拓扑]] | [[alpha-beta性能模型]] | [[placement group]] | [[Megatron-LM]] | [[Tensor Parallel]] | [[Pipeline Parallel]] | [[3D parallelism]] | [[overlap strategy]] | [[18-集群网络存储与数据系统]]
