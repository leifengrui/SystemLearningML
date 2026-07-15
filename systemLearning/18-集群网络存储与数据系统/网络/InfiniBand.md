# InfiniBand

> **所属章节**: [[网络]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: IB / InfiniBand / 子网管理器 / Subnet Manager / SM / LID
> **难度**: 中高(需懂 [[RDMA与GPUDirect]] / [[RoCE]] / [[NCCL核心机制]] / [[NVLink与NVSwitch]] / [[alpha-beta性能模型]])


## 1. 一句话定义

**InfiniBand(IB)** 是专为高性能计算与大规模 GPU 集群设计的**专用低延迟高带宽网络标准**(由 IBTA / InfiniBand Trade Association 维护),核心特征是**原生 RDMA**(零拷贝、绕过 CPU/OS)、**硬件级可靠传输**(无 TCP/IP 栈开销)、**专用交换机**、子网由 **Subnet Manager(SM)** 集中分配路径与地址(LID)。速率逐代:SDR/DDR/QDR/FDR/EDR/HDR/**NDR/XDR**,当前 AI 集群主流 **HDR 200 Gb/s、NDR 400 Gb/s、XDR 800 Gb/s**(单端口线速)。GPU 集群跨机互联走 **NCCL over IB**,是 H100/A100 万卡集群事实标准(另见以太网派 [[RoCE]])。**关键数字(必背):HDR=200 / NDR=400 / XDR=800 Gb/s,单向 payload 约 25/50/100 GB/s。**

> [!note] 三句话定位
> - **是什么**:IBTA 标准的专用网络架构(物理层 + 链路层 + 传输层 + 子网管理),原生 RDMA,硬件可靠传输,低延迟(~0.5-1μs 端到端)。
> - **为什么**:以太网+TCP/IP 有协议栈开销、丢包重传、延迟抖动,撑不住万卡 all-reduce;IB 用专用 ASIC 交换机 + RDMA 把端到端延迟压到亚微秒级,集合通信稳定不打嗝。
> - **与 [[RoCE]] 关系**:IB 是"原生 RDMA 专用网",RoCE 是"把 IB 传输层搬到以太网"的兼容方案——同 API(RDMA Verbs)、不同物理层。IB 贵稳,RoCE 便宜通用但需无损配置。


## 2. 为什么需要它(动机与背景)

### 2.1 以太网 + TCP 撑不住万卡训练

万卡 all-reduce 每步要同步所有卡的梯度,通信量 ∝ 参数量(GPT-3 175B 约 700 MB/步 × 多次 all-reduce)。以太网 TCP/IP 路径:GPU→CPU 内存→TCP 栈(拷贝+校验+分段)→NIC→交换机→…→NIC→TCP 栈→CPU 内存→GPU,**两次内核态拷贝 + 协议栈处理**,延迟 10~50 μs,丢包重传抖动巨大。万卡下 all-reduce 时间被网络主导,训练效率塌陷。

### 2.2 RDMA 砍掉 CPU/OS

IB 的核心是 **RDMA(Remote Direct Memory Access)**:NIC 硬件直接读写对端内存,**不进 CPU、不进 OS、零拷贝**。GPU 显存还能被 NIC 直接访问(GPUDirect RDMA),跨机 all-reduce 路径缩短为 GPU→NIC→…→NIC→GPU。详见 [[RDMA与GPUDirect]]。

### 2.3 专用交换机 + 集中子网管理

IB 交换机是专用 ASIC(Mellanox/NVIDIA Quantum 系列),硬件转发 + 硬件可靠传输(基于 credit-based 流控,无损)。子网由 **Subnet Manager(SM)** 集中管理:SM 给每个端口分配 LID(局部地址)、计算路由表、下发给交换机。这种"集中算路 + 交换机只转发"的设计让 IB 拓扑变化、链路故障都能被 SM 重新收敛,**没有以太网 ARP/STP 的分布式协议开销与广播风暴**。

### 2.4 GPU 集群跨机事实标准

DGX GH200 / DGX H100 SuperPOD 等万卡集群用 IB leaf-spine 拓扑跨机互联,NCCL `NET` transport 直接跑在 IB(RDMACM + verbs)。H100 集成 400 Gb/s NDR 网卡(ConnectX-7),GH200 4× 400 Gb/s。IB 是 NVIDIA 收购 Mellanox 后的"亲儿子"网络。

> [!warning] 误区:以为 IB 是"更快的以太网"
> IB 不是以太网的提速版,而是**完全独立的网络架构**(独立物理层 spec、独立帧格式、独立地址体系 LID、独立管理面 SM)。以太网是"分布式、IP 路由、TCP 可靠、广播泛洪",IB 是"集中管理、LID 路由、硬件可靠、credit 流控"。RoCE 才是两者的混合体(把 IB verbs 跑以太网)。


## 3. 核心概念详解

### 3.1 IB 速率代际全表(必背)

| 代 | 名称 | 单端口线速(Gb/s) | 单端口单向 payload(GB/s) | 双向(GB/s) | 典型网卡 |
|---|---|---|---|---|---|
| 1 | SDR | 10 | 2.5 | 5 | —— |
| 2 | DDR | 20 | 5 | 10 | —— |
| 3 | QDR | 40 | 10 | 20 | —— |
| 4 | FDR | 56 | 14 | 28 | —— |
| 5 | EDR | 100 | 25 | 50 | ConnectX-4/5 |
| 6 | **HDR** | 200 | 25 | 50 | ConnectX-6 |
| 7 | **NDR** | 400 | 50 | 100 | ConnectX-7 |
| 8 | **XDR** | 800 | 100 | 200 | ConnectX-8 / Quantum-X800 |

> 标准端口宽度 4X;1X/12X 较少。线速 Gb/s 转 GB/s payload:HDR 200 Gb/s ÷ 8 = 25 GB/s 单向(4096B IB MTU,扣开销实测 ~22-24)。表内为 IBTA 规范 / NVIDIA Quantum 交换机 datasheet 标称值,稳定可信。NDR 400、XDR 800 是 2022/2024 NVIDIA Quantum-2/Quantum-X800 主推速率。

### 3.2 IB 协议栈与帧结构

```
应用 (NCCL/MPI/verbs API: send/recv, RDMA read/write)
  ↓ libibverbs / librdmacm
传输层 (RC/UC/UD: Reliable/Unreliable Connected/Datagram)
链路层 (IB frame, MTU 256B~4096B, credit-based 流控)
物理层 (SerDes, 25/50/100 Gbaud PAM4)
```

- **RC(Reliable Connection)**:1 对 1 可靠队列,GPUDirect RDMA/NCCL 默认用 RC。
- **verbs API**:`ibv_post_send` / `ibv_post_recv` / `ibv_rdma_read` / `ibv_rdma_write`,用户态直接驱动 NIC(内核 bypass)。
- **MTU**:IB 支持 4096B(以太网 jumbo frame 9000B),大 MTU 减少头开销。

### 3.3 Subnet Manager (SM)

- **SM 是什么**:子网管理进程,管理一个 IB 子网内所有交换机与端口的地址、路径。常驻在**管理交换机**(NVIDIA Quantum 交换机内置 SM)或独立服务器。
- **SM 干什么**:
  1. **发现拓扑**:扫描子网,建立交换机/端口拓扑图(类似 STP 但集中)。
  2. **分配 LID**:给每个端口分配 16-bit(或扩展 32-bit)Local Identifier,**IB 路由基于 LID**(非 MAC/IP)。
  3. **计算路由表**:根据 LID 算转发路径(胖树最短路径等),下发给所有交换机。
  4. **持续维护**:链路故障/新设备加入时重收敛(几秒到几十秒)。
- **主备高可用**:SM 单点故障会让整个子网地址失效,生产集群**必须配主备 SM**(Master + Standby,通过 SMInfo 选举),主 SM 挂了备 SM 顶上。`sminfo` 命令可查当前 master SM。
- **OpenSM**:开源 SM 实现(随 OFED 发布),可跑在普通服务器,但生产多用交换机内置 SM。

### 3.4 IB vs 以太网 对比

| 维度 | InfiniBand | 以太网(+TCP/IP) |
|---|---|---|
| 设计目标 | HPC / 大规模 RDMA 集群 | 通用网络 |
| 可靠传输 | 硬件 credit-based,无丢包 | TCP 软件重传,丢包常见 |
| 协议栈 | 内核 bypass(verbs 用户态) | TCP/IP 内核栈,多次拷贝 |
| 地址 | LID(SM 集中分配) | MAC + IP(ARP/分布式) |
| 管理面 | Subnet Manager 集中算路 | STP/ARP/OSPF/BGP 分布式 |
| 延迟(端到端) | ~0.5-1 μs | ~10-50 μs(TCP) |
| 流控 | credit-based 无损 | PFC(需配置)/ 尾丢包 |
| MTU | 4096B | 1500B(标准)/ 9000B(jumbo) |
| 交换机 | 专用 ASIC(贵) | 通用 ToR(便宜) |
| 生态 | NVIDIA/Mellanox 准垄断 | 开放,多厂商 |

### 3.5 NCCL over IB

NCCL 的跨机 `NET` transport 用 IB verbs(RC 连接 + RDMA read/write)。每个 GPU↔GPU 跨机 pair 建一条 RC QP(Queue Pair),NCCL ring 上的跨机跳就用这条 QP GPUDirect RDMA 直读对端 GPU 显存。`NCCL_IB_HCA=mlx5_0` 选网卡,`NCCL_IB_PCI_RELAXED_ORDERING=1` 关 PCIe 顺序约束提性能。

### 3.6 IB 与 RoCE 的关系

IB 是"原生 RDMA 专用网"。**RoCE(RDMA over Converged Ethernet)** 把 IB 的 verbs/RDMA 传输层**搬到以太网物理层上**,让通用以太网交换机也能跑 RDMA。两者**上层 API 相同**(都叫 verbs,NCCL 同一份代码),**物理层与管理不同**:IB 用 LID + SM,RoCE 用 IP + PFC。详见 [[RoCE]]。


## 4. 数学原理 / 公式

### 4.1 线速转 payload 带宽

IB 单端口线速 $R$ Gb/s,4X 宽度,扣 8b/10b(IB 用 64b/66b 或 8b/10b 视代)与帧头开销,单向 payload 带宽近似:

$$
B_{\text{dir}} \approx \frac{R}{8} \;\eta_{\text{hdr}} \quad [\text{GB/s}], \quad \eta_{\text{hdr}}\approx 0.95\sim0.98
$$

HDR(200 Gb/s):$B_{\text{dir}}\approx 25\times0.95\approx23.7$ GB/s,实测 22~24。NDR(400):≈47~50。XDR(800):≈95~100。

### 4.2 跨机 all-reduce 延迟(alpha-beta)

$N$ 卡跨机 all-reduce ring,$\sqrt N \times \sqrt N$ 二维 torus 或 tree,跨机跳数 $H=\sqrt{N/\text{node\_size}}$。单步消息 $M$,链路单向带宽 $B$、延迟 $\alpha$:

$$
T \approx 2H\left(\alpha + \frac{M}{B}\right)
$$

NDR $B\approx50$ GB/s,$\alpha\approx1\mu s$,$M=700$ MB,$H=4$:$T\approx 8(1 + 700/50\times10^3\mu s)\approx 8\times 14001 = 112008\mu s$... 实际 ring 算法每卡只收发 $2(N-1)/N\cdot M$,上述是粗略上界,优化后实测 all-reduce 在万卡 NDR 上达 TB/s 聚合带宽。**精确建模见 [[alpha-beta性能模型]] 与 [[NCCL通信拓扑]]。**

### 4.3 SM 路径数与胖树

胖树(fat tree)$k$ 端口交换机可支撑 $\frac{k^3}{4}$ 端口、$\frac{k^3}{4}$ 端口的非阻塞网络。NDR 64 口交换机 → $\frac{64^3}{4}=65536$ 端口非阻塞(实际 52 口用),够万卡。这是 IB 胖树能 scale 到几万卡的数学依据。


## 5. 代码示例

### 5.1 查 IB 设备状态

```bash
# 列出本机 IB 端口与速率 (State=Active, Rate 字段: 200=HDR, 400=NDR, 800=XDR)
ibstat
# 期望输出:
#   mlx5_0 ... Port 1: State: Active, ... Rate: 400 (NDR)

# 详细端口信息 (含 LID, SM info)
ibv_devinfo -v          # verbose, 看 LID, SM_LID, active_width, active_speed
ibv_devinfo -d mlx5_0

# 看 SM (子网管理器) master 在哪
sminfo                   # 输出 SM LID, priority, 一行一条 (主备)
# 看 IB 路由 (LID -> 端口转发)
ibroute                  # 显示交换机转发表
ibnetdiscover | head     # 拓扑发现 (画 IB 子网拓扑图)
```

### 5.2 测 RDMA 带宽(perftest)

```bash
# 服务端 (机器 A):
ib_write_bw -d mlx5_0 --report_gbits
# 客户端 (机器 B):
ib_write_bw -d mlx5_0 <机器A_IP> --report_gbits
# 期望 HDR ~ 23-24 GB/s, NDR ~ 47-50 GB/s (单口单向)

# RDMA 延迟测试 (1 byte, 看微秒延迟)
ib_write_lat -d mlx5_0  <机器A_IP>   # NDR ~0.7-1.2 μs
```

### 5.3 NCCL over IB 测跨机 all-reduce

```bash
# 关掉 NVLink 只测跨机 IB (单机1卡 ×多机, 或强制 NET)
NCCL_DEBUG=INFO NCCL_NET=IB NCCL_IB_HCA=mlx5_0 \
  ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8   # 8 机 ×1卡 或 1机8卡 看 cross
# 日志应出现 "Channel 00/02 : 0[rx] -> 1[tx] via NET/IB/..." 说明走 IB
```


## 6. 与其他知识点的关系

- **上游(依赖)**: [[RDMA与GPUDirect]](IB 是 RDMA 的原生实现)、[[NVLink与NVSwitch]](机内 vs 机外)、[[PCIe]](NIC 经 PCIe 接 CPU)
- **下游(应用)**: [[NCCL核心机制]](NCCL NET transport over IB)、[[NCCL backend]]、[[集群拓扑感知placement]](胖树 leaf-spine)、[[alpha-beta性能模型]](跨机集合通信建模)
- **对比 / 易混**: [[RoCE]](RDMA 跑以太网,同 API 不同物理层)、[[PCIe]](机内 vs 跨机)、NVLink-Network(机柜级,介于二者)


## 7. 常见误区与易错点

> [!warning] 误区1:把 Gb/s 当 GB/s
> IB 速率标的是 **Gb/s(线速)**,换算 payload 要 ÷8 再扣开销。"400 Gb/s NDR" 单向 payload 约 50 GB/s,不是 400 GB/s。算集合通信时务必换算。

> [!warning] 误区2:以为 SM 挂了网络就断
> SM 挂了**已建路由表仍在交换机内存生效**(转发不停),但**新拓扑/故障无法收敛**。生产必须配主备 SM。但 SM 主切换瞬间会有秒级感知,NCCL 可能短暂报错重连。

> [!warning] 误区3:以为 IB 与以太网能直接互通
> IB 帧与以太网帧**格式不同**,不能直连。要互通需 RoCE 网关或 NVIDIA 的 Ethernet-IB 桥接卡。GPU 集群通常 IB 子网与以太网管理网**物理分离**。

> [!tip] 实践:生产 IB 集群必查
> 1. `ibstat` 全部端口 State=Active、Rate 对(降速排查线缆/光模块);
> 2. 主备 SM(`sminfo` 有 ≥2 条,且 priority 不同);
> 3. `ibnetdiscover` 拓扑是否是预期胖树、有无单点链路;
> 4. NCCL `NCCL_DEBUG=INFO` 日志确认跨机走 `NET/IB` 而非 socket;
> 5. 打开 `NCCL_IB_PCI_RELAXED_ORDERING=1` + BIOS Resizable BAR + ACS 关闭(见 [[PCIe]])。


## 8. 延伸细节

### 8.1 Quantum 交换机代际

| 平台 | 代 | 端口数 | 速率 | 模式 |
|---|---|---|---|---|
| Quantum | HDR | 40 口 | 200 Gb/s | HDR/ split HDR |
| Quantum-2 | NDR | 64 口(52 用) | 400 Gb/s | NDR / split 2× HDR |
| Quantum-X800 | XDR | 144 口 | 800 Gb/s | XDR / split NDR |

> 端口数为 NVIDIA Quantum 系列 datasheet 标称。实际非阻塞可用端口与 split 模式有关。

### 8.2 IB 分裂(split)端口

NDR 400 Gb/s 端口可分裂(split)成 2× HDR 200 Gb/s(用 split cable),给更多 GPU 各自一条 HDR。NDR-2 模式常见于 H100 集成网卡 4× 100 = 400 的组合。算带宽时注意 split 后单子端口带宽减半。

### 8.3 Adaptive Routing(自适应路由)

IB 交换机支持 **Adaptive Routing(AR)**:SM 下发多路径表,交换机按拥塞动态选下一跳出口,避免胖树单路径拥塞。NCCL 需 `NCCL_IB_AR=1` 配合。以太网用 ECMP 静态哈希,会因 hash 冲突导致单链路拥塞,这是 IB 性能稳定的关键之一。

### 8.4 IB 与以太网融合趋势

NVIDIA Spectrum-X 以太网平台用"以太网 + RoCE + Adaptive Routing + 主机端遥测"逼近 IB 性能,是"以太网派"对 IB 的正面竞争。GB200 NVL72 机柜内用 NVLink,跨机柜仍以 IB 或 Spectrum-X 二选一。详见 [[RoCE]] §8。

---
相关: [[RDMA与GPUDirect]] · [[RoCE]] · [[NVLink与NVSwitch]] · [[PCIe]] · [[NCCL核心机制]] · [[alpha-beta性能模型]] · [[集群拓扑感知placement]] · [[18-集群网络存储与数据系统]]
