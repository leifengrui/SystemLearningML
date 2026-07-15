# RDMA与GPUDirect

> **所属章节**: [[网络]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: RDMA / Remote Direct Memory Access / GPUDirect / GPUDirect RDMA / GPUDirect Storage / NVLink-NT / peer-2-peer / P2P
> **难度**: 中高(需懂 [[InfiniBand]] / [[RoCE]] / [[PCIe]] / [[NVLink与NVSwitch]] / [[NCCL核心机制]] / [[all-reduce]])


## 1. 一句话定义

**RDMA(Remote Direct Memory Access,远程直接内存访问)** 是一种让一端网卡(NIC)硬件**直接读写对端内存/显存、不经对端 CPU 与 OS、零拷贝**的通信范式;**GPUDirect** 是 NVIDIA 在 RDMA 上的 GPU 专用扩展家族,核心是 **GPUDirect RDMA**(NIC 与 GPU 之间做 PCIe P2P DMA,跨机 NCCL all-reduce 时数据直接 GPU↔NIC,不绕 CPU 内存四次拷贝)。GPUDirect 家族还有 **GPUDirect Storage**(GPU↔NVMe 直连)、**GPUDirect P2P**(同机 GPU↔GPU 直连)与 **NVLink-NT**(NVLink 跨节点网络化扩展)。**为什么需要它**:不用 RDMA 跨机 all-reduce 的路径是 GPU0→CPU0 内存→NIC0→…→NIC1→CPU1 内存→GPU1 四次拷贝 + 两次 CPU 参与,延迟大、带宽吃 CPU 内存带宽;GPUDirect RDMA 把它压成 GPU0→NIC0→…→NIC1→GPU1 两次 PCIe/DMA 转发、零 CPU 参与。**关键关系**:GPUDirect RDMA 是 [[NCCL核心机制]] 跨机通信的底层(NCCL `NET` transport 默认走它),也是 [[InfiniBand]]/[[RoCE]] 上跑 GPU 集合通信的物理基础。

> [!note] 三句话定位
> - **是什么**:RDMA = 网卡硬件直接读写对端内存(零拷贝、绕 CPU/OS);GPUDirect = RDMA 的 GPU 版本,NIC 直 DMA GPU 显存(NIC↔GPU PCIe P2P),跨机 all-reduce 不绕 CPU 内存。
> - **为什么**:传统 socket 跨机 all-reduce 要 GPU→CPU 内存→NIC→…→NIC→CPU 内存→GPU 四次拷贝,延迟大、占 CPU 内存带宽;GPUDirect RDMA 把路径压成 GPU→NIC→…→NIC→GPU,零拷贝、零 CPU 参与、低延迟。
> - **与 [[InfiniBand]]/[[RoCE]] 关系**:RDMA 是通信"范式",IB/RoCE 是它的两种物理实现(IB 原生 / 以太网兼容);GPUDirect RDMA 在两者上都跑,是 NCCL 跨机通信底层。


## 2. 为什么需要它(动机与背景)

### 2.1 传统 socket 跨机通信的"四次拷贝"

不用 RDMA,跨机 GPU all-reduce 一条消息的路径:

```
GPU0.vram 
  → (cudaMemcpy) CPU0.dram   [拷贝1: GPU→CPU内存]
  → (内核socket send) 内核send buffer  [拷贝2: user→kernel]
  → NIC0 (DMA) 
  → ... 网络转发 ...
  → NIC1 (DMA to) CPU1.dram 内核recv buffer  [拷贝3: kernel→user]
  → (cudaMemcpy) GPU1.vram  [拷贝4: CPU内存→GPU]
```

四个拷贝 + CPU 两次系统调用,延迟 10~50μs,CPU 内存带宽也被吃(DDR4/5 ~100-200 GB/s,大集群 all-reduce 持续走 CPU 内存会成瓶颈)。CPU 还要参与数据搬运,占核。

### 2.2 RDMA 砍掉 CPU/OS

RDMA 让 NIC 硬件直接用 `ibv_rdma_write/read` verbs 把数据从本地内存 DMA 到对端内存(或反向),**不进 CPU、不进 OS 内核、零拷贝**(用户态 buffer 直接被 NIC DMA)。

### 2.3 GPUDirect RDMA 砍掉 CPU 内存这一跳

普通 RDMA 还要"GPU 显存 → CPU 内存"再 NIC。**GPUDirect RDMA** 让 NIC 直接 DMA GPU 显存:NIC 与 GPU 在同一 PCIe switch 下做 P2P DMA(或 Grace-Hopper 用 NVLink-C2C),省掉"GPU↔CPU 内存"两跳。路径压成:

```
GPU0.vram → (PCIe P2P DMA) NIC0 → ... → NIC1 → (PCIe P2P DMA) GPU1.vram
```

两次 DMA 转发,零 CPU 参与,延迟 ~1-3μs(看网络速率),带宽打满 NIC 线速。这是万卡训练能跑起来的物理前提。

> [!warning] 误区:以为 GPUDirect "把 GPU 直连网络线缆"
> GPUDirect 不改物理拓扑——NIC 仍挂 PCIe(或 GH200 上挂 NVLink-C2C),GPU 也仍挂 PCIe。GPUDirect 是让 NIC 与 GPU **走 PCIe switch 片上做 P2P DMA**,省"绕 CPU 内存"这一跳。前提:NIC 与 GPU 同 PCIe switch(拓扑感知)且 ACS 关闭。不在同一 switch / ACS 开着,会退化成走 CPU 内存 bounce buffer,失去 GPUDirect 优势。


## 3. 核心概念详解

### 3.1 RDMA verbs(动词 API)

RDMA 通信用 verbs(用户态库 libibverbs),核心三类原语:

- **send/recv**:语义像 socket send/recv,但 NIC 直接 DMA buffer,零拷贝。需接收端预先 post recv buffer。
- **RDMA write**:发送端直接写对端内存,**不需对端 CPU 参与**(对端 NIC 自动接收,应用以后才 poll)。一端发起,单边。
- **RDMA read**:发送端直接读对端内存,单边,不需对端参与。

GPUDirect RDMA 让 buffer 可以是 GPU 显存指针(NIC 直接 DMA 显存)。NCCL 的 ring/tree 跨机跳用 RC 连接 + RDMA write/read 实现 zero-copy。

### 3.2 GPUDirect 家族

| 名称 | 做什么 | 典型用途 |
|---|---|---|
| **GPUDirect P2P** | 同机 GPU↔GPU 直连(PCIe P2P 或 NVLink) | 机内 NCCL、单机 all-reduce |
| **GPUDirect RDMA** | NIC↔GPU 直接 DMA(跨机零拷贝) | NCCL 跨机 all-reduce(NET transport) |
| **GPUDirect Storage** | NVMe SSD↔GPU 直接 DMA(经 DPU/NVMe-oF) | 训练数据加载、checkpoint 直接进显存 |
| **Unified Virtual Memory** | CPU/GPU 统一地址(软件层) | 程序员便利(慢,page fault) |
| **NVLink-NT(NVLink-Network)** | NVLink 跨节点网络化扩展 | NVL72 跨机柜全连接 |

### 3.3 GPUDirect RDMA 路径与拓扑前提

```
GPU0 --[PCIe P2P]-- NIC0 ====网络==== NIC1 --[PCIe P2P]-- GPU1
```

- **拓扑前提**:NIC 与目标 GPU 在**同一 PCIe switch** 下,P2P 才高效(跨 switch 走 root port 慢,且 ACS 可能阻断)。
- **ACS 关闭**:见 [[PCIe]] §3.2。NCCL 报 `GPUDirect RDMA is not supported` 多半是 ACS 开着或拓扑不通。
- **Grace-Hopper 特例**:GH200 的 NIC(ConnectX-7)与 GPU 通过 NVLink-C2C 互联,NIC↔GPU 不走 PCIe 走 NVLink-C2C(900 GB/s),GPUDirect 更快。

### 3.4 GPUDirect Storage

存储直接进显存:NVMe SSD → DPU/网卡 → 直接 DMA GPU 显存,省"SSD→CPU内存→GPU"。训练数据加载(从 NVMe/共享 FS 读 batch)、checkpoint 保存/加载(几百 GB 模型权重直接 GPU↔SSD)都受益。NVIDIA Magnum IO + GPUDirect Storage 是典型栈。详见 [[存储]] / [[数据加工]]。

### 3.5 NVLink-NT(NVLink-Network)

NVLink-NT 是 NVLink 的**跨节点网络化扩展**:把 NVLink 链路经 NVLink Switch tray 扩展到机柜级(NVL72 = 72 卡单域),甚至跨机柜(NVL576)。NVLink-NT 让 NVLink "出机箱",跨机不再必然走 IB/RoCE。Blackwell GB200 NVL72 是首个大规模产品。具体带宽随拓扑变化(NVL72 内每卡 14 NVLink 留给本地 + 4 留给跨机?**待核实**,以 NVIDIA NVL72 spec 为准)。

### 3.6 peer-2-peer(P2P)与 GPUDirect P2P

"peer-2-peer"在 GPU 语境指同机 GPU 间 P2P DMA(经 PCIe switch 或 NVLink),是机内 GPU↔GPU 通信的硬件基础。有 NVLink 走 NVLink(快),无 NVLink 走 PCIe P2P(慢但能用)。NCCL `P2P` transport 用它。见 [[PCIe]] §3.3。

### 3.7 与 NCCL 的关系(关键)

NCCL 跨机 `NET` transport 默认走 GPUDirect RDMA:
- 探测拓扑,NIC 与 GPU 同 PCIe switch → 启用 GPUDirect RDMA。
- 建 RC QP(每跨机 pair 一条),NCCL ring 上跨机跳用 RDMA write/read 直读对端 GPU 显存。
- 失败(ACS/拓扑不通)退化走 CPU bounce buffer(NIC↔CPU 内存↔GPU),带宽掉一半。

`NCCL_DEBUG=INFO` 日志看 `NET/IB`/`NET/IB/RoCE` + 是否 `via GPU P2P` 即 GPUDirect RDMA 生效。详见 [[NCCL核心机制]] / [[NCCL backend]]。


## 4. 数学原理 / 公式

### 4.1 路径拷贝数对比

设单条消息 $M$ 字节,链路带宽 $B_{\text{PCIe}}$(NIC↔GPU PCIe)、$B_{\text{net}}$(网络)、CPU 内存带宽 $B_{\text{DRAM}}$、各跳延迟 $\alpha_i$。

**socket(四次拷贝 + CPU 参与)**:

$$
T_{\text{sock}} \approx \sum_i \alpha_i + \frac{M}{B_{\text{PCIe}}} + \frac{M}{B_{\text{DRAM}}} + \frac{M}{B_{\text{net}}} + \frac{M}{B_{\text{DRAM}}} + \frac{M}{B_{\text{PCIe}}}
$$

**GPUDirect RDMA(两次 P2P DMA,零 CPU)**:

$$
T_{\text{GDRDMA}} \approx \alpha_{\text{PCIe}} + \alpha_{\text{net}} + \frac{M}{B_{\text{PCIe}}} + \frac{M}{B_{\text{net}}} + \frac{M}{B_{\text{PCIe}}}
$$

省掉两跳 DRAM 带宽(每跳 $M/B_{\text{DRAM}}\approx M/150$ GB/s)与两次系统调用延迟(各 ~5-10μs)。对 $M=100$ MB、NDR 网络:

$$
\text{省的延迟} \approx 2\alpha_{\text{sys}}\approx 10-20\mu s,\quad \text{省的带宽开销} \approx \frac{2M}{B_{\text{DRAM}}} = \frac{200}{150}\times10^3\mu s \approx 1333\mu s
$$

GPUDirect RDMA 在大消息场景省 ms 级,在小消息场景省 10-20μs 系统调用。是跨机集合通信低延迟的关键。

### 4.2 PCIe P2P 与 GPUDirect 的带宽上限

单 GPU↔NIC 路径带宽上限 = $\min(B_{\text{PCIe}}, B_{\text{net}})$。H100 PCIe Gen5 x16 单向 63 GB/s,NIC NDR 单向 50 GB/s,$\min=50$,GPUDirect 受网络限制;若用 800G RoCE 单向 ~94 GB/s,则受 PCIe 限制(63)。所以**高带宽网卡 + PCIe Gen5 必须配套**,否则 PCIe 是瓶颈。GH200 用 NVLink-C2C 接 NIC(900 GB/s),突破 PCIe 瓶颈,这是 GH200 的关键优势。

### 4.3 集合通信跨机跳延迟(all-reduce ring)

$N$ 卡跨机 all-reduce ring,跨机跳数 $H$,单跳 GPUDirect RDMA 延迟 $\alpha_{\text{GDRDMA}}$,消息 $M$:

$$
T \approx H\left(\alpha_{\text{GDRDMA}} + \frac{M}{B_{\text{net}}}\right)
$$

GPUDirect 把 $\alpha$ 从 socket 的 ~20μs 压到 ~2-3μs,且消掉 CPU 内存带宽瓶颈。万卡 ring $H=\sqrt{N}$,$N=10^4$ 时 $H=100$,延迟省 $100\times 18 \approx 1.8$ ms/步,全天训练累计是小时级。**精确建模见 [[alpha-beta性能模型]]。**


## 5. 代码示例

### 5.1 确认 GPUDirect RDMA 是否生效

```bash
# 1. 网卡与 GPU 是否同 PCIe switch (看 PCI BDF 拓扑)
nvidia-smi topo -m            # NIC 不直接出现, 看 GPU 对的链接
lspci -tv | grep -i mellanox  # NIC 的 PCIe 路径, 与 GPU 比较 root

# 2. 确认 NIC 支持 GPUDirect RDMA (看 peer_memory 支持)
ls /sys/kernel/mm/peermem/    # 存在 peermem 模块 = 支持 GPUDirect RDMA 注册
# 或
cat /proc/driver/nvidia/gpus/*/information | grep -i p2p
modinfo nvidia-peermem        # 内核模块 nvidia-peermem (旧名 nvidia_p2p_persistent)

# 3. NCCL debug 看是否用 GPUDirect RDMA
NCCL_DEBUG=INFO ./all_reduce_perf -g 8 2>&1 | grep -E "NET|P2P|GPU Direct|via"
# 期望: "Channel ... via NET/IB/... " + "using GPU Direct RDMA"
# 失败退化: 日志报 "P2P is not supported" / "falling back to copy"
```

### 5.2 用 nvidia-peer-memory 验证 NIC↔GPU 直 DMA

```bash
# 启用 nvidia-peermem (内核模块注册 GPU 内存给 NIC DMA)
modprobe nvidia-peermem
lsmod | grep peermem          # 确认加载

# 跑 RDMA perftest, --use_cuda 让 buffer 在 GPU 显存 (验证 GPUDirect RDMA)
# 服务端:
ib_write_bw -d mlx5_0 --use_cuda=0
# 客户端:
ib_write_bw -d mlx5_0 <server_IP> --use_cuda=0 --report_gbits
# 若 bandwidth 接近线速 = GPUDirect RDMA 生效 (数据在 GPU 显存, NIC 直接 DMA)
# 若带宽掉半 = 退化走 CPU bounce (peermem 没加载/ACS 开)
```

### 5.3 Python NCCL 跨机验证(PyTorch)

```python
import torch
import torch.distributed as dist
dist.init_process_group("nccl", rank=0, world_size=8)
t = torch.randn(1<<24, device='cuda')   # 64MB 在 GPU0 显存
dist.all_reduce(t)                       # 跨机 all-reduce
# 设置 NCCL_DEBUG=INFO 跑, 看日志确认 "via NET/IB" + "GPU Direct RDMA"
```

```bash
NCCL_DEBUG=INFO NCCL_NET=IB NCCL_IB_HCA=mlx5_0 \
  NCCL_IB_PCI_RELAXED_ORDERING=1 \
  python train.py
```


## 6. 与其他知识点的关系

- **上游(依赖)**: [[InfiniBand]](RDMA 原生实现)、[[RoCE]](RDMA 以太网实现)、[[PCIe]](NIC↔GPU P2P 通道)、[[NVLink与NVSwitch]](机内 P2P/Grace-Hopper NIC↔GPU)
- **下游(应用)**: [[NCCL核心机制]](GPUDirect RDMA 是 NCCL NET transport 底层)、[[NCCL backend]]、[[all-reduce]](跨机 all-reduce 走 GPUDirect RDMA)、[[存储]](GPUDirect Storage)、[[数据加工]](训练数据加载)
- **对比 / 易混**: [[NVLink与NVSwitch]](机内专用 vs 跨机通用 RDMA)、[[PCIe]](P2P 物理层)、socket 通信(对比基准)


## 7. 常见误区与易错点

> [!warning] 误区1:以为 GPUDirect RDMA 自动启用
> 需满足:(1) NIC 与 GPU 同 PCIe switch;(2) 加载 `nvidia-peermem` 内核模块;(3) BIOS 关 ACS;(4) NCCL `NET` transport 选 IB/RoCE;(5) Resizable BAR 开。任一缺失退化走 CPU bounce,带宽掉半。务必 `NCCL_DEBUG=INFO` 看 "using GPU Direct RDMA"。

> [!warning] 误区2:RDMA 一定零拷贝
> 只有 buffer 是 NIC 直接 DMA 的内存(注册过的 MR / GPU 显存经 peermem)才零拷贝。用普通 malloc 内存 + send/recv,仍有 NIC↔buffer 一次 DMA 但仍 user 态零拷贝;但 GPU 显存不经 peermem 则要拷到 CPU 内存。

> [!warning] 误区3:把 NVLink-NT 当成 IB 替代
> NVLink-NT 只在 NVL72/NVL576 等 NVIDIA 专用整机柜产品内,且仍限于单 NVLink domain(72/576 卡)。普通跨机集群仍走 IB/RoCE GPUDirect RDMA。NVLink-NT 是机柜级专用,不是通用跨机网络。

> [!tip] 实践:跨机 all-reduce 调优清单
> 1. `nvidia-peermem` 已加载(`lsmod`);
> 2. NIC 与 GPU 同 PCIe switch(`lspci -tv` 看路径);
> 3. ACS 关 / Resizable BAR 开;
> 4. NCCL `NCCL_IB_PCI_RELAXED_ORDERING=1` + `NCCL_IB_HCA=mlx5_0`;
> 5. `NCCL_DEBUG=INFO` 日志确认 "GPU Direct RDMA";
> 6. perftest `--use_cuda` 跑接近线速 → 再跑 NCCL all-reduce 对比。
> 不达标先解决 GPUDirect 再调 NCCL,否则在错误基础上调参。

> [!note] 补充:为什么 GH200 让 GPUDirect 更强
> GH200 的 NIC(ConnectX-7)与 GPU 通过 **NVLink-C2C(900 GB/s)** 互联,而非 PCIe(63 GB/s)。NIC↔GPU 路径带宽从 PCIe 的 63 GB/s 跳到 NVLink-C2C 的 450 GB/s 单向,GPUDirect RDMA 受网络(50 GB/s NDR)限制而非 PCIe 限制,跨机带宽利用率更高。这是 GH200 训练吞吐提升的关键之一。


## 8. 延伸细节

### 8.1 RDMA 两种连接类型

- **RC(Reliable Connection)**:1 对 1 可靠,GPUDirect RDMA/NCCL 默认。每对 QP 状态在两端 NIC,丢包重传 NIC 硬件处理。
- **UD/UC**:不可靠,少用。大规模集群 RC 的 QP 数量随节点数平方增长(每对一 QP),需 QP 规模优化(SRQ / DC target)。

### 8.2 PCIe Relaxed Ordering

GPUDirect RDMA 大吞吐时,PCIe 写顺序约束会让 NIC 与 GPU DMA 互相阻塞。`NCCL_IB_PCI_RELAXED_ORDERING=1`(配合 BIOS/CPU 支持)放开 PCIe 顺序约束,大幅提吞吐。Intel ICE/Sapphire Rapids + ConnectX-6/7 都支持,是 NDR/RoCE 400G 必开项。

### 8.3 GPUDirect Storage 落地

训练数据(图片/tokenized shard)从共享 FS / NVMe 直接 DMA 到 GPU 显存,省"FS→CPU内存→GPU"一跳。Magnum IO + cuFile API + GDS 是典型栈。对 I/O 受限训练(图像/多模态)提速明显。详见 [[数据加工]] / [[存储]]。

### 8.4 RDMA 与 [[Ray与分布式调度]] 的关系

Ray 的 actor/worker 间通信默认用 TCP/Arrow Plasma,不走 RDMA。但 Ray 在 GPU 集群上跑训练时,真正的 GPU 集合通信(梯度同步)仍走 NCCL over RDMA,Ray 只管调度与数据搬。Ray Data 可用 Arrow/Parquet + GDS 读存储。详见 [[Ray与分布式调度]] / [[数据加工]]。

### 8.5 一句话总结五者关系

> **RDMA** 是范式(NIC 直接读写对端内存),**[[InfiniBand]] / [[RoCE]]** 是它的两种物理实现,**GPUDirect RDMA** 是 RDMA 的 GPU 版(NIC 直 DMA 显存),**[[NVLink与NVSwitch]]** 是机内 GPU 专用互连(替代跨机 RDMA 的角色),**[[NCCL核心机制]]** 在机内用 NVLink、跨机用 GPUDirect RDMA,二者协同构成大模型训练的通信底层。PCIe Gen5 / NVLink-C2C 是 NIC↔GPU 这一跳的物理通道。

---
相关: [[InfiniBand]] · [[RoCE]] · [[PCIe]] · [[NVLink与NVSwitch]] · [[NCCL核心机制]] · [[NCCL backend]] · [[all-reduce]] · [[alpha-beta性能模型]] · [[存储]] · [[数据加工]] · [[Ray与分布式调度]] · [[memory bandwidth]] · [[18-集群网络存储与数据系统]]
