# PCIe

> **所属章节**: [[GPU互连]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: PCI Express / PCIe / PCI-Express / 外设组件互连高速总线 / Gen4 / Gen5
> **难度**: 中（需懂 [[NVLink与NVSwitch]] / [[RDMA与GPUDirect]] / [[alpha-beta性能模型]] / [[拓扑感知]]）


## 1. 一句话定义

**PCIe（PCI Express，Peripheral Component Interconnect Express）** 是现代服务器里 CPU 与 GPU、网卡、NVMe SSD 等几乎所有外设之间通信的**通用高速串行总线标准**——由 PCI-SIG 维护,用"差分信号对 + 多 lane 并行 + 分代提速(Gen1~Gen6)"提供点对点带宽,是 GPU 接入系统的**默认最低标准互连**;在 GPU 集群里 PCIe 既负责 CPU↔GPU 控制面与数据搬运,也(在没 NVLink 时)兜底 GPU↔GPU 数据面,但带宽远低于 [[NVLink与NVSwitch]],大模型张量并行(TP)强依赖 NVLink,PCIe 只做"配角 + 跨机兜底"。**关键带宽数字(必背):Gen4 x16 ≈ 64 GB/s 双向(31.5 GB/s 单向),Gen5 x16 ≈ 128 GB/s 双向(63 GB/s 单向),约为 NVLink(A100 600 / H100 900 / Blackwell 1800 GB/s 双向)的 1/10~1/14。**

> [!note] 三句话定位
> - **是什么**:CPU-root port → PCIe switch → GPU/NIC/SSD 的树状点对点串行总线,分代(Gen1~6)每代翻倍带宽,每条 lane 是一对收一发差分对,xN 表示 N 条 lane 并行。
> - **为什么**:GPU 必须挂在 PCIe 上才能被 CPU 发现/配置/驱动、做 DMA、传权重与激活;没有 NVLink 的卡(GTX/RTX 消费卡、部分 T4/L4 推理卡)只能靠 PCIe 做 GPU↔GPU,即"PCIe P2P"。
> - **与 [[NVLink与NVSwitch]] 关系**:PCIe 是"通用总线(慢但什么都能挂)",NVLink 是"GPU 专用高速互连(快但只连 GPU)";TP/SP 通信走 NVLink,控制面、跨厂商设备、无 NVLink 的兜底走 PCIe。


## 2. 为什么需要它(动机与背景)

### 2.1 GPU 要被 CPU 管,就得有条"总线"

GPU 不是独立计算机,它**没有自己的 PCI 配置空间就无法被 OS/CUDA 枚举、映射显存到 CPU 地址空间、加载驱动**。这条"被发现 + 被驱动 + DMA 传权重"的总线历史上是 PCI,后来 AGP,现在统一是 PCIe。哪怕 H100 也有 PCIe 接口版(SXM 版才用 NVLink 全连接),PCIe 版 H100 仍需 PCIe 挂载才能被系统看见。

### 2.2 没 NVLink 时,GPU↔GPU 只能走 PCIe

消费卡(RTX 4090)与部分数据中心推理卡(T4/L4/A10)没有 NVLink,GPU↔GPU 通信用 **PCIe peer-to-peer (P2P) DMA**——同 PCIe switch 下的两张卡可以直接 DMA 对方显存,不绕 CPU 内存。但带宽只有 Gen4 x16 的 64 GB/s,远低于 NVLink 的 600~900 GB/s。

### 2.3 大模型训练为什么 PCIe 不够

TP(Tensor Parallel)的每层 all-reduce 通信量 ∝ 激活大小(batch × seq × hidden)。A100 上 hidden=12288 的 TP8 all-allgather,单卡收发约 96 MB/层 × 多层,要求**数百 GB/s 级带宽 + 微秒级延迟**;PCIe Gen4 x16 双向 64 GB/s、延迟 5~10 μs(远端跨 switch 还要加 switch 转发延迟),训练吞吐直接腰斩。所以 TP 组必须放同 NVSwitch domain(NVLink 全连接),PCIe 只做跨机兜底或 DP/PP(通信量小)。

> [!warning] 误区:以为"插 PCIe Gen5 就能训大模型"
> Gen5 x16 双向 128 GB/s 看着大,但:(1) 这是**理论上限**,实际 sustained 受 TLP、协议头、链路训练、switch 转发损耗打 80~90 折;(2) TP all-reduce 是**双向同步**的集合通信,要把带宽折半看单向;(3) NVLink 不仅带宽高 7~14 倍,延迟还低 3~5 倍(1~2 μs vs 5~10 μs)。结论:**Gen5 也救不了 TP,只对 PP/EP 级通信量小、或单卡推理 + 跨机 DP 有意义。**


## 3. 核心概念详解

### 3.1 分代(Gen)与 lane

- **Gen(代)** 决定单 lane 速率。每代**单向带宽翻倍**:Gen3≈1 GB/s/lane、Gen4≈2 GB/s/lane、Gen5≈4 GB/s/lane、Gen6≈8 GB/s/lane(PAM4)。
- **lane(通道)** 是一对差分发送 + 一对差分接收(共 4 对线,双向)。**xN** 表示 N 条 lane 并行,x16 是 GPU 标配,x8/x4 见于复用拆分。
- **编码开销**:Gen1/2 用 8b/10b(20% 开销),Gen3+ 用 128b/130b(1.54% 开销),Gen6 用 PAM4 + FLIT 1b/2b(开销回升但速率翻倍)。

### 3.2 拓扑:root port → switch → endpoint

典型 8 卡 GPU 服务器(DGX A100/H100 SXM 版不用 PCIe 连 GPU,这里指 PCIe 版或推理机型):

```
CPU(root complex)
 ├── root port ── PCIe switch ── GPU0, GPU1   (同 switch 下 P2P 快)
 ├── root port ── PCIe switch ── GPU2, GPU3
 └── root port ── PCIe switch ── NIC (ConnectX-6/7)
```

- **root complex / root port**:CPU 侧的 PCIe 根,发起配置读写、做地址翻译。
- **PCIe switch**:多端口片上 crossbar,把多个 endpoint 互连;**同 switch 下**的 GPU 间 P2P 走 switch 片上,延迟低(2~4 μs);**跨 switch** 要上行到 root port 再下行,延迟高、带宽竞争 CPU 根端口。
- **endpoint**:GPU、NIC、NVMe,每个有自己的 BAR(Base Address Register)映射内存窗口。
- **PCIe 桥/ACS**:Access Control Service 会**阻断 P2P**(出于安全,IOMMU 隔离),做 GPUDirect P2P 前要确认 ACS 关闭或支持 P2P 转发。

### 3.3 peer-to-peer (P2P) DMA

P2P 让同机 GPU0 显存直接 DMA 到 GPU1 显存,**不 copy 到 CPU 内存**:

```
不 P2P:  GPU0.vram → CPU.dram → GPU1.vram   (两次 PCIe 横穿 + CPU 内存拷贝, ~2x 延迟)
P2P:     GPU0.vram ────(PCIe switch 片上)──── GPU1.vram  (一次, 不碰 CPU)
```

CUDA 用 `cudaDeviceEnablePeerAccess()` + `cudaMemcpyPeer()` 或统一内存地址访问(UVA)做 P2P。NCCL 在没有 NVLink 时退化用 `cudaIpc` + P2P 通道(`P2P`/`P2P/CL` transport)。

### 3.4 拓扑感知:同 switch 快、跨 switch 慢

`nvidia-smi topo -m` 输出每对 GPU 间的链接类型(`NV#`=NVLink、`PIX`=同 PCIe switch、`PHB`=同 NUMA 但跨 PCIe bridge、`SYS`=跨 CPU/NUMA)。调度器([[拓扑感知]] placement)把通信频繁的 TP 组放同 switch 下,跨 switch 的卡做 DP/PP。

### 3.5 NPUs(网络处理单元,选读)

部分高端服务器在 PCIe switch 上挂 **BlueField DPU(Data Processing Unit)** 或专用网络加速核,做网络协议卸载、加密、存储虚拟化。GPU 集群里更常见的是 BlueField-3 DPU 做存储/网络卸载,与 GPU 之间也走 PCIe P2P(GPUDirect Storage 经 DPU)。**不展开**,详见 [[存储]]。

### 3.6 BAR 空间与 64-bit

GPU 的 BAR1 映射显存窗口给 CPU 访问。大显存(A100 80GB)需 **Resizable BAR / PCIe Above 4G Decoding**,否则 BAR1 只有 256MB 窗口、CPU 访问显存要分页映射,大块传输效率低。BIOS 必须开 Resizable BAR。


## 4. 数学原理 / 公式

### 4.1 单 lane 单向带宽推导

设 PCIe 第 $k$ 代线速为 $R_k$ GT/s(每秒 giga-transfer),编码效率 $\eta$($8\text{b}/10\text{b} \Rightarrow \eta=8/10=0.8$;$128\text{b}/130\text{b} \Rightarrow \eta=128/130\approx0.9846$),每 transfer 携带 1 byte payload 需 8 bit。则单 lane 单向**有效带宽**:

$$
B_{\text{lane/dir}}(k) = \frac{R_k \cdot \eta}{8} \quad [\text{GB/s}]
$$

例:Gen4 线速 $R_4=16$ GT/s,$\eta=128/130$:

$$
B_{\text{lane/dir}}(4) = \frac{16 \times 128/130}{8} = \frac{16 \times 0.9846}{8} = 1.969 \;\text{GB/s/lane/dir}
$$

### 4.2 x16 双向带宽

$$
B_{\text{x}N,\text{bi}}(k) = 2 \cdot N \cdot B_{\text{lane/dir}}(k) = 2 \cdot N \cdot \frac{R_k \eta}{8}
$$

代入 Gen4 $N=16$:$B = 2 \times 16 \times 1.969 = 63.0$ GB/s(常说 64 GB/s,取整)。
Gen5 $R_5=32$:$B_{\text{lane/dir}}=3.938$,$B_{\text{x16,bi}}=2\times16\times3.938=126$ GB/s(常说 128 GB/s)。

### 4.3 PCIe vs NVLink 通信时间对比(TP all-reduce)

设 TP all-reduce 每卡收发 $M$ 字节。通信时间(单向,粗略 $\alpha$-$\beta$ 模型,见 [[alpha-beta性能模型]]):

$$
T_{\text{PCIe}} = \alpha_{\text{PCIe}} + \frac{M}{B_{\text{PCIe,dir}}}, \quad
T_{\text{NVLink}} = \alpha_{\text{NVLink}} + \frac{M}{B_{\text{NVLink,dir}}}
$$

取 $M=96$ MB(典型一层 activation),A100 Gen4 PCIe $B_{\text{dir}}\approx31.5$ GB/s、$\alpha\approx6\mu s$;NVLink $B_{\text{dir}}=300$ GB/s(A100 单向)、$\alpha\approx1.5\mu s$:

$$
T_{\text{PCIe}} \approx 6 + \frac{96}{31.5}\times10^3\mu s \approx 6 + 3046 = 3052\,\mu s
$$
$$
T_{\text{NVLink}} \approx 1.5 + \frac{96}{300}\times10^3\mu s \approx 1.5 + 320 = 322\,\mu s
$$

**比值 ≈ 9.5×**:TP 走 PCIe 慢约 10 倍,这就是 NVLink 存在的理由。


## 5. 代码示例

### 5.1 看拓扑(GPU↔GPU 走啥)

```bash
# 输出 8 卡两两链接类型: NV# = NVLink, PIX = 同 PCIe switch, PHB = 同 NUMA 跨 bridge, SYS = 跨 NUMA
nvidia-smi topo -m

# 典型输出(DGX A100 SXM, 全 NVLink):
#         GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
#  GPU0     X   NV12  NV12  NV12  NV12  NV12  NV12  NV12
#  GPU1   NV12    X   NV12  NV12  NV12  NV12  NV12  NV12
#  ... (NV12 = 12 条 NVLink)
#
# PCIe 版机器(无 NVLink):
#  GPU0     X    PIX   PHB   SYS   ...
#  (PIX = 同 PCIe switch, 跨 switch 变 PHB/SYS)
```

### 5.2 看 PCIe 链路协商速率与宽度

```bash
# -q 详细查询, 看 PCIe Link gen 与 width (GPU 一般 x16)
nvidia-smi -q | grep -A3 "PCIe"

# 或直接 lspci
lspci -vvv -s $(lspci | grep -i nvidia | head -1 | cut -d' ' -f1) | grep -E "LnkCap|LnkSta"
# LnkCap: Speed 16GT/s, Width x16   <- Gen4 x16
# LnkSta: Speed 16GT/s, Width x16   <- 实际协商成功(否则降速降宽排查线缆/BIOS)
```

### 5.3 测 P2P/PCIe 带宽(NCCL 无 NVLink 退化场景)

```bash
# nccl-tests 测 all-reduce, 设 NCCL_P2P_DISABLE=1 强制走 PCIe 看 PCIe 兜底带宽
NCCL_P2P_DISABLE=1 NCCL_NET_DISABLE=1 ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8
# 对比不 disable 的 NVLink 带宽, 差距 ~10x

# nvidia-smi 看 GPU 利用率与 PCIe 吞吐(RX/TX 计数器)
nvidia-smi dmon -s p   # p = PCIe rx/tx throughput (MB/s)
```

### 5.4 CUDA P2P 拷贝(最小可跑)

```python
import torch
# 假设两块卡都支持 P2P (同机 + ACS 关)
a = torch.randn(1024*1024, device='cuda:0')   # GPU0
b = a.to('cuda:1')                              # 默认走 P2P 或 NVLink, 不经 CPU
# 显式启用 P2P 访问 (无 NVLink 时)
torch.cuda.cudart().cudaDeviceEnablePeerAccess(1)  # GPU0 开对 GPU1 的 peer 访问
# 之后 GPU0 可直接指针访问 GPU1 显存 (UVA), NCCL 也在底层用此通道
```


## 6. 与其他知识点的关系

- **上游(依赖)**: [[NVLink与NVSwitch]](对比基准)、[[RDMA与GPUDirect]](P2P 同理)、[[alpha-beta性能模型]](带宽延迟建模)、[[拓扑感知]](同 switch 优先)
- **下游(应用)**: [[NCCL核心机制]](无 NVLink 退化走 PCIe P2P transport)、[[Tensor Parallel]](TP 通信强依赖 NVLink,PCIe 兜底慢)、[[推理系统]](单卡推理 + 跨机 DP 用 PCIe 够)、[[存储]](NVMe over PCIe)
- **对比 / 易混**: [[NVLink与NVSwitch]](专用高速 vs 通用总线)、[[InfiniBand]]/[[RoCE]](跨机网络 vs 机内总线)、消费卡 NVLink Bridge(双卡 NVLink,数据中心 SXM 才全连接)


## 7. 常见误区与易错点

> [!warning] 误区1:把 x16 双向带宽当成单向
> 厂商与营销常标"Gen5 x16 = 128 GB/s"是**双向总和**,单向只有 63 GB/s。算 all-reduce(双向都用)时单向才决定单条消息延迟,别算翻倍。

> [!warning] 误区2:以为 PCIe switch 能当 NVSwitch 用
> PCIe switch 是**共享总线式 crossbar**,端口数少(常见 16~24 口),且 GPU↔GPU P2P 仍受 PCIe 速率限制;NVSwitch 是**NVLink 专用 crossbar**,带宽高 10 倍、端口全双工 NVLink。两者不可互换。

> [!warning] 误区3:跨 switch GPU 当同 switch 用
> `nvidia-smi topo -m` 显示 `SYS`(跨 NUMA/CPU)的对通信延迟最高(走 QPI/UPI 互联),TP 组放这种位置会严重掉速。必须看拓扑再放 TP 组。

> [!tip] 实践:训练前必查
> 1. `nvidia-smi topo -m` 确认 TP 组全是 `NV#`(NVLink)而非 `PIX/PHB/SYS`;
> 2. `lspci` 确认 GPU 协商到 Gen4/5 x16(降速 x4 是线缆/BIOS/ riser 问题);
> 3. 开 BIOS Resizable BAR + Above 4G Decoding,否则大显存映射低效。


## 8. 延伸细节

### 8.1 各代 PCIe 单 lane 速率全表(含 Gen6/7)

| 代 | 线速(GT/s) | 编码 | 单 lane 单向(GB/s) | x16 单向(GB/s) | x16 双向(GB/s) | 年份 |
|---|---|---|---|---|---|---|
| Gen1 | 2.5 | 8b/10b | 0.250 | 4.0 | 8.0 | 2003 |
| Gen2 | 5.0 | 8b/10b | 0.500 | 8.0 | 16.0 | 2007 |
| Gen3 | 8.0 | 128b/130b | 0.985 | 15.75 | 31.5 | 2010 |
| Gen4 | 16.0 | 128b/130b | 1.969 | 31.5 | 63.0 | 2017 |
| **Gen5** | 32.0 | 128b/130b | 3.938 | 63.0 | **126.0** | 2019 |
| Gen6 | 64.0 | PAM4+FLIT | 7.877 | 126.0 | 252.0 | 2022 |
| Gen7 | 128.0 | PAM4 | 15.75 | 252.0 | 504.0 | 待定 |

> 数据为 PCI-SIG 规范值,稳定可信。来源:PCI-SIG 官方规范 / 各代 PCIe datasheet。

### 8.2 PCIe 版 vs SXM 版 GPU 对比

| 项 | PCIe 版(如 H100 PCIe) | SXM 版(如 H100 SXM5) |
|---|---|---|
| GPU↔GPU 互连 | PCIe Gen5 x16(128 GB/s) | NVLink 4.0(900 GB/s) |
| 单卡功耗 | 350W | 700W |
| 8 卡全连接 | ❌(无 NVSwitch) | ✅(NVSwitch 全连接) |
| 适用 | 推理、单卡训练、跨机 DP | 大模型 TP 训练 |
| 显存带宽 | 同(HBM) | 同 |

### 8.3 PCIe 与 IOMMU / ACS

IOMMU(IOMMU/VT-d)做设备地址隔离,**默认会阻断 P2P**(P2P 绕过 CPU,破坏隔离)。做 GPUDirect RDMA/Storage 前要:(1) BIOS 关 ACS 或设为 trust;(2) 内核 `pcie_acs_override`;(3) 或用支持 P2P 转发的 switch。否则 NCCL 报 `P2P is not supported` 退化走 CPU bounce。

---
相关: [[NVLink与NVSwitch]] · [[RDMA与GPUDirect]] · [[alpha-beta性能模型]] · [[拓扑感知]] · [[NCCL核心机制]] · [[18-集群网络存储与数据系统]]
