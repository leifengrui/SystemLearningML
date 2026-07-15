# RoCE

> **所属章节**: [[网络]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: RoCE / RDMA over Converged Ethernet / RoCEv1 / RoCEv2 / 无损以太网
> **难度**: 中高(需懂 [[InfiniBand]] / [[RDMA与GPUDirect]] / [[NCCL核心机制]] / [[拓扑感知]])


## 1. 一句话定义

**RoCE(RDMA over Converged Ethernet,读作 "rocky")** 是把 **InfiniBand 的 RDMA 传输层(verbs/RDMA verbs)跑在以太网物理层上**的网络协议族——让通用的以太网交换机(比 IB 专用交换机便宜、多厂商、生态广)也能做零拷贝、绕过 CPU/OS 的 RDMA 通信。分两版:**RoCE v1**(链路层 / L2,只能同子网,需无损以太)、**RoCE v2**(UDP/IP / L3,可跨路由、跨子网,当前主流)。要让 RoCE 真正无损,必须在以太网上配 **PFC(Priority Flow Control)+ ECN 拥塞控制**(DCB / Data Center Bridging),否则丢包会让 RDMA 性能塌陷。RoCE 是"以太网派"对 [[InfiniBand]] 的兼容替代:同上层 API(NCCL/verbs 同一份代码)、不同物理层,**成本敏感的 AI 集群(RoCEv2 + Spectrum-X)越来越普及**。**关键带宽数字同以太网速率:400G / 800G 以太网口,单向 payload 约 50/100 GB/s。**

> [!note] 三句话定位
> - **是什么**:IB 的 verbs 传输层封装进以太网帧(RoCEv1 链路层 / RoCEv2 UDP+IP 网络层),NIC 硬件做 RDMA,以太网交换机做转发。
> - **为什么**:IB 交换机贵且准垄断(NVIDIA/Mellanox),以太网便宜、多厂商、与运维/管理网统一;RoCE 让通用以太网也能 RDMA,大幅降本。
> - **与 [[InfiniBand]] 关系**:上层 API 完全一致(NCCL/verbs 同代码),**物理层与管理面不同**——IB 用 LID + SM 集中算路,RoCE 用 IP + MAC + PFC + ECMP;IB 硬件无损,RoCE 需配 PFC 防丢包。


## 2. 为什么需要它(动机与背景)

### 2.1 IB 太贵、生态封闭

IB 交换机与网卡由 NVIDIA/Mellanox 准垄断,价格高、供货单一;以太网交换机多厂商(思科/Arista/Juniper/ NVIDIA Spectrum),价格竞争、运维与现存数据中心以太网统一。超大规模云厂(微软/Google/字节)更愿意用 RoCEv2 自建,避免被 NVIDIA 锁定。

### 2.2 以太网带宽已追上 IB

以太网 400G(2020)、800G(2022+)口速率已与 IB HDR/NDR 同量级,差距在**延迟与无损**而非带宽。只要把以太网做成无损(配 PFC/ECN),RoCE 性能可逼近 IB。

### 2.3 同 API 切换无成本

NCCL、MPI、verbs 程序**同一份代码**在 IB 与 RoCE 上跑(transport 都是 `NET`,内核 libibverbs 屏蔽差异),切换物理层无需改应用。这让 RoCE 成为 IB 的"无缝平替"候选。

> [!warning] 误区:以为 RoCE "就是更便宜的 IB"
> RoCE 上层 API 同 IB,但**物理层是无损以太网,需手动配 PFC/ECN/DCB**,配错(丢包)性能塌陷甚至不如普通 TCP。IB 是"开箱即无损"(硬件 credit 流控),RoCE 是"花功夫才能无损"。运维成本 RoCE 高于 IB。


## 3. 核心概念详解

### 3.1 RoCE v1 vs v2

| 维度 | RoCE v1 | RoCE v2 |
|---|---|---|
| 封装 | 以太网帧头 + IB 传输层 | 以太网 + IP + UDP + IB 传输层 |
| 网络层 | L2 链路层(同子网) | L3 可路由(跨子网) |
| 路由 | MAC 转发(同广播域) | IP 路由(标准 L3) |
| 可扩展 | 受同子网限制 | 可跨 L3 fabric |
| 当前主流 | 已淘汰 | ✅ 主流 |

RoCEv2 把 IB GRH(全局路由头)映射到 IP+UDP:UDP 目的端口 **4791** 是 RoCEv2 标识。v2 能跨 L3,这是大集群多 leaf-spine 多 fabric 必备。

### 3.2 必须配无损以太网(DCB)

以太网默认**丢包**(队列满 tail-drop),而 RDMA RC 连接丢一个包会触发 go-back-N 重传,性能雪崩。所以 RoCE 必须配 DCB(Data Center Bridging)使以太网无损:

- **PFC(Priority Flow Control,IEEE 802.1Qbb)**:给 RoCE 流量划分优先级(常见 priority 3),当入口队列接近满,向**上游**发 PAUSE 帧,上游暂停该优先级发送,**防止丢包**(用反压代替丢包)。等价 IB credit-based 流控的以太网实现。
- **ECN(Explicit Congestion Notification,IEEE 802.1Qau / IP ECN)**:交换机在拥塞时给报文打 ECN 标记,接收端 NIC 收到 ECN 后回 CNP(Congestion Notification Packet),发送端降速(类似 IB 的 CC)。控速避免持续拥塞压垮 PFC。
- **ETS(Enhanced Transmission Selection,802.1Qaz)**:带宽分配给各优先级,保证 RoCE 流量最小带宽。
- **DCBX(Data Center Bridging eXchange,802.1Qaz)**:交换机与 NIC 间自动协商 PFC/ETS 能力。

### 3.3 RoCE vs IB vs 普通以太网

| 维度 | InfiniBand | RoCEv2 | 普通以太网+TCP |
|---|---|---|---|
| 物理/链路 | IB 专用 | 以太网(配 DCB) | 以太网 |
| RDMA | ✅ 原生硬件 | ✅ verbs(同 IB) | ❌(TCP 软件拷贝) |
| 可靠性 | 硬件 credit 无损 | PFC/ECN 配无损 | TCP 重传 |
| 路由 | LID + SM | IP + MAC(标准 L3) | IP + MAC |
| 跨子网 | 子网内(SM) | ✅(L3 路由) | ✅ |
| 延迟 | ~0.5-1μs | ~1-2μs | ~10-50μs |
| 交换机 | 专用贵 | 通用便宜 | 通用最便宜 |
| 多厂商 | ❌(NVIDIA) | ✅ | ✅ |
| 运维 | SM 简单 | PFC/ECN 调复杂 | 最熟 |
| 丢包风险 | 极低 | 配错易丢包塌陷 | 常见 |

### 3.4 AI 集群 RoCE 普及与 NVIDIA Spectrum-X

成本敏感的 AI 集群越来越用 RoCEv2 + Adaptive Routing。NVIDIA 收购 Mellanox 后推出 **Spectrum-X Ethernet 平台**:以太网 ASIC + BlueField DPU + RoCEv2 + 自适应路由 + 主机端遥测(telemetry),性能逼近 IB,主打"以太网派"市场(GB200 NVL72 跨机柜、H100 SuperPOD 以太网变体)。Spectrum-4 800G 端口以太网 RoCE 是当前顶配。详见 [[InfiniBand]] §8.4。

### 3.5 PFC 优先级与配置

NVIDIA RoCE 网卡 ConnectX 默认 priority 3 给 RoCEv2,交换机侧配 PFC 在 priority 3 暂停。常见命令(Dell/Mellanox 交换机):
```
mlnx_qos -i eth1 --pfc 0,0,0,1,0,0,0,0  # priority 3 开 PFC
```
配错(比如忘开 PFC 或 priority 不匹配)会丢包,RoCE 性能从 100 GB/s 掉到几 GB/s。

### 3.6 与 RDMA/GPUDirect 的关系

RoCE 是 **RDMA 的一种以太网实现**(物理层用以太网,RDMA verbs 传输层同 IB)。GPUDirect RDMA 在 RoCE 上同样工作:NIC 直接 DMA GPU 显存(NIC↔GPU P2P),NCCL 跨机 all-reduce 走 RoCE GPUDirect RDMA。详见 [[RDMA与GPUDirect]]。


## 4. 数学原理 / 公式

### 4.1 带宽(同以太网速率)

RoCEv2 端口速率 $R$ Gb/s,扣开销(UDP/IP 头 + 以太网帧头 + 编码)约 $\eta\approx 0.94$:

$$
B_{\text{dir}} \approx \frac{R}{8}\eta \quad [\text{GB/s}]
$$

400G 以太网:$B\approx 50\times0.94\approx47$ GB/s 单向(实测 ~46-48)。800G:≈94~98。与 IB NDR/XDR 同物理速率下 payload 接近,差异在延迟与稳定性。

### 4.2 丢包对 RoCE 性能影响

RC(可靠连接)丢包触发 **go-back-N**:从丢包点重传其后所有未确认报文。设丢包率 $p$,窗口 $W$,一次重传代价 $\propto W$:

$$
T_{\text{eff}} \approx T_{\text{ideal}}\cdot(1 + p\cdot W)
$$

$p=0.1\%$、$W=64$ KB 大块传输:$T_{\text{eff}}\approx T_{\text{ideal}}\times1.064$,看似小;但**瞬时突发丢包(微突发)**会让某条流瞬间重传雪崩,实测 RoCE 在 PFC 配置错误时带宽从 47 GB/s 掉到 3-5 GB/s。这就是必须无损的量化依据。

### 4.3 PFC 反压延迟

入口队列阈值 $Q_{\text{th}}$ 触发 PAUSE,反压传播到上游,上游暂停 $T_{\text{pause}}\approx$ 链路 RTT + 处理延迟(μ秒级)。反压级联可能形成 **PFC 死锁**(环形暂停),需配 **PFC watchdog** 检测死锁并恢复。死锁是 RoCE 运维最难的问题之一。

### 4.4 ECN 降速

ECN 标记率 $r$ → 接收端回 CNP → 发送端按 AIRED/DCQCN 降速到 $(1-r\cdot\beta)$,$\beta$ 降速系数。DCQCN(数据中心量化拥塞通知)是 RoCE 主流 CC 算法,类似 IB CC 但调参更难。


## 5. 代码示例

### 5.1 确认网卡支持 RoCE 与版本

```bash
# 看 RoCE 网卡能力 (mlx5 系列 ConnectX)
ibv_devinfo            # 输出 mlx5_X (RoCE 网卡也走 ibverbs, Transport: InfiniBand / RoCE)
ibv_devinfo -d mlx5_0 -v
# RoCEv2 网卡 Transport: 通常是 "InfiniBand"(因为 verbs 抽象), 看 link_layer:
#   link_layer = Ethernet  -> RoCE
#   link_layer = InfiniBand -> 纯 IB
show_gids               # RoCE 用 GID (IPv6/IPv4) 寻址, 列出所有 GID (含 RoCEv2 的 GID)
```

### 5.2 交换机配 PFC / ECN(以 Mellanox 交换机为例)

```bash
# 进入交换机 CLI (ONIE/Mellanox Onyx):
mlnx_qos -i eth1 --pfc 0,0,0,1,0,0,0,0   # priority 3 开 PFC (对应 RoCE)
mlnx_qos -i eth1 --trust dscp            # 用 DSCP 信任(配合 RoCEv2 L3)
mcumgrctl set_priority_remap 26=3        # DSCP 26->priority 3 (RoCE 常用 DSCP 26)
# 开 ECN (WRED):
tune_ECN enable
# 验证:
mlnx_qos -i eth1                          # 看 PFC 配置
pfc_counter                               # 看 PFC PAUSE 次数 (持续增长=拥塞)
```

### 5.3 测 RoCE 带宽与丢包

```bash
# RoCE perftest (与 IB 同工具, 自动走 RoCEv2)
ib_write_bw -d mlx5_0 --report_gbits         # 服务端
ib_write_bw -d mlx5_0 <server_IP> --report_gbits   # 客户端, 用 IP (RoCEv2 走 IP)
# 期望 400G RoCE ~ 46-48 GB/s 单向

# 看 RoCE 丢包统计 (perftest 性能塌陷时先看)
ethtool -S eth1 | grep -i -E "drop|discard|pfc"   # rx_pfc_x_xoff 持续涨=反压正常; discard=真丢包
mlxup -i eth1                                     # NIC 健康 + PFC 状态
```

### 5.4 NCCL over RoCE

```bash
NCCL_DEBUG=INFO NCCL_NET=SOCKET NCCL_IB_DISABLE=0 \
  NCCL_IB_GID_INDEX=3 NCCL_IB_QPS_PER_GPI=2 \
  ./build/all_reduce_perf -g 8 ...
# 日志: "via NET/Socket"(若 IB_DISABLE) 或 "via NET/IB/RoCE"(RoCEv2)
# RoCEv2 需正确 GID_INDEX 选 RoCEv2 GID (非 IB GID)
```


## 6. 与其他知识点的关系

- **上游(依赖)**: [[InfiniBand]](verbs 传输层来源)、[[RDMA与GPUDirect]](RoCE 是 RDMA 的一种以太网实现)、[[PCIe]](NIC 经 PCIe)
- **下游(应用)**: [[NCCL核心机制]](NCCL NET over RoCE)、[[拓扑感知]](leaf-spine + ECMP)、[[集群拓扑感知placement]]
- **对比 / 易混**: [[InfiniBand]](原生专用 vs 以太网平替)、[[PCIe]](机内 vs 跨机)、NVLink-Network(机柜级专用)


## 7. 常见误区与易错点

> [!warning] 误区1:RoCE "开箱即用"
> 不配 PFC/ECN 的 RoCE 性能可能比普通 TCP 还差(丢包雪崩)。生产前必须:(1) 交换机与网卡都配 DCB;(2) priority 一致;(3) ECN+WRED;(4) DCQCN 调参;(5) PFC watchdog 防死锁。

> [!warning] 误区2:RoCEv2 与 RoCEv1 通用
> RoCEv1 已淘汰,跨 L3 大集群必须 RoCEv2。新购网卡全默认 v2,但混合老卡要确认。

> [!warning] 误区3:以为 RoCE 一定比 IB 便宜"还没性能损失"
> 同速率下 RoCE 单向带宽接近 IB,但**延迟高 1-3 倍、抖动大、有 PFC 死锁风险**,极端情况(微突发)吞吐波动远大于 IB。极致延迟场景(SuperPOD 万卡 all-reduce)仍倾向 IB。

> [!tip] 实践:RoCE 性能塌陷排查
> 1. perftest 跑不到线速 → 看交换机 `discard`(真丢包)与 `pfc_xoff`(反压是否生效);
> 2. 持续重传 → 检查 PFC priority 一致、DSCP->priority 映射;
> 3. 随机卡死 → 疑 PFC 死锁,看 watchdog;
> 4. NCCL 报 timeout → `NCCL_IB_GID_INDEX` 选错 GID(选了 IB GID 而非 RoCEv2 GID)。

> [!note] 补充:Adaptive Routing 是 RoCE 接近 IB 的关键
> 静态 ECMP 按 hash 选路径,大流量易 hash 冲突导致单链路拥塞触发 PFC。Spectrum-X / Arista 的 Adaptive Routing 按队列长度动态选下一跳,避免拥塞堆积,这是 RoCE 性能逼近 IB 的关键。NCCL 需配合 `NCCL_IB_AR=1`(IB 也用)。


## 8. 延伸细节

### 8.1 以太网速率演进

| 代 | 端口速率 | 单向 payload(GB/s) | 年份 |
|---|---|---|---|
| 100G | 100 | ~12 | 2016 |
| 200G | 200 | ~25 | 2018 |
| 400G | 400 | ~47 | 2020 |
| 800G | 800 | ~94 | 2022+ |
| 1.6T | 1600 | ~190 | 2025+ |

> RoCEv2 在 400G/800G 上与 IB NDR/XDR 同物理速率同量级 payload,差异在无损保证与延迟。

### 8.2 DCQCN 拥塞控制

DCQCN(Data Center Quantitative Congestion Notification)是 RoCE 主流 CC 算法:Mellanox/微软提出,ECN→CNP→发送端按 alpha 衰减降速。比 IB CC 调参复杂,需设 `np/alpha/...` 等参数,且对拓扑敏感。深度调参是 RoCE 运维专业活,详见 NVIDIA/Mellanox 白皮书。

### 8.3 PFC 死锁与 watchdog

PFC 环形反压形成死锁(多个流互相暂停),队列全冻结。交换机配 **PFC watchdog**(周期性主动发恢复帧)打破死锁,但 watchdog 触发会导致短暂丢包。拓扑设计应避免环路(leaf-spine 无环)减少死锁概率。

### 8.4 RoCE 与 [[RDMA与GPUDirect]] 协同

GPUDirect RDMA 在 RoCE 上同样工作:ConnectX 网卡通过 PCIe P2P 直接 DMA GPU 显存(NIC↔GPU 不经 CPU 内存)。NCCL 跨机 all-reduce 走 RoCE GPUDirect RDMA,路径 GPU0→NIC0→…→NIC1→GPU1,全链路零拷贝(除 PCIe/以太网硬件转发)。这是 RoCE 大集群训练能打满带宽的底层。

### 8.5 选型建议(粗略)

- 追求极致延迟与稳态 + 不差钱 + 单厂商接受度 → **IB**(开箱无损)
- 成本敏感 + 多厂商 + 自运维能力强 + 与现网以太网统一 → **RoCEv2 + Spectrum-X / Arista**
- 单机/机柜内 → **NVLink**(根本不走网络)
- 机内无 NVLink 兜底 → [[PCIe]] P2P

---
相关: [[InfiniBand]] · [[RDMA与GPUDirect]] · [[NVLink与NVSwitch]] · [[PCIe]] · [[NCCL核心机制]] · [[拓扑感知]] · [[集群拓扑感知placement]] · [[18-集群网络存储与数据系统]]
