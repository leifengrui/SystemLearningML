# NVLink与NVSwitch

> **所属章节**: [[GPU互连]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: NVLink / NVSwitch / NVLink-C2C / NVLink-Network / NVL72
> **难度**: 中高(需懂 [[PCIe]] / [[Tensor Parallel]] / [[all-reduce]] / [[NCCL核心机制]] / [[拓扑感知]])


## 1. 一句话定义

**NVLink** 是 NVIDIA **GPU 间的高速专用点对点互连**(及 Grace CPU↔Hopper GPU 的 NVLink-C2C 变体),带宽逐代翻倍式增长——P100(1.0,160 GB/s 双向)、V100(2.0,300)、A100(3.0,600)、H100(4.0,900)、Blackwell B200(5.0,1800 GB/s = 1.8 TB/s 双向),**单向带宽约是 PCIe Gen5 x16 的 7~14 倍、延迟约 1/3**;**NVSwitch** 是 GPU 之间的片上 crossbar 交换芯片,让单机 8 卡(或多机 NVL72)**两两 NVLink 全连接**(否则 NVLink 是点对点,只能连有限邻居)。大模型训练的**张量并行(TP)强依赖 NVLink+NVSwitch**:TP 每层 all-reduce/all-gather 通信量 ∝ 激活,只有 NVLink 的几百 GB/s + 微秒延迟才能把通信开销压到可接受。**关键带宽数字(必背):NVLink 单向 = 各代总带宽/2:P100 80 / V100 150 / A100 300 / H100 450 / B200 900 GB/s 单向。**

> [!note] 三句话定位
> - **是什么**:NVIDIA 私有的高速差分串行 GPU 互连(2-4 对线/lane,单 link 50~100 GB/s 双向)+ NVSwitch 片上 crossbar,实现单机/多机全连接 GPU 池。
> - **为什么**:PCIe 太慢(64 GB/s)且非 GPU 专用,TP/SP/EP 集合通信要几百 GB/s + 微秒延迟,只有 NVLink 满足;NVSwitch 解决"NVLink 是点对点、单 GPU 引脚有限无法两两直连"的问题。
> - **与 [[PCIe]] 关系**:NVLink 只连 GPU(专用、快、贵、NVIDIA 闭源),PCIe 连一切外设(通用、慢、便宜、开放);两者共存,NVLink 跑 GPU 数据面,PCIe 跑控制面与无 NVLink 的设备。


## 2. 为什么需要它(动机与背景)

### 2.1 PCIe 是大模型训练的瓶颈

DGX-1(P100 时代)用 PCIe 连 8 卡,TP all-reduce 在 PCIe Gen3(32 GB/s 双向)上,通信占训练时间 40~60%。NVLink 1.0 把 P100 间带宽提到 160 GB/s,通信占比降到 10% 以内。详见 [[PCIe]] §4.3 的 9.5× 通信时间对比。

### 2.2 GPU 引脚有限,无法两两直连

NVLink 是**点对点**串行链路,单 GPU 的 NVLink 引脚数有限(P100 4 link、V100 6、A100 12、H100 18)。8 卡要"任意两卡直连"需 $C_8^2 = 28$ 条逻辑链路,远超单 GPU 引脚。**NVSwitch**(片上 crossbar)用"每 GPU 把自己的 NVLink 全打到 NVSwitch,NVSwitch 内部 crossbar 两两交换"实现全连接——DGX A100 用 6 颗 NVSwitch 芯片,每 GPU 12 link 分散打到 6 颗 NVSwitch,任意两 GPU 间聚合带宽 12 link(600 GB/s)。

### 2.3 TP 强依赖 NVLink

[[Tensor Parallel]] 把单层权重切到多卡,前向/反向每层都要 all-reduce(激活同形状)。A100 hidden=12288、batch×seq 适中时单层 all-reduce 收发约 96 MB,80 层 × 3 阶段(前向+反向 input grad + weight grad)通信极频繁。**只有 NVLink 的 300 GB/s 单向(A100)+ 1~2 μs 延迟**才能把通信/计算比压到 <20%。同 NVSwitch domain 内全连接是 TP8 的物理前提。

> [!warning] 误区:以为 NVLink 让 8 卡"显存合并"
> NVLink 让**数据传输快**,但**不改变每个 GPU 的显存是独立地址空间**这一事实。要让上层"看到一块大显存"需 CUDA Unified Memory + page migration(慢,page fault)或框架层手动 placement(NVLink 让手动通信够快)。NVLink ≠ NUMA 自动合并显存。


## 3. 核心概念详解

### 3.1 NVLink 各代规格全表(必背)

| NVLink 代 | GPU 架构 | 典型卡 | link 数 | 单 link 双向带宽 | **单 GPU 总双向带宽** | 单向带宽 | 年份 |
|---|---|---|---|---|---|---|---|
| 1.0 | Pascal | P100 | 4 | 40 GB/s | **160 GB/s** | 80 | 2016 |
| 2.0 | Volta | V100 | 6 | 50 GB/s | **300 GB/s** | 150 | 2017 |
| 3.0 | Ampere | A100 | 12 | 50 GB/s | **600 GB/s** | 300 | 2020 |
| 4.0 | Hopper | H100 | 18 | 50 GB/s | **900 GB/s** | 450 | 2022 |
| 5.0 | Blackwell | B200 | 18 | 100 GB/s | **1.8 TB/s** | 900 | 2024 |

> 单 link 单向带宽:1.0=20、2.0/3.0/4.0=25、5.0=50 GB/s;双向×2。表内"单 GPU 总双向带宽"是 NVIDIA datasheet 官方标称值(每代 GPU 产品页与 whitepaper 均载),稳定可信。单 link 规格(50 GB/s)依据 NVIDIA NVLink whitepaper / Ampere/Hopper 架构 whitepaper。

> [!note] 容易记混的点
> NVLink 2.0/3.0/4.0 **单 link 双向都是 50 GB/s**(单 link 没提速),靠**加 link 数**翻总带宽(6→12→18);只有 NVLink 5.0 把单 link 提到 100 GB/s,18 link 才到 1.8 TB/s。所以"代际翻倍"主要靠**加 link**,5.0 才靠**单 link 翻倍**。

### 3.2 NVSwitch:NVLink 的 crossbar

- **DGX-1(V100)** 用混合拓扑:6 NVLink × 部分 GPU 两两直连,**不是全连接**(每对 GPU 只有 2-6 NVLink,带宽不均)。
- **DGX A100/H100** 引入 **NVSwitch 芯片**:8 GPU 各 12(A100)/18(H100) link 全打到 6 颗 NVSwitch,NVSwitch 片上 crossbar 让**任意两 GPU 间聚合到全部 link 带宽**(A100 600、H100 900 GB/s),**全连接无空洞**。
- **NVSwitch 3.0(H100)**:36 路 NVLink 4.0 端口,总交换容量 1.8 TB/s(注意:这是"对外总端口带宽",非单对带宽);每颗 NVSwitch 接所有 8 GPU。
- **第四代 NVSwitch(Blackwell)**:集成到同一 baseboard,B200 18 NVLink 全连,机内全连接无独立 NVSwitch 芯片(SXM baseboard 集成)。

### 3.3 NVSwitch domain 与跨机

- **NVSwitch domain**:同一组 NVSwitch 下的 GPU 池,**域内全连接**。DGX A100/H100 单机 = 1 个 NVSwitch domain(8 卡)。
- **跨 domain**:NVLink 是机内互连,跨机要走 InfiniBand/RoCE(NVLink 本身不出机箱)。但 **NVLink-Network / NVLink Switch**(NVLink 的网络化扩展,见 NVL72)可让 NVLink 跨机柜扩展——Blackwell NVL72 用 NVLink Switch tray 把 72 卡全连接(域 = 72 卡),跨 NVL72 域才走 IB。

### 3.4 NVLink-C2C(Grace-Hopper)

NVLink-C2C 是 NVLink 的**芯片间(chip-to-chip)变体**,连 Grace CPU 与 Hopper GPU(GH200),带宽 **900 GB/s 双向**(同 H100 NVLink 4.0 量级),CPU↔GPU 共享内存一致性,免去传统 CPU↔GPU 走 PCIe 的瓶颈。是 NVLink 走向"CPU-GPU 异构一致性内存"的关键。

### 3.5 NVL72 与 NVLink-Network

Blackwell GB200 NVL72:72 个 B200 GPU + 36 个 Grace CPU 组成单一 NVLink domain,**所有 72 卡 NVLink 全连接**(NVLink 5.0),跨节点用 NVLink Switch tray 扩展(NVLink-Network)。这是 NVLink 第一次"出机箱"跨机柜扩展。跨 NVL72(>72 卡)才退到 IB/XDR。具体 NVLink-Network 跨机带宽随拓扑变化,**待核实**(NVL72 内部 NVLink Switch tray 规格以官方文档为准)。

### 3.6 NCCL 怎么用 NVLink

NCCL 探测拓扑后选 `NV#`(NVLink)作为机内 channel transport,把 all-reduce 的 ring/tree 拓扑铺到 NVLink 上,双 ring 双向打满。无 NVLink 退化用 `P2P`(PCIe)或跨机 `NET`(IB/RoCE)。见 [[NCCL核心机制]] / [[NCCL通信拓扑]]。


## 4. 数学原理 / 公式

### 4.1 单 GPU 总双向带宽

设 NVLink 第 $g$ 代的 link 数 $L_g$、单 link 双向带宽 $b_g$,则:

$$
B_{\text{NVLink}}(g) = L_g \cdot b_g \quad [\text{GB/s, 双向}]
$$

代入学表:A100 $L_3=12, b_3=50 \Rightarrow 600$;H100 $L_4=18, b_4=50 \Rightarrow 900$;B200 $L_5=18, b_5=100 \Rightarrow 1800$。

### 4.2 NVLink vs PCIe 倍率

以 Gen5 x16(双向 126 GB/s,单向 63 GB/s)为基准:

| GPU | NVLink 双向 | / PCIe Gen5 双向 | NVLink 单向 | / PCIe Gen5 单向 |
|---|---|---|---|---|
| A100 | 600 | 4.8× | 300 | 4.8× |
| H100 | 900 | 7.1× | 450 | 7.1× |
| B200 | 1800 | 14.3× | 900 | 14.3× |

> 营销常说"NVLink 是 PCIe 的 7~14 倍"即指 Gen5 基准。对 Gen4(A100 实际机内 GPU↔CPU 仍是 Gen4),倍率更高(9.5×~28×)。

### 4.3 全连接所需 NVSwitch 端口数

8 卡全连接,每 GPU 有 $L$ link,需把这些 link 全打到 NVSwitch。NVSwitch 需 $\ge 8L$ 个 NVLink 端口(每 GPU $L$ 条上 NVSwitch)。A100: $8\times12=96$ 端口 → 用 6 颗 16-port NVSwitch? 实际 DGX A100 用 6 颗 NVSwitch 3.0,每颗 18 端口(实际每 GPU 每 NVSwitch 分到 2 link × 6 = 12 ✓)。逻辑:$L=12$ 拆成 6 组各 2 link,每组打 1 颗 NVSwitch,每颗 NVSwitch 接 8 GPU × 2 link = 16 端口。$C_8^2=28$ 对中每对在某颗 NVSwitch 上有 2 link 通路,6 颗 NVSwitch 全覆盖且每对 12 link 全打通。

### 4.4 TP all-reduce 时间(NVLink 打满)

[[all-reduce]] ring 算法单卡通信量 $\frac{2(n-1)}{n}M$(n 卡,$M$ 总数据),带宽 $B_{\text{dir}}$ 单向:

$$
T_{\text{comm}} \approx \alpha + \frac{2(n-1)}{n} \cdot \frac{M}{B_{\text{dir}}}
$$

A100 TP8,$M=768$ MB(一层激活),$B_{\text{dir}}=300$ GB/s,$\alpha=1.5\mu s$:

$$
T \approx 1.5 + \frac{14}{8}\cdot\frac{768}{300}\times10^3\mu s \approx 1.5 + 448 = 450\,\mu s
$$

同数据 PCIe Gen4($B_{\text{dir}}=31.5$, $\alpha=6$):$T\approx 6 + 1.75\times 24381 \approx 42673\mu s$ ≈ 95× 慢。这就是 TP 必须用 NVLink 的量化依据。


## 5. 代码示例

### 5.1 看 NVLink 拓扑与协商速率

```bash
# 两两 GPU 间 NVLink 链路数 (NV12 = 12 条 NVLink 连通, A100 全连接)
nvidia-smi topo -m

# 单 GPU 的 NVLink 状态与速率
nvidia-smi nvlink -l           # 列出每个 NVLink link
nvidia-smi nvlink -s 0          # GPU0 各 NVLink 状态 (Active/Inactive)
nvidia-smi -q | grep -A20 "NVLink"  # 看 NVLink 代际 / 速率 (H100: 50 GB/s/link)
```

### 5.2 测 NVLink 实际带宽

```bash
# NVIDIA 官方带宽测试工具 (nvbandwidth), 测 P2P NVLink 真实带宽
git clone https://github.com/NVIDIA/nvbandwidth && cd nvbandwidth && make
./nvbandwidth -t 22           # test 22 = bi-dir P2P read, 看是否打满 900 GB/s (H100)

# nccl-tests 跨机/跨卡 all-reduce
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8
# 关掉 NVLink 看退化: NCCL_P2P_DISABLE=1 ./build/all_reduce_perf ...
# 对比可见 NVLink 带宽 ~10x PCIe
```

### 5.3 NCCL 选 transport 日志

```bash
# 启用 NCCL debug 看 transport 选择 (应选 NV# )
NCCL_DEBUG=INFO ./all_reduce_perf ... 2>&1 | grep -E "Channel|via|NV"
# 期望: "via NV12" (12 link NVLink), 而非 "via PIX"(PCIe) 或 "via NET"(跨机)
```


## 6. 与其他知识点的关系

- **上游(依赖)**: [[PCIe]](对比基准)、[[alpha-beta性能模型]](带宽延迟建模)、[[拓扑感知]](NVSwitch domain 放置)
- **下游(应用)**: [[NCCL核心机制]](NCCL 选 NVLink 作 channel transport)、[[Tensor Parallel]](TP 强依赖 NVLink)、[[all-reduce]](ring/tree 打在 NVLink 上)、[[Megatron-LM]](TP/SP 调度按 NVSwitch domain 分组)、[[RDMA与GPUDirect]](机内 NVLink + 机外 RDMA 协同)
- **对比 / 易混**: [[PCIe]](通用总线)、[[InfiniBand]]/[[RoCE]](跨机网络,NVLink 是机内)、NVLink-C2C(CPU↔GPU vs GPU↔GPU)


## 7. 常见误区与易错点

> [!warning] 误区1:把"总带宽"当单向带宽
> NVIDIA 标的"900 GB/s"(H100)是**双向总和**,单向 450 GB/s。算一条 all-reduce 消息延迟要用单向。见 §4.4。

> [!warning] 误区2:NVLink 跨机也能用
> NVLink 默认**不出机箱**(SXM 版)。跨机只能走 IB/RoCE;只有 Blackwell NVL72 这种特殊产品用 NVLink-Network 扩到机柜级,且仍限于单 NVL72(72 卡)域内。

> [!warning] 误区3:PCIe 版 H100 也有 NVSwitch
> **PCIe 版 H100 没有 NVSwitch、没有机内全连接 NVLink**(只有有限点对点 NVLink Bridge 或完全无 NVLink)。TP 训练必须买 SXM 版。消费卡(4090)完全无 NVLink(3090/4090 双卡 NVLink Bridge 已被砍,仅 4090 无)。

> [!tip] 实践:训练前确认 NVSwitch domain
> `nvidia-smi topo -m` 必须显示**所有 GPU 对都是 `NV18`(H100)或 `NV12`(A100)**才证明全连接 NVSwitch domain;出现 `PIX/PHB` 说明该对走 PCIe,不能放同一 TP 组。

> [!note] 补充:NVLink 与 NVLink Bridge 的区别
> NVLink Bridge 是消费/工作站双卡 NVLink 物理连接器(2 卡直连,无 switch);数据中心 SXM 版用 baseboard 上的 NVSwitch 实现多卡全连接。两者拓扑不同。


## 8. 延伸细节

### 8.1 NVSwitch 各代

| NVSwitch 代 | 配套 GPU | 端口/带宽 | 平台 |
|---|---|---|---|
| NVSwitch 1.0 | V100 | 18 端口 × 50 GB/s | DGX-1 |
| NVSwitch 2.0 | —— | —— | (跳过,A100 直接 3.0) |
| NVSwitch 3.0 | A100 | 36 端口 NVLink 3.0,1.8 TB/s 交换容量 | DGX A100,6 颗 |
| NVSwitch 4.0 | H100 | 64 端口 NVLink 4.0,3.2 TB/s | DGX H100,4 颗(集成) |
| NVSwitch 5.0 / NVLink Switch | Blackwell | NVLink-Network,跨机柜 | GB200 NVL72 |

> 端口与交换容量数为 NVIDIA 官方 NVSwitch datasheet / 产品页标称。NVSwitch 5.0/NVLink-Network 跨机柜规格随 NVL72/GB200 NVL576 等产品变化,**具体端口数以官方最新文档为准**。

### 8.2 NVLink 单 link 物理层

NVLink 单 link 物理上是多对差分线(V100 每对 25 GB/s,多对组成一 link 50 GB/s 双向)。NVLink 4.0 每 link 4 对(差分对),物理速率与 PCIe Gen5 相近的 50 Gbaud PAM4,但协议是 NVIDIA 私有、开销低、专为 GPU 突发大块传输优化。**物理层细节待核实**(NVIDIA 未完全公开物理层 spec)。

### 8.3 为什么 TP 而非 DP 用 NVLink

- **DP(Data Parallel)**:每步 only all-reduce 梯度(通信量 ∝ 参数量),跨机走 IB/RoCE,机内 NVLink 帮助有限。
- **TP(Tensor Parallel)**:每层 all-reduce/all-gather 激活(通信量 ∝ 激活 × 层数),频率极高,必须机内 NVLink。
- **SP(Sequence Parallel)** / **EP(Expert Parallel)**:介于二者,SP 借机内 NVLink,EP 跨机走 IB。

调度原则:**把 TP 组放同 NVSwitch domain,DP/EP 跨机跨 domain。** 详见 [[集群拓扑感知placement]] / [[Megatron-LM]]。

### 8.4 与 Grace-Hopper / Blackwell 一致性内存

GH200 用 NVLink-C2C 让 CPU(LPBDDR)与 GPU(HBM)共享地址空间 + 一致性 cache,框架层无需手动 copy。这是 NVLink 从"GPU↔GPU 高速通道"演进到"异构一致性内存互联"的标志。Blackwell GH200/GB200 进一步扩展。与 [[memory bandwidth]] 协同:NVLink 解决卡间带宽,HBM 解决卡内带宽。

---
相关: [[PCIe]] · [[RDMA与GPUDirect]] · [[InfiniBand]] · [[NCCL核心机制]] · [[Tensor Parallel]] · [[all-reduce]] · [[alpha-beta性能模型]] · [[拓扑感知]] · [[memory bandwidth]] · [[18-集群网络存储与数据系统]]
