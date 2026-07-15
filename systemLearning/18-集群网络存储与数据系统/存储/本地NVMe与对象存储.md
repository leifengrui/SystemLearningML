# 本地 NVMe 与对象存储

> **所属章节**: [[存储]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: 本地 NVMe / 对象存储 / S3-like / OSS / 冷存 / 热存 / 本地盘 vs 云存储 / NVMe SSD
> **难度**: 中（需懂 [[offloading (CPU-NVMe)]] + [[ZeRO (DeepSpeed)]] + [[训练显存估计]] + 数据吞吐模型）


## 1. 一句话定义

大模型训练的数据与 checkpoint 落盘有**两类存储后端**：① **本地 NVMe SSD**——插在节点 PCIe 槽上的本地固态盘（U.2/U.3/M.2 形态，PCIe Gen4/5 直连），**单盘读带宽 3–7 GB/s、写 3–5 GB/s、随机 4 KB IOPS 上百万、延迟 10–100 μs**，是训练节点**热数据（shard / activation / checkpoint 落盘）的首选**，但**容量有限（单节点 4–30 TB）、不可跨节点共享、盘坏即丢**；② **对象存储（S3 / OSS / GCS 等）**——把数据当"对象"（key→blob + 元数据）存在远端集群，**容量近乎无限、~$0.023/GB/月便宜、按 GET/PUT 次数计费**，但**单次 GET 延迟 10–100 ms、LIST 慢（1000 key/页）、随机/小文件性能差、带宽受 API 与每前缀速率限制**，适合**冷存 / 备份 / 训练数据的源头仓库**。实际大模型训练几乎都是**混合分层**：原始数据与历史 checkpoint 沉在对象存储，训练时把 shard 拉到本地 NVMe 顺序读、checkpoint 先同步落本地 NVMe（秒级，不卡 GPU）再异步传对象存储（分钟级，不阻塞训练）。

> [!note] 三句话定位
> - **是什么**：两类存储后端——本地 NVMe（快、贵、容量小、节点本地）与对象存储（慢、便宜、容量无限、远端按 key 访问）。
> - **为什么需要**：训练要高吞吐顺序/随机读（喂饱 GPU），又要海量冷存（原始数据 + 历史 ckpt），单一存储都做不到，必须分层。
> - **与 [[并行文件系统]] 关系**：并行 FS 是"多节点共享 + 并发读"的中间层，本地 NVMe 是节点内最快的热层，对象存储是最底冷层。三者构成"热—温—冷"存储栈。


## 2. 为什么需要它（动机与背景）

### 2.1 GPU 算力被存储喂不饱

单 H100 做 bf16 矩阵乘峰值 ~990 TFLOPS（等效 ~2 PB/s 的"乘加字节流"）。要喂饱它，训练数据读取不能成瓶颈。一条 token 的样本（含注意力、标签）约几 KB，batch=8、seq=4096 时**每 step 输入 ~几十 MB**，看似不大；但**预取、shuffle 窗口、多 worker 缓存、解码中间产物**把实际 IO 放大 10–100×，单 step IO 可达 GB 级。若读存储只有 100 MB/s（机械盘或慢 NFS），这一 step 的 IO 就要 10 秒，GPU 空转。

### 2.2 Checkpoint 落盘慢会卡 GPU

70B bf16 权重 = 140 GB，加 fp32 master + Adam(m,v) 状态约 **700–980 GB**（见 [[训练显存估计]]、[[state_dict分片]]）。若每 1000 step 存一次，**同步写**到 3 GB/s 的单 NVMe 要 230+ 秒（全状态），到 100 MB/s 的网络共享盘要上千秒。这段时间 GPU 全程空转，按 H100 ~$3/h、8 卡 ~$24/h 算，一次保存浪费几十美元，训练全程累计浪费可观。必须让 ckpt **本地快落 + 后台异步上传**，这是本地 NVMe + 对象存储分工的核心动机。

### 2.3 原始数据海量且冷

互联网级预训练语料可达数十 TB–PB，**不可能全装本地 NVMe**（贵、单节点装不下）。这些数据是**冷/温**的，访问模式是"训练开始拉一次 + 偶尔重洗"，正好匹配对象存储"便宜 + 容量大 + 低频访问"。所以原始 shard 仓沉在 S3/OSS，训练前批量拉到本地或并行 FS。

### 2.4 ZeRO offload 要本地 NVMe

[[offloading (CPU-NVMe)]] 与 [[ZeRO (DeepSpeed)]] 的 ZeRO-Infinity 把 optimizer state / 参数 offload 到 **NVMe**（不是对象存储，因为 offload 要在 step 间 prefetch 回 GPU，延迟必须低）。这强制训练节点配大本地 NVMe（如 7.3 TB ×4–8 盘）。对象存储延迟太高（ms 级），无法做 offload 后端。

> [!warning] 误区：对象存储能当 offload 后端
> 不能。ZeRO-Infinity 的 NVMe offload 假设 NVMe 的 ~10 μs 延迟与 GB/s 带宽。对象存储单次 GET 10–100 ms，prefetch overlap 会严重 stall，throughput 崩。offload 后端必须是本地 NVMe（或 [[并行文件系统]] 的 SSD 层）。

### 2.5 小文件 + 随机读的陷阱

百万级小文件（如逐张图、逐条 json）若直接放对象存储：每个文件一次 GET = 10–100 ms × 100 万 = 1–10 万秒，且 LIST 1000/页翻页慢。放本地 NVMe 随机读虽快（4K IOPS 上百万，单文件仍几十 μs），但元数据/打开句柄开销仍大。所以**结构化数据要打包成大 shard**（[[WebDataset]] 的 tar、[[Parquet]] 的 row group），变随机为顺序读，把单盘 GB/s 带宽吃满。


## 3. 核心概念详解

### 3.1 本地 NVMe SSD

- **形态**：U.2 / U.3（2.5"）企业盘（如 Samsung PM9A3、Micron 7450），或 M.2（消费级 980/990 Pro），插 PCIe Gen4/5 ×4 槽。NVMe 协议绕开 AHCI/SCSI 栈，多队列（64K 队列 ×64K 深）直连 CPU。
- **带宽**：单盘 Gen4 ×4 读 ~7 GB/s、写 ~3–5 GB/s；Gen5 ×4 读 ~14 GB/s。**多盘 stripe**（软件 RAID0 / md / LVM striping）可达 20–50 GB/s（8 盘 Gen4）。
- **延迟**：4K 随机读 QD1 ~50–100 μs，QD 深 ~10 μs 量级；顺序读更接近 PCIe 极限带宽。
- **IOPS**：4K 随机读 ~1M（Gen4）、~1.5M（Gen5），随机写 ~300K–1M。
- **容量**：单盘 1.9/3.84/7.68/15.36/30.72 TB；单节点 4–8 盘 = 15–60 TB。
- **持久性**：盘坏数据丢（非 RAID）；企业盘 PLP（掉电保护电容）保证写已提交不丢。

> [!note] 带宽数字来源
> 单盘 Gen4 ×4 ~7 GB/s、Gen5 ×4 ~14 GB/s 来自 NVMe SSD 厂商 datasheet（Samsung PM9A3、Micron 7450 PRO、Kioxia CM7 等）与 PCIe Gen4/5 ×4 理论上限（Gen4 单 lane ~2 GB/s ×4 ≈ 7.88 GB/s，编码后 ~7 GB/s；Gen5 ×4 ≈ 15.75 GB/s）。具体型号有出入，**标 ~7/~14 是稳定可信的经验值**。

### 3.2 对象存储（S3 / OSS / GCS）

- **模型**：`bucket / key → blob + 元数据`，无目录层级（key 里的 `/` 只是字符串）。无 POSIX open/read/write，只有 `PUT`（写）、`GET`（读整对象或 Range）、`LIST`（列 key）、`DELETE`。
- **延迟**：单次 GET first-byte ~30–100 ms（同区，跨区/跨洲 +几十到几百 ms）；PUT 类似。
- **吞吐**：受**每前缀（prefix）速率限制**（S3 ~5500 GET/s、3500–5500 PUT/s 每 prefix，可水平分 prefix 突破）；单 TCP 连接带宽受限（~几十到几百 MB/s），靠**并发连接 × 分 prefix**叠加到 GB/s–数十 GB/s。
- **LIST**：分页 1000 key/请求（v2 可调到 1000），大 prefix 翻页慢、是最终一致的（list 不保证看到刚 PUT 的对象）。
- **一致性**：S3 自 2020-12 起对 PUT/DELETE **强一致读后写**（overwrite 后立即可读到新值）；**LIST 仍最终一致**。OSS/GCS 类似（待核实细节）。
- **计费**：存储 ~$0.023/GB/月（标准）、GET ~$0.0004/1000、PUT ~$0.005/1000；还有**出口带宽费**（跨区/公网下行，~$0.09/GB）。

> [!tip] 实践：分 prefix 突破对象存储吞吐
> 把 shard 命名分散到多前缀：`s3://bkt/shard-0000/`、`shard-0001/`…而非全堆一个前缀。S3 按 prefix 负载均衡到不同分区，每 prefix 各给 5500 req/s，10 个前缀 ×并发 = 55K req/s，吞吐线性扩展。这是大规模读对象存储的标准技巧。

### 3.3 为什么大模型训练要本地 NVMe

| 用途 | 为什么必须本地 NVMe |
|---|---|
| **训练 shard 读** | 顺序读 GB/s，喂饱 GPU prefetch；对象存储延迟/速率限制撑不住高频读 |
| **checkpoint 同步落盘** | 140 GB+ 全状态写盘，NVMe GB/s 级 → 秒级；对象存储写要分钟级且会卡训练 |
| **ZeRO offload 后端** | step 间 prefetch optimizer state，延迟必须 μs–低 ms，对象存储 ms 级会 stall |
| **临时 shuffle / 缓存** | 数据 pipeline 中间产物，放本地快、隔离 |

### 3.4 对象存储的局限

- **LIST 慢**：百万 key 的 prefix 要翻千页，分钟级。
- **随机/小文件差**：每文件一次 GET = 10–100 ms，吞吐崩。
- **最终一致（LIST）**：训练前拉数据若依赖 LIST 看全量，可能漏刚写入的对象。
- **出口带宽费**：跨区/公网下行按 GB 收费，拉 PB 级数据可能比存储费还贵。
- **无 POSIX**：不能 `mmap`、不能随机 seek 改写（对象不可变，改 = 重写整对象 + 版本）。

### 3.5 混合分层策略

```
对象存储 (冷, 海量, 便宜)          ← 原始 shard 仓 / 历史 ckpt / 备份
       |  (训练前 batch pull / 异步上传 ckpt)
并行 FS 或本地 NVMe (热/温)        ← 当前 epoch shard / 最新 ckpt
       |
GPU HBM (最热, GB-TB/s)            ← 训练
```

- **热数据**（当前 epoch shard、当前 ckpt、offload state）→ 本地 NVMe（或并行 FS SSD 层）。
- **冷数据**（历史 shard、历史 ckpt、原始语料）→ 对象存储。
- **checkpoint**：先 `torch.save` 到本地 NVMe（秒级，训练只等这步），起后台线程 `s3.upload_file` 异步传（分钟级，不阻塞训练），传完可选删本地腾空间。详见 [[async distributed checkpoint]]。

### 3.6 三者横向对比

| 维度 | 本地 NVMe | 对象存储 S3/OSS | [[并行文件系统]]（Lustre/GPFS） |
|---|---|---|---|
| **单盘/单对象带宽** | 3–14 GB/s | 单连接 ~数十–数百 MB/s（并发分 prefix 到 GB/s+） | 单 OST 1–10 GB/s，多 OST 聚合 10–100+ GB/s |
| **延迟** | 10–100 μs | 10–100 ms | ms 级（网络往返） |
| **容量** | 单节点 4–60 TB | 近乎无限 | 跨节点共享，PB 级 |
| **共享** | ❌ 节点本地 | ✅ 远端共享 | ✅ POSIX 共享挂载 |
| **一致性** | 强（本地） | PUT 强 / LIST 最终 | POSIX 强一致 |
| **随机/小文件** | 强（IOPS 上百万） | 差 | 中（依赖元数据服务） |
| **成本** | 贵（$/GB） | 便宜（$0.023/GB/月） | 中（集群自建/商用） |
| **典型用途** | 热数据 / ckpt / offload | 冷存 / 源仓 / 备份 | 多节点共享热数据 |


## 4. 数学原理 / 公式

### 4.1 Checkpoint 落盘时间模型

checkpoint 数据量 $D$（字节），写带宽 $BW_{\text{w}}$，落盘时间：

$$T_{\text{save}} = \frac{D}{BW_{\text{w}}}$$

例：70B bf16 权重 $D=140\text{ GB}$。
- 单盘 NVMe 写 $BW_{\text{w}}\approx 3\text{ GB/s}$ → $T\approx 47\text{ s}$。
- 8 盘 stripe $BW_{\text{w}}\approx 24\text{ GB/s}$ → $T\approx 6\text{ s}$。
- 对象存储单连接上传 $\approx 0.1\text{ GB/s}$，并发 16 路 $\approx 1.6\text{ GB/s}$ → $T\approx 90\text{ s}$（且不含每请求开销）。

全训练状态（bf16 param + fp32 master + Adam m,v）$D \approx 70B\times(2+4+4)\text{B}=700\text{ GB}$，单盘 → ~233 s，8 盘 → ~30 s。这就是为什么必须**先快落本地、再异步上传**。

### 4.2 异步上传的 GPU 空闲消除

设每 $N$ step 存一次，同步策略 GPU 空闲 $T_{\text{save}}$；异步策略 GPU 只等 snapshot+入队 $T_{\text{snap}} \ll T_{\text{save}}$。训练总时间节省：

$$\Delta T = \frac{\text{total\_steps}}{N}\cdot(T_{\text{save}} - T_{\text{snap}})$$

若 $T_{\text{save}}=30\text{ s}$、$T_{\text{snap}}=1\text{ s}$、每 1000 step 存、共 10 万 step → 节省 $\frac{100000}{1000}\times 29 \approx 2900\text{ s}\approx 48\text{ min}$ GPU 时间。

### 4.3 顺序读吞吐 vs 小文件随机读

单盘顺序带宽 $BW_{\text{seq}}\approx 7\text{ GB/s}$。若样本平均 $s$ 字节、顺序打包成 shard：

$$\text{throughput}_{\text{seq}} = \frac{BW_{\text{seq}}}{s}\text{ 样本/秒}$$

若百万小文件随机读，受限于每文件打开/GET 延迟 $t_{\text{open}}$（NVMe ~50 μs，对象存储 ~50 ms）：

$$\text{throughput}_{\text{rand}} \approx \frac{1}{t_{\text{open}}}\text{ 样本/秒/连接}$$

对象存储单连接 $\frac{1}{0.05}\approx 20$ 样本/s，并发 1000 才 2 万/s；而顺序读 shard 7 GB/s ÷ 10 KB/样本 ≈ 70 万样本/s。差 35×。这就是 [[WebDataset]] 打包 shard 的根本动机。

### 4.4 NVMe offload 的带宽匹配

ZeRO-Infinity 把 optimizer state 卸 NVMe，每 step 前 prefetch 回 GPU。需带宽：

$$BW_{\text{need}} = \frac{D_{\text{state}}/P}{T_{\text{comp\_step}}}$$

其中 $D_{\text{state}}/P$ 是每卡 state 量，$T_{\text{comp\_step}}$ 是该 step 计算时间。若 prefetch 与计算 overlap，要求 $BW_{\text{NVMe}}\ge BW_{\text{need}}$，否则 stall。70B、8 卡、$D_{\text{state}}=700\text{ GB}$、每卡 87.5 GB、step 0.5 s → 需 175 GB/s（单盘远不够，需多盘 stripe 或换 CPU 内存 offload）。详见 [[offloading (CPU-NVMe)]] §4。


## 5. 代码示例（可选）

### 5.1 混合 checkpoint：本地同步落盘 + 异步传 S3

```python
import os, threading, torch
import boto3

s3 = boto3.client('s3')
LOCAL_CKPT_DIR = "/local_nvme/ckpt"   # 本地 NVMe 挂载点
S3_BUCKET = "my-ckpt-bucket"
S3_PREFIX = "run-001"

def save_checkpoint_local(state, step):
    """同步落本地 NVMe: 训练只等这步 (秒级)"""
    path = os.path.join(LOCAL_CKPT_DIR, f"step-{step}.pt")
    torch.save(state, path)            # NVMe ~3-24 GB/s, 140GB → 数秒~数十秒
    return path

def async_upload_to_s3(local_path, s3_key):
    """后台传对象存储: 分钟级, 不阻塞训练"""
    def _upload():
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        os.remove(local_path)          # 传完删本地腾 NVMe
    t = threading.Thread(target=_upload, daemon=True)
    t.start()
    return t

# 训练循环
state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}
local = save_checkpoint_local(state, step)            # GPU 只等这步
async_upload_to_s3(local, f"{S3_PREFIX}/step-{step}.pt")  # 异步传, 训练继续
```

### 5.2 读对象存储 shard 时分前缀并发

```python
from concurrent.futures import ThreadPoolExecutor
import boto3

s3 = boto3.client('s3')

def download_shard(idx):
    # 分散到不同前缀, 突破每前缀速率限制
    prefix = f"shard-{idx//100:04d}"        # 每 100 shard 一组前缀
    key = f"{prefix}/train-{idx:06d}.tar"
    s3.download_file("my-bucket", key, f"/local_nvme/{key.split('/')[-1]}")

# 并发拉 1000 个 shard, 利用对象存储水平扩展
with ThreadPoolExecutor(max_workers=64) as ex:
    list(ex.map(download_shard, range(1000)))
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[PCIe]]（NVMe 走 PCIe Gen4/5）、[[offloading (CPU-NVMe)]]（offload 后端是本地 NVMe）、[[ZeRO (DeepSpeed)]]（ZeRO-Infinity 用 NVMe）、[[训练显存估计]]（ckpt 数据量核算）。
- **下游（应用）**: [[async distributed checkpoint]]（本地快落 + 异步上传对象存储）、[[WebDataset]]（shard 顺序读对象存储）、[[Parquet]]（结构化冷存对象存储）、[[checkpoint转换]]（ckpt 在对象存储间转格式）。
- **对比 / 易混**:
  - **本地 NVMe vs 对象存储**：快/贵/本地 vs 慢/便宜/远端；热 vs 冷。互补非替代。
  - **对象存储 vs [[并行文件系统]]**：对象存储 key-value 无 POSIX、最终一致 LIST；并行 FS 是 POSIX 共享挂载、强一致、多节点并发读。AI 训练热数据多选并行 FS 或本地 NVMe，对象存储做源仓/备份。
  - **本地 NVMe vs CPU 内存 offload**：NVMe 容量大但带宽低（3–14 GB/s vs CPU 内存 32–100 GB/s），offload 巨量 state 选 NVMe，offload 高频访问的选 CPU 内存。见 [[offloading (CPU-NVMe)]]。


## 7. 常见误区与易错点

> [!warning] 误区 1：对象存储能直接当训练数据热读
> 不行。单 GET 10–100 ms、LIST 慢、速率限制。必须先拉到本地 NVMe / 并行 FS，或用 [[WebDataset]] 打包成大 shard 顺序读。

> [!warning] 误区 2：ckpt 同步写对象存储没事
> 140 GB 写对象存储要分钟级且卡训练。必须本地 NVMe 先落、再异步上传。见 [[async distributed checkpoint]]。

> [!warning] 误区 3：本地 NVMe 多盘就一定快
> 软件单线程 IO 跑不满多盘。要 stripe（md/LVM RAID0）或应用层并发，把多盘带宽聚合。单线程 `torch.save` 也只能吃单盘一部分带宽，可分卡分盘并行写。

> [!warning] 误区 4：NVMe 不会坏，ckpt 只存本地就行
> 盘坏即丢、节点挂也丢。本地 NVMe 只做"快取/中转"，**最终归宿必须是对象存储/并行 FS 的冗余存储**。

> [!warning] 误区 5：对象存储 LIST 能看全最新对象
> S3 的 LIST 仍最终一致（PUT 已强一致，但 LIST 可能滞后看到刚 PUT 的 key）。依赖 LIST 拉全量要小心，建议维护 manifest 文件而非靠 LIST。

> [!tip] 实践：ckpt 上传用 multipart + 断点续传
> 大 ckpt 上传 S3 用 multipart upload（分片 5–100 MB），单分片失败可重传，整体可断点续传；并发分片上传把吞吐拉满。boto3 的 `upload_file` 默认自动 multipart（>8 MB）。

> [!note] 补充：本地盘 vs 弹性云盘
> 云上训练机有两种本地盘：①**实例本地盘**（如 AWS P5 的 instance store、阿里云本地盘），临时、停机即丢、最快；②**EBS/云盘**，网络块存储、持久但带宽低且共享。训练热数据用①，持久用对象存储/并行 FS。EBS 当热读会成瓶颈。


## 8. 延伸细节

### 8.1 NVMe 多队列与 CPU

NVMe 每核可绑一个 IO 队列，避免跨核锁争用。PyTorch/数据 loader 多 worker 时，每 worker 的读请求分散到多队列，单盘 IOPS 能跑满。这是为什么 NVMe 比 SATA SSD（单队列 AHCI）在大并发下快得多。

### 8.2 对象存储的 S3 Select / Glacier

S3 Select 支持对单个对象做 SQL 过滤（谓词下推到对象存储），减少传回数据量；Glacier 是极便宜归档层，恢复要分钟–小时，只适合长期冷备（不参与训练热路径）。

### 8.3 数据本地性

把训练调度到数据所在节点（数据本地性）能省网络传输。云上对象存储与计算节点同区可省出口费；本地集群用并行 FS 时尽量让 worker 读本地副本（[[拓扑感知]]）。

### 8.4 内容来源

本地 NVMe 带宽来自 Samsung/Micron/Kioxia 企业盘 datasheet 与 PCIe Gen4/5 规格；对象存储特性来自 AWS S3 官方文档（consistency model、request rates、pricing）与阿里云 OSS 文档对照。混合策略与 ckpt 异步上传参考 [[async distributed checkpoint]] 与 DeepSpeed/Megatron 工程实践。带宽/延迟具体值随型号变化，关键数量级（GB/s vs 百 MB/s、μs vs ms）稳定。

---
相关: [[存储]] | [[并行文件系统]] | [[async distributed checkpoint]] | [[offloading (CPU-NVMe)]] | [[ZeRO (DeepSpeed)]] | [[训练显存估计]] | [[WebDataset]] | [[Parquet]] | [[拓扑感知]]
