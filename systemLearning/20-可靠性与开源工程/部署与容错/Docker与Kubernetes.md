# Docker 与 Kubernetes

> **所属章节**: [[部署与容错]]
> **所属模块**: [[20-可靠性与开源工程]]
> **别名**: Docker / Kubernetes / K8s / 容器编排 / Pod / 容器化 / 镜像分层 / GPU 调度 / Kubeflow / Training-Operator / CRD
> **难度**: 中（需懂 Linux namespace/cgroup、GPU 资源模型、[[torch.distributed]]、[[rank与world size]]、[[Python multiprocessing与GIL]] 进程模型）


## 1. 一句话定义

**Docker** 是把"应用 + 依赖（库/运行时/系统文件）"打包成**镜像（image）**再跑成**容器（container）**的引擎——靠 Linux **namespace**（隔离视图：pid/net/mnt/uts/ipc/user）+ **cgroup**（限制资源：cpu/mem/blkio）实现进程级隔离，解决"我机器能跑、你机器不能"的环境一致性问题；**Kubernetes（K8s）** 是**容器编排系统**——在集群上自动调度、扩缩容、滚动更新、自愈容器，把"单机跑容器"升级成"几千台机器上跑几万个容器并保活"。在 ML 场景里，Docker 用于固化训练/推理镜像（CUDA + PyTorch + 依赖版本锁死），K8s 用于调度 GPU 节点上的训练 Job（通过 **nvidia.com/gpu** 资源 + **NVIDIA Device Plugin** 暴露 GPU）和推理服务（Deployment + Service + HPA）；训练任务通常不直接写 Pod，而是用 **Operator**（Kubeflow **Training-Operator** 的 `PyTorchJob`/`MPIJob` CRD、Volcano 的 gang scheduling）声明式提交。注意 K8s 调度 GPU **不能像 CPU 那样切片/超卖**——GPU 是**整数设备**，`nvidia.com/gpu: 1` 必须是整数，请求 1 卡就独占 1 卡，没有 fractional GPU（除非用 **MIG** 或 **time-slicing**，但 time-slicing 是自愿共享非隔离、官方明确说不适合强隔离 ML 训练）。

> [!note] 三句话定位
> - **Docker**：镜像分层（base + layers，Union FS）+ 运行时隔离（namespace + cgroup），让"环境"变成可复制的不可变制品。
> - **K8s**：Pod（最小调度单位，1+ 容器共享网络/存储）+ Deployment（副本数 + 滚动更新）+ Service（负载均衡 + 稳定 IP/DNS）+ Scheduler（按 resource request 调度到节点）。
> - **ML 上的角色**：固化 CUDA/PyTorch 版本镜像；用 `nvidia.com/gpu` 资源 + device plugin 调度 GPU；用 Training-Operator 的 `PyTorchJob` CRD 跑分布式训练；与 [[SLURM]] 互补——K8s 通用/弹性/服务化，SLURM 批 HPC 更强。


## 2. 为什么需要它（动机与背景）

### 2.1 环境不一致——"我机器能跑"

ML 工程最常见的痛：本地训得好好的，上集群报 `CUDA version mismatch`、`undefined symbol: ...`、`libstdc++.so.6: version not found`。根因是**依赖未锁**——CUDA driver/runtime 版本、cuDNN 版本、NCCL 版本、PyTorch 编译时的 CUDA tag、系统 glibc、Python ABI 全要一致。Docker 把这些**整套打包成镜像**，镜像到哪环境到哪，根除"环境漂移"。

### 2.2 隔离——同机多任务不互踩

GPU 集群上同节点跑多个任务（训练 + 推理服务 + 数据预处理）。裸跑会：进程互抢 CPU/内存、文件路径冲突、`CUDA_VISIBLE_DEVICES` 配错看到别人的卡。容器用 namespace 隔离 PID/网络/文件系统，cgroup 限制 CPU/内存/GPU，互不干扰。是 [[Python multiprocessing与GIL]] 之外的另一层"进程隔离"——容器级隔离比进程级更彻底（连网络栈、mount 树都隔离）。

### 2.3 编排——几千卡集群手动管不动

单机 Docker 只解决"跑一个容器"。但训练集群有几千节点、每节点多卡，要管：节点挂了把任务迁到哪、节点维护时优雅驱逐、滚动升级不停服、故障自愈（Pod 挂了自动重启）。手动 ssh 起进程不可行——K8s 用**声明式 API**（你写"我要 4 副本 PyTorchJob"，K8s 持续 reconcile 到这个状态）+ **controller**（watch + reconcile 循环）自动管。

### 2.4 GPU 调度——训练卡的特殊性

CPU/内存是**可切片可超卖**资源（request 0.5 CPU 能调度，limit 1 CPU 能限流）。GPU 是**设备资源**——一张物理卡要么给一个容器独占、要么不给，不能"请求 0.5 张卡"。K8s 通过 **Extended Resource** 机制（`nvidia.com/gpu`）+ **Device Plugin**（NVIDIA Device Plugin 向 kubelet 上报节点 GPU 数）让 scheduler 按"卡数"调度。这与 CPU 调度的数学本质不同（见 §4）。

### 2.5 训练任务的特殊性——不是无状态服务

Web 服务是**无状态**（挂了重启就行，请求级幂等）。训练任务是**有状态长时计算**（几小时到几天，中间有 checkpoint），且分布式训练多 worker 强同步（[[Distributed Data Parallel]] allreduce），**一个 worker 挂了整 job 通常得重启或弹性恢复**。所以训练用 K8s 不能照搬 Web 的 Deployment，要用 **Job/CRD**（跑完退出、失败重试、`parallelism`/`completions`、elastic 模式配合 [[elastic training]]）。


## 3. 核心概念详解

### 3.1 Docker：镜像 / 容器 / 分层

```
镜像 (image): 只读模板 (应用 + 依赖 + 系统)
  ├─ base image (如 nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04)
  ├─ layer 1: apt install -y python3.10 (只存 diff)
  ├─ layer 2: pip install torch==2.3.0 (只存 diff)
  └─ layer 3: COPY train.py /app/ (只存 diff)
容器 (container): 镜像 + 可写层 (copy-on-write) + namespace + cgroup
```

- **镜像（image）**：只读、分层（layered）。每条 Dockerfile 指令产出一个 layer，layer 只存**与上层的 diff**（Union FS，Docker 默认 **OverlayFS**）。分层让**复用**：base image + 公共 layer 跨镜像共享，拉镜像只下缺失 layer。
- **容器（container）**：镜像的运行实例。在镜像顶层加一层**可写层**（copy-on-write，改文件先复制到可写层再改），靠 namespace 隔离视图、cgroup 限资源。
- **namespace**（隔离"看得见什么"）：`pid`（进程号从 1 起）、`net`（独立网卡/端口/路由）、`mnt`（独立 mount 树）、`uts`（hostname）、`ipc`（消息队列）、`user`（uid 映射）。
- **cgroup**（限制"用多少"）：`cpu`（cpu shares / CFS quota）、`memory`（OOM killer）、`blkio`（IO 带宽）、`devices`（设备访问，GPU 通过这个 + `nvidia-container-runtime` 注入）。

### 3.2 Dockerfile 关键指令

```dockerfile
# base: 官方 CUDA 镜像 (已含 driver runtime + cudnn)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 装 Python (新 layer)
RUN apt-get update && apt-get install -y python3.10 python3-pip && rm -rf /var/lib/apt/lists/*

# 装 PyTorch (新 layer, 用 CUDA 12.1 wheel)
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 拷代码 (新 layer, 频繁变放最后, 利用缓存)
COPY train.py /app/train.py
WORKDIR /app

# 默认命令 (可被 docker run 覆盖)
CMD ["python3", "train.py"]
```

> [!tip] 实践：layer 缓存顺序
> Dockerfile **变化频率从低到高排**：base → 装系统包 → 装 Python 依赖（`requirements.txt` 单独 COPY 再 `pip install`）→ COPY 代码。改代码不触发重装 torch（上一层没变，缓存命中）。把 `COPY . .` 放最前会让每次改代码都重跑后面所有 layer，构建巨慢。

### 3.3 K8s：Pod / Deployment / Service / Scheduler

```
集群 (cluster)
├─ Node (节点, 物理机/VM)
│   ├─ kubelet (节点 agent, 管 Pod 生命周期, 接 device plugin)
│   ├─ 容器运行时 (containerd / CRI-O, 跑容器)
│   └─ device plugin (如 NVIDIA Device Plugin, 上报 GPU)
└─ Control Plane (master)
    ├─ API server (唯一入口, etcd 存状态)
    ├─ Scheduler (按 resource request 选 node)
    ├─ Controller manager (Deployment/Job controller, watch + reconcile)
    └─ etcd (一致性 KV 存储, 集群状态)
```

- **Pod**：K8s **最小调度单位**（不是容器！）。一个 Pod 含 **1 个或多个紧密耦合容器**，共享**网络栈**（同 Pod 容器 `localhost` 互通、共享 IP）、**IPC**（共享消息队列）、**存储卷**（同 Pod 挂同 volume）。一个训练 Pod 通常 1 容器（跑 torchrun）+ init 容器（拉代码/装 SSH key）。**Pod 是临时的**——挂了不会原地复活，是 controller 起新 Pod（IP 变）。这是为什么要用 Service 做"稳定入口"。
- **Deployment**：管 **Pod 副本数 + 滚动更新**。声明 `replicas: 4`，controller 持续保证 4 个 Pod 在跑；改镜像版本触发**滚动更新**（先起新 Pod 再杀旧 Pod，`maxSurge`/`maxUnavailable` 控节奏）。适合**无状态服务**（推理服务）。**不适合训练**（训练不是多副本对等服务）。
- **Service**：给一组 Pod（按 label selector 选）**稳定 ClusterIP + DNS 名**，做**负载均衡**（kube-proxy 用 iptables/IPVS 把流量转发到 Pod）。Pod IP 会变，Service IP 不变，依赖方连 Service 名。类型：`ClusterIP`（集群内）、`NodePort`（节点端口）、`LoadBalancer`（云 LB）、`Headless`（DNS 直返 Pod IP，StatefulSet 用）。
- **Scheduler**：Pod 创建后 `Pending`，scheduler 按 **resource request**（CPU/mem/GPU 的 request 字段）+ **node affinity/taint** 选节点。**关键：调度看 request 不看 limit**——request 是"保底"（调度时承诺节点有这么多），limit 是"上限"（运行时 cgroup 限死）。CPU 可 request 0.5 limit 2（超卖），GPU 必须 request 整数且 request=limit（不可超卖，见 §4）。

### 3.4 GPU 调度：nvidia.com/gpu + Device Plugin

```
节点装:
  ├─ NVIDIA driver (宿主机, GPU kernel module)
  ├─ nvidia-container-toolkit (改 containerd 配置, 让容器能访问 GPU)
  └─ NVIDIA Device Plugin (DaemonSet, 起在每 GPU 节点)
        │
        ├─ 向 kubelet 上报: nvidia.com/gpu = 8 (本节点 8 卡)
        └─ kubelet 把这数报给 scheduler
        │
 scheduler 调度 Pod (request nvidia.com/gpu: 1):
        ├─ 找有剩余 GPU 的节点
        ├─ 绑定到该节点
        └─ kubelet 起容器时, device plugin 把 /dev/nvidia0 + CUDA libs 注入容器
```

- **Extended Resource**：GPU 不在 K8s 内置资源里，是 **extended resource**（`nvidia.com/gpu`）。Device Plugin 向 kubelet `ListAndWatch` 上报节点设备数，kubelet 存到 API server，scheduler 据此调度。
- **整数设备**：`nvidia.com/gpu` 只能是**整数**（`1`/`2`/`8`，不能 `0.5`）。这是**设备资源**的本质——物理 GPU 不能切片给两个容器独立用（不像 CPU 能分时切片）。请求 1 卡 = 该 Pod 独占 1 物理卡。
- **不可超卖（默认）**：节点 8 卡，已分 8 个 Pod 各 1 卡，第 9 个 Pod 永远 `Pending`（即使别的卡闲置也不行——extended resource 不支持 overcommit，request=limit 强制）。
- **MIG（Multi-Instance GPU）**：A100/H100 支持把一卡硬件切成最多 7 个实例（`mig-instance`），每个实例独立显存/SM，可分给不同容器。这是**硬件级切片**（不是软件超卖），需节点预先配 MIG 策略，device plugin 暴露成 `nvidia.com/mig-1g.10gb` 这种 fractional 资源。适合"小模型推理多租"。
- **time-slicing**：NVIDIA Device Plugin 支持**时间片共享**（一卡多 Pod 轮转），但**无内存隔离、无算力隔离**（一个 Pod OOM 可能拖垮另一 Pod），官方明确**不建议用于强隔离生产训练**，只适合"开发/低优推理"。

> [!warning] 误区：GPU 像 CPU 一样能 request 0.5
> 不能。GPU 是设备资源，K8s extended resource 只接受整数，且不允许超卖。要 fractional 用 MIG（硬件切片）或 MPS（软件共享，仍是共享非隔离），time-slicing 是裸轮转不隔离。把 K8s 的 CPU 模型套到 GPU 是初学者最常踩的坑。

### 3.5 训练任务：Job CRD + Operator

训练不是"常驻服务"，是"跑完退出"的批任务。K8s 内置 `Job`（`parallelism`/`completions`/`backoffLimit`）能跑单进程任务，但**分布式训练**（多 worker + torchrun + rendezvous）用裸 Job 很笨拙——要手写 Pod 互找、SSH 互信、rank 分配。**Operator 模式**封装这些：

```
Kubeflow Training-Operator 提供 CRD:
  PyTorchJob  (PyTorch DDP)
  MPIJob      (MPI allreduce, Megatron)
  TFJob       (TensorFlow)
  JAXJob      (JAX)

你写 PyTorchJob YAML (声明 worker 副本数 + 镜像 + 命令 + GPU request):
  operator controller watch 到 -> 自动起 N 个 Pod (worker)
                         -> 注入 rendezvous env (MASTER_ADDR/PORT, WORLD_SIZE, RANK)
                         -> 跑 torchrun
                         -> 挂了按 elastic policy 重启/缩容
```

- **CRD（CustomResourceDefinition）**：用户自定义资源类型（如 `PyTorchJob`），存到 etcd，有专属 controller（operator）watch 并 reconcile 成真实 Pod。把"跑分布式训练的领域知识"编码进 controller。
- **Training-Operator**：Kubeflow 项目的训练 operator 集合。`PyTorchJob` spec 里写 `worker` 角色 `replicas: 4`、每副本 `nvidia.com/gpu: 1`，controller 自动起 4 Pod，注入 `MASTER_ADDR`/`WORLD_SIZE`/`RANK` env，跑 `torchrun`。
- **Volcano / gang scheduling**：默认 K8s scheduler 是"逐 Pod 调度"——一个 4-worker 训练 job，可能只调度成功 3 个（第 4 个节点没 GPU），这 3 个 `Pending` 等着，但占着别人的资源（**死锁/资源碎片**）。**Volcano** 是批调度器，支持 **gang scheduling**（"all-or-nothing"——要么 4 个 Pod 全调度成功要么都不调度），避免分布式 job 调度半截卡死。生产大模型训练几乎必上 Volcano 或类似的 gang scheduler。

### 3.6 与 SLURM 对比

| 维度 | K8s | SLURM |
|---|---|---|
| 出身 | Google 内部 Borg，2014 开源，**通用容器编排** | Lawrence Livermore 1996，**HPC 批作业调度** |
| 调度单位 | Pod（容器） | Job（脚本 + 资源请求） |
| 资源模型 | CPU/mem 可切片可超卖，GPU 整数设备 | 节点/cgroup/任务槽，GPU 整数 |
| 容器化 | 原生（容器是一等公民） | 后加（`--container-image`，srun 起容器） |
| 弹性 | 强（HPA/VPA/Cluster Autoscaler/抢占） | 弱（sinfo 静态分区，弹性弱） |
| 服务化 | 强（Deployment/Service/Ingress） | 弱（不是为长服设计的） |
| 多租 | 强（namespace/ResourceQuota/NetworkPolicy） | 弱（partition/account） |
| 批 HPC | 一般（需 Volcano gang） | 强（原生批 + 拓扑感知） |
| InfiniBand / 拓扑 | 需插件（如 NIC plugin、scheduling 拓扑约束） | 原生（`--gres=gpu`、`--constraint` 拓扑） |
| ML 典型用法 | 推理服务、训练 Operator、Notebook、CI | 大模型训练（很多超算仍用 SLURM） |
| 学习曲线 | 平缓（声明式 YAML + 大生态） | 陡（脚本语言 + HPC 黑话） |

> [!tip] 实践：什么时候用哪个
> - **云上、推理服务、训练 + 推理混部、CI/CD 流水线、多租平台** → K8s。
> - **超算中心、纯训练大 job、IB 拓扑敏感、InfiniBand P2P allreduce** → SLURM。
> - **混用**：大厂自研平台常是"SLURM 管训练裸金属 + K8s 管推理/服务"，或"K8s + Volcano 逐步吞训练"。两者不是替代，是分工。


## 4. 数学原理 / 公式

### 4.1 CPU 超卖 vs GPU 不可超卖的数学

K8s 的 CPU 是 **compressible resource**（可压缩）：节点有 $C$ 核，$P$ 个 Pod 各 request $r_i$、limit $l_i$。调度条件是**request 之和 ≤ C**（只看 request）：

$$\sum_i r_i \le C \quad (\text{调度约束, request})$$

但 **limit 之和可远超 C**（超卖）：

$$\sum_i l_i \gg C \quad (\text{合法, 因 CPU 可分时})$$

运行时若所有 Pod 同时满载，CFS 按 limit 比例分时（每 Pod 实际 $\approx C \cdot l_i / \sum l_j$，被 throttle 但不崩）。

GPU 是 **non-compressible device**（不可压缩）：节点 $G$ 卡，$P$ 个 Pod 各 request $g_i$（**必须整数**）。调度条件：

$$\sum_i g_i \le G, \quad g_i \in \mathbb{Z}_{\ge 0}, \quad g_i = l_i \ (\text{request == limit, 强制})$$

extended resource **禁止 $\sum l_i > G$**（不超卖）。原因：GPU 不能像 CPU 分时切片到任意精度给独立容器（CUDA context 切换重、显存不隔离），强行共享要么 MPS（软件共享非隔离）要么 time-slicing（裸轮转，一 Pod OOM 拖垮他人）。所以 $g_i$ 必须整数且严格独占。

> [!note] 推论：为什么 GPU 集群常"调度碎片"
> 节点 8 卡，已分 7 个 1 卡 Pod + 1 个被残留 Pod 占着（没清理），新 1 卡 Pod Pending。即便集群总卡够，单节点凑不齐整数也卡。这催生 **Volcano gang + bin-pack + 碎片整理（descheduler）** 的需求。

### 4.2 镜像分层与磁盘占用

镜像 $I$ 由 base $B$ 与 $n$ 层 $L_1,\dots,L_n$ 叠加，每层 $L_i$ 存与下层 diff。**实际磁盘**（含共享层）：

$$\text{disk}(I) = \sum_{i} \text{size}(L_i)$$

但**集群总下载量**因 layer 跨镜像/跨节点共享而远小于 $\sum_{\text{所有镜像}} \sum_{\text{层}} \text{size}$。设 $S$ 为 layer 集合，$N(l)$ 为引用 layer $l$ 的节点镜像实例数，全局实际下载 = $\sum_{l \in S} \text{size}(l)$（每 layer 在每节点只下一次）。这就是分层设计——**层是去重单位**。改 Dockerfile 重 build，只有变了的层及之后层重下，前面层命中缓存。

### 4.3 滚动更新的"可不可用"约束

Deployment 滚动更新：旧 Pod 数 $R_{old}$，新 Pod 数 $R_{new}$，目标 $N$ 副本。约束（`maxSurge`=$S$、`maxUnavailable`=$U$）：

$$R_{old} + R_{new} \le N + S, \quad R_{old} + R_{new} \ge N - U$$

即"最多多 $S$ 个、最少不少过 $U$ 个"。设 $S=1, U=0, N=4$：滚动期最多 5 个 Pod（多 1 起新）、最少 4 个（旧的不准先杀）。保证更新期间始终 ≥4 个可用。推理服务滚动升级不停服靠这个；**训练不滚动**（worker 不可对等替换），训练升级要重起 job 或用 [[elastic training]] 滚动加节点。

### 4.4 Pod 资源 request/limit 的 cgroup 映射

| K8s 字段 | cgroup 子系统 | 含义 |
|---|---|---|
| `requests.cpu: 500m` | cpu.shares (502) | 调度承诺 + 空闲时按比例抢 |
| `limits.cpu: "2"` | cpu.cfs_quota_us | CFS 限 2 核，超了 throttle |
| `requests.memory: 4Gi` | - | 调度承诺（节点 mem 够） |
| `limits.memory: 8Gi` | memory.limit_in_bytes | 超了 OOMKilled |
| `nvidia.com/gpu: 1` | devices + nvidia runtime | 独占 1 卡，request=limit |

注：`500m` = 500 millicores = 0.5 核。CPU 的 millicore 模型印证"可切片"；GPU 没有 millicard 概念，最小单位是 1。


## 5. 代码示例（可选）

### 5.1 ML 训练镜像 Dockerfile（CUDA + PyTorch）

```dockerfile
# 锁 CUDA 12.1 + cuDNN8 + Ubuntu22.04 base
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 系统包 (OpenMPI 给分布式, ssh 给 torchrun rendezvous)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip openssh-client openssh-server \
    openmpi-bin libopenmpi-dev wget git vim \
 && rm -rf /var/lib/apt/lists/*

# Python 依赖 (单独 COPY requirements 利用缓存)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# SSH 互信 (多 Pod torchrun 需 SSH 或 c10d rendezvous)
RUN mkdir -p /root/.ssh && \
    ssh-keygen -t rsa -b 2048 -f /root/.ssh/id_rsa -N "" && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys

# 配置 NCCL 环境 (IB/RDMA 集群关键)
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=PHB

# 代码
COPY train.py /app/train.py
WORKDIR /app
CMD ["torchrun", "--nproc_per_node=8", "train.py"]
```

### 5.2 单 Pod 多卡训练（K8s Pod YAML）

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: train-single
  labels:
    app: train
spec:
  containers:
  - name: trainer
    image: myregistry/train:cuda12.1-torch2.3
    command: ["torchrun", "--nproc_per_node=8", "train.py"]
    resources:
      limits:
        nvidia.com/gpu: 8          # 8 卡 (整数, 独占)
        cpu: "64"
        memory: "512Gi"
      requests:
        nvidia.com/gpu: 8          # GPU 必须 request==limit
        cpu: "64"
        memory: "512Gi"
    volumeMounts:
    - name: dshm                   # 共享内存 (dataloader 多 worker 要)
      mountPath: /dev/shm
    - name: data
      mountPath: /data
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi
  - name: data
    persistentVolumeClaim:
      claimName: train-data-pvc
  restartPolicy: OnFailure
```

> [!warning] 误区：忘配 `/dev/shm` 共享内存
> K8s 默认 `emptyDir` 给 64MB 的 `/dev/shm`。PyTorch DataLoader `num_workers>0` 用共享内存传 batch，64MB 必爆 `Bus error`/`Too large section`。必须显式挂 `medium: Memory` 的 `emptyDir`（吃内存）或 `sizeLimit` 调大。是 PyTorch on K8s 最经典坑之一。详见 [[Python multiprocessing与GIL]] 的 DataLoader 段。

### 5.3 PyTorchJob CRD（Training-Operator 分布式训练）

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: ddp-train
spec:
  pytorchReplicaSpecs:
    Worker:
      replicas: 4                   # 4 节点, 每节点 8 卡 = 32 卡
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: myregistry/train:cuda12.1-torch2.3
            command:
            - torchrun
            - --nproc_per_node=8
            - --nnodes=4            # 4:4 固定 (elastic 用 min:max)
            - --node_rank=$(POD_NAME_IDX)   # operator 注入
            - --master_addr=ddp-train-worker-0
            - --master_port=29500
            - train.py
            resources:
              limits:
                nvidia.com/gpu: 8
          volumes:
          - name: dshm
            emptyDir: { medium: Memory, sizeLimit: 64Gi }
          - name: data
            persistentVolumeClaim:
              claimName: train-data-pvc
```

Training-Operator controller 自动：起 4 个 `Worker` Pod，注入 `MASTER_ADDR`/`WORLD_SIZE`/`RANK`（或 `POD_NAME_IDX`），监控 Pod 状态，挂了按 `restartPolicy` 处理。这比手写 4 个裸 Pod + SSH 互信脚本干净得多。

### 5.4 Volcano gang scheduling（避免半截调度）

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: ddp-train-gang
spec:
  minMember: 4                      # 4 个 Pod 必须同时调度成功, 否则都不调度
  priorityClassName: high
---
# 在 PyTorchJob 的 Pod template 加 annotation 让 Volcano 接管
# metadata.annotations:
#   scheduling.volcano.sh/podgroup: ddp-train-gang
```

`minMember: 4` = gang（"要么 4 个一起调度要么 0 个"），避免 3 个调度成功占着卡、第 4 个等不到的死锁。生产大训练必上。

### 5.5 推理服务 Deployment + Service（vLLM）

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 3                       # 3 副本对等推理服务
  selector:
    matchLabels: { app: vllm }
  template:
    metadata:
      labels: { app: vllm }
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args: [--model, llama3-8b, --tensor-parallel-size, 1]
        resources:
          limits: { nvidia.com/gpu: 1 }
        readinessProbe:             # 就绪探针, ready 才收流量
          httpGet: { path: /health, port: 8000 }
        livenessProbe:              # 存活探针, 挂了 K8s 重启 Pod
          httpGet: { path: /health, port: 8000 }
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-svc
spec:
  selector: { app: vllm }           # 按 label 选 Pod
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
# 集群内访问 vllm-svc:8000 -> 负载均衡到 3 个 Pod (与 [[OpenAI API server]] 接口兼容)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: Linux namespace/cgroup（隔离/限制的底层）、[[Python multiprocessing与GIL]]（容器内进程模型、DataLoader `num_workers` 与 `/dev/shm`）、[[rank与world size]]（operator 注入 rank）、[[torch.distributed]]（operator 跑 torchrun）、SLURM（对照的 HPC 调度）。
- **下游（应用）**: [[elastic training]]（K8s + Training-Operator + elastic rendezvous 跑弹性训练）、[[故障检测与恢复]]（K8s liveness/readiness probe、Pod 自动重启）、推理服务部署（[[OpenAI API server]] 兼容接口、vLLM Deployment）、CI/CD 流水线（[[GitHub issue到PR流程]] 的 CI 在容器里跑）。
- **对比 / 易混**:
  - **Docker vs 裸机 vs VM**：见 §3.6 表。Docker 进程级隔离无 hypervisor 开销，VM 硬件级隔离重，裸机无隔离但最快。ML 训练常用 Docker（环境一致）+ 裸金属调度（性能）。
  - **K8s vs SLURM**：见 §3.6 表。K8s 通用/弹性/服务化，SLURM 批 HPC/拓扑/IB 强。不是替代是分工。
  - **K8s Deployment vs Job vs PyTorchJob CRD**：Deployment 管无状态副本（推理），Job 管单进程批（跑完退出），PyTorchJob CRD 管分布式训练（operator 注入 rank/world + rendezvous）。三者针对不同工作负载。
  - **K8s Pod vs 裸 docker run**：Pod 多容器共享网络/存储卷（sidecar/init 模式），裸 docker run 单容器自管。Pod 是 K8s 抽象，docker run 是 Docker CLI。
  - **MIG vs time-slicing vs MPS**：MIG 硬件切片（隔离强，A100/H100 支持，粒度固定）、time-slicing 裸轮转（不隔离，开发用）、MPS 软件共享 context（算力隔离弱，生产慎用）。三者都是"一卡多租"手段，隔离强度 MIG > MPS > time-slicing。


## 7. 常见误区与易错点

> [!warning] 误区 1：GPU 能像 CPU 一样 request 0.5 卡
> 不能。`nvidia.com/gpu` 是 extended resource，只接受整数且 request=limit 强制、禁止超卖。要 fractional 用 MIG（硬件切片）或 MPS（软件共享）。详见 §4.1。

> [!warning] 误区 2：K8s 默认 `/dev/shm` 够大
> 不够。默认 `emptyDir` 给 64MB。PyTorch DataLoader 多 worker 用共享内存传 batch，64MB 必爆 `Bus error`。必须显式挂 `medium: Memory` 的 emptyDir 并设 `sizeLimit`（如 64Gi）。

> [!warning] 误区 3：用 Deployment 跑训练
> 错。Deployment 是"常驻副本对等服务"，训练是"跑完退出 + 强同步多 worker"。训练用 `Job` 或 `PyTorchJob` CRD。Deployment 跑训练会：job 跑完 Pod 被 controller 又起一遍（违反"跑完退出"）、worker 挂了整 job 不重起对等替换。

> [!warning] 误区 4：分布式训练裸 Pod 手动配 SSH 就行
> 能跑但脆弱。Training-Operator 的 `PyTorchJob` CRD 自动注入 `MASTER_ADDR`/`WORLD_SIZE`/`RANK`，比手写 SSH 互信 + rank 分配脚本干净、不易错。生产用 Operator。

> [!warning] 误区 5：time-slicing 适合多租训练
> 不适合。time-slicing 是裸轮转，无显存/算力隔离，一 Pod OOM 拖垮同卡他人。官方明确只建议开发/低优。强隔离多租用 MIG 或单卡独占。

> [!warning] 误区 6：K8s 能直接跑大模型训练（像跑推理）
> 理论能，实践坑多。K8s 默认逐 Pod 调度会"半截卡死"（4 worker 只调度 3 个），需 Volcano gang；IB/RDMA 拓扑感知调度弱，P2P allreduce 性能可能掉；checkpoint 到 PVC 有 IO 瓶颈。很多大厂仍用 SLURM 跑训练 + K8s 跑推理，混用而非纯 K8s。

> [!warning] 误区 7：Pod 挂了原地重启
> 不。Pod 是临时的，挂了 controller 起新 Pod（新 Pod 新 IP 新名）。要"稳定 IP"用 Service。要"稳定身份 + 顺序"用 StatefulSet（推理很少用，训练几乎不用，因训练 worker 无状态可乱序）。

> [!warning] 误区 8：镜像越大越好（一次装齐）
> 反。镜像越大拉取越慢（几千节点并发拉会打爆 registry），启动越慢。用 multi-stage build（builder 阶段编译，runtime 阶段只拷产物 + 运行时），runtime 镜像精简。ML 镜像仍偏大（CUDA + PyTorch 几 GB），但应避免再塞无关文件。


## 8. 延伸细节

### 8.1 NVIDIA Container Toolkit 与 CDI

`nvidia-container-toolkit`（旧名 `nvidia-docker2`）是让容器访问 GPU 的核心——它改 `containerd`/`dockerd` 配置，在起容器时用 `--gpus` 或 device plugin 注入 `/dev/nvidiaN` + CUDA 用户态库 + 设置环境变量。新趋势是 **CDI（Container Device Interface）**——K8s 1.29+ 推荐用 CDI 标准化设备注入（不止 GPU，含 RDMA NIC、FPGA 等），比老的 hook 模式更规范，待核实生产可用性。

### 8.2 K8s 上的 NCCL / RDMA

`NCCL_IB_DISABLE=0`、`NCCL_NET_GDR_LEVEL` 等环境变量在镜像里设。但要让容器看到 RDMA NIC（Mellanox）需额外 device plugin（如 `k8s-rdma-sriov-device-plugin`），把 `/dev/infiniband/...` 注入容器，并配 SR-IOV 网络插件。这是 K8s 跑高性能训练的"最后一公里"，很多公司直接裸金属 + SLURM 绕开。NCCL 故障定位见 [[NCCL hang与timeout]]。

### 8.3 弹性训练与 K8s 的天然契合

K8s 的弹性（HPA/Cluster Autoscaler/抢占）配 [[elastic training]] 的"worker 动态进出"是绝配——节点被抢占时 Pod 被杀，torchrun elastic 检测 worker 缩、从 ckpt 恢复继续训，新节点加入时新 Pod 起、elastic 把新 worker 加回。这是"云原生弹性大模型训练"的核心范式。详见 [[elastic training]]。

### 8.4 Nodepool / 拓扑感知调度

GPU 节点常分池：H100 训练池、A10 推理池、A100 通用池。用 `nodeSelector`/`nodeAffinity`/`taint+toleration` 把训练 Pod 调到训练池。跨节点训练需拓扑感知（同 Pod 多卡走 NVLink、跨 Pod 走 IB），K8s 默认 scheduler 不感知拓扑，需 **Topology Manager**（kubelet 级，NUMA/设备亲和）+ 自研拓扑插件，是 K8s 跑训练的难点。[[Ray与分布式调度]] 的 Ray 在这点上对 ML 更友好（actor placement 可感知）。

### 8.5 与 CI/CD 的衔接

GitHub Actions / GitLab CI 的 runner 常是 K8s Pod。[[GitHub issue到PR流程]] 里 PR 触发的 lint/test 在容器 Pod 里跑，跑完销毁。这把"开发环境一致性"延伸到 CI——Docker 镜像既是部署制品也是 CI 环境。

### 8.6 内容来源

K8s 概念整理自 K8s 官方文档（Pod/Deployment/Service/Scheduler/Device Plugin/Extended Resource）、NVIDIA GPU Operator 与 device plugin 官方文档、Kubeflow Training-Operator README 与 `PyTorchJob` spec、Volcano gang scheduling 文档、`nvidia/cuda` 官方镜像 tag 说明。MIG/time-slicing 隔离强度按 NVIDIA Data Center GPU 文档。与 SLURM 的分工与混用参考大厂工程实践。CDI 与 K8s 1.29+ 集成待核实最新状态。关联见 [[elastic training]]、[[故障检测与恢复]]、[[NCCL hang与timeout]]、[[Python multiprocessing与GIL]]、[[Ray与分布式调度]]、[[OpenAI API server]]。

---
相关: [[部署与容错]] | [[elastic training]] | [[故障检测与恢复]] | [[metrics logging tracing]] | [[Python multiprocessing与GIL]] | [[rank与world size]] | [[torch.distributed]] | [[Distributed Data Parallel]] | [[NCCL hang与timeout]] | [[Ray与分布式调度]] | [[OpenAI API server]] | [[GitHub issue到PR流程]]
