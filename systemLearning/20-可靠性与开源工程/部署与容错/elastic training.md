# elastic training

> **所属章节**: [[部署与容错]]
> **所属模块**: [[20-可靠性与开源工程]]
> **别名**: elastic training / 弹性训练 / torchrun elastic / torchelastic / rendezvous / 动态成员训练 / 节点动态加入退出 / `torch.distributed.run` --elastic / `torch.distributed.elastic`
> **难度**: 中高（需懂 [[torch.distributed]]、[[Distributed Data Parallel|DDP]]、[[rank与world size]]、[[async distributed checkpoint]]、rendezvous 协议、c10d/etcd backend）

## 1. 一句话定义

**elastic training（弹性训练）** 是训练过程中 **worker 可动态加入/退出而训练不崩溃、不从头重训**的机制——传统 [[Distributed Data Parallel|DDP]] 假设 **固定 world size**（`WORLD_SIZE=8`，8 卡全程不变），任一 worker 掉了 allreduce 就死锁、整 job 必须重启；弹性训练用 **rendezvous（会合）协议** 让 worker 在加入/退出时**重新协商 rank/world_size**，配 **checkpoint 恢复**让新成员从最近 ckpt 接着训。PyTorch 的实现是 **`torchrun --elastic`**（即 `torch.distributed.run` 的 elastic 模式，包在 `torch.distributed.elastic` / `torch.distributed.launcher` 里），关键参数 `--nnodes=min:max`（节点数允许在 min..max 间动态变）+ `--rdzv_backend`（`c10d`/`etcd`/`file`/`tcp`）+ `--rdzv_endpoint`（会合地址）。它是"云上抢被回收就续训"的核心范式——配云的**抢占式实例**（preemptible / spot 实例）或 K8s 抢占，节点被回收时 torchrun 检测 worker 掉、缩小 world、从 ckpt 恢复继续；新节点加入时 torchrun rendezvous 新 worker、扩大 world、allreduce 重新组网。**代价**：每次成员变更要重新 rendezvous + 从 ckpt 恢复 + 重组 allreduce（有秒级到分钟级停顿），且 DDP 的 batch 不再全球均匀切（data sampler 要按新 world 重切分）。**与"固定训练"的本质区别**：固定训练把 world_size 当**常量**，弹性训练把 world_size 当**可变量**（运行时由 rendezvous 决定当前成员）。

> [!note] 三句话定位
> - **是什么**：训练中 worker 动态进出，rendezvous 重新协商 rank/world_size，从 ckpt 恢复续训，不崩溃不重头。
> - **为什么**：云上抢占式实例会被回收、节点会故障、滚动升级要不停训——固定 world_size 一掉就死锁。
> - **怎么用**：`torchrun --nnodes=2:8 --rdzv_backend=c10d --rdzv_endpoint=host:port train.py`，代码里用 `elastic launch` + `init_process_group` + 周期 ckpt，agent（torchrun）监控成员变更触发 re-rendezvous。


## 2. 为什么需要它（动机与背景）

### 2.1 固定 world size 的脆弱

传统 DDP 训练脚本：

```python
# torchrun --nnodes=4 --nproc_per_node=8 train.py  (固定 4 节点 32 卡)
dist.init_process_group(backend="nccl")  # world_size=32 固定
model = DDP(model)
for batch in loader:
    loss = model(batch)          # forward
    loss.backward()              # allreduce grads
    optimizer.step()
```

`init_process_group` 后 `world_size=32` 被钉死。若第 1000 步某 worker 节点 OOM 崩了，该 rank 的 allreduce 永远等不到 → **NCCL hang**（见 [[NCCL hang与timeout]]），整个 job 死锁，只能 kill 全部重启。几小时训练成果若没 ckpt 就丢光。**几千卡训练一卡掉整 job 死**——成本不可承受。

### 2.2 云上抢占式实例

云厂商（AWS Spot/GCP Preemptible/Azure Spot/Aliyun 抢占式）给"可被随时回收的实例"打折（最多 80% off）。但"随时回收"意味着训练 worker 随时被杀。固定训练完全用不了 spot（一回收就死）。弹性训练让 spot 回收时缩容续训、新 spot 拍到时扩容加回——把"不可用的便宜算力"变成可用，**降本核心**。

### 2.3 节点故障 / 滚动升级

GPU 集群节点会故障（GPU ECC 错误、XID error、网卡掉、磁盘满）。也会滚动升级（换驱动、换 CUDA 版本、内核 patch）。固定训练遇到任一就得停 job。弹性训练让坏节点退出、新节点加入，训练**连续不停**。这与 [[故障检测与恢复]] 的"自愈"理念一致。

### 2.4 不想从头重训

大模型训练动辄几千卡天、成本 Millions。一次 NCCL hang 全重来 = 烧钱。弹性训练 + 周期 ckpt = 只丢最近一个 ckpt 间隔（见 §4.3 RPO），把"灾难性失败"降成"秒级停顿"。

### 2.5 与 DDP 固定假设的冲突

DDP 的 allreduce 假设**全员在线**（每个 rank 都参与 bucket allreduce）。成员变更时（一 rank 退出），剩余 rank 的 allreduce 若还按原 world_size 就死等。弹性训练的做法是 **re-rendezvous + 重组 group**——成员变更后所有存活 rank 重新 `init_process_group`（新 world_size），从 ckpt 恢复参数，data sampler 按新 world 重切 batch。这是"重启 DDP"而非"热修补 DDP"。


## 3. 核心概念详解

### 3.1 torchrun elastic 模式

```bash
# torchrun --elastic: --nnodes=min:max (节点数范围) + rdzv_backend + rdzv_endpoint
torchrun \
  --nnodes=2:8 \                         # 最少 2 节点起训, 最多 8 节点
  --nproc_per_node=8 \                   # 每节点 8 卡
  --rdzv_backend=c10d \                  # 会合后端 (c10d=PyTorch 自带, etcd=外部 KV)
  --rdzv_endpoint=master:29500 \         # 会合地址 (任意可达节点)
  --rdzv_id=job123 \                     # 会话 id (同 id 才能 rendezvous)
  --max-restarts=3 \                     # 每个 worker 最多重启 3 次
  train.py
```

- `--nnodes=min:max`：节点数范围。`2:8` = 2..8 节点可变；`:4` = `1:4`（min 默认 1）；`4`（单值）= `4:4`（固定，等价传统）。**min 是"起训门槛"**（凑齐 min 才 rendezvous 起训），**max 是"上限"**（超过 max 的新节点 standby 等）。
- `--rdzv_backend`：会合后端。
  - `c10d`：PyTorch 自带的会合实现，无需外部依赖，**推荐**。
  - `etcd`：用 etcd 做 KV 存储（旧推荐，需额外部署 etcd 集群）。
  - `file`：用共享文件系统（NFS）做会合，单机或共享 FS 跨机，**生产慎用**（NFS 一致性/性能坑）。
  - `tcp`：TCP 静态地址会合，类似传统 `init_method='tcp://'`，弹性弱。
- `--rdzv_endpoint`：会合地址。`c10d` 时是任一可达节点的 `host:port`；所有 worker 都连这个地址做会合。
- `--rdzv_id`：会话 id。同一 id 才能互相 rendezvous。防止两个独立 job 串台。
- `--max-restarts`：单 worker 崩溃后 agent（torchrun）自动重启该 worker 的上限。超了整个 job 退出。

### 3.2 rendezvous（会合）协议

rendezvous 是 elastic 的核心——worker 加入/退出时**所有成员重新协商**：

```
1. worker 启动, 连 rdzv_endpoint, 报名 "我要加入 rdzv_id=job123"
2. rdzv backend 收报名, 凑齐 min_nodes (如 2) 才开始:
   a. 分配 rank (按报名顺序 0,1,2,...)
   b. 协商 world_size (= 当前报名数, 但 <= max_nodes)
   c. 广播 MASTER_ADDR/MASTER_PORT/WORLD_SIZE/RANK 给全员
3. 所有 worker 调 init_process_group(backend='nccl', rank=..., world_size=...)
4. 训练
5. 某 worker 退出/加入 -> 打破 rendezvous -> 所有存活 worker 重新走 1-4 (re-rendezvous)
   -> 新 rank/world_size, 从 ckpt 恢复
```

- **会合 = 状态同步点**。所有 worker 必须到齐（达 min）才往下走。这是"全员同意新成员"的共识。
- **re-rendezvous 的代价**：秒级到分钟级停顿（NCCL 重组、ckpt 加载）。频繁变更成员会让训练吞吐抖动。
- **rank 是会合时分配的**，不是节点固定属性。一节点退出再回来，rank 可能变。

### 3.3 quorum（法定人数）= min_nodes

rendezvous 要"凑齐 min_nodes 才起训"——这是 **quorum**（法定人数）机制：

- $N$ 个 worker 报名，$N \ge \text{min}$ → 开始 rendezvous，world_size=$N$（若 $N \le \text{max}$）。
- $N < \text{min}$ → 等待（pending），不开始。
- 训练中某 worker 退出，剩 $N'$ 个：
  - $N' \ge \text{min}$ → re-rendezvous，新 world_size=$N'$，继续训。
  - $N' < \text{min}$ → **暂停训练**（standby），等新 worker 加入凑齐 min 再 re-rendezvous。

min 是"训练的法定人数"——低于 min 训练暂停但不退出（等凑齐）。这是"软容错"——单 worker 退出不杀 job，只暂停。详见 §4.1。

### 3.4 standby（待命）

$N > \text{max}$ 时多出的 worker **standby**（待命）——已连 rdzv_backend 但不参与当前 world。当某个 active worker 退出，standby 的被"提拔"进 active（re-rendezvous）。这让扩容平滑——新节点先 standby，等下次 re-rendezvous 才正式加入。常配 K8s HPA 或 Cluster Autoscaler：自动扩节点 → standby → 下次成员变更加入。

### 3.5 checkpoint-based 恢复

成员变更后，**所有 rank 都从最近 ckpt 恢复**（不是只恢复新 rank）：

```python
# train.py (elastic-aware)
def main():
    init_distributed()                  # rendezvous 协商 rank/world_size
    model = DDP(build_model().cuda())
    optimizer = optim.AdamW(model.parameters())
    sampler = DistributedSampler(dataset)  # 按 world_size 切分
    step = 0
    if os.path.exists(ckpt_path):
        step, model, optimizer = load_ckpt(ckpt_path)  # 从 ckpt 恢复
        print(f"recovered from step {step}, world_size={dist.get_world_size()}")
    for batch in loader:
        # ... train ...
        if step % 100 == 0:
            save_ckpt(step, model, optimizer)   # 周期 ckpt
        step += 1
```

- **全员从 ckpt 恢复**：成员变更后新 group 的所有 rank（包括没崩的）都重载 ckpt。因为 allreduce 重新组网，旧 optimizer state 不对应新 world，最稳是全员 reload（优化器 state 若用 sharded 可只 reload 自己 shard，见 [[async distributed checkpoint]]）。
- **sampler 重切分**：`DistributedSampler` 默认按 `world_size` 均匀切。world_size 变了，sampler 自动按新 world 重切（每个 rank 的数据范围变）。这避免"同一样本被新旧行 rank 重复训"。
- **step 同步**：恢复后 step 接着 ckpt 的值，不从头。是"续训"不是"重训"。
- **ckpt 间隔决定 RPO**：见 §4.3。间隔越短丢得越少但 IO 开销越大。

### 3.6 与固定训练的对比

| 维度 | 固定训练（DDP） | 弹性训练（torchrun elastic） |
|---|---|---|
| world_size | 常量（启动钉死） | 变量（rendezvous 协商，运行时变） |
| 节点数参数 | `--nnodes=4`（单值） | `--nnodes=min:max`（范围） |
| 任一 worker 崩 | allreduce 死锁，整 job 重启 | re-rendezvous，缩容续训 |
| 新节点加入 | 不支持（world 固定） | rendezvous 加入，扩容 |
| 数据切分 | 启动时按 world 切 | 成员变更后按新 world 重切 |
| 启动门槛 | 全部到齐才起 | min_nodes 到齐即起 |
| 停顿 | 无（除非崩） | 成员变更时秒级停顿 |
| 适用 | 固定集群、强一致 HPC | 云上抢占、滚动升级、故障多 |
| 复杂度 | 简单 | 高（rendezvous + ckpt + sampler 重切） |
| 容错 | 弱（崩即死） | 强（崩即缩） |
| 成本 | 高（不能用 spot） | 低（能用 spot） |

### 3.7 rdzv_backend 选型

| backend | 依赖 | 适用 | 备注 |
|---|---|---|---|
| `c10d` | 无（PyTorch 自带） | 生产首选 | 一个进程当 rendezvous 主，需 rdzv_endpoint 可达 |
| `etcd` | 部署 etcd 集群 | 旧推荐、大规模 | 强一致 KV，但运维 etcd 增成本，新版渐被 c10d 取代 |
| `file` | 共享 FS（NFS） | 单机/小集群 | NFS 一致性/性能坑，跨机慎用 |
| `tcp` | 无 | 静态弹性弱场景 | 类似 `init_method='tcp://'`，弹性弱 |


## 4. 数学原理 / 公式

### 4.1 quorum（法定人数）判定

设当前报名 worker 数 $N$，min/max 为 $\text{min}, \text{max}$。rendezvous 状态：

$$\text{state}(N) = \begin{cases}
\text{pending (等)} & N < \text{min} \\
\text{active (起训, } W=N\text{)} & \text{min} \le N \le \text{max} \\
\text{active } (\text{取 max 个}, \text{ 余 standby}) & N > \text{max}
\end{cases}$$

active 时 world_size $W = \min(N, \text{max})$。$N<\text{min}$ 时**不退出不报错，只 pending**——等凑齐 min。这是"软等待"而非"硬失败"。

训练中 active worker 数 $N'$ 降到 $<\text{min}$：

$$\text{transition: active} \to \text{standby} \quad (N' < \text{min})$$

job 不退出，等新成员凑齐 min 再 re-rendezvous。把"少数 worker 故障"降级成"暂停等恢复"。

### 4.2 成员变更的 allreduce 重组

固定训练 allreduce：$W$ 个 rank，每 rank grad $\mathbf{g}_i$，allreduce 得平均 $\bar{\mathbf{g}} = \frac{1}{W}\sum_{i=0}^{W-1}\mathbf{g}_i$，全员拿到 $\bar{\mathbf{g}}$，`optimizer.step()` 同步推进。

成员变更（$W \to W'$，如 $W=32$ 一 rank 退 → $W'=31$）后**重组 group**：

$$\bar{\mathbf{g}}' = \frac{1}{W'}\sum_{i \in \text{survivors}} \mathbf{g}_i$$

数学上只是参与求和的 rank 变了，但工程上 NCCL communicator 必须重建（旧 communicator 绑了死 rank 集，新成员加不进）。重建 = 重新 `init_process_group` + 新 NCCL communicator 初始化（走 IB/NCCL 握手，秒级）。这就是"re-rendezvous 代价"。

> [!note] 推论：为什么不能"热加 rank"
> DDP 的 NCCL communicator 在 `init_process_group` 时钉死 rank 集。新 rank 加进来要新 communicator——必须所有 rank 一起重建（没法只加一个）。这是"冷重组"而非"热修补"。所以弹性训练的扩缩容都是"全员 re-rendezvous + 重组"，不是渐进加。

### 4.3 RPO（数据丢失）= ckpt 间隔

设 ckpt 间隔 $T_{\text{ckpt}}$（如每 100 步或每 10 分钟），单步耗时 $\tau$。成员变更时丢的"训练进度"：

$$\text{RPO} = T_{\text{since\_last\_ckpt}} \in [0, T_{\text{ckpt}}]$$

即最多丢一个 ckpt 间隔的步数（$\le \lceil T_{\text{ckpt}} / \tau \rceil$ 步）。期望丢 $\approx T_{\text{ckpt}}/2$（均匀分布）。$T_{\text{ckpt}}$ 越短 RPO 越小（丢得少）但 IO 开销越大（每 $T_{\text{ckpt}}$ 停一次写盘）。详见 [[async distributed checkpoint]] 的异步 ckpt 减 IO 阻塞。

### 4.4 RTO（恢复时间）= rendezvous + ckpt load + NCCL 重组

成员变更到恢复训练的停顿时间：

$$\text{RTO} \approx T_{\text{re-rendezvous}} + T_{\text{ckpt\_load}} + T_{\text{nccl\_init}} + T_{\text{sampler\_reset}}$$

典型量级：re-rendezvous 秒级、ckpt load（大模型几十 GB）分钟级、NCCL init 秒到十秒级、sampler reset 毫秒级。大模型 RTO 主要被 **ckpt load** 主导（几十 GB 从 NFS/PVC 读）。减 RTO 用 [[async distributed checkpoint]] 异步 ckpt 或 RAM disk。详见 [[故障检测与恢复]] 的 RTO/RPO 节。


## 5. 代码示例（可选）

### 5.1 torchrun elastic 启动命令

```bash
# 最小弹性: 2 节点起训, 上限 8 节点
torchrun \
  --nnodes=2:8 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=10.0.0.10:29500 \
  --rdzv_id=my-elastic-job-$(date +%s) \
  --max-restarts=3 \
  train_elastic.py \
    --batch-size 32 --lr 1e-4
```

### 5.2 elastic-aware 训练脚本（含 ckpt 恢复）

```python
import os, torch, torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

@record()                                  # torchrun agent 抓异常上报
def main():
    # 1. rendezvous (torchrun 已注入 env: RANK/WORLD_SIZE/MASTER_ADDR)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % 8)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[elastic] rank={rank} world_size={world} (动态)")

    # 2. 模型 + DDP
    model = build_model().cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 3. 数据 (按动态 world 切分)
    dataset = build_dataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler,
                                         num_workers=8, pin_memory=True)

    # 4. 从 ckpt 恢复 (成员变更后全员 reload)
    ckpt_path = "/shared/ckpt.pt"
    step = 0
    if rank == 0 and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        step = ckpt["step"]
        sampler.set_epoch(step // len(loader))  # 对齐 epoch
        print(f"[elastic] recovered from step {step}, new world={world}")
    dist.barrier()                          # 等 rank 0 load 完

    # 5. 训练 (周期 ckpt)
    for batch in loader:
        if step > 0 and step % 100 == 0:    # 每 100 步 ckpt
            if rank == 0:
                torch.save({"model": model.state_dict(),
                            "optim": optimizer.state_dict(),
                            "step": step}, ckpt_path)
            dist.barrier()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

if __name__ == "__main__":
    main()
```

> [!tip] 实践：成员变更后 sampler.set_epoch
> `DistributedSampler` 用 epoch 确定切分种子。成员变更后 world_size 变了，若不 `set_epoch`，sampler 可能重复切到旧 world 的样本范围。恢复后 `set_epoch(step // len(loader))` 让切分对齐当前进度，避免重复训。

### 5.3 Python API 起 elastic agent（不用 CLI torchrun）

```python
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

cfg = LaunchConfig(
    min_nodes=2,
    max_nodes=8,
    nproc_per_node=8,
    run_id="my-elastic-job-123",
    rdzv_config={"backend": "c10d", "endpoint": "10.0.0.10:29500"},
    max_restarts=3,
)
elastic_launch(cfg, main_fn)(arg1, arg2)
# 在 Python 里起 elastic agent, 等价 torchrun CLI, 但能嵌进 Ray/Airflow 调度
```

这是 [[Ray与分布式调度]] 的 `torch.distributed.run` 集成方式——Ray Train 用 elastic_launch 包 DDP，把 elastic 的成员管理与 Ray 的 actor 调度统一。

### 5.4 K8s + Training-Operator + elastic

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata: { name: elastic-train }
spec:
  pytorchReplicaSpecs:
    Worker:
      replicas: 4                   # 初始 4 副本 (弹性可扩缩)
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: myregistry/train:elastic
            command:
            - torchrun
            - --nnodes=2:8          # 弹性范围 (K8s HPA 扩 Pod 时自动加)
            - --nproc_per_node=8
            - --rdzv_backend=c10d
            - --rdzv_endpoint=elastic-train-worker-0:29500
            - --rdzv_id=elastic-job-1
            - --max-restarts=3
            - train_elastic.py
            resources:
              limits: { nvidia.com/gpu: 8 }
          volumes:
          - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 64Gi } }
          - { name: ckpt, persistentVolumeClaim: { claimName: ckpt-pvc } }
```

配 K8s HPA 或 Cluster Autoscaler：节点池空闲 → 扩 Pod → standby → re-rendezvous 加入；节点被回收 → Pod 杀 → torchrun 检测缩 → re-rendezvous 续训。这是"云原生弹性训练"全栈。详见 [[Docker与Kubernetes]] §3.5。


## 6. 与其他知识点的关系

- **上游（依赖）**: [[torch.distributed]]（init_process_group/rendezvous 底层）、[[rank与world size]]（rendezvous 协商 rank/world）、[[Distributed Data Parallel]]（弹性是 DDP 的"可变 world"扩展，allreduce 重组）、[[Docker与Kubernetes]]（K8s HPA/抢占 + Training-Operator 跑弹性训练）、云 spot 实例（弹性存在的动机）。
- **下游（应用）**: [[故障检测与恢复]]（worker 崩溃的"自愈"——torchrun elastic 是恢复手段之一）、云原生弹性大模型训练（spot + elastic + ckpt）、[[RL worker故障恢复]]（RL 框架的 rollout worker 故障用类似 rendezvous 重连）、[[async distributed checkpoint]]（弹性恢复的 ckpt 后端，减 RPO/RTO）。
- **对比 / 易混**:
  - **elastic vs 固定训练**：见 §3.6 表。前者 world 可变，后者常量。弹性容错强但有 re-rendezvous 代价。
  - **elastic vs fault-tolerant DDP（FT-DDP）**：FT-DDP 是 PyTorch 实验性的"热修补"——成员变更不全员 re-rendezvous，只重组受影响的 communicator，停顿更短。elastic 是"冷重组"全员 re-rendezvous。FT-DDP 更先进但成熟度低，待核实生产可用。
  - **rdzv_backend: c10d vs etcd**：c10d 无外部依赖（推荐），etcd 需部署 etcd 集群（旧）。功能等价，运维成本不同。
  - **elastic training vs [[async distributed checkpoint]]**：前者是"成员动态 + 续训"机制，后者是"ckpt 不阻塞训练"技术。正交（弹性训练用异步 ckpt 减 RTO）。前者管成员，后者管 ckpt IO。


## 7. 常见误区与易错点

> [!warning] 误区 1：弹性训练 = 多卡能并行变快
> 不。弹性训练是"成员可变 + 续训"，不是"加卡提速"。加卡后吞吐提升是"同成本下用更多 spot 卡跑更多步"，但单步不算更快（DDP allreduce 通信开销随 world 增）。弹性是省钱/容错，不是加速单步。

> [!warning] 误区 2：成员变更后只有新 rank 恢复 ckpt
> 错。全员都 re-rendezvous + reload。因为 allreduce communicator 重建，旧 optimizer state 不对应新 world，最稳是全员 reload（除非用 sharded optimizer state，每 rank 只 reload 自己 shard）。这是"冷重组"非"热修补"。

> [!warning] 误区 3：re-rendezvous 没开销
> 有。秒到分钟级停顿（NCCL init + ckpt load + sampler reset）。频繁成员变更会让吞吐抖动。设计 spot 策略要平衡"省钱"与"抖动"——太激进抢 spot 会频繁 re-rendezvous 反损。

> [!warning] 误区 4：`--nnodes=4` 就是弹性
> 不。单值 `--nnodes=4` = `4:4` 固定，等价传统。弹性要 `min:max` 范围（如 `2:8`）。看冒号是关键。

> [!warning] 误区 5：min_nodes 设 1 即可
> 可但傻。min=1 意味着"单节点也能起训"，但单节点训练的 allreduce 没意义（无通信），且成员频繁降到 1 时实际退化单机训练，吞吐崩。min 应设到"训练有效性的下限"（如最少 2-4 节点才有 allreduce 意义）。

> [!warning] 误区 6：torchrun elastic 自动处理 OOM
> 不。elastic 处理"worker 进程退出"（被回收/崩/kill）。OOM 崩了会触发 re-rendezvous，但 OOM 本身不解决。配 [[故障检测与恢复]] 的 OOM 监控 + 降 batch 重启更稳。

> [!warning] 误区 7：rdzv_endpoint 用 localhost
> 错（跨节点时）。`c10d` 的 rdzv_endpoint 必须所有 worker 都可达。跨节点用某节点的内网 IP（如 `10.0.0.10:29500`）。localhost 只对单机 elastic 有效。这是常见连接失败原因。

> [!warning] 误区 8：弹性训练的 ckpt 与固定训练一样
> 不完全。固定训练 ckpt 是"给重启用"，弹性 ckpt 是"给 re-rendezvous 用"，且**每 rank 都要能读完整 ckpt 或自己 shard**（[[async distributed checkpoint]] 的 sharded 形式更适合弹性，因 reload 自己 shard 快）。弹性 + sharded ckpt 是大模型标配。


## 8. 延伸细节

### 8.1 torch.distributed.elastic 的 agent 模型

torchrun 本身是个 **elastic agent**——它在每个节点上跑一个 agent 进程，agent 监控本节点的 worker 进程（nproc_per_node 个）。agent 职责：① 起本节点 worker；② 监控 worker 健康；③ worker 崩时按 `max-restarts` 重启；④ 与其他节点 agent 通过 rdzv_backend 协调 re-rendezvous。这是"两层架构"——agent 管本节点 worker，rdzv_backend 管跨节点协调。详见 PyTorch `torch.distributed.elastic.agent` 源码。

### 8.2 与 Ray Train 的集成

[[Ray与分布式调度]] 的 Ray Train 用 `torch.distributed.run` 的 elastic_launch 包 DDP——Ray actor 当 worker，Ray 调度当 agent，rendezvous 走 c10d。Ray 的优势是 actor placement 可感知 NUMA/GPU 拓扑，且 Ray 自带容错（actor 死 Ray 重启），与 elastic 的 re-rendezvous 互补。这是"Ray 原生弹性训练"路径，与 K8s + Training-Operator 路径并存。

### 8.3 FT-DDP（Fault-Tolerant DDP）

PyTorch 实验性的 FT-DDP（`torch.distributed.elastic` 的进阶）做"热修补"——成员变更不全员 re-rendezvous，只重组受影响 communicator，停顿更短。比 elastic 更先进，但成熟度低于 elastic，大模型生产仍以 elastic + ckpt 为主。待核实 FT-DDP 最新状态。

### 8.4 MaaS / 大厂弹性训练平台

大厂自研的弹性训练平台（如字节 Steel、阿里 PACP、腾讯 Angel）多在 torchrun elastic 基础上加：① 自研 scheduler 抢占感知（spot 回收前提前告警让 worker 优雅存 ckpt 退出）；② sharded ckpt + RAM disk 减 RTO；③ 数据 reshard 自动化。把 elastic 的 re-rendezvous 代价压到秒级。

### 8.5 与 RL 框架的衔接

[[RL worker故障恢复]] 里 RL 的 rollout worker（actor 跑环境）故障率高于 trainer（模型推理轻、环境多样易崩）。弹性 rendezvous 思路用到 RL——rollout worker 崩了 trainer 不死，重新起 rollout worker + rendezvous + 从 rollout ckpt 恢复轨迹。GRPO/PPO RL 框架多借鉴 elastic 的 re-rendezvous 设计。

### 8.6 内容来源

elastic 概念整理自 PyTorch 官方文档 `torch.distributed.elastic`、`torchrun` 文档、`torch.distributed.launcher` API、PyTorch elastic design doc（rendezvous 协议、agent 模型、min/max nodes 语义）、Kubeflow Training-Operator 的 `PyTorchJob` elastic 示例、社区 spot + elastic 实践文章。FT-DDP 与 Ray Train 集成的最新状态待核实。关联见 [[故障检测与恢复]]、[[Docker与Kubernetes]]、[[async distributed checkpoint]]、[[rank与world size]]、[[Distributed Data Parallel]]、[[NCCL hang与timeout]]、[[Ray与分布式调度]]、[[RL worker故障恢复]]。

---
相关: [[部署与容错]] | [[Docker与Kubernetes]] | [[故障检测与恢复]] | [[async distributed checkpoint]] | [[rank与world size]] | [[torch.distributed]] | [[Distributed Data Parallel]] | [[NCCL hang与timeout]] | [[Ray与分布式调度]] | [[RL worker故障恢复]] | [[Python multiprocessing与GIL]]
