# weight sync mechanism

> **所属章节**: [[训推分离]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（需懂分布式通信 + 版本管理）

## 1. 一句话定义

**weight sync mechanism（参数同步机制）** 是 [[训推分离]] 中训练侧（[[policy training (PT)]] / learner）向采样/部署侧（[[rollout worker]] / [[policy deployment (PD)]]）**推送新 $\theta$ 的机制**，用以减小 [[stale policy problem]]、保持版本一致。核心权衡：**全量 vs 增量**（差分/压缩）、**push vs pull**、**同步频率 $S$**（减 stale vs 带宽开销）。$\theta$ 大（GB 级）带来带宽挑战，常用 [[NCCL通信拓扑]] AllGather / 参数服务器 / RPC + 压缩实现。PD 侧配合**热切换**（blue-green/canary）零停机升级。它是 PT-PD/learner-worker 的"参数动脉"，verl/OpenRLHF/DeepSpeed-Chat 都有 weight broadcast 实现。

> [!note] 三种 sync 场景
> - **learner → worker**（训练内部）：异步训练中 learner 把新 $\theta$ 推 worker，减 stale data。
> - **PT → PD**（部署）：PT 训新版本推 PD，PD 热切换升级服务。
> - **worker → learner**（反向，本篇不重点）：worker 采的数据回传 learner（数据 sync，非 weight）。

> [!note] 解答：现在主流的权重传输方法（2025–2026 联网调研）
> 用户问"现在主流传输方法是什么？checkpoint engine？"——**checkpoint engine 确实是主流方法之一**（SGLang 主推），但**最主流的 RLHF 热路径 sync 仍是 NCCL broadcast**。下面按场景把当前业界四条主流路线讲清。
>
> ### 路线一：NCCL broadcast（训练热路径绝对主流）
> **适用**：trainer↔rollout engine 同集群、需低延迟高频 sync（RLHF 每步/每 batch 推一次）。
> - **vLLM 原生 `WeightTransferEngine`**（RFC #31848 已落地，见 `vllm/distributed/weight_transfer/nccl_engine.py`）：trainer rank 0 与所有 vLLM worker 组一个 `StatelessProcessGroup`（torch.distributed 无关的轻量组），trainer broadcast 权重、worker 接收加载。四阶段协议 `init_weight_transfer_engine` → `start_weight_update` → `update_weights`（可分块多次）→ `finish_weight_update`。**packed tensor** 把多个小张量打包成大连续 buffer + 双/三缓冲 + 独立 CUDA stream 做 pack/broadcast/unpack overlap，吞吐显著提升（实现参考 NeMo-RL 的 packed tensor）。支持多机多卡、NVLink/IB。还派生 `sparse_nccl` 后端做稀疏 in-place patch（TP=PP=1 MVP）。
> - **OpenRLHF**：`--vllm.sync_backend nccl`，trainer 每 step 后 broadcast 给 vLLM；Hybrid Engine 模式下 vLLM sleep/wake 切换，sync 用 NCCL；async+partial rollout 模式下 vLLM 不停，pause 在飞请求→swap 权重→resume（混入少量 off-policy 噪声换全 overlap）。
> - **verl**：`WeightSyncClient/Server` + NCCL broadcast，每 episode batch sync。
> - **为什么主流**：GPU 间 NVLink/IB 带宽最高（数百 GB/s），broadcast 一对多语义正好匹配"trainer 单源推多 worker"，延迟毫秒级，是唯一能扛住 RLHF 每 step sync 的方案。本笔记 §3.4/§8.1 写的 broadcast vs AllGather 即此。
>
> ### 路线二：CUDA IPC（同机 hybrid engine 专用）
> **适用**：trainer 与 vLLM **同一台机器同卡组** colocate（Hybrid Engine 模式）。
> - OpenRLHF `ColocateWorkerExtension.update_weights_from_ipc_handles`：trainer 把权重张量的 CUDA IPC handle 传给 vLLM worker，worker 通过共享内存句柄直接拿到显存指针 `load_weights`，**零拷贝**。
> - 比 NCCL broadcast 更快（同机不走网卡），但**只限同机**；跨机退回 NCCL。
> - vLLM 的 `WeightTransferEngine` 也把 IPC 作为可插拔 backend 之一（`backend="ipc"`）。
>
> ### 路线三：Checkpoint Engine（SGLang 主推，冷启动+多机加载加速）
> **适用**：**PD 部署侧冷启动/大模型多机加载**，以及非 colocated rollout 的权重更新。**这正是用户问的那个。**
> - SGLang 的 **checkpoint engine**（`sglang.srt.checkpoint_engine`）是一个**分布式 checkpoint 加载系统**：把权重加载并行化到多进程多节点，每节点只从磁盘读一部分→有效放大磁盘带宽；`--load-format dummy` 让磁盘→CPU 搬运与 CUDA graph 捕获等其他初始化**overlap**。三种 update method：`broadcast`（加载进程 broadcast 到推理进程）、`p2p`（点对点直传）、`all`（两者结合）。实测 DeepSeek-R1 on H20-3e 双节点加载加速 ~20s。
> - **SGLang 还在整合 ckpt-engine 做 weight sync 加速**（issue #10464, 2025-09），目标是用它加速**在线权重同步**而非只是冷启动。
> - **本质区别**：NCCL broadcast 是"GPU 显存→GPU 显存"的热路径流式传输；checkpoint engine 是"磁盘/共享存储→多机并行加载→推理进程"的冷/温路径加载加速器。前者管"每步推一次"，后者管"快速把一个新版本从盘上搬起来"。**PD 升级、新副本拉起、大模型多机首次加载**用 checkpoint engine；**RLHF 训练中每 step sync** 用 NCCL。
>
> ### 路线四：共享文件系统 delta pull（非 colocated rollout / PD 多副本）
> **适用**：trainer 与 rollout/PD 不在同一集群、靠共享存储解耦。
> - SGLang `/pull_weights`（PR #30366）：trainer 每次权重更新发布一个版本目录 `weight_v{N:06d}/` 到共享 FS——可以是完整 HF checkpoint，也可以是 **zstd 压缩的 per-tensor byte delta**（xor/overwrite + per-tensor checksum）。engine 端 `POST /pull_weights` 把本机 checkpoint 拉到目标版本：从 ≤target 的最新 full 版起播，再按序 in-place mmap apply delta 链，跨 tensor 并行；checksum 不符或乱序立即 raise（绝不服务坏权重）；同机多 rank 用 flock 折叠成一次 pull。trainer 只需对接每 engine 一个 endpoint。实测 tp16 跨 2 节点、3 次 delta sync（~0.4% 密度、~0.7–0.8GB 线上）全 checksum 通过。
> - 之后 engine 用普通 `/update_weights_from_disk` reload——weight loader 完全不感知 delta 格式。
> - 这就是本笔记 §3.2 的 **pull 模式** + §3.1 的**增量/差分**在工业界的落地形态。
>
> ### 四条路线对比
> | 路线 | 传输介质 | 典型场景 | 延迟 | 带宽 | 代表实现 |
> |---|---|---|---|---|---|
> | **NCCL broadcast** | GPU↔GPU (NVLink/IB) | RLHF 每 step sync、colocated | 毫秒级 | 数百 GB/s | vLLM `WeightTransferEngine`/OpenRLHF/verl |
> | **CUDA IPC** | 同机共享显存句柄 | Hybrid Engine 同机 colocate | 微秒级、零拷贝 | 显存带宽 | OpenRLHF `update_weights_from_ipc_handles` |
> | **Checkpoint Engine** | 磁盘→多机并行加载 | PD 冷启动/大模型多机首载/版本升级 | 秒~十秒级 | 放大磁盘带宽 | SGLang `checkpoint_engine` (broadcast/p2p/all) |
> | **共享 FS delta pull** | 共享存储 + zstd delta | 非 colocated rollout/PD 多副本 | 秒级 | 增量后 ~0.7GB/次 | SGLang `/pull_weights` |
>
> ### 还有：HTTP 控制面 + NCCL 数据面（在线 serving）
> vLLM 还支持 **HTTP 控制面 + NCCL 数据面**分离：`POST /init_weight_transfer`、`POST /update_weights`、`POST /finalize_weight_update` 走 HTTP 下发元数据（names/shapes/dtype），实际张量走 NCCL broadcast。这样在线 serving 的 vLLM HTTP server 也能做高频 weight sync，不必停服。见 vLLM `examples/rl/rlhf_http_nccl.py`。
>
> ### 与本笔记的映射
> - 本笔记 §3.4 列的"NCCL AllGather / 参数服务器 / RPC+存储 / 压缩"四项，对应：NCCL broadcast=路线一、参数服务器≈路线三的 checkpoint engine（CPU 内存中转）、RPC+存储=路线四的共享 FS pull、压缩=路线四的 zstd delta。**笔记里写的方向是对的，只是 2025 后业界把它们工程化成了具体产品**（vLLM WeightTransferEngine / SGLang checkpoint engine / SGLang /pull_weights）。
> - **结论**：RLHF 训练热路径主流 = **NCCL broadcast**（vLLM/OpenRLHF/verl 都用）；PD 冷启动/多机加载主流 = **checkpoint engine**（SGLang 主推，用户猜对了）；非 colocated rollout = **共享 FS delta pull**；同机 colocate = **CUDA IPC**。四者按场景共存，不是互斥。
>
> **来源**：vLLM RFC #31848 / `docs/training/weight_transfer/nccl.md` / `docs/training/layerwise/`；SGLang issue #10464 / `checkpoint-engine.html` / PR #30366 / `pd_disaggregation.md`；OpenRLHF `architecture.html` / `async_training.html` / `hybrid_engine.rst` / vLLM blog 2025-04-23。截至 2026-07。

## 2. 为什么需要它（动机与背景）

异步 [[训推分离]] 必然产生 [[stale policy problem]]：
1. **必然 stale**：worker/PD 与 learner/PT 时间解耦，lag 期间 $\theta$ 已更新；
2. **需主动 sync**：不 sync 则 stale 累积，ratio 失真，训练不稳/服务落后；
3. **$\theta$ 大有带宽挑战**：LLM 的 $\theta$ GB~TB 级，全量 sync 带宽/延迟重；
4. **频率 tradeoff**：频繁 sync 减 stale 但开销大，稀疏则 stale 重；
5. **PD 需零停机**：PD 服务中升级 $\theta$，需热切换不中断；
6. **版本管理**：多副本 PD 需版本一致，避免部分副本旧部分新。

weight sync mechanism 是这些挑战的工程解：选合适模式（全量/增量、push/pull）、频率 $S$、压缩、热切换协议，把 stale 控制在可容忍范围（clip frequency 0.1~0.3）。

## 3. 核心概念详解

### 3.1 全量 vs 增量 sync

| 模式 | 机制 | 带宽 | 适用 |
|---|---|---|---|
| **全量 sync** | 推整个 $\theta$ | $O(P)$（$P$ 参数量） | 首次加载、版本大变、简单可靠 |
| **增量 sync** | 推 $\Delta\theta=\theta_{\text{new}}-\theta_{\text{old}}$ | $O(\|\Delta\theta\|_0)$（稀疏） | 连续小步更新（如每 N 步） |
| **差分+压缩** | $\Delta\theta$ 量化/稀疏化/TopK | $O(\text{压缩比}\cdot P)$ | 带宽紧、大模型 |
| **层分片 sync** | 分层轮流 sync，减单次带宽 | $O(P/L)$ per sync | 平滑带宽 |

LLM 训练 $\theta$ 变化连续（小步），增量/压缩有效；版本大变（如 SFT→RLHF 切换）用全量。

### 3.2 push vs pull 模式

| 模式 | 机制 | 触发 | 适用 |
|---|---|---|---|
| **push** | learner/PT 主动推 $\theta$ 给 worker/PD | learner 训到一定步数 | 异步训练（learner 主导） |
| **pull** | worker/PD 主动拉 $\theta$ | worker 需要时 | PD 升级、worker 启动 |
| **hybrid** | push 通知 + pull 拉取 | push 通知有新版本，worker/PD 按需 pull | 大 $\theta$ + 按需升级 |

PD 常用 pull（按升级计划拉 checkpoint）；训练内部常用 push（learner 主导同步）。

### 3.3 同步频率 $S$ 的 tradeoff

$$
\text{最大 staleness}\le S\text{（两次 sync 间步数）}
$$

- **$S$ 小（频繁）**：stale 轻，ratio 准，但 sync 开销大（带宽/延迟），打断 worker 生成；
- **$S$ 大（稀疏）**：stale 重，ratio 失真，但 sync 开销小；
- **调参信号**：监控 clip frequency（维持 0.1~0.3）、approx_kl、staleness 分布尾部；
- **典型 $S$**：异步 RLHF 中每 10~100 训练步 sync 一次（verl 默认每 episode batch sync）。

### 3.4 带宽与压缩

$\theta$ 大（70B 模型 ~140GB FP16），全量 sync 带宽挑战：
- **NCCL AllGather**：GPU 间高速集合通信（IB/NVLink）；
- **参数服务器**：CPU 内存中转，带宽受限网络；
- **RPC + 存储**：PT 写 checkpoint 到共享存储，PD/worker 拉（适合 PD，非训练热路径）；
- **压缩**：FP16→INT8/INT4、TopK 稀疏、差分编码，减带宽（损精度可控）；
- **分层 sync**：分片轮流，平滑带宽。

### 3.5 PD 的热切换

PD 升级 $\theta$ 需零停机：
- **blue-green**：新版本副本起来验证后切流量，旧版本待命回滚；
- **canary**：1% → 10% → 50% → 100% 渐进切，异常回滚；
- **shadow**：新版本影子跑（不返用户），对比指标；
- **预热**：新副本加载 $\theta$ + KV cache 预热后再切流量；
- **回滚**：异常秒级回退（旧副本待命）。

### 3.6 框架实现

| 框架 | weight sync 实现 |
|---|---|
| **verl** | `WeightSyncClient/Server`，NCCL broadcast，每 episode batch sync worker |
| **OpenRLHF** | `torch.distributed` broadcast，trainer → rollout 节点 |
| **DeepSpeed-Chat** | checkpoint save/load，PT 写存储 PD 拉 |
| **NeMo-Aligner** | 类似，支持增量同步 |

## 4. 数学原理 / 公式

### 4.1 sync 后的 staleness 上界

$$
k_{\max}\le S\text{（两次 sync 间隔的训练步数）}
$$

- $S=1$：每步 sync，$k_{\max}=1$（近同步）；
- $S=100$：每 100 步 sync，$k_{\max}\le100$。

### 4.2 带宽开销

$$
B=\frac{P\cdot\text{size}}{S\cdot\Delta t}\cdot\text{压缩比}^{-1}
$$

- $P$：参数量，size：每参数字节（FP16=2）；
- $S\cdot\Delta t$：sync 间隔时间；
- 压缩比：增量+量化减带宽。

例：70B FP16，$S=50$ 步，每步 1s，全量 → $B=140\text{GB}/50\text{s}=2.8\text{GB/s}$；INT8 差分（压缩 4×）→ $0.7\text{GB/s}$。

### 4.3 增量 sync 的差分

$$
\Delta\theta=\theta_{\text{new}}-\theta_{\text{old}},\quad\theta_{\text{new}}=\theta_{\text{old}}+\Delta\theta
$$

- $\Delta\theta$ 稀疏（小步更新多数分量接近 0）；
- TopK 保留大分量，减传输量；
- 量化 $\Delta\theta$（FP16→INT8）进一步减。

### 4.4 sync 的开销-收益

$$
\text{收益}=\text{stale 减少带来的训练稳定性}\quad\text{开销}=\text{带宽 + sync 延迟 + 中断}
$$

最优 $S$ 平衡两者，常通过监控 clip frequency 调。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.distributed as dist, copy

class LM(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)

V, D = 50, 32

# ===== 权重同步(模拟,实际用 dist.broadcast/AllGather) =====
class WeightSyncServer:
    """learner/PT 侧:推 θ"""
    def __init__(self, model):
        self.model = model
        self.version = 0
    def get_full_weights(self):
        """全量 θ"""
        self.version += 1
        return copy.deepcopy(self.model.state_dict()), self.version
    def get_delta_weights(self, old_state):
        """增量 Δθ(差分)"""
        new_state = self.model.state_dict()
        delta = {k: new_state[k] - old_state[k] for k in new_state}
        self.version += 1
        return delta, self.version, new_state

class WeightSyncClient:
    """worker/PD 侧:接收 θ"""
    def __init__(self):
        self.model = LM(V, D)
        self.version = 0
    def apply_full(self, state, version):
        """全量加载"""
        self.model.load_state_dict(state)
        self.version = version
        print(f"worker 全量 sync 到 v{version}")
    def apply_delta(self, delta, version):
        """增量应用 θ_new = θ_old + Δθ"""
        state = self.model.state_dict()
        for k in state: state[k] += delta[k]
        self.model.load_state_dict(state)
        self.version = version
        print(f"worker 增量 sync 到 v{version}(带宽省)")

# ===== 训练内部 sync(learner → worker) =====
learner_model = LM(V, D)
server = WeightSyncServer(learner_model)
worker = WeightSyncClient()

# worker 首次全量拉
state, ver = server.get_full_weights()
worker.apply_full(state, ver)

# learner 训练若干步
opt = torch.optim.Adam(learner_model.parameters(), lr=0.01)
for _ in range(10):
    x = torch.randint(0, V, (4, 8))
    loss = learner_model(x).float().mean()
    opt.zero_grad(); loss.backward(); opt.step()

# 增量同步(省带宽)
old_state = worker.model.state_dict()
delta, ver, _ = server.get_delta_weights(old_state)
worker.apply_delta(delta, ver)
print(f"staleness 已重置,worker v{worker.version} ≈ learner v{server.version}")

# ===== PD 热切换(PT → PD) =====
class PDDeployment:
    def __init__(self):
        self.model = LM(V, D); self.version = 0; self.old_model = None
    def hot_swap(self, new_state, new_ver):
        """blue-green 热切换:旧版本待命回滚"""
        self.old_model = copy.deepcopy(self.model)  # 旧版本待命
        self.old_version = self.version
        self.model.load_state_dict(new_state)
        self.version = new_ver
        print(f"PD 热切换: v{self.old_version} → v{self.version}(零停机,可回滚)")
    def rollback(self):
        if self.old_model:
            self.model = self.old_model
            self.version = self.old_version
            print(f"PD 回滚到 v{self.version}(异常恢复)")

pd = PDDeployment()
new_state, new_ver = server.get_full_weights()
pd.hot_swap(new_state, new_ver)
pd.rollback()  # 模拟异常回滚
print("weight sync 完成:训练内部减 stale + PD 热切换零停机")
```

> [!tip] 实际工程
> - **训练内部**：verl 用 `WeightSyncClient/Server` + NCCL broadcast，每 episode batch sync；
> - **PD 升级**：PT 写 checkpoint 到共享存储（如 SFS），PD pull + blue-green（Kubernetes Argo Rollouts）；
> - **大模型**：增量 + INT8 量化 + 分层 sync 减带宽。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[训推分离]]、[[asynchronous training]]、[[stale policy problem]]（sync 解决的问题）、[[policy training (PT)]]/[[policy deployment (PD)]]、[[learner worker split]]、[[rollout worker]]/[[inference worker]]。
- **下游（应用）**: 异步 RLHF 框架（verl/OpenRLHF/DeepSpeed-Chat）、PD serving（vLLM + Kubernetes）、分布式通信（[[NCCL通信拓扑]]/[[AllGather]]，待展开）、参数服务器。
- **对比 / 易混**:
  - **weight sync（本篇）vs gradient sync**：前者训练侧→采样/部署侧推 $\theta$；后者数据并行 worker→learner 聚合梯度（[[Data Parallel]]/[[Distributed Data Parallel]]）。方向与内容不同。
  - **weight sync vs checkpoint load**：前者流式/增量同步（热路径）；后者离线加载（冷启动）。机制不同。
  - **weight sync vs data sync**：前者参数；后者 worker 采的数据回传 learner（数据 sync）。互补。
  - **全量 vs 增量 sync**：见 §3.1。带宽/适用不同。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"sync 免费"**：错。$\theta$ 大，带宽/延迟/中断开销重，需 tradeoff 频率。
> 2. **"全量总低效"**：错。版本大变/首次加载用全量更简单可靠；增量适合连续小步。
> 3. **"热切换瞬切"**：错。需新副本预热 + 渐进切流量，非瞬间。
> 4. **"sync 越频繁越好"**：错。频繁减 stale 但开销大 + 打断 worker 生成，有最优 $S$。
> 5. **"push 一定优于 pull"**：错。PD 常用 pull（按升级计划）；训练内部 push 多。
> 6. **"增量无精度损"**：错。差分+量化有损，需可控（如 INT8 差分损小）。
> 7. **"PD 多副本自动版本一致"**：错。需 sync 协议保证，否则部分旧部分新。
> 8. **"weight sync 解决一切 stale"**：减 stale 但不消除（异步本质），需 + clip/V-trace 协同。
> 9. **"sync = DDP AllReduce"**：错。AllReduce 是梯度聚合（训练数据并行）；weight sync 是参数推送（训推分离）。不同。

## 8. 延伸细节

### 8.1 NCCL broadcast vs AllGather

- **broadcast**：一节点 $\theta$ 广播到所有节点（$O(P)$ 带宽，适合 weight sync）；
- **AllGather**：所有节点聚全（适合梯度聚合后分发）；
- weight sync 常用 broadcast（learner/PT 单源推送）。

### 8.2 参数服务器（parameter server）

- learner/PT 把 $\theta$ 写参数服务器（CPU 内存）；
- worker/PD 从参数服务器拉/pull；
- 适合 PD 升级（按需 pull），非训练热路径；
- 带宽受限网络（vs NCCL 的 IB/NVLink）。

### 8.3 差分 + 压缩

- **差分**：$\Delta\theta$ 稀疏（小步更新），TopK 保留大分量；
- **量化**：$\Delta\theta$ FP16→INT8，损小；
- **编码**：哈夫曼/游程编码进一步压；
- 大模型（70B+）必用，否则带宽扛不住。

### 8.4 verl 的 weight broadcast

verl 的实现（参考）：
- `WeightSyncServer` 在 trainer，每 episode batch 后 broadcast $\theta$；
- `WeightSyncClient` 在 rollout worker，接收并加载；
- 用 NCCL broadcast（IB 网络）；
- 支持增量（差分）模式减带宽。
OpenRLHF 类似，用 `torch.distributed.broadcast`。

### 8.5 PD 多副本版本一致性

- **版本协议**：PD 副本标当前 $\theta$ 版本；
- **同步升级**：blue-green 全副本切，或 canary 渐进；
- **不一致检测**：监控各副本版本，不一致告警；
- **避免部分旧**：负载均衡按版本路由（避免用户跨版本体验不一）。

### 8.6 sync 与训练稳定性的关系

weight sync 减 stale → ratio 准 → PPO 更新稳 → 避 [[KL explosion]]/[[policy collapse]]。故 sync 频率是 RLHF 稳定性工程的调参点（[[训练稳定性工程]]），与 target_kl/clip 协同。

---
相关: [[训推分离]]、[[stale policy problem]]、[[asynchronous training]]、[[policy training (PT)]]、[[policy deployment (PD)]]、[[learner worker split]]、[[rollout worker]]、[[inference worker]]、[[Data Parallel]]、[[Distributed Data Parallel]]、[[NCCL通信拓扑]]、[[AllGather]]、[[KL explosion]]、[[policy collapse]]、[[训练稳定性工程]]
