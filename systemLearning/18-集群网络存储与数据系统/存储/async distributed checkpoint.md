# async distributed checkpoint

> **所属章节**: [[存储]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: async ckpt / 异步 checkpoint / background save / 后台保存 / 训练不阻塞 ckpt / async save
> **难度**: 中（需懂 [[distributed checkpoint]] + [[state_dict分片]] + [[训练显存估计]] + [[本地NVMe与对象存储]] + 显存/带宽模型）


## 1. 一句话定义

**async distributed checkpoint（异步 checkpoint）** 是把大模型 checkpoint 的**落盘（写 NVMe / 上传对象存储）放到后台线程或独立进程**、**训练不等它就继续跑下一 step** 的策略——因为大模型 ckpt 极大（70B bf16 权重 140 GB、含 fp32 master + Adam(m,v) 全状态约 700–980 GB，见 [[训练显存估计]]），**同步保存**会把训练卡住几十秒到分钟，GPU 空转烧钱；async ckpt 的机制是**先把当前参数/优化器状态做一个快照副本（snapshot copy，detach+clone 到独立显存/内存）**，保证落盘期间原始参数继续被优化器更新也不会污染 ckpt 的**一致性**，然后**后台线程把快照异步写盘 / 传对象存储**，落盘完成后再通知（回调 / 标志位）训练侧。**关键认知**：不能简单异步——因为参数每 step 都在更新，若直接后台引用原 state_dict 会读到"半新半旧"的不一致权重，必须先 **snapshot 保证一致性**；与 [[distributed checkpoint]] 的关系是分工而非同义——后者是 **PyTorch DTensor 元数据跨拓扑 resharding 的保存格式**（解决"换 GPU 数/并行度时 ckpt 怎么存/载"），本条讲的是**落盘的异步调度策略**（解决"存盘不卡训练"），两者正交可叠加。

> [!note] 三句话定位
> - **是什么**：ckpt 落盘放后台，训练不等就跑下一 step；落盘前先 snapshot 保证一致性。
> - **为什么需要**：70B 全状态 ~700 GB，同步写盘几十秒~分钟，GPU 空转烧钱；异步让保存几乎不占 GPU 时间。
> - **与 [[distributed checkpoint]] 关系**：后者管"存什么格式/跨拓扑 reshard"，本条管"何时落盘/不阻塞训练"。正交可叠加。verl/Megatron 都支持。


## 2. 为什么需要它（动机与背景）

### 2.1 大模型 ckpt 极大

70B bf16 权重 = 140 GB。训练全状态（bf16 param + fp32 master + Adam m,v）≈ 70B×(2+4+4)B = **700 GB**（[[训练显存估计]]）。100B/175B 更大。这是 ckpt 落盘慢的根因。

### 2.2 同步保存卡 GPU

$T_{\text{save}}=D/BW_{\text{w}}$。700 GB 写单盘 NVMe 3 GB/s → ~233 s；写对象存储并发 1.6 GB/s → ~7 min。这段时间**全集群 GPU 空转**。按每 1000 step 存一次、训练 10 万 step，累计 GPU 空闲可达数小时，烧钱可观。

### 2.3 snapshot 保证一致性

若直接起线程引用 `model.state_dict()` 后台写，写盘期间优化器还在更新这些 tensor，后台读到的可能是"step 1000 半 + step 1001 半"的混杂权重，恢复训练会出错甚至发散。**必须先 snapshot**：把当前 step 的参数/状态 detach+clone 到独立显存/内存（或用 copy-on-write / 借助 framework 的 `_snapshot` 接口），后台写的是这个不可变快照，原始参数继续更新无影响。

### 2.4 多 rank 并发写的协调

分布式训练每 rank 只存自己的 state 分片（[[state_dict分片]] / [[distributed checkpoint]]），多 rank 并发写各自文件。async ckpt 还需协调：所有 rank 的快照在**同一 step** 取（barrier 保证 step 一致），各自后台写，全部写完后才算这次 ckpt 完成（rank 0 汇总标志）。

> [!warning] 误区：async ckpt 就是 `torch.save` 放线程里
> 不够。直接放线程会读到不一致的更新中权重。必须先 snapshot（detach+clone 快照），后台写的是快照。否则 ckpt 损坏。


## 3. 核心概念详解

### 3.1 同步 vs 异步的时间线

```
同步:
  step 1000 ─ train ─ [save: GPU 冻结 30s] ─ step 1001 train ...
                ^^^^^^^^^^^^^^^^^^^^^^^^ GPU 空转

异步:
  step 1000 ─ snapshot(1s) ─ step 1001 train ... step 1004 train ...
                └─── 后台落盘 30s (与训练 overlap) ──┘
  GPU 只等 snapshot(~1s), 不等落盘
```

### 3.2 snapshot 的几种实现

1. **clone 到 CPU/独立显存**：`{k: v.detach().clone().cpu() for k,v in sd.items()}`。简单通用，但 700 GB clone 占内存且 copy 也要几秒（带宽限）。verl/Megatron 多用此或变体。
2. **framework snapshot 接口**：PyTorch 的 `torch.distributed.checkpoint` / FSDP 提供 `_snapshot` / state_dict copy-on-write 语义，框架层高效快照。见 [[distributed checkpoint]]。
3. **双 buffer**：维护两份 state，训练用 A 时保存 B 的上一版，轮换。省 clone 但翻倍显存。

### 3.3 一致性：snapshot 在同一 step

```
rank 0..N  step 1000:
  barrier()              # 保证所有 rank 到 step 1000
  snap = snapshot(state) # 各 rank 快照本 rank 分片
  入后台队列               # 训练继续 step 1001
  后台: 各 rank 写 snap 到 /local_nvme/step-1000/rank-XX.pt
  全 rank 写完 -> rank0 写 done 标志 / 通知
```

snapshot 取在同一 step（barrier 后），保证这次 ckpt 是 step 1000 的完整一致快照。

### 3.4 落盘目标：本地 NVMe 先，对象存储后

- **本地 NVMe**：同步快落（秒级，训练只等 snapshot+少量本地写？实际本地写也异步）。见 [[本地NVMe与对象存储]]。
- **对象存储**：后台异步传（分钟级，不阻塞训练），传完删本地。

两段式 async：snapshot → 后台写本地 NVMe → 后台上传对象存储 → 删本地。训练全程只等 snapshot。

### 3.5 与 distributed checkpoint 的分工

| 维度 | **本条 async ckpt** | [[distributed checkpoint]] |
|---|---|---|
| 解决什么 | 落盘不阻塞训练（**何时存**） | 跨拓扑 reshard 的存/载格式（**存什么格式**） |
| 机制 | snapshot + 后台线程 | DTensor 元数据 + sharded 保存格式 |
| 层次 | 调度策略 | 数据格式/布局 |
| 关系 | 正交，可叠加 | 正交，可叠加 |

Megatron/PyTorch 的 distributed checkpoint（DCP）本身是格式，可配合 async 调度：async 调 snapshot 后用 DCP 写分片文件，后台落盘。

### 3.6 verl / Megatron 的 async ckpt

- **verl**（RL 训推一体框架）：rollout/训练循环里 ckpt 大，支持 async save 让 actor 训练不阻塞。snapshot 后后台写，配合 [[checkpoint转换]] 导出推理格式。
- **Megatron-LM**：`save_checkpoint` 支持 async（`async_save=True`），用独立进程/thread 写，训练继续。复用 torch.distributed.checkpoint 的 async save API。
- **PyTorch**：`torch.distributed.checkpoint.async_save` 自 2.x 起 native 支持 async save，底层用多进程/线程池。


## 4. 数学原理 / 公式

### 4.1 同步保存的 GPU 空闲成本

ckpt 数据量 $D$，写带宽 $BW_{\text{w}}$，每 $N$ step 存一次，共 $S$ step。同步策略 GPU 空闲总时间：

$$T_{\text{idle}}^{\text{sync}} = \frac{S}{N}\cdot\frac{D}{BW_{\text{w}}}$$

例：$D=700\text{ GB}$、$BW_{\text{w}}=24\text{ GB/s}$（8 盘 NVMe stripe）、$N=1000$、$S=100000$ → $\frac{100000}{1000}\times\frac{700}{24}\approx 2917\text{ s}\approx 48.6\text{ min}$。

### 4.2 异步保存的 GPU 占用

异步策略 GPU 只等 snapshot（clone/复制时间）$T_{\text{snap}}\approx D_{\text{snap}}/BW_{\text{copy}}$（$BW_{\text{copy}}$ 是显存→CPU/独立显存带宽）。

$$T_{\text{idle}}^{\text{async}} = \frac{S}{N}\cdot T_{\text{snap}}$$

若 $T_{\text{snap}}=2\text{ s}$ → $\frac{100000}{1000}\times 2 = 200\text{ s}\approx 3.3\text{ min}$。省 $\approx 45\text{ min}$。

### 4.3 落盘必须跑赢下次保存

后台落盘时间 $T_{\text{save}}$ 必须 $< N\times T_{\text{step}}$（两次保存间隔），否则积压：

$$\frac{D}{BW_{\text{w}}} < N\cdot T_{\text{step}}$$

否则后台队列堆积，内存/磁盘被未写完的快照占满。设计 async 时要核这个不等式，带宽不够就加大保存间隔 $N$ 或多盘 stripe。

### 4.4 snapshot 内存代价

snapshot 至少占一份 state 副本（clone 到 CPU 则占 CPU 内存 $D$，clone 到显存则占 $D$ 显存——显存放不下就 offload CPU）。所以 async 用 CPU 内存/NVMe 做快照中转，与 [[offloading (CPU-NVMe)]] 的显存-内存-NVMe 层次一致。


## 5. 代码示例（可选）

### 5.1 最小 async checkpointer（snapshot + 后台线程）

```python
import threading, queue, torch

class AsyncCheckpointer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.q = queue.Queue()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def _loop(self):
        while True:
            snap, step = self.q.get()
            if snap is None:
                break
            # 后台落盘 (与训练 overlap, 不阻塞)
            torch.save(snap, f"{self.save_dir}/step-{step}.pt")
            self.q.task_done()

    def save(self, model, optimizer, step):
        # 关键: 先 snapshot (detach+clone), 不持有原始引用
        snap = {
            "model": {k: v.detach().clone().cpu()     # clone 到 CPU, 显存释放
                      for k, v in model.state_dict().items()},
            "optimizer": {k: v.detach().clone().cpu()
                           for k, v in optimizer.state_dict().items()},
            "step": step,
        }
        self.q.put((snap, step))      # 入队, 训练继续下一 step

ckpt = AsyncCheckpointer("/local_nvme/ckpt")

for step, batch in enumerate(loader):
    train_step(batch)
    if step % 1000 == 0:
        ckpt.save(model, optimizer, step)   # 几乎不阻塞 (只 snapshot+入队)
```

> [!warning] 上面 `.clone().cpu()` 把 700 GB 状态拷到 CPU 内存，单机内存可能放不下。实际用 sharded save：每 rank 只 clone 自己分片（[[state_dict分片]]），且可分块流式写盘避免整份占内存。Megatron/PyTorch DCP async save 即分片流式。

### 5.2 用 PyTorch native async save（分片 + 后台）

```python
import torch.distributed.checkpoint as dcp

# 分布式训练中, 每 rank 存自己的分片
state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
# async save: 框架内部 snapshot + 后台线程池落盘
dcp.async_save(state, checkpoint_id=f"ckpt/step-{step}")
# 立即返回, 训练继续; 落盘完成回调
```

### 5.3 两段式：本地 NVMe + 异步传对象存储

```python
def save_two_stage(state, step, local_dir, s3_bucket, s3_prefix):
    snap = {k: v.detach().clone().cpu() for k, v in state.items()}
    def _bg():
        path = f"{local_dir}/step-{step}.pt"
        torch.save(snap, path)                      # 本地 NVMe (秒级)
        s3.upload_file(path, s3_bucket, f"{s3_prefix}/step-{step}.pt")
        os.remove(path)                             # 删本地
    threading.Thread(target=_bg, daemon=True).start()
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[state_dict分片]]（每 rank 存自己分片）、[[distributed checkpoint]]（存什么格式）、[[训练显存估计]]（ckpt 数据量）、[[本地NVMe与对象存储]]（落盘目标）。
- **下游（应用）**: 大模型训练 ckpt 不阻塞、RL 框架（[[verl]] / Megatron）actor 训练连续、配合 [[checkpoint转换]] 导出推理格式。
- **对比 / 易混**:
  - **async ckpt vs [[distributed checkpoint]]**：前者是落盘调度（何时存不卡训练），后者是存/载格式（跨拓扑 reshard）。正交可叠加。
  - **async ckpt vs [[offloading (CPU-NVMe)]]**：offload 是 step 间把 state 卸 NVMe 换显存（高频 prefetch）；async ckpt 是周期性保存整份 ckpt（低频）。都涉及 NVMe，但频率/目的不同。
  - **async ckpt vs gradient checkpointing**：前者是保存训练状态到盘（持久化），后者是前向时丢弃激活反向重算省显存。完全不同，只是都叫 checkpointing。


## 7. 常见误区与易错点

> [!warning] 误区 1：async 就是 `torch.save` 放线程
> 不够。参数在更新，直接后台引用原 state_dict 会读到不一致权重。必须先 snapshot（detach+clone）保证一致性。

> [!warning] 误区 2：async ckpt 不占额外内存
> snapshot 至少多一份 state 副本（700 GB 级），CPU 内存/显存要预留。分片 + 流式写盘可降低峰值，但仍有中转代价。

> [!warning] 误区 3：后台落盘随便慢没事
> 必须跑赢下次保存间隔 $D/BW < N\cdot T_{\text{step}}$，否则快照积压撑爆内存/磁盘。带宽不够要加 $N$ 或多盘。

> [!warning] 误区 4：async ckpt 和 distributed checkpoint 是一回事
> 不是。distributed checkpoint 管格式/跨拓扑 reshard，async 管何时落盘。正交。Megatron 可同时用 DCP 格式 + async 调度。

> [!warning] 误区 5：snapshot 用 `v.clone()` 不够
> `clone()` 复制值但不 detach 仍可能共享计算图/梯度。应 `detach().clone()` 切断图，且最好 `.cpu()` 或移到独立 buffer，避免显存占住。

> [!tip] 实践：保存频率与带宽匹配
> 先测 $BW_{\text{w}}$ 与 $T_{\text{step}}$，按 $N > D/(BW_{\text{w}}\cdot T_{\text{step}})$ 选保存间隔，确保后台落盘不积压。频繁存（小 $N$）+ 慢盘会撑爆。

> [!note] 补充：故障恢复的"最后一份"
> async 保存完成（所有 rank 写完 + 标志位）才算这次 ckpt 可用。训练挂了恢复时只载"已确认完成"的最近 ckpt，未写完的要丢弃。恢复逻辑要检查 done 标志。


## 8. 延伸细节

### 8.1 多进程 vs 多线程

CPU 密集（如 CPU 上的 optimizer state 序列化）用多进程避 GIL；纯 IO（写盘/网络）用线程即可。Megatron/PyTorch async save 多用进程池 + 共享内存传快照。

### 8.2 与 RL 框架的结合

verl 等 RL 框架训练循环里 actor/critic 频繁更新，ckpt 大，async save 让训练不阻塞，rollout 可继续。权重同步（[[权重同步]]）与 ckpt save 配合：save 后可触发权重下发到 rollout 引擎。

### 8.3 内容来源

ckpt 数据量与落盘时间来自 [[训练显存估计]] 与 [[本地NVMe与对象存储]] 的带宽模型。PyTorch `torch.distributed.checkpoint.async_save`、Megatron `--async-save` 为真实 API。snapshot 一致性、两段式本地+对象存储策略参考 DeepSpeed/Megatron/verl 工程实践。具体带宽/延迟随硬件，关键数量级（秒级 vs 分钟级、GB/s）稳定。

---
相关: [[存储]] | [[distributed checkpoint]] | [[state_dict分片]] | [[checkpoint转换]] | [[训练显存估计]] | [[本地NVMe与对象存储]] | [[offloading (CPU-NVMe)]] | [[ZeRO (DeepSpeed)]] | [[FLOPs计算]] | [[batch与mini-batch]]
