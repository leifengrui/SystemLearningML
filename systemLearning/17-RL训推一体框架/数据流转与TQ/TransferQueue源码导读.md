# TransferQueue 源码导读

> **所属章节**: [[数据流转与TQ]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §81（catalog 归在 trainer+rollout 下）
> **别名**: TQ / TransferQueue / AsyncFlow 数据面 / verl 数据网关 / transfer_queue
> **难度**: 高（需懂 [[colocated与disaggregated部署]]、[[V1 async trainer]]、[[RL角色拓扑]]、[[synchronous与asynchronous rollout]]、[[训推分离]]）
> **源码**: `https://github.com/Ascend/TransferQueue`（本地克隆 `transfer_queue/`，Apache-2.0，华为 + TQ Team）。论文为 [[AsyncFlow论文全文中文翻译]] 对应的 *AsyncFlow*（arXiv:2507.01663）。
> **本文件定位**: 不重复 README 的功能罗列，而是按「研究路径」逐层拆源码——控制面 `controller.py`、数据面 `storage/`、元数据 `metadata.py`、接口 `interface.py`，并对照 verl `trainer_base.py` 的调用点，说清 TQ 到底把 PPO dataflow 的哪一步解耦了。

## 1. 一句话定义

**TransferQueue（TQ）** 是一个面向后训练（post-training）的**异步流式数据管理与传输模块**：它用一张「行=样本、列=字段」的 2D 元数据表做**全景可见**的控制面（`TransferQueueController`，追踪每个样本每个字段的产出/消费状态），用可插拔的分布式存储后端做数据面（`StorageManager` + `SimpleStorageUnit`/Mooncake/Yuanrong/RayStore），再通过 Redis 风格的 KV API / PyTorch 风格的 `StreamingDataLoader` / 低层 `TransferQueueClient` 三档接口暴露给上层——本质是把 verl `RayPPOTrainer` 单控制器里「数据搬运」的活儿从控制流剥离，控制器只移动元数据，worker 直接从分布式存储取张量，从而把单点带宽瓶颈换成可扩展的分布式数据网关。

一句话：**TQ = 一个能按「字段就绪」粒度调度样本消费的分布式二维表 + 可插拔存储后端 + 三档 API**。

## 2. 为什么需要它（动机）

### 2.1 单控制器的数据流瓶颈

旧 verl `RLTrainer` 是单进程同时扛控制流与数据流：一步 PPO 要把整 batch 在控制器里序列化/反序列化 5–6 次（`rollout→reward→old_log_prob→ref→adv→update`）。多模态海量张量下，控制器进程的 CPU + 网络成单点瓶颈，所有 worker 都得排队过这个"收费站"。TQ（verl PR #5401，2026-04 合入）把数据流剥离：**控制器只移动元数据（`BatchMeta`），张量走 worker→存储后端直连**。

### 2.2 显式数据依赖耦合了任务

PPO 各步之间是隐式的"等上一步全跑完才能开始下一步"。TQ 把这种隐式依赖改成**字段级就绪**：下游任务只要声明"我要 `input_ids` + `log_prob` + `advantage` 这几个字段"，TQ 就能在这些字段全部就绪的样本上立即调度消费，不必等整 batch——这正是流式 / partial rollout 加速的底层支撑（见 [[V1 async trainer]] §2.4 partial rollout）。

### 2.3 分发逻辑耦合在控制器里

单控制器范式要手动决定每个 rank 拿哪些数据，适配不同并行策略（PP/DP/TP）很麻烦。TQ 把取数逻辑下放到 `Sampler` + `StreamingDataLoader`，每个 rank 自己向 TQ 拉数据，控制器退化为「side-controller for data dispatching」，大幅简化 disaggregated 框架设计。

## 3. 整体架构：三层

```
┌─────────────────────────────────────────────────────────┐
│  用户接口层 (interface.py / dataloader/ / client.py)     │
│   KV API    |  StreamingDataLoader  |  TransferQueueClient│
│   Put/Get/List/Clear | PyTorch 风格    |  原子元数据操作   │
└──────────────┬──────────────────────────┬───────────────┘
               │ 元数据 (BatchMeta)         │ 张量
┌──────────────▼─────────────────┐ ┌────────▼──────────────┐
│ 控制面 controller.py            │ │ 数据面 storage/        │
│  TransferQueueController        │ │  StorageManager        │
│   - DataPartitionStatus (2D表)  │ │   - SimpleStorageUnit  │
│   - production/consumption 张量 │ │   - Mooncake/Yuanrong  │
│   - Sampler (取数策略)          │ │   - RayStore            │
│   - ZMQ ROUTER (握手+请求)      │ │   ZMQ DEALER / 原生 client│
└─────────────────────────────────┘ └───────────────────────┘
```

控制面与数据面**解耦**：控制器只维护元数据与就绪状态，张量从不经过控制器；两者通过 ZMQ（SimpleStorage）或各后端原生 client 通信。

## 4. 核心数据模型：2D 表与三张状态

这是理解整个 TQ 的钥匙。TQ 把一个 partition（逻辑数据容器，如 `train@global_batch_0`）建模成一张**动态可扩张的二维表**：

- **行（row）= 一个样本**，分配一个全局唯一 `global_index`（跨 partition 不重复，由 `PartitionIndexManager` 分配，可回收复用）。
- **列（column）= 一个计算任务的输出字段**，如 `input_ids` / `log_prob` / `advantage` / `values`。

围绕这张表有**三类状态**（`controller.py: DataPartitionStatus`）：

| 状态 | 数据结构 | 含义 |
|---|---|---|
| `production_status` | `torch.int8` 张量 `[N_samples, N_fields]` | 0=未产出, 1=可消费。某行所有需要列=1 → 该样本就绪 |
| `consumption_status` | `dict[task_name, int8[N]]` | 每个**消费任务**独立一行，记录哪些样本被它消费过 |
| `field_metadata` | `dict[field, FieldMeta]` | 列级 schema：dtype/shape/is_nested/is_non_tensor，O(F) 存储 |

关键设计：**消费按任务隔离**。`compute_log_prob` 和 `compute_ref_log_prob` 可能要同一个 `input_ids` 字段，但各自有独立的 `consumption_status`，互不干扰——同一条数据可被多个下游独立消费。这正对应 PPO 里 advantage 计算和 ref 计算都要旧策略 log_prob 的场景。

表可**动态扩张**：`ensure_samples_capacity` / `ensure_fields_capacity` 负责给状态张量加行加列（一次性扩到所需容量，amortized）。还支持**预分配**（`register_pre_allocated_indexes`）：consumer 在 producer 还没生成完所有样本时就能看到预期总样本数，这对 `StreamingDataLoader` 提前规划消费节奏很关键。

## 5. 控制面深入：`controller.py`

`TransferQueueController` 是一个 `@ray.remote(num_cpus=1)` actor，单实例全局命名（`name="TransferQueueController", namespace="transfer_queue"`），后启动的进程通过 `_init_from_existing` 复用它（`interface.py: init`）。它内部跑两个 daemon 线程：握手线程（接 storage unit 注册）+ 请求处理线程（ROUTER socket 收 ZMQ 请求）。

### 5.1 `get_metadata` 的三模式（核心入口）

`controller.get_metadata(...)` 是控制面的总开关，三种 `mode`：

| mode | 用途 | 行为 |
|---|---|---|
| `insert` | 写入侧调用 | 激活/分配 `global_indexes`，生成**未就绪**的 `BatchMeta`（`production_status=0`），待 `put_data` 完成后由 `update_production_status` 翻成 1 |
| `fetch` | 消费侧调用 | `scan_data_status` 找就绪且未消费的样本 → 交给 `Sampler.sample` 选 batch → `mark_consumed` 标记。不够则阻塞重试或 `polling_mode` 返回空 |
| `force_fetch` | 调试/快照 | 取 partition 内全部 index，不做就绪/消费过滤 |

`fetch` 模式的就绪判定在 `DataPartitionStatus.scan_data_status`（`controller.py:712`）：用 `row_mask`（未消费）& `col_mask`（所需字段）切片 `production_status`，`torch.all(dim=1)` 找出所有需要列都为 1 的行。这是纯张量运算，批量大时也比逐样本判断快。

### 5.2 `Sampler`：可定制的取数逻辑

取数逻辑从控制器里解耦成 `BaseSampler`（`sampler/base.py`），接口只有一个方法：

```python
def sample(self, ready_indexes, batch_size, *args, **kwargs) -> tuple[list[int], list[int]]:
    """返回 (要取的 indexes, 要标记为已消费的 indexes)"""
```

**两个返回值分离**是精髓：取数和消费解耦，能轻松实现"样本可替换"（同一条数据被多次采样训练）。内置实现：

- `SequentialSampler` — 顺序取，最简单。
- `GRPOGroupNSampler` — **GRPO 专用**：保证同一 prompt 的 n 条 rollout 要么整组取出要么不取。它扫描 `ready_indexes` 找**连续**的 n 个 index 为一组（要求用户按 `[p1_s1,p1_s2,...,p2_s1,...]` 顺序 put），组不全就跳过。还按 `(dp_rank, batch_index)` 缓存采样结果保证确定性（`grpo_group_n_sampler.py:141`）。
- `SeqlenBalancedSampler` — 按 seqlen 负载均衡采样，把 verl legacy 的 `_balance_batch` 下放到 control plane 做。
- `RankAwareSampler` — 给 `StreamingDataLoader` 用，按 rank 分发。

自定义 sampler 传类或实例给 `TransferQueueController.remote(sampler=...)` 即可。

### 5.3 KV 语义层：key ↔ global_index 双向映射

KV API 用用户可读的 `key`（字符串）寻址，但底层仍是 `global_index`。`DataPartitionStatus` 维护 `keys_mapping`（key→idx）和 `revert_keys_mapping`（idx→key），`kv_retrieve_meta` 负责翻译：遇到新 key 且 `create=True` 就分配 index 并登记。所以 KV 接口不是另一套存储，而是**同一张 2D 表的字符串寻址层**。

### 5.4 ZMQ 通信骨架

控制器开两个 ZMQ ROUTER socket：`handshake_socket`（storage unit 注册握手，ACK 即可）+ `request_handle_socket`（处理业务请求）。请求是 `ZMQMessage` 帧，`_handle_request` 用一张 `request_type → handler` 字典分发（`controller.py:1936`），覆盖 `GET_META / NOTIFY_DATA_UPDATE / KV_* / SAVE_CHECKPOINT` 等十几种。请求循环**任何异常都不杀线程**，转成 `REQUEST_ERROR` 回复（避免 ROUTER 静默丢消息让调用方挂死在 `recv`）。

## 6. 数据面深入：`storage/`

数据面是**可插拔**的，统一抽象在 `StorageManager`：

```python
async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None
async def get_data(self, metadata: BatchMeta) -> TensorDict
async def clear_data(self, metadata: BatchMeta) -> None
```

### 6.1 SimpleStorage（默认，最该先读）

`AsyncSimpleStorageManager` + `SimpleStorageUnit`。每个 unit 是一个独立 Ray actor，部署在单独节点，**round-robin 跨 Ray 节点调度**（`num_data_storage_units` 建议 ≥ 2×节点数，让每节点扛多个 unit 分摊内存/带宽）。

unit 内部就是那张 2D 表的物化：行=样本、列=字段，按 `global_index` 精确寻址，支持细粒度并发读写。通信走 ZMQ DEALER↔ROUTER，`ZMQSocketPool` 复用连接。请求有重试（`TQ_SIMPLE_STORAGE_MAX_ATTEMPTS=3`，每次新建连接——因为 DEALER 给不可达对端会静默排队，复用连接抓不到超时）、超时探测（`accept_probe.py`）。

### 6.2 KV 后端

- **MooncakeStore**：KV-based 分层存储，支持 **RDMA**（GPU↔DRAM 直连，可开 GPUDirect RDMA `use_gdr`）+ **SSD offload**（内存超 high watermark 时 evict 到 NVMe，单节点集中式 offload pool，未来改多节点）。配置见 `config.yaml: MooncakeStore`。
- **Yuanrong**（openEuler）：昇腾原生数据系统，分层 HBM/DRAM/SSD，可开 `remote_h2d_device_ids`（RH2D 跨节点直传）+ huge_tlb。配置 `config.yaml: Yuanrong`。
- **RayStore**：Ray 新特性 RayRDT，让 Ray actor 间直接存传对象（alpha）。

KV 后端可直接继承 `KVStorageManager`（KV 通用 manager）+ 写一个 `storage/clients/` 下的低层 adapter，工厂类 `StorageManagerFactory` / `StorageClientFactory` 负责注册。

### 6.3 Checkpoint

`interface.py: save_checkpoint / load_checkpoint`。流程：controller 状态（partitions 快照 + index_manager + sampler 状态）pickle 到 `controller_state.pkl`，storage 状态由后端自决（SimpleStorage 内存态**强制保存**，否则重启丢数据；Mooncake 等 KV 后端数据外部持久化时可 `include_storage=False` 只存控制器元数据）。写入用 `tmp→rename` 原子替换，崩溃能从 `.old` 回滚。多节点要求 checkpoint_dir 在共享网络文件系统（NFS/GPFS/Lustre）上。

## 7. 用户接口：三档 API

| 档位 | 风格 | 细粒度 | 流式 | Sampler | 多后端 | 文件 |
|---|---|---|---|---|---|---|
| **KV Interface** | Redis Put/Get/List/Clear | ✓（按字段列） | ○ | ✗ | ✓ | `interface.py` |
| **StreamingDataLoader** | PyTorch DataLoader drop-in | ✓ | ✓ | ✓ | ✓ | `dataloader/streaming_dataloader.py` |
| **TransferQueueClient** | 原子元数据操作 | ✓ | ✓ | ✓ | ✓ | `client.py` |

### 7.1 KV Interface（verl 用的就是这个）

`interface.py` 暴露 `init/close/kv_put/kv_batch_put/kv_batch_get/kv_list/kv_clear` + `async_*` 版本 + `save/load_checkpoint`。典型流：

```python
import transfer_queue as tq
from tensordict import TensorDict
import torch

tq.init()  # 首进程起 controller+storage；其它进程 init() 只建 client 复用现有 controller

keys = ["s1","s2","s3"]
fields = TensorDict({"input_ids": torch.randn(3,10),
                     "attn_mask": torch.ones(3,10)}, batch_size=3)
meta = tq.kv_batch_put(keys=keys, partition_id="train", fields=fields,
                      tags=[{"score":0.9},{"score":0.85},{"score":0.95}])
# meta.fields == ['input_ids','attn_mask']

data = tq.kv_batch_get(keys=["s1","s2"], partition_id="train",
                       select_fields="input_ids")  # 只取一列
```

要点：
- `kv_batch_put` 内部三步：`async_kv_retrieve_meta(create=True)` 拿/建 index → `update_custom_meta` 挂 tag → `async_put` 写张量（见 `interface.py:686`）。
- `kv_batch_get` 取前检查 `batch_meta.is_ready`（所有请求字段全就绪），否则抛 `ValueError`——**消费侧就绪门控**就在这里。
- `select_fields` 允许只取一行的某些列，不必整行操作——细粒度访问。
- `data_parser` 钩子可在 put 前把引用（如 URL）解析成真实内容（仅 SimpleStorage 支持）。

### 7.2 StreamingDataLoader

PyTorch `DataLoader` 风格 drop-in，每个 rank 自动消费、无需单控制器干预。`controller` 退化为 side-controller，用户定义 `Sampler` 组织 dataflow。封装了各种并行策略下的调度与数据搬运逻辑，是 disaggregated 框架的关键抽象。

### 7.3 低层 TransferQueueClient

`client.py` 提供原子操作 `async_get_meta / async_put / async_get_data / async_clear_samples / async_kv_retrieve_meta / async_check_consumption_status` 等，最大灵活性，用于写需要细粒度控制的全流式调度。它内部维护 ZMQ async 上下文 + socket pool，`AsyncTransferQueueClient` 是异步版，`TransferQueueClient` 是其同步包装（`_run_coroutine` 跑事件循环）。

## 8. 端到端：verl PPO dataflow 怎么被 TQ 解耦

把 TQ 的 `kv_batch_put/get` 对照 verl `trainer/ppo/v1/trainer_base.py` 的调用点，就能看清每一步。一步 PPO 的字段流水线：

```
generate_sequences  →写字段: input_ids, responses, attention_mask
compute_log_prob    →读 input_ids/responses,  写 old_log_prob
compute_ref_log_prob→读 input_ids/responses,  写 ref_log_prob      (独立消费,不干扰上一步)
compute_values      →读 ..., 写 values
compute_advantage   →读 old_log_prob/reward, 写 advantage
update_policy       →读 advantage+old_log_prob+values, 反传更新
```

在 TQ 里每一步都是「下游声明 `data_fields` → `get_metadata(mode="fetch")` → TQ 等这些字段就绪的样本出现 → `Sampler` 选 batch → `get_data` 拉张量 → 算完 `kv_batch_put` 写新字段 → TQ 通知 controller 更新 `production_status`」。

- **generate 与 compute_log_prob 可重叠**：只要前几条样本的 `responses` 写完，log_prob worker 就能取这几条开算，不必等整 rollout batch。
- **同字段多消费者独立**：`ref_log_prob` 和 `log_prob` 都读 `input_ids`，但 `consumption_status` 按 task 隔离，谁也不挡谁。
- **控制器只动元数据**：张量从 storage unit 直接到计算 worker，不再过控制器进程——这就是 PR #5401 那个 49.1% 端到端加速（128×H100 多模态）的来源。

## 9. 关键工程细节（源码级）

- **全局 index 复用**：`PartitionIndexManager` 维护 `reusable_indexes` 池，partition 释放后 index 回收再分配，避免长期跑 index 无限增长。
- **预分配样本**：`TQ_PRE_ALLOC_SAMPLE_NUM` 环境变量，consumer 在 producer 写入前就能看到预期总样本数，`StreamingDataLoader` 据此规划。
- **polling_mode**：controller 数据不足时不阻塞而返回空 `BatchMeta`，调用方自行重试——异步流水线避免在控制器侧卡线程。
- **ZMQ 大 socket 预算**：`TQ_CONTROLLER_ZMQ_MAX_SOCKETS=4096`（默认 1023），因为 metrics exporter 共用 controller 的 ZMQ context 去查询 storage unit，预算要覆盖整个集群。
- **请求永不静默丢弃**：`_process_request_loop` 任何 handler 异常都转 `REQUEST_ERROR` 回复，调用方不会挂在 `recv`。
- **Ray Arrow 只读数组**：`BatchMeta.__setstate__` 检测 `production_status` 不可写则 `.copy()`，绕过 Ray 零拷贝反序列化的只读 ndarray 问题。
- **metrics**：`TQMetricsExporter` 暴露 Prometheus `/metrics`，controller 周期推 partition 产出/消费进度快照，storage unit 也注册端点被采集。
- **Mooncake offload**：单节点集中式 SSD pool，master 触发 eviction 时从任意节点内存经 TCP/RDMA 搬到 offload client；适合 CPU DRAM < GPU HBM 的环境。

## 10. 推荐研究顺序（按用户给的地图）

1. **tutorial 跑通**（`tutorial/`，循序渐进）：
   - `01_core_components.py` — Controller / StorageManager / Sampler 三大件
   - `02_kv_interface.py` — **最该先读**，verl 用的就是这个 API
   - `03_metadata_concepts.py` — BatchMeta / KVBatchMeta，理解 2D 行列模型
   - `04_understanding_controller.py` — control plane 如何追踪字段就绪
   - `05_custom_sampler.py` — 自定义取数逻辑
   - `06_streaming_dataloader.py` — 流式消费端
2. **读 `interface.py` + `controller.py`**：KV API 实现 + 核心数据结构 `DataPartitionStatus`。
3. **看存储后端（按需）**：先 `simple_storage_manager.py`（CPU 内存 2D 表，最易懂），生产再 Mooncake / Yuanrong。
4. **回 verl 对照**：TQ `kv_batch_put/get` ↔ verl `trainer/ppo/v1/trainer_base.py` 调用点，看清 PPO 每步如何解耦。

## 11. 与 verl 直接相关的关键点（速查）

- `sampler/grpo_group_n_sampler.py`：GRPO 同 prompt n 条 rollout 保持一组取出，对应 verl GRPO advantage 计算。
- `sampler/seqlen_balanced_sampler.py`：对应 verl legacy `_balance_batch`，下放到 control plane 做负载均衡——即文档说的"offload single-controller 的数据管理能力"。
- `save_checkpoint / load_checkpoint`（`interface.py` 导出）：verl 在 `trainer_base.py:845-853` 用 `_tq_supports_checkpoint()` 判断是否支持。
- verl PR #5401（2026-04）正式集成，DAPO 64 节点 1024 卡验证过，显著优化 host 内存利用与数据传输加速。
- 已被采纳：Meshy（角色即服务，TQ 连接各角色）、Dressage（32 节点 GLM-5.2，master 数据面峰值内存降 91%）、ROLL（`RemoteBatch` 兼容 `DataProto`）、UniRL（腾讯混元多模态）、Relax（micro-batch 级跨集群调度）。

## 12. 一句话总结

TQ 的全部精妙就在于那张「行=样本、列=字段」的二维状态表 + 字段级就绪门控：它把 PPO 流水线里"等上一步全完才能开始下一步"的隐式同步，换成"字段就绪即可消费"的细粒度流式调度，控制器只动元数据、张量走分布式存储直连，从而把单控制器带宽瓶颈变成可水平扩展的数据网关——这是 verl v0.7 异步管线（[[V1 async trainer]]）能做 partial rollout + 跨模型版本恢复的数据面根基。

---

**相关阅读**: [[V1 async trainer]]（verl 怎么用 TQ）｜[[AsyncFlow论文全文中文翻译]]（论文层论证）｜[[colocated与disaggregated部署]]（部署形态）｜[[RL角色拓扑]]｜[[synchronous与asynchronous rollout]]｜[[DAPO动态采样与AsyncFlow实现对照]]
