# Ray Data

> **所属章节**: [[Ray Data]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: Ray Data / Ray 流式数据管道 / streaming execution / dataset pipeline / backpressure
> **难度**: 中（需懂 [[Spark执行模型]] 的 stage/shuffle 对比 + [[Ray与分布式调度]] 的 actor/task 抽象 + [[WebDataset]] 的 shard 格式）


## 1. 一句话定义

**Ray Data** 是 Ray 生态的**流式数据处理库**——基于 [[Ray与分布式调度]] 的 actor/task 模型，把数据集当**逐批流式处理**（**streaming execution**：不一次性载入内存、按需读一批处理一批，适合超大数据集）、内置 **backpressure**（下游慢则上游自动降速反压、防 OOM）、提供 **pipeline 算子链**（`map` / `batch` / `map_batches` / `filter` / `random_shuffle` / `split` 等）。它是 AI 训练/推理数据管道的核心：**读 shard（[[WebDataset]]/[[Parquet]]）→ 预处理（解码、增广、tokenize）→ batch → 喂 trainer**，全流式不停训练、内存可控、可分布式扩展。与 [[Spark执行模型]] 比，**Spark 是 stage 落盘 batch 模型**（shuffle 落盘、stage 边界有显式中断）、**Ray Data 是流式 + actor 进程模型**（actor 间直接传数据、无 stage 落盘、低延迟），**Ray Data 更适合喂 GPU 训练的低延迟流式**（训练 step 边产数据边消费、不停顿）、**Spark 更适合离线 ETL**（大 join、批统计、SQL 生态成熟）。是 [[Ray与分布式调度]] 在数据并行场景的专门库——把 actor 模型用于数据 pipeline，**消除 Spark 的 stage 边界**，让数据像流水线一样在 actor 间流动。

> [!note] 三句话定位
> - **是什么**：Ray Data = 流式数据处理库，逐批流式执行（不载全量）、背压防 OOM、actor 间直接传数据。
> - **为什么用**：AI 训练数据管道（读 shard→预处理→batch→喂 trainer），流式不停训练、内存可控、分布式可扩展。
> - **与 [[Spark执行模型]] 区别**：Spark 是 stage 落盘 batch 模型（shuffle 落盘、有 stage 边界）；Ray Data 是流式 actor 进程模型（无 stage、低延迟喂 GPU）。

## 2. 为什么需要它（动机与背景）

### 2.1 AI 训练数据管道的痛点

AI 训练（尤其 LLM 预训练、多模态训练）数据量动辄 TB-PB：

- 全量载内存：**OOM**（一台机器装不下）。
- 全量载磁盘后按需读：**I/O 阻塞**（trainer 等 I/O，GPU 闲）。
- 用 Spark 预处理再喂 trainer：**两阶段、stage 边界落盘**，trainer 与 preprocessing 解耦但中间需落盘，慢且占存储。

理想：**读 shard → 预处理 → batch → 喂 trainer，全流式不停训练、内存可控、I/O 与算力重叠（trainer 算时下一批预处理）**。

### 2.2 Spark 不适合喂 GPU

Spark 的执行模型（[[Spark执行模型]]）是**stage 落盘 batch**——每个 stage 把所有 partition 跑完、shuffle 落盘后才进下一 stage。问题：

- **延迟高**：trainer 要等整个 stage 跑完才能拿到数据，秒-分钟级延迟，trainer 闲。
- **stage 边界硬**：actor 间不直接传数据、必落盘，不能流式喂。
- **批处理优化目标不同**：Spark 优化吞吐（大 join/SQL），不优化"逐批喂 GPU 低延迟"。

### 2.3 Ray Data 的方案：流式 + actor

Ray Data 基于 [[Ray与分布式调度]] 的 **actor 模型**（[[actor model]]）：

- 每个 actor 是常驻进程、有状态、处理一批数据。
- actor 间直接传数据（通过 object store 内零拷贝、跨节点序列化）。
- **无 stage 边界**——actor 1 处理完一批直接传 actor 2，actor 2 处理完传 actor 3。
- **背压**：actor 2 慢 → actor 1 发不出去 → actor 1 等待（block）→ actor 0 等待 → 上游降速。**自动反压、防 OOM**。

这正好喂 GPU：**prefetch actor → decode actor → batch actor → trainer**，trainer 算完一批直接拿下一批，GPU 不闲。

## 3. 核心概念详解

### 3.1 streaming execution（流式执行）

Ray Data 不一次载全量数据集。它**按需读一批（如 64 个样本）**、流过 pipeline、产出后读下一批。内存占用 = **几个 batch × pipeline 长度**（常数级），与数据集总大小无关。

```
源 shard 0 (256 samples)
  |  读 64
  v
[prefetch actor] (buffer 64)
  | 流式传
  v
[decode actor] (batch 64)
  | 流式传
  v
[batch actor] (batch 64)
  | 流式传
  v
trainer (consume 64)
  ↑ 反压: trainer 慢 -> batch actor buffer 满 -> decode 等 -> prefetch 等 -> 源降速
```

> [!note] 补充：与 Spark stage 的对照
> Spark 把 pipeline 切成 stage，**所有 partition 跑完一 stage 才进下 stage**。Ray Data 不切 stage——actor 间是流式管道，**一批数据可在 actor 1 还没处理完第二批时已到 actor 2**。这是 stage 边界消除的关键。

### 3.2 backpressure（背压）

**背压**：下游慢 → 上游自动降速。Ray Data 通过 **bounded buffer** 实现：

- 每个 actor 有固定大小 buffer（如 2 batch）。
- 上游发数据给下游，下游 buffer 满 → 上游阻塞等待（block on send）→ 上游降速。
- 链式反压：trainer 慢 → batch buffer 满 → decode 阻塞 → prefetch 阻塞 → 源降速。

**结果**：数据流速率 = **最慢 actor 的速率**（瓶颈），但**无 OOM**（buffer 有界）、**无丢数据**（背压而非丢弃）。

```
若 trainer 10 step/s, decode 100 step/s:
  - 无背压: decode 疯狂产, buffer 爆, OOM
  - 有背压: decode 产 2 batch 后阻塞, 等 trainer 消费再产, 速率 = 10 step/s (trainer 速率)
```

### 3.3 pipeline 算子链

Ray Data 提供 dataset pipeline 算子：

| 算子 | 语义 | 对应 Spark |
|---|---|---|
| `map(fn)` | 逐元素处理（如解码、tokenize） | `rdd.map` |
| `batch(n)` | 元素聚成 batch（n 个一组） | 无直接对应 |
| `map_batches(fn)` | 逐 batch 处理（如 GPU 上的 tokenize、collate） | `rdd.mapPartitions` |
| `filter(pred)` | 过滤 | `rdd.filter` |
| `random_shuffle()` | 全量 shuffle（落盘） | `rdd.repartition` |
| `random_sample(p)` | 抽样 | 无 |
| `split(n)` | 切 n 份（多 trainer 并行） | `rdd.randomSplit` |
| `window(n)` | 滑窗 | 无 |
| `flat_map(fn)` | 一对多展开 | `rdd.flatMap` |

```python
import ray

ds = ray.data.read_parquet("/path/to/data")   # 流式读

pipeline = ds.map(decode_image) \               # 逐样本解码
              .map(tokenize) \                  # 逐样本 tokenize
              .batch(32) \                      # 聚 32 一批
              .map_batches(collate_for_gpu) \   # batch 级 collate
              .split(4)                          # 切 4 份喂 4 个 trainer worker

# trainer worker 0
for batch in pipeline[0].iter_torch_batches():
    train_step(batch)
```

### 3.4 map vs map_batches

| 算子 | 粒度 | 适合 |
|---|---|---|
| `map(fn)` | 逐元素 | 简单逐样本操作（CPU、低开销） |
| `map_batches(fn)` | 逐 batch | batch 级处理（GPU 加速、collate、向量化） |

**经验**：CPU 密集逐样本用 `map`；GPU/向量化操作（如 tokenizer batch tokenize、CLIP score）用 `map_batches`（batch 利用 GPU、降低 per-sample overhead）。

### 3.5 split：多 trainer worker 并行

分布式训练（DDP/FSDP）多 worker 各取数据。Ray Data `split(n)` 把 pipeline 切 n 份、每 worker 一份独立流式、**互不干扰**。无需手动 partition/split。

```python
pipeline = ds.map(preprocess).batch(64).split(world_size)   # world_size=8

for epoch in range(num_epochs):
    for batch in pipeline[rank].iter_torch_batches():
        train_step(batch)
    pipeline[rank] = ...   # 下 epoch
```

### 3.6 prefetch（预取）

为消除 I/O 阻塞 trainer，Ray Data pipeline 默认 **prefetch**——上游 actor 提前生产 batch 进下游 buffer，trainer 算当前 batch 时下一批已在 buffer。pipeline 深度 = actor 数、buffer 大 = `pipeline_budget`，**trainer 几乎不等 I/O**。

### 3.7 与 Spark 的对比

| 维度 | Spark | Ray Data |
|---|---|---|
| 数据流模型 | stage 落盘 batch | 流式 actor 间传 |
| 中间介质 | 磁盘（shuffle 文件） | 内存（object store / actor） |
| 延迟 | 秒-分钟（stage 边界） | 毫秒-秒（流式） |
| 背压 | 隐式（shuffle 文件堆积） | 显式 bounded buffer |
| 计算单元 | task（一次性） | actor（常驻进程） |
| 状态管理 | 无状态 task | actor 有状态、可变异 |
| 适合场景 | 离线 ETL、大 join、SQL | 喂 GPU 训练、RL rollout、推理数据 |
| shuffle | shuffle 落盘（[[Spark shuffle与partition]]） | 无 shuffle（流式无 stage） |
| 数据格式 | [[Parquet]] 为主 | [[WebDataset]] / [[Parquet]] / [[Arrow]] |

> [!tip] 实践：何时用 Ray Data 何时用 Spark
> - **训练/推理数据 pipeline、流式喂 GPU、RL rollout** → Ray Data（低延迟、流式、背压）。
> - **离线 ETL、大 join、批 SQL、特征工程** → Spark（shuffle 优化深、SQL 生态、批吞吐高）。
> - 二者常配合：Spark 离线预处理生成 Parquet，Ray Data 流式读 Parquet 喂 trainer。

### 3.8 与 WebDataset / Parquet 的结合

- **[[WebDataset]]**：tar shard 格式，**流式读**（不解 tar、按 sample 流出）。Ray Data `read_webdataset` 流式读 tar shard、喂 pipeline。适合多模态大训练（图片+文本 shard）。
- **[[Parquet]]**：列存格式，**按 row group 读**（不载全文件）。Ray Data `read_parquet` 流式按 row group 读、批处理。
- **[[Arrow]]**：Ray Data 底层用 Arrow 内部表示（零拷贝跨进程传、列存高效）。

```python
# 流式读 WebDataset (tar shard)
ds = ray.data.read_webdataset("s3://bucket/shard-{000000..000999}.tar")
pipeline = ds.map(decode).batch(64).split(8)

# 流式读 Parquet (按 row group)
ds = ray.data.read_parquet("s3://bucket/data/*.parquet")
pipeline = ds.map(preprocess).batch(128).split(8)
```

## 4. 数学原理 / 公式

### 4.1 流式内存占用

设 pipeline 有 $K$ 个 stage（actor）、每 stage buffer $B$ batch、每 batch $n$ 样本、每样本 $s$ 字节。则：

$$\text{memory} \approx K \cdot B \cdot n \cdot s$$

与数据集总大小 $D$ 无关——**常数级内存**、可处理 TB-PB 数据集。Spark 需把 partition 全载内存（cache）或落盘，无法做到 pipeline 全流式常驻内存。

### 4.2 背压吞吐

设各 actor 处理速率 $\rho_i$（batch/s）。理想无背压下 pipeline 吞吐 $\rho^* = \min_i \rho_i$（瓶颈 actor）。

- **无背压**：$\rho_i$ 高的 actor 不断产、buffer 爆 OOM。
- **有背压**：上游 actor 发下游 buffer 满时阻塞，**实际吞吐** $\rho_{\text{actual}} = \rho^*$、无 OOM。

### 4.3 prefetch 消除空闲

trainer 每步耗时 $T_t$、I/O + 预处理耗时 $T_p$。

- **无 prefetch**：trainer 算 $T_t$ → 等预处理 $T_p$ → 算 $T_t$ → 等 $T_p$... 周期 $T_t + T_p$。
- **有 prefetch**：预处理在 trainer 算当前 batch 时并行产下一 batch，只要 $T_p \le T_t$，**trainer 周期 = $T_t$**（不空闲）。

条件 $T_p \le T_t$：预处理慢于 trainer 才瓶颈、否则 prefetch 把 I/O 隐藏。

### 4.4 split 多 worker 并行

$W$ 个 worker 各取一 split pipeline、总吞吐 $\approx W \cdot \rho^*$（每 worker 独立流式、背压独立）。

$$\text{total\_throughput} = \sum_{w=0}^{W-1} \rho_w^* \approx W \cdot \rho^*$$

无 Spark 的 shuffle（split 是数据源级切分、不重分布）。

### 4.5 actor 间通信代价

actor 跨节点传 batch（$n \cdot s$ 字节）代价 $\approx \frac{n \cdot s}{\text{bandwidth}} + \text{serialize}$。Ray 用 [[Arrow]] 零拷贝（同节点共享内存、跨节点序列化 + 网络传输）。设计 pipeline 时**减少跨节点**（actor 调度到同节点）省通信。

## 5. 代码示例（可选）

### 5.1 简单 pipeline

```python
import ray

ray.init()

# 读 parquet, 流式
ds = ray.data.read_parquet("/path/to/data")

# pipeline 算子链
pipeline = ds.map(lambda x: {"text": x["raw"].lower()}) \
              .filter(lambda x: len(x["text"]) > 0) \
              .batch(32) \
              .map_batches(lambda b: tokenize_batch(b))   # batch 级 tokenize

for batch in pipeline.iter_torch_batches():
    train_step(batch)
```

### 5.2 多 worker（DDP）split

```python
import ray
import torch.distributed as dist

ray.init()
dist.init_process_group("nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()

ds = ray.data.read_webdataset("s3://bucket/shard-*.tar")
pipeline = ds.map(decode_image).map(augment).batch(64).split(world_size)

for epoch in range(num_epochs):
    for batch in pipeline[rank].iter_torch_batches():
        train_step(batch)
```

### 5.3 自定义 actor（GPU 预处理）

```python
from ray.data import ActorPoolStrategy

class GPUPreprocessor:
    def __init__(self):
        self.model = load_clip().to("cuda:0")   # actor 常驻 GPU 模型
    def __call__(self, batch):
        return self.model(batch)   # GPU 批处理

# 用 actor 池跑 map_batches (复用 GPU 模型, 免每 task 重载)
pipeline = ds.map_batches(GPUPreprocessor, strategy=ActorPoolStrategy(8), batch_size=64)
```

> [!note] 补充：actor 池 vs task
> `map_batches` 默认用 **task 模型**（每 batch 起 task、无状态）。指定 `strategy=ActorPoolStrategy(N)` 改用 **actor 模型**（N 个常驻 actor、有状态、复用模型）。GPU 加载重的（如 CLIP、tokenizer）**必用 actor 池**，免每 batch 重载模型。是 Ray Data 喂 GPU 的关键。

### 5.4 backpressure 实验感

```python
# trainer 故意慢 (sleep), 看上游自动降速
import time
def slow_trainer(batch):
    time.sleep(1)
    return batch

ds = ray.data.range(10000)
pipeline = ds.map(lambda x: x).batch(10).map_batches(slow_trainer)

for batch in pipeline.iter_batches():
    print(time.time(), "got batch")   # 每 1s 一个, 而非每 0.001s (上游降速)
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[Ray与分布式调度]]（actor/task 调度基础）、[[actor model]]（actor 是常驻有状态进程）、[[WebDataset]]（流式 shard 格式）、[[Parquet]]（列存流式读）、[[Arrow]]（零拷贝跨进程传）、[[task model]]（map 默认 task 模型）。
- **下游（应用）**: AI 训练数据管道（喂 LLM/多模态 trainer）、RL rollout 数据收集、推理服务数据预处理、[[trajectory generation]]（RL 经验流式产出喂 trainer）。
- **对比 / 易混**:
  - **Ray Data vs [[Spark执行模型]]**：前者流式 actor 模型、无 stage 边界、低延迟喂 GPU；后者 stage 落盘 batch、shuffle 优化深、适合离线 ETL。
  - **Ray Data vs [[Spark shuffle与partition]]**：前者无 shuffle（流式无 stage）；后者 shuffle 是性能大头。
  - **Ray Data vs DataLoader（PyTorch）**：前者分布式（多节点 actor）、流式跨节点；后者单进程（多 worker 但同机）、不跨节点。前者是后者的分布式版。
  - **map vs map_batches**：前者逐元素；后者逐 batch。GPU/向量化用后者。

## 7. 常见误区与易错点

> [!warning] 误区 1：Ray Data 是 Spark 替代
> 不完全是。Ray Data 优化"流式喂训练"，不优化"大 join / 复杂 SQL ETL"。Spark 的 shuffle 优化、Catalyst 优化器、SQL 生态 Ray Data 没有。**两者常配合**：Spark 离线 ETL → Parquet → Ray Data 流式读喂训练。

> [!warning] 误区 2：Ray Data 全内存
> Ray Data 内存占用是**有界**（buffer 有界），不是全载数据集。但若 batch 太大 / pipeline 太深 / buffer 设太大，仍可能 OOM。需调 `target_max_block_size`、`prefetch` 等。

> [!warning] 误区 3：map 和 map_batches 等价
> 不等价。`map` 逐元素、单样本处理、CPU 友好；`map_batches` 逐 batch、可向量化/GPU、减少 per-sample overhead。**GPU 加载重的算子必用 `map_batches` + actor 池**，否则每 task 重载模型慢死。

> [!warning] 误区 4：背压自动即无瓶颈
> 背压防 OOM，但**不消除瓶颈**。最慢 actor 仍是瓶颈。要识别瓶颈 actor（profile）、并行化或调大其并行度（actor 池大小）。

> [!warning] 误区 5：split 后 worker 独立即均衡
> split 是**数据源级切分**（每 worker 读不同 shard），若某 worker 的 shard 慢（如某节点盘慢）、该 worker 慢成瓶颈。需 shuffle shard 或用共享 shuffle 后的 pipeline。

> [!tip] 实践：调优三问
> (1) GPU 利用率低？→ 增 prefetch 深度、用 actor 池 GPU 预处理。(2) OOM？→ 减 batch size / buffer / pipeline 深度。(3) 喂慢？→ 找瓶颈 actor（profile）、并行化。

## 8. 延伸细节

### 8.1 pipeline vs dataset

Ray Data 区分：
- **Dataset**：静态数据集，可重复迭代（每个 epoch 全跑）。
- **DatasetPipeline**：流式 pipeline，**窗口化**处理（不一次性构全 pipeline，按需逐段构）。

`ds.window(blocks_per_window=N)` 把 dataset 分窗，每窗构一段 pipeline，**适合超大数据集**（不一次构全 pipeline）。

### 8.2 与 object store 关系

Ray actor 间传数据走 **object store**（共享内存/磁盘）。同节点 actor 共享内存零拷贝（[[Arrow]]），跨节点序列化 + 网络。Ray Data 默认 batch 传 object、不大不小（典型 64-256MB object），平衡零拷贝与 GC。

### 8.3 与 streaming source

Ray Data 支持流式 source（Kafka、Kinesis、文件流）——source 不停产、pipeline 流式消费。适合在线推理数据流式喂模型。详见 Ray Data 文档"Streaming Ingestion"。

### 8.4 与 [[trajectory generation]]（RL）

强化学习 RL rollout：rollout worker 产 trajectory（state/action/reward），需流式喂 trainer 训 policy。Ray Data 的流式 + actor 模型适合——rollout worker 是 actor、产 trajectory 进 pipeline、trainer 消费。是 [[RL训推一体框架]] 的数据流基础。详见 [[trajectory generation]]。

### 8.5 内容来源

整理自 Ray Data 官方文档（"Ray Data"、"Streaming Ingestion"、API reference）、Ray 论文（Moritz et al. "Ray"）、Ray 源码（`ray/data/`）。与 Spark 的对比参考 [[Spark执行模型]]、[[Spark shuffle与partition]]。WebDataset/Parquet 结合见 [[WebDataset]]、[[Parquet]]。

---
相关: [[Spark执行模型]]、[[Spark shuffle与partition]]、[[Ray与分布式调度]]、[[WebDataset]]、[[Parquet]]、[[Arrow]]、[[actor model]]、[[trajectory generation]]
