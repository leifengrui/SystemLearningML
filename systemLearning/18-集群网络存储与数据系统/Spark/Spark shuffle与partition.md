# Spark shuffle 与 partition

> **所属章节**: [[Spark]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: Spark shuffle / Spark partition / shuffle write & read / spill to disk / 数据倾斜 skew
> **难度**: 中-高（需懂 [[Spark执行模型]] 的 stage/task/partition 概念，shuffle 是 stage 切分根因）


## 1. 一句话定义

**Spark shuffle** 是 stage 之间数据**按 key 重分布**的过程——上游 stage 的 task 把自己 partition 的数据**按 key hash 分桶写本地磁盘**（shuffle write），下游 stage 的 task 跨网络**拉属于自己桶的数据**（shuffle read），是 [[Spark执行模型]] 切 stage 的边界、也是 Spark 性能大头（**网络 + 磁盘 IO**）；**partition** 是数据切片，决定并行度（partition 数 = task 数 = 同 stage 并行度）；**spill to disk** 是 shuffle write 时内存不够、数据**溢写磁盘**（默认 sort-based shuffle，先内存排序后写盘，最后 merge）；**skew（数据倾斜）** 是某 key 数据量极大、对应单个下游 task 慢成**长尾**。shuffle 慢的根因是"**必跨网络 + 必落盘**"——网络带宽和磁盘 IO 是硬开销，无法靠 stage 内流水消除，只能靠减少 shuffle 量（`repartition` 改 partition 数、`coalesce` 合并、skew 治理）、调优 partition 数（太少并行低、太多 overhead 大、shuffle 后 partition 数 = 上游 partition × 下游 partition 在 worst case 下）来缓解。

> [!note] 三句话定位
> - **是什么**：shuffle = stage 间按 key 重分布；partition = 数据切片（决定并行度）；spill = 内存不够溢写盘；skew = 某 key 倾斜致长尾。
> - **为什么 shuffle 慢**：必跨网络 + 必落盘，硬 IO 开销无法靠流水消除。
> - **与 [[Spark执行模型]] 关系**：shuffle 是 stage 切分边界，每个 shuffle 边界切一刀 stage。

## 2. 为什么需要它（动机与背景）

### 2.1 为什么要有 shuffle

[[Spark执行模型]] 的 stage 内只能流水**窄依赖**（一对一/多对一 partition 传递，如 map/filter）。但很多算子是**宽依赖**——**按 key 重分布**：

- `reduceByKey`：相同 key 必须**聚到同一 task** 才能合并，否则同一 key 散在多个 task、合并结果错误。
- `groupByKey`：相同 key 的所有 value 要聚一起。
- `join`：两个 RDD 的同 key 行要聚一起。

为了"相同 key 在同一下游 task"，上游 task 必须把数据**按 key 哈希分桶**（key hash → 桶编号 → 该桶数据写盘 → 下游 task 拉自己桶）。这就是 **shuffle**——本质是"按 key 重分布数据"。

### 2.2 为什么 shuffle 慢

shuffle 慢的根因是**必跨网络 + 必落盘**：

- **网络**：上游 task 的数据要发给多个下游 task（多对多），走网络/磁盘共享。
- **落盘**：上游 task 不能直接把数据塞下游 task 内存（下游可能还没调度、可能慢、可能内存不够），**先写本地磁盘**作中转。下游 task 跨网络拉磁盘上的 shuffle 数据。

所以 shuffle 的代价 = **磁盘写 + 磁盘读 + 网络传输 + 序列化**。一个 stage 内可流水（内存内连续跑），shuffle 没法流水——这是 stage 切分的根因。

### 2.3 为什么 partition 重要

**partition 数 = 并行度**。同 stage 内 partition 数越多、task 越多、并行越高（直到 CPU 饱和）。但 partition 数不是越多越好——

- **太少**：并行度低、CPU 不吃饱、长尾（一个 task 数据多慢）。
- **太多**：每 task 数据少、调度 overhead 大（task 启动/序列化/网络建链）、小文件多。

shuffle 后 partition 数 = `spark.sql.shuffle.partitions`（默认 200）或算子参数。**shuffle 是 partition 数重设点**——上游 partition 数与下游无关。

### 2.4 为什么 spill to disk

shuffle write 时，上游 task 要把所有输出按 key 分桶（每桶对应一个下游 partition）。若数据量大、内存装不下所有桶，**按桶溢写磁盘**（spill），最后多份 spill 文件 merge 成一份 shuffle 数据。这是**内存换磁盘**的妥协，**防止 OOM**。

> [!warning] 误区：shuffle 全在内存
> shuffle 默认走磁盘（shuffle write 写盘 + shuffle read 读盘）。有 `spark.shuffle.file.buffer`、`spark.reducer.maxSizeInFlight` 等缓冲参数加速，但**不脱离磁盘**。把 `spark.shuffle` 全内存需 unsafe 配置，常态不推荐。

## 3. 核心概念详解

### 3.1 shuffle write：上游按 key 分桶写盘

上游 stage 的 task 处理完自己的 partition，输出 `(key, value)`。要按 key 重分布到下游 partition：

1. **hash 分桶**：对 key 做 hash → 桶编号（`partition_id = hash(key) % num_partitions`）。
2. **写盘**：每桶数据写到本地磁盘的 shuffle 文件（`shuffle_{shuffle_id}_{map_id}_{reduce_id}.data` + `.index`）。
3. **spill**：内存不够先按桶写 spill 文件，最后 merge。

```
上游 task 0 (partition 0): 输出 (a,1) (b,2) (a,3) (c,4)
  按 key hash 分桶:
    桶 0 (下游 partition 0): (a,1) (a,3)
    桶 1 (下游 partition 1): (b,2)
    桶 2 (下游 partition 2): (c,4)
  写本地磁盘: shuffle_0_0_0.data, shuffle_0_0_1.data, shuffle_0_0_2.data
```

### 3.2 shuffle read：下游跨网络拉桶

下游 stage 的 task 处理一个 partition $j$，要**所有上游 task** 的桶 $j$：

1. **定位**：从 driver 拿 shuffle 元数据（哪个 executor 有桶 $j$ 的哪段）。
2. **跨网络拉**：从各上游 executor 拉自己桶 $j$ 的数据（并发拉、限并发数 `spark.reducer.maxSizeInFlight`）。
3. **合并**：把所有上游拉来的桶 $j$ 数据合并、排序、给 reduce 函数。

```
下游 task 0 (partition 0): 桶 0 = 上游 task 0 的桶0 + 上游 task 1 的桶0 + 上游 task 2 的桶0
  跨网络拉各 executor 的桶 0 数据
  合并排序 -> reduce 函数处理
```

### 3.3 sort-based shuffle（默认）

Spark 默认 **sort-based shuffle manager**（自 1.2 起）。流程：

1. 上游 task 输出先进内存 buffer（按 `partition_id` 排序）。
2. buffer 满 → spill 到磁盘（已排序）。
3. 多份 spill 文件 **merge** 成一份 shuffle 输出文件（带 `.index` 索引各 partition 偏移）。
4. 下游 task 读 `.index` 拿偏移 → 拉该 partition 数据。

**优点**：输出文件少（一上游 task 一份数据 + 一份 index，不爆炸）、可外部排序处理大数据。

> [!note] 补充：tungsten sort
> Spark 1.4+ 的 **tungsten sort**（ unsafe 内存 + 直接操作序列化字节）比 sort-based 更快——避免反序列化、原地排序二进制。是 `spark.shuffle.manager=sort` 的优化路径。

### 3.4 partition 数调优

| 场景 | partition 数 | 现象 |
|---|---|---|
| 太少（如 10） | 10 | 并行低、长尾、CPU 闲 |
| 适中（如 200） | 200 | 并行饱和、调度 overhead 可控 |
| 太多（如 10000） | 10000 | 调度 overhead 大、每 task 数据少、小文件多 |
| shuffle 后 | 默认 200（`spark.sql.shuffle.partitions`） | 与上游 partition 数无关 |

**经验**：partition 数 ≈ 总 CPU 核 × 2-4（防长尾）。大数据 shuffle 用 `spark.sql.shuffle.partitions` 调大、小数据调小（避免每 task 数据极少）。

```python
# 调全局 shuffle partition 数
spark.conf.set("spark.sql.shuffle.partitions", "400")

# 算子级指定
rdd.reduceByKey(func, numPartitions=400)
rdd.repartition(400)       # 强制 shuffle 重切
rdd.coalesce(50)           # 合并 partition, 尽量不 shuffle (适合写文件前减少小文件)
```

### 3.5 skew：数据倾斜致长尾

**skew** 是某 key 数据量极大（远超平均），shuffle 后该 key 对应的下游 task 拉巨量数据、慢成**长尾**——其他 task 秒级完成、它跑分钟级，整作业被它拖住。

```
均匀: task0 [1G] task1 [1G] task2 [1G]  -> 3 task 同步跑完 (1 分钟)
skew:  task0 [100M] task1 [100M] task2 [10G]  -> task2 跑 10 分钟 (长尾)
```

常见原因：某 key 占比极高（如 null、空字符串、热门 ID）。Spark UI 的 Stage 页面看 task 时间/读数据量分布，**最长 task 是长尾**。

### 3.6 治理 skew

| 方法 | 思路 | 代价 |
|---|---|---|
| **salting（加盐）** | 给倾斜 key 加随机盐 `$key_n` 分散到多 task，先局部 reduce，再去盐全局 reduce | 多一次 shuffle |
| **广播 join** | 小表 broadcast 到 executor 内存、避免 shuffle join | 小表要装内存 |
| **自定义 partitioner** | 按 key 量重分桶，避开 hash 均匀假设 | 业务知识 |
| **过滤异常 key** | null/空 key 先 filter 再 shuffle | 数据丢失风险 |
| **repartition** | 增 partition 数、稀释倾斜 | 总 shuffle 量不变 |
| **AQE 自适应**（Spark 3+） | `spark.sql.adaptive.enabled=true`，自动 split 长 partition | 框架代劳 |

```python
# salting 治理 skew (经典套路)
from pyspark.sql.functions import lit, col, rand, explode, array, split, concat

# 原始: 按 user_id join, user_id 某 ID 极多
# df1.join(df2, "user_id")  # skew!

# 加盐: 给 df1 的 user_id 加 1..N 随机盐
N = 100
df1_salted = df1.withColumn("salt", (rand() * N).cast("int")) \
                .withColumn("user_id_salted", concat(col("user_id"), lit("_"), col("salt")))

# 给 df2 的 user_id 复制 N 份 (每份加一个盐)
df2_salted = df2.withColumn("salt", array([lit(i) for i in range(N)])) \
               .withColumn("salt", explode(col("salt"))) \
               .withColumn("user_id_salted", concat(col("user_id"), lit("_"), col("salt")))

# 加盐后 join: 某 ID 分散到 100 个 task, 不长尾
joined = df1_salted.join(df2_salted, "user_id_salted")
```

### 3.7 shuffle 与 stage 关系

shuffle 是 [[Spark执行模型]] 切 stage 的**唯一规则**：

- 每个 shuffle 边界切一刀 → 两个 stage（上游 + 下游）。
- stage 内全是窄依赖、流水跑、不 shuffle。
- stage 间靠 shuffle 衔接、shuffle write + read。

shuffle 边界数 = stage 数 - 1。一个 DAG 有 $S$ 个 stage 就有 $S-1$ 次 shuffle。

### 3.8 partition 数 worst case 通信量

理论上，shuffle 是多对多：上游 $M$ 个 task、下游 $N$ 个 task，**最坏每对都通信** = $M \times N$ 条逻辑链路。实际通过磁盘中转 + 批量拉，**实际网络流量 = 总数据量** $D$（每字节传一次），但**连接数大**、磁盘随机读多。`spark.reducer.maxSizeInFlight` 限并发拉数（默认 48MB）防过载。

## 4. 数学原理 / 公式

### 4.1 shuffle 通信量

设上游 stage 有 $M$ 个 task、下游 $N$ 个 task、总 shuffle 数据量 $D$。

- **每下游 task 平均拉**：$D / N$（均匀分布时）。
- **若 skew**：某 key 占比 $\alpha$（$0 < \alpha \le 1$），对应 task 拉 $\alpha D$，远超平均 $D/N$。当 $\alpha \gg 1/N$，长尾。
- **网络总流量** $\approx D$（每字节传一次，假设无 broadcast）。

### 4.2 partition 与并行度

设集群有 $C$ 个 CPU 核、partition 数 $P$、每 partition 数据量 $D/P$。

- 并行度 $\min(P, C)$（同时跑的 task 数）。
- 每 task 处理 $D/P$ 数据。
- $P \ll C$ → CPU 闲、并行低、长尾。
- $P \gg C$ → 每 task 数据少、调度 overhead（每 task 固定启动开销 $\tau$）总 overhead $\approx P \cdot \tau$。

经验 $P \approx 2C \sim 4C$（防长尾 + 不过度调度）。

### 4.3 sort-based shuffle 内存模型

每上游 task 内存 buffer 大小 $B$（`spark.shuffle.file.buffer` 默认 32KB），数据量 $d_i$：

- 若 $d_i \le B$ → 一次写盘，无 spill。
- 若 $d_i > B$ → 多次 spill（每次 $B$），最后 merge $k = \lceil d_i / B \rceil$ 份。

merge 阶段需同时打开 $k$ 文件、$k$ 路归并，merge 内存 $\propto k$。$k$ 太大可能 OOM → 调大 `spark.shuffle.file.buffer` 减少 spill 数。

### 4.4 skew 加盐后的分散度

原倾斜 key 占 $\alpha$ 比例（数据量 $\alpha D$）。加 $N$ 盐后该 key 分散到 $N$ 个 task，每 task $\alpha D / N$。要 $< D/N$（均匀）则需 $N > \alpha N^2 / D$... 简化：选 $N$ 使 $\alpha D / N \le D/N'$，即 $N \ge \alpha N'$（$N'$ 为下游 partition 数）。

### 4.5 broadcast join 省网络

小表数据量 $B$、大表 $D$、shuffle join 网络流量 $\approx D + B$（两表都 shuffle）。

broadcast join：小表广播到所有 executor 内存（流量 $\approx B \cdot E$，$E$ 为 executor 数），大表不 shuffle，**网络流量 $\approx B \cdot E \ll D$ 当 $B \ll D$**。

| join 方式 | 网络流量 | 条件 |
|---|---|---|
| shuffle join | $D + B$ | 通用 |
| broadcast join | $B \cdot E$ | $B \ll D$ 且 $B$ 装内存 |

## 5. 代码示例（可选）

### 5.1 看 shuffle 量（Spark UI + 日志）

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("shuffle-demo") \
    .config("spark.sql.shuffle.partitions", "50") \
    .master("local[*]") \
    .getOrCreate()

df1 = spark.read.parquet("/path/to/big1")   # 大表
df2 = spark.read.parquet("/path/to/big2")   # 大表

# 触发 shuffle join
joined = df1.join(df2, "user_id")
joined.count()   # <- 触发: 看 Spark UI -> Stages -> shuffle read/write 量
```

### 5.2 调优：改 broadcast join 省网络

```python
# 小表广播到大表, 不 shuffle
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB 以下自动 broadcast

# 手动 broadcast
from pyspark.sql.functions import broadcast
joined = big_df.join(broadcast(small_df), "user_id")   # small_df 广播, 大表不 shuffle
```

### 5.3 治理 skew：salting

```python
# 见 3.6 节示例, 经典 salting 套路
# 也可开 AQE 让 Spark 自适应切长 partition
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

### 5.4 自定义 partitioner（业务知识分桶）

```python
rdd = sc.parallelize([("hot_key", 1)] * 1000 + [("cold", 1)] * 10, 4)

# 默认 hash partitioner: hot_key 全到一个 partition (skew)
# 自定义: 按 key 量手动分桶
from pyspark import Partitioner

class SkewAwarePartitioner(Partitioner):
    def __init__(self, numPartitions, hot_keys):
        self.numPartitions = numPartitions
        self.hot_keys = hot_keys   # 已知热 key 集合
    def numPartitions(self):
        return self.numPartitions
    def getPartition(self, key):
        if key in self.hot_keys:
            # 热 key 按其 hash 分散到多个 partition
            return (hash(key) % 1000) % self.numPartitions
        # 冷 key 按常规分
        return hash(key) % self.numPartitions

partitioned = rdd.partitionBy(SkewAwarePartitioner(64, {"hot_key"}))
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[Spark执行模型]]（shuffle 是 stage 切分边界、partition 是 task 切分依据）、[[Parquet]]（输入按 row group 切 partition）、[[Arrow]]（shuffle 数据格式优化、可零拷贝）。
- **下游（应用）**: 大数据 ETL（join/groupBy 全靠 shuffle）、特征工程聚合、Spark ML 训练数据准备。
- **对比 / 易混**:
  - **shuffle vs stage 内流水**：shuffle 必落盘跨网络、不能流水；stage 内窄依赖内存流水。前者慢、后者快。
  - **shuffle write vs shuffle read**：前者上游按 key 分桶写盘、后者下游跨网络拉桶。两阶段。
  - **partition vs block**：partition 是 RDD 切片（逻辑），block 是 HDFS 物理切片。partition 可对齐 block。
  - **repartition vs coalesce**：前者强制 shuffle 重切（全数据 shuffle），后者尽量不 shuffle 合并（窄依赖）。前者可加可减，后者主要减。
  - **shuffle vs [[Ray Data]] 的流式**：Spark shuffle 落盘 batch、有 stage 边界；Ray Data 流式无 stage、actor 直接流。

## 7. 常见误区与易错点

> [!warning] 误区 1：shuffle 在内存里
> 默认不在。shuffle write 写盘 + read 读盘，是磁盘 IO 大头。可通过增大 buffer 减少 spill 次数，但本质是磁盘。

> [!warning] 误区 2：partition 越多越快
> 不一定。过多 → 调度 overhead 大、小文件多、每 task 数据少（启动开销占主导）。最优 partition ≈ 核数 × 2-4。

> [!warning] 误区 3：repartition 不 shuffle
> 必 shuffle。`repartition` 是宽依赖、全数据 shuffle 重切。**减 partition 用 coalesce**（窄依赖合并、不 shuffle）。

> [!warning] 误区 4：skew 只能靠加核解决
> 加核治标不治本——倾斜 key 的 task 仍处理巨量数据。要靠 salting / broadcast join / AQE 自适应分散倾斜 key。

> [!warning] 误区 5：groupByKey 和 reduceByKey 等价
> **不等价**。`groupByKey` 把所有 value 原样 shuffle 到下游（网络量大、OOM 风险）；`reduceByKey` 在上游**先本地 reduce**（合并同 key）再 shuffle，**shuffle 量大幅减少**。**永远优先 reduceByKey**。

> [!warning] 误区 6：shuffle 数据均匀分布
> 默认 hash partitioner 假设 key 均匀。实际数据常 skew（热 key、null key），要监控 task 时间分布、用 salting/AQE 治理。

> [!tip] 实践：shuffle 调优三问
> (1) shuffle 量大不大？UI Stages 看 shuffle write/read。太大 → 改 broadcast join 或预聚合。(2) partition 数合不合适？太少加、太多减。(3) 有没有 skew？看最长 task 与平均 task 时间比，差距大则 salting/AQE。

## 8. 延伸细节

### 8.1 shuffle service（External Shuffle Service）

为**动态资源分配**（dynamic allocation）设计——executor 可被回收，但 shuffle 数据要保留供后续下游 read。ESS 把 shuffle 数据管理从 executor 抽出，独立守护进程（YARN 的 `NodeManager Auxiliary`）。executor 释放后 shuffle 数据不丢。

### 8.2 AQE（Adaptive Query Execution，Spark 3+）

Spark 3 引入自适应：作业运行时根据**实际统计**（partition 数据量、skew）动态调整——

- **自动合并小 partition**（post-shuffle coalesce）。
- **自动切大 partition**（skew join split）。
- **动态切换 join 策略**（小表自动 broadcast）。

开 `spark.sql.adaptive.enabled=true` 即可，是治理 shuffle/skew 的现代手段。

### 8.3 shuffle 与磁盘格式

shuffle 文件布局（sort-based）：

```
${local_dir}/blockmgr-*/shuffle_${shuffle_id}_${map_id}_${reduce_id}.data  # 数据
${local_dir}/blockmgr-*/shuffle_${shuffle_id}_${map_id}_0.index           # 索引 (各 reduce 偏移)
```

`.index` 让下游 task 按 reduce_id 直接 seek 到对应数据段，免扫整文件。`spark.local.dir` 可配多盘、用 SSD 加速。

### 8.4 与 [[Ray Data]] 的对比

| 维度 | Spark shuffle | [[Ray Data]] streaming |
|---|---|---|
| 数据流模型 | stage 落盘 + 批 shuffle | 流式、actor 间直接传 |
| 中间介质 | 磁盘（shuffle 文件） | 内存（object store / actor 内存） |
| 延迟 | 秒-分钟（批） | 毫秒-秒（流式低延迟） |
| 背压 | 隐式（下游慢则上游 shuffle 文件堆积） | 显式 backpressure |
| 适合场景 | 离线 ETL、大 join | 喂 GPU 训练、RL rollout |

详见 [[Ray Data]]。

### 8.5 内容来源

整理自 Spark 官方文档（"Tuning Spark"、"Shuffle Behavior"、AQE）、Spark 源码（`SortShuffleManager`、`IndexShuffleBlockResolver`、`MapStatus`）、Spark 论文与社区博客（skew 治理经典套路）。stage 切分关系见 [[Spark执行模型]]。

---
相关: [[Spark执行模型]]、[[Ray Data]]、[[Parquet]]、[[Arrow]]、[[Ray与分布式调度]]
