# Spark 执行模型

> **所属章节**: [[Spark]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: Spark execution model / DAG 执行 / stage-task 模型 / Spark 运行时架构
> **难度**: 中（需懂分布式数据并行 + RDD 抽象 + MapReduce 对比 + [[Ray与分布式调度]] 的通用调度思想）


## 1. 一句话定义

**Spark 执行模型** 是 Apache Spark 把**用户写的 transformation/action 代码**编译成 **DAG（有向无环图）**、再由 **DAGScheduler** 按 shuffle 边把 DAG **切成 stage**、每个 stage 内按 partition **切成 task**、由常驻进程 **executor** 跑 task 的全流程——driver 负责 DAG 构建 + 调度、executor 负责实际计算、cluster manager（YARN/K8s/Standalone）负责资源分配，三层协作完成分布式批作业。核心抽象是 **RDD/DataFrame 的 transformation 懒执行建 DAG，action 触发实际执行**，**partition 是数据切片**，**stage 是 shuffle 边界划分的 task 集合**，**task 是单个 partition 上的计算单元**。相比 MapReduce 只能"一个 map → 一个 reduce"两段式、中间结果必落盘，Spark 用 **DAG + 内存缓存**把多个窄依赖算子**流水化在一个 stage 内**、shuffle 边界才落盘，大大减少磁盘 IO，是 Spark 比 MR 快一个数量级的关键。与 [[Ray与分布式调度]] 比，Spark 是**面向数据并行的批处理框架**（stage 落盘、actor 不常驻），Ray 是**面向通用 actor/task 的调度框架**（actor 常驻、低延迟、强化学习/推理友好），定位不同。

> [!note] 三句话定位
> - **是什么**：用户代码 → DAG → DAGScheduler 切 stage（按 shuffle 边）→ TaskScheduler 切 task（按 partition）→ executor 跑 task。
> - **为什么**：用 DAG 把可流水的窄依赖合并到同 stage、减少中间落盘；shuffle 边界强制切 stage（shuffle 需跨网络+落盘，不能流水）。
> - **与 [[Ray与分布式调度]] 区别**：Spark 是数据并行批框架（stage 落盘、task 一次性），Ray 是通用 actor/task 调度（actor 常驻、流式喂 GPU）。

## 2. 为什么需要它（动机与背景）

### 2.1 MapReduce 的痛点

MapReduce（Hadoop MR）模型是 **map → shuffle → reduce** 三段：map 把输入切片并行处理、shuffle 按 key 重分布、reduce 聚合。问题在于**每段中间结果必须落 HDFS**（磁盘 + 三副本）：

- 一个多轮的迭代作业（如 PageRank、机器学习梯度下降），每轮 map 输出都写 HDFS、下轮 reduce 再读 HDFS，**磁盘 IO 占大半时间**。
- MR 只能"map+reduce"两段，**复杂作业要手写多个 MR 串联**，每个串联处都落盘。
- 中间结果磁盘 + 网络 = 性能瓶颈大头。

Spark 的核心动机：**能不能把多步算子合并到一个 DAG，内存里流水执行，只在必要时（shuffle）落盘？**

### 2.2 RDD：可容错的内存抽象

Spark 提出 **RDD（Resilient Distributed Dataset）**——只读、分区的分布式数据集，记录血缘（lineage：怎么从父 RDD 算来）。**RDD 不存数据本身，存"怎么算出它"的函数链**。算到一半失败时，按血缘重算丢失 partition，**不需落盘副本**。这给了"内存计算 + 容错"基础。

### 2.3 DAG：把算子链合并

用户写 `rdd.map().filter().reduceByKey()` 时，**transformation 是懒执行**（只建 DAG 节点，不真算）。当遇到 **action**（如 `collect()`、`count()`）才**触发 DAGScheduler** 把这些节点组成 DAG、切 stage、发 task。

- **窄依赖**（map、filter、union）：父 partition 一对一/多对一给子 partition，**可流水**（同 task 内顺序跑）。
- **宽依赖**（reduceByKey、groupByKey、join）：父 partition 多对多给子 partition，**需 shuffle**（跨网络重分布 + 落盘），**不能流水**。

DAGScheduler 在**宽依赖边界切 stage**：一个 stage 内全是窄依赖，可流水成一个 task chain；stage 之间靠 shuffle 衔接。

### 2.4 三层架构：driver / executor / cluster manager

- **driver**：跑用户 main 函数，建 SparkContext，**建 DAG + 切 stage + 调度 task**，与 executor 通信收结果。是"指挥官"。
- **executor**：在 worker 节点起的**常驻 JVM 进程**，跑 task、存 cache（内存数据）、写 shuffle 数据。是"工人"。
- **cluster manager**（YARN/K8s/Standalone/Mesos）：给 driver 分配 executor（CPU/内存）。

> [!tip] 实践：driver 是单点
> driver 是单 JVM，**不跑 task**（只调度）。若 DAG 巨大、task 数极多，driver 调度成瓶颈（事件循环忙）。`spark.driver.cores` / `spark.driver.memory` 要给够。driver 挂 = 作业挂（无 HA 默认）。

## 3. 核心概念详解

### 3.1 transformation vs action

| 类型 | 语义 | 是否触发执行 | 典型算子 |
|---|---|---|---|
| **transformation** | 返回新 RDD/DataFrame，**懒** | ❌ 只建 DAG 节点 | `map`、`filter`、`flatMap`、`union`、`mapPartitions`、`join`、`groupByKey`、`reduceByKey`、`distinct` |
| **action** | 返回非 RDD 结果（标量 / 集合 / 写盘） | ✅ 触发 DAG 执行 | `collect`、`count`、`take`、`reduce`、`saveAsTextFile`、`foreach` |

> [!warning] 误区：以为 transformation 立刻执行
> `rdd.filter(x => x > 0)` 调用完**啥都没发生**——只是 DAG 多了一个节点。直到 `rdd.count()` 才真正触发从 source 读数据 → filter → count 的全链路执行。这是 Spark 懒执行的本质。调试时常见"代码没报错但没输出"，因为没 action。

### 3.2 partition：数据切片

**partition** 是 RDD/DataFrame 的**水平切片**，每个 partition 是一段数据，**分配给一个 task 处理**。partition 数 = **并行度**（同 stage 内同时跑的 task 数）。

```
RDD[1,2,3,4,5,6,7,8]  partitions=4
  -> partition 0: [1, 2]
  -> partition 1: [3, 4]
  -> partition 2: [5, 6]
  -> partition 3: [7, 8]
  -> 4 个 task 并行处理
```

- 初始 partition 数 = 输入切片数（如 HDFS block 数、`defaultParallelism`）。
- **shuffle 后 partition 数** = `spark.sql.shuffle.partitions`（默认 200）或 `reduceByKey` 的 numPartition 参数。
- **partition 太少**：并行度低，CPU 不吃饱，长尾。
- **partition 太多**：调度 overhead 大（task 启动/序列化/网络），小 partition 浪费。

### 3.3 stage：shuffle 边界划分的 task 集合

DAGScheduler 把 DAG 按 **宽依赖**（shuffle 边）切 stage。一个 stage = 一段**可流水**的算子链（窄依赖），**stage 间靠 shuffle 衔接**。

```
源 RDD -> map    -> filter -> reduceByKey(shuffle!) -> map -> collect
  |_______ stage 0 (窄, 流水) _____|____ stage 1 (窄) ___|
                       ^
                  shuffle 边界 (切 stage)
```

- **stage 内**：窄依赖，partition 一对一传递，**同 task chain 跑完所有算子**，不落盘。
- **stage 间**：宽依赖，shuffle，**上游 task 写 shuffle 数据到本地盘**，下游 stage 的 task 跨网络**拉自己 key 对应的数据**。

> [!note] 补充：stage 编号
> DAGScheduler 从最后（action 端）开始倒推切 stage，**最后的 stage 是 stage 0**（final stage），往前递增 stage 1、2…。所以日志里 stage 0 通常是 final stage，stage 编号越大越靠数据源。

### 3.4 task：单个 partition 上的计算单元

**task = stage 内对单个 partition 执行算子链**。一个 stage 有 $P$ 个 partition 就有 $P$ 个 task。task 是 Spark 最小调度单位。

- task 在 executor 的**线程池**里跑（一个 executor 多线程、共享内存）。
- task 失败自动重试（默认 4 次），重试换 executor 跑（推测执行 speculative 防慢节点）。
- task 数 = partition 数，**task 数过少** = 并行低，**过多** = 调度 overhead。

### 3.5 executor：常驻进程跑 task

executor 是 driver 通过 cluster manager 在 worker 上启动的 **JVM 进程**，**常驻作业生命周期**。executor 负责：

- 跑 task（线程池内）。
- 存 RDD cache（`rdd.persist()` 把 partition 存 executor 内存/磁盘）。
- 写 shuffle 数据到本地盘（shuffle write）。
- 与 driver 通信（心跳 + task 状态 + 结果）。

```
┌─────── driver (1 JVM) ──────────┐
│ SparkContext / DAGScheduler     │
│ TaskScheduler / 调度 task       │
└──┬───────────────┬──────────────┘
   │  分配 executor │  发 task
   ▼               ▼
cluster manager (YARN/K8s/...)
   │
   ▼
┌── executor (worker 1) ──┐  ┌── executor (worker 2) ──┐
│ task 线程池 (4 线程)     │  │ task 线程池 (4 线程)     │
│ RDD cache / shuffle 数据 │  │ RDD cache / shuffle 数据 │
└─────────────────────────┘  └─────────────────────────┘
```

### 3.6 DAGScheduler / TaskScheduler / cluster manager 三层

| 组件 | 职责 | 数量 |
|---|---|---|
| **driver**（含 SparkContext、DAGScheduler、TaskScheduler） | 建 DAG、切 stage、切 task、发 task、收结果 | 1 个 JVM |
| **DAGScheduler** | 把 DAG 按 shuffle 切 stage，提交 stage 的 TaskSet 给 TaskScheduler | driver 内 |
| **TaskScheduler** | 把 TaskSet 调度到 executor（按 locality、资源） | driver 内 |
| **executor** | 跑 task、存 cache、写 shuffle | N 个（按资源） |
| **cluster manager** | 给 driver 分 executor 资源 | 集群层 |

### 3.7 为什么 DAG 切 stage

切 stage 的唯一规则：**宽依赖（shuffle）边界**。

- **窄依赖**：父 partition 一对一/多对一给子 partition（map/filter/union），**可流水**——同 task 内连续跑算子链，数据不落盘、不过网络。
- **宽依赖**：父 partition 多对多给子 partition（reduceByKey/groupByKey/join），**必须 shuffle**——上游 task 写磁盘、下游 task 跨网络拉，**不能流水**。

所以 DAGScheduler 在**每个宽依赖处切一刀**，**stage 内全是窄依赖、可流水**，**stage 间靠 shuffle 落盘**。这把"必须 shuffle 的次数"压到最少——可流水的算子全合并，**减少中间落盘**。

```
rdd.map(f).filter(g).reduceByKey(h).map(k).collect()
  |___ stage 1 (map+filter 流水) ___|___ shuffle ___|___ stage 0 (map, final) ___|
```

### 3.8 与 MapReduce 对比

| 维度 | MapReduce | Spark |
|---|---|---|
| 计算模型 | map → shuffle → reduce 两段 | DAG，多算子链 + 多 stage |
| 中间结果 | **每段落 HDFS**（三副本磁盘） | stage 内流水内存、**仅 shuffle 落盘** |
| 内存利用 | 几乎不用（落盘为主） | **RDD cache 进内存**、shuffle 也可内存 |
| 多轮迭代 | 每轮新 MR job、重新读盘 | **同一 DAG 多次 action**、cache 复用 |
| 容错 | 落盘三副本 | RDD 血缘重算（落盘 cache 也行） |
| 编程 | map/reduce 两函数 | `map/filter/reduceByKey/...` 丰富算子 |
| 性能 | 慢（磁盘大头） | 内存 + DAG 流水，**快 10-100×**（典型） |

> [!note] 补充：Spark 不是"全内存"
> Spark 也落盘——shuffle 数据默认写本地盘、cache 满了 spill 磁盘。Spark 的"内存计算"指的是 **stage 内流水 + cache 倾向内存**，不是不碰盘。详见 [[Spark shuffle与partition]]。

### 3.9 与 Ray 对比

| 维度 | Spark | [[Ray与分布式调度]] |
|---|---|---|
| 定位 | 数据并行**批处理**框架 | **通用** actor/task 调度框架 |
| 计算单元 | task（一次性，跑完释放） | actor（常驻进程）+ task |
| 中间数据 | stage 间 shuffle 落盘 | actor 内存共享、object store |
| 调度粒度 | DAG stage → task 批量发 | 单 task / 单 actor method |
| 延迟 | 秒-分钟（批） | 毫秒-秒（流式、低延迟） |
| 典型场景 | 离线 ETL、批统计 | RL rollout、在线推理、训练 pipeline |
| 数据并行 | ✅ 内置 | 通过 [[Ray Data]] / 数据并行 API |
| 状态管理 | RDD cache（短生命周期） | actor 长期状态、可变异 |

> [!tip] 实践：选 Spark 还是 Ray
> 离线大数据 ETL/统计 → Spark（成熟、SQL 生态、shuffle 优化深）。AI 训练数据 pipeline / RL rollout / 在线推理 → Ray（actor 常驻、低延迟、喂 GPU 流式）。详见 [[Ray Data]] 的对比。

## 4. 数学原理 / 公式

### 4.1 task 数与 stage 数关系

设一个 DAG 有 $S$ 个 stage，stage $i$ 有 $P_i$ 个 partition。则：

$$\text{total\_tasks} = \sum_{i=0}^{S-1} P_i$$

每个 stage 的并行度 = 该 stage 的 $P_i$。若某 stage 的 $P_i$ 远小于其他（瓶颈），整体被它拖住。**shuffle 后 partition 数**通常取 `spark.sql.shuffle.partitions`（默认 200），与上游无关，**shuffle 是 partition 数重设点**。

### 4.2 窄 vs 宽依赖的 partition 关系

窄依赖（map/filter）：子 partition $i$ ← 父 partition $i$（一对一）：

$$P_{\text{child}} = P_{\text{parent}}$$

宽依赖（reduceByKey）：子 partition $j$ ← 父所有 partition 的部分（多对多）：

$$P_{\text{child}} = \text{numPartitions}_{\text{param}} \quad (\text{默认 } \texttt{spark.sql.shuffle.partitions}=200)$$

### 4.3 stage 内流水省落盘

MapReduce 模型下，$K$ 个算子串联（map → filter → ... → reduce）需 $K$ 次中间落盘，总 IO：

$$\text{IO}_{\text{MR}} = K \cdot N \cdot B$$

$N$ 为数据量、$B$ 为每字节副本因子（HDFS 三副本 = 3）。

Spark DAG 把 $K$ 个**窄依赖**算子流水到同 stage、同 task 内顺序跑，**零中间落盘**：

$$\text{IO}_{\text{Spark, narrow}} = 0 \quad (\text{stage 内})$$

仅 stage 间 shuffle 落盘（$S-1$ 次 shuffle）：

$$\text{IO}_{\text{Spark, shuffle}} = (S-1) \cdot N$$

当 $K \gg S$（窄依赖多、shuffle 少），Spark 比 MR 节省大量 IO。

### 4.4 容错重算代价

RDD 失败 partition 按血缘重算。设血缘链长 $L$、单算子重算代价 $C_i$：

$$\text{recompute\_cost} = \sum_{i=1}^{L} C_i \cdot (\text{该 partition 涉及数据量})$$

只有失败 partition 重算（不是全数据），但若 cache 全丢、血缘长，重算代价大。**所以宽依赖前置 checkpoint**（lineage 太长时落盘一次切断）。

## 5. 代码示例（可选）

### 5.1 wordcount 看 stage / partition

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("wordcount-stage-demo") \
    .master("local[*]")      # local 模式, * = 用本机所有核
    .getOrCreate()

sc = spark.sparkContext

# 读 HDFS/本地文件, 默认按 block 切 partition
rdd = sc.textFile("/path/to/input.txt")

# transformation: 懒, 只建 DAG 节点
words = rdd.flatMap(lambda line: line.split(" "))   # 窄依赖
pairs = words.map(lambda w: (w, 1))                 # 窄依赖
counts = pairs.reduceByKey(lambda a, b: a + b)      # 宽依赖 (shuffle!)

# action: 触发执行, DAGScheduler 切 2 个 stage:
#   stage 1: textFile -> flatMap -> map (窄, 流水) -> shuffle write (按 word hash 分桶)
#   stage 0: reduceByKey (shuffle read) -> collect  (final stage)
result = counts.collect()   # <- 这里才真正跑
print(result[:5])
```

> [!note] 补充：怎么看 stage
> 在 Spark UI（http://driver:4040）的 Stages 页能看到 stage 列表、task 数、shuffle read/write 量。本地跑可用 `sc.setLogLevel("INFO")` 看日志的 `Submitting ... tasks` 行。

### 5.2 调 partition 数（并行度）

```python
# 方式 1: 算子参数
pairs.reduceByKey(lambda a, b: a + b, numPartitions=64)  # shuffle 后 partition=64

# 方式 2: 全局配置
spark.conf.set("spark.sql.shuffle.partitions", "64")

# 方式 3: repartition 强制重切 (触发 shuffle)
rdd.repartition(100)  # 宽依赖, 全数据 shuffle

# 方式 4: coalesce 减 partition (尽量不 shuffle, 合并)
rdd.coalesce(10)      # 窄依赖合并 partition, 不 shuffle (适合写文件前减少小文件)
```

### 5.3 cache 复用

```python
# 缓存: action 触发后, partition 存 executor 内存, 下次 action 不重算
counts.persist()                       # 默认 MEMORY_AND_DISK
counts.cache()                          # = persist(MEMORY_ONLY)
# 多次 action 复用同一中间结果
print(counts.count())                  # 第 1 次: 算 + cache
print(counts.filter(lambda x: x[1]>10).count())  # 第 2 次: 直接读 cache, 不重算
counts.unpersist()                      # 用完释放
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[Parquet]]（输入格式、按 row group 切 partition）、[[Arrow]]（DataFrame 底层列存、Spark 与 Pandas/Python 交换格式）、[[Ray与分布式调度]]（通用分布式调度思想对照）。
- **下游（应用）**: [[Spark shuffle与partition]]（shuffle 是 stage 切分边界、是性能大头）、大数据 ETL、批统计、特征工程、ML 训练数据预处理。
- **对比 / 易混**:
  - **Spark 执行模型 vs MapReduce**：前者 DAG + 内存流水多 stage、中间仅 shuffle 落盘；后者两段式 map+reduce、每段落 HDFS。
  - **Spark vs [[Ray与分布式调度]]**：前者数据并行批框架（task 一次性、stage 落盘）；后者通用 actor/task 调度（actor 常驻、低延迟、喂 GPU/RL 友好）。
  - **Spark vs [[Ray Data]]**：前者 stage 落盘 batch 模型；后者流式 streaming execution + actor 进程模型，更适合喂 GPU 训练低延迟流式。
  - **DAGScheduler vs TaskScheduler**：前者切 stage（DAG 级），后者调度 task 到 executor（执行级）。

## 7. 常见误区与易错点

> [!warning] 误区 1：transformation 会执行
> 不会。`map/filter/reduceByKey` 都是懒，只建 DAG 节点。必须有 action（collect/count/save...）才触发执行。调试时"代码没报错没输出"往往是没 action。

> [!warning] 误区 2：partition 数等于 CPU 核数
> 不是。partition 数是数据切片数，决定并行度。CPU 核数决定能同时跑几个 task。partition > 核数则排队；partition < 核数则 CPU 闲。最优 partition 数 ≈ 核数 × 2-4（防长尾）。

> [!warning] 误区 3：Spark 不用磁盘、全内存
> 不是。shuffle 数据**默认写本地盘**、cache 满了 spill 磁盘、大 shuffle 也 spill。Spark 是"倾向内存 + 必要时落盘"，不是全内存。shuffle 落盘是性能大头（详见 [[Spark shuffle与partition]]）。

> [!warning] 误区 4：stage 0 是数据源端
> 反了。DAGScheduler 从 action 端倒推切 stage，**最后（final stage）是 stage 0**，往前递增。stage 0 通常是输出/聚合端，最大 stage 编号靠数据源。

> [!warning] 误区 5：driver 跑 task
> 不跑。driver 只建 DAG + 调度 + 收结果。task 在 executor 跑。若 driver 也算大数据（如 `collect()` 把全量拉回 driver），driver OOM 挂。大数据用 `take(n)` 或 `saveAsTextFile`。

> [!warning] 误区 6：窄依赖都不落盘
> stage 内窄依赖流水不落盘。但**cache 满了**窄依赖结果也可能 spill 磁盘。流水是"可流水"，不是"必内存"。

> [!tip] 实践：调优三步
> (1) 看 Spark UI Stages 页找**长尾 stage**（最慢的）。(2) 看该 stage 是 shuffle 重（看 shuffle read/write 量）还是 task 数少（partition 少）。(3) shuffle 重 → 调 `spark.sql.shuffle.partitions` 或解决 skew；task 少 → 加 partition。详见 [[Spark shuffle与partition]]。

## 8. 延伸细节

### 8.1 广播变量与累加器

- **广播变量（broadcast）**：driver 把只读大对象（如 lookup 表）广播到所有 executor 内存，**避免每 task 序列化拷贝**。
- **累加器（accumulator）**：executor 端只增的计数器，driver 端聚合。常做 task 内统计（如过滤掉的样本数）。

### 8.2 数据本地性

TaskScheduler 调度 task 时优先**数据本地性**（data locality）：把 task 调度到数据所在节点（process local）→ 同机（node local）→ 同机架（rack local）→ 跨机架（any）。本地性越好越省网络，但等待本地性会延迟调度（`spark.locality.wait` 默认 3s 等待）。

### 8.3 推测执行

慢 task（speculative）会被复制到另一 executor 并行跑，先完成的为准。防慢节点（straggler）拖整作业。开 `spark.speculation=true`。

### 8.4 与 SQL / DataFrame

DataFrame/SQL 也是同套执行模型——Catalyst 优化器把 SQL/DF 算子优化后转 RDD DAG，仍走 DAGScheduler → stage → task。区别在 DataFrame 有**整体优化**（谓词下推、列裁剪、Join 重排），RDD 是用户手写无优化。

### 8.5 内容来源

整理自 Spark 官方文档（"Spark Programming Guide"、"Tuning Spark"、"Job Scheduling"）、Spark 论文（Zahari et al. "Resilient Distributed Datasets"）、Spark 源码（`DAGScheduler.scala`、`TaskSchedulerImpl.scala`）与社区教程。与 MapReduce 的对比参考 Hadoop MR 文档。与 Ray 的对比见 [[Ray与分布式调度]]、[[Ray Data]]。

---
相关: [[Spark shuffle与partition]]、[[Ray Data]]、[[Ray与分布式调度]]、[[Parquet]]、[[Arrow]]
