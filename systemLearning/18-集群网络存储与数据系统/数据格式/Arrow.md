# Arrow

> **所属章节**: [[数据格式]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: Apache Arrow / Arrow / 内存列存 / 列式内存格式 / Arrow Flight / 零拷贝 IPC
> **难度**: 中（需懂 [[Parquet]] + 列存 + 进程间通信 + 向量化计算）


## 1. 一句话定义

**Apache Arrow** 是一个**内存列式数据格式标准**——规定了一组**语言无关的列存内存布局规范**（同列数据在内存里连续排列、带类型与元数据），使得数据在**不同进程、不同语言（Python/R/Java/C++/Rust）之间共享内存时无需序列化/反序列化即可零拷贝传递**，并原生支持**向量化计算**（同列连续 → SIMD/批量算子高效）。三大支柱：**列存**（同列连续内存、类型一致、向量化友好）、**零拷贝 IPC**（进程间共享同一段内存，跨语言无需 serde）、**Arrow Flight**（基于 gRPC/Arrow IPC 的高速数据传输 RPC，流式批量传输）。AI 数据管道用 Arrow 的理由：**列存让向量化预处理（ tokenize、过滤、统计）高效**；**跨语言零拷贝**让 Python↔C++↔Rust 组件（如 tokenizer、ray task）无缝传数据；**与 [[Parquet]] 同列存理念**——Parquet 是磁盘列存、Arrow 是内存列存，常"Parquet 读盘 → Arrow 内存 → 计算 → Parquet 写盘"全列式无转换。对比 Pandas：Pandas 默认行存 + 跨进程要 pickle/serialize 开销大；Arrow 是现代数据栈（Pandas 2.0 Arrow backend、Ray Data、Polars、DuckDB）的内存底座。

> [!note] 三句话定位
> - **是什么**：语言无关的内存列存格式标准，跨进程/语言零拷贝共享 + 向量化计算 + Arrow Flight 高速传输。
> - **为什么需要**：数据管道跨语言/进程传数据若要序列化（pickle）开销大、列存缺失使向量化低效；Arrow 统一内存格式消除这两类开销。
> - **与 [[Parquet]] 关系**：Parquet 是磁盘列存（持久、压缩），Arrow 是内存列存（临时、零拷贝）。常配套：Parquet 读盘解压成 Arrow 内存，计算完再压回 Parquet。


## 2. 为什么需要它（动机与背景）

### 2.1 跨语言/进程传数据的序列化税

Python 数据处理常要跨进程/语言传表（如 Ray worker 之间、Python↔C++ tokenizer、R↔Python 统计）。传统做法：Pandas DataFrame → pickle/JSON/CSV 序列化 → 传输 → 反序列化 → 另一端 DataFrame。这每步都拷贝+转码，大数据集耗时耗内存。Arrow 让两端**共享同一段 Arrow 内存**，指针传递即用，零拷贝。

### 2.2 列存让向量化高效

机器学习数据预处理（tokenize、归一化、过滤、特征统计）对同列批量操作。列存把同列连续放，SIMD/向量化算子一次处理一整列，CPU cache 友好。行存（Pandas 默认按行 block）则要跨列跳读，向量化低效。

### 2.3 与 Parquet 同理念无缝衔接

[[Parquet]] 是磁盘列存。若内存格式也列存（Arrow），则"读 Parquet → 内存 Arrow → 算 → 写 Parquet"全列式无行列转换。若内存用行存（Pandas），则每次 IO 都要行列转换，开销大。Arrow 是 Parquet 的天然内存搭子。

### 2.4 现代数据栈的底座

Polars、DuckDB、Pandas 2.0（ArrowDtype backend）、Ray Data、Spark（用 Arrow 做 Python UDF 传输）都以 Arrow 为内存格式。统一格式让组件即插即用、零拷贝组装 pipeline。

> [!warning] 误区：Arrow 是数据库
> 不是。Arrow 是**内存格式标准 + 库**，不是存储引擎/数据库。它规定"数据在内存里怎么排"，不存盘（存盘用 [[Parquet]]）、不调度查询。配合 Flight/库做传输与计算，但本身是格式层。


## 3. 核心概念详解

### 3.1 列存内存布局

Arrow 表 = `Schema`（列名/类型/metadata）+ 一组 `Array`（每列一个，连续内存）。每个 Array 由：
- **validity bitmap**（null 位图，1 bit/元素）
- **values buffer**（连续的同类型值，如 int32 紧排）
- 可选 **offsets/child buffers**（变长 string/list/嵌套）

同列连续 → SIMD 向量化、cache 友好、压缩友好（同类型）。

### 3.2 零拷贝 IPC

两进程共享同一段 Arrow 字节缓冲（共享内存 mmap / 传 buffer 指针），接收方按 Arrow 规范直接解释为 Array/Table，**不拷贝、不序列化**。Arrow 的 IPC（`RecordBatch` 流）就是"把内存 buffer + schema 打包成连续字节流"，接收方零拷贝重建。

### 3.3 Arrow Flight

基于 gRPC/HTTP2 的数据传输 RPC：客户端 `do_get(flow)` 流式拉数据，`do_put` 流式写，`do_action` 调命令。传输用 Arrow IPC 格式，**零拷贝到内存表**。适合大规模数据服务（替代 ODBC/JDBC 拉表）。Flight SQL 在此上跑 SQL。

### 3.3 列存 vs 行存

```
行存 (Pandas block):        列存 (Arrow):
  [row0: a,b,c]              colA: [a0,a1,a2,a3...]   <- 连续, 同类型
  [row1: a,b,c]              colB: [b0,b1,b2,b3...]
  [row2: a,b,c]              colC: [c0,c1,c2,c3...]
跨列跳读, SIMD 难            连续, SIMD 友好, cache 好列操作
```

### 3.4 与 Parquet 的衔接

```
磁盘 Parquet (列存, 压缩)
   | 读 + 解压
内存 Arrow (列存, 零拷贝)
   | 向量化计算
内存 Arrow (结果)
   | 压缩
磁盘 Parquet
```

全列式无行列转换。PyArrow 的 `pq.read_table` 直接返回 Arrow Table（列存），计算完 `pq.write_table` 写回。

### 3.5 Ray Data / Pandas 用 Arrow

- **Ray Data**：内部用 Arrow Dataset，跨进程 task 传 Arrow record batch 零拷贝（Ray 对象序列化用 Arrow IPC）。
- **Pandas 2.0**：ArrowDtype backend，把底层从 NumPy block（行存倾向）换 Arrow 列存，省内存、快 string 操作。
- **Spark**：Python UDF 用 Arrow 把 JVM 数据传 Python（`spark.sql.execution.arrow.pyspark.enabled`），替代逐行 pickle，大幅提速。


## 4. 数学原理 / 公式

### 4.1 序列化税的消除

传统行存 + pickle，传 $D$ 字节表要：序列化（$O(D)$ CPU + 临时副本）+ 传输 + 反序列化（$O(D)$ CPU + 副本）。Arrow 零拷贝：共享内存，传 buffer 指针 $O(1)$ 拷贝（仅元数据），CPU 与内存都省。

$$T_{\text{serde}} \approx 2\cdot\frac{D}{BW_{\text{mem}}}\quad\text{vs}\quad T_{\text{arrow}}\approx O(1)$$

大数据集（GB 级）省几百毫秒到秒级。

### 4.2 列存向量化加速

列存同列连续，SIMD 一次处理 $W$ 元素（向量宽度）。行存跨列跳读无法向量化。同列批量算子（加、比较、filter）：

$$\text{speedup} \approx \frac{W_{\text{simd}}}{1}\cdot\frac{\text{cache hit}}{\text{improvement}}$$

string 列、过滤、统计类算子，Arrow/Polars 比 Pandas 快 5–50×（待核实，强依赖数据与算子）。

### 4.3 内存占用

列存同类型紧排，null 用位图（1 bit）而非整行占位，比行存省内存（尤其多 null）。与 [[Parquet]] 压缩后磁盘占用互补：Arrow 是内存未压缩（或轻量编码），Parquet 是磁盘压缩。


## 5. 代码示例（可选）

### 5.1 Arrow Table + 与 Pandas 零拷贝互转

```python
import pyarrow as pa
import pandas as pd

# 列存 Table
table = pa.table({
    "id": pa.array([1, 2, 3], type=pa.int32()),
    "text": pa.array(["a", "bb", "ccc"]),
})
print(table.column("id"))           # 列视图, 连续 int32

# Arrow <-> Pandas 零拷贝 (同内存, 不序列化)
df = table.to_pandas()              # 零拷贝包装
table2 = pa.Table.from_pandas(df)   # 零拷贝还原
```

### 5.2 零拷贝 IPC（进程间共享）

```python
import pyarrow as pa

# 生产端: 把 Table 打包成 IPC 字节流 (不拷贝内容, 只加元数据)
sink = pa.BufferOutputStream()
writer = pa.ipc.new_stream(sink, table.schema)
writer.write_table(table)
writer.close()
buf = sink.getvalue()

# 消费端: 零拷贝重建 Table (直接解释这段字节)
reader = pa.ipc.open_stream(buf)
table2 = reader.read_all()
assert table2.equals(table)
```

### 5.3 从 Parquet 读到 Arrow 内存再算

```python
import pyarrow.parquet as pq

table = pq.read_table("data/train.parquet",
                      columns=["text", "label"])   # 直接得 Arrow 列存 Table
# 向量化算 (过滤, 用 Arrow compute)
import pyarrow.compute as pc
filt = pc.greater(table["label"], 0.5)
filtered = table.filter(filt)                       # 向量化过滤, 快
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Parquet]]（磁盘列存搭子）、向量化计算、SIMD、进程间通信/共享内存。
- **下游（应用）**: [[Ray Data]]（Arrow 为内存底座）、Pandas 2.0 Arrow backend、Polars/DuckDB、Spark Python UDL、数据预处理 pipeline。
- **对比 / 易混**:
  - **Arrow vs [[Parquet]]**：Arrow 是**内存**列存（临时、未压缩/轻编码、零拷贝）；Parquet 是**磁盘**列存（持久、压缩、row group）。配套使用。
  - **Arrow vs Pandas**：Pandas 传统行存 block + 跨进程 pickle 开销；Arrow 列存 + 零拷贝 IPC。Pandas 2.0 可切 Arrow backend 兼得两者。
  - **Arrow vs [[WebDataset]]**：Arrow 是结构化表内存格式；WebDataset 是媒体/任意文件 tar 流。前者表格向量化，后者样本流式。


## 7. 常见误区与易错点

> [!warning] 误区 1：Arrow 是存储格式
> 不是。Arrow 是**内存**格式，不持久化。存盘用 [[Parquet]]。Arrow 与 Parquet 配套：内存 Arrow，磁盘 Parquet。

> [!warning] 误区 2：零拷贝就完全不拷贝
> 零拷贝指跨进程/语言传递时**不序列化**（共享同段内存）。但首次把数据建成 Arrow 仍要拷贝一次（从原格式转）。后续传递才零拷贝。

> [!warning] 误区 3：Arrow 一定比 Pandas 快
> 列存向量化对**列操作**（filter/agg/string）快；逐行 Python 回调（apply）仍慢，因要拆成 Python 对象。用 Arrow compute（`pyarrow.compute`）向量化才快。

> [!warning] 误区 4：Arrow Flight 是文件系统
> 不是。Flight 是 RPC 协议（拉/推数据流），不是 POSIX 文件系统。配合 Arrow IPC 格式传，但本身是传输层。

> [!tip] 实践：数据管道全程用 Arrow
> 从读 [[Parquet]] → Arrow Table → `pyarrow.compute` 向量化预处理 → Ray Data 跨进程零拷贝传 → 写 Parquet，全程列式零序列化，pipeline 吞吐最高。混入 Pandas 行存会引入行列转换与 pickle 税。


## 8. 延伸细节

### 8.1 嵌套类型

Arrow 原生支持 list/struct/union/dense&sparse list 等嵌套（与 Parquet 的 Dremel 编码对应），处理嵌套 JSON/特征无 flattening 损失。

### 8.2 Dataset 与分区

`pyarrow.dataset` 可按分区（hive/directory partition）扫描大量 Parquet，谓词下推 + 并行读。是 [[Ray Data]] / Polars 扫海量文件的后端。

### 8.3 内容来源

Apache Arrow 格式规范、IPC/Flight 协议、`pyarrow` API 来自 Arrow 官方文档与 python 库。列存向量化性能对比参考 Polars/DuckDB benchmark（具体倍数标"待核实"，强依赖算子/数据）。Ray Data / Pandas 2.0 用 Arrow 来自各自文档。

---
相关: [[数据格式]] | [[Parquet]] | [[WebDataset]] | [[Ray Data]] | [[Spark执行模型]] | [[本地NVMe与对象存储]]
