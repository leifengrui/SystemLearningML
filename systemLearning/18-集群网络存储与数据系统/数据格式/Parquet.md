# Parquet

> **所属章节**: [[数据格式]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: Parquet / 列存压缩 / row group / predicate pushdown / 谓词下推 / Dremel 编码
> **难度**: 中（需懂 [[Arrow]] + 列存 + 压缩编码 + 数据扫描）


## 1. 一句话定义

**Parquet** 是面向**分析型负载**的**磁盘列式存储格式**——把一张表按 **row group（行组分块，每 group 几千~几十万行）** 切分，每个 row group 内部按**列**连续存储与压缩，同列数据类型一致因而**压缩比高**，并通过 **predicate pushdown（谓词下推）** 把过滤条件推到存储层（只读满足条件的 row group / 用 min-max 统计跳过无关页）、**column pruning（列裁剪）** 只读需要的列，大幅减少 IO；嵌套结构用 **Dremel 编码**（repetition/definition level）无损列式存储。AI 训练用 Parquet 的理由：**海量结构化/表格数据省存储省带宽**（语料元数据、特征表、标注表，压缩比 3–10×）、**列裁剪只读需要的列**（不要的列零 IO）、**谓词下推过滤行**（只读高质量样本）让数据扫描高效。与 [[Arrow]] 关系：Parquet 是**磁盘**列存（持久、压缩），Arrow 是**内存**列存（临时、零拷贝），常"Parquet 读盘解压 → Arrow 内存 → 计算 → 压回 Parquet"全列式无转换。与 [[WebDataset]] 对比：Parquet 适合**表格/结构化**（文本+标签+特征），WebDataset 适合**媒体流式**（图片/音频/视频任意二进制）。

> [!note] 三句话定位
> - **是什么**：磁盘列存格式——row group 分块 + 列式压缩 + 谓词下推 + Dremel 嵌套，省存储省 IO。
> - **为什么需要**：结构化训练数据海量，行存压缩低且读全行浪费；列存按需读列、按统计跳行，扫描快。
> - **与 [[Arrow]] 关系**：Parquet 磁盘列存 ↔ Arrow 内存列存，配套无转换。读 Parquet 直接得 Arrow Table。


## 2. 为什么需要它（动机与背景）

### 2.1 结构化数据海量，行存浪费

预训练语料元数据、特征表、标注表常是几亿~几十亿行的结构化表。CSV/JSON 行存：压缩比低（类型混杂）、读全行（不要的列也读）、无统计跳行。Parquet 列存：同列同类型压缩高、只读需要的列、用 min-max 跳行。

### 2.2 只读需要的列（列裁剪）

训练只用 `text` + `label` 两列，行存要读全行（含几十个不要的列）。Parquet 按列存，`columns=["text","label"]` 只读这两列，其余列零 IO。列越多收益越大。

### 2.3 按统计跳行（谓词下推）

每个 row group 的每列存 min/max/bloom。查询 `label > 0.5` 时，row group 的 `label` max ≤ 0.5 则整 group 跳过，不读。行存做不到这种跳过。

### 2.4 压缩省存储省带宽

同列同类型（如一列 int、一列重复 string）压缩比 3–10×。存对象存储 / [[并行文件系统]] 省容量，扫描省带宽。

### 2.5 与 Arrow 衔接

读 Parquet 用 pyarrow 直接得 Arrow Table（列存内存），无行列转换，向量化算。

> [!warning] 误区：Parquet 适合存图片/二进制
> 不适合。Parquet 是**列式表格**格式，存任意二进制（图片/音频）虽支持但压缩/扫描不优，且要全行结构。媒体数据用 [[WebDataset]] tar 流更合适。Parquet 存"图路径 + 文本 + 标签"的元数据表，图本身存 shard。


## 3. 核心概念详解

### 3.1 文件结构

```
Parquet 文件:
  Magic "PAR1"
  Row Group 1 (行组分块, 每组 N 行)
    Column Chunk (text)  [Page1, Page2...]   <- 列连续+压缩
    Column Chunk (label) [Page1, Page2...]
  Row Group 2
    ...
  Footer (schema, row group 统计: min/max/...
          + 列编码、压缩信息)
  Footer length + Magic "PAR1"
```

元数据在 footer，读 footer 即知每个 row group/列的统计与布局，据此跳过。

### 3.2 row group（行组）

表按行分块，每 row group 含所有列的 column chunk。row group 是**并行读 + 谓词下推**的单元：可多线程并发读不同 row group；用 row group 的列统计决定跳不跳。大小经验 128 MB–1 GB（HDFS block 对齐）。

### 3.3 列式压缩

每列独立压缩（同类型、值相似）。编码 + 通用压缩：
- **Dictionary encoding**：重复 string 存字典 id（"中国"→1），极大省空间。
- **Delta encoding**：递增 int 存差值。
- **RLE (Run-Length)**：重复值存 (值,次数)。
- 通用压缩：**Snappy**（快、压缩比中）、**ZSTD**（压缩比高、稍慢）、**Gzip**（高、慢）。ZSTD 是存储推荐（平衡）。

### 3.4 predicate pushdown（谓词下推）

把过滤条件（`label > 0.5 AND source = "web"`）推到存储层：
1. 用列 **min/max** 跳过整个 row group（group 的 label max ≤ 0.5 → 跳）。
2. 用 **bloom filter** 判断等值列（`source="web"`）是否在 group 出现过。
3. **page index**（可选）在 page 级再跳。

只读满足的行，IO 大减。

### 3.5 Dremel 嵌套编码

支持嵌套（list/struct/map）。用 **repetition level**（该层重复到第几层）+ **definition level**（该值定义到第几层，即哪些可选层存在）把嵌套结构无损列式存。读时按 level 还原。

### 3.6 与 Arrow 的衔接

`pyarrow.parquet.read_table` 读 Parquet → 直接返回 Arrow Table（内存列存），计算用 Arrow compute 向量化，写回 `write_table`。全列式无行列转换。详见 [[Arrow]] §3.4。

### 3.7 Parquet vs WebDataset vs Arrow

| 维度 | **Parquet** | [[Arrow]] | [[WebDataset]] |
|---|---|---|---|
| 层次 | 磁盘 | 内存 | 磁盘（tar 流） |
| 数据 | 表格/结构化 | 表格/结构化 | 媒体/任意二进制 |
| 存储 | 列存压缩 | 列存未压缩 | tar 打包 |
| 读 | 按列裁剪+跳行 | 零拷贝 | 顺序流式 |
| 场景 | 特征/标注/元数据 | 内存计算 | 图片/音频/视频 |


## 4. 数学原理 / 公式

### 4.1 压缩比与存储/带宽节省

原始行存大小 $D$，Parquet 列存压缩后 $D' = D / r$，$r$ 为压缩比（典型 3–10）。存储与扫描带宽都 ÷ $r$。

$$\text{存储} = \frac{D}{r},\quad \text{带宽} = \frac{D_{\text{read}}}{r}$$

例：1 TB 特征表，$r=5$ → 200 GB 存储，扫描 200 GB。行存 CSV 几乎不压缩仍 ~1 TB。

### 4.2 列裁剪的 IO 省

表 $C$ 列，只用 $c$ 列（$c\ll C$）。行存读全行 $D$；Parquet 列裁剪只读：

$$D_{\text{read}} \approx \frac{c}{C}\cdot\frac{D}{r}$$

$c=2$、$C=20$、$r=5$ → 读 $\frac{2}{20}\cdot\frac{1}{5} = 2\%$。省 50×。

### 4.3 谓词下推的行跳过

设过滤条件选出 $p$ 比例行（$0<p<1$），row group 统计能跳到只读含命中的 group，约读 $p+\epsilon$（含部分未命中 page）。结合列裁剪：

$$D_{\text{read}} \approx \frac{c}{C}\cdot\frac{D}{r}\cdot p$$

例 $p=0.1$ → 再 ×0.1。三重叠加（列裁剪 + 压缩 + 跳行）可让实际 IO 降到原行存的 0.2% 量级。

### 4.4 row group 并行

$G$ 个 row group，可 $G$ 线程并发读，吞吐 ≈ $\min(G\cdot BW_1, BW_{\text{storage}})$。配合对象存储分前缀 / [[并行文件系统]] 多 OST，扫描大规模 Parquet 可达数十 GB/s。


## 5. 代码示例（可选）

### 5.1 写 Parquet（指定压缩与 row group）

```python
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.table({
    "text": ["hello world"] * 1_000_000,
    "label": [0.7] * 1_000_000,
    "source": ["web"] * 1_000_000,
})
pq.write_table(table, "train.parquet",
               compression="zstd",        # 高压缩比
               row_group_size=128 * 1024) # 128K 行/row group
```

### 5.2 读：列裁剪 + 谓词下推

```python
import pyarrow.parquet as pq

# 只读需要的列 + 存储层过滤行 (predicate pushdown)
table = pq.read_table(
    "train.parquet",
    columns=["text", "label"],                  # 列裁剪
    filters=[("label", ">", 0.5),               # 谓词下推: 跳 row group
             ("source", "=", "web")],
)
print(table.num_rows)   # 已过滤, 只剩命中行
```

### 5.3 row group 级并行读

```python
pf = pq.ParquetFile("train.parquet")
print(pf.num_row_groups)
for i in range(pf.num_row_groups):
    rg = pf.read_row_group(i, columns=["text"])  # 单 row group, 可并发
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Arrow]]（内存列存搭子）、压缩编码、对象存储 / [[并行文件系统]]（存 Parquet 文件）。
- **下游（应用）**: AI 训练数据扫描（特征/标注/元数据表）、[[Ray Data]] / [[Spark执行模型]]（扫 Parquet 做 ETL）、湖仓（Iceberg/Delta 用 Parquet 底层）。
- **对比 / 易混**:
  - **Parquet vs [[Arrow]]**：磁盘列存压缩 ↔ 内存列存零拷贝。配套。
  - **Parquet vs [[WebDataset]]**：表格结构化（列存）↔ 媒体流式（tar 行存）。数据类型决定选型。
  - **Parquet vs CSV/JSON**：行存未压缩无统计跳行；Parquet 列存压缩有下推，分析负载快几倍~几十倍。


## 7. 常见误区与易错点

> [!warning] 误区 1：Parquet 适合所有数据
> 不。表格/结构化最优。图片/音频/任意二进制虽能存（BINARY 列）但压缩/扫描不优，且要全行结构。媒体用 [[WebDataset]]。

> [!warning] 误区 2：压缩越强越好
> ZSTD 压缩比高但解压 CPU 稍多。训练扫描若 CPU 是瓶颈，Snappy 解压更快。存储用 ZSTD，热扫描可 Snappy，权衡。

> [!warning] 误区 3：谓词下推一定过滤行
> 只对**列统计能判定**的条件（min/max/等值/范围）有效。复杂表达式（UDF、跨列计算）不能下推，回退到读后过滤。下推是"能跳 row group"的优化，非万能。

> [!warning] 误区 4：Parquet 小文件也快
> 小文件 footer 读取代价 + 对象存储每文件 GET 延迟，百万小 Parquet 仍慢。应合并成大文件（每文件 128 MB–1 GB）或用 [[WebDataset]] tar。

> [!tip] 实践：row group 与对象存储对齐
> row group 128 MB–1 GB，文件 512 MB–1 GB，既利压缩又利对象存储顺序读（少 GET）。配合分区目录（按 date/source 分），谓词下推按分区裁剪更高效。

> [!note] 补充：列式存储不可变
> Parquet 文件写后不可改（要改 = 重写整文件）。频繁更新用湖仓（Iceberg/Delta）在 Parquet 上加 manifest 做行级 upsert。训练数据一般批量写一次读多次，不可变无碍。


## 8. 延伸细节

### 8.1 元数据 footer

footer 含 schema、每 row group 每列的 min/max/null_count、编码、压缩。读时先读 footer 决定读哪些 page，随机定位高效。这也是谓词下推的依据。

### 8.2 page index

Parquet 2.9+ 可选 page index，每 page 也存 min/max，谓词下推可到 page 级（更细跳过），代价是 footer 更大。

### 8.3 内容来源

Parquet 格式规范、row group/predicate pushdown/Dremel encoding、压缩编码来自 Apache Parquet 官方文档与 Dremel 论文。`pyarrow.parquet` API 来自 pyarrow 文档。压缩比/扫描加速倍数为典型经验值，强依赖数据，关键数量级（3–10× 压缩、列裁剪/下推降 IO）稳定。

---
相关: [[数据格式]] | [[Arrow]] | [[WebDataset]] | [[Ray Data]] | [[Spark执行模型]] | [[本地NVMe与对象存储]] | [[并行文件系统]] | [[batch与mini-batch]]
