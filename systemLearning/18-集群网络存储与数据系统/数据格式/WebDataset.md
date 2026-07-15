# WebDataset

> **所属章节**: [[数据格式]]
> **所属模块**: [[18-集群网络存储与数据系统]]
> **别名**: WebDataset / tar shard / shard / 顺序读 / 大规模训练数据加载 / wds
> **难度**: 中（需懂 [[本地NVMe与对象存储]] + 顺序 vs 随机读 + 分布式数据并行 + [[batch与mini-batch]]）


## 1. 一句话定义

**WebDataset** 是面向**大规模训练**的**数据加载格式与库**——把大量样本（图/文/音/任意二进制）打包成**大 tar 文件（称为 shard）**，每个 tar 内含按样本分组的多个文件（如 `sample-000001.jpg`、`sample-000001.txt`、`sample-000001.cls`），训练时**顺序流式读 tar**（不解压到磁盘、不需随机 seek），**对每个 shard 内顺序读、跨 shard 分布式分配**（不同 worker 读不同 shard），把"百万小文件随机读慢"变成"少量大 shard 顺序读吞吐高"，**特别契合对象存储**（对象存储顺序 GET 大对象高效、随机 GET 小文件慢且 LIST 慢）。三大要素：**shard**（tar 分片，64 MB–1 GB，分布式读单元）、**顺序读**（流式 tar，无随机访问，对象存储友好）、**shuffle**（shard 内可随机乱序 buffer、跨 shard 顺序保留）。大规模训练必须用 WebDataset 的理由：百万小文件随机读——对象存储每 GET 10–100 ms、LIST 1000/页慢、本地盘随机 IOPS 也撑不住，**打包成大 shard 顺序读**把单盘/单对象带宽（GB/s 级）吃满，吞吐比小文件随机读高几十倍；同时 tar 是流式格式，可直接 `pipe:` 从 `aws s3 cp -` 流式拉，不落本地盘。与 [[Parquet]] 对比：Parquet 是**列存结构化表格**（文本+标签+特征，列裁剪压缩），WebDataset 是**行式媒体流**（图/音/任意二进制样本流式）；与 [[Arrow]] 对比：Arrow 是内存列存，WebDataset 是磁盘 tar 流。常配套：媒体样本用 WebDataset shard，元数据/特征用 Parquet，内存用 Arrow。

> [!note] 三句话定位
> - **是什么**：样本打包成大 tar shard，顺序流式读 + 按 shard 分布式分配，变随机小文件读为顺序大对象读。
> - **为什么需要**：百万小文件随机读慢（对象存储 GET/LIST 慢、本地盘随机 IOPS 撑不住），打包顺序读吞吐高几十倍且对象存储友好。
> - **与 [[Parquet]] 关系**：Parquet 列存表格结构化，WebDataset 行式媒体流。数据类型决定选型：媒体用 wds，表格用 Parquet。


## 2. 为什么需要它（动机与背景）

### 2.1 百万小文件随机读慢

预训练语料/多模态数据常百万~亿级样本。若每样本一文件直接读：
- **对象存储**：每文件一次 GET（10–100 ms），单连接 20 样本/s，百万样本要小时级；且 LIST 1000/页翻页慢。
- **本地 NVMe**：随机 4K IOPS 虽百万级，但每文件 open/stat 元数据 + 句柄开销大，吞吐仍远低于顺序 GB/s。
- **[[并行文件系统]]**：元数据服务被百万 open 打爆。

打包成大 shard 顺序读，把吞吐从"每请求几十 μs–几十 ms 的随机小请求"提升到"GB/s 的顺序流"，几十倍提升。

### 2.2 对象存储顺序读友好

对象存储对"一个大对象一次 GET（或 Range）"高效（带宽打满），对"百万小对象各 GET"差（每请求延迟 + 速率限制）。WebDataset tar 是一个大对象，顺序流式 GET 最契合对象存储。可直接 `pipe:aws s3 cp s3://bkt/shard-001.tar -` 流式拉不落盘。

### 2.3 分布式并行读

多 worker / 多节点训练，每个 worker 读不同 shard，互不冲突，天然并行。shard 是分布式读的最小分配单元。

### 2.4 不解压、流式

tar 读时流式解包到内存，不落本地磁盘（省盘空间、省二次 IO）。`IterableDataset` 风格，适配大规模训练。

> [!warning] 误区：shard 内强随机 shuffle
> shard 是 tar 顺序流，shard 内**无法高效随机跳读**（要 seek 整个 tar）。随机靠**缓冲式 shuffle**（读一段进 buffer 乱序吐）或**跨 shard shuffle**（多 shard 交错）。要严格逐样本随机需先打乱再打包。

> [!warning] 误区：WebDataset 只能从对象存储读
> 也能从本地 NVMe、[[并行文件系统]]、HTTP 流读。`wds.WebDataset(path)` path 可是本地 tar、`pipe:` 命令、`http://` URL。只是顺序流式天然契合对象存储。


## 3. 核心概念详解

### 3.1 shard（tar 分片）

一个大 tar 文件含很多样本，每个样本一组文件（key 相同）：

```
train-000000.tar
  sample-000001.jpg   sample-000001.txt   sample-000001.cls
  sample-000002.jpg   sample-000002.txt   sample-000002.cls
  ...
train-000001.tar
  ...
```

shard 是**顺序读单元** + **分布式分配单元**。大小经验 **64 MB–1 GB**（对齐对象存储高效读 + 内存缓冲可控）。

### 3.2 顺序读（流式）

```
worker 读 shard:
  stream = open tar (本地 / pipe / http)
  for entry in tar:          # 顺序, 不 seek
      sample = decode(entry) # 边读边解
      yield sample
```

无随机访问，对对象存储顺序 GET 极友好。

### 3.3 分布式按 shard 分配

```
shards = [000000..000099].tar
worker 0 读 [000000..000024]
worker 1 读 [000025..000049]
worker 2 读 [000050..000074]
worker 3 读 [000075..000099]
（每 worker 不同 shard, 无冲突）
```

多节点/多 worker 用 `split_by_node` / `split_by_worker` 把 shard 范围切给各 worker。每 epoch 可错开起止（reshuffle shard 顺序）增随机性。

### 3.4 shuffle 策略

- **shard 内 shuffle**：tar 顺序读，进一个**缓冲区**（如 1000 样本）乱序吐。`shuffle(1000)`。
- **跨 shard shuffle**：交错多个 shard 的流（`wds.ShardShuffle` / `resampled`），近似全局随机。
- **真随机**：先打乱样本再打包成 shard（离线预处理），训练时顺序读已随机化的 shard。

### 3.5 shard 大小选择

| shard 大小 | 优 | 劣 |
|---|---|---|
| 小（~64 MB） | 缓冲内存小、shuffle 灵活 | 元数据/shard 数多、对象存储请求多 |
| 大（~1 GB） | 顺序读带宽吃满、请求少 | 缓冲内存大、单 shard 失败重读成本高 |

经验 **64 MB–1 GB**，常见 ~256 MB–500 MB。要权衡对象存储 GET 效率（大好）、内存缓冲（小好）、容错（单 shard 坏不致命）。

### 3.6 与 Parquet / Arrow 对比

| 维度 | **WebDataset** | [[Parquet]] | [[Arrow]] |
|---|---|---|---|
| 数据 | 媒体/任意二进制 | 表格/结构化 | 表格/结构化 |
| 存储 | tar（行式打包） | 列存压缩 | 内存列存 |
| 读 | 顺序流式 | 列裁剪+跳行 | 零拷贝 |
| 场景 | 图/音/视频/多模态 | 特征/标注/元数据 | 内存计算 |
| 随机 | 弱（流式） | 强（按 row group） | 强（内存） |


## 4. 数学原理 / 公式

### 4.1 顺序 vs 随机读吞吐

设单盘/单连接顺序带宽 $BW_{\text{seq}}$（NVMe ~7 GB/s、对象存储并发 ~GB/s），每样本 $s$ 字节，随机读每样本延迟 $t_{\text{rand}}$（对象存储 ~50 ms、NVMe ~50 μs）。

顺序读 shard 吞吐：

$$\text{TP}_{\text{seq}} = \frac{BW_{\text{seq}}}{s}$$

随机读小文件吞吐（单连接）：

$$\text{TP}_{\text{rand}} \approx \frac{1}{t_{\text{rand}}}$$

对象存储 $s=10$ KB：顺序 $\frac{7\text{ GB/s}}{10\text{ KB}}\approx 7\times10^5$ 样本/s（理论，单连接实际受 API 限），随机 $\frac{1}{0.05}\approx 20$ 样本/s。差量级巨大。打包 shard 把随机变顺序是核心收益。

### 4.2 shard 数与分布式并行

总样本数 $N$，每 shard $n_s$ 样本，shard 数 $G=N/n_s$。$W$ 个 worker，每 worker 读 $G/W$ shard。要 $G \ge W$（每 worker 至少一 shard）且 $G \gg W$（负载均衡 + epoch 错开）。$N=10^9$、$n_s=10^4$ → $G=10^5$ shard，$W=1000$ → 每 worker 100 shard，充分并行。

### 4.3 缓冲 shuffle 的随机度

缓冲 $B$ 样本的乱序，近似 $B$-local 随机（样本与原序距离 $\le B$）。要全局接近 iid 需 $B\approx N$（不可行）或跨 shard 交错 $k$ 路（近似 $k$-local）。真随机靠离线打乱打包。$B$ 取 1000–10000 通常是训练够用的随机度（结合 shard 顺序随机化）。

### 4.4 带宽与 GPU 喂饱

单卡样本消耗 $\frac{\text{batch}}{T_{\text{step}}}$ 样本/s。8 卡 ×batch 64 ×step 0.1s → 5120 样本/s。$s=10$ KB → 51 MB/s，单 NVMe 够；但多模态 $s=1$ MB（图）→ 5 GB/s，需多盘/并行 FS。shard 顺序读能吃满这带宽，小文件随机读吃不满。


## 5. 代码示例（可选）

### 5.1 写 shard（打包样本）

```python
import webdataset as wds

with wds.TarWriter("train-000000.tar") as sink:
    for i, sample in enumerate(samples):
        sink.write({
            "__key__": f"sample-{i:06d}",
            "jpg": image_bytes,       # 图二进制
            "txt": text.encode(),     # 文本
            "cls": str(label).encode(),
        })
```

### 5.2 顺序流式读（对象存储友好）

```python
import webdataset as wds

# 100 个 shard, pipe: 从 s3 流式拉 (不落本地)
url = "pipe:aws s3 cp s3://my-bucket/train-{000000..000099}.tar -"
ds = (wds.WebDataset(url)
        .shuffle(1000)        # shard 内缓冲乱序
        .decode("pil")        # 自动解码 jpg->PIL
        .to_tuple("jpg", "cls"))   # 取图+标签
for img, label in ds:
    train_step(img, label)
```

### 5.3 分布式按 worker/node 切 shard

```python
import webdataset as wds

# 多 node 多 worker: 每 worker 读不同 shard 子集
ds = (wds.WebDataset("train-{000000..000999}.tar",
                     nodesplitter=wds.split_by_node,   # 按 node 切
                     shardshuffle=True)                 # 跨 epoch shard 顺序随机
      .shuffle(1000).decode())
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[本地NVMe与对象存储]]（顺序读对象存储/本地 NVMe）、[[并行文件系统]]（shard 放共享 FS）、[[batch与mini-batch]]（batch 组装）。
- **下游（应用）**: 大规模预训练/多模态数据加载、RL rollout 数据、[[Ray Data]]（wds 可作 source）。
- **对比 / 易混**:
  - **WebDataset vs [[Parquet]]**：媒体行式 tar 流 ↔ 表格列存压缩。图/音用 wds，特征/标注用 Parquet。
  - **WebDataset vs [[Arrow]]**：磁盘 tar 流 ↔ 内存列存。wds 喂数据进内存，Arrow 做内存计算，可衔接。
  - **WebDataset vs ImageFolder**：ImageFolder 逐文件随机读，百万文件慢；wds 打包顺序读快。是 ImageFolder 的大规模替代。


## 7. 常见误区与易错点

> [!warning] 误区 1：shard 内能高效随机跳读
> 不能。tar 是顺序流，跳读要 seek 整 tar。随机靠缓冲 shuffle 或跨 shard 交错。要真随机需离线打乱打包。

> [!warning] 误区 2：shuffle(1000) = 全局随机
> 只是 1000-local 缓冲随机。要更强随机需跨 shard 交错 + shard 顺序随机化，或离线打乱。

> [!warning] 误区 3：shard 越大越好
> 大 shard 顺序带宽好但缓冲内存大、单 shard 坏损失大。64 MB–1 GB 折中，常见 ~256–500 MB。

> [!warning] 误区 4：WebDataset 只读对象存储
> 本地 NVMe、[[并行文件系统]]、HTTP 流都行。`pipe:` / `http://` / 本地路径皆可。只是顺序流天然契合对象存储。

> [!warning] 误区 5：用 wds 就不用再想数据本地性
> 跨节点读对象存储仍有网络/出口费。大规模训练把当前 epoch shard 预取到本地 NVMe 或就近的 [[并行文件系统]] 更稳。

> [!tip] 实践：shard 内样本数与缓冲
> 每 shard ~1 万–10 万样本（对应几百 MB），shuffle 缓冲 1000–10000，跨 epoch 用 `shardshuffle` 错开 shard 顺序，综合随机性够大多数训练。严格 iid 需求才离线打乱。

> [!note] 补充：与 DLai / tarstreaming
> WebDataset（tmbdev）是 PyTorch 生态主流；另有 `tarstreaming`（MLCommons）类似，支持流式+索引。两者理念同：tar shard + 顺序读。


## 8. 延伸细节

### 8.1 命名约定与扩展名

WebDataset 按**扩展名**自动解码（`.jpg`→PIL、`.txt`→str、`.cls`→int、`.json`→dict），`__key__` 是样本 key。多模态一组文件 key 相同即同一样本。

### 8.2 复合管道

`wds.WebDataset(...).shuffle().decode().map().batched().unbatched()` 函数式组合，`IterableDataset` 风格，适配大规模流式预处理（tokenize、augment）。

### 8.3 容错

单 shard 损坏只丢该 shard 样本，可跳过。`wds` 支持忽略坏 shard，比"一个坏文件训练崩"健壮。大集群训练必备容错。

### 8.4 内容来源

WebDataset 概念、shard/shuffle/decode API、`pipe:` 流式来自 tmbdev/WebDataset 官方文档与 Paddle webdataset 对照。shard 大小 64 MB–1 GB、吞吐对比为典型经验值，强依赖数据/硬件，关键数量级（顺序 vs 随机几十倍、GB/s 顺序带宽）稳定。`tarstreaming` 来自 MLCommons 文档。

---
相关: [[数据格式]] | [[Parquet]] | [[Arrow]] | [[本地NVMe与对象存储]] | [[并行文件系统]] | [[Ray Data]] | [[Spark执行模型]] | [[batch与mini-batch]] | [[FLOPs计算]]
