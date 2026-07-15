# DeviceMesh

> **所属章节**: [[DTensor与DeviceMesh]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: DeviceMesh / device mesh / mesh / nested mesh / 逻辑拓扑 / DeviceMesh HSDP / mesh 维度
> **难度**: 中高（需懂 [[torch.distributed]] + [[process group]] + [[3D parallelism]]）


## 1. 一句话定义

**DeviceMesh** 是 PyTorch 2.x 的**逻辑设备拓扑抽象**——把一组物理设备（GPU/TPU）组织成**多维逻辑网格**（mesh），如 2D mesh $[dp, tp]$ 表示"前维做 data parallel、后维做 tensor parallel"，每维对应一个**子通信域**（可做 all-reduce/all-gather/reduce-scatter）。它是 [[DTensor]] sharding 与 [[FSDP2]] per-parameter sharding 的**坐标基础**（DTensor 的 placement 沿 mesh 维声明 shard/replicate），也是 [[3D parallelism]] 多维并行的**统一表达**（一个 mesh 表达 TP+DP+PP 多维，nested mesh 表达层级）。DeviceMesh 取代了旧 `process group` 的扁平单通信域模型（一个 pg 对应一个并行组，多维并行要手拼多个 pg 且不感知拓扑层级），让分布式代码"按 mesh 维声明 sharding"而非"手动管理通信组"。

> [!note] 三句话定位
> - **是什么**：多维逻辑设备网格，每维一个子通信域，是 DTensor/FSDP2 的坐标基础。
> - **为什么**：多维并行（TP+DP+PP）用扁平 pg 手拼繁琐且不感知层级；mesh 用多维表达"哪维做什么并行"，sharding 沿 mesh 维声明。
> - **与 [[process group]] 关系**：mesh 是 pg 的多维升级——每 mesh 维对应一个子 pg，但 mesh 感知拓扑层级（nested mesh 表达 HSDP/3D）。


## 2. 为什么需要它（动机与背景）

### 2.1 多维并行的表达需求

大模型训练用多维并行：TP（tensor parallel，切权重维）+ DP（data parallel，切 batch 维）+ PP（pipeline parallel，切层维）+ EP（expert parallel，切 expert 维）。每维一个通信组（TP 组内 all-reduce、DP 组内 all-reduce、PP 组间 send/recv、EP 组内 all-to-all）。旧 `process group` 是**扁平单通信域**——每维手动建一个 pg（`tp_group`/`dp_group`/...），用户代码管理哪个 pg 做什么。多维组合时手拼易错、不感知层级（如 HSDP 的 DP 内嵌 TP）。

### 2.2 mesh 的多维表达

DeviceMesh 把设备组织成多维网格：`mesh = DeviceMesh("cuda", shape=(dp, tp))` 表示 $dp \times tp$ 个 GPU 的 2D mesh。mesh 的每维是一个**子通信域**（mesh 维切出该维的 rank 组）。DTensor 的 placement 沿 mesh 维声明（如 `shard(mesh, 0)` 沿 dp 维 shard batch 维，`shard(mesh, 1)` 沿 tp 维 shard 权重维）。用户声明 sharding，runtime 据 mesh 维自动选通信。**声明式 vs 手拼 pg**。

### 2.3 nested mesh 表达层级

HSDP（Hybrid Sharded DP）：DP 外层（跨机）+ FSDP 内层（机内）。用 nested mesh：外 mesh 的 dp 维 + 内 mesh 的 fsdp 维。3D parallelism 的 TP+DP+PP 用 3D mesh 或 nested。mesh 感知拓扑层级（NVLink 域内 vs IB 跨机），优化通信选 NVLink 内 reduce、IB 跨机 all-reduce。扁平 pg 不感知层级。

### 2.4 取代 process group 的扁平模型

旧 `process group`：`dist.init_process_group` 建一个 world pg，再 `dist.new_group(ranks)` 建子 pg。多维并行手拼多 pg，用户代码 `if tp_group: all_reduce(...)`。DeviceMesh：建 mesh，DTensor sharding 声明沿哪维，runtime 自动通信。mesh 是 pg 的**多维 + 拓扑感知**升级，是 [[DTensor]]/[[FSDP2]] 的基础设施。

### 2.5 与 NCCL 拓扑协同

DeviceMesh 可从 `torch.distributed` 的 topology（NVLink/IB/PCIe）建，让 mesh 维对齐硬件层级（如 dp 维跨 IB、tp 维在 NVLink 域内）。通信自动选最优路径（NVLink 内 all-reduce 快、IB 跨机慢）。是 [[NCCL通信拓扑]] 的逻辑化抽象。


## 3. 核心概念详解

### 3.1 mesh 的创建

```python
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# 2D mesh: dp x tp, 总 dp*tp 个 GPU
mesh = init_device_mesh("cuda", (8, 4))   # 32 GPU, mesh[0]=dp(8), mesh[1]=tp(4)
# 或显式:
mesh = DeviceMesh("cuda", torch.arange(32).reshape(8, 4))
```

mesh 的 shape 是多维网格，每维一个子通信域。

### 3.2 子通信域（mesh 维）

```python
mesh = init_device_mesh("cuda", (dp, tp))
dp_mesh = mesh["dp"]   # dp 维的子 mesh (子 pg)
tp_mesh = mesh["tp"]   # tp 维的子 mesh
# 沿 dp 维做 all-reduce (DP grad sync)
dp_mesh.all_reduce(grad)
# 沿 tp 维做 all-reduce (TP)
tp_mesh.all_reduce(grad)
```

每 mesh 维对应一个子 pg，可做集合通信。命名 mesh 维（`mesh["dp"]`）让代码可读。

### 3.3 DTensor 的 sharding 坐标

```python
from torch.distributed.tensor import DTensor, Shard, Replicate
# weight 沿 tp 维 shard col
weight_dt = distribute_tensor(weight, tp_mesh, [Shard(1)])   # shard 第 1 维沿 tp_mesh
# batch 沿 dp 维 shard
x_dt = distribute_tensor(x, dp_mesh, [Shard(0)])   # shard 第 0 维 (batch) 沿 dp_mesh
```

DTensor 的 placement（Shard/Replicate/Partial）沿 mesh 维声明。mesh 是 sharding 的坐标。详见 [[DTensor]]。

### 3.4 nested mesh（HSDP）

```python
# HSDP: 外 dp (跨机) + 内 fsdp (机内 8 卡)
mesh = init_device_mesh("cuda", (dp_replicate, dp_shard), mesh_dim_names=("dp_replicate", "dp_shard"))
# 外维 dp_replicate: replicate 跨机
# 内维 dp_shard: shard 机内 (FSDP2)
# DTensor: weight 的 placement = [Replicate(dp_replicate), Shard(dp_shard)]
weight_dt = distribute_tensor(weight, mesh, [Replicate("dp_replicate"), Shard("dp_shard")])
# FSDP2 沿 dp_shard 维 shard/unshard
```

nested mesh 表达层级并行（外层 replicate、内层 shard），是 HSDP 的基础。

### 3.5 3D parallelism 的 mesh

```python
# TP + DP + PP, 3D mesh
mesh = init_device_mesh("cuda", (pp, dp, tp), mesh_dim_names=("pp","dp","tp"))
# pp 维: pipeline stage
# dp 维: data parallel
# tp 维: tensor parallel
# DTensor sharding 沿 dp/tp 维, PP 沿 pp 维切层
```

一个 3D mesh 表达三维并行，比手拼 3 个 pg 清晰。详见 [[3D parallelism]]。

### 3.6 拓扑感知建 mesh

```python
# 从 hardware topology 建 mesh, 让维对齐 NVLink/IB 层级
# torch.distributed 的 DeviceMesh 可读 NCCL topology
# 让 tp 维在 NVLink 域内 (快), dp 维跨 IB (慢但必要)
# 通信自动走最优路径
```

mesh 的建可考虑硬件拓扑，让高带宽通信（NVLink）对齐需高频通信的维（如 tp 的 all-reduce）。

### 3.7 mesh 与 process group 的关系

mesh 内部用 `process group`（每 mesh 维建一个子 pg），但 mesh 在其上加**多维 + 命名 + 拓扑感知**。旧代码用 pg 仍可（兼容），新代码（DTensor/FSDP2）用 mesh。两者并存，mesh 是 2.x 的推荐。

### 3.8 与 FSDP2 的关系

[[FSDP2]] 用 DeviceMesh 声明 sharding mesh（沿哪维 shard weight）。FSDP2 的 per-parameter sharding 是 `distribute_tensor(weight, mesh, [Shard(dim)])`，runtime 沿 mesh 维 all-gather/reduce-scatter。mesh 是 FSDP2 的坐标基础。详见 [[FSDP2]]。

### 3.9 与 Megatron-LM 的关系

Megatron-LM 的 TP/DP/PP 用手拼 pg + `ParallelState`（`tp_rank`/`dp_rank`/...）。DeviceMesh 是 PyTorch 原生的等价抽象（多维 mesh + 命名维）。两者理念同，DeviceMesh 是 PyTorch 主线（与 DTensor/FSDP2 集成）。详见 [[Megatron-LM]]。

### 3.10 单机模拟（无分布式）

```python
# 单机多 GPU 或 CPU 模拟, mesh 仍可建 (用于测试/开发)
mesh = DeviceMesh("cpu", torch.arange(4).reshape(2, 2))   # 2x2 CPU mesh
# DTensor 在单机 mesh 上仍可 distribute (local shard)
```


## 4. 数学原理 / 公式

### 4.1 mesh 的多维网格

设 mesh shape $(d_0, d_1, ..., d_k)$，总设备 $\prod_i d_i$。每维 $i$ 的子通信域含 $d_i$ 个 rank（mesh 沿维 $i$ 切的切片）。

### 4.2 rank 到 mesh 坐标的映射

global rank $r$ 映射到 mesh 坐标 $(c_0, ..., c_k)$（row-major）：

$$
c_i = \left\lfloor \frac{r \mod \prod_{j \le i} d_j}{\prod_{j < i} d_j} \right\rfloor
$$

每维 $i$ 的子 pg：固定其余坐标、变 $c_i$ 的 rank 组。

### 4.3 通信量与 mesh 维

DTensor `Shard(dim)` 沿 mesh 维 $i$：all-gather 通信量 $\propto$ tensor size（unshard），reduce-scatter $\propto$ grad size（sync）。Replicate 沿 mesh 维：无通信（各 replica）。Partial 沿 mesh 维：all-reduce（DP grad sync）。mesh 维的选择决定哪维承担哪通信。详见 [[DTensor]] 4 节。

### 4.4 nested mesh 的层级

nested mesh 外维 $d_{\text{outer}}$ + 内维 $d_{\text{inner}}$。外维的子 pg 跨 $d_{\text{outer}}$ 个内 mesh 组（每个内 mesh 是 $d_{\text{inner}}$ 设备）。Replicate 沿外维（每内 mesh 一份 replica）、Shard 沿内维（内 mesh 内 shard）。HSDP 的 $F = \text{replicate outer} + \text{shard inner}$。


## 5. 代码示例（可选）

### 5.1 2D mesh 与 DTensor

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor

mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))  # 8 GPU
dp_mesh, tp_mesh = mesh["dp"], mesh["tp"]

weight = torch.randn(768, 768)
# shard weight col 沿 tp 维
w_dt = distribute_tensor(weight, tp_mesh, [Shard(1)])
x = torch.randn(32, 768)
# shard batch 沿 dp 维
x_dt = distribute_tensor(x, dp_mesh, [Shard(0)])
```

### 5.2 nested mesh (HSDP)

```python
# 32 GPU: 4 机 x 8 卡. 外 dp_replicate(跨机 4), 内 dp_shard(机内 8)
mesh = init_device_mesh("cuda", (4, 8), mesh_dim_names=("dp_replicate","dp_shard"))
weight = torch.randn(768, 768)
# HSDP placement: replicate 跨机, shard 机内
w_dt = distribute_tensor(weight, mesh, [Replicate("dp_replicate"), Shard("dp_shard")])
# FSDP2 用 w_dt, 沿 dp_shard 维 all-gather (前向) / reduce-scatter (反向)
```

### 5.3 子通信域使用

```python
mesh = init_device_mesh("cuda", (dp, tp))
# 沿 tp 维 all-reduce (TP 的 col parallel 后 sync)
tp_pg = mesh.get_group("tp")
dist.all_reduce(grad, group=tp_pg)
# 或用 mesh api
mesh["tp"].all_reduce(grad)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[torch.distributed]]（mesh 基于其 pg/NCCL）、[[process group]]（mesh 维对应子 pg）、硬件拓扑（NVLink/IB，mesh 可感知）。
- **下游（应用）**: [[DTensor]]（sharding 沿 mesh 维声明）、[[FSDP2]]（per-param sharding 的 mesh）、[[3D parallelism]]（多维 mesh 表达 TP+DP+PP）、HSDP（nested mesh）、[[distributed checkpoint]]（DTensor metadata 含 mesh）、[[Megatron-LM]]（ParallelState 的 PyTorch 原生对应）。
- **对比 / 易混**:
  - **DeviceMesh vs [[process group]]**：mesh 是 pg 的多维 + 命名 + 拓扑感知升级；pg 是扁平单通信域。mesh 内部用 pg，但加多维抽象。新代码用 mesh。
  - **DeviceMesh vs [[Megatron-LM]] ParallelState**：两者理念同（多维并行拓扑），DeviceMesh 是 PyTorch 原生（与 DTensor/FSDP2 集成），ParallelState 是 Megatron 自有。趋同。
  - **mesh 维 vs DTensor placement 维**：mesh 维是设备拓扑维（dp/tp）；DTensor placement 维是 tensor 的逻辑维（如 batch 维/hidden 维）。Shard(tensor_dim) 沿 mesh_dim 声明"tensor 的哪维沿设备的哪 mesh 维 shard"。


## 7. 常见误区与易错点

> [!warning] 误区 1：mesh 维与 tensor 维混淆
> mesh 维是设备拓扑维（dp/tp）；tensor 维是逻辑维（batch/hidden）。`Shard(tensor_dim)` 沿 `mesh_dim` 声明。`Shard(0)` 是 shard tensor 第 0 维，沿哪个 mesh 维由 distribute_tensor 的 mesh 参数定。

> [!warning] 误区 2：忽视 nested mesh 的层级
> HSDP 用 nested mesh（外 replicate + 内 shard），不是单维 shard。单维 shard 是 FSDP（全 shard），nested 才是 HSDP（部分 replicate 部分 shard）。

> [!warning] 误区 3：mesh 维不感知硬件拓扑
> mesh 维可对齐硬件层级（tp 维在 NVLink 域内、dp 维跨 IB）。不感知拓扑乱建会让高频通信（tp all-reduce）走慢的 IB。建 mesh 考虑 topology。

> [!warning] 误区 4：以为 mesh 取代 pg
> mesh 内部用 pg（每 mesh 维一子 pg），是 pg 的多维抽象层，非取代。旧 pg 代码仍兼容。新 DTensor/FSDP2 用 mesh。

> [!warning] 误区 5：mesh 命名维可选
> 不命名维用 `mesh[0]`/`mesh[1]` 索引，易混。命名（`mesh["dp"]`）可读。生产代码命名。

> [!warning] 误区 6：单机不能用 mesh
> 单机多 GPU 或 CPU 可建 mesh（用于测试/开发，DTensor 仍 distribute local shard）。不需真分布式也能用 mesh API。

> [!warning] 误区 7：mesh 与 FSDP2 的 mesh 必须相同
> FSDP2 的 sharding mesh 是 mesh 的某维（如 dp_shard），但模型的不同参数可用不同 placement（有的 shard、有的 replicate）。mesh 是拓扑，placement 是 per-param 声明。


## 8. 延伸细节

### 8.1 DeviceMesh 的 SPMD 语义

DeviceMesh + DTensor 让分布式代码写成 SPMD（单程序多数据）：每 rank 跑同一代码，DTensor 的 placement 自动处理 sharding/通信。用户写 `y = x @ w`（DTensor op），runtime 据 placement 自动 all-gather/reduce-scatter。是 PyTorch 原生 SPMD 的基础（vs Megatron 手写 `RowParallelLinear` 等）。

### 8.2 mesh 的 resharding

不同 op 需不同 placement（如 attention 需 replicate KV、MLP 可 shard）。DTensor 的 resharding（`redistribute`）沿 mesh 维做 all-gather/reduce-scatter 转换 placement。是 [[distributed checkpoint]] 跨拓扑 resharding 的基础。

### 8.3 与 torch.compile 的关系

DeviceMesh/DTensor 可被 [[TorchDynamo]] trace（DTensor op 是 registered op）。FSDP2 的 mesh-aware op 可 compile。是 compile 与分布式结合的接口。但 NCCL 通信 op 在 graph 里是特殊 node（不融合）。

### 8.4 mesh 的 hardware-aware 建

`init_device_mesh` 可传 hardware topology（NVLink 域、IB 拓扑），让 mesh 维对齐。实验性 `DeviceMesh.from_nvidia_topology` 等。是拓扑感知建 mesh 的方向。

### 8.5 内容来源

DeviceMesh 整理自 PyTorch dev docs "DeviceMesh"、`torch/distributed/device_mesh/` 源码、RFC "PyTorch DTensor + DeviceMesh"。nested mesh/HSDP 见 FSDP2 design doc。

---
相关: [[DTensor与DeviceMesh]] | [[DTensor]] | [[FSDP2]] | [[3D parallelism]] | [[torch.distributed]] | [[process group]] | [[NCCL通信拓扑]] | [[Megatron-LM]] | [[distributed checkpoint]] | [[Fully Sharded Data Parallel]] | [[AllGather]] | [[reduce-scatter]] | [[all-reduce]]
