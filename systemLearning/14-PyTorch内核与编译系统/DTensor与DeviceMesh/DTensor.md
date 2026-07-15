# DTensor

> **所属章节**: [[DTensor与DeviceMesh]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: DTensor / Distributed Tensor / 分布式张量 / placement / Shard / Replicate / Partial / local tensor / global tensor / SPMD
> **难度**: 高（需懂 [[DeviceMesh]] + [[Fully Sharded Data Parallel]] + [[3D parallelism]]）


## 1. 一句话定义

**DTensor（Distributed Tensor）** 是 PyTorch 2.x 的**分布式张量抽象**——一个 DTensor 代表**逻辑上的全局张量**（global tensor，完整 shape），但物理上沿 [[DeviceMesh]] 的某些维 **shard**（切分）或 **replicate**（复制）到各 rank 的 local tensor。每个 tensor 维的分布用 **placement** 描述：`Shard(dim)`（该维沿某 mesh 维切分，每 rank 持一片）、`Replicate()`（该维全 replicate，每 rank 持完整副本）、`Partial()`（该维是 partial sum，需 all-reduce 才完整，典型用于 DP 梯度）。DTensor op（如 `x @ w`）是 **SPMD**（单程序多数据）——用户写全局语义的 `x @ w`，runtime 据 placement 自动插入 all-gather/reduce-scatter/all-reduce 让结果 placement 正确。它是 [[FSDP2]] per-parameter sharding、[[3D parallelism]] 统一表达、[[distributed checkpoint]] 跨拓扑 resharding 的基础抽象，取代了手写 `RowParallelLinear`/`ColumnParallelLinear` 的 Megatron 风格。

> [!note] 三句话定位
> - **是什么**：全局张量 + 沿 DeviceMesh 的 placement（Shard/Replicate/Partial），op 自动插通信保 placement 正确。
> - **为什么**：手写 `RowParallelLinear` 等每并行模式一 kernel 繁琐；DTensor 用 placement 声明 + SPMD op 让"切分策略"与"计算代码"解耦。
> - **与 [[FSDP2]] 关系**：FSDP2 是 DTensor 的 placement = `Shard(weight_dim)` 沿 dp mesh 的应用；DTensor 是更通用的抽象（支持任意维任意 mesh 的 shard/replicate/partial）。


## 2. 为什么需要它（动机与背景）

### 2.1 手写并行模式的痛点

Megatron-LM 风格：每种并行模式手写 module（`ColumnParallelLinear`/`RowParallelLinear`/`...ParallelEmbedding`），内部手动 all-reduce/all-gather/reduce-scatter。要改并行策略（如 TP→FSDP）需重写 module。并行策略与计算代码**强耦合**，难实验、难组合多维。

### 2.2 DTensor 的声明式

DTensor 把"切分策略"抽成 placement 声明（`Shard(1)` 沿 tp mesh shard 第 1 维），计算代码写全局语义 `y = x @ w`（DTensor op）。runtime 据 x/w 的 placement 自动插通信让 y 的 placement 正确。改策略只改 placement 声明，不动计算代码。**策略与计算解耦**。

### 2.3 SPMD 的简洁

每 rank 跑同一代码（SPMD），DTensor op 内部自动通信。用户不需写 `if tp_rank: all_reduce(...)`，写 `y = x @ w` 即可。runtime 据 placement 决定通信。是 PyTorch 原生 SPMD 的核心。

### 2.4 统一多维并行

DTensor + [[DeviceMesh]] 表达 TP（Shard weight 维沿 tp mesh）+ DP（Shard batch 维沿 dp mesh）+ FSDP（Shard weight 维沿 fsdp mesh）+ EP（Shard expert 维）的任意组合。一个 placement 列表表达多维 sharding。取代手拼多 module。

### 2.5 resharding 的灵活

不同 op 需不同 placement（attention 需 replicate KV、MLP 可 shard）。DTensor 的 `redistribute` 自动沿 mesh 维插通信转换 placement。是 [[distributed checkpoint]] 跨拓扑（不同 mesh）加载 resharding 的基础。

### 2.6 取代手写 module

Megatron 的 `ColumnParallelLinear` 等手写 module 仍可用（性能极致），但 DTensor 是 PyTorch 原生主线（与 FSDP2/torch.compile 集成）。新代码用 DTensor，极致性能场景仍可手写（与 DTensor 互补）。


## 3. 核心概念详解

### 3.1 三种 placement

| placement | 含义 | 典型用 |
|---|---|---|
| **`Shard(dim)`** | tensor 第 dim 维沿某 mesh 维切分，每 rank 持一片 | TP shard weight、DP shard batch、FSDP shard weight |
| **`Replicate()`** | 该维全 replicate，每 rank 持完整副本 | KV cache replicate、bias replicate |
| **`Partial()`** | 该维是 partial sum，每 rank 持部分和，需 all-reduce 才完整 | DP 梯度（各 rank 算部分 grad，all-reduce 合） |

每 tensor 维一个 placement（或全 Replicate 默认）。placement 列表沿 mesh 维声明（见 [[DeviceMesh]] 3.3）。

### 3.2 global vs local tensor

```python
# global: 逻辑完整 shape (如 [32, 768])
# local: 每 rank 持的物理片 (如 shard batch 沿 dp, 每 rank [32/dp, 768])
# DTensor 封装: _local_tensor (各 rank 的片) + _placement (声明)
x_dt = DTensor.from_local(x_local, mesh, [Shard(0)])   # local -> DTensor
x_local = x_dt.to_local()                              # DTensor -> local (取本 rank 片)
```

DTensor 的 `.shape` 是 global（逻辑），`.to_local()` 取本 rank 的物理片。

### 3.3 SPMD op 的自动通信

```python
# x: DTensor, placement [Shard(0)] 沿 dp (batch shard)
# w: DTensor, placement [Replicate()] (全 replicate)
y = x @ w   # DTensor matmul op
# runtime: 各 rank 本地 x_local @ w_local (w 是完整 replica)
# 结果 y: [Shard(0)] 沿 dp (batch 仍 shard)
# 无需通信 (batch shard 的 matmul 与 replicate weight 不需 sync)
```

vs DP 梯度场景：

```python
# 反向: grad_y: [Shard(0)] 沿 dp
# grad_w = x.T @ grad_y: 各 rank 算部分和 -> [Partial()] 沿 dp
# 需 all-reduce 沿 dp 维得完整 grad_w
# DTensor op 自动插 all-reduce (因 grad_w placement 是 Partial)
```

用户写全局语义，runtime 据 placement 自动通信。

### 3.4 placement 推导规则

每 op 有 placement 推导规则（propagate）：输入 placement → 输出 placement + 需通信。如：
- `x @ w`：x `[Shard(0)]` + w `[Replicate()]` → y `[Shard(0)]`，无通信。
- `x @ w`：x `[Replicate()]` + w `[Shard(1)]` → y `[Replicate()]`，无通信（TP row parallel）。
- `x @ w`：x `[Shard(0)]` + w `[Shard(1)]` → y `[Partial()]`，需 all-reduce（TP 后）。
- `x @ w`：x `[Shard(1)]` + w `[Shard(0)]` → 需 all-gather x 先（resharding）。

这些规则在 DTensor op 注册（[[Custom C++ CUDA Operator]] 的 TORCH_LIBRARY 风格）。runtime 查规则插通信。

### 3.5 resharding

```python
# x: [Shard(0)] 沿 dp, 某 op 需 [Replicate()]
x_repl = x_dt.redistribute(mesh, [Replicate()])
# runtime: all-gather 沿 dp 维 (各 rank 的 batch 片合)
# 或 [Shard(0)] -> [Shard(1)]: 需 all-to-all 重排
```

不同 op 的 placement 需求不同，redistribute 自动转换。是 FSDP2 前向 all-gather（shard→replicate）、反向 reduce-scatter（replicate→shard）的基础。

### 3.6 与 FSDP2 的关系

FSDP2 的 per-parameter sharding：weight 的 placement = `[Shard(weight_dim)]` 沿 fsdp mesh。前向：`redistribute` 到 `[Replicate()]`（all-gather 合片，用完整 weight 算）。反向：grad 是 `[Partial()]`（各 rank 算部分），all-reduce 后 `redistribute` 回 `[Shard()]`（reduce-scatter 切片存回）。DTensor 的 placement + redistribute 是 FSDP2 的机制。详见 [[FSDP2]]。

### 3.7 与 TP 的关系

Megatron TP 的 `ColumnParallelLinear`（w shard col，`[Shard(1)]` 沿 tp mesh）+ `RowParallelLinear`（w shard row，`[Shard(0)]`，后 all-reduce）。DTensor 表达：w `[Shard(1)]` 沿 tp，x `[Replicate()]` → y `[Replicate()]`（col parallel，无通信）；w `[Shard(0)]`，x `[Shard(1)]` → y `[Partial()]`（row parallel，all-reduce）。DTensor 统一表达，不必手写两 module。

### 3.8 与 DP 的关系

DDP 的 batch shard（x `[Shard(0)]` 沿 dp）+ grad all-reduce（grad `[Partial()]` → all-reduce）。DTensor 表达：x `[Shard(0)]`，w `[Replicate()]` → y `[Shard(0)]`；反向 grad_w `[Partial()]` → all-reduce 沿 dp。DTensor op 自动插 grad all-reduce（vs DDP 的 hook）。

### 3.9 与 torch.compile 的关系

DTensor op 是 registered op（[[Custom C++ CUDA Operator]]），被 [[TorchDynamo]] trace、[[Inductor]] codegen。FSDP2 的 DTensor-aware op 可 compile。但 NCCL 通信（all-gather/reduce-scatter/all-reduce）在 graph 里是特殊 node，Inductor 不融合（单独执行）。是 compile 与分布式结合的接口。

### 3.10 与 distributed checkpoint 的关系

DTensor 的 placement + mesh 是 checkpoint 的 metadata。保存时存 local tensor + placement + mesh。加载时若 mesh 不同（如换拓扑），DTensor 的 redistribute 自动 resharding。是 [[distributed checkpoint]] 跨拓扑的基础。详见 [[distributed checkpoint]]。


## 4. 数学原理 / 公式

### 4.1 Shard 的切分

`Shard(dim)` 沿 mesh 维 $m$（$d_m$ 个 rank）：tensor 第 dim 维大小 $D$ 切成 $d_m$ 片，每 rank $r$ 持片 $[r \cdot D/d_m, (r+1) \cdot D/d_m)$。local shape 第 dim 维 $= D/d_m$。

### 4.2 Replicate 的副本

`Replicate()` 沿 mesh 维 $m$：每 rank 持完整副本（local = global）。无通信（各 rank 独立）。

### 4.3 Partial 的求和

`Partial()` 沿 mesh 维 $m$：每 rank 持部分和 $P_r$，完整 = $\sum_r P_r$（all-reduce）。典型 DP 梯度（各 rank 算不同 batch 的梯度部分和）。

### 4.4 matmul 的 placement 推导

设 x placement $p_x$、w placement $p_w$，沿 mesh 维 $m$：

| $p_x$ | $p_w$ | $y = x@w$ placement | 通信 |
|---|---|---|---|
| Shard(0) | Replicate | Shard(0) | 无 |
| Replicate | Shard(1) | Replicate | 无 |
| Shard(0) | Shard(1) | Partial | all-reduce |
| Shard(1) | Shard(0) | — | all-gather x 先 (resharding) |

规则：shard 在 contracting 维（被 reduction 的维）→ Partial 需 all-reduce；shard 在 non-contracting 维 → 输出也 shard，无通信。

### 4.5 resharding 的通信量

`Shard(dim)` → `Replicate()`：all-gather，通信 $\propto$ tensor size（合片）。

`Partial()` → `Replicate()`：all-reduce，通信 $\propto$ tensor size。

`Replicate()` → `Shard(dim)`：reduce-scatter，通信 $\propto$ tensor size（切片）。

`Shard(0)` → `Shard(1)`：all-to-all，通信 $\propto$ tensor size（重排）。

### 4.6 SPMD 的本地 FLOP

每 rank 算 local tensor 的 FLOP。Shard(0) batch + Replicate weight：每 rank 算 $B/d_m \times D$ 的 matmul（batch 切 $d_m$ 份）。总 FLOP $\sum_r = B \times D$（= 全局，因 batch 切分算力也切分）。Partial weight + Replicate batch：每 rank 算 $B \times D/d_m$，总 $B \times D$。SPMD 的算力切分与数据切分对齐。


## 5. 代码示例（可选）

### 5.1 DTensor 基本用法

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate, Partial, distribute_tensor

mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
x = torch.randn(32, 768)
w = torch.randn(768, 768)
# x: shard batch 沿 dp
x_dt = distribute_tensor(x, mesh["dp"], [Shard(0)])
# w: shard col 沿 tp
w_dt = distribute_tensor(w, mesh["tp"], [Shard(1)])
# DTensor op (SPMD)
y_dt = x_dt @ w_dt   # runtime 据 placement 自动通信
# y 的 placement 自动推导 (Shard(0) batch + Replicate tp)
y = y_dt.to_local()  # 取本 rank 物理片
```

### 5.2 resharding

```python
# x_dt: [Shard(0)] 沿 dp, 某 op 需 [Replicate]
x_repl = x_dt.redistribute(mesh["dp"], [Replicate()])
# runtime: all-gather 沿 dp 维合片
# 或 Partial -> Replicate (DP grad all-reduce)
grad_dt = ...   # [Partial] 沿 dp
grad_repl = grad_dt.redistribute(mesh["dp"], [Replicate()])   # all-reduce
```

### 5.3 FSDP2 的 placement

```python
# FSDP2: weight [Shard(weight_dim)] 沿 fsdp mesh
mesh = init_device_mesh("cuda", (fsdp,), mesh_dim_names=("fsdp",))
w = torch.randn(768, 768)
w_dt = distribute_tensor(w, mesh, [Shard(0)])   # shard row 沿 fsdp
# 前向: redistribute to [Replicate] (all-gather 合片, 用完整 w 算)
w_full = w_dt.redistribute(mesh, [Replicate()])
y = x @ w_full.to_local()
# 反向: grad [Partial] -> all-reduce -> [Replicate] -> reduce-scatter -> [Shard] (存回)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[DeviceMesh]]（sharding 的坐标基础）、[[AllGather]]/[[reduce-scatter]]/[[all-reduce]]（resharding 的通信原语）、[[Tensor]]（DTensor 是其分布式版）、[[ATen与c10]]（DTensor op 注册）。
- **下游（应用）**: [[FSDP2]]（placement = Shard weight 的应用）、[[3D parallelism]]（多维 placement）、[[distributed checkpoint]]（placement + mesh 作 metadata）、[[torch.compile]]/[[TorchDynamo]]（DTensor op 可 trace）、[[Megatron-LM]]（手写 module 的 DTensor 等价）、HSDP（nested placement）。
- **对比 / 易混**:
  - **DTensor vs [[Tensor]]**：Tensor 是本地张量（单 device）；DTensor 是全局逻辑张量 + 沿 mesh 的 placement（多 device）。DTensor 的 `.to_local()` 取本 rank 的 Tensor。
  - **DTensor vs Megatron `ColumnParallelLinear`**：DTensor 用 placement 声明 + SPMD op（策略与计算解耦）；Megatron 手写 module（策略硬编码）。DTensor 是声明式，Megatron 是命令式。两者可互补（DTensor 通用，Megatron 极致）。
  - **DTensor vs [[Fully Sharded Data Parallel|FSDP]]**：FSDP1 是手写 module（`FullyShardedData Parallel` wrap）；FSDP2 用 DTensor 的 placement（per-param 声明）。FSDP2 是 DTensor 的应用。详见 [[FSDP2]]。
  - **Shard vs Partial vs Replicate**：Shard 是切分（每 rank 一片）；Partial 是部分和（需 all-reduce）；Replicate 是副本（无通信）。三者正交，组合表达任意分布。


## 7. 常见误区与易错点

> [!warning] 误区 1：DTensor 是新数据类型
> DTensor 是 `Tensor` 的子类（`__torch_function__` 拦截），用现有 op 但加 placement 语义。不是完全独立类型，互转（`from_local`/`to_local`）零拷贝（local tensor 共享 storage）。

> [!warning] 误区 2：忽视 placement 推导规则
> DTensor op 的输出 placement 由输入 placement 按规则推导。若规则未注册（custom op），DTensor op 可能报错或 fallback。custom op 要 DTensor-aware 需注册 placement 规则。

> [!warning] 误区 3：混淆 Shard 维与 mesh 维
> `Shard(tensor_dim)` 沿哪个 mesh 维由 distribute_tensor 的 mesh 参数定。`Shard(0)` 是 shard tensor 第 0 维，沿 mesh 的哪维（dp/tp/fsdp）看 mesh 参数。

> [!warning] 误区 4：Partial 不需通信
> `Partial()` 是部分和，**必须 all-reduce** 才完整（如 DP 梯度）。不 all-reduce 直接用是错的（部分和 ≠ 完整梯度）。DTensor op 在输出 Partial 时自动插 all-reduce。

> [!warning] 误区 5：以为 DTensor 比 Megatron 慢
> DTensor 的通信与 Megatron 手写等价（同 all-gather/reduce-scatter/all-reduce），性能相当。DTensor 优势是声明式（易改策略），非性能差。极致场景 Megatron 手写可能微优（定制 kernel），但 DTensor 是 PyTorch 主线。

> [!warning] 误区 6：DTensor resharding 免费
> redistribute 插通信（all-gather/reduce-scatter/all-to-all），有通信开销。频繁 resharding（每 op 不同 placement）通信多。设计 placement 让相邻 op placement 兼容减 resharding。

> [!warning] 误区 7：DTensor 的 global shape 与 local 混淆
> `.shape` 是 global（逻辑完整），`.to_local().shape` 是 local（本 rank 片）。Shard(dim) 的 local 第 dim 维 = global/d_mesh。调试时别看错。


## 8. 延伸细节

### 8.1 DTensor 的 op 注册

DTensor op 用 `torch.distributed.tensor.ops` 的注册机制（类似 [[Custom C++ CUDA Operator]] 的 TORCH_LIBRARY，但加 placement 推导规则）。custom op 要 DTensor-aware 需注册规则（输入 placement → 输出 placement + 通信）。

### 8.2 DTensor 与 eager autograd

DTensor op 经 dispatcher，反向自动（[[Autograd Engine]] 跑 grad_fn，grad 也是 DTensor，placement 自动推导）。用户写 `y = x @ w`（DTensor），`y.sum().backward()` 自动算 grad，grad 的 placement 由 matmul 反向规则推导（如 grad_w 是 Partial，自动 all-reduce）。

### 8.3 HSDP 的 nested placement

HSDP：weight placement = `[Replicate(dp_replicate), Shard(dp_shard)]`（外 replicate 跨机、内 shard 机内）。DTensor 的 nested placement 沿 nested mesh（见 [[DeviceMesh]] 3.4）。是 FSDP2 + DDP 混合的基础。

### 8.4 DTensor 的 experimental 状态

DTensor 是 PyTorch 2.x 的实验性 API（`torch.distributed.tensor`），API 可能变。FSDP2 依赖 DTensor。生产用前查最新文档。详见 PyTorch DTensor RFC。

### 8.5 内容来源

DTensor 整理自 PyTorch dev docs "DTensor"、`torch/distributed/tensor/` 源码、RFC "PyTorch DTensor"。placement 推导规则见 `torch/distributed/tensor/ops/`。与 FSDP2/Megatron 的关系见 [[FSDP2]]/[[Megatron-LM]]。

---
相关: [[DTensor与DeviceMesh]] | [[DeviceMesh]] | [[FSDP2]] | [[Fully Sharded Data Parallel]] | [[3D parallelism]] | [[Megatron-LM]] | [[distributed checkpoint]] | [[torch.compile]] | [[TorchDynamo]] | [[Custom C++ CUDA Operator]] | [[AllGather]] | [[reduce-scatter]] | [[all-reduce]] | [[Tensor]] | [[ATen与c10]] | [[Autograd Engine]] | [[ZeRO (DeepSpeed)]]
