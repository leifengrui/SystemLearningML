# Data Parallel

> **所属章节**: [[并行训练]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 基础（已被 DDP 取代，作为对比理解）

## 1. 一句话定义

**`nn.DataParallel`（DP）** 是 PyTorch 早期的**单进程多线程数据并行**方案：在一个进程里开多个线程，把模型**复制**到各 GPU，把一个 batch **按维度 0 切分**到各卡上前向、反向时把各卡梯度**归约回主卡（cuda:0）** 再更新。它因**主卡瓶颈、GIL 与线程开销、不支持跨机、被 DDP 全面取代**而基本退出主流，仅在调试或极小规模偶有使用。

> [!note] DP 已弃用
> PyTorch 官方文档明确推荐用 [[Distributed Data Parallel]]（DDP）替代 DP。DP 仅作为"最简单的多卡一行代码"保留：`model = nn.DataParallel(model)`。但生产/LLM 训练绝不要用 DP。学 DP 的价值是**对比理解 DDP 为什么更好**。

## 2. 为什么需要它（动机与背景）

最朴素的多卡想法：一个 batch 切成 N 份分给 N 卡各算各的梯度，再把梯度合起来更新一份参数。DP 就是这个想法的最直接实现——单进程、多线程、主卡聚合。它一行代码就能跑，早期（2017 前）很流行。

但问题随之暴露：

- **主卡瓶颈**：梯度都归约到 cuda:0，cuda:0 显存/通信先打满，其他卡闲置，扩展性差（4 卡以上效率急剧下降）。
- **线程 + GIL**：Python GIL 使多线程前向无法真正并行 CPU 部分，且线程同步开销大。
- **模型复制开销**：每次前向都要 scatter 模型副本，慢。
- **不支持跨机**：单进程不能跨节点。
- **与 BN 不兼容**：DP 下各卡只看到自己 batch 的 BN 统计，不是 sync BN，行为与单卡不同。

于是有了 [[DDP]]：多进程、all-reduce 梯度（无主卡瓶颈）、支持跨机、与 sync BN 兼容。DP 退居"原型调试"角色。

## 3. 核心概念详解

### 3.1 工作流（DP）

```
batch (N, ...) ──scatter(dim=0)──> [GPU0: (N/8,...), GPU1: ..., ..., GPU7: ...]
                                          │ 各卡前向（模型副本）
                                          ▼
                                  各卡 loss
                                          │ 各卡反向
                                          ▼
                          梯度 gather/reduce 回 GPU0
                                          ▼
                              GPU0 用合梯度更新参数
                                          ▼
                          新参数 broadcast 到各卡（下一前向 scatter 时带）
```

关键：**梯度只在 GPU0 聚合更新**，GPU0 是 master。

### 3.2 API

```python
model = MyModel().cuda()
model = nn.DataParallel(model, device_ids=[0,1,2,3], output_device=0)
out = model(x)   # 内部自动 scatter/broadcast
```

- `device_ids`：参与的 GPU 列表。
- `output_device`：结果汇总到哪卡，默认 `device_ids[0]`。
- 模型需先 `.cuda()` 或在 DataParallel 内搬。

### 3.3 与 DDP 的核心差异

| 维度 | DP（nn.DataParallel） | DDP（DistributedDataParallel） |
|---|---|---|
| 进程模型 | 单进程多线程 | **多进程**（一进程一 GPU） |
| 梯度归约 | gather 到主卡 | **all-reduce**（无主卡） |
| 通信后端 | 朴素（CUDA 流） | **NCCL**（ring all-reduce） |
| 跨机 | ❌ 不支持 | ✅ 支持 |
| 扩展性 | 4 卡后效率急降 | 线性扩展到数百卡 |
| BN 行为 | 各卡独立统计（异步） | 可配 `SyncBatchNorm` |
| 启动 | 一行代码 | `torchrun` + `init_process_group` |
| GIL | 受限 | 无（多进程） |
| 状态 | 弃用 | 主流 |

### 3.4 DP 的显存不对称

因为梯度只在 GPU0 聚合、且 output_device=0 的 loss 计算在 GPU0，GPU0 显存明显大于其他卡 → 8 卡时 GPU0 先 OOM，其他卡还空。这是 DP 扩展性差的直接表现。

## 4. 数学原理 / 公式

数据并行的等价性：设 batch $B$ 切成 $N$ 份 $B_0,\dots,B_{N-1}$，各卡算 $\ell_i=\frac{1}{|B_i|}\sum_{j\in B_i}\ell(f_\theta(x_j),y_j)$。梯度聚合（DP 在 GPU0、DDP 用 all-reduce）后：

$$
\nabla_\theta\mathcal{L} = \frac{1}{N}\sum_{i=0}^{N-1}\nabla_\theta \ell_i
$$

即平均梯度，等价于用整个 batch $B$ 算一次梯度（shuffle 不影响期望）。**数据并行的数学正确性**由此保证：DP 与 DDP 在数学上等价，差异只在**实现效率**。

DP 的瓶颈可形式化：GPU0 要接收 $N-1$ 份梯度并求和，通信量 $(N-1)M$，是瓶颈；DDP 的 ring all-reduce 每卡 $2M$，无瓶颈。

## 5. 代码示例

```python
import torch, torch.nn as nn

# DP 用法（仅演示，生产请用 DDP）
model = nn.Linear(1000, 10).cuda()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0,1,2,3])

x = torch.randn(64, 1000).cuda()
out = model(x)
print(out.shape)   # (64, 10)，结果在 cuda:0

# 取回单卡模型（保存时）
real_model = model.module   # DataParallel 包了一层，.module 取原模型
torch.save(real_model.state_dict(), "dp.pt")
```

> [!warning] DP 下保存要 `.module`
> `nn.DataParallel` 把原模型包在 `.module` 属性下。直接 `torch.save(model.state_dict())` 会让键带 `module.` 前缀，加载到非 DP 模型报错。务必 `model.module.state_dict()`。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[nn.Module]]、[[forward与backward]]、CUDA 多卡。
- **下游（应用）**: 历史用途；现已被 [[Distributed Data Parallel]] 取代。
- **对比 / 易混**:
  - **DP vs DDP**：见 §3.3 表，这是本节核心。
  - **Data Parallel（概念）vs `nn.DataParallel`（DP 类）vs DDP**：前者是"数据并行"这一并行范式的统称；`nn.DataParallel` 是其旧实现；DDP 是现代实现。注意 [[3D parallelism]] 里的 "DP" 指**数据并行范式**，不特指 `nn.DataParallel`。
  - **DP vs FSDP**：FSDP 还把参数分片省显存，DP/DDP 都复制全量参数。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **生产用 DP** → 4 卡以上效率差、不支持跨机；用 DDP。
> 2. **DP 下保存忘 `.module`** → 键带 `module.` 前缀，加载到普通模型 mismatch。
> 3. **以为 DP 自动是 sync BN** → 不是，各卡 BN 独立统计，与单卡行为不同；要 sync BN 用 DDP + `SyncBatchNorm`。
> 4. **output_device 不在 device_ids 里** → 报错或低效。
> 5. **batch 不能被 8 整除** → DP 按 dim0 切分，不整除时某卡少一份，行为还正确但负载不均。
> 6. **以为 DP 的"多线程"=多卡并行** → GIL 下 CPU 段并不真并行，前向的 Python 调度串行。
> 7. **DP 与 DataLoader 的 `num_workers`** → DP 是 GPU 并行，DataLoader worker 是 CPU 数据加载并行，二者正交，可同时用。

## 8. 延伸细节

### 8.1 为什么 DP 仍没被删

历史包袱：很多老代码、教程、Colab 单机多卡 demo 用 DP 一行跑通。删了会破坏兼容。但新代码一律 DDP。

### 8.2 单机多卡的"最简"选择

单机多卡且不想学 `torchrun`，DP 一行最快——但性能比 DDP 差 20~50%。值得多花 10 行写 DDP。

### 8.3 DP 的思想遗产

"切 batch 到各卡 + 聚合梯度"这个范式被 DDP/FSDP 继承，只是聚合方式从"主卡 gather"换成"all-reduce"。理解 DP 的等价性，就理解了数据并行为何在数学上正确。

### 8.4 与 [[batch与mini-batch]] 的关系

DP/DDP 把 effective batch 从单卡的 micro_batch 放大到 `micro_batch × world_size`（再 × accum）。这是大 batch 训练的基础，也带来 [[学习率调度]] 的 linear scaling rule。

---
相关: [[并行训练]]、[[Distributed Data Parallel]]、[[Fully Sharded Data Parallel]]、[[batch与mini-batch]]、[[nn.Module]]、[[3D parallelism]]
