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

> [!note] 解答：GLI 是什么？（应为 GIL，Python 全局解释器锁）
>
> 你写的"GLI"应是 **GIL**（**Global Interpreter Lock，全局解释器锁**）的笔误——这是理解 DP 为什么慢、DDP 为什么非要多进程的关键概念，单独讲透。
>
> ### GIL 是什么
>
> GIL 是 **CPython**（你日常用的那个 Python 官方实现）的一个**互斥锁**：在任何时刻，**只允许一个线程执行 Python 字节码**。注意三个限定：
>
> 1. **针对的是 CPython 解释器本身**，不是 Python 语言的限制。Jython、IronPython 没 GIL；PyPy 也有 GIL；PEP 703（Python 3.13+ 可选 free-threaded）正在尝试去掉它，但截至 2026 仍是实验性、默认仍带 GIL。
> 2. **锁的是"Python 字节码执行"**——纯 Python 代码段。C 扩展里**主动释放 GIL** 的部分（如 NumPy 的矩阵运算、PyTorch 的 CUDA kernel 调用）可以多线程真并行。
> 3. **线程级**，不是进程级。多进程（每进程一个独立解释器、各自一把 GIL）不受影响。
>
> ### 为什么 DP（`nn.DataParallel`）被 GIL 拖慢
>
> DP 是**单进程多线程**：主进程开 N 个线程，每个线程绑一张 GPU 跑前向。问题在于前向里有大量 **Python 层调度**（`Module.forward` 的 Python 调用、参数遍历、hook 触发），这些是 Python 字节码——GIL 下**只能有一个线程在跑**，其他线程干等。真正并行的只有"调用 CUDA kernel"那瞬间（C 扩展释放 GIL）。于是：
>
> - 8 卡 DP 的 CPU 端调度**串行化**，前向 Python 段总耗时 ≈ 单卡 × 8 而非单卡；
> - 线程切换/GIL 争抢本身有开销；
> - 4 卡以上效率急剧下降，GIL 是主因之一。
>
> ### 为什么 DDP 不受 GIL 影响
>
> DDP 是**多进程**：每张卡一个独立 Python 进程，**各自有自己的 GIL**，互不争抢。进程间靠 NCCL 通信（C 扩展，不持 GIL）交换梯度。所以 DDP 的 Python 调度是真并行的——8 卡 8 进程同时跑前向 Python 段。这是 DDP 比 DP 快的根本原因之一（另一个是 all-reduce 无主卡瓶颈）。
>
> ### GIL 的"释放窗口"——为什么 GPU 训练仍能并行
>
> PyTorch 调用 CUDA kernel（如 `torch.mm`）时，C++ 端会**主动释放 GIL**，让其他线程能跑 Python；kernel 在 GPU 上执行时 CPU 线程可以继续。所以即便 DP 下，**GPU kernel 是真并行的**，只是 Python 调度串行。这就是为什么 DP 不是完全没用、小模型还能跑，但大模型 Python 调度占比高就崩。
>
> ### 速记
>
> | 场景 | GIL 影响 | 多线程能否真并行 |
> |---|---|---|
> | DP 纯 Python 调度段 | 受 GIL 锁，串行 | ❌ |
> | DP 的 CUDA kernel 调用段 | C 扩展释放 GIL | ✅（GPU 端） |
> | DDP 多进程 | 每进程独立 GIL，不互相锁 | ✅ |
> | DataLoader `num_workers` 多进程 | 同 DDP | ✅ |
>
> 所以铁律：**GPU 多卡训练一律多进程（DDP），别用多线程（DP）**，GIL 是其中一个核心理由。详见 [[Distributed Data Parallel]]。
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

> [!note] 解答：batch 里装的是什么？是一条条 prompt 吗？
>
> **是的，但在训练的每个阶段"一条"的形态不同**。这里把数据并行的"切 batch"和 LLM 训练的数据形态一起讲透。
>
> ### 通用定义：batch = 一组样本
>
> 一个 **batch**（批）= 若干条**样本**（sample/instance）组成的列表。数据并行就是把一个 batch 按维度 0 切成 N 份分给 N 张卡，每卡拿 `batch_size / N` 条。所以"batch 里装什么"取决于"一条样本是什么"。
>
> ### 训练阶段：一条样本 = 一条 prompt + 它的"答案"
>
> LLM 训练分几种范式，每种的"一条"内容不同：
>
> | 训练范式 | 一条样本装什么 | 例子 |
> |---|---|---|
> | **预训练（Next Token Prediction）** | 一段连续文本（可能拼接多条文档），模型预测每个 token 的下一个 | 一段 4096 token 的网页/代码文本，loss = 每个位置预测下一 token 的交叉熵平均 |
> | **SFT（监督微调）** | 一条 **prompt + 期望输出**（即 instruction + response） | `{"prompt": "把这句话翻译成英文：你好", "response": "Hello"}`，loss 只算 response 部分 |
> | **RLHF / PPO** | 一条 **prompt**（响应由 policy 模型自己生成） | `{"prompt": "写一首关于秋天的诗"}`，rollout 时模型生成 response，loss 来自 reward |
> | **DPO** | 一条 prompt + 两个 response（chosen / rejected） | `{"prompt": ..., "chosen": 好回答, "rejected": 差回答}` |
>
> 所以"是一条条 prompt 吗"——**预训练严格说不是单条 prompt（是连续文本段），SFT/RLHF/DPO 是 prompt（外加配套的 response/label）**。
>
> ### 数据并行里 batch 的张量形态
>
> 不管原始样本是什么，进模型前都被 tokenize + pad + stack 成张量。一条样本 = 一串 token id，一个 batch 就是：
>
> ```
> input_ids: (batch_size, seq_len)   # 每行一条样本的 token id 序列
> labels:     (batch_size, seq_len)   # 监督微调时每行的"正确下一个 token"
> attention_mask: (batch_size, seq_len)  # 哪些位置是真实 token（非 padding）
> ```
>
> DP/DDP 把这个 `(batch_size, seq_len, ...)` 张量按维度 0（batch 维）切成 N 份：每卡拿到 `(batch_size/N, seq_len, ...)`，即 `batch_size/N` 条样本。**每条样本本身不被切**（seq_len 维不动），切的是"样本数"。
>
> ### 一个具体例子
>
> 假设 `batch_size=8`、4 卡 DP、SFT 任务、`seq_len=128`：
>
> - 全局 batch 是 8 条样本，比如 8 个 `{"prompt":..., "response":...}`；
> - tokenize 后 `input_ids` 是 `(8, 128)`；
> - DP 切成 4 份：GPU0 拿第 0,1 条 `(2,128)`、GPU1 拿第 2,3 条 ……；
> - 每卡各自前向算这 2 条样本的 loss、反向算梯度；
> - 梯度聚合（DP 在 GPU0、DDP 用 all-reduce）后更新——等价于用 8 条样本算的梯度。
>
> ### 为什么不"切一条 prompt"
>
> 有人会想：能不能把一条 prompt 切成几段分给几张卡？**那是另一种并行**（[[Pipeline Parallel]] 或 sequence parallelism），不是数据并行。数据并行的语义是"各卡跑不同样本"，样本之间独立。切同一条样本要处理样本内依赖，复杂得多，只有显存装不下长序列时才考虑（见 [[3D parallelism]] 的 sequence parallel 维度）。
>
> ### 速记
>
> - **batch = 一组样本**，每条样本在 LLM 里 ≈ 一条 prompt（+ 配套 response/label，视训练范式）；
> - **数据并行切 batch 维（dim 0）**，每卡拿 `batch_size/N` 条完整样本；
> - effective batch = `micro_batch × N(卡) × accum`，这是大 batch 训练的基础，见 [[batch与mini-batch]] §8.4；
> - RLHF 的 rollout 阶段 batch 装的是纯 prompt（response 还没生成），训练阶段 batch 装的是 prompt+生成的 response，见 [[trajectory generation]]。
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

> [!note] 解答：all-reduce 是让每个卡上都有完整的一份吗？
>
> **是的，这正是 all-reduce 的定义。** 把它和容易混的几个集合通信原语一起讲清。
>
> ### all-reduce 的语义
>
> **all-reduce**：每个 rank 各持有一个同样 shape 的张量，操作完成后**每个 rank 都拿到"所有 rank 张量的和（或均值/积/max）"**。关键词是 **all**——结果人手一份。
>
> ```
> 操作前:  rank0=[1], rank1=[2], rank2=[3]   (各 rank 不同值)
> all-reduce(SUM):
> 操作后:  rank0=[6], rank1=[6], rank2=[6]   (每 rank 都有完整和)
> ```
>
> 在 DDP 里，各 rank 拿不同 batch 算出**不同的梯度**，all-reduce(MEAN) 后**每个 rank 都拿到"全部 rank 梯度的平均"**——这就是为什么所有 rank 用一致梯度更新、参数保持同步。见 [[Distributed Data Parallel]] §3.4。
>
> ### 和它对比的几个原语（容易混）
>
> | 原语 | 操作后每 rank 拿到什么 | 是否每 rank 都有"完整一份" |
> |---|---|---|
> | **all-reduce** | 全体 rank 张量的和/均值，**完整结果** | ✅ 每 rank 都有完整一份 |
> | **reduce** | 只 rank 0 拿到和，其他 rank 拿不到 | ❌ 只有 root 有 |
> | **broadcast** | rank 0 的数据复制给所有 rank（不求和，就是复制） | ✅ 但只是"复制 root 的值"，不是聚合 |
> | **all-gather** | 各 rank 持有不同段，拼成完整张量，每 rank 都有完整 | ✅ 但语义是"拼装"不是"求和" |
> | **reduce-scatter** | 全体求和后**按段分给各 rank**，每 rank 只拿一段 | ❌ 每 rank 只有一段 |
>
> ### 为什么 DDP 用 all-reduce 而不是 reduce
>
> 如果只用 `reduce`（只 rank 0 拿到聚合梯度），那只有 rank 0 能更新参数，其他 rank 没有梯度无法更新，参数就不同步了。DDP 要**所有 rank 都用一致梯度更新**，所以必须 all-reduce——每 rank 都拿到完整平均梯度。
>
> ### DP 为什么没用 all-reduce
>
> DP（`nn.DataParallel`）是"各卡梯度 gather 到 GPU0、GPU0 更新、再把新参数 broadcast 给各卡"——本质是 `reduce`（到 GPU0）+ `broadcast`（新参数）的两步，GPU0 是瓶颈。DDP 用 ring all-reduce 一步搞定且无主卡瓶颈，通信量 $\approx 2M$ 与卡数无关。详见 [[NCCL核心机制]] §3.1。
>
> ### 速记
>
> - **all-reduce = 人手一份完整结果**，DDP 梯度同步靠它；
> - **reduce = 只有 root 有**，DP 的 GPU0 聚合就是它（加 broadcast）；
> - **all-gather = 拼装**，FSDP 聚合参数靠它；
> - **reduce-scatter = 分片结果**，FSDP/ZeRO 切梯度靠它。
>
> 这套原语的全套解释见 [[NCCL核心机制]] §3.5 与 [[all-reduce]]。
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
