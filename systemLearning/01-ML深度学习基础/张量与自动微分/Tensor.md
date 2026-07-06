# Tensor

> **所属章节**: [[张量与自动微分]]
> **所属模块**: [[01-ML深度学习基础]]
> **难度**: 基础

## 1. 一句话定义

**张量（Tensor）** 是多维数组的泛化形式，是 PyTorch 中最基本的数据结构；它由 **shape（形状）**、**dtype（数据类型）**、**device（所在设备）** 三个核心属性定义，可以是一维标量向量、二维矩阵，也可以是任意高维数组。

## 2. 为什么需要它（动机与背景）

深度学习的本质是**用数值计算去拟合函数**，而现实世界的数据（图像、文本、语音、参数权重）天然是多维的：

- 一张灰度图是 `[H, W]` 二维；
- 一张彩色图是 `[C, H, W]` 三维；
- 一个 batch 的彩色图是 `[N, C, H, W]` 四维；
- Transformer 的注意力分数是 `[B, H, S, S]` 四维。

如果只用 Python list 或 NumPy，无法做到三件事：

1. **GPU 加速**：list/np.ndarray 不能直接放到 GPU 上做大规模并行计算。
2. **自动微分**：训练需要梯度，普通数组没有"记录运算历史"的能力。
3. **批量高效算子**：深度学习里 matmul、conv、layernorm 要针对张量形状做高度优化。

所以需要一种"**可放 GPU、可记录梯度、形状灵活**"的数据结构 —— 这就是 Tensor。

> [!note] 解答：tensor 是怎么做到自动微分、怎么记录梯度的？
> Tensor 本身只是"多维数组 + 三个属性"，它**自己不会求导**。自动微分是靠 PyTorch 的 **autograd 引擎 + 计算图**完成的，但 Tensor 提供了三个"钩子"让这套机制能挂上去：
>
> 1. **`requires_grad`（求导开关）**：给一个 Tensor 设 `requires_grad=True`，就等于告诉 autograd "这个张量是变量，请追踪对它的运算"。`nn.Parameter` 自动设为 `True`，所以模型权重天然可训练。
> 2. **`grad_fn`（反向函数指针，建图的证据）**：一旦 `requires_grad=True` 的张量参与了运算，PyTorch 在**执行前向的同时**（define-by-run，见 [[计算图]]）会顺手给"运算结果"挂一个 `grad_fn` 属性，指向"产生它的那个算子的反向实现"。比如 `y = W @ x + b`，`y.grad_fn` 就是 `AddBackward0`，它内部又链到 `MatMulBackward`、指向 `b` 的叶子。这样一串 `grad_fn` 指针，就把前向所有运算**串成了一张隐式 DAG**——这就是"记录梯度/记录运算历史"的本质：**不是存梯度值，而是存一张可反推的图**。
> 3. **`.grad`（梯度累加槽）**：反向传播时，autograd 沿 `grad_fn` 链从 loss 反向走，每经过一个算子就调用它的 `backward`（向量-雅可比积 vjp），把算出的梯度**累加**到对应**叶子节点**的 `.grad` 属性里。注意是累加不是覆盖，所以每步训练前要 `optimizer.zero_grad()`。
>
> 一句话：**`requires_grad` 决定"要不要记"，`grad_fn` 负责"记成图"，`.grad` 负责"存结果"**。三者在 Tensor 上合体，调用 `.backward()` 时 autograd 沿 `grad_fn` 图反推，梯度落进 `.grad`。完整机制见 [[autograd机制]]、[[backward过程]]。
>
> ```python
> import torch
> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # 开启：要记
> y = (x ** 2).sum()                                     # 前向时自动挂 grad_fn
> print(y.grad_fn)                                       # <SumBackward0> —— 图建好了
> y.backward()                                           # 沿图反推
> print(x.grad)                                          # tensor([2., 4., 6.]) —— 梯度落进 .grad
> ```

## 3. 核心概念详解

### 3.1 三大属性

> [!note] 解答：为什么要记录 device？
> 张量底层是一块**真实内存**，而 CPU 内存和每张 GPU 显存是**物理上隔开的不同地址空间**——一个张量的数据要么在 CPU 内存里，要么在某张 GPU 卡的显存里，不可能"同时在两处"。所以每个 Tensor 必须记 `device`，原因有四：
>
> 1. **算子要派发到正确的 kernel**：同一个 `+`，CPU 张量走 CPU 加法核，`cuda:0` 张量走 GPU 加法核。框架拿到 `device` 才知道该调哪个后端的实现——这就是 **dispatcher（派发器）** 的核心依据。不记 device，连"用哪个核函数"都定不下来。
> 2. **跨设备运算必须先搬运**：GPU 和 CPU 之间走 PCIe/NVLink，搬一次数据比算本身慢几个量级（见 [[CPU-GPU pipeline stall]]、[[memory bandwidth]]）。框架要靠 `device` 字段**提前判断**两个张量是否同处一地：不同设备直接报错 `Expected all tensors to be on the same device`，逼你显式 `.to()` 搬运，而不是偷偷触发一次昂贵的隐式拷贝导致性能黑洞。
> 3. **多卡场景要区分是哪张卡**：`cuda:0`、`cuda:1` 是两块独立显存。分布式训练里张量分在不同 rank 的不同卡上（见 [[rank / world size]]、[[Tensor Parallel]]），`device` 字段决定了它属于哪块卡、和谁通信。记错卡号会导致跨卡拷贝或通信错乱。
> 4. **显存预算与调度**：训练时要精确知道"哪些张量占了哪块卡的多少显存"（见 [[activation memory]]、[[optimizer state memory]]）。没有 `device` 字段，显存账根本算不清，[[ZeRO]]、[[gradient checkpointing]] 这类显存优化也无从下手。
>
> 一句话：**device 不是元信息装饰，而是"数据物理位置"的硬约束**——它决定算子派发、跨设备拷贝判定、多卡归属和显存记账，是分布式/混合精度训练能正确运转的地基。

| 属性       | 含义          | 常见取值                                       |
| -------- | ----------- | ------------------------------------------ |
| `shape`  | 每个维度的大小（元组） | `(2, 3)`, `(4, 768, 768)`                  |
| `dtype`  | 元素的数据类型     | `float32`, `float16/bf16`, `int64`, `bool` |
| `device` | 张量所在设备      | `cpu`, `cuda:0`, `cuda:1`                  |

> [!note] 三者缺一不可
> 同样的数值，`shape` 不同就是不同张量；`dtype` 不同算出来的精度/速度差几个量级；`device` 不同直接决定能不能算（跨设备运算会报错）。

### 3.2 维度术语

- **scalar**（标量）：`shape = ()`，0 维
- **vector**（向量）：`shape = (n,)`，1 维
- **matrix**（矩阵）：`shape = (m, n)`，2 维
- **nd-tensor**：维度 ≥ 3，常叫"高维张量"

在深度学习语境里，维度（dimension / axis / rank）这几个词混用，注意：

- 数学里的"**秩（rank）**"= 维度数（`len(shape)`），不是矩阵的秩。
- PyTorch 里 `.dim()` 和 `.ndim` 都返回维度数。

### 3.3 dtype 的选择

| dtype | 字节 | 典型用途 |
|---|---|---|
| `float32` | 4 | 训练默认（精度/速度平衡） |
| `float16` | 2 | 混合精度训练（快、省显存） |
| `bfloat16` | 2 | LLM 训练主流（动态范围大，不易溢出） |
| `int64` | 8 | 索引、token id |
| `bool` | 1 | mask（如 causal mask、padding mask） |

> [!note] 已展开为独立笔记
> 数值类型（float32/16/bf16/fp8/int 的位级结构、范围与精度权衡、为何 LLM 选 bf16）已单独成篇，见 [[数值类型与精度]]。

> [!warning] bf16 vs fp16
> 两者都是 16 位，但分配不同：fp16 尾数多、指数少（精度高但易溢出）；bf16 指数多、尾数少（动态范围 ≈ fp32，但精度低）。LLM 训练几乎都用 bf16，因为梯度量级大、fp16 容易 inf。

### 3.4 device 与跨设备

- 张量默认在 `cpu`。
- `.to('cuda')` / `.cuda()` 把它搬到 GPU。
- **两个张量做运算必须在同一 device**，否则报错：`Expected all tensors to be on the same device`。

## 4. 数学原理 / 公式

张量本身就是一个**多维数组**，数学上记作 $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \dots \times d_n}$，其中 $\text{shape} = (d_1, d_2, \dots, d_n)$。

### 广播规则（Broadcasting）

当两个张量形状不同但要做逐元素运算时，按以下规则对齐：

1. 从**最右边的维度**开始对齐。
2. 每个维度要么相等，要么其中一个为 1，否则报错。
3. 为 1 的维度被"复制"扩成另一方的尺寸。

例：$A$ 形状 $(3, 1, 5)$，$B$ 形状 $(2, 5)$ → 对齐后都变成 $(3, 2, 5)$。

$$
A_{i,1,k} + B_{j,k} \;\longrightarrow\; C_{i,j,k} = A_{i,1,k} + B_{j,k}
$$

> [!tip] 理解广播
> 广播本质是"在不真正复制内存的前提下，逻辑上把维度扩成对齐尺寸"，所以省显存又省时间。

### stride 与内存布局

张量底层是一块连续内存，`stride` 表示**在每个维度上前进一个元素需要跳过的内存步长**。

- `contiguous`：stride 满足递减且紧凑，可被高效算子直接用。
- `view` 要求张量 contiguous；不连续时要用 `reshape`（必要时自动 copy）。

## 5. 代码示例

```python
import torch

# 1. 创建 —— 三大属性一次性看清
x = torch.randn(2, 3)          # 标准正态分布
print(x.shape, x.dtype, x.device)
# torch.Size([2, 3]) torch.float32 cpu

# 2. 指定 dtype / device
w = torch.zeros(4, 5, dtype=torch.bfloat16, device='cuda')   # 放到 GPU、用 bf16
idx = torch.tensor([0, 2, 1], dtype=torch.int64)             # 索引张量常用 int64
mask = torch.ones(3, 3, dtype=torch.bool).tril()             # 下三角 causal mask

# 3. 形状操作：view vs reshape
a = torch.arange(12)           # shape (12,)
b = a.view(3, 4)               # 不复制内存，仅改形状（要求 contiguous）
c = a.reshape(2, 6)            # 不要求 contiguous，必要时自动 copy
# b 和 a 共享内存：改 b 会影响 a
b[0, 0] = 99
print(a[0].item())             # 99

# 4. 广播
# torch.randn(*size)：从"标准正态分布 N(0,1)"（均值0、方差1）中采样填满指定形状的张量。
# 区分：randn=正态分布(可负); rand=均匀分布[0,1); zeros/ones/empty=常量; arange=等差数列。
# 初始化权重、造模拟数据/激活都用它，因为模型参数初值常需"零均值、小方差"的随机扰动。
A = torch.randn(3, 1, 5)
B = torch.randn(2, 5)
C = A + B                      # 广播成 (3, 2, 5)
print(C.shape)                 # torch.Size([3, 2, 5])

# 5. 跨设备
x_gpu = x.to('cuda')           # cpu -> cuda
# y = x_gpu + x                # 报错！x 在 cpu，x_gpu 在 cuda，不能直接相加
y = x_gpu + x.to('cuda')       # 正确：先搬到同一设备

# 6. contiguous / stride
t = torch.randn(3, 4)
print(t.stride())              # (4, 1)：行间跳 4，列间跳 1
t_t = t.t()                    # 转置，不复制内存
print(t_t.is_contiguous())     # False —— 转置后通常不连续
t_t_c = t_t.contiguous()       # 显式复制成连续
```

## 6. 与其他知识点的关系

- **上游（依赖）**: 无，张量是最底层的数据结构。
- **下游（应用）**: [[计算图]]（张量是计算图的节点）、[[autograd机制]]（张量携带 `requires_grad` 开启自动微分）、[[backward过程]]（梯度存回 `.grad`）。
- **对比 / 易混**:
  - **Tensor vs ndarray**：ndarray 不能放 GPU、不能记录梯度；但做纯数值预处理 ndarray 更轻量。
  - **Tensor vs Parameter**：`nn.Parameter` 就是 `requires_grad=True` 的 Tensor，专用于"可学习参数"。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **跨 device 运算不报"形状错"而报"设备错"**：两个张量形状对、device 不同直接 RuntimeError，新手常以为是形状问题。
> 2. **`view` 用在不连续张量上报错**：转置、切片后张量不 contiguous，要用 `reshape` 或先 `.contiguous()`。
> 3. **dtype 隐式提升**：`float32 + float16` 不是报错而是自动提升到 float32，混合精度里要小心，否则白省显存。
> 4. **`int64` 占 8 字节很大**：索引张量默认 int64 很占显存，超大词表场景可考虑 int32。
> 5. **`a.view(...)` 和 `a` 共享内存**：改 view 出来的张量会改原数据，调试时容易出 bug。
> 6. **0 维张量 vs Python 标量**：`loss` 是 `shape=()` 的 0 维张量，`loss.item()` 才转成 Python float；直接 `float(loss)` 在非 cpu 张量上报错。

## 8. 延伸细节

### 8.1 稀疏张量（Sparse Tensor）

当 90% 元素是 0 时（如大图邻接矩阵），用 `torch.sparse_coo_tensor` 只存非零坐标和值，省显存。GNN、推荐系统里常用。

### 8.2 内存布局与性能

- **C-contiguous**（行优先）：PyTorch 默认，访问最后一维连续时 cache 友好。
- 算子内核通常针对 contiguous 优化，不连续张量会触发隐式 copy 或慢路径。

### 8.3 pinned memory

`torch.tensor(...).pin_memory()` 把 CPU 张量锁页（pinned），异步拷贝到 GPU 时更快，DataLoader 默认开启 `pin_memory=True` 以加速数据加载。

### 8.4 张量的"梯度槽"

当 `requires_grad=True` 时，张量会分配一个 `.grad` 槽位，反向传播后梯度**累加**到这里（不是覆盖）。这就是为什么每个 step 前要 `optimizer.zero_grad()`。

---
相关: [[张量与自动微分]]、[[计算图]]、[[autograd机制]]
