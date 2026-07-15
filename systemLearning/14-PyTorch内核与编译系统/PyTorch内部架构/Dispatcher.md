# Dispatcher

> **所属章节**: [[PyTorch内部架构]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: Dispatcher / dispatch key / DispatchKeySet / operator registration / dispatch table / 分发器 / 算子分发
> **难度**: 中高（需懂 [[ATen与c10]] + autograd）


## 1. 一句话定义

**Dispatcher** 是 PyTorch 的**算子路由中枢**——每个 tensor 携带一组 **DispatchKey**（如 `CUDA` + `AutogradCUDA` + `Batched` + ...）组成 **DispatchKeySet**（位编码），调用 `at::add(t1, t2)` 时，dispatcher 取输入 tensor 的 key_set，按**优先级**（功能性键如 Autograd 优先于后端键如 CUDA）查**算子注册表**（dispatch table），找到对应后端/功能的 kernel 执行。它让"一个 op 声明、多后端实现 + autograd + functional passes"成为可能——新增 backend（如 NPU/TPU）或功能（如 autograd/`torch.compile` tracing/`vmap`）只需注册新 DispatchKey 的 kernel，不动算子声明。是 [[ATen与c10]] 分层设计落到"动态路由"的关键一环。

> [!note] 三句话定位
> - **是什么**：算子路由中枢，按 tensor 携带的 DispatchKeySet 查注册表选 kernel。
> - **为什么**：一个 `add` 要在 CPU/CUDA/Meta/...跑，且要插 autograd、编译 tracing 等功能层；dispatcher 把"选哪个 kernel + 插哪些 pass"统一成一次查表。
> - **核心机制**：DispatchKeySet 位编码 + 优先级路由（Autograd 优先于后端），算子经 `m.impl(name, &kernel)` 注册到 dispatch table。


## 2. 为什么需要它（动机与背景）

### 2.1 一个 op 多后端

`torch.add` 在 CPU、CUDA、MPS、XLA 都要能跑。若 Python 层 if/else 选后端，臃肿且难扩展。dispatcher 把"选后端"下沉到 C++ 内核，按 tensor 的 device/dtype 自动路由。新增 backend 只加 DispatchKey + 注册 kernel，不动用户代码。

### 2.2 功能层叠加

除了"选后端"，还要插：
- **Autograd**：`at::add` 实跑前要经 Autograd 层（记录 grad_fn，构造计算图，见 [[Autograd Engine]]）。
- **tracing/编译**：`torch.compile`/FX tracing 要捕获 op 调用，不真算而建图。
- **`vmap`/Batched**：批量维度变换。
- **Python fallback**：某些 op 只在 Python 实现。

这些是"功能键"（Autograd/Batched/Python/...），与"后端键"（CPU/CUDA/...）正交。dispatcher 按"功能优先于后端"的优先级，让 autograd 在真 kernel 前先插（记录 grad），再调真 kernel。一个查表统一多 pass。

### 2.3 动态分发 vs 静态 if/else

tensor 的 key_set 运行时才知（dtype/device/grad flag 变）。dispatcher 是**运行时分发**（按 key_set 查表），不是编译期 if/else。这让 PyTorch 动态图语义天然支持混合 dtype/device 的灵活调度。

### 2.4 扩展性

[[Custom C++ CUDA Operator]] 用 `torch.library`/`TORCH_LIBRARY` 把自定义 op 注册进 dispatcher，与 native op 同等。新 backend（如 NPU）注册自己的 DispatchKey kernel。dispatcher 是 PyTorch 内核可扩展性的核心抽象。


## 3. 核心概念详解

### 3.1 DispatchKey

`DispatchKey` 是枚举（c10 定义），分两类：

| 类别 | 示例 | 作用 |
|---|---|---|
| **后端键** | `CPU`/`CUDA`/`MPS`/`XLA`/`Meta` | 选具体硬件后端 kernel |
| **功能性键** | `Autograd`/`AutogradCPU`/`AutogradCUDA`/`Batched`/`Python`/`Functionalize`/`PythonTLSSnapshot` | 插功能层（autograd/vmap/...） |

`Meta` 是特殊后端键——只推 shape 不真算（见 [[ATen与c10]] 3.7）。`CompositeImplicitAutograd` 是组合实现键（见 [[ATen与c10]] 3.6）。

### 3.2 DispatchKeySet

一个 tensor 携带**多个** DispatchKey（如 CUDA tensor 且 `requires_grad=True` → `{CUDA, AutogradCUDA}`）。`DispatchKeySet` 是 64-bit 位集，每位对应一键。`tensor.key_set()` 返回它。dispatcher 按 key_set 路由。

### 3.3 优先级路由

dispatcher 不是"取第一个匹配键"，而是按**预定义优先级**遍历 key_set 的键，取第一个有注册 kernel 的：

```
Autograd* (功能性, 最高) > 后端键 (CPU/CUDA) > Composite* (fallback)
```

故 `at::add(cuda_tensor_requires_grad)` 先命中 `AutogradCUDA`（autograd 层），autograd kernel 内部记录 grad_fn 后**手动 redispatch** 到下一优先级 `CUDA`（真 kernel）。这种"逐层 redispatch"是 autograd/tracing/vmap 插层的机制。

### 3.4 算子注册（OperatorRegistration）

```cpp
// 简化: 注册 add 的 CUDA kernel
m.impl("add.Tensor",
       torch::dispatch(c10::DispatchKey::CUDA, &add_kernel_cuda));
// autograd kernel
m.impl("add.Tensor",
       torch::dispatch(c10::DispatchKey::AutogradCUDA, &add_autograd));
```

`native_functions.yaml`（[[ATen与c10]]）声明的 op 由 codegen 自动注册；自定义 op 用 `TORCH_LIBRARY` 手动注册（见 [[Custom C++ CUDA Operator]]）。注册表 = `OperatorEntry`，含每 DispatchKey 的 kernel 指针。

### 3.5 dispatch 表与查找

每 op 一个 `OperatorEntry`，内部 `dispatch_table_` 是 `DispatchKey → kernel` 的数组（按 key 索引）。调用时 dispatcher：

1. 取输入 tensor 的 `key_set`。
2. 按 key_set 优先级遍历，找第一个注册了 kernel 的 key。
3. 调该 kernel。kernel 内可 `redispatch`（手动调下一优先级）。

O(1) 查表（位集 + 数组索引），无哈希开销。

### 3.6 autograd 层的 redispatch

`AutogradCUDA` 的 kernel 不真算，而是：
1. 创建 `grad_fn`（`AddBackward`），连入 autograd 图。
2. 调 `redispatch` 到 `CUDA` 真 kernel，得输出 tensor。
3. 把输出 `grad_fn` 设为上述 `grad_fn`，`requires_grad=True`。

下次 `.backward()` 时 [[Autograd Engine]] 遍历 grad_fn 图算梯度。autograd 层是 dispatcher 最常用的功能性插层。

### 3.7 与 torch.compile 的关系

`torch.compile`（见 [[TorchDynamo]]）把 Python 函数 trace 成 graph，graph 里是 ATen op 调用。Inductor 后端（见 [[Inductor]]）把这些 op 重新 codegen 成 fused kernel。但 dispatcher 仍是运行时 fallback：编译失败的 op 走 dispatcher 原 kernel。graph capture 与 dispatcher 互补。

### 3.8 TLS（thread-local state）与 fallback

某些键（如 `Python`/`Autograd`）的状态是 thread-local（如当前是否在 autograd 模式、当前 Python 上下文）。dispatcher 读 TLS 决定 key_set 补充。这让 `torch.no_grad()` 临时关 autograd（从 key_set 移除 Autograd 键）成为可能。

### 3.9 DispatchKey 的"运行时拼装"

tensor 的 key_set 不是固定——`requires_grad=True` 加 Autograd 键，`torch.no_grad()` 移除，`torch.compile` 加 tracing 键，`vmap` 加 Batched 键。dispatcher 在每次调用时**实时拼装 key_set**（含 TLS），按当前上下文路由。这是动态图灵活性的来源。


## 4. 数学原理 / 公式

无复杂数学。核心是路由算法：

### 4.1 路由优先级

设 key_set $K$，预定义优先级序 $\prec$（Autograd $>$ 后端 $>$ Composite）。kernel 注册表 $R: (\text{op}, \text{key}) \to \text{kernel}$。

$$
\text{kernel}(op, K) = R\left(op,\ \min_{\prec}\{k \in K \mid (op,k) \in R\}\right)
$$

取优先级最高且已注册的 key 的 kernel。Autograd kernel 内部 $\text{redispatch}$ 到 $\min_{\prec}\{k \in K \setminus \{\text{Autograd}\}\}$，逐层下沉到真后端。

### 4.2 DispatchKeySet 位编码

$K = \sum_{i} 2^{k_i}$（位集）。`has(key)` 是位与，`add(key)` 是位或。O(1)。

### 4.3 调用开销

每次 `at::op` 调用 = 取 key_set（读 TLS + tensor 字段）+ 优先级遍历（~常数）+ kernel 指针调用。开销 ~几十 ns。是 Python 层 op 调用的主要 C++ 内开销（除 kernel 本身）。torch.compile 通过融合/去 dispatcher 提速（见 [[torch.compile]]）。


## 5. 代码示例（可选）

### 5.1 Python 层观察 key_set

```python
import torch
x = torch.randn(4, device='cuda', requires_grad=True)
print(x._has_rmeta)  # 内部
# torch._C._dispatch_key_set(x) 概念: {CUDA, AutogradCUDA, ...}
# at::add(x, x) 先命中 AutogradCUDA -> 记 grad_fn -> redispatch CUDA -> 真 kernel

with torch.no_grad():   # TLS 关 autograd
    y = x + x           # key_set 无 Autograd, 直奔 CUDA kernel, 不建图
```

### 5.2 C++ 注册（概念）

```cpp
#include <torch/library.h>
TORCH_LIBRARY(myops, m) {
  m.def("foo(Tensor x) -> Tensor");       // 声明 op
  m.impl("foo", torch::dispatch(c10::DispatchKey::CPU, &foo_cpu));
  m.impl("foo", torch::dispatch(c10::DispatchKey::CUDA, &foo_cuda));
  m.impl("foo", torch::dispatch(c10::DispatchKey::AutogradCUDA, &foo_autograd));
}
// 调 at::foo(x) -> dispatcher 按 x.key_set() 路由
```

### 5.3 autograd redispatch（概念）

```cpp
Tensor foo_autograd(const Tensor& x) {
  auto grad_fn = std::make_shared<FooBackward>(x);  // 建图
  auto out = at::_redispatch_from_autograd(x);      // redispatch 到 CUDA 真 kernel
  out.grad_fn_ = grad_fn;                            // 挂 grad_fn
  return out;
}
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[ATen与c10]]（DispatchKey/DispatchKeySet/TensorImpl 定义在 c10）、`OperatorEntry`/dispatch table 数据结构。
- **下游（应用）**: [[Autograd Engine]]（Autograd 键的 grad_fn 插层）、[[Custom C++ CUDA Operator]]（`TORCH_LIBRARY` 注册自定义 op）、[[torch.compile]]/[[TorchDynamo]]（trace 时捕获 dispatcher op 调用，编译失败 fallback 到 dispatcher）、[[Inductor]]（codegen 的 op 仍是 dispatcher 注册的 ATen op）、`vmap`/`funcionalize`/`torch.jit` 等功能键。
- **对比 / 易混**:
  - **Dispatcher vs [[Autograd Engine]]**：Dispatcher 是 op 调用的路由（每次 op 调用都过）；Autograd Engine 是反向遍历 grad_fn 图的执行器（只 `.backward()` 时跑）。Dispatcher 在前向插 autograd 层建图，Autograd Engine 在反向跑图。
  - **Dispatcher vs [[TorchDynamo]] tracing**：Dispatcher 是运行时路由（每调用一次路由一次）；Dynamo 是 Python 字节码 trace 成静态 graph（一次 trace 多次跑）。Dynamo graph 的 op 仍是 dispatcher 注册的，但跑时不逐次路由（已 codegen）。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 dispatcher 选第一个匹配的键
> 是按**预定义优先级**（Autograd > 后端 > Composite），不是 key_set 里的顺序。Autograd 键优先于 CUDA，故 `requires_grad` 的 CUDA tensor 先过 autograd 层。

> [!warning] 误区 2：忽视 redispatch
> autograd kernel 不真算，而是记录 grad_fn 后 redispatch 到后端真 kernel。不是"autograd 算一遍 + 后端算一遍"。一次 op 调用经多层，但真算只在最底后端。

> [!warning] 误区 3：以为 key_set 是静态的
> key_set 运行时拼装：`requires_grad`、`torch.no_grad()`、`torch.compile` tracing、`vmap` 都动态改 key_set（加/移键）。同一 tensor 在不同上下文 key_set 不同。

> [!warning] 误区 4：混淆 DispatchKey 与 device
> device（CPU/CUDA）是后端键的一部分，但 key_set 还含功能键（Autograd/Batched/...）。`tensor.device()` 只看后端，`tensor.key_set()` 看全部。dispatcher 用完整 key_set。

> [!warning] 误区 5：忽视 TLS 对 key_set 的影响
> `torch.no_grad()` 通过 TLS 临时移除 Autograd 键，影响当前线程所有 op 的路由。跨线程（如 DataLoader worker）TLS 不共享，autograd 状态独立。

> [!warning] 误区 6：以为自定义 op 不经 dispatcher
> `torch.library`/`TORCH_LIBRARY` 注册的 op 与 native op 同等经 dispatcher 路由。自定义 op 也要声明 dispatch key + autograd 公式，否则 `requires_grad` 的输入不会建图。


## 8. 延伸细节

### 8.1 DispatchKey 的演化

早期 PyTorch 用 `Backend`（CPU/CUDA）+ `VariableType`（autograd）双层。后重构为统一 DispatchKey 体系（更灵活、支持多功能键叠加）。现 DispatchKey 数十个，含 Autograd 变体、Python、Functionalize、PythonTLSSnapshot 等。

### 8.2 boxed vs unboxed kernel

dispatcher 的 kernel 有 boxed（接受 `Stack` 参数，通用）与 unboxed（直接 C++ 签名，快）两种注册方式。codegen 默认生成 unboxed（快），自定义/动态场景用 boxed。性能差 ~ns 级，热路径用 unboxed。

### 8.3 与 [[torch.compile]] 的协同与 fallback

Dynamo trace 出的 graph 调 ATen op，Inductor codegen 成 fused kernel。但若某 op 编译失败（见 [[graph break]]），fallback 到 dispatcher 原 kernel 跑。dispatcher 是编译路径的安全网。详见 [[TorchDynamo]]/[[Inductor]]。

### 8.4 DispatchKey 与新 backend

新 backend（如华为 NPU `PrivateUse1`、TPU）注册自己的 DispatchKey kernel，用户代码不变。是 PyTorch 多硬件生态的关键。详见 PyTorch dev docs "Registering a Dispatch Key"。

### 8.5 内容来源

Dispatcher 机制整理自 PyTorch `c10/core/DispatchKey.h`、`aten/src/ATen/core/Dispatcher`、`torch/library.h`，以及 PyTorch dev blog "PyTorch Dispatcher" 与 ARCHITECTURE.md。优先级与 redispatch 见 `DispatchKeySet` 的 `highestPriorityTypeId` 实现。

---
相关: [[PyTorch内部架构]] | [[ATen与c10]] | [[Autograd Engine]] | [[Custom C++ CUDA Operator]] | [[torch.compile]] | [[TorchDynamo]] | [[Inductor]] | [[graph break]] | [[计算图]] | [[autograd机制]] | [[forward与backward]]
