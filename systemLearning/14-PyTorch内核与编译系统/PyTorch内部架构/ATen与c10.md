# ATen与c10

> **所属章节**: [[PyTorch内部架构]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: ATen / c10 / Core 10 / A Tensor Library / TensorImpl / Storage / ScalarType / PyTorch C++ 内核
> **难度**: 中高（需懂 C++ + [[Tensor]] 使用层）


## 1. 一句话定义

**ATen 与 c10** 是 PyTorch 的 **C++ 底层**：**c10（Core 10）** 是最底层基础库（无依赖），定义 **`TensorImpl`/`Storage`/`ScalarType`/`DispatchKey`/`DispatchKeySet`** 等核心数据结构与分发原语；**ATen（A Tensor Library）** 是建在 c10 之上的算子库，定义 `torch::add`/`torch::mm` 等数百个 native tensor operation 的实现（CPU/CUDA/backend 无关声明 + 各后端 kernel）。Python 层的 `torch.Tensor` 只是一个**指向 c10 `TensorImpl` 的智能指针**（`intrusive_ptr`，handle 语义），所有 op 经 [[Dispatcher]] 路由到 ATen 的对应后端实现。简言之：**c10 = 数据结构 + 分发骨架，ATen = 算子声明与实现，二者构成 PyTorch 内核的 C++ 地基**。

> [!note] 三句话定位
> - **是什么**：c10 是底层基础库（TensorImpl/Storage/ScalarType/DispatchKey），ATen 是其上的算子库（add/mm/matmul 等的声明 + 各后端实现）。
> - **为什么**：Python 太慢，需 C++ 内核承载真实计算；分层让算子声明与后端实现解耦（同一 `add` 路由到 CPU/CUDA/MPS/Meta 等）。
> - **与 [[Tensor]] 关系**：用户层 `torch.Tensor`（[[Tensor]]）是 c10 `TensorImpl*` 的 handle，所有 `.add()` 经 [[Dispatcher]] 落到 ATen 后端 kernel。


## 2. 为什么需要它（动机与背景）

### 2.1 Python 慢，需 C++ 内核

PyTorch 前端是 Python（易用、动态图），但真实计算不能在 Python 跑（GIL + 解释器开销）。故内核用 C++ 写（c10 + ATen），Python 层只是 binding（`pybind11` 或 `torch._C` 扩展）。`torch.add(a, b)` 实际进 C++ 的 `at::add`。

### 2.2 算子声明与后端实现解耦

同一 `add` 要在 CPU、CUDA、MPS、XLA、Meta（shape 推导）、CompositeImplicitAutograd（autograd 公式组合）等多后端跑。若每后端各自声明，重复且难维护。ATen 的 native functions（`native_functions.yaml`）声明一次"算子签名 + autograd 公式"，各后端（CPU/CUDA/...）各自提供 kernel 实现，[[Dispatcher]] 按输入 tensor 的 dispatch key 路由。**声明与实现解耦**是核心设计。

### 2.3 c10 的"零依赖"原则

c10 只依赖 C++ STL 与极少数头文件，不依赖 ATen/Python/CUDA。这让 c10 可被最底层组件（如 `CUDACachingAllocator`、`intrusive_ptr`、logging）复用而不引入循环依赖。ATen 依赖 c10，Python binding（`torch._C`）依赖 ATen。分层清晰。

### 2.4 handle 语义的 Tensor

`torch::Tensor`（C++）/ `torch.Tensor`（Python）是**值类型 handle**，内部 `intrusive_ptr<TensorImpl>` 指向共享的 `TensorImpl`。拷贝 Tensor 只拷指针（O(1)），不拷数据；多个 Tensor 可 view 同一 Storage。这是 [[Tensor]] view/stride 语义的底层。


## 3. 核心概念详解

### 3.1 c10 的核心数据结构

| 结构 | 作用 |
|---|---|
| **`TensorImpl`** | tensor 的实现：sizes_/strides_/storage_offset_/dtype_/device_/`Storage`_，是 `torch::Tensor` 指向的对象 |
| **`Storage`** | 数据缓冲：`data_ptr`（`DataPtr`，含 deleter）+ size，是 `TensorImpl` 持有的底层 |
| **`ScalarType`** | dtype 枚举：`Float`/`Double`/`Long`/`BFloat16`/`Bool`/...（见 [[数值类型与精度]]） |
| **`DispatchKey`** | 分发键：`CPU`/`CUDA`/`AutogradCPU`/`Meta`/`CompositeImplicitAutograd`/...（见 [[Dispatcher]]） |
| **`DispatchKeySet`** | DispatchKey 的位集，表示一个 tensor 携带的所有分发键（如 CUDA + AutogradCUDA + ...） |
| **`intrusive_ptr`** | 引用计数智能指针（非线程安全 refcount，c10 自有版），Tensor/TensorImpl/Storage 共享语义 |
| **`Device`** / **`DeviceGuard`** | 设备抽象（type=index）+ RAII 设备切换 |
| **`Allocator`** | 内存分配器接口，CPU `CPUAllocator`/CUDA `CUDACachingAllocator`（见 [[CUDA allocator与memory pool]]） |

### 3.2 TensorImpl 内部（简化）

```cpp
struct TensorImpl : public c10::intrusive_ptr_target {
  Storage storage_;
  DispatchKeySet key_set_;          // 该 tensor 的 dispatch keys
  int64_t sizes_[...];              // shape
  int64_t strides_[...];
  int64_t storage_offset_;
  ScalarType data_type_;
  Device device_;
  bool requires_grad_;
  // ...
};
```

`torch::Tensor` = `intrusive_ptr<TensorImpl>` 的薄包装。`.sizes()` / `.strides()` / `.data_ptr()` 都转发到 `TensorImpl`。

### 3.3 ATen 的算子声明

ATen 用 `native_functions.yaml` 声明算子（签名 + autograd 公式 + dispatch 分类）：

```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  dispatch:
    CPU, CUDA, Meta, CompositeImplicitAutograd  # 哪些后端
```

代码生成器（`tools/gen_*.py`）从 yaml 生成 C++ 声明 `at::add`、dispatch 模板、Python binding、autograd 公式调用。`native_functions.yaml` 是 ATen 的单一真相源。

### 3.4 ATen 的后端实现

每后端有 `aten/src/ATen/native/`（CPU、Composite）、`aten/src/ATen/native/cuda/`（CUDA）等，实现 yaml 声明的 kernel：

```cpp
// aten/src/ATen/native/cuda/BinaryOps.cu
Tensor add_kernel_cuda(const Tensor& self, const Tensor& other, Scalar alpha) {
  // 调 thrust/cub 或手写 kernel
}
```

[[Dispatcher]] 按 tensor 的 DispatchKeySet 路由到对应 `add.*` 实现。

### 3.5 分层依赖

```
torch (Python) ──> torch._C (pybind11 binding)
                         │
                        ATen (at::add 算子声明 + 各后端 kernel + codegen)
                         │
                        c10 (TensorImpl/Storage/ScalarType/DispatchKey/Allocator)
                         │
                        C++ STL
```

c10 不依赖 ATen；ATen 依赖 c10；binding 依赖 ATen。循环依赖被分层杜绝。

### 3.6 CompositeImplicitAutograd

很多 op 不需每后端单独写——用 autograd 公式组合已有 op 即可（如 `elu = exp(x)-1` 用 `exp`/`sub` 组合）。ATen 标 `CompositeImplicitAutograd`，dispatcher 自动用组合实现 + autograd 公式，免每后端重复。减少代码量，是 ATen 算子数膨胀可控的关键。

### 3.7 Meta backend（shape 推导）

`Meta` dispatch key 不真算，只推导输出 shape（返回空数据 tensor 带 size）。`torch.compile`/tracing/Meta tensor 用它静态推 shape，不执行真计算（见 [[dynamic shape]]/[[TorchDynamo]]）。

### 3.8 与 CUDACachingAllocator

c10 的 `Allocator` 接口被 CUDA 后端的 `c10::cuda::CUDACachingAllocator` 实现，Tensor 的 `Storage` 持有的 `DataPtr` 由它分配。详见 [[CUDA allocator与memory pool]]。


## 4. 数学原理 / 公式

无复杂公式。核心是数据结构 + 分发机制。关键量：

### 4.1 Storage 与 view 的关系

tensor $T$ 的 view $T'$ 共享 Storage：

$$
\text{offset}(T') + \text{index}(i) = \text{offset}(T) + \sum_k s_k \cdot \text{idx}_k
$$

stride $s_k$ 与 storage_offset 共同定位元素。多 view 共享 data_ptr，零拷贝。

### 4.2 DispatchKeySet 的位编码

`DispatchKeySet` 是 64-bit，每位对应一个 DispatchKey。`tensor.key_set()` 返回其所有键（如 CUDA + AutogradCUDA + ...）。[[Dispatcher]] 用 key_set 选 kernel。位编码 O(1)。

### 4.3 intrusive_ptr refcount

`intrusive_ptr` refcount 非原子（默认），快但非线程安全；线程安全版 `intrusive_ptr_target` 的 `refcount` 用原子。Tensor 拷贝 ++ref，析构 --ref，归零 deleter 释放 Storage。view 共享 storage 故 storage 有独立 refcount。


## 5. 代码示例（可选）

### 5.1 Python 层观察 handle 语义

```python
import torch
a = torch.randn(4, 4)
b = a.view(2, 8)              # view: 共享 storage
print(a.storage().data_ptr() == b.storage().data_ptr())  # True, 同 Storage
# a/b 是不同 TensorImpl, 但指向同一 Storage
```

### 5.2 C++ 层观察 TensorImpl（概念）

```cpp
#include <ATen/ATen.h>
at::Tensor a = at::randn({4, 4});     // 调 at::randn -> Dispatcher -> CUDA kernel
// a 是 intrusive_ptr<TensorImpl>
auto* impl = a.unsafeGetTensorImpl(); // 取底层 impl (仅概念, 实际用 unsafeReleaseTensorImpl 等)
impl->sizes();                        // {4,4}
impl->strides();                     // {4,1}
impl->storage();                     // Storage (data_ptr + size)
// ScalarType
at::ScalarType dt = a.scalar_type();  // at::kFloat / kBFloat16 / ...
```

### 5.3 native_functions.yaml 声明（简化）

```yaml
# 声明一次, 多后端实现
- func: mm(Tensor self, Tensor mat2) -> Tensor
  dispatch:
    CPU, CUDA, Meta
  autogen:  # 自动生成 autograd 公式
    - mm
```

codegen 生成 `at::mm` 声明 + dispatch 模板 + Python binding + autograd 公式。


## 6. 与其他知识点的关系

- **上游（依赖）**: C++ STL、`intrusive_ptr` 智能指针模式。
- **下游（应用）**: [[Dispatcher]]（用 c10 的 DispatchKey/DispatchKeySet 路由）、[[Autograd Engine]]（用 c10 的 `Node`/`grad_fn`）、[[Custom C++ CUDA Operator]]（经 `torch.library`/`TORCH_LIBRARY` 注册到 dispatcher）、[[torch.compile]]/[[Inductor]]（codegen 生成 ATen op 调用）、[[CUDA allocator与memory pool]]（c10 `Allocator` 接口的 CUDA 实现）。
- **对比 / 易混**:
  - **c10 vs ATen**：c10 是数据结构 + 分发骨架（零依赖）；ATen 是算子声明 + 各后端实现（依赖 c10）。
  - **ATen vs [[Tensor]]（使用层）**：[[Tensor]] 是用户视角（Python `torch.Tensor`）；ATen 是其 C++ 实现 + 算子库。
  - **ScalarType vs [[数值类型与精度]]**：c10 ScalarType 是 dtype 枚举（C++ 层）；[[数值类型与精度]] 是 dtype 的数值语义（fp16/bf16/fp8 的范围/精度/用途）。
  - **ATen Storage vs [[CUDA allocator与memory pool]]**：Storage 是 tensor 持有的数据缓冲抽象；CUDACachingAllocator 是 Storage 的 CUDA 内存池实现。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 torch.Tensor 直接含数据
> `torch.Tensor` 是指向 `TensorImpl` 的 handle（`intrusive_ptr`），`TensorImpl` 持有 `Storage`，`Storage` 持有 `DataPtr`。拷贝 tensor 只拷指针，view 共享 Storage。

> [!warning] 误区 2：混淆 c10 与 ATen 的职责
> c10 是数据结构 + DispatchKey 骨架（零依赖）；ATen 是算子声明（native_functions.yaml）+ 各后端 kernel 实现。改算子行为找 ATen，改数据结构找 c10。

> [!warning] 误区 3：以为每后端各自声明算子
> 算子在 `native_functions.yaml` 声明一次，各后端提供 kernel 实现，codegen 生成 dispatch 模板。新增后端只需加 dispatch key + 实现，不动声明。

> [!warning] 误区 4：忽视 CompositeImplicitAutograd
> 很多 op 不需每后端手写——用已有 op 组合 + autograd 公式即可（标 `CompositeImplicitAutograd`）。滥用单独实现会重复且难维护。

> [!warning] 误区 5：混淆 ScalarType 与 device
> dtype（ScalarType）与 device 是 tensor 的两个独立属性。`at::kFloat` 在 CPU 与 CUDA 是同一 ScalarType，但 device 不同 → dispatcher 路由到不同后端 kernel。

> [!warning] 误区 6：以为 intrusive_ptr 是 std::shared_ptr
> c10 的 `intrusive_ptr` 是自有版（非线程安全 refcount 默认，快），不是 `std::shared_ptr`。误用 `std::` 版会引入性能开销。


## 8. 延伸细节

### 8.1 native_functions.yaml 与 codegen

ATen 的算子声明集中在 `aten/src/ATen/native/native_functions.yaml`，codegen 脚本（`tools/gen_*.py`）生成：C++ 声明 `at::add`、dispatch 模板、Python binding（`torch._C`）、autograd 公式调用、`CompositeImplicitAutograd` 组合。新增算子改 yaml + 加后端实现，不手写 binding。

### 8.2 DispatchKey 的层次

DispatchKey 有功能性与后端性两类：`CPU`/`CUDA`/`MPS` 是后端键，`Autograd`/`AutogradCPU` 是功能性键，`Meta`/`CompositeImplicitAutograd` 是推导/组合键。dispatcher 按"功能性优先 + 后端 fallback"的优先级路由。详见 [[Dispatcher]]。

### 8.3 与 [[Custom C++ CUDA Operator]] 的接口

自定义 op 用 `torch.library`/`TORCH_LIBRARY` 宏注册到 dispatcher，与 ATen native op 同等地位（有 DispatchKey、autograd 公式、Meta 实现）。是扩展 PyTorch 内核的标准接口。详见 [[Custom C++ CUDA Operator]]。

### 8.4 Meta tensor 与编译

Meta backend 返回空数据但带正确 shape 的 tensor。[[torch.compile]]/[[TorchDynamo]] 用 Meta tensor 静态推 shape（不执行真计算），是编译期 shape 推导的基础。详见 [[dynamic shape]]。

### 8.5 内容来源

c10/ATen 架构整理自 PyTorch 源码 `c10/`、`aten/src/ATen/` 目录结构与 `native_functions.yaml`，以及 PyTorch ARCHITECTURE.md / dev docs 的分层说明。ScalarType/DispatchKey 见 `c10/core/`。

---
相关: [[PyTorch内部架构]] | [[Dispatcher]] | [[Autograd Engine]] | [[Custom C++ CUDA Operator]] | [[torch.compile]] | [[Inductor]] | [[CUDA allocator与memory pool]] | [[Tensor]] | [[数值类型与精度]] | [[计算图]] | [[nn.Module]] | [[forward与backward]]
