# Custom C++ CUDA Operator

> **所属章节**: [[Custom Operator]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: Custom C++ CUDA Operator / torch.library / TORCH_LIBRARY / 自定义算子 / custom op / extension op / C++ extension
> **难度**: 高（需懂 [[Dispatcher]] + [[ATen与c10]] + CUDA）


## 1. 一句话定义

**Custom C++ CUDA Operator** 是用 `torch.library` / `TORCH_LIBRARY` 宏把**自定义 C++/CUDA kernel 注册进 PyTorch [[Dispatcher]]** 的标准机制——声明算子签名（schema，如 `foo(Tensor x) -> Tensor`）、为各 DispatchKey（CPU/CUDA/Meta/Autograd...）注册 kernel 实现、可选提供 autograd 公式（让 `requires_grad` 输入自动建图）、可选提供 Meta 实现（让 [[torch.compile]]/tracing 推 shape）。注册后自定义 op 与原生 `at::add`/`at::mm` **同等地位**：经 dispatcher 路由、支持 autograd、被 torch.compile trace、可 export。它是把 [[Triton kernel开发]]/[[CUTLASS与GEMM]] kernel 或手写 CUDA kernel 接入 PyTorch 训练流程的标准接口，取代了旧版 `torch.autograd.Function` + `pybind11` 的手拼方案。

> [!note] 三句话定位
> - **是什么**：`TORCH_LIBRARY` 注册自定义 op 到 dispatcher，含 schema + 各后端 kernel + autograd + Meta。
> - **为什么**：手写 kernel（CUDA/Triton/CUTLASS）要接入训练流程（autograd、compile tracing、export），需 dispatcher 注册而非手拼 pybind11。
> - **与 `torch.autograd.Function` 对比**：Function 是 Python 层临时封装（每次 forward 手写 grad）；TORCH_LIBRARY 是 C++ 层一次注册永久可用、与原生 op 同等、被 compile trace。新代码用后者。


## 2. 为什么需要它（动机与背景）

### 2.1 手写 kernel 要接入训练

[[FlashAttention]]、[[fused kernel]]、[[Triton kernel开发]] 的 kernel 性能好，但要进训练流程：`requires_grad` 的输入要自动建 autograd 图、`.backward()` 要自动调对应反向 kernel、torch.compile 要能 trace 它。若只在 Python 写个函数包 kernel，autograd 与 compile 都不认识它。需把 kernel **注册成 dispatcher 认识的 op**。

### 2.2 旧方案的痛点

旧方案是 `torch.autograd.Function`（Python 层定义 forward/backward）+ `pybind11` 把 C++ kernel 绑到 Python。痛点：
- 每次调用经 Python 层 Function，有 Python 开销。
- compile 不认识 Function 内的 C++ kernel（graph break，见 [[graph break]]）。
- 不支持 Meta（shape 推导）、export（torch.export）。
- 多后端（CPU/CUDA）散在 Python if/else。

新方案 `TORCH_LIBRARY` 把 op 注册到 C++ dispatcher，所有上述问题一次解决：autograd 自动、compile 可 trace、Meta 支持、多后端统一。

### 2.3 schema 是单一真相源

`TORCH_LIBRARY` 用 **schema**（`foo(Tensor x, *, Scalar alpha=1) -> Tensor`）声明 op 的类型，codegen 生成 C++ 声明 + Python binding + autograd 公式调用骨架。与 ATen 的 `native_functions.yaml` 同机制（见 [[ATen与c10]]）。自定义 op 与原生 op 同等。

### 2.4 与 torch.compile / export 的兼容

`TORCH_LIBRARY` 注册的 op 被 [[TorchDynamo]] trace、[[Inductor]] codegen、`torch.export` 序列化识别。是 torch.compile 生态对自定义 op 的硬要求——未注册的 Python kernel 会触发 [[graph break]]。


## 3. 核心概念详解

### 3.1 注册三件套：schema + kernel + autograd

```cpp
#include <torch/library.h>
TORCH_LIBRARY(myops, m) {
  m.def("foo(Tensor x, *, int n=1) -> Tensor");   // schema 声明
  m.impl("foo", torch::dispatch(c10::DispatchKey::CPU, &foo_cpu));   // CPU kernel
  m.impl("foo", torch::dispatch(c10::DispatchKey::CUDA, &foo_cuda));  // CUDA kernel
  m.impl("foo", torch::dispatch(c10::DispatchKey::Meta, &foo_meta));  // shape 推导
}
TORCH_LIBRARY_IMPL(myops, AutogradCUDA, m) {
  m.impl("foo", &foo_autograd);   // autograd 公式 (建图)
}
```

注册后 `at::foo(x)` 经 dispatcher 路由（见 [[Dispatcher]]），与 `at::add` 同等。

### 3.2 schema（算子签名）

schema 用 TorchScript 风格类型语言：

```
foo(Tensor x, *, Scalar alpha=1) -> (Tensor, Tensor)
```

- 类型：`Tensor`/`Tensor?`（optional）/`Tensor[]`/`Scalar`/`int`/`float`/`bool`/`str`。
- `*` 后是 keyword-only 参数。
- 默认值。
- 多返回 `(Tensor, Tensor)`。

codegen 从 schema 生成 C++ 声明与 Python binding，**不手写 binding**。

### 3.3 各 DispatchKey 的 kernel

| DispatchKey | 用途 |
|---|---|
| `CPU`/`CUDA` | 真 kernel 实现 |
| `Meta` | shape 推导（返回空数据但正确 size 的 tensor，见 [[ATen与c10]] 3.7） |
| `AutogradCUDA` | autograd 公式（建 grad_fn，见 [[Dispatcher]] 3.6） |
| `CompositeImplicitAutograd` | 用已有 op 组合 + autograd 自动（免每后端写，见 [[ATen与c10]] 3.6） |

### 3.4 autograd 公式（autograd kernel）

```cpp
Tensor foo_autograd(const Tensor& x, int64_t n) {
  auto grad_fn = std::make_shared<FooBackward>(x);  // Node, 见 Autograd Engine
  auto out = at::_redispatch_from_autograd(x, n);    // redispatch 到 CUDA 真 kernel
  out.grad_fn_ = grad_fn;
  return out;
}
// FooBackward::apply 实现 backward 公式
```

或用 `m.impl("foo", torch::autograd::AutoGradMode)` 的 helper 自动包装。注册后 `requires_grad` 的输入自动建图。

### 3.5 Meta 实现

```cpp
Tensor foo_meta(const Tensor& x, int64_t n) {
  return at::empty_like(x);  // 不真算, 只推 shape
}
```

`torch.compile`/[[TorchDynamo]] tracing 时调 Meta kernel 推 shape（不执行真计算）。未注册 Meta 的 op 在 compile tracing 时会失败（无法静态推 shape）。

### 3.6 Python 端调用

注册后 Python 直接 `torch.ops.myops.foo(x, n=1)`。codegen 从 schema 生成 Python binding。也可 `torch.library.opcheck` 验证 schema/kernel 一致性。

### 3.7 与 torch.compile 的兼容

`TORCH_LIBRARY` 注册的 op：
- [[TorchDynamo]] trace 时识别为 ATen op（不 graph break）。
- [[AOTAutograd]] 分解时被当 native op。
- [[Inductor]] codegen 时若没匹配的 codegen 规则则 fallback 到 dispatcher 原 kernel（不融但能跑）。
- `torch.export` 序列化时作为 op 记录。

未注册的 Python kernel → Dynamo 不认识 → [[graph break]]。故自定义 op 必须用 TORCH_LIBRARY 注册才能 compile-friendly。

### 3.8 与 `torch.autograd.Function` 的关系

`Function` 是 Python 层的 ad-hoc 封装（forward/backward 手写），适合快速 prototype 但不 compile-friendly、不 export-friendly。`TORCH_LIBRARY` 是 C++ 层正式注册，是生产级方案。新代码（尤其要 compile/export 的）用后者；老代码（FlashAttention 早期 `_flash_attn_forward`）逐步迁移到后者。


## 4. 数学原理 / 公式

无复杂数学。核心是注册机制。关键量：

### 4.1 schema 的类型系统

schema 类型语言保证签名静态已知（codegen 生成类型安全 binding）。vs 旧 pybind11 手写 binding（类型松散、易错）。

### 4.2 autograd 公式的链式

自定义 op 的 backward 公式与原生 op 同理（见 [[Autograd Engine]]）：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial x}, \quad y = \text{foo}(x)
$$

autograd kernel 在前向建 `FooBackward` grad_fn（存 $\partial y/\partial x$ 所需的前向值），反向 `apply` 算 $\partial y/\partial x \cdot \frac{\partial L}{\partial y}$。

### 4.3 Meta 的 shape 推导

Meta kernel 返回 $\text{shape}(y) = f(\text{shape}(x), n)$，不分配真数据。是编译期 shape 推导的依据（见 [[dynamic shape]]）。


## 5. 代码示例（可选）

### 5.1 完整 C++ CUDA 自定义 op

```cpp
// myops.cpp
#include <torch/library.h>
#include <ATen/ATen.h>

// CPU kernel
at::Tensor foo_cpu(const at::Tensor& x, int64_t n) {
  return x * n;   // 简化
}
// CUDA kernel (.cu 里)
at::Tensor foo_cuda(const at::Tensor& x, int64_t n) {
  // 调 CUDA kernel (见 Triton/CUTLASS/手写)
  return x * n;
}
// Meta (shape 推导)
at::Tensor foo_meta(const at::Tensor& x, int64_t n) {
  return at::empty_like(x);
}
// autograd
at::Tensor foo_autograd(const at::Tensor& x, int64_t n) {
  auto grad_fn = std::make_shared<FooBackward>(n);
  auto out = at::_redispatch_foo(x, n);   // redispatch 真 kernel
  out.grad_fn_ = grad_fn;
  return out;
}

TORCH_LIBRARY(myops, m) {
  m.def("foo(Tensor x, *, int n=1) -> Tensor");
  m.impl("foo", torch::dispatch(c10::DispatchKey::CPU, &foo_cpu));
  m.impl("foo", torch::dispatch(c10::DispatchKey::CUDA, &foo_cuda));
  m.impl("foo", torch::dispatch(c10::DispatchKey::Meta, &foo_meta));
}
TORCH_LIBRARY_IMPL(myops, AutogradCUDA, m) {
  m.impl("foo", &foo_autograd);
}

// FooBackward (Node 子类)
struct FooBackward : torch::autograd::Node {
  int64_t n;
  FooBackward(int64_t n_) : n(n_) {}
  variable_list apply(variable_list&& grads) override {
    auto grad = grads[0];
    return {grad * n};   // dy/dx = n
  }
};
```

```python
# Python 端
import torch
x = torch.randn(4, device='cuda', requires_grad=True)
y = torch.ops.myops.foo(x, n=3)    # 经 dispatcher -> AutogradCUDA -> CUDA
y.sum().backward()
print(x.grad)   # 3 * ones
# torch.compile 友好
@torch.compile
def f(x): return torch.ops.myops.foo(x, n=3)   # 不 graph break
```

### 5.2 opcheck 验证

```python
torch.library.opcheck(torch.ops.myops.foo.default, (x,), {'n': 3})
# 检查 schema 与 kernel 签名一致、autograd 公式正确、Meta 一致
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Dispatcher]]（注册机制）、[[ATen与c10]]（DispatchKey/Schema/codegen）、[[Autograd Engine]]（autograd kernel 建 grad_fn）、C++/CUDA 编程。
- **下游（应用）**: [[FlashAttention]]/[[fused kernel]]/[[Triton kernel开发]]/[[CUTLASS与GEMM]]（手写 kernel 接入 PyTorch 的标准接口）、[[torch.compile]]/[[TorchDynamo]]/[[Inductor]]（注册的 op 才能 trace 不 graph break）、`torch.export`/torch.fx graph（注册的 op 才能序列化）、[[PyBind11与CMake]]（编译扩展的 build 系统）。
- **对比 / 易混**:
  - **TORCH_LIBRARY vs `torch.autograd.Function`**：前者 C++ 层正式注册（compile/export 友好、多后端统一、codegen binding）；后者 Python 层 ad-hoc（快 prototype 但 graph break、单后端）。新代码用前者。
  - **TORCH_LIBRARY vs [[PyBind11与CMake]]**：TORCH_LIBRARY 注册 op 到 dispatcher（PyTorch 原生机制）；PyBind11 是把 C++ 类/函数绑到 Python 的通用工具（更底层、更通用但不自动 dispatcher/autograd/compile）。两者可并用：TORCH_LIBRARY 注册 op，PyBind11 绑辅助类。
  - **Meta 实现 vs 真实现**：Meta 只推 shape 不算，用于 compile tracing；真实现（CPU/CUDA）才算。两者都要注册才完整。


## 7. 常见误区与易错点

> [!warning] 误区 1：只注册 CUDA kernel 不注册 Meta
> 不注册 Meta，torch.compile tracing 无法推 shape → graph break 或报错。要 compile-friendly 必须给 Meta 实现（哪怕只 `at::empty_like`）。

> [!warning] 误区 2：不注册 autograd kernel
> 不注册 AutogradCUDA，`requires_grad=True` 的输入不会建图 → backward 报错或 grad 为 None。要训练必须注册 autograd 公式（或用 CompositeImplicitAutograd 让 dispatcher 自动组合）。

> [!warning] 误区 3：schema 与 kernel 签名不一致
> schema 声明 `foo(Tensor, int)` 但 kernel 写 `foo(Tensor, int64_t*)` 类型不符 → codegen 报错或运行 UB。用 `opcheck` 验证。

> [!warning] 误区 4：用 pybind11 手拼算子
> 旧 pybind11 方案不 compile/export 友好、不自动 autograd、类型松散。新代码用 TORCH_LIBRARY。pybind11 只用于绑辅助 Python 类（非 op）。

> [!warning] 误区 5：忽视 redispatch
> autograd kernel 不真算，要 `redispatch` 到真后端 kernel。若 autograd kernel 内直接调真 kernel 的私有名而不经 redispatch，dispatcher 链断裂（功能键如 Batched/vmap 不生效）。

> [!warning] 误区 6：自定义 op 不被 Inductor 融合
> 注册的 op 能被 Dynamo trace，但 Inductor 只对已知 op 生成融合 codegen。未知自定义 op 走 dispatcher fallback（不融但仍可跑）。极致性能需 Inductor 加 codegen 规则或手工 Triton kernel 在 graph 外。

> [!warning] 误区 7：混淆 `TORCH_LIBRARY` 与 `TORCH_LIBRARY_IMPL`
> `TORCH_LIBRARY` 注册 namespace + schema + 主后端 kernel；`TORCH_LIBRARY_IMPL` 追加特定 DispatchKey（如 AutogradCUDA）到已有 namespace。分开写因 autograd 常与真 kernel 在不同编译单元（.cpp vs .cu）。


## 8. 延伸细节

### 8.1 torch.library 的 Python 注册

PyTorch 也支持纯 Python 用 `torch.library` 装饰器注册（不必写 C++）：

```python
@torch.library.custom_op("myops::foo", mutates_args=())
def foo(x: torch.Tensor, n: int) -> torch.Tensor:
  return x * n   # 真 kernel (可调 triton/cuda)
@foo.register_fake   # Meta
def _(x, n): return torch.empty_like(x)
@foo.register_autograd
def _(x, n, grad):
  return grad * n
```

适合快速 prototype，但性能不如 C++（Python 调用开销）。生产用 C++ `TORCH_LIBRARY`。

### 8.2 native_functions.yaml vs 自定义

ATen 原生 op 在 `native_functions.yaml` 声明（PyTorch 源码内）；自定义 op 用 `TORCH_LIBRARY` 在扩展内声明。机制相同，只是声明位置不同。第三方库（FlashAttention/xformers/vLLM）用后者。

### 8.3 与 torch.export 的关系

`torch.export`（替代 torch.jit.trace）序列化模型为 graph，自定义 op 若用 TORCH_LIBRARY 注册会被当 graph node 记录（可跨进程/跨后端复用）。未注册的 Python 函数无法 export。详见 torch.export dev docs。

### 8.4 与 ABI 的关系

C++ 扩展编译需与 PyTorch 二进制 ABI 兼容（见 [[PyBind11与CMake]]），否则注册符号找不到。用 PyTorch 提供的 CMake helper（`TORCH_LIBRARY` 依赖的头文件/库）保证 ABI。

### 8.5 内容来源

TORCH_LIBRARY 机制整理自 PyTorch `torch/library.h`、dev docs "Custom C++ and CUDA Extensions" 与 "Registering a Dispatch Key"、以及 torch.export/torch.compile 对自定义 op 的兼容性文档。schema 语言见 TorchScript type reference。

---
相关: [[Custom Operator]] | [[Dispatcher]] | [[ATen与c10]] | [[Autograd Engine]] | [[PyBind11与CMake]] | [[torch.compile]] | [[TorchDynamo]] | [[Inductor]] | [[graph break]] | [[Triton kernel开发]] | [[CUTLASS与GEMM]] | [[FlashAttention]] | [[fused kernel]]
