# graph break

> **所属章节**: [[图捕获问题]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: graph break / 图中断 / Dynamo 跳出点 / 副作用函数 / fullgraph / graph split / guard 失败
> **难度**: 中高（需懂 [[TorchDynamo]] + [[torch.compile]]）


## 1. 一句话定义

**graph break（图中断）** 是 [[TorchDynamo]] 在 trace 用户函数时，遇**无法静态编译的 Python 结构**（副作用函数如 print/IO、不支持的第三方库、data-dependent control flow 如 `if tensor.item() > 0`、某些动态 shape 操作）时，**把当前累积的 FX Graph 交下游编译、剩余代码回 eager 执行、下个可 trace 处再开新图**的回退机制。它保证 `torch.compile` 在任意 Python 上"尽量编译、不能则回退"而不崩溃——但代价是：多处 break → 多段编译 + 中间 eager，融合被打断、launch 开销回归，收益打折。`fullgraph=True` 可强制不 break（有 break 报错，用于调试确保全图）。graph break 是 compile 与用户代码的**兼容性边界**——理解哪些结构会 break、如何避免，是写 compile-friendly 代码的关键。

> [!note] 三句话定位
> - **是什么**：Dynamo 遇不支持结构时把当前图交编译、剩余回 eager、再开新图的回退机制。
> - **为什么**：用户写任意 Python（副作用/控制流/第三方库），编译器无法全部静态化；graph break 让 compile 不崩溃而尽量编译。
> - **代价**：break 打断融合 + 回归 launch，收益打折。用 `TORCH_LIBRARY` 注册 custom op、避免副作用、`fullgraph=True` 调试可减 break。


## 2. 为什么需要它（动机与背景）

### 2.1 compile 要兼容任意 Python

用户在 `@torch.compile` 的函数里写任意 Python：可能 `print` 调试、调 numpy、有 `if tensor.item() > 0` 的 data-dependent 分支、用未注册的第三方库。Dynamo 无法把这些全静态化成图（data-dependent 分支运行时才知走哪、副作用难分析、第三方库无 trace 支持）。若遇不支持就崩溃，compile 不可用。

### 2.2 安全回退而非崩溃

graph break 是 Dynamo 的**安全网**：遇不支持的结构，不报错崩溃，而是把当前已累积的图（前面可 trace 的部分）交下游 [[AOTAutograd]]/[[Inductor]] 编译，break 点回 eager 执行，下个可 trace 处再开新图。结果是"多段编译图 + 中间 eager 段"——尽量编译、不能则回退，用户代码不需改。

### 2.3 代价：收益打折

break 的代价：
- **融合被打断**：break 前后是独立编译图，跨 break 边界的 op 不融合（中间 eager 的 op 各自 HBM 往返）。
- **launch 回归**：中间 eager 段的 op 逐个 launch（无 [[CUDA Graph与graph capture]] 压缩）。
- **Python/dispatcher 开销回归**：eager 段走 Python + dispatcher。

多处 break → 收益严重打折，甚至接近 eager。故写 compile-friendly 代码（减 break）是关键。

### 2.4 fullgraph=True 强制全图

`@torch.compile(fullgraph=True)` 强制不 break——遇会 break 的结构直接报错（而非回退）。用于调试：确保你的代码 0 break，拿到最大收益。生产代码应尽量 fullgraph（先调试到 0 break）。

### 2.5 常见 break 源

理解哪些结构会 break，才能写 compile-friendly 代码（见第 3 节）。


## 3. 核心概念详解

### 3.1 触发 graph break 的常见结构

| break 源 | 示例 | 原因 | 避免方式 |
|---|---|---|---|
| **副作用函数** | `print(x)`、`logging.info(...)`、文件 IO | 副作用难静态化（每次执行都发生） | 移出函数或用 `torch._logging` |
| **data-dependent control flow** | `if x.item() > 0: ...` | 运行时才知分支，静态图无法表达 | 重写成 tensor 操作（`torch.where`）或 `dynamic=True` |
| **不支持的第三方库** | `numpy_op(x.numpy())` | 非 ATen op，Dynamo 不认识 | 用 `TORCH_LIBRARY` 注册或换 ATen 等价 op |
| **未注册的 custom op** | pybind11 绑的 kernel | Dynamo 不识别 → break | 用 `TORCH_LIBRARY` 注册（[[Custom C++ CUDA Operator]]） |
| **某些动态 shape** | `x.reshape(unknown_dim)` | shape 运行时才知 | `dynamic=True` 或固定 shape |
| **Python 反射/元编程** | `getattr(obj, name)` 动态 | 静态分析难 | 用显式调用 |
| **不支持的 builtin** | 某些 Python builtin | Dynamo 未覆盖 | 换等价 ATen |

### 3.2 break 的执行流程

```
用户函数 f(x):
  op1 (进图)
  op2 (进图)
  print(...)        # 副作用 -> graph break!
    → 当前图 {op1, op2} 交 AOTAutograd/Inductor 编译
    → print(...) 回 eager 执行
  op3 (新图开始)
  op4 (进新图)
  return op4
```

结果：两段编译图 + 中间 eager（print）。两段图各自融合，但 op2→op3 跨 break 不融合。

### 3.3 break 的检测与调试

```python
@torch.compile
def f(x):
  y = x * 2
  print(y.sum().item())   # break
  z = y + 1
  return z
f(torch.randn(4))
# 警告: [WARNING] graph break at print, 2 graphs compiled
# TORCH_COMPILE_DEBUG=1 看详细 break 原因与位置
```

`TORCH_COMPILE_DEBUG=1` 环境变量让 Dynamo 打印详细 break 日志（哪行、什么结构、为什么）。是定位 break 的工具。

### 3.4 fullgraph=True

```python
@torch.compile(fullgraph=True)
def f(x):
  y = x * 2
  print(y.sum().item())   # break -> 报错 (不回退)
  ...
f(torch.randn(4))  # 报错: graph break at print
```

强制 0 break，有 break 报错。用于调试：逐个消除 break 源，直到 fullgraph 不报错。

### 3.5 break 对融合的影响

```
无 break: op1-op2-op3-op4 全融合 (1 kernel)
有 break (op2 后): {op1,op2} 融合 + {op3,op4} 融合 (2 kernel, op2->op3 中间 HBM 往返)
```

跨 break 边界的中间结果（op2 输出 → op3 输入）落 HBM（不融合则不寄存器传）。$I$ 降，[[Roofline模型]] 的 memory-bound 回归。多处 break → 多段，收益线性打折。

### 3.6 break 对 CUDA Graph 的影响

`reduce-overhead` 模式叠加 [[CUDA Graph与graph capture]]，但 CUDA Graph 要求**整段连续 kernel 序列**。graph break 把序列打断 → CUDA Graph 只能捕获 break 之间的段，不能整段。launch 压缩收益打折。故 reduce-overhead 模式需尽量 fullgraph。

### 3.7 常见 break 源的避免

| break 源 | 避免 |
|---|---|
| print/log | 移出 compile 函数，或用 `torch._logging`（compile-aware） |
| `if x.item() > 0` | 重写成 `torch.where(cond, a, b)`（tensor 操作，静态） |
| numpy 操作 | 换 ATen 等价 op（`torch.sum` 代替 `np.sum`） |
| custom kernel (pybind11) | 用 `TORCH_LIBRARY` 注册（[[Custom C++ CUDA Operator]]） |
| `.item()` 拉值到 Python | 避免在 compile 函数内 `.item()`（移出或用 `torch.where`） |

### 3.8 guard 失败 ≠ graph break

- **guard 失败**：shape/dtype 变，Dynamo 重 trace（重新 trace 同函数，非回退 eager）。是 [[dynamic shape]] 问题。
- **graph break**：遇不支持结构，回退 eager 执行。是兼容性问题。

两者都让 compile 收益打折，但机制不同。guard 失败可重 trace 恢复（同代码不同 shape），graph break 是结构性的（代码本身有不支持结构）。


## 4. 数学原理 / 公式

### 4.1 break 数与收益损失

设函数有 $k$ 个 op，$b$ 处 break → $b+1$ 段编译图。无 break 融合 $k$ op 成 1 kernel（byte $\sim 2N$）。$b$ 处 break：每段独立融合，段间 $b$ 次 HBM 往返（$\sim 2bN$ byte）。byte $\sim 2N + 2bN$。$b$ 大时接近 eager（$2kN$）。

### 4.2 break 与 launch

无 break + CUDA Graph：launch $\sim 1$ 次。$b$ 处 break：CUDA Graph 只能捕获段内，段间 eager launch $\sim b \cdot (\text{段内 op 数}) \cdot 5$ μs。launch 收益打折。

### 4.3 fullgraph 的收益

fullgraph（$b=0$）：最大融合 + 整段 CUDA Graph。byte $\sim 2N$，launch $\sim 1$。是 compile 的理想态。生产代码应调试到 fullgraph。


## 5. 代码示例（可选）

### 5.1 触发与观察 break

```python
import torch
@torch.compile
def f(x):
  y = x * 2
  print("debug", y.sum().item())   # break: 副作用 + .item()
  z = y + 1
  return z
# 警告: graph break, 2 graphs compiled
f(torch.randn(4))
```

### 5.2 消除 break

```python
# 原 (break)
@torch.compile
def f(x, cond):
  y = x * 2
  if cond.item() > 0:      # data-dependent control flow -> break
    y = y + 1
  return y

# 改 (无 break)
@torch.compile(fullgraph=True)
def f(x, cond):
  y = x * 2
  y = torch.where(cond > 0, y + 1, y)   # tensor 操作, 静态
  return y
f(torch.randn(4), torch.tensor([1.0]))  # 无 break
```

### 5.3 调试 break

```python
# 环境变量: TORCH_COMPILE_DEBUG=1 python script.py
# 输出: 每个 break 的行号、结构、原因
# 或:
import torch._dynamo
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False   # break 报错而非回退 (调试)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[TorchDynamo]]（break 是 Dynamo 的机制）、Python 字节码（Dynamo trace 的对象）。
- **下游（应用）**: [[torch.compile]]（break 影响收益）、[[Inductor]]（break 间的段各自 codegen）、[[CUDA Graph与graph capture]]（break 打断整段 graph 捕获）、[[Custom C++ CUDA Operator]]（注册避免 break）、[[dynamic shape]]（guard 失败 ≠ break）。
- **对比 / 易混**:
  - **graph break vs guard 失败**：见 3.8，break 是结构不支持回退 eager；guard 失败是 shape 变重 trace（同代码不同 shape）。
  - **graph break vs eager**：eager 是全程不编译；graph break 是部分编译 + 部分 eager。有 break 的 compile 仍优于纯 eager（段内融合）。
  - **fullgraph=True vs default**：fullgraph 强制 0 break（有 break 报错）；default 遇 break 回退。生产调试到 fullgraph 后用 default（防新 break 静默回退）。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 break 报错
> break 是**安全回退**（回 eager），不报错。`fullgraph=True` 才报错。default 模式遇 break 静默回退，可能不知收益打折。用 `TORCH_COMPILE_DEBUG=1` 或 fullgraph 调试。

> [!warning] 误区 2：忽视 `.item()` 的 break
> `x.item()` 把 tensor 拉成 Python 标量，是 data-dependent（运行时才知值），触发 break。compile 函数内避免 `.item()`，用 `torch.where` 等 tensor 操作替代。

> [!warning] 误区 3：用 pybind11 绑的 kernel 不 break
> pybind11 绑的 Python 函数，Dynamo 不认识 → break。必须用 `TORCH_LIBRARY` 注册（[[Custom C++ CUDA Operator]]）才被 Dynamo trace。

> [!warning] 误区 4：break 影响小
> 单个 break 把融合分两段，跨边界的 op 不融合 + launch 回归。多处 break 收益严重打折，可能接近 eager。生产应尽量 fullgraph。

> [!warning] 误区 5：fullgraph 报错就放弃 compile
> fullgraph 报错是调试信号——逐个消除 break 源（移 print、重写控制流、注册 op）。消除后用 default 模式（防新 break 静默回退）拿收益。

> [!warning] 误区 6：混淆 break 与 [[dynamic shape]] 重编译
> break 是代码结构不支持（回退 eager）；dynamic shape 重编译是 shape 变 guard 失败（重 trace 同代码）。前者改代码消除，后者用 `dynamic=True` 减重编译。

> [!warning] 误区 7：忽视 break 对 CUDA Graph 的影响
> reduce-overhead 模式叠加 CUDA Graph，但 break 打断整段序列 → graph 只能捕获段内。要 reduce-overhead 最大收益需 fullgraph（无 break）。


## 8. 延伸细节

### 8.1 Dynamo 的 break 源清单

`torch/_dynamo/` 维护"会 break 的结构"清单（unsupported Python builtins、副作用函数、data-dependent control flow 等）。随版本演进，Dynamo 覆盖更多结构（break 减少）。某些之前 break 的现在可 trace（如部分控制流用 symbolic）。

### 8.2 symbolic control flow

PyTorch 实验 `torch.cond`/`torch.while_loop`（symbolic 控制流）让 data-dependent control flow 在编译期用 symbolic 表达（不 break）。是减少 break 的方向。详见 torch.export dev docs。

### 8.3 与 torch.export 的关系

`torch.export`（AOT 序列化）要求 fullgraph（export 的图必须完整无 break）。故 export 前需调试到 0 break。运行时 compile 的 default 模式可容忍 break（回退），但 export 不容忍。

### 8.4 break 的 perf 诊断

`torch.profiler` 的 compile view 可看每段编译图的融合情况、break 位置。是定位 break 影响的工具。详见 [[torch profiler]]。

### 8.5 内容来源

graph break 整理自 PyTorch dev docs "TorchDynamo graph breaks"、`torch/_dynamo/` 源码、`TORCH_COMPILE_DEBUG` 输出格式。常见 break 源见 Dynamo unsupported list。

---
相关: [[图捕获问题]] | [[TorchDynamo]] | [[torch.compile]] | [[Inductor]] | [[dynamic shape]] | [[CUDA Graph与graph capture]] | [[Custom C++ CUDA Operator]] | [[torch profiler]] | [[fused kernel]] | [[kernel launch overhead]]
