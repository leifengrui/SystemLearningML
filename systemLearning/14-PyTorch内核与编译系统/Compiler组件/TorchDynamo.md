# TorchDynamo

> **所属章节**: [[Compiler组件]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: TorchDynamo / Dynamo / dynamo / Python 字节码追踪 / guard / frame analysis / FX graph / PEP 657
> **难度**: 高（需懂 Python 字节码 + [[torch.compile]] + [[Dispatcher]]）


## 1. 一句话定义

**TorchDynamo（Dynamo）** 是 `torch.compile` 的**第一段**——在 **Python 字节码层**追踪用户函数的执行：用 CPython 的 **PEP 657（frame evaluation hook）** 拦截函数帧，逐字节码指令分析，把支持的 ATen op 调用累积成 **FX Graph**（PyTorch 的 IR），同时为图建立 **guard**（输入 shape/dtype/值 的断言）。下次调用时先查 guard：全过则复用已编译图（μs 级），任一失败则重新 trace。遇不支持的 Python 结构（副作用函数、不支持的 builtin、data-dependent control flow）触发 **[[graph break]]**：把当前累积图交下游编译，剩余回 eager 执行，下个可 trace 处再开新图。Dynamo 是 compile 流水线的**入口与缓存守门员**，把"任意 Python 函数"变成"可编译的静态图 + 安全回退"。它取代了 1.x 的 `torch.jit.trace`（需手动 trace、不支持控制流）。

> [!note] 三句话定位
> - **是什么**：Python 字节码追踪器，用 PEP 657 frame hook 拦截帧，累积 ATen op 成 FX Graph + 建 guard。
> - **为什么**：要在 Python 层（用户写任意函数）与编译层（需静态图）之间桥接——Dynamo 自动 trace 而不需手写，guard 提供安全复用/重 trace。
> - **与 [[AOTAutograd]]/[[Inductor]] 关系**：Dynamo 出 FX Graph（前向 op 级）→ AOTAutograd 加反向 → Inductor codegen。三段流水。


## 2. 为什么需要它（动机与背景）

### 2.1 用户写任意 Python，编译需静态图

用户在 `def f(x): return torch.relu(x @ w + b)` 写任意 Python（含 if/print/循环/第三方库）。编译器（[[Inductor]]）要静态图才能融合/codegen。手动 trace（旧 `torch.jit.trace`）限制大、不支持控制流、副作用难处理。Dynamo 用**字节码追踪**自动把任意 Python 函数 trace 成图，且对不支持的**安全回退**（graph break → eager），不需用户改代码。

### 2.2 PEP 657 frame hook

CPython 的 frame evaluation hook 允许在函数帧执行前拦截。Dynamo 注册 hook，每次函数被调用时，hook 接管帧，逐字节码分析（`LOAD_FAST`/`CALL_FUNCTION`/`BINARY_MULTIPLY`...），识别 ATen op 调用（如 `torch.relu`）累积进 FX Graph，识别纯 Python 计算（如 `a + b` 其中 a,b 是 int）内联求值，识别副作用（print/IO）触发 graph break。是**在字节码层而非 AST 层**追踪——因为 AST 不含运行时类型/值，字节码追踪能拿到 tensor 的真实 shape/dtype 建 guard。

### 2.3 guard 的复用机制

首次 trace 生成图 + guard（如 `x.shape == [4,4]`、`x.dtype == float32`、`x.requires_grad == True`、`w.is_contiguous()`）。下次调用先查 guard：全过 → 复用图（μs 级）；任一失败 → 重 trace（shape 变了/属性变了）。guard 让 compile 安全缓存：稳定输入复用，变化重 trace。是 [[dynamic shape]] 问题的来源。

### 2.4 graph break 的安全回退

Dynamo 遇不支持的 Python 结构不崩溃，而是 graph break：把当前累积图交下游编译，剩余代码回 eager 执行，下个可 trace 处再开新图。多处 break → 多段编译 + 中间 eager。这保证 `torch.compile` 在任意 Python 上"尽量编译、不能则回退"，不会因一处不支持就整体失败。详见 [[graph break]]。

### 2.5 取代 torch.jit.trace

`torch.jit.trace` 需手动调、只 trace 一次（不支持变 shape/控制流）、副作用难处理。Dynamo 自动装饰、guard 支持变 shape 重 trace、graph break 安全回退。是 2.0+ 主线。


## 3. 核心概念详解

### 3.1 frame evaluation hook（PEP 657）

CPython 的 frame hook：Dynamo 注册后，每次 Python 函数被调用，CPython 在执行帧前调 Dynamo 的 callback。callback 决定：trace（接管执行，逐字节码分析）、或跳过（让 CPython 正常执行）。这是 Dynamo 能"透明"拦截任意函数的底层机制。

### 3.2 字节码追踪

Dynamo 不看 AST（静态语法树），而是**执行时逐字节码指令分析**：

```
LOAD_FAST x          # 加载 x (tensor)
LOAD_GLOBAL relu     # 加载 torch.relu
CALL_FUNCTION 1      # 调用
STORE_FAST y         # y = relu(x)
```

Dynamo 把 `torch.relu(x)` 识别为 ATen op，加进 FX Graph（`graph.call_function(torch.ops.aten.relu.default, args=(x,))`）。纯 Python 计算（`int + int`）内联求值不进图。副作用（print）触发 graph break。

### 3.3 FX Graph

Dynamo 输出 FX Graph（PyTorch 的 IR）：节点 = `torch.ops.aten.*` 调用，边 = tensor 数据流。是 [[AOTAutograd]] 与 [[Inductor]] 的输入。FX Graph 是可分析/变换的（可加反向、可融合、可 codegen）。

### 3.4 guard

guard 是图复用的条件断言，分几类：

| guard 类型 | 示例 | 失败触发 |
|---|---|---|
| **shape** | `x.shape == [4,4]` | shape 变 → 重 trace（[[dynamic shape]]） |
| **dtype** | `x.dtype == float32` | dtype 变 → 重 trace |
| **属性** | `x.requires_grad == True` | 属性变 → 重 trace |
| **值** | `x.item() > 0`（data-dependent） | 值变 → 重 trace |
| **contiguous** | `x.is_contiguous()` | 布局变 → 重 trace |

guard 在首次 trace 时记录输入的"指纹"，下次比对。全过则复用图（跳过 trace 与下游编译，μs 级）。

### 3.5 guard 失败与重 trace

shape 变（如 batch 64→128）→ shape guard 失败 → Dynamo 重 trace 该函数（新 shape），生成新图 + 新 guard，下游 [[Inductor]] 重新 codegen（新 shape 的 kernel）。多个 shape → 多个编译图（缓存膨胀）。`dynamic=True` 让 guard 放宽（shape 用 symbolic 变量），减少重编译（见 [[dynamic shape]]）。

### 3.6 graph break

Dynamo 遇以下触发 graph break（见 [[graph break]]）：
- 副作用函数（print/IO/log）。
- 不支持的 builtin 或第三方库（如 numpy 操作）。
- data-dependent control flow（`if x.item() > 0`，x 是 tensor，运行时才知分支）。
- 动态 shape 的 `.reshape(dynamic)` 某些场景。

break 时：当前累积图交下游编译，break 点回 eager 执行，下个可 trace 处开新图。多段图 + 中间 eager。`fullgraph=True` 强制不 break（有 break 报错，用于调试）。

### 3.7 guard 与 eager 的关系

Dynamo 仍是 eager 模式的扩展：trace 用的 op 经 [[Dispatcher]] 跑（真算 + 建 autograd 图），只是同时记录进 FX Graph。故 Dynamo trace 时**真算**（不是 symbolic 推导）。symbolic 推导（不真算）在 [[AOTAutograd]]/Meta backend。Dynamo 是"边算边记"。

### 3.8 与 AOTAutograd/Inductor 的流水

```
用户函数 → Dynamo (trace + guard) → FX Graph (前向 op 级)
         → AOTAutograd (加反向, joint 分解) → FX Graph (前向+反向)
         → Inductor (融合 + codegen) → Triton/C++ kernel
```

Dynamo 只出前向 op 级图。反向由 AOTAutograd 加（它读 FX Graph + autograd 公式生成反向图）。融合/codegen 由 Inductor。三段解耦，Dynamo 是入口。

### 3.9 TorchDynamo 的不对称设计

Dynamo trace 时调 op 真算（eager），但 trace 出的图后续复用是**编译后的 fused kernel**（不走 eager）。故首次调用慢（trace + 编译），后续快（编译图）。这种"首次 trace 真算 + 后续编译图"是不对称的，是 compile 的开销模型基础（见 [[torch.compile]] 4.2）。


## 4. 数学原理 / 公式

### 4.1 guard 的数量与重编译概率

设 guard 数 $G$，每 guard 失败概率 $p_i$（独立假设）。guard 全过概率 $\prod_i (1-p_i)$。shape 频繁变（$p_i$ 高）→ 复用率低 → 反复 trace + 编译。`dynamic=True` 把 shape guard 移除（$G$ 减）→ 复用率升。

### 4.2 trace 开销

trace 开销 $\propto$ 字节码数（逐指令分析）。大函数 trace 慢。但只在首次或 guard 失败时 trace，后续复用图（μs 级 lookup）。

### 4.3 graph break 的收益损失

$n$ 段 graph break → $n$ 段编译 + $n-1$ 段中间 eager。中间 eager 的 op 不融合、有 launch 开销。break 多 → 收益打折。`fullgraph=True` 调试确保 0 break（生产应尽量）。

### 4.4 缓存膨胀

$S$ 个不同 shape → $S$ 个编译图（每 shape 一份 Inductor codegen 缓存）。缓存 $\propto S \cdot \text{graph size}$。shape 频繁变 → 缓存膨胀 + 反复编译开销。`dynamic=True` 减 $S$（通用 kernel 跨 shape）。


## 5. 代码示例（可选）

### 5.1 Dynamo 的基本行为

```python
import torch
@torch.compile
def f(x, w):
  y = x @ w          # ATen op, 进图
  z = torch.relu(y)  # ATen op, 进图
  return z

x = torch.randn(4, 4); w = torch.randn(4, 4)
# 首次: Dynamo trace (字节码分析) -> FX Graph -> AOTAutograd -> Inductor codegen
out = f(x, w)   # 慢 (编译)
# 第二次: guard 查 (x.shape==[4,4] 等) 全过 -> 复用编译图
out2 = f(x, w)  # 快 (μs lookup + fused kernel)
# shape 变: guard shape 失败 -> 重 trace + 重编译
out3 = f(torch.randn(8, 4), w)  # 慢 (重编译)
```

### 5.2 观察 FX Graph

```python
# Dynamo 的 explained graph 可导出
import torch._dynamo
def f(x): return torch.relu(x * 2)
code, graph = torch._dynamo.export(f)(torch.randn(4))
print(graph.graph)   # FX Graph: 节点是 aten.mul / aten.relu
```

### 5.3 graph break 观察

```python
@torch.compile
def f(x):
  y = x * 2
  print("debug", y.sum().item())   # 副作用 + data-dependent -> graph break
  z = y + 1
  return z
# Dynamo 警告: graph break at print, 2 graphs compiled
# 用 TORCH_COMPILE_DEBUG=1 看详细 break 原因
```


## 6. 与其他知识点的关系

- **上游（依赖）**: CPython PEP 657 frame hook、Python 字节码、[[Dispatcher]]（trace 时 op 经 dispatcher 跑）、FX Graph IR。
- **下游（应用）**: [[torch.compile]]（Dynamo 是其第一段）、[[AOTAutograd]]（接 Dynamo 的 FX Graph 加反向）、[[Inductor]]（接 AOTAutograd 的图 codegen）、[[graph break]]（Dynamo 的回退机制）、[[dynamic shape]]（guard 失败的重 trace）、[[Custom C++ CUDA Operator]]（注册的 op 才被 Dynamo 识别）。
- **对比 / 易混**:
  - **Dynamo vs `torch.jit.trace`**：Dynamo 字节码追踪 + guard + graph break（自动、安全回退）；jit.trace 一次 trace 不支持控制流/变 shape。Dynamo 是 2.0+ 主线。
  - **Dynamo vs [[AOTAutograd]]**：Dynamo 出前向 op 级图；AOTAutograd 加反向 + joint 分解。Dynamo 是入口，AOTAutograd 是中段。
  - **Dynamo vs [[Dispatcher]]**：Dispatcher 是 op 调用路由（每 op 一次）；Dynamo 是 Python 函数级 trace（把多 op 串成图）。Dynamo trace 时调 op 经 dispatcher。
  - **Dynamo trace vs symbolic 推导**：Dynamo trace 时真算（eager）；Meta backend/AOTAutograd 的 symbolic 是不真算推 shape。Dynamo 是"边算边记"。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 Dynamo 是 symbolic 推导
> Dynamo trace 时 op 真算（eager，经 dispatcher），只是同时记进 FX Graph。symbolic 推导（不真算）是 Meta backend/AOTAutograd 的事。Dynamo 是"边算边记"。

> [!warning] 误区 2：忽视 guard 的重编译
> shape/dtype/属性变 → guard 失败 → 重 trace + 重编译。变长 batch 不设 `dynamic=True` 反复编译，可能负收益。见 [[dynamic shape]]。

> [!warning] 误区 3：以为 graph break 报错
> graph break 是**安全回退**（回 eager），不是报错。多处 break → 多段编译 + 中间 eager，收益打折。`fullgraph=True` 才在有 break 时报错（调试用）。

> [!warning] 误区 4：Dynamo 不认识自定义 op
> 用 pybind11 绑的 Python kernel，Dynamo 不认识 → graph break。必须用 `TORCH_LIBRARY` 注册（[[Custom C++ CUDA Operator]]）才被 Dynamo trace。

> [!warning] 误区 5：忽视首次 trace 的"真算"
> Dynamo trace 时真算 op（eager），故首次调用既慢（trace + 编译）又真跑了一遍。不是 symbolic 推导。

> [!warning] 误区 6：混淆 FX Graph 与 autograd 图
> Dynamo 出的 FX Graph 是前向 op 级 IR；autograd 图（grad_fn DAG）是 [[Dispatcher]] Autograd 层建的运行时图。两者不同层次。AOTAutograd 把 FX Graph + autograd 公式合成"前向+反向 joint graph"。

> [!warning] 误区 7：guard 全过就一定数值一致
> guard 只保证输入 shape/dtype/属性一致，不保证内核数值一致。编译图融合改浮点顺序有 $O(\epsilon)$ 差异（[[训推不一致]]）。需严格复现用确定性算法。


## 8. 延伸细节

### 8.1 Dynamo 的可配置

`torch._dynamo.config` 可调：`cache_size_limit`（每函数缓存图数上限）、`accumulated_cache_size_limit`、`suppress_errors`（trace 失败是否回退 eager 还是报错）、`verbose`。生产调 cache_size_limit 避免变 shape 反复编译撑爆缓存。

### 8.2 Dynamo 的 backend 选择

`torch.compile(backend=...)` 的 backend 在 Dynamo 之后接：`inductor`（默认）、`aot_eager`（AOTAutograd 不 codegen，调试）、`cudagraphs`（只 CUDA Graph）、自定义。Dynamo 出 FX Graph，backend 决定怎么处理。

### 8.3 与 torch.export 的关系

`torch.export`（AOT 序列化）不依赖运行时 trace，用 Dynamo 的 symbolic tracing（不真算）+ `torch.library.opcheck` 验证，导出可移植 graph。与运行时 Dynamo（边算边记）互补。详见 torch.export dev docs。

### 8.4 Dynamo 的递归

Dynamo trace 时若调另一被 compile 的函数，会递归 trace（嵌套图）。但跨函数的 graph break 传播要小心（内层 break 致外层也 break）。`nopython=True`/`fullgraph=True` 强制全图避免。

### 8.5 内容来源

Dynamo 整理自 PyTorch dev docs "TorchDynamo"、PEP 657（frame evaluation hook）、`torch/_dynamo/` 源码与 design doc。guard/graph break 见 [[graph break]] 与 [[dynamic shape]]。

---
相关: [[Compiler组件]] | [[torch.compile]] | [[AOTAutograd]] | [[Inductor]] | [[graph break]] | [[dynamic shape]] | [[Dispatcher]] | [[Custom C++ CUDA Operator]] | [[计算图]] | [[训推不一致]] | [[torch profiler]]
