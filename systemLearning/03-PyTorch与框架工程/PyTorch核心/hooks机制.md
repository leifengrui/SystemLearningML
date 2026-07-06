# hooks机制

> **所属章节**: [[PyTorch核心]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 中等（调试与中间件式改造的核心）

## 1. 一句话定义

**Hooks（钩子）** 是 [[nn.Module]] 与 autograd 暴露的**"在某阶段自动触发的用户回调"**接口——你注册一个函数，它在前向/反向的固定时机被调用，拿到该层的输入/输出/梯度，便于**调试、可视化、特征提取、梯度监控、自定义正则、hook 式注入逻辑**，而不必改动模型本身的 [[forward与backward]] 代码。

> [!note] 名字辨析
> PyTorch 里 "hook" 至少分两层：**Module hooks**（挂在 `nn.Module` 上，按层粒度，前向/反向触发）与 **Tensor hooks / autograd hooks**（挂在单个 `Tensor.grad_fn` 上，反向时触发）。两者 API 不同、粒度不同，别混用。

## 2. 为什么需要它（动机与背景）

很多需求是"想干预训练但不想改模型源码"：

- **调试**：想知道第 50 层激活有没有爆炸/消失、梯度流向对不对——总不能在 `forward` 里到处 `print`。
- **可视化**：把中间特征存下来画分布、画 attention map。
- **监控**：每 N 步对每层梯度范数做日志，定位哪层炸了。
- **自定义正则**：对某层激活加 L1、对某层梯度加噪声、做梯度惩罚（WGAN-GP）。
- **修改流**：做 early-exit、动态路由、prompt 注入。

若每改一次都要侵入模型代码，会污染 [[nn.Module]] 的 forward。Hooks 提供**非侵入式中间件**：`register_xxx_hook(fn)` 一行挂上，`remove_hook()` 一行卸下，模型代码保持干净。

## 3. 核心概念详解

### 3.1 四种 Module hook

| Hook | 触发时机 | 回调签名 | 用途 |
|---|---|---|---|
| `register_forward_pre_hook` | `forward` 之前 | `fn(module, input) -> None\|modified_input` | 改输入、early-exit、打日志 |
| `register_forward_hook` | `forward` 之后 | `fn(module, input, output) -> None\|modified_output` | 改输出、抓特征、可视化 |
| `register_full_backward_hook` | 反向时（该层输出对输入的梯度算完） | `fn(module, grad_input, grad_output) -> None\|tuple` | 查/改反向梯度流向 |
| `register_full_backward_pre_hook` | 反向算该层之前 | `fn(module, grad_output) -> None\|tuple` | 改进入该层的上游梯度 |

> [!warning] 旧 `register_backward_hook` 已废弃
> 旧版 `register_backward_hook` 语义不精确（对部分算子漏触发），PyTorch 推荐改用 `register_full_backward_hook` / `register_full_backward_pre_hook`。

### 3.2 触发条件：必须走 `__call__`

hooks 只在 `model(x)`（即 `nn.Module.__call__`）里挂触发逻辑。**直接 `model.forward(x)` 不会触发 forward hook**。这也是为什么养成"永远 `model(x)`"习惯的又一理由（见 [[nn.Module]] §8.1）。

### 3.3 Tensor / autograd hook

```python
y.register_hook(lambda g: g * 2)   # 反向时把流经 y 的梯度乘2
```

- 粒度是**单个非叶子张量**，反向时在其 `grad_fn` 触发。
- 常用于：手动改某中间梯度的流向、实现 stop-gradient（用 `.detach()` 更干净）、梯度调试。
- 缺点：只能改"流入该张量的梯度"，且 `y.grad` 不会被它填（hook 只在反向路径上改 grad，叶子 `.grad` 仍由 autograd 填）。

### 3.4 全局 hook：`register_forward_hook` 遍历 vs `nn.Module` 全局 API

要对所有层统一挂 hook，可用循环：

```python
handles = [m.register_forward_hook(hook_fn) for m in model.modules()]
```

PyTorch 还提供 `torch.nn.modules.module.register_module_forward_hook` 等全局注册 API（全局触发，慎用，影响所有 Module）。

### 3.5 返回值改输入/输出

forward hook 返回非 None 时会**替换该层输出**；pre_hook 返回非 None 时替换输入。这是"hook 式改模型行为"的关键，但不透明，调试完要记得卸。

### 3.6 句柄与清理

每个 `register_*` 返回一个 `RemovableHandle`，`handle.remove()` 卸载。不卸会**内存泄漏**（hook 闭包常持有大张量引用）+ 触发意外行为。推荐用 `with` 或显式清理。

## 4. 数学原理 / 公式

hooks 不引入新数学，但 backward hook 的语义可形式化。设层 $f$ 输入 $x$、输出 $y=f(x)$，损失 $\mathcal{L}$。反向到该层时：

- `grad_output` $= \partial\mathcal{L}/\partial y$（从下游传上来）；
- `grad_input` $= \partial\mathcal{L}/\partial x = J_f(x)^\top \cdot \text{grad\_output}$（链式法则，autograd 已算好）。

backward hook 拿到的就是这两个，返回新元组可改写它们再传给上下游。等价于手工插入一个"梯度变换算子"在链式链上。见 [[梯度与Jacobian]]、[[forward与backward]] §4。

## 5. 代码示例

```python
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,1))
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

# 1) forward hook：抓每层输出 shape 与激活范数
stats = {}
def fwd_hook(module, inp, out):
    stats[module.__class__.__name__] = (tuple(out.shape), float(out.norm()))
handles = [m.register_forward_hook(fwd_hook) for m in model.modules() if isinstance(m, nn.Linear)]

x = torch.randn(4,8); y = torch.randn(4,1)
out = model(x)                  # 走 __call__，hook 才触发
print(stats)                    # {'Linear': ((4,16), ...), ...}
for h in handles: h.remove()    # 调试完务必卸

# 2) full backward hook：监控梯度范数，定位哪层爆炸
grad_stats = {}
def bwd_hook(module, grad_input, grad_output):
    grad_stats[module.__class__.__name__] = float(grad_output[0].norm())
handles = [m.register_full_backward_hook(bwd_hook) for m in model.modules() if isinstance(m, nn.Linear)]

opt.zero_grad(); loss_fn(model(x), y).backward()
print(grad_stats)
for h in handles: h.remove()

# 3) forward pre hook：改输入（演示注入噪声）
def pre_hook(module, inp):
    return (inp[0] + 0.01*torch.randn_like(inp[0]),)   # 返回元组替换输入
h = model[0].register_forward_pre_hook(pre_hook)
out2 = model(x); h.remove()

# 4) Tensor hook：手改某中间梯度
y = model(x).sum()
y.register_hook(lambda g: g*2)   # 反向时把流入 y 的梯度乘2
y.backward()

# 5) 句柄上下文式清理（推荐）
from contextlib import contextmanager
@contextmanager
def with_hooks(model, fn):
    hs = [m.register_forward_hook(fn) for m in model.modules()]
    try: yield
    finally:
        for h in hs: h.remove()
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[nn.Module]]（hook 是 Module 的方法）、[[forward与backward]]（hook 挂在这两阶段）、[[autograd机制]]（Tensor hook 依赖反向图）。
- **下游（应用）**: 调试/可视化（torch profiler 底层也用 hook 思路）、WGAN-GP 梯度惩罚（对中间输出求二阶梯度，需 `create_graph=True`）、[[梯度裁剪]]（可在 backward 后读 `.grad` 也可 hook）、[[tensorboard]]/特征图可视化、自定义正则注入。
- **对比 / 易混**:
  - Module hook vs Tensor hook：粒度与 API 不同，见 §3。
  - `forward_hook` vs `forward_pre_hook`：一个在 forward 后改输出，一个在前改输入。
  - `register_full_backward_hook` vs 废弃的 `register_backward_hook`：前者语义完整。
  - hook vs 直接改 forward：hook 非侵入但隐式；改 forward 显式但污染代码。调试用 hook，上线用 forward。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **用 `model.forward(x)` 后抱怨 hook 没触发** → 必须 `model(x)`。
> 2. **忘了 `handle.remove()`** → hook 闭包持有大张量 → 显存泄漏 + 意外触发。推荐用 `with` 或调试脚本里显式清理。
> 3. **forward hook 返回新张量** → 默默替换该层输出，下游拿到的不是 forward 真实结果，调试结论失真。不用就返回 None。
> 4. **用旧 `register_backward_hook`** → 部分算子漏触发；改用 `register_full_backward_hook`。
> 5. **以为 forward hook 在 `torch.no_grad()` 下也触发** → 触发（forward 本就触发）；但 backward hook 在 `no_grad` 下**不会**触发（无反向图）。
> 6. **Tensor hook 改梯度但叶子 `.grad` 没变** → Tensor hook 改的是流经该非叶子节点的梯度，传到叶子的链路仍由 autograd 决定；要改叶子梯度直接改 `p.grad` 或用 `p.register_hook`。
> 7. **在 hook 里做需要梯度的运算但没 `create_graph`** → 二阶梯度场景（WGAN-GP）必须 `backward(create_graph=True)`，否则 hook 内 `.backward()` 报图已释放。
> 8. **全局 hook 影响所有模型实例** → 调试时只挂在目标 model 的子模块上，别用全局 API。

## 8. 延伸细节

### 8.1 hook 与 `torch.nn.modules.module` 全局钩子

PyTorch 提供 `register_module_forward_hook` 等全局钩子，对所有 Module 实例生效。Profiler、FSDP 的参数收集等内部机制大量使用全局钩子，普通用户少用以免副作用。

### 8.2 用 hook 实现"梯度范数日志"标准范式

```python
def log_grad_norm(model, step, log_every=100):
    if step % log_every: return
    for n, p in model.named_parameters():
        if p.grad is not None:
            print(f"[{n}] grad_norm={p.grad.norm().item():.4f}")
```
> [!tip] 更稳的做法
> 直接遍历 `named_parameters` 读 `.grad` 比 backward hook 简单且无副作用；hook 优势在"按层"和"改梯度"场景。监控用前者，干预用后者。

### 8.3 hook 与 DDP/FSDP 的交互

DDP 在反向时 hook 会触发，但梯度已被 all-reduce（hook 看到的是同步后梯度）；FSDP 的参数分片使 forward hook 看到的可能是 unshard 后的全量参数，需注意时序。调试分布式训练时，hook 的输出可能与单卡不一致，要结合 [[rank与world size]] 与 [[NCCL backend]] 的同步点理解。

### 8.4 forward hook 抓特征用于可解释性

抓某层激活做 CAM/Grad-CAM、attention 可视化、激活分布直方图，是 hook 最经典的可解释性用法。生产里也可用它做"中间层蒸馏"（teacher 中间特征 → student）。

### 8.5 替代品：`torch.func.transform` / FX

PyTorch 2.x 的 `torch.fx` 图变换、`torch.func.vmap/grad` 函数式变换，能在更结构化的层面做"不改源码改行为"，适合编译/优化场景；hook 更适合临时、动态、调试场景。二者互补。

---
相关: [[PyTorch核心]]、[[nn.Module]]、[[forward与backward]]、[[autograd机制]]、[[梯度与Jacobian]]、[[梯度裁剪]]
