# nn.Module

> **所属章节**: [[PyTorch核心]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 基础（入门必学）

## 1. 一句话定义

**`nn.Module`** 是 PyTorch 中**所有神经网络层/模型的基类**——任何自定义模型、层（Linear、Conv2d、TransformerBlock…）都继承它；它负责统一管理**子模块（submodules）、可训练参数（Parameter）、forward 调用、Hooks、状态字典（state_dict）、设备搬运、训练/评估模式切换**等一切"模型骨架"事务，用户只需重写 `__init__`（组装零件）和 `forward`（定义数据流）即可。

> [!note] 名字辨析
> `nn.Module` 中的 "Module" 是**广义模块**：一个 `Linear` 是 Module，一个 `TransformerBlock` 是 Module，整个 `GPT` 也是 Module。它**不等于"一层"**，而是"任意可组合的计算单元"。整个模型 = 一棵 Module 树（根 Module 持有若干子 Module，子 Module 再持有更细的子 Module，叶子是 Parameter/Buffer）。

## 2. 为什么需要它（动机与背景）

没有 `nn.Module` 之前，你得手写每个层：自己维护 `W`、`b` 张量、自己写 `forward`、自己遍历所有参数交给优化器、自己实现 `.to(device)`、自己实现保存/加载、自己实现 `.eval()`/`.train()`。一旦模型变深（几十层、上百个子层），这套样板代码会爆炸式重复。

`nn.Module` 把这套**通用机制**抽到基类里：

- 自动注册子模块与参数 → `model.parameters()` 一行拿到所有可训练张量；
- 自动 `.to(device)` 递归搬到 GPU；
- 自动 `state_dict()` 序列化所有参数 → 保存/加载统一；
- 自动 `.train()/.eval()` 切换 Dropout/BN 行为；
- 自动 hooks 挂载点 → 调试/特征提取。

用户因此只需关心**两件事**：`__init__` 里搭什么零件、`forward` 里数据怎么走。这是 PyTorch"显式但简洁"哲学的核心。

## 3. 核心概念详解

### 3.1 三大组成：Parameter / Buffer / 子 Module

一个 Module 内部可挂三类对象：

| 类别            | 类型                                                  | 是否被 `parameters()` 收集 | 是否进 `state_dict` | 典型例子                                       |
| ------------- | --------------------------------------------------- | --------------------- | ---------------- | ------------------------------------------ |
| **Parameter** | `nn.Parameter`（`Tensor` 子类，`requires_grad=True` 默认） | ✅                     | ✅                | `Linear.weight`、`LayerNorm.weight`         |
| **Buffer**    | 普通 `Tensor`，经 `self.register_buffer(name, t)` 注册    | ❌                     | ✅                | `BatchNorm.running_mean`、RoPE 的 `inv_freq` |
| **子 Module**  | 另一个 `nn.Module`                                     | （递归收集其参数）             | （递归进 state_dict） | `self.l1 = nn.Linear(...)`                 |

> [!tip] 何时用 Buffer？
> 既**不可训练**（不进优化器）又**要随模型保存/搬设备**的张量，必须用 `register_buffer`。例如 BatchNorm 的 `running_mean`（推理要用但不学习）、位置编码的常量、注意力 mask 模板。若只赋值 `self.x = torch.tensor(...)`（没注册），`.to(cuda)` 不会搬它、`state_dict` 不存它，多卡会出错。

### 3.2 `__init__` + `forward` 二段式

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__().____()           # 必须先调父类初始化，否则注册机制不工作
        self.l1 = nn.Linear(in_dim, hid)   # 子 Module 自动注册
        self.act = nn.GELU()
        self.l2 = nn.Linear(hid, out_dim)

    def forward(self, x):            # 定义前向，不要在里面建新层
        return self.l2(self.act(self.l1(x)))
```

- `__init__` 只做**结构声明**（搭零件、注册参数），**绝不在 forward 里 new 层**——否则每步前向都新建参数，永远不进优化器、永远不学习。
- `forward` 只写**数据流**（怎么调零件）。
- 调用 `model(x)` 实际走的是 `nn.Module.__call__` → 内部触发 hooks → 再调 `forward(x)`。**永远用 `model(x)` 而非 `model.forward(x)`**，否则 hooks 不触发。

### 3.3 自动注册机制

`nn.Module` 重写了 `__setattr__`：当你 `self.xxx = y`，
- 若 `y` 是 `nn.Parameter` → 加入 `self._parameters`；
- 若 `y` 是 `nn.Module` → 加入 `self._modules`；
- 若 `y` 是 `Tensor` 但没 `register_buffer` → 进 `self._buffers`? **不是**，普通 Tensor 赋值会进 `_non_persistent_buffers_set`? 也不对——它只是被当作普通属性，**不进 state_dict、不随 `.to()` 搬运**。要让它跟随，必须显式 `register_buffer`。
- 其他（int/str/函数）→ 普通属性。

这导致 `model.parameters()` 能**递归**遍历整棵 Module 树，把所有叶子 Parameter 一次返回给优化器。

### 3.4 模式切换：`.train()` / `.eval()`

`self.training` 是布尔标志，`.train(mode=True)` 递归把所有子 Module 的 `training` 置位。某些层行为依赖它：

- `nn.Dropout`：`training=True` 时随机置零，`training=False` 时恒等；
- `nn.BatchNorm*`：训练用 batch 统计并更新 running stats，推理用 running stats；
- `nn.LayerNorm`：**不依赖** training（永远用当前 batch 统计，无 running stats）。

> [!warning] 忘记 `.eval()` 是经典 bug
> 验证/推理前不调 `model.eval()`，Dropout 还在丢、BN 还在用 batch 统计 → 验证指标抖动、推理结果每次不同。

### 3.5 设备与 dtype 搬运：`.to()`

`model.to(device="cuda", dtype=torch.bfloat16)` 会递归把所有 Parameter 和 **已注册 Buffer** 搬过去并转 dtype。普通属性 Tensor 不受影响（这是 buffer 必须注册的又一原因）。

### 3.6 冻结参数 / 替换

- 冻结：`for p in model.parameters(): p.requires_grad_(False)`（仍进 state_dict，但不学）。
- 仅训部分：先全冻结，再对要训的 `p.requires_grad_(True)`。
- 替换某层：`model.l2 = nn.Linear(...)` 直接赋值，旧层被替换、新层自动注册。

## 4. 数学原理 / 公式

`nn.Module` 本身不涉及数学公式，但其中 `nn.Parameter` 的"自动求导"链路：

$$
\theta \in \mathbb{R}^d,\quad \text{requires\_grad}=\text{True}\ \Rightarrow\ \frac{\partial \mathcal{L}}{\partial \theta}\ \text{由 autograd 在 backward 时填入 } \theta.\text{grad}
$$

`nn.Linear` 的前向：

$$
y = x W^\top + b,\quad x\in\mathbb{R}^{B\times d_{in}},\ W\in\mathbb{R}^{d_{out}\times d_{in}},\ b\in\mathbb{R}^{d_{out}}
$$

PyTorch 把 `Linear.weight` 存成 `(out_features, in_features)` 并用 `xW^T` 而非 `xW`，是历史约定，便于梯度形状对齐。

## 5. 代码示例

```python
import torch
import torch.nn as nn

# 1) 自定义 Module：演示 Parameter / Buffer / 子Module 三件套
class TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)          # 子Module（自带Parameter weight/bias）
        self.gamma = nn.Parameter(torch.ones(dim))   # 显式Parameter：可学习
        self.register_buffer("const_bias", torch.zeros(dim))  # Buffer：不学但要保存/搬设备

    def forward(self, x):
        return self.fc(x) * self.gamma + self.const_bias

model = TinyBlock(8)

# 2) 三类对象如何被收集
print("params:", [n for n,_ in model.named_parameters()])         # fc.weight, fc.bias, gamma
print("buffers:", [n for n,_ in model.named_buffers()])           # const_bias
print("modules:", [n for n,_ in model.named_modules()][:3])       # '', fc

# 3) 模式切换
model.eval()            # training=False，Dropout/BN行为改变
model.train()           # training=True

# 4) 设备/dtype搬运（递归）
model.to("cpu", dtype=torch.float32)

# 5) forward调用：用 model(x) 而非 model.forward(x)
x = torch.randn(4, 8)
y = model(x)            # 走 __call__ -> hooks -> forward
assert y.shape == (4, 8)
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[Tensor]]（Parameter 即 Tensor 子类）、[[autograd机制]]（Module 的参数会被反向传播）。
- **下游（应用）**: [[forward与backward]]（forward 在此定义、backward 由 autograd 自动）、[[state_dict与load_state_dict]]（Module 的状态序列化）、[[optimizer state管理]]（优化器收集 `model.parameters()`）、[[hooks机制]]（Module 提供 forward/backward hook 挂载点）。
- **对比 / 易混**:
  - **`nn.Module` vs `nn.functional`**：`nn.functional.linear(x,w,b)` 是无状态函数；`nn.Linear` 是有状态的 Module（把 w/b 当 Parameter 持有）。需要学习参数→用 Module；纯计算/复用已有参数→用 functional。
  - **`nn.Module` vs `torch.jit.ScriptModule`**：后者是图捕获后的可序列化产物，见 [[计算图]]。
  - **Parameter vs Buffer**：见 3.1 表。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **`__init__` 里忘记 `super().__init__()`** → 参数注册机制不工作，`parameters()` 返回空，模型不学任何东西。
> 2. **在 `forward` 里 `self.w = nn.Parameter(...)` 新建参数** → 每次前向重建，参数不进优化器，等于没学。参数必须在 `__init__` 里建。
> 3. **用 `model.forward(x)` 而非 `model(x)`** → 跳过 hooks，调试/特征提取失效。
> 4. **普通 Tensor 当 Buffer**：`self.mask = torch.tensor(...)` 而非 `register_buffer` → `.to(cuda)` 不搬、`state_dict` 不存、多卡各 rank 形状/设备不一致。
> 5. **推理忘 `model.eval()`** → Dropout/BN 行为错误，指标抖。
> 6. **以为冻结 `requires_grad=False` 就不进优化器** → 仍会进（`parameters()` 不过滤 requires_grad），只是不更新；想完全排除要用 `filter` 或 `params=[p for p in m.parameters() if p.requires_grad]`。
> 7. **`model.to(device)` 没接住返回值？** 其实 `.to` 是 in-place 修改 Parameter（返回 self），可以不接；但 `tensor.to()` 不是 in-place，别混。

## 8. 延伸细节

### 8.1 `__call__` 与 hook 触发顺序

`model(x)` 实际执行（简化）：

1. 遍历 `forward_pre_hook` → 可改输入；
2. 调 `forward(x)`；
3. 遍历 `forward_hook` → 可改输出；
4. （若 `requires_grad`）注册 `full_backward_hook` → 反向时触发。

这就是为什么必须走 `__call__`：hooks 只在这里挂。

### 8.2 Module 树与 FSDP/DDP 的关系

[[DDP]] 和 [[FSDP]] 都依赖"Module 是树"这一性质：它们递归遍历整棵树，分别对参数做 broadcast（DDP）或 sharding（FSDP）。所以**自定义 Module 时保证结构清晰、参数都在 `__init__` 注册**，是分布式训练正确的前提。

### 8.3 `add_module` 与点访问

`self.l1 = nn.Linear(...)` 等价于 `self.add_module("l1", nn.Linear(...))`。也可以 `getattr(model, "l1")` / 字符串路径访问，这是配置驱动搭模型（如 from_config）的基础。

### 8.4 `nn.ModuleList` vs `nn.Sequential` vs Python `list`

- 用普通 `list` 存子 Module → **不会被注册**，`parameters()` 找不到，参数不学。这是高频 bug。
- `nn.ModuleList([...])` → 注册为一串子 Module，但**不定义 forward 顺序**（自己写 for 循环）。
- `nn.Sequential(a, b, c)` → 注册且自动按顺序串接 forward。

---
相关: [[PyTorch核心]]、[[forward与backward]]、[[state_dict与load_state_dict]]、[[optimizer state管理]]、[[hooks机制]]、[[Tensor]]
