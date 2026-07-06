# state_dict与load_state_dict

> **所属章节**: [[PyTorch核心]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 基础（工程必备）

## 1. 一句话定义

**`state_dict`（状态字典）** 是把一个 [[nn.Module]] 里的**全部可训练参数（Parameter）与已注册 Buffer** 序列化成的 `OrderedDict{名字: 张量}`；**`load_state_dict`** 是它的逆操作——用一份外部状态字典把同名同形状的张量灌回 Module。两者是 PyTorch **模型保存/加载、断点续训、多卡同步、权重迁移**的统一基础设施。

> [!note] 名字辨析
> `state_dict` 不是优化器状态，也不是某个层的私有属性，而是**整棵 Module 树**的扁平化键值视图：键用点号路径表示层级（如 `"layers.0.attn.q_proj.weight"`），值是 Tensor。优化器自己也有一个 `optimizer.state_dict()`，见 [[optimizer state管理]]。

> [!note] 解答：state_dict 是"存权重本身"吗？safetensors 存的是它吗？
> 
> **基本是，但比"权重"多一点点**：
> 
> | 问 | 答 |
> |---|---|
> | state_dict = 存权重本身吗？ | **是，但还含 Buffer**。它是 `{名字: 张量}` 的字典，值 = 所有 `nn.Parameter`（权重/bias/γ/β）+ 所有 `register_buffer` 的张量（BN running stats、RoPE inv_freq 等）。说"存权重"是大白话，严格说是"存模型全部数值状态"。 |
> | 验证模型加载有没有正确？ | 三步：(1) `load_state_dict(strict=True)` 默认校验**键名+形状**，报错即没对上；(2) 看返回值 `res = model.load_state_dict(sd)`，`res.missing_keys`/`res.unexpected_keys` 为空才算全灌进去（见 [[#3.4 load_state_dict 的关键参数]]、[[#7. 常见误区与易错点]] 误区 2）；(3) **你说的"把 state_dict 存下来"正是更严的数值校验**：加载后再 `got = model.state_dict()` 取一份，与原 sd 逐键 `torch.equal(sd[k], got[k])` 或算 `abs diff`——strict 只查键/形状，这步查**数值**有没有在 dtype 转换/拷贝中走样。 |
> | HF 下载的 `.safetensors` 存的是 state_dict 吗？ | **是**。`.safetensors` 本质就是 `{name: tensor_buffer}` 的序列化视图，与 `model.state_dict()` 一一对应（见 [[#8.3 safetensors]]）。加载时 `model.load_state_dict(safetensors.torch.load_file(path))` 即可。区别只在格式：`.pt` 用 pickle（有代码执行风险），`.safetensors` 零拷贝、安全、加载快。 |
> 
> **一句话**：state_dict ≈ 模型的"权重+非学习状态"打包；safetensors 是它在 HF 生态的标准存储格式；`load_state_dict` 的返回值 + 存下来逐键比对是验证加载是否正确的两道关。
## 2. 为什么需要它（动机与背景）

训练中需要频繁做这些事，每件都依赖"把模型当前所有权重/Buffer打包成可搬运的东西"：

> [!note] 解答：Buffer 是什么？
> 
> **Buffer** 是 `nn.Module` 里用 `self.register_buffer(name, tensor)` 注册的、**不随梯度更新、但要随模型一起搬设备/存盘/加载**的张量。它是"非学习的状态"。
> 
> | | Parameter（权重） | Buffer |
> |---|---|---|
> | 注册方式 | `nn.Parameter(t)` / `register_parameter` | `register_buffer(name, t)` |
> | requires_grad | 默认 True（反向更新） | 默认 False（不学） |
> | 进 state_dict | ✅ | ✅（`persistent=True` 时，默认） |
> | 谁更新 | 梯度下降 | forward 里手动更新（如 `running_mean = momentum*running_mean + ...`） |
> | 典型例子 | Linear 的 W/b、LayerNorm 的 γ/β | BatchNorm 的 running_mean/running_var（推理用、训练统计）、RoPE 的 inv_freq（预计算频率表）、绝对位置编码 |
> 
> **为什么要单独有 Buffer**：这些量**不是学出来的**（不需要 `.grad`），但**推理时必须带着**（BN 没有 running stats 就算不出正确输出），且**多卡/存盘要同步**。如果只塞 `self.x = tensor` 而不 register，`.to(cuda)` 不搬它、`state_dict()` 不存它、加载时丢失——多卡不一致、续训断在 BN 上。Buffer 把"非学习但需搬运"做成一等公民。见 [[#3.1 什么样的对象进 state_dict]]。
> 
> **记忆口诀**：Parameter = 学的权重（有 grad）；Buffer = 不学但要带的状态（BN running/RoPE）；两者都进 state_dict，普通 `self.x = tensor` 啥都不进。

- **保存 checkpoint**：训练中断后能从同一权重继续，不能丢失。
- **断点续训**：恢复模型 + 优化器状态 + LR 调度 + step 计数。
- **多卡权重同步**：rank 0 拿到新权重，broadcast 给所有 rank（[[weight sync mechanism]]、[[DDP]]）。
- **权重迁移 / 微调**：拿预训练权重灌进自己改过的模型（部分键匹配、strict=False）。
- **推理部署**：导出纯权重给 C++ / ONNX / 推理引擎。

如果靠用户手动遍历 `model.parameters()`，会漏 Buffer、丢名字、丢结构、多卡对不齐。`state_dict` 把"序列化整个 Module"做成一等公民，统一处理以上所有场景。

> [!note] 解答：LR 调度（Learning Rate Scheduling）是什么？
> 
> **LR 调度** = 训练过程中按 step / epoch **改变学习率**的策略（由 `torch.optim.lr_scheduler` 提供）。学习率 `lr` 是梯度下降的步长 $\theta \leftarrow \theta - \text{lr}\cdot\nabla\mathcal{L}$，固定不变的 lr 很难两全：太大前期快但后期震荡不收敛，太小稳但前期慢。调度器让 lr **随训练进程变化**，兼顾"前期快、后期稳"。
> 
> **常见策略**：
> 
> | 策略 | 形状 | 用途 |
> |---|---|---|
> | **Warmup + Cosine** | 线性升到峰值 → 余弦降到 ~0 | LLM 标配（GPT/Llama 都用）；前期小步防发散，后期平滑收敛 |
> | **Warmup + Linear** | 升到峰值 → 线性降 | 大模型常替代 cosine |
> | **StepLR** | 每 N step 乘 γ | 经典 CV，少用于 LLM |
> | **ConstantLR** | 不变 | LoRA / 微调小模型 |
> 
> **为什么续训要存 `scheduler.state_dict()`**：调度器内部记着"现在到第几步、当前 lr 是多少、是否到 decay 阶段"。断点续训若不恢复，lr 会从初始值重跑（如重新 warmup），破坏训练曲线。所以 [[#3.5 断点续训的完整四件套]] 里 scheduler 和 model/optimizer/step 一起存。见 [[学习率调度]]。
> 
> ```python
> from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
> sched = CosineAnnealingLR(opt, T_max=total_steps)
> # 续训：ckpt["scheduler"] = sched.state_dict(); sched.load_state_dict(ckpt["scheduler"])
> ```
> 
> 相关：[[学习率调度]]、[[SGD]]、[[Adam与AdamW]]、[[optimizer state管理]]。
## 3. 核心概念详解

### 3.1 什么样的对象进 `state_dict`

| 对象 | 是否进 `model.state_dict()` | 备注 |
|---|---|---|
| `nn.Parameter` | ✅ | 权重、bias、LayerNorm 的 γ/β 等 |
| `register_buffer` 注册的 Buffer（默认 `persistent=True`） | ✅ | BN running stats、RoPE inv_freq |
| `register_buffer(..., persistent=False)` | ❌ | 仅存活在内存、不序列化（如临时 mask） |
| 普通 `self.x = torch.tensor(...)`（没注册） | ❌ | `.to(cuda)` 也不搬它 |
| 子 Module 的以上对象 | ✅（递归） | 键带路径前缀 `sub.fc.weight` |
| 优化器内部状态（momentum 等） | ❌（在 `optimizer.state_dict()`） | 见 [[optimizer state管理]] |

### 3.2 键的命名规则

`state_dict()` 递归遍历 Module 树，键 = **从根到叶的点号路径**：

> [!note] 解答：怎么从 state_dict 的键识别 dense / MoE？命名规则是什么？
> 
> state_dict 的键 = **从根 Module 到叶张量的点号路径**（见上文），所以**键名就是模型结构的投影**。看键里有没有 `experts.{i}.` 和 `gate`/`router`，就能判断 dense 还是 MoE：
> 
> | 模型类型 | FFN 部分的键（典型） | 路由器键 | 说明 |
> |---|---|---|---|
> | **Dense**（GPT/Llama） | `layers.0.mlp.gate_proj.weight`<br>`layers.0.mlp.up_proj.weight`<br>`layers.0.mlp.down_proj.weight` | 无 | 每个 block **一组** FFN 权重，所有 token 共享 |
> | **MoE**（Mixtral/DeepSeek） | `layers.0.mlp.experts.0.gate_proj.weight`<br>`layers.0.mlp.experts.1.gate_proj.weight`<br>...（共 E 组） | `layers.0.mlp.gate.weight`（路由器） | 每个 block **E 组**专家权重 + 1 个路由器；键里多 `experts.{i}.` 一段 |
> 
> **命名规则总结**：
> - Dense FFN：`...mlp.{gate_proj,up_proj,down_proj}.{weight,bias}`，**每个 block 一份**；
> - MoE FFN：`...mlp.experts.{0..E-1}.{gate_proj,up_proj,down_proj}.{weight,bias}`，**每个 block E 份** + `...mlp.gate.weight`（路由，也叫 `router`）；
> - 共享专家（DeepSeek-V3）：额外 `...mlp.shared_experts.gate_proj.weight` 等。
> 
> **怎么自动识别**（脚本思路）：
> 
> ```python
> import re
> keys = list(state_dict.keys())
> is_moe = any("experts." in k or k.endswith("gate.weight") for k in keys if "mlp" in k)
> # 数专家数：取含 "experts." 的键，提取 {i} 的最大值+1
> expert_ids = {int(m.group(1)) for k in keys if (m := re.search(r"experts\.(\d+)\.", k))}
> num_experts = max(expert_ids) + 1 if expert_ids else 0
> ```
> 
> **为什么重要**：加载权重时键对不上就报错（[[#3.4 load_state_dict 的关键参数]] `strict`），看键名能预判结构是否匹配、要不要 `strict=False`。HuggingFace `config.json` 的 `num_experts`/`expert_inter_size` 也对应这套命名。见 [[MoE]]、[[参数量计算]]。
```text
model = Transformer(
    layers = ModuleList([ Block(attn = Attention(q_proj=Linear(...),
                                               k_proj=Linear(...)) ) ])
)
# state_dict 键示例：
# "layers.0.attn.q_proj.weight"
# "layers.0.attn.q_proj.bias"
# "layers.0.attn.k_proj.weight"
# ...
```

> [!note] 解答：mlp 指的是什么意思？
>
> **MLP = Multi-Layer Perceptron（多层感知机）**，但在 Transformer / LLM 语境里，`mlp` 几乎总是指 **每个 block 里的前馈子网络（FFN, Feed-Forward Network）**，即 attention 之后那段"全连接 + 激活 + 全连接"的小网络。它和"经典机器学习里的 MLP"是同一个东西的本质（堆叠的线性层 + 非线性激活），只是被当作 Transformer block 的一个标准子模块来用，所以代码里直接命名成 `self.mlp`。
>
> ### 1. 经典 MLP 是什么
>
> 一个**多层感知机** = 多个线性层（`nn.Linear`）之间插入非线性激活函数堆叠而成的前馈网络：
>
> $$
> \text{MLP}(x) = W_2\,\sigma(W_1 x + b_1) + b_2
> $$
>
> - 输入 $x\in\mathbb{R}^{d}$，第一层把它升到隐维 $d_{ff}$（通常 $d_{ff}=4d$），第二层再降回 $d$；
> - $\sigma$ 是激活（ReLU / GELU / SwiGLU 等）；
> - "多层"指至少两层线性变换夹一个非线性，否则就退化成单层线性映射 $Wx$，表达力不够。
>
> 它是最早的"深度网络"雏形（MLP → CNN → RNN → Transformer 都建立在它之上），核心思想就是**用非线性激活把线性变换串起来，逼近任意函数**（universal approximation theorem）。
>
> ### 2. Transformer 里的 `mlp` 子模块
>
> 在 GPT / Llama 等 decoder-only 模型里，每个 Transformer block = `Attention` + `MLP`（+ 残差 + LayerNorm）。这里的 `mlp` 就是 FFN，典型实现：
>
> | 模型 | FFN 结构 | state_dict 键 |
> |---|---|---|
> | **标准 GPT**（ReLU/GELU） | `Linear(d, 4d)` → GELU → `Linear(4d, d)` | `mlp.fc1.weight`、`mlp.fc2.weight` |
> | **Llama / Llama2**（SwiGLU） | 三个并行 `Linear`：`gate_proj`、`up_proj` → SiLU(gate)*up → `down_proj` | `mlp.gate_proj.weight`、`mlp.up_proj.weight`、`mlp.down_proj.weight` |
> | **MoE**（Mixtral/DeepSeek） | 把单个 MLP 换成 E 个专家 MLP + 一个路由器 | `mlp.experts.{i}.gate_proj.weight` … + `mlp.gate.weight` |
>
> Llama 的 SwiGLU FFN 公式：
>
> $$
> \text{FFN}_{\text{SwiGLU}}(x) = \text{down_proj}\big(\text{SiLU}(\text{gate_proj}(x)) \odot \text{up_proj}(x)\big)
> $$
>
> 其中 $\odot$ 是逐元素乘、$\text{SiLU}(x)=x\sigma(x)$。注意它有**三个**权重矩阵而不是两个，所以键里看到 `gate_proj/up_proj/down_proj` 三个就是 Llama 系的标志。
>
> ### 3. 为什么 attention 后面非要跟一个 MLP
>
> - **Attention 负责跨 token 信息混合**（"谁和谁相关"），但它是**线性**的、对每个 token 的变换是低秩的加权平均；
> - **MLP 负责每个 token 自己的非线性变换**（"这个 token 的特征该怎么加工"），逐位置独立、提供高维非线性表达力；
> - 二者分工：attention = 跨位置路由信息，MLP = 位置内的函数逼近器。缺了 MLP，模型只剩线性 attention 堆叠，等价于一个很深的线性映射，表达力塌掉。
>
> 这也是为什么 `state_dict` 键里 `mlp.*` 几乎占了整个 block 参数量的大头：$d_{ff}=4d$ 时，FFN 参数 $\approx 8d^2$，而 attention 的 QKV+O $\approx 4d^2$，**FFN 是 block 里参数最多的子模块**。见 [[参数量计算]]。
>
> ### 4. 常见误区
>
> - ❌ 把 `mlp` 当成"多任务学习协议/Multi-Learning Protocol"之类缩写——不是，就是 Multi-Layer Perceptron；
> - ❌ 以为 Llama 的 `mlp` 只有两个矩阵——它有**三个**（gate/up/down），因为是 SwiGLU 而不是普通 GELU-FFN；
> - ❌ 把 `mlp` 和 `lm_head`（输出分类头）混为一谈——`lm_head` 是最终映射到词表 logits 的投影，`mlp` 是每个 block 内部的 FFN，二者位置和作用不同；
> - ❌ 看到 MoE 的 `mlp.experts.0.gate_proj` 以为和路由器 `mlp.gate` 是同一个——前者是专家内部 SwiGLU 的 gate_proj，后者是路由器（router），命名撞车但含义不同。
>
> ### 5. 关联
>
> - 上游：[[Transformer]]（block 结构）、[[Attention]]（MLP 的搭档）；
> - 同级：[[SwiGLU]]（Llama 系 FFN 的激活）、[[MoE]]（把 MLP 换成多专家）；
> - 参数视角：[[参数量计算]]（FFN 占 block 大头）、本笔记 [[#3.2 键的命名规则]]（`mlp.*` 键的结构投影）。

所以**模型结构变了，键就变**——这是迁移学习时键对不上、需要 `strict=False` 的根因。

> [!note] 解答：`strict=False` 是什么意思？
> 
> `load_state_dict(sd, strict=False)` = **只灌同名的键，多余/缺失的键不报错，静默跳过**。对比 `strict=True`（默认）要求模型与 sd 的键**完全一致**，多一个少一个都报错。
> 
> | strict | 键完全一致 | 模型比 sd 多键 | 模型比 sd 少键 |
> |---|---|---|---|
> | `True`（默认） | 正常加载 | 报 `unexpected_keys` 错 | 报 `missing_keys` 错 |
> | `False` | 正常加载 | 静默跳过，返回 `unexpected_keys` | 静默跳过，返回 `missing_keys` |
> 
> **什么时候用 `strict=False`**：
> - **微调 / 改结构**：预训练模型没有你新增的分类头 `lm_head`（或 vice versa），只灌能对上的部分；
> - **加载部分层**：只用预训练的 encoder，不加载 decoder；
> - **兼容旧 checkpoint**：模型升级后某些层改名/删了。
> 
> **陷阱**：`strict=False` **默默**忽略不匹配，你以为加载成功，其实大部分键没灌进去，模型像随机初始化。务必打印返回值核对：
> 
> ```python
> res = model.load_state_dict(sd, strict=False)
> print(res.missing_keys, res.unexpected_keys)  # 空列表才算全灌进去
> ```
> 
> 详见 [[#3.4 load_state_dict 的关键参数]]、[[#7. 常见误区与易错点]] 误区 2。
### 3.3 `torch.save(model.state_dict(), path)` 与两种保存粒度

> [!note] 解答：§3.3 两种保存粒度细说
> 
> PyTorch 存模型有两种"粒度"，区别在**存的是权重字典还是整个对象**：
> 
> **粒度 A：只存 state_dict（推荐）**
> 
> ```python
> torch.save(model.state_dict(), "weights.pt")        # 只存 {名字: 张量}
> # 加载：必须先 new 一个空模型，再灌权重
> model2 = MyModel()                                   # 先实例化（要类定义可导入）
> model2.load_state_dict(torch.load("weights.pt"))    # 再灌权重
> ```
> - 存的是**纯权重字典**（几 MB ~ 几百 GB 的张量），与代码解耦；
> - 加载时**必须先有类定义**（`MyModel`），先 `new` 再 `load`；
> - 好处：代码重构/改名/换 PyTorch 版本都不影响权重加载——只要层结构（键名+形状）对得上。
> 
> **粒度 B：存整个 model 对象（不推荐）**
> 
> ```python
> torch.save(model, "model.pt")        # pickle 整个 Python 对象（含类引用）
> # 加载：直接得对象，不用先 new
> model2 = torch.load("model.pt")
> ```
> - 存的是**整个 Python 对象**（pickle 序列化），里面记着"我是哪个类的实例 + 权重"；
> - 加载时 `torch.load` 直接还原对象，**但要求那个类定义的 Python 路径当时还在**；
> - 坑：一旦你重构代码（改类名、移文件、改 `__init__` 签名），`torch.load` 找不到原来的类就崩——`AttributeError` / `ModuleNotFoundError`。
> 
> **为什么官方推荐 A**：权重与代码解耦。模型结构变了只要键还能对上（或 `strict=False`）就能灌；而 B 把权重和代码路径绑死，迁移性差。生产/续训/跨团队一律用 A。见 [[#7. 常见误区与易错点]] 误区 4。
> 
> **类比**：A = 只存"存档数据"，加载时重新开游戏再导入；B = 把"游戏 + 存档"一起打包，换台机器/换版本就打不开。
```python
# 粒度 A：只存权重（小、跨代码结构兼容性好）
torch.save(model.state_dict(), "weights.pt")
model2 = MyModel(); model2.load_state_dict(torch.load("weights.pt"))

# 粒度 B：存整个 Module 对象（pickle 整个类实例，依赖代码路径）
torch.save(model, "model.pt")          # 不推荐：加载时类定义必须可导入
```

> [!warning] 保存整个 model vs 保存 state_dict
> `torch.save(model)` 用 pickle 序列化整个对象，加载时依赖**当时那个类定义的 Python 路径**；代码重构/改名后加载会崩。**官方推荐只存 `state_dict`**，加载时先 `model = MyModel()` 实例化再 `load_state_dict`，解耦权重与代码。

### 3.4 `load_state_dict` 的关键参数

```python
model.load_state_dict(sd, strict=True, assign=False)
```

| 参数 | 作用 |
|---|---|
| `strict=True`（默认） | 要求 **键完全一致**（多/少/形状不符都报错）。续训/同结构加载用。 |
| `strict=False` | 只灌同名的，多余/缺失的键不报错，返回 `missing_keys`/`unexpected_keys`。微调/改结构时用。 |
| `assign=True`（PyTorch≥2.0） | 直接把传入张量**赋值**给 Parameter，不做 in-place 拷贝。用于跨设备/避免拷贝，但会改变 Parameter 对象身份。 |
| `copy=False` | 内部是否拷贝。默认 True 防止改 sd 影响模型。 |

> [!tip] 续训务必校验返回值
> `load_state_dict(strict=False)` 默默忽略不匹配的键，是微调时常见陷阱：你以为加载成功，其实大部分键没灌进去。打印 `missing_keys`/`unexpected_keys` 核对一遍再开训。

### 3.5 断点续训的完整四件套

续训不止恢复模型权重，还要恢复优化器状态、调度器、step：

```python
ckpt = {
    "model":     model.state_dict(),          # 权重+Buffer
    "optimizer": optimizer.state_dict(),      # 动量/Adam 矩等
    "scheduler": scheduler.state_dict(),     # LR 位置
    "step":      global_step,                 # 训练进度
    "rng":       torch.get_rng_state(),       # 随机性复现
}
torch.save(ckpt, "ckpt.pt")
```

恢复顺序：先 `model.load_state_dict` → 再 `optimizer.load_state_dict`（依赖参数已就位）→ `scheduler.load_state_dict` → step → rng。

### 3.6 跨设备 / 跨 dtype 加载

- `torch.load(path, map_location="cpu")`：把存在 GPU 上的 checkpoint 拉回 CPU，避免无 GPU 环境加载失败。
- 加载 bf16 权重到 fp32 模型：`load_state_dict` 会按形状校验，dtype 不一致时若 `assign=False` 会拷贝并转 dtype；若要保留低精度，先 `model.to(torch.bfloat16)` 再加载。

## 4. 数学原理 / 公式

本条不涉及新公式，但加载行为可形式化为：对每个键 $k$，若 $\text{shape}(sd[k])=\text{shape}(m_k)$ 且名字匹配，则 $m_k \leftarrow sd[k]$（in-place 拷贝或 assign）；否则按 `strict` 决定报错或跳过。

$$
\forall k\in\text{keys}(m):\quad m_k \leftarrow sd[k]\ \ \text{iff}\ \ k\in sd\ \text{and}\ \text{shape}(m_k)=\text{shape}(sd[k])
$$

## 5. 代码示例

```python
import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.register_buffer("const", torch.ones(d))   # persistent buffer
        self.register_buffer("tmp_mask", torch.zeros(d), persistent=False)  # 不进 state_dict
    def forward(self, x):
        return self.fc(x) + self.const

m = MLP(8)
sd = m.state_dict()
print(list(sd.keys()))   # ['fc.weight', 'fc.bias', 'const']   没有 tmp_mask

# 1) 保存/加载（粒度 A）
torch.save(sd, "mlm.pt")
m2 = MLP(8)
missing, unexpected = [], []
res = m2.load_state_dict(torch.load("mlm.pt"), strict=True)
print(res)   # <All keys matched successfully>

# 2) strict=False 演示（改结构后迁移）
class MLP2(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.new_layer = nn.Linear(d, d)   # 新增 -> 加载时 unexpected=无，missing=new_layer.*
    def forward(self, x): return self.fc(x)

m3 = MLP2(8)
res = m3.load_state_dict(torch.load("mlm.pt"), strict=False)
print("missing:", res.missing_keys, "unexpected:", res.unexpected_keys)
# missing: ['new_layer.weight','new_layer.bias']  unexpected: ['const']

# 3) 断点续训四件套
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
ckpt = {"model": m.state_dict(), "optimizer": opt.state_dict(), "step": 1000}
torch.save(ckpt, "ckpt.pt")
# 恢复
m.load_state_dict(ckpt["model"])
opt.load_state_dict(ckpt["optimizer"])   # 必须在 model 加载之后

# 4) 跨设备加载
sd = torch.load("mlm.pt", map_location="cpu")
m.to("cpu").load_state_dict(sd)
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[nn.Module]]（state_dict 是 Module 的方法）、[[Tensor]]（值都是 Tensor）、[[数值类型与精度]]（加载涉及 dtype）。
- **下游（应用）**: [[optimizer state管理]]（续训要同时存它）、[[DDP]]/[[FSDP]]（启动时同步 state_dict；FSDP 的分片参数要先 unshard 再 save）、[[weight sync mechanism]]（多卡权重广播本质是传 state_dict）、推理部署（导出权重给 vLLM 等推理引擎）。
- **对比 / 易混**:
  - `model.state_dict()` vs `optimizer.state_dict()`：前者是权重+Buffer，后者是动量/Adam 矩+step；**两个都要存才算完整续训**。
  - `torch.save(model)` vs `torch.save(model.state_dict())`：见 3.3，后者解耦代码。
  - `persistent=True/False` buffer：决定是否进 state_dict。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **续训只存 `model.state_dict()`** → 优化器动量丢失，Adam 的二阶矩归零，相当于重新 warmup，损失曲线会跳。
> 2. **`strict=False` 不看返回值** → 静默漏灌大部分权重，模型像随机初始化。务必打印 `missing_keys/unexpected_keys` 核对。
> 3. **改了模型结构没改 checkpoint** → `strict=True` 报 size/key mismatch；`strict=False` 又可能漏灌关键层。
> 4. **保存整个 model（`torch.save(model)`）** → 代码重构后加载失败；只存 state_dict。
> 5. **加载 GPU checkpoint 不用 `map_location`** → 无 GPU 机器直接崩，或被加载到原卡上 OOM。
> 6. **FSDP 下直接 `model.state_dict()`** → 得到的是分片后的 shape，灌回非分片模型会报形状不符；要用 `FSDP.state_dict_type` 配置 `FULL_STATE_DICT` 先 unshard。
> 7. **以为 Buffer 自动跟随** → 没 `register_buffer` 的 `self.x = tensor` 不进 state_dict，加载后丢失，多卡不一致。
> 8. **`assign=True` 后改了原 sd** → 反向影响模型；`assign` 是别名而非拷贝，慎用。

## 8. 延伸细节

### 8.1 FSDP 的 state_dict 三种类型

`FullySharded Data Parallel` 提供 `ShardedStateDictType / LocalStateDictType / FullStateDictType`：分片保存省显存、全量保存便于跨框架迁移。LLM 训练常用 Sharded（每个 rank 存自己那片），推理部署前再 gather 成 Full。

### 8.2 `meta` 设备与 `load_state_dict` 的 lazy 初始化

`torch.device("meta")` 上的张量**不分配真实显存**，只记 shape/dtype/stride。可以先 `model = MyModel(...).to_empty(device="meta")` 建出巨大模型骨架（0 显存），再 `load_state_dict` 灌真实权重——这是 HuggingFace `low_cpu_mem_usage` 加载百亿参数模型的原理，避免先在 CPU 分配一份完整权重再搬 GPU 的双倍峰值。

### 8.3 safetensors

HuggingFace 推的 `.safetensors` 格式相比 `.pt`（pickle）更安全（无任意代码执行风险）、零拷贝、加载快；本质仍是 `{name: tensor_buffer}` 的 state_dict 视图，可互转。生产部署优先 safetensors。

### 8.4 与 ONNX / TorchScript 导出的关系

导出推理图时一般先 `model.load_state_dict(...)` 再 `model.eval()` 再 `torch.onnx.export(model, ...)` 或 `torch.jit.trace`。state_dict 是"权重层"的统一中间表示，图导出是"结构层"，二者分离使权重可跨后端复用。

---
相关: [[PyTorch核心]]、[[nn.Module]]、[[optimizer state管理]]、[[Tensor]]、[[DDP]]、[[FSDP]]、[[weight sync mechanism]]
