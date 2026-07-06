# batch与mini-batch

> **所属章节**: [[训练工程基础]]
> **所属模块**: [[01-ML深度学习基础]]
> **难度**: 基础

## 1. 一句话定义

**batch（批）** 是一次前向 + 反向所用的**样本子集**；**mini-batch（小批）** 指用"几十~几千样本"的小批做 [[SGD]] 更新，介于全量（BGD）和单样本（纯 SGD）之间，是深度学习标准训练单元；batch size 就是这一批的样本数 $|B|$。

## 2. 为什么需要它（动机与背景）

三个极端都不好用：

- **全量（BGD, batch=全部数据）**：梯度准但一次更新要算百万样本，几小时一步，还易陷尖锐极小值。
- **单样本（batch=1）**：更新快但梯度噪声极大、且无法利用 GPU 并行（一个样本填不满算力）。
- **GPU 需要"吃饱"**：现代 GPU 算力远超单样本需求，batch 太小 = 算力浪费。

**mini-batch 是三者的甜点**：batch 足够大让 GPU 吃饱 + 梯度噪声可控 + 更新频率够高。同时噪声还有正则化效果（逃离尖锐极小值，见 [[SGD]]）。

## 3. 核心概念详解

### 3.1 三种梯度下降

| 名称 | batch size | 梯度 | 特点 |
|---|---|---|---|
| BGD | 全部数据 | 精确 | 准但慢、易陷鞍点 |
| SGD（严格） | 1 | 噪声极大 | 快但抖、GPU 浪费 |
| **mini-batch SGD** | $|B|$（几十~几千） | 近似 | **主流**，速度/精度/GPU 利用率平衡 |

### 3.2 batch size 的三个影响

1. **梯度方差**：$\text{Var}(\hat g)\propto 1/|B|$，batch 越大方差越小、越接近真梯度。
2. **GPU 利用率**：batch 太小吃不饱 GPU（compute-bound 算子如 matmul 利用率低）；到一定大小后饱和。
3. **泛化**：小 batch 噪声大 → 平坦极小值 → 泛化好；大 batch 噪声小 → 尖锐极小值 → 泛化有时差（但可用更大 lr + warmup 缓解）。

### 3.3 gradient accumulation（梯度累积）

显存放不下大 batch 时，**分多次小前向+反向，梯度累加到 `.grad`，凑够目标 batch 再 step**：

```
for i, (x,y) in enumerate(loader):
    loss = model(x, y) / accum_steps      # 缩放
    loss.backward()                       # 梯度累加
    if (i+1) % accum_steps == 0:
        opt.step(); opt.zero_grad()       # 凑够才更新
```

效果：等效 batch = micro_batch × accum_steps。显存只够 micro_batch=8 时，accum=8 可等效 batch=64。

> [!note] 解答：accum 是什么？
> **accum 就是"梯度累积步数"（accumulation steps / accum_steps），代码里常简写成 `accum`。** 它的意思是：**连续做 accum 次小前向+小反向、把梯度累加进 `.grad`，然后才调用一次 `opt.step()` 更新参数。** 目的是在显存放不下大 batch 时，用多次小批"拼"出一个等效大 batch。
>
> - **等效关系**：`等效 batch = micro_batch × accum`。例如显存只够 micro_batch=8，设 accum=8，等效就是 batch=64 的一次更新。
> - **为什么 loss 要除以 accum**：`loss = loss_fn(...) / accum`。因为 `.grad` 是**累加**的（见 [[autograd机制]] §3.4），accum 次累加会让梯度放大 accum 倍；除以 accum 后累加 = 这 accum 个小批梯度的**平均**，量级才和真大 batch 的一次平均梯度一致。不除的话等效 lr 被放大 accum 倍，训练会发散。
> - **为什么能等效**：mini-batch 梯度是样本梯度的平均，把 64 个样本拆成 8 组×8 个分别反向再相加（除以 8 后）= 64 个样本的平均梯度，数学期望和真 batch=64 一次反向一致（见本节 §4 的方差公式）。
> - **代价**（见下面 warning）：多次反向更慢；[[layer norm]] / BN 的统计仍按 micro batch 算；噪声结构和真大 batch 不同；是显存受限的**妥协**，不是完美等价。
>
> ```python
> micro_bs, accum = 8, 8            # 等效 batch = 64
> opt.zero_grad()
> for i, (x, y) in enumerate(loader_micro):
>     loss = loss_fn(model(x), y) / accum   # 必须缩放！
>     loss.backward()                        # 梯度累加进 .grad
>     if (i+1) % accum == 0:                 # 攒够 accum 次
>         opt.step()                         # 才真正更新一次参数
>         opt.zero_grad()
> ```
>
> 一句话：**accum = 凑一次参数更新所需的小反向次数**，`等效 batch = micro_batch × accum`，是"显存不够、batch 来凑"的标准手段。详见本节 §3.3、[[epoch与step]] §3.1 的 micro-step。

> [!warning] grad accum ≠ 真大 batch
> 数学上梯度期望一致，但：(1) [[layer norm]] 的运行统计、dropout 模式仍按 micro batch 算；(2) 与真大 batch 的噪声结构不同；(3) 多次反向更慢。是显存受限的妥协，不是完美等价。

### 3.4 token batch（LLM 特有）

LLM 不按"样本条数"算 batch，而按 **token 数** 算（一条样本是变长序列）。常见 `batch_size × seq_len` = 总 token 数，如 32 × 4096 = 131072 tokens/step。大模型训练常把"每 step 处理的 token 数"作为关键指标（如 0.5M tokens/step）。

### 3.5 micro-batch（流水线并行）

[[Pipeline Parallel]] 里把一个 mini-batch 再切成多个 **micro-batch**，让流水线各级交替处理，填满气泡。这是与"训练 batch"不同的工程概念。

## 4. 数学原理 / 公式

### 梯度方差与 batch

设单样本梯度 $g_i$ 独立同分布，方差 $\sigma^2$。mini-batch 平均 $\hat g=\frac1{|B|}\sum g_i$ 的方差：

$$
\text{Var}(\hat g)=\frac{\sigma^2}{|B|}
$$

batch 翻倍，梯度方差减半 → 更稳，但收益递减。

### 线性缩放法则（linear scaling rule）

大 batch 训练经验：batch 放大 $k$ 倍，lr 也放大 $k$ 倍，可保持"单样本对参数的等效移动"不变：

$$
\eta_{\text{new}} = k\,\eta_{\text{old}}
$$

因为 step 量 $\Delta\theta=-\eta\hat g$，batch 大 $k$ 倍时 $\hat g$ 方差小、量级相近，lr 放 $k$ 倍让 $\|\Delta\theta\|$ 量级匹配。但 lr 不能无限放大（有上限），故有 **critical batch size**（收益开始递减的 batch）。

### 临界 batch size

收益曲线：batch 从小增大，吞吐先线性涨（GPU 吃饱前），到 critical batch 后饱和。超过它再增大 batch 不再加速（compute-bound），只省步数但每步更贵。

## 5. 代码示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(1000, 10); Y = torch.randn(1000, 2)
loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True,
                    pin_memory=True, drop_last=True)   # drop_last 防最后不满 batch
model = nn.Linear(10, 2)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 标准 mini-batch 训练
for x, y in loader:                # 每个_batch 32 条
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()

# gradient accumulation（显存放不下大 batch 时等效大 batch）
micro_bs, accum = 8, 8             # 等效 batch=64
loader_micro = DataLoader(TensorDataset(X, Y), batch_size=micro_bs)
opt.zero_grad(set_to_none=True)
for i, (x, y) in enumerate(loader_micro):
    loss = loss_fn(model(x), y) / accum    # 缩放，使累加后量级=全 batch 平均
    loss.backward()                        # 梯度累加进 .grad
    if (i+1) % accum == 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[SGD]]（mini-batch 是 SGD 的批大小）、[[backward过程]]（每批一次反向）。
- **下游（应用）**: [[epoch与step]]（batch 决定 steps_per_epoch）、[[学习率调度]]（线性缩放、total_steps）、[[mixed precision training]]（batch 影响显存决定能否开 AMP）、[[Pipeline Parallel]]（micro-batch）。
- **对比 / 易混**:
  - **batch vs mini-batch vs micro-batch**：batch 泛指一批；mini-batch 是训练用小批；micro-batch 是流水线切分的更小单元。
  - **batch size vs sequence length（LLM）**：LLM 按 token 数算，增大 seq_len 也能"喂更多 token"但显存涨得更快（attention $O(n^2)$）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **batch 越大越好**：超过 critical batch 后不再加速，且泛化可能变差、lr 要重调。
> 2. **gradient accumulation 当真大 batch**：BN/layernorm 统计、dropout、噪声结构都不同，不完美等价。
> 3. **drop_last 忘开**：最后不满 batch 的一批样本少，梯度噪声不同，尤其带 BN 时影响大。
> 4. **batch 改了不改 lr**：违反线性缩放，大 batch 配小 lr 收敛慢、小 batch 配大 lr 发散。
> 5. **把 token batch 当样本 batch**：LLM 里 32×4096 是 32 条样本但 131072 token，比"32"大得多。
> 6. **pin_memory 不开**：CPU→GPU 拷贝慢，DataLoader 默认该开 `pin_memory=True`。

## 8. 延伸细节

### 8.1 大 batch 训练的代价

大 batch 噪声小，需更大 lr + warmup + 更长训练才能收敛到好解（LARS/LAMB 等大 batch 优化器专为超大批设计）。LLM 用百万 token batch，lr 峰值 3e-4~6e-4 + 长预热。

### 8.2 动态 batch / padding

变长序列（NLP）要 padding 到等长，或按长度分桶（bucketing）减少 padding 浪费。LLM 训练常用 packing（把多条短序列拼进一个长序列）提升 token 密度。

### 8.3 batch 与 BatchNorm

[[layer norm]] 之外的 BatchNorm 对 batch size 极敏感（小 batch 统计不稳），故 LLM 几乎不用 BN 而用 LayerNorm/RMSNorm（统计沿特征维，不依赖 batch）。

### 8.4 显存随 batch 线性增长

激活显存 $\propto$ batch × seq_len × hidden，这是 [[activation memory]] 的主项，也是 gradient checkpointing 要优化的对象。

---
相关: [[训练工程基础]]、[[SGD]]、[[epoch与step]]、[[学习率调度]]、[[mixed precision training]]
