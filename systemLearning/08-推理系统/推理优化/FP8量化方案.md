# FP8量化方案

> **所属章节**: [[推理优化]]
> **所属模块**: [[08-推理系统]]
> **别名**: FP8 quantization / FP8 量化 / 八位浮点量化 / E4M3 / E5M2 / MXFP8 / delayed scaling / block scaling
> **难度**: 中高（需懂 [[数值类型与精度|浮点格式]] + Transformer 训练/推理精度路径）
> **关联**: 本笔记讲"FP8 的不同量化方案/格式/缩放策略"，是 [[训推不一致]] 里"精度路径"来源的具体展开；与 [[数值类型与精度]] 的 FP8 位级结构互补（那篇讲位怎么排，本篇讲怎么用）。


## 1. 一句话定义

**FP8 量化方案**指把训练/推理中的张量（权重、激活、梯度、KV cache）从 BF16/FP16 表示成 **8 位浮点** 后，**用哪种 FP8 浮点格式（E4M3 / E5M2）、哪种缩放策略（per-tensor / per-block / per-channel / delayed / MX block）、哪个范围/方向（前向权重激活 vs 反向梯度）** 的具体组合。**同一个"FP8"在不同框架/版本/硬件下的方案并不相同**——这正是 [[训推不一致]] 里"精度路径"行写"fp8 的不同量化方案"的根因：训练侧用 Transformer Engine 的 E4M3 per-tensor delayed scaling，推理侧用 vLLM/TRT-LLM 的 E4M3 per-token/per-tensor，两套方案即便权重相同，算出的 logits 也有数值差。

> [!note] 三句话定位
> - **是什么**：FP8 不是一种格式，而是一族"格式 × 缩放"的方案组合。
> - **为什么**：8 位浮点比 BF16 省 2× 显存/带宽、比 INT8 不需要复杂的反量化查表（浮点自带指数），是 Hopper 起的训练/推理通用甜点精度。
> - **代价**：格式选错（E5M2 用于前向会精度不够）或缩放粒度不对（per-tensor 对激活太粗）会引入误差，且训推方案不一致 → TIM。


## 2. 为什么需要它（动机与背景）

### 2.1 为什么是 FP8 而非 INT8

INT8 量化在 LLM 上有几个痛点：
- **反量化开销**：INT8 是定点，参与 float 运算前要先 `dequant = int8 * scale + zero_point`，反量化是额外算子；
- **大动态范围难覆盖**：attention scores、梯度等张量动态范围大（数十个量级），定点 INT8 只有 256 个台阶且均匀分布，要么溢出要么精度塌；
- **KV cache 量化敏感**：$K$ 量化误差被 attention score $QK^\top$ 放大，INT8/INT4 常需 mixed-precision 细调。

**FP8 是浮点**：1 位符号 + 指数 + 尾数，指数让它**自动覆盖大动态范围**（小数用小台阶、大数用大台阶），**无需反量化查表**可直接喂进 Tensor Core（Hopper FP8 Tensor Core 原生支持）。所以 FP8 比 INT8 更"drop-in"、对训练更友好，是 2023+ H100/Blackwell 训练与推理的事实标准。

### 2.2 为什么"方案"会不一致

FP8 标准其实给了**两种格式**（E4M3、E5M2），加上**缩放粒度**和**应用范围**是各家自选，导致：
- 训练（NVIDIA Transformer Engine）默认：前向权重/激活 E4M3 + per-tensor delayed scaling，反向梯度 E5M2；
- 推理（vLLM、TRT-LLM）默认：权重/激活 E4M3，常见 per-tensor 或 per-token，KV cache 可选 E4M3 FP8；
- 新标准 **OCP MXFP8**（Open Compute Project Microscaling）：E4M3 + **32 个元素共享一个 scale 的 block scaling**（来自 OCP 的 MX format 规范），Blackwell 起原生支持，和上面的"per-tensor delayed"又不同。

这些方案**数学上不等价**，所以训推各用一套 → logp 不等 → TIM。


## 3. 核心概念详解

### 3.1 两种 FP8 浮点格式

| 格式 | 符号位 | 指数位 | 尾数位 | 指数偏置 | 最大正数 | 最小正正规数 | 适用 |
|---|---|---|---|---|---|---|---|
| **E4M3**（fnuz 变体） | 1 | 4 | 3 | 7 | ≈ 448 | ≈ $2^{-6}$ ≈ 0.0156 | **前向**（权重/激活）：精度优先，范围够 |
| **E5M2** | 1 | 5 | 2 | 15 | ≈ 57344 | ≈ $2^{-14}$ ≈ $6.1\times10^{-5}$ | **反向梯度**：范围优先（梯度动态范围大） |

> [!note] 关键直觉
> - **E4M3**：指数少（4 位 → 16 个台阶）、尾数多（3 位 → 8 个台阶/数），**精度高、范围小**。前向的权重和激活通常数值分布集中、范围可控，用 E4M3 吃精度。
> - **E5M2**：指数多（5 位 → 32 个台阶）、尾数少（2 位 → 4 个台阶/数），**范围大、精度低**。反向梯度数值跨度可达数十量级（接近 0 的梯度要表示，又不能溢出），用 E5M2 吃范围防溢出。
> - 这是 NVIDIA TE 的**默认分工**：前向 E4M3 + 反向 E5M2，两格式混用一把覆盖训推。

> [!warning] 误区：以为"FP8"只有一个格式
> 不。**E4M3 和 E5M2 是两个不同的 8 位浮点**，位级布局不同、表示的数集不同，混用要明确每张量用哪个。文档里写"FP8 training"通常指 E4M3/E5M2 混合；写"FP8 KV cache"通常指 E4M3。算 logp 时训推若一个用 E4M3 一个用 E5M2 算同一张量 → 直接错。

### 3.2 缩放策略（scaling）：怎么把 BF16 数塞进 FP8 的有限范围

FP8 能表示的 max 是 448（E4M3）。但 BF16 权重可能 >448 或 <0.0156，直接转必溢出/截断。**缩放**就是乘个 scale 把张量"拉"进 FP8 可表示范围，用完再除回来。

数学：量化 $x_{fp8} = \text{round}(x / s)$ 到 FP8，反量化 $x \approx x_{fp8} \times s$，$s$ 是缩放因子。**问题**：一个 scale 给整个张量用，还是给每个小块各用一个？

| 缩放粒度 | 一个 scale 管多大 | 优点 | 缺点 | 代表实现 |
|---|---|---|---|---|
| **per-tensor** | 整个张量 1 个 scale | 最省显存、scale 只 1 个标量 | 粒度粗，张量内有大数有小数时小数被压成 0 | NVIDIA TE 早期默认、TRT-LLM 权重 |
| **per-channel** | 每输出通道 1 个 scale | 权重维度天然分组，精度好 | 激活没"通道"概念不适用 | 训练权重常见 |
| **per-token** | 每个 token 1 个 scale | 激活按 token 切，对长序列/异常 token 友好 | scale 数量 = batch×seq | vLLM/TRT-LLM 激活、KV cache |
| **per-block (MX)** | **每 32 元素 1 个 scale**（MXFP8） | 粒度细、精度高、scale 量可控 | 需硬件支持 block scaling | OCP MXFP8 规范、Blackwell |
| **delayed scaling** | 用历史若干步的 amax 统计延迟更新 scale | 免每步重算 amax，训练快 | 启动期/数值突变期 scale 滞后 | NVIDIA TE 默认 `FP8DiamondithDelayedScaling` |

> [!tip] delayed scaling 是训练专属的工程优化
> 严格 per-tensor 要每步算当前张量的 $\max|x|$（amax）当 scale，但每步重算 amax 有开销。**delayed scaling**：用前 $N$ 步的 amax 滑动平均/最大值当本步的 scale，scale 更新"延迟"一步。这免了 amax kernel 但引入"scale 滞后"——数值突变时（如训练初期、loss spike）scale 不准 → 量化误差大。推理没有"多步"概念，不做 delayed scaling，直接 per-tensor/per-token 算 amax。**这条差异本身就是 TIM 来源**：训练 scale 是历史延迟值，推理是即时值。

### 3.3 主流 FP8 方案对照

| 方案 | 格式 | 缩放 | 出处 | 典型场景 |
|---|---|---|---|---|
| **NVIDIA Transformer Engine（TE）** | 前向 E4M3 / 反向 E5M2 | per-tensor + delayed scaling | NVIDIA，H100+ 训练主力 | Megatron-LM / NeMo 训练 |
| **vLLM FP8**（权重/激活） | E4M3 | 权重 per-channel / 激活 per-token | vLLM `quantization="fp8"` | 推理 |
| **vLLM FP8 KV cache** | E4M3 | per-tensor 或 per-token | `kv_cache_dtype="fp8"` | 推理 KV cache（见 [[KV cache management]] §3.4） |
| **TensorRT-LLM FP8** | E4M3 | per-tensor / per-channel | NVIDIA 推理 | 极致低延迟推理 |
| **OCP MXFP8**（Microscaling FP8） | E4M3 | **per-block 32 元素共享 scale** | OCP MX 规范 v1.3 | Blackwell 原生，未来训推统一 |
| **AMD/其他** | E4M3/E5M2 变体 | 各家 | ROCm 生态 | MI300 等 |

### 3.4 与训推不一致的耦合点

[[训推不一致]] §3.1 表里"精度路径"行指的就是**这些方案的差异**，具体落到几处：

1. **格式不同**：训练某层可能反向用 E5M2 算梯度相关张量，推理只做前向用 E4M3，前向 logp 计算路径上一致（都 E4M3）但**累计的舍入误差不同**（训练前向多步累积）。
2. **缩放粒度不同**：训练 per-tensor delayed scale vs 推理 per-token 即时 scale → 同一激活量化结果不同 → 矩阵乘微差 → logp 差。
3. **scale 更新时机不同**：训练 delayed（用历史 amax），推理即时（用当前 amax）→ scale 值本身就不等。
4. **KV cache FP8 化是推理独有**：训练无 KV cache，推理若开 `kv_cache_dtype="fp8"`，历史 K/V 被量化引入误差，attention score 累积放大 → 训推 logp 差更大（[[训推不一致]] 表里 KV cache 量化行严重度标"高"正是此故）。


## 4. 数学原理 / 公式

### 4.1 FP8 量化-反量化基本式

量化（BF16 → FP8）：

$$
x_{fp8} = \text{Quant}_{E4M3}\!\left(\frac{x}{s}\right), \quad s = \frac{448}{\max|x|} \cdot \alpha
$$

其中 $448$ 是 E4M3 的最大可表示正数，$\alpha \in (0,1]$ 是安全余量（防溢出，常用 0.9–1.0）。反量化（FP8 → BF16 参与计算）：

$$
\hat x = x_{fp8} \times s
$$

误差 $|x - \hat x| \le$ 半个 FP8 台阶（在 $x$ 所在量级）。

### 4.2 per-tensor vs per-block 的精度差异

设张量 $X \in \mathbb{R}^{N}$，最大元素 $\max|x_i| = M$，但有少数大元素主导 $M$，多数元素 $|x_i| \ll M$。

- **per-tensor**：一个 scale $s = 448/M$。小元素 $|x_i| \ll M$ 量化后 $x_i/s = |x_i| \cdot M/448 \ll 448$，可能落入 FP8 最小台阶以下被截成 0 → **相对误差巨大**。
- **per-block（MXFP8，32 元素一组）**：每块自己算 $\max$，小元素所在的块 $s$ 小 → $x_i/s$ 落在 FP8 有效台阶 → **相对误差小**。

代价是每 32 元素存一个 scale（额外 2 字节/32 元素 = 6.25% overhead）。

### 4.3 delayed scaling 的 scale 更新

$$
s_t = f\left(\max_{i \in [t-K, t-1]} \text{amax}_i\right), \quad \text{amax}_i = \max|x|_{\text{step } i}
$$

即第 $t$ 步用的 scale 是过去 $K$ 步 amax 的函数（max 或滑动平均）。本步真实 amax $\text{amax}_t$ 不参与 $s_t$（延迟）。若 $\text{amax}_t > s_t$ 对应的物理范围 → 本步张量**溢出**（被 saturate 截断到 448），引入大误差。这正是 delayed scaling 在数值突变期（loss spike、训练初期 warmup）不稳的根因。


## 5. 代码示例（可选

### 5.1 用 NVIDIA Transformer Engine 做一次 FP8 前向（看 E4M3 + delayed scaling）

```python
# pip install transformer_engine[pytorch]
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# TE 的 Linear 自带 FP8 路径，在 autocast 上下文里启用
model = te.Linear(4096, 4096, bias=False).cuda()
x = torch.randn(8, 4096, device="cuda")

# FP8 autocast：指定格式 + 缩放策略
recipe = DelayedScaling(
    fp8_format=Format.HYBRID,   # 前向 E4M3 / 反向 E5M2，TE 默认混合
    amax_history_len=16,         # delayed scaling 用过去 16 步 amax
    amax_compute_algo="max",     # 取历史窗口内的 max
)
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = model(x)                  # 内部权重/激活走 FP8 Tensor Core
print("out dtype:", y.dtype, "(TE 在 ctx 内用 FP8 计算, 出 ctx 自动回 BF16)")
```

### 5.2 vLLM 里开 FP8 KV cache（推理侧方案）

```python
from vllm import LLM
# 权重 FP8(per-channel) + KV cache FP8(per-tensor)
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    quantization="fp8",            # 权重/激活 E4M3 per-channel
    kv_cache_dtype="fp8",          # KV cache E4M3 per-tensor（或 "fp8_e4m3" 显式）
)
out = llm.generate(["用 FP8 KV cache 推理"], use_tqdm=False)
```

### 5.3 手写一个 per-tensor vs per-block FP8 量化对比（看清精度差）

```python
import numpy as np

def fp8_e4m3_quantize(x, scale):
    """简化 E4M3 量化：除以 scale, clip 到 [-448,448], 取整."""
    q = np.clip(x / scale, -448, 448)
    return np.round(q)          # 真实 FP8 还要做指数/尾数取舍，这里近似成 round

x = np.array([0.001, 0.002, 50.0, 100.0])     # 一大数主导 + 小数

# per-tensor: 整个张量一个 scale = 448 / max|x|
s_pt = 448 / np.abs(x).max()
q_pt = fp8_e4m3_quantize(x, s_pt) * s_pt
print("per-tensor:", q_pt)     # 小数 0.001 被压成 0 (相对误差 100%)

# per-block (2 元素一组): 各组自己 scale
for i in range(0, len(x), 2):
    block = x[i:i+2]
    s_blk = 448 / np.abs(block).max()
    q_blk = fp8_e4m3_quantize(block, s_blk) * s_blk
    print(f"block {i}: {q_blk}")   # 小数组内 scale 小, 0.001 保得住
```

### 5.4 显存/带宽收益速算

```python
# FP16 KV cache -> FP8 KV cache 显存减半
def kv_mem(seq, batch, layers, kv_heads, head_dim, bytes_per):
    return 2 * layers * kv_heads * head_dim * seq * batch * bytes_per

fp16 = kv_mem(8192, 64, 32, 8, 128, 2) / 1e9   # ~2.1 GB
fp8  = kv_mem(8192, 64, 32, 8, 128, 1) / 1e9   # ~1.05 GB
print(f"FP16 KV: {fp16:.2f} GB  ->  FP8 KV: {fp8:.2f} GB  (省 {(1-fp8/fp16)*100:.0f}%)")
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[数值类型与精度]]（FP8 的位级结构 E4M3/E5M2 在那篇详述，本篇讲怎么用）、[[mixed precision training]]（FP8 是 BF16 AMP 的进一步量化）、Transformer Engine / Blackwell 硬件（FP8 Tensor Core 原生支持）。
- **下游（应用）**: [[训推不一致]]（本笔记是其中"精度路径"行的展开，FP8 方案差异是 TIM 主因之一）、[[KV cache management]] §3.4（FP8 KV cache 是推理显存省 2× 的主流手段）、[[量化]]（待建总览，FP8 是浮点量化分支，INT8/INT4 是定点量化分支）。
- **对比 / 易混**:
  - **FP8 vs INT8**：浮点 vs 定点。FP8 浮点自带指数覆盖大动态范围、无需反量化查表、对训练友好；INT8 定点均匀台阶、需反量化、对大动态范围张量差。LLM 训推 Hopper+ 偏 FP8，边缘/老硬件偏 INT8。
  - **FP8 vs BF16**：BF16 是训练"主精度"，FP8 是"量化精度"。BF16 直接参与反传 master 累积，FP8 用于权重/激活/梯度的低比特表示，精度更低但显存/带宽更省。
  - **E4M3 vs E5M2**：见 §3.1，前向精度优先用 E4M3，反向范围优先用 E5M2。
  - **per-tensor vs per-block(MX)**：见 §4.2，per-block 精度更高但需硬件支持（Blackwell）。
  - **FP8 KV cache 量化 vs FP8 权重/激活量化**：前者是推理 KV cache 的低比特存储（历史 K/V），后者是当前前向计算的低比特；KV 量化误差被 attention 累积放大，对精度更敏感。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为"FP8"是一种格式
> 不是。E4M3 和 E5M2 是两个不同的 8 位浮点，位级布局不同。文档/代码里写"FP8"必须问清是 E4M3 还是 E5M2 还是混合。训推各用一种 → TIM。

> [!warning] 误区 2：以为 E5M2 能用于前向
> E5M2 尾数只有 2 位（4 个台阶/数），前向权重/激活用它精度严重不足（数值台阶稀疏）。E5M2 是为反向梯度的大动态范围设计的，前向应始终用 E4M3。

> [!warning] 误区 3：以为 per-tensor 够用
> per-tensor 对激活/梯度这种少数大元素主导的张量，会把多数小元素压成 0。激活应至少 per-token，权重 per-channel，新硬件上优先 per-block(MXFP8)。

> [!warning] 误区 4：以为 delayed scaling 没代价
> delayed scaling 省了每步 amax kernel，但 scale 滞后——数值突变期（loss spike、warmup、学习率骤变）scale 不准会溢出/截断。训练监控要盯 FP8 overflow/underflow 计数器，异常多时降 history_len 或切即时 amax。

> [!warning] 误区 5：以为训推都开 FP8 就一致了
> 都开"FP8"≠ 一致。训练 TE 的 E4M3+per-tensor+delayed 与推理 vLLM 的 E4M3+per-token+即时，方案不同 → logp 仍差。要消 TIM 需训推**方案级对齐**（格式+粒度+scale 更新策略全对齐），详见 [[训推不一致]] §消除手段。

> [!warning] 误区 6：以为 FP8 KV cache 无损
> FP8 KV cache 几乎无损是相对 INT8 而言，长序列下 K 量化误差经 attention 累积仍可见。对极长上下文/高精度要求场景，用 mixed-precision（K 保 BF16、V 量化）或干脆全 BF16。


## 8. 延伸细节

### 8.1 OCP Microscaling Formats（MX）与 MXFP8

Open Compute Project 的 **Microscaling Formats (MX) v1.3** 规范定义了一族"小块共享 scale"的低比特格式，**MXFP8 = E4M3 + 每 32 元素一个 8 位共享 scale**。它把缩放粒度从 per-tensor 下沉到 per-block-32，兼顾精度和 scale 存储开销。NVIDIA Blackwell（B100/200）原生支持 MXFP8 Tensor Core，AMD MI300 也支持。**MXFP8 是训推统一的最大希望**——同一规范、同一位布局、同一缩放粒度，训推都用它即可消除"方案不一致"这层 TIM。但截至 2026 仍在推广期，TE/vLLM 默认还是各自的老方案。

### 8.2 NVIDIA Transformer Engine 的演进

- **TE v1.x**：per-tensor + delayed scaling，H100 默认，简单但激活精度受限；
- **TE v2.x**：支持 per-tensor / per-channel 权重、per-token 激活，精度提升；
- **TE + Blackwell**：支持 MXFP8 per-block scaling，进一步逼近 BF16 精度。

### 8.3 FP8 训练的 amax 与 overflow 监控

TE 暴露 `fp8_meta` 张量记录每步 amax、overflow 计数。训练脚本应打印：
- `fp8_meta.amax_history`：各张量的 amax 历史，看是否突变；
- overflow/underflow 计数：持续增长说明 scale 不准，需调 `amax_history_len` 或降余量 $\alpha$。

### 8.4 与 [[mixed precision training]] 的层级关系

精度栈从高到低：**FP32 master → BF16 计算 → FP8 量化**。FP8 不是替代 BF16，而是在 BF16 计算路径的**特定张量**（权重/激活/梯度/KV）上再压一档。master 权重始终 FP32，FP8 不碰 master。所以"FP8 训练"本质是"BF16 AMP + 选择性 FP8 量化"，不是纯 FP8 训练。

### 8.5 何时不上 FP8

- **小模型/短序列**：BF16 显存够用，FP8 的精度/工程复杂度不划算；
- **极高精度要求**（如对数概率精确输出、评测基准）：FP8 引入的微差会影响可比性，留 BF16；
- **老硬件**（A100 无 FP8 Tensor Core）：只能 INT8 或留 BF16，FP8 是 Hopper+ 特性。

---
相关: [[推理优化]] | [[训推不一致]] | [[数值类型与精度]] | [[mixed precision training]] | [[KV cache management]] | [[量化]]
