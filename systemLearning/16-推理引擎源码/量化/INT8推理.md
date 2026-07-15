# INT8推理

> **所属章节**: [[量化]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: INT8 推理 / W8A8 / weight-only int8 / W8A16 / int8 量化推理 / 8-bit 定点推理
> **难度**: 中高（需懂 [[数值类型与精度]] + [[memory bandwidth]] + [[Roofline模型]] + 激活 outlier + [[FP8量化方案]] 对照 + 量化基本式）
> **关联**: 本笔记讲 INT8 推理的**两种模式**（weight-only int8 vs W8A8）、激活量化的 outlier 难点与 SmoothQuant、int8 GEMM kernel 与 FP8 的关系；与 [[AWQ]] / [[GPTQ]]（int4 weight-only）互补，与 [[FP8量化方案]]（浮点 8-bit）对照，同属 [[量化]] 体系。


## 1. 一句话定义

**INT8 推理** 是把 LLM 推理的**权重和/或激活从 fp16/bf16 量化到 8-bit 定点整数（int8）** 以省显存/带宽/算力的量化方案，分两种模式：① **weight-only int8（W8A16）**——只量化权重到 int8、激活仍 fp16/bf16，省权重显存与 decode 权重读取带宽（decode 是 memory-bound）；② **W8A8**——权重与激活都 int8，用 **INT8 Tensor Core GEMM**（int8 矩阵乘，吞吐约 2× bf16）省算力、提 prefill/大 batch compute-bound 场景吞吐，但需校准激活的量化参数并处理**激活 outlier**（少数通道激活幅值远大于其他，均匀 int8 量化会饱和或毁精度，用 per-token/per-channel 量化 + SmoothQuant 抑制）。INT8 是定点（均匀 256 台阶），与浮点 8-bit 的 [[FP8量化方案]]（带指数、动态范围大）形成对照——Hopper 前用 INT8，Hopper+ 偏 FP8。核心权衡：**weight-only 省带宽（decode 友好）vs W8A8 省算力（prefill 友好），精度损失靠 per-token 量化 + outlier 抑制控制**。

> [!note] 三句话定位
> - **是什么**：int8 量化推理两模式——weight-only（W8A16，省带宽）与 W8A8（权重+激活 int8，省算力用 int8 GEMM）。
> - **为什么**：decode memory-bound → weight-only 省权重带宽；prefill/大 batch compute-bound → W8A8 用 int8 Tensor Core 2× 算力。激活 outlier 是 W8A8 难点。
> - **与 [[FP8量化方案]] 关系**：INT8 定点（均匀、需反量化、动态范围小）；FP8 浮点（带指数、无需反量化、动态范围大、Hopper+ 原生）。FP8 是 INT8 的浮点替代，Hopper+ 训推通吃，INT8 在边缘/老硬件仍主流。


## 2. 为什么需要它（动机与背景）

### 2.1 两种瓶颈，两种 int8 模式

LLM 推理两阶段瓶颈不同（见 [[Roofline模型]]、[[TTFT与TPOT]]）：

- **prefill / 大 batch**：compute-bound（算术强度高），瓶颈是**算力**。W8A8 用 int8 Tensor Core GEMM（吞吐 2× bf16）直接提 prefill 吞吐。
- **decode / 小 batch**：memory-bound（算术强度 I≈2），瓶颈是**权重读取带宽**。weight-only int8 把权重字节压 2×（fp16→int8），省 decode 带宽。

故 INT8 推理分两模式对症下药。int4（[[AWQ]]/[[GPTQ]]）只 weight-only 省 4× 带宽（decode 更猛），但不能省算力（无 int4 GEMM，dequant 回 fp16 算）；int8 W8A8 兼顾省算力。

### 2.2 激活量化的难点：outlier

W8A8 要量化激活。但 LLM 激活有 **outlier（异常值）**：少数输入通道的激活幅值远大于其他（数十~百倍），均匀 per-tensor int8 量化要么为不溢出把步长 $\Delta$ 设很大（小值精度全丢），要么为保小值精度 $\Delta$ 设小（outlier 饱和截断）。这是 W8A8 精度的主要威胁，催生 per-token/per-channel 量化与 **SmoothQuant**（把激活 outlier "平滑"迁移到权重）。

### 2.3 INT8 vs FP8

INT8 是定点（8-bit 整数，均匀 256 台阶），参与浮点运算前要 `dequant = int8*scale + zero_point`（反量化开销）。[[FP8量化方案]]（E4M3/E5M2）是浮点（带指数，动态范围大），Tensor Core 原生吃（Hopper+），无需反量化查表。Hopper 前（A100 等）FP8 硬件缺失，INT8 是唯一 8-bit 加速选项；Hopper+ FP8 更友好（训推通吃），INT8 在边缘部署/老硬件/纯推理带宽优化仍主流。详见 [[FP8量化方案]] §2.1。


## 3. 核心概念详解

### 3.1 weight-only int8（W8A16）

- **量化对象**：仅权重 int8（per-channel 或 per-group），激活 fp16/bf16。
- **收益**：权重显存/带宽压 2×（fp16 2B → int8 1B + 元数据）。decode memory-bound，带宽省 2× → decode 吞吐近 2×。
- **计算**：int8 权重先 dequant 回 fp16 再做 fp16 GEMM（fused dequant+GEMM kernel，见 [[fused kernel]]）。**不**用 int8 Tensor Core（因激活是 fp16）。算力不省，省带宽。
- **校准**：静态 per-channel 量化（权重幅值固定，离线算 scale 即可），**不需**校准激活。简单。
- **精度**：int8 权重 2× 压缩，精度损失小（远好于 int4），几乎无损。
- **场景**：纯 decode 优化、边缘/低带宽硬件、不愿校准激活的简单部署。

### 3.2 W8A8（权重+激活都 int8）

- **量化对象**：权重 int8 + 激活 int8，用 **INT8 Tensor Core GEMM**（A100 624 TOPS int8 vs 312 TFLOPS bf16 = 2×；H100 类似）。
- **收益**：算力 2×（prefill/compute-bound 提速），权重带宽也省 2×。
- **校准**：权重 per-channel 静态（离线）；**激活需校准**（per-token 动态或 per-tensor 静态），因激活幅值运行时变。
- **难点**：激活 outlier（见 §3.3）。
- **计算**：$Y = (X_{int8}\cdot\Delta_X)(W_{int8}\cdot\Delta_W)$，int8 GEMM 算 $\text{int32} = X_{int8}W_{int8}$，再乘 $\Delta_X\Delta_W$ 缩放（dequant），通常 GEMM kernel 内部累加 int32 后乘 scale 出 fp16/bf16。
- **场景**：prefill 重 / 大 batch 服务 / compute-bound 高吞吐。

### 3.3 激活 outlier 与应对

LLM 激活（尤其 FFN/attention 后、某些层）有 outlier：少数通道幅值远大。应对：

- **per-token / per-channel 量化**：每个 token（或每个输出通道）各算一组 scale，让大值通道用大 scale 不影响小值通道。per-token 是激活（每 token 一 scale），per-channel 是权重（每输出通道一 scale）。比 per-tensor 精度好得多。
- **SmoothQuant**（Xiao et al.）：把激活的 outlier"迁移"到权重——对显著通道 $i$，激活除 $s_i$（压小 outlier）、权重乘 $s_i$（权重本就幅值均匀，放大无损），$y=(x/s)(s\odot W)$ 不变，但激活幅值变平坦 → per-tensor int8 也够精度。免动态 per-token，全静态 W8A8。
- **mixed-precision**：outlier 通道保 fp16，其余 int8（分支 kernel，复杂）。

### 3.4 INT8 GEMM（int8 矩阵乘）

W8A8 的核心算子是 **INT8 Tensor Core GEMM**：$C_{int32} = A_{int8}\times B_{int8}$（累加用 int32 防溢），输出乘 $\Delta_A\Delta_B$ dequant 回 fp16。cuBLAS/cutlass 提供 `int8Gemm`。A100/H100 Tensor Core 原生支持 int8。是 [[fused kernel]]（dequant+GEMM 融合）的典型。注意 int8 GEMM 要处理 zero-point（asymmetric）或 symmetric。

### 3.5 与 FP8 的关系

| | INT8（定点） | FP8（浮点，[[FP8量化方案]]） |
|---|---|---|
| 表示 | 8-bit 整数，均匀 256 台阶 | 8-bit 浮点（E4M3/E5M2），带指数 |
| 动态范围 | 小（均匀，大值易饱和） | 大（指数自适应，小值小台阶大值大台阶） |
| 反量化 | 需 `int8*scale+zp` 查表/乘 | 无需（浮点直接喂 Tensor Core） |
| 激活 outlier | 敏感（需 SmoothQuant/per-token） | 友好（指数吸收大值） |
| 硬件 | A100+ INT8 Tensor Core | Hopper+ FP8 Tensor Core（A100 无） |
| 训练 | 难（定点梯度表示差） | 友好（TE 训练主流） |
| 推理 | 老硬件/边缘主流 | Hopper+ 训推通吃 |

FP8 是 INT8 的浮点"升级"——动态范围更好、训推统一、免反量化。Hopper+ FP8 逐步替代 W8A8 INT8（训练侧 FP8 已主流，推理侧 FP8 与 INT8 并存）。详见 [[FP8量化方案]]。


## 4. 数学原理 / 公式

### 4.1 int8 量化基本式（asymmetric，带 zero point）

$$\Delta = \frac{x_{\max}-x_{\min}}{2^b-1},\quad z = \text{round}\!\left(-\frac{x_{\min}}{\Delta}\right),\quad x_{int}=\text{clamp}\!\left(\text{round}\!\left(\frac{x}{\Delta}\right)+z,\,0,\,2^b-1\right)$$

$b=8$，$x_{int}\in[0,255]$。反量化 $\hat x = (x_{int}-z)\Delta$。symmetric 则 $z=0$，$x_{int}\in[-127,127]$。

### 4.2 W8A8 GEMM 的 scale 缩放

设激活 $X$ 量化 scale $\Delta_X$（per-token，每个 token 一值），权重 $W$ scale $\Delta_W$（per-channel，每输出通道一值）。int8 GEMM 算 int32 累加：

$$C_{int32} = X_{int8}\,W_{int8}\in\mathbb{Z}^{n\times d_{out}}$$

dequant：$Y \approx C_{int32}\odot(\Delta_X\,\Delta_W^\top)$（$\Delta_X$ 广播行、$\Delta_W$ 广播列）。per-token + per-channel 组合是 W8A8 标配。

### 4.3 per-tensor vs per-token 的 outlier 误差

per-tensor 单 scale $\Delta=\max|X|/127$。若 outlier 幅值 $M_o\gg$ 其余 $M_n$，则 $\Delta\approx M_o/127$ 极大，普通通道（幅值 $\sim M_n$）量化后 $M_n/\Delta\sim M_n\cdot127/M_o\ll1$ → 截 0，**精度全毁**。per-token 每 token 一 $\Delta$，outlier token 自己大 scale，不影响其他 token。误差从"全局毁"降到"单 token 内仍存但隔离"。

### 4.4 SmoothQuant 的迁移推导

设激活 $x$、权重 $W$，逐通道 scale $s$（显著通道 $s_i>1$）：

$$y = xW = (x\oslash s)(s\odot W) = x'W',\quad x'=x\oslash s,\;W'=s\odot W$$

浮点不变。但 $x'$ 的 outlier 被压（除 $s_i$），幅值平坦 → per-tensor int8 也够；$W'$ 放大但权重本就幅值均匀、放大无损。**把激活难量化（outlier）转嫁给权重好量化（均匀）**。$s$ 据激活幅值离线搜（per-channel，静态 W8A8）。误差分析与 [[AWQ]] §4.2 镜像（AWQ 是放大权重保护，SmoothQuant 是缩小激活抑 outlier，方向相反目的一致）。


## 5. 代码示例（可选）

### 5.1 手写 weight-only int8 vs W8A8 + per-token 量化对比

```python
import torch

def quant_int8_sym(x):
    """per-tensor symmetric int8 量化, 返回 int8 与 scale."""
    delta = x.abs().max() / 127.0
    q = torch.round(x / delta).clamp(-128, 127).to(torch.int8)
    return q, delta

def dequant(q, delta):
    return q.to(torch.float32) * delta

def gemm_int8_w8a8(X, W):
    """W8A8: 激活 per-token int8, 权重 per-channel int8, int8 GEMM + scale 缩放."""
    # 激活 per-token (每行一 scale)
    Xt = X.t()                                   # [d_in, n]
    dx = Xt.abs().amax(dim=0, keepdim=True) / 127  # [1, n]
    Xi = torch.round(Xt / dx).clamp(-128,127).to(torch.int8)
    # 权重 per-channel (每输出列一 scale)
    dw = W.abs().amax(dim=0, keepdim=True) / 127   # [1, d_out]
    Wi = torch.round(W / dw).clamp(-128,127).to(torch.int8)
    # int8 GEMM (这里用 int32 累加模拟, 真实走 Tensor Core)
    Ci = (Xi.to(torch.int32) @ Wi.to(torch.int32))  # [d_in, d_out] int32
    Y = Ci.to(torch.float32) * (dx.t() * dw)        # dequant (scale 缩放)
    return Y.t()                                   # [n, d_out]

def gemm_weightonly_int8(X, W):
    """weight-only int8 (W8A16): 权重 int8 per-channel, 激活 fp16, dequant+GEMM."""
    dw = W.abs().amax(dim=0, keepdim=True) / 127
    Wi = torch.round(W / dw).clamp(-128,127)
    W_hat = Wi * dw                                 # 反量化回 fp
    return (X.to(torch.float32) @ W_hat)

torch.manual_seed(0)
n, d_in, d_out = 128, 512, 512
W = torch.randn(d_in, d_out) * 0.05
X = torch.randn(n, d_in); X[:, ::64] *= 20          # 制造 outlier 通道
y_true = X @ W
print("W8A8 (per-token) MSE:     ", ((gemm_int8_w8a8(X, W) - y_true)**2).mean().item())
print("weight-only int8 MSE:      ", ((gemm_weightonly_int8(X, W) - y_true)**2).mean().item())
```

### 5.2 vLLM / TRT-LLM INT8 配置

```bash
# vLLM: weight-only int8 (W8A16, 静态 per-channel, 无需校准激活)
python -m vllm.entrypoints.openai.api_server \
    --model <fp16-model> --quantization fp8     # 注: vLLM int8 weight-only 多走 bitsandbytes

# vLLM: W8A8 (需校准激活, 用 SmoothQuant 或 INT8 calibration)
#   通常离线用 llm-compression/AutoAWQ/AutoGPTQ 产 int8 校准表, 或 TRT-LLM build
```

```python
# llm-compressor (vLLM 生态) 跑 W8A8 INT8 + SmoothQuant
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
# SmoothQuant 迁移 outlier + int8 静态量化
oneshot(model=model, \
        recipe=GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]))
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[数值类型与精度]]（int8 位级、定点 vs 浮点）、[[mixed precision training]]（量化是 AMP 之外的压低）、[[memory bandwidth]] / [[Roofline模型]]（两模式对症 decode/prefill 瓶颈）、激活 outlier / SmoothQuant。
- **下游（应用）**: [[新模型接入]]（vLLM/TRT-LLM int8 配置）、[[model runner]]（int8 GEMM fused kernel）、[[sampling throughput]]（W8A8 提 prefill 吞吐、weight-only 提 decode）、[[Tensor Parallel]]（int8 权重 TP 切）、KV cache 量化（int8 KV cache 省显存，见 [[KV block manager]]）。
- **对比 / 易混**:
  - **weight-only int8 vs W8A8**：见下表。前者省带宽（decode），后者省算力（prefill）。
  - **INT8 vs [[AWQ]]/[[GPTQ]] int4**：int4 weight-only 省 4× 带宽（decode 更猛）但无 int4 GEMM（不省算力）；int8 W8A8 省 2× 算力 + 2× 带宽，精度更好。int4 用于极致 decode，int8 用于均衡/算力。
  - **INT8 vs [[FP8量化方案]]**：定点 vs 浮点，见 §3.5 表。Hopper+ FP8 替代 W8A8。
  - **INT8 vs [[FP4与低比特]]**：4-bit 省 2× 于 int8 但精度降更多、无 GEMM。int8 是 8-bit 甜点。

### weight-only int8 vs W8A8 对比表

| 维度 | weight-only int8 (W8A16) | W8A8 |
|---|---|---|
| 量化对象 | 仅权重 int8，激活 fp16 | 权重+激活都 int8 |
| 算力 | 不省（fp16 GEMM） | 省 2×（int8 Tensor Core） |
| 带宽 | 省 2×（权重字节减半） | 省 2×（同） |
| 主受益阶段 | decode（memory-bound） | prefill / 大 batch（compute-bound） |
| 校准激活 | 不需（权重静态） | 需（激活 outlier） |
| 精度 | 几乎无损 | 需 per-token/SmoothQuant 控误差 |
| 复杂度 | 低 | 中（校准 + outlier 处理） |
| 硬件 | A100+（fp16 GEMM + dequant） | A100+ INT8 Tensor Core |


## 7. 常见误区与易错点

> [!warning] 误区 1：int8 推理一定省算力
> 不一定。**只有 W8A8**（激活也 int8、走 int8 Tensor Core GEMM）才省算力 2×。**weight-only int8**（W8A16）激活仍是 fp16，算 fp16 GEMM，**不省算力**，只省权重带宽。把两者混为一谈会高估 weight-only 的算力收益。

> [!warning] 误区 2：INT8 和 FP8 等价
> 不等价。INT8 定点（均匀 256 台阶、需反量化、outlier 敏感）；FP8 浮点（带指数、动态范围大、Tensor Core 原生、训推友好）。Hopper+ FP8 训推通吃，INT8 在老硬件/边缘仍主流。详见 [[FP8量化方案]]。

> [!warning] 误区 3：W8A8 不需校准激活
> 需。W8A8 激活 int8，激活幅值运行时变，要校准（per-token 动态或 SmoothQuant 静态）。weight-only 才不需校准激活（权重固定）。误以为 W8A8 零校准会精度崩（outlier 毁）。

> [!warning] 误区 4：per-tensor 够 W8A8
> 不够。LLM 激活有 outlier，per-tensor 单 scale 会让小值通道精度全毁。必须 per-token（激活）+ per-channel（权重），或 SmoothQuant 抑 outlier。per-tensor 只在无 outlier 的简单网络够。

> [!warning] 误区 5：int8 GEMM 输出 int8
> 不是。int8 GEMM 内部累加用 **int32**（防 int8×int8 溢出），输出乘 scale dequant 回 fp16/bf16。int8 是输入与权重，不是全程 int8。INT8 Tensor Core 输入 int8、累加 int32、输出 fp16。

> [!tip] 实践：decode 选 weight-only / int4，prefill 选 W8A8/FP8
> 纯 decode 服务（小 batch、低延迟）选 weight-only int8 甚至 int4（[[AWQ]]/[[GPTQ]]），省带宽。prefill 重 / 大 batch 高吞吐选 W8A8 或 FP8，省算力。混合负载可 int4 权重 + FP8 激活（W4A8）。

> [!tip] 实践：KV cache 也可 int8
> KV cache 量化到 int8（或 fp8）省 2× 显存 → 增并发 → 提吞吐。K 量化误差被 attention score $QK^\top$ 放大，对精度敏感，常 mixed-precision（K 保 fp16、V int8）。见 [[KV block manager]]。


## 8. 延伸细节

### 8.1 W4A8（int4 权重 + int8 激活）混合

极致优化：权重 int4（省 4× 带宽，decode）、激活 int8（省 2× 算力，prefill），用 int8 GEMM + int4 权重 dequant。精度靠 AWQ/GPTQ 保护权重 + SmoothQuant 抑激活 outlier。vLLM/llm-compressor 支持。是 int4 与 int8 的折中。

### 8.2 硬件 INT8 Tensor Core 算力（参考）

A100：bf16 312 TFLOPS，int8 624 TOPS（2×）。H100 SXM5：bf16/fp16 ~1979 TFLOPS，int8 ~3958 TOPS（2×）。INT8 = 2× bf16 是 8-bit 定点的算力红利（待核实具体 SKU 数字，以 NVIDIA datasheet 为准）。FP8 在 H100 与 int8 同档（~1979/3958），Blackwell FP4 再 2× 于 FP8（见 [[FP4与低比特]]）。

### 8.3 内容来源

INT8 推理整理自 SmoothQuant（Xiao et al. 2022）、TensorRT-LLM INT8/SmoothQuant 文档、vLLM `--quantization` 与 llm-compressor `scheme="W8A8"`、cuBLAS int8 GEMM、[[FP8量化方案]] 对照。算力数字来自 NVIDIA A100/H100 datasheet（标"待核实"者以官方为准）。

---
相关: [[量化]] | [[AWQ]] | [[GPTQ]] | [[FP4与低比特]] | [[FP8量化方案]] | [[数值类型与精度]] | [[mixed precision training]] | [[memory bandwidth]] | [[Roofline模型]] | [[新模型接入]] | [[model runner]] | [[Tensor Parallel]] | [[fused kernel]] | [[sampling throughput]] | [[loss scaling]] | [[KV block manager]]
