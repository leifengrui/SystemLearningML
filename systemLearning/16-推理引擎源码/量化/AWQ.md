# AWQ

> **所属章节**: [[量化]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: AWQ / Activation-aware Weight Quantization / 激活感知权重量化 / activation-aware weight only quantization
> **难度**: 中高（需懂 [[数值类型与精度]] + [[mixed precision training]] + [[memory bandwidth]] + [[Roofline模型]] + 量化基本式）
> **关联**: 本笔记讲 AWQ 这一**权重低比特量化方法**的原理（per-channel scaling 保护显著通道）与工程集成；与 [[GPTQ]]（另一主流 int4 PTQ）互为对照，与 [[INT8推理]] / [[FP4与低比特]] 同属 [[量化]] 体系。


## 1. 一句话定义

**AWQ（Activation-aware Weight Quantization，激活感知的权重量化）** 是一种 **weight-only 的后训练量化（PTQ）方法**——它不训练、不反向传播，只用一小批校准数据（128~512 样本）前向统计激活幅值，**找出"显著权重"（salient weights，即对应激活幅值大的输入通道的那些权重列）**，通过 **per-channel scaling**（给显著通道的权重乘一个大 scale 抬高再量化、激活除以同 scale，浮点下乘积不变但量化相对误差被压低）保护这些少数但关键的权重，把其余权重压到 **int4 / int3**，从而在几乎不损模型精度（<1% perplexity 退化）的前提下把 LLM 权重显存压到 1/4、decode 阶段（memory-bandwidth-bound）的权重读取带宽省 4×，是 vLLM / TRT-LLM / SGLang 上 7B~70B 模型 int4 推理事实标准之一。核心洞见：**不是所有权重同等重要——保护 1% 的显著权重就能保住绝大部分精度**。

> [!note] 三句话定位
> - **是什么**：weight-only int4 PTQ。用激活幅值找显著通道，per-channel scaling 保护它们，其余 int4。
> - **为什么**：decode 阶段是 [[memory bandwidth]] bound（[[Roofline模型]] 算术强度 I≈2），瓶颈是读权重，int4 把权重字节压 4× 直接提 decode 吞吐；但均匀 int4 量化少数显著权重误差大、毁精度，AWQ 用 scaling 保护它们。
> - **与 [[GPTQ]] 关系**：GPTQ 用 Hessian 二阶信息逐列校准补偿误差（重、校准成本高、精度略好）；AWQ 用激活幅值一阶统计找显著通道（快、无反向、校准秒级、精度接近）。两者是 int4 PTQ 的两大主流路线。


## 2. 为什么需要它（动机与背景）

### 2.1 decode 是带宽瓶颈，权重量化直接提吞吐

LLM 自回归 decode 每步生成 1 个 token，每个 token 都要**把整份模型权重从 HBM 读一遍**。按 [[Roofline模型]]，decode 的算术强度 $I = \frac{\text{FLOP}}{\text{bytes}} \approx 2$（每读 1 个参数 2 字节做约 2 次乘加），远低于 GPU 的脊点 $I^* = \frac{\text{峰值算力}}{\text{带宽}}$（A100 约 156、H100 约 211、B200 约 500）。故 decode 是 **memory-bandwidth-bound**，瓶颈是**权重读取带宽**。把权重从 fp16（2 B/param）压到 int4（0.5 B/param + 量化元数据），权重字节压 4×，decode 吞吐近线性涨 4×。这是 [[memory bandwidth]] 与 decode 优化的核心手段。详见 [[memory bandwidth]]、[[Roofline模型]]、[[sampling throughput]]。

### 2.2 均匀 int4 量化毁精度：少数权重是"显著"的

朴素 per-channel int4 量化（$\hat W = \text{round}(W/s)\cdot s$）的问题是：**量化误差在所有权重上均匀分布**（$|\hat W - W| \leq \Delta/2$，$\Delta$ 是量化步长），但**少数权重对输出影响极大**（它们对应的输入通道激活幅值大，$y_j = \sum_i W_{ij} x_i$ 中 $|x_i|$ 大的项贡献大），这些权重稍一量化不准就毁输出。实验表明 LLM 里约 **0.1%~1% 的输入通道是"显著"的**（激活幅值远大于其他通道），它们对应的权重列必须保护。

### 2.3 保护显著权重 → AWQ 的 scaling 思想

直接把显著权重留在 fp16（混合精度，其余 int4）可行但实现复杂（要分支 kernel）。AWQ 的巧思：**不用混合精度，而用 per-channel scaling 等效保护**——给显著通道的权重乘大 scale $s$（$s>1$）使其幅值变大、相对量化误差变小，同时给激活除以 $s$（激活仍 fp16 不引入量化误差），浮点下 $y = (s\odot W)(x \oslash s) = Wx$ 不变，但量化后 $\hat W = Q(s\odot W)$ 的显著列误差被压低。


## 3. 核心概念详解

### 3.1 显著通道（salient channels）

对一个线性层 $y = xW$（$x\in\mathbb{R}^{d_{in}}$，$W\in\mathbb{R}^{d_{in}\times d_{out}}$），输出 $y_j = \sum_i W_{ij} x_i$。某个输入通道 $i$ 若其激活幅值 $|x_i|$ 在校准集上平均很大，则连到它的权重 $W_{i,:}$（第 $i$ 行）对输出贡献大，是"显著"的。**AWQ 用校准数据前向、统计每层每输入通道的平均激活幅值 $s_i^{\text{act}} = \mathbb{E}(|x_i|)$，幅值大的通道即显著通道**。注意是"激活感知"——用激活而非权重本身判断重要性。

> [!note] 为什么不直接挑 top-1% 权重留 fp16（混合精度）
> 可以，但混合精度需要分支 kernel（int4 路径 + fp16 路径），实现复杂、访存不连续。AWQ 用 scaling 把"保护"伪装成"统一 int4 量化"——所有权重都是 int4，只是显著列的量化误差被 scale 压低了，kernel 仍统一。这是 AWQ 工程上的关键。

### 3.2 per-channel scaling（AWQ 的核心机制）

对显著通道 $i$，乘 scale $s_i > 1$：

- 权重：$W'_{i,:} = W_{i,:} \cdot s_i$（显著行放大）
- 激活：$x'_i = x_i / s_i$（对应激活缩小，保持 $y$ 不变）
- 量化：$\hat W' = \text{Quantize}(W')$（int4，步长 $\Delta$）
- 推理时：$\hat y = \hat W' \cdot (x \oslash s)$

**关键**：量化误差 $|\hat W'_{i,:} - W'_{i,:}| \leq \Delta/2$（被步长约束，且因为只放大少数通道，group/tensor 的 $\max|W'|$ 几乎不变，$\Delta$ 几乎不变）。而输出误差 $\Delta y_i = (\hat W'_{i,:} - W'_{i,:}) \cdot (x_i/s_i)$，多了个 $1/s_i < 1$ 因子，**显著通道的输出误差被压 $s_i$ 倍**。$s$ 越大保护越强，但会略微抬高非显著通道的步长 $\Delta$（因为放大了 group max），所以需搜最优 $s$。

### 3.3 scale 的搜索

AWQ 不是手动设 $s$，而是搜索每层最优 per-channel $s$ 使量化误差最小。论文用启发式：scale 正比于激活幅值、反比于权重幅值，并引入混合系数 $\alpha \in [0,1]$ 网格搜索（典型 20 个点）：

$$s_i = \left(\frac{\mathbb{E}(|x_i|)}{\max(|W_{i,:}|)}\right)^\alpha$$

$\alpha=0$ 即不 scale（均匀量化），$\alpha>0$ 越保护显著通道。网格搜 $\alpha$ 取校准集 MSE 最小者。搜索成本极低（一次前向 + 几十次量化对比）。

### 3.4 zero point 与量化格式

AWQ 支持 **symmetric / asymmetric int4**（带 zero point $z$），per-group 量化（典型 group=128 channel 共享一组 scale/zp）。反量化 $\hat W = (\hat W_{int} - z)\cdot \Delta$。多数实现默认 per-group symmetric。

### 3.5 weight-only：激活仍 fp16/bf16

**AWQ 只量化权重，激活保持 fp16/bf16 不量化**（W4A16）。原因：① decode 带宽瓶颈在权重读取，量化权重即省带宽；② 激活量化（W4A4/W8A8）需校准激活量化参数且对 outlier 敏感（见 [[INT8推理]] 的 SmoothQuant），weight-only 简单且精度好。计算时 int4 权重先 dequant 回 fp16 再做 fp16 GEMM（dequant + GEMM 融合 kernel，见 [[fused kernel]]）。


## 4. 数学原理 / 公式

### 4.1 量化基本式

per-group symmetric int4 量化（group $g$ 内共享 scale $\Delta_g$）：

$$\hat W_{int} = \text{round}\left(\frac{W}{\Delta_g}\right),\quad \Delta_g = \frac{\max_{i\in g}|W_i|}{2^{b-1}-1}$$

$b=4$ 时 $\Delta_g = \max|W|/7$。反量化 $\hat W = \hat W_{int}\cdot\Delta_g$。误差 $|\hat W - W|\leq \Delta_g/2$。

### 4.2 AWQ scaling 推导（核心）

设 per-channel scale 向量 $s\in\mathbb{R}^{d_{in}}_{>0}$，线性层：

$$y = xW = (x\oslash s)\cdot (s\odot W) \quad \text{（浮点不变）}$$

其中 $\oslash,\odot$ 是逐通道除/乘。记 $W' = s\odot W$（行缩放），$x' = x\oslash s$。量化 $W'$ 得 $\hat W' = Q(W')$。推理输出：

$$\hat y = \hat W'\cdot x'$$

输出误差（per 通道 $i$ 贡献）：

$$\hat y - y = (\hat W' - W')\cdot x' = \sum_i \underbrace{(\hat W'_{i,:} - W'_{i,:})}_{\text{权重量化误差}}\cdot \underbrace{\frac{x_i}{s_i}}_{\text{激活缩放后}}$$

**关键**：权重量化误差 $|\hat W'_{i,:} - W'_{i,:}|\leq \Delta/2$（被步长约束，且因为只放大少数通道，group/tensor 的 $\max|W'|$ 几乎不变，$\Delta$ 几乎不变）。而激活项 $x_i/s_i$ 对显著通道（$x_i$ 大）配大 $s_i$ → 该项被压小 $s_i$ 倍。故**显著通道的输出误差被压 $s_i$ 倍**，而非显著通道（$x_i$ 小）本来贡献就小，略增无妨。净效果：输出 MSE 降。

> [!warning] 误区：以为 scale 越大越好
> $s$ 大压显著通道误差，但会抬高 group 的 $\max|W'|$ → $\Delta$ 变大 → 非显著通道绝对误差变大（它们 $x$ 小，影响小但有）。存在最优点，故要搜 $\alpha$。不是无脑放大。

### 4.3 scale 搜索目标

最小化校准集输出 MSE：

$$s^* = \arg\min_s \mathbb{E}_{x}\left[\left\|(\hat W'(s) - W'(s))\cdot (x\oslash s)\right\|^2\right]$$

AWQ 用启发式 $s_i = (\mathbb{E}|x_i|/\max|W_{i,:}|)^\alpha$ + $\alpha$ 网格搜（成本低，免反传）。

### 4.4 误差分析

设显著通道占比 $\rho\approx1\%$，scale $s$。AWQ 后显著通道输出误差 $\sim O(\Delta/(2s))$，非显著 $\sim O(\Delta\cdot s^\gamma/2)$（$\gamma<1$，因 group max 缓增）。总 MSE $\propto \rho\frac{\Delta^2}{4s^2} + (1-\rho)\frac{\Delta^2 s^{2\gamma}}{4}$，对 $s$ 求最优有非零解。典型 $s\in[1, 3]$ 区间最优。


## 5. 代码示例（可选）

### 5.1 手写 per-channel int4 量化 + AWQ scaling 对比误差

```python
import torch

def quantize_per_group_sym(w, g=128):
    """per-group symmetric int4 量化 (group=g channels 共享 scale), 返回反量化后的 w_hat 与误差."""
    orig = w.shape
    w = w.reshape(-1, g)                      # [out*g//g_in?, g] 简化: 按 in 维 group
    mx = w.abs().amax(dim=-1, keepdim=True)
    delta = mx / 7.0                          # int4 symmetric: 7 = 2^(4-1)-1
    q = torch.round(w / delta).clamp(-8, 7)   # int4 范围 [-8,7]
    w_hat = q * delta
    return w_hat.reshape(orig)

def awq_quantize(w, x_calib, alpha=0.5, g=128):
    """AWQ: 先算 per-channel scale s (基于激活幅值), 放大显著权重再量化."""
    # x_calib: [n, d_in], 统计每输入通道平均激活幅值
    act_mag = x_calib.abs().mean(dim=0)       # [d_in]
    w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [d_in,1]
    s = (act_mag / w_max.squeeze(-1)).clamp(min=1e-8) ** alpha  # [d_in]
    s = s.clamp(min=1.0)                      # scale>=1 (保护)
    w_scaled = w * s.unsqueeze(-1)            # 行(输入通道)放大
    w_hat = quantize_per_group_sym(w_scaled, g)   # 量化放大后的
    return w_hat / s.unsqueeze(-1)            # 反 scale 回原量纲

torch.manual_seed(0)
d_in, d_out, n = 4096, 4096, 128
w = torch.randn(d_in, d_out) * 0.02
x = torch.randn(n, d_in); x[:, ::200] *= 12   # 制造 1% 显著通道 (幅值大 12x)

# 朴素 per-group int4
hat_naive = quantize_per_group_sym(w, g=128)
# AWQ
hat_awq = awq_quantize(w, x, alpha=0.5, g=128)

y_true = x @ w
err_naive = ((x @ hat_naive - y_true)**2).mean()
err_awq   = ((x @ hat_awq   - y_true)**2).mean()
print(f"朴素 int4 输出 MSE: {err_naive:.4e}")
print(f"AWQ    int4 输出 MSE: {err_awq:.4e}")   # 通常显著降 (显著通道被保护)
```

### 5.2 用 AutoAWQ 跑 AWQ 量化（生产）

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "Llama-2-7b-awq"
tokenizer = AutoTokenizer.from.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto")
# 校准数据 (如 wikitext/PEAA)
model.quantize(tokenizer, quant_config={
    "zero_point": True, "q_group_size": 128,
    "w_bit": 4, "version": "GEMM",   # GEMM 版 (Marlin kernel 最快)
})
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### 5.3 vLLM 加载 AWQ 模型推理

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Llama-2-7b-awq \
    --quantization awq \
    --dtype float16          # 权重 int4, 激活/计算 fp16 (W4A16)
# vLLM 用 Marlin/fused dequant+GEMM kernel, decode 吞吐约 3-4x fp16
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[数值类型与精度]]（int4 的位级结构、量化基本式）、[[mixed precision training]]（量化是 AMP 之外的进一步压低）、[[memory bandwidth]]（decode 带宽瓶颈是动机）、[[Roofline模型]]（算术强度 I≈2 → memory-bound → 量化权重省带宽）、后训练量化 PTQ 概念。
- **下游（应用）**: [[新模型接入]]（vLLM 接入 AWQ 量化模型，`--quantization awq`）、[[model runner]]（runner 调 fused dequant+GEMM kernel）、[[sampling throughput]]（int4 decode 吞吐 3-4×）、[[Tensor Parallel]]（AWQ 权重按 TP 切分，每卡持 int4 片）、QLoRA 微调（4-bit 基座 + LoRA，虽 QLoRA 用 NF4 非 AWQ，思路相通）。
- **对比 / 易混**:
  - **AWQ vs [[GPTQ]]**：见下表。AWQ 一阶激活统计 + scaling（快、无反传），GPTQ 二阶 Hessian + 逐列补偿（重、精度略好）。
  - **AWQ vs [[INT8推理]] W8A8**：AWQ 是 weight-only int4（W4A16，省带宽）；W8A8 是权重+激活都 int8（省算力）。前者省带宽提 decode，后者省算力提 prefill。
  - **AWQ vs [[FP4与低比特]] NF4**：NF4 是浮点 4-bit（为正态分布权重优化，用于 QLoRA 微调）；AWQ 是定点 int4 + scaling（用于推理 PTQ）。推理 AWQ，微调 NF4。
  - **AWQ vs [[FP8量化方案]]**：FP8 是浮点 8-bit（Hopper+ 原生、训练友好）；AWQ 是定点 4-bit（省更多但仅推理 weight-only）。

### AWQ vs GPTQ 对比表

| 维度 | AWQ | GPTQ |
|---|---|---|
| 核心思想 | 激活感知找显著通道，per-channel scaling 保护 | Hessian 二阶信息逐列量化 + 误差补偿剩余列 |
| 信息阶 | 一阶（激活幅值统计） | 二阶（Hessian $H=2XX^\top$） |
| 校准数据 | 128~512 样本前向，秒级 | 同量样本前向算 Hessian，分钟级 |
| 是否反传 | 否（纯前向统计） | 否（PTQ，但算 Hessian 较重） |
| 量化粒度 | per-group (128) + per-channel scaling | per-group (128)，逐列补偿 |
| 精度（int4） | 略逊 GPTQ（0.1~0.5% ppl 差） | 略优（二阶补偿更精） |
| 速度（校准） | 快（秒级） | 慢（分钟级，大模型更慢） |
| 混合精度 | 可（显著列 fp16，但默认全 int4 + scale） | 可（group-wise） |
| 框架集成 | vLLM(`--quantization awq`)、TRT-LLM、AutoAWQ | vLLM(`--quantization gptq`)、AutoGPTQ、TRT-LLM |
| 推理 kernel | Marlin/AWQ-GEMM (dequant+GEMM fused) | Marlin/GPTQ-GEMM |


## 7. 常见误区与易错点

> [!warning] 误区 1：AWQ 是训练时量化（QAT）
> 不是。AWQ 是**后训练量化（PTQ）**——模型训好后，用一小批校准数据前向统计激活幅值找显著通道，不训练、不反传、不更新权重。"activation-aware"指用激活信息指导，不是用激活做训练。

> [!warning] 误区 2：显著权重 = 权重幅值大的
> 不对。**显著 = 对应激活幅值大的通道的权重**（activation-aware，不是 weight-aware）。一个权重本身幅值小，但若其输入通道激活大，它就显著。反之权重大但激活小的未必关键。这是 AWQ 与朴素 weight-magnitude 量化的区别。

> [!warning] 误区 3：scaling 改了输出值
> 浮点下 $(s\odot W)(x\oslash s) = Wx$，输出不变。scaling 只改变**量化后的相对误差分布**，不改真值。误区是以为 scale 把权重"变大"会改模型行为——不会，因为激活同比例变小补偿。

> [!warning] 误区 4：AWQ 量化激活
> AWQ 是 **weight-only**（W4A16），激活全程 fp16/bf16 不量化。若要量化激活（省算力）走 [[INT8推理]] W8A8 或 [[FP8量化方案]]。AWQ 省的是 decode **带宽**（权重读取），不是算力。

> [!warning] 误区 5：int4 权重直接做 int4 GEMM
> 不对。当前 GPU（A100/H100）无 int4 Tensor Core GEMM。AWQ int4 权重先 **dequant 回 fp16** 再做 fp16 GEMM（dequant 与 GEMM 融合成一个 kernel，见 [[fused kernel]]）。省的是**带宽**（int4 读取 4× 少），算力仍是 fp16。Blackwell 才有 FP4 Tensor Core（见 [[FP4与低比特]]）。

> [!tip] 实践：校准数据要代表部署分布
> AWQ 找显著通道靠校准激活幅值。若校准数据与实际部署输入分布差大（如校准用英文 wiki、部署用中文代码），显著通道判错 → 精度退化。用与部署同分布的校准集（至少 128 样本）。

> [!tip] 实践：group_size 调
> 默认 group=128（128 channel 共享一组 scale/zp）。小 group 精度好但元数据多（省的带宽被 scale 抵消），大 group 省带宽但精度差。128 是经验最优。极端 group=1 即 per-channel。


## 8. 延伸细节

### 8.1 Marlin kernel（AWQ/GPTQ 通用 fast kernel）

AWQ/GPTQ int4 推理的 fastest kernel 是 **Marlin**（vLLM 集成），把 int4 权重解包 + dequant + fp16 GEMM 融合成单 kernel，最大化带宽利用。是 [[fused kernel]] 在量化的应用。老 AWQ-GEMM kernel 慢于 Marlin，新 vLLM 默认 Marlin。

### 8.2 与 GPTQ 的精度差

GPTQ 二阶补偿在 3-bit（int3）精度优势更明显（int4 两者接近，int3 GPTQ 略胜）。AWQ 在 int4 + 大模型（7B+）与 GPTQ 差距 <0.5% ppl，可忽略。选 AWQ 多因校准快、kernel 生态成熟。

### 8.3 内容来源

AWQ 整理自 Lin et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"（MIT Han Lab, NeurIPS 2023）、AutoAWQ GitHub、vLLM `--quantization awq` 文档与 `vllm/model_executor/layers/quantization/awq/` 源码、Marlin kernel 论文。与 GPTQ 对照见 [[GPTQ]]。NF4 对照见 [[FP4与低比特]]。

---
相关: [[量化]] | [[GPTQ]] | [[INT8推理]] | [[FP4与低比特]] | [[FP8量化方案]] | [[数值类型与精度]] | [[memory bandwidth]] | [[Roofline模型]] | [[新模型接入]] | [[model runner]] | [[Tensor Parallel]] | [[fused kernel]] | [[mixed precision training]] | [[sampling throughput]]
