# FP4与低比特

> **所属章节**: [[量化]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: FP4 / 4-bit 浮点量化 / NF4 / NormalFloat4 / MXFP4 / Microscaling FP4 / NVFP4 / mBlock 量化 / microscaling block quantization / per-block shared scale
> **难度**: 高（需懂 [[数值类型与精度]] 浮点位级 + [[FP8量化方案]] 8→4 递进 + 量化信息论 + per-block 共享 scale + Blackwell 硬件 FP4 Tensor Core）
> **关联**: 本笔记讲 **4-bit 量化的浮点格式族**（NF4 / MXFP4 / NVFP4 / per-block mBlock）与定点 INT4 的区别；是 [[FP8量化方案]] 从 8-bit 到 4-bit 的递进，与 [[AWQ]] / [[GPTQ]]（定点 int4）对照，同属 [[量化]] 体系。


## 1. 一句话定义

**FP4 与低比特量化**指把权重/激活压到 **4-bit（每元素 0.5 字节，比 int8 再省 2×、比 fp16 省 4×）** 的量化格式族——因 4-bit 动态范围极小（仅 16 个表示值），定点 INT4（均匀 16 台阶）对 LLM 权重/激活的大动态范围表示力不足，故发展出**浮点 4-bit**格式：① **NF4（NormalFloat4）**——QLoRA 提出，基于标准正态分布的 16 个分位点，**信息论意义上对正态分布权重最优**，配 blockwise absmax 归一化，用于 QLoRA 4-bit 微调；② **MXFP4（OCP Microscaling FP4）**——4-bit 浮点元素（E2M1）+ **per-block 共享 8-bit 指数 scale（block=32）**，用共享指数把动态范围扩大数十量级，是 OCP 2023 微缩放格式标准、OpenAI gpt-oss 等采用；③ **NVFP4（NVIDIA Blackwell FP4）**——E2M1 + per-block scale（block=16，更细粒度），Blackwell B200/B300 原生 **FP4 Tensor Core**（吞吐约 2× FP8）。核心：**4-bit 浮点带指数 + per-block 共享 scale = 比定点 INT4 更大动态范围、比 per-tensor FP4 更精细**，是 Blackwell 时代 4-bit 训推的新方向。

> [!note] 三句话定位
> - **是什么**：4-bit 量化格式族——NF4（正态分位点，QLoRA）、MXFP4（E2M1 + block32 共享 E8M0 scale，OCP 标准）、NVFP4（E2M1 + block16 scale，Blackwell 原生）。浮点 + per-block 共享 scale。
> - **为什么**：4-bit 动态范围太小，定点 INT4 不够；浮点带指数 + per-block 共享 scale 把动态范围扩到可用，且 Blackwell FP4 Tensor Core 2× FP8 算力红利。
> - **与 [[FP8量化方案]] 关系**：FP4 是 FP8 从 8-bit 到 4-bit 的递进——同样"浮点 + scale"，只是元素更窄、更依赖 per-block 共享 scale 补动态范围。FP8 是训推通用甜点，FP4 是极致省显存/算力的前沿。


## 2. 为什么需要它（动机与背景）

### 2.1 4-bit 是省显存/带宽/算力的极致

- **显存/带宽**：4-bit = 0.5 B/param，比 fp16（2B）省 4×、比 int8（1B）省 2×。70B 模型 fp16 140GB → FP4 ~35GB，单 80GB 卡可装。decode memory-bound，权重字节省 4× → 吞吐近 4×。
- **算力（Blackwell）**：Blackwell FP4 Tensor Core 吞吐约 2× FP8（B200 FP4 ~9 PFLOPS vs FP8 ~4.5 PFLOPS，待核实具体 SKU），4-bit 既省带宽又省算力（前提硬件支持 GEMM）。

### 2.2 4-bit 动态范围小，定点 INT4 不够

4-bit 只有 16 个表示值。定点 INT4（均匀 16 台阶，symmetric $[-7,7]$）动态范围 $\approx 14\times$（max/min nonzero），对 LLM 权重/激活的大动态范围（数十量级）表示力不足：要么大值饱和、要么小值截 0。**浮点 4-bit（带指数）** 让小值小台阶、大值大台阶，动态范围大得多（E2M1 max 6，配合 per-block 共享指数 scale 可达数十量级）。

### 2.3 per-block 共享 scale（mBlock）补动态范围

单 FP4 元素动态范围仍有限（E2M1 仅 ±{0,0.5,1,1.5,2,3,4,6}）。**Microscaling（MX）思路**：一组（block）元素**共享一个 8-bit 指数 scale**（E8M0，即 2 的幂），block 内元素相对 scale 归一化。共享指数把整 block 的动态范围扩到 $2^{-127}\sim2^{128}$（E8M0 覆盖），远超单 FP4。这就是 **mBlock（microscaling block）量化**——per-block 共享 scale 是 MX 格式族（MXFP4/MXFP8）的核心机制。

### 2.4 训练 vs 推理用 FP4

- **推理**：weight-only FP4（省带宽，decode）或 W4A4（省算力，Blackwell GEMM）。NF4 用于 QLoRA 微调基座、MXFP4/NVFP4 用于推理部署。
- **训练**：FP4 训练尚前沿（动态范围对梯度敏感，易不稳），Blackwell 出现后探索中，主流训练仍 fp8/bf16。FP4 训练需特殊稳定性手段（loss scaling、per-block scale 动态更新，见 [[loss scaling]]）。


## 3. 核心概念详解

### 3.1 NF4（NormalFloat4，QLoRA）

QLoRA（Dettmers et al. 2023）提出的 4-bit 数据类型，**信息论意义上对正态分布权重的最优表示**：

- **构造**：取标准正态 $N(0,1)$ 的 16 个等概率分位点（把 $[-1,1]$ 归一化的正态 CDF 等分成 16 段，取每段代表值），得 16 个非均匀间距的码值。因 LLM 权重近似正态，这些分位点恰好让每个码值等概率命中 → 信息熵最大（最优）。
- **NF4 码本（16 值，对称）**：$\pm\{0, 0.0696, 0.1415, 0.2121, 0.2844, 0.3726, 0.4839, 0.6962, 1.0\}$（即 $[-1.0, -0.6962, -0.4839, -0.3726, -0.2844, -0.2121, -0.1415, -0.0696, 0.0696, ..., 1.0]$，来自 bitsandbytes NF4 codebook，精确小数待核实）。
- **blockwise absmax 归一化**：每 block（如 64 元素）算 absmax，权重除以 absmax 归一化到 $[-1,1]$ 再查 NF4 码本，存 4-bit 索引 + block scale。**双量化**（double quantization）：把 block scale 本身再量化到 8-bit 省元数据存储。
- **用途**：QLoRA 微调——基座权重 NF4 冻结（4-bit），LoRA 增量 fp16 反传，让单 48GB 卡微调 65B。是 NF4 的标志性应用。
- **硬件**：A100 等**无 FP4 Tensor Core**，NF4 权重推理时 dequant 回 fp16 做 GEMM（开销大，需 fused kernel）。Blackwell 才原生 FP4。

### 3.2 MXFP4（OCP Microscaling FP4）

OCP（Open Compute Project）2023 微缩放格式（MX）标准的 4-bit 成员：

- **元素格式**：E2M1（4-bit：1 符号 + 2 指数 + 1 尾数），正值集合 $\{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$（8 正 8 负，含 0，max 6）。
- **per-block 共享 scale**：每 **block=32 个 E2M1 元素共享一个 8-bit 指数 scale（E8M0，即 2 的幂，bias 127）**。实际值 $=$ E2M1 元素 $\times 2^{(\text{E8M0}-127)}$。共享指数把 block 动态范围扩到 $\sim 2^{\pm127}$。
- **与 FP8 的递进**：MXFP8（8-bit E4M3 元素 + block32 共享 E8M0）是 MX 格式的 8-bit 版；MXFP4 把元素压到 4-bit、保留 per-block 共享 scale。思路同 [[FP8量化方案]] 的 per-block/MX block scaling，只是元素更窄。
- **采用**：OpenAI gpt-oss（开源模型用 MXFP4 降推理成本）、部分国产 GPU（昇腾/寒武纪适配 MXFP4）、训练探索。是 OCP 推的训推统一 4-bit 标准希望。

### 3.3 NVFP4（NVIDIA Blackwell FP4）

NVIDIA Blackwell（B200/B300）的原生 FP4 格式：

- **元素**：E2M1（同 MXFP4）。
- **scale 粒度**：per-block，但 **block=16**（比 MXFP4 的 32 更细），scale 为 FP8（E4M3）格式（待核实 exact scale dtype，部分资料称 E4M3 per-block scale）。更细粒度 → 精度更好但元数据更多。
- **硬件**：Blackwell **FP4 Tensor Core 原生支持**，吞吐约 2× FP8（B200 FP4 ~9 PFLOPS vs FP8 ~4.5 PFLOPS，待核实 SKU）。是 Blackwell 推理/训练算力翻倍的关键。
- **与 MXFP4 区别**：都是 E2M1 + per-block scale，但 NVFP4 block=16 + FP8 scale（NVIDIA 私有/Blackwell 原生），MXFP4 block=32 + E8M0 scale（OCP 开放标准）。vLLM 已支持 NVFP4 W4A 量化。

### 3.4 mBlock（microscaling block）量化通则

mBlock 量化的通用模式：把张量分 block，**block 内元素共享一个 scale（指数或浮点）**，元素本身低 bit。好处：① 共享 scale 扩动态范围（单 FP4 不够）；② 元数据少（一 block 一 scale vs per-element）；③ 适配硬件（MX/NVFP4 Tensor Core 按 block 处理）。是 [[FP8量化方案]] per-block/MX scaling 在 4-bit 的延续。block size 是精度 vs 元数据成本 trade-off（小 block 精度好元数据多，大反之；MXFP4=32、NVFP4=16）。

### 3.5 group quantization（per-128 channel group）

定点 int4（[[AWQ]]/[[GPTQ]]）用 **per-128-channel group**：128 个 channel 共享一组 scale/zp（定点，非浮点指数）。与 mBlock（per-block 共享浮点指数 scale）区别：group 量化是定点 scale（连续值），mBlock 是浮点指数 scale（2 的幂，硬件友好）。两者都是"per-block 共享 scale 减元数据"思路，定点用 group、浮点用 microscaling block。

### 3.6 与 INT4 的区别

| | INT4（定点） | FP4（浮点，NF4/MXFP4/NVFP4） |
|---|---|---|
| 表示 | 均匀 16 台阶（symmetric ±7） | 非均匀，带指数（E2M1：±{0,.5,1,1.5,2,3,4,6}）或正态分位点（NF4） |
| 动态范围 | 小（~14×） | 大（指数自适应 + per-block 共享 scale 达数十量级） |
| 小值精度 | 差（均匀台阶，小值台阶大） | 好（浮点小值小台阶） |
| 硬件 GEMM | 无 int4 Tensor Core（dequant 回 fp16） | Blackwell FP4 Tensor Core 原生（MXFP4/NVFP4） |
| 信息论最优 | 否（均匀非最优） | NF4 对正态分布最优 |
| 典型用 | AWQ/GPTQ 推理 weight-only | QLoRA 微调(NF4) / Blackwell 推理(MXFP4/NVFP4) |


## 4. 数学原理 / 公式

### 4.1 E2M1 浮点 4-bit 表示

4-bit：1 符号 $s$ + 2 指数 $e$ + 1 尾数 $m$。指数 bias=1（$2^{2-1}-1=1$）。

- $e=0$（subnormal）：值 $=(-1)^s\cdot 0.m\times 2^{1-1}=(-1)^s\cdot\{0, 0.5\}$
- $e\in\{1,2\}$（normal）：值 $=(-1)^s\cdot 1.m\times 2^{e-1}=\pm\{1.0,1.5,2.0,3.0\}$
- $e=3$（max，非 inf）：值 $=(-1)^s\cdot 1.m\times 2^{3-1}=\pm\{4.0, 6.0\}$

正值集合 $\{0,0.5,1.0,1.5,2.0,3.0,4.0,6.0\}$，max $6.0$。带符号共 16 值（含 ±0）。

### 4.2 per-block 共享 scale（mBlock）

设 block $B$ 有 32 个 E2M1 元素 $v_i$，共享 scale $S=2^{(E-127)}$（$E$ 是 E8M0 的 8-bit 指数域）。实际值：

$$x_i = v_i \times S = v_i \times 2^{(E-127)},\quad i\in B$$

scale 选取：$S=\text{absmax}(B)/6$（让 block 最大值映射到 E2M1 的 max 6），$E=\text{round}(\log_2 S)+127$（E8M0 量化到 2 的幂，有 $\leq$ 半步误差）。per-block scale 把 block 动态范围扩到 $2^{\pm127}$，远超单 E2M1 的 6/0=∞ 但实际有效 ~12×。

### 4.3 NF4 分位点构造（信息论最优）

设权重 $\sim N(0,\sigma^2)$，归一化到 $[-1,1]$（除 absmax）。取标准正态 CDF $\Phi$，把 $[0,1]$ 等分 16 段，分位点 $p_k=\Phi^{-1}((k+0.5)/16)$，$k=0..15$，归一化到 $[-1,1]$ 得 NF4 码本 $\{c_k\}$。因每个码值等概率命中（$P(x\in[c_k,c_{k+1}))=1/16$），**离散熵 $H=-\sum\frac1{16}\log\frac1{16}=4$ bit 达满**——信息论意义下对正态分布 4-bit 最优（最大化熵）。量化 $x\to\arg\min_k|x-c_k|$（最近码值）。

### 4.4 量化误差与动态范围

- INT4 per-group：误差 $\leq\Delta/2=\max|W|/(2\cdot7)$，动态范围 $\max/\min_{\text{nonzero}}\approx 7/1=7$（symmetric），小值相对误差大。
- FP4 E2M1 + per-block scale：小值台阶 0.5（相对大值 6 的 1/12），动态范围（含 scale）$\sim 6\times 2^{127}/（0.5\times 2^{-127})$ 巨大。小值精度好。
- NF4：非均匀分位点，正态分布尾部（大值）稀疏码值、中部密集，匹配权重分布，期望量化误差低于均匀 INT4。

### 4.5 Blackwell FP4 算力红利

FP4 元素 0.5 字节（vs FP8 1 字节），同带宽下喂 2× 元素；Tensor Core 一拍处理 2× FP4 vs FP8 → 吞吐 ~2×。$P_{FP4}\approx 2 P_{FP8}$（B200 约 9 vs 4.5 PFLOPS，待核实）。是 4-bit 既省带宽又省算力的根（前提原生 FP4 GEMM，Blackwell 满足，A100/H100 不满足）。


## 5. 代码示例（可选）

### 5.1 手写 E2M1 FP4 + per-block 共享 scale 量化

```python
import torch
import math

E2M1_POS = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # 正值码本

def quant_fp4_mxfp4(w, block=32):
    """MXFP4 风格: E2M1 元素 + per-block 共享 E8M0 scale (2 的幂)."""
    orig = w.shape
    w = w.reshape(-1, block)
    # per-block scale: absmax/6, 量化到 2 的幂 (E8M0)
    absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale_f = absmax / 6.0
    # E8M0: scale 量化到最近的 2 的幂
    E = torch.round(torch.log2(scale_f))
    scale = 2.0 ** E
    w_norm = w / scale                            # 归一化到 [-6,6]
    # 量化到 E2M1 (最近码值)
    q = torch.argmin((w_norm.unsqueeze(-1).abs() - E2M1_POS).abs(), dim=-1)
    sign = torch.sign(w_norm)
    w_hat = sign * E2M1_POS[q] * scale            # 反量化
    # 存: q (4-bit) + E (8-bit) per block
    return w_hat.reshape(orig), q, E.int()

def quant_nf4(w, block=64):
    """NF4 风格: 正态分位点码本 + blockwise absmax 归一化."""
    nf4 = torch.tensor([-1.,-.6962,-.4839,-.3726,-.2844,-.2121,-.1415,-.0696,
                         .0696,.1415,.2121,.2844,.3726,.4839,.6962,1.])
    orig = w.shape; w = w.reshape(-1, block)
    absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    w_norm = w / absmax                           # 归一化到 [-1,1]
    q = torch.argmin((w_norm.unsqueeze(-1) - nf4).abs(), dim=-1)
    w_hat = nf4[q] * absmax
    return w_hat.reshape(orig)

torch.manual_seed(0)
W = torch.randn(4096, 4096) * 0.03                 # 近似正态权重
hat_mxfp4, _, _ = quant_fp4_mxfp4(W, block=32)
hat_nf4 = quant_nf4(W, block=64)
def naive_int4(w, g=128):
    d = w.abs().amax()/7
    return torch.round(w/d).clamp(-8,7)*d
hat_int4 = naive_int4(W)
err = lambda h: ((h-W)**2).mean().item()
print(f"INT4    MSE: {err(hat_int4):.4e}")
print(f"MXFP4   MSE: {err(hat_mxfp4):.4e}")
print(f"NF4     MSE: {err(hat_nf4):.4e}")          # NF4 通常最优 (正态分布)
```

### 5.2 QLoRA 用 NF4 微调（bitsandbytes）

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 (正态分位点, 信息论最优)
    bnb_4bit_use_double_quant=True,    # 双量化 (量化 scale 本身省元数据)
    bnb_4bit_compute_dtype=torch.bfloat16,  # 反传/计算 bf16
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf",
                                             quantization_config=bnb)
# 基座 NF4 冻结, LoRA 增量 bf16 反传
peft = LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj","v_proj"])
model = get_peft_model(model, peft)    # 单卡微调 70B
```

### 5.3 vLLM 用 NVFP4 推理（Blackwell）

```bash
# vLLM (Blackwell B200+) 加载 NVFP4 W4A 量化模型
python -m vllm.entrypoints.openai.api_server \
    --model <nvfp4-quantized-model> \
    --quantization fp4 \
    --dtype bfloat16      # 激活/累加 bf16, 权重 FP4 + per-block scale
# 走 Blackwell FP4 Tensor Core, decode 吞吐 ~2x FP8 / ~4x fp16
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[数值类型与精度]]（浮点位级 E2M1/E4M3/E8M0、正态分布分位点、信息熵）、[[FP8量化方案]]（FP4 是 FP8 的 8→4 递进，per-block/MX scaling 同源）、[[mixed precision training]]（4-bit 是 AMP 之外的极致压低）、[[loss scaling]]（FP4 训练防梯度下溢）、[[memory bandwidth]] / [[Roofline模型]]（4-bit 省 4× 带宽的 decode 动机）。
- **下游（应用）**: [[新模型接入]]（vLLM NVFP4/MXFP4 接入）、[[model runner]]（FP4 Tensor Core GEMM kernel）、QLoRA 微调（NF4 基座 + LoRA）、[[sampling throughput]]（FP4 decode 4× 吞吐）、[[Tensor Parallel]]（FP4 权重 TP 切）、Blackwell 硬件。
- **对比 / 易混**:
  - **FP4 vs [[FP8量化方案]]**：8→4 递进。FP8 训推通用甜点（Hopper+），FP4 极致省（Blackwell+，训练尚前沿）。per-block 共享 scale 思路同源。
  - **FP4 vs INT4（[[AWQ]]/[[GPTQ]]）**：浮点 vs 定点。FP4 带指数动态范围大、小值精度好、Blackwell 原生 GEMM；INT4 均匀、需 AWQ/GPTQ 保护、无原生 GEMM。见 §3.6 表。
  - **NF4 vs MXFP4 vs NVFP4**：NF4 正态分位点（QLoRA 微调、无原生 GEMM、A100 可用）；MXFP4 E2M1+block32 E8M0（OCP 标准、gpt-oss）；NVFP4 E2M1+block16 FP8 scale（Blackwell 原生、vLLM）。见 §3 表。
  - **per-block mBlock vs per-128 group（[[AWQ]]/[[GPTQ]]）**：前者浮点指数 scale（2 的幂，硬件按 block 处理），后者定点连续 scale。都是 per-block 共享 scale 减元数据。

### 4-bit 格式对比表

| 格式 | 元素 | 码本 | per-block scale | block size | 硬件 GEMM | 典型用 |
|---|---|---|---|---|---|---|
| INT4 | 定点 | 均匀 ±7 | 定点 scale/zp | 128 (group) | 无（dequant fp16） | AWQ/GPTQ 推理 |
| NF4 | 浮点(分位点) | 正态 16 分位 | absmax (连续) | 64 | 无（dequant fp16） | QLoRA 微调 |
| MXFP4 | E2M1 | ±{0,.5,1,1.5,2,3,4,6} | E8M0 (2 的幂) | 32 | 部分(国产/gpt-oss) | 推理/训练探索 |
| NVFP4 | E2M1 | 同上 | FP8(E4M3) | 16 | Blackwell 原生 | Blackwell 推理 |


## 7. 常见误区与易错点

> [!warning] 误区 1：FP4 和 INT4 等价
> 不等价。INT4 定点（均匀台阶，动态范围小、小值精度差）；FP4 浮点（带指数，非均匀，小值小台阶大值大台阶，动态范围大）。对 LLM 权重/激活（动态范围大），FP4 表示力优于 INT4。Blackwell 才有 FP4 GEMM，此前 FP4 也要 dequant 回 fp16（无算力红利，只省带宽）。

> [!warning] 误区 2：4-bit 一定省算力
> 不一定。**只有硬件有 4-bit Tensor Core GEMM（Blackwell FP4）** 才省算力 2×。A100/H100 无 FP4/int4 GEMM，4-bit 权重要 dequant 回 fp16 算 fp16 GEMM，**只省带宽不省算力**（甚至 dequant 有开销）。把 4-bit 当省算力会误判 A100 上的收益。

> [!warning] 误区 3：NF4 和 MXFP4 一样
> 不同。NF4 是正态分位点码本（信息论最优正态分布，QLoRA 专用，无原生 GEMM）；MXFP4 是 E2M1 + per-block E8M0 共享指数（OCP 标准，码本固定 ±{0,.5,...,6}）。码本构造、scale 机制、用途都不同。

> [!warning] 误区 4：MXFP4 和 NVFP4 一样
> 都是 E2M1 + per-block scale，但 block size（32 vs 16）和 scale 格式（E8M0 2 的幂 vs FP8 E4M3 连续）不同，精度与硬件原生支持不同（NVFP4 Blackwell 原生，MXFP4 OCP 开放）。别混用。

> [!warning] 误区 5：NF4 随便用于推理
> NF4 是为**正态分布权重 + blockwise absmax**设计，QLoRA 微调场景（基座冻结、反传 LoRA）最优。推理部署若硬件无 FP4 GEMM，NF4 要 dequant 回 fp16（开销），不如 MXFP4/NVFP4（原生 GEMM）或直接 AWQ/GPTQ int4（kernel 成熟）。NF4 推理非首选（除 QLoRA）。

> [!warning] 误区 6：FP4 训练已成熟
> 未成熟。FP4 训练（前向/反向用 FP4）对梯度动态范围极敏感（梯度分布非正态、有 outlier），易下溢/不稳。Blackwell 出现后才探索，主流训练仍 FP8/bf16。FP4 训练需特殊 loss scaling + per-block scale 动态更新，是前沿研究，非生产标配。

> [!tip] 实践：选 4-bit 格式
> 微调 → NF4（QLoRA，bitsandbytes）；Blackwell 推理 → NVFP4（原生 GEMM，vLLM）；非 Blackwell 推理 → AWQ/GPTQ int4（kernel 成熟，Marlin）或 MXFP4（若硬件支持）；极致省显存但无 GEMM → NF4/MXFP4 weight-only（dequant 开销换带宽）。

> [!tip] 实践：block size 调
> mBlock block 小（16）精度好但元数据多（scale 占比升，省的带宽被抵消）；block 大（32/64）省元数据但精度降。MXFP4=32、NVFP4=16 是各自工程权衡。精度敏感用小 block，带宽敏感用大 block。


## 8. 延伸细节

### 8.1 双量化（double quantization，NF4）

QLoRA 的 NF4 配 blockwise absmax scale（每 block 一个 fp32 scale）。**双量化**：把这些 fp32 scale 再量化到 int8（+ 一个上层 scale），把元数据从 32 bit/block 降到 ~8 bit/block，额外省 ~0.5 bit/param 存储。是 QLoRA 省 65B 微调显存的细节之一。

### 8.2 FP4 dequant kernel 开销（无原生 GEMM 时）

A100/H100 无 FP4 Tensor Core，NF4/MXFP4 权重推理要 dequant 回 fp16 做 fp16 GEMM。dequant 是额外算子（查码本 + 乘 scale），需 fused kernel（dequant+GEMM 融合，见 [[fused kernel]]）摊销开销。Fast NF4 dequant kernel（arxiv 2026）优化此开销。Blackwell 原生 FP4 GEMM 免此开销。

### 8.3 Blackwell FP4 算力（参考，待核实）

B200 FP4 ~9 PFLOPS vs FP8 ~4.5 PFLOPS（2×），vs bf16 ~2.25 PFLOPS（4×）。B300（Blackwell Ultra）更高。NVIDIA 称 Blackwell "FP4 性能 4× Hopper"（含架构 + 精度双重红利）。具体 SKU 数字以 NVIDIA datasheet 为准，本笔记标"待核实"。

### 8.4 与 FP8 的递进关系（再强调）

[[FP8量化方案]] 讲 8-bit 浮点（E4M3/E5M2 + per-tensor/per-block/delayed scaling），FP4 是其 8→4 的递进：同样"浮点元素 + scale"，但元素更窄（4 vs 8 bit）→ 更依赖 per-block 共享 scale 补动态范围（FP8 per-block 可选，FP4 per-block 几乎必须）。FP8 是训推通用甜点（Hopper+），FP4 是极致省 + Blackwell 前沿。两者 per-block/MX block scaling 机制同源，是 [[FP8量化方案]] "MXFP8 per-block" 行在 4-bit 的对应。

### 8.5 内容来源

NF4 整理自 QLoRA 论文（Dettmers et al. 2023, arXiv:2305.14314）、bitsandbytes NF4 codebook、[[FP8量化方案]] 双量化对照。MXFP4 整理自 OCP Microscaling Formats (MX) v1.0 spec（2023）、OpenAI gpt-oss MXFP4 用法、知乎/CSDN MXFP4 vs NVFP4 解析。NVFP4 整理自 NVIDIA Blackwell 架构白皮书、vLLM `--quantization fp4`（NVFP4 W4A）文档与源码。E2M1 码本由 IEEE 754 风格浮点定义推算。算力数字标"待核实"者以 NVIDIA datasheet 为准。与 FP8 递进见 [[FP8量化方案]]。

---
相关: [[量化]] | [[AWQ]] | [[GPTQ]] | [[INT8推理]] | [[FP8量化方案]] | [[数值类型与精度]] | [[mixed precision training]] | [[loss scaling]] | [[memory bandwidth]] | [[Roofline模型]] | [[新模型接入]] | [[model runner]] | [[Tensor Parallel]] | [[fused kernel]] | [[sampling throughput]]
