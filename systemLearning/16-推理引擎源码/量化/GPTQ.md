# GPTQ

> **所属章节**: [[量化]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: GPTQ / Hessian-based post-training quantization / 二阶后训练量化 / OBQ-based 量化
> **难度**: 中高（需懂 [[数值类型与精度]] + 二阶优化/Hessian + 矩阵求逆/Cholesky + 量化基本式 + [[memory bandwidth]]）
> **关联**: 本笔记讲 GPTQ 这一**基于 Hessian 的逐列误差补偿量化方法**的原理（OBQ → Cholesky 稳定 → 逐列量化 + 剩余列补偿）与工程集成；与 [[AWQ]]（一阶激活感知路线）互为对照，同属 [[量化]] 体系。


## 1. 一句话定义

**GPTQ（Generalized Post-Training Quantization，基于 Hessian 的后训练量化）** 是一种 **one-shot 后训练量化（PTQ）方法**——它把最优脑外科（OBS）/最优脑量化（OBQ）的二阶误差补偿思路用于 LLM 权重量化：用一小批校准数据前向算出每层的 **Hessian 矩阵 $H = 2XX^\top$**（$X$ 是该层输入，二阶信息刻画权重间耦合），对权重矩阵**逐列（沿输入维度）量化**到 int4/int3，每量化一列就**用 $H$ 的逆把该列的量化误差"补偿"到尚未量化的剩余列**（让剩余列预先吸收误差，使最终输出误差最小），并用 **Cholesky 分解**保证大模型下数值稳定（这是 GPTQ 相对原始 OBQ 的核心改进）。一次校准（128~1024 样本前向）即完成，不训练、不反传，精度在 4-bit 下接近 fp16（<1% ppl 退化），是 AutoGPTQ / vLLM / TRT-LLM 上 int4 LLM 推理的另一事实标准。核心思想：**量化不是独立地逐个权重 round，而是带着"补偿后续"的序列决策——用二阶信息指导误差分配**。

> [!note] 三句话定位
> - **是什么**：基于 Hessian 的逐列量化 + 误差补偿 PTQ。算 $H=2XX^\top$，逐列量化，每列误差用 $H^{-1}$ 补偿到剩余列，Cholesky 保稳定。
> - **为什么**：朴素逐个 round 权重误差独立累积、大；GPTQ 用二阶信息把误差"推"到尚可调的权重上，序列化最小化输出误差，4-bit 几乎无损。
> - **与 [[AWQ]] 关系**：GPTQ 二阶（重、校准慢、精度略好）；AWQ 一阶（轻、校准秒级、精度接近）。两者 int4 PTQ 双主流。


## 2. 为什么需要它（动机与背景）

### 2.1 朴素逐个 round 误差大

朴素 int4 量化 $\hat W = \text{round}(W/\Delta)\cdot\Delta$，每个权重独立 round，误差 $|\hat W-W|\leq\Delta/2$ 独立累积。对 LLM 大权重矩阵，这些独立误差叠乘 $X$ 后输出误差可观（尤其 3-bit）。问题：**量化各权重时不考虑它们对输出的联合影响（耦合）**——某些权重对输出极敏感，独立 round 会毁精度。

### 2.2 OBS/OBQ：用二阶信息决定"怎么量化"

**OBS（Optimal Brain Surgeon，Hassibi & Stahl 1993）** 原用于剪枝：基于二阶泰勒展开，剪某权重时用 Hessian 逆算"剪它后其余权重怎么调最优"。**OBQ（Optimal Brain Quantization）** 把它推广到量化（固定一个权重到量化值、调其余补偿误差）。思路：量化是**序列决策**——逐个（或逐列）量化，每次把当前误差补偿到尚未量化的权重上，使总输出二阶损失最小。GPTQ 即 OBQ 用于 LLM + Cholesky 稳定 + 批量高效。

### 2.3 GPTQ 的贡献：让 OBQ 在 LLM 上可用

原始 OBQ 两个问题：① 每步要更新 $H^{-1}$（Sherman-Morrison），$d$ 大（4096+）时 O($d^3$) 太慢且数值不稳；② LLM 权重矩阵大，逐权重量化（$d^2$ 步）不可行。GPTQ 改进：**逐列（而非逐权重）批量处理 + Cholesky 分解 $H$ 避免显式求逆更新 + 惰性批处理**，使 OBQ 在 175B 模型上几小时完成。Cholesky 是 GPTQ 相对 OBQ 的关键工程贡献。


## 3. 核心概念详解

### 3.1 Hessian $H = 2XX^\top$

对线性层 $Y = XW$（$X\in\mathbb{R}^{n\times d_{in}}$ 校准输入，$W\in\mathbb{R}^{d_{in}\times d_{out}}$，$Y\in\mathbb{R}^{n\times d_{out}}$），量化损失 $L = \|XW - X\hat W\|_F^2 = \sum_j \|Xw_j - X\hat w_j\|^2 = \text{tr}((W-\hat W)^\top H (W-\hat W))$，其中 **Hessian**：

$$H = 2X^\top X \in \mathbb{R}^{d_{in}\times d_{in}}$$

$H$ 刻画**输入通道间的耦合**（哪些通道共变）。$H_{ij}$ 大 → 通道 $i,j$ 强相关，量化其一要考虑另一。$H$ 是每层算一次（校准数据前向收集每层输入 $X$），之后量化全程复用，不反传。

### 3.2 逐列量化 + 误差补偿（OBQ 核心）

GPTQ 沿 $d_{in}$（输入维）逐列量化 $W$（把 $W$ 看作 $d_{in}$ 列，每列是 $d_{out}$ 维——一个输入通道连到所有输出）。设当前量化第 $i$ 列：

1. **量化**：$\hat w_i = \text{Quantize}(W_{:,i})$（int4，per-group 128），误差 $\delta_i = \hat w_i - W_{:,i} \in \mathbb{R}^{d_{out}}$。
2. **补偿剩余列**：把 $\delta_i$ 的影响"推"到尚未量化的列 $\mathcal{S}=\{i+1,\dots,d_{in}\}$，使它们预先调整以抵消误差。按 OBQ 二阶最优：

$$W_{:,\mathcal{S}} \leftarrow W_{:,\mathcal{S}} - \delta_i \otimes \frac{(H^{-1})_{\mathcal{S},i}}{(H^{-1})_{i,i}}$$

其中 $\frac{(H^{-1})_{\mathcal{S},i}}{(H^{-1})_{i,i}}$ 是 $|\mathcal{S}|$ 维向量（输入位置敏感度），$\delta_i$ 是 $d_{out}$ 维，外积得 $d_{out}\times|\mathcal{S}|$ 更新。**直观**：$H^{-1}$ 告诉"通道 $i$ 量化出错时，其余通道该怎么挪来补偿最优"，二阶信息指导误差分配。

> [!note] 为什么逐列而非逐权重
> LLM $W$ 是 $d_{in}\times d_{out}$（如 4096×4096），逐权重要 $1.6\times10^7$ 步不可行。GPTQ 逐列（沿 $d_{in}$，每列含 $d_{out}$ 个权重一次量化），$d_{in}$ 步（4096 步），且因 $H$ 共享、跨 $d_{out}$ 可批量（vectorize），秒~分钟级完成。

### 3.3 Cholesky 分解（GPTQ 的稳定关键）

原始 OBQ 每步用 Sherman-Morrison 更新 $H^{-1}$，但 $H=2X^\top X$ 在 $d_{in}=4096+$ 时病态（条件数大，因 LLM 输入通道强相关/有 dead 通道 $H_{ii}=0$），显式求逆/更新数值爆炸。**GPTQ 用 Cholesky 分解**：$H = LL^\top$（$L$ 下三角），或在 $H^{-1}$ 上做 Cholesky，把 $(H^{-1})_{\mathcal{S},i}/(H^{-1})_{i,i}$ 的递推变成基于 $L$ 的稳定前代换。同时处理 dead 通道（$H_{ii}=0$ 置 1 防除零）。这是 GPTQ 能在 175B 上跑通的核心——OBQ 在中等模型就崩。

### 3.4 per-group int4（per-128-channel group）

GPTQ 量化用 per-group symmetric int4，group=128（128 个输入通道共享一组 scale $\Delta$）。$\hat W_{int}=\text{round}(W/\Delta)$，$\Delta=\max_g|W|/7$。group 内 round + 跨列 Hessian 补偿结合。也支持 int3/int2（精度递降）。

### 3.5 one-shot PTQ

**一次校准**：用 128~1024 样本前向，收集每层输入 $X$（约 1024 token 序列 × $d_{in}$，内存可接受），算 $H$，逐层逐列量化 + 补偿，存 int4 权重 + scale。之后推理复用，不再校准。不训练、不反传、不更新梯度。是 PTQ（post-training），不是 QAT（quantization-aware training）。

### 3.6 weight-only（W4A16）

与 AWQ 同，GPTQ 是 **weight-only int4**（激活 fp16/bf16），省 decode 带宽，计算时 dequant 回 fp16 做 GEMM（fused kernel）。动机同 [[AWQ]] §2.1（decode memory-bound）。


## 4. 数学原理 / 公式

### 4.1 量化损失与 Hessian

$$L = \|XW - X\hat W\|_F^2 = \text{tr}\!\left((W-\hat W)^\top \underbrace{2X^\top X}_{H} (W-\hat W)\right)$$

$H=2X^\top X$ 半正定。$H_{ij}=2\sum_n X_{ni}X_{nj}$ 度量通道 $i,j$ 在校准数据上的共现强度。

### 4.2 OBQ 误差补偿推导

固定第 $i$ 列量化值 $\hat w_i = w_i + \delta_i$（$\delta_i$ 误差）。剩余列 $\hat w_\mathcal{S}$ 待定，使 $L$ 最小。令 $\partial L/\partial \hat w_\mathcal{S}=0$：

$$H_{\mathcal{S},\mathcal{S}}\,\Delta_\mathcal{S} + H_{\mathcal{S},i}\,\delta_i = 0 \;\Rightarrow\; \Delta_\mathcal{S} = -H_{\mathcal{S},\mathcal{S}}^{-1}H_{\mathcal{S},i}\,\delta_i$$

其中 $\Delta_\mathcal{S}=\hat w_\mathcal{S}-w_\mathcal{S}$ 是剩余列调整量。用**分块逆恒等式**（Sherman-Morrison 递推的等价形式）：

$$\Delta_\mathcal{S} = -\delta_i\,\frac{(H^{-1})_{\mathcal{S},i}}{(H^{-1})_{i,i}}$$

这是 OBS/OBQ 的标准结果：**量化第 $i$ 列后，剩余列最优调整为误差 $\delta_i$ 乘以 $H^{-1}$ 第 $i$ 列归一**。$(H^{-1})_{i,i}$ 越小（该方向敏感）补偿越大。

> [!note] 逐列（沿 $d_{in}$）的合理性
> $L=\sum_j\|Xw_j-X\hat w_j\|^2$ 按**输出列** $j$ 分离（各输出列独立，共享 $H$）。故补偿沿 $d_{in}$（输入维）进行：量化输入通道 $i$ 影响所有输出列，用共享 $H$ 跨 $d_{out}$ 批量补偿。这就是"逐输入列、跨输出批量"。

### 4.3 Cholesky 稳定化

$H=2X^\top X$ 病态时，显式 $(H^{-1})_{\mathcal{S},i}/(H^{-1})_{i,i}$ 数值炸。GPTQ：对 $H$ 加阻尼 $H \leftarrow H + \lambda I$（$\lambda\sim10^{-2}\sim10^{-3}$ 平滑 dead 通道），算 Cholesky $H=LL^\top$。则 $H^{-1}=L^{-\top}L^{-1}$，补偿递推用 $L$ 的前代换稳定计算，免每步显式求逆。复杂度从 OBQ 的 $O(d^3)$ 降到 $O(d^2\cdot d_{out})$（每列 $O(d\cdot d_{out})$，$d$ 列）。

### 4.4 误差上界

单列量化误差 $\|\delta_i\|\leq\sqrt{d_{out}}\,\Delta/2$（$\Delta$ group 步长）。补偿后剩余输出误差二阶项 $\propto \delta_i^\top (H^{-1})_{ii}^{-1}\delta_i$——OBQ 把"独立 round 误差"降到"最优分配后的二阶下界"，理论优于朴素量化。实测 int4 接近 fp16，int3 退化可控。


## 5. 代码示例（可选）

### 5.1 手写 GPTQ 逐列量化 + Hessian 补偿（最小实现）

```python
import torch

def gptq_quantize_layer(W, X, group_size=128, nbits=4, damping=0.01):
    """
    W: [d_in, d_out]  (待量化, 行=输入通道)
    X: [n, d_in]      (校准输入)
    返回 int4 反量化后的 W_hat
    """
    d_in, d_out = W.shape
    H = 2.0 * (X.t() @ X)                       # Hessian [d_in, d_in]
    H += damping * torch.eye(d_in, device=W.device)  # 阻尼防病态
    Hinv = torch.linalg.inv(H)                  # 真实 GPTQ 用 Cholesky, 这里简化用 inv
    W = W.clone()
    qmax = 2**(nbits-1) - 1                      # 7 for int4

    for i in range(d_in):                        # 逐输入列量化
        # per-group scale (简化: 每 group_size 通道一组; 这里按列 i 的 group)
        g = i // group_size
        gs = g * group_size; ge = min((g+1)*group_size, d_in)
        Wg = W[gs:ge]                            # 该 group 的行
        delta_g = Wg.abs().amax() / qmax
        # 量化第 i 列 (输入通道 i, 即 W 的第 i 行)
        q = torch.round(W[i] / delta_g).clamp(-qmax-1, qmax)
        d = q * delta_g - W[i]                   # 量化误差 delta_i [d_out]
        W[i] = q * delta_g                       # 写入量化值
        # OBQ 补偿: 剩余列 W[i+1:] -= d ⊗ (Hinv[i+1:, i] / Hinv[i,i])
        if i < d_in - 1:
            coef = Hinv[i+1:, i] / Hinv[i, i]    # [d_in-i-1]
            W[i+1:] -= torch.outer(coef, d)      # [d_in-i-1, d_out]
    return W

torch.manual_seed(0)
d_in, d_out, n = 256, 256, 64
W = torch.randn(d_in, d_out) * 0.05
X = torch.randn(n, d_in)
W_hat = gptq_quantize_layer(W, X, group_size=128, nbits=4)
y_true, y_hat = X @ W, X @ W_hat
print(f"GPTQ 输出 MSE: {((y_hat-y_true)**2).mean():.4e}")
print(f"朴素 int4 MSE: {((X@(torch.round(W/ (W.abs().amax()/7))* (W.abs().amax()/7)) - y_true)**2).mean():.4e}")
```

### 5.2 用 AutoGPTQ 跑 GPTQ 量化（生产）

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "Llama-2-7b-gptq"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True)
model = AutoGPTQForCausalLM.from_pretrained(model_path, config)
# 校准数据 (如 wikitext/pile 切 1024 样本)
calib = [tokenizer("...").input_ids for _ in range(128)]
model.quantize(calib)              # 算 H + 逐列量化 + Cholesky 补偿
model.save_quantized(quant_path)
```

### 5.3 vLLM 加载 GPTQ 模型推理

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Llama-2-7b-gptq \
    --quantization gptq \
    --dtype float16          # W4A16, Marlin kernel
# vLLM 默认用 Marlin kernel (比原 GPTQ kernel 快), decode ~3-4x fp16
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[数值类型与精度]]（int4 位级、量化基本式）、二阶优化/Hessian（OBS/OBQ 理论基础）、矩阵求逆/Cholesky 分解（线性代数）、[[mixed precision training]]（量化是 AMP 之外的压低）、[[memory bandwidth]] / [[Roofline模型]]（weight-only int4 省 decode 带宽的动机）。
- **下游（应用）**: [[新模型接入]]（vLLM `--quantization gptq`）、[[model runner]]（Marlin/GPTQ fused dequant+GEMM kernel）、[[sampling throughput]]（int4 decode 提吞吐）、[[Tensor Parallel]]（GPTQ 权重按 TP 切）、AutoGPTQ/TRT-LLM 工具链。
- **对比 / 易混**:
  - **GPTQ vs [[AWQ]]**：见下表与 [[AWQ]] §6。GPTQ 二阶 Hessian 补偿（重、精），AWQ 一阶激活 scaling（轻、快）。
  - **GPTQ vs QAT**：GPTQ 是 PTQ（训后、校准、不反传）；QAT 是训练时量化（前向模拟量化、反传更新）。GPTQ 不碰训练。
  - **GPTQ vs [[INT8推理]] W8A8**：GPTQ weight-only int4（省带宽），W8A8 权重+激活 int8（省算力）。
  - **GPTQ vs [[FP8量化方案]]**：FP8 浮点 8-bit 训练推理通用；GPTQ 定点 4-bit 仅推理 weight-only。
  - **GPTQ vs [[FP4与低比特]] NF4**：NF4 浮点 4-bit（正态分布最优，QLoRA 微调）；GPTQ 定点 4-bit + Hessian 补偿（推理 PTQ）。

### GPTQ vs AWQ 对比表

| 维度 | GPTQ | AWQ |
|---|---|---|
| 理论基础 | OBS/OBQ 二阶 Hessian 误差补偿 | 激活幅值一阶 + per-channel scaling |
| Hessian | 算 $H=2XX^\top$，用 $H^{-1}$ 补偿 | 不算 Hessian |
| 校准成本 | 中（算 H + Cholesky，分钟级） | 低（前向统计，秒级） |
| 逐列补偿 | 是（误差推到剩余列） | 否（只 scale 保护显著列） |
| 精度 int4 | 略优（二阶补偿） | 略逊（0.1~0.5% ppl 差） |
| 精度 int3 | 明显优于 AWQ | 退化较大 |
| 数值稳定 | 需 Cholesky + 阻尼 | 天然稳定 |
| 框架 | AutoGPTQ、vLLM(`--quantization gptq`)、TRT-LLM | AutoAWQ、vLLM(`--quantization awq`)、TRT-LLM |
| 推理 kernel | Marlin（与 AWQ 通用） | Marlin |


## 7. 常见误区与易错点

> [!warning] 误区 1：GPTQ 是训练时量化（QAT）
> 不是。GPTQ 是**后训练量化（PTQ）**——模型训完，用校准数据前向算 Hessian 逐列量化，不训练、不反传、不更新权重梯度。虽涉及 Hessian（二阶信息），但 Hessian 由校准前向统计 $H=2XX^\top$ 得到，不是反传算的。PTQ 与 QAT 的区别见 [[mixed precision training]]。

> [!warning] 误区 2：GPTQ 量化激活
> GPTQ 是 **weight-only int4**（W4A16），激活 fp16/bf16 不量化。"用 Hessian"指指导**权重**量化误差分配，不是量化激活。要量化激活走 [[INT8推理]] W8A8 / SmoothQuant 或 [[FP8量化方案]]。

> [!warning] 误区 3：Hessian 是反传算的
> 不是。$H=2X^\top X$ 是**校准数据前向**收集每层输入 $X$ 算的（外积和），无需反传。Hessian 在这里是"输入通道二阶矩"，不是训练梯度的二阶导。别和训练里的"Hessian-vector product 反传"混。

> [!warning] 误区 4：GPTQ 不需校准数据
> 需要。要算 $H=2X^\top X$ 必须有校准输入 $X$（128~1024 样本）。"one-shot"指一次校准即完成（不像 QAT 要多 epoch 训练），不是说零校准。校准数据要代表部署分布。

> [!warning] 误区 5：逐列量化是逐输出列
> 易混。GPTQ 逐**输入维**（$d_{in}$）量化（沿输入通道序列决策），跨输出维（$d_{out}$）批量补偿——因 $L=\sum_j\|Xw_j-X\hat w_j\|^2$ 按输出列分离但共享 $H$。权重矩阵存储约定（行/列）不同实现不同，但补偿沿 $d_{in}$ 是本质。

> [!warning] 误区 6：Cholesky 可省
> 小模型可，大 LLM（$d_{in}\geq4096$）原始 OBQ 无 Cholesky 会数值爆炸（$H$ 病态、dead 通道 $H_{ii}=0$）。Cholesky + 阻尼是 GPTQ 能 scale 到 175B 的关键，不是可选优化。

> [!tip] 实践：desc_act（按 Hessian 对角排序）
> GPTQ 可选 `desc_act=True`——按 $H_{ii}$ 降序量化（先量化敏感列），精度略好但需重排、慢。默认 False（顺序量化）够用。int3/低比特时建议开。

> [!tip] 实践：校准样本数
> 128 够 int4，1024 更稳（尤其大模型/低比特）。太多无益且慢。用与部署同分布数据。


## 8. 延伸细节

### 8.1 与 OBQ 的渊源

OBQ（Frantar et al. 2022）是 GPTQ 的直接前身，把 OBS 从剪枝推广到量化。GPTQ = OBQ + Cholesky 稳定 + 逐列批量，使 OBQ 实用于 LLM。OBS 又源自 OBD（Optimal Brain Damage，LeCun 1990）——一脉二阶信息网络压缩。

### 8.2 Marlin kernel（GPTQ/AWQ 通用）

GPTQ 与 AWQ 的 int4 权重推理最终都走 **Marlin**（vLLM 集成的 universal int4 kernel）——dequant + GEMM 融合，[[fused kernel]] 在量化的应用。原 GPTQ-GEMM kernel 慢，新 vLLM 默认 Marlin，AWQ/GPTQ 权重格式 Marlin 都吃。故推理速度两者趋同，选型主要看校准成本与精度偏好。

### 8.3 content来源

GPTQ 整理自 Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"（ICLR 2023）、OBS（Hassibi & Stahl 1993）、OBQ（Frantar et al. 2022 "Optimal Brain Quantization"）、AutoGPTQ GitHub、vLLM `--quantization gptq` 文档与 `vllm/model_executor/layers/quantization/gptq/` 源码、Marlin kernel。与 AWQ 对照见 [[AWQ]]。

---
相关: [[量化]] | [[AWQ]] | [[INT8推理]] | [[FP4与低比特]] | [[FP8量化方案]] | [[数值类型与精度]] | [[mixed precision training]] | [[memory bandwidth]] | [[Roofline模型]] | [[新模型接入]] | [[model runner]] | [[Tensor Parallel]] | [[fused kernel]] | [[sampling throughput]] | [[loss scaling]]
