# FP16-fix

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §83
> **别名**: Defeating TIM via FP16 / FP16 defeats TIM / 精度根治 TIM
> **难度**: 中（需懂 [[训推不一致]]、[[块缩放浮点格式]]、[[mixed precision training]]、[[rollout train reference logprob一致性]]）
> **来源**: Qi et al. (Sea AI Lab / NUS), *Defeating the Training-Inference Mismatch via FP16*, arXiv:2510.26788 (2025-10). 代码: https://github.com/sail-sg/Precision-RL

## 1. 一句话定义

**FP16-fix** 指出 [[训推不一致]]（TIM）的**根因是浮点精度**：BF16 只有 7 位 mantissa，训推两套引擎的舍入误差累积破坏一致性；改用 **FP16（10 位 mantissa，比 BF16 多 3 位 = 精度高 8 倍）** 能把序列级 logp 比值的 mismatch 降低约 **24×**，从精度路径上根治 TIM，使多种 RL 算法在 sanity test 上从崩溃/震荡提升到 ~99%。

## 2. 为什么需要它（动机与背景）

### 2.1 BF16 是 RL 微调的"精度短板"

预训练用 BF16 是为了**动态范围**（8 位 exponent，防溢出）。但 RL 微调阶段，权重/激活的动态范围已在预训练期确立，BF16 的宽范围**用不上**，而它牺牲的 mantissa 精度反而成了 TIM 的主因：

| 格式 | exponent | mantissa | 动态范围 | 精度 |
|---|---|---|---|---|
| BF16 | 8 | **7** | 大（≈FP32） | 低 |
| FP16 | 5 | **10** | 小 | 高（8×） |
| FP32 | 8 | 23 | 大 | 最高 |

FP16 比 BF16 多 3 位 mantissa → 精度高 $2^3=8$ 倍。这多出的精度"吸收"了两套引擎实现间的细微差异。

### 2.2 误差的自回归累积

RL rollout 自回归生成几百~几千 token，每步 forward 的舍入误差沿序列累积。BF16 的粗 mantissa 让累积误差快速越过"PPO $\rho$ 偏离 1"的容忍线（见 [[rollout train reference logprob一致性]] §4.1）。FP16 的细 mantissa 把单步误差压小，累积后仍在线下。

### 2.3 为什么不是 FP32

FP32 推理稳定但**~3× 慢**，工程不可行。FP16 在 H100/A100 上有硬件加速，速度接近 BF16，是"精度够、速度不亏"的甜点。

## 3. 核心概念详解

### 3.1 TIM 的有偏梯度

论文给出有偏策略梯度：从行为策略 $\mu$（推理引擎算）采样但梯度在 $\pi$（训练引擎算）下计算：

$$\nabla_\theta J_{\text{biased}}(x,\theta)=\mathbb{E}_{y\sim\mu(\cdot|x,\theta)}\left[\nabla_\theta\log\pi(y|x,\theta)\cdot R(x,y)\right]\ne\nabla_\theta J(x,\theta)$$

部署 gap：$\arg\max_\theta\mathbb{E}[R]\text{ under }\mu\ne\arg\max_\theta\mathbb{E}[R]\text{ under }\pi$。FP16 让 $\mu\approx\pi$（同权重下两引擎算出的分布几乎一致），消除 gap。

### 3.2 序列级重要性比

$$\rho=\frac{\pi(y|x,\theta')}{\mu(y|x,\theta')}$$

BF16 下 $\rho$ 偏离 1（mismatch 24×大）；FP16 下 $\rho\approx1$。

### 3.3 关键结论

> **只改推理端精度没用，必须训推同改 FP16。** mismatch 是两套引擎的**差**，单边改一边等于没改。

## 4. 数学原理 / 公式

### 4.1 mantissa 位数 → 精度

相对舍入误差 $\approx 2^{-(\text{mantissa}+1)}$：
- BF16：$2^{-8}\approx3.9\times10^{-3}$
- FP16：$2^{-11}\approx4.9\times10^{-4}$

单步精度差 ~8×。经 $T$ 步累积（保守按 $\sqrt T$ 根号增长），序列级 mismatch 放大，FP16 相对 BF16 降 ~24×（论文 Fig.2 实测）。

### 4.2 Sanity test（可完美子集 MATH，1460 题）

| 配置 | 训练准确率 |
|---|---|
| BF16 GRPO | 峰值 73%/84% 后**崩溃** |
| BF16 Seq-MIS | 稳定但峰值 ~95% |
| **FP16（所有算法）** | **~99%** |

### 4.3 下游基准

- AIME 2024：BF16 Seq-MIS = 34% → **FP16 = 39%**
- MoE（Qwen3-30B-A3B）：FP16 更稳、训练准确率更高
- OctoThinker-3B：BF16 ~150 步后失稳，FP16 平滑
- mismatch 分析（Fig.2）：FP16 把序列级 logp 比值 mismatch 降 ~24×；BF16 下响应越长分歧越大
- 受 Andrej Karpathy 背书

## 5. 代码示例

```python
# verl/Oat 配置: 训推同改 FP16 (单改一边无效!)
# training engine (FSDP/DeepSpeed)
bf16: false
fp16: true          # ★ 训练 forward 用 FP16 (非 bf16)

# rollout engine (vLLM)
# vLLM 启动:
#   --dtype float16   # ★ 推理也用 FP16 (与训练一致), 不是 bfloat16
#   # 关掉 FP8 KV cache (--kv-cache-dtype 不设 fp8), 否则又引入精度路径 TIM

# 验证: θ_new=θ_old 时 ratio 应≈1
import torch
T = 1000
logp_old = torch.randn(T)*2 - 5       # rollout (FP16 引擎)
tim_bf16 = torch.randn(T)*0.03        # BF16 级分歧
tim_fp16 = torch.randn(T)*0.00125     # FP16 级分歧 (约 24× 小)
rho_bf16 = torch.exp(logp_old - (logp_old - tim_bf16))
rho_fp16 = torch.exp(logp_old - (logp_old - tim_fp16))
print(f"BF16 ratio std: {rho_bf16.std():.4f}  (偏离 1, clip 误触发)")
print(f"FP16 ratio std: {rho_fp16.std():.6f} (≈1, 稳)")
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[训推不一致]]（TIM 根因=精度，本条是其精度路径的具体根治）、[[块缩放浮点格式]]（BF16/FP16/MXFP8 格式定义）、[[mixed precision training]]（FP32 master + BF16 forward 的训练侧精度）、[[FP8量化方案]]（FP8 路径 TIM 更严重，FP16 是"够用且稳"的折中）
- **下游（应用）**: [[训练推理不一致TIM技术综述]]（FP16-fix 是引擎层"精度根治"代表）、低精度 RL 稳定性
- **对比 / 易混**:
  - **FP16-fix（精度根治）vs [[Bitwise-consistent-RL]]（kernel 根治）**：前者改 dtype 降误差量级，后者对齐 kernel 让误差为零。FP16 是"够小到不伤 PPO"，Bitwise 是"严格为零"。前者几乎无开销，后者慢 2.4×。
  - **FP16 vs FP32**：FP32 也消 TIM 但 3× 慢，工程不可行；FP16 是速度-精度甜点。
  - **FP16-fix vs [[R3 rollout routing replay]]**：FP16 治 dense + MoE 数值型 TIM；MoE 路由**结构型** TIM 仍需 R3（FP16 把概率差压小但 TopK 边界仍可能跳变）。
  - **FP16-fix vs [[VeXact]]**：VeXact 是诊断工具（零 mismatch 测真值），FP16-fix 是生产手段。

## 7. 常见误区与易错点

> [!warning] 误区 1：只把推理改 FP16 就行
> 不行。mismatch 是两套引擎的**差**，单改推理端、训练仍是 BF16，差不变。必须训推**同改** FP16。

> [!warning] 误区 2：FP16 一定不会溢出
> FP16 动态范围小（$\sim\pm65504$）。极端大模型/大梯度场景可能溢出。论文未声称 FP16 普遍最优，超大模型需评估溢出风险。

> [!warning] 误区 3：FP16 能完全消 MoE 路由 TIM
> 不能。FP16 把数值差压 ~24×，但 MoE TopK 边界 token 只要 router logit 差一点仍可能跳专家。MoE 需 FP16 + [[R3 rollout routing replay]] 叠加。

> [!warning] 误区 4：FP8 推理 + FP16 训练能对齐
> 精度路径不同（FP8 vs FP16）仍是 TIM 来源。FP8 rollout 需专门对齐方案（见 [[FP8量化方案]]、综述 §8 场景③）。

## 8. 延伸细节

### 8.1 框架差异残留
FP16 大幅降 mismatch 但**不归零**——VeRL 即便 FP16 仍偶现数值尖峰（框架特定差异）。要完全归零需 [[Bitwise-consistent-RL]]。

### 8.2 与算法层方法正交
FP16-fix（精度）可与 IcePop/TIS/DPPO（算法层）叠加：FP16 降 baseline mismatch，算法层兜底残余。论文在 FP16 下测了 GRPO/Dr.GRPO/TIS/MIS/GSPO/PG-Seq-IS/DAPO 多算法均达 ~99% sanity。

### 8.3 内容来源
mantissa 位数、24× mismatch 降低、sanity test 99%、AIME24 34%→39%、Karpathy 背书均来自 arXiv:2510.26788（2025-10-30 提交）全文已联网核实。代码库 sail-sg/Precision-RL。截至 2026-09。

---
相关: [[训练推理不一致TIM技术综述]] | [[训推不一致]] | [[块缩放浮点格式]] | [[mixed precision training]] | [[FP8量化方案]] | [[Bitwise-consistent-RL]] | [[VeXact]] | [[R3 rollout routing replay]] | [[rollout train reference logprob一致性]] | [[17-RL训推一体框架]]
