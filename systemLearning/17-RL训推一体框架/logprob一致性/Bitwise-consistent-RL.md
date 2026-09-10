# Bitwise-consistent-RL

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §83
> **别名**: Bitwise Consistent Train-Inference / batch-invariant RL / bit 级一致 RL
> **难度**: 高（需懂 [[训推不一致]]、[[forward与backward]]、[[torch.compile]]、vLLM/TorchTitan）
> **来源**: vLLM Blog, *No More Train-Inference Mismatch: Bitwise Consistent On-Policy RL with vLLM and TorchTitan*, 2025-11-10. URL: https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference

## 1. 一句话定义

**Bitwise-consistent RL** 是 vLLM 项目组提出的**工程侧根治**方案：把 vLLM 的 **batch-invariant 推理 kernel** 直接注入 TorchTitan 训练框架的 forward pass，并为每个算子注册 PyTorch 自定义 backward，使训练与推理走**逐 bit 相同**的 kernel 代码，KL 恒等于 0，从工程上彻底消除 [[训推不一致]]（TIM）——与算法层（TIS/IcePop/DPPO 等）是**替代而非互补**关系。

## 2. 为什么需要它（动机与背景）

### 2.1 标准 RL 的两套 kernel

训练框架（Megatron/FSDP/TorchTitan）与推理框架（vLLM/SGLang）因工作负载不同，用**截然不同**的 kernel 库。即便加载同一权重，kernel 实现差异 + 浮点非结合性让两套 logp 不等。算法层方法（TIS/IcePop/DPPO）是"patch around"——用 IS 修正/裁剪/丢弃来**补偿** mismatch，永远追不上 mismatch 的增长。

### 2.2 根治思路：让两套用同一套 kernel

不是"修 mismatch"，而是"让 mismatch 不存在"——把 vLLM 的 batch-invariant kernel 搬进训练 forward，训推共享 kernel → bit 一致 → KL≡0。

### 2.3 batch invariance 是关键

Thinking Machines（Horace He, 2025-09 博客）澄清：单 kernel run-to-run 是确定的（同 matmul 跑 1000 次结果相同），真正的非确定来自**批量变化触发不同归约策略**（matmul Split-K、RMSNorm 分卡归约、attention split-size）。VeXact（[[VeXact]]）也独立诊断到同一结论。batch-invariant kernel 固定归约顺序与 tiling，使"批量变结果不变"。

## 3. 核心概念详解

### 3.1 三步工程实现

1. **Forward**：训练 forward 直接调用 vLLM batch-invariant kernel（含融合算子：SiLU MLP、带残差的 RMSNorm），保证训推前向数值相同
2. **Backward**：为每个 kernel 写简单 vanilla PyTorch 的自定义反向，注册为 autograd function，保持等价
3. **同步执行**：trainer 与 generator 在单 host 交替同步运行——"exactly on-policy execution" 的演示

### 3.2 验证

- 开 `batch_inv_ON`：KL **恒等于 0.0**，证实 bit 一致
- 关 `batch_inv_OFF`（kernel 不一致）：100 步内 reward 下降
- bit 一致不仅训得步数少，且总 reward 更高
- Demo：Qwen3-1.7B + GSM8K + correctness reward
- 代价：**2.4× 慢**

### 3.3 与算法层的关系：替代非互补

> 算法层（IS 权重、KL penalty、clip）是"补偿 mismatch"；bit 一致是"消除 mismatch 源头"。一旦 bit 一致，KL penalty 恒为 0、IS 修正无对象——**根本不需要算法层补丁**。这是哲学转向：从"打补丁"到"修根因"。

## 4. 数学原理 / 公式

bit 一致下：

$$\forall t:\ \log\pi_{\text{train}}(a_t|s_t)\stackrel{!}{=}\log\pi_{\text{infer}}(a_t|s_t)\ \Rightarrow\ \rho_{\text{TIM}}=\exp(\log\pi_{\text{train}}-\log\pi_{\text{infer}})=1$$

$$D_{KL}[\pi_{\text{train}}\Vert\pi_{\text{infer}}]\equiv0$$

即 [[rollout train reference logprob一致性]] §4.1 里的 $\epsilon_t\equiv0$，TIM 引入的 $\rho$ 噪声与有偏完全消失。

## 5. 代码示例

```python
# 概念: 训练 forward 直接用 vLLM 的 batch-invariant kernel
# 而非 Megatron/FSDP 自己的 kernel, 并注册自定义 backward

# import torch
# class BatchInvariantRMSNorm(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, w, eps):
#         # 用 vLLM 的 batch-invariant RMSNorm (固定归约顺序, 不分卡)
#         out = vllm_rmsnorm(x, w, eps)
#         ctx.save_for_backward(x, w, out)
#         return out
#     @staticmethod
#     def backward(ctx, g):
#         # 简单 vanilla PyTorch 反向, 保持与 forward 等价
#         x, w, out = ctx.saved_tensors
#         ...  # 略
#         return gx, gw, None

# 验证 bit 一致: 同输入训/推 forward, KL 应=0
# logp_train = forward_train(x)   # 用 vLLM kernel
# logp_infer = forward_infer(x)  # 同一 kernel
# assert torch.equal(logp_train, logp_infer)  # bit 级一致
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[训推不一致]]（TIM 根因，本条是工程根治）、[[forward与backward]]（自定义 backward）、[[torch.compile]]（当前需 eager 模式）、batch-invariant kernel 设计（Horace He blog）
- **下游（应用）**: [[训练推理不一致TIM技术综述]]（工程根治代表）、exactly on-policy RL
- **对比 / 易混**:
  - **Bitwise（kernel 根治）vs 算法层（TIS/IcePop/DPPO）**：**替代**关系——bit 一致后算法层补丁无对象。但 bit 一致 2.4× 慢，生产吞吐受限时算法层仍作兜底。
  - **Bitwise vs [[FP16-fix]]**：FP16 降 mismatch ~24×（够小到不伤 PPO但非零）；Bitwise 让 mismatch=0。前者几乎无开销、后者慢。
  - **Bitwise vs [[VeXact]]**：VeXact 是诊断标尺（rollout 端零 mismatch 做基准）；Bitwise 是生产链路（训推全链 bit 一致）。
  - **Bitwise vs [[R3 rollout routing replay]]**：R3 只对齐 MoE 路由离散选择（保留 softmax 数值自由度）；Bitwise 连续数值也对齐。R3 <3% 开销，Bitwise 2.4×。

## 7. 常见误区与易错点

> [!warning] 误区 1：以为 bit 一致是"训推都关非确定算法"
> 不只是关非确定算法。还要**训推共享同一套 kernel 实现**（vLLM kernel 注入 TorchTitan），否则两套不同 kernel 即使各自确定，结果仍不同。

> [!warning] 误区 2：bit 一致就一定比算法层好
> 2.4× 慢是实打实的吞吐损失。生产追求极致吞吐时，FP16-fix + 算法层兜底可能更经济。bit 一致是"最严但最贵"档，见综述 §8 场景选型。

> [!warning] 误区 3：bit 一致后还要 KL penalty
> 不需要。KL 恒为 0，KL penalty 项无意义。算法层补丁全部可撤。

## 8. 延伸细节

### 8.1 当前局限与未来方向
- 两份模型代码副本（脆弱，维护成本高）
- 需 eager 模式（禁 vLLM 编译优化）
- 2.4× 慢待优化
- 模型支持窄（仅 Qwen3-1.7B demo）

### 8.2 与 Thinking Machines batch-invariant kernel 的关系
vLLM 的 batch-invariant kernel（matmul 固定配置避 Split-K、RMSNorm 数据并行避分卡归约、attention 固定 split-size）正是 Horace He 博客主张的实现。VeXact 也用同思路。

### 8.3 内容来源
三步实现、KL=0、2.4× 慢、Qwen3-1.7B demo、"替代非互补"定位均来自 vLLM blog 2025-11-10 已联网核实。batch invariance 根因来自 Thinking Machines blog 2025-09-10。截至 2026-09。

---
相关: [[训练推理不一致TIM技术综述]] | [[训推不一致]] | [[FP16-fix]] | [[VeXact]] | [[R3 rollout routing replay]] | [[forward与backward]] | [[torch.compile]] | [[rollout train reference logprob一致性]] | [[17-RL训推一体框架]]
