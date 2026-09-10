# IcePop

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §83
> **别名**: IcePop / token-level discrepancy masking / 双边 ratio masking
> **难度**: 高（需懂 [[训推不一致]]、[[rollout train reference logprob一致性]]、[[importance sampling与off-policy correction]]、[[MoE路由]]）
> **来源**: Ling Team (Ant Group / inclusionAI), *Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model* (Ring-1T), arXiv:2510.18855 (2025-10). 代码随 Ring-1T 开源。

## 1. 一句话定义

**IcePop** 是 Ring-1T 提出的 **token 级"训推概率比"双边 masking** 方法：对每个 token 算训练引擎与推理引擎在**同一旧权重 $\theta_{old}$** 下算出的概率比 $k=\pi_{\text{train}}(a;\theta_{old})/\pi_{\text{infer}}(a;\theta_{old})$，若 $k$ 落在 $[\alpha,\beta]$ 之外（训推分歧过大）就把该 token 的**梯度贡献置零**（mask=0），从而抑制 [[训推不一致]]（TIM）带来的噪声梯度，稳定 MoE/大模型 RL 训练。

> [!note] 关键区分
> IcePop 作用在**训推概率比 $k$**（衡量两套引擎的数值分歧），**不是** PPO 的 importance ratio $r_{i,t}=\pi_\theta/\pi_{\theta_{old}}$（衡量权重更新）。两套 ratio 正交：$k$ 管"同权重下引擎算得准不准"，$r$ 管"权重动没动"。详见 [[rollout train reference logprob一致性]] §3.7。

## 2. 为什么需要它（动机与背景）

### 2.1 TIM 让 $\rho$ 在 $\theta$ 未动时偏离 1

见 [[训推不一致]] §3.3、[[rollout train reference logprob一致性]] §2.3：rollout（vLLM）与 trainer（FSDP/Megatron）即便加载同一 $\theta_{old}$，算出的 $\log\pi$ 也有数值差 $\delta_{\text{TIM}}$，导致 PPO 的 $\rho=\exp(\delta_{\text{TIM}})\ne1$，梯度被污染。

### 2.2 概率分歧的指数放大

IcePop 论文给出 **Theorem 1（概率分歧复合）**：设单步训推 KL $\delta_t=D_{KL}(\pi_{\text{infer}}(\cdot;\theta_t)\Vert\pi_{\text{train}}(\cdot;\theta_t))$，步长 $\mu>0$，则存在 $\eta>0$ 使

$$\delta_{t+1}\ge\left(1+\tfrac{\eta}{2}\mu\right)\delta_t$$

即训推分歧**跨迭代指数增长**。证明思路：把梯度拆成 $g_t=g_t^*+b_t$（$b_t$ 为 TIM 引入的偏差项），当 $\delta_t\ge 2\kappa/\eta$ 时偏差复合放大。这解释了为什么 MoE RL 训练会在 180–200 步附近突然崩溃——分歧累积到临界点后爆炸。

### 2.3 为何不直接 recompute/R3

[[rollout train reference logprob一致性]] §3.6 的 recompute 策略能消数值型 TIM 但多一次 forward（贵）；[[R3 rollout routing replay]] 只治 MoE 路由结构性 TIM。IcePop 走**算法层兜底**：不消根因，而是把"训推分歧过大"的 token 梯度丢弃，让噪声不进梯度。代价是丢少量数据，收益是不改引擎、即插即用。

## 3. 核心概念详解

### 3.1 训推概率比 $k$

$$k=\frac{\pi_{\text{train}}(a;\theta_{old})}{\pi_{\text{infer}}(a;\theta_{old})}$$

- 分子：trainer（FSDP/Megatron）用 $\theta_{old}$ forward 算的 token 概率
- 分母：rollout（vLLM/SGLang）生成时记的 token 概率
- $k\approx1$：两引擎一致；$k$ 远离 1：该 token 训推分歧大，梯度不可信

### 3.2 双边 masking 函数

$$M(k;\alpha,\beta)=\begin{cases}k,& k\in[\alpha,\beta]\\0,&\text{otherwise}\end{cases}$$

- $\alpha=0.5,\ \beta=5.0$（Ring-1T 默认）：训推概率比在 $[0.5,5]$ 内的 token 保留，超出则梯度置零
- 双边 = 既截"训远大于推"（$k>\beta$）也截"训远小于推"（$k<\alpha$）

### 3.3 IcePop 目标

$$J_{\text{IcePop}}(\theta)=\mathbb{E}\left[\frac1G\sum_i\frac1{|y_i|}\sum_t\left[M\!\left(\frac{\pi_{\text{train}}(y_{i,t}|\cdot;\theta_{old})}{\pi_{\text{infer}}(y_{i,t}|\cdot;\theta_{old}};\alpha,\beta\right)\cdot\min\!\left(r_{i,t}\hat A_{i,t},\,\text{clip}(r_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t}\right)-\gamma D_{KL}(\pi_\theta\Vert\pi_{\text{ref}})\right]\right]$$

- $r_{i,t}=\pi_{\text{train}}(y_{i,t}|\cdot;\theta)/\pi_{\text{train}}(y_{i,t}|\cdot;\theta_{old})$：标准 PPO ratio（分子分母都出自训练引擎，无 TIM）
- $M(\cdot)$ 乘在 PPO clip 项之前：训推分歧大的 token 整项被置零
- Ring-1T 配置：$\gamma=0$（无 KL）、温度 1.0、480 prompts×8 rollouts/step、max_len 65536

> [!tip] 为什么用 $M$ 乘 PPO 项而不是改 $\rho$
> 若直接把 $\rho=\pi_\theta/\pi_{\text{infer}}$ 用训推混合比，会把引擎数值差混进策略更新信号。IcePop 只用训练引擎算的 $r$ 做策略更新，另用 $k$ 做"这个 token 该不该信"的 gate——把"可信度"与"策略梯度"解耦。

## 4. 数学原理 / 公式

### 4.1 masking 体积

Ring-1T 实测：默认 $[\alpha,\beta]=[0.5,5.0]$ 下只裁剪 **~1–2‰（0.1–0.2%）** 的训练 token；这些被裁 token 的熵偏高（恰是 TopK 边界、路由易跳变的 token）。这是 IcePop"少数据换稳训练"的关键数据点。

### 4.2 敏感性（Ring-mini-2.0, 16.8B/0.75B active MoE, AIME-2025 Avg@64）

| 区间 | 行为 |
|---|---|
| $[0.5,5.0]$ 默认 | 稳定，多样性丰富 |
| $[0.5,2.0]$ 收窄 | **立即失稳** |
| $[0.4,5.0]$ 放宽 | 稳定但放进高 logp token |

### 4.3 与 TIS 的对比实验

- 基线 vanilla GRPO（无 KL）在 step 180–200 **崩溃**
- TIS 比 IcePop 低 ~6%（AIME-2025）
- IcePop 保持比 TIS 更低的 logp → 更大探索空间
- 训练动力学（Fig.6）：无 IcePop 时梯度范数 + 概率分歧快速攀升；有 IcePop 时分歧 400 步内下降

### 4.4 Ring-1T 最终基准

| 基准 | Ring-1T | DeepSeek-V3.1-Terminus | Qwen3-235B |
|---|---|---|---|
| AIME 2025 (Avg@64) | **93.40** | 89.06 | 92.30 |
| HMMT25 (Avg@16) | **86.72** | 86.10 | 83.90 |
| CodeForces (rating) | **2088** | 2073 | 2055 |

## 5. 代码示例

```python
import torch

def icepop_mask(k, alpha=0.5, beta=5.0):
    """双边 token 级 ratio masking.
    k: (T,) 训推概率比 π_train/π_infer (同 θ_old)
    返回 mask: (T,) ∈ {1, 0}（保留/置零梯度）
    """
    return torch.where((k >= alpha) & (k <= beta), 1.0, 0.0)

# 模拟: θ 未动, 纯引擎数值差
torch.manual_seed(0)
T = 1000
logp_train = torch.randn(T) * 2 - 5          # trainer 算的 logp
tim_noise = torch.randn(T) * 0.3            # 引擎分歧 (MoE 路由边界 token 偏大)
logp_infer = logp_train - tim_noise          # vLLM 算的 logp
k = torch.exp(logp_train - logp_infer)       # 训推概率比
mask = icepop_mask(k, alpha=0.5, beta=5.0)
print(f"masking 比例: {(1-mask).mean():.4f} (期望 ~0.2%)")
# 被裁的是 k 超出 [0.5,5] 的 token, 即训推分歧大的边界/路由跳变 token
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[训推不一致]]（TIM 机理，IcePop 的治理对象）、[[rollout train reference logprob一致性]]（三端 logp 与 $\rho$ 失真）、[[importance sampling与off-policy correction]]（PPO ratio/clip）、[[MoE路由]]（路由跳变是 $k$ 偏离的主因之一）
- **下游（应用）**: [[训练推理不一致TIM技术综述]]（IcePop 是引擎层 masking 代表）、MoE 大模型 RL 稳定性（Ring-1T）
- **对比 / 易混**:
  - **IcePop（mask 训推比 $k$）vs TIS（截 IS 权重上限）**：IcePop 双边、基于训推分歧；TIS 单边截上限、基于 IS 权重。详见 [[TIS]]、综述 §4。
  - **IcePop vs [[R3 rollout routing replay]]**：IcePop 丢弃分歧 token（算法层兜底）；R3 让训推走同一路由（工程根因对齐）。论文里 R3 定位为"治根因"，IcePop"治症状"。
  - **IcePop 的 $k$ vs PPO 的 $r$**：$k$ 衡量引擎分歧（同权重），$r$ 衡量权重更新（不同权重），正交。
  - **IcePop vs DeepSeek [[KSM与Keep-Routing]]**：KSM 复用截断 mask（采样动作空间一致）；IcePop mask 训推概率比 token（梯度可信度）。不同对象。

## 7. 常见误区与易错点

> [!warning] 误区 1：以为 IcePop 截的是 PPO ratio
> 它截的是**训推概率比 $k=\pi_{\text{train}}/\pi_{\text{infer}}$**（同 $\theta_{old}$），不是 PPO 的 $r=\pi_\theta/\pi_{\theta_{old}}$。两个 ratio 一个管引擎一致、一个管权重更新，别混。

> [!warning] 误区 2：以为双边区间越窄越稳
> 实测 $[0.5,2.0]$ 收窄会**立即失稳**——裁太多 token，有效梯度不足。默认 $[0.5,5]$ 是经验最优。

> [!warning] 误区 3：以为 IcePop 治了 TIM 根因
> 它只是把分歧大的 token 梯度丢掉，根因（引擎数值差/路由跳变）仍在。MoE 场景建议叠加 [[R3 rollout routing replay]] 治根因，见综述 §8 场景选型。

> [!warning] 误区 4：固定 $[\alpha,\beta]$ 假设误差均匀
> 这是 IcePop 的核心局限——不同 token/层的训推分歧分布不同，全局固定阈值不是最优。这正是后续 **KPop** 用自适应区域替代的动机（注：KPop 目前**未找到可靠 arXiv 来源**，见综述 §9）。

## 8. 延伸细节

### 8.1 与 Ring-1T 三件套的关系
IcePop 是 Ring-1T 稳定性三件套之一（另两个是 C3PO++ 算法改进与 ASystem 系统）。论文将其定位为"token-level discrepancy masking and clipping"。

### 8.2 IcePop 之后
IcePop 的固定阈值局限 motivates 自适应 masking（KPop，binary KL 替代固定 ratio，masking 比例从 0.2% 升到 10–30%）——但**该后续工作未在 arXiv 检索到可靠来源**（知乎科普文 403 不可读），综述 [[训练推理不一致TIM技术综述]] §9 标注「未找到可靠来源」。

### 8.3 内容来源
公式 $M(k;\alpha,\beta)$、目标 $J_{\text{IcePop}}$、Theorem 1、$\alpha=0.5/\beta=5.0$、裁剪 ~0.2%、Ring-mini-2.0 AIME +14%、Ring-1T 基准表均来自 arXiv:2510.18855 全文（v1 2025-10-21，v2 2025-10-25）已联网核实。截至 2026-09。

---
相关: [[训练推理不一致TIM技术综述]] | [[训推不一致]] | [[rollout train reference logprob一致性]] | [[importance sampling与off-policy correction]] | [[R3 rollout routing replay]] | [[TIS]] | [[KSM与Keep-Routing]] | [[MoE路由]] | [[17-RL训推一体框架]]
