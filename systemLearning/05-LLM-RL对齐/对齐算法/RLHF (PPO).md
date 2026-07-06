# RLHF (PPO)

> **所属章节**: [[对齐算法]]
> **所属模块**: [[05-LLM-RL对齐]]
> **难度**: 难（对齐算法的"显式 RL"基线，理解 DPO/IPO/KTO 的参照点）

## 1. 一句话定义

**RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）** 是 InstructGPT（OpenAI 2022）确立的 LLM 对齐方法：先 [[SFT]] 学指令服从 → 再 [[Reward Model]] 学人类偏好 → 最后用 [[PPO optimization]] 做 KL-constrained reward maximization，把"判别信号"转成"生成能力提升"。**PPO 是 RLHF 默认的 RL 引擎**，故"RLHF (PPO)"常被当作同义词，但严格说 RLHF 是**方法范式**（三阶段 + RM-driven RL），PPO 是其**求解器**之一（也可换 GRPO/RLAIF）。本篇是第 16 节"对齐算法"的总览入口，定位 RLHF (PPO) 在对齐谱系中的位置，并为 [[DPO]]/[[IPO]]/[[KTO]] 提供对比基线。

> [!note] 本篇与 [[PPO optimization]] 的分工
> - [[PPO optimization]]：**工程组装篇**——四模型协同、rollout→reward→GAE→update 的逐行流程、token 级 ratio、critic 训练、verl/TRL 落地。
> - 本篇：**方法总览篇**——RLHF 三阶段范式、为什么用 PPO、PPO 作为对齐算法一员的谱系定位、训练动态监控、与 DPO 家族的工程权衡、历史与现状。
> 算法内核与公式细节见 [[PPO optimization]] §4，本篇不重复。

## 2. 为什么需要它（动机与背景）

预训练 LLM 学会了"接龙"但不会"听指令、守价值观"：
1. **指令不对齐**：预训练目标是预测下一 token，没有"听从用户"信号 → 需 [[SFT]] 注入指令服从；
2. **偏好无法示范**：好/坏回答难用 SFT 示范（同一 prompt 多个可接受答案，且"坏"的样本难写）→ 需 [[Reward Model]] 从偏好对学"判别器"；
3. **判别器≠生成器**：RM 能给已有回答打分，但不能直接生成更好回答 → 需 **RL** 把判别信号转成生成能力提升（策略梯度用 reward 优化 $\pi_\theta$）；
4. **目标带 KL 约束**：$\max_\theta\mathbb{E}[r_{\text{RM}}]-\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$ 既有 reward 最大化又有 KL 约束保语言能力 → SFT 无此机制，需 RL 的 KL-constrained PO 框架；
5. **PPO 是工业验证过的稳定 PG**：[[REINFORCE]] 方差大、TRPO 慢，[[PPO clipped objective]] 用 clip 软约束 + 多 epoch 复用，在 LLM 生成昂贵的场景样本效率够用。

故 RLHF = "SFT + RM + PPO"三阶段，PPO 是把 RM 信号落地为 $\pi_\theta$ 提升的关键引擎。它是对齐算法的**显式 RL 基线**，后续 [[DPO]]/[[IPO]]/[[KTO]] 都是"绕过 PPO 的轻量化替代"，需先懂 RLHF (PPO) 才能理解它们为什么省、省在哪、代价是什么。

## 3. 核心概念详解

### 3.1 RLHF 三阶段 pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐
│  阶段1 SFT  │ -> │  阶段2 RM    │ -> │  阶段3 PPO          │
│  示范微调   │    │  偏好→打分器 │    │  KL-constrained RL  │
│  π_SFT      │    │  r_φ         │    │  π_θ ← 优化         │
└─────────────┘    └──────────────┘    └─────────────────────┘
   人类示范          人类偏好对           prompt + 在线 rollout
   (高质量回答)      (chosen/rejected)   (actor 生成,RM 打分)
   监督学习          判别学习             强化学习
```

- **阶段 1 [[SFT]]**：$\pi_{\text{pretrained}}$ + 高质量 (prompt, response) 示范 → $\pi_{\text{SFT}}$（学指令服从，当 reference $\pi_{\text{ref}}$ 与 actor 初始化）；
- **阶段 2 [[Reward Model]]**：人类标 (prompt, chosen, rejected) 偏好对 → 训 $r_\phi$（Bradley-Terry 损失，输出标量 reward）；
- **阶段 3 [[PPO optimization]]**：$\pi_{\text{SFT}}$ 当 actor 初始化 + $r_\phi$ 当 reward + $\pi_{\text{ref}}=\pi_{\text{SFT}}$ 做 KL 锚点 → PPO 优化得 $\pi_\theta$（对齐后的 LLM）。

### 3.2 PPO 在 RLHF 中的角色（为什么是 PPO）

RLHF 第三阶段的目标是 KL-constrained reward maximization：
$$
\max_\theta\ \mathbb{E}_{x,y\sim\pi_\theta}[r_\phi(x,y)]-\beta\,\mathbb{E}_{x,y\sim\pi_\theta}[\text{KL}(\pi_\theta(\cdot|x)\|\pi_{\text{ref}}(\cdot|x))]
$$

这是**约束策略优化（constrained PO）**问题。可选求解器：
| 求解器 | 思路 | 在 RLHF 的命运 |
|---|---|---|
| [[REINFORCE]] | 蒙特卡洛 PG | 方差太大，不稳 |
| TRPO | KL 硬约束 + 二阶解 | 慢，LLM 规模不友好 |
| **[[PPO clipped objective]]** | clip 软约束 + 多 epoch | **InstructGPT 选定，事实标准** |
| GRPO | PPO 去 critic，组内相对 | 数学推理场景崛起（[[GRPO]] 待展开） |

PPO 胜出原因：clip 提供软稳定性、多 epoch 复用样本（LLM 生成贵，复用提效率）、工业验证（游戏 RL 已成熟）。所以"RLHF (PPO)"成了 RLHF 的默认实现，但**严格说 PPO 只是 RLHF 的 RL 引擎**，换 GRPO/RLAIF 仍是 RLHF 范式。

### 3.3 对齐算法谱系（第 16 节全景）

| 算法 | 类型 | 是否 rollout | 是否需 RM | 是否需 critic | 核心思路 |
|---|---|---|---|---|---|
| **RLHF (PPO)** | 显式 RL | ✅ | ✅ | ✅ | PPO 解 KL-constrained reward max |
| [[DPO]] | 隐式（闭式） | ❌ | ❌ | ❌ | 目标闭式解 → 直接 BT 损失训 $\pi_\theta$ |
| [[IPO]] | 隐式（闭式） | ❌ | ❌ | ❌ | DPO 的非概率过拟合版（Kelley-Mesterton-leaping） |
| [[KTO]] | 隐式（闭式） | ❌ | ❌ | ❌ | 只需"好/坏"二元标签（非成对偏好） |

RLHF (PPO) 是**最重但最灵活**的：能在线探索、RM 可持续标新数据。DPO 家族是"离线、轻、快"的替代，用闭式解绕过 rollout/RM/critic。详见各专篇。

### 3.4 训练动态与监控信号

PPO 阶段需盯的核心曲线（详见 [[PPO optimization]] §8.7）：
- **reward**：应缓慢上升（若猛涨且人类评估降 → [[reward hacking]]）；
- **KL（to ref）**：维持 target 附近（adaptive β，若爆 → [[KL explosion]]）；
- **clip_fraction**：0.1~0.3 健康；
- **response length**：稳定（若猛涨 → length bias / [[reward hacking]]）；
- **entropy**：缓降但不到 0（若到 0 → [[policy collapse]]/[[mode collapse]]）；
- **critic explained variance**：>0.5 算可用（critic 不准则 advantage 噪声大）。

这些曲线是 RLHF 调参的诊断信号，是 PPO 阶段成败的"仪表盘"。

### 3.5 数据与超参（InstructGPT 标定）

| 阶段 | 数据量（InstructGPT） | 关键超参 |
|---|---|---|
| SFT | 13k 示范 | lr 1e-6~5e-6, 2~3 epoch |
| RM | 33k 偏好对 | BT 损失, lr 1e-6, RM 常从 SFT 初始化 |
| PPO | 31k prompt | β=0.01（早期 fixed，后 adaptive）, K=2, ε=0.2, lr 1e-6~5e-6 |

现代开源实践（Llama-2/3-chat）：SFT 用更多指令数据，RM 用更大偏好集（10万~百万对），PPO 用 adaptive β + target_kl 早停 + verl/TRL 工程。

## 4. 数学原理 / 公式

### 4.1 RLHF 目标（KL-constrained reward maximization）

$$
\max_\theta\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\bigl[r_\phi(x,y)\bigr]\quad\text{s.t.}\quad\mathbb{E}_{x,y\sim\pi_\theta}\bigl[\text{KL}(\pi_\theta(\cdot|x)\big\|\pi_{\text{ref}}(\cdot|x))\bigr]\le\delta
$$

拉格朗日松弛 → 无约束 $\max_\theta\mathbb{E}[r]-\beta\,\text{KL}$，β 是约束价格（[[KL penalty与KL control]] §4.1）。**PPO 是此目标的 RL 求解器**：用 rollout 估 $\mathbb{E}_{y\sim\pi_\theta}$、用 [[GAE]] 算 advantage、用 clip + KL penalty + target_kl 三重约束近似求解。

### 4.2 PPO 损失（actor 部分，token 级）

$$
L^{CLIP}(\theta)=-\mathbb{E}_t\bigl[\min(\rho_t\hat A_t,\,\text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)\bigr]
$$

- $\rho_t=\pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$，token 级 ratio；
- $\hat A_t$：GAE 倒推的 advantage（detach）；
- 加 critic loss + entropy bonus 成总 loss（详见 [[PPO optimization]] §4.2、[[PPO clipped objective]] §3.6）。

### 4.3 闭式最优解（DPO 的出发点，关键对照）

§4.1 目标有**闭式最优解**（这是 [[DPO]] 的推导起点）：
$$
\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\!\Bigl(\frac{r_\phi(x,y)}{\beta}\Bigr)
$$
反解得 $r_\phi(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$。代入 Bradley-Terry 偏好损失 → DPO 损失（无需 rollout/RM/critic）。

> [!tip] PPO vs DPO 的本质对照
> - **PPO**：显式跑 RL 求 $\pi^*$（rollout + RM + critic + clip）——重但灵活，能在线探索；
> - **DPO**：用闭式解直接训 $\pi^*$（只需偏好对 + 两个 logp）——轻但离线。
> 两者优化的是**同一个目标** §4.1，只是路径不同。理解此闭式解是理解 DPO 家族（[[IPO]]/[[KTO]]）的前提。

## 5. 代码示例

极简三阶段调度（每阶段细节见各自专篇）：

```python
import torch, torch.nn as nn, torch.nn.functional as F, copy

class LM(nn.Module):
    """极简 LLM:emb + lstm + head"""
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)
    def logp(s, x, a):
        return F.log_softmax(s(x),-1).gather(-1,a.unsqueeze(-1)).squeeze(-1)

V, D = 50, 32
pi = LM(V, D)                                  # 预训练模型(简化)

# ========== 阶段1: SFT(示范微调) ==========
sft_data = [(torch.randint(0,V,(1,8)), torch.randint(0,V,(1,8))) for _ in range(20)]  # (prompt,resp)
opt_sft = torch.optim.Adam(pi.parameters(), 1e-5)
for ep in range(3):
    for x, y in sft_data:
        loss = -pi.logp(y, y).mean()           # teacher forcing 负对数似然
        opt_sft.zero_grad(); loss.backward(); opt_sft.step()
pi_sft = copy.deepcopy(pi)                      # π_SFT = reference = actor 初始化

# ========== 阶段2: Reward Model(偏好对) ==========
class RM(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.s=nn.Linear(d,1)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.s(h[:,-1]).squeeze(-1)
rm = RM(V, D)
pref_data = [((torch.randint(0,V,(1,8)), torch.randint(0,V,(1,8))),   # (prompt, chosen, rejected)
             torch.tensor(1), torch.tensor(0)) for _ in range(20)]
opt_rm = torch.optim.Adam(rm.parameters(), 1e-5)
for ep in range(3):
    for (x, yc), _, _ in pref_data:            # 简化:用 yc/yr 算 BT 损失
        yr = torch.randint(0, V, (1, 8))
        rc, rr = rm(torch.cat([x, yc], 1)), rm(torch.cat([x, yr], 1))
        loss = -F.logsigmoid(rc - rr)          # Bradley-Terry
        opt_rm.zero_grad(); loss.backward(); opt_rm.step()
for p in rm.parameters(): p.requires_grad_(False)

# ========== 阶段3: PPO(详见 PPO optimization.md) ==========
# 这里只标框架,完整实现见 [[PPO optimization]] §5
actor = copy.deepcopy(pi_sft)                  # actor 初始化 = π_SFT
ref = copy.deepcopy(pi_sft)                     # reference 冻结
for p in ref.parameters(): p.requires_grad_(False)
opt_ppo = torch.optim.Adam(actor.parameters(), 1e-6)
# PPO loop: rollout(actor 生成) -> reward(rm - β·KL) -> advantage(GAE) -> update(clip+vf)
# ... 见 PPO optimization.md §5 ...
print("RLHF 三阶段调度完成: π_pretrained -> π_SFT -> (RM) -> π_θ(对齐后)")
```

> [!note] 完整 PPO 实现见 [[PPO optimization]] §5，本篇只示三阶段调度骨架。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[SFT]]（阶段1）、[[Reward Model]]（阶段2）、[[PPO optimization]]（阶段3 工程实现）、[[PPO clipped objective]]（PPO 内核）、[[GAE]]、[[KL penalty与KL control]]、[[entropy bonus]]、[[MDP]]（LLM-RL 的 MDP 映射）。
- **下游（应用）**: 对齐后 LLM（InstructGPT/Llama-chat）、[[DPO]]/[[IPO]]/[[KTO]]（绕过 PPO 的替代）、[[reward hacking]]/[[KL explosion]]/[[policy collapse]]/[[mode collapse]]（PPO 失败模式）、[[RLAIF]]（用 AI 替人类标偏好）、[[GRPO]]（PPO 变体）。
- **对比 / 易混**:
  - **RLHF vs PPO**：RLHF 是**方法范式**（三阶段 + RM-driven RL），PPO 是其**RL 引擎**之一。说"RLHF (PPO)"是强调用 PPO 实现，换 GRPO/RLAIF 仍是 RLHF。
  - **RLHF (PPO) vs [[DPO]]**：PPO 显式 rollout+RM+critic（重工程），DPO 闭式无 rollout（轻）。等价目标不同路径，详见 [[DPO]] §4。
  - **RLHF (PPO) vs [[PPO optimization]]**：本篇是方法总览（三阶段/谱系/动态），[[PPO optimization]] 是工程组装（四模型/rollout/GAE 逐行）。
  - **RLHF vs SFT**：SFT 用示范教服从（监督），RLHF 用 reward 教偏好（强化）；SFT 是 RLHF 阶段1。
  - **RLHF (PPO) vs [[RLAIF]]**：人类标偏好 vs AI 标偏好，RM 来源不同，PPO 引擎可共用。
  - **RLHF (PPO) vs [[GRPO]]**：PPO 有 critic，GRPO 去 critic 用组内相对 advantage，省显存。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"RLHF = PPO"**：错。RLHF 是三阶段方法范式，PPO 只是阶段3的 RL 引擎之一（可换 GRPO/RLAIF）。
> 2. **"RLHF = SFT + RM"**：错。还差阶段3 RL，否则 RM 信号无法落地为生成能力提升。
> 3. **"PPO 阶段不需要 SFT"**：错。$\pi_{\text{ref}}$ 与 actor 初始化都来自 SFT，没 SFT 锚点 PPO 会把模型训崩（[[KL explosion]]）。
> 4. **"reward 涨就是对齐成功"**：错。reward 涨但人类评估降是 [[reward hacking]]，需看 KL/length/entropy/人类评估。
> 5. **"PPO 可离线"**：错。PPO 需 actor 在线 rollout 生成（autoregressive），不能用固定数据集（那是 DPO）。训推分离只是工程加速，rollout 仍是在线的。
> 6. **"KL 约束只是正则"**：错。KL 是 §4.1 目标的**硬约束**（保语言能力），β 是约束价格，adaptive β 是动态调整约束强度。
> 7. **"DPO 一定比 PPO 好"**：不一定。PPO 能在线探索、RM 持续标新数据；DPO 离线固定偏好数据，灵活性差。现代实践常 PPO+DPO 混用。
> 8. **"InstructGPT 用 adaptive β"**：早期 InstructGPT 用 fixed β=0.01，adaptive 是后续改进。
> 9. **"PPO 阶段 RM 也更新"**：错。RM 全程冻结，否则 reward 信号漂移、训练不稳。
> 10. **"四个模型一样大"**：不一定。critic 常更大（提 advantage 准度），RM 可小，actor/ref 与 SFT 同规模。

## 8. 延伸细节

### 8.1 InstructGPT 的历史定位

InstructGPT（Ouyang et al. 2022）确立 SFT→RM→PPO 三阶段，是现代 RLHF 的标准范式。GPT-3.5/4 的对齐即基于此。此后 Llama-2/3-chat、Claude、Gemini 的对齐均沿用此框架（细节有改进：RLAIF/Constitutional AI/更大数据/GRPO 等）。PPO 阶段是事实标准，但 DPO（2023）后轻量化对齐崛起，现代实践常 DPO 打底 + PPO 在线精修。

### 8.2 RLHF (PPO) 的工程权衡（vs DPO 家族）

| 维度 | RLHF (PPO) | [[DPO]]/[[IPO]]/[[KTO]] |
|---|---|---|
| 工程 | 4 模型 + rollout + critic，重 | 1 模型 + 偏好数据，轻 |
| 算力 | 高（生成贵，4 模型显存 ~4× SFT） | 低（无生成，1 模型） |
| 灵活性 | 在线 RM 可持续标新数据 | 离线，固定偏好数据 |
| 探索 | 能学偏好数据外行为 | 不能 |
| 稳定性 | 易 [[reward hacking]]/[[KL explosion]] | 较稳但易过拟合偏好 |
| 适用 | 在线对齐、持续优化、高要求场景 | 离线对齐、快速出模型、资源紧 |

现代实践：DPO 快速打底，PPO 在线精修；或 GRPO 在数学推理场景崛起（DeepSeekMath/R1）。PPO 不再是唯一选择，但仍是"显式 RL 对齐"的基线与参照。

### 8.3 PPO 的失败模式（第 17 节预告）

PPO 阶段最易出现的问题（详见各专篇）：
- [[reward hacking]]：钻 RM 漏洞拿高分但实际差；
- [[KL explosion]]：$\pi_\theta$ 漂离 ref，语言能力崩；
- [[policy collapse]]：actor 塌缩到少数 mode；
- [[mode collapse]]：多样性丧失；
- [[exposure bias]]：rollout 与 SFT 分布偏移。
这些是 PPO 调参的核心诊断对象，第 17 节专门展开。

### 8.4 RLHF 的变体与演进

- [[RLAIF]]：用 AI（如更强 LLM）替人类标偏好，降标注成本（Constitutional AI）；
- [[GRPO]]：PPO 去 critic，组内相对 advantage，省显存，数学推理效果好（DeepSeek）；
- 多轮 RLHF：SFT→RM→PPO→再用 PPO 产出数据回训 RM→再 PPO（迭代对齐）；
- 过程监督（PRM）：不只看最终答案 reward，还标中间步骤（OpenAI o1 思路）。
这些是 RLHF (PPO) 范式的演进方向。

### 8.5 PPO 的开源实现现状

| 框架 | 特点 | 适用 |
|---|---|---|
| **verl** | 字节/开源，Ray 调度，训推分离，大模型友好 | 生产、大模型 |
| **TRL** | HuggingFace，集成 transformers，社区主流 | 研究、中等规模 |
| **DeepSpeed-Chat** | 微软，DeepSpeed 加速，3 阶段全流程 | DeepSpeed 用户 |
| **OpenRLHF** | 注重可读性与大模型支持 | 学习、大模型 |

PPO 阶段的工程门槛高（四模型协同 + rollout + GAE），是 RLHF 落地的核心瓶颈，verl 是当前主流选择。

---
相关: [[对齐算法]]、[[SFT]]、[[Reward Model]]、[[PPO optimization]]、[[PPO clipped objective]]、[[GAE]]、[[KL penalty与KL control]]、[[entropy bonus]]、[[DPO]]、[[IPO]]、[[KTO]]、[[RLAIF]]、[[GRPO]]、[[reward hacking]]、[[KL explosion]]、[[policy collapse]]、[[mode collapse]]、[[exposure bias]]、[[MDP]]
