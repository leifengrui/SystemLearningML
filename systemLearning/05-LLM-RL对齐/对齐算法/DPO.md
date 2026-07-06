# DPO

> **所属章节**: [[对齐算法]]
> **所属模块**: [[05-LLM-RL对齐]]
> **难度**: 难（推导是核心，需懂 RLHF 目标 + Bradley-Terry + KL-constrained PO 闭式解）

## 1. 一句话定义

**DPO（Direct Preference Optimization，直接偏好优化）** 是 Rafailov et al. 2023 提出的 RLHF 替代方案：**绕过显式 [[Reward Model]] 与 [[PPO optimization]] rollout**，直接用偏好对 $(x, y_w, y_l)$ 通过一个**闭式损失**微调 $\pi_\theta$。核心洞察是——RLHF 目标 $\max_\theta\mathbb{E}[r]-\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$ 有**闭式最优解** $\pi^*=\pi_{\text{ref}}\exp(r/\beta)/Z$，反解出 $r=\beta\log(\pi^*/\pi_{\text{ref}})+\beta\log Z$，代入 Bradley-Terry 偏好损失即得 DPO 损失。结果：**无需 RM、无需 critic、无需 rollout**，只需偏好对 + 两个模型（$\pi_\theta$ 与冻结 $\pi_{\text{ref}}$），一个 loss 走监督学习式训练。它和 [[RLHF (PPO)]] 优化的是**同一个 KL-constrained 目标**，只是用闭式解代替显式 RL。

> [!note] DPO = 闭式 RLHF
> 一句话记忆：**"RLHF 的目标有闭式解，把解代回 BT 损失就免了 PPO"**。DPO 不是新目标，是 RLHF 目标的**直接求解器**。理解 §4 的推导是理解 DPO 的全部。

## 2. 为什么需要它（动机与背景）

[[RLHF (PPO)]] 三阶段（SFT→RM→PPO）能对齐，但 PPO 阶段工程极重：
1. **四模型协同**：actor + critic + reference + RM 同显存，约 SFT 的 4 倍显存（[[PPO optimization]] §8.2）；
2. **rollout 昂贵**：actor 需在线自回归生成，是训练瓶颈（[[训推分离]]）；
3. **多环节误差累积**：RM 学不准 → PPO 优化错误 reward → [[reward hacking]]；
4. **超参敏感**：β/K/ε/lr 调不好就 [[KL explosion]]/[[policy collapse]]；
5. **数据利用率低**：PPO 在线采样，每个偏好对只用一次（隐式）。

DPO 的动机：**既然 RLHF 目标有闭式解，何必跑 PPO？** 用闭式解把 reward 用 $\pi_\theta$ 表出，直接代入偏好损失，就把"RL 问题"变成"监督学习问题"——一个 loss、两个模型、无 rollout。代价：失去在线探索能力、依赖离线偏好数据质量。但工程门槛大降，使对齐民主化（小团队也能做），故 2023 后成主流。

## 3. 核心概念详解

### 3.1 DPO 的核心洞察（三步法）

```
RLHF 目标 max E[r]-βKL  ──(1) 闭式最优解──>  π* = π_ref·exp(r/β)/Z
                                            │
                                            │(2) 反解 r
                                            ▼
                       r = β·log(π*/π_ref) + β·log Z
                                            │
                                            │(3) 代入 BT 偏好损失
                                            ▼
              L_DPO = -log σ( β·log(π_θ(y_w)/π_ref(y_w)) - β·log(π_θ(y_l)/π_ref(y_l)) )
```

- **(1)**：RLHF 目标是 KL-constrained reward max，其最优策略有闭式形式（用变分法/拉格朗日）；
- **(2)**：反解出 reward 用策略表出（$Z$ 是配分函数，与 $y$ 无关）；
- **(3)**：把 reward 表达代入 Bradley-Terry 偏好损失 $P(y_w\succ y_l)=\sigma(r_w-r_l)$，$Z(x)$ 在相减中消掉，得 DPO 损失。

结果：reward 被策略表出，无需显式 RM；最优策略就是 $\pi_\theta$，无需 PPO rollout。

### 3.2 DPO 损失的直观读法

$$
\mathcal{L}_{\text{DPO}}(\theta)=-\log\sigma\!\Bigl(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Bigr)
$$

读法：让 $\pi_\theta$ 相对 $\pi_{\text{ref}}$ **提升 chosen $y_w$ 的概率、压低 rejected $y_l$ 的概率**，比值差越大 loss 越小。
- $\beta$：温度，控制偏离 $\pi_{\text{ref}}$ 的强度（类比 [[KL penalty与KL control]] 的 β）；
- $\pi_{\text{ref}}$：冻结的 SFT 模型，KL 锚点；
- 无 RM、无 critic、无 rollout：只需偏好对 + 两次前向（$\pi_\theta$ 与 $\pi_{\text{ref}}$ 各算 $y_w, y_l$ 的 logp）。

### 3.3 DPO 的梯度（隐式 reward 学习）

对 $\pi_\theta(y_w)$ 求梯度（简化）：
$$
\nabla_\theta\mathcal{L}_{\text{DPO}}\propto\bigl[\underbrace{\beta\nabla_\theta\log\pi_\theta(y_l|x)}_{\text{压低 rejected}}-\underbrace{\beta\nabla_\theta\log\pi_\theta(y_w|x)}_{\text{提升 chosen}}\bigr]\cdot\bigl[\text{当前 margin 的 sigmoid 残差}\bigr]
$$

- **奖励 chosen**：提升 $\log\pi_\theta(y_w)$；
- **惩罚 rejected**：压低 $\log\pi_\theta(y_l)$；
- **残差加权**：当 $\pi_\theta$ 已拉开足够 margin 时梯度衰减（已学好），自适应聚焦难样本。

这等价于一个**隐式 reward** $\hat r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 在做 BT 损失——DPO 把"训 RM"和"训 $\pi_\theta$"合并成一步。

### 3.4 DPO 的训练流程

```
输入:偏好对数据 {(x, y_w, y_l)}, SFT 模型 π_ref, β
1. π_θ ← π_ref(初始化)
2. for each batch (x, y_w, y_l):
     a. 算 logp_θ(y_w|x), logp_θ(y_l|x)        # π_θ 前向
     b. 算 logp_ref(y_w|x), logp_ref(y_l|x)    # π_ref 前向(冻结,no_grad)
     c. margin = β·(logp_θ(y_w)-logp_ref(y_w)) - β·(logp_θ(y_l)-logp_ref(y_l))
     d. loss = -log σ(margin)
     e. backward, step
3. 输出 π_θ(对齐后)
```

相比 PPO 的四模型 + rollout + GAE + 多 epoch，DPO 就是一个标准监督学习 loop。

### 3.5 DPO vs PPO 的关键差异

| 维度 | [[RLHF (PPO)]] | DPO |
|---|---|---|
| 目标 | $\max\mathbb{E}[r]-\beta\,\text{KL}$（显式 RL） | 同一目标（闭式求解） |
| 模型数 | 4（actor+critic+ref+RM） | 2（$\pi_\theta$ + ref） |
| rollout | ✅ 在线生成 | ❌ 无 |
| RM | ✅ 需训 | ❌ 隐式（用 $\pi_\theta$ 表出） |
| 数据 | prompt + 在线采样 | 偏好对 $(x,y_w,y_l)$ |
| 算力 | 高（4× SFT 显存 + 生成） | 低（~SFT） |
| 探索 | ✅ 能学偏好外行为 | ❌ 离线固定 |
| 稳定性 | 易 reward hacking/KL explosion | 较稳，但易过拟合偏好 |
| 超参 | β/K/ε/lr 多 | β/lr 少 |

## 4. 数学原理 / 公式

### 4.1 RLHF 目标与闭式最优解

RLHF 目标（KL-constrained reward maximization）：
$$
\max_\theta\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\bigl[r(x,y)\bigr]-\beta\,\mathbb{E}_{x,y\sim\pi_\theta}\bigl[\text{KL}(\pi_\theta(\cdot|x)\big\|\pi_{\text{ref}}(\cdot|x))\bigr] \tag{1}
$$

**变分法求最优策略**：对单个 $x$，目标等价于最大化
$$
J(\pi_\theta|x)=\mathbb{E}_{y\sim\pi_\theta}[r(x,y)]-\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})
$$

展开 KL：$\text{KL}=\sum_y\pi_\theta(y|x)\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，代入并用变分（或拉格朗日乘子法约束 $\sum_y\pi_\theta=1$），得最优解：
$$
\boxed{\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\!\Bigl(\frac{r(x,y)}{\beta}\Bigr)} \tag{2}
$$

其中 $Z(x)=\sum_y\pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$ 是配分函数（归一化常数，与 $y$ 无关）。

> [!note] 直觉
> 最优策略 = reference 按 reward 的玻尔兹曼权重重整：reward 高的 $y$ 被放大，reward 低的被压制，β 是温度（小 β → 偏移大）。

### 4.2 反解 reward（用策略表出）

由 (2) 反解 $r$：
$$
\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}=\frac{1}{Z(x)}\exp\!\Bigl(\frac{r(x,y)}{\beta}\Bigr)
$$

取对数：
$$
\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}=\frac{r(x,y)}{\beta}-\log Z(x)
$$

解出 $r$：
$$
\boxed{r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)} \tag{3}
$$

> [!tip] 关键
> reward 现在被**最优策略** $\pi^*$ 表出。若把 $\pi^*$ 当成正在训的 $\pi_\theta$，则 reward = $\beta\log(\pi_\theta/\pi_{\text{ref}})$ + 常数。**RM 不再需要显式训**——只要训 $\pi_\theta$，reward 自动跟着出来。

### 4.3 代入 Bradley-Terry 偏好损失

[[Reward Model]] 用 Bradley-Terry 模型建模人类偏好：
$$
P(y_w\succ y_l|x)=\sigma\bigl(r(x,y_w)-r(x,y_l)\bigr) \tag{4}
$$

把 (3) 代入：
$$
r(x,y_w)-r(x,y_l)=\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}+\beta\log Z(x)-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}-\beta\log Z(x)
$$

**$Z(x)$ 相减消掉**（关键！这就是 DPO 不需 $Z$ 的原因）：
$$
r(x,y_w)-r(x,y_l)=\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \tag{5}
$$

### 4.4 DPO 损失

把 (5) 代入 BT 的负对数似然，得 DPO 损失：
$$
\boxed{\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_w,y_l)}\Bigl[\log\sigma\!\Bigl(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Bigr)\Bigr]} \tag{6}
$$

- 只含 $\pi_\theta$ 与冻结 $\pi_{\text{ref}}$ 的 logp；
- 无 $r$、无 $Z$、无 RM、无 rollout；
- 训练就是标准监督学习：最小化 (6)。

### 4.5 梯度推导（隐式 reward 的力学）

对 $\theta$ 求梯度（记 $u=\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}$，$v=\beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}$，$m=u-v$）：
$$
\nabla_\theta\mathcal{L}_{\text{DPO}}=-\nabla_\theta\log\sigma(m)=-(1-\sigma(m))\nabla_\theta m
$$

其中 $1-\sigma(m)=\sigma(-m)$ 是"当前 margin 不足"的残差，$\nabla_\theta m=\beta\nabla_\theta\log\pi_\theta(y_w)-\beta\nabla_\theta\log\pi_\theta(y_l)$。故：
$$
\nabla_\theta\mathcal{L}_{\text{DPO}}=\beta\,\sigma(-m)\bigl[\nabla_\theta\log\pi_\theta(y_l)-\nabla_\theta\log\pi_\theta(y_w)\bigr] \tag{7}
$$

读法：
- $\nabla_\theta\log\pi_\theta(y_w)$ 项系数为负 → **提升 chosen 概率**；
- $\nabla_\theta\log\pi_\theta(y_l)$ 项系数为正 → **压低 rejected 概率**；
- $\sigma(-m)$：margin 已大时梯度衰减（已学好），margin 不足时梯度大（聚焦难样本）。

这就是 DPO 的"隐式 reward 学习"——梯度自动实现"奖励 chosen、惩罚 rejected"。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F, copy

class LM(nn.Module):
    """极简 LLM:emb + lstm + head"""
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)
    def logp(s, x, y):
        """算 log π(y|x) = Σ_t log π(y_t | x + y_{<t})(teacher forcing)"""
        ctx = torch.cat([x, y[:, :-1]], 1)            # 输入:prompt + y 前缀
        logits = s(ctx)                                # (B, T, V)
        logp = F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, T_y)
        return logp.sum(-1)                            # Σ_t -> (B,) 序列级 logp

V, D = 50, 32
pi_sft = LM(V, D)   # 假设已完成 SFT

# DPO 初始化
pi_theta = copy.deepcopy(pi_sft)                      # π_θ ← π_ref
pi_ref = copy.deepcopy(pi_sft)                        # reference 冻结
for p in pi_ref.parameters(): p.requires_grad_(False)
opt = torch.optim.Adam(pi_theta.parameters(), 1e-6)
beta = 0.1

# 偏好对数据(prompt, chosen, rejected)
pref_data = [(torch.randint(0,V,(4,4)), torch.randint(0,V,(4,8)), torch.randint(0,V,(4,8))) for _ in range(20)]

for ep in range(3):
    for x, yw, yl in pref_data:
        # 两次前向:π_θ 与 π_ref 各算 yw, yl 的序列 logp
        logp_w = pi_theta.logp(x, yw)                 # (B,)
        logp_l = pi_theta.logp(x, yl)
        with torch.no_grad():
            ref_w = pi_ref.logp(x, yw)
            ref_l = pi_ref.logp(x, yl)
        # DPO 损失 (6)
        margin = beta * (logp_w - ref_w) - beta * (logp_l - ref_l)
        loss = -F.logsigmoid(margin).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        # 监控
        with torch.no_grad():
            acc = (margin > 0).float().mean()         # 偏好准确率
            kl = (logp_w - ref_w).abs().mean()        # 偏离 ref 的程度(近似 KL)
    print(f"ep{ep} loss={loss:.3f} acc={acc:.2f} drift={kl:.2f}")

print("DPO 训练完成:π_SFT -> π_θ(对齐后),无 RM 无 rollout")
```

> [!tip] 实际工程（TRL）
> TRL 的 `DPOTrainer` 封装上述 loop，支持：
> - 大模型用 `AutoModelForCausalLM`，序列级 logp 用 teacher forcing 累加；
> - $\pi_{\text{ref}}$ 用 `no_grad` + 半精度省显存；
> - `beta`、`loss_type`（default/ipo/hinge/robust）可配；
> - 支持 LoRA（只训 adapter，$\pi_\theta$ 与 $\pi_{\text{ref}}$ 共享 base）。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[RLHF (PPO)]]（DPO 是其闭式替代，同目标）、[[SFT]]（$\pi_{\text{ref}}$ 与 $\pi_\theta$ 初始化）、[[Reward Model]]（BT 损失来源，DPO 隐式化了它）、[[KL penalty与KL control]]（β 的角色）、[[Bradley-Terry模型]]（偏好建模，待展开）。
- **下游（应用）**: 对齐后 LLM（Zephyr/Llama-chat 的 DPO 版）、[[IPO]]（DPO 的过拟合修正版）、[[KTO]]（非成对版）、DPO 衍生（SimPO、rDPO、Iterative DPO）。
- **对比 / 易混**:
  - **DPO vs [[RLHF (PPO)]]**：同目标（§4.1），DPO 闭式无 rollout（轻），PPO 显式 RL（重但灵活）。详见 §3.5 表。
  - **DPO vs [[IPO]]**：DPO 用 BT 损失（概率模型），IPO 用 Kelley-Mesterton 损失防概率过拟合，损失形式略变。
  - **DPO vs [[KTO]]**：DPO 需成对 $(y_w, y_l)$，KTO 只需二元标签（好/坏，非成对），数据门槛更低。
  - **DPO vs SFT**：SFT 只学 chosen（teacher forcing），DPO 同时学 chosen 提升 + rejected 压低，且用 $\pi_{\text{ref}}$ 做 KL 锚点。
  - **DPO 的 reward vs RM 的 reward**：DPO 的隐式 reward $\hat r=\beta\log(\pi_\theta/\pi_{\text{ref}})$ 是策略表出，RM 是显式学；两者在 §4.1 目标下等价，但 DPO 一步到位。
  - **β vs [[KL penalty与KL control]] 的 β**：DPO 的 β 是偏离 $\pi_{\text{ref}}$ 的温度，类比 PPO 的 KL β，但 DPO 是固定值（无 adaptive，因无在线 KL 监控）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"DPO 是新目标"**：错。DPO 与 RLHF (PPO) 是**同一目标** §4.1，DPO 是闭式求解，PPO 是显式 RL 求解。
> 2. **"$Z(x)$ 需要算"**：错。$Z(x)$ 在 BT 损失的 $r_w - r_l$ 相减中消掉（§4.3），DPO 损失不含 $Z$。这是 DPO 可行的关键。
> 3. **"DPO 不需要 reference"**：错。$\pi_{\text{ref}}$ 是 KL 锚点，必须冻结前向算 logp；没有 ref，DPO 退化为只提升 chosen 压低 rejected 的无约束优化，会崩（漂移过大）。
> 4. **"DPO 不需要 SFT"**：错。$\pi_{\text{ref}}$ 与 $\pi_\theta$ 初始化都来自 SFT；没 SFT 锚点，DPO 训不出对齐模型（语言能力不足）。
> 5. **"β 越大越好"**：错。β 大 → 偏离 ref 强 → 易过拟合偏好数据 + 漂移；β 小 → 偏移弱 → 学不动。典型 0.1~0.5。
> 6. **"DPO 比 PPO 一定更好"**：错。DPO 离线、不能探索、依赖偏好数据质量；PPO 能在线探索、RM 持续标新。场景不同。
> 7. **"DPO 不会 reward hacking"**：不完全。DPO 无显式 RM 故无 RM 漏洞，但仍可过拟合偏好数据的偏差（"偏好 hacking"），需看人类评估。
> 8. **"DPO 的 logp 是 token 级"**：部分错。DPO 损失用**序列级** logp（Σ_t log π(y_t)），但计算时逐 token teacher forcing 累加；与 PPO 的 token 级 ratio 不同。
> 9. **"DPO 损失 = SFT 损失"**：错。SFT 是 $-\log\pi_\theta(y_w)$（只学 chosen），DPO 是 $-\log\sigma(\beta\log(\pi_\theta(y_w)/\pi_{\text{ref}}(y_w))-\cdots)$（同时学 chosen/rejected + KL 锚点）。
> 10. **"DPO 可在线探索"**：错。DPO 用固定偏好对，无 rollout，不能学偏好数据外行为。这是与 PPO 的根本差异。
> 11. **"DPO 不需调参"**：错。β、lr、偏好数据质量、ref 选择都敏感；β 过大或 lr 过大仍会漂移崩坏。
> 12. **"DPO 隐式 reward 等于显式 RM"**：理论上等价（同目标），但实践上 DPO 的隐式 reward 受 $\pi_\theta$ 容量限制，RM 独立训可能更准；两者各有优劣。

## 8. 延伸细节

### 8.1 DPO 的假设与局限

DPO 成立的关键假设：
1. **BT 模型成立**：人类偏好可由 $\sigma(r_w-r_l)$ 建模。若偏好非传递性/有噪声，BT 失准；
2. **偏好对质量**：DPO 对偏好数据敏感，标注噪声/偏差会直接学进 $\pi_\theta$；
3. **离线固定**：不能在线探索，学不到偏好数据外的行为；
4. **$\pi_{\text{ref}}$ 质量**：ref 是 SFT，若 SFT 差则 DPO 起点差；
5. **β 固定**：无在线 KL 监控，β 不能 adaptive（PPO 可 adaptive β）。
这些是 DPO 的代价——换来工程简单。

### 8.2 DPO 的衍生

- **[[IPO]]**：用 Kelley-Mesterton-leaping 损失替换 BT，防概率过拟合（DPO 易把 margin 推过大）；
- **[[KTO]]**：只需二元好/坏标签（非成对），数据门槛低；
- **SimPO**：去 reference，用序列长度归一化的 margin，更省；
- **rDPO**：robust DPO，对偏好噪声鲁棒；
- **Iterative DPO**：多轮 DPO，用上一轮 $\pi_\theta$ 当新 ref，迭代对齐；
- **Online DPO**：结合在线采样 + DPO 损失，折中 PPO 与 DPO。
这些是 DPO 的改进方向。

### 8.3 DPO 与 PPO 的实践选择

| 场景 | 推荐 |
|---|---|
| 资源紧、快速出对齐模型 | DPO |
| 离线偏好数据充足且干净 | DPO |
| 在线持续对齐、RM 可不断标新 | PPO |
| 高要求、需探索偏好外行为 | PPO |
| 数学推理/有 verifier 的场景 | GRPO（PPO 变体） |
| 小团队/研究 | DPO（TRL 一行起） |

现代实践：DPO 打底快速对齐，PPO 在线精修；或 Iterative DPO 多轮迭代逼近 PPO 效果。

### 8.4 DPO 与 Bradley-Terry 的深度关联

DPO 的推导高度依赖 BT 模型（§4.3）。BT 假设偏好是传递的、可由 reward 差的 sigmoid 建模。若偏好数据有：
- 非传递性（A>B, B>C, C>A）：BT 失准；
- 噪声大：BT 拟合差；
- 标注者分歧大：BT 用单一 reward 难表达。
这些场景需更复杂的偏好模型（Thurstone/Plackett-Luce），DPO 也可推广（如 Plackett-Luce DPO 排序多候选）。BT 是最简且实用的选择。

### 8.5 DPO 的"重参数化"视角

DPO 可视为**重参数化技巧**：把 reward $r$ 用策略 $\pi_\theta$ 重参数化（$r=\beta\log(\pi_\theta/\pi_{\text{ref}})$ + const），然后直接在重参数化空间做 BT 损失。这绕过了"先训 RM 再训 $\pi_\theta$"的两步，合并为一步。此视角解释了 DPO 为何能一步到位——重参数化消除了 RM 与 $\pi_\theta$ 的分离。

### 8.6 DPO 的失败模式

- **过拟合偏好数据**：β 大或 epoch 多，$\pi_\theta$ 把 margin 推过大，泛化差（[[IPO]] 专为修此）；
- **漂移过大**：β 大 + lr 大，$\pi_\theta$ 漂离 ref，语言能力崩（[[KL explosion]] 的 DPO 版）；
- **多样性丧失**：DPO 倾向压低 rejected，可能把合理但未被偏好的 mode 也压掉（[[mode collapse]] 的 DPO 版）；
- **偏好数据偏差放大**：DPO 忠实学偏好，若偏好有偏差（标注者偏好/采样偏差），$\pi_\theta$ 放大该偏差。
这些是 DPO 调参的诊断对象。

---
相关: [[对齐算法]]、[[RLHF (PPO)]]、[[SFT]]、[[Reward Model]]、[[PPO optimization]]、[[KL penalty与KL control]]、[[Bradley-Terry模型]]、[[IPO]]、[[KTO]]、[[reward hacking]]、[[KL explosion]]、[[mode collapse]]、[[MDP]]
