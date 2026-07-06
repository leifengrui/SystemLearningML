# IPO

> **所属章节**: [[对齐算法]]
> **所属模块**: [[05-LLM-RL对齐]]
> **难度**: 中（DPO 的小改进，关键是理解为何 DPO 会过拟合 + IPO 的 target margin）

## 1. 一句话定义

**IPO（Identity Preference Optimization，恒等偏好优化）** 是 Azar et al. 2023 提出的 [[DPO]] 改进版：用 **IPO metric（squared loss）** 替换 DPO 的 Bradley-Terry 概率损失，给 margin 一个**有界 target** $\frac{1}{2\beta}$，防止 DPO 把 chosen/rejected 的 log-ratio 差无限推大的**概率过拟合**。IPO 损失 $\mathcal{L}_{\text{IPO}}=\bigl(\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}-\frac{1}{2\beta}\bigr)^2$，当 margin 达到 $\frac{1}{2\beta}$ 时梯度为 0 自然停，而 DPO 的 BT 损失 margin→∞ 时才趋 0、会一直推。它和 DPO 仍是同一 KL-constrained 目标，只是偏好损失的**正则形式**不同。

> [!note] IPO = DPO 换损失函数
> 一句话记忆：**"DPO 用 BT 损失会把 margin 推太大，IPO 换成 squared loss 给个 target margin 就停"**。目标、闭式解、数据、模型结构与 DPO 全同，**唯一区别是损失函数**。

## 2. 为什么需要它（动机与背景）

[[DPO]] 用 Bradley-Terry 损失 $\mathcal{L}=-\log\sigma(m)$（$m$ 是 margin）。问题是 BT 是**概率模型损失**，其梯度 $1-\sigma(m)=\sigma(-m)$ 在 $m\to\infty$ 时才趋 0：
- 即使 $\pi_\theta$ 已把 chosen 推得远高于 rejected，BT 损失仍有梯度继续推；
- 实践中 DPO 倾向把 $\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}$ 推到很大，**margin 过大 → 过拟合偏好数据 → 泛化差**；
- 这叫**概率过拟合**：BT 损失最优解在 $m\to\infty$，但实际偏好数据有噪声，不该无限推。

IPO 的动机：**给 margin 一个有界 target**。用 squared loss $(m-\frac{1}{2\beta})^2$，当 $m=\frac{1}{2\beta}$ 时 loss=0、梯度=0，自然停。这等价于"偏好数据告诉我 chosen 比 rejected 好 $\frac{1}{2\beta}$ 这么多就够了，别再推了"。IPO 来自一个更一般的理论框架（IPS，Identity Preference Shift），把偏好优化看成在重参数化空间做回归，target 由 β 决定。

## 3. 核心概念详解

### 3.1 DPO 的过拟合问题（IPO 要修的）

DPO 损失 $\mathcal{L}_{\text{DPO}}=-\log\sigma(m)$，$m=\beta(\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})$：
- 最优解在 $m\to\infty$（loss→0 但永不达 0）；
- 梯度 $\nabla\mathcal{L}=-\sigma(-m)\nabla m$，$\sigma(-m)$ 衰减慢（指数而非 squared）；
- 结果：DPO 训练越久 margin 越大，$\pi_\theta$ 漂离 $\pi_{\text{ref}}$ 越远 → 过拟合 + 漂移。

> [!warning] 概率过拟合
> BT 损失的最优解在无穷大 margin，但偏好数据有噪声/偏差，无限推 margin 会把噪声也放大。这是 DPO 实践中过拟合的根源。

### 3.2 IPO 的解法：target margin + squared loss

IPO 用 squared loss 替换 BT：
$$
\mathcal{L}_{\text{IPO}}=\bigl(\underbrace{\log\tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\log\tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}}_{\text{log-ratio 差 }m'}-\underbrace{\tfrac{1}{2\beta}}_{\text{target}}\bigr)^2
$$

- **target margin** $\frac{1}{2\beta}$：偏好数据暗示的"理想 log-ratio 差"；
- **squared loss**：当 $m'=\frac{1}{2\beta}$ 时 loss=0、梯度=0，**自然停**；
- 不再无限推，防过拟合；
- β 控制 target：β 小 → target 大 → 偏移强；β 大 → target 小 → 偏移弱（与 DPO 的 β 角色一致）。

### 3.3 IPO 的理论框架（IPS）

IPO 来自 Azar et al. 的 **Identity Preference Shift (IPS)** 框架：把 RLHF 目标的闭式解 $\pi^*=\pi_{\text{ref}}\exp(r/\beta)/Z$ 取恒等变换（identity），得 $\log\frac{\pi^*}{\pi_{\text{ref}}}=\frac{r}{\beta}-\log Z$。对一对 $(y_w,y_l)$，理想 log-ratio 差 $=\frac{r_w-r_l}{\beta}$。若假设偏好数据的"真实 reward 差"约 $1/2$（归一化假设），则 target $=\frac{1}{2\beta}$。这给了 squared loss 的回归 target。DPO 用 BT 是另一种选择（概率建模），IPO 用 squared 是回归建模——后者有界、防过拟合。

### 3.4 IPO vs DPO 的损失对比

| 维度 | DPO（BT 损失） | IPO（squared 损失） |
|---|---|---|
| 损失形式 | $-\log\sigma(m)$ | $(m'-\frac{1}{2\beta})^2$ |
| 最优 margin | $\to\infty$（无界） | $\frac{1}{2\beta}$（有界） |
| 梯度衰减 | $\sigma(-m)$，指数慢 | $2(m'-\frac{1}{2\beta})$，达 target 即 0 |
| 过拟合风险 | 高（margin 被推过大） | 低（有 target 停） |
| 建模视角 | 概率（BT 模型） | 回归（identity metric） |
| β 角色 | 温度（偏离 ref 强度） | 同 + 决定 target margin |

### 3.5 IPO 的训练流程

与 DPO **完全相同**，只改 loss 一行：
```
输入:偏好对 {(x, y_w, y_l)}, SFT 模型 π_ref, β
1. π_θ ← π_ref
2. for each batch:
     a. 算 logp_θ(y_w), logp_θ(y_l)        # π_θ 前向
     b. 算 logp_ref(y_w), logp_ref(y_l)    # π_ref 冻结前向
     c. margin' = log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))
     d. loss = (margin' - 1/(2β))²          # ← IPO 唯一改动
     e. backward, step
3. 输出 π_θ
```

## 4. 数学原理 / 公式

### 4.1 IPO 损失推导（从 IPS 框架）

由 [[DPO]] §4.2 的闭式解 $\pi^*(y|x)=\pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)/Z(x)$，取恒等变换：
$$
\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}=\frac{r(x,y)}{\beta}-\log Z(x) \tag{1}
$$

对偏好对 $(y_w,y_l)$，相减消 $Z$：
$$
\log\frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\log\frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}=\frac{r(x,y_w)-r(x,y_l)}{\beta} \tag{2}
$$

IPO 假设偏好数据蕴含的"真实 reward 差" $r(x,y_w)-r(x,y_l)$ 归一化为 $1/2$（即 chosen 比 rejected 在 reward 上高半个单位），则理想 log-ratio 差 target 为：
$$
\log\frac{\pi^*(y_w)}{\pi_{\text{ref}}(y_w)}-\log\frac{\pi^*(y_l)}{\pi_{\text{ref}}(y_l)}=\frac{1}{2\beta} \tag{3}
$$

把 $\pi^*$ 换成正在训的 $\pi_\theta$，用 squared loss 回归到此 target：
$$
\boxed{\mathcal{L}_{\text{IPO}}(\theta)=\mathbb{E}_{(x,y_w,y_l)}\Bigl[\Bigl(\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}-\frac{1}{2\beta}\Bigr)^2\Bigr]} \tag{4}
$$

### 4.2 与 DPO 损失的对照

DPO（[[DPO]] §4.4）：
$$
\mathcal{L}_{\text{DPO}}=-\mathbb{E}\bigl[\log\sigma\bigl(\beta\log\tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\beta\log\tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\bigr)\bigr] \tag{5}
$$

记 $m'=\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}$，则 DPO 的 margin $m=\beta m'$，IPO target $m'=\frac{1}{2\beta}$ 即 $m=\beta\cdot\frac{1}{2\beta}=\frac{1}{2}$。

| | DPO | IPO |
|---|---|---|
| 用 $m'$ 表达 | $-\log\sigma(\beta m')$ | $(m'-\frac{1}{2\beta})^2$ |
| 最优 $m'$ | $\to\infty$ | $\frac{1}{2\beta}$ |
| 最优 $m=\beta m'$ | $\to\infty$ | $\frac{1}{2}$ |

> [!tip] 直觉
> DPO 把 margin 推向无穷；IPO 把 margin 推到 $\frac{1}{2}$（在 $\beta m'$ 空间）就停。后者有界，防过拟合。

### 4.3 梯度对比

DPO 梯度（[[DPO]] §4.5）：$\nabla\mathcal{L}_{\text{DPO}}=-\sigma(-\beta m')\cdot\beta\nabla m'$，$\sigma(-\beta m')$ 衰减慢。

IPO 梯度：
$$
\nabla_\theta\mathcal{L}_{\text{IPO}}=2\Bigl(m'-\frac{1}{2\beta}\Bigr)\nabla_\theta m' \tag{6}
$$

- 当 $m'<\frac{1}{2\beta}$：梯度正，推大 $m'$（提升 chosen 压低 rejected）；
- 当 $m'=\frac{1}{2\beta}$：梯度 0，停；
- 当 $m'>\frac{1}{2\beta}$：梯度负，拉回（防过推）。

这是 IPO 防过拟合的力学：**有界 target + squared 梯度自然刹车**。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F, copy

class LM(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)
    def logp(s, x, y):
        ctx = torch.cat([x, y[:, :-1]], 1)
        return F.log_softmax(s(ctx),-1).gather(-1, y.unsqueeze(-1)).squeeze(-1).sum(-1)

V, D = 50, 32
pi_sft = LM(V, D)
pi_theta = copy.deepcopy(pi_sft)
pi_ref = copy.deepcopy(pi_sft)
for p in pi_ref.parameters(): p.requires_grad_(False)
opt = torch.optim.Adam(pi_theta.parameters(), 1e-6)
beta = 0.1

pref_data = [(torch.randint(0,V,(4,4)), torch.randint(0,V,(4,8)), torch.randint(0,V,(4,8))) for _ in range(20)]

for ep in range(3):
    for x, yw, yl in pref_data:
        logp_w = pi_theta.logp(x, yw); logp_l = pi_theta.logp(x, yl)
        with torch.no_grad():
            ref_w = pi_ref.logp(x, yw); ref_l = pi_ref.logp(x, yl)
        # IPO 损失 (4):squared loss,target = 1/(2β)
        margin = (logp_w - ref_w) - (logp_l - ref_l)      # m'
        loss = ((margin - 1.0/(2*beta))**2).mean()        # ← IPO 唯一改动(DPO 是 -logσ(β·m'))
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (margin > 0).float().mean()
            drift = (logp_w - ref_w).abs().mean()
    print(f"ep{ep} loss={loss:.3f} margin={margin.mean():.2f} (target={1/(2*beta):.2f}) acc={acc:.2f} drift={drift:.2f}")

print("IPO 训练完成:margin 收敛到 target 1/(2β),防 DPO 的过推")
```

> [!tip] TRL 支持
> TRL 的 `DPOTrainer` 设 `loss_type="ipo"` 即用 IPO 损失，其余流程与 DPO 全同。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[DPO]]（IPO 是其损失改进版，同目标同闭式解）、[[SFT]]、[[KL penalty与KL control]]、[[Bradley-Terry模型]]（DPO 用它，IPO 替换它）。
- **下游（应用）**: 对齐后 LLM（IPO 版，实践中不如 DPO 主流）、[[KTO]]（另一 DPO 变体，非成对）。
- **对比 / 易混**:
  - **IPO vs [[DPO]]**：同目标同数据同模型，**唯一区别是损失**（squared vs BT）。IPO 防 DPO 的 margin 过推。
  - **IPO vs [[KTO]]**：IPO 仍需成对偏好，KTO 只需二元标签。
  - **IPO 的 squared vs hinge**：IPO 原文是 squared $(m'-t)^2$，有些实现用 hinge $\max(0,m'-t)^2$ 或 $\max(0,t-m')$；squared 是原始形式。
  - **IPO 的 target $\frac{1}{2\beta}$ vs DPO 的无 target**：IPO 有界，DPO 无界。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"IPO 是新目标"**：错。与 DPO 同一 KL-constrained 目标，只改损失函数。
> 2. **"IPO 用 hinge loss"**：不完全。原始 IPO 是 squared loss $(m'-\frac{1}{2\beta})^2$，hinge 是变体。
> 3. **"IPO 一定比 DPO 好"**：不一定。IPO 防 margin 过推，但 target $\frac{1}{2\beta}$ 假设 reward 差归一化为 $1/2$，若实际偏好 reward 差与此假设偏离大，IPO 也可能欠拟合。实践中 DPO 仍主流，IPO 是特定场景的改进。
> 4. **"IPO 的 target 是 $\frac{1}{2}$"**：要看空间。在 $m'$（log-ratio 差）空间 target 是 $\frac{1}{2\beta}$；在 $m=\beta m'$ 空间 target 是 $\frac{1}{2}$。别混。
> 5. **"IPO 不需要 β"**：错。β 决定 target margin 与偏离 ref 强度，仍需调。
> 6. **"IPO 不会漂移"**：会，但比 DPO 轻。squared loss 在达 target 后梯度为 0，但若 lr 大或数据偏差，仍会漂离 ref。
> 7. **"IPO 和 DPO 的代码差异大"**：错。只改 loss 一行，其余（数据/前向/π_ref 冻结）全同。

## 8. 延伸细节

### 8.1 DPO 过拟合的实证

实践中 DPO 训练久了 margin 持续涨、人类评估先升后降，是 BT 损失无界最优的体现。IPO 的 squared loss 在 margin 达 target 后训练自然停滞，margin 曲线趋于平台，是诊断 IPO 起效的信号。

### 8.2 IPO 的假设局限

IPO 的 target $\frac{1}{2\beta}$ 假设偏好数据的真实 reward 差归一化为 $1/2$。若：
- 偏好数据 reward 差异大（有的容易有的难）：单一 target 不准；
- 偏好有噪声：target 被噪声拉偏；
- β 选不对：target 偏离实际。
这些场景 IPO 可能欠拟合或仍过拟合。更鲁棒的变体（rDPO/robust DPO）对此改进。

### 8.3 IPO 的实践地位

IPO 在学术上重要（揭示 DPO 的概率过拟合问题 + 提供一般框架），但**实践中不如 DPO 主流**。原因：
- DPO 简单且多数场景够用；
- IPO 的 target 假设强，调参更微妙；
- 社区已习惯 DPO + 早停（用 epoch 控制防过推，替代 IPO 的 target）。
但 IPO 的理论洞察（BT 损失无界最优）是理解 DPO 失败模式的关键，值得学。

### 8.4 DPO 家族的损失谱系

| 损失 | 形式 | 最优 margin | 特点 |
|---|---|---|---|
| BT（DPO） | $-\log\sigma(m)$ | $\infty$ | 概率模型，主流 |
| squared（IPO） | $(m'-\frac{1}{2\beta})^2$ | $\frac{1}{2\beta}$ | 有界 target，防过推 |
| hinge | $\max(0,t-m')^2$ | $t$ | 类 SVM，鲁棒 |
| robust（rDPO） | 噪声鲁棒损失 | 自适应 | 抗偏好噪声 |

这些是 DPO 损失的变体，目标与闭式解全同，只改最后的损失形式。

---
相关: [[对齐算法]]、[[DPO]]、[[RLHF (PPO)]]、[[SFT]]、[[KL penalty与KL control]]、[[Bradley-Terry模型]]、[[KTO]]、[[reward hacking]]、[[KL explosion]]
