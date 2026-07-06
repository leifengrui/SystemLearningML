# entropy bonus

> **所属章节**: [[PPO]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 中等（探索与收敛的平衡项）

## 1. 一句话定义

**entropy bonus（熵奖励 / 熵正则）** 是在策略梯度 loss 里加一项 **$+\beta\,H(\pi_\theta(\cdot|s_t))$**（最大化 entropy），鼓励策略 $\pi_\theta$ 在每个状态保持一定**随机性 / 探索性**，防止策略过早收敛到确定性分布（deterministic policy）而陷入局部最优。它是 [[PPO clipped objective]] 完整 loss 的第三项（$-c_e\,S[\pi_\theta]$，因做梯度下降故取负），系数 $c_e$ 通常 0.01（游戏 RL）或 0/极小（LLM-RL）。来自 Williams 1992 的"entropy regularization"思想，也是 Soft Actor-Critic（SAC）"最大熵 RL"的根基。

> [!note] 一句话直觉
> 策略梯度天然会**走向确定性**——某个动作 advantage 大就猛增其概率，其他动作概率趋零。一旦确定，探索停止，可能卡在局部最优。entropy bonus 像一根"反向弹簧"：策略越确定（熵越低），弹簧拉得越用力，把概率往均匀方向拽，逼智能体继续试别的动作。$\beta$ 是弹簧强度：大→过度探索不收敛、小→探索不足卡局部。

## 2. 为什么需要它（动机与背景）

[[REINFORCE]]/[[Actor-Critic]]/[[PPO clipped objective]] 的梯度本质是"增大高 advantage 动作概率、减小低 advantage 动作概率"。这有一个隐忧：

1. **正反馈导致确定性化**：某动作 advantage 略正 → 概率增大 → 下次更可能采到它 → 估计的 advantage 更准、更可能仍正 → 概率继续增大……正反馈循环让策略**快速塌缩到少数动作**，熵 $\to0$；
2. **探索死亡**：策略一旦确定， rollout 永远走同一条路，看不到环境其他区，可能错过更优策略（局部最优）；
3. **早期 advantage 估计噪声大**：训初各动作 advantage 估得不准，若据此确定性化会锁死错误策略。

entropy bonus 提供**持续探索压力**：在 loss 里奖励高熵（接近均匀分布），抵消"确定性化"倾向。它与 [[variance reduction]] 的方向不同——后者降梯度方差，前者保策略多样性。两者协同让训练既稳又探。

LLM-RL 中 entropy bonus 的角色较微妙：LLM 已有 $\pi_{ref}$（SFT）当锚点（[[KL penalty与KL control]]），探索需求弱；且 LLM 分布本就多 mode（数千 token 都有非零概率），熵天然高。故 RLHF 常把 $c_e$ 设 0 或极小（1e-3 以下），避免与 KL penalty 拉扯导致训不稳。游戏 RL（离散动作少）则 $c_e$ 较大（0.01）促探索。

## 3. 核心概念详解

### 3.1 熵 $H$ 的定义

对离散分布 $\pi(\cdot|s)$：

$$
H(\pi(\cdot|s))=-\sum_a\pi(a|s)\log\pi(a|s)=\mathbb{E}_{a\sim\pi}\bigl[-\log\pi(a|s)\bigr]
$$

- $H\ge0$；
- **均匀分布熵最大** $H_{\max}=\log|\mathcal{A}|$（$|\mathcal{A}|$ 动作数）；
- **确定性分布熵为 0**（某动作概率 1，其余 0）；
- 熵高 → 分布平、随机性强、探索多；熵低 → 分布尖、确定性强、利用多。

连续动作（高斯策略 $\mathcal{N}(\mu,\sigma)$）：$H=\frac12\log(2\pi e\sigma^2)$（1D），调 $\sigma$ 即调熵——$\sigma$ 大则探索多。SAC 用此控制连续探索。

### 3.2 entropy bonus 的形式

加到策略梯度目标（**最大化**）：

$$
J_{\text{aug}}(\theta)=\mathbb{E}\left[\sum_t\nabla\log\pi_\theta(a_t|s_t)\,\hat A_t\right]+\beta\,\mathbb{E}_t\bigl[H(\pi_\theta(\cdot|s_t))\bigr]
$$

在 PPO loss（**梯度下降**，取负）里写成：

$$
L^{PPO}=\underbrace{-L^{CLIP}}_{\text{actor}}+\underbrace{c_v\,L^{VF}}_{\text{critic}}\underbrace{-c_e\,\mathbb{E}_t[H(\pi_\theta)]}_{\text{entropy bonus}}
$$

- $c_e$（即 $\beta$）：entropy 系数；
- 最大化 $H$ → 梯度推 $\pi_\theta$ **远离**确定性、**靠近**均匀；
- 注意符号：loss 里是 $-c_e H$（梯度下降时减小 loss = 增大 $H$）。

### 3.3 梯度方向

$\nabla_\theta H(\pi_\theta(\cdot|s))$ 把 logits 推向均匀。对 categorical logits $z$：$\pi=\text{softmax}(z)$，$\nabla_z H=\pi-1/|\mathcal{A}|\cdot\mathbf{1}$ 的某种形式（实际 $\nabla_z H=(\log|\mathcal{A}| - H)\pi - \pi\odot\log\pi$ 等价变形），本质是"压低最大 logit、抬高最小 logit"，让分布变平。PyTorch 的 `Categorical(logits=z).entropy()` 直接可微反传。

### 3.4 与 KL penalty 的对比（关键易混）

| 维度 | entropy bonus | [[KL penalty与KL control]] |
|---|---|---|
| 加什么 | $+H(\pi_\theta)$（自身熵） | $-\text{KL}(\pi_\theta\|\pi_{ref})$（与参考的距离） |
| 拉向 | **均匀分布**（最大熵） | **$\pi_{ref}$**（SFT 模型） |
| 方向 | 推**离**确定性（探索） | 拉**向**参考（保人格/防 OOD） |
| 防什么 | mode collapse / 局部最优 | policy collapse / reward hacking / KL explosion |
| LLM-RL | 常关或极小（$c_e<10^{-3}$） | 必开（$\beta\sim0.01\sim0.1$） |
| 游戏 RL | 较大（$c_e\approx0.01$） | 用 $\pi_{old}$ 早停，无 ref |

> [!note] 为何 LLM-RL 慎用 entropy bonus
> LLM 已有 $\pi_{ref}$ 当"探索锚点"——KL penalty 把 $\pi_\theta$ 拉向 $\pi_{ref}$，而 $\pi_{ref}$（SFT 模型）本身是高熵、多 mode 的语言分布。再加 entropy bonus 会让 $\pi_\theta$ 被两个方向拉扯（KL 拉向 ref、entropy 拉向均匀），可能训不稳且偏离语言分布。故 RLHF 常关 entropy 或用极小系数，靠 KL + reward 多样性保证探索。游戏 RL 无 ref，必须靠 entropy bonus 防确定性化。

### 3.5 与 temperature 的关系

策略的 temperature $\tau$ 控制 softmax 锐度：$\pi=\text{softmax}(z/\tau)$。$\tau\uparrow$ → 分布平、熵高、探索多；$\tau\downarrow$ → 分布尖、熵低、利用多。entropy bonus 等价于**自适应调高 $\tau$**——当熵低时梯度推 logits 变平（相当于升 $\tau$）。区别：temperature 是手动超参，entropy bonus 是 loss 里的自适应项（与 advantage 权衡）。SAC 用 entropy 自动学 $\alpha$（temperature 的逆），更系统。

### 3.6 系数 $\beta$ 的选择

- **太大**：策略被推成接近均匀，advantage 信号被淹没，不收敛或收敛到随机策略；
- **太小**：探索不足，策略快速确定性化，卡局部最优；
- **游戏 RL**：$c_e\approx0.01$ 是经验值（A3C/PPO baselines）；
- **LLM-RL**：$c_e=0$ 或 $<10^{-3}$，因有 KL penalty；
- **SAC**：$\alpha$ 自适应（让 entropy 维持 target），最系统。
- 监控：训练中看平均熵曲线，熵持续下降到 0 → 探索死亡，需调大 $c_e$。

### 3.7 最大熵 RL（Maximum Entropy RL）

SAC（Haarnoja 2018）的目标是 **最大熵 RL**：

$$
J_{\text{MaxEnt}}=\mathbb{E}_{\pi}\left[\sum_t\gamma^t\bigl(r_t+\alpha\,H(\pi(\cdot|s_t))\bigr)\right]
$$

即"在最大化 reward 的同时最大化 entropy"。$\alpha$ 自适应调到 entropy 维持 target $\bar{H}$。这把 entropy bonus 从"辅助正则"升格为"目标的一部分"，理论上有更好的探索性与收敛性。PPO 的 entropy bonus 是其简化版（fixed $\beta$）。

## 4. 数学原理 / 公式

### 4.1 entropy bonus 的梯度

对 categorical 策略 logits $z$，$\pi=\text{softmax}(z)$：

$$
H(\pi)=-\sum_a\pi_a\log\pi_a
$$

$$
\frac{\partial H}{\partial z_i}=\pi_i\bigl(H-\log\pi_i-1\bigr)+\pi_i=\pi_i\bigl(H-\log\pi_i\bigr)
$$

（用 $\partial\log\pi_i/\partial z_i=\pi_i-1$ 与链式法则整理）。该梯度在 $\pi_i$ 大（确定性）时为负（推 $z_i$ 下降，降该动作概率），在 $\pi_i$ 小时为正（推 $z_i$ 上升，增该动作概率）——即"削峰填谷"使分布变平。

### 4.2 与 KL 到均匀分布的关系

记均匀分布 $u(a)=1/|\mathcal{A}|$：

$$
\text{KL}(\pi\|u)=\sum_a\pi_a\log\frac{\pi_a}{1/|\mathcal{A}|}=\sum_a\pi_a\log\pi_a+\log|\mathcal{A}|=-H(\pi)+\log|\mathcal{A}|
$$

故 $H(\pi)=\log|\mathcal{A}|-\text{KL}(\pi\|u)$。**最大化 entropy 等价于最小化 $\pi$ 到均匀分布的 KL**——即 entropy bonus 是"把 $\pi_\theta$ 拉向均匀"的软约束。这与 [[KL penalty与KL control]] 把 $\pi_\theta$ 拉向 $\pi_{ref}$ 形式同构，只是目标分布不同（均匀 vs ref）。

### 4.3 在策略梯度定理里的位置

带 entropy bonus 的策略梯度：

$$
\nabla_\theta J_{\text{aug}}=\mathbb{E}\left[\sum_t\nabla\log\pi_\theta(a_t|s_t)\,\hat A_t+\beta\,\nabla_\theta H(\pi_\theta(\cdot|s_t))\right]
$$

第一项是标准 PG（[[REINFORCE]]），第二项是 entropy 梯度。二者相加：advantage 项做"利用"，entropy 项做"探索"。

### 4.4 与 reward shaping 的等价

把 $+\beta\,H$ 等价为 per-step reward shaping $r_t\leftarrow r_t+\beta\,H(\pi_\theta(\cdot|s_t))$，则策略梯度天然含 entropy 项（无需改 loss）。这种"reward shaping 视角"与 [[KL penalty与KL control]] 的 reward KL penalty 同构。但实践中多直接加在 loss 上，避免污染 advantage 估计。

## 5. 代码示例

```python
import torch, torch.nn as nn

class AC(nn.Module):
    def __init__(s,o,n): super().__init__(); s.a=nn.Sequential(nn.Linear(o,64),nn.Tanh(),nn.Linear(64,n)); s.c=nn.Sequential(nn.Linear(o,64),nn.Tanh(),nn.Linear(64,1))
    def pi(s,x): return torch.distributions.Categorical(logits=s.a(x))
    def v(s,x): return s.c(x).squeeze(-1)

ac=AC(4,2); opt=torch.optim.Adam(ac.parameters(),3e-3); gamma=0.99; c_e=0.01; c_v=0.5

def collect(n=200):
    buf=[]; s=torch.randn(4)
    for _ in range(n):
        d=ac.pi(s); a=d.sample(); sn=torch.randn(4); r=float(a==1)
        buf.append((s,a,r,sn)); s=sn
    return buf

buf=collect()
S=torch.stack([t[0] for t in buf]); A=torch.tensor([t[1] for t in buf])
R=torch.tensor([t[2] for t in buf],dtype=torch.float32); SN=torch.stack([t[3] for t in buf])

# advantage(MC 版简化,真实用 GAE)
with torch.no_grad():
    G=torch.zeros(len(buf)); g=0
    for t in reversed(range(len(buf))): g=R[t]+gamma*g; G[t]=g
    V=ac.v(S); adv=(G-V).detach(); adv=(adv-adv.mean())/(adv.std()+1e-8)

d=ac.pi(S); logp=d.log_prob(A); ent=d.entropy()           # ← entropy
actor_loss=-(logp*adv).mean()                              # 标准 PG
critic_loss=((ac.v(S)-G).mean())**2*0+((ac.v(S)-G)**2).mean()  # critic
entropy_bonus=ent.mean()                                   # 最大化
loss=actor_loss+c_v*critic_loss-c_e*entropy_bonus          # -c_e*H: 降 loss=增 H
opt.zero_grad(); loss.backward(); opt.step()
print(f"entropy={ent.mean().item():.3f} | actor={actor_loss.item():.3f} | critic={critic_loss.item():.3f}")

# 诊断:看熵是否随训练下降到 0(探索死亡)
# for step in ...: 记录 ent.mean(),若 →0 且 reward 停滞 → 调大 c_e
```

> [!tip] PyTorch 写法
> `torch.distributions.Categorical(logits=z).entropy()` 直接返回每个样本的熵，可微。`.mean()` 后加 `-c_e*` 进 loss 即可。连续动作（Normal）同理 `.entropy()`，但要注意 $\sigma$ 别被推到爆炸。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[policy与value function]]（策略分布）、信息论（熵定义）、[[PPO clipped objective]]（loss 的承载者）、Williams 1992 entropy regularization。
- **下游（应用）**: PPO/A3C 的第三项 loss、SAC（最大熵 RL，自适应 $\alpha$）、exploration 策略设计、RLHF（常关闭）。
- **对比 / 易混**:
  - **entropy bonus vs [[KL penalty与KL control]]**：见 §3.4，推离确定性 vs 拉向 ref，方向相反但可协同。
  - **entropy bonus vs temperature**：前者是 loss 自适应项，后者是手动超参；SAC 把二者合一（自适应 $\alpha$）。
  - **entropy bonus vs [[variance reduction]]**：前者保策略多样性（探索），后者降梯度方差（稳定），目标不同。
  - **entropy bonus vs advantage 归一化**：前者约束分布形状，后者约束权重尺度，互补。
  - **最大熵 RL vs 标准 RL**：前者目标含 $+\alpha H$，后者不含；最大熵 RL 探索更系统但需调 $\alpha$。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **符号反**：loss 里应是 **$-c_e\,H$**（梯度下降减小 loss = 增大 $H$）。写成 $+c_e H$ 会**压低**熵，加速确定性化，越训越差。
> 2. **entropy 没 detach 期望、但梯度要流**：$H$ 的梯度要流回 actor logits（这是 bonus 的作用），不能 detach；但 `ent.mean()` 作为监控值可 detach。区分"梯度路径"与"监控值"。
> 3. **$c_e$ 调过大**：策略被推成均匀，advantage 信号被淹没，不收敛或收敛到随机；游戏 RL 别超 0.05。
> 4. **LLM-RL 照搬游戏 $c_e=0.01$**：与 KL penalty 拉扯导致训不稳；RLHF 用 $c_e=0$ 或 $<10^{-3}$。
> 5. **以为 entropy bonus 能防 reward hacking**：它只防"确定性化导致的局部最优"，不防"在分布内钻空子"；reward hacking 要靠 RM 设计 + KL（[[KL penalty与KL control]]）。
> 6. **连续动作 $\sigma$ 被 entropy 推爆**：高斯策略 $H\propto\log\sigma$，$c_e$ 大会让 $\sigma$ 持续涨、采样发散；要 clamp $\sigma$ 或限幅。
> 7. **不看熵曲线**：熵→0 是探索死亡的信号，需早发现调 $c_e$；只看 reward 会误判"收敛了"。
> 8. **混淆 $H$ 与 $\text{KL}$**：$H(\pi)$ 是自身熵，$\text{KL}(\pi\|q)$ 是与 $q$ 的距离；二者通过 $H=\log|\mathcal{A}|-\text{KL}(\pi\|u)$ 联系（§4.2），但作用方向不同。
> 9. **entropy bonus 与 KL penalty 同时大开**：两个相反方向的大力拉扯，policy 在均匀与 ref 间震荡；选一个为主（LLM 选 KL，游戏选 entropy）。

## 8. 延伸细节

### 8.1 最大熵 RL 的理论优势

最大熵 RL（SAC）的 target $J=\mathbb{E}[\sum\gamma^t(r+\alpha H)]$ 有几个好处：
- **探索性**：高熵策略天然多 mode 探索；
- **多解学习**：reward 平坦区（多个等优动作）也被学（不只挑一个），policy 更鲁棒；
- **收敛性**：Q 函数在最大熵下更平滑，收敛更好；
- **可逆性**：最大熵策略是 RL 的"玻尔兹曼分布"，有统计物理含义。

SAC 用双 Q 网络（clipped double Q）+ 自适应 $\alpha$（让 entropy 维持 target $\bar H$），是连续控制 SOTA。

### 8.2 LLM-RL 中 entropy 的现实

RLHF 中 $\pi_\theta$ 是 LLM，token 分布本就高熵（vocab 数千）。entropy bonus 的边际作用：
- **不开**：靠 KL penalty 拉 $\pi_{ref}$ 保多 mode，足够；
- **开小系数**：进一步防"塌缩到重复 token"（mode collapse 的一种），但易与 KL 冲突；
- **诊断**：监控 token 级熵，若某层 logits 变尖（熵↓）且 response 重复 → 可能 mode collapse，可临时升 $c_e$。

实践中 RLHF 多关 entropy，靠 KL + 多样性 reward + temperature 采样保证探索。

### 8.3 entropy 与 intrinsic reward

另一种探索机制是 **intrinsic reward / curiosity**（ICM、RND）：给"新奇状态"额外 reward。它与 entropy bonus 都促探索，但机制不同：
- entropy bonus：约束策略分布形状（直接推 $\pi$ 变平）；
- intrinsic reward：约束状态访问（奖励到未见过的状态），通过 reward 间接影响 $\pi$。

二者可叠加，但 LLM-RL 都少用（成本高、LLM 探索需求弱）。

### 8.4 entropy bonus 在 PPO 中的占比

PPO loss 三项的典型量级（游戏 RL）：
- actor loss（$L^{CLIP}$）：~1
- critic loss（$L^{VF}$）：~1，系数 0.5
- entropy：~0.7（$\log|\mathcal{A}|$ 量级），系数 0.01 → 贡献 ~0.007

故 entropy bonus 是**小但持续**的项，作用是"稳态偏置"而非主导。LLM-RL 中 actor loss 与 KL penalty 主导，entropy 贡献更小。

### 8.5 与 DPO 的关系

DPO（[[DPO]]）是 RLHF 的闭式重写，**不含 entropy bonus**——它优化的是"reward - β·KL"，没有显式 entropy 项。故 DPO 训出的 policy 的探索性完全由 $\pi_{ref}$ 与偏好数据决定，无 entropy 调节空间。这是 DPO 与 PPO 的一个差异：PPO 可用 entropy bonus 调探索，DPO 不能。

### 8.6 entropy 系数的自适应（SAC 式）

SAC 的 $\alpha$ 自适应：让 $\mathbb{E}[H(\pi)]$ 维持 target $\bar H$（如 $\log|\mathcal{A}|$ 的一定比例）：

```
alpha_loss = -(alpha * (log_pi + target_entropy).detach()).mean()
opt_alpha.zero_grad(); alpha_loss.backward(); opt_alpha.step()
```

这把 entropy bonus 的系数从手动超参变成自动调节，是最大熵 RL 的核心工程。PPO 一般不自适应 $c_e$（fixed），因 PPO 已有 clip + KL 多重约束，entropy 退居辅助。

---
相关: [[PPO]]、[[PPO clipped objective]]、[[KL penalty与KL control]]、[[variance reduction]]、[[reward normalization]]、[[policy与value function]]、[[reward hacking]]、[[mode collapse]]、[[DPO]]、[[policy collapse]]
