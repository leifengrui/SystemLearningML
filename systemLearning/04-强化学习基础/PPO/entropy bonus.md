# entropy bonus

> **所属章节**: [[PPO]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**entropy bonus（熵奖励 / 熵正则）** 是在策略梯度 loss 里加一项 **$-\beta\,H(\pi_\theta)$**（梯度下降时减小 loss = 增大 entropy），鼓励策略 $\pi_\theta$ 保持一定**随机性 / 探索性**，防止过早收敛到确定性分布而陷入局部最优。它是 [[PPO clipped objective]] 完整 loss 的第三项，系数 $\beta$ 通常 0.01（游戏 RL）或 0/极小（LLM-RL）。来自 Williams 1992 的 entropy regularization，也是 SAC"最大熵 RL"的根基。

> [!note] 一句话直觉
> 策略梯度天然走向确定性——某动作 advantage 大就猛增其概率，其他趋零，正反馈让策略快速塌缩。entropy bonus 像反向弹簧：策略越确定（熵越低）拉得越用力，把概率往均匀方向拽。$\beta$ 是弹簧强度：大→过度探索不收敛、小→探索不足卡局部。

## 2. 为什么需要它

[[REINFORCE]]/[[Actor-Critic]]/[[PPO clipped objective]] 的梯度本质是"增大高 advantage 动作概率、减小低 advantage 动作概率"。隐忧：

1. **正反馈导致确定性化**：某动作 advantage 略正 → 概率增大 → 下次更可能采到 → advantage 估计更准、更可能仍正 → 概率继续增大……策略快速塌缩到少数动作，熵 $\to0$。
2. **探索死亡**：一旦确定，rollout 永远走同一条路，看不到环境其他区，可能错过更优策略。
3. **早期 advantage 噪声大**：训初各动作 advantage 估不准，据此确定性化会锁死错误策略。

entropy bonus 在 loss 里奖励高熵（接近均匀），抵消确定性化倾向。与 [[variance reduction]] 方向不同——后者降梯度方差，前者保策略多样性。

LLM-RL 角色较微妙：LLM 已有 $\pi_{ref}$（SFT）当锚点（[[KL penalty与KL control]]），探索需求弱；且 LLM 分布本就多 mode（数千 token 都有非零概率），熵天然高。故 RLHF 常把 $\beta$ 设 0 或极小（$<10^{-3}$），避免与 KL penalty 拉扯导致训不稳。游戏 RL（离散动作少）则 $\beta$ 较大（0.01）促探索。

## 3. 核心概念详解

### 3.1 熵 $H$ 的定义

对离散分布 $\pi(\cdot|s)$：

$$
H(\pi)=-\sum_a\pi(a|s)\log\pi(a|s)=\mathbb{E}_{a\sim\pi}[-\log\pi(a|s)]
$$

- $H\ge0$；均匀分布熵最大 $H_{\max}=\log|\mathcal{A}|$；确定性分布熵为 0；
- 熵高 → 分布平、随机性强、探索多；熵低 → 分布尖、确定性强、利用多。
- 连续动作（高斯 $\mathcal{N}(\mu,\sigma)$）：$H=\frac12\log(2\pi e\sigma^2)$（1D），调 $\sigma$ 即调熵。SAC 用此控制连续探索。

### 3.2 entropy bonus 的形式

PPO loss（**梯度下降**，取负）：

$$
L^{PPO}=\underbrace{-L^{CLIP}}_{\text{actor}}+\underbrace{c_v\,L^{VF}}_{\text{critic}}\underbrace{-c_e\,\mathbb{E}_t[H(\pi_\theta)]}_{\text{entropy bonus}}
$$

- $c_e$（即 $\beta$）：entropy 系数；
- 最大化 $H$ → 推 $\pi_\theta$ 远离确定性、靠近均匀；
- 注意符号：loss 里是 $-c_e H$。

### 3.3 与 KL penalty 的对比（关键易混）

| 维度 | entropy bonus | [[KL penalty与KL control]] |
|---|---|---|
| 加什么 | $+H(\pi_\theta)$（自身熵） | $-\text{KL}(\pi_\theta\|\pi_{ref})$（与参考的距离） |
| 拉向 | **均匀分布**（最大熵） | **$\pi_{ref}$**（SFT 模型） |
| 方向 | 推**离**确定性（探索） | 拉**向**参考（保人格/防 OOD） |
| 防什么 | mode collapse / 局部最优 | policy collapse / reward hacking / KL explosion |
| LLM-RL | 常关或极小（$c_e<10^{-3}$） | 必开（$\beta\sim0.01\sim0.1$） |
| 游戏 RL | 较大（$c_e\approx0.01$） | 用 $\pi_{old}$ 早停，无 ref |

> [!note] 为何 LLM-RL 慎用 entropy bonus
> LLM 已有 $\pi_{ref}$ 当探索锚点——KL penalty 把 $\pi_\theta$ 拉向 $\pi_{ref}$，而 $\pi_{ref}$ 本身是高熵多 mode 的语言分布。再加 entropy bonus 会让 $\pi_\theta$ 被两个方向拉扯（KL 拉向 ref、entropy 拉向均匀），可能训不稳且偏离语言分布。故 RLHF 常关 entropy 或用极小系数。

### 3.4 系数 $\beta$ 的选择

- **太大**：策略被推成接近均匀，advantage 信号被淹没，不收敛或收敛到随机策略；
- **太小**：探索不足，策略快速确定性化，卡局部最优；
- **游戏 RL**：$c_e\approx0.01$ 是经验值（A3C/PPO baselines）；
- **LLM-RL**：$c_e=0$ 或 $<10^{-3}$，因有 KL penalty；
- **SAC**：$\alpha$ 自适应（让 entropy 维持 target $\bar H$），最系统。
- 监控：训练中看平均熵曲线，熵持续下降到 0 → 探索死亡，需调大 $c_e$。

## 4. 关键推导

### 4.1 entropy bonus 的梯度

对 categorical logits $z$，$\pi=\text{softmax}(z)$：

$$
\frac{\partial H}{\partial z_i}=\pi_i\bigl(H-\log\pi_i\bigr)
$$

该梯度在 $\pi_i$ 大（确定性）时为负（推 $z_i$ 下降），在 $\pi_i$ 小时为正（推 $z_i$ 上升）——即"削峰填谷"使分布变平。PyTorch 的 `Categorical(logits=z).entropy()` 直接可微反传。

### 4.2 与 KL 到均匀分布的关系

记均匀分布 $u(a)=1/|\mathcal{A}|$：

$$
\text{KL}(\pi\|u)=\sum_a\pi_a\log\frac{\pi_a}{1/|\mathcal{A}|}=-H(\pi)+\log|\mathcal{A}|
$$

故 $H(\pi)=\log|\mathcal{A}|-\text{KL}(\pi\|u)$。**最大化 entropy 等价于最小化 $\pi$ 到均匀分布的 KL**——entropy bonus 是"把 $\pi_\theta$ 拉向均匀"的软约束，与 [[KL penalty与KL control]] 把 $\pi_\theta$ 拉向 $\pi_{ref}$ 形式同构，只是目标分布不同。

## 5. 代码示例

```python
import torch
# d = Categorical(logits=z)
ent = d.entropy().mean()                          # 每样本熵,可微
loss = actor_loss + 0.5*critic_loss - c_e*ent     # -c_e*H: 降 loss = 增 H
```

> [!tip] PyTorch 写法
> `torch.distributions.Categorical(logits=z).entropy()` 直接返回每个样本的熵，可微。`.mean()` 后加 `-c_e*` 进 loss。连续动作（Normal）同理 `.entropy()`，但要注意 $\sigma$ 别被推到爆炸。

## 6. 与其他知识点的关系

- **上游**: [[policy与value function]]（策略分布）、信息论（熵定义）、[[PPO clipped objective]]（loss 的承载者）。
- **下游**: PPO/A3C 的第三项 loss、SAC（最大熵 RL，自适应 $\alpha$）、RLHF（常关闭）。
- **对比**:
  - **entropy bonus vs [[KL penalty与KL control]]**：见 §3.3，推离确定性 vs 拉向 ref，方向相反但可协同。
  - **entropy bonus vs temperature**：前者是 loss 自适应项，后者是手动超参；SAC 把二者合一（自适应 $\alpha$）。
  - **entropy bonus vs [[variance reduction]]**：前者保策略多样性（探索），后者降梯度方差（稳定），目标不同。

## 7. 常见误区

1. **符号反**：loss 里应是 $-c_e\,H$（梯度下降减小 loss = 增大 $H$）。写成 $+c_e H$ 会压低熵，加速确定性化，越训越差。
2. **entropy 梯度被误 detach**：$H$ 的梯度要流回 actor logits（这是 bonus 的作用），不能 detach；但 `ent.mean()` 作为监控值可 detach。
3. **$c_e$ 调过大**：策略被推成均匀，advantage 信号被淹没，不收敛或收敛到随机；游戏 RL 别超 0.05。
4. **LLM-RL 照搬游戏 $c_e=0.01$**：与 KL penalty 拉扯导致训不稳；RLHF 用 $c_e=0$ 或 $<10^{-3}$。
5. **以为 entropy bonus 能防 reward hacking**：它只防"确定性化导致的局部最优"，不防"在分布内钻空子"；reward hacking 要靠 RM 设计 + KL。
6. **连续动作 $\sigma$ 被 entropy 推爆**：高斯策略 $H\propto\log\sigma$，$c_e$ 大会让 $\sigma$ 持续涨、采样发散；要 clamp $\sigma$ 或限幅。
7. **不看熵曲线**：熵→0 是探索死亡的信号，需早发现调 $c_e$；只看 reward 会误判"收敛了"。
8. **entropy bonus 与 KL penalty 同时大开**：两个相反方向的大力拉扯，policy 在均匀与 ref 间震荡；选一个为主（LLM 选 KL，游戏选 entropy）。

---
相关: [[PPO]]、[[PPO clipped objective]]、[[KL penalty与KL control]]、[[variance reduction]]、[[policy与value function]]、[[reward hacking]]、[[DPO]]
