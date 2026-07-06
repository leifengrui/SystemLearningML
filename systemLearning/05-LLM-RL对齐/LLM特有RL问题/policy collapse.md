# policy collapse

> **所属章节**: [[LLM特有RL问题]]
> **所属模块**: [[05-LLM-RL对齐]]
> **难度**: 中（机制清晰，与 entropy/KL/reward 集中相关）

## 1. 一句话定义

**policy collapse（策略坍缩）** 是 RLHF 中 PPO 阶段的失败模式：actor $\pi_\theta$ 的**策略概率分布塌缩**——熵 $H(\pi_\theta)\to0$，概率质量集中到少数 token/序列上，策略失去探索能力。机制是 PPO 的 reward 信号集中（少数 mode 拿高分）+ 缺乏 entropy bonus + [[KL explosion]] 推动 + advantage 偏置，使 $\pi_\theta$ 朝单一高 reward mode 坍缩。与 [[mode collapse]] 紧密关联但侧重不同：**policy collapse 是机制层（策略熵塌缩、概率集中）**，[[mode collapse]] 是现象层（输出 mode 单一、多样性丧失）。表现:生成高度雷同、重复同一回答、无法应对多样输入。防法:entropy bonus、KL 约束、temperature、advantage 归一化、reward 分散化。

> [!note] 与 [[mode collapse]] 的分工
> - **policy collapse（本篇）**：机制层——为什么 PPO 会让策略熵塌缩、概率质量集中，数学上是 $H(\pi_\theta)\to0$。
> - **[[mode collapse]]**：现象层——输出 mode 单一、多样性丧失的诊断与防法（temperature/采样/多样化），类比 GAN 的 mode collapse。
> 两者常并发，本篇讲"为什么会塌"，[[mode collapse]] 讲"塌了怎么救"。

## 2. 为什么需要它（动机与背景）

PPO 最大化 $\mathbb{E}[r_\phi]$，若 reward 信号集中（少数 mode 拿高分）：
1. **reward 集中**：RM 对某类 response 给高分，其余低分，actor 朝高 reward mode 集中；
2. **无探索压力**：[[entropy bonus]] 关闭或系数过小，策略熵无保底；
3. **KL 爆推塌缩**：[[KL explosion]] 时 $\pi_\theta$ 漂移后易塌缩到某 mode（漂移+塌缩常连锁）；
4. **advantage 偏置**：advantage 归一化不当，少数样本优势被放大；
5. **PPO clip 的副作用**：clip 限制 ratio，已塌缩的策略难扩散回。

结果：$\pi_\theta$ 熵→0，概率集中，生成雷同。这是策略优化的通病（强化学习中"探索-利用"失衡），在 LLM 中因 reward 集中 + 高维 action 空间而尤甚。理解机制才能对症防。

## 3. 核心概念详解

### 3.1 policy collapse 的机制

```
reward 集中(少数 mode 高分)
    ↓
PPO 推 π_θ 朝高 reward mode
    ↓
熵 H(π_θ) 下降(概率集中)
    ↓
无 entropy bonus 阻止
    ↓
KL 爆推漂移 + 塌缩连锁
    ↓
policy collapse: H→0,策略失探索能力
```

### 3.2 策略熵塌缩的数学信号

- $H(\pi_\theta(\cdot|s))=-\sum_a\pi_\theta(a|s)\log\pi_\theta(a|s)$；
- collapse 时 $H\to0$（概率质量给单一 token）；
- 监控：训练中 token 级 entropy 曲线，塌缩 = 单调降趋 0。

### 3.3 policy collapse vs mode collapse（关键区分）

| 维度 | policy collapse（本篇） | [[mode collapse]] |
|---|---|---|
| 层次 | 机制（策略熵塌缩） | 现象（输出 mode 单一） |
| 信号 | $H(\pi_\theta)\to0$ | 生成样本雷同/重复 |
| 成因 | reward 集中 + 无 entropy bonus + KL 爆 | policy collapse 的下游表现 + 采样策略 |
| 防法 | entropy bonus / KL 约束 / advantage 归一 | temperature / nucleus / 多样化采样 |
| 关系 | 上游（机制） | 下游（现象） |

### 3.4 PPO 各组件对 collapse 的影响

| 组件 | 对 collapse 作用 |
|---|---|
| **reward 集中** | 主因，推策略朝单一 mode |
| **entropy bonus** | 防塌缩，保策略熵底（[[entropy bonus]] §3.2） |
| **KL penalty** | 间接防，限制漂移后塌缩 |
| **advantage 归一化** | 防少数样本优势放大 |
| **clip** | 副作用，已塌缩难扩散回 |
| **K epoch** | 过多加剧塌缩（反复强化同一 mode） |

### 3.5 防法总览

| 防法 | 机制 |
|---|---|
| **entropy bonus** | loss 加 $-c_e H(\pi_\theta)$，保熵底 |
| **KL 约束** | 限制漂移后塌缩 |
| **advantage 归一化** | 防优势集中 |
| **reward 分散化** | RM 设计让 reward 不过度集中 |
| **temperature 调节** | 推理时升温恢复多样性（详见 [[mode collapse]]） |
| **数据多样** | rollout prompt 多样，防策略过拟合单一模式 |

## 4. 数学原理 / 公式

### 4.1 策略熵与塌缩

$$
H(\pi_\theta(\cdot|s))=-\sum_a\pi_\theta(a|s)\log\pi_\theta(a|s)\ge0
$$

当 $\pi_\theta$ 塌缩到单点 $a^*$（$\pi_\theta(a^*|s)\to1$），$H\to0$。均匀分布时 $H=\log|A|$ 最大。

### 4.2 entropy bonus 的防塌缩

PPO loss 加 entropy 项（[[entropy bonus]] §3.2）：
$$
L=L^{CLIP}+c_v L^{VF}-c_e\,\mathbb{E}[H(\pi_\theta(\cdot|s))]
$$

梯度上 $-c_e H$ 推 $\pi_\theta$ 朝高熵（均匀）方向，对抗 reward 朝低熵（集中）的拉力。$c_e$ 大 → 熵强保底但 reward 学慢；$c_e$ 小 → 易塌缩。LLM-RL 常取 $c_e$ 极小或 0（因 LLM action 空间大，entropy bonus 信号弱），故 LLM 更易 policy collapse。

### 4.3 reward 集中与塌缩的动力学

设 reward $r(a)$ 在 $a^*$ 处最高。PPO 策略梯度 $\nabla_\theta\mathbb{E}[r]\approx\nabla\log\pi(a^*)\cdot r(a^*)$ 推 $\pi(a^*))$ 涨，其余降。若无 entropy bonus/KL 平衡，$\pi(a^*)\to1$，塌缩。

### 4.4 KL 约束的间接防护

$\max\mathbb{E}[r]-\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$ 中，$\pi_{\text{ref}}$（SFT）是相对均匀的多 mode 分布，KL 项拉 $\pi_\theta$ 朝 $\pi_{\text{ref}}$（保多 mode）。β 足够大时防塌缩；β 小（[[KL explosion]]）时塌缩自由。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F, copy

class LM(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)

V, D = 50, 32
actor = LM(V, D); ref = copy.deepcopy(actor)
for p in ref.parameters(): p.requires_grad_(False)
opt = torch.optim.Adam(actor.parameters(), 1e-4)

x = torch.randint(0, V, (4, 8))
# 模拟 reward 集中:只给 token 5 高分
def fake_reward(y):
    return (y == 5).float().sum(-1)

entropy_history = []
for step in range(30):
    logits = actor(x)
    # 模拟策略梯度(无 entropy bonus)
    logp = F.log_softmax(logits, -1)
    # 假梯度:推朝 token 5
    target = torch.full_like(logits, 0); target[..., 5] = 1
    loss = F.kl_div(logp, target, reduction='batchmean')
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = F.softmax(actor(x), -1)
        ent = -(p * (p.add(1e-8).log())).sum(-1).mean()   # token 级熵
        entropy_history.append(ent.item())
    print(f"step{step} entropy={ent:.3f}", "← 塌缩!" if ent < 0.5 else "")

print(f"\n熵历史: {[f'{e:.2f}' for e in entropy_history]}")
print("诊断:熵单调降趋 0 → policy collapse")
print("防法:entropy bonus(c_e>0)/KL 约束/advantage 归一/reward 分散")

# entropy bonus 防法演示
ce = 0.01
print(f"加 entropy bonus c_e={ce} 后 loss 增加熵保底项 -{ce}·H,对抗塌缩")
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[RLHF (PPO)]]、[[PPO optimization]]、[[entropy bonus]]（防法核心）、[[KL penalty与KL control]]、[[advantage estimation]]。
- **下游（应用）**: [[mode collapse]]（现象层）、PPO 调参诊断、RLHF 稳定性。
- **对比 / 易混**:
  - **policy collapse vs [[mode collapse]]**：见 §3.3 表。机制 vs 现象。
  - **policy collapse vs [[KL explosion]]**：前者是熵塌缩（概率集中），后者是漂移过大。常连锁（漂移后塌缩）。
  - **policy collapse vs [[reward hacking]]**：reward hacking 可引发 policy collapse（actor 塌缩到 RM 高分 mode），但前者是策略层，后者是 proxy 层。
  - **entropy bonus vs KL**：entropy bonus 保策略熵底（防塌缩），KL 限制漂移（防爆）。两者都间接防 collapse，机制不同。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"policy collapse = mode collapse"**：常混用，但本库区分：机制 vs 现象。
> 2. **"entropy bonus 一定能防"**：LLM action 空间大，entropy bonus 信号弱，$c_e$ 需调，过大反扰 reward。
> 3. **"只看 reward 调参"**：错。reward 涨但熵塌 → collapse。必须看 entropy 曲线。
> 4. **"KL 约束防 collapse"**：间接防（限制漂移后塌缩），非直接。β 小时仍塌。
> 5. **"policy collapse 只在 PPO"**：任何策略优化都可能（REINFORCE/GRPO），PPO 因 clip+多 epoch 更易。
> 6. **"DPO 不会 policy collapse"**：DPO 无在线探索，但 β 过大仍塌缩到偏好 mode（机制略不同）。
> 7. **"生成雷同就是 policy collapse"**：可能是采样温度低，非策略塌缩。需看策略熵区分。
> 8. **"advantage 归一化无关"**：归一化不当放大少数样本优势，加速塌缩。

## 8. 延伸细节

### 8.1 LLM-RL 的 entropy bonus 困境

LLM action 空间是词表（~10^4~10^5），entropy $H\le\log|V|\approx11\sim12$ nats。entropy bonus $-c_e H$ 的梯度信号相对 reward 弱（reward 尺度大），$c_e$ 需大才有效，但大 $c_e$ 扰 reward 学习。故 LLM-RL 常取 $c_e$ 极小或 0，这是 LLM 比 game RL 更易 policy collapse 的原因。替代方案：token 级 entropy 监控 + 手动调 prompt 多样性 + reward 分散化。

### 8.2 policy collapse 的检测

- **token 级 entropy 曲线**：单调降趋 0 → 塌缩；
- **生成多样性**：多个 prompt 生成 response，看独特 n-gram 比例；
- **top-k 概率集中度**：$\pi_\theta$ 的 top-1 概率，趋 1 → 塌缩；
- **重复率**：生成中 n-gram 重复比例。
多信号联合诊断。

### 8.3 与 GAN mode collapse 的类比

GAN 的 mode collapse（生成器塌缩到少数 mode）与 policy collapse 机制相通：
- GAN：判别器 reward 集中 → 生成器塌缩；
- RLHF：RM reward 集中 → actor 塌缩；
- 防法也相通：entropy 正则 / 多样性奖励 / minibatch 判别（GAN）/ prompt 多样（RLHF）。
此类比有助理解。

### 8.4 GRPO 的防塌缩思路

[[GRPO]]（PPO 变体）用组内相对 advantage：同一 prompt 生成 G 条 response，$\hat A_i=(r_i-\text{mean})/\text{std}$。组内相对化天然分散 advantage（不让单一 mode 独占优势），缓解 policy collapse。是 PPO 在 LLM 的改进方向。

### 8.5 policy collapse 的不可逆性

策略一旦塌缩到熵~0，PPO clip 限制 ratio，难扩散回（梯度在塌缩方向饱和）。故 collapse 常需回滚 checkpoint + 调参重启，而非在线恢复。预防 > 治疗。

---
相关: [[LLM特有RL问题]]、[[RLHF (PPO)]]、[[PPO optimization]]、[[entropy bonus]]、[[KL penalty与KL control]]、[[advantage estimation]]、[[mode collapse]]、[[KL explosion]]、[[reward hacking]]、[[GRPO]]
