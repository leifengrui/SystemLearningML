# variance reduction

> **所属章节**: [[Policy Gradient体系]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 中等偏难（决定 PG 能否训练）

## 1. 一句话定义

**variance reduction（方差缩减）** 是策略梯度方法里**一切让梯度估计方差变小、从而能用更大学习率更快收敛的技术统称**——核心包括减 [[baseline]]、用 [[advantage function]] 替代 return、[[GAE]] 的偏差-方差插值、advantage/reward 归一化、reward clipping、并行多环境采样、entropy 正则等。策略梯度原始形式（[[REINFORCE]]）方差极大，几乎不可训；现代 PG（[[Actor-Critic]]/[[PPO clipped objective]]）可用，正是靠一整套方差缩减叠加。

> [!note] 为什么方差是头号敌人
> 策略梯度是**蒙特卡洛估计**：$\hat g=\frac1N\sum\nabla\log\pi\cdot A$。若 $\text{Var}[\hat g]$ 大，则 $\hat g$ 与真值 $\nabla J$ 偏差大，梯度方向噪声大 → 学习率必须极小才不发散 → 收敛慢、易震荡、易陷局部最优。方差缩减 = 把噪声压下去 = 学习率能放大 = 训得动。方差 vs 偏差是 trade-off：很多技巧用小偏差换大方差下降（如 TD/GAE）。

## 2. 为什么需要它（动机与背景）

[[REINFORCE]] 证明策略梯度可解，但实测发现：

- 一条轨迹的 $G_t$ 动辄差几个数量级（赢 +100 vs 输 -10）；
- 用它作权重，梯度 $\hat g$ 噪声爆表；
- 学习率 $\eta>10^{-3}$ 就发散，$\eta\le10^{-4}$ 才稳但慢到不可用；
- 不同 seed 结果差异巨大。

根因：**蒙特卡洛方差**。每条轨迹是高维随机过程，return 累积了所有随机性。要让 PG 实用，必须系统降噪。于是发展出一系列手段，从最朴素的"减均值"到"用 critic 估条件期望"到"偏差-方差插值"，构成现代 PG 的工程内核。LLM-RL（PPO-RLHF）尤其依赖方差缩减：reward 只在序列末尾、return 单点、方差本就巨大，不靠 GAE + critic + KL 几乎训不动。

## 3. 核心概念详解

### 3.1 方差来源拆解

策略梯度估计 $\hat g=\frac1N\sum_\tau\sum_t\nabla\log\pi\,c_t$（$c_t$ 是 $G_t$/$A_t$ 等权重）的方差来自：

1. **轨迹间方差**：不同轨迹 return 差异大（环境随机 + 策略随机）。
2. **时间间方差**：一条轨迹内不同 $t$ 的 $c_t$ 量级差。
3. **状态相关方差**：状态本身好坏（baseline 可减）。
4. **动作采样方差**：$\pi$ 采样的 $a_t$ 不同。
5. **小批量方差**：batch 太小估不准。

每种方差有对应缩减手段。

### 3.2 方差缩减技术清单

| 技术 | 原理 | 效果 | 代价 |
|---|---|---|---|
| **[[baseline]] $V^\pi(s)$** | 从 $G$ 减状态底色 | 大降、无偏 | 需学 critic |
| **[[advantage function]] $A=Q-V$** | 用相对量代绝对 | 同上 | 同上 |
| **[[GAE]] ($\lambda$)** | MC 与 TD 插值 | 调偏差-方差 | 引入小偏差 |
| **advantage 归一化** | batch 内标准化 | 降尺度方差 | 微小偏差 |
| **reward 归一化/clipping** | 控 reward 尺度 | 防梯度爆炸 | 改 reward 语义（需谨慎） |
| **多环境并行** | 增大 N | $\propto1/\sqrt N$ | 算力 |
| **entropy 正则** | 防策略过早确定 | 稳探索 | 微小偏差 |
| **ratio clipping（PPO）** | 限 off-policy 偏差 | 防发散 | 限步长 |
| **状态表征标准化** | 输入归一化 | 降输入方差 | - |
| **梯度裁剪** | 限梯度范数 | 防爆炸 | 截断真梯度 |

### 3.3 偏差-方差权衡（bias-variance tradeoff）

- **无偏高方差**：MC（$G_t$）。
- **有偏低方差**：TD（依赖 $V_\phi$ 估准，未准则有偏差）。
- **插值**：GAE 的 $\lambda$ 在二者间连续过渡。

PG 实践常接受小偏差换大方差下降（TD/GAE），因为方差是收敛瓶颈。

### 3.4 baseline 与归一化的叠加

合法 baseline 不含 $a_t$ 即无偏，多个独立 baseline 可叠加：$V_\phi$（状态）+ batch mean（全局）+ reward running std。三者正交，叠加降方差。注意 reward 归一化若改变 reward 的相对尺度可能改最优策略（如非单调变换），用 std 归一化（线性变换）安全。

### 3.5 GAE 的核心地位

$$
\hat A_t^{GAE(\lambda)}=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}
$$

$\lambda\in[0,1]$：0=纯 TD（低方差高偏差）、1=纯 MC（高方差低偏差）。LLM 常用 0.95~1.0。GAE 是 [[advantage estimation]] 的工业标准，见 [[GAE]]。

### 3.6 ratio clipping 的方差控制

off-policy 用旧样本训当前 $\pi$，importance ratio $\rho=\pi_\theta/\pi_{old}$ 方差随 $\pi$ 漂移爆炸。PPO 的 `clip(ρ,1±ε)` 把极端 ratio 截断，虽引入偏差但防发散，是 [[PPO clipped objective]] 能多 epoch 复用样本的关键。

## 4. 数学原理 / 公式

### 4.1 蒙特卡洛估计的方差

$\hat g=\frac1N\sum_i X_i$，$X_i=\sum_t\nabla\log\pi\,c_t$ 独立同分布。$\text{Var}[\hat g]=\text{Var}[X]/N$。降方差三条路：降 $\text{Var}[X]$（baseline/advantage/GAE）、增大 $N$（并行/多采）、减小 $X$ 的尺度（归一化）。

### 4.2 baseline 降方差的机制

$\text{Var}[\nabla\log\pi\,(G-b)]=\mathbb{E}[(\nabla\log\pi)^2(G-b)^2]-\mathbb{E}[\nabla\log\pi(G-b)]^2$。期望项不变（[[baseline]] 无偏证明），第一项在 $b=V^\pi$ 时最小化（$G-V$ 是残差）。方差下降量 $\approx\text{Var}[V]$（状态底色部分被扣掉）。

### 4.3 GAE 的偏差-方差

$\hat A^{GAE(\lambda)}$ 是 $\lambda$-return 的 advantage，其偏差来自 $V_\phi$ 估不准，方差来自 $\gamma\lambda$ 的折扣长度。$\lambda\to0$ 偏差大（依赖 $V_\phi$ 的当前值）、方差小（只看一步）；$\lambda\to1$ 偏差小（回到 MC）、方差大。数学上 GAE 是 n-step advantage 的指数加权平均。

### 4.4 归一化的无偏性

`A' = (A - mean)/std` 在 batch 内是**仿射变换**。若 mean/std 视为常数（detach），则 $\mathbb{E}[\nabla\log\pi\,A']=\mathbb{E}[\nabla\log\pi\,A]/\text{std}$（只改学习率尺度），方向不变；若不 detach 会有微小耦合。实践中 detach 处理，近似无偏。

## 5. 代码示例

```python
import torch, torch.nn as nn
from collections import deque

class AC(nn.Module):
    def __init__(s, obs, n): super().__init__(); s.a=nn.Sequential(nn.Linear(obs,64),nn.Tanh(),nn.Linear(64,n)); s.c=nn.Sequential(nn.Linear(obs,64),nn.Tanh(),nn.Linear(64,1))
    def pi(s,x): return torch.distributions.Categorical(logits=s.a(x))
    def v(s,x): return s.c(x).squeeze(-1)

ac=AC(4,2); opt=torch.optim.Adam(ac.parameters(),3e-3); gamma=0.99; lam=0.95
rew_std = deque([1.0], maxlen=100)   # reward running std

def collect(n=256):
    buf=[]
    s=torch.randn(4)
    for _ in range(n):
        d=ac.pi(s); a=d.sample(); sn=torch.randn(4); r=float(a==1)*2-1
        buf.append((s,a,r,sn,False)); s=sn
    return buf

buf=collect()
# 1) 算 GAE advantage
with torch.no_grad():
    Vs=torch.stack([ac.v(t[0]) for t in buf]); Vn=torch.stack([ac.v(t[3]) for t in buf])
    rs=torch.tensor([t[2] for t in buf],dtype=torch.float32)
    dones=torch.tensor([t[4] for t in buf],dtype=torch.float32)
    # reward 归一化（running std）
    rs = rs/(list(rew_std)[-1]+1e-8); rew_std.append(rs.std().item()+1e-8)
    deltas = rs + gamma*Vn*(1-dones) - Vs
    A=torch.zeros_like(deltas); Gt=torch.zeros_like(deltas)
    A[-1]=deltas[-1]; Gt[-1]=deltas[-1]+Vs[-1]
    for t in reversed(range(len(buf)-1)):
        A[t]=deltas[t]+gamma*lam*A[t+1]*(1-dones[t])
        Gt[t]=A[t]+Vs[t]   # return target for critic
# 2) advantage 归一化
A_norm = (A - A.mean())/(A.std()+1e-8)
logps=torch.stack([ac.pi(t[0]).log_prob(t[1]) for t in buf])
actor_loss=-(logps*A_norm).mean()
critic_loss=((ac.v(torch.stack([t[0] for t in buf]))-Gt)**2).mean()
entropy=ac.pi(torch.stack([t[0] for t in buf])).entropy().mean()
loss=actor_loss+0.5*critic_loss-0.01*entropy
opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ac.parameters(),0.5); opt.step()
print("adv std",A.std().item(),"actor",float(actor_loss),"critic",float(critic_loss))
```

> [!tip] 方差缩减叠加清单
> 上面同时用了：GAE(λ) + reward 归一化 + advantage 归一化 + entropy 正则 + 梯度裁剪。这就是工业 PG 的标准配置，每一项都在压方差/稳训练。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[REINFORCE]]（待减方差的原型）、[[baseline]]、[[advantage function]]、[[return与reward]]、概率论（蒙特卡洛方差、bias-variance）。
- **下游（应用）**: [[Actor-Critic]]（critic=baseline 的载体）、[[GAE]]（主力 advantage 估计）、[[advantage estimation]]、[[PPO clipped objective]]（ratio clipping 防发散）、[[reward normalization]]、[[梯度裁剪]]（限梯度范数防爆炸）、[[entropy regularization]]（稳探索）。
- **对比 / 易混**:
  - **方差缩减 vs 偏差引入**：baseline 无偏、GAE/归一化有小偏差；权衡取舍。
  - **reward 归一化 vs advantage 归一化**：前者控 reward 尺度，后者控梯度权重尺度，互补。
  - **方差缩减 vs 探索**：降方差别降过头成确定性策略，需 entropy 正则平衡。
  - **方差缩减 vs off-policy 偏差**：ratio clipping 是"用偏差换稳定性"的典型，与 baseline 的"无偏差换方差下降"不同。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **裸 REINFORCE 不加任何方差缩减** → 几乎不可训，学习率必须极小。至少 batch 归一化 + baseline。
> 2. **把 reward 归一化当万能**：非线性归一化（如 tanh/clamp 极端值）会改最优策略；用 std 线性归一化安全。
> 3. **advantage 归一化不 detach mean/std**：会有耦合梯度；detach 处理。
> 4. **GAE λ 调到 0** → 纯 TD 偏差大，critic 没学好时梯度方向错；LLM 用 0.95+。
> 5. **梯度裁剪阈值太小**：截断真梯度方向，学不动；0.5~1.0 是经验值，太小有害。
> 6. **多环境并行但 batch 仍小**：并行增大 N 才有用，batch 内归一化才有统计意义。
> 7. **过度降方差到策略熵塌缩**：方差压太狠 + entropy bonus 不够 → 策略变确定性 → 陷局部最优。
> 8. **reward 归一化与 reward shaping 混**：归一化是线性不改最优，shaping 可能改最优（见 [[reward hacking]]）。
> 9. **LLM-RL 不用 critic 裸跑**：return 单点方差爆，必上 critic + GAE。
> 10. **以为方差缩减越多越好**：每项都有代价（偏差/算力/超参），叠加需验证每项收益。

## 8. 延伸细节

### 8.1 方差缩减与样本效率

降方差 = 同样 N 下估更准 = 可用更大学习率 = 样本效率高。这是 on-policy PG（样本利用率本低）能实用的关键。off-policy 靠重加权复用旧样本，但重加权本身增方差，需 ratio clipping 兜底。

### 8.2 pop-art 与 reward 归一化

pop-art（van Hasselt 2016）对 value target 做自适应归一化并对应反归一化输出，稳定大规模 off-policy 训练。RLHF 里 reward model 输出尺度可能漂移，running std 归一化是简化版。

### 8.3 entropy 正则的双重作用

entropy bonus $+\beta H(\pi)$ 既鼓励探索（防确定性塌缩）也起到平滑策略、间接降梯度方差（策略分布更平 → $\nabla\log\pi$ 数值更稳）。$\beta$ 太大则策略过于随机不收敛，太小则方差/探索不足。LLM-RL 中 $\beta$ 常由 KL 约束隐式承担（见 [[entropy regularization]]/[[KL penalty与KL control]]）。

### 8.4 LLM-RL 的方差挑战

RLHF reward 只在序列末尾、单条轨迹 reward 单点 → return 方差天然极大。靠 GAE 把单点 reward 倒推到每个 token、critic 减状态底色、KL 约束限制漂移，三者合力才把方差压到可训。这也是 RLHF 比 CV/Atari RL 更"娇气"的根因。

### 8.5 并行采样的方差收益

N 个并行环境采的样本 i.i.d（近似），方差 $\propto1/\sqrt N$。A3C/A2C/PPO 的多 env 并行既是吞吐也是方差缩减。RLHF 的 rollout worker 群（见 [[rollout worker]]/[[trajectory generation]]）同理。

---
相关: [[Policy Gradient体系]]、[[REINFORCE]]、[[baseline]]、[[advantage function]]、[[GAE]]、[[Actor-Critic]]、[[PPO clipped objective]]、[[reward normalization]]、[[梯度裁剪]]、[[entropy regularization]]
