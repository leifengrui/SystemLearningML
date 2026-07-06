# advantage estimation

> **所属章节**: [[Actor-Critic]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 中等偏难（PG 工程核心）

## 1. 一句话定义

**advantage estimation（优势估计）** 是把理论 advantage $A^\pi(s,a)=Q^\pi-V^\pi$ **用采样数据 + critic $V_\phi$ 算出可用估计 $\hat A_t$** 的方法族——核心包括 MC 估计 $\hat A_t=G_t-V_\phi$、TD(0) 估计 $\hat A_t=\delta_t=r+\gamma V_\phi(s')-V_\phi(s)$、n-step 估计、以及工业标准 **[[GAE]]**（$\lambda$-return 的 advantage 版）。估计得好坏直接决定 [[Actor-Critic]]/[[PPO clipped objective]] 的梯度质量，是 RL 工程最核心的"算权重"环节。

> [!note] 估的是谁
> advantage $A^\pi$ 是期望量（对策略 $\pi$ 与环境 $P$ 的期望），未知。advantage estimation 用一条/一批 rollout + 一个 critic $V_\phi$ 去估它。不同估法在**偏差-方差-算力**三轴上权衡：MC 无偏高方差、TD(0) 有偏低方差、GAE 居中可调。选哪种、参数怎么设，是 PG 调参的主战场。

## 2. 为什么需要它（动机与背景）

[[advantage function]] 说策略梯度用 $A^\pi$ 作权重，但 $A^\pi$ 需要 $Q^\pi,V^\pi$ 都已知——现实未知。要得到可用的 $\hat A_t$：

- 直接用 $G_t$（[[REINFORCE]]）→ 是 $Q^\pi$ 的 MC 估计，没减 $V^\pi$ → 高方差。
- 减 [[baseline]] $V_\phi$ → $\hat A=G_t-V_\phi$（MC advantage）→ 无偏但方差仍大、需等回合。
- 用 TD 误差 $\delta_t$ → 有偏（依赖 $V_\phi$）但低方差、在线。
- 折中 → GAE。

advantage estimation 的本质是"**在偏差、方差、算力、延迟之间找平衡**"：

- 偏差小（无偏）的 MC 方差大、需等回合、样本利用率低；
- 方差小的 TD 偏差大（critic 不准则方向错）；
- GAE 用 $\lambda$ 在二者间连续调节，是工业主流。

LLM-RL 尤其依赖：reward 只在序列末尾、return 单点方差爆，必须靠 GAE 倒推 + critic 减底色才能压到可训。估错 advantage → 梯度方向错 → policy 跑飞（[[KL explosion]]/[[policy collapse]]）。

## 3. 核心概念详解

### 3.1 四种估计法对比

| 方法 | 公式 | 偏差 | 方差 | 算力 | 用途 |
|---|---|---|---|---|---|
| MC | $\hat A_t=G_t-V_\phi(s_t)$ | 无偏 | 高 | 需等回合 | REINFORCE+baseline |
| TD(0) | $\hat A_t=\delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$ | 有偏（依赖 $V_\phi$） | 低 | 在线 | A2C 简版 |
| n-step | $\hat A_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V_\phi(s_{t+n})-V_\phi(s_t)$ | 中 | 中 | 等n步 | n-step Q |
| **GAE(λ)** | $\hat A_t=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}$ | 可调 | 可调 | 一次轨迹 | **PPO/A3C 工业标配** |

### 3.2 MC advantage

$\hat A_t=G_t-V_\phi(s_t)$，其中 $G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}$ 是实际 return。无偏（$\mathbb{E}[G_t\mid s,a]=Q^\pi$），但单条轨迹 $G_t$ 噪声大、且需等回合结束才能算。适合短回合或无 critic 时。

### 3.3 TD(0) advantage

$\hat A_t=\delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$，即 TD 误差。$\mathbb{E}[\delta_t\mid s,a]=A^\pi$（$V_\phi=V^\pi$ 时），故无偏（理论）但实践中 $V_\phi\ne V^\pi$ 引入偏差。优点：低方差、在线（不需等回合）。缺点：critic 不准则方向错。

### 3.4 n-step advantage

$\hat A_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V_\phi(s_{t+n})-V_\phi(s_t)$，前 n 步用实际 reward、之后用 $V_\phi$ 收尾。n 大→偏 MC、n 小→偏 TD。需等 n 步才能算。

### 3.5 GAE(λ)

$$
\hat A_t^{GAE(\lambda)}=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}
$$

- $\lambda=0$：$\hat A=\delta_t$（TD(0)）；
- $\lambda=1$：$\hat A=G_t-V_\phi$（MC）；
- $0<\lambda<1$：n-step 的指数加权平均。

GAE 在偏差与方差间连续插值，一次轨迹前向即可算（从后往前递推），是 [[GAE]] 专篇的主题、PPO/A3C 的标配。

### 3.6 critic 的训练 target 与 advantage 的关系

- critic target：$V_\phi$ 回归的 $y_t$，常取 $G_t$（MC）或 $r+\gamma V_\phi(s')$（TD）或 GAE-based return $V_\phi(s_t)+\hat A_t$。
- advantage：给 actor 的权重 $\hat A_t$。
- 常用技巧：$\hat A_t$ 算完后，critic target 取 $V_\phi(s_t)+\hat A_t$（"advantage + baseline" 形式），数值稳定。

### 3.7 advantage 归一化

$\hat A_t$ 算出后常做 batch 归一化 `(A-A.mean())/(A.std()+1e-8)`，进一步降方差稳尺度。detach mean/std 避免耦合。见 [[variance reduction]]。

### 3.8 LLM-RL 的特殊处理

RLHF reward 仅序列末尾一次（$r_{T}=r_\phi$，其余 0）。用 GAE 倒推：$\delta_t=\gamma V(s_{t+1})-V(s_t)$（中间步 r=0）+$\delta_{T-1}=r_\phi+\gamma V(s_T)-V(s_{T-1})$。return 单点被 GAE 的指数衰减分到每个 token，critic 减底色。$\gamma$ 常取 1（episodic），$\lambda$ 取 0.95~1.0。见 [[GAE]] §8。

## 4. 数学原理 / 公式

### 4.1 $\delta_t$ 是 $A^\pi$ 的无偏估计（$V_\phi=V^\pi$ 时）

$$
\mathbb{E}_{s_{t+1}\sim P}[\delta_t]=R(s_t,a_t)+\gamma\mathbb{E}[V^\pi(s_{t+1})]-V^\pi(s_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)=A^\pi(s_t,a_t)
$$

（用 [[policy与value function]] §3.4 的 $Q^\pi=R+\gamma\mathbb{E}[V^\pi]$）。故 TD 误差理论无偏，偏差全来自 $V_\phi\ne V^\pi$。

### 4.2 GAE 的等价形式

$$
\hat A_t^{GAE(\lambda)}=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)}
$$

即 GAE 是 n-step advantage 的 $\lambda$-加权平均。证明见 [[GAE]] §4。

### 4.3 GAE return target

$$
\hat G_t = V_\phi(s_t)+\hat A_t^{GAE(\lambda)}
$$

作为 critic 回归目标，比裸 $G_t$ 或裸 TD 更稳（含 critic 的先验 + advantage 修正）。

### 4.4 偏差-方差的 $\lambda$ 调节

$\lambda\uparrow$：向 MC 靠，偏差↓方差↑；$\lambda\downarrow$：向 TD 靠，偏差↑方差↓。最优 $\lambda$ 取决于 critic 准度：critic 越准可用越小 $\lambda$（信 critic），越不准要用大 $\lambda$（信实际 return）。

## 5. 代码示例

```python
import torch, torch.nn as nn

class AC(nn.Module):
    def __init__(s,o,n): super().__init__(); s.a=nn.Sequential(nn.Linear(o,64),nn.Tanh(),nn.Linear(64,n)); s.c=nn.Sequential(nn.Linear(o,64),nn.Tanh(),nn.Linear(64,1))
    def pi(s,x): return torch.distributions.Categorical(logits=s.a(x))
    def v(s,x): return s.c(x).squeeze(-1)

ac=AC(4,2); opt=torch.optim.Adam(ac.parameters(),3e-3); gamma=0.99; lam=0.95

def collect(n=128):
    buf=[]; s=torch.randn(4)
    for _ in range(n):
        d=ac.pi(s); a=d.sample(); sn=torch.randn(4); r=float(a==1)
        buf.append((s,a,r,sn)); s=sn
    return buf

buf=collect()
states=torch.stack([t[0] for t in buf]); nexts=torch.stack([t[3] for t in buf])
rs=torch.tensor([t[2] for t in buf],dtype=torch.float32)
with torch.no_grad():
    Vs=ac.v(states); Vn=ac.v(nexts)
    deltas=rs+gamma*Vn-Vs
    # GAE: 从后往前递推
    A=torch.zeros_like(deltas)
    A[-1]=deltas[-1]
    for t in reversed(range(len(buf)-1)):
        A[t]=deltas[t]+gamma*lam*A[t+1]
    Gt = Vs + A   # critic target
# advantage 归一化
A_norm = (A - A.mean())/(A.std()+1e-8)
logps=torch.stack([ac.pi(t[0]).log_prob(t[1]) for t in buf])
v=ac.v(states)
actor_loss=-(logps*A_norm).mean()
critic_loss=((v-Gt.detach())**2).mean()
loss=actor_loss+0.5*critic_loss
opt.zero_grad(); loss.backward(); opt.step()
print("A_std",A.std().item(),"Gt_mean",Gt.mean().item())
```

> [!tip] 选择估计法的经验
> - 回合短、无 critic：MC。
> - 在线、critic 快收敛：TD(0)。
> - 通用、PPO/A3C：GAE(λ=0.95)。
> - LLM-RL：GAE(λ=0.95~1.0, γ=1)，reward 末尾倒推。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[advantage function]]（要估的目标）、[[value network]]（critic 提供减项）、[[return与reward]]（$G_t$）、[[baseline]]（$G-V$）。
- **下游（应用）**: [[Actor-Critic]]（advantage 给 actor）、[[PPO clipped objective]]（GAE 是 PPO 标配）、[[GAE]]（主力方法专篇）、[[variance reduction]]（estimation 即在调偏差-方差）。
- **对比 / 易混**:
  - **advantage estimation vs advantage function**：前者是估计方法，后者是定义量。
  - **MC vs TD vs GAE**：见 §3.1 表，偏差-方差轴。
  - **critic target vs advantage**：target 回归 $V$，advantage 给 actor；同轨迹同时算两份。
  - **advantage 归一化 vs reward 归一化**：前者控梯度权重，后者控 reward 尺度。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **$\delta_t$ 没 detach critic**：advantage 给 actor 时若 $V_\phi$ 梯度混入，actor 去最小化 $V$。detach。
> 2. **GAE 递推方向错**：必须从后往前（$A_t$ 依赖 $A_{t+1}$），正推会漏项。
> 3. **critic target 用 advantage 不加 $V$**：`target = A` 会丢 baseline，critic 学偏；用 `target = V + A` 或 `G_t`。
> 4. **λ=0 直接用 TD**：critic 未收敛时偏差大；先用大 λ 过渡或 MC。
> 5. **reward 末尾 reward 没正确倒推**：LLM-RL 中中间步 r=0，GAE 靠 $\gamma V$ 传递；若 $\gamma=0$ 则 advantage 全 0。
> 6. **advantage 不归一化**：量级漂移，学习率难调；batch 归一化。
> 7. **off-policy 用旧 advantage**：策略变了 advantage 变，需重算或 importance reweight。
> 8. **n-step 等待未处理**：n-step 需等 n 步，在线算法要 cache buffer。
> 9. **critic target 与 advantage 用同一 $V$ 不分离**：critic 用 $V_\phi(s_t)$ 当 target 会自举；用 $V_\phi(s_t)+A$ 或独立 target network。
> 10. **GAE 的 $\gamma$ 与 reward 末尾不匹配**：episodic 可 $\gamma=1$；continuing 必须 $<1$。

## 8. 延伸细节

### 8.1 GAE 的物理直觉

GAE 把"未来每步的 TD 误差"按 $\gamma\lambda$ 衰减累加 —— 远处的 $\delta$ 权重小（因 critic 已含其期望，信它即可），近处 $\delta$ 权重大（信实际）。$\lambda$ 控制"信 critic 多远"。

### 8.2 V-trace（IMPALA）

off-policy 版 GAE，用截断 importance ratio 修正旧策略样本， IMPALA 用。LLM-RL 少用（on-policy 为主），但概念相关。

### 8.3 reward 末尾的 advantage 倒推（LLM-RL 详解）

设 reward 只在 $T$ 处给 $r$。GAE：$\delta_{T-1}=r+\gamma V(s_T)-V(s_{T-1})$（$s_T$ 为终止态 $V=0$），$\delta_{t<T-1}=\gamma V(s_{t+1})-V(s_t)$。$A_t=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}$ 把末尾 $r$ 按指数衰减分到每个 token。$\gamma=1$ 时衰减纯靠 $\lambda$。这是 RLHF 信用分配的核心。

### 8.4 critic 准度的自检

观察 critic loss 与 advantage 方差比：critic loss 大、advantage 方差相对小 → critic 不准但 advantage 用得多 → 偏差风险；critic loss 小 → advantage 可信。可动态调 $\lambda$（critic 准则降 λ 信它）。

### 8.5 与 reward normalization 的协同

reward 归一化让 $V$ 尺度稳定，advantage 估计更稳；advantage 归一化再压一层。两者与 GAE 一起构成 PPO 的"三件套稳方差"。

---
相关: [[Actor-Critic]]、[[GAE]]、[[value network]]、[[advantage function]]、[[baseline]]、[[variance reduction]]、[[PPO clipped objective]]、[[KL penalty与KL control]]、[[entropy bonus]]、[[return与reward]]、[[reward normalization]]
