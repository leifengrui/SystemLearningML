# GAE

> **所属章节**: [[PPO]]
> **所属模块**: [[04-强化学习基础]]
> **难度**: 中等偏难（PG 信用分配的工业标准）

## 1. 一句话定义

**GAE（Generalized Advantage Estimation，广义优势估计）** 是 [[advantage estimation]] 的主力方法，用一个参数 $\lambda\in[0,1]$ 在 **MC advantage**（$\hat A=G_t-V$，无偏高方差）与 **TD(0) advantage**（$\hat A=\delta_t$，有偏低方差）之间做**指数加权插值**：

$$
\hat A_t^{GAE(\lambda)}=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k},\qquad \delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

- $\lambda=0$ → $\hat A_t=\delta_t$（TD(0)）；
- $\lambda=1$ → $\hat A_t=G_t-V_\phi$（MC）；
- $0<\lambda<1$ → n-step advantage 的 $\lambda$-加权平均（偏差-方差可调）。

GAE 是 [[PPO clipped objective]]、A3C、IMPALA 等现代策略梯度算法的**标配 advantage 估计法**——它把"未来每一步的 TD 误差"按 $\gamma\lambda$ 衰减累加，用 critic $V_\phi$ 当"远处已知项"、用实际 reward 当"近处修正项"，在偏差与方差间取得工程最佳折中。由 Schulman 2016（High-Dimensional Continuous Control Using GAE）提出。

> [!note] 一句话直觉
> GAE 在问"这个动作对未来有多大好处"。近处的 reward 我信得过（实际发生的），远处的我信 critic（$V_\phi$ 已含其期望）。$\gamma\lambda$ 控制"信 critic 的距离"——$\lambda$ 小：只信一步实际，剩下全靠 critic（低方差高偏差）；$\lambda$ 大：信很多步实际（高方差低偏差）。$\gamma$ 是 discount（[[return与reward]]），$\lambda$ 是 GAE 自己的"信用分配衰减"——两者乘积 $\gamma\lambda$ 才是真正的衰减率。

## 2. 为什么需要它（动机与背景）

[[advantage function]] 说策略梯度用 $A^\pi=Q^\pi-V^\pi$ 作权重。但 $A^\pi$ 未知，要估（[[advantage estimation]]）。两种朴素估法各有硬伤：

| 估法 | 形式 | 偏差 | 方差 | 问题 |
|---|---|---|---|---|
| MC | $\hat A=G_t-V_\phi$ | 无偏 | **高** | 单条轨迹 $G_t$ 噪声大、需等回合结束、样本利用率低 |
| TD(0) | $\hat A=\delta_t$ | **有偏**（依赖 $V_\phi$） | 低 | critic 不准时方向错、只看一步太短视 |

实际想要的：**用几步实际 reward + critic 收尾**（n-step），且能**连续调节 n 的远近**。固定 n-step（如 n=5）只选一个点；GAE 的洞见是——**对所有 n 的 n-step advantage 做 $\lambda$ 加权平均**，得到一条从 TD 到 MC 的**连续谱**，一个参数 $\lambda$ 即可调。这既避免 MC 方差爆，又避免 TD(0) 偏差大，是 PPO/A3C 的工业选择。

LLM-RL 尤其依赖 GAE：reward 只在序列末尾给一次（[[return与reward]] §8.1），return 退化为单点，方差爆炸。必须靠 GAE 的指数衰减把末尾 reward 倒推到每个 token，并用 critic 减底色，才能压到可训。

## 3. 核心概念详解

### 3.1 TD 误差 $\delta_t$（GAE 的构造块）

$$
\delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

- 它是 **TD(0) 的 advantage 估计**（[[advantage estimation]] §3.3）；
- 理论上 $\mathbb{E}[\delta_t|s_t,a_t]=A^\pi(s_t,a_t)$（$V_\phi=V^\pi$ 时无偏，见 §4.1）；
- 实际 $V_\phi\ne V^\pi$ 引入偏差，但单步方差小；
- $\delta_t$ 把"这步实际 reward + critic 对下一步的预测 - critic 对这步的预测"汇成一个标量，是"单步惊喜"。

### 3.2 GAE 的定义

$$
\boxed{\hat A_t^{GAE(\lambda)}=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}=\delta_t+\gamma\lambda\,\delta_{t+1}+(\gamma\lambda)^2\,\delta_{t+2}+\cdots}
$$

- 衰减率是 $\gamma\lambda$（不是 $\gamma$！），把未来每步的 $\delta$ 加权累加；
- 系数 $(\gamma\lambda)^k$ 当 $k\to\infty$ 衰减（需 $\gamma\lambda<1$）；
- $\lambda$ 是"信实际 reward 的距离"：大→看远、小→看近。

### 3.3 两端点的退化

| $\lambda$ | GAE | 等价于 |
|---|---|---|
| $\lambda=0$ | $\hat A_t=\delta_t$ | TD(0) advantage |
| $\lambda=1$ | $\hat A_t=\sum_k\gamma^k\delta_{t+k}=G_t-V_\phi(s_t)$ | MC advantage |

证明 $\lambda=1$ 退化为 MC：$\sum_{k\ge0}\gamma^k\delta_{t+k}$ 是**望远镜和**（telescoping），中间的 $V_\phi$ 项两两抵消，只剩 $G_t-V_\phi(s_t)$（§4.2）。这是 GAE 的关键性质——它**严格包含 MC 与 TD 为特例**。

### 3.4 从后往前递推（工程实现）

GAE 不是按定义 $O(n^2)$ 算，而是 $O(n)$ 递推：

$$
\hat A_t=\delta_t+\gamma\lambda\,\hat A_{t+1}
$$

从轨迹末尾往前推，$\hat A_{T}=0$（或 $\delta_T$，终止态 $V=0$）。证明见 §4.3。这是所有 PPO 实现的标准写法（[[PPO clipped objective]] §5 代码）。

### 3.5 GAE return target（critic 的回归目标）

$$
\hat G_t=V_\phi(s_t)+\hat A_t^{GAE(\lambda)}
$$

作为 critic $V_\phi$ 的回归目标，比裸 $G_t$（MC，高方差）或裸 $r+\gamma V$（TD，高偏差）更稳：它含 critic 的先验 $V_\phi$ + advantage 的修正。这是 [[advantage estimation]] §3.6 的"advantage + baseline"形式，PPO 标配。

### 3.6 与 n-step advantage 的等价

n-step advantage：$\hat A_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V_\phi(s_{t+n})-V_\phi(s_t)$。GAE 是它的 $\lambda$-加权平均：

$$
\hat A_t^{GAE(\lambda)}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\,\hat A_t^{(n)}
$$

（§4.4 证明）。故 GAE = "所有 n-step 的几何加权"，比固定 n 自适应：critic 准时小 $\lambda$（信 critic，n 小），不准时大 $\lambda$（信实际，n 大）。

### 3.7 $\gamma$ 与 $\lambda$ 的分工（易混点）

| 参数 | 角色 | 调大 | 典型值 |
|---|---|---|---|
| $\gamma$ | **折扣**（[[return与reward]]，时间偏好/收敛） | 远视、有效视野 $1/(1-\gamma)$ 增 | 0.99 |
| $\lambda$ | **GAE 偏差-方差**（信 critic 的距离） | 偏 MC、方差↑偏差↓ | 0.95 |

两者**独立**：$\gamma$ 是 MDP 的属性（定义 return），$\lambda$ 是估计器的属性（定义 advantage 估计）。GAE 的实际衰减率是**乘积 $\gamma\lambda$**，但 $\gamma$ 改变 return 本身、$\lambda$ 只改估计法不改变 RL 目标。混用会让 advantage 尺度与 target 错配。

### 3.8 LLM-RL 的 GAE 倒推（核心应用）

RLHF 中 reward 仅序列末尾给一次 $r_\phi$（其余步 $r=0$），$\gamma$ 常取 1（episodic 固定长度）。GAE 简化为：

$$
\delta_{T-1}=r_\phi+\gamma V(s_T)-V(s_{T-1}),\quad \delta_{t<T-1}=\gamma V(s_{t+1})-V(s_t)
$$

（$s_T$ 终止态 $V=0$）。递推 $\hat A_t=\delta_t+\lambda\hat A_{t+1}$ 把末尾 $r_\phi$ 按 $\lambda^k$ 衰减分到每个 token——靠 $\lambda$ 完成信用分配（$\gamma=1$ 时衰减纯靠 $\lambda$）。$\lambda$ 常取 0.95~1.0：太小（如 0.9）衰减太快，远端 token 拿不到信号；太大（=1）退化为所有 token 同分（MC），方差爆。critic 减底色 $V_\phi$ 后，advantage 才有正负区分（哪些 token 比平均好/差）。

## 4. 数学原理 / 公式

### 4.1 $\delta_t$ 是 $A^\pi$ 的无偏估计（$V_\phi=V^\pi$ 时）

$$
\mathbb{E}_{s_{t+1}\sim P}[\delta_t]=R(s_t,a_t)+\gamma\mathbb{E}[V^\pi(s_{t+1})]-V^\pi(s_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)=A^\pi(s_t,a_t)
$$

（用 [[policy与value function]] §3.4 的 $Q^\pi=R+\gamma\mathbb{E}[V^\pi]$）。故 TD 误差理论无偏，偏差全来自 $V_\phi\ne V^\pi$。这是用 $\delta_t$ 构造 GAE 的合理性根基。

### 4.2 $\lambda=1$ 退化为 MC（望远镜和证明）

$$
\sum_{k=0}^{T-t-1}\gamma^k\delta_{t+k}=\sum_{k=0}^{T-t-1}\gamma^k\bigl(r_{t+k+1}+\gamma V(s_{t+k+1})-V(s_{t+k})\bigr)
$$

展开后 $V$ 项交错出现 $\gamma^k\gamma V(s_{t+k+1})=\gamma^{k+1}V(s_{t+k+1})$ 与 $-\gamma^k V(s_{t+k})$，相邻两项的 $V(s_{t+k+1})$ 抵消，最终只剩 $V$ 在首尾：

$$
=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}+\gamma^{T-t}V(s_T)-V(s_t)=G_t-V(s_t)\quad(V(s_T)=0)
$$

证毕。$\lambda=1$ 时 GAE = MC advantage。

### 4.3 后向递推 $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$

从定义展开：$\hat A_t=\delta_t+\gamma\lambda\,\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots=\delta_t+\gamma\lambda\bigl(\delta_{t+1}+\gamma\lambda\,\delta_{t+2}+\cdots\bigr)=\delta_t+\gamma\lambda\,\hat A_{t+1}$。故 $O(n)$ 递推成立，$\hat A_T=0$ 起步。

### 4.4 GAE = n-step 的 $\lambda$-加权平均

记 $\hat A_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V(s_{t+n})-V(s_t)$。需证：

$$
\hat A_t^{GAE(\lambda)}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)}
$$

**思路**：把 GAE 的 $\delta$ 展开，按"未来第 $n$ 步的 $V$ 出现几次"重组。每个 $V(s_{t+n})$ 在 GAE 中以 $(\gamma\lambda)^{n-1}\cdot\gamma\cdot[\text{系数}]$ 出现，恰好匹配 n-step 的权重 $\lambda^{n-1}$。完整证明见 Schulman 2016 附录。直观：n-step 是"前 n 步实际 + 第 n 步 critic"，GAE 把所有 n 的这种组合按 $\lambda^{n-1}$ 加权，$(1-\lambda)$ 是归一化（$\sum\lambda^{n-1}=1/(1-\lambda)$）。这把"固定 n"升级为"几何加权 n"。

### 4.5 偏差-方差随 $\lambda$

$\lambda\uparrow$：向 MC 靠，偏差↓方差↑；$\lambda\downarrow$：向 TD 靠，偏差↑方差↓。最优 $\lambda$ 取决于 critic 准度：critic 越准可用越小 $\lambda$（信 critic），越不准要用大 $\lambda$（信实际 return）。$\lambda=0.95$ 是经验甜点（critic 中等准时兼顾）。

### 4.6 GAE 的期望与方差

$\mathbb{E}[\hat A_t^{GAE}]\to A^\pi$（$V_\phi\to V^\pi$ 时无偏）。方差：$\text{Var}(\hat A^{GAE})\approx\text{Var}(\text{MC})\cdot\frac{1-\gamma\lambda}{1+\gamma\lambda}$（粗略），随 $\lambda$ 增大而增大。GAE 通过 $\lambda$ 在 $[\text{Var(TD)},\text{Var(MC)}]$ 间插值。

## 5. 代码示例

```python
import torch, torch.nn as nn

class AC(nn.Module):
    def __init__(s,o,n): super().__init__(); s.a=nn.Sequential(nn.Linear(o,64),nn.Tanh(),nn.Linear(64,n)); s.c=nn.Sequential(nn.Linear(o,64),nn.Tanh(),nn.Linear(64,1))
    def pi(s,x): return torch.distributions.Categorical(logits=s.a(x))
    def v(s,x): return s.c(x).squeeze(-1)

ac=AC(4,2); opt=torch.optim.Adam(ac.parameters(),3e-3); gamma=0.99; lam=0.95

def collect(n=200):
    buf=[]; s=torch.randn(4)
    for _ in range(n):
        d=ac.pi(s); a=d.sample(); sn=torch.randn(4); r=float(a==1)
        buf.append((s,a,r,sn)); s=sn
    return buf

buf=collect()
S=torch.stack([t[0] for t in buf]); SN=torch.stack([t[3] for t in buf])
R=torch.tensor([t[2] for t in buf],dtype=torch.float32)

# GAE 核心:从后往前递推 O(n)
with torch.no_grad():
    Vs=ac.v(S); Vn=ac.v(SN)
    deltas=R+gamma*Vn-Vs              # δ_t = r + γV(s') - V(s)
    A=torch.zeros_like(deltas)
    A[-1]=deltas[-1]                  # 终止步起步
    for t in reversed(range(len(buf)-1)):
        A[t]=deltas[t]+gamma*lam*A[t+1]   # A_t = δ_t + γλ A_{t+1}
    ret=Vs+A                          # critic target = V + A

# advantage 归一化 + 简单 PG 更新
A_norm=(A-A.mean())/(A.std()+1e-8)
logps=torch.stack([ac.pi(t[0]).log_prob(t[1]) for t in buf])
actor_loss=-(logps*A_norm.detach()).mean()
critic_loss=((ac.v(S)-ret.detach())**2).mean()
loss=actor_loss+0.5*critic_loss
opt.zero_grad(); loss.backward(); opt.step()
print("A mean/std:", A.mean().item(), A.std().item(), "| ret mean:", ret.mean().item())
```

> [!tip] LLM-RL 的 GAE 片段（reward 末尾倒推）
> ```python
> # response 长 T,reward 只在末尾 r_phi,中间 r=0,γ=1
> rewards=torch.zeros(T); rewards[-1]=r_phi
> V=v(states); V_next=v(next_states); V_next[-1]=0       # 终止 V=0
> deltas=rewards+gamma*V_next-V
> A=torch.zeros_like(deltas); A[-1]=deltas[-1]
> for t in reversed(range(T-1)): A[t]=deltas[t]+lam*A[t+1]   # γ=1 → 衰减纯靠 λ
> ret=V+A
> ```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[advantage function]]（要估的量 $A^\pi$）、[[advantage estimation]]（方法族总览，GAE 是其中主力）、[[return与reward]]（$G_t$、$\gamma$）、[[value network]]（critic $V_\phi$）、[[baseline]]（$G-V=A$ 的来源）、TD 学习（$\delta_t$）。
- **下游（应用）**: [[PPO clipped objective]]（GAE 是 PPO 标配 advantage）、A3C/A2C、IMPALA（V-trace 是 off-policy 版 GAE）、RLHF 的 token 级信用分配。
- **对比 / 易混**:
  - **GAE vs n-step**：n-step 固定一个 n；GAE 是所有 n 的 $\lambda$ 加权平均，自适应。
  - **GAE vs MC**：$\lambda=1$ 时 GAE=MC；GAE 严格更广。
  - **GAE vs TD(0)**：$\lambda=0$ 时 GAE=TD(0)。
  - **GAE vs V-trace**：V-trace 是 off-policy 版（带截断 importance ratio），IMPALA 用；GAE 是 on-policy。
  - **$\gamma$ vs $\lambda$**：见 §3.7，前者是折扣后者是估计器参数。
  - **GAE return target $V+A$ vs 裸 $G_t$**：前者稳（critic 先验+修正），后者高方差。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **递推方向错（正推）**：必须**从后往前**（$A_t$ 依赖 $A_{t+1}$），正推会漏项。`for t in reversed(range(...))`。
> 2. **衰减率用 $\lambda$ 而非 $\gamma\lambda$**：GAE 的衰减是**乘积 $\gamma\lambda$**，写成 `A[t]=deltas[t]+lam*A[t+1]` 漏 $\gamma$ 会算错尺度。
> 3. **$\delta_t$ 没 detach critic**：$V_\phi$ 给 actor 用时若带梯度，actor 去最小化 $V$。`with torch.no_grad()` 或 `.detach()`。
> 4. **critic target 用裸 advantage `A` 不加 `V`**：`target = A` 丢 baseline，critic 学偏；用 `target = V + A` 或裸 $G_t$。
> 5. **终止步 $V$ 不置 0**：$s_T$ 是终止态，$V(s_T)=0$；不置 0 则 $\delta_{T-1}$ 含一个虚假的 $V$，advantage 全错。
> 6. **$\gamma=1$ 用于 continuing**：无穷级数发散，GAE 不收敛；episodic 固定长度才可 $\gamma=1$（衰减靠 $\lambda$）。
> 7. **LLM-RL 中间步 $r=0$ 时 $\delta$ 写错**：$\delta_{t<T-1}=\gamma V(s_{t+1})-V(s_t)$（无 reward 项），$\delta_{T-1}=r_\phi+\gamma V(s_T)-V(s_{T-1})$。漏 $r_\phi$ 或漏倒推会让所有 token advantage 相同。
> 8. **$\lambda$ 调过小（LLM-RL）**：$\lambda=0.9$ 衰减太快，远端 token 拿不到末尾 reward 信号，长 response 训不动；取 0.95~1.0。
> 9. **advantage 不归一化**：量级随训练漂移，PPO 学习率难调；batch 归一化。
> 10. **GAE 与 reward normalization 冲突**：reward 归一化改 reward 尺度，$V$ 也随之变；要协调：先 reward 归一化再算 GAE，critic 学归一化后的尺度。
> 11. **off-policy 用旧 GAE**：策略变了 advantage 变，需重算或 importance reweight（V-trace）；GAE 是 on-policy 估计。

## 8. 延伸细节

### 8.1 GAE 的"信 critic 距离"直觉

$\hat A_t=\delta_t+\gamma\lambda\,\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots$。$(\gamma\lambda)^k$ 衰减意味着：远处（大 $k$）的 $\delta$ 权重小——因为远处的信息 critic $V_\phi$ 已部分包含（$V$ 是未来 reward 的期望），信 critic 即可；近处（小 $k$）的 $\delta$ 权重大——实际发生的 reward 比 critic 预测更可信。$\lambda$ 调"信 critic 多远"。

### 8.2 GAE 的偏差来源

GAE 的偏差全来自 $V_\phi\ne V^\pi$：$\delta_t$ 的期望 $=A^\pi$ 仅在 $V_\phi=V^\pi$ 时成立。critic 不准时，每个 $\delta$ 都带偏差，GAE 累加放大偏差。故 critic 越准 GAE 越可用小 $\lambda$（信 critic）；critic 差时大 $\lambda$ 反而引入更多 $\delta$ 偏差——这是 $\lambda$ 调节的权衡本质。LLM-RL 的 critic 常用更大网络 + 预训练 + 多 step 更新以提准。

### 8.3 reward 末尾倒推的衰减曲线（LLM-RL）

设 $r_\phi=1$，$\gamma=1$，$V=0$（critic 暂置 0 看纯 reward 传播）。$\delta_{T-1}=1$，$\delta_{t<T-1}=0$。GAE：$A_t=\lambda^{T-1-t}$（从末尾往前指数衰减）。$\lambda=0.95$、$T=200$ 时 $A_0\approx0.95^{199}\approx4e-5$——开头 token 几乎拿不到信号。这正是 LLM-RL 长 response 信用分配难、critic 必须非零减底色的原因：critic 学到 $V_\phi$ 后，$A_t=\delta_t+\lambda A_{t+1}$ 不再纯衰减，而是"比 critic 预测高/低"的相对量，每个 token 都有信号。

### 8.4 V-trace（off-policy GAE）

IMPALA 的 V-trace：$\hat A_t=\sum_{k}(\gamma\lambda)^k\bigl(\prod_{i<k}\rho_{t+i}^{(\bar\rho)}\bigr)\rho_{t+k}^{(\rho)}\delta_{t+k}^{V}$，用截断 importance ratio $\rho^{(\bar\rho)}=\min(\bar\rho,\rho)$ 修正旧策略样本。LLM-RL 少用（on-policy 为主），但概念上 GAE + off-policy 修正 = V-trace。

### 8.5 GAE 与 reward normalization / advantage normalization 的"三件套"

PPO 稳方差的标配组合：
1. **reward normalization**（running std 归一化 reward，[[reward normalization]]）→ $V$ 尺度稳定；
2. **GAE**（$\lambda$ 调偏差-方差）→ advantage 估计；
3. **advantage 归一化**（batch 内 `(A-A.mean())/(A.std()+1e-8)`）→ 再压一层。

三者协同：reward 归一化让 GAE 输入稳定，advantage 归一化让 GAE 输出尺度可控，PPO 学习率才好调。

### 8.6 GAE 在 PPO 多 epoch 中的不变性

PPO 一个 rollout batch 做 $K$ epoch，**$\hat A_t^{GAE}$ 全程不变**（用 rollout 时的 critic 算好、detach）。每 epoch 只重算 ratio $\rho_t$ 与 $L^{CLIP}$，不重算 advantage。若每 epoch 重算 advantage（用更新后的 critic）会引入 actor-critic 耦合、不稳。见 [[PPO clipped objective]] §7 误区 5。

### 8.7 $\lambda$ 与 critic 准度的自适应

理论上可动态调 $\lambda$：监控 critic loss（或 explained variance），critic 准则降 $\lambda$（信 critic），不准则升 $\lambda$（信实际）。实践中多固定 $\lambda=0.95$，因动态调引入额外超参与不稳。LLM-RL 的 $\lambda$ 常取 0.95~1.0，配合 $\gamma=1$。

---
相关: [[PPO]]、[[advantage function]]、[[advantage estimation]]、[[return与reward]]、[[value network]]、[[baseline]]、[[variance reduction]]、[[PPO clipped objective]]、[[reward normalization]]、[[policy与value function]]
