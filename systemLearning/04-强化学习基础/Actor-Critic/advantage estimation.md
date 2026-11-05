# advantage estimation

> **所属章节**: [[Actor-Critic]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**advantage estimation（优势估计）** 是用采样数据 + critic $V_\phi$ 把理论 $A^\pi(s,a)=Q^\pi-V^\pi$ 算出可用估计 $\hat A_t$ 的方法族——核心包括 MC $\hat A=G_t-V_\phi$、TD(0) $\hat A=\delta_t=r+\gamma V_\phi(s')-V_\phi(s)$、n-step、以及工业标准 **[[GAE]]**。估计好坏直接决定 [[Actor-Critic]]/[[PPO clipped objective]] 的梯度质量，是 RL 工程最核心的"算权重"环节。

## 2. 为什么需要它

[[advantage function]] 说策略梯度用 $A^\pi$ 作权重，但 $A^\pi$ 需 $Q^\pi,V^\pi$ 都已知——现实未知。要得到可用 $\hat A_t$：

- 直接用 $G_t$（[[REINFORCE]]）→ 是 $Q^\pi$ 的 MC 估计，没减 $V^\pi$ → 高方差。
- 减 [[baseline]] $V_\phi$ → $\hat A=G_t-V_\phi$（MC advantage）→ 无偏但方差仍大、需等回合。
- 用 TD 误差 $\delta_t$ → 有偏（依赖 $V_\phi$）但低方差、在线。
- 折中 → GAE。

本质是"**在偏差、方差、算力、延迟之间找平衡**"：MC 无偏高方差需等回合；TD 有偏低方差但 critic 不准则方向错；GAE 用 $\lambda$ 在二者间连续调节，工业主流。LLM-RL 尤其依赖：reward 只在序列末尾、return 单点方差爆，必须靠 GAE 倒推 + critic 减底色才能压到可训。估错 advantage → 梯度方向错 → policy 跑飞（[[KL explosion]]/[[policy collapse]]）。

## 3. 核心估计法对比

| 方法 | 公式 | 偏差 | 方差 | 用途 |
|---|---|---|---|---|
| MC | $\hat A_t=G_t-V_\phi(s_t)$ | 无偏 | 高 | REINFORCE+baseline |
| TD(0) | $\hat A_t=\delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$ | 有偏（依赖 $V_\phi$） | 低 | A2C 简版 |
| n-step | $\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V_\phi(s_{t+n})-V_\phi(s_t)$ | 中 | 中 | n-step Q |
| **GAE(λ)** | $\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}$ | 可调 | 可调 | **PPO/A3C 标配** |

- **MC**：$\mathbb E[G_t\mid s,a]=Q^\pi$，无偏；单条轨迹噪声大、需等回合。适合短回合或无 critic。
- **TD(0)**：$\mathbb E[\delta_t\mid s,a]=A^\pi$（$V_\phi=V^\pi$ 时），理论无偏，实践偏差全来自 $V_\phi\ne V^\pi$；低方差、在线。
- **n-step**：前 n 步用实际 reward、之后用 $V_\phi$ 收尾。n 大→偏 MC、n 小→偏 TD。
- **GAE(λ)**：$\lambda=0$ 退化为 TD(0)，$\lambda=1$ 退化为 MC，$0<\lambda<1$ 是 n-step 的指数加权平均。一次轨迹从后往前递推即可算，详见 [[GAE]] 专篇。

### critic target 与 advantage 的关系

- critic target：$V_\phi$ 回归的 $y_t$，常取 $G_t$（MC）/ $r+\gamma V_\phi(s')$（TD）/ GAE-based return $V_\phi(s_t)+\hat A_t$。
- advantage：给 actor 的权重 $\hat A_t$。
- 常用技巧：$\hat A_t$ 算完后 critic target 取 $V_\phi(s_t)+\hat A_t$（"advantage + baseline" 形式），数值稳定。

### advantage 归一化

$\hat A_t$ 算出后常做 batch 归一化 `(A-A.mean())/(A.std()+1e-8)`，进一步降方差稳尺度。detach mean/std 避免耦合。见 [[variance reduction]]。

### LLM-RL 的特殊处理

RLHF reward 仅序列末尾一次（$r_T=r_\phi$，其余 0）。GAE 倒推：中间步 $\delta_t=\gamma V(s_{t+1})-V(s_t)$（r=0），$\delta_{T-1}=r_\phi+\gamma V(s_T)-V(s_{T-1})$。末尾 return 经指数衰减分到每个 token，critic 减底色。$\gamma$ 常取 1（episodic），$\lambda$ 取 0.95~1.0。详见 [[GAE]]。

## 4. 数学原理

### 4.1 $\delta_t$ 是 $A^\pi$ 的无偏估计（$V_\phi=V^\pi$ 时）

$$
\mathbb E_{s_{t+1}\sim P}[\delta_t]=R(s_t,a_t)+\gamma\mathbb E[V^\pi(s_{t+1})]-V^\pi(s_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)=A^\pi(s_t,a_t)
$$

（用 [[policy与value function]] 的 $Q^\pi=R+\gamma\mathbb E[V^\pi]$）。故 TD 误差理论无偏，偏差全来自 $V_\phi\ne V^\pi$。

### 4.2 GAE 的等价形式

$$
\hat A_t^{GAE(\lambda)}=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)}
$$

即 GAE 是 n-step advantage 的 $\lambda$-加权平均。证明见 [[GAE]]。

### 4.3 偏差-方差的 λ 调节

$\lambda\uparrow$：向 MC 靠，偏差↓方差↑；$\lambda\downarrow$：向 TD 靠，偏差↑方差↓。最优 $\lambda$ 取决于 critic 准度：critic 越准可用越小 $\lambda$（信 critic），越不准要用大 $\lambda$（信实际 return）。

## 5. 代码示例

```python
with torch.no_grad():
    Vs, Vn = ac.v(states), ac.v(nexts)
    deltas = rs + gamma * Vn - Vs
    A = torch.zeros_like(deltas); A[-1] = deltas[-1]
    for t in reversed(range(len(buf)-1)):
        A[t] = deltas[t] + gamma * lam * A[t+1]   # GAE(λ) 倒推
    Gt = Vs + A                                    # critic target

A_norm = (A - A.mean()) / (A.std() + 1e-8)          # advantage 归一化
v, logps = ac.v(states), ac.pi(states).log_prob(actions)
actor_loss  = -(logps * A_norm.detach()).mean()
critic_loss = ((v - Gt.detach())**2).mean()
loss = actor_loss + 0.5 * critic_loss
```

> [!tip] 选择经验
> 回合短、无 critic：MC；在线、critic 快收敛：TD(0)；通用、PPO/A3C：GAE(λ=0.95)；LLM-RL：GAE(λ=0.95~1.0, γ=1)。

## 6. 与其他知识点的关系

- **上游**: [[advantage function]]（要估的目标）、[[value network]]（critic 提供减项）、[[return与reward]]（$G_t$）、[[baseline]]（$G-V$）。
- **下游**: [[Actor-Critic]]、[[PPO clipped objective]]（GAE 是 PPO 标配）、[[GAE]]（主力方法专篇）、[[variance reduction]]。
- **对比**:
  - **advantage estimation vs advantage function**：前者是估计方法，后者是定义量。
  - **critic target vs advantage**：target 回归 $V$，advantage 给 actor；同轨迹同时算两份。
  - **advantage 归一化 vs reward 归一化**：前者控梯度权重，后者控 reward 尺度。

## 7. 常见误区与易错点

1. **$\delta_t$ 没 detach critic**：advantage 给 actor 时若 $V_\phi$ 梯度混入，actor 去最小化 $V$。detach。
2. **GAE 递推方向错**：必须从后往前（$A_t$ 依赖 $A_{t+1}$），正推漏项。
3. **critic target 用 advantage 不加 $V$**：`target = A` 丢 baseline，critic 学偏；用 `target = V + A` 或 `G_t`。
4. **λ=0 直接用 TD**：critic 未收敛时偏差大；先用大 λ 过渡。
5. **LLM-RL 中间步 reward 没正确倒推**：中间步 r=0，GAE 靠 $\gamma V$ 传递；$\gamma=0$ 则 advantage 全 0。
6. **advantage 不归一化**：量级漂移，学习率难调。
7. **off-policy 用旧 advantage**：策略变了 advantage 变，需重算或 importance reweight。
8. **critic target 与 advantage 用同一 $V$ 不分离**：critic 用 $V_\phi(s_t)$ 当 target 会自举；用 $V_\phi(s_t)+A$ 或独立 target network。
9. **GAE 的 $\gamma$ 与任务不匹配**：episodic 可 $\gamma=1$；continuing 必须 $<1$。

## 8. 延伸细节

### 8.1 GAE 物理直觉

GAE 把"未来每步的 TD 误差"按 $\gamma\lambda$ 衰减累加——远处 $\delta$ 权重小（critic 已含其期望，信它即可），近处 $\delta$ 权重大（信实际）。$\lambda$ 控制"信 critic 多远"。

### 8.2 V-trace（IMPALA）

off-policy 版 GAE，用截断 importance ratio 修正旧策略样本，IMPALA 用。LLM-RL 少用（on-policy 为主），概念相关。

### 8.3 与 reward normalization 协同

reward 归一化让 $V$ 尺度稳定，advantage 估计更稳；advantage 归一化再压一层。两者与 GAE 构成 PPO 的"三件套稳方差"。

---
相关: [[Actor-Critic]]、[[GAE]]、[[value network]]、[[advantage function]]、[[baseline]]、[[variance reduction]]、[[PPO clipped objective]]
