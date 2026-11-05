# GAE

> **所属章节**: [[PPO]]
> **所属模块**: [[04-强化学习基础]]

## 1. 一句话定义

**GAE（Generalized Advantage Estimation）** 是 [[advantage estimation]] 的主力方法，用参数 $\lambda\in[0,1]$ 在 **MC advantage**（$\hat A=G_t-V$，无偏高方差）与 **TD(0) advantage**（$\hat A=\delta_t$，有偏低方差）之间做**指数加权插值**：

$$
\hat A_t^{GAE(\lambda)}=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k},\qquad \delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

- $\lambda=0$ → $\hat A_t=\delta_t$（TD(0)，低方差高偏差）；
- $\lambda=1$ → $\hat A_t=G_t-V_\phi$（MC，无偏高方差）；
- $0<\lambda<1$ → n-step advantage 的 $\lambda$ 加权平均（偏差-方差可调）。

GAE 是 [[PPO clipped objective]]、A3C、IMPALA 等现代策略梯度的**标配 advantage 估计法**。由 Schulman 2016 提出。

> [!note] 一句话直觉
> 近处 reward 信得过（实际发生的），远处信 critic（$V_\phi$ 已含期望）。衰减率 $\gamma\lambda$ 控制"信 critic 的距离"：$\lambda$ 小只信一步实际、剩下靠 critic；$\lambda$ 大信很多步实际。**$\gamma$ 是折扣，$\lambda$ 是估计器参数，实际衰减率是乘积 $\gamma\lambda$**。

## 2. 为什么需要它

[[advantage function]] 说策略梯度用 $A^\pi=Q^\pi-V^\pi$ 作权重，但 $A^\pi$ 未知要估。两种朴素估法各有硬伤：

| 估法 | 形式 | 偏差 | 方差 | 问题 |
|---|---|---|---|---|
| MC | $G_t-V_\phi$ | 无偏 | 高 | 单条轨迹 $G_t$ 噪声大、需等回合结束 |
| TD(0) | $\delta_t$ | 有偏（依赖 $V_\phi$） | 低 | critic 不准时方向错、只看一步太短视 |

想要的是"用几步实际 reward + critic 收尾"的 n-step，且能连续调节 n。GAE 的洞见：**对所有 n-step advantage 做 $\lambda$ 加权平均**，得到从 TD 到 MC 的连续谱，一个参数 $\lambda$ 即可调。

LLM-RL 尤其依赖 GAE：reward 只在序列末尾给一次，return 退化为单点、方差爆炸，必须靠 GAE 的指数衰减把末尾 reward 倒推到每个 token。

## 3. 核心概念详解

### 3.1 TD 误差 $\delta_t$（构造块）

$$
\delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

- TD(0) 的 advantage 估计；
- $\mathbb{E}[\delta_t|s_t,a_t]=A^\pi$（$V_\phi=V^\pi$ 时无偏，§4.1）；
- 实际 $V_\phi\ne V^\pi$ 引入偏差，但单步方差小；
- "这步实际 reward + critic 对下一步的预测 - critic 对这步的预测" = 单步惊喜。

### 3.2 GAE 定义与两端点

$$
\hat A_t^{GAE(\lambda)}=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}=\delta_t+\gamma\lambda\,\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots
$$

衰减率是 $\gamma\lambda$（不是 $\gamma$）；系数 $(\gamma\lambda)^k$ 需 $\gamma\lambda<1$ 才收敛。

| $\lambda$ | GAE 等价于 |
|---|---|
| $0$ | $\delta_t$（TD(0) advantage） |
| $1$ | $\sum_k\gamma^k\delta_{t+k}=G_t-V_\phi$（MC advantage，望远镜和证明见 §4.2） |

GAE 严格包含 MC 与 TD 为特例。

### 3.3 从后往前递推（工程实现 $O(n)$）

$$
\hat A_t=\delta_t+\gamma\lambda\,\hat A_{t+1}
$$

从轨迹末尾往前推，$\hat A_T=0$（终止态 $V=0$）。这是所有 PPO 实现的标准写法。

### 3.4 GAE return target（critic 回归目标）

$$
\hat G_t=V_\phi(s_t)+\hat A_t^{GAE(\lambda)}
$$

比裸 $G_t$（MC 高方差）或裸 $r+\gamma V$（TD 高偏差）更稳：含 critic 先验 $V_\phi$ + advantage 修正。

### 3.5 $\gamma$ 与 $\lambda$ 的分工（易混点）

| 参数 | 角色 | 调大 | 典型值 |
|---|---|---|---|
| $\gamma$ | **折扣**（定义 return，时间偏好/收敛） | 远视、视野 $1/(1-\gamma)$ 增 | 0.99 |
| $\lambda$ | **GAE 偏差-方差**（信 critic 的距离） | 偏 MC、方差↑偏差↓ | 0.95 |

两者独立：$\gamma$ 是 MDP 的属性（定义 return），$\lambda$ 是估计器的属性（不改变 RL 目标）。GAE 的实际衰减率是乘积 $\gamma\lambda$，但 $\gamma$ 改 return 本身、$\lambda$ 只改估计法。混用会让 advantage 尺度与 target 错配。

### 3.6 LLM-RL 的 GAE 倒推（核心应用）

RLHF 中 reward 仅序列末尾给一次 $r_\phi$（其余步 $r=0$），$\gamma$ 常取 1（episodic 固定长度）。GAE 简化为：

$$
\delta_{T-1}=r_\phi+\gamma V(s_T)-V(s_{T-1}),\quad \delta_{t<T-1}=\gamma V(s_{t+1})-V(s_t)
$$

（$s_T$ 终止 $V=0$）。递推 $\hat A_t=\delta_t+\lambda\hat A_{t+1}$ 把末尾 $r_\phi$ 按 $\lambda^k$ 衰减分到每个 token——靠 $\lambda$ 完成信用分配（$\gamma=1$ 时衰减纯靠 $\lambda$）。$\lambda$ 常取 0.95~1.0：太小（如 0.9）衰减太快，远端 token 拿不到信号；=1 退化为所有 token 同分（MC），方差爆。critic 减底色 $V_\phi$ 后 advantage 才有正负区分。

## 4. 关键推导

### 4.1 $\delta_t$ 是 $A^\pi$ 的无偏估计（$V_\phi=V^\pi$ 时）

$$
\mathbb{E}_{s_{t+1}\sim P}[\delta_t]=R+\gamma\mathbb{E}[V^\pi(s_{t+1})]-V^\pi(s_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)=A^\pi(s_t,a_t)
$$

用 $Q^\pi=R+\gamma\mathbb{E}[V^\pi]$。故 TD 误差理论无偏，偏差全来自 $V_\phi\ne V^\pi$。GAE 偏差也全来自此。

### 4.2 $\lambda=1$ 退化为 MC（望远镜和）

$$
\sum_{k=0}^{T-t-1}\gamma^k\delta_{t+k}=\sum_{k=0}^{T-t-1}\gamma^k\bigl(r_{t+k+1}+\gamma V(s_{t+k+1})-V(s_{t+k})\bigr)
$$

相邻两项的 $V(s_{t+k+1})$ 抵消（$\gamma^k\cdot\gamma V(s_{t+k+1})$ 与 $-\gamma^{k+1}V(s_{t+k+1})$），最终只剩首尾：

$$
=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}+\gamma^{T-t}V(s_T)-V(s_t)=G_t-V(s_t)\quad(V(s_T)=0)
$$

### 4.3 后向递推

$\hat A_t=\delta_t+\gamma\lambda(\delta_{t+1}+\gamma\lambda\,\delta_{t+2}+\cdots)=\delta_t+\gamma\lambda\,\hat A_{t+1}$。

### 4.4 GAE = n-step 的 $\lambda$ 加权平均

n-step：$\hat A_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V(s_{t+n})-V(s_t)$。GAE = $(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)}$（$(1-\lambda)$ 归一化）。故 GAE 是"所有 n-step 的几何加权"，比固定 n 自适应：critic 准时小 $\lambda$（信 critic，n 小），不准时大 $\lambda$（信实际，n 大）。

### 4.5 偏差-方差随 $\lambda$

$\lambda\uparrow$：向 MC 靠，偏差↓方差↑；$\lambda\downarrow$：向 TD 靠，偏差↑方差↓。最优 $\lambda$ 取决于 critic 准度：critic 越准可用越小 $\lambda$，越不准用大 $\lambda$。$\lambda=0.95$ 是经验甜点（critic 中等准时兼顾）。

## 5. 代码示例

```python
import torch

# response 长 T,reward 只在末尾 r_phi,中间 r=0,γ=1 (LLM-RL 场景)
T=200; gamma=1.0; lam=0.95; r_phi=1.0
rewards=torch.zeros(T); rewards[-1]=r_phi
states=torch.randn(T,8); next_states=torch.randn(T,8)

def v(s): return torch.randn(T)   # 占位 critic

V=v(states); V_next=v(next_states); V_next[-1]=0           # 终止 V=0
deltas=rewards+gamma*V_next-V                               # δ_t
A=torch.zeros_like(deltas); A[-1]=deltas[-1]
for t in reversed(range(T-1)):
    A[t]=deltas[t]+gamma*lam*A[t+1]                          # γ=1 → 衰减纯靠 λ
ret=V+A                                                      # critic target = V + A
adv=(A-A.mean())/(A.std()+1e-8)                              # advantage 归一化
```

> [!tip] 关键接口
> - 递推必须**从后往前**（`reversed`），正推会漏项；
> - 衰减率是 $\gamma\lambda$，写成 `lam*A[t+1]` 漏 $\gamma$ 会算错尺度；
> - $\delta$ 计算时 critic 用 `torch.no_grad()` 或 detach，防 actor 去最小化 $V$。

## 6. 与其他知识点的关系

- **上游**: [[advantage function]]（要估的量）、[[advantage estimation]]（方法族总览）、[[return与reward]]（$G_t$、$\gamma$）、[[Actor-Critic]]（critic $V_\phi$）、TD 学习（$\delta_t$）。
- **下游**: [[PPO clipped objective]]（GAE 是 PPO 标配 advantage）、A3C/A2C、IMPALA（V-trace 是 off-policy 版 GAE）、RLHF 的 token 级信用分配。
- **对比**:
  - **GAE vs n-step**：n-step 固定一个 n；GAE 是所有 n 的 $\lambda$ 加权平均，自适应。
  - **GAE vs V-trace**：V-trace 是 off-policy 版（带截断 importance ratio），IMPALA 用；GAE 是 on-policy。
  - **$\gamma$ vs $\lambda$**：见 §3.5，前者折扣后者估计器参数。
  - **GAE return $V+A$ vs 裸 $G_t$**：前者稳（critic 先验+修正），后者高方差。

## 7. 常见误区

1. **递推方向错（正推）**：必须从后往前（$A_t$ 依赖 $A_{t+1}$），`for t in reversed(range(...))`。
2. **衰减率用 $\lambda$ 而非 $\gamma\lambda$**：写成 `A[t]=deltas[t]+lam*A[t+1]` 漏 $\gamma$ 会算错尺度。
3. **$\delta_t$ 没 detach critic**：$V_\phi$ 给 actor 用时若带梯度，actor 去最小化 $V$。
4. **critic target 用裸 advantage `A` 不加 `V`**：丢 baseline，critic 学偏；用 `target = V + A`。
5. **终止步 $V$ 不置 0**：$s_T$ 终止态 $V(s_T)=0$；不置 0 则 $\delta_{T-1}$ 含虚假 $V$，advantage 全错。
6. **$\gamma=1$ 用于 continuing**：无穷级数发散，GAE 不收敛；episodic 固定长度才可 $\gamma=1$（衰减靠 $\lambda$）。
7. **LLM-RL 中间步 $r=0$ 时 $\delta$ 写错**：$\delta_{t<T-1}=\gamma V(s_{t+1})-V(s_t)$（无 reward），$\delta_{T-1}=r_\phi+\gamma V(s_T)-V(s_{T-1})$。
8. **$\lambda$ 调过小（LLM-RL）**：$\lambda=0.9$ 衰减太快，远端 token 拿不到末尾 reward 信号；取 0.95~1.0。
9. **advantage 不归一化**：量级随训练漂移，PPO 学习率难调；batch 内归一化。
10. **off-policy 用旧 GAE**：策略变了 advantage 变，需重算或 importance reweight（V-trace）；GAE 是 on-policy 估计。

## 8. PPO 多 epoch 中的不变性

PPO 一个 rollout batch 做 $K$ epoch，**$\hat A_t^{GAE}$ 全程不变**（用 rollout 时的 critic 算好、detach）。每 epoch 只重算 ratio $\rho_t$ 与 $L^{CLIP}$，不重算 advantage。若每 epoch 重算 advantage（用更新后的 critic）会引入 actor-critic 耦合、不稳。见 [[PPO clipped objective]] §7 误区 5。

---
相关: [[PPO]]、[[advantage function]]、[[advantage estimation]]、[[PPO clipped objective]]、[[return与reward]]、[[Actor-Critic]]
