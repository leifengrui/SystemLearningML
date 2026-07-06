# PPO optimization

> **所属章节**: [[RLHF流程]]
> **所属模块**: [[05-LLM-RL对齐]]
> **难度**: 难（RLHF 第三阶段，多组件协同的工程核心）

## 1. 一句话定义

**PPO optimization（RLHF 的 PPO 阶段）** 是 RLHF 的**第三阶段**：用 [[PPO clipped objective]] 算法在 LLM 上做强化学习，把 [[Reward Model]] 的分数当 reward、把 [[SFT]] 模型当冻结的 reference $\pi_{\text{ref}}$、用 [[GAE]] 算 token 级 advantage、用 [[KL penalty与KL control]] 约束 $\pi_\theta$ 不漂离 $\pi_{\text{ref}}$，最终得到对齐人类偏好的 $\pi_\theta$。一条迭代 = **rollout（actor 生成 response）→ reward（RM 打分 − KL penalty）→ advantage（GAE 倒推）→ update（PPO clip + value loss + entropy）**，循环多轮。它是把 RLHF 各零件（SFT/RM/PPO/GAE/KL）**组装成一个可跑训练 loop** 的工程核心，由 InstructGPT（OpenAI 2022）确立，verl/TRL/DeepSpeed-Chat 等开源框架实现。

> [!note] 与第四章 PPO 的关系
> 第四章 [[PPO clipped objective]] 讲的是 **PPO 算法本身**（clipped surrogate 怎么来、为何 clip）。本篇讲的是 **PPO 在 RLHF 里怎么组装**——多了 LLM 特有的：RM 当 reward、reference model 做 KL 锚点、token 级 ratio、rollout 生成昂贵。算法内核仍第四章那套，但工程外壳是为 LLM 定制的。

## 2. 为什么需要它（动机与背景）

[[SFT]] 教模型"听指令"，[[Reward Model]] 学会"判好坏"，但还差一步——**用 RM 的分数把 SFT 模型进一步优化成对齐偏好的**。这一步用 RL 的原因：

1. **RM 是判别器，不是生成器**：RM 能给已有回答打分，但不能直接生成更好回答。要把"判别信号"变成"生成能力提升"，需 RL（策略梯度用 reward 优化 $\pi_\theta$）。
2. **目标带 KL 约束**：RLHF 目标是 $\max_\theta\mathbb{E}[r_{\text{RM}}]-\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$，既有 reward 最大化又有 KL 约束——监督学习（SFT）没有"约束在 $\pi_{\text{ref}}$ 附近"的机制，需 RL 的 KL-constrained PO 框架。
3. **PPO 是工业验证过的稳定 PG**：[[REINFORCE]] 方差太大，TRPO 慢，[[PPO clipped objective]] 用 clip 软约束 + 多 epoch 复用，在 LLM 生成昂贵的场景样本效率够用。InstructGPT 选 PPO 后成事实标准。

故 RLHF 第三阶段 = "用 PPO 解 KL-constrained reward maximization"。这阶段最复杂：要同时跑 actor（生成）、critic（估 $V$）、reference（算 KL）、reward model（算 reward）四个模型，还要 rollout 与多 epoch 训练 loop 协同。是 RLHF 工程的瓶颈与核心。

## 3. 核心概念详解

### 3.1 RLHF-PPO 的目标函数

$$
\max_\theta\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\bigl[r_\phi(x,y)\bigr]-\beta\,\mathbb{E}_{x,y\sim\pi_\theta}\bigl[\text{KL}(\pi_\theta(\cdot|x)\big\|\pi_{\text{ref}}(\cdot|x))\bigr]
$$

- $r_\phi$：[[Reward Model]]（冻结）；
- $\pi_{\text{ref}}=\pi_{\text{SFT}}$：reference model（冻结）；
- $\beta$：KL 系数（adaptive，[[KL penalty与KL control]] §3.6）；
- 第一项：reward 最大化（学人类偏好）；第二项：KL 约束（保 SFT 语言能力）。
- 这是 [[DPO]] 推导的出发点——DPO 证明此目标有闭式解，从而绕过 PPO（[[DPO]] §4）。

### 3.2 四个模型与各自角色

| 模型 | 角色 | 是否更新 | 大小 |
|---|---|---|---|
| **actor $\pi_\theta$** | 生成 response、被优化 | ✅ 更新 | 与 SFT 同规模（如 7B） |
| **critic $V_\phi$** | 估 value 算 advantage | ✅ 更新 | 常与 actor 同规模（或更大） |
| **reference $\pi_{\text{ref}}$** | 算 KL penalty | ❌ 冻结 | 与 SFT 同规模 |
| **reward model $r_\phi$** | 打 reward 分 | ❌ 冻结 | 常从 SFT 初始化，可小 |

- actor 与 critic 可**共享 body**（省显存）或**分离**（更稳，主流）；
- reference 与 reward model 全程冻结，只前向；
- 四个模型同时在显存里 → RLHF 显存是 SFT 的 ~4 倍（详见 [[activation memory]]），是显存瓶颈。

### 3.3 一条 PPO 迭代的完整流程

```
1. Rollout(生成)
   - 用 actor π_θ 采 N 条 prompt,每条生成 response y
   - 同时算:旧 actor 的 token 级 logp_old(采样快照)
   - 用 reference π_ref 算每 token 的 logp_ref(用于 KL)
   - 用 reward model 算每条 response 的标量 r_φ(x,y)
   - 存 buffer:(s_t, a_t, logp_old, logp_ref, r_φ)

2. Reward 组装
   - per-token reward:r_t = (末尾 r_φ 分摊) - β·(logp_θ - logp_ref)
     · 末尾 token 加 r_φ;中间 token 只扣 KL
   - (或:序列级 r_φ 在末尾,中间 r=0,GAE 倒推)

3. Advantage 估计
   - 用 critic V_φ 算 GAE:δ_t = r_t + γV(s_{t+1}) - V(s_t)
   - 倒推 A_t = δ_t + γλ A_{t+1}(γ≈1, λ≈0.95)
   - critic target = V + A
   - advantage 归一化

4. PPO Update(K 个 epoch,每个 epoch 多个 minibatch)
   - 重算 ratio ρ_t = exp(logp_θ(a_t|s_t) - logp_old)
   - actor loss = -min(ρ·A, clip(ρ,1-ε,1+ε)·A).mean()
   - critic loss = (V_φ(s_t) - target)².mean()
   - entropy bonus = -c_e·H(π_θ)(常关或极小)
   - 总 loss = actor + c_v·critic - c_e·entropy
   - backward + step
   - 每 epoch 后算 approx_kl,超 target_kl 则 break

5. 监控
   - reward、KL、clip_fraction、response length、critic loss
   - 异常 → 调 β/lr/K/ε
```

### 3.4 token 级 ratio 与 advantage（LLM 特有）

- **ratio**：$\rho_t=\pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$，token 级（每 token 一个）。$a_t$ 是 token，$s_t$ 是 prompt+已生成前缀；
- **advantage**：$\hat A_t$ 用 GAE 倒推，token 级；
- 一条 response 长 $T$ → $T$ 个样本进 PPO loss；
- 这与游戏 RL 的"每步一个 $(s,a)$"完全同构，只是 action 是 token、state 是前缀（[[MDP]] §8.1）。

### 3.5 reward 的两种组装方式

| 方式 | 形式 | 用例 |
|---|---|---|
| **KL-per-token + 末尾 RM** | $r_t=-\beta\,\text{KL}_t$（中间）+ $r_{T-1}=r_\phi-\beta\,\text{KL}_{T-1}$ | InstructGPT、TRL 主流 |
| **序列末尾单一 reward** | $r_{T-1}=r_\phi-\beta\,\text{KL}_{seq}$，中间 0 | 简化版，GAE 倒推 |

主流是第一种：per-token KL penalty（每个 token 都扣偏离 ref 的代价），末尾加 RM 分。RM 分通过 GAE 倒推分给每个 token（[[GAE]] §3.8）。

### 3.6 critic 的训练

critic $V_\phi$ 学"从状态 $s_t$（prompt+前缀）平均能拿多少 reward"：
- target：$V_\phi(s_t)+\hat A_t$（GAE return，[[GAE]] §3.5）；
- loss：$(V_\phi(s_t)-\text{target})^2$；
- critic 准度直接决定 advantage 质量 → critic 常用更大网络 + 多 step 预训练 + 更大 lr；
- 监控：explained variance（critic 解释了多少 return 方差），>0.5 算可用。

### 3.7 多 epoch 复用与 target_kl 早停

- 一个 rollout batch 做 $K=2\sim4$ epoch（LLM 生成贵，复用样本提效率）；
- 每 epoch 后算 `approx_kl`，超 `target_kl`（如 6 nats/token）则 break（[[KL penalty与KL control]] §3.7）；
- clip_fraction 监控 0.1~0.3 健康；
- $K$ 太大 → 样本失真；太小 → 利用率低。

### 3.8 RLHF-PPO 的工程框架

| 框架 | 来源 | 特点 |
|---|---|---|
| **verl** | 字节/开源 | 高性能、Ray 调度、支持大模型、训推分离 |
| **TRL** | HuggingFace | 易用、集成 transformers、社区主流 |
| **DeepSpeed-Chat** | 微软 | DeepSpeed 加速、3阶段全流程 |
| **OpenRLHF** | 开源 | 注重可读性与大模型支持 |

这些框架封装了上述四模型协同、rollout、GAE、PPO loop，是落地 RLHF 的基础。

## 4. 数学原理 / 公式

### 4.1 RLHF 目标与 KL-constrained PO

$$
\max_\theta\mathbb{E}_{x,y\sim\pi_\theta}[r_\phi(x,y)]\quad\text{s.t.}\quad\mathbb{E}[\text{KL}(\pi_\theta\|\pi_{\text{ref}})]\le\delta
$$

拉格朗日松弛 → $\max_\theta\mathbb{E}[r]-\beta\,\text{KL}$，β 是约束价格（[[KL penalty与KL control]] §4.1）。PPO 用 clip + KL penalty + target_kl 三重约束近似此目标。

### 4.2 token 级 PPO loss

$$
L^{PPO}(\theta)=\underbrace{-\mathbb{E}_t[\min(\rho_t\hat A_t,\,\text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)]}_{\text{actor }L^{CLIP}}+\underbrace{c_v\,\mathbb{E}_t[(V_\phi(s_t)-\hat G_t)^2]}_{\text{critic }L^{VF}}\underbrace{-c_e\,\mathbb{E}_t[H(\pi_\theta(\cdot|s_t))]}_{\text{entropy}}
$$

- $\rho_t=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{old}}(a_t|s_t))$，token 级；
- $\hat A_t$：GAE 倒推，detach；
- $\hat G_t=V_\phi(s_t)+\hat A_t$：critic target；
- $c_v\approx0.5$，$c_e\approx0$（LLM-RL 常关）；
- $\epsilon\approx0.2$，$K=2\sim4$ epoch。

详见 [[PPO clipped objective]] §3.6、[[GAE]] §3.5、[[entropy bonus]] §3.2。

### 4.3 reward 的 per-token 形式

$$
r_t = \begin{cases}-\beta\,\log\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} & t<T-1\\ r_\phi(x,y)-\beta\,\log\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} & t=T-1\end{cases}
$$

（per-token log-ratio 是 $\text{KL}(\pi_\theta\|\pi_{\text{ref}})$ 的无偏估计，[[KL penalty与KL control]] §4.3）。然后 $r_t$ 进 GAE。

### 4.4 GAE 倒推（γ=1 简化）

$$
\delta_t=r_t+V_\phi(s_{t+1})-V_\phi(s_t),\quad \hat A_t=\delta_t+\lambda\,\hat A_{t+1},\quad \hat A_{T}=0
$$

把末尾 $r_\phi$ 按 $\lambda^k$ 衰减分到每个 token，critic 减底色（[[GAE]] §3.8）。

### 4.5 与 DPO 的闭式对照

DPO 证明 §4.1 目标最优解 $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$，代入 BT 损失得 DPO loss（无需 rollout/RM/critic）。PPO 是"显式跑 RL 求 $\pi^*$"，DPO 是"闭式直接训 $\pi^*$"。等价目标，不同路径（[[DPO]] §4）。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F, copy

# ========= 四个模型(极简演示) =========
class LM(nn.Module):
    """actor / reference 共用此结构:emb + lstm + head(logits)"""
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):           # x:(B,T) -> logits (B,T,V)
        h,_=s.lstm(s.emb(x)); return s.head(h)
    def logp(s, x, a):           # token 级 log π(a|x)
        return F.log_softmax(s(x),-1).gather(-1,a.unsqueeze(-1)).squeeze(-1)

class Critic(nn.Module):
    """critic:emb + lstm + value_head(标量 V)"""
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.vh=nn.Linear(d,1)
    def forward(s, x):           # x:(B,T) -> V (B,T)
        h,_=s.lstm(s.emb(x)); return s.vh(h).squeeze(-1)

class RM(nn.Module):
    """reward model:emb + lstm + scalar(取最后 token)"""
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.s=nn.Linear(d,1)
    def forward(s, x):           # x:(B,T) -> r (B,)
        h,_=s.lstm(s.emb(x)); return s.s(h[:,-1]).squeeze(-1)

V,D=50,32
actor=LM(V,D); critic=Critic(V,D)
ref=copy.deepcopy(actor); [p.requires_grad_(False) for p in ref.parameters()]   # reference 冻结
rm=RM(); [p.requires_grad_(False) for p in rm.parameters()]    # RM 冻结(已预训练好)

opt=torch.optim.Adam(actor.parameters(),1e-6)   # PPO lr 很小
opt_c=torch.optim.Adam(critic.parameters(),1e-5)
gamma,lam,eps,K,beta,cv,ce=1.0,0.95,0.2,4,0.05,0.5,0.0

# ========= 一条 PPO 迭代 =========
B,T=4,8
x=torch.randint(0,V,(B,T))                      # prompt(简化:整段当 prompt+response)
# 1) Rollout:actor 采样(这里简化用 x 当生成结果)
with torch.no_grad():
    a=x; logp_old=actor.logp(x,a)
    logp_ref=ref.logp(x,a)
    r_rm=rm(x)                                  # (B,) 序列级 RM 分
# 2) per-token reward:中间 -β·KL,末尾 +r_rm
with torch.no_grad():
    kl_per_tok=logp_old-logp_ref                # log(π_θ/π_ref) 近似
    r=torch.zeros(B,T)
    r[:,:-1]=-beta*kl_per_tok[:,:-1]
    r[:,-1]=r_rm-beta*kl_per_tok[:,-1]
# 3) GAE(γ=1)
with torch.no_grad():
    V=critic(x).squeeze(-1)                     # (B,T)
    Vn=torch.zeros_like(V); Vn[:,:-1]=V[:,1:]   # V(s_{t+1})
    deltas=r+gamma*Vn-V
    A=torch.zeros_like(deltas); A[:,-1]=deltas[:,-1]
    for t in reversed(range(T-1)): A[:,t]=deltas[:,t]+lam*A[:,t+1]
    ret=V+A
    A=(A-A.mean())/(A.std()+1e-8)
# 4) PPO update K epoch
for _ in range(K):
    logp=actor.logp(x,a)
    ratio=(logp-logp_old).exp()
    surr1=ratio*A; surr2=torch.clamp(ratio,1-eps,1+eps)*A
    actor_loss=-torch.min(surr1,surr2).mean()
    v=critic(x).squeeze(-1)
    critic_loss=((v-ret)**2).mean()
    ent=F.log_softmax(actor(x),-1).exp().mul(F.log_softmax(actor(x),-1)).neg().mean()
    loss=actor_loss+cv*critic_loss-ce*ent
    opt.zero_grad(); opt_c.zero_grad(); loss.backward(); opt.step(); opt_c.step()
    with torch.no_grad():
        approx_kl=((ratio-1)-torch.log(ratio+1e-8)).mean()
        if approx_kl>0.02: break              # target_kl 早停
print(f"reward={r_rm.mean():.2f} actor_loss={actor_loss:.3f} critic_loss={critic_loss:.3f} kl={approx_kl:.4f}")
```

> [!tip] 实际工程（verl/TRL）
> - actor 用 `AutoModelForCausalLM`，critic 单独建模（加 value head）；
> - rollout 用 vLLM/SGLang 加速生成（生成是瓶颈）；
- reference/RM 前向用 `torch.no_grad` + 半精度省显存；
- Ray 做训推分离（rollout worker 与 trainer 分离，见 [[rollout worker]] 待展开）；
- 大模型用 FSDP/ZeRO 切分四模型显存。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[SFT]]（$\pi_{\text{ref}}$ 与 actor 初始化）、[[Reward Model]]（reward 来源）、[[PPO clipped objective]]（算法内核）、[[GAE]]（advantage）、[[KL penalty与KL control]]（KL 约束）、[[entropy bonus]]（loss 第三项）、[[MDP]]（LLM-RL 的 MDP 映射）。
- **下游（应用）**: 对齐后的 LLM（InstructGPT/Llama-chat）、[[reward hacking]]/[[KL explosion]]/[[policy collapse]]（PPO 失败模式）、[[DPO]]（绕过此阶段的替代路径）、verl/TRL 工程框架、[[rollout worker]]/[[训推分离]]（系统支撑）。
- **对比 / 易混**:
  - **PPO optimization vs [[PPO clipped objective]]**：前者是 RLHF 组装篇（四模型+rollout+LLM 特化），后者是 PPO 算法本身（clip 公式）。本篇复用前者内核。
  - **RLHF-PPO vs DPO**：PPO 显式 rollout + RM + critic（重工程），DPO 闭式无 rollout（轻）。等价目标不同路径，见 [[DPO]]。
  - **RLHF-PPO vs SFT**：SFT 用示范教服从（监督），PPO 用 reward 教偏好（强化）；PPO 的 KL 锚点是 SFT。
  - **actor vs critic vs reference vs RM**：见 §3.2 表，角色与是否更新不同。
  - **token 级 ratio vs 序列级**：PPO 用 token 级，DPO 用序列级（整段 logp 比）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **reference/RM 没冻结**：$\pi_{\text{ref}}$、$r_\phi$ 应全程 `requires_grad_(False)` + 不进 optimizer；若随 actor 更新，KL 与 reward 失效。
> 2. **logp_old 没 detach**：采样快照应冻结，每 epoch 只重算当前 logp；漏 detach 让 ratio 恒 1，clip 失效（[[PPO clipped objective]] §7）。
> 3. **advantage 没 detach**：$\hat A$ 给 actor 时带 critic 梯度，actor 去最小化 $V$；detach。
> 4. **per-token KL 算成序列级**：应是 token 级 log-ratio，序列级丢细粒度信用分配（[[KL penalty与KL control]] §7）。
> 5. **reward 不 normalization**：RM 分数尺度漂移，advantage 失控；running std 归一化。
> 6. **critic 与 actor 共享 body 不分离梯度**：critic loss 梯度污染 actor；分离网络或 careful detach。
> 7. **K epoch 过大**：LLM-RL $K>4$ 时 clip_fraction 飙升、样本失真；取 2~4 + target_kl 早停。
> 8. **rollout 用 teacher forcing**：rollout 必须 actor **自生成**（autoregressive），不能用 ground-truth 前缀（那是 SFT）；否则分布偏移、PPO 无意义。
> 9. **γ 用 0.99 在长 response**：episodic 固定长度用 γ=1，衰减靠 λ；γ<1 会让远端 token 拿不到末尾 reward 信号。
> 10. **不看监控盲目训**：reward/KL/clip_fraction/length 是诊断必看；reward 涨但 KL 爆或 length 涨可能是 [[reward hacking]]/[[KL explosion]]。
> 11. **critic 太小或不预训练**：critic 不准则 advantage 噪声大、PPO 不稳；critic 用更大网络 + 预训练。
> 12. **四模型显存爆**：actor+critic+ref+RM 同显存，大模型需 FSDP/ZeRO/offload 切分（[[activation memory]]）。

## 8. 延伸细节

### 8.1 InstructGPT 的三阶段确立

OpenAI InstructGPT（2022）确立 SFT → RM → PPO 三阶段：
- SFT：13k 示范；
- RM：33k 偏好对；
- PPO：31k prompt 做强化；
- $\beta=0.01$（fixed 早期，后 adaptive），K=2，PPO clip ε=0.2。
- 这是现代 RLHF 的标准范式，所有开源框架遵循。

### 8.2 LLM-PPO 的显存账（4P）

四模型同显存：actor $P$ + critic $P$ + reference $P$ + RM $P$ = $4P$ 参数；加 optimizer state（actor+critic）+ activation + KV cache（rollout）→ RLHF 显存约 SFT 的 4 倍以上。是大模型 RLHF 的核心瓶颈，需 FSDP/ZeRO-3 切分 + offload + 重算（[[activation memory]]、[[gradient checkpointing]]）。

### 8.3 训推分离与 rollout 加速

rollout 生成是 LLM-PPO 的瓶颈（自回归生成慢）：
- 用 vLLM/SGLang 做 rollout（连续批处理、PagedAttention 加速）；
- rollout worker 与 trainer 分离（[[训推分离]]），生成与训练重叠；
- weight sync：actor 更新后同步给 rollout worker（[[weight sync mechanism]]，stale policy 问题）；
- Ray 调度（[[Ray基础]] 待展开）。
verl 是此架构的代表。

### 8.4 RLHF-PPO vs DPO 的工程权衡

| 维度 | RLHF-PPO | DPO |
|---|---|---|
| 工程 | 4 模型 + rollout + critic，重 | 1 模型 + 偏好数据，轻 |
| 算力 | 高（生成贵） | 低（无生成） |
| 灵活性 | 在线 RM 可不断标新数据 | 离线，固定偏好数据 |
| 探索 | 能学偏好数据外行为 | 不能 |
| 稳定性 | 易 KL 爆/reward hacking | 较稳但易过拟合偏好 |
| 适用 | 在线对齐、持续优化 | 离线对齐、快速出模型 |

现代实践：DPO 快速打底，PPO 在线精修；或 GRPO（PPO 变体，去 critic，组内相对 advantage）在数学推理场景崛起。

### 8.5 reward hacking 与 KL explosion 的 PPO 视角

- [[reward hacking]]：actor 钻 RM 漏洞拿高分但实际差。PPO 视角：reward 涨但人类评估降。防：RM 集成 + KL 约束 + 长度惩罚 + 在线 RM 更新。
- [[KL explosion]]：β 太小或 lr 太大，$\pi_\theta$ 漂离 ref，语言能力崩。PPO 视角：KL 曲线爆。防：adaptive β + target_kl 早停 + 小 lr。
- [[policy collapse]]：actor 塌缩到少数 mode。防：entropy bonus + KL。
三者是 PPO 失败的主要模式，详见第17节。

### 8.6 GRPO：PPO 在 LLM 的演进

GRPO（Group Relative Policy Optimization，DeepSeek 2024）：去掉 critic，用"同一 prompt 生成 G 条 response 的组内相对 advantage"（$\hat A_i=(r_i-\text{mean})/\text{std}$）。省 critic 显存，数学推理场景效果好（DeepSeekMath/R1）。是 PPO 在 LLM 的精简变体。详见 [[GRPO]]（待展开）。

### 8.7 监控仪表盘（PPO 调参核心）

- **reward**：应缓慢上升；
- **KL（to ref）**：维持 target 附近（adaptive β）；
- **clip_fraction**：0.1~0.3 健康；
- **response length**：稳定，若猛涨可能 length bias；
- **critic loss / explained variance**：critic 准度；
- **entropy**：缓降但不到 0；
- **KL（actor vs old）**：每 epoch 漂移，超 target 早停。
这些是 RLHF 调参的诊断信号，缺一不可。

---
相关: [[RLHF流程]]、[[SFT]]、[[Reward Model]]、[[PPO clipped objective]]、[[GAE]]、[[KL penalty与KL control]]、[[entropy bonus]]、[[value network]]、[[reward hacking]]、[[KL explosion]]、[[policy collapse]]、[[DPO]]、[[MDP]]、[[reward normalization]]、[[activation memory]]
