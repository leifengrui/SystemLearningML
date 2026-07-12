# trajectory generation

> **所属章节**: [[Rollout系统]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（需懂 RL 数据格式 + LLM 采样）

## 1. 一句话定义

**trajectory generation（轨迹生成）** 是 [[rollout worker]] 的核心任务：用当前 $\theta$ 在环境/提示上**采样生成训练轨迹**——记录 $(s,a,r,\log\pi,v)$ 序列，回传 learner 供 PPO 训练。RLHF 中轨迹 $=$ prompt → response（token 序列 + reward）。采样方式（greedy / temperature / top-k / top-p）控制**探索-利用**，决定数据多样性。它是 on-policy RL 的数据源头，与 [[inference worker]] 的部署生成不同——rollout 生成带 logp/value 供训练，inference 只返用户 output。

> [!note] trajectory vs inference output
> - **trajectory generation（本篇）**：训练侧采数据，记录 $(s,a,r,\log\pi,v)$ 全量训练信息。
> - **[[inference worker]]**：部署侧服务，只返生成 token 给用户。
> 前者重训练信息完整性（logp/value），后者重延迟/质量。

> [!note] 解答：为什么把这个东西叫"轨迹（trajectory）"？
> 这名字来自**经典强化学习的 MDP 形式化**，不是 LLM 圈新造的词。
>
> ### 一、源头：MDP 里的 trajectory
> 在马尔可夫决策过程（MDP）里，agent 在**时间维度**上一边走一边决策，走出的一条"路径"就叫 trajectory，记作 $\tau$：
> $$
> \tau=(s_0,a_0,r_1,s_1,a_1,r_2,\dots,s_T)
> $$
> 它本质是 $(s_t,a_t,r_{t+1})$ 这条**时序三元组序列**。"轨迹"是个**时空比喻**：agent 在**状态空间**里随**时间**移动留下的一条路径，就像物理里抛物运动在时空中画出的一道轨迹——只不过这里的"位置"是状态 $s$，"每一步的推力"是动作 $a$，"沿途收到的反馈"是奖励 $r$。所以 trajectory = agent 走过的一条带时间戳的路径。
>
> ### 二、为什么不直接叫"sample / output / 数据"
> 因为 trajectory 强调的不是"一段文字"，而是**带完整训练信息的时序决策序列**：
>
> | 叫法 | 隐含信息 | 够不够训练 |
> |---|---|---|
> | sample / output | 只有一段生成 token | ❌ 丢了 logp/value/reward |
> | sequence | 只是 token 串 | ❌ 没有"谁采的、概率多少、回报多少" |
> | **trajectory（轨迹）** | $(s,a,r,\log\pi,v)$ 全套 + 时序结构 | ✅ PPO 直接吃 |
>
> PPO 算 ratio $\rho=\pi_{\theta_{\text{new}}}/\pi_{\theta_{\text{old}}}$ 需要 $\log\pi_{\theta_{\text{old}}}$，算 advantage 需要 value $V(s_t)$，算回报需要 reward——这些只有 trajectory 这个概念容器才装得下。叫"输出"会让人误以为只要一段文字就够了。
>
> ### 三、LLM/RLHF 里"轨迹"对应什么
> RLHF 把 token 当 action，prompt 当初始状态，于是 trajectory 退化成 token 级：
> $$
> \tau = \text{prompt}(s_0)\to a_0\to a_1\to\dots\to a_T,\quad\text{带每 token 的 }(r_t,\log\pi_t,V_t)
> $$
> 也就是说，"prompt→response"不是一段静态文本，而是**策略 $\pi_\theta$ 在语言状态空间里走出的那条路径**。每生成一个 token = agent 在那个状态选了一个 action = 走了一步。整条 response 就是这条轨迹。
>
> ### 四、易混澄清
> - **trajectory = episode = rollout = 一次采样**：四个词在 RLHF 里基本同义，都指"一条从开始到结束的采样序列"，只是不同社区习惯。
>   - trajectory：偏 RL 理论，强调时序状态-动作路径（本篇用此，贴合 PPO 形式化）。
>   - episode：偏控制论/游戏，强调"一回合"。
>   - rollout：偏工程，强调"用当前策略跑一遍采数据"这个动作。
> - **trajectory vs transition**：trajectory 是**整条**序列 $\tau$；transition 是**一步** $(s_t,a_t,r_{t+1},s_{t+1})$。一条 trajectory 由多个 transition 组成。
> - **trajectory vs batch**：trajectory 是**一条**采样；batch 是**一批** trajectory 凑成训练 batch。
>
> ### 五、一句话记忆
> **trajectory（轨迹）= agent 在状态空间里随时间走出的一条带训练信息的路径。** 叫"轨迹"而不是"输出"，是因为它要装的不只是文字，还有"每一步选了啥、概率多大、回报多少、价值几何"——这些才是 PPO 能算的料。
>
> 相关：[[trajectory generation]]、[[rollout worker]]、[[MDP]]、[[PPO clipped objective]]、[[GAE]]

## 2. 为什么需要它（动机与背景）

on-policy RL 必须用当前 $\theta$ 采数据：
1. **on-policy 本质**：PPO 的 ratio $\rho=\pi_{\theta_{\text{new}}}/\pi_{\theta_{\text{old}}}$ 需 $\theta_{\text{old}}$ 是采数据的策略，故必须当前 $\theta$ 采；
2. **探索需求**：greedy 生成无多样性，训练数据单一，策略无法改进；需采样（temperature/top-k/top-p）探索；
3. **训练信息完整**：需记录 $\log\pi$（算 ratio）、value（算 advantage）、reward（信号），非只 output；
4. **RLHF 的轨迹**：prompt → response → reward（RM 打分），整条轨迹训练；
5. **批量生成**：需多样本并行（高吞吐），见 [[sampling throughput]]。

trajectory generation 是 RLHF 数据流的起点：rollout worker 用 $\theta$ 生成 $\to$ 打包 trajectory $\to$ 回传 learner。它的质量（多样性、logp 准确性）直接决定 PPO 训练效果。

## 3. 核心概念详解

### 3.1 轨迹的格式

通用 RL 轨迹：
$$
\tau=\{(s_t,a_t,r_t,\log\pi_\theta(a_t|s_t),V_\theta(s_t))\}_{t=0}^{T}
$$

RLHF 轨迹（token 级）：
| 字段 | 说明 |
|---|---|
| **prompt** | 输入 token 序列（$s_0$） |
| **response** | 生成 token 序列（$\{a_t\}$） |
| **logp** | 每token $\log\pi_\theta(a_t\|s_t)$（算 ratio） |
| **value** | 每token $V_\theta(s_t)$（算 advantage） |
| **reward** | 终止 reward（RM 打分）+ per-token penalty（KL to ref） |
| **done** | 终止标志 |

### 3.2 采样方式（探索-利用）

| 方式                      | 机制                                         | 探索性    | 用途            |
| ----------------------- | ------------------------------------------ | ------ | ------------- |
| **greedy**              | $\arg\max$ 最高概率                            | 无（确定性） | 评估/部署，**不训练** |
| **temperature**         | $\text{softmax}(\text{logits}/T)$，$T$ 大越随机 | $T$ 大高 | 训练采数据         |
| **top-k**               | 仅 top-k token 采样                           | 中      | 限低概率噪声        |
| **top-p (nucleus)**     | 累积概率 $p$ 内采样                               | 中      | 动态裁剪长尾        |
| **top-p + temperature** | 先 top-p 再温度缩放                              | 可调     | 训练常用          |

训练采数据用 top-p + temperature（如 $p=0.9,T=0.7\sim1.0$），平衡多样性与质量。greedy 无探索，数据单一，策略无法改进。

### 3.3 生成的流程

```
rollout worker 流程:
1. 接收 θ(从 learner,weight sync)              ← [[weight sync mechanism]]
2. 加载 θ 到推理引擎(vLLM)                       ← 高吞吐生成
3. 从 prompt 池(batch)采样:
   a. prompt → 前向 → logits
   b. 采样 next token(top-p+temperature)         ← 探索
   c. 记录 logp(算 ratio 用)
   d. 重复到 done/maxLength
4. RM 打分 response(终止 reward)                 ← [[Reward Model]]
5. (可选)KL to ref penalty(per-token)            ← [[KL penalty与KL control]]
6. (可选)Critic 算 value(per-token)              ← GAE 用
7. 打包 trajectory (s,a,r,logp,v)
8. 回传 learner(数据 sync)                       ← [[asynchronous training]]
```

### 3.4 RLHF 的轨迹细节

- **prompt 池**：通常来自 SFT/预训练数据，或线上反馈数据；
- **response 生成**：vLLM 高吞吐生成（continuous batching）；
- **reward**：RM 对整条 response 打分（终止 reward），或 token 级（如格式奖励）；
- **KL penalty**：每 token 加 $-\beta\log(\pi_\theta/\pi_{\text{ref}})$，防偏离 ref（[[KL penalty与KL control]]）；
- **value**：Critic 对每 token 算 $V(s_t)$，GAE 用（[[GAE]]）。

### 3.5 多样本并行

为高吞吐（见 [[sampling throughput]]），rollout worker 同时生成多条轨迹：
- **batch 生成**：多 prompt 一起前向（continuous batching）；
- **并行 worker**：多 GPU/worker 各生成（数据并行 rollout）；
- **异步生成**：worker 生成与 learner 训练重叠（[[asynchronous training]]）。

## 4. 数学原理 / 公式

### 4.1 采样概率

$$
\pi_\theta(a_t|s_t)=\text{softmax}(\text{logits}_\theta(s_t)/T)_{a_t}
$$

- $T\to0$：greedy（$\arg\max$）；
- $T=1$：原始 softmax；
- $T$ 大：越均匀随机（高探索）。

### 4.2 top-p 截断

$$
\mathcal{V}_p=\min\{V: \sum_{a\in V}\pi_\theta(a|s_t)\ge p\},\quad \pi'_\theta(a|s_t)=\frac{\pi_\theta(a|s_t)\mathbb{1}[a\in\mathcal{V}_p]}{\sum_{a\in\mathcal{V}_p}\pi_\theta(a|s_t)}
$$

累积概率 $\ge p$ 的最小集合，截掉长尾低概率 token，再归一化采样。

### 4.3 logp 的记录

$$
\log\pi_\theta(a_t|s_t)=\log\text{softmax}(\text{logits}_\theta(s_t)/T)_{a_t}
$$

必须记录采样时的 $\log\pi$，PPO 算 ratio $\rho=\exp(\log\pi_{\theta_{\text{new}}}-\log\pi_{\theta_{\text{old}}})$ 用。

### 4.4 reward 的组成

RLHF per-token reward：
$$
r_t=\begin{cases}-\beta\log\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} & t<T\text{（KL penalty）}\\ R_{\text{RM}}(\text{response}) & t=T\text{（终止 reward）}\end{cases}
$$

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Policy(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)
    def logp_and_logits(s, x, a):
        logits = s(x); logp = F.log_softmax(logits,-1).gather(-1,a.unsqueeze(-1)).squeeze(-1); return logp, logits

class Critic(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,1)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h).squeeze(-1)

class RM(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,1)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h[:,-1,:]).squeeze(-1)

V, D = 50, 32
policy, critic, rm = Policy(V,D), Critic(V,D), RM(V,D)
ref = Policy(V,D); ref.load_state_dict(policy.state_dict())  # 冻结 ref

# ===== trajectory generation =====
def top_p_sample(logits, T=0.7, p=0.9):
    """top-p + temperature 采样(探索)"""
    probs = F.softmax(logits/T, -1)
    sorted_p, sorted_i = probs.sort(descending=True)
    cum = sorted_p.cumsum(-1); mask = (cum - sorted_p) < p     # 保留累积<p 的
    sorted_p = sorted_p * mask
    sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)       # 归一化
    idx = torch.multinomial(sorted_p, 1)
    return sorted_i.gather(-1, idx), probs.gather(-1, idx)    # token, 概率

def generate_trajectory(policy, critic, rm, ref, prompt, max_len=8, T=0.7, p=0.9, beta=0.05):
    """生成一条 RLHF 轨迹"""
    policy.eval(); out = prompt.clone(); logps, values, rewards = [], [], []
    with torch.no_grad():
        for t in range(max_len):
            logits = policy(out)[:, -1, :]
            a, prob = top_p_sample(logits, T, p)                # 采样(探索)
            logp_old = torch.log(prob + 1e-10)                  # 记录 logp(算 ratio 用)
            logp_ref = F.log_softmax(ref(out)[:, -1, :], -1).gather(-1, a).squeeze(-1)
            kl_pen = -beta * (logp_old - logp_ref)              # KL penalty
            out = torch.cat([out, a], 1)
            logps.append(logp_old); values.append(critic(out)[:, -1]); rewards.append(kl_pen)
        rewards[-1] = rewards[-1] + rm(out)                     # 终止 + RM reward
    traj = {
        "prompt": prompt, "response": out[:, prompt.shape[1]:],
        "logp": torch.stack(logps, 1), "value": torch.stack(values, 1),
        "reward": torch.stack(rewards, 1), "done": True
    }
    return traj

# 生成一条
prompt = torch.randint(0, V, (1, 4))
traj = generate_trajectory(policy, critic, rm, ref, prompt)
print(f"轨迹: prompt{traj['prompt'].shape} → response{traj['response'].shape}")
print(f"logp{traj['logp'].shape} value{traj['value'].shape} reward{traj['reward'].shape}")
print(f"终止 reward={traj['reward'][0,-1]:.3f}(含 RM)")
print("trajectory 回传 learner,供 PPO 训练")

# 多样本并行(batch 生成)
batch = torch.randint(0, V, (8, 4))
trajs = [generate_trajectory(policy, critic, rm, ref, b.unsqueeze(0)) for b in batch]
print(f"batch 生成 {len(trajs)} 条轨迹(高吞吐)")
```

> [!tip] 实际工程
> - **生成引擎**：vLLM（continuous batching + PagedAttention）高吞吐生成；
> - **logp 提取**：vLLM 支持返回 logprob，或生成后重算；
> - **并行**：多 GPU worker 各生成，verl/OpenRLHF 支持；
> - **prompt 池**：从 SFT/线上数据采样，保证多样性。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[rollout worker]]（执行单元）、[[policy training (PT)]]/learner（$\theta$ 来源，weight sync）、[[vLLM基础]]/[[continuous batching]]（生成引擎）、[[Reward Model]]、[[KL penalty与KL control]]、[[GAE]]/[[Critic]]（value 估计）。
- **下游（应用）**: [[sampling throughput]]（生成性能）、[[batching strategy]]（批次组织）、[[PPO clipped objective]]（消费 trajectory）、[[asynchronous training]]（生成-训练重叠）、[[stale policy problem]]（生成时 $\theta$ 滞后）。
- **对比 / 易混**:
  - **trajectory generation vs [[inference worker]]**：训练采数据（带 logp/value） vs 部署服务（只 output）。见 §1 note。
  - **trajectory generation vs inference generation**：前者记录训练信息，后者只返 token。
  - **greedy vs sampling**：greedy 无探索（评估用），sampling 探索（训练用）。
  - **trajectory vs episode**：同义（轨迹 = 一条 episode）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"rollout = inference"**：错。rollout 记录 logp/value/reward 供训练，inference 只返 output。
> 2. **"greedy 能训练"**：错。greedy 无探索，数据单一，策略无法改进。需 sampling。
> 3. **"不用记 logp"**：错。PPO ratio 需 $\log\pi_{\theta_{\text{old}}}$，不记则无法算。
> 4. **"logp 用新 θ 算"**：错。logp 必须用采样时的 $\theta_{\text{old}}$（否则 ratio=1 失意义）。
> 5. **"temperature 越高越好"**：错。过高数据噪声大，过低无探索。典型 $0.7\sim1.0$。
> 6. **"reward 只在终止"**：不一定。RLHF 常有 per-token KL penalty + 终止 RM reward。
> 7. **"生成不瓶颈"**：错。生成是 RLHF 主要耗时（auto-regressive），需 vLLM 优化（[[sampling throughput]]）。
> 8. **"prompt 池随便"**：错。prompt 多样性决定策略泛化，需 SFT/线上数据平衡。
> 9. **"value 用 policy 算"**：错。value 用独立 Critic（防 policy 自评偏差）。

## 8. 延伸细节

### 8.1 vLLM 生成 + logp 提取

- vLLM 生成时可选返回每 token 的 logprob（`top_logprobs`）；
- 或生成后用 policy 重算 logp（多一次前向，但准确）；
- 工程上 vLLM 内置 logprob 更省（不重算）。

### 8.2 多样本并行的吞吐

- **batch 生成**：vLLM continuous batching 多 prompt 一起（[[sampling throughput]]）；
- **并行 worker**：多 GPU 各生成，汇总（数据并行 rollout）；
- **异步生成**：worker 生成时 learner 训上一批（[[asynchronous training]]）；
- 吞吐是 RLHF 效率关键，详见 [[sampling throughput]]。

### 8.3 off-policy 修正

若用旧 $\theta$ 采的数据训新 $\theta$（异步/复用数据），需 importance sampling 修正：
$$
\rho=\frac{\pi_{\theta_{\text{new}}}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}
$$

PPO clip 截断 $\rho$，V-trace 截断容忍 stale（[[stale policy problem]]）。这是复用旧数据的代价。

### 8.4 探索-利用的调参

- **$T$ 低（0.3）**：近 greedy，数据质量高但多样性低，策略易局部最优；
- **$T$ 高（1.0+）**：高探索，数据多样但噪声大，训练不稳；
- **$p$ 低（0.8）**：裁剪狠，质量高多样性低；
- **$p$ 高（0.95）**：宽容，多样性高；
- 典型：$T=0.7\sim1.0,p=0.9$，监控 entropy 调（防 [[policy collapse]]）。

### 8.5 trajectory 的存储与回传

- **回传 learner**：trajectory 打包（prompt/response/logp/value/reward）传 learner；
- **[[replay buffer]]**：可选存 buffer 供复用（off-policy 修正）；
- **压缩**：logp/value 浮点数据，可 FP16/INT8 减带宽；
- **数据 sync 带宽**：trajectory 比 $\theta$ 小但仍可观（多 token × 多字段）。

### 8.6 生成与训练的稳定性

- **stale logp**：异步中 logp 用旧 $\theta$，ratio 失真（[[stale policy problem]]）；
- **entropy 塌缩**：$T$ 过低或训练后期 entropy → 0，[[policy collapse]]；
- **KL 漂移**：reward 信号推 $\theta$ 偏 ref，KL penalty 平衡（[[KL penalty与KL control]]）。
trajectory generation 的参数直接影响训练稳定性（[[训练稳定性工程]]）。

---
相关: [[Rollout系统]]、[[rollout worker]]、[[sampling throughput]]、[[batching strategy]]、[[policy training (PT)]]、[[asynchronous training]]、[[stale policy problem]]、[[weight sync mechanism]]、[[PPO clipped objective]]、[[GAE]]、[[Reward Model]]、[[KL penalty与KL control]]、[[continuous batching]]、[[vLLM基础]]、[[replay buffer]]、[[policy collapse]]、[[训练稳定性工程]]
