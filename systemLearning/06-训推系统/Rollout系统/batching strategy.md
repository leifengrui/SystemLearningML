# batching strategy

> **所属章节**: [[Rollout系统]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（需懂 PPO minibatch + 显存管理）

## 1. 一句话定义

**batching strategy（批次策略）** 是 [[rollout worker]] 生成与 [[policy training (PT)]]/learner 消费的**批次组织策略**：包括 batch 层级（rollout batch → learner batch → PPO minibatch）、**长度分桶**（length bucketing 减 padding 浪费）、动态 padding、rollout-learner batch 对齐、batch size 调优（吞吐 vs 稳定性 vs 显存）。好的 batching strategy 提 [[sampling throughput]]、稳 PPO 训练、省显存。它是 [[trajectory generation]] 产数据到 [[PPO clipped objective]] 消费数据的"中间调度"，verl/OpenRLHF 都有 batch 配置参数（`rollout_batch_size`/`train_batch_size`/`ppo_mini_batch_size`/`ppo_micro_batch_size`）。

> [!note] batch 层级
> - **rollout batch**：rollout worker 一次生成的 prompt 数（生成并发）。
> - **learner batch**：learner 一次 PPO 更新消费的 trajectory 数（= rollout batch 或其倍数）。
> - **PPO minibatch**：learner batch 切分成 K 份，每份一次梯度更新（SGD-style）。
> - **micro batch**：minibatch 再切分进 GPU 显存（gradient accumulation）。

## 2. 为什么需要它（动机与背景）

batch 组织影响吞吐、稳定性、显存三方面：
1. **吞吐**：大 batch 提 GPU 利用率（[[sampling throughput]]），但需长度分桶减 padding；
2. **稳定性**：PPO minibatch size 影响梯度噪声，过大稳但慢，过小快但噪；
3. **显存**：micro batch 限 GPU 显存（前向+反向+KV cache），需 gradient accumulation；
4. **rollout-learner 对齐**：rollout batch 与 learner batch 不对齐则数据 sync 浪费/等待；
5. **长度异构**：LLM 轨迹长度差异大，需分桶+动态 padding。

故 batching strategy 是连接生成与训练的调度核心，参数调优直接影响 RLHF 效率与稳定。

## 3. 核心概念详解

### 3.1 batch 层级（自顶向下）

| 层级 | 含义 | 决定 |
|---|---|---|
| **rollout batch** | rollout 一次生成的 prompt 数 | 生成并发、KV cache 显存 |
| **learner batch** | 一次 PPO 更新的 trajectory 数 | PPO 期望估计精度、稳定性 |
| **PPO minibatch** | learner batch 切 K 份，每份一次梯度 | SGD 步数、梯度噪声 |
| **micro batch** | minibatch 再切分进 GPU | GPU 显存（前向+反向+KV） |

例：rollout_batch=512, learner_batch=512, minibatch=64（K=8 次梯度）, micro_batch=4（每 minibatch 累积 16 次）。

### 3.2 长度分桶（length bucketing）

LLM 轨迹长度差异大（如 100~2000 token），若混 batch padding 到最长，浪费严重：
- **分桶**：按长度分桶（如 [0-200], [200-500], [500-1000]...），桶内 padding 到桶长；
- **动态 padding**：每 batch padding 到该 batch 最长，非全局最长；
- **增益**：减 padding 计算 30~50%；
- **vLLM**：continuous batching 内置动态长度，无显式分桶。
> [!note] 解答：为什么要 padding 到桶长（而不是别的长度）
> 先说**为什么要 padding**：GPU 批处理要求一个 batch 内所有序列**等长**（张量是矩形 `[B, L]`，不等长没法拼成张量送进 kernel）。所以短序列必须补到某个长度 $L_{\text{pad}}$，补的那些位是无效计算（pad token，注意力要 mask 掉不参与梯度）。padding 是"为了能批处理"付出的浪费。
>
> 那为什么是 padding 到**桶长**，而不是 padding 到 **batch 内最长**或**全局最长**？三种策略对比：
>
> | 策略 | $L_{\text{pad}}$ 取谁 | 浪费率 | 问题 |
> |---|---|---|---|
> | **全局最长** | 整个数据集最长序列 | 极高（$\bar L=100$ 但 $L_{\max}=2000$ → waste 95%） | 一条超长样本拖累所有 batch |
> | **batch 内最长**（动态 padding） | 本 batch 最长那条 | 中等（取决于 batch 内长度分布） | 长短混在一个 batch 仍浪费；长度异构时 batch 内最长也很长 |
> | **桶长**（length bucketing） | 该桶的预设上界 | **最低**（桶内长度接近，$L_{\max}\approx\bar L$） | 需先分桶、桶间 batch size 可能不同 |
>
> 桶长的核心逻辑：**让一个 batch 里的序列长度尽量接近**，这样 padding 到桶上界时几乎不浪费。做法是**先按长度分桶**（如 `[0-200], [200-500], [500-1000], ...`），每个 batch 只从**同一个桶**里采样，桶内序列最长也就到桶上界，短的不用补太长。
>
> 举个例子，8 条轨迹长度 `[5,8,10,15,50,100,150,200]`：
> - **不分桶混 batch**：若随机抽到 `{5, 200}` 一组，pad 到 200 → waste = (200-5)+(200-200) / 400 ≈ 49%。
> - **分桶**：短桶 `{5,8,10,15}` pad 到 15 → waste ≈ 38%；中桶 `{50,100}` pad 到 100 → waste 25%；长桶 `{150,200}` pad 到 200 → waste 12.5%。整体 waste 大降。
>
> 公式上（见 §4.1）：
> $$\text{waste} = 1 - \frac{\bar L}{L_{\max}}$$
> 桶长的目标就是把桶内的 $L_{\max}$ 压到接近 $\bar L$，让 waste 趋近 0。论文/实测可省 30~50% padding 计算。
>
> **额外好处（静态 shape）**：桶内长度接近 → 同桶的 batch shape 几乎固定，便于编译优化（torch.compile / CUDA Graph 喜欢静态 shape，动态 shape 会触发重编译、掉性能）。分桶是"减 padding 浪费"和"保静态 shape 利于编译"的双赢。
>
> **为什么 vLLM 不用显式分桶**：vLLM 的 [[continuous batching]] + PagedAttention 把 KV cache 切成页（block），按 token 实际长度分配物理页，**物理上不存在 padding**（每个 token 占自己该占的页，不补齐），所以不需要桶。但传统框架（Megatron 训练侧、早期 serving）张量是矩形必须等长，就得靠分桶。所以工程上常见组合：**rollout 用 vLLM（无 padding）+ learner 端分桶**（若 learner 是传统框架）。
>
> **一句话**：padding 是为批处理凑等长；padding 到桶长（而非全局/batch 最长）是为了让同 batch 长度接近、把浪费率 `1-ḹ/L_max` 压到最低，顺带换静态 shape 利于编译。vLLM 靠 paged 内存从物理上消除了 padding，传统框架则靠分桶。

### 3.3 rollout-learner batch 对齐

- **对齐**：rollout batch = learner batch，rollout 生成完直接 learner 消费，无等待；
- **倍数**：learner batch = N × rollout batch（攒多批再训），减 PPO 噪声但增 stale；
- **异步**：rollout 持续生成，learner 按需取（[[asynchronous training]]），不强对齐；
- **tradeoff**：对齐减等待增同步开销；倍数减噪声增 stale。

### 3.4 PPO minibatch 的作用

learner batch 切 K 个 minibatch，每 minibatch 一次梯度更新（K 次 SGD）：
- **多 epoch**：每 minibatch 可多 epoch 复用（PPO 典型 4 epoch）；
- **clip 防偏离**：多 epoch 复用旧数据，ratio 偏离，clip 截断（[[PPO clipped objective]]）；
- **size tradeoff**：minibatch 大→稳但步数少；小→快但噪声大；
- **典型**：learner_batch=512, minibatch=64, epoch=4 → 32 次梯度更新。

### 3.5 batch size 调优

| 因素 | 大 batch | 小 batch |
|---|---|---|
| **吞吐** | 高（GPU 利用率高） | 低 |
| **稳定性** | 高（期望估计准） | 低（噪声大） |
| **显存** | 紧（需 micro accumulation） | 松 |
| **收敛** | 慢（步数少） | 快（步数多） |
| **stale** | 重（生成慢，lag 长） | 轻 |

权衡：rollout/learner batch 大（吞吐+稳定），minibatch 适中（步数+稳定），micro 限显存。

### 3.6 框架的 batch 配置（verl 为例）

```
verl config:
  data.rollout_batch=512       # rollout 一次生成
  ppo.train_batch_size=512     # learner 一次消费
  ppo.mini_batch_size=64       # PPO minibatch
  ppo.micro_batch_size=4       # GPU 显存限
  ppo.epochs=4                 # 每 minibatch epoch 数
```

OpenRLHF/DeepSpeed-Chat 类似参数。

## 4. 数学原理 / 公式

### 4.1 padding 浪费

$$
\text{waste}=\frac{\sum_{i=1}^{B}(L_{\max}-L_i)}{\sum_{i=1}^{B}L_{\max}}=1-\frac{\bar L}{L_{\max}}
$$

- $L_{\max}$：batch 最长，$\bar L$：平均；
- 不分桶：长度差异大则 waste 高（如 $\bar L=100,L_{\max}=2000$，waste=95%）；
- 分桶：桶内 $L_{\max}\approx\bar L$，waste 低。

### 4.2 PPO minibatch 的梯度噪声

$$
\text{Var}(\nabla)\propto\frac{1}{B_{\text{minibatch}}}
$$

- minibatch 大→方差小→稳但步数少；
- minibatch 小→方差大→噪但步数多；
- 多 epoch 复用：每 epoch 同 minibatch 多次梯度，但 ratio 偏离需 clip。

### 4.3 batch 吞吐

$$
\text{throughput}\propto\frac{B\cdot\text{util}(B)}{\text{time}(B)}
$$

- $B$ 大→util 高（GPU 利用率）但 time 增；
- 最优 $B$：util 接近饱和且 time 线性增长前。

### 4.4 gradient accumulation

$$
g=\frac{1}{M}\sum_{m=1}^{M}g_m\text{（micro batch 累积）}
$$

- micro batch $m$ 算梯度 $g_m$，累积 $M$ 次平均；
- 等效 batch size = micro × M；
- 解决显存放不下大 batch 的问题。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Policy(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)

V, D = 50, 32
policy = Policy(V, D)

# ===== 长度分桶 =====
def length_bucket(trajs, n_buckets=3):
    """按长度分桶,桶内 padding 减浪费"""
    lengths = [t.shape[1] for t in trajs]
    sorted_i = sorted(range(len(trajs)), key=lambda i: lengths[i])
    buckets = [sorted_i[i::n_buckets] for i in range(n_buckets)]  # 简化:交错分桶
    batched = []
    for b in buckets:
        if not b: continue
        maxL = max(lengths[i] for i in b)
        padded = torch.full((len(b), maxL), -1, dtype=torch.long)  # -1 = pad
        for j, i in enumerate(b):
            t = trajs[i]; padded[j, :t.shape[1]] = t
        batched.append((padded, b))
    return batched

trajs = [torch.randint(0, V, (1, l)) for l in [5, 50, 10, 200, 8, 100, 15, 150]]
buckets = length_bucket(trajs, n_buckets=3)
print("长度分桶:")
for padded, idx in buckets:
    print(f"  桶 {len(idx)} 条, max_len={padded.shape[1]}, waste={1-(sum(trajs[i].shape[1] for i in idx)/len(idx))/padded.shape[1]:.0%}")

# ===== PPO minibatch 切分 =====
def ppo_update(policy, batch, logp_old, adv, minibatch_size=64, epochs=4, eps=0.2):
    """learner batch 切 K minibatch,每 minibatch 多 epoch 梯度"""
    B = batch.shape[0]
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    for epoch in range(epochs):
        perm = torch.randperm(B)
        for i in range(0, B, minibatch_size):
            idx = perm[i:i+minibatch_size]
            mb_x, mb_logp_old, mb_adv = batch[idx], logp_old[idx], adv[idx]
            logits = policy(mb_x); logp = F.log_softmax(logits,-1).gather(-1, mb_x).squeeze(-1)
            ratio = (logp - mb_logp_old).exp()
            surr = torch.min(ratio*mb_adv, ratio.clamp(1-eps,1+eps)*mb_adv)
            loss = -surr.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return policy

# 模拟 learner batch
B = 16
batch = torch.randint(0, V, (B, 8))
logp_old = torch.randn(B, 8).log_softmax(-1)
adv = torch.randn(B, 8)
policy = ppo_update(policy, batch, logp_old, adv, minibatch_size=4, epochs=2)
print(f"PPO 更新完成: batch={B} → minibatch=4(K=4) × epoch=2 = 8 次梯度")

# ===== gradient accumulation(显存不够时) =====
def update_with_accum(policy, batch, micro_size=2, accum_steps=4):
    """micro batch 累积,等效 batch = micro × accum"""
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    opt.zero_grad()
    for i in range(0, batch.shape[0], micro_size):
        mb = batch[i:i+micro_size]
        loss = policy(mb).float().mean() / accum_steps  # 缩放
        loss.backward()                                   # 累积梯度
    opt.step()                                            # 等效 batch=micro×accum
    return policy

policy = update_with_accum(policy, batch, micro_size=2, accum_steps=4)
print(f"gradient accumulation: micro=2 × accum=4 = 等效 batch 8(显存只装 micro=2)")
print("batching strategy: 分桶减 waste + minibatch 切 K + micro accumulation")
```

> [!tip] 实际工程
> - **verl/OpenRLHF**：配置 `rollout_batch/train_batch/mini_batch/micro_batch`；
> - **vLLM**：continuous batching 内置动态长度，无需显式分桶；
> - **PPO epoch**：典型 4，超则 ratio 偏离（clip freq 高）；
> - **micro accumulation**：显存不够时用，等效大 batch。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[trajectory generation]]（产数据）、[[sampling throughput]]（生成性能）、[[rollout worker]]/[[policy training (PT)]]（执行单元）、[[PPO clipped objective]]（消费 minibatch）、[[Data Parallel]]/gradient accumulation。
- **下游（应用）**: RLHF 框架 batch 配置（verl/OpenRLHF/DeepSpeed-Chat）、PPO 训练稳定性、显存管理。
- **对比 / 易混**:
  - **batching strategy（本篇）vs [[sampling throughput]]**：前者讲批次组织（分桶/层级/对齐），后者讲生成性能（tokens/sec）。互补。
  - **PPO minibatch vs gradient accumulation**：前者是 SGD 多步（每步独立梯度），后者是显存不够模拟大 batch（累积后一次更新）。不同。
  - **rollout batch vs learner batch**：前者生成并发，后者训练消费。可对齐或倍数。
  - **length bucketing vs continuous batching**：前者显式分桶（传统），后者动态长度（vLLM 内置）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"batch 越大越好"**：错。大 batch 吞吐高但显存紧 + stale 重 + 收敛慢，有最优。
> 2. **"padding 无害"**：错。padding 是无效计算，长度异构时浪费严重（可达 90%+），需分桶。
> 3. **"rollout-learner 无需对齐"**：错。不对齐则数据 sync 等待/浪费，影响效率。
> 4. **"PPO minibatch = gradient accumulation"**：错。前者多步独立梯度（SGD），后者累积一次更新（模拟大 batch）。
> 5. **"minibatch 越小越好"**：错。小则噪声大，训练不稳；大则步数少，收敛慢。有 tradeoff。
> 6. **"epoch 越多越好"**：错。PPO 多 epoch 复用旧数据，ratio 偏离，clip 频发，需 target_kl 早停。
> 7. **"分桶 vLLM 不用"**：对（vLLM continuous 内置动态长度），但传统框架需显式分桶。
> 8. **"micro batch = minibatch"**：错。micro 是显存限制的切片（gradient accumulation），minibatch 是 PPO 的梯度单元。
> 9. **"batch size 与稳定性无关"**：错。batch 影响梯度噪声 + stale，直接关系 [[KL explosion]]/[[policy collapse]]。

## 8. 延伸细节

### 8.1 verl 的 batch 配置详解

verl（参考）：
- `data.rollout_batch=512`：rollout 一次生成 512 prompt；
- `ppo.train_batch_size=512`：learner 消费 512 trajectory（对齐）；
- `ppo.mini_batch_size=64`：切 8 份 minibatch；
- `ppo.micro_batch_size=4`：每 minibatch 累积 16 次（显存限）；
- `ppo.epochs=4`：每 minibatch 4 epoch；
- 总梯度步数 = 8 × 4 = 32 per PPO round。
OpenRLHF/DeepSpeed-Chat 类似。

### 8.2 PPO minibatch 与 clip 的关系

- 每 epoch 复用旧数据，ratio $\rho=\pi_{\theta_{\text{new}}}/\pi_{\theta_{\text{old}}}$ 偏离；
- clip 截断超 $[1-\epsilon,1+\epsilon]$ 的样本梯度；
- epoch 多 → ratio 偏离大 → clip freq 高 → 有效梯度信号弱；
- **target_kl 早停**：每 epoch 算 KL，超 target（如 0.05）则 break，防过度偏离（[[KL penalty与KL control]]）。

### 8.3 长度分桶的实现

- **传统框架**：数据预处理时按长度分桶，每 batch 从同桶采样；
- **vLLM**：continuous batching 动态长度，无显式分桶（PagedAttention 支持）；
- **混合**：rollout 用 vLLM（动态），learner 端可分桶（若传统框架）；
- **tradeoff**：分桶减 padding 但增批次异构（不同桶 batch size 不同）。

### 8.4 动态 shape 与编译优化

- **静态 shape**：batch 长度固定，编译优化好（如 torch.compile）；
- **动态 shape**：长度异构，编译优化难；
- **分桶 + 静态**：桶内 shape 近静态，兼顾优化与 padding；
- **vLLM**：内部处理动态 shape，用户无感。

### 8.5 batch 与训练稳定性的连锁

- batch 过小 → 梯度噪声大 → PPO 更新抖 → [[KL explosion]]/[[policy collapse]]；
- batch 过大 → stale 重（生成慢） → ratio 失真 → 同样不稳；
- minibatch epoch 多 → ratio 偏离 → clip 频发 → 信号弱；
- 故 batching strategy 是 RLHF 稳定性工程的调参点（[[训练稳定性工程]]），与 target_kl/clip 协同。

### 8.6 异步训练的 batch 策略

- **异步**：rollout 持续生成，learner 按需取（[[asynchronous training]]）；
- **batch 不对齐**：rollout batch 灵活，learner 取可用数据组 batch；
- **stale 管理**：buffer 容量限最大 stale（旧数据过期）；
- **tradeoff**：异步提吞吐但增 stale，需 weight sync + clip 协同（[[stale policy problem]]）。

---
相关: [[Rollout系统]]、[[trajectory generation]]、[[sampling throughput]]、[[rollout worker]]、[[policy training (PT)]]、[[PPO clipped objective]]、[[KL penalty与KL control]]、[[Data Parallel]]、[[asynchronous training]]、[[stale policy problem]]、[[continuous batching]]、[[KV cache management]]、[[KL explosion]]、[[policy collapse]]、[[训练稳定性工程]]
