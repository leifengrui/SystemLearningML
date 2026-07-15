# sequence packing与动态采样

> **所属章节**: [[采样工程]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: sequence packing / 序列打包 / padding→packing / 变长样本负载均衡 / Dynamic Sampling / 动态采样 / DAPO 动态采样 / 过滤全对全错组 / timely rollout / RL 采样工程优化
> **难度**: 中高（需懂 [[trajectory generation]]、[[synchronous与asynchronous rollout]]、[[GRPO]]、[[DAPO]]、[[advantage function]]、[[batching strategy]]、[[continuous batching的调度实现]]、[[chunked prefill]]、[[reward hacking]]）

> **所属小节**: §85


## 1. 一句话定义

**sequence packing 与动态采样** 是 [[17-RL训推一体框架]] **采样工程侧的两个吞吐与有效性优化**——① **sequence packing（序列打包）**：把**多条变长 response 拼进一个定长序列**（把传统的 **padding 改成 packing**），用 **block-diagonal attention mask** 隔离各 trajectory（让 trajectory A 的 token 不 attend trajectory B），消除 padding 浪费、提升 token 密度（useful tokens/total → ~100%）、把长短样本混 pack 做负载均衡；它与训练侧 packing（如预训练 packing）的区别是 **RL 还要 per-trajectory 隔离 logp/reward/advantage**（每条打包的 response 保留自己的 outcome reward、GRPO 组内相对 advantage、`old_log_prob`，不能混），否则 IS ratio 和 advantage 算错。② **Dynamic Sampling（动态采样，DAPO 核心技术之一）**：当一批采样结果**全对（accuracy=1）或全错（accuracy=0）**——即 group 内 reward 全相同、$\sigma_r=0$、advantage $(r_i-\bar r)/\sigma_r=0/0$ 无信号、梯度为零——**整组丢弃重采**，**采够"有效样本"（advantage 非零，即 $0<\text{正确数}<G$）才训练**；过滤时机分 **pre-filter（进 batch 前在 rollout 阶段当场过滤）** vs **post-filter（累积成 batch 后再过滤）**，**timely rollout** 特指在 rollout 阶段当场做过滤（不浪费训练算力在无信号样本上）。两者联合解决 LLM-RL 采样的两大浪费——**padding 的算力浪费**与**全对/全错组的梯度信号浪费**。是 [[sampling throughput]] 与 [[GRPO]]/[[DAPO]] 的采样工程展开。

> [!note] 三句话定位
> - **是什么**：packing 把多条变长 response 拼进定长序列 + block-diagonal mask 隔离（消除 padding 浪费、长短混 pack 均衡）；Dynamic Sampling 把全对/全错组（advantage=0 无信号）丢弃重采，凑够有效样本才训练。
> - **为什么**：LLM-RL rollout 变长（200~4096 token），padding 浪费可达 ~50% 算力；全对/全错组 advantage=0 梯度为零，占着 batch 却无信号，稀释有效样本数、增梯度方差。两者把"浪费"换成"有效"。
> - **与 [[GRPO]]/[[DAPO]] 关系**：Dynamic Sampling 是 DAPO（arXiv:2503.14476）四技术之一（另三：Clip-Higher、Token-Level Loss、Overlong Reward Shaping）；GRPO 的 group-mean baseline 在全对/全错组失效（σ=0），Dynamic Sampling 是其补丁。


## 2. 为什么需要它（动机与背景）

### 2.1 RL rollout 变长：padding 浪费严重

LLM-RL 的 response 长度**极度不均**：数学题有的 200 token 解完、有的写到 4096 token 还在推。传统做法把一个 batch 的 response **padding 到最长那条** $L_{max}$ 对齐后堆叠成 `[B, L_max]` tensor 送 forward。短样本被 pad 大量无用 token。

设 batch $B$ 条，长度 $L_1,\ldots,L_B$，$L_{max}=\max_i L_i$。padding 后总 token $=B\cdot L_{max}$，有效 token $=\sum_i L_i$。

**padding 浪费率**：

$$
\text{waste}=1-\frac{\sum_i L_i}{B\cdot L_{max}}=1-\frac{\bar L}{L_{max}}
$$

若 $\bar L=L_{max}/2$（高度变长），**浪费 50%**——一半算力/带宽耗在 pad token 上（pad token 既占显存又做无用 FLOPs，还参与 attention 被 mask 掉）。详见 §4.1。这是 LLM-RL 比 NLP 预训练更严重的痛点——RL 的 response 变长且不可预测（CoT 长度随难度/策略变化）。

### 2.2 全对/全错组：advantage=0 梯度信号消失

GRPO（[[GRPO]]）对同一 prompt 采 $G$ 条 response，用**组内 reward 均值当 baseline**、组内相对差当 advantage：

$$
\hat A_i=\frac{r_i-\bar r}{\sigma_r},\quad \bar r=\frac1G\sum_j r_j,\quad \sigma_r=\sqrt{\frac1G\sum_j(r_j-\bar r)^2}
$$

**致命情况**：若一个 group 内 $G$ 条 response **全对**（accuracy=1，全得 $r=1$）或**全错**（accuracy=0，全得 $r=0$）——则 $r_1=\ldots=r_G=\bar r$，$\sigma_r=0$，**所有 advantage $=0/0$ 无定义 → 取 0**。零 advantage → 零 policy gradient（$\nabla\propto\hat A$）→ **该组对更新无贡献**，却占着 batch 名额、稀释有效样本数、增大梯度方差。

DAPO 论文（arXiv:2503.14476 §3.2）实测：随训练进行，**accuracy=1 的 prompt 比例持续上升**（模型变强，简单题全做对），导致每 batch 有效 prompt 数下降、梯度方差增大、收敛变慢。

### 2.3 两个浪费的本质：算力 vs 信号

| 浪费 | 性质 | 解法 |
|---|---|---|
| padding 浪费 | **算力/带宽**浪费（pad token 做无用 FLOPs） | sequence packing（拼真实 token 消除 pad） |
| 全对/全错组 | **梯度信号**浪费（advantage=0 无更新） | Dynamic Sampling（丢弃重采，凑有效样本） |

两者正交：packing 治"算力浪费"，动态采样治"信号浪费"。常联合用（partial 凑够就训 + 动态过滤 + packing 提密度）。

### 2.4 为何 RL 比 pretraining 更需要 packing

预训练的 packing（如 Megatron/LLaMA 训练把短 doc 拼长序列）只是 throughput 优化；**RL packing 多一层 per-trajectory 隔离**——每条打包的 response 有自己的 outcome reward、GRPO advantage、`old_log_prob`（采样时 $\theta_{old}$ 算的 logp）。若 packing 时 trajectory 边界没隔好，trajectory A 的 token attend 到 B，会**算错 logp/ratio/advantage**（IS 修正失真、reward 归属错位）。故 RL packing 的 attention mask 必须 **block-diagonal**（每条只 attend 自己），且 loss/reward 按 trajectory 分段记账。详见 §3.2、§4.4。


## 3. 核心概念详解

### 3.1 padding → packing

```
padding (传统, 浪费):
  batch = [ resp1(200 tok) | pad.... | resp2(4096 tok) ]   ← 短的被 pad 到 4096
  [B, L_max=4096]: 有效 token 密度 = (200+...+4096)/(B·4096) ≈ 50% (浪费 ~50%)

packing (拼进定长):
  seq = [ resp1(200) | resp2(300) | resp3(1500) | ... ]   ← 多条短 response 拼满 4096
  block-diagonal mask: resp1 只 attend resp1, resp2 只 attend resp2, ... (隔离)
  token 密度 → ~100% (无 pad), 同样算力跑更多有效 token
```

- **padding**：每条 response pad 到 $L_{max}$，`[B, L_max]`。短样本浪费 $(L_{max}-L_i)$ pad token。
- **packing**：把多条变长 response **拼进一个定长序列**（长度 = packing unit，如 4096/8192），用 **block-diagonal attention mask** 让各 trajectory 只 attend 自己（A 的 token 不看 B）。消除 pad，token 密度 → ~100%。
- 长短样本**混 pack** 做负载均衡——短样本凑满一个 pack，长样本单独或与短样本凑 pack，避免长尾拖死整 batch（与 [[synchronous与asynchronous rollout]] 的 partial 思路一致）。

### 3.2 RL packing 的 per-trajectory 隔离（与 pretraining packing 的区别）

| 维度 | pretraining packing | RL packing |
|---|---|---|
| 拼什么 | 短 doc 拼长序列 | 变长 response 拼定长 |
| attention mask | block-diagonal（doc 间不 attend） | block-diagonal（trajectory 间不 attend） |
| **per-trajectory 隔离** | 无（只是 throughput） | **必须有**：每条 trajectory 独立 reward/advantage/old_logp |
| loss 记账 | 统一 next-token prediction | **分段记账**：每条 trajectory 的 ratio/advantage 分开算 |
| 边界错误后果 | 轻（doc 间 leakage 影响小） | **重**：logp 算错→ratio 失真→IS 修正错；reward 归属错位→advantage 错 |

RL packing 必须保证：
1. **attention 隔离**：block-diagonal mask，trajectory A 的 token 不 attend B（否则 logp 被污染，详见 [[rollout train reference logprob一致性]]）。
2. **reward/advantage 分段**：每条 trajectory 保留自己的 outcome reward $r_i$、GRPO advantage $\hat A_i$、`old_log_prob`，loss 按 trajectory 分段算 $\frac1{\sum|o_i|}\sum_t\min(\rho_{i,t}\hat A_{i,t},\text{clip}\cdot\hat A_{i,t})$。
3. **position id 重置**：每条 trajectory 在 pack 内从 position 0 重新编号（RoPE position 不跨 trajectory 累加），否则位置编码错位。

### 3.3 Dynamic Sampling（DAPO）：过滤全对/全错组

DAPO（arXiv:2503.14476 §3.2 "The More the Merrier: Dynamic Sampling"）的核心：

**问题**：GRPO 的 group 内若 $G$ 条 response 全对（accuracy=1）或全错（accuracy=0），reward 全相同 → $\sigma_r=0$ → advantage $=0$ → 零梯度。该组占 batch 名额却无信号。随训练进行 accuracy=1 的 prompt 越来越多（模型变强），**有效 prompt 数下降**，梯度方差增大。

**解法**：**over-sample + 过滤**——对每个 prompt 采 $G$ 条后，若该 group **全对或全错**（$|\{o_i|\text{正确}\}|=0$ 或 $=G$），**整组丢弃**，**继续采直到凑够"有效 prompt"**（满足 $0<|\{o_i|\text{正确}\}|<G$，即组内既有对的又有错的，advantage 非零）。DAPO 的约束写进目标：

$$
0<\Big|\{o_i\mid\texttt{is\_equivalent}(a,o_i)\}\Big|<G
$$

即**至少一条对且至少一条错**——保证 $\sigma_r>0$、advantage 非零、有梯度信号。

**关键收益**：每 batch 的有效 prompt 数保持一致（不再随训练下降），梯度方差稳定、收敛更快（DAPO 实测同性能更快达到）。

### 3.4 过滤时机：pre-filter vs post-filter vs timely rollout

| 时机 | 做什么 | 优点 | 缺点 |
|---|---|---|---|
| **pre-filter（rollout 阶段当场过滤）** | 采完一个 group 立刻判 reward，全对/全错就当场丢弃重采，**不进 batch** | 不浪费训练算力在无信号样本；rollout 端就筛好 | rollout 端需能算 reward（verifier 要在 rollout 侧跑）；增加 rollout 端逻辑 |
| **post-filter（累积成 batch 后过滤）** | 先采满一整 batch，再过滤全对/全错组，丢弃后 batch 变小 | 实现简单（trainer 侧统一过滤） | 浪费了已采但被丢的无信号样本的**采样算力** |
| **timely rollout** | 特指在 **rollout 阶段当场**做过滤/判 reward（pre-filter 的强调） | 采样算力不浪费在注定无信号的样本上 | 需 rollout 端集成 verifier |

> [!tip] 实践：pre-filter 是 DAPO 的隐含选择
> DAPO 论文说"keep sampling until the batch is fully filled with samples whose accuracy is neither 0 nor 1"——即**采的过程中就过滤**（pre-filter），而非采满再筛。且指出"generation time is typically dominated by the generation of long-tail samples if the RL system is synchronized"——同步系统下长尾样本拖慢，动态采样的额外采样开销被长尾主导，故不显著拖慢。这支持 pre-filter 的及时性（timely rollout）：在 rollout 端边采边筛，不让无信号样本浪费后续训练算力。

### 3.5 padding vs packing 算术强度对比

LLM 推理/训练的线性层是 **memory-bound**（权重加载是瓶颈，算术强度 AI = FLOPs/bytes $\propto$ token 数）。详见 [[continuous batching的调度实现]]、[[chunked prefill]] 的算力带宽分析。

- **padding**：处理 $B\cdot L_{max}$ 个 token，但只有 $\sum L_i$ 个有效。权重加载次数不变，但 **pad token 做无用 FLOPs**、占无用带宽。**有效算术强度** $=\sum L_i / (\text{weight bytes})$ 被 pad 稀释。
- **packing**：把同样显存/算力预算填满**有效 token**（无 pad），**有效算术强度** $\to L_{max}/(\text{weight bytes})$（满载）。同样算力跑更多有效 token。

**有效吞吐提升** $\approx L_{max}/\bar L = 1/(1-\text{waste})$。若 waste=50%，packing 提速 ~2× 有效吞吐。详见 §4.2。

### 3.6 为何 RL 需要 packing（rollout 变长 vs 预训练定长）

预训练的 sequence length 常固定（如 2048/4096），doc 拼接后切等长，padding 少。**RL rollout 变长且不可预测**——CoT 长度随 prompt 难度/策略长度变化（数学题 200~4096），同一 group 的 $G$ 条 response 长度都可能差 10×。padding 对齐到最长浪费尤其严重。故 RL 比 pretraining 更需要 packing。且 RL 还要混长短 pack 均衡负载（避免长尾拖死整 batch，呼应 partial rollout）。


## 4. 数学原理 / 公式

### 4.1 推导：padding 浪费率

设 batch $B$ 条 response，长度 $L_1,\ldots,L_B$，$L_{max}=\max_i L_i$，均值 $\bar L=\frac1B\sum_i L_i$。

padding 对齐到 $L_{max}$，总 token $N_{pad}=B\cdot L_{max}$，有效 token $N_{useful}=\sum_i L_i=B\bar L$。

**浪费率**：

$$
\boxed{\;\text{waste}=1-\frac{N_{useful}}{N_{pad}}=1-\frac{B\bar L}{B\cdot L_{max}}=1-\frac{\bar L}{L_{max}}\;}
$$

- $\bar L=L_{max}$（等长）：waste=0（无浪费）。
- $\bar L=L_{max}/2$（高度变长）：waste=0.5（**浪费一半**）。
- $\bar L=L_{max}/4$（极端变长，长尾主导）：waste=0.75（浪费 3/4）。

LLM-RL 的 CoT 变长常使 $\bar L/L_{max}\in[0.3,0.6]$，即 waste 40~70%，padding 极浪费。

### 4.2 推导：packing 的有效吞吐提升

memory-bound 线性层：算术强度 $\text{AI}=\text{FLOPs}/\text{bytes}\propto$ token 数（权重加载一次，跑 $N$ 个 token）。**有效吞吐** $\propto$ 有效 token 数 / 时间。

设同算力预算（同卡同时间 $T$）能处理的**总 token 上限** $N_{budget}=T\cdot\text{throughput}$（受带宽/算力上限）。

- **padding**：处理 $B\cdot L_{max}\le N_{budget}$ 个 token，其中有效 $\sum L_i=B\bar L$。受 $B\cdot L_{max}\le N_{budget}$ 约束 → $B\le N_{budget}/L_{max}$ → 有效 token $\le (N_{budget}/L_{max})\bar L=N_{budget}\cdot(\bar L/L_{max})$。

- **packing**：处理全是有效 token，$\sum L_i\le N_{budget}$ → 有效 token $\le N_{budget}$。

**有效吞吐提升**：

$$
\boxed{\;\text{speedup}_{\text{useful}}=\frac{N_{budget}}{N_{budget}\cdot(\bar L/L_{max})}=\frac{L_{max}}{\bar L}=\frac{1}{1-\text{waste}}\;}
$$

- waste=0.5：speedup=2×（有效吞吐翻倍）。
- waste=0.75：speedup=4×。

即 packing 把"浪费在 pad 上的算力/带宽"换成有效 token，提升正比于 $1/(1-\text{waste})$。是 memory-bound 场景的吞吐红利（compute-bound 场景收益小，因算力本就是瓶颈）。

### 4.3 推导：全对/全错组 advantage=0（Dynamic Sampling 的数学根）

GRPO 的 advantage $\hat A_i=(r_i-\bar r)/\sigma_r$。设 group $G$ 条 reward $r_1,\ldots,r_G$。

**全对**（accuracy=1，$r_1=\ldots=r_G=1$）或**全错**（accuracy=0，$r_1=\ldots=r_G=0$）：

$$
\bar r=\frac1G\sum_j r_j=r\ (\text{常数}),\quad \sigma_r=\sqrt{\frac1G\sum_j(r_j-\bar r)^2}=\sqrt{0}=0
$$

$$
\hat A_i=\frac{r_i-\bar r}{\sigma_r}=\frac{0}{0}\ \text{（无定义）}\ \Rightarrow\ \hat A_i:=0\ \text{（工程取 0）}
$$

PPO/GRPO 目标 $\propto\min(\rho_t\hat A_t,\text{clip}\cdot\hat A_t)$，$\hat A_t=0$ → **该组梯度为 0**。即全对/全错组**对更新无贡献**，却占 batch 名额、稀释有效样本数 $\to$ 梯度方差 $\uparrow$、有效信号 $\downarrow$。

**Dynamic Sampling 的约束** $0<|\{o_i|\text{正确}\}|<G$ 保证 $\sigma_r>0$（组内有差异）→ advantage 非零 → 有梯度信号。把"无信号组"换成"有信号组"，保持有效 prompt 数一致。

> [!note] 补充：accuracy=1 上升的动态
> 随训练进行模型变强，**简单题越来越多全对**（accuracy=1 比例上升，DAPO Figure 3(b) 实测）。若不过滤，每 batch 有效 prompt 数单调下降，后期 batch 几乎无信号——这是 GRPO 训练后期收敛变慢、方差增大的主因之一。Dynamic Sampling 是对这个退化动态的对症补丁。

### 4.4 推导：RL packing 的 block-diagonal mask 与 per-trajectory 隔离

设一个 pack 内拼了 $K$ 条 trajectory，第 $k$ 条长 $L_k$，pack 总长 $L=\sum_k L_k$。token $i$ 属于 trajectory $\tau(i)$（其起止 $[s_{\tau(i)},e_{\tau(i)}]$）。

**block-diagonal attention mask**（让 trajectory 内自回归 attend、跨 trajectory 不 attend）：

$$
M_{ij}=\begin{cases}0,&\tau(i)=\tau(j)\text{ 且 }j\le i\ (\text{同 trajectory, 因果})\\0,&\tau(i)\ne\tau(j)\ ?\ \text{否}\end{cases}
$$

即 $M_{ij}=0$（允许 attend）当且仅当 $i,j$ 同属一条 trajectory 且 $j\le i$；否则 $=-\infty$（屏蔽）。

**per-trajectory 记账**：第 $k$ 条 trajectory 的 token 级 ratio/advantage：

$$
\rho_{k,t}=\frac{\pi_{\theta_{new}}(o_{k,t})}{\pi_{\theta_{old}}(o_{k,t})},\quad \hat A_{k,t}=\frac{r_k-\bar r_k}{\sigma_{r,k}}\ (\text{GRPO 组内, }r_k\text{ 是该 trajectory 的 outcome reward})
$$

loss 按 trajectory 分段：

$$
L=\frac{1}{\sum_k L_k}\sum_k\sum_{t=1}^{L_k}\min(\rho_{k,t}\hat A_{k,t},\ \text{clip}(\rho_{k,t},1-\varepsilon,1+\varepsilon)\hat A_{k,t})
$$

**关键**：若 mask 没 block-diagonal（trajectory A token attend B），则 $\log\pi_{\theta}(o_{A,t})$ 被污染（混入 B 的 KV）→ ratio $\rho_{A,t}$ 失真 → IS 修正错（与 [[rollout train reference logprob一致性]] 的 logp 污染同构）；reward $r_k$ 若归属错位（A 的 reward 算给 B）→ advantage $\hat A_{k,t}$ 错。故 RL packing 的 mask + 分段记账是正确性的前提，不止 throughput 优化。


## 5. 代码示例（可选）

### 5.1 sequence packing 的 attention mask 构造

```python
import torch

def build_block_diagonal_mask(lengths, L_max):
    # lengths: 每条 trajectory 的长度 [L_1, L_2, ...], sum <= L_max
    # 返回 [L_max, L_max] 的 attention mask, block-diagonal (trajectory 间不 attend)
    mask = torch.full((L_max, L_max), float('-inf'))
    offset = 0
    for L in lengths:
        # 该 trajectory 占 [offset, offset+L), 内部自回归 (下三角允许 attend)
        mask[offset:offset+L, offset:offset+L] = torch.triu(
            torch.full((L, L), float('-inf')), diagonal=1
        )  # diagonal=1: 上三角(不含对角)屏蔽 -> 自回归
        offset += L
    # pack 尾部 padding 位 (offset:L_max) 全屏蔽 (-inf)
    return mask  # 0=允许 attend, -inf=屏蔽

# 例: 3 条 trajectory 拼进 L_max=8, 长度 [2, 3, 1], 尾部 2 位 pad
mask = build_block_diagonal_mask([2, 3, 1], L_max=8)
# block 0 (traj1, idx0-1) 只 attend 自己, block 1 (traj2, idx2-4) 只 attend 自己,
# block 2 (traj3, idx5) 只 attend 自己, idx6-7 (pad) 全屏蔽 -> 隔离 + 无 pad 浪费
```

### 5.2 RL packing 的 per-trajectory 分段记账（概念）

```python
def packed_rl_loss(packed_tokens, lengths, rewards, old_logps, theta_new):
    # packed_tokens: [L] 拼接的 token, lengths: 各 trajectory 长 [L_k], rewards: 各 outcome [r_k]
    new_logps = theta_new.logp(packed_tokens)          # 一次性 forward 整个 pack (省算力)
    # 按 trajectory 分段算 ratio/advantage (不能混)
    offset = 0
    loss, n_tok = 0.0, 0
    r_mean = rewards.mean(); r_std = rewards.std() + 1e-8  # GRPO group baseline
    for k, L in enumerate(lengths):
        sl = slice(offset, offset + L)
        ratio = (new_logps[sl] - old_logps[k]).exp()       # 该 trajectory 的 ρ
        adv = (rewards[k] - r_mean) / r_std                 # 该 trajectory 的 advantage
        adv = adv.expand(L)                                   # sequence-level advantage 广播到 token
        loss_k = (-torch.min(ratio * adv,
                             ratio.clamp(1-EPS, 1+EPS) * adv)).sum()  # PPO clip
        loss = loss + loss_k; n_tok += L; offset += L
    return loss / n_tok   # token-level 归一 (呼应 DAPO Token-Level Loss)
```

### 5.3 Dynamic Sampling 的过滤循环（DAPO §3.2）

```python
def dynamic_sampling_batch(prompts, g_size, target_n, rollout_fn, verifier):
    # 过采样: 全对/全错组丢弃重采, 凑够 target_n 个"有效"prompt 才训练
    valid = []
    while len(valid) < target_n:
        q = sample_prompt(prompts)
        group = [rollout_fn(q) for _ in range(g_size)]        # 采 G 条 response
        rewards = [verifier(q, o) for o in group]             # rule-based 判 reward
        n_correct = sum(r > 0 for r in rewards)               # 正确数
        if 0 < n_correct < g_size:                            # 既非全对也非全错 (DAPO 约束)
            valid.append((q, group, rewards))                 # 有效 (advantage 非零), 进 batch
        # 否则 (全对/全错) 丢弃, 继续采 —— timely: 在 rollout 阶段当场过滤
    return valid[:target_n]
# -> 每 batch 有效 prompt 数一致, 不随 accuracy=1 上升而退化
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[trajectory generation]]（采的就是变长 response）、[[sampling throughput]]（packing 提有效吞吐）、[[batching strategy]]（packing 是 batch 组成策略）、[[continuous batching的调度实现]]（rollout 端变长混合调度）、[[chunked prefill]]（变长混合的算力带宽分析）、[[GRPO]]（group sampling + group-mean baseline，全对/全错组失效）、[[verifier与function reward]]（Dynamic Sampling 靠 rule-based verifier 判全对/全错）。
- **下游（应用）**: [[17-RL训推一体框架]] 采样工程、[[DAPO]]（Dynamic Sampling 是其四技术之一）、[[synchronous与asynchronous rollout]]（partial 常配动态采样 + packing）、[[GRPO体系]]（GRPO 的采样补丁）。
- **对比 / 易混**:
  - **RL packing vs 预训练 packing**：预训练 packing 只 throughput 优化（doc 间 leakage 影响小）；RL packing **必须 per-trajectory 隔离**（reward/advantage/logp 不能混，block-diagonal mask + 分段记账）。详见 §3.2。
  - **Dynamic Sampling vs [[rollout worker]] 端 continuous batching**：前者是**样本有效性**过滤（丢无信号组）；后者是**采样吞吐**调度（请求动态进出）。正交，常联合（边采边筛）。
  - **pre-filter vs post-filter**：pre-filter（rollout 当场过滤，不浪费训练算力）；post-filter（trainer 侧采满再筛，浪费采样算力）。DAPO 隐含 pre-filter（timely rollout）。详见 §3.4。
  - **Dynamic Sampling vs Overlong Reward Shaping**：前者丢"无信号"组（advantage=0）；后者处理"超长截断"样本的 reward 噪声。两者都是 DAPO 技术，治不同浪费。
  - **本条 vs [[synchronous与asynchronous rollout]]**：后者讲采训**时序**（同步/异步/partial），本条讲采**样本本身**的工程优化（变长 packing、无信号过滤）。partial（凑够就训）常配 dynamic sampling（凑的是有效样本）+ packing（凑的是高密度 token）。

- **三篇互链**: [[synchronous与asynchronous rollout]]（partial + 动态采样 + packing 联合）、[[verifier与function reward]]（verifier 判 reward 决定动态采样过滤）。


## 7. 常见误区与易错点

> [!warning] 误区 1：packing 就是简单拼接
> 不。RL packing 必须 **block-diagonal attention mask** + **per-trajectory 分段记账**。若 trajectory 间 token 互相 attend，会算错 logp → ratio 失真（与 TIM 同构污染）；reward/advantage 若不分段会归属错位。预训练 packing 可松（doc 间 leakage 影响小），RL packing 严格。详见 §3.2、§4.4。

> [!warning] 误区 2：Dynamic Sampling 是过滤"低 reward"样本
> 不是。是过滤**全对或全错**的 group（advantage=0 无信号），不是过滤低 reward。一条全错的 group（reward=0）和一条全对的（reward=1）**同样被丢**——两者 $\sigma_r=0$、advantage=0。要的是**组内有差异**（$0<\text{正确数}<G$），advantage 非零。低 reward 但组内有差异的 group 保留（它提供"该 prompt 上哪些 response 更差"的信号）。详见 §4.3。

> [!warning] 误区 3：packing 一定提速 2×
> 仅 memory-bound 场景且 waste 高时。compute-bound 场景（算力是瓶颈）packing 收益小（pad token 也耗算力但权重已加载，省的是带宽不是算力）。且 packing 有 mask 构造/分段记账开销。极端等长（waste=0）packing 反增开销。收益 $\propto 1/(1-\text{waste})$，看变长程度。详见 §4.2。

> [!warning] 误区 4：Dynamic Sampling 拖慢训练（要过采样）
> DAPO 论文指出：同步系统下 generation time 被**长尾样本**主导，动态采样的额外采样开销被长尾吸收，**不显著拖慢**。且过滤后每 batch 有效 prompt 一致，**同性能更快达到**（少跑无用 step）。但若 rollout 不平衡（rollout 已饱和），过采样会增开销——需配 partial/timely rollout 在 rollout 端边采边筛。详见 §3.4。

> [!warning] 误区 5：position id 跨 trajectory 累加
> 错。pack 内每条 trajectory 的 **position id 从 0 重新编号**（RoPE position 不跨 trajectory 累加），否则位置编码错位、logp 算错。与 block-diagonal mask 配套。是 RL packing 易漏的点。

> [!warning] 误区 6：post-filter 等价 pre-filter
> 不等价。post-filter（采满 batch 再筛）浪费了被丢样本的**采样算力**；pre-filter（rollout 端边采边筛，timely rollout）不浪费。DAPO 选 pre-filter。两者对 rollout 算力预算影响不同。详见 §3.4。

> [!warning] 误区 7：全对组 reward 高所以该保留
> 逻辑反了。RL 不是最大化 reward，是**最大化 advantage 信号**做策略更新。全对组 reward 高但 advantage=0（无相对差异）→ 无梯度信号 → 对更新无用。保留它反而稀释有效样本、增方差。Dynamic Sampling 丢的是"无信号"组，不是"低 reward"组。与 [[reward hacking]] 区别：reward hacking 是刷高 reward 没真变强；全对组是 reward 真高但无相对学习信号。


## 8. 延伸细节

### 8.1 packing 与 continuous batching / chunked prefill 的关系

rollout 端（vLLM/SGLang）的采样吞吐受 [[continuous batching的调度实现]]（请求动态进出）+ [[chunked prefill]]（长 prompt 切块与 decode 混）影响。packing 是**训练/重算侧**的变长优化（把采好的多条 response 拼进 forward 序列），与 rollout 端的 continuous batching 正交——一个管"怎么采"，一个管"采好后怎么 forward"。两者都解决变长，但层面不同。详见 [[continuous batching的调度实现]]、[[chunked prefill]]。

### 8.2 DAPO 四技术与本条的耦合

DAPO（arXiv:2503.14476）四技术中，**本条直接关联两条**：

- **Dynamic Sampling**（§3.2）：本条核心，过滤全对/全错组。
- **Token-Level Policy Gradient Loss**（§3.3）：loss 按**总 token 数** $\sum|o_i|$ 归一（而非 GRPO 的 sample-level 先 per-sequence mean 再平均）。这让长样本的每个 token 有等权贡献，与 packing 的"按 trajectory 分段、token 级归一"天然契合——packing 后 loss 正好按 $\sum_k L_k$ 归一，每 token 等权。本条 §5.2 的 `loss/n_tok` 即 token-level 归一。
- 另两技术：**Clip-Higher**（解耦 $\varepsilon_{low}/\varepsilon_{high}$，提探索，详见 [[DAPO]]）、**Overlong Reward Shaping**（超长截断样本 mask loss 减 reward 噪声，详见 [[verifier与function reward]]）。

### 8.3 timely rollout 的工程含义

"timely rollout" 强调**在 rollout 阶段当场**做过滤/判 reward（pre-filter），而非攒到 trainer 侧。工程要求：rollout 端集成 verifier（数学题对答案、代码题跑测试用例，详见 [[verifier与function reward]]），采完一个 group 立刻判、全对/全错当场丢、继续采。好处：不把无信号样本送回 trainer（省传输/省训练算力），且 batch 凑满时已是全有效样本。代价：rollout 端要跑 verifier（代码沙箱等），增加 rollout 端复杂度。详见 [[verifier与function reward]] §3.3（沙箱执行）。

### 8.4 内容来源

sequence packing 与 padding 浪费率/算术强度推导基于 memory-bound 线性层算术强度分析（与 [[continuous batching的调度实现]]、[[chunked prefill]] 一致）。Dynamic Sampling、DAPO 约束 $0<|\{o_i|\text{is\_equivalent}\}|<G$、over-sample+过滤、accuracy=1 上升退化动态均来自 DAPO 论文（arXiv:2503.14476 §3.2，已联网核实原文）。DAPO 四技术（Clip-Higher / Dynamic Sampling / Token-Level Loss / Overlong Reward Shaping）与"built on verl"来自同论文 §3。block-diagonal mask 与 per-trajectory 隔离是 RL packing 工程标准（verl/OpenRLHF 实现，与 [[rollout train reference logprob一致性]] 的 logp 污染同构）。timely rollout / pre-filter 的工程含义综合自 DAPO §3.2 论述与社区实践。截至 2026-07。

---
相关: [[采样工程]] | [[trajectory generation]] | [[sampling throughput]] | [[batching strategy]] | [[continuous batching的调度实现]] | [[chunked prefill]] | [[GRPO]] | [[DAPO]] | [[synchronous与asynchronous rollout]] | [[verifier与function reward]] | [[rollout train reference logprob一致性]] | [[reward hacking]]
