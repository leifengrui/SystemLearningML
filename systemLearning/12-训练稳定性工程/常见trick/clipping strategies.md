# clipping strategies

> **所属章节**: [[常见trick]]
> **所属模块**: [[12-训练稳定性工程]]
> **别名**: clipping strategies / clip / gradient clipping / PPO clip / ratio clip
> **难度**: 入门（需懂 [[gradient explosion and vanishing]] + PPO）


## 1. 一句话定义

**clipping（截断）** 是训练中对**梯度（gradient clip）**或 **PPO 的 importance ratio（PPO clip）**做硬上界截断，防更新过大，是训练稳定性的核心机制。**gradient clip**：梯度 norm 超 $c$ 时缩到 $c$（by global norm）或截断元素值（by value），防 [[gradient explosion and vanishing|梯度爆炸]]传权重。**PPO clip**：截断 importance ratio $r_t$ 到 $[1-\epsilon, 1+\epsilon]$（$\epsilon \approx 0.2$），防 policy 更新偏离 $\pi_{\text{old}}$ 过远。两者不同：gradient clip 稳梯度（数值维），PPO clip 稳 policy 更新（优化维）。clip 是 LLM 训练（防 spike）和 RLHF（防 policy 崩）的标配。与 [[warmup]] / [[KL divergence control]] 正交组合。

> [!note] 三句话定位
> - **是什么**：对梯度 norm（gradient clip）或 PPO ratio（PPO clip）硬截断，防更新过大。
> - **为什么**：梯度爆炸致权重炸；PPO ratio 过大致 policy 偏离；clip 硬限上界防发散。
> - **怎么用**：gradient clip by global norm（$c \sim 1.0$）+ PPO clip（$\epsilon \sim 0.2$）。


## 2. 为什么需要它（动机与背景）

### 2.1 梯度爆炸的硬限

[[gradient explosion and vanishing|梯度爆炸]]：深层连乘 $>1$，梯度指数大，权重更新炸 → loss spike/NaN。gradient clip 硬限梯度 norm：

$$
\text{if } \|g\| > c: \quad g \leftarrow g \cdot \frac{c}{\|g\|}
$$

超阈值的梯度缩到阈值，防炸传权重。是 LLM 训练防 [[loss spike handling|spike]] 的第一道防线。

### 2.2 PPO ratio 的偏离

PPO 用 importance ratio $r_t = \pi_\theta(y)/\pi_{\text{old}}(y)$ 做 off-policy 修正。$r_t$ 过大（policy 偏离 $\pi_{\text{old}}$ 远）→ 更新大 → policy 震荡/崩。PPO clip 截断 $r_t$ 到 $[1-\epsilon, 1+\epsilon]$：

$$
L = \min(r_t A, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A)
$$

防 policy 单步偏离过远。是 PPO 的核心稳定性机制（区别于 TRPO 的 KL 约束，clip 更简单）。

### 2.3 clip vs KL 的不同

| 维度 | gradient clip | PPO clip | KL penalty |
|---|---|---|---|
| 截什么 | 梯度 norm | importance ratio | policy 偏离 reference |
| 维度 | 数值（防爆炸） | 优化（防 policy 偏 old） | 优化（防 policy 偏 ref） |
| 场景 | 所有训练 | PPO | RLHF |
| 硬/软 | 硬限（截断） | 硬限（截断） | 软（penalty） |

三者正交，可组合：LLM 预训练用 gradient clip；RLHF 用 gradient clip + PPO clip + KL penalty。

### 2.4 clip 的局限性

clip 是硬限（截断），非根治：

- gradient clip 防梯度炸传权重，但不防梯度产生（根因是连乘/init/lr）。
- PPO clip 防 ratio 偏离，但不防 policy 整体偏离 reference（KL 才管）。
- clip 过严欠优化（截断有用梯度），过松无效。

clip 是稳定性手段，需与 [[warmup]] / init / lr schedule 配合根治。


## 3. 核心概念详解

### 3.1 gradient clip by global norm（主流）

算所有参数梯度的 global norm $\|g\| = \sqrt{\sum_p \sum_i g_{p,i}^2}$，若 $> c$ 则整体缩：

$$
g \leftarrow g \cdot \min(1, c/\|g\|)
$$

保留梯度方向，只缩幅度。PyTorch 的 `clip_grad_norm_`。主流（LLM/RLHF）。

### 3.2 gradient clip by value

截断每个梯度元素到 $[-v, v]$：

$$
g_i \leftarrow \max(-v, \min(v, g_i))
$$

不保留方向（元素独立截），不如 global norm。少用。PyTorch 的 `clip_grad_value_`。

### 3.3 global norm vs per-layer

- **global norm**：所有参数合算 norm，整体缩。主流。
- **per-layer**：每层独立 clip。某些场景（如极深模型某些层梯度特别大）更精细，但少用。

global norm 简单且多数够用，LLM 主流。

### 3.4 PPO clip

PPO objective：

$$
L = \mathbb{E}\left[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)\right]
$$

- $r_t > 1+\epsilon$（policy 更偏 $\pi_\theta$）且 $A > 0$：clip 限更新（防过增）。
- $r_t < 1-\epsilon$（policy 更偏 $\pi_{\text{old}}$）且 $A < 0$：clip 限更新（防过减）。
- 其他：min 选保守，clip 不激活。

$\epsilon \approx 0.2$（经验）。clip 使 policy 在 $\pi_{\text{old}}$ 附近 $\epsilon$-ball 内更新，防单步偏离远。

### 3.5 clip 阈值的选择

| clip | 典型阈值 | 调节 |
|---|---|---|
| gradient clip $c$ | 1.0（LLM）/ 0.5-5.0 | 太松无效，太严欠优化 |
| PPO clip $\epsilon$ | 0.2 | 0.1-0.3，大模型可 0.2 |

gradient clip $c$：观察训练中梯度 norm 分布，设为 50-90 分位（截大异常，保常规）。PPO $\epsilon$：0.2 是经验最优（TRPO 启发），少调。

### 3.6 clip 与 warmup 的配合

[[warmup]]（前期小 lr）+ gradient clip（限梯度幅度）+ init（He/Xavier）= 防 [[gradient explosion and vanishing|爆炸]] 栈。warmup 防 Adam 冷启动 + 早期不稳，clip 防梯度突大，init 防连乘偏离。三者正交，组合用。


## 4. 数学原理 / 公式

### 4.1 gradient clip by global norm

$$
\|g\| = \sqrt{\sum_{p,i} g_{p,i}^2}, \quad g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|}\right)
$$

若 $\|g\| \leq c$ 不变，$> c$ 缩到 $c$。

### 4.2 gradient clip by value

$$
g_i \leftarrow \max(-v, \min(v, g_i))
$$

### 4.3 PPO clip

$$
L = \min\left(r_t A_t, \, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)
$$

$\text{clip}(r, a, b) = \max(a, \min(b, r))$。$\epsilon \approx 0.2$。

### 4.4 clip 的效果

- gradient clip：$\|g\| > c$ 时 $\|g'\| = c$，权重更新 $\Delta\theta = \eta \cdot g'$ 的 norm 上界 $\eta c$。防 $\Delta\theta$ 炸。
- PPO clip：$r_t \in [1-\epsilon, 1+\epsilon]$，policy 单步偏离 $\pi_{\text{old}}$ 限 $\epsilon$-ball。

### 4.5 clip 与 KL 的关系

PPO clip 约束 $\pi_\theta$ 不偏离 $\pi_{\text{old}}$（局部，上一迭代），KL penalty 约束不偏离 $\pi_{\text{ref}}$（全局，SFT）。两者正交：clip 防 single-step 跳跃，KL 防 cumulative 偏离。详见 [[KL divergence control]]。


## 5. 代码示例（可选

### 5.1 gradient clip（模拟 PyTorch clip_grad_norm_）

```python
import math
def clip_grad_norm(grads, c=1.0):
    """global norm clip: 若 ||g||>c, 整体缩到 c."""
    total_norm = math.sqrt(sum(g*g for g in grads))
    scale = min(1.0, c / (total_norm + 1e-6))
    return [g * scale for g in grads], total_norm

grads = [3.0, 4.0]  # ||g|| = 5
clipped, norm = clip_grad_norm(grads, c=1.0)
print(f"original norm={norm:.2f}, clipped={clipped}, new norm={math.sqrt(sum(c*c for c in clipped)):.4f}")
```

### 5.2 PPO clip

```python
def ppo_clip(ratio, advantage, epsilon=0.2):
    """min(r*A, clip(r, 1-eps, 1+eps)*A)."""
    r_clipped = max(1-epsilon, min(1+epsilon, ratio))
    return min(ratio * advantage, r_clipped * advantage)

for r, a in [(1.5, 1.0), (0.5, -1.0), (1.3, 1.0), (0.7, -1.0)]:
    print(f"r={r}, A={a} -> L={ppo_clip(r, a):.3f}")
```

### 5.3 PyTorch 实际用法

```python
# gradient clip (训练循环中)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[gradient explosion and vanishing]]（clip 防爆炸）、PPO（PPO clip 的场景）、梯度/norm 数学。
- **下游（应用）**: [[loss spike handling]]（clip 防 spike）、[[warmup]]（与 clip 并列防 spike）、[[KL divergence control]]（与 PPO clip 正交约束 policy）、[[reward normalization]]（稳 advantage 使 PPO clip 有效）、[[entropy regularization]]（与 PPO clip 稳 PPO）、LLM 预训练/RLHF 训练实践、PyTorch clip_grad_norm_。
- **对比 / 易混**:
  - **gradient clip vs PPO clip**：前者稳梯度 norm（数值维，所有训练），后者稳 importance ratio（优化维，PPO）。详见 §2.3。
  - **clip vs [[KL divergence control\|KL penalty]]**：clip 硬限（截断），KL 软（penalty）。clip 防 single-step 跳跃，KL 防 cumulative 偏离。PPO 同时用。
  - **clip vs [[warmup]]**：clip 限梯度幅度（per-step），warmup 调 lr（全局 schedule）。不同层级的稳定性手段。
  - **global norm vs by value**：前者保留方向（主流），后者元素独立截（少用）。详见 §3.1-3.2。


## 7. 常见误区与易错点

> [!warning] 误区 1：clip 无害可任意紧
> 不是。clip 过严截断有用梯度，欠优化（收敛慢/不收敛）。$c$ 应设为梯度 norm 分布的 50-90 分位，截大异常保常规，非越小越好。

> [!warning] 误区 2：clip 解决梯度爆炸
> 缓解非根治。clip 截断大梯度防传权重，但不防梯度产生（根因是连乘/init/lr）。需配合 [[warmup]] / He init / lr schedule 根治。clip 是症状治疗。

> [!warning] 误区 3：PPO clip = KL penalty
> 不同。PPO clip 约束不偏离 $\pi_{\text{old}}$（局部，硬限），KL penalty 约束不偏离 $\pi_{\text{ref}}$（全局，软）。PPO 同时用：clip 防 single-step 跳跃，KL 防 cumulative 偏离。详见 [[KL divergence control]]。

> [!warning] 误区 4：clip 阈值任意
> 需调。gradient clip $c$ 按梯度 norm 分布设（观察训练中 norm），PPO $\epsilon$ 有经验最优（0.2）。任意设可能无效（过松）或欠优化（过严）。

> [!warning] 误区 5：by value = global norm
> 不同。by value 元素独立截（不保方向），global norm 整体缩（保方向）。global norm 更合理（保留梯度方向），主流。by value 少用。

> [!warning] 误区 6：clip 后无需 warmup
> 不足。clip 防梯度突大，但早期不稳还涉及 Adam $v$ 冷启动、LN statistics 未稳。warmup 与 clip 正交，需组合。详见 [[warmup]]。

> [!warning] 误区 7：PPO clip ε 越小越稳
> 不是。$\epsilon$ 过小 policy 更新受限，收敛慢/欠优化；过大防偏离失效。0.2 是经验最优（TRPO 启发），少调。


## 8. 延伸细节

### 8.1 PyTorch clip_grad_norm_ 的实现

`torch.nn.utils.clip_grad_norm_(params, max_norm)`：

1. 算所有参数梯度的 global norm $\|g\|$。
2. 若 $> \text{max\_norm}$，所有梯度乘 $\text{max\_norm}/\|g\|$。
3. 返回原 norm（可监测）。

是 LLM/RLHF 训练循环的标配（`loss.backward(); clip_grad_norm_; optimizer.step()`）。

### 8.2 clip 与 fp16/bf16

fp16 易溢出（max 65504），梯度大时 clip 前已 NaN。需：

- bf16（范围大，不易溢出）。
- 或 fp16 + loss scaling（先放大 loss 防下溢，clip 后 unscale）。

bf16 主流，clip 在 bf16 上正常工作。详见 [[混合精度]]。

### 8.3 PPO clip 的来源

PPO（Schulman 2017）用 clip 替代 TRPO 的 KL 约束（TRPO 解 constrained optimization 贵）。clip 简单（截断）且效果相近，使 PPO 比 TRPO 快。$\epsilon=0.2$ 是原论文经验值，多数场景最优。

### 8.4 gradient clip 的阈值监测

训练中监测梯度 norm：

- 多数 step $\|g\| < c$（clip 不激活，正常）。
- 少数 step $\|g\| > c$（clip 激活，截异常）。
- 频繁激活（>10% step）→ $c$ 过小或梯度真不稳（查 init/lr/data）。

`clip_grad_norm_` 返回原 norm，记 tensorboard 监测。

### 8.5 RLHF 的 clip 栈

RLHF/PPO 训练用多层 clip：

- gradient clip（防梯度炸）。
- PPO clip（防 ratio 偏离）。
- reward clip（防 reward 极端值，可选）。
- KL penalty（防 policy 偏离 reference，软约束）。

四层稳 RLHF。clip 是硬限，KL 是软约束，互补。

### 8.6 clip 与 mode collapse

PPO clip 限 policy 偏离 $\pi_{\text{old}}$，间接防 mode collapse（policy 塌到单一输出）。但 clip 非专门防 mode collapse（[[entropy regularization]] 才是）。clip 稳更新幅度，entropy reg 鼓励多样，两者正交。

---
相关: [[常见trick]] | [[warmup]] | [[loss spike handling]] | [[gradient explosion and vanishing]] | [[KL divergence control]] | [[reward normalization]] | [[entropy regularization]] | [[PPO]] | [[混合精度]] | [[AdamW]] | [[He initialization]] | [[torch.compile]] | [[torch profiler]]
