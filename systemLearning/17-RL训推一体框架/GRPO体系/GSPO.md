# GSPO

> **所属章节**: [[GRPO体系]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §81
> **别名**: Group Sequence Policy Optimization / 序列级 ratio / 几何均值 ratio
> **难度**: 高（需懂 [[GRPO]]、[[importance sampling与off-policy correction]]、[[token-level与sequence-level objective]]、[[MoE路由]]、[[训推不一致]]）
> **来源**: Zheng et al. (Qwen Team, Alibaba), *Group Sequence Policy Optimization*, arXiv:2507.18071 (2025-07). 全文翻译见 [[GSPO论文全文中文翻译]]。

## 1. 一句话定义

**GSPO（Group Sequence Policy Optimization）** 把 GRPO 的 **token 级** importance ratio 换成**序列级** importance ratio $s_i$（逐 token ratio 的**长度归一几何均值**），裁剪作用在整条 response 上而非每 token——解决 token 级 ratio 在长序列/MoE 上的**方差爆炸**（MoE 每 step ~10% 激活专家在 old/new 间跳变，token ratio 剧烈波动），稳定 Qwen3 MoE RL 训练。但它仍用 ratio 作信任域代理，遗留"ratio 难 proxy 真实分布偏移"问题，为 [[DPPO]] 铺垫。

## 2. 为什么需要它（动机与背景）

### 2.1 token 级 ratio 的统计缺陷

GRPO 每 token 算 $w_{i,t}=\pi_\theta(y_{i,t})/\pi_{\theta_{old}}(y_{i,t})$，这是对 IS 的**误用**——IS 原理要求在行为分布上**多样本平均**（$N\gg1$），而 GRPO 在每个 token 位置只从**单样本** $y_{i,t}$ 算权重：
1. 没做分布修正
2. 给训练梯度注入高方差噪声
3. 噪声在长序列上累积

### 2.2 MoE 让 token ratio 更不稳

MoE 每步 ~10% 激活专家在 old/new 间跳变，token ratio 大幅波动。长序列累积 → 方差爆炸 → 模型崩溃。

### 2.3 核心原则：优化单元 = 奖励单元

reward 是序列级，token 级 correction 与之错配。GSPO 聚合到序列级，平滑噪声——"只要语言建模保持，序列似然对个别 token 波动不敏感"。

## 3. 核心概念详解

### 3.1 序列级 importance ratio（几何均值）

$$s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{1/|y_i|}=\exp\!\left(\frac1{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}}\right)$$

- $1/|y_i|$ 次幂 = 长度归一，把 $s_i$ 控制在统一数值范围，降方差
- 几何均值（非算术）：exp(算术均值 of log-ratio)

### 3.2 GSPO 目标

$$\mathcal{J}_{\text{GSPO}}(\theta)=\mathbb{E}\!\left[\frac1G\sum_{i=1}^G\min\!\Big(s_i(\theta)\widehat A_i,\ \text{clip}(s_i(\theta),1-\varepsilon,1+\varepsilon)\widehat A_i\Big)\right]$$

裁剪作用在**整条 response** $s_i$，而非每 token。advantage 是序列级 $\widehat A_i$。

### 3.3 与 Routing Replay 的关系

GSPO 内含一种 "Recompute Routing Replay"：recompute 阶段记路由、update 阶段回放——只修 mini-step 间的模型更新差，**不修训推引擎差**（见 [[R3 rollout routing replay]] §1 区分 R2/R3）。论文称 GSPO "eliminates the dependency on Routing Replay for MoE"——指其序列级聚合已能稳住 MoE，不依赖外部 R3。详见 [[token-level与sequence-level objective]]。

## 4. 数学原理 / 公式

### 4.1 几何均值降方差

逐 token ratio $r_t$ 的方差在序列上累乘；取几何均值 $s_i=(\prod r_t)^{1/T}$ 后 $\log s_i=\frac1T\sum\log r_t$ 是均值，方差 $\sim1/T$ 降。长序列收益大。

### 4.2 实验结果

- 比 GRPO 训练效率更高，稳 MoE RL（Qwen3 核心）
- 裁剪比例比 GRPO 高**两个数量级**，但效率更高
- 成功用于 Qwen3 训练

## 5. 代码示例

```python
import torch
def gspo_ratio(logp_new, logp_old):
    """序列级几何均值 ratio.
    logp_new/old: (G, T) 每条 response 逐 token logp
    返回 s_i: (G,) 每条一个 ratio
    """
    log_r = logp_new - logp_old            # (G,T)
    s = torch.exp(log_r.mean(dim=-1))      # 几何均值 = exp(算术均值 of log-ratio)
    return s
# 对比 GRPO token 级: 每 token 一个 r_t, 方差累乘爆炸
# GSPO: 每条一个 s_i, 方差 ~1/T
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[GRPO]]（GSPO 是序列级变体）、[[importance sampling与off-policy correction]]（IS/几何均值 ratio）、[[token-level与sequence-level objective]]（粒度权衡）、[[MoE路由]]（专家跳变是 token ratio 方差源）
- **下游（应用）**: [[训练推理不一致TIM技术综述]]（优化层 clip 族中点）、Qwen3 MoE RL、[[R3 rollout routing replay]]（GSPO 的 Recompute Routing Replay vs R3）
- **对比 / 易混**:
  - **GSPO（序列级）vs GRPO（token 级）**：粒度，GSPO 解方差爆炸。
  - **GSPO vs [[DPPO]]**：GSPO 仍是 ratio 代理（只改粒度）；DPPO 用散度替代 ratio（改代理本身）。GSPO 遗留 ratio-as-proxy 问题 → DPPO。
  - **GSPO 的 Recompute Routing Replay vs [[R3 rollout routing replay]] 的 Rollout Routing Replay**：前者只修 mini-step 更新差，后者修训推引擎差。mini_step=1 时前者失效，R3 仍有效。

## 7. 常见误区与易错点

> [!warning] 误区 1：序列级就一定更好
> 短序列上 sequence 级 IS 方差比 token 级高（VESPO 分析）。GSPO 在长序列/MoE 收益大。

> [!warning] 误区 2：GSPO 彻底解决了 ratio 代理问题
> 没有。它只改粒度，ratio 仍是分布偏移的代理。长尾词表上 ratio clip 的结构性偏差仍在 → 这正是 [[DPPO]] 的切入点。

> [!warning] 误区 3：GSPO 的 Routing Replay = R3
> GSPO 的是 Recompute Routing Replay（recompute 阶段记、update 阶段放，只修更新差）；R3 是 Rollout Routing Replay（rollout 阶段记、recompute+update 都放，修引擎差）。

## 8. 延伸细节

### 8.1 与 VESPO 的对照
VESPO（arXiv:2602.10693）理论分析：naive IS 无偏但高方差（序列级方差随长指数爆）；token 级 IS 有偏低方差；GSPO 长归一引入偏差换方差可控。

### 8.2 内容来源
$s_i$ 公式、目标、MoE ~10% 专家跳变、裁剪比例高两个数量级、Qwen3 应用均来自 arXiv:2507.18071（2025-07-24）全文已联网核实。全文翻译见 [[GSPO论文全文中文翻译]]。截至 2026-09。

---
相关: [[训练推理不一致TIM技术综述]] | [[GRPO]] | [[importance sampling与off-policy correction]] | [[token-level与sequence-level objective]] | [[R3 rollout routing replay]] | [[DPPO]] | [[DRPO]] | [[MoE路由]] | [[17-RL训推一体框架]]
