# rollout correction与IS weight

> **所属章节**: [[logprob一致性]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §83
> **别名**: verl rollout correction / IS-RS correction framework / 三策略 off-policy 修正 / decoupled PPO
> **难度**: 高（需懂 [[importance sampling与off-policy correction]]、[[TIS]]、[[rollout train reference logprob一致性]]、[[PPO clipped objective]]、[[训推不一致]]、[[训练推理不一致TIM技术综述]]）
> **来源**: verl 源码与官方文档已读全文：`verl/trainer/ppo/rollout_corr_helper.py`（完整代码）、`verl/trainer/ppo/ray_trainer.py:1570-1660`（bypass/decoupled 调用点）、`docs/algo/rollout_corr.md`（2025-10-30 用法指南）、`docs/algo/rollout_corr_math.md`（2025-11-04 数学推导）、`verl/trainer/config/algorithm/rollout_correction.yaml`。引用论文/博客：Liu, Li et al. *When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch* (richardli.xyz/rl-collapse, 2025-09)；Li et al. *Trust Region Masking for Long-Horizon LLM RL* (arXiv:2512.23075)；Hilton et al. *Batch size-invariance for policy optimization* (arXiv:2110.00641, Decoupled PPO 理论源头)；Williams 1992 REINFORCE；Schulman 2017 PPO。

## 1. 一句话定义

**verl rollout correction** 是 verl 的统一 off-policy 修正框架，把"数据采集分布 ≠ 训练分布"的偏差用两个**正交**手段一起治：**(a) IS 权重**（连续软性重加权，`rollout_is_weights`，降方差）+ **(b) Rejection Sampling**（硬二值 mask，改 `response_mask` 丢 outlier）；建立在上层 **三策略框架**（π_rollout 行为策略 / π_old 近端锚点 / π_θ 当前策略）之上，用 **decoupled mode**（3 策略，重算 π_old，pre-batch 修正，batch-size 不变）或 **bypass mode**（2 策略，π_old=π_rollout，跳过重算，in-loss 修正）两种操作模式驱动，是 [[TIS]]/IcePop/MIS/K1·K2·K3-RS 等方法在 verl 里的统一实现底座。

## 2. 为什么需要它（动机与背景）

### 2.1 on-policy 假设的三个 off-policy 来源
RL 标准假设 on-policy——用当前 π_θ 采样、用当前 π_θ 算梯度。实际有三个 off-policy 来源（`rollout_corr_helper.py` 模块 docstring §1-3、`rollout_corr.md` §Overview）：
1. **实现不一致（policy mismatch）**：rollout 用 vLLM（BFloat16、有采样截断/kernel 差），训练用 FSDP/Megatron（FP32），同权重两边 logp 不等——即 [[训推不一致]]（TIM）
2. **权重 staleness**：async 模式下 rollout 用几步前的旧权重（`parameter_sync_step>1` 时尤甚，见 [[V1 async trainer]]）
3. **partial rollout**：一条 trajectory 跨多个模型版本（KV 前缀续采）

直接当 on-policy 用会崩——"When Speed Kills Stability" 博客（richardli.xyz/rl-collapse）讲的就是这个。

### 2.2 朴素 PPO 的实现错误（§3.7 关键）
许多 LLM-RL 实现错误地用 PPO 时**忽略真实 π_rollout**、假设 $\pi_{old}=\pi_{rollout}$：

$$L_{\text{PPO}}(\theta)=-\mathbb{E}_t[\min(r_t A_t,\ \text{clip}(r_t,1\pm\varepsilon)A_t)],\quad r_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}\ \text{(忽略 }\pi_{\text{rollout}}\text{)}$$

**为什么错**：LLM-RL 里 rollout 与训练用不同精度/后端/checkpoint，$\pi_{\text{rollout}}\ne\pi_{\text{old}}$ 即便同权重。**这不是 PPO 的错**——PPO 本身数学正确，错在"π_old=π_rollout"的实现假设。该错误被 "When Speed Kills Stability" 博客识别为 LLM-RL 崩溃主因，催生了本框架。

### 2.3 正确做法：三策略解耦
- **Decoupled mode**：三策略（π_rollout, π_old, π_θ），IS 修正 rollout→old 的漂移
- **Bypass mode**：两策略（π_rollout=π_old, π_θ），直接用 rollout 当 PPO 锚
- **Bypass + PG mode**：两策略（π_rollout, π_θ），IS/RS 修正 + 无 PPO clip

## 3. 核心概念详解（rollout correction / IS weight 细说）

### 3.1 三策略框架与两类漂移
verl 用三个策略各司其职（`rollout_corr_math.md` §2.1）：

| 策略 | 角色 | 何时建 | 是否冻结 |
|---|---|---|---|
| **π_rollout**（行为策略 μ） | 数据采集分布 | rollout 阶段 | 训练该 batch 时冻结 |
| **π_old**（近端策略 π_prox） | PPO clip 锚点 | decoupled: epoch 开头 `actor.compute_log_prob()` 重算；bypass: =π_rollout | 整个 epoch 所有 mini-batch 冻结 |
| **π_θ**（当前策略） | 被优化的策略 | 每梯度步更新 | — |

由此分**两类漂移**，分别由不同机制修正：
- **Drift 1（rollout→old，off-policy gap）**：$\rho_t=\frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$，由 **IS 权重**修正（本笔记主题）
- **Drift 2（old→θ，策略更新漂移）**：$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$，由 **PPO clip** 修正（见 [[PPO clipped objective]]、[[importance sampling与off-policy correction]]）

### 3.2 两种操作模式（ray_trainer.py:1570-1660）
调用点（已读源码）：

```python
# ray_trainer.py:1574-1575
rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
```

**Decoupled mode（`bypass_mode=False`）**：
- `ray_trainer.py:1584-1610`：`_compute_old_log_prob()` 重算 π_old（3 策略）
- `ray_trainer.py:1647-1660`：**pre-batch** 调 `compute_rollout_correction_and_add_to_batch()`——用稳定 π_old vs π_rollout 一次性算 IS 权重 + RS mask + metrics，加进 batch
- 代码注释明示："Only runs in decoupled mode (computes once per batch using stable π_old)"
- 性质：✅ batch-size 不变 ✅ 分开修正两类漂移 ❌ 多一次 forward

**Bypass mode（`bypass_mode=True`）**：
- `ray_trainer.py:1576-1583`：`apply_bypass_mode()` 设 `old_log_probs=rollout_log_probs`（2 策略）、`policy_loss_config["loss_mode"]="bypass_mode"`、把 `rollout_correction` config 透传给 actor
- `ray_trainer.py:1647-1654`：**跳过** pre-batch `compute_rollout_correction_and_add_to_batch`（注释："In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout"）
- IS/RS 改在 actor 的 `compute_policy_loss_bypass_mode()`（`core_algos.py`）里用 π_θ/π_rollout **in-loss 现算**
- 性质：✅ 跳过重算 forward（快）✅ in-loss 用 π_θ 而非冻结 π_old ⚠️ 不达 batch-size 不变

> [!note] bypass 不等于"不做 correction"
> 用户摘要里"bypass 不重算、不做 correction"是**半对**：pre-batch helper 被跳过，但 `compute_policy_loss_bypass_mode` 仍按 `rollout_correction` config 做 IS/RS（只是分母用 π_rollout、分子用 π_θ 而非 π_old）。preset `bypass_pg_geo_rs`/`bypass_ppo_clip_k3_rs` 就是 bypass 下做 RS 的明证。

### 3.3 (a) IS 权重：连续软性重加权
核心（`rollout_corr_helper.py:compute_rollout_correction_weights`，log 空间避免溢出）：

$$\text{log\_ratio}=\text{old\_log\_probs}-\text{rollout\_log\_probs}=\log\frac{\pi_{\text{old}}}{\pi_{\text{rollout}}}$$

$$\text{IS\_weight}=\exp(\text{clamp}(\text{log\_ratio},\ -\text{SAFETY\_BOUND},\ \text{SAFETY\_BOUND})),\quad \text{SAFETY\_BOUND}=20$$

`exp(±20)≈[2e-9, 4.85e8]`，防溢出。两档粒度（`rollout_is` 参数）：
- **token**：每 token 一个权重 $w_t=\exp(\text{log\_ratio}_t)$，独立截断，**低方差有偏**（bias $\sim O(T^2\Delta_{\max})$）
- **sequence**：整条序列一个权重 $w_{\text{seq}}=\exp(\sum_t\text{log\_ratio}_t)=\prod_t\rho_t$，广播到所有 token，**无偏高方差**

**截断（TIS）**：`.clamp(max=rollout_is_threshold)`——**只截上限**，默认 `2.0`。不截下限以保小权重无偏。下界 `1/threshold` 仅用于**metrics 诊断**（`ratio_fraction_low`），不参与实际截断。

**IcePop 模式**：`rollout_is_threshold="0.5_5.0"` 字符串（lower_upper）→ 超出 `[lower,upper]` 的 token 权重**置零**（不是 clamp），双边 mask。与 [[IcePop]] 同源。

**Batch normalize**（`rollout_is_batch_normalize=True`）：把权重归一化到均值 1（token 级对所有 token 归一；sequence 级对序列均值归一），分布式下用 `distributed_masked_mean`，在截断**之后**应用以保留截断语义。

**关键 `.detach()`**：`rollout_is_weights = rollout_is_weights.detach()`——IS 权重"改变测度"不参与梯度，符合 IS 理论（§3.2.2 数学推导）。若不 detach，autograd 会多算一项 $\log\pi_\theta\cdot\nabla_\theta w(\theta)$ 的 bias term。

### 3.4 (b) Rejection Sampling：硬二值丢 outlier
不靠权重软性修正，而是直接把 outlier 样本从训练里踢掉——通过修改 `response_mask`（被踢 token mask 置 0）。`SUPPORTED_ROLLOUT_RS_OPTIONS`（`rollout_corr_helper.py:77-89`）：

| 粒度 | K1（对称 ratio） | K2（$\frac12(\log r)^2$） | K3（$e^{\log r}-1-\log r$） |
|---|---|---|---|
| token | `token_k1` | `token_k2` | `token_k3` |
| seq 求和 | `seq_sum_k1` | `seq_sum_k2` | `seq_sum_k3` |
| seq 均值 | `seq_mean_k1` | `seq_mean_k2` | `seq_mean_k3` |
| seq 最大 | — | `seq_max_k2` | `seq_max_k3` |

- **K1** = $-\text{log\_ratio}$，对称区间 $[\log(\text{lower}),\log(\text{upper})]$，threshold 用 `"0.5_2.0"` lower_upper 字符串；只给 float 则下界默认倒数
- **K2** = $0.5\cdot\text{log\_ratio}^2$，只设上界（$\ge0$），近似 $\frac12\text{Var}[\log\rho]$
- **K3** = $\exp(\text{log\_ratio})-1-\text{log\_ratio}$，只设上界，$\ge0$ per token，期望 $=\text{KL}(\pi_{\text{rollout}}\|\pi_{\text{old}})$，小 KL 时比直接 KL 稳
- 多选项逗号分隔（`"token_k1,seq_mean_k3"`），所有选项都过才保留（**logical AND**）

### 3.5 两者关系：解耦设计
- **RS 总是修改 `response_mask`**（当 `rollout_rs` 设了），无条件
- **IS 权重仅当 `rollout_is` 配置时才加进 batch**
- 可以"先只开 RS 看 metrics，再决定要不要上 IS 权重"

四象限（`rollout_corr.md` 表）：
| `rollout_is` | `rollout_rs` | 行为 |
|---|---|---|
| null | null | 禁用（只算 metrics） |
| null | 设了 | **只丢**（无权重修正） |
| 设了 | null | **只加权**（不丢） |
| 设了 | 设了 | **全修正**（既加权又丢） |

## 4. 数学原理 / 公式

### 4.1 IS 权重处理流水线（4 阶段，`rollout_corr.md` §Summary）
1. **Safety bound**：$\exp(\text{clamp}(\text{log\_ratio},-20,20))$ → 每 token $\in[2\text{e-}9,4.85\text{e8}]$
2. **TIS 截断**：$\text{clamp}(\cdot,\max=C_{\text{IS}})$（只上限）
3. **Padding 置零**：$\times\ \text{response\_mask}$
4. **可选 batch 归一**：$\tilde w_t=w_t/\mathbb{E}[w]$（截断后应用）

### 4.2 K1/K2/K3 散度估计
$$K_1=-\log\rho,\quad K_2=\frac12(\log\rho)^2,\quad K_3=e^{\log\rho}-1-\log\rho=\rho-\log\rho-1$$
- $K_3$ 期望 $=\text{KL}(\pi_{\text{rollout}}\|\pi_{\text{old}})$（因 $\mathbb{E}_{\pi_r}[\rho]=1$，$\mathbb{E}_{\pi_r}[\log\rho]=-\text{KL}$）
- 小漂移 $K_3\approx\frac12\text{Var}(\log\rho)\approx K_2$

### 4.3 stopgrad 理论（§3.2.2）
目标 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[\sum_t A_t]$，off-policy 下：
$$\nabla_\theta J=\mathbb{E}_{\tau\sim\pi_{\text{rollout}}}\!\left[w(\tau;\theta)\sum_t A_t\nabla_\theta\log\pi_\theta(a_t|s_t)\right]$$
$w(\tau;\theta)$ 是换测度的系数，**不是被优化的对象**。若不 detach：
$$\nabla_\theta[w(\theta)\log\pi_\theta]=\underbrace{\log\pi_\theta\cdot\nabla_\theta w(\theta)}_{\text{WRONG: bias}}+\underbrace{w(\theta)\cdot\nabla_\theta\log\pi_\theta}_{\text{CORRECT}}$$
PyTorch 里必须 `loss = -A * log_prob * rollout_is_weights.detach()`。

### 4.4 诊断 metrics（`compute_offpolicy_metrics`）
- **ESS**：$1/\mathbb{E}[\tilde w^2]\in[0,1]$，越低权重越集中
- **χ²_token**：$\mathbb{E}[\rho^2]-1$；**χ²_seq**：$\mathbb{E}[(\prod_t\rho_t)^2]-1$
- **KL**：$\mathbb{E}[\log\pi_r-\log\pi_{\text{old}}]$（direct，可负）；**K3_KL**：$\mathbb{E}[\rho-\log\rho-1]$（$\ge0$，小 KL 更稳）
- **PPL_ratio**：$\exp(\mathbb{E}[\log\text{PPL}_{\text{old}}-\log\text{PPL}_r])$，$>1$ 训练不如 rollout 自信

## 5. 代码示例

### 5.1 核心 API（`rollout_corr_helper.py`）
```python
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

# 返回 (IS权重DataProto, 修改后的response_mask, metrics)
weights_proto, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
    old_log_prob=old_log_probs,        # (B,T) 训练端 π_old logp（decoupled 重算 / bypass=rollout）
    rollout_log_prob=rollout_log_probs, # (B,T) rollout 端 π_rollout logp
    response_mask=response_mask,        # (B,T) 1=valid
    rollout_is="token",                 # "token" | "sequence" | None
    rollout_is_threshold=2.0,           # float=TIS上限 | "0.5_5.0"=IcePop双边
    rollout_is_batch_normalize=False,
    rollout_rs="seq_mean_k3",           # None | "token_k1,seq_max_k2" | ...
    rollout_rs_threshold=0.01,          # K1用"lo_hi"，K2/K3用float
)
# IS 权重: safety-bound → TIS截断 → padding置零 → (可选)归一 → .detach()
# RS: 修改 response_mask（超阈 token/序列置0）
# metrics 全带 "rollout_corr/" 前缀
```

### 5.2 preset API（`RolloutCorrectionConfig`，13 个验证预设）
```python
from verl.trainer.config.algorithm import RolloutCorrectionConfig as RCfg

# Decoupled（3 策略）
cfg = RCfg.decoupled_token_is()          # Token-TIS
cfg = RCfg.decoupled_seq_is_rs()        # Seq-MIS（IS+RS）
cfg = RCfg.decoupled_geo_rs_token_tis() # Geo-RS + Token-TIS（长 CoT 推荐）
cfg = RCfg.decoupled_k3_rs_token_tis()  # K3 filter + Token-TIS（小 KL）

# Bypass（2 策略，快）
cfg = RCfg.bypass_ppo_clip()            # PPO-clip only（ratio 已处理 IS）
cfg = RCfg.bypass_ppo_clip_geo_rs()     # PPO-clip + Geo-RS
cfg = RCfg.bypass_pg_is()               # REINFORCE + Seq-TIS（无 clip）
cfg = RCfg.disabled()                   # 只看 metrics
```

### 5.3 YAML
```yaml
algorithm:
  rollout_correction:
    rollout_is: token            # null | "token" | "sequence"
    rollout_is_threshold: 2.0    # float=TIS | "0.5_5.0"=IcePop
    rollout_is_batch_normalize: false
    rollout_rs: null             # "token_k1,seq_mean_k3" | ...
    rollout_rs_threshold: null   # K1:"lo_hi" | K2/K3:float
    bypass_mode: false           # false=Decoupled(3策略) | true=Bypass(2策略)
    loss_type: ppo_clip           # bypass下: "ppo_clip" | "reinforce"
actor_rollout_ref:
  rollout:
    calculate_log_probs: true    # 必须！rollout 端要算 rollout_log_probs
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[importance sampling与off-policy correction]]（IS 理论、ratio、V-trace——本笔记是其 verl 实现底座）、[[TIS]]（本笔记 IS 权重的默认截断模式 $w=\min(\rho,C)$）、[[PPO clipped objective]]（Drift 2 的 clip 修正）、[[rollout train reference logprob一致性]]（bypass/recompute 的 TIM 治理上下文）、[[训推不一致]]（off-policy 来源①的实现差机理）、[[训练推理不一致TIM技术综述]]（本笔记是综述里版本/staleness 层 + 引擎层 IS 兜底的统一框架）、[[IcePop]]（`rollout_is_threshold="lo_hi"` 即 IcePop 模式）、[[TRM]]（arXiv:2512.23075，本框架 K3-RS 的长程理论延伸）
- **下游（应用）**: verl 生产 RL 训练（`ray_trainer.py` + `core_algos.py`）、[[V1 async trainer]]（异步 staleness 的 IS 兜底）、[[R3 rollout routing replay]]（MoE 路由 TIM，与本框架正交：R3 治引擎层路由差，本框架治 off-policy 分布差）
- **对比 / 易混**:
  - **IS 权重（软加权）vs RS（硬丢）**：前者改测度降方差、保留所有样本信号；后者改 mask 删 outlier、降有效样本量。正交可独立开关。
  - **decoupled（3 策略）vs bypass（2 策略）**：前者重算 π_old、pre-batch 修正、batch-size 不变；后者 π_old=π_rollout、跳重算、in-loss 修正、快。
  - **TIS（只截上限）vs IcePop（双边置零）**：`rollout_is_threshold=float` 是 TIS；`="lo_hi"` 是 IcePop。
  - **rollout correction vs R3**：R3 修 MoE 路由离散跳变（引擎层根因）；rollout correction 修 off-policy 分布差（版本/staleness/实现差）。叠加正交。
  - **本框架 vs 朴素 PPO**：朴素 PPO 错误假设 π_old=π_rollout；本框架显式三策略解耦，是"正确实现的 PPO"。

## 7. 常见误区与易错点

> [!warning] 误区 1：TIS 把权重截断在 $[1/C,\ C]$
> 不是。TIS 是 `.clamp(max=C)` **只截上限**，不截下限（保小权重无偏）。$1/C$ 只用于 metrics 诊断 `ratio_fraction_low`，不参与实际截断。双边截断（置零）是 IcePop 模式（`"lo_hi"` 字符串），不是 TIS。

> [!warning] 误区 2：bypass 模式不做 correction
> 半对。pre-batch `compute_rollout_correction_and_add_to_batch` 在 bypass 被跳过（`ray_trainer.py:1653` 的 `not bypass_recomputing_logprobs` 条件），但 `compute_policy_loss_bypass_mode` 仍按 config 做 IS/RS——只是分母用 π_rollout、分子用 π_θ 而非冻结 π_old。preset `bypass_pg_geo_rs` 就是 bypass 下做 RS 的明证。

> [!warning] 误区 3：IS 权重参与梯度
> 不能。`.detach()` 是 IS 理论的数学要求（§3.2.2），不是实现细节。不 detach 会多算 $\log\pi_\theta\cdot\nabla_\theta w$ 的 bias term，优化错误目标。

> [!warning] 误区 4：PPO ratio 和 IS 权重一起用
> 看模式。bypass+ppo_clip 里 PPO ratio $r=\pi_\theta/\pi_{\text{rollout}}$ 已处理 IS，再加显式 IS weights 是 double-counting。bypass+reinforce 才用显式 IS weights（无 clip）。decoupled 里 IS 权重 $w=\pi_{\text{old}}/\pi_{\text{rollout}}$ 修 Drift1、PPO ratio $r=\pi_\theta/\pi_{\text{old}}$ 修 Drift2，两者管不同漂移不冲突。

> [!warning] 误区 5：朴素 PPO 假设 π_old=π_rollout 没问题
> 错。rollout 用 vLLM BF16 与训练 FP32 不等（[[训推不一致]]），$\pi_{\text{rollout}}\ne\pi_{\text{old}}$。"When Speed Kills Stability" 把这列为 LLM-RL 崩溃主因。必须三策略解耦或 bypass 显式承认。

> [!warning] 误区 6：K1/K2/K3 是三种不同散度
> 是同一 log_ratio 的三种变换：K1=$-\log\rho$（对称 ratio）、K2=$\frac12(\log\rho)^2$（二次）、K3=$\rho-\log\rho-1$（=反 KL，$\ge0$）。期望都关联 $\text{KL}(\pi_r\|\pi_{\text{old}})$，但数值稳定性不同：K3 per-token 非负最稳，K1 对称可判方向，K2 近似方差。

## 8. 延伸细节

### 8.1 preset 全表（13 个，`rollout_corr_math.md` §5.1）
| preset | 估计器 | 模式 | IS 级 | RS 级 |
|---|---|---|---|---|
| `decoupled_token_is` | Token-TIS | Decoupled | token | — |
| `decoupled_seq_is` | Seq-TIS | Decoupled | sequence | — |
| `decoupled_seq_is_rs` | Seq-MIS | Decoupled | sequence | sequence(seq_sum_k1) |
| `decoupled_geo_rs` | Geo-RS | Decoupled | — | sequence(seq_mean_k1) |
| `decoupled_geo_rs_token_tis` | Geo-RS-Token-TIS | Decoupled | token | sequence |
| `decoupled_k3_rs` | K3-RS | Decoupled | — | sequence(seq_mean_k3) |
| `decoupled_k3_rs_token_tis` | K3-RS-Token-TIS | Decoupled | token | sequence |
| `bypass_ppo_clip` | — | Bypass(ppo_clip) | — | — |
| `bypass_ppo_clip_geo_rs` | Geo-RS | Bypass(ppo_clip) | — | sequence |
| `bypass_ppo_clip_k3_rs` | K3-RS | Bypass(ppo_clip) | — | sequence |
| `bypass_pg_is` | Seq-TIS | Bypass(reinforce) | sequence | — |
| `bypass_pg_geo_rs` | Geo-RS | Bypass(reinforce) | — | sequence |
| `bypass_pg_geo_rs_token_tis` | Geo-RS-Token-TIS | Bypass(reinforce) | token | sequence |

### 8.2 Length Trap（长序列陷阱，§3.3.3）
标准 IS 估计器有系统性**长度偏差**：$\rho(y)=\prod_t\rho_t$，每 token 平均 ratio=1.1 时，10 token → $\rho\approx2.6$（保留），100 token → $\rho\approx13780$（超阈被丢）→ **Context Collapse**：模型只学短答、丢长 CoT。**Geo-RS** 用几何均值 $\rho_{\text{geo}}=\rho^{1/T}$ 归一化长度，长短序列同 trust score。阈值极紧（`"0.999_1.001"` ~±0.1%）。长 CoT/agent 必用 Geo-RS 或 K3-RS。

### 8.3 推荐工作流（`rollout_corr.md` §Example Workflow）
1. 先 `rollout_is=null, rollout_rs=null, bypass_mode=true` 只看 metrics（`kl`/`log_ppl_abs_diff`/`chi2_token`）评估 off-policy gap
2. gap 大则开 RS（`rollout_rs="seq_mean_k1"` 或 `"seq_mean_k3"`）丢 outlier
3. 仍不稳再上 IS 权重（`rollout_is="sequence"` + `loss_type="reinforce"`）

### 8.4 性能与文件
- 开销：~1% 显存、1-3% 计算（`rollout_corr.md` §Performance）
- 文件：`verl/trainer/ppo/rollout_corr_helper.py`（核心）、`verl/trainer/ppo/core_algos.py`（`compute_policy_loss_bypass_mode`/`compute_policy_loss_reinforce`）、`verl/trainer/config/algorithm.py`（`RolloutCorrectionConfig`）、`verl/trainer/config/algorithm/rollout_correction.yaml`、测试 `tests/trainer/ppo/test_rollout_corr.py`/`test_rollout_corr_integration.py`、示例 `examples/rollout_correction/`

### 8.5 内容来源
API、公式、三策略框架、bypass/decoupled 调用点（ray_trainer.py:1570-1660）、K1/K2/K3 定义、stopgrad 推导、preset 表、Length Trap、metrics 清单均来自 verl 源码与 `docs/algo/rollout_corr.md`/`rollout_corr_math.md` 全文已读核实。Decoupled PPO 理论源于 Hilton et al. arXiv:2110.00641。REINFORCE 源于 Williams 1992。截至 2026-09。

---
相关: [[importance sampling与off-policy correction]] | [[TIS]] | [[IcePop]] | [[TRM]] | [[rollout train reference logprob一致性]] | [[PPO clipped objective]] | [[训推不一致]] | [[训练推理不一致TIM技术综述]] | [[V1 async trainer]] | [[R3 rollout routing replay]] | [[17-RL训推一体框架]]
