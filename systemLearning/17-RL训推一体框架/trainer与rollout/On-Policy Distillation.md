# On-Policy Distillation

> **所属章节**: [[trainer与rollout]]
> **所属模块**: [[17-RL训推一体框架]]
> **所属小节**: §81
> **别名**: OPD / On-Policy KD / GKD + PG 蒸馏 / 学生自采状态蒸馏
> **难度**: 高（需懂 [[SFT]]、[[RLHF (PPO)]]、[[PPO clipped objective]]、[[exposure bias]]、[[Agent rollout]]、teacher–student 双引擎）
> **来源**: ① Thinking Machines, *On-Policy Distillation*, https://thinkingmachines.ai/blog/on-policy-distillation/ （OPD 概念提出方，PG-as-reward 路线出处）；② verl 实现 `docs/algo/opd.md`（Jacob Helwig, 2026-05）、`verl/trainer/distillation/losses.py`、`verl/experimental/teacher_loop/`、`verl/experimental/agent_loop/`；③ 监督回传路径引 `arXiv:2306.13649`（verl 代码注释原文引用）。

> [!note] OPD ≠ OPO
> OPO（On-Policy *Optimization*）泛指"同策略采样训练"的一类范式（on-policy PG/PPO 都是），不是蒸馏方法。OPD（On-Policy *Distillation*）特指 **学生自采状态 + 教师逐 token 监督** 的蒸馏。上一轮把 OPD 误记成 OPO——本条按 OPD（蒸馏）写。

## 1. 一句话定义

**OPD（On-Policy Distillation）** 让**学生用自己当前策略 $\pi_\theta$ 采样得到的状态序列** $s_t$ 去问教师 $\nu$，拿回教师在这些**真实 reached 状态**上的逐 token 分布，再用 **GKD 前向 KL（稠密 token 监督）+ PG 反向 KL 单样本当 reward（策略梯度）** 两条路径联合训练——既保留了 RL 的 on-policy 状态对齐（消除 [[exposure bias]]），又把 KD 的逐 token 稠密监督注入每一步，而不是只在序列末尾给一个稀疏 reward。

## 2. 为什么需要它（动机与背景）

### 2.1 三法对比

| 范式 | 状态来源 | 监督信号 | 暴露偏差 | 信号密度 |
|---|---|---|---|---|
| 标准 KD / [[SFT]] | 教师强制（teacher-forced，用教师/数据自带 token） | 逐 token 教师分布 | **有**（学生推理时偏离教师轨迹就崩） | 稠密 |
| RLVR（[[RLHF (PPO)]] / GRPO 类） | 学生自采（on-policy） | 序列末尾稀疏 reward | 无 | 稀疏 |
| **OPD** | **学生自采（on-policy）** | **逐 token 教师分布** | **无** | **稠密** |

### 2.2 暴露偏差（exposure bias）

标准 KD/SFT 用 teacher-forcing：训练时输入的是教师/数据自带的 token，学生从不经历自己"走错一步后接下来怎么走"。推理时学生一旦在某步偏离，后续状态分布与训练分布不符——[[exposure bias]]，错误累积。RLVR 用学生自采状态解决了状态对齐，但只在序列末尾给 reward，信号稀疏、信用分配难。OPD 取两者之长：**状态用学生的（on-policy，无暴露偏差），监督用教师的逐 token（稠密）**。

### 2.3 驾校类比

- **标准 KD**：教练开车，学员坐副驾看（学员从不自己握方向盘走自己的路线）。
- **RLVR**：学员独自开完全程，到终点才被告知"及格/不及格"（不知道哪步错）。
- **OPD**：学员自己开车走自己的路线，教练坐副驾，每一步都给 token 级指点（"这里该变道""刚才那步偏了"）。既对齐学员真实驾驶状态，又有稠密反馈。

## 3. 核心概念详解

### 3.1 两条损失路径

OPD 的 loss 由两条路径组成，对应 `distillation_ppo_loss` 的两次调用（见 §5）：

| 路径 | 监督形式 | KL 方向 | 粒度 | 角色 |
|---|---|---|---|---|
| **GKD**（Generalized KD） | 教师分布稠密 token 监督 | 前向 KL $\nu\log\nu/\pi_\theta$ | Top-K token | 主监督（logits processor 里算） |
| **PG**（Policy Gradient） | 反向 KL 单样本当 reward | 反向 KL（单样本） | 采样 token | 当优势做策略梯度 |

### 3.2 为什么是前向 + 反向两条

- **GKD 用前向 KL**（$\mathbb{E}_{v\sim\nu}$）：前向 KL 是 mean-seeking，鼓励学生覆盖教师所有高概率模式，配合 Top-K 只在教师最可能的几个 token 上算，省显存又稳。这是"稠密监督"主路径。
- **PG 用反向 KL 单样本**（$\mathbb{E}_{y\sim\pi_\theta}$）：学生采样的 token $y_t$ 上算 $\log\nu(y_t)-\log\pi_\theta(y_t)$ 当 reward，再用 [[PPO clipped objective]] / REINFORCE 做策略梯度。反向 KL 是 mode-seeking，鼓励学生集中到教师最可能的模式。把蒸馏 loss 取负当 reward（见 §4.2）——**verl 注释原文**：`Use negative distillation loss as reward`（Thinking Machines 博客路线）。

两条路径可选其一或都开：`use_policy_gradient=True` 走 PG；`False` 则直接把蒸馏 loss 当监督 loss 回传（`arXiv:2306.13649` 路线）。

## 4. 数学原理 / 公式

### 4.1 GKD 前向 KL（Top-K）

$$L_{\text{GKD}}^{(k)}(s_t)=\sum_{v\in\text{TopK}(\nu(\cdot|s_t))}\nu(v|s_t)\Big[\log\nu(v|s_t)-\log\pi_\theta(v|s_t)\Big]$$

仅对教师 Top-K 个 token 求前向 KL，避开整张词表 softmax。student logits 在 logits processor 阶段就有，提前算出逐 token loss，再 all-gather 跨 SP/CP 组。

### 4.2 PG：蒸馏 loss 取负当 reward

$$\hat r_t=\log\nu(y_t|s_t)-\log\pi_\theta(y_t|s_t)$$

即学生采样 token $y_t$ 上的反向 KL 单样本。把它取负当 PPO 的 advantage：

$$L_{\text{PG}}=-\mathbb{E}\Big[\min\big(r_t(\theta)\hat A_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big],\quad \hat A_t=-L_{\text{distill}}\cdot(\text{detach})$$

verl 实现：`advantages=-distillation_losses.detach()`（[losses.py:283](../../../codes/verl/verl/trainer/distillation/losses.py#L283)）。`.detach()` 保证 reward 不传梯度到教师 logprob——教师 logprob 是常数 target。

## 5. 代码：verl 的 OPD 实现

### 5.1 双调用结构（`distillation_ppo_loss`）

[losses.py:165](../../../codes/verl/verl/trainer/distillation/losses.py#L165) 的 docstring 自述：**同一个函数被当两次用**——logit processor 与最终策略 loss：

```
[split seq across sp/cp] → [forward 出 student logits] → [logits processor 算 topk loss]
   → [all-gather topk loss] → [combine topk loss 与 policy loss]
```

- **第一次调用（logits processor）**：`student_logits is not None` → [L205](../../../codes/verl/verl/trainer/distillation/losses.py#L205) `return compute_topk_loss(...)`，在 SP/CP 切片上算 GKD Top-K loss，返回逐 token 张量，随后 all-gather。
- **第二次调用（final policy loss）**：`student_logits is None` → [L208](../../../codes/verl/verl/trainer/distillation/losses.py#L208) 调 `distillation_loss(...)` 算总蒸馏 loss，再 [L210](../../../codes/verl/verl/trainer/distillation/losses.py#L210) 看 `use_task_rewards`/`use_policy_gradient` 决定是否叠加 PPO loss。

这样设计是为**把 logits processor 的 forward 与最终 loss 的 forward 合一**：student logits 算一次既喂 GKD（processor 阶段）又喂 PG（final 阶段），省一次前向。

### 5.2 策略分发（`compute_topk_loss`）

[losses.py:137](../../../codes/verl/verl/trainer/distillation/losses.py#L137) 按 `config.strategy` 分发到不同后端的 `compute_forward_kl_topk`：

| strategy | 实现 |
|---|---|
| `fsdp` / `veomni` | `fsdp_losses.compute_forward_kl_topk` |
| `megatron` | `megatron_losses.compute_forward_kl_topk` |
| 其它 | `NotImplementedError` |

### 5.3 PG / 监督 分叉（`distillation_loss`）

[losses.py:268](../../../codes/verl/verl/trainer/distillation/losses.py#L268) `if loss_config.use_policy_gradient:` 分叉：

- **PG 路径**（[L270–283](../../../codes/verl/verl/trainer/distillation/losses.py#L270)）：`policy_loss_fn=get_policy_loss_fn(loss_config.policy_loss_mode)` → `policy_loss_fn(..., advantages=-distillation_losses.detach(), ...)`。蒸馏 loss 取负当 reward，走标准 PPO/REINFORCE 策略梯度。
- **监督路径**（[L289+](../../../codes/verl/verl/trainer/distillation/losses.py#L289) `else`）：`distillation_loss=agg_loss(distillation_losses, response_mask, ...)` 直接把蒸馏 loss 当监督 loss 回传（`arXiv:2306.13649` 路线）。

### 5.4 两条注册 loss

[losses.py:305](../../../codes/verl/verl/trainer/distillation/losses.py#L305) `@register_distillation_loss(names=["forward_kl_topk"], use_topk=True)` → `compute_forward_kl_topk`：GKD 前向 KL Top-K 主路径。
[losses.py:370](../../../codes/verl/verl/trainer/distillation/losses.py#L370) `@register_distillation_loss(...)`：PG 路径，[L398–399](../../../codes/verl/verl/trainer/distillation/losses.py#L398) `distillation_losses=kl_penalty(logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode)` 用 `kl_penalty`（复用 [[PPO clipped objective]] 体系的 `kl_penalty`，支持 k1/k2/k3 等估计器）算学生–教师 KL 当蒸馏 loss。

### 5.5 教师 logprob：Agent Loop

教师 $\nu$ 不在训练进程里，靠独立推理服务出 logprob，链路（`verl/experimental/`）：

```
AgentLoopManager → AgentLoopWorker._compute_teacher_logprobs
  → AsyncTeacherLLMServerManager → LLMServerClient
```

教师推理参数（关键）：`max_tokens=1`（只回填 1 个 token 的位置即可拿 prompt_logprobs）、`prompt_logprobs=topk`（返回每个 prompt 位置的 Top-K logprob）、`temperature=1.0`（教师采样分布，非贪婪）。文件：`teacher_loop/teacher_manager.py`、`teacher_loop/teacher_model.py`、`agent_loop/agent_loop.py`、`agent_loop/single_turn_agent_loop.py`、`agent_loop/tool_agent_loop.py`、`agent_loop/tool_parser.py`、`agent_loop/utils.py`。

教师 logprob 以 `teacher_logprobs (bsz,seqlen,topk)` + `teacher_ids (bsz,seqlen,topk)` 形式塞进 data，供 GKD Top-K 对齐用（见 docstring `data` 字段）。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[SFT]]（OPD 是其 on-policy 升级，解决暴露偏差）、[[RLHF (PPO)]] / [[PPO clipped objective]]（PG 路径复用 PPO loss 与 `kl_penalty`）、[[exposure bias]]（OPD 的核心动机）、[[Agent rollout]]（教师推理走 Agent Loop，与多轮 tool-calling rollout 同源）
- **下游（应用）**: verl `distillation_ppo_loss`（student–teacher 双引擎蒸馏训练）、小模型蒸馏、reasoning 模型蒸馏
- **对比 / 易混**:
  - **OPD vs 标准 KD/[[SFT]]**：状态来源不同——OPD 学生自采（on-policy），KD teacher-forced（off-policy 状态）→ OPD 无暴露偏差。
  - **OPD vs RLVR**：信号密度不同——OPD 逐 token 稠密，RLVR 序列末稀疏。
  - **OPD vs OPO**：OPD 是蒸馏（有教师逐 token 监督）；OPO（On-Policy Optimization）泛指同策略采样训练，无教师。同名易混，见顶部 note。
  - **GKD vs PG 两条路径**：GKD 前向 KL 稠密监督（mean-seeking）；PG 反向 KL 单样本当 reward（mode-seeking）。可单用可合用。

## 7. 常见误区与易错点

- ❌ 把 OPD 当成普通 offline KD——普通 KD 用 teacher-forced 状态，学生训练分布与推理分布不一致（暴露偏差）；OPD 的状态是学生自己采的，对齐推理分布。
- ❌ 以为 GKD 与 PG 二选一必须——verl 用 `use_policy_gradient` 开关：`True` 走 PG（蒸馏 loss 当 reward），`False` 直接监督回传（`arXiv:2306.13649`），两条路径独立可配。
- ❌ 忘了 `.detach()`——PG 路径 `advantages=-distillation_losses.detach()`，reward 不能传梯度回教师 logprob（教师是常数 target）；漏 detach 会把梯度灌进 teacher logprob 通道。
- ❌ 以为教师要完整 softmax——GKD 只取教师 Top-K（`prompt_logprobs=topk`），省显存；`max_tokens=1` 因为只需 prompt 位置的 logprob，不需教师续写。
- ❌ 把 OPD 与 OPO 混记——见顶部 note，OPD=蒸馏，OPO=同策略优化泛称。

## 8. 延伸细节

- **概念出处**：OPD 作为命名方法由 Thinking Machines 博客提出（PG-as-reward 路线）；verl `docs/algo/opd.md`（Jacob Helwig, 2026-05）给出工程实现。监督回传分支 verl 注释引 `arXiv:2306.13649`（按代码注释原文引用，未独立核实其标题，待核实）。
- **双调用的工程动机**：logits processor 阶段与最终 loss 阶段都需要 student logits——合一避免重复前向，SP/CP 切片上算 Top-K 再 all-gather，省通信与显存。
- **教师推理复用 Agent Loop**：教师 logprob 获取走 `agent_loop/`（本用于多轮 tool-calling rollout），`max_tokens=1 + prompt_logprobs=topk` 把它退化成"只回填 prompt 位置 logprob"的批量推理服务，与 [[多轮Agent rollout与tool calling]] 同源。
- **`kl_penalty` 复用**：PG 路径的 `kl_penalty(...)` 与 [[KL penalty与KL control]] / [[rollout correction与IS weight]] 体系共用 k1/k2/k3 估计器，蒸馏 loss 的 KL 形式可配。

---
相关: [[SFT]]｜[[RLHF (PPO)]]｜[[PPO clipped objective]]｜[[exposure bias]]｜[[Agent rollout]]｜[[多轮Agent rollout与tool calling]]｜[[KL penalty与KL control]]｜[[trainer与rollout引擎组合]]｜[[V1 async trainer]]｜[[Decoupled PPO]]｜[[17-RL训推一体框架]]
