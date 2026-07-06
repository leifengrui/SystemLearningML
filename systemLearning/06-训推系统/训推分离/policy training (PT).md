# policy training (PT)

> **所属章节**: [[训推分离]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（总览性，整合训练架构与 RLHF 流程）

## 1. 一句话定义

**policy training（PT，策略训练侧）** 是 [[训推分离]] 的**训练侧**：负责 LLM 训练全流程（预训练 → [[SFT]] → [[Reward Model]] → [[PPO optimization]]），用 [[learner worker split]] 架构（learner 训练 + [[rollout worker]] 采样）。PT 专注**训练优化**（FSDP/DeepSpeed/Megatron 的显存通信优化、大规模训练 GPU 集群），产出训练好的 $\theta$，通过 [[weight sync mechanism]] 推送给 [[policy deployment (PD)]] 部署。与 PD 分离是因为训练与部署的**负载/框架/资源/优化点完全不同**——训练重算力显存、部署重延迟并发。在 RLHF 中，PT = SFT + RM + PPO 三阶段，是产出对齐 LLM 的"工厂"。

> [!note] PT 的范畴
> PT 不只是 RLHF 的 PPO，而是**整个训练侧**：预训练（next-token）、SFT（指令微调）、RM（偏好学习）、PPO（强化对齐）。[[PPO optimization]] 是 PT 在 RLHF 阶段的核心，但 PT 更广。

## 2. 为什么需要它（动机与背景）

LLM 的训练与部署是两种截然不同的负载：
1. **负载不同**：训练重 forward+backward+optimizer（算力/显存密集），部署重前向生成（延迟/并发密集）；
2. **框架不同**：训练用 FSDP/DeepSpeed/Megatron（显存切分/通信优化），部署用 vLLM/TensorRT-LLM（KV cache/连续批处理/量化）；
3. **资源不同**：训练需大集群（多节点多卡，IB 网络），部署需服务副本（多卡，负载均衡）；
4. **稳定性要求不同**：训练容忍偶发重启（checkpoint 恢复），部署要求高可用（零停机）；
5. **持续训练 vs 固定部署**：PT 持续训新版本，PD 部署固定版本（定期升级）。

故需 PT/PD 分离：PT 专注训练优化，PD 专注部署服务，通过 weight sync 解耦。PT 持续训练不干扰 PD 服务，PD 热切换新版本不停机。

## 3. 核心概念详解

### 3.1 PT 的训练全流程

```
PT 侧流程:
1. 预训练(next-token)        → π_pretrained
2. SFT(指令微调)             → π_SFT        ← [[SFT]]
3. RM(偏好学习)              → r_φ          ← [[Reward Model]]
4. PPO(KL-constrained RL)    → π_θ(对齐后)  ← [[PPO optimization]]
   └─ learner + rollout worker(批1架构)
5. checkpoint 保存            → θ 推送给 PD  ← [[weight sync mechanism]]
```

### 3.2 PT 的架构（批1的具体化）

PT 在 RLHF-PPO 阶段用 [[learner worker split]]：
- **learner**：跑 actor + critic 的 forward/backward/optimizer step（FSDP/DeepSpeed）；
- **rollout worker**：跑 actor 自回归生成（vLLM/SGLang）；
- **reward worker**（可选）：跑冻结 RM/reference；
- **buffer**：trajectory 数据管道；
- **weight sync**：learner → rollout worker 同步新 $\theta$。
详见 [[PPO optimization]] §8.3、[[learner worker split]] §3.5。

### 3.3 PT 的资源与框架

| 维度 | PT 侧 |
|---|---|
| **硬件** | 训练 GPU 集群（多节点多卡，A100/H100，IB/NVLink 网络） |
| **框架** | FSDP/DeepSpeed/Megatron-LM（显存切分/通信优化） |
| **并行** | DP + TP + PP（3D parallelism，[[3D parallelism]]） |
| **显存** | $\theta$ + optimizer state + activation + gradient checkpointing |
| **优化** | 混合精度（BF16）、梯度累积、activation checkpointing |

### 3.4 PT 与 PD 的接口

| 接口 | 机制 | 详见 |
|---|---|---|
| **checkpoint** | PT 保存 $\theta$ 到存储（S3/HDFS），PD 加载 | [[weight sync mechanism]] |
| **weight sync** | PT 流式推送 $\theta$ 给 PD（热切换） | [[weight sync mechanism]] |
| **版本协议** | PT 标记版本号/commit，PD 按版本切换 | §3.5 |
| **回滚** | PD 出问题回退到上一 PT 版本 | checkpoint 管理 |

### 3.5 PT 的版本管理

- **版本号**：每次 PT 训出 $\theta$ 标记版本（如 `v1.3-rlhf-20260705`）；
- **checkpoint**：保存 $\theta$ + optimizer state（可恢复训练）+ metadata（超参/数据版本）；
- **PD 升级**：PD 加载新 checkpoint，blue-green 热切换（新版本起来验证后切流量）；
- **回滚**：新版本异常，PD 回退上一 checkpoint。

### 3.6 PT 的监控

- **训练指标**：loss、reward、KL、entropy、clip_fraction（[[PPO optimization]] §8.7）；
- **资源**：GPU 利用率、显存、通信带宽；
- **吞吐**：steps/sec、tokens/sec；
- **稳定性**：loss NaN、重启次数、checkpoint 完整性。
这些是 PT 运维的仪表盘。

## 4. 数学原理 / 公式

### 4.1 PT 的训练时间

$$
T_{\text{PT}}=\frac{N_{\text{steps}}}{\text{throughput}}=\frac{N_{\text{data}}/\text{batch}}{\text{steps/sec}}
$$

throughput 受显存（batch 限）、通信（多节点）、算力（GPU 数）限制。RLHF-PPO 的 throughput 主要受 rollout 生成瓶颈（[[PPO optimization]] §8.3）。

### 4.2 PT 的显存账

$$
\text{mem}_{\text{PT}}=\underbrace{\theta}_{P}+\underbrace{\text{optimizer}}_{2P\text{(Adam)}}+\underbrace{\text{activation}}_{A}+\underbrace{\text{grad}}_{P}
$$

RLHF-PPO 还加 reference $\theta_{\text{ref}}$ + RM $\theta_{\text{rm}}$（冻结，$+2P$）+ rollout worker 的 KV cache。总约 $4P$+（[[PPO optimization]] §8.2）。FSDP/ZeRO 切分减每卡显存。

### 4.3 PT-PD 的版本滞后

PT 训出 $\theta_{\text{new}}$ 后，PD 需时间加载/热切换，存在**版本滞后**：
$$
\text{lag}_{\text{PT-PD}}=T_{\text{sync}}+T_{\text{switch}}
$$

PD 期间服务的是 $\theta_{\text{old}}$（比 PT 滞后）。若 PT 持续训练，PD 定期升级追赶。详见 [[stale policy problem]] 的 PT-PD 变体。

## 5. 代码示例

```python
import torch, torch.nn as nn, copy, json

class LM(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)

V, D = 50, 32
pi = LM(V, D)   # 预训练模型

# ===== PT 侧:训练全流程(极简) =====
def pretrain(model, data, steps=10):
    opt = torch.optim.Adam(model.parameters(), 1e-4)
    for _ in range(steps):
        loss = -torch.log_softmax(model(data), -1).gather(-1, data[:, 1:]).mean()  # next-token
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def sft(model, data, steps=10):
    opt = torch.optim.Adam(model.parameters(), 1e-5)
    for _ in range(steps):
        loss = -torch.log_softmax(model(data), -1).gather(-1, data[:, 1:]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def ppo_optimize(actor, prompts, steps=5):
    # 详见 PPO optimization.md(learner + rollout worker 架构)
    opt = torch.optim.Adam(actor.parameters(), 1e-6)
    for _ in range(steps):
        loss = -actor(prompts).mean()  # 简化
        opt.zero_grad(); loss.backward(); opt.step()
    return actor

# PT 流程
pi = pretrain(pi, torch.randint(0, V, (4, 8)))
pi_sft = sft(copy.deepcopy(pi), torch.randint(0, V, (4, 8)))
pi_aligned = ppo_optimize(copy.deepcopy(pi_sft), torch.randint(0, V, (4, 4)))

# ===== PT → PD 接口:checkpoint 保存 + 版本管理 =====
version = "v1.0-rlhf-20260705"
checkpoint = {"theta": pi_aligned.state_dict(), "version": version,
              "config": {"V": V, "D": D}}
torch.save(checkpoint, "pt_checkpoint.pt")
print(f"PT 训练完成,保存 checkpoint {version}")
print("PD 将加载此 checkpoint 部署(见 policy deployment (PD).md)")
```

> [!tip] 实际工程
> PT 侧用 DeepSpeed/FSDP 训练，checkpoint 存 S3/HDFS，metadata 存数据库（版本/超参/数据）。PD 侧从存储拉 checkpoint 加载。verl/OpenRLHF 管理 PT 全流程，Triton/vLLM server 管 PD。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[learner worker split]]、[[rollout worker]]、[[asynchronous training]]（PT 内部架构）、[[SFT]]、[[Reward Model]]、[[PPO optimization]]（PT 的 RLHF 阶段）、[[3D parallelism]]（PT 的并行）、FSDP/DeepSpeed。
- **下游（应用）**: [[policy deployment (PD)]]（PT 产出给 PD）、[[weight sync mechanism]]（PT→PD 接口）、[[stale policy problem]]（PT-PD 滞后）、checkpoint 管理、RLHF 全流程。
- **对比 / 易混**:
  - **PT vs [[policy deployment (PD)]]**：训练侧 vs 部署侧。负载/框架/资源/优化点全不同。
  - **PT vs [[PPO optimization]]**：PT 是训练侧总览（含预训练/SFT/RM/PPO），PPO optimization 是 PT 在 RLHF 阶段的核心。
  - **PT vs [[learner worker split]]**：PT 是高层概念（训练侧），split 是 PT 内部的架构（采训分离）。
  - **PT 的 $\theta$ vs PD 的 $\theta$**：PT 是最新训练版，PD 是部署版（可能滞后，定期升级）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"PT 只是 PPO"**：错。PT 含预训练/SFT/RM/PPO 全流程，PPO 是 RLHF 阶段核心。
> 2. **"PT 直接服务用户"**：错。PT 只训练，服务用户是 PD 侧。PT 与 PD 分离。
> 3. **"PT 的 $\theta$ 就是部署的"**：不一定。PD 可能滞后（版本升级延迟），PT 最新 $\theta$ 需经 weight sync 推送。
> 4. **"PT 用部署框架"**：低效。PT 应用 FSDP/DeepSpeed（训练优化），非 vLLM（部署优化）。
> 5. **"PT 不需 checkpoint 管理"**：错。训练耗时长，checkpoint 是恢复+版本管理+PD 接口的关键。
> 6. **"PT 与 PD 同集群"**：常分离。PT 大集群训练，PD 服务副本，资源/网络不同。
> 7. **"PT 训完立即部署"**：需经验证（eval/人评/安全），合格才推 PD。
> 8. **"PT 的监控只看 loss"**：错。RLHF-PT 需看 reward/KL/entropy/clip/吞吐/资源综合。

## 8. 延伸细节

### 8.1 PT 的资源管理

- **GPU 集群调度**：Slurm/Kubernetes 调度训练任务，多 job 共享集群；
- **显存切分**：FSDP/ZeRO-3 切分 $\theta$/optimizer/grad 到多卡，支持大模型；
- **通信优化**：梯度 all-reduce/gather-scatter 用 NCCL，IB/NVLink 网络加速；
- **混合精度**：BF16 训练减显存/加速，A100/H100 原生支持。
详见 [[3D parallelism]]、[[通信机制]]。

### 8.2 PT 的 checkpoint 策略

- **频率**：每 N 步保存（tradeoff：恢复丢失 vs 存储开销）；
- **格式**：$\theta$ + optimizer state（恢复训练）+ scheduler state；
- **存储**：本地 SSD（快）+ 远程 S3/HDFS（持久）；
- **版本**：metadata 标版本/超参/数据 commit，可追溯。

### 8.3 PT-PD 版本协议

- PT 每次产出标记版本（semantic versioning）；
- PD 按版本拉 checkpoint，验证（eval/安全）后热切换；
- 异常回滚到上一版本；
- A/B 测试：部分流量切新版本验证。
这是 [[训推分离]] 的版本管理核心。

### 8.4 RLHF 的 PT 全流程实例

以 Llama-3-chat 为例：
1. 预训练：Llama-3 base，15T tokens；
2. SFT：指令数据微调；
3. RM：偏好对训 reward model；
4. PPO：RLHF 对齐（verl/OpenRLHF）；
5. checkpoint → PD（Llama-3-chat API）。
PT 侧耗月级，PD 侧持续服务。

### 8.5 PT 的持续训练

现代 PT 不是一次性的，而是**持续训练**：
- 用线上 PD 数据（用户反馈/偏好）回标，增量训 RM + PPO；
- 在线 RLHF（PT 持续 rollout + 训练）；
- PD 定期升级追 PT。
这是 [[asynchronous training]] 在 PT-PD 层的延伸。

---
相关: [[训推分离]]、[[policy deployment (PD)]]、[[weight sync mechanism]]、[[stale policy problem]]、[[learner worker split]]、[[rollout worker]]、[[asynchronous training]]、[[SFT]]、[[Reward Model]]、[[PPO optimization]]、[[RLHF (PPO)]]、[[3D parallelism]]、[[通信机制]]、[[activation memory]]
