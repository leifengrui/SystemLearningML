# RL权重同步

> **所属章节**: [[权重同步]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: RL weight sync / RL 权重下发 / trainer→rollout 权重刷新 / actor→rollout 参数同步 / weight update in RLHF / θ sync
> **难度**: 高（需懂 [[trainer与rollout引擎组合]]、[[colocated与disaggregated部署]]、[[weight sync mechanism]]、[[NCCL核心机制]]、[[Fully Sharded Data Parallel]]、[[Megatron-LM]]、[[importance sampling与off-policy correction]]、[[rollout train reference logprob一致性]]、[[stale policy problem]]）


## 1. 一句话定义

**RL 权重同步** 是 [[17-RL训推一体框架]] 中 trainer（actor，FSDP/Megatron 训练态）每做完一次或若干次梯度更新后，把**新策略参数 $\theta_{new}$ 下发给 rollout 端（vLLM/SGLang 推理态副本）和 reference 端（若 ref 也随训练漂移或需对齐）** 的过程——它的根因是 **RL 采样的 off-policy 性质**：rollout 用旧权重 $\theta_{old}$ 采的样本，喂给用 $\theta_{new}$ 训练的 trainer，两者越偏离 importance sampling ratio $\rho=\pi_\theta/\pi_{old}$ 越爆，故必须**频繁把 rollout 的 $\theta$ 刷到尽量新**以降 staleness。主流手段四条——① **NCCL broadcast**（trainer→rollout 全量 broadcast，RLHF 热路径绝对主流，verl/OpenRLHF `--vllm-sync-backend nccl`）② **CUDA IPC**（同机零拷贝句柄传递，OpenRLHF `update_weights_from_ipc_handles`，最快但限同机 colocate）③ **Checkpoint Engine**（SGLang 主推，写磁盘→多机并行加载，冷启动/超大模型多机首载）④ **增量更新**（只传变化的参数 delta，或 LoRA 只传 adapter，通信量从全量降到 delta）。注意它**与 [[weight sync mechanism]] 是同源不同频**：[[weight sync mechanism]] 是**训推分离 PT-PD 的版本级** sync（训练出新版本推给线上服务，低频、蓝绿/canary）；本条是 **RL 训练 step 级** sync（每个 PPO/GRPO step 或每 N step 一次，高频、热路径），两者机制同构（trainer 出权重→转换→reshard→load 进推理引擎）但频率/场景/容忍度天差地别。是 [[trainer与rollout引擎组合]] §3.4 "权重同步路径"的专章展开，是 [[colocated与disaggregated部署]] colocate vs disaggregate 选型的决定性变量，也是 [[rollout train reference logprob一致性]] 的时机根因（rollout 用的 $\theta_{old}$ 与 trainer 的 $\theta_{new}$ 不同步是 importance sampling 修正的由来）。

> [!note] 三句话定位
> - **是什么**：trainer 训完把新 $\theta$ 下发给 rollout/ref，手段四选一（NCCL broadcast / CUDA IPC / Checkpoint Engine / 增量 delta），含 all-gather 还原 + dtype 转 + reshard 重切 + load 四步。
> - **为什么**：rollout 用旧 $\theta_{old}$ 采的是 off-policy 数据，staleness 越大 ratio $\rho$ 越偏离 1、训练越不稳；频繁 sync 把 rollout 的 $\theta$ 刷新降 staleness，是 RL 稳定的命脉。
> - **与 [[weight sync mechanism]] 关系**：同源不同频——后者是 PT-PD 训推分离的**版本级** sync（低频、蓝绿/canary），本条是 RL **step 级** sync（高频、每 batch）。机制同构（四步转换 reshard），频率/场景/容忍度不同。


## 2. 为什么需要它（动机与背景）

### 2.1 RL 天生 off-policy：rollout 用旧 θ，trainer 用新 θ

RLHF/GRPO 的核心数据流是 **rollout 采 → trainer 训**，两段用**不同权重**：
- **rollout 端**：用**采样时刻的权重** $\theta_{old}$ 跑 vLLM/SGLang 生成 response $o\sim\pi_{\theta_{old}}(\cdot|q)$，记下每 token 的 $\log\pi_{\theta_{old}}(o_t|q,o_{<t})$（即 `old_log_prob`）。
- **trainer 端**：拿到 $(q,o,\log\pi_{\theta_{old}})$ 后，用**当前权重** $\theta_{new}$ 重新 forward 算 $\log\pi_{\theta_{new}}$，算 importance sampling ratio $\rho=\exp(\log\pi_{\theta_{new}}-\log\pi_{\theta_{old}})$，再算 PPO/GRPO 的 clip loss 更新 $\theta$。

这天然是 **off-policy**——样本是 $\pi_{\theta_{old}}$ 采的，梯度是对 $\pi_{\theta_{new}}$ 算的。靠 importance sampling（[[importance sampling与off-policy correction]]）的 ratio 重加权修正。**关键**：$\theta_{new}$ 与 $\theta_{old}$ 偏离越大，ratio $\rho$ 越偏离 1，修正的方差越大、越易爆（详见 §4.2）。故必须**尽量让 rollout 的 $\theta_{old}$ 接近 trainer 的 $\theta_{new}$**——这就要求**频繁把 trainer 的新 $\theta$ 下发给 rollout**。

> [!warning] 误区：rollout 用最新 θ 就 on-policy 了
> 即使每 step sync，rollout 采的那一刻是 $\theta_{old}$，trainer 训完变 $\theta_{new}$，样本对 $\theta_{new}$ 仍是 off-policy（差一步）。sync 只是把 staleness 从 $k$ 步压到 1 步，不是消除 off-policy。PPO/GRPO 仍靠 ratio clip 修这一步差。真正 on-policy（每采一条立刻训）在 LLM-RL 不可行（rollout 太贵）。详见 [[stale policy problem]]、[[importance sampling与off-policy correction]]。

### 2.2 staleness 不控制会爆 KL、崩 policy

若 rollout 不 sync 或 sync 间隔大（异步训练、disaggregate 跨机），rollout 用 $\theta_{old-k}$（落后 $k$ 步）采的样本喂 trainer，ratio $\rho_k=\exp(\log\pi_\theta-\log\pi_{old-k})$ 随 $k$ 增大急剧偏离 1。后果链：
- ratio 偏离 → PPO clip 频率 `pg_clipfrac` 飙（大量 token 被截断，梯度信号失真）；
- clip 频率高 → 有效梯度方向被 clip 扭曲，policy 朝错误方向更新；
- $\pi_\theta$ 远离 $\pi_{ref}$ → KL 约束爆（[[KL explosion]]），或 policy 漂成乱码（[[policy collapse]]）；
- 更糟：reward hacking（policy 找到 RM 漏洞刷分而非真变强）。

这是 [[stale policy problem]] 的全貌。**控制 staleness 的第一手段就是 frequent weight sync**——每 step 或每 N step 把 rollout 的 $\theta$ 刷新。colocate 因为 in-process sync 几乎零开销可每 step sync（$k\approx1$）；disaggregate 跨机 sync 重需权衡，常 $k\le S$（$S$ 为 sync 间隔），靠 PPO clip $\varepsilon$ 兜底。

### 2.3 两端权重切分布局不同：sync 不是裸传张量

trainer 用 FSDP（每 DP rank 持 $\theta$ 的一片）/ Megatron（TP 切 + PP 段 + DP 复制）持有权重，**不是完整 $\theta$**；rollout 用 vLLM/SGLang 按**推理态 TP** 切分权重。两端切法不同，sync 不能裸传 trainer 的 shard，必须：
1. **all-gather 还原全量**：FSDP 各 rank 的 shard 聚成完整 $\theta$；Megatron 的 TP 列切片 all-gather 还原完整层、跨 PP 聚层段、DP 去重。
2. **dtype 转**：训练常持 FP32 master + BF16 param；推理用 BF16/FP16。
3. **reshard 重切**：按 rollout 的 TP 并行度重新切分（训练 TP/PP/DP → 推理 TP）。
4. **load_weights 进 rollout**：vLLM/SGLang 的 `load_weights`/`update_weights` API 载入 + 重建 CUDA Graph/paged KV 元数据。

通信量 = 全模型权重（70B BF16 ≈ 140GB），是 RL 训推一体**最频繁的跨引擎通信**。详见 [[trainer与rollout引擎组合]] §3.4。

### 2.4 同步 vs 异步：频率的 trade-off

- **同步训练**（synchronous）：每 step 训完立刻 sync，rollout 用最新 $\theta$ 采下一 batch。staleness $k\approx1$，稳。但 rollout 与 training 串行（colocate 无法 overlap），GPU 利用率低。
- **异步训练**（asynchronous）：trainer 训 batch N 时 rollout 池同时采 batch N+1（disaggregate 训推 overlap），提吞吐。但 rollout 用 $\theta_{old-k}$，staleness 大，需 clip + 频繁 sync 兜底。

sync 频率（$S$=两次 sync 间隔步数）是核心 trade-off：$S$ 小（频繁 sync）staleness 小但 sync 开销重；$S$ 大 sync 开销小但 staleness 大训练不稳。colocate 因 sync 极快常 $S=1$；disaggregate 常 $S\in[1,8]$ 权衡。详见 [[asynchronous training]]、[[synchronous与asynchronous rollout]]。


## 3. 核心概念详解

### 3.1 四种主流手段总览

| 手段 | 机制 | 通信介质 | 延迟（70B） | 适用场景 | 代表实现 |
|---|---|---|---|---|---|
| **① NCCL broadcast** | trainer all-gather 还原全量→按 rollout TP reshard→NCCL broadcast 给 rollout 各 rank | NVLink（同机）/ IB（跨机） | 同机毫秒级、跨机十毫秒级 | RLHF 热路径绝对主流，colocate/disaggregate 通用 | verl `WeightSyncClient/Server`、OpenRLHF `--vllm-sync-backend nccl`、vLLM `WeightTransferEngine` |
| **② CUDA IPC** | trainer 把权重张量的 CUDA IPC handle 传给同机 rollout 进程，rollout 映射进自己地址空间零拷贝 | 同机共享显存 | 微秒级（零拷贝） | 同机 colocate（trainer 与 rollout 同节点），最快但限同机 | OpenRLHF `update_weights_from_ipc_handles`、vLLM `update_weights_from_ipc` |
| **③ Checkpoint Engine** | trainer 写权重到磁盘（常 safetensors）→ rollout 池多机并行从磁盘加载 | 共享 FS / 本地磁盘 | 秒~十秒级 | 冷启动、超大模型多机首载、非热路径 | SGLang `sglang.srt.checkpoint_engine`（broadcast/p2p/all 三模式） |
| **④ 增量更新** | 只传变化的参数 delta（或 LoRA 只传 adapter，通信量从全量降到 delta） | NCCL / IPC / FS | 随 delta 大小 | LoRA-RL、稀疏更新、参数高效微调 | verl LoRA sync、OpenRLHF adapter sync |

> [!note] 补充：四手段不互斥
> 四条路线**按场景共存非互斥**。典型组合：colocate 同机用 IPC（最快）+ 跨机 fallback NCCL broadcast；冷启动/换模型用 Checkpoint Engine；LoRA-RL 用增量只传 adapter。SGLang 还整合 ckpt-engine 加速在线 sync（issue #10464）。详见 [[weight sync mechanism]] 已联网核实的四路线总结。

### 3.2 手段①：NCCL broadcast（热路径主流）

trainer all-gather 还原全量 $\theta$ → dtype 转 + 按 rollout TP reshard → 用 NCCL `broadcast` 把各 rollout rank 应得的切片从 trainer 对应 rank 推过去。是 RLHF 每 step sync 的绝对主流。

**协议细节（vLLM `WeightTransferEngine` 四阶段，来自 [[weight sync mechanism]] 已核实）**：
1. **register**：rollout 启动时向 trainer 注册自己的权重张量元信息（name/shape/dtype/TP 切分）。
2. **send name list**：trainer 决定 sync 时，把要更新的张量名列表发给 rollout（控制流，小对象）。
3. **broadcast tensors**：trainer 各 rank 用 NCCL broadcast 把对应 rollout rank 应得的权重切片推过去（数据流，大对象）。
4. **ack**：rollout 收完回 ack，trainer 才继续。

**packed tensor 优化**：weight sync 传大量小张量（每层 weight/bias）时，per-tensor NCCL launch 开销累积。vLLM `WeightTransferEngine` 用 **packed tensor**——把多个小张量打包成大连续 buffer，一次 broadcast，再 unpack。双/三缓冲 + 独立 CUDA stream 做 pack/broadcast/unpack overlap，吞吐显著提升。详见 [[trainer与rollout引擎组合]] §8.2。

**verl / OpenRLHF 配置**：
```bash
# OpenRLHF: NCCL broadcast weight sync
python -m openrlhf.cli.train.ppo_ray \
  --vllm_sync_backend nccl \   # ★ NCCL broadcast 后端
  --colocate                   # colocate 同机 NVLink；去此开 disaggregate 走跨机 NCCL
```
verl 的 `WeightSyncClient/Server` 在 colocate 走 in-process、disaggregate 走跨机 NCCL，同一 API 切后端。详见 [[Ray与分布式调度]] §3.5。

### 3.3 手段②：CUDA IPC（同机零拷贝最快）

同机 colocate 下，trainer 与 rollout 在同一节点的不同进程（或同进程），可共享 GPU 显存。CUDA IPC 机制允许一个进程把**自己 GPU 张量的 IPC handle**（一个句柄）传给另一进程，另一进程用 `cudaIpcOpenMemHandle` 映射进自己地址空间，**零拷贝**直接访问同一块显存。

**OpenRLHF `update_weights_from_ipc_handles`**：trainer 把每层权重的 IPC handle 序列化传给 vLLM，vLLM `open` handle 映射，无需 broadcast 数据——通信只传 handle（小对象），数据零拷贝。延迟微秒级，是同机 colocate 的极致。

**限制**：
- **限同机**：CUDA IPC handle 只在同一节点的 GPU 间有效，跨机不可用（跨机必须 fallback NCCL broadcast）。
- **进程生命周期**：handle 依赖发送进程存活，trainer 进程不能释放张量。
- **vLLM 内部布局对齐**：IPC 传的是裸张量，需 trainer 的张量布局与 vLLM 期望布局对齐（或 reshard 后再 IPC）。

详见 [[weight sync mechanism]] 路线二已核实内容。

### 3.4 手段③：Checkpoint Engine（冷启动/大模型）

SGLang 的 `sglang.srt.checkpoint_engine` 把权重写成 checkpoint 文件（safetensors），rollout 池多机并行从共享 FS 加载。支持 broadcast / p2p / all 三种加载模式：
- **broadcast**：一台从 FS 读，broadcast 给其他（省 FS 带宽，但单点读瓶颈）。
- **p2p**：各机各自从 FS 读（并行，但 FS 带宽瓶颈）。
- **all**：所有机都读全量（最简但浪费，仅小模型）。

**适用场景**：
- **冷启动**：训练开始 rollout 首次加载 base 权重，ckpt-engine 从 FS 多机并行加载。
- **超大模型多机首载**：70B+ 跨机 NCCL broadcast 仍重，ckpt-engine 多机并行读 FS 更快。
- **非 colocated rollout 多副本**：rollout 池有多副本，ckpt-engine 一次写多副本并行读。

**不适合 RLHF 热路径**：秒级延迟远超每 step sync 的容忍度（colocate 毫秒级）。故 SGLang 在整合 ckpt-engine 加速在线 sync（issue #10464），但 RLHF step 级仍以 NCCL broadcast 为主。

SGLang 另有 **`/pull_weights`**（共享 FS zstd per-tensor delta + checksum，用于非 colocated rollout/PD 多副本增量同步），详见 [[weight sync mechanism]] 路线四。

### 3.5 手段④：增量更新（delta / LoRA）

全量 sync 通信量 = 全模型权重（140GB/70B），即便 NVLink 也重。若只部分参数变化，可只传 delta：
- **LoRA-RL**：只训 adapter（A/B 矩阵，参数量 ≪ base），sync 只传 adapter（几十 MB~几 GB），通信量降两个数量级。verl/OpenRLHF 的 LoRA recipe 支持。
- **稀疏更新**：若只部分层/部分参数训（如只训最后几层），sync 只传变化部分。
- **delta diff**：trainer 算 $\Delta\theta=\theta_{new}-\theta_{old}$，只传非零部分（但 LLM 训练通常全参数都变，delta 稀疏性有限，不如 LoRA 显著）。

增量更新大幅降通信，但需 rollout 端支持"接收 delta 并叠加"或"接收 adapter 并与 base 合并"的 load 逻辑。LoRA-RL 是当前最实用的增量场景。

### 3.6 reshard 问题：trainer 切法 ≠ rollout 切法

这是 weight sync 最复杂的工程点。trainer 与 rollout 的**并行切分布局不同**：

```
trainer (训练态)                         rollout (推理态)
┌──────────────────────┐                ┌──────────────────────┐
│ FSDP: 每 DP rank 持一片│                │ vLLM: 按 TP 切        │
│  (不是完整 θ)          │ → all-gather → │  (每 rank 持层列切片)  │
│ Megatron: TP 切+PP 段 │   还原全量 θ   │                       │
│  + DP 复制            │ → 按 TP reshard│                       │
└──────────────────────┘                └──────────────────────┘
```

- **FSDP → rollout**：FSDP 每 DP rank 持 $\theta$ 的 $1/N_{dp}$ 片。rollout 需完整 $\theta$ 按其 TP 切，故必须先 all-gather 还原全量，再按 rollout TP 切。即使 trainer DP 与 rollout TP 度数相同，切法不同（DP 是参数维切、TP 是层内列切），仍需 all-gather + 重切。
- **Megatron → rollout**（最复杂）：训练态 TP/PP/DP 三维切分，要 ① TP all-gather 还原完整层 ② 跨 PP 聚层段成完整层序列 ③ DP 去重（各 DP rank 同权重取一份）④ 按 rollout TP 重切。HybridFlow 的 **3D-HybridEngine** 把 actor 训推态放同组 GPU，靠 NVLink all-gather + 重切在组内完成 reshard，避免跨机 weight sync（详见 [[colocated与disaggregated部署]] §3.5、[[trainer与rollout引擎组合]] §3.5）。

reshard 通信量 $\approx O(P\cdot\text{size})$（与全量 sync 同量级，因要还原全量再重切）。

### 3.7 同步频率：step 级 vs 版本级

| 维度 | RL step 级 sync（本条） | PT-PD 版本级 sync（[[weight sync mechanism]]） |
|---|---|---|
| 触发 | 每 PPO/GRPO step 或每 N step | 训出新版本（小时~天级） |
| 频率 | 高（colocate 每 step，disaggregate 每 N step） | 低（版本发布频率） |
| 容忍度 | 毫秒~十毫秒级（热路径，不能慢） | 秒~分钟级（可慢，非实时） |
| stale 容忍 | 低（ratio clip 兜底，但 $k$ 大仍不稳） | 高（线上服务有版本管理） |
| 手段 | NCCL broadcast / IPC 为主 | Checkpoint Engine / 共享 FS delta 为主 |
| 版本管理 | 无（rollout 始终追 trainer 最新） | 有（蓝绿/canary/灰度） |

两者机制同构（trainer 出权重→转换→reshard→load），但频率/场景/容忍度天差地别。RL step 级 sync 是热路径，对延迟极敏感，故 colocate IPC/同机 NCCL 是首选。

### 3.8 与 logprob 一致性的时机根因

weight sync 的**时机**直接决定 logprob 一致性的格局：
- **sync 前**：rollout 用 $\theta_{old}$ 采，记 `old_log_prob`=$\log\pi_{\theta_{old}}$。trainer 此时权重已是 $\theta_{new}$（训完上一 step），重算得 `new_log_prob`=$\log\pi_{\theta_{new}}$。ratio $\rho=\exp(\log\pi_{\theta_{new}}-\log\pi_{\theta_{old}})$——这正是 importance sampling 修正，**不同步本身就是 IS 的由来**，不是 bug。
- **sync 后**：rollout 的 $\theta$ 刷成 $\theta_{new}$，下一 batch 采的 `old_log_prob`=$\log\pi_{\theta_{new}}$（与新 trainer 重算的 $\log\pi_{\theta_{new+}}$ 只差一步），ratio 接近 1，staleness 小。

**真正的 logp 一致性问题**不在 $\theta$ 不同步（那是 IS 本意），而在**同 $\theta$ 下 rollout 与 trainer 算的 logp 数值不一致**（TIM 训推不一致，见 [[rollout train reference logprob一致性]]、[[训推不一致]]）。即 $\theta_{rollout}=\theta_{trainer}$ 时，$\log\pi_{rollout}\ne\log\pi_{trainer}$（数值差），这才是要治理的。weight sync 保证 $\theta$ 对齐，TIM 治理保证同 $\theta$ 下 logp 对齐——两者正交但常被混淆。

> [!warning] 误区：weight sync 做好了 logp 就一致
> 不对。weight sync 保证 rollout 的 $\theta$ 与 trainer 对齐（$\theta$ 相同），但同 $\theta$ 下 vLLM（BF16+优化 kernel）与 trainer（FP32 master+BF16 forward）算的 logp 因精度/算子/MoE 路由不一致仍有数值差（TIM）。weight sync 是 $\theta$ 对齐，TIM 治理是同 $\theta$ 下 logp 对齐，两层正交。详见 [[rollout train reference logprob一致性]]。


## 4. 数学原理 / 公式

### 4.1 通信量公式

一次全量 weight sync 的通信体积：

$$
V = P \cdot \text{size}
$$

- $P$：模型参数量（70B ≈ $70\times10^9$）。
- size：每参数字节数（BF16/FP16=2，FP32=4，FP8=1）。

例：70B BF16 → $V=70\times10^9\times2=140\text{GB}$。7B BF16 → $14\text{GB}$。

同步延迟（NCCL broadcast）：

$$
T_{\text{sync}} = \frac{V}{\text{BW}_{\text{eff}}} + T_{\text{overhead}}
$$

- $\text{BW}_{\text{eff}}$：有效带宽。同机 NVLink 数百 GB/s，跨机 IB 数十~数百 GB/s，同机 IPC 共享显存 ~TB/s。
- $T_{\text{overhead}}$：协议开销（register/name list/ack）+ launch 开销（packed tensor 优化此）。

| 模式 | $\text{BW}_{\text{eff}}$ | $T_{\text{sync}}$（70B BF16） |
|---|---|---|
| colocate IPC（零拷贝） | 显存带宽 ~TB/s | 微秒级 |
| colocate 同机 NCCL | NVLink ~数百 GB/s | 毫秒级 |
| disaggregate 跨机 NCCL | IB ~数十~数百 GB/s | 十毫秒级 |
| Checkpoint Engine | FS/磁盘 | 秒~十秒级 |
| LoRA 增量（只 adapter） | 同上介质 | 降 2 个数量级（adapter 几十 MB~几 GB） |

### 4.2 推导：staleness 如何让 ratio 爆（sync 频率的数学根）

设 rollout 用 $\theta_{old-k}$（落后 trainer $k$ 步）采，trainer 当前 $\theta_{new}$。token 级 ratio：

$$
\rho_t = \frac{\pi_{\theta_{new}}(o_t|q,o_{<t})}{\pi_{\theta_{old-k}}(o_t|q,o_{<t})} = \exp\!\big(\log\pi_{\theta_{new}}(o_t) - \log\pi_{\theta_{old-k}}(o_t)\big)
$$

假设单步更新 $\theta_{new}-\theta_{old}$ 是小量 $\Delta\theta$，对 $\log\pi$ 做一阶 Taylor 展开：

$$
\log\pi_{\theta_{new}} \approx \log\pi_{\theta_{old}} + \nabla_\theta\log\pi_{\theta_{old}}\cdot\Delta\theta
$$

落后 $k$ 步则 $\theta_{old-k}$ 与 $\theta_{new}$ 差 $\approx k\Delta\theta$：

$$
\log\pi_{\theta_{new}} - \log\pi_{\theta_{old-k}} \approx k\cdot\nabla_\theta\log\pi_{\theta_{old}}\cdot\Delta\theta
$$

故：

$$
\rho_t \approx \exp\!\big(k\cdot g_t\big),\quad g_t=\nabla_\theta\log\pi_{\theta_{old}}\cdot\Delta\theta
$$

**关键结论**：ratio 的对数偏离 $\propto k$（staleness 步数）。$k$ 越大，$\rho$ 偏离 1 越剧烈（指数放大）：
- $k=1$（每 step sync）：$\rho\approx\exp(g_t)$，偏离小，PPO clip $\varepsilon=0.2$ 基本不触发。
- $k=8$（每 8 step sync）：$\rho\approx\exp(8g_t)$，偏离大，大量 token 超 clip 范围被截断，梯度失真。

这就是 **sync 频率 $S$（=两次 sync 间隔步数）必须控制小**的数学根——$S$ 即 $k_{\max}$，staleness 上界。colocate $S=1$ 最稳；disaggregate 权衡 $S$ 与 sync 开销。PPO clip $\varepsilon$ 是 staleness 的兜底：当 $\rho$ 超出 $[1-\varepsilon,1+\varepsilon]$ 截断，防梯度爆，但截断过多会失真。故 **sync 频率 + clip $\varepsilon$ 共同控制 off-policy 程度**。详见 [[importance sampling与off-policy correction]] §4、[[stale policy problem]]。

### 4.3 推导：reshard 通信量为何与全量 sync 同量级

FSDP 每 DP rank 持 $\theta$ 的 $1/N_{dp}$ 片。要给 rollout 完整 $\theta$（按其 TP 切），必须先 all-gather 还原全量：

$$
\text{Vol}_{\text{gather}} = \frac{N_{dp}-1}{N_{dp}}\cdot P\cdot\text{size} \approx P\cdot\text{size}\quad(N_{dp}\gg1)
$$

即 all-gather 通信量 $\approx$ 全模型（ring all-gather 每卡 $(N-1)/N\cdot V$）。还原全量后再按 rollout TP 切，**切分本身不产生跨机通信**（本地 slice），但把切片下发到 rollout 各 rank 又是一次 broadcast（$\approx V$）。

故 reshard 总通信量 $\approx O(P\cdot\text{size})$，与全量 sync 同量级——因为**必须还原全量才能重切**，无法绕过（除非 trainer 与 rollout 切分布局完全一致可直传 shard，但罕见）。3D-HybridEngine 把这步放同组 NVLink（高带宽）避免跨机，是优化关键。详见 [[trainer与rollout引擎组合]] §4.2。

### 4.4 sync 总延迟分解

$$
T_{\text{sync}} = T_{\text{gather}} + T_{\text{convert}} + T_{\text{reshard}} + T_{\text{transfer}} + T_{\text{load}}
$$

- $T_{\text{gather}}$：FSDP all-gather / Megatron TP+PP 还原（同组 NVLink 快、跨机慢）。
- $T_{\text{convert}}$：dtype 转 + tensor 重组（CPU 算，常与 gather overlap）。
- $T_{\text{reshard}}$：按 rollout TP 重切（本地 slice，快）。
- $T_{\text{transfer}}$：传给 rollout 进程（colocate in-process / 同机 NCCL / 跨机 NCCL broadcast / IPC 零拷贝）。
- $T_{\text{load}}$：vLLM/SGLang `load_weights`（写显存 + 重建 CUDA Graph/paged KV 元数据）。

| 模式 | $T_{\text{sync}}$（70B） |
|---|---|
| FSDP colocate IPC | 微秒~毫秒级（IPC 零拷贝 + 同机 NVLink gather） |
| FSDP colocate 同机 NCCL | 毫秒级 |
| FSDP disaggregate 跨机 NCCL | 十毫秒级 |
| Megatron colocate 3D-HybridEngine | 十毫秒级（同组 reshard + NVLink） |
| Megatron disaggregate 跨机 | 百毫秒~秒级（跨机 reshard 重） |

### 4.5 与 IS 修正的关系：sync 频率决定 ratio 的"正确偏离"

注意一个微妙点：weight sync 的目的是**让 rollout 的 $\theta$ 尽量新**，但即便每 step sync，rollout 采那一刻是 $\theta_{old}$、trainer 训完变 $\theta_{new}$，ratio $\rho=\exp(\log\pi_{\theta_{new}}-\log\pi_{\theta_{old}})\ne1$——**这个偏离是 IS 修正的本意，不是要消除的**。要消除的是**过度偏离**（$k>1$ 的 staleness 放大）。故 sync 频率不是"让 ratio=1"，而是"让 ratio 偏离控制在 clip $\varepsilon$ 范围内"。colocate $k=1$ 时偏离由单步 $\Delta\theta$ 决定（小，clip 基本不触发）；disaggregate $k$ 大时偏离被指数放大，clip 频繁触发，需更频繁 sync 降 $k$。详见 [[importance sampling与off-policy correction]] §4.4。


## 5. 代码示例（可选）

### 5.1 手算 staleness 对 ratio 的影响（最小可跑）

```python
import numpy as np

def ratio_vs_staleness(g_t, k):
    """
    g_t: 单步更新对 logpi 的一阶影响 (标量, 近似 grad*logpi * delta_theta)
    k  : staleness 步数 (rollout 落后 trainer k 步)
    返回 ratio rho = exp(k * g_t)
    """
    return np.exp(k * g_t)

g = 0.05  # 假设单步更新让 logpi 变化 0.05 (中等步)
for k in [1, 2, 4, 8]:
    rho = ratio_vs_staleness(g, k)
    clip_frac = np.mean(np.abs(rho - 1) > 0.2)  # PPO clip eps=0.2 触发率(示意)
    print(f"k={k}: rho={rho:.3f}, 偏离1={abs(rho-1):.3f}, clip触发={'是' if abs(rho-1)>0.2 else '否'}")
# k=1: rho=1.051, 偏离1=0.051, clip触发=否   (每 step sync, 稳)
# k=2: rho=1.105, 偏离1=0.105, clip触发=否
# k=4: rho=1.221, 偏离1=0.221, clip触发=是   (4 step sync, clip 开始触发)
# k=8: rho=1.492, 偏离1=0.492, clip触发=是   (8 step sync, 大量截断, 梯度失真)
# -> staleness 指数放大 ratio 偏离, 故 sync 频率必须控制小
```

### 5.2 verl FSDP trainer → vLLM rollout weight sync（概念）

```python
import torch
import torch.distributed as dist

def rl_weight_sync_fsdp_to_vllm(actor_fsdp, vllm_engine, rollout_tp_size, sync_backend="nccl"):
    """
    RL step 级 weight sync: trainer (FSDP) 新 θ -> rollout (vLLM)
    sync_backend: "nccl" (broadcast) 或 "ipc" (同机零拷贝)
    """
    # 1. FSDP all-gather: 各 rank 的 shard 聚成完整 θ
    full_state = {}
    # FSDP all_gather 把各 rank shard 聚成完整参数 (FSDP API 或 manual)
    actor_fsdp.state_dict(all_gather=True, out=full_state)

    # 2. dtype 转 + 按 vLLM TP 重切 (reshard)
    for name, param in full_state.items():
        param = param.to(torch.bfloat16)  # 训练 FP32 -> 推理 BF16
        # 训练 FSDP 是 DP 切, 推理是 TP 切, 切法不同需重切
        shards = split_for_tp(param, rollout_tp_size)
        for rank, shard in enumerate(shards):
            if sync_backend == "nccl":
                # NCCL broadcast: trainer rank 推给 rollout rank
                dist.broadcast(shard, src=trainer_rank_for(rank), group=rollout_group)
                vllm_engine.workers[rank].load_weight(name, shard)
            elif sync_backend == "ipc":
                # CUDA IPC: 同机零拷贝, 传 handle 不传数据
                handle = shard.to("cuda").ipc_handle()  # 拿 IPC handle
                vllm_engine.workers[rank].update_weights_from_ipc_handles(
                    {name: handle})  # vLLM 映射进地址空间

    # 3. vLLM 重建 CUDA Graph / paged KV 元数据
    vllm_engine.after_weight_update()
```

### 5.3 OpenRLHF 配置：四种 sync 后端切换

```bash
# OpenRLHF: DeepSpeed ZeRO-3 trainer + vLLM rollout, 四种 sync 后端

# ① NCCL broadcast (主流, colocate/disaggregate 通用)
python -m openrlhf.cli.train.ppo_ray \
  --zero_stage 3 --bf16 \
  --vllm_num_engines 4 \
  --vllm_sync_backend nccl \    # ★ NCCL broadcast
  --colocate                    # colocate 同机; 去此开 disaggregate 走跨机 NCCL

# ② CUDA IPC (同机 colocate 最快, 零拷贝)
python -m openrlhf.cli.train.ppo_ray \
  --zero_stage 3 --bf16 \
  --vllm_num_engines 4 \
  --vllm_sync_backend ipc \     # ★ CUDA IPC handle
  --colocate                    # 必须同机 (IPC 限同节点)

# ③ Checkpoint Engine (SGLang, 冷启动/大模型多机首载)
# SGLang 用 sglang.srt.checkpoint_engine, 从 FS 多机并行加载
# RLHF 热路径仍用 NCCL, ckpt-engine 用于冷启动 (配置以 SGLang 官方文档为准)

# ④ 增量更新 (LoRA-RL, 只传 adapter)
python -m openrlhf.cli.train.ppo_ray \
  --zero_stage 3 --bf16 \
  --lora_dim 64 \               # LoRA, 只训 adapter
  --vllm_num_engines 4 \
  --vllm_sync_backend nccl \   # 仍 NCCL, 但只传 adapter (几十 MB~几 GB, 降两数量级)
  --colocate
```

### 5.4 verl colocate sleep/wake + weight sync 主循环（来自 [[Ray与分布式调度]] §3.5）

```python
# verl colocate: rollout 副本与 actor 同卡, sleep/wake 分时 + weight sync
for batch in train_dataloader:
    # rollout 阶段: vLLM 副本在线采 (用当前 θ_old), actor 闲
    gen = self.async_rollout_manager.generate_sequences(batch)
    self.checkpoint_manager.sleep_replicas()   # 释放 rollout 副本显存给训练

    # 训练阶段: actor forward/backward 更新 θ
    batch = batch.union(gen)
    ...reward, old_log_prob, ref_log_prob, advantage...
    self._update_actor(batch)                  # backward, θ: old -> new

    # ★ RL weight sync: 新 θ -> rollout 副本 (colocate in-process 极快)
    self.checkpoint_manager.update_weights(self.global_steps)
    # update_weights = wake 副本 + 把 actor 新 θ 原地刷给 vLLM (IPC/in-process)
    # 下一 batch rollout 用刷新后的 θ 采, staleness k≈1
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[trainer与rollout引擎组合]]（两端引擎组合，本条是其 weight sync 桥接的专章）、[[colocated与disaggregated部署]]（部署模式决定 sync 后端与延迟）、[[weight sync mechanism]]（同源机制，本条是其 step 级高频特化）、[[NCCL核心机制]]（broadcast 的底层）、[[Fully Sharded Data Parallel]]/[[Megatron-LM]]（trainer 切分布局，决定 reshard 复杂度）、[[Ray与分布式调度]]（verl 用 Ray 编排 sync）。
- **下游（应用）**: [[17-RL训推一体框架]] 全章、[[importance sampling与off-policy correction]]（sync 频率决定 ratio 偏离程度）、[[rollout train reference logprob一致性]]（sync 时机是 IS 的根因，但 TIM 是同 θ 下 logp 不一致，两者正交）、[[stale policy problem]]（sync 是控制 staleness 的第一手段）、[[asynchronous training]]/[[synchronous与asynchronous rollout]]（sync 频率是同步异步 trade-off 的核心）、[[RL角色拓扑]]（actor→rollout 的权重流）。
- **对比 / 易混**:
  - **RL step 级 sync（本条）vs [[weight sync mechanism]] 版本级 sync**：前者是 RL 训练内每 step/N step 一次（热路径，毫秒级容忍，NCCL/IPC 为主，无版本管理）；后者是 PT-PD 训推分离版本发布（低频，秒~分钟级容忍，Checkpoint Engine/FS 为主，有蓝绿/canary）。机制同构，频率/场景/容忍度不同。
  - **RL weight sync vs 梯度 AllReduce**：前者是 trainer→rollout 推**权重**（推理端刷新采样权重）；后者是 trainer 内各 rank 聚**梯度**（训练端数据并行同步）。方向相反、对象不同。
  - **RL weight sync vs [[训推分离]] PT→PD push**：前者是 RLHF 训练内部 trainer→rollout（采数据用）；后者是训练↔部署分离（线上服务用）。前者高频每 step，后者低频版本级。
  - **weight sync（θ 对齐）vs [[训推不一致]] TIM 治理（同 θ 下 logp 对齐）**：前者保证 $\theta_{rollout}=\theta_{trainer}$；后者保证同 $\theta$ 下 $\log\pi_{rollout}=\log\pi_{trainer}$（数值一致）。两层正交，常被混淆。详见 [[rollout train reference logprob一致性]]。


## 7. 常见误区与易错点

> [!warning] 误区 1：weight sync 只是传张量
> 错。两端权重切分布局不同（FSDP shard / Megatron TP/PP/DP / vLLM TP），sync 含 **all-gather 还原 + dtype 转 + reshard 重切 + load**，通信量 = 全模型权重。Megatron reshard 尤复杂。详见 [[trainer与rollout引擎组合]] §3.4。

> [!warning] 误区 2：rollout 用最新 θ 就 on-policy 了
> 即使每 step sync，rollout 采那一刻是 $\theta_{old}$，trainer 训完变 $\theta_{new}$，样本对 $\theta_{new}$ 仍 off-policy（差一步）。sync 只把 staleness 从 $k$ 压到 1，不消除 off-policy。PPO/GRPO 仍靠 ratio clip 修这一步差。

> [!warning] 误区 3：sync 越频繁越好
> 不一定。colocate sync 极快（IPC 微秒级）可每 step sync。但 disaggregate 跨机 sync 重（十毫秒级），过于频繁 sync 的开销累积拖死吞吐。需权衡 sync 频率 $S$ 与 staleness——$S$ 小稳但慢，$S$ 大快但 stale。colocate $S=1$；disaggregate $S\in[1,8]$。

> [!warning] 误区 4：weight sync 做好了 logp 就一致
> 错。weight sync 保证 $\theta$ 对齐（rollout 与 trainer 同权重），但同 $\theta$ 下 vLLM 与 trainer 算的 logp 因精度/算子/MoE 路由不一致仍有数值差（TIM）。weight sync 是 $\theta$ 对齐，TIM 治理是同 $\theta$ 下 logp 对齐，两层正交。详见 [[rollout train reference logprob一致性]]、[[训推不一致]]。

> [!warning] 误区 5：CUDA IPC 跨机也能用
> 错。CUDA IPC handle 只在同一节点的 GPU 间有效，跨机不可用。跨机必须 fallback NCCL broadcast。IPC 是 colocate 同机的极致，非跨机方案。

> [!warning] 误区 6：增量更新总是更好
> 不一定。全参数训练时 $\Delta\theta$ 通常全参数非零，delta diff 稀疏性有限，压缩收益不大且需额外 diff 计算。增量更新真正显著受益是 LoRA-RL（只训 adapter，通信量降两个数量级）或稀疏更新场景。全参数 RLHF 仍以全量 NCCL broadcast 为主。

> [!warning] 误区 7：RL weight sync 与训推分离 PT-PD sync 是一回事
> 机制同构（trainer 出权重→转换→reshard→load），但频率/场景/容忍度不同。前者是 RL step 级（热路径，毫秒级，无版本管理）；后者是版本级（低频，秒~分钟级，有蓝绿/canary）。详见 §3.7、[[weight sync mechanism]]。

> [!tip] 实践：选 sync 后端看部署模式
> - colocate 同机：CUDA IPC（最快，零拷贝微秒级）或 in-process。
> - colocate/disaggregate 同机跨进程：同机 NCCL（NVLink 毫秒级）。
> - disaggregate 跨机：NCCL broadcast（IB 十毫秒级，热路径主流）。
> - 冷启动/超大模型多机首载：SGLang Checkpoint Engine（多机并行 FS 加载）。
> - LoRA-RL：增量只传 adapter（NCCL/IPC 均可，通信量降两数量级）。


## 8. 延伸细节

### 8.1 vLLM WeightTransferEngine 的协议细节

vLLM 的 `WeightTransferEngine` 是 RLHF weight sync 的代表实现，四阶段协议（来自 [[weight sync mechanism]] 已核实）：
1. **register**：rollout 启动时向 trainer 注册权重张量元信息（name/shape/dtype/TP 切分），trainer 建立张量注册表。
2. **send name list**：trainer 决定 sync 时，把要更新的张量名列表发给 rollout（控制流，小对象，走 Ray/HTTP）。
3. **broadcast tensors**：trainer 各 rank 用 NCCL broadcast 把对应 rollout rank 应得的权重切片推过去（数据流，大对象，走 NCCL）。
4. **ack**：rollout 收完回 ack，trainer 才继续。

**packed tensor 优化**：把多个小张量打包成大连续 buffer，一次 broadcast 减 launch 开销，双/三缓冲 + 独立 CUDA stream 做 pack/broadcast/unpack overlap。是 weight sync 吞吐优化的关键。详见 [[trainer与rollout引擎组合]] §8.2、[[weight sync mechanism]] 路线一。

### 8.2 SGLang 的 weight sync 后端

SGLang 的 weight sync 与 vLLM 思路同但实现异：
- **Checkpoint Engine**（`sglang.srt.checkpoint_engine`）：写磁盘→多机并行加载，broadcast/p2p/all 三模式，冷启动/大模型多机首载加速。
- **`/pull_weights`**：共享 FS zstd per-tensor delta + checksum，用于非 colocated rollout/PD 多副本增量同步。
- **在线 sync 整合**：SGLang 在整合 ckpt-engine 加速在线 RLHF sync（issue #10464），但 RLHF step 级热路径仍以 NCCL broadcast 为主。

详见 [[weight sync mechanism]] 路线三/四已核实内容。

### 8.3 reference 端的 weight sync

reference policy（ref）通常是**冻结的 base/已对齐模型**，不随训练更新，故不需 weight sync。但有例外：
- **多阶段 RL**：每阶段（如 SFT→RL1→RL2）ref 可能换（上一阶段的 actor 当下一阶段的 ref），换阶段时需把新 ref 权重 load 进 ref 引擎（一次性，非每 step）。
- **ref 与 actor colocate**：ref 副本与 actor 同卡，ref 不更新但需在阶段切换时刷新。verl 的 `actor_rollout_ref` 把 ref 塞同进程，ref 只 forward 算 logp，权重只在阶段切换时 load 一次。

故 reference 端的 weight sync 是**低频/阶段级**，与 rollout 的 step 级不同。详见 [[RL角色拓扑]]。

### 8.4 weight sync 与 prefix cache 的交互

weight sync 推新 $\theta$ 后，rollout 的 vLLM prefix cache 里存的旧 KV 是用旧 $\theta$ 算的，与新 $\theta$ 不一致。若不清 cache，下次采命中前缀会用旧 KV 拼 attention，引入 logp 偏差（TIM 的 KV 路径）。故 weight sync 后通常**清空 prefix cache**（或失效相关前缀）。详见 [[Radix Tree prefix cache]] §3.4（已批注解答）。verl/SGLang 的 weight sync 接口常内置 cache invalidation。

### 8.5 异步训练的 partial rollout 与 sync 时机

disaggregate 异步训练下，trainer 训 batch N 时 rollout 采 batch N+1，两阶段 overlap。但 sync 时机复杂：
- 若 sync 时 rollout 还在采（用旧 $\theta$ 采到一半），新 $\theta$ 刷入会打断正在采的 batch（$\theta$ 中途变，logp 不一致）。
- 故异步常用 **partial rollout**：rollout 采完一个完整 batch 才接受 sync，未采完的 batch 标记 stale 喂 trainer 或丢弃。
- 或 **版本号机制**：rollout 采时记 $\theta$ 版本号，sync 后新 batch 用新版本号，trainer 按版本号算对应 ratio。

详见 [[asynchronous training]]、[[synchronous与asynchronous rollout]]。

### 8.6 内容来源

RL weight sync 四手段综合自 [[weight sync mechanism]] 已联网核实的四路线总结（vLLM `WeightTransferEngine` 四阶段协议+packed tensor、OpenRLHF `--vllm_sync_backend nccl/ipc`+`update_weights_from_ipc_handles`、SGLang `sglang.srt.checkpoint_engine`+`/pull_weights`）。verl colocate sleep/wake+update_weights 时序来自 [[Ray与分布式调度]] §3.5 已核实代码解答。staleness→ratio 指数放大推导基于 importance sampling 一阶 Taylor 展开（与 [[importance sampling与off-policy correction]] §4 一致）。reshard 通信量与 3D-HybridEngine 来自 [[trainer与rollout引擎组合]] §3.5/§4.2 与 HybridFlow 论文 arXiv:2409.19256。截至 2026-07。

---
相关: [[权重同步]] | [[trainer与rollout引擎组合]] | [[colocated与disaggregated部署]] | [[weight sync mechanism]] | [[importance sampling与off-policy correction]] | [[rollout train reference logprob一致性]] | [[训推不一致]] | [[stale policy problem]] | [[asynchronous training]] | [[synchronous与asynchronous rollout]] | [[NCCL核心机制]] | [[Fully Sharded Data Parallel]] | [[Megatron-LM]] | [[Ray与分布式调度]] | [[RL角色拓扑]] | [[训推分离]] | [[rollout worker]] | [[trainer与rollout引擎组合]] | [[KL explosion]] | [[policy collapse]] | [[17-RL训推一体框架]]
