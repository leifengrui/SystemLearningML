# colocated与disaggregated部署

> **所属章节**: [[角色拓扑]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: colocate vs disaggregate / 同机共置 vs 跨机分离 / RLHF 部署模式 / RL 角色部署拓扑 / hybrid 部署 / 3D-HybridEngine
> **难度**: 高（需懂 [[RL角色拓扑]]、[[trainer与rollout引擎组合]]、[[weight sync mechanism]]、[[训推分离]]、[[stale policy problem]]、[[asynchronous training]]、[[Ray与分布式调度]]）


## 1. 一句话定义

**colocated 与 disaggregated 部署** 是 [[17-RL训推一体框架]] 五角色（actor/rollout/reference/critic/reward）在**物理布局维度的两种部署模式**——**colocated**（同机共置）把多个角色放**同一组 GPU、同进程或同节点**，靠显存分时复用 + 权重 in-process 共享省卡、把 weight sync 降到微秒~毫秒级同机 NCCL/CUDA IPC（[[weight sync mechanism]] 路线一/二），代价是资源争用、显存挤、role 切换时 GPU 利用率剧烈波动（rollout 时训练 SM 闲置、训练时 rollout 副本 sleep）；**disaggregated**（跨机分离）把各角色放**独立 GPU 池跨节点**，资源独立扩缩、可按角色算力特征配比（rollout 池大训推分离式扩、trainer 池按梯度算力配），代价是 weight sync 走跨机 NCCL/共享 FS/checkpoint engine（路线三/四）开销重 + [[stale policy problem]] 风险；**hybrid**（混合）是主流——trainer+ref colocate 同卡组（ref 只算 forward 可分时）、rollout 独立池按推理算力配、critic/RM 按需——兼顾省卡与隔离。verl 的 `actor_rollout_ref` colocate + 独立 critic/reward 池、OpenRLHF 的 colocate/hybrid 模式、HybridFlow 的 3D-HybridEngine（actor 训推态同组 reshard）都是 hybrid 的具体形态。是 [[RL角色拓扑]] §3.4 拓扑图在"同机 vs 跨机"轴的专篇展开，与 [[trainer与rollout引擎组合]]（引擎轴）正交。

> [!note] 三句话定位
> - **是什么**：colocate = 多角色同机同卡（省显存/sync 极快/但争用）；disaggregate = 各角色独立跨机池（隔离稳/可配比/但 sync 重 + stale）；hybrid = 混合（主流）。
> - **为什么**：RLHF 五角色若全 disaggregate，每 batch 一次 weight sync（~140GB）走跨机带宽扛不住、stale 重；若全 colocate，大模型显存挤爆 + role 切换利用率骤降。hybrid 取两者甜区。
> - **与 [[RL角色拓扑]] / [[trainer与rollout引擎组合]] 关系**：本篇是"角色放哪"（部署轴）；[[RL角色拓扑]] 是"有哪些角色、数据怎么流"（角色轴）；[[trainer与rollout引擎组合]] 是"角色用什么框架跑"（引擎轴）。三轴正交组合出具体部署。


## 2. 为什么需要它（动机与背景）

### 2.1 全 colocate 的痛：显存挤 + 利用率波动

colocate 把 actor/rollout/ref/critic/rm 全塞同组卡。好处明显——weight sync 走 in-process 共享显存（rollout 副本与 actor 同权重），**近乎零开销**。但两痛：

1. **显存挤**：大模型 RLHF 时，训练态权重 + 优化器状态（Adam = 2×权重 FP32 m/v，[[Fully Sharded Data Parallel]] zero-3 后仍有 shard）+ 梯度 + rollout 的 vLLM 副本权重 + KV cache + ref 权重 + critic 权重 + rm，全压一组卡。70B 模型同屏 4~5 份权重，即便 zero-3 也常 OOM。GRPO 砍 critic 省一份，仍紧。
2. **利用率波动**：rollout 阶段（采样）训练 SM 闲置（vLLM 在用卡算生成，但 backward/optimizer 闲）；训练阶段（forward+backward）rollout 副本 sleep 释放显存但 rollout 算力空转。两阶段交替，**GPU 平均利用率常只有 30~50%**，墙钟时间浪费。verl 的 `sleep_replicas()`/`wake()` 正是为缓解——rollout 完 sleep 让显存给训练、训完 wake 推新权重，但切换本身有开销。

### 2.2 全 disaggregate 的痛：sync 重 + stale

disaggregate 把 rollout 独立池、trainer 独立池。好处也明显——资源独立扩缩（rollout 池可远大于 trainer、按推理算力配）、不争用、训推 overlap（trainer 训上一 batch 时 rollout 采下一 batch）。但两痛：

1. **weight sync 重**：每 batch 后 trainer 把新 $\theta$（70B ~140GB）跨机推给 rollout 池，走 NCCL broadcast（数十~数百 GB/s IB）十毫秒级 + 协议开销。是 [[weight sync mechanism]] 路线一，但跨机延迟比 colocate in-process 高 1~2 个数量级。
2. **staleness**：rollout 池用旧 $\theta_{old-k}$ 采的样本喂 trainer，ratio $\rho=\pi_\theta/\pi_{old-k}$ 偏离 1，训练不稳。需 clip + 频繁 sync 兜底，但仍比 colocate 重。详见 [[stale policy problem]]、[[asynchronous training]]。

### 2.3 hybrid：取两者甜区

主流不是纯 colocate 也不是纯 disaggregate，而是 hybrid——把"可分时复用"的角色（ref 只 forward、critic 与 actor 可交替、rm 小）colocate 省显存，把"算力特征不同、需独立扩缩"的角色（rollout 需推理优化+大 KV、吞吐受 continuous batching 影响）独立池。这是 verl/OpenRLHF/HybridFlow 的实际默认。

### 2.4 选型即工程决策

部署模式选型直接决定：显存够不够、weight sync 开销多大、GPU 利用率多高、stale 多重、能否训推 overlap。是 RLHF 工程的"部署命脉"，与算法（PPO/GRPO）正交——同一算法可跑不同部署，同一部署可跑不同算法。


## 3. 核心概念详解

### 3.1 colocated（同机共置）

多个角色放**同一组 GPU、同进程或同节点**。核心机制：

- **显存分时复用**：role 在时间上不重叠用——rollout 阶段只 rollout 副本在线（actor 训练态权重 shard 留着但不 forward）、训练阶段 rollout 副本 sleep（释放显存给 forward/backward）。verl `sleep_replicas()`/`wake()` 实现。ref 只 forward 算 logp，与 critic forward 可穿插。
- **权重 in-process 共享**：rollout 副本与 actor 训练态常**同一份权重内存**（或 sleep 后 update_weights 把 actor 新 $\theta$ 原地刷给 rollout），weight sync 走 in-process（[[weight sync mechanism]] 路线二 CUDA IPC，零拷贝微秒级）或同机 NCCL（路线一，毫秒级）。
- **进程组同机**：trainer 的 NCCL 进程组与 rollout 的 vLLM TP 组在同一机内 NVLink 通信，跨 role 张量同步同机极快。

```
colocated（verl actor_rollout_ref 模式）:
┌──── 同一组 GPU（如 8 卡 H100）────┐
│  Ray actor 进程 (create_colocated_worker_cls 塞多 role)
│   ├ actor shard (FSDP/Megatron 训练态)
│   ├ rollout 副本 (vLLM, sleep/wake 切换)
│   └ ref 副本 (冻结, forward 算 logp)
│  显存分时: rollout 期 rollout 在线, actor 闲;
│          训练期 rollout sleep, actor forward/backward
└──────────────────────────────────┘
weight sync: in-process 原地刷 / 同机 NCCL, 微秒~毫秒级
```

**优点**：省卡（显存复用，一份卡跑多 role）、weight sync 极快（in-process）、无跨机 staleness（rollout 用最新 $\theta$）。
**缺点**：显存挤（大模型易 OOM）、role 切换 GPU 利用率波动（rollout/train 交替闲置）、扩缩不灵活（增 rollout 必增 trainer，无法独立配比）。

### 3.2 disaggregated（跨机分离）

各角色放**独立 GPU 池跨节点**。核心机制：

- **资源独立扩缩**：rollout 池、trainer 池、critic 池、rm 池各按算力特征配 GPU 数。rollout 池可大（吞吐受 continuous batching 影响，常需多副本）、trainer 池按梯度算力配、rm 池小。
- **weight sync 跨机**：trainer 每 batch 后把新 $\theta$ broadcast 到 rollout 池，走跨机 NCCL（路线一，十毫秒级）/共享 FS delta pull（路线四，秒级）/checkpoint engine（路线三，冷启动）。开销比 colocate in-process 高 1~2 个数量级。
- **staleness**：rollout 池用旧 $\theta_{old-k}$ 采，样本喂 trainer 有 lag。$k_{\max}\le S$（sync 间隔步数）。需 clip + 频繁 sync 兜底。
- **训推 overlap**：disaggregate 下 trainer 训 batch N、rollout 池同时采 batch N+1，两阶段 overlap 提吞吐（但 stale）。colocate 无法 overlap（同卡串行）。

```
disaggregated:
┌─trainer 池─┐         ┌─rollout 池─┐        ┌─critic 池─┐  ┌─rm 池─┐
│ 跨机 8×N    │ ─sync──►│ 跨机 8×M    │        │ 跨机     │   │ 小池  │
│ FSDP/Megra │  θ跨机   │ vLLM/SGLang │        │          │   │      │
└────────────┘         └────────────┘        └──────────┘   └──────┘
weight sync: 跨机 NCCL broadcast / 共享 FS / ckpt engine, 十毫秒~秒级 + stale
训推 overlap: trainer 训 batch N 时 rollout 采 N+1 (异步提吞吐)
```

**优点**：资源独立扩缩（按 role 配比）、不争用（隔离稳）、训推 overlap 提吞吐。
**缺点**：weight sync 重（跨机带宽）、staleness 风险、版本管理复杂（多池版本一致）。

### 3.3 hybrid（混合，主流）

主流部署是 hybrid——按角色特性混合 colocate/disaggregate：

- **trainer + ref colocate**：ref 只 forward 算 logp，与 actor 可分时（actor backward 时 ref 闲），colocate 省一份独立 ref 池。
- **rollout 独立池**（或部分 colocate 部分独立）：rollout 算力特征与训练不同（受 continuous batching/prefix caching/KV 影响），独立池按推理算力配比扩缩；大模型 rollout 池常需多副本提吞吐。
- **critic/RM 按需**：小 RM colocate（forward 完释放）、大 RM 或 PRM（process reward）独立池。

```
hybrid（verl/OpenRLHF 主流）:
┌── trainer 池（actor+ref colocate 同卡组）──┐   ┌─ rollout 池（独立）─┐  ┌─ rm 池 ─┐
│  actor shard + ref 副本 (ref forward 分时)  │   │ vLLM/SGLang 多副本   │  │ 小/独立  │
└────────────────────────────────────────────┘   └──────────────────────┘  └─────────┘
         │ weight sync (colocate 部分 in-process)        │ 跨机 NCCL/FS sync + stale
         └────────────────────────────────────────────────┘
critic (PPO): 与 trainer colocate 或独立池; GRPO 无 critic
```

verl 的 `actor_rollout_ref`（actor+rollout+ref 三塞同进程）是 hybrid 的极致 colocate 形态——连 rollout 都 colocate 进 actor 进程，靠 sleep/wake 分时。OpenRLHF 的 colocate 模式类似；hybrid 模式则 rollout 独立 vLLM 副本。详见 §3.5。

### 3.4 选型 trade-off 决策矩阵

| 决策因素 | 倾向 colocate | 倾向 disaggregate |
|---|---|---|
| 模型大小 | 中小模型（显存够 colocate） | 大模型（70B+，显存挤需分离） |
| GPU 资源 | 紧（省卡优先） | 充裕（隔离稳优先） |
| weight sync 频率 | 高频（每 batch sync，colocate 极快） | 可低频（sync 重可容忍） |
| 训推 overlap 需求 | 不需（同步串行可接受） | 需（异步提吞吐，容忍 stale） |
| rollout 吞吐瓶颈 | rollout 不是瓶颈 | rollout 是瓶颈（需独立扩缩池） |
| reward 性质 | rule-based（rm 不占 GPU） | 大 RM/PRM（需独立池） |
| 网络带宽 | 跨机带宽紧 | 跨机带宽充足（IB 400G+） |

### 3.5 verl / OpenRLHF / HybridFlow 的实践

| 框架 | 部署模式 | 关键机制 |
|---|---|---|
| **verl** | hybrid（默认 `actor_rollout_ref` colocate + 独立 critic/reward 池可配） | `create_colocated_worker_cls` 把 actor+rollout+ref 塞同 Ray actor 进程省显存；`spawn` 拆逻辑 wg；`sleep_replicas()`/`update_weights()` 分时显存 + in-process weight sync；`ResourcePoolManager` 灵活配各 role colocate 标志。见 [[Ray与分布式调度]] §3.5 |
| **OpenRLHF** | colocate 模式（actor+ref+critic+rm 同卡 sleep/wake）/ hybrid 模式（rollout 独立 vLLM 副本） | `--colocate` 开 colocate；`--vllm_sync_backend nccl/ipc` 选 sync 路线；colocate 时 sleep/wake 切换显存，hybrid 时 vLLM 独立副本 |
| **HybridFlow**（论文 arXiv:2409.19256） | 3D-HybridEngine（actor 训推态同组 colocate + reshard） | actor 角色既训练又其副本采样，训练态 3D 并行（TP/PP/DP）↔ rollout 推理态 TP，同组内 reshard 切换，避免跨机 weight sync；critic/ref/rm 按 placement 配。是 hybrid 的"训推同组 reshard"极致形态 |

> [!note] 补充：3D-HybridEngine 的 colocate 含义
> HybridFlow 的 3D-HybridEngine 把 actor 的训练态与生成态放**同一组 GPU**（colocate），靠 reshard（训练态 TP/PP/DP 分片 ↔ 推理态 TP 分片）在组内切换并行布局，**避免跨机 weight sync**——这是 colocate 的极致：连 rollout 都不独立池，靠 reshard 在同组内切。代价是 reshard 本身有开销（重切分片）、且仍受显存挤与利用率波动困扰。3D-HybridEngine 是 [[trainer与rollout引擎组合]] §3.5 的核心，本篇只点其部署含义。

> [!tip] 实践：大模型 RLHF 常需 hybrid 甚至偏 disaggregate
> - 7B 模型：全 colocate（actor_rollout_ref）够，省卡。
> - 30B+：rollout 独立池（避免与 trainer 争显存/KV），trainer+ref colocate。
> - 70B+：critic 也可独立（GRPO 直接无 critic 省事），rollout 多副本独立池，trainer Megatron 3D 并行跨机。weight sync 用跨机 NCCL broadcast + 频率 tradeoff（每 N step sync 一次减开销，但 stale 增）。


## 4. 数学原理 / 公式

### 4.1 colocate 的显存复用核算

colocate 多角色同卡，显存峰值非"各角色简单相加"——靠分时复用，峰值是"同时在线角色"之和：

$$
M_{\text{colocate}} = \max_{\text{phase}} \sum_{\text{role online in phase}} M_{\text{role}}
$$

- rollout 阶段：rollout 副本（权重+KV）在线，actor 训练态权重 shard 留（但 forward/backward/optim 闲）→ $M\approx M_{\text{actor shard}}+M_{\text{rollout}}+M_{\text{ref}}$。
- 训练阶段：rollout sleep（释放），actor 全态（权重+grad+optim states）+ ref forward → $M\approx M_{\text{actor full}}+M_{\text{ref}}$。

vs 全 disaggregate（各池独立）总显存 $M_{\text{disagg}}=\sum_{\text{all role}} M_{\text{role}}$（各池常驻全态）。colocate 靠分时把峰值压到"同 phase 在线角色"和，省 $M$ 的量级 = 不重叠角色之和。这是 colocate 省卡的数学根。

> [!warning] 注意：colocate 显存不是简单"一份权重"
> colocate 仍要存 actor 训练态全显存（权重 shard + 梯度 + Adam 状态，zero-3 后 shard 仍大）+ rollout 副本权重 + KV cache + ref。sleep 只释放 rollout 副本给训练 forward，不是"角色共用一份权重内存"那么简单。大模型仍易 OOM，故 hybrid。

### 4.2 weight sync 通信量与延迟对比

一次 weight sync 体积 $V=P\cdot\text{size}$（70B FP16 ≈ 140GB）。

$$
T_{\text{sync}} = \frac{V}{\text{BW}_{\text{eff}}} + T_{\text{overhead}}
$$

| 模式 | 介质 | $\text{BW}_{\text{eff}}$ | $T_{\text{sync}}$（70B） | overhead |
|---|---|---|---|---|
| colocate in-process（CUDA IPC） | 同机共享显存 | 显存带宽（~TB/s） | 微秒级 | 零拷贝，最小 |
| colocate 同机 NCCL | NVLink | 数百 GB/s | 毫秒级 | 协议轻 |
| disaggregate 跨机 NCCL broadcast | IB | 数十~数百 GB/s | 十毫秒级 | 跨机协议 |
| disaggregate 共享 FS delta pull | 共享存储+zstd | 受网络/磁盘 | 秒级 | delta 压缩+checksum |
| disaggregate checkpoint engine | 磁盘→多机并行 | 放大磁盘带宽 | 秒~十秒级 | 冷路径 |

colocate 的 in-process/同机路径比 disaggregate 跨机快 **1~3 个数量级**——这是 colocate 吞吐优势的主因（高频每 batch sync 下，跨机开销累积致命）。详见 [[weight sync mechanism]] §3.4 已核实四路线对比。

### 4.3 disaggregate 的 staleness 上界

disaggregate 下 rollout 用 $\theta_{old-k}$（落后 $k$ 步），最大 staleness：

$$
k_{\max} \le S\quad(\text{两次 sync 间隔训练步数})
$$

- $S=1$（每步 sync）：近同步，$k_{\max}=1$，但 sync 开销重。
- $S$ 大：sync 开销小，但 $k_{\max}$ 大，ratio $\rho=\pi_\theta/\pi_{old-k}$ 偏离 1 多，训练不稳。

ratio 偏离：

$$
\rho_t = \frac{\pi_\theta(o_t)}{\pi_{\theta_{old-k}}(o_t)} = \exp(\log\pi_\theta - \log\pi_{old-k})
$$

$k$ 越大 $\rho$ 越偏离 1，需 PPO clip $\varepsilon$ 兜底（维持 clip frequency 0.1~0.3）。colocate $k\approx0$（rollout 用最新 $\theta$），无此问题。详见 [[stale policy problem]]、[[weight sync mechanism]] §4.1。

### 4.4 GPU 利用率：colocate 的波动 vs disaggregate 的稳

colocate 两阶段交替，利用率：

$$
U_{\text{colocate}} \approx \frac{T_{\text{rollout useful}} + T_{\text{train useful}}}{T_{\text{rollout}} + T_{\text{train}} + T_{\text{switch}}}
$$

rollout 阶段训练 SM 闲、训练阶段 rollout 闲 + sleep/wake 切换开销 $T_{\text{switch}}$ → 平均 $U$ 常 30~50%。

disaggregate 各池独立，trainer 池全跑训练 $U\approx100\%$、rollout 池全跑生成 $U\approx100\%$（含 continuous batching 满载），无切换。但训推 overlap 下 trainer 用 stale 数据，且总卡数多（各池独立）。

$$
\text{吞吐 trade-off}: \text{colocate 省 sync + 省卡但 }U\text{ 低; disaggregate }U\text{ 高 + overlap 但 sync 重 + stale}
$$

大模型下 colocate 的 $U$ 低 + 显存挤常抵消省 sync 优势，故偏 disaggregate/hybrid。


## 5. 代码示例（可选）

### 5.1 verl colocate 配置（概念，配置名以官方文档为准）

```python
# verl ResourcePoolManager 决定各 role colocate 还是独立池
# 来自 verl single_controller（见 Ray与分布式调度 §3.5）
from verl.single_controller.ray import RayResourcePool

# 模式 A: 全 colocate（中小模型省卡）
pools_colocate = {
    "actor_rollout_ref_pool": RayResourcePool(
        process_on_nodes=8,          # 8 卡一机
        use_gpu=True, num_gpus=8,
        colocate=True,               # actor+rollout+ref 塞同进程
        colocate_with=["actor_rollout_ref"]  # 共置
    ),
    # critic/rm 也塞同池 (role_worker_mapping 都映到同一 pool)
}

# 模式 B: hybrid（大模型，rollout 独立池）
pools_hybrid = {
    "actor_ref_pool": RayResourcePool(      # trainer + ref colocate
        process_on_nodes=8, num_gpus=8, colocate=True,
    ),
    "rollout_pool": RayResourcePool(        # rollout 独立 (推理算力配比)
        process_on_nodes=8, num_gpus=16,    # rollout 池可更大
        colocate=False,
    ),
    "critic_pool": RayResourcePool(num_gpus=8, colocate=False),  # PPO critic 独立; GRPO 无
    "reward_pool": RayResourcePool(num_gpus=2, colocate=True),   # 小 rm colocate
}
# create_colocated_worker_cls(class_dict) 把同池多 role 塞同 Ray actor 进程
# spawn(prefix_set) 按 role 拆逻辑 wg (物理共置, 逻辑分离)
```

### 5.2 verl sleep/wake 分时显存（fit 主循环片段，来自 [[Ray与分布式调度]] §3.5）

```python
# colocate 核心: rollout 副本与 actor 训练态同卡, 靠 sleep/wake 分时
for batch in train_dataloader:
    # rollout 阶段: vLLM 副本在线采, actor 闲
    gen = self.async_rollout_manager.generate_sequences(batch)
    self.checkpoint_manager.sleep_replicas()   # ★ 释放 rollout 副本显存给训练
    batch = batch.union(gen)

    # 训练阶段: actor forward/backward, rollout 副本 sleep
    ...reward, old_log_prob, ref_log_prob, values, advantage...
    self._update_actor(batch)                 # backward 更新 θ

    # weight sync: 新 θ → rollout 副本 (colocate in-process 极快)
    self.checkpoint_manager.update_weights(self.global_steps)  # ★ wake + 原地刷权重
```

> [!note] sleep_replicas / update_weights = colocate 的 weight sync
> colocate 下 rollout 副本与 actor 同卡，`sleep_replicas()` 把 vLLM 副本显存让给训练 forward，训完 `update_weights()` 把新 $\theta$ 原地刷给 vLLM 副本并 wake——这就是 [[weight sync mechanism]] 路线二（CUDA IPC in-process）在 colocate 的落地，零跨机开销。disaggregate 下 `update_weights` 改走跨机 NCCL broadcast（路线一）。同一 API、不同 sync 后端。

### 5.3 OpenRLHF 模式切换（配置项以官方文档为准）

```bash
# OpenRLHF colocate 模式: actor+ref+critic+rm 同卡 sleep/wake
python -m openrlhf.cli.train.ppo_ray \
  --colocate \                          # ★ colocate 模式
  --vllm_num_engines 4 --vllm_sync_backend nccl \  # rollout vLLM, sync NCCL
  --zero_stage 3 --bf16

# OpenRLHF hybrid 模式: rollout 独立 vLLM 副本跨机
python -m openrlhf.cli.train.ppo_ray \
  --vllm_num_engines 8 \                # 独立 rollout 副本
  --vllm_sync_backend ipc \             # 跨机用 IPC/NCCL (同机 IPC)
  --colocate_critic_model \             # critic 与 actor colocate, rollout 独立
  --zero_stage 3 --bf16
# disaggregate: 去 --colocate, 全角色独立池, sync 走跨机 NCCL/共享 FS
```

### 5.4 吞吐模拟：colocate vs disaggregate 利用率

```python
# 概念: 比较 colocate(串行, U 低) vs disaggregate(overlap, U 高但 stale)
def colocate_throughput(T_roll, T_train, T_sync_in=0.001, T_switch=0.05):
    # 串行: rollout → sleep → train → wake → sync(in-process)
    return 1.0 / (T_roll + T_train + T_switch + T_sync_in)   # U ~ (useful)/(total)

def disaggregate_throughput(T_roll, T_train, T_sync_cross=0.05, overlap=True):
    # overlap: max(roll, train) + sync (stale)
    if overlap:
        return 1.0 / (max(T_roll, T_train) + T_sync_cross)   # U 高, 但 stale
    return 1.0 / (T_roll + T_train + T_sync_cross)

# 大模型: T_roll, T_train 都大, colocate 串行 U 低;
# disaggregate overlap 提吞吐但 T_sync_cross 重 + stale
# hybrid: trainer+ref colocate(省 ref 池 + in-process sync), rollout 独立(overlap)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[RL角色拓扑]]（五角色定义，本篇是其部署轴展开）、[[trainer与rollout引擎组合]]（引擎轴，与本篇正交）、[[weight sync mechanism]]（colocate in-process vs disaggregate 跨机的四路线）、[[训推分离]]（disaggregate 是 RLHF 内的训推分离）、[[Ray与分布式调度]]（colocate 用 `create_colocated_worker_cls`+`spawn` 落地）、[[Fully Sharded Data Parallel]]/[[Megatron-LM]]（trainer 训练态显存与并行，决定 colocate 显存够不够）。
- **下游（应用）**: [[17-RL训推一体框架]] 全章、[[synchronous与asynchronous rollout]]（disaggregate 训推 overlap 是异步基础）、[[stale policy problem]]（disaggregate 的代价）、[[asynchronous training]]、[[RL权重同步]]。
- **对比 / 易混**:
  - **colocate vs disaggregate（本篇）vs [[训推分离]] PT/PD**：前者是 RLHF 训练内部角色同机/跨机；后者是训练↔部署分离。disaggregate RLHF 是"训练内训推分离"，PT/PD 是"训推部署分离"，机制同构但场景不同。
  - **colocate vs [[PD分离]] prefill/decode 分离**：前者是角色级同机/跨机；后者是推理内 prefill/decode 阶段分离。不同层。
  - **3D-HybridEngine（colocate 极致）vs disaggregate rollout 池**：前者 rollout 不独立池靠同组 reshard；后者 rollout 独立池跨机。前者省跨机 sync 但 reshard 开销 + 显存挤；后者 sync 重但隔离。


## 7. 常见误区与易错点

> [!warning] 误区 1：colocate 就是"一份权重共用"
> 不完全是。colocate 是多角色同卡，靠 sleep/wake 分时复用显存——rollout 副本与 actor 训练态仍是各自显存区域（rollout 副本 vLLM 权重 + actor FSDP shard + optim states），sleep 释放 rollout 副本给训练，不是"共用同一份内存"。weight sync 仍要把 actor 新 $\theta$ 刷给 rollout 副本（in-process，但非零拷贝除非用 IPC handle）。

> [!warning] 误区 2：disaggregate 一定比 colocate 快
> 不一定。disaggregate U 高 + 训推 overlap，但每 batch 一次跨机 weight sync（~140GB）开销重，小模型/高频 sync 下 colocate in-process 反而快。大模型/rollout 瓶颈时 disaggregate 才显优势。看规模与瓶颈。

> [!warning] 误区 3：colocate 无 staleness
> 同步 colocate 下 rollout 用最新 $\theta$，$k\approx0$。但若 colocate + 多 epoch 复用同一 rollout batch（PPO 风格 off-policy），$\theta$ 已更新而样本还是旧 $\theta_{old}$ 采的，仍有 epoch 内 staleness（靠 ratio clip 修）。colocate 消除的是"跨步 sync staleness"，非"epoch 内 off-policy"。

> [!warning] 误区 4：GRPO 必须 colocate
> 错。GRPO 只是算法（去 critic），与部署模式正交——GRPO 可跑 colocate 也可 disaggregate。GRPO 省 critic 显存使 colocate 更易（少一份权重），故 GRPO 常配 colocate，但非强制。

> [!warning] 误区 5：3D-HybridEngine 是 disaggregate
> 相反。3D-HybridEngine 是 colocate 的极致——actor 训推态同组 GPU 靠 reshard 切换，避免跨机 weight sync。是"同组 reshard"，非"跨机分离"。

> [!warning] 误区 6：全 colocate 省卡所以总最优
> 错。省卡但 U 低（30~50%）+ 显存挤易 OOM，墙钟时间常更长。大模型下 hybrid/disaggregate 的 U 高 + overlap 抵消多卡开销。选型看规模/瓶颈/资源，非"省卡即优"。

> [!tip] 实践调参信号
> 选 colocate/disaggregate 看：①显存是否够 colocate（OOM 则分离）；②weight sync 频率（高频 favor colocate）；③rollout 是否瓶颈（瓶颈则独立池扩缩）；④clip frequency（disaggregate stale 重则 clip 频率升，>0.3 则需更频繁 sync）。


## 8. 延伸细节

### 8.1 verl 的 colocate 实现：create_colocated_worker_cls + spawn

verl 把同池多 role 的 worker 类用 `create_colocated_worker_cls(class_dict)` 塞进**同一个 Ray actor 进程**（物理共置省显存：rollout 与 actor 可共享权重内存），`spawn(prefix_set)` 再按 role 前缀拆出多个 `RayWorkerGroup` 视图——**物理同一组 actor，逻辑多组 wg**。这是 verl 区别于 OpenRLHF 的核心工程优化之一，详见 [[Ray与分布式调度]] §3.5 解答。`ResourcePoolManager` 的 `colocate` 标志控制是否共置。

### 8.2 sleep/wake 的显存时序

verl colocate 的显存时序：rollout 采完 → `sleep_replicas()` 把 vLLM 副本权重+KV 卸载（常 offload 到 CPU 或释放）→ actor forward/backward/optim 用腾出的显存 → `update_weights()` 把 actor 新 $\theta$ 刷给 vLLM 副本 + `wake()` 重新加载副本 → 下 batch rollout。sleep/wake 本身有开销（CPU↔GPU 搬运 + vLLM 重新初始化 CUDA Graph 等），是 colocate $T_{\text{switch}}$ 的来源。大模型下 sleep/wake 开销显著，部分场景用"不 sleep 直接共存"换更简但更挤的显存。待核实各框架 sleep 策略细节。

### 8.3 disaggregate 的训推 overlap 与异步

disaggregate 下 trainer 训 batch N、rollout 池同时采 batch N+1，两阶段 overlap。这是 [[synchronous与asynchronous rollout]] 的异步基础——但要处理：①rollout 用 $\theta_{old-k}$ 的 stale（ratio clip 修）；②样本回传 trainer 的数据 sync；③多版本样本混 batch（partial rollout）。OpenRLHF 的 async/partial rollout 模式、verl fully_async 模式都是 disaggregate + overlap 的工程化。详见 [[asynchronous training]]。

### 8.4 weight sync 后端选择

colocate 选 IPC/in-process（最快）；disaggregate 同机跨进程选 NCCL（同机 NVLink）；disaggregate 跨机选 NCCL broadcast（热路径）/共享 FS delta pull（非 colocated rollout 多副本）/checkpoint engine（冷启动）。四路线详见 [[weight sync mechanism]] 已联网核实的总结。verl `WeightSyncClient/Server` + NCCL；OpenRLHF `--vllm_sync_backend nccl/ipc`；SGLang checkpoint engine + `/pull_weights`。

### 8.5 大模型部署演进趋势

社区趋势：中小模型仍 colocate（省卡简单）；70B+ 偏 hybrid（rollout 独立池 + trainer colocate ref）；超大规模 RL（如 R1 范式）偏 disaggregate + 异步 overlap（提吞吐容忍 stale，靠 clip + 频繁 sync 兜底）。HybridFlow 3D-HybridEngine 的"同组 reshard"是 colocate 极致的学术探索，工业上多按规模分档选 colocate/hybrid/disaggregate。待核实各框架最新默认。

### 8.6 内容来源

colocate/disaggregate 概念综合自 verl 文档（`single_controller`/`ResourcePoolManager`/`create_colocated_worker_cls`）、OpenRLHF 文档（`architecture.html`/`hybrid_engine.rst`/`async_training.html`）、HybridFlow 论文 arXiv:2409.19256（3D-HybridEngine）。weight sync 路线与延迟对比来自 [[weight sync mechanism]] 已联网核实的四路线总结（vLLM WeightTransferEngine/SGLang checkpoint engine/SGLang /pull_weights/OpenRLHF sync_backend）。verl fit 时序与 sleep_replicas/update_weights 来自 [[Ray与分布式调度]] §3.5 已核实代码解答。截至 2026-07。

---
相关: [[角色拓扑]] | [[RL角色拓扑]] | [[trainer与rollout引擎组合]] | [[weight sync mechanism]] | [[RL权重同步]] | [[训推分离]] | [[stale policy problem]] | [[asynchronous training]] | [[synchronous与asynchronous rollout]] | [[Ray与分布式调度]] | [[Fully Sharded Data Parallel]] | [[Megatron-LM]] | [[continuous batching的调度实现]] | [[rollout worker]] | [[inference worker]] | [[learner worker split]] | [[新模型接入]] | [[17-RL训推一体框架]]
