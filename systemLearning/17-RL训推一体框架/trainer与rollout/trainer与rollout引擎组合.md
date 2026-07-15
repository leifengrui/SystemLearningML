# trainer与rollout引擎组合

> **所属章节**: [[trainer与rollout]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: FSDP/Megatron trainer + vLLM/SGLang rollout / 训推两端引擎组合 / trainer-rollout engine pairing / 训练框架×推理框架组合矩阵 / weight reshard / HybridFlow 统一抽象
> **难度**: 高（需懂 [[RL角色拓扑]]、[[colocated与disaggregated部署]]、[[weight sync mechanism]]、[[Fully Sharded Data Parallel]]、[[Megatron-LM]]、[[continuous batching的调度实现]]、[[rollout train reference logprob一致性]]）


## 1. 一句话定义

**trainer + rollout 引擎组合** 是 [[17-RL训推一体框架]] 在**引擎维度的两端组合**——训练端（trainer）用 [[Fully Sharded Data Parallel]] / [[Megatron-LM]] / DeepSpeed-ZeRO 这类**训练框架**，管训练态显存（权重 shard + 梯度 + Adam 优化器状态 + 并行：ZeRO/FSDP 切分或 Megatron 3D TP/PP/DP）；采样端（rollout）用 vLLM/SGLang 这类**推理框架**，享推理优化（[[PagedAttention]] KV cache、[[continuous batching的调度实现]]、CUDA Graph、prefix caching）；因为两端是**不同框架、不同权重切分布局**，actor 训练完一个 batch 后必须把 $\theta$ 做**格式转换 + 并行 reshard** 再 `load_weights` 进 rollout 引擎（FSDP 要 all-gather 还原全量、Megatron 要从 TP/PP/DP 训练态 reshard 到推理态 TP）。之所以必须拆两端——训练框架无推理优化（采样吞吐低一个数量级）、推理框架无梯度/优化器/并行训练支持（训不了）；故 RL 训推一体的工程核心就是"两端引擎各自最优 + 中间 weight sync 桥接"。verl 的 FSDP/Megatron trainer + vLLM/SGLang rollout 组合、OpenRLHF 的 DeepSpeed trainer + vLLM rollout 同构、HybridFlow 的统一抽象（control flow 与 distributed execution 解耦，trainer/rollout 后端可插拔）是代表实现。是 [[RL角色拓扑]] 的 actor/rollout 角色在"用什么引擎跑"轴的专篇，与 [[colocated与disaggregated部署]]（同机/跨机轴）正交。

> [!note] 三句话定位
> - **是什么**：训练用训练框架（FSDP/Megatron/DeepSpeed，管梯度/optim/并行），采样用推理框架（vLLM/SGLang，KV cache/batching/CUDA Graph），两端靠 weight sync（转换+reshard）桥接。
> - **为什么**：训练框架采样吞吐低（无推理优化）、推理框架训不了（无梯度/optim/并行训练）——两端各有不可替代优势，必须组合 + 中间转换。
> - **与 [[RL角色拓扑]] / [[colocated与disaggregated部署]] 关系**：本篇是"角色用什么引擎"（引擎轴）；[[RL角色拓扑]] 是"有哪些角色"（角色轴）；[[colocated与disaggregated部署]] 是"角色放哪"（部署轴）。三轴正交组合出具体部署。
> 
> > [!note] 文件名说明
> > 此笔记对应 整体目录.md §81 的 `[[FSDP/Megatron trainer + vLLM/SGLang rollout]]`，因原名含 `/` 作文件名会成路径歧义，故本笔记文件名为 `trainer与rollout引擎组合.md`。主控会更新 整体目录.md 用管道别名 `[[trainer与rollout引擎组合|FSDP/Megatron trainer + vLLM/SGLang rollout]]`。


## 2. 为什么需要它（动机与背景）

### 2.1 训练框架采样吞吐低

用 FSDP/Megatron 裸跑采样（forward 逐 token 生成，无 KV cache 复用、无 continuous batching 动态混、无 CUDA Graph）：
- **无 KV cache**：每生成一个 token 重新算所有历史 token 的 attention（$O(L^2)$ 重复算），autoregressive 生成几百 token 极慢。
- **无 continuous batching**：训练框架 batch 静态，长短不一的 response 要 padding 到同长，浪费算力；请求不能动态进出。
- **无 CUDA Graph**：每步 forward kernel launch 开销重，小 token 生成被 launch 开销吃掉。
- **显存态重**：训练框架带梯度/优化器状态/activations，forward 时显存占用高、batch 上限受制。

实测训练框架裸采样吞吐比 vLLM 低 **一个数量级**。而 rollout 是 RLHF 墙钟时间大头（生成几百~几千 token × 成千上万条），低吞吐直接拖死整个训练。→ **必须用推理框架采样**。

### 2.2 推理框架训不了

用 vLLM/SGLang 裸跑训练：
- **无 backward**：推理框架为低延迟 forward 优化，没有 autograd engine、不记计算图、不算梯度。
- **无 optimizer**：无 Adam/AdamW 状态管理、无 lr scheduler、无 gradient clipping。
- **无并行训练**：无 FSDP/Megatron 的 ZeRO 切分/TP/PP/DP AllReduce/AllGather，无法多机多卡训大模型。
- **权重布局为推理优化**：vLLM 的权重常按 TP 切分 + 为 CUDA Graph/paged KV 优化布局，不是训练态的 FSDP shard/Megatron TP/PP 切分。

→ **必须用训练框架训练**。两端各有不可替代优势，必须组合。

### 2.3 两端组合的代价：weight sync 桥接

两端不同框架 → actor 训练完一个 batch 后，新 $\theta$ 不能直接给 rollout 用，要：
1. **从训练框架导出**：trainer 的 $\theta$ 按 FSDP shard（每 rank 持一片）/Megatron TP/PP/DP 切分（每 rank 持 TP 切片+PP 层段+DP 复制）持有，不是完整权重。
2. **转换格式**：训练态数据类型（常 FP32 master weight + BF16 param）、布局（FSDP shard/Megatron 分片）→ 推理态布局（vLLM TP 切分、BF16/FP16、按 layer 重组）。
3. **reshard 并行布局**：训练态并行度（如 Megatron TP=4/PP=2/DP=2）→ 推理态并行度（vLLM TP=8），切分方式不同需重切。
4. **load_weights 进 rollout**：vLLM/SGLang 的 `load_weights`/`update_weights` API 载入。

这步是 [[weight sync mechanism]] 的核心，通信量 = 全模型权重（~140GB/70B），是 RL 训推一体最频繁的跨引擎通信。colocate 走 in-process 极快、disaggregate 走跨机 NCCL 重。详见 [[colocated与disaggregated部署]]。

### 2.4 为什么不写"一个框架既训练又采样"

理论上可——PyTorch 原生 forward 既能训练（有 autograd）又能采样（forward 逐 token）。但如 §2.1 所述，采样吞吐低一个数量级，RLHF 训不动。故业界共识：**训练用训练框架、采样用推理框架、中间桥接 weight sync**，是性能与工程复杂度的最优 trade-off。HybridFlow 论文（arXiv:2409.19256）正是把这套组合抽象成"统一接口"。


## 3. 核心概念详解

### 3.1 训练端框架：FSDP / Megatron / DeepSpeed

| 训练框架 | 并行策略 | 权重切分 | 优势 | 代表 |
|---|---|---|---|---|
| **FSDP**（[[Fully Sharded Data Parallel]]） | ZeRO-3 风格数据并行 | 每 DP rank 持一片（params/grads/optim 都切） | 简单、PyTorch 原生、中小~大模型通用、无 TP/PP 复杂度 | verl FSDP trainer、TRL PPO、OpenRLHF DeepSpeed ZeRO-3 |
| **Megatron-LM** | 3D 并行（TP/PP/DP） | TP 切 attention/FFN 维、PP 切层段、DP 复制 | 超大模型（70B+）必用、TP 减单卡显存、PP 切深度 | verl Megatron trainer、Megatron-DeepSpeed |
| **DeepSpeed-ZeRO** | ZeRO-1/2/3 | optim/grad/param 逐级切分 | 与 Megatron 可组合、ZeRO-3≈FSDP | OpenRLHF `--zero_stage 3` |

训练框架管：**权重 shard + 梯度 + Adam 状态（m/v）+ AllReduce/AllGather 通信 + lr/clip + checkpoint**。forward 算 logp（供 ratio）、backward 算梯度、optimizer step 更新 $\theta$。

### 3.2 采样端框架：vLLM / SGLang

| 推理框架 | 核心优化 | 权重布局 | 代表 |
|---|---|---|---|
| **vLLM** | [[PagedAttention]]（非 contiguous KV block）、[[continuous batching的调度实现]]（动态进出）、CUDA Graph、prefix caching | 按 TP 切分，BF16/FP16 | verl/OpenRLHF 默认 rollout |
| **SGLang** | RadixAttention（prefix caching）、continuous batching、CUDA Graph、复杂结构（结构化生成/JSON） | 按 TP 切分 | verl 可选 rollout |

推理框架管：**KV cache 管理 + continuous batching + CUDA Graph + 高吞吐 forward**。无梯度/无 optim/无训练并行。是 rollout 角色的高吞吐引擎。

### 3.3 为什么不能直接用训练框架采样 / 推理框架训练

| 维度 | 训练框架（FSDP/Megatron）采样 | 推理框架（vLLM）训练 |
|---|---|---|
| KV cache | 无（每 token 重算历史 attention $O(L^2)$） | 有（增量算 $O(L)$） | 
| continuous batching | 无（静态 batch + padding 浪费） | 有（动态进出，padding-free） |
| CUDA Graph | 无（kernel launch 开销重） | 有（捕获重放，launch 近零） |
| backward/autograd | 有（能训） | 无（训不了） |
| optimizer/状态 | 有（Adam m/v，lr/clip） | 无 |
| 多机多卡训练并行 | 有（AllReduce/AllGather/TP/PP） | 无 |
| → 适合 | 训练 | 采样 |

训练框架采样吞吐低 1 个数量级（无 KV/batching/CUDA Graph）；推理框架训不了（无 backward/optim/并行）。故必须拆两端组合。

### 3.4 权重同步路径：trainer → rollout

两端权重切分布局不同，sync 不仅是"传张量"，还要**格式转换 + reshard**：

```
trainer (训练态)                    转换层                      rollout (推理态)
┌─────────────────┐                ┌──────────────────┐         ┌─────────────────┐
│ FSDP shard:     │                │ 1. all-gather    │         │ vLLM TP shard:  │
│  每 rank 持一片  │ ──state_dict──►│    还原全量 θ    │ ────────►│  按 TP 重切     │
│  (FP32 master + │                │ 2. dtype 转 BF16 │ load_   │  (BF16, layer  │
│   BF16 param)   │                │ 3. 按 rollout TP │ weights │   重组)         │
├─────────────────┤                │    重切 (reshard)│         ├─────────────────┤
│ Megatron 3D:    │                │ 4. packed tensor │         │ SGLang 同构,    │
│  TP 切 + PP 段  │ ──state_dict──►│    打包(可选)    │ ────────►│  RadixAttention │
│  + DP 复制      │                │ 5. NCCL/IPC 传输  │         │  cache 预热     │
└─────────────────┘                └──────────────────┘         └─────────────────┘
```

**FSDP trainer → rollout**：
1. **all-gather 还原全量**：FSDP 每 rank 只持 $\theta$ 的一片（shard），要 sync 给 rollout（需完整权重按 TP 切）必须先 all-gather 把各 rank 片聚成完整 $\theta$。
2. **dtype 转**：训练态常持 FP32 master weight + BF16 param；rollout 用 BF16/FP16，转 dtype。
3. **按 rollout TP 重切**：rollout vLLM 的 TP 并行度可能与 trainer DP 不同，按 rollout TP 重新切分。
4. **load_weights**：vLLM `load_weights`/`update_weights` 载入。

**Megatron trainer → rollout**（更复杂）：
1. **TP 还原**：Megatron TP 切 attention/FFN 维（每 rank 持列切片），要 all-gather 还原完整层。
2. **PP 重组**：Megatron PP 切层段（每 rank 持若干层），要跨 PP rank 聚成完整层序列。
3. **DP 去重**：Megatron DP 复制（各 DP rank 同权重），取一份即可。
4. **按 rollout TP 重切**：训练态 TP/PP/DP → 推理态 TP，reshard 重切。
5. **load_weights**。

### 3.5 HybridFlow 3D-HybridEngine：actor 训推态同组 reshard

HybridFlow 论文（arXiv:2409.19256）的 3D-HybridEngine 把 actor 的训练态与生成态放**同一组 GPU**（colocate），靠 reshard 在组内切换并行布局，**避免跨机 weight sync**：
- 训练态：Megatron 3D 并行（TP/PP/DP），权重按 TP 列切 + PP 层段 + DP 复制。
- 生成态：vLLM 风格 TP 并行，权重只按 TP 切。
- **reshard**：同组 GPU 内，把训练态 TP/PP/DP 分片 → 推理态 TP 分片，靠 NVLink all-gather + 重新切分完成，不跨机。

这是 §3.4 reshard 的"同组极致"——省跨机 weight sync（actor 与 rollout 同组），代价是 reshard 本身有开销（all-gather + 重切）+ 显存挤（两态权重同组）。是 [[colocated与disaggregated部署]] §3.5 的核心。verl 的 Megatron trainer + colocate rollout 即此实现。

### 3.6 verl / OpenRLHF / HybridFlow 的组合实现

| 框架 | trainer 后端 | rollout 后端 | weight sync 桥接 |
|---|---|---|---|
| **verl** | FSDP / Megatron（可选 `ActorRolloutRefWorker` 配 trainer backend） | vLLM / SGLang（可选） | `update_weights`：trainer state_dict → 转换 → rollout `load_weights`；FSDP all-gather 还原全量、Megatron reshard；colocate 走 in-process IPC，disaggregate 走 NCCL broadcast |
| **OpenRLHF** | DeepSpeed ZeRO-3（≈FSDP）/ Megatron | vLLM（`RayvLLMRollout`） | `--vllm_sync_backend nccl/ipc`：trainer broadcast → vLLM `update_weights`；ZeRO-3 all-gather 还原全量再下发 |
| **HybridFlow**（论文） | 统一抽象：trainer 后端可插拔（FSDP/Megatron） | rollout 后端可插拔（vLLM/SGLang/sglang） | control flow 与 distributed execution 解耦，weight sync 是统一接口的桥接层；3D-HybridEngine 是 colocate 同组 reshard 的特化 |

### 3.7 组合矩阵：训练框架 × 推理框架

| 训练框架 \ 推理框架 | vLLM | SGLang | (训练框架裸采样) |
|---|---|---|---|
| **FSDP** | ✅ verl/OpenRLHF 主流（all-gather 还原全量→TP 重切） | ✅ verl 可选 | ❌ 吞吐低 1 数量级 |
| **Megatron-LM** | ✅ verl 大模型（reshard TP/PP/DP→TP） | ✅ verl 可选 | ❌ 吞吐低 |
| **DeepSpeed-ZeRO-3** | ✅ OpenRLHF（≈FSDP+vLLM） | ✅ 可选 | ❌ 吞吐低 |
| **(推理框架裸训练)** | ❌ 无 backward/optim/并行 | ❌ 同 | — |

主流组合：**FSDP trainer + vLLM rollout**（中小~大模型通用，最简单，verl/OpenRLHF 默认）与 **Megatron trainer + vLLM/SGLang rollout**（超大模型，需 3D reshard，verl/HybridFlow 3D-HybridEngine）。


## 4. 数学原理 / 公式

### 4.1 FSDP all-gather 还原全量的通信量

FSDP 每 DP rank 持 $\theta$ 的 $1/N_{\text{dp}}$ 片（$N_{\text{dp}}$=DP 并行度）。要给 rollout 完整 $\theta$，需 all-gather：

$$
\text{Vol}_{\text{gather}} = P\cdot\text{size}\quad(\text{全量 } \theta,\ \text{各 rank 片聚成完整})
$$

- $P$：参数量（70B ≈ $70\times10^9$），size：BF16=2 → ~140GB。
- all-gather 后每 rank 持完整 $\theta$（临时），再按 rollout TP 切下发。
- 通信走 NVLink（colocate 同机，数百 GB/s）或 IB（disaggregate 跨机，数十~数百 GB/s）。

vs 不 all-gather 直接传 shard：rollout 需完整 $\theta$（按其 TP 切），不能直接用 FSDP shard，故 all-gather 不可省（除非 rollout 与 trainer 同并行度同切分，则可直接复用，但罕见）。

### 4.2 Megatron reshard 的通信量

Megatron 训练态 TP=$T_t$/PP=$P_t$/DP=$D_t$，rollout 推理态 TP=$T_r$。reshard：

1. **TP 还原**：每 TP rank 持 attention/FFN 列切片，all-gather 还原完整层：$\text{Vol}\approx P\cdot\text{size}$（全模型）。
2. **PP 重组**：跨 PP rank 聚层段，通信量 = 全模型（各 PP rank 互发自己层的完整副本）。
3. **按 $T_r$ 重切**：完整 $\theta$ 按 rollout TP 切，每 rollout rank 取一片。

总 resh ard 通信量 $\approx O(P\cdot\text{size})$（与全量 sync 同量级，因要还原全量再重切）。3D-HybridEngine 在同组靠 NVLink 高带宽做这步，避免跨机；disaggregate 则跨机，开销大。HybridFlow 论文核心贡献之一是优化这步的通信模式（如层间流水 reshard、packed tensor 打包减 launch 开销）。

### 4.3 weight sync 总通信与延迟

$$
T_{\text{sync}} = T_{\text{gather}} + T_{\text{convert}} + T_{\text{reshard}} + T_{\text{transfer}} + T_{\text{load}}
$$

- $T_{\text{gather}}$：FSDP all-gather / Megatron TP+PP 还原（同组 NVLink 快、跨机慢）。
- $T_{\text{convert}}$：dtype 转 + tensor 重组（CPU 算，常与 gather overlap）。
- $T_{\text{reshard}}$：按 rollout TP 重切（同组快）。
- $T_{\text{transfer}}$：传给 rollout 进程（colocate in-process / 同机 NCCL / 跨机 NCCL broadcast）。
- $T_{\text{load}}$：vLLM/SGLang `load_weights`（写显存 + 重建 CUDA Graph/paged KV 元数据）。

| 模式 | $T_{\text{sync}}$（70B） |
|---|---|
| FSDP colocate in-process | 毫秒级（all-gather 同机 NVLink + IPC 零拷贝） |
| FSDP disaggregate 跨机 | 十毫秒级（all-gather 跨机 + NCCL broadcast） |
| Megatron colocate 3D-HybridEngine | 十毫秒级（同组 reshard + NVLink） |
| Megatron disaggregate 跨机 | 百毫秒~秒级（跨机 reshard 重） |

详见 [[weight sync mechanism]] §3.4 已联网核实的四路线对比。

### 4.4 logprob 一致性：两端引擎的隐藏风险

rollout（vLLM/SGLang）算的 $\log\pi_{old}$ 与 actor（FSDP/Megatron）重算的 $\log\pi_\theta$ **可能不一致**——TIM 训推不一致：
- 数值精度：vLLM 用 BF16 + 优化 kernel、trainer 用 FP32 master + BF16 forward，浮点累加顺序不同 → logp 差异累积。
- 实现差异：vLLM 的 attention kernel（PagedAttention）/RoPE/归一化与 trainer 的实现细节微差 → logp 偏移。
- MoE 路由：vLLM 推理时路由与 trainer 训练态路由不一致（[[R3 rollout routing replay]]，待核实）→ logp 失真。

不一致让 ratio $\rho=\exp(\log\pi_\theta-\log\pi_{old})$ 失真（$\theta=\theta_{old}$ 时 $\rho$ 应=1，实际偏离）→ 训练不稳/崩。故 verl/OpenRLHF 常让 **actor 重算 old_log_prob**（用 trainer 重新 forward 算 $\log\pi_{old}$，而非用 rollout 算的）做锚点，保证 ratio 一致性。详见 [[rollout train reference logprob一致性]]。这是"两端引擎组合"引入的核心副作用。


## 5. 代码示例（可选）

### 5.1 verl FSDP trainer + vLLM rollout 组合（概念，配置名以官方文档为准）

```python
# verl 的 actor_rollout worker 同时持有 trainer (FSDP) + rollout (vLLM)
# 来自 verl actor_rollout worker (见 Ray与分布式调度 §3.5)
class ActorRolloutRefWorker:
    def __init__(self, config):
        # 训练态: FSDP/Megatron (管梯度/optim/并行)
        self.actor = FSDPWorker(config.actor)          # 或 MegatronWorker
        # 推理态: vLLM/SGLang (管 KV/batching/CUDA Graph)
        self.rollout = vLLMRollout(config.rollout)       # 或 SGLangRollout

    def update_weights(self):
        """trainer 新 θ → rollout (weight sync 桥接)"""
        # 1. 从 FSDP trainer 导出全量 θ (all-gather 还原 shard)
        state_dict = self.actor.get_full_state_dict()   # FSDP all-gather
        # 2. 转 dtype + 按 rollout TP 重切 (reshard)
        rollout_weights = convert_and_reshard(
            state_dict,
            src_layout="fsdp",          # 训练态 FSDP shard
            dst_layout="vllm_tp",       # 推理态 vLLM TP
            dtype=torch.bfloat16,
        )
        # 3. load_weights 进 vLLM (colocate in-process / disaggregate NCCL)
        self.rollout.update_weights(rollout_weights)    # vLLM load_weights API
```

### 5.2 verl 配置：FSDP trainer + vLLM rollout（YAML 概念）

```yaml
# verl config: FSDP trainer + vLLM rollout 组合
actor_rollout_ref:
  model:
    path: /path/to/base_model
  actor:
    strategy: fsdp                 # ★ 训练端: FSDP (也可选 megatron)
    fsdp_size: 8                  # DP 并行度 (all-gather 还原全量)
  rollout:
    engine: vllm                  # ★ 采样端: vLLM (也可选 sglang)
    tensor_parallel_size: 8       # rollout TP (与 trainer DP 可不同, 需 reshard)
    gpu_memory_utilization: 0.5   # rollout KV cache 显存 (colocate 时与 trainer 分时)
    enable_prefix_caching: true
    enable_sleep_mode: true       # colocate sleep/wake
critic:
  strategy: fsdp                  # PPO critic; GRPO 无此段
reward:
  ...
# weight sync: update_weights 桥接 FSDP shard → vLLM TP (all-gather + reshard)
```

### 5.3 FSDP all-gather + vLLM load 权重同步片段

```python
# 概念: FSDP trainer → vLLM rollout weight sync
import torch.distributed as dist

def fsdp_to_vllm_weight_sync(actor_fsdp, vllm_engine, tp_size):
    # 1. FSDP all-gather: 每 rank 的 shard 聚成完整 θ
    full_state = {}                     # full params (CPU 或 GPU)
    # FSDP all_gather 把各 rank shard 聚成完整参数 (用 FSDP API 或 manual)
    actor_fsdp.state_dict(all_gather=True, out=full_state)

    # 2. dtype 转 + 按 vLLM TP 重切 (reshard)
    for name, param in full_state.items():
        param = param.to(torch.bfloat16)            # 训练 FP32 → 推理 BF16
        # 按 vLLM TP 切分 (训练 FSDP 是 DP 切, 推理是 TP 切, 切法不同需重切)
        shards = split_for_tp(param, tp_size)
        for rank, shard in enumerate(shards):
            vllm_engine.workers[rank].load_weight(name, shard)  # load_weights

    # 3. vLLM 重建 CUDA Graph / paged KV 元数据
    vllm_engine.after_weight_update()  # 重建 graph (若有)
```

### 5.4 OpenRLHF DeepSpeed trainer + vLLM rollout（配置项以官方文档为准）

```bash
# OpenRLHF: DeepSpeed ZeRO-3 (≈FSDP) trainer + vLLM rollout
ray job submit -- python3 -m openrlhf.cli.train.ppo_ray \
  --pretrain {MODEL} \
  --zero_stage 3 \                          # ★ 训练端: DeepSpeed ZeRO-3 (≈FSDP)
  --bf16 \
  --vllm_num_engines 4 \                    # ★ 采样端: vLLM
  --vllm_sync_backend nccl \                # weight sync 后端: NCCL broadcast
  --colocate                                # colocate (in-process) 或去此开 disaggregate
# weight sync: DeepSpeed ZeRO-3 all-gather → vLLM TP reshard → load_weights
```

### 5.5 Megatron reshard 概念（3D-HybridEngine 同组）

```python
# 概念: Megatron 训练态 (TP_t/PP_t/DP_t) → vLLM 推理态 (TP_r) reshard
def megatron_reshard_to_vllm(megatron_model, vllm_engine, tp_r):
    # 训练态: TP 切 + PP 层段 + DP 复制
    # 1. TP 还原: 每 TP rank 的列切片 all-gather 成完整层
    for layer in megatron_model.get_layers_per_pp_rank():
        full_layer = all_gather_tp_shards(layer)   # TP 还原 (NVLink)
        # 2. PP 重组: 跨 PP rank 聚层段成完整层序列 (all-to-all)
        # 3. DP 去重: 各 DP rank 同权重, 取一份
        # 4. 按 vLLM TP_r 重切
        tp_shards = split_for_tp(full_layer, tp_r)
        for rank, shard in enumerate(tp_shards):
            vllm_engine.workers[rank].load_weight(layer.name, shard)
    # 同组 NVLink reshard, 避免跨机 weight sync (3D-HybridEngine 核心)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[RL角色拓扑]]（actor/rollout 角色定义，本篇是其在引擎轴展开）、[[colocated与disaggregated部署]]（部署轴，与本篇正交——引擎组合可跑 colocate 也可 disaggregate）、[[weight sync mechanism]]（两端桥接的机制）、[[Fully Sharded Data Parallel]]/[[Megatron-LM]]（训练端）、[[continuous batching的调度实现]]/[[PagedAttention]]（采样端）。
- **下游（应用）**: [[17-RL训推一体框架]] 全章、[[RL权重同步]]、[[rollout train reference logprob一致性]]（两端引擎引入的 logp 不一致风险）、[[新模型接入]]（新模型接两端引擎需适配 weight loader/reshard）、[[Ray与分布式调度]]（verl 用 Ray 编排两端 worker）。
- **对比 / 易混**:
  - **trainer-rollout 引擎组合 vs [[训推分离]] PT/PD**：前者是 RLHF 训练内部两端引擎（trainer 训 actor、rollout 采样，weight sync 每 batch）；后者是训练↔部署分离（PT 训新版本推 PD 服务）。前者高频每 batch sync，后者低频版本升级。weight sync 机制同构但频率/场景不同。
  - **trainer-rollout 引擎组合 vs 单框架训推一体**：前者两端各自最优 + 桥接；后者一个框架既训又采（PyTorch 原生，吞吐低）。前者主流，后者性能差。
  - **FSDP trainer + vLLM rollout vs Megatron trainer + vLLM rollout**：前者 FSDP all-gather 还原全量再 TP 切（简单，中小~大模型）；后者 Megatron 3D reshard（复杂，超大模型）。组合矩阵见 §3.7。


## 7. 常见误区与易错点

> [!warning] 误区 1：rollout 用训练框架就行
> 错。训练框架无 KV cache/continuous batching/CUDA Graph，采样吞吐低一个数量级。rollout 是 RLHF 墙钟大头，低吞吐直接拖死训练。必须用推理框架（vLLM/SGLang）。

> [!warning] 误区 2：用推理框架直接训练
> 错。推理框架无 backward/autograd/optimizer/并行训练，训不了。必须用训练框架（FSDP/Megatron）。

> [!warning] 误区 3：weight sync 只是传张量
> 错。两端权重切分布局不同（FSDP shard / Megatron TP/PP/DP / vLLM TP），sync 含 **all-gather 还原 + dtype 转 + reshard 重切 + load**，通信量 = 全模型权重，是 RL 训推一体最频繁的跨引擎通信。Megatron reshard 尤复杂。

> [!warning] 误区 4：rollout 与 trainer 同 TP 就不用 reshard
> 不一定。即使 TP 度同，FSDP 是 DP 切（每 rank 持参数片）、vLLM 是 TP 切（每 rank 持层列切），切法不同仍需 all-gather + 重切。只有完全同切分布局才能直接复用（罕见）。Megatron 更需 TP/PP/DP 三维 reshard。

> [!warning] 误区 5：两端 logp 天然一致
> 错。vLLM（BF16 + 优化 kernel）与 trainer（FP32 master + BF16 forward）算的 logp 因数值精度/实现细节/MoE 路由不一致，$\theta=\theta_{old}$ 时 $\rho$ 应=1 实际偏离。需 actor 重算 old_log_prob 做锚点。详见 [[rollout train reference logprob一致性]]。

> [!warning] 误区 6：3D-HybridEngine 是跨机
> 相反。3D-HybridEngine 是 colocate 同组 reshard（actor 训推态同组 GPU，靠 NVLink all-gather + 重切），避免跨机 weight sync。是 colocate 极致，非跨机分离。详见 [[colocated与disaggregated部署]] §3.5。

> [!tip] 实践：选 trainer 后端看模型大小
> - 中小~大模型（≤30B）：FSDP trainer + vLLM rollout（简单，all-gather 还原全量，verl/OpenRLHF 默认）。
> - 超大模型（70B+）：Megatron trainer + vLLM/SGLang rollout（3D 并行 + 3D-HybridEngine reshard，verl/HybridFlow）。
> - rollout 引擎选 vLLM（通用）或 SGLang（复杂结构化生成/强 prefix caching）。
> - weight sync 后端：colocate IPC/in-process，disaggregate NCCL broadcast。


## 8. 延伸细节

### 8.1 HybridFlow 的统一抽象

HybridFlow 论文（arXiv:2409.19256，ByteDance）核心：**control flow**（RL 算法 PPO/GRPO 的步骤序列，与分布式无关）与 **distributed execution**（各 role 的 trainer/rollout 后端怎么跑）解耦。trainer 后端可插拔（FSDP/Megatron）、rollout 后端可插拔（vLLM/SGLang/sglang），weight sync 是统一接口的桥接层。这就是 verl 能同时支持"FSDP/Megatron × vLLM/SGLang"组合矩阵的根——组合矩阵是配置选择，非改代码。详见 [[RL角色拓扑]] §8.1。

### 8.2 packed tensor 打包减 launch 开销

weight sync 传大量小张量（每层 weight/bias）时，NCCL broadcast 的 per-tensor launch 开销累积。vLLM `WeightTransferEngine` 用 **packed tensor**——把多个小张量打包成大连续 buffer，一次 broadcast，再 unpack。双/三缓冲 + 独立 CUDA stream 做 pack/broadcast/unpack overlap，吞吐显著提升（实现参考 NeMo-RL）。这是两端 weight sync 的工程优化。详见 [[weight sync mechanism]] 路线一已核实内容。

### 8.3 SGLang 的 weight sync 后端

SGLang 的 weight sync 与 vLLM 思路同但实现异：支持 checkpoint engine（`sglang.srt.checkpoint_engine`，磁盘→多机并行加载，broadcast/p2p/all 三模式，冷启动加速）与 `/pull_weights`（共享 FS zstd per-tensor delta + checksum，非 colocated rollout/PD 多副本）。RLHF 热路径仍以 NCCL broadcast 为主，SGLang 在整合 ckpt-engine 加速在线 sync（issue #10464）。详见 [[weight sync mechanism]] 路线三/四。

### 8.4 新模型接入两端的适配成本

新模型接入 RL 训推一体需适配两端引擎：
- **trainer 端**：FSDP/Megatron 需模型符合其分片接口（FSDP 需 `fully_shard` 兼容、Megatron 需 TP/PP 切分点标注）。
- **rollout 端**：vLLM/SGLang 需该模型的 weight loader（HF format → vLLM 内部布局）、attention kernel（PagedAttention 适配）、CUDA Graph 支持。
- **weight sync**：trainer state_dict 字段名/布局 ↔ vLLM `load_weights` 期望，需 converter。

故新模型/新架构（如 Mamba/Diffusion LLM 等）接入两端都有适配成本，常是 RL 训推一体的工程瓶颈。详见 [[新模型接入]]。待核实各框架对新架构的支持现状。

### 8.5 MoE 模型的两端组合特殊问题

MoE（Mixture of Experts）模型在两端组合有额外问题：训练态 expert 路由与推理态（vLLM）路由可能不一致 → logp 失真（[[R3 rollout routing replay]]，待核实）。需 rollout 时记录路由、trainer 重算时 replay 路由保证一致。是 [[rollout train reference logprob一致性]] 在 MoE 的特化。verl/OpenRLHF 对 MoE 的两端适配待以最新文档为准。

### 8.6 内容来源

两端引擎组合综合自 verl 文档（`actor_rollout worker`/`update_weights`/FSDP/Megatron trainer backend + vLLM/SGLang rollout backend）、OpenRLHF 文档（`--zero_stage`/`--vllm_num_engines`/`--vllm_sync_backend`）、HybridFlow 论文 arXiv:2409.19256（统一抽象 + 3D-HybridEngine）。weight sync 路线与 packed tensor 来自 [[weight sync mechanism]] 已联网核实的四路线总结（vLLM WeightTransferEngine/SGLang checkpoint engine/SGLang /pull_weights/OpenRLHF sync_backend）。logp 一致性风险来自 [[rollout train reference logprob一致性]]。verl fit 时序与 update_weights 来自 [[Ray与分布式调度]] §3.5 已核实代码解答。截至 2026-07。

---
相关: [[trainer与rollout]] | [[RL角色拓扑]] | [[colocated与disaggregated部署]] | [[weight sync mechanism]] | [[RL权重同步]] | [[Fully Sharded Data Parallel]] | [[Megatron-LM]] | [[continuous batching的调度实现]] | [[PagedAttention]] | [[rollout train reference logprob一致性]] | [[importance sampling与off-policy correction]] | [[训推分离]] | [[stale policy problem]] | [[asynchronous training]] | [[Ray与分布式调度]] | [[rollout worker]] | [[inference worker]] | [[learner worker split]] | [[新模型接入]] | [[17-RL训推一体框架]]
