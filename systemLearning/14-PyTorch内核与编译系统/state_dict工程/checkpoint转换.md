# checkpoint转换

> **所属章节**: [[state_dict工程]]
> **所属模块**: [[14-PyTorch内核与编译系统]]
> **别名**: checkpoint 转换 / 检查点转换 / 格式互转 / HF ↔ FSDP2 ↔ Megatron / weight conversion / resharding 转换
> **难度**: 中高（需懂 [[state_dict分片]] + [[distributed checkpoint]] + [[Megatron-LM]] + HuggingFace 格式）


## 1. 一句话定义

**checkpoint 转换** 是在三种主流模型权重存储格式间**互转**的工程流程——**HuggingFace 格式**（`safetensors`/`.bin`，权重 **replicate**（完整），按 `model.layers.N.self_attn.q_proj.weight` 等 HF 命名组织，跨框架通用、是分发/推理的 lingua franca）、**FSDP2 / DCP 格式**（PyTorch 原生，per-rank 分片 + [[DTensor]] 的 placement+[[DeviceMesh]] metadata，支持跨拓扑 resharding，是 PyTorch 分布式训练的原生存档）、**Megatron 格式**（`mp_rank_XX`/`tp_rank_XX` 目录，权重按 **TP/PP 分片**，padding 到等大，命名用 `model.layers.N.attention.query_key_value.weight` 等合并 QKV 风格）。转换的核心是：**权重数据**的 resharding（replicate ↔ shard 沿某维，靠 all-gather/reduce-scatter/切片，[[DTensor]] 的 `redistribute` 或手写）+ **命名映射**（HF 的 `q_proj`/`k_proj`/`v_proj` 三权重 ↔ Megatron 的合并 `query_key_value` 一权重）+ **padding 处理**（Megatron TP 要求每 rank shard 等大，HF 无 padding 需补/裁）。中间枢纽是 **`FULL_STATE_DICT`**（rank 0 聚合完整 model，HF 认）或 **DCP metadata**（placement 层转）。工具链：`torchtune` 的 `convert_checkpoints.py`、HuggingFace `Accelerate` 的 `load_checkpoint_and_dispatch`、Megatron 的 `convert_checkpoint.py`。

> [!note] 三句话定位
> - **是什么**：HF（replicate+safetensors）↔ FSDP2（DCP 分片）↔ Megatron（TP/PP 分片+padding）三格式互转，含 resharding+命名映射+padding。
> - **为什么**：训练用 FSDP2/Megatron（分布式），分发/推理用 HF（通用），需互转。各格式命名/sharding 策略不同，转换是工程刚需。
> - **与 [[distributed checkpoint]] 关系**：DCP 的 placement metadata 是 resharding 的枢纽（HF→DCP 加 placement→FSDP2，或→Megatron TP shard）。`FULL_STATE_DICT` 是 HF 互转的中间态（聚合完整，HF 认）。


## 2. 为什么需要它（动机与背景）

### 2.1 训练与分发的格式鸿沟

LLM 训练用分布式框架（[[FSDP2]] per-param sharding、[[Megatron-LM]] TP/PP），存的是分片权重（省显存、跨拓扑）。但模型分发/推理（HuggingFace Hub、vLLM、SGLang）用 HF 格式（replicate 完整、safetensors）。训练完的 FSDP2/Megatron checkpoint 不能直接给推理引擎用，需转 HF。

### 2.2 框架间并行策略不同

FSDP2 shard weight 沿 fsdp mesh（前向 all-gather 合片用）。Megatron TP shard weight 沿 tp mesh（每 rank 算部分，后 all-reduce），且 QKV 合并、padding 等大。从 Megatron 转到 FSDP2（或反向）不仅是 unshard，还要拆 QKV、去 padding、换命名。

### 2.3 继续训练/微调的格式对接

社区预训练权重是 HF 格式（如 Llama-3 在 HF Hub）。要用自己的分布式框架（FSDP2/Megatron）继续预训练或微调，需 HF → FSDP2/Megatron 转换。微调完再转回 HF 分发。双向转换是常态。

### 2.4 切换训练框架

从 Megatron 切到 FSDP2（或反向），权重格式不同，需互转。或从 DeepSpeed [[ZeRO (DeepSpeed)]] 转 FSDP2。转换让框架可切换，不被锁定。

### 2.5 推理引擎的格式要求

[[vLLM]]、[[SGLang]] 等推理引擎认 HF 格式（safetensors），部分支持 Megatron 格式（需配置 TP）。训练用 Megatron，推理用 vLLM（HF），需 Megatron → HF 转。或推理用 Megatron 格式（vLLM 支持 `--model-format megatron`）。

### 2.6 工程复杂度

转换不是简单 copy：resharding（数据切片/聚合）、命名映射（结构不同）、padding（补/裁）、optimizer state（是否转）。需工具链自动化。手写易错（offset、命名、padding）。


## 3. 核心概念详解

### 3.1 三种格式对比

| 维度 | HuggingFace | FSDP2/DCP | Megatron |
|---|---|---|---|
| **存储** | safetensors/`.bin`，replicate 完整 | DCP：per-rank 片+metadata | `mp_rank_XX`/`tp_rank_XX` 目录，分片 |
| **sharding** | 无（完整 model） | Shard 沿 fsdp mesh（DTensor） | TP shard weight、PP shard layer |
| **命名** | `q_proj`/`k_proj`/`v_proj` 分开 | 按 `named_parameters()`（如 HF 命名） | `query_key_value` 合并、`gate_up_proj` 合并 |
| **padding** | 无 | 无 | TP 每 rank shard **padding 等大**（如 vocab 不整除 TP） |
| **optimizer state** | 通常不含（仅 model） | SHARDED 含分片 opt | 含（`optim.pt`） |
| **跨拓扑** | ✅（完整，任意加载） | ✅（DCP resharding） | ❌（固定 TP/PP，需聚合转换） |
| **通用性** | lingua franca（分发/推理） | PyTorch 训练原生 | Megatron 训练 |

### 3.2 resharding：数据层转换

权重数据从一种 sharding 转到另一种：
- **HF（replicate）→ FSDP2（shard）**：HF 加载完整 → `distribute_tensor` 沿 fsdp mesh shard → 存 DCP。或直接 FSDP2 加载 HF 完整（rank 各取自己片）。
- **FSDP2（shard）→ HF（replicate）**：切 `FULL_STATE_DICT` → rank 0 all-gather 聚合完整 → 存 safetensors。
- **HF → Megatron（TP shard）**：HF 加载完整 → 沿 tp mesh 切（QKV 合并 + padding）→ 存 `tp_rank_XX`。
- **Megatron → HF**：聚合所有 TP 片（去 padding、拆 QKV）→ 存 safetensors。

resharding 靠 [[DTensor]] 的 `redistribute`（FSDP2）或手写切片/聚合（Megatron）。

### 3.3 命名映射：结构层转换

HF 与 Megatron 的层命名/结构不同：

| 组件 | HuggingFace 命名 | Megatron 命名 |
|---|---|---|
| Attention Q | `model.layers.N.self_attn.q_proj.weight` | `model.layers.N.attention.query_key_value.weight`（QKV 合并） |
| Attention K | `model.layers.N.self_attn.k_proj.weight` | （合并在上面） |
| Attention V | `model.layers.N.self_attn.v_proj.weight` | （合并在上面） |
| Attention O | `model.layers.N.self_attn.o_proj.weight` | `model.layers.N.attention.dense.weight` |
| MLP gate/up | `model.layers.N.mlp.gate_proj.weight` / `up_proj.weight` | `model.layers.N.mlp.gate_up_proj.weight`（合并） |
| MLP down | `model.layers.N.mlp.down_proj.weight` | `model.layers.N.mlp.dense_4h_to_h.weight` |
| Embedding | `model.embed_tokens.weight` | `model.embedding.word_embeddings.weight` |
| LM head | `lm_head.weight` | `model.language_model.output_layer.weight`（或 tied） |

转换需映射表。HF 的 QKV 三权重 → Megatron 的一权重（concat 后按 TP 切）。反向拆开。命名映射是转换的易错点。

### 3.4 padding：TP 的等大要求

Megatron TP 要求每 rank 的 shard **等大**（NCCL all-reduce 需等量）。若某维不整除 TP 数（如 vocab=32000, TP=8，每 rank 4000，整除；但 vocab=32001 不整除），需 **padding** 补到整除（如补到 32008，每 rank 4001）。HF 无 padding。转换：
- **HF → Megatron**：合并 QKV → 按 TP 切 → 若不整除 padding。
- **Megatron → HF**：聚合所有 TP 片 → **去 padding**（裁回原 vocab）。

padding 是 Megatron 特有，FSDP2 的 DTensor 无强制等大（per-param 各自 shard，但通信原语仍需等量，实践中也 padding）。转换需处理 padding 边界。

### 3.5 QKV 合并/拆分

Megatron 把 QKV 合并成一个 `query_key_value` 权重（按 `[Q; K; V]` concat 后按 TP 列切）。HF 分开存 `q_proj`/`k_proj`/`v_proj`。转换：
- **HF → Megatron**：concat(`q`, `k`, `v`) → 按 TP 列切（每 rank 持 `[Q_i; K_i; V_i]`）。
- **Megatron → HF**：聚合所有 TP 片 → split 回 `q`/`k`/`v`。

注意 Q/K 的 head 数可能不等（GQA，K/V head 少），合并/拆分需按 head 切，不能简单 concat。GQA 的转换更复杂（K/V 的 TP 切分按 group）。

### 3.6 转换枢纽：FULL_STATE_DICT 与 DCP

两条转换路径：
- **经 `FULL_STATE_DICT`**：分布式（FSDP2/Megatron）→ rank 0 聚合完整 model → 存 → HF 加载/保存。简单（HF 认完整），但 rank 0 内存爆（大模型不行）。适合小模型或单次转换。
- **经 DCP metadata**：HF 完整 → 加 placement metadata 转 DCP → FSDP2 加载（resharding）。或 → Megatron TP shard。省内存（per-rank），但需工具支持 DC↔HF 转换。大模型用。

### 3.7 optimizer state 的转换

checkpoint 含 optimizer state（Adam 的 $m, v$）时，转换更复杂：
- **HF**：通常不含 optimizer state（仅 model 权重，供推理）。
- **FSDP2**：`SHARDED_STATE_DICT` 含分片 optimizer state。
- **Megatron**：`optim.pt` 含 TP/PP 分片 optimizer state。

HF ↔ FSDP2/Megatron 转换若要带 optimizer（继续训练），需转 optimizer state 的 sharding。若仅转 model（推理/重新初始化 optimizer），丢弃 optimizer state。工具链（torchtune）可选是否转 optimizer。

### 3.8 工具链

- **`torchtune`** 的 `convert_checkpoints.py`：HF ↔ FSDP2/Meta（PyTorch 原生），支持 `--checkpoint_format` 选 hf/dcp/torchsnapshot。
- **HuggingFace `Accelerate`** 的 `load_checkpoint_and_dispatch`：HF → big model inference（自动 shard 加载，非训练格式转）。
- **Megatron `tools/checkpoint/loader.py` / `convert_checkpoint.py`**：HF ↔ Megatron TP/PP，处理 QKV 合并、padding、命名映射。
- **`torchsnapshot`**：PyTorch 的另一 checkpoint 格式（与 DCP 类似），torchtune 用。

### 3.9 转换的精度

转换是数据搬运（切片/聚合/concat），不改数值（fp16/bf16/fp32 不变）。但 padding 的补零/去零、concat 顺序需严格一致，否则数值错。转换后应校验（如跑 forward 对比 logits）。见 [[训推不一致]] 的数值一致性。

### 3.10 跨拓扑（不同 TP/PP）

Megatron checkpoint 的 TP/PP 数固定（如存时 TP=8）。转 HF 后再转 Megatron 不同 TP（如 TP=4）需 re-shard（聚合完整再按新 TP 切）。DCP 的 resharding 自动（任意 mesh），Megatron 需经 HF 完整态中转。


## 4. 数学原理 / 公式

### 4.1 resharding 的通信

HF→FSDP2 shard：每 rank 取自己片（无通信，HF 完整在每 rank，各取片）。FSDP2→HF：rank 0 all-gather 聚合（通信 $\propto N$）。Megatron→HF：聚合 TP 片（$\propto N$）。HF→Megatron：每 rank 切片（无通信，HF 完整在每 rank）。

### 4.2 QKV 合并的维度

设 Q/K/V 各 $[d_{\text{model}}, d_{\text{model}}]$（MHA）或 V/K 少 head（GQA）。Megatron 合并 `query_key_value` = $[d_{\text{model}}, d_{\text{model}} \cdot 3]$（MHA，QKV concat）或 $[d_{\text{model}}, d_Q + d_K + d_V]$（GQA，不等）。按 TP 列切：每 rank 持 $[d_{\text{model}}, (d_Q+d_K+d_V)/T]$。GQA 的 $d_K \neq d_Q$ 需按 head group 切，不能简单等分。是转换的复杂点。

### 4.3 padding 的边界

vocab $V$，TP 数 $T$。若 $V \bmod T \neq 0$，padding 到 $V' = \lceil V/T \rceil \cdot T$。每 rank shard $V'/T$。转换去 padding 裁回 $V$（去掉补的 $V'-V$ 行）。padding 在 vocab 维（embedding/lm_head）或 FFN hidden 维（若不整除）。

### 4.4 内存

FULL 路径：rank 0 聚合完整 $N$，内存 $N$。DCP 路径：per-rank $N/d$，省。大模型（如 70B = 140GB fp16）FULL 装不下，用 DCP。


## 5. 代码示例（可选）

### 5.1 FSDP2 → HF（经 FULL_STATE_DICT）

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

model = FSDP(MyModel().cuda(), ...)  # FSDP2 shard
# 切 FULL 聚合完整
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    sd = model.state_dict()  # rank 0 完整, 其他空
    if dist.get_rank() == 0:
        # 存 HF safetensors
        from safetensors.torch import save_file
        save_file(sd, "model.safetensors")
        # 或 model.save_pretrained("hf_ckpt")
```

### 5.2 HF → FSDP2（加载 HF 完整，shard 存）

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("hf_ckpt")  # HF 完整
# 转 FSDP2: shard param 沿 fsdp mesh
mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("fsdp",))
for p in model.parameters():
    p.data = distribute_tensor(p.data, mesh, [Shard(0)])
model = FSDP(model, mesh=mesh)
# 存 SHARDED (DCP)
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    DCP.save(model.state_dict(), writer)
```

### 5.3 HF ↔ Megatron（用 Megatron 工具）

```bash
# HF -> Megatron TP/PP
python tools/checkpoint/loader.py \
    --load-dir hf_ckpt \
    --save-dir megatron_ckpt \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1
# 内部: 加载 HF -> 合并 QKV -> padding -> 按 TP 切 -> 存 tp_rank_XX

# Megatron -> HF
python tools/checkpoint/loader.py \
    --load-dir megatron_ckpt \
    --save-dir hf_ckpt \
    --model-type GPTModel
# 内部: 聚合 TP 片 -> 去 padding -> 拆 QKV -> 存 safetensors
```

### 5.4 torchtune 转换

```bash
# HF <-> DCP/torchsnapshot (FSDP2 友好)
tune run convert_checkpoints \
    --checkpoint-format torchsnapshot \
    --hf-ckpt-path meta-llama/Llama-3-8B \
    --output-path ckpt
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[state_dict分片]]（FULL/SHARDED 格式）、[[distributed checkpoint]]（DCP metadata + resharding）、[[DTensor]]（placement + redistribute）、[[Megatron-LM]]（Megatron 格式 + QKV 合并/padding）、[[state_dict与load_state_dict]]（state_dict 抽象）。
- **下游（应用）**: [[vLLM]]/[[SGLang]]（推理引擎认 HF）、HuggingFace Hub（分发）、跨框架训练切换、继续预训练/微调、[[checkpointing策略]]（转换是策略的一环）。
- **对比 / 易混**:
  - **checkpoint 转换 vs [[distributed checkpoint]]**：DCP 是 FSDP2 的 checkpoint 机制（per-rank+metadata+resharding，**单框架内**）；checkpoint 转换是**跨框架/格式**互转（HF↔FSDP2↔Megatron）。DCP 是转换的枢纽之一（metadata 层转），但转换还含命名映射/padding（DCP 不涉及）。
  - **checkpoint 转换 vs [[checkpointing策略]]**：前者是格式互转（数据结构层），后者是何时存/async/incremental（策略层）。转换是策略执行后的一步（存完转 HF 分发）。
  - **FULL 路径 vs DCP 路径**：FULL 经 `FULL_STATE_DICT` 聚合完整（简单爆内存），DCP 经 metadata resharding（省内存复杂）。大模型用 DCP，小模型或单次用 FULL。
  - **checkpoint 转换 vs [[resharding]]**：resharding 是 DTensor/DCP 内的 placement 转换（单框架跨拓扑），checkpoint 转换是跨框架格式互转（含 resharding + 命名 + padding）。resharding 是转换的数据层子步骤。


## 7. 常见误区与易错点

> [!warning] 误区 1：转换是简单 copy
> 转换含 resharding（切片/聚合）、命名映射（QKV 合并/拆）、padding（补/裁）、optimizer state（是否转）。非简单文件 copy。手写易错，用工具链。

> [!warning] 误区 2：忽视 QKV 合并顺序
> Megatron 的 `query_key_value` 是 `[Q; K; V]` concat。拆分时顺序必须 Q 在前、K 中、V 后（或按 Megatron 约定）。顺序错则权重乱（数值全错）。GQA 的 K/V head 少，需按 head group 切，不能等分。

> [!warning] 误区 3：忘记去 padding
> Megatron TP 可能 padding（补到整除）。转 HF 需去 padding（裁回原 vocab/hidden）。忘去则 HF 模型维度错（如 vocab 32008 vs 32000），forward 报错或 logits 多 padding 行。

> [!warning] 误区 4：GQA 的 TP 切分用 MHA 逻辑
> GQA（K/V head 少）的 TP 切分按 head group（每 rank 持整数 group），不是简单等分列。MHA 的 QKV 等大可等分，GQA 不行。转换 GQA 模型（如 Llama-3）需 GQA-aware 工具。

> [!warning] 误区 5：FULL 转换大模型 OOM
> FULL_STATE_DICT 路径 rank 0 聚合完整 model，70B 装不下。大模型用 DCP 路径（per-rank 分片，省内存）。误用 FULL 存 70B 会 OOM。

> [!warning] 误区 6：转换后不校验
> 转换是数据搬运，数值应不变。但 padding/concat 顺序/QKV 拆分易错。转换后应跑 forward 对比 logits（与原 checkpoint）确保一致。见 [[训推不一致]]。

> [!warning] 误区 7：optimizer state 默认转
> HF 格式通常不含 optimizer state（仅 model）。FSDP2/Megatron 含。转 HF 时 optimizer state 丢弃（重新初始化）。若继续训练需保留 optimizer，用 DCP/Megatron 格式互转（含 opt），或工具选 `--save-optimizer`。


## 8. 延伸细节

### 8.1 tied embedding 的处理

若 model 的 `lm_head` 与 `embed_tokens` tied（共享权重，如 Llama），checkpoint 只存一份。转换时注意 tied 关系（HF 用 `tie_word_embeddings=True`，Megatron 用 `share_embeddings_and_output_weights`）。转换工具需识别 tied，避免重复存/错灌。

### 8.2 跨精度转换

转换时精度不变（fp16/bf16/fp32 各自转）。但若要跨精度（如 fp32 训练 → bf16 推理），转换工具可选 `--dtype`（如 `load_checkpoint` 的 `--target-dtype bf16`）。是数据类型转换（cast），与 resharding 独立。

### 8.3 LayerNorm/RMSNorm 的处理

Norm 权重（gamma/beta）通常 replicate（不 shard，因小）。HF 与 Megatron 命名不同（`input_layernorm` vs `pre_attention_layernorm`）。转换需映射。Norm 无 padding（小，replicate）。

### 8.4 内容来源

checkpoint 转换整理自 HuggingFace `safetensors` 文档、Megatron-LM `tools/checkpoint/loader.py` 源码、torchtune `convert_checkpoints.py` 源码、PyTorch DCP 文档。三种格式的对比与命名映射见 Megatron/HF 模型实现。QKV 合并/padding 见 [[Megatron-LM]] 的 TP 实现。

---
相关: [[state_dict工程]] | [[state_dict分片]] | [[distributed checkpoint]] | [[DTensor]] | [[DeviceMesh]] | [[Megatron-LM]] | [[FSDP2]] | [[state_dict与load_state_dict]] | [[checkpointing策略]] | [[ZeRO (DeepSpeed)]] | [[vLLM]] | [[SGLang]] | [[训推不一致]] | [[AllGather]] | [[reduce-scatter]] | [[Fully Sharded Data Parallel]]
