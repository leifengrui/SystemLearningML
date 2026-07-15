# HF与Megatron checkpoint转换

> **所属章节**: [[distributed checkpoint]]
> **所属模块**: [[15-训练框架源码]]
> **别名**: HF↔Megatron 转换 / weight name mapping / QKV merge/split / TP/PP recombine / 框架间 ckpt 转换 / Megatron↔HF converter
> **难度**: 高（需懂 [[Megatron distributed checkpoint]] + [[state_dict分片]] + [[ColumnParallelLinear]] + [[RowParallelLinear]] + [[checkpoint转换|ch14 通用转换器]] + HF Transformers 结构）


## 1. 一句话定义

**HF 与 Megatron checkpoint 转换** 是把 [[Megatron-LM]] 训练格式的分布式 checkpoint（`mp_rank/tp_rank/pp_rank` 多文件 + flatten param + TP/PP 切分）与 [[Hugging Face]] Transformers 推理格式的 checkpoint（单目录 `safetensors` + `config.json` + 标准 weight name，无 TP/PP 切分）**互相转换**的流程——两个核心难点是 **(1) weight name mapping**（Megatron 用 `model.language_model.encoder.layers.{i}.self_attention.query_key_value.weight`（QKV 合并）+ `dense` + `mlp.dense_h_to_4h`（gate_up 合并）等 Megatron 命名，HF 用 `transformer.h.{i}.attn.c_attn.weight`（GPT-NeoX 风格）或 `model.layers.{i}.self_attn.q_proj/v_proj/k_proj`（Llama 风格）分开 Q/K/V，需按模型结构建 name 映射表）、**(2) TP/PP recombine**（Megatron 把 QKV 在 TP 维切（ColumnParallel），HF 存合并的 QKV；Megatron 把 FFN up 在 TP 维切，HF 存合并；Megatron 在 PP 维切层（stage），HF 存全部层——转换需把 TP 维切片拼接、PP 维 stage 串接还原完整模型）。工具用 Megatron 自带 `convert.py`、社区 `transformers` 的 `convert_xxx_to_pytorch.py`、或 [[checkpoint转换|ch14 通用转换器]] 的统一接口。是训练（Megatron）→推理/发布（HF）的必经桥梁，也是大模型生态互通的关键。

> [!note] 三句话定位
> - **是什么**：Megatron 训练 ckpt（分布式多文件 + TP/PP 切分 + QKV 合并命名）↔ HF 推理 ckpt（单目录 safetensors + 无切分 + Q/K/V 分开命名）互转，核心是 weight name mapping + TP/PP recombine。
> - **为什么**：Megatron 训练高效但生态封闭（格式非标准），HF 推理/发布生态广但无训练并行——训练完需转 HF 发布/推理，HF 开源模型需转 Megatron 继续训。两者格式天差地别，需转换器。
> - **与 [[checkpoint转换|ch14 通用转换器]] 关系**：ch14 通用转换器是 HF↔FSDP2↔Megatron 三格式统一接口，本笔记是 Megatron↔HF 这条边的深度展开（weight name mapping + QKV/PP 细节）。是 ch14 的一个分支深化。


## 2. 为什么需要它（动机与背景）

### 2.1 训练与推理生态分离

- **Megatron 训练**：高效并行（TP/PP/DP/EP/CP），分布式 ckpt（`mp_rank` 多文件），但生态封闭（非 HF 标准）。
- **HF 推理/发布**：生态广（Transformers/Diffusers/社区下载），标准格式（safetensors + config），但无训练并行。

训练用 Megatron，发布/推理用 HF。训练完需转 HF 发布、HF 开源模型需转 Megatron 继续训。两者格式天差地别，需转换器。

### 2.2 weight name 不同

Megatron 与 HF 的 weight 命名不同：
- Megatron：`model.language_model.encoder.layers.{i}.self_attention.query_key_value.weight`（QKV 合并，ColumnParallel 切 TP）
- HF Llama：`model.layers.{i}.self_attn.q_proj.weight`（Q 分开，无 TP 切）

需按模型结构建 name 映射表，逐 weight 对应。

### 2.3 QKV 合并 vs 分开

Megatron 把 Q/K/V 合并成 `query_key_value`（ColumnParallel 一起切 TP，高效）。HF Llama 把 Q/K/V 分开（`q_proj`/`k_proj`/`v_proj`，适配多头/分组/多查询注意力差异）。转换需**拆 Megatron 合并的 QKV 成 HF 分开**（M→HF），或**合并 HF 分开成 Megatron 合并**（HF→M）。是转换的核心难点。

### 2.4 TP 切分需拼接

Megatron TP 把 weight 切 T 片（ColumnParallel 切列、RowParallel 切行）。HF 存完整 weight（无切分）。M→HF 需把 T 片拼接还原完整。HF→M 需把完整切 T 片。是 recombine。

### 2.5 PP 切层需串接

Megatron PP 把层切 stage（每 stage 持部分层）。HF 存全部层（无 stage）。M→HF 需把各 stage 串接还原全部层。HF→M 需把全部层按 stage 切。是 recombine。

### 2.6 flatten param 需 unflatten

Megatron flatten param 成一维 buffer。HF 存 per-tensor。转换需 unflatten（Megatron buffer → per-tensor）+ name mapping。是 [[state_dict分片]] 的 SHARDED→FULL。


## 3. 核心概念详解

### 3.1 weight name mapping

```
Megatron (Llama 风格)                HF (Llama)
attention.query_key_value.weight  <-> self_attn.q_proj.weight + k_proj + v_proj (QKV 合并->分开)
attention.dense.weight            <-> self_attn.o_proj.weight
mlp.dense_h_to_4h.weight          <-> mlp.gate_proj.weight + up_proj.weight (gate_up 合并->分开)
mlp.dense_4h_to_h.weight          <-> mlp.down_proj.weight
input_layernorm.weight            <-> input_layernorm.weight
post_attention_layernorm.weight   <-> post_attention_layernorm.weight
embedding.word_embeddings.weight  <-> model.embed_tokens.weight
output_layer.weight               <-> lm_head.weight
final_layernorm.weight            <-> model.norm.weight
```

逐 weight 对应。需按模型结构（Llama/GPT-NeoX/Qwen 等）建 mapping 表。是转换的基础。

### 3.2 QKV 合并 vs 分开

```
Megatron (ColumnParallel, QKV 合并):
  query_key_value.weight: [d, 3*d] (Q|K|V 拼接, TP 切列)
  TP=2: rank0 持 [d, 3*d/2] (Q0|K0|V0), rank1 持 [d, 3*d/2] (Q1|K1|V1)

HF (Llama, Q/K/V 分开):
  q_proj.weight: [d, d]
  k_proj.weight: [d, d] (或 [d, d_kv] for GQA/MQA)
  v_proj.weight: [d, d] (或 [d, d_kv])

M->HF 转换:
  1. 拼接 TP 片: [d, 3*d] (完整)
  2. 拆 QKV: [d, d] (Q) + [d, d] (K) + [d, d] (V)
  3. 赋给 q_proj, k_proj, v_proj
```

M→HF：拼接 TP → 拆 QKV → 赋 Q/K/V。HF→M 反向。是转换核心。

> [!warning] GQA/MQA 的 QKV 拆分更复杂
> GQA（Grouped-Query Attention）的 K/V 头数 < Q 头数。Megatron 的 `query_key_value` 把 Q（多头）+ K（少头，repeat）+ V（少头，repeat）拼接。HF 分开存 `q_proj`（多头）、`k_proj`/`v_proj`（少头，不 repeat）。转换需按 head 数拆/合，不能简单按 dim 切。是 QKV 转换的难点。

### 3.3 gate_up 合并 vs 分开

```
Megatron (ColumnParallel, gate_up 合并):
  dense_h_to_4h.weight: [d, 2*d_ffn] (gate|up 拼接, TP 切列)
HF (Llama, gate/up 分开):
  gate_proj.weight: [d, d_ffn]
  up_proj.weight: [d, d_ffn]

M->HF: 拼接 TP -> 拆 gate|up -> 赋 gate_proj, up_proj
HF->M: 反向
```

FFN 的 gate/up 合并/分开。类似 QKV。是 SwiGLU 结构的转换。

### 3.4 TP recombine（拼接）

```
Megatron TP=4 (ColumnParallel, 切列):
  rank0: W[:, 0:d/4], rank1: W[:, d/4:d/2], rank2: W[:, d/2:3d/4], rank3: W[:, 3d/4:d]
HF (无 TP, 完整):
  W: [d, d_full]

M->HF: 沿列拼接 4 片 -> [d, d_full]
```

M→HF 把 TP 片沿列（ColumnParallel）或行（RowParallel）拼接。HF→M 反向切。是 recombine。

### 3.5 PP recombine（串接）

```
Megatron PP=4 (切层):
  stage0: layers 0-19, stage1: layers 20-39, stage2: layers 40-59, stage3: layers 60-79
HF (无 PP, 全部层):
  layers 0-79

M->HF: 串接各 stage 的层 -> 0-79
```

M→HF 把各 stage 的层串接（按 layer idx 排序）。HF→M 反向按 stage 切。是 recombine。

### 3.6 转换流程（M→HF）

```
1. 读 Megatron meta (TP, PP, layer 数, 结构)
2. 每 rank 读自己 shard (torch.load)
3. unflatten param (buffer -> per-tensor)
4. 拼接 TP 片 (TP recombine, 沿列/行)
5. 串接 PP stage (PP recombine, 按 layer idx)
6. name mapping (Megatron -> HF name)
7. 拆 QKV (query_key_value -> q_proj/k_proj/v_proj)
8. 拆 gate_up (dense_h_to_4h -> gate_proj/up_proj)
9. 存 HF safetensors + config.json
```

全流程：读 shard → unflatten → 拼接 TP/PP → name map → 拆 QKV/gate_up → 存 HF。

### 3.7 转换流程（HF→M）

```
1. 读 HF safetensors + config
2. 读 Megatron 目标配置 (TP, PP)
3. name mapping (HF -> Megatron name)
4. 合 QKV (q_proj/k_proj/v_proj -> query_key_value)
5. 合 gate_up (gate_proj/up_proj -> dense_h_to_4h)
6. 切 TP (TP recombine 反向, 沿列/行切)
7. 切 PP (按 stage 切层)
8. flatten param (per-tensor -> buffer)
9. 每 rank 写自己 shard (torch.save)
```

M→HF 的逆流程。

### 3.8 转换工具

```bash
# Megatron 自带
python tools/checkpoint/loader.py --load-dir megatron_ckpt --save-dir hf_ckpt \
    --model-type llama --target-pipeline-parallel-size 1 --target-tensor-parallel-size 1

# HF Transformers 社区
python transformers/src/transformers/models/llama/convert_llama_original_to_pytorch.py

# ch14 通用转换器 (统一接口)
from universal_ckpt_converter import convert
convert(src_format='megatron', dst_format='hf', src_dir=..., dst_dir=...)
```

Megatron 自带 `tools/checkpoint/loader.py`（M↔HF）、HF 社区脚本、ch14 通用转换器（统一）。三者可选。

### 3.9 EP 的 MoE 转换

```
Megatron (EP 切 expert):
  每 rank 持部分 expert 的 weight
HF (无 EP, 全 expert):
  全部 expert

M->HF: 拼接 EP 片 -> 全 expert
HF->M: 切 EP
```

MoE 模型还需 EP recombine（expert 维）。是 EP 的转换。详见 [[grouped GEMM]] 与 [[MoE token dispatcher]]。


## 4. 数学原理 / 公式

### 4.1 QKV 拆分

Megatron `query_key_value` 权重 $W_{qkv} \in \mathbb{R}^{d \times 3d}$（Q|K|V 拼接，每 $d$ 一段）：

$$W_{qkv} = [W_Q; W_K; W_V], \quad W_Q = W_{qkv}[:, 0:d], W_K = W_{qkv}[:, d:2d], W_V = W_{qkv}[:, 2d:3d]$$

HF 分别存 $W_Q, W_K, W_V$。M→HF 按 $d$ 切 3 段。GQA 时 $W_K, W_V$ 的列数 $< d$（少头），需按 head 数调整。

### 4.2 TP 拼接

Megatron TP=T（ColumnParallel，切列）：

$$W_{\text{full}} = [W_0; W_1; \dots; W_{T-1}] \quad (\text{沿列拼接})$$

每 rank 持 $W_i \in \mathbb{R}^{d \times d/T}$（或行切 RowParallel $\mathbb{R}^{d/T \times d}$）。HF 存 $W_{\text{full}}$。M→HF 拼接 T 片。

### 4.3 PP 串接

Megatron PP=P（切层）：

$$L_{\text{full}} = [L_{\text{stage 0}}; L_{\text{stage 1}}; \dots; L_{\text{stage P-1}}] \quad (\text{按 layer idx 串接})$$

HF 存 $L_{\text{full}}$（全部层）。M→HF 串接 P stage。

### 4.4 flatten 的 unflatten

Megatron flatten：$\text{buffer} = [w_0.\text{flatten}(); w_1.\text{flatten}(); \dots]$（拼接所有 param 成一维）。unflatten：据 metadata（每 param 的 offset/shape）切回 per-tensor。是 [[state_dict分片]] 的 SHARDED→FULL。

### 4.5 显存

转换器需持完整模型（拼接 TP/PP/EP 后）。显存 $\propto N_{\text{param}}$（全模型）。大模型（671B）单进程持不下，需流式转换（逐层处理）或分块。是转换的工程难点。


## 5. 代码示例（可选）

### 5.1 Megatron → HF（概念）

```python
import torch
from safetensors.torch import save_file

# 1. 读 Megatron shards (多 rank)
shards = []
for rank in range(tp * pp):
    sd = torch.load(f'megatron_ckpt/mp_rank_{rank}/model_optimizers.ckpt', map_location='cpu')
    shards.append(sd['model'])

# 2. unflatten + 拼接 TP + 串接 PP
full_sd = recombine(shards, tp_size=tp, pp_size=pp, layer_count=L)

# 3. name mapping + 拆 QKV/gate_up
hf_sd = {}
for mname, tensor in full_sd.items():
    if 'query_key_value' in mname:
        q, k, v = split_qkv(tensor)  # 拆 3 段
        hf_name = mname.replace('self_attention.query_key_value', 'self_attn.q_proj')
        hf_sd[hf_name] = q
        hf_sd[hf_name.replace('q_proj', 'k_proj')] = k
        hf_sd[hf_name.replace('q_proj', 'v_proj')] = v
    elif 'dense_h_to_4h' in mname:
        gate, up = split_gate_up(tensor)
        hf_sd[mname.replace('mlp.dense_h_to_4h', 'mlp.gate_proj')] = gate
        hf_sd[mname.replace('mlp.dense_h_to_4h', 'mlp.up_proj')] = up
    else:
        hf_sd[name_map(mname)] = tensor

# 4. 存 HF safetensors + config
save_file(hf_sd, 'hf_ckpt/model.safetensors')
write_config('hf_ckpt/config.json', model_config)
```

### 5.2 HF → Megatron（概念）

```python
# 反向: name mapping HF->Megatron, 合 QKV/gate_up, 切 TP/PP, flatten, 每 rank 写
from safetensors.torch import load_file
hf_sd = load_file('hf_ckpt/model.safetensors')
# 合 QKV: q_proj + k_proj + v_proj -> query_key_value
# 合 gate_up: gate_proj + up_proj -> dense_h_to_4h
# 切 TP: 沿列切 T 片
# 切 PP: 按 stage 切层
# flatten: per-tensor -> buffer
# 每 rank 写自己 shard
for rank in range(tp * pp):
    torch.save({'model': shard[rank]}, f'megatron_ckpt/mp_rank_{rank}/model_optimizers.ckpt')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Megatron distributed checkpoint]]（Megatron 格式）、[[checkpoint转换|ch14 通用转换器]]（统一接口）、[[state_dict分片]]（FULL/SHARDED）、[[ColumnParallelLinear]]（TP 切列）、[[RowParallelLinear]]（TP 切行）、[[EP TP DP CP组合规则]]（TP/PP/EP 维）、[[Hugging Face]]（HF 格式）。
- **下游（应用）**: Megatron 训练→HF 发布/推理、HF 开源模型→Megatron 继续训、RLHF/SFT 的模型流转、生态模型互通。
- **对比 / 易混**:
  - **HF↔Megatron vs [[checkpoint转换|ch14 通用转换器]]**：前者是 Megatron↔HF 这条边的深度（QKV/PP 细节），后者是 HF↔FSDP2↔Megatron 三格式统一接口。前者是后者的一个分支深化。
  - **HF↔Megatron vs [[Megatron distributed checkpoint]]**：前者是格式转换（M↔HF），后者是 Megatron 内部存储格式（mp_rank 多文件）。前者是后者的对外转换层。
  - **QKV 合并 vs [[ColumnParallelLinear]]**：Megatron 把 QKV 合并成 `query_key_value` 并 ColumnParallel 切 TP（高效）。转换时拆开是还原 HF 的分开存储。是训练高效与推理标准化的折中。


## 7. 常见误区与易错点

> [!warning] 误区 1：name mapping 一对一
> 不全是一对一。QKV 合并的 `query_key_value` 一对三（q_proj/k_proj/v_proj），gate_up 一对二（gate_proj/up_proj）。需按结构建。误以为一对一会漏 weight。

> [!warning] 误区 2：QKV 简单按 dim 切 3 段
> 标准 MHA 可以（$3d$ 切 3 段），但 GQA/MQA 的 K/V 头数 < Q 头数，需按 head 数拆，不能简单按 dim。是 QKV 转换的坑。

> [!warning] 误区 3：只拼 TP 忘 PP
> Megatron 同时切 TP 和 PP。转换需拼接 TP（weight 维）+ 串接 PP（layer 维）。漏 PP 会缺层。

> [!warning] 误区 4：flatten 不 unflatten
> Megatron flatten 成一维 buffer。转换需 unflatten（据 metadata 切回 per-tensor）。直接用 buffer 会 shape 错。详见 [[state_dict分片]]。

> [!warning] 误区 5：单进程转换大模型
> 大模型（671B）完整模型单进程持不下。需流式（逐层）或分块转换。误以为单进程会 OOM。

> [!warning] 误区 6：optimizer state 也转
> M→HF 只转 weight（推理不需 opt）。HF→M 续训才需 opt（但 HF 通常无 opt，需重新初始化）。误以为全转会错。

> [!warning] 误区 7：MoE 的 EP 忘转
> MoE 模型还需 EP recombine（expert 维拼接/切分）。漏 EP 会缺 expert。是 MoE 转换的坑。


## 8. 延伸细节

### 8.1 GQA/MQA 的 QKV 拆分细节

GQA（Grouped-Query Attention）：Q 头数 $h_q$，K/V 头数 $h_{kv} < h_q$（如 $h_q=32, h_{kv}=8$）。Megatron 的 `query_key_value` 把 Q（$h_q$ 头）+ K（$h_{kv}$ 头，repeat 到 $h_q$）+ V（同 K）拼接。HF 存 `q_proj`（$h_q$ 头）、`k_proj`/`v_proj`（$h_{kv}$ 头，不 repeat）。转换需按 head 数拆/合，Megatron 的 repeat 部分需去重或展开。是 QKV 转换最复杂处。

### 8.2 流式转换

大模型转换为避免 OOM，逐层处理（读一层、转一层、写一层），不持完整模型。是工程优化。Megatron 的 `loader.py` 支持流式。

### 8.3 safetensors vs pickle

HF 用 `safetensors`（安全，无代码执行），Megatron 用 `torch.save`（pickle，可执行代码）。转换 M→HF 从 pickle 到 safetensors。是安全性的提升。

### 8.4 内容来源

HF↔Megatron 转换整理自 Megatron-LM `tools/checkpoint/loader.py` 源码、HF Transformers `convert_xxx_to_pytorch.py` 脚本、社区教程。QKV/GQA 拆分细节见 HF Llama/GPT-NeoX 模型代码。与 ch14 通用转换器的关系见 [[checkpoint转换]]。

---
相关: [[distributed checkpoint]] | [[Megatron distributed checkpoint]] | [[checkpoint转换]] | [[state_dict分片]] | [[ColumnParallelLinear]] | [[RowParallelLinear]] | [[EP TP DP CP组合规则]] | [[MoE token dispatcher]] | [[grouped GEMM]] | [[Hugging Face]] | [[Megatron Core目录与执行链路]] | [[Megatron-LM]]
