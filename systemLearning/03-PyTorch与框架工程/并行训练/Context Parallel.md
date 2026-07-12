# Context Parallel

> **所属章节**: [[并行训练]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **别名**: Context Parallelism / CP / 序列并行（注意与 Sequence Parallelism 区分）/ Ring Attention / DeepSpeed Ulysses
> **难度**: 中高（需懂 [[自注意力]] 的 $\mathcal{O}(S^2)$ 复杂度、[[KV cache]]、[[all-reduce]]/AllGather/点对点通信、3D 并行基础）
> **资料时效**: 2026-07 核对 NVIDIA Megatron Core 官方文档与 Dynamic-CP 博客

## 1. 一句话定义

**Context Parallelism（CP，上下文并行 / 序列并行）** 是把**输入序列的长度维度 $S$** 切成 $C$ 份分给 $C$ 张 GPU，每卡只持有 $S/C$ 个 token，从而把注意力 $\mathcal{O}(S^2)$ 的算力与 $\mathcal{O}(S)$ 的激活显存**按 $C$ 线性摊薄**的并行范式。与 [[Data Parallel|DP]]（切 batch）、[[Tensor Parallel|TP]]（切层内权重）、[[Pipeline Parallel|PP]]（切层间）正交，是**长序列训练（8K+ token，乃至百万级上下文）** 的标配。关键技术是**跨序列块注意力**：各卡只有自己那一段 token，但要算"本段 token 对全序列 token"的注意力，于是用 **Ring Attention**（环形 P2P 传 KV 块）或 **DeepSpeed Ulysses**（all-to-all 重排 head）让每卡都能访问到完整的 K/V。它是 [[Megatron-LM]] Core 一等公民（`--context-parallel-size`），在 LLaMA-3 70B/405B、DeepSeek-V3 等长上下文模型训练里是第 4~5 维并行。

> [!note] CP vs Sequence Parallelism（易混，务必分清）
> 名字都像"序列并行"，但切的位置不同：
> | 概念 | 切什么 | 是否切注意力计算 | 通信 | 解决什么 |
> |---|---|---|---|---|
> | **CP（本笔记）** | 序列维度贯穿全程 | **是**，注意力也切，跨块用 ring/all-to-all 算 | P2P 传 KV（ring）或 all-to-all | 长序列的算力 $\mathcal{O}(S^2)$ + 激活 $\mathcal{O}(S)$ |
> | **Sequence Parallelism（SP）** | 仅 LayerNorm/Dropout 阶段切序列，注意力前 AllGather 回完整序列 | **否**，注意力仍用完整序列 | AllGather / reduce-scatter | 省 LayerNorm 阶段的激活显存，配合 TP 用 |
>
> 一句话：**SP 只在非注意力处切序列省显存，CP 连注意力一起切降算力+显存**。CP 是真正的"长序列并行"，SP 是 TP 的显存优化伴生技术。Megatron 里两者可叠加（`--sequence-parallel` + `--context-parallel-size`）。

## 2. 为什么需要它（动机与背景）

长上下文是 LLM 的核心能力演进方向（4K → 8K → 32K → 128K → 1M+）。但注意力的**算力与序列长度平方成正比**、**激活显存与序列长度线性成正比**，单卡很快装不下也算不动：

1. **算力墙**：自注意力 $\text{FLOPs}\propto S^2\cdot d$。$S$ 从 4K→128K，算力翻 1024 倍。一个 70B 模型训 128K 上下文，单步注意力算力是 4K 的千倍量级，单卡算到天荒地老。
2. **显存墙**：前向激活里 attention scores 形状 $\propto B\cdot H\cdot S^2$（即使 FlashAttention 不物化 $S^2$ 矩阵，KV 与中间激活仍 $\propto S$）。$S=128\text{K}$ 时单卡激活轻松上百 GB，80GB 卡直接 OOM。
3. **DP/TP/PP 都救不了长序列**：
   - **DP** 切 batch，不切序列长度——单条序列再长也得整条放一卡；
   - **TP** 切权重（head/hidden），单条序列仍完整在一卡上，注意力 $\mathcal{O}(S^2)$ 没降；
   - **PP** 切层，单层内序列仍完整，注意力算力/显存没动。
   - 三者都把序列维度 $S$ 当成不可分割的整体，长 $S$ 直接打穿单卡。
4. **CP 的切入**：既然墙在序列长度，就**正交地切 $S$**——把序列分给 $C$ 卡，每卡只算 $S/C$ 个 token 的本地注意力，再通过通信补上"跨块"注意力。算力 $\mathcal{O}(S^2/C)$、激活 $\mathcal{O}(S/C)$，线性摊薄。
5. **与现有 3D 正交**：CP 切的是 DP/TP/PP 都没切的第 4 个维度（序列），可以叠加进 4D/5D 并行（`TP×PP×CP×DP(+EP)`），总卡数 = 各维乘积。

一句话：**CP 是为长上下文而生的第 4 维并行，专治"序列太长单卡装不下/算不动"。**

## 3. 核心概念详解

### 3.1 切什么：序列维度 $S$

输入 $X\in\mathbb{R}^{B\times S\times d}$（batch $B$、序列长 $S$、隐藏 $d$）。CP=$C$ 时，沿 $S$ 切成 $C$ 段，第 $r$ 卡持有：

$$X_r = X[:,\, r\cdot S/C : (r+1)\cdot S/C, :]\in\mathbb{R}^{B\times (S/C)\times d}$$

每卡的 Q/K/V 也只有自己那 $S/C$ 个 token。**问题**：注意力 $A=\text{softmax}(QK^\top)V$ 要求每个 query token 看到所有 key token，但 key 现在散在 $C$ 卡上。所以 CP 的核心难题就是**跨块注意力**——让每卡只拿本地 Q，却能对所有 K/V 算注意力。

### 3.2 两条主流实现路线

跨块注意力有两种工业实现，理解它们的差异是理解 CP 的关键：

#### 路线 A：Ring Attention（环形 KV 传递）

- **思路**：每卡持有本地 Q + 本地 KV 块。把 $C$ 张卡排成**环**，KV 块沿环**P2P send/recv** 一圈一卡地传；每收到一块 KV，就用本地 Q 对这块 KV 做一次 blockwise attention（FlashAttention 的逐块累加），累加到输出。
- **通信**：$C-1$ 次 P2P 传 KV 块（每块 $\propto B\cdot H\cdot (S/C)\cdot d$），可和"对当前 KV 块的注意力计算"**重叠**（传下一块的同时算上一块）。
- **代表**：Megatron-LM 的 CP（`--cp-comm-type p2p`）、Ring Attention with Inter-Sequence Attention（Liu et al. 2023）。
- **优点**：通信量与 head 数无关，**适合 head 少/序列极长**；P2P 通信在 IB 上可铺开。
- **代价**：通信步数 $\mathcal{O}(C)$，$C$ 大时环长、延迟累积；需精细 overlap 调度。

#### 路线 B：DeepSpeed Ulysses（all-to-all 重排 head）

- **思路**：先各卡持有 $S/C$ token 的完整 Q/K/V（所有 head）。做一次 **all-to-all**：把"切序列"重排成"切 head"——每卡拿到全序列但只一部分 head。此时每卡有完整 $S$ 的部分 head，可直接用本地算完整序列注意力（head 间独立）。算完再 all-to-all 回"切序列"。
- **通信**：2 次 all-to-all（前向各一次，反向对称），通信量 $\propto B\cdot S\cdot d$（与 $C$ 无关，只与序列总量有关）。
- **代表**：DeepSpeed Ulysses、YaRN 长上下文训练栈。
- **优点**：通信步数 $\mathcal{O}(1)$（就两次 all-toall），**适合 head 多**（head 越多越能摊 all-toall）；调度简单。
- **代价**：要求 $H$ 能被 $C$ 整除；head 少时 all-toall 切不动、退化。

> [!tip] 选哪条路
> - **head 多（如 GPT-3 175B 有 96 head）、$C$ 不太大** → Ulysses，all-toall 通信更省；
> - **head 少、序列极长、跨节点多** → Ring Attention，P2P 可 overlap 且不卡 head 数；
> - Megatron-LM 默认 Ring 风格（`p2p`），也可配 `aflash`；DeepSpeed 用 Ulysses。两者都是"CP"这个范式下的具体实现，**切序列**这一思想一致。

### 3.3 CP 与其它维度的正交组合

CP 是第 4 维，rank 网格扩成 $(D, T, P, C)$：

- **CP 域**：同 $(D,T,P)$ 坐标、沿 $C$ 维度的 $C$ 张卡，构成一个环（Ring Attention）或 all-toall 组（Ulysses）；
- 通信域与 TP/PP/DP 隔离（不同 process group），互不干扰；
- 总 GPU = $D\cdot T\cdot P\cdot C$（加 EP 还 $\cdot E$）。

> [!note] CP 域放节点内还是跨节点？
> CP 通信量中等（$\propto$ 序列总量，但分块传），且**关键路径可 overlap**（Ring 边传边算）。所以 CP **可以跨节点走 IB**，不像 TP 那样必须锁死节点内 NVLink。Megatron 推荐配置里 LLaMA-3 70B 用 `CP=2`、405B 用 `CP=2`，常跨节点。但 $C$ 太大时环长，延迟累积，需 overlap 兜底。

### 3.4 通信量与算力摊薄

| 项 | 单卡（无 CP） | CP=$C$（Ring） | CP=$C$（Ulysses） |
|---|---|---|---|
| 注意力算力 | $\mathcal{O}(S^2\cdot d)$ | $\mathcal{O}(S^2\cdot d/C)$（本地块 $\frac{S}{C}$ 的 $\mathcal{O}((S/C)^2)\times C$ 块） | $\mathcal{O}(S^2\cdot d/C)$（部分 head 算全序列） |
| 激活显存（KV） | $\mathcal{O}(S\cdot d)$ | $\mathcal{O}(S\cdot d/C)$ | $\mathcal{O}(S\cdot d/C)$（all-toall 后瞬时翻倍，需缓冲） |
| 通信/step | 0 | $\mathcal{O}(C-1)$ 次 P2P，每块 $\propto B H (S/C) d$，可 overlap | 2 次 all-toall，每次 $\propto B S d$ |
| 通信是否关键路径 | — | 可与计算 overlap（非完全阻塞） | 阻塞（all-toall 同步点） |

要点：**算力与显存都按 $C$ 线性降**，代价是引入通信。Ring 把通信藏进计算，Ulysses 用更少步数但同步点更硬。

### 3.5 位置编码与因果性的跨块处理

CP 把序列切开后，两个细节必须对齐，否则数值就错：

1. **位置编码**（[[RoPE]] 等）：RoPE 是按 token 的**绝对位置**旋转的。切块后每卡的 token 位置不再是 $0..S/C$，而是 $r\cdot S/C .. (r+1)\cdot S/C$。每卡必须用**全局位置**算 RoPE，不能按本地序号。Megatron 的 CP 会把每卡的 position offset 传给 attention kernel。
2. **因果掩码（causal mask）**：[[causal mask]] 要求 query $i$ 只看 key $j\le i$。切块后，第 $r$ 卡的 query 块对第 $r'<r$ 的 key 块是**全可见**（都更早），对 $r'>r$ 的 key 块是**全不可见**，对 $r'=r$ 的本地块是**下三角**。Ring Attention 在累加各 KV 块时要按 $r$ vs $r'$ 关系加掩码：本地块用 causal，左侧块全 1，右侧块全 0。漏了这步会导致"未来 token 泄漏"，训练崩。

### 3.6 Dynamic CP（2026 新进展，重点）

固定 CP 在**变长序列**数据上低效——这是 2026 年 NVIDIA 的核心改进点（[Dynamic CP 博客](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)）：

- **问题**：真实数据（代码、多模态、对话）序列长度长尾分布。若按"batch 里最长序列"定 CP size，短序列也被迫切片，引入**本不需要的 CP 通信**；且不同 DP rank 的算力不均（attention $\propto S^2$），快的等慢的，DP 同步空泡 + PP 气泡放大。
- **解法**：**每个 microbatch 动态选 CP size**。Megatron Core 在初始化时预建一组 CP group（size 从 1 到 $D\cdot C$，2 的幂），运行时由 solver 按 packing 结果挑合适的 group。
- **为什么改 CP 不改 TP/PP**：改 TP/PP 要重新分权重/重构 pipeline 图，开销大；改 CP 只需重切序列块、重组 CP 通信组，开销最小。
- **配套**：THD packing 布局（变长序列拼包）、`PackedSeqParams` 携带动态 `cp_size`/`cp_group`、按有效 token 算 loss（`loss = sum_loss / num_valid_tokens`）、FLOPs 按实际序列长算（之前按 max_seqlen 高估）。
- **收益**：Llama-13B、GBS=2048、PP=8、CP=8，GitHub 数据集 **1.48×**、CommonCrawl **1.25×**；千卡级工业环境**端到端 +35%**。

> [!note] 这是 CP 的"自适应并行度"趋势
> 固定并行度在变长/异构负载上必然浪费。Dynamic CP 是"按数据动态调并行度"的代表（同类思路有 ByteScale、WLB-LLM）。未来 TP/EP 也可能走向动态化（如 NVIDIA 同期 Nonuniform TP）。

## 4. 数学原理 / 公式

### 4.1 注意力复杂度与 CP 摊薄

标准注意力（单卡，序列长 $S$）：

$$
\text{FLOPs}_{\text{attn}} = \mathcal{O}(B\cdot H\cdot S^2\cdot d_{\text{head}}),\quad \text{mem}_{\text{KV}}=\mathcal{O}(B\cdot H\cdot S\cdot d_{\text{head}})
$$

CP=$C$ 后，每卡序列长 $S/C$，每卡算 $C$ 个 query-block × 1 个 key-block 的注意力（Ring）或部分 head 的全序列注意力（Ulysses），总单卡算力：

$$
\text{FLOPs}_{\text{attn per GPU}} = \mathcal{O}\!\left(\frac{B\cdot H\cdot S^2\cdot d_{\text{head}}}{C}\right)
$$

——**线性降**。KV 显存同理 $\mathcal{O}(B H S d/C)$。

### 4.2 Ring Attention 的分块累加

第 $r$ 卡对收到的第 $k$ 个 KV 块（来自 rank $(r-k)\bmod C$）做 blockwise attention，用 FlashAttention 的在线 softmax 累加：

$$
O_r = \sum_{k=0}^{C-1} \text{AttnBlock}(Q_r,\; KV_{(r-k)\bmod C},\; \text{mask}_{r,(r-k)})
$$

其中 $\text{mask}_{r,r'}$ 按因果性：$r'<r$ 全 1、$r'=r$ 下三角、$r'>r$ 全 0。在线累加需维护 running max $m$ 与 running sum $l$，每块更新：

$$
m_{\text{new}}=\max(m_{\text{old}}, m_{\text{block}}),\quad l_{\text{new}}=l_{\text{old}}e^{m_{\text{old}}-m_{\text{new}}}+l_{\text{block}}e^{m_{\text{block}}-m_{\text{new}}}
$$

这是 FlashAttention 能"逐块累加不物化 $S^2$ 矩阵"的数学基础，Ring Attention 借它实现"逐块传 KV、逐块累加"。

### 4.3 Ulysses 的 all-to-all 重排

all-toall 把 $[B, S/C, H, d]$（切序列）重排成 $[B, S, H/C, d]$（切 head）：

$$
X_{\text{seq-shard}}\xrightarrow{\text{all-to-all}}X_{\text{head-shard}}
$$

重排后每卡有**全序列 $S$** 但只 $H/C$ 个 head，head 间独立，直接算完整序列注意力（无跨块问题）。算完再 all-toall 回切序列。通信量 2 次 $\propto B S d$。

### 4.4 总并行度

$$
N = D\cdot T\cdot P\cdot C\quad(\text{加 EP 还 }\cdot E)
$$

Megatron 官方公式：`Total GPUs = TP × PP × CP × EP × DP`。

## 5. 代码示例

```python
import torch
import torch.distributed as dist

# ===== 概念：CP 把序列切成 C 段，每卡持 S/C token =====
def cp_shard_sequence(x, cp_rank, cp_size):
    """
    x: (B, S, d) 完整序列(只在概念上存在,实际每卡只持有自己那段)
    返回本卡持有的 (B, S/C, d) 子段。
    实际工程里数据加载器直接给每卡喂对应的子段,不会先有完整 x。
    """
    S = x.size(1)
    assert S % cp_size == 0
    chunk = S // cp_size
    return x[:, cp_rank*chunk:(cp_rank+1)*chunk, :].contiguous()

# ===== Ring Attention 的通信骨架(伪代码,展示 ring KV 传递 + overlap) =====
def ring_attention_forward(q_local, k_local, v_local, cp_group, cp_rank, cp_size):
    """
    q_local: (B, H, S/C, d)  本卡 query(始终本地)
    k_local/v_local: (B, H, S/C, d)  本卡 KV(会沿环传走,接收别人的)
    实际实现是 fused kernel(FlashAttention/Transformer Engine),这里只示通信流。
    """
    out = torch.zeros_like(q_local)
    # 在线 softmax 的 running 统计量
    run_max = torch.full((q_local.shape[0], q_local.shape[1], q_local.shape[2]), -1e9, device=q_local.device)
    run_sum = torch.zeros((q_local.shape[0], q_local.shape[1], q_local.shape[2]), device=q_local.device)

    k_cur, v_cur = k_local, v_local
    for step in range(cp_size):
        # 当前 KV 块来自哪个 rank → 决定 causal mask 类型
        kv_owner = (cp_rank - step) % cp_size
        if kv_owner < cp_rank:
            mask_type = "full"      # 左侧块,全可见
        elif kv_owner == cp_rank:
            mask_type = "causal"    # 本地块,下三角
        else:
            mask_type = "zero"      # 右侧块,全不可见(因果)

        # 1) 用本地 q 对当前 k_cur/v_cur 做一块 attention,在线累加
        out, run_max, run_sum = block_attention_accumulate(
            q_local, k_cur, v_cur, mask_type, out, run_max, run_sum)

        # 2) 异步把当前 KV 块发给下一卡,同时从上一卡收下一块(overlap)
        k_next = torch.empty_like(k_cur)
        v_next = torch.empty_like(v_cur)
        req_k = dist.P2POp(dist.isend, k_cur, dst=(cp_rank+1)%cp_size, group=cp_group)
        req_v = dist.P2POp(dist.irecv, k_next, src=(cp_rank-1)%cp_size, group=cp_group)
        # ... 实际用 batched_isend_irecv 一次性调度 send/recv 对 ...
        dist.batch_isend_irecv([req_k, req_v]).wait()
        k_cur, v_cur = k_next, v_next
    # 归一化
    out = out / run_sum.unsqueeze(-1)
    return out

# ===== 配置估算:长上下文训练为何要 CP =====
S = 131072            # 128K 上下文
d = 8192              # hidden
H = 64                # head
B = 1
# 单卡(无 CP): attention FLOPs ~ B*H*S^2*d_head
d_head = d // H
flops_single = B * H * S**2 * d_head
print(f"单卡 attn FLOPs(128K): {flops_single/1e12:.1f} TFLOPs/层  ← 单卡算不动")
# CP=8
C = 8
print(f"CP=8 每卡: {flops_single/C/1e12:.1f} TFLOPs/层  ← 线性降 8 倍")
# KV 显存
kv_mem_single = B * H * S * d * 2 * 2 / 1e9  # 2(K&V) * 2(fp16)
print(f"单卡 KV 显存: {kv_mem_single:.1f} GB  ← 80GB 卡单层就吃紧")
print(f"CP=8 每卡 KV: {kv_mem_single/C:.1f} GB  ← 线性降 8 倍")
```

> [!tip] 实际怎么开（Megatron-LM）
> ```bash
> torchrun --nproc_per_node=8 pretrain_gpt.py \
>     --tensor-model-parallel-size 4 \
>     --pipeline-model-parallel-size 4 \
>     --context-parallel-size 2 \
>     --seq-length 8192 \
>     --cp-comm-type p2p \
>     --sequence-parallel        # SP 可与 CP 叠加
> ```
> 官方推荐：LLaMA-3 70B → `TP=4,PP=4,CP=2`；405B → `TP=8,PP=8,CP=2`。`cp-comm-type` 可选 `p2p`（Ring 风格）或 `aflash`。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[自注意力]]（$\mathcal{O}(S^2)$ 复杂度是 CP 的根因）、[[KV cache]]（CP 切的就是 KV 的序列维）、[[all-reduce]]/[[AllGather]]/all-to-all/点对点（CP 通信原语）、[[FlashAttention]]（Ring Attention 借其在线累加）、[[RoPE]]/[[causal mask]]（跨块位置与掩码必须对齐）、[[Data Parallel]]/[[Tensor Parallel]]/[[Pipeline Parallel]]/[[3D parallelism]]（CP 是叠加在 3D 上的第 4 维）。
- **下游（应用）**: [[Megatron-LM]]（CP 一等公民，`--context-parallel-size`）、DeepSpeed Ulysses、长上下文 LLM 训练（LLaMA-3 128K、Qwen 长上下文、DeepSeek-V3）、[[policy training (PT)]] 中长 response 的 RLHF rollout 重算、多模态/视频生成（高分辨率长 token）。
- **对比 / 易混**:
  - **CP vs Sequence Parallelism (SP)**：见 §1 易混表。SP 只在 LayerNorm/Dropout 切序列省显存，注意力前 AllGather 回；CP 连注意力一起切，降算力+显存。可叠加。
  - **CP vs DP**：DP 切 batch 维，单条序列仍整条在一卡；CP 切序列维，单条序列跨多卡。互补不冲突。
  - **CP vs TP**：TP 切权重（head/hidden），单条序列完整；CP 切序列，权重不切。注意力算力 TP 降不下来（仍 $\propto S^2$），CP 才降。
  - **Ring Attention vs DeepSpeed Ulysses**：见 §3.2。前者 P2P 传 KV 适合 head 少/超长，后者 all-toall 重排 head 适合 head 多。
  - **CP vs DP/TP/PP 的通信层次**：TP 必须节点内 NVLink，CP 可跨节点 IB（通信可 overlap）。

## 7. 常见误区与易错点

> [!warning] 误区 1：把 CP 等同于 Sequence Parallelism
> 名字都带"序列"，但 SP 只在非注意力处切序列省显存（注意力前 AllGather 回完整序列），CP 连注意力也切（跨块用 ring/all-toall 算）。SP 是 TP 的显存优化伴生，CP 是独立的长序列并行维度。Megatron 里两者可叠加。

> [!warning] 误区 2：以为 CP 省的是权重显存
> CP 切的是序列维（激活/KV），不切权重。权重显存要靠 TP/PP/ZeRO。CP 解决的是"序列太长导致激活/算力爆炸"，不是"模型太大"。

> [!warning] 误区 3：跨块忘记对齐 RoPE 全局位置
> 切块后每卡 token 的全局位置是 $r\cdot S/C ..$，必须用全局位置算 RoPE，不能用本地序号 $0..S/C$。否则位置编码全错，模型学到的相对位置关系崩坏。

> [!warning] 误区 4：跨块 causal mask 漏掉"右侧全 0"
> 因果掩码下，query 块 $r$ 对 key 块 $r'>r$ 应全不可见（全 0 mask），对 $r'<r$ 全可见，对 $r'=r$ 下三角。漏掉"右侧全 0"会导致未来 token 泄漏，训练崩。

> [!warning] 误区 5：以为 CP 通信像 TP 一样必须 NVLink
> CP 通信量中等且可 overlap（Ring 边传边算），跨节点 IB 可承。不像 TP 每层阻塞通信必须锁 NVLink。但 $C$ 太大环长会累积延迟，需 overlap 兜底。

> [!warning] 误区 6：固定 CP 用在变长数据上
> 变长序列按"最长序列"定 CP size，短序列被无谓切片、DP rank 算力不均。应用 Dynamic CP（2026 Megatron）按 microbatch 动态调 CP size，长尾数据可 +35%。

> [!warning] 误区 7：以为 CP 能完全消除 $\mathcal{O}(S^2)$
> CP 把单卡算力降到 $\mathcal{O}(S^2/C)$，但**全局总算力仍是 $\mathcal{O}(S^2)$**（只是摊到 $C$ 卡）。要真正把平方复杂度降下来，还得靠稀疏注意力/线性注意力等结构改进，CP 只是"横向摊薄"。

## 8. 延伸细节

### 8.1 Megatron-LM 的 CP 实现

- 配置：`--context-parallel-size C`、`--cp-comm-type p2p|aflash`；
- 自动建 CP 通信域（沿 $C$ 维的 process group）；
- attention kernel（Transformer Engine 提供）内置跨块 ring 累加 + 全局位置 RoPE + 因果掩码；
- 与 TP/PP/DP/EP 正交组合，总 GPU = 各维乘积；
- Dynamic CP：预建多组 CP group（size 2 的幂 1..$D\cdot C$），运行时 `PackedSeqParams` 携带动态 `cp_size`/`cp_group`，solver 按 packing 挑组。

### 8.2 何时该上 CP

按 NVIDIA 官方建议：
- **序列 < 8K**：不用 CP，DP/TP/PP 足够；
- **8K ~ 32K**：CP=2~4，省激活显存；
- **32K ~ 128K+**：CP=4~16，算力与显存双摊薄；
- **变长数据**：Dynamic CP。
- 优先级：先把 TP/PP/DP 配好（模型装下、吞吐够），序列仍嫌长再加 CP。CP 不是默认开，是长上下文才开。

### 8.3 CP 与 RLHF 的关系

RLHF 里 rollout 生成几百~几千 token 的长 response，learner 重算 logprob 时序列很长（prompt+response+可能 packing）。CP 让 learner 侧长序列重算不 OOM。CP 也是 [[policy training (PT)]] 长上下文 RL 的并行底座之一。注意 CP 切序列会让 [[训推不一致]] 多一个来源（CP 与非 CP 的归约顺序差异），排查 TIM 时要查 CP 配置。

### 8.4 与 FlashAttention 的协同

Ring Attention 本质是"把 FlashAttention 的逐块累加从'单卡内分块'扩展到'跨卡分块'"。FlashAttention 的在线 softmax 数学（running max/sum）让跨块累加数值正确，是 CP 能成立的数学底座。没有 FlashAttention，跨块 attention 要物化 $S^2$ 矩阵，CP 的显存优势就没了。

### 8.5 趋势

- **Dynamic CP**（2026）：变长数据的自适应 CP，Megatron Core 已落地；
- **CP + SP 叠加**：长上下文训练标配组合；
- **CP 与 EP 协同**：MoE 长上下文（DeepSeek-V3）需 CP+EP 共同切；
- **非 NVLink 友好**：CP 让长上下文训练能跨节点铺开，不像 TP 锁节点内；
- **CP-aware 算子**：Transformer Engine / FlashAttention-3 内置 CP kernel，降低用户上手门槛。

---
相关: [[并行训练]]、[[Data Parallel]]、[[Tensor Parallel]]、[[Pipeline Parallel]]、[[3D parallelism]]、[[自注意力]]、[[KV cache]]、[[FlashAttention]]、[[RoPE]]、[[causal mask]]、[[all-reduce]]、[[AllGather]]、[[Megatron-LM]]、[[policy training (PT)]]、[[训推不一致]]
