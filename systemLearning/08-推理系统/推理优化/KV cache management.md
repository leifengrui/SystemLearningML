# KV cache management

> **所属章节**: [[推理优化]]
> **所属模块**: [[08-推理系统]]
> **别名**: KV cache / 键值缓存 / KV 缓存 / KV-Cache
> **难度**: 中高（需懂 [[attention]] + 显存系统）


## 1. 一句话定义

**KV cache（键值缓存）** 是 Transformer **自回归解码**时缓存的历史 token 的 **Key / Value 矩阵**：每生成一个新 token，只计算当前 token 的 $K, V$ 并 append 到 cache，再用当前 query 对整个 cache 做 attention，从而**避免每步重算历史 token 的 $K/V$ 投影**。它是 LLM 推理系统的"显存动脉"——既让解码从 $O(n^3)$ 计算降到 $O(n^2)$，又是推理时**最大的动态显存开销**（常超过模型权重本身）。

> [!note] 三句话定位
> - **是什么**：解码时把历史层的 $K, V$ 张量存显存里复用。
> - **为什么**：自回归每步只多 1 个 token，重算全部历史 $K/V$ 是浪费。
> - **代价**：显存 $\propto \text{layers} \times \text{seq} \times \text{batch}$，是推理吞吐的主要瓶颈。


## 2. 为什么需要它（动机与背景）

### 2.1 朴素自回归的浪费

Transformer 生成文本是**自回归**的：第 $t$ 步用已生成的 $t-1$ 个 token 预测第 $t$ 个。第 $t$ 步要计算 attention：

$$
\text{Attn}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\!\left(\frac{Q_t K_{1:t}^\top}{\sqrt{d}}\right) V_{1:t}
$$

这里 $K_{1:t}, V_{1:t}$ 是前 $t$ 个 token 经线性投影得到的 Key/Value。**问题**：第 $t+1$ 步又要算 $K_{1:t+1}, V_{1:t+1}$，其中前 $t$ 个 token 的 $K, V$ 与第 $t$ 步**完全相同**却被重算一遍。

| 项 | 朴素（每步重算全部） | KV cache（只算新 token） |
|---|---|---|
| 每步 $K/V$ 投影 | $O(t \cdot d)$（全部历史） | $O(1 \cdot d)$（仅当前） |
| $n$ 步总投影 | $\sum_{t=1}^{n} t = O(n^2)$ | $O(n)$ |
| 每步 attention scores | $O(t \cdot d)$（query 对全部 key） | 同左（无法省） |
| $n$ 步总 attention | $O(n^2)$ | $O(n^2)$ |
| **显存** | 低（不存历史） | **高**（存全部历史 $K/V$） |

> [!tip] KV cache 省的是"投影重算"，省不掉"attention 本身"
> attention scores $Q K^\top$ 每步仍要对全部历史 key 算（新 query 要看所有旧 key），这部分 $O(n^2)$ 是 attention 机制固有的，cache 无法消除。**KV cache 用显存换的是"重复的 $W_K, W_V$ 线性投影计算"**。所以长序列下 attention 的 $O(n^2)$ 计算（[[FlashAttention]] 优化）和 KV cache 的 $O(n)$ 显存是两个并行的瓶颈。

### 2.2 没有它会怎样

- 每生成一个 token 都把整段历史过一遍 $W_K/W_V$ 投影 → 序列越长越慢，实际不可用。
- 推理框架（vLLM / TGI / TensorRT-LLM / llama.cpp）的核心优化都围绕 KV cache 的**分配、复用、压缩**展开。


## 3. 核心概念详解

### 3.1 cache 的形状与生命周期

每个请求、每层、每个 head 都有一份 $K$ 和 $V$。以 Llama 风格多头注意力为例，单个请求的 cache 形状：

$$
K_{\text{cache}}, V_{\text{cache}} \in \mathbb{R}^{n_{\text{layers}} \times n_{\text{heads}} \times L \times d_{\text{head}}}
$$

其中 $L$ 是目前已生成的序列长度（随解码增长），$d_{\text{head}}$ 是每头维度，$n_{\text{heads}} \cdot d_{\text{head}} = d_{\text{model}}$。

> [!note] prefill 与 decode 两阶段
> - **Prefill（预填）**：处理输入 prompt。prompt 有多个 token，**一次性并行**计算它们的 $K/V$ 并填入 cache（这一步是 compute-bound，算力密集）。
> - **Decode（解码）**：逐 token 生成。每步喂 1 个 token，算其 $K/V$，**append** 到 cache 末尾，再对全 cache 做 attention（这一步是 memory-bound，每步只动一点点算力却要读整个 cache）。
>
> 两阶段的瓶颈不同：prefill 拼 [[GPU utilization]]（算力），decode 拼显存带宽与 [[batching tradeoff]]。vLLM 的 [[continuous batching]] 就是为了让 decode 阶段多个请求凑批、提升带宽利用率。

### 3.2 增量更新（append 语义）

解码第 $t$ 步：
1. 当前 token $x_t$ 过 $W_Q, W_K, W_V$ 得 $q_t, k_t, v_t$（各 $1$ 个 token）。
2. $K_{\text{cache}} \leftarrow [K_{\text{cache}};\, k_t]$, $V_{\text{cache}} \leftarrow [V_{\text{cache}};\, v_t]$（沿序列维 append）。
3. $\text{out}_t = \text{Attn}(q_t, K_{\text{cache}}, V_{\text{cache}})$。

> [!warning] 误区：append 不是原地写
> 直接 `torch.cat` 会每步复制整个 cache（$O(L)$ 拷贝），浪费且碎片化。生产系统用**预分配 buffer + 填充指针**（如 `cache_k[:, :, pos, :] = k_t`），避免反复分配。下文代码示例为清晰用 `cat`，工程实现见 [[#5. 代码示例（可选]] 的 buffer 版。

### 3.3 PagedAttention（vLLM 的分页 KV）

朴素 cache 要求**连续显存**且**预分配最大 seq_len**，导致：
- **碎片化**：请求长短不一，固定块浪费。
- **无法共享**：公共前缀（system prompt）的 KV 不能复用。

**PagedAttention**（vLLM）借鉴 OS 虚拟内存分页：把每个请求的逻辑 KV 序列切成固定大小 **block**（如 16 token/block），用 **block table** 把逻辑 block 映射到物理 block 池。物理 block 可不连续，空闲即用。

| 特性 | 朴素连续 cache | PagedAttention |
|---|---|---|
| 分配 | 一次分配 max_len | 按 block 动态增长 |
| 碎片 | 高（padding 到 max_len） | 低（仅末尾 block 可能不满 16） |
| 显存利用率 | ~30–50% | ~95%+ |
| 前缀共享 | 不支持 | 支持（指向同一物理 block）→ [[prefix caching]] |
| copy-on-write | 无 | 支持（分支生成/beam 不必复制全 cache） |

### 3.4 KV cache 量化

把 FP16 的 $K/V$ 存成 **FP8 / INT8 / INT4**，显存直接减半 / 减 4× / 减 8×。代价是精度损失（$K$ 对量化敏感因 attention score 放大误差，$V$ 相对鲁棒）。典型：FP8 KV 几乎无损，INT4 KV 需配 grouped-query attention + 细调。详见 [[量化]]（待展开，Chapter 1/2）。

### 3.5 cache 管理策略

当显存不够装所有并发请求的 cache 时，需要**驱逐/换出**：

| 策略 | 做法 | 场景 |
|---|---|---|
| **LRU eviction** | 按最近最少使用丢弃旧请求的 cache | 长尾请求多 |
| **recomputing** | 被驱逐的请求回来时重算其 prefill KV | 比换出便宜（compute < bandwidth）时 |
| **offload to CPU/SSD** | 把冷 cache 挪到 CPU 内存或 NVMe | [[activation memory]] 不足，DeepSpeed-MII / FlexGen |
| **prefix sharing** | 多请求共享公共前缀 KV | [[prefix caching]]、SGLang radix tree |


## 4. 数学原理 / 公式

### 4.1 显存占用公式

单个请求、单层、单个 head、单个 token 的 $K$ 或 $V$ 占 $d_{\text{head}}$ 个元素。$K$ 和 $V$ 共 2 份，全 $L$ 层、$n_{\text{heads}}$ 头、序列长 $S$、batch $B$、精度 $p$ 字节/元素：

$$
\boxed{M_{\text{KV}} = 2 \cdot n_{\text{layers}} \cdot n_{\text{heads}} \cdot d_{\text{head}} \cdot S \cdot B \cdot p}
$$

用 $d_{\text{model}} = n_{\text{heads}} \cdot d_{\text{head}}$ 化简：

$$
M_{\text{KV}} = 2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot S \cdot B \cdot p
$$

> [!note] 推导关键
> - "2" = $K$ 和 $V$ 两份。
> - $n_{\text{heads}} \cdot d_{\text{head}}$ 合并为 $d_{\text{model}}$（总维度）。
> - 与 batch $B$、seq $S$、精度 $p$ **线性**相关——这就是长序列 / 大 batch 下 KV cache 显存爆炸的根因。

### 4.2 数值实例：KV cache 常超过权重

以 **Llama-2 7B**（$n_{\text{layers}}=32$, $d_{\text{model}}=4096$, FP16 $p=2$）为例。

**模型权重显存**：7B 参数 × 2 字节 ≈ **14 GB**。

**单条请求 KV cache**（$S=2048, B=1$）：

$$
M = 2 \times 32 \times 4096 \times 2048 \times 1 \times 2 = 1{,}073{,}741{,}824 \text{ B} \approx 1.07 \text{ GB}
$$

**batch=64 并发**：

$$
M = 1.07 \times 64 \approx 68.7 \text{ GB} \gg 14 \text{ GB} \text{（权重）}
$$

> [!warning] 工程直觉：推理显存主项是 KV cache，不是权重
> 大 batch + 长序列时，KV cache 显存可达权重的 4–10×。所以推理引擎的"能开多大 batch / 多少并发"几乎完全由 KV cache 显存决定，而非权重。这是 [[batching tradeoff]] 与 [[continuous batching]] 的出发点。

### 4.3 GQA / MQA 对 cache 的影响

- **MHA**：$n_{\text{heads}}$ 个 KV 头 → cache 全量。
- **MQA**（Multi-Query）：所有 query head 共享 **1** 个 KV 头 → cache 缩 $\times \frac{1}{n_{\text{heads}}}$。
- **GQA**（Grouped-Query）：分 $g$ 组，每组共享 1 KV 头（$g$ 个 KV 头）→ cache 缩 $\times \frac{g}{n_{\text{heads}}}$。

Llama-2 70B / Llama-3 全系用 GQA，正是为压 KV cache 显存、提升 decode 吞吐。详见 [[attention]]（Chapter 2，待展开 / 已有）。


## 5. 代码示例（可选

### 5.1 教学版：带 KV cache 的多头注意力（cat 语义，看清 append）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CachedMHA(nn.Module):
    """简化多头注意力 + KV cache（教学用，省略 RoPE/GQA/causal mask 细节）。"""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None, use_cache=False):
        # x: [B, S, D]  (prefill: S>1; decode: S=1)
        B, S, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)            # 各 [B, S, D]
        # 拆头: [B, S, D] -> [B, H, S, hd]
        split = lambda t: t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)

        # —— KV cache: 把新 k/v 拼到历史末尾 ——
        if kv_cache is not None:
            k_past, v_past = kv_cache                      # 各 [B, H, S_past, hd]
            k = torch.cat([k_past, k], dim=2)             # [B, H, S_past+S, hd]
            v = torch.cat([v_past, v], dim=2)
        new_cache = (k, v) if use_cache else None

        # attention: q 对更新后的 k/v
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)                  # [B, H, S_q, S_kv]
        ctx = torch.matmul(attn, v)                       # [B, H, S_q, hd]
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(ctx), new_cache


# 演示: prefill 一次建 cache, decode 逐步 append
torch.manual_seed(0)
layer = CachedMHA(d_model=64, n_heads=8)
prompt = torch.randn(1, 5, 64)                            # 假设 prompt 5 token

# prefill: 一次算 prompt 全部, 建 cache
out, cache = layer(prompt, use_cache=True)
print("prefill out:", tuple(out.shape), "cache K:", tuple(cache[0].shape))
# (1, 5, 64) (1, 8, 5, 8)

# decode: 每步喂 1 个新 token, cache 增长
for step in range(3):
    new_tok = torch.randn(1, 1, 64)
    out, cache = layer(new_tok, kv_cache=cache, use_cache=True)
    print(f"step {step} out:", tuple(out.shape), "cache len:", cache[0].shape[2])
# step 0 out: (1, 1, 64) cache len: 6
# step 1 out: (1, 1, 64) cache len: 7
# step 2 out: (1, 1, 64) cache len: 8
```

### 5.2 工程版要点：预分配 buffer（避免 cat 复制）

```python
class CachedMHA_Buffer(nn.Module):
    """工程风格: 预分配 max_len 的 KV buffer, 用指针 pos 填充, 避免 cat 拷贝。"""
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.max_len = max_len
        # 预分配 [B, H, max_len, hd] —— 实际由框架(vLLM PagedAttention)按 block 管理
        self.register_buffer("k_buf", torch.zeros(1, n_heads, max_len, self.head_dim), persistent=False)
        self.register_buffer("v_buf", torch.zeros(1, n_heads, max_len, self.head_dim), persistent=False)

    def forward(self, x, pos):
        # x: [B, S, D], pos: 当前要写入的起始位置
        B, S, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # 原地写入 buffer 的 [pos, pos+S) 段, 不复制历史
        self.k_buf[:, :, pos:pos+S] = k
        self.v_buf[:, :, pos:pos+S] = v
        # attention 只看 [0, pos+S)
        k_used = self.k_buf[:, :, :pos+S]
        v_used = self.v_buf[:, :, :pos+S]
        scores = torch.matmul(q, k_used.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v_used).transpose(1, 2).contiguous().view(B, S, D)
        return ctx   # 省略 out 投影
```

### 5.3 显存估算小工具

```python
def kv_cache_bytes(n_layers, d_model, seq_len, batch, bytes_per=2):
    """KV cache 显存(字节). K+V 共 2 份."""
    return 2 * n_layers * d_model * seq_len * batch * bytes_per

# Llama-2-7B, seq 2048, batch 64, FP16
print(f"{kv_cache_bytes(32, 4096, 2048, 64, 2)/1e9:.1f} GB")   # ~68.7 GB
# 对比权重 7B*2 = 14 GB
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[attention]]（自注意力机制，K/V 从哪来）、[[Transformer]]（整体结构）、[[混合精度训练|FP16/BF16 精度]]（决定 $p$）。
- **下游（应用）**: [[continuous batching]]（基于 KV cache 的迭代级调度）、[[prefix caching]]（复用公共前缀 KV）、[[speculative decoding]]（草稿模型也要写主模型 cache）、[[batching tradeoff]]（batch 大小受 KV 显存约束）、[[GPU utilization]]（decode 阶段带宽瓶颈在 cache 读取）、[[activation memory]]（KV 是激活显存的最大组成）。
- **对比 / 易混**:
  - **KV cache vs 模型权重**：权重是共享参数（固定，推理时不改）；KV cache 是**每个请求独立**的激活（动态，随序列增长）。推理显存 = 权重 + 所有并发请求的 KV cache 之和。
  - **KV cache vs activation checkpointing**：前者是推理时存历史 K/V 供复用；后者是训练时省前向激活的重算策略（[[gradient checkpointing]]，Chapter 11）。名字都含"cache/checkpoint"但场景完全不同。
  - **PagedAttention vs radix tree**：前者按固定 block 分页（vLLM）；后者按前缀树组织共享前缀（SGLang），见 [[prefix caching]]。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 KV cache 能消除 attention 的 $O(n^2)$ 计算
> 不能。$Q K^\top$ 每步新 query 仍要对**全部历史 key**算分。cache 省的是"$K/V$ 线性投影的重复计算"，attention scores 计算量不变。$O(n^2)$ 那部分靠 [[FlashAttention]] / 稀疏注意力减。

> [!warning] 误区 2：以为 KV cache 显存很小、可忽略
> 恰恰相反，大 batch + 长序列时 KV cache 显存 **远超权重**（见 §4.2，64×2048 → 64GB vs 14GB 权重）。推理吞吐的硬约束常是 KV cache 显存。

> [!warning] 误区 3：以为只有 decode 阶段才有 cache
> prefill 阶段也在**建** cache（一次性算 prompt 全部 K/V 填入）。prefill 是 compute-bound，decode 是 memory-bound，两者瓶颈不同。

> [!warning] 误区 4：以为请求结束 cache 立即无效
> cache 本身可被 [[prefix caching]] / SGLang radix tree **跨请求复用**（公共前缀）。只有权重更新（如在线学习）后 cache 才真正失效（版本不匹配）。

> [!warning] 误区 5：MHA/GQA/MQA 搞混 cache 倍率
> 记住：cache 大小正比于 **KV 头数**。MHA = $n_h$ 倍，GQA = $g$ 倍（$g$ 组），MQA = 1 倍。Llama-3 用 GQA 正是为此。


## 8. 延伸细节

### 8.1 PagedAttention 的 block table（类比 OS 页表）

逻辑上请求看到连续的 $[0, 1, \dots, L-1]$，物理上每个逻辑 block 映射到 block 池中某个空闲物理 block：

```
逻辑 block 序号:    [0]      [1]      [2]      [3]
block table:        → 物理block#5  → #2  → #9  → #1
物理 block 池:       #0(空) #1(用) #2(用) ... #5(用) #9(用)
```

请求增长时申请新物理 block；结束时归还。前缀相同时多个请求的 block table 前几项指向**同一物理 block**（[[prefix caching]]）。beam search 分支用 **copy-on-write**：只在写入分歧 block 时才复制，不必克隆整个 cache。

### 8.2 KV cache 与带宽瓶颈（decode 阶段）

decode 每步只算 1 个 token 的 forward（计算量极小），但要**读整个 KV cache**进 SRAM 算 attention。对 7B 模型 batch=1，每步读 ~1GB cache → 受 HBM 带宽限制（A100 1.5TB/s → 约 1.5ms/token → ~660 tok/s 上限，实际更低）。这是 decode 阶段 [[GPU utilization]] 低、必须靠 [[continuous batching]] 凑大 batch 提升带宽利用率的根因。

### 8.3 KV 量化与 offload

- **FP8 KV**（vLLM/TensorRT-LLM 支持）：显存减半，精度几乎无损，是当前主流。
- **INT4 KV**：省 4×，但 $K$ 量化误差被 attention score 放大，需配 GQA + 细调或 mixed-precision（$K$ 保高精度、$V$ 量化）。
- **KV offload**：DeepSpeed-MII / FlexGen 把冷 cache 换到 CPU 内存或 NVMe，用 PCIe/NVMe 带宽换 HBM 容量。适合离线批量推理（延迟不敏感）。

### 8.4 cache 与 RLHF / 训推分离

在 [[policy training (PT)]]-[[policy deployment (PD)]] 架构（[[训推分离]]）中，PD 侧的 KV cache 是**采样吞吐**的关键资源。[[weight sync mechanism]] 推新权重时，旧 cache 因对应旧 $\theta$ 而失效（logits 不一致），通常**清空 cache 或等当前请求完成再切换**。这是 [[stale policy problem]] 在推理侧的投影。

### 8.5 与 vLLM / SGLang / TRT-LLM 的对应

| 框架 | KV cache 管理 | 特色 |
|---|---|---|
| vLLM | PagedAttention（block 16） | 显存利用率高、prefix caching 内置 |
| SGLang | Radix tree 前缀复用 | 结构化 prompt 共享更激进 |
| TensorRT-LLM | 连续/int8 KV | 极致 kernel 融合、低延迟 |
| DeepSpeed-FastGen | Dynamic Splitfuse | prefill/decode 融合调度 |

---
相关: [[推理优化]] | [[continuous batching]] | [[prefix caching]] | [[attention]] | [[activation memory]]
