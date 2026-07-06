# KV cache

> **所属章节**: [[推理关键机制]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中高级

## 1. 一句话定义

**KV cache（键值缓存，KV cache，KV 缓存）** 是 LLM 推理时的一项**核心优化**：在 [[autoregressive decoding]] 的逐词生成中，把**已生成位置每层的 K 和 V** 缓存下来，下一步新 token 只需计算它自己的 Q/K/V，用新 Q 与缓存的所有历史 K 做 attention、取历史 V 加权，**避免每步重算历史 K/V**；它把生成长度 $n$ 的总算力从 $O(n^2 L)$（每步重算全部）降到 $O(n L)$（每步只算新位置），是 LLM 推理能实用的前提，但**以显存换算力**——cache 显存随序列线性增长，是长上下文的显存瓶颈。

## 2. 为什么需要它（动机与背景）

[[decoder-only architecture]] 自回归生成时，第 $t$ 步要算 attention $q_t$ 与 $k_{1..t},v_{1..t}$ 的关系。注意 $k_{1..t-1},v_{1..t-1}$ 在前几步**已经算过且不变**（causal：历史 token 的 K/V 不受后续影响）。

- **无 cache**：每生成一词，重新对全部 $1..t$ 位置跑一遍 attention 的 Q/K/V 投影 → 历史 K/V 被反复重算，算力随生成长度平方增长，完全不可行。
- **有 cache**：历史 K/V 算一次就存住，每步只算新 token 的 Q/K/V，新 K/V append 到 cache，新 Q 与 cache 的 K 算 attention。算力随长度**线性**增长。

KV cache 是用**显存**（存历史 K/V）换**算力**（不重算），让自回归推理从 $O(n^2)$ 降到 $O(n)$。代价：长序列下 cache 显存巨大，成为推理显存瓶颈（催生 GQA/PagedAttention/量化等优化）。
> [!note] 解答：训练时有 KV cache 吗？
> 
> **没有，也不需要。KV cache 是推理专属优化，训练时不存在**，根因有三：
> 
> 1. **训练是一次性并行前向，不是逐词生成**：LLM 预训练用 teacher forcing（token 序列已知），一次前向用 [[causal mask]] 把整段序列所有位置的 attention **一次性并行算出**（$n\times n$ 矩阵），每个位置的 K/V 在这一次前向内计算并消费掉，**没有"下一步复用历史 K/V"的场景**——cache 失去存在意义。
> 2. **训练时权重每步在更新，历史 K/V 对下一步无效**：KV cache 的前提是"权重不变 → 历史 K/V 不变 → 可复用"。训练每步反向更新权重，前一步算的 K/V 用当前权重看已经过时，**不能跨步复用**。
> 3. **训练里 K/V 是中间激活，用完即丢**（甚至用 [[gradient checkpointing]] 主动丢弃以省显存，需要时反向重算）——这与推理"缓存 K/V 省算力"**正好相反**：训练宁可重算也要省显存，推理宁可占显存也要省算力。
> 
> **训练 vs 推理对比**：
> 
> | 维度 | 训练 | 推理（decode） |
> |---|---|---|
> | 是否逐词 | 否（整序列并行） | 是（每步 1 token） |
> | K/V 角色 | 中间激活，用完丢 | 缓存，跨步复用 |
> | 权重 | 每步更新 | 冻结 |
> | 复用前提 | 不成立（权重变） | 成立（权重不变） |
> | 算力特征 | compute-bound（$n\times n$） | memory-bound（$1\times n$） |
> 
> **一句话**：训练并行算整序列、权重在变、K/V 用完即丢 → 无 KV cache；推理逐词生成、权重冻结、历史 K/V 不变 → 才需要且必须有 KV cache。推理时有没有、占多少显存见下方 callout。

> [!note] 解答：KV cache 占多少显存？推理时有 KV cache 吗？
> 
> **先回答"推理时有没有"**：**有，且是默认标配**。所有主流 LLM 推理引擎（vLLM / TGI / TensorRT-LLM / HuggingFace Transformers 的 `use_cache=True`）都默认开 KV cache——不开则每生成一个词都要对所有历史位置重算 K/V，算力随长度平方增长，完全不可行（见 [[#2. 为什么需要它（动机与背景）]]）。只有**训练**时不叫"cache"：训练一次性对整序列前向，K/V 是中间激活，不跨步复用。**推理 = 一定有 KV cache**。
> 
> **占多少显存——公式**：
> 
> $$M_{\text{KV}} = 2 \cdot B \cdot L \cdot n_{\text{kv}} \cdot n_{\text{seq}} \cdot d_h \cdot b$$
> 
> - 2：K 和 V 两份。
> - $B$：batch size。
> - $L$：Transformer 层数。
> - $n_{\text{kv}}$：**KV 头数**（GQA 下远小于 Q 头数，见 [[#3.4 GQA/MQA 省 cache]]）。
> - $n_{\text{seq}}$：当前序列长度（随 decode 线性增长）。
> - $d_h$：每头维度。
> - $b$：每元素字节数（fp16/bf16=2, fp8=1, int4=0.5）。
> 
> **速算口诀**：每 token 的 KV cache 显存 ≈ $2 \cdot L \cdot n_{\text{kv}} \cdot d_h \cdot b$ 字节。
> 
> **具体数字**（batch=1, fp16）：
> 
> | 模型 | $L$ | $n_{\text{kv}}$ | $d_h$ | 每 token | seq=4k | seq=8k | seq=32k | seq=128k |
> |---|---|---|---|---|---|---|---|---|
> | Llama-2-7B（MHA） | 32 | 32 | 128 | 512 KB | 2 GB | 4 GB | 16 GB | 64 GB |
> | Llama-3-8B（GQA） | 32 | 8 | 128 | 128 KB | 0.5 GB | 1 GB | 4 GB | 16 GB |
> | Llama-3-70B（GQA） | 80 | 8 | 128 | 320 KB | 1.25 GB | 2.5 GB | 10 GB | 40 GB |
> 
> （上表为单 batch；实际服务 batch=32/64 时再 ×B；fp8 cache 减半，int4 减到 1/4。）
> 
> **要点**：
> 
> - **随序列线性增长**：$M_{\text{KV}} \propto n_{\text{seq}}$，长上下文下 cache 显存可**超过模型权重本身**（如 Llama-3-70B 权重 fp16 约 140 GB，128k 上下文单 batch cache 就 40 GB，多 batch 更甚）——这是长上下文推理的最大显存瓶颈（[[#3.5 长上下文的 cache 瓶颈]]）。
> - **GQA 是省 cache 的关键**：Llama-2-7B（MHA, 32 KV 头）每 token 512 KB，Llama-3-8B（GQA, 8 KV 头）仅 128 KB，**省 4 倍**——这是 Llama-3 全系上 GQA 的主因之一。
> - **batch 越大 cache 越爆**：服务端高吞吐靠大 batch，但 cache 也 ×B，是显存主要消耗项，催生 PagedAttention（[[#8.1 PagedAttention（vLLM）]]）、KV 量化（[[#8.2 KV cache 量化]]）、prefix 共享（[[#8.3 prefix caching / RADIT]]）、sliding window（[[#8.4 sliding window cache]]）等优化。
> - **与权重对比**：权重是参数（不变，常驻显存），cache 是激活（随请求长度/批量动态增长）。短请求 cache < 权重，长请求/大 batch cache > 权重。
> 
> 相关：[[#4. 数学原理 / 公式]]（显存公式出处）、[[多头注意力]]（GQA/MQA）、[[参数量计算]]。
## 3. 核心概念详解

### 3.1 prefill 与 decode 两阶段

LLM 推理分两阶段：

1. **prefill（预填充）**：处理输入 prompt，一次前向算出所有 prompt 位置的 K/V，存入 cache。这一步是**并行**的（一次算多个 token），compute-bound。
2. **decode（解码）**：逐词生成。每步：新 token → 各层 forward（用 cache，只算新位置 K/V 并 append）→ 取最后 logits → 采样 → append。每步算量小但读全部权重，memory-bound。
> [!note] 已展开为独立笔记：[[PD分离]]
> 本条批注的"pd分离"指 **Prefill-Decode Disaggregation**（预填充-解码分离部署），是推理系统把 prefill 与 decode 两阶段物理分离到不同实例的架构——正是上面 §3.1 两阶段算力特征相反（prefill compute-bound、decode memory-bound）的直接推论。完整内容（动机、KV cache 迁移、池化、代表系统、与 continuous batching / prefix caching / 训推分离的关系）见 [[PD分离]]（归入 [[08-推理系统]]/推理优化）。
> 
> > [!warning] 注意"PD"撞名：本条 PD = Prefill-Decode（推理内部阶段分离），**不是** [[policy deployment (PD)|Policy Deployment]]（训推分离，第二十章）。
### 3.2 cache 的内容与结构

每层、每 KV 头、每历史位置存一对 $(K_{l,h,t}, V_{l,h,t})\in\mathbb{R}^{d_h}$。整个 cache 张量形状：

$$
\text{cache shape}: (L_{\text{layers}},\, n_{\text{kv}},\, n_{\text{seq}},\, d_h)
$$

- $L_{\text{layers}}$：Transformer 层数。
- $n_{\text{kv}}$：**KV 头数**（GQA 下 $< $ Q 头数，见 [[多头注意力]]）。
- $n_{\text{seq}}$：当前已生成的序列长度（随 decode 增长）。
- $d_h$：每头维度。
> [!note] 解答：KV cache 选择性保留/驱逐——依据什么、怎么控制？
> 
> **背景**：长上下文下 KV cache 随序列线性增长（见 [[#4. 数学原理 / 公式]] 显存公式），$n_{\text{seq}}$ 上去后显存爆，于是出现一批"**只保留部分 token 的 K/V，其余驱逐**"的工作（H2O、StreamingLLM、SnapKV、NSA 等）。核心问题就是：**留谁、删谁、依据什么？**
> 
> **判断依据（按什么给 token 打"重要性分"）**：
> 
> | 依据 | 思路 | 代表 | 优点 | 缺点 |
> |---|---|---|---|---|
> | **attention score 累积** | 该 token 被后续 token 关注越多越重要 | **H2O**（Heavy-Hitter Oracle） | 有理论支撑（重击者决定生成） | 需统计历史 attention，开销 |
> | **recency（近期性）** | 越近越可能被关注 | **sliding window** | 实现极简、O(1) | 丢失远距离信息 |
> | **sink + recency** | 保留开头几个 sink token + 最近窗口 | **StreamingLLM** | 稳定注意力分布、可无限生成 | 中段全丢，长文档召回差 |
> | **importance 评分** | 综合注意力/范数/信息量打分 | SnapKV、PyramidKV | 更精细 | 实现复杂 |
> | **学习式稀疏** | 模型本身学 token 的保留/压缩 | **DeepSeek NSA**（Native Sparse Attention） | 端到端最优 | 要改架构、重训 |
> 
> **几个关键直觉**：
> 
> - **attention sink（注意力沉）**：softmax 要求注意力权重和为 1，当没有真正想关注的 token 时，模型倾向把概率"泄"到**开头几个 token**（它们被当作无意义的概率垃圾桶）。所以开头 token 的 attention score 偏高但语义上不重要——驱逐它们会让分布崩，**必须保留**。StreamingLLM 正是发现这点：只留 sink + 滑窗，外推稳定。
> - **heavy hitter**：H2O 发现一小部分 token 累积被关注很多（重击者），它们主导生成质量，保留它们、驱逐轻击者，质量损失可控。
> - **recent > remote**：多数任务里近期 token 更相关，但**召回/长文档 QA** 例外——远处的关键事实不能丢，纯 recency 会失败。
> 
> **怎么控制（留几个、何时删）**：
> 
> - **预算 $k$**：每层保留的 token 数（或比例）。$k$ 越小越省显存、质量越降。常按层分配置（PyramidKV：浅层多留、深层少留，因浅层更需细粒度）。
> - **触发条件**：cache 长度达阈值、显存压力、或每个新 token 进来时驱逐一个最不重要的。
> - **驱逐策略**：选 importance 最低的驱逐；被驱逐的 slot 可复用（[[#8.1 PagedAttention（vLLM）]] 的分页让驱逐/复用高效）。
> - **窗口 $w$ + sink 数 $s$**：sliding window 系的两个超参；$w$ 决定保留近期多少，$s$ 决定留开头几个 sink。
> 
> **代价与权衡**：
> 
> - 驱逐会损失远距离信息，**长文档召回、多跳推理、key-value 检索**任务质量明显下降（"needle in a haystack"会失败）。
> - 越激进的驱逐省越多显存但质量越降，需按任务调（生成式对话可激进，召回任务要保守）。
> - 量化（fp8/int4 cache）是"省显存不丢信息"的正交手段，常与驱逐叠加。
> 
> **与跨请求 cache 管理的区别**：本条讲的是**单请求内**按重要性驱逐 token；跨请求的 cache 共享/驱逐（prefix caching 的 LRU、Mooncake 池化的冷热分层）是另一层，见 [[PD分离]] §8.2、[[KV cache management]]。
> 
> 相关：[[#3.5 长上下文的 cache 瓶颈]]、[[#8.1 PagedAttention（vLLM）]]、[[#8.2 KV cache 量化]]、[[#8.4 sliding window cache]]、[[PD分离]]。
### 3.3 与 causal mask 的协同

[[causal mask]] 保证位置 $t$ 只看 $j\le t$。KV cache 存的正是 $j\le t$ 的 K/V，新 token 的 Q 与 cache 的 K 做 attention **天然只看过去**——cache 是 causal mask 的运行时实现。无需对 cache 再加 mask（decode 时新 token 是最新位置，可见所有历史）。

### 3.4 GQA/MQA 省 cache

KV cache 显存 $\propto n_{\text{kv}}$。[[多头注意力]] 的 GQA/MQA 降低 KV 头数：

- MHA：$n_{\text{kv}}=h$（全头）。
- GQA：$n_{\text{kv}}=g<h$（如 Llama-3 8 头 KV）。
- MQA：$n_{\text{kv}}=1$。

cache 显存按 $n_{\text{kv}}/h$ 缩减。这是 Llama 系选 GQA 的主因之一——长上下文下 KV cache 是大头开销。

### 3.5 长上下文的 cache 瓶颈

显存 $\propto n_{\text{seq}}$。32k 上下文、百亿参数模型，KV cache 可达数十 GB，超过模型权重本身。是长上下文推理的最大显存瓶颈，催生 PagedAttention（分页）、量化（fp8/int4 cache）、offload（卸载到 CPU/SSD）等优化。

## 4. 数学原理 / 公式

### 算力对比（生成 $n$ 词）

设层 $L$、每层 attention 对 $n$ 位置的 Q/K/V 投影+attention 算力 $\approx c\cdot n^2 d$。

- **无 cache**：每步重算全部，总 $\sum_{t=1}^n c t^2 d = O(n^3 L d)$ → 不可行。
- **有 cache**：prefill $O(n_0^2 L d)$ + decode 每步 $O(n\cdot L d)$（新 Q 与 cache K 的 attention 是 $O(n)$），总 $O(n^2 L d)$ → 实用。

核心：cache 把"每步重算历史"省掉，历史 K/V 只算一次。

### 显存公式

$$
M_{\text{KV}} = 2 \cdot B \cdot L \cdot n_{\text{kv}} \cdot n_{\text{seq}} \cdot d_h \cdot b
$$

- 2：K 和 V 两份。
- $B$：batch。
- $b$：每元素字节数（fp16=2, fp8=1, int4=0.5）。
- 例：Llama-2-7B（$L=32, h=32, d_h=128$，MHA），batch=1, seq=4096, fp16：$2\times32\times32\times4096\times128\times2\approx2\text{GB}$。

### cache append 的增量

第 $t$ 步新增 $(K_t,V_t)\in\mathbb{R}^{n_{\text{kv}}\times d_h}$ 每层，append 到 cache 第 $t$ 槽。attention 用 $q_t$ 对 $K_{1..t}$ 算 $\alpha$，加权 $V_{1..t}$ 得输出。

## 5. 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CachedAttention(nn.Module):
    """极简 KV cache attention（单头，示意）。"""
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.d = d
    def forward(self, x, cache=None):
        # x: (B, n_new, d)，n_new 在 decode 时=1
        B, n, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)     # 各 (B,n,d)
        if cache is not None:
            k = torch.cat([cache['k'], k], dim=1)  # append 历史 K
            v = torch.cat([cache['v'], v], dim=1)  # append 历史 V
        cache = {'k': k, 'v': v}                   # 更新 cache
        attn = F.softmax(q @ k.transpose(-2,-1) / d**0.5, dim=-1)  # (B,n,n_total)
        out = attn @ v                            # (B,n,d)
        return out, cache

d = 64
attn = CachedAttention(d)

# prefill：一次处理 prompt
prompt = torch.randn(1, 10, d)
out, cache = attn(prompt)
print(cache['k'].shape)                           # (1,10,d) 存了 10 个 K

# decode：逐词生成，每步 n_new=1
for step in range(5):
    new_tok = torch.randn(1, 1, d)
    out, cache = attn(new_tok, cache)             # cache 复用历史，只算新 K/V
    print(step, cache['k'].shape)                 # K 长度 11,12,...,15
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[多头注意力]]（GQA/MQA 决定 KV 头数）、[[自注意力]]、[[causal mask]]（cache 与因果性协同）。
- **下游（应用）**: [[autoregressive decoding]]（每步复用 cache）、推理服务系统（vLLM/TensorRT-LM 的核心优化对象）、长上下文工程。
- **对比 / 易混**:
  - **KV cache vs 模型权重**：权重是参数（不变），cache 是激活（随序列增长）。
  - **KV cache vs activation cache**：前者特指 K/V，后者泛指中间激活（[[gradient-checkpointing]] 那种）。
  - **prefill cache vs decode cache**：同一个 cache，prefill 初始化、decode 增量更新。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 cache 省显存**：cache 省**算力**，但**占显存**（且随序列线性增长）。长上下文下 cache 显存可能超权重。
> 2. **按总 head 数算 cache**：GQA/MQA 应按 **KV 头数 $n_{\text{kv}}$** 算，不是 Q 头数 $h$，否则高估 $h/g$ 倍（见 [[多头注意力]] 误区6）。
> 3. **忘了 prefill/decode 分阶段**：prefill 并行算 prompt（compute-bound），decode 逐词（memory-bound），两者算力特征不同，服务系统要分别调度。
> 4. **decode 时还加 causal mask**：decode 新 token 是最新位置，可见所有历史 cache，无需再 mask（cache 天然只含过去）。prefill 才需 causal mask。
> 5. **cache 不更新**：每步必须把新 K/V append 进 cache，否则模型看不到历史。
> 6. **以为 cache 能跨请求共享**：不同请求的 cache 不通（除非 prefix 共享/RADIT）。每个请求独立 cache。
> 7. **batch 内不同长度不处理**：变长 batch 要 padding + 每序列独立 cache 长度管理（PagedAttention 解决）。

## 8. 延伸细节

### 8.1 PagedAttention（vLLM）

借鉴 OS 虚拟内存：把 KV cache 分成固定大小"页"，按需分配，避免每请求预留最大长度导致的显存碎片与浪费。是 vLLM 高吞吐的核心。配合 continuous batching 实现动态拼批。

### 8.2 KV cache 量化

cache 显存大 → 用 fp8/int4 存 cache，省 2~4 倍显存，质量损失可控。是长上下文推理的主流优化。进一步可按需 offload 到 CPU/SSD（带 IQKV 等）。

### 8.3 prefix caching / RADIT

不同请求共享相同前缀（system prompt）的 KV cache，避免重复 prefill。vLLM/SGLang 支持，大幅降首 token 延迟。

### 8.4 sliding window cache

长序列下 cache 无限增长不行，用**滑动窗口**只保留最近 $w$ 个 K/V（Mistral 等），牺牲远距离换取显存可控。配合 sink token 保留全局锚点。

### 8.5 cache 与 attention kernel

decode 时新 Q (1 个) 对 cache K ($n$ 个) 的 attention 是 "Q 短 K 长" 的 GEMV/小 GEMM，memory-bound（读全部 cache 与权重）。flash decoding、split-K 等优化针对该模式。见 [[flash attention]] 延伸。

### 8.6 多模态的 cache

视觉/音频 token 也进 KV cache，长视频/高分辨率图像下视觉 cache 显存巨大，是多模态推理的新瓶颈。

---
相关: [[推理关键机制]]、[[autoregressive decoding]]、[[多头注意力]]、[[自注意力]]、[[causal mask]]、[[flash attention]]、[[decoder-only architecture]]
