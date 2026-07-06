# prefix caching

> **所属章节**: [[推理优化]]
> **所属模块**: [[08-推理系统]]
> **别名**: prefix caching / 前缀缓存 / Automatic Prefix Caching (APC) / prompt caching
> **难度**: 中（需懂 [[KV cache management]] + PagedAttention）


## 1. 一句话定义

**prefix caching（前缀缓存）** 是 LLM 推理引擎复用**多个请求公共前缀**的 [[KV cache management]] 的技术：当多个请求共享相同的开头（system prompt、few-shot 示例、文档前缀等），只对第一个请求算这部分前缀的 KV 并存入缓存，后续请求直接复用这部分 KV、只 prefill **各自不同的后缀**。vLLM 称 Automatic Prefix Caching (APC)，SGLang 用 radix tree（基数树）组织共享前缀，OpenAI API 称 prompt caching。

> [!note] 三句话定位
> - **是什么**：把"公共前缀的 KV"当成可跨请求复用的资源，避免重复 prefill。
> - **为什么**：system prompt 动辄 1K–8K token，每个请求都重算 prefill 是巨大浪费。
> - **怎么做**：按 token 序列哈希/前缀树定位已缓存的公共前缀 KV，复用其 block，只 prefill 增量后缀。


## 2. 为什么需要它（动机与背景）

### 2.1 公共前缀的重复 prefill 浪费

实际 LLM 服务中，请求常带**相同前缀**：

- **system prompt**：所有请求共享（如 "You are a helpful assistant..." + 工具说明），1K–4K token。
- **few-shot 示例**：同一任务的多个请求共享示例段。
- **RAG 上下文**：同一文档的不同提问共享检索到的文档段。
- **多轮对话**：历史轮次是后续请求的前缀。

每次请求都从 token 0 开始 prefill 整个前缀，算这段的 $K/V$。若前缀长 4K token，每个请求为此消耗 4K token 的 prefill 算力——而这段 KV 与其他请求**完全相同**。

> [!tip] 量化浪费
> Llama-7B prefill 4K token 前缀 ≈ $2 \times 7\text{B} \times 4096 \approx 5.7 \times 10^{13}$ FLOPs/请求。1000 req/s 的服务，每秒 $5.7 \times 10^{16}$ FLOPs 全在重算相同前缀——浪费。prefix caching 把它降到几乎 0（仅复用 + 小段后缀 prefill）。

### 2.2 没有它会怎样

- TTFT（首 token 延迟）被长前缀 prefill 拖高。
- prefill 算力被重复消耗，GPU 利用在"重算"而非"新请求"。
- 长 context 应用（RAG、agent）成本居高不下。

prefix caching 把"每个请求独立 prefill 全段"变成"首次 prefill + 后续复用"，是降本降延迟的关键。


## 3. 核心概念详解

### 3.1 前缀匹配：block 粒度 vs token 粒度

KV cache 在 PagedAttention 下按 **block**（如 16 token/block）组织。prefix caching 自然也按 block 粒度匹配：

- 把请求的 token 序列按 block 切分。
- 逐 block 计算哈希，查缓存是否已有该 block 的 KV。
- 命中的 block 直接复用物理 block（block table 指向同一物理 block），不重算。
- 从首个未命中 block 开始 prefill 增量部分。

> [!note] 为什么按 block 而非整段哈希
> 整段哈希要求前缀**完全相同**才命中，灵活性差。按 block 哈希可命中"部分公共前缀"——如两个请求前 10 个 block 相同、第 11 个 block 才分歧，仍能复用前 10 个 block 的 KV。SGLang 的 radix tree 进一步把这种"前缀共享"组织成树，自动发现任意长度的公共前缀。

### 3.2 radix tree（基数树，SGLang）

SGLang 用 **radix tree**（一种压缩前缀树）管理 KV cache 的共享：

- 每个节点代表一段 token 子序列及其对应 KV block。
- 从根到某节点的路径 = 一个请求的前缀。
- 多个请求共享根附近的节点（公共前缀），分歧处才分叉。
- 引用计数管理：某节点无引用时可回收。

```
root
 └─ [system_prompt 4K tok] (KV block #1..#256, ref=5)
     ├─ [few_shot_A] (KV #257..#260, ref=2)
     │   └─ 请求1 后缀
     └─ [few_shot_B] (KV #257..#259, ref=3)
         └─ 请求2 后缀
```

5 个请求共享 system_prompt 节点（ref=5），其 KV 只算一次。比朴素 block 哈希更高效（自动结构化发现共享）。

### 3.3 vLLM APC（Automatic Prefix Caching）

vLLM 的 APC 基于 PagedAttention 的 block 共享：

- 每个 block 计算哈希（基于 token 内容 + 位置）。
- 请求 prefill 时逐 block 查哈希表，命中则复用物理 block。
- **copy-on-write**：若某 block 需被修改（罕见，KV 不可变），才复制。
- 不命中的 block 正常 prefill 并写入缓存供后续复用。

### 3.4 版本与失效

KV cache 对应**特定模型权重版本 $\theta$**。若权重更新（在线学习、[[weight sync mechanism]] 推新 $\theta$），旧 KV 与新 $\theta$ 不匹配，前缀缓存**必须失效**——否则用旧 KV 算出的 attention 与新模型不一致（[[stale policy problem]] 在推理侧的投影）。所以缓存需带版本号，权重切换时清空或标记失效。

### 3.5 与 PagedAttention 的关系

prefix caching 几乎必然基于 PagedAttention：

- 朴素连续 KV cache 下，复用前缀要拷贝整段 KV（昂贵）。
- PagedAttention 下，复用 = 让多个请求的 block table 前几项指向**同一物理 block**（零拷贝，仅指针）。

所以 vLLM 的 APC 与 PagedAttention 是一体的。


## 4. 数学原理 / 公式

### 4.1 节省的 prefill 计算量

设请求前缀长 $P$ token，后缀长 $S$ token（$P \gg S$ 常见）。模型参数量 $N$，prefill 单 token 计算量 $\approx 2N$ FLOPs（前向）。

**无 prefix caching**：每请求 prefill $P + S$ token：

$$
C_{\text{no}} = (P + S) \cdot 2N
$$

**有 prefix caching**（前缀命中）：仅 prefill 后缀 $S$ token（前缀复用零计算）：

$$
C_{\text{with}} = S \cdot 2N
$$

**节省比**：

$$
\text{节省} = \frac{C_{\text{no}} - C_{\text{with}}}{C_{\text{no}}} = \frac{P}{P+S}
$$

> [!note] 推导要点
> 节省比例 = 前缀占比 $P/(P+S)$。前缀越长（相对总长），节省越多。RAG/agent 场景 $P \gg S$（如 $P=8000, S=100$），节省 $\approx 98.8\%$——这就是 prefix caching 在长 context 应用中收益巨大的根因。

### 4.2 数值实例

RAG 服务，文档前缀 $P=8000$ token，提问后缀 $S=100$ token，Llama-7B（$N=7\text{B}$）。

- 无缓存：$(8000+100) \times 2 \times 7\text{B} = 1.13 \times 10^{14}$ FLOPs/请求。
- 有缓存：$100 \times 2 \times 7\text{B} = 1.4 \times 10^{12}$ FLOPs/请求。
- 节省 $\approx 98.8\%$，TTFT 从 ~80ms 降到 ~1ms（量级）。

### 4.3 缓存命中率与 LRU

prefix cache 容量有限（显存预算）。命中率 $h$ 取决于请求分布与缓存大小。有效吞吐：

$$
\text{TPS}_{\text{eff}} \approx \frac{1}{(1-h) \cdot C_{\text{no}}/\text{throughput}_{\text{prefill}}}
$$

$h \to 1$ 时趋近"只 prefill 后缀"的极高吞吐。LRU 驱逐冷前缀以维持命中率。


## 5. 代码示例（可选

### 5.1 简化 prefix cache（block 哈希 + 最长前缀匹配）

```python
import hashlib

class PrefixCache:
    """简化 prefix caching: 按 token 块哈希, 复用最长公共前缀的 KV."""
    def __init__(self, block_size=4):
        self.block_size = block_size
        self.store = {}        # prefix_hash -> kv (示意, 实际是物理 block)
    def _hash(self, tokens):
        return hashlib.md5(str(tokens).encode()).hexdigest()[:8]
    def get_or_compute(self, tokens, compute_kv):
        """返回 (kv_list, reused_blocks). 复用最长已缓存前缀."""
        # 按 block 切分
        blocks = [tokens[i:i+self.block_size] for i in range(0, len(tokens), self.block_size)]
        reused = 0
        kv_list = []
        # 找最长连续命中前缀
        for blk in blocks:
            h = self._hash(blk)
            if h in self.store:
                kv_list.append(self.store[h]); reused += 1
            else:
                break   # 首个未命中即停 (前缀必须连续)
        # 剩余 block 需 compute, 并写入缓存
        for blk in blocks[reused:]:
            kv = compute_kv(blk)
            self.store[self._hash(blk)] = kv
            kv_list.append(kv)
        return kv_list, reused * self.block_size

# —— 演示: 三请求共享 system prompt 前缀 ——
cache = PrefixCache(block_size=4)
compute_kv = lambda blk: f"KV({blk})"   # 假装算 KV

sys_prompt = [10,11,12,13, 20,21,22,23]   # 8 token = 2 block
r1 = sys_prompt + [100,101,102,103]       # 前缀 + 后缀A
r2 = sys_prompt + [200,201,202,203]       # 前缀 + 后缀B
r3 = sys_prompt + [300,301]               # 前缀 + 短后缀

kv1, r1_used = cache.get_or_compute(r1, compute_kv)
print(f"r1: reused {r1_used} tokens")     # 0 (首次, 全算)
kv2, r2_used = cache.get_or_compute(r2, compute_kv)
print(f"r2: reused {r2_used} tokens")     # 8 (复用 system prompt)
kv3, r3_used = cache.get_or_compute(r3, compute_kv)
print(f"r3: reused {r3_used} tokens")     # 8 (同上)
```

预期：r1 无复用（首次），r2/r3 各复用 8 token 的 system prompt——只 prefill 自己的后缀。

### 5.2 radix tree 风格（示意）

```python
class RadixNode:
    def __init__(self, tokens):
        self.tokens = tokens; self.children = {}; self.kv = None; self.ref = 0
class RadixCache:
    def __init__(self): self.root = RadixNode([])
    def insert(self, tokens, compute_kv):
        """插入请求, 复用已有前缀, 返回复用长度."""
        node, i = self.root, 0
        # 沿树找最长公共前缀
        while i < len(tokens):
            head = tokens[i]
            if head not in node.children: break
            child = node.children[head]
            # 比较该 child 的 token 段
            j = 0
            while j < len(child.tokens) and i+j < len(tokens) and child.tokens[j] == tokens[i+j]:
                j += 1
            if j == len(child.tokens):
                node = child; i += j; node.ref += 1
            else:
                # 部分匹配: 需 split (略, 工程实现细节)
                break
        reused = i
        # 剩余 token 作为新子节点
        if i < len(tokens):
            new_child = RadixNode(tokens[i:]); new_child.kv = compute_kv(tokens[i:]); new_child.ref = 1
            node.children[tokens[i]] = new_child
        return reused
```

这是 SGLang radix tree 的极简示意（省略 split/copy-on-write）。


## 6. 与其他知识点的关系

- **上游（依赖）**: [[KV cache management]]（prefix caching 是 KV cache 复用的特化）、PagedAttention（block 共享是零拷贝复用的前提）、[[attention]]（前缀 KV 的来源）。
- **下游（应用）**: [[continuous batching]]（新请求 prefill 时先查 prefix cache）、[[batching tradeoff]]（减少 prefill 开销影响 batch 调度）、[[sampling throughput]]（RLHF 中 system prompt 固定时大幅提速采样）、[[policy deployment (PD)]]（PD 服务的降本关键）。
- **对比 / 易混**:
  - **prefix caching vs [[KV cache management]]**：后者是单请求内缓存历史 KV 供自回归复用；前者是**跨请求**复用公共前缀 KV。层次不同：单请求 vs 多请求。
  - **prefix caching vs checkpointing**：无关。后者是训练省激活（[[gradient checkpointing]]）。
  - **prefix caching vs [[量化|KV 量化]]**：前者复用计算（省 prefill FLOPs），后者压缩存储（省显存）。可叠加：量化 KV + 前缀复用。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为前缀"相似"就能复用
> 不能。prefix caching 要求**逐 token 完全相同**的前缀才命中（block 哈希严格匹配）。一个 token 不同，从该 block 起就不复用。"语义相似"不算数。

> [!warning] 误区 2：以为权重更新后缓存仍有效
> KV cache 是**对应特定 $\theta$** 的中间结果。权重更新（[[weight sync mechanism]]）后，旧 KV 与新 $\theta$ 不匹配，必须失效缓存，否则 attention 结果不一致（[[stale policy problem]]）。生产系统要给缓存打版本号，切换即清空。

> [!warning] 误区 3：以为 prefix caching 优化 decode
> 它优化的是 **prefill**（省前缀的 prefill 计算）。decode 阶段每步仍正常写自己的 KV cache，prefix caching 不直接干预 decode。

> [!warning] 误区 4：忽略缓存容量与命中率
> 显存有限，prefix cache 装不下所有可能前缀。冷前缀被 LRU 驱逐后，新请求若恰好用该前缀会 miss。设计时要预留 cache 显存预算，监控命中率。

> [!warning] 误区 5：以为 block 越小命中率越高越好
> 小 block 命中更细（部分前缀也能复用），但哈希表更大、查找开销高、碎片多。典型 block 16 token 是权衡点。


## 8. 延伸细节

### 8.1 OpenAI prompt caching 的实际形态

OpenAI API 的 prompt caching：对长 prompt（>1024 token）自动缓存，相同前缀的后续请求按缓存命中 token 数打折计费（如缓存命中部分 0.5× 费率）。这推动应用层把稳定 system prompt / few-shot 放最前面、变量放最后，以最大化命中。

### 8.2 SGLang 的结构化共享

SGLang 不仅做前缀缓存，还支持**结构化提示**的共享（如 few-shot 多分支、多轮对话树），通过 radix tree 自动发现任意共享结构，比朴素前缀匹配更激进。

### 8.3 与 RLHF 的关系

[[rollout worker]] 采样时，system prompt / prompt template 固定。prefix caching 让每个采样请求只 prefill 用户输入 + 生成部分，system prompt 段复用，大幅提升 [[sampling throughput]]。在 [[policy training (PT)]]-[[policy deployment (PD)]] 架构里，PD 侧的 prefix caching 是 RLHF 训练加速的关键一环。但注意 [[weight sync mechanism]] 推新 $\theta$ 时需失效缓存。

### 8.4 chunked prefill 与 prefix caching 的协同

长前缀即使缓存 miss 也要 chunked prefill（不阻塞 decode）；缓存 hit 时直接复用，连 chunked prefill 都省了。两者在 vLLM/SGLang 调度器里协同决定每个 iteration 做什么。

---
相关: [[推理优化]] | [[KV cache management]] | [[continuous batching]] | [[batching tradeoff]] | [[sampling throughput]]
