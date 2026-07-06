# tokenization

> **所属章节**: [[LLM结构]]
> **所属模块**: [[02-Transformer与LLM核心结构]]
> **难度**: 中级

## 1. 一句话定义

**分词（tokenization，tokenization，分词/分词器）** 是把文本切成**子词单元（token）** 并映射成整数 id 序列的预处理步骤，是 LLM 的"入口"（与 [[logits generation]] 的"出口"对偶）；主流方法是 **BPE（Byte-Pair Encoding）** 和 **SentencePiece**，目标是在**词表大小 $V$ 与序列长度**之间取平衡、并消除 OOV（未登录词），让模型用有限词表覆盖任意文本。

## 2. 为什么需要它（动机与背景）

神经网络只懂数值，文本要变成 id。最朴素的几种切法各有问题：

- **字符级**（按 char 切）：词表极小（几十~几百），但序列极长（一个词几十 token），语义稀疏、建模难。
- **词级**（按空格切词）：序列短，但词表爆炸（百万级），且**OOV**严重——训练没见过的词（新词、拼写错误、屈折变体）无法表示。
- **子词级**：折中——常见词整词一个 token，稀有词拆成若干子词 token，**任何文本都能被覆盖**（最坏拆到字符/字节）。

子词兼顾词表大小与序列长度，且**无 OOV**（任意文本都能编码），成为 LLM 标准。BPE/SentencePiece 是子词算法的代表。

## 3. 核心概念详解

### 3.1 BPE（Byte-Pair Encoding）

**训练**：从字符（或字节）序列开始，反复合并**出现频率最高的相邻 token 对**为新 token，加入词表，直到词表大小达 $V$。

1. 初始词表 = 所有字符（或字节）。
2. 统计语料中所有相邻 token 对的频率。
3. 取频率最高的对 $(a,b)$，合并成新 token $ab$，加入词表。
4. 把语料里所有 $a,b$ 相邻处替换为 $ab$。
5. 重复直到词表达 $V$。

**编码**：对新文本，按训练学到的合并规则，从最长/最高优先级的对开始贪心合并，切出 token。

- 词表是"字符 + 学到的合并单元"。
- 高频词成整词 token（短），低频词拆成子词（长但可覆盖）。

### 3.2 SentencePiece

- **语言无关**：直接在**原始 Unicode 字节流**上工作，不预先按空格切词（空格当普通字符 `▁`）。
- 支持两种算法：BPE 或 **Unigram**（基于似然的语言模型，删词使语料概率最大）。
- 任意文本可逆编码/解码（包括中文、emoji）。
- Llama/Qwen/T5 用 SentencePiece（BPE 或 Unigram 模式）。

### 3.3 byte-level BPE

GPT-2/4 用 **byte-level BPE**：先把文本编码成 UTF-8 **字节**（256 种），再在字节序列上跑 BPE。

- 任何文本都是字节序列 → **绝对无 OOV**（连罕见字符、emoji 都是字节组合）。
- 词表 = 256 字节 + 学到的合并。

### 3.4 词表、special token、编码解码

- **词表 $V$**：所有 token 的集合，大小常 3万~13万+（GPT-2 5万、Llama 3.2万、Llama-3 12.8万、Qwen2 15万+）。
- **special token**：`<bos>`(begin)、`<eos>`(end)、`<pad>`(padding)、`<unk>`(unknown，子词后基本不用) 等，用于控制流。
- **编码** `encode(text)→ids`，**解码** `decode(ids)→text`，应**可逆**（除 special token/不可见字符）。

### 3.5 中文分词特点

- 中文无空格，按字或词切。BPE/SentencePiece 常把**高频汉字组合**成 token（如"中国"一个 token）、低频字单独 token。
- 一个汉字**不一定是 1 token**：低频字可能被拆成字节（byte-level）或多 token。
- 词表含中文多 → 中文效率高（每字 token 少）；词表偏英文 → 中文费 token（影响上下文长度与成本）。

### 3.6 预分词（pre-tokenization）

BPE/SentencePiece 在训练和编码前，通常先用正则把文本切成"块"（按空格、标点、数字等），再在**块内**做合并，而不是在整个文本流上直接跑 BPE。

- **作用**：防止跨词产生奇怪合并（如 "the" + "y" 合成 "they" 占词表名额但语义混乱）、提速（块内 BPE 比全文 BPE 快）、统一空格的 token 化方式。
- **GPT-2 的正则模式**（经典）：
  ```
  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
  ```
  先抽缩写（'s 't 're...），再抽"可选空格+字母串""可选空格+数字串""可选空格+标点串"，最后处理空格。这样 `"Hello world"` → `["Hello", " world"]`，**空格粘在词前**（前导空格 token）。
- **SentencePiece 反过来**：不预切，把空格替换成元字符 `▁`（U+2581），直接在 `▁` 分隔的整流上做 BPE，解码时 `▁` → 空格。好处是空格处理更灵活、可逆。
- **预分词决定 token 边界**：换预分词正则 = 换 token 切法 = 换 tokenizer，权重不通用。

## 4. 数学原理 / 公式

### BPE 合并的选择

每步选使**合并后语料总长度（token 数）减少最多**的对——等价于选频率最高的对（合并一次减少 $|ab|$ 次相邻）。形式上：

$$
(a^*,b^*)=\arg\max_{(a,b)} \text{count}(a,b)
$$

### 压缩率（compression ratio）

衡量 tokenizer 质量的核心指标，定义为：

$$
\text{压缩率} = \frac{\text{字节数}}{\text{token 数}}
$$

- 越高越好：同样文本用更少 token → 上下文利用率高、计算省、API 便宜。
- 不同 tokenizer 在不同语言上压缩率差异巨大（见 8.6 多语言不公平），是选 tokenizer 的关键维度。
- 与 perplexity 互补：压缩率衡量"文本→token 的效率"，perplexity 衡量"模型对 token 的预测质量"，二者独立。

### Unigram（SentencePiece 选项）

给每个候选 token 一个概率 $p(v)$，语料似然 $\prod p(v_i)$。迭代删除使似然下降最小的 token，直到词表达 $V$。编码时用**最大似然分词**（动态规划找概率最大的切分）。

### 编码的贪心匹配

BPE 编码按**合并规则的优先级**（先学的先匹配）贪心切分，保证训练/编码一致：

$$
\text{encode}(x)=\text{apply\_merges\_in\_order}(x,\text{rules})
$$

## 5. 代码示例

```python
# 1. tiktoken：GPT 系的 byte-level BPE，超快
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')          # GPT-4/3.5 用
ids = enc.encode('Hello, 世界!')                     # 含中文
print(ids, enc.decode(ids))                          # 可逆

# 2. HuggingFace transformers：加载任意模型 tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
ids = tok('Hello, 世界!', return_tensors='pt')['input_ids']
print(ids, tok.decode(ids[0]))
print('vocab size:', tok.vocab_size)                 # 32000
print('special tokens:', tok.special_tokens_map)     # bos/eos/pad

# 3. 手写 BPE 训练（极简示意）
from collections import Counter
def train_bpe(corpus, V):
    # corpus: list of list of symbols（初始字符）
    vocab = set(s for w in corpus for s in w)
    while len(vocab) < V:
        pairs = Counter()
        for w in corpus:
            for i in range(len(w)-1):
                pairs[(w[i], w[i+1])] += 1
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]            # 频率最高对
        vocab.add(best)
        # 合并语料
        new_corpus = []
        for w in corpus:
            nw, i = [], 0
            while i < len(w):
                if i+1 < len(w) and (w[i], w[i+1]) == best:
                    nw.append(best); i += 2
                else:
                    nw.append(w[i]); i += 1
            new_corpus.append(nw)
        corpus = new_corpus
    return vocab
```

## 6. 与其他知识点的关系

- **上游（依赖）**: 文本语料（无 ML 依赖）。
- **下游（应用）**: [[decoder-only architecture]]（token id → embedding）、[[logits generation]]（输出投影到 vocab $V$）、上下文长度（token 数决定能否塞进模型）、成本（API 按 token 计费）。
- **对比 / 易混**:
  - **tokenizer vs model**：tokenizer 是预处理，模型只见 id；换 tokenizer 等于换词表，权重不通用。
  - **BPE vs Unigram**：合并频次 vs 似然最大化，训练目标不同。
  - **token vs 词 vs 字**：token 是子词单元，可能=词、=字、或=字节片段。
  - **tokenization vs [[位置编码]]**：前者把文本→id，后者给 id 序列注入位置。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为按空格切词**：BPE/SentencePiece 不一定按空格，SentencePiece 把空格当普通字符 `▁`，中文无空格也照切。
> 2. **token = 词**：一个词可能 1 token，也可能拆成多个子词 token；一个 token 也可能跨"词"边界。
> 3. **不同模型 tokenizer 通用**：词表/规则不同，token id 不能跨模型复用，需用对应 tokenizer。
> 4. **中文一字一 token**：低频字可能拆成多 token（byte-level），高频词可能整词一 token。中文 token 效率因模型而异（Llama-2 中文差、Qwen 中文好）。
> 5. **忘了 special token**：`<bos>`/`<eos>` 要计入输入，否则模型不知序列边界；padding 用 `<pad>` 并配合 attention 的 padding mask。
> 6. **编码不可逆**：标准 tokenizer 编码解码可逆，但若丢空格/normalization（如 NFC/小写）则可能失真，需注意 tokenizer 的 normalization 配置。
> 7. **词表越大越好**：大词表覆盖好但 embedding 参数多、softmax 算力大；要平衡（[[logits generation]] 8.2）。
> 8. **glitch token（异常 token）**：训练语料里罕见但词表里有的 token（如一长串空格、或某些无意义字符串），模型对它们的 embedding 学得差，可能引发重复输出、异常行为或"踩中即崩"。排查方法：遍历 vocab 每个 token 跑模型看输出是否异常，或用 `tiktoken` 反查稀有 token。Llama/GPT 都被发现过 glitch token。
> 9. **跨 tokenizer 的 perplexity 不可比**：perplexity 依赖词表粒度（字符级 vs 子词级数值差几个数量级），换 tokenizer 后 perplexity 数值不能直接比较；必须换算到 **bits/byte** 或 **nats/char**（按底层字节数归一）才公平。这也是为什么论文报 perplexity 必须说明 tokenizer。
> [!note] 解答：主流模型用什么 tokenizer？算法公开、词表/规则"专用"
> 
> **一句话结论**：现代 LLM 几乎**全部用 BPE 系**（byte-level BPE 或 SentencePiece+BPE），算法本身是公开的、非"专属算法"；但**词表 $V$ 和合并规则是每个模型独立训练的**，所以每个模型的 tokenizer 实例是"专用"的——token id 跨模型不通用，必须用对应模型的 tokenizer。
> 
> **主流模型对照表**（算法 + 词表 + 实现库，数字为公开配置近似值）：
> 
> | 模型 | tokenizer 算法 | 实现库 | 词表大小 | 备注 |
> |---|---|---|---|---|
> | GPT-3.5 / GPT-4 | byte-level BPE | **tiktoken** (`cl100k_base`) | ~100k | OpenAI 自训词表 |
> | GPT-4o / GPT-5 | byte-level BPE | **tiktoken** (`o200k_base`) | ~200k | 多语言大幅扩展，GPT-5 沿用 o200k 系，无新算法 |
> | Llama 1 / 2 | SentencePiece + BPE + byte fallback | SentencePiece | 32k | 偏英文，中文吃亏（见 [[#8.6 多语言不公平的量化]]） |
> | Llama 3 / 3.1 | byte-level BPE | **tiktoken** 风格（弃 SP） | 128k | 转向纯 BBPE，扩多语言 |
> | Qwen / Qwen2 / Qwen3 | byte-level BPE（BBPE） | tiktoken 风格 | 152k+ | 中文优先，中文最省 token |
> | DeepSeek-V3 / R1 | byte-level BPE | tiktoken 风格 | 128k | 与 Llama3 同族 BBPE |
> | Mistral / Mixtral | SentencePiece + BPE | SentencePiece | 32k | |
> | Gemma / Gemma2 | SentencePiece（+ byte fallback） | SentencePiece | 256k | 词表最大档之一 |
> | Claude | byte-level BPE（未完全公开） | — | ~100k+ | 推断为 BBPE 系 |
> 
> **关键区分：算法公开 vs 词表专用**
> 
> - **算法公开**：BPE（[[#3.1 BPE（Byte-Pair Encoding）]]）、SentencePiece、Unigram 都是公开算法，任何人都可用。
> - **词表/合并规则专用**：每个模型在自己的预训练语料上**训练**出一份词表 + 合并顺序，这份"字典"是模型专属的，换 tokenizer = 换词表 = 权重不通用（见 [[#7. 常见误区与易错点]] 第3条）。
> - **类比**：BPE 是"造字典的工艺"，每个模型的词表是"印出来的那本字典"——工艺公开，字典各印各的。
> 
> **趋势（2024–2025）**：
> 
> 1. **统一到 byte-level BPE**：Llama 3 放弃 SentencePiece 转用 tiktoken 风格的纯 BBPE，因为字节级天然无 OOV、实现简单、速度快、byte fallback 内建。
> 2. **大词表 + 多语言**：从 32k → 128k/152k/200k，牺牲一点 embedding/softmax 参数换中文/多语言压缩率（[[#8.1 词表大小的权衡]]）。
> 3. **数字/代码特殊处理**：Qwen/Llama3 对数字按"每 3 位一组"等规则切，提升数学与代码能力。
> 
> **常见误区**：
> 
> - ❌ 以为 GPT-5 有"专属 tokenizer 算法"：没有，仍是 byte-level BPE（o200k 系），只是词表更大。
> - ❌ 以为 Qwen/DeepSeek 用了新算法：本质都是 BBPE，区别只在词表和合并规则。
> - ❌ 以为同一族（如 Llama 系）tokenizer 可互用：Llama2 用 SentencePiece、Llama3 用 tiktoken 风格，**不互通**，id 表不同。
> - ❌ 跨模型直接喂 id：必须重新 `encode` 成该模型的 id，否则 embedding 查表错位、输出乱码。
> 
> 相关：[[tokenization]] 本节、[[logits generation]]（softmax 把 logits 归一为词表概率）、[[整体目录]]。
## 8. 延伸细节

### 8.1 词表大小的权衡

- 词表大：每词 token 少（序列短、上下文多）、但 embedding/lm_head 参数 $Vd$ 大、softmax 慢。
- 词表小：参数少、softmax 快，但序列长（同样字符数 token 多）、上下文紧。
- 多语言/中文倾向大词表（Qwen2 15万+），单语言/英文可小（GPT-2 5万）。

### 8.2 tokenizer 对模型能力的影响

- 中文 token 效率低 → 同样上下文窗口能塞的中文少，影响长文任务与成本。
- 代码/数字的 token 化影响推理（如"123"是 1 token 还是 3 token 影响数学能力）。
- tokenizer 是"训练后冻结"的，换 tokenizer 要重训，故预训练前要定好。

### 8.3 minijoin 与 byte fallback

SentencePiece 常配 **byte fallback**：词表里的 token 覆盖不到时回退到 UTF-8 字节，保证无 `<unk>`。Llama-2/3 用 SentencePiece + BPE + byte fallback。

### 8.4 特殊的 token 化

- 数字：有些 tokenizer 把"连续数字"每 3 位一组（GPT-2 的 `@@` 规则），影响数学。
- 代码：缩进/标点的 token 化影响代码模型。
- 多语言：混合语料的 tokenizer 要平衡各语言覆盖。

### 8.5 tokenizer 的不可学习性与冻结

tokenizer 在预训练前用规则训练好，**预训练中不更新**（embedding 学的是 token 的语义，不是 token 的切分）。这是它与模型权重的根本区别——tokenizer 是"字典"，权重是"语法/知识"。

### 8.6 多语言不公平的量化

在英语上训练的 tokenizer，对中文/低资源语言压缩率低 → 同样内容花更多 token → 这些语言用户上下文更短、API 更贵、长文任务更吃亏。量化例子（近似，词表与训练语料决定）：

| 模型 | 中文一字约 | 英文一词约 | 说明 |
|---|---|---|---|
| Llama-2（32k 词表，偏英文） | 1.5–2 token | ~1 token | 中文严重吃亏 |
| Llama-3（128k 词表，扩多语言） | ~1 token | ~1 token | 大幅改善 |
| Qwen2（152k 词表，中文优先） | ~0.6 token | ~1 token | 中文反而更省 |

这是多语言模型选词表大小时的核心考量：词表越大覆盖越好，但 embedding/softmax 成本线性增长（$Vd$ 参数 + $O(V)$ softmax）。

### 8.7 内容来源

本笔记 3.6（预分词）、压缩率公式、glitch token、跨 tokenizer 不可比、8.6（多语言不公平）等内容整理自 CS336 Lecture 1（概览与分词）。

---
相关: [[LLM结构]]、[[decoder-only architecture]]、[[logits generation]]、[[采样策略]]、[[autoregressive decoding]]
