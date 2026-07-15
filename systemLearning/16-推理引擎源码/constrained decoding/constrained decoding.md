# constrained decoding

> **所属章节**: [[constrained decoding]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: constrained decoding / 约束解码 / logit mask / logit masking / guided decoding / FSM-guided generation / grammar-guided decoding / 约束采样
> **难度**: 中（需懂 [[autoregressive decoding]] + [[采样策略]] + [[logits generation]] + [[tokenization]]）


## 1. 一句话定义

**constrained decoding（约束解码）** 是 LLM [[autoregressive decoding|自回归生成]] 时的一种**采样期干预技术**——在每一步对词表做采样之前，根据某个外部约束（语法 / 格式 / 词表子集），把"违反约束的 token"的 **logit 置为 $-\infty$**，从而令其 softmax 概率严格为 0，使模型**只能在合法 token 子集中采样**，最终输出 **100% 满足约束**的序列。它是 [[structured output]]（结构化输出）的**底层技术手段总称**，而 structured output 是它的**应用目标**（保证输出是合法 JSON / 正则 / EBNF / schema）。在 vLLM / SGLang 等推理引擎里，它以 **logits processor 钩子** 的形式挂进 [[logits generation]] 与 [[采样策略]] 之间。

> [!note] 三句话定位
> - **是什么**：采样前把违反约束的 token logit 置 $-\infty$，令其概率为 0，模型只能在合法集内采。
> - **为什么**：原生 LLM 不保证输出合法 JSON/正则/格式，事后 `json.loads` 解析常崩；约束解码在**生成期**就杜绝非法 token，做到 100% 合法。
> - **与 [[structured output]] 的区分**：constrained decoding 是**技术手段**（logit mask / FSM / grammar engine），structured output 是**能力目标**（输出符合 schema）。前者实现后者。

> [!warning] 误区：constrained decoding ≠ beam search
> 约束解码**不是**搜索算法的变种，它不改变"采哪个 token"的搜索策略（greedy / sample / beam 都可以），而是**在搜索前缩小候选集**。可以理解为：先 mask，再走原有的 [[采样策略]]（temperature / top-k / top-p）或 [[beam search]]。两者正交可叠加。


## 2. 为什么需要它（动机与背景）

### 2.1 原生 LLM 不保证语法合法

LLM 训练目标是"下一个 token 概率最大"，不是"输出合法 JSON"。即便 prompt 明确要求 `请输出 JSON`，模型仍可能产出：

- 漏闭合括号：`{"name": "Alice"` （缺 `}`）
- 字段类型错：`{"age": "二十"}`（schema 要 int）
- 多余文本：`好的，这是结果：{"name":"Alice"}`（前后有自然语言）
- 引号不配对、转义错误、数组逗号缺失……

这些在 `json.loads()` 时直接抛 `JSONDecodeError`，导致下游 **function calling / 工具调用 / API 解析失败**。模型越大越"像"会输出 JSON，但**概率上永远不等于 100%**。

### 2.2 事后修复（post-hoc fixing）脆弱

传统做法是生成完再修补：

```
text = model(prompt)
text = extract_first_json(text)   # 正则抠 {...}
obj = json.loads(text)            # 还可能崩
```

问题：
- **抠取靠正则**，模型若把 JSON 拆进 markdown 代码块、或字段含 `{`，正则就错。
- **修补靠启发式**（补括号、去尾逗号），改坏了语义（`{"age": 20,}` → 补成合法但语义变了）。
- **重试**（fail 了重新采样）浪费算力且不保证收敛。
- **类型约束做不到**：schema 要 `age: int`，模型可能给 `age: "20"`，事后修不了类型。

### 2.3 解码期约束 = 100% 合法

constrained decoding 把约束前移到**生成期**：每一步只允许合法 token 进入采样池。结果是——

- 输出**结构性 100% 合法**（不是"大概率合法"）。
- **零重试、零后处理**，省算力、省延迟。
- **类型安全**：`age` 字段只允许数字 token，不会出字符串。
- 与 [[speculative decoding]]、[[continuous batching]] 等引擎优化正交可叠加。

代价是每步多一次 **mask 计算**（CPU 上跑 FSM / grammar engine，再 GPU 同步），有开销——见第 8 节。


## 3. 核心概念详解

### 3.1 logit mask（核心机制）

回顾 [[autoregressive decoding]]：每步模型输出一个长度为 $|V|$（词表大小，如 128k / 152k）的 logit 向量 $\mathbf{z} \in \mathbb{R}^{|V|}$，[[采样策略]] 对它做 softmax 得概率，再采一个 token。

constrained decoding 在"softmax 之前"插入一步：根据当前约束状态，算出一个 **mask 向量** $\mathbf{m} \in \{0,1\}^{|V|}$（1=合法，0=非法），然后：

$$
\mathbf{z}'_i = \begin{cases} \mathbf{z}_i & m_i = 1 \\ -\infty & m_i = 0 \end{cases}
$$

即合法 token 的 logit 不动，非法 token 的 logit 置 $-\infty$。之后 softmax、top-k、top-p、temperature 一切照旧，只是非法 token 永远采不到。**这是 constrained decoding 的唯一硬核操作**，所有变种（FSM / grammar engine / regex compiler）都只是"怎么算出 $\mathbf{m}$"的不同实现。

```
原始:  logits = model(...)              # [V]
       probs  = softmax(logits)        # 任意 token 可采
约束:  mask   = fsm.legal_tokens(state) # {合法 token id 集合}
       logits = logits.masked_fill(~mask, -inf)
       probs  = softmax(logits)        # 只在合法集内分布
       token  = sample(probs)
       state  = fsm.next(state, token) # 推进状态
```

### 3.2 约束类型

约束可以是任意"对 token 序列的合法性判定"，常见三类：

| 约束类型 | 例子 | 编译成 |
|---|---|---|
| **语法约束** | JSON、正则、EBNF 上下文无关文法 | FSM / CFG → 状态机 |
| **格式约束** | 日期 `YYYY-MM-DD`、电话、IP、枚举 `{red,green,blue}` | 正则 → FSM |
| **词表约束** | 只能从 token id 子集采（如只允许中文 token / 停用词屏蔽） | 直接 set |

语法约束最常见、最复杂，是 [[structured output]] 的主力。词表约束最简单（一个集合），常用于禁词、领域限定。

### 3.3 FSM（有限状态机）驱动

把约束**编译成一个自动机**（DFA / NFA / CFG 状态机），每步：

1. 查当前状态 $s_t$ 下**所有合法转移**（从 $s_t$ 出发能走哪些字符 / token）。
2. 把这些合法字符 / token 映射到词表 id，得 mask。
3. 采样得 token $w_t$。
4. 用 $w_t$ 对应的字符序列**推进状态**：$s_{t+1} = \delta(s_t, \text{decode}(w_t))$。
5. 回到 1，直到接受态或 EOS。

```
约束: "回答只能是 yes 或 no"  →  DFA:
   s0 --y--> s1 --e--> s2 --s--> s3 (accept)
   s0 --n--> s4 --o--> s5 (accept)

生成 step 0 (state=s0): 合法 token = {y, n} 的首 token → mask
  采到 "y" → state = s1
step 1 (state=s1): 合法 = {e} → mask
  采到 "e" → state = s2
... 直到 accept
```

> [!note] FSM 的关键难点：token ≠ 字符
> 自动机的转移是**字符级**的（`y`→`e`→`s`），但 LLM 采的是 **token**（可能多字符，如 `" yes"` 一个 token，或 `"ye"` 一个 token）。所以 FSM 要预计算"从当前状态出发，哪些 **token** 是合法前缀"——这叫 **token 对齐**，是 outlines / xgrammar 的核心工程难点。详见第 8 节。

### 3.4 与 sampling 的关系（temperature / top-k / top-p 仍在合法集内作用）

约束解码**不替代**采样，而是**先约束后采样**：

```
logits ── mask ──> masked_logits ──> temperature ──> top_k ──> top_p ──> multinomial
```

- **temperature**：在合法集内再调温度（合法 token 的 logit 除以 $T$）。
- **top-k**：在合法集内取前 $k$ 个（若合法集 < $k$，则全留）。
- **top-p**：在合法集内做 nucleus。
- **greedy**：合法集内 argmax。

所以 constrained decoding 与 [[采样策略]] **正交叠加**：约束保证合法，采样决定"合法里挑哪个"。这是常见误区——以为约束解码就等于 greedy，其实完全可以约束 + 高温采样（合法集内随机）。

### 3.5 与 beam search 的关系

[[beam search]] 也可以加约束：每步对**每条 beam** 算各自 mask（各 beam 状态可能不同），在合法集内维护 $k$ 条最优 beam。称 **constrained beam search**。开销是 $k$ 倍 mask 计算。推理引擎里少见（beam 慢），多数用 sampling + 约束。

### 3.6 token healing（修复首 token 合法性）

一个相关但反向的小技术：**token healing** 不在生成时约束，而是在生成完一个可能"前缀截断"的 token 后，**回头补全 / 修复**首 token 的合法性。例如 prompt 以 `def foo(` 结尾，模型可能把 `)` 当独立 token 生成，破坏语法；token healing 会把上下文末尾的几个 token "重写"成更合法的开头。多见于 guidance 风格的库，vLLM/SGLang 主线不常用。**与 constrained decoding 互补**：一个是预防（mask），一个是事后修补（healing）。

### 3.7 vLLM / SGLang 的实现（logits processor 钩子）

在 vLLM v1 架构里，constrained decoding 挂在 **logits processor** 这一层——[[model runner]] 算完 logits 后、[[采样策略|sampler]] 采样前，调用一组 `logits_processor` 函数，每个接收 `(token_ids, logits)` 返回修改后的 logits。结构化输出的 backend（xgrammar / outlines / guidance）就是一个 logits processor，内部维护 FSM / grammar state，每步算 mask 并 `logits[~mask] = -inf`。

```python
# vLLM 概念伪码（简化）
class GrammarLogitsProcessor:
    def __init__(self, grammar_engine, tokenizer):
        self.engine = grammar_engine          # xgrammar / outlines engine
        self.state = engine.init_state()
    def __call__(self, token_ids, logits):    # 每个 token 调一次
        mask = self.engine.next_mask(self.state)   # [V] bool
        logits = logits.masked_fill(~mask, -float("inf"))
        return logits
```

SGLang 同理：在 sampler 前有 `LogitsProcessor` 钩子，grammar backend（默认 xgrammar）注册一个 processor。详见 [[structured output]] 第 5 节的 vLLM/SGLang API。


## 4. 数学原理 / 公式

### 4.1 logit mask → 概率为 0 的推导

设原始 logit $\mathbf{z} \in \mathbb{R}^{|V|}$，mask $\mathbf{m}\in\{0,1\}^{|V|}$，masked logit：

$$
z'_i = m_i\, z_i + (1-m_i)\cdot(-\infty)
$$

softmax 后概率：

$$
p_i = \frac{e^{z'_i}}{\sum_j e^{z'_j}}
$$

对非法 token（$m_i=0$）：

$$
p_i = \frac{e^{-\infty}}{\sum_j e^{z'_j}} = \frac{0}{\sum_{j:m_j=1} e^{z_j}} = 0
$$

（因 $e^{-\infty}=\lim_{x\to-\infty}e^x=0$，且分母只剩合法 token 的项，非法项贡献 0。）

对合法 token（$m_i=1$）：

$$
p_i = \frac{e^{z_i}}{\sum_{j:m_j=1} e^{z_j}}
$$

即**在合法集上做了一次 renormalized softmax**——相当于把概率质量全部重新分配给合法 token，比例由原始 logit 决定。**关键结论**：约束不是"截断后不归一"，而是"截断后重新归一"，所以合法 token 的相对概率被放大（原本被非法 token 占的质量回流到合法 token）。

> [!note] 数值实现：用 -inf 还是极小值？
> 工程上 `torch.full(..., -float("inf"))` 在 softmax 时 `exp(-inf)=0` 安全。但若用 `float32` 且整个 batch 某行 mask 全 0（无合法 token，约束矛盾），`softmax` 会出 `nan`。所以引擎需保证**任意状态至少有一个合法 token**（FSM 不能死锁）；xgrammar 等会在 schema 编译期校验可达性。

### 4.2 FSM 状态转移方程

设约束编译成 DFA $M=(Q,\Sigma,\delta,q_0,F)$，$Q$ 状态集，$\Sigma$ 字符表，$\delta:Q\times\Sigma\to Q$ 转移函数，$q_0$ 初态，$F\subseteq Q$ 接受态集。已生成 token 序列 $w_1..w_{t-1}$ 解码为字符序列 $c_1..c_n$，当前状态：

$$
s_t = \delta^*(q_0,\, c_1c_2\cdots c_n) \quad\text{其中}\quad \delta^*(q,\epsilon)=q,\ \delta^*(q,ca)=\delta^*(\delta(q,c),a)
$$

第 $t$ 步合法字符集：

$$
\mathrm{Legal}(s_t) = \{\, c\in\Sigma \mid \delta(s_t,c)\neq \varnothing \,\} \cup \{\,\text{EOS}\mid s_t\in F \,\}
$$

合法 token 集（经 token 对齐）：

$$
\mathrm{LegalTok}(s_t) = \{\, v\in V \mid \mathrm{prefix}(\mathrm{decode}(v)) \in \mathrm{PrefixLang}(s_t) \,\}
$$

其中 $\mathrm{PrefixLang}(s_t)$ 是"从 $s_t$ 出发、存在某条路径接受的所有字符前缀"集合。采样 $w_t\in\mathrm{LegalTok}(s_t)$ 后推进：

$$
s_{t+1} = \delta^*(s_t,\, \mathrm{decode}(w_t))
$$

迭代至 $s_t\in F$ 且采到 EOS。

### 4.3 约束采样分布

最终序列概率（约束下）：

$$
P_{\text{constr}}(w_{1:T}) = \prod_{t=1}^{T} \frac{\mathbb{1}[w_t\in\mathrm{LegalTok}(s_t)]\, e^{z_{t,w_t}/T}}{\sum_{v\in\mathrm{LegalTok}(s_t)} e^{z_{t,v}/T}}
$$

即每步在合法集内做带温度 $T$ 的归一化采样。**与无约束分布的关系**：约束分布是无约束分布在"合法序列子空间"上的**条件分布**——$P_{\text{constr}}(w) \propto P_{\text{free}}(w)\cdot\mathbb{1}[w\text{ 合法}]$，再归一。


## 5. 代码示例（可选）

### 5.1 手写一个最小 logit mask（直觉理解）

不依赖任何库，演示"只允许输出 `yes` 或 `no`"的约束解码。用字符级 FSM + 一个 toy 模型（随机 logit）。

```python
import torch

V = 5                                  # toy 词表: 0=y 1=e 2=s 3=n 4=o
decode = {0:"y",1:"e",2:"s",3:"n",4:"o"}
# DFA: s0 -y-> s1 -e-> s2 -s-> s3(acc) ; s0 -n-> s4 -o-> s5(acc)
trans = {
  0: {"y":1, "n":4},
  1: {"e":2},
  2: {"s":3},
  3: {},     # accept (EOS)
  4: {"o":5},
  5: {},     # accept (EOS)
}
accept = {3, 5}

def legal_mask(state):
    m = torch.full((V,), float("-inf"))
    for c, ns in trans[state].items():
        m[list(decode)[::-1].index(c) if False else
          next(k for k,v in decode.items() if v==c)] = 0.0
    if state in accept:
        m = torch.full((V,), float("-inf"))   # 只允许 EOS（这里用 -1 占位）
    return m

state = 0
out = []
torch.manual_seed(0)
for _ in range(3):
    logits = torch.randn(V)            # 假装是模型输出
    logits = logits + legal_mask(state)
    probs = torch.softmax(logits, dim=0)
    tok = torch.multinomial(probs, 1).item()
    out.append(decode[tok])
    state = trans[state][decode[tok]]
print("".join(out))   # 形如 "yes" 或 "no"，结构 100% 合法
```

> [!tip] 实践：这是教学版，真实引擎要处理 token≠字符、多字节 UTF-8、BPE 合并、EOS、batch 不同状态等，远比这复杂。但核心 = mask + 状态推进，就此一招。

### 5.2 用 outlines 做 FSM 约束（正则 → 合法输出）

outlines 把正则 / JSON / Pydantic / CFG 编译成 FSM，挂进任意 transformers 模型：

```python
# pip install outlines transformers
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(name, device_map="auto"),
    AutoTokenizer.from_pretrained(name),
)

# 正则约束：只输出 email
import re
email_model = model(outlines.types.regex(r"[a-z]+\.[a-z]+@[a-z]+\.com"))
print(email_model("生成一个 Alan Turing 的邮箱"))
# -> alan.turing@enigma.com  (100% 符合正则)

# Pydantic 约束：只输出合法 JSON
from pydantic import BaseModel
class Person(BaseModel):
    name: str
    age: int

p = model("生成一个人的名字和年龄", Person, max_new_tokens=64)
obj = Person.model_validate_json(p)     # 永远不会抛 JSONDecodeError
print(obj.name, obj.age)
```

### 5.3 在 vLLM 里挂 logits processor（底层视角）

```python
# pip install vllm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="HuggingFaceTB/SmolLM2-135M-Instruct")

# 新 API (v0.12.0+)：structured_outputs 参数
sp = SamplingParams(
    temperature=0.0,
    structured_outputs=StructuredOutputsParams(
        regex=r"\w+@\w+\.com"           # 只输出 email
    ),
)
out = llm.generate("Alan Turing 在 enigma 公司的邮箱", sp)
print(out[0].outputs[0].text)
```

> [!warning] 误区：guided_json / guided_regex 已废弃
> vLLM v0.12.0 起移除了旧 API `guided_json` / `guided_regex` / `guided_choice` / `guided_grammar` / `guided_whitespace_pattern` / `structural_tag` / `guided_decoding_backend`。新写法统一用 `StructuredOutputsParams(...)` 或 `extra_body={"structured_outputs": {...}}`。旧教程的 `guided_json=...` 已失效。


## 6. 与其他知识点的关系

- **上游（依赖）**: [[logits generation]]（mask 作用在 logit 上）、[[autoregressive decoding]]（每步采一个 token 才能推进状态）、[[tokenization]]（token 对齐 FSM 的难点所在）、[[采样策略]]（mask 后照旧 softmax/top-k/p）
- **下游（应用）**: [[structured output]]（本技术的应用目标，保证 JSON/正则/schema 合法）、function calling / tool use、RL 中的 reward shaping（约束输出便于解析）、[[speculative decoding]]（约束可叠加在 draft/verify 上）
- **对比 / 易混**:
  - vs [[structured output]]：本条是**技术手段**（logit mask / FSM / grammar engine 的总称），后者是**能力目标**（输出符合 schema）。关系类似"排序算法" vs"得到有序列表"。
  - vs [[beam search]]：本条只缩小候选集，不改搜索策略；beam 是搜索策略。可叠加为 constrained beam search。
  - vs [[speculative decoding]]：speculative 是**加速**（draft 预测 + verify），本条是**约束**（保证合法）。正交，可同时用（draft 也加约束）。
  - vs token healing：本条是**预防**（生成期 mask），healing 是**事后修补**（重写首 token）。互补。


## 7. 常见误区与易错点

> [!warning] 误区1：约束解码 = greedy
> 错。约束只缩小候选集，采样策略照旧。完全可以约束 + temperature=1.0 随机采（合法集内随机）。greedy 只是合法集内 argmax 的一种特例。

> [!warning] 误区2：约束解码能提升模型"质量"
> 不一定。约束只保证**结构合法**，不保证**内容正确**。模型可能输出合法但语义错的 JSON（`{"age": -999}` 合法但荒谬）。语义质量仍靠模型本身 + prompt。

> [!warning] 误区3：mask 全 0 会优雅降级
> 不会，会 `nan`。若约束矛盾（schema 写死、FSM 死锁，某状态无合法 token），softmax 出 nan 导致整 batch 崩。引擎必须在编译期校验可达性 / 提供 fallback（如允许任意 token + 警告）。

> [!warning] 误区4：token 级 mask = 字符级 mask
> 错。FSM 是字符级转移，LLM 采 token（多字符、BPE 合并、UTF-8 多字节）。必须做 **token 对齐**：预计算每个 token 在每个 FSM 状态下是否合法。这是 outlines/xgrammar 的核心工程，不是简单 set 查询。

> [!tip] 实践：约束越紧，多样性越低
> 约束把概率质量重分配给合法集。约束太紧（如 schema 字段极多）→ 合法集极小 → 采样几乎 greedy → 输出高度重复、缺乏创造性。生成代码 / JSON 宜紧，生成创意文本不宜上约束。

> [!note] 补充：EOS 的处理
> FSM 未到接受态时，EOS token 必须 mask 掉（否则模型提前 EOS，输出半截非法 JSON）。到接受态才放开 EOS。这是结构化输出"不提前截断"的关键。


## 8. 延伸细节

### 8.1 mask 计算的 CPU 开销与 GPU sync

每步要：① 在 CPU 上跑 FSM / grammar engine 算 `legal_tokens(state)`（涉及词表遍历、状态查表）；② 把 bool mask 传到 GPU；③ `logits.masked_fill`；④ 采样后把采到的 token 传回 CPU 推进状态。**②④是 CPU↔GPU 往返**，每 token 两次 sync，破坏 [[CUDA Graph执行|CUDA Graph]] 的捕获（Graph 要求静态控制流，mask 每步变）。xgrammar 用 **bitmask + GPU kernel** 把 mask 计算搬上 GPU，降到 near-zero overhead（其论文卖点）。vLLM v1 的 structured output worker 也在 GPU 端维护 grammar state 以减 sync。

### 8.2 token 对齐的预计算

对每个 FSM 状态 $s$，预计算 $\mathrm{LegalTok}(s)$：遍历词表，把每个 token 的字符串喂进 FSM 看能否从 $s$ 出发不撞死。这步 $O(|Q|\cdot|V|\cdot\text{token 长度})$，词表 128k 时一次性预计算几秒，缓存复用。outlines 把这步叫 `create_fsm_index`，xgrammar 用 adaptive mask（只算必要的 token 前缀树）。

### 8.3 约束与 continuous batching 的交互

[[continuous batching]] 的 batch 里每条请求约束不同、状态不同、推进速度不同。每步要对 batch 内每条独立算 mask，再 `scatter` 到 padded logits。vLLM 用 per-request 的 `GrammarLogitsProcessor` 实例，SGLang 同理。约束请求与无约束请求可混 batch（无约束的 mask 全 1）。

### 8.4 CFG 比 DFA 更强（xgrammar / outlines 支持）

正则 → DFA 够用。但 JSON schema、嵌套结构、递归（如"任意深度括号配对"）超出正则表达力，需 **上下文无关文法（CFG）**。xgrammar / outlines / llguidance 都把 JSON schema / EBNF 编译成 CFG 的 pushdown 自动机变体或其近似。这是为什么 vLLM 的 `grammar` 参数比 `regex` 更"强"但也更慢。

### 8.5 与 token healing、guidance 的关系

guidance（微软）库走另一路：用"闭标签"语法 + token healing，部分场景更灵活。vLLM v0.12+ 把 `guidance` / `llguidance` 作为可选 backend（与 xgrammar / outlines / lm-format-enforcer 并列，经 `--structured-outputs-config.backend` 选）。

### 8.6 SGLang 的实现（待核实具体参数名）

SGLang 的 `SamplingParams` 支持 `json_schema` / `regex` / `ebnf`（EBNF 文法）/ `choice` 等字段（参数名以官方文档为准，待核实），server 端 OpenAI 兼容接口经 `extra_body` 传。默认 grammar backend 为 **xgrammar**（2024/11 起 xgrammar 官方宣布集成进 SGLang），也可选 outlines / llguidance。底层同样是 logits processor 钩子，挂进 [[model runner]] 的 sampler 前。

---
相关: [[structured output]] | [[采样策略]] | [[autoregressive decoding]] | [[logits generation]] | [[beam search]] | [[speculative decoding]] | [[tokenization]] | [[vLLM scheduler]] | [[model runner]] | [[OpenAI API server]]
