# structured output

> **所属章节**: [[constrained decoding]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: structured output / 结构化输出 / structured generation / guided decoding / guided generation / grammar-constrained generation / schema-constrained output
> **难度**: 中（需懂 [[constrained decoding]] + [[autoregressive decoding]] + [[logits generation]] + [[tokenization]]）


## 1. 一句话定义

**structured output（结构化输出）** 是 LLM 推理的一项**能力目标**——让模型输出**严格符合**预定义结构（JSON、正则、EBNF 上下文无关文法、JSON Schema、Pydantic 模型），做到 100% 可被下游 `json.loads()` / 正则 / 解析器直接消费、零解析失败。它通过 [[constrained decoding]]（约束解码：每步采样前用 logit mask 把违反语法的 token 概率置 0）实现，主流实现库为 **xgrammar**（vLLM / SGLang / TensorRT-LLM 默认后端）、**outlines**（dottxt-ai，FSM 编译）、以及 **guidance / llguidance / lm-format-enforcer** 等 backend。在 vLLM / SGLang 中以 `structured_outputs` / `response_format` / `extra_body` 接口暴露。

> [!note] 三句话定位
> - **是什么**：让 LLM 输出 100% 符合 JSON / 正则 / EBNF / schema 的能力，靠 [[constrained decoding]] 的 logit mask 实现。
> - **为什么**：API / 工具调用要 JSON，原生 LLM 不保证合法，`json.loads` 常崩；结构化输出在生成期杜绝非法 token。
> - **与 [[constrained decoding]] 的关系**：后者是**技术手段**（logit mask / FSM / grammar engine），本条是**应用目标**（输出符合 schema）。手段服务目标。

> [!warning] 误区：structured output ≠ 普通采样 + 后处理
> 它不是"生成完再 `json.loads` + 修补"，而是**生成期**就用 mask 杜绝非法 token。前者概率性合法，后者结构性 100% 合法。两者代价不同：前者快但脆，后者每步多 mask 开销但稳。


## 2. 为什么需要它（动机与背景）

### 2.1 工具调用 / API 集成要 JSON

LLM 做 [[OpenAI API server|function calling]] / tool use / agent 时，输出要被下游代码 `json.loads` 成对象再 dispatch 到函数。任何 JSON 语法错（漏括号、尾逗号、引号不配、类型不符）都让调用链断。生产环境不能容忍"99% 合法"，要 100%。

### 2.2 原生模型不保证语法合法

LLM 训练目标是 token 概率，不是"合法 JSON"。即便 prompt 写"请输出 JSON"，模型仍可能：
- 漏闭合：`{"name":"Alice"`
- 类型错：schema 要 `int`，给 `"二十"`
- 夹自然语言：`好的：{"name":"Alice"}`
- 转义错、数组逗号缺……

模型越大越"像"会输出 JSON，但**概率上永远 < 100%**，长尾 case 总会崩。

### 2.3 事后修复脆弱、重试浪费

生成完修补（正则抠 + 补括号 + 去尾逗号）改坏语义、修不了类型错；失败重试浪费算力且不保证收敛。结构化输出把约束前移到生成期——**零后处理、零重试、100% 合法**。

### 2.4 类型安全与 schema 一致

不仅保证"是 JSON"，还保证"字段名 / 类型 / 枚举值 / 必填 / 嵌套结构"全符合 schema。如 `age:int` 字段只允许数字 token，`status:Enum[active,inactive]` 字段只允许这两串。这是普通采样做不到的。


## 3. 核心概念详解

### 3.1 什么是结构化输出（强制模型输出符合语法）

结构化输出 = **生成期约束** + **语法编译器**：把用户给的 schema / 正则 / EBNF 编译成状态机（FSM / CFG engine），挂进推理引擎的 logits processor，每步算 mask 杜绝非法 token。最终输出结构性 100% 合法。支持的结构类型：

| 结构类型 | 输入形式 | 编译目标 |
|---|---|---|
| **JSON** | JSON Schema（手写或 `pydantic.model_json_schema()`） | CFG / FSM |
| **正则** | `r"\w+@\w+\.com"` | DFA（正则→自动机） |
| **EBNF 文法** | 上下文无关文法字符串 | CFG engine |
| **枚举/choice** | `["positive","negative"]` | 简单 FSM |
| **structural_tag** | 在 `<tag>...</tag>` 内套 schema | 混合 FSM |

### 3.2 实现机制①：FSM / logit mask

最基础路线（outlines 原始论文 Willard & Louf 2023）：把正则 / JSON schema 编译成**有限状态机**，每步：
1. 查当前状态 $s_t$ 的合法转移字符集；
2. 经 **token 对齐**（处理 token≠字符）映射到词表 id，得 mask；
3. `logits[~mask] = -inf`；
4. 在合法集内做 [[采样策略]]（temperature/top-k/p）；
5. 采到 token 后推进状态 $s_{t+1}=\delta(s_t,\text{decode}(w_t))$。

数学与状态方程详见 [[constrained decoding]] 第 4 节。

### 3.3 实现机制②：xgrammar（CFG → GPU grammar engine）

xgrammar（MLC-ai）把 **JSON Schema / EBNF / 正则**编译成**上下文无关文法引擎**，支持正则表达不了的递归嵌套（任意深度 JSON 数组/对象）。核心优化：
- **adaptive bitmask**：只对必要 token 算前缀合法性，避免遍历全词表；
- **GPU kernel**：mask 计算与 apply 搬上 GPU，减 CPU↔GPU sync，做到 JSON 生成 **near-zero overhead**（其论文卖点）；
- **预编译**：schema 编译期算一次 token 对齐表，运行时只查表 + 推进状态。

xgrammar 是 vLLM / SGLang / TensorRT-LLM / MLC-LLM 的**默认**结构化生成后端。

### 3.4 实现机制③：outlines（正则 / JSON → FSM）

outlines（dottxt-ai）走 FSM 路线，论文 *"Efficient Guided Generation for LLMs"*（arXiv:2307.09702）系统化了"正则/JSON→DFA→token 对齐表→logit mask"流程。特点：
- **统一接口** `model(prompt, output_type)`：`output_type` 可以是 `Literal[...]`、`int`、pydantic `BaseModel`、正则、CFG——outlines 自动编译成 FSM 挂进模型；
- **模型无关**：同一份代码跑 transformers / llama.cpp / vLLM / Ollama / OpenAI / Gemini；
- 也是 vLLM 的可选 backend 之一（`--structured-outputs-config.backend=outlines`）。

### 3.5 实现机制④：生成时增量推进语法状态

无论 FSM 还是 CFG engine，每步都要**用刚采的 token 推进状态**。这是有状态的——同一条请求的 grammar state 跨 token 累积。在 [[continuous batching]] 里，batch 中每条请求各持一个 state，独立推进，互不干扰。state 跨 iteration 持久化在 worker 上（见 [[model runner]]）。

### 3.6 与 grammar 的关系

"grammar"（文法）是结构化输出的**描述语言**——EBNF、JSON Schema、正则都是 grammar 的不同表达。结构化输出的本质 = "把 grammar 编译成解码期约束"。vLLM 的 `grammar` 参数专门接收 EBNF 字符串（最通用、最强，但也最难手写），而 `json` / `regex` 是 grammar 的特例（更易用）。

### 3.7 vLLM / SGLang 的 guided decoding 接口

**vLLM**（v0.12.0+ 新 API，旧 `guided_*` 已移除）：

```python
# Offline
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
sp = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json=MyPydantic.model_json_schema(),   # 或 regex= / choice= / grammar= / structural_tag=
    ),
)
llm.generate(prompt, sp)

# Online (OpenAI 兼容 server)
client.chat.completions.create(
    ..., extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com"}},
    # 或 response_format={"type":"json_schema","json_schema":{"name":..,"schema":..}}
)
```

启动时选 backend：`vllm serve ... --structured-outputs-config.backend=auto`（默认 auto，按请求自动选；可选 `xgrammar` / `guidance` / `outlines` / `lm-format-enforcer`）。

> [!warning] 误区：还在用 guided_json
> vLLM v0.12.0 起移除 `guided_json` / `guided_regex` / `guided_choice` / `guided_grammar` / `guided_whitespace_pattern` / `structural_tag` / `guided_decoding_backend`。旧教程代码会报错，改用 `StructuredOutputsParams(...)` 或 `extra_body={"structured_outputs":{...}}`。

**SGLang**（参数名以官方文档为准，待核实）：`SamplingParams` 支持 `json_schema` / `regex` / `ebnf` / `choice` 等字段；server OpenAI 兼容接口经 `extra_body` 传。默认 grammar backend = **xgrammar**（2024/11 起 xgrammar 官方宣布集成），可切 outlines / llguidance。

### 3.8 性能影响（mask 计算开销、GPU sync）

- **CPU 开销**：每步在 CPU 跑 FSM/grammar engine 算 mask（xgrammar 用 GPU kernel 缓解）。
- **GPU sync**：mask 传 GPU、token 传回 CPU 推进状态，每步两次 CPU↔GPU 往返，破坏 [[CUDA Graph执行|CUDA Graph]] 捕获（控制流每步变）。xgrammar 的 GPU-side state 维护就是为减这个 sync。
- **吞吐**：JSON 生成 xgrammar 已做到 near-zero overhead（其论文）；复杂 CFG / 大 schema 仍有可见降速（10–30% 量级，待核实具体数）。
- **首 token 延迟**：schema 编译 + token 对齐表预计算一次性开销（几秒级，cache 复用）。

### 3.9 与 [[constrained decoding]] 的关系（技术手段 vs 应用目标）

[[constrained decoding]] 是**技术手段总称**（logit mask / FSM / grammar engine 的统称），structured output 是**应用目标**（输出符合 schema）。关系类似"排序算法" vs "得到有序列表"——前者是实现后者的方法。在 vLLM 文档里两者常混用（"structured outputs" = "constrained generation"），但概念上：结构化输出 = 约束解码 + 语法编译器 + 引擎接口封装。


## 4. 数学原理 / 公式

结构化输出的数学本质是 [[constrained decoding]] 第 4 节的**约束采样分布**，这里从"结构合法性"角度再写一遍。

### 4.1 合法序列集与条件分布

设 grammar $G$ 定义了合法字符串集合 $\mathcal{L}(G)\subseteq\Sigma^*$（如"所有合法 JSON"）。LLM 词表 $V$，token 序列 $w_{1:T}$ 解码为字符串 $\mathrm{decode}(w_{1:T})$。结构化输出要求：

$$
\mathrm{decode}(w_{1:T}) \in \mathcal{L}(G)
$$

通过每步 logit mask 实现的条件采样分布：

$$
P_{\text{struct}}(w_{1:T}) = \prod_{t=1}^{T} \frac{\mathbb{1}[\,\mathrm{decode}(w_{1:t})\text{ 是 }\mathcal{L}(G)\text{ 中某串的合法前缀}\,]\,e^{z_{t,w_t}/\tau}}{\sum_{v:\,\mathrm{decode}(w_{1:t-1}v)\text{ 合法前缀}} e^{z_{t,v}/\tau}}
$$

其中 $\tau$ 是 temperature，$\mathbb{1}[\cdot]$ 是指示函数（合法前缀才放行）。关键：不是"采完再判合法"，而是"每步只采能延展成合法串的前缀 token"。

### 4.2 logit mask → 概率为 0（同 [[constrained decoding]]）

mask $\mathbf{m}\in\{0,1\}^{|V|}$，$z'_i=m_i z_i+(1-m_i)(-\infty)$，softmax：

$$
p_i=\frac{e^{z'_i}}{\sum_j e^{z'_j}}\quad\Rightarrow\quad p_i=0\ \text{当}\ m_i=0
$$

（$e^{-\infty}=0$，非法 token 概率严格 0；合法 token 概率在合法集上 renormalize。）推导细节见 [[constrained decoding]] §4.1。

### 4.3 FSM / CFG 状态转移

把 $G$ 编译成自动机 $M$，状态 $s_t$ 编码"已生成前缀对应的语法位置"，合法 token 集：

$$
\mathrm{LegalTok}(s_t)=\{v\in V\mid \mathrm{decode}(v)\text{ 可从 }s_t\text{ 出发延展成 }\mathcal{L}(G)\text{ 中串的前缀}\}
$$

采样 $w_t$ 后推进 $s_{t+1}=\delta(s_t,\mathrm{decode}(w_t))$，至接受态放行 EOS。状态方程同 [[constrained decoding]] §4.2。CFG 比 DFA 强：能处理递归嵌套（JSON 对象套数组套对象），代价是状态机更复杂、mask 计算更贵。


## 5. 代码示例（可选）

### 5.1 vLLM：用 Pydantic schema 约束输出 JSON（在线 server）

```python
# 启动: vllm serve HuggingFaceTB/SmolLM2-1.7B-Instruct \
#         --structured-outputs-config.backend=xgrammar
from openai import OpenAI
from pydantic import BaseModel, Field
from enum import Enum

client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
model = client.models.list().data[0].id

class CarType(str, Enum):
    sedan = "sedan"; suv = "SUV"; truck = "Truck"; coupe = "Coupe"

class Car(BaseModel):
    brand: str
    model: str
    car_type: CarType
    price_usd: int = Field(ge=0)

completion = client.chat.completions.create(
    model=model,
    messages=[{"role":"user","content":"生成一辆 90 年代标志性车的信息"}],
    response_format={                       # OpenAI 风格 json_schema
        "type": "json_schema",
        "json_schema": {"name": "car", "schema": Car.model_json_schema()},
    },
)
car = Car.model_validate_json(completion.choices[0].message.content)  # 100% 合法
print(car.brand, car.car_type, car.price_usd)
```

### 5.2 vLLM：用 regex / grammar（EBNF）（在线 server）

```python
# 正则：只输出 email
client.chat.completions.create(
    model=model,
    messages=[{"role":"user","content":"生成 Alan Turing 在 enigma 的邮箱"}],
    extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com"}},
)

# EBNF 文法：简化 SQL
sql_grammar = '''
root ::= select_statement
select_statement ::= "SELECT " column " from " table
column ::= "col_1 " | "col_2 "
table ::= "table_1 " | "table_2 "
'''
client.chat.completions.create(
    model=model,
    messages=[{"role":"user","content":"生成查询 username,email 的 SQL"}],
    extra_body={"structured_outputs": {"grammar": sql_grammar}},
)
```

### 5.3 vLLM：离线（StructuredOutputsParams）

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
sp = SamplingParams(
    temperature=0.0,
    structured_outputs=StructuredOutputsParams(choice=["Positive","Negative"]),
)
print(llm.generate("分类情感：vLLM 真棒！", sp)[0].outputs[0].text)
```

### 5.4 outlines：直接挂 transformers 模型

```python
# pip install outlines transformers
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal
from pydantic import BaseModel

name = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(name, device_map="auto"),
    AutoTokenizer.from_pretrained(name),
)

# 枚举
print(model("情感？'太好用了'", Literal["Positive","Negative","Neutral"]))
# -> "Positive"

# Pydantic JSON
class Person(BaseModel):
    name: str
    age: int
p = model("生成一个人", Person, max_new_tokens=64)
obj = Person.model_validate_json(p)   # 永远不抛 JSONDecodeError
```

### 5.5 xgrammar：底层直接用（教学，看 mask 怎么来）

```python
# pip install xgrammar
import xgrammar as xgr
# 编译 JSON schema → grammar engine → matcher（每步给 mask）
# 实际多由 vLLM/SGLang 内部调用，用户层很少直接写。这里示意 API 形态。
compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))
compiled = compiler.compile_json_schema(Person.model_json_schema())
matcher = xgr.GrammarMatcher(compiled)
# 生成循环里：mask = matcher.find_next_token_bitmask(); logits[mask==0]=-inf;
#             采 token; matcher.accept_token(tok)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[constrained decoding]]（本能力的底层技术）、[[logits generation]]（mask 作用对象）、[[autoregressive decoding]]（逐 token 推进状态）、[[tokenization]]（token 对齐 FSM 的工程难点）
- **下游（应用）**: [[OpenAI API server|function calling / tool use]]（输出 JSON 给下游 dispatch）、agent 系统（结构化规划）、RL 训练的 reward 解析（约束输出便于打分）、数据合成（批量生成结构化训练数据）
- **对比 / 易混**:
  - vs [[constrained decoding]]：本条是**应用目标**（输出符合 schema），后者是**技术手段**（logit mask/FSM 总称）。本条 = 后者 + 语法编译器 + 引擎接口。
  - vs 普通采样 + 后处理：前者概率性合法、脆；本条结构性 100% 合法、稳，代价是每步 mask 开销。
  - vs [[speculative decoding]]：speculative 是**加速**，本条是**约束**。正交可叠加（draft 也加约束）。
  - vs [[beam search]]：beam 是搜索策略，不保证合法；可叠加约束成 constrained beam search。


## 7. 常见误区与易错点

> [!warning] 误区1：结构化输出保证"内容正确"
> 错。只保证**结构合法**（是合法 JSON / 符合 schema 字段名类型），不保证**语义正确**。`{"age":-999}` 合法但荒谬。语义质量仍靠模型 + prompt。

> [!warning] 误区2：structured output 与 constrained decoding 是同义词
> 概念上不同：constrained decoding 是**技术手段**（logit mask / FSM / grammar engine 的统称），structured output 是**应用目标**（输出符合 schema）。引擎文档常混用，但严谨说"结构化输出 = 约束解码 + 语法编译 + 接口封装"。详见 [[constrained decoding]] §3 与本条 §3.9。

> [!warning] 误区3：用旧的 guided_json 参数
> vLLM v0.12.0 起移除 `guided_json` / `guided_regex` / `guided_choice` / `guided_grammar` / `guided_whitespace_pattern` / `structural_tag` / `guided_decoding_backend`。新代码用 `StructuredOutputsParams(...)` 或 `extra_body={"structured_outputs":{...}}`，启动用 `--structured-outputs-config.backend=...` 选后端。

> [!warning] 误区4：正则用 Python re 语法
> 不同 backend 正则方言不同：xgrammar / guidance / outlines 用 **Rust 风格正则**（`regex` crate，不支持回溯、反向引用），lm-format-enforcer 用 Python `re`。跨 backend 迁移正则可能失效（如 `\d` vs `[0-9]`、lookahead 不支持）。vLLM 文档明确提示这点。

> [!tip] 实践：prompt 里仍要写清 schema
> 即使用结构化输出，prompt 里**明示字段含义 / 取值范围 / 示例**仍能显著提升**内容**质量（结构由 mask 保证，内容靠模型理解）。vLLM 文档也建议："normally it's better to indicate in the prompt the JSON schema and how fields should be populated."

> [!note] 补充：reasoning 模型 + 结构化输出
> vLLM 支持在 reasoning（如 DeepSeek-R1）输出后接结构化输出：reasoning 内容进 `reasoning` 字段，`content` 部分受 schema 约束。需 `--reasoning-parser` + 结构化参数。Qwen3-Coder 等模型需显式 `--structured-outputs-config.enable_in_reasoning=True`（v0.11.2+）。


## 8. 延伸细节

### 8.1 xgrammar vs outlines 对比

| 维度 | xgrammar | outlines |
|---|---|---|
| 出品 | MLC-ai（陈天奇团队） | dottxt-ai（.txt） |
| 编译目标 | CFG engine（支持递归嵌套） | FSM（DFA，正则/JSON schema） |
| 性能优化 | adaptive bitmask + GPU kernel，JSON near-zero overhead | token 对齐表预计算（`create_fsm_index`） |
| 表达力 | EBNF / JSON schema / 正则 / 递归 | 正则 / JSON schema / Pydantic / CFG（部分） |
| 核心论文 | XGrammar（MLSys 2025）/ XGrammar-2（2026） | Willard & Louf 2023（arXiv:2307.09702） |
| vLLM backend 名 | `xgrammar`（默认之一） | `outlines` |
| 用户层 API | 多由引擎内部调用 | `model(prompt, output_type)` 独立可用 |
| 模型无关 | 经引擎集成 | 原生支持 transformers/llama.cpp/vLLM/Ollama/OpenAI/Gemini |
| 语言 | C++ + Python（+ JS/Swift） | Python |

两者都做"语法→状态机→logit mask"，差异在编译目标（CFG vs DFA）、优化重心（GPU kernel vs 预计算表）、与用户层 API 距离。xgrammar 更"引擎内核"，outlines 更"用户框架"。

### 8.2 backend 选型（vLLM `--structured-outputs-config.backend`）

- `auto`（默认）：按请求类型自动选（JSON 倾向 xgrammar）。
- `xgrammar`：JSON / EBNF / 正则，GPU 优化好，vLLM/SGLang/TensorRT-LLM 默认。
- `guidance` / `llguidance`：微软 guidance 底层，支持 structural_tag、token healing 风格。
- `outlines`：FSM，模型无关，正则友好。
- `lm-format-enforcer`：Python `re` 风格正则，约束最松但兼容性高。

### 8.3 structural_tag：在自由文本里嵌套结构化片段

`structural_tag` 允许模型输出自由文本，但遇到指定标签（如 `<answer>...</answer>`）时，标签内内容受 schema 约束。适合"先思考再结构化作答"场景。vLLM v0.12+ 经 `StructuredOutputsParams(structural_tag=...)` 或 `extra_body={"structured_outputs":{"structural_tag":...}}` 配置。

### 8.4 与 [[continuous batching]] / [[vLLM scheduler]] 的交互

每条请求 schema 不同、grammar state 独立推进。[[vLLM scheduler]] 组 batch 时，约束请求与无约束请求可混（无约束的 mask 全 1）。[[model runner]] 在 sampler 前调一组 per-request 的 `GrammarLogitsProcessor`，各自算 mask 再 scatter 到 padded logits。state 跨 iteration 持久化在 worker 上，preempt 重算时 state 也要重置（与 KV 重算同步）。

### 8.5 与 [[speculative decoding]] 叠加

draft model 与 target model 都可挂约束（draft 用简单 mask，target verify 时也 mask）。难点：draft 的 token 序列要同时通过 grammar 约束与 target 概率 verify。xgrammar + eagle/vLLM spec decode 已有支持（细节待核实）。

### 8.6 不确定 / 待核实

- SGLang `SamplingParams` 的具体字段名（`json_schema` / `regex` / `ebnf` / `choice`）与 server `extra_body` 字段名，以 SGLang 官方文档为准（本次 `docs.sglang.ai/structured_outputs.html` 抓取 404，未能核实）。
- 复杂 CFG / 大 schema 的具体降速比例（10–30% 为经验量级估计，未实测）。

---
相关: [[constrained decoding]] | [[采样策略]] | [[autoregressive decoding]] | [[logits generation]] | [[beam search]] | [[speculative decoding]] | [[tokenization]] | [[vLLM scheduler]] | [[model runner]] | [[OpenAI API server]]
