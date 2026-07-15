# 多轮Agent rollout与tool calling

> **所属章节**: [[Agent rollout]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: multi-turn agent rollout / 多轮智能体采样 / tool env action loop / agentic rollout / tool-augmented RL rollout / ReAct rollout / Toolformer-style RL / 函数调用强化学习 / turn-based rollout / 环境-交互-采样循环
> **难度**: 高（需懂 [[trajectory generation]]、[[rollout worker]]、[[autoregressive decoding]]、[[verifier与function reward]]、[[synchronous与asynchronous rollout]]、[[rollout train reference logprob一致性]]、[[采样策略]]、[[constrained decoding]]、[[RL角色拓扑]]、[[GRPO]]）

> **所属小节**: §87


## 1. 一句话定义

**多轮 Agent rollout 与 tool calling** 是 [[17-RL训推一体框架]] 中把 RL 采样从**单轮**（`prompt→response→reward`，LLM 一次性生成完整 response、算一次 reward）演进到**多轮交互式**的范式——Agent（被训练的 LLM）在一条 trajectory 内**分多步与环境交互**：每步 LLM 基于当前上下文生成一段**推理（reasoning）+ 一个动作（action/tool call）**，环境（tool env）执行该动作（调搜索/计算器/代码执行/API/数据库）、返回一个 **observation（观测）**，LLM 把 observation 拼进上下文继续推理下一步，如此**action loop 循环**直到任务完成（给最终答案）或达步数上限；rollout 的产物从"一段 response 序列"变成"一条**多轮 trajectory**"（含每步的 reasoning、tool call、env observation、中间状态），其中**只有 LLM 生成的 token 参与 policy logp 与梯度**（observation 是环境给的、不来自 $\pi_\theta$，要 mask 掉），reward 来源也多元化（最终任务成功 / 中间步反馈 / 过程 reward）。它是 **ReAct**（Reason+Act，Yao et al. 2022）/ **Toolformer**（Schick et al. 2023）范式从**推理时增强**升级到**用 RL 训练**的工程化落地——不再只是推理时让 LLM 调工具，而是**用 RL 优化 LLM 何时调、调什么、怎么用 observation 推理**。与单轮 RL 的本质区别：rollout 不再是**纯 GPU 自回归解码**（[[autoregressive decoding]] 一次 forward 出一串 token），而是**GPU 生成 + CPU/外部环境执行 交替**的混合循环——LLM 生成 action 后要**等环境返回 observation** 才能继续，环境可能是网络请求（搜索 API）、代码沙箱（[[sandbox与轨迹回放]]）、数据库查询，**非纯 GPU 算力**，引入**等待延迟、并发管理、超时、状态隔离**等工程挑战。verl、OpenRLHF、RAGEN 等框架已陆续支持多轮 Agent rollout。

> [!note] 三句话定位
> - **是什么**：单轮 `prompt→response→reward` 升级为多轮 action loop——LLM 每步生成 reasoning+action（tool call），env 执行返回 observation，LLM 拼 observation 继续推理，循环到任务完成。trajectory 含多步、每步有 tool call/observation/中间推理；只有 LLM 生成的 token 进 logp/梯度，observation 要 mask。
> - **为什么**：Agent 任务（用工具解数学题、修代码 bug、查资料答题、操作网页）需要**长程规划+工具交互**，单轮 RL reward 稀疏（只有最终对错）、且 LLM 无法在生成中获取外部信息/执行计算，必须多轮交互才能完成任务、才能拿到有信号的 reward。
> - **与 [[synchronous与asynchronous rollout]] 关系**：后者讲 rollout↔trainer 的**时序协同**（采与训怎么排队），本条讲**单条 trajectory 内部**的多轮交互循环（一条 trajectory 内 LLM↔env 怎么交替）。两者正交：多轮 rollout 的每条 trajectory 内部是 action loop，多条 trajectory 之间仍可同步/异步采。

> [!warning] arxiv ID 勘误
> 任务描述称"RAGEN arXiv:2504.11442"，**经联网核实有误**——arXiv:2504.11442 实为 *TextArena*（一个文本竞技游戏 benchmark）。RAGEN 的正确编号是 **arXiv:2504.20073**（*RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning*，Wang et al. 2025，提出 StarPO）。本文一律采用正确编号 2504.20073。另其续作 RAGEN-2（*Reasoning Collapse in Agentic RL*，arXiv:2604.06268，ICML 2026 Oral）研究 agent RL 中"看似 entropy 高实则退化为固定模板"的 reasoning collapse。


## 2. 为什么需要它（动机与背景）

### 2.1 单轮 RL 的天花板：reward 稀疏 + 无法交互

单轮 RL（GRPO/PPO 数学题范式）是 `prompt → response（一整段 CoT）→ verifier 判对错 → reward`。这对**纯推理任务**（数学题心算、代码题一次性写完）够用，因为：

- LLM 可以在 response 内部**纯靠 token 生成**完成全部推理（CoT 把答案推出来）；
- reward 虽稀疏（只最终对错）但靠 **group sampling**（GRPO 组内相对 advantage）+ **Dynamic Sampling**（[[sequence packing与动态采样]] 过滤全对/全错组）造出信号。

但**真实 Agent 任务**单轮做不了：

| 任务 | 为什么单轮不行 |
|---|---|
| **查资料答题**（"2024 年诺贝尔物理学奖得主是谁"） | LLM 参数里没有 2024 实时信息，必须**调搜索 API** 拿到 observation 才能答 |
| **复杂数学**（大数因式分解、精确数值计算） | LLM 算不准，必须**调计算器/代码执行**算 observation |
| **修代码 bug**（SWE-bench，给仓库改 bug 跑测试） | 要**跑代码/跑测试**看报错、再改，多轮试错 |
| **操作网页/API**（订机票、查数据库） | 要**调外部 API**，根据返回结果决定下一步 |
| **多步规划**（迷宫、Sokoban、交互式对话博弈） | 环境状态随动作变化，必须**一步步交互**才能推进 |

这些任务的共同点：**LLM 无法在生成 response 时凭空获得外部信息/执行外部计算**，必须**中断生成 → 调工具 → 拿 observation → 继续生成**。这正是 ReAct/Toolformer 范式的动机——把 LLM 从"封闭 token 生成器"变成"能调工具的 agent"。

### 2.2 从"推理时调工具"到"用 RL 训练调工具"

ReAct/Toolformer 最初是**推理时**的增强——给定一个**已经会调工具**的 LLM（靠 SFT 教会它输出 `<tool>search("query")</tool>` 这种格式），推理时解析 tool call、执行、拼 observation。但问题是：

- **什么时候该调工具**、**调什么工具**、**怎么用 observation 推理**——这些**策略**靠 SFT 教不好（SFT 只能模仿示范轨迹，没法优化"调对工具能解任务"这个目标）；
- SFT 示范数据贵且有限，模型容易**该调不调**（直接 hallucinate 答案）或**乱调**（调无关工具浪费步数）。

**RL 的引入**：把"用工具解任务"的**成功**当 reward，让模型自己探索"怎么调工具最有效"，用 policy gradient 优化调工具的策略。这就是多轮 Agent rollout 的本质——**rollout 不再是生成一段文本，而是跑一个完整的 agent episode**（在环境里多轮交互、调工具、拿结果），用 episode 的成败算 reward 训 policy。

### 2.3 为什么 reward 仍稀疏但必须多轮

即便多轮，**reward 常仍是稀疏的**（只有最终任务成功 $r=1$，失败 $r=0$）。为什么单轮稀疏 reward 不行、多轮稀疏 reward 可以？

- 单轮稀疏：LLM 一口气生成完，对就是对错就是错，**中间没法纠错**——一次 hallucinate 就整条废，信号噪声比极低；
- 多轮稀疏：LLM 可以**试错**——第 1 步调搜索拿到线索、第 2 步调计算器验证、第 3 步给答案。即便最终失败，**trajectory 里有结构化的中间动作**，可以靠 group sampling（同一任务采多条 trajectory 横向比）+ 过程 reward（PRM 给中间步打分）+ hindsight 等手段**从稀疏最终 reward 提炼中间信号**。

但 RAGEN（arXiv:2504.20073）指出一个反直觉发现：**没有精细的 reasoning-aware reward，纯靠多轮 RL 的最终 outcome reward，agent 的 reasoning 很难涌现**——模型会退化成"浅层策略"（直接猜答案、机械调工具）或"幻觉推理"（reasoning 文本看似合理实则与 observation 无关）。这是多轮 Agent RL 当前的主要瓶颈。

### 2.4 工程挑战总览

多轮 rollout 把单轮"纯 GPU forward"变成"GPU 生成 + 环境执行 交替"的混合循环，引入一堆单轮没有的工程问题：

| 挑战 | 单轮 | 多轮 |
|---|---|---|
| **算力性质** | 纯 GPU（[[autoregressive decoding]]，一次 forward 一串 token） | GPU 生成 + CPU/网络环境执行 交替（非纯 GPU） |
| **等待延迟** | 无（GPU 自己算） | 有（LLM 生成 action 后等环境返回 observation） |
| **并发** | batch 内各样本独立 forward | batch 内各 trajectory 处于不同步、需异步管理 |
| **状态管理** | 无状态（每条独立） | 每条 trajectory 的**环境状态**需隔离管理（[[sandbox与轨迹回放]]） |
| **超时/容错** | 生成完即止 | 某步 tool 超时/报错，trajectory 怎么处理 |
| **trajectory 结构** | `[prompt, response]` 扁平 | 多轮嵌套：每步含 `[reasoning, action, observation]` |
| **logp 记录** | 整段 response 逐 token logp | 只 LLM 生成 token 进 logp，observation 要 mask |
| **显存** | 一条 sequence 的 KV | 长trajectory 的 KV 累积（跨步复用 KV cache） |

这些挑战的具体解法见 §3、§8。


## 3. 核心概念详解

### 3.1 单轮 vs 多轮 rollout 对比

```
单轮 rollout (GRPO 数学题范式):
  prompt = "解 x^2+5x+6=0"
  LLM 一次性生成 response: "因式分解 (x+2)(x+3)=0, x=-2 或 -3. \\boxed{-2,-3}"
  verifier 判 \\boxed{} 对错 -> reward=1
  trajectory = [prompt, response]
  logp 累积: 所有 response token

多轮 Agent rollout (查资料答题范式):
  prompt = "2024 诺贝尔物理学奖得主是谁?"
  step 1:
    LLM 生成: <think>我不确定,需要查</think><tool>search("2024 Nobel Physics")</tool>
    env 执行 search -> observation: "2024 Nobel Physics: Hopfield, Hinton"
  step 2:
    LLM 生成: <think>Hopfield 和 Hinton, 因为研究神经网络</think><answer>Hopfield 与 Hinton</answer>
    verifier 判对 -> reward=1
  trajectory = [prompt, (reason1, action1, obs1), (reason2, action2)]
  logp 累积: 只 reason/action token, obs1 是 env 给的 mask 掉
```

### 3.2 action loop：多轮 rollout 的核心循环

多轮 rollout 的本质是一个 **action loop（动作循环）**，伪代码：

```python
def multi_turn_rollout(llm, env, prompt, max_steps=10):
    trajectory = {"prompt": prompt, "steps": []}
    context = prompt                       # 初始上下文 = prompt
    for step in range(max_steps):
        # 1. LLM 生成: reasoning + action (到 stop token 或 tool call 触发)
        segment = llm.generate(context, stop=["</tool>", "<answer>"])
        action = parse_tool_call(segment)  # 解析出要调的工具
        trajectory["steps"].append({"llm_output": segment, "action": action})
        context += segment
        # 2. 任务完成?
        if is_final_answer(segment):
            break
        # 3. env 执行 action -> observation
        observation = env.execute(action)   # 调搜索/计算器/代码沙箱/API
        trajectory["steps"][-1]["observation"] = observation
        context += observation               # observation 拼进上下文, LLM 下步能看到
    reward = env.reward(trajectory)          # 最终任务成功? 或过程 reward
    return trajectory, reward
```

关键点：
1. **LLM 生成被 stop token 打断**——生成到 `<tool>` 或 `</tool>` 或 `<answer>` 时停，让 env 执行；
2. **env 执行 action 返回 observation**——observation 是**非 LLM 生成**的文本，拼进 context 给下步用；
3. **循环直到 final answer 或 max_steps**——防止死循环。

### 3.3 ReAct / Toolformer 范式

多轮 Agent rollout 的范式来源：

- **ReAct**（Yao et al. 2022, *ReAct: Synergizing Reasoning and Acting in Language Models*）：让 LLM 交替输出 **Thought（推理）→ Action（动作）→ Observation（观测）**。Thought 是 LLM 内部推理，Action 是调工具，Observation 是环境返回。核心思想是"推理指导动作、动作反馈推理"的协同。
- **Toolformer**（Schick et al. 2023）：让 LLM 自学在文本中插入工具调用（`[Search("query")]`），用自监督方式筛选"加了 tool 输出后 perplexity 降"的位置。Toolformer 偏**单轮内嵌 tool call**（生成中穿插调用），ReAct 偏**显式多轮**（每轮一个 thought+action+obs）。

现代多轮 Agent rollout（verl/RAGEN）多采用 **ReAct 式显式多轮**——因为显式分轮便于**逐 step 管理 env 状态、记录 trajectory、mask observation**。但 LLM 的 prompt 里会教它用 ReAct 格式输出：

```
你是一个能调工具的 agent。每步输出:
<think>...推理...</think>
<tool>tool_name(args)</tool>
环境会返回 <observation>...</observation>, 你基于它继续。
最后用 <answer>...</answer> 给最终答案。
```

### 3.4 trajectory 结构：多轮嵌套

单轮 trajectory 是扁平的 `[prompt, response]`，多轮 trajectory 是嵌套的：

```python
trajectory = {
    "prompt": "...",
    "steps": [
        {
            "llm_tokens": [t1, t2, ...],      # LLM 生成的 token (reasoning + action)
            "action": {"tool": "search", "args": "..."},
            "observation_tokens": [o1, o2, ...],  # env 返回, 非 LLM 生成
        },
        {
            "llm_tokens": [...],
            "action": {"tool": "calculator", "args": "..."},
            "observation_tokens": [...],
        },
        # ... 多步
        {
            "llm_tokens": [..., "<answer>...</answer>"],  # 最终答案
            "action": None,
            "observation_tokens": [],
        }
    ],
    "reward": 1.0,
}
```

训练时要把这条嵌套 trajectory **展平成一条 token 序列**送进模型，但用一个 **mask** 标记哪些 token 是 LLM 生成的（参与 logp/梯度）、哪些是 env observation（不参与）：

```
序列:  [prompt] [llm gen step1] [obs1] [llm gen step2] [obs2] ... [llm gen final]
mask:   no       yes              no     yes              no        yes
logp:   skip     累加             skip   累加             skip      累加
```

这是多轮 rollout logp 记录的核心——**observation 不进 logp**，详见 §4。

### 3.5 reward 来源

多轮 rollout 的 reward 比单轮多元：

| reward 类型 | 何时给 | 例子 | 稀疏度 |
|---|---|---|---|
| **最终 outcome reward** | trajectory 结束 | 任务成功 $r=1$，失败 $r=0$ | 极稀疏 |
| **中间步 reward** | 每步 | 调对工具 $+0.1$，调错 $-0.05$ | 中稠密 |
| **过程 reward（PRM）** | 每步推理 | 该步推理是否正确/有效 | 稠密但需 PRM |
| **格式 reward** | 每步 | tool call 格式合法 | 辅助 |
| **步数惩罚** | 结束 | $-0.01 \times \text{steps}$ 鼓励简洁 | 辅助 |

实践中**最终 outcome reward 最常用**（与 GRPO/R1 范式一致，靠 group sampling 造信号），但 RAGEN 指出**纯 outcome reward 不够**——需要 reasoning-aware 的过程 reward 才能让推理真正涌现。中间步 reward 容易引入 reward hacking（模型刷中间步 reward 而非真解任务），需谨慎。详见 [[verifier与function reward]]。

### 3.6 tool env 接口设计

tool env 是 action loop 里"执行 action"的部分，接口设计关键是**function calling schema**：

```python
class ToolEnv:
    """多轮 Agent rollout 的工具环境接口"""
    def __init__(self, tools: dict):
        self.tools = tools   # {"search": search_fn, "calculator": calc_fn, ...}
        self.tool_schemas = [
            {"name": "search", "description": "搜索", 
             "parameters": {"query": "str"}},
            {"name": "calculator", "description": "计算",
             "parameters": {"expr": "str"}},
        ]

    def parse_action(self, llm_output: str):
        """从 LLM 输出解析 tool call (JSON / 特殊 token)"""
        # 方式1: JSON 格式 {"name": "search", "arguments": {"query": "..."}}
        # 方式2: 特殊 token <tool>search("...")</tool>
        match = re.search(r'<tool>(\w+)\((.*?)\)</tool>', llm_output)
        if match:
            return {"name": match.group(1), "args": eval_args(match.group(2))}
        return None   # 没调工具 (可能是最终答案)

    def execute(self, action: dict) -> str:
        """执行 tool, 返回 observation 字符串"""
        if action is None:
            return ""   # 无动作, 不加 observation
        tool_fn = self.tools[action["name"]]
        try:
            result = tool_fn(**action["args"])
            return f"<observation>{result}</observation>"
        except Exception as e:
            return f"<observation>ERROR: {e}</observation>"

    def reward(self, trajectory, ground_truth=None) -> float:
        """从 trajectory 算 reward (最终答案对错 / 过程)"""
        final = extract_answer(trajectory)
        return 1.0 if verify(final, ground_truth) else 0.0
```

**function calling schema** 的两种主流形式：

1. **JSON tool call**（OpenAI function calling 风格，[[OpenAI API server]] 兼容）：LLM 输出结构化 JSON `{"name": "search", "arguments": {...}}`，env 解析 JSON 执行。优点是结构清晰、易解析；缺点是 LLM 要学会输出合法 JSON（格式 reward 辅助）。
2. **特殊 token 触发**（ReAct 风格）：LLM 输出 `<tool>search("query")</tool>`，用 `</tool>` 当 **stop token** 触发生成中断、解析。verl/vLLM 多用这种——stop token 让生成引擎在该处暂停、把控制权交回 rollout loop。详见 §3.7。

### 3.7 stop token 触发生成中断

多轮 rollout 在工程上最关键的点是**如何让 LLM 生成引擎（vLLM/SGLang）在 tool call 处暂停、等 env 执行完再继续**。主流做法是 **stop token**：

```
LLM 生成: <think>要查</think><tool>search("2024 Nobel")
                                         ^ stop token 在 <tool> 之后? 或 </tool>?
```

两种实现：
- **`<tool>` 后即停**：生成到 `<tool>` 就停，rollout loop 解析 `<tool>` 后的参数（要求 LLM 把参数写在 `<tool>` 同一行）、执行、把 `<observation>...</observation>` 拼进 context、续生成。
- **`</tool>` 处停**：LLM 生成完整 `<tool>...</tool>` 块到 `</tool>` 停，解析中间内容执行。更稳（参数完整）但多生成几个 token。

vLLM/SGLang 的 `stop` 参数支持指定停止字符串——生成到该字符串即停（不包含该串在输出里或包含，看配置）。verl 的多轮 rollout 用 vLLM 的 stop 机制实现生成-暂停-续生成循环。**KV cache 跨步复用**：每次续生成不是重新 prefill 整个 context，而是**复用上一步已算的 KV**（上一步的 KV 还在 cache 里），只对新增的 observation + 新生成 token 算。这依赖 vLLM 的 **prefix caching / session-based generation**（同 session 的请求复用前缀 KV）。详见 §8.4、[[autoregressive decoding]]。

> [!tip] 实践：KV cache 跨步复用是多轮 rollout 的吞吐关键
> 多轮 trajectory 越来越长（每步加 reasoning+action+observation），若每步重 prefill 整个 context 会极慢。必须让推理引擎**保留上一步 KV、只增量算新 token**——vLLM 的 `prefix_caching`、SGLang 的 `session-based` 机制。verl 多轮 rollout 显式依赖这个。


## 4. 数学原理 / 公式

### 4.1 多轮 trajectory 的 token 序列与 mask

把多轮 trajectory 展平成一条 token 序列。设 trajectory 有 $K$ 步，每步 LLM 生成 token 集 $A_k$（reasoning+action，action 的 token）、环境返回 observation token 集 $O_k$（$O_K=\emptyset$ 若最后一步给答案不调工具）。展平序列：

$$
\tau = \underbrace{p}_{\text{prompt}} \; \underbrace{A_1}_{\text{LLM gen}} \; \underbrace{O_1}_{\text{env}} \; \underbrace{A_2}_{\text{LLM gen}} \; \underbrace{O_2}_{\text{env}} \; \cdots \; \underbrace{A_K}_{\text{LLM gen}}
$$

定义 mask $m_t$：$m_t=1$ 若 token $t$ 是 LLM 生成的（$A_*$ 中的），$m_t=0$ 若是 env observation（$O_*$ 中的）或 prompt（$p$ 中的）。

### 4.2 policy logp：只对 LLM 生成 token 累积

policy 对该 trajectory 的对数概率——**只对 LLM 实际生成的 token 累加 logp**，observation token 不来自 $\pi_\theta$ 故跳过：

$$
\log \pi_\theta(\tau) = \sum_{t=1}^{|\tau|} m_t \cdot \log \pi_\theta(\tau_t \mid \tau_{<t})
$$

其中 $\tau_{<t}$ 是**整个上文**（含 prompt + 之前所有 LLM 生成 token + 所有 observation token）——即 LLM 生成 $A_k$ 时**能看到**之前所有 observation（这正是多轮交互的意义）。但 logp 只算 LLM 自己生成的 token（mask 把 observation 排除）。

**为什么 observation 不进 logp**：observation 的 token 不是从 $\pi_\theta$ 采样来的（是 env 决定的），它的"概率"与 policy 无关。若把 observation token 也算进 $\log\pi_\theta$，相当于让 policy 对 env 输出"负责"，梯度方向就错了——policy gradient 应只优化**自己能控制的动作**（reasoning+tool call），不优化**环境返回**。这是多轮 RL 与单轮的关键数学差异。

### 4.3 importance sampling ratio：同样只 mask LLM token

PPO/GRPO 的 ratio（详见 [[importance sampling与off-policy correction]]、[[rollout train reference logprob一致性]]）：

$$
\rho_t = \frac{\pi_\theta(\tau_t \mid \tau_{<t})}{\pi_{\theta_{old}}(\tau_t \mid \tau_{<t})} = \exp\big(\log\pi_\theta(\tau_t|\tau_{<t}) - \log\pi_{\theta_{old}}(\tau_t|\tau_{<t})\big)
$$

ratio 同样只在 $m_t=1$ 的 token 上算（observation token 两端 logp 都跳过，ratio 不定义或视作 1）。policy gradient objective：

$$
\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{old}}} \left[ \sum_{t} m_t \cdot \text{clip}\big(\rho_t, 1-\varepsilon, 1+\varepsilon\big) \hat{A}_t \right]
$$

GRPO 版本（[[GRPO]]，去 critic、group-mean baseline）：对同一 prompt 采 $G$ 条多轮 trajectory，advantage $\hat A_i = (r_i - \bar r)/\sigma_r$，每条按上式在 LLM token 上累加 loss。

### 4.4 return / 折扣回报

多步 trajectory 的 return。设第 $k$ 步的 reward $r_k$（常 $r_1=\cdots=r_{K-1}=0$，$r_K$=最终成败，稀疏），折扣 $\gamma$：

$$
G = \sum_{k=1}^{K} \gamma^{k-1} r_k
$$

若只有最终 outcome reward $r_K \in \{0,1\}$，则 $G = \gamma^{K-1} r_K$。折扣 $\gamma<1$ 鼓励**少步数解决**（步数越多折扣越大、return 越小），常配步数惩罚 $r_K \leftarrow r_K - \lambda K$。

advantage（critic 版，PPO）：$\hat A_k = \delta_k + \gamma\delta_{k+1} + \cdots$（GAE），需 critic 估 $V(s_k)$。GRPO 无 critic，用 group-relative $\hat A_i$ 作用整条 trajectory（trajectory-level advantage，不分步）。

> [!note] 补充：trajectory-level vs step-level advantage
> GRPO 的 advantage 是 **trajectory-level**（整条 trajectory 一个 $\hat A$，所有 LLM token 共享），不分步。若要 step-level credit assignment（哪步调对/调错），需 PRM 给每步 reward + critic 估每步 $V$，回到 PPO 式。RAGEN 的 StarPO-S 引入 critic incorporation 就是为做更细的 step-level 信号。纯 trajectory-level + outcome reward 在长 trajectory 上**信用分配困难**（步数多、最终成败难归因到具体步），是多轮 RL 训练难的根因之一。

### 4.5 多轮 vs 单轮 logp 的差异小结

| 维度 | 单轮 | 多轮 |
|---|---|---|
| token 序列 | `[prompt, response]` | `[prompt, A1, O1, A2, O2, ..., AK]` |
| logp 累积 | response 所有 token | 只 $A_*$ token（mask 掉 $O_*$） |
| 上下文依赖 | response token 依赖 prompt+前文 response | $A_k$ 依赖 prompt + 之前所有 $A$ 和 $O$ |
| ratio 范围 | response 所有 token | 只 $A_*$ token |
| advantage 粒度 | sequence/group-level | trajectory-level（GRPO）或 step-level（PPO+PRM） |
| return | $r$（一个） | $\sum \gamma^{k-1} r_k$（多步） |


## 5. 代码示例（可选）

### 5.1 最简多轮 tool env action loop

```python
import re, random

class FakeLLM:
    """模拟一个会 ReAct 格式输出的 LLM (实际用 vLLM/SGLang)"""
    def __init__(self, policy):
        self.policy = policy  # 实际是 vLLM 引擎
    def generate(self, context, stop=None):
        # 模拟: 根据上下文生成一段 (reasoning + tool call 或 final answer)
        # 实际: vLLM.generate(context, stop=["</tool>","<answer>"])
        return self.policy(context, stop)

class ToolEnv:
    def __init__(self):
        self.tools = {"search": self._search, "calculator": self._calc}
    def _search(self, query):
        db = {"2024 nobel physics": "Hopfield, Hinton", "pi": "3.14159"}
        return db.get(query.lower(), "no result")
    def _calc(self, expr):
        try: return str(eval(expr))
        except Exception: return "error"
    def parse(self, text):
        m = re.search(r'<tool>(\w+)\((.*?)\)</tool>', text)
        if m:
            return {"name": m.group(1), "args": eval(f"dict({m.group(2)})")}
        return None
    def execute(self, action):
        if action is None: return ""
        fn = self.tools[action["name"]]
        try: return f"<observation>{fn(**action['args'])}</observation>"
        except Exception as e: return f"<observation>ERROR:{e}</observation>"

def multi_turn_rollout(llm, env, prompt, max_steps=5):
    trajectory = {"prompt": prompt, "steps": [], "context": prompt}
    ctx = prompt
    for step in range(max_steps):
        seg = llm.generate(ctx, stop=["</tool>", "<answer>"])
        # vLLM 在 stop token 处暂停, 返回已生成片段
        seg += "</tool>" if "<tool>" in seg and "</tool>" not in seg else ""  # 补全
        action = env.parse(seg)
        trajectory["steps"].append({"llm_seg": seg, "action": action, "obs": ""})
        ctx += seg
        if "<answer>" in seg:   # 给了最终答案, 结束
            break
        obs = env.execute(action)
        trajectory["steps"][-1]["obs"] = obs
        ctx += obs
    trajectory["context"] = ctx
    return trajectory

# 假装 LLM policy: 看到不懂的就调 search, 第二步给答案
def demo_policy(ctx, stop):
    if "2024" in ctx and "observation" not in ctx:
        return '<think>要查</think><tool>search(query="2024 Nobel Physics")</tool>'
    if "Hopfield" in ctx:
        return '<think>查到了</think><answer>Hopfield 与 Hinton</answer>'
    return '<answer>不知道</answer>'

env = ToolEnv()
llm = FakeLLM(demo_policy)
traj = multi_turn_rollout(llm, env, "2024 诺贝尔物理学奖得主是谁?")
print(traj["context"])
# prompt + <tool>search(...)</tool> + <observation>Hopfield, Hinton</observation> + <answer>...</answer>
```

### 5.2 mask 与 logp 计算（模拟）

```python
import torch

def build_token_sequence_and_mask(trajectory, tokenizer):
    """把多轮 trajectory 展平成 token 序列 + mask (1=LLM gen, 0=env/prompt)"""
    ids, mask = [], []
    # prompt
    p_ids = tokenizer(trajectory["prompt"])["input_ids"]
    ids += p_ids; mask += [0] * len(p_ids)
    for step in trajectory["steps"]:
        # LLM 生成 segment
        a_ids = tokenizer(step["llm_seg"])["input_ids"]
        ids += a_ids; mask += [1] * len(a_ids)
        # env observation
        if step["obs"]:
            o_ids = tokenizer(step["obs"])["input_ids"]
            ids += o_ids; mask += [0] * len(o_ids)   # observation: mask 0!
    return torch.tensor(ids), torch.tensor(mask)

def policy_logp(logits, token_ids, mask):
    """只对 mask=1 的 token 累积 logp"""
    logp = torch.nn.functional.log_softmax(logits[:-1], dim=-1)
    tgt = token_ids[1:]
    gathered = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (gathered * mask[1:].float()).sum()   # 只 LLM token 累加
```

### 5.3 用 vLLM stop token 实现生成-暂停-续生成（verl 式）

```python
# verl 多轮 rollout 的简化版: 用 vLLM 的 stop 参数 + prefix caching
from vllm import LLM, SamplingParams

llm = LLM(model="...", enable_prefix_caching=True)   # 关键: 跨步复用 KV

def vllm_multi_turn(env, prompt, max_steps=5):
    ctx = prompt
    for step in range(max_steps):
        # stop 在 </tool> 或 <answer> 处暂停生成
        sp = SamplingParams(stop=["</tool>", "<answer>"], temperature=0.7, max_tokens=512)
        out = llm.generate([ctx], sp)[0]              # 复用前缀 KV (prefix caching)
        seg = out.outputs[0].text
        seg += "</tool>" if "<tool>" in seg and "</tool>" not in seg else ""
        ctx += seg
        if "<answer>" in seg:
            break
        action = env.parse(seg)
        obs = env.execute(action)
        ctx += obs                                     # observation 拼进, 下次 generate 复用 KV
    return ctx
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[trajectory generation]]（多轮 trajectory 是单轮 trajectory 的多步嵌套）、[[rollout worker]]（rollout 端要管 action loop、tool env、observation 记录）、[[autoregressive decoding]]（每步 LLM 生成仍是自回归，但被 stop token 打断）、[[采样策略]]（多轮 rollout 同一 prompt 采多条 trajectory 做 group sampling）、[[constrained decoding]]（tool call 格式约束可看成一种 constrained decoding）、[[RL角色拓扑]]（rollout 角色要集成 tool env）。
- **下游（应用）**: [[sandbox与轨迹回放]]（多轮 rollout 的 env 状态隔离与 trajectory 回放是其直接配套）、[[verifier与function reward]]（多轮 reward 常用 function reward——tool 执行结果当 reward）、[[synchronous与asynchronous rollout]]（多条多轮 trajectory 间仍可同步/异步采）、[[sequence packing与动态采样]]（多条多轮 trajectory packing 时 per-trajectory 隔离更关键——observation 不能跨 trajectory attend）、[[rollout train reference logprob一致性]]（observation token 不进 logp，但 LLM token 的训推 logp 一致性仍要保）、[[RL worker故障恢复]]（多轮 trajectory 更长更贵，故障恢复要能续 rollout 状态）。
- **对比 / 易混**:
  - **单轮 RL rollout（GRPO 数学题）vs 多轮 Agent rollout**：前者一次生成完、纯 GPU、reward 稀疏靠 group sampling；后者 action loop、GPU+env 交替、observation mask、需 env 状态管理。见 §3.1。
  - **ReAct 推理时 vs 多轮 RL rollout**：ReAct 是推理时调工具（模型已会），多轮 RL 是用 RL 训练调工具策略。前者不更新权重，后者 policy gradient 优化。
  - **[[R3 rollout routing replay]] vs 多轮 trajectory 回放**：前者是 MoE **路由**回放（保训推选同专家），后者是 **env observation** 回放（重算 logp 时还原 observation context）。两者都叫"回放"但对象不同。详见 [[sandbox与轨迹回放]] §3.5。
  - **与 [[OpenAI API server]]**：OpenAI function calling schema 是 tool env 接口的事实标准，多轮 rollout 的 tool call 格式常兼容它。
- **三篇互链**: [[sandbox与轨迹回放]]（多轮 rollout 的 env 隔离与轨迹回放）、[[RL worker故障恢复]]（多轮 rollout 故障更难恢复）。


## 7. 常见误区与易错点

> [!warning] 误区 1：observation token 也进 logp / 梯度
> 错。observation 是**环境返回**的，不来自 $\pi_\theta$，绝不能进 logp 累积或梯度。若把 observation 当 LLM 生成 token 算 logp，policy gradient 会让模型去"拟合环境的输出"（学怎么生成搜索结果），方向完全错。必须 mask 掉 observation token（§4.2）。这是多轮 RL 最易犯的实现 bug——展平 trajectory 时忘了区分 LLM token 和 env token。

> [!warning] 误区 2：多轮 rollout 还是纯 GPU
> 错。单轮 rollout 是纯 GPU 自回归（[[autoregressive decoding]]，一次 forward 一串）。多轮 rollout 在 LLM 生成 action 后要**等 env 返回 observation**——env 可能是网络请求（搜索 API，几十~几百 ms）、代码沙箱（秒级）、数据库查询。**GPU 在等待时空转**，吞吐受 env 延迟拖累。这是多轮 rollout 比 single-turn 慢的根因，需**并发管理**（一批 trajectory 处于不同步，异步推进）+ **prefill/decode 与 env 执行 overlap**。详见 §8.3。

> [!warning] 误区 3：每步重 prefill 整个 context
> 错，极慢。多轮 trajectory 越来越长（每步加 reasoning+action+observation），若每步 LLM 生成时重 prefill 整个 context（O(L) 算力/步），K 步共 O(KL)。必须用 **prefix caching** 让推理引擎**保留上一步 KV、只增量算新 token**——vLLM `enable_prefix-caching`、SGLang session-based。verl 多轮 rollout 强依赖此。详见 §8.4、[[autoregressive decoding]]。

> [!warning] 误区 4：多轮 RL 一定能涌现 reasoning
> 不一定。RAGEN（arXiv:2504.20073）实测：**纯 outcome reward + 多轮 RL，agent reasoning 难涌现**——模型退化成"浅层策略"（直接猜答案）或"幻觉推理"（reasoning 文本与 observation 无关）。需精细的 reasoning-aware 过程 reward（PRM）才涌现。这是当前多轮 Agent RL 的核心瓶颈，不是"上了多轮就行"。

> [!tip] 实践：tool call 格式用 stop token + 特殊 token，别依赖 JSON
> JSON tool call（OpenAI 风格）虽结构清晰，但 LLM 学输出合法 JSON 难、易生成非法 JSON 导致 parse 失败。ReAct 式特殊 token（`<tool>...</tool>`）+ vLLM stop token 更稳——stop token 保证生成在 tool call 边界暂停、parse 简单（正则即可）。verl 多用此。格式合法性可加辅助 format reward 引导。

> [!tip] 实践：env 超时/报错要 graceful 降级
> 某步 tool 超时/报错时，trajectory 不能直接崩——应把错误当 observation 返回给 LLM（`<observation>ERROR: timeout</observation>`），让 LLM 自己决定重试/换工具/放弃。这种 graceful 降级让 agent 学会处理工具故障。但超时步数要计 max_steps 防死循环。

> [!note] 补充：action loop 的"步"不等于"token"
> 一个 step = LLM 生成一段 reasoning+action（可能几十~几百 token）+ env 返回 observation。不是 token 级别的 loop，是**段级别**的 loop。step 数 $K$ 常限制在 5~20（防过长）。return 的 $\gamma$ 折扣按 step 不按 token。


## 8. 延伸细节

### 8.1 verl 的多轮 Agent rollout 支持

verl（volcengine/verl，原 ByteDance）已支持多轮 tool-calling rollout，是社区较早完整支持多轮 Agent RL 的框架。其设计要点（以最新官方文档/代码为准，部分细节待核实）：

- **multi-turn rollout worker**：在 rollout 端集成 tool env，用 vLLM/SGLang 的 stop token 实现"生成-暂停-env 执行-observation 拼接-续生成"循环；
- **数据结构**：多轮 trajectory 用嵌套结构存储（每步的 prompt/response/observation 分段），训练前展平成 token 序列 + observation mask；
- **examples**：verl repo 有多轮/工具调用示例（如 `examples/multi_turn`、tool calling、SWE-bench、search-augmented QA 等，具体路径以最新仓库为准）；
- **KV 复用**：依赖 vLLM `prefix_caching` / SGLang session 机制实现跨步 KV 复用；
- **与 GRPO/PPO 集成**：多轮 trajectory 的 advantage 用 GRPO group-relative 或 PPO critic，logp 只在 LLM token 上算（observation mask）。

具体 API（如 `AgentRolloutWrapper`、`ToolManager` 类名、配置项）随版本变动，建议查 verl 官方文档与 `examples/multi_turn` 目录确认。

### 8.2 OpenRLHF 的多轮支持

OpenRLHF 也在推进多轮/tool-use rollout 支持。其架构（single-controller + Ray 驱动，详见 [[Ray与分布式调度]]）便于集成外部 env——rollout worker 调用 remote env（HTTP/RPC）执行 tool，拿 observation 拼 context 续生成。具体 API 与示例以 OpenRLHF 官方仓库为准（待核实最新进度）。

### 8.3 并发与环境等待：多轮 rollout 的吞吐瓶颈

多轮 rollout 的吞吐瓶颈在**env 等待**而非 GPU。一批 $N$ 条 trajectory 并行采，各处于不同 step（有的在生成、有的在等 env、有的已结束）。若 naive 串行等每条 env 返回，GPU 大量空转。解法：

- **异步推进**：用 event loop / asyncio 管理一批 trajectory 的 state machine——哪条生成完就送 env、哪条 env 返回就续生成，不让 GPU 等 env。类似 [[synchronous与asynchronous rollout]] 的异步思路下放到单 trajectory 内部。
- **batch 生成 + 异步 env**：把"同处生成态"的多条 trajectory 凑一个 batch 送 vLLM 一起 forward（GPU 利用率高），env 调用并发（asyncio.gather）；
- **env 并发限流**：env（如搜索 API）有 QPS 限，需限流+队列；
- **超时与丢弃**：某条 env 超时太长直接丢该 trajectory（不送训练，类似 Dynamic Sampling 过滤）。

### 8.4 KV cache 跨步复用的细节

多轮 trajectory 第 $k$ 步生成时，上文是 `prompt + A1 + O1 + ... + A_{k-1} + O_{k-1} + O_{k-1}`（越来越长）。若每步重新 prefill 整个上文，算力 $O(K^2 L)$（$L$ 平均步长）。用 prefix caching：

- 第 $k$ 步的 KV = 第 $k-1$ 步已算的 KV（prompt+...+O_{k-1} 的 KV 都在 cache）+ 新增 $A_k$ 的 KV；
- 续生成时只对 $A_k$ 的 token 算 KV（$O(L)$/步），共 $O(KL)$。

前提：推理引擎**保留该 session/request 的 KV 不被驱逐**（vLLM 的 prefix caching 按 prefix hash 匹配，需 context 前缀稳定；SGLang 的 session-based 显式保留 session KV）。多轮 rollout 需显式启用此特性。详见 [[autoregressive decoding]]、[[chunked prefill]] 的 KV 管理机制。

### 8.5 长_trajectory 的显存与 logp 记录

多轮 trajectory 长（K 步 × 每步几百 token + observation），单条可达数千~上万 token。显存压力：

- **rollout 端**：vLLM 的 KV cache 要装下整条 trajectory（越长越占），多条并发时 KV cache 紧张；
- **train 端**：重算 logp 时 forward 整条 trajectory，activation 显存随长度涨，需 [[sequence packing与动态采样]] 式 packing + block-diagonal mask 隔离多条 trajectory（observation 不能跨 trajectory attend，否则 logp 算错）；
- **logp 记录**：rollout 时逐 token 记 `old_log_prob`（只 LLM token），observation token 记占位（mask=0）。trajectory 越长，logp tensor 越大，传输/存储开销涨。

### 8.6 与 RAGEN / StarPO 的关系

RAGEN（arXiv:2504.20073，Wang et al. 2025，Stanford/Meta）是专门研究多轮 Agent RL 的系统，核心贡献：

- **StarPO**（State-Thinking-Actions-Reward Policy Optimization）：多轮 trajectory-level agent RL 的通用框架，把 state（observation）、thinking（reasoning）、actions（tool call）、reward 显式建模成多步 MDP 式结构；
- **Echo Trap**：发现多轮 RL 中 reward 方差悬崖 + 梯度尖峰的反复失效模式，用 StarPO-S（trajectory filtering + critic incorporation + gradient stabilization）缓解；
- **关键发现**：纯 outcome reward 不够，需 reasoning-aware 过程 reward 才能让推理涌现；rollout shaping 受初始状态多样性、交互粒度、采样频率影响；
- 续作 RAGEN-2（arXiv:2604.06268，ICML 2026 Oral）揭示 **reasoning collapse**——即使 entropy 高，agent 仍可能退化到固定模板（看似多样实则僵化）。

RAGEN 与 verl/OpenRLHF 的关系：RAGEN 偏**研究诊断**（分析多轮 RL 的失效模式），verl/OpenRLHF 偏**工程框架**（提供可跑的多轮 rollout 基础设施）。两者互补——RAGEN 的发现指导 verl/OpenRLHF 改进多轮训练稳定性。

### 8.7 多轮 RL 的失效模式总结

| 失效模式 | 表现 | 来源 |
|---|---|---|
| **Echo Trap** | reward 方差悬崖、梯度尖峰、训练反复崩 | RAGEN §3 |
| **reasoning collapse** | entropy 高但退化成固定模板 | RAGEN-2 |
| **浅层策略** | 直接猜答案不调工具 | 缺 reasoning-aware reward |
| **幻觉推理** | reasoning 文本与 observation 无关 | 缺过程 reward |
| **tool 滥用** | 乱调工具刷中间步 reward | 中间步 reward 设计不当 |
| **observation 漏 mask** | 把 env token 进梯度 | 实现 bug（§7 误区1） |
| **死循环** | 反复调同一工具不推进 | max_steps 没限 |

### 8.8 与单轮的统一视角

多轮 Agent rollout 可视为单轮 rollout 的**泛化**：单轮 = $K=1$ 步、无 observation（$O_1=\emptyset$）、reward 只 outcome 的特例。多轮的 mask 机制退化为单轮时就是"response 全进 logp"。这样看，[[GRPO]]/[[DAPO]] 是多轮框架在 $K=1$ 的特化，多轮 rollout 把"一条 response"泛化成"一条交互 episode"。统一视角有助于把单轮的优化（group sampling、Dynamic Sampling、clip）迁移到多轮（group 采多条 episode、过滤无信号 episode、clip ratio）。详见 [[GRPO]]、[[sequence packing与动态采样]]、[[importance sampling与off-policy correction]]。

> [!note] 补充：观测（observation）的来源与可信度
> RAGEN 强调 agent 要"reasoning-aware"——即 reasoning 要**真正利用 observation**（如 observation 说 A，reasoning 要基于 A 推下一步），而非无视 observation 自说自话。纯 outcome reward 无法区分"reasoning 用了 observation"和"reasoning 没用 observation 但碰巧答对"，故模型可能学到后者（幻觉推理）。过程 reward（PRM 判断每步 reasoning 是否引用了 observation）是治此的方向，但 PRM 本身又回到 RM 的痛点（详见 [[verifier与function reward]]）。当前多轮 Agent RL 的核心开放问题。

RAGEN（arXiv:2504.20073，StarPO/Echo Trap/reasoning 涌现条件）已联网核实（任务描述的 2504.11442 实为 TextArena，已勘误）。RAGEN-2（arXiv:2604.06268，reasoning collapse，ICML 2026 Oral）已联网核实。ReAct（Yao et al. 2022）、Toolformer（Schick et al. 2023）为经典范式。verl/OpenRLHF 多轮 rollout 支持基于框架设计与社区实践，具体 API/示例路径随版本变动，已标注待核实项建议查最新官方文档。多轮 logp mask 推导基于 RL 标准（policy gradient 只对可控动作）+ [[rollout train reference logprob一致性]] 的 logp 体系。截至 2026-07。

---
相关: [[Agent rollout]] | [[sandbox与轨迹回放]] | [[RL worker故障恢复]] | [[trajectory generation]] | [[rollout worker]] | [[autoregressive decoding]] | [[verifier与function reward]] | [[synchronous与asynchronous rollout]] | [[sequence packing与动态采样]] | [[rollout train reference logprob一致性]] | [[importance sampling与off-policy correction]] | [[采样策略]] | [[constrained decoding]] | [[RL角色拓扑]] | [[GRPO]] | [[RL权重同步]] | [[OpenAI API server]] | [[R3 rollout routing replay]] | [[17-RL训推一体框架]]
