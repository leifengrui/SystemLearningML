# sandbox与轨迹回放

> **所属章节**: [[Agent rollout]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: sandbox / 沙箱 / 环境状态隔离 / trajectory replay / 轨迹回放 / env state isolation / 多轮环境隔离 / action-observation replay / observation replay / 重算 logp 时的 observation 还原
> **难度**: 高（需懂 [[多轮Agent rollout与tool calling]]、[[rollout train reference logprob一致性]]、[[R3 rollout routing replay]]、[[verifier与function reward]]、[[synchronous与asynchronous rollout]]、[[rollout worker]]、[[sandbox与轨迹回放]]）

> **所属小节**: §87


## 1. 一句话定义

**sandbox 与轨迹回放** 是 [[多轮Agent rollout与tool calling]] 的两个关键工程支柱——① **sandbox（沙箱 / 环境状态隔离）**：在并行采 $N$ 条多轮 trajectory 时，**每条 trajectory 的环境状态必须独立隔离**——代码执行的全局变量、文件系统改动、网络会话、数据库连接等不能跨 trajectory 串扰，否则 trajectory A 调 `os.environ["X"]=1` 会污染 trajectory B 的代码执行结果，observation 错乱、reward 算错；隔离手段是**容器（Docker namespace）/ 进程（subprocess）/ cgroups（资源限制）/ seccomp（系统调用白名单）**，并配**超时与 CPU/内存/网络资源上限**防止恶意/死循环代码拖垮 rollout 池。② **轨迹回放（trajectory replay / observation replay）**：在 trainer 端**重算 logp 时要还原 rollout 时 LLM 看到的完整 context**——因为多轮 trajectory 的 context 里夹着 env 返回的 observation token，这些 token 不是 LLM 生成的、不进梯度，但 LLM 生成 $A_k$ 时**依赖**它们（context 包含 $O_1,\ldots,O_{k-1}$）。trainer 用新 $\theta_{new}$ 重算 $\log\pi_{\theta_{new}}$ 时，必须把 rollout 时记下的 observation **原样拼回 context**，否则 LLM token 的 logp 算错（context 对不上）。回放 vs 重算的关键区分：**回放只重现 env 输出（observation 不变，因为 env 不是 $\pi_\theta$ 决定的）、重算 logp 用新 $\theta$（LLM token 的 logp 随 $\theta$ 变）**——observation 是确定性回放、logp 是用新权重重算。它与 [[R3 rollout routing replay]] 都叫"回放"但对象不同：**R3 回放的是 MoE 的路由选择（专家索引），本条回放的是 env 的 observation（环境输出）**。是 [[多轮Agent rollout与tool calling]] §3.4 trajectory 结构与 §4.2 logp mask 的工程落地——让"多轮 trajectory 能并行采、能在 trainer 端忠实重算 logp"的两块基础设施。

> [!note] 三句话定位
> - **是什么**：① sandbox 给每条并行 trajectory 一个隔离的 env（容器/进程/cgroups/seccomp，超时+资源限），防状态串扰；② 轨迹回放把 rollout 时 env 返回的 observation 原样拼回 context，让 trainer 用新 θ 重算 logp 时 context 与 rollout 时一致（observation 确定性回放、logp 用新权重重算）。
> - **为什么**：并行采 N 条 trajectory 若共享 env 状态会互相污染（trajectory A 改的全局变量影响 B），observation/reward 错乱；trainer 重算 logp 若不还原 observation，LLM token 的 context 对不上、logp 算错 → ratio 失真 → 梯度有偏（[[rollout train reference logprob一致性]] 在多轮的特化）。
> - **与 [[R3 rollout routing replay]] 关系**：都叫"回放"但 R3 回放 MoE 路由（专家索引，保训推选同专家），本条回放 env observation（环境输出，保重算 logp 时 context 一致）。对象不同，机制也不同。


## 2. 为什么需要它（动机与背景）

### 2.1 并行采样的状态污染

多轮 Agent rollout 采一个 batch 时，$N$ 条 trajectory **并行**跑 action loop。若它们**共享一个 env 实例**（同一个 Python 进程、同一个文件系统、同一个网络会话池），会互相污染：

```
trajectory A step 3: env.execute(代码 "import os; os.environ['DEBUG']='1'")
trajectory B step 2: env.execute(代码 "print(os.environ.get('DEBUG'))")
                        ↑ 若共享进程, B 看到了 A 设的 '1'!
                           -> B 的 observation 被污染, 与 A 无关的任务却受影响
```

后果：

- **observation 错乱**：B 拿到不该有的 observation（A 的副作用），LLM 基于错误 observation 推理；
- **reward 错乱**：若 reward 依赖 env 状态（如代码跑测试 pass 数），A 的污染让 B 的测试结果不真实；
- **不可复现**：同一组并行度下结果与并行度耦合（并行 N=8 与 N=16 结果不同），训练**不可复现**、调试地狱。

**根因**：多轮 rollout 的 env 是**有状态的**（与单轮 RL 的无状态 verifier 不同——单轮数学 verifier 只判对错、无副作用）。有状态 env 并行必须隔离。

### 2.2 隔离的层次：从进程到容器到系统调用

状态污染的层次决定隔离手段的层次：

| 污染源 | 隔离手段 | 粒度 |
|---|---|---|
| **Python 全局变量 / 模块状态** | 独立进程（subprocess/multiprocessing） | 进程级 |
| **文件系统**（模型生成代码写文件、改环境） | 临时目录 / 容器 mount namespace | 文件系统级 |
| **进程 / PID** | 容器 PID namespace / 进程隔离 | 进程级 |
| **网络**（模型代码发外网请求、外泄） | 网络命名空间 / 禁网 / 代理白名单 | 网络级 |
| **系统资源**（CPU、内存、磁盘） | cgroups 限额 | 资源级 |
| **系统调用**（恶意 syscall） | seccomp 白名单 | 内核级 |
| **死循环 / 超时** | `signal.alarm` / 容器超时杀进程 | 时间级 |

生产级沙箱常**多层叠加**：Docker 容器（namespace 隔文件/进程/网络）+ cgroups（限资源）+ seccomp（限系统调用）+ 超时杀。基础级（verl/OpenRLHF 的 code rollout 常用）可能只 subprocess + 禁危险 import + 超时，但有绕过风险（详见 [[verifier与function reward]] §3.4）。多轮 Agent rollout 的 sandbox 是 code sandbox 的强化版——**每条 trajectory 一个隔离环境实例**，且要支持**跨 step 状态保持**（同条 trajectory 内多步共享状态、不同 trajectory 间隔离）。

### 2.3 trainer 端重算 logp 的 context 还原问题

rollout 时 LLM 用 $\theta_{old}$ 生成 trajectory，记了 `old_log_prob`。trainer 用 $\theta_{new}$ 重算 `new_log_prob` 算 ratio $\rho$（详见 [[rollout train reference logprob一致性]]）。单轮时 context 就是 `prompt`，trainer 重算时把 prompt + response 喂模型即可。

多轮时 context 是 `prompt + A1 + O1 + A2 + O2 + ...`（夹着 observation）。trainer 重算 LLM token $A_k$ 的 logp 时，**context 必须包含之前的 observation $O_1,\ldots,O_{k-1}$**——否则 LLM 生成 $A_k$ 时看到的 context 与 rollout 时不一致，logp 算错。

**问题**：observation 是 env 在 rollout 时返回的，trainer 不可能重跑 env（env 可能是搜索 API、随机环境、状态已变）。解法：**rollout 时把每步 observation 记下来，trainer 重算时原样拼回 context**——这就是**轨迹回放**。

### 2.4 回放 vs 重算：observation 不变、logp 随 θ 变

关键区分（与 [[rollout train reference logprob一致性]] 的 recompute 策略呼应）：

- **回放（replay）**：重现 **env 的输出**（observation）。observation 是 env 决定的、与 $\pi_\theta$ 无关，故**确定性回放**——rollout 时记下什么、trainer 原样拼回，不重新调 env（也调不了）。
- **重算（recompute）**：用 **$\theta_{new}$ 重算 LLM token 的 logp**。logp 随 $\theta$ 变（这是 IS 的本意），故 trainer 用新权重 forward 重算。

合起来：trainer 重算 logp 时，**observation 用回放（原样拼 context）、LLM token 的 logp 用重算（新 θ forward）**。这是多轮 RL 训推一致性的核心——observation 部分靠回放保证 context 一致，LLM 部分靠重算保证 ratio 用新 θ。

> [!warning] 误区：回放 = 重跑 env
> 错。回放是**重放已记录的 observation 文本**，不是重新调 env。env 可能是随机的（搜索结果每次不同）、有状态的（已消费的 API 额度）、有副作用的（已改的数据库），重跑得不到同结果。回放只取 rollout 时记下的 observation 字符串原样拼回。若 env 有随机性需确定性回放，要记种子（§3.4）。


## 3. 核心概念详解

### 3.1 sandbox：每条 trajectory 一个隔离 env 实例

多轮 rollout 的并行采样架构（简化）：

```
rollout driver:
  for traj_i in batch (N 条并行, 异步):
    env_i = Sandbox()         # 每条 trajectory 独立沙箱实例
    for step in range(max_steps):
      seg = llm.generate(ctx_i)            # LLM 生成
      action = parse(seg)
      obs = env_i.execute(action)         # 在 traj_i 自己的沙箱里执行, 不串扰
      ctx_i += obs
```

每条 trajectory 一个 `env_i`（独立容器/进程），同条 trajectory 内多步共享该 env_i 的状态（step 1 写的文件 step 2 能读），不同 trajectory 间完全隔离。这是 sandbox 的核心契约——**trajectory 内状态共享、trajectory 间状态隔离**。

### 3.2 sandbox 实现层次对比

| 隔离方案 | 隔离强度 | 启动开销 | 适用 |
|---|---|---|---|
| **subprocess + 禁 import** | 弱（同 OS、可绕过） | 极低（ms） | 原型/教学 |
| **multiprocessing** | 中（独立进程，同文件系统） | 低 | 纯计算无 IO |
| **临时目录 + subprocess** | 中（文件隔离、进程隔离） | 低 | 代码执行 RL |
| **Docker 容器** | 强（namespace 全隔离） | 中（100ms~s） | 生产级 code/agent RL |
| **Docker + cgroups + seccomp** | 最强（资源+系统调用都限） | 中 | 安全要求高 |
| **VM / 微 VM（firecracker）** | 极强（独立内核） | 高（s） | 不信任代码、多租户 |
| **gVisor / Kata** | 强（用户态内核） | 中 | 兼顾安全与性能 |

verl/OpenRLHF 的 code rollout 常用 **Docker 容器** + 资源限制（[[verifier与function reward]] §3.4）。多轮 Agent rollout 若 env 复杂（如 SWE-bench 要 git checkout 整个仓库、跑仓库测试），沙箱要支持**持久状态**（跨 step 保持仓库改动）+ **隔离**（不同 trajectory 不同仓库副本）。

### 3.3 资源限制与超时

sandbox 必须限制资源防失控：

| 资源 | 限制手段 | 防什么 |
|---|---|---|
| **CPU** | cgroups `cpu.cfs_quota` / `ulimit -t` | 死循环吃满 CPU、拖慢其他 trajectory |
| **内存** | cgroups `memory.limit` / `ulimit -v` | `list(range(1e9))` 爆内存 OOM 全机 |
| **时间** | `signal.alarm` / 容器 `--timeout` | 死循环/慢查询拖死整条 trajectory |
| **磁盘** | cgroups / quota / 临时目录大小限 | 写爆磁盘 |
| **网络** | 禁网 / 代理白名单 / 带宽限 | 外泄数据、DDoS、爬虫 |
| **进程数** | `ulimit -u` / cgroups pids | fork 炸弹 |

超时机制：每步 tool call 设 `timeout`（如 5s for 计算器、30s for 代码执行、60s for 仓库测试），超时杀进程、该步 observation 返回 `ERROR: timeout`、trajectory 继续或终止。整条 trajectory 设总时长上限防卡死。

### 3.4 确定性回放与随机种子

env 若有随机性（随机环境如 Frozen Lake、Sokoban 随机关卡、搜索 API 返回随机结果），同一 action 不同次调用 observation 不同。回放时必须**确定性**——记下 rollout 时的 env 随机种子/observation，trainer 回放时原样重现。

两种确定性策略：

1. **记种子**：rollout 时给 env 设种子，记下来；回放时用同种子重跑 env（要求 env 是确定性的——给定种子+action 序列，observation 唯一）。适于随机但可复现的 env（如游戏环境）。
2. **记 observation**：rollout 时把每步 observation 的**文本**记下来；回放时不重跑 env、直接拼回 observation 文本。适于不可复现的 env（搜索 API、有状态 DB）。**这是 RL 实践的主流**——因为 env 常不可复现（网络 API、状态已变），干脆记 observation 原样回放。

```python
# 方式1: 记种子, 回放时重跑 env (确定性 env)
env.seed(42); trajectory["env_seed"] = 42
# trainer 回放:
env2.seed(42); for action in trajectory["actions"]: env2.execute(action)  # 重现 observation

# 方式2: 记 observation 文本, 回放时直接拼 (不可复现 env, 主流)
trajectory["observations"] = [obs1, obs2, ...]   # rollout 时记
# trainer 回放: 直接把 observation 文本拼进 context, 不重跑 env
```

> [!tip] 实践：优先记 observation 文本而非种子
> RL 的 env 常不可复现（搜索 API 结果每次不同、代码执行有副作用、网络波动）。记种子要求 env 给定种子+action 序列 observation 唯一——网络 API 做不到。故**主流是记 observation 文本**，trainer 回放时直接拼回，不重跑 env。简单、稳健、env 不可复现也能用。代价是 observation 文本占存储（trajectory 越长越大），但比重跑 env 靠谱。

### 3.5 与 R3 rollout routing replay 的区分

两者都叫"回放"（replay），但**对象与目的不同**：

| 维度 | [[R3 rollout routing replay]] | 本条 trajectory replay |
|---|---|---|
| **回放对象** | MoE **路由选择**（每 token 每层 TopK 专家**索引**） | env **observation**（环境返回的文本） |
| **目的** | 保训推选**同一专家子网络**（消除 MoE 结构性 TIM） | 保 trainer 重算 logp 时 **context 一致**（observation 拼回） |
| **记什么** | 专家索引 int32，张量契约 `(seq-1, n_layers, topk)` | observation token 序列 + mask |
| **何时记** | rollout（推理引擎）记路由 | rollout（env）记 observation |
| **何时回放** | trainer 重算 logp 时强制走同专家 | trainer 重算 logp 时把 observation 拼回 context |
| **针对的 TIM** | MoE 结构性（logp 差 $O(1)$） | 多轮 context 错配（logp 差因 context 对不上） |
| **dense 模型** | 不需要（无路由） | 多轮时仍需要（observation 要回放） |

**关键**：R3 只对 MoE 模型有意义（dense 无路由）；本条的 observation 回放对**任何**多轮 rollout 都需要（dense 与 MoE 都要）。两者正交——MoE 多轮 RL 同时需要两者（路由回放 + observation 回放）。详见 [[R3 rollout routing replay]]、[[rollout train reference logprob一致性]]。

### 3.6 verl 的多轮数据结构

verl 的多轮 trajectory 数据结构（以最新代码为准，部分待核实）支持：

- **多步分段存储**：trajectory 按步分段，每步记 LLM 生成 token + env observation token；
- **observation mask**：token 序列展平后有 mask 区分 LLM token（进 logp）与 observation token（不进）；
- **observation 字段**：observation 文本单独存储，trainer 重算时按步拼回 context；
- **env 状态**（若需跨 step）：sandbox 实例或状态快照随 trajectory 存。

这种结构让 trainer 能"回放 observation + 重算 LLM logp"——把 observation 按步拼回 context、对 LLM token 用新 $\theta$ forward 算 logp，observation token mask 掉不进梯度。详见 [[多轮Agent rollout与tool calling]] §5.2。


## 4. 数学原理 / 公式

### 4.1 隔离的正确性：trajectory 独立性

设并行采 $N$ 条 trajectory，每条 trajectory $\tau_i$ 的 env 状态 $S_i$。**隔离要求**：$S_i \perp S_j$（$i\ne j$），即 trajectory $i$ 的 env 状态不影响 trajectory $j$。

若不隔离（共享 env 状态 $S$），trajectory $j$ 的 observation $O_j^{(k)}$ 依赖 $S$，而 $S$ 被 trajectory $i$ 的 action 修改：

$$
O_j^{(k)} = \text{env}(A_j^{(k)}, S), \quad S \leftarrow \text{update}(S, A_i^{(k')})
$$

则 $O_j^{(k)}$ 受 $A_i^{(k')}$ 影响，trajectory $j$ 的 reward $r_j$ 也受污染：

$$
r_j = f(\tau_j, S) \ne f(\tau_j, S_j) \quad (\text{污染后} \ne \text{隔离应有的})
$$

隔离后 $S_i$ 独立：$O_j^{(k)} = \text{env}(A_j^{(k)}, S_j)$，$S_j$ 只被 trajectory $j$ 自己的 action 改，reward 干净。这是并行采样数学正确性的前提。

### 4.2 回放正确性：context 一致性

trainer 重算 LLM token $A_k$ 的 logp 时，context 应与 rollout 时一致。rollout 时 LLM 生成 $A_k$ 看到的 context：

$$
c_k^{\text{rollout}} = p \; A_1 \; O_1 \; \cdots \; A_{k-1} \; O_{k-1}
$$

trainer 重算时 context：

$$
c_k^{\text{train}} = p \; A_1' \; O_1' \; \cdots \; A_{k-1}' \; O_{k-1}'
$$

其中 $A'$ 是 rollout 时记下的 LLM 生成 token（不变）、$O'$ 是回放的 observation。**回放正确性要求** $O' = O$（observation 原样回放），则 $c_k^{\text{train}} = c_k^{\text{rollout}}$，context 一致。

若不回放（observation 缺失或不同），$c_k^{\text{train}} \ne c_k^{\text{rollout}}$，则：

$$
\log\pi_{\theta}(A_k | c_k^{\text{train}}) \ne \log\pi_{\theta}(A_k | c_k^{\text{rollout}})
$$

即使 $\theta$ 相同（$\theta=\theta_{old}$），logp 也算错——context 对不上。这正是 [[rollout train reference logprob一致性]] 在多轮的特化：单轮时 context 一致（只 prompt），多轮时 context 含 observation，必须回放保一致。

### 4.3 logp 累积（带回放 + mask）

trainer 用新 $\theta$ 重算 logp（observation 回放、LLM token 重算）：

$$
\log\pi_\theta(\tau) = \sum_{t} m_t \cdot \log\pi_\theta(\tau_t | \underbrace{c_t^{\text{train}}}_{\text{含回放 obs}})
$$

其中 $m_t$ 是 mask（$m_t=1$ 若 LLM token，$0$ 若 observation/prompt），$c_t^{\text{train}}$ 含回放的 observation（与 rollout 时 $c_t^{\text{rollout}}$ 一致）。ratio：

$$
\rho_t = \exp\big(\log\pi_\theta(\tau_t|c_t^{\text{train}}) - \log\pi_{\theta_{old}}(\tau_t|c_t^{\text{rollout}})\big), \quad m_t=1
$$

observation token 不算 ratio（两端都跳过）。回放保证 $c_t^{\text{train}}=c_t^{\text{rollout}}$ 在 observation 部分一致，故 ratio 只反映 $\theta$ 变化（$\theta$ vs $\theta_{old}$）在 LLM token 上的影响，不被 context 错配污染。这是多轮 RL 训推一致性的数学根。

### 4.4 沙箱开销与并行度

设单沙箱启动开销 $T_{\text{init}}$（容器启动 ~100ms-1s）、单步 env 执行 $T_{\text{exec}}$、单步 LLM 生成 $T_{\text{gen}}$。一条 trajectory $K$ 步：

$$
T_{\text{traj}} = T_{\text{init}} + \sum_{k=1}^{K}(T_{\text{gen}}^{(k)} + T_{\text{exec}}^{(k)})
$$

并行 $N$ 条（每条独立沙箱），若资源够：

$$
T_{\text{batch}} \approx \max_i T_{\text{traj}}^{(i)} + T_{\text{init}} \quad (\text{并行, 取最长})
$$

沙箱启动开销 $T_{\text{init}}$ 是固定成本——若 trajectory 短（$K$ 小），$T_{\text{init}}$ 占比大，吞吐受损。优化：**沙箱池化**（预启动一批容器复用，免每次启动）、**轻量沙箱**（subprocess 比 Docker 快）。详见 §8.2。

### 4.5 回放存储开销

回放要记 observation 文本。设平均 observation 长度 $L_o$ token、$K$ 步、batch $N$ 条：

$$
\text{存储} \approx N \cdot K \cdot L_o \cdot \text{bytes/token}
$$

例：$N=512$, $K=8$, $L_o=128$, 4 bytes/token → $512\times8\times128\times4 \approx 2$ MB（observation 文本，可接受）。但若 observation 含大输出（如代码执行的完整 stdout、搜索返回的长文档），$L_o$ 可达数千 token，存储与传输涨。优化：observation 截断（只留关键部分，但截断改变 context 会破坏回放一致性，需谨慎）、压缩。

> [!warning] 误区：observation 可以截断/改写
> 回放要求 observation **原样**——截断/改写会让 $c_t^{\text{train}} \ne c_t^{\text{rollout}}$，logp 算错。若 rollout 时 LLM 看到的 observation 是长文档全文，trainer 回放也必须是全文。若要省存储，应在 **rollout 时就截断**（让 LLM 看到截断版），则回放也用截断版（一致）。不能 rollout 看全文、回放用截断——context 不一致。


## 5. 代码示例（可选）

### 5.1 最简 sandbox（每条 trajectory 独立进程 + 临时目录）

```python
import subprocess, tempfile, os, signal, json

class TrajectorySandbox:
    """每条 trajectory 一个沙箱: 独立进程 + 临时目录 + 超时 + 禁危险模块"""
    def __init__(self):
        self.tmpdir = tempfile.mkdtemp(prefix="sandbox_")   # 独立文件系统
        self.env_state = {}                                    # 跨 step 的状态 (同 traj 内共享)

    def execute(self, code: str, timeout: float = 5.0) -> str:
        # 禁危险 import + 限制 + 捕获输出
        guarded = (
            "import sys, importlib.util\n"
            "for m in ['os','subprocess','socket','shutil','ctypes']:\n"
            "    sys.modules[m]=None\n"
            f"{code}\n"
        )
        try:
            proc = subprocess.run(
                ["python", "-c", guarded],
                capture_output=True, text=True,
                timeout=timeout,
                cwd=self.tmpdir,                # 独立工作目录
                env={"PYTHONPATH": "", "HOME": self.tmpdir},  # 干净环境
            )
            return f"<observation>{proc.stdout or proc.stderr}</observation>"
        except subprocess.TimeoutExpired:
            return "<observation>ERROR: timeout</observation>"
        except Exception as e:
            return f"<observation>ERROR: {e}</observation>"

    def cleanup(self):
        import shutil; shutil.rmtree(self.tmpdir, ignore_errors=True)

# 并行采 N 条 trajectory, 各自独立沙箱
def parallel_rollout(prompts, llm, max_steps=5):
    sandboxes = [TrajectorySandbox() for _ in prompts]   # 每条独立
    trajectories = []
    for i, p in enumerate(prompts):
        env = sandboxes[i]   # trajectory i 用自己的沙箱, 不串扰
        traj = run_action_loop(llm, env, p, max_steps)   # 见 [[多轮Agent rollout与tool calling]] §5.1
        traj["sandbox"] = env   # 持有引用 (跨 step 状态保持)
        trajectories.append(traj)
    for s in sandboxes: s.cleanup()
    return trajectories
```

### 5.2 轨迹回放：trainer 端重算 logp 时还原 observation

```python
import torch

def replay_and_recompute_logp(model, trajectory, tokenizer):
    """trainer 用新 θ 重算 logp: observation 回放 + LLM token 重算"""
    ids, mask = build_sequence_and_mask(trajectory, tokenizer)  # §5.3 of 多轮笔记
    # ids = [prompt, A1, O1, A2, O2, ...] 展平
    # mask = [0(p), 1(A1), 0(O1), 1(A2), 0(O2), ...]  observation=0!
    with torch.no_grad():
        logits = model(ids.unsqueeze(0))   # 新 θ forward, context 含回放的 obs
    logp = torch.log_softmax(logits[0, :-1], dim=-1)
    tgt = ids[1:]
    gathered = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    # 只 LLM token (mask=1) 累加, observation (mask=0) 跳过
    llm_logp = (gathered * mask[1:].float()).sum()
    return llm_logp

def build_sequence_and_mask(trajectory, tokenizer):
    """展平 trajectory: prompt + 每步(LLM seg + observation), 带mask"""
    ids, mask = [], []
    for t in tokenizer(trajectory["prompt"])["input_ids"]:
        ids.append(t); mask.append(0)
    for step in trajectory["steps"]:
        for t in tokenizer(step["llm_seg"])["input_ids"]:
            ids.append(t); mask.append(1)        # LLM 生成 -> 进 logp
        if step.get("obs"):
            for t in tokenizer(step["obs"])["input_ids"]:
                ids.append(t); mask.append(0)    # env observation -> 回放但不进 logp
    return torch.tensor(ids), torch.tensor(mask)
```

### 5.3 Docker 沙箱（生产级，简化）

```python
import subprocess, json

class DockerSandbox:
    """生产级: Docker 容器隔离 + 资源限制 + 超时"""
    def __init__(self, image="python:3.11-slim"):
        self.image = image
        self.container = None
    def execute(self, code, timeout=30):
        cmd = ["docker", "run", "--rm",
               "--memory=512m", "--cpus=1",          # cgroups 限资源
               "--network=none",                      # 禁网
               "--read-only",                         # 只读根文件系统
               "--tmpfs", "/tmp:rw,size=64m",         # 临时写区
               "--pids-limit=64",                     # 限进程防 fork bomb
               self.image, "python", "-c", code]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return f"<observation>{r.stdout or r.stderr}</observation>"
        except subprocess.TimeoutExpired:
            return "<observation>ERROR: timeout</observation>"
        # --rm 自动清理容器, 不留状态
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[多轮Agent rollout与tool calling]]（sandbox 与回放是多轮 rollout 的两个支柱）、[[verifier与function reward]]（code sandbox 是 function reward 的执行环境，本条是其在多轮并行场景的强化）、[[rollout train reference logprob一致性]]（回放是 logp 一致性在多轮的特化——observation 回放保 context 一致）、[[rollout worker]]（rollout 端要管理 sandbox 池与 observation 记录）。
- **下游（应用）**: [[synchronous与asynchronous rollout]]（异步采样下 sandbox 与回放更复杂——trajectory 跨 θ 版本，observation 仍按各自版本记）、[[sequence packing与动态采样]]（多条多轮 trajectory packing 时 observation mask + block-diagonal 隔离）、[[RL worker故障恢复]]（sandbox 崩了要能续、observation 回放支持 checkpoint 恢复）、[[17-RL训推一体框架]]（多轮 Agent RL 的基础设施）。
- **对比 / 易混**:
  - **[[R3 rollout routing replay]] vs 本条 trajectory replay**：R3 回放 MoE 路由（专家索引，保训推选同专家），本条回放 env observation（环境输出，保重算 logp context 一致）。对象、目的、机制均不同。见 §3.5。
  - **sandbox（状态隔离）vs checkpoint（状态保存）**：sandbox 防并行 trajectory 间串扰（空间隔离），checkpoint 防故障后状态丢失（时间保存）。一个防并发污染、一个防崩溃丢失。
  - **回放（replay observation）vs 重算（recompute logp）**：回放重现 env 输出（确定性、不随 θ 变），重算用新 θ 算 LLM logp（随 θ 变）。见 §2.4。
- **三篇互链**: [[多轮Agent rollout与tool calling]]（本条是其 sandbox 与回放支柱）、[[RL worker故障恢复]]（多轮 trajectory 的 sandbox 状态与 observation 回放影响故障恢复）。


## 7. 常见误区与易错点

> [!warning] 误区 1：多轮 rollout 共享一个 env 实例
> 错。共享 env 让 trajectory 间状态串扰（A 改的全局变量影响 B）。必须**每条 trajectory 独立 sandbox 实例**（独立进程/容器）。即使 env "看起来无状态"（如计算器），代码执行类 env 一定有状态（全局变量、文件、import 缓存）。默认隔离，别赌无状态。

> [!warning] 误区 2：回放 = 重跑 env
> 错。回放是**重放已记录的 observation 文本**，不是重新调 env。env 可能不可复现（网络 API、随机、有状态）。记 observation 文本、原样拼回。除非 env 是确定性可复现的（游戏环境 + 种子），才重跑。主流是记文本。见 §3.4、§2.4。

> [!warning] 误区 3：observation 可以截断/改写以省存储
> 错。回放要求 observation **原样**——截断让 $c^{\text{train}}\ne c^{\text{rollout}}$，logp 算错。要省存储应在 rollout 时就截断（LLM 看截断版、回放也截断版，一致），不能 rollout 看全文、回放截断。见 §4.5。

> [!warning] 误区 4：subprocess + 禁 import 就是沙箱
> 不够。`sys.modules[m]=None` 挡不住 `__import__('o'+'s')`、`builtins.__import__`、`ctypes` 直接 syscall。生产级必须 Docker 容器 + cgroups + seccomp。详见 [[verifier与function reward]] §3.4。原型可 subprocess，生产别。

> [!tip] 实践：沙箱池化降启动开销
> Docker 容器启动 100ms-1s，若 trajectory 短、每条都启容器，$T_{\text{init}}$ 占比大。解法：**预启动一批容器池**，采 trajectory 时从池取、用完归还复用（不 `--rm` 销毁、而是 reset 状态后复用）。或用 subprocess/轻量沙箱降开销。见 §4.4、§8.2。

> [!tip] 实践：observation 记文本 + mask，别只记"调了什么工具"
> 回放要还原 context 的 observation token，不只记"step 3 调了 search('query')"——还要记 search 返回的**结果文本**。否则 trainer 回放时不知道拼什么 observation。trajectory 结构里每步要有 `obs` 字段存完整 observation 文本。

> [!note] 补充：跨 step 状态保持 vs trajectory 间隔离的张力
> sandbox 的契约是"trajectory 内状态共享、trajectory 间隔离"。实现上，同 trajectory 多步要用**同一个** sandbox 实例（状态跨 step 保持），不同 trajectory 用**不同**实例。并行采 N 条 = N 个 sandbox 实例。若 sandbox 重（Docker），N 大时资源紧。权衡：重 sandbox + 池化复用 vs 轻 sandbox（subprocess）+ 弱隔离。看 env 是否信任（模型生成的代码不信任→重 sandbox）。


## 8. 延伸细节

### 8.1 沙箱层级与安全威胁模型

模型生成的代码/动作的威胁模型决定沙箱层级：

| 威胁 | 沙箱手段 |
|---|---|
| **无意死循环/OOM** | 超时 + 内存限（cgroups） |
| **误删文件** | 临时目录 + 只读根 + tmpfs |
| **网络外泄**（模型把数据发外网） | `--network=none` / 代理白名单 |
| **恶意 syscall**（ptrace、内核漏洞） | seccomp 白名单 |
| **逃逸容器**（CVE） | gVisor/Kata/微 VM（独立内核） |
| **fork bomb** | `--pids-limit` |
| **侧信道**（跨 trajectory 偷信息） | 强隔离（VM）或时间错峰 |

LLM-RL 的代码 env 威胁主要是**无意的**（模型不会主动攻击，但会生成死循环/OOM 代码）+ **reward hacking**（模型生成篡改测试的代码）。前者用超时+资源限够，后者要测试工程（禁 mock、独立断言，详见 [[verifier与function reward]] §8.3）。恶意逃逸（模型主动攻击宿主）在单方训练场景少见，多租户/在线服务才需 VM 级。

### 8.2 沙箱池化与复用

Docker 容器启动开销 $T_{\text{init}}$ 在短 trajectory 下占比大。池化策略：

- **预启动池**：训练开始预起 $M$ 个容器（$M \ge$ 并行度 $N$），采 trajectory 时取一个、reset 状态（清临时目录、重置环境变量）、用完归还；
- **不 `--rm`**：容器不销毁、复用，省启动；
- **reset 成本**：reset 比 start 快（清文件 vs 起容器）；
- **污染累积**：复用多次后可能有残留状态，需定期销毁重建。

verl/OpenRLHF 的 code rollout 常用池化（具体实现待核实）。轻量场景用 subprocess（启动 ms 级）免池化。

### 8.3 异步采样下的 sandbox 与回放

异步采样（[[synchronous与asynchronous rollout]]）下 trajectory 跨 θ 版本，sandbox 与回放更复杂：

- **sandbox 生命周期**：异步下 trajectory 采的时间拉长（不等 trainer），sandbox 占用时间拉长，池要更大；
- **observation 版本**：trajectory 用 $\theta_{old-k}$ 采，observation 是该版本下 env 返回的；trainer 用 $\theta_{new}$ 重算时 observation 仍回放原样（observation 不随 θ 变，env 不是 θ 决定的）——这是回放的**版本无关性**，与 LLM token 的 ratio（随 θ 变）形成对比；
- **staleness 对回放无影响**：observation 回放的是 env 输出，与 θ 版本无关，故 staleness 不破坏回放一致性。staleness 只影响 LLM token 的 ratio（需 IS 修正），不影响 observation 回放。

### 8.4 回放与 R3 在 MoE 多轮 RL 的叠加

MoE 模型做多轮 Agent RL 时，trainer 重算 logp 需**同时**：

1. **observation 回放**（本条）：把 env observation 原样拼回 context；
2. **路由回放**（[[R3 rollout routing replay]]）：强制 trainer 走与 rollout 相同的 MoE 专家（保训推选同专家子网络）。

两者叠加：trainer forward 时，context 含回放 observation（保 context 一致）+ 路由 mask（保子网络一致），logp 才既反映 θ 变化又不被 context/路由错配污染。这是 MoE 多轮 RL 训推一致性的完整要求。详见 [[R3 rollout routing replay]]、[[rollout train reference logprob一致性]]。

### 8.5 sandbox 故障与 trajectory 部分失败

某条 trajectory 的 sandbox 中途崩（OOM、超时、容器死），该 trajectory 部分失败。处理：

- **graceful 降级**：sandbox 崩返回 `ERROR` observation，LLM 继续（如 [[多轮Agent rollout与tool calling]] §7 实践）；
- **丢弃该 trajectory**：若崩得太早（前几步就崩），observation/reward 不全，丢弃不送训练（类似 Dynamic Sampling 过滤）；
- **sandbox 重启续**：复杂——sandbox 状态丢了，要重建状态才能续，常不如重采。详见 [[RL worker故障恢复]]。

### 8.6 确定性回放的边界：env 不可复现时的折中

若 env 严格不可复现（如真实网络搜索，每次结果不同），只能记 observation 文本回放。但若 observation 极大（搜索返回整页 HTML），存储/传输贵。折中：

- **rollout 时就摘要**：让 LLM 只看 search 返回的摘要（而非全文），则回放也只存摘要——一致且省。但摘要逻辑要是确定性的（不能随机摘要）；
- **observation 分层**：关键 observation（最终答案、报错）全存，辅助 observation（中间搜索结果）摘要——但 LLM 看到的是摘要，context 已变，回放用摘要（一致）。

原则：**rollout 时 LLM 看到什么、回放就拼什么**——不能事后压缩。要省就在 rollout 端省。

### 8.7 与单轮 RL 的对比

单轮 RL（GRPO 数学题）的 env 是无状态 verifier（判对错、无副作用），故**不需要 sandbox**（无状态污染）也**不需要 observation 回放**（trajectory 是 `prompt+response` 扁平、无 env observation 夹层）。sandbox 与回放是多轮 Agent rollout **独有**的工程负担——是多轮比单轮复杂的根因之一。把单轮 RL 框架（verl/OpenRLHF）扩展到多轮，主要工作量就在 sandbox 管理与 observation 回放的数据结构改造。详见 [[多轮Agent rollout与tool calling]] §8。

sandbox 与回放设计综合自多轮 Agent RL 工程实践（verl/OpenRLHF code rollout、SWE-agent 沙箱）与 [[verifier与function reward]] §3.4（沙箱执行安全隔离）。observation 回放与 logp 一致性的关系基于 [[rollout train reference logprob一致性]] 的 recompute 体系在多轮的特化。与 [[R3 rollout routing replay]] 的区分基于 R3（arXiv:2510.11370）回放 MoE 路由、本条回放 env observation 的对象差异。沙箱池化、cgroups/seccomp/namespace 层级综合自容器安全标准。verl 多轮数据结构的具体字段以最新官方代码为准，已标注待核实项。截至 2026-07。

---
相关: [[Agent rollout]] | [[多轮Agent rollout与tool calling]] | [[RL worker故障恢复]] | [[verifier与function reward]] | [[rollout train reference logprob一致性]] | [[R3 rollout routing replay]] | [[synchronous与asynchronous rollout]] | [[sequence packing与动态采样]] | [[rollout worker]] | [[17-RL训推一体框架]]
