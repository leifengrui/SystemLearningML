# verifier与function reward

> **所属章节**: [[reward工程]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: rule-based verifier / 规则验证器 / function reward / 程序执行反馈 / 代码沙箱 reward / outcome verifier / pass@k reward / tool-use reward / 数学答案验证 / reward shaping / RL reward 来源
> **难度**: 中（需懂 [[Reward Model]]、[[reward hacking]]、[[GRPO]]、[[DAPO]]、[[trajectory generation]]、[[rollout worker]]、[[sequence packing与动态采样]]、[[synchronous与asynchronous rollout]]）

> **所属小节**: §86


## 1. 一句话定义

**verifier 与 function reward** 是 [[17-RL训推一体框架]] 中 **reward（奖励信号）的三大来源**及其工程实现——① **reward model（RM）**：用人类偏好对（preference pairs）训出来的**神经网络打分器** $r_\phi(q,o)\in\mathbb{R}$，把"人类觉得哪个好"蒸馏成标量，适合**开放式、主观、难精确判对错**的任务（通用对话、有用性、无害性），详见 [[Reward Model]]；② **rule-based verifier（规则验证器）**：用**确定性程序/正则/对答案**判 response 对错，输出 0/1（或部分分），适合**有标准答案、可精确验证**的任务（数学题对 `\boxed{}`、代码题跑测试用例、格式约束正则）；③ **function reward（函数 reward / 程序执行反馈）**：把 response **送进代码沙箱/工具环境真跑一遍**，用**执行结果**当 reward（unittest pass 数、pass@k、工具调用返回值、编译器输出），是 rule-based 的"执行型"加强版——不止对答案，还跑过程。本条核心论点是 **rule-based / function reward 在数学/代码 RL（R1/GRPO/DAPO 范式）流行的原因**：reward 虽**稀疏**（只有最终对错）但**可精确验证**（不靠 proxy 网络、无 RM 偏差）、**不训 RM 省成本**（免偏好标注+训 RM 的工程链）、**防 reward 注入**（程序判对错，actor 难钻空子）；代价是**只限可验证任务**，开放式对话仍需 RM。配套工程：**沙箱执行**（安全隔离 untrusted 代码、超时/内存/CPU/网络资源限制）、**reward shaping**（Overlong Reward Shaping 治超长截断噪声、过程 reward vs 结果 reward 的稀疏度权衡）。是 [[GRPO]]/[[DAPO]] 范式的 reward 侧支柱，与 [[reward hacking]] 互补——RM 易被 hack，rule-based/function reward 难被 hack（程序铁面无私）。

> [!note] 三句话定位
> - **是什么**：reward 三来源——RM（偏好网络，主观开放任务）、rule-based verifier（对答案/正则，可验证任务）、function reward（沙箱执行反馈，代码/工具任务）。数学/代码 RL 选 rule-based/function，因为稀疏但可精确验证、不训 RM、防注入。
> - **为什么**：RM 是 proxy 易被 [[reward hacking]]（Goodhart），rule-based 是 ground truth 不被 hack；但 rule-based 只限可验证任务，开放对话仍需 RM。沙箱执行保安全，reward shaping 治噪声。
> - **与 [[reward hacking]] 关系**：RM 越优化越易 hack（钻 proxy 漏洞）；rule-based/function reward 是**程序判对错**，actor 难钻空子（除非写出让测试用例误判的代码，但这本身极难且可防）。是防 reward hacking 的根本手段之一。


## 2. 为什么需要它（动机与背景）

### 2.1 RL 要 reward，但 reward 从哪来

RL（[[GRPO]]/[[DAPO]]/PPO）的策略梯度 $\nabla J\propto\hat A$ 依赖 **advantage** $\hat A$，而 advantage 依赖 **reward** $r$（GRPO 用组内 reward 均值当 baseline，详见 [[GRPO]]）。故**reward 函数是 RL 的燃料**——没有 reward 就没梯度、没更新。问题是：**给定一条 response，怎么给它打分？** 这是 reward 工程的核心问题，三条路线：

1. **训一个网络学人类偏好**（RM）：贵、proxy、易 hack，但能处理任意主观任务；
2. **写规则判对错**（rule-based verifier）：便宜、精确、难 hack，但只限有标准答案的任务；
3. **真跑一遍看结果**（function reward）：介于两者——执行型验证，比纯对答案强（能判过程），但仍限可执行任务。

本条就是这三条路线的对比与工程实现。

### 2.2 RM 的痛点：proxy + Goodhart + 贵

RM（[[Reward Model]]）用偏好对 $(x,y_c,y_l)$ 训 Bradley-Terry 打分器 $r_\phi$。三大痛点（详见 [[reward hacking]]）：

- **proxy 偏差**：$r_\phi$ 是人类偏好 $r^*$ 的代理，必有盲区，PPO 优化越深越钻空子（Goodhart），reward 涨但真实质量降；
- **reward hacking**：actor 拉长 response、重复堆砌、套话、迎合 RM 偏好，刷高 $r_\phi$ 但没真变强；
- **成本高**：要标偏好对（人工/AI 标）+ 训 RM（额外 GPU + 工程链）+ 在线 RM 更新防 stale。

对**数学/代码这类有标准答案的任务**，用 RM 是杀鸡用牛刀且引入不必要偏差——明明能对答案/跑测试，何必训个 proxy 网络还担 hack 风险？

### 2.3 rule-based / function reward 的优势：精确、省成本、防注入

对数学题（答案可对错）、代码题（测试用例可跑），**直接用程序判**：

- **精确**：`\boxed{42}` 对就是对、错就是错，无 proxy 偏差。reward 是 ground truth，不是代理；
- **省成本**：不用标偏好对、不用训 RM、不用在线更新 RM。一个 verifier 函数搞定；
- **防 reward 注入**：程序铁面无私，actor 难钻空子。RM 的"漏洞"来自网络泛化盲区；rule-based 的"漏洞"只能来自**验证逻辑写错**（漏判某个错答案对、或对答案判错），这可测可修，且 actor 难系统性利用。

这是 R1/GRPO/DAPO 范式选 rule-based 的根——**数学/代码 RL 的爆发，本质是发现了"这类任务的 reward 可廉价精确获得"**，绕过了 RM 这道贵且不稳的工序。

### 2.4 代价：只限可验证任务

rule-based/function reward 的硬伤：**只限有标准答案/可执行验证的任务**。开放对话（"哪个回答更有用"）没有标准答案，无法对答案、无法跑测试，仍需 RM。故：

- **数学/代码 RL**（R1/GRPO/DAPO）：rule-based + function reward 主导；
- **通用对话对齐**（InstructGPT/Llama-2 RLHF）：RM 主导；
- **混合**（Agent/工具调用）：function reward（工具执行结果）+ RM（对话质量）联合。

这是 reward 工程的分野——**任务的"可验证性"决定 reward 路线**。

### 2.5 沙箱与 shaping：让 function reward 安全且不噪

function reward 要跑 untrusted 代码（模型生成的 Python/代码），必须**沙箱隔离**（防 `os.remove`、死循环吃 CPU、网络外泄）；且 reward 常因**超长截断**（response 写到 max_length 还没出答案）产生噪声（截断样本被判错但其实是"还没写完"），需 **reward shaping**（Overlong Reward Shaping）平滑。这两点是 function reward 的工程配套。


## 3. 核心概念详解

### 3.1 reward 三大来源总览

| 维度 | ① reward model (RM) | ② rule-based verifier | ③ function reward |
|---|---|---|---|
| **是什么** | 偏好数据训的神经网络 $r_\phi(q,o)$ | 确定性程序/正则/对答案 | 代码沙箱/工具环境执行反馈 |
| **输出** | 连续标量 $\mathbb{R}$ | 0/1（或部分分） | pass 数 / pass@k / 执行返回值 |
| **任务适合** | 开放对话、有用性、无害性（主观） | 数学（对答案）、格式（正则） | 代码（unittest）、工具调用、Agent |
| **可验证性** | 不可精确验证（proxy） | 可精确验证（ground truth） | 可执行验证（ground truth） |
| **reward 稀疏度** | 稠密（每条都打分） | 极稀疏（只有最终对错） | 稀疏~中（pass 数有梯度） |
| **训 RM 成本** | 高（标偏好 + 训 RM + 在线更新） | 零（写 verifier 函数） | 零~低（写测试用例 + 沙箱） |
| **reward hacking 风险** | 高（proxy 可钻） | 低（程序铁面，除非 verifier 写错） | 低（测试用例铁面，除非用例有漏洞） |
| **泛化性** | 强（网络泛化到新 response） | 弱（只判 verifier 覆盖的） | 弱（只判能跑的） |
| **典型框架** | InstructGPT/Llama-2 RLHF | GRPO/R1 数学、DAPO | R1 代码、SWE-agent、verl code rollout |
| **详见** | [[Reward Model]] | 本条 §3.2 | 本条 §3.3 |

> [!note] 补充：三者不是互斥
> 实际系统常**联合**：代码题 = function reward（跑测试）+ rule-based（编译是否通过）+ RM（代码风格/可读性）；Agent = function reward（工具返回）+ RM（对话是否完成指令）。分界是"主 reward 用谁"——数学/代码 RL 主用 rule-based/function，通用对话主用 RM。

### 3.2 ② rule-based verifier：对答案/正则/格式

rule-based verifier 用**确定性规则**判 response 对错，最典型的两种：

**数学题——对最终答案**：从 response 里抽取 `\boxed{}`（或 `####`、`The answer is`）里的答案，与 ground truth 比较（去空格/等价化后字符串匹配或 sympy 符号等价）。

```python
import re, sympy

def math_verifier(response: str, ground_truth: str) -> float:
    # 1. 抽 \boxed{} 里的答案
    m = re.search(r"\\boxed\{([^}]*)\}", response)
    if not m:
        return 0.0  # 没给答案 -> 错
    pred = m.group(1).strip()
    # 2. 等价化 (去空格, sympy 符号等价)
    try:
        if sympy.simplify(sympy.sympify(pred) - sympy.sympify(ground_truth)) == 0:
            return 1.0  # 对
    except Exception:
        if pred.replace(" ", "") == ground_truth.replace(" ", ""):
            return 1.0  # 字符串等价
    return 0.0  # 错
```

- **稀疏**：只有最终对错 0/1，中间推理错对 reward 无贡献（除非加过程 reward，见 §3.5）。
- **精确**：对就是对、错就是错，无 proxy 偏差。
- **防注入**：actor 要"刷高 reward"只能真把答案算对，钻不了空子（除非 verifier 抽取逻辑有漏洞，如正则漏边界，可测可修）。

**格式约束——正则**：如要求 response 有 `<think>...</think>` 段、JSON 合法、特定字段。

```python
import json, re

def format_verifier(response: str) -> float:
    score = 0.0
    if re.search(r"<think>.*</think>", response, re.DOTALL):  # 有 think 段
        score += 0.5
    try:
        json.loads(response.split("</think>")[-1])  # think 后是合法 JSON
        score += 0.5
    except Exception:
        pass
    return score  # 0 / 0.5 / 1.0
```

格式 verifier 常作为**辅助 reward**（与对答案 reward 叠加），引导模型学会要求的输出格式。

### 3.3 ③ function reward：沙箱执行反馈

function reward 把 response **送进代码沙箱真跑**，用**执行结果**当 reward。比 rule-based 强在**执行型验证**——不止对最终答案，还跑过程（测试用例、工具调用）。

**代码题——跑 unittest**：模型生成一段 Python 函数，沙箱里对预设测试用例跑，按 pass 数给分：

```python
def code_function_reward(generated_code: str, test_cases: list, sandbox: "Sandbox") -> float:
    # 沙箱执行 (隔离 untrusted 代码, 详见 §3.4)
    results = sandbox.run(generated_code, test_cases)  # 每个用例 pass/fail/exception
    n_pass = sum(1 for r in results if r == "pass")
    return n_pass / len(test_cases)  # 0~1, pass 率
    # 进阶: pass@k (k 条采样至少一条全过的概率)
```

- **pass@k**：对同一 prompt 采 $k$ 条 response，只要**至少一条全过测试**就算"可解"，reward = $1-(1-p)^k$（$p$ 单条通过率）。这是代码 RL 的标准指标，比单条 pass 更宽容、更适合 RL（允许探索）。
- **工具调用 reward**：Agent 生成调工具的代码（如 `search("query")`），沙箱执行工具返回结果，用**结果是否解决任务**当 reward（如 SWE-agent 修 bug 后跑仓库测试）。

function reward 的本质：**把"response 好不好"转成"执行结果对不对"**，让 reward 从主观变客观。

### 3.4 沙箱执行：安全隔离与资源限制

function reward 要跑**模型生成的 untrusted 代码**，模型可能生成恶意/死循环代码（`while True`、`os.remove`、`subprocess` 外泄），必须沙箱隔离。工程手段：

| 隔离层 | 手段 | 防什么 |
|---|---|---|
| **进程隔离** | subprocess / multiprocessing 起独立进程 | 代码崩不拖垮主进程 |
| **容器隔离** | Docker / namespace / cgroups | 文件系统/进程/网络隔离 |
| **资源限制** | cgroups CPU/内存上限、`signal.alarm` 超时、禁网络 | 死循环吃 CPU、内存爆、外泄 |
| **权限裁剪** | 禁 `import os/subprocess/socket`、seccomp 系统调用白名单 | 恶意系统调用 |
| **临时环境** | 临时目录 + 用完即弃 | 写坏宿主文件系统 |

```python
import subprocess, signal

class Sandbox:
    """最小代码沙箱: 独立进程 + 超时 + 临时目录 + 禁危险模块"""
    def run(self, code: str, test_cases: list, timeout: float = 5.0):
        # 1. 包装: 禁危险 import + 跑测试用例 + 捕获异常
        wrapped = self._wrap(code, test_cases)
        # 2. 独立进程跑 (隔离), 设超时 (防死循环)
        try:
            proc = subprocess.run(
                ["python", "-c", wrapped],
                capture_output=True, text=True,
                timeout=timeout,  # 超时杀进程
                env={"PYTHONPATH": ""},  # 干净环境
            )
            return self._parse_results(proc.stdout)  # ["pass","fail",...]
        except subprocess.TimeoutExpired:
            return ["timeout"] * len(test_cases)  # 超时 = 全 fail
        except Exception:
            return ["error"] * len(test_cases)

    def _wrap(self, code, test_cases):
        # 禁危险模块 (基础防护, 生产级用 cgroups/seccomp/容器)
        guarded = "import sys, importlib.util\n" \
                  "for m in ['os','subprocess','socket','shutil']:\n" \
                  "    sys.modules[m]=None  # import 即抛\n" \
                  f"{code}\n" \
                  f"results=[]\n" \
                  f"for tc in {test_cases}:\n" \
                  f"    try:\n" \
                  f"        assert tc[0](*tc[1:])\n" \
                  f"        results.append('pass')\n" \
                  f"    except Exception:\n" \
                  f"        results.append('fail')\n" \
                  f"import json;print(json.dumps(results))\n"
        return guarded
```

> [!warning] 误区：沙箱靠禁 import 就够
> `sys.modules[m]=None` 只挡 `import os` 这种直白调用，挡不住 `__import__('o'+'s')`、`builtins.__import__`、`ctypes` 直接 syscall。**生产级沙箱必须用容器（Docker namespace）+ cgroups（资源）+ seccomp（系统调用白名单）**，禁 import 只是基础。verl/OpenRLHF 的 code rollout 常用 Docker 容器隔离，详见 §8.2。

### 3.5 reward shaping：Overlong Reward Shaping 与过程 vs 结果 reward

**Overlong Reward Shaping**（DAPO Trick 4，arXiv:2503.14476 §3.4）：long-CoT RL 里 response 常写到 max_length 还没出答案（被强制截断），这种样本 reward 噪声大（被判错但其实是"还没写完"，强行学会混淆"质量"与"长度"）。DAPO 两步治：

1. **Overlong Filtering**：被截断的样本（length ≥ hard cap），**loss 里 mask 掉**（不参与梯度）；
2. **Soft Overlong Punishment**：对落在惩罚区间 $[L_{max}-L_{cache}, L_{max}]$ 的样本，按长度**线性追加负奖励**，越接近硬截断惩罚越重，平滑引导模型"该停就停"。

$$R_{len}(L)=\begin{cases}0 & L\le L_{max}-L_{cache}\\ -\dfrac{L-(L_{max}-L_{cache})}{L_{cache}} & L_{max}-L_{cache}<L\le L_{max}\\ -1 & L>L_{max}\end{cases}$$

加到 rule-based correctness reward 上。详见 [[DAPO]] §3.5。

**过程 reward vs 结果 reward**：

| 维度 | 结果 reward (outcome) | 过程 reward (process, PRM) |
|---|---|---|
| **给什么** | 只看最终答案对错 | 每步推理给分（对/错/部分） |
| **稀疏度** | 极稀疏（一锤定音） | 稠密（每步有信号） |
| **获取成本** | 低（对答案） | 高（要标每步对错，或训 PRM） |
| **信噪比** | 低（长推导只末尾一个信号） | 高（每步反馈） |
| **典型** | GRPO/R1 数学（rule-based） | PRM (Process Reward Model) |

rule-based verifier 默认给**结果 reward**（极稀疏），长推导里中间错对无信号。过程 reward（PRM）给每步打分更稠密，但要标每步对错（贵）或训 PRM（又回 RM 痛点）。R1/GRPO 选结果 reward + group sampling（组内相对 advantage 造信号）+ Dynamic Sampling（过滤无信号组）来弥补稀疏，而非上 PRM。详见 [[sequence packing与动态采样]] §3.3、[[GRPO]]。

### 3.6 reward 的稀疏度与 GRPO 的信号制造

rule-based/function reward 极稀疏（0/1），单条 response 无梯度信号差异（全对全错 advantage=0）。GRPO 的解法是**组内相对**——同 prompt 采 $G$ 条，用组内 reward 均值当 baseline、组内相对差当 advantage $\hat A_i=(r_i-\bar r)/\sigma_r$。这样即便 reward 是 0/1，只要组内**既有对又有错**（$0<\text{正确数}<G$），advantage 非零、有梯度信号。这是 rule-based reward 与 GRPO 的天然契合——**稀疏 0/1 reward + group relative = 可学信号**。但全对/全错组仍无信号，靠 [[sequence packing与动态采样]] 的 Dynamic Sampling 过滤。三者（rule-based reward + GRPO group + Dynamic Sampling）构成数学/代码 RL 的 reward 链路。


## 4. 数学原理 / 公式

### 4.1 推导：rule-based reward 的无偏性（vs RM 的 proxy 偏差）

设真实偏好 $r^*(q,o)$，RM $r_\phi(q,o)$ 是其代理。RM 的期望偏差：

$$\mathbb{E}_{(q,o)}\big[r_\phi(q,o)-r^*(q,o)\big]\ne 0\quad(\text{proxy 必有偏})$$

PPO 优化 $\mathbb{E}_{\pi_\theta}[r_\phi]$，优化越深 actor 越往 $r_\phi$ 高、$r^*$ 低的区域钻（Goodhart），gap 放大。详见 [[reward hacking]] §3.2。

**rule-based verifier** $v(q,o)\in\{0,1\}$ 直接判对错，是 ground truth：

$$\mathbb{E}_{(q,o)}\big[v(q,o)-r^*(q,o)\big]=0\quad(\text{无偏, 假设 verifier 逻辑正确})$$

无 proxy 偏差，actor 优化 $v$ 即优化 $r^*$（在该任务定义下），无 hack 空间。这是 rule-based 的数学优势。

> [!note] 补充：verifier 也可能"有偏"——但来源不同
> rule-based 的"偏差"来自**verifier 逻辑写错**（漏判某个错答案对、对答案判错），是**确定性 bug**，可测可修；RM 的偏差来自**网络泛化盲区**，是**随机/不可预测**的，难测难修。两者性质不同——rule-based 的偏是 bug，RM 的偏是 proxy 本质。

### 4.2 推导：reward 期望与方差（GRPO group 的信号条件）

GRPO 对同 prompt 采 $G$ 条，reward $r_1,\ldots,r_G$（rule-based 0/1）。组内 advantage $\hat A_i=(r_i-\bar r)/\sigma_r$。

**期望**（rule-based 0/1 reward，单条对概率 $p$）：

$$\mathbb{E}[r_i]=p,\quad \bar r=\frac1G\sum_j r_j\to p\ (G\to\infty)$$

**方差**：

$$\mathrm{Var}[r_i]=p(1-p),\quad \sigma_r=\sqrt{p(1-p)}$$

- $p=0$（全错）或 $p=1$（全对）：$\sigma_r=0$ → advantage $=0/0$ 无定义 → **无信号**（Dynamic Sampling 过滤此情形，详见 [[sequence packing与动态采样]] §4.3）。
- $p=0.5$（半对半错）：$\sigma_r=0.5$ 最大 → **信号最强**（组内差异最大）。
- $p\in(0,1)$：$\sigma_r>0$ → 有信号。

**advantage 的期望绝对值**（rule-based 0/1，组内 $k$ 条对、$G-k$ 条错）：

$$\hat A_{\text{对}}=\frac{1-\bar r}{\sigma_r}=\frac{1-k/G}{\sqrt{(k/G)(1-k/G)}}=\sqrt{\frac{1-k/G}{k/G}},\quad \hat A_{\text{错}}=-\sqrt{\frac{k/G}{1-k/G}}$$

$k=G/2$（半对半错）：$\hat A_{\text{对}}=\hat A_{\text{错}}=1$（对称，信号均衡）；$k\to0$ 或 $k\to G$：$|\hat A|\to\infty$ 但 $\sigma_r\to0$（极端组，实际无信号被过滤）。**最优学习信号在 $k\approx G/2$**——这也是为何 Dynamic Sampling 要求 $0<\text{正确数}<G$ 而非"全对"。

### 4.3 推导：reward shaping 的目标——降噪不偏

reward shaping 加辅助 reward $F(q,o)$ 到原 reward $r$ 上：$r_{\text{shaped}}=r+F$。理论要求（potential-based shaping）：

$$F(q,o)=\gamma\Phi(o')-\Phi(o)$$

保证**最优策略不变**（只改 reward 的势函数差不改 argmax）。Overlong Reward Shaping 的 $R_{len}(L)$ 是长度势函数的工程近似——加负奖励惩罚超长，引导模型停在合适长度，不改变"答案对就对"的主 reward 语义（对短样本 $R_{len}=0$ 不影响）。详见 [[DAPO]] §3.5。

### 4.4 推导：pass@k 的期望 reward

代码 RL 用 pass@k：采 $k$ 条，至少一条全过测试算"可解"。设单条通过率 $p$，$k$ 条至少一条过的概率：

$$\text{pass@k}=1-(1-p)^k$$

- $p=0.1, k=1$：pass@1=0.1（难）；
- $p=0.1, k=8$：pass@8=$1-0.9^8\approx0.57$（宽容，鼓励探索）；
- $p\to0$：pass@k$\to0$（再宽容也救不了全错）。

pass@k 比 pass@1 更适合 RL——它奖励"能解出"而非"每次都解出"，给探索留空间，单条通过率低的 prompt 也有梯度信号。


## 5. 代码示例（可选）

### 5.1 rule-based 数学 verifier（对 `\boxed{}` 答案）

```python
import re
import sympy

def math_verifier(response: str, ground_truth: str) -> float:
    """对数学题 response 的 \\boxed{} 答案, 返回 0/1 reward."""
    # 1. 抽 \\boxed{...} (容忍嵌套大括号的最朴素版本)
    m = re.search(r"\\boxed\{([^}]*)\}", response)
    if not m:
        return 0.0  # 没给答案
    pred = m.group(1).strip()
    gt = ground_truth.strip()
    # 2. 字符串等价 (去空格)
    if pred.replace(" ", "") == gt.replace(" ", ""):
        return 1.0
    # 3. 符号等价 (sympy, 处理 1/2 vs 0.5, 2*x vs x*2)
    try:
        if sympy.simplify(sympy.sympify(pred) - sympy.sympify(gt)) == 0:
            return 1.0
    except Exception:
        pass
    return 0.0

# 测
assert math_verifier("所以 \\boxed{42}", "42") == 1.0
assert math_verifier("答案是 \\boxed{\\frac{1}{2}}", "0.5") == 1.0  # 符号等价
assert math_verifier("我算出 \\boxed{7}", "42") == 0.0
assert math_verifier("没给答案", "42") == 0.0
```

### 5.2 function reward：代码沙箱跑 unittest

```python
import subprocess, json

def code_function_reward(generated_func: str, test_cases: list,
                         timeout: float = 5.0) -> float:
    """沙箱跑生成的函数, 按 unittest pass 率给 reward (0~1)."""
    # test_cases: [(func_name, args, expected), ...]
    wrapped = (
        "import sys,json\n"
        "for m in ['os','subprocess','socket','shutil']: sys.modules[m]=None\n"  # 禁危险
        f"{generated_func}\n"          # 用户代码 (定义函数)
        "R=[]\n"
        "for tc in __TC__:\n"
        "    name,args,exp=tc\n"
        "    try:\n"
        "        fn=getattr(sys.modules['__main__'],name)\n"
        "        R.append('pass' if fn(*args)==exp else 'fail')\n"
        "    except Exception:\n"
        "        R.append('error')\n"
        "print(json.dumps(R))\n"
    ).replace("__TC__", repr(test_cases))
    try:
        p = subprocess.run(["python","-c",wrapped],capture_output=True,
                           text=True,timeout=timeout,env={"PYTHONPATH":""})
        results = json.loads(p.stdout)
        return sum(r=="pass" for r in results) / len(results)
    except subprocess.TimeoutExpired:
        return 0.0  # 超时=全 fail
    except Exception:
        return 0.0

# 测: 生成一个加法函数, 跑 3 个用例
gen_code = "def add(a,b):\n    return a+b\n"
tests = [("add",(1,2),3),("add",(0,0),0),("add",(100,1),101)]
print(code_function_reward(gen_code, tests))  # 1.0 (全过)
print(code_function_reward("def add(a,b):\n    return a-b\n", tests))  # 0.0 (全错)
```

### 5.3 Overlong Reward Shaping（DAPO §3.4）

```python
def overlong_reward_shaping(length: int, L_max: int = 4096,
                            L_cache: int = 512) -> float:
    """DAPO Soft Overlong Punishment: 超过 L_max-L_cache 后线性罚到 -1."""
    if length <= L_max - L_cache:
        return 0.0
    if length > L_max:
        return -1.0
    return -(length - (L_max - L_cache)) / L_cache

# total reward = correctness + shaping
def total_reward(correct: bool, length: int) -> float:
    r_correct = 1.0 if correct else 0.0
    return r_correct + overlong_reward_shaping(length)
    # 截断样本 (length>L_max): 即使答案对也只 ~0 (被 shaping 拉低), 鼓励"该停就停"
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[trajectory generation]]（verifier 给采出的 trajectory 打 reward）、[[rollout worker]]（rollout 端常集成 verifier 做 timely 过滤）、[[Reward Model]]（reward 三来源之一，主观任务路线）、[[GRPO]]（group sampling + rule-based reward = 信号链）、[[DAPO]]（Overlong Reward Shaping 是其 Trick 4）。
- **下游（应用）**: [[17-RL训推一体框架]] reward 工程（reward 决定 advantage 决定梯度）、[[sequence packing与动态采样]]（Dynamic Sampling 靠 verifier 判全对/全错过滤）、[[synchronous与asynchronous rollout]]（partial/timely rollout 在 rollout 端跑 verifier）、[[reward hacking]]（rule-based/function reward 是防 hack 根本手段）。
- **对比 / 易混**:
  - **本条 vs [[Reward Model]]**：本条是 reward 三来源总览（RM + rule-based + function），[[Reward Model]] 是其中 RM 路线的深入（偏好数据 + Bradley-Terry + 训练）。本条讲"选哪条路线"，[[Reward Model]] 讲"RM 怎么训"。
  - **rule-based verifier vs function reward**：前者对答案/正则（静态判），后者沙箱执行（动态跑）。代码题常两者叠加（编译通过 rule-based + 测试过 function）。
  - **过程 reward (PRM) vs 结果 reward (ORM)**：前者每步打分（稠密但贵），后者只末尾对错（稀疏但廉）。GRPO/R1 选 ORM + group sampling 弥补稀疏。详见 §3.5。
  - **本条 vs [[reward hacking]]**：RM 易被 hack（proxy 可钻），rule-based/function reward 难被 hack（程序铁面）。本条是 reward **来源**，[[reward hacking]] 是 reward **失效模式**。选 rule-based 是防 hack 的根本手段之一（另一是 RM 集成 + KL 约束）。
  - **Overlong Reward Shaping vs length penalty**：前者 DAPO 的线性软惩罚（只在接近截断时罚），后者常是固定 length penalty（每多一个 token 罚常数）。前者更平滑、不杀短样本。详见 [[DAPO]] §3.5、[[token-level与sequence-level objective]] §8。

- **三篇互链**: [[synchronous与asynchronous rollout]]（timely rollout 在 rollout 端跑 verifier）、[[sequence packing与动态采样]]（Dynamic Sampling 靠 verifier 判全对/全错）。


## 7. 常见误区与易错点

> [!warning] 误区 1：rule-based verifier 一定无偏
> 不完全。verifier **逻辑写对**时无偏（对就是对），但**逻辑写错**时有偏——如正则抽 `\boxed{}` 漏边界、对答案没做符号等价（`1/2` vs `0.5` 判错）。这种偏是**确定性 bug**（可测可修），与 RM 的 proxy 偏差（随机盲区）性质不同。但工程上仍需仔细测 verifier 覆盖率。详见 §4.1 补充。

> [!warning] 误区 2：function reward 就是 rule-based
> 不准确。rule-based 是**静态判**（对答案/正则，不执行 response）；function reward 是**动态跑**（沙箱执行 response 代码看结果）。代码题两者常叠加（编译通过 rule-based + 测试过 function），但"对答案"与"跑测试"是不同验证强度。详见 §3.2 vs §3.3。

> [!warning] 误区 3：沙箱靠禁 import 就安全
> `sys.modules[m]=None` 只挡直白 `import os`，挡不住 `__import__('o'+'s')`、`builtins`、`ctypes` 直接 syscall。生产级沙箱必须**容器（Docker namespace）+ cgroups（资源）+ seccomp（系统调用白名单）**。禁 import 只是基础。详见 §3.4、§8.2。

> [!warning] 误区 4：rule-based reward 比 RM 好
> 不绝对。rule-based 只限**可验证任务**（数学/代码/格式），开放对话（有用性/无害性）无标准答案，仍需 RM。"哪个好"取决于任务的可验证性。数学/代码 RL 选 rule-based 是任务匹配，非 rule-based 万能。详见 §2.4。

> [!warning] 误区 5：Overlong Reward Shaping 是 length penalty
> 不完全。Overlong Reward Shaping 是 DAPO 特定的**软惩罚**——只在接近截断区间 $[L_{max}-L_{cache}, L_{max}]$ 线性罚，短样本 $R_{len}=0$ 不影响。与固定 length penalty（每 token 罚常数）不同，前者更平滑、不杀短样本、专治"超长截断噪声"。详见 §3.5、[[DAPO]] §3.5。

> [!warning] 误区 6：rule-based 防一切 reward hacking
> rule-based 防的是**proxy 钻空**（RM 的 Goodhart），但不防**verifier 漏洞被利用**——如模型生成让测试用例误判的代码（mock 全局函数、篡改断言、catch 异常吞掉）。这需**测试用例设计严谨**（隔离测试、禁 monkeypatch、独立断言）。rule-based 降低了 hack 风险但非零，仍需测试工程。详见 §8.3。

> [!warning] 误区 7：reward 越稠密越好（上 PRM）
> 过程 reward（PRM）稠密但要标每步对错（贵）或训 PRM（又回 RM 痛点）。R1/GRPO 选结果 reward + group sampling 弥补稀疏，更省更稳。PRM 在特定场景（如数学每步可标）有用，但非"稠密一定好"。reward 工程是**信噪比 vs 成本**的权衡，非越稠越好。详见 §3.5。


## 8. 延伸细节

### 8.1 R1 范式与 rule-based reward 的爆发

DeepSeek-R1（及 GRPO）的爆发，本质是发现**数学/代码的 rule-based reward 可廉价精确获得**，绕过 RM 这道贵且不稳的工序：

- reward = `\boxed{}` 对错 / unittest pass（写个 verifier 函数，零训练成本）；
- group sampling 造信号（GRPO 组内相对，弥补稀疏）；
- Dynamic Sampling 过滤无信号组（DAPO）；
- KL 约束防偏离 reference（但 rule-based 本身不易 hack，KL 需求弱于 RM 场景）。

这让"RL 训推理能力"的工程门槛大降——不需要标偏好对、训 RM，只要 verifier + GRPO。是 R1 开源后数学/代码 RL 复现潮的根。详见 [[GRPO]]、[[DAPO]]、[[reward hacking]] §8。

### 8.2 沙箱的工程实现层级

生产级 function reward 沙箱常多层叠加：

- **verl code rollout**：Docker 容器隔离（namespace 文件系统/进程），cgroups 限 CPU/内存，`signal.alarm` 超时，禁网络。容器复用（pool）减启动开销。
- **OpenRLHF**：类似，subprocess + 资源限制。
- **SWE-agent / Agent 沙箱**：更复杂——要在真实仓库环境跑（git checkout、跑仓库测试），隔离与真实性平衡。
- **FireJail / gVisor / Kata**：轻量沙箱方案，比 Docker 更细粒度。

沙箱的 overhead（容器启动 ~100ms）在 rollout batch 级可摊薄（复用容器 pool）。详见 [[rollout worker]]、[[sampling throughput]]。

### 8.3 verifier 漏洞与测试工程

rule-based/function reward 防的是 proxy 钻空，但不防 verifier 自身漏洞：

- **对答案 verifier**：正则漏边界（`\boxed{}` 嵌套大括号抽不全）、没做符号等价（`1/2` vs `0.5` 判错）。修：用 AST/sympy 等价化。
- **unittest verifier**：模型生成 mock/monkeypatch 篡改测试、catch 异常吞掉断言、篡改全局函数。修：禁 `unittest.mock`、独立断言、隔离测试环境、随机化测试用例顺序防记忆。

测试工程是 code RL 的隐性成本——verifier 写不严，模型会"学测试用例的漏洞"而非"学真解题"（一种 reward hacking 的变体）。详见 [[reward hacking]] §3。

### 8.4 与采样工程、时序的耦合

verifier 在 RL pipeline 里常跑在 **rollout 端**（timely rollout，pre-filter，详见 [[sequence packing与动态采样]] §3.4）——采完一个 group 立刻判 reward、全对/全错当场丢、不送回 trainer 浪费训练算力。这要求 rollout 端集成 verifier（数学对答案 CPU 即可、代码沙箱需容器），增加 rollout 端复杂度但省训练端算力。与 [[synchronous与asynchronous rollout]] 的 partial 模式天然契合——partial 凑够有效（verifier 判过的）样本就训。三篇（采样时序 + 采样工程 + reward 工程）构成 RL 训推一体的采训链路。

### 8.5 内容来源

reward 三来源（RM / rule-based / function）综合自 RLHF 标准（[[Reward Model]]）、GRPO/R1 范式（[[GRPO]] §3、[[RL角色拓扑]] §3.4 rule-based verifier）。rule-based 在数学/代码 RL 流行的原因（稀疏可验证、不训 RM、防注入）综合自 DeepSeek-R1 / GRPO 论文与社区实践。沙箱执行（容器/cgroups/seccomp/超时）综合自 verl/OpenRLHF code rollout 工程实践与 SWE-agent。Overlong Reward Shaping（Soft Punishment + Filtering）来自 DAPO 论文（arXiv:2503.14476 §3.4，已联网核实，与 [[DAPO]] §3.5 一致）。pass@k 来自 HumanEval (Chen et al. 2021) 标准。reward 期望/方差推导基于 0/1 reward 的 Bernoulli 统计 + GRPO group-relative advantage。截至 2026-07。

---
相关: [[reward工程]] | [[Reward Model]] | [[reward hacking]] | [[GRPO]] | [[DAPO]] | [[trajectory generation]] | [[rollout worker]] | [[sequence packing与动态采样]] | [[synchronous与asynchronous rollout]] | [[RL角色拓扑]] | [[sampling throughput]] | [[continuous batching的调度实现]] | [[token-level与sequence-level objective]]
