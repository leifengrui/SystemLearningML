# rollout worker

> **所属章节**: [[训练架构]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（需懂 RL rollout 流程 + 推理引擎基础）

## 1. 一句话定义

**rollout worker（rollout 工作器 / 采样 worker）** 是 [[learner worker split]] 架构中专门跑**强化学习 rollout**（生成 trajectory）的 worker：用当前 policy $\pi_\theta$ **自回归生成** response/trajectory，连同 `logp_old`、reward 等通过 buffer 传给 learner 训练。在 [[RLHF (PPO)]] 中，rollout worker 用 **vLLM/SGLang 等推理引擎**加速生成（[[continuous batching]] / PagedAttention / [[KV cache management]]），与 learner（FSDP/DeepSpeed 训练）分离，通过 [[weight sync mechanism]] 接收 learner 的新 $\theta$。它是 [[PPO optimization]] §8.3 训推分离架构的**采样侧核心**，生成吞吐是 RLHF 训练的关键瓶颈。与 [[inference worker]] 的区别：rollout worker **为训练采数据**，inference worker **为部署服务用户**。
> [!note] 解答：rollout worker vs [[inference worker]] 的彻底辨析（不止 §3.4 那张表）
> 两者表面都是"用大模型自回归生成 token"，初学者最易混。下面从**七个维度**逐一拆，最后给口诀。
>
> | 维度 | rollout worker（采样 worker） | [[inference worker]]（部署 worker） |
> |---|---|---|
> | **1. 存在目的** | 给 **learner 喂训练数据**，产物是更新后的 $\theta$ | 给 **用户返回回答**，产物是用户满意 |
> | **2. 数据去向** | 生成结果 → 进 **buffer/replay** → learner 消费 | 生成结果 → **HTTP/gRPC 流** → 用户屏幕 |
> | **3. 权重 $\theta$** | **频繁变**：learner 每 N 步更新后经 [[weight sync mechanism]] 推过来 | **冻结**：部署快照，只在版本升级时换 |
> | **4. 采样策略** | 按 $\pi_\theta$ 采样**保探索**（温度可调、nucleus，但目的=多样性，非质量） | 按部署配置（温度/nucleus/beam），目的=**质量+可控** |
> | **5. 优化目标** | **吞吐优先**（tokens/sec 越高越好，批越大越好） | **延迟优先**（TTFT<1s、TPOT<50ms），批为延迟让路 |
> | **6. 框架选型** | vLLM/SGLang（重生成吞吐、RadixAttention 适合 GRPO 组内多采样） | vLLM/TensorRT-LLM/Triton（重服务延迟、量化 INT4/FP8） |
> | **7. 所属阶段** | **PT（训练）内部**，是 [[learner worker split]] 的采样侧 | **PD（部署）**，是 [[训推分离]] 的服务侧 |
>
> **为什么易混**：因为"生成"这一动作的代码几乎一样（都是 `vLLM.generate(prompts)`）。区别不在"怎么生成"，而在**生成出来给谁、$\theta$ 变不变、为谁优化**——即"同动作，不同语境"。
>
> **一个判别小测验**：拿掉步骤 4/5（logp_ref/reward）后它还是不是 inference worker？见下一条批注的解答——**不是**，仍是 rollout worker，因为数据进 buffer 不进用户、$\theta$ 仍频繁 sync、采样仍为探索。
>
> **口诀**：**"rollout 为训采数据喂 learner，inference 为用服务喂用户；前者 θ 天天换重吞吐，后者 θ 冻成冰压延迟。"** 详见 §3.4 表与 [[inference worker]] §3.3。
> [!note] 别名
> rollout worker = sampling worker = generation worker = actor worker（在 IMPALA 语境）。都指 RL 中跑生成的 worker。

## 2. 为什么需要它（动机与背景）

[[PPO optimization]] 每轮 PPO 迭代需新 rollout（actor $\pi_\theta$ 自生成 response），但大模型自回归生成极贵：
1. **生成慢**：逐 token 自回归，长序列 $T$ 步，每步注意 $O(T)$，总 $O(T^2)$；秒~分钟级；
2. **生成是瓶颈**：RLHF 训练时间 60~80% 花在 rollout 生成（非训练），是吞吐瓶颈；
3. **需推理引擎优化**：生成用 [[continuous batching]]（动态拼批）、PagedAttention（KV cache 分页，减碎片）、[[speculative decoding]]（投机解码）等推理优化，这些 vLLM/SGLang 专长，训练框架（FSDP）不优；
4. **与训练分离**：生成（推理引擎）与训练（FSDP）框架异构，需不同进程；
5. **多 worker 并行**：N 个 rollout worker 并行生成，吞吐 N×，是 RLHF 扩展采样吞吐的主手段。
> [!note] 解答：多 rollout worker 并行 ≠ 严格意义的 DP，但"像 DP 的副本复制那一半"
> 直觉上像——N 个 worker 各持一份完整 $\pi_\theta$ 副本、各采不同 prompt、并行提吞吐——这和 **DP（[[Data Parallel]]）每 rank 持全量模型、切数据**的"副本复制"那半确实同构。但**只像了一半**，关键区别在另一半：
>
> | 维度 | 多 rollout worker 并行 | DP（Data Parallel） |
> |---|---|---|
> | **每副本干什么** | 只**前向生成** trajectory（采样） | **前向+反向**算梯度 |
> | **副本间通信** | **无梯度 all-reduce**（各采各的，互不交换） | **必须 all-reduce 梯度**，否则各 rank 梯度不同步→参数发散 |
> | **参数怎么来** | 从 learner 经 [[weight sync mechanism]] **单向下发**（learner→worker） | 各 rank 自己算梯度→step→**自己更新**，靠 all-reduce 保一致 |
> | **参数怎么走** | worker **不回传**任何东西给 learner（只回传 trajectory 数据，非梯度） | 梯度是回传给"自己"做 step 的 |
> | **目的** | 提**采样吞吐**（数据并行） | 提**训练算力**（梯度并行） |
> | **模型切分** | 大模型时每 worker 内部常再上 **TP**（[[Tensor Parallel]]），即"worker 间副本复制 + worker 内 TP 切分" | DP 就是整模型复制，不再切；要切权重上 Megatron TP/PP |
>
> **所以严格说**：多 rollout worker 是"**模型副本并行 / 数据并行的采样侧**"，借用了 DP 的"复制模型+切数据"思想，但**没有 DP 的梯度 all-reduce 环节**——因为 worker 不训，不产梯度，无需同步梯度。它是 [[learner worker split]] 里 worker 侧的扩展方式，与 learner 侧的 DP 是**对称但不同性质**的两套并行。
>
> **类比记忆**：DP 像"N 个工人各自算账本汇总"（算完要对账=all-reduce）；多 rollout worker 像"N 个采蜜蜂各采各的花"（采完直接入巢=buffer，互不对账）。采蜜不需要对账，因为蜂不记账本（不训）。
>
> **工程细节**：verl/OpenRLHF 里 rollout worker 数量与 learner 数量**独立配置**（如 8 worker + 4 learner），用 Ray 调度；weight sync 是 learner→所有 N 个 worker 广播新 $\theta$。详见 §8.2、[[Data Parallel]]、[[weight sync mechanism]]。
故 rollout worker 是大模型 RL 的工程必需——用专用进程 + 推理引擎，把生成优化到极致，与训练异步并行。

## 3. 核心概念详解

### 3.1 rollout worker 的标准流程

```
rollout worker 一轮:
1. 接收 weight sync(从 learner 拿最新 θ)      ← [[weight sync mechanism]]
2. 用 π_θ 对 N 个 prompt 自回归生成 response   ← vLLM/SGLang 加速
3. 算 logp_old(生成时的 token 级 logp)         ← 生成副产物
4. (可选)算 logp_ref(reference model 前向)     ← 可在 reward worker 或本 worker
5. (可选)算 reward(RM 前向)                    ← 可在 reward worker 或本 worker
6. 组装 trajectory:(prompt, response, logp_old, logp_ref, reward)
7. 写入 buffer                                 ← 传给 learner
8. 回到 1(异步:不等 learner)
```
> [!note] 解答：$\theta$ 是权重，$\pi_\theta$ 是"用这些权重构成的策略（模型）"
> **$\theta$（theta）**：模型的**全部可学习参数**堆成一个张量集合——Transformer 里就是所有 `Linear` 的 `weight`/`bias`、`Embedding` 的权重、LayerNorm 的 `weight`/`bias` 等。代码层面就是 `model.parameters()` 迭代出的那堆 `nn.Parameter`。说"从 learner 拿最新 $\theta$"="拿最新权重/最新 state_dict"=[[weight sync mechanism]]。所以**是的，$\theta$ 就是权重**。
>
> **$\pi_\theta$（pi-theta）**：把上面那堆权重 $\theta$ **装进网络结构**后，得到的"**给定状态 $s$ 输出动作分布**"的函数，即**策略**。在 LLM-RL 语境：
> $$\pi_\theta(a_t\mid s_t)=\text{softmax}(\text{logits}_\theta(s_t))_{a_t}$$
> 其中 $s_t$=已生成前缀（prompt+已生成 token），$a_t$=下一个 token，$\text{logits}_\theta$=用权重 $\theta$ 跑一次前向得到的词表得分向量。所以 $\pi_\theta$ 不是"另一个东西"，它就是**"带着权重 $\theta$ 的那个模型，被当作策略看"**——同一组权重，从"参数"角度叫 $\theta$，从"行为函数"角度叫 $\pi_\theta$。
>
> **关系一句话**：$\theta$ 是**参数值**（静态的数），$\pi_\theta$ 是**装上这组参数后的函数**（动态的行为）。改 $\theta$ → $\pi_\theta$ 的输出分布变 → 采样出的 trajectory 变。这就是为什么 learner 一更新 $\theta$ 就要 weight sync 给 rollout worker：让 worker 用**新 $\pi_\theta$** 采新数据，否则采的是旧策略的数据（[[stale policy problem]]）。
>
> **代码对照**（本文件 §5）：
> ```python
> policy = Policy(V, D)              # 网络结构(π 的壳)
> # policy.parameters() 里的所有张量 = θ
> probs = torch.softmax(policy(out)[:,-1,:], -1)  # 这一步 = π_θ(·|s_t)
> nxt = torch.multinomial(probs, 1)  # a_t ~ π_θ(·|s_t)
> ```
> **记法**：下标 $\theta$ 在数学里就是"此函数由参数 $\theta$ 决定"。$\pi$ 不带下标=抽象策略，$\pi_\theta$=具体用 $\theta$ 实例化的那个策略。RL 里 learner 在做的事就是 $\theta\leftarrow\theta+\alpha\nabla_\theta J(\pi_\theta)$，即调权重让策略变好。详见 [[Actor-Critic]]、[[MDP]]、[[PPO optimization]]。
> [!note] 解答：去掉步骤 4/5 仍是 rollout worker，不会变成 inference worker——判别看"四件套"不看可选附加
> 这个问题切中了易混点。步骤 4（算 logp_ref）和 5（算 reward）确实是**RLHF-PPO 特有**的附加产物（给 PPO 算 KL 和 advantage 用）。但**它们不是区分 rollout worker / inference worker 的判据**。真正的判据是下面这"四件套"，**只要四件套中任一为"训练侧"就是 rollout worker**：
>
> | 判据 | rollout worker（哪怕去掉 4/5） | inference worker |
> |---|---|---|
> | **① 输出去哪** | 进 **buffer** 给 learner 训 | 返**用户** |
> | **② $\theta$ 变不变** | 频繁 **weight sync** 更新 | **冻结** |
> | **③ 采样策略** | 按 $\pi_\theta$ 采样**保探索**（温度为多样性，非质量） | 部署配置（温度为**质量+可控**） |
> | **④ 优化目标** | **吞吐**优先（大批、批为吞吐） | **延迟**优先（TTFT/TPOT，批为延迟让路） |
>
> 去掉 4/5 后的 worker：照样把 response 写进 buffer（①训练侧）、$\theta$ 仍频繁 sync（②训练侧）、仍按 $\pi_\theta$ 探索采样（③训练侧）、仍重吞吐（④训练侧）——**四条全占**，所以**铁定还是 rollout worker**。
>
> **什么时候真的会去掉 4/5？** 现实中很常见，并不影响身份：
> - **GRPO / REINFORCE 无 KL**：不需要 logp_ref，去掉步骤 4——DeepSeek-R1 的 GRPO 就是这样（只在 loss 端加 β·KL，但 ref logp 可在 learner 端补算）；
> - **reward 外置到 reward worker**：把步骤 5 搬到独立的 reward worker（verl 支持），rollout worker 只管生成——此时 rollout worker 只剩 1/2/3/6/7，**它依然是 rollout worker**，只是把 RM/ref 活外包了；
> - **rule-based reward**：reward 是规则算的（数学题对错），不需要 RM 前向，步骤 5 本就没有。
>
> **那 inference worker 长什么样对照？** inference worker 的流程（见 [[inference worker]] §3.1）是：加载固定 $\theta$ → 接用户请求 → continuous batching 生成 → **流式返用户** → $\theta$ 不变。它**根本没有 buffer、没有 weight sync 频繁更新、采样为质量不为探索**——这四条全占部署侧，才是 inference worker。
>
> **一句话**：4/5 是"RLHF-PPO 给 rollout worker 加的活"，去掉只是"少干点活的 rollout worker"，不会跨过 PT/PD 那条线变成服务用户的 inference worker。判身份看四件套，不看附加任务。
### 3.2 rollout worker 的组件

| 组件                                   | 作用                     | 优化点                                  |
| ------------------------------------ | ---------------------- | ------------------------------------ |
| **actor $\pi_\theta$ 副本**            | 自回归生成 response         | 推理引擎加载                               |
| **reference $\pi_{\text{ref}}$（可选）** | 算 logp_ref 用于 KL       | 冻结，半精度                               |
| **RM $r_\phi$（可选）**                  | 算 reward               | 冻结，半精度                               |
| **推理引擎**                             | vLLM/SGLang 加速生成       | continuous batching / PagedAttention |
| **buffer 接口**                        | 写 trajectory 给 learner | 共享内存/Ray object store                |

> [!tip] reward/ref 的位置
> reference 与 RM 的前向可在 rollout worker 内（省数据传输），也可独立成 reward worker（专跑 RM/ref，省 rollout worker 显存）。verl 支持两种模式。
> [!note] 解答：reward worker 与 reference worker 是干什么的？要不要单起？
> §3.2 表里出现了两个"冻结的、只前向的"模型——**reference model $\pi_{\text{ref}}$** 和 **reward model $r_\phi$**。它们各自前向算一个数，给 PPO 用。这两个前向**可以塞在 rollout worker 里**，也可以**拆出去单独成 worker**。下面分别讲清"是干什么的"+"要不要单起"。
>
> ### reference model（参考模型 / $\pi_{\text{ref}}$）是干什么的
> 它是**训练开始前冻结的初始 actor 副本**（通常是 SFT 后的模型快照），整个 RLHF 过程**不更新**。作用：算 **logp_ref** = "如果用参考模型生成同一个 response，每个 token 的对数概率"。PPO 用它和 logp_old 算 **KL 散度**：
> $$\text{KL}(\pi_\theta\|\pi_{\text{ref}}) \approx \frac{1}{T}\sum_t \big[\log\pi_\theta(a_t|s_t)-\log\pi_{\text{ref}}(a_t|s_t)\big]$$
> KL 项（$\beta\cdot\text{KL}$）作为惩罚加进 reward，**防止 actor 跑偏到 reward hacking**（见 [[KL penalty与KL control]]、[[reward hacking]]）。没有 reference model，PPO 就失去"别离原模型太远"的锚，容易训崩（[[policy collapse]]）。
>
> ### reward model（$r_\phi$）是干什么的
> 它是用人类偏好数据训出来的**奖励模型**（[[Reward Model]]），输入 (prompt, response) 输出一个标量 reward $r$，衡量 response 的"人类偏好程度"。PPO 用 $r$ 减去 KL 惩罚得到**最终 reward**：$R = r - \beta\cdot\text{KL}$，再算 advantage 训 actor。**rule-based reward**（如数学题对错）则不需要 RM 前向，跳过此步。
>
> ### 为什么要"单起一个 reward worker"（即拆出去）
> 这两个前向（ref + RM）虽然只前向不训练，但**占显存 + 占算力**：
> - **显存**：ref 和 RM 各是一份完整模型副本（GB 级），塞在 rollout worker 里 → rollout worker 显存爆炸，挤压 KV cache → 生成并发下降；
> - **算力**：每条 trajectory 都要跑两次额外前向（ref + RM），与生成争 GPU；
> - **解耦**：ref/RM 是**冻结的、批处理友好的**前向，和 rollout 的"自回归逐 token 生成"负载特性不同，分开调度更优。
>
> 故工程上常把它们**独立成 reward worker**（专跑 ref + RM 前向，冻结半精度，可批处理），rollout worker 只管生成。两者解耦后：rollout worker 显存全给 KV cache 提生成吞吐，reward worker 批量算 ref/reward 提算力利用率。
>
> ### 三种部署模式对比
> | 模式 | ref/RM 在哪 | 优点 | 缺点 |
> |---|---|---|---|
> | **A. 合并到 rollout worker** | 同进程跑生成+ref+RM | 数据不搬运，简单 | rollout 显存挤 KV cache，吞吐降 |
> | **B. 独立 reward worker** | 单独进程专跑 ref+RM | rollout 显存全给生成，reward 批处理高效 | 多一次 trajectory 数据传输 |
> | **C. learner 端补算** | ref/RM 在 learner 侧前向 | rollout 最轻 | learner 显存增 + 传 logits 而非 logp |
>
> verl/OpenRLHF 支持 A 和 B（`actor.rollout.sleep_level` / `reward_model.inference_mode` 配置）。大模型（生成显存紧张）选 B；小模型选 A 省事。DeepSeek-R1 GRPO 无需 ref（或 learner 端补），走 C 简化。
>
> **注意身份**：reward worker 也是**为训练采数据**（算 reward/KL 给 learner），不是服务用户——所以它和 [[rollout worker]] 同属 PT 侧，**不是** [[inference worker]]。判别仍看本文件上面的"四件套"。详见 [[Reward Model]]、[[KL penalty与KL control]]、[[reward hacking]]、[[policy training (PT)]]。
### 3.3 推理引擎加速（rollout worker 的核心优化）

| 技术 | 机制 | 详见 |
|---|---|---|
| **[[continuous batching]]** | 动态拼批，新生成请求随时插入，GPU 不闲置 | [[continuous batching]] |
| **PagedAttention** | KV cache 分页存储，减显存碎片，提并发 | [[KV cache management]] |
| **[[speculative decoding]]** | 小模型草稿 + 大模型校验，减大模型前向次数 | [[speculative decoding]]（待展开） |
| **tensor parallel** | 生成时模型并行（大模型单卡装不下） | [[Tensor Parallel]] |

这些是 vLLM/SGLang 的核心，使 rollout worker 生成吞吐数倍于朴素 PyTorch 生成。

### 3.4 rollout worker vs inference worker（关键区分）

| 维度 | rollout worker | [[inference worker]] |
|---|---|---|
| 目的 | 为**训练**采数据 | 为**部署**服务用户 |
| 输入 | prompt 集（离线/在线采样） | 用户请求（实时） |
| 输出 | trajectory（进 buffer 给 learner） | response（返用户） |
| 采样策略 | 按 $\pi_\theta$ 采样（保探索，温度/nucleus） | 按部署配置（温度/nucleus/beam） |
| 延迟要求 | 吞吐优先（批处理） | 延迟敏感（单请求快） |
| θ 更新 | 频繁（weight sync 从 learner） | 不更新（部署固定 θ） |
| 框架 | vLLM/SGLang（吞吐优化） | vLLM/TensorRT-LLM（延迟优化） |

### 3.5 生成吞吐的瓶颈

rollout worker 的生成吞吐（tokens/sec）受：
- **模型规模**：大模型单 token 前向慢；
- **序列长度**：自回归 $O(T^2)$ 注意力，长序列慢；
- **批大小**：批大 GPU 利用率高，但显存限（KV cache）；
- **采样策略**：nucleus/top-k 比贪心略慢（排序+采样）；
- **weight sync 频率**：频繁同步打断生成。

优化：vLLM continuous batching + PagedAttention + 多 worker 并行 + 合理 sync 频率。详见批3 [[sampling throughput]]。

## 4. 数学原理 / 公式

### 4.1 自回归生成的时间复杂度

生成 $T$ token，每步 $t$ 的注意 $O(t)$（因 KV cache 累积），总：
$$
T_{\text{gen}}=\sum_{t=1}^T O(t)=O(T^2)
$$

长序列生成慢。vLLM 的 PagedAttention 不改复杂度，但减常数（显存碎片/并发）。

### 4.2 生成吞吐

$$
\text{throughput}=\frac{B\cdot T}{T_{\text{gen}}}\quad\text{(tokens/sec)}
$$

$B$ 批大小。vLLM continuous batching 让 $B$ 动态最大化（显存允许内），提吞吐。

### 4.3 按 $\pi_\theta$ 采样的随机性

rollout worker 不是贪心 $\arg\max$，而是按 $\pi_\theta$ 采样：
$$
a_t\sim\pi_\theta(\cdot|s_t)
$$

保探索（RL 需 off-policy 数据的多样性）。可用 temperature/nucleus 调随机性：
- 训练早期：温度高，增探索；
- 训练后期：温度低，利用已学。
详见 [[trajectory generation]]（批3）。

## 5. 代码示例

```python
import torch, torch.nn as nn, queue, time

class Policy(nn.Module):
    def __init__(self, v, d):
        super().__init__(); self.emb=nn.Embedding(v,d); self.lstm=nn.LSTM(d,d,batch_first=True); self.head=nn.Linear(d,v)
    def forward(self, x):
        h,_=self.lstm(self.emb(x)); return self.head(h)

V, D = 50, 32
policy = Policy(V, D)
buffer = queue.Queue(maxsize=10)

def rollout_worker(prompts, max_len=10, T=1.0):
    """rollout worker:用 π_θ 自回归生成 response,存 buffer"""
    with torch.no_grad():
        out = prompts.clone()
        logp_sum = torch.zeros(prompts.shape[0])
        for _ in range(max_len):
            logits = policy(out)[:, -1, :] / T          # temperature
            probs = torch.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)            # 按 π_θ 采样(非贪心)
            logp_sum += torch.log(probs.gather(1, nxt).squeeze(-1))
            out = torch.cat([out, nxt], 1)
        # 组装 trajectory
        traj = {'prompt': prompts, 'response': out[:, prompts.shape[1]:],
                'logp_old': logp_sum, 'len': max_len}
        buffer.put(traj)
    return traj

# 模拟多 prompt rollout
prompts = torch.randint(0, V, (4, 4))
traj = rollout_worker(prompts, max_len=8, T=1.0)
print(f"rollout 生成 {traj['response'].shape}, logp_old={traj['logp_old']}")
print("trajectory 进 buffer,learner 将取训;实际用 vLLM 加速生成")
print("weight sync:learner 更新后把新 θ 同步给此 worker")
```

> [!tip] 实际工程（verl + vLLM）
> verl 的 rollout worker 用 vLLM `LLM.generate()`：批量 prompt 进，continuous batching 生成，自动算 logp。weight sync 用 `vllm.load_format('dummy')` + 参数注入。这是 RLHF 高吞吐生成的标准。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[learner worker split]]（架构基础）、[[PPO optimization]] §8.3（RLHF rollout 需求）、[[MDP]]（rollout = MDP 交互）、[[RLHF (PPO)]]。
- **下游（应用）**: [[训推分离]]、[[weight sync mechanism]]、[[stale policy problem]]、[[trajectory generation]]、[[sampling throughput]]、[[batching strategy]]（rollout 的细化）、[[continuous batching]]、[[KV cache management]]。
- **对比 / 易混**:
  - **rollout worker vs [[inference worker]]**：见 §3.4 表。采数据 vs 服务用户。
  - **rollout worker vs learner**：前者采样（前向生成），后者训练（forward+backward+step）。
  - **rollout worker vs reward worker**：前者跑 actor 生成，后者（若有）跑 RM/ref 算 reward。可合并可分离。
  - **rollout worker 的采样 vs 推布的采样**：前者按 $\pi_\theta$ 采样保探索（温度可调），后者按部署配置（温度/nucleus）。目的不同。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"rollout worker = inference worker"**：错。前者为训练采数据，后者为部署服务用户。目的/输入/θ 更新都不同。
> 2. **"rollout 用贪心"**：错。RL 需按 $\pi_\theta$ 采样保探索，贪心无探索数据 → PPO 学不动。
> 3. **"rollout worker 也 backward"**：错。只前向生成，不 backward/step（那是 learner）。但需算 logp_old（前向副产物）。
> 4. **"rollout 生成免费"**：错。是大模型 RL 主要开销（60~80% 时间），是瓶颈。
> 5. **"rollout worker 用训练框架"**：低效。应用 vLLM/SGLang（推理优化），非 FSDP（训练优化）。
> 6. **"weight sync 越频繁越好"**：错。频繁同步打断生成 + 同步延迟大（θ 大）。需 tradeoff（[[weight sync mechanism]]）。
> 7. **"rollout worker 单个够"**：吞吐不足。多 worker 并行提采样吞吐，是 RLHF 扩展主手段。
> 8. **"rollout 温度固定"**：可调。训练早期高温探索，后期低温利用，是调参点。

## 8. 延伸细节

### 8.1 vLLM/SGLang 在 RLHF rollout 的应用

- **vLLM**：PagedAttention + continuous batching，吞吐优化，RLHF 主流（verl/OpenRLHF 默认）；
- **SGLang**：RadixAttention（前缀复用），对同 prompt 多采样场景（如 GRPO 组内 G 条）更优；
- **TensorRT-LLM**：NVIDIA 推理引擎，延迟优化，部署侧更多。
选择取决于吞吐 vs 延迟需求（rollout 重吞吐）。

### 8.2 多 rollout worker 并行

- N 个 worker 各自生成，buffer 汇集，吞吐 N×；
- 负载均衡：长 prompt 分配慢 worker，短 prompt 快 worker，动态调度；
- weight sync：N 个 worker 需都同步新 θ，广播或轮流；
- 显存：每个 worker 一份 $\pi_\theta$ 副本，N worker 显存 N×（大模型需 TP 切分）。

### 8.3 rollout 的采样策略调参

- **temperature**：高探索 vs 低利用；
- **top-p / top-k**：截断长尾，保质量；
- **repetition penalty**：防重复（[[mode collapse]] 的采样层防法）；
- **多候选生成**：同 prompt 生成多条（GRPO 的组内 G 条）。
详见 [[trajectory generation]]（批3）。

### 8.4 rollout worker 的 stale policy 问题

异步 split 中，rollout worker 用 $\theta_t$ 生成，但 learner 已更新到 $\theta_{t+k}$。trajectory 的 logp_old 对应 $\theta_t$，learner 用 $\theta_{t+k}$ 算 ratio $\rho=\pi_{\theta_{t+k}}/\pi_{\theta_t}$，$k$ 大时 ratio 失真（[[stale policy problem]]）。防法：weight sync 定期同步 + PPO clip 限制 ratio + target_kl 早停。

### 8.5 rollout 的 KV cache 显存账

rollout 生成时 KV cache 占显存：$2\cdot L\cdot d\cdot T\cdot B$（$L$ 层数，$d$ 隐维度，$T$ 序列长，$B$ 批大小）。长序列 + 大批 → KV cache 爆显存。PagedAttention 分页减碎片，continuous batching 动态调 $B$ 适配显存。详见 [[KV cache management]]、[[activation memory]]。

---
相关: [[训练架构]]、[[learner worker split]]、[[inference worker]]、[[asynchronous training]]、[[PPO optimization]]、[[RLHF (PPO)]]、[[训推分离]]、[[weight sync mechanism]]、[[stale policy problem]]、[[trajectory generation]]、[[sampling throughput]]、[[batching strategy]]、[[continuous batching]]、[[KV cache management]]、[[MDP]]
