# rollout worker

> **所属章节**: [[训练架构]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（需懂 RL rollout 流程 + 推理引擎基础）

## 1. 一句话定义

**rollout worker（rollout 工作器 / 采样 worker）** 是 [[learner worker split]] 架构中专门跑**强化学习 rollout**（生成 trajectory）的 worker：用当前 policy $\pi_\theta$ **自回归生成** response/trajectory，连同 `logp_old`、reward 等通过 buffer 传给 learner 训练。在 [[RLHF (PPO)]] 中，rollout worker 用 **vLLM/SGLang 等推理引擎**加速生成（[[continuous batching]] / PagedAttention / [[KV cache management]]），与 learner（FSDP/DeepSpeed 训练）分离，通过 [[weight sync mechanism]] 接收 learner 的新 $\theta$。它是 [[PPO optimization]] §8.3 训推分离架构的**采样侧核心**，生成吞吐是 RLHF 训练的关键瓶颈。与 [[inference worker]] 的区别：rollout worker **为训练采数据**，inference worker **为部署服务用户**。

> [!note] 别名
> rollout worker = sampling worker = generation worker = actor worker（在 IMPALA 语境）。都指 RL 中跑生成的 worker。

## 2. 为什么需要它（动机与背景）

[[PPO optimization]] 每轮 PPO 迭代需新 rollout（actor $\pi_\theta$ 自生成 response），但大模型自回归生成极贵：
1. **生成慢**：逐 token 自回归，长序列 $T$ 步，每步注意 $O(T)$，总 $O(T^2)$；秒~分钟级；
2. **生成是瓶颈**：RLHF 训练时间 60~80% 花在 rollout 生成（非训练），是吞吐瓶颈；
3. **需推理引擎优化**：生成用 [[continuous batching]]（动态拼批）、PagedAttention（KV cache 分页，减碎片）、[[speculative decoding]]（投机解码）等推理优化，这些 vLLM/SGLang 专长，训练框架（FSDP）不优；
4. **与训练分离**：生成（推理引擎）与训练（FSDP）框架异构，需不同进程；
5. **多 worker 并行**：N 个 rollout worker 并行生成，吞吐 N×，是 RLHF 扩展采样吞吐的主手段。

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

### 3.2 rollout worker 的组件

| 组件 | 作用 | 优化点 |
|---|---|---|
| **actor $\pi_\theta$ 副本** | 自回归生成 response | 推理引擎加载 |
| **reference $\pi_{\text{ref}}$（可选）** | 算 logp_ref 用于 KL | 冻结，半精度 |
| **RM $r_\phi$（可选）** | 算 reward | 冻结，半精度 |
| **推理引擎** | vLLM/SGLang 加速生成 | continuous batching / PagedAttention |
| **buffer 接口** | 写 trajectory 给 learner | 共享内存/Ray object store |

> [!tip] reward/ref 的位置
> reference 与 RM 的前向可在 rollout worker 内（省数据传输），也可独立成 reward worker（专跑 RM/ref，省 rollout worker 显存）。verl 支持两种模式。

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
