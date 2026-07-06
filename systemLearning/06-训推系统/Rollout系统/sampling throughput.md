# sampling throughput

> **所属章节**: [[Rollout系统]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（需懂生成瓶颈 + vLLM 优化）

## 1. 一句话定义

**sampling throughput（采样吞吐）** 是 [[rollout worker]] 做 [[trajectory generation]] 的**生成性能**（tokens/sec/GPU），是 RLHF 训练的**主要瓶颈**。auto-regressive 生成每 token 一次前向 + KV cache 增长，导致单请求慢；优化靠 [[continuous batching]]（动态批）、[[KV cache management]]（PagedAttention 减显存碎片）、[[speculative decoding]]（小模型预测减大模型前向）、[[Tensor Parallel]]（切分大模型）、batch size 调优。度量 tokens/sec/ replica、并发数、延迟。生成吞吐常占 RLHF 总耗时 60~80%，是工程优化重点。

> [!note] 生成 vs 训练吞吐
> - **生成吞吐（本篇）**：rollout 侧 auto-regressive 生成 tokens/sec，**RLHF 主要耗时**。
> - **训练吞吐**：learner 侧前向+反向 FLOPS，GPU compute-bound，相对快。
> RLHF 常是"生成 bound"，故 rollout 吞吐优化是关键。

## 2. 为什么需要它（动机与背景）

RLHF 训练的耗时主要在生成：
1. **auto-regressive 慢**：每 token 一次前向，$T$ token 则 $T$ 次前向（vs 训练 1 次）；
2. **生成是主耗时**：RLHF 中生成占 60~80%（PPO 每轮需大量 trajectory）；
3. **并发受显存限**：KV cache 随 token 增长，显存限并发数；
4. **大模型慢**：70B 模型单 token 前向重，吞吐低；
5. **需高吞吐**：RLHF 需百万级 trajectory，吞吐低则训练慢/贵。

故 sampling throughput 是 RLHF 工程的核心指标，决定训练成本与周期。vLLM 等 inference engine 的优化（continuous batching/PagedAttention）都为提生成吞吐。

## 3. 核心概念详解

### 3.1 生成的瓶颈

| 瓶颈 | 原因 | 影响 |
|---|---|---|
| **auto-regressive** | 每 token 一次前向，串行 | 长 response 慢 |
| **KV cache 增长** | 每 token 存 K/V，显存涨 | 限并发数 |
| **显存碎片** | 不同长度请求混合，碎片浪费 | 并发降 |
| **batch 静态** | 传统 batch 等齐长度，浪费 | GPU 利用率低 |
| **大模型单次前向重** | 70B 参数，每 token 计算大 | 吞吐低 |

### 3.2 吞吐的优化

| 优化 | 机制 | 增益 |
|---|---|---|
| **[[continuous batching]]** | 动态批，请求进出不等待 | 2~4× |
| **[[KV cache management]] (PagedAttention)** | 分页 KV cache，减碎片 | 并发 2~4× |
| **[[speculative decoding]]** | 小模型预测多 token，大模型并行验证 | 2~3× |
| **[[Tensor Parallel]]** | 大模型切多卡 | 单卡吞吐换多卡总吞吐 |
| **batch size 调优** | 大 batch 提 GPU 利用率 | 线性至显存限 |
| **prefix caching** | 复用 prompt 的 KV cache | 减 TTFT |
| **量化（INT8/INT4）** | 减显存+计算 | 吞吐 1.5~2× |

### 3.3 度量指标

| 指标 | 定义 | 典型 |
|---|---|---|
| **tokens/sec/replica** | 每副本每秒生成 token | vLLM 70B ~1000-2000 |
| **并发数** | 同时生成请求数 | 显存限 |
| **TTFT** | 首 token 延迟 | < 1s |
| **TPOT** | 每 token 延迟 | < 50ms |
| **GPU 利用率** | 生成时 GPU busy | 高效 > 80% |
| **生成/训练耗时比** | rollout 占 RLHF 总耗时 | 60~80% |

### 3.4 吞吐与并发的 tradeoff

- **并发高**：continuous batching 多请求，吞吐升（GPU 利用率高）；
- **但显存限**：每请求 KV cache 占显存，并发上限 = 显存/KV per request；
- **batch size**：大 batch 提吞吐，但显存限 + 延迟升（TPOT 增）；
- **延迟 vs 吞吐**：高吞吐需大 batch，但单请求延迟升（排队）。
rollout 场景重吞吐（非延迟），故 batch 尽大（至显存限）。

### 3.5 rollout 的吞吐优化策略

1. **vLLM 引擎**：continuous batching + PagedAttention（默认优化）；
2. **大 batch**：prompt 池大，batch 拉满显存；
3. **tensor parallel**：大模型切多卡（70B 用 TP=8）；
4. **speculative decoding**：减大模型前向次数；
5. **多 worker 并行**：数据并行 rollout，多 GPU 各生成；
6. **异步生成**：worker 生成与 learner 训练重叠（[[asynchronous training]]）；
7. **prompt 长度管理**：限 prompt 长度，控 KV cache 增长。

## 4. 数学原理 / 公式

### 4.1 auto-regressive 的耗时

$$
t_{\text{gen}}=T\cdot t_{\text{forward}}\quad\text{（T token,每 token 一次前向）}
$$

vs 训练 $t_{\text{train}}=1\cdot t_{\text{forward+backward}}\approx2t_{\text{forward}}$。生成 $T$ token 耗时 $\approx T/2$ 倍训练（故生成慢）。

### 4.2 吞吐与并发

$$
\text{throughput}=\frac{B_{\text{dynamic}}\cdot T}{\text{latency}},\quad B_{\text{dynamic}}\le\frac{\text{mem}_{\text{KV}}}{\text{KV per request}}
$$

- $B_{\text{dynamic}}$：continuous batching 的动态批大小（显存限）；
- KV per request $=O(L\cdot d)$（$L$ 长度，$d$ 维）；
- 并发上限 = 显存/KV per request。

### 4.3 continuous batching 的增益

$$
\text{throughput}_{\text{cont}}\approx\text{throughput}_{\text{static}}\cdot\frac{1}{\text{padded waste}+\text{sync wait}}
$$

静态 batch 需等齐长度 + padding 浪费，continuous 消除两者，增益 2~4×。

### 4.4 speculative decoding 的增益

$$
\text{throughput}_{\text{spec}}\approx\text{throughput}_{\text{base}}\cdot\frac{1}{1-\text{accept rate}\cdot(k-1)/k}
$$

- 小模型预测 $k$ token，大模型一次前向验证；
- accept rate 高则减大模型前向次数，增益 2~3×。

### 4.5 量化的显存-吞吐

$$
\text{mem}_{\text{KV}}^{\text{INT8}}=\text{mem}_{\text{KV}}^{\text{FP16}}/2\Rightarrow\text{concurrency}\times2\Rightarrow\text{throughput}\uparrow
$$

显存减 → 并发增 → 吞吐升。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F, time

class LM(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)

V, D = 50, 32
model = LM(V, D).eval()

# ===== 基准:静态 batch 生成(慢) =====
def gen_static(model, prompts, max_len=16):
    """静态 batch:等齐长度,padding 浪费"""
    out = prompts.clone()
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(out)[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)   # greedy 简化
            out = torch.cat([out, nxt], 1)
    return out

# ===== 优化:continuous batching(简化模拟) =====
def gen_continuous(model, prompts, max_len=16):
    """continuous:请求完成即退出,不等齐(简化)"""
    out = prompts.clone(); active = list(range(len(prompts)))
    for _ in range(max_len):
        if not active: break
        with torch.no_grad():
            logits = model(out[active])[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            out[active] = torch.cat([out[active], nxt], 1)
            # 模拟:部分请求提前完成退出(continuous 核心)
            active = active[1:]  # 简化:每步退一个
    return out

B, L = 16, 4
prompts = torch.randint(0, V, (B, L))

t0 = time.time(); _ = gen_static(model, prompts, 16); t_static = time.time()-t0
t0 = time.time(); _ = gen_continuous(model, prompts, 16); t_cont = time.time()-t0
print(f"静态 batch: {t_static*1000:.1f}ms")
print(f"continuous(简化): {t_cont*1000:.1f}ms")
print(f"实际 vLLM continuous batching + PagedAttention 增益 2-4x")

# ===== 吞吐度量 =====
def throughput_measure(model, B, L, max_len, gen_fn):
    t0 = time.time(); _ = gen_fn(model, torch.randint(0,V,(B,L)), max_len); t = time.time()-t0
    tokens = B * max_len
    return tokens / t  # tokens/sec

tp = throughput_measure(model, B=32, L=4, max_len=16, gen_fn=gen_static)
print(f"吞吐: {tp:.0f} tokens/sec(实际 vLLM 70B ~1000-2000)")
print(f"并发受 KV cache 显存限,大模型需 TP 切分")
print("RLHF 中生成占 60-80% 耗时,吞吐优化是关键")
```

> [!tip] 实际工程
> - **vLLM**：continuous batching + PagedAttention，开源主流，rollout 默认；
> - **tensor parallel**：70B 用 TP=8 切 8 卡；
> - **speculative**：vLLM 支持，小模型预测；
> - **监控**：tokens/sec、GPU 利用率、KV cache 占用。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[trajectory generation]]（任务）、[[rollout worker]]（执行单元）、[[vLLM基础]]、[[continuous batching]]、[[KV cache management]]、[[speculative decoding]]、[[Tensor Parallel]]、[[量化]]。
- **下游（应用）**: [[batching strategy]]（批次组织）、RLHF 训练成本/周期、[[asynchronous training]]（生成-训练重叠提总吞吐）。
- **对比 / 易混**:
  - **sampling throughput（本篇）vs training throughput**：生成 tokens/sec vs 训练 FLOPS。前者是 RLHF 瓶颈。
  - **sampling throughput vs [[inference worker]] 吞吐**：rollout 侧（训练采数据）vs 部署侧（服务用户）。优化相似但场景不同（rollout 重吞吐，PD 重延迟+吞吐）。
  - **throughput vs latency**：rollout 重吞吐（非延迟），PD 重延迟。tradeoff 不同。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"生成不瓶颈"**：错。auto-regressive 慢，生成占 RLHF 60~80% 耗时。
> 2. **"batch 越大越好"**：错。显存限 KV cache，且延迟升。有最优 batch。
> 3. **"吞吐只看 tokens/sec"**：错。还需看 GPU 利用率、并发、延迟（避免低利用高延迟）。
> 4. **"continuous batching 等于静态"**：错。前者动态进出，后者等齐 padding，增益 2~4×。
> 5. **"KV cache 不碎片"**：错。不同长度混合碎片严重，PagedAttention 解决。
> 6. **"大模型单卡够"**：错。70B 需 TP 切多卡，否则显存/计算扛不住。
> 7. **"speculative 总有益"**：不一定。小模型与大模型不近则 accept rate 低，无增益。
> 8. **"rollout 重延迟"**：错。rollout 重吞吐（非用户实时），batch 拉满。
> 9. **"量化损吞吐"**：反。量化减显存→增并发→提吞吐（精度损可控）。

## 8. 延伸细节

### 8.1 vLLM PagedAttention 的原理

- KV cache 分页（如 OS 虚拟内存），按需分配；
- 不同长度请求共享显存，无碎片；
- 并发提升 2~4×（vs 连续分配）；
- 详见 [[KV cache management]]（待展开）。

### 8.2 speculative decoding 详解

- 小模型（draft）预测 $k$ token；
- 大模型（target）一次前向验证 $k$ token；
- accept 则跳过 $k$ 次大模型前向，reject 则回退；
- accept rate 高（小模型近大模型）则增益 2~3×；
- 详见 [[speculative decoding]]（待展开）。

### 8.3 tensor parallel 在 rollout

- 大模型（70B）切多卡（TP=8），每卡存部分 $\theta$；
- 每 token 前向需卡间通信（AllReduce）；
- 单卡吞吐降，但总吞吐升（支持大模型）；
- 与训练 TP 类似（[[Tensor Parallel]]）。

### 8.4 rollout 与训练的吞吐比

- **生成 bound**：rollout 慢（auto-regressive），训练快（一次前向+反向）；
- **优化比**：生成占 60~80%，优化生成提总吞吐最有效；
- **异步重叠**：worker 生成时 learner 训上一批，隐藏生成延迟（[[asynchronous training]]）；
- **batch 对齐**：rollout batch 与 learner batch 对齐，减数据 sync 浪费（[[batching strategy]]）。

### 8.5 成本视角

- **GPU 时**：rollout 占 60~80% GPU 时，是成本大头；
- **优化 ROI**：生成吞吐翻倍 → 总成本近半；
- **量化部署**：rollout 用 INT8/INT4 减显存+提吞吐，精度损对采数据可接受（vs PD 服务质量）；
- **spot 实例**：rollout 容忍中断（数据可重采），可用 spot GPU 省钱。

### 8.6 吞吐监控与调优

- **tokens/sec/replica**：主指标；
- **GPU 利用率**：< 80% 则 batch 不够或瓶颈；
- **KV cache 占用**：满则并发限；
- **queue 长度**：worker 队列长则生成跟不上训练；
- **生成/训练耗时比**：监控 rollout 占比，优化 ROI。
调优循环：监控 → 瓶颈定位 → 优化（batch/TP/speculative/量化）→ 再测。

---
相关: [[Rollout系统]]、[[trajectory generation]]、[[batching strategy]]、[[rollout worker]]、[[continuous batching]]、[[KV cache management]]、[[speculative decoding]]、[[Tensor Parallel]]、[[量化]]、[[vLLM基础]]、[[asynchronous training]]、[[policy training (PT)]]、[[训练稳定性工程]]
