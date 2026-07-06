# inference worker

> **所属章节**: [[训练架构]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（部署侧概念，需懂推理优化基础）

## 1. 一句话定义

**inference worker（推理 worker / 服务 worker）** 是部署侧专门跑模型**推理服务用户请求**的 worker：加载训练好的 $\theta$（从 PT 侧同步），响应实时用户请求，返回生成结果。与 [[rollout worker]] 的区别：inference worker **为部署服务用户**（延迟敏感、$\theta$ 固定），rollout worker **为训练采数据**（吞吐优先、$\theta$ 频繁更新）。用 **vLLM/TensorRT-LLM/Triton** 等推理引擎优化延迟（[[KV cache management]] / [[continuous batching]] / 量化 / [[speculative decoding]]）。它是 [[训推分离]] 的 **PD（policy deployment）侧**核心，对应 [[policy deployment (PD)]]。

> [!note] 别名
> inference worker = serving worker = deployment worker = PD worker。都指部署侧服务用户的推理进程。

## 2. 为什么需要它（动机与背景）

训练好的 LLM 需部署服务用户，但推理与训练的负载特性完全不同：
1. **延迟敏感**：用户等不起秒级延迟，单请求需快返回（TTFT < 1s、TPOT < 50ms）；
2. **吞吐次要**：服务侧单请求延迟优先，批处理为延迟让路（虽 continuous batching 兼顾两者）；
3. **$\theta$ 固定**：部署期间不更新（除非版本升级），与 rollout worker 的频繁 weight sync 不同；
4. **推理优化专门**：量化（INT8/INT4）、[[speculative decoding]]（投机解码）、PagedAttention 等针对推理的优化，训练框架不优；
5. **多副本负载均衡**：高并发需多 inference worker 副本 + 负载均衡，与训练的分布式不同；
6. **与训练分离**：PT（训练）与 PD（部署）解耦，PD 用专用 worker + 推理引擎，PT 不干扰 PD。

故 inference worker 是 LLM 服务的工程必需——用专用进程 + 推理引擎，把延迟压到最低，多副本扛并发。

## 3. 核心概念详解

### 3.1 inference worker 的标准流程

```
inference worker 服务流程:
1. 启动:加载 θ(从 PT 同步或 checkpoint)         ← [[weight sync mechanism]] 或 checkpoint 加载
2. 初始化推理引擎:vLLM/TensorRT-LLM             ← KV cache 预分配
3. 接收用户请求(prompt)                          ← HTTP/gRPC
4. continuous batching:动态拼批                   ← [[continuous batching]]
5. 自回归生成 response(用 θ)                     ← KV cache 加速
6. 返回 response 给用户                          ← 流式或一次性
7. 回到 3(θ 不变,除非版本升级)
```

### 3.2 inference worker 的优化技术

| 技术 | 机制 | 效果 | 详见 |
|---|---|---|---|
| **[[KV cache management]]** | 缓存历史 K/V，免重算 | 减 $O(T^2)$ 到 $O(T)$ 每步 | [[KV cache management]] |
| **PagedAttention** | KV cache 分页，减碎片 | 提并发 | vLLM |
| **[[continuous batching]]** | 动态拼批，新请求随时插入 | 提 GPU 利用率 | [[continuous batching]] |
| **量化（INT8/INT4）** | 低精度推理 | 减显存/加速 | [[量化]]（待展开） |
| **[[speculative decoding]]** | 小模型草稿 + 大模型校验 | 减大模型前向 | [[speculative decoding]] |
| **prefix caching** | 共享前缀的 KV cache 复用 | 同 system prompt 的请求加速 | vLLM/SGLang |

### 3.3 inference worker vs rollout worker（再强化）

| 维度 | inference worker | [[rollout worker]] |
|---|---|---|
| 目的 | 部署服务用户 | 训练采数据 |
| 输入 | 用户实时请求 | prompt 集（采样） |
| 输出 | response 返用户 | trajectory 进 buffer |
| 延迟 vs 吞吐 | 延迟优先 | 吞吐优先 |
| $\theta$ 更新 | 固定（版本升级才换） | 频繁（weight sync） |
| 采样策略 | 部署配置（温度/nucleus/beam） | 按 $\pi_\theta$ 采样保探索 |
| 框架 | vLLM/TensorRT-LLM/Triton（延迟优化） | vLLM/SGLang（吞吐优化） |
| 副本 | 多副本 + 负载均衡 | 多 worker 并行采样 |
| 所属 | PD（[[policy deployment (PD)]]） | PT 内部（训练架构） |

### 3.4 延迟指标（服务侧核心）

| 指标 | 定义 | 目标 |
|---|---|---|
| **TTFT（Time To First Token）** | 请求到首 token 延迟 | < 1s |
| **TPOT（Time Per Output Token）** | 每 token 生成延迟 | < 50ms |
| **端到端延迟** | 请求到完整 response | 视序列长 |
| **吞吐** | tokens/sec/worker | 越高越好 |
| **并发** | 同时服务请求数 | KV cache 显存限 |

### 3.5 多副本部署与负载均衡

- **多副本**：N 个 inference worker 各加载 $\theta$，扛 N× 并发；
- **负载均衡**：请求路由到空闲 worker（round-robin / 最少连接）；
- **显存切分**：大模型单卡装不下，每副本用 TP 切分（[[Tensor Parallel]]）；
- **弹性扩缩**：流量高峰加副本，低谷减（Kubernetes autoscaling）。

## 4. 数学原理 / 公式

### 4.1 自回归推理的时间

生成 $T$ token，用 KV cache 后每步注意 $O(1)$（只算新 token），总：
$$
T_{\text{infer}}=T\cdot O(1)=O(T)
$$

（首 token 需处理 prompt $O(L)$，后续每 token $O(1)$，$L$ prompt 长）。对比无 KV cache 的 $O(T^2)$，KV cache 是推理加速的关键。

### 4.2 TTFT 与 TPOT

$$
\text{TTFT}=O(L)\text{（prompt 前向）},\quad \text{TPOT}=O(1)\text{（每 token 生成）}
$$

端到端 $\approx \text{TTFT}+T\cdot\text{TPOT}$。

### 4.3 量化的显存-精度 tradeoff

- FP16：$\theta$ 显存 $P$；
- INT8：$P/2$，精度损小；
- INT4：$P/4$，精度损中（AWQ/GPTQ）。
显存减 → 并发增（KV cache 空间大）→ 吞吐升，但精度降。详见 [[量化]]（待展开）。

### 4.4 continuous batching 的吞吐

$$
\text{throughput}=\frac{B_{\text{dynamic}}\cdot T}{T_{\text{infer}}}
$$

$B_{\text{dynamic}}$ 随请求到达动态调（显存允许内），比静态批 $B_{\text{static}}$ 利用率高。

## 5. 代码示例

```python
import torch, torch.nn as nn

class Policy(nn.Module):
    def __init__(self, v, d):
        super().__init__(); self.emb=nn.Embedding(v,d); self.lstm=nn.LSTM(d,d,batch_first=True); self.head=nn.Linear(d,v)
    def forward(self, x):
        h,_=self.lstm(self.emb(x)); return self.head(h)

V, D = 50, 32
model = Policy(V, D)
model.load_state_dict(torch.load('policy.pt'))  # 加载训练好的 θ(实际从 PT 同步)
model.eval()

# ===== inference worker 服务 =====
@torch.no_grad()
def serve(prompt, max_len=10, T=0.7, p=0.9):
    """inference worker:接收用户请求,生成 response 返回"""
    out = prompt.clone()
    for _ in range(max_len):
        logits = model(out)[:, -1, :] / T           # temperature
        probs = torch.softmax(logits, -1)
        # nucleus sampling(部署常用,保质量)
        sorted_p, sorted_i = probs.sort(descending=True)
        cum = sorted_p.cumsum(-1)
        mask = cum - sorted_p < p
        sorted_p = sorted_p * mask / (sorted_p * mask).sum(-1, keepdim=True)
        nxt = sorted_i.gather(-1, torch.multinomial(sorted_p, 1))
        out = torch.cat([out, nxt], 1)
    return out[:, prompt.shape[1]:]                  # 返回 response 给用户

# 模拟用户请求
prompt = torch.randint(0, V, (1, 4))
response = serve(prompt, max_len=8, T=0.7, p=0.9)
print(f"用户 prompt: {prompt[0].tolist()}")
print(f"生成 response: {response[0].tolist()}")
print("实际:vLLM server 用 continuous batching + PagedAttention 扛高并发")
```

> [!tip] 实际工程（vLLM server / Triton）
> - **vLLM server**：`python -m vllm.entrypoints.api_server`，HTTP API，continuous batching 扛并发；
> - **Triton Inference Server**：NVIDIA，支持多框架（vLLM/TensorRT-LLM/PyTorch），生产级部署；
> - **TensorRT-LLM**：NVIDIA 推理引擎，INT8/FP8 量化 + 算子融合，延迟最优。
> 生产 PD 侧常用 Triton + TensorRT-LLM 或 vLLM server。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[训推分离]]（PD 侧）、[[policy deployment (PD)]]、[[weight sync mechanism]]（从 PT 接收 θ）、训练好的 LLM。
- **下游（应用）**: LLM 服务（ChatGPT/Claude/API）、[[continuous batching]]、[[KV cache management]]、[[speculative decoding]]、[[量化]]、多副本负载均衡。
- **对比 / 易混**:
  - **inference worker vs [[rollout worker]]**：见 §3.3 表。部署服务 vs 训练采样。
  - **inference worker vs learner**：前者部署推理（只前向），后者训练（forward+backward）。无直接关系，但 learner 训出的 θ 给 inference worker 部署。
  - **PD vs PT**：PD = inference worker 部署侧，PT = learner + rollout worker 训练侧。[[训推分离]] 解耦两者。
  - **推理引擎 vs 训推框架**：推理引擎（vLLM/TensorRT-LLM）优化延迟/吞吐，训推框架（FSDP/DeepSpeed）优化显存/通信。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"inference worker = rollout worker"**：错。前者部署服务用户，后者训练采数据。目的/θ 更新/延迟需求都不同。
> 2. **"推理只加载模型"**：错。需 KV cache/continuous batching/量化等优化，否则延迟高/并发低。
> 3. **"inference worker 也更新 θ"**：错。部署期间 θ 固定，除非版本升级（weight sync 从 PT 拿新版本）。
> 4. **"延迟和吞吐兼得"**：tradeoff。continuous batching 兼顾，但极端延迟需求需小批/单请求。
> 5. **"大模型单卡部署"**：常装不下。需 TP 切分（[[Tensor Parallel]]）或量化（INT4）。
> 6. **"推理用训练框架"**：低效。应用 vLLM/TensorRT-LLM（推理优化），非 FSDP（训练优化）。
> 7. **"多副本无成本"**：N 副本显存 N×，需 GPU 资源 + 负载均衡。
> 8. **"量化无损"**：INT8/INT4 有精度损，需评估（perplexity/人评），AWQ/GPTQ 减损但非零。

## 8. 延伸细节

### 8.1 推理引擎对比

| 引擎 | 强项 | 适用 |
|---|---|---|
| **vLLM** | PagedAttention + continuous batching，吞吐优，开源主流 | 通用部署、RLHF rollout |
| **TensorRT-LLM** | INT8/FP8 量化 + 算子融合，延迟最优 | NVIDIA 生产部署 |
| **SGLang** | RadixAttention（前缀复用），多候选/结构化生成优 | 复杂采样、GRPO |
| **Triton** | 多框架支持，生产级服务管理 | 企业部署 |

### 8.2 量化技术

- **INT8**：W8A8，显存半，精度损小，TensorRT-LLM 支持；
- **INT4**（AWQ/GPTQ）：W4A16，显存 1/4，精度损中，大模型部署主流；
- **FP8**：H100 支持，精度损极小，新一代主流。
量化是 inference worker 部署大模型的核心（显存-延迟 tradeoff）。

### 8.3 speculative decoding（投机解码）

小模型（draft）先生成草稿，大模型（target）并行校验，减大模型前向次数：
- 适合 token 级预测高一致性的场景（代码/数学）；
- 减 TPOT，但增显存（draft model）+ 复杂度；
- vLLM/TensorRT-LLM 支持。
详见 [[speculative decoding]]（待展开）。

### 8.4 prefix caching

多个请求共享同一 system prompt（如"你是个助手..."），其 KV cache 可复用：
- vLLM 的 `enable_prefix_caching`；
- SGLang 的 RadixAttention 天然支持；
- 减 TTFT（共享前缀免重算）。
是服务侧降延迟的有效手段。

### 8.5 PD 的版本更新

inference worker 的 θ 固定，但 PT 侧持续训练新版本。版本更新流程：
1. PT 训出新 $\theta_{\text{new}}$；
2. checkpoint 保存；
3. PD 侧 inference worker 加载新 checkpoint（或 weight sync 流式更新）；
4. 热切换（blue-green deployment，零停机）。
这是 [[训推分离]] 的版本管理，详见 [[weight sync mechanism]]。

### 8.6 inference worker 的监控

服务侧必盯：
- **TTFT/TPOT**：延迟指标；
- **吞吐**：tokens/sec；
- **并发**：同时请求数；
- **GPU 利用率/显存**：资源使用；
- **错误率/超时**：服务健康。
这些是 LLM serving 的运维仪表盘。

---
相关: [[训练架构]]、[[rollout worker]]、[[learner worker split]]、[[训推分离]]、[[policy deployment (PD)]]、[[policy training (PT)]]、[[weight sync mechanism]]、[[continuous batching]]、[[KV cache management]]、[[speculative decoding]]、[[量化]]、[[Tensor Parallel]]、[[asynchronous training]]
