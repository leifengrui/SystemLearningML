# policy deployment (PD)

> **所属章节**: [[训推分离]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（总览性，与 PT 对称，整合推理服务）

## 1. 一句话定义

**policy deployment（PD，策略部署侧）** 是 [[训推分离]] 的**部署侧**：加载 [[policy training (PT)]] 训好的 $\theta$，用 [[inference worker]] 服务用户请求，返回生成结果。PD 专注**部署优化**（vLLM/TensorRT-LLM 的延迟优化、[[量化]]、多副本负载均衡），通过 [[weight sync mechanism]] 从 PT 接收新 $\theta$ 版本，**热切换**升级。与 PT 分离是因为训练与部署的负载/框架/资源/优化点完全不同——PD 重延迟并发、要求高可用零停机。PD 是用户接触 LLM 的"前台"，是 [[inference worker]] 的整体编排（多副本 + 负载均衡 + 版本管理 + SLO 保障）。

> [!note] PD vs [[inference worker]]
> - **PD（本篇）**：部署侧**整体**——多 inference worker 副本 + 负载均衡 + 版本管理 + SLO。
> - **[[inference worker]]**：PD 内部的**单个推理进程**——加载 $\theta$、服务请求。
> PD 是编排层，inference worker 是执行单元。

## 2. 为什么需要它（动机与背景）

LLM 训练完后需持续服务用户，但部署与训练的负载截然不同：
1. **延迟敏感**：用户等不起秒级延迟，PD 需压低 TTFT/TPOT；
2. **高并发**：多用户同时请求，PD 需多副本 + 负载均衡扛并发；
3. **高可用**：PD 需 7×24 零停机（PT 容忍重启，PD 不行）；
4. **推理优化专门**：[[量化]]（INT8/INT4）、[[speculative decoding]]、prefix caching 等推理优化，训练框架不优；
5. **版本管理**：PT 持续训新版本，PD 需热切换不停机；
6. **资源异构**：PD 用服务 GPU 副本（低延迟网络），PT 用训练大集群（IB 网络）。

故 PD 独立于 PT：用推理引擎 + 多副本 + 负载均衡，把延迟压低、并发扛高、服务稳定。PD 是 LLM 走向用户的"最后一公里"。

## 3. 核心概念详解

### 3.1 PD 的部署流程

```
PD 侧流程:
1. 从 PT 接收 θ(checkpoint 或 weight sync)        ← [[weight sync mechanism]]
2. 验证(eval/安全/人评)                            ← 上线前 gate
3. inference worker 加载 θ + 推理引擎初始化         ← vLLM/TensorRT-LLM
4. 服务用户请求(continuous batching + KV cache)    ← [[inference worker]]
5. 监控 SLO(TTFT/TPOT/吞吐/错误率)
6. 版本升级:PT 推新 θ → 热切换(blue-green)         ← 零停机
7. 异常回滚到上一版本
```

### 3.2 PD 的架构（批1的具体化）

PD 用多个 [[inference worker]] + 编排层：
- **inference worker 副本**：N 个，各加载 $\theta$，服务请求（vLLM/TensorRT-LLM）；
- **负载均衡器**：请求路由到空闲 worker（round-robin / 最少连接 / 最短队列）；
- **API gateway**：接收用户请求，鉴权/限流/路由；
- **版本管理器**：管理 $\theta$ 版本，热切换/回滚；
- **监控/告警**：SLO 监控，异常告警。

### 3.3 PD 的资源与框架

| 维度 | PD 侧 |
|---|---|
| **硬件** | 服务 GPU 副本（多卡，A100/H100，低延迟网络） |
| **框架** | vLLM server / TensorRT-LLM / Triton（推理优化） |
| **并行** | TP（大模型单副本切分）[[Tensor Parallel]] |
| **显存** | $\theta$ + KV cache（推理主要开销） |
| **优化** | 量化（INT8/INT4）、speculative decoding、prefix caching |
| **编排** | Kubernetes（副本管理/ autoscaling）+ 负载均衡 |

### 3.4 PD 的版本管理

| 机制 | 说明 |
|---|---|
| **checkpoint 加载** | PD 从存储拉 PT 的 checkpoint，加载 $\theta$ |
| **weight sync** | PT 流式推送 $\theta$，PD 热切换（不停机） |
| **blue-green** | 新版本起来验证后切流量，旧版本待命回滚 |
| **A/B 测试** | 部分流量切新版本，对比效果 |
| **灰度发布** | 渐进切流量（1% → 10% → 100%） |
| **回滚** | 异常时秒级回退上一版本 |

### 3.5 PD 的 SLO（服务等级目标）

| 指标 | 典型 SLO |
|---|---|
| **TTFT** | < 1s（首 token 延迟） |
| **TPOT** | < 50ms（每 token 延迟） |
| **可用性** | 99.9%+（零停机） |
| **吞吐** | tokens/sec 满足峰值 |
| **错误率** | < 0.1% |

PD 的所有优化围绕 SLO：延迟优化（量化/speculative）、并发优化（多副本/continuous batching）、可用性（多副本冗余/热切换）。

### 3.6 PD 的监控

- **延迟**：TTFT/TPOT 分位数（p50/p95/p99）；
- **吞吐**：tokens/sec/replica × N；
- **并发**：同时请求数、队列长度；
- **资源**：GPU 利用率/显存（KV cache 占用）；
- **错误/超时**：服务健康；
- **版本**：当前服务的 $\theta$ 版本。
这些是 PD 运维的仪表盘（[[inference worker]] §8.6 的延伸）。

## 4. 数学原理 / 公式

### 4.1 PD 的延迟

$$
\text{latency}=\text{TTFT}+T\cdot\text{TPOT}
$$

- TTFT $=O(L)$（prompt 前向，$L$ prompt 长）；
- TPOT $=O(1)$（每 token，用 KV cache）；
- 量化/speculative 减 TPOT，prefix caching 减 TTFT。

### 4.2 PD 的吞吐与并发

$$
\text{throughput}=\frac{B_{\text{dynamic}}\cdot T}{\text{latency}},\quad \text{concurrency}\le\frac{\text{mem}_{\text{KV}}}{\text{KV per request}}
$$

- $B_{\text{dynamic}}$：continuous batching 的动态批大小；
- 并发受 KV cache 显存限（每请求的 KV cache 占显存）；
- 多副本：N 副本吞吐 N×，并发 N×。

### 4.3 量化的显存-精度 tradeoff

| 精度 | $\theta$ 显存 | KV cache 显存 | 精度损 |
|---|---|---|---|
| FP16 | $P$ | $K$ | 0 |
| INT8 | $P/2$ | $K/2$ | 小 |
| INT4 | $P/4$ | $K$（常保 FP16） | 中 |

显存减 → 并发增 → 吞吐升。详见 [[量化]]（待展开）。

### 4.4 PD 的版本滞后

PD 服务的 $\theta_{\text{PD}}$ 可能滞后 PT 最新 $\theta_{\text{PT}}$：
$$
\text{lag}=\theta_{\text{PT}}-\theta_{\text{PD}}\text{（版本差）}
$$

PT 持续训练，PD 定期升级追赶。lag 大 → PD 服务旧版本（[[stale policy problem]] 的 PT-PD 变体）。

## 5. 代码示例

```python
import torch, torch.nn as nn, copy, time

class LM(nn.Module):
    def __init__(s, v, d):
        super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s, x):
        h,_=s.lstm(s.emb(x)); return s.head(h)

V, D = 50, 32

# ===== PD 侧:部署服务 =====
class PDEployment:
    def __init__(self):
        self.model = LM(V, D)
        self.version = None
        self.replicas = []   # 多 inference worker 副本
    def load_checkpoint(self, ckpt_path):
        """从 PT 接收 θ(checkpoint)"""
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt["theta"])
        self.version = ckpt["version"]
        print(f"PD 加载 θ 版本 {self.version}")
    def hot_swap(self, new_ckpt_path):
        """热切换(blue-green):新版本起来验证后切"""
        old_version = self.version
        self.load_checkpoint(new_ckpt_path)
        print(f"热切换: {old_version} -> {self.version}(零停机)")
    @torch.no_grad()
    def serve(self, prompt, max_len=8, T=0.7, p=0.9):
        """inference worker 服务请求"""
        out = prompt.clone()
        for _ in range(max_len):
            logits = self.model(out)[:, -1, :] / T
            probs = torch.softmax(logits, -1)
            sorted_p, sorted_i = probs.sort(descending=True)
            cum = sorted_p.cumsum(-1); mask = cum - sorted_p < p
            sorted_p = sorted_p * mask / (sorted_p * mask).sum(-1, keepdim=True)
            nxt = sorted_i.gather(-1, torch.multinomial(sorted_p, 1))
            out = torch.cat([out, nxt], 1)
        return out[:, prompt.shape[1]:]

# 模拟 PT → PD
ckpt = {"theta": LM(V, D).state_dict(), "version": "v1.0"}
torch.save(ckpt, "pt_checkpoint.pt")

pd = PDEployment()
pd.load_checkpoint("pt_checkpoint.pt")
response = pd.serve(torch.randint(0, V, (1, 4)))
print(f"PD 服务: {response[0].tolist()}")

# PT 训新版本 → PD 热切换
ckpt2 = {"theta": LM(V, D).state_dict(), "version": "v1.1"}
torch.save(ckpt2, "pt_checkpoint_v2.pt")
pd.hot_swap("pt_checkpoint_v2.pt")
print("PD 持续服务,零停机升级")
```

> [!tip] 实际工程
> PD 侧用 vLLM server / Triton + Kubernetes：
> - Kubernetes 管 N 副本（Deployment），autoscaling 扛峰值；
> - 负载均衡（Nginx/Envoy）路由请求；
> - 版本管理：Argo Rollouts / Flagger 做 blue-green/canary；
> - 监控：Prometheus + Grafana（TTFT/TPOT/吞吐）。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[policy training (PT)]]（PD 的 $\theta$ 来源）、[[weight sync mechanism]]（PT→PD 接口）、[[inference worker]]（PD 的执行单元）、[[continuous batching]]、[[KV cache management]]、[[量化]]、[[speculative decoding]]。
- **下游（应用）**: LLM 服务（ChatGPT/Claude/API）、A/B 测试、灰度发布、SRE/LLMOps。
- **对比 / 易混**:
  - **PD vs [[policy training (PT)]]**：部署侧 vs 训练侧。对称对照，负载/框架/资源全不同。
  - **PD vs [[inference worker]]**：PD 是编排整体（多副本+负载均衡+版本管理），inference worker 是单执行单元。
  - **PD vs [[rollout worker]]**：PD 服务用户（部署），rollout worker 采数据（训练）。目的不同。
  - **PD 的 $\theta$ vs PT 的 $\theta$**：PD 是部署版（可能滞后），PT 是最新训练版。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"PD 也训练"**：错。PD 只部署推理，不训练（训练是 PT）。
> 2. **"PD 只加载模型"**：错。需推理优化（量化/continuous batching/speculative）+ 多副本 + 负载均衡 + 版本管理。
> 3. **"PD 的 $\theta$ 最新"**：不一定。PD 可能滞后 PT（版本升级延迟），定期 sync 追赶。
> 4. **"PD 用训练框架"**：低效。应用 vLLM/TensorRT-LLM（推理优化），非 FSDP（训练）。
> 5. **"PD 单副本够"**：并发扛不住。需多副本 + 负载均衡。
> 6. **"PD 不需高可用"**：错。PD 是用户前台，需 7×24 零停机，多副本冗余 + 热切换。
> 7. **"版本升级停机"**：应零停机。blue-green/canary 热切换，旧版本待命回滚。
> 8. **"PD 的 SLO 只看吞吐"**：错。延迟（TTFT/TPOT）+ 可用性 + 错误率都关键。

## 8. 延伸细节

### 8.1 serving 框架对比

| 框架 | 强项 | 适用 |
|---|---|---|
| **vLLM server** | PagedAttention + continuous batching，开源主流 | 通用 PD |
| **TensorRT-LLM** | INT8/FP8 量化 + 算子融合，延迟最优 | NVIDIA 生产 PD |
| **Triton Inference Server** | 多框架支持 + 模型仓库 + 版本管理 | 企业 PD |
| **SGLang** | RadixAttention，结构化/多候选生成优 | 复杂采样 PD |

### 8.2 PD 的量化部署

- **INT8（W8A8）**：显存半，精度损小，TensorRT-LLM 支持；
- **INT4（AWQ/GPTQ）**：显存 1/4，大模型部署主流；
- **FP8**：H100，精度损极小，新一代主流；
- **KV cache 量化**：减 KV 显存，提并发。
量化是 PD 部署大模型的核心，详见 [[量化]]（待展开）。

### 8.3 PD 的弹性扩缩

- **峰值扛压**：流量高峰 autoscaling 加副本，低谷减（省钱）；
- **预测性扩容**：基于历史流量预测提前扩；
- **GPU 预热池**：常备预热副本，秒级扩容（避免冷启动）。
这是 Kubernetes + LLM serving 的运维核心。

### 8.4 A/B 测试与灰度

- **A/B**：部分流量切新 $\theta$，对比 SLO/人评；
- **canary**：1% → 10% → 50% → 100% 渐进切，异常回滚；
- **shadow**：新版本影子跑（不返用户），只对比指标；
- 这些是 PD 版本升级的安全网。

### 8.5 PD 的 stale policy（PT-PD 变体）

PD 服务 $\theta_{\text{PD}}$，PT 已训到 $\theta_{\text{PT}}$。若 lag 大：
- 用户接触的是旧版本（可能能力/对齐落后）；
- 线上反馈数据基于旧 $\theta$，回传 PT 训练时有分布偏移（[[exposure bias]] 变体）；
- 需定期 sync 追赶 + 线上数据标注时记 $\theta$ 版本。
详见 [[stale policy problem]] 的 PT-PD 变体。

### 8.6 PD 的成本优化

- **量化**：减显存 → 减 GPU 数；
- **speculative decoding**：减大模型前向 → 减算力；
- **prefix caching**：减 TTFT → 提并发；
- **batching**：continuous batching 提 GPU 利用率；
- **spot 实例**：低谷用 spot GPU 省钱（需冗余防中断）。
PD 的成本是 LLM 服务运营的关键。

---
相关: [[训推分离]]、[[policy training (PT)]]、[[weight sync mechanism]]、[[stale policy problem]]、[[inference worker]]、[[rollout worker]]、[[continuous batching]]、[[KV cache management]]、[[量化]]、[[speculative decoding]]、[[Tensor Parallel]]、[[exposure bias]]
