# Transformer与新型架构适配

> **所属章节**: [[新型架构适配]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: 训推框架对 SSM/Mamba 的支持 / SSM 系统适配 / Mamba serving / 新架构引擎改造 / vLLM Mamba / SGLang Mamba
> **难度**: 高（需懂 [[Mamba]] + [[Hybrid架构]] + [[KV cache]] + [[PagedAttention]] + [[continuous batching的调度实现]] + [[新模型接入]] + [[CUDA stream与event]]）


## 1. 一句话定义

**Transformer 与新型架构适配** 是把为 Transformer（[[自注意力]] + [[autoregressive decoding]] + [[KV cache]]）设计的训推框架（vLLM/SGLang/Megatron/PyTorch）**改造以支持新型架构**（SSM/[[Mamba]]/[[Hybrid架构]]）的工程命题——三大改造点：① **推理**：SSM **无 KV cache 而是固定 state**，引擎的 [[PagedAttention]]/[[continuous batching的调度实现]] 假设基于"每 token 增量 KV"需改（SSM 的 state 管理：固定大小、可预分配、不增长；可变长序列的 scan 调度：变长 batch 的 scan 前缀和）；② **训练**：SSM 的 **selective scan 算子**（[[CUDA stream与event]] 上的融合 kernel）、**序列并行 CP 切 SSM scan**（scan 是顺序依赖，CP 切分需 chunk 间传递 state）、**并行化**（scan 的前缀和并行算法、变长 scan）；③ **框架改动点**：attention 层替换为 SSM 层、**去掉 [[位置编码]]**（SSM 递归有序不需 RoPE）、**cache 接口从 KV 改 state**、CP/TP 的切分点改。**vLLM/SGLang 对 Mamba 的支持现状**：vLLM 已支持 Mamba/Mamba-2/Jamba（via `mamba`/`mamba2` 模型类、state cache 替代 KV cache）、SGLang 有 Mamba 支持；Megatron 训练侧有 selective scan kernel 与 SSM CP。**与 [[新模型接入]]、[[CUDA stream与event]] 的关系**：新模型接入是改 model runner/worker 接口、SSM kernel 走 CUDA stream 调度；本质是 AR 假设（增量 KV、causal、1 token decode）→ SSM 假设（固定 state、递归、全序列 scan）的系统重构。是 [[新型架构适配]] 的系统层落地。

> [!note] 三句话定位
> - **是什么**：vLLM/SGLang/Megatron 为 Transformer 设计，新架构（SSM/Mamba/Hybrid）需改——state 替 KV cache、selective scan 算子、去位置编码、CP 切 scan。
> - **为什么**：引擎的 PagedAttention/continuous batching 假设每 token 增量 KV + causal + 1 token decode；SSM 固定 state + 全序列 scan，假设全失效。
> - **与 [[新模型接入]] 关系**：后者是改 model runner/worker 通用接口；前者是 SSM-specific 的 cache/算子/并行改造，是 [[新模型接入]] 的特化。


## 2. 为什么需要它（动机与背景）

### 2.1 训推框架全为 Transformer 设计

vLLM/SGLang/Megatron/PyTorch 的核心组件都为 Transformer 优化：
- **[[KV cache]]**：每 token 增量 KV、PagedAttention 分页存储、只增不减。
- **[[continuous batching的调度实现]]**：按 decode step 调度、各请求 1 token。
- **[[PagedAttention]]**：非 contiguous 变长 KV 分页。
- **CP（context parallel）**：ring attention 切序列。
- **[[位置编码]]**：[[RoPE]] 在 attention 前。
- **attention kernel**：[[FlashAttention]] causal。

### 2.2 SSM 打破每个假设

[[Mamba]] SSM 与 Transformer 假设全不同：
- **无 KV cache，固定 state**：state $h$ 固定大小 $N$，不随序列长增长，不增量、可预分配。KV cache 的增量/分页/增长假设全失效。
- **selective scan 算子**：递归 $h_t=\bar A_t h_{t-1}+\bar B_t x_t$，参数输入相关，需专用 fused kernel（非 attention）。
- **全序列 forward**：每步 scan 全序列（非 1 token decode）。
- **不需位置编码**：递归有序。
- **CP 切 scan**：scan 顺序依赖，CP 切分需 chunk 间传 state（非 ring attention）。

### 2.3 Hybrid 需同时管 state + KV

[[Hybrid架构]]（Jamba）部分层 SSM（state）、部分层 attention（KV），引擎需同时管两类缓存 + 两类算子，最复杂。

### 2.4 框架必须改造才能支持

不改 → SSM 模型无法在 vLLM/SGLang 跑。改 → 接入新架构、复用 serving 栈（continuous batching、调度）。是 [[新型架构适配]] 的系统层命题。


## 3. 核心概念详解

### 3.1 推理侧：state cache 替 KV cache

```
Transformer KV cache:
  每 token 增量计算 KV_i, 存页表, 只增不减
  decode: 新 token j -> 只算 KV_j 增量
  PagedAttention: 非 contiguous 分页存

SSM state cache:
  固定大小 state h (N 维), 可预分配, 不增长
  decode: 新 token x_t -> h_t = bar_A_t h_{t-1} + bar_B_t x_t (递归更新)
  无 "增量 KV", 是 "状态递归"
  管理: 每请求一个 state (固定), 进出 batch 即分配/释放
```

vLLM 改造：把 `KVCache` 抽象替换/扩展为 `StateCache`，每请求预分配固定 state，连续 batching 时按请求管理 state 而非 KV block。state 小（N 维，如 16×d），可常驻、无需分页。

### 3.2 推理侧：变长序列的 scan 调度

continuous batching 下各请求序列长不同。SSM scan 需对每请求按其长度递归。引擎需支持**变长 scan**：一个 batch 内不同请求不同长度，scan kernel 需处理 padding/segment。

```
batch: [req_A (len 100), req_B (len 50), req_C (len 200)]
  SSM scan: 各请求按自己长度递归 state
  kernel: 变长 scan (segment-based, 非 padding)
```

### 3.3 训练侧：selective scan CUDA kernel

SSM 的 selective scan（参数输入相关）破坏卷积并行，需**专用 fused CUDA kernel**：
- 输入 $x_t$ → 投影出 $B_t,C_t,\Delta_t$（输入相关）。
- 离散化 $\bar A_t=\exp(\Delta_t A)$。
- 并行扫描（前缀和式）算 $h_{1:L}$，融合避免 materialization $h$ 的 IO。
- 走 [[CUDA stream与event]] 调度，与 attention kernel 共存。

mamba-ssm/mamba2-ssm 库提供 kernel。Megatron 集成。

### 3.4 训练侧：CP 切 SSM scan

Transformer CP 用 ring attention（序列切段、ring 传 KV）。SSM CP 需切 scan：
- 把序列分 chunk，每 GPU 算一个 chunk 的 scan。
- chunk 间**顺序依赖**（$h$ 从前传后），需 chunk 间传 state（不像 attention 可 ring 并行）。
- 实现：chunk 顺序流水，前 GPU 算完传 state 给后 GPU。是 SSM CP 的核心难点。

### 3.5 训练侧：scan 的并行化

scan 递归 $h_t=\bar A_t h_{t-1}+\bar B_t x_t$ 是顺序依赖，但可用**前缀和并行算法**（associative scan，结合律）并行：把递归视为 $(h, \text{acc})$ 的结合运算，用并行 reduce 树算。mamba-ssm 用此在 GPU 上并行 scan。

### 3.6 框架改动点清单

| 改动点 | Transformer | SSM/Mamba/Hybrid |
|---|---|---|
| **核心层** | attention + MLP | SSM + 门控 (+attention for Hybrid) |
| **位置编码** | [[RoPE]] | 去掉（SSM），Hybrid 仅 attention 层 |
| **cache 接口** | KV cache（增量分页） | state cache（固定预分配） + KV（Hybrid） |
| **CP 切分** | ring attention | chunk 间传 state |
| **算子 kernel** | [[FlashAttention]] causal | selective scan + full attention（Hybrid） |
| **decode 模式** | 1 token/step（[[autoregressive decoding]]） | 全序列 scan/递归 |
| **调度** | 按 decode step | 按请求长度 scan |

### 3.7 vLLM/SGLang 对 Mamba 的支持现状

- **vLLM**：已支持 Mamba/Mamba-2/Jamba。模型类 `mamba`/`mamba2`/`jamba`。把 KV cache manager 替换为 state cache（`A, B, C, dt` 投影 + state 递归）。continuous batching 复用（按请求管理 state）。Hybrid（Jamba）同时管 state + KV。
- **SGLang**：有 Mamba 支持，类似机制。
- **Megatron**：训练侧，selective scan kernel + SSM CP（chunk 间传 state）。
- **PyTorch / mamba-ssm**：提供 `mamba_ssm`/`mamba2_ssm` 算子，供框架调用。

### 3.8 与 AR serving 组件对照

| AR serving 组件 | SSM 下处理 |
|---|---|
| [[KV cache]] | 替为 state cache（固定预分配） |
| [[PagedAttention]] | 不需（state 小可常驻），Hybrid 仍用 |
| [[continuous batching的调度实现]] | 复用（按请求管理 state） |
| [[autoregressive decoding]] | 替为 SSM 递归（每步更新 state） |
| [[chunked prefill]] | 长 prompt 一次性 scan（不像 prefill+decode） |


## 4. 数学原理 / 公式

### 4.1 SSM 递归 vs KV 增量

- **Transformer decode**：新 token $x_t$，算 $K_t,V_t$ 增量存 cache，$y_t=\text{softmax}(q_t K_{\leq t})V_{\leq t}$。
- **SSM decode**：新 token $x_t$，$B_t=B(x_t),C_t=C(x_t),\Delta_t=\Delta(x_t)$，$\bar A_t=\exp(\Delta_t A)$，$\bar B_t=\Delta_t B_t$，$h_t=\bar A_t h_{t-1}+\bar B_t x_t$，$y_t=C_t h_t$。**只更新固定 state $h$**，无增量存储。

### 4.2 state cache 大小

SSM state $h\in\mathbb R^{N}$（$N$=state size，如 $16d$）。每请求一个 state，大小固定 $\propto N$（与序列长 $L$ 无关）。vs Transformer KV cache $\propto L$。

$$\text{SSM state mem} = O(N),\quad \text{Transformer KV mem} = O(L)$$

长序列 $L\gg N$ 时 SSM 省。

### 4.3 scan 的并行：associative scan

递归 $h_t=a_t h_{t-1}+b_t$（$a_t=\bar A_t$ 标量、$b_t=\bar B_t x_t$）可写为线性变换的复合：

$$\begin{pmatrix}h_t\\1\end{pmatrix} = \begin{pmatrix}a_t & b_t\\0 & 1\end{pmatrix}\begin{pmatrix}h_{t-1}\\1\end{pmatrix}$$

矩阵乘可结合 → **associative scan** 用并行 reduce 树算全部 $h_{1:L}$，深度 $O(\log L)$，并行度 $O(L/\log L)$。是 SSM 训练并行的数学基础。

### 4.4 CP 切 scan 的依赖

序列分 chunk $c_1,\dots,c_k$，GPU $i$ 算 $c_i$。$h$ 从 $c_1$ 流到 $c_k$（顺序）。GPU $i$ 需 GPU $i-1$ 的终态 $h^{(i-1)}_{\text{end}}$ 才能开始。是**顺序流水**（pipeline），非 ring 并行。延迟 $\propto$ chunk 数。优化：chunk 间 state 小，传输快；可 overlap 计算。

### 4.5 Hybrid 的混合 cache

[[Hybrid架构]]（Jamba）每层：SSM 层 state $O(N)$ + attention 层 KV $O(L)$（只 $\rho$ 层）。引擎总 cache $\approx \rho L d + N$（每请求）。需同时管两类。


## 5. 代码示例（可选）

### 5.1 state cache 接口（概念，替代 KV cache）

```python
class SSMStateCache:
    """替代 KV cache: 每请求一个固定 state, 预分配, 不增长."""
    def __init__(self, n_reqs, state_dim, device):
        self.state = torch.zeros(n_reqs, state_dim, device=device)  # 固定预分配
        self.req2idx = {}

    def alloc(self, req_id):
        idx = len(self.req2idx); self.req2idx[req_id] = idx; return idx

    def step(self, model, req_id, x_t):
        """decode 一步: 递归更新 state (非增量 KV)."""
        idx = self.req2idx[req_id]
        B_t, C_t, dt_t = model.proj_inputs(x_t)         # 输入相关 (selective)
        A_bar = torch.exp(dt_t * model.A)               # 离散化
        h = self.state[idx]
        self.state[idx] = A_bar * h + dt_t * B_t * x_t   # h_t = bar_A h_{t-1} + bar_B x_t
        return torch.matmul(C_t, self.state[idx])       # y_t = C_t h_t

    def free(self, req_id):
        # 请求结束释放 (state 可复用), 无需 free KV block
        del self.req2idx[req_id]
```

### 5.2 vLLM 风格 Mamba 注册（示意）

```python
# vLLM 模型注册 (伪代码, 概念)
@ SUPPORTS_DECODE_MODES  # 标记非 AR, SSM 递归
class MambaForCausalLM:
    def forward_decode(self, input_ids, state_cache):
        # 非 KV cache, 用 state_cache 递归
        x = self.embed(input_ids)
        for layer in self.mamba_layers:        # 无 attention, 无 RoPE
            x = layer(x, state=state_cache)
        return self.lm_head(x)
# vLLM scheduler 复用 continuous batching, 但 cache manager 用 state_cache
```

### 5.3 associative scan 并行（教学）

```python
def assoc_scan(a, b, init):
    """并行前缀和式 scan: h_t = a_t h_{t-1} + b_t. 教学串行, 真实用 reduce 树."""
    L = len(a); h = init; ys = []
    for t in range(L):
        h = a[t]*h + b[t]; ys.append(h)
    return ys
# 真实 mamba_ssm.selective_scan 用 CUDA reduce 树, depth O(log L)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[Mamba]]（SSM 数学/算子）、[[Hybrid架构]]（混合）、[[KV cache]]（要替换）、[[PagedAttention]]（Hybrid 仍用）、[[continuous batching的调度实现]]（复用）、[[autoregressive decoding]]（替换）、[[新模型接入]]（通用接口）、[[CUDA stream与event]]（kernel 调度）。
- **下游（应用）**: vLLM/SGLang Mamba serving、Megatron SSM 训练、Jamba/Mamba 部署、新架构 LLM 上线。
- **对比 / 易混**:
  - **SSM 适配 vs [[新模型接入]]**：前者是 SSM-specific（state/scan/CP）；后者是通用模型 runner/worker 接口。前者是后者特化。
  - **SSM state cache vs [[KV cache]]**：固定预分配 vs 增量分页；$O(N)$ vs $O(L)$；递归更新 vs 增量计算。
  - **SSM CP vs Transformer CP（ring attention）**：chunk 间传 state 顺序流水 vs ring 传 KV 并行。


## 7. 常见误区与易错点

> [!warning] 误区 1：SSM 能直接用 vLLM 的 KV cache
> 不能。SSM 无增量 KV，是固定 state 递归。需把 KV cache manager 替为 state cache。Hybrid（Jamba）需同时管 state + KV。见 3.1。

> [!warning] 误区 2：SSM CP 用 ring attention
> 不行。ring attention 切 attention（ring 传 KV）；SSM scan 顺序依赖，CP 需 chunk 间顺序传 state（流水，非 ring 并行）。见 3.4、4.4。

> [!warning] 误区 3：SSM 不需位置编码所以全局都不加
> SSM 层不需（递归有序），但 [[Hybrid架构]] 的 attention 层仍需 [[RoPE]]。混合模型只 attention 层加。别全去。

> [!warning] 误区 4：SSM 推理也是 1 token decode
> SSM decode 每步递归更新 state、输出 1 token（类似 AR 1 token），但训练是全序列 scan。推理可 1 token/step（递归）但 state 固定不增长。与 AR 的 KV 增长不同。

> [!warning] 误区 5：selective scan 就是普通 RNN forward
> 不。selective scan 参数输入相关（$B,C,\Delta$ 随 $x$）、$A$ 结构化、用 associative scan 并行 kernel。比朴素 RNN 串行快、可并行训练。朴素串行慢得不可用。

> [!tip] 实践：从 mamba-ssm 起步
> 落地 SSM 训练用官方 `mamba_ssm`/`mamba2_ssm` 算子（含 CUDA selective scan kernel + associative scan）。推理用 vLLM/SGLang 的 Mamba 注册（已支持 Mamba/Mamba-2/Jamba）。Hybrid 用 Jamba 现成。详见 [[新模型接入]]。


## 8. 延伸细节

### 8.1 vLLM 的 Mamba 实现要点

vLLM 支持 Mamba：模型类继承改 `forward` 用 state cache；scheduler 仍 continuous batching 但按请求管理 state（非 KV block）；Hybrid（Jamba）层内区分 SSM（state）与 attention（KV）两套 cache。位置编码只 attention 层。CP 走 chunk 传 state。

### 8.2 Megatron 的 SSM 训练

Megatron 训 SSM：selective scan kernel 集成（[[CUDA stream与event]] 调度）、SSM CP（chunk 间传 state）、associative scan 并行、TP 切 SSM 层（$A,B,C$ 矩阵切）。与 Transformer 的 TP/CP/PP 共存（Hybrid 模型）。

### 8.3 关键论文/工作（已核实）

- **Mamba**: arXiv:2312.00752（selective scan + 硬件感知并行算法）。
- **Mamba-2**: arXiv:2405.21060（SSD 对偶，可借 [[FlashAttention]] tiling，kernel 更高效）。
- **Jamba**: arXiv:2403.19887 / Jamba-1.5 arXiv:2408.12570（Hybrid 落地，vLLM 支持）。

### 8.4 与 Diffusion LLM 系统适配的对照

[[Diffusion LLM缓存与调度]] 是双向 attention 的 KV cache 适配（难复用、近似刷新）。本条是 SSM 的 state 替 KV（固定、无增长）。两个新架构都打破 AR 的 KV cache 增量假设，但方向不同：Diffusion 仍 attention 但双向、SSM 去 attention 用 state。

### 8.5 内容来源

整理自 Mamba (arXiv:2312.00752)、Mamba-2 (arXiv:2405.21060)、Jamba (arXiv:2403.19887) 论文及 vLLM/SGLang 的 Mamba/Mamba-2/Jamba 模型注册、Megatron SSM 训练实践。系统组件对照据 [[KV cache]]、[[PagedAttention]]、[[continuous batching的调度实现]]、[[新模型接入]]、[[CUDA stream与event]]。

---
相关: [[新型架构适配]] | [[Mamba]] | [[Hybrid架构]] | [[新模型接入]] | [[KV cache]] | [[PagedAttention]] | [[continuous batching的调度实现]] | [[autoregressive decoding]] | [[CUDA stream与event]] | [[位置编码]] | [[RoPE]] | [[FlashAttention]] | [[chunked prefill]]
