# 推理 CUDA Graph

> **所属章节**: [[CUDA Graph执行]]
> **所属模块**: [[16-推理引擎源码]]
> **别名**: 推理 CUDA Graph / decode graph capture / graph replay / 推理图捕获与回放 / CUDA Graph pool / cudagraph
> **难度**: 中高（需懂 [[CUDA Graph与graph capture]] + [[kernel launch overhead]] + [[vLLM scheduler]] + [[PagedAttention]] + [[continuous batching的调度实现]]）


## 1. 一句话定义

**推理 CUDA Graph** 是 vLLM / SGLang 等推理引擎在 **decode 路径**上把"一次 forward 内的一串 kernel launch"**捕获成一张可重放的 CUDA Graph**，之后每生成一个 token 只调一次 `cudaGraphLaunch` 整图回放（replay），从而把每步几十个 kernel 的 CPU **launch 开销**从"逐个提交"压成"一次提交"——**capture 一次、replay 多次**，是消除 [[kernel launch overhead]] 在推理侧的落地实现。它的前提是 decode 的 **shape 固定**（每个请求 query_len=1、batch 在一个 bucket 内 padding 后不变），所以图可以锁死参数/地址；运行时只把 `positions / block_table / seq_lens / input_ids` 等**数据** copy 进静态 buffer 再回放。与十三节的 [[CUDA Graph与graph capture]]（讲 CUDA 层 capture/replay 通用机制）不同，本条专讲**推理引擎如何把这套机制工程化**：多 batch bucket 预捕获、graph pool 共享显存、piecewise vs full、causal mask 与动态位置的回放处理、vLLM 的 `CUDAGraphWrapper`/`CudagraphDispatcher`/`BatchDescriptor` 与 SGLang 的 `DecodeCudaGraphRunner`/`cuda_graph_buffer_registry`。

> [!note] 三句话定位
> - **是什么**：把 decode 一次 forward 的所有 kernel 捕获成图，replay 时一次 `cudaGraphLaunch` 执行全图，省逐 kernel launch 开销。
> - **为什么**：decode 每 token 数十个小 kernel，单个 GPU 计算仅几 μs 但 launch 占 2–5 μs/次，launch 累积成 CPU 瓶颈；capture→replay 把它压到接近一次 launch。
> - **关键工程点**：静态 shape 要求 → 按 batch size bucket 预捕获多张图 + graph pool 共享显存；动态数据（position/block table）靠 replay 前 `static_buf.copy_(real)` 注入；prefill 因长度变化通常走 piecewise 或不捕获。


## 2. 为什么需要它（动机与背景）

### 2.1 decode 是"小 kernel 密集 + shape 固定"的完美靶子

LLM 自回归 **decode** 每生成一个 token 要跑：embedding lookup → 多层 {RMSNorm → QKV GEMM → RoPE → **attention**（query_len=1，访存主导）→ O proj GEMM → RMSNorm → gate/up GEMM → SiLU → down GEMM} → final norm → lm_head GEMM → **采样**。一个 7B 模型一层就有十几个 kernel，32 层即 **数百个 kernel / token**。

这些 kernel 有两个特点：

1. **小**：decode 的 GEMM 是 `[batch, 1, hidden] × [hidden, hidden]`，batch 小（如 32）时算力几 μs，但 [[kernel launch overhead]] 每次 2–5 μs（CPU 解析参数、走 runtime/driver、提交命令队列）。GPU 端算 1 μs、launch 5 μs → **80%+ 时间浪费在 launch**。
2. **shape 固定**：同一 batch bucket 内，每步的 tensor shape 完全一样（`[B,1,H]`、`[B,1,n_head,head_dim]`），地址也可固定（用静态 buffer）。

→ 完美匹配 CUDA Graph 的"捕获一次、参数锁死、replay 多次"模型。数百 kernel 的 launch 从 `300×5μs≈1.5ms` 压到一次 `~5μs`，**decode 吞吐提几十倍**。这是 vLLM/SGLang decode 路径的核心加速，没有它 decode 会被 CPU launch 主导。

> [!warning] 误区：以为 CUDA Graph 是"加速 GPU 计算"
> 不是。graph **不改 GPU 端 kernel 的执行耗时**（同一个 GEMM 在图内/图外跑一样快）。它省的是 **CPU 侧逐次 launch 的固定开销**——把 N 次 launch 提交压成 1 次。所以收益在"kernel 多而小、launch 占比高"的 decode 上巨大，在"kernel 少而大、算力主导"的 prefill 上很小。

### 2.2 为什么 prefill 不（太）用 graph

**prefill** 的 query_len 随 prompt 长度变化（512、2048、8192…），shape 不固定 → 捕获的图只能匹配一个 shape，换长度就得重捕或维护海量图。且 prefill 是 **compute-bound**（大 GEMM 算几十 ms），launch 占比 <1%，收益微小。故推理引擎默认 **decode 全图、prefill 走 piecewise 或 eager**（见 3.5 的 `FULL_AND_PIECEWISE`）。

### 2.3 与训练图的区别

训练里也用 CUDA Graph（如 `torch.cuda.graph` 包裹一个 step），但训练 **shape 动态**（batch 随数据变、序列长度变、梯度反传 shape 跟前向耦合），往往只能捕获静态部分或频繁重捕；训练一个 step 算几十/几百 ms，launch 占比本就低，graph 收益远不如推理 decode。详见十三节 [[CUDA Graph与graph capture]]。下表对比：

| 维度 | 训练 CUDA Graph | 推理 CUDA Graph |
|---|---|---|
| 主要受益阶段 | 一个 training step | **decode 每 token** |
| shape 稳定性 | 动态（batch/seq 变） | **固定**（query_len=1，bucket 内 padding 不变） |
| kernel 数 | 少（几个大 GEMM+反传） | **数百个小 kernel** |
| launch 占比 | 低（<5%） | **高（>50%，甚至 80%）** |
| 收益 | 中 | **巨大（几十倍 decode 吞吐）** |
| 捕获策略 | 单图 + 偶尔重捕 | **多 bucket 预捕获 + graph pool 共享** |
| 数据更新 | 少 | 多（position/block_table/seq_lens 每步 copy） |


## 3. 核心概念详解

### 3.1 capture 与 replay 两阶段

```
[阶段1: capture (启动时一次性)]
  for bs in capture_bs:            # 预捕获若干 batch size bucket
    分配静态 input/output buffer (地址固定)
    cudaStreamBeginCapture(stream)
      model(static_input)           # 跑一遍, 所有 kernel 被录入图
    cudaStreamEndCapture -> graph
    cudaGraphInstantiate -> exec    # 编译成可执行实例
    graphs[bs] = exec                # 存池, 按 bs 取用

[阶段2: replay (每步 decode)]
  bs = pad(current_batch)           # 当前 batch 向上取整到最近的 bucket
  static_input.copy_(real_input)    # 注入真实数据: input_ids/positions/block_table...
  cudaGraphLaunch(graphs[bs], stream)  # 一次提交整图, CPU 只走一次
  output = static_output            # 结果已在静态 output buffer
```

- **capture**：启动时用 `_dummy_run` 跑一遍 forward（带静态 buffer），把整条 kernel 序列录成图并实例化。**只发生一次**（per bucket）。
- **replay**：每步 decode 把真实数据 copy 进静态 buffer，一次 `cudaGraphLaunch`。**发生成千上万次**。

> [!tip] 核心：capture 一次 → replay 多次省 CPU launch
> 这就是"摊销"——把 N 次 launch 的固定成本（参数解析、driver 往返）在 capture 时付一次，replay 只走 GPU 端轻路径（≈一次 launch 的 5 μs，无论图内几个 kernel）。

### 3.2 静态 shape 是前提，数据靠 copy 注入

图捕获时**锁死每个 kernel 的参数与 tensor 地址**。replay 要求后续用**同样 shape、同样地址的 buffer**。所以：

- **静态 buffer**：input_ids、positions、block_table、seq_lens、output logits 等都用预分配的固定地址张量（capture 时即用它们）。
- **数据注入**：每步把真实值 `copy_` 进去（地址不变，只改内容）。图内 kernel 读的就是这些 buffer（已被刷新）。

```
# 静态 buffer (地址在 capture 时锁进图)
static_input_ids  : [max_bs, 1]        int64
static_positions  : [max_bs, 1]        int64
static_block_table: [max_bs, max_blocks] int32   # 指向 KV cache block
static_seq_lens   : [max_bs]           int32
static_out_logits : [max_bs, vocab]    fp16

# 每步 replay 前 (CPU→GPU copy, 地址不变)
static_input_ids[:bs].copy_(real_ids)
static_positions[:bs].copy_(real_pos)
static_block_table[:bs].copy_(real_bt)
static_seq_lens[:bs].copy_(real_sl)
cudaGraphLaunch(graphs[bs])   # 图内 attention 读 block_table -> 访问正确的 KV block
logits = static_out_logits[:bs]
```

### 3.3 causal mask 与动态位置的处理（重点易错）

这是初学者最困惑的点：**图是静态的，但每个请求的位置、序列长度、KV block 地址每步都变，怎么回放？**

答案：**计算结构固定，数据可变**。

- **位置 positions**：每步不同（请求 A 在第 50 token、B 在第 30），但 shape 固定 `[B,1]`。replay 前 `static_positions.copy_(real_pos)`，RoPE / QKV / attention 内部用这个 buffer 算，读到的就是新值。
- **序列长度 seq_lens**：每步 +1，但 shape 固定 `[B]`。`static_seq_lens.copy_(real_sl)`，attention kernel 据 `seq_lens` 决定每个请求 attend 到哪些 KV。
- **block_table（KV 地址）**：每步可能申请新 KV block，[[PagedAttention]] 用 block_table 映射逻辑 token→物理 block。block_table 内容变、shape 不变 `[B, max_blocks]`，`static_block_table.copy_(real_bt)`，attention kernel 据此访存正确的物理 block。
- **causal mask**：
  - **纯 decode（query_len=1）**：单 query token 对全部历史 KV，**不需要三角 mask**，attention 退化为 `[B,1,n_head,head_dim] × [B,seq_len,n_head,head_dim]`，mask 是全 1，故 decode 图天然友好。
  - **prefill / 混合（query_len>1）**：需要三角 mask，mask shape 随 `max_query_len × seq_len` 变。能捕获 full graph 的 attention backend（如 FlashAttention v3）在图内据 `seq_lens` 重新算 mask；不支持的就降级到 piecewise（见 3.5）。

> [!warning] 误区：以为图把"mask 也固化了"
> 不是。**图固化的是"调用哪些 kernel、kernel 的 shape/地址"**，不是"数据的值"。mask 在 attention kernel 内部据（每步被 copy 刷新的）`seq_lens` 重新构造，故每步 mask 不同但图能复用。这正是 capture 时要"用代表性 attn_metadata 跑一遍"以确保走对 kernel 分支。

### 3.4 多 batch bucket 预捕获与 graph pool 共享显存

**多 bucket**：continuous batching 每步 batch size 变（请求进出）。不可能每步重捕图，故预捕获一批 bucket（如 `1,2,4,8,...,256`），运行时把当前 batch 向上 **padding 到最近的 bucket** 再 replay 对应图。padding 浪费一点算力，但免重捕。

**graph pool 共享显存**：每张图都要静态 input/output buffer，N 个 bucket 全独立分配会爆显存。PyTorch 提供 `torch.cuda.graphs.graph_pool_handle()` 返回一个**显存池句柄**，多个图 capture 时传 `pool=handle`，使它们**复用同一片地址空间**（同一 bucket 大小的图共享 buffer 地址）。vLLM/SGLang 都用此机制，称 **cuda graph pool / graph pool**。一个池里"同 shape 的图共享地址、不同 shape 的图按最大需求分配"，大幅省显存。

> [!note] 补充：cudagraph trees / 多图复用
> PyTorch 内部用一个"图树"结构管理"共享同一 pool 的多个捕获图"的实例化与回放（捕获时把图挂到 pool 对应节点，replay 时按 pool 路径执行）。公开 API 是 `graph_pool_handle()` + `torch.cuda.CUDAGraph().capture_begin(pool=...)`；vLLM 老版 `CUDAGraphRunner` 即用它让 per-bucket 图共享 `mem_pool`。该内部树结构的具体类名随版本变（待核实），但"多图共享一个显存地址池"的语义稳定。

### 3.5 vLLM 的工程化：CUDAGraphMode / Wrapper / Dispatcher

vLLM v1 把图逻辑从 `torch.compile` 解耦，引入一套显式调度。核心组件（基于官方 design doc，2026 年 main 分支）：

| 组件 | 职责 |
|---|---|
| **`CUDAGraphMode`** | 枚举 knob：`NONE` / `PIECEWISE` / `FULL` / `FULL_DECODE_ONLY` / `FULL_AND_PIECEWISE`（默认） |
| **`CUDAGraphWrapper`** | 包裹一个 callable，负责单张图的 capture & replay；每个实例绑一个 `runtime_mode`（`FULL` 或 `PIECEWISE`） |
| **`CudagraphDispatcher`** | 中央控制器，维护 `FULL` / `PIECEWISE` 两套合法 dispatch key，运行时按 batch 选 mode + key |
| **`BatchDescriptor`** | dispatch key：`(num_tokens, num_reqs, uniform, has_lora)`，唯一标识一个（padding 后的）batch |
| **`AttentionCGSupport`** | attention backend 的 graph 能力枚举：`ALWAYS > UNIFORM_BATCH > UNIFORM_SINGLE_TOKEN_DECODE > NEVER` |

**两种 runtime mode**：

- **`FULL`（full graph）**：把整个 model forward 捕成一张图，attention 也在图内。要求 attention backend 支持（FlashAttention v3 = `ALWAYS`；FlashAttention v2 标 `UNIFORM_BATCH` 实为 `ALWAYS` 但降级走 piecewise 求性能）。`FULL` 对 prefill/mixed 和 uniform decode 都捕获，decode 复用同 batch_size 的非 uniform 图。
- **`PIECEWISE`（分片图）**：`torch.compile` 把图按 `splitting_ops`（如 attention）切成多片，**attention 留 eager、其余片各自捕获小图**。兼容所有 backend，但片间仍有 eager 切换开销。

**双 mode 默认 `FULL_AND_PIECEWISE`**：uniform decode 走 FULL（最快）、prefill/mixed 走 PIECEWISE（兼容），dispatcher 运行时按 `BatchDescriptor.uniform` 自动切换。`FULL_DECODE_ONLY` 专为 P/D 分离的 decode 实例（省掉 piecewise 的显存）。

**嵌套 wrapper 设计**：外层一个 `FULL` wrapper 包整个 model；每个 piecewise backend 内再包一个 `PIECEWISE` wrapper。FULL 激活时内层 piecewise 不激活，反之亦然，无冲突。

**dispatch 优先级**：`FULL > PIECEWISE > NONE`。找不到 key 则降级 eager。

```
# vLLM 运行时（简化，源自 design doc）
batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens, uniform=...)
runtime_mode, batch_descriptor = cudagraph_dispatcher.dispatch(batch_descriptor)
with set_forward_context(..., cudagraph_runtime_mode=runtime_mode,
                         batch_descriptor=batch_descriptor):
    output = self.model(...)   # wrapper 内: 命中则 replay, 否则 eager
```

**捕获时机**：`gpu_model_runner` 首次用非 `NONE` mode 调 `_dummy_run` 时触发；warmup 用 `NONE` mode 做 eager 跑一遍（含显式跑一次 attention），再切到目标 mode 捕获。`cudagraph_capture_sizes`（compilation config）列出要预捕获的 batch size 列表。

### 3.6 SGLang 的工程化：runner 包 + buffer registry

SGLang 把执行器拆成 `runner/` 包，按场景分（基于 GitHub main 分支 2026 年结构，待核实具体类名随版本演化）：

| 文件 | 角色 |
|---|---|
| `model_executor/runner/eager_runner.py` | `EagerRunner`：不捕获，逐 kernel eager |
| `model_executor/runner/decode_cuda_graph_runner.py` | `DecodeCudaGraphRunner`：decode 路径 full graph 捕获/回放 |
| `model_executor/runner/prefill_cuda_graph_runner.py` | `PrefillCudaGraphRunner`：prefill 路径捕获（仅对支持的长度/batch） |
| `model_executor/runner/base_cuda_graph_runner.py` | 公共 capture/replay/静态 buffer 逻辑 |
| `model_executor/runner/shape_key.py` | **shape key**：唯一标识一个可捕获 batch（≈ vLLM 的 BatchDescriptor） |
| `model_executor/cuda_graph_buffer_registry.py` | 注册所有静态 input/output buffer，跨图复用（≈ graph pool） |
| `model_executor/graph_shared_output.py` | 多图共享的 output buffer |
| `model_executor/pool_configurator.py` / `runner_utils/pool.py` | 显存池配置 |
| `model_executor/runner_backend/*` | 多种 backend：`full_cuda_graph_backend`、`breakable_cuda_graph_backend`（可中断图）、`tc_piecewise_cuda_graph_backend`（torch.compile piecewise，≈ vLLM PIECEWISE）、`cuda_graph_dedup_mixin`（去重） |

SGLang 同样有"full vs piecewise"对偶（`full_cuda_graph_backend` vs `tc_piecewise_cuda_graph_backend`），与 vLLM 的 `FULL`/`PIECEWISE` 思路一致。`breakable_cuda_graph` 是 SGLang 特色：允许图在运行中"打断"（如遇到需要 host 介入的条件），比纯 full graph 更灵活。`capture_bs` 列表预捕获多 bucket，运行时按 shape key 取图；`cuda_graph_buffer_registry` 统一管理静态 buffer 地址，实现跨图共享（即 SGLang 版的 cuda graph pool）。

### 3.7 padding 到 bucket 的取舍

```
capture_bs = [1,2,4,8,16,32,64,128,256]
当前 batch=50 -> pad 到 64 -> replay graphs[64]  (浪费 14 个槽的算力)
当前 batch=200 -> pad 到 256 -> replay graphs[256]
```

bucket 太密：捕获慢、显存多；太稀：padding 浪费大。经验取 2 的幂 + workload 常见 batch（vLLM 默认一组 2 的幂，可配 `cudagraph_capture_sizes`）。


## 4. 数学原理 / 公式

### 4.1 launch 开销占比模型

设 decode 单步有 $K$ 个 kernel，第 $i$ 个 kernel 的 **GPU 计算耗时** $t_{\text{compute},i}$、**CPU launch 开销** $t_{\text{launch}}\approx 2\sim5\,\mu s$（与 $i$ 无关，固定）。

不用 graph，单步总耗时（CPU/GPU 串行化时）：

$$
T_{\text{eager}} = \sum_{i=1}^{K}\big(t_{\text{launch}} + t_{\text{compute},i}\big) = K\cdot t_{\text{launch}} + \sum_{i} t_{\text{compute},i}
$$

用 graph 后，所有 launch 压成一次 `cudaGraphLaunch`（开销 $t_{\text{graph\_launch}}\approx t_{\text{launch}}$，与 $K$ 无关）：

$$
T_{\text{graph}} = t_{\text{graph\_launch}} + \sum_{i} t_{\text{compute},i}
$$

加速比（launch 主导时，即 $K\cdot t_{\text{launch}} \gg \sum t_{\text{compute},i}$）：

$$
\text{speedup} \approx \frac{K\cdot t_{\text{launch}}}{t_{\text{launch}}} = K
$$

即 **kernel 数越多、launch 开销占比越高，graph 收益越接近 $K$ 倍**。decode 的 $K$ 达数百，故收益巨大；prefill 的 $K$ 小且 $\sum t_{\text{compute}}$ 大，$\text{speedup}\to 1$，收益微。

### 4.2 显存池共享的地址复用

设 $S$ 为所有 bucket 的静态 buffer 集合。不共享时显存 $\propto \sum_{s\in S}\text{size}(s)$；共享 pool 后，同 shape 的图复用同一地址（同一物理页），不同 shape 取上确界：

$$
M_{\text{pool}} \approx \max_{\text{shape-class}} \text{size}(s) \quad\text{（而非求和）}
$$

故 N 个 bucket 的显存从 $O(N)$ 降到接近 $O(\text{bucket 种类数})$，是预捕获多 bucket 可行的关键。

### 4.3 padding 浪费率

当前真实 batch $b$，向上取整到 bucket $B=\text{next\_pow2}(b)$：

$$
\text{waste}(b) = \frac{B - b}{B} \in [0, 0.5)
$$

最坏 $b=B/2+1$ 时浪费接近 $1/2$；故 bucket 列表要覆盖 workload 常见 batch 以降低平均浪费。


## 5. 代码示例（可选）

### 5.1 最小 capture→replay（PyTorch 原生，可跑）

```python
import torch

device = "cuda"
model = torch.nn.Linear(4096, 4096).to(device).eval()
B = 32

# 静态 input/output (地址固定, capture 时锁进图)
static_in = torch.randn(B, 4096, device=device)
static_out = torch.empty(B, 4096, device=device)

# [阶段1] capture: 跑一遍, 录成图
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_in)          # kernel 被录入图, 输出地址固定

# [阶段2] replay: 每步只 copy 数据 + launch
real_in = torch.randn(B, 4096, device=device)
static_in.copy_(real_in)                  # 注入数据 (地址不变)
g.replay()                                 # 一次提交全图
assert torch.allclose(static_out, model(real_in), atol=1e-4)  # 与 eager 等价
```

> [!tip] 实践：replay 等价于重跑 model(real_in)，但只花一次 launch
> 把 `model(real_in)` 改成 `static_in.copy_; g.replay()` 后用 nsys 看：CPU 侧从"一串 cuda launch"变成"一次 cudaGraphLaunch"，GPU 侧 kernel 序列不变。

### 5.2 多图共享显存池（graph pool）

```python
import torch

pool = torch.cuda.graph_pool_handle()      # 共享显存池句柄
graphs, bufs = {}, {}
for bs in [1, 2, 4, 8, 16, 32]:            # 预捕获各 bucket
    inp = torch.randn(bs, 4096, device="cuda")
    out = torch.empty(bs, 4096, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):   # 传 pool -> 共享地址空间
        out = model(inp)
    graphs[bs] = g
    bufs[bs] = (inp, out)

# 运行时按 batch 取图 (这里示意, 不 padding)
cur_bs = 8
bufs[cur_bs][0].copy_(real_input_8)
graphs[cur_bs].replay()
result = bufs[cur_bs][1]
```

### 5.3 vLLM：开 FULL_AND_PIECEWISE（CLI / Python）

```python
# CLI: vllm serve --model meta-llama/Llama-3.1-8B-Instruct \
#       --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")
import vllm
from vllm.config import CUDAGraphMode  # 枚举: NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE

model = vllm.LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    compilation_config={"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"},
)
out = model.generate(["My name is John and"],
                     sampling_params=vllm.SamplingParams(temperature=0, max_tokens=1024))
```

> [!note] 补充：piecewise 需要 torch.compile
> `PIECEWISE` 系 mode 依赖 piecewise compilation（`torch.compile` 切片）；`FULL` 系 mode 依赖 attention backend 支持 graph。mode 不匹配时 vLLM 自动降级到最近的合法 mode（如 backend 只支持 uniform decode 就把 `FULL` 降到 `FULL_AND_PIECEWISE` 或 `PIECEWISE`）。


## 6. 与其他知识点的关系

- **上游（依赖）**: [[CUDA Graph与graph capture]]（CUDA 层 capture/replay 通用机制，本条是其推理落地）、[[kernel launch overhead]]（被消除的对象）、[[CUDA stream与event]]（graph 从 stream 捕获、回放到 stream）
- **下游（应用）**: [[model runner]]（runner 调 `_dummy_run` 触发捕获、execute 时 replay）、[[worker与executor]]（worker 进程内 runner 持图）、[[continuous batching的调度实现]]（batch 变化触发换 bucket）、[[vLLM scheduler]]（决定 batch 组成→决定用哪张图）
- **对比 / 易混**:
  - vs [[CUDA Graph与graph capture]]：十三节讲机制，本条讲推理工程化（bucket/pool/dispatcher）。
  - vs 训练 graph：训练 shape 动态、收益低；推理 decode shape 固定、收益几十倍（见 2.3 表）。
  - vs `torch.compile`：compile 是算子融合/图优化（减 kernel 数、改 kernel 实现）；graph 是压 launch 开销。两者正交，vLLM 的 piecewise 把 compile 切片与 graph capture 结合。
  - FULL vs PIECEWISE：full 把 attention 也进图（快但要求 backend 支持），piecewise 把 attention 留 eager（兼容但慢）。


## 7. 常见误区与易错点

> [!warning] 误区1：以为 graph 能处理任意 batch
> 不能。图锁死 shape，batch 超出预捕获 bucket 要么 padding 到更大 bucket、要么降级 eager。新 batch size 没有对应图 → 性能掉回 eager。

> [!warning] 误区2：以为 replay 不能改输入
> 能改 **数据**（值），不能改 **地址/shape**。`static_buf.copy_(real)` 是标准做法；但要换 tensor 地址/shape 必须重捕图。

> [!warning] 误区3：以为 graph 加速 GPU 计算
> graph 不改 kernel 执行耗时，只省 CPU launch。prefill 大 GEMM 算力主导，开 graph 几乎无收益甚至因 capture 浪费显存。

> [!warning] 误区4：捕获时忘记 warmup / 走错 attention 分支
> capture 时若 attention metadata 不代表运行时（如把 decode 当 prefill 捕获），图内 kernel 分支错，replay 结果错。vLLM warmup 显式用 `NONE` mode eager 跑一遍（含 attention）再捕获。

> [!warning] 误区5：多 bucket 全独立分配显存
> 不用 `graph_pool_handle()` 共享会爆显存。必须让 per-bucket 图共享 pool。

> [!warning] 误区6：把含 host 同步 / 动态控制流的算子进图
> graph 要求纯 GPU 序列。若算子内有 `torch.cuda.synchronize()`、CPU 端 if/loop 据张量值分支，捕获会失败或语义错。vLLM piecewise 即为此把不兼容算子（attention 某些 backend）留 eager。


## 8. 延伸细节

- **breakable cuda graph（SGLang）**：允许图执行中"打断"回到 host 处理条件逻辑再续跑，介于 full 与 piecewise 之间，兼顾性能与灵活性（`runner_backend/breakable_cuda_graph_backend.py`）。
- **capture 时的内存副作用**：capture 期间所有 CUDA 分配进图的工作内存，若模型有动态分配（如某些 MoE 的 token 重排）需先固定 shape 或排除出图。
- **speculative decode 与 graph**：spec decode 的"verify"步 query_len=1+num_spec_tokens，仍属 uniform batch，可捕获（`UNIFORM_BATCH` 级支持）。SGLang 有专门的 `eagle_draft_cuda_graph_runner.py` 等捕获 draft model。
- **多模态 ViT**：vision encoder 也可捕获图（vLLM 有 `cuda_graphs_multimodal` design；SGLang `vit_cuda_graph_runner.py`），但因图像 token 数变化，多按图像尺寸 bucket 捕获。
- **调试**：`cudagraph_mode=NONE` 关图走 eager 对照；用 nsys/ncu 看 CPU 侧 launch 是否压成单次 `cudaGraphLaunch`。
- **与 [[PagedAttention]] 协同**：graph 把 attention kernel 调用固化，attention 运行时据（被 copy 刷新的）`block_table` 访问正确 KV block——两者协同实现"静态图 + 动态 KV 寻址"。
- **TTFT/TPOT 影响**：graph 主要降 **TPOT**（每 token 延迟，decode 更快）；对 **TTFT**（首 token，走 prefill/eager）影响小。相关 [[TTFT与TPOT]]。

---
相关: [[CUDA Graph执行]]、[[model runner]]、[[worker与executor]]、[[PagedAttention]]、[[continuous batching的调度实现]]、[[vLLM scheduler]]、[[CUDA Graph与graph capture]]、[[kernel launch overhead]]
