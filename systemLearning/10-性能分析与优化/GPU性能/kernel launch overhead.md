# kernel launch overhead

> **所属章节**: [[GPU性能]]
> **所属模块**: [[10-性能分析与优化]]
> **别名**: kernel launch overhead / kernel 启动开销 / launch latency / CPU dispatch overhead
> **难度**: 中（需懂 CUDA stream + CPU-GPU 异步）


## 1. 一句话定义

**kernel launch overhead** 是 CPU 向 GPU **提交一个 kernel** 的开销（约 **5-10μs/kernel**），包括 CPU 端构建命令、经 driver/runtime 提交到 GPU 命令队列、GPU 前端解析启动。单个 launch 开销小，但**小 kernel 高频提交**时累积成显著瓶颈——如 LLM 推理每 token 生成涉及数十~数百 kernel（每层 attention+MLP+norm+激活多个），若每 kernel 计算量小（μs 级），launch overhead 淹没计算，GPU 大部分时间在等 CPU 提交下一 kernel（**CPU-bound**）。解法是 **kernel fusion**（融合多 kernel）+ **CUDA Graph**（预录命令批量回放），把 launch 次数从每 token 数百降到几次。

> [!note] 三句话定位
> - **是什么**：CPU 提交一个 kernel 到 GPU 的开销（~5-10μs）。
> - **为什么**：小 kernel 高频提交时，launch 开销累积淹没计算，GPU 等 CPU（CPU-bound）。
> - **解法**：kernel fusion（融合多操作）+ CUDA Graph（预录批量回放），降 launch 次数。


## 2. 为什么需要它（动机与背景）

### 2.1 CPU-GPU 异步模型

CUDA 的 kernel 提交是**异步**的：CPU 调 `kernel<<<...>>>()` 立即返回（不阻塞等 GPU 完成），命令进 GPU 的 stream 队列，GPU 按序执行。这让 CPU 与 GPU **并行**——CPU 提交下一 kernel 时 GPU 在执行上一 kernel。

但这要求**CPU 提交快于 GPU 执行**才能掩盖。若 CPU 提交慢（launch overhead 大）、GPU 执行快（小 kernel），GPU 执行完上一 kernel 后等 CPU 提交下一 kernel → **CPU-bound**（GPU 空闲等 CPU）。

### 2.2 小 kernel 的开销占比

LLM 推理 decode 单 token，单层 ~10 个 kernel（QKV proj、attention、O proj、FFN up、激活、FFN down、norm、residual）。32 层 → 320 kernel/token。若每 kernel 计算 10μs、launch 5μs：

- 计算：320 × 10μs = 3.2ms
- launch：320 × 5μs = 1.6ms

launch 占总时间 1.6/(3.2+1.6) = 33%。1/3 时间花在 launch 上，GPU 等 CPU。

若 kernel 更小（如 elementwise 2μs）：launch 5μs > 计算 2μs，**launch 比计算还久**，严重 CPU-bound。

### 2.3 launch overhead 的组成

一次 launch 的开销来自：

1. **CPU 端构建命令**：参数打包、grid/block 配置（μs 级）。
2. **driver/runtime 提交**：经 CUDA driver 提交到 stream 队列（μs 级）。
3. **GPU 前端解析**：GPU 取命令、解析、启动 kernel（μs 级）。

合计 ~5-10μs/kernel（依 CPU、driver、GPU 型号）。这个开销**与 kernel 计算量无关**——小 kernel 和大 kernel 的 launch 开销一样（都是 5-10μs）。所以小 kernel 受 launch overhead 影响最大。

### 2.4 LLM 推理的 launch 问题

LLM 推理（特别是 decode）每 token 数百 kernel：

- 每 layer：QKV proj, attention (QK, softmax, PV), O proj, FFN up, 激活, FFN down, norm × 2, residual × 2 → ~10-15 kernel。
- 32-80 layer → 320-1200 kernel/token。

若不优化，每 token 的 launch overhead 累积 1.6-6ms，严重拖慢 decode（本可几十 ms）。这是 LLM 推理框架（vLLM、SGLang）必优化项。


## 3. 核心概念详解

### 3.1 CUDA stream 与异步提交

- **stream**：GPU 命令队列，按序执行。默认 stream 或用户创建的多 stream。
- **异步提交**：`kernel<<<...>>>()` 立即返回，命令进 stream，GPU 异步执行。
- **同步点**：`cudaDeviceSynchronize()` 或 `torch.cuda.synchronize()` 阻塞等所有 stream 完成。

launch overhead 是 CPU 端提交的开销，与 GPU 执行并行。但若提交跟不上执行，GPU 空。

### 3.2 CPU-bound vs GPU-bound

- **GPU-bound**：计算重，GPU 满载，CPU 提交快，launch overhead 被掩盖。
- **CPU-bound**：小 kernel 多，CPU 提交慢，GPU 等 CPU，launch overhead 暴露。

判断：CPU 利用率高 + GPU 利用率低 → CPU-bound（launch 瓶颈）。

### 3.3 kernel fusion（融合）

把多个小 kernel 融合成一个大 kernel：

- elementwise + activation + residual → 一个 fused kernel。
- QKV proj + RoPE → 一个。

fusion 减 launch 次数 + 减中间结果写回 HBM（[[memory bandwidth]] 优化）。torch 的 `torch.compile`、Triton、FlashAttention 都大量用 fusion。

### 3.4 CUDA Graph

预录一段 kernel 序列为 **graph**，之后**批量回放**：

```python
g = cuda_graph()
for _ in iter:
    g.replay()   # 一次回放 = 整段 kernel, launch overhead 摊到 1 次
```

CUDA Graph 把 N 个 launch 合成 1 次回放，launch overhead 从 $N \times 5\mu s$ 降到 $\sim 5\mu s$。LLM 推理的每 token 前向用 graph 回放，launch overhead 几乎消除。

### 3.5 launch overhead 与 SM utilization 的关系

launch overhead 大 → GPU 等 CPU → [[SM utilization]] 低（SM 空闲）。但与 memory-bound 的 SM 低不同：launch-bound 的 SM 是**完全空闲**（没 kernel 执行），memory-bound 的 SM 是**有 kernel 但等数据**。


## 4. 数学原理 / 公式

### 4.1 launch overhead 占比

$N$ 个 kernel，每 kernel 计算时间 $T_{\text{exec}}$、launch 开销 $T_{\text{launch}}$（如 5μs）：

$$
T_{\text{total}} = N \cdot (T_{\text{exec}} + T_{\text{launch}}) \quad \text{(CPU-bound, 提交跟不上)}
$$

或（GPU-bound，launch 被掩盖）：

$$
T_{\text{total}} = N \cdot T_{\text{exec}} \quad \text{(若 } T_{\text{launch}} < T_{\text{exec}} \text{)}
$$

launch 占比：

$$
\text{overhead ratio} = \frac{N \cdot T_{\text{launch}}}{T_{\text{total}}}
$$

> [!note] 小 kernel 的 launch 占比
> $T_{\text{exec}}=10\mu s$、$T_{\text{launch}}=5\mu s$ → 占比 33%。$T_{\text{exec}}=2\mu s$ → 占比 71%。$T_{\text{exec}}=100\mu s$ → 占比 5%（可忽略）。**kernel 越小，launch overhead 占比越大**——大 kernel 不受影响，小 kernel 是优化重点。

### 4.2 fusion 收益

$K$ 个小 kernel（各 $T_{\text{exec}}$）融合成 1 个（$T_{\text{fused}} \le K \cdot T_{\text{exec}}$，可能更快因减内存访问）：

$$
T_{\text{before}} = K \cdot (T_{\text{exec}} + T_{\text{launch}}), \quad T_{\text{after}} = T_{\text{fused}} + T_{\text{launch}}
$$

节省：

$$
\Delta T = (K-1) \cdot T_{\text{launch}} + (K \cdot T_{\text{exec}} - T_{\text{fused}})
$$

$K=10$、$T_{\text{launch}}=5\mu s$ → 省 $45\mu s$ launch + 减内存访问。

### 4.3 CUDA Graph 收益

$N$ 个 launch 合成 1 次回放：

$$
T_{\text{launch, before}} = N \cdot 5\mu s, \quad T_{\text{launch, after}} \approx 5\mu s
$$

$N=320$ → 省 $319 \times 5 = 1595\mu s = 1.6$ms。LLM decode 每 token 省 1.6ms，显著。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 launch overhead 累积

```python
def simulate_inference(num_layers, kernels_per_layer, kernel_exec_us, launch_us=5):
    n = num_layers * kernels_per_layer
    total_exec = n * kernel_exec_us
    total_launch = n * launch_us
    overhead_pct = total_launch / (total_exec + total_launch) * 100
    return total_exec + total_launch, overhead_pct

# 7B decode: 32 layer × 10 kernel, 每 kernel 10μs, launch 5μs
total, pct = simulate_inference(32, 10, 10, 5)
print(f'未优化: {total/1e3:.2f}ms, launch 占比 {pct:.0f}%')

# kernel fusion: 10 kernel 融成 2
total2, pct2 = simulate_inference(32, 2, 30, 5)   # 2 kernel/layer, 算更多
print(f'fusion: {total2/1e3:.2f}ms, launch 占比 {pct2:.0f}%')

# CUDA graph: launch 摊到 1 次
n = 32 * 10
exec_total = n * 10
graph_launch = 5                                   # 1 次回放
print(f'CUDA graph: {(exec_total+graph_launch)/1e3:.2f}ms, launch 占比 {graph_launch/(exec_total+graph_launch)*100:.1f}%')
```

### 5.2 真实 PyTorch CUDA Graph 对照

```python
# import torch
# # 预录 graph
# static_input = torch.zeros(batch, seq, dim, device='cuda')
# g = torch.cuda.CUDAGraph()
# with torch.cuda.graph(g):
#     out = model(static_input)
# # 推理时回放 (copy 真输入到 static_input, replay)
# static_input.copy_(real_input)
# g.replay()
# # launch overhead 从 ~320 次降到 1 次
```


## 6. 与其他知识点的关系

- **上游（依赖）**: CUDA stream/异步模型、CPU-GPU 协同。
- **下游（应用）**: [[CPU-GPU pipeline stall]]（launch 瓶颈的下游表现）、[[SM utilization]]（launch-bound 时 SM 空闲）、[[memory bandwidth]]（fusion 同时优化 launch 和带宽）、[[torch profiler]]/[[nsys]]（测 launch 开销）。
- **对比 / 易混**:
  - **launch overhead vs [[memory bandwidth]] 瓶颈**：launch 是 CPU 提交开销（CPU-bound，GPU 空闲），bandwidth 是数据供给不足（memory-bound，GPU 有 kernel 但等数据）。两者都让 GPU 慢，但根因和解法不同。
  - **launch overhead vs [[kernel launch overhead]]**（自身，无对比）。
  - **kernel fusion vs CUDA Graph**：fusion 减 kernel 数（算法层），graph 减 launch 次数（运行时层）。两者互补，常并用。

> [!note] 此条不涉及
> 与"GPU 内存带宽"的混淆已在误区 1 澄清。


## 7. 常见误区与易错点

> [!warning] 误区 1：异步提交就没开销
> 异步只是不阻塞 CPU，但 launch overhead 仍存在（CPU 花 5μs 提交）。若提交跟不上 GPU 执行，GPU 等 CPU。异步 ≠ 零开销。

> [!warning] 误区 2：小 kernel 无害
> 小 kernel（μs 级）的 launch overhead（5μs）可能比计算还久。数百小 kernel 累积 launch 开销成显著瓶颈。小 kernel 是 launch overhead 的最大受害者。

> [!warning] 误区 3：忽略 CUDA Graph
> 不用 graph 时，每 token 数百 kernel 各自 launch，累积 ms 级开销。graph 把整段预录回放，launch 摊到 1 次，几乎消除开销。LLM 推理必用。

> [!warning] 误区 4：fusion 只为减 launch
> fusion 减 launch，但更大收益是**减中间结果写回 HBM**（[[memory bandwidth]] 优化）。如 elementwise+activation 融合，中间结果留 register/smem，不写 HBM。launch 是次要收益。

> [!warning] 误区 5：混淆 launch-bound 与 memory-bound
> launch-bound：GPU 完全空闲（没 kernel 执行），因 CPU 提交慢。memory-bound：GPU 有 kernel 但等数据。都让 [[GPU utilization]] 低，但解法不同（graph vs 带宽优化）。需 [[nsys]] trace 区分。


## 8. 延伸细节

### 8.1 CUDA Graph 的限制

graph 要求**固定形状**（grid/block/shape 预录时定）。LLM 推理的变长序列（每 token 长度变）需多 graph 或 padding。vLLM 等框架用 graph + padding（序列 pad 到固定长度回放）。

### 8.2 torch.compile 的 fusion

`torch.compile`（原 TorchDynamo/Inductor）自动 fuse elementwise + reduction，生成 Triton kernel。减 launch + 减内存访问。LLM 推理常 `torch.compile(model)` 自动优化。

### 8.3 LLM 推理的 launch 优化栈

- **vLLM**：CUDA Graph + 自定义 fused kernel（attention、MLP）。
- **SGLang**：RadixAttention + graph + fusion。
- **torch.compile**：自动 fusion。
- **Triton**：手写 fused kernel（FlashAttention 用此）。

### 8.4 launch overhead 的测量

```
nsys profile -t cuda ./my_program
# nsys timeline 看 kernel 间的 gap (CPU 提交延迟)
# 大 gap = launch overhead 暴露 = CPU-bound
```

[[nsys]] 的 timeline 直观显示 kernel 间空隙（GPU 空闲等 CPU）。

### 8.5 multi-stream 并发

多 stream 可让 GPU 交错执行不同 stream 的 kernel（一个等内存时换另一个）。但 launch overhead 仍在（每 kernel 仍 5μs 提交）。multi-stream 帮隐藏内存延迟，不解决 launch 瓶颈（launch 瓶颈用 graph/fusion 解）。

### 8.6 CPU 端开销

除 launch，CPU 端还有 Python 开销（PyTorch 的 dispatcher、autograd、Python 调用栈）。LLM 推理的 Python 端每 token 可能花 ms 级（远超 launch）。torch.compile 的 graph capture 把 Python 端也摊掉。生产 LLM 推理必走 compiled/graph 模式，避免 Python+launch 双重开销。

---
相关: [[GPU性能]] | [[CPU-GPU pipeline stall]] | [[SM utilization]] | [[memory bandwidth]] | [[GPU utilization]] | [[torch profiler]] | [[nsys]] | [[torch.compile]] | [[FlashAttention]] | [[continuous batching]]
