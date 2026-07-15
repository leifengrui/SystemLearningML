# CUDA Graph与graph capture

> **所属章节**: [[CUDA Graph]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: CUDA Graph / graph capture / graph replay / CUDA 图捕获
> **难度**: 中高（需懂 [[CUDA stream与event]] + [[kernel launch overhead]]）


## 1. 一句话定义

**CUDA Graph** 是把一串 CUDA 操作（kernel launch / memcpy）**捕获成一张可重放的图**（有向图，节点 = 操作，边 = 依赖），后续每次执行只调一次 `cudaGraphLaunch` replay 整图，省去每步重新解析参数、提交 launch、走 driver 的开销——本质是把**多次 launch 的固定开销**摊到捕获时一次付，replay 只走 GPU 端的轻路径。它是消除 [[kernel launch overhead]] 的最强手段，尤其统治 LLM 推理的 decode 阶段（每 token 数十个小 kernel，shape 固定，replay 收益巨大），vLLM/SGLang 的 decode 路径都基于 CUDA Graph。

> [!note] 三句话定位
> - **是什么**：捕获 stream 序列为图，replay 时一次 launch 执行全图，省逐次 launch 开销。
> - **为什么**：小 kernel 的耗时被 launch 开销（几 μs/次）主导，decode 每几十个 kernel 累积成 CPU 瓶颈，graph 把它压到一次。
> - **限制**：graph 要求**操作参数/shape 固定**（捕获时锁死），动态 shape / 控制流要重新捕获或多图池。


## 2. 为什么需要它（动机与背景）

### 2.1 launch overhead 在小 kernel 上成主导

[[kernel launch overhead]] 每次 launch ~2-5 μs（CPU 解析参数、走 runtime/driver、提交到 GPU 命令队列）。一个 elementwise kernel 算 1k 元素可能 GPU 端只跑 1 μs，但 launch 5 μs → 80% 时间是 launch 浪费。

LLM decode 每生成一个 token 要跑：attention、RMSNorm、RoPE、若干 GEMM、采样……几十个小 kernel。若每 kernel 5 μs launch × 30 kernel = 150 μs launch 开销/token，远超 GPU 真实计算（几十 μs）。**CPU launch 成 decode 的吞吐瓶颈**——这是 vLLM 早期 decode 慢的根因。

### 2.2 graph 把 launch 摊到捕获时

CUDA Graph 的思路：**第一次走 stream 时把整条序列捕获成图**（记录每个 kernel 的参数、依赖）。之后每步只 `cudaGraphLaunch`，driver 直接把整图丢给 GPU，**省去逐 kernel 的 CPU 解析与提交**。replay 开销接近一次 launch（几 μs），无论图里有几个 kernel。

故 decode 30 个 kernel 的 launch 从 150 μs 压到 ~5 μs，吞吐提几十倍。这是 LLM 推理框架 decode 路径的核心加速。

### 2.3 静态 shape 是前提

graph 捕获时**锁死每个 kernel 的参数与 tensor 地址**。replay 要求后续执行用同样的 shape、同样地址的 buffer。LLM decode 的 shape 恰好固定（batch × 1 × hidden），所以 graph 友好。但 prefill 长度变化、动态 batch、控制流分支 → 要重新捕获或维护多图池（见 [[请求生命周期]] 的 graph 池）。


## 3. 核心概念详解

### 3.1 两种捕获方式

**Stream-based capture**（最常用）：
```c
cudaStreamBeginCapture(stream, mode);       // 开始捕获
kernelA<<<...>>>(...);                        // 这些操作被录进图
kernelB<<<...>>>(...);
cudaMemcpyAsync(..., stream);
cudaStreamEndCapture(stream, &graph);        // 结束, 得到 graph
cudaGraphInstantiate(&exec, graph, ...);     // 实例化成可执行图
cudaGraphLaunch(exec, stream);                // replay
```

**Explicit construction**：手动 `cudaGraphCreate` + `cudaGraphAddKernelNode` 逐节点加。灵活但繁琐，少用。

### 3.2 graph 的元素

- **node**：kernel node / memcpy node / host node（CPU 回调）/ event node。
- **edge**：依赖关系（A 必须在 B 前完成）。capture 时由 stream 序隐式得；explicit 时手动 `cudaGraphAddDependencies`。
- **instantiate**：把 graph 编译成可执行实例（`cudaGraphInstantiate`），driver 内部做依赖分析、资源分配。一次实例化可多次 launch。
- **launch**：`cudaGraphLaunch(exec, stream)` 把实例丢给 GPU，单次开销几 μs。

### 3.3 graph 与 stream 的关系

graph 是从 stream 捕获来的（capture 模式），replay 时也提交到 stream。但 stream 内的多操作被"打包"成一次提交。**graph 不是 stream 的替代，是 stream 序列的压缩快照**。详见 [[CUDA stream与event]]。

### 3.4 动态参数：graph update

`cudaGraphExecKernelNodeSetParams` 可在 replay 前改某 node 的参数（如改指针/shape），避免重新 capture。但结构性变化（增删节点、改依赖）需重新 capture。LLM 的"换 batch 大小"通常直接维护多个 graph 实例的池（按 batch size bucket），不 update。

### 3.5 LLM 推理的 graph 池

vLLM/SGLang 的 decode 路径：

1. 每个 batch size bucket 预捕获一个 graph（shape 固定）。
2. decode 时按当前 batch 选对应 graph replay。
3. batch 变化（新请求加入、某请求完成）切到对应 bucket 的 graph。
4. prefill 通常不用 graph（长度变化），或对常见长度预捕获几个 graph。

这是 [[请求生命周期]] 与 [[CUDA Graph执行]] 的核心。

### 3.6 torch.cuda.graphs

```python
g = torch.cuda.CUDAGraph()
static_input = torch.randn(B, H, device='cuda')
torch.cuda.graph(g, pool=...)      # 捕获上下文
out = model(static_input)         # 录入图
torch.cuda.graph(g)               # 结束
# replay
static_input.copy_(real_input)   # 改输入 (地址不变)
g.replay()
result = out                     # 复用输出 buffer
```

PyTorch 的 `torch.cuda.make_graphed_callables` 把 nn.Module 包成 graph-callable，自动处理静态 buffer。torch.compile 也常与 graph capture 结合（[[torch.compile]] / [[Inductor]] 的 cudagraphs 模式）。

### 3.7 静态 buffer 约束

graph replay 时 input/output 必须用**捕获时同地址的 buffer**。不能 replay 后直接拿新 tensor——要用 `static_input.copy_(real)` 把真实数据拷进静态 buffer，再 replay，再从静态 out buffer 读。这要求框架用静态池管理 input/output。


## 4. 数学原理 / 公式

### 4.1 launch 开销对比

设每 kernel launch 开销 $L$，kernel 真实计算 $C_i$，共 $n$ 个 kernel：

$$
T_{\text{stream}} = \sum_{i=1}^n (L + C_i) = nL + \sum_i C_i
$$

$$
T_{\text{graph}} = L_{\text{replay}} + \sum_i C_i \approx L + \sum_i C_i
$$

收益 $\frac{nL + \sum C_i}{L + \sum C_i}$。当 $\sum C_i \ll nL$（小 kernel 串）时收益接近 $n$；当 $\sum C_i \gg nL$（大 GEMM）收益接近 1（launch 不重要）。

### 4.2 收益的 kernel 维度判据

graph 收益大当：$C_i$ 小（elementwise/norm/小 GEMM）、$n$ 大（多小 kernel 串）、shape 固定。收益小当：$C_i$ 大（训练的大 GEMM）、$n$ 小、shape 动态。

### 4.3 graph 池内存代价

每 graph 实例占一份静态 buffer。$k$ 个 bucket × 各 bucket buffer = 总静态显存。bucket 太多吃显存，太少覆盖不够。是 [[CUDA allocator与memory pool]] 的 trade-off。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟 launch 开销对比

```python
def time_stream_vs_graph(n_kernels, launch_us=3, compute_us=1):
    """n 个小 kernel: stream 逐次 launch vs graph replay."""
    stream = n_kernels * (launch_us + compute_us)
    graph = launch_us + n_kernels * compute_us      # replay 一次 launch
    return stream, graph, stream / graph

# decode 30 个小 kernel
s, g, gain = time_stream_vs_graph(30)
print(f"stream {s}us, graph {g}us, 加速 {gain:.1f}x")
# 大 GEMM 训练 (5 个大 kernel, compute=1000us)
s, g, gain = time_stream_vs_graph(5, compute_us=1000)
print(f"训练 stream {s}us, graph {g}us, 加速 {gain:.2f}x (收益小)")
```

### 5.2 PyTorch graph capture + replay

```python
import torch
B, H = 8, 4096
static_in = torch.randn(B, H, device='cuda')
static_out = [None]

g = torch.cuda.CUDAGraph()
# warmup (alloc/setup 不能在 capture 内)
for _ in range(3):
    y = static_in @ static_in.t()
torch.cuda.synchronize()

with torch.cuda.graph(g):
    static_out[0] = static_in @ static_in.t()      # 录入图

# replay: 改输入 (地址不变) 再 replay
for step in range(5):
    static_in.copy_(torch.randn(B, H, device='cuda'))
    g.replay()
    print(static_out[0].sum().item())              # 从静态 out 读
```

### 5.3 真实命令对照

```python
# nsys trace 看 graph replay 节点 (替代满屏小 kernel)
# nsys profile -t cuda,nvtx python decode.py
# vLLM: --enforce-eager=False 启用 cudagraph (默认 True for decode)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[CUDA stream与event]]（捕获源/执行载体）、[[kernel launch overhead]]（graph 要消除的对象）、[[GPU执行模型]]（launch 链路）。
- **下游（应用）**: [[推理 CUDA Graph]]（vLLM/SGLang decode 路径核心）、[[torch.compile]]/[[Inductor]]（cudagraphs 模式）、[[请求生命周期]]（graph 池管理）、[[continuous batching的调度实现]]（decode batch 切换 graph）、[[model runner]]（runner 调度 graph）。
- **对比 / 易混**:
  - **CUDA Graph vs stream**：stream 是命令队列（每次 launch 重新提交），graph 是 stream 序列的压缩快照（replay 一次提交）。graph 从 stream 捕获。
  - **CUDA Graph vs torch.compile**：graph 只压缩 launch，不改 kernel；torch.compile 重写/融合 kernel。常组合：compile 生成高效 kernel + graph 压 launch。
  - **graph capture vs explicit construction**：capture 从 stream 录（声明式，常用）；explicit 手动建（灵活但繁琐）。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 graph 能加速所有场景
> graph 只省 launch 开销。大 GEMM（训练的矩阵乘，compute 远大于 launch）几乎无收益。它统治小 kernel 串（decode、elementwise、norm），训练主 GEMM 不靠它。

> [!warning] 误区 2：捕获时含动态分配/同步
> graph capture 内不能有 CPU 同步（`cudaDeviceSynchronize`）、动态内存分配（除非用 graph-aware allocator）、条件分支。这些会 break capture。PyTorch 的 capture 需先 warmup 跑几步让 allocator 稳定。

> [!warning] 误区 3：replay 后直接读新 tensor
> graph replay 的 output 落在**捕获时同地址的静态 buffer**。新分配的 tensor 拿不到结果。要用 `static_in.copy_(real)` 改输入、`static_out` 读输出。框架要管静态池。

> [!warning] 误区 4：shape 变了不重新捕获
> graph 锁死 shape。decode 的 batch 变了（新请求加入）仍 replay 旧 graph → 越界或空算。必须切到对应 bucket 的 graph，或重新 capture。

> [!warning] 误区 5：忽视 graph 池显存代价
> 每 bucket 一个 graph 实例 + 静态 buffer，bucket 多显存吃紧。vLLM 默认 cudagraph bucket 数有限，过大 batch 走 eager fallback。

> [!warning] 误区 6：与 eager 混用忘了同步
> graph replay 异步，若马上接 eager 操作读 output，需 stream 序或显式同步，否则 race。


## 8. 延伸细节

### 8.1 stream capture 模式

- `cudaStreamCaptureModeGlobal`：全局捕获，capture 期间任何其他 stream 的操作会被忽略或警告。
- `cudaStreamCaptureModeThreadLocal`：仅本线程。
- `cudaStreamCaptureModeRelaxed`：允许部分跨流操作。

LLM 框架常用 thread-local 或 relaxed，避免与其他流（如 NCCL）冲突。

### 8.2 conditional graph

CUDA 12+ 支持 conditional node（`cudaGraphAddConditionalNode`），图内可按条件分支（如"if logit > threshold 走 A 路径 else B"），部分打破 graph 静态约束。用于 speculative decoding 的接受/拒绝分支。

### 8.3 graph 与 cudnn/cublas

cuDNN/cuBLAS 内部对固定 shape 的 op 也用 graph-like 思想缓存 plan。手动 graph capture 可包住这些库调用进一步压 launch。

### 8.4 vLLM/SGLang 的 graph 实践

- vLLM：`--enforce-eager=False` 启用 cudagraph，按 batch size bucket 预捕获，decode 复用。prefill 多变长通常 eager 或选少量 bucket。
- SGLang：类似，且与 [[Radix Tree prefix cache]] 配合，prefix 命中走 graph，miss 走重算路径。
- 两者的 graph 池管理与 [[block manager]] 的 KV 分配协同。

### 8.5 内容来源

CUDA Graph 语义整理自 NVIDIA CUDA C++ Programming Guide 的 Graphs 章节，PyTorch 包装见 `torch.cuda.graphs` 文档，vLLM/SGLang 实践见各自推理引擎源码章。

---
相关: [[CUDA Graph]] | [[CUDA stream与event]] | [[kernel launch overhead]] | [[推理 CUDA Graph]] | [[torch.compile]] | [[Inductor]] | [[请求生命周期]] | [[continuous batching的调度实现]] | [[model runner]] | [[CUDA allocator与memory pool]]
