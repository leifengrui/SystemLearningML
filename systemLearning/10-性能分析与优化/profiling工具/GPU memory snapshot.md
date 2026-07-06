# GPU memory snapshot

> **所属章节**: [[profiling工具]]
> **所属模块**: [[10-性能分析与优化]]
> **别名**: GPU memory snapshot / 显存快照 / memory snapshot / memory history
> **难度**: 中（需懂显存管理 + caching allocator）


## 1. 一句话定义

**GPU memory snapshot（显存快照）** 是 PyTorch 的**结构化显存占用分析**工具，记录某时刻（或一段时间）GPU 显存里**每个张量的分配**——名称、大小、分配栈、所属段（segment）——并按类型（权重/activation/gradient/optimizer state/buffer/碎片）分类汇总。它是 OOM 调试的核心：当显存超限，snapshot 告诉**谁吃了显存**（权重占多少？activation 占多少？optimizer state 占多少？碎片占多少？），而非只报"OOM"一个错误。与 [[torch profiler]] 的 trace（时间维事件）不同，memory snapshot 是**空间维快照**——看某时刻显存里有什么。它对接 [[activation memory]]/[[gradient memory]]/[[optimizer state memory]]（Ch11 显存系统）的诊断入口。

> [!note] 三句话定位
> - **是什么**：PyTorch 显存结构化快照，记录每张量占用、按类型分类。
> - **为什么**：OOM 需定位谁吃显存（权重/activation/optimizer/碎片），而非只报"OOM"。snapshot 给结构化答案。
> - **与 trace 的区别**：trace 是时间维事件流，snapshot 是空间维占用快照。OOM 调试用 snapshot，性能调试用 trace。


## 2. 为什么需要它（动机与背景）

### 2.1 OOM 的复杂性

训练大模型常遇 OOM（Out Of Memory）。显存被多种占用瓜分：

- **权重**（weight）：模型参数（7B fp16 = 14GB）。
- **activation**：前向中间结果（[[activation memory]]，与 batch/seq/层深相关）。
- **gradient**：反向梯度（[[gradient memory]]，≈权重）。
- **optimizer state**：Adam 的 m/v（[[optimizer state memory]]，2×权重）。
- **buffer**：通信 buffer、CUDA context、workspace。
- **碎片**（fragmentation）：caching allocator 的空闲碎片。

OOM 时只报"CUDA out of memory"，不知是哪类超了。是 activation 太大（减 batch/用 [[gradient checkpointing]]）？optimizer state（用 ZeRO）？权重（用 FSDP）？碎片（调 allocator）？需 snapshot 定位。

### 2.2 caching allocator 的复杂性

PyTorch 不直接用 `cudaMalloc`，而用 **caching allocator**：维护显存池（segment），分配/释放复用池内块，避免频繁 `cudaMalloc`（慢）。这带来：

- **segment**：大块显存（如 1-100MB），allocator 向 CUDA 申请的单元。
- **block**：segment 内分给张量的小块。
- **碎片**：segment 内释放后的空闲 block，可能太小不够新分配 → 显存"有但用不了"。

OOM 可能是碎片（总空闲够但无连续大块），不是真不够。snapshot 显示 segment/block 布局，诊断碎片。

### 2.3 snapshot 的结构化视图

`torch.cuda.memory_snapshot()` 返回当前 allocator 状态：

- 所有 segment 及其 block（size、allocated/free）。
- 每分配张量的 stack trace（哪行代码分配的）。

`torch.cuda.memory._record_memory_history()` 录制一段时间的分配历史，可导出可视化（火焰图）。这是 OOM 调试的利器。

### 2.4 与 trace 的分工

- **[[torch profiler]]/[[nsys]] trace**：时间维，看 kernel/算子/通信事件流。性能调优用。
- **memory snapshot**：空间维，看显存占用分布。OOM 调试用。

OOM 问题时先 snapshot 定位显存分布，再决定优化方向（[[gradient checkpointing]]/ZeRO/FSDP）。


## 3. 核心概念详解

### 3.1 memory snapshot API

```python
# 某时刻快照
snapshot = torch.cuda.memory_snapshot()
# 含 segments, 每 segment 的 blocks (allocated/free, size)

# 录制分配历史 (带 stack)
torch.cuda.memory._record_memory_history(max_entries=100000)
# ... 跑可能 OOM 的代码 ...
torch.cuda.memory._record_memory_history(enabled=None)
# 导出快照 (可视化)
torch.cuda.memory._dump_snapshot('snapshot.pickle')
```

### 3.2 segment 与 block

- **segment**：allocator 向 CUDA 申请的大块（1-100MB）。segment 不还给 CUDA（除非 `empty_cache`）。
- **block**：segment 内分给张量的小块。释放的 block 空闲，可复用。
- **碎片**：segment 内空闲 block 太小、不够新分配 → 该 segment 不能让出 → 显存"占用但空闲"。

snapshot 显示每 segment 的 block 布局，诊断碎片。

### 3.3 按类型分类

snapshot 的分配带 stack trace，可按调用源分类：

- 来自 `nn.Parameter` 初始化 → 权重。
- 来自 forward 的中间张量 → activation。
- 来自 backward 的梯度 → gradient。
- 来自 optimizer.step → optimizer state。
- 来自 all-reduce buffer → 通信 buffer。

汇总各类大小，看哪类占比异常。

### 3.4 碎片诊断

碎片率：

$$
\text{fragmentation} = \frac{\text{free but unusable}}{\text{total free}}
$$

snapshot 的 segment 视图直接看：某 segment 有多个小空闲 block（碎片）vs 大连续空闲（可用）。碎片高时 `torch.cuda.empty_cache()` 还 segment 给 CUDA，缓解。

### 3.5 峰值 vs 当前

- **current memory**：当前占用（snapshot 时刻）。
- **peak memory**：历史峰值（`torch.cuda.max_memory_allocated()`）。

OOM 常是峰值时刻超限（如 forward 到某层 activation 峰值）。snapshot 录制历史能定位峰值时刻哪个张量超。


## 4. 数学原理 / 公式

### 4.1 显存占用分解

总显存 $M_{\text{total}}$：

$$
M_{\text{total}} = M_{\text{weight}} + M_{\text{activation}} + M_{\text{gradient}} + M_{\text{optimizer}} + M_{\text{buffer}} + M_{\text{fragment}}
$$

snapshot 把各项分解，定位哪项超。

### 4.2 各项估算

- $M_{\text{weight}} = P \cdot b$（参数数 × 字节，7B fp16 = 14GB）。
- $M_{\text{gradient}} \approx M_{\text{weight}}$（fp16）或 $2M_{\text{weight}}$（fp32 梯度）。
- $M_{\text{optimizer}} = 2P \cdot b_{\text{state}}$（Adam 的 m+v，fp32 = 56GB for 7B）。
- $M_{\text{activation}} \approx L \cdot B \cdot S \cdot d \cdot \text{layers}$（层深 × batch × seq × dim，[[activation memory]]）。

详见 [[activation memory]]/[[gradient memory]]/[[optimizer state memory]]（Ch11）。

### 4.3 碎片率

$$
F = \frac{\sum_{\text{segments}} \text{free blocks too small}}{\sum_{\text{segments}} \text{total free in segment}}
$$

$F$ 高 → 显存有但用不了 → 需 `empty_cache` 或调 allocator（`PYTORCH_CUDA_ALLOC_CONF`）。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 memory snapshot

```python
class MemorySnapshot:
    """模拟 PyTorch caching allocator 的显存快照."""
    def __init__(self): self.allocations=[]
    def allocate(self, name, size):
        self.allocations.append({'name':name, 'size':size})
    def snapshot(self):
        total = sum(a['size'] for a in self.allocations)
        by_type = {}
        for a in self.allocations:
            t = a['name'].split('_')[0]      # 按前缀分类 (weight/activation/...)
            by_type[t] = by_type.get(t, 0) + a['size']
        return {'total':total, 'by_type':by_type, 'n_allocs':len(self.allocations)}

s = MemorySnapshot()
s.allocate('weight_attention', 1000)    # 权重
s.allocate('weight_mlp', 2000)
s.allocate('activation_layer1', 500)    # activation
s.allocate('activation_layer2', 300)
s.allocate('optimizer_mlp', 4000)       # optimizer state
snap = s.snapshot()
print(f'总显存: {snap["total"]} MB, {snap["n_allocs"]} 个分配')
print('按类型:')
for t, sz in sorted(snap['by_type'].items(), key=lambda x:-x[1]):
    print(f'  {t:12s} {sz} MB ({sz/snap["total"]:.0%})')
```

### 5.2 真实 PyTorch API 对照

```python
# import torch
# torch.cuda.memory._record_memory_history(max_entries=100000)
# model(input); loss.backward(); optimizer.step()   # 跑可能 OOM 的
# torch.cuda.memory._record_memory_history(enabled=None)
# torch.cuda.memory._dump_snapshot('snap.pickle')
# # 上传到 pytorch.org/memory_viz 可视化 (火焰图)
# # 或即时快照:
# print(torch.cuda.memory_snapshot())   # segments + blocks
# print(f'peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: PyTorch caching allocator、CUDA 显存管理。
- **下游（应用）**: [[activation memory]]/[[gradient memory]]/[[optimizer state memory]]（Ch11，snapshot 分解的类别）、[[gradient checkpointing]]/[[ZeRO (DeepSpeed)]]/[[offloading (CPU-NVMe)]]（Ch11，优化技术，snapshot 验证效果）、[[torch profiler]]（profile_memory 采内存事件）、OOM 调试、[[KV cache management]]（推理显存）。
- **对比 / 易混**:
  - **snapshot vs [[torch profiler]] trace**：snapshot 是空间维显存快照（OOM 调试），trace 是时间维事件流（性能调试）。不同问题用不同工具。
  - **snapshot vs `max_memory_allocated`**：后者只给峰值数字，snapshot 给结构化分解（哪类占多少）。snapshot 是"为什么"，数字是"多少"。
  - **current vs peak memory**：current 是当前占用，peak 是历史峰值。OOM 常是 peak，snapshot 录历史能定位峰值时刻。


## 7. 常见误区与易错点

> [!warning] 误区 1：OOM 时不看 snapshot 瞎调
> OOM 需先 snapshot 定位是哪类显存超（权重/activation/optimizer/碎片），再对症下药。瞎调可能优化错项（如 activation 超却去减权重）。snapshot 是 OOM 调试的第一步。

> [!warning] 误区 2：忽略碎片
> 显存"有但用不了"（碎片）也会 OOM。snapshot 的 segment 视图看碎片率，`empty_cache` 或调 `PYTORCH_CUDA_ALLOC_CONF` 缓解。不查碎片可能误判显存真不够。

> [!warning] 误区 3：snapshot = trace
> snapshot 是某时刻显存占用快照（空间维），trace 是事件时间流（时间维）。OOM 用 snapshot，性能用 trace。不要用 trace 调 OOM（信息不对）。

> [!warning] 误区 4：只看 current 不看 peak
> OOM 常发生在峰值时刻（forward 到某层 activation 峰值），当前占用可能不高。需录 memory history 找峰值时刻，或看 `max_memory_allocated`。

> [!warning] 误区 5：snapshot 开着跑全程
> `_record_memory_history` 有开销（采每分配 + stack）。只在调 OOM 时短时开，调完关。全程开会拖慢且文件巨大。


## 8. 延伸细节

### 8.1 caching allocator 的工作

PyTorch 的 caching allocator：

- 不直接 `cudaMalloc`（慢），维护 segment 池。
- 分配：从池找合适 block，没有则向 CUDA 申请新 segment。
- 释放：block 标记空闲，留在池（不还给 CUDA），复用。
- `empty_cache()`：把空闲 segment 还给 CUDA（缓解碎片，但下次分配又需申请新 segment，可能慢）。

理解这有助于解释 snapshot 的 segment/block 布局。

### 8.2 PYTORCH_CUDA_ALLOC_CONF

环境变量调 allocator：

- `max_split_size_mb`：限制 block 分裂，控碎片。
- `roundup_power2_divisions`：round-up 策略，控碎片。
- `expandable_segments:True`：动态扩 segment，减碎片。

碎片严重的训练调这些参数。

### 8.3 memory_viz 可视化

`torch.cuda.memory._dump_snapshot` 导出的 `.pickle` 可上传 [pytorch.org/memory_viz](https://pytorch.org/memory_viz) 生成火焰图——每段显存按分配栈展开，直观看哪类占用。是 OOM 调试的标准可视化。

### 8.4 LLM 训练的显存分布

7B 训练（Adam fp32 state）典型分布：

- weight: 14GB（fp16）
- gradient: 14GB（fp16）
- optimizer: 56GB（fp32 m+v）= 最大头
- activation: 数 GB-数十 GB（看 batch/seq/[[gradient checkpointing]]）

snapshot 验证：optimizer 占大头 → 用 [[ZeRO (DeepSpeed)]]/FSDP 分片；activation 大 → 用 [[gradient checkpointing]]。这是 Ch11 显存系统的诊断入口。

### 8.5 推理的显存

LLM 推理显存：

- weight: 14GB（7B fp16）
- KV cache: 随 batch/seq 增长（[[KV cache management]]）
- activation: 小（无反向）

snapshot 看 KV cache 占比，验证 [[continuous batching]]/[[prefix caching]] 的显存优化。

### 8.6 与 distributed 的交互

DDP/FSDP 下显存分布变（FSDP 分片权重/optimizer）。snapshot 在每 rank 独立看，需结合分布式策略解释。详见 [[Fully Sharded Data Parallel]]/[[ZeRO (DeepSpeed)]]。

---
相关: [[profiling工具]] | [[torch profiler]] | [[nsys (Nsight Systems)]] | [[activation memory]] | [[gradient memory]] | [[optimizer state memory]] | [[gradient checkpointing]] | [[ZeRO (DeepSpeed)]] | [[offloading (CPU-NVMe)]] | [[KV cache management]] | [[continuous batching]] | [[Fully Sharded Data Parallel]]
