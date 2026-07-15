# CUDA allocator与memory pool

> **所属章节**: [[CUDA显存管理]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: CUDACachingAllocator / 显存池 / memory pool / 显存碎片 / fragmentation / OOM / graph-aware allocator
> **难度**: 中（需懂 [[GPU内存层级]] + PyTorch 内存模型）


## 1. 一句话定义

**CUDA allocator 与 memory pool** 是 GPU 显存（HBM）的分配/释放管理机制——原生 `cudaMalloc/cudaFree` 是昂贵的同步调用（每次几十 μs、且会同步整 device），故 PyTorch 等框架用 **CUDACachingAllocator**：申请一块大显存做**池**，把"释放"的块挂回池（不真的 `cudaFree`），下次同尺寸请求直接复用，**把分配/释放降到 μs 级且不同步**。管理失效会导致**显存碎片（fragmentation）**——总显存够但分不出一块连续的大块 → 假 OOM；以及 [[CUDA Graph与graph capture]] 捕获时的地址漂移（graph-aware allocator 要保证 replay 时地址不变）。它是训练/推理稳定性的隐藏杀手，OOM 排查的第一站。

> [!note] 三句话定位
> - **是什么**：caching allocator 维护显存池，free 进池不真释放，malloc 优先复用池内块，绕开昂贵的 `cudaMalloc`。
> - **为什么**：原生 `cudaMalloc` 同步且慢（每次同步 device + 几十 μs），训练每步上百次分配/释放会卡死；池化让分配近乎免费。
> - **两大坑**：碎片（总够但连续不够 → 假 OOM）、graph 捕获时地址漂移（replay 错位 → 需 graph-aware allocator）。


## 2. 为什么需要它（动机与背景）

### 2.1 原生 cudaMalloc 的代价

`cudaMalloc` 是**同步调用**——它会 `cudaDeviceSynchronize` 整个 device（等所有 stream 完成），再做分配。一次分配几十 μs 起，且 serialize 所有 GPU 工作。训练每 step 上百次 forward op，每 op 分配几个临时 tensor → 几百次 `cudaMalloc`/`cudaFree` → 直接卡死。

### 2.2 池化思路

CUDACachingAllocator：

1. 首次 `malloc` 真调 `cudaMalloc` 拿一块大显存（>= 请求）。
2. 用完 `free` 时**不调 `cudaFree`**，把这块挂回池（标记 free）。
3. 下次同尺寸请求，从池里找匹配的 free 块直接给（零 `cudaMalloc`、零同步）。
4. 块比请求大时 split，比相邻 free 块小时 coalesce。

这样分配/释放降到 μs 级、不同步 device。训练 hot loop 几乎不触 `cudaMalloc`。

### 2.3 碎片问题

池里很多 free 块，但**地址不连续**：比如池里 100 个 1GB free 块（总 100GB），但请求一个 2GB 连续块 → 拿不出 → 报 OOM（即使"总 free 100GB ≥ 2GB"）。这是**外部碎片**。LLM 的 KV cache、动态 batch、graph 多 bucket 容易制造碎片。这是 OOM 排查最难的一类。

### 2.4 graph 的地址不变约束

[[CUDA Graph与graph capture]] replay 时要求 input/output 用**捕获时同地址的 buffer**。若 allocator 在 capture 期间给某块换地址（因 split/coalesce/重分配），replay 就错位。故 graph 模式需 **graph-aware allocator**（capture 时冻结地址、replay 时复用）。


## 3. 核心概念详解

### 3.1 块的生命周期与 split/coalesce

allocator 把 HBM 切成 block：

- **allocated**：被某个 tensor 占用。
- **free - cached**：被 free 但挂回池（不归 OS）。
- **free - large**：大块 free，可被 split 成小快满足请求。

split：请求 1GB，池里只有 free 2GB 块 → 切成 1GB+1GB，给 1GB、留 1GB free。
coalesce：相邻两 free 块合并成大块（减少碎片）。

### 3.2 内存统计指标

PyTorch 的 `torch.cuda.memory_allocated()` / `memory_reserved()` / `max_memory_allocated()`：

| 指标 | 含义 |
|---|---|
| allocated | 当前被 tensor 实际占用（业务视角） |
| reserved | allocator 从 `cudaMalloc` 拿的总池（含 free cached） |
| max_allocated | 历史峰值 allocated（峰值显存，规划用） |
| max_reserved | 历史峰值 reserved |

reserved - allocated = 池里 cached 但未用的碎片块。[[GPU memory snapshot]] 可视化这些块。

### 3.3 外部 vs 内部碎片

- **外部碎片**：池里 free 块总量够，但不连续 → 拿不出大块。常见。
- **内部碎片**：分配的块比请求大（split 后余下太小用不上）。allocator 的 bin 策略影响。

### 3.4 碎片化场景

- **动态 batch**：每步不同 batch size → 不同尺寸 activation tensor → 池里各种尺寸 free 块 → 碎片。
- **KV cache + model 混合**：KV cache 占固定块，model activation 占另一些，二者交错 → 碎片。
- **graph 多 bucket**：每 bucket 一套静态 buffer，互相交错 → 碎片。
- **forward/backward 不同 shape**（gradient checkpointing 重算）→ 碎片。

### 3.5 OOM 的真因与诊断

OOM 报"out of memory"可能是：

1. **真不够**：reserved ≈ 总 HBM，allocated 也接近峰值 → 物理不够，减 batch/模型。
2. **碎片**：reserved 远 < 总 HBM 但拿不出连续块 → 碎片，需 `empty_cache` 或调 allocator 策略。
3. **泄漏**：allocated 持续涨不回落 → tensor 没释放（参考环、hook 持有、graph static buffer 没复用）。

诊断顺序：看 `memory_allocated/reserved`、snapshot 找大块、grep 参考环。

### 3.6 PyTorch API

```python
torch.cuda.empty_cache()            # 把 cached free 块真 cudaFree 回 OS (降 reserved, 不降 allocated)
torch.cuda.memory_allocated()       # 当前 allocated
torch.cuda.memory_reserved()        # 当前 reserved
torch.cuda.memory_summary()        # 详细块状态
torch.cuda.memory._dump_snapshot() # 导出 snapshot 给 nvis ues (见 GPU memory snapshot)
```

`PYTORCH_CUDA_ALLOC_CONF` 环境变量调碎片策略：

- `expandable_segments:True`：用可扩展 segment 减碎片（新版）。
- `roundup_power2_quantum`：分配 round 到 2 的幂（减 split）。
- `max_split_size_mb`：限制 split，防大块被切碎。

### 3.7 graph-aware allocator

PyTorch 的 `torch.cuda.graph_pool_handle()` 给 graph 模式一个私有池，capture 与 replay 用同池同地址。vLLM/SGLang 自管 graph 的静态 buffer 池（[[推理 CUDA Graph]]）。

### 3.8 与框架的协作

- PyTorch：CUDACachingAllocator（默认）。
- vLLM：自管 KV cache 块池（[[PagedAttention]] 的 block manager）+ 模型 weight/static buffer。
- Megatron：分布式下每 rank 独立 allocator + NCCL buffer 单独管理。
- allocator 与 [[state_dict分片]] 的 checkpoint 交互：存盘时最好走 pinned + async（见 [[异步memcpy与pinned memory]]）。


## 4. 数学原理 / 公式

### 4.1 碎片的衡量

$$
\text{碎片率} = 1 - \frac{\text{最大可分配连续块}}{\text{总 free}}
$$

碎片率高：总 free 大但分不出大块。理想 0，实际 0.2-0.5 常见。

### 4.2 OOM 判据

$$
\text{OOM} \iff \text{request} > \text{max contiguous free}
$$

不依赖总 free。故"总 free 100GB 但 max contiguous 1GB" → 请求 2GB OOM。

### 4.3 池的收益

原生每次 malloc/free $L$ μs（含同步）× $n$ 次/step = $nL$ μs/step。池化后几乎 0（cache 命中）。100 次/step × 30 μs = 3 ms/step → 池化省 3 ms/step，训练百万 step 节省巨大。

### 4.4 caching 的 LRU 代价

池无限增长会吃光显存。allocator 有 LRU/limit 策略，超阈值触发真 `cudaFree` 回 OS。`empty_cache` 是手动 LRU flush。

### 4.5 graph 池的地址约束

设 graph $i$ 的 buffer 地址集合 $A_i$。replay 时要求 input/output 仍在 $A_i$。故 graph 池需"地址固定"——allocator 不能把 $A_i$ 里的地址 split 给别的请求。graph-aware allocator 在 capture 期间锁定这些地址。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟 caching allocator 与碎片

```python
class SimpleCachingAllocator:
    """简化 caching allocator: free 进池, malloc 优先复用."""
    def __init__(self, total):
        self.free_blocks = [(0, total)]   # (start, size)
        self.allocated = {}
    def malloc(self, size):
        for i, (start, bsize) in enumerate(self.free_blocks):
            if bsize >= size:
                self.free_blocks[i] = (start+size, bsize-size)
                self.allocated[start] = size
                return start
        raise MemoryError(f"OOM: 无法分配 {size}, 碎片 max_contig={max(b for _,b in self.free_blocks)}")
    def free(self, addr):
        size = self.allocated.pop(addr)
        self.free_blocks.append((addr, size))

a = SimpleCachingAllocator(1000)
p1, p2, p3 = a.malloc(300), a.malloc(300), a.malloc(300)   # 用 900
a.free(p1); a.free(p3)                                      # free 300 + 300, 但中间 p2 占着
# 总 free 600, 但 max contig 300 (两块被 p2 隔开) -> 碎片
print(f"总 free={sum(b for _,b in a.free_blocks)}, max_contig={max(b for _,b in a.free_blocks)}")
try: a.malloc(500)   # OOM 碎片
except MemoryError as e: print("OOM:", e)
```

### 5.2 PyTorch 内存统计

```python
import torch
x = torch.randn(1000, 1000, device='cuda')
print(f"allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")
print(f"reserved:  {torch.cuda.memory_reserved()/1e6:.1f} MB")
del x
print(f"free 后 allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB (cached 仍在池)")
print(f"free 后 reserved:  {torch.cuda.memory_reserved()/1e6:.1f} MB (不降)")
torch.cuda.empty_cache()
print(f"empty_cache 后 reserved: {torch.cuda.memory_reserved()/1e6:.1f} MB (降)")
```

### 5.3 碎片调优 env

```bash
# 减碎片 (新 PyTorch)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 限制 split 防大块被切碎
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GPU内存层级]]（HBM 是被分配的资源）、CUDA driver（`cudaMalloc` 底层）。
- **下游（应用）**: [[GPU memory snapshot]]（可视化块）、[[gradient checkpointing]]（重算减峰值 allocated）、[[ZeRO (DeepSpeed)]]/[[FSDP2]]（分片减单卡 allocated）、[[CUDA Graph与graph capture]]（graph-aware allocator 锁地址）、[[PagedAttention]]（KV cache 自管块池）、[[推理 CUDA Graph]]（静态 buffer 池）、[[activation memory]]/[[gradient memory]]/[[optimizer state memory]]（业务侧显存分类）。
- **对比 / 易混**:
  - **allocated vs reserved**：allocated 是业务实际占用，reserved 是 allocator 拿的总池（含 cached free）。OOM 看哪个分情况。
  - **CUDACachingAllocator vs 系统 malloc**：前者管 GPU 显存 + 池化不真 free；后者管 CPU 内存（也有 glibc malloc 池，但机制不同）。
  - **empty_cache vs reset**：empty_cache 把 cached free 真 cudaFree 回 OS（降 reserved，不降 allocated）；reset 彻底销毁 allocator。


## 7. 常见误区与易错点

> [!warning] 误区 1：OOM 一定是显存不够
> 不对。碎片 OOM 时总 free 充足但分不出连续块。看 reserved vs 总 HBM、看 memory_summary 的 max contiguous free，区分"真不够"与"碎片"。

> [!warning] 误区 2：调小 batch 一定救 OOM
> 减 batch 降 allocated，但若碎片是主因（reserved 仍高），减 batch 不解决碎片。需 `empty_cache` + 调 `expandable_segments`。

> [!warning] 误区 3：以为 empty_cache 释放了被引用的 tensor
> `empty_cache` 只回**已 free 的 cached 块**回 OS。被 tensor 引用的块（allocated）不动。它降 reserved，不降 allocated。

> [!warning] 误区 4：graph 模式不冻结地址
> 普通 allocator 在 graph capture 期间可能给某块换地址（split/coalesce），replay 错位。必须用 graph-aware 池（`torch.cuda.graph_pool_handle`）锁地址。

> [!warning] 误区 5：忽视 reserved 远大于 allocated
> reserved - allocated 大 = 池里 cached free 多。长期不释放可能是 fragmentation 或 tensor 泄漏（reference cycle / hook 持有）。用 snapshot 找大块、grep 参考环。

> [!warning] 误区 6：多进程共享 GPU 不限池
> 多进程同卡各自 caching allocator 抢显存，互相 cache 飙升 → 互相 OOM。生产用 MIG 或 `max_split_size` + limit。


## 8. 延伸细节

### 8.1 expandable_segments（PyTorch 2.0+）

传统 allocator 一次性 `cudaMalloc` 大块易碎片。expandable_segments 用多个可扩展 segment，按需 extend，减碎片、提 reserved 利用率。LLM 训练强烈推荐开。

### 8.2 memory snapshot 与可视化

`torch.cuda.memory._dump_snapshot('snap.pickle')` 导出，Nsight Systems/PyTorch snapshot viewer 看每块的分配栈、大小、生命周期。是碎片/泄漏排查的利器。见 [[GPU memory snapshot]]。

### 8.3 vLLM 的块池

vLLM 不依赖 PyTorch allocator 管 KV cache，自管 block pool（[[PagedAttention]]），把 KV 切成固定大小 block（如 16 token），用 block table 映射逻辑→物理。这是 KV cache 显存复用的核心。

### 8.4 NCCL buffer

NCCL 内部分配通信 buffer（all-reduce 的 ring buffer），常单独 stream + allocator 管理，不进 PyTorch 池。但与 PyTorch 显存共享 HBM，总量要算。见 [[NCCL核心机制]]。

### 8.5 内容来源

CUDACachingAllocator 机制整理自 PyTorch 源码 `c10/cuda/CUDACachingAllocator.{h,cpp}` 与文档，碎片策略见 `PYTORCH_CUDA_ALLOC_CONF` 说明，snapshot 见 PyTorch memory snapshot tutorial。

---
相关: [[CUDA显存管理]] | [[GPU内存层级]] | [[GPU memory snapshot]] | [[CUDA Graph与graph capture]] | [[PagedAttention]] | [[推理 CUDA Graph]] | [[gradient checkpointing]] | [[ZeRO (DeepSpeed)]] | [[FSDP2]] | [[activation memory]] | [[NCCL核心机制]]
