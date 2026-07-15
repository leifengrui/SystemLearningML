# 异步memcpy与pinned memory

> **所属章节**: [[CUDA stream与event]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: async memcpy / cudaMemcpyAsync / pinned memory / page-locked memory / zero-copy / DMA 传输
> **难度**: 中（需懂 OS 分页 + [[CUDA stream与event]]）


## 1. 一句话定义

**异步 memcpy** 是经 GPU 的 **copy engine** 在后台执行 H2D/D2H（host↔device）数据拷贝、与 compute engine 的 kernel 并发的机制；要让 copy engine 真正 DMA 直传，host 端内存必须是 **pinned memory（页锁定内存）**——OS 不允许换出、物理地址固定、可直接被 GPU 的 DMA 引擎寻址。两者配合（pinned host 内存 + `cudaMemcpyAsync` 入非默认 stream）才能实现"计算与传输 overlap"，把 CPU↔GPU 搬运藏于计算之内，是 data loader、KV cache offload、checkpoint 保存的底层加速手段。它是 [[CUDA stream与event]] 在数据搬运维度的典型应用。

> [!note] 三句话定位
> - **是什么**：pinned memory = 锁页 host 内存可被 GPU DMA 直传；async memcpy = 后台 copy engine 拷贝，与计算并发。
> - **为什么**：默认 pageable host 内存要先拷到 staging 再 DMA（两跳，慢且阻塞）；pinned + async 才能真并发、把传输藏于计算。
> - **代价**：pinned 内存吃物理 RAM、不能换出（系统内存压力增），且分配开销大（`cudaMallocHost` 比普通 malloc 慢）。


## 2. 为什么需要它（动机与背景）

### 2.1 默认 memcpy 的两跳问题

CPU 普通内存是 **pageable**（可被 OS 换出到 swap，物理地址不固定）。GPU DMA 引擎要求**物理地址固定**才能直传。所以默认 `cudaMemcpy`（同步、pageable）的实际链路是：

1. CUDA driver 在内部 staging buffer（pinned）里拷一份。
2. DMA 把 staging 拷到 GPU。

两跳：CPU→staging（CPU 拷）+ staging→GPU（DMA）。慢、且同步阻塞 CPU。

### 2.2 pinned 直传 + async 并发

用 `cudaMallocHost` 分配的 **pinned** 内存：OS 锁页不换出、物理地址固定 → DMA 直传（一跳）→ 快。再配合 `cudaMemcpyAsync(..., stream)` 入非默认 stream → copy engine 后台跑、CPU 不阻塞、可与 compute engine 的 kernel 并发。

这样 data loader 拷下一 batch（copy engine）能与本 batch 前向（compute engine）并发，藏传输于计算。是训练吞吐的基础优化。

### 2.3 训练/推理的真实场景

- **data loader**：pin_memory=True（PyTorch DataLoader）→ batch 落 pinned → `non_blocking=True` 的 `.to('cuda')` 走 async memcpy 与上一个 batch 的 forward 并发。
- **KV cache swap**：vLLM 把 KV cache 换到 CPU 再换回，靠 pinned + async（见 [[cache eviction与swap offload]]）。
- **checkpoint 保存**：把 weight 从 GPU 拷到 pinned host 再写盘，用 async 减训练阻塞（见 [[async distributed checkpoint]]）。
- **parameter server / weight sync**：[[RL权重同步]] 跨设备传权重用 pinned + async。


## 3. 核心概念详解

### 3.1 pageable vs pinned

| | pageable (普通 malloc) | pinned (cudaMallocHost) |
|---|---|---|
| 可换出 | 是（OS 可 page out） | 否（锁页） |
| 物理地址 | 不固定 | 固定 |
| GPU DMA 直传 | 否（需 staging 两跳） | 是（一跳） |
| 分配 | 快（普通 malloc） | 慢（要锁页，系统调用） |
| 占用 | 随 page table | 吃物理 RAM |
| 带宽 | ~1/2 PCIe 峰值 | 接近 PCIe/NVLink 峰值 |

### 3.2 同步 vs 异步 memcpy

| API | 阻塞 CPU | 与 kernel 并发 | 场景 |
|---|---|---|---|
| `cudaMemcpy` (sync, default stream) | 是 | 否（隐式同步） | 简单/一次性 |
| `cudaMemcpyAsync(..., stream)` | 否（返回即继续） | 是（需 pinned + 非默认 stream） | overlap 优化 |

注意：async + pageable 仍会被 driver 内部 staging（不能真并发，且行为不一致），**要真并发必须 pinned**。

### 3.3 copy engine 的并发

GPU 有 1-2 个 copy engine（A100 2 个，分 H2D / D2H 方向可同时跑）。多 stream + pinned + async 时：

- stream A：kernel 跑在 compute engine。
- stream B：H2D memcpy 跑在 copy engine。
- 二者并发，互不阻塞。

条件：不同 stream、pinned、资源空闲、无隐式同步（见 [[CUDA stream与event]] 的并发条件）。

### 3.4 PyTorch 里的用法

```python
# DataLoader pin_memory
loader = DataLoader(ds, batch_size=..., pin_memory=True, num_workers=4)

for batch in loader:                       # batch 在 pinned host
    x = batch[0].to('cuda', non_blocking=True)   # async memcpy 入非默认 stream
    # 上一 batch 的 forward 在 compute engine 并发
```

`non_blocking=True` 让 `.to()` 走 async（要求源是 pinned）。num_workers 多进程预取，配合 pinned 形成 pipeline。

### 3.5 zero-copy 内存

`cudaHostAlloc(..., cudaHostAllocMapped)` 让 GPU 直接映射 host pinned 地址，GPU 可读 CPU 内存（不经 HBM 拷贝）。适合数据小、只读一次、拷贝比计算还贵的场景。但每次访问走 PCIe（慢），不能替代真正的 pinned+async 传输用于大数据。LLM 训练一般不用，研究原型可。

### 3.6 与 NCCL/GPUDirect 的关系

跨 GPU 的数据传输（all-reduce）走 NVLink，不经 host。但"GPU↔host"的搬运走 PCIe + pinned。GPUDirect RDMA 让 GPU 直连网卡（不经 host bounce），是大规模分布式训练的关键，见 [[RDMA与GPUDirect]]。


## 4. 数学原理 / 公式

### 4.1 传输带宽

$$
T_{\text{copy}} = \frac{\text{Bytes}}{\text{PCIe/NVLink 带宽}}, \quad \text{PCIe Gen4 x16} \approx 32\text{ GB/s}, \quad \text{NVLink3} \approx 600\text{ GB/s}
$$

pageable 因两跳 + CPU 拷，实际带宽约峰值的 1/3-1/2；pinned + async 接近峰值。

### 4.2 overlap 收益

设单 stream 串行：$T = T_{\text{copy}} + T_{\text{compute}}$。pinned + async + 双 stream 稳态：

$$
T_{\text{overlap}} = \max(T_{\text{copy}}, T_{\text{compute}})
$$

当 $T_{\text{copy}} \approx T_{\text{compute}}$ 收益 ~2×；当一方远大趋近 1（瓶颈侧不变）。**若 $T_{\text{copy}} > T_{\text{compute}}$，说明搬运是瓶颈**——需减小数据量（fp16 vs fp32）、增 PCIe/NVLink、或 prefetch 更前。

### 4.3 流水线吞吐

$k$ 级流水（prefetch + compute）稳态吞吐 $\approx 1/\max_i(T_i)$，而非 $1/\sum T_i$。data loader 多 worker + pinned 形成多级流水，藏延迟。

### 4.4 pinned 占用约束

pinned 吃物理 RAM：若系统 RAM = $R$、其他进程用 $R_o$、pinned 用 $P$，则 $P \le R - R_o$。pin 太多会触发系统 OOM 或 swap thrash（pin 本来为防 swap）。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟 pageable vs pinned 带宽

```python
def copy_bandwidth(bytes_size, bw_pageable, bw_pinned):
    """pageable 两跳 ~ 峰值 1/3; pinned+async ~ 峰值."""
    t_page = bytes_size / bw_pageable
    t_pin = bytes_size / bw_pinned
    return t_page, t_pin

# 8GB 数据, PCIe Gen4 ~32GB/s 峰值
t1, t2 = copy_bandwidth(8e9, bw_pageable=11e9, bw_pinned=28e9)
print(f"pageable: {t1*1000:.0f} ms; pinned+async: {t2*1000:.0f} ms; 加速 {t1/t2:.2f}x")
```

### 5.2 PyTorch pin_memory + non_blocking

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

ds = TensorDataset(torch.randn(1_000_000, 1024))
loader = DataLoader(ds, batch_size=4096, pin_memory=True, num_workers=4)

s_copy = torch.cuda.Stream()
s_compute = torch.cuda.Stream()
it = iter(loader)
next_batch = next(it)
for step in range(3):
    # prefetch 下一批到 GPU (async, copy engine)
    with torch.cuda.stream(s_copy):
        x = next_batch[0].to('cuda', non_blocking=True)
    # 本批在 compute engine 算 (与 prefetch 并发)
    with torch.cuda.stream(s_compute):
        s_copy.synchronize()              # 等本批到 GPU
        out = x @ x
    try: next_batch = next(it)
    except StopIteration: pass
torch.cuda.synchronize()
```

### 5.3 显式 pinned 分配（对照 CUDA C）

```python
# CUDA C: float* h; cudaMallocHost(&h, N*sizeof(float));  // pinned
# PyTorch 等价: 用 pin_memory=True 或显式
pinned_buf = torch.empty(1024, 1024, pin_memory=True)   # pinned host tensor
# 或 cuda.HostAllocator
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[CUDA stream与event]]（async memcpy 入 stream）、OS 分页机制、GPU copy engine。
- **下游（应用）**: data loader pin_memory、[[overlap strategy]]（计算-拷贝 overlap 的最底层实现）、[[cache eviction与swap offload]]（KV cache 换出）、[[async distributed checkpoint]]（异步存盘）、[[RL权重同步]]（跨设备传权重）、[[NCCL核心机制]]（GPUDirect 跳过 host bounce）。
- **对比 / 易混**:
  - **pinned vs zero-copy**：pinned 是锁页 host 内存 + 拷到 GPU；zero-copy 是 GPU 直接映射 host 地址不拷（慢但省拷）。大数据用 pinned+async，小一次性数据可 zero-copy。
  - **async memcpy vs NCCL**：async memcpy 是 host↔device（CPU↔GPU）搬运；NCCL 是 device↔device（GPU↔GPU）搬运走 NVLink。两者方向不同。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 `non_blocking=True` 就一定能并发
> 必须源是 pinned。若源 pageable，`non_blocking=True` 仍会内部 staging（甚至阻塞或行为不一致）。DataLoader 的 `pin_memory=True` 才让 batch 是 pinned。

> [!warning] 误区 2：pinned 内存越多越好
> pinned 吃物理 RAM 且不换出。pin 太多挤占其他进程、触发 OOM。常见做法是 pin 一批/几批循环用，不 pin 全部数据。

> [!warning] 误区 3：忘了 synchronize 就读 async 结果
> async memcpy 入 stream 后立即返回，若马上 `tensor` 访问内容（如 `.cpu()` 或 numpy）会隐式同步阻塞。要么显式 `synchronize`，要么靠后续 stream 序保证。

> [!warning] 误区 4：混淆 H2D 与 D2H 方向
> 拷方向不同（H2D 走 PCIe/NVLink 写 GPU，D2H 反向），某些硬件两方向带宽不对称。NCCL 的 GPUDirect 是另一回事（不经 host）。

> [!warning] 误区 5：忽视 PCIe 与 NVLink 区别
> 单卡 host↔GPU 走 PCIe（~32GB/s）；多 GPU 间走 NVLink（600GB/s+）。把"跨 GPU 数据传输"误用 host 中转（GPU→host→另一 GPU）会慢几十倍，应直接 NCCL/P2P。


## 8. 延伸细节

### 8.1 GPUDirect 家族

- **GPUDirect P2P**：同机两 GPU 直接 NVLink 互访（不经 host）。
- **GPUDirect RDMA**：GPU 直连网卡，不经 host bounce（跨机分布式训练的关键）。
- **GPUDirect Storage**：GPU 直连 NVMe/并行 FS，不经 host bounce（checkpoint 保存加速）。

三者都靠 pinned + DMA 思想延伸，见 [[RDMA与GPUDirect]]。

### 8.2 pinned 池化

`cudaMallocHost` 单次分配慢。生产代码常预分配一个 pinned pool，循环复用（类似 [[CUDA allocator与memory pool]] 思想）。PyTorch 的 `pin_memory=True` 内部也有 cache。

### 8.3 异构内存与 unified memory

`cudaMallocManaged` 让 host/GPU 共享虚拟地址、运行时按需迁移。性能通常不如显式 pinned + async（页迁移 thrash），但代码简单。研究原型可用，生产训练一般不。

### 8.4 内容来源

pinned/async memcpy 语义整理自 NVIDIA CUDA C++ Programming Guide 的 Device Memory 与 Asynchronous Transfers 章节，PyTorch DataLoader pin_memory 见 `torch.utils.data` 文档。

---
相关: [[CUDA stream与event]] | [[overlap strategy]] | [[cache eviction与swap offload]] | [[async distributed checkpoint]] | [[RL权重同步]] | [[RDMA与GPUDirect]] | [[NCCL核心机制]] | [[CUDA allocator与memory pool]]
