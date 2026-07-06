# overlap strategy

> **所属章节**: [[性能问题]]
> **所属模块**: [[07-分布式与并行计算]]
> **难度**: 中（需懂计算-通信依赖 + 流水线调度）

## 1. 一句话定义

**overlap strategy（重叠策略）** 是让**计算与通信并行执行**以隐藏通信延迟的性能优化：反向计算时同步发起梯度 [[all-reduce]]（**梯度分桶 gradient bucketing**，小桶先发）、[[Pipeline Parallel|PP]] 气泡时做 [[Data Parallel|DP]] 通信、[[Fully Sharded Data Parallel|ZeRO-3]] 算时 AllGather 权重与前向重叠。串行时 $T=T_{\text{comp}}+T_{\text{comm}}$，重叠后 $T\approx\max(T_{\text{comp}},T_{\text{comm}})$，隐藏较小者。它是缓解 [[compute vs communication bottleneck|comm-bound]] 的核心手段（非消除——若 $T_{\text{comm}}>T_{\text{comp}}$ 仍有残余通信）。[[Distributed Data Parallel|DDP]] 默认梯度分桶 overlap，Megatron/DeepSpeed 在 3D 中广泛用。

> [!note] overlap 的本质
> - **串行**：$T=T_{\text{comp}}+T_{\text{comm}}$（通信时 GPU 空等）；
> - **重叠**：$T\approx\max(T_{\text{comp}},T_{\text{comm}})$（通信与计算并行，隐藏较小者）；
> - **前提**：通信与计算无数据依赖（如梯度桶 all-reduce 与后续层反向独立）；
> - **局限**：若 $T_{\text{comm}}>T_{\text{comp}}$，残余通信仍瓶颈（overlap 只隐藏 $T_{\text{comp}}$ 部分）。

## 2. 为什么需要它（动机与背景）

comm-bound 时通信串行浪费 GPU：
1. **GPU 空等**：通信时 GPU idle，算力浪费；
2. **comm-bound 主因**：$T_{\text{comm}}>T_{\text{comp}}$ 时通信是瓶颈，串行更糟；
3. **可重叠**：反向时各层梯度独立（不同参数），可边反向边 all-reduce 已算梯度；
4. **减感知延迟**：隐藏 $T_{\text{comp}}$ 部分（若 $T_{\text{comm}}<T_{\text{comp}}$ 则近消除通信感知）；
5. **免费午餐**：不改算法/精度，只调度，提效明显；
6. **3D 必备**：大模型 3D 训练通信重，overlap 是标配。

overlap 是分布式训练性能优化的"便宜且有效"手段，DDP/ZeRO/Megatron 都内置。

## 3. 核心概念详解

### 3.1 串行 vs 重叠

**串行**（无 overlap）：
```
反向 L1 -> 反向 L2 -> ... -> 反向 LN -> all-reduce 全梯度 -> 优化器
[---- T_comp ----][---- T_comm ----]
T = T_comp + T_comm
```

**重叠**（gradient bucketing）：
```
反向 L1 -> 反向 L2 -> 反向 L3 -> ...
           [all-reduce 桶1]  [all-reduce 桶2] ...
[---- T_comp ----]
   [---- T_comm(隐藏在 T_comp 内) ----]
T ≈ max(T_comp, T_comm)
```

### 3.2 梯度分桶（gradient bucketing）

DDP 的核心 overlap：
1. 参数按倒序（反向顺序）分桶（bucket，如 25MB 一桶）；
2. 反向算完一桶梯度 → 立即 all-reduce 该桶（与后续层反向并行）；
3. 小桶先发，大桶后发，流水线 overlap；
4. 反向结束时大部分梯度已 all-reduce 完（仅尾桶待）；
- **收益**：$T_{\text{comm}}$ 隐藏在 $T_{\text{comp}}$（反向）内；
- **前提**：$T_{\text{comm}}<T_{\text{comp}}$（否则残余通信瓶颈）。

### 3.3 PP 气泡 overlap

[[Pipeline Parallel|PP]] 有气泡（warmup/cooldown 时 stage 空闲）：
- 气泡时该 stage 可做 DP 通信（梯度 all-reduce）；
- 或做下一 micro-batch 的前向（interleaved）；
- 隐藏气泡 + 隐藏 DP 通信；
- Megatron 的 1F1B + DP overlap 实现。

### 3.4 ZeRO-3 的 overlap

ZeRO-3 算时 AllGather 权重：
- 前 forward 时 AllGather 下一层权重（与当前层 forward 重叠）；
- 反向时 AllGather 下一层权重 + reduce-scatter 当前层梯度；
- 隐藏权重 AllGather 通信；
- FSDP（PyTorch）支持 `forward_prefetch=True` 预取下一层。

### 3.5 overlap 的条件

- **无数据依赖**：通信的数据与计算的数据独立（如梯度桶 all-reduce 与后续层反向）；
- **通信 < 计算**：$T_{\text{comm}}<T_{\text{comp}}$ 才能完全隐藏（否则残余）；
- **硬件支持**：GPU 有独立通信引擎（NVLink/IB 与 SM 计算并行）；
- **调度复杂**：需框架支持（DDP/ZeRO/Megatron 内置）。

### 3.6 各策略的 overlap

| 策略 | overlap 方式 |
|---|---|
| **DDP** | 梯度分桶，反向时 all-reduce 桶 |
| **ZeRO-2/3** | reduce-scatter 梯度 + AllGather 权重 overlap 反向/前向 |
| **TP** | 较难（每层 all-reduce，计算通信紧耦合） |
| **PP** | 气泡时做 DP 通信 / interleaved 前向 |
| **weight sync** | 增量 sync 与生成 overlap |

## 4. 数学原理 / 公式

### 4.1 串行 vs 重叠

$$
T_{\text{串行}}=T_{\text{comp}}+T_{\text{comm}},\quad T_{\text{重叠}}\approx\max(T_{\text{comp}},T_{\text{comm}})
$$

- 重叠隐藏较小者；
- 若 $T_{\text{comm}}<T_{\text{comp}}$：$T_{\text{重叠}}\approx T_{\text{comp}}$（通信近消除）；
- 若 $T_{\text{comm}}>T_{\text{comp}}$：$T_{\text{重叠}}\approx T_{\text{comm}}$（残余通信瓶颈）。

### 4.2 梯度分桶的隐藏

$$
T_{\text{DDP}}\approx\max(T_{\text{backward}},T_{\text{all-reduce}})\approx T_{\text{backward}}\quad(\text{若 }T_{\text{all-reduce}}<T_{\text{backward}})
$$

- 反向时各桶 all-reduce 并行，隐藏在反向内；
- 仅尾桶（最后一桶梯度）需等 all-reduce 完才优化器更新；
- 尾桶延迟 $\sim T_{\text{all-reduce, bucket}}$（小桶，短）。

### 4.3 overlap 收益

$$
\text{speedup}=\frac{T_{\text{comp}}+T_{\text{comm}}}{\max(T_{\text{comp}},T_{\text{comm}})}=1+\frac{\min(T_{\text{comp}},T_{\text{comm}})}{\max(T_{\text{comp}},T_{\text{comm}})}
$$

- $T_{\text{comp}}=T_{\text{comm}}$：speedup = 2（最大）；
- $T_{\text{comm}}\ll T_{\text{comp}}$：speedup ≈ 1（本就 compute-bound，overlap 收益小）；
- $T_{\text{comm}}\gg T_{\text{comp}}$：speedup ≈ $T_{\text{comm}}/T_{\text{comp}}$（但残余通信仍瓶颈，overlap 有限）。

### 4.4 PP 气泡 overlap

$$
T_{\text{PP+overlap}}\approx\max(T_{\text{bubble}},T_{\text{DP comm}})\text{（气泡时做 DP 通信）}
$$

- 气泡时间用于 DP 通信，隐藏两者；
- 减气泡感知 + 减 DP 通信感知。

## 5. 代码示例

```python
import torch, torch.nn as nn, time

class LM(nn.Module):
    def __init__(s, v, d, L):
        super().__init__(); s.layers=nn.ModuleList([nn.Linear(d,d) for _ in range(L)]); s.head=nn.Linear(d,v)
    def forward(s,x):
        for l in s.layers: x=l(x); return s.head(x)

V, D, L = 50, 64, 8
model = LM(V, D, L)

# ===== 串行: 反向完再 all-reduce =====
def serial_step(model, x, N_sim=4):
    """模拟串行: 反向 -> all-reduce 全梯度"""
    out = model(x); loss = out.float().mean()
    loss.backward()
    # 模拟 all-reduce 全梯度(串行,反向后)
    for p in model.parameters():
        if p.grad is not None:
            _ = p.grad * N_sim  # 模拟 all-reduce sum
    return loss

# ===== 重叠: 梯度分桶,反向时 all-reduce 桶 =====
def overlap_step(model, x, N_sim=4, bucket_size=2):
    """模拟 overlap: 反向时按桶 all-reduce"""
    # 注册 hook: 梯度算完即 all-reduce(分桶)
    buckets = []
    cur_bucket = []
    for p in model.parameters():
        def make_hook(param):
            def hook(grad):
                # 模拟: 梯度算完立即 all-reduce(与后续反向并行)
                param.grad = grad * N_sim  # 模拟 all-reduce
            return hook
        p.register_hook(lambda g, p=p: g * N_sim)  # 简化: hook 模拟 all-reduce
    out = model(x); loss = out.float().mean()
    loss.backward()  # 反向时各层梯度 hook 触发,模拟 overlap
    return loss

x = torch.randint(0, V, (4, 8))
# 计时(概念,实际通信在 GPU 间)
t0=time.time(); serial_step(model,x); t_serial=time.time()-t0
t0=time.time(); overlap_step(model,x); t_overlap=time.time()-t0
print(f"串行(反向后 all-reduce): {t_serial*1000:.2f}ms")
print(f"overlap(梯度分桶反向时 all-reduce): {t_overlap*1000:.2f}ms")
print(f"实际: DDP overlap 隐藏 T_comm 在 T_backward 内,T≈max(T_comp,T_comm)")

# ===== 重叠收益分析 =====
print("\n重叠收益分析:")
T_comp, T_comm = 100, 80  # ms
print(f"  T_comp={T_comp}ms, T_comm={T_comm}ms")
print(f"  串行: T={T_comp+T_comm}ms")
print(f"  重叠: T≈max({T_comp},{T_comm})={max(T_comp,T_comm)}ms (speedup={(T_comp+T_comm)/max(T_comp,T_comm):.2f}x)")
T_comp, T_comm = 100, 150  # comm-bound
print(f"  T_comp={T_comp}ms, T_comm={T_comm}ms (comm-bound)")
print(f"  重叠: T≈max({T_comp},{T_comm})={max(T_comp,T_comm)}ms (残余通信瓶颈,overlap 有限)")

# ===== 实际 DDP 用法 =====
print("\n实际 DDP overlap:")
print("  model = DDP(model, device_ids=[0], bucket_cap_mb=25)  # 梯度分桶,25MB/桶")
print("  # 反向时自动 overlap all-reduce,无需手动")
print("  # ZeRO-3: FSDP(forward_prefetch=True) 预取下一层权重 overlap")
```

> [!tip] 实际工程
> - **DDP**：`bucket_cap_mb`（默认 25MB），自动梯度分桶 overlap；
> - **FSDP**：`forward_prefetch=True` 预取权重 overlap 前向；
> - **Megatron**：1F1B + DP overlap，PP 气泡填 DP 通信；
> - **监控**：profiler 查通信是否与计算重叠（[[torch profiler]]/[[nsys]]）。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[compute vs communication bottleneck]]（overlap 解决的问题）、[[all-reduce]]/[[AllGather]]/[[reduce-scatter]]（重叠的通信）、[[NCCL通信拓扑]]（异步通信支持）、[[Distributed Data Parallel|DDP]]/[[Fully Sharded Data Parallel|FSDP]]/[[Pipeline Parallel]]（实现 overlap 的框架）。
- **下游（应用）**: DDP 梯度分桶、ZeRO-3 权重预取、Megatron 3D overlap、weight sync overlap、分布式训练性能优化。
- **对比 / 易混**:
  - **overlap strategy（本篇）vs [[compute vs communication bottleneck]]**：前者是解法（重叠隐藏通信），后者是诊断（分析瓶颈）。互补。
  - **overlap vs 串行**：见 §3.1。重叠隐藏较小者，串行累加。
  - **gradient bucketing vs gradient checkpointing**：前者是分桶 all-reduce overlap（通信优化），后者是重算前向省显存（显存优化，但增计算）。不同。
  - **overlap vs 减通信量**：overlap 隐藏通信（时间），减通信量降通信本身。互补，常组合。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"overlap 完全消除通信"**：错。隐藏 $T_{\text{comp}}$ 部分，若 $T_{\text{comm}}>T_{\text{comp}}$ 残余仍瓶颈。
> 2. **"overlap 免费午餐"**：基本是，但有调度开销 + 复杂度 + 需框架支持。
> 3. **"overlap 不需硬件支持"**：错。需 GPU 独立通信引擎（NVLink/IB 与 SM 并行），否则通信阻塞计算。
> 4. **"TP 也能 overlap"**：难。TP 每 layer all-reduce 与计算紧耦合（下一 layer 需上一 layer 输出），overlap 空间小。
> 5. **"梯度分桶越大越好"**：错。桶大则 all-reduce 慢，与反向 overlap 空间小；桶小则 all-reduce 次数多，延迟增。有最优（DDP 默认 25MB）。
> 6. **"overlap 不需 profiler 验证"**：错。需 profiler 确认通信确实与计算重叠（[[torch profiler]]/[[nsys]]）。
> 7. **"ZeRO-3 overlap 无开销"**：错。预取权重需额外显存（存下一层权重），且调度复杂。
> 8. **"PP 气泡 overlap 消除气泡"**：错。气泡仍存在，overlap 用气泡时间做 DP 通信，隐藏两者（非消除气泡）。

## 8. 延伸细节

### 8.1 DDP 梯度分桶详解

- 参数按倒序（反向顺序）分桶；
- 每桶 25MB（默认），可调 `bucket_cap_mb`；
- 反向算完一桶 → 异步 all-reduce（与后续层反向并行）；
- 反向结束时大部分桶已 all-reduce 完；
- 仅尾桶待 all-reduce 完才优化器更新（尾延迟小）；
- 是 DDP 默认行为，无需手动。

### 8.2 ZeRO-3 的 overlap

- `forward_prefetch=True`：前向时预取下一层 AllGather 权重（与当前层 forward 重叠）；
- 反向时：AllGather 下一层权重 + reduce-scatter 当前层梯度（重叠）；
- 隐藏权重 AllGather 通信；
- 代价：预取占额外显存（下一层权重）。

### 8.3 Megatron 的 3D overlap

- **1F1B + DP overlap**：PP 气泡时做 DP all-reduce 梯度；
- **TP overlap**：有限（TP 紧耦合），但 layer 间 TP all-reduce 可与下一 layer 前向部分重叠；
- **interleaved PP**：减气泡，更多 overlap 空间；
- 是 3D 训练性能的关键。

### 8.4 overlap 的硬件支持

- **NVLink + IB**：独立通信引擎，与 SM 计算并行；
- **GPUDirect RDMA**：IB 直达 GPU，不经 CPU，减延迟；
- **CUDA stream**：计算 stream + 通信 stream 分离，并行；
- 无独立通信引擎（如 PCIe）则通信阻塞计算，overlap 失效。

### 8.5 overlap 的 profiler 验证

- [[torch profiler]]/[[nsys]] trace；
- 查 NCCL 调用是否与 CUDA kernel 时间重叠（非串行）；
- 查 GPU idle（空等 → overlap 未生效）；
- 查通信占比（overlap 后应降）；
- 详见 [[torch profiler]]/[[nsys]]（Chapter 10）。

### 8.6 overlap 的局限

- $T_{\text{comm}}>T_{\text{comp}}$ 时残余通信瓶颈（overlap 只隐藏 $T_{\text{comp}}$）；
- 紧耦合通信（TP 每 layer）overlap 空间小；
- 调度复杂，框架支持需求；
- 硬件依赖（独立通信引擎）；
- 故 overlap 是缓解非消除，需与减通信量/提带宽组合。

### 8.7 overlap 与其他优化的组合

- **+ 减通信量**：梯度压缩（TopK/量化）降 $T_{\text{comm}}$，再 overlap 隐藏；
- **+ 提带宽**：NVLink/多 ring 提带宽降 $T_{\text{comm}}$，overlap 隐藏；
- **+ 增 batch**：DP 增 B 提 $T_{\text{comp}}$，overlap 更易隐藏 $T_{\text{comm}}$；
- **+ 调并行策略**：comm-bound 时减 TP（紧耦合），增 DP/PP（可 overlap）；
- 组合用，详见 [[compute vs communication bottleneck]]。

---
相关: [[性能问题]]、[[compute vs communication bottleneck]]、[[all-reduce]]、[[AllGather]]、[[reduce-scatter]]、[[NCCL通信拓扑]]、[[Distributed Data Parallel]]、[[Fully Sharded Data Parallel]]、[[Pipeline Parallel]]、[[3D parallelism]]、[[torch profiler]]、[[nsys]]、[[gradient checkpointing]]、[[overlap strategy]]
