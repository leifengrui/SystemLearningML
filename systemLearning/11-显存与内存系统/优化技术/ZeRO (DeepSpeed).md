# ZeRO (DeepSpeed)

> **所属章节**: [[优化技术]]
> **所属模块**: [[11-显存与内存系统]]
> **别名**: ZeRO / Zero Redundancy Optimizer / ZeRO-1/2/3 / DeepSpeed ZeRO
> **难度**: 中高（需懂 [[optimizer state memory]] + [[gradient memory]] + [[Data Parallel]] + 通信）


## 1. 一句话定义

**ZeRO（Zero Redundancy Optimizer）** 是 DeepSpeed 提出的**数据并行显存优化技术**——通过把 optimizer state / gradient / 参数**分片（partition）到各卡**，消除标准 [[Data Parallel]]（DP）中每卡冗余存全量的浪费，让 $N$ 卡总显存承载 $N \times$ 大模型。分三个阶段：**ZeRO-1**（分片 [[optimizer state memory|optimizer state]]，无通信代价）、**ZeRO-2**（+分片 [[gradient memory|gradient]]，通信不增）、**ZeRO-3**（+分片参数，通信增但显存最大省）。ZeRO-3 与 [[Fully Sharded Data Parallel]]（FSDP）本质相同。ZeRO 让单卡放不下的大模型（如 7B+）在多卡训练可行，是大模型训练标配。可与 [[gradient checkpointing]]、[[offloading (CPU-NVMe)]]、TP/PP 叠加。

> [!note] 三句话定位
> - **是什么**：DP 的显存分片技术，把 state/gradient/参数分到各卡，消除冗余。
> - **为什么需要**：标准 DP 每卡存全量（冗余），大模型显存不够；ZeRO 让 N 卡承载 N× 模型。
> - **三阶段**：ZeRO-1（分 opt，无通信代价）、ZeRO-2（+grad）、ZeRO-3（+参数，通信增，=FSDP）。


## 2. 为什么需要它（动机与背景）

### 2.1 标准 DP 的冗余

[[Data Parallel]]：每卡存**完整**模型副本（权重 + gradient + optimizer state + activation），各算各 batch，反向 allreduce gradient 同步。问题：$N$ 卡各存一份完整 state/gradient/权重，**冗余 $N \times$**。7B Adam：单卡 88GB，8 卡总冗余存 $8 \times 88 = 704$GB（实际只需 1 份权重 + 1 份 state + 各卡 activation）。大模型单卡放不下时，DP 无法用。

### 2.2 ZeRO 消除冗余

ZeRO 的洞察：DP 的**计算**是并行的（各卡算不同 batch），但**存储**是冗余的（各卡存相同 state/gradient/权重）。把冗余存储**分片**到各卡，每卡只存 $1/N$，需要时聚合。计算仍并行（不改计算图），存储消除冗余。

- ZeRO-1：optimizer state 分片（每卡 $1/N$ state）。
- ZeRO-2：+ gradient 分片（每卡 $1/N$ gradient）。
- ZeRO-3：+ 参数分片（每卡 $1/N$ 权重）。

### 2.3 通信代价的 trade-off

分片需聚合使用：

- ZeRO-1：state 分片，optimizer step 时各卡只更新自己的 $1/N$ 参数的 state，**无额外通信**（state 本就在各卡，gradient 仍 allreduce）。
- ZeRO-2：gradient 分片，allreduce → reduce-scatter（每卡只收自己 $1/N$），**通信量不变**（同 allreduce 量）。
- ZeRO-3：参数分片，前向/反向前需 **all-gather** 聚合该层参数（用完释放），**通信增**（每层 all-gather）。

ZeRO-1/2 是"免费午餐"（显存省、通信不增），ZeRO-3 用通信换显存。

### 2.4 ZeRO-3 与 FSDP

[[Fully Sharded Data Parallel]]（FSDP，PyTorch 原生）与 ZeRO-3 本质相同：参数分片，前向/反向 all-gather 聚合，用完释放。FSDP 是 PyTorch 对 ZeRO-3 的实现。两者可互换理解。详见 [[Fully Sharded Data Parallel]]。

### 2.5 与 TP/PP 的区别

- **ZeRO/DP**：数据并行 + 显存分片。每卡算不同 batch，计算独立，通信是 gradient/参数聚合。模型完整（ZeRO-1/2）或分片（ZeRO-3）但每卡算完整层。
- **[[Tensor Parallel]]（TP）**：切模型层内的计算（matmul 分块）。每卡算同一层的不同部分，通信是层内 allreduce。需 NVLink 快互连。
- **[[Pipeline Parallel]]（PP）**：切模型层间（stage）。每卡算不同层，通信是 stage 间 send/recv。有 bubble。

ZeRO 是 DP 的优化（不改计算图），TP/PP 是模型并行的切分（改计算图）。可组合：3D parallel = DP(ZeRO) + TP + PP。

### 2.6 大模型训练的标配

7B 单卡（88GB）放不下 A100 40GB → 需 ZeRO-1（39GB）或 ZeRO-2（27GB）。70B 更需 ZeRO-3 + offloading。ZeRO 是大模型训练显存优化的基础，与 [[gradient checkpointing]]（减 activation）、混合精度（减半）叠加使用。


## 3. 核心概念详解

### 3.1 三阶段分片

| 阶段 | 分片对象 | 每卡显存（7B, 8卡） | 通信变化 |
|---|---|---|---|
| ZeRO-0 | 无（标准 DP） | 88 GB | allreduce grad（2V） |
| **ZeRO-1** | optimizer state | 39 GB | 不变 |
| **ZeRO-2** | + gradient | 27 GB | 不变（reduce-scatter） |
| **ZeRO-3** | + 参数 | 15 GB | **增**（all-gather 参数） |

### 3.2 ZeRO-1：optimizer state 分片

每卡只存 $1/N$ 的 optimizer state（Adam m, v）。optimizer step 时各卡只更新自己负责的 $1/N$ 参数的 state，然后用 all-gather 把更新后的参数广播给所有卡（或各卡只持有自己分片，下次用时聚合）。

- 显存：state $V_{\text{opt}} \to V_{\text{opt}}/N$。7B Adam 56GB → 7GB/卡。
- 通信：同标准 DP（allreduce gradient），state 分片无额外通信。**免费午餐**。

### 3.3 ZeRO-2：+ gradient 分片

反向后 gradient 不 allreduce（全量聚合），而是 **reduce-scatter**：每卡只收自己负责的 $1/N$ 参数的 gradient（其余释放）。optimizer step 用本地 $1/N$ gradient 更新 $1/N$ state 和参数。

- 显存：gradient $V \to V/N$。7B fp16 grad 14GB → 1.75GB/卡。
- 通信：reduce-scatter 通信量同 allreduce（$2(N-1)/N \cdot V$），不减。**仍免费**（显存省、通信不增）。

### 3.4 ZeRO-3：+ 参数分片

参数也分片，每卡只存 $1/N$ 权重。前向/反向到某层时，**all-gather** 聚合该层完整参数（用完释放），计算该层，然后释放参数（只留自己分片）。

- 显存：权重 $V \to V/N$。7B fp16 权重 14GB → 1.75GB/卡。总 ~15GB/卡。
- 通信：每层 all-gather 参数（前向 + 反向各一次），**通信增** $\sim 2V$/step（额外参数聚合）。trade 通信换显存。

### 3.5 通信量对比

| 阶段 | 通信/step | 通信类型 |
|---|---|---|
| DP/ZeRO-1 | $2V$（allreduce grad） | gradient 全量 |
| ZeRO-2 | $2V$（reduce-scatter grad） | gradient 分片收 |
| ZeRO-3 | $2V$（grad）+ $\sim 2V$（all-gather 参数） | 参数聚合增 |

ZeRO-3 通信量 ~2× DP，但显存减 $N$×。带宽富余（NVLink）时值得，带宽紧（跨节点 IB）时可能 [[communication bottleneck]]。

### 3.6 ZeRO 的层次选择

- 模型单卡放得下：ZeRO-1（省 state，无代价）或标准 DP。
- 模型单卡放不下 state 但放得下权重：ZeRO-2。
- 模型权重单卡也放不下：ZeRO-3（+ [[gradient checkpointing]] 减 activation）。
- 仍不够：+ [[offloading (CPU-NVMe)]]（ZeRO-Offload/Infinity）。

按显存瓶颈层级递进。


## 4. 数学原理 / 公式

### 4.1 各阶段显存（7B, 8卡, fp16 权重/梯度, fp32 state, FA+ckpt activation）

$$
M_{\text{DP}} = \underbrace{V_w}_{14} + \underbrace{V_g}_{14} + \underbrace{V_{\text{opt}}}_{56} + \underbrace{V_{\text{act}}}_{4.3} = 88 \text{ GB/卡}
$$

$$
M_{\text{ZeRO-1}} = V_w + V_g + \frac{V_{\text{opt}}}{N} + V_{\text{act}} = 14 + 14 + 7 + 4.3 = 39 \text{ GB}
$$

$$
M_{\text{ZeRO-2}} = V_w + \frac{V_g}{N} + \frac{V_{\text{opt}}}{N} + V_{\text{act}} = 14 + 1.75 + 7 + 4.3 = 27 \text{ GB}
$$

$$
M_{\text{ZeRO-3}} = \frac{V_w + V_g + V_{\text{opt}}}{N} + V_{\text{act}} = \frac{84}{8} + 4.3 = 15 \text{ GB}
$$

### 4.2 通信量

$$
C_{\text{ZeRO-1/2}} = \frac{2(N-1)}{N} V \approx 2V \quad (\text{同 DP})
$$

$$
C_{\text{ZeRO-3}} \approx 2V + \underbrace{2V}_{\text{all-gather 参数}} = 4V
$$

### 4.3 显存/通信 trade

ZeRO-3：显存 $\times 1/N$，通信 $\times 2$。$N$ 大时显存省显著（$N=8$ 减 5.6×），通信增 2×。NVLink 带宽富余时 throughput 损失小，跨节点 IB 紧时损失大。

### 4.4 与 TP 的显存对比

TP 切模型层内，每卡存 $1/N$ 每层参数（类似 ZeRO-3 的参数分片），但 TP 每层 allreduce 激活（通信频繁），且需 NVLink。ZeRO-3 切层间（每卡存若干完整层），all-gather 参数（通信 less frequent 但数据大）。TP 适合节点内 NVLink，ZeRO 适合节点内/跨节点。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟各阶段显存

```python
def zero_memory(n_params, stage, n_gpu=8, bpe_w=2, bpe_opt=4, act_bytes=4.3e9):
    """ZeRO 各阶段每卡显存. Adam (2 states fp32)."""
    w = n_params * bpe_w            # 权重 fp16
    g = n_params * bpe_w            # gradient fp16
    o = n_params * 2 * bpe_opt      # optimizer state (m,v fp32)
    a = act_bytes                   # activation (FA+ckpt)
    if stage == 0:   return w + g + o + a
    if stage == 1:   return w + g + o/n_gpu + a
    if stage == 2:   return w + g/n_gpu + o/n_gpu + a
    if stage == 3:   return (w + g + o)/n_gpu + a

print('7B, 8 GPU, Adam fp32 state, FA+ckpt activation:')
for s in [0, 1, 2, 3]:
    print(f'  ZeRO-{s}: {zero_memory(7e9, s)/1e9:6.1f} GB/卡')

# 70B
print('70B, 8 GPU:')
for s in [1, 2, 3]:
    print(f'  ZeRO-{s}: {zero_memory(70e9, s)/1e9:6.1f} GB/卡')
```

### 5.2 通信量对照

```python
def comm_volume(n_params, stage, n_gpu=8, bpe=2):
    V = n_params * bpe
    if stage <= 2:
        return 2*(n_gpu-1)/n_gpu * V   # allreduce/reduce-scatter grad
    else:
        return 2*(n_gpu-1)/n_gpu * V + 2*V   # + all-gather 参数

for s in [0, 1, 2, 3]:
    print(f'ZeRO-{s} 通信: {comm_volume(7e9, s)/1e9:.1f} GB/step')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[optimizer state memory]]/[[gradient memory]]/权重（分片对象）、[[Data Parallel]]（ZeRO 是 DP 的优化）、allreduce/reduce-scatter/all-gather（分片的通信原语）。
- **下游（应用）**: [[Fully Sharded Data Parallel]]（ZeRO-3 的 PyTorch 实现）、[[offloading (CPU-NVMe)]]（ZeRO-Offload/Infinity）、[[gradient checkpointing]]（叠加减 activation）、[[Tensor Parallel]]/[[Pipeline Parallel]]/[[3D parallelism]]（组合并行）、混合精度（叠加）、[[communication bottleneck]]（ZeRO-3 增通信）、大模型训练（7B+ 标配）。
- **对比 / 易混**:
  - **ZeRO vs [[Data Parallel\|DP]]**：DP 每卡存全量（冗余），ZeRO 分片消除冗余。ZeRO 是 DP 的显存优化版。
  - **ZeRO-3 vs [[Fully Sharded Data Parallel\|FSDP]]**：本质相同（参数分片 + all-gather）。FSDP 是 PyTorch 实现，ZeRO-3 是 DeepSpeed 实现。
  - **ZeRO vs [[Tensor Parallel\|TP]]**：ZeRO 切层间/数据并行（不改计算图），TP 切层内（改计算图，每层 allreduce 激活）。ZeRO 通信是参数/gradient 聚合，TP 通信是层内激活 allreduce。
  - **ZeRO-1/2 vs ZeRO-3**：前者无通信代价（免费），后者增通信换显存。按显存需求递进。


## 7. 常见误区与易错点

> [!warning] 误区 1：ZeRO 减通信
> 不完全。ZeRO-1/2 通信量同标准 DP（不减），ZeRO-3 **增**通信（all-gather 参数）。ZeRO 的核心是**显存优化**（消除冗余），不是通信优化。误以为 ZeRO 减通信会导致 throughput 预期错误。

> [!warning] 误区 2：ZeRO = TP
> 不同。ZeRO 是数据并行的显存分片（每卡算不同 batch，参数分片但计算独立），TP 是模型并行的层内切分（每卡算同一层的不同部分，计算依赖）。ZeRO 不改计算图，TP 改。ZeRO 通信是参数/gradient 聚合（less frequent），TP 通信是层内激活 allreduce（每层 frequent）。

> [!warning] 误区 3：ZeRO-3 总是优于 ZeRO-1/2
> 不一定。ZeRO-3 显存最省但通信增。模型单卡放得下时用 ZeRO-1（无通信代价）更优。ZeRO-3 只在权重单卡放不下时才需。盲目上 ZeRO-3 增通信得不偿失。

> [!warning] 误区 4：混淆 ZeRO 阶段的分片对象
> ZeRO-1 分 state，ZeRO-2 加 grad，ZeRO-3 加参数。递进关系。误以为 ZeRO-2 分参数、ZeRO-3 分 state 等会算错显存。口诀："1 state, 2 +grad, 3 +params"。

> [!warning] 误区 5：FSDP 与 ZeRO-3 不同
> 本质相同。[[Fully Sharded Data Parallel]] 是 PyTorch 对 ZeRO-3 的原生实现。两者机制一致（参数分片 + 前向/反向 all-gather）。可互换理解。DeepSpeed 用 ZeRO，PyTorch 用 FSDP，API 不同但原理同。

> [!warning] 误区 6：ZeRO 让单卡训练大模型
> 不完全。ZeRO 是多卡分片，单卡无意义（N=1 时分片=全量）。单卡训大模型需 [[offloading (CPU-NVMe)]]（ZeRO-Offload 把 state 卸 CPU）+ 量化 + checkpointing。ZeRO 是多卡技术。

> [!warning] 误区 7：ZeRO-3 的 all-gather 不阻塞
> 阻塞。前向/反向到某层需 all-gather 聚合该层参数，通信完成才能计算。无 [[overlap strategy|overlap]] 时通信串行阻塞计算，throughput 降。需 overlap（通信与上一层计算重叠）或 NVLink 高带宽缓解。


## 8. 延伸细节

### 8.1 ZeRO-Offload

ZeRO-Offload：ZeRO-2 的 optimizer state 卸到 CPU 内存，optimizer step 在 CPU 算（CPU Adam）。减 GPU 显存（state 大头），利用 CPU 大内存。代价：CPU Adam 慢 + PCIe 传输 gradient。适合单卡或少数卡训大模型（如 10GB 卡训 7B）。

### 8.2 ZeRO-Infinity

ZeRO-Infinity：进一步把 state/gradient/参数卸到 NVMe（不止 CPU）。GPU 显存不够时用 NVMe 暂存，按需 all-gather 到 GPU。让超大模型（100B+）在有限 GPU 训练。代价：NVMe 带宽远低于 GPU（GB/s vs TB/s），throughput 大降，主要用于显存极紧场景。

### 8.3 ZeRO 与 3D parallel 的组合

大模型训练（如 70B+）用 3D parallel = DP(ZeRO-3) + TP + PP：

- DP(ZeRO-3)：数据维度分片，跨节点（IB）。
- TP：层内切分，节点内 NVLink。
- PP：层间切分，跨节点或节点内。

组合需精细调（每维大小、通信模式）。Megatron-DeepSpeed 是代表实现。

### 8.4 ZeRO-3 的 prefetch overlap

ZeRO-3 的 all-gather 参数可与计算 overlap：当前层计算时，prefetch 下一层的 all-gather。隐藏通信。需 CUDA stream + NCCL 异步。[[overlap strategy]] 的应用。vLLM/Megatron 实现此优化。

### 8.5 ZeRO 与混合精度

混合精度（权重/gradient fp16）+ ZeRO（分片）叠加：显存进一步减。但 ZeRO-3 的 all-gather 参数按 fp16（减通信量），optimizer state 仍 fp32 分片。组合：混合精度 + ZeRO-2 + [[gradient checkpointing]] 是 7B 训练标配。

### 8.6 ZeRO 的实现

- **DeepSpeed**：`deepspeed.DeepSpeedEngine`，配置 `zero_optization.stage`。
- **PyTorch FSDP**：`torch.distributed.fsdp.FullyShardedDataParallel`，原生 ZeRO-3。
- **FSDP2**（新版）：改进的分片，per-parameter 而非 per-module，更细粒度。

工程上 FSDP 更主流（PyTorch 原生），DeepSpeed 功能更全（Offload/Infinity）。

### 8.7 ZeRO-3 与 activation 的关系

ZeRO 分片的是参数相关（state/grad/权重），**不分片 activation**。activation 仍每卡各算各 batch 的（数据并行）。减 activation 需 [[gradient checkpointing]] 或 sequence parallelism。ZeRO + checkpointing 正交叠加。

### 8.8 ZeRO 的通信瓶颈

ZeRO-3 增 all-gather 参数通信。跨节点（IB 带宽低）时易 [[communication bottleneck]]。缓解：

- 节点内 ZeRO-3 + 跨节点 DP（减少跨节点 all-gather）。
- 参数 prefetch overlap。
- ZeRO-1/2（无 all-gather 参数）跨节点，ZeRO-3 节点内。

按互连带宽选阶段。

---
相关: [[优化技术]] | [[optimizer state memory]] | [[gradient memory]] | [[activation memory]] | [[Fully Sharded Data Parallel]] | [[Data Parallel]] | [[offloading (CPU-NVMe)]] | [[gradient checkpointing]] | [[Tensor Parallel]] | [[Pipeline Parallel]] | [[3D parallelism]] | [[overlap strategy]] | [[communication bottleneck]] | [[混合精度]] | [[NCCL通信拓扑]] | [[rank与world size]]
