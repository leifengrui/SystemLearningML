# offloading (CPU-NVMe)

> **所属章节**: [[优化技术]]
> **所属模块**: [[11-显存与内存系统]]
> **别名**: offloading / 卸载 / CPU offload / NVMe offload / ZeRO-Offload / ZeRO-Infinity
> **难度**: 中（需懂 [[ZeRO (DeepSpeed)]] + 显存层次 + PCIe/NVMe 带宽）


## 1. 一句话定义

**offloading（卸载）** 是把训练显存的某些组成（[[optimizer state memory|optimizer state]] / [[gradient memory|gradient]] / 权重 / [[activation memory|activation]]）**从 GPU 显存卸到 CPU 内存或 NVMe SSD**，用 CPU 内存/NVMe 的大容量换 GPU 显存的稀缺容量，代价是 **PCIe/NVMe 带宽传输开销**（throughput 降）。它是显存优化的"最后手段"——当 [[ZeRO (DeepSpeed)]] 分片 + [[gradient checkpointing]] 重算 + 混合精度仍不够时，用 offloading 让超大模型（如 70B+）在有限 GPU 训练。代表实现：**ZeRO-Offload**（state 卸 CPU，CPU Adam）、**ZeRO-Infinity**（state/gradient/参数卸 NVMe）。offloading 的核心矛盾：CPU/NVMe 带宽远低于 GPU 显存（PCIe 32GB/s vs HBM 2TB/s，NVMe 3GB/s），传输是 [[CPU-GPU pipeline stall]]/[[memory bottleneck]] 风险源，需 prefetch overlap 缓解。

> [!note] 三句话定位
> - **是什么**：把显存组成卸到 CPU 内存/NVMe，用容量换显存，代价是带宽传输。
> - **为什么需要**：ZeRO+checkpointing 仍不够时，让超大模型在有限 GPU 训练（70B+ 单卡/少卡）。
> - **代价**：PCIe/NVMe 带宽远低于 HBM，传输慢，throughput 大降。需 prefetch overlap + 选择性卸载（卸大头）。


## 2. 为什么需要它（动机与背景）

### 2.1 显存层次与容量/带宽 trade-off

| 存储 | 容量 | 带宽 | 延迟 | 成本 |
|---|---|---|---|---|
| **GPU HBM** | 40-80 GB | 2-3 TB/s | ~ns | 极贵 |
| **CPU 内存** | 100s GB - TB | 32-100 GB/s (PCIe) | ~100ns | 中 |
| **NVMe SSD** | TB 级 | 3-7 GB/s | ~10μs | 便宜 |

GPU HBM 容量小但带宽高（快），CPU 内存容量大但带宽低 30-60×，NVMe 更大但带宽低 300×+。offloading 用低带宽大容量存储换 GPU 显存，容量够但传输慢。

### 2.2 ZeRO + checkpointing 的极限

7B 训练（8卡，ZeRO-3 + FA + ckpt）每卡 ~15GB，A100 40GB 够。但 70B：

| 配置 | 70B 8卡 每卡 |
|---|---|
| ZeRO-3 | 109 GB（超 A100 80GB） |
| ZeRO-3 + offload opt | 39 GB |
| ZeRO-3 + offload opt+param | 22 GB |

70B 8卡 ZeRO-3 仍超 A100，需 offload state（甚至参数）才能装下。offloading 是大模型在有限 GPU 训练的必需。

### 2.3 卸载什么

按显存占比和访问频率选：

- **optimizer state（首选）**：占大头（56GB for 7B Adam），访问频率低（每 step optimizer step 一次），卸载代价相对小。ZeRO-Offload 的核心。
- **gradient**：中等，访问频率中（每 step 反向 + allreduce）。可卸但传输频繁。
- **参数**：访问频率高（每层前向/反向），卸载代价大（all-gather 每层）。ZeRO-Infinity 才卸。
- **activation**：访问频率高（反向每层），卸载代价大。[[gradient checkpointing]] 的重算常比卸载优（算力 vs 带宽）。

优先卸大头 + 低频访问的 state。

### 2.4 CPU 卸载 vs NVMe 卸载

- **CPU 内存（ZeRO-Offload）**：state 卸 CPU，CPU 算 Adam（CPU 不闲置）。PCIe 32GB/s 传输。适合 state 大头（7B-30B）。
- **NVMe（ZeRO-Infinity）**：state/gradient/参数卸 NVMe。NVMe 3-7GB/s 传输（更慢）。适合超大模型（100B+）CPU 内存也不够。

NVMe 带宽极低（3GB/s），throughput 损失大，主要用于显存极紧的"能跑就行"场景。

### 2.5 传输开销与 overlap

卸载需 GPU↔CPU/NVMe 传输。PCIe 32GB/s，7B state 56GB 传输需 1.75s（远超计算时间 ~141ms）。故 offloading 必须 **overlap**（传输与计算重叠）+ 只卸低频访问的 state（每 step 传一次而非每层）。否则 throughput 暴跌。

### 2.6 与 ZeRO/checkpointing 的叠加

offloading 是 ZeRO 的延伸（ZeRO-Offload = ZeRO-2 + state 卸 CPU）。叠加：

- [[ZeRO (DeepSpeed)]]：参数相关分片到 GPU。
- [[gradient checkpointing]]：activation 重算省显存。
- **offloading**：state/参数卸 CPU/NVMe。
- 混合精度：权重/gradient fp16。

四者叠加让 70B+ 在有限 GPU 训练。但每层优化都有代价（通信/重算/传输），throughput 累加下降。offloading 是最后的手段。


## 3. 核心概念详解

### 3.1 卸载对象的选择

| 对象 | 大小（7B） | 访问频率 | 卸载代价 | 是否卸 |
|---|---|---|---|---|
| optimizer state | 56 GB | 每 step 1 次 | 中（低频） | **首选卸** |
| gradient | 14 GB | 每 step（反向+allreduce） | 中 | 可卸 |
| 参数 | 14 GB | 每层（前向+反向） | 高（高频） | ZeRO-Infinity 才卸 |
| activation | 4-47 GB | 每层（反向） | 高 | 用 checkpointing 更优 |

### 3.2 ZeRO-Offload

ZeRO-Offload（DeepSpeed）：ZeRO-2 的 optimizer state 卸到 CPU 内存，optimizer step 在 CPU 算（CPU Adam）。

- 显存：GPU 省 state（56GB→0 for 7B），CPU 存 state。
- 计算：CPU 算 Adam（CPU 有大内存 + 多核，Adam 是 element-wise 适合 CPU）。
- 传输：每 step GPU→CPU 传 gradient（14GB），CPU→GPU 传更新参数（14GB）。PCIe 32GB/s → 0.9s 传输。
- overlap：gradient 传输与反向计算 overlap，参数传输与下一 step 前向 overlap。

适合 7B-30B 单卡/少卡训练（CPU 内存够存 state）。

### 3.3 ZeRO-Infinity

ZeRO-Infinity（DeepSpeed）：state + gradient + 参数都卸到 NVMe（不止 CPU）。

- 显存：GPU 几乎只存 activation（+ 当前层参数），超大模型（100B+）也能跑。
- 传输：NVMe 3-7GB/s，传输极慢。需 prefetch + 多级缓存（NVMe→CPU→GPU）。
- throughput：大降（主要时间在传输），"能跑就行"。

适合 100B+ 在有限 GPU（如 8×A100 80GB 训 175B）。

### 3.4 CPU Adam

ZeRO-Offload 的 CPU Adam：optimizer step 在 CPU 算（不传 state 回 GPU，只传 gradient 到 CPU 和参数回 GPU）。CPU 大内存存 state，多核并行算 Adam（element-wise 操作适合 CPU）。减少 state 传输（state 留 CPU，只传 gradient 和参数）。

### 3.5 prefetch overlap

offloading 的传输与计算 overlap：

- 反向算 layer $i$ 时，传 layer $i-1$ 的 gradient 到 CPU（CPU 算 Adam）。
- 前向算 layer $i$ 时，prefetch layer $i+1$ 的参数从 CPU/NVMe 到 GPU。

隐藏传输延迟。需 CUDA stream + NCCL/PCIe 异步。overlap 效率高时传输几乎隐藏，低时暴露成 stall。

### 3.6 与其他显存优化的对比

| 技术 | 优化对象 | 代价 | 适用 |
|---|---|---|---|
| [[ZeRO (DeepSpeed)]] | state/grad/参数（分片） | ZeRO-3 增通信 | 多卡 |
| [[gradient checkpointing]] | activation | FLOP +33% | 长序列 |
| 混合精度 | 权重/grad/state | 精度损失 | 通用 |
| **offloading** | state/参数（卸载） | **带宽传输** | 显存极紧 |


## 4. 数学原理 / 公式

### 4.1 卸载后 GPU 显存

$$
M_{\text{GPU}} = \underbrace{(1-\alpha_w)V_w}_{\text{未卸权重}} + \underbrace{(1-\alpha_g)V_g}_{\text{未卸梯度}} + \underbrace{(1-\alpha_o)V_{\text{opt}}}_{\text{未卸state}} + V_{\text{act}}
$$

$\alpha \in [0,1]$ 是卸载比例。全卸 state（$\alpha_o=1$）+ ZeRO-3 分参数：

$$
M_{\text{GPU, full offload}} = \frac{V_g}{N} + V_{\text{act}}
$$

7B 8卡：$1.75 + 4.3 = 6$ GB/卡（state+参数全卸）。

### 4.2 传输时间

$$
T_{\text{transfer}} = \frac{\alpha_w V_w + \alpha_g V_g + \alpha_o V_{\text{opt}}}{\text{BW}_{\text{PCIe/NVMe}}}
$$

PCIe 32GB/s，7B state 56GB → $T = 1.75$s。NVMe 3GB/s → $T = 18.7$s。

### 4.3 step 时间

无 overlap：

$$
T_{\text{step}} = T_{\text{compute}} + T_{\text{transfer}}
$$

有 overlap（$\eta$ 重叠效率）：

$$
T_{\text{step}} = \max(T_{\text{compute}}, T_{\text{transfer}} \cdot (1-\eta)) + T_{\text{transfer}} \cdot \min(1,\eta) \cdot \mathbf{1}
$$

7B：compute ~141ms，transfer 1750ms。即使 $\eta=0.9$，暴露 175ms，step = 316ms（compute 141 + 暴露 175），throughput 减半。offloading 代价显著。

### 4.4 70B 实例

70B 8卡 ZeRO-3 + offload state+param：

- GPU 显存：$g/8 + \text{act} = 17.5 + 4.3 = 22$ GB/卡（A100 40GB 够）。
- 传输：state 560GB + 参数 280GB（前向+反向）/PCIe 32GB/s = 26s/step。
- throughput：极低（26s/step vs 计算 ~1.4s），offloading 让 70B 能跑但慢。

### 4.5 NVMe 的极限

NVMe 3GB/s，70B state 560GB → 187s/step 传输。throughput 灾难性。ZeRO-Infinity 用多 NVMe 并行（4-8 卡 NVMe × 3GB/s = 12-24GB/s）+ prefetch，但仍远慢于 GPU 计算。仅"能跑"场景。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟卸载显存 + 传输

```python
def train_mem_with_offload(n, offload_opt=False, offload_param=False,
                           n_gpu=8, use_ckpt=True):
    """ZeRO-3 + 可选 offload state/参数. 返回 GPU 显存."""
    w = n*2; g = n*2; o = n*8  # fp16 权重/grad, fp32 Adam state
    a = 4.3e9 if use_ckpt else 47e9
    w_gpu = 0 if offload_param else w/n_gpu
    g_gpu = g/n_gpu
    o_gpu = 0 if offload_opt else o/n_gpu
    return w_gpu + g_gpu + o_gpu + a

print('7B 8卡 ZeRO-3 + offload:')
print(f'  无 offload:          {train_mem_with_offload(7e9)/1e9:.1f} GB/卡')
print(f'  +offload opt:        {train_mem_with_offload(7e9,True)/1e9:.1f} GB/卡')
print(f'  +offload opt+param:  {train_mem_with_offload(7e9,True,True)/1e9:.1f} GB/卡')
print(f'70B offload opt+param: {train_mem_with_offload(70e9,True,True)/1e9:.1f} GB/卡')

def offload_transfer(n, offload_opt, offload_param, bw=32e9):
    """每 step 传输量 / 带宽 = 传输时间."""
    t = 0
    if offload_opt: t += n*8          # state 传输 (CPU Adam)
    if offload_param: t += n*2*2      # 参数 all-gather (前向+反向)
    return t / bw

print(f'7B offload 传输 (PCIe 32GB/s): {offload_transfer(7e9,True,True)*1e3:.0f}ms')
print(f'70B offload 传输 (PCIe):       {offload_transfer(70e9,True,True)*1e3:.0f}ms')
print(f'70B offload 传输 (NVMe 3GB/s): {offload_transfer(70e9,True,True,3e9)*1e3:.0f}ms')
```

### 5.2 overlap 效果

```python
def step_time(t_compute, t_transfer, eta=0.8):
    """overlap: 传输的 (1-eta) 暴露."""
    exposed = t_transfer * (1 - eta)
    return t_compute + exposed

tc = 0.141  # 7B compute
tt = offload_transfer(7e9, True, True)  # 2.6s
print(f'无 overlap: {step_time(tc, tt, 0)*1e3:.0f}ms')
print(f'overlap 0.8: {step_time(tc, tt, 0.8)*1e3:.0f}ms (暴露 {tt*0.2*1e3:.0f}ms)')
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[ZeRO (DeepSpeed)]]（offload 是 ZeRO 的延伸）、显存层次（HBM/CPU/NVMe）、PCIe/NVMe 带宽。
- **下游（应用）**: ZeRO-Offload/ZeRO-Infinity（实现）、大模型少卡训练（70B+ 单卡/少卡）、[[CPU-GPU pipeline stall]]（传输是 stall 源）、[[memory bottleneck]]（带宽瓶颈）、[[overlap strategy]]（prefetch overlap）、[[gradient checkpointing]]（叠加，activation 用重算不用卸载）。
- **对比 / 易混**:
  - **offloading vs [[ZeRO (DeepSpeed)]]**：ZeRO 分片到 GPU（多卡），offload 卸到 CPU/NVMe（异构存储）。ZeRO 用 NVLink 通信（快），offload 用 PCIe/NVMe（慢）。ZeRO 先用，不够再 offload。
  - **offloading vs [[gradient checkpointing]]**：前者用 CPU/NVMe 存（带宽换显存），后者用重算（算力换显存）。activation 用 checkpointing 更优（算力便宜），state 用 offload（算不动，存 CPU）。trade 带宽 vs 算力。
  - **CPU offload vs NVMe offload**：CPU 内存带宽 32GB/s（PCIe），NVMe 3-7GB/s。CPU 快但容量小，NVMe 慢但容量大。state 卸 CPU，参数卸 NVMe（ZeRO-Infinity）。


## 7. 常见误区与易错点

> [!warning] 误区 1：offloading 不慢
> 慢。PCIe 32GB/s 远低于 HBM 2TB/s（60× 慢），NVMe 3GB/s 更慢（600×）。传输是主要开销，throughput 大降。offloading 是"显存换 throughput"，不是免费午餐。只在显存不够时用。

> [!warning] 误区 2：卸全部显存组成
> 应卸大头 + 低频访问的。optimizer state（大头、每 step 1 次）首选卸；参数（高频、每层）卸代价大，只在 ZeRO-Infinity 才卸。盲目卸全部会让每层都传输，throughput 灾难。

> [!warning] 误区 3：CPU 卸载 = CPU 训练
> 不完全。CPU 卸载是 CPU **存** state/参数，GPU 仍算前向/反向。ZeRO-Offload 的 CPU Adam 是 CPU 也算 optimizer step（element-wise 适合 CPU），但前向/反向仍在 GPU。不是全部训练在 CPU（CPU 算 matmul 太慢）。

> [!warning] 误区 4：offloading 与 ZeRO 冲突
> 不冲突，offload 是 ZeRO 的延伸。ZeRO-Offload = ZeRO-2 + state 卸 CPU。ZeRO-Infinity = ZeRO-3 + state/grad/参数卸 NVMe。叠加使用，递进。

> [!warning] 误区 5：activation 用 offload 优化
> 不推荐。activation 访问频率高（每层反向），卸载传输代价大。[[gradient checkpointing]] 重算更优（算力便宜 vs 带宽贵）。activation 用 checkpointing，state 用 offload，分工。

> [!warning] 误区 6：NVMe offload 性能可接受
> 不可接受（for throughput）。NVMe 3GB/s，70B state 560GB → 187s/step 传输。throughput 灾难。ZeRO-Infinity 主要用于"能跑就行"（显存极紧、不在乎速度），如调试/小规模实验。生产训练用多 GPU + ZeRO，不用 NVMe offload。

> [!warning] 误区 7：忽略 prefetch overlap
> offloading 不 overlap 时传输串行阻塞计算，throughput 暴跌。必须 prefetch（当前层计算时取下一层数据）+ 异步传输。overlap 效率低（$\eta < 0.8$）时 offloading 几乎不可用。


## 8. 延伸细节

### 8.1 ZeRO-Offload 的 CPU Adam

CPU Adam 的优势：

- Adam 是 element-wise 操作（$m, v, \theta$ 逐参数），适合 CPU 多核并行。
- CPU 大内存存 state（无需传输 state，只传 gradient 和参数）。
- CPU Adam 与 GPU 反向 overlap（GPU 反向算下一层时，CPU 算上一层的 Adam）。

DeepSpeed 实现了优化的 CPU Adam（多线程 + SIMD），速度接近 GPU Adam（在 element-wise 场景）。

### 8.2 ZeRO-Infinity 的多级缓存

NVMe 带宽极低，ZeRO-Infinity 用多级缓存：

- NVMe → CPU 内存（ prefetch，大块读）。
- CPU 内存 → GPU 显存（按层聚合，小块读）。
- GPU 显存 → 计算（寄存器/smem）。

多级 prefetch overlap 隐藏延迟。但仍远慢于纯 GPU。

### 8.3 offloading 的适用场景

- **单卡/少卡训大模型**：7B 单卡（offload state 到 CPU 内存）、70B 少卡（offload state+参数）。
- **显存极紧的调试/实验**：跑通优先，速度次要。
- **CPU 内存富余**：服务器 CPU 内存 TB 级，可存大模型 state。

不适合：

- **生产大规模训练**：多 GPU + ZeRO + TP/PP 更快，不用 offload。
- **throughput 敏感**：offload 降 throughput 显著。

### 8.4 与 8-bit optimizer 的叠加

8-bit Adam（state int8）+ offload：state 减 4×（56→14GB for 7B），CPU 存更省，传输量减。是显存优化的叠加组合。QLoRA（4-bit 参数 + 8-bit Adam + offload）让单卡微调 70B 可行。

### 8.5 PCIe 带宽的限制

PCIe Gen4 32GB/s，Gen5 64GB/s。offloading 受 PCIe 带宽限。多 GPU 共享 PCIe 带宽（一棵 CPU 多 GPU 抢带宽），实际 effective 带宽更低。NVLink 不连 CPU（CPU 只走 PCIe），故 offloading 无法用 NVLink 加速。

### 8.6 offloading 与 CPU-GPU pipeline stall

offloading 的传输是 [[CPU-GPU pipeline stall]] 风险源：传输未完成时 GPU 等数据 idle。缓解：

- prefetch（提前取下一层）。
- 异步传输（CUDA stream + pinned memory）。
- overlap（传输与计算重叠）。

overlap 效率低时 stall 严重，throughput 大降。需 nsys 测传输/计算时间轴。

### 8.7 activation offload 的少见

理论上 activation 也可卸 CPU（前向存 CPU，反向取）。但 activation 访问高频（每层反向）+ 带宽贵，[[gradient checkpointing]] 重算常更优（算力便宜）。故 activation 用 checkpointing，不用 offload。少数场景（重算也撑不住、CPU 内存极大）才 activation offload。

### 8.8 offloading 的未来

GPU 显存增大（H100 80GB，B200 192GB）+ NVLink C2C（CPU-GPU 互连提带宽）让 offloading 代价降低。但大模型增长更快，offloading 仍是显存不够时的手段。趋势：更大显存 + 更高带宽 + ZeRO/TP 为主，offloading 为辅（极端场景）。

---
相关: [[优化技术]] | [[ZeRO (DeepSpeed)]] | [[optimizer state memory]] | [[gradient memory]] | [[activation memory]] | [[gradient checkpointing]] | [[CPU-GPU pipeline stall]] | [[memory bottleneck]] | [[overlap strategy]] | [[Fully Sharded Data Parallel]] | [[混合精度]] | [[GPU memory snapshot]] | [[nsys]]
