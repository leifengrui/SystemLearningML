# benchmark与regression test

> **所属章节**: [[测试]]
> **所属模块**: [[20-可靠性与开源工程]]
> **别名**: perf benchmark / 性能基准 / regression test / 性能回归守护 / MFU 测试 / 吞吐测试 / 延迟测试 / flaky 治理
> **难度**: 中（需懂 [[torch profiler]]、[[nsys (Nsight Systems)]]、GPU 时钟、benchmark 统计、CI 流程）


## 1. 一句话定义

**benchmark 与 regression test** 是 ML 性能测试的双支柱——**benchmark（性能基准）** 测速度/吞吐/显存（如测一个 attention kernel 的 **TFLOPS**、测端到端训练的 **MFU**（Model FLOPs Utilization，模型算力利用率）、测推理 server 的 **QPS/延迟**、测通信的 **总线带宽**）；**regression test（性能回归守护）** 在代码改动后跑 benchmark 对比**历史基线**，若性能退化超阈值则报警/拦截 PR——是性能的 CI 守护，防止"功能对但慢了"的回归偷偷进主分支；与功能测试的核心区别——**功能测对错**（数值/形状）、**性能测快慢**（吞吐/延迟/显存），两者正交（功能对但慢也是 bug）；**ML 性能 benchmark 的难点**——**GPU 时钟波动**（thermal throttling、boost 频率漂移）、**batch/seq/padding 影响**（变长使结果不稳）、**需多次平均 + 预热**（warmup 稳定 cache/时钟后测，多次 run 取中位/均值+方差）；**性能基线管理**（记录历史最优，CI 跑关键 benchmark 对比，退化超 X% 报警）；**perf benchmark 框架**——[[torch profiler]]（PyTorch 算子级）、**nccl-tests**（通信带宽）、**llmperf**（LLM 推理吞吐）；**flaky 性能测试治理**——性能测试天然抖动，需多次取中位、固定环境（时钟锁、独占 GPU）、容忍阈值（退化 <5% 不报警），否则 CI 噪音淹没真信号。

> [!note] 三句话定位
> - **是什么**：benchmark 测速度/吞吐/显存（TFLOPS/MFU/QPS/带宽）；regression test 跑 benchmark 对比基线，性能退化报警——性能的 CI 守护。
> - **为什么**：ML 框架代码改动（kernel/通信/调度）易引入隐性性能退化，功能测试抓不到，需专门的性能回归守护。
> - **难点**：GPU 时钟波动、batch/seq 影响、需多次平均+预热、flaky 治理——不像功能测试那么确定，要统计与基线管理。


## 2. 为什么需要它（动机与背景）

### 2.1 性能退化是隐性 bug

ML 框架迭代快——改一个 kernel、调通信 overlap、动 scheduler 逻辑——**功能可能仍对**（数值测过），但**性能可能退化**（如 fused kernel 拆回 unfused、通信 overlap 失效变成串行、padding 增加）。这种退化功能测抓不到，只在生产规模化时暴露（训练慢 20%、吞吐降），**返工成本极高**。需在 CI 阶段就跑 benchmark 守护，退化超阈值拦截。

### 2.2 GPU 性能的不可重复性

- **时钟波动**：GPU boost 频率随温度/功耗变，同代码跑两次 TFLOPS 差 ~5-10%。
- **cache 抖动**：L2/SM cache 命中率随调度变。
- **batch/seq 变长**：变长模型（LLM）的 batch/seq 每次不同，FLOPs 变，吞吐不稳。
- **多租户干扰**：共享 GPU 上别的 job 抢资源。
- 故性能测试**必须多次取统计量**（中位/均值+方差/置信区间），不能单次定论。

### 2.3 性能基线的必要性

无基线则无"退化"概念——每次跑的数 vs 历史最优（或上次主分支），退化才知。CI 维护性能基线库（每 benchmark 的历史值 + 容差范围），新跑值对比，超范围报警。

### 2.4 ML benchmark 比软件 benchmark 难

- 软件性能测（如 HTTP server QPS）：CPU 时钟稳定，单次复现性高。
- ML 性能测（GPU kernel/训练吞吐）：GPU 时钟波动 + cache + 变长 + 算子选择（cuDNN 选算法），复现性低，需更多次平均 + 统计处理。


## 3. 核心概念详解

### 3.1 benchmark 的三类指标

| 指标 | 定义 | 例 |
|---|---|---|
| **吞吐**（throughput） | 单位时间处理量 | 训练 tokens/s、推理 requests/s、batch/s |
| **延迟**（latency） | 单次请求耗时 | 首 token 延迟（TTFT）、每 token 延迟（TPOT）、step 时间 |
| **显存**（memory） | 峰值占用 | 训练峰值显存、KV cache 大小、激活显存 |
| **算力利用率** | 实际 FLOPs / 峰值 FLOPs | MFU（训练）、HFU（含 recomputation）、GPU 利用率 |

### 3.2 MFU（Model FLOPs Utilization）

$$\text{MFU} = \frac{\text{实际训练 FLOPs}}{\text{GPU 峰值 FLOPs} \times \text{时间}}$$

- 实际 FLOPs：模型 forward + backward 的理论 FLOPs（如 GPT forward ~$2\times$params×tokens，backward ~$4\times$，总 ~$6\times$params×tokens）。
- GPU 峰值：A100 BF16 ~312 TFLOPS、H100 BF16 ~989 TFLOPS。
- MFU 50-60% 算优秀（通信/调度/算子间隙有损耗）。MFU <30% 说明有瓶颈（通信未 overlap、kernel 低效、padding 浪费）。

### 3.3 TFLOPS（kernel 级 benchmark）

单个 kernel 的算力：$\text{TFLOPS} = \frac{\text{kernel FLOPs}}{\text{kernel 耗时 (s)} \times 10^{12}}$。用于评估 kernel 效率（如 attention kernel vs cuDNN）。需精确测 kernel 耗时（CUDA event，不是 wall clock，详见 §3.6）。

### 3.4 regression test 的流程

```
1. 基线建立: 主分支跑关键 benchmark, 记录历史值 + 容差 (如 MFU 55% ± 2%)
2. PR 触发: PR 分支跑同样 benchmark
3. 对比: 新值 vs 基线, 退化超阈值 (如 -5%) -> 报警/拦截
4. 人工/自动: 退化合理 (功能改动) 更新基线; 退化不合理 -> 修代码
```

### 3.5 benchmark 的统计处理

单次跑值不可靠，需：
- **预热**（warmup）：先跑几次稳 cache/时钟，弃数据。
- **多次 run**：跑 N 次（如 10-100），取**中位**（抗离群点）或**均值 ± 标准差**。
- **置信区间**：$\bar{x} \pm t_{\alpha/2}\cdot\sigma/\sqrt{n}$，看退化是否统计显著。
- **离群点剔除**：如 >3σ 的 run 剔除（thermal throttling 干扰）。

### 3.6 GPU 计时（CUDA event vs wall clock）

```python
import torch
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
out = model(x)  # kernel 异步提交
end.record()
torch.cuda.synchronize()  # 等完成
print(start.elapsed_time(end), "ms")  # GPU 时间 (准确, 不含 host 端 gap)
# wall clock (time.perf_counter) 含 host 端 launch gap, 不准
```

- **CUDA event**：测 GPU 上 kernel 实际耗时（不含 host 端 launch 间隙），准确。
- **wall clock**：含 host launch gap，测端到端可用但单 kernel 不准。
- 必 `cuda.synchronize()` 后读，否则 event 还没 record。

### 3.7 时钟波动与锁定

- **boost 频率漂移**：GPU boost 到最高频后 thermal throttling 降频，跑中变化。
- **锁定**：`nvidia-smi -lgc <min>,<max>`（锁 graphics clock）、`nvidia-smi -pm 1`（persistence mode）、`nvidia-smi --query-gpu=clocks.gr --format=csv`（查）。
- CI 跑前锁定时钟 + 预热，减波动。生产环境不一定锁（用户自便）。

### 3.8 perf benchmark 框架

| 框架 | 测什么 | 用途 |
|---|---|---|
| [[torch profiler]] | PyTorch 算子级 trace + GPU kernel 耗时 | 单算子/训练 step 的 profile |
| [[nsys (Nsight Systems)]] | 系统级 trace（CPU+GPU+NCCL） | 端到端瓶颈定位 |
| **nccl-tests** | NCCL 通信带宽/延迟 | all-reduce/all-gather 带宽测 |
| **llmperf** | LLM 推理吞吐/延迟 | 端到端推理 benchmark |
| **Megatron-LM benchmark** | 训练 MFU/step time | 训练框架性能 |
| **vLLM benchmark** | 推理吞吐/延迟 | 推理引擎性能 |

### 3.9 flaky 性能测试治理

- **现象**：同一代码同一输入，两次 run 差 >10%，CI 时过时不过。
- **根**：时钟波动、cache 抖动、多租户、变长。
- **治**：① 多次取中位（抗离群）；② 锁时钟 + 独占 GPU；③ 放宽容差（退化 <5% 不报警）；④ 剔除离群 run（>3σ）；⑤ 标 `@pytest.mark.flaky` 重试 N 次；⑥ 持续 flaky 的从 CI 移到 nightly（不阻塞 PR）。


## 4. 数学原理 / 公式

### 4.1 MFU 的推导

设模型参数量 $P$，序列长 $L$，batch $B$，训练一步（forward+backward）FLOPs $\approx 6PBL$（GPT 类，2×forward + 4×backward，稠密）。GPU 峰值 FLOPs $F_{\text{peak}}$，一步耗时 $t$：

$$\text{MFU} = \frac{6PBL}{F_{\text{peak}} \cdot t}$$

- $P=7\text{B}$, $L=2048$, $B=32$, $F_{\text{peak}}=312\text{TFLOPS}$（A100 BF16），$t=1\text{s}$：
  $\text{FLOPs}=6\times7\times10^9\times2048\times32\approx 2.75\times10^{15}=2750\text{TFLOPS}$。
  $\text{MFU}=2750/(312\times1)\approx 88\%$（不现实，实际 ~50-60% 因通信/padding/算子间隙）。
- MFU 上限受通信/算子间隙/padding 限制，实际 50-60% 是优秀。

### 4.2 均值、方差、置信区间

$N$ 次测量 $x_1,\ldots,x_N$：

$$\bar{x} = \frac{1}{N}\sum x_i, \quad s = \sqrt{\frac{1}{N-1}\sum(x_i-\bar{x})^2}$$

95% 置信区间：

$$\bar{x} \pm t_{0.025,N-1}\cdot\frac{s}{\sqrt{N}}$$

- $s$ 大（波动大）→ 区间宽，退化判定不可靠。
- 增大 $N$ 缩窄区间（$\propto 1/\sqrt{N}$），$N=10$ 到 $N=100$ 区间缩 3.16x。
- 退化判定：新基线均值 $\bar{x}_{\text{new}}$ vs 旧 $\bar{x}_{\text{old}}$，若 $\bar{x}_{\text{new}} < \bar{x}_{\text{old}} - t\cdot s/\sqrt{N}$（统计显著下降）→ 报警。

### 4.3 中位 vs 均值（抗离群点）

- 均值受离群点拉偏（一次 thermal throttle 低值拉低均值）。
- 中位抗离群（排序取中），更稳。性能 benchmark 常取中位 + IQR（四分位距）。
- 两者差异大 → 说明有离群点，需剔或多次。

### 4.4 通信带宽的 benchmark 数学

NCCL all-reduce 通信量：$\frac{2(N-1)}{N}\cdot D$（$N$ rank，$D$ 数据字节，ring 算法）。带宽 = $\frac{2(N-1)/N \cdot D}{\text{耗时}}$。

- 理论 NVLink 带宽 ~300 GB/s（A100），实测达 ~80% 算优。
- algbw（算法带宽）vs busbw（总线带宽）：busbw = algbw × 系数（反映实际链路利用率）。

### 4.5 warmup 与稳态

设前 $W$ 次 run 是 warmup（cache 冷、时钟未稳），数据弃；后 $M$ 次是稳态，取统计。$W$ 选 3-10（看稳态收敛），$M$ 选 10-100（看置信区间需求）。

### 4.6 退化的阈值设定

退化阈值 $\theta$（如 5%）需 > 噪声水平 $\sigma_{\text{noise}}$（时钟/cache 抖动），否则假阳。经验：$\theta \approx 2\sim3\times\sigma_{\text{noise}}$（如波动 $\sigma=2\%$，阈值设 5-6%）。阈值太严假阳多（CI 噪音），太松漏回归。


## 5. 代码示例（可选）

### 5.1 kernel 级 TFLOPS benchmark（CUDA event + 预热 + 多次中位）

```python
import torch
import statistics

def bench_kernel(fn, args, warmup=3, iters=20):
    torch.cuda.synchronize()
    # 预热 (稳 cache/时钟)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    # 计时 (CUDA event)
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*args)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))  # ms
    return statistics.median(times), statistics.stdev(times)

def matmul_tflops(M, N, K, ms):
    flops = 2 * M * N * K  # GEMM FLOPs
    return flops / (ms * 1e-3) / 1e12  # TFLOPS

med, std = bench_kernel(torch.mm, (torch.randn(4096, 4096, device="cuda"),
                                   torch.randn(4096, 4096, device="cuda")))
tflops = matmul_tflops(4096, 4096, 4096, med)
print(f"matmul: {med:.3f}ms ± {std:.3f}ms, {tflops:.1f} TFLOPS")
# matmul: ~2.1ms ± 0.05ms, ~66 TFLOPS (A100 BF16 峰值 312, MFU ~21% for 4096x4096)
```

### 5.2 端到端 MFU benchmark

```python
import torch
import statistics

def bench_train_mfu(model, dataloader, optim, params, seqlen, peak_tflops,
                    warmup=3, iters=10):
    model.train().cuda()
    times = []
    # 预热
    for _ in range(warmup):
        x, y = next(dataloader)
        x, y = x.cuda(), y.cuda()
        loss = model(x, y).loss
        loss.backward(); optim.step(); optim.zero_grad()
    torch.cuda.synchronize()
    # 计时
    for _ in range(iters):
        x, y = next(dataloader)
        x, y = x.cuda(), y.cuda()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        loss = model(x, y).loss
        loss.backward(); optim.step(); optim.zero_grad()
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))  # ms
    med = statistics.median(times)
    # FLOPs = 6 * params * batch * seqlen
    B = x.shape[0]
    flops = 6 * params * B * seqlen
    mfu = flops / (med * 1e-3) / 1e12 / peak_tflops
    return med, mfu

# A100 BF16 peak ~312 TFLOPS
med, mfu = bench_train_mfu(model, loader, optim, params=7e9, seqlen=2048,
                            peak_tflops=312)
print(f"step: {med:.1f}ms, MFU: {mfu*100:.1f}%")
```

### 5.3 regression test（对比基线）

```python
import json, statistics

BASELINE_FILE = "perf_baseline.json"  # 历史基线 (主分支跑的)

def load_baseline():
    with open(BASELINE_FILE) as f:
        return json.load(f)

def save_baseline(metrics):
    with open(BASELINE_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

def check_regression(name, new_val, baseline, threshold=0.05):
    old_val = baseline[name]["median"]
    std = baseline[name]["std"]
    deg = (old_val - new_val) / old_val  # 正=退化
    if deg > threshold:
        print(f"REGRESSION {name}: {new_val:.3f} vs baseline {old_val:.3f} "
              f"(退化 {deg*100:.1f}% > {threshold*100}%)")
        return False  # 拦截
    elif deg < -threshold:
        print(f"IMPROVEMENT {name}: +{-deg*100:.1f}%")
    return True

# CI 流程
baseline = load_baseline()
med, _ = bench_kernel(...)
if not check_regression("matmul_4k", med, baseline, threshold=0.05):
    exit(1)  # PR 拦截
```

### 5.4 多次取中位 + 离群剔除

```python
import statistics

def robust_bench(fn, args, runs=30, warmup=5, cutoff_sigma=3):
    # warmup
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(*args); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    # 剔离群 (>cutoff_sigma 标准差)
    med = statistics.median(times)
    mad = statistics.median([abs(t-med) for t in times])  # 中位绝对偏差
    cleaned = [t for t in times if abs(t-med) < cutoff_sigma * mad * 1.4826]
    return statistics.median(cleaned), statistics.stdev(cleaned)
```

### 5.5 pytest 标 flaky + mark

```python
import pytest

@pytest.mark.benchmark
@pytest.mark.flaky(reruns=3, reruns_delay=2)  # flaky 重试 3 次
def test_matmul_throughput():
    med, _ = bench_kernel(torch.mm, (torch.randn(4096,4096,device="cuda"),
                                     torch.randn(4096,4096,device="cuda")))
    tflops = matmul_tflops(4096,4096,4096,med)
    assert tflops > 50, f"matmul TFLOPS too low: {tflops}"  # 阈值下限
```

```ini
# pytest.ini
[pytest]
markers =
    benchmark: performance benchmark (slow)
    flaky: may fail sporadically, retry
addopts = -ra --strict-markers
```

跑：`pytest -m benchmark`（nightly）或 CI 单跑关键 benchmark。


## 6. 与其他知识点的关系

- **上游（依赖）**: [[torch profiler]]（算子级 trace）、[[nsys (Nsight Systems)]]（系统级 trace）、CUDA event、nccl-tests、统计方法
- **下游（应用）**: ML 框架 CI（vLLM/Megatron/Torch 的性能回归守护）、性能优化验证（优化前后对比）、[[训推系统]]的吞吐/延迟 benchmark
- **对比 / 易混**:
  - **benchmark vs 功能测试**：前者测快慢（吞吐/延迟/显存），后者测对错（数值/形状）。两者正交——功能对但慢也是 bug。
  - **benchmark vs [[数值一致性测试]]**：前者测性能，后者测数值正确性。优化 kernel 后要同时跑两者（快但对不对、对但快不快都要测）。
  - **regression test（性能）vs regression test（功能）**：前者对比性能基线防退化，后者（功能回归）跑历史功能测防逻辑退化。CI 都跑。
  - **MFU vs GPU 利用率**：MFU 是实际 FLOPs/峰值 FLOPs（算力利用率），GPU 利用率（nvidia-smi）是 SM 占用百分比。MFU 更准反映算力发挥，GPU 利用率高不代表 MFU 高（可能算子间隙被填充但实际算力低）。


## 7. 常见误区与易错点

> [!warning] 误区1：单次 benchmark 定论
> 错。GPU 时钟/cache/变长抖动，单次差 5-10%。必须多次取中位 + 预热 + 统计处理。

> [!warning] 误区2：用 wall clock 测 kernel
> 不准。wall clock 含 host launch gap（Python glue），单 kernel 计时用 CUDA event。端到端可用 wall clock 但要同步。

> [!warning] 误区3：不预热直接测
> 错。前几次 cache 冷、时钟未稳，数据偏低/不稳。warmup 3-10 次弃数据。

> [!warning] 误区4：退化阈值太严
> 假阳多。阈值需 > 噪声水平（$\sim 2\sim3\times\sigma$）。波动 2% 设阈值 5% 合理，设 1% 则 CI 噪音淹没信号。

> [!warning] 误区5：MFU 高就一定好
> 不全对。MFU 高说明算力发挥好，但若功能错（数值 bug）或显存爆（牺牲显存换算力）也不行。性能要综合 MFU + 显存 + 延迟。

> [!warning] 误区6：性能测试只跑 happy path
> 漏。要测多种 batch/seq/dtype 组合——短 seq 可能 bandwidth-bound、长 seq compute-bound，不同场景瓶颈不同，需分别基线。

> [!warning] 误区7：flaky 测试直接删
> 错。flaky 可能是真 bug 的间歇暴露。先治（多次取中位/锁时钟/剔离群），持续 flaky 移 nightly，但记录追踪。

> [!tip] 实践：性能 CI 三层
> ① PR 关键 benchmark（快速，单卡，防大退化）；② nightly 全 benchmark（多卡/多场景，建基线）；③ 性能优化 PR 跑全对比（前后 MFU/延迟）。三层覆盖。

> [!tip] 实践：锁时钟 + 独占
> CI 跑前 `nvidia-smi -lgc` 锁 graphics clock、`nvidia-smi -pm 1` persistence、独占 GPU（无多租户）。减时钟波动，复现性好。

> [!tip] 实践：基线版本化
> 性能基线存 git（或 CI artifact），每 benchmark 记 median+std+环境（GPU 型号/CUDA 版本/cuDNN）。换硬件/驱动时基线重置。退化对比同环境基线。


## 8. 延伸细节

### 8.1 GPU 利用率 vs MFU

- **GPU 利用率**（`nvidia-smi` 的 `utilization`）：SM 在某时刻是否有 kernel 跑（百分比）。高只说明"在跑"，不代表算力满（可能算子间隙被填充但 FLOPs 低）。
- **MFU**：实际 FLOPs/峰值 FLOPs。真反映算力发挥。
- 两者常背离——GPU 利用率 100% 但 MFU 20%（kernel 在跑但低效，或通信间隙被填充）。诊断用 [[torch profiler]]/[[nsys (Nsight Systems)]] 看实际算子占比。

### 8.2 通信 overlap 的 benchmark

DDP/TP/PP 的通信与计算 overlap 程序是性能关键。benchmark 测：通信耗时 vs 计算耗时，理想是通信完全藏在计算里（overlap 100%）。实测 overlap 率 = (计算 - (通信 - 隐藏部分)) / 计算。用 [[nsys (Nsight Systems)]] 看 NCCL stream 与计算 stream 的 timeline 重叠度。

### 8.3 变长模型的 benchmark 难点

LLM 推理的 batch/seq 变长（continuous batching、chunked prefill），每 step 的 FLOPs 变，吞吐不稳。benchmark 需：① 固定 workload（同 prompt 集）保复现；② 报 tokens/s（按实际 token 数）而非 requests/s；③ 测 P50/P99 延迟（不只均值，长尾重要）。llmperf 这么设计。

### 8.4 训推系统的端到端 benchmark

[[训推系统]] 的端到端：训练测 tokens/s + MFU + 显存 + step time；推理测 QPS + TTFT + TPOT + 显存（KV cache）。两者指标不同，benchmark 框架各异（Megatron 训练 benchmark、vLLM 推理 benchmark、llmperf 通用）。

### 8.5 显存 benchmark

显存是训练/推理的硬约束。benchmark 测峰值显存（`torch.cuda.max_memory_allocated()`），与理论值（激活+权重+优化器+梯度+KV）对比。退化（显存涨）也是回归——如重组件引入额外 buffer。CI 跑显存基线守护。

### 8.6 性能优化的验证流程

优化一个 kernel/通信 overlap 后：① 跑功能测（[[单元测试与分布式测试]] + [[数值一致性测试]]）保对；② 跑 benchmark（本条）保快；③ 对比基线（前后 MFU/延迟），确认显著提升。三步缺一——只跑 benchmark 不跑功能测，可能优化引入数值 bug。

### 8.7 与 [[单元测试与分布式测试]] / [[数值一致性测试]] 的关系

- **[[单元测试与分布式测试]]**：测逻辑/形状/切分对不对（功能对错）。
- **[[数值一致性测试]]**：测数值容差/跨设备一致性（数值合法性）。
- **本条**：测快慢（性能）。
- 三者构成 ML 测试体系——功能对（数值合法）+ 切分对 + 性能不退化，三者全过 PR 才合。性能优化要同时跑三者（快了对不对、对快不快都要测）。


---
相关: [[测试]] | [[单元测试与分布式测试]] | [[数值一致性测试]] | [[torch profiler]] | [[nsys (Nsight Systems)]] | [[torch.distributed]] | [[Distributed Data Parallel]] | [[数值类型与精度]] | [[mixed precision training]] | [[20-可靠性与开源工程]] | [[训推系统]]
