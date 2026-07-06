# asynchronous training

> **所属章节**: [[训练架构]]
> **所属模块**: [[06-训推系统]]
> **难度**: 中（需懂同步/异步 tradeoff + stale data 修正）

## 1. 一句话定义

**asynchronous training（异步训练）** 是 [[learner worker split]] 的**运行模式**：worker 采样与 learner 训练**并行流水线**，不互相等待——worker 持续采数据进 buffer，learner 持续从 buffer 取训，两者时间重叠。对比**同步训练**（worker 采完 batch 等 learner 训完再采，互相等、利用率低），异步**吞吐趋近 $\max(T_w, T_l)$**（重叠采训时间）。代价是 **stale data**：worker 用 $\theta_t$ 采的数据，learner 已更新到 $\theta_{t+k}$ 才消费，**staleness = $k$ 步**，导致 PPO ratio $\rho=\pi_{\theta_{t+k}}/\pi_{\theta_t}$ 失真（[[stale policy problem]]）。防法：[[weight sync mechanism]] 定期同步新 $\theta$ 给 worker + ratio clip / V-trace 修正。IMPALA（DeepMind 2018）是经典异步架构，RLHF 中 verl 支持同步/异步两种模式。

> [!note] 异步 vs 同步
> - **同步**：worker 采 → 等 learner 训 → 同步新 θ → worker 再采。利用率 < 1（快的等慢的），但无 stale data。
> - **异步**：worker 持续采进 buffer，learner 持续取训。吞吐高（重叠），但 stale data。
> 异步是 throughput 与 staleness 的 tradeoff。

## 2. 为什么需要它（动机与背景）

[[learner worker split]] 的同步模式有利用率问题：
1. **互相等**：worker 采完 batch 等 learner 训完，learner 训完等 worker 采完，GPU 互相闲置；
2. **利用率低**：快的等慢的，吞吐 = $\min(T_w, T_l)$，低于理论上限；
3. **大模型 RL 尤甚**：生成（worker）与训练（learner）耗时都长（秒级），互相等的浪费大；
4. **在线 RL 需持续采**：PPO 每轮需新 rollout，采训并行流水线才能匹配 RL 的数据需求。

异步训练解此：worker 与 learner 各自持续工作，通过 buffer 解耦时序，吞吐趋近 $\max(T_w, T_l)$。代价是 stale data，需修正。这是大模型 RL 提吞吐的关键模式。

## 3. 核心概念详解

### 3.1 同步 vs 异步的时间线

```
同步:
  worker: [采 batch1][等......][采 batch2][等......]
  learner:        [等][训 batch1][等......][训 batch2]
  ↑ 互相等,利用率低

异步:
  worker:  [采 b1][采 b2][采 b3][采 b4][采 b5]...   ← 持续采
  buffer:     ↓    ↓    ↓    ↓    ↓
  learner:    [训 b1][训 b2][训 b3][训 b4]...       ← 持续训
  ↑ 采训重叠,吞吐高;但 b2 用 θ_0 采,learner 训 b2 时已是 θ_1(stale)
```

### 3.2 stale data 与 staleness

- worker 在时刻 $t$ 用 $\theta_t$ 采数据 $D_t$；
- learner 已训练到 $\theta_{t+k}$ 才消费 $D_t$；
- **staleness = $k$ 步**（数据比训练滞后 $k$ 步）；
- $k$ 越大越 stale，ratio $\rho=\pi_{\theta_{t+k}}(a)/\pi_{\theta_t}(a)$ 偏离 1 越多；
- $k$ 太大 → PPO ratio 失真 → 训练不稳（[[stale policy problem]]）。

### 3.3 异步的防 stale 机制

| 机制 | 作用 | 详见 |
|---|---|---|
| **[[weight sync mechanism]]** | 定期把 learner 新 $\theta$ 同步给 worker，减小 $k$ | [[weight sync mechanism]] |
| **PPO ratio clip** | $\rho$ clip 在 $[1-\epsilon,1+\epsilon]$，限失真 | [[PPO clipped objective]] |
| **V-trace** | IMPALA 的多步 return + 截断 ratio，容忍 stale | §3.5 |
| **target_kl 早停** | 每 epoch 算 KL，超 target 则 break | [[KL penalty与KL control]] |
| **buffer 容量限制** | 旧数据过期丢弃，限制最大 $k$ | replay buffer 设计 |

### 3.4 异步的吞吐优势

- **同步**：吞吐 $=\min(T_w, T_l)$，利用率 $<1$；
- **异步**：吞吐 $\to\max(T_w, T_l)$（流水线重叠），利用率高；
- 当 $T_w\approx T_l$（生成与训练耗时近），异步约 2× 同步吞吐；
- 当 $T_w\ll T_l$ 或反之，异步优势小（瓶颈侧决定）。
RLHF 中生成（$T_w$）常是瓶颈且慢，异步让训练与生成重叠，吞吐提升显著。

### 3.5 IMPALA 与 V-trace

IMPALA（Importance Weighted Actor-Learner Architecture）是异步训练的经典：
- 多 actor（worker）并行采 trajectory，单 learner 异步训练；
- **V-trace**：用截断 ratio $\rho_{\bar\theta}=\min(\bar\rho, \pi_\theta/\pi_{\theta_{old}})$ 对 $V$ 做多步 return 估计，容忍一定 staleness；
- V-trace 是异步 stale data 的理论修正，使异步训练稳定。
RLHF 的异步 PPO 借鉴此思想（ratio clip + weight sync）。

## 4. 数学原理 / 公式

### 4.1 吞吐对比

记 worker 采样吞吐 $T_w$（trajectory/秒），learner 训练吞吐 $T_l$（batch/秒）。

- **同步**：每轮时间 $= 1/T_w + 1/T_l$，吞吐 $=\frac{1}{1/T_w+1/T_l}=\frac{T_w T_l}{T_w+T_l}$（调和平均，$<\min$）；
- **异步**：吞吐 $\to\max(T_w, T_l)$（瓶颈侧决定，另一侧重叠）。

例：$T_w=T_l=1$，同步吞吐 $=0.5$，异步 $\to 1$（2× 提升）。

### 4.2 staleness 与 ratio 失真

worker 用 $\theta_t$ 采，learner 在 $\theta_{t+k}$ 消费。PPO ratio：
$$
\rho=\frac{\pi_{\theta_{t+k}}(a|s)}{\pi_{\theta_t}(a|s)}
$$

$k=0$（同步）：$\rho=1$。$k$ 大：$\rho$ 偏离 1，clip $[1-\epsilon,1+\epsilon]$ 截断。$k$ 过大 → clip 频率高 → 有效梯度信号弱 → 训练慢/不稳。

### 4.3 weight sync 的 staleness 控制

[[weight sync mechanism]] 每 $S$ 步同步 $\theta$ 给 worker。worker 的最大 staleness $\le S$（在两次 sync 间采的数据）。$S$ 小 → stale 轻，但 sync 开销大（$\theta$ 大）；$S$ 大 → stale 重。tradeoff。

### 4.4 V-trace 的截断

IMPALA V-trace target：
$$
v_s=V_\theta(s_s)+\sum_{t=s}^{s+n-1}\gamma^{t-s}\rho_{\bar\theta}^{(t)}\bigl(r_t+\gamma V_\theta(s_{t+1})-V_\theta(s_t)\bigr)
$$

其中 $\rho_{\bar\theta}^{(t)}=\min(\bar\rho, \pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t))$ 截断 ratio，$\bar\rho\approx1$。容忍 stale 的多步 return。

## 5. 代码示例

```python
import torch, threading, queue, time

class Policy:
    def __init__(self):
        self.theta = torch.zeros(4)
        self.theta_version = 0           # θ 版本号(追踪 staleness)
    def generate(self, n):
        return torch.randn(n, 4) * self.theta
    def train_step(self, batch):
        self.theta += 0.01 * batch.mean(0)
        self.theta_version += 1

policy = Policy()
buffer = queue.Queue(maxsize=20)

def worker():
    """异步 worker:持续采,记录采时的 θ 版本"""
    while True:
        traj = policy.generate(8)
        data_version = policy.theta_version   # 采时的 θ 版本
        buffer.put((traj, data_version))
        time.sleep(0.1)

def learner():
    """异步 learner:持续训,算 staleness"""
    while True:
        traj, data_version = buffer.get()
        staleness = policy.theta_version - data_version   # k 步
        policy.train_step(traj)
        if staleness > 0:
            print(f"  stale data: k={staleness}(用 θ_{data_version} 采,θ_{policy.theta_version} 训)")
        time.sleep(0.15)

# 启动异步训练
threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=learner, daemon=True).start()
time.sleep(2)
print(f"异步训练后 θ = {policy.theta}, version = {policy.theta_version}")
print("异步:采训并行,吞吐高,但有 stale data(需 weight sync + ratio clip 修正)")
```

> [!tip] 实际工程（verl）
> verl 支持同步/异步两种模式：
> - **同步**：rollout worker 采完 → learner 训 → weight sync → 再采。简单稳定，吞吐低。
> - **异步**：rollout worker 持续采进 Ray object store，learner 持续取训，weight sync 定期广播新 θ。吞吐高，需 ratio clip + target_kl 修 stale。
> 大模型 RLHF 常用同步（稳定优先），异步在高吞吐场景用。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[learner worker split]]（异步是 split 的运行模式）、[[PPO optimization]]（PPO 的 ratio clip 修正 stale）、[[replay buffer]]（异步的数据管道）。
- **下游（应用）**: [[stale policy problem]]（异步的代价）、[[weight sync mechanism]]（防 stale）、IMPALA/V-trace、verl/OpenRLHF 的异步模式。
- **对比 / 易混**:
  - **异步 vs 同步训练**：见 §3.1。吞吐 vs 稳定 tradeoff。
  - **asynchronous training vs [[learner worker split]]**：前者是**运行模式**（并行流水线），后者是**架构**（采训分离）。split 可同步可异步。
  - **异步 vs 分布式数据并行（DDP）**：DDP 是同步数据并行（梯度 all-reduce 后同步更新），异步训练是采训流水线。不同层次。
  - **stale data vs [[stale policy problem]]**：前者是现象（数据滞后），后者是后果（ratio 失真训练不稳）。前者引出后者。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"异步一定比同步快"**：当 $T_w\approx T_l$ 时约 2×，但若 stale 修正开销大（clip 频繁/重采），净收益可能减。且瓶颈侧决定上限。
> 2. **"异步无代价"**：代价是 stale data → [[stale policy problem]]，需 weight sync + ratio clip 修正。
> 3. **"staleness k 越小越好"**：k 小 stale 轻，但需频繁 weight sync（$\theta$ 大，sync 慢），tradeoff。
> 4. **"异步 = 并发多线程"**：不只是线程，是多进程/多设备的采训流水线，用 Ray/分布式。
> 5. **"PPO 不能异步"**：可以，靠 ratio clip 修正 stale（off-policy 修正）。IMPALA 的 V-trace 是理论保证。
> 6. **"异步训练 = 异步梯度更新"**：不同。前者是采训流水线（数据层异步），后者是参数服务器异步梯度（PS 架构）。不同概念。
> 7. **"buffer 越大异步越好"**：buffer 大平滑波峰，但旧数据堆积 stale 重。需容量限制 + 过期丢弃。
> 8. **"RLHF 用异步"**：多数 RLHF 实践用同步（稳定优先），异步在高吞吐场景（如大规模在线 RLHF）才用。

## 8. 延伸细节

### 8.1 同步 vs 异步 PPO 的选择

| 场景 | 推荐 |
|---|---|
| 小规模、稳定优先 | 同步 PPO |
| 大规模在线 RLHF、吞吐瓶颈 | 异步 PPO + weight sync |
| 生成与训练耗时近 | 异步收益大（2×） |
| 生成远慢于训练 | 异步让训练与生成重叠，收益大 |
| 生成远快于训练 | 异步收益小（训练瓶颈） |

RLHF 中生成（rollout）常远慢于训练，异步收益大，但工程复杂度高，故开源框架默认同步，异步是高级选项。

### 8.2 IMPALA 的遗产

IMPALA 是异步训练的里程碑（DeepMind 2018）：
- 多 actor 并行采 + 单 learner 异步训；
- V-trace 修正 stale；
- throughput 大超同步 A3C。
其思想（采训分离 + 异步 + ratio 修正）被 RLHF 框架（verl/OpenRLHF）继承，只是 actor 换成 LLM rollout worker。

### 8.3 异步的 weight sync 策略

- **周期同步**：每 $S$ 步广播新 $\theta$，简单，$S$ 需调；
- **按需同步**：staleness 超 threshold 才 sync，自适应；
- **增量同步**：只传 $\theta$ 的差分（$\Delta\theta$），减带宽（但大模型 $\Delta$ 仍大）；
- **流式同步**：后台流式推 $\theta$，worker 渐进更新。
详见 [[weight sync mechanism]]。

### 8.4 异步与 stale policy problem 的深度关联

异步是 stale policy problem 的**主要成因**（同步无 stale）。stale 的修正：
- ratio clip（PPO）：限 ratio 失真；
- V-trace（IMPALA）：多步 return + 截断；
- weight sync：减小 $k$；
- target_kl 早停：监控 stale 引起的 KL 漂移。
四者协同，详见 [[stale policy problem]]。

### 8.5 异步训练的监控

- **staleness 分布**：buffer 中数据的 $k$ 分布，均值/最大值；
- **clip frequency**：ratio 触 clip 的比例，stale 重则高；
- **weight sync 延迟**：同步耗时，影响 $k$；
- **buffer 积压**：worker 产速度 vs learner 消费速度，积压说明 learner 慢。
这些是异步训练的诊断仪表盘。

---
相关: [[训练架构]]、[[learner worker split]]、[[rollout worker]]、[[inference worker]]、[[stale policy problem]]、[[weight sync mechanism]]、[[PPO optimization]]、[[PPO clipped objective]]、[[KL penalty与KL control]]、[[replay buffer]]、[[Ray基础]]
