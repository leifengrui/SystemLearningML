# resource starvation

> **所属章节**: [[工程问题]]
> **所属模块**: [[09-Ray与分布式调度]]
> **别名**: resource starvation / 资源饥饿 / 饿死 / starvation
> **难度**: 中（需懂 [[scheduling策略]] + [[resource request]]）


## 1. 一句话定义

**resource starvation（资源饥饿）** 是 Ray 中某些 task/actor **长期得不到资源**（被高优先级或大 task 持续抢占）而"饿死"的状态——它们能进入调度队列、但**永远排不上**执行。与 [[actor deadlock]]（循环等待、无进展）不同：starvation 的 task 理论上有进展（队列在动），但实际被持续插队/挤掉，等不到自己。根因是调度器缺 **fairness**（公平份额），一个 job 的大 task 把集群占满，其他 job 的 task 永远排队。

> [!note] 三句话定位
> - **是什么**：task 长期得不到资源（被抢/挤），饿在队列里。
> - **为什么**：调度器无 fairness，高优先级/大 job 持续占资源，低优先级/小 job 永远排不上。
> - **与 deadlock 的区别**：deadlock 是循环等待卡死（无进展）；starvation 是队列在动但轮不到你（有进展但饿）。两者都"卡"，但本质不同。


## 2. 为什么需要它（动机与背景）

### 2.1 资源稀缺与争抢

集群资源有限（如 8 GPU）。同时有：

- Job A：1000 个 `num_gpus=1` 的训练 task。
- Job B：10 个 `num_gpus=1` 的推理 task。

若无 fairness，调度器 first-come-first-served，Job A 的 1000 task 把 8 GPU 占满，Job B 的 10 task 在队列等——Job A 跑完才轮到 B。B 被饿死（长时间无资源）。

### 2.2 优先级反转

高优先级 task 持续到达，低优先级 task 永远被插队：

- 高优先级训练 task 流水线提交，每秒来一批。
- 低优先级推理 task 在队列，每次刚要调度，新来的高优先级插队。

低优先级 task 饿死。这是**优先级反转**（priority inversion）的变种——低优先级被持续挤掉。

### 2.3 head-of-line blocking

队列头部的大 task（如 `num_gpus=8` 训练）需 8 GPU 才能调度，但集群只有零碎空闲（4+2+2 跨 3 worker，无单 worker 8 GPU）。大 task 卡在队头，后面的小 task（`num_gpus=1`）本可调度，但若调度器严格 FIFO，小 task 被队头大 task 阻塞 → 间接 starvation。

### 2.4 RLHF 多实验场景

一个集群跑多个 RLHF 实验：

- 实验 A：大模型训练（8 GPU × 数小时）。
- 实验 B：小模型调参（1 GPU × 几分钟）。

无 fairness 时 A 占满集群，B 的快速实验等几小时——B 被饿死，研发效率低。


## 3. 核心概念详解

### 3.1 fairness（公平份额）

调度器给每 job 分**公平份额**（fair share）资源。如 8 GPU 集群、2 job，各得 4 GPU。即使 Job A 想跑 1000 task，它最多用 4 GPU，另 4 GPU 给 Job B。B 不饿。

主要公平算法：

- **max-min fairness**：最大化最小分配者。资源均分，剩余给需求大的。
- **DRF（Dominant Resource Fairness）**：多资源（GPU+CPU）下，按每 job 的主导资源（最紧缺的那个）公平。GPU 密集 job 和 CPU 密集 job 各按主导资源均分。

### 3.2 抢占（preemption）

高优先级 task 可**抢占**低优先级 task 的资源（暂停低优先级、释放资源给高优先级）。但若高优先级持续来，低优先级被反复抢占 → starvation。需配合 **priority aging**（等久了优先级渐升）避免饿死。

### 3.3 优先级反转

低优先级持资源，高优先级等它 → 中优先级插队（比低优先级先跑）→ 低优先级饿、高优先级间接饿。解法：**priority inheritance**（低优先级临时继承高优先级，让它快释放资源）。

### 3.4 与 deadlock 的区别

| 维度 | [[actor deadlock]] | resource starvation |
|---|---|---|
| 状态 | 卡死、无进展 | 队列在动、轮不到 |
| 根因 | 循环等待 | 缺 fairness/被抢占 |
| 资源 | 持有等对方 | 排队等可用 |
| 解法 | 避免环/async | fairness/aging |

### 3.5 Ray 的 fairness 机制

Ray 默认按 job 的 **pending task 数**加权分配（多 pending 的 job 多分资源，近似公平）。可配 **fairshare** 策略强化。生产多租户必开。


## 4. 数学原理 / 公式

### 4.1 max-min fairness

$N$ 个 job，总资源 $R$。max-min fairness 分配 $a_i$：

$$
a_i = \max\left(\min\left(\frac{R - \sum_{j<i} a_j}{N-i+1}, d_i\right), 0\right)
$$

其中 $d_i$ 是 job $i$ 的需求。先均分 $R/N$，需求小的拿完剩余给需求大的。

> [!note] max-min 示例
> 8 GPU，Job A 需 6、Job B 需 4。均分 4-4（A 拿 4 没到 6，B 拿 4 满足）。B 满足后剩余 4 全给 A → A 得 4。最终 A=4、B=4。无 starvation。

### 4.2 DRF（多资源公平）

两资源（GPU、CPU），job $i$ 的主导份额 $s_i = \max(g_i/G, v_i/V)$（$g_i$/$v_i$ 是分到的 GPU/CPU，$G$/$V$ 是总量）。DRF 最大化最小主导份额：

$$
\max \min_i s_i
$$

GPU 密集 job（多用 GPU）和 CPU 密集 job（多用 CPU）各按主导资源均分，不互相饿死。

### 4.3 starvation 等待时间

无 fairness 时，低优先级 task 的期望等待时间：

$$
E[W] = \sum_{k} \frac{\lambda_h \cdot T_h}{\lambda_l}
$$

其中 $\lambda_h$ 是高优先级到达率、$T_h$ 是高优先级执行时间、$\lambda_l$ 是低优先级到达率。若 $\lambda_h T_h \ge 1$（高优先级占满），$E[W] \to \infty$（饿死）。fairness 把 $\lambda_h$ 限流在 fair share 内，$E[W]$ 有限。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 starvation vs fairness

```python
class Worker:
    def __init__(self, gpus): self.gpus=gpus; self.used=0; self.per_job={}
    def available(self, g): return (self.gpus-self.used)>=g
    def allocate(self, g, jid):
        if self.available(g):
            self.used+=g; self.per_job[jid]=self.per_job.get(jid,0)+g
            return True
        return False
    def release(self, g, jid): self.used-=g; self.per_job[jid]-=g

class Scheduler:
    """模拟 Ray: 无 fairness (FCFS) vs fairness (fair share)."""
    def __init__(self, workers, fair=False):
        self.workers=workers; self.fair=fair; self.capacity=sum(w.gpus for w in workers)
    def schedule(self, g, jid):
        if self.fair:
            # fair share: 每 job 上限 = capacity / n_jobs (简化 2 job)
            job_use = sum(w.per_job.get(jid,0) for w in self.workers)
            if job_use >= self.capacity / 2:    # 超公平份额, 让其他 job
                return False
        for w in self.workers:
            if w.allocate(g, jid): return True
        return False

# 集群: 8 GPU
ws = [Worker(8)]
# 场景: Job A 提 8 task (各1 GPU), Job B 提 8 task (各1 GPU)
s = Scheduler([Worker(8)], fair=False)
a_ok = sum(s.schedule(1, 'A') for _ in range(8))   # A 抢光 8
b_ok = sum(s.schedule(1, 'B') for _ in range(8))   # B 全饿
print(f'无 fairness: A 调度 {a_ok}/8, B 调度 {b_ok}/8 (B 饿死)')

s2 = Scheduler([Worker(8)], fair=True)
a2 = sum(s2.schedule(1, 'A') for _ in range(8))    # A 受 fair share 限 4
b2 = sum(s2.schedule(1, 'B') for _ in range(8))    # B 得 4
print(f'有 fairness: A 调度 {a2}/8, B 调度 {b2}/8 (均分, B 不饿)')
```

### 5.2 真实 Ray API 对照

```python
# import ray
# ray.init(fairshare_enabled=True)  # 开启公平调度
# # 多 job 自动按 fair share 分配, 避免一个 job 饿死其他
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[scheduling策略]]（fairness 是其核心维度）、[[resource request]]（资源声明）、[[placement group]]（PG 预留可能加剧 starvation，预留资源不参与公平分配）。
- **下游（应用）**: [[task queue backlog]]（starvation 导致队列堆积）、[[actor deadlock]]（deadlock 也会导致资源占用不释放，间接 starvation）、RLHF 多实验调度。
- **对比 / 易混**:
  - **starvation vs [[actor deadlock]]**：deadlock 循环等待卡死（无进展）；starvation 队列在动但轮不到（饿）。两者都"卡"，根因与解法不同。
  - **starvation vs [[task queue backlog]]**：backlog 是队列堆积的**现象**，starvation 是某些 task **永远轮不到**的**根因**。backlog 不一定是 starvation（可能是吞吐不足）。
  - **starvation vs [[resource request]] 不满足**：request 不满足是单次调度失败（排队等）；starvation 是持续等不到（饿）。前者临时，后者持久。


## 7. 常见误区与易错点

> [!warning] 误区 1：混淆 starvation 与 deadlock
> deadlock 是循环等待、无进展（actor 互相等）；starvation 是队列在动、轮不到你（被抢占/挤掉）。deadlock 解法是避免环/async，starvation 解法是 fairness。两者不同。

> [!warning] 误区 2：以为 FCFS 公平
> first-come-first-served 看似公平，但大 job（1000 task）先来就占满集群，后到的小 job 饿死。FCFS 是"先到者通吃"，非真公平。需 fair share。

> [!warning] 误区 3：fairness 是免费的
> fairness 限制每 job 份额，可能降低总吞吐（如 Job A 本可独占 8 GPU 跑完，fairness 强制分 4 给 B，A 变慢）。fairness 是**公平与吞吐的权衡**——多租户场景必开，单租户可关。

> [!warning] 误区 4：placement group 加剧 starvation
> PG 的 bundle 预留资源**不参与公平分配**（预留给特定 actor 组）。若 PG 预留大量资源但闲置，其他 task 可能饿。生产需平衡 PG 预留与可调度池。

> [!warning] 误区 5：忽略 priority aging
> 高优先级持续来，低优先级饿死。无 aging（等久了优先级渐升）时低优先级永远等。生产应配 aging 避免饿死。

> [!warning] 误区 6：以为抢占能解 starvation
> 抢占让高优先级插队，但若高优先级持续来，被抢占的低优先级反复暂停 → 更饿。抢占是高优先级的工具，低优先级的 starvation 需 fairness/aging 解。


## 8. 延伸细节

### 8.1 DRF 在 K8s / Ray 的应用

K8s 的 scheduler 里 DRF 是常用公平算法（按主导资源）。Ray 的 fairshare 也近似 DRF。多资源（GPU+CPU+内存）场景 DRF 比单维 max-min 更公平。

### 8.2 RLHF 多实验调度

一个集群跑多个 RLHF 实验，每实验是独立 job。开 fairness 后各实验按 fair share 分 GPU，互不饿死。verl/OpenRLHF 支持多 job 并行，配 Ray fairshare。

### 8.3 head-of-line blocking 的解法

队头大 task（需 8 GPU）卡住时，调度器应 **backfill**（向后扫，找能用零碎资源的小 task 插队调度），不严格 FIFO。Ray 的调度器近似 backfill（first-fit 不严格队头）。

### 8.4 优先级反转与继承

低优先级持锁/资源，高优先级等它，中优先级插队 → 高低都饿。**priority inheritance**：低优先级临时继承高优先级，快释放资源。Ray 无内置继承，但可通过 watchdog/超时近似。

### 8.5 starvation 检测

监控每 job 的**等待时间分布**：若某 job 的 task 等待时间持续增长、远超平均 → starvation 倾向。结合 [[task queue backlog]] 监控定位。

### 8.6 与 [[object store]] 的关系

starvation 不只 GPU/CPU——[[object store]] 容量也有限，大对象占满后小 task 写不进 → object store starvation。生产需监控 object store 用量。

---
相关: [[工程问题]] | [[scheduling策略]] | [[resource request]] | [[placement group]] | [[actor deadlock]] | [[task queue backlog]] | [[policy training (PT)]] | [[rollout worker]] | [[object store]]
