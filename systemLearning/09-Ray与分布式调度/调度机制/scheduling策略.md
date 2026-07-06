# scheduling策略

> **所属章节**: [[调度机制]]
> **所属模块**: [[09-Ray与分布式调度]]
> **别名**: scheduling策略 / 调度策略 / scheduling policy / 调度算法
> **难度**: 中（需懂 [[resource request]] + [[placement group]]）


## 1. 一句话定义

**scheduling策略** 是 Ray 调度器为 task/actor 选 worker 的算法，主要有 **bin packing**（装箱优先，填满一个 worker 再用下一个，提高利用率与 locality）和 **spread**（分散优先，尽量用不同 worker，提高容错）两种，再叠加 **locality**（数据本地优先）、**fairness**（多租户公平）、**priority**（优先级）等考虑。它是 [[resource request]]（task 要什么）之后的第二步——task 声明资源后，调度器按策略选哪个 worker 执行。[[placement group]] 的 PACK/SPREAD 是 task/actor 组级的策略，scheduling策略 是单 task/actor 级的策略。

> [!note] 三句话定位
> - **是什么**：调度器选 worker 的算法（bin packing 装箱 / spread 分散 / locality 本地 / fairness 公平）。
> - **为什么**：不同负载需不同策略——计算密集要 bin packing（提利用率+locality），容错要 spread（分散）。
> - **与 PG 的关系**：PG 管"一组 task 的放置策略"，scheduling策略 管"单个 task 的选 worker 算法"。两者互补。


## 2. 为什么需要它（动机与背景）

### 2.1 选 worker 的自由度

task 声明 `num_gpus=1` 后，集群可能有多个满足的 worker（8 个 4-GPU worker 都有空闲 GPU）。选哪个？这影响：

- **利用率**：bin packing 填满一个再开下一个，空闲 worker 可关机省电/缩容。
- **locality**：选有 task 输入数据（[[object store]] 的生产方）的 worker，零拷贝传数据。
- **容错**：spread 分散，单 worker 挂掉影响小。
- **公平**：多租户/多 job 不应一个独占集群。

scheduling策略 让这些目标可配置。

### 2.2 负载特性差异

- **训练/推理**（计算密集、数据大）：bin packing + locality，填满 worker、数据本地传。
- **容错服务**（副本分散）：spread，分散不同 worker。
- **多租户**（多 job 共享）：fairness，每 job 公平分资源。

一种策略打天下不行，需可配置。

### 2.3 碎片与利用率权衡

bin packing 提高利用率（填满再开新）但易碎片（残留小块）。spread 降低利用率（分散，每 worker 都有零碎占用）但反碎片（不集中塞）。调度策略是利用率与碎片的权衡。


## 3. 核心概念详解

### 3.1 bin packing（装箱）

first-fit：按 worker 列表顺序，第一个能放下的就用。填满一个 worker 再开下一个。变种 best-fit（选剩余空间最小的，减少碎片）、worst-fit（选剩余最大的，反碎片）。Ray 默认近似 first-fit bin packing。

**优点**：利用率高（集中占用，空闲 worker 多）、locality 好（同 worker 的 task 数据本地）。
**缺点**：易碎片（塞满后残留小块放不下大 task）、容错差（同 worker 的 task 一起挂）。

### 3.2 spread（分散）

选当前**最空闲**的 worker（如 GPU 占用最低）。变种 round-robin（轮转）。

**优点**：容错好（task 分散，单 worker 挂影响小）、反碎片（不集中塞）。
**缺点**：利用率低（每 worker 都有零碎占用，难缩容）、locality 差（数据分散）。

### 3.3 locality（数据本地）

优先把消费方 task 调度到生产方 task 同 worker（[[object store]] 零拷贝）。Ray 的调度器把 task 的输入 ObjectRef 的 location 作为优先信号。

```python
@ray.remote
def produce(): return big_tensor            # 在 W0 产出
@ray.remote
def consume(x): return process(x)
ref = produce.remote()                       # 产出在 W0
# consume 优先调度到 W0 (零拷贝取 big_tensor)
ray.get(consume.remote(ref))
```

### 3.4 fairness（公平）

多租户/多 job 共享集群时，fairness 保证每 job 按权重分资源（如 job A:job B = 1:1）。Ray 的公平调度按 job 的 pending task 数加权分配 worker。避免一个 job 的 1000 个 task 把集群占满、其他 job 饿死（[[resource starvation]]）。

### 3.5 priority（优先级）

task 可声明优先级，高优先级优先调度（插队）。RLHF 中训练 task 优先于推理 task（训练不能等）。Ray 支持优先级队列。

### 3.6 与 placement group 的关系

PG 的策略（PACK/SPREAD）是**组级**（一组 bundle 的放置）。scheduling策略 是**单 task 级**（单个 task 选 worker）。一个 Ray 集群同时用两者：PG 放置一组 actor（如 Learner+PolicyServer colocate），单 task 用 bin packing+locality。PG 是"组规划"，scheduling 是"单点调度"。


## 4. 数学原理 / 公式

### 4.1 bin packing 与利用率

$W$ 个 worker，每 worker 容量 $C$。$N$ 个 task，各占 $c_j$。bin packing 目标：用最少 worker 装下所有 task：

$$
\min \text{used workers}, \quad \text{s.t.} \sum_{j \in \text{worker } i} c_j \le C
$$

利用率：

$$
U = \frac{\sum_j c_j}{\text{used workers} \cdot C}
$$

bin packing 让 used workers 最少 → $U$ 最大。spread 让 used workers = $W$（全用）→ $U$ 最低。

> [!note] bin packing 是 NP-hard
> 一维 bin packing 是 NP-hard，但 first-fit decreasing（按 task 大小降序，first-fit 放）是 1.7-近似（用 ≤1.7 × 最优 worker 数）。Ray 用启发式 first-fit，牺牲最优换调度速度（毫秒级）。

### 4.2 碎片率

bin packing 下，worker 塞到接近满时残留碎片。如 4-GPU worker 用了 3 GPU，剩 1 GPU 碎片——`num_gpus=2` 的 task 用不了。碎片率：

$$
F = \frac{\text{空闲但不能用的资源}}{\text{总空闲资源}}
$$

bin packing 的 $F$ 高（集中碎片），spread 的 $F$ 低（分散占用，少残留）。

### 4.3 locality 收益

task 输入数据 $D$，跨 worker 传输带宽 $\beta$，本地零拷贝：

$$
T_{\text{transfer}} = \frac{D}{\beta} \text{（跨）}, \quad 0 \text{（本地）}
$$

locality 命中省 $\frac{D}{\beta}$。大 $D$（GB 级）+ 慢 $\beta$（网络）时收益大。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 bin packing vs spread

```python
class Worker:
    def __init__(self, wid, gpus):
        self.wid=wid; self.gpus=gpus; self.used=0
    def available(self, g): return (self.gpus-self.used) >= g
    def allocate(self, g):
        if self.available(g): self.used+=g; return True
        return False
    def __repr__(self): return f'W{self.wid}({self.used}/{self.gpus}g)'

class Scheduler:
    def __init__(self, workers, strategy): self.workers=workers; self.strategy=strategy
    def schedule(self, g):
        if self.strategy == 'BIN_PACK':               # first-fit, 填满
            for w in self.workers:
                if w.allocate(g): return w
        elif self.strategy == 'SPREAD':               # 最空闲的 worker
            avail = [w for w in self.workers if w.available(g)]
            if avail:
                w = min(avail, key=lambda w: w.used)  # 最少占用
                w.allocate(g); return w
        return None

# 集群: 4 个 4-GPU worker
tasks = [1,1,2,1,3,1]   # 6 个 task 的 GPU 需求
for strat in ['BIN_PACK', 'SPREAD']:
    ws = [Worker(i, 4) for i in range(4)]
    s = Scheduler(ws, strat)
    for g in tasks: s.schedule(g)
    print(f'{strat}:', ws, f'used_workers={sum(1 for w in ws if w.used>0)}')
# BIN_PACK 集中塞 W0,W1 (used_workers 少)
# SPREAD 分散到所有 (used_workers 多)
```

### 5.2 真实 Ray API 对照

```python
# import ray
# ray.init(scheduler_config={"spread": False})  # bin packing (默认)
# # locality 自动: 消费方 task 优先调度到生产方同 worker
# @ray.remote
# def produce(): return big_tensor
# @ray.remote
# def consume(x): return process(x)
# ref = produce.remote()                  # W0 产出
# ray.get(consume.remote(ref))            # consume 优先到 W0 (locality)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[resource request]]（task 声明资源）、[[placement group]]（组级策略）、worker 资源上报、[[object store]]（locality 依据）。
- **下游（应用）**: [[actor deadlock]]（调度策略不当可能死锁）、[[resource starvation]]（fairness 缺失导致饿死）、[[task queue backlog]]（调度慢导致堆积）、[[GPU utilization]]（bin packing 提利用率）。
- **对比 / 易混**:
  - **bin packing vs spread**：装箱（集中、高利用、易碎片、容错差）vs 分散（散开、低利用、反碎片、容错好）。最核心对比。
  - **scheduling策略 vs [[placement group]] 策略**：前者单 task 选 worker，后者一组 bundle 放置。互补。
  - **Ray scheduling vs K8s scheduler**：Ray 细粒度动态（task 级、毫秒）、K8s 粗粒度静态（pod 级、秒）。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 bin packing 总是好的
> bin packing 提利用率，但易碎片（残留小块放不下大 task）+ 容错差（同 worker task 一起挂）。容错场景用 spread，碎片敏感用 PG 预留。没有万能策略。

> [!warning] 误区 2：spread 浪费资源
> spread 不是浪费，是分散。利用率低（每 worker 有零碎占用）但容错好、反碎片。服务高可用场景 spread 更优。利用率不是唯一目标。

> [!warning] 误区 3：忽略 locality
> task 输入大数据（GB 级）时，不 locality 命中会跨 worker 传输，慢。Ray 默认考虑 locality，但若 task 太小或集群大，locality 信号弱。大数据流应设计成 produce→consume colocate。

> [!warning] 误区 4：fairness 缺失导致饿死
> 多 job 共享集群时，若无 fairness，一个 job 的 1000 task 占满集群，其他 job 饿死（[[resource starvation]]）。生产环境必开 fairness。

> [!warning] 误区 5：混淆 PG 策略与 scheduling 策略
> PG 的 PACK/SPREAD 是**一组 bundle 的放置**（组级），scheduling 的 bin packing/spread 是**单 task 选 worker**（单点）。PG 创建后，PG 内 task 仍按 scheduling 策略选具体 bundle。两者不同层。


## 8. 延伸细节

### 8.1 Gang Scheduling

RLHF 训练需多 actor 同时启动（Learner + 所有 PolicyServer 一起）。若一个个调度，先启动的等后启动的，资源占着不干活。**Gang scheduling** 让一组 task 同时调度（全有或全无）。Ray 的 [[placement group]] 的 all-or-nothing 创建是 gang scheduling 的近似。

### 8.2 K8s 调度对比

K8s scheduler 也是 bin packing（默认）+ spread（pod anti-affinity）+ priority + fairness。但 K8s 调度 pod 级（秒级、静态），Ray 调度 task 级（毫秒级、动态）。ML 工作负载（task 短、频繁）用 Ray 更合适。

### 8.3 RLHF 的调度实践

- Learner + PolicyServer colocate：PG PACK。
- 采样请求 task：bin packing + locality（数据本地）。
- 多 job（多 RLHF 实验）：fairness，每 job 公平分 GPU。
- 训练 task 高优先级（不阻塞采样）。

verl/OpenRLHF 默认 bin packing + PG colocate，生产多租户开 fairness。

### 8.4 调度延迟与吞吐

调度器选 worker 的延迟（毫秒级）在高吞吐场景（每秒数千 task）可能成瓶颈。Ray 的调度器是分布式的（每 worker 本地调度本地 task），降低中心瓶颈。[[task queue backlog]] 监控调度延迟堆积。

### 8.5 自定义调度策略

Ray 支持自定义调度器插件（如按 GPU 拓扑、按网络亲和）。高级场景（如按 NVLink 拓扑绑 TP rank）需自定义。

---
相关: [[调度机制]] | [[resource request]] | [[placement group]] | [[object store]] | [[actor deadlock]] | [[resource starvation]] | [[task queue backlog]] | [[GPU utilization]] | [[policy training (PT)]] | [[rollout worker]]
