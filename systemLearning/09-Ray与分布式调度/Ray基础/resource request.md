# resource request

> **所属章节**: [[Ray基础]]
> **所属模块**: [[09-Ray与分布式调度]]
> **别名**: resource request / 资源声明 / resource requirement / num_gpus num_cpus
> **难度**: 中（需懂 [[remote function]] + 调度基础）


## 1. 一句话定义

**resource request** 是 Ray 中 task/actor 在 `@ray.remote(num_gpus=K, num_cpus=N, resources={...})` 声明的**资源需求**：调度器据此把 task/actor 分配到满足资源的 worker（如 `num_gpus=1` 的 task 到 GPU 节点），并保证不超额（一个 4-GPU worker 最多同时跑 4 个 `num_gpus=1` 的 task）。它是分布式调度的基石——没有资源声明，调度器无法知道 task 需要什么，会乱分配（GPU task 调到 CPU worker、超额调度导致 OOM）。

> [!note] 三句话定位
> - **是什么**：task/actor 声明的 GPU/CPU/自定义资源需求，调度器据此分配 worker。
> - **为什么**：集群多任务争抢资源，需声明需求让调度器合理分配、不超额。
> - **关键认知**：`num_gpus` 是**逻辑声明**（调度记账），不一定等于物理独占——可 fractional（0.5 GPU）、可超额（见误区）。


## 2. 为什么需要它（动机与背景）

### 2.1 集群资源争抢

一个 Ray 集群可能有：4-GPU 节点 × 8、CPU-only 节点 × 20。同时跑的任务：

- LLM 推理 task：需 1 GPU。
- 数据预处理 task：需 4 CPU、0 GPU。
- 训练 actor：需 8 GPU（整个节点）。

若无资源声明，调度器可能把 GPU task 调到 CPU 节点（跑不了）或把 8 个 GPU task 塞一个 4-GPU 节点（OOM）。resource request 让 task 携带"我需要什么"，调度器按可用资源分配。

### 2.2 防止超额调度

GPU 是稀缺资源。一个 4-GPU worker，若无记账，10 个 `num_gpus=1` task 可能全塞进来 → 显存 OOM。Ray 的资源记账保证：worker 上所有运行 task 的 `num_gpus` 之和 ≤ worker 总 GPU 数。`num_gpus=1` 的 task 在 4-GPU worker 上最多同时跑 4 个，第 5 个排队等释放。

### 2.3 与 K8s 资源的对比

K8s 的 pod 资源声明（requests/limits）是粗粒度（pod 级、整个 pod 占用）。Ray 的 resource request 是**细粒度、动态**的：task 级（比 pod 小）、执行时占用、完成即释放。这让 Ray 适合**动态、细粒度**的 ML 工作负载（如 RLHF 的采样 task 短时占用 GPU）。


## 3. 核心概念详解

### 3.1 资源类型

| 资源 | 语法 | 说明 |
|---|---|---|
| GPU | `num_gpus=1` | GPU 卡数（可 fractional: 0.5） |
| CPU | `num_cpus=4` | CPU 核数（可 fractional: 0.5） |
| 自定义 | `resources={"memory": 16e9}` | 任意命名资源（如内存、TPU） |

### 3.2 fractional GPU（分数 GPU）

```python
@ray.remote(num_gpus=0.5)
def f(x): ...   # 半个 GPU (2 个这样的 task 共占 1 GPU)
```

`num_gpus=0.5` 表示 task 用"半张 GPU"。Ray 记账：2 个 0.5-GPU task 可共享 1 GPU。这适合**显存够、算力共享**的场景（如小模型推理、MPS）。但注意：Ray 只做**记账**，不物理分割 GPU——两个 0.5 task 实际共享整张卡（可能争算力/显存），需应用层保证不超（如手动限显存）。

> [!warning] fractional GPU 不物理分割
> Ray 的 `num_gpus=0.5` 是调度记账（2 个 task 共占 1 GPU 配额），但物理上两个 task 共享整张卡。若两 task 各用满显存，仍会 OOM。fractional GPU 适合显存够、可共享的场景，不适合显存敏感的大模型。

### 3.3 资源原子性与调度

调度是**原子**的：一个 task 需 `num_gpus=2`，必须找到有 ≥2 空闲 GPU 的**单个** worker，不能跨 worker 拼 1+1。这让多 GPU task（如 `num_gpus=8` 的训练 actor）只能调度到 ≥8-GPU 节点。[[placement group]] 解决多 worker 联合分配（见 §8.3）。

### 3.4 调度算法：bin packing vs spread

Ray 默认 **bin packing**（先填满一个 worker 再用下一个，提高利用率、利于 locality）。可选 **spread**（打散到不同 worker，提高容错）。调度策略见 [[scheduling策略]]。

### 3.5 超额订阅（oversubscription）

`@ray.remote(num_gpus=0)` 声明 0 GPU，task 可调度到任意 worker（含 GPU 节点的 CPU 部分）但不占 GPU 配额。这常用于 CPU task 跑在 GPU 节点上（利用 GPU 节点的 CPU 富余算力），但需注意不与 GPU task 争 CPU。


## 4. 数学原理 / 公式

### 4.1 资源约束（背包问题）

worker $i$ 有总资源 $C_i = (G_i, V_i)$（GPU 数、CPU 数）。task $j$ 声明需求 $r_j = (g_j, v_j)$。调度需满足：

$$
\sum_{j \in \text{running on } i} g_j \le G_i, \quad \sum_{j \in \text{running on } i} v_j \le V_i \quad \forall i
$$

即每 worker 上运行 task 的资源声明之和不超过总量。这是**多维度背包问题**（NP-hard，Ray 用启发式：first-fit / best-fit）。

> [!note] 调度复杂度
> 多维背包是 NP-hard，但 Ray 集群规模（数千 task）用启发式（first-fit decreasing）足够快（毫秒级调度）。精确最优解不现实，Ray 牺牲最优性换调度速度。

### 4.2 碎片与反碎片

4-GPU worker 上已有 3 个 `num_gpus=1` task（用 3 GPU），剩 1 GPU。此时一个 `num_gpus=2` 的 task 无法用（需单 worker ≥2），只能等或换 worker。这是**碎片**问题。[[placement group]] 的 bundle 预留机制缓解（提前打包资源）。

### 4.3 利用率

集群 GPU 利用率：

$$
U = \frac{\sum_j g_j \cdot \mathbb{1}[\text{task } j \text{ running}]}{\sum_i G_i}
$$

bin packing 提高 $U$（填满再开新 worker，减少空闲碎片）。spread 降低 $U$ 但提高容错。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟资源调度（bin packing）

```python
class Worker:
    def __init__(self, wid, gpus, cpus):
        self.wid = wid; self.gpus = gpus; self.cpus = cpus
        self.used_g = 0; self.used_c = 0
    def available(self, g, c):
        return (self.gpus - self.used_g) >= g and (self.cpus - self.used_c) >= c
    def allocate(self, g, c):
        if self.available(g, c): self.used_g += g; self.used_c += c; return True
        return False
    def release(self, g, c): self.used_g -= g; self.used_c -= c

class Scheduler:
    """模拟 Ray 调度: bin packing, 按 num_gpus/num_cpus 分配."""
    def __init__(self, workers): self.workers = workers
    def schedule(self, g, c):
        for w in self.workers:                    # first-fit (bin packing)
            if w.allocate(g, c): return w
        return None                                # 资源不足, task 排队

# 集群: W0=4GPU/8CPU, W1=2GPU/4CPU
ws = [Worker(0, 4, 8), Worker(1, 2, 4)]
sched = Scheduler(ws)
# 提交带资源声明的 task
tasks = [(1, 2, 'task1'), (2, 2, 'task2'), (1, 1, 'task3'),
         (1, 1, 'task4'), (1, 1, 'task5')]
for g, c, name in tasks:
    w = sched.schedule(g, c)
    if w: print(f'{name}: need {g}gpu/{c}cpu -> W{w.wid} (used {w.used_g}g/{w.used_c}c)')
    else: print(f'{name}: need {g}gpu/{c}cpu -> QUEUED (资源不足)')
# task1->W0(1g/2c) task2->W0(3g/4c) task3->W0(4g/5c, W0 满)
# task4->W1(1g/1c) task5->W1(2g/2c, W1 满)
```

### 5.2 fractional GPU 演示

```python
@ray.remote(num_gpus=0.5)
def infer(x): ...   # 0.5 GPU 配额
# 4-GPU worker 可同时跑 8 个 0.5-GPU task (记账), 物理共享需应用保证显存
```

### 5.3 真实 Ray API 对照

```python
# import ray; ray.init(num_gpus=4, num_cpus=8)
# @ray.remote(num_gpus=1, num_cpus=2)
# def forward(batch): ...          # 调度到有 ≥1 GPU ≥2 CPU 的 worker
# refs = [forward.remote(b) for b in batches]   # 4 GPU 最多 4 个并行
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[remote function]]（resource request 作为 `@ray.remote` 参数）、Ray 调度器、worker 资源上报。
- **下游（应用）**: [[placement group]]（资源预留/打包）、[[scheduling策略]]（bin packing/spread）、[[GPU绑定]]（物理绑定强化逻辑声明）、[[actor model]]/[[task model]]（两者都声明资源）。
- **对比 / 易混**:
  - **resource request vs 物理资源独占**：request 是逻辑记账（调度器记账不超额），不一定物理独占。fractional GPU（0.5）是记账共享，非物理分割。[[GPU绑定]] 才物理绑定。
  - **Ray resource request vs K8s requests/limits**：Ray 细粒度（task 级、动态）、K8s 粗粒度（pod 级、静态占用）。
  - **num_gpus vs 实际 GPU 利用率**：声明 `num_gpus=1` 不保证 task 真用满 GPU（可能只算一点），只保证调度配额。利用率看 [[GPU utilization]]。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 num_gpus 是物理独占
> 不是。`num_gpus=1` 是调度记账（保证该 worker 上 ≤4 个这样的 task），但物理上 task 用的是 worker 的 GPU，可能与其他 task 共享（除非 [[GPU绑定]] 或独占模式）。fractional（0.5）更是显式共享。

> [!warning] 误区 2：fractional GPU 能防 OOM
> 不能。`num_gpus=0.5` 只保证 2 个 task 共享 1 GPU 配额，不物理分割显存。若两 task 各加载大模型，显存仍超。fractional 适合显存富余的小 task，大模型应用整数 GPU 或显式管显存。

> [!warning] 误区 3：忘记声明资源导致调度错误
> `@ray.remote` 不带 `num_gpus` → task 当 CPU task 调度，可能跑到 CPU-only worker，无法用 GPU。GPU task 必须显式 `num_gpus=1`。

> [!warning] 误区 4：多 GPU task 跨 worker
> `num_gpus=8` 的 task 必须找单个 ≥8-GPU worker，不能跨 2 个 4-GPU worker 拼。要跨 worker 联合用 [[placement group]] 的多 bundle。

> [!warning] 误区 5：忽略碎片
> 4-GPU worker 残留 1 GPU 空闲时，`num_gpus=2` task 用不了（需单 worker ≥2），导致碎片浪费。placement group 预留缓解。


## 8. 延伸细节

### 8.1 自定义资源

`resources={"memory_large": 1}` 声明自定义资源。worker 启动时 `ray start --resources='{"memory_large": 4}'` 上报。用于表达 GPU/CPU 之外的约束（如大内存节点、TPU、特殊硬件）。

### 8.2 RLHF 中的资源声明

- [[policy training (PT)]] 的 Learner actor：`@ray.remote(num_gpus=8)`（8 GPU 训练）。
- [[rollout worker]] 的 PolicyServer actor：`@ray.remote(num_gpus=1)`（1 GPU 推理）。
- 采样请求 task：`@ray.remote(num_gpus=0)`（不占额外 GPU，用 PolicyServer 的）或 `num_gpus=1`（独立 GPU task）。
- [[placement group]] 把 Learner + PolicyServer colocate（同节点，省权重传输）。

### 8.3 placement group 与资源预留

普通调度是"task 提交时找资源"，可能因碎片失败。[[placement group]] 的 bundle 是**预留**：提前打包一组资源（如 2 bundle × {4 GPU, 8 CPU}），bundle 内资源专用，task 调度到 bundle 不争抢。适合多 actor 联合放置（如训练 + 推理同节点）。

### 8.4 超额订阅与软约束

Ray 支持 `resources={"memory": 1e9}` 等自定义软约束。`num_gpus=0` 的 task 可跑在 GPU 节点的 CPU 部分（超额利用）。但需应用保证不与 GPU task 争资源。

### 8.5 与 GPU 利用率的区别

`num_gpus=1` 是**调度配额**（保证不超额调度），不是**实际利用率**。task 声明 1 GPU 但可能只算 10% 时间（利用率低）。利用率看 [[GPU utilization]]（roofline、SM 占用）。资源声明解决"分多少"，利用率解决"用多狠"。

---
相关: [[Ray基础]] | [[remote function]] | [[placement group]] | [[scheduling策略]] | [[GPU绑定]] | [[actor model]] | [[task model]] | [[policy training (PT)]] | [[rollout worker]] | [[GPU utilization]]
