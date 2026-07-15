# placement group

> **所属章节**: [[调度机制]]
> **所属模块**: [[09-Ray与分布式调度]]
> **别名**: placement group / PG / 放置组 / 资源放置 / placement
> **难度**: 中高（需懂 [[resource request]] + 调度基础）


## 1. 一句话定义

**placement group（PG, 放置组）** 是 Ray 的**资源预留 + 放置策略**机制：把一组 **bundle**（资源包，如 `{1 GPU, 2 CPU}`）按策略（**PACK** 共置 / **SPREAD** 分散 / **STRICT_PACK** 严格同节点 / **STRICT_SPREAD** 严格不同节点）打包放置到集群，保证这组 task/actor 满足**共置**（colocate，同节点省通信）或**分散**（spread，提容错）约束。普通调度只按单 task 找资源、无法表达"这组 actor 要同节点"，PG 让"一组资源 + 放置策略"成为一等公民——是 RLHF 训推分离 colocate、分布式训练拓扑感知的核心工具。

> [!note] 三句话定位
> - **是什么**：一组 bundle 按策略（共置/分散）预留并放置到集群的机制。
> - **为什么**：普通调度无法表达"这组 actor 要同节点"或"要分散不同节点"；PG 把放置约束变一等公民。
> - **典型用途**：RLHF 的 Learner+PolicyServer colocate（省权重同步）、tensor parallel 各 rank 同节点（用 NVLink）、容错分散。

> [!note] 解答：如何避免 PG 造成碎片（两 actor 各 4 卡却各占一个 8 卡节点）
> **先诊断你这个场景的根因**：两个 4-GPU actor 被放到了**两个不同节点**，每节点只用 4 卡、空 4 卡——这是典型的 **bin-packing 碎片**。根因不是 PG 本身，而是**策略选错了 + bundle 粒度没设计好**：你大概率用了 `SPREAD`（或默认调度把两个独立 actor 各自 first-fit 到不同节点），调度器"尊重"了分散意图，结果每节点剩半截没人能凑上。
>
> ### 一、碎片是怎么产生的
> 设集群有 $N$ 个节点，每节点 $G=8$ GPU。两个 actor 各要 4 GPU。
>
> | 放法 | 占用节点 | 每节点用 | 总利用率 $\eta$ | 碎片率 $1-\eta$ |
> |---|---|---|---|---|
> | 分散两节点（你遇到的） | 2 | 4/8 | $\frac{2\times4}{2\times8}=50\%$ | **50%** |
> | 装箱同节点（应做的） | 1 | 8/8 | $\frac{8}{1\times8}=100\%$ | 0% |
>
> 利用率 $\eta=\frac{\sum \text{已用 GPU}}{\sum \text{节点 GPU}}$，碎片率 $=1-\eta$。你那 50% 空着，就是因为调度器把两个本可凑满一个节点的 actor 拆开放了。
>
> ### 二、根因：策略选错 + bundle 粒度错
> 1. **用了 SPREAD/默认分散**：SPREAD 的语义就是"尽量不同节点"，4+4 自然各占一节点。但你这两个 actor **本该 colocate**（一起放省通信 / 一起填满节点），不该用 SPREAD。
> 2. **没用 PG / 用了两个独立 actor**：两个独立的 `@ray.remote(num_gpus=4)` actor 各自 first-fit，调度器看不到它俩的关系，谁先有空就放谁，极易分散。
> 3. **bundle 粒度 = 单 actor 需求**：建了含 2 个 `{GPU:4}` bundle 的 PG，但策略是 SPREAD → 必然分散。
>
> ### 三、解法：PACK/STRICT_PACK + 一个 PG 把它们装箱
> **核心思路**：把两个 actor 建成**同一个 PG 的两个 bundle**，策略用 `PACK`（尽量塞同节点）或 `STRICT_PACK`（必须同节点），让调度器一次性把它们装箱到同一节点：
>
> ```python
> import ray
> from ray.util.placement_group import placement_group
>
> # 关键：一个 PG 含 2 个 4-GPU bundle，策略 STRICT_PACK → 强制同节点
> pg = placement_group(
>     [{"GPU": 4, "CPU": 8} for _ in range(2)],   # 2 bundle × 4 GPU = 8 GPU
>     strategy="STRICT_PACK"                       # 必须全塞同一 8-GPU 节点
> )
> ray.get(pg.ready())                              # 等装箱完成，做不到则 PG 创建失败
>
> # 两个 actor 各绑一个 bundle，物理同节点
> a0 = MyActor.options(placement_group=pg, placement_group_bundle_index=0).remote()
> a1 = MyActor.options(placement_group=pg, placement_group_bundle_index=1).remote()
> ```
> - **STRICT_PACK** 保证两个 bundle 全在同节点 → 4+4=8 正好填满 8 卡节点，0 碎片；
> - 若用 **PACK**（非 strict），调度器"尽量"同节点，塞不下才溢出——若集群只剩一个 8 卡节点能放下就同节点，否则可能分散。要 100% 保证同节点用 STRICT_PACK。
>
> ### 四、verl 怎么避免这种碎片（实证）
> verl 的 `RayWorkerGroup._init_with_resource_pool`（`single_controller/ray/base.py` L552-555）默认就是装箱：
> ```python
> strategy = "PACK"
> if bin_pack:                 # RayWorkerGroup 默认 bin_pack=True
>     strategy = "STRICT_PACK"
> pgs = resource_pool.get_placement_groups(strategy=strategy, ...)
> ```
> 即 verl 把**同一 resource pool 的所有 worker 用 STRICT_PACK 强制同节点**，正是为了避免你遇到的"各占一半节点"碎片。所以用 verl 时一个 `RayResourcePool` 的 worker 会被装箱，跨 pool 才可能分散。
>
> ### 五、通用避免碎片的五条工程经验
> | 手段 | 怎么做 | 适用 |
> |---|---|---|
> | **1. 该 colocate 的用 STRICT_PACK** | 一个 PG 装 N 个 bundle，强制同节点 | actor 间有通信/要填满节点 |
> | **2. 调大 bundle 粒度** | 与其 2×4 不如 1×8（一个 actor 8 卡） | 需求可合并时 |
> | **3. 精确声明 num_gpus** | 不要 `num_gpus=4` 实际只用 2，多占即碎片 | 按 real footprint 声明 |
> | **4. 用 bin_pack 选项** | verl `RayWorkerGroup(bin_pack=True)` | 同 pool worker 装箱 |
> | **5. autoscaler + 反碎片调度** | Ray autoscaler 按 pending 需求扩节点，凑满再放 | 多节点集群 |
>
> ### 六、误区
> - ❌ "PG 自动避免碎片" → **PG 也可能制造碎片**。SPREAD 的 PG 必然分散造碎片；bundle 粒度和策略选错，PACK 也可能塞半节点。PG 是工具不是银弹，得设计。
> - ❌ "两个 actor 各 4 卡，调度器会自动凑满 8 卡节点" → 不会。两个**独立** actor 各 first-fit，调度器不管它俩能不能凑；必须用**同一 PG + STRICT_PACK** 让调度器看到它俩的关系。
> - ❌ "STRICT_PACK 永远最好" → 不。容错敏感（副本要分散）用 SPREAD；TP 必须 NVLink 同节点才用 STRICT_PACK；colocate 优先同节点但可容忍溢出用 PACK。策略选错要么造碎片要么造单点故障。
> - ❌ "碎片无害" → 大集群里 50% 碎片 = 一半卡空转 = 一半钱白烧；且待机 PG 不释放（见 §8.3），碎片会长期占着。
>
> 相关：[[scheduling策略]]、[[GPU绑定]]、[[Ray与分布式调度]] §3.2、verl `RayWorkerGroup` bin_pack。

## 2. 为什么需要它（动机与背景）

### 2.1 放置约束的需求

ML 工作负载常有**放置约束**：

- **RLHF 训推分离**：Learner（8 GPU 训练）+ PolicyServer（1 GPU 推理）需**同节点**——若跨节点，[[weight sync mechanism]] 的权重传输走网络，慢且占带宽。colocate 后走 NVLink/同机内存，快几个量级。
- **Tensor Parallel**：各 rank 需**同节点**（用 [[NCCL通信拓扑]] 的 NVLink 高速通信），跨节点走 PCIe/网络慢。
- **容错**：actor 副本需**分散不同节点**——一节点挂不全挂（如 3 副本分散 3 节点，挂 1 还剩 2）。

普通调度（[[resource request]] + first-fit）只看单 task 资源，不知道"Learner 和 PolicyServer 要一起放"。结果可能 Learner 在节点 A、PolicyServer 在节点 B，权重同步跨网络，慢。

### 2.2 碎片与预留

普通调度"提交时找资源"会因**碎片**失败：4-GPU worker 残 1 GPU 空闲，`num_gpus=2` 的 task 用不了（需单 worker ≥2）。PG 的 bundle **提前预留**，保证一组资源原子可用——创建 PG 时要么全部分配成功、要么全不分配（all-or-nothing），避免半途碎片。

### 2.3 让放置约束可表达

PG 把"一组资源 + 放置策略"打包，调度器整体规划：

```python
pg = placement_group([{1 GPU}, {1 GPU}, {1 GPU}], strategy='PACK')
ray.get(pg.ready())   # 等预留完成
ray.remote(Actor).options(placement_group=pg, placement_group_bundle_index=0).remote()
```

明确说"这 3 个 actor 要 PACK（共置）"，调度器据此一次放好。


## 3. 核心概念详解

### 3.1 bundle（资源包）

bundle 是 PG 的基本单元，一个 `{资源: 数量}` 字典，如 `{"GPU": 1, "CPU": 2}`。PG 由 $N$ 个 bundle 组成，每个 bundle 对应一个"放置槽"，task/actor 可绑定到某 bundle（`placement_group_bundle_index`）。bundle 是**原子**的——一个 bundle 必须整体放在单 worker，不能跨 worker 拼。

### 3.2 四种放置策略

| 策略 | 含义 | 典型用途 |
|---|---|---|
| **PACK** | 尽量塞同一节点（最少节点数），塞不下才用下一节点 | colocate（省通信） |
| **SPREAD** | 尽量分散不同节点（最多节点数） | 容错（一节点挂不全挂） |
| **STRICT_PACK** | **严格**所有 bundle 同一节点，做不到就失败 | 必须 NVLink 同节点（TP） |
| **STRICT_SPREAD** | **严格**每 bundle 不同节点，做不到就失败 | 严格容错（副本跨机架） |

> [!tip] STRICT 与非 STRICT 的区别
> PACK 是"尽量同节点"（塞不下可溢出到其他节点），STRICT_PACK 是"必须全同节点"（做不到则 PG 创建失败）。TP 必须全同节点用 NVLink → STRICT_PACK；RLHF colocate 优先同节点但可容忍部分跨节点 → PACK。

### 3.3 colocate（共置）

PG 的 PACK/STRICT_PACK 让一组 actor 同节点。RLHF 中 Learner + PolicyServer colocate 的好处：

- [[weight sync mechanism]] 走同机内存/NVLink（数十 GB/s），不走网络（十 GB/s）。
- 减少 NCCL 跨节点 all-reduce。
- 共享 object store 零拷贝。

### 3.4 拓扑感知

Ray 的 PG 可结合 `--resources` 表达拓扑（如不同机架、不同 NUMA）。高级放置（如"同机架不同节点"）用自定义资源 + STRICT_SPREAD 实现。GPU 拓扑感知（NVLink vs PCIe）见 [[GPU绑定]]。

### 3.5 创建与使用

```python
import ray
from ray.util.placement_group import placement_group

pg = placement_group(
    [{"GPU": 1, "CPU": 2} for _ in range(4)],  # 4 bundle
    strategy="PACK"
)
ray.get(pg.ready())                            # 等待预留完成

# 把 actor 绑到 PG 的 bundle 0
actor = MyActor.options(placement_group=pg, placement_group_bundle_index=0).remote()
```


## 4. 数学原理 / 公式

### 4.1 colocate 通信节省

Learner 同步权重给 PolicyServer，权重大小 $W$（GB）。跨节点带宽 $\beta_{\text{net}}$（如 25 GB/s IB），同节点 NVLink 带宽 $\beta_{\text{nvlink}}$（如 300 GB/s）。同步时间：

$$
T_{\text{sync}} = \frac{W}{\beta}, \quad \beta \in \{\beta_{\text{net}}, \beta_{\text{nvlink}}\}
$$

colocate（NVLink）vs 跨节点（网络）加速比：

$$
S = \frac{T_{\text{net}}}{T_{\text{nvlink}}} = \frac{\beta_{\text{nvlink}}}{\beta_{\text{net}}} \approx \frac{300}{25} = 12\times
$$

> [!note] 14GB 权重同步
> 14GB 模型（如 7B fp16）：网络 14/25=0.56s，NVLink 14/300=0.047s。RLHF 每轮迭代同步一次，colocate 省 ~0.5s/iter，长时间训练省显著。这是 RLHF 框架（verl/OpenRLHF）默认 colocate 的根因。

### 4.2 碎片率

集群有 $W$ 个 worker，每 worker $G$ GPU。普通调度下，随机提交 task 产生的碎片率随占用率上升。PG 的 bundle 预留是 all-or-nothing：要么 $N$ bundle 全放下、要么全不放（排队），避免半放置碎片。

### 4.3 策略对比

| 策略 | 节点数 | 通信 | 容错 | 适用 |
|---|---|---|---|---|
| PACK | 最少 | 最低（同节点） | 差（同节点同挂） | colocate |
| SPREAD | 最多 | 高（跨节点） | 好（分散） | 容错 |
| STRICT_PACK | 1（全同） | 最低 | 最差 | TP/NVLink |
| STRICT_SPREAD | =N bundle | 最高 | 最好 | 副本跨机架 |


## 5. 代码示例（可选

### 5.1 纯 Python 模拟 PG 调度（PACK vs SPREAD）

```python
class Worker:
    def __init__(self, wid, gpus, cpus):
        self.wid=wid; self.gpus=gpus; self.cpus=cpus; self.used_g=0; self.used_c=0
    def available(self, g, c):
        return (self.gpus-self.used_g)>=g and (self.cpus-self.used_c)>=c
    def allocate(self, g, c):
        if self.available(g,c): self.used_g+=g; self.used_c+=c; return True
        return False
    def __repr__(self): return f'W{self.wid}({self.used_g}/{self.gpus}g,{self.used_c}/{self.cpus}c)'

class Bundle:
    def __init__(self, gpus, cpus, tag=''): self.gpus=gpus; self.cpus=cpus; self.tag=tag

class PlacementGroup:
    def __init__(self, bundles, strategy): self.bundles=bundles; self.strategy=strategy; self.placed=[]

class Scheduler:
    """模拟 Ray PG 调度: 按策略放置 bundle."""
    def __init__(self, workers): self.workers=workers
    def place(self, pg):
        if pg.strategy == 'PACK':                    # 尽量塞同一 worker (co-locate)
            for b in pg.bundles:
                for w in self.workers:
                    if w.allocate(b.gpus, b.cpus):
                        pg.placed.append((b.tag, w.wid)); break
                else: return False
            return True
        elif pg.strategy == 'SPREAD':               # 每个 bundle 到不同 worker (容错)
            used = set()
            for b in pg.bundles:
                for w in self.workers:
                    if w.wid not in used and w.allocate(b.gpus, b.cpus):
                        pg.placed.append((b.tag, w.wid)); used.add(w.wid); break
                else: return False
            return True

# 集群: 2 个 4-GPU worker
ws1 = [Worker(0,4,8), Worker(1,4,8)]
pg1 = PlacementGroup([Bundle(1,2,'learner'), Bundle(1,2,'ps1'), Bundle(1,2,'ps2')], 'PACK')
print('PACK:', Scheduler(ws1).place(pg1), pg1.placed, ws1)
# 全塞 W0 (co-locate): W0(3/4g,6/8c), W1 空

ws2 = [Worker(0,4,8), Worker(1,4,8)]
pg2 = PlacementGroup([Bundle(1,2,'a'), Bundle(1,2,'b')], 'SPREAD')
print('SPREAD:', Scheduler(ws2).place(pg2), pg2.placed, ws2)
# 各到不同 worker: W0(1/4g), W1(1/4g)
```

### 5.2 真实 Ray API 对照

```python
# import ray
# from ray.util.placement_group import placement_group
# pg = placement_group([{"GPU":1,"CPU":2} for _ in range(4)], strategy="PACK")
# ray.get(pg.ready())
# actor = PolicyServer.options(placement_group=pg, placement_group_bundle_index=0).remote()
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[resource request]]（bundle 的资源声明）、Ray 调度器、worker 资源上报。
- **下游（应用）**: [[scheduling策略]]（PG 的策略是其一部分）、[[GPU绑定]]（PG + 绑定做拓扑感知）、[[weight sync mechanism]]（colocate 加速权重同步）、[[训推分离]]（PD colocate）、[[Tensor Parallel]]（各 rank 同节点用 NVLink）。
- **对比 / 易混**:
  - **PG vs 普通调度**：普通调度按单 task 找资源（无放置约束）；PG 把一组 bundle + 策略打包，保证共置/分散。
  - **PG vs K8s pod affinity/anti-affinity**：K8s 用 affinity 表达 pod 间放置；Ray PG 用 bundle + 策略表达，更细粒度（bundle 级、动态）。
  - **PG 预留 vs 普通调度动态分配**：PG 创建时预留（all-or-nothing），普通调度提交时动态找。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 PG 执行 task
> 不执行。PG 只**预留**资源 + **放置**，task/actor 仍需 `.remote()` 创建并绑到 PG（`options(placement_group=pg)`）。PG 是"地皮"，task 是"房子"——PG 划地、房子建在指定地块。

> [!warning] 误区 2：bundle 可跨 worker
> 不可。bundle 是**原子**的，必须整体放单 worker。`{"GPU":2}` 的 bundle 不能拆成 1+1 跨两 worker。要跨 worker 联合用多 bundle。

> [!warning] 误区 3：PACK 与 STRICT_PACK 混淆
> PACK 是"尽量同节点"（塞不下可溢出），STRICT_PACK 是"必须全同节点"（做不到则失败）。TP 必须 NVLink 全同节点 → STRICT_PACK；RLHF colocate 优先同节点可容忍溢出 → PACK。

> [!warning] 误区 4：PG 创建不 ready 就用
> `placement_group(...)` 异步创建，需 `ray.get(pg.ready())` 等待预留完成。不等就绑 actor 会绑到未就绪 PG，行为未定义。

> [!warning] 误区 5：以为 colocate 总是好的
> colocate 省通信但**减少容错**（同节点同挂）。容错敏感场景（服务高可用）用 SPREAD 分散。colocate vs spread 是通信效率与容错的权衡。


## 8. 延伸细节

### 8.1 RLHF 中的 PG 应用

verl/OpenRLHF 的典型 PG：

- Learner actor（8 GPU）+ PolicyServer actor（1 GPU）+ RewardServer（1 GPU）→ colocate 同节点（PACK）。
- 多个 PolicyServer 副本分散不同节点（SPREAD）做负载均衡。
- [[placement group]] 保证 Learner 与 PolicyServer 同节点，[[weight sync mechanism]] 走 NVLink。

### 8.2 与 Tensor Parallel 的关系

TP 的各 rank 需同节点用 NVLink（[[NCCL通信拓扑]]）。用 STRICT_PACK 的 PG 把 $N$ rank 放同节点，保证 NVLink 通信。Pipeline Parallel 跨节点则用普通调度 + SPREAD。

### 8.3 待机资源与成本

PG 预留的资源在 task 未绑定时**仍占用**（不释放）。若 PG 创建后不用，资源浪费。RLHF 中 PG 与 actor 生命周期绑定（actor 销毁时 PG 释放）。生产中需注意 PG 的待机成本。

### 8.4 动态 PG 与重调度

PG 创建后若 worker 挂了，Ray 可重调度 PG 的 bundle 到其他 worker（非 STRICT 策略）。STRICT_PACK 若节点挂了，PG 失败需重建。容错设计需考虑 PG 重建。

### 8.5 与 object store locality 的协同

PG 的 colocate 让生产-消费 task 同节点，[[object store]] 的 locality 命中（同 worker 零拷贝）。这是 PG + object store 的协同——PG 管"放哪"，object store 管"数据怎么传"，同节点时两者叠加最优。

---
相关: [[调度机制]] | [[resource request]] | [[scheduling策略]] | [[GPU绑定]] | [[weight sync mechanism]] | [[训推分离]] | [[Tensor Parallel]] | [[NCCL通信拓扑]] | [[policy training (PT)]] | [[rollout worker]] | [[object store]]
