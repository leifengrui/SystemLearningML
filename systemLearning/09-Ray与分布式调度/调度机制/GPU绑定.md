# GPU绑定

> **所属章节**: [[调度机制]]
> **所属模块**: [[09-Ray与分布式调度]]
> **别名**: GPU绑定 / GPU affinity / GPU pinning / CUDA_VISIBLE_DEVICES / visible_cuda_devices
> **难度**: 中（需懂 [[resource request]] + GPU 物理）


## 1. 一句话定义

**GPU绑定** 是把 Ray 的 task/actor **物理绑定到指定 GPU 卡**（如 `RAY_visible_cuda_devices=0,1` 或 worker 启动 `--num-gpus` + `CUDA_VISIBLE_DEVICES`），让 [[resource request]] 的逻辑声明（`num_gpus=1`）升级为**物理独占**——该 task 只看到并独占绑定的物理 GPU，避免与其他 task 争抢算力/显存。它是 [[placement group]] 之后的进一步约束：PG 管"放哪个 worker"，GPU绑定 管"用 worker 里哪几张卡"。

> [!note] 三句话定位
> - **是什么**：把 task/actor 物理钉到指定 GPU 卡，逻辑声明变物理独占。
> - **为什么**：`num_gpus` 只记账，多 task 可能物理共享一卡（fractional/超额）→ 算力争抢、显存 OOM；绑定强制隔离。
> - **关键认知**：`num_gpus=1` ≠ 绑定。绑定需显式 `visible_cuda_devices` 或 worker 独占配置。


## 2. 为什么需要它（动机与背景）

### 2.1 逻辑声明 vs 物理现实

[[resource request]] 的 `num_gpus=1` 是**调度记账**：Ray 保证该 worker 上 ≤4 个 `num_gpus=1` task（不超额调度）。但物理上：

- 4 个 task 共享 worker 的 4 GPU，**默认不绑定** → Ray 给每个 task 看 4 卡，task 可能全用 0 号卡（争抢），或分散（无保证）。
- `num_gpus=0.5` 更甚——2 个 task 共享 1 GPU 配额，物理共享整张卡。

结果：算力争抢（kernel 串行化）、显存 OOM（两 task 各加载大模型超显存）、性能不可预测。

### 2.2 物理绑定的好处

GPU绑定 把 task 钉到特定卡：

- **独占算力**：绑定的卡不被其他 task 抢，throughput 稳定。
- **显存隔离**：绑定的卡显存专用，不超显存（除非自己超）。
- **拓扑利用**：可绑定到 NVLink 互联的卡对（[[Tensor Parallel]] 高速通信）。

### 2.3 RLHF / 训练场景的需求

- 训练 actor（Learner）需独占 8 GPU 全算力 → 必须绑定，否则推理 task 抢算力拖慢训练。
- 推理 actor（PolicyServer）需独占 1 GPU → 绑定保证推理延迟稳定。
- 两者同节点 colocate（[[placement group]]）时，必须各自绑定不同 GPU，否则互抢。

没有绑定，colocate 的 Learner + PolicyServer 可能抢同一卡，性能崩。


## 3. 核心概念详解

### 3.1 Ray 的 GPU 暴露机制

worker 启动时 `ray start --num-gpus=8` 声明 8 GPU。Ray 默认通过 `CUDA_VISIBLE_DEVICES` 给 task 暴露 GPU：

- task 声明 `num_gpus=2` → Ray 把 `CUDA_VISIBLE_DEVICES` 设为 2 个 GPU id（如 `0,1`），task 内 `torch.cuda.device_count()` 看到 2。
- 但这是**逻辑暴露**——Ray 不保证这 2 卡不被其他 task 同时看到（除非独占模式）。

### 3.2 visible_cuda_devices（显式绑定）

`.options(runtime_env={"env_vars": {"RAY_visible_cuda_devices": "0,1"}})` 或 worker 启动 `--resources` 自定义，强制 task 只看到指定物理 GPU。这是**物理绑定**：被绑的卡对 task 独占（其他 task 看不到）。

```python
actor = PolicyServer.options(
    num_gpus=1,
    runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "0"}}  # 绑到物理 0 号
).remote()
```

### 3.3 独占模式 vs 共享模式

| 模式 | 机制 | 算力 | 显存 | 适用 |
|---|---|---|---|---|
| **独占（绑定）** | 一卡一 task | 独占 100% | 独占 | 大模型训练/推理 |
| **共享（fractional）** | 一卡多 task（0.5+0.5） | 瓜分/争抢 | 共享 | 小模型、低负载 |
| **MPS** | 多进程共享一卡，并行 kernel | 按配额瓜分 | 共享 | 推理低延迟复用 |
| **time-slicing** | 时间片轮转 | 轮转 | 共享 | 默认 CUDA 行为 |

### 3.4 与 placement group 的协同

PG 管"放哪个 worker"（节点级），GPU绑定 管"用 worker 里哪几张卡"（卡级）。RLHF colocate 流程：

1. PG（PACK）把 Learner + PolicyServer 放同 worker。
2. GPU绑定：Learner 绑 GPU 0-7（8 卡训练），PolicyServer 绑 GPU 8（推理，若节点 9+ 卡）或分时。

若节点只有 8 卡，Learner 全占，PolicyServer 无法 colocate 同节点（无空闲卡）→ 需另一节点或 Learner 让出 1 卡。


## 4. 数学原理 / 公式

### 4.1 共享算力瓜分

$K$ 个 task 共享 1 GPU（无绑定，争抢），GPU 算力 $P$。理想瓜分每 task 得 $P/K$，但因 kernel 串行化（CUDA stream 不真并行，除非 MPS），实际：

$$
P_i \approx \frac{P}{K} \cdot \eta, \quad \eta < 1 \text{（串行化损失）}
$$

绑定（独占）时 $P_i = P$（无争抢）。throughput 对比：

$$
\frac{T_{\text{bound}}}{T_{\text{shared}}} = \frac{K}{\eta} > K
$$

> [!note] 共享退化
> 2 个 task 共享 1 GPU 无绑定时，实际不是各得 50%——CUDA 默认 time-slicing 轮转，两 task 交替执行 kernel，总 throughput ≈ 1 个 task（算力没增加），每 task 得 ~50% 但延迟翻倍。绑定独占得 100%，throughput 翻倍。这是绑定的核心收益。

### 4.2 显存约束

GPU 显存 $M$。$K$ 个 task 共享，各用 $m_i$：

$$
\sum_i m_i \le M
$$

不绑定时不保证——两 task 可能各加载 14GB 模型到 16GB 卡 → OOM。绑定（独占）时每 task 有自己 $M$，互不影响。

### 4.3 绑定开销

绑定本身零开销（只是设置 `CUDA_VISIBLE_DEVICES` 环境变量）。但独占模式降低集群 GPU 复用率（一卡一 task，不能 fractional 复用小 task）。这是绑定与利用率的权衡。


## 5. 代码示例（可选

### 5.1 纯 Python 模拟绑定 vs 共享

```python
class GPUDevice:
    """模拟一张物理 GPU: 独占模式拒绝第二 task, 共享模式允许."""
    def __init__(self, gid): self.gid = gid; self.occupants = 0
    def assign(self, exclusive):
        if exclusive and self.occupants >= 1:
            return False                          # 独占, 已有人, 拒绝
        self.occupants += 1
        return True
    def release(self): self.occupants -= 1

class GPUNode:
    """模拟一个 worker 的多 GPU."""
    def __init__(self, n_gpus): self.gpus = [GPUDevice(i) for i in range(n_gpus)]
    def bind(self, n, exclusive=True):
        """绑 n 张卡 (独占优先, 找空闲)."""
        bound = []
        for g in self.gpus:
            if len(bound) >= n: break
            if g.assign(exclusive): bound.append(g.gid)
        if len(bound) < n:                          # 回滚
            for gid in bound: self.gpus[gid].release()
            return []
        return bound

node = GPUNode(8)
# 训练 actor 独占 4 卡
t1 = node.bind(4, exclusive=True)
print('训练绑定 4 卡:', t1)                  # [0,1,2,3]
# 推理 actor 独占 1 卡 (不冲突, 用 4)
t2 = node.bind(1, exclusive=True)
print('推理绑定 1 卡:', t2)                  # [4]
# 再独占 5 卡 (只剩 3, 失败)
t3 = node.bind(5, exclusive=True)
print('再绑 5 卡 (失败):', t3)               # []
# 共享模式: 两个小 task 共用 5 号卡
s1 = node.gpus[5].assign(exclusive=False)
s2 = node.gpus[5].assign(exclusive=False)
print('共享 5 号卡:', s1, s2)                # True True
```

### 5.2 真实 Ray API 对照

```python
# import ray
# ray.init(num_gpus=8)
# # 独占绑定: runtime_env 设 CUDA_VISIBLE_DEVICES
# actor = PolicyServer.options(
#     num_gpus=1,
#     runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "0"}}
# ).remote()
# # PG colocate + 各自绑不同 GPU
# pg = placement_group([{"GPU":1}]*2, strategy="PACK")
# ray.get(pg.ready())
# learner = Learner.options(placement_group=pg, placement_group_bundle_index=0).remote()
# ps = PolicyServer.options(placement_group=pg, placement_group_bundle_index=1).remote()
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[resource request]]（绑定的逻辑基础）、[[placement group]]（PG 定节点、绑定定卡）、CUDA_VISIBLE_DEVICES / NVIDIA driver。
- **下游（应用）**: [[policy training (PT)]]（Learner 绑 8 GPU）、[[policy deployment (PD)]]（PolicyServer 绑 1 GPU）、[[Tensor Parallel]]（各 rank 绑 NVLink 卡对）、[[weight sync mechanism]]（绑 NVLink 卡对加速同步）。
- **对比 / 易混**:
  - **GPU绑定 vs [[resource request]] num_gpus**：num_gpus 是逻辑记账（调度配额），绑定是物理独占（钉到卡）。num_gpus=1 不等于绑定。
  - **GPU绑定 vs [[placement group]]**：PG 管节点级放置，绑定管卡级独占。两者协同：PG 定节点、绑定定卡。
  - **绑定 vs fractional GPU**：绑定独占整卡，fractional 共享一卡。大模型用绑定，小模型用 fractional。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 num_gpus=1 就是绑定
> 不是。`num_gpus=1` 只保证调度配额（worker 上 ≤4 个这样的 task），不保证物理独占。task 可能与其他 task 看到同一卡（默认 Ray 暴露所有可见 GPU）。需显式 `CUDA_VISIBLE_DEVICES` 或独占模式才真绑定。

> [!warning] 误区 2：CUDA_VISIBLE_DEVICES 理解错
> 它是环境变量，限制 task **看到**的 GPU id（逻辑重映射）。设 `CUDA_VISIBLE_DEVICES=2,3` → task 内 `torch.cuda.device(0)` 实际是物理 2 号。绑定即用此变量钉到物理卡。

> [!warning] 误区 3：colocate 不绑定导致互抢
> PG 把 Learner + PolicyServer 放同节点，若都不绑定 GPU，两 actor 可能抢同一卡 → 训练慢、推理延迟不稳。colocate 必须 + 绑定各自不同 GPU。

> [!warning] 误区 4：fractional GPU 等于绑定
> `num_gpus=0.5` 是共享配额（2 task 共 1 卡），不是绑定。fractional 仍可能算力争抢，绑定才是独占。

> [!warning] 误区 5：绑定一定能防 OOM
> 绑定防**跨 task** 显存争抢，但不防**本 task 自己超显存**。若一个 task 加载 14GB 模型 + 大 KV cache 超 16GB 卡，绑定了也 OOM。需配合 [[KV cache management]] / 显存管理。


## 8. 延伸细节

### 8.1 NVIDIA MPS（Multi-Process Service）

MPS 让多进程**真并行**执行 kernel on 同一 GPU（不像 time-slicing 轮转）。算力按配额瓜分，适合推理低延迟复用（多个小推理服务共享一卡）。Ray 可配 MPS 模式。但 MPS 有版本/驱动要求，且单进程崩溃影响整卡，慎用。

### 8.2 time-slicing（默认 CUDA 行为）

无 MPS 时，多 task 共享 GPU 走 time-slicing（时间片轮转），kernel 交替执行。总 throughput ≈ 单 task（没增加算力），每 task 延迟翻倍。这是 fractional GPU 的默认物理现实——配额分了但算力没真分。

### 8.3 RLHF 中的绑定实践

verl/OpenRLHF 的典型：

- Learner actor 绑 8 GPU（独占训练，DDP/FSDP）。
- PolicyServer actor 绑 1 GPU（独占推理）。
- 同节点 colocate 时，节点需 9+ GPU，或 Learner 让出 1 卡（用 7 训练 + 1 推理）。
- [[weight sync mechanism]] 绑 NVLink 卡对，走 NVLink 而非网络。

### 8.4 与 vLLM / SGLang 的关系

推理引擎（vLLM/SGLang）常自己管 GPU（`tensor_parallel_size`），Ray 的绑定需与引擎配合——给引擎绑定的 GPU 集合，引擎内部自己分。Ray 不越俎代庖管引擎内部的 tensor parallel。

### 8.5 拓扑感知绑定

高级绑定考虑 NVLink 拓扑（哪些卡对 NVLink 直连、哪些走 PCIe）。NVML 查询拓扑，绑定优先 NVLink 直连的卡对（TP 各 rank）。这是 [[NCCL通信拓扑]] 的上层应用。

---
相关: [[调度机制]] | [[resource request]] | [[placement group]] | [[scheduling策略]] | [[policy training (PT)]] | [[policy deployment (PD)]] | [[Tensor Parallel]] | [[weight sync mechanism]] | [[NCCL通信拓扑]] | [[KV cache management]] | [[GPU utilization]]
