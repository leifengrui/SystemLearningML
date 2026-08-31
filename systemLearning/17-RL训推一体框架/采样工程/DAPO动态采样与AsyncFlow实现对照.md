# DAPO 动态采样与 AsyncFlow 实现对照

> 本笔记用于区分 DAPO 论文中的算法语义与 AsyncFlow/rollout 系统中的工程实现，不把二者混写成同一个模块。

## 1. 结论

DAPO 论文明确提出的是：**过滤全对/全错的 prompt-group，并持续补采样，直到 batch 被有效样本填满**。

在 AsyncFlow 设计中，这一语义被具体拆成：

```text
rollout -> filter -> 有效组进入 TQ
                 -> 无效组进入 Replay
                 -> replay 时调整 n_sample
                 -> refill，直到满足训练批次需求
```

因此：`filter` 直接来自论文；`Replay` 和 per-group 动态 `n_sample` 是系统实现层的扩展，不是论文中逐字定义的独立组件。

## 2. 论文中的 Dynamic Sampling

### 2.1 过滤条件

对同一个 prompt 采样 (G) 个回答。如果全对或全错，则组内 reward 方差为 0，GRPO advantage 没有有效信号。DAPO 保留满足下式的组：

$$
0 < \left|\{o_i\mid \operatorname{is\_equivalent}(a,o_i)\}\right| < G
$$

也就是正确回答数量既不是 0，也不是 (G)。

原文对应：DAPO 论文第 3.2 节、公式 (11)，以及算法 1 中“过滤输出并加入动态采样缓冲区”的步骤。

### 2.2 论文所说的“动态”

论文说的是每个 batch 的采样成本是动态的：被过滤的 prompt 越多，就需要额外采样越多的新 prompt。训练前持续采样，直到 batch 被有效 prompt 填满。

论文没有明确规定：

- 一个持久化的 ReplayBuffer；
- rejected group 的保存格式和优先级；
- replay 时的采样倍率；
- 每个 prompt 独立的 `n_sample` 调度策略；
- TQ、credit/seat、reserve/release 等资源生命周期。

## 3. 与 AsyncFlow 设计的对应关系

| AsyncFlow 设计概念 | 与论文的关系 | 说明 |
|---|---|---|
| `filter` | 直接对应 | 丢弃 reward 全相同、无法产生组内 advantage 的 group。 |
| `Replay` | 语义对应、实现新增 | 用存储的 rejected group 触发后续重采样；论文只描述“继续采样”，没有规定必须 replay。 |
| 动态 `n_sample` | 工程扩展 | 论文公式中的组大小 (G) 通常固定；设计允许某个 group 在 replay 时增加采样条数。 |
| `refill` | 对应论文“直到 batch 填满” | 将有效组补足到训练侧所需数量。 |
| TQ / credit-release | 系统层新增 | 用于异步流水线的席位、容量和在途轨迹管理，论文未涉及。 |

## 4. 一个容易混淆的差异

论文的过滤单位是固定 (G) 条回答组成的 prompt-group，要求：

```text
0 < correct_count < G
```

而 AsyncFlow 设计还要处理 partial rollout、动态 group、大于或小于原始组大小、staleness 和资源回收。因此设计中的“只有 1 条轨迹也可以保留”等规则，是为了适配系统生命周期的补充规则，不能直接当成 DAPO 论文的原始定义。

## 5. 论文与笔记入口

- [[DAPO]]：DAPO 算法梳理与四个核心 trick
- [[GRPO]]：GRPO 基础算法梳理
- [[sequence packing与动态采样]]：采样工程与动态采样背景
- [[05-LLM-RL对齐/对齐算法/论文全文翻译/DAPO论文全文中文翻译]]：DAPO 论文全文中文翻译
- [[17-RL训推一体框架/GRPO体系/论文全文翻译/DeepSeekMath_GRPO论文全文中文翻译]]：DeepSeekMath/GRPO 论文全文中文翻译

## 6. 术语边界

- **KL 惩罚项 / KL loss**：约束当前策略不要偏离 reference policy；它和 Dynamic Sampling 是不同维度的机制。
- **熵坍塌**：当前策略自身的分布变得过于集中，探索减少；DAPO 用 Clip-Higher 处理这一问题。
- **全对/全错组**：组内 reward 方差为 0，导致 GRPO advantage 无有效梯度；DAPO 用 Dynamic Sampling 处理。
