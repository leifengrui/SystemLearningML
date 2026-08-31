先记住一句：

> `train_batch_size` 决定“一轮收集多少数据”；  
> `mini batch` 决定“一次 PPO 优化用其中多少数据”；  
> `micro batch` 决定“一次前向/反向实际塞进一张 GPU 多少数据”。

假设 `rollout.n=1`，并且有 **8 张 GPU**：

```
train_batch_size = 1024
```

一个 step 先收集 1024 条样本。

```
ppo_mini_batch_size = 256
```

把 1024 条样本分成：

```
1024 / 256 = 4 个 mini-batch
```

每个 mini-batch 在 8 张 GPU 上分摊：

```
256 / 8 = 32 条/GPU
```

```
ppo_micro_batch_size_per_gpu = 8
```

所以每张 GPU 的 32 条还要切成：

```
32 / 8 = 4 个 micro-batch
```

因此一个 step 大致是：

```
1024 样本
→ 4 个 mini-batch
→ 每个 mini-batch 4 个 micro-batch
→ 每个 mini-batch 完成 1 次参数更新
→ 共 4 次 optimizer.step()
```

如果 `rollout.n=4`，每个 prompt 生成 4 条回答，实际 trajectory 数会变成 `1024×4=4096`。

## 1. verl 中的完整数据流

```
Dataset
  ↓
gen_batch_size              # DataLoader 每次取多少 prompt
  ↓
train_batch_size            # 一轮 rollout 收集多少 prompt
  ↓ rollout.n
trajectory batch             # 实际得到的 response 数
  ↓
ppo_mini_batch_size          # PPO 每次优化使用多少 trajectory
  ↓
ppo_micro_batch_size_per_gpu # 每张 GPU 一次前向/反向处理多少样本
  ↓
梯度累积
  ↓
optimizer.step()
```

对应代码：

- DataLoader 使用 `gen_batch_size`，未设置时退回 `train_batch_size`：[ray_trainer.py (line 405)](/Users/lei/Documents/codes/code4reading/verl/verl/trainer/ppo/ray_trainer.py:405)
- PPO mini-batch 切分：[engine_workers.py (line 242)](/Users/lei/Documents/codes/code4reading/verl/verl/workers/engine_workers.py:242)
- micro-batch 前向、反向和梯度累积：[fsdp/transformer_impl.py (line 700)](/Users/lei/Documents/codes/code4reading/verl/verl/workers/engine/fsdp/transformer_impl.py:700)

## 2. 三个核心概念

|名称|verl 配置|含义|
|---|---|---|
|Global batch size|`data.train_batch_size` 或元数据中的 `global_batch_size`|所有 GPU 合起来看到的数据量|
|Mini-batch|`actor.ppo_mini_batch_size`|PPO 算法层面的一次更新数据|
|Micro-batch|`actor.ppo_micro_batch_size_per_gpu`|单张 GPU 一次 forward/backward 的数据量|

### Global batch size

传统深度学习中，global batch size 通常是：

```
单卡 batch × GPU 数 × 梯度累积步数
```

但在 verl PPO 中，`data.train_batch_size` 有特殊含义：

```
data:
  train_batch_size: 1024
```

它表示一轮 rollout 采样多少个 prompt。代码文档明确说明：[ppo.md (line 32)](/Users/lei/Documents/codes/code4reading/verl/docs/algo/ppo.md:32)

如果：

```
train_batch_size = 1024
rollout.n = 4
```

那么实际得到：

```
trajectory 数 = 1024 × 4 = 4096
```

所以不要简单把 PPO 的 `train_batch_size` 当成“一次 optimizer.step 的 batch”。

### Mini-batch

```
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 256
```

表示 PPO 每次拿一部分 trajectory 做优化。verl 会把它乘以 `rollout.n`：

```
ppo_mini_batch_size = config.actor.ppo_mini_batch_size
ppo_mini_batch_size = ppo_mini_batch_size * config.rollout.n
```

见：[ray_trainer.py (line 1327)](/Users/lei/Documents/codes/code4reading/verl/verl/trainer/ppo/ray_trainer.py:1327)

例子：

```
train_batch_size = 1024 prompts
rollout.n = 4
总 trajectory = 4096

ppo_mini_batch_size = 256 prompts
实际 mini-batch = 256 × 4 = 1024 trajectories

mini-batch 数 = 4096 / 1024 = 4
```

verl 会为每个 mini-batch 执行一次 `train_batch()`，最终调用一次 `optimizer.step()`。

### Micro-batch

```
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: 8
```

表示每张 GPU 一次只处理 8 条样本，避免显存溢出。

假设：

```
GPU 数 = 8
global mini-batch = 1024 trajectories
```

每张 GPU 得到：

```
local mini-batch = 1024 / 8 = 128
```

每张 GPU 每次处理 8 条：

```
梯度累积步数 = 128 / 8 = 16
```

16 次 forward/backward 后，才进行一次：

```
optimizer.step()
```

代码中，global mini-batch 会先除以 data parallel size：[engine_workers.py (line 266)](/Users/lei/Documents/codes/code4reading/verl/verl/workers/engine_workers.py:266)

## 3. 一个完整数字例子

配置：

```
data:
  train_batch_size: 1024

actor_rollout_ref:
  rollout:
    n: 4
  actor:
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 8
    ppo_epochs: 2
```

假设有 8 张 GPU：

```
一轮 prompt 数                 = 1024
一轮 trajectory 数             = 1024 × 4 = 4096

每个 PPO mini-batch             = 256 × 4 = 1024 trajectory
mini-batch 数                  = 4096 / 1024 = 4

每张 GPU 的 mini-batch          = 1024 / 8 = 128
每张 GPU 的 micro-batch         = 8
每个 mini-batch 的累积步数      = 128 / 8 = 16

ppo_epochs = 2
总 optimizer step              = 4 × 2 = 8
```

关系可以写成：

```
global mini-batch
    = micro_batch_per_gpu
      × data_parallel_size
      × gradient_accumulation_steps
```

也就是：

```
1024 = 8 × 8 × 16
```

## 4. `ppo_micro_batch_size` 和 `per_gpu` 的区别

旧配置：

```
ppo_micro_batch_size: 64
```

这是 global micro-batch，已经被标记为 deprecated。

新配置：

```
ppo_micro_batch_size_per_gpu: 8
```

这是每张 GPU 的 micro-batch，推荐使用。

配置类中明确标注了旧字段已废弃：[actor.py (line 153)](/Users/lei/Documents/codes/code4reading/verl/verl/workers/config/actor.py:153)

## 5. Dynamic batch：不再按样本数切，而按 token 数切

当：

```
actor:
  use_dynamic_bsz: true
  ppo_max_token_len_per_gpu: 16384
```

micro-batch 不再固定为“每次 N 条样本”，而是根据 token 总量动态切分：

```
num_micro_batches = ceil(total_seqlen / max_token_len)
```

见：[seqlen_balancing.py (line 394)](/Users/lei/Documents/codes/code4reading/verl/verl/utils/seqlen_balancing.py:394)

例如：

```
样本 A：1000 tokens
样本 B：8000 tokens
样本 C：2000 tokens
样本 D：7000 tokens
```

虽然都是 4 条样本，但计算量差异很大。动态 batch 可能切成：

```
micro-batch 1：A + B = 9000 tokens
micro-batch 2：C + D = 9000 tokens
```

此时每个 micro-batch 的样本数可能不同，但 token 预算接近。

verl 文档也说明，micro-batch 配置主要用于控制单次 forward/backward 的显存，不应改变算法本身的收敛行为：[ppo.md (line 26)](/Users/lei/Documents/codes/code4reading/verl/docs/algo/ppo.md:26)

## 6. 其他常见 batch 描述

### `gen_batch_size`

```
data:
  gen_batch_size: 32
```

DataLoader 每次取 32 个 prompt，用于 rollout 生成。

它只是“生成侧取数据的粒度”，不等于 PPO 优化 batch。

代码：[ray_trainer.py (line 405)](/Users/lei/Documents/codes/code4reading/verl/verl/trainer/ppo/ray_trainer.py:405)

### `log_prob_micro_batch_size_per_gpu`

```
actor_rollout_ref:
  rollout:
    log_prob_micro_batch_size_per_gpu: 8
```

计算 rollout response 的 log probability 时，每张 GPU 一次处理多少样本。

这是推理/打分阶段的 micro-batch，不是训练阶段的 micro-batch。

### `forward_micro_batch_size_per_gpu`

只做 forward、不反向传播时使用，例如 validation、critic 推理等。

### `local batch size`

某个 GPU 或某个 DP rank 实际收到的 batch：

```
local batch = global batch / data_parallel_size
```

### `effective batch size`

真正用于一次参数更新的数据量。静态训练中通常是：

```
effective batch
= micro-batch per GPU
  × GPU 数
  × 梯度累积步数
```

在 PPO 中，通常对应一个 global `ppo_mini_batch_size`，而不是整个 `train_batch_size`。

### `token batch size`

以 token 数而不是样本数描述 batch：

```
token batch = 所有样本有效 token 数之和
```

大模型训练中，token batch 往往比 sample batch 更能反映真实显存和计算量。

### `gradient accumulation steps`

梯度累积次数：

```
gradient_accumulation_steps
= local_mini_batch_size / micro_batch_size_per_gpu
```

它让小显存 GPU 模拟更大的 batch，但参数不会在每个 micro-batch 后立即更新。

## 7. 最容易混淆的地方

1. `train_batch_size`：PPO 一轮收集多少 prompt。
2. `ppo_mini_batch_size`：PPO 一次优化多少 trajectory。
3. `ppo_micro_batch_size_per_gpu`：每张 GPU 一次 forward/backward 多少样本。
4. `gen_batch_size`：rollout DataLoader 一次取多少 prompt。
5. `log_prob_micro_batch_size_per_gpu`：计算 logprob 时一次处理多少样本。
6. `max_token_len_per_gpu`：动态 batch 模式下，每张 GPU 的 token 上限。

可以用这句话判断：

> 看它位于数据流的哪一层：采样层是 `gen/train batch`，算法更新层是 `mini-batch`，显存执行层是 `micro-batch`。