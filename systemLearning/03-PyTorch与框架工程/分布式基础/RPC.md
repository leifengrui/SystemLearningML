# RPC

> **所属章节**: [[分布式基础]]
> **所属模块**: [[03-PyTorch与框架工程]]
> **难度**: 中高（分布式模型/流水并行的旧基础设施）
> **别名**: `torch.distributed.rpc`、分布式远程调用

## 1. 一句话定义

**RPC（Remote Procedure Call，远程过程调用）** 在 PyTorch 里指 `torch.distributed.rpc` 框架——它在 [[torch.distributed]] 的集合通信之上提供**远程函数调用、远程对象引用（RRef）、跨进程参数托管**等能力，让一个进程可以**像调本地函数一样调用另一个进程上的函数/方法**，并自动管理跨 rank 的张量所有权。它是早期 **Pipeline Parallel、模型并行（MP）、跨卡参数分片托管**的底层基础，现正被 `torch.distributed.pipelining`、[[Fully Sharded Data Parallel|FSDP]]、`DTensor` 等更新方案逐步替代，但**概念仍是理解分布式模型并行的钥匙**。

> [!note] 名字辨析
> "RPC" 是通用术语（任何远程调用都叫 RPC，如 gRPC、Thrift）。**本笔记里的 RPC 特指 PyTorch 的 `torch.distributed.rpc` 子模块**，不要和通用 RPC 框架混。它和 [[Ray与分布式调度|Ray]] 的 `ray.remote` 在概念上同源（都是"远程调用 + 远程对象"），但实现独立：PyTorch RPC 是为模型并行/流水并行设计的、绑 NCCL/gloo；Ray 是通用分布式调度框架。两者可组合（Ray 调起 PyTorch RPC 进程组）。

## 2. 为什么需要它（动机与背景）

[[torch.distributed]] 只提供**集合通信原语**（all-reduce / all-gather / broadcast …）和**点对点 send/recv**。集合通信是"大家一起做同一个 op"，点对点是"我发你收"。但分布式模型并行/流水并行还需要第三种能力：**远程调用**——"让 rank B 上的某个 Module 帮我跑一次 forward，把结果还给我，且我能在 rank A 上引用 rank B 上的张量而不必立刻拷贝"。

具体场景：

1. **Pipeline Parallel（PP）**：模型按层切到多个 rank，rank A 算完 stage 0 要把激活送给 rank B 算 stage 1。早期 PP 实现里这个"递交 + 触发 B 计算"就是一次 RPC 调用（`rpc.rpc_sync(rank_b, run_stage1, args=(act,))`）。现 `torch.distributed.pipelining` 改用点对点 send/recv + schedule，但思路同源。
2. **模型并行参数托管**：超大模型某层权重放 rank B，rank A 想用它就得远程调 rank B 上的 forward。RRef 让 rank A 持有"指向 rank B 上张量的远程引用"，需要时再 fetch，不必每步全量拷贝。
3. **分布式优化器 / 参数服务器**：旧式 parameter server 里，worker 远程从 server 拉/推梯度，也是 RPC 语义。

没有 RPC，这些场景就得手写 send/recv + 协议解析 + 引用计数，工程极重。`torch.distributed.rpc` 把"远程调用 + 远程引用 + 跨进程 autograd"打包成一等公民。

## 3. 核心概念详解

### 3.1 三大支柱

| 支柱 | API | 作用 |
|---|---|---|
| **远程函数调用** | `rpc.rpc_sync` / `rpc.rpc_async` | 调用指定 rank 上的函数，拿结果（同步阻塞 / 异步返回 Future） |
| **远程对象引用 RRef** | `RRef`、`rpc.remote` | 持有"住在别的 rank 上"的对象引用，不立即拷贝；需要时 `rref.to_here()` 取回 |
| **分布式 autograd** | `distributed_autograd` | 跨 rank 的反向传播：在 worker 上 forward、在另一处 backward，梯度自动回流 |

### 3.2 工作流（远程调用一次）

```
rank A:                                  rank B:
  fut = rpc.rpc_async(B, fn, args=(x,))    # B 被 A 调用, 执行 fn(x)
  ... 干别的活 ...                          result = fn(x)   # B 本地算
  result = fut.wait()                      # B 把 result 传回 A
```

- `rpc_sync`：A 阻塞等结果返回；
- `rpc_async`：A 立刻拿 Future，可重叠其他计算，`fut.wait()` 取结果；
- 传参/返回的张量走 NCCL/gloo，自动序列化。

### 3.3 RRef（Remote Reference）

`RRef` 是"远程对象的本地句柄"：

```python
rref = rpc.remote(rank_b, BigTensor, args=(shape,))  # BigTensor 住在 rank_b
# rank A 不持有真正数据，只持引用
val = rref.to_here()          # 需要时把数据拉回 A（一次点对点通信）
rref.rpc_sync().method()      # 远程调 rank_b 上对象的 method，不搬对象
```

- 避免大张量来回拷贝；
- 支持引用计数与所有权管理（owner rank 持真值，其他 rank 持引用）；
- 是模型并行"参数托管在 owner、他人持引用调用"的基础。

### 3.4 分布式 autograd

跨 rank 的前向/反向：A 调 B 的 forward 产激活，最终 loss 在 A，反向要把梯度从 A 流回 B。`torch.distributed.autograd.context` 记录跨 rank 的前向算子，`backward` 时自动沿 RPC 调用图把梯度送回去。这是"用 RPC 拼流水并行还能端到端训练"的关键。

### 3.5 启动与初始化

```python
import torch.distributed.rpc as rpc
rpc.init_rpc(
    name=f"worker{rank}",
    rank=rank,
    world_size=world,
    rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
        rpc_backend_options... # 底层走 ProcessGroup (nccl/gloo)
    )
)
# ... 远程调用 ...
rpc.shutdown()
```

`init_rpc` 内部会调 `dist.init_process_group`，所以 RPC 自带通信层。`rpc.shutdown()` 会做 barrier 并清理。

## 4. 数学原理 / 公式

本条不涉及新公式，但远程调用的成本可形式化：

- **一次 RPC 调用延迟** = 序列化 + 网络往返 + 远端执行 + 反序列化：
$$T_{\text{rpc}} = T_{\text{ser}} + \text{RTT} + T_{\text{exec}} + T_{\text{deser}}$$
- **通信量**：传参张量大小 $\sum |x_i|$ + 返回张量大小 $|y|$。
- vs 集合通信：RPC 是点对点 + 函数语义，集合通信是全组同 op；RPC 灵活但每调用有固定开销，不适合高频小调用。

## 5. 代码示例

```python
# pipelined_rpc.py —— torchrun --nproc_per_node=2 pipelined_rpc.py
# 演示：rank 0 把一段计算委托给 rank 1 执行
import os, torch
import torch.distributed.rpc as rpc

def setup(rank, world):
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world,
        rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(),
    )

# 在任意 rank 上可被远程调用的函数（必须可 pickle）
def stage1(x: torch.Tensor) -> torch.Tensor:
    return x * 2 + 1      # 假装是 stage1 的 forward

def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    setup(rank, world)

    if rank == 0:
        x = torch.randn(4)
        # 同步远程调用 rank 1 上的 stage1
        y = rpc.rpc_sync(1, stage1, args=(x,))
        print(f"[rank 0] x={x.tolist()}  y={y.tolist()}")

        # 异步远程调用 + RRef
        rref = rpc.remote(1, torch.randn, args=(4,))
        print(f"[rank 0] rref owner={rref.owner_name()} val={rref.to_here().tolist()}")

    rpc.shutdown()

if __name__ == "__main__":
    main()
```

> [!tip] 现代替代
> 新代码尽量用 `torch.distributed.pipelining`（PP）、[[Fully Sharded Data Parallel|FSDP]]/`DTensor`（分片参数）、Ray（通用调度）替代手写 RPC。RPC 留给"必须跨 rank 调函数 + 远程引用"的老代码维护。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[torch.distributed]]（RPC 跑在 ProcessGroup 之上）、[[process group]]、[[rank与world size]]、Python pickle（函数/参数序列化）。
- **下游（应用）**: 早期 [[Pipeline Parallel]]、模型并行参数托管、分布式优化器、parameter server 风格训练。
- **对比 / 易混**:
  - **RPC vs 集合通信**：RPC = 点对点 + 函数语义（灵活、有调用开销）；集合通信 = 全组同 op（高效、不灵活）。DDP 用集合通信，PP 早期用 RPC。
  - **`torch.distributed.rpc` vs [[Ray与分布式调度|Ray]] `ray.remote`**：概念同（远程调用+远程对象），实现独立。Ray 通用、生态广；PyTorch RPC 绑 NCCL、为模型并行定制。新项目倾向 Ray。
  - **RRef vs `Future`**：RRef 是远程对象引用（指向远端常住对象）；Future 是一次调用的结果占位。`rpc.remote` 返 RRef，`rpc.rpc_async` 返 Future。
  - **RPC vs `torch.distributed.pipelining`**：后者是 PP 的新官方方案，用点对点 send/recv + schedule 表达流水，不用 RPC 函数语义，性能更好。
  - **RPC vs `DTensor`**：DTensor 把"分片/复制"做成张量属性，参数托管由 DTensor 自动管；RPC 要手动管 RRef。DTensor 是未来方向。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 RPC 是集合通信** → 不是。RPC 是点对点函数调用，集合通信是全组同 op。DDP 不用 RPC，PP 早期才用。
> 2. **远程函数不可 pickle** → RPC 靠 pickle 序列化函数与参数，闭包/lambda/local 函数无法 pickle，会报错；用顶层定义的函数。
> 3. **RPC 每步调一万次** → RPC 有固定序列化/RTT 开销，高频小调用会被延迟淹没；批量调用或改用集合通信。
> 4. **RRef 当本地张量用** → `rref` 不是张量，是引用；`rref + 1` 会报错，要 `rref.to_here()` 取回再算。
> 5. **不调 `rpc.shutdown()`** → 进程残留、端口占用、引用计数不释放；脚本末尾务必 shutdown。
> 6. **以为 RPC 自动 overlap** → `rpc_async` 才 overlap，`rpc_sync` 阻塞；要重叠得手动用 Future。
> 7. **新项目还猛写 RPC** → 现代方案（pipelining/DTensor/FSDP/Ray）更优，RPC 主要是维护老代码。
> 8. **跨 rank autograd 不配 context** → 没开 `distributed_autograd.context` 时跨 RPC 反向会断，梯度回不到远端。
> 9. **RPC 后端选错** → GPU 大张量走 gloo 慢；按场景配 nccl/gloo。
> 10. **和 Ray 的 remote 混为一谈** → 二者独立实现，API 不通；别在 PyTorch RPC 里调 Ray 的 `ray.remote`。

## 8. 延伸细节

### 8.1 RPC 后端

`torch.distributed.rpc` 支持两种后端：
- **`ProcessGroupRpcBackend`**：走 `torch.distributed` 的 ProcessGroup（nccl/gloo），与集合通信共享通信层，最常用。
- **`TensorPipeRpcBackend`**：基于 TensorPipe，更灵活的传输（TCP/UV/shared memory），曾是默认，后 TensorPipe 维护停滞。

### 8.2 与 Pipeline Parallel 的演进

早期 `torch.distributed.pipeline.sync` / `torch.distributed.pipelining` 用 RPC 实现 stage 间激活递交。新版 `pipelining` 改用显式 send/recv + schedule（`PipelineSchedule`），避免 RPC 的函数调用开销，性能更好、调试更清。RPC 退为"理解原理"的角色。

### 8.3 与 parameter server 的关系

旧式异步 PS：worker 远程从 server 拉参数、推梯度，靠 RPC 语义实现。同步训练（DDP/FSDP）兴起后 PS 模式式微，但概念（远程参数托管）仍在 RPC/RRef 里。

### 8.4 为什么被替代

- RPC 的函数语义有固定开销，不适合大模型每步高频通信；
- RRef 手动管理引用繁琐；
- `DTensor`/`FSDP` 把"分片"做成张量/Module 属性，自动管托管，更声明式；
- Ray 在通用分布式调度上生态更全（actor/task/调度/资源）。

### 8.5 何时还会用到 RPC

- 维护老 PP / MP 代码；
- 需要"跨 rank 调任意函数 + 远程引用"且不想引 Ray；
- 教学/理解分布式模型并行原理。

---
相关: [[分布式基础]]、[[torch.distributed]]、[[process group]]、[[rank与world size]]、[[Pipeline Parallel]]、[[Ray与分布式调度]]、[[Fully Sharded Data Parallel]]、[[3D parallelism]]
