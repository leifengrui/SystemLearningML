# Triton kernel开发

> **所属章节**: [[Triton]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: Triton / OpenAI Triton / triton.language / @triton.jit / tl.dot / autotune
> **难度**: 中高（需懂 [[GPU执行模型]] + Python）


## 1. 一句话定义

**Triton** 是 OpenAI 开源的 **Python-based GPU kernel 编程框架**——程序员写 Python 函数（`@triton.jit` 装饰），在函数体内用 `tl.load / tl.store / tl.dot` 等**块级（block-level）原语**操作整个 block 的数据（而非 CUDA 的 thread-level），编译器自动生成 warp 调度、shared memory tiling、bank conflict padding、访存合并、tensor core 指令——**把 CUDA 里最难的底层细节（smem 管理、warp 同步、wmma 调用）交给编译器**，让 Python 也能写出接近手写 CUDA 性能的 kernel。它是 [[torch.compile]]/[[Inductor]] 的默认 codegen 后端，也是 vLLM/SGLang/FlashAttention-2 的主力 kernel 语言，2024-2026 年 LLM 框架自定义算子的事实标准。

> [!note] 三句话定位
> - **是什么**：Python 写 block-level kernel，编译器自动生成 smem tiling/warp 调度/tensor core 指令。
> - **为什么**：手写 CUDA 要管 thread/warp/smem/bank/wmma，门槛极高且易错；Triton 把这些封装成块级原语，生产力 × 性能兼得。
> - **与 CUDA 的区别**：CUDA 是 thread-level（每 thread 写标量），Triton 是 block-level（每 "program" 写一个 block 的张量运算），编译器做 thread→warp 的映射。


## 2. 为什么需要它（动机与背景）

### 2.1 手写 CUDA 的痛点

写一个高效 GEMM kernel 要：

1. grid/block 配置、[[occupancy分析]] 调优。
2. shared memory tiling（手动 `__shared__` 声明 + 载入 + `__syncthreads`）。
3. bank conflict padding（`tile[N][N+1]`）。
4. tensor core 的 `wmma::mma_sync`（warp 协作提交 $16\times16\times16$）。
5. 访存合并、double buffering、software pipelining。

门槛极高，一个 kernel 几百行，易错且难调优。[[CUTLASS与GEMM]] 是 C++ 模板的解法（强大但更复杂），Triton 是 Python 的解法。

### 2.2 Triton 的块级抽象

Triton 程序员写的是一个 **"program"（≈ 一个 block）的运算**：

```python
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    pid = tl.program_id(0)              # 这个 program 的 id
    offs = ...                          # 这个 block 负责的行列
    a = tl.load(a_ptr + offs_a, mask=...)   # 整 block 载入 (编译器生成 smem tiling)
    b = tl.load(b_ptr + offs_b, mask=...)
    c = tl.dot(a, b)                    # 整 block 的矩阵乘 (编译器生成 wmma/tensor core)
    tl.store(c_ptr + offs_c, c, mask=...)   # 写回
```

编译器负责：thread→warp 映射、smem 分配与 tiling、bank padding、访存合并、tensor core 指令选择、`__syncthreads` 插入。程序员只管**块级语义**。

### 2.3 性能接近手写

Triton 编译出的 PTX 在很多算子（GEMM、attention、layernorm）上达到手写 CUDA 的 80-95%，且开发效率高一个数量级。PyTorch 的 `torch.compile` 默认用 Triton 生成融合算子，FlashAttention-2 官方也提供 Triton 实现。2024-2026 LLM 框架（vLLM、SGLang、Megatron-LM）的自定义 kernel 大量用 Triton。

### 2.4 autotune：自动 occupancy 调优

`@triton.autotune` 自动遍历不同 block size / num_warps / num_stages 配置，实测 throughput 选最优——相当于自动化的 [[occupancy分析]] + [[Roofline模型]] 调优，比手调 `launch_bounds` 省心。


## 3. 核心概念详解

### 3.1 program 与 program_id

Triton 的"program"≈ CUDA 的 block。kernel 被多个 program 并行执行，`tl.program_id(axis)` 给当前 program 在该轴的 id。grid 用 `tl.cdiv(N, BLOCK)` 配。

```python
pid = tl.program_id(0)         # 0 轴 program id
pid_m = pid // grid_n
pid_n = pid % grid_n
```

### 3.2 块级张量运算

- `tl.load(ptr + offs, mask=, other=)`：整 block 载入，mask 处理边界（不足一个 block 的尾）。
- `tl.dot(a, b)`：block 级矩阵乘，编译器选 tensor core / FMA 路径。
- `tl.store(ptr + offs, val, mask=)`：整 block 写回。
- `tl.arange`/`tl.zeros`/`tl.full`：块级张量构造。

### 3.3 mask 与边界

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
mask_m = offs_m < M                     # 尾部不足 BLOCK_M 时 mask False
a = tl.load(a_ptr + offs_m[:, None]*K + offs_k[None, :], mask=(mask_m[:, None] & mask_k[None, :]))
```

mask 是 bool 张量，False 处不读（防越界）。这是 Triton 比 CUDA 易用的关键——边界处理用一行 mask 表达，不用写 if 分支（还防 warp divergence）。

### 3.4 tl.dot 与 tensor core

`tl.dot(a, b)` 自动用 tensor core（若 shape/精度合适）：A100 用 `mma.sync` 的 $16\times8\times16$（bf16），编译器把 block 的 dot 分解成多个 wmma。程序员不直接调 wmma——编译器选指令、做寄存器分配、插 `mma.sync`。精度可指定（`tl.dot(a, b, allow_tf32=True)` 用 tf32）。

### 3.5 autotune

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M':128,'BLOCK_N':256,'BLOCK_K':32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M':64, 'BLOCK_N':128,'BLOCK_K':32}, num_warps=4, num_stages=4),
        # ... 多个配置
    ],
    key=['M','N','K'],                    # shape 变化时重选最优
)
@triton.jit
def gemm_kernel(...): ...
```

首次调用遍历所有 config，测 throughput，缓存最优。shape（key）变时重选。相当于自动 [[occupancy分析]] + [[Roofline模型]] 调优，比手调 launch_bounds 省心。

### 3.6 num_warps / num_stages

- `num_warps`：block 用几 warp（4/8 常见），影响 occupancy 与延迟隐藏。
- `num_stages`：软件流水级数，让下一 iter 的 load 与当前 compute overlap（编译器插 double/triple buffer）。

### 3.7 swizzle：bank conflict 与 cache 友好

```python
# tl.swizzle 改 program 到 block 的映射, 让访存合并 + 少 L2 bank conflict
pid = tl.swizzle(pid, GROUP_M, ...)
```

L2 cache 友好的 block 排布（如 grouped ordering）让相邻 program 访问相邻数据，提 L2 命中。Triton 提供原语，比 CUDA 手写简单。

### 3.8 与 torch.compile / Inductor

`torch.compile(model)` 用 [[Inductor]] 把 PyTorch graph 生成 Triton kernel（融合 elementwise/norm/_reduction 为一个 kernel）。这是 torch.compile 提速的底层机制，见 [[torch.compile]]/[[Inductor]]。

### 3.9 与 FlashAttention 的关系

FlashAttention-2 官方有 CUDA 与 Triton 两实现。Triton 版更易读、易改，[[FlashInfer]] 等也大量用 Triton。Triton 的 block-level 模型天然适合 attention 的 tiling 结构。


## 4. 数学原理 / 公式

### 4.1 block 级语义到 thread 级映射

Triton kernel 写的是"一个 block 的张量运算 $C_{BLOCK} = A_{BLOCK} \cdot B_{BLOCK}$"。编译器把它分解为：

1. 把 block 张量切到 warp 级（每 warp 处理一个子块）。
2. 每个 warp 用 `mma.sync` 提交 $16\times8\times16$ 的矩阵乘。
3. 插 shared memory 载入与 `__syncthreads`。
4. 处理 bank conflict padding、访存合并。

程序员写 1 个 `tl.dot`，编译器生成几十条 PTX 指令。这是块级抽象的本质。

### 4.2 tiling 的收益

GEMM $C=AB$，$A\in\mathbb{R}^{M\times K}$。不 tiling：$A$ 每 element 读 $N$ 次（$B$ 沿 $K$ 累加），HBM 读 $MNK$ 次。tiling 进 smem（$B_K$ 列）：$A$ tile 在 smem 复用 $N/B_N$ 次 → HBM 读 $MK \cdot (N/B_N) \cdot B_N = MNK$？实际：每 tile 读一次，全 $K$ 维累加 → $A$ 读 $M K$（每 element 一次），$B$ 读 $NK$。$I \approx 2MNK / (MK+NK)$ → 远高于非 tiling。

FlashAttention 的 tiling 同理（见 [[FlashAttention]]）。

### 4.3 autotune 的搜索空间

config = $\{$BLOCK 配置$\} \times \{$num_warps$\} \times \{$num_stages$\}$，每个 config 实测 throughput。组合爆炸时用启发式剪枝。shape（key）变时重搜，缓存按 shape 分桶。

### 4.4 software pipelining 收益

num_stages=$s$ 让 $s$ 级流水：第 $i$ iter 的 load 与第 $i-1$ 的 compute overlap。稳态 throughput $\approx 1/\max(T_{\text{load}}, T_{\text{compute}})$ 而非 $1/(T_{\text{load}}+T_{\text{compute}})$（见 [[Roofline模型]] overlap 收益）。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟 autotune

```python
def autotune(configs, bench_fn, key):
    """遍历 config, 测 throughput, 选最优."""
    best, best_t = None, float('inf')
    for cfg in configs:
        t = bench_fn(cfg, key)
        if t < best_t: best, best_t = cfg, t
    return best, best_t

configs = [
    {'BLOCK':128, 'num_warps':8, 'stages':3},
    {'BLOCK':256, 'num_warps':8, 'stages':2},
    {'BLOCK':64,  'num_warps':4, 'stages':4},
]
def bench(cfg, key):  # 模拟: 大 batch 喜欢大 block
    return 10 / (cfg['BLOCK'] * cfg['num_warps']) * (1 if key>1024 else 2)
print(autotune(configs, bench, key=4096))
```

### 5.2 真实 Triton GEMM

```python
import triton, triton.language as tl, torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M':128,'BLOCK_N':128,'BLOCK_K':32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M':64, 'BLOCK_N':128,'BLOCK_K':32}, num_warps=4, num_stages=4),
    ],
    key=['M','N','K'],
)
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid %  tl.cdiv(N, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None]*stride_am + offs_k[None, :]*stride_ak
    b_ptrs = b_ptr + offs_k[:, None]*stride_bk + offs_n[None, :]*stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_m[:, None]<M, other=0.)
        b = tl.load(b_ptrs, mask=offs_k[:, None]<K, other=0.)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None]*stride_cm + offs_n[None, :]*stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None]<M) & (offs_n[None,:]<N))

def gemm(a, b):
    M,K = a.shape; K2,N = b.shape
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    gemm_kernel[(tl.cdiv(M,128)*tl.cdiv(N,128),)](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c
```

### 5.3 torch.compile 后端

```python
# torch.compile 默认用 Triton 生成融合 kernel
@torch.compile
def f(x):
    return torch.relu(x * 2 + 1)   # 融合成一个 Triton kernel, 不落中间 HBM
# 见 Inductor / torch.compile 笔记
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GPU执行模型]]（block/warp/tensor core）、[[GPU内存层级]]（smem tiling/bank）、[[Roofline模型]]（autotune 用 throughput 判优劣）、[[occupancy分析]]（autotune 隐含调 occupancy）。
- **下游（应用）**: [[torch.compile]]/[[Inductor]]（默认 codegen 后端）、[[FlashAttention]]/[[FlashInfer]]（主力 kernel 语言）、vLLM/SGLang 的自定义算子、[[fused kernel]]（融合 elementwise/norm 用 Triton 写）。
- **对比 / 易混**:
  - **Triton vs CUDA C**：Triton block-level（Python），CUDA thread-level（C++）。Triton 生产力高，CUDA 极致控制强。
  - **Triton vs [[CUTLASS与GEMM|CUTLASS]]**：Triton Python + 自动 codegen；CUTLASS C++ 模板，更接近硬件、控制更细但门槛更高。
  - **Triton vs torch.compile**：Triton 是 kernel 语言；torch.compile 是编译框架，用 Triton 作后端之一。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 Triton 是"高层 API"，性能差
> 不。Triton 编译出的 PTX 在很多算子上达手写 CUDA 的 80-95%。FlashAttention-2、PyTorch Inductor 都用 Triton 作主力。

> [!warning] 误区 2：忘了 mask 处理边界
> `tl.load` 不加 mask 在尾部不足 block 时越界读（UB 或慢）。务必 `mask=offs < N`。mask 也防 warp divergence（不用 if 分支）。

> [!warning] 误区 3：autotune 的 key 没覆盖变化 shape
> key 只列固定 shape，运行时 shape 变（如动态 batch）会误用旧 config。key 应列所有影响最优配置的维度。

> [!warning] 误区 4：忽视 num_stages 的内存代价
> num_stages 高 → 软件流水级多 → smem 占用大 → occupancy 跌。autotune 会自动权衡，手调时要测。

> [!warning] 误区 5：与 Triton（AMD 的 CPU 编程模型）混淆
> OpenAI Triton（GPU，本笔记）与 AMD 的 Triton（CPU）同名不同物。LLM 场景说的"Triton"指 OpenAI 的。

> [!warning] 误区 6：以为 tl.dot 一定走 tensor core
> shape/精度不合适（如 int32 或 odd shape）时 tl.dot 退化用 FMA，性能差。需 shape 对齐（如 16 倍数）且精度是 fp16/bf16/fp8/tf32 才走 tensor core。


## 8. 延伸细节

### 8.1 Triton 的编译流程

`@triton.jit` → 解析 Python AST 为 MLIR → 优化（tiling、bank padding、warp 分配）→ LLVM IR → PTX → cubin。MLIR 中间表示让优化可移植（CPU/GPU/其他）。

### 8.2 triton.heuristics 与 autotune

`@triton.heuristics` 用启发式（shape 推 BLOCK 配置）而非全搜，比 autotune 快（不跑全部 config）。大 shape 变化少时用 heuristic 省 autotune 开销。

### 8.3 Triton 与 FlashAttention

FlashAttention-2 的 Triton 实现是学习 attention tiling 的最佳范例（见 [[FlashAttention]] 的 tiling 思想在 Triton 里更易读）。FlashInfer 的可变长度 attention 也大量 Triton。

### 8.4 Triton 在 vLLM/SGLang

vLLM 的部分自定义 op（如 RMSNorm、RoPE、特定 attention 变体）用 Triton；SGLang 的 [[Radix Tree prefix cache]] 相关算子也用。Triton 让框架作者能快速写高性能 kernel 而不陷 CUDA 细节。

### 8.5 与 CUTLASS 的关系

CUTLASS（C++ 模板）控制更细，但门槛高。Triton 易用但某些极致场景（如自定义 epilogue 流水）CUTLASS 更灵活。2026 年大框架常两者并用：Triton 写大多数 op，CUTLASS 写极致 GEMM/epilogue。

### 8.6 内容来源

Triton 编程模型整理自 OpenAI Triton 文档与 "Tiled Matrix Multiply" tutorial，autotune 语义见 `triton.autotune` 文档，与 [[torch.compile]]/[[Inductor]] 集成见 PyTorch 2.x 文档。

---
相关: [[Triton]] | [[GPU执行模型]] | [[GPU内存层级]] | [[Roofline模型]] | [[occupancy分析]] | [[CUTLASS与GEMM]] | [[FlashAttention]] | [[FlashInfer]] | [[fused kernel]] | [[torch.compile]] | [[Inductor]]
