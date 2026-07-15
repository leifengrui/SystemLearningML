# CUTLASS与GEMM

> **所属章节**: [[CUTLASS]]
> **所属模块**: [[13-GPU架构与CUDA编程]]
> **别名**: CUTLASS / CUTLASS GEMM / tile · threadblock · warp · instruction 层级 / CuTe
> **难度**: 高（需懂 [[GPU执行模型]] + 模板元编程 + tensor core）


## 1. 一句话定义

**CUTLASS**（CUDA Templates for Linear Algebra Subroutines）是 NVIDIA 开源的 **C++ 模板库**，把 GPU 上的矩阵乘（GEMM）及相关算子按 **instruction / warp / threadblock / tile 四级层次**模板化——程序员通过组合模板参数（tile 尺寸、warp 分块、data layout、epilogue、精度）声明一个 GEMM，库内部生成高效的 shared memory tiling + warp 协作 tensor core 指令 + 访存合并 + epilogue 融合代码。它是 NVIDIA 官方"手写极致 GEMM"的参考实现，cuBLAS 内部很多 kernel 源自 CUTLASS，也是 [[Megatron Core目录与执行链路]]/vLLM 等框架需要极致 GEMM 性能（grouped GEMM、特定 epilogue 融合）时的底层依赖；与 [[Triton kernel开发]] 是 2024-2026 年 LLM 框架写 GEMM 级算子的两条主流路线。

> [!note] 三句话定位
> - **是什么**：NVIDIA C++ 模板库，把 GEMM 拆成 tile/threadblock/warp/instruction 四级，模板参数化生成高效 tensor core 代码。
> - **为什么**：GEMM 是 LLM 训练/推理的计算主体（~70% FLOP），极致性能直接决定 MFU；CUTLASS 给出接近硬件峰值的参考实现。
> - **与 Triton 区别**：CUTLASS C++ 模板控制更细（epilogue/流水/double buffer 可手调）但门槛高；Triton Python block-level 易用，多数 op 用 Triton，极致 GEMM/特殊 epilogue 用 CUTLASS。


## 2. 为什么需要它（动机与背景）

### 2.1 GEMM 是 LLM 的计算主体

LLM 训练/推理的 FLOP ~70% 在 attention/FFN 的 GEMM（$QKV$ 投影、$O$ 投影、FFN 的两层 MLP）。MoE 的 [[grouped GEMM]] 更是。GEMM 性能直接决定 [[MFU与算术强度]]，是 [[Roofline模型]] compute-bound 段的核心算子。写不好 GEMM = 白费算力。

### 2.2 手写 GEMM 的复杂度

一个接近峰值的 GEMM 要管：

1. shared memory tiling（A/B 各 tile 进 smem，复用）。
2. double/triple buffering（load 与 compute overlap）。
3. warp 协作提交 tensor core（`mma.sync` $16\times8\times16$）。
4. epilogue（scale/bias/activation 融合，避免额外 HBM 往返）。
5. 访存合并、bank padding、warp swizzle。
6. shape 适配（M/N/K 不同尺寸选不同 tile 配置）。

手写一个通用高效 GEMM 库是几万行 C++ 的事。CUTLASS 把这些**模板化、参数化**，组合模板即得。

### 2.3 NVIDIA 官方参考

CUTLASS 是 NVIDIA 维护的"我们觉得 GEMM 该这么写"的参考实现。cuBLAS 内部很多 kernel 源自 CUTLASS 模板实例。研究/框架要写非标准 GEMM（grouped GEMM、低精度 fp8、特殊 epilogue）时常直接基于 CUTLASS 改。CuTe（CUTLASS 3.x 的 layout/coord 抽象）让跨硬件（SM80/90/100）的代码更易移植。

### 2.4 与 Triton 的分工

Triton 易用，多数自定义 op（elementwise/norm/attention）用它。但极致 GEMM 性能（cuBLAS 级）、特殊 epilogue 流水、fp8 的精细缩放、grouped GEMM 的 token 级调度，CUTLASS 控制更细。大框架常两者并用。


## 3. 核心概念详解

### 3.1 四级层次

CUTLASS 把 GEMM 自顶向下分四级：

| 层级 | 含义 | 硬件落点 | 典型尺寸 |
|---|---|---|---|
| **tile（block-level）** | 整个 block 负责的输出 tile | 一个 block（CTA） | $128\times128$ 输出 |
| **threadblock** | block 内部分给各 warp 的子 tile | warp 协作 | $64\times64$ 每 warp |
| **warp** | 一个 warp 用 tensor core 算的 tile | `mma.sync` | $16\times8\times16$ 等 |
| **instruction** | 单条 tensor core 指令 | `mma.sync` 一条 | $16\times8\times16$（bf16） |

每级有"tile size"参数，组合即得一个 GEMM 实例。

### 3.2 Shared memory tiling 与 double buffer

A/B 各分 tile 载入 smem，block 内多 warp 共享。**double/triple buffering**：开两份 smem buffer，一个在 load（下一 tile）一个在 compute（本 tile），靠 `cp.async`（异步 copy from global to smem）+ `mma` 流水，隐藏 load 延迟。是接近峰值的关键。

### 3.3 epilogue 融合

GEMM 后常接 scale + bias + activation（如 `C = act(alpha * AB + beta * C + bias)`）。CUTLASS 的 **epilogue** 把这些融合进 GEMM kernel 的写回阶段，中间结果 $AB+\text{bias}$ 不落 HBM，直接寄存器内算完写一次。这是 [[fused kernel]] 在 GEMM 维度的体现，省一次 HBM 往返，提 $I$（[[Roofline模型]]）。

### 3.4 data layout 参数

`RowMajor` / `ColumnMajor` / 任意 stride。CUTLASS 模板参数化 layout，编译期生成对应访存 pattern。CuTe（CUTLASS 3.x）用 `cute::Layout` 抽象任意张量布局，跨 SM 移植更易。

### 3.5 精度与 tensor core 代

```cpp
cutlass::gemm::kernel::Gemm<
  cutlass::gemm::GemmShape<128,128,32>,      // threadblock tile
  cutlass::gemm::GemmShape<64,64,32>,        // warp tile
  cutlass::gemm::GemmShape<16,8,16>,         // instruction (mma.sync) tile
  cutlass::half_t,                            // A 精度
  cutlass::half_t,                            // B 精度
  float,                                      // C 累加精度
  cutlass::layout::RowMajor, ...              // layout
>
```

精度决定走哪代 tensor core：bf16 → SM80 `mma.sync.aligned.m16n8k16`，fp8 → SM89/90 的 fp8 mma，tf32 → SM80 的 tf32 mma。CUTLASS 模板按精度实例化对应指令。

### 3.6 grouped GEMM（MoE）

`GroupedGemm` 一次跑多组不同尺寸的 GEMM（如 MoE 的多专家各算一组）。CUTLASS 提供 `GroupedGemm` kernel，用 problem list 调度，避免多个独立 kernel 的 launch 开销，且共享 tiling 代码。是 [[MoE token dispatcher]]/[[grouped GEMM]] 的底层。

### 3.7 CuTe（CUTLASS 3.x）

CUTLASS 3.x 重构为 CuTe（CUTLASS Tensor）库，引入 `Layout`/`Tensor` 抽象让 layout 与硬件分离，支持 SM90（Hopper）的 TMA（Tensor Memory Accelerator，硬件自动 global→smem 异步拷贝）与 warp specialization。让代码跨 SM80/90/100 移植更易。

### 3.8 与 cuBLAS 的关系

cuBLAS 是闭源库，内部很多 GEMM kernel 源自 CUTLASS 模板实例。框架用 cuBLAS 调标准 GEMM，用 CUTLASS 改非标准（grouped/特殊 epilogue/fp8 细节）。两者性能接近，cuBLAS 有时因 NVIDIA 内部优化略快，CUTLASS 更灵活。

### 3.9 与 Triton 的对比

| | CUTLASS | Triton |
|---|---|---|
| 语言 | C++ 模板 | Python |
| 抽象层级 | instruction/warp/threadblock/tile 全暴露 | block-level |
| 控制粒度 | 极细（epilogue/buffer/warp swizzle） | 粗（编译器决定） |
| 性能 | 接近峰值（手调极致） | 80-95% 峰值 |
| 门槛 | 高（模板元编程） | 低（Python） |
| 适用 | 极致 GEMM/grouped/特殊 epilogue | 多数自定义 op |
| 集成 | 需 C++ 编译/ABI 管理（见 [[PyBind11与CMake]]） | torch.compile 默认后端 |

大框架：Triton 写大多数 op，CUTLASS 写极致 GEMM/grouped/epilogue。


## 4. 数学原理 / 公式

### 4.1 GEMM 的 tiling 收益

$C = AB$, $A\in\mathbb{R}^{M\times K}, B\in\mathbb{R}^{K\times N}$。

- FLOP $= 2MNK$。
- 不 tiling：$A$ 每 element 读 $N$ 次，$B$ 读 $M$ 次 → HBM $\sim MNK$。
- tiling 进 smem（tile $B_M\times B_K$ for A, $B_K\times B_N$ for B）：A tile 在 smem 被 $B_N/B_{\text{warp}}$ 个 warp 复用，整 $K$ 累加后每 element 只 HBM 读 1 次 → $\text{HBM} \approx MK + NK + MN$。
- $I = 2MNK / (MK+NK+MN) \approx 2M$（$M\gg$ 时）→ compute-bound。

这是 CUTLASS 接近峰值的数学根因（与 [[Roofline模型]] 一致）。

### 4.2 double buffer 流水收益

单 buffer：$T = T_{\text{load}} + T_{\text{compute}}$ 串行。double buffer 稳态：$T = \max(T_{\text{load}}, T_{\text{compute}})$。$T_{\text{load}}\approx T_{\text{compute}}$ 时收益 ~2×。是接近峰值的另一关键。

### 4.3 epilogue 融合收益

不融合：GEMM 写 $AB$ 落 HBM（$MN$ bytes），epilogue 读 $AB$ + 写 $C$（$MN$ bytes）→ 多 $2MN$ HBM。融合：$AB$ 在寄存器直接接 scale/bias/activation → 只写 $C$ 一次。省 $2MN$ bytes，提 $I$。

### 4.4 tensor core 的峰值

A100 bf16 mma.sync $16\times8\times16$：每指令 $2\times16\times8\times16 = 4096$ FLOP/cycle × SM × 108 SM × freq → 峰值 312 TFLOPs。CUTLASS 的 warp tile 选 16 倍数对齐 mma 指令。

### 4.5 grouped GEMM 收益

$k$ 个独立 GEMM 各 launch 一次开销 $kL$；grouped 合一 $L$。$k$ 大时省 $kL$（见 [[kernel launch overhead]]）。且共享 tiling 代码、L2 局部性。


## 5. 代码示例（可选）

### 5.1 纯 Python 模拟四级 tiling

```python
def cutlass_gemm_levels(M, N, K,
                        tile=(128,128,32), warp=(64,64,32), inst=(16,8,16)):
    """模拟 CUTLASS 四级: tile -> warp -> instruction."""
    B_M, B_N, B_K = tile
    W_M, W_N, W_K = warp
    I_M, I_N, I_K = inst
    # 每 block 的 tile
    blocks = (M//B_M) * (N//B_N)
    # 每 block 内的 warp 数
    warps_per_block = (B_M//W_M) * (B_N//W_N)
    # 每 warp 的 mma 指令数 (沿 K 累加)
    mma_per_warp = (B_K//I_K) * (W_M//I_M) * (W_N//I_N)
    return {'blocks':blocks, 'warps/block':warps_per_block, 'mma/warp':mma_per_warp}

print(cutlass_gemm_levels(4096, 4096, 4096))
# blocks=1024, warps/block=4, mma/warp=... (体现层次分解)
```

### 5.2 CUTLASS C++ 声明（伪代码见 3.5）

```cpp
// 真实使用: 实例化一个 bf16 GEMM + bias+relu epilogue
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // A
    cutlass::half_t, cutlass::layout::ColumnMajor, // B
    float,            cutlass::layout::RowMajor,    // C
    float,                                              // 累加精度
    cutlass::gemm::GemmShape<128,128,32>,
    cutlass::gemm::GemmShape<64,64,32>,
    cutlass::gemm::GemmShape<16,8,16>,
    cutlass::epilogue::thread::LinearCombinationBiasRelu<...>  // 融合 epilogue
>;
Gemm gemm;
gemm({A,B,C,{M,N,K},alpha,bias});
```

### 5.3 PyTorch 视角

```python
# 用户一般不直接写 CUTLASS; 用 torch.mm (调 cuBLAS) 或 triton
# 框架作者要 grouped/fp8 极致时基于 CUTLASS
# torch 2.x: torch._C._cutlass 接口部分暴露 (AOTAutograd 的 epilogue fusion 用 CUTLASS)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[GPU执行模型]]（warp/tensor core）、[[GPU内存层级]]（smem tiling/bank/double buffer）、[[Roofline模型]]（GEMM 是 compute-bound 典范）。
- **下游（应用）**: [[grouped GEMM]]（MoE 的底层）、[[Megatron Core目录与执行链路]]（Column/RowParallel 用 cuBLAS/CUTLASS）、vLLM/SGLang 的 GEMM（大 batch prefill）、[[FP8量化方案]]/[[AWQ]]/[[GPTQ]]（低精度 GEMM）、[[fused kernel]]（epilogue 融合）。
- **对比 / 易混**:
  - **CUTLASS vs cuBLAS**：CUTLASS 开源模板可改，cuBLAS 闭源更黑盒但有时略快（NVIDIA 内部优化）。
  - **CUTLASS vs [[Triton kernel开发|Triton]]**：见 3.9，C++ 模板极致控制 vs Python 易用。
  - **CUTLASS vs torch.compile**：CUTLASS 是底层库；torch.compile 的 epilogue fusion 部分用 CUTLASS（AOTAutograd）。


## 7. 常见误区与易错点

> [!warning] 误区 1：以为 cuBLAS 总比 CUTLASS 快
> 不一定。CUTLASS 在非标准场景（grouped/fp8/特殊 epilogue）可能更优，因可定制。cuBLAS 优势在标准 GEMM 的批量调优。

> [!warning] 误区 2：忽视 epilogue 融合
> 不融合 epilogue，GEMM 写出 $AB$ 落 HBM 再读回算 bias/activation，多一次往返，降 $I$。CUTLASS 的 epilogue 融合是性能关键，不是锦上添花。

> [!warning] 误区 3：tile 配置照搬
> 不同 GPU/shape 的最优 tile 不同。盲目照搬某论文的配置可能劣化。需按 [[Roofline模型]] + [[occupancy分析]] 实测，或用 CuTe 的自动搜索。

> [!warning] 误区 4：混淆 CUTLASS 与 CuDNN
> CUTLASS 是 GEMM/linear algebra 模板库（开源）；cuDNN 是 DNN 高层 op 库（卷积/attention 等，闭源）。两者层次不同。

> [!warning] 误区 5：忽视 ABI / 编译
> CUTLASS 是 C++ 模板，集成到 PyTorch 要处理 ABI/编译（见 [[PyBind11与CMake]]）。Triton 是 Python 无此问题，是 Triton 更流行的原因之一。

> [!warning] 误区 6：以为 grouped GEMM = 多次普通 GEMM
> grouped 不是简单循环调普通 GEMM，而是单一 kernel 用 problem list 调度，省 launch 开销 + 共享 tiling + L2 局部性。MoE 的 [[grouped GEMM]] 性能依赖此。


## 8. 延伸细节

### 8.1 Hopper TMA 与 warp specialization

SM90（H100）引入 TMA（Tensor Memory Accelerator）——硬件自动 global→smem 异步拷贝（不经寄存器中转），warp specialization 让一组 warp 专做 load、另一组专做 compute。CUTLASS 3.x（CuTe）支持，比 SM80 的 `cp.async` + 软件流水更高效。

### 8.2 fp8 GEMM

SM89/90 支持 fp8 mma（E4M3/E5M2）。CUTLASS 提供 fp8 GEMM 模板，含 per-tensor/per-block scaling factor（fp8 动态范围窄，需 scale）。是 [[FP8量化方案]] 训练的底层。细节多易错，框架常直接用 CUTLASS 实例。

### 8.3 CUTLASS 与 FlashAttention

FlashAttention-2 有 CUTLASS 实现路径（attention 本质是带 softmax 的 grouped GEMM）。但 Triton 实现更主流（易读易改）。CUTLASS 实现追求极致。

### 8.4 内存层级与 CUTLASS

CUTLASS 的四级层次与 [[GPU内存层级]] 对应：instruction 在 register，warp 在 register+smem，threadblock 在 smem，tile 跨 global（HBM）。double buffer 用 smem 的两份。

### 8.5 内容来源

CUTLASS 四级层次与 CuTe 整理自 NVIDIA CUTLASS 文档与 examples，GEMM 性能模型见 [[Roofline模型]] 与 cuBLAS 技术说明，grouped GEMM 见 Megatron-LM 的 MoE 实现。

---
相关: [[CUTLASS]] | [[GPU执行模型]] | [[GPU内存层级]] | [[Roofline模型]] | [[occupancy分析]] | [[Triton kernel开发]] | [[fused kernel]] | [[grouped GEMM]] | [[Megatron Core目录与执行链路]] | [[FP8量化方案]] | [[AWQ]] | [[GPTQ]] | [[PyBind11与CMake]]
