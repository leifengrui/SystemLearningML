# mixed precision training

> **所属章节**: [[训练工程基础]]
> **所属模块**: [[01-ML深度学习基础]]
> **难度**: 中级~进阶

## 1. 一句话定义

**混合精度训练（mixed precision training，混合精度训练）** 指在一次训练里**对不同计算环节用不同 dtype**：数值密集但对精度不敏感的算子（matmul/conv/attention 投影）用低位宽（fp16/bf16）跑得又快又省显存，而数值敏感、需大动态范围或高精度累加的环节（master 权重、loss、reduction、softmax 分母、layernorm 统计、梯度累加）仍保留 fp32；现代 LLM 的事实标准是 **bf16 AMP + fp32 master 权重**。

## 2. 为什么需要它（动机与背景）

fp32 训练又慢又占显存：

- **速度**：GPU tensor core 对 fp16/bf16 的吞吐是 fp32 的 2~4 倍（A100 bf16 312TFLOPS vs fp32 19.5TFLOPS）。
- **显存**：权重/激活用 16bit，显存减半 → 能训更大模型或更大 batch。
- **通信**：分布式梯度用 16bit 传输，带宽减半（见 [[all-reduce]]）。

但**不能全程低精度**：

- fp16 动态范围窄（max 65504），梯度稍大就 inf；小梯度下溢成 0。
- bf16 精度低（~0.4% 相对误差），百万次累加会偏。
- 优化器更新需要精度（小步长 + 小梯度），低精度下参数更新会丢信息。

**混合精度的核心思想：该快的地方低精度，该准的地方高精度**。用 autocast 自动按算子类别分配 dtype，用 fp32 master 权重保证优化器更新精度，用 [[loss scaling]]（fp16 时）救小梯度。

> [!note] 解答：哪里是要用高精度的（必须保 fp32）？
> 以下环节**必须用 fp32**，否则训练会发散或掉点：
> 1. **master 权重副本**：参数本体存 fp32，前向/反向用 bf16 的低精度副本参与计算，优化器用 fp32 master 更新。低精度（bf16 ~0.4% 误差）直接更新参数，小步长会被量化吞掉。
> 2. **优化器状态**：Adam 的 $m_t,v_t$（一阶/二阶矩）保 fp32，更新量 $\eta\cdot\hat m/(\sqrt{\hat v}+\epsilon)$ 涉及小数除法，bf16 精度不够会失真。
> 3. **梯度累加 / reduction**：多步累加梯度、跨卡 all-reduce 求和，累加次数多，bf16 累加误差累积漂移 → 保 fp32 累加与归约。
> 4. **loss 计算**：最终 loss 标量用 fp32（softmax 交叉熵的 log-sum-exp 对精度敏感，bf16 易丢小概率）。
> 5. **数值敏感算子**：[[layer norm]] / RMSNorm 的均值方差统计、softmax 的分母累加、exp/log、reduction（mean/sum）——autocast 默认把这些算子**强制保 fp32**，即使全局开 bf16。
> 6. **嵌入层（可选）**：大词表 embedding 有时保 fp32 查表再转，或与 lm_head 同精度。
> 一句话：**累加、归约、小步长更新、统计量**这四类必保 fp32；大块 matmul（线性投影、attention QKV、FFN 升降维）用 bf16。autocast 已按此规则自动分派，但理解原理才能排查精度 bug。

## 3. 核心概念详解

### 3.1 三条路线

| 路线 | 配方 | 何时用 | 要 loss scaling? |
|---|---|---|---|
| **bf16 AMP**（主流） | 前向 bf16 + fp32 master 权重 | A100/H100 起，LLM 事实标准 | **否**（bf16 范围同 fp32） |
| **fp16 AMP** | 前向 fp16 + fp32 master + loss scaling | V100/早期 GPU（无 bf16 硬件） | **是**（防下溢） |
| **fp8 训练** | 前向 e4m3/反向 e5m2 + per-tensor scale | H100，极致省显存，普及中 | 配 scaling factor |

> [!note] 为什么 LLM 选 bf16
> bf16 **指数位同 fp32（8 位）→ 动态范围同 fp32，不溢出、不下溢、不需要 loss scaling**；尾数少（7 位）精度低，但深度学习对 ~0.4% 噪声鲁棒，关键累加回 fp32 即可。且 bf16 是 fp32 截断低 16 位，硬件转换零成本。V100 无 bf16 硬件才退回 fp16+scaling。详见 [[数值类型与精度]]。

### 3.2 AMP（Automatic Mixed Precision）

PyTorch 的 `torch.autocast` 上下文管理器：进入后，**注册的算子自动按白名单选 dtype**：

- **降精度（fp16/bf16）**：matmul、conv、attention 的 QKV 投影、linear —— 计算密集、对精度不敏感。
- **保 fp32**：softmax 分母、layernorm 均值方差、reduction/sum、loss、累加 —— 数值敏感。
- **升级回 fp32**：算子输出若要参与敏感运算，自动升回。

不用手动管 dtype，autocast 按"算子类别"自动分派。

### 3.3 fp32 master 权重（权重副本）

低精度下直接用 bf16 参数做优化器更新有问题：更新量 $\eta\cdot\hat m/(\sqrt{\hat v}+\epsilon)$ 很小，bf16 精度低（~0.4%）→ 小更新量被舍入吞掉，参数不动。

解法：**维护一份 fp32 master 权重**，优化器在 fp32 上精确更新，再拷贝（截断）回 bf16 用于前向：

```
fp32 master W → (前向用 bf16 副本) → 反向得 bf16 梯度 → 转 fp32 → 优化器更新 fp32 master → 截断回 bf16 副本
```

LLM 训练（bf16 参数 + fp32 optimizer state）即此模式。显存多一份 fp32 master（见 8.2 显存账）。

### 3.4 loss scaling（仅 fp16 路线）

fp16 小梯度（< 6e-5）下溢成 0，梯度全丢。**loss scaling**：前向前把 loss 乘大数 $S$（如 65536），梯度同步放大 $S$ 倍抬出下溢区；反向后、optimizer 前把梯度除 $S$ 缩回。详见 [[loss scaling]]。

**bf16 不需要**：范围同 fp32，小梯度不下溢。开了反而无益（甚至 GradScaler 会误判 skip step）。

## 4. 数学原理 / 公式

### tensor core 的收益

A100：fp32 19.5 TFLOPS，bf16 312 TFLOPS（16× 算力密度，因 tensor core 按 16×16 矩阵块算）。matmul 在 bf16 下吞吐约 fp32 的 16 倍（实际受内存带宽限制，约 2~4×）。

### master 权重防舍入

bf16 机器 epsilon $\varepsilon\approx 3.9\times10^{-3}$。若参数 $w=1.0$，更新 $\Delta w=10^{-4}$，在 bf16 下 $w+\Delta w$ 因 $10^{-4}<\varepsilon/2$ 被舍掉 → 参数不动。fp32 $\varepsilon\approx1.2\times10^{-7}$，能保留 $10^{-4}$ 更新。故 master 必须 fp32。

### bf16 累加偏移

每次 bf16 运算 ~0.4% 相对误差，累加 $N$ 次误差 $\propto\sqrt N\varepsilon$。百万次累加偏差显著，故 reduction 必须回 fp32（硬件累加器本就是 fp32 寄存器）。

## 5. 代码示例

```python
import torch
import torch.nn as nn

model = nn.Linear(1000, 1000).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
x = torch.randn(64, 1000).cuda(); y = torch.randn(64, 1000).cuda()
lossf = nn.MSELoss()

# ===== 路线 A：bf16 AMP（LLM 主流，无需 scaler） =====
for step in range(5):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):   # 自动选 dtype
        out = model(x)                                   # matmul 跑 bf16
        loss = lossf(out, y)                             # loss 保 fp32
    loss.backward()                  # 梯度 bf16，autocast 外反向
    opt.step()                      # AdamW 在 fp32 state 上更新

# ===== 路线 B：fp16 AMP（V100，需要 GradScaler） =====
scaler = torch.cuda.amp.GradScaler()      # dynamic loss scaling
for step in range(5):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        out = model(x); loss = lossf(out, y)
    scaler.scale(loss).backward()         # loss 放大后反向
    scaler.unscale_(opt)                  # 缩回梯度（clip 前必须 unscale）
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                      # 检查 inf/nan，无则 step
    scaler.update()                       # 动态调 scale

# ===== 显式 master 权重（框架内部已做，手动示范原理） =====
# fp32_master = {n: p.float().clone() for n,p in model.named_parameters()}
# 前向用 bf16 副本；反向得梯度转 fp32；优化器更新 fp32_master；再 .bfloat16() 拷回
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[数值类型与精度]]（dtype 位级结构是基础）、[[Tensor]]（device/dtype）。
- **下游（应用）**: [[loss scaling]]（fp16 路线必备）、[[Adam与AdamW]]（优化器在 fp32 state 上更新，state 占显存大头）、显存预算（[[activation memory]] 减半）、[[all-reduce]]（梯度 16bit 通信）。
- **对比 / 易混**:
  - **bf16 vs fp16 AMP**：前者无需 scaler、范围大；后者需 scaler、精度高。LLM 用 bf16，V100 用 fp16。
  - **AMP vs 全程低精度**：AMP 关键环节保 fp32；全程低精度会发散。
  - **autocast vs 手动 .half()**：autocast 自动按算子分派；手动 .half() 全量降精度，易错。
  - **混合精度训练 vs 推理量化**：训练保 fp32 master 保精度；推理可激进量化到 int8/fp8（不需反向，精度要求不同）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **bf16 也开 GradScaler**：bf16 不下溢，scaler 无益且会因 overflow 检测逻辑误 skip step。只 fp16 开。
> 2. **autocast 范围乱套**：autocast 应只包前向，反向在 autocast 外（梯度按 autocast 记录的 dtype 自动反传）；把反向也包进去可能错。
> 3. **手混 fp16+fp32 常量**：autocast 内手动混 dtype 会被隐式提升回 fp32，白开 AMP（见 [[数值类型与精度]] 3.5）。
> 4. **clip 前忘 unscale**：fp16 路线 scaler 放大了梯度，clip_grad_norm 前必须 `scaler.unscale_`，否则阈值被放大 $S$ 倍、形同虚设。
> 5. **不维护 master 权重**：直接用 bf16 参数做优化器更新，小步长被舍入吞掉，训练停滞。
> 6. **reduction 用低精度**：softmax/layernorm/loss 的累加必须 fp32，autocast 已处理，但自己写算子要注意。
> 7. **gradcheck 关 tf32/低精度**：数值验证要全程 fp64/fp32，关掉 TF32 和 AMP，否则误差大。

## 8. 延伸细节

### 8.1 TF32 —— 隐形加速

A100 在 fp32 张量上跑 matmul 时，硬件**自动用 TF32**（19 位：1+8+10，范围同 fp32、精度同 fp16）做 tensor core 运算，对外仍是 fp32。`torch.backends.cuda.matmul.allow_tf32` 控制开关。训练常开（白嫖加速），数值敏感验证关掉。

### 8.2 显存账（bf16 AMP 训练）

以 fp32 参数 $P$ 字节为单位（$P$ = 参数量×4）：

| 项 | 字节/参数 | 相对 |
|---|---|---|
| bf16 参数 | 2 | 0.5P |
| bf16 梯度 | 2 | 0.5P |
| fp32 master + Adam m,v | 12 | 3P |
| **合计（不含激活）** | 16 | **4P** |

AdamW state（fp32 master + m + v）是显存大头（3P），故有 [[ZeRO]]/[[FSDP]] 分片。bf16 把参数/梯度压到 0.5P，激活也减半。

### 8.3 fp8 训练

H100 支持 fp8（e4m3 精度高范围小、e5m2 范围大精度低）。训练常**前向 e4m3、反向 e5m2**，配 per-tensor scaling factor 把数值映射进 fp8 窄范围。仍在普及，对 scaler/校准要求更高。

### 8.4 通信精度

梯度 all-reduce 可在 bf16（省带宽）或 fp32（精度高）做。bf16 通信减半带宽但引入量化误差，大模型常用 bf16 通信 + fp32 累加。

### 8.5 与 gradient checkpointing 协同

[[gradient-checkpointing]] 重算前向省激活显存；和 AMP 叠加是 LLM 训练省显存标配。重算时 dtype 仍按 autocast 走。

---
相关: [[训练工程基础]]、[[数值类型与精度]]、[[loss scaling]]、[[Adam与AdamW]]、[[backward过程]]
