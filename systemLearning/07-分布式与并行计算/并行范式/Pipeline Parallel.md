# Pipeline Parallel

> **所属章节**: [[并行范式]]
> **所属模块**: [[07-分布式与并行计算]]
> **难度**: 中高（需懂流水线调度 + 气泡分析）

## 1. 一句话定义

**Pipeline Parallel（PP，流水线并行）** 是把模型**按层切分成若干段（stage）**，每张卡放一段，输入 micro-batch **像流水线一样穿过各 stage** 的并行范式。第 1 卡算前几层，激活传给第 2 卡算后续层，依此类推。多 micro-batch 交错执行填满流水线，减空闲。核心问题是**气泡（bubble）**——流水线填满/排空时各 stage 空闲，效率损失。调度策略 **1F1B**（一前一反）/ **interleaved**（交错）减气泡。与 [[Tensor Parallel]]（切层内）正交：**PP 切层间、TP 切层内**。PP 适合**深网络**（层多），通信是 stage 间点对点（激活），比 TP 的每层 AllReduce 稀疏。它是 [[3D parallelism]] 的"层间"维度。

> [!note] PP 的核心特征
> - **切的是层段**（按层切段），每卡一段层。
> - **流水线执行**：micro-batch 穿越各 stage。
> - **气泡（bubble）**：填满/排空时空闲，效率损失（调度减）。
> - **通信稀疏**：只在 stage 边界传激活（点对点），非每层。
> - **显存 $1/P$**：每卡只存一段层（$P$ = stage 数）。

## 2. 为什么需要它（动机与背景）

大模型深网络（如 80+ 层）单卡显存不足：
1. **层多显存大**：深网络权重 + 激活 + 梯度随层数线性增，单卡装不下全模型；
2. **按层切分**：[[Tensor Parallel]] 切单层（适合单层大），PP 切层段（适合层多），两者正交；
3. **流水线提并行**：朴素按层切分是串行（stage2 等 stage1），流水线让多 micro-batch 交错，多 stage 同时算；
4. **通信稀疏**：PP 只 stage 间传激活，比 TP 每层 AllReduce 通信少，适合跨节点（慢互联）；
5. **与 DP/TP 组合**：PP 是 3D parallelism 的层间维度，与 DP（数据）+ TP（层内）组合训任意大模型。

PP 让层多的深网络能跨卡/跨节点装下，流水线调度提并行度。GPipe/Megatron-LM 是标准实现。

## 3. 核心概念详解

### 3.1 Stage 切分

模型 $L$ 层切 $P$ 段（stage），每段 $L/P$ 层：
- stage 0：层 $[0, L/P)$；
- stage 1：层 $[L/P, 2L/P)$；
- ...
- stage $P-1$：层 $[(P-1)L/P, L)$。

每卡放一个 stage，显存 ~$1/P$。

### 3.2 流水线执行

朴素（GPipe）：
1. micro-batch $m_1$ 进 stage 0 → 算 → 激活传 stage 1 → ... → stage $P-1$；
2. 同时 $m_2$ 进 stage 0（不等 $m_1$ 走完全程）；
3. 多 micro-batch 交错填满流水线。

时间线（$M$ micro-batch，$P$ stage）：
```
stage0: [m1F][m2F][m3F]...[mMF]          (前向填满)
stage1:      [m1F][m2F]...[mMF]
...
stageP-1:              [m1F]...[mMF]
(然后反向类似,气泡大)
```

### 3.3 气泡（bubble）

气泡 = stage 空闲时间：
- **填满气泡**：开头 stage $i$ 等 $m_1$ 走到它（前 $i$ 步空闲）；
- **排空气泡**：结尾 stage $i$ 等 $m_M$ 反向走到它；
- **气泡占比**（GPipe）：$(P-1)/M$（$M$ micro-batch 数），$M\gg P$ 则气泡小。

### 3.4 调度策略

| 调度 | 机制 | 气泡 | 显存 |
|---|---|---|---|
| **GPipe（朴素）** | 先所有前向再所有反向 | 大 $(P-1)/M$ | 高（存所有 $M$ 激活） |
| **1F1B** | 一前向紧跟一反向，交错 | 中 $(P-1)/M$ | 低（只存 $P$ 激活） |
| **interleaved** | 每 stage 多段交错，减气泡 | 小 $(P-1)/(M\cdot V)$ | 中 |
| **zero bubble** | 反向优先调度，零气泡（近似） | 极小 | 中 |

1F1B 是主流（Megatron-LM 默认），interleaved 进一步减气泡（V 段交错）。

### 3.5 1F1B 调度

```
stage0: [m1F][m2F][m3F][m1B][m2B][m3B]...   (一前向一反向交错)
stage1:      [m1F][m2F][m1B][m3F][m2B]...
...
```
- 前 $P$ 步填满（warmup，只前向）；
- 之后 1F1B 交错（前向紧跟反向）；
- 显存：每 stage 只存 $P$ 个激活（vs GPipe 存 $M$）；
- 气泡：仍 $(P-1)/M$（warmup + cooldown）。

### 3.6 通信

- **stage 间**：前向传激活（点对点 send/recv），反向传梯度（点对点）；
- **通信量**：$O(B\cdot d)$ per stage 边界（激活大小）；
- **稀疏**：只 $P-1$ 个边界通信，非每层；
- **跨节点友好**：点对点 + 稀疏，适合慢互联（IB/以太网），vs TP 需 NVLink。

### 3.7 显存

| 项 | 单卡 |
|---|---|
| 权重 | $P_{\text{stage}}=P_{\text{total}}/P$ |
| 激活（1F1B） | $\sim P$ 个 micro-batch 的激活 |
| 梯度 | $P_{\text{stage}}$ |
| 优化器状态 | $O_{\text{stage}}=O_{\text{total}}/P$ |

每卡 ~$1/P$（权重）+ $O(P)$ 激活（1F1B 限）。

## 4. 数学原理 / 公式

### 4.1 气泡占比

GPipe/1F1B：
$$
\text{bubble ratio}=\frac{(P-1)\cdot t}{M\cdot t+（P-1)\cdot t}=\frac{P-1}{M+P-1}\approx\frac{P-1}{M}\quad(M\gg P)
$$

- $t$：单 micro-batch 单 stage 耗时；
- $M$ 大（多 micro-batch）则气泡小；
- interleaved（V 段）：$\frac{P-1}{M\cdot V}$（气泡减 V 倍）。

### 4.2 吞吐

$$
\text{throughput}\approx\frac{M}{M+P-1}\cdot\text{peak}\approx\text{peak}\cdot(1-\frac{P-1}{M})
$$

- 理想（无气泡）：$M/(M+P-1)\to1$；
- 气泡损失：$(P-1)/M$。

### 4.3 通信量

$$
\text{comm per step}=2(P-1)\cdot B\cdot d\quad\text{（前向+反向,各边界）}
$$

- $P-1$ 个 stage 边界；
- 每 boundary 传 $Bd$（激活）；
- 比 TP 的 $2LBd$（每层）稀疏 $L/P$ 倍。

### 4.4 显存

$$
\text{mem}_{\text{per GPU}}\approx\frac{P_{\text{total}}}{P}+P\cdot A_{\text{micro}}+G_{\text{stage}}
$$

- 权重 $1/P$；
- 激活 $O(P)$（1F1B，存 $P$ 个 micro-batch）；
- 1F1B 比 GPipe（存 $M$）省 $M/P$ 倍激活显存。

## 5. 代码示例

```python
import torch, torch.nn as nn

class StageBlock(nn.Module):
    """一个 stage 的若干层"""
    def __init__(self, d, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=4*d, batch_first=True) for _ in range(n_layers)])
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x

# ===== 流水线并行(单进程模拟 P stage) =====
class PipelineModel(nn.Module):
    def __init__(self, d, L, P):
        super().__init__()
        assert L % P == 0
        self.stages = nn.ModuleList([StageBlock(d, L//P) for _ in range(P)])  # P 段
        self.P = P
    def forward(self, x, M_micro=4):
        """GPipe 风格: M micro-batch 流水线"""
        B = x.shape[0]; mB = B // M_micro
        micros = [x[i*mB:(i+1)*mB] for i in range(M_micro)]
        # 各 stage 对各 micro 顺序前向(模拟流水)
        acts = [[None]*self.P for _ in range(M_micro)]
        for m in range(M_micro):
            acts[m][0] = micros[m]
            for s in range(self.P):
                # stage s 算 micro m(实际:流水线交错,这里简化顺序)
                inp = acts[m][s] if s==0 else acts[m][s]
                acts[m][s] = self.stages[s](inp if s==0 else acts[m][s-1])
            # 注:实际 PP 是跨 stage 交错,这里用顺序模拟各 stage
        return torch.cat([acts[m][self.P-1] for m in range(M_micro)], 0)
    def forward_1f1b(self, x, M_micro=4):
        """1F1B 调度(简化): warmup 前向 + 1F1B 交错 + cooldown"""
        # 真实 1F1B 复杂,这里示意调度逻辑
        print(f"1F1B: warmup {self.P} 前向 -> 1F1B 交错 -> cooldown(显存只存 {self.P} 激活)")
        return self.forward(x, M_micro)  # 简化用 forward

d, L, P, B = 64, 8, 4, 16
model = PipelineModel(d, L, P)
x = torch.rand(B, 5, d)  # (batch, seq, dim)
y = model.forward_1f1b(x, M_micro=4)
print(f"PP(P={P} stage): in{x.shape} -> out{y.shape}")
print(f"每卡显存 ~1/{P}: 每段 {L//P} 层, 气泡 ~(P-1)/M = {P-1}/4 = {(P-1)/4:.0%}")
print(f"通信: stage 间点对点(稀疏), 比 TP 每 layer AllReduce 少")

# 气泡可视化
M = 8
print(f"\n气泡分析(M={M} micro-batch, P={P} stage):")
print(f"  GPipe 气泡: (P-1)/M = {(P-1)/M:.0%}")
print(f"  interleaved(V=2): (P-1)/(M*V) = {(P-1)/(M*2):.0%}")
print(f"  M 越大气泡越小,但显存(激活)越大(GPipe),1F1B 限激活显存")
```

> [!tip] 实际工程
> - **Megatron-LM/DeepSpeed**：内置 PP + 1F1B/interleaved 调度；
> - **PyTorch**：`torch.distributed.pipeline`（实验性）；
> - **M 调参**：M 大减气泡但增显存（GPipe），1F1B 平衡；
> - **跨节点**：PP 适合跨节点（通信稀疏），TP 在节点内（需 NVLink）。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[Tensor Parallel]]（对照，层内 vs 层间）、[[Data Parallel]]（对照）、[[all-reduce]]/点对点通信、[[NCCL通信拓扑]]、[[nn.Module]]。
- **下游（应用）**: [[3D parallelism]]（PP 是其层间维度）、Megatron-LM/DeepSpeed/PipeDream 训练框架、大模型深网络训练。
- **对比 / 易混**:
  - **PP vs [[Tensor Parallel]]**：PP 切层间（段间点对点通信，适合层多/跨节点），TP 切层内（每层 AllReduce，适合单层大/节点内）。正交，常组合。
  - **PP vs [[Data Parallel]]**：PP 切模型层（每卡一段），DP 切数据（每卡全模型+不同数据）。正交。
  - **PP vs [[Fully Sharded Data Parallel]]**：PP 是层段切分（计算时各卡算自己段），FSDP 是权重切分（算时 AllGather 全层）。PP 通信稀疏，FSDP 通信是权重。
  - **1F1B vs GPipe vs interleaved**：见 §3.4。调度策略，气泡/显存不同。
  - **气泡 vs 计算空闲**：气泡是流水线填满/排空的 stage 空闲，不是计算错误。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"PP 省计算"**：错。总 FLOPS 不变，有气泡反而效率损失（vs 理想）。省显存。
> 2. **"气泡可消除"**：难。调度减但不可避免（warmup/cooldown）。zero bubble 近似消除但复杂。
> 3. **"PP 适合单层大"**：错。PP 适合层多（深网络），单层大用 TP。
> 4. **"PP 通信频繁"**：错。PP 通信稀疏（stage 间），比 TP 每 layer 少。
> 5. **"M 越大越好"**：部分。M 大减气泡但增显存（GPipe），1F1B 限激活但调度复杂。有 tradeoff。
> 6. **"PP 等于模型并行"**：部分对。PP 是模型并行（层间），TP 也是模型并行（层内）。
> 7. **"1F1B 气泡比 GPipe 小"**：错。气泡同 $(P-1)/M$，但 1F1B 显存省（存 $P$ vs $M$ 激活）。
> 8. **"PP 不需调度"**：错。朴素串行无并行，需流水线调度（1F1B/interleaved）填满。
> 9. **"PP 与 TP 互斥"**：错。正交，3D parallelism 组合 DP×TP×PP。
> 10. **"stage 切分均匀就好"**：基本对，但需考虑各 stage 计算量（若某 stage 层宽则慢，需均衡）。

## 8. 延伸细节

### 8.1 GPipe vs 1F1B vs interleaved

| 调度 | 气泡 | 激活显存 | 复杂度 |
|---|---|---|---|
| **GPipe** | $(P-1)/M$ | $M$（所有 micro） | 简单 |
| **1F1B** | $(P-1)/M$ | $P$（限 warmup 数） | 中 |
| **interleaved** | $(P-1)/(MV)$ | $PV$ | 高 |
| **zero bubble** | $\approx0$ | 中 | 极高 |

- GPipe 简单但显存爆；
- 1F1B 主流（Megatron 默认）；
- interleaved 减气泡但需 stage 内再切 V 段；
- zero bubble（PyTorch/PipeDream）反向优先，近似零气泡。

### 8.2 interleaved 1F1B

- 每 stage 内再分 V 段（虚拟 stage）；
- 调度时各虚拟段交错，减气泡 V 倍；
- 气泡 $(P-1)/(MV)$；
- 代价：通信次数增 V 倍（每虚拟段边界）；
- Megatron-LM 支持（`num_chunks_per_stage`）。

### 8.3 PP 与 TP 的组合（2D）

- 节点内 TP（层内，NVLink），节点间 PP（层间，IB）；
- 充分利用各互联层次；
- 3D parallelism 的基础（再加 DP）；
- 详见 [[3D parallelism]]。

### 8.4 PP 的负载均衡

- 朴素均匀切层，但若各层计算不同（如某层宽），stage 不均衡；
- 需按计算量切（动态规划找均衡切分点）；
- Megatron 支持 `num_layers_per_stage` 手动调；
- 不均衡则慢 stage 拖累全流水。

### 8.5 PP 的 activation 显存

- GPipe 存所有 $M$ micro-batch 的激活（爆）；
- 1F1B 限存 $P$ 个（warmup 数）；
- gradient checkpointing 进一步减（重算前向）；
- 详见 [[activation memory]]/[[gradient checkpointing]]（Chapter 11）。

### 8.6 PP 的历史

- **PipeDream**（2018）：1F1B 调度先驱；
- **GPipe**（2019）：朴素流水线 + gradient checkpointing；
- **Megatron-LM**：1F1B + interleaved，工业标准；
- **PyTorch RPC pipeline**：实验性实现；
- PP 是大模型训练的标准组件，与 TP/DP 组成 3D。

### 8.7 PP 在推理

- 推理一般用 TP（[[inference worker]]），PP 少（推理无反向，流水线优势弱）；
- 但超大模型（如 405B）推理也用 PP（单卡 TP 装不下）；
- 推理 PP 气泡更难填（无反向交错）。

---
相关: [[并行范式]]、[[Tensor Parallel]]、[[Data Parallel]]、[[3D parallelism]]、[[all-reduce]]、[[NCCL通信拓扑]]、[[Fully Sharded Data Parallel]]、[[nn.Module]]、[[activation memory]]、[[gradient checkpointing]]、[[compute vs communication bottleneck]]、[[overlap strategy]]
