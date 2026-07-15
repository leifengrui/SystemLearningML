# projector

> **所属章节**: [[VLM架构]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: projector / 投影层 / connector / 连接器 / vision-language adapter / visual projection / 视觉对齐层 / linear projector / MLP projector / resampler / Q-Former / Perceiver
> **难度**: 中（需懂 [[ViT与vision encoder]]、[[early与late fusion]]、[[自注意力]]、[[FFN]]、[[decoder-only architecture]]）


## 1. 一句话定义

**projector**（投影层 / connector）是 VLM 中 **vision encoder 与 LLM 之间的桥接模块**——vision encoder 出的视觉特征（维度 $D_v$、语义在视觉空间）与 LLM 期望的 token embedding（维度 $d$、语义在文本空间）**维度不同、分布不同**，projector 把视觉特征投影 + 对齐到 LLM 能直接吃的语义空间；主流三种：① **linear**（单层线性投影，LLaVA-1.5 早期 / LLaVA-NeXT 简版）② **MLP**（两层线性 + GELU，LLaVA-1.5 / LLaVA-NeXt 主力）③ **resampler**（用可学习 query 把**变长**视觉特征**压缩**成固定数量 token，Q-Former / Perceiver，Qwen-VL / Flamingo / BLIP-2）；projector 是 [[early与late fusion]] 范式中 early fusion 的对齐桥，决定视觉信息进 LLM 的"翻译质量"与"token 预算"。

> [!note] 三句话定位
> - **是什么**：vision encoder 输出 $N$ 个 $D_v$ 维视觉 token → projector → $N'$ 个 $d$ 维 LLM-对齐 token（$N' \le N$，linear/MLP 不压 $N$，resampler 压）。是视觉特征到文本语义的"翻译层"。
> - **为什么**：视觉特征维度 / 分布与文本 token 不同，直接拼 LLM 学不动；高分辨率视觉 token 数百上千，resampler 压到几十省 LLM 算力。
> - **与 [[early与late fusion]] 关系**：projector 是 early fusion 的对齐桥——把视觉特征对齐后与文本 token 拼接进 LLM。late fusion 用 cross-attention 不需 projector。


## 2. 为什么需要它（动机与背景）

### 2.1 维度不匹配

vision encoder（如 CLIP-ViT-L/14）输出 $D_v = 1024$ 维；LLM（如 LLaMA-7B）token embedding 是 $d = 4096$ 维。直接拼维度对不上，需投影层 $\mathbf{W} \in \mathbb{R}^{D_v \times d}$ 升维。

### 2.2 语义分布不匹配

更关键的是**语义空间不同**：CLIP 视觉特征是"与文本对齐的视觉表示"（对比学习对齐到文本 encoder 空间，不是 LLM 空间）；LLM token embedding 是"语言模型自回归分布"。两者分布差异大，LLM 直接吃原始视觉特征会**学不动**（梯度信号弱、token 分布偏离 LLM 训练分布）。projector（尤其 MLP）通过非线性变换把视觉特征"翻译"到 LLM 友好的分布。

### 2.3 token 数压缩

高分辨率图 → 视觉 token 数百上千（224×224 → 196；448×448 → 1024；4K 图 → 数千）。LLM 是 [[decoder-only architecture]]，attention $O(L^2)$，视觉 token 灌进去 → 序列长爆 → 计算量 / KV cache 显存爆。**resampler** 用少量可学习 query 把变长视觉特征**压缩**成固定 $N'$ 个 token（如 256），省 LLM 算力。是 [[动态分辨率与视觉token]] 的 token 预算控制关键。

### 2.4 训练稳定性

projector 是 VLM 训练的"缓冲"：冻结 vision encoder + LLM，只训 projector（LLaVA stage 1），让 projector 学会"翻译"而不破坏预训练两边。这是 LLaVA 之所以简单高效能训起来的关键——projector 参数小、易收敛、不毁两边预训练知识。


## 3. 核心概念详解

### 3.1 linear projector（单层线性）

最简形式：一个线性层把 $D_v$ → $d$。

$$\mathbf{H}_v^{LLM} = \mathbf{H}_v \mathbf{W}, \quad \mathbf{W} \in \mathbb{R}^{D_v \times d}$$

- token 数不变：$N' = N$。
- 参数量：$D_v \cdot d$（如 $1024 \times 4096 \approx 4M$，极小）。
- 出处：LLaVA-1.5 早期 / LLaVA-NeXT 简版（后改 MLP）。
- 优劣：极简、省参、易训；但表达能力弱，高分辨率 token 不压缩。

### 3.2 MLP projector（两层线性 + GELU）

两层线性中间夹 GELU 激活：

$$\mathbf{H}_v^{LLM} = \text{GELU}(\mathbf{H}_v \mathbf{W}_1) \mathbf{W}_2, \quad \mathbf{W}_1 \in \mathbb{R}^{D_v \times d}, \mathbf{W}_2 \in \mathbb{R}^{d \times d}$$

- token 数不变：$N' = N$。
- 参数量：$D_v \cdot d + d^2$（如 $1024 \times 4096 + 4096^2 \approx 21M$，仍小）。
- 出处：**LLaVA-1.5 / LLaVA-NeXt 主力**（2 层 MLP，GELU）。
- 优劣：非线性增强对齐能力，仍是 dense 不压缩；现代 LLaVA 标配。

> [!tip] 实践
> LLaVA-1.5 用 2 层 MLP（Pandres等 2023 实验显示 MLP > linear 显著），$D_v=1024 \to d=4096 \to d=4096$。Stage 1 只训 projector（冻结两边）；Stage 2 解冻 LLM 微调。

### 3.3 resampler（Q-Former / Perceiver，可学习 query 压缩）

用 $N'$ 个**可学习 query** $\mathbf{Q} \in \mathbb{R}^{N' \times d}$ 通过 [[自注意力]] / cross-attention 从变长视觉特征 $\mathbf{H}_v \in \mathbb{R}^{N \times D_v}$ 中"抽取"信息，输出 $N'$ 个固定 token。

#### Q-Former（BLIP-2）

Q-Former 是一个小 Transformer，含 **可学习 query**（如 32 / 64 / 256 个）+ cross-attention 层。query 与视觉特征 cross-attn：

$$\mathbf{Q}' = \text{CrossAttn}(\mathbf{Q}, \mathbf{H}_v, \mathbf{H}_v)$$

query 作 Q，视觉特征作 K/V。$N'$ 固定（如 32），$N$ 变长 → **压缩** $N \to N'$。

#### Perceiver Resampler（Flamingo）

类似 Q-Former，用可学习 latent query cross-attend 视觉特征。Flamingo 用 64 query。

#### Qwen-VL / Qwen2-VL 的 resampler

Qwen-VL 用 256 query 的 Perceiver-style resampler，把变长视觉 token 压到 256 固定 token 喂 LLM。

> [!note] resampler 的本质
> resampler 是 **可学习的"信息瓶颈"**——$N'$ 个 query 从 $N$ 个视觉 token 提取信息，$N' \ll N$ 时是**有损压缩**，但 $N'$ 个 token 可学成"视觉摘要"。trade-off：压得越狠省越多 LLM 算力，但丢细节。文档 OCR 等需细粒度的任务不宜压太狠。

### 3.4 三种 projector 对比

| projector | token 数 $N'$ | 参数量 | 压缩能力 | 出处 | 适用 |
|---|---|---|---|---|---|
| **linear** | $N' = N$（不压） | $D_v d$（极小，~4M） | 无 | LLaVA-1.5 早期 | 低分辨率、token 少、求简 |
| **MLP**（2 层 + GELU） | $N' = N$（不压） | $D_v d + d^2$（小，~21M） | 无（非线性对齐强） | LLaVA-1.5/NeXt 主力 | 中分辨率、token 可控 |
| **resampler**（Q-Former/Perceiver） | $N' \ll N$（压到固定，如 256） | ~ $N' d$ + cross-attn 层（中，~50M） | 强（变长 → 固定） | BLIP-2 / Flamingo / Qwen-VL | 高分辨率、token 多、需省 LLM |

> [!warning] resampler 不一定优
> resampler 压 token 省 LLM 算力，但**有损**——OCR / 细粒度任务压狠丢字。LLaVA 系（MLP 不压）反而在很多 benchmark 优于 Q-Former 系。resampler 适合"视觉 token 数会爆"的场景（高分辨率、长视频），MLP 适合"token 可控、求保真"。

### 3.5 projector 在 VLM 流水线中的位置

```
图 -> vision encoder (D_v, N token) -> projector -> (d, N' token) -> 与文本 token 拼接 -> LLM -> 输出文本
```

projector 是"翻译层"——把视觉特征翻译成 LLM 懂的 token。冻结 vision encoder + LLM，只训 projector（LLaVA stage 1）能让 VLM 快速学会看图，再解冻 LLM 微调（stage 2）。


## 4. 数学原理 / 公式

### 4.1 linear projector

$$\mathbf{H}_v^{LLM} = \mathbf{H}_v \mathbf{W} + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{D_v \times d}, \mathbf{H}_v \in \mathbb{R}^{N \times D_v}$$

输出 $\mathbf{H}_v^{LLM} \in \mathbb{R}^{N \times d}$。token 数不变 $N' = N$。

### 4.2 MLP projector

$$\mathbf{H}_v^{LLM} = \text{GELU}(\mathbf{H}_v \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

$\mathbf{W}_1 \in \mathbb{R}^{D_v \times d}, \mathbf{W}_2 \in \mathbb{R}^{d \times d}$。token 数不变 $N' = N$。

### 4.3 resampler 压缩比

设 vision encoder 输出 $N$ 个 token，resampler query 数 $N'$。**压缩比**：

$$r = \frac{N}{N'}$$

例：$N = 576$（CLIP-ViT-L@336），$N' = 256$（Qwen-VL resampler）→ $r \approx 2.25 \times$。$N = 4096$（高分辨率），$N' = 256$ → $r = 16 \times$。

> [!tip] 压缩比 trade-off
> $r$ 大省 LLM 算力但丢信息；$r$ 小保真但 LLM 算力大。文档 OCR / 细粒度任务 $r$ 小（或用 MLP 不压）；视频 / 高分辨率大场景 $r$ 大。

### 4.4 LLM 算力节省

LLM attention $O(L^2)$。设文本 token $L_t$，视觉 token $N'$。总序列长 $L = L_t + N'$。视觉 token 贡献的 LLM 算力 $\propto N'^2 + 2 N' L_t$（self + cross 项）。

linear/MLP（$N' = N$）：算力 $\propto N^2$。$N=576$ → $\approx 331K$。
resampler（$N'=256$）：算力 $\propto 256^2 = 65K$。**省 ~5×**。

高分辨率 $N=4096$：linear $\propto 16.7M$；resampler $N'=256$ → $65K$，**省 ~256×**。差距随 $N$ 拉大。

### 4.5 Q-Former 的 cross-attention

Q-Former 的第 $l$ 层 cross-attn：

$$\mathbf{Q}^{(l+1)} = \text{softmax}\left(\frac{(\mathbf{W}_Q \mathbf{Q}^{(l)})(\mathbf{W}_K \mathbf{H}_v)^T}{\sqrt{d_k}}\right) \mathbf{W}_V \mathbf{H}_v$$

$\mathbf{Q} \in \mathbb{R}^{N' \times d}$（query），$\mathbf{H}_v \in \mathbb{R}^{N \times D_v}$（视觉特征）。输出 $N'$ 个 token。$N'$ 固定，$N$ 变长 → 压缩。

### 4.6 projector 参数量

| projector | 参数量（$D_v=1024, d=4096$） |
|---|---|
| linear | $1024 \times 4096 + 4096 \approx 4.2M$ |
| MLP（2 层） | $1024 \times 4096 + 4096 \times 4096 \approx 21M$ |
| Q-Former（256 query，4 层 cross-attn） | $\sim 50M$（含 query embedding + cross-attn 层） |

projector 参数量远小于 vision encoder（300M+）和 LLM（7B+），是"小而关键"的桥。


## 5. 代码示例（可选）

### 5.1 linear projector

```python
import torch
import torch.nn as nn

class LinearProjector(nn.Module):
    """单层线性投影: D_v -> d。token 数不变。LLaVA-1.5 早期用。"""
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim)

    def forward(self, vision_features):
        # vision_features: (B, N, D_v)
        return self.proj(vision_features)  # (B, N, d)

p = LinearProjector()
out = p(torch.randn(2, 576, 1024))
print(out.shape)  # torch.Size([2, 576, 4096])  -- N 不变, 维度对齐到 LLM
```

### 5.2 MLP projector（LLaVA-1.5 风格）

```python
class MLPProjector(nn.Module):
    """两层 MLP + GELU。LLaVA-1.5 / LLaVA-NeXt 主力。token 数不变。"""
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),   # D_v -> d
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),       # d -> d
        )

    def forward(self, vision_features):
        return self.mlp(vision_features)      # (B, N, d), N 不变

p = MLPProjector()
out = p(torch.randn(2, 576, 1024))
print(out.shape)  # torch.Size([2, 576, 4096])
```

### 5.3 Q-Former resampler（概念示意）

```python
class Resampler(nn.Module):
    """Q-Former 风格 resampler: 用 N' 个可学习 query 把变长 N 视觉特征压到 N' token。"""
    def __init__(self, vision_dim=1024, llm_dim=4096, num_queries=256, num_heads=8, num_layers=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim))  # 可学习 latent query
        # 把视觉特征先投影到 llm_dim
        self.vision_proj = nn.Linear(vision_dim, llm_dim)
        # 多层 cross-attention (query attend 视觉特征)
        cross_layer = nn.TransformerDecoderLayer(
            d_model=llm_dim, nhead=num_heads, dim_feedforward=llm_dim * 4,
            batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(cross_layer, num_layers=num_layers)

    def forward(self, vision_features):
        # vision_features: (B, N, D_v)  N 可变长
        B = vision_features.size(0)
        q = self.queries.expand(B, -1, -1)                  # (B, N', d)
        kv = self.vision_proj(vision_features)              # (B, N, d)
        out = self.decoder(q, kv)                           # query attend kv -> (B, N', d)
        return out                                          # (B, N', d), N' 固定

r = Resampler(num_queries=256)
out = r(torch.randn(2, 576, 1024))   # N=576 变长
print(out.shape)                     # torch.Size([2, 256, 4096])  -- 压到 256 固定

out2 = r(torch.randn(2, 4096, 1024))  # N=4096 也行 (变长)
print(out2.shape)                     # torch.Size([2, 256, 4096])  -- 仍 256
```

### 5.4 完整 VLM 前端（vision encoder + projector + 拼接 LLM）

```python
class VLMVisionFront(nn.Module):
    """VLM 视觉前端: vision encoder + projector, 输出 LLM-ready 视觉 token。"""
    def __init__(self, vit, projector):
        super().__init__()
        self.vit = vit          # CLIP-ViT / DINOv2 (常冻结)
        self.projector = projector  # MLP / resampler

    def forward(self, images):
        # images: (B, 3, 336, 336)
        with torch.no_grad():   # vision encoder 冻结
            feats = self.vit(images)             # (B, N, D_v)
        visual_tokens = self.projector(feats)    # (B, N', d) -- LLM-ready
        return visual_tokens

# 后续: visual_tokens 与文本 token embedding 拼接 -> LLM forward
# visual_tokens (B, N', d) cat text_emb (B, L_t, d) -> (B, N'+L_t, d) -> LLM
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[ViT与vision encoder]]（视觉特征源）、[[自注意力]]（resampler cross-attn 机制）、[[多头注意力]]、[[FFN]]（MLP/Resampler 内部）、[[decoder-only architecture]]（下游 LLM 形态）、[[tokenization]]（视觉 token 对齐文本 token）、[[位置编码]]（resampler query 需位置）。
- **下游（应用）**: [[early与late fusion]]（projector 是 early fusion 桥）、[[动态分辨率与视觉token]]（resampler 压 token 数）、[[多模态sequence packing]]（变长视觉 token 经 resampler 后定长，简化 packing）、LLaVA / Qwen-VL / InternVL / BLIP-2 / Flamingo / CogVLM。
- **对比 / 易混**:
  - **projector vs cross-attention fusion**：projector 是早期融合的对齐桥（视觉特征对齐后与文本拼进 LLM）；cross-attention fusion 是在 LLM 每层用 cross-attn 让 LLM query 视觉特征（不需 projector）。前者是 [[early与late fusion]] 的 early，后者偏 late。详见 [[early与late fusion]]。
  - **linear vs MLP vs resampler**：见 §3.4 对比表。linear 最简不压，MLP 非线性强不压，resampler 压变长到固定。
  - **projector vs [[多模态MoE路由]]**：projector 是模态对齐桥，MoE 路由是专家稀疏激活提容量。正交（projector 后接 MoE LLM 也可）。
  - **projector vs [[新模型接入]]**：新接入 vision encoder 到推理引擎，projector 层实现要适配（resampler 的 cross-attn 在推理框架需支持变长）。


## 7. 常见误区与易错点

> [!warning] 误区 1：projector 只是改维度
> 不只是维度。更关键是**语义分布对齐**——CLIP 视觉特征与 LLM token embedding 分布差异大，projector（尤其 MLP）的非线性把视觉特征"翻译"到 LLM 友好分布。LLaVA 论文实验：MLP > linear 显著，证明非线性对齐重要。

> [!warning] 误区 2：resampler 一定优于 MLP
> 不一定。resampler 压 token 省 LLM 算力但**有损**。LLaVA-NeXt（MLP 不压）在多 benchmark 优于 Qwen-VL（resampler）。resampler 适合"视觉 token 会爆"场景（高分辨率、长视频），MLP 适合"token 可控、求保真"。

> [!warning] 误区 3：projector 必须压 token
> linear / MLP 不压（$N' = N$）。只有 resampler 压。LLaVA 系主力是 MLP 不压。

> [!warning] 误区 4：projector 是 VLM 最大的模块
> 不是。projector 参数量极小（4M~50M），远小于 vision encoder（300M+）和 LLM（7B+）。但它是"小而关键"的桥——训练时是 stage 1 唯一可学模块，决定对齐质量。

> [!warning] 误区 5：cross-attention fusion 也叫 projector
> 概念易混。projector 特指"视觉特征 → LLM-ready token"的桥接（linear/MLP/resampler）。cross-attention fusion 是在 LLM 内每层 query 视觉特征，是另一种融合范式，不叫 projector。见 [[early与late fusion]]。

> [!warning] 误区 6：冻结 vision encoder 就不需 projector
> 仍需。vision encoder 输出维度 / 分布与 LLM 不同，即使冻结 encoder 也需 projector 做对齐。projector 是必经的桥。

> [!warning] 误区 7：resampler query 数越多越好
> query 多保细节但 LLM 算力大。$N'=256$ 是 Qwen-VL 经验值。$N'$ 与任务粒度 trade-off：OCR 需细粒度 → $N'$ 大或用 MLP 不压；视频 / 大场景 → $N'$ 小。


## 8. 延伸细节

### 8.1 LLaVA-NeXt 的 MLP projector 高分辨率策略

LLaVA-NeXt（OneVision）用 MLP projector 不压 token，但高分辨率时切 tile → 每 tile 576 token × 4 tile = 2304 token 灌 LLM，靠 LLM 大上下文扛。这是"不压但切"的策略，保细节。

### 8.2 Qwen2-VL 的 resampler + 任意分辨率

Qwen2-VL 用 NaViT 风格 vision encoder + resampler，支持任意分辨率。resampler 把变长视觉 token 压到固定 256（或按分辨率动态调整）。是 [[动态分辨率与视觉token]] + resampler 的组合。

### 8.3 BLIP-2 的两阶段 Q-Former 训练

BLIP-2 的 Q-Former 分两阶段训：stage 1 视觉-语言对齐（query 学从视觉特征提文本相关信息）；stage 2 视觉到 LLM 生成对齐（query 输出对齐到 LLM 生成）。Q-Former 设计复杂，但 resampler 思想被 Qwen-VL 等简化继承。

### 8.4 CogVLM 的"视觉专家"MoE 路线

CogVLM 不用 projector，而是在 LLM 每 FFN 加"视觉专家"分支（与文本专家并联），让 LLM 内部模态分支。是 [[多模态MoE路由]] 思路，绕开 projector 直接在 LLM 内对齐。参数大但保细节。

### 8.5 projector 的训练策略

主流 VLM 两阶段：
1. **Stage 1（projector 预训练）**：冻结 vision encoder + LLM，只训 projector（LLaVA）或 Q-Former（BLIP-2）。让桥学会"翻译"。
2. **Stage 2（指令微调）**：解冻 LLM（+ 可选 vision encoder），全参 / LoRA 微调，对齐指令。

stage 1 projector 是唯一可学模块，收敛快、不毁两边预训练知识。是 LLaVA 之所以简单高效的关键设计。

### 8.6 内容来源

整理自 LLaVA / LLaVA-1.5 / LLaVA-NeXt 论文与代码（`llava/model/` 的 `mlp_proj` / `mq_proj`）、BLIP-2 Q-Former 论文、Flamingo Perceiver Resampler 论文、Qwen-VL / Qwen2-VL 技术报告、CogVLM 论文。具体超参（query 数、层数）以原论文为准。

---
相关: [[VLM架构]]、[[ViT与vision encoder]]、[[early与late fusion]]、[[动态分辨率与视觉token]]、[[多模态sequence packing]]、[[多模态MoE路由]]
