# early 与 late fusion

> **所属章节**: [[VLM架构]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: early fusion / late fusion / cross-attention fusion / 早期融合 / 晚期融合 / 交叉注意力融合 / 模态融合范式 / 视觉语言融合
> **难度**: 中（需懂 [[ViT与vision encoder]]、[[projector]]、[[自注意力]]、[[多头注意力]]、[[decoder-only architecture]]、[[residual]]）


## 1. 一句话定义

**模态融合（fusion）** 是 VLM 把视觉与文本信息合到一起的三种范式——① **early fusion（早期融合）**：视觉 token 与文本 token 在**浅层就拼成一个序列**进同一个 LLM backbone（LLaVA / Qwen-VL，视觉与文本平权参与 self-attention）② **late fusion（晚期融合）**：各模态**独立 encoder**，只在 **LLM 的高层**用 cross-attention 让文本 query 视觉特征（Flamingo / BLIP-2，视觉作"外部记忆"被按需查询）③ **cross-attention fusion**：在 LLM **每层**插 cross-attention 层，让 LLM 层 query 视觉特征（介于两者，CogVLM 的视觉专家也偏此）；三者权衡参数量 / 对齐难度 / 灵活性——**主流 VLM（LLaVA / Qwen-VL / InternVL）倾向 early fusion**（实现简单、复用 LLM、效果好），projector 是 early fusion 的对齐桥。

> [!note] 三句话定位
> - **是什么**：视觉与文本合到一起的三种方式——early（浅层拼成序列进 LLM）、late（高层 cross-attn 查询）、cross-attn（每层 cross-attn 交互）。
> - **为什么**：融合点决定模型架构与对齐方式。early 简单复用 LLM；late 保模态独立灵活但 cross-attn 层多参；cross-attn 折中。
> - **与 [[projector]] 关系**：projector 是 early fusion 的对齐桥（视觉特征 → LLM-ready token 后拼接）；late/cross-attn fusion 不需 projector（用 cross-attn 直接查视觉特征）。


## 2. 为什么需要它（动机与背景）

### 2.1 融合点决定架构

视觉语言模型要把"图"与"文"合到一起，最根本的问题是**在哪一层合**：
- 浅层合（early）：视觉 token 进 LLM 输入端，与文本平权参与所有层 self-attention。
- 深层合（late）：视觉特征作"外部"被 LLM 高层按需 cross-attn 查询。
- 中间层合（cross-attn）：LLM 每层都插 cross-attn 与视觉交互。

融合点决定：参数量、对齐难度、模态是否平权、能否复用 LLM 预训练、训练稳定性、推理效率。

### 2.2 early fusion 复用 LLM 预训练

LLM（如 LLaMA）在海量文本上预训练，self-attention 强大。early fusion 把视觉 token 拼进输入端，LLM 的 self-attention **天然处理混合序列**（视觉 token 与文本 token 同等参与 attention）。优点：**复用 LLM 全部预训练知识与算子**，实现极简（拼接 + projector 对齐）。LLaVA 证明这条简单路线效果惊人。

### 2.3 late fusion 保模态独立

视觉与文本语义差异大。early fusion 让两者平权竞争 LLM 容量，可能视觉信息淹没文本预训练知识。late fusion 让**视觉特征作外部记忆**——LLM 高层（或专门 decoder）用 cross-attention **按需查询**视觉特征，文本 LLM 主体不被视觉干扰，保文本生成能力。Flamingo / BLIP-2 走这条，但需训专门的 cross-attn 层。

### 2.4 cross-attention fusion 折中

纯 early（拼接）或纯 late（高层 cross-attn）都各有利。cross-attention fusion 在 LLM **每层**插 cross-attn（类似图像 captioning 的 Transformer decoder 结构），让 LLM 每层都能细粒度查视觉。参数多但保细节。CogVLM 的"视觉专家"MoE 思路偏此。

### 2.5 主流倾向 early

实测：**early fusion（LLaVA / Qwen-VL / InternVL）** 在多数 benchmark 优于 late / cross-attn，且实现简单、训练稳定、推理高效（无额外 cross-attn 层）。late / cross-attn 在长视频、跨模态检索等特殊场景仍有优势。


## 3. 核心概念详解

### 3.1 early fusion（早期融合，拼接进 LLM）

视觉特征经 [[projector]] 对齐到 LLM 维 $d$ 后，与文本 token embedding **拼接成一个序列**进 LLM：

$$\mathbf{H}_0 = [\mathbf{H}_v^{LLM};\; \mathbf{H}_t] \in \mathbb{R}^{(N' + L_t) \times d}$$

$\mathbf{H}_v^{LLM} \in \mathbb{R}^{N' \times d}$（视觉 token，$N'$ 个），$\mathbf{H}_t \in \mathbb{R}^{L_t \times d}$（文本 token）。LLM 的 self-attention 对**全序列**做 [[自注意力]]——视觉 token 与文本 token 平权互相 attend。

- 代表：**LLaVA / LLaVA-1.5 / LLaVA-NeXt / Qwen-VL / Qwen2-VL / InternVL**。
- projector：linear / MLP / resampler（见 [[projector]]）。
- 视觉 token 位置：通常在文本 prompt 前或后（如 `<image> 用户问题`，视觉在前）。
- 优势：复用 LLM 全部层与预训练知识，实现极简，推理无额外 cross-attn 层高效。
- 劣势：视觉 token 占序列长，长序列 LLM attention $O(L^2)$ 大；视觉信息与文本平权竞争 LLM 容量。

### 3.2 late fusion（晚期融合，高层 cross-attention）

视觉特征**不进 LLM 输入端**，而是作"外部"被 LLM 高层（或专门 decoder）的 **cross-attention 层**查询：

$$\mathbf{H}_{dec}^{(l+1)} = \text{SelfAttn}(\mathbf{H}_{dec}^{(l)}) + \text{CrossAttn}(\mathbf{H}_{dec}^{(l)}, \mathbf{H}_v)$$

LLM（decoder）每层先 self-attn 处理已生成文本，再 cross-attn 查视觉特征。视觉特征是"外部记忆"。

- 代表：**Flamingo**（Perceiver resampler + LLM 每 4 层插 cross-attn）、**BLIP-2**（Q-Former + OPT/Flan-T5 decoder）。
- 视觉特征不需 projector 对齐到 LLM 输入端（但要 cross-attn 投影到 LLM 内部维）。
- 优势：LLM 主体不被视觉 token 干扰，保文本生成能力；视觉特征可冻结复用；cross-attn 层参数小可控。
- 劣势：需训专门 cross-attn 层；视觉与文本不深层交互，细节任务（OCR）可能弱；架构偏离纯 LLM，工程复杂。

### 3.3 cross-attention fusion（每层 cross-attn）

介于 early 与 late：在 LLM **每层**都插 cross-attention 层，让 LLM 每层细粒度查视觉。

- 代表：**CogVLM**（LLM 每层 FFN 加"视觉专家"分支，与文本专家并联，类似 MoE 路由视觉 / 文本分支）。
- 与 late 的区别：late 仅高层或周期性插（Flamingo 每 4 层），cross-attn fusion 每层都插。
- 优势：细粒度视觉-文本交互，保细节。
- 劣势：参数量增（每层 cross-attn），推理慢。

### 3.4 三种范式对比

| 维度 | early fusion | late fusion | cross-attention fusion |
|---|---|---|---|
| **融合点** | 浅层（LLM 输入端拼接） | 高层（cross-attn 查询视觉） | 每层（每层 cross-attn） |
| **视觉位置** | LLM 输入序列内 | 外部"记忆"，被查 | 外部，每层查 |
| **视觉是否平权** | 是（与文本同等 self-attn） | 否（被按需查） | 否（每层按需查） |
| **需 projector** | 是（对齐到 LLM 输入维） | 否（cross-attn 投影到内部维） | 否（同 late） |
| **新增参数** | projector（小） | cross-attn 层（中） | 每层 cross-attn（大） |
| **复用 LLM 预训练** | 强（self-attn 全复用） | 弱（cross-attn 需新训） | 弱（cross-attn 需新训） |
| **训练稳定性** | 高（拼接简单） | 中（cross-attn 需调） | 中 |
| **推理效率** | 高（无额外层） | 中（cross-attn 层） | 低（每层 cross-attn） |
| **长序列** | 视觉 token 占长，$O(L^2)$ 大 | 视觉外部，LLM 序列短 | 同 late |
| **细节任务（OCR）** | 强（视觉平权） | 弱-中 | 强 |
| **代表模型** | LLaVA / Qwen-VL / InternVL | Flamingo / BLIP-2 | CogVLM |

> [!tip] 实践
> 多数 VLM 选 **early fusion**——实现简单、复用 LLM、效果好。LLaVA 是 early fusion 标杆，MLP projector + 拼接进 LLaMA，证明简单路线能跑赢复杂架构。late / cross-attn 在长视频、跨模态检索、保文本生成能力等特殊场景仍有价值。

### 3.5 为什么主流倾向 early fusion

1. **简单**：拼接 + projector，无需 cross-attn 层，工程易。
2. **复用 LLM**：self-attention 全复用预训练知识，不需新训 cross-attn。
3. **效果好**：实测 LLaVA / Qwen-VL 在多数 VLM benchmark 优于 Flamingo / BLIP-2。
4. **训练稳**：拼接是温和扰动，cross-attn 新模块难训。
5. **推理高效**：无额外层，可用标准 LLM 推理引擎（vLLM / SGLang）直接服务。
6. **scaling 友好**：early fusion 随 LLM scaling 直接受益，cross-attn 需重训。

> [!warning] early fusion 不总优
> 长视频（数百帧）early fusion 视觉 token 数千，LLM $O(L^2)$ 爆。此时 late / cross-attn（视觉外部）省 LLM 序列长。Flamingo 专为长视频设计。early fusion 需 [[动态分辨率与视觉token]] + resampler 压 token 才能扛长视频。


## 4. 数学原理 / 公式

### 4.1 early fusion 的序列拼接

视觉 token $\mathbf{H}_v^{LLM} \in \mathbb{R}^{N' \times d}$（经 projector 对齐），文本 token $\mathbf{H}_t \in \mathbb{R}^{L_t \times d}$。拼接：

$$\mathbf{H}_0 = [\mathbf{H}_v^{LLM};\; \mathbf{H}_t] \in \mathbb{R}^{(N' + L_t) \times d}$$

LLM self-attention 对全序列 $\mathbf{H}_0$ 做：

$$\text{Attn}(\mathbf{H}_0) = \text{softmax}\left(\frac{(\mathbf{W}_Q \mathbf{H}_0)(\mathbf{W}_K \mathbf{H}_0)^T}{\sqrt{d_k}}\right) \mathbf{W}_V \mathbf{H}_0$$

视觉与文本 token 平权互相 attend。$L = N' + L_t$，算力 $O(L^2)$。

### 4.2 late fusion 的 cross-attention

LLM decoder 每层（或周期层）：

$$\mathbf{H}_{dec}' = \text{SelfAttn}(\mathbf{H}_{dec})$$  （处理已生成文本）
$$\mathbf{H}_{dec}'' = \text{CrossAttn}(\mathbf{Q}=\mathbf{H}_{dec}',\; \mathbf{K}=\mathbf{H}_v,\; \mathbf{V}=\mathbf{H}_v)$$  （查视觉特征）

LLM 序列长仅 $L_t$（视觉外部），算力 $O(L_t^2 + L_t \cdot N)$。

### 4.3 算力对比

设 $N' = 256$ 视觉 token，$L_t = 512$ 文本 token。

- **early fusion**：$L = 768$，self-attn $\propto 768^2 \approx 590K$。
- **late fusion**：self-attn $\propto 512^2 \approx 262K$ + cross-attn $\propto 512 \times 256 \approx 131K$ → 总 $\approx 393K$。

late 省 ~33%。但若 $N'$ 大（高分辨率 1024），early $L = 1536$，$\propto 2.36M$；late $\propto 262K + 524K = 786K$，省 ~67%。视觉 token 越多 late 越省。

### 4.4 参数量对比

设 LLM 维 $d = 4096$，层数 $L_{layer} = 32$。

- **early fusion**：仅 projector（MLP ~21M）+ LLM（7B）。新增 ~21M。
- **late fusion**：每层 cross-attn（Q/K/V/O 投影，$4 d^2 \approx 67M$/层）。32 层全插 → 2.1B 新增（大！）。Flamingo 每 4 层插 → 8 层 → ~536M。
- **cross-attention fusion**（每层）：同 late 全插 → 2.1B 新增。

late / cross-attn fusion 新增参数远大于 projector。是工程权衡的关键。

### 4.5 视觉信息的"路径长"

- early：视觉 token 进 LLM 第 0 层，经 32 层 self-attn 与文本逐层交互 → 视觉信息路径长、交互深。
- late：视觉特征外部，decoder 每层 cross-attn 查 → 视觉信息从外部注入每层，但 visual encoder 不被 LLM self-attn 处理。
- early 视觉与文本"共同演化"，late 视觉作"外部知识库"被查。


## 5. 代码示例（可选）

### 5.1 early fusion（拼接 + LLM）

```python
import torch
import torch.nn as nn

class EarlyFusionVLM(nn.Module):
    """early fusion: 视觉 token (经 projector) 与文本 token 拼接进 LLM。"""
    def __init__(self, vit, projector, llm):
        super().__init__()
        self.vit = vit            # CLIP-ViT (常冻结)
        self.projector = projector  # MLP projector
        self.llm = llm            # LLaMA / Qwen decoder

    def forward(self, images, input_ids):
        # images: (B, 3, 336, 336), input_ids: (B, L_t)
        with torch.no_grad():
            feats = self.vit(images)                  # (B, N, D_v)
        visual_tokens = self.projector(feats)         # (B, N', d) -- 对齐到 LLM 维
        text_emb = self.llm.model.embed_tokens(input_ids)  # (B, L_t, d)
        # 拼接: 视觉 + 文本 (视觉在前)
        h = torch.cat([visual_tokens, text_emb], dim=1)    # (B, N'+L_t, d)
        # LLM forward (注意 attention mask 与 position id 要适配变长拼接)
        out = self.llm(inputs_embeds=h)
        return out.logits

# 推理: LLM 自回归生成, 视觉 token 只在首步 prefill (后续 decode 不重算视觉)
```

### 5.2 late fusion（cross-attention）

```python
class CrossAttentionLayer(nn.Module):
    """LLM 每层(或周期层)插: self-attn 后 cross-attn 查视觉特征。"""
    def __init__(self, d=4096, num_heads=32):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm_v = nn.LayerNorm(d)  # 视觉特征前置 norm
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.norm3 = nn.LayerNorm(d)

    def forward(self, h, vision_feats):
        # h: (B, L_t, d) 文本, vision_feats: (B, N, d_v) 视觉(外部)
        a, _ = self.self_attn(self.norm1(h), self.norm1(h), self.norm1(h), need_weights=False)
        h = h + a                                   # residual (self-attn)
        c, _ = self.cross_attn(self.norm2(h), self.norm_v(vision_feats),
                               self.norm_v(vision_feats), need_weights=False)
        h = h + c                                   # residual (cross-attn 查视觉)
        h = h + self.ffn(self.norm3(h))
        return h                                    # (B, L_t, d) -- 文本序列长不变

class LateFusionVLM(nn.Module):
    """late fusion: LLM 每 4 层插 cross-attn, 视觉特征作外部记忆。"""
    def __init__(self, vit, llm, num_cross_layers=8):
        super().__init__()
        self.vit = vit
        self.llm = llm
        self.cross_layers = nn.ModuleList([CrossAttentionLayer() for _ in range(num_cross_layers)])
        # 投影视觉特征到 LLM 内部维
        self.v_proj = nn.Linear(1024, 4096)

    def forward(self, images, input_ids):
        with torch.no_grad():
            feats = self.vit(images)                 # (B, N, D_v)
        v = self.v_proj(feats)                       # (B, N, d)
        h = self.llm.model.embed_tokens(input_ids)    # (B, L_t, d)
        # LLM 每 4 层后插 cross-attn
        for i, layer in enumerate(self.llm.layers):
            h = layer(h)
            if (i + 1) % 4 == 0:
                h = self.cross_layers[i // 4](h, v)  # 查视觉特征
        return self.llm.lm_head(h)
```

### 5.3 简化对比示意

```python
# early: 视觉进输入序列 -> [v1,v2,...,v_N', t1,t2,...,t_Lt] -> LLM self-attn 全序列
# late:  视觉外部, LLM decoder self-attn 文本 + cross-attn 查视觉
# cross: late 但每层都 cross-attn (不是每 4 层)

# 推理时:
# early: 首步 prefill 算视觉+文本, 后续 decode 只算文本 (视觉 KV cache 复用)
# late:  每 decode 步都要 cross-attn 查视觉 (视觉特征 cache, 但 cross-attn 层每步跑)
# -> early decode 快, late decode 慢 (多 cross-attn 层)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[ViT与vision encoder]]（视觉特征源）、[[projector]]（early fusion 的对齐桥）、[[自注意力]]（early fusion 的核心机制）、[[多头注意力]]（cross-attn 也是 MHSA 变体）、[[residual]]（融合层的残差结构）、[[decoder-only architecture]]（early fusion 下游 LLM 形态）、[[FFN]]（融合层内 FFN）。
- **下游（应用）**: [[动态分辨率与视觉token]]（early fusion 高分辨率 token 数控制）、[[多模态sequence packing]]（early fusion 变长视觉 token 的 packing）、LLaVA / Qwen-VL / InternVL（early）、Flamingo / BLIP-2（late）、CogVLM（cross-attn）、[[新模型接入]]（推理引擎支持 early / cross-attn 融合的算子）。
- **对比 / 易混**:
  - **early vs late vs cross-attn**：见 §3.4 对比表。
  - **fusion 范式 vs [[projector]]**：projector 是 early fusion 的对齐桥（视觉特征 → LLM-ready token）；late / cross-attn fusion 用 cross-attn 直接查视觉特征，不需 projector。两者是"拼接进 LLM" vs "外部 cross-attn" 的架构选择。
  - **fusion vs [[多模态MoE路由]]**：前者是模态怎么合（拼 / cross-attn）；后者是 LLM 内专家怎么路由（视觉专家 / 文本专家）。CogVLM 是 fusion + MoE 的混合。正交维度。
  - **early fusion vs [[VLM RL与多模态reward]]**：early fusion 是前向架构，VLM RL 是后向训练对齐。early fusion VLM 可用 RLHF / GRPO 对齐。


## 7. 常见误区与易错点

> [!warning] 误区 1：early fusion 就是不对齐直接拼
> 不对。仍需 [[projector]] 把视觉特征对齐到 LLM 维 + 分布。直接拼维度对不上、分布偏离，LLM 学不动。projector 是 early fusion 必经桥。

> [!warning] 误区 2：late fusion 比 early 高级
> 不一定。late fusion 在多数 VLM benchmark 不如 early（LLaVA 系）。late 适合长视频、保文本生成能力等特殊场景。简单路线（early + MLP projector）反而强。

> [!warning] 误区 3：cross-attention fusion = late fusion
> 概念重叠。cross-attn 是机制（query 视觉），late 是位置（高层）。Flamingo 的 late 也是用 cross-attn（每 4 层插）。CogVLM 的 cross-attn fusion 是每层插。严格说 cross-attn 是 mechanism，early/late 是 position。本文 cross-attn fusion 特指"每层插"。

> [!warning] 误区 4：early fusion 推理慢因视觉 token 多
> 视觉 token 只在首步 prefill 算，后续 decode 复用 KV cache 不重算。early fusion decode 每步只算新 1 token，反而比 late（每步 cross-attn 查视觉）快。early fusion 的问题是 prefill 长序列 $O(L^2)$，不是 decode 慢。

> [!warning] 误区 5：融合范式决定一切
> 融合点重要但不是全部。vision encoder 质量、projector 设计、训练数据、LLM 规模都关键。LLaVA 之所以强不只因 early fusion，还有 CLIP-ViT + 大量指令数据 + LLM scaling。

> [!warning] 误区 6：late fusion 不需任何对齐
> 仍需把视觉特征投影到 LLM 内部维 $d$（cross-attn 的 K/V 投影）。只是不进 LLM 输入端拼接，分布对齐由 cross-attn 的可学投影完成。

> [!warning] 误区 7：early fusion 视觉 token 必须在前
> 位置可配。LLaVA 视觉在前（`<image> 问题`），也有的在后或中间。位置由训练数据格式定。但视觉与文本的相对位置影响 attention，需训练时一致。


## 8. 延伸细节

### 8.1 Flamingo 的 late fusion 设计

Flamingo（Alayrac et al. 2022）是 late fusion 标杆：Perceiver Resampler 把变长视觉特征压成固定 64 token；LLM（Chinchilla）每 4 层插 cross-attn 层（gated，初始化为 0 保训练稳定）。专为**长视频 / 多图交错**设计——视觉特征外部，LLM 序列短。但 cross-attn 层需新训，偏离纯 LLM。

### 8.2 BLIP-2 的 Q-Former + decoder

BLIP-2（Li et al. 2023）：Q-Former（小型 Transformer，32 query）从冻结 image encoder 提视觉特征 → 固定 32 token → 喂冻结 LLM（OPT / Flan-T5）的 decoder。Q-Former 是"late fusion 的 projector"——既对齐又压缩，但与 LLM 解耦。

### 8.3 CogVLM 的"视觉专家"MoE

CogVLM（Wang et al. 2023）：LLM 每层 FFN 加"视觉专家"分支（与文本专家并联，门控路由）。视觉与文本在 LLM 内部分支处理，是 cross-attention fusion 的 MoE 变体。参数大（每层多视觉专家）但保细节。与 [[多模态MoE路由]] 相关。

### 8.4 LLaVA 证明 early fusion 简单有效

LLaVA（Liu et al. 2023）：CLIP-ViT-L/14 + 2 层 MLP projector + 拼接进 LLaMA。仅训 projector + 微调 LLM，简单架构在 VLM benchmark 接近或超过 GPT-4V。证明 early fusion 是 scaling 友好、工程简单的强基线。后续 LLaVA-1.5 / LLaVA-NeXt / Qwen-VL / InternVL 都延续 early fusion。

### 8.5 推理引擎对 fusion 范式的支持

vLLM / SGLang 主要支持 early fusion（标准 LLM 前向，视觉 token 拼接进 prompt）。late / cross-attn fusion 需推理框架支持 cross-attn 层（Flamingo 推理需定制）。是 [[新模型接入]] 的考量——early fusion VLM 易上推理引擎，late / cross-attn 需额外适配。

### 8.6 内容来源

整理自 Flamingo / BLIP-2 / LLaVA / LLaVA-1.5 / Qwen-VL / InternVL / CogVLM 论文与代码、多模态融合综述。具体超参与 cross-attn 层数以原论文为准。

---
相关: [[VLM架构]]、[[ViT与vision encoder]]、[[projector]]、[[动态分辨率与视觉token]]、[[多模态sequence packing]]、[[多模态MoE路由]]、[[新模型接入]]、[[VLM RL与多模态reward]]
