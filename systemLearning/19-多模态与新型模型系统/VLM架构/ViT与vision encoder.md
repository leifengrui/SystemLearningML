# ViT 与 vision encoder

> **所属章节**: [[VLM架构]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: Vision Transformer / 视觉 Transformer / ViT / vision encoder / 视觉编码器 / patch embedding / class token / [CLS] token
> **难度**: 中（需懂 [[自注意力]]、[[多头注意力]]、[[FFN]]、[[位置编码]]、[[residual]]、[[decoder-only architecture]]）


## 1. 一句话定义

**ViT（Vision Transformer）** 是把**图像当成一串 patch token 序列**喂给 Transformer encoder 的视觉骨干网络——先把图像切成固定大小 patch（如 16×16 像素一块），每 patch 经一个线性层投影成一个 token embedding（即 **patch embedding**），叠加 **position embedding**（告知每个 patch 的空间位置），可选地在序列最前面加一个可学习的 **class token（[CLS]）** 作全局表示位，过若干层 [[自注意力]] encoder 后输出视觉特征序列；CLIP / DINOv2 / SigLIP 等 **vision encoder** 在 VLM（视觉语言模型）中担任"图像 → token 序列"的前端——把图像编成与文本 token 同构的视觉 token，再喂给 LLM。

> [!note] 三句话定位
> - **是什么**：图像切 patch → 线性投影成 token → 加位置编码 → 过 Transformer encoder → 视觉特征序列。class token（[CLS]）或 global average pooling 取全局表示。
> - **为什么**：Transformer 在 NLP 称王，把图像"序列化"就能复用同一套 attention 机制，吃下大数据后超越 CNN。VLM 用预训练 vision encoder 把图像变成 LLM 能吃的 token。
> - **与 VLM 关系**：vision encoder 是 VLM 的"眼睛"——LLaVA 用 CLIP-ViT，Qwen-VL 用 ViT + resampler，InternVL 用 InternViT。vision encoder 出的视觉 token 经 [[projector]] 对齐后进 LLM。


## 2. 为什么需要它（动机与背景）

### 2.1 Transformer 想吃图像

NLP 里 Transformer 已称王（BERT / GPT），但图像是 2D 像素网格，CNN 长期统治。问题：能否把图像也变成"一串 token"，让 Transformer 直接吃？ViT 的答案是 **patch 切分 + 线性投影**：把 $H \times W \times 3$ 的图像重排成 $N$ 个 $P \times P \times 3$ 的小块，每块当一个"词"，线性投影成 embedding——就回到了 NLP 那一套（一串 token + 位置编码 + attention）。

### 2.2 无归纳偏置 → 需大数据

CNN 有**归纳偏置（inductive bias）**：平移不变性（卷积核扫过整图）、局部性（只看邻域）、层级性（低层边、高层语义）。ViT 的 [[自注意力]] 是**全局**的——任意两 patch 都能直接交互，**没有局部性 / 平移不变性偏置**。代价：缺先验 → 需**更大数据**才能学好（CNN 在小数据 ImageNet-1k 上更稳，ViT 需 ImageNet-21k / JFT-300M 这种量级才超越）。这是 ViT 与 CNN 的核心权衡。

### 2.3 VLM 要复用 LLM，就得有 vision encoder

LLM 是 [[decoder-only architecture]]，吃 token 序列。要让 LLM"看图"，必须把图像变成 token。两条路：
1. 从零训一个图像 tokenizer（如 VQ-VAE 离散 token）——但离散 token 损信息、训不稳。
2. **用预训练 vision encoder 提连续视觉特征**，再经 [[projector]] 对齐到 LLM 空间——主流 VLM（LLaVA / Qwen-VL / InternVL / CogVLM）都走这条。

vision encoder 本身在大量图文对上预训练（CLIP 对齐图文 / DINOv2 自监督），特征质量高，VLM 只需**冻结或微调**它即可复用。

### 2.4 高分辨率需动态处理

传统 ViT 要固定 224×224 输入，文档 / 图表 OCR 等高分辨率场景 resize 会丢小字。NaViT 支持任意分辨率；InternVL / Qwen-VL 用 tile 切分子图分别 encode。详见 [[动态分辨率与视觉token]]。


## 3. 核心概念详解

### 3.1 patch 切分

输入图像 $x \in \mathbb{R}^{H \times W \times C}$（$C=3$ RGB）。切成 $N$ 个 $P \times P$ 的 patch：

$$N = \frac{H}{P} \cdot \frac{W}{P}$$

例：224×224 图，$P=16$，则 $N = 14 \times 14 = 196$ 个 patch。每个 patch 是 $16 \times 16 \times 3 = 768$ 维向量（拉平）。

> [!note] patch 大小权衡
> - $P$ 小（如 8）：$N$ 多，token 长，attention $O(N^2)$ 爆，但分辨率细节多。
> - $P$ 大（如 32）：$N$ 少，省算力，但每 patch 信息粗。
> - 主流 $P=16$（ViT-Base/CLIP-ViT-L/14 用 14，patch 16）。

### 3.2 patch embedding（线性投影）

每 patch 拉平成 $P^2 \cdot C$ 维向量，再经一个线性层 $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$ 投到 $D$ 维（$D$=embedding dim，如 768）：

$$\mathbf{z}_i = \mathbf{x}_i^{patch} \mathbf{E}, \quad i = 1, \dots, N$$

实现上等价于 **Conv2d(stride=kernel=P)**：一个 kernel=16、stride=16 的卷积，输出通道=D，正好把图切成 $N$ 个 $D$ 维 token。这是工程上最常用写法（不用循环切 patch）。

### 3.3 class token（[CLS]）

在 token 序列最前加一个**可学习的 [CLS] token** $\mathbf{z}_{cls}$（$D$ 维，类似 BERT 的 [CLS]）：

$$\mathbf{Z}_0 = [\mathbf{z}_{cls};\; \mathbf{z}_1; \dots; \mathbf{z}_N]$$

过完 encoder 后，**取 [CLS] 的输出**作全图表示（接分类头）。设计动机来自 BERT [CLS]：一个专门聚合全局信息的"槽位"，由 attention 从所有 patch 收集。

### 3.4 class token vs global average pooling

取全图表示的两种方式：

| 方式 | 做法 | 出处 | 优劣 |
|---|---|---|---|
| **class token** | 序列前加可学习 [CLS]，取其输出 | ViT 原版 / DeiT | 显式全局槽位，但需专门训；CLS 未必真聚合全局 |
| **global average pooling (GAP)** | 对所有 patch 输出取平均 | DeiT / DINOv2 / 多数现代 ViT | 无额外参数，更稳；但所有 patch 平均未必优于专门 CLS |

> [!tip] 实践
> 现代预训练 ViT（DINOv2、SigLIP）多用 **GAP + LayerNorm** 取全图表示，比 [CLS] 更稳。VLM 取视觉特征时，**通常取所有 patch token 序列**（不只 [CLS]）喂给 [[projector]]——因为 LLM 要看每个 patch 的局部细节，不只全局向量。

### 3.5 position embedding

attention 本身**置换不变**（打乱 patch 顺序结果同），需加位置编码告知空间布局。ViT 用**可学习的 1D 绝对位置 embedding** $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$：

$$\mathbf{Z} = \mathbf{Z}_0 + \mathbf{E}_{pos}$$

> [!warning] 误区：1D 位置编码丢了 2D 结构
> ViT 原版用 1D 顺序位置编码（patch 按行扫描编号），看似丢 2D 邻域信息。实测：足够数据下模型自己学会 2D 关系（attention 全局可补）。但小数据下 2D-aware 位置编码（如相对位置、Axial）更稳。NaViT 用 factorized 2D 位置编码。

### 3.6 Transformer encoder body

$\mathbf{Z}$ 过 $L$ 层 encoder，每层：

$$\mathbf{Z}' = \mathbf{Z} + \text{MHSA}(\text{LN}(\mathbf{Z}))$$
$$\mathbf{Z}'' = \mathbf{Z}' + \text{FFN}(\text{LN}(\mathbf{Z}'))$$

即 [[多头注意力]] + [[residual]] + [[FFN]]，前置 LayerNorm（Pre-LN）。$L=12$（ViT-Base）/ 24（ViT-Large）/ 32（ViT-Huge）。

### 3.7 CLIP / DINOv2 / SigLIP 在 VLM 中的角色

| vision encoder | 预训练方式 | 特点 | VLM 用例 |
|---|---|---|---|
| **CLIP-ViT** | 图文对比学习（对比损失对齐图文 embedding） | 视觉特征与文本语义对齐好，泛化强 | LLaVA / LLaVA-1.5（CLIP-ViT-L/14）|
| **DINOv2** | 自监督（student-teacher 蒸馏 + SwAV 聚类） | 不需文本，特征质量高，分割 / 检测强 | Idefics2 / 一些开源 VLM 选 |
| **SigLIP** | sigmoid 对比损失（替 softmax InfoNCE） | 更适合大 batch、小数据，性能强 | PaliGemma / Llama 3-Vision / InternVL2 |
| **InternViT** | InternLM 自训 ViT（6B 参数） | 大 vision encoder，多分辨率 | InternVL 系列 |

> [!note] VLM 怎么用 vision encoder
> VLM 流水线：图 → **vision encoder**（出 $N$ 个 $D$ 维视觉 token）→ **[[projector]]**（对齐到 LLM 维 $d$）→ 与文本 token 拼接 → **LLM**（[[decoder-only architecture]]）→ 出文本。vision encoder 通常**冻结**（LLaVA）或**部分微调**（InternVL 解冻后几层 / Qwen-VL 微调）。冻结省算力、保预训练知识；微调适下游任务但易过拟。

### 3.8 高分辨率处理

固定 224×224 对文档 / OCR / 图表不够。策略：
1. **NaViT**：原生任意分辨率，patch 切变长序列，packing 进 batch（见 [[动态分辨率与视觉token]]、[[多模态sequence packing]]）。
2. **Tile 切分**（InternVL / Qwen-VL）：把高分辨率图切成多个固定大小子图（如 448×448 tile），每 tile 独立 encode，再合并 token。token 数 = $N_{tile} \times N_{per\_tile}$，易爆 → 需 [[projector]] resampler 压缩。


## 4. 数学原理 / 公式

### 4.1 patch embedding 维度

输入 $x \in \mathbb{R}^{H \times W \times C}$。切 $P \times P$ patch，共：

$$N = \frac{H}{P} \cdot \frac{W}{P}$$

每 patch 拉平 $P^2 C$ 维。线性投影 $\mathbf{E} \in \mathbb{R}^{P^2 C \times D}$ → 每 patch 成 $D$ 维 token。

总参数（patch embedding 层）：$P^2 C \cdot D$。例 $P=16, C=3, D=768$：$768 \times 768 \approx 0.59M$。

### 4.2 token 数与分辨率

$$N = \frac{H W}{P^2}$$

例：
- 224×224, $P=16$：$N=196$
- 336×336（CLIP-ViT-L/14@336）, $P=14$：$N=576$
- 448×448, $P=14$：$N=1024$
- 1024×1024（高分辨率 tile）, $P=16$：$N=4096$

> [!warning] token 数爆炸
> attention 是 $O(N^2)$。$N=4096$ 时 attention 矩阵 $4096^2 \approx 16.7M$ 元素，显存与算力爆。高分辨率需 [[动态分辨率与视觉token]] 的 tile + resampler 压缩。

### 4.3 attention 计算量

单层 [[多头注意力]]（$h$ 头，每头 $d_k = D/h$）：

$$\text{FLOPs}_{\text{attn}} \approx 2 \cdot (2 N^2 D + 2 N D^2) \cdot L$$

$N^2$ 项是 attention 矩阵（全局交互），$D^2$ 项是 Q/K/V/O 投影。$N$ 大（高分辨率）→ $N^2$ 主导。

### 4.4 class token 的信息聚合

[CLS] 的输出 $\mathbf{z}_{cls}^{(L)}$ 是第 $L$ 层后 CLS 位上的表示。由 attention：

$$\mathbf{z}_{cls}^{(l+1)} = \text{Attn}(\mathbf{z}_{cls}^{(l)}, [\mathbf{z}_{cls}^{(l)}; \mathbf{z}_1^{(l)}; \dots])$$

CLS 可对所有 patch 做 attention（全局聚合）。但实践中 CLS 未必真聚合所有 patch——DINOv2 论文显示 GAP 更稳。

### 4.5 position embedding 注入

$$\mathbf{Z}^{(0)} = [\mathbf{z}_{cls}; \mathbf{E} \mathbf{x}_1; \dots; \mathbf{E} \mathbf{x}_N] + \mathbf{E}_{pos}$$

$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$ 可学习。加法注入（非 concat）。


## 5. 代码示例（可选）

### 5.1 手写 ViT patch embedding（等价 Conv2d 切 patch）

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """图像 -> patch token 序列。等价于切 patch + 线性投影, 但用 Conv2d 一步到位。"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # N = (H/P)*(W/P)
        # Conv2d: kernel=patch_size, stride=patch_size -> 输出 (N, embed_dim, 1, 1)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)              # (B, D, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D), N=196
        return x

pe = PatchEmbed()
x = torch.randn(2, 3, 224, 224)
out = pe(x)
print(out.shape)  # torch.Size([2, 196, 768])  -- 196 个 768 维 token
```

### 5.2 完整 ViT encoder（class token + position embedding + 多层）

```python
class ViTBlock(nn.Module):
    """Pre-LN Transformer encoder block: MHSA + residual + FFN + residual."""
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, z):
        a, _ = self.attn(self.norm1(z), self.norm1(z), self.norm1(z), need_weights=False)
        z = z + a                                       # residual
        z = z + self.ffn(self.norm2(z))                # residual
        return z

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))   # [CLS] 可学习
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        z = self.patch_embed(x)                              # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, D)
        z = torch.cat([cls, z], dim=1)                       # (B, N+1, D)
        z = z + self.pos_embed                               # 加位置编码
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        return z[:, 0]                                       # 取 [CLS] 输出 -> 分类

vit = ViT()
out = vit(torch.randn(2, 3, 224, 224))
print(out.shape)  # torch.Size([2, 1000])
```

### 5.3 取所有 patch token（VLM 用法）

```python
# VLM 不取 [CLS], 而是取所有 patch token 序列喂给 projector
def vit_features_for_vlm(vit, x):
    z = vit.patch_embed(x)                                  # (B, N, D)
    z = z + vit.pos_embed[:, 1:]                            # 不要 CLS 位的位置 (VLM 常省 CLS)
    for blk in vit.blocks:
        z = blk(z)
    return vit.norm(z)                                      # (B, N, D) -- 喂 projector
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[自注意力]]（核心机制）、[[多头注意力]]（多子空间）、[[FFN]]（每层后处理）、[[位置编码]]（patch 顺序注入）、[[residual]]（每层残差）、[[decoder-only architecture]]（VLM 下游 LLM 形态）、[[tokenization]]（视觉 token 化类比文本 tokenization）。
- **下游（应用）**: [[projector]]（视觉 token 对齐到 LLM）、[[early与late fusion]]（vision encoder 是融合的视觉侧前端）、[[动态分辨率与视觉token]]（高分辨率切 patch）、[[多模态sequence packing]]（变长视觉 token pack 进 batch）、LLaVA / Qwen-VL / InternVL / CLIP / DINOv2 / SigLIP / PaliGemma / CogVLM。
- **对比 / 易混**:
  - **ViT vs CNN**：ViT 全局 attention 无归纳偏置需大数据；CNN 局部卷积有平移不变性小数据稳。详见下方对比表。
  - **class token vs GAP**：前者显式全局槽位（ViT 原版），后者所有 patch 平均（DINOv2 现代主流）。VLM 通常取全 patch 序列而非任一全局向量。
  - **vision encoder vs [[projector]]**：前者提视觉特征（冻结 / 微调预训练 ViT），后者把视觉特征对齐到 LLM 维。前者是"眼睛"，后者是"翻译"。
  - **ViT vs [[多模态MoE路由]]**：ViT 是 dense attention 全 patch 交互；MoE 路由是专家稀疏激活，可替 ViT FFN 提容量。

| 维度 | ViT | CNN（ResNet/ConvNeXt） |
|---|---|---|
| 机制 | 全局 [[自注意力]] | 局部卷积 |
| 归纳偏置 | 无（平移不变性 / 局部性都无） | 有（平移不变 + 局部） |
| 数据需求 | 大（ImageNet-21k / JFT 量级才超越） | 小（ImageNet-1k 稳） |
| 感受野 | 全局（第 1 层即可看全图） | 逐层扩（深层才全图） |
| 长程依赖 | 直接（attention） | 间接（多层堆叠） |
| 算力 | $O(N^2)$，token 多则爆 | $O(HW)$，与分辨率线性 |
| 可扩展性 | 强（堆层 / 增 dim 即扩） | 强（ConvNeXt 也能堆） |


## 7. 常见误区与易错点

> [!warning] 误区 1：ViT 必须有 [CLS] token
> 不必须。原版 ViT 有，但 DINOv2 / SigLIP 等现代 ViT 用 GAP 取全图表示，省 [CLS]。VLM 取视觉特征时更常取**全 patch 序列**（不只 [CLS]），因为 LLM 要看局部细节。

> [!warning] 误区 2：ViT 比 CNN 强因为它用 attention
> 不全面。ViT 在大数据下超越 CNN，但小数据下 CNN 更稳（归纳偏置优势）。ViT 强在**可扩展性 + 大数据**，不是 attention 本身更优。

> [!warning] 误区 3：patch embedding 是把图切块再喂
> 实现上是 **Conv2d(stride=kernel=P)** 一步到位——一个卷积把图切成 $N$ 个 $D$ 维 token。不是真"切 patch 再线性"。但**数学等价**于切 patch + 线性投影。

> [!warning] 误区 4：1D 位置编码丢 2D 信息
> 表面看是。但 attention 全局可补，足够数据下模型自学会 2D 关系。小数据下才需 2D-aware 编码。NaViT 用 factorized 2D 编码处理任意分辨率。

> [!warning] 误区 5：VLM 的 vision encoder 一定要训
> LLaVA-1.5 / 早期 Qwen-VL **冻结** CLIP-ViT 只训 projector + LLM，效果就很好。后期为提上限才解冻微调（InternVL 解冻后几层、Qwen2-VL 全微调）。冻结省算力、保预训练知识。

> [!warning] 误区 6：vision encoder 出的 [CLS] 就是全图表示
> [CLS] 未必真聚合全局（DINOv2 论文显示）。且 VLM 需局部细节（哪个 patch 是字 / 哪个是图），不只全局向量。**取全 patch 序列**是 VLM 标准做法。

> [!warning] 误区 7：高分辨率直接喂 ViT
> 传统 ViT 固定分辨率（224）。高分辨率需 NaViT 原生支持，或 tile 切子图分别 encode（见 [[动态分辨率与视觉token]]）。直接 resize 丢小字。


## 8. 延伸细节

### 8.1 Conv2d 切 patch 的等价性

设输入 $(B, 3, H, W)$，Conv2d(kernel=$P$, stride=$P$, out_channels=$D$) 输出 $(B, D, H/P, W/P)$。展平空间维 → $(B, N, D)$，$N = (H/P)(W/P)$。每输出通道对应一个 patch 的 $D$ 维 token。数学上**完全等价**于"切 patch + 线性投影"，但 GPU 友好（卷积算子高度优化）。

### 8.2 CLIP-ViT-L/14@336 的 VLM 经典

LLaVA-1.5 用 CLIP-ViT-L/14@336（$P=14$, 336×336 → $N=24^2=576$ token，$D=1024$）。冻结 vision encoder，训 2 层 MLP projector（见 [[projector]]）。视觉 token 与文本 token 拼接进 LLaMA。是 [[early与late fusion]] 的 early fusion 范式。

### 8.3 DINOv2 的 register token

DINOv2 发现 ViT 高 attention 出现"异常 patch"（artifacts，某些 patch attention 爆），引入 **register token**（额外几个无语义槽位吸收冗余 attention）缓解。现代 ViT（DINOv2、SigLIP2）多有 register。

### 8.4 SigLIP 的 sigmoid 损失

CLIP 用 softmax InfoNCE 对比损失（需全 batch 算 softmax）。SigLIP 用 **sigmoid 损失**（每图文对独立判断相关与否），省算力、更适合大 batch / 小数据，性能强。Llama 3-Vision / PaliGemma 用 SigLIP。

### 8.5 vision encoder 的参数量与算力

CLIP-ViT-L/14：~300M 参数。InternViT-6B：6B 参数（大 vision encoder）。vision encoder 在 VLM 推理时占算力（高分辨率 $N$ 大 → attention $O(N^2)$ 爆），需 tile + resampler 压 token 数（见 [[projector]]、[[动态分辨率与视觉token]]）。

### 8.6 内容来源

整理自 ViT 原论文（Dosovitskiy et al. 2020 "An Image is Worth 16×16 Words"）、DeiT、DINOv2、SigLIP 论文、NaViT 论文、LLaVA / Qwen-VL / InternVL 技术报告，及 HuggingFace `transformers` ViT 实现。具体数字以原论文为准，本文为知识整理，非逐字引用。

---
相关: [[VLM架构]]、[[projector]]、[[early与late fusion]]、[[动态分辨率与视觉token]]、[[多模态sequence packing]]
