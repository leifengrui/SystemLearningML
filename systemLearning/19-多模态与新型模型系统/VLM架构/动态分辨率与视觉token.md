# 动态分辨率与视觉 token

> **所属章节**: [[VLM架构]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: 动态分辨率 / Native Resolution ViT / NaViT / 任意分辨率 / tile 切分 / 视觉 token 数控制 / 任意纵横比 / packing
> **难度**: 中（需懂 [[ViT与vision encoder]]、[[projector]]、[[自注意力]]、[[位置编码]]、[[多模态sequence packing]]）


## 1. 一句话定义

**动态分辨率** 是 VLM 处理任意大小图像的机制——传统 ViT 要求固定输入分辨率（如 224×224），高分辨率图（文档 / 图表 / 4K）强行 resize 会**丢小字与细节**；**NaViT（Native Resolution ViT）** 支持任意分辨率——把任意大小图按 patch 切成**变长 token 序列**、用 attention mask 隔离不同图 pack 进一个 batch；高分辨率策略主流还有 **tile 切分**（InternVL / Qwen-VL 把高分辨率图切成多个固定大小子图分别 encode 再合并 token）；视觉 token 数与分辨率严格相关 $N = (H/P)(W/P)$，高分辨率 token 爆炸需 [[projector]] 的 resampler 压缩，是 VLM 视觉侧"清晰度 vs 算力"的核心权衡。

> [!note] 三句话定位
> - **是什么**：支持任意大小图——NaViT 原生变长 patch 序列 + packing；tile 切分子图分别 encode。token 数 $N=(H/P)(W/P)$ 随分辨率线性变。
> - **为什么**：文档 / OCR / 图表需高清，固定 224×224 resize 丢小字。但高分辨率 token 数千，LLM $O(N^2)$ 爆，需 resampler 压。
> - **与 [[projector]] 关系**：动态分辨率解决"看清"，resampler 解决"看清后省 LLM 算力"。前者提清晰度，后者控 token 预算。[[多模态sequence packing]] 处理 batch 内变长视觉 token。


## 2. 为什么需要它（动机与背景）

### 2.1 固定分辨率的局限

传统 ViT（如 CLIP-ViT-L/14）固定输入 224×224 或 336×336。问题：
1. **resize 丢信息**：高分辨率文档 / 图表强行 resize 成 224×224，小字 / 细线变模糊，OCR 失效。
2. **纵横比扭曲**：非方形图 resize 到方形（如 1080×1920 竖屏截图 → 224×224）扭曲，比例信息丢。
3. **算力浪费**：简单图（如纯色背景）也固定算 196 token，浪费。

文档理解（DocVQA / ChartQA / OCR）、长截图、表格、密集场景（目标检测小物体）等任务**必须高分辨率**才有信号。

### 2.2 任意分辨率的挑战

支持任意分辨率不是"直接喂任意大图"那么简单：
1. **token 数变长**：224×224 → 196 token；1024×1024 → 4096 token；4K → 数万 token。变长序列难 batching。
2. **attention $O(N^2)$ 爆**：$N=4096$ attention 矩阵 $16.7M$ 元素，算力 / 显存爆。
3. **位置编码**：1D 固定长度 position embedding 不能直接处理任意长度。
4. **纵横比任意**：不能强 reshape，要保原比例。

NaViT 与 tile 切分是两种主流解法。

### 2.3 NaViT 的解法（原生任意分辨率）

NaViT（Dehghani et al. 2023 "Patch n' Pack"）核心思想：**Patch n' Pack**——任意大小图按 patch 切成变长 token 序列，**多张图 pack 进一个固定长度序列**，用 attention mask 隔离不同图。

- patch 大小 $P$ 固定（如 16），但 $H, W$ 任意 → token 数 $N$ 变长。
- 多图 pack 进一个 batch 序列（如总长 1024），不同图 token 段用 block mask 隔离 attention（图 A 的 token 不 attend 图 B 的）。
- position embedding 用 **factorized 2D**（行 + 列分开编码）支持任意 $H, W$。
- 训练时 random resolution / random aspect ratio 数据增强，模型学多尺度。

### 2.4 tile 切分的解法（子图分别 encode）

InternVL / Qwen-VL 等用 tile 切分：
- 高分辨率图按固定大小（如 448×448）切成多个 tile。
- 每 tile 独立过 vision encoder（仍是固定分辨率 ViT），出固定 token 数。
- 多 tile 的 token 合并（+ 可选 thumbnail 全图 token）。
- token 数 = $N_{tile} \times N_{per\_tile}$，可控。

tile 切分用"多个固定分辨率子图"近似高分辨率，复用固定 ViT，但 token 数仍多，需 resampler 压。

### 2.5 token 爆炸 → resampler

无论 NaViT 还是 tile，高分辨率 → token 数百上千 → LLM $O(L^2)$ 爆。需 [[projector]] 的 resampler（Q-Former / Perceiver）压成固定少 token。是 [[projector]] 在高分辨率场景的关键作用。


## 3. 核心概念详解

### 3.1 token 数与分辨率

ViT 把 $H \times W$ 图切 $P \times P$ patch，token 数：

$$N = \frac{H}{P} \cdot \frac{W}{P} = \frac{HW}{P^2}$$

- 224×224, $P=16$：$N = 196$
- 336×336, $P=14$：$N = 576$（CLIP-ViT-L/14@336）
- 448×448, $P=14$：$N = 1024$
- 1024×1024, $P=16$：$N = 4096$
- 4K（3840×2160）, $P=16$：$N = 240 \times 135 = 32400$！

token 数与分辨率**线性**（$HW$）。高分辨率 token 爆炸，attention $O(N^2)$ 不可承受。

### 3.2 NaViT 的 packing

NaViT 把多张变长图 pack 进一个固定长度序列（如 max_len=1024）：

```
序列: [图A 196 token][图B 256 token][图C 100 token][pad]
attention mask:
  图A token 只 attend 图A
  图B token 只 attend 图B
  图C token 只 attend 图C
  -> block-diagonal mask
```

- 一张图不够 1024 → 多张拼。
- 超长图（如 $N=2000$）超出 max_len → 切两段分别 pack（或限制 max resolution）。
- attention mask 是 block-diagonal（每图一块，块外 mask 掉）。

> [!note] 与 [[多模态sequence packing]] 关系
> NaViT 的 packing 就是多模态 sequence packing 的视觉版——变长视觉 token pack 进 batch，用 attention mask 隔离。[[多模态sequence packing]] 处理"视觉 + 文本"变长序列的 packing 与 mask。NaViT 是视觉侧 packing，[[多模态sequence packing]] 是视觉 + 文本联合 packing。

### 3.3 factorized 2D position embedding

NaViT 用**因式分解 2D 位置编码**——把 patch 的行号 $r$ 与列号 $c$ 分开编码：

$$\text{PE}(r, c) = \text{PE}_{row}(r) + \text{PE}_{col}(c)$$

$\text{PE}_{row}, \text{PE}_{col}$ 各自是可学习表，行 / 列独立。任意 $H, W$ 都能查（行号 $r \in [0, H/P)$，列号 $c \in [0, W/P)$）。比 ViT 原版 1D 顺序编码更适合任意分辨率 + 任意纵横比。

### 3.4 tile 切分（InternVL / Qwen-VL）

高分辨率图 $H \times W$ 按固定 tile 大小 $T \times T$（如 448）切：

$$N_{tile} = \lceil H/T \rceil \cdot \lceil W/T \rceil$$

每 tile resize / pad 到 $T \times T$，独立过 ViT 出 $N_{per\_tile} = (T/P)^2$ token。总 token：

$$N_{total} = N_{tile} \cdot N_{per\_tile} + N_{thumb}$$

$N_{thumb}$ 是可选的全图缩略图 token（保全局上下文）。

例：1024×1024 图，$T=448$, $P=14$ → $N_{tile} = 3 \times 3 = 9$，$N_{per\_tile} = 32^2 = 1024$，$N_{total} = 9 \times 1024 + 1024 \approx 10K$ token。需 resampler 压到几百。

### 3.5 tile vs NaViT 对比

| 维度 | NaViT（原生任意分辨率） | tile 切分 |
|---|---|---|
| **vision encoder** | 改造 ViT 支持变长 + packing | 复用固定 ViT，子图分别 encode |
| **token 数** | $N = (H/P)(W/P)$（精确） | $N_{tile} \cdot N_{per\_tile}$（粗略，因 tile 需 pad） |
| **batching** | packing 进固定序列，attention mask 复杂 | 每 tile 独立 encode，batching 简单 |
| **实现复杂度** | 高（改 ViT + packing） | 低（复用固定 ViT） |
| **细粒度** | 高（按 patch 切，无 resize） | 中（tile 内 resize，但 tile 多则细） |
| **纵横比** | 保原比例 | tile 切分保局部比例，全局比例靠 thumbnail |
| **代表** | NaViT / SigLIP2 / Qwen2-VL | InternVL / Qwen-VL / LLaVA-NeXt |

> [!tip] 实践
> InternVL / Qwen-VL 用 tile 切分（实现简单，复用固定 ViT）。Qwen2-VL 转 NaViT 风格（原生任意分辨率 + packing）。两者效果都强，tile 是工程妥协，NaViT 是原生方案。

### 3.6 视觉 token 压缩

高分辨率 token 爆，主流压缩：
1. **resampler**（[[projector]]）：Q-Former / Perceiver 把变长 $N$ 压到固定 $N'$（如 256）。
2. **pixel shuffle / merge**：把相邻 $k \times k$ patch token 合并成 1 个（如 $2 \times 2$ → 1，token 数 /4）。Qwen-VL 用。
3. **downsample attention**：低层 ViT 高分辨率，高层下采。
4. **MLP projector 不压 + LLM 扛长上下文**（LLaVA-NeXt）。

> [!warning] token 数 trade-off
> 压狠省 LLM 算力但丢细节（OCR 失效）；不压保细节但 LLM $O(N^2)$ 大。文档 OCR 需细粒度 → 不压或轻压（$r=2$）；视频 / 大场景 → 重压（$r=16$）。


## 4. 数学原理 / 公式

### 4.1 token 数与分辨率

$$N = \frac{HW}{P^2}$$

$H, W$ 图分辨率，$P$ patch 大小。线性于 $HW$。

### 4.2 attention 计算量

ViT 单层 attention（$h$ 头，维 $D$）FLOPs $\propto 2N^2 D + 2 N D^2$。$N$ 大则 $N^2$ 主导：

| $N$ | $N^2$ | 注 |
|---|---|---|
| 196 | 38K | 224×224 |
| 576 | 331K | 336×336 |
| 1024 | 1M | 448×448 |
| 4096 | 16.7M | 1024×1024 |
| 32400 | 1.05B | 4K |

4K 图 ViT attention 1B+ FLOPs/层，30 层 → 30B+，爆。

### 4.3 tile 切分的 token 数

$$N_{total} = \left\lceil \frac{H}{T} \right\rceil \cdot \left\lceil \frac{W}{T} \right\rceil \cdot \left(\frac{T}{P}\right)^2 + N_{thumb}$$

例：$H=W=1024$, $T=448$, $P=14$：
- $N_{tile} = \lceil 1024/448 \rceil^2 = 3^2 = 9$
- $N_{per\_tile} = (448/14)^2 = 32^2 = 1024$
- $N_{total} = 9 \times 1024 + 1024 = 10240$ token

需 resampler 压到几百才能喂 LLM。

### 4.4 resampler 压缩比

$$r = \frac{N_{total}}{N'}$$

例：$N_{total} = 10240$, $N' = 256$ → $r = 40\times$。压 40 倍。

### 4.5 LLM 算力节省

LLM attention $\propto (N' + L_t)^2$。

- 不压（$N'=10240$, $L_t=512$）：$(10752)^2 \approx 115M$
- 压到 256：$(768)^2 \approx 590K$

省 ~200×。resampler 是高分辨率 VLM 的必需品。

### 4.6 pixel shuffle 压缩

把相邻 $k \times k$ patch token 合并：

$$N' = \frac{N}{k^2}$$

例 $k=2$：$N' = N/4$。Qwen-VL 用 pixel shuffle merge 相邻 2×2 patch。无参数压缩（reshape + linear），有损但简单。


## 5. 代码示例（可选）

### 5.1 tile 切分（InternVL 风格）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def split_into_tiles(image, tile_size=448):
    """把高分辨率图切成 (tile_size, tile_size) 的子图列表。"""
    # image: (B, 3, H, W), H, W 任意
    B, C, H, W = image.shape
    tiles = []
    n_h = (H + tile_size - 1) // tile_size  # ceil
    n_w = (W + tile_size - 1) // tile_size
    # pad 到 tile_size 整数倍
    pad_h = n_h * tile_size - H
    pad_w = n_w * tile_size - W
    image = F.pad(image, (0, pad_w, 0, pad_h))
    for i in range(n_h):
        for j in range(n_w):
            tile = image[:, :, i*tile_size:(i+1)*tile_size,
                         j*tile_size:(j+1)*tile_size]
            tiles.append(tile)  # (B, 3, tile_size, tile_size)
    return tiles, n_h, n_w  # 每 tile 独立 encode

# 例: 1024x1024 图 -> 9 个 448 tile (pad 后)
img = torch.randn(1, 3, 1024, 1024)
tiles, n_h, n_w = split_into_tiles(img, 448)
print(len(tiles), tiles[0].shape)  # 9, (1, 3, 448, 448)
# 每 tile 过固定 ViT -> 9 * 1024 = 9216 token -> resampler 压
```

### 5.2 NaViT 风格 patch + packing（概念）

```python
def navit_patch_embed(image, patch_size=16):
    """任意分辨率图 -> 变长 patch token 序列。"""
    # image: (3, H, W), H, W 任意
    C, H, W = image.shape
    n_h, n_w = H // patch_size, W // patch_size
    # 切 patch (用 unfold 等价 Conv2d stride=patch)
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # patches: (C, n_h, n_w, P, P)
    patches = patches.contiguous().view(C, -1, patch_size * patch_size)  # (C, n_h*n_w, P*P)
    patches = patches.transpose(0, 1).transpose(1, 2)  # (n_h*n_w, P*P*C)
    return patches, n_h, n_w  # (N, P*P*C), N = n_h * n_w 变长

# 多张变长图 pack 进固定序列 + block mask
def pack_images(images, max_tokens=1024, patch_size=16):
    """多张变长图 pack 进一个固定长度序列, block-diagonal attention mask。"""
    packed_tokens = []
    mask = []  # block-diagonal: 每图一块
    for img in images:
        tokens, n_h, n_w = navit_patch_embed(img, patch_size)
        packed_tokens.append(tokens)
        mask_block = torch.ones(len(tokens), len(tokens))
        mask.append(mask_block)
    # concat + pad to max_tokens
    packed = torch.cat(packed_tokens, dim=0)  # (sum_N, P*P*C)
    # block-diagonal mask (图 A 不 attend 图 B)
    full_mask = torch.block_diag(*mask)
    # pad 到 max_tokens
    pad = max_tokens - packed.size(0)
    if pad > 0:
        packed = torch.cat([packed, torch.zeros(pad, packed.size(1))], dim=0)
        full_mask = F.pad(full_mask, (0, pad, 0, pad))
    return packed[:max_tokens], full_mask[:max_tokens, :max_tokens]

img1 = torch.randn(3, 224, 224)  # 196 token
img2 = torch.randn(3, 336, 336)  # 441 token
packed, mask = pack_images([img1, img2], max_tokens=1024)
print(packed.shape, mask.shape)  # (1024, 768), (1024, 1024)
```

### 5.3 factorized 2D position embedding

```python
class FactorizedPosEmbed(nn.Module):
    """NaViT 风格 2D 因式分解位置编码: 行 + 列分开。支持任意 H, W。"""
    def __init__(self, max_h=64, max_w=64, dim=768):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(max_h, dim))
        self.col_embed = nn.Parameter(torch.randn(max_w, dim))

    def forward(self, n_h, n_w):
        # n_h, n_w: 当前图的 patch 行列数 (任意)
        r = self.row_embed[:n_h]  # (n_h, dim)
        c = self.col_embed[:n_w]  # (n_w, dim)
        # 广播相加: (n_h, n_w, dim)
        pos = r.unsqueeze(1) + c.unsqueeze(0)  # (n_h, n_w, dim)
        return pos.reshape(n_h * n_w, -1)  # (n_h*n_w, dim)

pe = FactorizedPosEmbed()
pos_224 = pe(14, 14)   # 196 token
pos_336 = pe(21, 21)   # 441 token
pos_arb = pe(13, 17)   # 任意纵横比也行
print(pos_224.shape, pos_arb.shape)  # (196, 768), (221, 768)
```

### 5.4 resampler 压高分辨率 tile token

```python
class Resampler(nn.Module):
    """把高分辨率 tile 产生的大量视觉 token 压成固定少 token。见 [[projector]]。"""
    def __init__(self, vision_dim=1024, llm_dim=4096, num_queries=256, num_layers=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim))
        self.v_proj = nn.Linear(vision_dim, llm_dim)
        layer = nn.TransformerDecoderLayer(llm_dim, 8, llm_dim*4, batch_first=True, activation='gelu')
        self.decoder = nn.TransformerDecoder(layer, num_layers)

    def forward(self, vision_feats):
        # vision_feats: (B, N_total, D_v), N_total 可能上万
        B = vision_feats.size(0)
        q = self.queries.expand(B, -1, -1)
        kv = self.v_proj(vision_feats)
        out = self.decoder(q, kv)  # (B, 256, d) -- 压到 256 固定
        return out

r = Resampler()
big_feats = torch.randn(2, 9216, 1024)   # 9 tile 各 1024 token = 9216
out = r(big_feats)
print(out.shape)  # torch.Size([2, 256, 4096])  -- 压到 256 喂 LLM
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[ViT与vision encoder]]（vision encoder 是 patch 切分 + encode 的主体）、[[自注意力]]（attention $O(N^2)$ 决定 token 预算）、[[位置编码]]（factorized 2D 编码支持任意分辨率）、[[projector]]（resampler 压高分辨率 token）、[[多头注意力]]。
- **下游（应用）**: [[多模态sequence packing]]（变长视觉 token 的 batch packing + mask）、InternVL / Qwen-VL / Qwen2-VL / LLaVA-NeXt（高分辨率策略实现）、DocVQA / ChartQA / OCR 任务（需高清）、[[多模态MoE路由]]（高分辨率下视觉 token 多，MoE 可分流）。
- **对比 / 易混**:
  - **NaViT vs tile 切分**：见 §3.5 对比表。NaViT 原生变长 + packing，tile 复用固定 ViT 切子图。
  - **动态分辨率 vs [[projector]] resampler**：前者解决"看清"（保高清细节），后者解决"省 LLM 算力"（压 token）。两者配合：动态分辨率提清晰度，resampler 压 token 预算。
  - **动态分辨率 vs [[多模态sequence packing]]**：前者是图像侧任意分辨率机制，后者是 batch 内变长序列 packing 机制。NaViT 的 packing 是 sequence packing 的视觉实例。
  - **tile 切分 vs resize**：tile 切分子图分别 encode（保细节），resize 是缩放整图（丢细节）。tile 是"以多换清晰"。
  - **动态分辨率 vs [[新模型接入]]**：推理引擎要支持变长视觉 token + block mask + resampler 的 cross-attn。是接入高分辨率 VLM 的工程考量。


## 7. 常见误区与易错点

> [!warning] 误区 1：动态分辨率就是直接喂大图
> 不简单。大图 token 数千 → attention $O(N^2)$ 爆。需 NaViT packing + resampler 压，或 tile 切分 + resampler。直接喂会爆显存 / 算力。

> [!warning] 误区 2：tile 切分等于 resize
> 不同。tile 切分是"把大图切成多个固定大小子图分别 encode"（每子图保原分辨率细节），resize 是"整图缩放到固定大小"（丢细节）。tile 是"以多换清晰"。

> [!warning] 误区 3：NaViT 不需 attention mask
> 需。多图 pack 进一个序列，不同图 token 不能互相 attend（否则信息串扰）。需 block-diagonal attention mask 隔离。

> [!warning] 误区 4：高分辨率一定更好
> token 数多 → LLM 算力爆 + 训练不稳 + 推理慢。需权衡。多数任务 1024 token 够，OCR 等才需 4096+。

> [!warning] 误区 5：position embedding 可任意长
> ViT 原版 1D 可学习 position embedding 有固定长度上限（如 196）。超长需 factorized 2D（NaViT）或插值。NaViT 用 factorized 2D 支持任意 $H, W$。

> [!warning] 误区 6：resampler 不损信息
> 损。$N' \ll N$ 是有损压缩。OCR / 细粒度任务压狠丢字。需权衡压缩比 $r$。

> [!warning] 误区 7：tile 切分保纵横比
> 每 tile 是固定方形（如 448×448），原纵横比靠 tile 数与 thumbnail 间接保。tile 内 resize 仍可能扭曲局部。NaViT 才真正保任意纵横比。


## 8. 延伸细节

### 8.1 NaViT 的训练数据增强

NaViT 训练时 random resolution（224 / 336 / 448 / ...）+ random aspect ratio（4:3 / 16:9 / 1:1 / ...），模型学多尺度。是"原生任意分辨率"的关键——模型见过各种大小，推理时能泛化。

### 8.2 InternVL 的 dynamic resolution

InternVL 用 InternViT（6B 大 vision encoder）+ tile 切分。根据图分辨率动态决定 tile 数（小图 1 tile，大图 9+ tile），每 tile 448×448 encode，token 数动态。再经 select-layer（取 ViT 倒数第几层）+ projector 喂 LLM。

### 8.3 Qwen2-VL 的 NaViT 风格

Qwen2-VL（Bai et al. 2024）转 NaViT 风格：原生任意分辨率 + factorized 2D 位置编码 + packing + resampler。支持任意纵横比、任意分辨率，处理长截图 / 文档 / 视频。

### 8.4 LLaVA-NeXt 的 tile + 不压

LLaVA-NeXt（OneVision）：tile 切分高分辨率 + MLP projector 不压 token，靠 LLM 大上下文（32K / 128K）扛长视觉序列。保细节但 LLM 算力大。

### 8.5 视频的动态分辨率

视频是"多帧图像"，每帧高分辨率 → token 数 × 帧数 爆。需帧采样（抽关键帧）+ 每帧 resampler + 时序 pooling。是 [[image video audio processor]] 的 video processor 处理对象。

### 8.6 内容来源

整理自 NaViT 论文（Dehghani et al. 2023 "Patch n' Pack: NaViT, Vision Transformer for Any Aspect Ratio and Resolution"）、InternVL / Qwen-VL / Qwen2-VL 技术报告、LLaVA-NeXt 论文、SigLIP2 论文。具体 tile 数 / resampler query 数以原论文为准。

---
相关: [[VLM架构]]、[[ViT与vision encoder]]、[[projector]]、[[多模态sequence packing]]、[[多模态MoE路由]]、[[image video audio processor]]、[[新模型接入]]
