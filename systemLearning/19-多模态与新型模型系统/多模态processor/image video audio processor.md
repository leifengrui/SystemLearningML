# image video audio processor

> **所属章节**: [[多模态processor]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: image processor / video processor / audio processor / 多模态 processor / 预处理器 / media processor / 媒体前处理 / token 化 / 时序对齐
> **难度**: 中（需懂 [[ViT与vision encoder]]、[[tokenization]]、[[自注意力]]、[[位置编码]]、[[动态分辨率与视觉token]]）


## 1. 一句话定义

**多模态 processor**（image / video / audio processor）是 VLM 把**原始媒体**（PNG 图、MP4 视频、WAV 音频）转成**模型可吃输入**的前处理模块——**image processor**（resize / normalize / patch 化 → 视觉 token）、**video processor**（抽帧 → 每帧当 image → 加时间 / 位置编码 → 时序 token）、**audio processor**（mel spectrogram / whisper encoder → 音频 token）；核心是**token 化**（图像 → patch token、音频 → 帧 token、视频 → 帧 + 时序）与**时序对齐**（视频 / 音频 token 与文本在时间轴对齐，哪些帧对应哪段文本）；在 VLM 框架中是 HF transformers 的 `Processor` 类（image processor + tokenizer 合一），是 [[ViT与vision encoder]] 前端、LLM 输入构建的起点。

> [!note] 三句话定位
> - **是什么**：image processor（resize/normalize/patch）、video processor（抽帧+时序编码）、audio processor（mel spectrogram/whisper encoder），把原始媒体变模型输入 token。token 化 + 时序对齐是核心。
> - **为什么**：原始 PNG/MP4/WAV 是原始字节，模型要的是数值 tensor + token 序列；视频/音频有额外时间维，需与文本时间轴对齐。
> - **与 [[ViT与vision encoder]] 关系**：image processor 是 vision encoder 的前端（输出 patch token 喂 ViT）；与 [[tokenization]] 类比（文本 tokenize vs 媒体 process）。在 VLM 框架中是 HF `Processor` 类。


## 2. 为什么需要它（动机与背景）

### 2.1 原始媒体是字节，模型要 tensor

VLM 用户给的是 PNG / JPG 图、MP4 视频、WAV 音频——这些是**编码字节**（PNG 压缩、H.264 编码、PCM 采样）。模型不能直接吃字节，要：
1. **解码**：PNG → RGB 像素 array，MP4 → 帧序列，WAV → PCM 采样。
2. **前处理**：resize / normalize / 增强（图）、抽帧 + 时间编码（视频）、mel spectrogram（音频）。
3. **token 化**：转成 token 序列（图像 patch token、音频帧 token、视频帧 + 时序 token）。
4. **对齐**：与文本 token 在序列 / 时间轴上对齐（视频某帧对应文本某段）。

processor 干 1-4，是 raw media → model input 的全流程。

### 2.2 三种模态各有特性

- **图像**：2D 空间。resize + normalize + patch 切分（[[ViT与vision encoder]]）→ $N$ 个 patch token。
- **视频**：2D 空间 × 时间。抽帧（关键帧采样）→ 每帧当 image → 加时间 / 帧位置编码 → $N_{frame} \times N_{patch}$ token + 时序。
- **音频**：1D 时间。mel spectrogram（时频表示）→ whisper encoder / audio encoder → 帧级 token。

每模态前处理不同，processor 要分模态实现。

### 2.3 时序对齐的挑战

视频 / 音频有**时间维**。文本问"第 3 秒说了什么"时，模型要知道哪些 token 是第 3 秒。需：
1. **时间戳**：每帧 / 每 audio chunk 带时间戳。
2. **时序编码**：时间位置 embedding 注入 token。
3. **对齐文本**：视频 token 与文本 token 在序列中对齐（如"第 3 秒"文本 token 后接对应帧视觉 token）。

是 video / audio VLM 比 image VLM 复杂的关键。

### 2.4 VLM 框架的 Processor 类

HF transformers 把 image processor + text tokenizer 合一为 `Processor` 类（如 `LlavaProcessor`、`Qwen2VLProcessor`）。`processor(text=..., images=..., videos=...)` → 返回 `input_ids` + `pixel_values` + `attention_mask`。是 VLM 输入构建的统一入口。


## 3. 核心概念详解

### 3.1 image processor

image processor 把原始图（PIL Image / numpy array）→ ViT 输入 tensor：

1. **resize**：图缩放到 ViT 期望分辨率（如 336×336，固定 ViT；或保原分辨率，NaViT / tile，见 [[动态分辨率与视觉token]]）。
2. **center crop / pad**：非方形图裁剪或填充到方形。
3. **normalize**：减均值除标准差（如 ImageNet mean/std）。
4. **to tensor**：HWC → CHW，$[0, 255] \to [0, 1]$。
5. **patch 切分**：在 ViT 内部做（Conv2d stride=P），不在 processor。

输出 `pixel_values: (B, 3, H, W)` 喂 ViT。

```python
from PIL import Image
import numpy as np

def image_process(img_pil, size=336, mean=(0.48, 0.46, 0.41), std=(0.27, 0.26, 0.28)):
    img = img_pil.convert("RGB").resize((size, size))  # resize
    arr = np.array(img, dtype=np.float32) / 255.0       # [0, 1]
    arr = (arr - np.array(mean)) / np.array(std)        # normalize
    arr = arr.transpose(2, 0, 1)                         # HWC -> CHW
    return arr  # (3, H, W)
```

### 3.2 video processor

video processor 把视频（MP4）→ 帧序列 tensor：

1. **解码**：MP4 → 帧序列（用 ffmpeg / decord / OpenCV）。
2. **抽帧**：均匀采样 $N_{frame}$ 帧（如 8 / 16 / 32 帧，受算力 / 上下文限制）。
3. **每帧 image process**：每帧当独立 image resize + normalize。
4. **时间 / 帧位置编码**：每帧加帧号或时间戳位置 embedding。
5. **token 化**：每帧过 ViT 出 patch token，加时间编码 → 视频视觉 token 序列。

输出 `pixel_values_videos: (B, T, 3, H, W)` 或展平 (B, T*N_patch, D) 喂 video ViT（如 VideoChat / Qwen2-VL 的 video tower）。

> [!note] 视频抽帧 trade-off
> 帧多保细节但 token 爆（$T \times N_{patch}$，如 16 帧 × 576 patch = 9216 token）；帧少省算力但丢动作。主流 VLM 8-32 帧。长视频需分段 + 时序 pooling。

### 3.3 audio processor

audio processor 把音频（WAV）→ 音频 token：

1. **解码**：WAV / MP4 音轨 → PCM 采样（1D 波形）。
2. **重采样**：统一采样率（如 16kHz，whisper 标准）。
3. **mel spectrogram**：PCM → mel 频谱（时频表示，2D：时间 × mel 频率 bin）。是 audio encoder 的标准输入。
4. **audio encoder**：mel 谱过 whisper encoder / audio tower → 帧级 audio token。
5. **时间编码**：每 audio token 带时间戳。

输出 `audio_features: (B, T_audio, D_audio)` 喂 audio encoder。

```python
import librosa
import numpy as np

def audio_process(wav_path, sr=16000, n_mels=80, n_fft=400, hop=160):
    y, _ = librosa.load(wav_path, sr=sr)  # PCM 1D 波形
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )
    mel = librosa.power_to_db(mel)  # log mel
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)  # normalize
    return mel.T  # (T_audio, n_mels) -> 喂 whisper encoder
```

### 3.4 token 化

processor 把媒体变 token：
- **图像**：patch token（[[ViT与vision encoder]]），$N = (H/P)(W/P)$ 个 $D$ 维 token。
- **音频**：帧 token（whisper encoder 输出），$T_{audio}$ 个 $D_{audio}$ 维 token（每 token 对应一小段时间窗，如 25ms）。
- **视频**：帧 + 时序 token，$T \times N_{patch}$ 个视觉 token + 帧号位置编码。

token 化让媒体与文本 token 同构（一串 embedding），可拼接进 LLM。

### 3.5 时序对齐

视频 / 音频 token 要与文本在时间轴对齐。两种方式：

1. **序列拼接对齐**：视频 token 段 + 文本 token 段按时间顺序拼（如 `<video 0-5s> 文本描述 0-5s <video 5-10s> 文本 5-10s`）。位置 / 时间 embedding 标注每段。
2. **时间戳 token**：文本里嵌入时间戳标记（如 `<time 3s>`），模型学会从该标记定位对应帧 token。

主流用拼接对齐 + 时间位置编码。复杂视频问答用时间戳 token。

### 3.6 HF Processor 类

```python
from transformers import LlavaProcessor, Qwen2VLProcessor

processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
# 输入: 文本 + 图
inputs = processor(
    text="<image> Describe this image.",
    images=img_pil,
    return_tensors="pt"
)
# 输出: {input_ids, pixel_values, attention_mask}
# input_ids: (B, L_t) 文本 token (含 <image> 占位)
# pixel_values: (B, 3, 336, 336) 图 tensor
# VLM forward: vision encoder(pixel_values) -> projector -> 替换 <image> -> LLM
```

### 3.7 processor 在 VLM 流水线中的位置

```
原始媒体 (PNG/MP4/WAV) + 文本
        |
   processor (image/video/audio processor + tokenizer)
        |
   input_ids (文本 token, 含 <image>/<video>/<audio> 占位)
   pixel_values (图/视频 tensor)
   audio_features (音频 mel)
        |
   vision encoder -> projector -> 替换占位 -> LLM 输入序列
        |
   LLM -> 输出文本
```

processor 是 raw media → model input 的全流程入口。

### 3.8 三模态 processor 对比

| 维度 | image processor | video processor | audio processor |
|---|---|---|---|
| **输入** | PNG / JPG | MP4 | WAV / MP3 |
| **解码** | PIL / cv2 | ffmpeg / decord | librosa / ffmpeg |
| **核心前处理** | resize / normalize | 抽帧 + 每帧 image process | mel spectrogram |
| **token 化** | patch token (ViT) | 帧 patch + 时序 token | 帧 token (whisper encoder) |
| **时间维** | 无（2D 空间） | 有（时间 × 空间） | 有（时间） |
| **token 数** | $N = (H/P)(W/P)$ | $T \cdot N_{patch}$ | $T_{audio}$ |
| **时序对齐** | 不需 | 需（帧对应文本段） | 需（音频段对应文本段） |
| **算力** | 低 | 高（T 倍图） | 中 |
| **代表** | CLIPImageProcessor | VideoChat / Qwen2VL video | WhisperProcessor |


## 4. 数学原理 / 公式

### 4.1 image token 数

$$N_{image} = \frac{H}{P} \cdot \frac{W}{P}$$

### 4.2 video token 数

抽 $T$ 帧，每帧 $N_{patch} = (H/P)(W/P)$ patch：

$$N_{video} = T \cdot N_{patch}$$

例：8 帧 × 576 patch = 4608 token；16 帧 × 576 = 9216 token。

### 4.3 mel spectrogram 维度

音频 PCM 波形 $y \in \mathbb{R}^{S \cdot D}$（$S$ 采样率，$D$ 秒数），mel 谱：

$$\text{mel}(t, f) = \text{mel\_filterbank}\left(\left|\text{STFT}(y)\right|^2\right)$$

输出 $\text{mel} \in \mathbb{R}^{T_{audio} \times N_{mel}}$，$T_{audio} = \lfloor S \cdot D / \text{hop} \rfloor$，$N_{mel} = 80$（whisper 标准）。每帧对应 $\text{hop} / S$ 秒（如 hop=160, sr=16k → 10ms）。

whisper encoder 把 mel 谱每 2 帧（20ms）压成 1 token → $T_{audio}/2$ 个 audio token。

### 4.4 时序位置编码

视频帧 $t$ 加帧位置 embedding：

$$\text{PE}_{video}(t) = \text{PE}_{time}(t) + \text{PE}_{patch}(r, c)$$

帧号 $t$ + patch 行列 $r, c$。或用 3D 因式分解（时间 + 行 + 列）。

音频帧 $t_a$ 加时间位置 embedding：

$$\text{PE}_{audio}(t_a) = \text{PE}_{time}(t_a)$$

### 4.5 token 预算

VLM 输入序列 $L = N_{text} + N_{image} + N_{video} + N_{audio}$。LLM attention $O(L^2)$。视频 / 音频 token 多易爆：

- 图：576
- 视频 16 帧：9216
- 音频 30s whisper：1500 token

需 [[动态分辨率与视觉token]] 的 tile + resampler，或帧采样减 $T$，或时序 pooling。


## 5. 代码示例（可选）

### 5.1 image processor（HF 风格）

```python
import torch
from PIL import Image
import numpy as np

class SimpleImageProcessor:
    """极简 image processor: resize + normalize + to tensor。"""
    def __init__(self, size=336, mean=(0.481, 0.458, 0.408), std=(0.269, 0.261, 0.276)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: Image.Image):
        img = img.convert("RGB").resize((self.size, self.size))  # resize
        arr = np.array(img, dtype=np.float32) / 255.0             # [0,1]
        arr = (arr - self.mean) / self.std                         # normalize
        arr = arr.transpose(2, 0, 1)                               # HWC -> CHW
        return torch.from_numpy(arr)                               # (3, H, W)

p = SimpleImageProcessor(size=336)
img = Image.new("RGB", (1000, 800), color=(123, 45, 67))
pv = p(img)
print(pv.shape)  # torch.Size([3, 336, 336])
```

### 5.2 video processor（抽帧 + 每帧 image process）

```python
import torch
import numpy as np
from PIL import Image

class SimpleVideoProcessor:
    """极简 video processor: 均匀抽 T 帧, 每帧走 image processor。"""
    def __init__(self, image_processor, num_frames=8):
        self.ip = image_processor
        self.num_frames = num_frames

    def __call__(self, frames):
        # frames: list[PIL.Image], 视频全帧
        total = len(frames)
        # 均匀抽 num_frames 帧
        idx = np.linspace(0, total - 1, self.num_frames).astype(int)
        sampled = [frames[i] for i in idx]
        # 每帧 image process
        tensors = [self.ip(f) for f in sampled]  # 每个 (3, H, W)
        video = torch.stack(tensors)             # (T, 3, H, W)
        return video

vp = SimpleVideoProcessor(SimpleImageProcessor(336), num_frames=8)
frames = [Image.new("RGB", (640, 480), color=(i*10, i*5, 255-i)) for i in range(100)]
v = vp(frames)
print(v.shape)  # torch.Size([8, 3, 336, 336])
# -> video ViT: 每帧 patch + 时序位置编码 -> T*N_patch token
```

### 5.3 audio processor（mel spectrogram）

```python
import numpy as np
import librosa

class SimpleAudioProcessor:
    """极简 audio processor: PCM -> log mel spectrogram。"""
    def __init__(self, sr=16000, n_mels=80, n_fft=400, hop_length=160):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop = hop_length

    def __call__(self, wav_path):
        y, _ = librosa.load(wav_path, sr=self.sr)   # PCM 1D
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop, n_mels=self.n_mels
        )                                              # (n_mels, T_audio)
        mel = librosa.power_to_db(mel)                 # log mel
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)  # normalize
        return mel.T                                   # (T_audio, n_mels) -> whisper encoder

# ap = SimpleAudioProcessor()
# mel = ap("audio.wav")
# -> whisper encoder(mel) -> T_audio/2 audio token
```

### 5.4 时序对齐：视频 token 与文本 token 拼接

```python
def align_video_text_tokens(video_tokens, text_tokens, frame_starts):
    """视频 token 与文本按时间轴拼接对齐。"""
    # video_tokens: (T, N_patch, D), T 帧
    # text_tokens: (L_text, D), 分段 (按 frame_starts 切)
    # frame_starts: 每帧的时间戳 (秒)
    aligned = []
    for t in range(video_tokens.size(0)):
        frame_tokens = video_tokens[t]  # (N_patch, D)
        # 加时间位置编码
        time_emb = time_position_embedding(frame_starts[t], D=frame_tokens.size(-1))
        aligned.append(frame_tokens + time_emb)
        # 该时间段的文本 token 接在帧 token 后
        if t < len(text_tokens):
            aligned.append(text_tokens[t])
    return torch.cat(aligned, dim=0)  # (T*(N_patch + text_per_frame), D)
```

### 5.5 完整多模态 VLM processor 调用（HF 风格）

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# 图 + 文
inputs = processor(
    text="<|im_start|>user <image> Describe.<|im_end|>",
    images=img_pil,
    return_tensors="pt"
)
# 输出: {input_ids, pixel_values, image_grid_thw, attention_mask}
# pixel_values: (1, 3, H, W), H/W 由动态分辨率决定
# image_grid_thw: tile 网格信息 (Qwen2-VL NaViT 风格)
# -> Qwen2VL: vision encoder(pixel_values) -> projector -> 替换 <image> -> LLM

# 视频
inputs = processor(
    text="<|im_start|>user <video> Summarize.<|im_end|>",
    videos=video_frames_list,
    return_tensors="pt"
)
# pixel_values_videos: (1, T, 3, H, W)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[ViT与vision encoder]]（image processor 是其前端，输出 patch token）、[[tokenization]]（文本 tokenize，processor 把媒体 tokenize 与文本 tokenize 合一）、[[位置编码]]（时间 / 空间位置 embedding）、[[自注意力]]（token 序列下游）、[[动态分辨率与视觉token]]（image processor 的 resize / tile / NaViT 策略）。
- **下游（应用）**: [[多模态sequence packing]]（processor 输出变长多模态 token 需 packing）、[[多模态batching]]（processor 输出 batch）、LLaVA / Qwen2-VL / VideoChat / PaliGemma / Whisper-VLM、[[新模型接入]]（推理引擎要支持各模态 processor 的输出格式）、[[VLM RL与多模态reward]]（RL 训练时 rollout 用 processor 处理多模态输入）。
- **对比 / 易混**:
  - **processor vs tokenizer**：tokenizer 是文本 tokenize（子词 → token id），processor 是多模态前处理（媒体 → tensor + 占位 token）。HF 的 `Processor` = `image_processor + tokenizer` 合一。
  - **image processor vs video processor**：前者 2D 空间，后者 3D（+时间），需抽帧 + 时序编码。
  - **audio processor vs [[tokenization]]**：音频先 mel spectrogram 再 audio encoder（whisper）出 token，是"两步 tokenize"；文本是一步子词 tokenize。
  - **processor vs [[ViT与vision encoder]]**：processor 是"raw media → tensor"，vision encoder 是"tensor → 视觉 token"。前者是 ViT 的前端。


## 7. 常见误区与易错点

> [!warning] 误区 1：processor 就是 tokenizer
> 不全是。tokenizer 是文本子词切分，processor 是多模态前处理（含 image / video / audio processor）。HF 的 `Processor` 是 image processor + tokenizer 合一。两者不同概念。

> [!warning] 误区 2：image processor 切 patch
> 不切。image processor 做 resize / normalize / to tensor，输出 `pixel_values: (B, 3, H, W)`。patch 切分在 [[ViT与vision encoder]] 内部（Conv2d stride=P）。

> [!warning] 误区 3：视频就是多张图
> 不只。视频有**时序**——帧之间有动作连续性，需时间 / 帧位置编码注入。不能把帧当独立 image 处理丢时序。

> [!warning] 误区 4：音频 tokenize = 文本 tokenize
> 不同。文本是一步子词 tokenize（BPE / WordPiece）。音频是 PCM → mel spectrogram → audio encoder（whisper）出 token，是"两步 tokenize"。

> [!warning] 误区 5：时序对齐可有可无
> 关键。视频 / 音频问答（"第 3 秒说什么"）需时序对齐——时间戳 token 或拼接对齐。不对齐模型不知哪帧对应哪段文本。

> [!warning] 误区 6：抽帧越多越好
> 帧多保细节但 token 爆（$T \times N_{patch}$）。主流 8-32 帧。长视频需分段 + 时序 pooling + 关键帧采样。

> [!warning] 误区 7：normalize 用 ImageNet 均值
> 不一定。不同 vision encoder 有自己的 normalize 均值 / 标准差（CLIP 用 (0.481, 0.458, 0.408) / (0.269, 0.261, 0.276)，SigLIP 不同）。用错会偏分布。需按模型 card 用。


## 8. 延伸细节

### 8.1 HF Processor 的设计

HF transformers 把 image processor + tokenizer 合一为 `Processor` 类。`__call__` 接 `text` + `images` + `videos` + `audio`，返回 `input_ids` + `pixel_values` + `pixel_values_videos` + `audio_features` + `attention_mask`。占位 token（`<image>` / `<video>` / `<audio>`）在 `input_ids` 中标记位置，VLM forward 时用 vision encoder 输出替换占位。

### 8.2 Qwen2-VL 的 NaViT processor

Qwen2-VL processor 支持任意分辨率 + 任意纵横比：image processor 不 resize 到固定，保原分辨率，输出 `pixel_values: (B, 3, H, W)` + `image_grid_thw`（tile / patch 网格信息）。vision encoder 用 NaViT 处理。

### 8.3 whisper 的 audio encoder

Whisper（Radford et al. 2022）：PCM → 80 mel bin spectrogram → encoder（Transformer，每 2 帧压 1 token，30s 音频 → 1500 token）。是 audio VLM（如 Whisper + LLM 转 audio caption）的标准 audio encoder。

### 8.4 VideoChat 的视频处理

VideoChat / Qwen2-VL video processor：抽 8-32 帧 + 每帧 image process + 时序位置编码 + （可选）每帧 resampler 压 token。长视频（几分钟）需分段 + 关键帧采样 + 时序 pooling。

### 8.5 多模态 sequence packing 与 processor

processor 输出变长多模态 token（不同图分辨率不同 token 数），batching 需 [[多模态sequence packing]]——变长 token pack 进固定 batch，attention mask 隔离。是 processor 输出 → batch 的桥梁。

### 8.6 内容来源

整理自 HF transformers `image_processing` / `video_processing` / `audio_processing` 模块源码、whisper 论文、CLIP / SigLIP / Qwen2-VL / VideoChat 技术报告、librosa 文档。具体 normalize 均值 / mel 配置以模型 card 为准。

---
相关: [[多模态processor]]、[[ViT与vision encoder]]、[[tokenization]]、[[动态分辨率与视觉token]]、[[多模态sequence packing]]、[[多模态batching]]、[[新模型接入]]、[[VLM RL与多模态reward]]
