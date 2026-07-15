# Diffusion LLM 训练目标

> **所属章节**: [[Diffusion LLM]]
> **所属模块**: [[19-多模态与新型模型系统]]
> **别名**: 离散扩散语言模型训练目标 / Diffusion Language Model objective / continuous diffusion LM / absorbing state diffusion / score entropy discrete diffusion / SEDD
> **难度**: 高（需懂 [[autoregressive decoding]] + [[采样策略]] + 连续扩散 [[score matching]] + ELBO + 离散马尔可夫链）


## 1. 一句话定义

**Diffusion LLM 训练目标** 是把"文本语言建模"从**自回归逐 token 预测**（[[autoregressive decoding]]，左到右、第 $i$ 个 token 只看前 $i-1$ 个）重构成一个**去噪扩散过程**（denoising diffusion）——给定一段真文本 $x_{1:L}$，先按某个**前向加噪核** $q(x_t \mid x_0)$ 把它逐步"弄脏"（连续版：在 token embedding 上加高斯噪声；离散版：把若干 token 替换成 `[mask]` 或随机 token），再训练一个网络 $\theta$ 去**反向预测**"被弄脏前的干净 token"，从而学到一个从噪声到数据的生成模型 $p_\theta(x_0)$；推理时从纯噪声出发，**多步迭代去噪**生成整段文本（可并行去噪多个位置，非 AR 的串行）。**与 AR 的根本区别**：AR 用链式分解 $p(x_{1:L})=\prod_i p(x_i\mid x_{<i})$，每个 token 只被前面约束、生成顺序固定左到右；Diffusion 用一个**全局去噪目标**（score matching / ELBO / 交叉熵等价形式），每个 token 都被"整段被噪声污染的上下文"双向约束，生成顺序可任意、可并行。当前主流有**三大训练目标路线**：① **continuous diffusion**（把 token embedding 当连续向量跑连续扩散，代表作 DiffuLLaMA/CDCD，arXiv:2410.17891）；② **absorbing state / masked diffusion**（离散吸收态，把 token 替换成 `[mask]` 再预测，代表作 MDLM，arXiv:2406.07524）；③ **SEDD（Score Entropy Discrete Diffusion，离散 score matching）**，把连续空间的 score matching 推广到离散词表，代表作 arXiv:2310.16834。三者训练 loss 数学上都能化简为某种"加权交叉熵"，但噪声类型（连续高斯 / 离散 mask / 离散随机转移）与推导路径不同。是 [[Diffusion LLM]] 范式的核心——训练目标决定了"什么算噪声、怎么去噪、能不能并行"。

> [!note] 三句话定位
> - **是什么**：把语言建模重构成去噪扩散——加噪（mask / 高斯）→ 网络预测干净 token → 学到 $p_\theta(x_0)$。三大路线：continuous（连续高斯）、absorbing（mask 吸收态）、SEDD（离散 score entropy）。
> - **为什么**：AR 逐 token 串行，长序列慢、生成顺序死板、不能 fill-in-the-middle；Diffusion 可并行去噪、双向上下文、可任意顺序生成。
> - **与 [[autoregressive decoding]] 关系**：AR 是 $\prod_i p(x_i\mid x_{<i})$ 的链式分解；Diffusion 是去噪目标（score/ELBO/CE 等价）。前者串行左到右，后者迭代并行去噪。


## 2. 为什么需要它（动机与背景）

### 2.1 AR 的三个痛点

自回归 LM（GPT 系）统治语言建模，但有三个结构性痛点：

1. **串行生成**：$L$ 个 token 要 $L$ 步 forward，[[autoregressive decoding]] 天然串行，长序列慢（推理延迟 $\propto L$）。
2. **生成顺序固定**：左到右，无法做"先写结尾再补开头"的 infilling、无法任意顺序生成。
3. **单向上下文**：第 $i$ 个 token 预测时只看 $x_{<i}$，丢失右侧信息，对完形填空、代码补全等双向任务不优。

### 2.2 扩散在图像上的成功

扩散模型（DDPM/[[score matching]]）在图像生成上大获成功——加噪去噪、迭代 refine、可并行。自然问：**能否把扩散搬到离散文本上**？图像是连续像素值，直接加高斯噪声即可；文本是离散 token，"加噪"没那么自然，催生三大路线。

### 2.3 三大路线的分流

- **continuous**：最直觉——把离散 token 先 embedding 成连续向量，再加高斯噪声跑标准 DDPM，最后 rounding 回 token。问题是 token embedding 空间不是欧氏的，连续去噪与离散词表有 gap。
- **absorbing**：离散友好——用 `[mask]` 当噪声态，token 逐步被 mask 吸收、再 unmask。等价于 BERT 式 masked LM 的扩散化，loss 可化简为加权交叉熵。
- **SEDD**：理论最干净——把连续 score matching（估计 $\nabla_x \log p(x)$）推广到离散，估计"数据分布的比值" $\frac{p(x+\text{一步}}{p(x)}$，构造 score entropy loss。

### 2.4 训练目标决定一切

训练目标（loss）直接决定：噪声类型、能否并行去噪、推理步数、loss 是否等价 CE。所以"训练目标"是 Diffusion LLM 的**核心命题**，比网络结构更关键。


## 3. 核心概念详解

### 3.1 扩散框架的两步：前向加噪 + 反向去噪

```
前向 (加噪, 固定, 不学):
  x_0 (真文本) --q(x_t|x_0)--> x_t (噪声态, t=0..T, t 越大越噪)
  continuous: x_t = sqrt(a_t)*emb(x_0) + sqrt(1-a_t)*eps   (高斯)
  absorbing:  x_t = x_0 若被保留, [mask] 若被吸收
  SEDD:       x_t = 一步随机转移 (token i -> j 按 q)

反向 (去噪, 学 θ):
  x_T (纯噪声) --p_θ(x_{t-1}|x_t)--> ... --> x_0 (生成文本)
  θ 学 "给定噪声态 x_t, 预测干净 x_0 (或 score)"
```

训练只学反向网络 $\theta$；前向加噪核 $q$ 是固定调度（像 DDPM 的 $\beta_t$ schedule）。

### 3.2 三大训练目标对比

| 维度 | continuous diffusion | absorbing state (masked) | SEDD |
|---|---|---|---|
| **代表作** | DiffuLLaMA (2410.17891), CDCD | MDLM (2406.07524), LLaDA | SEDD (2310.16834) |
| **噪声空间** | 连续（embedding 向量 + 高斯） | 离散（`[mask]` 吸收态） | 离散（随机 token 转移） |
| **加噪方式** | $x_t=\sqrt{\bar\alpha_t}\text{emb}(x_0)+\sqrt{1-\bar\alpha_t}\epsilon$ | 按 $t$ 比例随机选位置替成 `[mask]` | 按转移核 $q_{ij}$ 把 token 转移 |
| **网络预测** | 连续 $\hat x_0$ 或 $\epsilon$ 或 score | 各位置被 mask 的干净 token 概率 | 离散 score $s_\theta$（logit 差） |
| **loss 形式** | MSE / score matching | 加权交叉熵 | score entropy（离散 score matching） |
| **能否等价 CE** | 否（连续 loss） | **是**（加权 CE） | 近似 CE（score 形式） |
| **离散-连续 gap** | 有（需 rounding / embedding 对齐） | 无（纯离散） | 无（纯离散） |
| **推理并行** | 可（连续去噪） | 可（并行 unmask） | 可（并行转移） |
| **典型步数** | 数十~百步 | 数十~百步，可蒸馏到少步 | 数十步，可比 AR 少网络评估 |

### 3.3 continuous diffusion（连续扩散 on embedding）

**思路**：token $x_i$ 先过 embedding 表 $E$ 得连续向量 $e_i = E[x_i]$；在连续向量上跑标准 DDPM：

$$x_t = \sqrt{\bar\alpha_t}\,e_i + \sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal N(0,I)$$

网络 $\theta$（Transformer）在 $t$ 时刻输入整段 $x_t$，预测噪声 $\epsilon$ 或干净向量 $\hat e_0$；推理时连续去噪出向量，最后 **rounding**（最近邻 + 温度采样）回 token。

**DiffuLLaMA** 的关键贡献：发现 AR 与 continuous diffusion 目标**数学上可桥接**——AR 的 next-token CE 可视作 diffusion 的某种极限，从而**从 AR checkpoint 持续预训练**（continual pretrain）转成 diffusion，省去从 scratch 训。把 GPT2/LLaMA（127M~7B）转成 DiffuGPT/DiffuLLaMA，<200B token 即可，性能接近 AR 原版。

### 3.4 absorbing state / masked diffusion（离散吸收态）

**思路**：定义一个**吸收态** `[mask]`。前向：每个 token 以速率 $\beta_t$ 被"吸收"成 `[mask]`（一旦吸收不再变回）。$t$ 时刻，被 mask 的位置数 $\propto t$。反向：网络看带 mask 的序列，预测每个 mask 位置的干净 token。

这与 BERT 的 masked LM **形似但不同**：BERT 是固定 15% mask、单步预测；absorbing diffusion 是**连续时间** $t\in[0,1]$ 的 mask 比例调度、多步迭代 unmask，有完整 ELBO。

**MDLM** 的关键：证明在 absorbing 假设下，扩散 ELBO 可化简为**加权交叉熵**（每个位置、每个时间 $t$ 的 CE 加权和），训练极简（就是个加权 BERT loss），但能做生成（多步 unmask）。是当前最强离散扩散 LM 之一。

### 3.5 SEDD（Score Entropy Discrete Diffusion）

**思路**：连续扩散靠 score matching $\nabla_x\log p(x)$；离散空间没有梯度，SEDD 把"score"重定义为**对数比值**：

$$s(x)_i = \log \frac{p(x+\text{token}_i)}{p(x)}$$

即"从当前 token 转移到 token $i$ 的对数概率比"。SEDD 提出 **score entropy** loss，是连续 score matching 在离散空间的自然推广，**可证明是真实 score 的无偏估计**。训练一个网络输出离散 score，推理时按 score 引导离散转移去噪。

**SEDD** 论文（Lou, Meng, Ermon, ICML 2024 Oral）实验：在同等规模下，SEDD 比此前所有扩散 LM 困惑度降 25%~75%，**与 AR（GPT-2）竞争**；不需温度退火就能生成忠实文本（生成困惑度比未退火 GPT-2 好 6~8 倍）；可用 32 倍更少的网络评估达相似质量。

### 3.6 与 AR 的核心对比

| 维度 | AR (自回归) | Diffusion LLM |
|---|---|---|
| **分解** | $\prod_i p(x_i\mid x_{<i})$ 链式 | 去噪目标 $p_\theta(x_0)$ 全局 |
| **上下文** | 单向（左到右 causal mask） | 双向（全序列 attention，见 [[自注意力]]） |
| **生成顺序** | 固定左到右 | 任意（可并行 unmask） |
| **推理步数** | $L$ 步（每步 1 token） | $T$ 步（每步可并行多 token），$T\ll L$ 时快 |
| **单步算力** | 1 token forward | 全序列 forward（贵） |
| **能 fill-in-middle** | 需 prompt 重排 | 天然支持 |
| **KV cache** | 有（[[KV cache]]） | 难直接复用（见 [[Diffusion LLM缓存与调度]]） |
| **loss** | next-token CE | score / 加权 CE / ELBO |


## 4. 数学原理 / 公式

### 4.1 AR 目标回顾

AR LM 最大化链式分解对数似然：

$$\log p_\theta(x_{1:L}) = \sum_{i=1}^L \log p_\theta(x_i \mid x_{<i})$$

每步 CE：$\mathcal L_{\text{AR}} = -\sum_i \log p_\theta(x_i\mid x_{<i})$。生成必须 $i=1,2,\dots,L$ 串行。

### 4.2 连续扩散目标（continuous）

前向（固定调度 $\bar\alpha_t$）：

$$x_t = \sqrt{\bar\alpha_t}\,e_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal N(0,I)$$

去噪 score matching loss（预测 $\epsilon$）：

$$\mathcal L_{\text{cont}} = \mathbb E_{t, x_0, \epsilon}\left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2$$

或预测 $x_0$：

$$\mathcal L_{x_0} = \mathbb E_{t, x_0, \epsilon}\left\| e_0 - \hat e_\theta(x_t, t) \right\|^2$$

最后一步 rounding：$\hat x_i = \arg\max_{j} \langle \hat e_i, E[j]\rangle$（最近邻）+ 温度采样。

> [!warning] 连续 gap
> 连续去噪出的向量 $\hat e$ 不一定落在词表 embedding 的凸包上，rounding 有损。CDCD/DiffuLLaMA 用 embedding 对齐、扩散在"词表 simplex"上等技巧缓解。这是 continuous 路线的固有难点。

### 4.3 吸收态扩散目标（absorbing / MDLM）

定义吸收态 $m$（`[mask]`）。前向转移核（$t\in[0,1]$，$\beta$ 速率）：

$$q(x_t = m \mid x_0 = w) = 1 - e^{-\beta t},\quad q(x_t=w\mid x_0=w)=e^{-\beta t}$$

即 token 以 $\beta t$ 概率被吸收成 mask。扩散 ELBO 可化简（MDLM 关键结论）为：

$$\boxed{\mathcal L_{\text{MDLM}} = -\mathbb E_{t,\, M}\left[ \sum_{i\in M} \log p_\theta(x_i \mid x_{\setminus M}^{\text{masked}}) \right]}$$

其中 $M$ 是被 mask 的位置集合，$x_{\setminus M}^{\text{masked}}$ 是把 $M$ 位置替成 mask 后的序列。**这就是加权 masked LM 的交叉熵**——形式上像 BERT，但 $t$ 连续采样、加权，且推理多步 unmask。训练极简，等价 CE。

### 4.4 SEDD：离散 score 与 score entropy

定义离散 score（对数比值，对非吸收转移族）：

$$s_\theta(x)_i = \log \frac{p_\theta(x \text{ 转移到含 } i)}{p_\theta(x)}$$

SEDD loss（score entropy，是连续 score matching 的离散推广）：

$$\mathcal L_{\text{SEDD}} = \mathbb E_{x\sim\text{转移链}}\left[ \sum_{i} \text{softmax}(s_\theta(x))_i \cdot \left( s_\theta(x)_i - \log \frac{q(x\mid x_0)}{\sum_j q} \right) \right]$$

> [!note] 直觉
> SEDD 让网络学的"score"等于真实数据分布的转移比值 $\log\frac{p(x^+)}{p(x)}$。一旦学准，反向去噪按 score 采样即恢复数据分布。是唯一有**无偏 score 估计**保证的离散扩散方法。

### 4.5 三种 loss 与 CE 的等价关系

- **continuous**：MSE/score matching，**不等价 CE**（连续 loss）。
- **absorbing (MDLM)**：**等价加权 CE**（4.3 推导），训练像 BERT。
- **SEDD**：score entropy，**近似 CE**（score 形式），理论更干净。

### 4.6 算力模型：步数 vs 单步

Diffusion 推理算力 $\approx T_{\text{步数}} \times \text{单步 FLOPs}$。单步是全序列 forward（$O(Ld^2)$），步数 $T$ 通常 $16\sim 256$。总 $\approx T \cdot L d^2$。AR 是 $L$ 步、每步单 token（decode memory-bound），$L \cdot d^2$ 量级但串行。

> [!tip] 何时 Diffusion 更快
> 当 $T \cdot (\text{并行单步}) < L \cdot (\text{串行单步})$，即步数 $T \ll L$ 且单步可并行摊销，Diffusion 总延迟更低。但单步算力更重（全序列），所以 batch 小时 AR 常更快，batch 大时 Diffusion 并行优势显现。详见 [[Diffusion LLM缓存与调度]]。


## 5. 代码示例（可选）

### 5.1 absorbing diffusion 一步训练（概念，等价加权 CE）

```python
import torch, torch.nn.functional as F

def mdlm_loss(model, input_ids, vocab_size, mask_id, pad_id, device):
    """
    MDLM 式 absorbing diffusion 一步 loss (化简为加权 CE).
    input_ids: [B, L] 真 token
    """
    B, L = input_ids.shape
    # 1. 采时间 t in (0,1), 越大 mask 越多
    t = torch.rand(B, device=device)             # [B]
    mask_prob = 1 - torch.exp(-beta * t)         # 吸收概率 (per-token), [B]
    # 2. 随机选位置 mask (per-token Bernoulli)
    mask = (torch.rand(B, L, device=device) < mask_prob[:, None]) & (input_ids != pad_id)
    # 3. 构造噪声态: mask 位置替成 mask_id
    noisy = input_ids.clone()
    noisy[mask] = mask_id
    # 4. 网络预测 (双向 attention, 非 causal)
    logits = model(noisy)                          # [B, L, V]
    # 5. loss 只在 mask 位置算 (加权 CE, 权重 ~ 1/mask_prob 避免 bias)
    weight = (1.0 / mask_prob[:, None]).clamp(max=10.0)
    loss = F.cross_entropy(
        logits[mask].float(), input_ids[mask], reduction='mean'
    )
    return loss
```

### 5.2 continuous diffusion 一步（embedding + 高斯）

```python
def continuous_diffusion_step(model, embed_table, input_ids, alpha_bar_t, device):
    """
    continuous diffusion 一步 loss (预测 x0 向量).
    embed_table: [V, d]  token embedding
    """
    e0 = embed_table[input_ids]                    # [B, L, d] 干净向量
    eps = torch.randn_like(e0)
    xt = e0.sqrt_alpha_bar + eps * (1 - alpha_bar_t).sqrt()   # 加噪
    hat_e0 = model(xt, t)                          # 网络预测干净向量 [B,L,d]
    loss = F.mse_loss(hat_e0, e0)                  # 预测 x0
    return loss

def round_to_token(hat_e0, embed_table, temperature=1.0):
    # 最近邻 + 温度采样回离散 token
    sim = hat_e0 @ embed_table.T / temperature     # [B,L,V]
    return sim.argmax(-1)                          # 或 sample
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[autoregressive decoding]]（AR 是 Diffusion 的对比基准）、[[采样策略]]（去噪采样类似 score-based 采样）、[[score matching]]（continuous/SEDD 的理论基础）、ELBO（变分下界，absorbing/SEDD 推导用）、[[自注意力]]（Diffusion LM 用双向 attention）。
- **下游（应用）**: [[iterative与parallel decoding]]（训练目标决定解码方式）、[[Diffusion LLM缓存与调度]]（双向 attention 使 KV cache 难复用）、LLaDA/Dream/iLLaDA 等开源 dLLM、fill-in-the-middle、可控生成。
- **对比 / 易混**:
  - **Diffusion vs [[autoregressive decoding]]**：链式 vs 去噪；串行单向 vs 并行双向。见 3.6 表。
  - **continuous vs absorbing vs SEDD**：连续高斯 / 离散 mask / 离散 score。见 3.2 表。
  - **absorbing diffusion vs BERT**：形式像 masked LM，但 BERT 单步固定 mask、只做理解；absorbing 是连续时间多步、做生成。
  - **Diffusion 训练 vs [[speculative decoding]]**：前者改训练目标；后者不改训练、是 AR 推理加速（仍保 AR 顺序）。


## 7. 常见误区与易错点

> [!warning] 误区 1：Diffusion LLM 一定比 AR 快
> 不一定。Diffusion 单步是全序列 forward（$O(Ld^2)$）比 AR 单 token 重得多。只有步数 $T\ll L$ 且 batch 大能摊销时才快。小 batch AR 常更快。算力模型见 4.6。

> [!warning] 误区 2：continuous 与 absorbing 的 loss 一样
> 不一样。continuous 是 MSE/score matching（连续 loss，不等价 CE）；absorbing（MDLM）等价加权 CE；SEDD 是 score entropy。三者噪声类型与推导路径不同，见 3.2。

> [!warning] 误区 3：absorbing diffusion 就是 BERT
> 形似不同。BERT 单步固定 15% mask、只做判别理解；absorbing diffusion 是连续时间 $t\in[0,1]$ mask 比例调度、多步迭代 unmask、有完整 ELBO、做生成。loss 形式都是加权 CE，但调度与推理完全不同。

> [!warning] 误区 4：SEDD 和 continuous score matching 等价
> 不等价。连续 score matching 估 $\nabla_x\log p$（梯度，需连续空间）；SEDD 估对数比值 $\log\frac{p(x^+)}{p(x)}$（离散，无梯度）。是推广，但数学对象不同。

> [!warning] 误区 5：Diffusion LM 不需要位置编码
> 错。双向 attention 仍需 [[位置编码]]（[[RoPE]] 等）告诉模型 token 顺序。只是 attention 不 causal。详见 [[Transformer与新型架构适配]]。

> [!tip] 实践：从 AR checkpoint 起步
> DiffuLLaMA（2410.17891）证明可从 AR checkpoint（LLaMA）continual pretrain 转 diffusion，省去 from scratch。若要复现/落地 dLLM，优先此路线而非 scratch。


## 8. 延伸细节

### 8.1 关键论文清单（已核实 arXiv ID）

- **SEDD**: Lou, Meng, Ermon. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*. **arXiv:2310.16834**, ICML 2024 Oral.
- **MDLM**: Sahoo et al. *Simple and Effective Masked Diffusion Language Models*. **arXiv:2406.07524**, NeurIPS 2024.
- **DiffuLLaMA**: Gong et al. (HKUNLP) *Scaling Diffusion Language Models via Adaptation from Autoregressive Models*. **arXiv:2410.17891**, ICLR 2025.
- **LLaDA**: Nie et al. *LLaDA: Large Language Diffusion Models*（masked diffusion 8B，arXiv:2502.09992，待核实具体编号）。
- **Dream**: Dream LLC. diffusion LM（基于 LLaDA 路线）。

### 8.2 continuous 路线的 rounding gap

continuous diffusion 去噪出连续向量，必须 rounding 回 token。rounding 有损（向量不在词表凸包）。CDCD（Continuous Diffusion for Categorical Data）用"在 simplex 上扩散 + categorical 转移"绕开；DiffuLLaMA 用 embedding 对齐 + categorical head。这是 continuous 路线相对离散路线的工程复杂度来源。

### 8.3 SEDD 的无偏性

SEDD 的 score entropy 是连续 score matching 在离散空间的**无偏推广**（论文有定理证明估计的 score 期望等于真实 score）。这是它相对其他"拍脑袋离散化"方法的理论优势——不引入 bias，故能 scale。MDLM 的加权 CE 虽训练简单，但等价的是 ELBO（下界），有 gap。

### 8.4 与 RL 的关系

Diffusion LM 的训练目标本身是 likelihood-based（CE/score/ELBO）。要进一步对齐人类偏好，可在 dLLM 上做 RL（GRPO 等），但因 dLLM 的 marginal 似然难算（多步去噪），RL 设计比 AR 复杂。见 [[VLM RL与多模态reward]] 的思路迁移。

### 8.5 内容来源

整理自 SEDD (arXiv:2310.16834)、MDLM (arXiv:2406.07524)、DiffuLLaMA (arXiv:2410.17891) 论文及 LLaDA/Dream 开源工作。continuous/absorbing/SEDD 三分法是社区共识分类。具体模型编号（LLaDA arXiv:2502.09992）待核实。

---
相关: [[Diffusion LLM]] | [[iterative与parallel decoding]] | [[Diffusion LLM缓存与调度]] | [[autoregressive decoding]] | [[采样策略]] | [[score matching]] | [[自注意力]] | [[位置编码]] | [[speculative decoding]]
