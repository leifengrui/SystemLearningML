# mode collapse

> **所属章节**: [[LLM特有RL问题]]
> **所属模块**: [[05-LLM-RL对齐]]
> **难度**: 易（现象直观，防法是采样/RM 设计工程）

## 1. 一句话定义

**mode collapse（模式坍缩）** 是 RLHF 中 LLM 生成的**现象层失败模式**：输出 **mode 单一**——response 模式雷同（同一开头/结构/结尾/措辞），缺乏多样性，无法应对多样输入。它是 [[policy collapse]]（机制层，策略熵塌缩）的下游表现，也可能由 RM 偏好集中 + 数据单一引发。类比 GAN 的 mode collapse（生成器塌缩到少数 mode）。诊断：n-gram 多样性下降、self-BLEU 上升、unique response 比例降。防法分两层：**训练层**（entropy bonus、reward 分散、prompt/数据多样、KL 约束）治本；**推理层**（temperature 升温、nucleus sampling、diversity-promoting beam）治标。

> [!note] 与 [[policy collapse]] 的分工
> - **[[policy collapse]]**：机制层——策略熵塌缩、概率集中，$H(\pi_\theta)\to0$。
> - **mode collapse（本篇）**：现象层——输出 mode 单一、多样性丧失的诊断与防法。
> 本篇讲"塌了怎么救"（采样/RM 工程），[[policy collapse]] 讲"为什么会塌"（PPO 力学）。

## 2. 为什么需要它（动机与背景）

即使 [[policy collapse]] 不严重（策略熵未到 0），RLHF 后 LLM 仍可能 mode collapse：
1. **RM 偏好集中**：RM 训练数据偏好某类 response（如长/礼貌/结构化），actor 学到后所有 prompt 都输出该模式；
2. **SFT 数据单一**：SFT 示范若风格单一，$\pi_{\text{ref}}$ 本身 mode 少，PPO 在其上加剧；
3. **reward over-optimization**：[[reward hacking]] 推 actor 朝单一高 reward mode；
4. **采样温度低**：推理时贪心/低温采样放大策略的 mode 集中。

结果：模型对所有 prompt 输出雷同（"您好，关于您的问题，首先...其次...总结..."），用户体验差、实用性降。这是 RLHF 后常见现象，InstructGPT/Llama-chat 早期均有报告。防法是训练+推理双管齐下。

## 3. 核心概念详解

### 3.1 mode collapse 的典型表现

| 现象 | 示例 |
|---|---|
| **开头雷同** | 所有 response 以"您好"/"当然"/"首先"开头 |
| **结构雷同** | "首先...其次...最后...总结" 模板化 |
| **措辞雷同** | 高频词/套话重复（"综上所述"/"值得注意的是"） |
| **结尾雷同** | "希望对您有帮助" 等固定结尾 |
| **风格单一** | 无论 prompt 风格，输出总一种语气 |

### 3.2 mode collapse 的成因链

```
RM 偏好集中(标记者偏好某类风格) + SFT 数据单一
    ↓
PPO 朝单一高 reward mode 优化
    ↓
policy collapse(策略熵降) → mode collapse(输出单一)
    ↓
推理低温采样放大 → mode collapse 更显
```

### 3.3 训练层防法（治本）

| 防法 | 机制 |
|---|---|
| **entropy bonus** | 保策略熵底（[[entropy bonus]]、[[policy collapse]] §3.5） |
| **reward 分散化** | RM 设计让 reward 不集中单一 mode（如风格多样性奖励） |
| **prompt/数据多样** | rollout prompt 多样、SFT 示范风格多 |
| **KL 约束** | 限制漂离 ref（ref 多 mode），防过度集中 |
| **advantage 归一化** | 防少数样本优势放大 |
| **GRPO** | 组内相对 advantage 天然分散（[[GRPO]]） |

### 3.4 推理层防法（治标）

| 防法 | 机制 |
|---|---|
| **temperature 升温** | $T>1$ 软化分布，增随机性（[[temperature]] 待展开） |
| **nucleus sampling (top-p)** | 从累积概率 $p$ 的核内采样，截断长尾但保多样（[[nucleus sampling]] 待展开） |
| **top-k sampling** | 从 top-k token 采样，类似 |
| **diversity-promoting beam** | beam search 加多样性惩罚（如 diverse beam search） |
| **repetition penalty** | 惩罚已出现 token，防重复 |

> [!tip] 训练 vs 推理
> 训练层防法治本（让 $\pi_\theta$ 本身多 mode），推理层防法治标（从单一 $\pi_\theta$ 采出多样）。生产中两者结合：训练保熵底 + 推理适度升温/nucleus。

### 3.5 mode collapse 的诊断指标

| 指标 | 定义 | collapse 信号 |
|---|---|---|
| **unique response 比例** | 多 prompt 生成的独特 response 占比 | 降 |
| **n-gram 多样性** | unique n-gram / total n-gram | 降 |
| **self-BLEU** | 生成集内部 BLEU（自相似度） | 升 |
| **distinct-n** | 不同 n-gram 数 / 总 token | 降 |
| **token 级 entropy** | $\pi_\theta$ 输出熵 | 降 |

## 4. 数学原理 / 公式

### 4.1 temperature 软化

推理时 $\pi_\theta(a|s)$ 经 temperature $T$ 软化：
$$
\pi_T(a|s)=\frac{\exp(\logit_a/T)}{\sum_{a'}\exp(\logit_{a'}/T)}
$$

- $T\to0$：贪心（mode collapse 最显）；
- $T=1$：原分布；
- $T>1$：软化，增熵抗 collapse。

### 4.2 nucleus sampling

从累积概率 $\le p$ 的最小 token 集采样：
$$
\mathcal{N}_p(s)=\min\{V'\subseteq V:\sum_{a\in V'}\pi_\theta(a|s)\ge p\}
$$

在 $\mathcal{N}_p$ 内按 $\pi_\theta$ 重归一化采样。截断长尾低质 mode，但保 top-p 内多样。$p=0.9$ 常用。

### 4.3 entropy 与 mode 集中

$$
H(\pi_\theta(\cdot|s))=-\sum_a\pi_\theta(a|s)\log\pi_\theta(a|s)
$$

mode collapse 时 $H$ 降（[[policy collapse]] §4.1）。temperature 升温等价于在 softmax 前缩放 logits，使 $H$ 升。

### 4.4 self-BLEU

生成集 $Y=\{y_1,\dots,y_N\}$，self-BLEU = 平均每 $y_i$ 对其余 $Y\setminus\{y_i\}$ 的 BLEU。mode collapse 时生成互似 → self-BLEU 升。

## 5. 代码示例

```python
import torch, torch.nn.functional as F, collections

def generate(model, x, T=1.0, p=0.9, max_len=20):
    """带 temperature + nucleus sampling 的生成"""
    out = x.clone()
    for _ in range(max_len):
        logits = model(out)[:, -1, :] / T              # temperature 软化
        probs = F.softmax(logits, -1)
        # nucleus:取累积概率 ≤ p 的核
        sorted_p, sorted_i = probs.sort(descending=True)
        cum = sorted_p.cumsum(-1)
        mask = cum - sorted_p < p                       # 保留使累积刚超 p 的核
        sorted_p = sorted_p * mask                      # 截断长尾
        sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
        nxt = sorted_i.gather(-1, torch.multinomial(sorted_p, 1))
        out = torch.cat([out, nxt], 1)
    return out

def distinct_n(seqs, n=2):
    """distinct-n:不同 n-gram 数 / 总 n-gram 数"""
    total, uniq = 0, set()
    for s in seqs:
        grams = [tuple(s[i:i+n]) for i in range(len(s)-n+1)]
        total += len(grams); uniq |= set(grams)
    return len(uniq) / max(total, 1)

# 模拟 mode collapse 诊断
class FakeModel:
    """模拟塌缩模型:总输出 token 5"""
    def __call__(self, x):
        B, T_len = x.shape
        logits = torch.zeros(B, T_len, 50)
        logits[..., 5] = 10  # token 5 logits 极高 → collapse
        return logits

m = FakeModel()
x = torch.randint(0, 50, (4, 4))
# 贪心(T→0)生成:全 token 5
seqs_greedy = [generate(m, x[:1], T=0.01, p=1.0, max_len=10)[0].tolist() for _ in range(10)]
# nucleus(T=1.0, p=0.9):稍微多样但仍塌(token 5 占主)
seqs_nucleus = [generate(m, x[:1], T=1.5, p=0.9, max_len=10)[0].tolist() for _ in range(10)]
print(f"贪心 distinct-2: {distinct_n(seqs_greedy):.3f} (低 → collapse)")
print(f"升温 nucleus distinct-2: {distinct_n(seqs_nucleus):.3f} (稍升但仍受策略塌缩限制)")
print("结论:推理升温治标,治本需训练层 entropy bonus/reward 分散")
```

## 6. 与其他知识点的关系

- **上游（依赖）**: [[policy collapse]]（机制层）、[[RLHF (PPO)]]、[[Reward Model]]、[[entropy bonus]]、[[KL penalty与KL control]]。
- **下游（应用）**: 推理采样策略、RLHF 后处理、用户体验工程。
- **对比 / 易混**:
  - **mode collapse vs [[policy collapse]]**：见 §1 + [[policy collapse]] §3.3。现象 vs 机制。
  - **mode collapse vs [[exposure bias]]**：前者是输出 mode 单一（训练后策略问题），后者是训练-推布分布偏移（自回归与 teacher forcing 差异）。不同。
  - **mode collapse vs [[reward hacking]]**：reward hacking 可引发 mode collapse（actor 塌缩到 RM 高分 mode），但前者是多样性层，后者是 proxy 层。
  - **训练层 vs 推理层防法**：前者治本（让 $\pi_\theta$ 多 mode），后者治标（从单一 $\pi_\theta$ 采多样）。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **"升温解决 mode collapse"**：治标不治本。升温从单一 $\pi_\theta$ 采多样，但 $\pi_\theta$ 仍塌缩，升温过高质量降。需训练层治本。
> 2. **"mode collapse = policy collapse"**：本库区分：现象 vs 机制。实践中常混用。
> 3. **"RLHF 后必 mode collapse"**：不一定。好的 RM 设计 + 数据多样 + entropy bonus 可控。
> 4. **"只看生成样本就诊断"**：需量化（distinct-n/self-BLEU/entropy），肉眼易主观。
> 5. **"nucleus sampling 防一切"**：nucleus 截断长尾保 top-p 多样，但若 $\pi_\theta$ top-p 内也单一，nucleus 无效。需 $\pi_\theta$ 本身多 mode。
> 6. **"mode collapse 是 RLHF 独有"**：GAN/VAE/扩散等生成模型都有，是生成模型通病。LLM 中因 RLHF 的 reward 集中而尤甚。
> 7. **"SFT 不影响 mode collapse"**：SFT 数据单一 → $\pi_{\text{ref}}$ mode 少 → PPO 加剧。SFT 多样性是基础。
> 8. **"重复 = mode collapse"**：重复可能是 repetition penalty 不足（采样问题），非策略塌缩。需区分。

## 8. 延伸细节

### 8.1 Llama-chat / InstructGPT 的 mode collapse 案例

早期 InstructGPT/Llama-chat 报告：模型对所有 prompt 输出"首先/其次/总结"结构化套话，是 RM 偏好集中 + SFT 风格单一的 mode collapse。后续改进：RM 标注多样化 + 数据风格多样 + 推理升温，缓解。

### 8.2 RLHF 后的"对齐税"

mode collapse 是"对齐税（alignment tax）"的一种：对齐（RLHF）牺牲了预训练/SFT 的多样性。缓解：
- 用 SFT 当 KL 锚点（保多 mode 基底）；
- RLHF 后做"能力回填"（mix pretrain data）；
- 或 DPO（KL 约束天然保 ref 多 mode）。

### 8.3 多样性奖励

高级防法：在 reward 中显式加多样性项，如 negative self-BLEU、unique n-gram 奖励，鼓励 actor 输出多样。但设计复杂，实践少用。

### 8.4 nucleus vs beam

- **beam search**：最大化概率，倾向高 prob mode，**加剧** mode collapse（贪心式）；
- **nucleus/top-k**：随机采样，保多样，**缓解** mode collapse。
故 RLHF 后推理常采 nucleus，非 beam。

### 8.5 mode collapse 的检测 pipeline

生产中定期诊断：
1. 多 prompt（含多样风格）生成 response；
2. 算 distinct-n/self-BLEU/unique ratio；
3. 与 SFT/pretrained 基线对比；
4. 降 → mode collapse，需调训练（entropy/reward 分散）或推理（升温/nucleus）。

---
相关: [[LLM特有RL问题]]、[[policy collapse]]、[[RLHF (PPO)]]、[[Reward Model]]、[[entropy bonus]]、[[KL penalty与KL control]]、[[exposure bias]]、[[reward hacking]]、[[temperature]]、[[nucleus sampling]]
