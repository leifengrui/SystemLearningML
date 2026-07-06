# SFT

> **所属章节**: [[RLHF流程]]
> **所属模块**: [[05-LLM-RL对齐]]
> **难度**: 中等（RLHF 的起点，工程成熟但细节多）

## 1. 一句话定义

**SFT（Supervised Fine-Tuning，监督微调）** 是 RLHF 的**第一阶段**：拿一个**预训练好的基座语言模型**（base LLM，只会"续写"），用**人工编写的"指令-回答"示范数据** $(x, y)$ 做监督微调，让它学会**按指令做事**（回答问题、遵循格式、对话），得到一个能对话/服从的模型 $\pi_{\text{SFT}}$——它同时充当后续 [[PPO optimization]] 与 [[DPO]] 的 **reference model（参考策略 $\pi_{\text{ref}}$）** 与 RLHF 的起点。SFT 的损失就是语言建模的**交叉熵**（teacher forcing 下对示范回答 $y$ 的 next-token 负对数似然），但数据从"网页文本"换成"高质量指令-回答对"。

> [!note] 一句话直觉
> 预训练模型像"读完了整个互联网的文科生"，知识满脑但不会"答题"——你问它"写首诗"，它可能续写成"写首诗的人叫李白…"。SFT 就是教它"听到指令就按指令回答"，从"续写机器"变成"听话助手"。SFT 之后才有 [[Reward Model]] 打分、[[PPO optimization]] 强化，故 SFT 是 RLHF 的地基。

## 2. 为什么需要它（动机与背景）

预训练（pretraining）用海量网页文本做 next-token 预测，得到的是**续写模型**：给定前缀，续写最可能的下文。这有两个问题：

1. **不会"遵循指令"**：prompt "请把这句话翻译成英文：你好" → 续写模型可能输出"请把这句话翻译成英文：你好吗"（续写更多同类问题），而非真的翻译。它没学过"问→答"的对话范式。
2. **格式与风格不可控**：续写模型输出随机网页风格，不像助手（不安全、不简明、不服从格式）。

SFT 用**人工示范**直接教模型"指令→回答"的映射：

- 数据 $(x, y)$：$x$ 是用户指令（"写一首关于秋天的诗"），$y$ 是高质量示范回答；
- 损失：$\mathcal{L}=-\sum_t\log\pi_\theta(y_t|x,y_{<t})$（对 $y$ 的 next-token 交叉熵，teacher forcing）；
- 训完得到 $\pi_{\text{SFT}}$，能听指令做事。

SFT 在 RLHF 中的三重角色：
1. **RLHF 的第一阶段**：经典三阶段 SFT → RM → PPO 的起点（[[PPO optimization]]）；
2. **reference model $\pi_{\text{ref}}$ 的来源**：RLHF 的 KL 惩罚把 $\pi_\theta$ 约束在 $\pi_{\text{ref}}=\pi_{\text{SFT}}$ 附近保语言能力（[[KL penalty与KL control]] §3.3）；
3. **DPO/IPO 的起点与参考**：DPO 直接在 SFT 模型上用偏好数据微调，$\pi_{\text{ref}}$ 就是 SFT（[[DPO]]）。

故 SFT 不只是"训个能用的模型"，它**定义了 RLHF 的锚点**——后续所有对齐都围绕"SFT 模型的语言能力"做约束，防止跑飞。

> [!warning] SFT 不是"训练一个全新模型"
> SFT 是**微调**（fine-tune）一个**已预训练**的基座模型，不是从零训。基座已具备语言能力与知识，SFT 只是教"怎么用这些能力去服从指令"。从零训叫 pretraining，不是 SFT。

## 3. 核心概念详解

### 3.1 SFT 的损失（与预训练同形，数据不同）

$$
\mathcal{L}_{\text{SFT}}(\theta)=-\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t})
$$

- $\pi_\theta(y_t|x,y_{<t})$：模型在 prompt $x$ + 已生成前缀 $y_{<t}$ 下，对下一 token $y_t$ 的 softmax 概率；
- **teacher forcing**：训练时 $y_{<t}$ 用**真实**前缀（不是模型自己生成的），即"老师带着走"；
- 与预训练损失**形式相同**，区别只在数据：预训练用网页 $(x,y)$（$x$ 是前文、$y$ 是续写），SFT 用指令-回答 $(x,y)$（$x$ 是指令、$y$ 是示范回答）。
- 只对**回答 $y$ 算 loss**，不对 prompt $x$ 算（prompt 是输入，不是要学的目标）。

### 3.2 数据格式（三种主流）

| 格式 | 结构 | 代表 |
|---|---|---|
| **Alpaca** | `{"instruction":..., "input":..., "output":...}` 拼成模板 | Alpaca/_SELF-INGEST |
| **ShareGPT** | 多轮对话 `[{role:user,content:...},{role:assistant,content:...}]` | Vicuna/ShareGPT |
| **OpenAI ChatML** | `<\|im_start\|>user\n...<\|im_end\|>` 特殊 token 分隔 | 现代聊天模型 |

模板示例（Alpaca 风格）：
```
Below is an instruction... Instruction: {x} Response: {y}
```
模型学的是"看到 Instruction 就开始生成 Response"。模板设计影响泛化（不同模板同模型效果差异大）。

### 3.3 SFT 与预训练的对比

| 维度 | 预训练 | SFT |
|---|---|---|
| 目标 | 学语言+知识（next-token on web） | 学服从指令（next-token on 示范） |
| 数据量 | 万亿 token | 几万~几十万条指令 |
| 学习率 | 大（如 1e-4 量级） | 小（1e-5~2e-5，比预训练小一个量级） |
| epoch | ~1 | 1~3（多了过拟合） |
| 损失 | 交叉熵 | 交叉熵（同形） |
| 结果 | 续写模型 base | 助手模型 $\pi_{\text{SFT}}$ |

> [!tip] SFT 调参要点
> - **学习率要小**（1e-5~2e-5）：基座已收敛，大 lr 会破坏预训练知识（"灾难性遗忘"）；
> - **epoch 少**（1~3）：SFT 数据量小，多 epoch 易过拟合（模型背答案、泛化差）；
> - **只对回答算 loss**：prompt 部分 mask 掉，否则模型学着"生成 prompt"无意义；
> - **用基座模型初始化**：不要用已 SFT 的模型再 SFT（除非多阶段）。

### 3.4 SFT 的"对齐"意义

SFT 本身就是一种**对齐**（alignment）：把续写模型对齐成"有用的助手"。它解决"听指令"与"格式风格"，但**不解决价值观/偏好**：
- SFT 教"怎么回答"（格式、服从），但回答可能不安全/不诚实/有偏见；
- 这正是 [[Reward Model]] + [[PPO optimization]] 第二步要做的：用人类偏好信号进一步对齐"什么样的回答更好"。

故 RLHF = SFT（教服从）+ RM+PPO（教偏好）。SFT-only 也能用（很多开源模型只 SFT），但偏好对齐效果弱于 RLHF/DPO。

### 3.5 SFT 作为 reference model

在 [[PPO optimization]] 里，$\pi_{\text{ref}}=\pi_{\text{SFT}}$ 全程冻结，reward 里加 $-\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$（[[KL penalty与KL control]] §3.3）。意义：
- $\pi_{\text{SFT}}$ 是"已学会语言能力"的锚点；
- KL 惩罚防止 PPO 把 $\pi_\theta$ 训得偏离语言分布（输出乱码、钻 RM 空子）；
- $\beta$ 控制"保 SFT 人格的力度"。

故 SFT 的质量直接决定 RLHF 上限——SFT 不好，RM/PPO 都难补。

### 3.6 SFT 的变体

- **Instruction tuning**：SFT 的同义词，强调用指令数据；
- **Chain-of-Thought SFT (CoT-SFT)**：示范含推理过程，教模型"想一步答一步"；
- **Constitutional AI (CAI) / RLAIF**：用 AI（而非人工）生成 SFT 数据或偏好，降低人工成本（[[RLAIF]] 待展开）；
- **多阶段 SFT**：先通用 SFT，再领域 SFT（代码/数学），分层对齐。

## 4. 数学原理 / 公式

### 4.1 SFT 损失推导

给定数据集 $\mathcal{D}=\{(x^{(i)},y^{(i)})\}$，最大化示范的对数似然：

$$
\max_\theta\sum_{(x,y)\in\mathcal{D}}\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t})
$$

等价于最小化负对数似然（交叉熵）：

$$
\mathcal{L}_{\text{SFT}}=-\sum_{(x,y)}\sum_t\log\pi_\theta(y_t\mid x,y_{<t})
$$

- $\pi_\theta(y_t|x,y_{<t})=\text{softmax}(z_\theta(x,y_{<t}))_{y_t}$，$z$ 是模型 logits；
- 梯度 $\nabla_\theta\mathcal{L}=-\sum_t\nabla\log\pi_\theta(\cdot)$，正是 [[REINFORCE]] §3.4 所说的"与监督学习交叉熵同构"——SFT 就是权重恒为 1 的策略梯度。

### 4.2 teacher forcing 与 exposure bias

训练用真实前缀 $y_{<t}$，推理用模型自生成前缀 $\hat y_{<t}$。两者分布不同 → **exposure bias**：模型没学过"自己出错后怎么接着生成"，推理一旦某 token 偏，后续雪崩。详见 [[exposure bias]]。

### 4.3 SFT 与 REINFORCE 的同构

[[REINFORCE]] 策略梯度 $\nabla J=\mathbb{E}[\nabla\log\pi\cdot G_t]$，SFT 梯度 $\nabla\mathcal{L}=-\nabla\log\pi$（权重 1）。可看作"reward 恒 1、只对示范 token"的策略梯度——SFT 是 RLHF 梯度框架的"退化版"，故 LLM-RL 训练框架能直接复用 SFT 的工程（前向、logp、反传）。

### 4.4 灾难性遗忘的形式化

SFT 损失只覆盖指令分布 $\mathcal{D}_{\text{instr}}$，预训练分布 $\mathcal{D}_{\text{web}}$ 上没梯度 → 模型在 $\mathcal{D}_{\text{web}}$ 上的能力退化（"遗忘"）。缓解：小 lr、少 epoch、混预训练数据（REGMIX）、参数高效微调（LoRA）少动原权重。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ========= 1. SFT 数据:指令-回答对,拼成模板 =========
PROMPT_TEMPLATE = (
    "Below is an instruction... \n\n### Instruction:\n{x}\n\n### Response:\n"
)
class SFTDataset(Dataset):
    def __init__(self, pairs, tok, max_len=512):
        self.data=[]; self.tok=tok
        for x,y in pairs:
            ids = tok(PROMPT_TEMPLATE.format(x=x)+y, return_tensors='pt').input_ids[0]
            if len(ids) <= max_len:
                self.data.append(ids)
    def __len__(s): return len(s.data)
    def __getitem__(s,i): return s.data[i]

# 假 tokenizer(词表小,演示)
class Tok:
    def __init__(s): s.vocab={c:i for i,c in enumerate("你好吗写一首关于秋天诗是的 <>,.:abcdefghijklmnopqrstuvwxyz");}
    def __call__(s,t,return_tensors=None):
        ids=[s.vocab[c] for c in t if c in s.vocab]
        import torch as _t
        return type('R',(),{'input_ids':_t.tensor([ids])})()
tok=Tok()
pairs=[("写一首关于秋天的诗","秋天来了叶子黄了"),
       ("你好吗","我很好谢谢")]
ds=SFTDataset(pairs,tok); dl=DataLoader(ds,batch_size=2,shuffle=True,collate_fn=lambda b:torch.nn.utils.rnn.pad_sequence(b,batch_first=True))

# ========= 2. 极简语言模型(仅演示 SFT 损失) =========
class LM(nn.Module):
    def __init__(s,v,d): super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True); s.head=nn.Linear(d,v)
    def forward(s,x): h,_=s.lstm(s.emb(x)); return s.head(h)   # logits (B,n,v)
lm=LM(len(tok.vocab),32); opt=torch.optim.AdamW(lm.parameters(),2e-5)   # 小 lr(基座已收敛)

# ========= 3. SFT 训练循环:只对回答部分算 loss =========
def sft_loss(logits, ids, response_start):
    # logits/ids: (B,n,v)/(B,n); response_start: 回答起始 token 位置
    B,n,_=logits.shape
    tgt = ids[:,1:]                      # 预测下一个 token
    lp = F.log_softmax(logits[:,:-1],-1)
    mask=torch.zeros_like(tgt)
    for b in range(B): mask[b, response_start[b]:] = 1   # 只对回答段算 loss
    nll = -lp.gather(-1,tgt.unsqueeze(-1)).squeeze(-1)   # (B,n-1)
    return (nll*mask).sum()/mask.sum().clamp(min=1)

for ids in dl:
    logits=lm(ids)
    # 假设回答从第 10 个 token 开始(实际按模板定位 Response 标记)
    starts=[10]*ids.size(0)
    loss=sft_loss(logits,ids,starts)
    opt.zero_grad(); loss.backward(); opt.step()
print("SFT loss:", loss.item())
```

> [!tip] 实际 SFT 工程
> - 用 `transformers` 的 `AutoModelForCausalLM` + `Trainer`，数据用 `datasets`，模板用 `apply_chat_template`；
> - **DataCollator** 自动 pad + 只对回答算 loss（`labels` 里 prompt 位置置 -100）；
> - 2~3 epoch，lr 1e-5~2e-5，cosine decay + warmup；
> - 大模型用 LoRA/QLoRA 省显存（只训少量参数，少动基座）。

## 6. 与其他知识点的关系

- **上游（依赖）**: 预训练（基座模型来源）、[[nn.Module]]/[[forward与backward]]（模型与训练框架）、交叉熵/[[损失函数分类]]、teacher forcing。
- **下游（应用）**: [[Reward Model]]（RM 常从 SFT 模型初始化）、[[PPO optimization]]（$\pi_{\text{ref}}=\pi_{\text{SFT}}$）、[[KL penalty与KL control]]（KL 约束到 SFT）、[[DPO]]/[[IPO]]（以 SFT 为 $\pi_{\text{ref}}$ 与起点）、[[exposure bias]]（SFT 的 teacher forcing 副作用）。
- **对比 / 易混**:
  - **SFT vs 预训练**：前者微调教服从指令，后者从零学语言；损失同形，数据/学习率/epoch 不同。
  - **SFT vs RLHF (PPO)**：SFT 用示范教"怎么答"，PPO 用 reward 教"哪个答更好"；SFT 是 PPO 的起点与锚点。
  - **SFT vs DPO**：DPO 不需 RM/rollout，直接在 SFT 上用偏好数据微调；SFT 是 DPO 的 $\pi_{\text{ref}}$。
  - **SFT vs instruction tuning**：同义，instruction tuning 强调"用指令数据"。
  - **SFT vs few-shot prompting**：前者改模型权重，后者只在 prompt 里给示例不改权重。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 SFT 是从零训**：SFT 是微调已预训练的基座，不是 pretraining。从零训要万亿 token，SFT 只需几万~几十万条。
> 2. **学习率用预训练的**：基座已收敛，大 lr（如 1e-4）会"灾难性遗忘"预训练知识；SFT 用 1e-5~2e-5。
> 3. **epoch 训太多**：SFT 数据小，3 epoch 以上易过拟合（背答案、泛化崩）；监控验证集 loss 早停。
> 4. **对 prompt 也算 loss**：prompt 是输入不是目标，应 mask（`labels` 置 -100）；算了会让模型学"生成 prompt"，无意义且损质量。
> 5. **不用基座而用已 SFT 模型再 SFT**：除非多阶段设计，否则从 base 模型 SFT，避免累积偏移。
> 6. **数据质量差**：SFT 对数据质量极敏感（"garbage in garbage out"），低质示范会教坏模型；宁少而精。
> 7. **模板设计随意**：不同 prompt 模板效果差异大，要统一、可泛化、含清晰的角色分隔符。
> 8. **以为 SFT 解决偏好对齐**：SFT 只教服从与格式，不解决"哪个回答更好/更安全"；要 RM+PPO 或 DPO。
> 9. **不保留 SFT 模型当 reference**：RLHF 时 $\pi_{\text{ref}}$ 要冻结的 SFT 模型，别拿 PPO 中的 actor 当 ref。

## 8. 延伸细节

### 8.1 SFT 数据规模与质量

- 经典 InstructGPT：~13k 条 SFT 数据；
- Alpaca：52k 条（SELF-INGEST 用 text-davinci-003 生成）；
- LLaMA-2 chat：SFT 数据远多于 InstructGPT，质量严控；
- 共识：**数据质量 > 数量**，几千条高质量 > 几万条低质。

### 8.2 LoRA / QLoRA 微调

全量 SFT 要更新所有参数，大模型显存吃紧。LoRA 冻结原权重，只训低秩增量 $W+\Delta W=W+BA$（$B,A$ 小矩阵），显存大降、质量接近全量。QLoARA 进一步把基座 4bit 量化。是开源社区 SFT 主流。但注意：DPO/PPO 的 reference model 仍需完整权重算 KL，LoRA 增量要合并或单独存。

### 8.3 SFT 之后是否还要 RLHF

- 只 SFT：很多开源模型（Vicuna 早期、Alpaca）只 SFT 即可用，效果已不错；
- SFT + RLHF/DPO：进一步偏好对齐，安全性/有用性/诚实性更好（Llama-2-chat、InstructGPT）；
- 趋势：SFT 打底 + DPO/RLHF 精修，是现代对齐标配。

### 8.4 SFT 的评估

- 自动：验证集 perplexity、MMLU/Benchmark 准确率；
- 人工：帮 assist（helpful）、安全（safe）、诚实（honest）的人工评分；
- 注意：SFT 易在 benchmark 上过拟合（背答案），要 unseen 测试集。

### 8.5 SFT 与 RLAIF / Constitutional AI

Anthropic 的 CAI/RLAIF：用 AI（更强模型）生成 SFT 数据或偏好标注，降低人工成本。SFT 数据可由 AI 生成（"self-instruct""_backtranslation_"），质量接近人工但需过滤。这是 SFT 数据获取的规模化路径。

### 8.6 SFT 的"指令泛化"

好的 SFT 不仅学会具体示范，还**泛化到未见过的指令**（"instruction generalization"）。关键：数据多样性（覆盖多种任务类型）+ 模板多样性 + 适度规模。单一任务 SFT 会过拟合到该任务。

---
相关: [[RLHF流程]]、[[Reward Model]]、[[PPO optimization]]、[[KL penalty与KL control]]、[[DPO]]、[[IPO]]、[[exposure bias]]、[[nn.Module]]、[[损失函数分类]]、[[REINFORCE]]
