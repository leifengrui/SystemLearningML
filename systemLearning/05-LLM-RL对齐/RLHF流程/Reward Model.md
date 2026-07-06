# Reward Model

> **所属章节**: [[RLHF流程]]
> **所属模块**: [[05-LLM-RL对齐]]
> **难度**: 中等偏难（RLHF 第二阶段，偏好→标量的桥梁）

## 1. 一句话定义

**Reward Model（RM，奖励模型）** 是 RLHF 的**第二阶段**：训练一个**标量打分函数** $r_\phi(\text{prompt},\text{response})\in\mathbb{R}$，让它的分数**反映人类偏好**——给定一个 prompt 与一条 response，输出"这条回答有多好"。训练方法是**从人类偏好对（preference pairs）学习**：数据是 $(x, y_c, y_l)$（$y_c$ chosen 偏好胜出、$y_l$ rejected 落败），用 **Bradley-Terry 模型**建模 $P(y_c\succ y_l|x)=\sigma(r_\phi(x,y_c)-r_\phi(x,y_l))$，损失 $-\log\sigma(r_\phi(x,y_c)-r_\phi(x,y_l))$。RM 训好后作为 [[PPO optimization]] 的 reward 函数，把"人类觉得哪个好"变成可微、可优化的标量信号；也是 [[DPO]] 推导的隐含前提（DPO 证明 RLHF 目标有闭式解，从而绕过显式 RM）。

> [!note] 一句话直觉
> 人类偏好是模糊的"A 比 B 好"，PPO 需要的是"这条回答 0.73 分"的标量。RM 就是把"偏好对"翻译成"打分器"的桥梁：拿成对比较数据训一个网络，让它给 chosen 打高分、rejected 打低分，训完就能对任意新回答打分。PPO 拿这个分数当 reward 去优化 actor。RM 的本质是"把人类偏好蒸馏成一个标量函数"。

## 2. 为什么需要它（动机与背景）

[[SFT]] 之后模型能听指令，但"哪个回答更好"还需对齐：同一 prompt 不同回答，哪个更安全/有用/诚实？直接用 PPO 需要**标量 reward**（策略梯度要 $\hat A$、要 $G_t$，都需 reward 数值）。但人类只能给**相对偏好**（"A 比 B 好"），给不了绝对分数。两个鸿沟：

1. **偏好是相对的，PPO 要绝对的**：人类能比较，但不会给"A=0.83、B=0.41"这种绝对分。需要把相对偏好转成绝对标量。
2. **PPO 要可微、可泛化的 reward**：不能每次 PPO 更新都问人。需要一个学好的函数，对新 response 自动打分。

RM 解决这两点：
- 用 **Bradley-Terry** 把"偏好对"建模成"分数差的 sigmoid"（相对→绝对，BT 假设下分数差决定偏好概率）；
- 训一个**神经网络** $r_\phi$，泛化到未见过的 response，可批量打分；
- PPO 阶段把 $r_\phi(x,y)$ 当 reward 用（[[PPO optimization]] §3.8）。

RM 是 RLHF 的"偏好翻译器"，质量直接决定 RLHF 上限。RM 偏了，PPO 越优化越偏（[[reward hacking]]）。这也是 [[DPO]] 的动机之一——DPO 证明可绕过显式 RM，直接在偏好数据上训，避免 RM 的偏差与工程成本。

## 3. 核心概念详解

### 3.1 偏好数据

数据格式：三元组 $(x, y_c, y_l)$：
- $x$：prompt；
- $y_c$：chosen，人类偏好的（更好的）回答；
- $y_l$：rejected，落败的（更差的）回答；
- 标注：人工对同一 prompt 生成多个回答，两两比较标"哪个更好"（或排序）。

来源：
- **人工标注**：OpenAI/Anthropic 雇人标（贵、慢、但质量高）；
- **AI 标注（RLAIF）**：用更强模型（如 GPT-4）当裁判打偏好（便宜、快、规模化），Constitutional AI 用此；
- **隐式偏好**：从用户反馈（点赞/点踩）、编辑历史（初稿vs改稿）提取。

规模：InstructGPT RM 用 ~33k 偏好对；Llama-2 用更多；现代偏好数据集（UltraFeedback、HH-RLHF）数十万对。

### 3.2 Bradley-Terry 模型（RM 的数学基础）

BT 假设：每个对象有"实力分" $r$，比较胜负概率由分数差决定：

$$
P(y_c\succ y_l\mid x)=\sigma\bigl(r(x,y_c)-r(x,y_l)\bigr)=\frac{1}{1+e^{-(r_c-r_l)}}
$$

- $r_c>r_l$ → $P>0.5$，偏好 $y_c$ 概率高；
- $r_c=r_l$ → $P=0.5$，平手；
- 分数差越大，偏好概率越接近 1。

BT 是 RM 把"相对偏好"变成"绝对分数"的关键假设——它把"哪个好"建模成"分数差的 sigmoid"。**局限**：BT 假设偏好可传递（A>B, B>C → A>C），现实中人类偏好有时不可传递/有噪声。

### 3.3 RM 损失（pairwise ranking loss）

$$
\mathcal{L}_{\text{RM}}(\phi)=-\log\sigma\bigl(r_\phi(x,y_c)-r_\phi(x,y_l)\bigr)
$$

- 最大化 chosen 与 rejected 的分数差（让 $\sigma(r_c-r_l)\to1$）；
- 等价于"chosen 应比 rejected 分高"的 log-loss；
- 梯度：$\nabla\mathcal{L}=-(1-\sigma(r_c-r_l))\cdot(\nabla r_l-\nabla r_c)$，即"分数差不够大时拉大"。

> [!note] 只学相对，不学绝对
> RM 损失只约束 $r_c-r_l$（差），不约束 $r_c$、$r_l$ 的**绝对值**。故 RM 的分数是**相对的**——整体可加任意常数（$r\to r+C$ 不改变损失）。PPO 用 RM 时主要看相对差（advantage 减 baseline 后更只看相对），故绝对值漂移不致命，但需 reward normalization（[[reward normalization]]）稳尺度。

### 3.4 RM 架构

主流做法：**从 SFT 模型初始化**，改造输出头：
- 基座：$\pi_{\text{SFT}}$ 的 transformer body（参数共享/初始化自 SFT）；
- 去掉 unembedding（词表投影头）；
- 加一个 **scalar head**：把最后隐状态 $h_{\text{last}}$（或 pooled）投影到标量 $r=hW_s+b$。

为何从 SFT 初始化：
- SFT 已懂语言+指令，RM 复用其表示，不用从零学语言；
- 实证：从 SFT 初始化的 RM 比从 base 初始化更准、更稳。

变体：
- **PRM（Process Reward Model）**：对推理过程的每步打分（不只看最终结果），数学/代码场景提升大；
- **scalar head 位置**：常取 response 最后 token 的隐状态，或 mean-pool；
- **多 RM 集成**：训多个 RM 取平均/最小，降 reward hacking（[[reward hacking]] §8）。

### 3.5 RM 的"分布外不可信"

RM 只在它见过的偏好数据分布上可靠。若 PPO 把 actor $\pi_\theta$ 训到 RM 训练数据外的区域（OOD），RM 分数不可信 → actor 可能钻 RM 漏洞拿高分但实际差（[[reward hacking]]）。这是 [[KL penalty与KL control]] 把 $\pi_\theta$ 约束在 $\pi_{\text{ref}}$（SFT）附近的根本原因——保 actor 在 RM 可信域内。

### 3.6 PRM vs ORM（过程奖励 vs 结果奖励）

| 类型 | 打分时机 | 优点 | 缺点 |
|---|---|---|---|
| **ORM（Outcome RM）** | 仅最终 response 一个分 | 标注简单、通用 | 信用分配难、稀疏 |
| **PRM（Process RM）** | 推理每步一个分 | 信用分配精细、数学/代码提升大 | 标注贵、需步级标注 |

OpenAI 的 PRM800K（数学推理）证明 PRM > ORM 在数学场景。但 PRM 标注成本高，通用对话仍以 ORM 为主。

### 3.7 RM 与 DPO 的关系

[[DPO]] 的核心洞见：RLHF 目标 $\max_\theta\mathbb{E}[r]-\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$ 的**最优解有闭式** $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$。代入 BT 损失，得 DPO 损失（直接在偏好数据上训 $\pi_\theta$，**不训 RM、不 rollout**）。故 DPO 是"把 RM 隐式进 policy"——绕过显式 RM 的训练与推理成本。但 DPO 不能学 RM 未覆盖的偏好（无在线 rollout），且 RM 在线可不断标新数据更新（PPO+RM 更灵活）。

## 4. 数学原理 / 公式

### 4.1 Bradley-Terry 推导

假设每个 response 有"真实价值" $r^*(x,y)$，人类偏好由 BT 生成：

$$
P(y_c\succ y_l\mid x)=\sigma(r^*(x,y_c)-r^*(x,y_l))
$$

给定偏好数据 $\mathcal{D}=\{(x^{(i)},y_c^{(i)},y_l^{(i)})\}$，最大化对数似然：

$$
\max_\phi\sum_i\log\sigma\bigl(r_\phi(x^{(i)},y_c^{(i)})-r_\phi(x^{(i)},y_l^{(i)})\bigr)
$$

等价于最小化：

$$
\boxed{\mathcal{L}_{\text{RM}}=-\sum_i\log\sigma(r_\phi(x,y_c)-r_\phi(x,y_l))}
$$

### 4.2 梯度

$$
\nabla_\phi\mathcal{L}=-\sum_i\bigl(1-\sigma(\Delta r_i)\bigr)\bigl(\nabla r_\phi(x,y_l)-\nabla r_\phi(x,y_c)\bigr),\quad \Delta r_i=r_\phi(x,y_c)-r_\phi(x,y_l)
$$

- $\sigma(\Delta r)$ 小（分数差不够）→ 权重大，拉大差；
- $\sigma(\Delta r)\to1$（已分对）→ 权重小，不再更新；
- 故 RM 是"自适应聚焦难样本"的 ranking loss。

### 4.3 BT 的局限：偏好不可传递

BT 假设偏好可传递（$A>B,B>C\Rightarrow A>C$）。但人类偏好有时不传递（A 比 B 礼貌、B 比 C 准确、C 比 A 简洁——多维度不可比）。这使 BT 损失在多维偏好上有偏。改进：Elo/MCMC 排序、多 RM 分别管不同维度。这是 RM 误差的根源之一。

### 4.4 RM 分数的尺度不定

$\mathcal{L}_{\text{RM}}$ 只约束 $r_c-r_l$，对 $r\to r+C$ 不变（差不变）。故 RM 分数有"平移自由度"。PPO 用时配合 [[reward normalization]]（除 running std）稳尺度。但不影响相对排序与 advantage（advantage 减 baseline 后平移消去）。

### 4.5 从 RM 到 PPO reward

PPO 阶段，每条 response 的 reward：

$$
R(x,y)=r_\phi(x,y)-\beta\,\text{KL}\bigl(\pi_\theta(\cdot|x)\big\|\pi_{\text{ref}}(\cdot|x)\bigr)
$$

（[[KL penalty与KL control]] §3.3）。然后 $R$ 进 [[GAE]] 倒推 token 级 advantage。RM 是 $R$ 的主体，KL 是约束项。

## 5. 代码示例

```python
import torch, torch.nn as nn, torch.nn.functional as F

# ========= 1. Reward Model: 从 LM body + scalar head =========
class RewardModel(nn.Module):
    def __init__(self, lm_body, hidden):
        super().__init__()
        self.body = lm_body                  # 从 SFT 模型初始化的 transformer body
        self.scalar_head = nn.Linear(hidden, 1)  # 标量头
    def forward(self, input_ids):
        h = self.body(input_ids)             # (B, n, hidden)
        last = h[:, -1, :]                   # 取 response 最后 token 的隐状态
        return self.scalar_head(last).squeeze(-1)  # (B,) 标量 reward

# 极简 LM body(演示)
class LMBody(nn.Module):
    def __init__(s,v,d): super().__init__(); s.emb=nn.Embedding(v,d); s.lstm=nn.LSTM(d,d,batch_first=True)
    def forward(s,x): h,_=s.lstm(s.emb(x)); return h
body=LMBody(50,32); rm=RewardModel(body,32); opt=torch.optim.AdamW(rm.parameters(),1e-5)

# ========= 2. 偏好数据:(prompt+chosen, prompt+rejected) =========
# 每条: (ids_chosen, ids_rejected)
pairs=[(torch.tensor([1,2,3,4,5]), torch.tensor([1,2,3,4,1])),   # chosen 更好
       (torch.tensor([6,7,8,9,10]), torch.tensor([6,7,8,9,2]))]

# ========= 3. RM 训练循环: pairwise ranking loss =========
for ep in range(100):
    for yc, yl in pairs:
        rc = rm(yc.unsqueeze(0))             # chosen 分
        rl = rm(yl.unsqueeze(0))             # rejected 分
        loss = -F.logsigmoid(rc - rl).sum()  # -log σ(r_c - r_l)
        opt.zero_grad(); loss.backward(); opt.step()
    if ep%20==0: print(f"ep{ep} r_c={rc.item():.2f} r_l={rl.item():.2f} loss={loss.item():.3f}")

# ========= 4. 推理:对新 response 打分 =========
new_response = torch.tensor([[1,2,3,4,5]])
print("reward:", rm(new_response).item())
```

> [!tip] RM 工程要点
> - **从 SFT 初始化 body**：`rm.body.load_state_dict(sft_model.transformer.state_dict())`；
> - **scalar head 独立初始化**（小权重），避免初期梯度爆；
> - **数据均衡**：每 prompt 的 chosen/rejected 对数要均衡，否则 RM 学偏；
> - **集成**：训 3~5 个 RM 取 min/mean，降 reward hacking；
> - **监控**：验证集 accuracy（chosen 分 > rejected 分的比例），>65% 算可用。

## 6. 与其他知识点的关系

- **上游（依赖）**: [[SFT]]（RM 从 SFT 初始化）、[[损失函数分类]]（pairwise ranking loss）、Bradley-Terry 模型、偏好数据标注。
- **下游（应用）**: [[PPO optimization]]（RM 当 reward）、[[reward hacking]]（RM 漏洞被钻）、[[KL penalty与KL control]]（KL 把 actor 约束在 RM 可信域）、PRM（过程奖励）。
- **对比 / 易混**:
  - **RM vs [[SFT]]**：SFT 学"生成好回答"，RM 学"判别好回答"（一个生成一个判别）。
  - **RM vs DPO**：RM 显式训打分器再 PPO，DPO 闭式绕过 RM 直接训 policy。见 §3.7。
  - **ORM vs PRM**：见 §3.6，结果分 vs 步级分。
  - **RM vs critic（value network）**：RM 给"回答好坏"的标量 reward，critic 给"当前状态期望 return"的 $V$；PPO 中 RM 是 reward 来源、critic 是 advantage 减项，二者不同。
  - **RM 分数 vs reward normalization**：RM 分数有平移自由度，需 normalization 稳尺度。

## 7. 常见误区与易错点

> [!warning] 误区清单
> 1. **以为 RM 分数是绝对客观分**：RM 只学**相对**（差），绝对值可任意平移；别把"0.8 分"当客观质量，要看相对排序。
> 2. **RM 不从 SFT 初始化**：从 base 初始化的 RM 更难收敛、更不准；从 SFT 初始化复用语言表示。
> 3. **scalar head 取错位置**：应取 response 最后 token（或 pool），不能取 prompt token（那与回答无关）。
> 4. **偏好数据不可传递没处理**：BT 假设传递性，多维度偏好会冲突；用多 RM 或排序标注缓解。
> 5. **RM 过拟合偏好数据**：RM 在小数据上易背具体回答；用验证集 accuracy 早停、加正则。
> 6. **以为训好 RM 就万事大吉**：RM 在 OOD 不可信，PPO 会让 actor 钻 RM 漏洞（[[reward hacking]]）；必须配 KL 约束。
> 7. **RM 与 actor 共享 body 不冻结**：若 RM 与 actor 共享参数且都更新，互相干扰；RLHF 中 RM 训好后冻结，PPO 阶段只用作 reward。
> 8. **reward 不 normalization**：RM 分数尺度随训练漂移，PPO 优势尺度失控；running std 归一化。
> 9. **用 ORM 做数学/代码**：稀疏 reward 信用分配难，数学/代码场景应上 PRM。
> 10. **只训一个 RM**：单 RM 易被钻空子；集成 3~5 个取 min 提鲁棒。

## 8. 延伸细节

### 8.1 RM 的"长度偏置"

RM 常对**长回答**打高分（因训练数据中长回答更详尽→偏好）。PPO 拿这个 RM 优化会让 actor 越生成越长（"length bias"）。缓解：
- 训 RM 时均衡 chosen/rejected 长度；
- reward 显式扣长度惩罚 $r\to r-\alpha\cdot\text{len}$；
- 监控 response length 曲线。

### 8.2 PRM 与数学推理

OpenAI "Let's Verify Step by Step"（2023）：用 PRM800K（80万步级标注）训 PRM，在数学推理上 best-of-N + PRM 显著优于 ORM。PRM 能定位推理错在哪步，可做 process supervision。代价：步级标注成本高。后续 Meta-task 定位错步、自动步级标注是研究热点。

### 8.3 RM 集成与 reward hacking 防御

训多个 RM（不同种子/数据子集），PPO 用 $\min_i r_i$ 或 $\text{mean}-\text{std}$ 当 reward。直觉：单 RM 的漏洞被钻时，其他 RM 可能不认；取 min 让 actor 必须所有 RM 都高分才得分，难钻空子。InstructGPT 用此降 [[reward hacking]]。

### 8.4 RLAIF：AI 替代人工标注

Anthropic Constitutional AI：用 AI（更强模型或宪法规则）生成偏好标注，替代人工。流程：
1. 用 AI 对 response 两两比较打偏好（"哪个更helpful/安全"）；
2. 拿 AI 偏好训 RM；
3. PPO 用该 RM。
成本大降、规模化，但 RM 偏差继承自裁判 AI（"AI 偏好放大"）。详见 [[RLAIF]]（待展开）。

### 8.5 RM 的评估

- **偏好准确率**：验证集上 chosen 分 > rejected 分的比例（>65% 可用，>75% 好）；
- **KL/RM 一致性**：RM 分数与人类排序的 Spearman 相关；
- **在线 A/B**：PPO 后模型 A/B 测，间接验证 RM 质量。

### 8.6 RM 的"奖励弥散"问题

RM 是 ORM（仅末尾打分），PPO 要把它倒推到每个 token（靠 [[GAE]]）。若 RM 分数噪声大，倒推后每个 token 的 advantage 噪声大，训不稳。PRM 给每步直接打分，避免倒推，是数学场景 PRM 更稳的原因之一。

### 8.7 DPO 绕过 RM 的代价

DPO 不训 RM，省工程与算力，但：
- 不能学 RM 训练数据外的偏好（无在线 rollout）；
- 不能用 PRM 的步级信号（DPO 是序列级）；
- 偏好数据外的"探索性"行为学不到。
故在线 RM + PPO 在"持续探索/在线 reward"场景仍不可替代（[[PPO optimization]] §8.4）。

---
相关: [[RLHF流程]]、[[SFT]]、[[PPO optimization]]、[[reward hacking]]、[[KL penalty与KL control]]、[[DPO]]、[[reward normalization]]、[[损失函数分类]]、[[GAE]]、[[value network]]
