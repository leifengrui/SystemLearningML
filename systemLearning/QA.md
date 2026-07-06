# QA 索引

> 所有用户批注（笔记中以特定标记起首的行）的问题与答案落点。`[[双链]]` 点进去即看答案。
> 维护规则见 [[AGENTS]] 第九节：每答完一条用户批注立即在此追加。

---

## 一、ML/深度学习基础 / 张量与自动微分

### [[Tensor]]

- [[Tensor]] — tensor 怎么做到自动微分 / 记录梯度？（`requires_grad` + `grad_fn` + `.grad` 三者合体）
- [[Tensor]] — 为什么要记录 device？（算子派发 / 跨设备搬运判定 / 多卡归属 / 显存记账）
- [[Tensor]] — 代码里的 `randn` 是什么？（标准正态 N(0,1) 采样）

### [[数值类型与精度]]

- [[数值类型与精度]] — 数值类型应单独展开成笔记（新建笔记，由 Tensor 批注触发）
- [[数值类型与精度]] — 混合精度训练是什么？
- [[数值类型与精度]] — 现在最主流的用法是什么？（bf16 AMP + fp32 master；V100 用 fp16+loss scaling；推理 int8/fp8）
- [[mixed precision training]] — 混合精度训练完整展开（bf16 AMP 主流路线、autocast 算子分派、fp32 master 权重、显存账 4P）
- [[mixed precision training]] — 哪里要用高精度？（master 权重/优化器状态/梯度累加/loss/数值敏感算子=累加归约小步长统计量保 fp32）
- [[loss scaling]] — fp16 训练防梯度下溢的损失缩放机制（bf16 不需要、dynamic scaling、skip step）

### [[autograd机制]]

- [[autograd机制]] — 梯度就是导数吗？沿着该方向变化最快的量？（梯度=多变量偏导向量，指向上升最快，负梯度下降最快）

### [[计算图]]

- [[计算图]] — 这和推理的"入图"/CUDA Graph 是一个东西吗？（计算图 vs CUDA Graph vs 推理图捕获，三层不同）
- [[计算图]] — 什么是算子？用户什么时候要自己写算子？训练框架的人会感知吗？
- [[计算图]] — 前向 vs 反向那张图看不懂，展开讲（用 x=1,W=2,b=0.5,y=3 逐节点算前向值+反向梯度）

## 一、ML/深度学习基础 / 优化器

### [[SGD]]

- [[SGD]] — 什么是 mini-batch？（从全集随机抽一小批样本算平均梯度近似全数据梯度；速度/精度/算力甜点，详见 [[batch与mini-batch]]）
- [[SGD]] — 怎么理解"有噪"？（mini-batch 梯度是全集梯度的随机估计，Var=σ²/|B|，batch 越小越抖；噪声有正反两面）

### [[Adam与AdamW]]

- [[Adam与AdamW]] — 优化器优化了什么？起了什么作用？（优化模型参数 θ；作用=把梯度变成参数更新：定方向/定步长/定正则）

## 一、ML/深度学习基础 / 训练工程基础

### [[epoch与step]]

- [[epoch与step]] — 多次 epoch 就是数据集多跑几次？（是；每轮 shuffle 分批跑 K 个 step；同样数据不同模型状态还能降 loss；过多轮过拟合；LLM 谈 tokens 不谈 epoch）

### [[batch与mini-batch]]

- [[batch与mini-batch]] — accum 是什么？（梯度累积步数；等效 batch = micro_batch × accum；loss 要除以 accum；显存不够 batch 来凑）

### [[损失函数分类]]

- [[损失函数分类]] — 损失我明白，为什么叫"损失函数"呢？（loss = 把"预测错误代价"量化成可微标量的函数；来自统计决策论 Wald 1950；loss/cost/objective/risk 四者辨析；非负、零损失、可微三性质）

---

## 二、Transformer & LLM核心结构 / Transformer基础

### [[残差连接]]

- [[残差连接]] — 每个子层（attention/FFN）都包残差的话起到什么作用？（信息 highway 前向不丢、梯度 highway 反向不衰减、能学恒等映射、隐式集成、与 LN 协同稳定）
- [[残差连接]] — 白话说残差是为了什么？（给信号抄近路直连，"输入直连+子层只学增量"，深网能退化成浅网、训练稳，是深 Transformer 可堆百层的结构前提）

### [[FFN]]

- [[FFN]] — 这一章没看懂，重写本章节（已应反馈重写：加白话类比"attention=课堂讨论/FFN=课后消化"、ASCII block 结构图、d=512 算参数例子、逐步推导位置独立与 SwiGLU 参数调整、误区针对初学者；核心=FFN 是每 token 独立的两层 MLP，干"加工+记忆"，与 attention"交流"互补，占 2/3 参数）

---

## 二、Transformer & LLM核心结构 / LLM结构

### [[tokenization]]

- [[tokenization]] — 现在主流模型用什么 tokenizer？Qwen/DeepSeek/GPT-5 都是专用的吗？（算法公开 BPE 系、词表/规则专用；GPT-5 沿用 o200k BBPE 无新算法；Llama2 SP→Llama3 tiktoken 不互通）

### [[decoder-only architecture]]

- [[decoder-only architecture]] — 下面是扩展阅读 整合一下（三种架构定义/原理/优缺点已提炼整合到 3.2 表格后 callout，补 BERT MLM/Seq2Seq 场景，删除原文冗余）

### [[logits generation]]

- [[logits generation]] — logits 是什么？（softmax 前的未归一得分向量 $z\in\mathbb{R}^V$；词源 log-odds；几何=与词向量余弦相似度；hidden→logits→probability 三层递进）
- [[logits generation]] — lm_head 是什么？（最后一层线性层 `Linear(d,V,bias=False)`，语义→词表得分；命名=language modeling head；常与 embedding tied；$z=hW^T$）

---

## 二、Transformer & LLM核心结构 / 推理关键机制

### [[KV cache]]

- [[KV cache]] — KV cache 占多少显存？推理时有 KV cache 吗？（推理默认开 `use_cache=True`，不开则 $O(n^2)$ 不可行；公式 $M=2BLn_{kv}n_{seq}d_h b$；Llama-2-7B 2GB@4k、Llama-3-70B 40GB@128k；GQA 省 4 倍）
- [[PD分离]] — pd分离 展开一下 新建一个章节（**新建笔记**；PD=Prefill-Decode Disaggregation 预填充-解码分离部署，非训推分离 Policy Deployment；归 [[08-推理系统]]/推理优化）
- [[KV cache]] — 很多论文可以有选择的保存kvcache的驻留，依据什么判断？怎么控制要留要删哪些？（H2O/StreamingLLM/sink+滑窗/学习式稀疏；依据 attention score 累积/recency/sink；控制预算 k、窗口 w、sink 数 s、驱逐触发）
- [[KV cache]] — 训练时有 kvcache 吗？（训练无 KV cache：并行前向 + 权重每步变 + K/V 是中间激活用完即丢；推理才有，因逐词生成 + 权重冻结 + 历史 K/V 不变可复用）

### [[autoregressive decoding]]

- [[autoregressive decoding]] — 为什么要"采样一个词 + append 回输入"循环？（因果性→必须逐词；概率分布→采样而非 argmax；下一步要完整前文→append；串成 前向→logits→采样→append 闭环直到 eos）

---

## 二、Transformer & LLM核心结构 / MoE

### [[MoE]]

- [[MoE]] — 这里的"算力"指显存吗？（不是；算力=FLOP 计算量、显存=bytes 存储；dense 都∝N 易混，MoE 解耦：算力按激活参数省、显存按总参数费）
- [[MoE]] — 路由不可导"那又怎样"？（未选专家无梯度→冻死；路由器无法自我纠正→正反馈锁死→专家坍缩；必须靠负载均衡损失/偏置外部打破）
- [[MoE]] — 什么是超参？（人设定不随梯度更新的配置量，对比参数=模型自学权重；MoE 常见 E/k/α/c/σ；MoE 比 dense 敏感需小心调）

### [[负载均衡损失]]

- [[负载均衡损失]] — 负载均衡只针对训练？推理还用顾虑吗？（辅助损失训练专属；但推理仍需均衡：专家并行热点过载/吞吐不均/延迟不稳；改用容量因子/路由调度/expert placement/All-to-All 重排）

---

## 六、训推系统 / Rollout系统

### [[R3 rollout routing replay]]

- [[R3 rollout routing replay]] — 补充（原文件仅批注"补充"；R3=Rollout-Routing-Replay，按 RL rollout 路由与回放补全并移至第六章 Rollout系统；含 routing 策略 / replay buffer / 重要性采样修正 / 与推理系统复用）

---

## 三、PyTorch / 深度学习框架工程 / PyTorch核心

### [[state_dict与load_state_dict]]

- [[state_dict与load_state_dict]] — state_dict 是"存权重本身"吗？safetensors 存的是它吗？（是，但含 Buffer；加载验证看返回值 + 逐键 `torch.equal`；safetensors 即 state_dict 的安全格式）
- [[state_dict与load_state_dict]] — Buffer 是什么？（`register_buffer` 注册的不学但要搬的张量；BN running / RoPE inv_freq；对比 Parameter）
- [[state_dict与load_state_dict]] — LR 调度是什么？（按 step/epoch 改 lr 的策略；warmup+cosine 标配；续训要存 `scheduler.state_dict()`）
- [[state_dict与load_state_dict]] — 怎么从键识别 dense / MoE？命名规则？（看 `experts.{i}.` 和 `gate`；Dense 一组 FFN，MoE E 组+路由器；附自动识别脚本）
- [[state_dict与load_state_dict]] — `strict=False` 是什么意思？（只灌同名键，多余/缺失静默跳过；微调改结构用；务必看 missing/unexpected_keys）
- [[state_dict与load_state_dict]] — §3.3 两种保存粒度没看懂（粒度 A 只存 state_dict 推荐/解耦代码；粒度 B 存整个 model pickle 不推荐/绑死类路径）
- [[state_dict与load_state_dict]] — mlp 指的是什么意思？（MLP=Multi-Layer Perceptron 多层感知机；Transformer 语境下指每个 block 的 FFN 子模块；Llama SwiGLU 三矩阵 gate/up/down；MoE 把它换成多专家+路由器）

### [[forward与backward]]

- [[forward与backward]] — 输出 $\hat y$ 与损失 $\mathcal{L}$ 是什么？（$\hat y$=预测 y-hat；$\mathcal{L}$=损失标量；forward 数据流 $x\to f_\theta\to\hat y\to\ell\to\mathcal{L}$；$\mathcal{L}$ 是 backward 唯一入口）
- [[forward与backward]] — 训推一致性为什么比 forward logp？为什么不管 backward？Pearson？（logp 标量可比/PPO 直接用；forward 一致是 backward 一致充分条件；Pearson>0.99 且 max_abs_diff<1e-3）

### [[optimizer state管理]]

- [[optimizer state管理]] — 什么是优化器？到底在优化什么？（把梯度变成参数更新的算法；优化参数 $\theta$ 使 $\mathcal{L}$ 最小；$\theta^*=\arg\min\mathcal{L}(\theta)$；三类对象分工表）

> 本章另落盘笔记（无批注）：[[nn.Module]]、[[hooks机制]]、[[torch.distributed]]、[[process group]]、[[rank与world size]]、[[NCCL backend]]、[[Data Parallel]]、[[Distributed Data Parallel]]、[[Fully Sharded Data Parallel]]。

---

## 十一、显存与内存系统 / 资源核算

### [[FLOPs计算]]

- [[FLOPs计算]] — 前向 2 反向 4 怎么得出来的？（矩阵乘 2mn=2N_W 前向每参数 2 FLOP；反向算输入梯度+权重梯度两个 2N_W=4N_W；合计 6ND；直觉=反向算两次乘加是前向 2 倍）

---
相关: [[整体目录]]、[[张量与自动微分]]、[[优化器]]、[[训练工程基础]]、[[Transformer基础]]、[[03-PyTorch与框架工程]]
