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
- [[R3 rollout routing replay]] — 这章感觉在胡扯，联网搜一下，这是保证训推一致性的手段（**纠错重写**：之前把 R3 误读为"rollout/routing/replay 三段流水线"是错的；R3 实为 Ma et al. 2025 arXiv:2510.11370 的具体方法，在推理引擎记录 MoE router 专家选择、训练时回放，保证训推走同一专家子网络，消除 MoE 的 Training-Inference Mismatch 稳定 RL 训练；R1/R2/R3 分档；dense 模型不需要；工程坑见 vLLM/verl 修复 PR；与 weight sync/重要性采样正交）

### [[训推不一致]]（新建笔记）

- [[训推不一致]] — 细说一下 什么时候会有训推不一致 怎么衡量（pearson）会有什么影响 新建一个md（**新建型**：TIM=训推双引擎同权重算出不同 logp；来源分层=精度路径/算子实现/kernel融合/归约顺序/KV量化/RoPE base/MoE路由，分数值型与MoE结构型；衡量指标=逐token绝对差+Pearson(>0.99)+KL+$F(\tau)$+JS+ratio偏移，Pearson看形状max_abs_diff保绝对；影响四级=噪声→有偏→KL爆炸→崩溃(MoE)；与staleness区分=同θ引擎差异vs不同θ权重滞后；消除手段分层=精度/算子/超参对齐/bitwise/R3/KV量化对齐）
- [[R3 rollout routing replay]] — 假设项目里落地 要记录训练端的什么？记录路由选择？怎么记录（**行内型**：纠正方向——R3 在**推理端**记录、训练端回放，只有 R2 才训练端记录；记录对象=每token每MoE层TopK专家**索引**(int32)，不记logits/gating；张量契约`(seq-1,n_layers,topk)`；SGLang `--enable-return-routed-experts`原生/vLLM需patch+`--no-async-scheduling`；随轨迹经TransferQueue/packing/CP切片传到Megatron的`RouterReplay`安装mask，gating仍用softmax(s_train)保梯度；落地8步清单；工程坑见EPLB逻辑ID/dense层偏移/捕获时机）

### [[trajectory generation]]

- [[trajectory generation]] — 回答就回答呗 为什么要叫轨迹（trajectory 词源 MDP 形式化 τ=(s,a,r) 时序路径，时空比喻"agent 在状态空间随时间走出的一条带训练信息的路径"；为什么不叫 sample/output——trajectory 容器要装 logp/value/reward 全套 PPO 才能用；trajectory=episode=rollout 同义辨析；LLM 里 token=action、prompt=s_0、整条 response 即一条轨迹）

### [[batching strategy]]

- [[batching strategy]] — 为什么要 padding 到桶长（padding 是为批处理凑等长张量；pad 到桶长而非全局/batch 最长，是为了让同 batch 长度接近、把浪费率 `1-ḹ/L_max` 压到最低，顺带换静态 shape 利于编译；vLLM 靠 PagedAttention 物理消除 padding 故不用桶，传统框架靠分桶）

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
- [[optimizer state管理]] — 白话说一下优化器起什么作用 还没看懂（蒙眼下山比喻：梯度=探哪边下坡、优化器=迈那一脚；不定义目标不产梯度只翻译成参数更新；Momentum 抹噪声、Adam 自适应步长；状态=优化器记的小本本）
- [[optimizer state管理]] — 这里是不是格式写错了？（是；AdamW 多行公式块原用单 `$` 定界，跨行 `aligned` 不渲染，已改为 `$$...$$`；附行内 vs 独立公式块定界符规则表）

> 本章另落盘笔记（无批注）：[[nn.Module]]、[[hooks机制]]、[[torch.distributed]]、[[process group]]、[[rank与world size]]、[[NCCL backend]]、[[Data Parallel]]、[[Distributed Data Parallel]]、[[Fully Sharded Data Parallel]]。

## 三、PyTorch / 深度学习框架工程 / 分布式基础

### [[torch.distributed]]

- [[RPC]] — [[DDP]]/[[FSDP]]/[[RPC]]/[[Ray与分布式调度]]/集合通信原语 新建补全（**新建型**：DDP/FSDP 已有全名笔记，加 `aliases` frontmatter 让短名解析；新建 [[RPC]]（§10）、[[broadcast]]（§23 通信机制）、[[Ray与分布式调度]] 章总览（§28）；整体目录已加双链）
- [[torch.distributed]] — huawei hccl 有什么区别和联系？（**行内型**：HCCL 是华为 Ascend NPU 对标 NCCL 的集合通信库，CANN 栈里同位体；硬件/互联/算法/容错/开源/生态对比表；`torch_npu` 把 hccl 适配为 `torch.distributed` 后端，同接口不同实现；互不兼容、不能直接换 import）

### [[NCCL backend]]

- [[NCCL backend]] — gloo 是什么？第一次听说 CPU 训练（**行内型**：gloo=Meta 开源 CPU 集合通信库，`torch.distributed` 的 `backend="gloo"`；与 NCCL 并列、按张量设备二选一；CPU 训练真实场景=小模型/调试/无GPU集群/DDP逻辑单元测试；gloo vs NCCL 对比表+代码+误区"GPU 张量→nccl，CPU 张量→gloo"）
- [[NCCL核心机制]] — NCCL 五点核心价值展开/新建文件（**新建型**：①ring all-reduce 算法（通信量 $\frac{2(N-1)}{N}M\to2M$ 与 N 无关、无单点瓶颈）②拓扑感知（探测 NVLink/PCIe/IB 建 channel、自动选 ring/tree/collnet/nvls）③GPUDirect RDMA（跨机 GPU 直达网卡不绕 CPU 内存）④CUDA kernel 融合+stream overlap（async_op 独立 stream）⑤算子丰富全套（all-reduce/all-gather/reduce-scatter/broadcast/all-to-all/send-recv）+ async；附带宽量级表与调优经验值）

### [[rank与world size]]

- [[rank与world size]] — master rank/rank 0 有什么特殊？为什么要知道谁是 0？（**行内型**：rank 0 计算上平等，只在 IO/协调上被约定挑出来当"管事的"；必须知道谁是 0 才能写 `if rank==0` 守卫和 `broadcast(src=0)`；master 是社区约定非框架强制）
- [[rank与world size]] — 为什么叫 nnode/nnodes？（**行内型**：`nnodes`=number of nodes="节点数=机器数"；`n` 前缀=数量约定，node=一台机器；附三个 n 的关系表与误区）
- [[rank与world size]] — §8.3 两种绑卡流派是什么意思？（**行内型**：A 流派 `CUDA_VISIBLE_DEVICES` 隔离可见性每进程只见一卡 local_rank 恒 0；B 流派 torchrun 全可见+`set_device(local_rank)` 绑定，主流；对比表+为什么选 B+误区）

---

## 三、PyTorch / 深度学习框架工程 / 并行训练

### [[Data Parallel]]

- [[Data Parallel]] — GLI（应为 GIL）是什么？（**行内型**：GIL=CPython 全局解释器锁，任一时刻只允许一个线程执行 Python 字节码；DP 单进程多线程下 Python 调度段被 GIL 串行化是慢的主因之一，CUDA kernel 调用段 C 扩展释放 GIL 可并行；DDP 多进程每进程独立 GIL 不互锁；速记表）
- [[Data Parallel]] — batch 里装的是什么？是一条条 prompt 吗？（**行内型**：batch=一组样本；预训练是连续文本段、SFT 是 prompt+response、RLHF/PPO 是纯 prompt、DPO 是 prompt+chosen+rejected；DP/DDP 按 dim0 切样本数不切单条；input_ids shape=(B,seq_len) 切成 (B/N,seq_len) 每卡 N 条完整样本）
- [[Data Parallel]] — all-reduce 是让每个卡上都有完整的一份吗？（**行内型**：是，all-reduce 定义即"每 rank 都拿到全体和/均值"；对比 reduce(只 root 有)/broadcast(复制 root 值)/all-gather(拼装)/reduce-scatter(分片)；DDP 用 all-reduce 因需所有 rank 一致梯度更新；DP 用 reduce+broadcast 两步有主卡瓶颈）

### [[Distributed Data Parallel]]

- [[Distributed Data Parallel]] — 怎么理解各 rank 持全量参数？92B 装得下吗？参数指什么？（**行内型**：DDP=每卡一份完整模型副本切数据，参数指全部权重+buffer，N 份冗余换吞吐；92B bf16 权重 184GB>单卡 80GB，训练态 16P≈1472GB 根本装不下直接 OOM，必上 FSDP/Megatron TP；DDP 适用边界=单卡训练态显存，越过即 FSDP/TP/PP）

### [[Fully Sharded Data Parallel]]

- [[Megatron-LM]] — 提到 FSDP 就该讲讲 Megatron，新建一页（**新建型**：Megatron-LM=TP+PP+DP 三维并行工业参考实现；与 FSDP 是两条省显存主流路线——FSDP 分片存储 all-gather 全量计算通信∝参数可跨节点，Megatron 真切分权重每卡只算一片通信∝激活不随模型变大但需 NVLink；列切/行切数学推导、MLP 夹心组合减通信、PP 1F1B/interleaved、并行交叉熵、selective recombination、Megatron vs FSDP 对比表、何时用哪个）
- [[Megatron-LM]] — 联网搜索 megatron 到底和 fsdp 有什么区别和联系（**行内型+联网**：一句话=Megatron 真切权重每卡算一片通信∝激活需 NVLink 扩展≤8，FSDP 分片存储 all-gather 全量通信∝参数可跨节点扩展数百卡；正交可叠加(超大模型=TP+PP+ZeRO-3 工业配方)；血统不同(NVIDIA vs DeepSpeed/PyTorch FSDP2 DTensor)；正在 DTensor 层融合(Megatron 已内置 megatron-fsdp)；工程选型速记；来源 NVIDIA/PyTorch/ZeRO/Megatron 论文）

---

## 十一、显存与内存系统 / 资源核算

### [[FLOPs计算]]

- [[FLOPs计算]] — 前向 2 反向 4 怎么得出来的？（矩阵乘 2mn=2N_W 前向每参数 2 FLOP；反向算输入梯度+权重梯度两个 2N_W=4N_W；合计 6ND；直觉=反向算两次乘加是前向 2 倍）

### [[训练显存估计]]

- [[训练显存估计]] — 新建一章节 专门来讲解 训练显存估计 假设训练92b模型 需要多少显存？（**新建型**：训练峰值显存公式 $M \approx 16N + M_{\text{act}}$，Adam+bf16 AMP 训练态 = 权重 $2N$+梯度 $2N$+optimizer state $12N$（fp32 master $4N$ + Adam $m,v$ 各 $4N$）= $16N$；92B → 1472GB 训练态、单卡 80GB OOM 19 倍、DDP 不可行、FSDP/ZeRO-3 分片 $P_{\min}\approx 16N/64 \approx 23$ 卡实测 32-64 卡；加 FA+ckpt 激活 $\approx 21.5$GB、通信临时约 5%；ZeRO-1/2/3 三阶段单卡显存分级表、不同 optimizer state 大小对照、8-bit optimizer/offload 省显存手段、口诀 $P_{\min}\approx N/4$GB；区分训练显存 vs 推理显存(2N+KV) vs $16N$、漏算 master 权重误区、MoE 用总参数算显存）

---

## 四、强化学习基础 / RL基本概念

### [[MDP]]

- [[Actor-Critic]] — 什么是 [[Actor-Critic]]？（**新建型**：A-C=策略梯度两网络联合学习架构，actor 学 $\pi_\theta$、critic 学 $V_\phi$，critic 给 actor 低方差 advantage 权重 $\hat A_t$；把 REINFORCE 的"用实际 return 当权重、无偏高方差"升级为"用 critic 估 $V$ 减底色、有偏低方差、可在线"；是 A2C/A3C/PPO/RLHF-PPO 的统一骨架；含策略梯度定理→advantage 形式推导、双时间尺度收敛、GAE 标配、可跑代码、LLM-RL 映射表、与 REINFORCE/DQN/DDPG/SAC 对比、12 条误区）

## 四、强化学习基础 / Policy Gradient体系

### [[REINFORCE]]

- [[REINFORCE]] — 策略梯度用 $\log\pi$ 而不是 $\pi$ 的好处在哪里？（**行内型**：五大好处——①连乘变连加求导 $O(2^T)\to O(T)$ ②干净分离策略与环境→model-free 数学根基 ③数值防下溢长序列必须 ④梯度量级归一化 $\nabla\pi/\pi$ 训练稳 ⑤与 SFT 交叉熵同构工程可复用；附五维对比表与"取 log 是等价变换不改最优 $\theta^*$"误区）

### [[variance reduction]] / [[baseline]]

- [[variance reduction]] — "方差"意味着什么？（**行内型**：方差=梯度估计量 $\hat g$ 的随机波动 $\text{Var}[\hat g]=\text{Var}[X]/N$，非 reward/参数方差；方差大→学习率必须小→训不动；PG 蒙特卡洛天然大方差，variance reduction 全套技巧都在压 $\text{Var}[\hat g]$；含方差来源拆解/偏差-方差 trade-off/GAE 插值。同一问题另现于 [[baseline]] §4.2）
- [[baseline]] — §4.2 这里的"方差"指什么？（**行内型**：指梯度样本 $\nabla\log\pi\,(G_t-b)$ 的方差；baseline 通过压二阶矩 $\mathbb{E}[X^2]$ 降它、期望项不变故无偏；$b=V^\pi$ 时扣掉状态底色方差降最多，下降量 $\approx\text{Var}[V^\pi]$）

---

## 五、LLM RL（RLHF / RLAIF） / 对齐算法

### [[DAPO]]（新建笔记）

- [[DAPO]] — 增加相关章节 DAPO 在 Timely rollout 上提供 prompt-group 级 dynamic sampling、pre-filter、post-filter、replay 和 refill 语义。是什么意思（**新建型+联网**：DAPO=arXiv:2503.14476 字节 Seed，GRPO+4 trick 治 long-CoT RL 的熵坍塌/梯度消失/长度漂移/奖励噪声；批注那串术语是对 Dynamic Sampling 在 rollout 系统实现语义的工程拆解——timely rollout=采样过滤当场在 rollout 阶段做、prompt-group 级=过滤粒度按"一 prompt+G 条响应"整组判 reward std=0 整组扔、dynamic sampling=采不够动态多采、pre-filter=进训练 batch 前剔除 trivial 组、post-filter=累积阶段对齐切片、replay=不够 continue 重采一轮、refill=补满到 train_batch_size；一句话=rollout 端"当场判、整组过滤、不够重采、补满为止"；4 trick 完整推导/verl 配置/代码循环/误区/与 GRPO/PPO/DPO 对比表；来源 arXiv:2503.14476/verl docs 2025-06/dapo-sia.github.io/swift docs/腾讯云 2026-07-14 核实）

---

## 六、训推系统 / 训练架构

### [[learner worker split]]

- [[learner worker split]] — 训练就是 trainer？推理就是 worker？这是训推分离？（**行内型**：learner=trainer=训练器✓；worker 做的是"为采数据的前向"≠"为用户服务的前向"，后者叫 inference worker；split 是 PT 内部采训分离，[[训推分离]] 是 PT↔PD 分离，口诀"split 拆采与训，训推分离拆训与用"）
- [[learner worker split]] — Megatron 也属于这里？DeepSpeed/FSDP/ZeRO 四个差别？（**行内型**：Megatron 属 learner 框架✓；但"四个"是误会——ZeRO 是 DeepSpeed 内的算法非独立框架，FSDP 是 PyTorch 原生版 ZeRO-3；真正三个框架 DeepSpeed/FSDP/Megatron-LM+一个算法 ZeRO；附四维对照表与"何时用哪个"选型经验）
- [[learner worker split]] — 吞吐里的 trajectory 是什么？新起一章（**行内型**：trajectory=一次完整 rollout 采的全量数据(prompt,response,logp,v,reward)；LLM-RL 里默认一条 prompt→response=一条 trajectory；$T_w$ 用 trajectory/秒衡量产样本速度，=tokens/sec÷平均长度；专章见 [[trajectory generation]]）
- [[learner worker split]] — IMPALA V-trace 细说一下（**行内型**：V-trace=截断 IS 修正×多步 TD 回溯；$\rho_s=\min(\bar\rho,\pi_\theta/\pi_{\theta_{old}})$ 截断防 IS 爆炸、$c_s$ 截断控回溯深度；vs PPO clip 同思想不同位置；LLM-RL 多用 PPO clip+target_kl，V-trace 主流在传统 RL）
- [[训推分离]] — 新开一章节专门讲训推分离（**新建型**：[[训推分离]] 已建为第六章 §20 总览笔记——PT-PD 训练-部署分离，⊃ split；weight sync 版本级低频；撞名 [[PD分离]](Prefill-Decode)；整体目录 §20 已加双链）
- [[learner worker split]] — 吞吐指的就是吐字的量吗？（**行内型**：throughput=单位时间单位工作量，非"吐字量"；分三层——生成 tokens/sec、采样 trajectory/秒、训练 batch/秒；token 只是①，靠平均长度÷换算成②；split 总吞吐=瓶颈侧 min；vs 延迟辨析）
- [[learner worker split]] — split 与训推分离展开细说（**行内型**：两层嵌套训推分离⊃split；同=都分离+都 weight sync+都 stale；不同=分离两端/使用方/频率/产物；两条 weight sync 叠加内层 step 级外层版本级；图示+对比表；完整版见 [[训推分离]]）
- [[learner worker split]] — 联网调研负载均衡业界进展（**行内型+联网**：四主线——AReaL 全异步主动均衡+staleness PPO 2.77×(arXiv:2505.24298)、HybridFlow 3D-HybridEngine reshard+自动设备映射 1.53-20.57×(arXiv:2409.19256)、vLLM continuous batching+length bucketing 解变长短板、Ray autoscaler+colocate 弹性调度；附对比表与来源）

### [[rollout worker]]

- [[rollout worker]] — 与 [[inference worker]] 的区别细说（**行内型**：七维对比表(目的/数据去向/θ/采样/优化/框架/所属)；同动作不同语境；判别小测验→去 4/5 仍是 rollout；口诀"rollout 为训采数据喂 learner，inference 为用服务喂用户"）
- [[rollout worker]] — 多 worker 并行=多个 DP 吗？（**行内型**：只像一半——副本复制同构，但无梯度 all-reduce、不产梯度、weight sync 单向下发；是"模型副本并行/数据并行的采样侧"，与 learner 侧 DP 对称但不同质；类比"采蜜蜂不对账"；verl/OpenRLHF worker/learner 数独立配）
- [[rollout worker]] — 从 learner 拿最新 θ 是什么？是权重吗？π_θ 是什么？（**行内型**：θ=全部可学习参数(model.parameters()/state_dict)✓；π_θ=装上这组权重的模型当作策略看，=softmax(logits_θ)的 token 分布；θ 静态参数值 vs π_θ 动态行为函数；改 θ→π_θ 变→sync 给 worker 用新策略采）
- [[rollout worker]] — 不算步骤 4/5 就变成 inference worker 为部署服务用户？（**行内型**：否，判别看四件套(输出去哪/θ变不变/采样策略/优化目标)不看附加任务；去 4/5 仍四条全占训练侧；GRPO 无 ref/rule-based reward/reward 外置都是去 4/5 的现实场景，身份不变）
- [[rollout worker]] — 这两个 worker 干什么的？另起 reward worker / reference worker？（**行内型**：reference model=冻结的初始 actor 算 logp_ref 给 KL 防 reward hacking；reward model=偏好数据训的 RM 算 reward；两者只前向但占显存算力→常独立成 reward worker 省 rollout 显存；三种部署模式 A 合并/B 独立 reward worker/C learner 端补算对比表；reward worker 仍属 PT 侧非 inference worker）

### [[训推分离]]（新建章节笔记）

- [[训推分离]] — 训练-部署分离总览（PT-PD disaggregation；⊃ [[learner worker split]]；weight sync 版本级；撞名 [[PD分离]]；蓝绿/canary 版本管理；colocate vs 分离；RLHF 在线服务高频 sync 场景）
- [[PD分离]] — pd分离也应该新建一个md 我是说prefill decode 分离（**新建型-已存在**：用户要的 Prefill-Decode 分离已有独立笔记 [[PD分离]]，位于 `08-推理系统/推理优化/`，已完整展开两阶段算力特征/P-D池分离/KV cache迁移开销/配比/DistServe·Mooncake·SGLang/池化预取量化；与训推分离撞名不同物，互为对照）
- [[训推分离]] — 异步其实也属于一种训推分离？（**行内型**：抽象形式上"是"——同属"训/推解耦+weight sync"骨架，rollout侧真跑vLLM推理栈故形式同构+技术复用；严格语义上"不是本条PT-PD"——异步是learner-worker split的运行模式、split是训推分离PT内部子架构、二者嵌套训推分离⊃split∋{同步,异步}；差别锚点=推理侧服务谁：服务用户=训推分离、服务learner=split；三层分离对比表8维）

### [[weight sync mechanism]]

- [[weight sync mechanism]] — 联网搜索 现在主流传输的方法是什么？checkpoint engine？（**行内型+联网**：四条主流路线——①NCCL broadcast（RLHF 热路径绝对主流，vLLM `WeightTransferEngine` 四阶段协议+packed tensor、OpenRLHF `--vllm.sync_backend nccl`、verl）②CUDA IPC（同机 hybrid engine 零拷贝，OpenRLHF `update_weights_from_ipc_handles`）③**Checkpoint Engine**（SGLang 主推，磁盘→多机并行加载加速，broadcast/p2p/all 三模式，PD 冷启动/大模型多机首载，用户猜对了）④共享 FS delta pull（SGLang `/pull_weights`，zstd per-tensor delta+checksum，非 colocated rollout/PD 多副本）；另有 HTTP 控制面+NCCL 数据面在线 serving；四者按场景共存非互斥，附对比表与来源）

### [[asynchronous training]]

- [[asynchronous training]] — 联网搜业内最新动态，提名 areal/asyncflow/fullyasync（**行内型+联网**：三大主流系统——AReaL(arXiv:2505.24298, 蚂蚁+清华, 全异步+decoupled PPO 容忍8步旧, 2.77×)、AsyncFlow(arXiv:2507.01663, verl-recipe, TransferQueue+CheckpointEngine, 3.81×)、verl fully_async_policy(主仓4模式 on-policy/stream/async-stale/partial-rollout, 2-2.35×)；另附 StreamRL/Magistral/StaleFlow(2026)/Laminar(EuroSys2026)；趋势=全异步解耦+staleness主动治理+数据中枢+权重sync工程化+agentic原生）
- [[asynchronous training]] — KL这是什么（**行内型**：KL=Kullback-Leibler divergence 相对熵 D_KL(P‖Q)=ΣP log(P/Q)，非负不对称；RL/PPO 里三角色=①策略漂移监控 π_θ vs π_old ②target_kl 早停超则break ③KL penalty/控制 -β·KL(π_θ‖π_ref) 软约束；ratio clip 是单步防飞阀、KL 是累计漂移仪表，异步 stale 让 KL 更易涨故监控更关键；详见 [[KL penalty与KL control]]）

---

## 八、推理系统 / 延迟与吞吐优化

### [[batching tradeoff]]

- [[batching tradeoff]] — perf/throughput 是什么指标 细说一下（**行内型**：三类指标——①吞吐类 throughput=单位时间产出，QPS 请求/秒、TPS token/秒、goodput 满足 SLO 的有效吞吐；②延迟类 latency=一个请求等多久，TTFT 首 token、TPOT 每 token、E2E、P50/P99 分位数；③利用率类 utilization=硬件吃没吃满，GPU SM 占用、显存带宽利用率、KV cache 占用；三者被 batch size 反向拉扯，batching tradeoff 在延迟 SLO + 显存预算双约束下找 throughput/goodput 最大 batch，Little's Law `λ_max=B_max/W_max` 给上限）

### [[KV cache management]]

- [[KV cache management]] — kvcache 能由调度器自动管理吗？自动把请求调度到更多命中的实例上？（**行内型+联网**：能——多实例层有 KV-cache-aware / prefix-aware router 把同前缀请求路由到命中高的 pod；近似路由按历史预测、精确路由(vLLM KVEvents/llm-d kvcache.Index)实时窥探 block-hash→pod；llm-d 实测精确路由 TTFT 快 57×、吞吐翻倍 vs 盲调度；选型梯度 random→load→approximate→precise；与 §2 三层管理的 ③ 调度器层在多实例维度的延伸正交叠加）
- [[FP8量化方案]] — fp8 的不同量化方案展开（**新建型**：FP8=E4M3/E5M2 两格式 × per-tensor/per-channel/per-token/per-block(MX)/delayed scaling 缩放策略的组合；训练 TE 默认前向 E4M3+per-tensor delayed、反向 E5M2，推理 vLLM/TRT-LLM 权重 per-channel+激活 per-token+KV E4M3，方案不等价是 [[训推不一致]] 精度路径 TIM 主因；MXFP8 是训推统一希望；TE/vLLM 代码、per-tensor vs per-block 精度推导、6 条误区）
- [[KV cache management]] — prefill 和 decode 本质都是算 KV，分两阶段只因"token 拿到的时机不同"对吗？（**行内型**：大方向对但只抓到表象——根本原因是自回归生成的**串行因果依赖**使 decode 无法并行（第 t 个 query 要等 t-1 步 argmax 出来才能算），非仅 prompt 提前给；并衍生 compute-bound vs memory-bound 两套截然不同的 kernel(GEMM vs GEMV)/调度/指标体系(TTFT vs TPOT)；prefill 一次性并行填 N 个 K/V 进 cache，decode 每步 append 1 个；两阶段抢资源冲突催生 chunked prefill / PD 分离 / continuous batching 等调度策略）

### [[dynamic batching]]

- [[dynamic batching]] — 为什么要一个batch一起前向？一个请求不能前向吗？（**行内型**：能前向(batch=1)但亏——GPU 是吞吐器件(A100 312T/6912 core)，单请求算力利用率 <10% memory-bound；攒批让"一次访存服务多请求计算"算术强度↑吞吐近线性涨而单次耗时几乎不涨，64 条 batch 比串行快 ~40×；数学：T_fwd(B)≈launch+2NB/P 慢增、throughput=B/T_fwd；反问两答(单请求延迟低但成本最高/double 凑批等待正是 batching tradeoff 由 dynamic 双触发权衡)；单请求合法场景(调试/极敏延迟/小显存/流式 decode)；LLM decode 每步本质 batch=1 故 continuous/speculative/KV量化/kernel融合全在对抗此浪费；一句话=把312T GPU当3T用是浪费）
- [[dynamic batching]] — 矩阵相乘类比 m 越长算越多显存固定 对吗？（**行内型**：方向对但需修正——更准确是 X(B×d)×W(d×d)=Y(B×d)，W 权重固定(显存固定✓只读一次) B=batch 行数变长 FLOPs=2Bd² 随 B 线性涨 算术强度 I=B 随 B 涨→利用率↑(你抓对的部分)；但"显存固定"只对权重成立，KV cache 随 B 线性涨(每请求独立 cache, batch64 7B@2k ≈32GB)、激活随 B 涨，这是 batch 不能无限大的硬约束 OOM；收益来自权重固定、代价来自 KV cache 不固定=batching tradeoff 两面；修正后=权重矩阵固定被多行摊薄、每行拖随它走的 KV cache）

### [[continuous batching]]

- [[continuous batching]] — 分块预填是怎么做到节约时间的（**行内型**：核心澄清——chunked prefill 几乎不减 FLOPs，省的是等待/闲置/重算。五条收益：①长 prefill 不再独占 GPU 阻塞 decode（消除饿死与 TPOT 尖峰）②TTFT 降（首 chunk 算完即进 decode）③KV 增量分配免 preemption 重算（零存整取 vs 一次性大额申请）④prefill 算力与 decode 带宽交替吃满 SM/带宽更均衡⑤每 iteration 重新评估显存预算做动态反馈控。数学直觉对比 B·T_P 的 decode 产能冻结 vs 接近0损失。类比超长货车拆小车混行。误区=不减FLOPs/不是总提速/chunk太小反慢/与prefix caching正交非同物）

### [[prefix caching]]

- [[prefix caching]] — 这个技术现在是不是已经很容易实现了？直接调 API 就行？训推框架开发人员感知吗？（**行内型+联网**：三层回答——①API 用户近乎零成本白嫖(OpenAI 全自动 90% off 无 knob/GPT-5.5+ 默认 24h 扩展/Anthropic 显式 `cache_control` 90% off 但写费 1.25×/Gemini 90% off 但有存储租金/DeepSeek V4 Flash 98% off 最狠)；2026-07 价格表+5 条坑(静态前→变量后/低于阈值静默不缓存/Gemini 闲置计费/Batch 可叠加 95% off/健康命中率 60-90%)) ②自建推理一行开关即有(vLLM `--enable-prefix-caching` block 对齐有粒度损失/SGLang RadixAttention 默认零配置无对齐损失/TRT-LLM 内置；2026 vLLM+SGLang 唯二重要生产引擎) ③框架开发人员必须深度感知(推理引擎当核心 KPI+护城河,SGLang RadixAttention 快 vLLM 3-5× prefill+29% 吞吐,RadixArk 分拆融 $400M;训练框架靠它提 rollout 吞吐但须 weight sync 失效缓存治 staleness+TIM 对齐 KV 量化；在线学习权重新鲜度与复用冲突需版本化缓存)。一句话=容易实现对使用者,对开发者仍是性能与一致性硬仗。来源 vLLM docs/turion.ai 2026/inference.net 2026/leanlm.ai/bytecosts/aicost.ai 2026-07-12 核实)
- [[prefix caching]] — 我记得我的项目一打开很容易把显存打爆（**行内型**：prefix caching 本身几乎不额外占显存(指针复用物理 block,引用计数+1),真凶是四条——①KV cache 预分配过头(vLLM `--gpu-memory-utilization` 默认 0.9 顶太高,启动瞬间 90% 显存没了,forward 要 activation 直接 OOM)②模型权重本身大(70B bf16 140GB 单卡装不下须 TP)③多实例/多进程抢卡④CUDA context+临时张量没留余量；prefix caching 间接影响只在命中率低冷前缀堆积/weight sync 未失效旧 KV 残留/block_size 太小碎片；治法=util 降到 0.85-0.88+大模型 TP+`--swap-space` 兜底+监控命中率；附 vLLM 70B TP4 启动配方）

### [[PD分离]]

- [[PD分离]] — 为什么要把 gpu 叫做实例？（**行内型**：实例=一个独立运行的推理服务进程/部署单元=逻辑调度单位≠物理 GPU 硬件单位；一个实例可挂 1 卡(TP=1)也可挂多卡(TP=8)；叫实例因①逻辑服务单元与硬件解耦②路由/扩缩容/版本管理对象是实例非卡③云原生 borrow(K8s pod/EC2 instance)④weight sync 按实例 TP>1 时歧义；附 物理层/实例层/池层/路由层四层图 + instance/worker/pod/replica 术语对齐表；vLLM/SGLang 文档 worker/instance 混用都指推理服务单元,硬件语境才叫 GPU/device）
- [[PD分离]] — 这两个比例一样吗？1 个 node 8 卡 prefill 8 卡 decode 会不会 P 阻塞 D？（**行内型**：三答——①§3.4/§4 的 P:D 比例是同一个概念(都是 P 池与 D 池资源配比),非两套,§2 末尾只描述两池配置方向未给数；②1:1 是中性起点,合不合理看流量画像(偏生成→1:2~1:5 D 多/偏 prompt→2:1~3:1 P 多/均衡→1:1 起步动态调/长 context→池化优先比例次要),工业经验常见 1:2~1:5；8P8D 同 node 机内 NVLink 迁移快是甜点拓扑但比例要监控队列动态调；③分离架构核心目的就是消除 P 阻塞 D(P/D 不同实例不同 batch,decode TBT 稳),但冒出新问题=KV cache 迁移延迟(D 等 cache 到位非 P 阻塞 D)/D 池饱和排队(D 自己过载)/P 池饱和拖整体/跨机带宽瓶颈(规模化痛点催生池化)；监控别误判 P 阻塞 D 而错优化）

### [[GPU utilization]]

- [[GPU utilization]] — gpu-memory-utilization 为何设 0.8/0.9 不设 1.0？要留给谁？（**行内型**：vLLM `--gpu-memory-utilization` 指 KV cache 池占比非算力利用率；不设 1.0 是留运行时临时开销——CUDA context+allocator 元数据 300-600MB、forward 激活(prefill 长序列峰值高)、cuBLAS/FlashAttention workspace、临时张量/logits($B×V$ 对大词表可观)、speculative draft、其他进程、峰值尖刺；util=1.0 启动没事一跑长 prompt/大 batch 立刻 OOM；推荐值表 独占 0.90/大模型 TP 0.85-0.88/长 context 0.80-0.85/混部 0.80）
- [[GPU utilization]] — 为什么推理 OOM 反而要降低 gpu 利用率？（**行内型**：KV 池是启动时一次性预分配刚性大块,util 越高池越大把地占死,forward 要激活/workspace 抢不到→OOM；降 util=主动缩 KV 池给运行时临时腾地方,代价是并发数/max context/吞吐↓；本质是显存预算重分配=少给 KV 喂吞吐、多给临时喂不崩,属 batching tradeoff 显存维；排错顺序=先查权重太大该上 TP 没/同卡多实例/context 峰值/`--swap-space` 都治完还 OOM 再降 util）

## 八、推理系统 / 高级推理技术

### [[beam search]]

- [[beam search]] — beam search 用在哪？什么是序列解码 decoding？这是在加速 decoding 吗？（**行内型**：①用在序列生成质量优先场景——NMT/摘要/image captioning/ASR/结构化输出(SQL/代码),LLM 时代对话创作被 sampling 取代退守窄场景；②序列解码=从模型每步输出分布里挑 token 拼序列的后处理,argmax 贪心/sampling/beam/speculative 是不同 decoding 策略,同模型同权重不同策略产出不同输出；③**不是加速反而慢 B 倍**——beam 每步算 B 条候选剪枝是拿时间换质量近似全局最优,加速的是 speculative decoding(一次 forward 多 token 保分布无损),二者正交 beam 是搜索提质 spec 是采样加速；防混口诀"beam=慢但更准的搜索,speculative=快且无损的加速"）

### [[speculative decoding]]

- [[speculative decoding]] — 联网搜索该领域新进展 verl-speco/eagle/medusa 等（**行内型+联网**：2024-2026 演进主线=独立 draft→轻量 head→预测 feature→动态树+训练时对齐。方法家族表(vanilla 2-2.8× / Medusa K 并行头 1.8-2.5× 正被取代 / EAGLE-1 预测 hidden state 2.5-3× / EAGLE-2 动态树 3-3.5× / EAGLE-3 改回 token+多层 hidden 聚合+training-time test 3.5-4×(13B 5.6×) 2026 生产默认 / EAGLE-3.1 vLLM+TorchSpec 协作 2026-05 / NanoSpec 动态词表裁剪 40× 再省 51.6%)；另有 n-gram 零成本首试/self-speculation 跳层/streaming 重叠/MTP(DeepSeek V3/Llama4 原生)/Suffix+LSTM。2026 生产选型=新部署直接 EAGLE-3(预训练 head 拿来即用/vLLM 一行 --speculative-config/几乎零额外 VRAM/接受率最高),决策树 n-gram→EAGLE→draft model→Medusa。框架表(vLLM EAGLE-3/3.1 主推/SGLang/TensorRT-LLM Dynamic Tree+SA/HF/llama.cpp)。**RL rollout 新热点**(用户特别问)：verl PR#5925 EAGLE-3 加速 rollout ~15% 关键坑=weight sync 后重载 draft+重建 RoPE cache；**verl-SpeCo**(verl-speco,2026-06 pre-release)=co-training framework 训练时协同更新 draft 保持 policy 演化对齐；Decoupled Speculation(arXiv:2511.16193) draft/verify 异步流水线 verl issue#5559；MTP 训推一体 PR#6432；Speculator 训练框架 PR#4947。与训推一致性交汇=draft 须随 θ 对齐否则 acceptance 崩+logp 污染 TIM；无损性边界=经典无损但 draft 冻结+target 演化破坏保证。来源 EAGLE-3 arXiv:2503.01840/vLLM blog 2026-05/verl-SpeCo GitHub/verl PR#5925#5559#4947/NanoSpec arXiv:2605.26444/localaimaster/inference.net/vizuara 2026-07-12 核实）
- [[speculative decoding]] — 细说 EAGLE-3：什么是 target / 什么是轻量 draft head / 什么是 hidden state / 为什么一次能验证多分支（**行内型+联网**：四问四答。①target=被加速的大模型本身(Llama-3.1-8B/70B),最终裁判+给 draft 喂 hidden state,输出分布严格=p 保证无损;vanilla 的 target 配独立小模型,EAGLE 的 target 配自家上的轻量 head 一体。②轻量 draft head=target 上挂的 1 层 Llama decoder+FC 投影+小 LM head,num_hidden_layers=1,checkpoint ~1.2GB(vs 8B target ~16GB,1/13),复用 target hidden state 作输入,词表缩小到 32k,预训练 head 拿来即用;轻量因只1层/参数几十之一/复用target/小词表。③hidden state=transformer 每层 forward 后每位置产出的连续向量(hidden_size 维),是内部表示;EAGLE-1 只用顶层 feature(只承载下token信息),EAGLE-3 抓 target 早/中/晚3层(如{2,N//2,N-3})hidden concat 3k 维→FC→k 维 g 融合低中高语义+training-time test 训练时模拟多步自回归喂 draft 输出 a 当下一步输入抗误差累积;连续平滑好预测=EAGLE 接受率 70-80% vs vanilla 50% 的根因。④一次验多分支=tree attention mask:整棵候选树 token 拼成一条序列一次 forward,mask[u,v]=0 当且仅当 v 是 u 的祖先(含自身)否则 -inf,每 token 只看自己祖先链不跨分支泄漏,等价于各自正确上下文,transformer 一次矩阵乘并行算出所有分支位置 logits 再接受-拒绝挑最长前缀;含可跑的 8 节点树 mask 构造代码。tree decoding 取舍:低并发最快但高 QPS 时验证 64 候选只为多接受1-2 token 不划算,vLLM 默认 greedy/TRT-LLM 用 dynamic tree 权衡。来源 arXiv:2503.01840§3.1/vLLM speculators eagle3 docs/HF thoughtworks GLM-4.7 config/Red Hat 2025-07/EAGLE-Pangu arXiv:2603.08088§2.4/TRT-LLM PR#12062 2026-07-12 核实）

---

## 十、性能分析与系统优化 / 监控指标

### [[常见指标]] / [[RLHF训练监控指标详解]]

- [[RLHF训练监控指标详解]] — 这些都是什么指标 详细展开 补充完整 新起一个md（critic/rewards/mean、actor/pg_loss、actor/grad_norm、actor/kl_loss、response_length/mean、actor/pg_clipfrac、actor/ppo_kl、critic/advantages/mean、training/rollout_actor_probs_pearson_corr）（**新建型**：逐条拆"是什么/怎么算/看什么/坏了说明什么"；PPO总目标公式+GAE+Pearson推导；verl风格最小复现代码；三KL区分表(ppo_kl/kl_loss/kl_ref)；健康绿区参考表+调参联动决策表；与异步staleness治理对应；另附verl/OpenRLHF/TRL命名对照）

---

## 九、Ray / 分布式调度 / 工程问题

### [[placement group]]

- [[placement group]] — 如何避免 pg 造成碎片 比如两个 actor 各占 4 卡结果各自占了一个完整的 8 卡 node（**行内型**：根因是 SPREAD/默认分散 + 两个独立 actor 各 first-fit，调度器看不到它俩关系；碎片率公式 $\eta=\sum\text{已用}/\sum\text{节点}$，你那 50% 空着；解法=一个 PG 含 2 个 4-GPU bundle + STRICT_PACK 强制同节点装箱 4+4=8 填满；verl 实证 `RayWorkerGroup` 默认 `bin_pack=True`→`STRICT_PACK` 把同 pool worker 装箱；五条工程经验表；4 条误区含"PG 也可能造碎片""调度器不会自动凑满"）

### [[Ray与分布式调度]]

- [[Ray与分布式调度]] — 各个 actor 之间怎么传递信息？需要传递信息吗？主控是怎么控制各个 actor 的？（**行内型**：两条正交通道——张量数据走 NCCL 集体通信、控制信号/小对象走 Ray Object Store；需要传，RLHF 多角色流水线每步都跨 actor 交数据；主控=single-controller 模式，driver 进程持 RayWorkerGroup handle，靠 `actor.method.remote(args)` + `ray.get()` 编排时序，actor 内部各自起 torch.distributed 组做张量同步；verl RayWorkerGroup.execute_all_async/_execute_remote_single_worker 代码实证；MASTER_ADDR/PORT 由 driver 通过 env_vars 注入 actor）
- [[Ray与分布式调度]] — 这里能展开讲讲吗 verl 的代码在 ray_trainer.py（**行内型**：逐段拆 verl RayPPOTrainer——`__init__` driver 不持卡只存 role_worker_mapping/resource_pool_manager 等元数据；`init_workers()` 四步建 actor=资源池→RayClassWithInitArgs→create_colocated_worker_cls 多 role 塞同进程省显存→spawn 拆逻辑 wg；`fit()` 主循环 rollout→sleep_replicas→reward→old_log_prob→ref_log_prob→values→adv(本地)→update_critic→update_actor→save_ckpt→update_weights(weight sync)，driver 只 `.remote().get()` RPC 编排重活全在 actor；5 条关键观察 + 代码行号）

### [[Ray调试手段]]（新建笔记）

- [[Ray调试手段]] — 调试手段新建一章节吧 可以联网 ray debugger？（**新建型**：五层工具链——①Ray Dashboard(`8265` Web UI，Overview/Cluster/Jobs/Logs/Metrics/Events/Serve view) ②Ray Distributed Debugger(2026-04 GA VSCode 扩展，跨 actor 交互式断点 attach) ③ray timeline(Chrome tracing/Perfetto 看调度排队/搬运/exec 时序) ④ray list state APIs(`ray list actors/tasks/nodes/cluster-events` CLI) ⑤代码级 `ray.get(timeout)` + RayTaskError/GetTimeoutError 异常；调度延迟公式各项对应工具；verl RLHF 调试 5 步流程；与 torch profiler/nsys 四层分工表；8 条误区；来源 Anyscale blog 2026-04/05、Ray docs 2.55/2.56）

### [[actor deadlock]]

- [[actor deadlock]] — 细说一下规避方法（**行内型**：六种规避——①避免循环依赖/无环 DAG ②single-controller 主控编排消除多主控互等 ③async actor `max_concurrency>1` 线程池并发破死锁 ④不要嵌套 `ray.get`/future 直接传递不阻塞 ⑤`ray.get(timeout)` + 超时熔断 ⑥callback/continuation `.continue_with` 不等返回 ⑦watchdog 周期性健康探测自愈；每条含机制/代码/适用场景/RLHF 落点；优先级口诀"无环>单主控>异步>非阻塞>超时>回调>看门狗"）
- [[actor deadlock]] — `max_concurrency` 是什么意思？（**行内型**：Ray actor 并发度参数，决定同一 actor 同时能跑几个方法调用；默认 1 串行保证状态一致但易死锁/低吞吐；设 N 配 N 线程池并发执行，循环等待被打破不死锁但并发改状态须自己加锁、一致性责任从 Ray 转开发者；取值表（训练 actor 1/推理 10-100/编排 1/需响应回调 >1）；更优解是 asyncio actor 协程并发不阻塞又不数据竞争；RLHF 推理常调大训练保默认，verl single-controller 无环设计使大多数 actor 不需调大）

---

## 十、性能分析与系统优化 / GPU性能

### [[memory bandwidth]]

- [[memory bandwidth]] — 联网查询一下 HBM 规格表正确性（**行内型+联网**：原表四行(V100 900/A100 2039/H100 3350/H200 4800)全部正确，来源 NVIDIA 官方 datasheet；订正=A100 应区分 40GB(1555 GB/s)/80GB(2039 GB/s)；2025 新增 Blackwell 应补入——B200 192GB HBM3e 8.0TB/s、B300 288GB HBM3e 8.0TB/s、NVLink 1.8TB/s；2026-07-12 核实；带宽墙从 A100 2039→B200 8000 GB/s，7B decode 理论上限 145→571 tokens/s 但仍 memory-bound 因算术强度 I=2 不变）

---

## 七、分布式系统与并行计算 / 通信机制

### [[all-reduce]]

- [[all-reduce]] — 普通信号适合用 all-reduce 传递吗？性能影响如何？（**行内型**：不适合——all-reduce 是"大块张量归约"原语非"传消息"原语；语义不对(归约抹原值,要原值用 broadcast)、性能浪费(小消息 launch 开销 10-100μs 主导,大炮打蚊子)、同步过重(集合通信自带 barrier 绑架异步)、NCCL 只管 GPU tensor CPU 信号该走 gloo/Ray；正确做法表=状态用 broadcast/点对点用 send-recv/CPU 用 gloo·Ray·RPC/指标聚合才用 all-reduce；口诀"要原值用 broadcast,要归约用 all-reduce"）

---
相关: [[整体目录]]、[[张量与自动微分]]、[[优化器]]、[[训练工程基础]]、[[Transformer基础]]、[[03-PyTorch与框架工程]]
