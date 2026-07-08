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

---

## 十一、显存与内存系统 / 资源核算

### [[FLOPs计算]]

- [[FLOPs计算]] — 前向 2 反向 4 怎么得出来的？（矩阵乘 2mn=2N_W 前向每参数 2 FLOP；反向算输入梯度+权重梯度两个 2N_W=4N_W；合计 6ND；直觉=反向算两次乘加是前向 2 倍）

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

### [[asynchronous training]]

- [[asynchronous training]] — 联网搜业内最新动态，提名 areal/asyncflow/fullyasync（**行内型+联网**：三大主流系统——AReaL(arXiv:2505.24298, 蚂蚁+清华, 全异步+decoupled PPO 容忍8步旧, 2.77×)、AsyncFlow(arXiv:2507.01663, verl-recipe, TransferQueue+CheckpointEngine, 3.81×)、verl fully_async_policy(主仓4模式 on-policy/stream/async-stale/partial-rollout, 2-2.35×)；另附 StreamRL/Magistral/StaleFlow(2026)/Laminar(EuroSys2026)；趋势=全异步解耦+staleness主动治理+数据中枢+权重sync工程化+agentic原生）
- [[asynchronous training]] — KL这是什么（**行内型**：KL=Kullback-Leibler divergence 相对熵 D_KL(P‖Q)=ΣP log(P/Q)，非负不对称；RL/PPO 里三角色=①策略漂移监控 π_θ vs π_old ②target_kl 早停超则break ③KL penalty/控制 -β·KL(π_θ‖π_ref) 软约束；ratio clip 是单步防飞阀、KL 是累计漂移仪表，异步 stale 让 KL 更易涨故监控更关键；详见 [[KL penalty与KL control]]）

---

## 十、性能分析与系统优化 / 监控指标

### [[常见指标]] / [[RLHF训练监控指标详解]]

- [[RLHF训练监控指标详解]] — 这些都是什么指标 详细展开 补充完整 新起一个md（critic/rewards/mean、actor/pg_loss、actor/grad_norm、actor/kl_loss、response_length/mean、actor/pg_clipfrac、actor/ppo_kl、critic/advantages/mean、training/rollout_actor_probs_pearson_corr）（**新建型**：逐条拆"是什么/怎么算/看什么/坏了说明什么"；PPO总目标公式+GAE+Pearson推导；verl风格最小复现代码；三KL区分表(ppo_kl/kl_loss/kl_ref)；健康绿区参考表+调参联动决策表；与异步staleness治理对应；另附verl/OpenRLHF/TRL命名对照）

---

## 七、分布式系统与并行计算 / 通信机制

### [[all-reduce]]

- [[all-reduce]] — 普通信号适合用 all-reduce 传递吗？性能影响如何？（**行内型**：不适合——all-reduce 是"大块张量归约"原语非"传消息"原语；语义不对(归约抹原值,要原值用 broadcast)、性能浪费(小消息 launch 开销 10-100μs 主导,大炮打蚊子)、同步过重(集合通信自带 barrier 绑架异步)、NCCL 只管 GPU tensor CPU 信号该走 gloo/Ray；正确做法表=状态用 broadcast/点对点用 send-recv/CPU 用 gloo·Ray·RPC/指标聚合才用 all-reduce；口诀"要原值用 broadcast,要归约用 all-reduce"）

---
相关: [[整体目录]]、[[张量与自动微分]]、[[优化器]]、[[训练工程基础]]、[[Transformer基础]]、[[03-PyTorch与框架工程]]
