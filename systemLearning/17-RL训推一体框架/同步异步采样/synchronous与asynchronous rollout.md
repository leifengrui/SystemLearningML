# synchronous与asynchronous rollout

> **所属章节**: [[同步异步采样]]
> **所属模块**: [[17-RL训推一体框架]]
> **别名**: 同步 / 异步 rollout / partial rollout / off-policy rollout / RL 采样调度模式 / rollout-trainer 时序协同 / staleness-aware sampling / decoupled generation and training / RL pipeline overlap
> **难度**: 高（需懂 [[asynchronous training]]、[[RL权重同步]]、[[importance sampling与off-policy correction]]、[[stale policy problem]]、[[colocated与disaggregated部署]]、[[trainer与rollout引擎组合]]、[[trajectory generation]]、[[rollout worker]]、[[sampling throughput]]）

> **所属小节**: §84


## 1. 一句话定义

**synchronous 与 asynchronous rollout（同步 / 异步采样）** 是 [[17-RL训推一体框架]] 中 **rollout（采样）端与 trainer（训练）端在时间轴上的四种协同模式**——① **synchronous（同步）**：rollout 采完**一整组**样本后设置 barrier，trainer 才更新一步，更新完再 sync 权重、采下一组；两端**串行不 overlap**，importance sampling ratio $\rho\approx1$（staleness $k\approx1$）最稳，但**两端 GPU 互相闲置**（rollout 采时 trainer 空转、trainer 训时 rollout 空转，aggregate 利用率常只有 ~50%）。② **asynchronous（异步）**：rollout 与 trainer **解耦并行**——trainer 训第 $N$ batch 时 rollout 池同时采第 $N+1$ batch（[[colocated与disaggregated部署]] disaggregate + [[RL权重同步]] overlap），吞吐高（理想可达 2×、AReaL 实测 2.77×、AsyncFlow 1.59×），但 trainer 用的是**异步到达的旧 trajectory**（staleness $k>1$ 步，rollout 用 $\theta_{old-k}$ 采），ratio $\rho$ 指数偏离 1、需 [[importance sampling与off-policy correction]] 修正 + clip/V-trace 兜底。③ **partial rollout（部分采样）**：不要求一整组 rollout **全部完成**才训练，**凑够有效样本就训**，长短样本混 batch 做负载均衡，batch 内可能**跨多个 $\theta$ 版本**（混合 staleness，需每条记版本号独立算 ratio）。④ **off-policy（离策略采样）**：rollout **显式用旧 $\theta$ 采**、trainer **用 ratio $\rho=\pi_\theta/\pi_{old}$ 修正**，本质就是 importance sampling——这是让"旧数据训新策略"数学合法的根，**不是与前三者并列的第四种调度模式，而是异步（乃至多 epoch 复用）赖以成立的数据语义**。本条是 **rollout-trainer 时序协同的专章**，与 [[asynchronous training]] 分工——后者讲**训练侧异步**（trainer/optimizer/NCCL 等训练组件的非阻塞），本条讲 **rollout 与 trainer 两个角色之间的协同时序**（采与训怎么排队、谁等谁）。staleness 的数学根（ratio 随 $k$ 指数爆）详见 [[importance sampling与off-policy correction]] §4.4、[[RL权重同步]] §4.2，本条侧重**工程时序与吞吐**。

> [!note] 三句话定位
> - **是什么**：四种 rollout-trainer 时序——同步（串行 barrier，稳但 idle ~50%）、异步（并行 overlap，快但 stale $k>1$ 需 IS）、partial（凑够就训、跨版本混 batch）、off-policy（旧 θ 采 + ratio 修正，是前三者复用数据的数学根）。
> - **为什么**：LLM rollout 占墙钟 50~80%，同步让两池 GPU 互相闲置、吞吐浪费；异步 overlap 提吞吐但 staleness 让 ratio 爆，靠 PPO clip + 频繁 sync 兜底。是 throughput↔staleness 的核心 trade-off。
> - **与 [[asynchronous training]] 关系**：后者讲**训练侧**异步（trainer 内部组件非阻塞），本条讲 **rollout↔trainer 之间**的协同时序。staleness→ratio 爆的数学详见 [[importance sampling与off-policy correction]]、[[stale policy problem]]。


## 2. 为什么需要它（动机与背景）

### 2.1 LLM-RL 的墙钟瓶颈：rollout 太贵

LLM-RL 一次训练 step 的墙钟时间主要由两段组成：

- **rollout（采样）**：用当前 $\theta_{old}$ 跑 vLLM/SGLang，对 $N$ 个 prompt 各采 $G$ 条 response（GRPO group size），每条几百~几千 token forward。常占墙钟 **50~80%**。详见 [[trajectory generation]]、[[sampling throughput]]。
- **train（训练）**：trainer 用 $\theta_{new}$ forward+backward 算 ratio/advantage/clip loss，更新 $\theta$。占 20~50%。

**同步模式**把这两段**串行**：先采完一整组 → barrier → 训练 → barrier → sync 权重 → 采下一组。两端**不 overlap**，墙钟 $T_{step}=T_{rollout}+T_{train}$。问题在 **disaggregate 部署**（rollout 独立池、trainer 独立池）下：rollout 采时 trainer 池**空转**，trainer 训时 rollout 池**空转**，两池 GPU 互相闲置，**aggregate 利用率常只有 ~50%**（见 §4.1）。这是同步模式最大的浪费。

### 2.2 异步的诱惑：overlap 提吞吐

若把 rollout 与 trainer **解耦并行**——trainer 训第 $N$ batch 时 rollout 池同时采第 $N+1$ batch——两端 overlap，墙钟 $T_{step}\to\max(T_{rollout},T_{train})$（取较长那段）。若两段平衡（$T_{rollout}\approx T_{train}$），墙钟近乎**减半**，吞吐翻倍。这是异步的动机。

但代价立刻浮现：rollout 采第 $N+1$ batch 用的是**sync 之前的旧权重 $\theta_{old-k}$**（trainer 已更新 $k$ 步到 $\theta_{new}$），样本对 $\theta_{new}$ 是 **off-policy** 的，importance sampling ratio $\rho=\pi_{\theta_{new}}/\pi_{\theta_{old-k}}$ 随 staleness $k$ 指数偏离 1，方差爆、训练不稳。详见 [[stale policy problem]]、[[importance sampling与off-policy correction]] §4.4。

### 2.3 三段不可兼得：throughput / staleness / 稳定性

这是本条的核心 trade-off，三者互相拉扯：

| 追求 | 手段 | 代价 |
|---|---|---|
| **吞吐高** | 异步 overlap（disaggregate + 训推并行） | staleness $k$ 大 → ratio 爆 → 不稳 |
| **staleness 小** | 频繁 [[RL权重同步]]（sync 频率 $S$ 小） | sync 开销重（跨机 140GB broadcast）拖吞吐 |
| **稳定（ratio≈1）** | 同步串行（每步采最新 θ） | 两端 idle ~50%，吞吐浪费 |

同步最稳但慢；异步最快但 stale；频繁 sync 降 stale 但拖慢。**没有免费午餐**，工程就是在三者间找甜区——这也是 AReaL/AsyncFlow/verl fully_async 等系统存在的意义（用系统优化把甜区往三高推）。

### 2.4 partial：长尾样本的负载均衡

即便同步，一整组 rollout 里**长短样本极不均**（数学题有的 200 token 解完、有的写到 4096 token 还没完），同步 barrier 要等**最长那条**采完才能训——长尾样本拖死整组吞吐。partial rollout 解耦这个：**不要求一整组全完成**，凑够有效样本（够一个 batch）就训，长短样本混 batch 均衡负载，且可边采边训（rollout 不等长尾）。但 batch 内样本可能**跨 $\theta$ 版本**（前半采时 $\theta_{old-k}$、sync 后后半采时 $\theta_{old}$），需每条记版本号独立算 ratio（详见 §3.3）。

### 2.5 off-policy：让"旧数据训新策略"合法

异步 / partial / 多 epoch 复用本质都是"用旧 $\theta$ 采的数据训新 $\theta$"。若没有 importance sampling 的 ratio 修正，这是**有偏**的——梯度对 $\theta_{new}$ 算但样本来自 $\theta_{old}$。IS 的恒等式 $\mathbb{E}_{\pi_\theta}[f]=\mathbb{E}_{\pi_{old}}[\rho f]$（$\rho=\pi_\theta/\pi_{old}$）让旧数据可无偏估新策略期望，是异步能数学成立的根。故 off-policy 不是与同步/异步/partial 并列的"第四种调度模式"，而是**前三者（尤其异步）赖以成立的数据语义**——同步是近似 on-policy（$k\approx1$，$\rho\approx1$），异步是重度 off-policy（$k>1$，$\rho$ 偏离大需 IS）。详见 [[importance sampling与off-policy correction]]。


## 3. 核心概念详解

### 3.1 四种模式总览

| 维度 | ① synchronous | ② asynchronous | ③ partial rollout | ④ off-policy |
|---|---|---|---|---|
| **时序** | rollout→barrier→train→barrier→sync→rollout…（串行） | rollout ∥ train（并行 overlap） | 凑够就训、边采边训（不强求整组完成） | 旧 θ 采 + ratio 修正（数据语义） |
| **staleness $k$** | $\approx1$（每步采最新 θ） | $>1$（rollout 落后 trainer $k$ 步） | 混合（batch 内跨版本） | $>0$（显式承认旧 θ） |
| **ratio $\rho$** | $\approx1$（小偏离） | 指数偏离 1（随 $k$ 爆） | 每条独立（按版本号算） | $\pi_\theta/\pi_{old}$（IS 本意） |
| **IS 修正** | 几乎不需要（clip 基本不触发） | 必须（clip/V-trace 兜底） | 必须（每条独立 ratio） | 是其定义本身 |
| **吞吐** | 低（两池 idle ~50%） | 高（overlap，理想 2×） | 中高（长尾不拖死整组） | 取决于调度（off-policy 本身不决吞吐） |
| **GPU 利用率** | ~50%（disaggregate） | ~100%（overlap） | 中~高 | — |
| **稳定性** | 最稳 | 需 clip + 频繁 sync 兜底 | 需版本管理 | 靠 IS + clip |
| **典型实现** | verl/OpenRLHF 默认 colocate | AReaL（fully async）、AsyncFlow、verl `fully_async` | OpenRLHF partial/async rollout | PPO/GRPO 全是轻度 off-policy + clip |

> [!note] 补充：四者不是正交并列
> 严格说，**①②③ 是"调度时序"轴**（rollout 与 trainer 怎么排队），**④ 是"数据语义"轴**（样本是 on/off-policy、要不要 IS 修正）。同步是近似 on-policy（$\rho\approx1$），异步 / partial / 多 epoch 复用都是 off-policy（$\rho$ 偏离需 IS）。把 ④ 单列是因为它解释了 ②③ 为什么能数学成立——没有 IS，异步采的旧数据训新策略就是有偏的。详见 §2.5、[[importance sampling与off-policy correction]]。

### 3.2 ① synchronous（同步）：barrier 串行，最稳最浪费

```
时间轴 →
rollout:  [采 batch N 采完]          [采 batch N+1 采完] ...
trainer:                [训 batch N 训完][sync]          [训 batch N+1]...
          ↑barrier(采完才训)          ↑barrier(训完才采)
```

- rollout 采整组 → **barrier**（采完才训）→ trainer 更新 → **barrier**（训完才 sync/采）→ [[RL权重同步]] → 采下一组。
- 两端**严格串行不 overlap**。
- rollout 用每步 sync 后的最新 $\theta_{old}$ 采，staleness $k\approx1$（只差 trainer 这一更新步），ratio $\rho\approx1$，clip 基本不触发，**最稳**。
- 代价：disaggregate 两池 GPU **互相闲置**——rollout 采时 trainer 池空转、trainer 训时 rollout 池空转，aggregate 利用率 ~50%（若两段平衡）。colocate 同一组 GPU 分时虽不"空转"（总在跑），但**无法 overlap**，墙钟仍是 $T_{rollout}+T_{train}$，相对异步仍慢。

> [!warning] 误区：colocate 同步就 100% 利用
> colocate 同步同一组 GPU 分时跑 rollout 与 train，物理上 GPU 总在算（不空转），但**两阶段无法 overlap**——rollout 时"训练算力"概念上闲置，train 时"rollout 算力"闲置。墙钟仍是串行 $T_r+T_t$，比异步 overlap 慢。利用率高 ≠ 墙钟短。详见 [[colocated与disaggregated部署]] §2.1。

### 3.3 ② asynchronous（异步）：解耦并行，快但 stale

```
时间轴 →
rollout:  [采 batch N+1 ...............]   (用 θ_old-k 采, 与 trainer 并行)
trainer:  [训 batch N ...........] [训 batch N+1 ...]
sync:           ↑每 S 步 sync 一次 θ -> rollout
```

- trainer 训第 $N$ batch 时，rollout 池**同时采**第 $N+1$ batch（disaggregate 训推 overlap）。
- 两端**并行**，墙钟 $\to\max(T_{rollout},T_{train})$，吞吐大幅提升（理想 2×）。
- 但 rollout 采第 $N+1$ 用的是 **sync 之前的旧权重 $\theta_{old-k}$**（trainer 已更新 $k$ 步到 $\theta_{new}$），样本对 $\theta_{new}$ 是 off-policy，ratio $\rho=\pi_{\theta_{new}}/\pi_{\theta_{old-k}}$ 随 $k$ 指数偏离 1（推导见 §4.2）。
- 必须 [[importance sampling与off-policy correction]] 修正 + PPO clip $\varepsilon$ / V-trace $\bar\rho$ 兜底防方差爆。详见 [[stale policy problem]]、[[RL权重同步]] §2.2。
- $k$ 上界 = sync 间隔 $S$（两次 sync 间隔步数）。频繁 sync 降 $k$ 但 sync 开销重，需权衡。colocate $S=1$（in-process sync 几乎零开销）；disaggregate $S\in[1,8]$。

> [!tip] 实践：staleness 阈值
> 工程上常设 **staleness 上界 $k_{\max}$**（如 AReaL/AsyncFlow 控制数据 staleness），超过阈值的旧样本丢弃或降权，防止极端 stale 爆 ratio。这是"staleness-enhanced PPO"的工程化——不只用 clip 事后兜底，还在调度层**限制最大 $k$**，从源头控 off-policy 程度。详见 §8.1。

### 3.4 ③ partial rollout（部分采样）：凑够就训，长短均衡

```
时间轴 →
rollout:  [短样本 a 完] [短样本 b 完] ............ [长尾样本 z 完(很晚)]
partial:  |凑够 N 条→ 立即训 batch1 |  |再凑够→ 训 batch2|
          ↑不强求整组全完, 凑够就训, 不等长尾 z
```

- **不要求一整组 rollout 全部完成**才训练——凑够一个 batch 的有效样本就训。
- 解决**长尾样本拖死整组**：同步 barrier 要等最长那条采完，长尾（如 4096 token 的数学推导）拖死短样本（200 token）的吞吐。partial 凑够就训，长短混 batch 均衡负载。
- 但 batch 内样本可能**跨 $\theta$ 版本**：前半采时 $\theta_{old-k}$，sync 后后半采时 $\theta_{old}$。需**每条 response 记采样时的 $\theta$ 版本号**，trainer 按版本号算对应 ratio $\rho=\pi_{\theta_{new}}/\pi_{\theta_{old-version}}$（混合 staleness）。详见 [[importance sampling与off-policy correction]] §8（多版本样本混 batch）。
- 常与**动态采样**（[[sequence packing与动态采样]]）配合：partial 凑够就训 + 动态过滤全对/全错组，保证进 batch 的都是有效（advantage 非零）样本。

### 3.5 ④ off-policy（离策略采样）：旧 θ 采 + ratio 修正

$$
\rho_t=\frac{\pi_{\theta_{new}}(o_t\mid q,o_{<t})}{\pi_{\theta_{old}}(o_t\mid q,o_{<t})}=\exp\!\big(\log\pi_{\theta_{new}}(o_t)-\log\pi_{\theta_{old}}(o_t)\big)
$$

- rollout 用旧 $\theta_{old}$ 采，trainer 用 $\theta_{new}$ 训，靠 ratio $\rho=\pi_\theta/\pi_{old}$ 把旧策略采的样本重加权给新策略（IS 恒等式，推导见 [[importance sampling与off-policy correction]] §4.1）。
- 这是让"旧数据训新策略"数学合法的根——没有 IS，异步 / 多 epoch 复用就是有偏的。
- PPO clip $\min(\rho\hat A,\text{clip}(\rho,1\pm\varepsilon)\hat A)$ 截断 ratio 防 IS 高方差爆；V-trace $\bar\rho=\min(\rho,\bar\rho)$ 在异步 staleness 下进一步截顶。
- **同步也是轻度 off-policy**：即便每步 sync，rollout 采那一刻是 $\theta_{old}$、trainer 训完变 $\theta_{new}$，差一步，$\rho\ne1$。同步只是把 $k$ 压到 1，不是消除 off-policy。详见 [[RL权重同步]] §2.1 误区。

> [!note] 补充：ratio 偏离的两种来源
> ratio $\rho\ne1$ 有两种原因：①**正确偏离**（$\theta_{new}\ne\theta_{old}$，staleness 本意，IS 该修）；②**错误偏离**（$\theta_{new}=\theta_{old}$ 但 [[rollout train reference logprob一致性]] 的 TIM 让同 $\theta$ 下 logp 算错，污染）。前者靠 sync 降、靠 IS 修；后者靠精度/recompute 治。异步放大的是前者（$k$ 大偏离大），与后者正交。详见 [[importance sampling与off-policy correction]] §3.7。

### 3.6 与部署模式的耦合

四种采样模式与 [[colocated与disaggregated部署]] 耦合：

- **colocate + 同步**：同一组 GPU 分时跑 rollout/train，in-process sync 极快（$S=1$），staleness $\approx0$。但无法 overlap，墙钟串行。中小模型主流。
- **disaggregate + 异步**：rollout 独立池 ∥ trainer 独立池，overlap 提吞吐。但跨机 sync 重 + stale。70B+ / R1 范式主流。
- **disaggregate + partial**：rollout 独立池边采边喂，trainer 凑够就训。OpenRLHF partial/async rollout、verl fully_async 模式都是 disaggregate + overlap 的工程化。详见 [[colocated与disaggregated部署]] §8.3。

部署模式（角色放哪）与采样模式（采训怎么排队）正交组合出具体系统。


## 4. 数学原理 / 公式

### 4.1 推导：同步的 GPU 闲置率

设 disaggregate 两池：rollout 池 $N_r$ 卡（采 $T_r$ 时间）、trainer 池 $N_t$ 卡（训 $T_t$ 时间），同步串行 $T_{step}=T_r+T_t$。

任意时刻，**只有一池在算**（另一池空转）：

- rollout 阶段：trainer 池 idle，利用率 $0$；rollout 池满载。
- train 阶段：rollout 池 idle，利用率 $0$；trainer 池满载。

**aggregate GPU 利用率**（两池总卡数 $N_r+N_t$ 上平均）：

$$
U_{sync}=\frac{N_r T_r+N_t T_t}{(N_r+N_t)(T_r+T_t)}
$$

若两段平衡 $T_r=T_t$ 且 $N_r=N_t$：$U_{sync}=\frac{N_r T_r+N_r T_r}{2N_r\cdot 2T_r}=\frac{2N_r T_r}{4N_r T_r}=\boxed{0.5}$

即**同步两池平衡时 aggregate 利用率仅 50%**——一半卡在空转。这是同步最大的浪费。

### 4.2 推导：异步的 overlap 加速与 staleness 代价

异步 overlap 后，两端并行，墙钟由较长段决定：

$$
T_{async}=\max(T_r,T_t)
$$

**加速比**（相对同步串行）：

$$
\text{speedup}=\frac{T_{sync}}{T_{async}}=\frac{T_r+T_t}{\max(T_r,T_t)}
$$

- 两段平衡 $T_r=T_t=T$：$\text{speedup}=\frac{2T}{T}=2$（理想翻倍）。
- rollout 主导 $T_r\gg T_t$：$\text{speedup}\approx\frac{T_r}{T_r}=1$（rollout-bound，异步帮助小——这时该扩 rollout 池而非异步）。

**staleness 代价**：trainer 在 $T_r$ 时间内能更新 $k$ 步（每步 $T_t/k$），rollout 采的样本落后 trainer $k$ 步。一阶 Taylor 展开 ratio（与 [[importance sampling与off-policy correction]] §4.4、[[RL权重同步]] §4.2 一致）：

设每步梯度方向近似一致 $g$，$k$ 步后 $\theta_{new}\approx\theta_{old}+k\cdot g$：

$$
\log\pi_{\theta_{new}}(o_t)-\log\pi_{\theta_{old}}(o_t)\approx(\theta_{new}-\theta_{old})^\top\nabla_\theta\log\pi_{\theta_{old}}(o_t)\approx k\cdot g^\top\nabla_\theta\log\pi_{\theta_{old}}(o_t)
$$

$$
\boxed{\;\rho_t=\exp\!\big(k\cdot g^\top\nabla_\theta\log\pi_{\theta_{old}}(o_t)\big)\;}
$$

ratio 对数偏离 $\propto k$，**staleness 越大 ratio 指数偏离 1 越剧烈**。方差 $\mathrm{Var}[\rho]\propto\mathbb{E}[\rho^2]\sim\exp(2k\cdot\ldots)$ 随 $k$ 指数爆。这就是异步必须靠 clip/V-trace 截断 + 频繁 sync 降 $k$ 的数学根。

> [!tip] 实践：AReaL 为何能 2.77×（超 2× 理论上限）
> 纯 overlap 上限是 2×（两段平衡）。AReaL 报 2.77× 说明还有**系统级优化**叠加：① 完全解耦 generation/training（rollout 不等任何 barrier）；② workload balancing（控 data staleness 让 rollout/trainer 速率匹配，减少一端等另一端）；③ staleness-enhanced PPO（容忍更大 $k$ 用更激进的调度）。即不止 overlap，还把两端各自压榨 + 凑配比，故超 2×。详见 §8.1。

### 4.3 推导：staleness 上界 $k_{\max}$ 与 sync 频率 $S$

两次 [[RL权重同步]] 之间，trainer 最多更新 $S$ 步，rollout 最多落后 $S$ 步：

$$
k_{\max}=S
$$

- $S$ 小（频繁 sync）：$k_{\max}$ 小，$\rho$ 偏离小（稳），但 sync 开销重（每 $S$ 步跨机 broadcast 140GB）拖吞吐。
- $S$ 大（少 sync）：sync 开销小（快），但 $k_{\max}$ 大，$\rho$ 指数爆（不稳），clip 频繁触发失真。

PPO clip $\varepsilon$ 是 staleness 的**事后兜底**：当 $\rho$ 超出 $[1-\varepsilon,1+\varepsilon]$ 截断防梯度爆。但截断过多（clip frequency 高）会失真。故 **sync 频率 $S$（源头控 $k$）+ clip $\varepsilon$（事后兜底）共同控制 off-policy 程度**。colocate $S=1$（sync 极快）；disaggregate 异步 $S\in[1,8]$ 权衡。详见 [[RL权重同步]] §4.2、[[importance sampling与off-policy correction]] §4.4。

### 4.4 推导：partial 的混合 staleness

partial 模式 batch 内样本跨 $\theta$ 版本。设 batch 有 $G$ 条 response，第 $i$ 条用 $\theta_{old-v_i}$ 采（$v_i$ 是该条的版本号，$v_i\in\{0,1,\ldots,k_{\max}\}$）。trainer 当前 $\theta_{new}$，每条独立 ratio：

$$
\rho_{i,t}=\frac{\pi_{\theta_{new}}(o_{i,t})}{\pi_{\theta_{old-v_i}}(o_{i,t})}=\exp\!\big(\log\pi_{\theta_{new}}(o_{i,t})-\log\pi_{\theta_{old-v_i}}(o_{i,t})\big)
$$

- 不同 $i$ 的 $v_i$ 可能不同（混合 staleness），ratio 不再统一 $\pi_\theta/\pi_{old}$，而是**每条按自己版本号算**。
- batch 期望 staleness $\bar v=\frac1G\sum_i v_i$，方差也受 $v_i$ 离散度影响。
- 工程需**版本号机制**：rollout 每条记采样时的 $\theta$ 版本号（weight version），trainer 按版本号取对应 $\log\pi_{old-v_i}$ 算 ratio。这是异步 IS 的复杂化。详见 [[importance sampling与off-policy correction]] §8。


## 5. 代码示例（可选）

### 5.1 异步 rollout-trainer 时序（概念，staleness buffer）

```python
import queue, threading

# 异步: rollout 池持续采, trainer 凑够就训, 用旧 θ (staleness=k)
traj_queue = queue.Queue(maxsize=4)  # rollout→trainer 的 trajectory 缓冲
weight_version = [0]  # 当前 rollout 用的 θ 版本号 (trainer 更新后 +1)

def rollout_worker():
    while True:
        traj = generate_trajectories(theta=rollout_theta)  # 用旧 θ_old-k 采
        traj["version"] = weight_version[0]  # 记采样时的版本号
        traj_queue.put(traj)  # 异步投递, trainer 凑够就消费
        if traj_queue.full():
            weight_version[0] += 1  # 模拟 sync 后版本 +1 (rollout 用新 θ 采下一批)

def trainer_worker():
    k = 0
    while True:
        traj = traj_queue.get()  # 取异步到达的旧 trajectory (可能落后 k 步)
        k = weight_version[0] - traj["version"]  # 该 batch 的 staleness
        if k > K_MAX:  # staleness 超阈值, 丢弃防 ratio 爆 (staleness-aware)
            continue
        ratio = compute_ratio(theta_new=trainer_theta,
                              old_logp=traj["old_log_prob"])  # ρ=π_θ_new/π_θ_old-k
        loss = ppo_clip_loss(ratio, traj["advantage"])  # clip 兜底
        trainer_theta -= lr * grad(loss)
        # 每 S 步 sync: trainer_theta -> rollout_theta (跨机 NCCL broadcast)

threading.Thread(target=rollout_worker, daemon=True).start()
threading.Thread(target=trainer_worker, daemon=True).start()
# -> rollout 与 trainer 并行, 吞吐 ↑; 但 traj 落后 k 步, 靠 ratio + clip 修
```

### 5.2 staleness 对 ratio 偏离的手算（与 [[RL权重同步]] §5.1 一致）

```python
import math

def ratio_vs_staleness(g_logp_grad, k):
    # g_logp_grad: 单步梯度对 logπ 的扰动 (近似常数)
    # k: staleness 步数 (rollout 落后 trainer k 步)
    log_rho = k * g_logp_grad  # log ρ ≈ k·g·∇logπ
    return math.exp(log_rho)

g = 0.05  # 单步扰动
for k in [1, 4, 8]:
    rho = ratio_vs_staleness(g, k)
    print(f"staleness k={k}: ratio={rho:.2f}  (偏离 1 {'小' if k==1 else '剧烈'})")
# staleness k=1: ratio=1.05  (colocate 同步, 偏离小, clip 不触发)
# staleness k=4: ratio=1.22  (disaggregate 异步, 偏离明显, clip 偶触发)
# staleness k=8: ratio=1.49  (重度异步, 偏离大, clip 频繁触发, 需 sync 降 k)
# -> 异步 staleness 让 ratio 随 k 指数爆, 频繁 sync 降 k + clip 兜底是异步稳定的关键
```

### 5.3 partial rollout 的混合版本 ratio（概念）

```python
def partial_batch_ratio(batch, theta_new_logp_fn):
    # batch 内每条 trajectory 带自己采样时的版本号 version (混合 staleness)
    for traj in batch:
        v = traj["version"]                       # 该条采时的 θ 版本
        old_logp = traj[f"logp_v{v}"]             # 取对应版本的 old_log_prob
        new_logp = theta_new_logp_fn(traj)       # trainer θ_new 重算
        traj["ratio"] = (new_logp - old_logp).exp()  # 每条独立 ρ=π_θ_new/π_θ_old-v
    # batch 内 ratio 不统一 (混合 staleness), 每条按版本号算
    return batch
```


## 6. 与其他知识点的关系

- **上游（依赖）**: [[trajectory generation]]（rollout 采的就是 trajectory）、[[rollout worker]]（采样执行体）、[[sampling throughput]]（rollout 算力特征决定 $T_r$）、[[colocated与disaggregated部署]]（部署模式决定能否 overlap）、[[RL角色拓扑]]（rollout 与 trainer 两角色）。
- **下游（应用）**: [[17-RL训推一体框架]] 全章（采训协同是骨架）、[[RL权重同步]]（sync 频率 $S$=staleness 上界 $k_{\max}$）、[[importance sampling与off-policy correction]]（异步靠 IS 修正旧数据）、[[stale policy problem]]（staleness 是 ratio 爆的根）、[[sequence packing与动态采样]]（partial + 动态采样联合优化）、[[continuous batching的调度实现]]（rollout 端的采样吞吐引擎）。
- **对比 / 易混**:
  - **本条 vs [[asynchronous training]]**：本条讲 **rollout↔trainer 之间**的协同时序（采与训怎么排队）；[[asynchronous training]] 讲**训练侧**异步（trainer/optimizer/NCCL/all-reduce 等训练组件的非阻塞、用 future/async API 让 backward 与 next forward overlap）。两者正交——训练侧异步是 trainer 内部优化，rollout 异步是采训两角色间的调度。
  - **异步 vs [[colocated与disaggregated部署]] disaggregate**：disaggregate 是"角色放哪"（部署轴），异步是"采训怎么排队"（调度轴）。disaggregate 是异步的前提（两池独立才能 overlap），但 disaggregate 也可跑同步（两池串行）。详见 §3.6。
  - **partial vs 同步 group sampling**：GRPO 的 group sampling（同 prompt 采 $G$ 条）默认同步等整组完成；partial 不等整组，凑够就训。partial 是 group sampling 的负载均衡松弛。
  - **off-policy vs on-policy**：同步 $\approx$ on-policy（$k\approx1,\rho\approx1$），异步 $\approx$ off-policy（$k>1,\rho$ 偏离）。但同步也是轻度 off-policy（差一步），详见 [[importance sampling与off-policy correction]] §3.6。
  - **本条 vs [[RL权重同步]]**：[[RL权重同步]] 讲"怎么把新 θ 下发给 rollout"（手段：NCCL/IPC/ckpt/delta）；本条讲"采训两端怎么排队协同"（时序：同步/异步/partial）。sync 频率 $S$ 是两者的交叉点——$S$ 既是 sync 间隔，又是 staleness 上界 $k_{\max}$。

- **三篇互链**: [[sequence packing与动态采样]]（partial 常配动态采样）、[[verifier与function reward]]（reward 决定 advantage 是否为零，进而决定动态采样是否过滤）。


## 7. 常见误区与易错点

> [!warning] 误区 1：同步就是 on-policy、异步才是 off-policy
> 不准确。同步也是轻度 off-policy——rollout 采那一刻是 $\theta_{old}$，trainer 训完变 $\theta_{new}$，差一步，$\rho\ne1$。同步只是把 staleness $k$ 压到 1，不是消除 off-policy。真正 on-policy（每采一条立刻训）在 LLM-RL 不可行（rollout 太贵）。PPO/GRPO 全是轻度 off-policy + clip。详见 [[RL权重同步]] §2.1。

> [!warning] 误区 2：异步一定比同步快 2×
> 仅当 rollout 与 train **时间平衡**时理想 2×。若 rollout-bound（$T_r\gg T_t$），异步 help 小（受 rollout 拖），该扩 rollout 池而非异步。且异步的 staleness 代价（clip 失真、sync 开销、可能需更多 step 收敛）会侵蚀加速。AReaL 2.77× 是叠加系统优化的结果，非裸 overlap。详见 §4.2。

> [!warning] 误区 3：异步的 staleness 靠 clip 就能兜底
> clip 是事后截断，staleness 大时大量 token 超 clip 被截断，**有效梯度信号失真**（clip frequency 飙），policy 朝错误方向漂移 → KL 爆/policy collapse。治本要**频繁 sync 降 $k$**（从源头控 off-policy），clip 只是兜底。两者配合。详见 [[stale policy problem]]、[[RL权重同步]] §2.2。

> [!warning] 误区 4：partial rollout 不需要版本号
> 错。partial 凑够就训，batch 内样本可能跨 $\theta$ 版本（前半采时 $\theta_{old-k}$、sync 后后半采时 $\theta_{old}$）。若不记版本号、用统一 $\pi_\theta/\pi_{old}$ 算 ratio，会**算错 IS 修正**（把落后 $k$ 步的样本当成落后 0 步），梯度有偏。必须每条记版本号独立算 ratio。详见 §4.4、[[importance sampling与off-policy correction]] §8。

> [!warning] 误区 5：colocate 同步利用率 100% 所以最快
> colocate 同一组 GPU 分时跑，物理利用率高，但**无法 overlap**（两阶段串行），墙钟 $T_r+T_t$。disaggregate 异步虽两池独立卡多，但 overlap 后墙钟 $\max(T_r,T_t)$，大模型下常更快。选型看规模/瓶颈，非"利用率高即快"。详见 [[colocated与disaggregated部署]] §7。

> [!warning] 误区 6：off-policy 是第四种独立调度模式
> 不是。off-policy 是**数据语义**（样本来自旧 θ 需 IS 修正），同步/异步/partial 是**调度时序**。异步、partial、多 epoch 复用都是 off-policy（程度不同），靠 IS + clip 成立。把 off-policy 当并列调度模式会混淆"采训怎么排队"与"数据要不要修正"两个正交轴。详见 §3.1 补充。

> [!warning] 误区 7：异步对所有任务都好
> 异步的收益依赖 rollout/train 平衡 + 容忍 staleness。对**收敛敏感、ratio 易爆**的任务（如小 $\varepsilon$ 严格约束、reward 稀疏方差大），异步 staleness 可能 destabilize 训练，同步更稳。社区趋势：中小模型 colocate 同步；70B+/R1 范式 disaggregate 异步容忍 stale。详见 [[colocated与disaggregated部署]] §8.3。


## 8. 延伸细节

### 8.1 三大异步系统实践

- **AReaL**（arXiv:2505.24298，inclusionAI）：**fully asynchronous** RL 系统——完全解耦 generation 与 training，rollout workers **持续生成不等任何 barrier**，training workers **凑够一个 batch 就更新**。关键手段：① **workload balancing**（配比 rollout/train 速率控 data staleness）；② **staleness-enhanced PPO**（容忍过期样本的 PPO 变体）。实测 **2.77× 加速**（vs 同步同 GPU 数），数学/代码 reasoning 基准上 final performance 匹配或更优。代码 `github.com/inclusionAI/AReaL`。
- **AsyncFlow**（arXiv:2507.01663，ByteDance）：**asynchronous streaming** RL 框架——分布式数据存储/传输模块做统一数据管理 + 细粒度调度，**producer-consumer 异步工作流**，在 staleness 阈值内**策略性延迟参数更新**（strategically deferring parameter update within staleness thresholds）最小化计算空闲。架构与底层训练/推理引擎解耦（service-oriented 接口）。实测 **1.59× 平均吞吐提升** vs SOTA baseline。
- **verl `fully_async`**：verl 框架的 fully async 模式，rollout 与 trainer 解耦并行，靠 [[RL权重同步]] 的 NCCL broadcast 频率控 staleness。verl 是 DAPO 的基础框架（DAPO 建在 verl 上，arXiv:2503.14476）。verl 还支持 hybrid/colocate 模式（同步为主、可切异步）。具体配置名待以最新 verl 文档为准。

> [!note] 补充：三者共性与差异
> 三者都走"decouple generation from training + overlap + staleness 控制"路线，差异在工程侧重：AReaL 强调 fully-decoupled + workload balancing 配比；AsyncFlow 强调 streaming 数据流 + producer-consumer + staleness 阈值内延迟更新；verl fully_async 是框架内置的异步模式，与 verl 的 hybrid/colocate 可切换。共同数学根都是 [[importance sampling与off-policy correction]] + PPO clip/V-trace 兜底 staleness。

### 8.2 staleness-aware 调度

工程上不止"事后 clip 兜底"，还在**调度层源头控 staleness**：

- **staleness 上界 $k_{\max}$**：超过阈值的旧样本丢弃或降权（AReaL/AsyncFlow 的 staleness control）。
- **速率匹配**：配比 rollout/train worker 数量让两端速率匹配（AReaL workload balancing），减少一端等另一端的空闲。
- **延迟更新**：AsyncFlow 在 staleness 阈值内策略性延迟 trainer 更新（等 rollout 凑够更多有效样本），用可控 staleness 换更优 batch 质量。

这些把"事后兜底"前移到"源头预防"，比纯 clip 更稳。

### 8.3 与连续批处理 / 采样吞吐的关系

rollout 端的采样吞吐受 [[continuous batching的调度实现]] 影响（continuous batching 动态进出请求、[[chunked prefill]] 切长 prompt）。异步模式下 rollout 池持续采，continuous batching 的变长混合 + chunked prefill 让 rollout 在变长样本下也高吞吐。partial 模式的"凑够就训"天然适配 continuous batching 的动态进出。详见 [[sampling throughput]]、[[continuous batching的调度实现]]、[[chunked prefill]]。

### 8.4 内容来源

四种采样模式综合自 AReaL（arXiv:2505.24298，fully async + staleness-enhanced PPO + 2.77×）、AsyncFlow（arXiv:2507.01663，streaming + producer-consumer + staleness 阈值延迟更新 + 1.59×）、DAPO（arXiv:2503.14476，built on verl，Dynamic Sampling 的过滤动机）、verl/OpenRLHF 的 colocate/hybrid/async 模式（[[colocated与disaggregated部署]] §3.5、[[Ray与分布式调度]] §3.5 已核实）。staleness→ratio 指数爆推导与 [[importance sampling与off-policy correction]] §4.4、[[RL权重同步]] §4.2 一致（一阶 Taylor 展开）。GPU 闲置率与 overlap 加速推导基于标准 pipeline overlap 算术。verl fully_async / OpenRLHF async rollout 的具体配置名待以最新文档为准。截至 2026-07。

---
相关: [[同步异步采样]] | [[asynchronous training]] | [[RL权重同步]] | [[importance sampling与off-policy correction]] | [[stale policy problem]] | [[colocated与disaggregated部署]] | [[trainer与rollout引擎组合]] | [[trajectory generation]] | [[rollout worker]] | [[sampling throughput]] | [[sequence packing与动态采样]] | [[continuous batching的调度实现]] | [[verifier与function reward]]
