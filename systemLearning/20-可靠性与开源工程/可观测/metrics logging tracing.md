# metrics logging tracing

> **所属章节**: [[可观测]]
> **所属模块**: [[20-可靠性与开源工程]]
> **别名**: 可观测三支柱 / observability / metrics / logging / tracing / Prometheus / Grafana / OpenTelemetry / OTel / structured logging / DCGM exporter / distributed tracing
> **难度**: 中（需懂训练/推理性能指标、分布式 trace、Prometheus 采集模型、[[故障检测与恢复]] 的告警基础）

## 1. 一句话定义

**可观测性三支柱（observability three pillars）** 是把系统运行状态变成可查证据的三类数据——① **metrics（指标）**：数值时序（loss、吞吐、GPU 利用率、显存、NCCL 带宽），靠 **Prometheus** 周期 scrape + **Grafana** 看板看趋势；② **logging（日志）**：文本事件（某步某 rank 的 warn/异常/调度决策），结构化为 JSON 便于检索过滤，靠 log level（DEBUG/INFO/WARN/ERROR）分严重度；③ **tracing（分布式追踪）**：一个请求跨多服务/多阶段的**调用链**（如推理请求 `HTTP → scheduler → worker → GPU kernel`），靠 **OpenTelemetry** 给每段打 span + trace_id 串联，看哪段慢。在 ML 训推里：训练用 metrics 看几千卡的 loss spike/卡死/通信瓶颈（allreduce 带宽掉 = 某 rank 慢）、logging 记异常/调度决策/坏样本 skip、tracing 看一个 rollout 请求跨 trainer→rollout→verifier 的延迟分布（RL 框架尤其需要）；推理服务用 metrics 看 QPS/延迟分布/显存、tracing 看单请求在 scheduler/worker 的耗时分解。**GPU metrics** 主要靠 **DCGM exporter**（DaemonSet 采 GPU util/温度/显存/ECC/XID）暴露成 Prometheus 格式。**与 [[故障检测与恢复]] 的关系**：metrics 是检测的基础——watchdog 看吞吐、heartbeat 看 liveness、告警基于 metrics 阈值（如 `gpu_util < 10% for 5min` = 疑 hang）。**核心 trade-off**：采样密→看得清但开销大（metrics scrape、trace span 生成都有成本），稀疏→省但漏异常。生产常 metrics 全采、tracing 采样（如 1% 请求全 trace）。

> [!note] 三句话定位
> - **metrics**：数值时序，Prometheus scrape + Grafana 看板，看趋势（loss/吞吐/GPU util），告警基础。
> - **logging**：文本事件，结构化 JSON + log level，查"发生了什么"（某步异常/调度决策）。
> - **tracing**：跨服务调用链，OpenTelemetry span + trace_id，查"哪段慢"（推理请求各阶段耗时、RL rollout 链路）。
> - **关系**：metrics 是 [[故障检测与恢复]] 的数据源；三支柱互补——metrics 看趋势、logging 看细节、tracing 看链路。


## 2. 为什么需要它（动机与背景）

### 2.1 几千卡训练看不见就管不了

几千卡训练，肉眼盯不出问题——某 rank 的 allreduce 慢了 2x、某节点 GPU util 掉到 0、某 batch loss spike、通信带宽掉。这些要靠 **metrics 时序曲线**实时看。没有可观测 = 盲飞，崩了不知道哪崩、慢了不知道哪慢。

### 2.2 NCCL hang / 卡死要 metrics 才发现

NCCL hang 是静默死锁——进程活、零吞吐。靠 **watchdog 看步间隔** + **metrics 看吞吐曲线变平** + **NCCL 带宽 metrics 掉到 0** 才能发现。纯靠日志（无 error 输出）发现不了。详见 [[故障检测与恢复]] §3.1。

### 2.3 推理延迟分布要 trace 才分解

推理服务 P99 延迟高，是 scheduler 慢、worker 慢、还是 GPU kernel 慢？**trace** 把请求拆成 span（`scheduler.schedule`、`worker.forward`、`kv_cache.copy`），看哪段占比大。纯 metrics 只给"平均延迟"，trace 给"哪段慢"。是性能优化的起点。

### 2.4 RL 框架的跨服务链路

RL 训练一个 rollout 请求跨：trainer（生成 token）→ rollout worker（跑环境）→ verifier（判 reward）→ trainer（用 reward 更新）。这条链路跨多进程/多节点，**trace 串联**看哪段是瓶颈（rollout 慢、verifier 慢、还是同步等）。没有 trace 只能瞎猜。详见 [[RL worker故障恢复]] 的 RL 拓扑。

### 2.5 检测是恢复的前提

[[故障检测与恢复]] 的"检测"段全部基于可观测——watchdog 是 metrics、heartbeat 是 metrics、GPU 日志是 logging、告警是 metrics 阈值。没有可观测就没有自动检测，没有自动检测就没有自动恢复。可观测是容错的**前置基础设施**。


## 3. 核心概念详解

### 3.1 metrics（指标）

```
metrics: 数值时序 (name + timestamp + value + labels)
  ├─ counter (单调增, 如 request_count, step_count)
  ├─ gauge (可增减, 如 gpu_util, mem_used, loss)
  ├─ histogram/summary (分布, 如 latency P50/P95/P99)
  └─ 采集: Prometheus 周期 scrape /metrics endpoint (pull 模式)
     看板: Grafana 查 Prometheus 画曲线
```

- **数据模型**：metric = `{name, labels, value, timestamp}`。labels 是多维标签（如 `gpu="0", node="node-1", model="llama3"`），支持多维聚合查询。
- **四种类型**：
  - **counter**：单调递增（如 `training_steps_total`、`request_count`）。看速率用 `rate()`。
  - **gauge**：可增可减（如 `gpu_utilization`、`memory_used_bytes`、`loss`）。看瞬时值。
  - **histogram**：分布（如 `request_latency_seconds` 桶化的 P50/P95/P99）。看长尾。
  - **summary**：类似 histogram 但客户端预算分位数（少用，histogram 更灵活）。
- **采集**：Prometheus **pull 模式**——周期（如 15s）主动 scrape 各目标的 `/metrics` HTTP endpoint。目标自己暴露 metrics 文本格式。push 模式用 Pushgateway（短任务用）。
- **看板**：Grafana 接 Prometheus 数据源，写 PromQL 查询画曲线。典型 dashboard：训练 loss 曲线、GPU util 热力图、NCCL 带宽时序、吞吐 steps/s。

### 3.2 logging（日志）

```
logging: 文本事件 (timestamp + level + message + structured fields)
  ├─ level: DEBUG < INFO < WARN < ERROR
  ├─ 结构化: JSON {ts, level, msg, rank, step, loss, ...} 便于检索
  ├─ 采集: filebeat/fluentd 收集, 存 ELK/Loki
  └─ 查询: 按 level/rank/keyword 过滤
```

- **log level**：DEBUG（细节，生产关）、INFO（正常事件）、WARN（异常但可继续）、ERROR（错误需处理）。生产常 INFO+，排障临时开 DEBUG。
- **结构化日志**：JSON 格式（`{"ts":"...","level":"WARN","rank":3,"step":1000,"msg":"NCCL bandwidth low","bw_gbps":50}`）比纯文本好检索。能按 `rank=3 AND level=WARN` 过滤。
- **分布式训练的 rank 标注**：每条日志必带 `rank`/`node`，否则几千份日志混一起看不出谁的。是 ML 日志的特殊要求。
- **采集与存储**：filebeat/fluentd 收集容器/节点日志，存 Elasticsearch（ELK 栈）或 Loki（Grafana 系，更轻）。查用 Kibana/LogQL。

### 3.3 tracing（分布式追踪）

```
tracing: 一个请求跨多服务/多阶段的调用链
  ├─ trace: 一个请求的完整链路 (全局唯一 trace_id)
  ├─ span: 链路里一段 (有 span_id, parent_span_id, start/duration, name, tags)
  │     如 [scheduler.schedule] -> [worker.forward] -> [kv_cache.copy]
  ├─ 传播: OpenTelemetry 在跨进程边界注入/抽取 trace context (W3C TraceContext header)
  └─ 采集: OTel SDK 上报到 Jaeger/Tempo/OTLP collector
     分析: 看 span 树, 找最慢段 (critical path)
```

- **trace**：一个请求的完整调用链，全局唯一 `trace_id`。
- **span**：链路里一段。有 `span_id`、`parent_span_id`（构成树）、`start_time`/`duration`、`name`、`tags`（如 `gpu=0, batch_size=32`）。一个 trace = 多个 span 组成的树。
- **context propagation**：跨进程/服务边界时，OpenTelemetry 把 trace context（trace_id + 当前 span_id）注入 HTTP header 或 gRPC metadata，下游服务抽取后继续打 span，串成链。**这是分布式 trace 能串起来的关键**。
- **采样**：trace span 生成有开销，生产常**采样**（如 1% 请求全 trace，或"出错请求必 trace"）。head-based sampling（入口决定）或 tail-based sampling（出口按特征决定，如慢请求必采）。
- **后端**：Jaeger、Tempo（Grafana 系）、OTLP collector。看 span 树找最慢段。

### 3.4 三支柱对比

| 维度 | metrics | logging | tracing |
|---|---|---|---|
| 数据形态 | 数值时序 | 文本事件 | 调用链（span 树） |
| 粒度 | 聚合（每秒/每步） | 单事件 | 单请求链路 |
| 看什么 | 趋势、告警 | 发生了什么 | 哪段慢 |
| 采集 | Prometheus pull | filebeat 收集 | OTel SDK 上报 |
| 成本 | 低（数值） | 中（文本量大） | 高（span 生成） |
| 典型工具 | Prometheus + Grafana | ELK / Loki | Jaeger / Tempo |
| ML 训练用途 | loss/吞吐/GPU util 曲线 | 异常/调度决策/坏样本 | RL rollout 链路 |
| ML 推理用途 | QPS/延迟分布/显存 | 请求错误日志 | 单请求各阶段耗时 |
| 检测作用 | 告警基础（阈值告警） | 错误事件 | 性能瓶颈定位 |
| 采样 | 全采（量小） | 全采（或按 level 过滤） | 常采样（1% 或出错必采） |

### 3.5 ML 训练的关键 metrics

| metric | 类型 | 含义 | 异常信号 |
|---|---|---|---|
| `training_loss` | gauge | 当前 step loss | spike / 不降 |
| `grad_norm` | gauge | 梯度范数 | 爆炸（>1e3）/消失 |
| `learning_rate` | gauge | 当前 lr | scheduler 异常 |
| `steps_total` | counter | 累计步数 | 速率掉（卡死） |
| `throughput_tokens_s` | gauge | 每秒 token | 掉 = 通信瓶颈 |
| `gpu_utilization` | gauge | GPU SM 占用 | <10% 疑 hang/IO 瓶颈 |
| `gpu_memory_used` | gauge | 显存占用 | 接近上限 = OOM 风险 |
| `gpu_temp_celsius` | gauge | GPU 温度 | >85°C 降频风险 |
| `nccl_bandwidth_gbps` | gauge | allreduce 带宽 | 掉 = 通信瓶颈/某 rank 慢 |
| `nccl_avg_latency_ms` | gauge | allreduce 延迟 | 升 = 网络抖动 |
| `data_loader_time_s` | gauge | 数据加载耗时 | 占比高 = IO 瓶颈 |
| `ckpt_save_time_s` | gauge | ckpt 保存耗时 | 升 = IO/FS 瓶颈 |
| `dcgm_ecc_errors` | counter | ECC 错误计数 | 升 = 显存坏 |
| `dcgm_xid_errors` | counter | XID 错误计数 | >0 = GPU 故障 |

### 3.6 GPU metrics：DCGM exporter

```
节点装:
  ├─ NVIDIA driver
  ├─ DCGM exporter (DaemonSet, NVIDIA Data Center GPU Manager)
  │     ├─ 采 GPU util / mem / temp / ECC / XID / power
  │     └─ 暴露 :9400/metrics (Prometheus 文本格式)
  └─ node-exporter (采 CPU/mem/disk, 通用)
Prometheus scrape :9400/metrics
Grafana 画 GPU dashboard (NVIDIA 官方提供 Grafana dashboard JSON)
```

DCGM exporter 是 GPU metrics 的标准来源。采的 metric 名如 `DCGM_FI_DEV_GPU_UTIL`、`DCGM_FI_DEV_MEM_COPY_UTIL`、`DCGM_FI_DEV_FB_USED`（显存）、`DCGM_FI_DEV_GPU_TEMP`、`DCGM_FI_DEV_ECC_SBE_VOL_TOTAL`（单比特错）、`DCGM_FI_DEV_XID_ERRORS`。是 [[故障检测与恢复]] GPU 日志监控的**数据源**——告警规则基于这些（如 `DCGM_FI_DEV_XID_ERRORS > 0` 告警）。

### 3.7 RL 框架的分布式 trace

```
RL rollout 请求的 trace:
  trace_id=abc123
  ├─ span: trainer.generate (50ms)           # trainer 生成 token
  │   ├─ span: trainer.forward (40ms)
  │   └─ span: trainer.sample (10ms)
  ├─ span: rollout.run_env (200ms)          # rollout worker 跑环境
  │   ├─ span: env.step (180ms)
  │   └─ span: env.reward (20ms)
  └─ span: verifier.compute_reward (80ms)   # verifier 判 reward
看 span 树: rollout.run_env 占大头 (200ms), 是瓶颈
```

RL 框架（GRPO/PPO 训推一体）跨 trainer→rollout→verifier，trace 串联看哪段是瓶颈。是 [[RL worker故障恢复]] 的 RL 拓扑可观测。OpenTelemetry 在各角色进程注入 trace context，跨进程串成链。


## 4. 数学原理 / 公式

### 4.1 Prometheus 采集的 cardinality（基数）

metric 的 cardinality = label 组合数。如 `gpu_util{node, gpu, model}`，$N$ 节点 × 8 GPU × $M$ 模型 = $8NM$ 条时序。Prometheus 存储与查询成本与 cardinality 正相关：

$$\text{storage} \propto \sum_{\text{metrics}} \text{cardinality} \times \text{retention}$$

高基数（如把 `request_id` 当 label）会爆炸（每请求一条时序 = 几百万条）。**规则**：label 值的取值数要小（枚举类如 `model`、`node`），不要把高基数值（`request_id`、`user_id`）当 label。这是 Prometheus 的核心约束。

### 4.2 histogram 与分位数

延迟分布用 histogram——预定义桶（bucket），每桶计数。设桶上界 $b_1 < b_2 < \dots < b_k$，第 $i$ 桶累计计数 $c_i$（$\le b_i$ 的请求数）。P99（99 分位）近似：

$$\text{P99} \approx b_j, \quad \text{where } c_j \ge 0.99 \cdot C_{\text{total}} \text{ 且 } c_{j-1} < 0.99 \cdot C_{\text{total}}$$

桶越细 P99 越准，但桶多 storage 大。典型桶：`[0.005, 0.01, 0.025, ..., 10]` 秒。summary（客户端预算分位数）更准但不可跨实例聚合（每个客户端预算自己的分位，合并失真），故 histogram 更主流。

### 4.3 trace 采样率与开销

设单请求生成 $S$ 个 span，每 span 序列化与上报成本 $c_s$，请求率 $R$（req/s）。全采开销：

$$C_{\text{full}} = R \cdot S \cdot c_s$$

采样率 $\rho$（$\rho \in [0,1]$）下开销 $C = \rho \cdot R \cdot S \cdot c_s$。高 QPS 服务（如推理几千 req/s）全采会爆，故采样 1%（$\rho=0.01$）或 tail-based（慢/出错必采，正常采样）。trade-off：采样低省开销但漏异常。

### 4.4 告警的 false positive 率

设 metrics 阈值告警（如 `gpu_util < 10% for 5min` 判 hang）。正常低 util 偶发概率 $p$（如数据加载间隙），连续 5min（$T/\Delta = 5\text{min}/15\text{s}=20$ 次 scrape）全低于阈值的概率 $P_{\text{FP}} = p^{20}$。$p$ 越小误报越少，但真 hang 时 detect 延迟 = 5min。是"误报率 vs 检测延迟"的 trade-off（与 [[故障检测与恢复]] §4.1 watchdog 同构）。


## 5. 代码示例（可选）

### 5.1 训练脚本上报 metrics（Python + prometheus_client）

```python
from prometheus_client import Gauge, Counter, start_http_server

# 定义 metrics (带 labels 多维聚合)
loss_gauge = Gauge("training_loss", "current loss", ["model"])
grad_norm_gauge = Gauge("grad_norm", "gradient norm", ["model"])
steps_counter = Counter("training_steps_total", "total steps", ["model"])
gpu_util_gauge = Gauge("gpu_util_percent", "GPU utilization", ["node", "gpu"])

# 起 :8000/metrics endpoint, Prometheus scrape
start_http_server(8000)

for step, batch in enumerate(loader):
    loss = train_step(batch)
    loss_gauge.labels(model="llama3").set(loss.item())
    grad_norm_gauge.labels(model="llama3").set(grad_norm.item())
    steps_counter.labels(model="llama3").inc()
    # GPU util 从 nvidia-smi 或 NVML 读
    gpu_util_gauge.labels(node="node-1", gpu="0").set(get_gpu_util(0))
```

Prometheus 配 scrape 这个 :8000/metrics，Grafana 画 `training_loss` 曲线。

### 5.2 结构化日志（Python + structlog/标准 logging）

```python
import logging, json, socket

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts": self.formatTime(record),
            "level": record.levelname,
            "rank": int(os.environ.get("RANK", 0)),
            "node": socket.gethostname(),
            "msg": record.getMessage(),
            # 额外字段 (structured)
            "step": getattr(record, "step", None),
            "loss": getattr(record, "loss", None),
        })

logger = logging.getLogger("train")
h = logging.StreamHandler()
h.setFormatter(JsonFormatter())
logger.addHandler(h)
logger.setLevel(logging.INFO)

# 用
logger.warning("NCCL bandwidth low", extra={"step": 1000, "bw_gbps": 50})
# 输出: {"ts":"...","level":"WARNING","rank":3,"node":"node-1","msg":"NCCL bandwidth low","step":1000,"loss":null,"bw_gbps":50}
```

JSON 格式让 Loki/Kibana 能按 `level=WARNING AND rank=3` 过滤。

### 5.3 OpenTelemetry trace（推理请求各阶段）

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317"))
)
tracer = trace.get_tracer("vllm")

def handle_request(req):
    """推理请求处理, 每段打 span."""
    with tracer.start_as_current_span("handle_request") as root:
        with tracer.start_as_current_span("scheduler.schedule"):
            batch = schedule(req)          # scheduler 选 batch
        with tracer.start_as_current_span("worker.forward"):
            out = model.forward(batch)     # GPU forward
        with tracer.start_as_current_span("kv_cache.copy"):
            copy_kv(out)                  # KV cache 操作
        return out
# 上报到 Jaeger/Tempo, 看 span 树找最慢段
```

### 5.4 PromQL 告警规则（Grafana/Prometheus Alertmanager）

```yaml
# prometheus rules: GPU util 低 5min 疑 hang
groups:
- name: training
  rules:
  - alert: GpuUtilLowHang
    expr: avg by (node, gpu) (gpu_util_percent) < 10
    for: 5m                  # 持续 5min 才告警 (防抖)
    labels: { severity: critical }
    annotations:
      summary: "GPU {{ $labels.gpu }} on {{ $labels.node }} util < 10% for 5min, likely hang"
  - alert: NcclBandwidthDrop
    expr: nccl_bandwidth_gbps < 50
    for: 2m
    labels: { severity: warning }
    annotations:
      summary: "NCCL bandwidth dropped below 50 Gbps"
  - alert: GpuXidError
    expr: increase(dcgm_xid_errors[5m]) > 0
    labels: { severity: critical }
    annotations:
      summary: "GPU XID error on {{ $labels.gpu }}, hardware fault likely"
```

### 5.5 Grafana dashboard（NVIDIA 官方 GPU dashboard）

Grafana 导入 NVIDIA 官方 DCGM dashboard（ID 12239，待核实），自动有 GPU util 热力图、显存时序、温度、ECC 错误、XID 事件面板。配合训练 loss/吞吐面板，是训练观测标配。


## 6. 与其他知识点的关系

- **上游（依赖）**: 训练/推理性能模型（知道采哪些 metric）、[[Python multiprocessing与GIL]]（多进程采 metrics 不互抢）、Prometheus/Grafana/OTel 工具链、DCGM exporter。
- **下游（应用）**: [[故障检测与恢复]]（metrics 是检测基础、告警基于 metrics 阈值、watchdog 是 metrics 的步间隔超时）、训练平台告警系统、推理服务 SLO 监控（[[OpenAI API server]] 的延迟/QPS）、RL 框架 rollout trace（[[RL worker故障恢复]]）、性能优化（trace 找瓶颈→用 [[nsys (Nsight Systems)]]/[[ncu (Nsight Compute)]]/[[torch profiler]] 深挖）。
- **对比 / 易混**:
  - **metrics vs logging vs tracing**：见 §3.4 表。metrics 看趋势、logging 看事件、tracing 看链路。互补不替代。
  - **Prometheus pull vs push**：Prometheus 默认 pull（主动 scrape），短任务用 Pushgateway push。ML 训练 job 短时跑用 push gateway，常驻服务用 pull。
  - **head-based vs tail-based sampling**：前者入口决定采样（快但不能按结果采），后者出口按特征采（如慢请求必采，更智能但开销大）。
  - **tracing vs [[nsys (Nsight Systems)]]/[[torch profiler]]**：tracing 是**跨服务/进程**调用链（业务层），nsys/torch profiler 是**单进程内** GPU/CPU timeline（系统层）。前者看"哪个服务慢"，后者看"哪个 kernel 慢"。互补——trace 找到慢服务后用 nsys 深挖该服务内。
  - **metrics vs [[benchmark与regression test]]**：metrics 是**生产实时**观测，benchmark 是**离线**性能回归测试（CI 跑）。前者查线上，后者防退化。


## 7. 常见误区与易错点

> [!warning] 误区 1：把 request_id 当 metric label
> 灾难。高基数 label（request_id/user_id）让 cardinality 爆炸（每请求一条时序），Prometheus 撑不住。label 值取值数要小（枚举类）。高基数值只入 log/trace，不入 metric label。

> [!warning] 误区 2：只靠 metrics 不做 trace
> metrics 给"平均延迟高"但不给"哪段慢"。要 trace 拆 span 找瓶颈。纯 metrics 只能告警不能定位。

> [!warning] 误区 3：trace 全采
> 高 QPS 服务（推理几千 req/s）全采 span 会爆。生产常采样 1% 或 tail-based（慢/出错必采）。训练 QPS 低可全采。

> [!warning] 误区 4：日志不带 rank/node
> 分布式训练几千份日志混一起，不带 rank 看不出谁的。每条日志必带 rank/node/step，否则排查抓瞎。

> [!warning] 误区 5：纯文本日志
> 不便检索。结构化 JSON（带字段如 step/loss/rank）能按字段过滤（`level=ERROR AND rank=3 AND step>1000`）。纯文本只能全文搜。

> [!warning] 误区 6：DCGM exporter 等于 nvidia-smi
> 不。nvidia-smi 是 CLI 工具（人看一次），DCGM exporter 是常驻 daemon 暴露 Prometheus metrics（Prometheus 周期 scrape）。后者是可观测数据源，前者是排障工具。

> [!warning] 误区 7：告警基于瞬时值
> 易误报。用 `for: 5m`（持续 5min 才告警）防抖，避免单次抖动触发。是告警设计的基本规则。

> [!warning] 误区 8：trace 替代 nsys
> 不同层。trace 是业务层跨服务链路，nsys 是系统层单进程 GPU timeline。trace 找到慢服务后用 nsys 深挖该服务内的 kernel。互补不替代。


## 8. 延伸细节

### 8.1 OpenTelemetry 的统一野心

OpenTelemetry（OTel）是 CNCF 项目，目标统一 metrics/logging/tracing 三支柱的 API/SDK——一套 SDK 同时产 metrics、log、trace，跨语言（Python/Go/Java/C++）。旧项目（OpenTracing/OpenCensus/Jaeger client/Prometheus client）正被 OTel 整合。未来趋势是"OTel 一套 SDK 产三类数据，后端可换（Jaeger/Tempo/Prometheus/Loki）"。ML 框架（vLLM/PyTorch 生态）正逐步接 OTel。

### 8.2 训练 loss spike 的实时检测

训练 loss 异常 spike（某步 loss 突然飙）是训练不稳的信号（梯度爆炸/lr 太大/坏 batch）。metrics 采 `training_loss` + 告警规则（如 `loss > rolling_avg * 5`）实时抓。配 grad_norm 监控（爆炸前 grad_norm 先飙）。是训练稳定性工程（[[12-训练稳定性工程]]）的观测基础。

### 8.3 推理服务的 SLO 监控

推理服务 SLI/SLO：TTFT、TPOT、P99 延迟、QPS、错误率。metrics 采这些 + Grafana 画 + Alertmanager 告警（如 `p99_latency > 500ms` 告警）。trace 拆单请求找慢段。与 [[OpenAI API server]] 兼容接口的 SLO 通用此栈。

### 8.4 与 nsys/ncu 的衔接

trace 定位"某服务慢"后，在该服务内用 [[nsys (Nsight Systems)]] 抓 timeline（CPU/GPU kernel 对齐）、[[ncu (Nsight Compute)]] 抓 kernel metric（算力/带宽利用率）。trace 是业务层入口，nsys/ncu 是系统层深挖。详见 [[nvprof（旧）]]→[[ncu (Nsight Compute)]] 的演进。[[torch profiler]] 是 PyTorch 自带的可视化 profiler，介于两者间。

### 8.5 RL 框架的可观测

[[RL worker故障恢复]] 的 RL 训练跨 trainer/rollout/verifier 多角色，可观测尤重要——metrics 看各角色吞吐、tracing 看单 rollout 链路、logging 看各角色异常。OpenTelemetry 在角色间传 trace context 串联。是"训推一体"框架的观测核心。

### 8.6 内容来源

可观测三支柱整理自 Google SRE 书（Site Reliability Engineering）、Prometheus 官方文档（metric 类型/pull 模型/PromQL）、Grafana 官方文档、OpenTelemetry 官方文档（trace/span/context propagation）、NVIDIA DCGM exporter 文档与官方 Grafana dashboard（ID 12239 待核实）、Jaeger/Tempo 文档。ML 场景应用参考大模型训练平台工程实践与 vLLM/PyTorch 可观测文档。关联见 [[故障检测与恢复]]、[[RL worker故障恢复]]、[[OpenAI API server]]、[[nsys (Nsight Systems)]]、[[ncu (Nsight Compute)]]、[[torch profiler]]、[[nvprof（旧）]]、[[benchmark与regression test]]。

---
相关: [[可观测]] | [[故障检测与恢复]] | [[elastic training]] | [[Docker与Kubernetes]] | [[RL worker故障恢复]] | [[OpenAI API server]] | [[nsys (Nsight Systems)]] | [[ncu (Nsight Compute)]] | [[torch profiler]] | [[nvprof（旧）]] | [[Python multiprocessing与GIL]] | [[benchmark与regression test]]
