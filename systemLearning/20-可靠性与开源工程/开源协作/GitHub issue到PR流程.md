# GitHub issue 到 PR 流程

> **所属章节**: [[开源协作]]
> **所属模块**: [[20-可靠性与开源工程]]
> **别名**: issue 到 PR 流程 / 开源协作流程 / issue / PR / pull request / code review / CI / Continuous Integration / fork & branch / CONTRIBUTING.md / CLA / sign-off
> **难度**: 低中（需懂 Git、CI、[[metrics logging tracing]] 的 CI 监控、[[Docker与Kubernetes]] 的 CI runner 容器）

## 1. 一句话定义

**GitHub issue 到 PR 流程** 是开源协作的标准链路——① **issue**（在仓库提"报 bug / 提 feature"，用 **issue template** 模板化写复现步骤/期望/实际）；② **fork & branch**（fork 仓库到自己账号、建 **feature branch** 开发，不直接动 main）；③ **PR（pull request）**（把分支改动"请求合并"回上游仓库，PR template 描述改动/动机/测试）；④ **review（code review）**（reviewer 审代码质量/正确性/风格/测试覆盖，提 comment，作者改到 approve）；⑤ **CI（Continuous Integration）**（PR 触发 GitHub Actions / Jenkins 自动跑 lint/test/benchmark，**CI 全绿才可合并**）。分支策略上，开源用 **fork + feature branch**（外部贡献者无直接 push 权），内部研发用 **internal branch + feature branch**（有 push 权）；**release branch** 用于稳定版只接 bugfix。规范写在 **CONTRIBUTING.md**（怎么搭环境、跑测试、提 PR、sign-off/CLA 要求）。与公司内部研发的区别：开源需 **sign-off**（`git commit -s`，证明你有权提交，Linux 内核溯源）+ **CLA（Contributor License Agreement，贡献者许可协议，企业项目如 PyTorch/Megatron 母公司要签）**+ 邮箱合规（用真名邮箱，非 noreply）。大项目（**PyTorch / vLLM / Megatron-LM**）的 PR 流程是这套的工业级实例——CI 跑多 CUDA 版本/test/benchmark、需多 reviewer approve、大改动要 RFC issue 先讨论。

> [!note] 三句话定位
> - **链路**：issue（报问题）→ fork & branch（开发）→ PR（请求合并）→ review（审）→ CI（自动测）→ merge。
> - **规范**：CONTRIBUTING.md 写环境/测试/PR 要求；issue/PR template 模板化；sign-off + CLA + 真名邮箱（开源合规）。
> - **与内部研发区别**：开源 fork（无 push 权）+ sign-off/CLA（版权合规）+ 多 reviewer + 公开透明；内部直接 branch + 内部 CI。


## 2. 为什么需要它（动机与背景）

### 2.1 协作要规范否则乱

开源项目几百上千贡献者，若每人直接 push main，立刻乱——冲突、低质代码、风格不一、无测试。规范链路（fork + PR + review + CI）保证：① 改动经 review 才进；② CI 自动验质量（lint/test/benchmark）；③ 历史可追溯（PR 链接 issue、review 记录保留）；④ 贡献者合规（sign-off/CLA）。

### 2.2 issue 是需求/缺陷的单一入口

bug 报告、feature 提议、问题讨论都从 issue 开始。issue template 强制写复现步骤（版本/环境/命令/期望/实际），否则"我这跑不了"无法定位。issue 是 PR 的前置——**好 PR 链接好 issue**（`Fixes #123`），让 reviewer 知道为什么改。

### 2.3 review 是质量闸门

机器（CI）能验"能不能跑"，验不了"设计对不对/风格好不好/边界 case 漏没漏"。**人 review** 抓这些。大项目要求 1-2 个 reviewer approve 才可合并。review 也是知识传递——reviewer 的 comment 让作者学到项目规范。

### 2.4 CI 是自动化质量底线

人 review 会漏（视觉疲劳）。CI 全自动跑——lint（风格/ruff/black/clang-format）、unit test、integration test、benchmark（性能不退化）。CI 红的 PR 不准合并。把"质量"从"人盯"变成"机器强制"。CI 在容器里跑（[[Docker与Kubernetes]] 的 Pod runner），保证环境一致。

### 2.5 开源合规不能省

企业项目（PyTorch 母公司 Meta、vLLM 母公司 vLLM Project/Megatron 母公司 NVIDIA）要签 **CLA**——贡献者授权项目方使用你的代码（版权合规）。**sign-off**（`git commit -s`）是 commit 级证明"我有权提交这行"。Linux 内核溯源，开源圈沿用。不签 CLA / 不 sign-off 的 PR 不合并。这是开源与内部研发的关键区别——内部研发无版权合规要求（同公司）。


## 3. 核心概念详解

### 3.1 issue

```
issue (在仓库 Issues tab 提):
  ├─ 类型: bug report / feature request / question (用 issue template)
  ├─ 模板字段:
  │   - 版本 (PyTorch 2.3.0, CUDA 12.1, OS)
  │   - 复现步骤 (命令序列)
  │   - 期望 vs 实际
  │   - 最小可复现代码 (repro script)
  │   - 日志/截图
  └─ 标签 (label): bug / enhancement / P0 / needs-triage
```

- **issue template**：仓库 `.github/ISSUE_TEMPLATE/` 下放 YAML/MD 模板，提 issue 时预填字段。强制"版本/复现步骤/期望/实际"——杜绝"跑不了"无信息报告。
- **最小可复现（repro）**：要求贡献者给**最小脚本**（如 10 行 Python 复现 bug）。`import torch; x=torch.randn(...); ...`。无 repro 的 issue 难定位。大项目（PyTorch）有"repro script"硬要求。
- **label**：`bug`/`enhancement`/`P0`(紧急)/`needs-triage`(待分诊)/`good first issue`(新手友好)。维护者贴标管理优先级。

### 3.2 fork & branch

```
fork: 把上游仓库 fork 到自己账号 (github.com/me/upstream)
  -> 你有完整 push 权 (在自己的 fork)
branch: 在 fork 上建 feature branch 开发
  git checkout -b fix-nccl-hang
  ... 改代码 ...
  git commit -s -m "fix: handle NCCL timeout (fixes #123)"  # -s = sign-off
  git push origin fix-nccl-hang                              # push 到 fork
```

- **fork**：外部贡献者无上游仓库 push 权，先 fork（GitHub 给自己账号一份副本，有完整 push 权）。在自己 fork 上开发、push。
- **feature branch**：不直接动 main（main 保持干净），每改一特性建分支。分支名描述性（`fix-nccl-hang`、`feat-chunked-prefill`）。
- **sign-off**：`git commit -s` 在 commit message 末加 `Signed-off-by: Name <email>`。证明提交者有权贡献（Linux DCO - Developer Certificate of Origin 溯源）。
- **commit message 规范**：Conventional Commits（`fix:`/`feat:`/`docs:`/`refactor:`/`test:`/`chore:`）。首行简短、空行后详述、`Fixes #123` 链接 issue。

### 3.3 PR（pull request）

```
PR: 请求把 fork 的 feature branch 合并回上游 main
  ├─ source: me/upstream:fix-nccl-hang
  ├─ target: upstream:main
  ├─ PR template (自动填):
  │   - 改动描述 (what)
  │   - 动机 (why, 链接 issue)
  │   - 测试 (how tested)
  │   - breaking change? (是/否)
  │   - CLA 签署状态
  └─ CI 自动跑 (push 后触发)
```

- **PR template**：仓库 `.github/PULL_REQUEST_TEMPLATE.md`，新建 PR 自动填。强制写"改了什么/为什么/怎么测的/breaking change"。防"标题一行就提 PR"。
- **target branch**：通常 PR 到 `main`；release 的 bugfix PR 到 `release/v1.x`（cherry-pick 回 main）。
- **draft PR**：未完成先开 draft PR（CI/review 不触发或轻量），早收反馈。完成标"Ready for review"。

### 3.4 review（code review）

```
review (reviewer 审):
  ├─ 看什么:
  │   - 正确性 (逻辑对不对, 边界 case)
  │   - 设计 (接口/抽象合理, 是否过度工程)
  │   - 风格 (命名, 格式, 注释)
  │   - 测试 (有测试? 覆盖关键路径?)
  │   - 性能 (不该有 N+1 查询/无谓拷贝)
  │   - 文档 (改了 API 要更新 docstring)
  └─ 机制:
      - 行内 comment (点某行提意见)
      - approve / request changes / comment (三种结论)
      - 作者按 comment 改, push 新 commit, CI 重跑
      - 循环到 approve + CI 绿 -> 合并
```

- **reviewer 数**：大项目要求 1-2 reviewer approve（`CODEOWNERS` 文件自动指定负责模块的 reviewer）。
- **行内 comment**：点某行代码提具体意见，比整体 comment 精准。
- **approve / request changes**：approve = 认可可合并；request changes = 要改才能合；comment = 建议非阻塞。
- **CODEOWNERS**：仓库 `CODEOWNERS` 文件指定"哪路径改动需哪 reviewer 审"（如 `src/nccl/* @nccl-maintainer`），自动加 reviewer，防漏审。

### 3.5 CI（Continuous Integration）

```
CI (PR push 触发, GitHub Actions):
  ├─ lint (ruff/black for Python, clang-format for C++)
  ├─ type check (mypy/pyright)
  ├─ unit test (pytest, 跑所有 test_*.py)
  ├─ integration test (端到端, 跑训练/推理小 job)
  ├─ benchmark (性能不退化, 见 [[benchmark与regression test]])
  ├─ 多版本矩阵 (CUDA 11.8/12.1, Python 3.9/3.10/3.11)
  └─ CI 全绿才可合并 (branch protection rule)
```

- **触发**：PR push 自动触发 workflow（`.github/workflows/*.yml`）。
- **矩阵**：`strategy.matrix` 跑多版本组合（CUDA × Python × OS），保证跨版本兼容。
- **branch protection**：仓库设置"CI 必须绿 + reviewer 必 approve 才可 merge"，强制质量。
- **benchmark CI**：性能敏感项目（PyTorch/vLLM）CI 跑 benchmark，比上 baseline 不退化超阈值才过。详见 [[benchmark与regression test]]。

### 3.6 分支策略

```
main          ← 持续集成的主线 (随时可发布/可跑)
  ├─ feature/xxx  ← 功能开发分支 (PR 回 main)
  ├─ fix/yyy      ← bugfix 分支 (PR 回 main)
  └─ release/v1.0 ← 稳定版分支 (只接 cherry-pick 的 bugfix, 不接新功能)

fork (外部贡献者):
  me/upstream:main (跟上游 sync)
    └─ me/upstream:feature/zzz (开发) -> PR 回 upstream:main

internal (内部研发):
  upstream:feature/zzz (有 push 权, 不需 fork)
    -> PR 回 upstream:main
```

- **main**：持续可跑的主线，随时可发布。
- **feature/fix branch**：短命，合并后删。
- **release branch**：稳定版只接 bugfix（cherry-pick 自 main 的 fix），不接新功能。给生产用户稳定。
- **fork vs internal**：外部贡献者 fork（无上游 push 权），内部直接 branch（有 push 权）。

### 3.7 CONTRIBUTING.md 与 sign-off/CLA

- **CONTRIBUTING.md**：仓库根文档，写"怎么搭环境/跑测试/提 PR/sign-off 要求/CLA 签署/代码风格"。贡献者入门必读。
- **sign-off（DCO）**：`git commit -s` 加 `Signed-off-by: Name <email>`。证明提交者有权贡献（Developer Certificate of Origin）。Linux 内核溯源，开源圈沿用。GitHub 有 DCO action 自动检查 commit 是否带 sign-off。
- **CLA（Contributor License Agreement）**：企业项目（PyTorch-Meta、Megatron-NVIDIA）要贡献者签 CLA，授权项目方使用你的代码（版权合规）。一次性签，之后贡献都覆盖。不签 CLA 的 PR 不合并。
- **邮箱合规**：用真名邮箱（非 GitHub noreply），sign-off/CLA 邮箱要一致。

### 3.8 与内部研发的区别

| 维度 | 开源协作 | 公司内部研发 |
|---|---|---|
| push 权 | fork（无上游权） | 直接 branch（有权） |
| 版权合规 | sign-off + CLA（必签） | 无（同公司） |
| 透明度 | 全公开（PR/comment/CI 可见） | 内部可见 |
| reviewer | 社区 + 维护者（CODEOWNERS） | 团队内部 |
| CI | GitHub Actions / 公开 runner | 内部 Jenkins / 自建 runner |
| 节奏 | 异步（跨时区） | 同步（同团队） |
| 规范 | CONTRIBUTING.md + 强模板 | 内部 wiki/规约 |
| merge 权 | 维护者（非贡献者） | 团队 lead |


## 4. 数学原理 / 公式

### 4.1 CI 矩阵的组合数

设 CI 跑 $C$ 个 CUDA 版本 × $P$ 个 Python 版本 × $O$ 个 OS，总 job 数：

$$N_{\text{jobs}} = C \cdot P \cdot O$$

如 $C=3$（11.8/12.1/12.4）、$P=3$（3.9/3.10/3.11）、$O=2$（Linux/不跑 macOS）= 18 job/PR。每 job 几分钟到几十分钟。大项目 PR 频繁，CI 算力开销大——用**矩阵优化**（部分组合只在 PR 合并后跑，PR 阶段跑子集）减开销。

### 4.2 review 轮次的收敛

设 PR 经 $k$ 轮 review，每轮 reviewer 提 $n_i$ 个 comment，作者改后剩 $n_{i+1} < n_i$ 个未解决。合并条件：$n_k = 0$ 且 CI 绿。收敛速度取决于 reviewer 严格度与作者响应速度。大项目严格 PR 可能 $k=5-10$ 轮，持续数周。trade-off：严则质量高但慢，松则快但质量降。

### 4.3 branch protection 的"质量约束"

branch protection 规则形式化：

$$\text{merge allowed} \iff (\text{CI all green}) \land (\text{approve} \ge R_{\min}) \land (\text{no requested\_changes})$$

$R_{\min}$ 是最少 approve 数（大项目 1-2）。这是"机器 + 人"双闸门——CI 验能跑，review 验设计对。两者都过才可 merge。

### 4.4 release branch 的 cherry-pick 成本

main 上的 commit $c$ 要 cherry-pick 到 release/v1.x。cherry-pick 冲突概率 $p_{\text{conflict}}$ 与 main/release 分歧度正相关：

$$p_{\text{conflict}} \approx 1 - \exp(-\lambda \cdot \Delta t)$$

$\Delta t$ 是 release 与 main 的分歧时间，$\lambda$ 是 main 演进速率。分歧越久 cherry-pick 越易冲突。故 release branch 要定期从 main rebase 关键非破坏性改动，控制分歧。


## 5. 代码示例（可选）

### 5.1 commit message（Conventional Commits + sign-off）

```bash
# -s 加 sign-off (Signed-off-by: Your Name <you@email.com>)
git commit -s -m "fix(dds): handle NCCL timeout in elastic re-rendezvous

When a worker hangs in allreduce, the NCCL timeout was not propagated
to the elastic agent, causing the job to hang silently. Now we catch
the TimeoutError and trigger re-rendezvous.

Fixes #1234
"
```

### 5.2 issue template（`.github/ISSUE_TEMPLATE/bug_report.yml`）

```yaml
name: Bug report
description: Report a bug
labels: [bug, needs-triage]
body:
- type: input
  id: version
  attributes: { label: "PyTorch / CUDA version", placeholder: "torch 2.3.0 / CUDA 12.1" }
  validations: { required: true }
- type: textarea
  id: repro
  attributes:
    label: Repro script
    description: Minimal runnable script reproducing the bug
    placeholder: |
      import torch
      ...
  validations: { required: true }
- type: textarea
  id: expected
  attributes: { label: "Expected vs actual" }
  validations: { required: true }
```

### 5.3 PR template（`.github/PULL_REQUEST_TEMPLATE.md`）

```markdown
## What does this PR do?
Fixes #1234. Handles NCCL timeout in elastic re-rendezvous.

## Why?
Currently a hung worker causes the whole job to hang silently. We need to catch NCCL timeout and trigger re-rendezvous.

## How tested?
- [x] Unit test: `tests/test_elastic_timeout.py`
- [x] Integration: 4-node torchrun with simulated hang
- [x] Benchmark: no throughput regression (see [[benchmark与regression test]])

## Breaking change?
- [ ] Yes (describe migration)
- [x] No

## CLA
- [x] Signed CLA (my-email@company.com)
```

### 5.4 GitHub Actions CI（`.github/workflows/ci.yml`）

```yaml
name: CI
on: [pull_request, push]
jobs:
  lint:
    runs-on: ubuntu-latest
    container: pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel  # 容器跑 CI (环境一致)
    steps:
    - uses: actions/checkout@v4
    - run: pip install ruff black mypy
    - run: ruff check .
    - run: black --check .
    - run: mypy src/

  test:
    runs-on: [self-hosted, gpu]   # 自托管 GPU runner (跑 CUDA test)
    strategy:
      matrix:
        cuda: ["11.8", "12.1"]
        python: ["3.10", "3.11"]
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "${{ matrix.python }}" }
    - run: pip install -r requirements.txt
    - run: pytest tests/ -v --cov=src
    - run: python -m torch.distributed.run --nproc_per_node=2 tests/test_distributed.py  # 分布式 test

  benchmark:
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'pull_request'
    steps:
    - uses: actions/checkout@v4
    - run: python benchmarks/run.py --baseline main  # 比 main 不退化
```

### 5.5 branch protection（仓库设置，YAML 概念）

```yaml
# 仓库 Settings -> Branches -> Branch protection rules (main)
required_status_checks:
  - "CI / lint"
  - "CI / test (12.1, 3.11)"
required_pull_request_reviews:
  required_approving_review_count: 2   # 需 2 个 reviewer approve
  require_code_owner_reviews: true      # CODEOWNERS 路径必须其 owner 审
enforce_admins: true                   # 连 admin 也受保护
required_linear_history: true          # 不准 merge commit (rebase/squash only)
required_signatures: true               # commit 必 sign-off (DCO)
```

### 5.6 CODEOWNERS（自动指定 reviewer）

```
# .github/CODEOWNERS
# 每路径的 owner, 改该路径自动加 owner reviewer
*                       @maintainer-lead
src/distributed/        @distributed-team
src/nccl/               @nccl-maintainer
src/elastic/            @elastic-team
docs/                   @docs-team
```


## 6. 与其他知识点的关系

- **上游（依赖）**: Git（分支/合并/rebase/cherry-pick）、[[Docker与Kubernetes]]（CI runner 跑容器保证环境一致）、[[metrics logging tracing]]（CI 状态监控、benchmark CI 产 metrics）。
- **下游（应用）**: 开源项目协作（PyTorch/vLLM/Megatron 贡献）、公司内部研发流程（借鉴 fork+PR+CI 但去 sign-off/CLA）、[[benchmark与regression test]]（CI 跑 benchmark 防性能退化）、[[框架API抽象边界与兼容性管理]]（PR 改 API 要走 deprecation 周期、CLA 签署与 API 兼容承诺挂钩）。
- **对比 / 易混**:
  - **fork vs internal branch**：前者外部贡献者无 push 权先 fork，后者内部有权直接 branch。机制同（都 PR 回 main），权限不同。
  - **PR vs MR（merge request）**：GitHub 叫 PR（pull request），GitLab 叫 MR（merge request），概念同。
  - **review vs CI**：review 是人审设计/正确性/风格，CI 是机器验能跑/测过/性能。互补不替代。branch protection 要求两者都过。
  - **sign-off vs CLA**：sign-off 是 commit 级（DCO，每 commit 自带），CLA 是贡献者级（一次性签，覆盖所有）。都是版权合规，粒度不同。
  - **main vs release branch**：前者持续集成主线，后者稳定版只接 bugfix。bugfix 先 PR 到 main 再 cherry-pick 到 release。


## 7. 常见误区与易错点

> [!warning] 误区 1：直接 push main
> 开源严禁。必须 fork + feature branch + PR + review + CI。直接 push main 绕过质量闸门，破坏协作模型。即使有 push 权的维护者也走 PR（self-merge 也要过 CI）。

> [!warning] 误区 2：PR 无 issue
> 不好。好 PR 链接 issue（`Fixes #123`），让 reviewer 知道为什么改。无 issue 的 PR 缺动机上下文，难审。先开 issue 讨论，再开 PR 实现。

> [!warning] 误区 3：commit message 随便写
> 要 Conventional Commits（`fix:`/`feat:`）+ 详述 + sign-off + `Fixes #N`。一行 `update` 无法追溯。大项目 CI 检 commit message 格式与 sign-off。

> [!warning] 误区 4：忘签 CLA / sign-off
> 企业项目（PyTorch/Megatron）不签 CLA 不 merge。commit 不带 sign-off 被 DCO action 拦。首次贡献先签 CLA、commit 加 `-s`。

> [!warning] 误区 5：用 noreply 邮箱
> sign-off/CLA 邮箱要真名且一致。GitHub noreply 邮箱导致 sign-off 与 CLA 邮箱不匹配，DCO 检查失败。用真名邮箱。

> [!warning] 误区 6：CI 红了硬合
> branch protection 禁止。CI 红先修，不能绕。强制 CI 全绿才可 merge 是质量底线。

> [!warning] 误区 7：PR 巨大（几千行）
> 难审。大改动拆小 PR（每个 < 500 行，单职责）。大 PR reviewer 抵触、易漏 bug。先开 RFC issue 讨论方案，再拆小 PR 实现。

> [!warning] 误区 8：benchmark CI 不跑
> 性能敏感项目（PyTorch/vLLM）必须 CI 跑 benchmark 比上 baseline。不跑则性能退化（如某 PR 减 20% 吞吐）混进 main 才发现，已晚。详见 [[benchmark与regression test]]。


## 8. 延伸细节

### 8.1 大项目 PR 流程实例（PyTorch / vLLM）

PyTorch 的 PR：① 先开 issue 讨论（meta-issue 或 RFC）；② 大改动需在 `pytorch/rfcs` 提 RFC；③ 实现后 PR，CI 跑 lint/type/单元/多 CUDA 版本矩阵/test_windows/有 GPU 的 self-hosted runner；④ 需 1-2 个 PyTorch maintainer approve（CODEOWNERS 指定）；⑤ 大改动要等 review 数周；⑥ squash merge。vLLM 类似但更轻（CI 跑 model 覆盖测试、benchmark）。Megatron-LM（NVIDIA）要签 NVIDIA CLA。

### 8.2 CI runner 的 GPU 难题

跑 CUDA test 需 GPU runner，GitHub 公共 runner 无 GPU。大项目用**自托管 runner**（self-hosted, gpu label）——自家 GPU 机器跑 CI。成本：GPU 机器贵、runner 并发有限（PR 多要排队）。开源项目常 GPU runner 紧张，PR CI 排队几小时。这是 ML 开源 CI 的特殊难题。

### 8.3 fork 的 sync

fork 长期不 sync 上游会落后，PR 基于旧 main 冲突多。定期 `git fetch upstream && git rebase upstream/main`（或 GitHub 的 "Sync fork" 按钮）保持 fork 最新。rebase 后 force push 更新 PR。

### 8.4 依赖项的供应链

开源项目依赖其他开源库，CI 常跑 **dependabot/renovate** 自动提 PR 升依赖版本（如 PyTorch 升 NCCL 版本）。升依赖 PR 也走 review + CI，防 supply chain 风险。

### 8.5 与 API 兼容性的衔接

PR 改 API（删函数/改签名）要走 [[框架API抽象边界与兼容性管理]] 的 deprecation 周期——不能直接删，要先 `DeprecationWarning` + 文档标 deprecated + 2-3 版本后才删。CLA 的版权合规与 API 兼容承诺共同构成"开源可信"。

### 8.6 内容来源

issue/PR/review/CI 概念整理自 GitHub Docs（issues/PR/branch protection/CODEOWNERS/GitHub Actions）、Conventional Commits 规范、Linux DCO（Developer Certificate of Origin）文档、典型开源项目 CONTRIBUTING.md（PyTorch/vLLM/Megatron-LM）、CLA 实践（Meta/NVIDIA CLA 流程）。大项目 PR 流程实例参考 PyTorch/vLLM 实际 PR 历史。关联见 [[Docker与Kubernetes]]、[[metrics logging tracing]]、[[benchmark与regression test]]、[[框架API抽象边界与兼容性管理]]。

---
相关: [[开源协作]] | [[框架API抽象边界与兼容性管理]] | [[Docker与Kubernetes]] | [[metrics logging tracing]] | [[benchmark与regression test]]
