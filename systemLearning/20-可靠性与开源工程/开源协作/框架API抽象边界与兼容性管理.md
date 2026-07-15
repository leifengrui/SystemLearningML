# 框架 API 抽象边界与兼容性管理

> **所属章节**: [[开源协作]]
> **所属模块**: [[20-可靠性与开源工程]]
> **别名**: API 兼容性 / deprecation / semver / 语义化版本 / public API / internal API / backward compatibility / breaking change / API 治理 / 版本演进
> **难度**: 中（需懂框架工程、[[GitHub issue到PR流程]] 的 PR/review/CI、版本管理）

## 1. 一句话定义

**框架 API 抽象边界与兼容性管理** 是把框架 API 分成**稳定的 public surface（公开 API）**与**可变的 internal surface（内部 API）**，并对**改动 public API 设严格规则**以保证老用户代码不 break 的工程治理——**抽象边界**：用 `__all__`、`_` 前缀、`@api.deprecated`、文档明示哪些是 public（承诺稳定、有 deprecation 周期才改/删）、哪些是 internal（随时可变、用户别依赖）；**向后兼容（backward compatibility）**：改 public API 不能 break 老代码——加参数要有默认值、改行为要给 opt-in flag、删 API 要走 **deprecation 流程**（`DeprecationWarning` → 版本说明 → 2-3 个版本后才删，给用户迁移期）；**semver（Semantic Versioning，语义化版本）**：`MAJOR.MINOR.PATCH`——MAJOR 升可 break 兼容、MINOR 升加功能但兼容、PATCH 升只 bugfix 兼容；**ML 框架 API 兼容难**因迭代快（PyTorch 半年一版）、API 面巨大（torch.* 几千函数）、社区依赖多（huggingface/lightning/megatron 都依赖 torch，一 break 牵连）；**deprecation 的迁移成本**是"框架演进"与"用户稳定"的 trade-off——兼容代码累赘（一堆 `if version >= x` 分支）但保用户信任；**直接改**省维护但伤用户（老代码崩、用户流失）。大项目（**PyTorch / vLLM / Megatron-LM**）的 API 演进是这套的工业实例——PyTorch `torch.legacy_*`、`F.affrad`→`F.softmax`、`torch.distributed.launch`→`torchrun` 都走了完整 deprecation。

> [!note] 三句话定位
> - **边界**：public API（`__all__` + 无 `_` 前缀，稳定）vs internal API（`_` 前缀/私有，随时变）。文档明示。
> - **兼容**：改 public 不能 break 老代码——加参数给默认、改行为给 opt-in、删走 deprecation（warning → 2-3 版本 → 删）。
> - **semver**：`MAJOR.MINOR.PATCH`，major 可 break、minor 加功能兼容、patch bugfix。版本号是兼容性承诺。


## 2. 为什么需要它（动机与背景）

### 2.1 用户依赖 API 才能用框架

用户代码 `import torch; torch.nn.functional.softmax(x)` 依赖 `torch.nn.functional.softmax` 这个名字稳定。框架一删/改名，几百万行用户代码崩。**API 是框架与用户的契约**——契约稳，用户敢上生产；契约乱，用户流失。PyTorch 早期 `F.affrad`（拼写错的 softmax）改 `F.softmax` 走了完整 deprecation 才没炸用户。

### 2.2 ML 框架迭代快但用户怕变

PyTorch 半年一 major 版（2.0/2.1/2.2...），每版加新功能/改优化器/改分布式语义。但用户的生产代码跑几年不想动。若每版都 break API，用户被迫每半年改一次代码，迁移成本巨大 → 流失到更稳的框架。所以 ML 框架 API 兼容是"演进"与"稳定"的持续 trade-off。

### 2.3 社区生态依赖链

huggingface transformers / accelerate、lightning、megatron、deepspeed、vLLM 都依赖 PyTorch API。PyTorch 一 break，这些下游全崩，链式故障。所以上游框架 API 改动要**极其慎重**，走完整 deprecation 让下游有时间迁。

### 2.4 internal API 需要自由变

框架内部实现（如 `torch._C._nn.*`、`torch._dynamo.*`）需要自由重构——改算法、改数据结构、合并函数。若这些被当 public 承诺稳定，框架无法演进。所以**明确划 internal**（`_` 前缀），告诉用户"别依赖，随时变"。PyTorch `torch._C`、`torch._dynamo` 是 internal，改它们不需 deprecation。

### 2.5 直接改 vs 兼容的 trade-off

直接改（删旧 API、换语义）省维护（不留兼容分支），但伤用户（老代码崩）。兼容（保留旧 API + 加新）保用户但代码累赘（`if version >= 2.0: new() else: old()` 散布）。ML 框架选"兼容优先"——宁可累赘保用户信任，因流失用户代价远大于维护兼容分支。这叫 **backward compatibility debt** 的反向——"兼容债"是主动承担的成本。

### 2.6 与开源协作的衔接

[[GitHub issue到PR流程]] 的 PR 改 API 时，reviewer 必查"是否 break 兼容、是否走 deprecation、是否升 MAJOR 版本"。这是 code review 的硬性 checklist。CI 可跑 **API 兼容性测试**（对比上版本 API surface，break 则 CI 红）。CLA 签署与 API 兼容承诺共同构成"开源可信"。


## 3. 核心概念详解

### 3.1 API 抽象边界

```
public API (稳定, 承诺):
  ├─ 无 _ 前缀的公开名 (torch.nn.Linear, torch.optim.Adam)
  ├─ __all__ 列出的导出名
  └─ 文档明示 "stable" / "public"

internal API (可变, 无承诺):
  ├─ _ 前缀 (torch._C, torch._dynamo, torch.nn._reduction)
  ├─ __ 双下划线 (name mangling, 强私有)
  └─ 文档明示 "internal" / "do not use"
```

- **`__all__`**：模块级列表，`from module import *` 只导入列出的。是 public API 的**显式声明**——没列的默认不导出（Python 仍可显式 `from module import _internal`，但约定不依赖）。
- **`_` 前缀**：单下划线前缀的名称约定为"内部"（`_internal_func`）。Python 不强制（不像 Java private），但约定 + linter 警告。`__`（双下划线）触发 name mangling（更强私有，类内有效）。
- **文档明示**：API docstring 标 `.. versionadded::`、`.. deprecated::`、`Stable since`、`Internal`。Sphinx 渲染时区分。
- **PyTorch 实例**：`torch.nn.Linear`（public，稳定）、`torch._C._nn.linear`（internal，C++ binding，随时变）、`torch._dynamo`（internal，torch.compile 内部，2.0+ 重构频繁）。

### 3.2 向后兼容性（backward compatibility）

```
向后兼容 = 老 API 调用方式在新版本仍能跑 (行为不变或可 opt-in 新行为)

加参数: 默认值保持老行为
  def f(x, new_param=None):  # 默认 None = 老行为
      ...

改行为: 加 opt-in flag, 默认老行为
  def f(x, new_mode=False):
      if new_mode: new_behavior()
      else: old_behavior()  # 默认老

删 API: 走 deprecation 周期 (见 §3.3), 不直接删
```

- **加参数**：新参数有默认值，老调用 `f(x)` 仍工作。默认值保持老行为。
- **改行为**：不直接改（破坏老代码），加 opt-in flag（`new_mode=False` 默认老），用户显式 `new_mode=True` 才新。下版本再让新行为成默认，再下版本删 flag。**两步走**避免一次性 break。
- **删 API**：不直接删，走 deprecation（§3.3）。给用户迁移期。
- **改默认值**：慎。改默认值会改变所有依赖默认的老代码行为。需 deprecation warning 提示 + 版本说明。

### 3.3 deprecation 流程

```
deprecation 周期 (删 public API 的标准流程):

V1.0  API 正常用
  │
V1.1  发 DeprecationWarning ("will be removed in V2.0, use new_api")
  │    文档标 .. deprecated:: 1.1, 指向替代 API
  │
V1.2, V1.3  warning 继续 (给用户 2-3 版本迁移)
  │
V2.0  删 API (用 old_api -> ImportError, 文档标 .. versionremoved:: 2.0)
```

- **`DeprecationWarning`**：调用旧 API 时运行时警告。Python 标准库 `warnings.warn("...", DeprecationWarning)`。用户跑代码看到 warning 知道要迁。
- **文档 `.. deprecated::`**：Sphinx directive，标"自 X 版 deprecated，用 Y 替代"。API doc 渲染时显眼。
- **迁移期**：给 2-3 个版本（如半年一版 = 1-1.5 年）让用户迁。不能一版本就删（用户来不及）。
- **`FutureWarning`**：比 `DeprecationWarning` 更显眼（默认对用户可见，`DeprecationWarning` 默认对开发者可见）。改默认行为时用 `FutureWarning` 提示"行为要变"。
- **最终删除**：`.. versionremoved::` 文档标删除，调用直接 `ImportError`/`AttributeError`。

### 3.4 semver（语义化版本）

```
版本号: MAJOR.MINOR.PATCH  (如 2.3.1)

MAJOR (2): 可 break 向后兼容 (删 API / 改语义 / 改默认值)
  -> 用户升级要改代码
MINOR (3): 加功能, 向后兼容 (新 API, 不删不改老的)
  -> 用户升级不用改代码, 多了新功能
PATCH (1): bugfix, 向后兼容 (修 bug, 不改 API)
  -> 用户升级 bug 修了, 不用改代码

预发布: 2.4.0-alpha.1, 2.4.0-rc1 (不稳, 不保证兼容)
build metadata: 2.4.0+build123 (不影响兼容)
```

- **MAJOR 升**：允许 break。用户升级要改代码。如 PyTorch 1.x → 2.0（torch.compile 引入，部分行为变）。
- **MINOR 升**：加功能不 break。用户升级零成本（多 API 可用）。
- **PATCH 升**：bugfix 不 break。无脑升。
- **预发布**：alpha/beta/rc，不保证兼容，用户慎用生产。
- **范围约束**：用户在 `requirements.txt` 写 `torch>=2.0,<3.0` 锁 MAJOR（2.x 内兼容）；或 `torch~=2.3`（≥2.3,<3.0）；或 `torch==2.3.0` 精确锁。semver 让依赖约束有意义——这是它存在的根本。

### 3.5 ML 框架 API 兼容难的原因

| 原因 | 说明 |
|---|---|
| 迭代快 | PyTorch 半年一 major，vLLM 月度发，每版改语义 |
| API 面巨大 | `torch.*` 几千函数/类，`torch.nn` 几百模块，全保持兼容难 |
| 社区依赖多 | huggingface/lightning/megatron/deepspeed/vLLM 全依赖，一 break 链式崩 |
| 性能优化逼改语义 | torch.compile 改执行模型、FSDP2 重写 FSDP 语义，难保老行为 |
| 实验性 API 多 | 新功能先 experimental（`torch.experimental.*`），稳定后变 public，过程有变 |
| 跨语言 | PyTorch 是 Python + C++（ATen/c10），C++ ABI 兼容也烦 |

### 3.6 与"直接改"的 trade-off

| 维度 | 兼容优先（保留旧 API + 加新） | 直接改（删旧换新） |
|---|---|---|
| 维护成本 | 高（保留兼容分支、deprecation warning、文档） | 低（代码干净） |
| 用户信任 | 高（老代码不崩） | 低（频繁 break，用户流失） |
| 演进速度 | 慢（被兼容债拖） | 快（随时重构） |
| 生态健康 | 强（下游敢依赖） | 弱（下游不敢上） |
| 代码累赘 | 多（`if version` 分支） | 少 |
| 适合 | 成熟框架（PyTorch）、有大量用户 | 早期项目（vLLM 早期）、用户少 |

ML 框架多选兼容优先——PyTorch 极重兼容（deprecation 周期长），vLLM 早期快（minor 版都 break），成熟后转稳。Megatron-LM 偏研究代码，兼容较弱（用户少，敢直接改）。


## 4. 数学原理 / 公式

### 4.1 semver 的兼容性约束

版本号 $V = (M, m, p)$（major.minor.patch）。两个版本 $V_1 = (M_1, m_1, p_1)$、$V_2 = (M_2, m_2, p_2)$，$V_2 > V_1$。兼容性承诺：

$$\text{compatible}(V_1, V_2) \iff M_1 = M_2$$

即**同 MAJOR 内兼容**。MINOR/PATCH 升不 break。MAJOR 升允许 break。故依赖约束 `torch>=2.0,<3.0` 锁 MAJOR=2 内兼容——这是 semver 的"可推导兼容性"。

### 4.2 deprecation 周期的迁移窗口

设 deprecation 在 $V_{\text{dep}}$ 版发 warning，在 $V_{\text{rem}}$ 版删。迁移窗口：

$$\Delta V = V_{\text{rem}} - V_{\text{dep}} \quad (\text{版本数})$$

若每 $T$ 时间一版，迁移时间窗口 $= \Delta V \cdot T$。PyTorch 半年一版，$\Delta V = 2-3$ → 迁移期 1-1.5 年。这是"用户来得及迁"的时间保证。$\Delta V$ 太小（如 1）用户来不及，太大（如 5）兼容债堆积。典型 2-3。

### 4.3 兼容债的累积

每保留一个 deprecated API = 一份兼容债。设框架有 $D$ 个 deprecated API，每 API 维护成本 $c$（文档/warning/测试分支），总兼容成本：

$$C_{\text{compat}} = D \cdot c$$

随 deprecated 累积 $D$ 增。删（$D$ 减）需等 $\Delta V$ 周期。故周期性清债——MAJOR 升时批量删老 deprecated（如 PyTorch 2.0 删了一批 1.x 的 deprecated）。

### 4.4 依赖约束的可满足性

设下游 $P$ 依赖上游 $U$，约束 `U>=a,<b`。可满足条件：上游发的版本 $V$ 满足 $a \le V < b$。semver 让约束有意义——`a`/`b` 是 MAJOR 边界，MINOR/PATCH 永远满足同 MAJOR 内。若上游乱发版（minor 版 break 兼容），约束失效，下游锁定 `==V`（精确锁，无升级自由）。semver 是"约束可推导"的基础，破坏 semver = 破坏依赖管理。

### 4.5 API surface 的稳定性度量

设 API 集合 $A_V$（版本 $V$ 的 public API 名集合）。稳定性指标：

$$\text{stability}(V_1 \to V_2) = \frac{|A_{V_1} \cap A_{V_2}|}{|A_{V_1}|}$$

（保留的 API 占比）。$=1$ 完全兼容（无删），$<1$ 有删（break）。CI 可跑这个对比——若 $\text{stability} < 1$ 且非 MAJOR 升，CI 红警告。


## 5. 代码示例（可选）

### 5.1 API 抽象边界（`__all__` + `_` 前缀）

```python
# myframework/api.py
__all__ = ["Linear", "softmax", "train"]  # public: 这些承诺稳定

class Linear:           # 无 _ 前缀 + 在 __all__, public
    ...

def softmax(x):         # public
    return _softmax_impl(x, dim=-1)  # 调 internal

def _softmax_impl(x, dim):   # _ 前缀, internal (随时可变)
    ...                       # 用户不该 from api import _softmax_impl
```

### 5.2 加参数保持兼容（默认值）

```python
# V1: def f(x): ...
# V2: 加 eps 参数, 默认 None 保持老行为
def f(x, eps=None):           # 老调用 f(x) 仍工作
    if eps is None:
        return _old_behavior(x)   # 默认老
    return _new_behavior(x, eps)  # 显式传 eps 才新
```

### 5.3 改行为用 opt-in flag（两步走）

```python
# V2.0: 加 new_mode flag, 默认 False (老行为)
def softmax(x, new_mode=False):
    if new_mode:
        return _new_stable_softmax(x)   # 更稳的数值实现
    return _old_softmax(x)              # 默认老 (不 break)

# V2.1: 发 FutureWarning, 提示下版 new_mode 默认变 True
def softmax(x, new_mode=None):
    if new_mode is None:
        warnings.warn("new_mode will default to True in v3.0; "
                      "set explicitly to silence.",
                      FutureWarning)
        new_mode = False   # 本版仍默认老
    ...

# V3.0: new_mode 默认 True (新行为成默认), flag 仍保留兼容
def softmax(x, new_mode=True):
    ...

# V4.0: 删 flag (deprecation 走完), 直接 new 行为
def softmax(x):
    return _new_stable_softmax(x)
```

### 5.4 deprecation warning + 文档

```python
import warnings

def old_api(x):
    """Old API, use new_api instead.

    .. deprecated:: 1.1
        Use :func:`new_api` instead. Will be removed in 2.0.
    """
    warnings.warn(
        "old_api is deprecated since v1.1, use new_api. "
        "Will be removed in v2.0.",
        DeprecationWarning,
        stacklevel=2,           # 警告指向调用方 (而非本行)
    )
    return new_api(x)           # 内部转调 new_api, 行为兼容

def new_api(x):
    """New API (stable since v1.1)."""
    return _impl(x)
```

### 5.5 API 兼容性 CI 测试（对比 API surface）

```python
# tests/test_api_compatibility.py
# 对比上版本的 public API 名, 不准删 (除非 MAJOR 升)
import myframework, subprocess

def get_public_api():
    return set(name for name in dir(myframework)
               if not name.startswith("_"))  # 无 _ 前缀 = public

def test_no_api_removed():
    """CI: MINOR/PATCH 版不准删 public API (否则 break)."""
    cur_api = get_public_api()
    # 跑上一 tag 的 API (git stash + checkout + 跑)
    subprocess.run(["git", "stash"])
    subprocess.run(["git", "checkout", "v1.0"])
    old_api = get_public_api()  # 假设能 import
    subprocess.run(["git", "checkout", "-"])
    subprocess.run(["git", "stash", "pop"])
    removed = old_api - cur_api
    assert not removed, f"API removed without MAJOR bump: {removed}"
    # CI 红 = 有删但非 MAJOR 升, 提示要走 deprecation 或升 MAJOR
```

### 5.6 版本号与依赖约束（`requirements.txt` / `pyproject.toml`）

```toml
# pyproject.toml (下游项目锁上游依赖)
[project]
dependencies = [
    "torch>=2.0,<3.0",     # 锁 MAJOR=2 内兼容 (2.x 任意版本)
    "vllm~=0.6.0",         # ~= 兼容版: >=0.6.0, <0.7.0 (锁 minor 内)
    "transformers>=4.40,<5.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4"]
```

- `>=2.0,<3.0`：MAJOR=2 内任何 MINOR/PATCH，兼容保证。
- `~=0.6.0`：等价 `>=0.6.0,<0.7.0`（锁 minor，允许 patch 升）。
- `==2.3.0`：精确锁（无升级自由，只在不信 semver 时用）。


## 6. 与其他知识点的关系

- **上游（依赖）**: Python `__all__`/`_`/`warnings` 机制、Git tag 与版本管理、[[GitHub issue到PR流程]]（PR/review/CI 是 API 兼容的执行机制）、Sphinx 文档（`.. deprecated::`/`.. versionadded::`）。
- **下游（应用）**: 框架 API 演进（PyTorch/vLLM/Megatron 版本路线）、依赖管理（pip/poetry 的 semver 约束）、[[benchmark与regression test]]（API 改语义要 benchmark CI 验性能不退化）、开源项目可信度（兼容承诺 = 用户敢上生产）。
- **对比 / 易混**:
  - **backward compatibility vs forward compatibility**：前者老代码在新版跑（框架承诺），后者新代码在老版跑（罕见，通常不承诺）。ML 框架只承诺 backward。
  - **deprecation vs breaking change**：前者渐进（warning → 删，给迁移期），后者一次性（直接删/改，立即 break）。成熟框架用前者，早期项目用后者。
  - **semver vs Calver（calendar versioning）**：semver 按兼容性（MAJOR.MINOR.PATCH），Calver 按日期（YYYY.MM.X，如 Ubuntu 24.04）。semver 反映兼容性，Calver 反映时间。ML 框架多用 semver，部分工具（如 PyTorch 的 CUDA tag）混用。
  - **public API vs internal API vs experimental API**：public 稳定（承诺+deprecation）、internal 随意变（`_` 前缀）、experimental 明确标"可能变"（`torch.experimental.*`，无 deprecation 期可改）。三档稳定性。
  - **兼容优先 vs 直接改**：见 §3.6 表。前者保用户但累赘，后者干净但伤用户。成熟选兼容，早期选直接改。


## 7. 常见误区与易错点

> [!warning] 误区 1：minor 版删 API
> 违反 semver。minor 版只能加功能不 break。删 API 要等 MAJOR 升或走 deprecation 删。minor 删 = 破坏 semver，下游约束失效。

> [!warning] 误区 2：改默认值不当 breaking
> 是 breaking。改默认值改变所有依赖默认的老代码行为。要 `FutureWarning` 提示 + 版本说明 + opt-in flag 过渡。不能默默改。

> [!warning] 误区 3：用 `DeprecationWarning` 当默认对用户可见
> Python 默认 `DeprecationWarning` 只对开发者可见（用户跑代码看不到）。改用户可见行为要用 `FutureWarning`（默认对用户可见）。否则用户看不到迁移提示。

> [!warning] 误区 4：依赖 `_` 前缀的 internal API
> 用户坑。`_` 前缀随时变，无 deprecation。下游代码依赖 `torch._C._nn.*` 会在某版崩。只依赖 public API。

> [!warning] 误区 5：一版本就删 deprecated API
> 太快。用户来不及迁。给 2-3 版本（半年一版 = 1-1.5 年）迁移期。立刻删伤用户。

> [!warning] 误区 6：experimental API 承诺稳定
> 错。`torch.experimental.*` 明确标"可能变"，无 deprecation 期就能改/删。用户用 experimental 自担风险。框架不该把 experimental 当 public 承诺。

> [!warning] 误区 7：版本号乱发（minor 版 break）
> 破坏 semver = 破坏依赖管理。下游锁 `==V`（精确锁）防 break，无升级自由，生态僵化。要严守 semver 语义。

> [!warning] 误区 8：删 API 不更新文档
> 删了 API 但文档仍写旧用法，用户照文档写崩。删 API 要同步删文档 + 加 `.. versionremoved::` 标记 + 迁移指南。

> [!warning] 误区 9：兼容分支不测试
> deprecated API 保留但无测试 → 行为悄悄变（重构时改了 internal，deprecated wrapper 行为变了，用户代码崩）。deprecated API 也要有测试，直到删为止。


## 8. 延伸细节

### 8.1 PyTorch API 演进实例

- `torch.distributed.launch` → `torchrun`：1.9 起 `launch` deprecated，2.x 仍在但 warning，引导迁 `torchrun`。是 [[elastic training]] 的入口演进。
- `torch.nn.functional.aff_rad`（拼写错）→ `torch.nn.functional.softmax`：早期拼写错，走 deprecation 改正名。
- `F.legacy_relu` / `torch.jit.script` 旧 API 走 deprecation 周期删。
- `torch.compile`（2.0+）：新执行模型，但 `eager` 仍是默认（保老行为），`compile` 是 opt-in。两步走过渡。
- `FSDP` → `FSDP2`：2.x 起 FSDP2 重写语义，旧 FSDP 走 deprecation，两者并存迁移期。

### 8.2 vLLM API 演进

vLLM 早期（0.3-0.5）minor 版都 break（参数改名/删），用户频繁改代码。0.6+ 转稳，加强向后兼容（参数加 alias 兼容旧名）。是"早期快、成熟稳"的典型。`LLM`/`SamplingParams` 类的参数演进走 deprecation。

### 8.3 Megatron-LM 的 API 演进

Megatron 偏研究代码，兼容较弱（敢直接改 argument 名/删函数），用户少敢迁。但社区 fork 多（DeepSpeed-Megatron、Megatron-DeepSpeed），fork 间 API 不一致是另一层兼容问题。是"研究代码 vs 产品代码"的兼容性差异。

### 8.4 API 兼容性 CI 工具

工具如 `vcr`/`griffe`（Python API surface 提取）、`abi-compliance-checker`（C++ ABI 对比）能自动对比版本间 API 差异，CI 跑报警"非 MAJOR 升删了 API"。是 API 兼容治理的自动化基础。

### 8.5 与 benchmark CI 的衔接

改 API 语义（如改默认行为）可能影响性能。PR 改 API 时 CI 跑 [[benchmark与regression test]] 验性能不退化。是"API 改 + 性能验"的双重 CI 闸门。

### 8.6 版本号与发布节奏

PyTorch 半年一 major（2.0/2.1/2.2...，2.x 内兼容）、vLLM 月度发（0.6.x）、Megatron 跟 PyTorch/CUDA 版本。发布节奏决定 deprecation 周期长度——半年一版 × 2-3 版 = 1-1.5 年迁移期。快节奏项目的 deprecation 要更紧凑。

### 8.7 内容来源

semver 规范整理自 semver.org 官方规范、Python `__all__`/`warnings`/`DeprecationWarning`/`FutureWarning` 官方文档、PEP 384（`__all__` 与 public API）、PyTorch deprecation policy 与实际 API 演进历史（`torchrun`/`F.softmax`/`torch.compile`/FSDP2）、vLLM 版本日志、Megatron-LM README。API 兼容性 CI 实践参考 `griffe`/`abi-compliance-checker` 工具文档，待核实最新。关联见 [[GitHub issue到PR流程]]、[[benchmark与regression test]]、[[Docker与Kubernetes]]、[[metrics logging tracing]]、[[elastic training]]（torchrun 作为 torch.distributed.launch 的 successor）。

---
相关: [[开源协作]] | [[GitHub issue到PR流程]] | [[benchmark与regression test]] | [[Docker与Kubernetes]] | [[metrics logging tracing]] | [[elastic training]]
