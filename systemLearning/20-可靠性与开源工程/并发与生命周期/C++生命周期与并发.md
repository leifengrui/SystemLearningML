# C++生命周期与并发

> **所属章节**: [[并发与生命周期]]
> **所属模块**: [[20-可靠性与开源工程]]
> **别名**: C++ lifetime / RAII / shared_ptr / weak_ptr / std::thread / data race / UB / use-after-free / C++ 内存管理
> **难度**: 中高（需懂 C++ 对象生命周期、智能指针、多线程同步、与 Python GC 对比、推理引擎/训练框架的 C++ 层）


## 1. 一句话定义

**C++ 生命周期与并发**是 C++ 工程的两大核心地雷——**生命周期**指对象从构造（资源获取）到析构（资源释放）的全程，C++ **无 GC**（垃圾回收），全靠 **RAII（Resource Acquisition Is Initialization，资源获取即初始化）** 把资源（内存/fd/锁/socket/CUDA stream）绑定到对象——构造获取、析构释放、异常时栈展开自动析构（**异常安全**），配合 **`std::shared_ptr`（引用计数智能指针）** 自动管理共享对象的生命周期（最后一个引用归零自动释放，但**循环引用**需 `weak_ptr` 打断否则泄漏）；**并发**用 **`std::thread`**（多线程），多线程共享数据需 **mutex/atomic** 保护，否则 **data race（数据竞争）** 是 **UB（未定义行为，Undefined Behavior）**——程序可崩、可错、可"看起来对"随机失败，**极难复现调试**；C++ 的 **UB 三大杀手**：data race、use-after-free（悬垂指针 deref）、空指针/越界——它们不像 Python 抛异常，而是**静默错或随机崩**。在 ML 工程里，[[推理引擎源码|vLLM]]/Megatron/Torch 的 C++ 层（ATen/c10 kernel、CUDA stream 管理、block manager）踩 UB 会导致**间歇性崩溃/数值错乱**难定位——不像 Python 错误有 traceback，C++ 的 segfault 只给个 core dump，悬垂指针的 use-after-free 可能在**对象已释放后很久**才崩，调用栈对不上现场。是 [[Python multiprocessing与GIL]]（Python 端并发）的 C++ 端对照——Python 靠 GIL 保引用计数线程安全、靠 GC 环回收，C++ 靠 RAII + 智能指针 + 程序员自律，无安全网。

> [!note] 三句话定位
> - **是什么**：C++ 无 GC，对象生命周期全靠 RAII（构造获取/析构释放/异常安全）+ 智能指针（shared_ptr 引用计数）管；并发靠 std::thread + mutex/atomic，无保护的数据竞争是 UB。
> - **为什么 ML 要懂**：vLLM/Megatron/Torch 的 C++ 层（kernel、stream、block manager）踩悬垂指针/data race 会导致间歇崩/数值错，且难复现难调试（segfault 无 traceback）。
> - **与 Python 对比**：Python 有 GC（引用计数 + 循环检测），对象生命周期自动；C++ 无 GC，全靠 RAII/智能指针 + 程序员管，错一步就是 UB。


## 2. 为什么需要它（动机与背景）

### 2.1 为什么 C++ 无 GC

C++ 设计哲学是**零开销原则**（zero-overhead）——不为没用的功能付费。GC 的运行时开销（stop-the-world、内存占用、不可预测停顿）违背 C++ 对性能/确定性的追求。C++ 选择**确定性析构**（deterministic destruction）：对象出作用域立即析构（精确时机），资源立即释放，无 GC 延迟。代价：**程序员全权管生命周期**，错就是 UB。

### 2.2 ML 框架的 C++ 层为什么躲不开

- **PyTorch ATen/c10**：kernel、dispatcher、autograd engine 用 C++（性能 + CUDA 互操作）。
- **vLLM**：PagedAttention 的 block manager、CUDA stream 调度、sampling 的 C++ 优化。
- **Megatron**：通信 overlap、MoE dispatcher 的 C++。
- 这些 C++ 层一旦有悬垂指针（如 freed KV block 被 deref）或 data race（多线程改 KV table 无锁），症状是**间歇性 segfault / 数值错乱**——难复现、难定位、CI 抓不到、生产偶发。比 Python 错误难调试一个数量级。

### 2.3 RAII 是 C++ 的安全网

C++ 用 **RAII** 把资源绑对象——`std::lock_guard` 构造时 `lock()`、析构时 `unlock()`（异常也 unlock，防死锁）；`std::vector` 构造分配内存、析构释放；`cudaStream_t` 封装成类，析构 `cudaStreamDestroy`。资源不裸用，全包进 RAII 类，靠**作用域**自动管——这是 C++ 抵消无 GC 风险的核心范式。

### 2.4 智能指针自动化

`std::unique_ptr`（独占，零开销）、`std::shared_ptr`（共享，引用计数）把"何时 delete"自动化。`shared_ptr` 最后一个引用归零自动析构对象，像 Python 引用计数。但**循环引用**（A 持 B，B 持 A）会泄漏（refcnt 永不归零）——需 `weak_ptr` 打断。


## 3. 核心概念详解

### 3.1 RAII（Resource Acquisition Is Initialization）

RAII 的三要素：① 资源获取在**构造函数**（构造失败则抛异常，无泄漏）；② 资源释放在**析构函数**（出作用域自动调，异常时栈展开也调——**异常安全**）；③ 资源用**全生命周期**绑对象，对象在作用域结束自动析构释放资源。

```cpp
{
    std::lock_guard<std::mutex> lk(mtx);  // 构造: mtx.lock()
    do_work();                             // 若抛异常, lk 析构 unlock (异常安全)
}  // 作用域结束: lk 析构, mtx.unlock() (无论异常与否)
```

对比裸用：`mtx.lock(); do_work(); mtx.unlock();`——若 `do_work` 抛异常，`unlock` 不执行→死锁。RAII 解决。

### 3.2 shared_ptr（引用计数智能指针）

```cpp
auto p = std::make_shared<Foo>(args...);  // refcnt=1
auto q = p;                                 // refcnt=2 (p, q 都指向)
q.reset();                                  // refcnt=1
// p 出作用域: refcnt=0 -> 析构 Foo, 释放内存
```

- **引用计数**：每复制 +1，每销毁 -1，归零自动 `delete`。线程安全（refcnt 用 atomic）。
- **make_shared**：一次分配（对象 + control block），比 `new + shared_ptr` 快、省内存、异常安全。
- **循环引用泄漏**：A 的 `shared_ptr<B>` + B 的 `shared_ptr<A>`，互相 +1，refcnt 永不归零→泄漏。用 `weak_ptr` 打断：

```cpp
struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;   // weak 不增 refcnt, 打断循环
};
```

### 3.3 weak_ptr（弱引用，观察不持有）

`weak_ptr` 由 `shared_ptr` 构造，**不增 refcnt**（只观察）。用前 `lock()` 提升为 `shared_ptr`（若对象还在则成功，refcnt+1；若已释放则返回空）。用于：① 打断循环引用；② 观察者模式（不延长对象生命周期）；③ cache（对象可被释放时 cache 失效）。

### 3.4 std::thread（多线程）

```cpp
std::thread t([]{ work(); });  // 起线程
t.join();                       // 等结束 (或 detach 分离, 或 move 到容器)
// 警告: 析构未 join/detach 的 thread 会 std::terminate (崩溃)
```

- `std::thread` 析构前必须 `join()`（等子线程）或 `detach()`（分离独立跑）。析构一个 `joinable()` 的 thread 是 UB/crash。
- 传递参数：默认按值拷（避免悬垂引用），要传引用用 `std::ref`。

### 3.5 data race（数据竞争）与 UB

```cpp
int counter = 0;
std::thread t1([&]{ for(int i=0;i<1e6;i++) counter++; });
std::thread t2([&]{ for(int i=0;i<1e6;i++) counter++; });
t1.join(); t2.join();
// counter 期望 2e6, 实际随机 < 2e6 (data race: ++ 不是原子, 丢失更新)
```

- **data race**：≥2 线程并发访问同一内存，至少一个写，无同步 → UB。
- **`++`** 非原子：读-改-写三步，两线程交错丢失更新。
- **修复**：`std::mutex`（`std::lock_guard` RAII 包）或 `std::atomic<int>`。

### 3.6 UB（未定义行为）三大杀手

| UB 类型 | 触发 | 症状 | 难点 |
|---|---|---|---|
| **data race** | 多线程无锁改共享 | 随机错、间歇崩 | 难复现，CI 抓不到 |
| **use-after-free** | free 后仍 deref 指针 | 随机崩、数值错（内存被复用） | 崩在释放后很久，栈对不上 |
| **空指针 deref / 越界** | nullptr/越界访问 | segfault | 现场明确但根因在别处 |

> [!warning] UB 的恐怖
> UB **不是抛异常**，是**编译器可任意假设 UB 不发生**——优化后行为不可预测。可崩、可静默错、可"看起来对"随机失败。CI 跑 1000 次过了，生产偶发崩。ASan/TSan/UBSan 是唯一系统化检测手段（详见 §8）。

### 3.7 use-after-free 与悬垂指针

```cpp
int* p = (int*)malloc(sizeof(int));
*p = 42;
free(p);
*p = 99;        // use-after-free! UB: 那块内存可能被复用, 写错地方
// 或读:
int v = *p;     // 读已释放内存, UB, 可读到旧值或垃圾
```

- 智能指针根治：`shared_ptr`/`unique_ptr` 析构自动释放，无法被 deref（对象已析构，指针失效）。
- 裸指针 + 手动 `delete` 易踩——`delete p; ...; *p = ...`（use-after-free）。RAII/智能指针是防悬垂的根本。

### 3.8 mutex / atomic / condition_variable

- **`std::mutex`**：互斥锁，`lock()`/`unlock()`。用 `std::lock_guard`/`unique_lock` RAII 包（异常安全）。
- **`std::atomic<T>`**：无锁原子操作，`fetch_add`/`compare_exchange`。适合简单计数/标志。
- **`std::condition_variable`**：等待-通知，配合 mutex 实现生产者-消费者。

### 3.9 与 Python 的 GC 对比

| 维度 | C++（无 GC） | Python（GC + GIL） |
|---|---|---|
| 内存管理 | RAII + 智能指针 + 程序员 | 引用计数 + 分代 GC（自动） |
| 释放时机 | 确定（出作用域立即） | 不确定（refcnt=0 立即，循环靠 GC 周期） |
| 引用计数 | shared_ptr 手动（atomic 线程安全） | ob_refcnt（GIL 保线程安全） |
| 循环引用 | 需 weak_ptr 打断 | GC 检测回收 |
| 悬垂指针 | UB（use-after-free） | 不可能（GC 保活，除非 `__del__` 误用） |
| data race | UB，需 mutex/atomic | GIL 保引用计数安全，多线程仍受限 |
| 错误表现 | segfault/core dump（无 traceback） | 抛异常 + traceback |
| 调试难度 | 极高（ASan/TSan/UBSan） | 低（异常明确） |


## 4. 数学原理 / 公式

### 4.1 shared_ptr 引用计数的线程安全

`shared_ptr` 的 refcnt 用 atomic 操作（`std::atomic<int>` 或平台等价）：

- `shared_ptr(const shared_ptr& r)`：`refcnt.fetch_add(1, memory_order_relaxed)`（+1 原子）。
- `~shared_ptr()`：`if (refcnt.fetch_sub(1, memory_order_release)==1) ... release`（-1 原子，归零才释放）。

故 refcnt 本身线程安全（多线程复制/销毁同一 shared_ptr 不 race）。但**对象内部**不自动线程安全——多个 shared_ptr 持有同一对象，多线程通过它们改对象内部数据仍需 mutex（shared_ptr 只管生命周期，不管对象内容）。

### 4.2 循环引用的 refcnt 不归零

设 A 持 `shared_ptr<B>`，B 持 `shared_ptr<A>`。初始 A.refcnt=1（外部）+1（B 持）=2，B.refcnt=1（A 持）。外部释放 A 的 shared_ptr：A.refcnt=1（B 仍持），不析构。B 的 refcnt 仍 1（A 持，A 未析构）——**互相保活，永不归零**→泄漏。

用 `weak_ptr` 打断：A 持 `weak_ptr<B>`（不增 B.refcnt），B 持 `shared_ptr<A>`。外部释放 A：A.refcnt=0，A 析构→A 的 weak_ptr<B> 析构（不动 B.refcnt）。B.refcnt 独立，正常释放。循环断开。

### 4.3 data race 的"丢失更新"概率

`counter++` = load / add / store 三步（非原子）。两线程并发跑 $N$ 次：
- 每次"丢失"概率：两线程的 load-add-store 完全交错（A load → B load → A store → B store，B 覆盖 A）。
- 期望丢失次数 $\approx$ 交叠概率 $\times N$。对 $N=10^6$、高并发，丢失可达百万级，counter 远小于 $2N$。
- `atomic` 把三步合成一条原子指令（如 `lock inc`），无丢失。

### 4.4 Amdahl 与多线程加速

单线程段 $s$，并行段 $1-s$，$n$ 线程：$S_n=\dfrac{1}{s+(1-s)/n}$。但 C++ 多线程还需扣**锁开销**——若并行段大量抢锁（临界区占比 $c$），实际加速 $S_n\approx\dfrac{1}{s+c+(1-s-c)/n}$（临界段串行）。无锁（atomic/lock-free）压 $c$。

### 4.5 内存序（memory_order）与可见性

C++ atomic 的内存序：`relaxed`（只保原子，不保可见性）、`acquire`（读后操作不重排到前）、`release`（写前操作不重排到后）、`seq_cst`（全局顺序，默认，最严但最慢）。data race 不止"同时写"，还包括"一个线程写、另一个线程读但读不到新值"（缓存一致性/重排）。`acquire-release` 配对保证 release 的写在 acquire 后对其他线程可见。误用 relaxed 可能读到旧值——UB 的一种。


## 5. 代码示例（可选）

### 5.1 RAII（lock_guard 异常安全）

```cpp
#include <mutex>
std::mutex mtx;
void safe_work() {
    std::lock_guard<std::mutex> lk(mtx);  // 构造 lock
    do_critical();                         // 抛异常也安全 (栈展开析构 lk -> unlock)
}  // 析构 unlock

void unsafe_work() {
    mtx.lock();
    do_critical();   // 抛异常 -> unlock 不执行 -> 死锁
    mtx.unlock();
}
```

### 5.2 shared_ptr + weak_ptr 打断循环

```cpp
#include <memory>
struct Node : std::enable_shared_from_this<Node> {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;   // weak 打断循环
    ~Node() { /* 释放资源 */ }
};

auto a = std::make_shared<Node>();   // a.refcnt=1
auto b = std::make_shared<Node>();   // b.refcnt=1
a->next = b;   // b.refcnt=2
b->prev = a;   // a.refcnt 不变 (weak 不增) = 1
// a 出作用域: a.refcnt=0 -> a 析构 -> a->next 析构 -> b.refcnt=1 -> b 正常释放
// 无泄漏
```

### 5.3 data race vs atomic

```cpp
#include <thread>
#include <atomic>
#include <mutex>

int bad_counter = 0;                    // 非原子
std::atomic<int> good_counter{0};        // 原子
std::mutex mtx;
int locked_counter = 0;

void bad() { for(int i=0;i<1'000'000;i++) bad_counter++; }       // data race, 丢失更新
void good() { for(int i=0;i<1'000'000;i++) good_counter.fetch_add(1); }  // 原子
void locked() { for(int i=0;i<1'000'000;i++) { std::lock_guard<std::mutex> lk(mtx); locked_counter++; } }

int main() {
    std::thread t1(bad), t2(bad);  t1.join(); t2.join();
    std::thread t3(good), t4(good); t3.join(); t4.join();
    // bad_counter << 2e6 (丢失), good_counter == 2e6 (原子)
}
```

### 5.4 use-after-free（ASan 会抓）

```cpp
int* p = new int(42);
delete p;
*p = 99;   // use-after-free! UB
// 编译: g++ -fsanitize=address -> ASan 报 "heap-use-after-free"
// 无 ASan: 可静默错, 可 segfault, 可延迟崩
```

### 5.5 ML 场景：CUDA stream 的 RAII 封装

```cpp
class CudaStream {
    cudaStream_t stream_;
public:
    CudaStream() { cudaStreamCreate(&stream_); }   // 构造获取
    ~CudaStream() { cudaStreamDestroy(stream_); }  // 析构释放 (异常也安全)
    operator cudaStream_t() const { return stream_; }
    void sync() { cudaStreamSynchronize(stream_); }
};

void kernel_launch() {
    CudaStream s;                    // RAII, 出作用域自动 destroy
    my_kernel<<<grid, block, 0, s>>>(...);
    s.sync();
}   // s 析构 -> cudaStreamDestroy (即使 sync 抛异常)
```


## 6. 与其他知识点的关系

- **上游（依赖）**: C++ 对象模型、操作系统进程/线程/锁、CUDA stream/event 生命周期
- **下游（应用）**: [[推理引擎源码|vLLM]]/Megatron/Torch ATen 的 C++ 层、CUDA stream/event 管理、block manager 的 C++、性能优化（无锁数据结构）
- **对比 / 易混**:
  - **C++ 无 GC vs Python GC**：C++ 靠 RAII/智能指针 + 程序员，Python 靠引用计数 + GC（GIL 保引用计数线程安全）。C++ 错是 UB（难调），Python 错是异常（易调）。
  - **shared_ptr vs Python 引用计数**：两者都是引用计数，但 Python 的 ob_refcnt 由 GIL 保线程安全（无竞争），shared_ptr 的 refcnt 用 atomic（多线程安全但开销略大）。
  - **std::thread vs Python threading**：C++ 无 GIL（多线程真并行），但 data race 需程序员管；Python 有 GIL（引用计数安全但 CPU 密集串行）。
  - **C++ 多线程 vs [[Python multiprocessing与GIL]] 多进程**：C++ 多线程真并行多核（无 GIL），Python 多线程受 GIL 退化成多进程才并行。


## 7. 常见误区与易错点

> [!warning] 误区1：shared_ptr 线程安全所以对象也线程安全
> 错。shared_ptr 的 **refcnt** 线程安全（atomic），但**对象内部成员**不自动——多线程通过各自 shared_ptr 改对象成员仍 data race，需 mutex。shared_ptr 只管生命周期，不管对象内容。

> [!warning] 误区2：shared_ptr 万能防泄漏
> 错。循环引用（A 持 B 持 A）会泄漏，需 weak_ptr 打断。还有"this 指针返 shared_ptr"陷阱——直接 `shared_ptr<this>` 会新建 control block，与原 shared_ptr 脱节→双重释放。要用 `enable_shared_from_this` + `shared_from_this()`。

> [!warning] 误区3：data race 只是"同时写"
> 错。一个写一个读、甚至两个读（若有写重排可见性）都算。更广 UB 包括：编译器/硬件重排导致读不到新值（缓存一致性问题）。要用 atomic（带正确 memory_order）或 mutex 保可见性。

> [!warning] 误区4：UB 会抛异常
> 错。UB 是**未定义**——可崩、可静默错、可"看起来对"。不抛异常。编译器可假设 UB 不发生做激进优化，行为不可预测。这是 UB 比"明确错误"更危险的原因。

> [!warning] 误区5：析构未 join 的 thread 没事
> 错。`std::thread` 析构时若 `joinable()`（未 join/detach），调 `std::terminate` → 程序崩。必须 join 或 detach，或 move 到容器管理。

> [!warning] 误区6：用裸指针 delete 后置 nullptr 就安全
> 部分对（防再次 deref nullptr 易定位），但若有**其他指针副本**指向同一内存，那些仍悬垂（use-after-free）。根治是智能指针——一个所有权源，其他用 weak_ptr 观察。

> [!tip] 实践：RAII + 智能指针是 C++ 安全基石
> 资源（fd/锁/stream/socket）全包进 RAII 类，靠作用域自动管；共享对象用 shared_ptr（配 weak_ptr 断循环）；独占用 unique_ptr（零开销）。几乎不用裸 new/delete。

> [!tip] 实践：ML C++ 层必开 sanitizer
> 编译加 `-fsanitize=address,undefined,thread`（ASan/UBSan/TSan），跑测试时抓 use-after-free/data race/越界。是 C++ UB 的系统化检测，CI 必备。详见 §8.1。

> [!tip] 实践：多线程共享数据默认上锁
> 除非 profile 证明锁是瓶颈，否则用 mutex（最安全）。atomic 只用于简单计数/标志，复杂结构 lock-free 极易写错（ABA、内存序）。


## 8. 延伸细节

### 8.1 sanitizer（ASan/TSan/UBSan）

- **ASan（AddressSanitizer）**：抓 heap-use-after-free / heap-buffer-overflow / stack-use-after-return。编译 `-fsanitize=address -g`，运行慢 2x、内存 3x，但抓 90% 内存错误。
- **TSan（ThreadSanitizer）**：抓 data race。`-fsanitize=thread`，慢 5-10x，只测多线程。
- **UBSan（UndefinedBehaviorSanitizer）**：抓整数溢出/空指针 deref/越界/类型混淆。`-fsanitize=undefined`。
- 三者**不兼容同用**（ASan vs TSan 互斥），CI 分轮跑。ML C++ 层（vLLM/Megatron kernel）必开。

### 8.2 弱指针提升的 race（`weak_ptr::lock`）

```cpp
std::weak_ptr<Foo> w = p;   // p 是 shared_ptr
// 线程 A: auto sp = w.lock(); if (sp) use(sp);   // 提升, 若对象在则 sp 持有
// 线程 B: p.reset();   // 可能释放对象
// A 的 lock 是原子的 (要么提升成功拿到 live sp, 要么返回空) -> 安全
```

`weak_ptr::lock()` 是原子操作（check refcnt + 提升），线程安全。但提升后用 sp 期间对象保证活着（sp 持有 refcnt）。

### 8.3 移动语义与生命周期

`std::move` 把对象资源**转移所有权**而非拷贝——`unique_ptr` 只能 move（独占），`shared_ptr` move 不增 refcnt（转移控制块）。move 后源对象处于"valid but unspecified"状态（应不再用）。move 是 C++ 避免"拷贝大对象"的核心（如 `vector` 的 move 后源变空）。误用 move 后的源（如 `*moved_from`）是 UB 或无效。

### 8.4 CUDA 内存的生命周期

CUDA 的 `cudaMalloc`/`cudaFree`、stream 的 create/destroy 同样需 RAII。Torch 的 `c10::cuda::CUDAStream` 是 RAII 封装。vLLM 的 block manager 管理 KV cache 的 CUDA 内存池，块的生命周期（分配/引用计数/释放）用 RAII + 引用计数管理——悬垂 CUDA 块被复用会致 KV 错乱（采样到错的 token），是 vLLM C++ 层的典型 UB 风险。

### 8.5 C++ 与 Python 互操作的生命周期

PyTorch 的 C++ 扩展里，`torch::Tensor` 是 `shared_ptr` 管的（C++ 层也引用计数）。Python 持有 Tensor 时 C++ 的 intrusive_ptr 也持有，跨语言边界一致。自定义 C++ op 返回 tensor 时要正确管理生命周期（别返悬垂指针指向栈变量）。`pybind11` 的 `py::object` 也是 RAII（析构 decref Python 引用）。

### 8.6 无锁数据结构与 ABA

`std::atomic` 的 `compare_exchange` 可造无锁栈/队列，但有 **ABA 问题**：线程读 A→被中断→别的线程 pop A push B 再 push A→原线程 CAS 成功（值还是 A）但链表结构已变→错。无锁结构极难写对，ML 框架除极少数热路径外用 mutex。

### 8.7 与 [[Python multiprocessing与GIL]] 的分工

ML 工程的并发体系：Python 层用 multiprocessing（绕 GIL）+ asyncio（IO 并发），C++ 层用 std::thread（真并行多核，无 GIL）+ RAII/智能指针管生命周期。两者边界——Python 调度，C++ 执行 kernel/通信。C++ 层的 data race/悬垂指针是 Python 层遇不到的类问题（Python 有 GC/GIL 兜底）。详见 [[Python multiprocessing与GIL]] §3.8、[[asyncio]] §8.6。


---
相关: [[并发与生命周期]] | [[Python multiprocessing与GIL]] | [[asyncio]] | [[20-可靠性与开源工程]] | [[推理引擎源码]] | [[torch.distributed]] | [[数值类型与精度]]
