# llama.cpp 在 Apple Silicon 上的推理性能调查

一次 measurement-driven 的性能工程案例:在 Apple M2 上把 llama.cpp 的 Metal 推理从头跑通、建立可复现基线,然后对三个"看起来能优化"的点逐一提出假设、做隔离实验、用数据证伪——并讲清每一个"为什么没有空间"。

> **一句话结论**:llama.cpp 的 Metal 后端在 M2 上已经贴着硬件极限。三个优化假设全部被自己的数据否掉,第四个探针则定位到唯一还有空间的场景(长上下文注意力)。而"证伪 / 定位"的过程(微基准 / A-B / 正确性验证 / 瓶颈归因)正是本项目要展示的——**会测量、懂瓶颈、不发假 win**。
>
> *(English version: [README.md](README.md))*

## 为什么是这样一个项目

性能优化的真实工作不是"我把它改快了",而是"我能证明时间花在哪、哪里还有空间、哪里没有"。在一个被反复打磨过的推理引擎上,好拿的分早被代码或操作系统拿走了;能诚实地量出这一点并说清机制,比再做一个"变快 demo"更接近真实的 infra / perf 工程。

本仓库完全可复现:基线用 `bench/run_bench.sh`,三个实验各自独立可跑,原始数据见 [`results/measurements.md`](results/measurements.md)。

## 环境与方法

| | |
|---|---|
| 机器 | Apple M2(4P+4E / 16GB 统一内存),macOS 15.6,Metal 3 |
| 引擎 | llama.cpp `@3e390af`,`-DGGML_METAL=ON` Release,Apple clang 17 |
| 模型 | Qwen2.5-3B-Instruct-Q4_K_M,1.93GB,vocab 151,936 |
| 测量 | `llama-bench`(前向吞吐)· `llama-batched-bench`(并发)· 自写微基准(采样)· `llama-cli` 墙钟(冷启动) |

## 基线与"带宽墙"

`llama-bench`(全层 offload,10 次重复,方差极小):

| 场景 | t/s |
|---|---|
| 解码 tg128 | **42.7 ± 0.1** |
| 预填 pp512 | **437.4 ± 0.9** |
| 预填+解码 pp512+tg128 | 151.7 ± 0.5 |

框定后面所有结论的一步:

> 解码 42.7 tok/s × 模型 1.93 GB ≈ **82 GB/s ≈ M2 ~100 GB/s 带宽的 82%**。每生成 1 个 token 要把整个模型读一遍 → 解码是 **memory-bound**。推论:固定量化下,让单流解码变快的唯一物理路径是"少读权重"——要么更小的量化,要么把权重读摊到更多 token(批量 / 投机解码);compute 侧再优化都撞这堵墙。

## 实验一:采样器候选表的"去冗余"（证伪）

**假设**:每生成一个 token,CPU 采样路径都把整张词表(15.2 万项)的 `{id, logit, p}` 全量重建一遍(`common/sampling.cpp` 的 `set_logits`)。`id` 恒等于下标、`p` 随后被 softmax 覆盖,只有 `logit` 真变——跳过冗余写应该更快:

```diff
-  for (token_id = 0; token_id < n_vocab; token_id++)
-      cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
+  // id 只建一次,热路径只刷 logit
+  if (cur_ids_n != n_vocab) { for (...) cur[i].id = i; cur_ids_n = n_vocab; }
+  for (...) { cur[i].logit = logits[i]; cur[i].p = 0.0f; }
```

**结果**(`bench/micro_set_logits.cpp`,5 种写法隔离微基准):改法 **慢 2.1×**。

```
full_ramp (基线整体写)   ~29 µs/step
partial_skipid (去冗余)  ~61 µs/step   ← 慢 2.1×
soa_memcpy (理想 SoA)    ~9  µs/step   ← 快 3.2×
```

**机制**:`llama_token_data` 是 AoS(12 字节),整体顺序写能走向量化 / 流式存储;"跳过 id"变成**带空洞的部分写**(每 12 字节只写中间 8),打断顺序写、触发缓存行读-改-写。对照组 `full_cachedid`(整体写、id 从数组读)≈ 基线,证明"写 id"本就免费。真正的空间在 SoA 布局(快 3.2×),但那要重构贯穿整条采样链的 `llama_token_data_array`,是跨模块大改,且采样只占解码 **<0.2%**,端到端不可见。**放弃。**

## 实验二:Metal 着色器编译的冷启动（证伪）

**现象**:进程启动时从内嵌源码编译 35 个 Metal kernel 库,首次 **~14.2s**。
**假设**:transformer 用不到其中一多半(ssm / wkv / conv / tri、各种量化 KV 的 flash-attn 变体),跳过能砍冷启动——加了 `GGML_METAL_SKIP_LIBS` 开关验证。
**结果**:启动墙钟 14.2s → 6.83s → **1.1s** 一路下降,因为 **macOS 自己把编译结果缓存到磁盘**。稳态下全编(1.1s)与跳过 18 个(1.1s)**无差别**;跳过只在"刚 rebuild 后第一次"的冷缓存有用。输出逐 token 一致。
**结论**:不是 llama.cpp 的问题,是 OS 已兜底,稳态无空间。**放弃。**

## 实验三:K 量化小批量 matmul 的 kernel 路由（证伪）

`llama-batched-bench` 显示并发解码从 41 t/s 扩到 227 t/s(B=16,**5.5×**)——spare compute 确实存在。但有一道反常的坎:**B=1→2 每步耗时 +57%**(本该近乎免费)。

**根因**(`ggml/src/ggml-metal/ggml-metal-ops.cpp` 的 `mul_mat`):高效的小批量 mat-vec 核对传统量化(Q4_0…)开放 `ne11∈[2,8]`,但对 K 系量化(Q4_K/Q5_K/…)只开放 `[4,8]`。我们的模型是 Q4_K_M,于是 B=2/3 掉出快核:

```diff
-   false) && (ne11 >= 4 && ne11 <= 8)   // K-quants
+   false) && (ne11 >= 2 && ne11 <= 8)
```

**结果**:B=2 **从 53.2 掉到 42.7 t/s(−20%)**,输出逐 token 一致(正确但更慢)。那个 ext 核在 K 量化 ne11=2/3 时比它现在走的 MMA 路更慢——**上游把门槛设成 ≥4 是对的**,B=1→2 的坎是深思熟虑的正确路由,不是 bug。**放弃。**

## 一条值得记下的线索 —— 长上下文注意力(未坐实)

和前三个死靶子不同,这一个有"带宽没吃满"的证据。吞吐随 KV 深度:

| KV 深度 | 解码 tg128 | 预填 pp512 |
|---|---|---|
| 0 | 42.2 | 433 |
| 2048 | 38.6 | 387 |
| 4096 | 36.2 | 349 |
| 8192 | 32.4(−23%) | 294(−32%) |

变慢本身是预期的(注意力随上下文变长变贵);有意思的是 roofline:短上下文解码跑在 ~80 GB/s(82% 峰值);到 8K 时 KV cache 每 token 只多读 ~0.3 GB,纯带宽模型应给出 ~36 tok/s,实测却是 **32.4**——达成带宽掉到 **~71 GB/s**。注意力 / KV 读取这条路**没把带宽吃满**,不像权重 GEMV 贴着峰值。这是全篇唯一"还有空间"的点,而且正好是实际最重要的场景(长 prompt / RAG / 长对话)。

**诚实边界**:要坐实并拿下它,需要逐算子 GPU profiling(这台没 Xcode Metal GPU-capture 工具链,拿不到)或直接改 `fa_*` FlashAttention 核——深水区,胜率未知。它和前三个的区别在于:**这里数据显示带宽有余量,而不是一个被数据否掉的假设。**

## 那到底哪里还有空间?

- **并发吞吐**:已能扩 5.5×(B=16),llama.cpp 的 continuous batching 已实现且扩得不错。
- **投机解码**:摊薄带宽成本的正解,llama.cpp 已有大量实现(`--spec-type` 多种)。
- **解码最后的 18%**(82%→100% 带宽):要重写 Metal GEMV kernel(访存 / dequant 重叠),专家级深水区且上游已很接近——低胜率,本项目不碰。
- **换后端**:CUDA / CPU / 国产加速卡后端未必像 M2 Metal 这样被榨干,是另一台机器的战场。

## 这个项目展示了什么

- 从源码构建并读懂一个大型 C++/Objective-C/Metal 推理引擎的热路径(解码循环、采样链、Metal 派发、KV cache)
- 建立低方差、可复现的基线,用 roofline / 带宽模型做瓶颈归因
- 提出假设 → 隔离实验(微基准 / A-B / 正确性对拍)→ 用数据证伪,而不是提交"看起来对"的改动
- 诚实区分"我以为能优化"和"测完发现上游是对的"——性能工程里最值钱也最少见的习惯

## 复现

见 [`SETUP.md`](SETUP.md)。基线:`bench/run_bench.sh baseline`;原始数据:[`results/measurements.md`](results/measurements.md);三个实验改动在 [`patches/`](patches/)。

```
bench/     基线 / 对比 / 微基准 工具
results/   原始测量数据(measurements.md + baseline.json)
patches/   实验改动(h2/h3 为捕获的 git diff;h1 见上文 diff)
SETUP.md   构建 llama.cpp + 下模型 + 复现步骤
```


