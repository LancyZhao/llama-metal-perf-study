# llama.cpp inference performance study on Apple Silicon

A measurement-driven performance-engineering case study: build llama.cpp's Metal backend from source on an Apple M2, establish a reproducible baseline, then take three plausible "this looks optimizable" ideas, turn each into a hypothesis, run an isolated experiment, and let the data decide — explaining *why* there is no room in each case.

> **Bottom line:** llama.cpp's Metal backend is already up against the hardware limits on an M2. All three optimization hypotheses were falsified by their own measurements — and the *way* they were falsified (microbenchmarks, A/B tests, correctness checks, bottleneck attribution) is the point of this project: **measure, locate the bottleneck, and never ship a fake win.**

*(中文版见 [README.zh-CN.md](README.zh-CN.md))*

## Why this project

Real performance work isn't "I made it faster" — it's "I can prove where the time goes, where there's headroom, and where there isn't." On a mature, heavily-tuned inference engine the easy wins are already taken by the code or the OS; honestly measuring that and explaining the mechanism is closer to real infra/perf engineering than shipping another "it got faster" demo.

Everything here is reproducible: `bench/run_bench.sh` for the baseline, each experiment runs standalone. Raw data in [`results/measurements.md`](results/measurements.md).

## Setup

| | |
|---|---|
| Machine | Apple M2 (4P+4E / 16 GB unified memory), macOS 15.6, Metal 3 |
| Engine  | llama.cpp `@3e390af`, `-DGGML_METAL=ON` Release, Apple clang 17 |
| Model   | Qwen2.5-3B-Instruct-Q4_K_M, 1.93 GB, vocab 151,936 |
| Tools   | `llama-bench` (forward throughput) · `llama-batched-bench` (concurrency) · a custom microbench (sampling) · `llama-cli` wall clock (cold start) |

## Baseline and the bandwidth wall

`llama-bench` (all layers offloaded, 10 repetitions, tiny variance):

| workload | t/s |
|---|---|
| decode tg128 | **42.7 ± 0.1** |
| prefill pp512 | **437.4 ± 0.9** |
| prefill+decode pp512+tg128 | 151.7 ± 0.5 |

The one number that frames every result below:

> decode 42.7 tok/s × 1.93 GB model ≈ **82 GB/s ≈ 82 % of the M2's ~100 GB/s**. Every generated token re-reads the whole model → decode is **memory-bandwidth bound**. Corollary: at a fixed quantization the only physical way to speed up single-stream decode is to *read the weights less* — a smaller quant, or amortizing the weight read across more tokens (batching / speculative decoding). Optimizing the compute side just hits this wall.

## Experiment 1 — "de-duplicating" the sampler candidate array (falsified)

**Hypothesis.** Every generated token, the CPU sampler path rebuilds the entire vocabulary (151,936 entries) of `{id, logit, p}` in `common/sampling.cpp`'s `set_logits`. The `id` always equals the index and `p` is overwritten by softmax later, so only `logit` actually changes — skipping the redundant writes should be faster:

```diff
-  for (token_id = 0; token_id < n_vocab; token_id++)
-      cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
+  // write ids once; only refresh logit on the hot path
+  if (cur_ids_n != n_vocab) { for (...) cur[i].id = i; cur_ids_n = n_vocab; }
+  for (...) { cur[i].logit = logits[i]; cur[i].p = 0.0f; }
```

**Result** (`bench/micro_set_logits.cpp`, 5 layouts, isolated): the change is **2.1× slower**.

```
full_ramp (baseline full write)   ~29 µs/step
partial_skipid (the "de-dup")     ~61 µs/step   <- 2.1x slower
soa_memcpy (ideal SoA)            ~9  µs/step   <- 3.2x faster
```

**Mechanism.** `llama_token_data` is an array-of-structs (12 bytes). A full sequential write vectorizes / stream-stores well; "skipping id" turns it into a **strided partial write** (8 of every 12 bytes), which breaks the sequential store and triggers cache-line read-modify-write. The control `full_cachedid` (full write, id read from an array) ≈ baseline, proving the id write was already free. The real headroom is a struct-of-arrays layout (3.2×), but that means refactoring `llama_token_data_array` across the whole sampler chain — and the sampler is **< 0.2 % of decode**, so it is invisible end-to-end. **Dropped.**

## Experiment 2 — Metal shader-compile cold start (falsified)

**Observation.** On startup the backend compiles 35 Metal kernel libraries from embedded source — **~14.2 s** the first time.
**Hypothesis.** A transformer never uses most of them (ssm / wkv / conv / tri, the quantized-KV flash-attn variants); skipping them should cut cold start. Verified with a `GGML_METAL_SKIP_LIBS` switch.
**Result.** Startup wall fell 14.2 s → 6.83 s → **1.1 s** across successive runs, because **macOS caches compiled shaders on disk**. In steady state, compiling all 35 (1.1 s) and skipping 18 (1.1 s) are **indistinguishable**; the skip only helps the single cold compile after a rebuild. Output byte-identical.
**Conclusion.** Not a llama.cpp problem — the OS already handles it. No steady-state headroom. **Dropped.**

## Experiment 3 — K-quant small-batch matmul routing (falsified)

`llama-batched-bench` shows concurrent decode scaling from 41 to 227 tok/s (B=16, **5.5×**) — spare compute does exist. But there's one anomaly: **a +57 % per-step cost going from B=1 to B=2** (which should be nearly free).

**Root cause** (`ggml/src/ggml-metal/ggml-metal-ops.cpp`, `mul_mat`): the efficient small-batch mat-vec kernel is enabled for legacy quants (Q4_0…) at `ne11∈[2,8]`, but for K-quants (Q4_K/Q5_K/…) only at `[4,8]`. Our model is Q4_K_M, so B=2/3 fall out of the fast kernel:

```diff
-   false) && (ne11 >= 4 && ne11 <= 8)   // K-quants
+   false) && (ne11 >= 2 && ne11 <= 8)
```

**Result.** B=2 **drops from 53.2 to 42.7 tok/s (−20 %)**, output byte-identical (correct but slower). That ext kernel is slower for K-quants at ne11=2/3 than the MMA path it currently takes — **upstream's `≥ 4` gate is the right call**, and the B=1→2 cliff is a deliberate, correct routing decision, not a bug. **Dropped.**

## So where *is* there headroom?

- **Concurrent throughput** already scales 5.5× (B=16); llama.cpp's continuous batching is implemented and scales well.
- **Speculative decoding** is the right way to amortize the bandwidth cost, and llama.cpp already ships many variants (`--spec-type`).
- **The last 18 %** of decode (82 % → 100 % of bandwidth) needs a rewritten Metal GEMV kernel (access pattern / dequant overlap) — expert territory, and upstream is already close. Low odds; out of scope here.
- **Other backends** (CUDA / CPU / domestic accelerators) may not be as squeezed as M2 Metal — a different machine's battle.

## What this project demonstrates

- Building and reading the hot path of a large C++/Objective-C/Metal inference engine (decode loop, sampler chain, Metal dispatch, KV cache)
- Establishing a low-variance, reproducible baseline and attributing the bottleneck with a roofline / bandwidth model
- Hypothesis → isolated experiment (microbench / A-B / correctness diff) → falsification by data, instead of shipping a change that merely "looks right"
- Honestly separating "I thought this was optimizable" from "measured it, upstream was right" — the most valuable and least common habit in performance work

## Reproduce

See [`SETUP.md`](SETUP.md). Baseline: `bench/run_bench.sh baseline`; raw data: [`results/measurements.md`](results/measurements.md); the experiment diffs are in [`patches/`](patches/).

```
bench/     baseline / compare / microbench tools
results/   raw measurements (measurements.md + baseline.json)
patches/   experiment diffs (h2/h3 captured git diffs; h1 is the diff above)
SETUP.md   build llama.cpp + download model + reproduction steps
```


