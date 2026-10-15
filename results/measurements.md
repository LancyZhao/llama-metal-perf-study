# Raw measurements

Every number here was measured on the machine below — nothing is estimated.
These back the findings in the [README](../README.md).

## Environment
- Apple M2 (8-core: 4P + 4E), 16 GB unified memory, macOS 15.6
- GPU: Metal 3 (MTLGPUFamilyApple8), ~100 GB/s memory bandwidth, 11.4 GB recommended working set
- llama.cpp @ `3e390af`, built `-DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release`, Apple clang 17
- Model: Qwen2.5-3B-Instruct-Q4_K_M (bartowski), 1.93 GB, vocab 151,936

## Baseline throughput — `llama-bench -ngl 99 -t 4 -r 10`
| workload | t/s (mean ± sd) |
|---|---|
| decode `tg128` | 42.69 ± 0.11 |
| decode `tg128` @ KV depth 512 | 41.95 ± 0.25 |
| prefill `pp512` | 437.43 ± 0.92 |
| prefill `pp512` @ KV depth 512 | 424.18 ± 1.19 |
| prefill+decode `pp512+tg128` | 151.74 ± 0.54 |

**Memory-bandwidth check:** 42.69 tok/s × 1.93 GB/tok ≈ **82 GB/s ≈ 82 % of the
M2's ~100 GB/s**. Single-stream decode is memory-bandwidth bound.

## H1 — sampler candidate-array construction (`bench/micro_set_logits.cpp`)
`n_vocab=151936`, 20000 steps, `clang++ -O3`:
```
full_ramp        ~29000 ns/step   baseline: cur[i] = {i, logit, 0}
full_cachedid    ~29500 ns/step   full write, id read from array  ~ baseline
partial_skipid   ~61000 ns/step   the "de-dup": skip id, write logit+p   -> 2.1x SLOWER
logit_only       ~46000 ns/step   write only logit                       -> 1.6x slower
soa_memcpy        ~9300 ns/step   struct-of-arrays memcpy                -> 3.2x faster
```
Fill cost ≈ 29 µs vs a decode step ≈ 23 ms → the sampler build is **< 0.2 % of decode**.
## H2 — Metal shader compile cold start (`llama-cli` startup wall clock)
| run | wall |
|---|---|
| first ever (OS shader cache cold), all 35 libs | ~14.2 s |
| successive runs, all 35 libs | 6.83 s → 1.1 s |
| skip 8 non-transformer libs (`GGML_METAL_SKIP_LIBS`) | ~1.1 s |
| skip 18 unused libs (incl. quantized-KV flash-attn) | ~1.1 s |

Greedy output byte-identical with/without the skip. macOS caches compiled
shaders on disk, so the warm-cache start is ~1.1 s **regardless** — skipping
unused libraries only helps the single cold compile right after a rebuild.

## H3 — batched decode throughput — `llama-batched-bench -npp 128 -ntg 64`
Baseline (`S_TG` = aggregate generation tok/s across `B` parallel sequences):
| B | S_TG t/s | ms per decode step |
|---|---|---|
| 1 | 41.4 | 24.1 |
| 2 | 53.2 | 37.6 |
| 4 | 93.9 | 42.6 |
| 8 | 174.3 | 45.9 |
| 16 | 227.4 | 70.4 |

Scales 5.5× to B=16 (spare compute exists), but a **+57 % per-step cost cliff at
B=1→2** — the one anomaly worth chasing.

Lowering the K-quant small-batch mat-vec gate to `ne11 ≥ 2` (upstream uses `≥ 4`):
| B | baseline S_TG | modified S_TG |
|---|---|---|
| 1 | 41.4 | 39.6 |
| 2 | 53.2 | **42.7 (−20 %)** |
| 4 | 93.9 | 95.6 |
| 8 | 174.3 | 180.6 |
| 16 | 227.4 | 228.5 |

Generation byte-identical (numerically correct), but B=2 **regresses 20 %** →
upstream's `≥ 4` gate is already the right routing choice for K-quants.

## Long-context scaling (lead, not yet confirmed) — `llama-bench -d 0,512,2048,4096,8192`
| KV depth | decode tg128 t/s | prefill pp512 t/s |
|---|---|---|
| 0 | 42.17 ± 0.33 | 433.42 ± 1.89 |
| 512 | 39.34 ± 1.66 | 420.31 ± 3.95 |
| 2048 | 38.60 ± 1.20 | 386.89 ± 2.37 |
| 4096 | 36.16 ± 0.68 | 349.34 ± 1.48 |
| 8192 | 32.43 ± 0.95 | 294.48 ± 1.19 |

Roofline: at d=0, decode achieves 42.2 × 1.93 ≈ **80 GB/s**. At d=8192 the KV
cache adds ≈ 0.30 GB/token (36 layers × 2 × 2 KV heads × 128 × 2 B × 8192), so a
pure 80 GB/s model predicts 80/(1.93+0.30) ≈ **36 t/s** — but it measures **32.4**,
i.e. achieved bandwidth fell to ≈ **71 GB/s**. The attention / KV-read path is not
saturating bandwidth the way the weight GEMV does → the one place with plausible
headroom. Confirming needs per-op GPU profiling (unavailable here) or `fa_*` kernel work.

### KV-cache quantization at d=8192 (`-fa 1`, -r 3)
| KV cache | decode tg128 t/s |
|---|---|
| f16 | 31.98 ± 0.64 |
| q8_0 | 27.38 ± 0.52 (−14 %) |
| q4_0 | 27.85 ± 0.36 (−13 %) |

Quantizing the KV cache is **slower** on Metal: the quantized KV is dequantized to
F16 before the attention kernel, and that dequant costs more than the bandwidth it
saves (KV ≈ 0.3 GB vs 1.9 GB of weights). KV-quant saves memory, not time.
