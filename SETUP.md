# Reproduction

Every number depends on the exact machine (Apple M2 / 16 GB / macOS 15.6 / Metal 3). Re-measure on other hardware.

## 1. Build llama.cpp (Metal)

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && git fetch --depth 1 origin 3e390af && git checkout 3e390af   # version used here
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-bench llama-cli llama-batched-bench -j
```

## 2. Download the model (~1.9 GB)

```bash
curl -L --fail -o Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf?download=true"
```

## 3. Baseline

Edit the paths at the top of `bench/run_bench.sh`, then:

```bash
bench/run_bench.sh baseline                      # runs llama-bench, saves JSON
python3 bench/compare.py results/baseline.json   # render; two files -> compare.py before.json after.json
```

## 4. The three experiments

```bash
# H1  sampler candidate-array microbench (standalone, no llama.cpp needed)
clang++ -O3 -std=c++17 bench/micro_set_logits.cpp -o /tmp/micro && /tmp/micro

# H2  cold start: all libs vs skipping unused ones (apply patches/h2-*.patch and rebuild first)
time build/bin/llama-cli -m MODEL -p hi -st -n 1 -ngl 99
GGML_METAL_SKIP_LIBS="ssm,wkv,gated_delta_net,conv,tri,solve_tri,upscale,pool" \
  time build/bin/llama-cli -m MODEL -p hi -st -n 1 -ngl 99

# H3  concurrent throughput: baseline vs relaxed gate (git apply patches/h3-*.patch, rebuild, compare B=2)
build/bin/llama-batched-bench -m MODEL -c 4096 -b 2048 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8,16 -ngl 99
```

`patches/h2` and `patches/h3` are git diffs against the commit above (apply with `git apply`); `h1`'s change is the diff shown in the README (falsified, so no standalone patch).
