# 复现步骤

所有数字都依赖具体机器(Apple M2 / 16GB / macOS 15.6 / Metal 3)。换机器需重测。

## 1. 构建 llama.cpp (Metal)

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && git fetch --depth 1 origin 3e390af && git checkout 3e390af  # 本研究所用版本
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-bench llama-cli llama-batched-bench -j
```

## 2. 下载模型 (~1.9GB)

```bash
curl -L --fail -o Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf?download=true"
```

## 3. 基线

改 `bench/run_bench.sh` 顶部的路径后:

```bash
bench/run_bench.sh baseline                       # 跑 llama-bench,存 JSON
python3 bench/compare.py results/baseline.json    # 渲染表格;两份则 compare.py before.json after.json
```

## 4. 三个实验

```bash
# H1  采样器候选表微基准(独立,无需 llama.cpp)
clang++ -O3 -std=c++17 bench/micro_set_logits.cpp -o /tmp/micro && /tmp/micro

# H2  冷启动:全编 vs 跳过未用库(先 git apply patches/h2-*.patch 并重编)
time build/bin/llama-cli -m MODEL -p hi -st -n 1 -ngl 99
GGML_METAL_SKIP_LIBS="ssm,wkv,gated_delta_net,conv,tri,solve_tri,upscale,pool" \
  time build/bin/llama-cli -m MODEL -p hi -st -n 1 -ngl 99

# H3  并发吞吐:基线 vs 放宽门槛(git apply patches/h3-*.patch,重编,对比 B=2)
build/bin/llama-batched-bench -m MODEL -c 4096 -b 2048 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8,16 -ngl 99
```

`patches/h2`、`patches/h3` 是对上述 commit 的 git diff,`git apply` 即可;`h1` 的改动见 README 中的 diff(已证伪,未单独留 patch)。
