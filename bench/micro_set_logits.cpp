// Microbenchmark: how to refresh the full-vocab candidate array each decode step.
// llama_token_data is AoS: { int32 id; float logit; float p; } (12 bytes).
//
// Variants:
//   full_ramp     : cur[i] = {i, logits[i], 0}          baseline (full sequential write)
//   full_cachedid : cur[i] = {ids[i], logits[i], 0}      full write, id read from array
//   partial_skipid: cur[i].logit=..; cur[i].p=0          "de-dup" idea (skip id, partial write)
//   logit_only    : cur[i].logit=..                      most partial (4 of 12 bytes)
//   soa_memcpy    : memcpy(logit_arr, logits)            what a struct-of-arrays layout would get
//
// Build: clang++ -O3 -std=c++17 micro_set_logits.cpp -o micro_set_logits
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

struct token_data { int32_t id; float logit; float p; };

using clk = std::chrono::high_resolution_clock;
static long long ns(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

int main(int argc, char ** argv) {
    const int n_vocab = argc > 1 ? atoi(argv[1]) : 151936;
    const int steps   = argc > 2 ? atoi(argv[2]) : 20000;

    std::vector<float>      logits(n_vocab);
    std::vector<int32_t>    ids(n_vocab);
    std::vector<token_data> cur(n_vocab);
    std::vector<float>      logit_soa(n_vocab);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> d(-20.f, 20.f);
    for (int i = 0; i < n_vocab; ++i) { logits[i] = d(rng); ids[i] = i; cur[i].id = i; }

    volatile double sink = 0;

    auto full_ramp      = [&](int n){ for (int s=0;s<n;++s){ logits[s%n_vocab]+=1e-6f; for(int i=0;i<n_vocab;++i) cur[i]=token_data{i,logits[i],0.f};            sink+=cur[s%n_vocab].logit; } };
    auto full_cachedid  = [&](int n){ for (int s=0;s<n;++s){ logits[s%n_vocab]+=1e-6f; for(int i=0;i<n_vocab;++i) cur[i]=token_data{ids[i],logits[i],0.f};        sink+=cur[s%n_vocab].logit; } };
    auto partial_skipid = [&](int n){ for (int s=0;s<n;++s){ logits[s%n_vocab]+=1e-6f; for(int i=0;i<n_vocab;++i){ cur[i].logit=logits[i]; cur[i].p=0.f; }         sink+=cur[s%n_vocab].logit; } };
    auto logit_only     = [&](int n){ for (int s=0;s<n;++s){ logits[s%n_vocab]+=1e-6f; for(int i=0;i<n_vocab;++i)  cur[i].logit=logits[i];                          sink+=cur[s%n_vocab].logit; } };
    auto soa_memcpy     = [&](int n){ for (int s=0;s<n;++s){ logits[s%n_vocab]+=1e-6f; memcpy(logit_soa.data(), logits.data(), n_vocab*sizeof(float));              sink+=logit_soa[s%n_vocab]; } };

    struct V { const char * name; void (*noop)(); };
    auto time_it = [&](const char * name, auto fn){
        fn(50);                               // warm
        auto a = clk::now(); fn(steps); auto b = clk::now();
        printf("  %-15s %9.1f ns/step\n", name, (double) ns(a,b)/steps);
        return (double) ns(a,b)/steps;
    };

    printf("n_vocab=%d steps=%d\n", n_vocab, steps);
    double base = time_it("full_ramp",      full_ramp);
    time_it("full_cachedid",  full_cachedid);
    double mine = time_it("partial_skipid", partial_skipid);
    time_it("logit_only",     logit_only);
    double soa  = time_it("soa_memcpy",     soa_memcpy);
    printf("my change (partial_skipid) vs baseline (full_ramp): %.2fx  (%+.1f%% time)\n", base/mine, (mine/base-1)*100);
    printf("ideal SoA (soa_memcpy)     vs baseline (full_ramp): %.2fx\n", base/soa);
    printf("[sink=%g]\n", (double) sink);
    return 0;
}
