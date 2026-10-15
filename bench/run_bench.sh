#!/usr/bin/env bash
# Benchmark one build of llama.cpp and save a clean JSON result tagged by label+git-rev.
# Run it once before a change (label=baseline) and once after (label=optimized),
# then diff with compare.py.
#
# Usage: run_bench.sh <label>        e.g.  run_bench.sh baseline
# Env overrides: MODEL, NGL, THREADS, REPS, PROMPTS, GENS, PG, DEPTHS
set -euo pipefail

LABEL="${1:?usage: run_bench.sh <label>  (e.g. baseline | optimized)}"
ROOT="$HOME/llm-perf"
BR="$ROOT/llama.cpp"
BIN="$BR/build/bin/llama-bench"
MODEL="${MODEL:-$ROOT/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf}"
OUT="$ROOT/bench/results"
mkdir -p "$OUT"

# --- knobs held CONSTANT across before/after (only the code may change) ---
NGL="${NGL:-99}"        # offload all layers to the Metal GPU
THREADS="${THREADS:-4}" # pin to the M2 performance cores for low variance
REPS="${REPS:-10}"      # repetitions -> llama-bench reports mean +/- stddev
PROMPTS="${PROMPTS:-512}"   # prompt-processing (prefill) sizes
GENS="${GENS:-128}"        # text-generation (decode) sizes
PG="${PG:-512,128}"        # one combined prefill+decode run
DEPTHS="${DEPTHS:-0,512}"  # measure decode at KV depth 0 and 512

[ -x "$BIN" ]   || { echo "ERROR: llama-bench not built at $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: model not found at $MODEL" >&2; exit 1; }

REV="$(git -C "$BR" rev-parse --short HEAD 2>/dev/null || echo nogit)"
BRANCH="$(git -C "$BR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo nogit)"
STAMP="$(date +%Y%m%d-%H%M%S)"
TAG="${LABEL}_${BRANCH}_${REV}_${STAMP}"
JSON="$OUT/${TAG}.json"

echo "== llama-bench  label=$LABEL  branch=$BRANCH  rev=$REV =="
echo "   model=$(basename "$MODEL")  ngl=$NGL  threads=$THREADS  reps=$REPS"
echo "   prompt=$PROMPTS gen=$GENS pg=$PG depths=$DEPTHS"

# single invocation: md table -> stdout (human), clean json -> file (machine).
# -oe prints a second format; we send json to the file via a temp fd trick below.
"$BIN" -m "$MODEL" \
  -p "$PROMPTS" -n "$GENS" -pg "$PG" -d "$DEPTHS" \
  -ngl "$NGL" -t "$THREADS" -r "$REPS" \
  --progress -o json \
  >"$JSON" 2>"$OUT/${TAG}.stderr.log"

echo "saved: $JSON"
echo "---- table ----"
python3 "$ROOT/bench/compare.py" "$JSON" || cat "$JSON"
