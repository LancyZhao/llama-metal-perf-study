#!/usr/bin/env python3
"""Render / compare llama-bench JSON results.

Usage:
  compare.py RESULT.json                 # pretty table for one run
  compare.py BEFORE.json AFTER.json      # before/after delta table

llama-bench emits a JSON array; each row carries n_prompt/n_gen/n_depth and
avg_ts / stddev_ts (tokens per second). We key rows by the (prompt,gen,depth)
shape so the same workload lines up across two builds.
"""
import json, sys


def load(path):
    with open(path) as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        key = (r.get("n_prompt", 0), r.get("n_gen", 0), r.get("n_depth", 0))
        out[key] = r
    return out


def label(key):
    p, g, d = key
    if g == 0:
        name = f"pp{p}"          # prefill / prompt-processing
    elif p == 0:
        name = f"tg{g}"          # decode / text-generation
    else:
        name = f"pp{p}+tg{g}"
    return name + (f" @d{d}" if d else "")


def ts(r):
    return r.get("avg_ts", 0.0), r.get("stddev_ts", 0.0)


def one(path):
    data = load(path)
    print(f"| workload | t/s (mean +/- sd) |")
    print(f"|---|---|")
    for key in sorted(data):
        a, s = ts(data[key])
        print(f"| {label(key)} | {a:8.2f} +/- {s:5.2f} |")


def two(before, after):
    b, a = load(before), load(after)
    print(f"| workload | before t/s | after t/s | delta | signif? |")
    print(f"|---|---|---|---|---|")
    for key in sorted(set(b) | set(a)):
        if key not in b or key not in a:
            continue
        ba, bs = ts(b[key]); aa, as_ = ts(a[key])
        delta = (aa - ba) / ba * 100 if ba else 0.0
        # crude significance: gap larger than the summed stddevs
        signif = "yes" if abs(aa - ba) > (bs + as_) else "no"
        sign = "+" if delta >= 0 else ""
        print(f"| {label(key)} | {ba:8.2f} +/- {bs:5.2f} | {aa:8.2f} +/- {as_:5.2f} "
              f"| {sign}{delta:5.1f}% | {signif} |")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        one(args[0])
    elif len(args) == 2:
        two(args[0], args[1])
    else:
        print(__doc__)
        sys.exit(1)
