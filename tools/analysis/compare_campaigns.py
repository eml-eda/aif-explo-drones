"""Paired comparison of evaluation campaigns run on the same seeds.

Usage:
  python tools/analysis/compare_campaigns.py NAME=<campaign_dir> NAME=<campaign_dir> [...] [--seeds 1-30]

The first campaign is the reference for the paired statistics. Only seeds present in
all campaigns are compared. Runs are matched by seed; scenes are identical when the
seed and the scenario settings (tables, targets, forced empty tables) match.
"""
import argparse
import json
import math
import os
import statistics as st


def load(campaign_dir):
    with open(os.path.join(campaign_dir, "eval_report", "eval_report.json"), encoding="utf-8") as f:
        return {r["seed"]: r for r in json.load(f)["run_metrics"]}


def ok(r):
    return r["reason"] == "done"


def parse_seeds(text):
    lo, hi = text.split("-")
    return set(range(int(lo), int(hi) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaigns", nargs="+", help="NAME=path")
    ap.add_argument("--seeds", default=None, help="e.g. 1-30")
    args = ap.parse_args()

    named = [c.split("=", 1) for c in args.campaigns]
    data = {name: load(path) for name, path in named}
    seeds = set.intersection(*(set(d) for d in data.values()))
    if args.seeds:
        seeds &= parse_seeds(args.seeds)
    seeds = sorted(seeds)
    print(f"{len(seeds)} common seeds\n")

    print("seed | " + " | ".join(name for name, _ in named))
    for s in seeds:
        cells = [f"{'ok' if ok(d[s]) else 'TO'} {d[s]['elapsed_s']:4.0f}s {d[s]['found_count']}" for d in data.values()]
        print(f"{s:4d} | " + " | ".join(cells))
    print()
    for name, d in data.items():
        R = [d[s] for s in seeds]
        succ = [r for r in R if ok(r)]
        print(f"{name}: success {len(succ)}/{len(R)} = {len(succ) / len(R):.2f}; "
              f"mean time {st.mean(r['elapsed_s'] for r in R):.0f}s "
              f"(successes {st.mean(r['elapsed_s'] for r in succ) if succ else float('nan'):.0f}s); "
              f"found {st.mean(r['found_count'] for r in R):.2f}; "
              f"coverage {st.mean(r['coverage_ratio'] for r in R):.2f}")
    ref_name, ref = named[0][0], data[named[0][0]]
    print()
    for name, d in list(data.items())[1:]:
        a = sum(ok(d[s]) and not ok(ref[s]) for s in seeds)
        b = sum(ok(ref[s]) and not ok(d[s]) for s in seeds)
        n, k = a + b, min(a, b)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        diff = [int(ok(d[s])) - int(ok(ref[s])) for s in seeds]
        m, se = st.mean(diff), st.stdev(diff) / math.sqrt(len(diff))
        dt = [d[s]["elapsed_s"] - ref[s]["elapsed_s"] for s in seeds]
        mt, set_ = st.mean(dt), st.stdev(dt) / math.sqrt(len(dt))
        print(f"{name} vs {ref_name}: only {name} {a}, only {ref_name} {b}, McNemar exact p={p:.2f}; "
              f"success diff {m:+.3f} [95% CI {m - 1.96 * se:+.2f}, {m + 1.96 * se:+.2f}]; "
              f"time diff {mt:+.1f}s [{mt - 1.96 * set_:+.1f}, {mt + 1.96 * set_:+.1f}]")


if __name__ == "__main__":
    main()
