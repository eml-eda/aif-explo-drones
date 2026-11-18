#!/usr/bin/env python3
"""How the ZOI cost on GAP8 grows with the size of the map.

Reads gap8/main.c outputs for several map sizes (the Webots room and the tilings
of tools/c_port/make_tiled_room_header.py, all from the same recorded flights),
groups them by number of cells (the "map R x C cells" line; outputs without it
are the 13 x 11 room) and reports, per size:

- belief update per frame: geometry + evidence (the frame update), which works
  on the cells in view, not on the whole map;
- planner call: belief refresh + IG search + flight times, mean, 95th
  percentile and worst over all calls (main.c's plan_call_max: the costliest
  call of each frame with a plan);
- frame: everything from the detector output to the command, mean and worst;
- exact IG evaluations per call, out of 8 candidates per cell.

Then, for each quantity, the least-squares line y = a + b * cells (R^2) and the
power-law exponent k of y ~ cells^k (log-log fit): k close to 1 means linear
growth, close to 0 means independent of the map size.

    python3 tools/c_port/scaling_gap8.py embedded/zoi/gap8/batch/flight \\
        embedded/zoi/gap8/batch/tiles_1x2/flight embedded/zoi/gap8/batch/tiles_2x2/flight
    python3 tools/c_port/scaling_gap8.py --seeds-upto 24 DIR ...   # only flights of seeds 1..24 at every size
    python3 tools/c_port/scaling_gap8.py --host DIR ...   # host builds of main.c: "cycles" are ns
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import statistics

COLS = ["frame", "geometry", "update", "beliefs", "planner_IG", "planner_T", "navigator",
        "belief_update", "planner", "total", "seen_cells", "active_points", "IG_cands", "T_cands",
        "plans", "plan_ok", "plan_call_max", "IG_cands_all"]


def parse(path: str, host: bool) -> dict | None:
    rows, mhz, cells = [], None, 13 * 11
    with open(path, errors="replace") as f:
        for line in f:
            p = line.strip().split(";")
            if len(p) == len(COLS) and p[0].isdigit():
                rows.append(dict(zip(COLS, (int(x) for x in p))))
                continue
            m = re.search(r"map (\d+) x (\d+) cells", line)
            if m:
                cells = int(m.group(1)) * int(m.group(2))
            m = re.search(r"\(@(\d+(?:\.\d+)?) MHz", line)
            if m:
                mhz = float(m.group(1))
    if not rows or (mhz is None and not host):
        return None
    k = 1e-6 if host else 1.0 / (mhz * 1e3)  # -> ms
    planned = [r for r in rows if r["plans"] > 0]
    return {
        "cells": cells,
        "update_ms": [(r["geometry"] + r["update"]) * k for r in rows],
        "call_ms": [r["plan_call_max"] * k for r in planned],
        "call_mean_parts": [((r["beliefs"] + r["planner_IG"] + r["planner_T"]) * k, r["plans"]) for r in planned],
        "frame_ms": [r["total"] * k for r in rows],
        "ig": [(r["IG_cands_all"], r["plans"]) for r in planned],
    }


def pct(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def linfit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    a = my - b * mx
    ss = sum((c - my) ** 2 for c in y)
    r2 = 1.0 - sum((c - a - b * xx) ** 2 for xx, c in zip(x, y)) / ss if ss > 0 else 1.0
    return a, b, r2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="directories of gap8/main.c outputs (seed*.txt)")
    ap.add_argument("--host", action="store_true", help="host builds: per-frame values are ns, not cycles")
    ap.add_argument("--seeds-upto", type=int, default=0,
                    help="keep only outputs whose seeds are all <= N (seed012.txt, seed009_012.txt), "
                         "so that every size covers the same flights")
    args = ap.parse_args()

    by_cells: dict[int, list[dict]] = {}
    for d in args.dirs:
        files = sorted(glob.glob(os.path.join(d, "seed*.txt"))) if os.path.isdir(d) else [d]
        for p in files:
            if args.seeds_upto:
                seeds = [int(x) for x in re.findall(r"\d+", os.path.basename(p))]
                if not seeds or max(seeds) > args.seeds_upto:
                    continue
            fl = parse(p, args.host)
            if fl:
                by_cells.setdefault(fl["cells"], []).append(fl)
    if len(by_cells) < 2:
        raise SystemExit("need outputs for at least two map sizes")

    sizes = sorted(by_cells)
    base = sizes[0]
    stats = {}
    unit = "ms (host, not GAP8)" if args.host else "ms"
    print(f"{'cells':>5} {'x':>4} {'flights':>7} {'frames':>6} {'calls':>6} | {'update':>7} | "
          f"{'call mean':>9} {'call p95':>8} {'call max':>8} | {'frame mean':>10} {'frame max':>9} | {'IG/call':>11}   [{unit}]")
    for c in sizes:
        fls = by_cells[c]
        upd = [z for fl in fls for z in fl["update_ms"]]
        calls = [z for fl in fls for z in fl["call_ms"]]
        n_calls = sum(n for fl in fls for _, n in fl["call_mean_parts"])
        call_mean = sum(ms for fl in fls for ms, _ in fl["call_mean_parts"]) / n_calls
        frames = [z for fl in fls for z in fl["frame_ms"]]
        ig = sum(i for fl in fls for i, _ in fl["ig"]) / n_calls
        s = {"update mean": statistics.fmean(upd), "call mean": call_mean, "call p95": pct(calls, 0.95),
             "call max": max(calls), "frame mean": statistics.fmean(frames), "frame max": max(frames),
             "IG per call": ig}
        stats[c] = s
        print(f"{c:5d} {c / base:4.1f} {len(fls):7d} {len(frames):6d} {n_calls:6d} | {s['update mean']:7.2f} | "
              f"{s['call mean']:9.2f} {s['call p95']:8.2f} {s['call max']:8.2f} | {s['frame mean']:10.2f} "
              f"{s['frame max']:9.2f} | {ig:5.0f}/{8 * c:<5d}")

    print(f"\nfit over {len(sizes)} sizes: y = a + b * cells (R^2), and y ~ cells^k")
    for key in ["update mean", "call mean", "call p95", "call max", "frame mean", "frame max", "IG per call"]:
        x = [float(c) for c in sizes]
        y = [stats[c][key] for c in sizes]
        a, b, r2 = linfit(x, y)
        _, kexp, r2log = linfit([math.log(v) for v in x], [math.log(max(v, 1e-9)) for v in y])
        per = "" if key == "IG per call" else " ms"
        print(f"  {key:12s} a = {a:8.2f}{per}, b = {b:7.4f}{per}/cell, R^2 = {r2:.3f}   k = {kexp:.2f} (R^2 {r2log:.3f})"
              f"   x{y[-1] / y[0]:.2f} for x{sizes[-1] / sizes[0]:.0f} cells")


if __name__ == "__main__":
    main()
