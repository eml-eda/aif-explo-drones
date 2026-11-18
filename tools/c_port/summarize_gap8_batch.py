#!/usr/bin/env python3
"""Summary of the gvsoc batch (embedded/zoi/gap8/run_batch.sh).

Per detector frame, ZOI runs the frame update (geometry + evidence) and,
when the navigator asks for it, one or more planner calls. A planner call is the
belief refresh (beliefs are computed only when the planner reads them) plus the
search (IG and flight-time stages). Reported, over all flights:

- planner calls: per flight, per frame, exact IG evaluations per call;
- planner call latency: mean over all calls, 95th percentile, worst call;
- frame latency (frame update + the frame's planner calls + navigator, i.e. from
  the detector output to the new command): mean, worst; with the detector time
  and against the detector period, both read from each output;
- the later plan of a period (frames with two or more plans): exact IG
  evaluations of the frame's last plan, which the IG cache reduces;
- the flights whose run failed (seedNNN.txt.part: unfinished, or a first plan
  of a frame different from Python's), left out of the numbers above.

run_batch.sh writes one directory per mode, batch/<mode>.

Outputs without the plan_call_max column have no
per-call latency for frames with two plans; for those the frame's planner total
is used as the call latency (an upper bound), and the summary says so.

    python3 tools/c_port/summarize_gap8_batch.py                       # batch/flight
    python3 tools/c_port/summarize_gap8_batch.py --mode full           # batch/full
    python3 tools/c_port/summarize_gap8_batch.py --files a.txt b.txt   # any GAP8 outputs
    python3 tools/c_port/summarize_gap8_batch.py --per-flight          # one line per flight
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


def parse(path: str) -> dict:
    rows, det_ms, period_ms, mhz, plan_hash = [], None, None, None, None
    n_cand = 13 * 11 * 8  # the Webots room, unless the output prints the map size
    with open(path, errors="replace") as f:
        for line in f:
            p = line.strip().split(";")
            if len(p) >= 15 and p[0].isdigit():
                rows.append(dict(zip(COLS, (int(x) for x in p))))
                continue
            m = re.match(r"plan decisions hash: (0x[0-9a-f]+)", line)
            if m:
                plan_hash = m.group(1)
            m = re.search(r"map (\d+) x (\d+) cells", line)
            if m:
                n_cand = int(m.group(1)) * int(m.group(2)) * 8
            m = re.search(r"\(@(\d+(?:\.\d+)?) MHz", line)
            if m:
                mhz = float(m.group(1))
            m = re.match(r"\s*MobileNetV2\s+([\d.]+)", line)
            if m:
                det_ms = float(m.group(1))
            m = re.search(r"detector period ([\d.]+) ms", line)
            if m:
                period_ms = float(m.group(1))
    if not rows or mhz is None:
        raise ValueError(f"{path}: no per-frame rows or no clock line")
    k = 1.0 / (mhz * 1e3)
    calls_ms, call_max_ms, ig_per_frame, upper_bound = [], [], [], False
    for r in rows:
        n = r["plans"]
        if n <= 0:
            continue
        frame_plan_ms = (r["beliefs"] + r["planner_IG"] + r["planner_T"]) * k
        calls_ms.append((frame_plan_ms, n))
        if "plan_call_max" in r:
            call_max_ms.append(r["plan_call_max"] * k)
            ig_per_frame.append((r["IG_cands_all"], n))
        else:
            call_max_ms.append(frame_plan_ms)  # exact for one plan, upper bound for two
            upper_bound |= n > 1
            ig_per_frame.append((r["IG_cands"] * n, n))  # last plan's count for every plan
    return {
        "name": os.path.splitext(os.path.basename(path))[0],
        "frame_ms": [r["total"] * k for r in rows],
        "update_ms": [(r["geometry"] + r["update"]) * k for r in rows],
        "calls": calls_ms,          # (planner ms of the frame, calls in the frame)
        "call_max": call_max_ms,    # costliest call of each frame with a plan
        "ig": ig_per_frame,         # (exact IGs of the frame, calls in the frame)
        "upper_bound": upper_bound,
        # Exact IGs of the last plan of each frame with two or more plans (the later plan of the period).
        "ig_later": [r["IG_cands"] for r in rows if r["plans"] >= 2],
        "hash": plan_hash,
        "det_ms": det_ms,
        "period_ms": period_ms,
        "n_cand": n_cand,
    }


def ci95(values: list[float]) -> tuple[float, float, float]:
    m = statistics.fmean(values)
    if len(values) < 2:
        return m, m, m
    h = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return m, m - h, m + h


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[int(q * (len(s) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="embedded/zoi/gap8/batch", help="e.g. embedded/zoi/gap8/batch/tiles_2x2 for a larger room")
    ap.add_argument("--mode", default="flight", choices=["flight", "full"])
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--per-flight", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.batch, args.mode)
    files = args.files or sorted(glob.glob(os.path.join(out_dir, "seed*.txt")))
    if not files:
        raise SystemExit(f"no outputs in {out_dir}")
    flights = [parse(p) for p in files]

    n_frames = sum(len(fl["frame_ms"]) for fl in flights)
    n_calls = sum(n for fl in flights for _, n in fl["calls"])
    calls_per_flight = [sum(n for _, n in fl["calls"]) for fl in flights]
    planner_ms_total = sum(ms for fl in flights for ms, _ in fl["calls"])
    call_means = [sum(ms for ms, _ in fl["calls"]) / max(1, sum(n for _, n in fl["calls"])) for fl in flights]
    call_max = [c for fl in flights for c in fl["call_max"]]
    worst_call, worst_call_fl = max((max(fl["call_max"]), fl["name"]) for fl in flights if fl["call_max"])
    ig_calls = sum(ig for fl in flights for ig, _ in fl["ig"]) / max(1, n_calls)
    frame_all = [z for fl in flights for z in fl["frame_ms"]]
    frame_means = [statistics.fmean(fl["frame_ms"]) for fl in flights]
    worst_frame, worst_frame_fl, worst_frame_i = max(
        (z, fl["name"], i) for fl in flights for i, z in enumerate(fl["frame_ms"]))
    det, period = flights[0]["det_ms"], flights[0]["period_ms"]
    over = [(fl["name"], i, z + fl["det_ms"]) for fl in flights if fl["det_ms"] and fl["period_ms"]
            for i, z in enumerate(fl["frame_ms"]) if z + fl["det_ms"] > fl["period_ms"]]

    if args.per_flight:
        print(f"{'flight':34} {'frames':>6} {'calls':>5} {'call ms':>8} {'worst call':>10} {'frame ms':>8} {'worst frame':>11}")
        for fl, cm, nc in zip(flights, call_means, calls_per_flight):
            print(f"{fl['name']:34} {len(fl['frame_ms']):6d} {nc:5d} {cm:8.1f} {max(fl['call_max'], default=0):10.1f} "
                  f"{statistics.fmean(fl['frame_ms']):8.1f} {max(fl['frame_ms']):11.1f}")
        print()

    src = "files" if args.files else out_dir
    print(f"{len(flights)} flights, {n_frames} frames, {n_calls} planner calls ({src})")
    m, lo, hi = ci95(calls_per_flight)
    print("\nPlanner calls")
    print(f"  per flight: {m:.1f} [{lo:.1f}, {hi:.1f}] (95%), from {min(calls_per_flight)} to {max(calls_per_flight)}")
    print(f"  per frame: {n_calls / n_frames:.2f} (a call on {100 * sum(len(fl['calls']) for fl in flights) / n_frames:.0f}% of the frames)")
    print(f"  exact IG evaluations per call: {ig_calls:.0f} of {flights[0]['n_cand']} candidates")
    later = [n for fl in flights for n in fl["ig_later"]]
    if later:
        print(f"  later plan of a period ({len(later)} frames with two or more plans): exact IG evaluations "
              f"mean {statistics.fmean(later):.1f}, max {max(later)}")
    m, lo, hi = ci95(call_means)
    print("\nPlanner call latency (belief refresh + search)")
    print(f"  mean over all calls: {planner_ms_total / n_calls:.1f} ms; mean of the flight means {m:.1f} ms [{lo:.1f}, {hi:.1f}]")
    print(f"  95th percentile (costliest call of each frame): {pct(call_max, 0.95):.1f} ms")
    print(f"  worst call: {worst_call:.1f} ms ({worst_call_fl})"
          + ("  [upper bound: some outputs have no per-call column]" if any(fl["upper_bound"] for fl in flights) else ""))
    m, lo, hi = ci95(frame_means)
    print("\nFrame latency (frame update + planner calls + navigator, detector output -> command)")
    print(f"  mean over all frames: {statistics.fmean(frame_all):.1f} ms (frame update alone "
          f"{statistics.fmean([u for fl in flights for u in fl['update_ms']]):.1f} ms); "
          f"mean of the flight means {m:.1f} ms [{lo:.1f}, {hi:.1f}]")
    print(f"  95th percentile: {pct(frame_all, 0.95):.1f} ms; worst frame: {worst_frame:.1f} ms ({worst_frame_fl}, frame {worst_frame_i})")
    if det and period:
        print(f"  with the detector ({det:.2f} ms): mean {statistics.fmean(frame_all) + det:.1f} ms, "
              f"worst {worst_frame + det:.1f} ms, detector period {period:.0f} ms")
        print(f"  frames over the period: {len(over)} of {n_frames}"
              + ("" if not over else " (" + ", ".join(f"{n} f{i} {t:.0f} ms" for n, i, t in over[:10])
                 + (" ..." if len(over) > 10 else "") + ")"))

    if not args.files:
        failed = sorted(glob.glob(os.path.join(out_dir, "seed*.txt.part")))
        print(f"\nFailed runs (left out above): {len(failed)}")
        for p in failed:
            name = os.path.basename(p)[:-len(".txt.part")]
            with open(p, errors="replace") as f:
                line = next((ln.strip() for ln in f if ln.startswith("plans equal to Python")),
                            "unfinished (no final line)")
            print(f"  {name}: {line}")

if __name__ == "__main__":
    main()
