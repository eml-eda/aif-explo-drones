#!/usr/bin/env python3
"""GAP8 trace header for a larger room, tiled from recorded flights.

The Webots room is 13 x 11 cells of 0.5 m (6.5 m along y, 5.5 m along x). A
tiling TR x TC lays TR x TC copies of it side by side: a map of 13 TR x 11 TC
cells, centred on the origin like the original, TR TC times the area. The
recorded flights of TR TC seeds are replayed one after the other, one per tile
(rows of tiles in order, snaking so consecutive tiles are adjacent), without
resetting the state. The first frames therefore plan with the whole map
unexplored, the last flight with all tiles but one explored. Boxes are in image
coordinates and need no change; the cell boundaries of every tile fall on the
original ones.

Poses are replayed open loop, as in the gvsoc batch, so the timing covers the
belief update and the planner over the larger map. There is no Python decision
for these rooms: TRACE_PLAN is -1 everywhere and main.c skips the comparison
("plans equal to Python: 0 / 0").

    # one header from given flights (TR * TC of them)
    python3 tools/c_port/make_tiled_room_header.py --tiles 2x2 a.h b.h c.h d.h -o out.h
    # every group of TR * TC consecutive seeds of a directory: seed001_004.h, seed005_008.h, ...
    python3 tools/c_port/make_tiled_room_header.py --tiles 1x2 --group embedded/zoi/gap8/batch/headers \\
        -o embedded/zoi/gap8/batch/tiles_1x2/headers

Build on gvsoc with ROWS=13*TR COLS=11*TC (gap8/Makefile, run_batch.sh); the
tool prints them. 1x2: 13 x 22 (2x the cells), 2x2: 26 x 22 (4x).
"""

from __future__ import annotations

import argparse
import glob
import os
import re

ROOM_ROWS, ROOM_COLS = 13, 11  # the Webots room, cells
CELL = 0.5


def block(text: str, name: str) -> str:
    m = re.search(r"static const \w+ " + name + r"\[[^\]]*\] = \{(.*?)\};", text, re.S)
    if not m:
        raise ValueError(f"{name} not found")
    return m.group(1)


def define(text: str, name: str) -> str:
    m = re.search(r"#define " + name + r" (\S+)", text)
    if not m:
        raise ValueError(f"#define {name} not found")
    return m.group(1)


def floats(s: str) -> list[float]:
    return [float(x.rstrip("f")) for x in re.findall(r"-?[0-9.]+(?:e-?[0-9]+)?f?", s)]


def ints(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]


def tuples(s: str) -> list[str]:
    return re.findall(r"\{([^{}]*)\}", s)


def parse(path: str) -> dict:
    with open(path) as f:
        t = f.read()
    n = int(define(t, "TRACE_N_FRAMES"))
    if "TRACE_ROWS" in t:
        raise ValueError(f"{path} is already a synthetic room")
    poses = [floats(p) for p in tuples(block(t, "TRACE_POSE"))]
    boxes = tuples(block(t, "TRACE_BOXES"))
    d = {
        "path": path,
        "env": define(t, "TRACE_ENV"),
        "det": define(t, "TRACE_DET"),
        "grid_n": define(t, "TRACE_GRID_N"),
        "table": define(t, "TRACE_TABLE_CONTEXT"),
        "params": block(t, "TRACE_PARAMS").strip(),
        "pose": poses,
        "time": floats(block(t, "TRACE_TIME")),
        "box_start": ints(block(t, "TRACE_BOX_START")),
        "box_count": ints(block(t, "TRACE_BOX_COUNT")),
        "boxes": [b.strip() for b in boxes],
        "replans": ints(block(t, "TRACE_REPLANS")),
    }
    for k in ("pose", "time", "box_start", "box_count", "replans"):
        if len(d[k]) != n:
            raise ValueError(f"{path}: {k} has {len(d[k])} entries, {n} frames")
    if any(len(p) != 3 for p in d["pose"]):
        raise ValueError(f"{path}: a pose without 3 values")
    if sum(d["box_count"]) != len(d["boxes"]):
        raise ValueError(f"{path}: box counts do not match the boxes")
    return d


def f32(x: float) -> str:
    return repr(float(x)) + "f"


def tile_order(tr: int, tc: int) -> list[tuple[int, int]]:
    """Tiles (row, col), rows in order, snaking so consecutive tiles are adjacent."""
    return [(i, j if i % 2 == 0 else tc - 1 - j) for i in range(tr) for j in range(tc)]


def combine(flights: list[dict], tr: int, tc: int) -> str:
    a = flights[0]
    for b in flights[1:]:
        for k in ("env", "det", "grid_n", "table", "params"):
            if a[k] != b[k]:
                raise ValueError(f"{k} differs between {a['path']} and {b['path']}")
    rows, cols = ROOM_ROWS * tr, ROOM_COLS * tc
    poses, times, box_start, box_count, boxes, replans, notes = [], [], [], [], [], [], []
    t_next = 0.0
    for fl, (i, j) in zip(flights, tile_order(tr, tc)):
        # Tile centre: the original room is centred on the origin.
        dx = (j - 0.5 * (tc - 1)) * ROOM_COLS * CELL
        dy = (i - 0.5 * (tr - 1)) * ROOM_ROWS * CELL
        poses += [(x + dx, y + dy, yaw) for x, y, yaw in fl["pose"]]
        times += [t - fl["time"][0] + t_next for t in fl["time"]]
        step = fl["time"][-1] - fl["time"][-2] if len(fl["time"]) > 1 else 0.625
        t_next = times[-1] + step
        box_start += [s + len(boxes) for s in fl["box_start"]]
        box_count += fl["box_count"]
        boxes += fl["boxes"]
        replans += fl["replans"]
        notes.append(f" *   tile ({i}, {j}), shift ({dx:+.2f}, {dy:+.2f}) m: {os.path.basename(fl['path'])}, "
                     f"{len(fl['pose'])} frames")
    n = len(poses)

    def ilist(v):
        return ", ".join(str(x) for x in v)

    out = [
        "/* Generated by tools/c_port/make_tiled_room_header.py: the Webots room tiled",
        f" * {tr} x {tc} ({rows} x {cols} cells of {CELL} m, {tr * tc}x the area), one recorded",
        " * flight per tile, one after the other, state kept. Do not edit.",
        *notes,
        f" * No Python decision for this room: TRACE_PLAN is -1. Build with ROWS={rows} COLS={cols}. */",
        "#ifndef ZOI_TRACE_DATA_H",
        "#define ZOI_TRACE_DATA_H",
        "",
        '#include "zoi.h"',
        "",
        f"#define TRACE_ENV {a['env']}",
        f"#define TRACE_DET {a['det']}",
        f"#define TRACE_GRID_N {a['grid_n']}",
        f"#define TRACE_TABLE_CONTEXT {a['table']}",
        f"#define TRACE_ROWS {rows}",
        f"#define TRACE_COLS {cols}",
        f"#define TRACE_N_FRAMES {n}",
        "/* recall, class accuracy, false alarm, desk recall, desk false alarm, p_on_table */",
        f"static const float TRACE_PARAMS[6] = {{{a['params']}}};",
        "",
        "static const zoi_pose_t TRACE_POSE[TRACE_N_FRAMES] = {",
        *[f"    {{{f32(x)}, {f32(y)}, {f32(yaw)}}}," for x, y, yaw in poses],
        "};",
        "",
        "static const float TRACE_TIME[TRACE_N_FRAMES] = {",
        "    " + ", ".join(f32(t) for t in times),
        "};",
        "",
        f"static const int TRACE_BOX_START[TRACE_N_FRAMES] = {{{ilist(box_start)}}};",
        f"static const int TRACE_BOX_COUNT[TRACE_N_FRAMES] = {{{ilist(box_count)}}};",
        f"static const zoi_box_t TRACE_BOXES[{max(len(boxes), 1)}] = {{",
        *([f"    {{{bx}}}," for bx in boxes] or ["    {0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},"]),
        "};",
        "",
        "/* No Python decision in this room. */",
        f"static const int TRACE_PLAN[TRACE_N_FRAMES] = {{{ilist([-1] * n)}}};",
        "",
        "/* Plans the recorded flights made after each frame (-1: not recorded in the log). */",
        f"static const int TRACE_REPLANS[TRACE_N_FRAMES] = {{{ilist(replans)}}};",
        "",
        "#endif /* ZOI_TRACE_DATA_H */",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", required=True, help="TRxTC, e.g. 1x2 (2x the cells) or 2x2 (4x)")
    ap.add_argument("headers", nargs="*", help="TR * TC flight headers, in tile order")
    ap.add_argument("--group", metavar="DIR", help="every TR * TC consecutive seedNNN.h of DIR")
    ap.add_argument("-o", "--out", required=True, help="output header, or directory with --group")
    args = ap.parse_args()
    m = re.fullmatch(r"(\d+)x(\d+)", args.tiles)
    if not m or int(m.group(1)) < 1 or int(m.group(2)) < 1:
        raise SystemExit("--tiles must be TRxTC, e.g. 2x2")
    tr, tc = int(m.group(1)), int(m.group(2))
    k = tr * tc
    if args.group:
        files = sorted(glob.glob(os.path.join(args.group, "seed[0-9][0-9][0-9].h")))
        if len(files) < k:
            raise SystemExit(f"fewer than {k} headers in {args.group}")
        os.makedirs(args.out, exist_ok=True)
        n = len(files) // k
        for g in range(n):
            grp = files[g * k:(g + 1) * k]
            first, last = os.path.basename(grp[0])[:-2], os.path.basename(grp[-1])[4:-2]
            path = os.path.join(args.out, f"{first}_{last}.h" if k > 1 else f"{first}.h")
            with open(path, "w") as f:
                f.write(combine([parse(p) for p in grp], tr, tc))
        left = len(files) - n * k
        print(f"{n} headers in {args.out}" + (f" ({left} seeds left out)" if left else ""))
    else:
        if len(args.headers) != k:
            raise SystemExit(f"--tiles {args.tiles} needs {k} headers")
        with open(args.out, "w") as f:
            f.write(combine([parse(p) for p in args.headers], tr, tc))
        print(args.out)
    print(f"build with ROWS={ROOM_ROWS * tr} COLS={ROOM_COLS * tc}")


if __name__ == "__main__":
    main()
