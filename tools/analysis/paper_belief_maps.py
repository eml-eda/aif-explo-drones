#!/usr/bin/env python3
"""Paper figure: P(target) room maps of one ZOI flight, plus belief entropy over time.

The flight is replayed from its perception log with tools/c_port/export_trace.py
(same beliefs and rho/theta as the logged flight). Per snapshot the map shows
P(target) of every grid cell (cells of the policy grid, drawn as tiles), with
ground-truth targets on top and, optionally, the desks and the drone path.

Writes every variant asked for in one go:
    ptarget_<symbols>_<tables>_<path>.{pdf,png}
        symbols: generic (one symbol) | split (bottle / can)
        tables:  tables | notables
        path:    path | nopath
    entropy.{pdf,png}

Usage (from the repo root):
    python tools/analysis/paper_belief_maps.py <run_dir> -o figures/belief_maps
    python tools/analysis/paper_belief_maps.py <run_dir> -o out --times prior 30 end
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.legend_handler import HandlerBase  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CTRL = os.path.join(REPO, "webots", "controllers", "crazyflie_AIF")
sys.path.insert(0, CTRL)
sys.path.insert(0, os.path.dirname(CTRL))

# q_joint order: table_{empty, bottle, can}, floor_{empty, bottle, can}
TARGET_IDX = [1, 2, 4, 5]

# ---- style (colours of the paper's method figure) ----
def _pick_font(candidates=("Nimbus Sans", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")):
    """First installed Helvetica-like font (Nimbus Sans on Linux, Helvetica on macOS, Arial on Windows)."""
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((c for c in candidates if c in installed), "DejaVu Sans")


FONT = _pick_font()
YELLOW = LinearSegmentedColormap.from_list(
    "ptarget", ["#fcfaf2", "#f8f4e1", "#e8dea9", "#cbb86e", "#ac963f", "#76672a"])
P_MIN = 1e-3
TABLE_EDGE = "#6c757d"
TARGET = "#4662bc"
PATH = "#343a40"
INK = "#212529"
MUTED = "#868e96"
ROOM_EDGE = "#ced4da"


def load_log(run_dir):
    with open(os.path.join(run_dir, "perception_log.jsonl")) as f:
        header = json.loads(f.readline())
    return header


def get_trace(run_dir, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    parts = os.path.normpath(run_dir).split(os.sep)
    path = os.path.join(cache_dir, f"{parts[-3]}__{parts[-1]}.txt")
    if not os.path.isfile(path):
        subprocess.run([sys.executable, "-B", os.path.join(REPO, "tools", "c_port", "export_trace.py"),
                        run_dir, "-o", path, "--cand-every", "0"], check=True, stdout=subprocess.DEVNULL)
    return path


def parse_trace(path, rows, cols):
    frames, cur = [], None
    with open(path) as f:
        for line in f:
            tag, *v = line.split()
            if tag == "FRAME":
                cur = {"t": float.fromhex(v[1]), "x": float.fromhex(v[2]), "y": float.fromhex(v[3]),
                       "yaw": float.fromhex(v[4]), "q": np.zeros((rows, cols, 6)), "claims": []}
                frames.append(cur)
            elif tag == "Q":
                cur["q"][int(v[1]), int(v[0])] = [float.fromhex(s) for s in v[2:8]]
            elif tag == "CLAIM":
                cur["claims"].append((int(v[0]), int(v[1]), int(v[2])))
    return frames


def prior_q(run_dir, header):
    """Belief before the first frame: the policy right after initialisation."""
    from configuration.zoi_params import ZOIParams
    from policies.zoi_vectorized import ZOIVectorizedPolicy
    with open(os.path.join(run_dir, "policy_parameters.json")) as f:
        model = json.load(f)["model"]
    fields = set(ZOIParams.__dataclass_fields__)
    params = ZOIParams(**{k: v for k, v in model.items() if k in fields})
    pol = ZOIVectorizedPolicy(params=params, detector_period_s=float(header["detection_every_s"]))
    g = header["grid"]
    cs = float(g["cell_size_m"])
    gs = {"min_x": g["min_x"], "min_y": g["min_y"], "cell_size": cs, "rows": g["rows"], "cols": g["cols"],
          "max_x": g["min_x"] + g["cols"] * cs, "max_y": g["min_y"] + g["rows"] * cs}
    pol._ensure_initialized({"grid_spec": gs, "fov_spec": header["fov_spec"]["floor"],
                             "fov_spec_by_surface": header["fov_spec"]})
    return pol.q_joint.copy()


def entropy_bits(q):
    q = np.clip(q, 1e-300, 1.0)
    return -(q * np.log2(q)).sum(axis=-1)


class Flight:
    def __init__(self, run_dir, cache_dir):
        self.header = load_log(run_dir)
        g = self.header["grid"]
        self.rows, self.cols, self.cs = int(g["rows"]), int(g["cols"]), float(g["cell_size_m"])
        self.x0, self.y0 = float(g["min_x"]), float(g["min_y"])
        self.frames = parse_trace(get_trace(run_dir, cache_dir), self.rows, self.cols)
        self.q0 = prior_q(run_dir, self.header)
        self.t0 = self.frames[0]["t"]
        objs = self.header["spawned_objects"]["objects"]
        self.desks = [o["tabletop_rect"]["corners_xy"] for o in objs if o["class_name"] == "desk"]
        self.targets = [o for o in objs if o["class_name"] in ("target_bottle", "target_can")]
        claim_t = {}
        for fr in self.frames:
            for c in fr["claims"]:
                claim_t.setdefault(c, fr["t"] - self.t0)
        self.found_t = []
        for o in self.targets:
            key = (int(math.floor((o["x"] - self.x0) / self.cs)), int(math.floor((o["y"] - self.y0) / self.cs)),
                   0 if o["class_name"] == "target_bottle" else 1)
            self.found_t.append(claim_t.get(key, math.inf))
        self.t_end = self.frames[-1]["t"] - self.t0

    def snapshot(self, spec):
        """(label, time since start, frame index or None for the prior)."""
        if spec == "prior":
            return None
        t = {"mid": 0.5 * self.t_end, "end": self.t_end}.get(spec)
        t = float(spec) if t is None else t
        return min(range(len(self.frames)), key=lambda j: abs(self.frames[j]["t"] - self.t0 - t))


# ---- drawing ----

def draw_tiles(ax, fl, img, gap=0.06, radius=0.07):
    cs = fl.cs
    for r in range(fl.rows):
        for c in range(fl.cols):
            v = max(float(img[r, c]), P_MIN)
            x, y = fl.x0 + c * cs + gap / 2, fl.y0 + r * cs + gap / 2
            ax.add_patch(FancyBboxPatch((x, y), cs - gap, cs - gap,
                                        boxstyle=f"round,pad=0,rounding_size={radius}",
                                        fc=YELLOW(LogNorm(P_MIN, 1)(v)), ec="none", zorder=1))
    w, h = fl.cols * cs, fl.rows * cs
    ax.add_patch(FancyBboxPatch((fl.x0 - 0.06, fl.y0 - 0.06), w + 0.12, h + 0.12,
                                boxstyle="round,pad=0,rounding_size=0.14", fc="none", ec=ROOM_EDGE,
                                lw=0.8, zorder=0))


def drone_marker(ax, x, y, yaw, size=0.30):
    pts = np.array([[1.0, 0.0], [-0.7, 0.6], [-0.35, 0.0], [-0.7, -0.6]]) * size
    c, s = math.cos(yaw), math.sin(yaw)
    pts = pts @ np.array([[c, s], [-s, c]]) + [x, y]
    ax.add_patch(Polygon(pts, closed=True, fc=INK, ec="white", lw=0.6, zorder=6))


def draw_target(ax, x, y, kind, found, split):
    fc = TARGET if found else "white"
    if not split:
        ax.plot(x, y, "o", ms=5.2, mfc=fc, mec=TARGET if not found else "white", mew=1.0, zorder=7)
        return
    # rounded letter badge, as the B / C boxes of the method figure
    letter = "B" if kind == "target_bottle" else "C"
    ax.text(x, y, letter, ha="center", va="center", fontsize=5.6, fontweight="bold",
            color="white" if found else TARGET, zorder=8,
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=fc, ec=TARGET if not found else "white",
                      lw=0.8))


def map_panel(ax, fl, k, *, tables, path, split):
    if k is None:
        q, t = fl.q0, -1.0
    else:
        fr = fl.frames[k]
        q, t = fr["q"], fr["t"] - fl.t0
    draw_tiles(ax, fl, q[:, :, TARGET_IDX].sum(-1))
    if tables:
        for corners in fl.desks:
            ax.add_patch(Polygon(corners, closed=True, fc="none", ec="white", lw=2.8, zorder=3))
            ax.add_patch(Polygon(corners, closed=True, fc="none", ec=TABLE_EDGE, lw=1.1, zorder=3))
    if path and k is not None:
        p = np.array([(f["x"], f["y"]) for f in fl.frames[: k + 1]])
        ax.plot(p[:, 0], p[:, 1], "-", color="white", lw=2.2, alpha=0.8, solid_capstyle="round", zorder=4)
        ax.plot(p[:, 0], p[:, 1], "-", color=PATH, lw=0.9, solid_capstyle="round", zorder=5)
        ax.plot(p[0, 0], p[0, 1], "o", ms=3.2, mfc="white", mec=PATH, mew=0.9, zorder=5)
        drone_marker(ax, fl.frames[k]["x"], fl.frames[k]["y"], fl.frames[k]["yaw"])
    for o, ft in zip(fl.targets, fl.found_t):
        draw_target(ax, o["x"], o["y"], o["class_name"], ft <= t, split)
    ax.set_xlim(fl.x0 - 0.1, fl.x0 + fl.cols * fl.cs + 0.1)
    ax.set_ylim(fl.y0 - 0.1, fl.y0 + fl.rows * fl.cs + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")


class Badge:
    """Legend proxy for the B / C target badges."""

    def __init__(self, letter, found, label):
        self.letter, self.found, self.label = letter, found, label

    def get_label(self):
        return self.label


class HandlerBadge(HandlerBase):
    def create_artists(self, legend, orig, xdescent, ydescent, width, height, fontsize, trans):
        t = Text(width / 2 - xdescent, height / 2 - ydescent, orig.letter, ha="center", va="center",
                 fontsize=5.6, fontweight="bold", color="white" if orig.found else TARGET, transform=trans,
                 bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=TARGET if orig.found else "white",
                           ec="white" if orig.found else TARGET, lw=0.8))
        return [t]


def legend_handles(*, tables, path, split):
    hs = []
    if split:
        hs.append(Badge("B", True, "bottle"))
        hs.append(Badge("C", True, "can"))
        hs.append(Badge("B", False, "not yet found"))
    else:
        hs.append(Line2D([], [], ls="none", marker="o", ms=5.2, mfc=TARGET, mec="white", label="target, found"))
        hs.append(Line2D([], [], ls="none", marker="o", ms=5.2, mfc="white", mec=TARGET, mew=1.0,
                         label="target, not yet found"))
    if tables:
        hs.append(Line2D([], [], ls="none", marker="s", ms=7, mfc="none", mec=TABLE_EDGE, mew=1.1,
                         label="table"))
    if path:
        hs.append(Line2D([], [], color=PATH, lw=0.9, label="drone path"))
        hs.append(Line2D([], [], ls="none", marker=(3, 0, -90), ms=6, color=INK, label="drone"))
    return hs


def map_figure(fl, snaps, titles, *, tables, path, split):
    n = len(snaps)
    panel_w = 2.05
    fig = plt.figure(figsize=(n * panel_w + 0.55, panel_w * fl.rows / fl.cols + 0.55))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.05],
                          wspace=0.08, left=0.01, right=0.92, top=0.93, bottom=0.10)
    for i, (k, title) in enumerate(zip(snaps, titles)):
        ax = fig.add_subplot(gs[0, i])
        map_panel(ax, fl, k, tables=tables, path=path, split=split)
        ax.set_title(title, fontsize=8.5, fontweight="bold", color=INK, pad=4)
    cax = fig.add_subplot(gs[0, n])
    sm = plt.cm.ScalarMappable(norm=LogNorm(P_MIN, 1), cmap=YELLOW)
    cb = fig.colorbar(sm, cax=cax, ticks=[1e-3, 1e-2, 1e-1, 1])
    cb.ax.set_yticklabels([r"$\leq$0.001", "0.01", "0.1", "1"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, color=MUTED, labelcolor=INK)
    cb.set_label(r"$P(\mathrm{target})$", fontsize=8, color=INK, labelpad=3)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + pos.height * 0.12, pos.width, pos.height * 0.76])
    cb.ax.minorticks_off()
    fig.legend(handles=legend_handles(tables=tables, path=path, split=split), loc="lower center",
               ncol=7, frameon=False, fontsize=7, handletextpad=0.35, columnspacing=1.4,
               bbox_to_anchor=(0.47, 0.0), handler_map={Badge: HandlerBadge()})
    return fig


def entropy_figure(fl, snaps, titles):
    ts = np.array([0.0] + [f["t"] - fl.t0 for f in fl.frames])
    H = np.array([entropy_bits(fl.q0).sum()] + [entropy_bits(f["q"]).sum() for f in fl.frames])
    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    ax.fill_between(ts, H, color="#f8f4e1", lw=0, zorder=1)
    ax.plot(ts, H, color="#ac963f", lw=1.3, zorder=3)
    for k, title in zip(snaps, titles):
        tk = 0.0 if k is None else fl.frames[k]["t"] - fl.t0
        ax.axvline(tk, color=ROOM_EDGE, lw=0.8, ls=(0, (3, 2)), zorder=0)
    for ft in sorted(fl.found_t):
        if math.isfinite(ft):
            ax.plot(ft, np.interp(ft, ts, H), "o", ms=4.2, mfc=TARGET, mec="white", mew=0.8, zorder=4)
    ax.plot([], [], "o", ms=4.2, mfc=TARGET, mec="white", mew=0.8, label="target found")
    ax.legend(frameon=False, fontsize=7, loc="upper right", handletextpad=0.3)
    ax.set_xlim(0, ts[-1])
    ax.set_ylim(0, H.max() * 1.05)
    ax.set_xlabel("Time [s]", fontsize=8, color=INK)
    ax.set_ylabel("Belief entropy [bits]", fontsize=8, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(labelsize=7, color=MUTED, labelcolor=INK, width=0.6, length=2.5)
    fig.tight_layout(pad=0.3)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--times", nargs="+", default=["prior", "mid", "end"],
                    help="snapshots: 'prior', 'mid', 'end' or seconds from the start")
    ap.add_argument("--cache", default=None, help="trace cache directory (default <outdir>/.traces)")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": FONT, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
                         "mathtext.bf": f"{FONT}:bold", "text.color": INK})
    os.makedirs(args.outdir, exist_ok=True)
    fl = Flight(args.run_dir, args.cache or os.path.join(args.outdir, ".traces"))
    snaps = [fl.snapshot(s) for s in args.times]
    titles = []
    for s, k in zip(args.times, snaps):
        t = 0.0 if k is None else fl.frames[k]["t"] - fl.t0
        name = {"prior": "Start", "end": "End", "mid": "Mid-flight"}.get(s, "")
        titles.append(f"{name} (t = {t:.0f} s)" if name else f"t = {t:.0f} s")

    for split, tables, path in itertools.product((False, True), (True, False), (True, False)):
        fig = map_figure(fl, snaps, titles, tables=tables, path=path, split=split)
        name = "ptarget_{}_{}_{}".format("split" if split else "generic", "tables" if tables else "notables",
                                         "path" if path else "nopath")
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(args.outdir, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    fig = entropy_figure(fl, snaps, titles)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"entropy.{ext}"), dpi=300)
    plt.close(fig)
    found = sum(math.isfinite(t) for t in fl.found_t)
    print(f"{args.outdir}: {len(fl.frames)} frames, {fl.t_end:.1f} s, {found}/{len(fl.targets)} found, "
          f"snapshots at {[t for t in titles]}")


if __name__ == "__main__":
    main()
