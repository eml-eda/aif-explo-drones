from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:  # graceful fallback; scipy is preferred for the smooth monotone curve
    PchipInterpolator = None


# =============================================================================
# USER SETTINGS
# =============================================================================
# Each entry may be either a run directory or its episode_log_*.csv file.
# Keep between 2 and 6 runs.
RUN_PATHS = [
    # Relative paths are resolved from the current directory, for example:
    # "webots/controllers/crazyflie_AIF/logs/log_run_spiral_<timestamp>",
    # "webots/controllers/crazyflie_AIF/logs/log_run_zoi_<timestamp>",
]

# Relative paths are resolved from the repository root.
OUTPUT_DIR = "Visuals_DATE"
# Give only the base name; .pdf and .png are created automatically.
# Four files are produced: normal + _full, in both PDF and PNG.
OUTPUT_FILENAME = "coverage_comparison_SEED_42_targets_uniform_TRIO"

# Leave empty to infer canonical policy names. Otherwise provide one label per path.
CUSTOM_LABELS = []

EPISODE_DURATION_S = 180.0
FULL_COVERAGE_PERCENT = 100.0
MAX_RUNS = 6

# Smooth cumulative curve: only actual coverage change points are interpolated.
# PCHIP is shape-preserving and monotone, so it does not invent oscillations or
# overshoot. Set False to display the raw sampled curve instead.
SMOOTH_CURVES = True
SMOOTH_SAMPLES_PER_SECOND = 12

# Only TRUE-POSITIVE, supervisor-equivalent detections are shown. The script reads
# claim_log_*.csv, validates each claim against scene_snapshot.json ground truth,
# checks the predicted class, and keeps only the FIRST correct claim for each real
# target object. With the standard scenario this therefore produces at most 6 marks.
SHOW_DETECTION_MARKERS = True
DETECTION_MARKER_SIZE = 165
DETECTION_INNER_SIZE = 34

# Detection color mode:
#   "uniform" -> every confirmed detection uses the same dedicated event color.
#   "policy"  -> each confirmed detection uses the color of its policy curve.
# This is intentionally a one-line toggle so you can render both variants side by side.
DETECTION_MARKER_COLOR_MODE = "uniform"
UNIFORM_DETECTION_COLOR = "#E83E8C"   # vivid raspberry, separate from curve palette
UNIFORM_DETECTION_EDGE = "#76143F"

# Figure tuned for reduction in a two-column paper.
FIGSIZE = (8.8, 5.7)
SAVE_DPI = 600
AXIS_LABEL_FONTSIZE = 19
TICK_FONTSIZE = 14.5
LEGEND_FONTSIZE = 13.0

POLICY_LABELS = {
    "wall_following": "Wall Following",
    "spiral": "Spiral",
    "pseudo_random": "Pseudo-Random",
    "rotate_and_measure": "Rotate and Measure",
    "zoi": "ZOI",
}

# Vivid solid curves. Detection marker color is controlled independently above.
LINE_STYLES = [
    {"color": "#0B6BFF", "linewidth": 5.8},
    {"color": "#FF7A00", "linewidth": 5.8},
    {"color": "#7A3FF2", "linewidth": 5.7},
    {"color": "#00A6D6", "linewidth": 5.7},
    {"color": "#D64FA3", "linewidth": 5.6},
    {"color": "#3346B8", "linewidth": 5.6},
]

TARGET_CLASS_NAMES = {"target_bottle", "target_can"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelweight": "bold",
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    if len(here.parents) > 4:
        candidate = here.parents[4]
        if (candidate / "webots").exists():
            return candidate
    return Path.cwd().resolve()


def _resolve_output_dir() -> Path:
    p = Path(OUTPUT_DIR).expanduser()
    return p if p.is_absolute() else (_repo_root() / p)


def _output_base() -> Path:
    name = Path(OUTPUT_FILENAME).name
    stem = Path(name).stem if Path(name).suffix else name
    return _resolve_output_dir() / stem


def _latest_file(run_dir: Path, pattern: str) -> Path | None:
    matches = sorted(run_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _resolve_run(path_like: str) -> tuple[Path, Path, Path, Path | None]:
    p = Path(path_like).expanduser().resolve()
    if p.is_dir():
        run_dir = p
        episode_log = _latest_file(run_dir, "episode_log_*.csv")
        if episode_log is None:
            raise FileNotFoundError(f"No episode_log_*.csv found in {run_dir}")
    elif p.is_file():
        run_dir = p.parent
        episode_log = p
    else:
        raise FileNotFoundError(p)

    snapshot = run_dir / "scene_snapshot.json"
    if not snapshot.exists():
        raise FileNotFoundError(f"Missing scene_snapshot.json in {run_dir}")

    claim_log = _latest_file(run_dir, "claim_log_*.csv")
    return run_dir, episode_log, snapshot, claim_log


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_episode_log(path: Path) -> dict[str, np.ndarray]:
    wanted = {
        "episode_time_s", "sim_time_s", "coverage_ratio",
        "visited_cells", "total_cells",
    }
    out = {k: [] for k in wanted}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"episode_time_s", "sim_time_s", "coverage_ratio"}
        if not required.issubset(fields):
            raise ValueError(f"{path} is missing columns: {sorted(required - fields)}")
        for row in reader:
            for key in wanted:
                if key in fields:
                    out[key].append(float(row[key]))
    if not out["episode_time_s"]:
        raise ValueError(f"No episode samples in {path}")

    result = {
        "episode_time_s": np.asarray(out["episode_time_s"], dtype=float),
        "sim_time_s": np.asarray(out["sim_time_s"], dtype=float),
        "coverage_percent": 100.0 * np.asarray(out["coverage_ratio"], dtype=float),
    }
    if out["visited_cells"]:
        result["visited_cells"] = np.asarray(out["visited_cells"], dtype=float)
    if out["total_cells"]:
        result["total_cells"] = np.asarray(out["total_cells"], dtype=float)
    return result


def _grid_cell_from_xy(x: float, y: float, grid: dict) -> tuple[int, int] | None:
    min_x = float(grid["min_x"])
    min_y = float(grid["min_y"])
    max_x = float(grid.get("max_x", min_x + float(grid["cols"]) * float(grid["cell_size_m"])))
    max_y = float(grid.get("max_y", min_y + float(grid["rows"]) * float(grid["cell_size_m"])))
    cs = float(grid["cell_size_m"])
    if x < min_x or x >= max_x or y < min_y or y >= max_y:
        return None
    return int((x - min_x) / cs), int((y - min_y) / cs)


def _true_target_keys(scene: dict) -> set[tuple[int, int, str]]:
    grid = scene.get("grid", {}) or {}
    spawned = scene.get("spawned_objects", {}) or {}
    objects = spawned.get("objects", []) or []
    keys: set[tuple[int, int, str]] = set()

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        class_name = str(obj.get("class_name", obj.get("label", "")))
        if class_name not in TARGET_CLASS_NAMES:
            continue

        cx = obj.get("cx", None)
        cy = obj.get("cy", None)
        if cx is None or cy is None:
            if "x" not in obj or "y" not in obj or not grid:
                continue
            cell = _grid_cell_from_xy(float(obj["x"]), float(obj["y"]), grid)
            if cell is None:
                continue
            cx, cy = cell

        keys.add((int(cx), int(cy), class_name))
    return keys


def _read_confirmed_detection_events(path: Path | None, scene: dict) -> list[tuple[float, str]]:
    """Return first correct-claim simulator time + class for each real target.

    Validation mirrors the supervisor semantics: claim cell AND predicted class
    must match an actual spawned target. Duplicate claims for the same true object
    are ignored.
    """
    if path is None or not path.exists():
        return []

    true_targets = _true_target_keys(scene)
    if not true_targets:
        return []

    detections: list[tuple[float, tuple[int, int, str]]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"time_s", "claim_cx", "claim_cy", "class_name"}
        if not required.issubset(fields):
            print(
                f"[WARN] {path.name} lacks {sorted(required - fields)}; "
                "confirmed-detection markers disabled for this run."
            )
            return []

        for row in reader:
            try:
                key = (
                    int(float(row["claim_cx"])),
                    int(float(row["claim_cy"])),
                    str(row["class_name"]),
                )
                t = float(row["time_s"])
            except Exception:
                continue
            if key in true_targets:
                detections.append((t, key))

    detections.sort(key=lambda item: item[0])
    seen: set[tuple[int, int, str]] = set()
    first_events: list[tuple[float, str]] = []
    for t, key in detections:
        if key in seen:
            continue
        seen.add(key)
        first_events.append((float(t), str(key[2])))

    return first_events


def _detection_episode_events(
    det_events: list[tuple[float, str]],
    log: dict[str, np.ndarray],
) -> list[tuple[float, str]]:
    if not det_events:
        return []
    sim_t = log["sim_time_s"]
    ep_t = log["episode_time_s"]
    if sim_t.size == 0:
        return []

    out: list[tuple[float, str]] = []
    for sim_time, class_name in det_events:
        if sim_time < sim_t[0] or sim_time > sim_t[-1]:
            continue
        episode_time = float(np.interp(float(sim_time), sim_t, ep_t))
        out.append((episode_time, str(class_name)))
    return out

def _round_nested(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_round_nested(v) for v in value]
    if isinstance(value, dict):
        return {
            k: _round_nested(v)
            for k, v in value.items()
            if k not in {"id", "recognition_color_rgb"}
        }
    return value


def _scene_signature(scene: dict) -> dict:
    spawned = scene.get("spawned_objects", {}) or {}
    objects = []
    for obj in spawned.get("objects", []) or []:
        if not isinstance(obj, dict):
            continue
        keep = {
            k: obj.get(k)
            for k in (
                "def", "class_name", "model", "proto", "x", "y", "z", "yaw",
                "cx", "cy", "spawn_mode", "desk_id", "dimensions_m", "tabletop_rect",
            )
            if k in obj
        }
        objects.append(_round_nested(keep))
    objects.sort(
        key=lambda o: (
            str(o.get("class_name", "")),
            str(o.get("def", "")),
            float(o.get("x", 0.0)),
            float(o.get("y", 0.0)),
        )
    )

    init = scene.get("experiment_init", {}) or {}
    return {
        "environment": scene.get("environment"),
        "detector": scene.get("detector"),
        "seed": int(scene.get("seed", -1)),
        "grid": _round_nested(scene.get("grid", {}) or {}),
        "drone_spawn": _round_nested(init.get("drone_spawn", {}) or {}),
        "objects": objects,
    }


def _validate_same_scene(scenes: list[dict]) -> None:
    reference = _scene_signature(scenes[0])
    for idx, scene in enumerate(scenes[1:], start=2):
        if _scene_signature(scene) != reference:
            raise ValueError(
                f"Run {idx} does not describe the same environment/seed/detector/spawned scene as run 1."
            )


def _label_for(scene: dict, idx: int) -> str:
    if CUSTOM_LABELS:
        if len(CUSTOM_LABELS) != len(RUN_PATHS):
            raise ValueError("CUSTOM_LABELS must be empty or have one entry per RUN_PATHS item.")
        return str(CUSTOM_LABELS[idx])
    policy = str(scene.get("policy", f"run {idx + 1}"))
    return POLICY_LABELS.get(policy, policy.replace("_", " ").title())


def _smooth_curve(times: np.ndarray, coverage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    coverage = np.maximum.accumulate(np.asarray(coverage, dtype=float))
    if not SMOOTH_CURVES or times.size < 3:
        return times, coverage

    # Interpolate through the actual cumulative-coverage change points instead of
    # drawing every repeated plateau sample. PCHIP preserves monotonicity and does
    # not overshoot the measured coverage range.
    change = np.r_[True, np.diff(coverage) > 1e-10]
    idx = np.flatnonzero(change).tolist()
    if idx[-1] != len(times) - 1:
        idx.append(len(times) - 1)

    x = times[idx]
    y = coverage[idx]

    keep = np.r_[True, np.diff(x) > 1e-9]
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return times, coverage

    dense_n = max(300, int(max(1.0, x[-1] - x[0]) * SMOOTH_SAMPLES_PER_SECOND))
    xd = np.linspace(float(x[0]), float(x[-1]), dense_n)

    if PchipInterpolator is not None and x.size >= 3:
        yd = PchipInterpolator(x, y, extrapolate=False)(xd)
    else:
        yd = np.interp(xd, x, y)

    yd = np.maximum.accumulate(np.asarray(yd, dtype=float))
    yd = np.clip(yd, float(y.min()), float(y.max()))
    return xd, yd


def _scaled_limits(loaded: list[dict]) -> tuple[float, float]:
    max_t = max(float(item["smooth_t"][-1]) for item in loaded)
    max_cov = max(float(item["smooth_cov"].max()) for item in loaded)
    x_hi = min(EPISODE_DURATION_S, max(10.0, math.ceil((max_t + 3.0) / 10.0) * 10.0))
    y_hi = min(FULL_COVERAGE_PERCENT, max(10.0, math.ceil((max_cov + 3.0) / 5.0) * 5.0))
    return x_hi, y_hi


def _darken(hex_color: str, factor: float = 0.58) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(max(0.0, min(1.0, c * factor)) for c in rgb)


def _save_both(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {pdf_path.resolve()}")
    print(f"Saved: {png_path.resolve()}")


def _validate_detection_color_mode() -> str:
    mode = str(DETECTION_MARKER_COLOR_MODE).strip().lower()
    if mode not in {"uniform", "policy"}:
        raise ValueError(
            "DETECTION_MARKER_COLOR_MODE must be 'uniform' or 'policy', "
            f"got {DETECTION_MARKER_COLOR_MODE!r}."
        )
    return mode


def _draw_detection_markers(
    ax,
    *,
    times: np.ndarray,
    ys: np.ndarray,
    policy_color: str,
    zorder: int,
) -> None:
    if times.size == 0:
        return

    mode = _validate_detection_color_mode()
    if mode == "policy":
        color = policy_color
        edge = _darken(policy_color, factor=0.48)
    else:
        color = UNIFORM_DETECTION_COLOR
        edge = UNIFORM_DETECTION_EDGE

    # One consistent bullseye symbol. Only its color mode changes.
    for t, y in zip(times, ys):
        ax.scatter(
            [t], [y],
            s=DETECTION_MARKER_SIZE * 2.05,
            facecolor=color, edgecolor="none", alpha=0.15,
            zorder=zorder,
        )
        ax.scatter(
            [t], [y],
            s=DETECTION_MARKER_SIZE, marker="o",
            facecolor=color, edgecolor=edge, linewidth=2.8,
            zorder=zorder + 1,
        )
        ax.scatter(
            [t], [y],
            s=DETECTION_MARKER_SIZE * 0.43, marker="o",
            facecolor="white", edgecolor="white", linewidth=0.0,
            zorder=zorder + 2,
        )
        ax.scatter(
            [t], [y],
            s=DETECTION_INNER_SIZE, marker="o",
            facecolor=color, edgecolor="none",
            zorder=zorder + 3,
        )

def _draw(loaded: list[dict], base: Path, *, full_axes: bool) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor("#FBFCFE")

    for idx, item in enumerate(loaded):
        style = LINE_STYLES[idx]
        color = style["color"]
        lw = float(style["linewidth"])
        label = _label_for(item["scene"], idx)
        dark = _darken(color)

        # A soft glow plus a true two-stroke curve: dark outer rail and vivid core.
        # This adds depth without introducing dash patterns or extra data symbols.
        ax.plot(
            item["smooth_t"], item["smooth_cov"],
            color=color, linewidth=lw + 6.0, alpha=0.085,
            solid_capstyle="round", solid_joinstyle="round", zorder=1 + idx,
        )
        ax.plot(
            item["smooth_t"], item["smooth_cov"],
            color=dark, linewidth=lw + 2.4, linestyle="-", alpha=0.96,
            solid_capstyle="round", solid_joinstyle="round", zorder=3 + idx,
        )
        ax.plot(
            item["smooth_t"], item["smooth_cov"],
            color=color, linewidth=lw, linestyle="-", alpha=1.0,
            solid_capstyle="round", solid_joinstyle="round",
            label=label, zorder=5 + idx,
        )

        if SHOW_DETECTION_MARKERS and item["detection_events"]:
            valid_events = [
                (t, cls)
                for t, cls in item["detection_events"]
                if item["smooth_t"][0] <= t <= item["smooth_t"][-1]
            ]
            if valid_events:
                det_t = np.asarray([event[0] for event in valid_events], dtype=float)
                det_y = np.interp(det_t, item["smooth_t"], item["smooth_cov"])
                _draw_detection_markers(
                    ax,
                    times=det_t,
                    ys=det_y,
                    policy_color=color,
                    zorder=12 + idx * 5,
                )

    ax.set_xlabel("Time [s]", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
    ax.set_ylabel("Visited-cell ratio [%]", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, pad=4)
    ax.grid(True, color="#718096", linewidth=0.95, alpha=0.22)
    ax.spines["top"].set_alpha(0.60)
    ax.spines["right"].set_alpha(0.60)

    if full_axes:
        ax.set_xlim(0.0, EPISODE_DURATION_S)
        ax.set_ylim(0.0, FULL_COVERAGE_PERCENT)
        ax.set_xticks(np.arange(0.0, EPISODE_DURATION_S + 0.1, 30.0))
        ax.set_yticks(np.arange(0.0, FULL_COVERAGE_PERCENT + 0.1, 20.0))
    else:
        x_hi, y_hi = _scaled_limits(loaded)
        ax.set_xlim(0.0, x_hi)
        ax.set_ylim(0.0, y_hi)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))

    handles = [
        Line2D(
            [0], [0],
            color=LINE_STYLES[i]["color"],
            linewidth=LINE_STYLES[i]["linewidth"],
            linestyle="-",
            label=_label_for(item["scene"], i),
        )
        for i, item in enumerate(loaded)
    ]
    if SHOW_DETECTION_MARKERS and any(item["detection_events"] for item in loaded):
        mode = _validate_detection_color_mode()
        if mode == "uniform":
            legend_face = UNIFORM_DETECTION_COLOR
            legend_edge = UNIFORM_DETECTION_EDGE
            legend_label = "Confirmed target detection"
        else:
            # Neutral legend swatch: actual plot markers inherit each policy color.
            legend_face = "white"
            legend_edge = "#30343B"
            legend_label = "Confirmed target detection (policy color)"
        handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="None", markersize=10.8,
                markerfacecolor=legend_face,
                markeredgecolor=legend_edge, markeredgewidth=2.2,
                label=legend_label,
            )
        )

    ncol = min(4, max(2, math.ceil(len(handles) / 2)))
    ax.legend(
        handles=handles,
        loc="lower center", bbox_to_anchor=(0.5, 1.03),
        ncol=ncol, fontsize=LEGEND_FONTSIZE,
        frameon=True, framealpha=0.97, facecolor="white",
        edgecolor="#C3C7CC", handlelength=2.7,
        columnspacing=1.30, handletextpad=0.62,
        borderpad=0.68, labelspacing=0.55,
    )

    fig.subplots_adjust(top=0.80, left=0.13, right=0.985, bottom=0.16)
    _save_both(fig, base)
    plt.close(fig)


def main() -> None:
    if not (2 <= len(RUN_PATHS) <= MAX_RUNS):
        raise ValueError(f"RUN_PATHS must contain between 2 and {MAX_RUNS} runs.")

    loaded = []
    for raw in RUN_PATHS:
        run_dir, episode_path, snapshot_path, claim_path = _resolve_run(raw)
        scene = _read_json(snapshot_path)
        log = _read_episode_log(episode_path)
        smooth_t, smooth_cov = _smooth_curve(log["episode_time_s"], log["coverage_percent"])

        confirmed_sim_events = _read_confirmed_detection_events(claim_path, scene)
        confirmed_episode_events = _detection_episode_events(confirmed_sim_events, log)
        total_targets = len(_true_target_keys(scene))
        bottle_count = sum(1 for _, cls in confirmed_episode_events if cls == "target_bottle")
        can_count = sum(1 for _, cls in confirmed_episode_events if cls == "target_can")
        print(
            f"[{_label_for(scene, len(loaded))}] confirmed detections: "
            f"{len(confirmed_episode_events)}/{total_targets} "
            f"(bottles={bottle_count}, cans={can_count})"
        )

        loaded.append({
            "run_dir": run_dir,
            "scene": scene,
            "log": log,
            "smooth_t": smooth_t,
            "smooth_cov": smooth_cov,
            "detection_events": confirmed_episode_events,
        })

    _validate_same_scene([item["scene"] for item in loaded])

    base = _output_base()
    _draw(loaded, base, full_axes=False)
    _draw(loaded, base.with_name(base.name + "_full"), full_axes=True)


if __name__ == "__main__":
    main()
