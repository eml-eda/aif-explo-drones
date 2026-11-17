from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch, Polygon
from matplotlib.path import Path as MplPath
import numpy as np


# =============================================================================
# USER SETTINGS
# =============================================================================
# Each entry may be either:
#   - a run directory containing scene_snapshot.json + episode_log_*.csv, or
#   - the episode_log_*.csv file itself.
# Keep between 2 and 6 runs.
RUN_PATHS = [
    # Relative paths are resolved from the current directory, for example:
    # "webots/controllers/crazyflie_AIF/logs/log_run_spiral_<timestamp>",
    # "webots/controllers/crazyflie_AIF/logs/log_run_zoi_<timestamp>",
]

# Relative paths are resolved from the repository root.
OUTPUT_DIR = "Visuals_DATE"
# Give only the base name; .pdf and .png are created automatically.
OUTPUT_FILENAME = "trajectory_comparison_SEED_42_AIFs"

# Leave empty to infer canonical policy names. Otherwise provide one label per run.
CUSTOM_LABELS = []

# Optional transparent PNG resources used instead of the built-in target silhouettes.
# Relative paths are resolved from the repository root. Leave "" for built-in markers.
BOTTLE_ICON_PATH = ""
CAN_ICON_PATH = ""
BOTTLE_ICON_ZOOM = 0.128
CAN_ICON_ZOOM = 0.112

SHOW_TARGETS = True
SHOW_TABLES = True
SHOW_DISTRACTORS = True
SHOW_START = True
SHOW_ENDS = True

# The first logged autonomous-search positions must agree within this tolerance.
START_POSITION_TOLERANCE_M = 0.08

# Figure tuned for readability after reduction in a two-column paper.
FIGSIZE = (8.4, 8.0)
SAVE_DPI = 600
MAX_RUNS = 6

# Typography / grid.
AXIS_LABEL_FONTSIZE = 19
TICK_FONTSIZE = 14.5
LEGEND_FONTSIZE = 13.0
GRID_LINEWIDTH = 0.95
GRID_ALPHA = 0.38

# Semantic-marker palette deliberately kept separate from the first route colors.
START_COLOR = "#FFC857"     # luminous gold
START_EDGE = "#6A4B00"
END_COLOR = "#A855F7"       # vivid amethyst
END_EDGE = "#54218A"
BOTTLE_COLOR = "#35D07F"    # bright emerald
BOTTLE_EDGE = "#0D6840"
CAN_COLOR = "#F0445E"       # vivid coral-red
CAN_EDGE = "#8C1D31"
TABLE_EDGE = "#85562E"
TABLE_FACE = "#D9A86C"

POLICY_LABELS = {
    "wall_following": "Wall Following",
    "spiral": "Spiral",
    "pseudo_random": "Pseudo-Random",
    "rotate_and_measure": "Rotate and Measure",
    "zoi": "ZOI",
}

# Vivid route palette. The first two are intentionally far from the semantic-marker
# palette because most publication figures compare two policies.
LINE_STYLES = [
    # Main two-policy publication pairing.
    # The orange route keeps the richer double-stroke treatment. The blue route is
    # deliberately a SINGLE clean dashed stroke so overlaps remain easy to parse.
    {"color": "#FF7A00", "linestyle": "-",                "linewidth": 6.8, "outer_extra": 3.0, "glow_extra": 6.0},
    {"color": "#0B3B6E", "linestyle": (0, (13.0, 5.0)),    "linewidth": 7.4, "single_stroke": True},
    {"color": "#00A6D6", "linestyle": "-.",               "linewidth": 5.6, "outer_extra": 2.2, "glow_extra": 4.2},
    {"color": "#7A3FF2", "linestyle": ":",                "linewidth": 5.6, "outer_extra": 2.2, "glow_extra": 4.2},
    {"color": "#D64FA3", "linestyle": (0, (5, 1.6)),       "linewidth": 5.4, "outer_extra": 2.2, "glow_extra": 4.2},
    {"color": "#B88900", "linestyle": (0, (3, 1, 1, 1)),   "linewidth": 5.4, "outer_extra": 2.2, "glow_extra": 4.2},
]

DESK_NAMES = {"desk", "table", "tabletop"}
TARGET_CLASS_NAMES = {"target_bottle", "target_can"}


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelweight": "bold",
    "axes.linewidth": 1.35,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    # Keep custom dash lengths in absolute points instead of scaling them with
    # the thick publication linewidth. This prevents the dashed trajectory from
    # turning into a few oversized rounded capsules.
    "lines.scale_dashes": False,
})


# Stylized built-in target silhouettes, used when no PNG icons are supplied.
_BOTTLE_PATH = MplPath(
    [
        (-0.23, -0.50), (0.23, -0.50), (0.23, 0.12),
        (0.17, 0.27), (0.10, 0.34), (0.10, 0.56),
        (-0.10, 0.56), (-0.10, 0.34), (-0.17, 0.27),
        (-0.23, 0.12), (-0.23, -0.50),
    ],
    [
        MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
        MplPath.LINETO, MplPath.LINETO, MplPath.LINETO,
        MplPath.LINETO, MplPath.LINETO, MplPath.LINETO,
        MplPath.LINETO, MplPath.CLOSEPOLY,
    ],
)

_CAN_PATH = MplPath(
    [
        (-0.30, -0.48), (0.30, -0.48), (0.34, -0.38),
        (0.34, 0.38), (0.30, 0.48), (-0.30, 0.48),
        (-0.34, 0.38), (-0.34, -0.38), (-0.30, -0.48),
    ],
    [
        MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
        MplPath.LINETO, MplPath.LINETO, MplPath.LINETO,
        MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY,
    ],
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    # .../webots/controllers/crazyflie_AIF/utils/script.py -> repo is parents[4]
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


def _resolve_resource(path_value: str) -> Path | None:
    if not str(path_value).strip():
        return None
    p = Path(path_value).expanduser()
    p = p if p.is_absolute() else (_repo_root() / p)
    if not p.exists():
        raise FileNotFoundError(f"Target icon not found: {p}")
    return p


def _latest_file(run_dir: Path, pattern: str) -> Path | None:
    matches = sorted(run_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _resolve_run(path_like: str) -> tuple[Path, Path, Path]:
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
    return run_dir, episode_log, snapshot


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_episode_log(path: Path) -> dict[str, np.ndarray]:
    required = {"episode_time_s", "x_m", "y_m", "yaw_rad", "coverage_ratio"}
    out = {k: [] for k in required}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise ValueError(f"{path} is missing columns: {sorted(required - fields)}")
        for row in reader:
            for key in required:
                out[key].append(float(row[key]))
    if not out["x_m"]:
        raise ValueError(f"No trajectory samples in {path}")
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


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


def _validate_logged_starts(logs: list[dict[str, np.ndarray]]) -> tuple[float, float]:
    starts = np.asarray([[log["x_m"][0], log["y_m"][0]] for log in logs], dtype=float)
    ref = starts[0]
    distances = np.linalg.norm(starts - ref, axis=1)
    worst = float(distances.max(initial=0.0))
    if worst > START_POSITION_TOLERANCE_M:
        raise ValueError(
            "Provided runs do not begin from the same autonomous-search position: "
            f"largest discrepancy is {worst:.3f} m, tolerance={START_POSITION_TOLERANCE_M:.3f} m."
        )
    return float(starts[:, 0].mean()), float(starts[:, 1].mean())


def _world_to_grid(x, y, grid: dict):
    cs = float(grid["cell_size_m"])
    gx = (np.asarray(x, dtype=float) - float(grid["min_x"])) / cs - 0.5
    gy = (np.asarray(y, dtype=float) - float(grid["min_y"])) / cs - 0.5
    return gx, gy


def _rect_corners_from_object(obj: dict) -> list[tuple[float, float]]:
    tabletop = obj.get("tabletop_rect", {}) or {}
    corners = tabletop.get("corners_xy", None)
    if isinstance(corners, list) and len(corners) >= 4:
        return [(float(p[0]), float(p[1])) for p in corners[:4]]

    dims = obj.get("dimensions_m", {}) or {}
    length = float(dims.get("length", 0.0) or 0.0)
    width = float(dims.get("width", 0.0) or 0.0)
    if length <= 0.0 or width <= 0.0:
        return []

    cx = float(obj.get("x", 0.0))
    cy = float(obj.get("y", 0.0))
    yaw = float(obj.get("yaw", 0.0) or 0.0)
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for lx, ly in [
        (-length / 2, -width / 2),
        ( length / 2, -width / 2),
        ( length / 2,  width / 2),
        (-length / 2,  width / 2),
    ]:
        out.append((cx + c * lx - s * ly, cy + s * lx + c * ly))
    return out


def _add_icon(ax, *, x: float, y: float, image, zoom: float, zorder: int) -> None:
    artist = AnnotationBbox(
        OffsetImage(image, zoom=zoom),
        (float(x), float(y)),
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0.0,
        zorder=zorder,
    )
    ax.add_artist(artist)


def _plot_fallback_target(ax, *, gx: float, gy: float, class_name: str) -> None:
    if class_name == "target_bottle":
        marker = _BOTTLE_PATH
        face = BOTTLE_COLOR
        edge = BOTTLE_EDGE
        size = 485
    else:
        marker = _CAN_PATH
        face = CAN_COLOR
        edge = CAN_EDGE
        size = 380

    ax.scatter(
        [gx], [gy], marker=marker, s=size,
        facecolor="white", edgecolor="white", linewidth=5.2, zorder=15,
    )
    ax.scatter(
        [gx], [gy], marker=marker, s=size,
        facecolor=face, edgecolor=edge, linewidth=2.4, zorder=16,
    )


def _plot_scene(ax, scene: dict, grid: dict, bottle_icon, can_icon) -> None:
    objects = (scene.get("spawned_objects", {}) or {}).get("objects", []) or []

    if SHOW_TABLES:
        for obj in objects:
            name = str(obj.get("class_name", "")).strip().lower()
            if name not in DESK_NAMES:
                continue
            corners = _rect_corners_from_object(obj)
            if not corners:
                continue
            poly_xy = [tuple(map(float, _world_to_grid(x, y, grid))) for x, y in corners]
            ax.add_patch(
                Polygon(
                    poly_xy,
                    closed=True,
                    facecolor=TABLE_FACE,
                    edgecolor=TABLE_EDGE,
                    linewidth=2.5,
                    alpha=0.22,
                    zorder=1,
                )
            )

    if SHOW_TARGETS:
        for obj in objects:
            class_name = str(obj.get("class_name", ""))
            if class_name not in TARGET_CLASS_NAMES:
                continue
            gx, gy = _world_to_grid(float(obj["x"]), float(obj["y"]), grid)
            gx, gy = float(gx), float(gy)
            icon = bottle_icon if class_name == "target_bottle" else can_icon
            if icon is not None:
                icon_zoom = BOTTLE_ICON_ZOOM if class_name == "target_bottle" else CAN_ICON_ZOOM
                _add_icon(ax, x=gx, y=gy, image=icon, zoom=icon_zoom, zorder=16)
            else:
                _plot_fallback_target(ax, gx=gx, gy=gy, class_name=class_name)

    if SHOW_DISTRACTORS:
        pts = [
            obj for obj in objects
            if str(obj.get("class_name", "")) not in TARGET_CLASS_NAMES
            and str(obj.get("class_name", "")).strip().lower() not in DESK_NAMES
            and "x" in obj and "y" in obj
        ]
        if pts:
            coords = [_world_to_grid(float(o["x"]), float(o["y"]), grid) for o in pts]
            xs, ys = zip(*[(float(x), float(y)) for x, y in coords])
            ax.scatter(
                xs, ys, marker="x", s=78,
                color="#6A6F78", linewidths=2.2, zorder=4,
            )


def _configure_grid(ax, grid: dict) -> None:
    rows = int(grid["rows"])
    cols = int(grid["cols"])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#FBFCFE")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks([v - 0.5 for v in range(cols + 1)], minor=True)
    ax.set_yticks([v - 0.5 for v in range(rows + 1)], minor=True)
    ax.grid(which="minor", color="#6C7684", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, pad=5)
    ax.set_xlabel(r"Grid $c_x$ (column)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
    ax.set_ylabel(r"Grid $c_y$ (row)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)


def _label_for(scene: dict, idx: int) -> str:
    if CUSTOM_LABELS:
        if len(CUSTOM_LABELS) != len(RUN_PATHS):
            raise ValueError("CUSTOM_LABELS must be empty or have one entry per RUN_PATHS item.")
        return str(CUSTOM_LABELS[idx])
    policy = str(scene.get("policy", f"run {idx + 1}"))
    return POLICY_LABELS.get(policy, policy.replace("_", " ").title())


def _darken(hex_color: str, factor: float = 0.58) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(max(0.0, min(1.0, c * factor)) for c in rgb)


def _legend_handles(loaded: list[dict], *, has_tables: bool, has_distractors: bool) -> list:
    handles = []
    for idx, item in enumerate(loaded):
        style = LINE_STYLES[idx]
        handles.append(
            Line2D(
                [0], [0],
                color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], label=_label_for(item["scene"], idx),
            )
        )

    if SHOW_START:
        handles.append(
            Line2D([0], [0], marker="h", linestyle="None", markersize=12.0,
                   markerfacecolor=START_COLOR, markeredgecolor=START_EDGE, markeredgewidth=2.0,
                   label="Start")
        )
    if SHOW_ENDS:
        handles.append(
            Line2D([0], [0], marker="D", linestyle="None", markersize=11.5,
                   markerfacecolor=END_COLOR, markeredgecolor=END_EDGE, markeredgewidth=2.0,
                   label="End")
        )
    if SHOW_TARGETS:
        handles.extend([
            Line2D([0], [0], marker=_BOTTLE_PATH, linestyle="None", markersize=15.2,
                   markerfacecolor=BOTTLE_COLOR, markeredgecolor=BOTTLE_EDGE,
                   markeredgewidth=1.7, label="Bottle target"),
            Line2D([0], [0], marker=_CAN_PATH, linestyle="None", markersize=13.0,
                   markerfacecolor=CAN_COLOR, markeredgecolor=CAN_EDGE,
                   markeredgewidth=1.6, label="Can target"),
        ])
    if SHOW_TABLES and has_tables:
        handles.append(Patch(facecolor=TABLE_FACE, edgecolor=TABLE_EDGE, alpha=0.35, label="Table"))
    if SHOW_DISTRACTORS and has_distractors:
        handles.append(
            Line2D([0], [0], marker="x", linestyle="None", markersize=8,
                   color="#6A6F78", label="Distractor")
        )
    return handles


def _save_both(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {pdf_path.resolve()}")
    print(f"Saved: {png_path.resolve()}")


def main() -> None:
    if not (2 <= len(RUN_PATHS) <= MAX_RUNS):
        raise ValueError(f"RUN_PATHS must contain between 2 and {MAX_RUNS} runs.")

    loaded = []
    for raw in RUN_PATHS:
        run_dir, episode_path, snapshot_path = _resolve_run(raw)
        loaded.append({
            "run_dir": run_dir,
            "episode_path": episode_path,
            "scene": _read_json(snapshot_path),
            "log": _read_episode_log(episode_path),
        })

    scenes = [item["scene"] for item in loaded]
    logs = [item["log"] for item in loaded]
    _validate_same_scene(scenes)
    common_start_xy = _validate_logged_starts(logs)
    grid = scenes[0]["grid"]

    bottle_path = _resolve_resource(BOTTLE_ICON_PATH)
    can_path = _resolve_resource(CAN_ICON_PATH)
    bottle_icon = plt.imread(bottle_path) if bottle_path is not None else None
    can_icon = plt.imread(can_path) if can_path is not None else None

    objects = (scenes[0].get("spawned_objects", {}) or {}).get("objects", []) or []
    has_tables = any(
        str(o.get("class_name", "")).strip().lower() in DESK_NAMES
        for o in objects if isinstance(o, dict)
    )
    has_distractors = any(
        str(o.get("class_name", "")) not in TARGET_CLASS_NAMES
        and str(o.get("class_name", "")).strip().lower() not in DESK_NAMES
        for o in objects if isinstance(o, dict)
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _configure_grid(ax, grid)
    _plot_scene(ax, scenes[0], grid, bottle_icon, can_icon)

    for idx, item in enumerate(loaded):
        style = LINE_STYLES[idx]
        log = item["log"]
        gx, gy = _world_to_grid(log["x_m"], log["y_m"], grid)
        gx = np.asarray(gx, dtype=float)
        gy = np.asarray(gy, dtype=float)

        color = style["color"]
        lw = float(style["linewidth"])

        if bool(style.get("single_stroke", False)):
            # Clean publication-style dashed route: exactly one dark-blue stroke.
            # Dash lengths are fixed in points (see lines.scale_dashes=False), so
            # the path reads as one continuous trajectory with regular gaps instead
            # of a handful of oversized capsule-like segments.
            ax.plot(
                gx, gy,
                color=color, linestyle=style["linestyle"],
                linewidth=lw, alpha=1.0,
                solid_capstyle="round", solid_joinstyle="round",
                dash_capstyle="butt", dash_joinstyle="round",
                zorder=10 + idx,
            )
        else:
            # Rich two-stroke treatment retained for the solid route(s).
            outer_color = _darken(color, factor=0.50)
            glow_extra = float(style.get("glow_extra", 4.5))
            outer_extra = float(style.get("outer_extra", 2.3))
            ax.plot(
                gx, gy,
                color=color, linestyle=style["linestyle"],
                linewidth=(lw + glow_extra), alpha=0.075,
                solid_capstyle="round", solid_joinstyle="round",
                dash_capstyle="round", dash_joinstyle="round",
                zorder=3 + idx,
            )
            ax.plot(
                gx, gy,
                color=outer_color, linestyle=style["linestyle"],
                linewidth=(lw + outer_extra), alpha=0.98,
                solid_capstyle="round", solid_joinstyle="round",
                dash_capstyle="round", dash_joinstyle="round",
                zorder=6 + idx,
            )
            ax.plot(
                gx, gy,
                color=color, linestyle=style["linestyle"],
                linewidth=lw, alpha=1.0,
                solid_capstyle="round", solid_joinstyle="round",
                dash_capstyle="round", dash_joinstyle="round",
                zorder=8 + idx,
            )

        if SHOW_ENDS:
            # Fixed semantic styling: endpoint color never duplicates a route color.
            ax.scatter(
                [gx[-1]], [gy[-1]], marker="D", s=285,
                facecolor="white", edgecolor="white", linewidth=5.0, zorder=17 + idx,
            )
            ax.scatter(
                [gx[-1]], [gy[-1]], marker="D", s=285,
                facecolor=END_COLOR, edgecolor=END_EDGE, linewidth=2.5, zorder=18 + idx,
            )

    if SHOW_START:
        sx, sy = _world_to_grid(common_start_xy[0], common_start_xy[1], grid)
        ax.scatter(
            [sx], [sy], marker="h", s=315,
            facecolor="white", edgecolor="white", linewidth=5.3, zorder=20,
        )
        ax.scatter(
            [sx], [sy], marker="h", s=315,
            facecolor=START_COLOR, edgecolor=START_EDGE, linewidth=2.6, zorder=21,
        )
        ax.scatter(
            [sx], [sy], marker="o", s=34,
            facecolor="#FFF8D8", edgecolor="none", zorder=22,
        )

    handles = _legend_handles(loaded, has_tables=has_tables, has_distractors=has_distractors)
    ncol = min(4, max(2, math.ceil(len(handles) / 2)))
    ax.legend(
        handles=handles,
        loc="lower center", bbox_to_anchor=(0.5, 1.025),
        ncol=ncol, fontsize=LEGEND_FONTSIZE,
        frameon=True, framealpha=0.97, facecolor="white",
        edgecolor="#C3C7CC", handlelength=2.9,
        columnspacing=1.25, handletextpad=0.58,
        borderpad=0.68, labelspacing=0.55,
    )

    fig.subplots_adjust(top=0.82, left=0.12, right=0.98, bottom=0.11)
    _save_both(fig, _output_base())
    plt.close(fig)


if __name__ == "__main__":
    main()
