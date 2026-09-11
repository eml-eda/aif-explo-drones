# webots/controllers/crazyflie_AIF/utils/run_visualizations.py

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.ticker import FuncFormatter

PLOTS_SUBDIR = "trajectory plots"

# Simple low->high scale for occupancy time.
OCCUPANCY_CMAP = LinearSegmentedColormap.from_list(
    "occupancy_blue_to_red",
    ["#2166ac", "#f7f7f7", "#b2182b"],
)

TARGET_OBJECT_STYLES = {
    "target_bottle": {
        "label": "target bottle",
        "marker": "X",
        "color": "tab:green",
    },
    "target_can": {
        "label": "target can",
        "marker": "X",
        "color": "tab:blue",
    },
}

DESK_CLASS_NAMES = {"desk", "table", "tabletop"}


def _latest_file(directory: Path, pattern: str) -> Path | None:
    matches = list(directory.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _policy_name_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name

    if not name.startswith("log_run_"):
        return name

    body = name[len("log_run_"):]

    # Single run:
    #   aif_binary_20260507_151252
    # Eval run:
    #   aif_binary_run00_seed28_20260507_151252
    m = re.search(r"_(?:run\d+_seed[^_]+_)?\d{8}_\d{6}$", body)
    if m:
        return body[:m.start()]

    return body


def read_occupancy_summary(summary_csv_path: str | Path):
    summary_csv_path = Path(summary_csv_path)

    with open(summary_csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        header = next(reader, None)
        if header is None or header[0] != "arena_width_m":
            raise ValueError(f"Invalid occupancy summary header in {summary_csv_path}")

        meta_row = next(reader, None)
        if meta_row is None or len(meta_row) < 9:
            raise ValueError(f"Invalid occupancy summary metadata in {summary_csv_path}")

        meta = {
            "arena_width_m": float(meta_row[0]),
            "arena_height_m": float(meta_row[1]),
            "cell_size_m": float(meta_row[2]),
            "min_x": float(meta_row[3]),
            "max_x": float(meta_row[4]),
            "min_y": float(meta_row[5]),
            "max_y": float(meta_row[6]),
            "rows": int(meta_row[7]),
            "cols": int(meta_row[8]),
            "min_time_s": float(meta_row[9]) if len(meta_row) >= 10 else 0.0,
        }

        grid_header = next(reader, None)
        if grid_header is None or grid_header[:3] != ["row", "col", "time_s"]:
            raise ValueError(f"Invalid occupancy grid header in {summary_csv_path}")

        grid = np.zeros((meta["rows"], meta["cols"]), dtype=float)

        for row in reader:
            if len(row) < 3:
                continue

            r = int(row[0])
            c = int(row[1])
            t = float(row[2])

            if 0 <= r < meta["rows"] and 0 <= c < meta["cols"]:
                grid[r, c] = t

    return meta, grid


def read_occupancy_log(occupancy_log_path: str | Path):
    occupancy_log_path = Path(occupancy_log_path)

    times = []
    xs = []
    ys = []
    rows = []
    cols = []
    inside = []

    with open(occupancy_log_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        required = {"time_s", "x_m", "y_m", "row", "col", "inside"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Invalid occupancy log columns in {occupancy_log_path}")

        for row in reader:
            times.append(float(row["time_s"]))
            xs.append(float(row["x_m"]))
            ys.append(float(row["y_m"]))
            rows.append(int(row["row"]))
            cols.append(int(row["col"]))
            inside.append(int(row["inside"]) == 1)

    return {
        "time_s": np.asarray(times, dtype=float),
        "x_m": np.asarray(xs, dtype=float),
        "y_m": np.asarray(ys, dtype=float),
        "row": np.asarray(rows, dtype=int),
        "col": np.asarray(cols, dtype=int),
        "inside": np.asarray(inside, dtype=bool),
    }


def read_spawned_objects(gt_json_path: str | Path | None):
    if gt_json_path is None:
        return []

    gt_json_path = Path(gt_json_path)
    if not gt_json_path.exists():
        return []

    try:
        with open(gt_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        return []

    cleaned = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if "x" not in obj or "y" not in obj:
            continue

        cleaned.append(
            {
                "x": float(obj["x"]),
                "y": float(obj["y"]),
                "class_name": str(obj.get("class_name", obj.get("label", "object"))),
                "cx": obj.get("cx", None),
                "cy": obj.get("cy", None),
            }
        )

    return cleaned


def read_claim_log(claim_log_path: str | Path | None):
    """
    Optional claim-event log for trajectory red markers.

    Expected useful columns:
      time_s,x_m,y_m,class_name,claim_cx,claim_cy,confidence

    The visualization only requires x_m and y_m. Extra columns are ignored.
    If the file is missing or has no usable rows, this returns an empty list.
    """
    if claim_log_path is None:
        return []

    claim_log_path = Path(claim_log_path)
    if not claim_log_path.exists():
        return []

    out = []
    try:
        with open(claim_log_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if not {"x_m", "y_m"}.issubset(set(reader.fieldnames or [])):
                return []

            for row in reader:
                try:
                    out.append(
                        {
                            "time_s": None if row.get("time_s") in (None, "") else float(row["time_s"]),
                            "x": float(row["x_m"]),
                            "y": float(row["y_m"]),
                            "class_name": str(row.get("class_name", "claim")),
                            "confidence": None if row.get("confidence") in (None, "") else float(row["confidence"]),
                        }
                    )
                except Exception:
                    continue
    except Exception:
        return []

    return out


def _world_to_grid_xy(x, y, meta: dict):
    """
    Convert Webots world-frame x/y into continuous grid coordinates.

    Integer coordinates are cell centers, matching the belief heatmap convention:
      x axis = cx in [0, cols-1]
      y axis = cy in [0, rows-1]
    """
    cell_size = float(meta["cell_size_m"])
    gx = (np.asarray(x, dtype=float) - float(meta["min_x"])) / cell_size - 0.5
    gy = (np.asarray(y, dtype=float) - float(meta["min_y"])) / cell_size - 0.5
    return gx, gy


def _configure_cell_axis(ax, meta: dict):
    rows = int(meta["rows"])
    cols = int(meta["cols"])

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    ax.set_xticks([x - 0.5 for x in range(cols + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(rows + 1)], minor=True)
    ax.grid(which="minor", color="black", linewidth=0.45, alpha=0.18)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel("cx (col)")
    ax.set_ylabel("cy (row)")


def _plot_spawned_objects(ax, objects, meta: dict):
    """Plot spawned objects without per-object text labels.

    Targets keep class-specific symbols/colors, desks/tables use one shared
    symbol, and every other spawned object is grouped into one generic
    distractor category. This keeps the plot readable even when many
    distractor classes are present.
    """
    grouped = {
        "target_bottle": [],
        "target_can": [],
        "desk": [],
        "distractor": [],
    }

    for obj in objects:
        class_name = str(obj.get("class_name", "object")).strip().lower()

        if class_name in TARGET_OBJECT_STYLES:
            category = class_name
        elif class_name in DESK_CLASS_NAMES:
            category = "desk"
        else:
            category = "distractor"

        try:
            gx, gy = _world_to_grid_xy(float(obj["x"]), float(obj["y"]), meta)
            grouped[category].append((float(gx), float(gy)))
        except Exception:
            continue

    for class_name in ("target_bottle", "target_can"):
        points = grouped[class_name]
        if not points:
            continue

        style = TARGET_OBJECT_STYLES[class_name]
        xs, ys = zip(*points)
        ax.scatter(
            xs,
            ys,
            marker=style["marker"],
            s=185,
            linewidths=2.6,
            color=style["color"],
            edgecolors="black",
            label=style["label"],
            zorder=7,
        )

    if grouped["desk"]:
        xs, ys = zip(*grouped["desk"])
        ax.scatter(
            xs,
            ys,
            marker="s",
            s=145,
            facecolors="none",
            edgecolors="tab:gray",
            linewidths=2.7,
            label="desk / table",
            zorder=6,
        )

    if grouped["distractor"]:
        xs, ys = zip(*grouped["distractor"])
        ax.scatter(
            xs,
            ys,
            marker="x",
            s=145,
            linewidths=2.8,
            color="tab:gray",
            label="distractor",
            zorder=6,
        )


def _plot_claim_events(ax, claim_events, meta: dict):
    if not claim_events:
        return

    xs = [float(ev["x"]) for ev in claim_events]
    ys = [float(ev["y"]) for ev in claim_events]
    gx, gy = _world_to_grid_xy(xs, ys, meta)

    ax.scatter(
        gx,
        gy,
        marker="o",
        s=145,
        color="red",
        edgecolors="black",
        linewidths=1.2,
        label="detection claim",
        zorder=9,
    )


def plot_occupancy_heatmap(
    *,
    summary_csv_path: str | Path,
    output_dir: str | Path,
    policy_name: str | None = None,
    output_name: str = "occupancy_heatmap.png",
):
    summary_csv_path = Path(summary_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta, grid = read_occupancy_summary(summary_csv_path)

    if policy_name is None:
        policy_name = _policy_name_from_run_dir(summary_csv_path.parent)

    fig, ax = plt.subplots(figsize=(8, 9))

    positive = grid[grid > 0.0]

    occupancy_cmap = LinearSegmentedColormap.from_list(
        "occupancy_positive_red",
        [
            "#fee5d9",  # low positive occupancy
            "#fcae91",
            "#fb6a4a",
            "#de2d26",
            "#a50f15",  # high occupancy
        ],
    )

    # Mask true zero cells so they are rendered as neutral background,
    # not as "low value" cells.
    plot_grid = np.ma.masked_where(grid <= 0.0, grid)
    occupancy_cmap.set_bad("#f2f2f2")

    if positive.size > 0:
        vmin = float(max(positive.min(), 1e-6))
        vmax = float(positive.max())

        if vmax > vmin:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=0.0, vmax=max(1.0, vmax))
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)

    im = ax.imshow(
        plot_grid,
        origin="lower",
        extent=(-0.5, meta["cols"] - 0.5, -0.5, meta["rows"] - 0.5),
        cmap=occupancy_cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )

    _configure_cell_axis(ax, meta)

    ax.set_title(
        f"Cell occupancy time — {policy_name}\n"
        f"{meta['rows']}×{meta['cols']} grid, cell={meta['cell_size_m']:.2f}m"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cumulated time (s)")

    if positive.size > 0:
        tick_candidates = np.array(
            [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 60],
            dtype=float,
        )
        ticks = tick_candidates[
            (tick_candidates >= vmin) & (tick_candidates <= vmax)
        ]

        if ticks.size >= 2:
            cbar.set_ticks(ticks)

    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _pos: f"{x:g}")
    )

    fig.tight_layout()

    out_path = output_dir / output_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return out_path


def plot_trajectory(
    *,
    occupancy_log_path: str | Path,
    summary_csv_path: str | Path,
    output_dir: str | Path,
    policy_name: str | None = None,
    gt_json_path: str | Path | None = None,
    claim_log_path: str | Path | None = None,
    output_name: str = "trajectory_plot.png",
):
    occupancy_log_path = Path(occupancy_log_path)
    summary_csv_path = Path(summary_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta, _grid = read_occupancy_summary(summary_csv_path)
    log = read_occupancy_log(occupancy_log_path)
    objects = read_spawned_objects(gt_json_path)
    claim_events = read_claim_log(claim_log_path)

    if policy_name is None:
        policy_name = _policy_name_from_run_dir(occupancy_log_path.parent)

    inside = log["inside"]

    if inside.sum() == 0:
        raise ValueError(f"No inside-arena trajectory samples found in {occupancy_log_path}")

    x = log["x_m"][inside]
    y = log["y_m"][inside]
    gx, gy = _world_to_grid_xy(x, y, meta)

    # Slightly wider canvas leaves room for a compact legend outside the grid.
    fig, ax = plt.subplots(figsize=(9.5, 9))

    _configure_cell_axis(ax, meta)

    # Make the trajectory readable at normal figure size: one solid line, no
    # per-sample dots. Detection claims remain as explicit red event markers.
    ax.plot(
        gx,
        gy,
        linewidth=2.8,
        color="black",
        alpha=0.72,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="trajectory",
        zorder=3,
    )

    _plot_claim_events(ax, claim_events, meta)

    ax.scatter(
        [gx[0]],
        [gy[0]],
        marker="o",
        s=175,
        color="tab:green",
        edgecolors="black",
        linewidths=1.2,
        label="start",
        zorder=10,
    )
    ax.scatter(
        [gx[-1]],
        [gy[-1]],
        marker="*",
        s=260,
        color="tab:orange",
        edgecolors="black",
        linewidths=1.0,
        label="end",
        zorder=10,
    )

    if objects:
        _plot_spawned_objects(ax, objects, meta)

    # No plot title: the grid itself is the focus and the legend carries the
    # semantic information. Keep only the coordinate-axis labels.
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(which="minor", color="black", linewidth=0.60, alpha=0.22)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        frameon=True,
        framealpha=0.92,
        handlelength=2.0,
        handletextpad=0.7,
        labelspacing=0.65,
        borderpad=0.65,
    )

    fig.subplots_adjust(right=0.78)

    out_path = output_dir / output_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


def generate_run_visualizations(
    *,
    run_dir: str | Path,
    make_heatmap: bool,
    make_trajectory: bool,
    occupancy_summary_path: str | Path | None = None,
    occupancy_log_path: str | Path | None = None,
    claim_log_path: str | Path | None = None,
    gt_json_path: str | Path | None = None,
    plots_subdir: str = PLOTS_SUBDIR,
):
    run_dir = Path(run_dir)

    if occupancy_summary_path is None:
        occupancy_summary_path = _latest_file(run_dir, "occupancy_summary_*.csv")

    if occupancy_log_path is None:
        occupancy_log_path = _latest_file(run_dir, "occupancy_log_*.csv")

    if claim_log_path is None:
        claim_log_path = _latest_file(run_dir, "claim_log_*.csv")

    if occupancy_summary_path is None:
        raise FileNotFoundError(f"No occupancy_summary_*.csv found in {run_dir}")

    if make_trajectory and occupancy_log_path is None:
        raise FileNotFoundError(f"No occupancy_log_*.csv found in {run_dir}")

    output_dir = run_dir / plots_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_name = _policy_name_from_run_dir(run_dir)

    outputs = {}

    if make_heatmap:
        outputs["occupancy_heatmap"] = str(
            plot_occupancy_heatmap(
                summary_csv_path=occupancy_summary_path,
                output_dir=output_dir,
                policy_name=policy_name,
            )
        )

    if make_trajectory:
        outputs["trajectory_plot"] = str(
            plot_trajectory(
                occupancy_log_path=occupancy_log_path,
                summary_csv_path=occupancy_summary_path,
                output_dir=output_dir,
                policy_name=policy_name,
                gt_json_path=gt_json_path,
                claim_log_path=claim_log_path,
            )
        )

    return outputs