# utils/occupancy_heatmap.py
#
# Set SUMMARY_CSV_PATH below and run:
#   python3 webots/controllers/crazyflie_AIF/utils/occupancy_heatmap.py
#
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---- EDIT THIS ----
SUMMARY_CSV_PATH = "webots/controllers/crazyflie_AIF/logs/log_run_rotate_and_measure_20260324_212017/occupancy_summary_20260324_212017.csv"
# -------------------


def extract_policy_name_from_summary_path(summary_path: Path) -> str:
    """
    Extracts policy name from a path like:
      .../log_run_rotate_and_measure_20260324_212017/occupancy_summary_...

    Returns:
      rotate_and_measure
    """
    parent_name = summary_path.parent.name
    match = re.match(r"log_run_(.+)_\d{8}_\d{6}$", parent_name)
    if not match:
        raise ValueError(
            f"Could not extract policy name from parent folder '{parent_name}'. "
            f"Expected format like 'log_run_<policy>_YYYYMMDD_HHMMSS'."
        )
    return match.group(1)


def output_png_path_for_policy(policy_name: str) -> Path:
    out_dir = Path("webots/visuals/occ_heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{policy_name}.png"


def main():
    summary_path = Path(SUMMARY_CSV_PATH)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    with open(summary_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        header = next(reader, None)
        if header is None or header[0] != "arena_width_m":
            raise ValueError("Invalid summary file: missing arena metadata header.")

        meta = next(reader, None)
        if meta is None or len(meta) < 9:
            raise ValueError("Invalid summary file: missing arena metadata values.")

        arena_width = float(meta[0])
        arena_height = float(meta[1])
        cell_size = float(meta[2])
        min_x = float(meta[3])
        max_x = float(meta[4])
        min_y = float(meta[5])
        max_y = float(meta[6])
        rows = int(meta[7])
        cols = int(meta[8])

        min_time_s = float(meta[9]) if len(meta) >= 10 else 0.0

        grid_header = next(reader, None)
        if grid_header is None or grid_header[:3] != ["row", "col", "time_s"]:
            raise ValueError("Invalid summary file: missing grid header.")

        grid = np.zeros((rows, cols), dtype=float)
        for row in reader:
            if len(row) < 3:
                continue
            r = int(row[0])
            c = int(row[1])
            t = float(row[2])
            if 0 <= r < rows and 0 <= c < cols:
                grid[r, c] = t

    policy_name = extract_policy_name_from_summary_path(summary_path)
    out_path = output_png_path_for_policy(policy_name)

    extent = [min_x, max_x, min_y, max_y]

    fig, ax = plt.subplots(figsize=(8, 9))

    # Better occupancy colormap:
    # "magma" avoids near-zero pure black looking too dominant and gives a much clearer progression.
    im = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap="turbo",
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Cell Occupancy Time (s) — {policy_name} — from t >= {min_time_s:.1f}s")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Seconds")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[OCC HEATMAP] wrote {out_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()