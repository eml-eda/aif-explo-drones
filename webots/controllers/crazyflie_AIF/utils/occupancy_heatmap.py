# utils/occupancy_heatmap.py
#
# Set SUMMARY_CSV_PATH below and run:
#   python3 webots/controllers/crazyflie_AIF/utils/occupancy_heatmap.py
#

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---- EDIT THIS ----
SUMMARY_CSV_PATH = "webots/controllers/crazyflie_AIF/logs/log_run_wall_following_20260219_150648/occupancy_summary_20260219_150648.csv"
# -------------------


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

    extent = [min_x, max_x, min_y, max_y]

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap="inferno",
        aspect="equal",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Cell Occupancy Time (s)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Seconds")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
