import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_summary(path):
    with open(path, newline="", encoding="utf-8") as csvfile:
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

    return {
        "grid": grid,
        "arena_width": arena_width,
        "arena_height": arena_height,
        "cell_size": cell_size,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
    }


def render_heatmap(summary_path, output_path=None, show=True):
    data = _load_summary(summary_path)
    grid = data["grid"]
    extent = [data["min_x"], data["max_x"], data["min_y"], data["max_y"]]

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

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render occupancy heatmap from summary CSV.")
    parser.add_argument("summary_csv", help="Path to occupancy_summary_*.csv")
    parser.add_argument("--output", help="Optional output image path")
    parser.add_argument("--no-show", action="store_true", help="Do not open a window")
    args = parser.parse_args()

    render_heatmap(Path(args.summary_csv), args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
