"""Expected number of grid cells the belief update touches in one frame.

Usage: python tools/analysis/cells_in_view.py [--samples 20000]

The false-alarm rate f is per cell per frame, so its denominator is the number of
cells updated in a frame: cells whose floor-plane coverage passes the controller's
visible-cell threshold. The calibration images store no drone pose, so the count
comes from geometry alone: uniform drone position over the arena, uniform yaw, the
controller's floor footprint at 1.5 m and its coverage code. No flight data.
"""
import argparse
import os
import random
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "webots", "controllers", "crazyflie_AIF"))

from env.env_manager import FoVSpec, GridEnvManager, GridSpec  # noqa: E402

# Same values as crazyflie_AIF.py.
GRID = GridSpec(min_x=-2.75, max_x=2.75, min_y=-3.25, max_y=3.25, cell_size=0.5, rows=13, cols=11)
FLOOR_FOV = FoVSpec.from_width_height_asym(1.394, 0.553, 0.493)
COVERAGE_UPDATE_THRESHOLD = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    env = GridEnvManager(GRID, FLOOR_FOV)
    counts = []
    for _ in range(args.samples):
        x = rng.uniform(GRID.min_x, GRID.max_x)
        y = rng.uniform(GRID.min_y, GRID.max_y)
        yaw = rng.uniform(-3.141592653589793, 3.141592653589793)
        cov = env.compute_cell_coverages(x, y, yaw)
        counts.append(sum(1 for v in cov.values() if v >= COVERAGE_UPDATE_THRESHOLD))
    se = st.stdev(counts) / len(counts) ** 0.5
    print(f"cells updated per frame: mean {st.mean(counts):.2f} (s.e. {se:.2f}), "
          f"min {min(counts)}, max {max(counts)}, samples {len(counts)}")


if __name__ == "__main__":
    main()
