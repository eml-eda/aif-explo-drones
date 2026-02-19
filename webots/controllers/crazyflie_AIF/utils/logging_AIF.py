# utils/logging_AIF.py

import os
from datetime import datetime


class ExperimentLogger:
    """
    Depends only on grid_spec.
    Writes:
      - occupancy_log_<ts>.csv   (per step samples)
      - occupancy_summary_<ts>.csv (metadata + per-cell accumulated time)
      - step_log_<ts>.csv        (actions)
    """

    def __init__(self, enabled=False, log_dir=None, grid_spec=None, arena_meta=None):
        self.enabled = enabled
        self.grid_spec = grid_spec
        self.arena_meta = arena_meta or {}

        self._occupancy_log = None
        self._step_log = None
        self._summary_path = None
        self._write_count = 0

        self.occupancy_time_s = None

        if not self.enabled:
            return

        if self.grid_spec is None:
            raise ValueError("ExperimentLogger(enabled=True) requires grid_spec.")

        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._occupancy_log_path = os.path.join(log_dir, f"occupancy_log_{timestamp}.csv")
        self._summary_path = os.path.join(log_dir, f"occupancy_summary_{timestamp}.csv")
        self._step_log_path = os.path.join(log_dir, f"step_log_{timestamp}.csv")

        self._occupancy_log = open(self._occupancy_log_path, "w", encoding="utf-8")
        self._occupancy_log.write("time_s,x_m,y_m,row,col,dt_s,inside\n")

        self._step_log = open(self._step_log_path, "w", encoding="utf-8")
        self._step_log.write("time_s,forward,sideways,yaw_rate,altitude,detections\n")

        rows = int(self.grid_spec.rows)
        cols = int(self.grid_spec.cols)
        self.occupancy_time_s = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def log_occupancy(self, time_s, x_m, y_m, dt_s):
        if not self.enabled:
            return None

        cell = self.grid_spec.rowcol_from_xy(x_m, y_m)
        inside = cell is not None

        if inside:
            row, col = cell
            self.occupancy_time_s[row][col] += dt_s
            self._occupancy_log.write(
                f"{time_s:.6f},{x_m:.4f},{y_m:.4f},{row},{col},{dt_s:.6f},1\n"
            )
        else:
            self._occupancy_log.write(
                f"{time_s:.6f},{x_m:.4f},{y_m:.4f},-1,-1,{dt_s:.6f},0\n"
            )

        self._write_count += 1
        if self._write_count % 20 == 0:
            self._occupancy_log.flush()

        return cell

    def log_step(self, *, time, forward, sideways, yaw_rate, altitude, detections):
        if not self.enabled:
            return

        self._step_log.write(
            f"{time:.6f},{forward:.4f},{sideways:.4f},{yaw_rate:.4f},{altitude:.4f},{int(detections)}\n"
        )

        if self._write_count % 20 == 0:
            self._step_log.flush()

    def close(self):
        if not self.enabled:
            return

        if self._step_log:
            self._step_log.close()
        if self._occupancy_log:
            self._occupancy_log.close()

        # Write summary (metadata + per-cell accumulated time)
        if self._summary_path and self.occupancy_time_s is not None:
            gs = self.grid_spec

            # Keep same header your heatmap expects
            arena_width = float(gs.max_x - gs.min_x)
            arena_height = float(gs.max_y - gs.min_y)

            with open(self._summary_path, "w", encoding="utf-8") as summary_file:
                summary_file.write(
                    "arena_width_m,arena_height_m,cell_size_m,min_x,max_x,min_y,max_y,rows,cols\n"
                )
                summary_file.write(
                    f"{arena_width},{arena_height},{gs.cell_size},"
                    f"{gs.min_x},{gs.max_x},{gs.min_y},{gs.max_y},{gs.rows},{gs.cols}\n"
                )
                summary_file.write("row,col,time_s\n")
                for row in range(gs.rows):
                    for col in range(gs.cols):
                        summary_file.write(f"{row},{col},{self.occupancy_time_s[row][col]:.6f}\n")
