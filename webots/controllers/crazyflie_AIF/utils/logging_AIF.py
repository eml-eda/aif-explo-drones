# utils/logging_AIF

import os
from datetime import datetime


class ExperimentLogger:
    """
    Depends only on grid_spec.
    Writes:
      - occupancy_log_<ts>.csv    (per step samples)
      - occupancy_summary_<ts>.csv (metadata + per-cell accumulated time)
      - step_log_<ts>.csv         (actions)

    New behavior:
      - samples with time_s < min_time_s are ignored
      - this is useful for excluding warm-up / takeoff from the occupancy heatmap
    """

    def __init__(self, enabled=False, log_dir=None, grid_spec=None, arena_meta=None, min_time_s=0.0):
        self.enabled = enabled
        self.grid_spec = grid_spec
        self.arena_meta = arena_meta or {}
        self.min_time_s = float(min_time_s)

        self.log_dir = log_dir
        self.occupancy_log_path = None
        self.occupancy_summary_path = None
        self.step_log_path = None

        self._occupancy_write_count = 0
        self._step_write_count = 0

        self.occupancy_time_s = None

        if not self.enabled:
            return

        if self.grid_spec is None:
            raise ValueError("ExperimentLogger(enabled=True) requires grid_spec.")

        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.occupancy_log_path = os.path.join(log_dir, f"occupancy_log_{timestamp}.csv")
        self.occupancy_summary_path = os.path.join(log_dir, f"occupancy_summary_{timestamp}.csv")
        self.step_log_path = os.path.join(log_dir, f"step_log_{timestamp}.csv")

        self._occupancy_log_path = self.occupancy_log_path
        self._summary_path = self.occupancy_summary_path
        self._step_log_path = self.step_log_path

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

        time_s = float(time_s)
        if time_s < self.min_time_s:
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

        self._occupancy_write_count += 1
        if self._occupancy_write_count % 20 == 0:
            self._occupancy_log.flush()

        return cell

    def log_step(self, *, time, forward, sideways, yaw_rate, altitude, detections):
        if not self.enabled:
            return

        time = float(time)
        if time < self.min_time_s:
            return

        self._step_log.write(
            f"{time:.6f},{forward:.4f},{sideways:.4f},{yaw_rate:.4f},{altitude:.4f},{int(detections)}\n"
        )

        self._step_write_count += 1
        if self._step_write_count % 20 == 0:
            self._step_log.flush()
    
    def paths(self):
        return {
            "occupancy_log": self.occupancy_log_path,
            "occupancy_summary": self.occupancy_summary_path,
            "step_log": self.step_log_path,
        }

    def close(self):
        if not self.enabled:
            return self.paths()

        if self._step_log:
            self._step_log.close()
        if self._occupancy_log:
            self._occupancy_log.close()

        if self._summary_path and self.occupancy_time_s is not None:
            gs = self.grid_spec

            arena_width = float(gs.max_x - gs.min_x)
            arena_height = float(gs.max_y - gs.min_y)

            with open(self._summary_path, "w", encoding="utf-8") as summary_file:
                summary_file.write(
                    "arena_width_m,arena_height_m,cell_size_m,min_x,max_x,min_y,max_y,rows,cols,min_time_s\n"
                )
                summary_file.write(
                    f"{arena_width},{arena_height},{gs.cell_size},"
                    f"{gs.min_x},{gs.max_x},{gs.min_y},{gs.max_y},{gs.rows},{gs.cols},{self.min_time_s}\n"
                )
                summary_file.write("row,col,time_s\n")
                for row in range(gs.rows):
                    for col in range(gs.cols):
                        summary_file.write(f"{row},{col},{self.occupancy_time_s[row][col]:.6f}\n")

        return self.paths()