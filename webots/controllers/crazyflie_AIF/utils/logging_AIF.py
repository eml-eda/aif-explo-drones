import os
from datetime import datetime


class ExperimentLogger:
    def __init__(self, enabled=False, log_dir=None, grid=None):
        self.enabled = enabled
        self.grid = grid or {}
        self.occupancy_time_s = None
        self._occupancy_log = None
        self._step_log = None
        self._summary_path = None

        if not self.enabled:
            return

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

        rows = self.grid.get("rows")
        cols = self.grid.get("cols")
        if rows is not None and cols is not None:
            self.occupancy_time_s = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def _cell_from_position(self, x, y):
        min_x = self.grid.get("min_x")
        max_x = self.grid.get("max_x")
        min_y = self.grid.get("min_y")
        max_y = self.grid.get("max_y")
        cell_size = self.grid.get("cell_size")
        rows = self.grid.get("rows")
        cols = self.grid.get("cols")
        if None in (min_x, max_x, min_y, max_y, cell_size, rows, cols):
            return None
        if x < min_x or x >= max_x or y < min_y or y >= max_y:
            return None
        col = int((x - min_x) / cell_size)
        row = int((y - min_y) / cell_size)
        if col < 0 or col >= cols or row < 0 or row >= rows:
            return None
        return row, col

    def log_occupancy(self, time_s, x_m, y_m, dt_s):
        if not self.enabled:
            return None

        cell = self._cell_from_position(x_m, y_m)
        inside = cell is not None
        if inside and self.occupancy_time_s is not None:
            row, col = cell
            self.occupancy_time_s[row][col] += dt_s

        if inside:
            row, col = cell
            self._occupancy_log.write(
                f"{time_s:.6f},{x_m:.4f},{y_m:.4f},{row},{col},{dt_s:.6f},1\n"
            )
        else:
            self._occupancy_log.write(
                f"{time_s:.6f},{x_m:.4f},{y_m:.4f},-1,-1,{dt_s:.6f},0\n"
            )
        return cell

    def log_step(self, **data):
        if not self.enabled:
            return
        time_s = data.get("time", 0.0)
        forward = data.get("forward", 0.0)
        sideways = data.get("sideways", 0.0)
        yaw_rate = data.get("yaw_rate", 0.0)
        altitude = data.get("altitude", 0.0)
        detections = data.get("detections", 0)
        self._step_log.write(
            f"{time_s:.6f},{forward:.4f},{sideways:.4f},{yaw_rate:.4f},{altitude:.4f},{detections}\n"
        )
        self._write_count = getattr(self, "_write_count", 0) + 1
        if self._write_count % 20 == 0:
            self._occupancy_log.flush()
            self._step_log.flush()


    def close(self):
        if not self.enabled:
            return
        if self._step_log:
            self._step_log.close()
        if self._occupancy_log:
            self._occupancy_log.close()

        if self._summary_path and self.occupancy_time_s is not None:
            with open(self._summary_path, "w", encoding="utf-8") as summary_file:
                summary_file.write(
                    "arena_width_m,arena_height_m,cell_size_m,min_x,max_x,min_y,max_y,rows,cols\n"
                )
                summary_file.write(
                    f"{self.grid.get('arena_width')},{self.grid.get('arena_height')},"
                    f"{self.grid.get('cell_size')},{self.grid.get('min_x')},{self.grid.get('max_x')},"
                    f"{self.grid.get('min_y')},{self.grid.get('max_y')},{self.grid.get('rows')},"
                    f"{self.grid.get('cols')}\n"
                )
                summary_file.write("row,col,time_s\n")
                for row in range(self.grid.get("rows", 0)):
                    for col in range(self.grid.get("cols", 0)):
                        summary_file.write(f"{row},{col},{self.occupancy_time_s[row][col]:.6f}\n")
