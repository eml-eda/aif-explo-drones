# utils/navigator.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

Cell = Tuple[int, int]  # (cx, cy)


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Navigator:
    """
    Converts (target_cell, target_yaw) into continuous commands.
    Rule: yaw first, then translate.

    Arrival condition is handled via:
      - coverage(target_cell) >= min_cell_coverage
      - distance to target cell center <= dist_eps_m
      - |yaw_err| <= yaw_eps_rad
    """
    grid_spec: object  # expects GridSpec-like: min_x, min_y, cell_size, rows, cols

    max_forward: float = 0.30
    max_sideways: float = 0.30
    max_yaw_rate: float = 0.8

    k_pos: float = 1.2
    k_yaw: float = 1.5

    dist_eps_m: float = 0.2
    yaw_eps_rad: float = 0.12
    min_cell_coverage: float = 0.60

    target_cell: Optional[Cell] = None
    target_yaw: Optional[float] = None

    def set_target(self, cell: Cell, yaw: float) -> None:
        self.target_cell = cell
        self.target_yaw = float(yaw)

    def clear_target(self) -> None:
        self.target_cell = None
        self.target_yaw = None

    def has_target(self) -> bool:
        return (self.target_cell is not None) and (self.target_yaw is not None)

    def cell_center_xy(self, cx: int, cy: int) -> Tuple[float, float]:
        # cx -> col, cy -> row
        x0 = float(self.grid_spec.min_x) + cx * float(self.grid_spec.cell_size)
        y0 = float(self.grid_spec.min_y) + cy * float(self.grid_spec.cell_size)
        cs = float(self.grid_spec.cell_size)
        return (x0 + 0.5 * cs, y0 + 0.5 * cs)

    def arrived(self, pose: Dict, coverages_cxcy: Dict[Cell, float]) -> bool:
        if not self.has_target():
            return True

        x = float(pose["x"])
        y = float(pose["y"])
        yaw = float(pose["yaw"])

        cx, cy = self.target_cell
        tx, ty = self.cell_center_xy(cx, cy)

        dist = math.hypot(tx - x, ty - y)
        yaw_err = abs(wrap_to_pi(float(self.target_yaw) - yaw))
        cov = float(coverages_cxcy.get((cx, cy), 0.0))

        return (cov >= self.min_cell_coverage) and (dist <= self.dist_eps_m) and (yaw_err <= self.yaw_eps_rad)

    def compute_command(self, pose: Dict) -> Dict[str, float]:
        if not self.has_target():
            return {"forward": 0.0, "sideways": 0.0, "yaw_rate": 0.0}

        x = float(pose["x"])
        y = float(pose["y"])
        yaw = float(pose["yaw"])

        # 1) yaw first
        yaw_err = wrap_to_pi(float(self.target_yaw) - yaw)
        if abs(yaw_err) > self.yaw_eps_rad:
            yaw_rate = clamp(self.k_yaw * yaw_err, -self.max_yaw_rate, self.max_yaw_rate)
            return {"forward": 0.0, "sideways": 0.0, "yaw_rate": yaw_rate}

        # 2) translate toward cell center (command in BODY frame)
        cx, cy = self.target_cell
        tx, ty = self.cell_center_xy(cx, cy)

        dx_w = tx - x
        dy_w = ty - y

        c = math.cos(yaw)
        s = math.sin(yaw)

        # world -> body
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w

        fwd = clamp(self.k_pos * dx_b, -self.max_forward, self.max_forward)
        side = clamp(self.k_pos * dy_b, -self.max_sideways, self.max_sideways)

        return {"forward": fwd, "sideways": side, "yaw_rate": 0.0}