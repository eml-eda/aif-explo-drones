# vision/ground_truth_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from env.env_manager import GridSpec, FoVSpec, GridEnvManager, Cell, Point


@dataclass
class GroundTruthDetector:
    """
    Ground-truth detector that returns detections directly in grid-cell space.

    Inputs:
      - object_positions_xy: list of (x,y) world coordinates for spawned objects
      - grid_spec, fov_spec: used to project FoV footprint and map points to cells

    Output:
      - det_conf_rc: Dict[(row,col), confidence]
    """
    object_positions_xy: List[Tuple[float, float]]
    grid_spec: GridSpec
    fov_spec: FoVSpec
    confidence: float = 1.0  # returned conf for visible objects

    def __post_init__(self) -> None:
        self.env_mgr = GridEnvManager(self.grid_spec, self.fov_spec)

    def process(
        self,
        *,
        drone_x: float,
        drone_y: float,
        drone_yaw: float,
    ) -> Dict[Cell, float]:
        """
        Returns dict {(row,col): conf} for objects inside the FoV polygon.
        If multiple objects fall in the same cell, keeps the max conf.
        """
        fov_poly = self.env_mgr._fov_polygon_world(drone_x, drone_y, drone_yaw)

        # IMPORTANT: point-in-convex-polygon assumes CCW order.
        # _fov_polygon_world currently returns corners in clockwise order (fl, fr, br, bl),
        # so we flip if needed.
        if self.env_mgr._polygon_area(fov_poly) < 0.0:
            fov_poly = list(reversed(fov_poly))

        det_conf_rc: Dict[Cell, float] = {}
        for (ox, oy) in self.object_positions_xy:
            if not self._point_in_convex_polygon((ox, oy), fov_poly):
                continue

            cell_rc = self.grid_spec.rowcol_from_xy(ox, oy)
            if cell_rc is None:
                continue

            prev = det_conf_rc.get(cell_rc, 0.0)
            if self.confidence > prev:
                det_conf_rc[cell_rc] = float(self.confidence)

        return det_conf_rc

    @staticmethod
    def _point_in_convex_polygon(p: Point, poly_ccw: List[Point]) -> bool:
        """
        Check if point p is inside a convex polygon given CCW vertices.
        Uses left-of-edge test; allows points on the boundary.
        """
        if not poly_ccw or len(poly_ccw) < 3:
            return False

        px, py = p
        for i in range(len(poly_ccw)):
            ax, ay = poly_ccw[i]
            bx, by = poly_ccw[(i + 1) % len(poly_ccw)]
            # cross((b-a),(p-a)) >= 0  => p is to the left of edge a->b
            if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < 0.0:
                return False
        return True