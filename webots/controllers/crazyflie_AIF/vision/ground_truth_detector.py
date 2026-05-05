# vision/ground_truth_detector.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from env.env_manager import GridSpec, FoVSpec, GridEnvManager, Cell, Point


@dataclass
class GroundTruthDetector:
    """
    Ground-truth detector that returns detections directly in grid-cell space
    together with class metadata.

    Inputs:
      - objects: list of dicts with keys:
          x, y, class_id, class_name
      - grid_spec, fov_spec: used to project FoV footprint and map points to cells

    Output:
      - Dict[(row,col)] = {
            "confidence": float,
            "class_id": int,
            "label": str,        # backward compatibility
            "class_name": str,   # preferred canonical field
        }
    """
    objects: List[dict]
    grid_spec: GridSpec
    fov_spec: FoVSpec
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.env_mgr = GridEnvManager(self.grid_spec, self.fov_spec)

    def process(
        self,
        *,
        drone_x: float,
        drone_y: float,
        drone_yaw: float,
    ) -> Dict[Cell, dict]:
        fov_poly = self.env_mgr._fov_polygon_world(drone_x, drone_y, drone_yaw)

        if self.env_mgr._polygon_area(fov_poly) < 0.0:
            fov_poly = list(reversed(fov_poly))

        det_out: Dict[Cell, dict] = {}
        for o in self.objects:
            ox = float(o["x"])
            oy = float(o["y"])
            class_id = int(o["class_id"])
            class_name = str(o["class_name"])

            if not self._point_in_convex_polygon((ox, oy), fov_poly):
                continue

            cell_rc = self.grid_spec.rowcol_from_xy(ox, oy)
            if cell_rc is None:
                continue

            prev = det_out.get(cell_rc, None)
            if (prev is None) or (self.confidence > float(prev["confidence"])):
                det_out[cell_rc] = {
                    "confidence": float(self.confidence),
                    "class_id": int(class_id),
                    "label": str(class_name),
                    "class_name": str(class_name),
                }

        return det_out

    @staticmethod
    def _point_in_convex_polygon(p: Point, poly_ccw: List[Point]) -> bool:
        if not poly_ccw or len(poly_ccw) < 3:
            return False

        px, py = p
        for i in range(len(poly_ccw)):
            ax, ay = poly_ccw[i]
            bx, by = poly_ccw[(i + 1) % len(poly_ccw)]
            if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < 0.0:
                return False
        return True