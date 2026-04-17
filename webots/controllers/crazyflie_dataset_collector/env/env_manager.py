# env/env_manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import math

Cell = Tuple[int, int]  # (row, col)
Point = Tuple[float, float]
Pixel = Tuple[float, float]

@dataclass(frozen=True)
class GridSpec:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    cell_size: float
    rows: int
    cols: int

    def cell_bounds(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """Returns (x0, x1, y0, y1) bounds of a cell square in world coords."""
        x0 = self.min_x + col * self.cell_size
        x1 = x0 + self.cell_size
        y0 = self.min_y + row * self.cell_size
        y1 = y0 + self.cell_size
        return x0, x1, y0, y1

    def clamp_rowcol(self, row: int, col: int) -> Optional[Cell]:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return None

    def rowcol_from_xy(self, x: float, y: float) -> Optional[Cell]:
        """Which cell contains (x,y). Returns None if outside grid."""
        if x < self.min_x or x >= self.max_x or y < self.min_y or y >= self.max_y:
            return None
        col = int((x - self.min_x) / self.cell_size)
        row = int((y - self.min_y) / self.cell_size)
        return self.clamp_rowcol(row, col)


@dataclass(frozen=True)
class FoVSpec:
    half_left: float   # meters left of drone center (body frame)
    half_right: float  # meters right of drone center (body frame)
    forward: float     # meters forward of drone center (body frame)
    backward: float    # meters backward of drone center (body frame)

    @staticmethod
    def from_width_height_asym(width_m: float, forward_m: float, backward_m: float) -> "FoVSpec":
        half = width_m / 2.0
        return FoVSpec(half_left=half, half_right=half, forward=forward_m, backward=backward_m)


class GridEnvManager:
    """
    Computes FoV overlap with grid cells.

    Convention:
    - World frame: (x,y) from GPS, yaw in radians.
    - Drone body frame for footprint before rotation:
        +x_body = forward direction of drone
        +y_body = left direction of drone (common robotics convention)
      If your convention differs, swap signs accordingly.
    """

    def __init__(self, grid: GridSpec, fov: FoVSpec):
        self.grid = grid
        self.fov = fov
        self.cell_area = grid.cell_size * grid.cell_size

    # ---------- Public API ----------

    def compute_cell_coverages(self, x: float, y: float, yaw: float) -> Dict[Cell, float]:
        """
        Returns dict {(row,col): coverage_fraction} for cells intersecting FoV.
        coverage_fraction in [0,1] = (area of cell inside FoV) / (cell_area).
        """
        fov_poly = self._fov_polygon_world(x, y, yaw)  # list of 4 points
        minx, maxx, miny, maxy = self._aabb(fov_poly)

        # Candidate cell range (clamped)
        c0 = int(math.floor((minx - self.grid.min_x) / self.grid.cell_size))
        c1 = int(math.floor((maxx - self.grid.min_x) / self.grid.cell_size))
        r0 = int(math.floor((miny - self.grid.min_y) / self.grid.cell_size))
        r1 = int(math.floor((maxy - self.grid.min_y) / self.grid.cell_size))

        c0 = max(0, c0); r0 = max(0, r0)
        c1 = min(self.grid.cols - 1, c1); r1 = min(self.grid.rows - 1, r1)

        coverages: Dict[Cell, float] = {}

        # For each candidate cell, compute intersection area with FoV polygon
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                cell_poly = self._cell_polygon(row, col)
                inter_poly = self._convex_clip(subject=fov_poly, clipper=cell_poly)
                if not inter_poly:
                    continue
                area = abs(self._polygon_area(inter_poly))
                if area <= 0.0:
                    continue
                frac = min(1.0, max(0.0, area / self.cell_area))
                if frac > 0.0:
                    coverages[(row, col)] = frac

        return coverages
    
    def detection_cell_from_bbox(
        self,
        bbox_xyxy,
        img_w: int,
        img_h: int,
        drone_x: float,
        drone_y: float,
        drone_yaw: float,
    ) -> Optional[Cell]:
        """
        Map bbox center pixel -> body-frame point inside FoV footprint -> world -> (row,col).
        Simple linear mapping across the footprint rectangle.

        bbox_xyxy is [x1,y1,x2,y2] in pixel coords.
        """
        if img_w is None or img_h is None or img_w <= 0 or img_h <= 0:
            return None
        if bbox_xyxy is None or len(bbox_xyxy) != 4:
            return None

        x1, y1, x2, y2 = bbox_xyxy
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)

        # Normalize to [0,1]
        un = u / float(img_w)
        vn = v / float(img_h)

        # Pixel x (u): left->right maps to body y: +left to -right
        # So un=0 => +half_left, un=1 => -half_right
        y_body = (1.0 - un) * self.fov.half_left + un * (-self.fov.half_right)

        # Pixel y (v): top->bottom maps to body x: +forward to -backward
        # vn=0 => +forward, vn=1 => -backward
        x_body = (1.0 - vn) * self.fov.forward + vn * (-self.fov.backward)

        # Convert body -> world
        wx, wy = self._rot_trans((x_body, y_body), drone_yaw, drone_x, drone_y)
        return self.grid.rowcol_from_xy(wx, wy)


    @staticmethod
    def coverage_to_visible_mask(coverages: Dict[Cell, float], threshold: float) -> Dict[Cell, bool]:
        """Turns a coverage dict into a boolean mask dict using threshold."""
        return {cell: (cov >= threshold) for cell, cov in coverages.items()}

    # ---------- Geometry ----------

    def _fov_polygon_world(self, x: float, y: float, yaw: float) -> List[Point]:
        """
        FoV rectangle corners in world frame.
        Body-frame rectangle (counterclockwise):
          front-left, front-right, back-right, back-left
        """
        # Body-frame corners
        fl = ( self.fov.forward,  self.fov.half_left)
        fr = ( self.fov.forward, -self.fov.half_right)
        br = (-self.fov.backward, -self.fov.half_right)
        bl = (-self.fov.backward,  self.fov.half_left)

        return [self._rot_trans(p, yaw, x, y) for p in (fl, fr, br, bl)]

    @staticmethod
    def _rot_trans(p: Point, yaw: float, tx: float, ty: float) -> Point:
        """Rotate point p (in body frame) by yaw and translate by (tx,ty)."""
        px, py = p
        c = math.cos(yaw)
        s = math.sin(yaw)
        wx = tx + c * px - s * py
        wy = ty + s * px + c * py
        return (wx, wy)

    def _cell_polygon(self, row: int, col: int) -> List[Point]:
        """Axis-aligned cell square as convex polygon CCW."""
        x0, x1, y0, y1 = self.grid.cell_bounds(row, col)
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    @staticmethod
    def _aabb(poly: List[Point]) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return min(xs), max(xs), min(ys), max(ys)

    @staticmethod
    def _polygon_area(poly: List[Point]) -> float:
        """Shoelace formula; poly assumed closed implicitly."""
        if len(poly) < 3:
            return 0.0
        area = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        return 0.5 * area

    # Sutherland–Hodgman polygon clipping for convex polygons
    @staticmethod
    def _convex_clip(subject: List[Point], clipper: List[Point]) -> List[Point]:
        """
        Clip 'subject' polygon by 'clipper' polygon (both convex, CCW).
        Returns intersection polygon (convex), possibly empty.
        """
        def inside(p: Point, a: Point, b: Point) -> bool:
            # is p to the left of directed edge a->b ?
            return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0.0

        def intersection(p1: Point, p2: Point, a: Point, b: Point) -> Point:
            # line p1->p2 intersect line a->b
            x1, y1 = p1; x2, y2 = p2
            x3, y3 = a;  x4, y4 = b
            den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(den) < 1e-12:
                return p2  # nearly parallel; fallback
            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
            return (px, py)

        output = subject[:]
        for i in range(len(clipper)):
            a = clipper[i]
            b = clipper[(i + 1) % len(clipper)]
            input_list = output
            output = []
            if not input_list:
                break
            s = input_list[-1]
            for e in input_list:
                if inside(e, a, b):
                    if not inside(s, a, b):
                        output.append(intersection(s, e, a, b))
                    output.append(e)
                elif inside(s, a, b):
                    output.append(intersection(s, e, a, b))
                s = e
        return output
