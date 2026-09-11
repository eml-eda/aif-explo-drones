"""Surface-aware camera projection helpers for the AIF controller.

This module contains only geometry and representation helpers. It does not own
policy beliefs, detector scheduling, controller claims, or supervisor state.

Terminology
-----------
``persistent_table_cells``
    Cells that have accumulated perception-derived evidence of a tabletop and
    therefore persist across controller steps.

``transient_table_shadow_cells``
    Current-frame floor-plane counterparts of detected tabletop bbox probes.
    They are geometry corrections only and disappear on the next controller
    step; they never become positive table evidence by themselves.

``effective_table_projection_cells``
    The union used when selecting which surface footprint supplies *visual/FoV
    coverage* for the current controller step. This is distinct from the
    trajectory-based visited-cell coverage reported at the end of a run.
"""

from __future__ import annotations

import math
from typing import Iterable

from .env_manager import FoVSpec


def make_fov_spec(*, width_m: float, forward_m: float, backward_m: float) -> FoVSpec:
    return FoVSpec(
        half_left=float(width_m) / 2.0,
        half_right=float(width_m) / 2.0,
        forward=float(forward_m),
        backward=float(backward_m),
    )


def surface_fov_scale(
    *,
    camera_height_m: float,
    surface_height_m: float,
    reference_height_m: float,
) -> float:
    reference_height_m = max(1e-6, float(reference_height_m))
    distance_m = max(0.05, float(camera_height_m) - float(surface_height_m))
    return float(distance_m / reference_height_m)


def scaled_fov_spec_for_surface(
    surface_height_m: float,
    *,
    camera_height_m: float,
    reference_height_m: float,
    base_width_m: float,
    base_forward_m: float,
    base_backward_m: float,
) -> FoVSpec:
    """Scale a measured reference-plane FoV footprint to another surface."""
    scale = surface_fov_scale(
        camera_height_m=float(camera_height_m),
        surface_height_m=float(surface_height_m),
        reference_height_m=float(reference_height_m),
    )
    return make_fov_spec(
        width_m=float(base_width_m) * scale,
        forward_m=float(base_forward_m) * scale,
        backward_m=float(base_backward_m) * scale,
    )


def merge_visual_coverages_by_surface(
    *,
    floor_coverages_rc: dict,
    table_coverages_rc: dict,
    table_projection_cells_cxcy: set,
) -> dict:
    """Select floor- or table-plane visual coverage for every grid cell.

    ``floor_coverages_rc`` and ``table_coverages_rc`` use internal ``(row,col)``
    keys. ``table_projection_cells_cxcy`` uses policy/report ``(cx,cy)`` keys.
    A table-projection cell uses the smaller table-height camera footprint;
    every other cell uses the floor footprint.
    """
    out = {}
    all_cells = set(floor_coverages_rc.keys()) | set(table_coverages_rc.keys())
    for cell_rc in all_cells:
        r, c = int(cell_rc[0]), int(cell_rc[1])
        if (c, r) in table_projection_cells_cxcy:
            val = float(table_coverages_rc.get(cell_rc, 0.0))
        else:
            val = float(floor_coverages_rc.get(cell_rc, 0.0))
        if val > 0.0:
            out[(r, c)] = val
    return out


def effective_table_projection_cells(
    *,
    persistent_table_cells_cxcy: set,
    transient_table_shadow_cells_cxcy: set | None = None,
    enable_table_shadow_projection: bool = True,
) -> set:
    """Return cells that use table-height visual coverage this controller step."""
    effective = {
        (int(cx), int(cy))
        for (cx, cy) in (persistent_table_cells_cxcy or set())
    }
    if enable_table_shadow_projection:
        effective.update(
            (int(cx), int(cy))
            for (cx, cy) in (transient_table_shadow_cells_cxcy or set())
        )
    return effective


def surface_height_payload(
    persistent_table_cells_cxcy: set,
    table_confidence_cxcy=None,
    *,
    enabled: bool,
    floor_height_m: float,
    table_height_m: float,
) -> dict:
    """Serialize persistent geometric surface state for policy observations."""
    table_confidence_cxcy = table_confidence_cxcy or {}
    return {
        "enabled": bool(enabled),
        "floor_height_m": float(floor_height_m),
        "table_height_m": float(table_height_m),
        # Retain the established external key for compatibility. Internally
        # these are explicitly named persistent_table_cells.
        "table_cells": sorted(
            [[int(cx), int(cy)] for (cx, cy) in persistent_table_cells_cxcy]
        ),
        "height_by_cell": {
            f"{int(cx)},{int(cy)}": float(table_height_m)
            for (cx, cy) in sorted(persistent_table_cells_cxcy)
        },
        "table_confidence_by_cell": {
            f"{int(cx)},{int(cy)}": float(conf)
            for (cx, cy), conf in sorted(table_confidence_cxcy.items())
        },
    }


def clamp_bbox_xyxy(bbox_xyxy, *, img_w: int, img_h: int):
    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = max(0.0, min(float(img_w), x1))
    x2 = max(0.0, min(float(img_w), x2))
    y1 = max(0.0, min(float(img_h), y1))
    y2 = max(0.0, min(float(img_h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_area(bbox_xyxy) -> float:
    if bbox_xyxy is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_overlap_area(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _bbox_center_inside(inner_bbox, outer_bbox) -> bool:
    if inner_bbox is None or outer_bbox is None:
        return False
    x1, y1, x2, y2 = [float(v) for v in inner_bbox]
    ox1, oy1, ox2, oy2 = [float(v) for v in outer_bbox]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return (ox1 <= cx <= ox2) and (oy1 <= cy <= oy2)


def bbox_overlaps_any_desk(
    target_bbox,
    desk_bboxes: Iterable,
    *,
    min_overlap_frac: float = 0.05,
) -> bool:
    target_area = max(1e-6, _bbox_area(target_bbox))
    for desk_bbox in desk_bboxes:
        if _bbox_center_inside(target_bbox, desk_bbox):
            return True
        if (_bbox_overlap_area(target_bbox, desk_bbox) / target_area) >= float(min_overlap_frac):
            return True
    return False


def event_bbox_xyxy(event: dict):
    bbox = event.get("bbox_xyxy", None)
    if bbox is not None and len(bbox) == 4:
        return [float(v) for v in bbox]

    center = event.get("bbox_center_px", None)
    size = event.get("bbox_size_px", None)
    if center is not None and size is not None and len(center) >= 2 and len(size) >= 2:
        cx, cy = float(center[0]), float(center[1])
        w, h = float(size[0]), float(size[1])
        return [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]

    return None


def _bbox_probe_grid(bbox_xyxy, *, img_w: int, img_h: int, max_samples_per_axis: int):
    """Yield the exact deterministic probe bboxes used by the pre-extraction controller."""
    bbox = clamp_bbox_xyxy(bbox_xyxy, img_w=img_w, img_h=img_h)
    if bbox is None:
        return None, []

    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    nx = int(max(2, min(max_samples_per_axis, round(w / 32.0) + 1)))
    ny = int(max(2, min(max_samples_per_axis, round(h / 32.0) + 1)))

    probes = []
    for ix in range(nx):
        px = x1 + ((ix + 0.5) / float(nx)) * w
        for iy in range(ny):
            py = y1 + ((iy + 0.5) / float(ny)) * h
            probes.append([px - 0.5, py - 0.5, px + 0.5, py + 0.5])
    return bbox, probes


def sample_bbox_projected_cells_rc(
    env_mgr_surface,
    *,
    bbox_xyxy,
    img_w: int,
    img_h: int,
    drone_x: float,
    drone_y: float,
    drone_yaw: float,
    max_samples_per_axis: int = 5,
) -> set:
    """Project deterministic bbox probes to all grid cells they hit."""
    bbox, probes = _bbox_probe_grid(
        bbox_xyxy,
        img_w=img_w,
        img_h=img_h,
        max_samples_per_axis=max_samples_per_axis,
    )
    if bbox is None:
        return set()

    cells = set()
    for probe_bbox in probes:
        cell_rc = env_mgr_surface.detection_cell_from_bbox(
            bbox_xyxy=probe_bbox,
            img_w=img_w,
            img_h=img_h,
            drone_x=drone_x,
            drone_y=drone_y,
            drone_yaw=drone_yaw,
        )
        if cell_rc is not None:
            cells.add((int(cell_rc[0]), int(cell_rc[1])))

    center_cell = env_mgr_surface.detection_cell_from_bbox(
        bbox_xyxy=bbox,
        img_w=img_w,
        img_h=img_h,
        drone_x=drone_x,
        drone_y=drone_y,
        drone_yaw=drone_yaw,
    )
    if center_cell is not None:
        cells.add((int(center_cell[0]), int(center_cell[1])))
    return cells


def cell_center_xy_from_rc(
    cell_rc,
    *,
    min_x: float,
    min_y: float,
    cell_size_m: float,
) -> tuple[float, float]:
    row, col = int(cell_rc[0]), int(cell_rc[1])
    return (
        float(min_x) + (float(col) + 0.5) * float(cell_size_m),
        float(min_y) + (float(row) + 0.5) * float(cell_size_m),
    )


def forward_distance_m_for_cell_rc(
    cell_rc,
    *,
    drone_x: float,
    drone_y: float,
    drone_yaw: float,
    min_x: float,
    min_y: float,
    cell_size_m: float,
) -> float:
    cell_x, cell_y = cell_center_xy_from_rc(
        cell_rc,
        min_x=min_x,
        min_y=min_y,
        cell_size_m=cell_size_m,
    )
    dx = float(cell_x) - float(drone_x)
    dy = float(cell_y) - float(drone_y)
    return float(dx * math.cos(float(drone_yaw)) + dy * math.sin(float(drone_yaw)))


def sample_front_only_bbox_floor_counterparts_rc(
    env_mgr_table,
    env_mgr_floor,
    *,
    bbox_xyxy,
    img_w: int,
    img_h: int,
    drone_x: float,
    drone_y: float,
    drone_yaw: float,
    min_x: float,
    min_y: float,
    cell_size_m: float,
    forward_tolerance_m: float,
    max_samples_per_axis: int = 5,
) -> set:
    """Return the legacy forward-only floor counterparts for a desk bbox."""
    bbox, probes = _bbox_probe_grid(
        bbox_xyxy,
        img_w=img_w,
        img_h=img_h,
        max_samples_per_axis=max_samples_per_axis,
    )
    if bbox is None:
        return set()

    counterparts = set()

    def try_add_pair(probe_bbox) -> None:
        table_rc = env_mgr_table.detection_cell_from_bbox(
            bbox_xyxy=probe_bbox,
            img_w=img_w,
            img_h=img_h,
            drone_x=drone_x,
            drone_y=drone_y,
            drone_yaw=drone_yaw,
        )
        floor_rc = env_mgr_floor.detection_cell_from_bbox(
            bbox_xyxy=probe_bbox,
            img_w=img_w,
            img_h=img_h,
            drone_x=drone_x,
            drone_y=drone_y,
            drone_yaw=drone_yaw,
        )
        if table_rc is None or floor_rc is None:
            return

        table_rc = (int(table_rc[0]), int(table_rc[1]))
        floor_rc = (int(floor_rc[0]), int(floor_rc[1]))
        table_forward_m = forward_distance_m_for_cell_rc(
            table_rc,
            drone_x=drone_x,
            drone_y=drone_y,
            drone_yaw=drone_yaw,
            min_x=min_x,
            min_y=min_y,
            cell_size_m=cell_size_m,
        )
        floor_forward_m = forward_distance_m_for_cell_rc(
            floor_rc,
            drone_x=drone_x,
            drone_y=drone_y,
            drone_yaw=drone_yaw,
            min_x=min_x,
            min_y=min_y,
            cell_size_m=cell_size_m,
        )

        tolerance = float(forward_tolerance_m)
        if table_forward_m < -tolerance:
            return
        if floor_forward_m < -tolerance:
            return
        if floor_forward_m < (table_forward_m - tolerance):
            return
        counterparts.add(floor_rc)

    for probe_bbox in probes:
        try_add_pair(probe_bbox)
    try_add_pair(bbox)
    return counterparts
