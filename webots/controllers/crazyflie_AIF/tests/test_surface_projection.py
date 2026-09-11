from __future__ import annotations

import math
import unittest

from env.env_manager import GridEnvManager, GridSpec
from env.surface_projection import (
    bbox_overlaps_any_desk,
    effective_table_projection_cells,
    make_fov_spec,
    merge_visual_coverages_by_surface,
    sample_bbox_projected_cells_rc,
    sample_front_only_bbox_floor_counterparts_rc,
    scaled_fov_spec_for_surface,
    surface_height_payload,
)


MIN_X = -2.75
MIN_Y = -3.25
CELL_SIZE_M = 0.5


def _legacy_clamp_bbox_xyxy(bbox_xyxy, *, img_w: int, img_h: int):
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


def _legacy_sample_bbox_projected_cells_rc(
    env_mgr_surface,
    *,
    bbox_xyxy,
    img_w,
    img_h,
    drone_x,
    drone_y,
    drone_yaw,
    max_samples_per_axis=5,
):
    bbox = _legacy_clamp_bbox_xyxy(bbox_xyxy, img_w=img_w, img_h=img_h)
    if bbox is None:
        return set()
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    nx = int(max(2, min(max_samples_per_axis, round(w / 32.0) + 1)))
    ny = int(max(2, min(max_samples_per_axis, round(h / 32.0) + 1)))
    cells = set()
    for ix in range(nx):
        px = x1 + ((ix + 0.5) / float(nx)) * w
        for iy in range(ny):
            py = y1 + ((iy + 0.5) / float(ny)) * h
            tiny_bbox = [px - 0.5, py - 0.5, px + 0.5, py + 0.5]
            cell_rc = env_mgr_surface.detection_cell_from_bbox(
                bbox_xyxy=tiny_bbox,
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


def _legacy_forward_distance(cell_rc, *, drone_x, drone_y, drone_yaw):
    row, col = int(cell_rc[0]), int(cell_rc[1])
    cell_x = MIN_X + (col + 0.5) * CELL_SIZE_M
    cell_y = MIN_Y + (row + 0.5) * CELL_SIZE_M
    dx = cell_x - drone_x
    dy = cell_y - drone_y
    return dx * math.cos(drone_yaw) + dy * math.sin(drone_yaw)


def _legacy_front_counterparts(
    env_mgr_table,
    env_mgr_floor,
    *,
    bbox_xyxy,
    img_w,
    img_h,
    drone_x,
    drone_y,
    drone_yaw,
    max_samples_per_axis=5,
):
    bbox = _legacy_clamp_bbox_xyxy(bbox_xyxy, img_w=img_w, img_h=img_h)
    if bbox is None:
        return set()
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    nx = int(max(2, min(max_samples_per_axis, round(w / 32.0) + 1)))
    ny = int(max(2, min(max_samples_per_axis, round(h / 32.0) + 1)))
    counterparts = set()

    def try_add(probe_bbox):
        table_rc = env_mgr_table.detection_cell_from_bbox(
            bbox_xyxy=probe_bbox, img_w=img_w, img_h=img_h,
            drone_x=drone_x, drone_y=drone_y, drone_yaw=drone_yaw,
        )
        floor_rc = env_mgr_floor.detection_cell_from_bbox(
            bbox_xyxy=probe_bbox, img_w=img_w, img_h=img_h,
            drone_x=drone_x, drone_y=drone_y, drone_yaw=drone_yaw,
        )
        if table_rc is None or floor_rc is None:
            return
        table_rc = (int(table_rc[0]), int(table_rc[1]))
        floor_rc = (int(floor_rc[0]), int(floor_rc[1]))
        table_forward = _legacy_forward_distance(
            table_rc, drone_x=drone_x, drone_y=drone_y, drone_yaw=drone_yaw
        )
        floor_forward = _legacy_forward_distance(
            floor_rc, drone_x=drone_x, drone_y=drone_y, drone_yaw=drone_yaw
        )
        tol = 0.10
        if table_forward < -tol or floor_forward < -tol:
            return
        if floor_forward < (table_forward - tol):
            return
        counterparts.add(floor_rc)

    for ix in range(nx):
        px = x1 + ((ix + 0.5) / float(nx)) * w
        for iy in range(ny):
            py = y1 + ((iy + 0.5) / float(ny)) * h
            try_add([px - 0.5, py - 0.5, px + 0.5, py + 0.5])
    try_add(bbox)
    return counterparts


class SurfaceProjectionTests(unittest.TestCase):
    def setUp(self):
        grid = GridSpec(
            min_x=-2.75, max_x=2.75,
            min_y=-3.25, max_y=3.25,
            cell_size=0.5, rows=13, cols=11,
        )
        floor_fov = make_fov_spec(width_m=1.394, forward_m=0.553, backward_m=0.493)
        table_fov = scaled_fov_spec_for_surface(
            0.72,
            camera_height_m=1.5,
            reference_height_m=1.5,
            base_width_m=1.394,
            base_forward_m=0.553,
            base_backward_m=0.493,
        )
        self.floor = GridEnvManager(grid, floor_fov)
        self.table = GridEnvManager(grid, table_fov)

    def test_table_fov_scaling_matches_pre_extraction_math(self):
        fov = self.table.fov
        self.assertAlmostEqual(fov.half_left + fov.half_right, 1.394 * (0.78 / 1.5), places=12)
        self.assertAlmostEqual(fov.forward, 0.553 * (0.78 / 1.5), places=12)
        self.assertAlmostEqual(fov.backward, 0.493 * (0.78 / 1.5), places=12)

    def test_bbox_probe_projection_matches_legacy_formula(self):
        cases = [
            ([40, 30, 100, 80], -1.0, 1.0, 0.0),
            ([80, 60, 260, 210], 0.3, -0.7, 0.65),
            ([-20, -10, 350, 260], 1.4, 1.8, -1.1),
        ]
        for bbox, x, y, yaw in cases:
            with self.subTest(bbox=bbox, pose=(x, y, yaw)):
                old = _legacy_sample_bbox_projected_cells_rc(
                    self.table, bbox_xyxy=bbox, img_w=320, img_h=240,
                    drone_x=x, drone_y=y, drone_yaw=yaw,
                )
                new = sample_bbox_projected_cells_rc(
                    self.table, bbox_xyxy=bbox, img_w=320, img_h=240,
                    drone_x=x, drone_y=y, drone_yaw=yaw,
                )
                self.assertEqual(new, old)

    def test_front_shadow_counterparts_match_legacy_formula(self):
        cases = [
            ([40, 20, 280, 150], 0.0, 0.0, 0.0),
            ([90, 70, 260, 220], -1.2, 0.8, 0.8),
            ([10, 10, 310, 230], 1.0, -1.5, -0.7),
        ]
        for bbox, x, y, yaw in cases:
            with self.subTest(bbox=bbox, pose=(x, y, yaw)):
                old = _legacy_front_counterparts(
                    self.table, self.floor,
                    bbox_xyxy=bbox, img_w=320, img_h=240,
                    drone_x=x, drone_y=y, drone_yaw=yaw,
                )
                new = sample_front_only_bbox_floor_counterparts_rc(
                    self.table, self.floor,
                    bbox_xyxy=bbox, img_w=320, img_h=240,
                    drone_x=x, drone_y=y, drone_yaw=yaw,
                    min_x=MIN_X, min_y=MIN_Y, cell_size_m=CELL_SIZE_M,
                    forward_tolerance_m=0.10,
                )
                self.assertEqual(new, old)

    def test_effective_projection_cells_keep_transient_state_transient(self):
        persistent = {(2, 3), (4, 5)}
        transient = {(4, 5), (6, 7)}
        effective = effective_table_projection_cells(
            persistent_table_cells_cxcy=persistent,
            transient_table_shadow_cells_cxcy=transient,
            enable_table_shadow_projection=True,
        )
        self.assertEqual(effective, {(2, 3), (4, 5), (6, 7)})
        self.assertEqual(persistent, {(2, 3), (4, 5)})

    def test_shadow_projection_can_be_disabled_without_losing_persistent_cells(self):
        effective = effective_table_projection_cells(
            persistent_table_cells_cxcy={(2, 3)},
            transient_table_shadow_cells_cxcy={(6, 7)},
            enable_table_shadow_projection=False,
        )
        self.assertEqual(effective, {(2, 3)})

    def test_visual_coverage_surface_selection_is_cellwise(self):
        floor = {(3, 2): 0.8, (3, 3): 0.6}
        table = {(3, 2): 0.2, (3, 3): 0.1}
        merged = merge_visual_coverages_by_surface(
            floor_coverages_rc=floor,
            table_coverages_rc=table,
            table_projection_cells_cxcy={(2, 3)},
        )
        self.assertEqual(merged, {(3, 2): 0.2, (3, 3): 0.6})

    def test_surface_payload_contains_only_persistent_table_cells(self):
        payload = surface_height_payload(
            {(2, 3)},
            {(2, 3): 0.91},
            enabled=True,
            floor_height_m=0.0,
            table_height_m=0.72,
        )
        self.assertEqual(payload["table_cells"], [[2, 3]])
        self.assertEqual(payload["height_by_cell"], {"2,3": 0.72})
        self.assertEqual(payload["table_confidence_by_cell"], {"2,3": 0.91})

    def test_bbox_overlap_semantics_are_unchanged(self):
        desks = [[100, 100, 200, 200]]
        self.assertTrue(bbox_overlaps_any_desk([120, 120, 140, 140], desks))
        self.assertTrue(bbox_overlaps_any_desk([195, 120, 215, 160], desks, min_overlap_frac=0.05))
        self.assertFalse(bbox_overlaps_any_desk([220, 120, 240, 160], desks))


if __name__ == "__main__":
    unittest.main()
