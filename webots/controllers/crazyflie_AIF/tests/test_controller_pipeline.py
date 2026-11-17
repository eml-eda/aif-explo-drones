from __future__ import annotations

import unittest
from dataclasses import dataclass

from runtime.controller_pipeline import (
    PolicyPerceptionEvidence,
    TargetDistanceTracker,
    TaskStateSnapshot,
    build_policy_observation,
    build_progress_payload,
    detection_event_rc_to_cxcy,
    mapping_rc_to_cxcy,
    rc_to_cxcy,
)


@dataclass(frozen=True)
class _GridSpec:
    min_x: float = -2.75
    max_x: float = 2.75
    min_y: float = -3.25
    max_y: float = 3.25
    cell_size: float = 0.5
    rows: int = 13
    cols: int = 11


@dataclass(frozen=True)
class _FovSpec:
    half_left: float
    half_right: float
    forward: float
    backward: float


class ControllerPipelineTests(unittest.TestCase):
    def test_grid_coordinate_conversion_preserves_row_col_fields(self):
        self.assertEqual(rc_to_cxcy((3, 8)), (8, 3))
        self.assertEqual(mapping_rc_to_cxcy({(3, 8): 0.6}), {(8, 3): 0.6})

        event = detection_event_rc_to_cxcy(
            {
                "cell_rc": (3, 8),
                "covered_cells_rc": [[3, 8], [4, 8]],
                "cell_rc_floor": (5, 7),
                "cell_rc_table": (4, 7),
            }
        )
        self.assertEqual(event["cell"], (8, 3))
        self.assertEqual(event["covered_cells"], [[8, 3], [8, 4]])
        self.assertEqual(event["cell_floor"], (7, 5))
        self.assertEqual(event["cell_table"], (7, 4))

    def test_task_snapshot_normalizes_supervisor_payload(self):
        snapshot = TaskStateSnapshot.from_payload(
            {
                "status": "completed",
                "reason": "all_found",
                "total_objects": 6,
                "remaining_objects": 0,
                "validated_found_cells": [[2, 4], [7, 1]],
                "validated_found_objects": [
                    {"cx": 2, "cy": 4, "class_name": "target_bottle"},
                    {"cx": 7, "cy": 1, "class_name": "target_can"},
                ],
            }
        )
        self.assertEqual(snapshot.status, "completed")
        self.assertEqual(snapshot.total_objects, 6)
        self.assertEqual(snapshot.validated_found_cells, frozenset({(2, 4), (7, 1)}))
        self.assertIn((2, 4, "target_bottle"), snapshot.validated_found_objects)

    def test_skipped_perception_step_has_no_policy_evidence(self):
        evidence = PolicyPerceptionEvidence.for_step(
            perception_processed=False,
            visible_mask={(1, 2): True},
            det_confidence={(1, 2): 0.9},
            det_confidence_multiclass={(1, 2): 0.9},
            det_metadata_multiclass={(1, 2): {"class_name": "target_can"}},
            detections_all=[{"cell": (1, 2)}],
            target_detections=[{"cell": (1, 2)}],
            table_confidence={(4, 5): 0.8},
            table_visible_mask={(4, 5): True},
            table_coverages={(4, 5): 0.7},
        )
        self.assertEqual(evidence.visible_mask, {})
        self.assertEqual(evidence.det_confidence, {})
        self.assertEqual(evidence.detections_all, [])
        self.assertEqual(evidence.table_coverages, {})

    def test_policy_observation_keeps_established_schema(self):
        evidence = PolicyPerceptionEvidence.for_step(
            perception_processed=True,
            visible_mask={(1, 2): True},
            det_confidence={(1, 2): 0.91},
            det_confidence_multiclass={(1, 2): 0.91},
            det_metadata_multiclass={(1, 2): {"class_name": "target_bottle", "class_id": 0}},
            detections_all=[{"cell": (1, 2), "class_name": "target_bottle"}],
            target_detections=[{"cell": (1, 2), "class_name": "target_bottle"}],
            table_confidence={(3, 4): 0.7},
            table_visible_mask={(3, 4): True},
            table_coverages={(3, 4): 0.6},
        )
        task = TaskStateSnapshot.from_payload(
            {
                "total_objects": 6,
                "remaining_objects": 5,
                "validated_found_cells": [[1, 2]],
                "validated_found_objects": [
                    {"cx": 1, "cy": 2, "class_name": "target_bottle"}
                ],
            }
        )
        floor = _FovSpec(0.7, 0.694, 0.553, 0.493)
        table = _FovSpec(0.363, 0.362, 0.288, 0.256)
        sensor_state = {
            "time": 12.5,
            "gps": {"x": 0.0, "y": 0.0, "z": 1.5},
            "camera": {"width": 320, "height": 240},
        }

        obs = build_policy_observation(
            sensor_state=sensor_state,
            step=123,
            pose={"x": 0.0, "y": 0.0, "yaw": 0.2},
            grid_spec=_GridSpec(),
            detector_detections=[{"class_name": "target_bottle", "confidence": 0.91}],
            current_cell=(5, 6),
            visual_coverages={(5, 6): 0.8},
            perception_evidence=evidence,
            coverage_threshold=0.05,
            persistent_table_confidence={(3, 4): 0.7},
            perception_processed=True,
            surface_height_map={"enabled": True},
            persistent_table_cells={(3, 4)},
            effective_table_projection_cells={(3, 4), (3, 5)},
            claimed_found_cells={(1, 2)},
            task_state=task,
            claim_confidence_threshold=0.45,
            snapshots_enabled=True,
            snapshots_dir="/tmp/run/snapshots",
            snapshots_every_n=200,
            snapshots_on_stuck=True,
            stuck_replan_limit=100,
            floor_fov_spec=floor,
            table_fov_spec=table,
            floor_height_m=0.0,
            table_height_m=0.72,
        )

        self.assertEqual(obs["step"], 123)
        self.assertEqual(obs["grid"]["current_cell"], (5, 6))
        self.assertEqual(obs["grid"]["coverages"], {(5, 6): 0.8})
        self.assertEqual(obs["grid"]["det_confidence"], {(1, 2): 0.91})
        self.assertEqual(obs["grid"]["table_cells"], [(3, 4)])
        self.assertEqual(obs["task"]["remaining_objects"], 5)
        self.assertEqual(obs["task"]["claim_confidence_threshold"], 0.45)
        self.assertIn("detections", obs)
        self.assertEqual(obs["fov_spec"]["forward"], 0.553)
        self.assertEqual(obs["fov_spec_by_surface"]["table"]["height_m"], 0.72)

    def test_plane_coverages_are_additive_grid_keys(self):
        evidence = PolicyPerceptionEvidence.for_step(
            perception_processed=True,
            visible_mask={(5, 6): True},
            det_confidence={},
            det_confidence_multiclass={},
            det_metadata_multiclass={},
            detections_all=[],
            target_detections=[],
            table_confidence={},
            table_visible_mask={(5, 6): True},
            table_coverages={(5, 6): 0.8},
        )
        fov = _FovSpec(0.5, 0.5, 0.5, 0.5)
        kwargs = dict(
            sensor_state={"time": 1.0},
            step=1,
            pose={"x": 0.0, "y": 0.0, "yaw": 0.0},
            grid_spec=_GridSpec(),
            detector_detections=[],
            current_cell=(5, 6),
            visual_coverages={(5, 6): 0.8},
            perception_evidence=evidence,
            coverage_threshold=0.05,
            persistent_table_confidence={},
            perception_processed=True,
            surface_height_map={},
            persistent_table_cells=set(),
            effective_table_projection_cells=set(),
            claimed_found_cells=set(),
            task_state=TaskStateSnapshot(),
            claim_confidence_threshold=0.45,
            snapshots_enabled=False,
            snapshots_dir=None,
            snapshots_every_n=200,
            snapshots_on_stuck=True,
            stuck_replan_limit=100,
            floor_fov_spec=fov,
            table_fov_spec=fov,
            floor_height_m=0.0,
            table_height_m=0.72,
        )
        without = build_policy_observation(**kwargs)
        with_planes = build_policy_observation(
            **kwargs,
            floor_plane_coverages={(5, 6): 0.8, (5, 7): 0.3},
            table_plane_coverages={(5, 6): 0.4},
        )
        self.assertNotIn("floor_plane_coverages", without["grid"])
        self.assertEqual(with_planes["grid"].pop("floor_plane_coverages"), {(5, 6): 0.8, (5, 7): 0.3})
        self.assertEqual(with_planes["grid"].pop("table_plane_coverages"), {(5, 6): 0.4})
        self.assertEqual(with_planes, without)

    def test_target_distance_tracker_matches_legacy_aggregation(self):
        tracker = TargetDistanceTracker()
        self.assertFalse(tracker.record_policy_action({"forward": 0.1}))
        self.assertTrue(tracker.record_policy_action({"target_selection": {"distance_m": 2.0}}))
        self.assertTrue(tracker.record_policy_action({"target_selection": {"distance_m": 1.0}}))
        payload = tracker.payload(enabled=True, policy_supported=True)
        self.assertEqual(payload["count"], 2)
        self.assertAlmostEqual(payload["sum_m"], 3.0)
        self.assertAlmostEqual(payload["average_m"], 1.5)
        self.assertAlmostEqual(payload["min_m"], 1.0)
        self.assertAlmostEqual(payload["max_m"], 2.0)
        self.assertTrue(payload["tracked"])

    def test_progress_payload_keeps_report_contract(self):
        payload = build_progress_payload(
            policy_name="rotate_and_measure",
            run_dir="/tmp/run",
            autonomy_start_time_s=15.0,
            last_controller_time_s=20.0,
            visited_cells={(2, 1), (1, 1)},
            claimed_found_cells={(7, 8)},
            claimed_found_objects={(7, 8, "target_can")},
            detector_limitations={"mode": "time", "time_interval_s": 0.625},
            experiment_metadata={"environment": "easy"},
            target_distance_stats={"tracked": False},
            drone_spawn={"x": 0.0, "y": 0.0, "yaw": 0.0, "cx": 5, "cy": 6},
            surface_height_map={"enabled": True},
        )
        self.assertEqual(payload["visited_cells"], [[1, 1], [2, 1]])
        self.assertEqual(
            payload["claimed_found_objects"],
            [{"cx": 7, "cy": 8, "class_name": "target_can", "label": "target_can"}],
        )
        self.assertEqual(payload["experiment_metadata"]["environment"], "easy")


if __name__ == "__main__":
    unittest.main()
