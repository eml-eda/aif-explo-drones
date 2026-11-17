from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vision.perception_projection import (
    PerceptionProjectionConfig,
    project_perception_frame,
)


class _PerceptionManager:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def process_if_due(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _EnvManager:
    def __init__(self, cell):
        self.cell = cell

    def detection_cell_from_bbox(self, **_kwargs):
        return self.cell


class PerceptionProjectionTests(unittest.TestCase):
    def setUp(self):
        self.config = PerceptionProjectionConfig(
            target_class_names=frozenset({"target_bottle", "target_can"}),
            target_class_ids=frozenset({0, 1}),
            desk_class_names=frozenset({"desk"}),
            desk_class_ids=frozenset({2}),
            surface_detection_conf_threshold=0.30,
            enable_table_shadow_projection=True,
            table_shadow_forward_tolerance_m=0.10,
            min_x=-2.75,
            min_y=-3.25,
            cell_size_m=0.5,
        )
        self.camera = {"image": b"pixels", "width": 320, "height": 240, "time": 1.25}

    def test_skipped_frame_returns_no_new_evidence(self):
        raw_output = SimpleNamespace(detections=[{"class_name": "target_can"}])
        manager = _PerceptionManager(
            SimpleNamespace(processed=False, events=[], raw_output=raw_output)
        )
        frame = project_perception_frame(
            perception_manager=manager,
            config=self.config,
            step_count=10,
            time_now_s=1.25,
            camera_capture=self.camera,
            drone_x=0.0,
            drone_y=0.0,
            drone_yaw=0.0,
            env_mgr_floor=_EnvManager((1, 1)),
            env_mgr_table=_EnvManager((2, 2)),
            floor_visual_coverages={(1, 1): 0.5},
            persistent_table_cells=set(),
            persistent_table_confidence={},
        )
        self.assertFalse(frame.processed)
        self.assertIs(frame.raw_output, raw_output)
        self.assertEqual(frame.target_confidence_rc, {})
        self.assertEqual(frame.all_events_rc, [])
        self.assertEqual(frame.transient_table_shadow_cells, set())

    @patch("vision.perception_projection.sample_front_only_bbox_floor_counterparts_rc")
    @patch("vision.perception_projection.sample_bbox_projected_cells_rc")
    @patch("vision.perception_projection.bbox_overlaps_any_desk")
    def test_learned_frame_persists_desk_before_target_projection(
        self,
        overlaps_mock,
        sample_bbox_mock,
        shadow_mock,
    ):
        overlaps_mock.return_value = True
        sample_bbox_mock.return_value = {(2, 3)}
        shadow_mock.return_value = {(1, 1)}

        desk_bbox = [10.0, 20.0, 100.0, 120.0]
        target_bbox = [20.0, 30.0, 50.0, 90.0]
        events = [
            {
                "bbox_xyxy": desk_bbox,
                "confidence": 0.8,
                "class_id": 2,
                "class_name": "desk",
                "label": "desk",
            },
            {
                "bbox_xyxy": target_bbox,
                "confidence": 0.7,
                "class_id": 1,
                "class_name": "target_can",
                "label": "target_can",
            },
        ]
        raw_output = SimpleNamespace(detections=events)
        manager = _PerceptionManager(
            SimpleNamespace(processed=True, events=events, raw_output=raw_output)
        )
        persistent_cells = set()
        persistent_confidence = {}

        frame = project_perception_frame(
            perception_manager=manager,
            config=self.config,
            step_count=30,
            time_now_s=3.0,
            camera_capture=self.camera,
            drone_x=0.1,
            drone_y=-0.2,
            drone_yaw=0.3,
            env_mgr_floor=_EnvManager((6, 7)),
            env_mgr_table=_EnvManager((4, 5)),
            floor_visual_coverages={(1, 1): 0.4},
            persistent_table_cells=persistent_cells,
            persistent_table_confidence=persistent_confidence,
        )

        self.assertEqual(persistent_cells, {(3, 2)})
        self.assertAlmostEqual(persistent_confidence[(3, 2)], 0.8)
        self.assertEqual(frame.table_confidence_rc, {(2, 3): 0.8})
        self.assertEqual(frame.transient_table_shadow_cells, {(1, 1)})
        self.assertEqual(len(frame.target_events_rc), 1)
        target = frame.target_events_rc[0]
        self.assertEqual(target["cell_rc_floor"], (6, 7))
        self.assertEqual(target["cell_rc_table"], (4, 5))
        self.assertEqual(target["cell_rc"], (4, 5))
        self.assertEqual(target["surface_assumption"], "table")
        self.assertEqual(frame.target_confidence_rc, {(4, 5): 0.7})


if __name__ == "__main__":
    unittest.main()
