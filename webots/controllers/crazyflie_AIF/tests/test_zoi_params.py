from __future__ import annotations

import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parents[1]
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

from configuration.zoi_params import (
    DETECTOR_CALIBRATIONS, ZOIParams, DetectorCalibration, resolve_zoi_params, rule_of_succession)
from configuration.experiment_config import resolve_experiment_selection
from configuration.registry import ENVIRONMENTS, normalize_policy_id

HARD = ENVIRONMENTS["hard"]
CALIBRATED = {"ZOI_RECALL": "0.8", "ZOI_CLASS_ACCURACY": "0.95", "ZOI_FALSE_ALARM": "0.002",
              "ZOI_DESK_RECALL": "0.9", "ZOI_DESK_FALSE_ALARM": "0.1"}


class CalibrationTests(unittest.TestCase):
    def test_rule_of_succession(self):
        self.assertAlmostEqual(rule_of_succession(0, 0), 0.5)
        self.assertAlmostEqual(rule_of_succession(140, 167), 141 / 169)

    def test_mobilenet_values_from_the_calibration_counts(self):
        params = resolve_zoi_params(HARD, environ={})
        self.assertEqual(params, resolve_zoi_params(HARD, environ={}, detector_id="mobilenet_ssd"))
        self.assertEqual(params.calibration, "hard/mobilenet_ssd")
        self.assertAlmostEqual(params.recall, 141 / 169)
        self.assertAlmostEqual(params.class_accuracy, 141 / 142)
        self.assertAlmostEqual(params.false_alarm, 3 / 3493)
        self.assertAlmostEqual(params.desk_recall, 182 / 191)
        self.assertAlmostEqual(params.desk_false_alarm, 189 / 1490)
        self.assertEqual(params.p_target_on_table, 0.9)
        self.assertTrue(params.table_context_enabled)

    def test_yolo_nano_values_from_the_same_calibration_images(self):
        params = resolve_zoi_params(HARD, environ={}, detector_id="yolo_nano")
        self.assertEqual(params.calibration, "hard/yolo_nano")
        self.assertAlmostEqual(params.recall, 168 / 169)
        self.assertAlmostEqual(params.class_accuracy, 168 / 169)
        self.assertAlmostEqual(params.false_alarm, 7 / 3493)
        self.assertAlmostEqual(params.desk_recall, 183 / 191)
        self.assertAlmostEqual(params.desk_false_alarm, 199 / 1490)
        mbn, yolo = DETECTOR_CALIBRATIONS[("hard", "mobilenet_ssd")], DETECTOR_CALIBRATIONS[("hard", "yolo_nano")]
        for field in ("targets", "empty_cell_frames", "table_cells", "floor_cells"):
            self.assertEqual(getattr(mbn, field), getattr(yolo, field))

    def test_calibrated_constructor_matches_the_resolver(self):
        self.assertEqual(ZOIParams.calibrated("hard", "yolo_nano"),
                         resolve_zoi_params(HARD, environ={}, detector_id="yolo_nano"))

    def test_medium_calibrations(self):
        mbn = resolve_zoi_params(ENVIRONMENTS["medium"], environ={}, detector_id="mobilenet_ssd")
        yolo = resolve_zoi_params(ENVIRONMENTS["medium"], environ={}, detector_id="yolo_nano")
        self.assertEqual((mbn.calibration, yolo.calibration), ("medium/mobilenet_ssd", "medium/yolo_nano"))
        self.assertAlmostEqual(mbn.recall, 183 / 189)
        self.assertAlmostEqual(mbn.false_alarm, 1 / 3417)
        self.assertAlmostEqual(mbn.desk_recall, 228 / 240)
        self.assertAlmostEqual(mbn.desk_false_alarm, 253 / 1416)
        self.assertAlmostEqual(yolo.recall, 188 / 189)
        self.assertAlmostEqual(yolo.false_alarm, 4 / 3417)
        self.assertAlmostEqual(yolo.desk_recall, 230 / 240)
        self.assertAlmostEqual(yolo.desk_false_alarm, 261 / 1416)
        self.assertTrue(mbn.table_context_enabled)

    def test_easy_calibrations_have_no_desk_values(self):
        for det, (r, f) in (("mobilenet_ssd", (302 / 335, 24 / 3348)), ("yolo_nano", (322 / 335, 18 / 3348))):
            params = resolve_zoi_params(ENVIRONMENTS["easy"], environ={}, detector_id=det)
            self.assertEqual(params.calibration, f"easy/{det}")
            self.assertAlmostEqual(params.recall, r)
            self.assertAlmostEqual(params.false_alarm, f)
            self.assertEqual((params.desk_recall, params.desk_false_alarm), (None, None))
            self.assertFalse(params.table_context_enabled)

    def test_uncalibrated_pair_needs_every_detector_value(self):
        for env, det in (("hard", "yolo_small"), ("hard", "ground_truth"), ("easy", "yolo_small")):
            with self.assertRaises(ValueError):
                resolve_zoi_params(ENVIRONMENTS[env], environ={}, detector_id=det)
        params = resolve_zoi_params(HARD, environ=CALIBRATED, detector_id="yolo_small")
        self.assertEqual(params.calibration, "environment variables")
        self.assertEqual(params.recall, 0.8)


class ScenarioAndOverrideTests(unittest.TestCase):
    def test_overrides(self):
        params = resolve_zoi_params(HARD, environ={**CALIBRATED, "ZOI_P_ON_TABLE": "0.7"})
        self.assertEqual(
            (params.recall, params.class_accuracy, params.false_alarm, params.desk_recall,
             params.desk_false_alarm, params.p_target_on_table),
            (0.8, 0.95, 0.002, 0.9, 0.1, 0.7))
        self.assertEqual(params.calibration, "hard/mobilenet_ssd")

    def test_environment_without_desks(self):
        params = resolve_zoi_params(ENVIRONMENTS["easy"], environ=CALIBRATED)
        self.assertFalse(params.table_context_enabled)

    def test_environment_without_desks_needs_only_the_target_values(self):
        target_only = {k: v for k, v in CALIBRATED.items() if "DESK" not in k}
        params = resolve_zoi_params(ENVIRONMENTS["easy"], environ=target_only, detector_id="yolo_small")
        self.assertEqual((params.recall, params.desk_recall, params.desk_false_alarm), (0.8, None, None))
        # A desk value given alone is dropped (unused without desks).
        params = resolve_zoi_params(
            ENVIRONMENTS["easy"], environ={**target_only, "ZOI_DESK_RECALL": "0.9"}, detector_id="yolo_small")
        self.assertIsNone(params.desk_recall)
        # With desks, the desk values are required.
        with self.assertRaises(ValueError):
            resolve_zoi_params(HARD, environ=target_only, detector_id="yolo_small")

    def test_calibration_without_desk_counts(self):
        key = ("easy", "mobilenet_ssd")
        DETECTOR_CALIBRATIONS[key] = DetectorCalibration(
            targets=100, targets_detected=90, class_correct=89, false_boxes=1, empty_cell_frames=2000)
        try:
            params = resolve_zoi_params(ENVIRONMENTS["easy"], environ={}, detector_id="mobilenet_ssd")
        finally:
            del DETECTOR_CALIBRATIONS[key]
        self.assertEqual(params.calibration, "easy/mobilenet_ssd")
        self.assertAlmostEqual(params.recall, 91 / 102)
        self.assertIsNone(params.desk_recall)
        self.assertFalse(params.table_context_enabled)

    def test_desk_values_are_required_with_desks_and_go_together(self):
        with self.assertRaises(ValueError):
            ZOIParams(recall=0.8, class_accuracy=0.9, false_alarm=0.01)
        with self.assertRaises(ValueError):
            ZOIParams(recall=0.8, class_accuracy=0.9, false_alarm=0.01, desk_recall=0.9,
                           table_context_enabled=False)
        ZOIParams(recall=0.8, class_accuracy=0.9, false_alarm=0.01, table_context_enabled=False)

    def test_removed_model_switches_are_rejected(self):
        for name in ("ZOI_SURFACE_MODEL", "ZOI_RHO_MODEL", "ZOI_N_TARGETS", "ZOI_LAMBDA_TABLE"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    resolve_zoi_params(HARD, environ={name: "joint"})

    def test_rejects_exact_zero_or_one(self):
        for name in ("recall", "class_accuracy", "false_alarm", "desk_recall", "desk_false_alarm", "p_target_on_table"):
            for value in (0.0, 1.0, -0.1, 1.5):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        ZOIParams.calibrated(**{name: value})

    def test_planner_implementation(self):
        self.assertEqual(resolve_zoi_params(HARD, environ={}).planner_impl, "loop")
        params = resolve_zoi_params(HARD, environ={"ZOI_PLANNER_IMPL": "vectorized"})
        self.assertEqual(params.planner_impl, "vectorized")
        with self.assertRaises(ValueError):
            resolve_zoi_params(HARD, environ={"ZOI_PLANNER_IMPL": "gpu"})

    def test_grid_size(self):
        self.assertEqual(resolve_zoi_params(HARD, environ={}).grid_size, 32)
        self.assertEqual(resolve_zoi_params(HARD, environ={"ZOI_GRID_SIZE": "64"}).grid_size, 64)
        for bad in ("64.5", "4", "big"):
            with self.assertRaises(ValueError):
                resolve_zoi_params(HARD, environ={"ZOI_GRID_SIZE": bad})

    def test_desk_recall_must_exceed_desk_false_alarm(self):
        with self.assertRaises(ValueError):
            ZOIParams.calibrated(desk_recall=0.2, desk_false_alarm=0.3)

    def test_non_numeric_value_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_zoi_params(HARD, environ={"ZOI_RECALL": "high"})

    def test_policy_is_selectable_in_every_environment(self):
        self.assertEqual(normalize_policy_id("ZOI"), "zoi")
        for env, spec in ENVIRONMENTS.items():
            self.assertIn("zoi", spec.supported_policies)
            selection = resolve_experiment_selection(
                environment=env, run_mode="evaluation", policy="zoi")
            self.assertEqual(selection.policy, "zoi")


if __name__ == "__main__":
    unittest.main()
