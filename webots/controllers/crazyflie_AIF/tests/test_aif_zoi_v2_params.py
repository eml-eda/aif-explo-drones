from __future__ import annotations

import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parents[1]
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

from configuration.aif_zoi_v2_params import (
    DETECTOR_CALIBRATIONS, AIFZOIV2Params, resolve_aif_zoi_v2_params, rule_of_succession)
from configuration.experiment_config import resolve_experiment_selection
from configuration.registry import ENVIRONMENTS, normalize_policy_id

HARD = ENVIRONMENTS["hard"]
CALIBRATED = {"AIF_ZOI_V2_RECALL": "0.8", "AIF_ZOI_V2_CLASS_ACCURACY": "0.95", "AIF_ZOI_V2_FALSE_ALARM": "0.002",
              "AIF_ZOI_V2_DESK_RECALL": "0.9", "AIF_ZOI_V2_DESK_FALSE_ALARM": "0.1"}


class CalibrationTests(unittest.TestCase):
    def test_rule_of_succession(self):
        self.assertAlmostEqual(rule_of_succession(0, 0), 0.5)
        self.assertAlmostEqual(rule_of_succession(140, 167), 141 / 169)

    def test_mobilenet_values_from_the_calibration_counts(self):
        params = resolve_aif_zoi_v2_params(HARD, environ={})
        self.assertEqual(params, resolve_aif_zoi_v2_params(HARD, environ={}, detector_id="mobilenet_ssd"))
        self.assertEqual(params.calibration, "hard/mobilenet_ssd")
        self.assertAlmostEqual(params.recall, 141 / 169)
        self.assertAlmostEqual(params.class_accuracy, 141 / 142)
        self.assertAlmostEqual(params.false_alarm, 3 / 3493)
        self.assertAlmostEqual(params.desk_recall, 182 / 191)
        self.assertAlmostEqual(params.desk_false_alarm, 189 / 1490)
        self.assertEqual(params.p_target_on_table, 0.9)
        self.assertTrue(params.table_context_enabled)

    def test_yolo_nano_values_from_the_same_calibration_images(self):
        params = resolve_aif_zoi_v2_params(HARD, environ={}, detector_id="yolo_nano")
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
        self.assertEqual(AIFZOIV2Params.calibrated("hard", "yolo_nano"),
                         resolve_aif_zoi_v2_params(HARD, environ={}, detector_id="yolo_nano"))

    def test_uncalibrated_pair_needs_every_detector_value(self):
        for env, det in (("hard", "yolo_small"), ("hard", "ground_truth"), ("medium", "mobilenet_ssd")):
            with self.assertRaises(ValueError):
                resolve_aif_zoi_v2_params(ENVIRONMENTS[env], environ={}, detector_id=det)
        params = resolve_aif_zoi_v2_params(HARD, environ=CALIBRATED, detector_id="yolo_small")
        self.assertEqual(params.calibration, "environment variables")
        self.assertEqual(params.recall, 0.8)


class ScenarioAndOverrideTests(unittest.TestCase):
    def test_overrides(self):
        params = resolve_aif_zoi_v2_params(HARD, environ={**CALIBRATED, "AIF_ZOI_V2_P_ON_TABLE": "0.7"})
        self.assertEqual(
            (params.recall, params.class_accuracy, params.false_alarm, params.desk_recall,
             params.desk_false_alarm, params.p_target_on_table),
            (0.8, 0.95, 0.002, 0.9, 0.1, 0.7))
        self.assertEqual(params.calibration, "hard/mobilenet_ssd")

    def test_environment_without_desks(self):
        params = resolve_aif_zoi_v2_params(ENVIRONMENTS["easy"], environ=CALIBRATED)
        self.assertFalse(params.table_context_enabled)

    def test_removed_model_switches_are_rejected(self):
        for name in ("AIF_ZOI_V2_SURFACE_MODEL", "AIF_ZOI_V2_RHO_MODEL", "AIF_ZOI_V2_N_TARGETS", "AIF_ZOI_V2_LAMBDA_TABLE"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    resolve_aif_zoi_v2_params(HARD, environ={name: "joint"})

    def test_rejects_exact_zero_or_one(self):
        for name in ("recall", "class_accuracy", "false_alarm", "desk_recall", "desk_false_alarm", "p_target_on_table"):
            for value in (0.0, 1.0, -0.1, 1.5):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        AIFZOIV2Params.calibrated(**{name: value})

    def test_planner_implementation(self):
        self.assertEqual(resolve_aif_zoi_v2_params(HARD, environ={}).planner_impl, "loop")
        params = resolve_aif_zoi_v2_params(HARD, environ={"AIF_ZOI_V2_PLANNER_IMPL": "vectorized"})
        self.assertEqual(params.planner_impl, "vectorized")
        with self.assertRaises(ValueError):
            resolve_aif_zoi_v2_params(HARD, environ={"AIF_ZOI_V2_PLANNER_IMPL": "gpu"})

    def test_grid_size(self):
        self.assertEqual(resolve_aif_zoi_v2_params(HARD, environ={}).grid_size, 128)
        self.assertEqual(resolve_aif_zoi_v2_params(HARD, environ={"AIF_ZOI_V2_GRID_SIZE": "64"}).grid_size, 64)
        for bad in ("64.5", "4", "big"):
            with self.assertRaises(ValueError):
                resolve_aif_zoi_v2_params(HARD, environ={"AIF_ZOI_V2_GRID_SIZE": bad})

    def test_desk_recall_must_exceed_desk_false_alarm(self):
        with self.assertRaises(ValueError):
            AIFZOIV2Params.calibrated(desk_recall=0.2, desk_false_alarm=0.3)

    def test_non_numeric_value_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_aif_zoi_v2_params(HARD, environ={"AIF_ZOI_V2_RECALL": "high"})

    def test_policy_is_selectable_in_every_environment(self):
        self.assertEqual(normalize_policy_id("AIF_ZOI_V2"), "aif_zoi_v2")
        for env, spec in ENVIRONMENTS.items():
            self.assertIn("aif_zoi_v2", spec.supported_policies)
            selection = resolve_experiment_selection(
                environment=env, run_mode="evaluation", policy="aif_zoi_v2")
            self.assertEqual(selection.policy, "aif_zoi_v2")


if __name__ == "__main__":
    unittest.main()
