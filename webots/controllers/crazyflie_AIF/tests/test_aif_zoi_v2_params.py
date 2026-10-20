from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parents[1]
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

from configuration.aif_zoi_v2_params import (
    SUCCESSION_CLASS_ACCURACY, SUCCESSION_FALSE_ALARM, SUCCESSION_RECALL, TABLE_MAP_RULES,
    AIFZOIV2Params, resolve_aif_zoi_v2_params, scenario_priors)
from configuration.experiment_config import resolve_experiment_selection
from configuration.registry import ENVIRONMENTS, normalize_policy_id


class AIFZOIV2ParamsTests(unittest.TestCase):
    def test_defaults_with_tables(self):
        # The 73/100 model (eval_aif_zoi_v2_mbnetv2_hard_20261014_160738).
        for env in ("medium", "hard"):
            params = resolve_aif_zoi_v2_params(ENVIRONMENTS[env], environ={})
            self.assertEqual(
                params.as_dict(),
                {
                    "recall": 0.84,
                    "class_accuracy": 0.98,
                    "false_alarm": 0.001,
                    "lambda_table": 0.54,
                    "lambda_floor": 0.0045,
                    "rho": 0.07,
                    "table_map_rule": "touched",
                    "surface_model": "typed",
                    "planner_objective": "distance_cost",
                    "planner_terms": "info_and_class",
                    "desk_recall": 182 / 191,
                    "desk_false_alarm": 189 / 1490,
                    "table_context_enabled": True,
                },
            )

    def test_succession_values_from_calibration_counts(self):
        # (k + 1) / (n + 2) on the calibration counts; not defaults yet.
        self.assertAlmostEqual(SUCCESSION_RECALL, 141 / 169)
        self.assertAlmostEqual(SUCCESSION_CLASS_ACCURACY, 141 / 142)
        self.assertAlmostEqual(SUCCESSION_FALSE_ALARM, 3 / 3493)

    def test_joint_model_defaults(self):
        params = resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={"AIF_ZOI_V2_SURFACE_MODEL": "joint"})
        self.assertEqual(params.surface_model, "joint")
        # All from the one calibration set and the scenario description.
        self.assertAlmostEqual(params.recall, 141 / 169)
        self.assertAlmostEqual(params.class_accuracy, 141 / 142)
        self.assertAlmostEqual(params.false_alarm, 3 / 3493)
        self.assertAlmostEqual(params.desk_recall, 182 / 191)
        self.assertAlmostEqual(params.desk_false_alarm, 189 / 1490)
        c = 3 * 1.2 * 0.7 / 0.25
        self.assertAlmostEqual(params.lambda_table, 5.4 / c)
        self.assertAlmostEqual(params.lambda_floor, 0.6 / (143 - c))
        self.assertAlmostEqual(params.rho, c / 143)

    def test_planner_objective(self):
        params = resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={"AIF_ZOI_V2_PLANNER": "rate"})
        self.assertEqual(params.planner_objective, "rate")
        with self.assertRaises(ValueError):
            resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={"AIF_ZOI_V2_PLANNER": "fastest"})
        with self.assertRaises(ValueError):
            AIFZOIV2Params(planner_objective="fastest")

    def test_planner_terms(self):
        params = resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={"AIF_ZOI_V2_PLANNER_TERMS": "info_only"})
        self.assertEqual(params.planner_terms, "info_only")
        with self.assertRaises(ValueError):
            resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={"AIF_ZOI_V2_PLANNER_TERMS": "class_only"})
        with self.assertRaises(ValueError):
            AIFZOIV2Params(planner_terms="class_only")

    def test_joint_model_overrides_and_validation(self):
        params = resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={
            "AIF_ZOI_V2_SURFACE_MODEL": "joint", "AIF_ZOI_V2_DESK_RECALL": "0.8",
            "AIF_ZOI_V2_DESK_FALSE_ALARM": "0.2"})
        self.assertEqual((params.desk_recall, params.desk_false_alarm), (0.8, 0.2))
        with self.assertRaises(ValueError):
            resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={"AIF_ZOI_V2_SURFACE_MODEL": "both"})
        with self.assertRaises(ValueError):
            AIFZOIV2Params(desk_recall=0.2, desk_false_alarm=0.3)
        for value in (0.0, 1.0):
            with self.assertRaises(ValueError):
                AIFZOIV2Params(desk_recall=value)

    def test_centre_rule_priors(self):
        params = resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={"AIF_ZOI_V2_TABLE_MAP": "centre"})
        self.assertEqual(params.table_map_rule, "centre")
        # C = 3 desks x cells whose centre is inside a desk box.
        c = 3 * (1.2 * 0.7 + (1.2 ** 2 + 0.7 ** 2) / math.pi) / 0.25
        self.assertAlmostEqual(params.lambda_table, 5.4 / c)
        self.assertAlmostEqual(params.lambda_floor, 0.6 / (143 - c))
        self.assertAlmostEqual(params.rho, c / 143)
        self.assertEqual(params.recall, 0.84)

    def test_unobserved_prior_is_uniform_for_every_rule(self):
        # rho * lambda_T + (1 - rho) * lambda_F = N / M whatever C is.
        for rule in TABLE_MAP_RULES:
            pr = scenario_priors(rule)
            self.assertAlmostEqual(
                pr["rho"] * pr["lambda_table"] + (1 - pr["rho"]) * pr["lambda_floor"], 6 / 143)

    def test_touched_scenario_priors_use_the_centre_on_desk_count(self):
        pr = scenario_priors("touched")
        c = 3 * 1.2 * 0.7 / 0.25
        self.assertAlmostEqual(pr["lambda_table"], 5.4 / c)
        self.assertAlmostEqual(pr["rho"], c / 143)
        # The 73/100 defaults are these values rounded.
        self.assertAlmostEqual(pr["lambda_table"], 0.54, places=2)
        self.assertAlmostEqual(pr["lambda_floor"], 0.0045, places=4)
        self.assertAlmostEqual(pr["rho"], 0.07, places=2)

    def test_unknown_table_map_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={"AIF_ZOI_V2_TABLE_MAP": "corners"})
        with self.assertRaises(ValueError):
            AIFZOIV2Params(table_map_rule="corners")

    def test_defaults_without_tables(self):
        params = resolve_aif_zoi_v2_params(ENVIRONMENTS["easy"], environ={})
        self.assertFalse(params.table_context_enabled)
        self.assertAlmostEqual(params.lambda_floor, 6.0 / 143.0)
        self.assertEqual(params.rho, 0.0)

    def test_environment_overrides(self):
        environ = {
            "AIF_ZOI_V2_RECALL": "0.7",
            "AIF_ZOI_V2_CLASS_ACCURACY": "0.9",
            "AIF_ZOI_V2_FALSE_ALARM": "0.02",
            "AIF_ZOI_V2_LAMBDA_TABLE": "0.4",
            "AIF_ZOI_V2_LAMBDA_FLOOR": "0.01",
            "AIF_ZOI_V2_RHO": "0.1",
        }
        params = resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ=environ)
        self.assertEqual(
            (params.recall, params.class_accuracy, params.false_alarm,
             params.lambda_table, params.lambda_floor, params.rho),
            (0.7, 0.9, 0.02, 0.4, 0.01, 0.1),
        )

    def test_rejects_exact_zero_or_one(self):
        for name in ("recall", "class_accuracy", "false_alarm", "lambda_table", "lambda_floor"):
            for value in (0.0, 1.0, -0.1, 1.5):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        AIFZOIV2Params(**{name: value})

    def test_rho_bounds(self):
        AIFZOIV2Params(rho=0.0)
        AIFZOIV2Params(rho=1.0)
        with self.assertRaises(ValueError):
            AIFZOIV2Params(rho=1.01)

    def test_non_numeric_value_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={"AIF_ZOI_V2_RECALL": "high"})

    def test_policy_is_selectable_in_every_environment(self):
        self.assertEqual(normalize_policy_id("AIF_ZOI_V2"), "aif_zoi_v2")
        for env, spec in ENVIRONMENTS.items():
            self.assertIn("aif_zoi_v2", spec.supported_policies)
            selection = resolve_experiment_selection(
                environment=env, run_mode="evaluation", policy="aif_zoi_v2")
            self.assertEqual(selection.policy, "aif_zoi_v2")


if __name__ == "__main__":
    unittest.main()
