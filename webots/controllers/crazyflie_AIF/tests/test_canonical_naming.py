from __future__ import annotations

import sys
import unittest
from pathlib import Path

AIF_DIR = Path(__file__).resolve().parents[1]
CONTROLLERS_DIR = AIF_DIR.parent
if str(AIF_DIR) not in sys.path:
    sys.path.insert(0, str(AIF_DIR))
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from configuration.registry import ENVIRONMENTS, normalize_environment_id
from dataset_collection.class_config import normalize_collection_environment
from dataset_collection.collection_config import get_collection_config


class CanonicalNamingTests(unittest.TestCase):
    def test_runtime_environment_ids_are_canonical_only(self):
        self.assertEqual(set(ENVIRONMENTS), {"easy", "medium", "hard"})
        for env in ("easy", "medium", "hard"):
            self.assertEqual(normalize_environment_id(env), env)
        for legacy in ("mixed_easy", "table", "tables", "mixed_tables", "mixed_hard"):
            with self.subTest(legacy=legacy):
                with self.assertRaises(ValueError):
                    normalize_environment_id(legacy)

    def test_easy_supports_aif_zoi(self):
        self.assertIn("aif_zoi", ENVIRONMENTS["easy"].supported_policies)

    def test_registry_uses_environment_explicit_supervisors(self):
        self.assertEqual(ENVIRONMENTS["easy"].single_supervisor, "worldgen_easy_supervisor")
        self.assertEqual(ENVIRONMENTS["easy"].evaluation_supervisor, "worldgen_easy_eval_supervisor")
        self.assertEqual(ENVIRONMENTS["medium"].single_supervisor, "worldgen_medium_supervisor")
        self.assertEqual(ENVIRONMENTS["medium"].evaluation_supervisor, "worldgen_medium_eval_supervisor")
        self.assertEqual(ENVIRONMENTS["hard"].single_supervisor, "worldgen_hard_supervisor")
        self.assertEqual(ENVIRONMENTS["hard"].evaluation_supervisor, "worldgen_hard_eval_supervisor")

    def test_dataset_environment_ids_are_canonical_only(self):
        for env in ("easy", "medium", "hard"):
            self.assertEqual(normalize_collection_environment(env), env)
        for legacy in ("mixed_easy", "table", "tables", "mixed_tables", "mixed_hard"):
            with self.subTest(legacy=legacy):
                with self.assertRaises(ValueError):
                    normalize_collection_environment(legacy)

    def test_medium_dataset_config_contains_no_table_env_alias(self):
        cfg = get_collection_config("medium")
        env_names = (
            cfg.policy_env_names
            + cfg.capture_interval_env_names
            + cfg.samples_per_run_env_names
            + cfg.save_empty_env_names
            + cfg.save_segmentation_env_names
            + cfg.save_manifest_env_names
            + cfg.save_metadata_env_names
            + cfg.output_root_env_names
        )
        self.assertTrue(env_names)
        self.assertTrue(all(name.startswith("MEDIUM_DATASET_") for name in env_names))
        self.assertFalse(any(name.startswith("TABLE_DATASET_") for name in env_names))

    def test_webots_controller_directories_are_canonical(self):
        canonical = (
            "crazyflie_easy_dataset_collector",
            "crazyflie_medium_dataset_collector",
            "crazyflie_hard_dataset_collector",
            "worldgen_easy_supervisor",
            "worldgen_easy_eval_supervisor",
            "worldgen_easy_dataset_supervisor",
            "worldgen_medium_supervisor",
            "worldgen_medium_eval_supervisor",
            "worldgen_medium_dataset_supervisor",
            "worldgen_hard_supervisor",
            "worldgen_hard_eval_supervisor",
            "worldgen_hard_dataset_supervisor",
        )
        for name in canonical:
            with self.subTest(name=name):
                controller_dir = CONTROLLERS_DIR / name
                self.assertTrue(controller_dir.is_dir())
                self.assertTrue((controller_dir / f"{name}.py").is_file())

        obsolete = (
            "crazyflie_dataset_collector",
            "crazyflie_table_dataset_collector",
            "worldgen_supervisor",
            "worldgen_eval_supervisor",
            "worldgen_dataset_supervisor",
            "worldgen_table_supervisor",
            "worldgen_table_eval_supervisor",
            "worldgen_table_dataset_supervisor",
        )
        for name in obsolete:
            with self.subTest(obsolete=name):
                self.assertFalse((CONTROLLERS_DIR / name).exists())

    def test_active_environment_variable_prefixes_are_canonical(self):
        medium_single = (CONTROLLERS_DIR / "worldgen_medium_supervisor" / "worldgen_medium_supervisor.py").read_text(encoding="utf-8")
        medium_eval = (CONTROLLERS_DIR / "worldgen_medium_eval_supervisor" / "worldgen_medium_eval_supervisor.py").read_text(encoding="utf-8")
        medium_dataset = (CONTROLLERS_DIR / "worldgen_medium_dataset_supervisor" / "worldgen_medium_dataset_supervisor.py").read_text(encoding="utf-8")
        easy_single = (CONTROLLERS_DIR / "worldgen_easy_supervisor" / "worldgen_easy_supervisor.py").read_text(encoding="utf-8")
        easy_eval = (CONTROLLERS_DIR / "worldgen_easy_eval_supervisor" / "worldgen_easy_eval_supervisor.py").read_text(encoding="utf-8")

        self.assertNotIn("TABLE_SCENARIO_", medium_single)
        self.assertNotIn("TABLE_EVAL_", medium_eval)
        self.assertNotIn("TABLE_DATASET_", medium_dataset)
        self.assertNotIn("BASE_SCENARIO_", easy_single)
        self.assertNotIn("BASE_SCENARIO_", easy_eval)
        self.assertIn("MEDIUM_SCENARIO_", medium_single)
        self.assertIn("MEDIUM_EVAL_", medium_eval)
        self.assertIn("MEDIUM_DATASET_", medium_dataset)
        self.assertIn("EASY_SCENARIO_", easy_single)
        self.assertIn("EASY_SCENARIO_", easy_eval)

    def test_all_eval_state_writes_preserve_campaign_identity(self):
        import ast

        for environment in ("easy", "medium", "hard"):
            path = (
                CONTROLLERS_DIR
                / f"worldgen_{environment}_eval_supervisor"
                / f"worldgen_{environment}_eval_supervisor.py"
            )
            tree = ast.parse(path.read_text(encoding="utf-8"))
            persisted_state_dicts = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if not any(isinstance(target, ast.Name) and target.id == "eval_state" for target in node.targets):
                    continue
                if not isinstance(node.value, ast.Dict):
                    continue
                keys = {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                if "completed" in keys:
                    persisted_state_dicts.append((node.lineno, keys))

            self.assertGreaterEqual(len(persisted_state_dicts), 3, environment)
            for lineno, keys in persisted_state_dicts:
                with self.subTest(environment=environment, lineno=lineno):
                    self.assertIn("scenario", keys)
                    self.assertIn("environment", keys)
                    self.assertIn("run_mode", keys)


if __name__ == "__main__":
    unittest.main()
