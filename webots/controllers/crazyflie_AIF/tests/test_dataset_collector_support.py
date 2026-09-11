from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


AIF_DIR = Path(__file__).resolve().parents[1]
CONTROLLERS_DIR = AIF_DIR.parent
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from dataset_collection import collector


class DatasetCollectorSupportTests(unittest.TestCase):
    def test_shared_state_uses_canonical_controllers_shared_directory(self):
        self.assertEqual(
            collector.SHARED_STATE.root,
            (CONTROLLERS_DIR / "shared").resolve(),
        )

    def test_runtime_support_modules_resolve_to_crazyflie_aif(self):
        expected = {
            "devices": AIF_DIR / "devices.py",
            "env.env_manager": AIF_DIR / "env" / "env_manager.py",
            "pid.pid_controller": AIF_DIR / "pid" / "pid_controller.py",
            "policies.pseudo_random": AIF_DIR / "policies" / "pseudo_random.py",
            "policies.rotate_and_measure": AIF_DIR / "policies" / "rotate_and_measure.py",
            "policies.spiral": AIF_DIR / "policies" / "spiral.py",
            "policies.wall_following": AIF_DIR / "policies" / "wall_following.py",
        }
        for module_name, expected_path in expected.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(Path(module.__file__).resolve(), expected_path.resolve())

    def test_policy_loader_uses_canonical_seeded_pseudo_random_policy(self):
        policy = collector._load_policy("pseudo_random", seed=17)
        module = importlib.import_module(policy.__class__.__module__)
        self.assertEqual(
            Path(module.__file__).resolve(),
            (AIF_DIR / "policies" / "pseudo_random.py").resolve(),
        )
        self.assertEqual(policy.seed, 17)

    def test_dataset_manifest_jsonl_append_writes_multiple_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "manifest.jsonl"
            collector._append_jsonl(path, {"entry_id": "a", "seed": 1})
            collector._append_jsonl(path, {"entry_id": "b", "seed": 2})

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                rows,
                [
                    {"entry_id": "a", "seed": 1},
                    {"entry_id": "b", "seed": 2},
                ],
            )


if __name__ == "__main__":
    unittest.main()
