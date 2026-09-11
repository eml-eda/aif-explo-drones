from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

CONTROLLERS_DIR = Path(__file__).resolve().parents[2]
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from dataset_collection.collection_config import get_collection_config


class DatasetCollectionConfigTests(unittest.TestCase):
    def test_medium_uses_canonical_environment_variable(self):
        cfg = get_collection_config("medium")
        with patch.dict(os.environ, {"MEDIUM_DATASET_CAPTURE_EVERY_N_STEPS": "123"}, clear=True):
            self.assertEqual(cfg.capture_every_n_steps, 123)

    def test_legacy_table_environment_variable_is_not_supported(self):
        cfg = get_collection_config("medium")
        with patch.dict(os.environ, {"TABLE_DATASET_CAPTURE_EVERY_N_STEPS": "123"}, clear=True):
            self.assertEqual(cfg.capture_every_n_steps, cfg.default_capture_every_n_steps)

    def test_dataset_ipc_filenames_are_environment_explicit(self):
        easy = get_collection_config("easy")
        medium = get_collection_config("medium")
        hard = get_collection_config("hard")
        self.assertEqual(easy.experiment_init_filename, "easy_dataset_experiment_init.json")
        self.assertEqual(easy.dataset_objects_filename, "easy_dataset_objects.json")
        self.assertEqual(easy.status_filename, "easy_dataset_task_state.json")
        self.assertEqual(medium.experiment_init_filename, "medium_dataset_experiment_init.json")
        self.assertEqual(medium.dataset_objects_filename, "medium_dataset_objects.json")
        self.assertEqual(medium.status_filename, "medium_dataset_task_state.json")
        self.assertEqual(medium.collection_state_filename, "medium_dataset_collection_state.json")
        self.assertEqual(hard.experiment_init_filename, "hard_dataset_experiment_init.json")


if __name__ == "__main__":
    unittest.main()
