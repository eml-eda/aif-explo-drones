from __future__ import annotations

import sys
import unittest
from pathlib import Path

CONTROLLERS_DIR = Path(__file__).resolve().parents[2]
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from dataset_collection.class_config import (
    get_dataset_class_config,
    normalize_collection_environment,
)
from dataset_collection.collector import _object_records_by_id


class DatasetClassConfigTests(unittest.TestCase):
    def test_easy_has_two_foreground_classes(self):
        cfg = get_dataset_class_config("easy")
        self.assertEqual(cfg.class_id_to_name, {0: "target_bottle", 1: "target_can"})
        self.assertIsNone(cfg.desk_class_name)

    def test_medium_and_hard_share_three_class_schema(self):
        expected = {0: "target_bottle", 1: "target_can", 2: "desk"}
        self.assertEqual(get_dataset_class_config("medium").class_id_to_name, expected)
        self.assertEqual(get_dataset_class_config("hard").class_id_to_name, expected)

    def test_hard_distractors_are_not_labelable(self):
        cfg = get_dataset_class_config("hard")
        payload = {
            "objects": [
                {"id": 1, "class_name": "target_bottle", "model": "target_bottle"},
                {"id": 2, "class_name": "desk", "model": "desk"},
                {"id": 3, "class_name": "distractor_plate", "model": "plate"},
            ]
        }
        self.assertEqual(set(_object_records_by_id(payload, cfg)), {1, 2})

    def test_legacy_environment_aliases_are_rejected(self):
        for value in ("mixed_easy", "table", "tables", "mixed_tables", "mixed_hard"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_collection_environment(value)


if __name__ == "__main__":
    unittest.main()
