from __future__ import annotations

import sys
import unittest
from pathlib import Path

CONTROLLERS_DIR = Path(__file__).resolve().parents[2]
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from common.experiment_naming import evaluation_directory_name


class ExperimentNamingTests(unittest.TestCase):
    def test_detector_labels_are_included_in_campaign_directories(self):
        expected = {
            "mobilenet_ssd": "eval_spiral_mbnetv2_easy_20260921_142313",
            "yolo_nano": "eval_spiral_yolonano_easy_20260921_142313",
        }
        for detector, directory in expected.items():
            with self.subTest(detector=detector):
                self.assertEqual(
                    evaluation_directory_name(
                        policy="spiral",
                        detector=detector,
                        environment="easy",
                        timestamp="20260921_142313",
                    ),
                    directory,
                )

    def test_unknown_detector_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluation_directory_name(
                policy="spiral",
                detector="unknown",
                environment="easy",
                timestamp="20260921_142313",
            )


if __name__ == "__main__":
    unittest.main()
