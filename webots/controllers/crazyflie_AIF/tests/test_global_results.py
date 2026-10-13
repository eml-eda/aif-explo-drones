from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CONTROLLERS_DIR = Path(__file__).resolve().parents[2]
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from common.global_results import build_global_result_entry, update_global_result


def _report() -> dict:
    return {
        "policy": "aif_binary",
        "detector_used": "mobilenet_ssd",
        "run_metrics": [
            {
                "reason": "done",
                "elapsed_s": 10.0,
                "detection_rate_percent": 100.0,
                "coverage_ratio": 0.1,
            },
            {
                "reason": "timeout",
                "elapsed_s": 180.0,
                "detection_rate_percent": 50.0,
                "coverage_ratio": 0.3,
            },
            {
                "reason": "done",
                "elapsed_s": 20.0,
                "detection_rate_percent": 100.0,
                "coverage_ratio": 0.2,
            },
        ],
    }


class GlobalResultsTests(unittest.TestCase):
    def test_builds_compact_metrics_with_success_only_time(self):
        entry = build_global_result_entry(
            campaign_id="eval_aif_binary_mbnetv2_easy_20260921_142313",
            environment="easy",
            eval_report=_report(),
        )

        self.assertEqual(entry["num_runs"], 3)
        self.assertEqual(entry["success_rate_percent"], 66.6667)
        self.assertEqual(
            entry["target_recovery_percent"],
            {"mean": 83.3333, "std": 28.8675},
        )
        self.assertEqual(
            entry["time_to_success_s"],
            {"mean": 15.0, "std": 7.0711},
        )
        self.assertEqual(
            entry["cell_visited_percent"],
            {"mean": 20.0, "std": 10.0},
        )

    def test_update_preserves_existing_results_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "global_results.json"
            path.write_text(
                json.dumps({"results": [{"campaign_id": "manual", "num_runs": 100}]}),
                encoding="utf-8",
            )

            update_global_result(
                path,
                campaign_id="campaign-two",
                environment="easy",
                eval_report=_report(),
            )
            update_global_result(
                path,
                campaign_id="campaign-two",
                environment="easy",
                eval_report=_report(),
            )

            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(document["results"]), 2)
            self.assertEqual(document["results"][0]["campaign_id"], "manual")
            self.assertEqual(document["results"][1]["campaign_id"], "campaign-two")
            self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_rejects_an_unexpected_existing_document_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "global_results.json"
            path.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaises(ValueError):
                update_global_result(
                    path,
                    campaign_id="campaign",
                    environment="easy",
                    eval_report=_report(),
                )


if __name__ == "__main__":
    unittest.main()
