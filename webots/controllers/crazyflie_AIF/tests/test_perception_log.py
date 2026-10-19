import json
import os
import sys
import tempfile
import unittest

CONTROLLER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, CONTROLLER_DIR)

from utils.perception_log import PERCEPTION_LOG_FILENAME, PerceptionFrameLogger


class PerceptionFrameLoggerTest(unittest.TestCase):
    def test_disabled_logger_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = PerceptionFrameLogger(enabled=False, run_dir=tmp)
            logger.write_header({"seed": 1})
            logger.close()
            self.assertFalse(os.path.exists(os.path.join(tmp, PERCEPTION_LOG_FILENAME)))

    def test_frame_record_uses_cxcy_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = PerceptionFrameLogger(enabled=True, run_dir=tmp)
            logger.write_header({"seed": 7})
            logger.log_frame(
                time_s=20.0,
                step=100,
                autonomous=True,
                x=0.1,
                y=-0.2,
                z=1.5,
                yaw=0.3,
                floor_coverages_rc={(2, 5): 1.0, (2, 6): 0.4},
                table_coverages_rc={(2, 5): 0.5},
                merged_coverages_rc={(2, 5): 0.5, (2, 6): 0.4},
                table_projection_cells_cxcy={(5, 2)},
                persistent_table_cells_cxcy={(5, 2)},
                detection_events_rc=[{
                    "class_name": "target_can",
                    "class_id": 1,
                    "confidence": 0.91,
                    "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
                    "cell_rc": (2, 5),
                    "cell_rc_floor": (2, 6),
                    "cell_rc_table": (2, 5),
                    "surface_assumption": "table",
                    "bbox_overlaps_desk": True,
                }],
            )
            logger.close()

            with open(os.path.join(tmp, PERCEPTION_LOG_FILENAME), encoding="utf-8") as f:
                header, frame = [json.loads(line) for line in f]

        self.assertEqual(header, {"type": "header", "seed": 7})
        self.assertEqual(frame["cells"], [[5, 2, 0.5, 1.0, 0.5], [6, 2, 0.4, 0.4, 0.0]])
        self.assertEqual(frame["table_cells"], [[5, 2]])
        det = frame["detections"][0]
        self.assertEqual(det["cell"], [5, 2])
        self.assertEqual(det["cell_floor"], [6, 2])
        self.assertEqual(det["cell_table"], [5, 2])
        self.assertTrue(det["bbox_overlaps_desk"])


if __name__ == "__main__":
    unittest.main()
