from __future__ import annotations

import sys
import unittest
from pathlib import Path

CONTROLLERS_DIR = Path(__file__).resolve().parents[2]
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from dataset_collection.segmentation_annotator import SegmentationAnnotator


class _NoSegmentationCamera:
    def hasRecognition(self):
        return True

    def recognitionEnable(self, _period):
        return None

    def hasRecognitionSegmentation(self):
        return False


class _PixelAnnotator(SegmentationAnnotator):
    def _segmentation_pixels(self, width, height):
        for y in range(height):
            for x in range(width):
                rgb = (255, 0, 0) if (1 <= x <= 3 and 2 <= y <= 4) else (0, 0, 0)
                yield x, y, rgb


class SegmentationAnnotatorTests(unittest.TestCase):
    def test_missing_segmentation_is_fatal(self):
        annotator = SegmentationAnnotator(
            camera=_NoSegmentationCamera(),
            model_to_class={"target_bottle": 0},
        )
        with self.assertRaisesRegex(RuntimeError, "segmentation is not available"):
            annotator.enable(32)

    def test_segmentation_mask_produces_expected_bbox_and_yolo_line(self):
        annotator = _PixelAnnotator(
            camera=object(),
            model_to_class={"target_bottle": 0},
            class_id_to_name={0: "target_bottle"},
            min_box_pixels=2,
            min_target_mask_pixels=1,
        )
        infos = [
            {
                "object_key": 7,
                "class_id": 0,
                "class_name": "target_bottle",
                "model": "target_bottle",
                "candidate_colors_rgb_tuples": [(255, 0, 0)],
            }
        ]
        boxes = annotator._segmentation_bboxes_for_objects(infos, width=8, height=8)
        self.assertEqual(boxes[7]["bbox_xyxy_px"], [1.0, 2.0, 4.0, 5.0])
        self.assertEqual(boxes[7]["mask_area_px"], 9)

        lines = annotator.yolo_lines(
            [{"class_id": 0, "bbox_xywh_norm": [0.5, 0.5, 0.25, 0.5]}]
        )
        self.assertEqual(lines, ["0 0.500000 0.500000 0.250000 0.500000"])


if __name__ == "__main__":
    unittest.main()
