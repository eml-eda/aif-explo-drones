import unittest

import numpy as np

from env.env_manager import GridSpec
from vision.ground_truth_detector import GroundTruthDetector


class _FakeSegmentationCamera:
    def __init__(self, image: bytes, width: int, height: int):
        self._image = image
        self._width = width
        self._height = height
        self.recognition_period = None
        self.segmentation_enabled = False

    def hasRecognition(self):
        return True

    def recognitionEnable(self, sampling_period_ms):
        self.recognition_period = int(sampling_period_ms)

    def recognitionEnableSegmentation(self):
        self.segmentation_enabled = True

    def getRecognitionSegmentationImage(self):
        return self._image

    def getWidth(self):
        return self._width

    def getHeight(self):
        return self._height


class _FakeCameraWithoutSegmentation:
    def hasRecognition(self):
        return True

    def recognitionEnable(self, sampling_period_ms):
        self.recognition_period = int(sampling_period_ms)


class GroundTruthDetectorTests(unittest.TestCase):
    @staticmethod
    def grid():
        return GridSpec(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            cell_size=0.5,
            rows=4,
            cols=4,
        )

    def test_segmentation_pixels_emit_manifest_cell_and_bbox(self):
        width, height = 4, 3
        bgra = np.zeros((height, width, 4), dtype=np.uint8)
        bgra[:, :, 3] = 255
        # Two red recognition pixels: RGB=(255,0,0) -> BGRA=(0,0,255,255).
        bgra[1, 1] = [0, 0, 255, 255]
        bgra[1, 2] = [0, 0, 255, 255]

        camera = _FakeSegmentationCamera(bgra.tobytes(), width, height)
        detector = GroundTruthDetector(
            camera=camera,
            objects=[
                {
                    "id": 12,
                    "cx": 3,
                    "cy": 2,
                    "class_id": 0,
                    "class_name": "target_bottle",
                    "model": "target_bottle",
                    "recognition_color_rgb": [1.0, 0.0, 0.0],
                }
            ],
            grid_spec=self.grid(),
            class_name_to_id={"target_bottle": 0, "target_can": 1},
            min_box_pixels=1,
            color_tolerance=0,
        )
        detector.enable(640)
        events = detector.process()

        self.assertTrue(camera.segmentation_enabled)
        self.assertEqual(camera.recognition_period, 640)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["cell_rc"], (2, 3))
        self.assertEqual(events[0]["bbox_xyxy"], [1.0, 1.0, 3.0, 2.0])
        self.assertEqual(events[0]["mask_area_px"], 2)
        self.assertEqual(events[0]["detector"], "ground_truth")

    def test_missing_segmentation_support_is_fatal(self):
        detector = GroundTruthDetector(
            camera=_FakeCameraWithoutSegmentation(),
            objects=[],
            grid_spec=self.grid(),
            class_name_to_id={"target_bottle": 0, "target_can": 1},
        )
        with self.assertRaisesRegex(RuntimeError, "recognition segmentation"):
            detector.enable(640)

    def test_process_before_enable_is_fatal(self):
        camera = _FakeSegmentationCamera(b"", 1, 1)
        detector = GroundTruthDetector(
            camera=camera,
            objects=[],
            grid_spec=self.grid(),
            class_name_to_id={"target_bottle": 0, "target_can": 1},
        )
        with self.assertRaisesRegex(RuntimeError, "before recognition segmentation was enabled"):
            detector.process()


if __name__ == "__main__":
    unittest.main()
