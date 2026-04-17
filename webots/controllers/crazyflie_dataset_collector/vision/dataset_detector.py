# vision/dataset_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _call_method(obj: Any, names: List[str], *args):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)(*args)
    raise AttributeError(f"None of these methods exist on {type(obj).__name__}: {names}")


def _try_call_method(obj: Any, names: List[str], *args):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)(*args)
    return None


@dataclass
class DatasetDetector:
    """
    Reads Webots camera recognition outputs and converts them to YOLO annotations.

    The primary source of truth is CameraRecognitionObject:
      - id
      - model
      - position_on_image (pixel center)
      - size_on_image (pixel width/height)

    The controller uses this class to:
      - enable recognition
      - fetch visible objects
      - convert them into clipped pixel boxes
      - convert them into normalized YOLO labels
    """

    camera: Any
    model_to_class: Dict[str, int]
    min_box_pixels: int = 4
    enable_segmentation_debug: bool = False

    def enable(self, sampling_period_ms: int) -> None:
        has_recognition = _try_call_method(
            self.camera,
            ["hasRecognition", "has_recognition"],
        )
        if not has_recognition:
            raise RuntimeError(
                "Camera has no Recognition node. Add a Recognition node in the Camera.recognition field."
            )

        _call_method(
            self.camera,
            ["recognitionEnable", "enableRecognition", "recognition_enable", "enable_recognition"],
            sampling_period_ms,
        )

        if self.enable_segmentation_debug:
            has_seg = _try_call_method(
                self.camera,
                ["hasRecognitionSegmentation", "has_recognition_segmentation"],
            )
            if has_seg:
                _call_method(
                    self.camera,
                    [
                        "enableRecognitionSegmentation",
                        "recognitionEnableSegmentation",
                        "enable_recognition_segmentation",
                        "recognition_enable_segmentation",
                    ],
                )

    def get_image_size(self) -> tuple[int, int]:
        width = _call_method(self.camera, ["getWidth", "get_width"])
        height = _call_method(self.camera, ["getHeight", "get_height"])
        return int(width), int(height)

    def get_annotations(self) -> List[Dict]:
        width, height = self.get_image_size()

        objects = _call_method(
            self.camera,
            ["getRecognitionObjects", "get_recognition_objects"],
        )

        annotations: List[Dict] = []

        for obj in objects:
            model = _try_call_method(obj, ["getModel", "get_model"])
            if model not in self.model_to_class:
                continue

            obj_id = _try_call_method(obj, ["getId", "get_id"])
            pos_img = _try_call_method(obj, ["getPositionOnImage", "get_position_on_image"])
            size_img = _try_call_method(obj, ["getSizeOnImage", "get_size_on_image"])
            rel_pos = _try_call_method(obj, ["getPosition", "get_position"])
            rel_size = _try_call_method(obj, ["getSize", "get_size"])

            if pos_img is None or size_img is None:
                continue

            cx_px = float(pos_img[0])
            cy_px = float(pos_img[1])
            w_px = float(size_img[0])
            h_px = float(size_img[1])

            if w_px < self.min_box_pixels or h_px < self.min_box_pixels:
                continue

            x1 = cx_px - 0.5 * w_px
            y1 = cy_px - 0.5 * h_px
            x2 = cx_px + 0.5 * w_px
            y2 = cy_px + 0.5 * h_px

            # Clip to image bounds
            x1 = max(0.0, min(x1, width - 1.0))
            y1 = max(0.0, min(y1, height - 1.0))
            x2 = max(0.0, min(x2, width - 1.0))
            y2 = max(0.0, min(y2, height - 1.0))

            clipped_w = x2 - x1
            clipped_h = y2 - y1
            if clipped_w < self.min_box_pixels or clipped_h < self.min_box_pixels:
                continue

            cx_clip = 0.5 * (x1 + x2)
            cy_clip = 0.5 * (y1 + y2)

            ann = {
                "id": None if obj_id is None else int(obj_id),
                "model": str(model),
                "class_id": int(self.model_to_class[model]),
                "bbox_xyxy_px": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_xywh_px": [float(cx_clip), float(cy_clip), float(clipped_w), float(clipped_h)],
                "bbox_xywh_norm": [
                    float(cx_clip / width),
                    float(cy_clip / height),
                    float(clipped_w / width),
                    float(clipped_h / height),
                ],
                "camera_relative_position_m": None if rel_pos is None else [float(v) for v in rel_pos],
                "camera_relative_size_m": None if rel_size is None else [float(v) for v in rel_size],
            }
            annotations.append(ann)

        return annotations

    @staticmethod
    def yolo_lines(annotations: List[Dict]) -> List[str]:
        lines = []
        for ann in annotations:
            c = ann["class_id"]
            cx, cy, w, h = ann["bbox_xywh_norm"]
            lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines

    @staticmethod
    def save_yolo_txt(label_path: str, annotations: List[Dict]) -> None:
        lines = DatasetDetector.yolo_lines(annotations)
        with open(label_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

    def save_segmentation_debug(self, path: str, quality: int = 100) -> bool:
        """
        Optional helper for QA/debug only.
        Returns False if the Python API method is not available.
        """
        result = _try_call_method(
            self.camera,
            [
                "saveRecognitionSegmentationImage",
                "recognitionSaveSegmentationImage",
                "save_recognition_segmentation_image",
                "recognition_save_segmentation_image",
            ],
            path,
            quality,
        )
        return result is not None