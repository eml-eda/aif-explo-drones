from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO


class DetectorOutput:
    def __init__(self, detections=None, embedding=None, raw=None):
        self.detections = detections or []
        self.embedding = embedding
        self.raw = raw


def _normalize_label(label: str) -> str:
    aliases = {
        "bottle": "target_bottle",
        "can": "target_can",
        "table": "desk",
        "tabletop": "desk",
        "desk_surface": "desk",
    }
    value = str(label).strip()
    return aliases.get(value, value)


def _ordered_model_names(names) -> List[str]:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda v: int(v))]
    if isinstance(names, (list, tuple)):
        return [str(v) for v in names]
    return []


class Detector:
    """YOLO detector used by the easy environment.

    Checkpoint selection belongs to the central configuration layer. This
    wrapper only loads the supplied model and validates that its output classes
    match the environment class schema before inference begins.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cuda:0",
        conf: float = 0.45,
        iou: float = 0.40,
        img_size: int = 320,
        class_names: Optional[List[str]] = None,
        expected_model_size: Optional[str] = None,
    ):
        if not model_name:
            raise ValueError(
                "YOLO checkpoint path was not supplied. Select it through "
                "configuration/model_registry.py or AIF_MODEL_PATH."
            )

        self.model_name = str(Path(model_name).expanduser())
        self.device = str(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.img_size = int(img_size)
        self.class_names = [
            _normalize_label(v)
            for v in (class_names or ["target_bottle", "target_can"])
        ]

        self.model = YOLO(self.model_name)
        raw_model_names = _ordered_model_names(getattr(self.model, "names", {}))
        normalized_model_names = [_normalize_label(v) for v in raw_model_names]

        if normalized_model_names != self.class_names:
            raise ValueError(
                "YOLO class schema mismatch.\n"
                f"  checkpoint: {normalized_model_names}\n"
                f"  environment: {self.class_names}\n"
                f"  checkpoint path: {self.model_name}"
            )

        expected_scale = {"nano": "n"}.get(
            None if expected_model_size is None else str(expected_model_size).strip().lower()
        )
        model_yaml = getattr(getattr(self.model, "model", None), "yaml", {}) or {}
        actual_scale = str(model_yaml.get("scale", "")).strip().lower() if isinstance(model_yaml, dict) else ""
        if expected_scale and actual_scale and actual_scale != expected_scale:
            raise ValueError(
                "YOLO model-size mismatch.\n"
                f"  expected profile size: {expected_model_size} ({expected_scale})\n"
                f"  checkpoint model scale: {actual_scale}\n"
                f"  checkpoint path: {self.model_name}"
            )
        if expected_scale and not actual_scale:
            print(
                f"[YOLO] WARNING: checkpoint did not expose a model scale; "
                f"could not verify expected size '{expected_model_size}'.",
                flush=True,
            )

        self.classes = list(range(len(self.class_names)))

        print(
            f"[YOLO] model={self.model_name} device={self.device} "
            f"classes={dict(enumerate(self.class_names))} "
            f"conf={self.conf:.3f} iou={self.iou:.3f} imgsz={self.img_size}",
            flush=True,
        )

    @staticmethod
    def _image_from_webots(image, width, height):
        if image is None or width is None or height is None:
            return None

        frame = np.frombuffer(image, dtype=np.uint8)
        expected = int(width * height * 4)
        if frame.size != expected:
            return None

        frame = frame.reshape((height, width, 4))
        return frame[:, :, :3]  # Webots BGRA -> BGR for Ultralytics/OpenCV.

    def process(self, image, width=None, height=None, timestamp=None):
        frame = self._image_from_webots(image, width, height)
        if frame is None:
            return DetectorOutput(detections=[], embedding=None, raw=None)

        results = self.model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if results:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                clss = boxes.cls.detach().cpu().numpy().astype(int)

                for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss):
                    if not (0 <= int(cls_id) < len(self.class_names)):
                        continue
                    class_name = self.class_names[int(cls_id)]
                    detections.append(
                        {
                            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(score),
                            "class_id": int(cls_id),
                            "label": class_name,
                            "class_name": class_name,
                        }
                    )

        return DetectorOutput(detections=detections, embedding=None, raw=results)
