from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO


TARGET_CLASS_NAMES = {"target_bottle", "target_can"}
DESK_CLASS_NAME = "desk"


@dataclass
class DetectorOutput:
    detections: List[Dict[str, Any]]
    timestamp: Optional[float] = None
    raw: Any = None


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
    """YOLO detector used by medium and hard environments.

    The wrapper keeps the existing RGB conversion and context-event output
    semantics, but checkpoint/class selection is now owned by central config.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
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
            for v in (class_names or ["target_bottle", "target_can", "desk"])
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

        print(
            f"[YOLO CONTEXT] model={self.model_name} device={self.device} "
            f"classes={dict(enumerate(self.class_names))} "
            f"conf={self.conf:.3f} iou={self.iou:.3f} imgsz={self.img_size}",
            flush=True,
        )

    def _label_for_class_id(self, cls_id: int) -> str:
        if 0 <= int(cls_id) < len(self.class_names):
            return self.class_names[int(cls_id)]
        return str(cls_id)

    @staticmethod
    def _webots_image_to_pil_rgb(image, *, width: int, height: int) -> Image.Image:
        # Preserve the previous table/hard conversion path exactly.
        try:
            from controller import Camera  # type: ignore
        except Exception as exc:  # pragma: no cover - Webots runtime only
            raise RuntimeError("Could not import Webots Camera helper functions.") from exc

        w = int(width)
        h = int(height)
        arr = np.empty((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                arr[y, x, 0] = int(Camera.imageGetRed(image, w, x, y))
                arr[y, x, 1] = int(Camera.imageGetGreen(image, w, x, y))
                arr[y, x, 2] = int(Camera.imageGetBlue(image, w, x, y))
        return Image.fromarray(arr, mode="RGB")

    def process(
        self,
        image,
        *,
        width: int,
        height: int,
        timestamp: Optional[float] = None,
    ) -> DetectorOutput:
        pil_img = self._webots_image_to_pil_rgb(
            image,
            width=int(width),
            height=int(height),
        )

        results = self.model.predict(
            source=pil_img,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results:
            return DetectorOutput(detections=detections, timestamp=timestamp, raw=results)

        result0 = results[0]
        boxes = getattr(result0, "boxes", None)
        if boxes is None:
            return DetectorOutput(detections=detections, timestamp=timestamp, raw=results)

        for i, box in enumerate(boxes):
            try:
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
                score = float(box.conf[0].detach().cpu().item())
                cls_id = int(box.cls[0].detach().cpu().item())
            except Exception:
                continue

            if not (0 <= cls_id < len(self.class_names)):
                continue

            x1, y1, x2, y2 = [float(v) for v in xyxy]
            x1 = max(0.0, min(float(width), x1))
            x2 = max(0.0, min(float(width), x2))
            y1 = max(0.0, min(float(height), y1))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                continue

            label = self._label_for_class_id(cls_id)
            detections.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [
                        0.5 * (x1 + x2),
                        0.5 * (y1 + y2),
                        x2 - x1,
                        y2 - y1,
                    ],
                    "confidence": score,
                    "class_id": cls_id,
                    "label": label,
                    "class_name": label,
                    "is_target": label in TARGET_CLASS_NAMES,
                    "is_desk": label == DESK_CLASS_NAME,
                    "timestamp": None if timestamp is None else float(timestamp),
                    "detector": "yolo_context",
                    "index": int(i),
                }
            )

        return DetectorOutput(detections=detections, timestamp=timestamp, raw=results)
