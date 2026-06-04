# webots/controllers/crazyflie_AIF/vision/detector_yolo_tables.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO

import config as aif_config


DEFAULT_TABLE_MODEL = (
    "/home/ale/dev/thesis_project/aif-explo-drones/"
    "detection_model_training/yolo11_s_mixed_tables/"
    "mixed_table_detector_runs/yolo11s_ft_bottle_can_table_v1/weights/best.pt"
)

TABLE_CLASS_NAMES = ["target_bottle", "target_can", "desk"]
TARGET_CLASS_NAMES = {"target_bottle", "target_can"}
DESK_CLASS_NAME = "desk"


@dataclass
class DetectorOutput:
    detections: List[Dict[str, Any]]
    timestamp: Optional[float] = None
    raw: Any = None


def _normalize_label(label: str) -> str:
    """Normalize the few class-name aliases that may come from training YAMLs."""
    label = str(label).strip()
    aliases = {
        "bottle": "target_bottle",
        "can": "target_can",
        "table": "desk",
        "tabletop": "desk",
        "desk_surface": "desk",
    }
    return aliases.get(label, label)


class Detector:
    """
    YOLO detector for the mixed-table scenario.

    Output contract matches the existing controller expectations:
      Detector.process(...) -> object with .detections list

    Each detection dictionary includes at least:
      bbox_xyxy, confidence, class_id, label, class_name

    Unlike the old easy-scenario detector, this version keeps the desk class in
    the output. The controller/policy decide which classes are target evidence
    and which classes are only surface/table evidence.
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
    ):
        self.model_name = str(
            model_name
            or os.environ.get("AIF_YOLO_TABLES_MODEL")
            or os.environ.get("AIF_YOLO_MODEL")
            or getattr(aif_config, "AIF_YOLO_MODEL", DEFAULT_TABLE_MODEL)
            or DEFAULT_TABLE_MODEL
        )
        self.device = str(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.img_size = int(img_size)
        # The current controller may still pass the old easy-scenario
        # class_names=["target_bottle", "target_can"]. Do not let that hide
        # class id 2; the table detector must expose desk as well.
        if class_names is None or len(class_names) < len(TABLE_CLASS_NAMES):
            self.class_names = list(TABLE_CLASS_NAMES)
        else:
            self.class_names = [_normalize_label(v) for v in class_names]

        print(
            f"[YOLO TABLE DETECTOR] model={self.model_name} device={self.device} "
            f"conf={self.conf:.3f} iou={self.iou:.3f} imgsz={self.img_size}",
            flush=True,
        )
        self.model = YOLO(self.model_name)

    def _label_for_class_id(self, cls_id: int) -> str:
        # Prefer explicit controller-provided class_names because the table
        # detector is trained with target_bottle/target_can/desk ids.
        if 0 <= int(cls_id) < len(self.class_names):
            return _normalize_label(self.class_names[int(cls_id)])

        names = getattr(self.model, "names", None)
        if isinstance(names, dict) and int(cls_id) in names:
            return _normalize_label(str(names[int(cls_id)]))
        if isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
            return _normalize_label(str(names[int(cls_id)]))
        return str(cls_id)

    @staticmethod
    def _webots_image_to_pil_rgb(image, *, width: int, height: int) -> Image.Image:
        """
        Convert the Webots camera image to a PIL RGB image.

        Webots' raw camera buffer is easy to misread directly, so use the
        Camera.imageGetRed/Green/Blue helpers. This is only 320x240 in this
        setup and runs at detector cadence, so correctness is preferred here.
        """
        try:
            from controller import Camera  # type: ignore
        except Exception as exc:  # pragma: no cover - only in Webots runtime
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

    def process(self, image, *, width: int, height: int, timestamp: Optional[float] = None) -> DetectorOutput:
        pil_img = self._webots_image_to_pil_rgb(image, width=int(width), height=int(height))

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
                conf = float(box.conf[0].detach().cpu().item())
                cls_id = int(box.cls[0].detach().cpu().item())
            except Exception:
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
                    "bbox_xywh": [0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "label": label,
                    "class_name": label,
                    "is_target": bool(label in TARGET_CLASS_NAMES),
                    "is_desk": bool(label == DESK_CLASS_NAME),
                    "timestamp": None if timestamp is None else float(timestamp),
                    "detector": "yolo_tables",
                    "index": int(i),
                }
            )

        return DetectorOutput(detections=detections, timestamp=timestamp, raw=results)