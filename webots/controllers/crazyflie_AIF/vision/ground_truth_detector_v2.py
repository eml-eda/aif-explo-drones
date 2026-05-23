# webots/controllers/crazyflie_AIF/vision/ground_truth_detector_v2.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from env.env_manager import GridSpec, Cell


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
class CameraRecognitionGroundTruthDetector:
    """
    Ground-truth detector based on Webots Camera Recognition.

    Source of visibility:
      Webots camera recognition output.

    Source of true class/cell:
      spawned_objects.json, matched by Webots recognition object id.

    Output format matches the old GroundTruthDetector:
      Dict[(row, col)] = {
          "confidence": float,
          "class_id": int,
          "label": str,
          "class_name": str,
      }
    """

    camera: Any
    objects: List[dict]
    grid_spec: GridSpec
    confidence: float = 1.0
    model_to_class: Optional[Dict[str, int]] = None
    min_box_pixels: int = 2
    require_known_id: bool = True

    def __post_init__(self) -> None:
        self.model_to_class = self.model_to_class or {
            "target_bottle": 0,
            "target_can": 1,
        }

        self.objects_by_id: Dict[int, dict] = {}
        for obj in self.objects:
            obj_id = obj.get("id", None)
            if obj_id is not None:
                self.objects_by_id[int(obj_id)] = obj

        self._warned_missing_ids = False
        self._warned_unknown_id = False

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
            [
                "recognitionEnable",
                "enableRecognition",
                "recognition_enable",
                "enable_recognition",
            ],
            sampling_period_ms,
        )

    def process(
        self,
        *,
        drone_x: float = 0.0,
        drone_y: float = 0.0,
        drone_yaw: float = 0.0,
    ) -> Dict[Cell, dict]:
        # drone pose is accepted only to keep the old detector interface.
        # This detector does not use the analytic drone FoV polygon.
        recognition_objects = _call_method(
            self.camera,
            ["getRecognitionObjects", "get_recognition_objects"],
        )

        det_out: Dict[Cell, dict] = {}

        for rec_obj in recognition_objects:
            model = _try_call_method(rec_obj, ["getModel", "get_model"])
            if model is None:
                continue

            model = str(model)
            if model not in self.model_to_class:
                continue

            pos_img = _try_call_method(
                rec_obj,
                ["getPositionOnImage", "get_position_on_image"],
            )
            size_img = _try_call_method(
                rec_obj,
                ["getSizeOnImage", "get_size_on_image"],
            )

            # Recognition says the object exists, but for safety ignore tiny or
            # invalid image boxes.
            if pos_img is None or size_img is None:
                continue

            w_px = float(size_img[0])
            h_px = float(size_img[1])
            if w_px < self.min_box_pixels or h_px < self.min_box_pixels:
                continue

            rec_id = _try_call_method(rec_obj, ["getId", "get_id"])
            if rec_id is None:
                if not self._warned_missing_ids:
                    print(
                        "[GTv2] recognition object has no id; cannot map it to spawned_objects.json",
                        flush=True,
                    )
                    self._warned_missing_ids = True
                continue

            rec_id = int(rec_id)
            manifest_obj = self.objects_by_id.get(rec_id, None)

            if manifest_obj is None:
                if self.require_known_id:
                    if not self._warned_unknown_id:
                        print(
                            f"[GTv2] recognition object id={rec_id} not found in spawned_objects.json; "
                            "skipping unknown recognized objects.",
                            flush=True,
                        )
                        self._warned_unknown_id = True
                    continue

                class_id = int(self.model_to_class[model])
                class_name = model
                cell_rc = None
            else:
                class_id = int(manifest_obj.get("class_id", self.model_to_class[model]))
                class_name = str(manifest_obj.get("class_name", model))

                if manifest_obj.get("cx", None) is not None and manifest_obj.get("cy", None) is not None:
                    # stored as cx,cy but controller expects row,col internally
                    cell_rc = (int(manifest_obj["cy"]), int(manifest_obj["cx"]))
                else:
                    x = float(manifest_obj["x"])
                    y = float(manifest_obj["y"])
                    cell_rc = self.grid_spec.rowcol_from_xy(x, y)

            if cell_rc is None:
                continue

            prev = det_out.get(cell_rc, None)
            if prev is None or self.confidence > float(prev["confidence"]):
                det_out[cell_rc] = {
                    "confidence": float(self.confidence),
                    "class_id": int(class_id),
                    "label": str(class_name),
                    "class_name": str(class_name),
                    "recognition_id": int(rec_id),
                    "model": str(model),
                    "bbox_center_px": [float(pos_img[0]), float(pos_img[1])],
                    "bbox_size_px": [float(size_img[0]), float(size_img[1])],
                }

        return det_out