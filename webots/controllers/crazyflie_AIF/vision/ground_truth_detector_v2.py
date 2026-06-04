# webots/controllers/crazyflie_AIF/vision/ground_truth_detector_v2.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.env_manager import GridSpec, Cell


def _call_method(obj: Any, names: List[str], *args):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)(*args)
    raise AttributeError(f"None of these methods exist on {type(obj).__name__}: {names}")


def _try_call_method(obj: Any, names: List[str], *args):
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)(*args)
            except TypeError:
                continue
    return None


def _color_to_rgb255(color) -> Optional[Tuple[int, int, int]]:
    """Convert manifest recognition_color_rgb to integer RGB.

    Accepts either normalized floats in [0, 1] or already-255 values.
    """
    if color is None:
        return None
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return None

    vals = [float(color[0]), float(color[1]), float(color[2])]
    if max(vals) <= 1.0:
        vals = [v * 255.0 for v in vals]
    return tuple(max(0, min(255, int(round(v)))) for v in vals)  # type: ignore[return-value]


def _rgb_close(a: Tuple[int, int, int], b: Tuple[int, int, int], tol: int) -> bool:
    return (
        abs(int(a[0]) - int(b[0])) <= int(tol)
        and abs(int(a[1]) - int(b[1])) <= int(tol)
        and abs(int(a[2]) - int(b[2])) <= int(tol)
    )


@dataclass
class CameraRecognitionGroundTruthDetector:
    """
    Ground-truth detector based on Webots Camera Recognition segmentation.

    The old GT detector used getRecognitionObjects() as its visibility gate.
    That can be stricter than a pixel detector for partial/edge views. This
    version instead uses the recognition segmentation image as a pixel-level
    visibility oracle:

      visible colored pixels for manifest object -> emit true manifest cell

    Target events remain oracle-like: the emitted target cell is the true
    spawned_objects.json cell, not a floor/table bbox projection. Desk events
    include bbox_xyxy/bbox_xywh so the controller can use the same multi-cell
    table-surface marking path used by YOLO desk detections.
    """

    camera: Any
    objects: List[dict]
    grid_spec: GridSpec
    confidence: float = 1.0
    model_to_class: Optional[Dict[str, int]] = None
    min_box_pixels: int = 1
    require_known_id: bool = True
    min_mask_pixels: int = 1
    color_tolerance: int = 3
    fallback_to_recognition_objects: bool = True

    def __post_init__(self) -> None:
        self.model_to_class = self.model_to_class or {
            "target_bottle": 0,
            "target_can": 1,
            "desk": 2,
        }

        self.objects_by_id: Dict[int, dict] = {}
        self.objects_by_color: Dict[Tuple[int, int, int], List[dict]] = {}

        for obj in self.objects:
            obj_id = obj.get("id", None)
            if obj_id is not None:
                self.objects_by_id[int(obj_id)] = obj

            class_name = str(obj.get("class_name", obj.get("model", "")))
            model = str(obj.get("model", class_name))
            if class_name not in self.model_to_class and model not in self.model_to_class:
                continue

            rgb = _color_to_rgb255(obj.get("recognition_color_rgb", None))
            if rgb is None:
                continue
            self.objects_by_color.setdefault(rgb, []).append(obj)

        self._warned_missing_ids = False
        self._warned_unknown_id = False
        self._warned_no_segmentation = False
        self._segmentation_enabled = False

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

        # Recognition segmentation is the important part: it gives the actual
        # visible pixels of recognized objects in the camera frame.
        enabled = False
        for name in [
            "recognitionEnableSegmentation",
            "enableRecognitionSegmentation",
            "recognition_enable_segmentation",
            "enable_recognition_segmentation",
        ]:
            if hasattr(self.camera, name):
                try:
                    getattr(self.camera, name)()
                    enabled = True
                    break
                except TypeError:
                    try:
                        getattr(self.camera, name)(sampling_period_ms)
                        enabled = True
                        break
                    except Exception:
                        pass
                except Exception:
                    pass
        self._segmentation_enabled = bool(enabled)

        if not self._segmentation_enabled:
            print(
                "[GTv2] WARNING: could not enable recognition segmentation; "
                "falling back to getRecognitionObjects() if allowed.",
                flush=True,
            )

    def _camera_width_height(self) -> tuple[int, int]:
        w = _call_method(self.camera, ["getWidth", "get_width"])
        h = _call_method(self.camera, ["getHeight", "get_height"])
        return int(w), int(h)

    def _segmentation_image(self):
        return _try_call_method(
            self.camera,
            [
                "getRecognitionSegmentationImage",
                "get_recognition_segmentation_image",
            ],
        )

    @staticmethod
    def _raw_image_to_rgb_array(image, width: int, height: int) -> Optional[np.ndarray]:
        """Convert Webots raw BGRA segmentation image to an RGB uint8 array.

        Webots documents camera/segmentation images as a contiguous 32-bit BGRA
        buffer. Reading this buffer with NumPy avoids 3 * width * height calls to
        Camera.imageGetRed/Green/Blue on every processed frame.
        """
        if image is None:
            return None

        try:
            raw = np.frombuffer(image, dtype=np.uint8)
        except Exception:
            return None

        n = int(raw.size)
        expected4 = int(width) * int(height) * 4
        expected3 = int(width) * int(height) * 3

        try:
            if n >= expected4:
                bgra = raw[:expected4].reshape((int(height), int(width), 4))
                return bgra[:, :, [2, 1, 0]]  # BGRA -> RGB
            if n >= expected3:
                rgb = raw[:expected3].reshape((int(height), int(width), 3))
                return rgb
        except Exception:
            return None

        return None

    @staticmethod
    def _image_rgb_at(image, width: int, x: int, y: int) -> Tuple[int, int, int]:
        # Slow fallback only used if NumPy raw-buffer conversion is unavailable.
        from controller import Camera  # type: ignore

        return (
            int(Camera.imageGetRed(image, width, x, y)),
            int(Camera.imageGetGreen(image, width, x, y)),
            int(Camera.imageGetBlue(image, width, x, y)),
        )

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int]]:
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None
        return (
            int(xs.size),
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        )

    def _cell_rc_for_manifest_obj(self, obj: dict) -> Optional[Cell]:
        if obj.get("cx", None) is not None and obj.get("cy", None) is not None:
            return (int(obj["cy"]), int(obj["cx"]))
        if "x" in obj and "y" in obj:
            return self.grid_spec.rowcol_from_xy(float(obj["x"]), float(obj["y"]))
        return None

    def _event_from_manifest_obj(
        self,
        *,
        obj: dict,
        bbox_xyxy: List[float],
        mask_area_px: int,
        source: str,
    ) -> Optional[dict]:
        cell_rc = self._cell_rc_for_manifest_obj(obj)
        if cell_rc is None:
            return None

        class_name = str(obj.get("class_name", obj.get("model", "unknown")))
        model = str(obj.get("model", class_name))
        class_id = int(obj.get("class_id", self.model_to_class.get(model, self.model_to_class.get(class_name, -1))))
        if class_id < 0:
            return None

        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        return {
            "cell_rc": (int(cell_rc[0]), int(cell_rc[1])),
            "confidence": float(self.confidence),
            "class_id": int(class_id),
            "label": str(class_name),
            "class_name": str(class_name),
            "recognition_id": None if obj.get("id", None) is None else int(obj["id"]),
            "model": str(model),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1],
            "bbox_center_px": [0.5 * (x1 + x2), 0.5 * (y1 + y2)],
            "bbox_size_px": [x2 - x1, y2 - y1],
            "mask_area_px": int(mask_area_px),
            "source": str(source),
            "detector": "ground_truth_recognition_segmentation",
        }

    def _process_from_segmentation(self) -> Optional[List[dict]]:
        img = self._segmentation_image()
        if img is None:
            if not self._warned_no_segmentation:
                print("[GTv2] recognition segmentation image is unavailable.", flush=True)
                self._warned_no_segmentation = True
            return None

        width, height = self._camera_width_height()
        colors = list(self.objects_by_color.keys())
        if not colors:
            return []

        # Fast path: read the raw Webots BGRA segmentation buffer once and use
        # vectorized NumPy masks. This replaces the old 76,800 * 3 Python/Webots
        # channel-helper calls per processed 320x240 frame.
        rgb_arr = self._raw_image_to_rgb_array(img, width, height)

        stats: Dict[Tuple[int, int, int], Dict[str, int]] = {
            rgb: {
                "count": 0,
                "min_x": width,
                "min_y": height,
                "max_x": -1,
                "max_y": -1,
            }
            for rgb in colors
        }

        tol = int(self.color_tolerance)

        if rgb_arr is not None:
            arr = rgb_arr.astype(np.int16, copy=False)

            for rgb in colors:
                target = np.array(rgb, dtype=np.int16)
                if tol <= 0:
                    mask = (arr[:, :, 0] == target[0]) & (arr[:, :, 1] == target[1]) & (arr[:, :, 2] == target[2])
                else:
                    diff = np.abs(arr - target.reshape((1, 1, 3)))
                    mask = (diff[:, :, 0] <= tol) & (diff[:, :, 1] <= tol) & (diff[:, :, 2] <= tol)

                bbox = self._bbox_from_mask(mask)
                if bbox is None:
                    continue

                count, min_x, min_y, max_x, max_y = bbox
                st = stats[rgb]
                st["count"] = int(count)
                st["min_x"] = int(min_x)
                st["min_y"] = int(min_y)
                st["max_x"] = int(max_x)
                st["max_y"] = int(max_y)
        else:
            # Slow fallback for unusual Webots bindings that do not expose the
            # segmentation image as a buffer-compatible Python object.
            for y in range(height):
                for x in range(width):
                    pix = self._image_rgb_at(img, width, x, y)
                    if pix == (0, 0, 0):
                        continue

                    matched = None
                    if pix in stats:
                        matched = pix
                    elif tol > 0:
                        for rgb in colors:
                            if _rgb_close(pix, rgb, tol):
                                matched = rgb
                                break

                    if matched is None:
                        continue

                    st = stats[matched]
                    st["count"] += 1
                    if x < st["min_x"]:
                        st["min_x"] = x
                    if y < st["min_y"]:
                        st["min_y"] = y
                    if x > st["max_x"]:
                        st["max_x"] = x
                    if y > st["max_y"]:
                        st["max_y"] = y

        events: List[dict] = []
        for rgb, st in stats.items():
            count = int(st["count"])
            if count < int(self.min_mask_pixels):
                continue

            x1 = float(st["min_x"])
            y1 = float(st["min_y"])
            # Use +1 so width/height are pixel extents, consistent with image boxes.
            x2 = float(st["max_x"] + 1)
            y2 = float(st["max_y"] + 1)

            if (x2 - x1) < float(self.min_box_pixels) or (y2 - y1) < float(self.min_box_pixels):
                continue

            for obj in self.objects_by_color.get(rgb, []):
                ev = self._event_from_manifest_obj(
                    obj=obj,
                    bbox_xyxy=[x1, y1, x2, y2],
                    mask_area_px=count,
                    source="recognition_segmentation_mask",
                )
                if ev is not None:
                    events.append(ev)

        return events

    def _process_from_recognition_objects(self) -> List[dict]:
        recognition_objects = _call_method(
            self.camera,
            ["getRecognitionObjects", "get_recognition_objects"],
        )

        det_out: List[dict] = []

        for rec_obj in recognition_objects:
            model = _try_call_method(rec_obj, ["getModel", "get_model"])
            if model is None:
                continue

            model = str(model)
            if model not in self.model_to_class:
                continue

            pos_img = _try_call_method(rec_obj, ["getPositionOnImage", "get_position_on_image"])
            size_img = _try_call_method(rec_obj, ["getSizeOnImage", "get_size_on_image"])
            if pos_img is None or size_img is None:
                continue

            cx_px = float(pos_img[0])
            cy_px = float(pos_img[1])
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

            manifest_obj = self.objects_by_id.get(int(rec_id), None)
            if manifest_obj is None:
                if self.require_known_id:
                    if not self._warned_unknown_id:
                        print(
                            f"[GTv2] recognition object id={int(rec_id)} not found in spawned_objects.json; "
                            "skipping unknown recognized objects.",
                            flush=True,
                        )
                        self._warned_unknown_id = True
                    continue
                manifest_obj = {
                    "id": int(rec_id),
                    "class_id": int(self.model_to_class[model]),
                    "class_name": model,
                    "model": model,
                }

            ev = self._event_from_manifest_obj(
                obj=manifest_obj,
                bbox_xyxy=[cx_px - 0.5 * w_px, cy_px - 0.5 * h_px, cx_px + 0.5 * w_px, cy_px + 0.5 * h_px],
                mask_area_px=0,
                source="recognition_object_bbox_fallback",
            )
            if ev is not None:
                det_out.append(ev)

        return det_out

    def process(
        self,
        *,
        drone_x: float = 0.0,
        drone_y: float = 0.0,
        drone_yaw: float = 0.0,
    ) -> List[dict]:
        # drone pose is accepted only to keep the old detector interface.
        # Visibility comes from the Webots recognition segmentation image.
        events = self._process_from_segmentation()
        if events is not None:
            return events

        if self.fallback_to_recognition_objects:
            return self._process_from_recognition_objects()

        return []