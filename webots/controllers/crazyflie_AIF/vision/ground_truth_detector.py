from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.env_manager import Cell, GridSpec


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
    if color is None or not isinstance(color, (list, tuple)) or len(color) < 3:
        return None

    vals = [float(color[0]), float(color[1]), float(color[2])]
    if max(vals) <= 1.0:
        vals = [v * 255.0 for v in vals]
    return tuple(max(0, min(255, int(round(v)))) for v in vals)  # type: ignore[return-value]


@dataclass
class GroundTruthDetector:
    """Rendered-visibility ground-truth detector using Webots recognition segmentation.

    Recognition-segmentation pixels determine whether an object is visible in the
    rendered camera image. Target localization remains oracle-like: target events
    use the true grid cell stored in ``spawned_objects.json``. Desk events include
    segmentation-derived image boxes so the controller can run the same table
    surface-projection logic used for learned detector outputs.

    Recognition segmentation is mandatory. If it cannot be enabled or a processed
    frame has no segmentation image, the detector raises a clear runtime error
    rather than silently switching to a different visibility mechanism.
    """

    camera: Any
    objects: List[dict]
    grid_spec: GridSpec
    class_name_to_id: Dict[str, int]
    confidence: float = 1.0
    min_box_pixels: int = 1
    min_mask_pixels: int = 1
    color_tolerance: int = 3

    def __post_init__(self) -> None:
        self.class_name_to_id = {
            str(name): int(class_id)
            for name, class_id in self.class_name_to_id.items()
        }
        self.objects_by_color: Dict[Tuple[int, int, int], List[dict]] = {}

        for obj in self.objects:
            class_name = str(obj.get("class_name", obj.get("model", "")))
            model = str(obj.get("model", class_name))
            if class_name not in self.class_name_to_id and model not in self.class_name_to_id:
                continue

            rgb = _color_to_rgb255(obj.get("recognition_color_rgb", None))
            if rgb is None:
                continue
            self.objects_by_color.setdefault(rgb, []).append(obj)

        self._segmentation_enabled = False

    def enable(self, sampling_period_ms: int) -> None:
        has_recognition = _try_call_method(
            self.camera,
            ["hasRecognition", "has_recognition"],
        )
        if not has_recognition:
            raise RuntimeError(
                "Ground-truth perception requires a Camera Recognition node, but the active camera has none."
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

        enabled = False
        for name in [
            "recognitionEnableSegmentation",
            "enableRecognitionSegmentation",
            "recognition_enable_segmentation",
            "enable_recognition_segmentation",
        ]:
            if not hasattr(self.camera, name):
                continue
            method = getattr(self.camera, name)
            try:
                method()
                enabled = True
                break
            except TypeError:
                try:
                    method(sampling_period_ms)
                    enabled = True
                    break
                except Exception:
                    continue
            except Exception:
                continue

        if not enabled:
            raise RuntimeError(
                "Ground-truth perception requires Webots recognition segmentation, "
                "but segmentation could not be enabled on the active camera."
            )

        self._segmentation_enabled = True

    def _camera_width_height(self) -> tuple[int, int]:
        width = _call_method(self.camera, ["getWidth", "get_width"])
        height = _call_method(self.camera, ["getHeight", "get_height"])
        return int(width), int(height)

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
        if image is None:
            return None

        try:
            raw = np.frombuffer(image, dtype=np.uint8)
        except Exception:
            return None

        expected4 = int(width) * int(height) * 4
        expected3 = int(width) * int(height) * 3

        try:
            if raw.size >= expected4:
                bgra = raw[:expected4].reshape((int(height), int(width), 4))
                return bgra[:, :, [2, 1, 0]]
            if raw.size >= expected3:
                return raw[:expected3].reshape((int(height), int(width), 3))
        except Exception:
            return None

        return None

    @staticmethod
    def _image_rgb_at(image, width: int, x: int, y: int) -> Tuple[int, int, int]:
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
    ) -> Optional[dict]:
        cell_rc = self._cell_rc_for_manifest_obj(obj)
        if cell_rc is None:
            return None

        class_name = str(obj.get("class_name", obj.get("model", "unknown")))
        model = str(obj.get("model", class_name))
        class_id = obj.get("class_id", None)
        if class_id is None:
            class_id = self.class_name_to_id.get(class_name, self.class_name_to_id.get(model, -1))
        class_id = int(class_id)
        if class_id < 0:
            return None

        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        return {
            "cell_rc": (int(cell_rc[0]), int(cell_rc[1])),
            "confidence": float(self.confidence),
            "class_id": class_id,
            "label": class_name,
            "class_name": class_name,
            "recognition_id": None if obj.get("id", None) is None else int(obj["id"]),
            "model": model,
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1],
            "bbox_center_px": [0.5 * (x1 + x2), 0.5 * (y1 + y2)],
            "bbox_size_px": [x2 - x1, y2 - y1],
            "mask_area_px": int(mask_area_px),
            "source": "recognition_segmentation_mask",
            "detector": "ground_truth",
        }

    def process(
        self,
        *,
        drone_x: float = 0.0,
        drone_y: float = 0.0,
        drone_yaw: float = 0.0,
    ) -> List[dict]:
        del drone_x, drone_y, drone_yaw  # kept for the shared detector interface

        if not self._segmentation_enabled:
            raise RuntimeError(
                "GroundTruthDetector.process() called before recognition segmentation was enabled."
            )

        image = self._segmentation_image()
        if image is None:
            raise RuntimeError(
                "Webots recognition segmentation returned no image on a processed GT frame. "
                "Check the camera Recognition node and sampling period."
            )

        width, height = self._camera_width_height()
        colors = list(self.objects_by_color.keys())
        if not colors:
            return []

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

        rgb_arr = self._raw_image_to_rgb_array(image, width, height)
        tolerance = int(self.color_tolerance)

        if rgb_arr is not None:
            arr = rgb_arr.astype(np.int16, copy=False)
            for rgb in colors:
                target = np.array(rgb, dtype=np.int16)
                if tolerance <= 0:
                    mask = (
                        (arr[:, :, 0] == target[0])
                        & (arr[:, :, 1] == target[1])
                        & (arr[:, :, 2] == target[2])
                    )
                else:
                    diff = np.abs(arr - target.reshape((1, 1, 3)))
                    mask = (
                        (diff[:, :, 0] <= tolerance)
                        & (diff[:, :, 1] <= tolerance)
                        & (diff[:, :, 2] <= tolerance)
                    )

                bbox = self._bbox_from_mask(mask)
                if bbox is None:
                    continue
                count, min_x, min_y, max_x, max_y = bbox
                stats[rgb].update(
                    count=int(count),
                    min_x=int(min_x),
                    min_y=int(min_y),
                    max_x=int(max_x),
                    max_y=int(max_y),
                )
        else:
            # Binding compatibility fallback: still uses the segmentation image,
            # only the pixel-access mechanism changes.
            for y in range(height):
                for x in range(width):
                    pixel = self._image_rgb_at(image, width, x, y)
                    if pixel == (0, 0, 0):
                        continue

                    matched = None
                    if pixel in stats:
                        matched = pixel
                    elif tolerance > 0:
                        for rgb in colors:
                            if all(abs(int(pixel[i]) - int(rgb[i])) <= tolerance for i in range(3)):
                                matched = rgb
                                break
                    if matched is None:
                        continue

                    stat = stats[matched]
                    stat["count"] += 1
                    stat["min_x"] = min(stat["min_x"], x)
                    stat["min_y"] = min(stat["min_y"], y)
                    stat["max_x"] = max(stat["max_x"], x)
                    stat["max_y"] = max(stat["max_y"], y)

        events: List[dict] = []
        for rgb, stat in stats.items():
            count = int(stat["count"])
            if count < int(self.min_mask_pixels):
                continue

            x1 = float(stat["min_x"])
            y1 = float(stat["min_y"])
            x2 = float(stat["max_x"] + 1)
            y2 = float(stat["max_y"] + 1)
            if (x2 - x1) < float(self.min_box_pixels) or (y2 - y1) < float(self.min_box_pixels):
                continue

            for obj in self.objects_by_color.get(rgb, []):
                event = self._event_from_manifest_obj(
                    obj=obj,
                    bbox_xyxy=[x1, y1, x2, y2],
                    mask_area_px=count,
                )
                if event is not None:
                    events.append(event)

        return events
