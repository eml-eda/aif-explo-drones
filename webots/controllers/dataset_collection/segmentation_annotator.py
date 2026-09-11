# vision/dataset_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

RGB = Tuple[int, int, int]


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
class SegmentationAnnotator:
    """
    Webots Recognition segmentation -> YOLO detection annotations.

    Behavior:
      - Recognition objects are used for identity/class/id only.
      - Recognition segmentation image is the only bbox source.
      - Runtime CameraRecognitionObject.getColors(), read with
        getNumberOfColors(), is the authoritative color source.
      - No recognition-box fallback is written.
    """

    camera: Any
    model_to_class: Dict[str, int]
    class_id_to_name: Optional[Dict[int, str]] = None
    object_records_by_id: Optional[Dict[int, Dict[str, Any]]] = None
    min_box_pixels: int = 4
    enable_segmentation_debug: bool = False

    segmentation_color_tolerance: int = 3
    min_target_mask_pixels: int = 4
    min_desk_mask_pixels: int = 16
    desk_class_name: str = "desk"

    log_missing_segmentation_masks: bool = True
    missing_segmentation_warning_limit: int = 40
    _missing_segmentation_warning_count: int = 0

    def enable(self, sampling_period_ms: int) -> None:
        has_recognition = _try_call_method(self.camera, ["hasRecognition", "has_recognition"])
        if not has_recognition:
            raise RuntimeError(
                "Camera has no Recognition node. Add a Recognition node in the Camera.recognition field."
            )

        _call_method(
            self.camera,
            ["recognitionEnable", "enableRecognition", "recognition_enable", "enable_recognition"],
            sampling_period_ms,
        )

        has_seg = _try_call_method(
            self.camera,
            ["hasRecognitionSegmentation", "has_recognition_segmentation"],
        )
        if not has_seg:
            raise RuntimeError(
                "Camera recognition segmentation is not available. "
                "Set Recognition.segmentation TRUE in the Camera recognition node."
            )

        _call_method(
            self.camera,
            [
                "enableRecognitionSegmentation",
                "recognitionEnableSegmentation",
                "enable_recognition_segmentation",
                "recognition_enable_segmentation",
            ],
        )

    def get_image_size(self) -> Tuple[int, int]:
        width = _call_method(self.camera, ["getWidth", "get_width"])
        height = _call_method(self.camera, ["getHeight", "get_height"])
        return int(width), int(height)

    def _class_name_for(self, class_id: int, model: str) -> str:
        if self.class_id_to_name is not None:
            return str(self.class_id_to_name.get(class_id, model))
        return str(model)

    def _is_desk(self, *, class_id: int, class_name: str, model: str) -> bool:
        return (
            class_name == self.desk_class_name
            or model == self.desk_class_name
            or (self.class_id_to_name is not None and self.class_id_to_name.get(class_id) == self.desk_class_name)
        )

    def _min_mask_pixels_for(self, *, class_id: int, class_name: str, model: str) -> int:
        if self._is_desk(class_id=class_id, class_name=class_name, model=model):
            return int(self.min_desk_mask_pixels)
        return int(self.min_target_mask_pixels)

    @staticmethod
    def _rgb_from_three(values: Iterable[float]) -> RGB:
        vals = list(values)[:3]
        if len(vals) != 3:
            raise ValueError("Need exactly three color values")
        out: List[int] = []
        for value in vals:
            v = float(value)
            if 0.0 <= v <= 1.0:
                out.append(int(round(v * 255.0)))
            else:
                out.append(int(round(v)))
        return (
            max(0, min(255, out[0])),
            max(0, min(255, out[1])),
            max(0, min(255, out[2])),
        )

    def _runtime_recognition_colors_rgb(self, obj: Any) -> List[RGB]:
        """
        Safely read CameraRecognitionObject colors.

        Webots documents colors as a double array of length
        3 * number_of_colors. Do not iterate/list() the raw object blindly;
        index exactly the documented number of entries.
        """
        n_raw = _try_call_method(obj, ["getNumberOfColors", "get_number_of_colors"])
        colors_raw = _try_call_method(obj, ["getColors", "get_colors"])
        if n_raw is None or colors_raw is None:
            return []
        try:
            n = int(n_raw)
        except Exception:
            return []

        colors: List[RGB] = []
        for i in range(max(0, n)):
            j = 3 * i
            try:
                rgb = self._rgb_from_three([
                    float(colors_raw[j]),
                    float(colors_raw[j + 1]),
                    float(colors_raw[j + 2]),
                ])
            except Exception:
                continue
            if rgb not in colors:
                colors.append(rgb)
        return colors

    def _manifest_recognition_colors_rgb(self, object_record: Optional[Dict[str, Any]]) -> List[RGB]:
        if not object_record:
            return []
        raw = object_record.get("recognition_color_rgb")
        if raw is None:
            return []
        try:
            rgb = self._rgb_from_three([float(raw[0]), float(raw[1]), float(raw[2])])
        except Exception:
            return []
        return [rgb]

    def _object_color_candidates_rgb(self, obj: Any, object_record: Optional[Dict[str, Any]]) -> List[RGB]:
        # Runtime object colors are authoritative for the segmentation image.
        colors = self._runtime_recognition_colors_rgb(obj)
        if colors:
            return colors
        # Only use the manifest color if runtime colors cannot be read at all.
        return self._manifest_recognition_colors_rgb(object_record)

    def _segmentation_image(self):
        return _try_call_method(
            self.camera,
            [
                "getRecognitionSegmentationImage",
                "recognitionGetSegmentationImage",
                "get_recognition_segmentation_image",
                "recognition_get_segmentation_image",
            ],
        )

    def _segmentation_image_array(self):
        return _try_call_method(
            self.camera,
            [
                "getRecognitionSegmentationImageArray",
                "recognitionGetSegmentationImageArray",
                "get_recognition_segmentation_image_array",
                "recognition_get_segmentation_image_array",
            ],
        )

    def _image_rgb_at(self, image, width: int, x: int, y: int) -> RGB:
        red_fn = getattr(self.camera, "imageGetRed", None) or getattr(self.camera, "image_get_red", None)
        green_fn = getattr(self.camera, "imageGetGreen", None) or getattr(self.camera, "image_get_green", None)
        blue_fn = getattr(self.camera, "imageGetBlue", None) or getattr(self.camera, "image_get_blue", None)
        if red_fn and green_fn and blue_fn:
            return (
                int(red_fn(image, width, x, y)),
                int(green_fn(image, width, x, y)),
                int(blue_fn(image, width, x, y)),
            )
        try:
            from controller import Camera  # type: ignore
        except Exception as exc:
            raise RuntimeError("Could not import Webots Camera helpers for segmentation image decoding.") from exc
        return (
            int(Camera.imageGetRed(image, width, x, y)),
            int(Camera.imageGetGreen(image, width, x, y)),
            int(Camera.imageGetBlue(image, width, x, y)),
        )

    def _colors_match(self, observed: RGB, target: RGB) -> bool:
        tol = int(self.segmentation_color_tolerance)
        return (
            abs(observed[0] - target[0]) <= tol
            and abs(observed[1] - target[1]) <= tol
            and abs(observed[2] - target[2]) <= tol
        )

    def _segmentation_pixels(self, width: int, height: int):
        """Yield (x, y, rgb) from the recognition segmentation image.

        Use the raw Webots image buffer plus imageGetRed/Green/Blue.

        Do not use getRecognitionSegmentationImageArray() here. Its nested-list
        orientation is easy to misread, and mixing arr[x][y] with arr[y][x]
        transposes part of the segmentation mask. That is what creates the
        absurd tall/skinny YOLO boxes.
        """
        img = self._segmentation_image()
        if img is None:
            raise RuntimeError(
                "Recognition segmentation image is unavailable even though segmentation was enabled."
            )

        for y in range(height):
            for x in range(width):
                yield x, y, self._image_rgb_at(img, width, x, y)

    def _segmentation_bboxes_for_objects(self, object_infos: List[Dict[str, Any]], *, width: int, height: int) -> Dict[int, Dict[str, Any]]:
        if not object_infos:
            return {}

        targets: List[Tuple[int, List[RGB]]] = []
        for info in object_infos:
            colors = info.get("candidate_colors_rgb_tuples") or []
            if colors:
                targets.append((int(info["object_key"]), colors))

        if not targets:
            return {}

        accum: Dict[int, Dict[str, Any]] = {
            object_key: {
                "x1": width,
                "y1": height,
                "x2": -1,
                "y2": -1,
                "count": 0,
                "segmentation_color_rgb": None,
                "candidate_colors_rgb": [list(c) for c in colors],
            }
            for object_key, colors in targets
        }

        for x, y, observed in self._segmentation_pixels(width, height):
            if observed == (0, 0, 0):
                continue
            for object_key, colors in targets:
                if any(self._colors_match(observed, c) for c in colors):
                    a = accum[object_key]
                    if x < a["x1"]:
                        a["x1"] = x
                    if y < a["y1"]:
                        a["y1"] = y
                    if x > a["x2"]:
                        a["x2"] = x
                    if y > a["y2"]:
                        a["y2"] = y
                    a["count"] += 1
                    if a["segmentation_color_rgb"] is None:
                        a["segmentation_color_rgb"] = list(observed)
                    break

        out: Dict[int, Dict[str, Any]] = {}
        infos_by_key = {int(info["object_key"]): info for info in object_infos}
        for object_key, a in accum.items():
            info = infos_by_key.get(object_key)
            if info is None:
                continue
            min_pixels = self._min_mask_pixels_for(
                class_id=int(info["class_id"]),
                class_name=str(info["class_name"]),
                model=str(info["model"]),
            )
            if int(a["count"]) < min_pixels:
                continue

            x1 = float(max(0, min(int(a["x1"]), width)))
            y1 = float(max(0, min(int(a["y1"]), height)))
            x2 = float(max(0, min(int(a["x2"]) + 1, width)))
            y2 = float(max(0, min(int(a["y2"]) + 1, height)))
            if (x2 - x1) < self.min_box_pixels or (y2 - y1) < self.min_box_pixels:
                continue

            out[object_key] = {
                "bbox_xyxy_px": [x1, y1, x2, y2],
                "mask_area_px": int(a["count"]),
                "segmentation_color_rgb": a["segmentation_color_rgb"],
                "candidate_colors_rgb": a["candidate_colors_rgb"],
            }
        return out

    def _warn_missing_mask(self, info: Dict[str, Any]) -> None:
        if not self.log_missing_segmentation_masks:
            return
        if self._missing_segmentation_warning_count >= int(self.missing_segmentation_warning_limit):
            return
        self._missing_segmentation_warning_count += 1
        print(
            "[DATASET DETECTOR WARNING] recognized object has no segmentation mask bbox; skipping label. "
            f"model={info.get('model')} class={info.get('class_name')} "
            f"id={info.get('recognition_id')} "
            f"object_def={None if info.get('object_record') is None else info['object_record'].get('def')} "
            f"candidate_colors={info.get('candidate_colors_rgb')}",
            flush=True,
        )

    def get_annotations(self) -> List[Dict]:
        width, height = self.get_image_size()
        objects = _call_method(self.camera, ["getRecognitionObjects", "get_recognition_objects"])

        object_infos: List[Dict[str, Any]] = []
        for idx, obj in enumerate(objects):
            model = _try_call_method(obj, ["getModel", "get_model"])
            if model not in self.model_to_class:
                continue

            obj_id = _try_call_method(obj, ["getId", "get_id"])
            pos_img = _try_call_method(obj, ["getPositionOnImage", "get_position_on_image"])
            size_img = _try_call_method(obj, ["getSizeOnImage", "get_size_on_image"])
            rel_pos = _try_call_method(obj, ["getPosition", "get_position"])
            rel_size = _try_call_method(obj, ["getSize", "get_size"])

            class_id = int(self.model_to_class[model])
            class_name = self._class_name_for(class_id, str(model))
            rec_id = None if obj_id is None else int(obj_id)
            object_record = None
            if rec_id is not None and self.object_records_by_id:
                object_record = self.object_records_by_id.get(rec_id)

            raw_bbox_xyxy_px: Optional[List[float]] = None
            raw_bbox_xywh_px: Optional[List[float]] = None
            if pos_img is not None and size_img is not None:
                cx_px = float(pos_img[0])
                cy_px = float(pos_img[1])
                w_px = float(size_img[0])
                h_px = float(size_img[1])
                raw_bbox_xyxy_px = [cx_px - 0.5 * w_px, cy_px - 0.5 * h_px, cx_px + 0.5 * w_px, cy_px + 0.5 * h_px]
                raw_bbox_xywh_px = [cx_px, cy_px, w_px, h_px]

            colors = self._object_color_candidates_rgb(obj, object_record)
            object_infos.append(
                {
                    "obj": obj,
                    "object_key": rec_id if rec_id is not None else -1000000 - idx,
                    "model": str(model),
                    "class_id": class_id,
                    "class_name": class_name,
                    "recognition_id": rec_id,
                    "object_record": object_record,
                    "rel_pos": rel_pos,
                    "rel_size": rel_size,
                    "raw_bbox_xyxy_px": raw_bbox_xyxy_px,
                    "raw_bbox_xywh_px": raw_bbox_xywh_px,
                    "candidate_colors_rgb": [list(c) for c in colors],
                    "candidate_colors_rgb_tuples": colors,
                }
            )

        segmentation_bboxes = self._segmentation_bboxes_for_objects(object_infos, width=width, height=height)

        annotations: List[Dict] = []
        for info in object_infos:
            mask_info = segmentation_bboxes.get(int(info["object_key"]))
            if mask_info is None:
                self._warn_missing_mask(info)
                continue

            x1, y1, x2, y2 = mask_info["bbox_xyxy_px"]
            clipped_w = x2 - x1
            clipped_h = y2 - y1
            if clipped_w < self.min_box_pixels or clipped_h < self.min_box_pixels:
                continue

            cx_clip = 0.5 * (x1 + x2)
            cy_clip = 0.5 * (y1 + y2)
            rec_id = info["recognition_id"]
            object_record = info["object_record"]
            edge_touch = {
                "left": bool(x1 <= 0.0),
                "top": bool(y1 <= 0.0),
                "right": bool(x2 >= float(width)),
                "bottom": bool(y2 >= float(height)),
            }

            annotations.append(
                {
                    "id": rec_id,
                    "recognition_id": rec_id,
                    "model": info["model"],
                    "class_id": int(info["class_id"]),
                    "class_name": info["class_name"],
                    "object_def": None if object_record is None else object_record.get("def"),
                    "spawn_mode": None if object_record is None else object_record.get("spawn_mode"),
                    "desk_id": None if object_record is None else object_record.get("desk_id"),
                    "is_empty_desk": bool(object_record.get("is_empty", False)) if object_record else False,
                    "bbox_source": "recognition_segmentation_mask",
                    "mask_area_px": int(mask_info["mask_area_px"]),
                    "segmentation_color_rgb": mask_info.get("segmentation_color_rgb"),
                    "candidate_colors_rgb": mask_info.get("candidate_colors_rgb"),
                    "raw_bbox_xyxy_px": None if info["raw_bbox_xyxy_px"] is None else [float(v) for v in info["raw_bbox_xyxy_px"]],
                    "raw_bbox_xywh_px": None if info["raw_bbox_xywh_px"] is None else [float(v) for v in info["raw_bbox_xywh_px"]],
                    "bbox_xyxy_px": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_xywh_px": [float(cx_clip), float(cy_clip), float(clipped_w), float(clipped_h)],
                    "bbox_xywh_norm": [
                        max(0.0, min(1.0, float(cx_clip / width))),
                        max(0.0, min(1.0, float(cy_clip / height))),
                        max(0.0, min(1.0, float(clipped_w / width))),
                        max(0.0, min(1.0, float(clipped_h / height))),
                    ],
                    "touches_image_edge": bool(any(edge_touch.values())),
                    "edge_touch": edge_touch,
                    "camera_relative_position_m": None if info["rel_pos"] is None else [float(v) for v in info["rel_pos"]],
                    "camera_relative_size_m": None if info["rel_size"] is None else [float(v) for v in info["rel_size"]],
                }
            )
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
        lines = SegmentationAnnotator.yolo_lines(annotations)
        with open(label_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

    def save_segmentation_debug(self, path: str, quality: int = 100) -> bool:
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