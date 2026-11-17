from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional



@dataclass(frozen=True)
class PerceptionSchedule:
    mode: str = "time"
    frames_interval: int = 20
    time_interval_s: float = 0.625

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        object.__setattr__(self, "mode", mode)
        if mode not in {"none", "frames", "time"}:
            raise ValueError(
                f"Unknown perception throttle mode '{self.mode}'. Use 'none', 'frames', or 'time'."
            )
        if mode == "frames" and int(self.frames_interval) <= 0:
            raise ValueError("frames_interval must be > 0 when mode='frames'.")
        if mode == "time" and float(self.time_interval_s) <= 0.0:
            raise ValueError("time_interval_s must be > 0 when mode='time'.")

    def should_process(
        self,
        *,
        step_count: int,
        time_now_s: float,
        last_step: int,
        last_time_s: float,
    ) -> bool:
        if self.mode == "none":
            return True
        if self.mode == "frames":
            return (int(step_count) - int(last_step)) >= int(self.frames_interval)
        return (float(time_now_s) - float(last_time_s)) >= float(self.time_interval_s)


    def metadata(self) -> dict:
        payload = {
            "mode": self.mode,
            "frames_interval": None,
            "time_interval_s": None,
        }
        if self.mode == "frames":
            payload["frames_interval"] = int(self.frames_interval)
        elif self.mode == "time":
            payload["time_interval_s"] = float(self.time_interval_s)
        return payload


@dataclass
class PerceptionResult:
    processed: bool
    source: str
    events: List[dict]
    raw_output: Any = None


class PerceptionManager:
    """Own detector selection, sampling cadence, execution, and event normalization.

    The detector/model choice is already resolved and validated by the central
    configuration layer. This class intentionally stops before geometric
    bbox-to-grid projection; floor/table projection remains in the controller.
    """

    def __init__(
        self,
        *,
        camera: Any,
        grid_spec: Any,
        detector_selection: Any,
        class_name_to_id: Dict[str, int],
        timestep_ms: int,
        schedule: PerceptionSchedule,
    ) -> None:
        self.camera = camera
        self.grid_spec = grid_spec
        self.selection = detector_selection
        self.environment = str(detector_selection.environment).strip().lower()
        self.detector_id = str(detector_selection.detector_id).strip().lower()
        self.backend = str(detector_selection.backend).strip().lower()
        self.class_name_to_id = {
            str(name): int(class_id)
            for name, class_id in class_name_to_id.items()
        }
        self.class_names = [
            name
            for name, _ in sorted(self.class_name_to_id.items(), key=lambda item: item[1])
        ]
        self.schedule = schedule
        self.detector = None
        self.source = self.detector_id

        configured_classes = list(getattr(detector_selection, "class_names", ()))
        if configured_classes and configured_classes != self.class_names:
            raise ValueError(
                "Perception class schema mismatch.\n"
                f"  detector selection: {configured_classes}\n"
                f"  controller: {self.class_names}"
            )

        # The detector can run on the first control step.
        self.last_processed_step = -10**9
        self.last_processed_time_s = -1e9

        detector_cls = self._load_learned_detector_class(self.backend)
        model_path = getattr(detector_selection, "model_path", None)
        if not model_path:
            raise ValueError(
                f"Learned detector '{self.detector_id}' has no resolved checkpoint path."
            )

        self.detector = detector_cls(
            model_name=str(model_path),
            device=str(detector_selection.device),
            conf=float(detector_selection.confidence_threshold),
            iou=float(detector_selection.iou_threshold),
            img_size=int(detector_selection.image_size),
            class_names=list(self.class_names),
            expected_model_size=getattr(detector_selection, "model_size", None),
        )
        print(
            f"[PERCEPTION] detector={self.detector_id} backend={self.backend} "
            f"model={getattr(detector_selection, 'model_path_report', model_path)}",
            flush=True,
        )

    def _load_learned_detector_class(self, backend: str):
        if backend == "yolo":
            if self.environment in {"medium", "hard"}:
                from vision.detector_yolo_context import Detector

                return Detector
            from vision.detector_yolo import Detector

            return Detector

        if backend == "mobilenet_ssd":
            from vision.detector_mobilenet_ssd import Detector

            return Detector

        raise ValueError(
            f"Unknown learned detector backend '{backend}'. "
            "Use the central detector registry to select a supported detector."
        )

    @staticmethod
    def _clamp_bbox_xyxy(bbox_xyxy, *, img_w: int, img_h: int) -> Optional[List[float]]:
        if bbox_xyxy is None or len(bbox_xyxy) < 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
        except Exception:
            return None

        x1 = max(0.0, min(float(img_w), x1))
        x2 = max(0.0, min(float(img_w), x2))
        y1 = max(0.0, min(float(img_h), y1))
        y2 = max(0.0, min(float(img_h), y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _normalize_learned_events(self, detector_output, *, width: int, height: int) -> List[dict]:
        events: List[dict] = []
        for detection in getattr(detector_output, "detections", []) or []:
            bbox = self._clamp_bbox_xyxy(
                detection.get("bbox_xyxy"),
                img_w=int(width),
                img_h=int(height),
            )
            if bbox is None:
                continue

            class_id = int(detection.get("class_id", -1))
            label = str(detection.get("label", "unknown"))
            class_name = str(detection.get("class_name", label))
            events.append(
                {
                    **detection,
                    "bbox_xyxy": bbox,
                    "confidence": float(detection.get("confidence", 0.0)),
                    "class_id": class_id,
                    "label": label,
                    "class_name": class_name,
                }
            )
        return events

    def process_if_due(
        self,
        *,
        step_count: int,
        time_now_s: float,
        camera_image,
        width: int,
        height: int,
        timestamp: Optional[float],
    ) -> PerceptionResult:
        should_process = self.schedule.should_process(
            step_count=step_count,
            time_now_s=time_now_s,
            last_step=self.last_processed_step,
            last_time_s=self.last_processed_time_s,
        )
        if not should_process:
            return PerceptionResult(
                processed=False,
                source=self.source,
                events=[],
                raw_output=None,
            )

        assert self.detector is not None
        raw_output = self.detector.process(
            camera_image,
            width=int(width),
            height=int(height),
            timestamp=timestamp,
        )
        events = self._normalize_learned_events(
            raw_output,
            width=int(width),
            height=int(height),
        )

        self.last_processed_step = int(step_count)
        self.last_processed_time_s = float(time_now_s)
        return PerceptionResult(
            processed=True,
            source=self.source,
            events=events,
            raw_output=raw_output,
        )
