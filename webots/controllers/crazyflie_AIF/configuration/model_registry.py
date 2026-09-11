from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelProfile:
    environment: str
    detector_id: str
    backend: str
    model_family: str
    model_size: Optional[str]
    model_variant: Optional[str]
    expected_classes: Tuple[str, ...]
    model_path: Optional[str]


# Paths are repository-relative. Learned detector profiles use the canonical
# released checkpoints under detection_model_training/checkpoints/.
MODEL_PROFILES: Dict[tuple[str, str], ModelProfile] = {
    ("easy", "ground_truth"): ModelProfile(
        environment="easy",
        detector_id="ground_truth",
        backend="ground_truth",
        model_family="ground_truth",
        model_size=None,
        model_variant=None,
        expected_classes=("target_bottle", "target_can"),
        model_path=None,
    ),
    ("easy", "yolo_nano"): ModelProfile(
        environment="easy",
        detector_id="yolo_nano",
        backend="yolo",
        model_family="yolo11",
        model_size="nano",
        model_variant="yolo11n",
        expected_classes=("target_bottle", "target_can"),
        model_path="detection_model_training/checkpoints/easy/yolo_nano.pt",
    ),
    ("easy", "yolo_small"): ModelProfile(
        environment="easy",
        detector_id="yolo_small",
        backend="yolo",
        model_family="yolo11",
        model_size="small",
        model_variant="yolo11s",
        expected_classes=("target_bottle", "target_can"),
        model_path="detection_model_training/checkpoints/easy/yolo_small.pt",
    ),
    ("easy", "mobilenet_ssd"): ModelProfile(
        environment="easy",
        detector_id="mobilenet_ssd",
        backend="mobilenet_ssd",
        model_family="mobilenetv2_ssd",
        model_size="1.0",
        model_variant="mbv2ssd_sigmoid",
        expected_classes=("target_bottle", "target_can"),
        model_path="detection_model_training/checkpoints/easy/mobilenet_ssd.pt",
    ),
    ("medium", "ground_truth"): ModelProfile(
        environment="medium",
        detector_id="ground_truth",
        backend="ground_truth",
        model_family="ground_truth",
        model_size=None,
        model_variant=None,
        expected_classes=("target_bottle", "target_can", "desk"),
        model_path=None,
    ),
    ("medium", "yolo_nano"): ModelProfile(
        environment="medium",
        detector_id="yolo_nano",
        backend="yolo",
        model_family="yolo11",
        model_size="nano",
        model_variant="yolo11n",
        expected_classes=("target_bottle", "target_can", "desk"),
        model_path="detection_model_training/checkpoints/medium/yolo_nano.pt",
    ),
    ("medium", "yolo_small"): ModelProfile(
        environment="medium",
        detector_id="yolo_small",
        backend="yolo",
        model_family="yolo11",
        model_size="small",
        model_variant="yolo11s",
        expected_classes=("target_bottle", "target_can", "desk"),
        model_path="detection_model_training/checkpoints/medium/yolo_small.pt",
    ),
    ("medium", "mobilenet_ssd"): ModelProfile(
        environment="medium",
        detector_id="mobilenet_ssd",
        backend="mobilenet_ssd",
        model_family="mobilenetv2_ssd",
        model_size="1.0",
        model_variant="mbv2ssd_sigmoid",
        expected_classes=("target_bottle", "target_can", "desk"),
        model_path="detection_model_training/checkpoints/medium/mobilenet_ssd.pt",
    ),
    ("hard", "ground_truth"): ModelProfile(
        environment="hard",
        detector_id="ground_truth",
        backend="ground_truth",
        model_family="ground_truth",
        model_size=None,
        model_variant=None,
        expected_classes=("target_bottle", "target_can", "desk"),
        model_path=None,
    ),
    ("hard", "yolo_nano"): ModelProfile(
        environment="hard",
        detector_id="yolo_nano",
        backend="yolo",
        model_family="yolo11",
        model_size="nano",
        model_variant="yolo11n",
        expected_classes=("target_bottle", "target_can", "desk"),
        model_path="detection_model_training/checkpoints/hard/yolo_nano.pt",
    ),
    ("hard", "yolo_small"): ModelProfile(
        environment="hard",
        detector_id="yolo_small",
        backend="yolo",
        model_family="yolo11",
        model_size="small",
        model_variant="yolo11s",
        expected_classes=("target_bottle", "target_can", "desk"),
        model_path="detection_model_training/checkpoints/hard/yolo_small.pt",
    ),
    ("hard", "mobilenet_ssd"): ModelProfile(
        environment="hard",
        detector_id="mobilenet_ssd",
        backend="mobilenet_ssd",
        model_family="mobilenetv2_ssd",
        model_size="1.0",
        model_variant="mbv2ssd_sigmoid",
        expected_classes=("target_bottle", "target_can", "desk"),
        model_path="detection_model_training/checkpoints/hard/mobilenet_ssd.pt",
    ),
}


def get_model_profile(environment: str, detector_id: str) -> ModelProfile:
    key = (str(environment), str(detector_id))
    try:
        return MODEL_PROFILES[key]
    except KeyError as exc:
        raise ValueError(
            f"No detector profile is registered for environment='{environment}' "
            f"and detector='{detector_id}'."
        ) from exc
