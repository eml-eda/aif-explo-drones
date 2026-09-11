from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .model_registry import ModelProfile, get_model_profile
from .registry import get_environment_spec, normalize_detector_id, normalize_environment_id


@dataclass(frozen=True)
class DetectorSelection:
    environment: str
    detector_id: str
    backend: str
    model_family: str
    model_size: Optional[str]
    model_variant: Optional[str]
    class_names: Tuple[str, ...]
    model_path: Optional[str]
    model_path_report: Optional[str]
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    device: str

    @property
    def is_ground_truth(self) -> bool:
        return self.detector_id == "ground_truth"

    def metadata(self) -> dict:
        return {
            "detector_id": self.detector_id,
            "detector_used": self.detector_id,
            "detector_backend": self.backend,
            "use_ground_truth_detections": self.is_ground_truth,
            "detector_model_family": self.model_family,
            "detector_model_size": self.model_size,
            "detector_model_variant": self.model_variant,
            "detector_model_path": self.model_path_report,
            "detector_confidence_threshold": (
                None if self.is_ground_truth else float(self.confidence_threshold)
            ),
            "detector_iou_threshold": (
                None if self.is_ground_truth else float(self.iou_threshold)
            ),
            "detector_image_size": (
                None if self.is_ground_truth else int(self.image_size)
            ),
            "detector_device": None if self.is_ground_truth else str(self.device),
        }


def _repo_root() -> Path:
    # detector_config.py -> configuration -> crazyflie_AIF -> controllers -> webots -> repo
    return Path(__file__).resolve().parents[4]


def _resolve_model_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path.resolve()


def _portable_report_path(path: Path) -> str:
    repo = _repo_root().resolve()
    try:
        return path.resolve().relative_to(repo).as_posix()
    except ValueError:
        return str(path.resolve())


def validate_model_profile(profile: ModelProfile) -> None:
    environment_spec = get_environment_spec(profile.environment)
    expected = tuple(environment_spec.target_classes + environment_spec.context_classes)
    if tuple(profile.expected_classes) != expected:
        raise ValueError(
            "Detector profile class schema does not match the environment.\n"
            f"  environment: {profile.environment}\n"
            f"  environment classes: {list(expected)}\n"
            f"  profile classes: {list(profile.expected_classes)}\n"
            f"  detector: {profile.detector_id}"
        )


def resolve_detector_selection(
    *,
    environment: str,
    detector: str,
    model_path_override: Optional[str] = None,
    confidence_threshold: float = 0.45,
    iou_threshold: float = 0.40,
    image_size: int = 320,
    device: str = "cuda:0",
    require_model_file: bool = True,
) -> DetectorSelection:
    environment_id = normalize_environment_id(environment)
    detector_id = normalize_detector_id(detector)
    profile = get_model_profile(environment_id, detector_id)
    validate_model_profile(profile)

    if not (0.0 <= float(confidence_threshold) <= 1.0):
        raise ValueError("Detector confidence threshold must be in [0, 1].")
    if not (0.0 <= float(iou_threshold) <= 1.0):
        raise ValueError("Detector IoU threshold must be in [0, 1].")
    if int(image_size) <= 0:
        raise ValueError("Detector image size must be > 0.")

    if detector_id == "ground_truth":
        return DetectorSelection(
            environment=environment_id,
            detector_id=detector_id,
            backend=profile.backend,
            model_family=profile.model_family,
            model_size=profile.model_size,
            model_variant=profile.model_variant,
            class_names=tuple(profile.expected_classes),
            model_path=None,
            model_path_report=None,
            confidence_threshold=float(confidence_threshold),
            iou_threshold=float(iou_threshold),
            image_size=int(image_size),
            device=str(device),
        )

    selected_path = model_path_override or profile.model_path
    if not selected_path:
        raise ValueError(
            "No checkpoint path is configured for this learned detector.\n"
            f"  environment: {environment_id}\n"
            f"  detector: {detector_id}\n"
            "Add the repository-relative path in configuration/model_registry.py "
            "or set AIF_MODEL_PATH=/path/to/best.pt for this run."
        )

    resolved_path = _resolve_model_path(selected_path)
    if require_model_file and not resolved_path.is_file():
        raise FileNotFoundError(
            "Configured detector checkpoint does not exist.\n"
            f"  environment: {environment_id}\n"
            f"  detector: {detector_id}\n"
            f"  resolved path: {resolved_path}"
        )

    return DetectorSelection(
        environment=environment_id,
        detector_id=detector_id,
        backend=profile.backend,
        model_family=profile.model_family,
        model_size=profile.model_size,
        model_variant=profile.model_variant,
        class_names=tuple(profile.expected_classes),
        model_path=str(resolved_path),
        model_path_report=_portable_report_path(resolved_path),
        confidence_threshold=float(confidence_threshold),
        iou_threshold=float(iou_threshold),
        image_size=int(image_size),
        device=str(device),
    )
