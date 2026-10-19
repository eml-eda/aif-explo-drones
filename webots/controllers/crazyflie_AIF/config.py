"""User-facing runtime configuration for Crazyflie experiments.

The Webots world still launches the controllers. Runtime choices can be made
without editing source through environment variables:

  AIF_ENVIRONMENT=easy|medium|hard
  AIF_RUN_MODE=single|evaluation
  AIF_POLICY=<policy id>
  AIF_DETECTOR=ground_truth|yolo_nano|yolo_small|mobilenet_ssd

For learned detectors, model paths are resolved from the central model registry.
A one-run override can be supplied with AIF_MODEL_PATH=/path/to/best.pt.
"""

import os

try:  # Imported as crazyflie_AIF.config by supervisors.
    from .configuration.aif_zoi_v2_params import resolve_aif_zoi_v2_params
    from .configuration.detector_config import resolve_detector_selection
    from .configuration.experiment_config import resolve_experiment_selection
    from .configuration.registry import (
        normalize_detector_id,
        normalize_environment_id,
        normalize_policy_id,
    )
    from .configuration.validation import validate_runtime_selection
except ImportError:  # Executed from the crazyflie_AIF controller directory.
    from configuration.aif_zoi_v2_params import resolve_aif_zoi_v2_params
    from configuration.detector_config import resolve_detector_selection
    from configuration.experiment_config import resolve_experiment_selection
    from configuration.registry import (
        normalize_detector_id,
        normalize_environment_id,
        normalize_policy_id,
    )
    from configuration.validation import validate_runtime_selection


# -----------------------------------------------------------------------------
# Experiment selection
# -----------------------------------------------------------------------------

ENVIRONMENT = os.environ.get("AIF_ENVIRONMENT", "medium")
RUN_MODE = os.environ.get("AIF_RUN_MODE", "evaluation")
AIF_POLICY = os.environ.get("AIF_POLICY", "rotate_and_measure")

EXPERIMENT_SELECTION = resolve_experiment_selection(
    environment=ENVIRONMENT,
    run_mode=RUN_MODE,
    policy=AIF_POLICY,
)

# Re-export normalized canonical values to existing callers.
ENVIRONMENT = EXPERIMENT_SELECTION.environment
RUN_MODE = EXPERIMENT_SELECTION.run_mode
AIF_POLICY = EXPERIMENT_SELECTION.policy

# aif_zoi_v2 model parameters (AIF_ZOI_V2_* environment variables). Invalid
# values fail at startup; other policies ignore these variables.
AIF_ZOI_V2_PARAMS = (
    resolve_aif_zoi_v2_params(EXPERIMENT_SELECTION.environment_spec)
    if AIF_POLICY == "aif_zoi_v2"
    else None
)


# -----------------------------------------------------------------------------
# Detector selection
# -----------------------------------------------------------------------------

AIF_DETECTOR = os.environ.get("AIF_DETECTOR", "ground_truth")
_detector_id = normalize_detector_id(AIF_DETECTOR)

# One generic override is the preferred mechanism. The old backend-specific
# environment variables remain accepted during the migration period.
_model_path_override = os.environ.get("AIF_MODEL_PATH", None)
if not _model_path_override and _detector_id in {"yolo_nano", "yolo_small"}:
    _model_path_override = os.environ.get("AIF_YOLO_MODEL", None)
elif not _model_path_override and _detector_id == "mobilenet_ssd":
    _model_path_override = os.environ.get("AIF_MBV2SSD_MODEL", None)

DETECTOR_CONFIDENCE_THRESHOLD = float(
    os.environ.get("AIF_DETECTOR_CONFIDENCE", "0.45")
)
DETECTOR_IOU_THRESHOLD = float(
    os.environ.get("AIF_DETECTOR_IOU", "0.40")
)
DETECTOR_IMAGE_SIZE = int(
    os.environ.get("AIF_DETECTOR_IMAGE_SIZE", "320")
)
DETECTOR_DEVICE = os.environ.get("AIF_DETECTOR_DEVICE", "cuda:0")

DETECTOR_SELECTION = resolve_detector_selection(
    environment=ENVIRONMENT,
    detector=_detector_id,
    model_path_override=_model_path_override,
    confidence_threshold=DETECTOR_CONFIDENCE_THRESHOLD,
    iou_threshold=DETECTOR_IOU_THRESHOLD,
    image_size=DETECTOR_IMAGE_SIZE,
    device=DETECTOR_DEVICE,
)

# Canonical public detector id.
AIF_DETECTOR = DETECTOR_SELECTION.detector_id

# Compatibility exports for code outside the new configuration package. New
# controller code should use DETECTOR_SELECTION directly.
DETECTOR_BACKEND = DETECTOR_SELECTION.backend
USE_GROUND_TRUTH_DETECTIONS = DETECTOR_SELECTION.is_ground_truth
AIF_MODEL_PATH = DETECTOR_SELECTION.model_path
