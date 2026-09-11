"""Experiment and detector configuration helpers for the Crazyflie controller."""

from .detector_config import DetectorSelection, resolve_detector_selection
from .experiment_config import ExperimentSelection, resolve_experiment_selection
from .model_registry import MODEL_PROFILES, ModelProfile, get_model_profile
from .registry import (
    DETECTORS,
    ENVIRONMENTS,
    POLICIES,
    get_environment_spec,
    get_policy_spec,
    normalize_detector_id,
    normalize_environment_id,
    normalize_policy_id,
)
from .validation import validate_runtime_selection

__all__ = [
    "DETECTORS",
    "ENVIRONMENTS",
    "MODEL_PROFILES",
    "POLICIES",
    "DetectorSelection",
    "ExperimentSelection",
    "ModelProfile",
    "get_environment_spec",
    "get_model_profile",
    "get_policy_spec",
    "normalize_detector_id",
    "normalize_environment_id",
    "normalize_policy_id",
    "resolve_detector_selection",
    "resolve_experiment_selection",
    "validate_runtime_selection",
]
