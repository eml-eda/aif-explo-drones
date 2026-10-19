from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    display_name: str


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    display_name: str
    world_file: str
    single_supervisor: str
    evaluation_supervisor: str
    target_classes: Tuple[str, ...]
    context_classes: Tuple[str, ...]
    supported_policies: Tuple[str, ...]


@dataclass(frozen=True)
class DetectorSpec:
    detector_id: str
    display_name: str
    aliases: Tuple[str, ...] = ()


POLICIES: Dict[str, PolicySpec] = {
    "wall_following": PolicySpec("wall_following", "Wall Following"),
    "spiral": PolicySpec("spiral", "Spiral"),
    "pseudo_random": PolicySpec("pseudo_random", "Pseudo Random"),
    "rotate_and_measure": PolicySpec("rotate_and_measure", "Rotate and Measure"),
    "aif_binary": PolicySpec("aif_binary", "AIF Binary"),
    "aif_zoi": PolicySpec("aif_zoi", "AIF ZOI"),
    "aif_zoi_v2": PolicySpec("aif_zoi_v2", "AIF ZOI v2"),
}

POLICY_ALIASES = {
}

_NAIVE = ("wall_following", "spiral", "pseudo_random", "rotate_and_measure")

ENVIRONMENTS: Dict[str, EnvironmentSpec] = {
    "easy": EnvironmentSpec(
        environment_id="easy",
        display_name="Easy",
        world_file="AIF_Drone_World_easy.wbt",
        single_supervisor="worldgen_easy_supervisor",
        evaluation_supervisor="worldgen_easy_eval_supervisor",
        target_classes=("target_bottle", "target_can"),
        context_classes=(),
        supported_policies=(*_NAIVE, "aif_binary", "aif_zoi", "aif_zoi_v2"),
    ),
    "medium": EnvironmentSpec(
        environment_id="medium",
        display_name="Medium",
        world_file="AIF_Drone_World_medium.wbt",
        single_supervisor="worldgen_medium_supervisor",
        evaluation_supervisor="worldgen_medium_eval_supervisor",
        target_classes=("target_bottle", "target_can"),
        context_classes=("desk",),
        supported_policies=(*_NAIVE, "aif_binary", "aif_zoi", "aif_zoi_v2"),
    ),
    "hard": EnvironmentSpec(
        environment_id="hard",
        display_name="Hard",
        world_file="AIF_Drone_World_hard.wbt",
        single_supervisor="worldgen_hard_supervisor",
        evaluation_supervisor="worldgen_hard_eval_supervisor",
        target_classes=("target_bottle", "target_can"),
        context_classes=("desk",),
        supported_policies=(*_NAIVE, "aif_binary", "aif_zoi", "aif_zoi_v2"),
    ),
}

ENVIRONMENT_IDS = frozenset(ENVIRONMENTS)

DETECTORS: Dict[str, DetectorSpec] = {
    "ground_truth": DetectorSpec(
        "ground_truth",
        "Ground Truth",
        aliases=("gt",),
    ),
    "yolo_nano": DetectorSpec(
        "yolo_nano",
        "YOLO11 Nano",
        aliases=("yolo11n", "nano"),
    ),
    "yolo_small": DetectorSpec(
        "yolo_small",
        "YOLO11 Small",
        aliases=("yolo11s", "small"),
    ),
    "mobilenet_ssd": DetectorSpec(
        "mobilenet_ssd",
        "MobileNetV2-SSD",
        aliases=("mobilenet", "mobilenetv2_ssd", "mbv2ssd"),
    ),
}

DETECTOR_ALIASES = {
    alias: detector_id
    for detector_id, spec in DETECTORS.items()
    for alias in (detector_id, *spec.aliases)
}


def normalize_environment_id(value: str) -> str:
    key = str(value).strip().lower()
    if key not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment '{value}'. Available: {', '.join(sorted(ENVIRONMENTS))}"
        )
    return key


def normalize_policy_id(value: str) -> str:
    key = str(value).strip().lower()
    key = POLICY_ALIASES.get(key, key)
    if key not in POLICIES:
        raise ValueError(
            f"Unknown policy '{value}'. Available: {', '.join(sorted(POLICIES))}"
        )
    return key


def normalize_detector_id(value: str) -> str:
    key = str(value).strip().lower()
    if key == "yolo":
        raise ValueError(
            "Detector id 'yolo' is ambiguous. Use 'yolo_nano' or 'yolo_small'."
        )
    try:
        return DETECTOR_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown detector '{value}'. Available: {', '.join(sorted(DETECTORS))}"
        ) from exc


def get_environment_spec(value: str) -> EnvironmentSpec:
    return ENVIRONMENTS[normalize_environment_id(value)]


def get_policy_spec(value: str) -> PolicySpec:
    return POLICIES[normalize_policy_id(value)]
