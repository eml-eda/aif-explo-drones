from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from .class_config import normalize_collection_environment


def _env_first(names: Tuple[str, ...], default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def _bool_env(names: Tuple[str, ...], default: bool) -> bool:
    raw = _env_first(names, "1" if default else "0")
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class DatasetCollectionConfig:
    environment: str
    policy_env_names: Tuple[str, ...]
    capture_interval_env_names: Tuple[str, ...]
    samples_per_run_env_names: Tuple[str, ...]
    save_empty_env_names: Tuple[str, ...]
    save_segmentation_env_names: Tuple[str, ...]
    save_manifest_env_names: Tuple[str, ...]
    save_metadata_env_names: Tuple[str, ...]
    output_root_env_names: Tuple[str, ...]

    default_policy: str
    default_capture_every_n_steps: int
    default_samples_per_run: int
    default_save_empty_labels: bool
    default_save_segmentation_debug: bool
    default_save_manifest: bool
    default_save_metadata: bool
    min_box_pixels: int

    experiment_init_filename: str
    dataset_objects_filename: str
    status_filename: str
    collection_state_filename: str | None
    collection_prefix: str
    dataset_type: str

    @property
    def policy(self) -> str:
        return _env_first(self.policy_env_names + ("DATASET_POLICY",), self.default_policy)

    @property
    def capture_every_n_steps(self) -> int:
        value = int(_env_first(self.capture_interval_env_names, str(self.default_capture_every_n_steps)))
        if value <= 0:
            raise ValueError("Dataset capture interval must be positive.")
        return value

    @property
    def samples_per_run(self) -> int:
        value = int(_env_first(self.samples_per_run_env_names, str(self.default_samples_per_run)))
        if value <= 0:
            raise ValueError("Dataset samples per run must be positive.")
        return value

    @property
    def save_empty_labels(self) -> bool:
        return _bool_env(self.save_empty_env_names, self.default_save_empty_labels)

    @property
    def save_segmentation_debug(self) -> bool:
        return _bool_env(self.save_segmentation_env_names, self.default_save_segmentation_debug)

    @property
    def save_manifest(self) -> bool:
        return _bool_env(self.save_manifest_env_names, self.default_save_manifest)

    @property
    def save_metadata(self) -> bool:
        return _bool_env(self.save_metadata_env_names, self.default_save_metadata)

    def output_root_override(self) -> str | None:
        for name in self.output_root_env_names + ("DATASET_OUTPUT_ROOT",):
            value = os.environ.get(name)
            if value:
                return value
        return None


_CONFIGS = {
    "easy": DatasetCollectionConfig(
        environment="easy",
        policy_env_names=("EASY_DATASET_POLICY",),
        capture_interval_env_names=("EASY_DATASET_CAPTURE_EVERY_N_STEPS",),
        samples_per_run_env_names=("EASY_DATASET_SAMPLES_PER_RUN",),
        save_empty_env_names=("EASY_DATASET_SAVE_EMPTY_LABELS",),
        save_segmentation_env_names=("EASY_DATASET_SAVE_SEGMENTATION_DEBUG",),
        save_manifest_env_names=("EASY_DATASET_SAVE_MANIFEST",),
        save_metadata_env_names=("EASY_DATASET_SAVE_METADATA",),
        output_root_env_names=("EASY_DATASET_OUTPUT_ROOT",),
        default_policy="pseudo_random",
        default_capture_every_n_steps=400,
        default_samples_per_run=800,
        default_save_empty_labels=True,
        default_save_segmentation_debug=False,
        default_save_manifest=True,
        default_save_metadata=True,
        min_box_pixels=4,
        experiment_init_filename="easy_dataset_experiment_init.json",
        dataset_objects_filename="easy_dataset_objects.json",
        status_filename="easy_dataset_task_state.json",
        collection_state_filename=None,
        collection_prefix="collection_easy",
        dataset_type="easy_yolo_detection",
    ),
    "medium": DatasetCollectionConfig(
        environment="medium",
        policy_env_names=("MEDIUM_DATASET_POLICY",),
        capture_interval_env_names=("MEDIUM_DATASET_CAPTURE_EVERY_N_STEPS",),
        samples_per_run_env_names=("MEDIUM_DATASET_SAMPLES_PER_RUN",),
        save_empty_env_names=("MEDIUM_DATASET_SAVE_EMPTY_LABELS",),
        save_segmentation_env_names=("MEDIUM_DATASET_SAVE_SEGMENTATION_DEBUG",),
        save_manifest_env_names=("MEDIUM_DATASET_SAVE_MANIFEST",),
        save_metadata_env_names=("MEDIUM_DATASET_SAVE_METADATA",),
        output_root_env_names=("MEDIUM_DATASET_OUTPUT_ROOT",),
        default_policy="pseudo_random",
        default_capture_every_n_steps=300,
        default_samples_per_run=400,
        default_save_empty_labels=True,
        default_save_segmentation_debug=True,
        default_save_manifest=False,
        default_save_metadata=False,
        min_box_pixels=2,
        experiment_init_filename="medium_dataset_experiment_init.json",
        dataset_objects_filename="medium_dataset_objects.json",
        status_filename="medium_dataset_task_state.json",
        collection_state_filename="medium_dataset_collection_state.json",
        collection_prefix="collection_medium",
        dataset_type="medium_yolo_detection",
    ),
    "hard": DatasetCollectionConfig(
        environment="hard",
        policy_env_names=("HARD_DATASET_POLICY",),
        capture_interval_env_names=("HARD_DATASET_CAPTURE_EVERY_N_STEPS",),
        samples_per_run_env_names=("HARD_DATASET_SAMPLES_PER_RUN",),
        save_empty_env_names=("HARD_DATASET_SAVE_EMPTY_LABELS",),
        save_segmentation_env_names=("HARD_DATASET_SAVE_SEGMENTATION_DEBUG",),
        save_manifest_env_names=("HARD_DATASET_SAVE_MANIFEST",),
        save_metadata_env_names=("HARD_DATASET_SAVE_METADATA",),
        output_root_env_names=("HARD_DATASET_OUTPUT_ROOT",),
        default_policy="pseudo_random",
        default_capture_every_n_steps=300,
        default_samples_per_run=400,
        default_save_empty_labels=True,
        default_save_segmentation_debug=True,
        default_save_manifest=False,
        default_save_metadata=False,
        min_box_pixels=2,
        experiment_init_filename="hard_dataset_experiment_init.json",
        dataset_objects_filename="hard_dataset_objects.json",
        status_filename="hard_dataset_task_state.json",
        collection_state_filename="hard_dataset_collection_state.json",
        collection_prefix="collection_hard",
        dataset_type="hard_yolo_detection",
    ),
}


def get_collection_config(environment: str) -> DatasetCollectionConfig:
    return _CONFIGS[normalize_collection_environment(environment)]
