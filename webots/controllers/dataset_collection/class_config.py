from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class DatasetClassConfig:
    environment: str
    class_id_to_name: Dict[int, str]

    @property
    def class_name_to_id(self) -> Dict[str, int]:
        return {name: class_id for class_id, name in self.class_id_to_name.items()}

    @property
    def model_to_class_id(self) -> Dict[str, int]:
        return {
            name: class_id
            for class_id, name in self.class_id_to_name.items()
        }

    @property
    def target_class_names(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name in self.class_id_to_name.values()
            if name in ("target_bottle", "target_can")
        )

    @property
    def desk_class_name(self) -> str | None:
        return "desk" if "desk" in self.class_name_to_id else None

    def is_labelable_record(self, record: dict) -> bool:
        class_name = str(record.get("class_name", ""))
        model = str(record.get("model", class_name))
        return (
            class_name in self.class_name_to_id
            or model in self.model_to_class_id
        )

    def yolo_yaml_text(self, dataset_path: str) -> str:
        names = "\n".join(
            f"  {class_id}: {name}"
            for class_id, name in self.class_id_to_name.items()
        )
        return (
            f"path: {dataset_path}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "\n"
            "names:\n"
            f"{names}\n"
        )


_CONFIGS = {
    "easy": DatasetClassConfig(
        environment="easy",
        class_id_to_name={
            0: "target_bottle",
            1: "target_can",
        },
    ),
    "medium": DatasetClassConfig(
        environment="medium",
        class_id_to_name={
            0: "target_bottle",
            1: "target_can",
            2: "desk",
        },
    ),
    "hard": DatasetClassConfig(
        environment="hard",
        class_id_to_name={
            0: "target_bottle",
            1: "target_can",
            2: "desk",
        },
    ),
}


def normalize_collection_environment(environment: str) -> str:
    key = str(environment).strip().lower()
    if key not in _CONFIGS:
        raise ValueError(
            f"Unknown dataset collection environment '{environment}'. "
            f"Expected one of: {', '.join(sorted(_CONFIGS))}."
        )
    return key


def get_dataset_class_config(environment: str) -> DatasetClassConfig:
    return _CONFIGS[normalize_collection_environment(environment)]
