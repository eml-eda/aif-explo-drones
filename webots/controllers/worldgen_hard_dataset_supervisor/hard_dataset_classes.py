"""Class mapping for the mixed_hard dataset collection stack.

The hard scene contains additional non-target distractors, but the detector is
trained only on bottle, can, and desk. Distractors therefore deliberately do
not appear in this mapping or in the generated YOLO labels.
"""

from __future__ import annotations

CLASS_ID_TO_NAME = {
    0: "target_bottle",
    1: "target_can",
    2: "desk",
}

CLASS_NAME_TO_ID = {name: class_id for class_id, name in CLASS_ID_TO_NAME.items()}

MODEL_TO_CLASS_ID = {
    "target_bottle": CLASS_NAME_TO_ID["target_bottle"],
    "target_can": CLASS_NAME_TO_ID["target_can"],
    "desk": CLASS_NAME_TO_ID["desk"],
}

TARGET_CLASS_NAMES = ("target_bottle", "target_can")
DESK_CLASS_NAME = "desk"


def yolo_yaml_text(dataset_path: str) -> str:
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in CLASS_ID_TO_NAME.items())
    return (
        f"path: {dataset_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        f"{names}\n"
    )
