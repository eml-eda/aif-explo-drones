#!/usr/bin/env python3
"""Train YOLO11-small on dataset_mixed_tables.

Intended location:
    detection_model_training/yolo11_s_mixed_tables/train_yolo_tables.py

This script deliberately writes a local runtime YAML before training, because
dataset_mixed_tables/target_objects.yaml may contain a stale absolute `path:`
from the original Webots collector folder if the dataset was moved.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def find_project_root() -> Path:
    """Return detection_model_training/."""
    script_dir = Path(__file__).resolve().parent

    if (script_dir / "dataset_mixed_tables" / "target_objects.yaml").exists():
        return script_dir

    if (script_dir.parent / "dataset_mixed_tables" / "target_objects.yaml").exists():
        return script_dir.parent

    raise FileNotFoundError(
        "Could not find dataset_mixed_tables/target_objects.yaml next to this script "
        "or one directory above it. Put this script in detection_model_training/"
        "yolo11_s_mixed_tables/."
    )


def write_local_dataset_yaml(source_yaml: Path, dataset_dir: Path, output_yaml: Path) -> Path:
    """Copy source_yaml but force `path:` to the current dataset_mixed_tables folder."""
    if not source_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    required_dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "images" / "test",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
        dataset_dir / "labels" / "test",
    ]
    missing = [p for p in required_dirs if not p.exists()]
    if missing:
        missing_text = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(f"Dataset split folders are missing:\n{missing_text}")

    lines = source_yaml.read_text(encoding="utf-8").splitlines()
    new_lines: list[str] = []
    replaced_path = False

    for line in lines:
        if line.strip().startswith("path:"):
            new_lines.append(f"path: {dataset_dir}")
            replaced_path = True
        else:
            new_lines.append(line)

    if not replaced_path:
        new_lines.insert(0, f"path: {dataset_dir}")

    output_yaml.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return output_yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root()
DATASET_DIR = PROJECT_ROOT / "dataset_mixed_tables"
SOURCE_DATA_YAML = DATASET_DIR / "target_objects.yaml"
DATA_YAML = write_local_dataset_yaml(
    source_yaml=SOURCE_DATA_YAML,
    dataset_dir=DATASET_DIR,
    output_yaml=SCRIPT_DIR / "target_objects_local.yaml",
)

LOCAL_WEIGHTS = PROJECT_ROOT / "yolo11s.pt"
MODEL_WEIGHTS = str(LOCAL_WEIGHTS) if LOCAL_WEIGHTS.exists() else "yolo11s.pt"

PROJECT_NAME = SCRIPT_DIR / "mixed_table_detector_runs"
RUN_NAME = "yolo11s_ft_bottle_can_table_v1"
EVAL_PROJECT = SCRIPT_DIR / "eval_runs_mixed_tables"

TRAIN_ARGS = dict(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    workers=16,
    pretrained=True,
    optimizer="auto",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    patience=20,
    cos_lr=False,
    amp=True,
    cache=True,
    freeze=0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    project=str(PROJECT_NAME),
    name=RUN_NAME,
    exist_ok=False,
    verbose=True,
    plots=True,
    save=True,
    save_period=-1,
    val=True,
)


def main() -> None:
    print(f"Script dir:       {SCRIPT_DIR}")
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Dataset dir:      {DATASET_DIR}")
    print(f"Source YAML:      {SOURCE_DATA_YAML}")
    print(f"Training YAML:    {DATA_YAML}")
    print(f"Model:            {MODEL_WEIGHTS}")
    print(f"Run dir:          {PROJECT_NAME / RUN_NAME}")
    print(f"Eval project:     {EVAL_PROJECT}")

    print("\nTraining YAML contents:")
    print(DATA_YAML.read_text(encoding="utf-8"))

    model = YOLO(MODEL_WEIGHTS)
    model.train(**TRAIN_ARGS)

    print("\nTraining finished.")
    print(f"Run directory: {PROJECT_NAME / RUN_NAME}")
    print("Best weights should be under: weights/best.pt")
    print("Last weights should be under: weights/last.pt")

    best_weights = PROJECT_NAME / RUN_NAME / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"\nBest weights not found, skipping final eval: {best_weights}")
        return

    best_model = YOLO(str(best_weights))

    val_metrics = best_model.val(
        data=str(DATA_YAML),
        split="val",
        device=0,
        project=str(EVAL_PROJECT),
        name="val_eval",
        exist_ok=True,
        plots=True,
    )
    print("\nFinal validation complete.")
    print(val_metrics)

    test_metrics = best_model.val(
        data=str(DATA_YAML),
        split="test",
        device=0,
        project=str(EVAL_PROJECT),
        name="test_eval",
        exist_ok=True,
        plots=True,
    )
    print("\nTest-set evaluation complete.")
    print(test_metrics)


if __name__ == "__main__":
    main()
