#!/usr/bin/env python3
"""Train YOLO11 nano on the mixed bottle/can dataset.

This is the nano counterpart of train_mixed_yolo.py:
- model weights: yolo11n.pt
- output folder: detection_model_training/yolov11_n_all/mixed_detector_runs/yolo11n_ft_bottle_can_v1

The training hyperparameters intentionally match the YOLO11-small run so the
comparison is mostly architecture-size driven.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def find_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "dataset_bottle_can_mixed" / "target_objects.yaml").exists():
        return script_dir
    if (script_dir.parent / "dataset_bottle_can_mixed" / "target_objects.yaml").exists():
        return script_dir.parent
    raise FileNotFoundError(
        "Could not find dataset_bottle_can_mixed/target_objects.yaml next to this script "
        "or one directory above it."
    )


PROJECT_ROOT = find_project_root()
DATA_YAML = PROJECT_ROOT / "dataset_bottle_can_mixed" / "target_objects.yaml"

MODEL_WEIGHTS = "yolo11n.pt"

OUTPUT_GROUP = "yolov11_n_all"
PROJECT_NAME = PROJECT_ROOT / OUTPUT_GROUP / "mixed_detector_runs"
RUN_NAME = "yolo11n_ft_bottle_can_v1"

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
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data YAML:    {DATA_YAML}")
    print(f"Model:        {MODEL_WEIGHTS}")
    print(f"Run dir:      {PROJECT_NAME / RUN_NAME}")

    model = YOLO(MODEL_WEIGHTS)
    model.train(**TRAIN_ARGS)

    print("\nTraining finished.")
    print(f"Run directory: {PROJECT_NAME / RUN_NAME}")
    print("Best weights should be under: weights/best.pt")
    print("Last weights should be under: weights/last.pt")

    best_weights = PROJECT_NAME / RUN_NAME / "weights" / "best.pt"
    if best_weights.exists():
        best_model = YOLO(str(best_weights))
        eval_project = PROJECT_ROOT / OUTPUT_GROUP / "eval_runs_mixed"

        val_metrics = best_model.val(
            data=str(DATA_YAML),
            split="val",
            device=0,
            project=str(eval_project),
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
            project=str(eval_project),
            name="test_eval",
            exist_ok=True,
            plots=True,
        )
        print("\nTest-set evaluation complete.")
        print(test_metrics)


if __name__ == "__main__":
    main()
