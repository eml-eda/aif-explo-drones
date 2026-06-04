# detection_model_training/train_bottle_yolo.py

from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "dataset_bottles_only" / "target_bottle.yaml"

# ---------------------------------------------------------------------
# Model choice
# ---------------------------------------------------------------------
# Good starting options:
# - "yolo11n.pt" : fastest, smallest
# - "yolo11s.pt" : usually a strong first fine-tuning choice
# - "yolo11m.pt" : try later if you want more capacity
#
# Start with "yolo11s.pt" unless you want the fastest debugging loop.
MODEL_WEIGHTS = "yolo11s.pt"

# ---------------------------------------------------------------------
# Run naming
# ---------------------------------------------------------------------
PROJECT_NAME = "bottle_detector_runs"
RUN_NAME = "yolo11s_ft_bottles_v2"

# ---------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------
TRAIN_ARGS = dict(
    # Core
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=32,          # if OOM, try 24, 16, or 8
    device=0,          # first CUDA GPU
    workers=16,
    pretrained=True,

    # Optimization
    optimizer="auto",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    # Scheduling / stopping
    patience=20,
    cos_lr=False,

    # Precision / memory / I/O
    amp=True,
    cache=True,

    # Fine-tuning behavior
    freeze=0,

    # Augmentation
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

    # Logging / outputs
    project=str(PROJECT_ROOT / PROJECT_NAME),
    name=RUN_NAME,
    exist_ok=False,
    verbose=True,
    plots=True,
    save=True,
    save_period=-1,

    # Validation
    val=True,
)

def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    model = YOLO(MODEL_WEIGHTS)
    results = model.train(**TRAIN_ARGS)

    print("\nTraining finished.")
    print(f"Run directory: {PROJECT_ROOT / PROJECT_NAME / RUN_NAME}")
    print("Best weights should be under: weights/best.pt")
    print("Last weights should be under: weights/last.pt")

    # Optional final validation on val split using best weights
    best_weights = PROJECT_ROOT / PROJECT_NAME / RUN_NAME / "weights" / "best.pt"
    if best_weights.exists():
        best_model = YOLO(str(best_weights))
        metrics = best_model.val(data=str(DATA_YAML), split="val", device=0)
        print("\nFinal validation complete.")
        print(metrics)

    # Optional final test-set evaluation
    if best_weights.exists():
        best_model = YOLO(str(best_weights))
        test_metrics = best_model.val(data=str(DATA_YAML), split="test", device=0)
        print("\nTest-set evaluation complete.")
        print(test_metrics)

if __name__ == "__main__":
    main()  