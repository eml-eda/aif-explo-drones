# detection_model_training/eval_bottle_yolo.py

from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_YAML = PROJECT_ROOT / "dataset_bottles_only" / "target_bottle.yaml"

# Set this to the actual successful run weights
BEST_WEIGHTS = PROJECT_ROOT / "bottle_detector_runs" / "yolo11s_ft_bottles_v2" / "weights" / "best.pt"

# Change if your successful run ended up elsewhere
# Example alternative:
# BEST_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "trainX" / "weights" / "best.pt"

DEVICE = 0


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {DATA_YAML}")

    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing weights: {BEST_WEIGHTS}")

    model = YOLO(str(BEST_WEIGHTS))

    print("\n===== VALIDATION SPLIT =====")
    val_metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        device=DEVICE,
        project=str(PROJECT_ROOT / "eval_runs"),
        name="val_eval",
        exist_ok=True,
        plots=True,
    )
    print("\nVAL results_dict:")
    print(val_metrics.results_dict)

    print("\n===== TEST SPLIT =====")
    test_metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        device=DEVICE,
        project=str(PROJECT_ROOT / "eval_runs"),
        name="test_eval",
        exist_ok=True,
        plots=True,
    )
    print("\nTEST results_dict:")
    print(test_metrics.results_dict)


if __name__ == "__main__":
    main()