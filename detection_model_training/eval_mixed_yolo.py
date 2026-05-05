# detection_model_training/eval_mixed_yolo.py

from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_YAML = PROJECT_ROOT / "dataset_bottle_can_mixed" / "target_objects.yaml"

BEST_WEIGHTS = (
    PROJECT_ROOT
    / "mixed_detector_runs"
    / "yolo11s_ft_bottle_can_v1"
    / "weights"
    / "best.pt"
)

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
        project=str(PROJECT_ROOT / "eval_runs_mixed"),
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
        project=str(PROJECT_ROOT / "eval_runs_mixed"),
        name="test_eval",
        exist_ok=True,
        plots=True,
    )
    print("\nTEST results_dict:")
    print(test_metrics.results_dict)


if __name__ == "__main__":
    main()