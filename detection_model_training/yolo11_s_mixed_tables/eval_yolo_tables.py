#!/usr/bin/env python3
"""Evaluate YOLO11-small on dataset_mixed_tables and print metrics clearly.

Intended location:
    detection_model_training/yolo11_s_mixed_tables/eval_yolo_tables.py

This script deliberately writes a local runtime YAML before evaluation, because
dataset_mixed_tables/target_objects.yaml may contain a stale absolute `path:`
from the original Webots collector folder if the dataset was moved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
DEFAULT_DATA_YAML = write_local_dataset_yaml(
    source_yaml=SOURCE_DATA_YAML,
    dataset_dir=DATASET_DIR,
    output_yaml=SCRIPT_DIR / "target_objects_local.yaml",
)

DEFAULT_WEIGHTS = (
    SCRIPT_DIR
    / "mixed_table_detector_runs"
    / "yolo11s_ft_bottle_can_table_v1"
    / "weights"
    / "best.pt"
)
DEFAULT_EVAL_PROJECT = SCRIPT_DIR / "eval_runs_mixed_tables"


def repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    try:
        return float(obj)
    except Exception:
        return str(obj)


def extract_yolo_metrics(metrics: Any, model: YOLO) -> dict[str, Any]:
    results_dict = dict(getattr(metrics, "results_dict", {}) or {})
    box = getattr(metrics, "box", None)

    summary: dict[str, Any] = {
        "results_dict": to_jsonable(results_dict),
        "precision": None,
        "recall": None,
        "mAP50": None,
        "mAP50_95": None,
        "mAP75": None,
        "fitness": results_dict.get("fitness"),
        "per_class_mAP50_95": {},
    }

    if box is not None:
        for out_key, attr in [
            ("precision", "mp"),
            ("recall", "mr"),
            ("mAP50", "map50"),
            ("mAP50_95", "map"),
            ("mAP75", "map75"),
        ]:
            if hasattr(box, attr):
                try:
                    summary[out_key] = float(getattr(box, attr))
                except Exception:
                    pass

        maps = getattr(box, "maps", None)
        if maps is not None:
            names = getattr(model, "names", {}) or {}
            maps_list = maps.tolist() if hasattr(maps, "tolist") else list(maps)
            for idx, val in enumerate(maps_list):
                class_name = names.get(idx, str(idx)) if isinstance(names, dict) else str(idx)
                summary["per_class_mAP50_95"][class_name] = float(val)

    key_aliases = {
        "precision": ["metrics/precision(B)", "precision"],
        "recall": ["metrics/recall(B)", "recall"],
        "mAP50": ["metrics/mAP50(B)", "mAP50"],
        "mAP50_95": ["metrics/mAP50-95(B)", "mAP50-95", "mAP50_95"],
    }
    for out_key, aliases in key_aliases.items():
        if summary[out_key] is None:
            for key in aliases:
                if key in results_dict:
                    summary[out_key] = float(results_dict[key])
                    break

    return summary


def print_metric_summary(split: str, summary: dict[str, Any]) -> None:
    print(f"\n{split.upper()} main metrics")
    print(f"precision(B):  {summary['precision']:.4f}" if summary["precision"] is not None else "precision(B):  n/a")
    print(f"recall(B):     {summary['recall']:.4f}" if summary["recall"] is not None else "recall(B):     n/a")
    print(f"mAP50(B):      {summary['mAP50']:.4f}" if summary["mAP50"] is not None else "mAP50(B):      n/a")
    print(f"mAP50-95(B):   {summary['mAP50_95']:.4f}" if summary["mAP50_95"] is not None else "mAP50-95(B):   n/a")

    if summary.get("mAP75") is not None:
        print(f"mAP75(B):      {summary['mAP75']:.4f}")
    if summary.get("fitness") is not None:
        print(f"fitness:       {float(summary['fitness']):.4f}")

    per_class = summary.get("per_class_mAP50_95") or {}
    if per_class:
        print("\nPer-class mAP50-95")
        for name, val in per_class.items():
            print(f"  {name:>18s}: {float(val):.4f}")

    print("\nRaw Ultralytics results_dict")
    print(json.dumps(summary["results_dict"], indent=2))


def evaluate_split(model: YOLO, data_yaml: Path, split: str, device: str, eval_project: Path) -> dict[str, Any]:
    print(f"\n===== {split.upper()} SPLIT =====")
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        device=device,
        project=str(eval_project),
        name=f"{split}_eval",
        exist_ok=True,
        plots=True,
    )

    summary = extract_yolo_metrics(metrics, model)
    print_metric_summary(split, summary)

    split_dir = eval_project / f"{split}_eval"
    split_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = split_dir / "metrics_pretty.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved readable metrics: {metrics_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11-small mixed bottle/can/table detector with explicit metric printing."
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML))
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--eval-project", type=str, default=str(DEFAULT_EVAL_PROJECT))
    parser.add_argument("--split", type=str, default="both", choices=["val", "test", "both"])
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = repo_path(args.data)
    weights = repo_path(args.weights)
    eval_project = repo_path(args.eval_project)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {data_yaml}")
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")

    print(f"Script dir:       {SCRIPT_DIR}")
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Dataset dir:      {DATASET_DIR}")
    print(f"Source YAML:      {SOURCE_DATA_YAML}")
    print(f"Eval YAML:        {data_yaml}")
    print(f"Weights:          {weights}")
    print(f"Eval project:     {eval_project}")
    print(f"Device:           {args.device}")

    print("\nEval YAML contents:")
    print(data_yaml.read_text(encoding="utf-8"))

    model = YOLO(str(weights))
    splits = ["val", "test"] if args.split == "both" else [args.split]
    summary = {
        split: evaluate_split(model, data_yaml, split, args.device, eval_project)
        for split in splits
    }

    eval_project.mkdir(parents=True, exist_ok=True)
    summary_path = eval_project / "summary_metrics_pretty.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
