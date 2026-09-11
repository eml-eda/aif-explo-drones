#!/usr/bin/env python3
"""Generic YOLO11 evaluator for the thesis detector datasets.

Place directly in detection_model_training/. Supports YOLO11-nano and
YOLO11-small on the easy, medium, and hard datasets.

Unlike the previous hard evaluator, this script explicitly passes imgsz=320 by
default to model.val(). This makes offline detector evaluation match the Webots
YOLO inference size instead of silently evaluating a 640-trained checkpoint at
its saved 640 training size.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torchvision.ops import box_iou
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ENV_ALIASES = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}

MODEL_ALIASES = {
    "n": "nano",
    "nano": "nano",
    "yolo11n": "nano",
    "s": "small",
    "small": "small",
    "yolo11s": "small",
}

ENV_PRESETS = {
    "easy": {
        "dataset_dir": "datasets/easy",
        "dataset_yaml": "target_objects.yaml",
        "model_suffix": "easy",
        "run_dir": "detector_runs",
        "eval_dir": "eval_runs",
        "run_tag": "bottle_can",
    },
    "medium": {
        "dataset_dir": "datasets/medium",
        "dataset_yaml": "target_objects.yaml",
        "model_suffix": "medium",
        "run_dir": "detector_runs",
        "eval_dir": "eval_runs",
        "run_tag": "bottle_can_desk",
    },
    "hard": {
        "dataset_dir": "datasets/hard",
        "dataset_yaml": "target_objects.yaml",
        "model_suffix": "hard",
        "run_dir": "detector_runs",
        "eval_dir": "eval_runs",
        "run_tag": "bottle_can_desk",
    },
}

MODEL_PRESETS = {
    "nano": {"short": "n", "checkpoint": "yolo11n.pt"},
    "small": {"short": "s", "checkpoint": "yolo11s.pt"},
}


@dataclass(frozen=True)
class DetectorPaths:
    model_root: Path
    train_project: Path
    run_name: str
    eval_project: Path
    default_weights: Path
    base_checkpoint: str


def repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def canonical_env(name: str) -> str:
    try:
        return ENV_ALIASES[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown environment {name!r}; use easy, medium, or hard") from exc


def canonical_model_size(name: str) -> str:
    try:
        return MODEL_ALIASES[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown YOLO model size {name!r}; use nano or small") from exc


def build_paths(env_name: str, model_size: str) -> DetectorPaths:
    env_name = canonical_env(env_name)
    model_size = canonical_model_size(model_size)
    ep = ENV_PRESETS[env_name]
    mp = MODEL_PRESETS[model_size]

    model_root = PROJECT_ROOT / f"yolo11_{mp['short']}_{ep['model_suffix']}"
    train_project = model_root / ep["run_dir"]
    # Explicit img320 label makes new deployment-matched runs unambiguous when
    # they are later moved alongside the historical 640-trained checkpoints.
    run_name = f"yolo11{mp['short']}_ft_{ep['run_tag']}_img320_v1"
    eval_project = model_root / ep["eval_dir"]
    default_weights = train_project / run_name / "weights" / "best.pt"
    return DetectorPaths(
        model_root=model_root,
        train_project=train_project,
        run_name=run_name,
        eval_project=eval_project,
        default_weights=default_weights,
        base_checkpoint=mp["checkpoint"],
    )


def validate_dataset(dataset_dir: Path) -> None:
    required = [
        dataset_dir / kind / split
        for kind in ("images", "labels")
        for split in ("train", "val", "test")
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset split folders are missing:\n" + "\n".join(f"  - {p}" for p in missing)
        )


def write_runtime_dataset_yaml(env_name: str, model_root: Path) -> Path:
    """Create a self-contained local YAML inside the generated detector folder."""
    env_name = canonical_env(env_name)
    preset = ENV_PRESETS[env_name]
    dataset_dir = PROJECT_ROOT / preset["dataset_dir"]
    source_yaml = dataset_dir / preset["dataset_yaml"]

    validate_dataset(dataset_dir)
    if not source_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")

    model_root.mkdir(parents=True, exist_ok=True)
    output_yaml = model_root / "target_objects_local.yaml"

    lines = source_yaml.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith("path:"):
            out.append(f"path: {dataset_dir}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.insert(0, f"path: {dataset_dir}")

    output_yaml.write_text("\n".join(out) + "\n", encoding="utf-8")
    return output_yaml


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

    aliases = {
        "precision": ["metrics/precision(B)", "precision"],
        "recall": ["metrics/recall(B)", "recall"],
        "mAP50": ["metrics/mAP50(B)", "mAP50"],
        "mAP50_95": ["metrics/mAP50-95(B)", "mAP50-95", "mAP50_95"],
    }
    for out_key, keys in aliases.items():
        if summary[out_key] is None:
            for key in keys:
                if key in results_dict:
                    summary[out_key] = float(results_dict[key])
                    break
    return summary


def print_metric_summary(split: str, summary: dict[str, Any]) -> None:
    print(f"\n{split.upper()} main metrics")
    for label, key in [
        ("precision(B)", "precision"),
        ("recall(B)", "recall"),
        ("mAP50(B)", "mAP50"),
        ("mAP50-95(B)", "mAP50_95"),
    ]:
        value = summary[key]
        print(f"{label + ':':15s} {value:.4f}" if value is not None else f"{label + ':':15s} n/a")

    if summary.get("mAP75") is not None:
        print(f"{'mAP75(B):':15s} {summary['mAP75']:.4f}")
    if summary.get("fitness") is not None:
        print(f"{'fitness:':15s} {float(summary['fitness']):.4f}")

    per_class = summary.get("per_class_mAP50_95") or {}
    if per_class:
        print("\nPer-class mAP50-95")
        for name, val in per_class.items():
            print(f"  {name:>18s}: {float(val):.4f}")

    print("\nRaw Ultralytics results_dict")
    print(json.dumps(summary["results_dict"], indent=2))


def _dataset_layout(data_yaml: Path, split: str) -> tuple[Path, Path, list[str]]:
    """Resolve the canonical images/<split> and labels/<split> directories."""
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    else:
        root = root.resolve()

    split_entry = cfg.get(split)
    if not isinstance(split_entry, str):
        raise ValueError(
            f"Expected a single directory path for split={split!r} in {data_yaml}; "
            f"got {split_entry!r}."
        )

    image_dir = (root / split_entry).resolve()
    rel = Path(split_entry)
    parts = list(rel.parts)
    try:
        images_idx = parts.index("images")
    except ValueError as exc:
        raise ValueError(
            f"Expected split path to contain an 'images' component, got {split_entry!r}."
        ) from exc
    parts[images_idx] = "labels"
    label_dir = (root / Path(*parts)).resolve()

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image split directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label split directory: {label_dir}")

    raw_names = cfg["names"]
    if isinstance(raw_names, dict):
        names = [raw_names[k] for k in sorted(raw_names, key=lambda x: int(x))]
    else:
        names = list(raw_names)
    return image_dir, label_dir, names


def _read_yolo_gt(label_path: Path, width: int, height: int) -> dict[int, torch.Tensor]:
    """Read zero-based YOLO labels and return per-class xyxy boxes in pixels."""
    by_class: dict[int, list[list[float]]] = {}
    if label_path.exists():
        for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed YOLO label at {label_path}:{line_no}: {raw!r}")
            cls_f, cx, cy, bw, bh = map(float, parts[:5])
            cls_id = int(cls_f)
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            by_class.setdefault(cls_id, []).append([x1, y1, x2, y2])

    return {
        cls_id: torch.tensor(boxes, dtype=torch.float32)
        for cls_id, boxes in by_class.items()
    }


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_deployment_metrics(
    model: YOLO,
    data_yaml: Path,
    split: str,
    device: str,
    imgsz: int,
    conf_threshold: float,
    nms_iou: float,
    match_iou: float,
) -> dict[str, Any]:
    """Compute TP/FP/FN at the detector settings used by the Webots runtime.

    `conf_threshold` and `nms_iou` control the detector output itself. `match_iou`
    is the evaluation criterion used to decide whether a prediction matches a
    ground-truth box; it is deliberately separate from the NMS IoU threshold.
    """
    image_dir, label_dir, class_names = _dataset_layout(data_yaml, split)
    image_paths = sorted(
        p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")

    counts = {
        class_id: {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
        for class_id in range(len(class_names))
    }

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=int(imgsz),
        conf=float(conf_threshold),
        iou=float(nms_iou),
        device=device,
        rect=True,
        verbose=False,
        save=False,
        stream=True,
    )

    seen = 0
    for result in results:
        # Ultralytics does not reliably preserve the original absolute source
        # path in Results.path when predict() receives a list of image paths;
        # some versions expose synthetic names such as "image0.jpg".  The
        # prediction iterator does preserve source order, so pair each result
        # with the corresponding path from our explicit, sorted input list.
        if seen >= len(image_paths):
            raise RuntimeError(
                "Deployment metric pass returned more predictions than input images."
            )

        image_path = image_paths[seen]
        rel = image_path.relative_to(image_dir)
        label_path = label_dir / rel.with_suffix(".txt")

        height, width = map(int, result.orig_shape)
        gt_by_class = _read_yolo_gt(label_path, width, height)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            pred_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            pred_cls = torch.zeros((0,), dtype=torch.int64)
            pred_conf = torch.zeros((0,), dtype=torch.float32)
        else:
            pred_xyxy = boxes.xyxy.detach().cpu().float()
            pred_cls = boxes.cls.detach().cpu().to(torch.int64)
            pred_conf = boxes.conf.detach().cpu().float()

        for class_id in range(len(class_names)):
            gt_boxes = gt_by_class.get(class_id, torch.zeros((0, 4), dtype=torch.float32))
            class_keep = pred_cls == class_id
            pboxes = pred_xyxy[class_keep]
            pscores = pred_conf[class_keep]

            counts[class_id]["gt"] += int(len(gt_boxes))
            matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)

            if len(pscores):
                order = torch.argsort(pscores, descending=True)
                pboxes = pboxes[order]

            for pbox in pboxes:
                if len(gt_boxes) == 0:
                    counts[class_id]["fp"] += 1
                    continue

                ious = box_iou(pbox.view(1, 4), gt_boxes).view(-1)
                if len(matched_gt):
                    ious[matched_gt] = -1.0
                best_iou, best_idx = torch.max(ious, dim=0)
                best_idx_i = int(best_idx.item())

                if float(best_iou) >= match_iou and not bool(matched_gt[best_idx_i]):
                    counts[class_id]["tp"] += 1
                    matched_gt[best_idx_i] = True
                else:
                    counts[class_id]["fp"] += 1

            counts[class_id]["fn"] += int((~matched_gt).sum().item())
        seen += 1

    if seen != len(image_paths):
        raise RuntimeError(f"Deployment metric pass processed {seen}/{len(image_paths)} images.")

    per_class: dict[str, Any] = {}
    for class_id, class_name in enumerate(class_names):
        c = counts[class_id]
        per_class[class_name] = {
            "gt": int(c["gt"]),
            **_prf(int(c["tp"]), int(c["fp"]), int(c["fn"])),
        }

    def aggregate(class_ids: list[int]) -> dict[str, Any]:
        gt = sum(counts[i]["gt"] for i in class_ids)
        tp = sum(counts[i]["tp"] for i in class_ids)
        fp = sum(counts[i]["fp"] for i in class_ids)
        fn = sum(counts[i]["fn"] for i in class_ids)
        return {"gt": int(gt), **_prf(int(tp), int(fp), int(fn))}

    all_ids = list(range(len(class_names)))
    target_ids = [i for i, name in enumerate(class_names) if str(name).startswith("target_")]

    return {
        "num_images": len(image_paths),
        "confidence_threshold": float(conf_threshold),
        "nms_iou_threshold": float(nms_iou),
        "match_iou_threshold": float(match_iou),
        "imgsz": int(imgsz),
        "per_class": per_class,
        "all_classes_micro": aggregate(all_ids),
        "targets_micro": aggregate(target_ids) if target_ids else None,
    }


def print_deployment_metrics(split: str, metrics: dict[str, Any]) -> None:
    print(
        f"\n{split.upper()} deployment-threshold metrics "
        f"(conf={metrics['confidence_threshold']:.3f}, "
        f"NMS IoU={metrics['nms_iou_threshold']:.3f}, "
        f"match IoU={metrics['match_iou_threshold']:.3f})"
    )
    print(
        f"{'class':>18s} {'GT':>6s} {'TP':>6s} {'FP':>6s} {'FN':>6s} "
        f"{'P':>8s} {'R':>8s} {'F1':>8s}"
    )
    for name, vals in metrics["per_class"].items():
        print(
            f"{name:>18s} {vals['gt']:6d} {vals['tp']:6d} {vals['fp']:6d} {vals['fn']:6d} "
            f"{vals['precision']:8.4f} {vals['recall']:8.4f} {vals['f1']:8.4f}"
        )

    all_vals = metrics["all_classes_micro"]
    print(
        f"{'all (micro)':>18s} {all_vals['gt']:6d} {all_vals['tp']:6d} "
        f"{all_vals['fp']:6d} {all_vals['fn']:6d} "
        f"{all_vals['precision']:8.4f} {all_vals['recall']:8.4f} {all_vals['f1']:8.4f}"
    )
    target_vals = metrics.get("targets_micro")
    if target_vals is not None:
        print(
            f"{'targets (micro)':>18s} {target_vals['gt']:6d} {target_vals['tp']:6d} "
            f"{target_vals['fp']:6d} {target_vals['fn']:6d} "
            f"{target_vals['precision']:8.4f} {target_vals['recall']:8.4f} {target_vals['f1']:8.4f}"
        )


def evaluate_split(
    model: YOLO,
    data_yaml: Path,
    split: str,
    device: str,
    eval_project: Path,
    imgsz: int,
    deployment_conf: float,
    deployment_nms_iou: float,
    deployment_match_iou: float,
) -> dict[str, Any]:
    print(f"\n===== {split.upper()} SPLIT =====")
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=int(imgsz),
        rect=True,
        device=device,
        project=str(eval_project),
        name=f"{split}_eval",
        exist_ok=True,
        plots=True,
    )

    summary = extract_yolo_metrics(metrics, model)
    print_metric_summary(split, summary)

    deployment_metrics = compute_deployment_metrics(
        model=model,
        data_yaml=data_yaml,
        split=split,
        device=device,
        imgsz=imgsz,
        conf_threshold=deployment_conf,
        nms_iou=deployment_nms_iou,
        match_iou=deployment_match_iou,
    )
    print_deployment_metrics(split, deployment_metrics)
    summary["deployment_threshold"] = deployment_metrics

    split_dir = eval_project / f"{split}_eval"
    split_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = split_dir / "metrics_pretty.json"
    deployment_path = split_dir / "deployment_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    deployment_path.write_text(json.dumps(deployment_metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved readable metrics:   {metrics_path}")
    print(f"Saved deployment metrics: {deployment_path}")
    return summary


def evaluate_checkpoint(
    *,
    env: str,
    model_size: str,
    weights: str | Path | None = None,
    data: str | Path | None = None,
    eval_project: str | Path | None = None,
    split: str = "both",
    device: str = "0",
    imgsz: int = 320,
    deployment_conf: float = 0.45,
    deployment_nms_iou: float = 0.40,
    deployment_match_iou: float = 0.50,
) -> dict[str, Any]:
    env_name = canonical_env(env)
    model_size = canonical_model_size(model_size)
    paths = build_paths(env_name, model_size)
    paths.model_root.mkdir(parents=True, exist_ok=True)

    data_yaml = repo_path(data) if data is not None else write_runtime_dataset_yaml(env_name, paths.model_root)
    weights_path = repo_path(weights) if weights is not None else paths.default_weights
    eval_path = repo_path(eval_project) if eval_project is not None else paths.eval_project

    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {data_yaml}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Environment:      {env_name}")
    print(f"YOLO variant:     {model_size}")
    print(f"Eval YAML:        {data_yaml}")
    print(f"Weights:          {weights_path}")
    print(f"Eval imgsz:       {imgsz}")
    print(f"Eval project:     {eval_path}")
    print(f"Device:           {device}")

    if imgsz != 320:
        print(
            f"\n[WARNING] eval imgsz={imgsz}; Webots inference uses 320. "
            "Use a different size only for an intentional ablation.",
            flush=True,
        )

    print("\nEval YAML contents:")
    print(data_yaml.read_text(encoding="utf-8"))

    model = YOLO(str(weights_path))
    splits = ["val", "test"] if split == "both" else [split]
    summary = {
        s: evaluate_split(
            model, data_yaml, s, device, eval_path, imgsz,
            deployment_conf, deployment_nms_iou, deployment_match_iou,
        )
        for s in splits
    }

    eval_path.mkdir(parents=True, exist_ok=True)
    summary_path = eval_path / "summary_metrics_pretty.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    eval_config = {
        "environment": env_name,
        "model_size": model_size,
        "weights": str(weights_path),
        "data": str(data_yaml),
        "imgsz": int(imgsz),
        "deployment_conf": float(deployment_conf),
        "deployment_nms_iou": float(deployment_nms_iou),
        "deployment_match_iou": float(deployment_match_iou),
        "rect": True,
        "device": str(device),
        "splits": splits,
    }
    (eval_path / "evaluation_config.json").write_text(
        json.dumps(eval_config, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nSaved summary metrics: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLO11 nano/small on easy, medium, or hard data")
    p.add_argument("--env", default="medium", choices=sorted(ENV_ALIASES))
    p.add_argument("--model-size", default="nano", choices=sorted(MODEL_ALIASES))
    p.add_argument("--data", default=None, help="Optional YAML override")
    p.add_argument("--weights", default=None, help="Optional checkpoint override")
    p.add_argument("--eval-project", default=None, help="Optional output-directory override")
    p.add_argument("--split", default="both", choices=["val", "test", "both"])
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Explicit validation input size; default matches Webots inference.")
    p.add_argument("--deployment-conf", type=float, default=0.45,
                   help="Confidence threshold for the deployment-matched TP/FP/FN report.")
    p.add_argument("--deployment-nms-iou", type=float, default=0.40,
                   help="NMS IoU used by the deployment-matched prediction pass.")
    p.add_argument("--deployment-match-iou", type=float, default=0.50,
                   help="GT/prediction IoU required to count a deployment-pass true positive.")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    evaluate_checkpoint(
        env=a.env,
        model_size=a.model_size,
        weights=a.weights,
        data=a.data,
        eval_project=a.eval_project,
        split=a.split,
        device=a.device,
        imgsz=a.imgsz,
        deployment_conf=a.deployment_conf,
        deployment_nms_iou=a.deployment_nms_iou,
        deployment_match_iou=a.deployment_match_iou,
    )


if __name__ == "__main__":
    main()
