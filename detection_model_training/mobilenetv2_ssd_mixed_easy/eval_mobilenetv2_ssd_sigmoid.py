#!/usr/bin/env python3
"""Evaluate the custom MobileNetV2-SSD sigmoid detector on YOLO-format splits.

Expected current repo layout:

    detection_model_training/
      eval_mobilenetv2_ssd_sigmoid.py
      dataset_bottle_can_mixed/
        target_objects.yaml
        images/{train,val,test}/...
        labels/{train,val,test}/...
      mobilenetv2_ssd/
        __init__.py
        mobilenetv2_ssd_sigmoid.py
      mobilenetv2_ssd_all/
        runs/...

Outputs are intentionally kept separate from training artifacts:

    eval_runs_mobilenet/
      val_eval/
        metrics.json
        predictions.json
        sample_predictions.txt
      test_eval/
        metrics.json
        predictions.json
        sample_predictions.txt
      summary_metrics.json

YOLO labels are 0-based. For this torchvision-style SSD model, labels are
shifted by +1 because label 0 is reserved internally for background.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou


# -----------------------------------------------------------------------------
# Import the local mobilenetv2_ssd package.
#
# Important: add PROJECT_ROOT, not PROJECT_ROOT/mobilenetv2_ssd, to sys.path.
# If PROJECT_ROOT/mobilenetv2_ssd is inserted first, Python may import
# mobilenetv2_ssd/mobilenetv2_ssd.py as a plain module, which causes:
# "'mobilenetv2_ssd' is not a package".
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from mobilenetv2_ssd.mobilenetv2_ssd_sigmoid import mobilenet_v2_ssd_sigmoid
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import mobilenet_v2_ssd_sigmoid. Expected file: "
        f"{PROJECT_ROOT / 'mobilenetv2_ssd' / 'mobilenetv2_ssd_sigmoid.py'}. "
        "Run this script from detection_model_training/ or keep the script next "
        "to the mobilenetv2_ssd package folder."
    ) from exc


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DataConfig:
    root: Path
    names: list[str]


def repo_path(path: str | Path) -> Path:
    """Resolve relative paths against detection_model_training/."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def read_data_yaml(path: str | Path) -> DataConfig:
    path = repo_path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", path.parent)).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    else:
        root = root.resolve()

    raw_names = cfg["names"]
    if isinstance(raw_names, dict):
        names = [raw_names[k] for k in sorted(raw_names, key=lambda x: int(x))]
    else:
        names = list(raw_names)

    if not names:
        raise ValueError(f"No class names found in {path}")

    return DataConfig(root=root, names=names)


class YoloDetectionDataset(Dataset):
    def __init__(self, data: DataConfig, split: str, min_box_size_px: float = 1.0) -> None:
        self.data = data
        self.split = split
        self.min_box_size_px = min_box_size_px
        self.image_dir = data.root / "images" / split
        self.label_dir = data.root / "labels" / split

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {self.label_dir}")

        self.images = sorted(
            p for p in self.image_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.images:
            raise FileNotFoundError(f"No images found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def _read_yolo_label(self, label_path: Path, width: int, height: int) -> tuple[Tensor, Tensor]:
        boxes: list[list[float]] = []
        labels: list[int] = []

        if not label_path.exists():
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        with label_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_f, cx, cy, bw, bh = map(float, parts[:5])
                cls_id = int(cls_f)
                if cls_id < 0 or cls_id >= len(self.data.names):
                    raise ValueError(
                        f"Invalid class id {cls_id} in {label_path}:{line_no}; "
                        f"valid YOLO ids are 0..{len(self.data.names) - 1}"
                    )

                x1 = (cx - bw / 2.0) * width
                y1 = (cy - bh / 2.0) * height
                x2 = (cx + bw / 2.0) * width
                y2 = (cy + bh / 2.0) * height

                x1 = max(0.0, min(float(width), x1))
                y1 = max(0.0, min(float(height), y1))
                x2 = max(0.0, min(float(width), x2))
                y2 = max(0.0, min(float(height), y2))

                if (x2 - x1) >= self.min_box_size_px and (y2 - y1) >= self.min_box_size_px:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id + 1)  # SSD foreground labels start at 1.

        if not boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor | str]]:
        image_path = self.images[idx]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, labels = self._read_yolo_label(label_path, width, height)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

        target: dict[str, Tensor | str] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "image_path": str(image_path),
        }
        return image_tensor, target


def collate_fn(batch: list[tuple[Tensor, dict[str, Any]]]) -> tuple[list[Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def parse_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[dict[str, Tensor], dict[str, Any]]:
    ckpt_path = repo_path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing weights: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], dict(ckpt.get("args", {}))
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # Allows loading a raw state_dict as a fallback.
        return ckpt, {}
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def compute_class_ap(
    predictions: list[tuple[int, float, Tensor]],
    gt_for_class: dict[int, Tensor],
    iou_threshold: float,
) -> float:
    npos = sum(int(boxes.shape[0]) for boxes in gt_for_class.values())
    if npos == 0:
        return math.nan
    if not predictions:
        return 0.0

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    matched = {
        img_id: torch.zeros(len(boxes), dtype=torch.bool)
        for img_id, boxes in gt_for_class.items()
    }
    tp = np.zeros(len(predictions), dtype=np.float64)
    fp = np.zeros(len(predictions), dtype=np.float64)

    for i, (img_id, _score, pred_box) in enumerate(predictions):
        gt_boxes = gt_for_class.get(img_id, torch.zeros((0, 4), dtype=torch.float32))
        if gt_boxes.numel() == 0:
            fp[i] = 1.0
            continue

        ious = box_iou(pred_box.view(1, 4), gt_boxes).view(-1)
        best_iou, best_idx = torch.max(ious, dim=0)
        best_idx_i = int(best_idx.item())
        if float(best_iou) >= iou_threshold and not bool(matched[img_id][best_idx_i]):
            tp[i] = 1.0
            matched[img_id][best_idx_i] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(1, npos)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    recall_grid = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in recall_grid:
        precision_at_r = precisions[recalls >= r]
        ap += float(np.max(precision_at_r)) if precision_at_r.size else 0.0
    return ap / 101.0


def summarize_metrics(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor | str]],
    class_names: list[str],
) -> dict[str, Any]:
    num_classes = len(class_names)
    thresholds = [round(float(x), 2) for x in np.arange(0.50, 0.951, 0.05)]

    gt_by_class: list[dict[int, Tensor]] = [dict() for _ in range(num_classes + 1)]
    preds_by_class: list[list[tuple[int, float, Tensor]]] = [[] for _ in range(num_classes + 1)]

    for image_idx, (target, pred) in enumerate(zip(targets, predictions)):
        target_boxes = target["boxes"]
        target_labels = target["labels"]
        assert isinstance(target_boxes, Tensor)
        assert isinstance(target_labels, Tensor)

        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]

        for cls in range(1, num_classes + 1):
            gt_by_class[cls][image_idx] = target_boxes[target_labels == cls].cpu()

            keep = pred_labels.cpu() == cls
            boxes_c = pred_boxes.cpu()[keep]
            scores_c = pred_scores.cpu()[keep]
            for box, score in zip(boxes_c, scores_c):
                preds_by_class[cls].append((image_idx, float(score), box))

    per_class: dict[str, dict[str, float]] = {}
    ap_matrix: list[list[float]] = []

    for cls in range(1, num_classes + 1):
        cls_aps = [compute_class_ap(preds_by_class[cls], gt_by_class[cls], thr) for thr in thresholds]
        ap_matrix.append(cls_aps)

        finite_aps = [x for x in cls_aps if not math.isnan(x)]
        per_class[class_names[cls - 1]] = {
            "AP50": cls_aps[0],
            "AP75": cls_aps[5],
            "AP50_95": float(np.mean(finite_aps)) if finite_aps else math.nan,
            "num_gt": float(sum(int(v.shape[0]) for v in gt_by_class[cls].values())),
            "num_predictions": float(len(preds_by_class[cls])),
        }

    ap_array = np.array(ap_matrix, dtype=np.float64)
    return {
        "num_images": len(targets),
        "num_gt": int(sum(v["num_gt"] for v in per_class.values())),
        "num_predictions": int(sum(v["num_predictions"] for v in per_class.values())),
        "mAP50_95": float(np.nanmean(ap_array)),
        "AP50": float(np.nanmean(ap_array[:, 0])),
        "AP75": float(np.nanmean(ap_array[:, 5])),
        "per_class": per_class,
    }


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor | str]]]:
    model.eval()
    all_predictions: list[dict[str, Tensor]] = []
    all_targets: list[dict[str, Tensor | str]] = []

    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            all_predictions.append(
                {
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                }
            )
            all_targets.append(
                {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                    "image_path": str(target["image_path"]),
                }
            )

        if step == 1 or step % 25 == 0 or step == len(loader):
            print(f"inference step={step:04d}/{len(loader):04d}", flush=True)

    return all_predictions, all_targets


def class_name_from_label(label: int, class_names: list[str]) -> str:
    if 1 <= label <= len(class_names):
        return class_names[label - 1]
    return f"unknown_label_{label}"


def best_same_class_iou(pred_box: Tensor, pred_label: int, target: dict[str, Tensor | str]) -> float | None:
    target_boxes = target["boxes"]
    target_labels = target["labels"]
    assert isinstance(target_boxes, Tensor)
    assert isinstance(target_labels, Tensor)

    same_class = target_labels == pred_label
    if not bool(same_class.any()):
        return None
    ious = box_iou(pred_box.view(1, 4), target_boxes[same_class]).view(-1)
    return float(ious.max().item()) if ious.numel() else None


def format_box(box: Tensor) -> str:
    vals = [float(x) for x in box.tolist()]
    return "[" + ", ".join(f"{v:.1f}" for v in vals) + "]"


def build_sample_prediction_text(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor | str]],
    class_names: list[str],
    split: str,
    num_samples: int,
    min_score: float,
    max_dets: int,
) -> str:
    lines: list[str] = []
    lines.append(f"===== SAMPLE PREDICTIONS: {split.upper()} =====")
    lines.append(
        f"Showing up to {num_samples} images, up to {max_dets} detections/image, "
        f"score >= {min_score:.3f}"
    )
    lines.append("")

    n = min(num_samples, len(predictions))
    for sample_idx in range(n):
        pred = predictions[sample_idx]
        target = targets[sample_idx]
        image_path = Path(str(target["image_path"]))

        lines.append(f"[{sample_idx + 1:02d}] {image_path.name}")

        target_boxes = target["boxes"]
        target_labels = target["labels"]
        assert isinstance(target_boxes, Tensor)
        assert isinstance(target_labels, Tensor)

        if len(target_boxes) == 0:
            lines.append("    GT: none")
        else:
            gt_parts = []
            for box, label in zip(target_boxes, target_labels):
                label_i = int(label.item())
                gt_parts.append(f"{class_name_from_label(label_i, class_names)} xyxy={format_box(box)}")
            lines.append("    GT: " + "; ".join(gt_parts))

        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]
        keep = scores >= min_score
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        if scores.numel() == 0:
            lines.append("    Pred: none above threshold")
            lines.append("")
            continue

        order = torch.argsort(scores, descending=True)[:max_dets]
        lines.append("    Pred:")
        for rank, det_idx in enumerate(order, start=1):
            box = boxes[det_idx]
            label_i = int(labels[det_idx].item())
            score = float(scores[det_idx].item())
            best_iou = best_same_class_iou(box, label_i, target)
            iou_text = "no_same_class_gt" if best_iou is None else f"best_same_class_iou={best_iou:.3f}"
            lines.append(
                f"      {rank:02d}. {class_name_from_label(label_i, class_names):>14s} "
                f"score={score:.3f}, xyxy={format_box(box)}, {iou_text}"
            )
        lines.append("")

    return "\n".join(lines)


def write_predictions_json(
    path: Path,
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor | str]],
    class_names: list[str],
) -> None:
    records: list[dict[str, Any]] = []

    for pred, target in zip(predictions, targets):
        dets: list[dict[str, Any]] = []
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            label_i = int(label.item())
            dets.append(
                {
                    "class_id_yolo": label_i - 1,
                    "label_ssd": label_i,
                    "class_name": class_name_from_label(label_i, class_names),
                    "score": float(score.item()),
                    "bbox_xyxy": [float(x) for x in box.tolist()],
                }
            )

        gts: list[dict[str, Any]] = []
        target_boxes = target["boxes"]
        target_labels = target["labels"]
        assert isinstance(target_boxes, Tensor)
        assert isinstance(target_labels, Tensor)
        for box, label in zip(target_boxes, target_labels):
            label_i = int(label.item())
            gts.append(
                {
                    "class_id_yolo": label_i - 1,
                    "label_ssd": label_i,
                    "class_name": class_name_from_label(label_i, class_names),
                    "bbox_xyxy": [float(x) for x in box.tolist()],
                }
            )

        records.append({"image_path": str(target["image_path"]), "targets": gts, "detections": dets})

    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)



def _match_counts_for_class(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor | str]],
    class_label: int,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Return TP, FP, FN for one 1-based SSD class label."""
    tp = 0
    fp = 0
    npos = 0

    for pred, target in zip(predictions, targets):
        target_boxes = target["boxes"]
        target_labels = target["labels"]
        assert isinstance(target_boxes, Tensor)
        assert isinstance(target_labels, Tensor)

        gt_boxes = target_boxes[target_labels == class_label].cpu()
        npos += int(gt_boxes.shape[0])
        matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        keep = (pred["labels"].cpu() == class_label) & (pred["scores"].cpu() >= conf_threshold)
        pred_boxes = pred["boxes"].cpu()[keep]
        pred_scores = pred["scores"].cpu()[keep]

        if pred_scores.numel() > 0:
            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]

        for box in pred_boxes:
            if gt_boxes.numel() == 0:
                fp += 1
                continue
            ious = box_iou(box.view(1, 4), gt_boxes).view(-1)
            best_iou, best_idx = torch.max(ious, dim=0)
            best_idx_i = int(best_idx.item())
            if float(best_iou) >= iou_threshold and not bool(matched[best_idx_i]):
                tp += 1
                matched[best_idx_i] = True
            else:
                fp += 1

    fn = npos - tp
    return tp, fp, fn


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def compute_confidence_curves(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor | str]],
    class_names: list[str],
    iou_threshold: float = 0.50,
    num_thresholds: int = 101,
) -> dict[str, Any]:
    """Compute YOLO-like precision/recall/F1 curves over confidence thresholds."""
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    per_class: dict[str, dict[str, list[float]]] = {}

    for cls_idx, name in enumerate(class_names, start=1):
        precision: list[float] = []
        recall: list[float] = []
        f1: list[float] = []
        for conf in thresholds:
            tp, fp, fn = _match_counts_for_class(
                predictions, targets, cls_idx, float(conf), iou_threshold
            )
            p = _safe_div(tp, tp + fp)
            r = _safe_div(tp, tp + fn)
            f = _safe_div(2.0 * p * r, p + r)
            precision.append(p)
            recall.append(r)
            f1.append(f)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}

    p_arr = np.array([v["precision"] for v in per_class.values()], dtype=np.float64)
    r_arr = np.array([v["recall"] for v in per_class.values()], dtype=np.float64)
    f_arr = np.array([v["f1"] for v in per_class.values()], dtype=np.float64)
    mean_precision = np.mean(p_arr, axis=0) if len(class_names) else np.zeros_like(thresholds)
    mean_recall = np.mean(r_arr, axis=0) if len(class_names) else np.zeros_like(thresholds)
    mean_f1 = np.mean(f_arr, axis=0) if len(class_names) else np.zeros_like(thresholds)
    best_idx = int(np.argmax(mean_f1)) if mean_f1.size else 0

    return {
        "thresholds": thresholds.tolist(),
        "per_class": per_class,
        "mean_precision": mean_precision.tolist(),
        "mean_recall": mean_recall.tolist(),
        "mean_f1": mean_f1.tolist(),
        "best_f1": float(mean_f1[best_idx]) if mean_f1.size else 0.0,
        "best_confidence": float(thresholds[best_idx]) if thresholds.size else 0.0,
        "iou_threshold": iou_threshold,
    }


def save_curve_plots(curves: dict[str, Any], class_names: list[str], out_dir: Path) -> None:
    thresholds = np.array(curves["thresholds"], dtype=np.float64)
    mean_precision = np.array(curves["mean_precision"], dtype=np.float64)
    mean_recall = np.array(curves["mean_recall"], dtype=np.float64)
    mean_f1 = np.array(curves["mean_f1"], dtype=np.float64)
    best_f1 = curves["best_f1"]
    best_conf = curves["best_confidence"]

    def save_metric_curve(filename: str, title: str, ylabel: str, key: str, mean_values: np.ndarray) -> None:
        plt.figure(figsize=(12, 8))
        for name in class_names:
            vals = np.array(curves["per_class"][name][key], dtype=np.float64)
            plt.plot(thresholds, vals, linewidth=1.5, label=name)
        if key == "f1":
            label = f"all classes {best_f1:.2f} at {best_conf:.3f}"
        else:
            label = "all classes"
        plt.plot(thresholds, mean_values, linewidth=4.0, label=label)
        plt.title(title)
        plt.xlabel("Confidence")
        plt.ylabel(ylabel)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()

    save_metric_curve("BoxF1_curve.png", "F1-Confidence Curve", "F1", "f1", mean_f1)
    save_metric_curve("BoxP_curve.png", "Precision-Confidence Curve", "Precision", "precision", mean_precision)
    save_metric_curve("BoxR_curve.png", "Recall-Confidence Curve", "Recall", "recall", mean_recall)

    plt.figure(figsize=(12, 8))
    for name in class_names:
        p = np.array(curves["per_class"][name]["precision"], dtype=np.float64)
        r = np.array(curves["per_class"][name]["recall"], dtype=np.float64)
        order = np.argsort(r)
        plt.plot(r[order], p[order], linewidth=1.5, label=name)
    order = np.argsort(mean_recall)
    plt.plot(mean_recall[order], mean_precision[order], linewidth=4.0, label="all classes")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "BoxPR_curve.png", dpi=200)
    plt.close()


def compute_confusion_matrix(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor | str]],
    class_names: list[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> np.ndarray:
    """
    Confusion matrix with rows=true class, columns=predicted class.
    Last row/column is background: FP predictions go to background row;
    missed GT boxes go to background column.
    """
    n = len(class_names)
    bg = n
    matrix = np.zeros((n + 1, n + 1), dtype=np.float64)

    for pred, target in zip(predictions, targets):
        target_boxes = target["boxes"].cpu()
        target_labels = target["labels"].cpu() - 1  # to 0-based
        assert isinstance(target_boxes, Tensor)
        assert isinstance(target_labels, Tensor)

        keep = pred["scores"].cpu() >= conf_threshold
        pred_boxes = pred["boxes"].cpu()[keep]
        pred_labels = pred["labels"].cpu()[keep] - 1
        pred_scores = pred["scores"].cpu()[keep]

        order = torch.argsort(pred_scores, descending=True) if pred_scores.numel() else torch.empty(0, dtype=torch.long)
        matched_gt: set[int] = set()

        for pred_idx in order.tolist():
            pbox = pred_boxes[pred_idx]
            plabel = int(pred_labels[pred_idx].item())
            if target_boxes.numel() == 0:
                matrix[bg, plabel] += 1
                continue

            ious = box_iou(pbox.view(1, 4), target_boxes).view(-1)
            # Do not allow one GT to absorb multiple predictions.
            for gt_i in matched_gt:
                ious[gt_i] = -1.0
            best_iou, best_gt = torch.max(ious, dim=0)
            best_gt_i = int(best_gt.item())
            if float(best_iou) >= iou_threshold:
                tlabel = int(target_labels[best_gt_i].item())
                matrix[tlabel, plabel] += 1
                matched_gt.add(best_gt_i)
            else:
                matrix[bg, plabel] += 1

        for gt_idx, label in enumerate(target_labels.tolist()):
            if gt_idx not in matched_gt:
                matrix[int(label), bg] += 1

    return matrix


def save_confusion_matrix_plot(matrix: np.ndarray, class_names: list[str], out_path: Path, normalized: bool) -> None:
    labels = class_names + ["background"]
    values = matrix.copy()
    if normalized:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)

    plt.figure(figsize=(9, 8))
    im = plt.imshow(values, interpolation="nearest", aspect="auto")
    plt.title("Confusion Matrix" + (" Normalized" if normalized else ""))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    max_val = values.max() if values.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if normalized:
                text = f"{val:.2f}" if val > 0 else ""
            else:
                text = f"{int(val)}" if val > 0 else ""
            if text:
                plt.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if max_val > 0 and val > max_val / 2 else "black",
                    fontsize=9,
                )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _draw_box(draw: ImageDraw.ImageDraw, box: Tensor, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    text_bbox = draw.textbbox((x1, max(0, y1 - 16)), label)
    draw.rectangle(text_bbox, fill=color)
    draw.text((x1, max(0, y1 - 16)), label, fill=(255, 255, 255))


def _make_grid(images: list[Image.Image], cols: int = 4) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), "white")
    widths, heights = zip(*(img.size for img in images))
    cell_w = max(widths)
    cell_h = max(heights)
    rows = int(math.ceil(len(images) / cols))
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(245, 245, 245))
    for idx, img in enumerate(images):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        grid.paste(img, (x, y))
    return grid


def save_sample_batch_images(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor | str]],
    class_names: list[str],
    split: str,
    out_dir: Path,
    num_batches: int,
    batch_size: int,
    min_score: float,
) -> None:
    colors = [(56, 142, 60), (25, 118, 210), (245, 124, 0), (123, 31, 162)]
    total = min(len(targets), num_batches * batch_size)
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(total, start + batch_size)
        if start >= end:
            break

        label_images: list[Image.Image] = []
        pred_images: list[Image.Image] = []
        for idx in range(start, end):
            image_path = Path(str(targets[idx]["image_path"]))
            base = Image.open(image_path).convert("RGB")
            label_img = base.copy()
            pred_img = base.copy()
            label_draw = ImageDraw.Draw(label_img)
            pred_draw = ImageDraw.Draw(pred_img)

            target_boxes = targets[idx]["boxes"]
            target_labels = targets[idx]["labels"]
            assert isinstance(target_boxes, Tensor)
            assert isinstance(target_labels, Tensor)
            for box, label in zip(target_boxes, target_labels):
                label_i = int(label.item())
                color = colors[(label_i - 1) % len(colors)]
                _draw_box(label_draw, box, class_name_from_label(label_i, class_names), color)

            keep = predictions[idx]["scores"] >= min_score
            boxes = predictions[idx]["boxes"][keep]
            labels = predictions[idx]["labels"][keep]
            scores = predictions[idx]["scores"][keep]
            order = torch.argsort(scores, descending=True)[:20] if scores.numel() else torch.empty(0, dtype=torch.long)
            for det_idx in order:
                label_i = int(labels[det_idx].item())
                score = float(scores[det_idx].item())
                color = colors[(label_i - 1) % len(colors)]
                _draw_box(
                    pred_draw,
                    boxes[det_idx],
                    f"{class_name_from_label(label_i, class_names)} {score:.2f}",
                    color,
                )

            label_images.append(label_img)
            pred_images.append(pred_img)

        _make_grid(label_images).save(out_dir / f"{split}_batch{batch_idx}_labels.jpg", quality=95)
        _make_grid(pred_images).save(out_dir / f"{split}_batch{batch_idx}_pred.jpg", quality=95)


def save_eval_plots(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor | str]],
    class_names: list[str],
    split: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    curves = compute_confidence_curves(
        predictions,
        targets,
        class_names,
        iou_threshold=args.curve_iou,
        num_thresholds=args.curve_points,
    )
    save_curve_plots(curves, class_names, out_dir)
    with (out_dir / "confidence_curves.json").open("w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)

    matrix = compute_confusion_matrix(
        predictions,
        targets,
        class_names,
        conf_threshold=args.confusion_conf,
        iou_threshold=args.confusion_iou,
    )
    np.savetxt(out_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%.6g")
    save_confusion_matrix_plot(matrix, class_names, out_dir / "confusion_matrix.png", normalized=False)
    save_confusion_matrix_plot(matrix, class_names, out_dir / "confusion_matrix_normalized.png", normalized=True)

    save_sample_batch_images(
        predictions,
        targets,
        class_names,
        split,
        out_dir,
        num_batches=args.plot_batches,
        batch_size=args.plot_batch_size,
        min_score=args.sample_min_score,
    )

def evaluate_split(
    split: str,
    model: torch.nn.Module,
    data: DataConfig,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"\n===== {split.upper()} SPLIT =====")

    dataset = YoloDetectionDataset(data, split=split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )

    predictions, targets = run_inference(model, loader, device)
    metrics = summarize_metrics(predictions, targets, data.names)

    print(f"\n{split.upper()} metrics")
    print(f"num_images:      {metrics['num_images']}")
    print(f"num_gt:          {metrics['num_gt']}")
    print(f"num_predictions: {metrics['num_predictions']}")
    print(f"mAP50-95:        {metrics['mAP50_95']:.4f}")
    print(f"AP50:            {metrics['AP50']:.4f}")
    print(f"AP75:            {metrics['AP75']:.4f}")
    print("\nPer class")
    for name, vals in metrics["per_class"].items():
        print(
            f"  {name:>14s} | AP50-95={vals['AP50_95']:.4f} "
            f"AP50={vals['AP50']:.4f} AP75={vals['AP75']:.4f} "
            f"GT={int(vals['num_gt'])} Pred={int(vals['num_predictions'])}"
        )

    split_dir = repo_path(args.eval_project) / f"{split}_eval"
    split_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = split_dir / "metrics.json"
    predictions_path = split_dir / "predictions.json"
    samples_path = split_dir / "sample_predictions.txt"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_predictions_json(predictions_path, predictions, targets, data.names)

    sample_text = build_sample_prediction_text(
        predictions=predictions,
        targets=targets,
        class_names=data.names,
        split=split,
        num_samples=args.print_samples,
        min_score=args.sample_min_score,
        max_dets=args.sample_max_dets,
    )
    samples_path.write_text(sample_text + "\n", encoding="utf-8")

    if args.print_samples > 0:
        print("\n" + sample_text)

    if args.plots:
        print(f"\nSaving YOLO-like diagnostic plots to: {split_dir}")
        save_eval_plots(predictions, targets, data.names, split, split_dir, args)

    print(f"\nSaved {split} metrics:     {metrics_path}")
    print(f"Saved {split} predictions: {predictions_path}")
    print(f"Saved {split} samples:     {samples_path}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2-SSD sigmoid on YOLO-format data")
    parser.add_argument("--data", type=str, default="dataset_bottle_can_mixed/target_objects.yaml")
    parser.add_argument(
        "--weights",
        type=str,
        default="mobilenetv2_ssd_all/runs/mbv2ssd_sigmoid_bottle_can_v1/best.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "val", "test", "both"],
        help="Use 'both' to evaluate val and test, matching your YOLO eval style.",
    )
    parser.add_argument("--eval-project", type=str, default="eval_runs_mobilenet")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, or 0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width-mult", type=float, default=None, choices=[0.5, 0.75, 1.0])
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.001)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--detections-per-img", type=int, default=100)
    parser.add_argument("--print-samples", type=int, default=8)
    parser.add_argument("--sample-min-score", type=float, default=0.25)
    parser.add_argument("--sample-max-dets", type=int, default=10)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Save YOLO-like curves, confusion matrices, and sample image grids.")
    parser.add_argument("--curve-iou", type=float, default=0.50, help="IoU threshold used for P/R/F1 confidence curves.")
    parser.add_argument("--curve-points", type=int, default=101, help="Number of confidence thresholds for curves.")
    parser.add_argument("--confusion-conf", type=float, default=0.25, help="Confidence threshold for confusion matrix and image plots.")
    parser.add_argument("--confusion-iou", type=float, default=0.50, help="IoU threshold for confusion matrix matching.")
    parser.add_argument("--plot-batches", type=int, default=3, help="Number of image-grid batches to save per split.")
    parser.add_argument("--plot-batch-size", type=int, default=8, help="Images per saved image-grid batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_data_yaml(args.data)
    num_classes = len(data.names)
    device = parse_device(args.device)

    weights_path = repo_path(args.weights)
    model_state, ckpt_args = load_checkpoint(weights_path, device)

    width_mult = args.width_mult if args.width_mult is not None else float(ckpt_args.get("width_mult", 1.0))
    image_width = args.image_width if args.image_width is not None else int(ckpt_args.get("image_width", 320))
    image_height = args.image_height if args.image_height is not None else int(ckpt_args.get("image_height", 240))

    print(f"Dataset root: {data.root}")
    print(f"Classes: {dict(enumerate(data.names))}")
    print(f"Weights: {weights_path}")
    print(f"Eval project: {repo_path(args.eval_project)}")
    print(f"Device: {device}")
    print(f"Model size: width_mult={width_mult}, image_size=({image_width}, {image_height})")

    model = mobilenet_v2_ssd_sigmoid(
        num_classes=num_classes,
        size=(image_width, image_height),
        width_mult=width_mult,
        relu6=bool(ckpt_args.get("relu6", True)),
        match_init=bool(ckpt_args.get("match_init", True)),
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        detections_per_img=args.detections_per_img,
    ).to(device)
    model.load_state_dict(model_state)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    summary: dict[str, Any] = {}
    for split in splits:
        summary[split] = evaluate_split(split, model, data, device, args)

    summary_path = repo_path(args.eval_project) / "summary_metrics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
