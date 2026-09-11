#!/usr/bin/env python3
"""Generic MobileNetV2-SSD sigmoid trainer for the thesis datasets.

Place directly in detection_model_training/. It supports all three current
environments:
    easy   -> datasets/easy
    medium -> datasets/medium
    hard   -> datasets/hard

The custom detector keeps its native 320x240 model input, matching both the
Webots camera and the previous MobileNet training/deployment pipeline. All new
training and evaluation artifacts are created in detector-specific folders
under detection_model_training/. Final deployment checkpoints are published
separately under detection_model_training/checkpoints/.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageEnhance
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou


# -----------------------------------------------------------------------------
# Import the local mobilenetv2_ssd package.
# Current expected layout:
#
#   detection_model_training/
#     train_mobilenetv2_ssd_sigmoid.py
#     mobilenetv2_ssd/
#       __init__.py
#       mobilenetv2_ssd_sigmoid.py
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

ENV_ALIASES = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}
ENV_PRESETS = {
    "easy": {
        "dataset_dir": "datasets/easy",
        "model_dir": "mobilenetv2_ssd_easy",
        "run_name": "mbv2ssd_sigmoid_bottle_can_img320_v1",
        "eval_dir": "eval_runs",
    },
    "medium": {
        "dataset_dir": "datasets/medium",
        "model_dir": "mobilenetv2_ssd_medium",
        "run_name": "mbv2ssd_sigmoid_bottle_can_desk_img320_v1",
        "eval_dir": "eval_runs",
    },
    "hard": {
        "dataset_dir": "datasets/hard",
        "model_dir": "mobilenetv2_ssd_hard",
        "run_name": "mbv2ssd_sigmoid_bottle_can_desk_img320_v1",
        "eval_dir": "eval_runs",
    },
}


def repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def canonical_env(name: str) -> str:
    try:
        return ENV_ALIASES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown environment {name!r}") from exc


def write_runtime_dataset_yaml(env_name: str, model_root: Path) -> Path:
    preset = ENV_PRESETS[env_name]
    dataset_dir = PROJECT_ROOT / preset["dataset_dir"]
    source_yaml = dataset_dir / "target_objects.yaml"
    required = [dataset_dir / kind / split for kind in ("images", "labels") for split in ("train", "val", "test")]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Dataset split folders are missing:\n" + "\n".join(f"  - {p}" for p in missing))
    if not source_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")

    model_root.mkdir(parents=True, exist_ok=True)
    output_yaml = model_root / "target_objects_local.yaml"
    lines = source_yaml.read_text(encoding="utf-8").splitlines()
    out = []
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


def apply_env_defaults(args: argparse.Namespace) -> argparse.Namespace:
    env_name = canonical_env(args.env)
    args.env = env_name
    preset = ENV_PRESETS[env_name]
    model_root = PROJECT_ROOT / preset["model_dir"]
    model_root.mkdir(parents=True, exist_ok=True)
    if args.data is None:
        args.data = str(write_runtime_dataset_yaml(env_name, model_root))
    if args.output_dir is None:
        args.output_dir = str(model_root / "runs" / preset["run_name"])
    if args.eval_project is None:
        args.eval_project = str(model_root / preset["eval_dir"])
    return args


@dataclass
class DataConfig:
    root: Path
    names: list[str]


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
    def __init__(
        self,
        data: DataConfig,
        split: str,
        augment: bool = False,
        hflip_prob: float = 0.5,
        color_jitter: bool = True,
        min_box_size_px: float = 1.0,
    ) -> None:
        self.data = data
        self.split = split
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
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
                    # torchvision detection models reserve 0 for background.
                    labels.append(cls_id + 1)

        if not boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def _maybe_augment(self, image: Image.Image, boxes: Tensor) -> tuple[Image.Image, Tensor]:
        if not self.augment:
            return image, boxes

        width, _ = image.size

        if random.random() < self.hflip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - x2
                boxes[:, 2] = width - x1

        if self.color_jitter:
            # Conservative photometric augmentation. This avoids changing boxes.
            if random.random() < 0.5:
                factor = random.uniform(0.75, 1.25)
                image = ImageEnhance.Brightness(image).enhance(factor)
            if random.random() < 0.5:
                factor = random.uniform(0.75, 1.25)
                image = ImageEnhance.Contrast(image).enhance(factor)
            if random.random() < 0.5:
                factor = random.uniform(0.75, 1.25)
                image = ImageEnhance.Color(image).enhance(factor)

        return image, boxes

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_path = self.images[idx]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, labels = self._read_yolo_label(label_path, width, height)
        image, boxes = self._maybe_augment(image, boxes)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch: list[tuple[Tensor, dict[str, Tensor]]]) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_loader(
    data: DataConfig,
    split: str,
    batch_size: int,
    workers: int,
    shuffle: bool,
    augment: bool,
) -> DataLoader:
    dataset = YoloDetectionDataset(data, split=split, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        persistent_workers=workers > 0,
    )


def move_targets_to_device(targets: list[dict[str, Tensor]], device: torch.device) -> list[dict[str, Tensor]]:
    return [
        {
            "boxes": t["boxes"].to(device),
            "labels": t["labels"].to(device),
        }
        for t in targets
    ]


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_clip_norm: float,
    print_freq: int,
) -> dict[str, float]:
    model.train()
    running: dict[str, float] = {"loss": 0.0, "bbox_regression": 0.0, "classification": 0.0}
    n_batches = 0
    start = time.time()

    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}")
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}")
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        n_batches += 1
        running["loss"] += float(loss.detach().cpu())
        for key in ("bbox_regression", "classification"):
            if key in loss_dict:
                running[key] += float(loss_dict[key].detach().cpu())

        if step == 1 or step % print_freq == 0 or step == len(loader):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            avg_loss = running["loss"] / max(1, n_batches)
            print(
                f"epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
                f"loss={avg_loss:.4f} lr={lr:.3g} elapsed={elapsed:.1f}s",
                flush=True,
            )

    return {k: v / max(1, n_batches) for k, v in running.items()}


@torch.no_grad()
def evaluate_ap50(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> float:
    """Lightweight AP50 used during training for best-checkpoint selection."""
    model.eval()
    gt_by_class: list[dict[int, Tensor]] = [dict() for _ in range(num_classes + 1)]
    preds_by_class: list[list[tuple[int, float, Tensor]]] = [[] for _ in range(num_classes + 1)]
    image_idx = 0

    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            for cls in range(1, num_classes + 1):
                gt_by_class[cls][image_idx] = target["boxes"][target["labels"] == cls].cpu()

                keep = output["labels"].detach().cpu() == cls
                boxes = output["boxes"].detach().cpu()[keep]
                scores = output["scores"].detach().cpu()[keep]
                for box, score in zip(boxes, scores):
                    preds_by_class[cls].append((image_idx, float(score), box))
            image_idx += 1

    aps: list[float] = []
    for cls in range(1, num_classes + 1):
        aps.append(compute_class_ap(preds_by_class[cls], gt_by_class[cls], iou_threshold=0.50))
    return float(np.mean(aps)) if aps else 0.0


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
    matched = {img_id: torch.zeros(len(boxes), dtype=torch.bool) for img_id, boxes in gt_for_class.items()}
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

    # COCO-style 101-point interpolation.
    recall_grid = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in recall_grid:
        precision_at_r = precisions[recalls >= r]
        ap += float(precision_at_r.max()) if precision_at_r.size else 0.0
    return ap / 101.0


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_ap50: float,
    class_names: list[str],
    args: argparse.Namespace,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_ap50": best_ap50,
        "class_names": class_names,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": vars(args),
    }
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2-SSD sigmoid on easy, medium, or hard data")
    parser.add_argument("--env", type=str, default="medium", choices=sorted(ENV_ALIASES))
    parser.add_argument("--data", type=str, default=None, help="Optional dataset YAML override")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional training output override")
    parser.add_argument("--eval-project", type=str, default=None, help="Optional final-evaluation output override")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--lr-step", type=int, default=24)
    parser.add_argument("--lr-gamma", type=float, default=0.95)
    parser.add_argument("--width-mult", type=float, default=1.0, choices=[0.5, 0.75, 1.0])
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--relu6", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--match-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=1, help="Run AP50 validation every N epochs; 0 disables it")
    parser.add_argument("--print-freq", type=int, default=25)
    parser.add_argument("--final-eval", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = apply_env_defaults(parse_args())

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data = read_data_yaml(args.data)
    num_classes = len(data.names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Dataset root: {data.root}")
    print(f"Classes: {dict(enumerate(data.names))}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Model input: ({args.image_width}, {args.image_height})")
    if (args.image_width, args.image_height) != (320, 240):
        print(
            f"[WARNING] MobileNet input is {args.image_width}x{args.image_height}; "
            "the Webots camera/deployment path uses 320x240.",
            flush=True,
        )

    train_loader = build_loader(
        data,
        split="train",
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        augment=True,
    )
    val_loader = build_loader(
        data,
        split="val",
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        augment=False,
    )

    model = mobilenet_v2_ssd_sigmoid(
        num_classes=num_classes,
        size=(args.image_width, args.image_height),
        width_mult=args.width_mult,
        relu6=args.relu6,
        match_init=args.match_init,
    ).to(device)

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_gamma,
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 1
    best_ap50 = -1.0
    if args.resume:
        resume_path = repo_path(args.resume)
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler is not None and "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_ap50 = float(ckpt.get("best_ap50", -1.0))
        print(f"Resumed from {resume_path} at epoch {start_epoch}; best_ap50={best_ap50:.4f}")

    history_path = output_dir / "history.csv"
    if not history_path.exists():
        history_path.write_text("epoch,loss,bbox_regression,classification,val_ap50,lr\n", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            grad_clip_norm=args.grad_clip_norm,
            print_freq=args.print_freq,
        )

        val_ap50 = math.nan
        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            val_ap50 = evaluate_ap50(model, val_loader, device, num_classes)
            print(f"epoch={epoch:03d} val_AP50={val_ap50:.4f}", flush=True)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        with history_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_stats['loss']:.8f},{train_stats['bbox_regression']:.8f},"
                f"{train_stats['classification']:.8f},{val_ap50:.8f},{lr:.10g}\n"
            )

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_ap50,
            data.names,
            args,
            scaler,
        )

        if not math.isnan(val_ap50) and val_ap50 > best_ap50:
            best_ap50 = val_ap50
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_ap50,
                data.names,
                args,
                scaler,
            )
            print(f"Saved new best checkpoint: AP50={best_ap50:.4f}", flush=True)

    print("Training complete.")
    print(f"Last checkpoint: {output_dir / 'last.pt'}")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")

    best_path = output_dir / "best.pt"
    if args.final_eval:
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found for final evaluation: {best_path}")
        from eval_mobilenetv2_ssd_sigmoid import evaluate_checkpoint
        evaluate_checkpoint(
            env=args.env,
            weights=best_path,
            data=args.data,
            eval_project=args.eval_project,
            split="both",
            device="auto",
        )


if __name__ == "__main__":
    main()
