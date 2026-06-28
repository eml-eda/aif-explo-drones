# tools/inspect_boxes_tables.py
# python3 inspect_boxes_tables.py ../datasets/<collection_name> \
# --min-target-visible-px 5 \
# --prune-tiny-targets
# or:   --delete-tiny-targets


from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

CLASS_NAMES = {
    0: "target_bottle",
    1: "target_can",
    2: "desk",
}
TARGET_CLASS_IDS = {0, 1}
DESK_CLASS_ID = 2

MIN_TARGET_AREA_FRAC = 0.00035
MIN_DESK_AREA_FRAC = 0.01
MAX_TARGET_ASPECT_RATIO = 7.0
MAX_DESK_ASPECT_RATIO = 18.0
FULL_FRAME_AREA_FRAC = 0.92

DEFAULT_MIN_TARGET_VISIBLE_PX = 8

COLORS = {
    0: (30, 144, 255),
    1: (255, 140, 0),
    2: (0, 180, 90),
}


def parse_manifest(dataset_dir: Path) -> Dict[str, dict]:
    manifest_path = dataset_dir / "manifest.jsonl"
    out: Dict[str, dict] = {}
    if not manifest_path.exists():
        return out
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = row.get("label_filename")
            image = row.get("image_filename")
            if label:
                out[str(label)] = row
            if image:
                out[str(image)] = row
    return out


def parse_yolo_label_file(label_path: Path) -> Tuple[List[Tuple[int, float, float, float, float]], List[str]]:
    if not label_path.exists():
        return [], ["missing_label"]
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return [], []
    anns = []
    errors = []
    for line_idx, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"malformed_line_{line_idx}")
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"non_numeric_line_{line_idx}")
            continue
        anns.append((cls_id, cx, cy, w, h))
    return anns, errors


def yolo_to_xyxy(ann, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    _, cx, cy, w, h = ann
    cx_px = cx * img_w
    cy_px = cy * img_h
    w_px = w * img_w
    h_px = h * img_h
    return (
        cx_px - 0.5 * w_px,
        cy_px - 0.5 * h_px,
        cx_px + 0.5 * w_px,
        cy_px + 0.5 * h_px,
    )


def clamp_box(box, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(int(round(x1)), img_w)),
        max(0, min(int(round(y1)), img_h)),
        max(0, min(int(round(x2)), img_w)),
        max(0, min(int(round(y2)), img_h)),
    )


def box_width_height(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1), max(0, y2 - y1)


def touches_edge(box: Tuple[int, int, int, int], img_w: int, img_h: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= 0 or y1 <= 0 or x2 >= img_w or y2 >= img_h


def target_visible_side_too_small(
    cls_id: int,
    box: Tuple[int, int, int, int],
    min_target_visible_px: int,
) -> bool:
    if cls_id not in TARGET_CLASS_IDS:
        return False
    bw, bh = box_width_height(box)
    return bw < min_target_visible_px or bh < min_target_visible_px


def classify_suspicions(
    cls_id: int,
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    manifest_entry: dict | None,
    *,
    min_target_visible_px: int,
) -> List[str]:
    bw, bh = box_width_height(box)
    image_area = max(1, img_w * img_h)
    area_frac = (bw * bh) / float(image_area)
    aspect = max(bw / max(1, bh), bh / max(1, bw))
    reasons: List[str] = []

    if bw <= 0 or bh <= 0:
        reasons.append("invalid_or_zero_size")
    if touches_edge(box, img_w, img_h):
        reasons.append("touches_edge")

    if cls_id in TARGET_CLASS_IDS:
        if target_visible_side_too_small(cls_id, box, min_target_visible_px):
            reasons.append("target_visible_side_too_small")
        if area_frac < MIN_TARGET_AREA_FRAC:
            reasons.append("target_tiny_area")
        if aspect > MAX_TARGET_ASPECT_RATIO:
            reasons.append("target_extreme_aspect")
    elif cls_id == DESK_CLASS_ID:
        if area_frac < MIN_DESK_AREA_FRAC:
            reasons.append("desk_tiny_area")
        if aspect > MAX_DESK_ASPECT_RATIO:
            reasons.append("desk_extreme_aspect")
        if area_frac >= FULL_FRAME_AREA_FRAC and manifest_entry is None:
            reasons.append("desk_full_frame_missing_manifest")
        elif area_frac >= FULL_FRAME_AREA_FRAC and manifest_entry and not manifest_entry.get("contains_desk", False):
            reasons.append("desk_full_frame_manifest_inconsistent")
    else:
        reasons.append("unknown_class")

    return reasons


def draw_boxes(image: Image.Image, anns, boxes, reasons_per_box) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for ann, box, reasons in zip(anns, boxes, reasons_per_box):
        cls_id = ann[0]
        color = COLORS.get(cls_id, (255, 0, 0))
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = CLASS_NAMES.get(cls_id, str(cls_id))
        if reasons:
            text += "|" + ",".join(reasons[:2])
        tw = max(40, 7 * len(text))
        ty = max(0, y1 - 12)
        draw.rectangle([x1, ty, min(image.width, x1 + tw), ty + 11], fill=color)
        draw.text((x1 + 2, ty), text, fill=(255, 255, 255))
    return out


def _move_or_delete_pair(
    *,
    img_path: Path,
    label_path: Path,
    rejected_images_dir: Path,
    rejected_labels_dir: Path,
    delete: bool,
) -> str:
    if delete:
        img_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)
        return "deleted"

    rejected_images_dir.mkdir(parents=True, exist_ok=True)
    rejected_labels_dir.mkdir(parents=True, exist_ok=True)

    img_dst = rejected_images_dir / img_path.name
    label_dst = rejected_labels_dir / label_path.name

    if img_dst.exists():
        img_dst.unlink()
    if label_dst.exists():
        label_dst.unlink()

    shutil.move(str(img_path), str(img_dst))
    if label_path.exists():
        shutil.move(str(label_path), str(label_dst))
    return "moved_to_rejected_tiny_targets"


def inspect_dataset(
    dataset_dir: Path,
    *,
    min_target_visible_px: int = DEFAULT_MIN_TARGET_VISIBLE_PX,
    prune_tiny_targets: bool = False,
    delete_tiny_targets: bool = False,
) -> None:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Expected images/ and labels/ under {dataset_dir}")

    inspection_dir = dataset_dir / "inspection"
    boxed_all_dir = inspection_dir / "boxed_all"
    edge_dir = inspection_dir / "suspect_edge_touching"
    target_dir = inspection_dir / "suspect_targets"
    desk_dir = inspection_dir / "suspect_desks"
    rejected_dir = inspection_dir / "rejected_tiny_targets"
    rejected_images_dir = rejected_dir / "images"
    rejected_labels_dir = rejected_dir / "labels"

    for path in (boxed_all_dir, edge_dir, target_dir, desk_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = parse_manifest(dataset_dir)
    rows = []
    image_paths = sorted(images_dir.glob("*.png"))

    composition_counts = {
        "empty": 0,
        "target_only": 0,
        "table_only": 0,
        "target_and_table": 0,
    }
    tiny_target_reject_count = 0
    prune_action_counts = {
        "none": 0,
        "moved_to_rejected_tiny_targets": 0,
        "deleted": 0,
    }

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        anns, parse_errors = parse_yolo_label_file(label_path)

        has_target = any(ann[0] in TARGET_CLASS_IDS for ann in anns)
        has_desk = any(ann[0] == DESK_CLASS_ID for ann in anns)
        if not has_target and not has_desk:
            composition_category = "empty"
        elif has_target and not has_desk:
            composition_category = "target_only"
        elif has_desk and not has_target:
            composition_category = "table_only"
        else:
            composition_category = "target_and_table"
        composition_counts[composition_category] += 1

        manifest_entry = manifest.get(img_path.name) or manifest.get(label_path.name)
        with Image.open(img_path) as im:
            image = im.convert("RGB")
        img_w, img_h = image.size

        boxes = [clamp_box(yolo_to_xyxy(ann, img_w, img_h), img_w, img_h) for ann in anns]
        reasons_per_box = [
            classify_suspicions(
                ann[0],
                box,
                img_w,
                img_h,
                manifest_entry,
                min_target_visible_px=min_target_visible_px,
            )
            for ann, box in zip(anns, boxes)
        ]
        if parse_errors:
            reasons_per_box.append(parse_errors)

        tiny_target_boxes = [
            {
                "class_name": CLASS_NAMES.get(ann[0], str(ann[0])),
                "box_xyxy": list(box),
                "width_px": box_width_height(box)[0],
                "height_px": box_width_height(box)[1],
            }
            for ann, box in zip(anns, boxes)
            if target_visible_side_too_small(ann[0], box, min_target_visible_px)
        ]
        reject_tiny_target = bool(tiny_target_boxes)
        if reject_tiny_target:
            tiny_target_reject_count += 1

        overlay = draw_boxes(image, anns, boxes, reasons_per_box[: len(anns)])
        overlay_path = boxed_all_dir / img_path.name
        overlay.save(overlay_path)

        target_reasons = sorted({r for ann, reasons in zip(anns, reasons_per_box) if ann[0] in TARGET_CLASS_IDS for r in reasons})
        desk_reasons = sorted({r for ann, reasons in zip(anns, reasons_per_box) if ann[0] == DESK_CLASS_ID for r in reasons})
        edge_classes = sorted({CLASS_NAMES.get(ann[0], str(ann[0])) for ann, box in zip(anns, boxes) if touches_edge(box, img_w, img_h)})
        all_reasons = sorted(set(parse_errors + target_reasons + desk_reasons))

        if edge_classes:
            edge_class_dir = edge_dir / ("desk" if edge_classes == ["desk"] else "targets_or_mixed")
            edge_class_dir.mkdir(exist_ok=True)
            shutil.copy2(overlay_path, edge_class_dir / overlay_path.name)
        if target_reasons:
            shutil.copy2(overlay_path, target_dir / overlay_path.name)
        if desk_reasons:
            shutil.copy2(overlay_path, desk_dir / overlay_path.name)

        prune_action = "none"
        if reject_tiny_target and (prune_tiny_targets or delete_tiny_targets):
            prune_action = _move_or_delete_pair(
                img_path=img_path,
                label_path=label_path,
                rejected_images_dir=rejected_images_dir,
                rejected_labels_dir=rejected_labels_dir,
                delete=delete_tiny_targets,
            )
        prune_action_counts[prune_action] = prune_action_counts.get(prune_action, 0) + 1

        rows.append(
            {
                "entry_id": img_path.stem,
                "image": img_path.name,
                "label": label_path.name,
                "num_boxes": len(anns),
                "classes_present": ";".join(sorted({CLASS_NAMES.get(a[0], str(a[0])) for a in anns})),
                "edge_touching_classes": ";".join(edge_classes),
                "target_suspect_reasons": ";".join(target_reasons),
                "desk_suspect_reasons": ";".join(desk_reasons),
                "parse_errors": ";".join(parse_errors),
                "suspect": int(bool(all_reasons or edge_classes)),
                "all_reasons": ";".join(sorted(set(all_reasons + (["edge_touching"] if edge_classes else [])))),
                "composition_category": composition_category,
                "has_target": int(has_target),
                "has_desk": int(has_desk),
                "min_target_visible_px": int(min_target_visible_px),
                "reject_tiny_target": int(reject_tiny_target),
                "tiny_target_boxes_json": json.dumps(tiny_target_boxes, sort_keys=True),
                "prune_action": prune_action,
            }
        )

    report_path = inspection_dir / "inspection_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["entry_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] inspected {len(image_paths)} images")
    print(f"[DONE] overlays: {boxed_all_dir}")
    print(f"[DONE] report: {report_path}")

    total_images = max(1, len(image_paths))
    print("[COMPOSITION]")
    for key in ("empty", "target_only", "table_only", "target_and_table"):
        count = composition_counts[key]
        pct = 100.0 * count / total_images
        print(f"  {key}: {count}/{len(image_paths)} ({pct:.2f}%)")

    print("[TINY TARGET FILTER]")
    pct = 100.0 * tiny_target_reject_count / total_images
    print(
        f"  reject_tiny_target: {tiny_target_reject_count}/{len(image_paths)} "
        f"({pct:.2f}%) using min visible side >= {min_target_visible_px}px"
    )
    if prune_tiny_targets or delete_tiny_targets:
        print("[PRUNE]")
        print(f"  moved_to_rejected_tiny_targets: {prune_action_counts.get('moved_to_rejected_tiny_targets', 0)}")
        print(f"  deleted: {prune_action_counts.get('deleted', 0)}")
        if prune_action_counts.get("moved_to_rejected_tiny_targets", 0):
            print(f"  rejected images: {rejected_images_dir}")
            print(f"  rejected labels: {rejected_labels_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw and flag YOLO boxes for table dataset collections.")
    parser.add_argument("dataset_dir", type=Path, help="Path to collection_table_* dataset folder.")
    parser.add_argument(
        "--min-target-visible-px",
        type=int,
        default=DEFAULT_MIN_TARGET_VISIBLE_PX,
        help=(
            "Minimum visible bbox side length in pixels for target_bottle/target_can. "
            "If width OR height is below this value, the image is marked reject_tiny_target."
        ),
    )
    parser.add_argument(
        "--prune-tiny-targets",
        action="store_true",
        help=(
            "Move images/labels with at least one tiny target bbox to "
            "inspection/rejected_tiny_targets/{images,labels}. Safer than deletion."
        ),
    )
    parser.add_argument(
        "--delete-tiny-targets",
        action="store_true",
        help=(
            "Permanently delete images/labels with at least one tiny target bbox. "
            "Use only after checking the inspection report/overlays."
        ),
    )
    args = parser.parse_args()

    if args.min_target_visible_px < 1:
        raise ValueError("--min-target-visible-px must be >= 1")

    inspect_dataset(
        args.dataset_dir.resolve(),
        min_target_visible_px=int(args.min_target_visible_px),
        prune_tiny_targets=bool(args.prune_tiny_targets),
        delete_tiny_targets=bool(args.delete_tiny_targets),
    )


if __name__ == "__main__":
    main()