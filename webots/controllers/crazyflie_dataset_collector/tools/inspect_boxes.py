# tools/inspect_boxes.py

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------
# Paste the relative path to the dataset run folder you want to inspect.
# This path is interpreted relative to the location of this script file.
# ---------------------------------------------------------------------
DATASET_RUN_REL_PATH = "../datasets/mixed_bottle_can_dataset_raw/run_seed0050_pseudo_random_20260504_145955"

# Save projected copies even for empty-label frames
SAVE_EMPTY_OVERLAYS = True

# If True, copies suspicious projected images into bbox_projections/suspects/
COPY_SUSPECTS = True

# Heuristics for suspicious boxes
THIN_BOX_MIN_SIDE_PX = 10
SMALL_BOX_AREA_PX = 120
BORDER_MARGIN_PX = 0
CENTER_SAFE_MARGIN_PX = 20

# Class legend for overlay text
CLASS_NAMES = {
    0: "bottle",
    1: "can",
}


def parse_yolo_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    out = []
    for line_idx, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO line in {label_path} at line {line_idx}: expected 5 fields, got {len(parts)}"
            )
        cls_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        out.append((cls_id, cx, cy, w, h))
    return out


def yolo_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    cx_px = cx * img_w
    cy_px = cy * img_h
    w_px = w * img_w
    h_px = h * img_h

    x1 = cx_px - w_px / 2.0
    y1 = cy_px - h_px / 2.0
    x2 = cx_px + w_px / 2.0
    y2 = cy_px + h_px / 2.0
    return x1, y1, x2, y2


def clamp_xyxy(
    box: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(0, min(int(round(x2)), img_w - 1))
    y2 = max(0, min(int(round(y2)), img_h - 1))
    return x1, y1, x2, y2


def touches_border(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    margin: int = 0,
) -> bool:
    x1, y1, x2, y2 = box
    return (
        x1 <= margin
        or y1 <= margin
        or x2 >= (img_w - 1 - margin)
        or y2 >= (img_h - 1 - margin)
    )


def box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def box_size(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1), max(0, y2 - y1)


def classify_box_suspicions(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> List[str]:
    reasons: List[str] = []

    w_px, h_px = box_size(box)
    area_px = w_px * h_px
    cx_px, cy_px = box_center(box)

    border = touches_border(box, img_w, img_h, margin=BORDER_MARGIN_PX)
    center_far_from_border = (
        CENTER_SAFE_MARGIN_PX <= cx_px <= (img_w - 1 - CENTER_SAFE_MARGIN_PX)
        and CENTER_SAFE_MARGIN_PX <= cy_px <= (img_h - 1 - CENTER_SAFE_MARGIN_PX)
    )

    if w_px < THIN_BOX_MIN_SIDE_PX or h_px < THIN_BOX_MIN_SIDE_PX:
        reasons.append("thin_box")

    if area_px < SMALL_BOX_AREA_PX:
        reasons.append("small_area")

    if border:
        reasons.append("touches_border")

    if center_far_from_border and area_px < SMALL_BOX_AREA_PX:
        reasons.append("tiny_but_centered")

    return reasons


def draw_boxes(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    classes: List[int],
    per_box_reasons: List[List[str]],
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)

    for i, (box, cls_id, reasons) in enumerate(zip(boxes, classes, per_box_reasons), start=1):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        reason_suffix = ""
        if reasons:
            reason_suffix = "|" + ",".join(reasons[:2])

        label = f"{cls_name}:{i}{reason_suffix}"
        text_x = x1 + 2
        text_y = max(0, y1 - 12)
        text_w = max(42, 7 * len(label))
        draw.rectangle([text_x - 1, text_y - 1, text_x + text_w, text_y + 10], fill=(255, 0, 0))
        draw.text((text_x, text_y), label, fill=(255, 255, 255))

    return out


def resolve_run_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    return (script_dir / DATASET_RUN_REL_PATH).resolve()


def inspect_run(run_dir: Path) -> None:
    images_dir = run_dir / "images"
    labels_dir = run_dir / "labels"

    output_dir = run_dir / "bbox_projections"
    suspects_dir = output_dir / "suspects"

    output_dir.mkdir(parents=True, exist_ok=True)
    if COPY_SUSPECTS:
        suspects_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "bbox_projection_summary.csv"
    empty_txt = output_dir / "empty_images.txt"
    empty_csv = output_dir / "empty_images.csv"

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")

    rows = []
    suspect_count = 0
    empty_images: List[str] = []

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"

        with Image.open(img_path) as im:
            image = im.convert("RGB")
            img_w, img_h = image.size

        anns = parse_yolo_label_file(label_path)

        pixel_boxes: List[Tuple[int, int, int, int]] = []
        class_ids: List[int] = []
        per_box_reasons: List[List[str]] = []

        for cls_id, cx, cy, w, h in anns:
            xyxy = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            box = clamp_xyxy(xyxy, img_w, img_h)
            reasons = classify_box_suspicions(box, img_w, img_h)

            pixel_boxes.append(box)
            class_ids.append(cls_id)
            per_box_reasons.append(reasons)

        if len(pixel_boxes) == 0:
            empty_images.append(img_path.stem)

        should_save_overlay = bool(pixel_boxes) or SAVE_EMPTY_OVERLAYS
        overlay_path = output_dir / img_path.name

        if should_save_overlay:
            with Image.open(img_path) as im:
                image = im.convert("RGB")
            projected = draw_boxes(image, pixel_boxes, class_ids, per_box_reasons)
            projected.save(overlay_path)

        image_reasons = sorted({reason for reasons in per_box_reasons for reason in reasons})
        class_names_present = sorted({CLASS_NAMES.get(cls_id, str(cls_id)) for cls_id in class_ids})
        is_suspect = len(image_reasons) > 0

        if COPY_SUSPECTS and is_suspect and overlay_path.exists():
            shutil.copy2(overlay_path, suspects_dir / overlay_path.name)
            suspect_count += 1

        rows.append(
            {
                "image": img_path.name,
                "label": label_path.name,
                "num_boxes": len(pixel_boxes),
                "classes_present": ";".join(class_names_present),
                "empty": int(len(pixel_boxes) == 0),
                "suspect": int(is_suspect),
                "suspect_reasons": ";".join(image_reasons),
                "any_border_touch": int(any("touches_border" in r for r in per_box_reasons)),
                "all_border_touch": int(all("touches_border" in r for r in per_box_reasons) if pixel_boxes else 0),
            }
        )

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "label",
                "num_boxes",
                "classes_present",
                "empty",
                "suspect",
                "suspect_reasons",
                "any_border_touch",
                "all_border_touch",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with empty_txt.open("w", encoding="utf-8") as f:
        for stem in empty_images:
            f.write(stem + "\n")

    with empty_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_stem"])
        for stem in empty_images:
            writer.writerow([stem])

    total_images = len(image_paths)
    empty_count = len(empty_images)
    empty_pct = (100.0 * empty_count / total_images) if total_images > 0 else 0.0

    print(f"[DONE] run_dir: {run_dir}")
    print(f"[DONE] wrote overlays to: {output_dir}")
    print(f"[DONE] wrote summary CSV to: {summary_csv}")
    if COPY_SUSPECTS:
        print(f"[DONE] suspect overlays copied to: {suspects_dir}")
        print(f"[DONE] suspect image count: {suspect_count}")
    print(f"[DONE] processed {total_images} images")
    print(f"[DONE] empty images: {empty_count}/{total_images} ({empty_pct:.2f}%)")
    print(f"[DONE] empty image list saved to: {empty_txt}")
    print(f"[DONE] empty image CSV saved to: {empty_csv}")


if __name__ == "__main__":
    inspect_run(resolve_run_dir())