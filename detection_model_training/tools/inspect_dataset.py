from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw


CLASS_SCHEMAS = {
    "easy": {0: "target_bottle", 1: "target_can"},
    "medium": {0: "target_bottle", 1: "target_can", 2: "desk"},
    "hard": {0: "target_bottle", 1: "target_can", 2: "desk"},
}

TARGET_CLASS_IDS = {0, 1}
DESK_CLASS_ID = 2

COMPOSITION_CATEGORIES = (
    "empty",
    "target_only",
    "desk_only",
    "target_and_desk",
)


def _iter_images(dataset_dir: Path) -> Iterable[Tuple[Path, Path, str]]:
    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"

    if not images_root.exists() or not labels_root.exists():
        raise FileNotFoundError(
            f"Expected images/ and labels/ under dataset directory: {dataset_dir}"
        )

    for image_path in sorted(images_root.rglob("*.png")):
        rel = image_path.relative_to(images_root)
        label_path = labels_root / rel.with_suffix(".txt")
        split = str(rel.parent) if str(rel.parent) != "." else "raw"
        yield image_path, label_path, split


def _read_labels(path: Path):
    rows = []
    if not path.exists():
        return rows

    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) != 5:
            continue

        rows.append(
            (
                int(float(parts[0])),
                *(float(v) for v in parts[1:]),
            )
        )

    return rows


def _composition_category(labels) -> str:
    has_target = any(class_id in TARGET_CLASS_IDS for class_id, *_ in labels)
    has_desk = any(class_id == DESK_CLASS_ID for class_id, *_ in labels)

    if not labels:
        return "empty"
    if has_target and has_desk:
        return "target_and_desk"
    if has_target:
        return "target_only"
    if has_desk:
        return "desk_only"

    # A canonical dataset should never reach this branch because all valid
    # classes are targets or desks. Keep it explicit instead of silently
    # counting an unknown-class sample as empty.
    raise ValueError(
        "Encountered a non-empty label file containing no target/desk classes."
    )


def _print_summary(
    *,
    total_images: int,
    composition_counts: Counter,
    class_counts: Counter,
    schema: dict[int, str],
) -> None:
    print()
    print("[COMPOSITION]")

    for category in COMPOSITION_CATEGORIES:
        count = int(composition_counts.get(category, 0))
        pct = (100.0 * count / total_images) if total_images else 0.0
        print(f"  {category:16s}: {count:5d}/{total_images:<5d} ({pct:6.2f}%)")

    non_empty = total_images - int(composition_counts.get("empty", 0))
    non_empty_pct = (100.0 * non_empty / total_images) if total_images else 0.0
    print(
        f"  {'non_empty':16s}: {non_empty:5d}/{total_images:<5d} "
        f"({non_empty_pct:6.2f}%)"
    )

    print()
    print("[CLASS INSTANCES]")
    total_instances = 0
    for class_id, class_name in schema.items():
        count = int(class_counts.get(class_id, 0))
        total_instances += count
        print(f"  {class_name:16s}: {count}")

    print(f"  {'total':16s}: {total_instances}")


def inspect_dataset(
    dataset_dir: Path,
    *,
    environment: str,
    min_target_side_px: int = 3,
) -> Path:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    schema = CLASS_SCHEMAS[environment]

    out_dir = dataset_dir / "inspection"
    boxed_dir = out_dir / "boxed"
    boxed_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    composition_counts: Counter = Counter()
    class_counts: Counter = Counter()

    # Useful when inspecting an already prepared dataset. Raw collections use
    # only the "raw" bucket.
    split_composition_counts = defaultdict(Counter)
    split_image_counts = Counter()

    for image_path, label_path, split in _iter_images(dataset_dir):
        with Image.open(image_path) as im:
            image = im.convert("RGB")

        draw = ImageDraw.Draw(image)
        width, height = image.size
        labels = _read_labels(label_path)

        for class_id, *_ in labels:
            if class_id not in schema:
                raise ValueError(
                    f"Unexpected class id {class_id} in {label_path}; "
                    f"expected one of {sorted(schema)} for environment={environment!r}."
                )
            class_counts[class_id] += 1

        composition = _composition_category(labels)
        composition_counts[composition] += 1
        split_composition_counts[split][composition] += 1
        split_image_counts[split] += 1

        tiny_target = False
        edge_touching = False

        for class_id, cx, cy, bw, bh in labels:
            x1 = max(0, int(round((cx - bw / 2.0) * width)))
            y1 = max(0, int(round((cy - bh / 2.0) * height)))
            x2 = min(width, int(round((cx + bw / 2.0) * width)))
            y2 = min(height, int(round((cy + bh / 2.0) * height)))

            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)

            tiny_target = tiny_target or (
                class_id in TARGET_CLASS_IDS
                and (
                    box_w < min_target_side_px
                    or box_h < min_target_side_px
                )
            )

            edge_touching = (
                edge_touching
                or x1 <= 0
                or y1 <= 0
                or x2 >= width
                or y2 >= height
            )

            draw.rectangle((x1, y1, x2, y2), width=2)
            draw.text(
                (x1 + 2, max(0, y1 - 12)),
                schema.get(class_id, f"class_{class_id}"),
            )

        prefix = "raw" if split == "raw" else split.replace("/", "_")
        image.save(boxed_dir / f"{prefix}__{image_path.name}")

        per_image_class_counts = Counter(class_id for class_id, *_ in labels)
        rows.append(
            {
                "image": str(image_path.relative_to(dataset_dir)),
                "label": str(label_path.relative_to(dataset_dir)),
                "split": split,
                "num_boxes": len(labels),
                "composition": composition,
                "num_target_bottle": per_image_class_counts.get(0, 0),
                "num_target_can": per_image_class_counts.get(1, 0),
                "num_desk": per_image_class_counts.get(2, 0),
                "tiny_target": int(tiny_target),
                "edge_touching": int(edge_touching),
            }
        )

    report = out_dir / "inspection_report.csv"
    fieldnames = [
        "image",
        "label",
        "split",
        "num_boxes",
        "composition",
        "num_target_bottle",
        "num_target_can",
        "num_desk",
        "tiny_target",
        "edge_touching",
    ]

    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_images = len(rows)

    print(f"[INSPECT] images={total_images} overlays={boxed_dir}")
    print(f"[INSPECT] report={report}")

    _print_summary(
        total_images=total_images,
        composition_counts=composition_counts,
        class_counts=class_counts,
        schema=schema,
    )

    if len(split_image_counts) > 1:
        print()
        print("[COMPOSITION BY SPLIT]")
        for split in sorted(split_image_counts):
            split_total = int(split_image_counts[split])
            print(f"  {split}:")
            for category in COMPOSITION_CATEGORIES:
                count = int(split_composition_counts[split].get(category, 0))
                pct = (100.0 * count / split_total) if split_total else 0.0
                print(
                    f"    {category:16s}: "
                    f"{count:5d}/{split_total:<5d} ({pct:6.2f}%)"
                )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only visual QA and composition statistics for raw or "
            "prepared YOLO datasets."
        )
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--environment",
        required=True,
        choices=sorted(CLASS_SCHEMAS),
    )
    parser.add_argument(
        "--min-target-side-px",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if args.min_target_side_px < 1:
        raise ValueError("--min-target-side-px must be >= 1")

    inspect_dataset(
        args.dataset_dir,
        environment=args.environment,
        min_target_side_px=args.min_target_side_px,
    )


if __name__ == "__main__":
    main()
