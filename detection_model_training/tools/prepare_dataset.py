from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

from PIL import Image

CLASS_SCHEMAS = {
    "easy": {0: "target_bottle", 1: "target_can"},
    "medium": {0: "target_bottle", 1: "target_can", 2: "desk"},
    "hard": {0: "target_bottle", 1: "target_can", 2: "desk"},
}
TARGET_CLASS_IDS = {0, 1}

DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.10
DEFAULT_SPLIT_SEED = 2026

DEFAULT_MIN_TARGET_WIDTH_PX = 3.0
DEFAULT_MIN_TARGET_HEIGHT_PX = 3.0
DEFAULT_MIN_TARGET_AREA_PX = 9.0


def normalize_environment(value: str) -> str:
    env = str(value).strip().lower()
    if env not in CLASS_SCHEMAS:
        raise ValueError(f"Unknown environment '{value}'. Expected one of: {', '.join(CLASS_SCHEMAS)}")
    return env


def parse_seed(text: str) -> int:
    match = re.search(r"seed(\d+)", text)
    return int(match.group(1)) if match else -1


def load_bad_entries(values: Iterable[str], files: Iterable[Path]) -> Set[str]:
    bad = {str(v).strip() for v in values if str(v).strip()}
    for path in files:
        if path is None:
            continue
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            value = raw.strip()
            if value and not value.startswith("#"):
                bad.add(value)
    return bad


def load_manifest(dataset_dir: Path) -> List[dict]:
    path = dataset_dir / "manifest.jsonl"
    if not path.exists():
        return []
    rows: List[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))
    return rows


def image_entries(dataset_dir: Path) -> List[dict]:
    manifest_rows = load_manifest(dataset_dir)
    if manifest_rows:
        return manifest_rows

    entries: List[dict] = []
    images_dir = dataset_dir / "images"
    for image_path in sorted(images_dir.glob("*.png")):
        entries.append(
            {
                "entry_id": image_path.stem,
                "seed": parse_seed(image_path.stem),
                "image_filename": image_path.name,
                "label_filename": image_path.stem + ".txt",
            }
        )
    return entries


def entry_keys(entry: dict, sequential_index: int) -> Set[str]:
    image = str(entry.get("image_filename", ""))
    label = str(entry.get("label_filename", ""))
    entry_id = str(entry.get("entry_id") or Path(image).stem)
    return {
        str(sequential_index),
        entry_id,
        image,
        label,
        Path(image).stem,
        Path(label).stem,
    }


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    labels: List[Tuple[int, float, float, float, float]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label in {label_path} line {line_number}: {raw_line!r}")
        class_id = int(float(parts[0]))
        cx, cy, width, height = (float(v) for v in parts[1:])
        labels.append((class_id, cx, cy, width, height))
    return labels


def validate_labels(label_path: Path, allowed_class_ids: Set[int]) -> None:
    for class_id, cx, cy, width, height in read_yolo_labels(label_path):
        if class_id not in allowed_class_ids:
            raise ValueError(
                f"Label {label_path} contains class id {class_id}, expected {sorted(allowed_class_ids)}"
            )
        for name, value in (("cx", cx), ("cy", cy), ("width", width), ("height", height)):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Label {label_path} has {name}={value}, expected normalized value in [0,1]")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Label {label_path} has a non-positive bbox size")


def tiny_target_boxes(
    *,
    image_path: Path,
    label_path: Path,
    min_width_px: float,
    min_height_px: float,
    min_area_px: float,
) -> List[dict]:
    labels = read_yolo_labels(label_path)
    if not labels:
        return []
    with Image.open(image_path) as image:
        image_w, image_h = image.size

    tiny: List[dict] = []
    for class_id, _cx, _cy, box_w_norm, box_h_norm in labels:
        if class_id not in TARGET_CLASS_IDS:
            continue
        box_w_px = max(0.0, box_w_norm * image_w)
        box_h_px = max(0.0, box_h_norm * image_h)
        box_area_px = box_w_px * box_h_px
        if box_w_px < min_width_px or box_h_px < min_height_px or box_area_px < min_area_px:
            tiny.append(
                {
                    "class_id": int(class_id),
                    "bbox_width_px": float(box_w_px),
                    "bbox_height_px": float(box_h_px),
                    "bbox_area_px": float(box_area_px),
                }
            )
    return tiny


def split_by_seed(entries: Sequence[dict]) -> dict[int, str]:
    seeds = sorted({int(e.get("seed", parse_seed(str(e.get("image_filename", ""))))) for e in entries})
    seeds = [s for s in seeds if s >= 0]
    if len(seeds) < 3:
        raise ValueError("Seed-based splitting requires at least three distinct non-negative seeds")
    mapping = {seed: "train" for seed in seeds}
    mapping[seeds[-2]] = "val"
    mapping[seeds[-1]] = "test"
    return mapping


def _ratio_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")
    if n <= 0:
        return 0, 0, 0
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n
    if n >= 3:
        if val_n == 0:
            val_n = 1
            train_n = max(1, train_n - 1)
        if test_n == 0:
            test_n = 1
            train_n = max(1, train_n - 1)
        train_n = n - val_n - test_n
    return train_n, val_n, test_n


def split_entries_randomly(
    entries: Sequence[dict],
    *,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[int, str]:
    order = list(range(len(entries)))
    random.Random(int(split_seed)).shuffle(order)
    train_n, val_n, test_n = _ratio_counts(len(order), train_ratio, val_ratio, test_ratio)
    result: dict[int, str] = {}
    for position, idx in enumerate(order):
        if position < train_n:
            result[idx] = "train"
        elif position < train_n + val_n:
            result[idx] = "val"
        else:
            result[idx] = "test"
    return result


def yolo_yaml_text(class_id_to_name: dict[int, str]) -> str:
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in sorted(class_id_to_name.items()))
    return (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"{names}\n"
    )


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    environment: str,
    split_mode: str = "auto",
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    bad_entries: Set[str] | None = None,
    tiny_target_filter: bool | None = None,
    min_target_width_px: float = DEFAULT_MIN_TARGET_WIDTH_PX,
    min_target_height_px: float = DEFAULT_MIN_TARGET_HEIGHT_PX,
    min_target_area_px: float = DEFAULT_MIN_TARGET_AREA_PX,
) -> Path:
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    environment = normalize_environment(environment)
    bad_entries = bad_entries or set()

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Expected images/ and labels/ under {source_dir}")

    schema = CLASS_SCHEMAS[environment]
    allowed_class_ids = set(schema)
    entries = image_entries(source_dir)
    if not entries:
        raise RuntimeError(f"No PNG images found under {images_dir}")

    if tiny_target_filter is None:
        tiny_target_filter = environment == "hard"

    candidate_entries: List[dict] = []
    removed_entries: List[dict] = []
    removal_counts = {"manual_bad_entry": 0, "missing_image": 0, "tiny_target_bbox": 0}

    for idx, entry in enumerate(entries):
        image_name = str(entry.get("image_filename", ""))
        label_name = str(entry.get("label_filename") or (Path(image_name).stem + ".txt"))
        image_path = images_dir / image_name
        label_path = labels_dir / label_name

        if entry_keys(entry, idx) & bad_entries:
            removed = dict(entry, preparation_reason="manual_bad_entry")
            removed_entries.append(removed)
            removal_counts["manual_bad_entry"] += 1
            continue
        if not image_path.exists():
            removed = dict(entry, preparation_reason="missing_image")
            removed_entries.append(removed)
            removal_counts["missing_image"] += 1
            continue

        validate_labels(label_path, allowed_class_ids)
        if tiny_target_filter:
            tiny = tiny_target_boxes(
                image_path=image_path,
                label_path=label_path,
                min_width_px=min_target_width_px,
                min_height_px=min_target_height_px,
                min_area_px=min_target_area_px,
            )
            if tiny:
                removed = dict(entry, preparation_reason="tiny_target_bbox", tiny_target_boxes=tiny)
                removed_entries.append(removed)
                removal_counts["tiny_target_bbox"] += 1
                continue

        candidate_entries.append(
            {
                **entry,
                "_source_image": image_path,
                "_source_label": label_path,
                "seed": int(entry.get("seed", parse_seed(image_name))),
            }
        )

    if not candidate_entries:
        raise RuntimeError("All dataset entries were removed during preparation")

    unique_seeds = sorted({int(e["seed"]) for e in candidate_entries if int(e["seed"]) >= 0})
    mode = str(split_mode).lower()
    if mode == "auto":
        mode = "seed" if len(unique_seeds) >= 3 else "shuffled"
    if mode == "seed":
        seed_split = split_by_seed(candidate_entries)
        entry_split = {idx: seed_split[int(entry["seed"])] for idx, entry in enumerate(candidate_entries)}
    elif mode == "shuffled":
        entry_split = split_entries_randomly(
            candidate_entries,
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    else:
        raise ValueError("split_mode must be one of: auto, seed, shuffled")

    split_counts = {"train": 0, "val": 0, "test": 0}
    for split in split_counts:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []
    for idx, entry in enumerate(candidate_entries):
        split = entry_split[idx]
        split_counts[split] += 1
        new_stem = f"{split}_{split_counts[split]:06d}"
        dst_image = output_dir / "images" / split / f"{new_stem}.png"
        dst_label = output_dir / "labels" / split / f"{new_stem}.txt"
        shutil.copy2(entry["_source_image"], dst_image)
        if entry["_source_label"].exists():
            shutil.copy2(entry["_source_label"], dst_label)
        else:
            dst_label.write_text("", encoding="utf-8")

        row = {k: v for k, v in entry.items() if not k.startswith("_source_")}
        row.update(
            {
                "source_collection": source_dir.name,
                "prepared_split": split,
                "prepared_index": split_counts[split],
                "prepared_image_filename": f"images/{split}/{new_stem}.png",
                "prepared_label_filename": f"labels/{split}/{new_stem}.txt",
            }
        )
        manifest_rows.append(row)

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "removed_entries.jsonl").open("w", encoding="utf-8") as f:
        for row in removed_entries:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    metadata = {
        "environment": environment,
        "source_collection": source_dir.name,
        "class_id_to_name": schema,
        "preparation": {
            "split_mode_requested": split_mode,
            "split_mode_used": mode,
            "split_seed": int(split_seed),
            "split_counts": split_counts,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "source_seeds": unique_seeds,
            "kept_count": len(manifest_rows),
            "removed_count": len(removed_entries),
            "removal_counts": removal_counts,
            "tiny_target_filter": {
                "enabled": bool(tiny_target_filter),
                "target_class_ids": sorted(TARGET_CLASS_IDS),
                "min_target_width_px": float(min_target_width_px),
                "min_target_height_px": float(min_target_height_px),
                "min_target_area_px": float(min_target_area_px),
                "discard_rule": "discard whole image/label pair when any bottle/can bbox violates a threshold",
            },
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "target_objects.yaml").write_text(yolo_yaml_text(schema), encoding="utf-8")

    print(f"[DATASET PREP] source={source_dir}")
    print(f"[DATASET PREP] output={output_dir}")
    print(f"[DATASET PREP] environment={environment} split_mode={mode} counts={split_counts}")
    print(f"[DATASET PREP] kept={len(manifest_rows)} removed={len(removed_entries)}")
    if tiny_target_filter:
        print(
            "[DATASET PREP] hard target filter: "
            f"w<{min_target_width_px:g}px h<{min_target_height_px:g}px area<{min_target_area_px:g}px^2"
        )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one raw Webots collection folder as a train/val/test YOLO dataset."
    )
    parser.add_argument("source_dir", type=Path, help="Raw collection containing images/ and labels/.")
    parser.add_argument("output_dir", type=Path, help="New prepared dataset directory (must not exist).")
    parser.add_argument("--environment", required=True, choices=sorted(CLASS_SCHEMAS))
    parser.add_argument("--split-mode", choices=("auto", "seed", "shuffled"), default="auto")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--bad-entry", action="append", default=[])
    parser.add_argument("--bad-file", action="append", type=Path, default=[])
    tiny = parser.add_mutually_exclusive_group()
    tiny.add_argument("--tiny-target-filter", dest="tiny_target_filter", action="store_true")
    tiny.add_argument("--no-tiny-target-filter", dest="tiny_target_filter", action="store_false")
    parser.set_defaults(tiny_target_filter=None)
    parser.add_argument("--min-target-width-px", type=float, default=DEFAULT_MIN_TARGET_WIDTH_PX)
    parser.add_argument("--min-target-height-px", type=float, default=DEFAULT_MIN_TARGET_HEIGHT_PX)
    parser.add_argument("--min-target-area-px", type=float, default=DEFAULT_MIN_TARGET_AREA_PX)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_dataset(
        args.source_dir,
        args.output_dir,
        environment=args.environment,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        bad_entries=load_bad_entries(args.bad_entry, args.bad_file),
        tiny_target_filter=args.tiny_target_filter,
        min_target_width_px=args.min_target_width_px,
        min_target_height_px=args.min_target_height_px,
        min_target_area_px=args.min_target_area_px,
    )


if __name__ == "__main__":
    main()
