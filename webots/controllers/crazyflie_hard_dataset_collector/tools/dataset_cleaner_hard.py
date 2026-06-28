from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from PIL import Image

CLASS_ID_TO_NAME = {
    0: "target_bottle",
    1: "target_can",
    2: "desk",
}
TARGET_CLASS_IDS = {0, 1}

# The raw camera is 320 x 240. These defaults remove samples where a visible
# target reduces to only one or two pixels in either dimension, or fewer than
# nine pixels in total. Desk boxes never trigger this filter.
DEFAULT_MIN_TARGET_WIDTH_PX = 3.0
DEFAULT_MIN_TARGET_HEIGHT_PX = 3.0
DEFAULT_MIN_TARGET_AREA_PX = 9.0

# =============================================================================
# USER SETTINGS — edit only this block, then run:
#
#     python dataset_cleaner_hard_paste_paths.py
# =============================================================================

RAW_DATASET_DIR = Path(
    "/home/ale/dev/thesis_project/aif-explo-drones/webots/controllers/"
    "crazyflie_hard_dataset_collector/datasets/"
    "collection_hard_pseudo_random_20260628_132258"
)

OUTPUT_DATASET_DIR = Path(
    "/home/ale/dev/thesis_project/aif-explo-drones/detection_model_training/"
    "yolo11_s_mixed_hard/dataset_mixed_hard"
)

# Optional manual exclusions. Leave empty unless you later inspect the dataset
# and want to reject specific image stems, filenames, label filenames, or indices.
MANUAL_BAD_ENTRIES: list[str] = []
MANUAL_BAD_ENTRIES_FILE: Path | None = None

# Only bottle/can boxes are checked. Any frame containing a target box below
# any one of these limits is excluded completely. Desk labels are not filtered.
MIN_TARGET_WIDTH_PX = 3.0
MIN_TARGET_HEIGHT_PX = 3.0
MIN_TARGET_AREA_PX = 9.0



def load_bad_entries(values: Iterable[str], files: Iterable[Path]) -> Set[str]:
    bad = {v.strip() for v in values if v.strip()}
    for path in files:
        if not path:
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if value and not value.startswith("#"):
                    bad.add(value)
    return bad


def load_manifest(dataset_dir: Path) -> List[dict]:
    path = dataset_dir / "manifest.jsonl"
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_seed(text: str) -> int:
    match = re.search(r"seed(\d+)", text)
    if not match:
        return -1
    return int(match.group(1))


def image_entries(dataset_dir: Path, manifest_rows: List[dict]) -> List[dict]:
    if manifest_rows:
        return manifest_rows

    entries = []
    for image_path in sorted((dataset_dir / "images").glob("*.png")):
        stem = image_path.stem
        entries.append(
            {
                "entry_id": stem,
                "seed": parse_seed(stem),
                "image_filename": image_path.name,
                "label_filename": stem + ".txt",
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


def split_for_seed(seed: int, ordered_seeds: List[int]) -> str:
    if not ordered_seeds:
        return "train"
    if seed == ordered_seeds[-1]:
        return "test"
    if len(ordered_seeds) >= 2 and seed == ordered_seeds[-2]:
        return "val"
    return "train"


def yolo_yaml_text(dataset_path: Path) -> str:
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in CLASS_ID_TO_NAME.items())
    return (
        f"path: {dataset_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        f"{names}\n"
    )


def copy_optional(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Read valid YOLO detection lines as (class_id, cx, cy, width, height)."""
    if not label_path.exists():
        return []

    labels: List[Tuple[int, float, float, float, float]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label line in {label_path} at line {line_number}: {raw_line!r}")
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = (float(v) for v in parts[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric YOLO label in {label_path} at line {line_number}: {raw_line!r}") from exc
        labels.append((class_id, cx, cy, width, height))
    return labels


def _tiny_target_boxes(
    *,
    image_path: Path,
    label_path: Path,
    min_width_px: float,
    min_height_px: float,
    min_area_px: float,
) -> List[dict]:
    """Return target annotations too small to be meaningful training labels.

    Any such target causes the complete image/label pair to be discarded. This
    avoids silently converting a visibly tiny target into an unlabeled negative
    sample. Desk annotations are intentionally ignored by this quality rule.
    """
    labels = _read_yolo_labels(label_path)
    if not labels:
        return []

    with Image.open(image_path) as image:
        width_px, height_px = image.size

    tiny: List[dict] = []
    for class_id, _cx, _cy, box_w_norm, box_h_norm in labels:
        if class_id not in TARGET_CLASS_IDS:
            continue
        box_w_px = max(0.0, float(box_w_norm) * float(width_px))
        box_h_px = max(0.0, float(box_h_norm) * float(height_px))
        box_area_px = box_w_px * box_h_px
        if (
            box_w_px < float(min_width_px)
            or box_h_px < float(min_height_px)
            or box_area_px < float(min_area_px)
        ):
            tiny.append(
                {
                    "class_id": int(class_id),
                    "class_name": CLASS_ID_TO_NAME.get(int(class_id), str(class_id)),
                    "bbox_width_px": float(box_w_px),
                    "bbox_height_px": float(box_h_px),
                    "bbox_area_px": float(box_area_px),
                }
            )
    return tiny


def clean_dataset(
    dataset_dir: Path,
    bad_entries: Set[str],
    output_dir: Path | None = None,
    *,
    min_target_width_px: float = DEFAULT_MIN_TARGET_WIDTH_PX,
    min_target_height_px: float = DEFAULT_MIN_TARGET_HEIGHT_PX,
    min_target_area_px: float = DEFAULT_MIN_TARGET_AREA_PX,
) -> Path:
    if output_dir is None:
        output_dir = dataset_dir.with_name(dataset_dir.name + "_clean")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing clean dataset: {output_dir}")

    if min_target_width_px < 0 or min_target_height_px < 0 or min_target_area_px < 0:
        raise ValueError("Tiny-target thresholds must be non-negative.")

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Expected images/ and labels/ under {dataset_dir}")

    manifest_rows = load_manifest(dataset_dir)
    entries = image_entries(dataset_dir, manifest_rows)
    seeds = sorted({int(e.get("seed", parse_seed(str(e.get("image_filename", ""))))) for e in entries})

    split_counts = {"train": 0, "val": 0, "test": 0}
    kept_manifest: List[dict] = []
    removed_manifest: List[dict] = []
    removal_counts = {
        "manual_bad_entry": 0,
        "missing_image": 0,
        "tiny_target_bbox": 0,
    }
    tiny_target_box_count = 0

    for split in split_counts:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for idx, entry in enumerate(entries):
        image_name = str(entry.get("image_filename"))
        label_name = str(entry.get("label_filename") or (Path(image_name).stem + ".txt"))
        src_img = images_dir / image_name
        src_lbl = labels_dir / label_name

        if entry_keys(entry, idx) & bad_entries:
            removed = dict(entry)
            removed["cleaner_reason"] = "manual_bad_entry"
            removed_manifest.append(removed)
            removal_counts["manual_bad_entry"] += 1
            continue

        if not src_img.exists():
            removed = dict(entry)
            removed["cleaner_reason"] = "missing_image"
            removed_manifest.append(removed)
            removal_counts["missing_image"] += 1
            continue

        tiny_boxes = _tiny_target_boxes(
            image_path=src_img,
            label_path=src_lbl,
            min_width_px=min_target_width_px,
            min_height_px=min_target_height_px,
            min_area_px=min_target_area_px,
        )
        if tiny_boxes:
            removed = dict(entry)
            removed["cleaner_reason"] = "tiny_target_bbox"
            removed["tiny_target_boxes"] = tiny_boxes
            removed_manifest.append(removed)
            removal_counts["tiny_target_bbox"] += 1
            tiny_target_box_count += len(tiny_boxes)
            continue

        seed = int(entry.get("seed", parse_seed(image_name)))
        split = split_for_seed(seed, seeds)
        split_counts[split] += 1
        new_stem = f"{split}_{split_counts[split]:06d}"
        new_img = output_dir / "images" / split / f"{new_stem}.png"
        new_lbl = output_dir / "labels" / split / f"{new_stem}.txt"
        shutil.copy2(src_img, new_img)
        if src_lbl.exists():
            shutil.copy2(src_lbl, new_lbl)
        else:
            new_lbl.write_text("", encoding="utf-8")

        updated = dict(entry)
        updated.update(
            {
                "clean_split": split,
                "clean_index": split_counts[split],
                "clean_image_filename": f"images/{split}/{new_stem}.png",
                "clean_label_filename": f"labels/{split}/{new_stem}.txt",
                "original_image_filename": image_name,
                "original_label_filename": label_name,
            }
        )
        kept_manifest.append(updated)

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in kept_manifest:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    with (output_dir / "removed_entries.jsonl").open("w", encoding="utf-8") as f:
        for row in removed_manifest:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    copy_optional(dataset_dir / "metadata.json", output_dir / "metadata.json")
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {}
    metadata["cleaning"] = {
        "source_dataset": str(dataset_dir),
        "bad_entries_requested": sorted(bad_entries),
        "removed_count": len(removed_manifest),
        "kept_count": len(kept_manifest),
        "split_counts": split_counts,
        "split_rule": "last seed -> test, second-last seed -> val, earlier seeds -> train",
        "seeds": seeds,
        "tiny_target_bbox_filter": {
            "enabled": True,
            "target_class_ids": sorted(TARGET_CLASS_IDS),
            "target_class_names": [CLASS_ID_TO_NAME[class_id] for class_id in sorted(TARGET_CLASS_IDS)],
            "min_target_width_px": float(min_target_width_px),
            "min_target_height_px": float(min_target_height_px),
            "min_target_area_px": float(min_target_area_px),
            "discard_rule": "Discard the complete image/label pair when any target bbox violates a threshold.",
            "removed_samples": int(removal_counts["tiny_target_bbox"]),
            "removed_target_boxes": int(tiny_target_box_count),
        },
        "removal_counts": removal_counts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    (output_dir / "target_objects.yaml").write_text(yolo_yaml_text(output_dir), encoding="utf-8")

    print(f"[DONE] source: {dataset_dir}")
    print(f"[DONE] clean dataset: {output_dir}")
    print(
        f"[DONE] kept={len(kept_manifest)} removed={len(removed_manifest)} "
        f"split_counts={split_counts}"
    )
    print(
        "[DONE] tiny-target filter: "
        f"samples_removed={removal_counts['tiny_target_bbox']} "
        f"boxes_removed={tiny_target_box_count} "
        f"thresholds=(w<{min_target_width_px:g}px, h<{min_target_height_px:g}px, area<{min_target_area_px:g}px^2)"
    )
    return output_dir


def main() -> None:
    raw_dataset_dir = RAW_DATASET_DIR.expanduser().resolve()
    output_dataset_dir = OUTPUT_DATASET_DIR.expanduser().resolve()

    if not raw_dataset_dir.exists():
        raise FileNotFoundError(
            "RAW_DATASET_DIR does not exist. Edit the USER SETTINGS block at the top of this file:\n"
            f"  {raw_dataset_dir}"
        )

    bad_files = []
    if MANUAL_BAD_ENTRIES_FILE is not None:
        bad_files.append(MANUAL_BAD_ENTRIES_FILE.expanduser().resolve())

    print("\n=== Hard YOLO dataset preparation ===")
    print(f"Raw collection: {raw_dataset_dir}")
    print(f"Output dataset: {output_dataset_dir}")
    print(
        "Tiny-target filter: discard whole frame if a bottle/can bbox has "
        f"width < {MIN_TARGET_WIDTH_PX:g}px, height < {MIN_TARGET_HEIGHT_PX:g}px, "
        f"or area < {MIN_TARGET_AREA_PX:g}px²."
    )
    print()

    clean_dataset(
        raw_dataset_dir,
        load_bad_entries(MANUAL_BAD_ENTRIES, bad_files),
        output_dataset_dir,
        min_target_width_px=float(MIN_TARGET_WIDTH_PX),
        min_target_height_px=float(MIN_TARGET_HEIGHT_PX),
        min_target_area_px=float(MIN_TARGET_AREA_PX),
    )


if __name__ == "__main__":
    main()