# controllers/crazyflie_table_dataset_collector/tools/dataset_cleaner.py

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set

CLASS_ID_TO_NAME = {
    0: "target_bottle",
    1: "target_can",
    2: "desk",
}


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
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def image_entries(dataset_dir: Path, manifest_rows: List[dict]) -> List[dict]:
    if manifest_rows:
        return manifest_rows
    entries = []
    for image_path in sorted((dataset_dir / "images").glob("*.png")):
        stem = image_path.stem
        seed = parse_seed(stem)
        entries.append(
            {
                "entry_id": stem,
                "seed": seed,
                "image_filename": image_path.name,
                "label_filename": stem + ".txt",
            }
        )
    return entries


def parse_seed(text: str) -> int:
    match = re.search(r"seed(\d+)", text)
    if not match:
        return -1
    return int(match.group(1))


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


def clean_dataset(dataset_dir: Path, bad_entries: Set[str], output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = dataset_dir.with_name(dataset_dir.name + "_clean")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing clean dataset: {output_dir}")

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

    for split in split_counts:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for idx, entry in enumerate(entries):
        if entry_keys(entry, idx) & bad_entries:
            removed_manifest.append(entry)
            continue

        image_name = str(entry.get("image_filename"))
        label_name = str(entry.get("label_filename") or (Path(image_name).stem + ".txt"))
        src_img = images_dir / image_name
        src_lbl = labels_dir / label_name
        if not src_img.exists():
            removed = dict(entry)
            removed["cleaner_reason"] = "missing_image"
            removed_manifest.append(removed)
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
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    (output_dir / "target_objects.yaml").write_text(yolo_yaml_text(output_dir), encoding="utf-8")
    print(f"[DONE] source: {dataset_dir}")
    print(f"[DONE] clean dataset: {output_dir}")
    print(f"[DONE] kept={len(kept_manifest)} removed={len(removed_manifest)} split_counts={split_counts}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a non-destructive cleaned/split table dataset.")
    parser.add_argument("dataset_dir", type=Path, help="Path to collection_table_* dataset folder.")
    parser.add_argument("--bad-entry", action="append", default=[], help="Bad entry id, image stem/name, label stem/name, or sequential index.")
    parser.add_argument("--bad-file", action="append", type=Path, default=[], help="Text file with one bad entry id/name/index per line.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output path. Defaults to <dataset>_clean.")
    args = parser.parse_args()

    bad_entries = load_bad_entries(args.bad_entry, args.bad_file)
    clean_dataset(args.dataset_dir.resolve(), bad_entries, args.output_dir.resolve() if args.output_dir else None)


if __name__ == "__main__":
    main()
