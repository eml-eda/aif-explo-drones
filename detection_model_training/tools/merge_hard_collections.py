from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

HARD_CLASS_IDS = {0, 1, 2}
HARD_CLASS_NAMES = {0: "target_bottle", 1: "target_can", 2: "desk"}


def _read_label_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.exists():
        return ids
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label in {path} line {line_number}: {raw!r}")
        ids.add(int(float(parts[0])))
    return ids


def _validate_collection(path: Path) -> List[Path]:
    images_dir = path / "images"
    labels_dir = path / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Expected images/ and labels/ under {path}")
    images = sorted(images_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No PNG images found in {images_dir}")
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        unknown = _read_label_ids(label) - HARD_CLASS_IDS
        if unknown:
            raise ValueError(f"{label} contains non-hard class ids: {sorted(unknown)}")
    return images


def merge_hard_collections(collection_a: Path, collection_b: Path, output_dir: Path) -> Path:
    collection_a = Path(collection_a).expanduser().resolve()
    collection_b = Path(collection_b).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing merged collection: {output_dir}")

    sources = [("hard_collection_a", collection_a), ("hard_collection_b", collection_b)]
    image_lists = {name: _validate_collection(path) for name, path in sources}

    (output_dir / "images").mkdir(parents=True)
    (output_dir / "labels").mkdir(parents=True)
    has_any_seg = any((path / "segmentation_debug").exists() for _, path in sources)
    if has_any_seg:
        (output_dir / "segmentation_debug").mkdir(parents=True)

    manifest_rows: List[dict] = []
    source_counts: Dict[str, int] = {}

    for source_name, source_dir in sources:
        source_counts[source_name] = 0
        for image_path in image_lists[source_name]:
            label_path = source_dir / "labels" / f"{image_path.stem}.txt"
            new_stem = f"{source_name}__{image_path.stem}"
            dst_image = output_dir / "images" / f"{new_stem}.png"
            dst_label = output_dir / "labels" / f"{new_stem}.txt"
            shutil.copy2(image_path, dst_image)
            if label_path.exists():
                shutil.copy2(label_path, dst_label)
            else:
                dst_label.write_text("", encoding="utf-8")

            seg_candidates = [
                source_dir / "segmentation_debug" / f"{image_path.stem}_seg.png",
                source_dir / "segmentation_debug" / f"{image_path.stem}.png",
            ]
            for seg_path in seg_candidates:
                if seg_path.exists() and has_any_seg:
                    shutil.copy2(seg_path, output_dir / "segmentation_debug" / f"{new_stem}_seg.png")
                    break

            manifest_rows.append(
                {
                    "entry_id": new_stem,
                    "seed": _parse_seed(image_path.stem),
                    "image_filename": f"{new_stem}.png",
                    "label_filename": f"{new_stem}.txt",
                    "source_domain": source_name,
                    "source_collection": source_dir.name,
                    "source_image_filename": image_path.name,
                    "source_label_filename": label_path.name,
                }
            )
            source_counts[source_name] += 1

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    metadata = {
        "environment": "hard",
        "collection_role": "combined_hard_training_collection",
        "class_id_to_name": HARD_CLASS_NAMES,
        "source_domains": {
            "hard_collection_a": collection_a.name,
            "hard_collection_b": collection_b.name,
        },
        "source_counts": source_counts,
        "total_samples": len(manifest_rows),
        "note": "Collection A and B differ in visual context only; labelled classes are identical.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "target_objects.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n"
        "  0: target_bottle\n  1: target_can\n  2: desk\n",
        encoding="utf-8",
    )

    print(f"[HARD MERGE] A={collection_a} ({source_counts['hard_collection_a']} samples)")
    print(f"[HARD MERGE] B={collection_b} ({source_counts['hard_collection_b']} samples)")
    print(f"[HARD MERGE] output={output_dir} total={len(manifest_rows)}")
    return output_dir


def _parse_seed(text: str) -> int:
    import re

    match = re.search(r"seed(\d+)", text)
    return int(match.group(1)) if match else -1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the two raw hard collection domains A+B before dataset preparation."
    )
    parser.add_argument("collection_a", type=Path)
    parser.add_argument("collection_b", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    merge_hard_collections(args.collection_a, args.collection_b, args.output_dir)


if __name__ == "__main__":
    main()
