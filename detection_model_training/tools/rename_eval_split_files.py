from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATASET_ROOT = "../detection_model_training/dataset_bottles_only"

# Splits to rename
SPLITS = ["val", "test"]

# File extensions
IMAGE_EXT = ".png"
LABEL_EXT = ".txt"

# Safety: if False, script only prints what it would do
APPLY_RENAME = True


def resolve_dataset_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return (script_dir / DATASET_ROOT).resolve()


def collect_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    image_paths = sorted(images_dir.glob(f"*{IMAGE_EXT}"))
    pairs: List[Tuple[Path, Path]] = []

    if not image_paths:
        print(f"[WARN] No images found in {images_dir}")
        return pairs

    for img_path in image_paths:
        lbl_path = labels_dir / f"{img_path.stem}{LABEL_EXT}"
        if not lbl_path.exists():
            raise FileNotFoundError(
                f"Missing mirrored label for image:\n"
                f"  image: {img_path}\n"
                f"  label: {lbl_path}"
            )
        pairs.append((img_path, lbl_path))

    return pairs


def rename_split(dataset_root: Path, split: str) -> None:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

    pairs = collect_pairs(images_dir, labels_dir)
    if not pairs:
        print(f"[INFO] Nothing to rename for split '{split}'")
        return

    print(f"[INFO] Split '{split}': found {len(pairs)} image/label pairs")

    # Two-stage rename to avoid filename collisions
    temp_records = []
    for idx, (img_path, lbl_path) in enumerate(pairs):
        tmp_img = images_dir / f"__tmp__{split}_{idx:06d}{IMAGE_EXT}"
        tmp_lbl = labels_dir / f"__tmp__{split}_{idx:06d}{LABEL_EXT}"

        print(f"  TEMP  {img_path.name} -> {tmp_img.name}")
        print(f"  TEMP  {lbl_path.name} -> {tmp_lbl.name}")

        if APPLY_RENAME:
            img_path.rename(tmp_img)
            lbl_path.rename(tmp_lbl)

        temp_records.append((tmp_img, tmp_lbl, idx))

    for tmp_img, tmp_lbl, idx in temp_records:
        final_stem = f"{split}_{idx:06d}"
        final_img = images_dir / f"{final_stem}{IMAGE_EXT}"
        final_lbl = labels_dir / f"{final_stem}{LABEL_EXT}"

        print(f"  FINAL {tmp_img.name} -> {final_img.name}")
        print(f"  FINAL {tmp_lbl.name} -> {final_lbl.name}")

        if APPLY_RENAME:
            tmp_img.rename(final_img)
            tmp_lbl.rename(final_lbl)

    print(f"[DONE] Renamed split '{split}'")


def main() -> None:
    dataset_root = resolve_dataset_root()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] APPLY_RENAME={APPLY_RENAME}")

    for split in SPLITS:
        rename_split(dataset_root, split)


if __name__ == "__main__":
    main()