# detection_model_training/combine_clean_mixed_runs.py

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
# Paste ONLY the cleaned training runs here.
# Do NOT include val/test runs.
TRAIN_RUN_REL_PATHS = [
    "../webots/controllers/crazyflie_dataset_collector/datasets/mixed_bottle_can_dataset_clean/run_seed0017_pseudo_random_20260504_141545",
    "../webots/controllers/crazyflie_dataset_collector/datasets/mixed_bottle_can_dataset_clean/run_seed0023_pseudo_random_20260504_141929",
    "../webots/controllers/crazyflie_dataset_collector/datasets/mixed_bottle_can_dataset_clean/run_seed0034_pseudo_random_20260504_142107",
    "../webots/controllers/crazyflie_dataset_collector/datasets/mixed_bottle_can_dataset_clean/run_seed0045_pseudo_random_20260504_142223",
    "../webots/controllers/crazyflie_dataset_collector/datasets/mixed_bottle_can_dataset_clean/run_seed0050_pseudo_random_20260504_145955",
]

# Destination TRAIN folders inside the final YOLO dataset structure.
TRAIN_IMAGES_REL_PATH = "dataset_bottle_can_mixed/images/train"
TRAIN_LABELS_REL_PATH = "dataset_bottle_can_mixed/labels/train"

# New filename prefix in the final training set.
TRAIN_PREFIX = "train"

# If True, existing files in the train destination folders are deleted first.
CLEAR_DESTINATION_FIRST = False


def resolve_script_relative(path_str: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    return (script_dir / path_str).resolve()


def validate_run_dir(run_dir: Path) -> Tuple[Path, Path]:
    images_dir = run_dir / "images"
    labels_dir = run_dir / "labels"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    return images_dir, labels_dir


def clear_directory_contents(folder: Path) -> None:
    if not folder.exists():
        return
    for item in folder.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def collect_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    image_paths = sorted(images_dir.glob("*.png"))
    pairs: List[Tuple[Path, Path]] = []

    if not image_paths:
        print(f"[WARN] No PNG files found in {images_dir}")

    for img_path in image_paths:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            raise FileNotFoundError(
                f"Missing mirrored label for image:\n"
                f"  image: {img_path}\n"
                f"  label: {lbl_path}"
            )
        pairs.append((img_path, lbl_path))

    return pairs


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    train_images_dir = resolve_script_relative(TRAIN_IMAGES_REL_PATH)
    train_labels_dir = resolve_script_relative(TRAIN_LABELS_REL_PATH)

    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    if CLEAR_DESTINATION_FIRST:
        print(f"[INFO] Clearing destination: {train_images_dir}")
        print(f"[INFO] Clearing destination: {train_labels_dir}")
        clear_directory_contents(train_images_dir)
        clear_directory_contents(train_labels_dir)

    all_pairs: List[Tuple[Path, Path, str]] = []

    print("[INFO] Resolving training runs...")
    for rel_path in TRAIN_RUN_REL_PATHS:
        run_dir = resolve_script_relative(rel_path)
        images_dir, labels_dir = validate_run_dir(run_dir)
        pairs = collect_pairs(images_dir, labels_dir)

        print(f"  - {run_dir.name}: {len(pairs)} pairs")
        for img_path, lbl_path in pairs:
            all_pairs.append((img_path, lbl_path, run_dir.name))

    if not all_pairs:
        raise RuntimeError("No training pairs found. Check TRAIN_RUN_REL_PATHS.")

    print(f"[INFO] Total training pairs to copy: {len(all_pairs)}")

    copied = 0
    manifest_rows = []

    for idx, (src_img, src_lbl, run_name) in enumerate(all_pairs):
        new_stem = f"{TRAIN_PREFIX}_{idx:06d}"
        dst_img = train_images_dir / f"{new_stem}.png"
        dst_lbl = train_labels_dir / f"{new_stem}.txt"

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

        manifest_rows.append(
            {
                "new_stem": new_stem,
                "source_run": run_name,
                "source_image": src_img.name,
                "source_label": src_lbl.name,
            }
        )
        copied += 1

    manifest_path = script_dir / "train_merge_manifest_mixed.csv"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("new_stem,source_run,source_image,source_label\n")
        for row in manifest_rows:
            f.write(
                f"{row['new_stem']},{row['source_run']},{row['source_image']},{row['source_label']}\n"
            )

    print(f"[DONE] Copied {copied} image/label pairs into the TRAIN split.")
    print(f"[DONE] Train images dir: {train_images_dir}")
    print(f"[DONE] Train labels dir: {train_labels_dir}")
    print(f"[DONE] Merge manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()