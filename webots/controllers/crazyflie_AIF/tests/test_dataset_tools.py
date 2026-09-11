from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
TOOLS_DIR = REPO_ROOT / "detection_model_training" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from prepare_dataset import normalize_environment, prepare_dataset, split_entries_randomly
from merge_hard_collections import merge_hard_collections


def _write_pair(root: Path, stem: str, label_text: str, size=(320, 240)) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(root / "images" / f"{stem}.png")
    (root / "labels" / f"{stem}.txt").write_text(label_text, encoding="utf-8")


class DatasetToolTests(unittest.TestCase):
    def test_environment_aliases_are_rejected(self):
        for value in ("mixed_easy", "table", "tables", "mixed_tables", "mixed_hard"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_environment(value)

    def test_shuffled_split_is_deterministic(self):
        entries = [{"entry_id": str(i)} for i in range(20)]
        first = split_entries_randomly(entries, split_seed=123, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        second = split_entries_randomly(entries, split_seed=123, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        self.assertEqual(first, second)

    def test_single_seed_collection_auto_uses_shuffled_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "collection"
            output = root / "prepared"
            for i in range(10):
                _write_pair(source, f"seed0050_frame{i:04d}", "")
            prepare_dataset(source, output, environment="easy")
            meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["preparation"]["split_mode_used"], "shuffled")
            self.assertEqual(sum(meta["preparation"]["split_counts"].values()), 10)
            self.assertGreater(meta["preparation"]["split_counts"]["train"], 0)
            self.assertGreater(meta["preparation"]["split_counts"]["val"], 0)
            self.assertGreater(meta["preparation"]["split_counts"]["test"], 0)

    def test_hard_tiny_target_filter_discards_whole_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "collection"
            output = root / "prepared"
            # width=2px, height=4px target on 320x240 -> discard.
            _write_pair(source, "seed0001_frame0001", "0 0.5 0.5 0.00625 0.016667\n")
            # Small desk should not trigger target filter.
            _write_pair(source, "seed0001_frame0002", "2 0.5 0.5 0.00625 0.016667\n")
            # Normal target.
            _write_pair(source, "seed0001_frame0003", "1 0.5 0.5 0.1 0.1\n")
            prepare_dataset(source, output, environment="hard", split_mode="shuffled")
            removed = (output / "removed_entries.jsonl").read_text(encoding="utf-8")
            self.assertIn("tiny_target_bbox", removed)
            manifest = (output / "manifest.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("frame0001", manifest)
            self.assertIn("frame0002", manifest)
            self.assertIn("frame0003", manifest)

    def test_hard_merge_preserves_both_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a"
            b = root / "b"
            out = root / "merged"
            _write_pair(a, "seed0001_frame0001", "0 0.5 0.5 0.1 0.1\n")
            _write_pair(b, "seed0001_frame0001", "2 0.5 0.5 0.2 0.2\n")
            merge_hard_collections(a, b, out)
            self.assertTrue((out / "images" / "hard_collection_a__seed0001_frame0001.png").exists())
            self.assertTrue((out / "images" / "hard_collection_b__seed0001_frame0001.png").exists())
            rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({r["source_domain"] for r in rows}, {"hard_collection_a", "hard_collection_b"})

    def test_hard_merge_rejects_unknown_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a"
            b = root / "b"
            _write_pair(a, "seed0001_frame0001", "4 0.5 0.5 0.1 0.1\n")
            _write_pair(b, "seed0001_frame0001", "0 0.5 0.5 0.1 0.1\n")
            with self.assertRaises(ValueError):
                merge_hard_collections(a, b, root / "merged")


if __name__ == "__main__":
    unittest.main()
