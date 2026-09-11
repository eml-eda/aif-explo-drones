from __future__ import annotations

import os
import sys

HERE = os.path.dirname(__file__)
BASE_SUPERVISOR_DIR = os.path.abspath(
    os.path.join(HERE, "..", "worldgen_hard_dataset_supervisor")
)
if BASE_SUPERVISOR_DIR not in sys.path:
    sys.path.insert(0, BASE_SUPERVISOR_DIR)

# This controller name fixes the collection domain. Other HARD_DATASET_* options
# (run count, samples/run, policy, spawn density, etc.) remain normal overrides.
os.environ["HARD_DATASET_SCENE_VARIANT"] = "hard_collection_b"

from worldgen_hard_dataset_supervisor import main


if __name__ == "__main__":
    main()
