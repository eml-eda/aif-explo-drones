from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROLLERS_DIR = HERE.parent
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from dataset_collection.collector import run_collection


if __name__ == "__main__":
    run_collection("medium", controller_dir=HERE)
