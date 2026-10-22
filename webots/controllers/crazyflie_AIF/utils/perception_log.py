# utils/perception_log.py
"""Per-frame perception log for measuring cell-level detector behaviour.

Writes perception_log.jsonl in the run directory: one header record, then one
record per processed detector frame. Each frame stores the drone pose, every
grid cell intersecting the camera footprint with its coverage, the persistent
table map, and every detection with its bbox and projected cells. The header
stores the grid, the camera footprints and the spawned scene, so cell-level
recall, false alarms and misassignments can be computed offline against
ground truth.

Cells are (cx, cy) = (col, row), like policy inputs and reports.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Mapping, Optional

PERCEPTION_LOG_FILENAME = "perception_log.jsonl"


def _cxcy(rc) -> Optional[list]:
    if rc is None:
        return None
    return [int(rc[1]), int(rc[0])]


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def detection_record(event: Mapping) -> dict:
    """Compact, JSON-safe record of one projected detection event (rc keys)."""
    bbox = event.get("bbox_xyxy")
    record = {
        "class_name": str(event.get("class_name", event.get("label", "unknown"))),
        "class_id": int(event.get("class_id", -1)),
        "confidence": _round(event.get("confidence", 0.0)),
        "bbox_xyxy": None if bbox is None else [_round(v, 2) for v in bbox],
        "cell": _cxcy(event.get("cell_rc")),
        "cell_floor": _cxcy(event.get("cell_rc_floor")),
        "cell_table": _cxcy(event.get("cell_rc_table")),
        "surface_assumption": event.get("surface_assumption"),
    }
    if "bbox_overlaps_desk" in event:
        record["bbox_overlaps_desk"] = bool(event["bbox_overlaps_desk"])
    if "covered_cells_rc" in event:
        record["covered_cells"] = [_cxcy(rc) for rc in event["covered_cells_rc"]]
    return record


class PerceptionFrameLogger:
    def __init__(self, *, enabled: bool, run_dir: Optional[str]) -> None:
        self.enabled = bool(enabled and run_dir)
        self.path = os.path.join(run_dir, PERCEPTION_LOG_FILENAME) if self.enabled else None
        self._file = open(self.path, "w", encoding="utf-8") if self.enabled else None

    def _write(self, record: dict) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()

    def write_header(self, header: dict) -> None:
        self._write({"type": "header", **header})

    def log_frame(
        self,
        *,
        time_s: float,
        step: int,
        autonomous: bool,
        x: float,
        y: float,
        z: float,
        yaw: float,
        floor_coverages_rc: Mapping,
        table_coverages_rc: Mapping,
        merged_coverages_rc: Mapping,
        table_projection_cells_cxcy: Iterable,
        persistent_table_cells_cxcy: Iterable,
        detection_events_rc: Iterable[Mapping],
        policy_state: Optional[Mapping] = None,
    ) -> None:
        if self._file is None:
            return
        cells = []
        for rc in sorted(set(floor_coverages_rc) | set(table_coverages_rc) | set(merged_coverages_rc)):
            cells.append([
                int(rc[1]),
                int(rc[0]),
                _round(merged_coverages_rc.get(rc, 0.0)),
                _round(floor_coverages_rc.get(rc, 0.0)),
                _round(table_coverages_rc.get(rc, 0.0)),
            ])
        record = {
            "type": "frame",
            "t": _round(time_s, 3),
            "step": int(step),
            "autonomous": bool(autonomous),
            "pose": {"x": _round(x), "y": _round(y), "z": _round(z), "yaw": _round(yaw)},
            # [cx, cy, coverage used by the policy, floor-plane coverage, table-plane coverage]
            "cells": cells,
            "table_projection_cells": sorted([int(c[0]), int(c[1])] for c in table_projection_cells_cxcy),
            "table_cells": sorted([int(c[0]), int(c[1])] for c in persistent_table_cells_cxcy),
            "detections": [detection_record(e) for e in detection_events_rc],
        }
        if policy_state:
            # Policy-side values (e.g. the learned rho), as of the previous frame.
            record["policy"] = dict(policy_state)
        self._write(record)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
