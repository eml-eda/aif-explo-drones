from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

Cell = Tuple[int, int]
ObjectKey = Tuple[int, int, str]


def rc_to_cxcy(rc: Optional[Sequence[int]]) -> Optional[Cell]:
    if rc is None:
        return None
    r, c = int(rc[0]), int(rc[1])
    return (c, r)


def mapping_rc_to_cxcy(values: Mapping[Tuple[int, int], Any]) -> Dict[Cell, Any]:
    return {(int(c), int(r)): value for (r, c), value in values.items()}


def detection_event_rc_to_cxcy(event: Mapping[str, Any]) -> dict:
    """Copy one projected detection event and expose policy/report cx/cy cells.

    Grid/environment code uses row/column tuples internally. Policy inputs and
    report payloads use (cx, cy). Both representations remain present so no
    information is discarded during the conversion.
    """
    out = dict(event)

    cell_rc = out.get("cell_rc")
    if cell_rc is not None:
        r, c = int(cell_rc[0]), int(cell_rc[1])
        out["cell_rc"] = (r, c)
        out["cell"] = (c, r)

    covered_rc = out.get("covered_cells_rc")
    if covered_rc is not None:
        converted = []
        for rc in covered_rc:
            try:
                r, c = int(rc[0]), int(rc[1])
                converted.append([c, r])
            except Exception:
                continue
        out["covered_cells"] = converted

    for rc_key, cell_key in (
        ("cell_rc_floor", "cell_floor"),
        ("cell_rc_table", "cell_table"),
    ):
        rc = out.get(rc_key)
        if rc is None:
            continue
        try:
            r, c = int(rc[0]), int(rc[1])
            out[cell_key] = (c, r)
        except Exception:
            pass

    return out


@dataclass(frozen=True)
class TaskStateSnapshot:
    status: str = "running"
    reason: Any = None
    total_objects: int = 0
    remaining_objects: int = 0
    validated_found_cells: frozenset[Cell] = frozenset()
    validated_found_objects: frozenset[ObjectKey] = frozenset()

    @classmethod
    def from_payload(cls, payload: Optional[Mapping[str, Any]]) -> "TaskStateSnapshot":
        payload = payload or {}

        cells = frozenset(
            (int(cx), int(cy))
            for (cx, cy) in (payload.get("validated_found_cells", []) or [])
        )
        objects = frozenset(
            (int(raw["cx"]), int(raw["cy"]), str(raw["class_name"]))
            for raw in (payload.get("validated_found_objects", []) or [])
        )

        return cls(
            status=str(payload.get("status", "running")),
            reason=payload.get("reason"),
            total_objects=int(payload.get("total_objects", 0)),
            remaining_objects=int(payload.get("remaining_objects", 0)),
            validated_found_cells=cells,
            validated_found_objects=objects,
        )


@dataclass(frozen=True)
class PolicyPerceptionEvidence:
    visible_mask: Dict[Cell, Any]
    det_confidence: Dict[Cell, Any]
    det_confidence_multiclass: Dict[Cell, Any]
    det_metadata_multiclass: Dict[Cell, Any]
    detections_all: list[dict]
    target_detections: list[dict]
    table_confidence: Dict[Cell, Any]
    table_visible_mask: Dict[Cell, Any]
    table_coverages: Dict[Cell, Any]

    @classmethod
    def for_step(
        cls,
        *,
        perception_processed: bool,
        visible_mask: Mapping[Cell, Any],
        det_confidence: Mapping[Cell, Any],
        det_confidence_multiclass: Mapping[Cell, Any],
        det_metadata_multiclass: Mapping[Cell, Any],
        detections_all: Iterable[Mapping[str, Any]],
        target_detections: Iterable[Mapping[str, Any]],
        table_confidence: Mapping[Cell, Any],
        table_visible_mask: Mapping[Cell, Any],
        table_coverages: Mapping[Cell, Any],
    ) -> "PolicyPerceptionEvidence":
        if not perception_processed:
            return cls({}, {}, {}, {}, [], [], {}, {}, {})

        return cls(
            visible_mask=dict(visible_mask),
            det_confidence=dict(det_confidence),
            det_confidence_multiclass=dict(det_confidence_multiclass),
            det_metadata_multiclass=dict(det_metadata_multiclass),
            detections_all=[dict(event) for event in detections_all],
            target_detections=[dict(event) for event in target_detections],
            table_confidence=dict(table_confidence),
            table_visible_mask=dict(table_visible_mask),
            table_coverages=dict(table_coverages),
        )


@dataclass
class TargetDistanceTracker:
    count: int = 0
    sum_m: float = 0.0
    min_m: Optional[float] = None
    max_m: Optional[float] = None

    def record_policy_action(self, policy_action: Mapping[str, Any]) -> bool:
        target_selection = policy_action.get("target_selection")
        if not isinstance(target_selection, dict):
            return False

        distance_m = target_selection.get("distance_m")
        if distance_m is None:
            return False

        distance_m = float(distance_m)
        self.count += 1
        self.sum_m += distance_m
        if self.min_m is None or distance_m < self.min_m:
            self.min_m = distance_m
        if self.max_m is None or distance_m > self.max_m:
            self.max_m = distance_m
        return True

    def payload(self, *, enabled: bool, policy_supported: bool) -> dict:
        tracked = bool(enabled and policy_supported)
        average_m = (self.sum_m / float(self.count)) if self.count > 0 else None
        return {
            "enabled": bool(enabled),
            "policy_supported": bool(policy_supported),
            "tracked": tracked,
            "count": int(self.count),
            "sum_m": float(self.sum_m),
            "average_m": None if average_m is None else float(average_m),
            "min_m": None if self.min_m is None else float(self.min_m),
            "max_m": None if self.max_m is None else float(self.max_m),
        }


def build_policy_observation(
    *,
    sensor_state: Mapping[str, Any],
    step: int,
    pose: Mapping[str, float],
    grid_spec: Any,
    detector_detections: Optional[Iterable[Mapping[str, Any]]],
    current_cell: Optional[Cell],
    visual_coverages: Mapping[Cell, Any],
    perception_evidence: PolicyPerceptionEvidence,
    coverage_threshold: float,
    persistent_table_confidence: Mapping[Cell, float],
    perception_processed: bool,
    surface_height_map: Mapping[str, Any],
    persistent_table_cells: Iterable[Cell],
    effective_table_projection_cells: Iterable[Cell],
    claimed_found_cells: Iterable[Cell],
    task_state: TaskStateSnapshot,
    claim_confidence_threshold: float,
    snapshots_enabled: bool,
    snapshots_dir: Optional[str],
    snapshots_every_n: int,
    snapshots_on_stuck: bool,
    stuck_replan_limit: int,
    floor_fov_spec: Any,
    table_fov_spec: Any,
    floor_height_m: float,
    table_height_m: float,
    floor_plane_coverages: Optional[Mapping[Cell, Any]] = None,
    table_plane_coverages: Optional[Mapping[Cell, Any]] = None,
) -> dict:
    """Build the policy input without changing the established schema.

    floor_plane_coverages / table_plane_coverages are optional additions: the
    coverage of each cell by the camera footprint at floor and at table height,
    independent of the table map. They are added as new grid keys only when
    given, so existing keys are unchanged.
    """
    policy_observation = {
        **dict(sensor_state),
        "step": int(step),
        "pose": {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "yaw": float(pose["yaw"]),
        },
        "grid_spec": {
            "min_x": grid_spec.min_x,
            "max_x": grid_spec.max_x,
            "min_y": grid_spec.min_y,
            "max_y": grid_spec.max_y,
            "cell_size": grid_spec.cell_size,
            "rows": grid_spec.rows,
            "cols": grid_spec.cols,
        },
        "grid": {
            "current_cell": current_cell,
            "coverages": dict(visual_coverages),
            "visible_mask": dict(perception_evidence.visible_mask),
            "coverage_threshold": float(coverage_threshold),
            "det_confidence": dict(perception_evidence.det_confidence),
            "det_confidence_multiclass": dict(perception_evidence.det_confidence_multiclass),
            "det_metadata_multiclass": dict(perception_evidence.det_metadata_multiclass),
            "detections_all": list(perception_evidence.detections_all),
            "target_detections": list(perception_evidence.target_detections),
            "table_confidence": dict(perception_evidence.table_confidence),
            "table_visible_mask": dict(perception_evidence.table_visible_mask),
            "table_coverages": dict(perception_evidence.table_coverages),
            "table_confidence_persistent": dict(persistent_table_confidence),
            "perception_processed_this_step": bool(perception_processed),
            "surface_height_map": dict(surface_height_map),
            "table_cells": sorted(list(persistent_table_cells)),
            "effective_table_projection_cells": sorted(list(effective_table_projection_cells)),
        },
        "task": {
            "found_cells": sorted(list(claimed_found_cells)),
            "validated_found_cells": sorted(list(task_state.validated_found_cells)),
            "validated_found_objects": sorted(
                [
                    {"cx": int(cx), "cy": int(cy), "class_name": str(class_name)}
                    for (cx, cy, class_name) in task_state.validated_found_objects
                ],
                key=lambda d: (d["cy"], d["cx"], d["class_name"]),
            ),
            "remaining_objects": int(task_state.remaining_objects),
            "total_objects": int(task_state.total_objects),
            "claim_confidence_threshold": float(claim_confidence_threshold),
            "snapshot": {
                "enabled": bool(snapshots_enabled),
                "dir": snapshots_dir,
                "every_n": int(snapshots_every_n),
                "on_stuck": bool(snapshots_on_stuck),
                "stuck_replan_limit": int(stuck_replan_limit),
            },
        },
        "fov_spec": {
            "half_left": floor_fov_spec.half_left,
            "half_right": floor_fov_spec.half_right,
            "forward": floor_fov_spec.forward,
            "backward": floor_fov_spec.backward,
        },
        "fov_spec_by_surface": {
            "floor": {
                "height_m": float(floor_height_m),
                "half_left": floor_fov_spec.half_left,
                "half_right": floor_fov_spec.half_right,
                "forward": floor_fov_spec.forward,
                "backward": floor_fov_spec.backward,
            },
            "table": {
                "height_m": float(table_height_m),
                "half_left": table_fov_spec.half_left,
                "half_right": table_fov_spec.half_right,
                "forward": table_fov_spec.forward,
                "backward": table_fov_spec.backward,
            },
        },
    }

    if floor_plane_coverages is not None:
        policy_observation["grid"]["floor_plane_coverages"] = dict(floor_plane_coverages)
    if table_plane_coverages is not None:
        policy_observation["grid"]["table_plane_coverages"] = dict(table_plane_coverages)

    policy_observation["detections"] = list(detector_detections or [])

    return policy_observation


def build_progress_payload(
    *,
    policy_name: str,
    run_dir: str,
    autonomy_start_time_s: Optional[float],
    last_controller_time_s: float,
    visited_cells: Iterable[Cell],
    claimed_found_cells: Iterable[Cell],
    claimed_found_objects: Iterable[ObjectKey],
    detector_limitations: Mapping[str, Any],
    experiment_metadata: Mapping[str, Any],
    target_distance_stats: Mapping[str, Any],
    drone_spawn: Mapping[str, Any],
    surface_height_map: Mapping[str, Any],
) -> dict:
    return {
        "policy_name": str(policy_name),
        "run_dir": str(run_dir),
        "autonomy_start_time_s": autonomy_start_time_s,
        "last_controller_time_s": float(last_controller_time_s),
        "visited_cells": sorted([[int(c[0]), int(c[1])] for c in visited_cells]),
        "claimed_found_cells": sorted([[int(c[0]), int(c[1])] for c in claimed_found_cells]),
        "claimed_found_objects": sorted(
            [
                {
                    "cx": int(cx),
                    "cy": int(cy),
                    "class_name": str(class_name),
                    "label": str(class_name),
                }
                for (cx, cy, class_name) in claimed_found_objects
            ],
            key=lambda d: (d["cy"], d["cx"], d["class_name"]),
        ),
        "detector_limitations": dict(detector_limitations),
        "experiment_metadata": dict(experiment_metadata),
        "target_distance_stats": dict(target_distance_stats),
        "drone_spawn": dict(drone_spawn),
        "surface_height_map": dict(surface_height_map),
    }
