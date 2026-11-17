from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, MutableSet, Optional, Sequence, Set, Tuple

from env.surface_projection import (
    bbox_overlaps_any_desk,
    event_bbox_xyxy,
    sample_bbox_projected_cells_rc,
    sample_front_only_bbox_floor_counterparts_rc,
)

CellRC = Tuple[int, int]
Cell = Tuple[int, int]


def _matches_class(
    class_name: Any,
    class_id: Any,
    *,
    names: frozenset[str],
    ids: frozenset[int],
) -> bool:
    name = "" if class_name is None else str(class_name)
    if name in names:
        return True
    if class_id is None:
        return False
    try:
        return int(class_id) in ids
    except Exception:
        return False


def _rc_to_cxcy(rc: Optional[Sequence[int]]) -> Optional[Cell]:
    if rc is None:
        return None
    return (int(rc[1]), int(rc[0]))


@dataclass(frozen=True)
class PerceptionProjectionConfig:
    target_class_names: frozenset[str]
    target_class_ids: frozenset[int]
    desk_class_names: frozenset[str]
    desk_class_ids: frozenset[int]
    surface_detection_conf_threshold: float
    enable_table_shadow_projection: bool
    table_shadow_forward_tolerance_m: float
    min_x: float
    min_y: float
    cell_size_m: float


@dataclass
class ProjectedPerceptionFrame:
    processed: bool
    raw_output: Any
    target_confidence_rc: Dict[CellRC, float]
    target_metadata_rc: Dict[CellRC, dict]
    all_events_rc: list[dict]
    target_events_rc: list[dict]
    table_confidence_rc: Dict[CellRC, float]
    transient_table_shadow_cells: Set[Cell]


def project_perception_frame(
    *,
    perception_manager: Any,
    config: PerceptionProjectionConfig,
    step_count: int,
    time_now_s: float,
    camera_capture: Mapping[str, Any],
    drone_x: float,
    drone_y: float,
    drone_yaw: float,
    env_mgr_floor: Any,
    env_mgr_table: Any,
    floor_visual_coverages: Mapping[CellRC, float],
    persistent_table_cells: MutableSet[Cell],
    persistent_table_confidence: MutableMapping[Cell, float],
) -> ProjectedPerceptionFrame:
    """Run due perception and project detector events onto the controller grid.

    This is a direct extraction of the pre-Pass-10 controller logic. Persistent
    table state is intentionally mutated in-place, matching the old loop.
    """
    target_confidence_rc: Dict[CellRC, float] = {}
    target_metadata_rc: Dict[CellRC, dict] = {}
    all_events_rc: list[dict] = []
    target_events_rc: list[dict] = []
    table_confidence_rc: Dict[CellRC, float] = {}
    transient_table_shadow_cells: Set[Cell] = set()

    perception_result = perception_manager.process_if_due(
        step_count=step_count,
        time_now_s=time_now_s,
        camera_image=camera_capture["image"],
        width=camera_capture["width"],
        height=camera_capture["height"],
        timestamp=camera_capture["time"],
    )
    processed = bool(perception_result.processed)
    raw_output = perception_result.raw_output

    if not processed:
        return ProjectedPerceptionFrame(
            processed=False,
            raw_output=raw_output,
            target_confidence_rc=target_confidence_rc,
            target_metadata_rc=target_metadata_rc,
            all_events_rc=all_events_rc,
            target_events_rc=target_events_rc,
            table_confidence_rc=table_confidence_rc,
            transient_table_shadow_cells=transient_table_shadow_cells,
        )

    is_target = lambda name, class_id: _matches_class(
        name,
        class_id,
        names=config.target_class_names,
        ids=config.target_class_ids,
    )
    is_desk = lambda name, class_id: _matches_class(
        name,
        class_id,
        names=config.desk_class_names,
        ids=config.desk_class_ids,
    )

    img_w = int(camera_capture["width"])
    img_h = int(camera_capture["height"])
    raw_detection_events = list(perception_result.events)

    # First pass: persist desk/table projection before mapping targets from
    # the same detector frame, exactly as in the pre-extraction controller.
    desk_bboxes = []
    for det in raw_detection_events:
        class_id = int(det.get("class_id", -1))
        class_name = str(det.get("class_name", det.get("label", "unknown")))
        conf = float(det.get("confidence", 0.0))
        if not is_desk(class_name, class_id):
            continue

        bbox = det.get("bbox_xyxy")
        desk_bboxes.append(bbox)

        covered_cells_rc = sample_bbox_projected_cells_rc(
            env_mgr_table,
            bbox_xyxy=bbox,
            img_w=img_w,
            img_h=img_h,
            drone_x=drone_x,
            drone_y=drone_y,
            drone_yaw=drone_yaw,
        )
        primary_cell_rc = env_mgr_table.detection_cell_from_bbox(
            bbox_xyxy=bbox,
            img_w=img_w,
            img_h=img_h,
            drone_x=drone_x,
            drone_y=drone_y,
            drone_yaw=drone_yaw,
        )
        if primary_cell_rc is None and covered_cells_rc:
            primary_cell_rc = sorted(covered_cells_rc)[0]

        if conf >= config.surface_detection_conf_threshold:
            for rc in covered_cells_rc:
                cell_cxcy = _rc_to_cxcy(rc)
                if cell_cxcy is None:
                    continue
                rc_key = (int(rc[0]), int(rc[1]))
                persistent_table_cells.add(cell_cxcy)
                persistent_table_confidence[cell_cxcy] = max(
                    float(conf),
                    float(persistent_table_confidence.get(cell_cxcy, 0.0)),
                )
                table_confidence_rc[rc_key] = max(
                    float(conf),
                    float(table_confidence_rc.get(rc_key, 0.0)),
                )

        if primary_cell_rc is not None:
            all_events_rc.append(
                {
                    **det,
                    "cell_rc": (int(primary_cell_rc[0]), int(primary_cell_rc[1])),
                    "cell_rc_table": (int(primary_cell_rc[0]), int(primary_cell_rc[1])),
                    "covered_cells_rc": sorted(
                        [[int(r), int(c)] for (r, c) in covered_cells_rc]
                    ),
                    "surface_cells_already_persisted": True,
                    "surface_assumption": "table",
                }
            )

    # Second pass: target projection may use floor or table plane.
    for det in raw_detection_events:
        class_id = int(det.get("class_id", -1))
        class_name = str(det.get("class_name", det.get("label", "unknown")))
        if not is_target(class_name, class_id):
            continue

        bbox = det.get("bbox_xyxy")
        conf = float(det.get("confidence", 0.0))
        label = str(det.get("label", class_name))

        floor_cell_rc = env_mgr_floor.detection_cell_from_bbox(
            bbox_xyxy=bbox,
            img_w=img_w,
            img_h=img_h,
            drone_x=drone_x,
            drone_y=drone_y,
            drone_yaw=drone_yaw,
        )
        table_cell_rc = env_mgr_table.detection_cell_from_bbox(
            bbox_xyxy=bbox,
            img_w=img_w,
            img_h=img_h,
            drone_x=drone_x,
            drone_y=drone_y,
            drone_yaw=drone_yaw,
        )

        overlaps_desk = bbox_overlaps_any_desk(bbox, desk_bboxes)
        table_cell_cxcy = _rc_to_cxcy(table_cell_rc) if table_cell_rc is not None else None
        table_cell_is_known = (
            table_cell_cxcy in persistent_table_cells
            if table_cell_cxcy is not None
            else False
        )

        use_table_projection = bool(overlaps_desk or table_cell_is_known)
        cell_rc = table_cell_rc if use_table_projection else floor_cell_rc
        if cell_rc is None:
            cell_rc = floor_cell_rc if table_cell_rc is None else table_cell_rc
        if cell_rc is None:
            continue
        cell_rc = (int(cell_rc[0]), int(cell_rc[1]))

        event = {
            **det,
            "cell_rc": cell_rc,
            "cell_rc_floor": None
            if floor_cell_rc is None
            else (int(floor_cell_rc[0]), int(floor_cell_rc[1])),
            "cell_rc_table": None
            if table_cell_rc is None
            else (int(table_cell_rc[0]), int(table_cell_rc[1])),
            "bbox_overlaps_desk": bool(overlaps_desk),
            "table_cell_is_known": bool(table_cell_is_known),
            "surface_assumption": "table" if use_table_projection else "floor",
            "confidence": conf,
            "class_id": class_id,
            "label": label,
            "class_name": class_name,
        }
        all_events_rc.append(event)
        target_events_rc.append(event)

        previous = target_confidence_rc.get(cell_rc, 0.0)
        if conf > previous:
            target_confidence_rc[cell_rc] = conf
            target_metadata_rc[cell_rc] = {
                "class_id": class_id,
                "label": label,
                "class_name": class_name,
                "surface_assumption": event["surface_assumption"],
            }

    # Desk-event pass: current-frame shadow correction. Desk cells were already
    # persisted in the first pass.
    for event in all_events_rc:
        class_id = int(event.get("class_id", -1))
        class_name = str(event.get("class_name", event.get("label", "unknown")))
        conf = float(event.get("confidence", 0.0))
        if conf < config.surface_detection_conf_threshold:
            continue
        if not is_desk(class_name, class_id):
            continue

        if config.enable_table_shadow_projection:
            bbox = event_bbox_xyxy(event)
            if bbox is not None:
                try:
                    floor_counterpart_cells_rc = sample_front_only_bbox_floor_counterparts_rc(
                        env_mgr_table,
                        env_mgr_floor,
                        bbox_xyxy=bbox,
                        img_w=int(camera_capture["width"]),
                        img_h=int(camera_capture["height"]),
                        drone_x=drone_x,
                        drone_y=drone_y,
                        drone_yaw=drone_yaw,
                        min_x=config.min_x,
                        min_y=config.min_y,
                        cell_size_m=config.cell_size_m,
                        forward_tolerance_m=config.table_shadow_forward_tolerance_m,
                    )
                    for rc in floor_counterpart_cells_rc:
                        rc_key = (int(rc[0]), int(rc[1]))
                        if rc_key not in floor_visual_coverages:
                            continue
                        cell_cxcy = _rc_to_cxcy(rc_key)
                        if cell_cxcy is not None:
                            transient_table_shadow_cells.add(cell_cxcy)
                except Exception:
                    pass

    return ProjectedPerceptionFrame(
        processed=processed,
        raw_output=raw_output,
        target_confidence_rc=target_confidence_rc,
        target_metadata_rc=target_metadata_rc,
        all_events_rc=all_events_rc,
        target_events_rc=target_events_rc,
        table_confidence_rc=table_confidence_rc,
        transient_table_shadow_cells=transient_table_shadow_cells,
    )
