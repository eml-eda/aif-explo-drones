# controllers/crazyflie_AIF/crazyflie_AIF.py

import os
import time
from datetime import datetime
import json
import math

from controller import Keyboard
from controller import Robot

from devices import CrazyflieDevices
from pid.pid_controller import pid_velocity_fixed_height_controller
from policies.wall_following import WallFollowingPolicy
from policies.pseudo_random import PseudoRandomPolicy
from policies.spiral import SpiralPolicy
from policies.rotate_and_measure import RotateAndMeasurePolicy
from policies.aif_saccade import AIFSaccadePolicy
from policies.aif_saccade_table import AIFSaccadeTablePolicy
from utils.capture_util import save_capture
from utils.logging_AIF import ExperimentLogger
from vision.ground_truth_detector import GroundTruthDetector
from utils.run_visualizations import generate_run_visualizations

from env.env_manager import GridSpec, FoVSpec, GridEnvManager

from config import AIF_POLICY as DEFAULT_POLICY
import config as aif_config
# ---- Detector backend ----
DETECTOR_BACKEND = "yolo"  # options: "yolo", "mobilenet_ssd"

GT_DETECTOR_BACKEND = "camera_recognition"  # options: "analytic_fov", "camera_recognition"

# ---- Run toggles ----
ENABLE_CSV_LOGGING = False

ENABLE_OCCUPANCY_HEATMAP = True
ENABLE_TRAJECTORY_PLOT = True
TRACK_AIF_TARGET_DISTANCE = True

ENABLE_LOGGING = ENABLE_CSV_LOGGING or ENABLE_OCCUPANCY_HEATMAP or ENABLE_TRAJECTORY_PLOT

ENABLE_CAPTURES = False
LOG_EVERY_N_STEPS = 20
CAPTURE_EVERY_N_STEPS = 20

TRAJECTORY_PLOTS_SUBDIR = "trajectory plots"

# ---- Mode toggles ----
EVAL_MODE = True
EVAL_STATE_FILENAME = "eval_state.json"

# ---- Experiment labels for report aggregation - EVAL mode ----
SPAWN_CONDITIONS = "spawn15"
TARGET_SETUP = "mixed_tables"  # options: "bottles_only", "mixed_easy", "mixed_tables", "mixed_hard"

# ---- Flight preference ----
FLYING_ATTITUDE = 1.5

# ---- Spatial parameters ----
ARENA_WIDTH_M = 5.5
ARENA_HEIGHT_M = 6.5
CELL_SIZE_M = 0.5
GRID_COLS = int(ARENA_WIDTH_M / CELL_SIZE_M)
GRID_ROWS = int(ARENA_HEIGHT_M / CELL_SIZE_M)
MIN_X = -ARENA_WIDTH_M / 2.0
MIN_Y = -ARENA_HEIGHT_M / 2.0
MAX_X = ARENA_WIDTH_M / 2.0
MAX_Y = ARENA_HEIGHT_M / 2.0

# ---- Camera FoV footprint on floor (meters) at 1.5m altitude ----
# Measured from the Webots camera FoV setup. Forward/backward asymmetry comes
# from the downward camera offset relative to the drone center.
FOV_WIDTH_M = 1.394
FOV_FORWARD_M = 0.553
FOV_BACKWARD_M = 0.493

# ---- Height-aware FoV setup ----
# The base FoV above is the floor-plane projection at FLYING_ATTITUDE.
# For raised surfaces, footprint dimensions scale with camera-to-surface distance.
ENABLE_HEIGHT_AWARE_FOV = True
FLOOR_HEIGHT_M = 0.0
TABLE_HEIGHT_M = 0.72
DESK_CLASS_NAME = "desk"
DESK_CLASS_ID = 2
SURFACE_DETECTION_CONF_THRESHOLD = 0.30

# ---- Table-height shadow projection ----
# Current-frame geometry correction: if a detected/persistent table-height cell
# lies in the hypothetical floor FoV, cells behind it along the same camera ray
# should use table-height coverage rather than floor-height coverage. This avoids
# giving no-target/no-table evidence to floor cells that are actually occluded or
# truncated by the raised table surface. These shadow cells are NOT persisted as
# confirmed table cells and do NOT create positive table evidence.
ENABLE_TABLE_SHADOW_PROJECTION = True
TABLE_SHADOW_LATERAL_TOLERANCE_M = CELL_SIZE_M * 0.75
TABLE_SHADOW_MAX_DEPTH_M = FOV_FORWARD_M + CELL_SIZE_M

# ---- FoV → cell thresholds ----
COVERAGE_UPDATE_THRESHOLD = 0.05

# ---- Single debug config ----
DEBUG = {
    "every_n": 100,
    "fov": False,
    "cmd": False,
    "policy_grid": False,
    "policy_detections": False,
}

# ---- Snapshot / debugging ----
ENABLE_SNAPSHOTS = False
SNAPSHOT_EVERY_N_STEPS = 200
SNAPSHOT_ON_STUCK = True
STUCK_REPLAN_LIMIT = 100

# ---- Detection mode ----
USE_GROUND_TRUTH_DETECTIONS = True
GT_CONFIDENCE = 1.0
GT_JSON_FILENAME = "spawned_objects.json"

# ---- Perception sampling / throttling ----
# "none"   -> process detector every control step
# "frames" -> process once every DETECTION_EVERY_N_FRAMES control steps
# "time"   -> process once every DETECTION_EVERY_S simulation seconds
DETECTION_THROTTLE_MODE = "time"   # "none", "frames", "time"

DETECTION_EVERY_N_FRAMES = 20
DETECTION_EVERY_S = 0.625

# ---- Shared init / task / progress files ----
EXPERIMENT_INIT_FILENAME = "experiment_init.json"
TASK_STATE_FILENAME = "task_state.json"
CONTROLLER_PROGRESS_FILENAME = "controller_progress.json"

# ---- Task manager: "found" definition ----
FOUND_DET_THRESHOLD = 0.45
FOUND_STREAK_STEPS = 1

# ---- Takeoff warm-up ----
AUTO_START_AUTONOMY = True
AUTONOMY_START_DELAY_S = 15.0

# ---- Startup file wait ----
STARTUP_WAIT_TIMEOUT_S = 10.0
STARTUP_WAIT_POLL_S = 0.05

# Classes that are task targets and can update the old AIF target-belief grid / create found claims.
# Desk observations are preserved separately in per-frame detection lists, but they are not
# treated as target evidence until a table-aware policy consumes them explicitly.
TARGET_OBJECT_CLASS_NAMES = {"target_bottle", "target_can"}
TARGET_OBJECT_CLASS_IDS = {0, 1}
DESK_OBJECT_CLASS_NAMES = {DESK_CLASS_NAME}
DESK_OBJECT_CLASS_IDS = {DESK_CLASS_ID}

AIF_TARGET_DISTANCE_POLICIES = {"aif_saccade", "aif_saccade_table"}


def _is_target_object_class(class_name, class_id=None) -> bool:
    name = "" if class_name is None else str(class_name)
    if name in TARGET_OBJECT_CLASS_NAMES:
        return True
    if class_id is None:
        return False
    try:
        return int(class_id) in TARGET_OBJECT_CLASS_IDS
    except Exception:
        return False


def _is_desk_object_class(class_name, class_id=None) -> bool:
    name = "" if class_name is None else str(class_name)
    if name in DESK_OBJECT_CLASS_NAMES:
        return True
    if class_id is None:
        return False
    try:
        return int(class_id) in DESK_OBJECT_CLASS_IDS
    except Exception:
        return False


def _make_fov_spec(*, width_m: float, forward_m: float, backward_m: float) -> FoVSpec:
    return FoVSpec(
        half_left=float(width_m) / 2.0,
        half_right=float(width_m) / 2.0,
        forward=float(forward_m),
        backward=float(backward_m),
    )


def _surface_fov_scale(*, camera_height_m: float, surface_height_m: float, reference_height_m: float) -> float:
    reference_height_m = max(1e-6, float(reference_height_m))
    distance_m = max(0.05, float(camera_height_m) - float(surface_height_m))
    return float(distance_m / reference_height_m)


def _scaled_fov_spec_for_surface(surface_height_m: float, *, camera_height_m: float) -> FoVSpec:
    # The floor FoV constants were measured at FLYING_ATTITUDE above floor.
    # Scale them by camera-to-surface distance for raised surfaces.
    scale = _surface_fov_scale(
        camera_height_m=float(camera_height_m),
        surface_height_m=float(surface_height_m),
        reference_height_m=float(FLYING_ATTITUDE),
    )
    return _make_fov_spec(
        width_m=FOV_WIDTH_M * scale,
        forward_m=FOV_FORWARD_M * scale,
        backward_m=FOV_BACKWARD_M * scale,
    )


def _merge_coverages_by_surface(
    *,
    floor_coverages_rc: dict,
    table_coverages_rc: dict,
    table_cells_cxcy: set,
) -> dict:
    """Select floor or table coverage per cell based on discovered surface cells.

    Keys are row/col internally. table_cells_cxcy is maintained in policy/report
    coordinates (cx, cy). If a cell is flagged as table-height, use the smaller
    table-plane projection coverage for that cell; otherwise use floor coverage.
    """
    out = {}
    all_cells = set(floor_coverages_rc.keys()) | set(table_coverages_rc.keys())
    for cell_rc in all_cells:
        r, c = int(cell_rc[0]), int(cell_rc[1])
        if (c, r) in table_cells_cxcy:
            val = float(table_coverages_rc.get(cell_rc, 0.0))
        else:
            val = float(floor_coverages_rc.get(cell_rc, 0.0))
        if val > 0.0:
            out[(r, c)] = val
    return out


def _cell_center_xy_from_cxcy(grid_spec: GridSpec, cell_cxcy) -> tuple:
    """Return metric center of a policy/report cell (cx, cy)."""
    cx, cy = int(cell_cxcy[0]), int(cell_cxcy[1])
    x0 = float(grid_spec.min_x) + cx * float(grid_spec.cell_size)
    y0 = float(grid_spec.min_y) + cy * float(grid_spec.cell_size)
    cs = float(grid_spec.cell_size)
    return (x0 + 0.5 * cs, y0 + 0.5 * cs)


def _cell_camera_forward_lateral(
    *,
    grid_spec: GridSpec,
    cell_cxcy,
    drone_x: float,
    drone_y: float,
    drone_yaw: float,
) -> tuple:
    """Cell-center coordinates in the drone/camera frame.

    forward is positive in the drone yaw direction. lateral is the signed
    sideways offset. Only relative forward and absolute lateral differences are
    used by the table-shadow projection logic.
    """
    x, y = _cell_center_xy_from_cxcy(grid_spec, cell_cxcy)
    dx = float(x) - float(drone_x)
    dy = float(y) - float(drone_y)
    cyaw = math.cos(float(drone_yaw))
    syaw = math.sin(float(drone_yaw))
    forward = dx * cyaw + dy * syaw
    lateral = -dx * syaw + dy * cyaw
    return (float(forward), float(lateral))


def _effective_table_projection_cells(
    *,
    table_cells_cxcy: set,
    floor_coverages_rc: dict,
    drone_x: float,
    drone_y: float,
    drone_yaw: float,
    grid_spec: GridSpec,
    lateral_tolerance_m: float = TABLE_SHADOW_LATERAL_TOLERANCE_M,
    max_shadow_depth_m: float = TABLE_SHADOW_MAX_DEPTH_M,
) -> set:
    """Return current-frame cells that should use table-height FoV projection.

    This is a geometry/visibility correction, not table-belief propagation. The
    returned set includes persistent perception-derived table cells plus transient
    current-view shadow cells lying behind such table cells along the drone/camera
    ray. Shadow cells are recomputed every control step and are not added to
    table_surface_cells or table_confidence.
    """
    effective = {
        (int(cx), int(cy))
        for (cx, cy) in (table_cells_cxcy or set())
    }

    if not ENABLE_TABLE_SHADOW_PROJECTION or not table_cells_cxcy or not floor_coverages_rc:
        return effective

    floor_cells_cxcy = {
        (int(c), int(r))
        for (r, c) in floor_coverages_rc.keys()
    }

    lateral_tol = max(0.0, float(lateral_tolerance_m))
    max_depth = max(0.0, float(max_shadow_depth_m))

    table_cells_in_current_floor_fov = [
        cell for cell in effective
        if cell in floor_cells_cxcy
    ]
    if not table_cells_in_current_floor_fov:
        return effective

    table_coords = [
        (
            cell,
            *_cell_camera_forward_lateral(
                grid_spec=grid_spec,
                cell_cxcy=cell,
                drone_x=drone_x,
                drone_y=drone_y,
                drone_yaw=drone_yaw,
            ),
        )
        for cell in table_cells_in_current_floor_fov
    ]

    for candidate in floor_cells_cxcy:
        cand_forward, cand_lateral = _cell_camera_forward_lateral(
            grid_spec=grid_spec,
            cell_cxcy=candidate,
            drone_x=drone_x,
            drone_y=drone_y,
            drone_yaw=drone_yaw,
        )

        for _table_cell, table_forward, table_lateral in table_coords:
            depth = cand_forward - table_forward
            if depth <= 1e-9:
                continue
            if depth > max_depth:
                continue
            if abs(cand_lateral - table_lateral) > lateral_tol:
                continue
            effective.add(candidate)
            break

    return effective


def _surface_height_payload(table_cells_cxcy: set, table_confidence_cxcy=None) -> dict:
    table_confidence_cxcy = table_confidence_cxcy or {}
    return {
        "enabled": bool(ENABLE_HEIGHT_AWARE_FOV),
        "floor_height_m": float(FLOOR_HEIGHT_M),
        "table_height_m": float(TABLE_HEIGHT_M),
        "table_cells": sorted([[int(cx), int(cy)] for (cx, cy) in table_cells_cxcy]),
        "height_by_cell": {
            f"{int(cx)},{int(cy)}": float(TABLE_HEIGHT_M)
            for (cx, cy) in sorted(table_cells_cxcy)
        },
        # Persistent max confidence with which each cell has been mapped as table-height.
        # This lets table-belief policies seed q_table if the controller discovered
        # a desk before the first autonomous policy update.
        "table_confidence_by_cell": {
            f"{int(cx)},{int(cy)}": float(conf)
            for (cx, cy), conf in sorted(table_confidence_cxcy.items())
        },
    }


def _detection_event_rc_to_cxcy(event: dict) -> dict:
    """Return a copy of a detection event with explicit (cx, cy) cells.

    Internally the controller/env code uses row/col cells, while policies and
    reports mostly use cx/cy. Keeping both fields avoids lossy conversion and
    makes future table-aware policies able to consume all detections.
    """
    out = dict(event)
    cell_rc = out.get("cell_rc", None)
    if cell_rc is not None:
        r, c = int(cell_rc[0]), int(cell_rc[1])
        out["cell_rc"] = (r, c)
        out["cell"] = (c, r)

    covered_rc = out.get("covered_cells_rc", None)
    if covered_rc is not None:
        converted = []
        for rc in covered_rc:
            try:
                r, c = int(rc[0]), int(rc[1])
                converted.append([int(c), int(r)])
            except Exception:
                continue
        out["covered_cells"] = converted

    floor_rc = out.get("cell_rc_floor", None)
    if floor_rc is not None:
        try:
            r, c = int(floor_rc[0]), int(floor_rc[1])
            out["cell_floor"] = (c, r)
        except Exception:
            pass

    table_rc = out.get("cell_rc_table", None)
    if table_rc is not None:
        try:
            r, c = int(table_rc[0]), int(table_rc[1])
            out["cell_table"] = (c, r)
        except Exception:
            pass

    return out


def _clamp_bbox_xyxy(bbox_xyxy, *, img_w: int, img_h: int):
    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = max(0.0, min(float(img_w), x1))
    x2 = max(0.0, min(float(img_w), x2))
    y1 = max(0.0, min(float(img_h), y1))
    y2 = max(0.0, min(float(img_h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_area(bbox_xyxy) -> float:
    if bbox_xyxy is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_overlap_area(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _bbox_center_inside(inner_bbox, outer_bbox) -> bool:
    if inner_bbox is None or outer_bbox is None:
        return False
    x1, y1, x2, y2 = [float(v) for v in inner_bbox]
    ox1, oy1, ox2, oy2 = [float(v) for v in outer_bbox]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return (ox1 <= cx <= ox2) and (oy1 <= cy <= oy2)


def _bbox_overlaps_any_desk(target_bbox, desk_bboxes, *, min_overlap_frac: float = 0.05) -> bool:
    target_area = max(1e-6, _bbox_area(target_bbox))
    for desk_bbox in desk_bboxes:
        if _bbox_center_inside(target_bbox, desk_bbox):
            return True
        if (_bbox_overlap_area(target_bbox, desk_bbox) / target_area) >= float(min_overlap_frac):
            return True
    return False


def _event_bbox_xyxy(event: dict):
    bbox = event.get("bbox_xyxy", None)
    if bbox is not None and len(bbox) == 4:
        return [float(v) for v in bbox]

    center = event.get("bbox_center_px", None)
    size = event.get("bbox_size_px", None)
    if center is not None and size is not None and len(center) >= 2 and len(size) >= 2:
        cx, cy = float(center[0]), float(center[1])
        w, h = float(size[0]), float(size[1])
        return [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]

    return None


def _sample_bbox_projected_cells_rc(
    env_mgr_surface,
    *,
    bbox_xyxy,
    img_w: int,
    img_h: int,
    drone_x: float,
    drone_y: float,
    drone_yaw: float,
    max_samples_per_axis: int = 5,
) -> set:
    """Project a detection bbox to all grid cells hit by sampled bbox slices.

    This is used for desk detections. It is still perception-derived: the
    controller marks cells based on the detected image-space desk bbox, not from
    supervisor geometry or arbitrary neighbor propagation.
    """
    bbox = _clamp_bbox_xyxy(bbox_xyxy, img_w=img_w, img_h=img_h)
    if bbox is None:
        return set()

    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    # Use a deterministic probe grid, not random points.
    # Cap at 5x5 = 25 probe points so large desk detections mark several
    # table-height cells without becoming too aggressive/noisy.
    # The 32 px step gives roughly:
    #   small bbox  -> 2x2 / 3x2 probes
    #   medium bbox -> 3x3 / 4x3 probes
    #   large bbox  -> at most 5x5 probes
    nx = int(max(2, min(max_samples_per_axis, round(w / 32.0) + 1)))
    ny = int(max(2, min(max_samples_per_axis, round(h / 32.0) + 1)))

    cells = set()
    for ix in range(nx):
        px = x1 + ((ix + 0.5) / float(nx)) * w
        for iy in range(ny):
            py = y1 + ((iy + 0.5) / float(ny)) * h
            tiny_bbox = [px - 0.5, py - 0.5, px + 0.5, py + 0.5]
            cell_rc = env_mgr_surface.detection_cell_from_bbox(
                bbox_xyxy=tiny_bbox,
                img_w=img_w,
                img_h=img_h,
                drone_x=drone_x,
                drone_y=drone_y,
                drone_yaw=drone_yaw,
            )
            if cell_rc is not None:
                cells.add((int(cell_rc[0]), int(cell_rc[1])))

    # Always include the projection of the bbox center/whole box too.
    center_cell = env_mgr_surface.detection_cell_from_bbox(
        bbox_xyxy=bbox,
        img_w=img_w,
        img_h=img_h,
        drone_x=drone_x,
        drone_y=drone_y,
        drone_yaw=drone_yaw,
    )
    if center_cell is not None:
        cells.add((int(center_cell[0]), int(center_cell[1])))

    return cells


def rc_to_cxcy(rc):
    if rc is None:
        return None
    r, c = rc
    return (c, r)


def dict_rc_to_cxcy(d):
    return {(c, r): v for (r, c), v in d.items()}


def _shared_dir() -> str:
    here = os.path.dirname(__file__)
    shared_dir = os.path.abspath(os.path.join(here, "..", "shared"))
    os.makedirs(shared_dir, exist_ok=True)
    return shared_dir


def _shared_path(filename: str) -> str:
    return os.path.join(_shared_dir(), filename)


def _read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _atomic_write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _wait_for_json(path: str, validator, timeout_s: float, poll_s: float):
    t0 = time.time()
    last_data = None
    while (time.time() - t0) < timeout_s:
        data = _read_json(path, None)
        if isinstance(data, dict):
            last_data = data
            try:
                if validator(data):
                    return data
            except Exception:
                pass
        time.sleep(poll_s)
    return last_data


def load_gt_objects(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("objects", []))


def load_gt_seed(json_path: str) -> int:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("seed", -1))
    except Exception:
        return -1


def make_run_dir(base_log_dir: str, prefix: str = "log_run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_log_dir, f"{prefix}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def load_policy(name, debug=None):
    registry = {
        "wall_following": WallFollowingPolicy,
        "aif_saccade": AIFSaccadePolicy,
        "aif_saccade_table": AIFSaccadeTablePolicy,
        "pseudo_random": PseudoRandomPolicy,
        "spiral": SpiralPolicy,
        "rotate_and_measure": RotateAndMeasurePolicy,
    }
    if name not in registry:
        raise ValueError(f"Unknown policy '{name}'. Available: {', '.join(sorted(registry))}")
    return registry[name](debug=debug)

def load_detector_class(backend: str):
    backend = str(backend).lower().strip()
    target_setup = str(TARGET_SETUP).lower().strip()

    if backend == "yolo":
        if target_setup in {"mixed_tables", "mixed_hard"}:
            from vision.detector_yolo_tables import Detector
            return Detector

        from vision.detector_yolo import Detector
        return Detector

    if backend == "mobilenet_ssd":
        from vision.detector_mobilenet_ssd import Detector
        return Detector

    raise ValueError(
        f"Unknown DETECTOR_BACKEND='{backend}'. "
        "Use 'yolo' or 'mobilenet_ssd'."
    )


def _resolve_run_dir(base_log_dir: str, policy_name: str, eval_state_path: str) -> str:
    """
    Single-run mode:
      logs/log_run_<policy>_<ts>

    Evaluation mode:
      <eval_dir>/runs/log_run_<policy>_runXX_seedYY_<ts>

    Falls back to single-run layout if eval_state is missing or malformed.
    """
    if not EVAL_MODE:
        return make_run_dir(base_log_dir, prefix=f"log_run_{policy_name}")

    eval_state = _read_json(eval_state_path, {})
    eval_dir = eval_state.get("eval_dir", None)
    run_index = int(eval_state.get("current_run_index", 0))
    current_seed = eval_state.get("current_seed", "unknown")

    if not eval_dir:
        return make_run_dir(base_log_dir, prefix=f"log_run_{policy_name}")

    runs_dir = os.path.join(eval_dir, "runs")
    prefix = f"log_run_{policy_name}_run{run_index:02d}_seed{current_seed}"
    return make_run_dir(runs_dir, prefix=prefix)


def _should_process_detection(
    *,
    mode: str,
    step_count: int,
    time_now_s: float,
    last_det_step: int,
    last_det_time_s: float,
) -> bool:
    mode = str(mode).lower()

    if mode == "none":
        return True

    if mode == "frames":
        if DETECTION_EVERY_N_FRAMES <= 0:
            raise ValueError("DETECTION_EVERY_N_FRAMES must be > 0 when using mode='frames'")
        return (step_count - last_det_step) >= int(DETECTION_EVERY_N_FRAMES)

    if mode == "time":
        if DETECTION_EVERY_S <= 0.0:
            raise ValueError("DETECTION_EVERY_S must be > 0 when using mode='time'")
        return (time_now_s - last_det_time_s) >= float(DETECTION_EVERY_S)

    raise ValueError(
        f"Unknown DETECTION_THROTTLE_MODE='{mode}'. Use 'none', 'frames', or 'time'."
    )

def _recognition_sampling_period_ms(timestep_ms: int) -> int:
    """Match Webots recognition/segmentation sampling to detector cadence.

    The controller may run every basic timestep, but GT recognition segmentation
    only needs to refresh when the detector is actually consumed. Use a sampling
    period aligned to an integer number of basic timesteps to avoid asking for a
    recognition frame before Webots has produced a fresh one.
    """
    base = max(1, int(timestep_ms))
    mode = str(DETECTION_THROTTLE_MODE).lower().strip()

    if mode == "none":
        return base

    if mode == "frames":
        return max(base, int(DETECTION_EVERY_N_FRAMES) * base)

    if mode == "time":
        requested_ms = max(base, int(round(float(DETECTION_EVERY_S) * 1000.0)))
        n_steps = max(1, int(math.ceil(float(requested_ms) / float(base))))
        return int(n_steps * base)

    return base

def _detector_limitations_payload() -> dict:
    mode = str(DETECTION_THROTTLE_MODE).lower()

    payload = {
        "mode": mode,
        "frames_interval": None,
        "time_interval_s": None,
    }

    if mode == "frames":
        payload["frames_interval"] = int(DETECTION_EVERY_N_FRAMES)
    elif mode == "time":
        payload["time_interval_s"] = float(DETECTION_EVERY_S)

    return payload

def _detector_model_metadata_payload() -> dict:
    if USE_GROUND_TRUTH_DETECTIONS:
        return {
            "detector_model_family": "ground_truth",
            "detector_model_size": None,
            "detector_model_variant": None,
            "detector_model_path": None,
        }

    backend = str(DETECTOR_BACKEND).lower().strip()

    if backend == "yolo":
        return {
            "detector_model_family": "yolo11",
            "detector_model_size": str(getattr(aif_config, "AIF_YOLO_MODEL_SIZE", "unknown")),
            "detector_model_variant": str(getattr(aif_config, "AIF_YOLO_MODEL_VARIANT", "unknown")),
            "detector_model_path": str(
                os.environ.get(
                    "AIF_YOLO_MODEL",
                    getattr(aif_config, "AIF_YOLO_MODEL", "unknown"),
                )
            ),
        }

    if backend == "mobilenet_ssd":
        return {
            "detector_model_family": "mobilenetv2_ssd",
            "detector_model_size": str(getattr(aif_config, "AIF_MOBILENET_MODEL_SIZE", "unknown")),
            "detector_model_variant": str(getattr(aif_config, "AIF_MOBILENET_MODEL_VARIANT", "unknown")),
            "detector_model_path": str(
                os.environ.get(
                    "AIF_MBV2SSD_MODEL",
                    getattr(aif_config, "AIF_MOBILENET_MODEL", "unknown"),
                )
            ),
        }

    return {
        "detector_model_family": backend,
        "detector_model_size": "unknown",
        "detector_model_variant": "unknown",
        "detector_model_path": "unknown",
    }

def _experiment_metadata_payload() -> dict:
    detector_used = "ground_truth" if USE_GROUND_TRUTH_DETECTIONS else str(DETECTOR_BACKEND)
    model_meta = _detector_model_metadata_payload()

    return {
        "detector_used": detector_used,
        **model_meta,
        "detector_backend": str(DETECTOR_BACKEND),
        "use_ground_truth_detections": bool(USE_GROUND_TRUTH_DETECTIONS),
        "found_det_threshold": float(FOUND_DET_THRESHOLD),
        "found_streak_steps": int(FOUND_STREAK_STEPS),
        "spawn_conditions": str(SPAWN_CONDITIONS),
        "target_setup": str(TARGET_SETUP),
    }

def _tracks_aif_target_distance(policy_name: str) -> bool:
    return bool(
        TRACK_AIF_TARGET_DISTANCE
        and str(policy_name).lower().strip() in AIF_TARGET_DISTANCE_POLICIES
    )

def _target_distance_stats_payload(
    *,
    policy_name: str,
    count: int,
    sum_m: float,
    min_m,
    max_m,
) -> dict:
    enabled = bool(TRACK_AIF_TARGET_DISTANCE)
    policy_supported = str(policy_name).lower().strip() in AIF_TARGET_DISTANCE_POLICIES
    tracked = bool(enabled and policy_supported)

    count = int(count)
    sum_m = float(sum_m)

    average_m = (sum_m / float(count)) if count > 0 else None

    return {
        "enabled": enabled,
        "policy_supported": policy_supported,
        "tracked": tracked,
        "count": count,
        "sum_m": sum_m,
        "average_m": None if average_m is None else float(average_m),
        "min_m": None if min_m is None else float(min_m),
        "max_m": None if max_m is None else float(max_m),
    }

if __name__ == "__main__":
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    devices = CrazyflieDevices(robot, timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    policy_name = os.environ.get("AIF_POLICY", DEFAULT_POLICY)
    policy = load_policy(policy_name, debug=DEBUG)

    experiment_init_path = _shared_path(EXPERIMENT_INIT_FILENAME)
    gt_path = _shared_path(GT_JSON_FILENAME)
    task_state_path = _shared_path(TASK_STATE_FILENAME)
    controller_progress_path = _shared_path(CONTROLLER_PROGRESS_FILENAME)
    eval_state_path = _shared_path(EVAL_STATE_FILENAME)

    experiment_init = _wait_for_json(
        experiment_init_path,
        validator=lambda d: isinstance(d.get("drone_spawn", None), dict) and ("x" in d["drone_spawn"]) and ("y" in d["drone_spawn"]),
        timeout_s=STARTUP_WAIT_TIMEOUT_S,
        poll_s=STARTUP_WAIT_POLL_S,
    )

    if experiment_init is None:
        print(f"[WARN] experiment init file not available at startup: {experiment_init_path}", flush=True)
        experiment_init = {
            "seed": -1,
            "drone_spawn": {"x": 0.0, "y": 0.0, "yaw": 0.0, "cx": None, "cy": None},
        }

    drone_spawn = experiment_init.get("drone_spawn", {}) or {}
    drone_spawn_x = float(drone_spawn.get("x", 0.0))
    drone_spawn_y = float(drone_spawn.get("y", 0.0))
    drone_spawn_yaw = float(drone_spawn.get("yaw", 0.0))
    drone_spawn_cx = drone_spawn.get("cx", None)
    drone_spawn_cy = drone_spawn.get("cy", None)

    print(
        f"[INIT] drone spawn x={drone_spawn_x:.3f} y={drone_spawn_y:.3f} "
        f"yaw={drone_spawn_yaw:.3f} cell={(drone_spawn_cx, drone_spawn_cy)}",
        flush=True,
    )

    grid_spec = GridSpec(
        min_x=MIN_X,
        max_x=MAX_X,
        min_y=MIN_Y,
        max_y=MAX_Y,
        cell_size=CELL_SIZE_M,
        rows=GRID_ROWS,
        cols=GRID_COLS,
    )

    floor_fov_spec = _make_fov_spec(
        width_m=FOV_WIDTH_M,
        forward_m=FOV_FORWARD_M,
        backward_m=FOV_BACKWARD_M,
    )
    table_fov_spec = _scaled_fov_spec_for_surface(
        TABLE_HEIGHT_M,
        camera_height_m=FLYING_ATTITUDE,
    )

    # Keep the old variable name as the floor/default FoV for backwards compatibility.
    fov_spec = floor_fov_spec
    env_mgr_floor = GridEnvManager(grid_spec, floor_fov_spec)
    env_mgr_table = GridEnvManager(grid_spec, table_fov_spec)
    env_mgr = env_mgr_floor

    Detector = load_detector_class(DETECTOR_BACKEND)

    detector = None
    gt_detector = None

    claimed_found_objects = set()  # (cx, cy, class_name)
    claimed_found_cells = set()

    # key = (cx, cy, class_name) -> consecutive processed detections count
    _found_streak_counts = {}

    validated_found_cells = set()
    validated_found_objects = set()
    TOTAL_OBJECTS = 0
    remaining_objects = 0
    supervisor_status = "running"
    supervisor_reason = None

    visited_cells = set()
    table_surface_cells = set()  # discovered table-height cells, stored as (cx, cy)
    table_surface_confidence = {}  # persistent max desk confidence by discovered table cell (cx, cy)

    autonomy_start_time = None
    gt_seed = int(experiment_init.get("seed", -1))
    final_snapshot_written = False

    if USE_GROUND_TRUTH_DETECTIONS:
        gt_json = _wait_for_json(
            gt_path,
            validator=lambda d: isinstance(d.get("objects", None), list),
            timeout_s=STARTUP_WAIT_TIMEOUT_S,
            poll_s=STARTUP_WAIT_POLL_S,
        )
        if gt_json is None:
            raise FileNotFoundError(f"Required GT file not found or invalid: {gt_path}")

        gt_seed = int(gt_json.get("seed", gt_seed))
        gt_objects = load_gt_objects(gt_path)

        print(f"[GT] Loaded {len(gt_objects)} objects from {gt_path}")
        print("[GT] Example objects:", gt_objects[:3])

        if GT_DETECTOR_BACKEND == "analytic_fov":
            gt_detector = GroundTruthDetector(
                objects=gt_objects,
                grid_spec=grid_spec,
                fov_spec=fov_spec,
                confidence=GT_CONFIDENCE,
            )

        elif GT_DETECTOR_BACKEND == "camera_recognition":
            from vision.ground_truth_detector_v2 import CameraRecognitionGroundTruthDetector

            gt_detector = CameraRecognitionGroundTruthDetector(
                camera=camera,
                objects=gt_objects,
                grid_spec=grid_spec,
                confidence=GT_CONFIDENCE,
                model_to_class={
                    "target_bottle": 0,
                    "target_can": 1,
                    "desk": 2,
                },
                min_box_pixels=2,
                require_known_id=True,
            )
            gt_detector.enable(_recognition_sampling_period_ms(timestep))

        else:
            raise ValueError(
                f"Unknown GT_DETECTOR_BACKEND='{GT_DETECTOR_BACKEND}'. "
                "Use 'analytic_fov' or 'camera_recognition'."
            )
    else:
        print(f"[DETECTOR] backend={DETECTOR_BACKEND}", flush=True)
        detector_class_names = (
            ["target_bottle", "target_can", "desk"]
            if str(TARGET_SETUP).lower().strip() in {"mixed_tables", "mixed_hard"}
            else ["target_bottle", "target_can"]
        )
        detector = Detector(
            model_name=None,
            device="cuda:0",
            conf=0.45,
            iou=0.40,
            img_size=320,
            class_names=detector_class_names,
        )

    print("\n[FoV setup]")
    print(
        f"  cell_size={CELL_SIZE_M}m grid={GRID_ROWS}x{GRID_COLS} "
        f"x∈[{MIN_X:.2f},{MAX_X:.2f}] y∈[{MIN_Y:.2f},{MAX_Y:.2f}]"
    )
    print(
        f"  floor footprint width={FOV_WIDTH_M:.3f}m "
        f"forward={FOV_FORWARD_M:.3f}m backward={FOV_BACKWARD_M:.3f}m"
    )
    print(
        f"  table footprint height={TABLE_HEIGHT_M:.3f}m "
        f"width={(table_fov_spec.half_left + table_fov_spec.half_right):.3f}m "
        f"forward={table_fov_spec.forward:.3f}m backward={table_fov_spec.backward:.3f}m"
    )

    base_log_dir = os.path.join(os.path.dirname(__file__), "logs")
    run_dir = _resolve_run_dir(base_log_dir, policy_name, eval_state_path)
    print("Run output dir:", run_dir)

    claim_log_path = os.path.join(run_dir, f"claim_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    claim_log_file = open(claim_log_path, "w", encoding="utf-8")
    claim_log_file.write("time_s,x_m,y_m,current_cx,current_cy,claim_cx,claim_cy,class_name,confidence\n")

    logger = ExperimentLogger(
        enabled=ENABLE_LOGGING,
        log_dir=run_dir,
        grid_spec=grid_spec,
        min_time_s=AUTONOMY_START_DELAY_S,
    )

    pid_controller = pid_velocity_fixed_height_controller()

    height_desired = FLYING_ATTITUDE
    autonomous_mode = False
    step_count = 0
    occupancy_dt_accum = 0.0

    # Average target distance tracking
    target_distance_count = 0
    target_distance_sum_m = 0.0
    target_distance_min_m = None
    target_distance_max_m = None

    def write_progress(last_time_s: float):
        progress_payload = {
            "policy_name": policy_name,
            "run_dir": run_dir,
            "autonomy_start_time_s": autonomy_start_time,
            "last_controller_time_s": float(last_time_s),
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
            "detector_limitations": _detector_limitations_payload(),
            "experiment_metadata": _experiment_metadata_payload(),
            "target_distance_stats": _target_distance_stats_payload(
                policy_name=policy_name,
                count=target_distance_count,
                sum_m=target_distance_sum_m,
                min_m=target_distance_min_m,
                max_m=target_distance_max_m,
            ),
            "drone_spawn": {
                "x": drone_spawn_x,
                "y": drone_spawn_y,
                "yaw": drone_spawn_yaw,
                "cx": drone_spawn_cx,
                "cy": drone_spawn_cy,
            },
            "surface_height_map": _surface_height_payload(table_surface_cells, table_surface_confidence),
        }
        _atomic_write_json(controller_progress_path, progress_payload)

    print("\n====== AIF Controller ======\n")
    print("AIF_POLICY =", policy_name)
    print("Controls: Arrow keys for planar motion, Q/E yaw, W/S altitude")
    print("Press A to enable autonomous mode, D to disable\n")

    write_progress(last_time_s=0.0)

    last_detection_step = -10**9
    last_detection_time_s = -1e9

    try:
        while robot.step(timestep) != -1:
            step_count += 1
            observation = devices.read()
            dt = observation["dt"]
            time_now = float(observation["time"])

            occupancy_dt_accum += float(dt)

            task_state = _read_json(
                task_state_path,
                {
                    "status": "running",
                    "reason": None,
                    "total_objects": 0,
                    "remaining_objects": 0,
                    "validated_found_cells": [],
                    "validated_found_objects": [],
                },
            )
            supervisor_status = str(task_state.get("status", "running"))
            supervisor_reason = task_state.get("reason", None)
            TOTAL_OBJECTS = int(task_state.get("total_objects", 0))
            remaining_objects = int(task_state.get("remaining_objects", 0))
            validated_found_cells = {
                (int(cx), int(cy))
                for (cx, cy) in task_state.get("validated_found_cells", [])
            }
            validated_found_objects = {
                (
                    int(o["cx"]),
                    int(o["cy"]),
                    str(o["class_name"]),
                )
                for o in task_state.get("validated_found_objects", [])
            }

            if AUTO_START_AUTONOMY and (not autonomous_mode):
                if time_now >= AUTONOMY_START_DELAY_S and supervisor_status == "running":
                    autonomous_mode = True
                    autonomy_start_time = float(time_now)
                    print(f"Autonomous mode: ON (auto at t={time_now:.1f}s)", flush=True)

            forward_desired = 0.0
            sideways_desired = 0.0
            yaw_desired = 0.0
            height_diff_desired = 0.0

            key = keyboard.getKey()
            while key > 0:
                if key == Keyboard.UP:
                    forward_desired += 0.5
                elif key == Keyboard.DOWN:
                    forward_desired -= 0.5
                elif key == Keyboard.RIGHT:
                    sideways_desired -= 0.5
                elif key == Keyboard.LEFT:
                    sideways_desired += 0.5
                elif key == ord("Q"):
                    yaw_desired = +1
                elif key == ord("E"):
                    yaw_desired = -1
                elif key == ord("W"):
                    height_diff_desired = 0.1
                elif key == ord("S"):
                    height_diff_desired = -0.1
                elif key == ord("A"):
                    if not autonomous_mode and supervisor_status == "running":
                        autonomous_mode = True
                        autonomy_start_time = float(time_now)
                        print("Autonomous mode: ON", flush=True)
                elif key == ord("D"):
                    if autonomous_mode:
                        autonomous_mode = False
                        print("Autonomous mode: OFF", flush=True)
                key = keyboard.getKey()

            height_desired = max(0.1, height_desired + height_diff_desired * dt)

            if ENABLE_CAPTURES and (step_count % CAPTURE_EVERY_N_STEPS == 0):
                save_capture(
                    observation["camera"]["image"],
                    width=observation["camera"]["width"],
                    height=observation["camera"]["height"],
                    timestamp=observation["camera"]["time"],
                    output_dir=os.path.join(run_dir, "captures"),
                )

            x_pos = observation["gps"]["x"]
            y_pos = observation["gps"]["y"]
            yaw = observation["imu"]["yaw"]

            floor_coverages = env_mgr_floor.compute_cell_coverages(x_pos, y_pos, yaw)
            table_coverages = env_mgr_table.compute_cell_coverages(x_pos, y_pos, yaw)
            effective_table_projection_cells = (
                _effective_table_projection_cells(
                    table_cells_cxcy=table_surface_cells,
                    floor_coverages_rc=floor_coverages,
                    drone_x=x_pos,
                    drone_y=y_pos,
                    drone_yaw=yaw,
                    grid_spec=grid_spec,
                )
                if ENABLE_HEIGHT_AWARE_FOV
                else set(table_surface_cells)
            )
            coverages = (
                _merge_coverages_by_surface(
                    floor_coverages_rc=floor_coverages,
                    table_coverages_rc=table_coverages,
                    table_cells_cxcy=effective_table_projection_cells,
                )
                if ENABLE_HEIGHT_AWARE_FOV
                else dict(floor_coverages)
            )
            visible_mask = env_mgr_floor.coverage_to_visible_mask(
                coverages,
                threshold=COVERAGE_UPDATE_THRESHOLD,
            )
            current_cell_rc = grid_spec.rowcol_from_xy(x_pos, y_pos)
            current_cell_cxcy = rc_to_cxcy(current_cell_rc)

            if current_cell_cxcy is not None:
                visited_cells.add(current_cell_cxcy)

            # Target-belief evidence remains a max-confidence map per cell because the
            # old AIF policy expects that shape. The raw/event lists below keep every
            # detection, including multiple detections in the same cell and non-target
            # classes such as desks, so claim logic and future table-aware policies do
            # not lose information.
            det_conf_rc = {}          # target-only max confidence per cell, for old AIF belief updates
            det_meta_rc = {}          # metadata for the target event that set det_conf_rc[cell]
            det_all_events_rc = []    # all accepted detections, including desks
            det_target_events_rc = [] # all accepted target detections; no per-cell collapse
            table_conf_rc = {}        # per-frame positive table evidence: max desk confidence per mapped cell
            detector_output = None

            process_detection_this_step = _should_process_detection(
                mode=DETECTION_THROTTLE_MODE,
                step_count=step_count,
                time_now_s=time_now,
                last_det_step=last_detection_step,
                last_det_time_s=last_detection_time_s,
            )

            if process_detection_this_step:
                if USE_GROUND_TRUTH_DETECTIONS:
                    assert gt_detector is not None
                    det_raw = gt_detector.process(
                        drone_x=x_pos,
                        drone_y=y_pos,
                        drone_yaw=yaw,
                    )

                    # New GT detectors return a list of detection events. Keep a compatibility
                    # path for the old dict[(row,col)] = meta format.
                    if isinstance(det_raw, dict):
                        raw_events = []
                        for cell, meta in det_raw.items():
                            ev = dict(meta)
                            ev["cell_rc"] = cell
                            raw_events.append(ev)
                    else:
                        raw_events = list(det_raw or [])

                    for raw_event in raw_events:
                        cell_rc = raw_event.get("cell_rc", None)
                        if cell_rc is None:
                            continue
                        cell_rc = (int(cell_rc[0]), int(cell_rc[1]))

                        # Analytic GT still needs the controller FoV mask. Camera-recognition
                        # GT already means Webots saw the object in the rendered camera frame.
                        if GT_DETECTOR_BACKEND == "analytic_fov" and not visible_mask.get(cell_rc, False):
                            continue

                        conf = float(raw_event.get("confidence", 0.0))
                        class_id = int(raw_event.get("class_id", -1))
                        class_name = str(raw_event.get("class_name", raw_event.get("label", "unknown")))
                        label = str(raw_event.get("label", class_name))

                        event = {
                            **raw_event,
                            "cell_rc": cell_rc,
                            "confidence": conf,
                            "class_id": class_id,
                            "label": label,
                            "class_name": class_name,
                        }
                        det_all_events_rc.append(event)

                        if not _is_target_object_class(class_name, class_id):
                            continue

                        det_target_events_rc.append(event)
                        prev = det_conf_rc.get(cell_rc, 0.0)
                        if conf > prev:
                            det_conf_rc[cell_rc] = conf
                            det_meta_rc[cell_rc] = {
                                "class_id": class_id,
                                "label": label,
                                "class_name": class_name,
                            }
                else:
                    assert detector is not None
                    detector_output = detector.process(
                        observation["camera"]["image"],
                        width=observation["camera"]["width"],
                        height=observation["camera"]["height"],
                        timestamp=observation["camera"]["time"],
                    )

                    img_w = int(observation["camera"]["width"])
                    img_h = int(observation["camera"]["height"])

                    raw_detection_events = []
                    for det in detector_output.detections:
                        bbox = _clamp_bbox_xyxy(det.get("bbox_xyxy"), img_w=img_w, img_h=img_h)
                        if bbox is None:
                            continue

                        conf = float(det.get("confidence", 0.0))
                        class_id = int(det.get("class_id", -1))
                        label = str(det.get("label", "unknown"))
                        class_name = str(det.get("class_name", label))

                        raw_detection_events.append(
                            {
                                **det,
                                "bbox_xyxy": bbox,
                                "confidence": conf,
                                "class_id": class_id,
                                "label": label,
                                "class_name": class_name,
                            }
                        )

                    # First pass: collect desk/table bboxes and use the table-plane
                    # projection to mark all grid cells covered by the detected desk bbox.
                    # This is perception-derived surface discovery, not supervisor injection.
                    desk_bboxes = []
                    for det in raw_detection_events:
                        class_id = int(det.get("class_id", -1))
                        class_name = str(det.get("class_name", det.get("label", "unknown")))
                        conf = float(det.get("confidence", 0.0))
                        if not _is_desk_object_class(class_name, class_id):
                            continue

                        bbox = det.get("bbox_xyxy")
                        desk_bboxes.append(bbox)

                        covered_cells_rc = _sample_bbox_projected_cells_rc(
                            env_mgr_table,
                            bbox_xyxy=bbox,
                            img_w=img_w,
                            img_h=img_h,
                            drone_x=x_pos,
                            drone_y=y_pos,
                            drone_yaw=yaw,
                        )
                        primary_cell_rc = env_mgr_table.detection_cell_from_bbox(
                            bbox_xyxy=bbox,
                            img_w=img_w,
                            img_h=img_h,
                            drone_x=x_pos,
                            drone_y=y_pos,
                            drone_yaw=yaw,
                        )
                        if primary_cell_rc is None and covered_cells_rc:
                            primary_cell_rc = sorted(covered_cells_rc)[0]

                        if conf >= SURFACE_DETECTION_CONF_THRESHOLD:
                            for rc in covered_cells_rc:
                                cell_cxcy = rc_to_cxcy(rc)
                                if cell_cxcy is not None:
                                    cell_cxcy = (int(cell_cxcy[0]), int(cell_cxcy[1]))
                                    rc_key = (int(rc[0]), int(rc[1]))
                                    table_surface_cells.add(cell_cxcy)
                                    table_surface_confidence[cell_cxcy] = max(
                                        float(conf),
                                        float(table_surface_confidence.get(cell_cxcy, 0.0)),
                                    )
                                    table_conf_rc[rc_key] = max(
                                        float(conf),
                                        float(table_conf_rc.get(rc_key, 0.0)),
                                    )

                        if primary_cell_rc is not None:
                            event = {
                                **det,
                                "cell_rc": (int(primary_cell_rc[0]), int(primary_cell_rc[1])),
                                "cell_rc_table": (int(primary_cell_rc[0]), int(primary_cell_rc[1])),
                                "covered_cells_rc": sorted([[int(r), int(c)] for (r, c) in covered_cells_rc]),
                                "surface_assumption": "table",
                            }
                            det_all_events_rc.append(event)

                    # Second pass: map target bboxes. For targets, compute both the
                    # floor-plane and table-plane cell. Use table projection when the
                    # target overlaps a desk bbox in the same frame, or when the
                    # table-plane cell is already known as a table-height cell.
                    for det in raw_detection_events:
                        class_id = int(det.get("class_id", -1))
                        class_name = str(det.get("class_name", det.get("label", "unknown")))
                        if not _is_target_object_class(class_name, class_id):
                            continue

                        bbox = det.get("bbox_xyxy")
                        conf = float(det.get("confidence", 0.0))
                        label = str(det.get("label", class_name))

                        floor_cell_rc = env_mgr_floor.detection_cell_from_bbox(
                            bbox_xyxy=bbox,
                            img_w=img_w,
                            img_h=img_h,
                            drone_x=x_pos,
                            drone_y=y_pos,
                            drone_yaw=yaw,
                        )
                        table_cell_rc = env_mgr_table.detection_cell_from_bbox(
                            bbox_xyxy=bbox,
                            img_w=img_w,
                            img_h=img_h,
                            drone_x=x_pos,
                            drone_y=y_pos,
                            drone_yaw=yaw,
                        )

                        overlaps_desk = _bbox_overlaps_any_desk(bbox, desk_bboxes)
                        table_cell_cxcy = rc_to_cxcy(table_cell_rc) if table_cell_rc is not None else None
                        table_cell_is_known = (table_cell_cxcy in table_surface_cells) if table_cell_cxcy is not None else False

                        use_table_projection = bool(overlaps_desk or table_cell_is_known)
                        cell_rc = table_cell_rc if use_table_projection else floor_cell_rc
                        if cell_rc is None:
                            # If the preferred projection failed, use the other one as a
                            # last-resort mapping for a real image detection.
                            cell_rc = floor_cell_rc if table_cell_rc is None else table_cell_rc
                        if cell_rc is None:
                            continue
                        cell_rc = (int(cell_rc[0]), int(cell_rc[1]))

                        event = {
                            **det,
                            "cell_rc": cell_rc,
                            "cell_rc_floor": None if floor_cell_rc is None else (int(floor_cell_rc[0]), int(floor_cell_rc[1])),
                            "cell_rc_table": None if table_cell_rc is None else (int(table_cell_rc[0]), int(table_cell_rc[1])),
                            "bbox_overlaps_desk": bool(overlaps_desk),
                            "table_cell_is_known": bool(table_cell_is_known),
                            "surface_assumption": "table" if use_table_projection else "floor",
                            "confidence": conf,
                            "class_id": class_id,
                            "label": label,
                            "class_name": class_name,
                        }
                        det_all_events_rc.append(event)
                        det_target_events_rc.append(event)

                        prev = det_conf_rc.get(cell_rc, 0.0)
                        if conf > prev:
                            det_conf_rc[cell_rc] = conf
                            det_meta_rc[cell_rc] = {
                                "class_id": class_id,
                                "label": label,
                                "class_name": class_name,
                                "surface_assumption": event["surface_assumption"],
                            }

                # Surface discovery: any desk/table detection flags its mapped cells as
                # table-height. This is intentionally perception-derived, not injected
                # from supervisor geometry. Real YOLO desk detections may already carry
                # covered_cells_rc from bbox sampling; GT desk detections can be sampled
                # here from their bbox_center_px/bbox_size_px if available.
                for event in det_all_events_rc:
                    class_id = int(event.get("class_id", -1))
                    class_name = str(event.get("class_name", event.get("label", "unknown")))
                    conf = float(event.get("confidence", 0.0))
                    if conf < SURFACE_DETECTION_CONF_THRESHOLD:
                        continue
                    if not _is_desk_object_class(class_name, class_id):
                        continue

                    cells_to_mark_rc = set()
                    for rc in event.get("covered_cells_rc", []) or []:
                        try:
                            cells_to_mark_rc.add((int(rc[0]), int(rc[1])))
                        except Exception:
                            continue

                    if not cells_to_mark_rc:
                        bbox = _event_bbox_xyxy(event)
                        if bbox is not None:
                            try:
                                img_w_s = int(observation["camera"]["width"])
                                img_h_s = int(observation["camera"]["height"])
                                cells_to_mark_rc |= _sample_bbox_projected_cells_rc(
                                    env_mgr_table,
                                    bbox_xyxy=bbox,
                                    img_w=img_w_s,
                                    img_h=img_h_s,
                                    drone_x=x_pos,
                                    drone_y=y_pos,
                                    drone_yaw=yaw,
                                )
                            except Exception:
                                pass

                    if not cells_to_mark_rc:
                        cell_rc = event.get("cell_rc", None)
                        if cell_rc is not None:
                            try:
                                cells_to_mark_rc.add((int(cell_rc[0]), int(cell_rc[1])))
                            except Exception:
                                pass

                    for rc in cells_to_mark_rc:
                        cell_cxcy = rc_to_cxcy(rc)
                        if cell_cxcy is not None:
                            cell_cxcy = (int(cell_cxcy[0]), int(cell_cxcy[1]))
                            rc_key = (int(rc[0]), int(rc[1]))
                            table_surface_cells.add(cell_cxcy)
                            table_surface_confidence[cell_cxcy] = max(
                                float(conf),
                                float(table_surface_confidence.get(cell_cxcy, 0.0)),
                            )
                            table_conf_rc[rc_key] = max(
                                float(conf),
                                float(table_conf_rc.get(rc_key, 0.0)),
                            )

                if ENABLE_HEIGHT_AWARE_FOV:
                    # Recompute the selected coverage map after incorporating desk detections
                    # from this frame, so current-frame target/table evidence uses table-height
                    # geometry for both directly detected table cells and transient cells
                    # geometrically behind them from the current viewpoint.
                    effective_table_projection_cells = _effective_table_projection_cells(
                        table_cells_cxcy=table_surface_cells,
                        floor_coverages_rc=floor_coverages,
                        drone_x=x_pos,
                        drone_y=y_pos,
                        drone_yaw=yaw,
                        grid_spec=grid_spec,
                    )
                    coverages = _merge_coverages_by_surface(
                        floor_coverages_rc=floor_coverages,
                        table_coverages_rc=table_coverages,
                        table_cells_cxcy=effective_table_projection_cells,
                    )
                    visible_mask = env_mgr_floor.coverage_to_visible_mask(
                        coverages,
                        threshold=COVERAGE_UPDATE_THRESHOLD,
                    )

                last_detection_step = step_count
                last_detection_time_s = time_now

            coverages_cxcy = dict_rc_to_cxcy(coverages)
            visible_mask_cxcy = dict_rc_to_cxcy(visible_mask)

            det_conf_cxcy = dict_rc_to_cxcy(det_conf_rc)
            det_meta_cxcy = dict_rc_to_cxcy(det_meta_rc)
            det_all_events_cxcy = [_detection_event_rc_to_cxcy(e) for e in det_all_events_rc]
            det_target_events_cxcy = [_detection_event_rc_to_cxcy(e) for e in det_target_events_rc]
            table_conf_cxcy = dict_rc_to_cxcy(table_conf_rc)

            # q_table positive evidence comes from per-frame desk bbox projections.
            # q_table negative evidence must use the same surface-selected visibility
            # and coverage as target belief updates: floor coverage for normal cells,
            # table-height coverage for known/effective table-projection cells.
            table_visible_mask_cxcy = visible_mask_cxcy
            table_coverages_cxcy = coverages_cxcy

            # Generic detection field for AIF remains target-only for the old policy.
            det_conf_generic_cxcy = dict(det_conf_cxcy)

            # Policy only receives evidence on processed perception steps. The old AIF
            # policy uses det_confidence; future table-aware policies can read the event lists.
            policy_visible_mask_cxcy = visible_mask_cxcy if process_detection_this_step else {}
            policy_det_conf_generic_cxcy = det_conf_generic_cxcy if process_detection_this_step else {}
            policy_det_conf_cxcy = det_conf_cxcy if process_detection_this_step else {}
            policy_det_meta_cxcy = det_meta_cxcy if process_detection_this_step else {}
            policy_det_all_events_cxcy = det_all_events_cxcy if process_detection_this_step else []
            policy_det_target_events_cxcy = det_target_events_cxcy if process_detection_this_step else []
            policy_table_confidence_cxcy = table_conf_cxcy if process_detection_this_step else {}
            policy_table_visible_mask_cxcy = table_visible_mask_cxcy if process_detection_this_step else {}
            policy_table_coverages_cxcy = table_coverages_cxcy if process_detection_this_step else {}
            surface_height_map_payload = _surface_height_payload(table_surface_cells, table_surface_confidence)

            best_obj = None
            if det_conf_cxcy:
                best_cell, best_conf = max(det_conf_cxcy.items(), key=lambda kv: float(kv[1]))
                best_meta = det_meta_cxcy.get(best_cell, {})
                best_class_name = str(best_meta.get("class_name", best_meta.get("label", "unknown")))
                best_class_id = int(best_meta.get("class_id", -1))
                best_obj = {
                    "cell": best_cell,
                    "confidence": float(best_conf),
                    "class_name": best_class_name,
                    "class_id": best_class_id,
                }

            # ---- Per-candidate streak accumulation ----
            # IMPORTANT:
            # Update/reset streaks ONLY on processed perception steps.
            # Skipped steps are "no new evidence", not negative evidence.
            if process_detection_this_step:
                current_seen_keys = set()

                for event in det_target_events_cxcy:
                    cell = event.get("cell", None)
                    if cell is None:
                        continue

                    conf = float(event.get("confidence", 0.0))
                    class_id = int(event.get("class_id", -1))
                    class_name = str(event.get("class_name", event.get("label", "unknown")))

                    # Desk and any future context classes are allowed to exist in the
                    # detection stream, but only target objects can become found claims.
                    if not _is_target_object_class(class_name, class_id):
                        continue

                    candidate_key = (
                        int(cell[0]),
                        int(cell[1]),
                        class_name,
                    )

                    if candidate_key in claimed_found_objects:
                        continue

                    if conf < FOUND_DET_THRESHOLD:
                        continue

                    current_seen_keys.add(candidate_key)
                    _found_streak_counts[candidate_key] = int(_found_streak_counts.get(candidate_key, 0)) + 1

                    if _found_streak_counts[candidate_key] >= FOUND_STREAK_STEPS:
                        claimed_found_objects.add(candidate_key)
                        claimed_found_cells = {(cx, cy) for (cx, cy, _) in claimed_found_objects}

                        print(
                            f"[FOUND CLAIM] cell={(int(cell[0]), int(cell[1]))} class={class_name} "
                            f"conf={conf:.2f} streak={_found_streak_counts[candidate_key]}",
                            flush=True,
                        )

                        if claim_log_file is not None:
                            cur_cx, cur_cy = current_cell_cxcy if current_cell_cxcy is not None else (-1, -1)
                            claim_log_file.write(
                                f"{time_now:.6f},{x_pos:.6f},{y_pos:.6f},"
                                f"{int(cur_cx)},{int(cur_cy)},"
                                f"{int(cell[0])},{int(cell[1])},"
                                f"{class_name},{conf:.6f}\n"
                            )
                            claim_log_file.flush()

                        _found_streak_counts.pop(candidate_key, None)



                stale_keys = [k for k in list(_found_streak_counts.keys()) if k not in current_seen_keys]
                for k in stale_keys:
                    _found_streak_counts.pop(k, None)

            claimed_found_cells = {(cx, cy) for (cx, cy, _) in claimed_found_objects}

            if DEBUG.get("fov", False) and (step_count % int(DEBUG.get("every_n", 50)) == 0):
                top_rc = sorted(coverages.items(), key=lambda kv: kv[1], reverse=True)[:12]
                num_vis = sum(1 for v in visible_mask.values() if v)

                print("\n[FOV DEBUG]")
                print(
                    f"  step={step_count}  pos=({x_pos:.2f},{y_pos:.2f}) "
                    f"yaw={yaw:.3f} rad ({yaw * 180.0 / 3.14159:.1f} deg)"
                )
                print(f"  current_cell(cx,cy)={current_cell_cxcy}")
                print(
                    f"  coverages: {len(coverages)} cells intersect FoV; {num_vis} >= {COVERAGE_UPDATE_THRESHOLD:.2f}; "
                    f"table_cells={len(table_surface_cells)} effective_table_projection_cells={len(effective_table_projection_cells)}"
                )
                for (r, c), frac in top_rc:
                    print(f"    cell=(cx={c},cy={r}) coverage={frac*100:5.1f}% visible={visible_mask[(r,c)]}")

            if ENABLE_LOGGING and (step_count % LOG_EVERY_N_STEPS == 0):
                logger.log_occupancy(
                    time_s=observation["time"],
                    x_m=x_pos,
                    y_m=y_pos,
                    dt_s=occupancy_dt_accum,
                )
                occupancy_dt_accum = 0.0

            if (step_count % LOG_EVERY_N_STEPS == 0) or (best_obj is not None and best_obj["cell"] in claimed_found_cells):
                write_progress(last_time_s=time_now)

            obs_for_policy = {
                **observation,
                "step": step_count,
                "pose": {"x": x_pos, "y": y_pos, "yaw": yaw},
                "grid_spec": {
                    "min_x": grid_spec.min_x,
                    "max_x": grid_spec.max_x,
                    "min_y": grid_spec.min_y,
                    "max_y": grid_spec.max_y,
                    "cell_size": grid_spec.cell_size,
                    "rows": grid_spec.rows,
                    "cols": grid_spec.cols,
                },
                **(
                    {}
                    if USE_GROUND_TRUTH_DETECTIONS
                    else {"detections": [] if detector_output is None else detector_output.detections}
                ),
                "grid": {
                    "current_cell": current_cell_cxcy,
                    "coverages": coverages_cxcy,
                    "visible_mask": policy_visible_mask_cxcy,
                    "coverage_threshold": COVERAGE_UPDATE_THRESHOLD,
                    "det_confidence": policy_det_conf_generic_cxcy,
                    "det_confidence_multiclass": policy_det_conf_cxcy,
                    "det_metadata_multiclass": policy_det_meta_cxcy,
                    "detections_all": policy_det_all_events_cxcy,
                    "target_detections": policy_det_target_events_cxcy,
                    # Per-frame table-observation fields for policies that maintain q_table.
                    # table_confidence is positive desk/table evidence for cells hit by
                    # bbox probing. table_visible_mask/table_coverages provide negative
                    # no-table evidence using the already surface-selected coverage map.
                    "table_confidence": policy_table_confidence_cxcy,
                    "table_visible_mask": policy_table_visible_mask_cxcy,
                    "table_coverages": policy_table_coverages_cxcy,
                    "table_confidence_persistent": dict(table_surface_confidence),
                    "perception_processed_this_step": bool(process_detection_this_step),
                    "surface_height_map": surface_height_map_payload,
                    "table_cells": sorted(list(table_surface_cells)),
                    # Current-frame geometry-only projection mask. Includes table_cells plus
                    # transient cells behind perceived table cells that use table-height
                    # coverage for this observation. It is diagnostic only; policy logic
                    # consumes the already merged coverages/visible masks above.
                    "effective_table_projection_cells": sorted(list(effective_table_projection_cells)),
                },
                "task": {
                    "found_cells": sorted(list(claimed_found_cells)),
                    "validated_found_cells": sorted(list(validated_found_cells)),
                    "validated_found_objects": sorted(
                        [
                            {"cx": int(cx), "cy": int(cy), "class_name": str(class_name)}
                            for (cx, cy, class_name) in validated_found_objects
                        ],
                        key=lambda d: (d["cy"], d["cx"], d["class_name"]),
                    ),
                    "remaining_objects": remaining_objects,
                    "total_objects": TOTAL_OBJECTS,
                    "found_det_threshold": FOUND_DET_THRESHOLD,
                    "found_streak_steps": FOUND_STREAK_STEPS,
                    "snapshot": {
                        "enabled": ENABLE_SNAPSHOTS,
                        "dir": os.path.join(run_dir, "snapshots") if run_dir else None,
                        "every_n": SNAPSHOT_EVERY_N_STEPS,
                        "on_stuck": SNAPSHOT_ON_STUCK,
                        "stuck_replan_limit": STUCK_REPLAN_LIMIT,
                    },
                },
                "fov_spec": {
                    "half_left": fov_spec.half_left,
                    "half_right": fov_spec.half_right,
                    "forward": fov_spec.forward,
                    "backward": fov_spec.backward,
                },
                "fov_spec_by_surface": {
                    "floor": {
                        "height_m": float(FLOOR_HEIGHT_M),
                        "half_left": floor_fov_spec.half_left,
                        "half_right": floor_fov_spec.half_right,
                        "forward": floor_fov_spec.forward,
                        "backward": floor_fov_spec.backward,
                    },
                    "table": {
                        "height_m": float(TABLE_HEIGHT_M),
                        "half_left": table_fov_spec.half_left,
                        "half_right": table_fov_spec.half_right,
                        "forward": table_fov_spec.forward,
                        "backward": table_fov_spec.backward,
                    },
                },
            }

            if supervisor_status != "running":
                if autonomous_mode:
                    autonomous_mode = False
                    print(f"Autonomous mode: OFF ({supervisor_status})", flush=True)

                if (not final_snapshot_written) and hasattr(policy, "save_snapshot_from_controller"):
                    try:
                        policy.save_snapshot_from_controller(
                            obs_for_policy,
                            reason=str(supervisor_reason or supervisor_status),
                        )
                        final_snapshot_written = True
                    except Exception as e:
                        print(f"[SNAPSHOT] final snapshot failed: {e}", flush=True)

            if autonomous_mode and supervisor_status == "running":
                policy_action = policy.compute_action(obs_for_policy)
                forward_desired = policy_action.get("forward", 0.0)
                sideways_desired = policy_action.get("sideways", 0.0)
                yaw_desired = policy_action.get("yaw_rate", 0.0)

                if _tracks_aif_target_distance(policy_name):
                    target_selection = policy_action.get("target_selection", None)

                    if isinstance(target_selection, dict):
                        distance_m = target_selection.get("distance_m", None)

                        if distance_m is not None:
                            distance_m = float(distance_m)

                            target_distance_count += 1
                            target_distance_sum_m += distance_m

                            if target_distance_min_m is None or distance_m < target_distance_min_m:
                                target_distance_min_m = distance_m

                            if target_distance_max_m is None or distance_m > target_distance_max_m:
                                target_distance_max_m = distance_m

                            write_progress(last_time_s=time_now)

            if DEBUG.get("cmd", False) and (step_count % int(DEBUG.get("every_n", 50)) == 0):
                print(
                    f"[cmd] step={step_count} auto={autonomous_mode} "
                    f"fwd={forward_desired:.2f} side={sideways_desired:.2f} yaw={yaw_desired:.2f} "
                    f"h_des={height_desired:.2f} z={observation['gps']['z']:.2f}"
                )

            motor_power = pid_controller.pid(
                dt,
                forward_desired,
                sideways_desired,
                yaw_desired,
                height_desired,
                observation["imu"]["roll"],
                observation["imu"]["pitch"],
                observation["gyro"]["yaw_rate"],
                observation["gps"]["z"],
                observation["velocity"]["vx"],
                observation["velocity"]["vy"],
            )

            devices.set_motor_speeds(motor_power)

            if ENABLE_LOGGING and (step_count % LOG_EVERY_N_STEPS == 0):
                if USE_GROUND_TRUTH_DETECTIONS:
                    det_count = len(det_all_events_rc)
                else:
                    det_count = len(det_all_events_rc)

                logger.log_step(
                    time=observation["time"],
                    forward=forward_desired,
                    sideways=sideways_desired,
                    yaw_rate=yaw_desired,
                    altitude=observation["gps"]["z"],
                    detections=det_count,
                )

    finally:
        try:
            write_progress(last_time_s=float(time_now) if "time_now" in locals() else 0.0)
        except Exception:
            pass

        logger_paths = logger.close()

        try:
            if claim_log_file is not None:
                claim_log_file.close()
        except Exception:
            pass

        if ENABLE_OCCUPANCY_HEATMAP or ENABLE_TRAJECTORY_PLOT:
            try:
                visual_outputs = generate_run_visualizations(
                    run_dir=run_dir,
                    make_heatmap=ENABLE_OCCUPANCY_HEATMAP,
                    make_trajectory=ENABLE_TRAJECTORY_PLOT,
                    occupancy_summary_path=logger_paths.get("occupancy_summary"),
                    occupancy_log_path=logger_paths.get("occupancy_log"),
                    gt_json_path=gt_path if os.path.exists(gt_path) else None,
                    plots_subdir=TRAJECTORY_PLOTS_SUBDIR,
                )

                for name, path in visual_outputs.items():
                    print(f"[VIS] wrote {name}: {path}", flush=True)

            except Exception as e:
                print(f"[VIS] failed to generate run visualizations: {e}", flush=True)