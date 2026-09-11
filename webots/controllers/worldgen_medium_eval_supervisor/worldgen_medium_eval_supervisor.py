# webots/controllers/worldgen_medium_eval_supervisor/worldgen_medium_eval_supervisor.py

from __future__ import annotations

import colorsys
import math
import os
import random
import statistics
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from controller import Supervisor

ENVIRONMENT_ID = "medium"
RUN_MODE = "evaluation"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROLLERS_DIR = os.path.dirname(CURRENT_DIR)

if CONTROLLERS_DIR not in sys.path:
    sys.path.insert(0, CONTROLLERS_DIR)

from common.shared_state import (
    CONTROLLER_READY_FILENAME,
    IPC_SESSION_FILENAME,
    SharedStateStore,
    atomic_write_json,
    establish_supervisor_session,
    read_json,
)

SHARED_STATE = SharedStateStore.from_controller_file(__file__)

from crazyflie_AIF.config import (
    AIF_DETECTOR as CONFIGURED_DETECTOR,
    AIF_POLICY as DEFAULT_POLICY,
    DETECTOR_SELECTION as CONFIGURED_DETECTOR_SELECTION,
    normalize_policy_id,
)

CONFIGURED_DETECTOR_SIGNATURE = CONFIGURED_DETECTOR_SELECTION.metadata()

# -----------------------------------------------------------------------------
# Multi-run evaluation supervisor for the medium scenario
# -----------------------------------------------------------------------------
# Keeps the same shared-file/reporting lifecycle as the original eval supervisor,
# while replacing the floor-only spawn process with table-aware target spawns.
# Desks are exported for detection/GT/surface-height discovery but are not task
# objects to be found.

# ---- Evaluation settings ----
EVAL_NUM_RUNS = int(os.environ.get("MEDIUM_EVAL_NUM_RUNS", "100"))
FIXED_10_EVAL_SEEDS = [28, 31, 34, 37, 40, 43, 46, 49, 52, 55]

WARMUP_SECONDS = float(os.environ.get("MEDIUM_EVAL_WARMUP_SECONDS", "15.0"))
EPISODE_SECONDS = float(os.environ.get("MEDIUM_EVAL_EPISODE_SECONDS", "180.0"))
TOTAL_SECONDS = WARMUP_SECONDS + EPISODE_SECONDS
FINALIZATION_GRACE_SECONDS = 1.0

# ---- Scenario setup ----
N_BOTTLES = int(os.environ.get("MEDIUM_EVAL_N_BOTTLES", "3"))
N_CANS = int(os.environ.get("MEDIUM_EVAL_N_CANS", "3"))
N_TABLES = int(os.environ.get("MEDIUM_EVAL_N_TABLES", "3"))

P_TARGET_ON_TABLE = float(os.environ.get("MEDIUM_EVAL_P_TARGET_ON_TABLE", "0.90"))

# Number of eval runs, starting from run_index=0, where one randomly selected
# table is forcibly excluded from on-table target spawning. Other tables can
# still end up empty probabilistically.
FORCE_ONE_EMPTY_TABLE_NUM_RUNS = int(
    os.environ.get("MEDIUM_EVAL_FORCE_ONE_EMPTY_TABLE_NUM_RUNS", "5")
)

# ---- Class / proto mapping ----
CLASS_BOTTLE = "target_bottle"
CLASS_CAN = "target_can"
CLASS_DESK = "desk"

CLASS_NAME_TO_ID = {
    CLASS_BOTTLE: 0,
    CLASS_CAN: 1,
    CLASS_DESK: 2,
}
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}
TARGET_CLASS_NAMES = {CLASS_BOTTLE, CLASS_CAN}

BOTTLE_PROTO_NODE = os.environ.get("MEDIUM_EVAL_BOTTLE_PROTO", "DatasetBeerBottle")
CAN_PROTO_NODE = os.environ.get("MEDIUM_EVAL_CAN_PROTO", "DatasetCan")

# ---- Arena / grid ----
ARENA_X = 5.5
ARENA_Y = 6.5
CELL_SIZE_M = 0.5
GRID_COLS = int(ARENA_X / CELL_SIZE_M)
GRID_ROWS = int(ARENA_Y / CELL_SIZE_M)

WALL_MARGIN = 0.35
SPAWN_CENTER_DISC_RADIUS_M = float(os.environ.get("MEDIUM_EVAL_SPAWN_CENTER_DISC_RADIUS_M", "0.15"))
DRONE_TARGET_EXCLUSION_RADIUS_M = float(os.environ.get("MEDIUM_EVAL_DRONE_TARGET_EXCLUSION_RADIUS_M", "1.0"))
DESK_DRONE_EXCLUSION_RADIUS_M = float(os.environ.get("MEDIUM_EVAL_DESK_DRONE_EXCLUSION_RADIUS_M", "1.25"))
TARGET_MIN_SEPARATION_M = float(os.environ.get("MEDIUM_EVAL_TARGET_MIN_SEPARATION_M", "0.22"))

# ---- Desk/table geometry ----
DESK_LENGTH_M = float(os.environ.get("MEDIUM_EVAL_DESK_LENGTH_M", "1.2"))
DESK_WIDTH_M = float(os.environ.get("MEDIUM_EVAL_DESK_WIDTH_M", "0.7"))
DESK_HEIGHT_M = float(os.environ.get("MEDIUM_EVAL_DESK_HEIGHT_M", "0.72"))
DESK_TOP_THICKNESS_M = float(os.environ.get("MEDIUM_EVAL_DESK_TOP_THICKNESS_M", "0.08"))
DESK_SPAWN_EDGE_MARGIN_M = float(os.environ.get("MEDIUM_EVAL_DESK_SPAWN_EDGE_MARGIN_M", "0.05"))
DESK_FLOOR_EXCLUSION_MARGIN_M = float(os.environ.get("MEDIUM_EVAL_DESK_FLOOR_EXCLUSION_MARGIN_M", "0.50"))
DESK_MIN_CENTER_DISTANCE_M = float(os.environ.get("MEDIUM_EVAL_DESK_MIN_CENTER_DISTANCE_M", "1.35"))
DESK_WALL_MARGIN_M = float(os.environ.get("MEDIUM_EVAL_DESK_WALL_MARGIN_M", "0.18"))
DESK_TABLETOP_TEXTURE_URL = os.environ.get(
    "MEDIUM_EVAL_DESK_TABLETOP_TEXTURE_URL",
    "world_resources/wooden_tabletop.jpeg",
)

# ---- Target physical metadata ----
BOTTLE_BBOX_RADIUS_M = 0.035
BOTTLE_BBOX_HEIGHT_M = 0.245
CAN_BBOX_RADIUS_M = 0.018
CAN_BBOX_HEIGHT_M = 0.125

OBJECT_SPECS = [
    {
        "count": N_BOTTLES,
        "proto": BOTTLE_PROTO_NODE,
        "class_id": CLASS_NAME_TO_ID[CLASS_BOTTLE],
        "class_name": CLASS_BOTTLE,
        "model": CLASS_BOTTLE,
        "def_prefix": "TARGET_BOTTLE",
        "bbox_radius_m": BOTTLE_BBOX_RADIUS_M,
        "bbox_height_m": BOTTLE_BBOX_HEIGHT_M,
    },
    {
        "count": N_CANS,
        "proto": CAN_PROTO_NODE,
        "class_id": CLASS_NAME_TO_ID[CLASS_CAN],
        "class_name": CLASS_CAN,
        "model": CLASS_CAN,
        "def_prefix": "TARGET_CAN",
        "bbox_radius_m": CAN_BBOX_RADIUS_M,
        "bbox_height_m": CAN_BBOX_HEIGHT_M,
    },
]

# ---- Drone placement ----
DRONE_DEF = "CRAZYFLIE"
DRONE_SPAWN_X_MIN = float(os.environ.get("MEDIUM_EVAL_DRONE_SPAWN_X_MIN", "-2.5"))
DRONE_SPAWN_X_MAX = float(os.environ.get("MEDIUM_EVAL_DRONE_SPAWN_X_MAX", "2.5"))
DRONE_SPAWN_Y_MIN = float(os.environ.get("MEDIUM_EVAL_DRONE_SPAWN_Y_MIN", "-3.0"))
DRONE_SPAWN_Y_MAX = float(os.environ.get("MEDIUM_EVAL_DRONE_SPAWN_Y_MAX", "3.0"))
DRONE_SPAWN_YAW = float(os.environ.get("MEDIUM_EVAL_DRONE_SPAWN_YAW", "0.0"))

# ---- Shared files ----
SPAWNED_OBJECTS_FILENAME = "spawned_objects.json"
EXPERIMENT_INIT_FILENAME = "experiment_init.json"
TASK_STATE_FILENAME = "task_state.json"
CONTROLLER_PROGRESS_FILENAME = "controller_progress.json"
EVAL_STATE_FILENAME = "eval_state.json"

# ---- Reports ----
REPORT_SUBDIR = "run_report"
EVAL_REPORT_SUBDIR = "eval_report"
INCLUDE_RUNS_IN_EVAL_REPORT = False
INCLUDE_RUN_METRICS_IN_EVAL_REPORT = True

Point = Tuple[float, float]
Cell = Tuple[int, int]


def dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _controller_logs_dir() -> str:
    here = os.path.dirname(__file__)
    logs_dir = os.path.abspath(os.path.join(here, "..", "crazyflie_AIF", "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _node_id(node) -> Optional[int]:
    if hasattr(node, "getId"):
        return int(node.getId())
    if hasattr(node, "get_id"):
        return int(node.get_id())
    return None


def _instance_color(index: int) -> tuple[float, float, float]:
    hue = (index * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (max(0.1, r), max(0.1, g), max(0.1, b))


def _configured_eval_seeds() -> list[int]:
    n = int(EVAL_NUM_RUNS)
    if n == len(FIXED_10_EVAL_SEEDS):
        return list(FIXED_10_EVAL_SEEDS)
    if n > len(FIXED_10_EVAL_SEEDS):
        return list(range(1, n + 1))
    raise ValueError(
        f"EVAL_NUM_RUNS={n} is below the fixed baseline size "
        f"{len(FIXED_10_EVAL_SEEDS)}. Use 10 or any value above 10."
    )


def _normalize_detector_limitations(payload: dict | None) -> dict:
    payload = payload or {}
    mode = str(payload.get("mode", "unknown")).lower()
    out = {"mode": mode, "frames_interval": None, "time_interval_s": None}
    if mode == "frames":
        val = payload.get("frames_interval", None)
        out["frames_interval"] = None if val is None else int(val)
    elif mode == "time":
        val = payload.get("time_interval_s", None)
        out["time_interval_s"] = None if val is None else float(val)
    return out


def _normalize_experiment_metadata(payload: dict | None) -> dict:
    payload = payload or {}
    claim_confidence_threshold = payload.get("claim_confidence_threshold", payload.get("found_det_threshold", None))
    return {
        "environment": str(payload.get("environment", payload.get("target_setup", ENVIRONMENT_ID))),
        "run_mode": str(payload.get("run_mode", RUN_MODE)),
        "detector_used": str(payload.get("detector_used", "unknown")),
        "detector_model_family": None if payload.get("detector_model_family", None) is None else str(payload.get("detector_model_family")),
        "detector_model_size": None if payload.get("detector_model_size", None) is None else str(payload.get("detector_model_size")),
        "detector_model_variant": None if payload.get("detector_model_variant", None) is None else str(payload.get("detector_model_variant")),
        "detector_model_path": None if payload.get("detector_model_path", None) is None else str(payload.get("detector_model_path")),
        "detector_confidence_threshold": None if payload.get("detector_confidence_threshold", None) is None else float(payload.get("detector_confidence_threshold")),
        "detector_iou_threshold": None if payload.get("detector_iou_threshold", None) is None else float(payload.get("detector_iou_threshold")),
        "detector_image_size": None if payload.get("detector_image_size", None) is None else int(payload.get("detector_image_size")),
        "detector_device": None if payload.get("detector_device", None) is None else str(payload.get("detector_device")),
        "detector_backend": str(payload.get("detector_backend", "unknown")),
        "use_ground_truth_detections": bool(payload.get("use_ground_truth_detections", False)),
        "claim_confidence_threshold": None if claim_confidence_threshold is None else float(claim_confidence_threshold),
        "spawn_conditions": str(payload.get("spawn_conditions", "unknown")),
        "target_setup": str(payload.get("target_setup", ENVIRONMENT_ID)),
    }


def _normalize_target_distance_stats(payload: dict | None) -> dict:
    payload = payload or {}
    count = int(payload.get("count", 0) or 0)
    sum_m = float(payload.get("sum_m", 0.0) or 0.0)
    average_m = payload.get("average_m", None)
    min_m = payload.get("min_m", None)
    max_m = payload.get("max_m", None)
    return {
        "enabled": bool(payload.get("enabled", False)),
        "policy_supported": bool(payload.get("policy_supported", False)),
        "tracked": bool(payload.get("tracked", False)),
        "count": count,
        "sum_m": sum_m,
        "average_m": None if average_m is None else float(average_m),
        "min_m": None if min_m is None else float(min_m),
        "max_m": None if max_m is None else float(max_m),
    }


def _cxcy_from_xy(x: float, y: float) -> Optional[Cell]:
    min_x = -ARENA_X / 2.0
    min_y = -ARENA_Y / 2.0
    max_x = ARENA_X / 2.0
    max_y = ARENA_Y / 2.0
    if x < min_x or x >= max_x or y < min_y or y >= max_y:
        return None
    return (int((x - min_x) / CELL_SIZE_M), int((y - min_y) / CELL_SIZE_M))


def _cell_center_xy(cx: int, cy: int) -> Point:
    min_x = -ARENA_X / 2.0
    min_y = -ARENA_Y / 2.0
    return (min_x + (float(cx) + 0.5) * CELL_SIZE_M, min_y + (float(cy) + 0.5) * CELL_SIZE_M)


def _uniform_point_in_disc(center_x: float, center_y: float, radius: float) -> Point:
    theta = random.uniform(0.0, 2.0 * math.pi)
    r = radius * math.sqrt(random.random())
    return (center_x + r * math.cos(theta), center_y + r * math.sin(theta))


def _world_to_local(px: float, py: float, cx: float, cy: float, yaw: float) -> Point:
    dx = px - cx
    dy = py - cy
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * dx + s * dy, -s * dx + c * dy)


def _local_to_world(lx: float, ly: float, cx: float, cy: float, yaw: float) -> Point:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (cx + c * lx - s * ly, cy + s * lx + c * ly)


def _point_in_rotated_rect(px: float, py: float, rect: Dict[str, float], eps: float = 1e-9) -> bool:
    lx, ly = _world_to_local(px, py, rect["x"], rect["y"], rect["yaw"])
    return abs(lx) <= rect["length"] / 2.0 + eps and abs(ly) <= rect["width"] / 2.0 + eps


def _rotated_rect_corners(rect: Dict[str, float]) -> List[Point]:
    hx = rect["length"] / 2.0
    hy = rect["width"] / 2.0
    return [_local_to_world(lx, ly, rect["x"], rect["y"], rect["yaw"]) for lx, ly in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]]


def _rect_record(x: float, y: float, yaw: float, length: float, width: float) -> Dict[str, object]:
    rect: Dict[str, object] = {"x": float(x), "y": float(y), "yaw": float(yaw), "length": float(length), "width": float(width)}
    corners = _rotated_rect_corners(rect)  # type: ignore[arg-type]
    rect["corners_xy"] = [[float(px), float(py)] for px, py in corners]
    return rect


def _desk_fits_room(rect: Dict[str, float]) -> bool:
    min_x = -ARENA_X / 2.0 + DESK_WALL_MARGIN_M
    max_x = ARENA_X / 2.0 - DESK_WALL_MARGIN_M
    min_y = -ARENA_Y / 2.0 + DESK_WALL_MARGIN_M
    max_y = ARENA_Y / 2.0 - DESK_WALL_MARGIN_M
    return all(min_x <= x <= max_x and min_y <= y <= max_y for x, y in _rotated_rect_corners(rect))


def _target_within_floor_bounds(x: float, y: float) -> bool:
    return -ARENA_X / 2.0 + WALL_MARGIN <= x <= ARENA_X / 2.0 - WALL_MARGIN and -ARENA_Y / 2.0 + WALL_MARGIN <= y <= ARENA_Y / 2.0 - WALL_MARGIN


def _eligible_floor_cells() -> List[Cell]:
    cells: List[Cell] = []
    for cy in range(GRID_ROWS):
        for cx in range(GRID_COLS):
            x, y = _cell_center_xy(cx, cy)
            if _target_within_floor_bounds(x, y):
                cells.append((cx, cy))
    return cells


def _desk_spawn_rect(desk: Dict[str, object]) -> Dict[str, object]:
    return _rect_record(float(desk["x"]), float(desk["y"]), float(desk["yaw"]), DESK_LENGTH_M - 2.0 * DESK_SPAWN_EDGE_MARGIN_M, DESK_WIDTH_M - 2.0 * DESK_SPAWN_EDGE_MARGIN_M)


def _desk_floor_exclusion_rect(desk: Dict[str, object]) -> Dict[str, object]:
    spawn_rect = desk["spawn_rect"]
    return _rect_record(float(desk["x"]), float(desk["y"]), float(desk["yaw"]), float(spawn_rect["length"]) + 2.0 * DESK_FLOOR_EXCLUSION_MARGIN_M, float(spawn_rect["width"]) + 2.0 * DESK_FLOOR_EXCLUSION_MARGIN_M)


def _in_any_desk_spawn_zone(x: float, y: float, desks: List[Dict[str, object]]) -> bool:
    return any(_point_in_rotated_rect(x, y, desk["spawn_rect"]) for desk in desks)


def _in_any_desk_floor_exclusion_zone(x: float, y: float, desks: List[Dict[str, object]]) -> bool:
    return any(_point_in_rotated_rect(x, y, desk["floor_exclusion_rect"]) for desk in desks)


def _sample_drone_spawn_xy() -> Point:
    return (random.uniform(DRONE_SPAWN_X_MIN, DRONE_SPAWN_X_MAX), random.uniform(DRONE_SPAWN_Y_MIN, DRONE_SPAWN_Y_MAX))


def _sample_desks(drone_xy: Point, max_tries: int = 12000) -> List[Dict[str, object]]:
    desks: List[Dict[str, object]] = []
    for desk_id in range(N_TABLES):
        placed = None
        for _ in range(max_tries):
            yaw = random.uniform(-math.pi, math.pi)
            x = random.uniform(-ARENA_X / 2.0 + DESK_WALL_MARGIN_M, ARENA_X / 2.0 - DESK_WALL_MARGIN_M)
            y = random.uniform(-ARENA_Y / 2.0 + DESK_WALL_MARGIN_M, ARENA_Y / 2.0 - DESK_WALL_MARGIN_M)
            tabletop_rect = _rect_record(x, y, yaw, DESK_LENGTH_M, DESK_WIDTH_M)
            if not _desk_fits_room(tabletop_rect):
                continue
            if dist2((x, y), drone_xy) < DESK_DRONE_EXCLUSION_RADIUS_M ** 2:
                continue
            if any(dist2((x, y), (float(d["x"]), float(d["y"]))) < DESK_MIN_CENTER_DISTANCE_M ** 2 for d in desks):
                continue
            placed = {
                "desk_id": int(desk_id),
                "def": f"MEDIUM_EVAL_DESK_{desk_id:03d}",
                "class_id": CLASS_NAME_TO_ID[CLASS_DESK],
                "class_name": CLASS_DESK,
                "model": CLASS_DESK,
                "x": float(x),
                "y": float(y),
                "z": float(DESK_HEIGHT_M),
                "yaw": float(yaw),
                "dimensions_m": {"length": DESK_LENGTH_M, "width": DESK_WIDTH_M, "height": DESK_HEIGHT_M, "top_thickness": DESK_TOP_THICKNESS_M},
                "tabletop_rect": tabletop_rect,
                "is_empty": False,
                "forced_empty": False,
            }
            placed["spawn_rect"] = _desk_spawn_rect(placed)
            placed["floor_exclusion_rect"] = _desk_floor_exclusion_rect(placed)
            break
        if placed is None:
            raise RuntimeError(f"Could not place desk/table {desk_id} under current constraints.")
        desks.append(placed)
    return desks


def _valid_on_desk_cells(desk: Dict[str, object]) -> List[Cell]:
    rect = desk["spawn_rect"]
    cells: List[Cell] = []
    for cy in range(GRID_ROWS):
        for cx in range(GRID_COLS):
            x, y = _cell_center_xy(cx, cy)
            if _point_in_rotated_rect(x, y, rect):
                cells.append((cx, cy))
    return cells


def _sample_target_on_desk(desk: Dict[str, object], occupied_xy: List[Point], occupied_cells: set[Cell], max_tries: int = 2000) -> Optional[Tuple[float, float, int, int]]:
    cells = desk.get("valid_spawn_cells")
    if not cells:
        return None
    rect = desk["spawn_rect"]
    for _ in range(max_tries):
        cx, cy = random.choice(cells)
        if (cx, cy) in occupied_cells:
            continue
        center_x, center_y = _cell_center_xy(cx, cy)
        x, y = _uniform_point_in_disc(center_x, center_y, SPAWN_CENTER_DISC_RADIUS_M)
        if _cxcy_from_xy(x, y) != (cx, cy):
            continue
        if not _point_in_rotated_rect(x, y, rect):
            continue
        if any(dist2((x, y), p) < TARGET_MIN_SEPARATION_M ** 2 for p in occupied_xy):
            continue
        return (x, y, cx, cy)
    return None


def _sample_target_on_any_desk(desks: List[Dict[str, object]], occupied_xy: List[Point], occupied_cells: set[Cell], max_tries: int = 2500) -> Optional[Tuple[float, float, int, int, Dict[str, object]]]:
    eligible_desks = [d for d in desks if not bool(d.get("forced_empty", False))]
    if not eligible_desks:
        return None
    for _ in range(max_tries):
        desk = random.choice(eligible_desks)
        sampled = _sample_target_on_desk(desk, occupied_xy, occupied_cells, max_tries=1)
        if sampled is None:
            continue
        x, y, cx, cy = sampled
        return (x, y, cx, cy, desk)
    return None


def _sample_target_off_desk(*, floor_cells: List[Cell], desks: List[Dict[str, object]], drone_xy: Point, occupied_xy: List[Point], occupied_cells: set[Cell], max_tries: int = 5000) -> Optional[Tuple[float, float, int, int]]:
    for _ in range(max_tries):
        cx, cy = random.choice(floor_cells)
        if (cx, cy) in occupied_cells:
            continue
        center_x, center_y = _cell_center_xy(cx, cy)
        x, y = _uniform_point_in_disc(center_x, center_y, SPAWN_CENTER_DISC_RADIUS_M)
        if _cxcy_from_xy(x, y) != (cx, cy):
            continue
        if not _target_within_floor_bounds(x, y):
            continue
        if dist2((x, y), drone_xy) < DRONE_TARGET_EXCLUSION_RADIUS_M ** 2:
            continue
        if _in_any_desk_spawn_zone(x, y, desks):
            continue
        if _in_any_desk_floor_exclusion_zone(x, y, desks):
            continue
        if any(dist2((x, y), p) < TARGET_MIN_SEPARATION_M ** 2 for p in occupied_xy):
            continue
        return (x, y, cx, cy)
    return None


def _desk_node_string(desk: Dict[str, object], color_rgb) -> str:
    r, g, b = color_rgb
    tabletop_center_z = DESK_HEIGHT_M - DESK_TOP_THICKNESS_M / 2.0
    return f'''
DEF {desk["def"]} Solid {{
  translation {float(desk["x"]):.6f} {float(desk["y"]):.6f} {tabletop_center_z:.6f}
  rotation 0 0 1 {float(desk["yaw"]):.6f}
  model "desk"
  recognitionColors [
    {r:.6f} {g:.6f} {b:.6f}
  ]
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 1 1 1
        roughness 0.85
        metalness 0
        baseColorMap ImageTexture {{
          url [
            "{DESK_TABLETOP_TEXTURE_URL}"
          ]
        }}
      }}
      geometry Box {{
        size {DESK_LENGTH_M:.6f} {DESK_WIDTH_M:.6f} {DESK_TOP_THICKNESS_M:.6f}
      }}
    }}
  ]
  boundingObject Box {{
    size {DESK_LENGTH_M:.6f} {DESK_WIDTH_M:.6f} {DESK_TOP_THICKNESS_M:.6f}
  }}
}}
'''.strip()


def _desk_legs_node_string(desk: Dict[str, object]) -> str:
    leg_x = DESK_LENGTH_M / 2.0 - 0.12
    leg_y = DESK_WIDTH_M / 2.0 - 0.10
    leg_h = DESK_HEIGHT_M - DESK_TOP_THICKNESS_M
    leg_center_z = leg_h / 2.0
    legs = "\n".join(
        f'''
    Pose {{
      translation {sx * leg_x:.6f} {sy * leg_y:.6f} 0
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.18 0.18 0.18 roughness 0.7 metalness 0 }}
          geometry Box {{ size 0.055 0.055 {leg_h:.6f} }}
        }}
      ]
    }}'''.rstrip()
        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    )
    return f'''
DEF {desk["def"]}_LEGS Pose {{
  translation {float(desk["x"]):.6f} {float(desk["y"]):.6f} {leg_center_z:.6f}
  rotation 0 0 1 {float(desk["yaw"]):.6f}
  children [
{legs}
  ]
}}
'''.strip()


def _target_node_string(*, def_name: str, proto: str, model: str, x: float, y: float, z: float, color_rgb) -> str:
    r, g, b = color_rgb
    return f'''
DEF {def_name} {proto} {{
  translation {x:.6f} {y:.6f} {z:.6f}
  name "{def_name}"
  model "{model}"
  recognitionColors [
    {r:.6f} {g:.6f} {b:.6f}
  ]
}}
'''.strip()


def _write_spawned_objects_json(path: str, *, seed: int, objects: List[dict], table_scenario: dict) -> None:
    payload = {
        "seed": int(seed),
        "objects": objects,
        "class_name_to_id": CLASS_NAME_TO_ID,
        "target_class_names": sorted(TARGET_CLASS_NAMES),
        "table_scenario": table_scenario,
    }
    atomic_write_json(path, payload)


def _write_experiment_init_json(path: str, *, seed: int, drone_spawn_xy: Point, drone_spawn_yaw: float, table_scenario: dict):
    x, y = drone_spawn_xy
    cxcy = _cxcy_from_xy(float(x), float(y))
    payload = {
        "seed": int(seed),
        "environment": ENVIRONMENT_ID,
        "run_mode": RUN_MODE,
        "scenario": ENVIRONMENT_ID,
        "target_setup": ENVIRONMENT_ID,
        "drone_spawn": {
            "x": float(x),
            "y": float(y),
            "yaw": float(drone_spawn_yaw),
            "cx": int(cxcy[0]) if cxcy is not None else None,
            "cy": int(cxcy[1]) if cxcy is not None else None,
        },
        "table_scenario": table_scenario,
    }
    atomic_write_json(path, payload)


def _write_task_state(path: str, *, status: str, reason: str | None, seed: int, total_objects: int, remaining_objects: int, validated_found_objects: set[tuple[int, int, str]], timeout_s: float, warmup_s: float, episode_s: float, now_time_s: float, report_written: bool) -> None:
    validated_found_cells = {(cx, cy) for (cx, cy, _) in validated_found_objects}
    payload = {
        "status": str(status),
        "reason": reason,
        "seed": int(seed),
        "total_objects": int(total_objects),
        "remaining_objects": int(remaining_objects),
        "validated_found_cells": sorted([[int(c[0]), int(c[1])] for c in validated_found_cells]),
        "validated_found_objects": sorted(
            [
                {"cx": int(cx), "cy": int(cy), "class_name": str(class_name), "class_id": int(CLASS_NAME_TO_ID[class_name])}
                for (cx, cy, class_name) in validated_found_objects
            ],
            key=lambda d: (d["cy"], d["cx"], d["class_name"]),
        ),
        "timeout_s": float(timeout_s),
        "warmup_s": float(warmup_s),
        "episode_s": float(episode_s),
        "now_time_s": float(now_time_s),
        "report_written": bool(report_written),
    }
    atomic_write_json(path, payload)


def _empty_classification_stats() -> dict:
    return {
        "unique_claims_evaluated": 0,
        "correct_class_claims": 0,
        "wrong_class_claims": 0,
        "empty_cell_claims": 0,
        "object_cell_claims": 0,
        "object_cell_class_accuracy": 0.0,
        "overall_claim_precision": 0.0,
        "confusion_matrix": {},
    }


def _finalize_classification_stats(stats: dict) -> dict:
    object_cell_claims = int(stats["correct_class_claims"]) + int(stats["wrong_class_claims"])
    unique_claims = int(stats["unique_claims_evaluated"])
    correct_claims = int(stats["correct_class_claims"])
    stats["object_cell_claims"] = object_cell_claims
    stats["object_cell_class_accuracy"] = float(correct_claims) / float(object_cell_claims) if object_cell_claims > 0 else 0.0
    stats["overall_claim_precision"] = float(correct_claims) / float(unique_claims) if unique_claims > 0 else 0.0
    return stats


def _update_confusion_matrix(confusion_matrix: dict, true_name: str, pred_name: str) -> None:
    if true_name not in confusion_matrix:
        confusion_matrix[true_name] = {}
    confusion_matrix[true_name][pred_name] = int(confusion_matrix[true_name].get(pred_name, 0)) + 1


def _claim_key(claim: dict) -> tuple:
    pred_name = claim.get("class_name", None)
    if pred_name is None:
        pred_name = claim.get("label", None)
    return (int(claim.get("cx")), int(claim.get("cy")), str(pred_name) if pred_name is not None else "unknown")


def _extract_claims_from_progress(progress: dict) -> list[dict]:
    claims = []
    raw_objects = progress.get("claimed_found_objects", [])
    if isinstance(raw_objects, list):
        for obj in raw_objects:
            if not isinstance(obj, dict):
                continue
            if "cx" not in obj or "cy" not in obj:
                continue
            class_name = obj.get("class_name", None)
            if class_name is None:
                class_name = obj.get("label", None)
            # Desks are not task objects. Ignore them rather than count them as
            # wrong target claims.
            if class_name is not None and str(class_name) not in TARGET_CLASS_NAMES:
                continue
            claims.append(
                {
                    "cx": int(obj["cx"]),
                    "cy": int(obj["cy"]),
                    "class_id": None if obj.get("class_id", None) is None else int(obj["class_id"]),
                    "class_name": None if class_name is None else str(class_name),
                    "label": None if obj.get("label", None) is None else str(obj.get("label")),
                    "confidence": None if obj.get("confidence", None) is None else float(obj["confidence"]),
                }
            )
    return claims


def _evaluate_new_claims(*, claims: list[dict], seen_claim_keys: set, true_objects_by_cell: dict, validated_found_objects: set[tuple[int, int, str]], classification_stats: dict) -> set[tuple[int, int, str]]:
    new_validated = set(validated_found_objects)
    for claim in claims:
        key = _claim_key(claim)
        if key in seen_claim_keys:
            continue
        seen_claim_keys.add(key)
        classification_stats["unique_claims_evaluated"] = int(classification_stats["unique_claims_evaluated"]) + 1
        cell = (int(claim["cx"]), int(claim["cy"]))
        pred_class_id = claim.get("class_id", None)
        pred_class_name = claim.get("class_name", None) or claim.get("label", None)
        gt_obj = true_objects_by_cell.get(cell, None)
        if gt_obj is None:
            classification_stats["empty_cell_claims"] = int(classification_stats["empty_cell_claims"]) + 1
            continue
        gt_class_id = int(gt_obj["class_id"])
        gt_class_name = str(gt_obj["class_name"])
        is_correct = False
        if pred_class_name is not None:
            is_correct = str(pred_class_name) == gt_class_name
        elif pred_class_id is not None:
            is_correct = int(pred_class_id) == gt_class_id
        if is_correct:
            classification_stats["correct_class_claims"] = int(classification_stats["correct_class_claims"]) + 1
            _update_confusion_matrix(classification_stats["confusion_matrix"], gt_class_name, str(pred_class_name or gt_class_name))
            new_validated.add((int(cell[0]), int(cell[1]), gt_class_name))
        else:
            classification_stats["wrong_class_claims"] = int(classification_stats["wrong_class_claims"]) + 1
            _update_confusion_matrix(classification_stats["confusion_matrix"], gt_class_name, str(pred_class_name or (f"class_{pred_class_id}" if pred_class_id is not None else "unknown")))
    return new_validated


def _write_run_report(*, run_dir: str, policy_name: str, gt_seed: int, total_objects: int, validated_found_objects: set[tuple[int, int, str]], remaining_objects: int, autonomy_start_time_s: float | None, now_time_s: float, timeout_s: float, visited_cells: set[Cell], grid_rows: int, grid_cols: int, reason: str, classification_stats: dict, detector_limitations: dict, experiment_metadata: dict, target_distance_stats: dict, table_scenario: dict) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    report_dir = os.path.join(run_dir, REPORT_SUBDIR)
    os.makedirs(report_dir, exist_ok=True)
    found_count = int(len(validated_found_objects))
    elapsed = float(now_time_s - autonomy_start_time_s) if autonomy_start_time_s is not None else float(now_time_s)
    coverage_cells = int(len(visited_cells))
    total_cells = int(grid_rows * grid_cols)
    coverage_ratio = float(coverage_cells) / float(total_cells) if total_cells > 0 else 0.0
    cls_stats = _finalize_classification_stats(dict(classification_stats))
    payload = {
        "policy": str(policy_name),
        "gt_seed": int(gt_seed),
        "reason": str(reason),
        "detector_used": str(experiment_metadata.get("detector_used", "unknown")),
        "detector_model_family": experiment_metadata.get("detector_model_family", None),
        "detector_model_size": experiment_metadata.get("detector_model_size", None),
        "detector_model_variant": experiment_metadata.get("detector_model_variant", None),
        "detector_model_path": experiment_metadata.get("detector_model_path", None),
        "detector_confidence_threshold": experiment_metadata.get("detector_confidence_threshold", None),
        "detector_iou_threshold": experiment_metadata.get("detector_iou_threshold", None),
        "detector_image_size": experiment_metadata.get("detector_image_size", None),
        "detector_device": experiment_metadata.get("detector_device", None),
        "spawn_conditions": str(experiment_metadata.get("spawn_conditions", "unknown")),
        "environment": ENVIRONMENT_ID,

        "run_mode": RUN_MODE,

        "target_setup": str(experiment_metadata.get("target_setup", "medium")),
        "claim_confidence_threshold": experiment_metadata.get("claim_confidence_threshold", None),
        "target_distance_stats": dict(target_distance_stats),
        "autonomy_start_time_s": autonomy_start_time_s,
        "now_time_s": float(now_time_s),
        "elapsed_s": float(elapsed),
        "timeout_s": float(timeout_s),
        "total_objects": int(total_objects),
        "found_count": int(found_count),
        "remaining_objects": int(remaining_objects),
        "found_cells": sorted([[int(cx), int(cy)] for (cx, cy, _) in validated_found_objects]),
        "found_objects": sorted(
            [{"cx": int(cx), "cy": int(cy), "class_name": str(class_name), "class_id": int(CLASS_NAME_TO_ID[class_name])} for (cx, cy, class_name) in validated_found_objects],
            key=lambda d: (d["cy"], d["cx"], d["class_name"]),
        ),
        "coverage": {"visited_cells": int(coverage_cells), "total_cells": int(total_cells), "coverage_ratio": float(coverage_ratio)},
        "classification": cls_stats,
        "detector_limitations": dict(detector_limitations),
        "table_scenario": dict(table_scenario),
    }
    out_path = os.path.join(report_dir, "run_report.json")
    atomic_write_json(out_path, payload)
    print(f"[RUN REPORT] wrote {out_path}", flush=True)
    return payload


def _write_eval_report(eval_dir: str, policy_name: str, seeds: list[int], runs: list[dict]) -> str:
    report_dir = os.path.join(eval_dir, EVAL_REPORT_SUBDIR)
    os.makedirs(report_dir, exist_ok=True)
    n = len(runs)

    def _std(vals) -> float:
        return float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0

    if n == 0:
        payload = {"policy": policy_name, "num_runs": 0, "seeds": seeds}
        if INCLUDE_RUN_METRICS_IN_EVAL_REPORT:
            payload["run_metrics"] = []
        if INCLUDE_RUNS_IN_EVAL_REPORT:
            payload["runs"] = []
    else:
        elapsed_vals = [float(r["elapsed_s"]) for r in runs]
        found_vals = [int(r["found_count"]) for r in runs]
        remaining_vals = [int(r.get("remaining_objects", 0)) for r in runs]
        total_object_vals = [int(f + rem) for f, rem in zip(found_vals, remaining_vals)]
        coverage_vals = [float(r["coverage_ratio"]) for r in runs]
        success_vals = [1 if str(r["reason"]) == "done" else 0 for r in runs]
        total_found = int(sum(found_vals))
        total_target_objects = int(sum(total_object_vals))
        detection_rate_percent = 100.0 * float(total_found) / float(total_target_objects) if total_target_objects > 0 else 0.0
        total_unique_claims = sum(int(r["classification"]["unique_claims_evaluated"]) for r in runs)
        total_correct = sum(int(r["classification"]["correct_class_claims"]) for r in runs)
        total_wrong = sum(int(r["classification"]["wrong_class_claims"]) for r in runs)
        total_empty = sum(int(r["classification"]["empty_cell_claims"]) for r in runs)
        total_object_cell_claims = sum(int(r["classification"]["object_cell_claims"]) for r in runs)
        target_stats = [_normalize_target_distance_stats(r.get("target_distance_stats", {})) for r in runs]
        target_distance_total_count = sum(int(s["count"]) for s in target_stats)
        target_distance_total_sum_m = sum(float(s["sum_m"]) for s in target_stats)
        target_distance_average_m = float(target_distance_total_sum_m) / float(target_distance_total_count) if target_distance_total_count > 0 else None
        target_distance_mins = [float(s["min_m"]) for s in target_stats if s.get("min_m", None) is not None]
        target_distance_maxs = [float(s["max_m"]) for s in target_stats if s.get("max_m", None) is not None]
        target_distance_summary = {
            "enabled": any(bool(s["enabled"]) for s in target_stats),
            "policy_supported": any(bool(s["policy_supported"]) for s in target_stats),
            "tracked": any(bool(s["tracked"]) for s in target_stats),
            "total_target_selections": int(target_distance_total_count),
            "total_target_distance_m": float(target_distance_total_sum_m),
            "average_target_distance_m": target_distance_average_m,
            "min_target_distance_m": min(target_distance_mins) if target_distance_mins else None,
            "max_target_distance_m": max(target_distance_maxs) if target_distance_maxs else None,
        }
        run_metrics = []
        for r, total_objects_for_run in zip(runs, total_object_vals):
            cls = r.get("classification", {}) or {}
            td_stats = _normalize_target_distance_stats(r.get("target_distance_stats", {}))
            found_count = int(r.get("found_count", 0))
            run_detection_rate_percent = 100.0 * float(found_count) / float(total_objects_for_run) if total_objects_for_run > 0 else 0.0
            table_info = r.get("table_scenario", {}) or {}
            run_metrics.append(
                {
                    "run_index": int(r.get("run_index", len(run_metrics))),
                    "seed": int(r.get("seed", -1)),
                    "reason": str(r.get("reason", "unknown")),
                    "elapsed_s": float(r.get("elapsed_s", 0.0)),
                    "found_count": found_count,
                    "total_objects": int(total_objects_for_run),
                    "detection_rate_percent": float(run_detection_rate_percent),
                    "coverage_ratio": float(r.get("coverage_ratio", 0.0)),
                    "target_distance_average_m": td_stats.get("average_m", None),
                    "overall_claim_precision": float(cls.get("overall_claim_precision", 0.0)),
                    "object_cell_class_accuracy": float(cls.get("object_cell_class_accuracy", 0.0)),
                    "targets_on_table": int((table_info.get("target_spawn_counts", {}) or {}).get("on_table", 0)),
                    "targets_off_table": int((table_info.get("target_spawn_counts", {}) or {}).get("off_table", 0)),
                    "num_empty_tables": int(len(table_info.get("empty_table_ids", []) or [])),
                    "forced_empty_table_id": table_info.get("forced_empty_table_id", None),
                }
            )
        detector_limitations = dict(runs[0].get("detector_limitations", {})) if runs else {}
        first_run = runs[0] if runs else {}
        table_runs = [r.get("table_scenario", {}) or {} for r in runs]
        forced_empty_runs = sum(1 for t in table_runs if t.get("forced_empty_table_id", None) is not None)
        payload = {
            "policy": policy_name,
            "num_runs": n,
            "seeds": seeds,
            "detector_used": str(first_run.get("detector_used", "unknown")),
            "detector_model_family": first_run.get("detector_model_family", None),
            "detector_model_size": first_run.get("detector_model_size", None),
            "detector_model_variant": first_run.get("detector_model_variant", None),
            "detector_model_path": first_run.get("detector_model_path", None),
            "detector_confidence_threshold": first_run.get("detector_confidence_threshold", None),
            "detector_iou_threshold": first_run.get("detector_iou_threshold", None),
            "detector_image_size": first_run.get("detector_image_size", None),
            "detector_device": first_run.get("detector_device", None),
            "spawn_conditions": str(first_run.get("spawn_conditions", "unknown")),
            "environment": ENVIRONMENT_ID,

            "run_mode": RUN_MODE,

            "target_setup": str(first_run.get("target_setup", "medium")),
            "claim_confidence_threshold": first_run.get("claim_confidence_threshold", first_run.get("found_det_threshold", None)),
            "detector_limitations": detector_limitations,
            "target_distance_summary": target_distance_summary,
            "table_scenario_summary": {
                "num_tables": int(N_TABLES),
                "p_target_on_table": float(P_TARGET_ON_TABLE),
                "force_one_empty_table_num_runs": int(FORCE_ONE_EMPTY_TABLE_NUM_RUNS),
                "forced_empty_runs": int(forced_empty_runs),
                "total_targets_on_table": int(sum(int((t.get("target_spawn_counts", {}) or {}).get("on_table", 0)) for t in table_runs)),
                "total_targets_off_table": int(sum(int((t.get("target_spawn_counts", {}) or {}).get("off_table", 0)) for t in table_runs)),
            },
            "summary": {
                "success_rate": float(sum(success_vals) / n),
                "average_simulation_time_s": float(sum(elapsed_vals) / n),
                "std_simulation_time_s": _std(elapsed_vals),
                "average_found_count": float(sum(found_vals) / n),
                "std_found_count": _std(found_vals),
                "detection_rate_percent": float(detection_rate_percent),
                "average_coverage_ratio": float(sum(coverage_vals) / n),
                "std_coverage_ratio": _std(coverage_vals),
                "min_simulation_time_s": float(min(elapsed_vals)),
                "max_simulation_time_s": float(max(elapsed_vals)),
                "min_found_count": int(min(found_vals)),
                "max_found_count": int(max(found_vals)),
                "min_coverage_ratio": float(min(coverage_vals)),
                "max_coverage_ratio": float(max(coverage_vals)),
            },
            "classification_summary": {
                "total_unique_claims_evaluated": int(total_unique_claims),
                "total_correct_class_claims": int(total_correct),
                "total_wrong_class_claims": int(total_wrong),
                "total_empty_cell_claims": int(total_empty),
                "total_object_cell_claims": int(total_object_cell_claims),
                "global_overall_claim_precision": float(total_correct) / float(total_unique_claims) if total_unique_claims > 0 else 0.0,
                "global_object_cell_class_accuracy": float(total_correct) / float(total_object_cell_claims) if total_object_cell_claims > 0 else 0.0,
            },
        }
        if INCLUDE_RUN_METRICS_IN_EVAL_REPORT:
            payload["run_metrics"] = run_metrics
        if INCLUDE_RUNS_IN_EVAL_REPORT:
            payload["runs"] = runs
    out_path = os.path.join(report_dir, "eval_report.json")
    atomic_write_json(out_path, payload)
    print(f"[EVAL REPORT] wrote {out_path}", flush=True)
    return out_path


def _init_or_resume_eval_state(
    eval_state_path: str,
    policy_name: str,
    seeds: list[int],
    *,
    detector_id: str,
    detector_signature: dict,
) -> dict:
    eval_state = read_json(eval_state_path, None)

    if isinstance(eval_state, dict) and not bool(eval_state.get("completed", False)):
        state_environment = str(
            eval_state.get("environment", eval_state.get("scenario", ""))
        )
        environment_matches = state_environment == ENVIRONMENT_ID
        policy_matches = str(eval_state.get("policy_name", "")) == str(policy_name)
        detector_matches = str(eval_state.get("detector_id", "")) == str(detector_id)
        detector_signature_matches = dict(eval_state.get("detector_signature", {}) or {}) == dict(detector_signature)
        seeds_match = [int(v) for v in eval_state.get("seeds", [])] == [int(v) for v in seeds]

        if (
            environment_matches
            and policy_matches
            and detector_matches
            and detector_signature_matches
            and seeds_match
        ):
            return eval_state

        print(
            "[MEDIUM EVAL] existing incomplete eval_state does not match the requested "
            "environment/policy/detector/seeds; starting a new evaluation.",
            flush=True,
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(_controller_logs_dir(), f"eval_{policy_name}_medium_{ts}")

    eval_state = {
        "policy_name": policy_name,
        "detector_id": str(detector_id),
        "detector_signature": dict(detector_signature),
        "seeds": list(seeds),
        "total_runs": len(seeds),
        "current_run_index": 0,
        "current_seed": int(seeds[0]),
        "eval_dir": eval_dir,
        "aggregate_runs": [],
        "completed": False,
        "scenario": ENVIRONMENT_ID,
        "environment": ENVIRONMENT_ID,
        "run_mode": RUN_MODE,
    }
    atomic_write_json(eval_state_path, eval_state)
    return eval_state

def _place_drone(supervisor: Supervisor, drone_node, *, x: float, y: float, yaw: float) -> None:
    if drone_node is None:
        raise RuntimeError(f"Drone DEF '{DRONE_DEF}' not found. Set DRONE_DEF to the drone DEF used in your world.")
    translation_field = drone_node.getField("translation")
    rotation_field = drone_node.getField("rotation")
    current_translation = translation_field.getSFVec3f()
    z = float(current_translation[2])
    translation_field.setSFVec3f([float(x), float(y), z])
    rotation_field.setSFRotation([0.0, 0.0, 1.0, float(yaw)])
    drone_node.resetPhysics()
    supervisor.simulationResetPhysics()


def _cleanup_spawned_nodes(spawn_children) -> None:
    for idx in range(spawn_children.getCount() - 1, -1, -1):
        spawn_children.getMFNode(idx).remove()


def _spawn_scene(supervisor: Supervisor, spawn_children, *, seed: int, run_index: int, drone_xy: Point) -> tuple[List[dict], List[dict], List[dict], dict]:
    desks = _sample_desks(drone_xy)
    force_empty_this_run = (run_index < int(FORCE_ONE_EMPTY_TABLE_NUM_RUNS)) and len(desks) > 0
    forced_empty_table_id = None
    if force_empty_this_run:
        forced_empty_table_id = int(random.choice(desks)["desk_id"])
        for desk in desks:
            desk["forced_empty"] = int(desk["desk_id"]) == forced_empty_table_id

    spawned_objects: List[dict] = []
    target_records: List[dict] = []
    global_index = 0

    for desk in desks:
        desk["valid_spawn_cells"] = _valid_on_desk_cells(desk)
        if not desk["valid_spawn_cells"] and not bool(desk.get("forced_empty", False)):
            raise RuntimeError(f"Desk/table {desk['desk_id']} has no valid target spawn cells.")
        color_rgb = _instance_color(global_index)
        spawn_children.importMFNodeFromString(-1, _desk_node_string(desk, color_rgb))
        desk_node = spawn_children.getMFNode(spawn_children.getCount() - 1)
        desk["id"] = _node_id(desk_node)
        spawn_children.importMFNodeFromString(-1, _desk_legs_node_string(desk))
        desk["recognition_color_rgb"] = [float(v) for v in color_rgb]
        global_index += 1

    floor_cells = _eligible_floor_cells()
    if not floor_cells:
        raise RuntimeError("No eligible floor cells available under current wall margin.")

    occupied_xy: List[Point] = []
    occupied_target_cells: set[Cell] = set()

    for spec in OBJECT_SPECS:
        for local_i in range(int(spec["count"])):
            prefer_table = random.random() < P_TARGET_ON_TABLE
            sampled = None
            spawn_mode = "on_table" if prefer_table else "off_table"
            desk_for_target = None
            z = 0.0
            if prefer_table:
                sampled_on = _sample_target_on_any_desk(desks, occupied_xy, occupied_target_cells)
                if sampled_on is not None:
                    x, y, cx, cy, desk_for_target = sampled_on
                    sampled = (x, y, cx, cy)
                    z = DESK_HEIGHT_M
                else:
                    spawn_mode = "off_table"
            if sampled is None:
                sampled_off = _sample_target_off_desk(floor_cells=floor_cells, desks=desks, drone_xy=drone_xy, occupied_xy=occupied_xy, occupied_cells=occupied_target_cells)
                if sampled_off is not None:
                    x, y, cx, cy = sampled_off
                    sampled = (x, y, cx, cy)
                    desk_for_target = None
                    z = 0.0
                    spawn_mode = "off_table"
            if sampled is None:
                print(f"WARNING: Could not place {spec['class_name']} #{local_i} under constraints.", flush=True)
                continue
            x, y, cx, cy = sampled
            occupied_xy.append((float(x), float(y)))
            occupied_target_cells.add((int(cx), int(cy)))
            color_rgb = _instance_color(global_index)
            def_name = f"{spec['def_prefix']}_{global_index:03d}"
            node_string = _target_node_string(def_name=def_name, proto=str(spec["proto"]), model=str(spec["model"]), x=float(x), y=float(y), z=float(z), color_rgb=color_rgb)
            spawn_children.importMFNodeFromString(-1, node_string)
            target_node = spawn_children.getMFNode(spawn_children.getCount() - 1)
            target_record = {
                "def": def_name,
                "id": _node_id(target_node),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "cx": int(cx),
                "cy": int(cy),
                "class_id": int(spec["class_id"]),
                "class_name": str(spec["class_name"]),
                "proto": str(spec["proto"]),
                "model": str(spec["model"]),
                "recognition_color_rgb": [float(v) for v in color_rgb],
                "spawn_mode": spawn_mode,
                "desk_id": None if desk_for_target is None else int(desk_for_target["desk_id"]),
                "grid_cell_center_xy": [float(v) for v in _cell_center_xy(cx, cy)],
                "spawn_center_disc_radius_m": float(SPAWN_CENTER_DISC_RADIUS_M),
            }
            target_records.append(target_record)
            spawned_objects.append(target_record)
            global_index += 1

    target_desk_ids = {int(t["desk_id"]) for t in target_records if t.get("desk_id", None) is not None}
    for desk in desks:
        desk["is_empty"] = int(desk["desk_id"]) not in target_desk_ids
        desk_export = dict(desk)
        desk_export.pop("valid_spawn_cells", None)
        spawned_objects.append(desk_export)

    target_counts = {name: sum(1 for r in target_records if r["class_name"] == name) for name in (CLASS_BOTTLE, CLASS_CAN)}
    on_table_count = sum(1 for t in target_records if t["spawn_mode"] == "on_table")
    off_table_count = sum(1 for t in target_records if t["spawn_mode"] == "off_table")
    empty_table_ids = sorted([int(d["desk_id"]) for d in desks if bool(d.get("is_empty", False))])
    table_scenario = {
        "scenario": ENVIRONMENT_ID,
        "environment": ENVIRONMENT_ID,
        "run_mode": RUN_MODE,
        "num_tables": int(len(desks)),
        "num_targets": int(len(target_records)),
        "num_bottles": int(target_counts[CLASS_BOTTLE]),
        "num_cans": int(target_counts[CLASS_CAN]),
        "p_target_on_table": float(P_TARGET_ON_TABLE),
        "target_spawn_counts": {"on_table": int(on_table_count), "off_table": int(off_table_count)},
        "empty_table_ids": empty_table_ids,
        "forced_empty_table_id": forced_empty_table_id,
        "force_empty_this_run": bool(force_empty_this_run),
        "desk_spawn_edge_margin_m": float(DESK_SPAWN_EDGE_MARGIN_M),
        "desk_floor_exclusion_margin_m": float(DESK_FLOOR_EXCLUSION_MARGIN_M),
        "spawn_center_disc_radius_m": float(SPAWN_CENTER_DISC_RADIUS_M),
    }
    return spawned_objects, target_records, desks, table_scenario


def main():
    if not (0.0 < SPAWN_CENTER_DISC_RADIUS_M < 0.5 * CELL_SIZE_M):
        raise ValueError(f"SPAWN_CENTER_DISC_RADIUS_M must be in (0, {0.5 * CELL_SIZE_M:.3f}), got {SPAWN_CENTER_DISC_RADIUS_M}")
    if not (0.0 <= P_TARGET_ON_TABLE <= 1.0):
        raise ValueError(f"P_TARGET_ON_TABLE must be in [0, 1], got {P_TARGET_ON_TABLE}")

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    policy_name = normalize_policy_id(os.environ.get("AIF_POLICY", DEFAULT_POLICY))

    spawned_objects_path = SHARED_STATE.path(SPAWNED_OBJECTS_FILENAME)
    experiment_init_path = SHARED_STATE.path(EXPERIMENT_INIT_FILENAME)
    task_state_path = SHARED_STATE.path(TASK_STATE_FILENAME)
    controller_progress_path = SHARED_STATE.path(CONTROLLER_PROGRESS_FILENAME)
    eval_state_path = SHARED_STATE.path(EVAL_STATE_FILENAME)

    SHARED_STATE.clear(
        CONTROLLER_READY_FILENAME,
        IPC_SESSION_FILENAME,
        SPAWNED_OBJECTS_FILENAME,
        EXPERIMENT_INIT_FILENAME,
        TASK_STATE_FILENAME,
        CONTROLLER_PROGRESS_FILENAME,
    )

    print("[MEDIUM EVAL] shared state cleared", flush=True)

    establish_supervisor_session(
        SHARED_STATE,
        environment=ENVIRONMENT_ID,
        run_mode=RUN_MODE,
    )
    print("[IPC] supervisor session established", flush=True)

    configured_seeds = _configured_eval_seeds()
    eval_state = _init_or_resume_eval_state(
        eval_state_path,
        policy_name,
        configured_seeds,
        detector_id=CONFIGURED_DETECTOR,
        detector_signature=CONFIGURED_DETECTOR_SIGNATURE,
    )
    seeds = [int(s) for s in eval_state["seeds"]]
    total_runs = int(eval_state["total_runs"])
    run_index = int(eval_state["current_run_index"])
    current_seed = int(eval_state["current_seed"])
    eval_dir = str(eval_state["eval_dir"])
    aggregate_runs = list(eval_state.get("aggregate_runs", []))

    random.seed(current_seed)

    _write_task_state(task_state_path, status="running", reason=None, seed=current_seed, total_objects=0, remaining_objects=0, validated_found_objects=set(), timeout_s=TOTAL_SECONDS, warmup_s=WARMUP_SECONDS, episode_s=EPISODE_SECONDS, now_time_s=0.0, report_written=False)

    atomic_write_json(
        controller_progress_path,
        {
            "policy_name": policy_name,
            "run_dir": None,
            "autonomy_start_time_s": None,
            "last_controller_time_s": 0.0,
            "visited_cells": [],
            "claimed_found_cells": [],
            "claimed_found_objects": [],
            "detector_limitations": {"mode": "unknown", "frames_interval": None, "time_interval_s": None},
            "drone_spawn": None,
        },
    )

    print(f"[MEDIUM EVAL] initialized run {run_index + 1}/{total_runs} policy={policy_name} seed={current_seed}", flush=True)

    spawn_root = supervisor.getFromDef("SPAWN_ROOT")
    if spawn_root is None:
        print("ERROR: DEF SPAWN_ROOT Group not found in the world.", flush=True)
        return
    spawn_children = spawn_root.getField("children")

    drone_node = supervisor.getFromDef(DRONE_DEF)
    drone_xy = _sample_drone_spawn_xy()
    _place_drone(supervisor, drone_node, x=drone_xy[0], y=drone_xy[1], yaw=DRONE_SPAWN_YAW)

    spawned_objects, target_records, desks, table_scenario = _spawn_scene(supervisor, spawn_children, seed=current_seed, run_index=run_index, drone_xy=drone_xy)

    print(
        f"[MEDIUM EVAL] run {run_index + 1}/{total_runs} policy={policy_name} seed={current_seed} "
        f"drone_spawn=({drone_xy[0]:.3f},{drone_xy[1]:.3f}) tables={len(desks)} "
        f"targets={len(target_records)} on_table={table_scenario['target_spawn_counts']['on_table']} "
        f"off_table={table_scenario['target_spawn_counts']['off_table']} empty_tables={table_scenario['empty_table_ids']} "
        f"forced_empty={table_scenario['forced_empty_table_id']}",
        flush=True,
    )

    _write_experiment_init_json(experiment_init_path, seed=current_seed, drone_spawn_xy=drone_xy, drone_spawn_yaw=DRONE_SPAWN_YAW, table_scenario=table_scenario)
    print(f"Experiment init JSON exported: {experiment_init_path}", flush=True)

    _write_spawned_objects_json(spawned_objects_path, seed=current_seed, objects=spawned_objects, table_scenario=table_scenario)
    print(f"GT JSON exported: {spawned_objects_path}", flush=True)

    true_objects_by_cell: Dict[Cell, dict] = {}
    for obj in target_records:
        if obj.get("cx", None) is None or obj.get("cy", None) is None:
            continue
        true_objects_by_cell[(int(obj["cx"]), int(obj["cy"]))] = obj

    total_objects = len(true_objects_by_cell)
    validated_found_objects: set[tuple[int, int, str]] = set()
    classification_stats = _empty_classification_stats()
    seen_claim_keys = set()
    report_written = False
    final_reason = None
    final_time_s = None

    _write_task_state(task_state_path, status="running", reason=None, seed=current_seed, total_objects=total_objects, remaining_objects=total_objects, validated_found_objects=validated_found_objects, timeout_s=TOTAL_SECONDS, warmup_s=WARMUP_SECONDS, episode_s=EPISODE_SECONDS, now_time_s=0.0, report_written=False)

    spawn_cxcy = _cxcy_from_xy(drone_xy[0], drone_xy[1])
    atomic_write_json(
        controller_progress_path,
        {
            "policy_name": None,
            "run_dir": None,
            "autonomy_start_time_s": None,
            "last_controller_time_s": 0.0,
            "visited_cells": [],
            "claimed_found_cells": [],
            "claimed_found_objects": [],
            "detector_limitations": {"mode": "unknown", "frames_interval": None, "time_interval_s": None},
            "drone_spawn": {"x": float(drone_xy[0]), "y": float(drone_xy[1]), "yaw": float(DRONE_SPAWN_YAW), "cx": spawn_cxcy[0] if spawn_cxcy is not None else None, "cy": spawn_cxcy[1] if spawn_cxcy is not None else None},
        },
    )

    start_time = supervisor.getTime()
    while supervisor.step(timestep) != -1:
        t = supervisor.getTime()
        elapsed_total = float(t - start_time)
        controller_progress = read_json(
            controller_progress_path,
            {
                "policy_name": None,
                "run_dir": None,
                "autonomy_start_time_s": None,
                "last_controller_time_s": 0.0,
                "visited_cells": [],
                "claimed_found_cells": [],
                "claimed_found_objects": [],
                "detector_limitations": {"mode": "unknown", "frames_interval": None, "time_interval_s": None},
            },
        )
        detector_limitations = _normalize_detector_limitations(controller_progress.get("detector_limitations", {}))
        experiment_metadata = _normalize_experiment_metadata(controller_progress.get("experiment_metadata", {}))
        if experiment_metadata.get("target_setup") in (None, "unknown"):
            experiment_metadata["target_setup"] = "medium"
        target_distance_stats = _normalize_target_distance_stats(controller_progress.get("target_distance_stats", {}))

        claims = _extract_claims_from_progress(controller_progress)
        validated_found_objects = _evaluate_new_claims(claims=claims, seen_claim_keys=seen_claim_keys, true_objects_by_cell=true_objects_by_cell, validated_found_objects=validated_found_objects, classification_stats=classification_stats)
        remaining_objects = max(0, int(total_objects) - int(len(validated_found_objects)))

        if len(validated_found_objects) >= total_objects and total_objects > 0:
            final_reason = "done"
            final_time_s = float(t)
        elif elapsed_total >= TOTAL_SECONDS:
            final_reason = "timeout"
            final_time_s = float(t)

        if final_reason is None:
            _write_task_state(task_state_path, status="running", reason=None, seed=current_seed, total_objects=total_objects, remaining_objects=remaining_objects, validated_found_objects=validated_found_objects, timeout_s=TOTAL_SECONDS, warmup_s=WARMUP_SECONDS, episode_s=EPISODE_SECONDS, now_time_s=float(t), report_written=report_written)
            continue

        if not report_written:
            _write_task_state(task_state_path, status=final_reason, reason=final_reason, seed=current_seed, total_objects=total_objects, remaining_objects=remaining_objects, validated_found_objects=validated_found_objects, timeout_s=TOTAL_SECONDS, warmup_s=WARMUP_SECONDS, episode_s=EPISODE_SECONDS, now_time_s=float(final_time_s), report_written=False)
            run_dir = controller_progress.get("run_dir", None)
            autonomy_start_time_s = controller_progress.get("autonomy_start_time_s", None)
            visited_cells = {(int(cx), int(cy)) for (cx, cy) in controller_progress.get("visited_cells", [])}

            if run_dir:
                run_payload = _write_run_report(run_dir=run_dir, policy_name=policy_name, gt_seed=current_seed, total_objects=total_objects, validated_found_objects=validated_found_objects, remaining_objects=remaining_objects, autonomy_start_time_s=autonomy_start_time_s, now_time_s=float(final_time_s), timeout_s=TOTAL_SECONDS, visited_cells=visited_cells, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, reason=final_reason, classification_stats=classification_stats, detector_limitations=detector_limitations, experiment_metadata=experiment_metadata, target_distance_stats=target_distance_stats, table_scenario=table_scenario)
            else:
                print("[RUN REPORT] run_dir not available yet, using in-memory payload", flush=True)
                coverage_cells = int(len(visited_cells))
                total_cells = int(GRID_ROWS * GRID_COLS)
                coverage_ratio = float(coverage_cells) / float(total_cells) if total_cells > 0 else 0.0
                cls_stats = _finalize_classification_stats(dict(classification_stats))
                run_payload = {
                    "policy": policy_name,
                    "gt_seed": current_seed,
                    "reason": final_reason,
                    "detector_used": str(experiment_metadata.get("detector_used", "unknown")),
                    "detector_model_family": experiment_metadata.get("detector_model_family", None),
                    "detector_model_size": experiment_metadata.get("detector_model_size", None),
                    "detector_model_variant": experiment_metadata.get("detector_model_variant", None),
                    "detector_model_path": experiment_metadata.get("detector_model_path", None),
                    "detector_confidence_threshold": experiment_metadata.get("detector_confidence_threshold", None),
                    "detector_iou_threshold": experiment_metadata.get("detector_iou_threshold", None),
                    "detector_image_size": experiment_metadata.get("detector_image_size", None),
                    "detector_device": experiment_metadata.get("detector_device", None),
                    "spawn_conditions": str(experiment_metadata.get("spawn_conditions", "unknown")),
                    "environment": ENVIRONMENT_ID,

                    "run_mode": RUN_MODE,

                    "target_setup": str(experiment_metadata.get("target_setup", "medium")),
                    "claim_confidence_threshold": experiment_metadata.get("claim_confidence_threshold", None),
                    "target_distance_stats": dict(target_distance_stats),
                    "autonomy_start_time_s": autonomy_start_time_s,
                    "now_time_s": float(final_time_s),
                    "elapsed_s": float(final_time_s - autonomy_start_time_s) if autonomy_start_time_s is not None else float(final_time_s),
                    "timeout_s": TOTAL_SECONDS,
                    "total_objects": total_objects,
                    "found_count": len(validated_found_objects),
                    "remaining_objects": remaining_objects,
                    "coverage": {"visited_cells": coverage_cells, "total_cells": total_cells, "coverage_ratio": coverage_ratio},
                    "classification": cls_stats,
                    "detector_limitations": dict(detector_limitations),
                    "table_scenario": dict(table_scenario),
                }

            aggregate_runs.append(
                {
                    "run_index": run_index,
                    "seed": current_seed,
                    "policy": policy_name,
                    "reason": str(run_payload["reason"]),
                    "detector_used": str(run_payload.get("detector_used", "unknown")),
                    "detector_model_family": run_payload.get("detector_model_family", None),
                    "detector_model_size": run_payload.get("detector_model_size", None),
                    "detector_model_variant": run_payload.get("detector_model_variant", None),
                    "detector_model_path": run_payload.get("detector_model_path", None),
                    "detector_confidence_threshold": run_payload.get("detector_confidence_threshold", None),
                    "detector_iou_threshold": run_payload.get("detector_iou_threshold", None),
                    "detector_image_size": run_payload.get("detector_image_size", None),
                    "detector_device": run_payload.get("detector_device", None),
                    "spawn_conditions": str(run_payload.get("spawn_conditions", "unknown")),
                    "environment": ENVIRONMENT_ID,

                    "run_mode": RUN_MODE,

                    "target_setup": str(run_payload.get("target_setup", "medium")),
                    "claim_confidence_threshold": run_payload.get("claim_confidence_threshold", run_payload.get("found_det_threshold", None)),
                    "target_distance_stats": dict(run_payload.get("target_distance_stats", {})),
                    "elapsed_s": float(run_payload["elapsed_s"]),
                    "found_count": int(run_payload["found_count"]),
                    "remaining_objects": int(run_payload["remaining_objects"]),
                    "coverage_ratio": float(run_payload["coverage"]["coverage_ratio"]),
                    "classification": dict(run_payload["classification"]),
                    "detector_limitations": dict(run_payload["detector_limitations"]),
                    "table_scenario": dict(run_payload.get("table_scenario", table_scenario)),
                    "run_dir": run_dir,
                }
            )
            report_written = True
            is_last_run = (run_index + 1) >= total_runs
            if is_last_run:
                _write_eval_report(eval_dir=eval_dir, policy_name=policy_name, seeds=seeds, runs=aggregate_runs)
                eval_state = {"policy_name": policy_name, "detector_id": str(CONFIGURED_DETECTOR), "detector_signature": dict(CONFIGURED_DETECTOR_SIGNATURE), "seeds": seeds, "total_runs": total_runs, "current_run_index": run_index, "current_seed": current_seed, "eval_dir": eval_dir, "aggregate_runs": aggregate_runs, "completed": True, "scenario": ENVIRONMENT_ID, "environment": ENVIRONMENT_ID, "run_mode": RUN_MODE}
                atomic_write_json(eval_state_path, eval_state)
                _write_task_state(task_state_path, status=final_reason, reason=final_reason, seed=current_seed, total_objects=total_objects, remaining_objects=remaining_objects, validated_found_objects=validated_found_objects, timeout_s=TOTAL_SECONDS, warmup_s=WARMUP_SECONDS, episode_s=EPISODE_SECONDS, now_time_s=float(final_time_s), report_written=True)
                print(f"[MEDIUM EVAL] completed all {total_runs} runs", flush=True)
                grace_start = supervisor.getTime()
                while supervisor.step(timestep) != -1:
                    if (supervisor.getTime() - grace_start) >= FINALIZATION_GRACE_SECONDS:
                        break
                _cleanup_spawned_nodes(spawn_children)
                print("Cleanup complete. Quitting simulation.", flush=True)
                supervisor.simulationQuit(0)
                return
            next_run_index = run_index + 1
            next_seed = int(seeds[next_run_index])
            eval_state = {"policy_name": policy_name, "detector_id": str(CONFIGURED_DETECTOR), "detector_signature": dict(CONFIGURED_DETECTOR_SIGNATURE), "seeds": seeds, "total_runs": total_runs, "current_run_index": next_run_index, "current_seed": next_seed, "eval_dir": eval_dir, "aggregate_runs": aggregate_runs, "completed": False, "scenario": ENVIRONMENT_ID, "environment": ENVIRONMENT_ID, "run_mode": RUN_MODE}
            atomic_write_json(eval_state_path, eval_state)
            _write_task_state(task_state_path, status=final_reason, reason=final_reason, seed=current_seed, total_objects=total_objects, remaining_objects=remaining_objects, validated_found_objects=validated_found_objects, timeout_s=TOTAL_SECONDS, warmup_s=WARMUP_SECONDS, episode_s=EPISODE_SECONDS, now_time_s=float(final_time_s), report_written=True)
            print(f"[MEDIUM EVAL] finished run {run_index + 1}/{total_runs}; reloading world for next seed={next_seed}", flush=True)
            grace_start = supervisor.getTime()
            while supervisor.step(timestep) != -1:
                if (supervisor.getTime() - grace_start) >= FINALIZATION_GRACE_SECONDS:
                    break
            _cleanup_spawned_nodes(spawn_children)
            supervisor.worldReload()
            return


if __name__ == "__main__":
    main()
