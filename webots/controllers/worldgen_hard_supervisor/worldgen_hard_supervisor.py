# webots/controllers/worldgen_hard_supervisor/worldgen_hard_supervisor.py

from __future__ import annotations

import colorsys
import math
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from controller import Supervisor

ENVIRONMENT_ID = "hard"
RUN_MODE = "single"

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

from crazyflie_AIF.config import AIF_POLICY as DEFAULT_POLICY, normalize_policy_id

# -----------------------------------------------------------------------------
# Single-run supervisor for the hard scenario
# -----------------------------------------------------------------------------
# Keeps the mixed-table shared-file/reporting lifecycle while spawning the
# hard scenario: the same task targets and desks plus non-target visual
# distractors. Desks are exported for GT/surface-height discovery but are not
# task objects to be found; distractors are exported for traceability only.

# ---- Single-run settings ----
SEED = int(os.environ.get("HARD_SEED", "63"))

WARMUP_SECONDS = float(os.environ.get("HARD_WARMUP_SECONDS", "15.0"))
EPISODE_SECONDS = float(os.environ.get("HARD_EPISODE_SECONDS", "180.0"))
TOTAL_SECONDS = WARMUP_SECONDS + EPISODE_SECONDS
FINALIZATION_GRACE_SECONDS = 1.0

# ---- Scenario setup ----
N_BOTTLES = int(os.environ.get("HARD_N_BOTTLES", "3"))
N_CANS = int(os.environ.get("HARD_N_CANS", "3"))
N_TABLES = int(os.environ.get("HARD_N_TABLES", "3"))

P_TARGET_ON_TABLE = float(os.environ.get("HARD_P_TARGET_ON_TABLE", "0.90"))

# Single runs do not force an empty table by default. Set this to a positive
# value only when deliberately reproducing the early-run table-eval condition.
FORCE_ONE_EMPTY_TABLE_NUM_RUNS = int(
    os.environ.get("HARD_FORCE_ONE_EMPTY_TABLE_NUM_RUNS", "0")
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

BOTTLE_PROTO_NODE = os.environ.get("HARD_BOTTLE_PROTO", "DatasetBeerBottle")
CAN_PROTO_NODE = os.environ.get("HARD_CAN_PROTO", "DatasetCan")

# ---- Arena / grid ----
ARENA_X = 5.5
ARENA_Y = 6.5
CELL_SIZE_M = 0.5
GRID_COLS = int(ARENA_X / CELL_SIZE_M)
GRID_ROWS = int(ARENA_Y / CELL_SIZE_M)

WALL_MARGIN = 0.35
SPAWN_CENTER_DISC_RADIUS_M = float(os.environ.get("HARD_SPAWN_CENTER_DISC_RADIUS_M", "0.15"))
DRONE_TARGET_EXCLUSION_RADIUS_M = float(os.environ.get("HARD_DRONE_TARGET_EXCLUSION_RADIUS_M", "1.0"))
DESK_DRONE_EXCLUSION_RADIUS_M = float(os.environ.get("HARD_DESK_DRONE_EXCLUSION_RADIUS_M", "1.25"))
TARGET_MIN_SEPARATION_M = float(os.environ.get("HARD_TARGET_MIN_SEPARATION_M", "0.22"))

# ---- Desk/table geometry ----
DESK_LENGTH_M = float(os.environ.get("HARD_DESK_LENGTH_M", "1.2"))
DESK_WIDTH_M = float(os.environ.get("HARD_DESK_WIDTH_M", "0.7"))
DESK_HEIGHT_M = float(os.environ.get("HARD_DESK_HEIGHT_M", "0.72"))
DESK_TOP_THICKNESS_M = float(os.environ.get("HARD_DESK_TOP_THICKNESS_M", "0.08"))
DESK_SPAWN_EDGE_MARGIN_M = float(os.environ.get("HARD_DESK_SPAWN_EDGE_MARGIN_M", "0.05"))
DESK_FLOOR_EXCLUSION_MARGIN_M = float(os.environ.get("HARD_DESK_FLOOR_EXCLUSION_MARGIN_M", "0.50"))
DESK_MIN_CENTER_DISTANCE_M = float(os.environ.get("HARD_DESK_MIN_CENTER_DISTANCE_M", "1.35"))
DESK_WALL_MARGIN_M = float(os.environ.get("HARD_DESK_WALL_MARGIN_M", "0.18"))
DESK_TABLETOP_TEXTURE_URL = os.environ.get(
    "HARD_DESK_TABLETOP_TEXTURE_URL",
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


# ---- Hard-environment distractors ----
# These are standard Webots PROTOs declared as IMPORTABLE EXTERNPROTOs in the
# hard world. They are deliberately not task classes and are ignored by the GT
# detector, which only maps target_bottle, target_can, and desk manifest entries.
N_SOCCER_BALLS = int(os.environ.get("HARD_N_SOCCER_BALLS", "1"))
N_CEREAL_BOXES = int(os.environ.get("HARD_N_CEREAL_BOXES", "2"))
N_PLATES = int(os.environ.get("HARD_N_PLATES", "3"))
N_FLOWER_POTS = int(os.environ.get("HARD_N_FLOWER_POTS", "2"))

# Optional distractors use a flat floor/table preference. A failed preferred
# placement falls back to the other admissible surface; plates are table-only.
P_DISTRACTOR_ON_TABLE = float(os.environ.get("HARD_P_DISTRACTOR_ON_TABLE", "0.50"))

DISTRACTOR_SPECS = [
    {
        "count": N_SOCCER_BALLS,
        "proto": "SoccerBall",
        "class_name": "distractor_soccer_ball",
        "model": "soccer ball",
        "def_prefix": "DISTRACTOR_SOCCER_BALL",
        "z_floor_m": 0.113,
        "z_table_offset_m": 0.113,
        "placement": "optional_table",
    },
    {
        "count": N_CEREAL_BOXES,
        "proto": "CerealBox",
        "class_name": "distractor_cereal_box",
        "model": "cereal box",
        "def_prefix": "DISTRACTOR_CEREAL_BOX",
        "z_floor_m": 0.0,
        "z_table_offset_m": 0.0,
        "placement": "optional_table",
    },
    {
        "count": N_PLATES,
        "proto": "Plate",
        "class_name": "distractor_plate",
        "model": "plate",
        "def_prefix": "DISTRACTOR_PLATE",
        "z_floor_m": 0.0,
        "z_table_offset_m": 0.0,
        "placement": "table_only",
    },
    {
        "count": N_FLOWER_POTS,
        "proto": "FlowerPot",
        "class_name": "distractor_flower_pot",
        "model": "flower pot",
        "def_prefix": "DISTRACTOR_FLOWER_POT",
        "z_floor_m": 0.0,
        "z_table_offset_m": 0.0,
        "placement": "optional_table",
    },
]

# ---- Drone placement ----
DRONE_DEF = "CRAZYFLIE"
DRONE_SPAWN_X_MIN = float(os.environ.get("HARD_DRONE_SPAWN_X_MIN", "-2.5"))
DRONE_SPAWN_X_MAX = float(os.environ.get("HARD_DRONE_SPAWN_X_MAX", "2.5"))
DRONE_SPAWN_Y_MIN = float(os.environ.get("HARD_DRONE_SPAWN_Y_MIN", "-3.0"))
DRONE_SPAWN_Y_MAX = float(os.environ.get("HARD_DRONE_SPAWN_Y_MAX", "3.0"))
DRONE_SPAWN_YAW = float(os.environ.get("HARD_DRONE_SPAWN_YAW", "0.0"))

# ---- Shared files ----
SPAWNED_OBJECTS_FILENAME = "spawned_objects.json"
EXPERIMENT_INIT_FILENAME = "experiment_init.json"
TASK_STATE_FILENAME = "task_state.json"
CONTROLLER_PROGRESS_FILENAME = "controller_progress.json"
EVAL_STATE_FILENAME = "eval_state.json"

# ---- Reports ----
REPORT_SUBDIR = "run_report"

Point = Tuple[float, float]
Cell = Tuple[int, int]


def dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


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
                "def": f"HARD_DESK_{desk_id:03d}",
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


def _sample_target_on_desk(
    desk: Dict[str, object],
    occupied_xy: List[Point],
    occupied_cells: set[Cell],
    *,
    reserve_cell: bool = True,
    max_tries: int = 2000,
) -> Optional[Tuple[float, float, int, int]]:
    cells = desk.get("valid_spawn_cells")
    if not cells:
        return None
    rect = desk["spawn_rect"]
    for _ in range(max_tries):
        cx, cy = random.choice(cells)
        if reserve_cell and (cx, cy) in occupied_cells:
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


def _sample_target_on_any_desk(
    desks: List[Dict[str, object]],
    occupied_xy: List[Point],
    occupied_cells: set[Cell],
    *,
    reserve_cell: bool = True,
    include_forced_empty: bool = False,
    max_tries: int = 2500,
) -> Optional[Tuple[float, float, int, int, Dict[str, object]]]:
    eligible_desks = [
        d
        for d in desks
        if include_forced_empty or not bool(d.get("forced_empty", False))
    ]
    if not eligible_desks:
        return None
    for _ in range(max_tries):
        desk = random.choice(eligible_desks)
        sampled = _sample_target_on_desk(
            desk,
            occupied_xy,
            occupied_cells,
            reserve_cell=reserve_cell,
            max_tries=1,
        )
        if sampled is None:
            continue
        x, y, cx, cy = sampled
        return (x, y, cx, cy, desk)
    return None


def _sample_target_off_desk(
    *,
    floor_cells: List[Cell],
    desks: List[Dict[str, object]],
    drone_xy: Point,
    occupied_xy: List[Point],
    occupied_cells: set[Cell],
    reserve_cell: bool = True,
    max_tries: int = 5000,
) -> Optional[Tuple[float, float, int, int]]:
    for _ in range(max_tries):
        cx, cy = random.choice(floor_cells)
        if reserve_cell and (cx, cy) in occupied_cells:
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


def _distractor_node_string(
    *,
    def_name: str,
    proto: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
) -> str:
    """Instantiate a standard imported Webots PROTO without target metadata."""
    return f'''
DEF {def_name} {proto} {{
  translation {x:.6f} {y:.6f} {z:.6f}
  rotation 0 0 1 {yaw:.6f}
  name "{def_name}"
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

        "target_setup": str(experiment_metadata.get("target_setup", "hard")),
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


def _spawn_scene(
    supervisor: Supervisor,
    spawn_children,
    *,
    seed: int,
    run_index: int,
    drone_xy: Point,
) -> tuple[List[dict], List[dict], List[dict], dict]:
    """Spawn tables, task targets, and non-target hard-environment distractors."""
    desks = _sample_desks(drone_xy)
    force_empty_this_run = (
        run_index < int(FORCE_ONE_EMPTY_TABLE_NUM_RUNS)
    ) and len(desks) > 0
    forced_empty_table_id = None
    if force_empty_this_run:
        forced_empty_table_id = int(random.choice(desks)["desk_id"])
        for desk in desks:
            desk["forced_empty"] = int(desk["desk_id"]) == forced_empty_table_id

    spawned_objects: List[dict] = []
    target_records: List[dict] = []
    distractor_records: List[dict] = []
    global_index = 0

    for desk in desks:
        desk["valid_spawn_cells"] = _valid_on_desk_cells(desk)
        if not desk["valid_spawn_cells"] and not bool(desk.get("forced_empty", False)):
            raise RuntimeError(
                f"Desk/table {desk['desk_id']} has no valid target spawn cells."
            )
        color_rgb = _instance_color(global_index)
        spawn_children.importMFNodeFromString(
            -1,
            _desk_node_string(desk, color_rgb),
        )
        desk_node = spawn_children.getMFNode(spawn_children.getCount() - 1)
        desk["id"] = _node_id(desk_node)
        spawn_children.importMFNodeFromString(-1, _desk_legs_node_string(desk))
        desk["recognition_color_rgb"] = [float(v) for v in color_rgb]
        global_index += 1

    floor_cells = _eligible_floor_cells()
    if not floor_cells:
        raise RuntimeError("No eligible floor cells available under current wall margin.")

    # Target cells remain unique because task validation is cell based. The same
    # position and separation constraints are shared with distractors, but
    # distractors may use another valid point within an already occupied cell
    # when physical separation permits it.
    occupied_xy: List[Point] = []
    occupied_target_cells: set[Cell] = set()

    # Plates are table-only, so place them before targets. Targets keep their
    # table preference but can safely fall back to the floor when table capacity
    # is consumed; required plates never need an invalid floor fallback.
    for spec in DISTRACTOR_SPECS:
        if str(spec["placement"]) != "table_only":
            continue
        for local_i in range(int(spec["count"])):
            sampled_on = _sample_target_on_any_desk(
                desks,
                occupied_xy,
                occupied_target_cells,
                reserve_cell=False,
                include_forced_empty=True,
                max_tries=5000,
            )
            if sampled_on is None:
                raise RuntimeError(
                    f"Could not place required table-only distractor "
                    f"{spec['class_name']} #{local_i} under hard-scene constraints."
                )

            x, y, cx, cy, desk_for_object = sampled_on
            occupied_xy.append((float(x), float(y)))
            def_name = f"{spec['def_prefix']}_{global_index:03d}"
            yaw = random.uniform(-math.pi, math.pi)
            z = DESK_HEIGHT_M + float(spec["z_table_offset_m"])
            spawn_children.importMFNodeFromString(
                -1,
                _distractor_node_string(
                    def_name=def_name,
                    proto=str(spec["proto"]),
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    yaw=float(yaw),
                ),
            )
            node = spawn_children.getMFNode(spawn_children.getCount() - 1)
            record = {
                "def": def_name,
                "id": _node_id(node),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "cx": int(cx),
                "cy": int(cy),
                "class_id": None,
                "class_name": str(spec["class_name"]),
                "proto": str(spec["proto"]),
                "model": str(spec["model"]),
                "is_distractor": True,
                "spawn_mode": "on_table",
                "desk_id": int(desk_for_object["desk_id"]),
                "yaw": float(yaw),
                "grid_cell_center_xy": [float(v) for v in _cell_center_xy(cx, cy)],
                "spawn_center_disc_radius_m": float(SPAWN_CENTER_DISC_RADIUS_M),
            }
            distractor_records.append(record)
            spawned_objects.append(record)
            global_index += 1

    for spec in OBJECT_SPECS:
        for local_i in range(int(spec["count"])):
            prefer_table = random.random() < P_TARGET_ON_TABLE
            sampled = None
            spawn_mode = "on_table" if prefer_table else "off_table"
            desk_for_object = None
            z = 0.0

            if prefer_table:
                sampled_on = _sample_target_on_any_desk(
                    desks,
                    occupied_xy,
                    occupied_target_cells,
                    reserve_cell=True,
                )
                if sampled_on is not None:
                    x, y, cx, cy, desk_for_object = sampled_on
                    sampled = (x, y, cx, cy)
                    z = DESK_HEIGHT_M
                else:
                    spawn_mode = "off_table"

            if sampled is None:
                sampled_off = _sample_target_off_desk(
                    floor_cells=floor_cells,
                    desks=desks,
                    drone_xy=drone_xy,
                    occupied_xy=occupied_xy,
                    occupied_cells=occupied_target_cells,
                    reserve_cell=True,
                )
                if sampled_off is not None:
                    x, y, cx, cy = sampled_off
                    sampled = (x, y, cx, cy)
                    desk_for_object = None
                    z = 0.0
                    spawn_mode = "off_table"

            if sampled is None:
                print(
                    f"WARNING: Could not place {spec['class_name']} #{local_i} "
                    "under constraints.",
                    flush=True,
                )
                continue

            x, y, cx, cy = sampled
            occupied_xy.append((float(x), float(y)))
            occupied_target_cells.add((int(cx), int(cy)))
            color_rgb = _instance_color(global_index)
            def_name = f"{spec['def_prefix']}_{global_index:03d}"
            node_string = _target_node_string(
                def_name=def_name,
                proto=str(spec["proto"]),
                model=str(spec["model"]),
                x=float(x),
                y=float(y),
                z=float(z),
                color_rgb=color_rgb,
            )
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
                "recognition_color_rgb": [
                    float(color_rgb[0]),
                    float(color_rgb[1]),
                    float(color_rgb[2]),
                ],
                "spawn_mode": spawn_mode,
                "desk_id": (
                    None
                    if desk_for_object is None
                    else int(desk_for_object["desk_id"])
                ),
                "grid_cell_center_xy": [
                    float(v) for v in _cell_center_xy(cx, cy)
                ],
                "spawn_center_disc_radius_m": float(SPAWN_CENTER_DISC_RADIUS_M),
            }
            target_records.append(target_record)
            spawned_objects.append(target_record)
            global_index += 1

    # The remaining optional distractors use the same floor/table admissibility,
    # wall margin, centre-disc sampling, drone exclusion, and physical-separation
    # conditions as targets. They are non-target objects and are therefore not
    # inserted into occupied_target_cells or task validation records.
    for spec in DISTRACTOR_SPECS:
        if str(spec["placement"]) == "table_only":
            continue
        for local_i in range(int(spec["count"])):
            placement = str(spec["placement"])
            prefer_table = placement == "table_only" or (
                placement == "optional_table"
                and random.random() < P_DISTRACTOR_ON_TABLE
            )
            sampled = None
            spawn_mode = "on_table" if prefer_table else "off_table"
            desk_for_object = None

            if prefer_table:
                sampled_on = _sample_target_on_any_desk(
                    desks,
                    occupied_xy,
                    occupied_target_cells,
                    reserve_cell=False,
                    include_forced_empty=True,
                    max_tries=5000,
                )
                if sampled_on is not None:
                    x, y, cx, cy, desk_for_object = sampled_on
                    sampled = (x, y, cx, cy)
                    z = DESK_HEIGHT_M + float(spec["z_table_offset_m"])
                elif placement != "table_only":
                    spawn_mode = "off_table"

            if sampled is None and placement != "table_only":
                sampled_off = _sample_target_off_desk(
                    floor_cells=floor_cells,
                    desks=desks,
                    drone_xy=drone_xy,
                    occupied_xy=occupied_xy,
                    occupied_cells=occupied_target_cells,
                    reserve_cell=False,
                    max_tries=5000,
                )
                if sampled_off is not None:
                    x, y, cx, cy = sampled_off
                    sampled = (x, y, cx, cy)
                    desk_for_object = None
                    z = float(spec["z_floor_m"])
                    spawn_mode = "off_table"

            if sampled is None:
                raise RuntimeError(
                    f"Could not place required distractor {spec['class_name']} "
                    f"#{local_i} under hard-scene constraints."
                )

            x, y, cx, cy = sampled
            occupied_xy.append((float(x), float(y)))
            def_name = f"{spec['def_prefix']}_{global_index:03d}"
            yaw = random.uniform(-math.pi, math.pi)
            node_string = _distractor_node_string(
                def_name=def_name,
                proto=str(spec["proto"]),
                x=float(x),
                y=float(y),
                z=float(z),
                yaw=float(yaw),
            )
            spawn_children.importMFNodeFromString(-1, node_string)
            node = spawn_children.getMFNode(spawn_children.getCount() - 1)
            record = {
                "def": def_name,
                "id": _node_id(node),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "cx": int(cx),
                "cy": int(cy),
                "class_id": None,
                "class_name": str(spec["class_name"]),
                "proto": str(spec["proto"]),
                "model": str(spec["model"]),
                "is_distractor": True,
                "spawn_mode": spawn_mode,
                "desk_id": (
                    None
                    if desk_for_object is None
                    else int(desk_for_object["desk_id"])
                ),
                "yaw": float(yaw),
                "grid_cell_center_xy": [
                    float(v) for v in _cell_center_xy(cx, cy)
                ],
                "spawn_center_disc_radius_m": float(SPAWN_CENTER_DISC_RADIUS_M),
            }
            distractor_records.append(record)
            spawned_objects.append(record)
            global_index += 1

    target_desk_ids = {
        int(t["desk_id"])
        for t in target_records
        if t.get("desk_id", None) is not None
    }
    for desk in desks:
        desk["is_empty"] = int(desk["desk_id"]) not in target_desk_ids
        desk_export = dict(desk)
        desk_export.pop("valid_spawn_cells", None)
        spawned_objects.append(desk_export)

    target_counts = {
        name: sum(1 for r in target_records if r["class_name"] == name)
        for name in (CLASS_BOTTLE, CLASS_CAN)
    }
    on_table_count = sum(
        1 for t in target_records if t["spawn_mode"] == "on_table"
    )
    off_table_count = sum(
        1 for t in target_records if t["spawn_mode"] == "off_table"
    )
    distractor_counts = {
        str(spec["class_name"]): sum(
            1
            for record in distractor_records
            if record["class_name"] == str(spec["class_name"])
        )
        for spec in DISTRACTOR_SPECS
    }
    distractors_on_table = sum(
        1 for d in distractor_records if d["spawn_mode"] == "on_table"
    )
    distractors_off_table = sum(
        1 for d in distractor_records if d["spawn_mode"] == "off_table"
    )
    empty_table_ids = sorted(
        [int(d["desk_id"]) for d in desks if bool(d.get("is_empty", False))]
    )
    table_scenario = {
        "scenario": ENVIRONMENT_ID,
        "environment": ENVIRONMENT_ID,
        "run_mode": RUN_MODE,
        "num_tables": int(len(desks)),
        "num_targets": int(len(target_records)),
        "num_bottles": int(target_counts[CLASS_BOTTLE]),
        "num_cans": int(target_counts[CLASS_CAN]),
        "p_target_on_table": float(P_TARGET_ON_TABLE),
        "target_spawn_counts": {
            "on_table": int(on_table_count),
            "off_table": int(off_table_count),
        },
        "empty_table_ids": empty_table_ids,
        "forced_empty_table_id": forced_empty_table_id,
        "force_empty_this_run": bool(force_empty_this_run),
        "desk_spawn_edge_margin_m": float(DESK_SPAWN_EDGE_MARGIN_M),
        "desk_floor_exclusion_margin_m": float(DESK_FLOOR_EXCLUSION_MARGIN_M),
        "spawn_center_disc_radius_m": float(SPAWN_CENTER_DISC_RADIUS_M),
        "distractor_scenario": {
            "num_distractors": int(len(distractor_records)),
            "p_optional_distractor_on_table": float(P_DISTRACTOR_ON_TABLE),
            "spawn_counts": {
                "on_table": int(distractors_on_table),
                "off_table": int(distractors_off_table),
            },
            "class_counts": distractor_counts,
            "table_only_class_names": ["distractor_plate"],
        },
    }
    return spawned_objects, target_records, desks, table_scenario


def main():
    if not (0.0 < SPAWN_CENTER_DISC_RADIUS_M < 0.5 * CELL_SIZE_M):
        raise ValueError(
            f"SPAWN_CENTER_DISC_RADIUS_M must be in (0, {0.5 * CELL_SIZE_M:.3f}), "
            f"got {SPAWN_CENTER_DISC_RADIUS_M}"
        )
    if not (0.0 <= P_TARGET_ON_TABLE <= 1.0):
        raise ValueError(
            f"P_TARGET_ON_TABLE must be in [0, 1], got {P_TARGET_ON_TABLE}"
        )
    if not (0.0 <= P_DISTRACTOR_ON_TABLE <= 1.0):
        raise ValueError(
            "P_DISTRACTOR_ON_TABLE must be in [0, 1], "
            f"got {P_DISTRACTOR_ON_TABLE}"
        )

    random.seed(SEED)

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    policy_name = normalize_policy_id(os.environ.get("AIF_POLICY", DEFAULT_POLICY))

    spawned_objects_path = SHARED_STATE.path(SPAWNED_OBJECTS_FILENAME)
    experiment_init_path = SHARED_STATE.path(EXPERIMENT_INIT_FILENAME)
    task_state_path = SHARED_STATE.path(TASK_STATE_FILENAME)
    controller_progress_path = SHARED_STATE.path(CONTROLLER_PROGRESS_FILENAME)
    eval_state_path = SHARED_STATE.path(EVAL_STATE_FILENAME)

    # Clear stale inter-process state so a single run cannot inherit a previous
    # evaluation seed, progress record, or evaluation output directory.
    SHARED_STATE.clear(
        CONTROLLER_READY_FILENAME,
        IPC_SESSION_FILENAME,
        SPAWNED_OBJECTS_FILENAME,
        EXPERIMENT_INIT_FILENAME,
        TASK_STATE_FILENAME,
        CONTROLLER_PROGRESS_FILENAME,
        EVAL_STATE_FILENAME,
    )

    print("[HARD] shared state cleared", flush=True)

    establish_supervisor_session(
        SHARED_STATE,
        environment=ENVIRONMENT_ID,
        run_mode=RUN_MODE,
    )
    print("[IPC] supervisor session established", flush=True)

    spawn_root = supervisor.getFromDef("SPAWN_ROOT")
    if spawn_root is None:
        print("ERROR: DEF SPAWN_ROOT Group not found in the world.", flush=True)
        return
    spawn_children = spawn_root.getField("children")

    drone_node = supervisor.getFromDef(DRONE_DEF)
    drone_xy = _sample_drone_spawn_xy()
    _place_drone(
        supervisor,
        drone_node,
        x=drone_xy[0],
        y=drone_xy[1],
        yaw=DRONE_SPAWN_YAW,
    )

    spawned_objects, target_records, desks, table_scenario = _spawn_scene(
        supervisor,
        spawn_children,
        seed=SEED,
        run_index=0,
        drone_xy=drone_xy,
    )

    print(
        f"[HARD] policy={policy_name} seed={SEED} "
        f"drone_spawn=({drone_xy[0]:.3f},{drone_xy[1]:.3f}) "
        f"tables={len(desks)} targets={len(target_records)} "
        f"on_table={table_scenario['target_spawn_counts']['on_table']} "
        f"off_table={table_scenario['target_spawn_counts']['off_table']} "
        f"distractors={table_scenario['distractor_scenario']['num_distractors']} "
        f"empty_tables={table_scenario['empty_table_ids']} "
        f"forced_empty={table_scenario['forced_empty_table_id']}",
        flush=True,
    )

    _write_experiment_init_json(
        experiment_init_path,
        seed=SEED,
        drone_spawn_xy=drone_xy,
        drone_spawn_yaw=DRONE_SPAWN_YAW,
        table_scenario=table_scenario,
    )
    print(f"Experiment init JSON exported: {experiment_init_path}", flush=True)

    _write_spawned_objects_json(
        spawned_objects_path,
        seed=SEED,
        objects=spawned_objects,
        table_scenario=table_scenario,
    )
    print(f"GT JSON exported: {spawned_objects_path}", flush=True)

    true_objects_by_cell: Dict[Cell, dict] = {}
    for obj in target_records:
        true_objects_by_cell[(int(obj["cx"]), int(obj["cy"]))] = obj

    total_objects = len(true_objects_by_cell)
    validated_found_objects: set[tuple[int, int, str]] = set()
    classification_stats = _empty_classification_stats()
    seen_claim_keys = set()
    report_written = False
    final_reason = None
    final_time_s = None

    _write_task_state(
        task_state_path,
        status="running",
        reason=None,
        seed=SEED,
        total_objects=total_objects,
        remaining_objects=total_objects,
        validated_found_objects=validated_found_objects,
        timeout_s=TOTAL_SECONDS,
        warmup_s=WARMUP_SECONDS,
        episode_s=EPISODE_SECONDS,
        now_time_s=0.0,
        report_written=False,
    )

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
            "detector_limitations": {
                "mode": "unknown",
                "frames_interval": None,
                "time_interval_s": None,
            },
            "drone_spawn": {
                "x": float(drone_xy[0]),
                "y": float(drone_xy[1]),
                "yaw": float(DRONE_SPAWN_YAW),
                "cx": None if spawn_cxcy is None else int(spawn_cxcy[0]),
                "cy": None if spawn_cxcy is None else int(spawn_cxcy[1]),
            },
        },
    )

    start_time = supervisor.getTime()
    while supervisor.step(timestep) != -1:
        now_time_s = float(supervisor.getTime())
        elapsed_total = now_time_s - float(start_time)

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
                "detector_limitations": {
                    "mode": "unknown",
                    "frames_interval": None,
                    "time_interval_s": None,
                },
            },
        )
        detector_limitations = _normalize_detector_limitations(
            controller_progress.get("detector_limitations", {})
        )
        experiment_metadata = _normalize_experiment_metadata(
            controller_progress.get("experiment_metadata", {})
        )
        if experiment_metadata.get("target_setup") in (None, "unknown"):
            experiment_metadata["target_setup"] = "hard"
        target_distance_stats = _normalize_target_distance_stats(
            controller_progress.get("target_distance_stats", {})
        )

        claims = _extract_claims_from_progress(controller_progress)
        validated_found_objects = _evaluate_new_claims(
            claims=claims,
            seen_claim_keys=seen_claim_keys,
            true_objects_by_cell=true_objects_by_cell,
            validated_found_objects=validated_found_objects,
            classification_stats=classification_stats,
        )
        remaining_objects = max(0, total_objects - len(validated_found_objects))

        if len(validated_found_objects) >= total_objects and total_objects > 0:
            final_reason = "done"
            final_time_s = now_time_s
        elif elapsed_total >= TOTAL_SECONDS:
            final_reason = "timeout"
            final_time_s = now_time_s

        if final_reason is None:
            _write_task_state(
                task_state_path,
                status="running",
                reason=None,
                seed=SEED,
                total_objects=total_objects,
                remaining_objects=remaining_objects,
                validated_found_objects=validated_found_objects,
                timeout_s=TOTAL_SECONDS,
                warmup_s=WARMUP_SECONDS,
                episode_s=EPISODE_SECONDS,
                now_time_s=now_time_s,
                report_written=False,
            )
            continue

        _write_task_state(
            task_state_path,
            status=final_reason,
            reason=final_reason,
            seed=SEED,
            total_objects=total_objects,
            remaining_objects=remaining_objects,
            validated_found_objects=validated_found_objects,
            timeout_s=TOTAL_SECONDS,
            warmup_s=WARMUP_SECONDS,
            episode_s=EPISODE_SECONDS,
            now_time_s=float(final_time_s),
            report_written=False,
        )

        run_dir = controller_progress.get("run_dir", None)
        autonomy_start_time_s = controller_progress.get("autonomy_start_time_s", None)
        visited_cells = {
            (int(cx), int(cy))
            for cx, cy in controller_progress.get("visited_cells", [])
        }

        if run_dir:
            _write_run_report(
                run_dir=run_dir,
                policy_name=policy_name,
                gt_seed=SEED,
                total_objects=total_objects,
                validated_found_objects=validated_found_objects,
                remaining_objects=remaining_objects,
                autonomy_start_time_s=autonomy_start_time_s,
                now_time_s=float(final_time_s),
                timeout_s=TOTAL_SECONDS,
                visited_cells=visited_cells,
                grid_rows=GRID_ROWS,
                grid_cols=GRID_COLS,
                reason=final_reason,
                classification_stats=classification_stats,
                detector_limitations=detector_limitations,
                experiment_metadata=experiment_metadata,
                target_distance_stats=target_distance_stats,
                table_scenario=table_scenario,
            )
        else:
            print("[RUN REPORT] run_dir not available yet, skipping file write", flush=True)

        report_written = True
        _write_task_state(
            task_state_path,
            status=final_reason,
            reason=final_reason,
            seed=SEED,
            total_objects=total_objects,
            remaining_objects=remaining_objects,
            validated_found_objects=validated_found_objects,
            timeout_s=TOTAL_SECONDS,
            warmup_s=WARMUP_SECONDS,
            episode_s=EPISODE_SECONDS,
            now_time_s=float(final_time_s),
            report_written=report_written,
        )
        print(f"[HARD] end-of-run state: {final_reason}", flush=True)

        grace_start = supervisor.getTime()
        while supervisor.step(timestep) != -1:
            if (supervisor.getTime() - grace_start) >= FINALIZATION_GRACE_SECONDS:
                break
        _cleanup_spawned_nodes(spawn_children)
        print("Cleanup complete. Quitting simulation.", flush=True)
        supervisor.simulationQuit(0)
        return


if __name__ == "__main__":
    main()
