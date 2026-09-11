# webots/controllers/worldgen_medium_supervisor/worldgen_medium_supervisor.py

from __future__ import annotations

import colorsys
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

from controller import Supervisor

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

ENVIRONMENT_ID = "medium"
RUN_MODE = "single"

# -----------------------------------------------------------------------------
# Single-run table scenario supervisor
# -----------------------------------------------------------------------------
# Purpose:
#   - Same role as worldgen_easy_supervisor.py for the current easy scenario.
#   - Spawns a room with target objects plus tabletop/desk objects.
#   - Writes the same shared files consumed by crazyflie_AIF.py:
#       spawned_objects.json
#       experiment_init.json
#       task_state.json
#       controller_progress.json
#   - Validates task completion against TARGET objects only.
#     Desks/tables are exported in spawned_objects.json for perception/GT use,
#     but they are not counted as objects that need to be "found".

# ---- Experiment settings ----
SEED = int(os.environ.get("MEDIUM_SCENARIO_SEED", "34"))

WARMUP_SECONDS = float(os.environ.get("MEDIUM_SCENARIO_WARMUP_SECONDS", "15.0"))
EPISODE_SECONDS = float(os.environ.get("MEDIUM_SCENARIO_EPISODE_SECONDS", "180.0"))
TOTAL_SECONDS = WARMUP_SECONDS + EPISODE_SECONDS
FINALIZATION_GRACE_SECONDS = 1.0

# ---- Object setup ----
N_BOTTLES = int(os.environ.get("MEDIUM_SCENARIO_N_BOTTLES", "3"))
N_CANS = int(os.environ.get("MEDIUM_SCENARIO_N_CANS", "3"))
N_TABLES = int(os.environ.get("MEDIUM_SCENARIO_N_TABLES", "3"))

P_TARGET_ON_TABLE = float(os.environ.get("MEDIUM_SCENARIO_P_TARGET_ON_TABLE", "0.90"))

# ---- Class / proto mapping ----
CLASS_BOTTLE = "target_bottle"
CLASS_CAN = "target_can"
CLASS_DESK = "desk"

CLASS_NAME_TO_ID = {
    CLASS_BOTTLE: 0,
    CLASS_CAN: 1,
    CLASS_DESK: 2,
}

TARGET_CLASS_NAMES = {CLASS_BOTTLE, CLASS_CAN}

# These custom PROTOs should expose model and recognitionColors and forward them
# into the root Solid. They keep the same visuals/textures as the original objects.
BOTTLE_PROTO_NODE = os.environ.get("MEDIUM_SCENARIO_BOTTLE_PROTO", "DatasetBeerBottle")
CAN_PROTO_NODE = os.environ.get("MEDIUM_SCENARIO_CAN_PROTO", "DatasetCan")

# ---- Arena / grid ----
ARENA_X = 5.5
ARENA_Y = 6.5
CELL_SIZE_M = 0.5
GRID_COLS = int(ARENA_X / CELL_SIZE_M)
GRID_ROWS = int(ARENA_Y / CELL_SIZE_M)

# Keep floor targets away from walls.
WALL_MARGIN = 0.35

# The target itself must be sampled within this radius from the selected grid-cell center.
SPAWN_CENTER_DISC_RADIUS_M = float(os.environ.get("MEDIUM_SCENARIO_SPAWN_CENTER_DISC_RADIUS_M", "0.15"))

# Keep targets away from the drone start.
DRONE_TARGET_EXCLUSION_RADIUS_M = float(os.environ.get("MEDIUM_SCENARIO_DRONE_TARGET_EXCLUSION_RADIUS_M", "1.0"))

# Keep desks away from the drone start.
DESK_DRONE_EXCLUSION_RADIUS_M = float(os.environ.get("MEDIUM_SCENARIO_DESK_DRONE_EXCLUSION_RADIUS_M", "1.25"))

# Avoid overlapping target objects. This is intentionally much smaller than the
# old floor-only MIN_SEPARATION because on-table objects can be close, but should
# not visually overlap.
TARGET_MIN_SEPARATION_M = float(os.environ.get("MEDIUM_SCENARIO_TARGET_MIN_SEPARATION_M", "0.22"))

# ---- Desk/table geometry ----
DESK_LENGTH_M = float(os.environ.get("MEDIUM_SCENARIO_DESK_LENGTH_M", "1.2"))
DESK_WIDTH_M = float(os.environ.get("MEDIUM_SCENARIO_DESK_WIDTH_M", "0.7"))
DESK_HEIGHT_M = float(os.environ.get("MEDIUM_SCENARIO_DESK_HEIGHT_M", "0.72"))  # tabletop top surface
DESK_TOP_THICKNESS_M = float(os.environ.get("MEDIUM_SCENARIO_DESK_TOP_THICKNESS_M", "0.08"))

# Tabletop spawn zone = tabletop rectangle shrunk by 5 cm from each edge.
DESK_SPAWN_EDGE_MARGIN_M = float(os.environ.get("MEDIUM_SCENARIO_DESK_SPAWN_EDGE_MARGIN_M", "0.05"))

# No floor target should spawn in the 0.5 m border outside the table spawn zone.
DESK_FLOOR_EXCLUSION_MARGIN_M = float(os.environ.get("MEDIUM_SCENARIO_DESK_FLOOR_EXCLUSION_MARGIN_M", "0.50"))

# Desk centers should not be too close to one another.
DESK_MIN_CENTER_DISTANCE_M = float(os.environ.get("MEDIUM_SCENARIO_DESK_MIN_CENTER_DISTANCE_M", "1.35"))

# Keep desk tabletop corners inside the room by this margin.
DESK_WALL_MARGIN_M = float(os.environ.get("MEDIUM_SCENARIO_DESK_WALL_MARGIN_M", "0.18"))

DESK_TABLETOP_TEXTURE_URL = os.environ.get(
    "MEDIUM_SCENARIO_DESK_TABLETOP_TEXTURE_URL",
    "world_resources/wooden_tabletop.jpeg",
)

# ---- Target approximate physical sizes / z placement metadata ----
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
DRONE_SPAWN_X = float(os.environ.get("MEDIUM_SCENARIO_DRONE_SPAWN_X", "0.0"))
DRONE_SPAWN_Y = float(os.environ.get("MEDIUM_SCENARIO_DRONE_SPAWN_Y", "0.0"))
DRONE_SPAWN_YAW = float(os.environ.get("MEDIUM_SCENARIO_DRONE_SPAWN_YAW", "0.0"))

# ---- Shared files ----
SPAWNED_OBJECTS_FILENAME = "spawned_objects.json"
EXPERIMENT_INIT_FILENAME = "experiment_init.json"
TASK_STATE_FILENAME = "task_state.json"
CONTROLLER_PROGRESS_FILENAME = "controller_progress.json"

# ---- Run report ----
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
    """Stable, distinct-ish per-instance recognition color."""
    hue = (index * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (max(0.1, r), max(0.1, g), max(0.1, b))


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
    return (
        min_x + (float(cx) + 0.5) * CELL_SIZE_M,
        min_y + (float(cy) + 0.5) * CELL_SIZE_M,
    )


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
    return [
        _local_to_world(lx, ly, rect["x"], rect["y"], rect["yaw"])
        for lx, ly in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    ]


def _rect_record(x: float, y: float, yaw: float, length: float, width: float) -> Dict[str, object]:
    rect = {
        "x": float(x),
        "y": float(y),
        "yaw": float(yaw),
        "length": float(length),
        "width": float(width),
    }
    corners = _rotated_rect_corners(rect)  # type: ignore[arg-type]
    rect["corners_xy"] = [[float(px), float(py)] for px, py in corners]
    return rect


def _desk_fits_room(rect: Dict[str, float]) -> bool:
    min_x = -ARENA_X / 2.0 + DESK_WALL_MARGIN_M
    max_x = ARENA_X / 2.0 - DESK_WALL_MARGIN_M
    min_y = -ARENA_Y / 2.0 + DESK_WALL_MARGIN_M
    max_y = ARENA_Y / 2.0 - DESK_WALL_MARGIN_M

    for x, y in _rotated_rect_corners(rect):
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
    return True


def _target_within_floor_bounds(x: float, y: float) -> bool:
    return (
        -ARENA_X / 2.0 + WALL_MARGIN <= x <= ARENA_X / 2.0 - WALL_MARGIN
        and -ARENA_Y / 2.0 + WALL_MARGIN <= y <= ARENA_Y / 2.0 - WALL_MARGIN
    )


def _eligible_floor_cells() -> List[Cell]:
    cells: List[Cell] = []
    for cy in range(GRID_ROWS):
        for cx in range(GRID_COLS):
            x, y = _cell_center_xy(cx, cy)
            if _target_within_floor_bounds(x, y):
                cells.append((cx, cy))
    return cells


def _desk_spawn_rect(desk: Dict[str, object]) -> Dict[str, object]:
    return _rect_record(
        float(desk["x"]),
        float(desk["y"]),
        float(desk["yaw"]),
        DESK_LENGTH_M - 2.0 * DESK_SPAWN_EDGE_MARGIN_M,
        DESK_WIDTH_M - 2.0 * DESK_SPAWN_EDGE_MARGIN_M,
    )


def _desk_floor_exclusion_rect(desk: Dict[str, object]) -> Dict[str, object]:
    spawn_rect = desk["spawn_rect"]
    return _rect_record(
        float(desk["x"]),
        float(desk["y"]),
        float(desk["yaw"]),
        float(spawn_rect["length"]) + 2.0 * DESK_FLOOR_EXCLUSION_MARGIN_M,
        float(spawn_rect["width"]) + 2.0 * DESK_FLOOR_EXCLUSION_MARGIN_M,
    )


def _in_any_desk_spawn_zone(x: float, y: float, desks: List[Dict[str, object]]) -> bool:
    return any(_point_in_rotated_rect(x, y, desk["spawn_rect"]) for desk in desks)


def _in_any_desk_floor_exclusion_zone(x: float, y: float, desks: List[Dict[str, object]]) -> bool:
    return any(_point_in_rotated_rect(x, y, desk["floor_exclusion_rect"]) for desk in desks)


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
                "def": f"MEDIUM_SCENARIO_DESK_{desk_id:03d}",
                "class_id": CLASS_NAME_TO_ID[CLASS_DESK],
                "class_name": CLASS_DESK,
                "model": CLASS_DESK,
                "x": float(x),
                "y": float(y),
                "z": float(DESK_HEIGHT_M),
                "yaw": float(yaw),
                "dimensions_m": {
                    "length": DESK_LENGTH_M,
                    "width": DESK_WIDTH_M,
                    "height": DESK_HEIGHT_M,
                    "top_thickness": DESK_TOP_THICKNESS_M,
                },
                "tabletop_rect": tabletop_rect,
                "is_empty": False,
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
    if not desks:
        return None

    for _ in range(max_tries):
        desk = random.choice(desks)
        sampled = _sample_target_on_desk(desk, occupied_xy, occupied_cells, max_tries=1)
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
    max_tries: int = 5000,
) -> Optional[Tuple[float, float, int, int]]:
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


def _target_node_string(
    *,
    def_name: str,
    proto: str,
    model: str,
    x: float,
    y: float,
    z: float,
    color_rgb,
) -> str:
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


def _write_spawned_objects_json(path: str, *, seed: int, objects: List[dict]) -> None:
    payload = {
        "seed": int(seed),
        "objects": objects,
        "class_name_to_id": CLASS_NAME_TO_ID,
        "target_class_names": sorted(TARGET_CLASS_NAMES),
        "table_scenario": {
            "num_tables": int(N_TABLES),
            "p_target_on_table": float(P_TARGET_ON_TABLE),
            "desk_spawn_edge_margin_m": float(DESK_SPAWN_EDGE_MARGIN_M),
            "desk_floor_exclusion_margin_m": float(DESK_FLOOR_EXCLUSION_MARGIN_M),
            "spawn_center_disc_radius_m": float(SPAWN_CENTER_DISC_RADIUS_M),
        },
    }
    atomic_write_json(path, payload)


def _write_experiment_init_json(path: str, *, seed: int, drone_spawn_xy: Point, drone_spawn_yaw: float):
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
        "table_scenario": {
            "num_tables": int(N_TABLES),
            "num_bottles": int(N_BOTTLES),
            "num_cans": int(N_CANS),
            "p_target_on_table": float(P_TARGET_ON_TABLE),
        },
    }
    atomic_write_json(path, payload)


def _write_task_state(
    path: str,
    *,
    status: str,
    reason: str | None,
    seed: int,
    total_objects: int,
    remaining_objects: int,
    validated_found_objects: set[tuple[int, int, str]],
    timeout_s: float,
    warmup_s: float,
    episode_s: float,
    now_time_s: float,
    report_written: bool,
) -> None:
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
                {
                    "cx": int(cx),
                    "cy": int(cy),
                    "class_name": str(class_name),
                    "class_id": int(CLASS_NAME_TO_ID[class_name]),
                }
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


def _write_run_report(
    *,
    run_dir: str,
    policy_name: str,
    gt_seed: int,
    total_objects: int,
    validated_found_objects: set[tuple[int, int, str]],
    remaining_objects: int,
    autonomy_start_time_s: float | None,
    now_time_s: float,
    timeout_s: float,
    visited_cells: set[Cell],
    grid_rows: int,
    grid_cols: int,
    reason: str,
    spawned_objects: List[dict],
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    report_dir = os.path.join(run_dir, REPORT_SUBDIR)
    os.makedirs(report_dir, exist_ok=True)

    found_count = int(len(validated_found_objects))
    elapsed = (
        float(now_time_s - autonomy_start_time_s)
        if autonomy_start_time_s is not None
        else float(now_time_s)
    )

    coverage_cells = int(len(visited_cells))
    total_cells = int(grid_rows * grid_cols)
    coverage_ratio = float(coverage_cells) / float(total_cells) if total_cells > 0 else 0.0

    target_objects = [o for o in spawned_objects if o.get("class_name") in TARGET_CLASS_NAMES]
    desk_objects = [o for o in spawned_objects if o.get("class_name") == CLASS_DESK]

    payload = {
        "policy": str(policy_name),
        "gt_seed": int(gt_seed),
        "reason": str(reason),
        "scenario": ENVIRONMENT_ID,
        "environment": ENVIRONMENT_ID,
        "run_mode": RUN_MODE,
        "autonomy_start_time_s": autonomy_start_time_s,
        "now_time_s": float(now_time_s),
        "elapsed_s": float(elapsed),
        "timeout_s": float(timeout_s),
        "total_objects": int(total_objects),
        "found_count": int(found_count),
        "remaining_objects": int(remaining_objects),
        "found_objects": sorted(
            [
                {
                    "cx": int(cx),
                    "cy": int(cy),
                    "class_name": str(class_name),
                    "class_id": int(CLASS_NAME_TO_ID[class_name]),
                }
                for (cx, cy, class_name) in validated_found_objects
            ],
            key=lambda d: (d["cy"], d["cx"], d["class_name"]),
        ),
        "coverage": {
            "visited_cells": int(coverage_cells),
            "total_cells": int(total_cells),
            "coverage_ratio": float(coverage_ratio),
        },
        "table_scenario": {
            "num_tables": int(len(desk_objects)),
            "num_targets": int(len(target_objects)),
            "p_target_on_table": float(P_TARGET_ON_TABLE),
            "target_spawn_counts": {
                "on_table": int(sum(1 for o in target_objects if o.get("spawn_mode") == "on_table")),
                "off_table": int(sum(1 for o in target_objects if o.get("spawn_mode") == "off_table")),
            },
            "empty_table_ids": sorted([int(o["desk_id"]) for o in desk_objects if bool(o.get("is_empty", False))]),
        },
    }

    out_path = os.path.join(report_dir, "run_report.json")
    atomic_write_json(out_path, payload)
    print(f"[RUN REPORT] wrote {out_path}", flush=True)
    return out_path


def _place_drone(supervisor: Supervisor, drone_node, *, x: float, y: float, yaw: float) -> None:
    if drone_node is None:
        raise RuntimeError(
            f"Drone DEF '{DRONE_DEF}' not found. Set DRONE_DEF to the drone DEF used in your world."
        )

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


def main():
    random.seed(SEED)

    if not (0.0 < SPAWN_CENTER_DISC_RADIUS_M < (0.5 * CELL_SIZE_M)):
        raise ValueError(
            f"SPAWN_CENTER_DISC_RADIUS_M must be in (0, {0.5 * CELL_SIZE_M:.3f}), "
            f"got {SPAWN_CENTER_DISC_RADIUS_M}"
        )

    if not (0.0 <= P_TARGET_ON_TABLE <= 1.0):
        raise ValueError(f"P_TARGET_ON_TABLE must be in [0, 1], got {P_TARGET_ON_TABLE}")

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    spawned_objects_path = SHARED_STATE.path(SPAWNED_OBJECTS_FILENAME)
    experiment_init_path = SHARED_STATE.path(EXPERIMENT_INIT_FILENAME)
    task_state_path = SHARED_STATE.path(TASK_STATE_FILENAME)
    controller_progress_path = SHARED_STATE.path(CONTROLLER_PROGRESS_FILENAME)

    SHARED_STATE.clear(
        CONTROLLER_READY_FILENAME,
        IPC_SESSION_FILENAME,
        SPAWNED_OBJECTS_FILENAME,
        EXPERIMENT_INIT_FILENAME,
        TASK_STATE_FILENAME,
        CONTROLLER_PROGRESS_FILENAME,
    )

    print("[MEDIUM] shared state cleared", flush=True)

    establish_supervisor_session(
        SHARED_STATE,
        environment=ENVIRONMENT_ID,
        run_mode=RUN_MODE,
    )
    print("[IPC] supervisor session established", flush=True)

    spawn_root = supervisor.getFromDef("SPAWN_ROOT")
    if spawn_root is None:
        print("ERROR: DEF SPAWN_ROOT Group not found in the world.")
        return
    spawn_children = spawn_root.getField("children")

    drone_node = supervisor.getFromDef(DRONE_DEF)
    _place_drone(
        supervisor,
        drone_node,
        x=DRONE_SPAWN_X,
        y=DRONE_SPAWN_Y,
        yaw=DRONE_SPAWN_YAW,
    )
    drone_xy = (DRONE_SPAWN_X, DRONE_SPAWN_Y)

    spawned_objects: List[dict] = []
    target_records: List[dict] = []
    global_index = 0

    # ---- Spawn desks/tables first ----
    desks = _sample_desks(drone_xy)
    for desk in desks:
        desk["valid_spawn_cells"] = _valid_on_desk_cells(desk)
        if not desk["valid_spawn_cells"]:
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
        raise RuntimeError("No eligible floor cells available under the current wall margin.")

    occupied_xy: List[Point] = []
    occupied_target_cells: set[Cell] = set()

    # ---- Spawn targets ----
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
                    # Fall back to floor if all table zones are exhausted. This should be rare
                    # with the default 3 tables / 6 targets setup, but avoids failing the run.
                    spawn_mode = "off_table"

            if sampled is None:
                sampled_off = _sample_target_off_desk(
                    floor_cells=floor_cells,
                    desks=desks,
                    drone_xy=drone_xy,
                    occupied_xy=occupied_xy,
                    occupied_cells=occupied_target_cells,
                )
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
                "recognition_color_rgb": [float(v) for v in color_rgb],
                "spawn_mode": spawn_mode,
                "desk_id": None if desk_for_target is None else int(desk_for_target["desk_id"]),
                "grid_cell_center_xy": [float(v) for v in _cell_center_xy(cx, cy)],
                "spawn_center_disc_radius_m": float(SPAWN_CENTER_DISC_RADIUS_M),
            }
            target_records.append(target_record)
            spawned_objects.append(target_record)
            global_index += 1

    # Mark desk emptiness after target placement and export desks after the final state is known.
    target_desk_ids = {int(t["desk_id"]) for t in target_records if t.get("desk_id", None) is not None}
    for desk in desks:
        desk["is_empty"] = int(desk["desk_id"]) not in target_desk_ids
        # Avoid leaking helper-only cells into the public JSON.
        desk_export = dict(desk)
        desk_export.pop("valid_spawn_cells", None)
        spawned_objects.append(desk_export)

    on_table_count = sum(1 for t in target_records if t["spawn_mode"] == "on_table")
    off_table_count = sum(1 for t in target_records if t["spawn_mode"] == "off_table")

    print(
        f"[MEDIUM SCENARIO] seed={SEED} tables={len(desks)} targets={len(target_records)} "
        f"on_table={on_table_count} off_table={off_table_count} "
        f"empty_tables={sorted([int(d['desk_id']) for d in desks if d.get('is_empty')])}",
        flush=True,
    )

    _write_experiment_init_json(
        experiment_init_path,
        seed=SEED,
        drone_spawn_xy=drone_xy,
        drone_spawn_yaw=DRONE_SPAWN_YAW,
    )
    print(f"Experiment init JSON exported: {experiment_init_path}", flush=True)

    _write_spawned_objects_json(
        spawned_objects_path,
        seed=SEED,
        objects=spawned_objects,
    )
    print(f"GT JSON exported: {spawned_objects_path}", flush=True)

    true_target_objects: set[tuple[int, int, str]] = set()
    for obj in target_records:
        if obj.get("cx", None) is None or obj.get("cy", None) is None:
            continue
        true_target_objects.add((int(obj["cx"]), int(obj["cy"]), str(obj["class_name"])))

    total_objects = len(true_target_objects)
    validated_found_objects: set[tuple[int, int, str]] = set()
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

    spawn_cxcy = _cxcy_from_xy(DRONE_SPAWN_X, DRONE_SPAWN_Y)

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
            "drone_spawn": {
                "x": DRONE_SPAWN_X,
                "y": DRONE_SPAWN_Y,
                "yaw": DRONE_SPAWN_YAW,
                "cx": spawn_cxcy[0] if spawn_cxcy is not None else None,
                "cy": spawn_cxcy[1] if spawn_cxcy is not None else None,
            },
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
            },
        )

        claimed_found_objects = set()
        for o in controller_progress.get("claimed_found_objects", []):
            try:
                class_name = str(o.get("class_name", o.get("label", "unknown")))
                if class_name not in TARGET_CLASS_NAMES:
                    continue
                claimed_found_objects.add((int(o["cx"]), int(o["cy"]), class_name))
            except Exception:
                continue

        new_validated = claimed_found_objects.intersection(true_target_objects)
        if new_validated != validated_found_objects:
            validated_found_objects = set(new_validated)

        remaining_objects = max(0, int(total_objects) - int(len(validated_found_objects)))

        if len(validated_found_objects) >= total_objects and total_objects > 0:
            final_reason = "done"
            final_time_s = float(t)
        elif elapsed_total >= TOTAL_SECONDS:
            final_reason = "timeout"
            final_time_s = float(t)

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
                now_time_s=float(t),
                report_written=report_written,
            )
            continue

        if not report_written:
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
            policy_name = controller_progress.get("policy_name", None) or "unknown_policy"
            autonomy_start_time_s = controller_progress.get("autonomy_start_time_s", None)
            visited_cells = {
                (int(cx), int(cy))
                for (cx, cy) in controller_progress.get("visited_cells", [])
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
                    spawned_objects=spawned_objects,
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
                report_written=True,
            )

            print(f"[MEDIUM SUPERVISOR] end-of-run state: {final_reason}", flush=True)

            grace_start = supervisor.getTime()
            while supervisor.step(timestep) != -1:
                if (supervisor.getTime() - grace_start) >= FINALIZATION_GRACE_SECONDS:
                    break
            break

    _cleanup_spawned_nodes(spawn_children)
    print("Cleanup complete. Quitting simulation.", flush=True)
    supervisor.simulationQuit(0)


if __name__ == "__main__":
    main()
