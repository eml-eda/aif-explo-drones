# controllers/worldgen_table_dataset_supervisor/worldgen_table_dataset_supervisor.py

from __future__ import annotations

import colorsys
import json
import math
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from controller import Supervisor

HERE = os.path.dirname(__file__)
SHARED_DIR = os.path.abspath(os.path.join(HERE, "..", "shared"))
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

from table_dataset_classes import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, MODEL_TO_CLASS_ID

# ---------- Run config ----------
BASE_SEED = int(os.environ.get("TABLE_DATASET_BASE_SEED", "1"))
NUM_RUNS = int(os.environ.get("TABLE_DATASET_NUM_RUNS", "6"))
SAMPLES_PER_RUN = int(os.environ.get("TABLE_DATASET_SAMPLES_PER_RUN", "400"))
DEFAULT_POLICY = os.environ.get("TABLE_DATASET_POLICY", os.environ.get("DATASET_POLICY", "pseudo_random"))

WARMUP_SECONDS = 15.0
RUN_TIMEOUT_SECONDS = float(os.environ.get("TABLE_DATASET_RUN_TIMEOUT_SECONDS", "12000"))
FINALIZATION_GRACE_SECONDS = 1.0

N_BOTTLES = int(os.environ.get("TABLE_DATASET_N_BOTTLES", "11"))
N_CANS = int(os.environ.get("TABLE_DATASET_N_CANS", "11"))

# ---------- Arena / grid ----------
ARENA_X = 5.5
ARENA_Y = 6.5
CELL_SIZE_M = 0.5
GRID_COLS = int(ARENA_X / CELL_SIZE_M)
GRID_ROWS = int(ARENA_Y / CELL_SIZE_M)
WALL_MARGIN = 0.35

SPAWN_CENTER_DISC_RADIUS_M = 0.15
DRONE_TARGET_EXCLUSION_RADIUS_M = 1.0
TARGET_MIN_SEPARATION_M = 0.22

# ---------- Desks ----------
N_DESKS = 5
DESK_LENGTH_M = 1.2
DESK_WIDTH_M = 0.7
DESK_HEIGHT_M = 0.72
DESK_TOP_THICKNESS_M = 0.08
DESK_RECOGNITION_HEIGHT_M = DESK_TOP_THICKNESS_M
DESK_WALL_MARGIN_M = 0.18
DESK_MIN_CENTER_DISTANCE_M = 1.8
DESK_DRONE_EXCLUSION_RADIUS_M = 1.25
DESK_HIGH_PROB_SHRINK_M = 0.05
DESK_EXCLUSION_MARGIN_M = 0.50
P_TARGET_ON_DESK = 0.60
DESK_TABLETOP_TEXTURE_URL = os.environ.get(
    "DESK_TABLETOP_TEXTURE_URL",
    "world_resources/wooden_tabletop.jpeg",
)

# ---------- Target wrappers ----------
BOTTLE_BBOX_RADIUS_M = 0.035
BOTTLE_BBOX_HEIGHT_M = 0.245
CAN_BBOX_RADIUS_M = 0.018
CAN_BBOX_HEIGHT_M = 0.125

OBJECT_SPECS = [
    {
        "class_name": "target_bottle",
        "class_id": CLASS_NAME_TO_ID["target_bottle"],
        "model": "target_bottle",
        "proto": "DatasetBeerBottle",
        "def_prefix": "TARGET_BOTTLE",
        "count": N_BOTTLES,
        "bbox_radius_m": BOTTLE_BBOX_RADIUS_M,
        "bbox_height_m": BOTTLE_BBOX_HEIGHT_M,
    },
    {
        "class_name": "target_can",
        "class_id": CLASS_NAME_TO_ID["target_can"],
        "model": "target_can",
        "proto": "DatasetCan",
        "def_prefix": "TARGET_CAN",
        "count": N_CANS,
        "bbox_radius_m": CAN_BBOX_RADIUS_M,
        "bbox_height_m": CAN_BBOX_HEIGHT_M,
    },
]

# ---------- Drone ----------
DRONE_DEF = "CRAZYFLIE"
DRONE_SPAWN_X = 0.0
DRONE_SPAWN_Y = 0.0
DRONE_SPAWN_YAW = 0.0

# ---------- Shared files ----------
EXPERIMENT_INIT_FILENAME = "table_dataset_experiment_init.json"
DATASET_OBJECTS_FILENAME = "table_dataset_objects.json"
TASK_STATE_FILENAME = "table_dataset_task_state.json"
COLLECTION_STATE_FILENAME = "table_dataset_collection_state.json"
RUN_STATE_FILENAME = "table_dataset_run_state.json"


Point = Tuple[float, float]


def dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _shared_path(filename: str) -> str:
    os.makedirs(SHARED_DIR, exist_ok=True)
    return os.path.join(SHARED_DIR, filename)


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


def _cxcy_from_xy(x: float, y: float) -> Optional[Tuple[int, int]]:
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
    rect = {"x": float(x), "y": float(y), "yaw": float(yaw), "length": float(length), "width": float(width)}
    corners = _rotated_rect_corners(rect)
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


def _eligible_floor_cells() -> List[Tuple[int, int]]:
    cells = []
    for cy in range(GRID_ROWS):
        for cx in range(GRID_COLS):
            x, y = _cell_center_xy(cx, cy)
            if _target_within_floor_bounds(x, y):
                cells.append((cx, cy))
    return cells


def _desk_high_prob_rect(desk: Dict[str, object]) -> Dict[str, object]:
    return _rect_record(
        float(desk["x"]),
        float(desk["y"]),
        float(desk["yaw"]),
        DESK_LENGTH_M - 2.0 * DESK_HIGH_PROB_SHRINK_M,
        DESK_WIDTH_M - 2.0 * DESK_HIGH_PROB_SHRINK_M,
    )


def _desk_exclusion_rect(desk: Dict[str, object]) -> Dict[str, object]:
    return _rect_record(
        float(desk["x"]),
        float(desk["y"]),
        float(desk["yaw"]),
        DESK_LENGTH_M - 2.0 * DESK_HIGH_PROB_SHRINK_M + 2.0 * DESK_EXCLUSION_MARGIN_M,
        DESK_WIDTH_M - 2.0 * DESK_HIGH_PROB_SHRINK_M + 2.0 * DESK_EXCLUSION_MARGIN_M,
    )


def _in_any_high_prob_zone(x: float, y: float, desks: List[Dict[str, object]]) -> bool:
    return any(_point_in_rotated_rect(x, y, desk["high_prob_rect"]) for desk in desks)


def _in_any_desk_exclusion_ring(x: float, y: float, desks: List[Dict[str, object]]) -> bool:
    for desk in desks:
        in_large = _point_in_rotated_rect(x, y, desk["exclusion_rect"])
        in_high = _point_in_rotated_rect(x, y, desk["high_prob_rect"])
        if in_large and not in_high:
            return True
    return False


def _sample_desks(drone_xy: Point, max_tries: int = 12000) -> List[Dict[str, object]]:
    desks: List[Dict[str, object]] = []
    for desk_id in range(N_DESKS):
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
                "def": f"TABLE_DATASET_DESK_{desk_id:03d}",
                "class_id": CLASS_NAME_TO_ID["desk"],
                "class_name": "desk",
                "model": "desk",
                "x": float(x),
                "y": float(y),
                "z": float(DESK_HEIGHT_M),
                "yaw": float(yaw),
                "dimensions_m": {
                    "length": DESK_LENGTH_M,
                    "width": DESK_WIDTH_M,
                    "height": DESK_HEIGHT_M,
                    "recognition_height": DESK_RECOGNITION_HEIGHT_M,
                },
                "tabletop_rect": tabletop_rect,
                "high_prob_rect": _desk_high_prob_rect({"x": x, "y": y, "yaw": yaw}),
                "exclusion_rect": _desk_exclusion_rect({"x": x, "y": y, "yaw": yaw}),
                "is_empty": False,
            }
            break
        if placed is None:
            raise RuntimeError(f"Could not place desk {desk_id} under table dataset constraints.")
        desks.append(placed)
    return desks


def _valid_on_desk_cells(desk: Dict[str, object]) -> List[Tuple[int, int]]:
    rect = desk["high_prob_rect"]
    cells = []
    for cy in range(GRID_ROWS):
        for cx in range(GRID_COLS):
            x, y = _cell_center_xy(cx, cy)
            if _point_in_rotated_rect(x, y, rect):
                cells.append((cx, cy))
    return cells


def _sample_target_on_desk(desk: Dict[str, object], max_tries: int = 2000) -> Optional[Tuple[float, float, int, int]]:
    cells = desk.get("valid_spawn_cells")
    if not cells:
        return None
    rect = desk["high_prob_rect"]
    for _ in range(max_tries):
        cx, cy = random.choice(cells)
        center_x, center_y = _cell_center_xy(cx, cy)
        x, y = _uniform_point_in_disc(center_x, center_y, SPAWN_CENTER_DISC_RADIUS_M)
        if _cxcy_from_xy(x, y) != (cx, cy):
            continue
        if not _point_in_rotated_rect(x, y, rect):
            continue
        return (x, y, cx, cy)
    return None


def _sample_target_off_desk(
    *,
    floor_cells: List[Tuple[int, int]],
    desks: List[Dict[str, object]],
    drone_xy: Point,
    occupied_xy: List[Point],
    max_tries: int = 5000,
) -> Optional[Tuple[float, float, int, int]]:
    for _ in range(max_tries):
        cx, cy = random.choice(floor_cells)
        center_x, center_y = _cell_center_xy(cx, cy)
        x, y = _uniform_point_in_disc(center_x, center_y, SPAWN_CENTER_DISC_RADIUS_M)
        if _cxcy_from_xy(x, y) != (cx, cy):
            continue
        if not _target_within_floor_bounds(x, y):
            continue
        if dist2((x, y), drone_xy) < DRONE_TARGET_EXCLUSION_RADIUS_M ** 2:
            continue
        if _in_any_high_prob_zone(x, y, desks) or _in_any_desk_exclusion_ring(x, y, desks):
            continue
        if any(dist2((x, y), p) < TARGET_MIN_SEPARATION_M ** 2 for p in occupied_xy):
            continue
        return (x, y, cx, cy)
    return None


def _desk_node_string(desk: Dict[str, object], color_rgb) -> str:
    r, g, b = color_rgb

    # DESK_HEIGHT_M is the tabletop top-surface height.
    # The Solid origin is at the tabletop center.
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

def _target_node_string(*, def_name: str, proto: str, model: str, x: float, y: float, z: float, color_rgb, bbox_radius_m: float, bbox_height_m: float) -> str:
    r, g, b = color_rgb

    # Important:
    # Do NOT wrap BeerBottle/Can inside another recognized Solid.
    # The rendered pixels belong to the PROTO's own root Solid, so recognitionColors
    # must be applied directly to that PROTO instance. Otherwise Webots may
    # recognize the wrapper while the segmentation image is colored by the child
    # object, making deterministic segmentation-mask labeling impossible.
    return f"""
DEF {def_name} {proto} {{
  translation {x:.6f} {y:.6f} {z:.6f}
  name "{def_name}"
  model "{model}"
  recognitionColors [
    {r:.6f} {g:.6f} {b:.6f}
  ]
}}
""".strip()


def _place_drone(supervisor: Supervisor, drone_node, *, x: float, y: float, yaw: float) -> None:
    if drone_node is None:
        raise RuntimeError(f"Drone DEF '{DRONE_DEF}' not found.")
    translation_field = drone_node.getField("translation")
    rotation_field = drone_node.getField("rotation")
    current_translation = translation_field.getSFVec3f()
    translation_field.setSFVec3f([float(x), float(y), float(current_translation[2])])
    rotation_field.setSFRotation([0.0, 0.0, 1.0, float(yaw)])
    drone_node.resetPhysics()
    supervisor.simulationResetPhysics()


def _cleanup_spawned_nodes(spawn_children) -> None:
    for idx in range(spawn_children.getCount() - 1, -1, -1):
        spawn_children.getMFNode(idx).remove()


def _init_or_resume_run_state(path: str, seeds: List[int], policy_name: str) -> Dict[str, object]:
    existing = _read_json(path, None)
    if isinstance(existing, dict) and existing.get("completed") is not True:
        if existing.get("seeds") == seeds and existing.get("policy_name") == policy_name:
            return existing
    collection_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "policy_name": policy_name,
        "collection_start_time": collection_start_time,
        "collection_name": f"collection_table_{policy_name}_{collection_start_time}",
        "seeds": seeds,
        "total_runs": len(seeds),
        "current_run_index": 0,
        "current_seed": int(seeds[0]),
        "runs": [],
        "completed": False,
    }


def _write_task_state(path: str, *, status: str, seed: int, run_index: int, now_time_s: float, reason: Optional[str] = None) -> None:
    _atomic_write_json(
        path,
        {
            "status": str(status),
            "reason": reason,
            "seed": int(seed),
            "run_index": int(run_index),
            "now_time_s": float(now_time_s),
            "warmup_s": float(WARMUP_SECONDS),
            "timeout_s": float(RUN_TIMEOUT_SECONDS),
            "samples_per_run": int(SAMPLES_PER_RUN),
        },
    )


def main() -> None:
    if not (0.0 < SPAWN_CENTER_DISC_RADIUS_M < 0.5 * CELL_SIZE_M):
        raise ValueError("SPAWN_CENTER_DISC_RADIUS_M must keep samples inside their selected grid cell.")

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    policy_name = DEFAULT_POLICY

    seeds = [BASE_SEED + i for i in range(NUM_RUNS)]
    run_state_path = _shared_path(RUN_STATE_FILENAME)
    experiment_init_path = _shared_path(EXPERIMENT_INIT_FILENAME)
    dataset_objects_path = _shared_path(DATASET_OBJECTS_FILENAME)
    task_state_path = _shared_path(TASK_STATE_FILENAME)
    collection_state_path = _shared_path(COLLECTION_STATE_FILENAME)

    _safe_remove(experiment_init_path)
    _safe_remove(dataset_objects_path)
    _safe_remove(task_state_path)
    _safe_remove(collection_state_path)

    run_state = _init_or_resume_run_state(run_state_path, seeds, policy_name)
    run_index = int(run_state["current_run_index"])
    current_seed = int(run_state["current_seed"])
    total_runs = int(run_state["total_runs"])
    random.seed(current_seed)

    _atomic_write_json(run_state_path, run_state)
    _write_task_state(task_state_path, status="running", seed=current_seed, run_index=run_index, now_time_s=0.0)

    spawn_root = supervisor.getFromDef("SPAWN_ROOT")
    if spawn_root is None:
        print("ERROR: DEF SPAWN_ROOT Group not found in the world.", flush=True)
        return
    spawn_children = spawn_root.getField("children")

    drone_xy = (DRONE_SPAWN_X, DRONE_SPAWN_Y)
    _place_drone(supervisor, supervisor.getFromDef(DRONE_DEF), x=drone_xy[0], y=drone_xy[1], yaw=DRONE_SPAWN_YAW)

    desks = _sample_desks(drone_xy)
    empty_desk_id = random.choice([int(d["desk_id"]) for d in desks])
    for desk in desks:
        desk["is_empty"] = int(desk["desk_id"]) == empty_desk_id
        desk["valid_spawn_cells"] = _valid_on_desk_cells(desk)

    non_empty_desks = [d for d in desks if not d["is_empty"]]
    if not all(d.get("valid_spawn_cells") for d in non_empty_desks):
        raise RuntimeError("At least one non-empty desk has no valid center-disc spawn cells.")

    objects_manifest: List[Dict[str, object]] = []
    global_index = 0

    for desk in desks:
        color_rgb = _instance_color(global_index)
        spawn_children.importMFNodeFromString(-1, _desk_node_string(desk, color_rgb))
        node = spawn_children.getMFNode(spawn_children.getCount() - 1)
        desk["id"] = _node_id(node)
        spawn_children.importMFNodeFromString(-1, _desk_legs_node_string(desk))
        desk["recognition_color_rgb"] = [float(v) for v in color_rgb]
        objects_manifest.append(dict(desk))
        global_index += 1

    floor_cells = _eligible_floor_cells()
    occupied_xy: List[Point] = []
    target_records: List[Dict[str, object]] = []

    for spec in OBJECT_SPECS:
        for local_i in range(int(spec["count"])):
            spawn_mode = "on_desk" if random.random() < P_TARGET_ON_DESK else "off_desk"
            desk_for_target = None
            sampled = None
            z = 0.0

            if spawn_mode == "on_desk":
                for _ in range(200):
                    desk_for_target = random.choice(non_empty_desks)
                    sampled = _sample_target_on_desk(desk_for_target)
                    if sampled is None:
                        continue
                    x, y, _, _ = sampled
                    if any(dist2((x, y), p) < TARGET_MIN_SEPARATION_M ** 2 for p in occupied_xy):
                        sampled = None
                        continue
                    z = DESK_HEIGHT_M
                    break

            if sampled is None:
                spawn_mode = "off_desk"
                desk_for_target = None
                sampled = _sample_target_off_desk(
                    floor_cells=floor_cells,
                    desks=desks,
                    drone_xy=drone_xy,
                    occupied_xy=occupied_xy,
                )
                z = 0.0

            if sampled is None:
                print(f"WARNING: Could not place {spec['class_name']} #{local_i}.", flush=True)
                continue

            x, y, cx, cy = sampled
            occupied_xy.append((x, y))
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
                bbox_radius_m=float(spec["bbox_radius_m"]),
                bbox_height_m=float(spec["bbox_height_m"]),
            )
            spawn_children.importMFNodeFromString(-1, node_string)
            node = spawn_children.getMFNode(spawn_children.getCount() - 1)

            target_record = {
                "def": def_name,
                "id": _node_id(node),
                "class_id": int(spec["class_id"]),
                "class_name": str(spec["class_name"]),
                "model": str(spec["model"]),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "cx": int(cx),
                "cy": int(cy),
                "spawn_mode": spawn_mode,
                "desk_id": None if desk_for_target is None else int(desk_for_target["desk_id"]),
                "recognition_color_rgb": [float(v) for v in color_rgb],
                "grid_cell_center_xy": [float(v) for v in _cell_center_xy(cx, cy)],
                "spawn_center_disc_radius_m": float(SPAWN_CENTER_DISC_RADIUS_M),
            }
            target_records.append(target_record)
            objects_manifest.append(target_record)
            global_index += 1

    target_counts = {
        name: sum(1 for r in target_records if r["class_name"] == name)
        for name in ("target_bottle", "target_can")
    }
    on_desk_count = sum(1 for r in target_records if r["spawn_mode"] == "on_desk")
    off_desk_count = sum(1 for r in target_records if r["spawn_mode"] == "off_desk")

    print(
        f"[TABLE DATASET] run {run_index + 1}/{total_runs} seed={current_seed} "
        f"desks={len(desks)} empty_desk={empty_desk_id} targets={len(target_records)} "
        f"on_desk={on_desk_count} off_desk={off_desk_count}",
        flush=True,
    )

    experiment_init = {
        "seed": int(current_seed),
        "run_index": int(run_index),
        "total_runs": int(total_runs),
        "collection_name": run_state["collection_name"],
        "collection_start_time": run_state["collection_start_time"],
        "policy_name": policy_name,
        "samples_per_run": int(SAMPLES_PER_RUN),
        "drone_spawn": {
            "x": float(drone_xy[0]),
            "y": float(drone_xy[1]),
            "yaw": float(DRONE_SPAWN_YAW),
            "cx": _cxcy_from_xy(drone_xy[0], drone_xy[1])[0],
            "cy": _cxcy_from_xy(drone_xy[0], drone_xy[1])[1],
        },
    }
    _atomic_write_json(experiment_init_path, experiment_init)

    metadata = {
        "seed": int(current_seed),
        "run_index": int(run_index),
        "room_dimensions_m": {"x": ARENA_X, "y": ARENA_Y},
        "grid": {"cols": GRID_COLS, "rows": GRID_ROWS, "cell_size_m": CELL_SIZE_M},
        "drone_spawn": experiment_init["drone_spawn"],
        "target_class_counts": target_counts,
        "target_on_desk_probability": P_TARGET_ON_DESK,
        "target_spawn_counts": {"on_desk": on_desk_count, "off_desk": off_desk_count},
        "target_spawn_constraints": {
            "center_disc_radius_m": SPAWN_CENTER_DISC_RADIUS_M,
            "drone_exclusion_radius_m": DRONE_TARGET_EXCLUSION_RADIUS_M,
            "min_target_separation_m": TARGET_MIN_SEPARATION_M,
        },
        "desk_dimensions_m": {
            "length": DESK_LENGTH_M,
            "width": DESK_WIDTH_M,
            "height": DESK_HEIGHT_M,
            "recognition_height": DESK_RECOGNITION_HEIGHT_M,
            "high_prob_shrink_m": DESK_HIGH_PROB_SHRINK_M,
            "exclusion_margin_m": DESK_EXCLUSION_MARGIN_M,
        },
        "desk_spawn_constraints": {
            "wall_margin_m": DESK_WALL_MARGIN_M,
            "min_center_distance_m": DESK_MIN_CENTER_DISTANCE_M,
            "drone_exclusion_radius_m": DESK_DRONE_EXCLUSION_RADIUS_M,
        },
        "empty_desk_id": int(empty_desk_id),
        "desks": desks,
        "targets": target_records,
        "class_id_to_name": CLASS_ID_TO_NAME,
        "model_to_class_id": MODEL_TO_CLASS_ID,
    }
    _atomic_write_json(dataset_objects_path, {"seed": int(current_seed), "objects": objects_manifest, "metadata": metadata})

    start_time = supervisor.getTime()
    final_reason = None
    final_time_s = None

    while supervisor.step(timestep) != -1:
        t = supervisor.getTime()
        elapsed = float(t - start_time)
        collection_state = _read_json(collection_state_path, {})
        if (
            collection_state.get("status") == "complete"
            and int(collection_state.get("seed", -999999)) == current_seed
            and int(collection_state.get("run_index", -1)) == run_index
        ):
            final_reason = "complete"
            final_time_s = float(t)
        elif elapsed >= RUN_TIMEOUT_SECONDS:
            final_reason = "timeout"
            final_time_s = float(t)

        if final_reason is None:
            _write_task_state(task_state_path, status="running", seed=current_seed, run_index=run_index, now_time_s=float(t))
            continue

        _write_task_state(
            task_state_path,
            status=final_reason,
            reason=final_reason,
            seed=current_seed,
            run_index=run_index,
            now_time_s=float(final_time_s),
        )

        run_record = {
            "run_index": int(run_index),
            "seed": int(current_seed),
            "reason": str(final_reason),
            "time_s": float(final_time_s),
            "num_examples_collected": int(collection_state.get("num_examples_collected", 0)),
            "dataset_folder": collection_state.get("dataset_folder"),
            "empty_desk_id": int(empty_desk_id),
            "target_counts": target_counts,
            "target_spawn_counts": {"on_desk": on_desk_count, "off_desk": off_desk_count},
        }
        run_state["runs"] = list(run_state.get("runs", [])) + [run_record]

        is_last = (run_index + 1) >= total_runs
        if is_last:
            run_state["completed"] = True
            _atomic_write_json(run_state_path, run_state)
            print(f"[TABLE DATASET] completed all {total_runs} runs.", flush=True)
            grace_start = supervisor.getTime()
            while supervisor.step(timestep) != -1:
                if supervisor.getTime() - grace_start >= FINALIZATION_GRACE_SECONDS:
                    break
            _cleanup_spawned_nodes(spawn_children)
            supervisor.simulationQuit(0)
            return

        next_run_index = run_index + 1
        run_state["current_run_index"] = int(next_run_index)
        run_state["current_seed"] = int(seeds[next_run_index])
        run_state["completed"] = False
        _atomic_write_json(run_state_path, run_state)
        print(
            f"[TABLE DATASET] finished run {run_index + 1}/{total_runs}; "
            f"reloading for seed={seeds[next_run_index]}",
            flush=True,
        )
        grace_start = supervisor.getTime()
        while supervisor.step(timestep) != -1:
            if supervisor.getTime() - grace_start >= FINALIZATION_GRACE_SECONDS:
                break
        _cleanup_spawned_nodes(spawn_children)
        supervisor.worldReload()
        return


if __name__ == "__main__":
    main()