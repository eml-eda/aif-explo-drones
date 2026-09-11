# controllers/worldgen_hard_dataset_supervisor/worldgen_hard_dataset_supervisor.py

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
CONTROLLERS_DIR = os.path.abspath(os.path.join(HERE, ".."))
SHARED_DIR = os.path.join(CONTROLLERS_DIR, "shared")
for path in (CONTROLLERS_DIR, SHARED_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset_collection.class_config import get_dataset_class_config

_CLASS_CONFIG = get_dataset_class_config("hard")
CLASS_ID_TO_NAME = _CLASS_CONFIG.class_id_to_name
CLASS_NAME_TO_ID = _CLASS_CONFIG.class_name_to_id
MODEL_TO_CLASS_ID = _CLASS_CONFIG.model_to_class_id

# ---------- Dataset run configuration ----------
# ``hard_dataset`` preserves the existing reference collection scene. The two
# collection-only variants are used to train the final hard detector on visual
# domains that are deliberately different from the hard evaluation world.
SCENE_VARIANT = str(os.environ.get("HARD_DATASET_SCENE_VARIANT", "hard_dataset")).strip().lower()
_VALID_SCENE_VARIANTS = {"hard_dataset", "hard_collection_a", "hard_collection_b"}
if SCENE_VARIANT not in _VALID_SCENE_VARIANTS:
    raise ValueError(
        f"Unsupported HARD_DATASET_SCENE_VARIANT={SCENE_VARIANT!r}; "
        f"expected one of {sorted(_VALID_SCENE_VARIANTS)}"
    )

BASE_SEED = int(os.environ.get("HARD_DATASET_BASE_SEED", "1"))
NUM_RUNS = int(os.environ.get("HARD_DATASET_NUM_RUNS", "6"))
_DEFAULT_SAMPLES_PER_RUN = 200 if SCENE_VARIANT in {"hard_collection_a", "hard_collection_b"} else 400
SAMPLES_PER_RUN = int(os.environ.get("HARD_DATASET_SAMPLES_PER_RUN", str(_DEFAULT_SAMPLES_PER_RUN)))
DEFAULT_POLICY = os.environ.get(
    "HARD_DATASET_POLICY",
    os.environ.get("DATASET_POLICY", "pseudo_random"),
)

_COLLECTION_PREFIX = {
    "hard_dataset": "collection_hard",
    "hard_collection_a": "collection_hard_a",
    "hard_collection_b": "collection_hard_b",
}[SCENE_VARIANT]
_LOG_TAG = {
    "hard_dataset": "HARD DATASET",
    "hard_collection_a": "HARD A DATASET",
    "hard_collection_b": "HARD B DATASET",
}[SCENE_VARIANT]

WARMUP_SECONDS = 15.0
RUN_TIMEOUT_SECONDS = float(os.environ.get("HARD_DATASET_RUN_TIMEOUT_SECONDS", "12000"))
FINALIZATION_GRACE_SECONDS = 1.0

# ---------- Hard-dataset scene density ----------
# All hard collection variants use the same labelled classes, target/desk
# density, placement model, and tabletop geometry. The A/B collection-only
# variants deliberately change floor appearance and distractor identities while
# leaving the search geometry unchanged.
N_BOTTLES = int(os.environ.get("HARD_DATASET_N_BOTTLES", "10"))
N_CANS = int(os.environ.get("HARD_DATASET_N_CANS", "10"))
N_DESKS = int(os.environ.get("HARD_DATASET_N_DESKS", "4"))

# Reference-scene distractors.
N_SOCCER_BALLS = int(os.environ.get("HARD_DATASET_N_SOCCER_BALLS", "1"))
N_CEREAL_BOXES = int(os.environ.get("HARD_DATASET_N_CEREAL_BOXES", "2"))
N_PLATES = int(os.environ.get("HARD_DATASET_N_PLATES", "3"))
N_FLOWER_POTS = int(os.environ.get("HARD_DATASET_N_FLOWER_POTS", "2"))

# Hard collection A distractors.
N_RUBBER_DUCKS = int(os.environ.get("HARD_DATASET_N_RUBBER_DUCKS", "3"))
N_GLASSES = int(os.environ.get("HARD_DATASET_N_GLASSES", "3"))
N_APPLES = int(os.environ.get("HARD_DATASET_N_APPLES", "3"))
N_BISCUIT_BOXES = int(os.environ.get("HARD_DATASET_N_BISCUIT_BOXES", "3"))

# Hard collection B distractors.
N_CARAFES = int(os.environ.get("HARD_DATASET_N_CARAFES", "3"))
N_HONEY_JARS = int(os.environ.get("HARD_DATASET_N_HONEY_JARS", "3"))
N_ORANGES = int(os.environ.get("HARD_DATASET_N_ORANGES", "3"))
N_WOODEN_SPOONS = int(os.environ.get("HARD_DATASET_N_WOODEN_SPOONS", "3"))

P_TARGET_ON_DESK = float(os.environ.get("HARD_DATASET_P_TARGET_ON_DESK", "0.90"))
P_OPTIONAL_DISTRACTOR_ON_DESK = float(
    os.environ.get("HARD_DATASET_P_OPTIONAL_DISTRACTOR_ON_DESK", "0.50")
)

# ---------- Arena / grid ----------
ARENA_X = 5.5
ARENA_Y = 6.5
CELL_SIZE_M = 0.5
GRID_COLS = int(ARENA_X / CELL_SIZE_M)
GRID_ROWS = int(ARENA_Y / CELL_SIZE_M)
WALL_MARGIN = 0.35

SPAWN_CENTER_DISC_RADIUS_M = float(
    os.environ.get("HARD_DATASET_SPAWN_CENTER_DISC_RADIUS_M", "0.15")
)
DRONE_TARGET_EXCLUSION_RADIUS_M = float(
    os.environ.get("HARD_DATASET_DRONE_TARGET_EXCLUSION_RADIUS_M", "1.0")
)
TARGET_MIN_SEPARATION_M = float(
    os.environ.get("HARD_DATASET_MIN_OBJECT_SEPARATION_M", "0.22")
)

# ---------- Desks ----------
DESK_LENGTH_M = 1.2
DESK_WIDTH_M = 0.7
DESK_HEIGHT_M = 0.72
DESK_TOP_THICKNESS_M = 0.08
DESK_RECOGNITION_HEIGHT_M = DESK_TOP_THICKNESS_M
DESK_WALL_MARGIN_M = 0.18
DESK_MIN_CENTER_DISTANCE_M = 1.35
DESK_DRONE_EXCLUSION_RADIUS_M = 1.25
DESK_SPAWN_EDGE_MARGIN_M = 0.05
DESK_FLOOR_EXCLUSION_MARGIN_M = 0.50
DESK_TABLETOP_TEXTURE_URL = os.environ.get(
    "HARD_DATASET_DESK_TABLETOP_TEXTURE_URL",
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
        "def_prefix": "HARD_DATASET_BOTTLE",
        "count": N_BOTTLES,
        "bbox_radius_m": BOTTLE_BBOX_RADIUS_M,
        "bbox_height_m": BOTTLE_BBOX_HEIGHT_M,
    },
    {
        "class_name": "target_can",
        "class_id": CLASS_NAME_TO_ID["target_can"],
        "model": "target_can",
        "proto": "DatasetCan",
        "def_prefix": "HARD_DATASET_CAN",
        "count": N_CANS,
        "bbox_radius_m": CAN_BBOX_RADIUS_M,
        "bbox_height_m": CAN_BBOX_HEIGHT_M,
    },
]

# Standard imported Webots PROTOs. They are visually present in the collected
# images but intentionally absent from the shared hard class schema, so they never
# become YOLO annotations.
DISTRACTOR_SPECS_BY_VARIANT = {
    "hard_dataset": [
        {
            "count": N_SOCCER_BALLS,
            "proto": "SoccerBall",
            "class_name": "distractor_soccer_ball",
            "model": "soccer ball",
            "def_prefix": "HARD_DATASET_SOCCER_BALL",
            "z_floor_m": 0.113,
            "z_table_offset_m": 0.113,
            "placement": "optional_desk",
        },
        {
            "count": N_CEREAL_BOXES,
            "proto": "CerealBox",
            "class_name": "distractor_cereal_box",
            "model": "cereal box",
            "def_prefix": "HARD_DATASET_CEREAL_BOX",
            "z_floor_m": 0.0,
            "z_table_offset_m": 0.0,
            "placement": "optional_desk",
        },
        {
            "count": N_PLATES,
            "proto": "Plate",
            "class_name": "distractor_plate",
            "model": "plate",
            "def_prefix": "HARD_DATASET_PLATE",
            "z_floor_m": 0.0,
            "z_table_offset_m": 0.0,
            "placement": "desk_only",
        },
        {
            "count": N_FLOWER_POTS,
            "proto": "FlowerPot",
            "class_name": "distractor_flower_pot",
            "model": "flower pot",
            "def_prefix": "HARD_DATASET_FLOWER_POT",
            "z_floor_m": 0.0,
            "z_table_offset_m": 0.0,
            "placement": "optional_desk",
        },
    ],
    "hard_collection_a": [
        {
            "count": N_RUBBER_DUCKS,
            "proto": "RubberDuck",
            "class_name": "distractor_rubber_duck",
            "model": "rubber duck",
            "def_prefix": "HARD_A_RUBBER_DUCK",
            "z_floor_m": 0.02,
            "z_table_offset_m": 0.02,
            "placement": "optional_desk",
        },
        {
            "count": N_GLASSES,
            "proto": "Glass",
            "class_name": "distractor_glass",
            "model": "glass",
            "def_prefix": "HARD_A_GLASS",
            "z_floor_m": 0.0,
            "z_table_offset_m": 0.0,
            "placement": "desk_only",
        },
        {
            "count": N_APPLES,
            "proto": "Apple",
            "class_name": "distractor_apple",
            "model": "apple",
            "def_prefix": "HARD_A_APPLE",
            "z_floor_m": 0.05,
            "z_table_offset_m": 0.05,
            "placement": "optional_desk",
        },
        {
            "count": N_BISCUIT_BOXES,
            "proto": "BiscuitBox",
            "class_name": "distractor_biscuit_box",
            "model": "biscuit box",
            "def_prefix": "HARD_A_BISCUIT_BOX",
            "z_floor_m": 0.0,
            "z_table_offset_m": 0.0,
            "placement": "optional_desk",
        },
    ],
    "hard_collection_b": [
        {
            "count": N_CARAFES,
            "proto": "Carafe",
            "class_name": "distractor_carafe",
            "model": "carafe",
            "def_prefix": "HARD_B_CARAFE",
            "z_floor_m": 0.0,
            "z_table_offset_m": 0.0,
            "placement": "optional_desk",
        },
        {
            "count": N_HONEY_JARS,
            "proto": "HoneyJar",
            "class_name": "distractor_honey_jar",
            "model": "honey jar",
            "def_prefix": "HARD_B_HONEY_JAR",
            "z_floor_m": 0.0,
            "z_table_offset_m": 0.0,
            "placement": "desk_only",
        },
        {
            "count": N_ORANGES,
            "proto": "Orange",
            "class_name": "distractor_orange",
            "model": "orange",
            "def_prefix": "HARD_B_ORANGE",
            "z_floor_m": 0.05,
            "z_table_offset_m": 0.05,
            "placement": "optional_desk",
        },
        {
            "count": N_WOODEN_SPOONS,
            "proto": "WoodenSpoon",
            "class_name": "distractor_wooden_spoon",
            "model": "wooden spoon",
            "def_prefix": "HARD_B_WOODEN_SPOON",
            "z_floor_m": 0.0,
            "z_table_offset_m": 0.0,
            "placement": "optional_desk",
        },
    ],
}

DISTRACTOR_SPECS = DISTRACTOR_SPECS_BY_VARIANT[SCENE_VARIANT]

# ---------- Drone ----------
DRONE_DEF = "CRAZYFLIE"
DRONE_SPAWN_X_MIN = -2.5
DRONE_SPAWN_X_MAX = 2.5
DRONE_SPAWN_Y_MIN = -3.0
DRONE_SPAWN_Y_MAX = 3.0
DRONE_SPAWN_YAW = 0.0

# ---------- Shared files ----------
EXPERIMENT_INIT_FILENAME = "hard_dataset_experiment_init.json"
DATASET_OBJECTS_FILENAME = "hard_dataset_objects.json"
TASK_STATE_FILENAME = "hard_dataset_task_state.json"
COLLECTION_STATE_FILENAME = "hard_dataset_collection_state.json"
RUN_STATE_FILENAME = {
    "hard_dataset": "hard_dataset_run_state.json",
    "hard_collection_a": "hard_a_dataset_run_state.json",
    "hard_collection_b": "hard_b_dataset_run_state.json",
}[SCENE_VARIANT]

Point = Tuple[float, float]
Cell = Tuple[int, int]


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


def _cxcy_from_xy(x: float, y: float) -> Optional[Cell]:
    min_x = -ARENA_X / 2.0
    min_y = -ARENA_Y / 2.0
    if x < min_x or x >= ARENA_X / 2.0 or y < min_y or y >= ARENA_Y / 2.0:
        return None
    return (int((x - min_x) / CELL_SIZE_M), int((y - min_y) / CELL_SIZE_M))


def _cell_center_xy(cx: int, cy: int) -> Point:
    return (
        -ARENA_X / 2.0 + (float(cx) + 0.5) * CELL_SIZE_M,
        -ARENA_Y / 2.0 + (float(cy) + 0.5) * CELL_SIZE_M,
    )


def _uniform_point_in_disc(center_x: float, center_y: float, radius: float) -> Point:
    theta = random.uniform(0.0, 2.0 * math.pi)
    radius_sample = radius * math.sqrt(random.random())
    return (
        center_x + radius_sample * math.cos(theta),
        center_y + radius_sample * math.sin(theta),
    )


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
    rect: Dict[str, object] = {
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
    return all(min_x <= x <= max_x and min_y <= y <= max_y for x, y in _rotated_rect_corners(rect))


def _object_within_floor_bounds(x: float, y: float) -> bool:
    return (
        -ARENA_X / 2.0 + WALL_MARGIN <= x <= ARENA_X / 2.0 - WALL_MARGIN
        and -ARENA_Y / 2.0 + WALL_MARGIN <= y <= ARENA_Y / 2.0 - WALL_MARGIN
    )


def _eligible_floor_cells() -> List[Cell]:
    cells: List[Cell] = []
    for cy in range(GRID_ROWS):
        for cx in range(GRID_COLS):
            x, y = _cell_center_xy(cx, cy)
            if _object_within_floor_bounds(x, y):
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


def _sample_drone_spawn_xy() -> Point:
    return (
        random.uniform(DRONE_SPAWN_X_MIN, DRONE_SPAWN_X_MAX),
        random.uniform(DRONE_SPAWN_Y_MIN, DRONE_SPAWN_Y_MAX),
    )


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
                "def": f"HARD_DATASET_DESK_{desk_id:03d}",
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
                "is_empty": False,
            }
            placed["spawn_rect"] = _desk_spawn_rect(placed)
            placed["floor_exclusion_rect"] = _desk_floor_exclusion_rect(placed)
            break
        if placed is None:
            raise RuntimeError(f"Could not place hard-dataset desk {desk_id} under current constraints.")
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


def _sample_object_on_desk(
    desk: Dict[str, object],
    occupied_xy: List[Point],
    *,
    max_tries: int = 2000,
) -> Optional[Tuple[float, float, int, int]]:
    cells = desk.get("valid_spawn_cells")
    if not cells:
        return None
    rect = desk["spawn_rect"]
    for _ in range(max_tries):
        cx, cy = random.choice(cells)
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


def _sample_object_on_any_desk(
    desks: List[Dict[str, object]],
    occupied_xy: List[Point],
    *,
    max_tries: int = 5000,
) -> Optional[Tuple[float, float, int, int, Dict[str, object]]]:
    for _ in range(max_tries):
        desk = random.choice(desks)
        sampled = _sample_object_on_desk(desk, occupied_xy, max_tries=1)
        if sampled is None:
            continue
        x, y, cx, cy = sampled
        return (x, y, cx, cy, desk)
    return None


def _sample_object_off_desk(
    *,
    floor_cells: List[Cell],
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
        if not _object_within_floor_bounds(x, y):
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


def _desk_node_string(desk: Dict[str, object], color_rgb: tuple[float, float, float]) -> str:
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
    color_rgb: tuple[float, float, float],
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


def _distractor_node_string(
    *,
    def_name: str,
    proto: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
) -> str:
    return f'''
DEF {def_name} {proto} {{
  translation {x:.6f} {y:.6f} {z:.6f}
  rotation 0 0 1 {yaw:.6f}
  name "{def_name}"
}}
'''.strip()


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


def _init_or_resume_run_state(
    path: str,
    seeds: List[int],
    policy_name: str,
    scene_variant: str,
) -> Dict[str, object]:
    existing = _read_json(path, None)
    if isinstance(existing, dict) and existing.get("completed") is not True:
        if (
            existing.get("seeds") == seeds
            and existing.get("policy_name") == policy_name
            and existing.get("scene_variant") == scene_variant
        ):
            return existing

    collection_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "policy_name": policy_name,
        "scene_variant": scene_variant,
        "collection_start_time": collection_start_time,
        "collection_name": f"{_COLLECTION_PREFIX}_{policy_name}_{collection_start_time}",
        "seeds": seeds,
        "total_runs": len(seeds),
        "current_run_index": 0,
        "current_seed": int(seeds[0]),
        "runs": [],
        "completed": False,
    }


def _write_task_state(
    path: str,
    *,
    status: str,
    seed: int,
    run_index: int,
    now_time_s: float,
    reason: Optional[str] = None,
) -> None:
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


def _spawn_hard_dataset_scene(
    spawn_children,
    *,
    drone_xy: Point,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Spawn a visually hard scene while exposing labels only for 3 classes."""
    desks = _sample_desks(drone_xy)
    for desk in desks:
        desk["valid_spawn_cells"] = _valid_on_desk_cells(desk)
        if not desk["valid_spawn_cells"]:
            raise RuntimeError(f"Desk {desk['desk_id']} has no valid spawn cells.")

    objects_manifest: List[Dict[str, object]] = []
    target_records: List[Dict[str, object]] = []
    distractor_records: List[Dict[str, object]] = []
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
    if not floor_cells:
        raise RuntimeError("No eligible floor cells available under the hard-dataset wall margin.")

    occupied_xy: List[Point] = []

    def place_distractor(spec: Dict[str, object], *, force_desk: bool) -> Dict[str, object]:
        nonlocal global_index
        prefer_desk = force_desk or random.random() < P_OPTIONAL_DISTRACTOR_ON_DESK
        sampled = None
        spawn_mode = "on_desk" if prefer_desk else "off_desk"
        desk_for_object = None
        z = 0.0

        if prefer_desk:
            sampled_on = _sample_object_on_any_desk(desks, occupied_xy)
            if sampled_on is not None:
                x, y, cx, cy, desk_for_object = sampled_on
                sampled = (x, y, cx, cy)
                z = DESK_HEIGHT_M + float(spec["z_table_offset_m"])
            elif not force_desk:
                spawn_mode = "off_desk"

        if sampled is None and not force_desk:
            sampled_off = _sample_object_off_desk(
                floor_cells=floor_cells,
                desks=desks,
                drone_xy=drone_xy,
                occupied_xy=occupied_xy,
            )
            if sampled_off is not None:
                x, y, cx, cy = sampled_off
                sampled = (x, y, cx, cy)
                z = float(spec["z_floor_m"])
                spawn_mode = "off_desk"

        if sampled is None:
            raise RuntimeError(
                f"Could not place distractor {spec['class_name']} under hard-dataset constraints."
            )

        x, y, cx, cy = sampled
        occupied_xy.append((float(x), float(y)))
        def_name = f"{spec['def_prefix']}_{global_index:03d}"
        yaw = random.uniform(-math.pi, math.pi)
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
        global_index += 1
        return {
            "def": def_name,
            "id": _node_id(node),
            "class_id": None,
            "class_name": str(spec["class_name"]),
            "model": str(spec["model"]),
            "proto": str(spec["proto"]),
            "is_distractor": True,
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "yaw": float(yaw),
            "cx": int(cx),
            "cy": int(cy),
            "spawn_mode": spawn_mode,
            "desk_id": None if desk_for_object is None else int(desk_for_object["desk_id"]),
            "grid_cell_center_xy": [float(v) for v in _cell_center_xy(cx, cy)],
            "spawn_center_disc_radius_m": float(SPAWN_CENTER_DISC_RADIUS_M),
        }

    # Desk-only distractors are placed first so they can never lose table capacity
    # to the denser target set.
    for spec in DISTRACTOR_SPECS:
        if str(spec["placement"]) == "desk_only":
            for _ in range(int(spec["count"])):
                record = place_distractor(spec, force_desk=True)
                distractor_records.append(record)
                objects_manifest.append(record)

    for spec in OBJECT_SPECS:
        for local_i in range(int(spec["count"])):
            prefer_desk = random.random() < P_TARGET_ON_DESK
            sampled = None
            spawn_mode = "on_desk" if prefer_desk else "off_desk"
            desk_for_target = None
            z = 0.0

            if prefer_desk:
                sampled_on = _sample_object_on_any_desk(desks, occupied_xy)
                if sampled_on is not None:
                    x, y, cx, cy, desk_for_target = sampled_on
                    sampled = (x, y, cx, cy)
                    z = DESK_HEIGHT_M
                else:
                    spawn_mode = "off_desk"

            if sampled is None:
                sampled_off = _sample_object_off_desk(
                    floor_cells=floor_cells,
                    desks=desks,
                    drone_xy=drone_xy,
                    occupied_xy=occupied_xy,
                )
                if sampled_off is not None:
                    x, y, cx, cy = sampled_off
                    sampled = (x, y, cx, cy)
                    spawn_mode = "off_desk"
                    z = 0.0

            if sampled is None:
                raise RuntimeError(
                    f"Could not place {spec['class_name']} #{local_i} under hard-dataset constraints."
                )

            x, y, cx, cy = sampled
            occupied_xy.append((float(x), float(y)))
            color_rgb = _instance_color(global_index)
            def_name = f"{spec['def_prefix']}_{global_index:03d}"
            spawn_children.importMFNodeFromString(
                -1,
                _target_node_string(
                    def_name=def_name,
                    proto=str(spec["proto"]),
                    model=str(spec["model"]),
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    color_rgb=color_rgb,
                ),
            )
            node = spawn_children.getMFNode(spawn_children.getCount() - 1)
            target_record = {
                "def": def_name,
                "id": _node_id(node),
                "class_id": int(spec["class_id"]),
                "class_name": str(spec["class_name"]),
                "model": str(spec["model"]),
                "proto": str(spec["proto"]),
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

    for spec in DISTRACTOR_SPECS:
        if str(spec["placement"]) != "desk_only":
            for _ in range(int(spec["count"])):
                record = place_distractor(spec, force_desk=False)
                distractor_records.append(record)
                objects_manifest.append(record)

    target_desk_ids = {
        int(record["desk_id"])
        for record in target_records
        if record.get("desk_id") is not None
    }
    desk_empty_by_id = {}
    for desk in desks:
        desk["is_empty"] = int(desk["desk_id"]) not in target_desk_ids
        desk_empty_by_id[int(desk["desk_id"])] = bool(desk["is_empty"])

    # Desks were inserted into the manifest before targets were sampled. Mirror
    # their final empty/non-empty status back into those copied records so the
    # segmentation-label collector can preserve this optional metadata.
    for record in objects_manifest:
        if str(record.get("class_name", "")) == "desk" and "desk_id" in record:
            record["is_empty"] = bool(desk_empty_by_id[int(record["desk_id"])])

    target_counts = {
        class_name: sum(1 for r in target_records if r["class_name"] == class_name)
        for class_name in ("target_bottle", "target_can")
    }
    distractor_counts = {
        str(spec["class_name"]): sum(
            1 for r in distractor_records if r["class_name"] == str(spec["class_name"])
        )
        for spec in DISTRACTOR_SPECS
    }
    scene_metadata: Dict[str, object] = {
        "scenario": SCENE_VARIANT,
        "scene_variant": SCENE_VARIANT,
        "num_desks": int(len(desks)),
        "num_targets": int(len(target_records)),
        "num_bottles": int(target_counts["target_bottle"]),
        "num_cans": int(target_counts["target_can"]),
        "p_target_on_desk": float(P_TARGET_ON_DESK),
        "target_spawn_counts": {
            "on_desk": int(sum(1 for r in target_records if r["spawn_mode"] == "on_desk")),
            "off_desk": int(sum(1 for r in target_records if r["spawn_mode"] == "off_desk")),
        },
        "empty_desk_ids": sorted(int(d["desk_id"]) for d in desks if d["is_empty"]),
        "distractor_scenario": {
            "num_distractors": int(len(distractor_records)),
            "p_optional_distractor_on_desk": float(P_OPTIONAL_DISTRACTOR_ON_DESK),
            "spawn_counts": {
                "on_desk": int(sum(1 for r in distractor_records if r["spawn_mode"] == "on_desk")),
                "off_desk": int(sum(1 for r in distractor_records if r["spawn_mode"] == "off_desk")),
            },
            "class_counts": distractor_counts,
            "table_only_class_names": sorted(
                str(spec["class_name"])
                for spec in DISTRACTOR_SPECS
                if str(spec["placement"]) == "desk_only"
            ),
        },
    }
    return objects_manifest, target_records, desks, scene_metadata


def main() -> None:
    if not (0.0 < SPAWN_CENTER_DISC_RADIUS_M < 0.5 * CELL_SIZE_M):
        raise ValueError("HARD_DATASET_SPAWN_CENTER_DISC_RADIUS_M must keep samples inside their selected grid cell.")
    if not (0.0 <= P_TARGET_ON_DESK <= 1.0):
        raise ValueError("HARD_DATASET_P_TARGET_ON_DESK must be in [0, 1].")
    if not (0.0 <= P_OPTIONAL_DISTRACTOR_ON_DESK <= 1.0):
        raise ValueError("HARD_DATASET_P_OPTIONAL_DISTRACTOR_ON_DESK must be in [0, 1].")

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

    run_state = _init_or_resume_run_state(run_state_path, seeds, policy_name, SCENE_VARIANT)
    run_index = int(run_state["current_run_index"])
    current_seed = int(run_state["current_seed"])
    total_runs = int(run_state["total_runs"])
    random.seed(current_seed)

    _atomic_write_json(run_state_path, run_state)
    _write_task_state(task_state_path, status="running", seed=current_seed, run_index=run_index, now_time_s=0.0)

    spawn_root = supervisor.getFromDef("SPAWN_ROOT")
    if spawn_root is None:
        print("ERROR: DEF SPAWN_ROOT Group not found in the hard dataset world.", flush=True)
        return
    spawn_children = spawn_root.getField("children")

    drone_xy = _sample_drone_spawn_xy()
    _place_drone(
        supervisor,
        supervisor.getFromDef(DRONE_DEF),
        x=drone_xy[0],
        y=drone_xy[1],
        yaw=DRONE_SPAWN_YAW,
    )

    objects_manifest, target_records, desks, scene_metadata = _spawn_hard_dataset_scene(
        spawn_children,
        drone_xy=drone_xy,
    )

    target_counts = {
        class_name: sum(1 for r in target_records if r["class_name"] == class_name)
        for class_name in ("target_bottle", "target_can")
    }

    print(
        f"[{_LOG_TAG}] run {run_index + 1}/{total_runs} seed={current_seed} "
        f"desks={len(desks)} targets={len(target_records)} "
        f"distractors={scene_metadata['distractor_scenario']['num_distractors']} "
        f"on_desk={scene_metadata['target_spawn_counts']['on_desk']} "
        f"off_desk={scene_metadata['target_spawn_counts']['off_desk']}",
        flush=True,
    )

    drone_cell = _cxcy_from_xy(drone_xy[0], drone_xy[1])
    experiment_init = {
        "seed": int(current_seed),
        "run_index": int(run_index),
        "total_runs": int(total_runs),
        "collection_name": run_state["collection_name"],
        "collection_start_time": run_state["collection_start_time"],
        "policy_name": policy_name,
        "samples_per_run": int(SAMPLES_PER_RUN),
        "environment": "hard",
        "run_mode": "dataset_collection",
        "target_setup": "hard",
        "scene_variant": SCENE_VARIANT,
        "drone_spawn": {
            "x": float(drone_xy[0]),
            "y": float(drone_xy[1]),
            "yaw": float(DRONE_SPAWN_YAW),
            "cx": None if drone_cell is None else int(drone_cell[0]),
            "cy": None if drone_cell is None else int(drone_cell[1]),
        },
    }
    _atomic_write_json(experiment_init_path, experiment_init)

    metadata = {
        "seed": int(current_seed),
        "run_index": int(run_index),
        "environment": "hard",
        "run_mode": "dataset_collection",
        "target_setup": "hard",
        "scene_variant": SCENE_VARIANT,
        "room_dimensions_m": {"x": ARENA_X, "y": ARENA_Y},
        "grid": {"cols": GRID_COLS, "rows": GRID_ROWS, "cell_size_m": CELL_SIZE_M},
        "drone_spawn": experiment_init["drone_spawn"],
        "target_class_counts": target_counts,
        "target_on_desk_probability": float(P_TARGET_ON_DESK),
        "target_spawn_constraints": {
            "center_disc_radius_m": float(SPAWN_CENTER_DISC_RADIUS_M),
            "drone_exclusion_radius_m": float(DRONE_TARGET_EXCLUSION_RADIUS_M),
            "min_object_separation_m": float(TARGET_MIN_SEPARATION_M),
        },
        "desk_dimensions_m": {
            "length": DESK_LENGTH_M,
            "width": DESK_WIDTH_M,
            "height": DESK_HEIGHT_M,
            "recognition_height": DESK_RECOGNITION_HEIGHT_M,
            "spawn_edge_margin_m": DESK_SPAWN_EDGE_MARGIN_M,
            "floor_exclusion_margin_m": DESK_FLOOR_EXCLUSION_MARGIN_M,
        },
        "desk_spawn_constraints": {
            "wall_margin_m": DESK_WALL_MARGIN_M,
            "min_center_distance_m": DESK_MIN_CENTER_DISTANCE_M,
            "drone_exclusion_radius_m": DESK_DRONE_EXCLUSION_RADIUS_M,
        },
        "scene": scene_metadata,
        "desks": desks,
        "targets": target_records,
        "class_id_to_name": CLASS_ID_TO_NAME,
        "model_to_class_id": MODEL_TO_CLASS_ID,
    }
    _atomic_write_json(
        dataset_objects_path,
        {"seed": int(current_seed), "objects": objects_manifest, "metadata": metadata},
    )

    start_time = supervisor.getTime()
    final_reason = None
    final_time_s = None

    while supervisor.step(timestep) != -1:
        now_time_s = float(supervisor.getTime())
        elapsed = now_time_s - float(start_time)
        collection_state = _read_json(collection_state_path, {})

        if (
            collection_state.get("status") == "complete"
            and int(collection_state.get("seed", -999999)) == current_seed
            and int(collection_state.get("run_index", -1)) == run_index
        ):
            final_reason = "complete"
            final_time_s = now_time_s
        elif elapsed >= RUN_TIMEOUT_SECONDS:
            final_reason = "timeout"
            final_time_s = now_time_s

        if final_reason is None:
            _write_task_state(
                task_state_path,
                status="running",
                seed=current_seed,
                run_index=run_index,
                now_time_s=now_time_s,
            )
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
            "scene_variant": SCENE_VARIANT,
            "target_counts": target_counts,
            "scene": scene_metadata,
        }
        run_state["runs"] = list(run_state.get("runs", [])) + [run_record]

        is_last = (run_index + 1) >= total_runs
        if is_last:
            run_state["completed"] = True
            _atomic_write_json(run_state_path, run_state)
            print(f"[{_LOG_TAG}] completed all {total_runs} runs.", flush=True)
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
            f"[{_LOG_TAG}] finished run {run_index + 1}/{total_runs}; "
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