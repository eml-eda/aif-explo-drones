# controllers/worldgen_dataset_supervisor/worldgen_dataset_supervisor.py

from __future__ import annotations

import colorsys
import json
import os
import random

from controller import Supervisor

# ---------- Dataset run config ----------
SEED = 50

WARMUP_SECONDS = 15.0
DATA_COLLECTION_SECONDS = 10000.0
TOTAL_SECONDS = WARMUP_SECONDS + DATA_COLLECTION_SECONDS
FINALIZATION_GRACE_SECONDS = 1.0

N_BOTTLES = 10
N_CANS = 10

# ---------- Arena ----------
ARENA_X = 5.5
ARENA_Y = 6.5
CELL_SIZE_M = 0.5

WALL_MARGIN = 0.35
DRONE_EXCLUSION_RADIUS = 0.75
MIN_SEPARATION = 0.70

# ---------- Bottle wrapper ----------
BOTTLE_CLASS_ID = 0
BOTTLE_CLASS_NAME = "target_bottle"
BOTTLE_MODEL_NAME = "target_bottle"

# Approximate bounding cylinder for recognition/size estimation
BOTTLE_BBOX_RADIUS_M = 0.035
BOTTLE_BBOX_HEIGHT_M = 0.245

# ---------- Can wrapper ----------
CAN_CLASS_ID = 1
CAN_CLASS_NAME = "target_can"
CAN_MODEL_NAME = "target_can"

# Can dimensions from Webots docs: 0.03175 x 0.03175 x 0.1222 m
CAN_BBOX_RADIUS_M = 0.018
CAN_BBOX_HEIGHT_M = 0.125

# ---------- Drone placement ----------
DRONE_DEF = "CRAZYFLIE"
DRONE_SPAWN_X = 0.0
DRONE_SPAWN_Y = 0.0
DRONE_SPAWN_YAW = 0.0

# ---------- Shared files ----------
EXPERIMENT_INIT_FILENAME = "experiment_init.json"
DATASET_OBJECTS_FILENAME = "dataset_objects.json"
DATASET_STATE_FILENAME = "dataset_state.json"


def dist2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def _shared_dir() -> str:
    here = os.path.dirname(__file__)
    shared_dir = os.path.abspath(os.path.join(here, "..", "shared"))
    os.makedirs(shared_dir, exist_ok=True)
    return shared_dir


def _shared_path(filename: str) -> str:
    return os.path.join(_shared_dir(), filename)


def _atomic_write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _cxcy_from_xy(x: float, y: float) -> tuple[int, int] | None:
    min_x = -ARENA_X / 2.0
    min_y = -ARENA_Y / 2.0
    max_x = ARENA_X / 2.0
    max_y = ARENA_Y / 2.0

    if x < min_x or x >= max_x or y < min_y or y >= max_y:
        return None

    cx = int((x - min_x) / CELL_SIZE_M)
    cy = int((y - min_y) / CELL_SIZE_M)
    return (cx, cy)


def _write_experiment_init_json(path: str, *, seed: int, drone_spawn_xy, drone_spawn_yaw: float):
    x, y = drone_spawn_xy
    cxcy = _cxcy_from_xy(float(x), float(y))
    payload = {
        "seed": int(seed),
        "drone_spawn": {
            "x": float(x),
            "y": float(y),
            "yaw": float(drone_spawn_yaw),
            "cx": int(cxcy[0]) if cxcy is not None else None,
            "cy": int(cxcy[1]) if cxcy is not None else None,
        },
    }
    _atomic_write_json(path, payload)


def _write_dataset_state(
    path: str,
    *,
    status: str,
    reason: str | None,
    seed: int,
    total_objects: int,
    now_time_s: float,
    timeout_s: float,
    warmup_s: float,
    collection_s: float,
):
    payload = {
        "status": str(status),
        "reason": reason,
        "seed": int(seed),
        "total_objects": int(total_objects),
        "now_time_s": float(now_time_s),
        "timeout_s": float(timeout_s),
        "warmup_s": float(warmup_s),
        "collection_s": float(collection_s),
    }
    _atomic_write_json(path, payload)


def _node_id(node):
    if hasattr(node, "getId"):
        return int(node.getId())
    if hasattr(node, "get_id"):
        return int(node.get_id())
    return None


def _instance_color(index: int) -> tuple[float, float, float]:
    """
    Unique-ish per-instance recognition color.
    Useful if you later want segmentation-based QA.
    """
    hue = (index * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (max(0.1, r), max(0.1, g), max(0.1, b))


def _bottle_wrapper_node_string(index: int, x: float, y: float, color_rgb) -> str:
    r, g, b = color_rgb
    def_name = f"TARGET_BOTTLE_{index:03d}"

    return f'''
DEF {def_name} Solid {{
  translation {x:.6f} {y:.6f} 0
  model "{BOTTLE_MODEL_NAME}"
  recognitionColors [
    {r:.6f} {g:.6f} {b:.6f}
  ]
  children [
    BeerBottle {{
    }}
  ]
  boundingObject Cylinder {{
    height {BOTTLE_BBOX_HEIGHT_M:.6f}
    radius {BOTTLE_BBOX_RADIUS_M:.6f}
    subdivision 16
  }}
}}
'''.strip()


def _can_wrapper_node_string(index: int, x: float, y: float, color_rgb) -> str:
    r, g, b = color_rgb
    def_name = f"TARGET_CAN_{index:03d}"

    return f'''
DEF {def_name} Solid {{
  translation {x:.6f} {y:.6f} 0
  model "{CAN_MODEL_NAME}"
  recognitionColors [
    {r:.6f} {g:.6f} {b:.6f}
  ]
  children [
    Can {{
    }}
  ]
  boundingObject Cylinder {{
    height {CAN_BBOX_HEIGHT_M:.6f}
    radius {CAN_BBOX_RADIUS_M:.6f}
    subdivision 16
  }}
}}
'''.strip()


def _write_dataset_objects_json(path: str, *, seed: int, objects):
    payload = {
        "seed": int(seed),
        "objects": objects,
    }
    _atomic_write_json(path, payload)


def _place_drone(supervisor: Supervisor, drone_node, *, x: float, y: float, yaw: float) -> None:
    if drone_node is None:
        raise RuntimeError(f"Drone DEF '{DRONE_DEF}' not found.")

    translation_field = drone_node.getField("translation")
    rotation_field = drone_node.getField("rotation")

    current_translation = translation_field.getSFVec3f()
    z = float(current_translation[2])

    translation_field.setSFVec3f([float(x), float(y), z])
    rotation_field.setSFRotation([0.0, 0.0, 1.0, float(yaw)])

    drone_node.resetPhysics()
    supervisor.simulationResetPhysics()


def main():
    random.seed(SEED)

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    experiment_init_path = _shared_path(EXPERIMENT_INIT_FILENAME)
    dataset_objects_path = _shared_path(DATASET_OBJECTS_FILENAME)
    dataset_state_path = _shared_path(DATASET_STATE_FILENAME)

    # Clear stale shared files from previous runs.
    _safe_remove(experiment_init_path)
    _safe_remove(dataset_objects_path)
    _safe_remove(dataset_state_path)

    spawn_root = supervisor.getFromDef("SPAWN_ROOT")
    if spawn_root is None:
        print("ERROR: DEF SPAWN_ROOT Group not found in the world.", flush=True)
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

    xmin = -ARENA_X / 2.0 + WALL_MARGIN
    xmax = +ARENA_X / 2.0 - WALL_MARGIN
    ymin = -ARENA_Y / 2.0 + WALL_MARGIN
    ymax = +ARENA_Y / 2.0 - WALL_MARGIN

    spawned_positions = []
    spawned_nodes = []
    objects_manifest = []

    def sample_position(max_tries=5000):
        for _ in range(max_tries):
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)

            if dist2((x, y), drone_xy) < DRONE_EXCLUSION_RADIUS ** 2:
                continue

            ok = True
            for (px, py) in spawned_positions:
                if dist2((x, y), (px, py)) < MIN_SEPARATION ** 2:
                    ok = False
                    break
            if ok:
                return (x, y)
        return None

    bottle_index = 0
    can_index = 0
    global_instance_index = 0

    # Spawn bottles
    for _ in range(N_BOTTLES):
        pos = sample_position()
        if pos is None:
            print(f"WARNING: could not place bottle {bottle_index} under constraints.", flush=True)
            break

        x, y = pos
        color_rgb = _instance_color(global_instance_index)
        node_string = _bottle_wrapper_node_string(bottle_index, x, y, color_rgb)

        spawn_children.importMFNodeFromString(-1, node_string)
        new_node = spawn_children.getMFNode(spawn_children.getCount() - 1)

        spawned_positions.append((x, y))
        spawned_nodes.append(new_node)

        cxcy = _cxcy_from_xy(x, y)
        obj_record = {
            "def": f"TARGET_BOTTLE_{bottle_index:03d}",
            "id": _node_id(new_node),
            "class_id": int(BOTTLE_CLASS_ID),
            "class_name": BOTTLE_CLASS_NAME,
            "model": BOTTLE_MODEL_NAME,
            "recognition_color_rgb": [float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2])],
            "x": float(x),
            "y": float(y),
            "z": 0.0,
            "cx": None if cxcy is None else int(cxcy[0]),
            "cy": None if cxcy is None else int(cxcy[1]),
        }
        objects_manifest.append(obj_record)

        bottle_index += 1
        global_instance_index += 1

    # Spawn cans
    for _ in range(N_CANS):
        pos = sample_position()
        if pos is None:
            print(f"WARNING: could not place can {can_index} under constraints.", flush=True)
            break

        x, y = pos
        color_rgb = _instance_color(global_instance_index)
        node_string = _can_wrapper_node_string(can_index, x, y, color_rgb)

        spawn_children.importMFNodeFromString(-1, node_string)
        new_node = spawn_children.getMFNode(spawn_children.getCount() - 1)

        spawned_positions.append((x, y))
        spawned_nodes.append(new_node)

        cxcy = _cxcy_from_xy(x, y)
        obj_record = {
            "def": f"TARGET_CAN_{can_index:03d}",
            "id": _node_id(new_node),
            "class_id": int(CAN_CLASS_ID),
            "class_name": CAN_CLASS_NAME,
            "model": CAN_MODEL_NAME,
            "recognition_color_rgb": [float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2])],
            "x": float(x),
            "y": float(y),
            "z": 0.0,
            "cx": None if cxcy is None else int(cxcy[0]),
            "cy": None if cxcy is None else int(cxcy[1]),
        }
        objects_manifest.append(obj_record)

        can_index += 1
        global_instance_index += 1

    print(
        f"[SPAWN] spawned {bottle_index} bottles + {can_index} cans = {len(spawned_nodes)} objects, seed={SEED}",
        flush=True,
    )

    _write_experiment_init_json(
        experiment_init_path,
        seed=SEED,
        drone_spawn_xy=drone_xy,
        drone_spawn_yaw=DRONE_SPAWN_YAW,
    )
    _write_dataset_objects_json(
        dataset_objects_path,
        seed=SEED,
        objects=objects_manifest,
    )
    _write_dataset_state(
        dataset_state_path,
        status="running",
        reason=None,
        seed=SEED,
        total_objects=len(objects_manifest),
        now_time_s=0.0,
        timeout_s=TOTAL_SECONDS,
        warmup_s=WARMUP_SECONDS,
        collection_s=DATA_COLLECTION_SECONDS,
    )

    print(f"[FILES] wrote {experiment_init_path}", flush=True)
    print(f"[FILES] wrote {dataset_objects_path}", flush=True)

    start_time = supervisor.getTime()
    final_reason = None
    final_time_s = None

    while supervisor.step(timestep) != -1:
        t = supervisor.getTime()
        elapsed_total = float(t - start_time)

        if elapsed_total >= TOTAL_SECONDS:
            final_reason = "timeout"
            final_time_s = float(t)
        else:
            _write_dataset_state(
                dataset_state_path,
                status="running",
                reason=None,
                seed=SEED,
                total_objects=len(objects_manifest),
                now_time_s=float(t),
                timeout_s=TOTAL_SECONDS,
                warmup_s=WARMUP_SECONDS,
                collection_s=DATA_COLLECTION_SECONDS,
            )
            continue

        _write_dataset_state(
            dataset_state_path,
            status=final_reason,
            reason=final_reason,
            seed=SEED,
            total_objects=len(objects_manifest),
            now_time_s=float(final_time_s),
            timeout_s=TOTAL_SECONDS,
            warmup_s=WARMUP_SECONDS,
            collection_s=DATA_COLLECTION_SECONDS,
        )

        print(f"[SUPERVISOR] end-of-run state: {final_reason}", flush=True)

        grace_start = supervisor.getTime()
        while supervisor.step(timestep) != -1:
            if (supervisor.getTime() - grace_start) >= FINALIZATION_GRACE_SECONDS:
                break
        break

    for idx in range(spawn_children.getCount() - 1, -1, -1):
        node = spawn_children.getMFNode(idx)
        node.remove()

    print("Cleanup complete. Quitting simulation.", flush=True)
    supervisor.simulationQuit(0)


if __name__ == "__main__":
    main()