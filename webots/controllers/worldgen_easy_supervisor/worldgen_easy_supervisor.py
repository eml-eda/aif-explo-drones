# webots/controllers/worldgen_easy_supervisor/worldgen_easy_supervisor.py

from __future__ import annotations

import colorsys
import math
import os
import random
import sys
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

ENVIRONMENT_ID = "easy"
RUN_MODE = "single"

# ---- Experiment settings ----
SEED = 28

WARMUP_SECONDS = 15.0
EPISODE_SECONDS = 180.0
TOTAL_SECONDS = WARMUP_SECONDS + EPISODE_SECONDS
FINALIZATION_GRACE_SECONDS = 1.0

N_BOTTLES = 3
N_CANS = 3

# Arena in your world
ARENA_X = 5.5
ARENA_Y = 6.5
CELL_SIZE_M = 0.5

# Keep objects away from walls and drone start
WALL_MARGIN = 0.35
DRONE_EXCLUSION_RADIUS = 0.75

# Avoid overlap between spawned objects
MIN_SEPARATION = 0.70

# ---- In-cell spawn restriction ----
SPAWN_CENTER_DISC_RADIUS_M = 0.20

# Approximate bounding cylinders used only for Webots recognition/size estimation.
BOTTLE_BBOX_RADIUS_M = 0.035
BOTTLE_BBOX_HEIGHT_M = 0.245
CAN_BBOX_RADIUS_M = 0.018
CAN_BBOX_HEIGHT_M = 0.125

# ---- Class / proto mapping ----
CLASS_BOTTLE = "target_bottle"
CLASS_CAN = "target_can"

CLASS_NAME_TO_ID = {
    CLASS_BOTTLE: 0,
    CLASS_CAN: 1,
}

# These custom PROTOs should expose model and recognitionColors and forward them
# into the root Solid. They keep the same visuals/textures as the original objects.
BOTTLE_PROTO_NODE = os.environ.get("EASY_SCENARIO_BOTTLE_PROTO", "DatasetBeerBottle")
CAN_PROTO_NODE = os.environ.get("EASY_SCENARIO_CAN_PROTO", "DatasetCan")

# Proto/world z placement for non-wrapper legacy spawning.
BOTTLE_SPAWN_Z = 0.0
CAN_SPAWN_Z = 0.0611

# ---- Drone placement ----
DRONE_DEF = "CRAZYFLIE"
DRONE_SPAWN_X = 0.0
DRONE_SPAWN_Y = 0.0
DRONE_SPAWN_YAW = 0.0

# ---- Shared files ----
SPAWNED_OBJECTS_FILENAME = "spawned_objects.json"
EXPERIMENT_INIT_FILENAME = "experiment_init.json"
TASK_STATE_FILENAME = "task_state.json"
CONTROLLER_PROGRESS_FILENAME = "controller_progress.json"

# ---- Run report ----
REPORT_SUBDIR = "run_report"


def dist2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _node_id(node):
    if hasattr(node, "getId"):
        return int(node.getId())
    if hasattr(node, "get_id"):
        return int(node.get_id())
    return None


def _instance_color(index: int) -> tuple[float, float, float]:
    """Unique-ish per-instance recognition color."""
    hue = (index * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (max(0.1, r), max(0.1, g), max(0.1, b))

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


def _cell_center_xy(cx: int, cy: int) -> tuple[float, float]:
    min_x = -ARENA_X / 2.0
    min_y = -ARENA_Y / 2.0
    x = min_x + (cx + 0.5) * CELL_SIZE_M
    y = min_y + (cy + 0.5) * CELL_SIZE_M
    return (x, y)


def _uniform_point_in_disc(center_x: float, center_y: float, radius: float) -> tuple[float, float]:
    r = radius * math.sqrt(random.random())
    theta = random.uniform(0.0, 2.0 * math.pi)
    x = center_x + r * math.cos(theta)
    y = center_y + r * math.sin(theta)
    return (x, y)


def _eligible_spawn_cells(radius: float) -> list[tuple[int, int]]:
    cols = int(ARENA_X / CELL_SIZE_M)
    rows = int(ARENA_Y / CELL_SIZE_M)

    xmin = -ARENA_X / 2.0 + WALL_MARGIN
    xmax = +ARENA_X / 2.0 - WALL_MARGIN
    ymin = -ARENA_Y / 2.0 + WALL_MARGIN
    ymax = +ARENA_Y / 2.0 - WALL_MARGIN

    cells = []
    for cy in range(rows):
        for cx in range(cols):
            ccx, ccy = _cell_center_xy(cx, cy)

            if (ccx - radius) < xmin:
                continue
            if (ccx + radius) > xmax:
                continue
            if (ccy - radius) < ymin:
                continue
            if (ccy + radius) > ymax:
                continue

            cells.append((cx, cy))
    return cells


def _write_spawned_objects_json(path: str, *, seed: int, objects):
    payload_objects = []
    for o in objects:
        x = float(o["x"])
        y = float(o["y"])
        cxcy = _cxcy_from_xy(x, y)
        obj = {
            "x": x,
            "y": y,
            "class_name": str(o["class_name"]),
            "class_id": int(o["class_id"]),
            "proto": str(o["proto"]),
            "model": str(o.get("model", o["class_name"])),
            "def": str(o.get("def", "")),
            "id": o.get("id", None),
            "recognition_color_rgb": o.get("recognition_color_rgb", None),
        }
        if cxcy is not None:
            obj["cx"] = int(cxcy[0])
            obj["cy"] = int(cxcy[1])
        payload_objects.append(obj)

    payload = {
        "seed": int(seed),
        "objects": payload_objects,
    }
    atomic_write_json(path, payload)


def _write_experiment_init_json(path: str, *, seed: int, drone_spawn_xy, drone_spawn_yaw: float):
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
    validated_found_objects: set,
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
    validated_found_objects: set,
    remaining_objects: int,
    autonomy_start_time_s: float | None,
    now_time_s: float,
    timeout_s: float,
    visited_cells: set,
    grid_rows: int,
    grid_cols: int,
    reason: str,
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

    payload = {
        "policy": str(policy_name),
        "gt_seed": int(gt_seed),
        "reason": str(reason),
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


def main():
    random.seed(SEED)

    if not (0.0 < SPAWN_CENTER_DISC_RADIUS_M < (0.5 * CELL_SIZE_M)):
        raise ValueError(
            f"SPAWN_CENTER_DISC_RADIUS_M must be in (0, {0.5 * CELL_SIZE_M:.3f}), "
            f"got {SPAWN_CENTER_DISC_RADIUS_M}"
        )

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    spawned_objects_path = SHARED_STATE.path(SPAWNED_OBJECTS_FILENAME)
    experiment_init_path = SHARED_STATE.path(EXPERIMENT_INIT_FILENAME)
    task_state_path = SHARED_STATE.path(TASK_STATE_FILENAME)
    controller_progress_path = SHARED_STATE.path(CONTROLLER_PROGRESS_FILENAME)

    # Clear stale shared files from previous runs so controllers cannot
    # accidentally bootstrap from an old seed/config.
    SHARED_STATE.clear(
        CONTROLLER_READY_FILENAME,
        IPC_SESSION_FILENAME,
        SPAWNED_OBJECTS_FILENAME,
        EXPERIMENT_INIT_FILENAME,
        TASK_STATE_FILENAME,
        CONTROLLER_PROGRESS_FILENAME,
    )

    print("[EASY] shared state cleared", flush=True)

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

    spawned_objects = []
    spawned_nodes = []

    eligible_cells = _eligible_spawn_cells(SPAWN_CENTER_DISC_RADIUS_M)
    if not eligible_cells:
        raise RuntimeError(
            "No eligible spawn cells available under current wall margin and center-disc radius."
        )

    print(
        f"[SPAWN] using center-disc spawn restriction with radius={SPAWN_CENTER_DISC_RADIUS_M:.3f} m "
        f"over {len(eligible_cells)} eligible cells",
        flush=True,
    )

    def sample_position(max_tries=2000):
        for _ in range(max_tries):
            cx, cy = random.choice(eligible_cells)
            center_x, center_y = _cell_center_xy(cx, cy)
            x, y = _uniform_point_in_disc(center_x, center_y, SPAWN_CENTER_DISC_RADIUS_M)

            sampled_cell = _cxcy_from_xy(x, y)
            if sampled_cell != (cx, cy):
                continue

            if dist2((x, y), drone_xy) < DRONE_EXCLUSION_RADIUS ** 2:
                continue

            ok = True
            for o in spawned_objects:
                px = float(o["x"])
                py = float(o["y"])
                if dist2((x, y), (px, py)) < MIN_SEPARATION ** 2:
                    ok = False
                    break
            if not ok:
                continue

            return (x, y)

        return None

    spawn_plan = (
        [{"class_name": CLASS_BOTTLE, "class_id": CLASS_NAME_TO_ID[CLASS_BOTTLE], "proto": BOTTLE_PROTO_NODE, "z": BOTTLE_SPAWN_Z}] * N_BOTTLES
        + [{"class_name": CLASS_CAN, "class_id": CLASS_NAME_TO_ID[CLASS_CAN], "proto": CAN_PROTO_NODE, "z": CAN_SPAWN_Z}] * N_CANS
    )
    random.shuffle(spawn_plan)

    for i, spec in enumerate(spawn_plan):
        pos = sample_position()
        if pos is None:
            print(f"WARNING: Could not place object {i} under constraints.")
            break

        x, y = pos
        class_name = str(spec["class_name"])
        proto = str(spec["proto"])

        def_name = f"TARGET_{class_name.upper()}_{i:03d}"
        color_rgb = _instance_color(i)
        node_string = _target_node_string(
            def_name=def_name,
            proto=proto,
            model=class_name,
            x=float(x),
            y=float(y),
            z=float(spec["z"]),
            color_rgb=color_rgb,
        )

        spawn_children.importMFNodeFromString(-1, node_string)
        new_node = spawn_children.getMFNode(spawn_children.getCount() - 1)
        spawned_nodes.append(new_node)

        spawned_objects.append(
            {
                "def": def_name,
                "id": _node_id(new_node),
                "x": float(x),
                "y": float(y),
                "class_name": class_name,
                "class_id": int(spec["class_id"]),
                "proto": proto,
                "model": class_name,
                "recognition_color_rgb": [float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2])],
            }
        )

    print(
        f"Spawned {len(spawned_nodes)} objects "
        f"({sum(1 for o in spawned_objects if o['class_name'] == CLASS_BOTTLE)} bottles, "
        f"{sum(1 for o in spawned_objects if o['class_name'] == CLASS_CAN)} cans). "
        f"Seed={SEED}",
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

    true_objects = set()
    for o in spawned_objects:
        cxcy = _cxcy_from_xy(float(o["x"]), float(o["y"]))
        if cxcy is not None:
            true_objects.add((int(cxcy[0]), int(cxcy[1]), str(o["class_name"])))

    total_objects = len(true_objects)
    validated_found_objects = set()
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

        claimed_found_objects = {
            (
                int(o["cx"]),
                int(o["cy"]),
                str(o["class_name"]),
            )
            for o in controller_progress.get("claimed_found_objects", [])
        }

        new_validated = claimed_found_objects.intersection(true_objects)
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
                    grid_rows=int(ARENA_Y / CELL_SIZE_M),
                    grid_cols=int(ARENA_X / CELL_SIZE_M),
                    reason=final_reason,
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

            print(f"[EASY SUPERVISOR] end-of-run state: {final_reason}", flush=True)

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
