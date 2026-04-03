# controllers/worldgen_supervisor/worldgen_supervisor.py

from __future__ import annotations

import json
import os
import random
from controller import Supervisor

# ---- Experiment settings ----
SEED = 28

WARMUP_SECONDS = 15.0
EPISODE_SECONDS = 180.0
TOTAL_SECONDS = WARMUP_SECONDS + EPISODE_SECONDS
FINALIZATION_GRACE_SECONDS = 1.0

N_BOTTLES = 6

# Arena in your world
ARENA_X = 5.5
ARENA_Y = 6.5
CELL_SIZE_M = 0.5

# Keep bottles away from walls and drone start
WALL_MARGIN = 0.35
DRONE_EXCLUSION_RADIUS = 0.75

# Avoid overlap between spawned objects
MIN_SEPARATION = 0.70

# Bottle object we spawn
BOTTLE_PROTO_NODE = "BeerBottle"

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


def _write_spawned_objects_json(path: str, *, seed: int, positions_xy):
    objects = []
    for (x, y) in positions_xy:
        cxcy = _cxcy_from_xy(float(x), float(y))
        obj = {"x": float(x), "y": float(y)}
        if cxcy is not None:
            obj["cx"] = int(cxcy[0])
            obj["cy"] = int(cxcy[1])
        objects.append(obj)

    payload = {
        "seed": int(seed),
        "objects": objects,
    }
    _atomic_write_json(path, payload)


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


def _write_task_state(
    path: str,
    *,
    status: str,
    reason: str | None,
    seed: int,
    total_objects: int,
    remaining_objects: int,
    validated_found_cells: set,
    timeout_s: float,
    warmup_s: float,
    episode_s: float,
    now_time_s: float,
    report_written: bool,
) -> None:
    payload = {
        "status": str(status),
        "reason": reason,
        "seed": int(seed),
        "total_objects": int(total_objects),
        "remaining_objects": int(remaining_objects),
        "validated_found_cells": sorted([[int(c[0]), int(c[1])] for c in validated_found_cells]),
        "timeout_s": float(timeout_s),
        "warmup_s": float(warmup_s),
        "episode_s": float(episode_s),
        "now_time_s": float(now_time_s),
        "report_written": bool(report_written),
    }
    _atomic_write_json(path, payload)


def _write_run_report(
    *,
    run_dir: str,
    policy_name: str,
    gt_seed: int,
    total_objects: int,
    validated_found_cells: set,
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

    found_count = int(len(validated_found_cells))
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
        "found_cells": sorted([[int(c[0]), int(c[1])] for c in validated_found_cells]),
        "coverage": {
            "visited_cells": int(coverage_cells),
            "total_cells": int(total_cells),
            "coverage_ratio": float(coverage_ratio),
        },
    }

    out_path = os.path.join(report_dir, "run_report.json")
    _atomic_write_json(out_path, payload)
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

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    spawned_objects_path = _shared_path(SPAWNED_OBJECTS_FILENAME)
    experiment_init_path = _shared_path(EXPERIMENT_INIT_FILENAME)
    task_state_path = _shared_path(TASK_STATE_FILENAME)
    controller_progress_path = _shared_path(CONTROLLER_PROGRESS_FILENAME)

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

    spawned_positions = []
    spawned_nodes = []

    xmin = -ARENA_X / 2.0 + WALL_MARGIN
    xmax = +ARENA_X / 2.0 - WALL_MARGIN
    ymin = -ARENA_Y / 2.0 + WALL_MARGIN
    ymax = +ARENA_Y / 2.0 - WALL_MARGIN

    def sample_position(max_tries=2000):
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
            if not ok:
                continue

            return (x, y)

        return None

    for i in range(N_BOTTLES):
        pos = sample_position()
        if pos is None:
            print(f"WARNING: Could not place bottle {i} under constraints.")
            break

        x, y = pos
        spawned_positions.append((x, y))

        node_string = (
            f'DEF BOTTLE_{i:03d} {BOTTLE_PROTO_NODE} {{ '
            f'  translation {x:.4f} {y:.4f} 0 '
            f'}}'
        )

        spawn_children.importMFNodeFromString(-1, node_string)
        new_node = spawn_children.getMFNode(spawn_children.getCount() - 1)
        spawned_nodes.append(new_node)

    print(f"Spawned {len(spawned_nodes)} bottles. Seed={SEED}", flush=True)

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
        positions_xy=spawned_positions,
    )
    print(f"GT JSON exported: {spawned_objects_path}", flush=True)

    true_cells = set()
    for (x, y) in spawned_positions:
        cxcy = _cxcy_from_xy(x, y)
        if cxcy is not None:
            true_cells.add(cxcy)

    total_objects = len(true_cells)
    validated_found_cells = set()
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
        validated_found_cells=validated_found_cells,
        timeout_s=TOTAL_SECONDS,
        warmup_s=WARMUP_SECONDS,
        episode_s=EPISODE_SECONDS,
        now_time_s=0.0,
        report_written=False,
    )

    _atomic_write_json(
        controller_progress_path,
        {
            "policy_name": None,
            "run_dir": None,
            "autonomy_start_time_s": None,
            "last_controller_time_s": 0.0,
            "visited_cells": [],
            "claimed_found_cells": [],
            "drone_spawn": {
                "x": DRONE_SPAWN_X,
                "y": DRONE_SPAWN_Y,
                "yaw": DRONE_SPAWN_YAW,
                "cx": _cxcy_from_xy(DRONE_SPAWN_X, DRONE_SPAWN_Y)[0] if _cxcy_from_xy(DRONE_SPAWN_X, DRONE_SPAWN_Y) else None,
                "cy": _cxcy_from_xy(DRONE_SPAWN_X, DRONE_SPAWN_Y)[1] if _cxcy_from_xy(DRONE_SPAWN_X, DRONE_SPAWN_Y) else None,
            },
        },
    )

    start_time = supervisor.getTime()

    while supervisor.step(timestep) != -1:
        t = supervisor.getTime()
        elapsed_total = float(t - start_time)

        controller_progress = _read_json(
            controller_progress_path,
            {
                "policy_name": None,
                "run_dir": None,
                "autonomy_start_time_s": None,
                "last_controller_time_s": 0.0,
                "visited_cells": [],
                "claimed_found_cells": [],
            },
        )

        claimed_found_cells = {
            (int(cx), int(cy))
            for (cx, cy) in controller_progress.get("claimed_found_cells", [])
        }

        new_validated = claimed_found_cells.intersection(true_cells)
        if new_validated != validated_found_cells:
            validated_found_cells = set(new_validated)

        remaining_objects = max(0, int(total_objects) - int(len(validated_found_cells)))

        if len(validated_found_cells) >= total_objects and total_objects > 0:
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
                validated_found_cells=validated_found_cells,
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
                validated_found_cells=validated_found_cells,
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
                    validated_found_cells=validated_found_cells,
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
                validated_found_cells=validated_found_cells,
                timeout_s=TOTAL_SECONDS,
                warmup_s=WARMUP_SECONDS,
                episode_s=EPISODE_SECONDS,
                now_time_s=float(final_time_s),
                report_written=True,
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