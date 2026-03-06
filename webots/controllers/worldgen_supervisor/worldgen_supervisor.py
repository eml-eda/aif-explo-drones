# controllers/worldgen_supervisor/worldgen_supervisor.py

from __future__ import annotations

import random
import math
from controller import Supervisor
import os
import json

# ---- Experiment settings ----
SEED = 21

WARMUP_SECONDS = 15.0
EPISODE_SECONDS = 180.0
TOTAL_SECONDS = WARMUP_SECONDS + EPISODE_SECONDS

N_BOTTLES = 6

# Arena in your world: floorSize 6.5 5.5
ARENA_X = 5.5
ARENA_Y = 6.5

# Keep bottles away from walls and drone start
WALL_MARGIN = 0.35
DRONE_EXCLUSION_RADIUS = 0.75

# Avoid overlap between spawned objects
MIN_SEPARATION = 0.30

# Bottle object we spawn (from IMPORTABLE EXTERNPROTO)
# We'll set translation explicitly.
BOTTLE_PROTO_NODE = "BeerBottle"

# ---- GT export ----
EXPORT_GT_JSON = True
GT_JSON_FILENAME = "spawned_objects.json"

def dist2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _shared_gt_json_path() -> str:
    # worldgen_supervisor.py is in: webots/controllers/worldgen_supervisor/
    # shared dir is:            webots/controllers/shared/
    here = os.path.dirname(__file__)
    shared_dir = os.path.abspath(os.path.join(here, "..", "shared"))
    os.makedirs(shared_dir, exist_ok=True)
    return os.path.join(shared_dir, GT_JSON_FILENAME)


def _write_gt_json(path: str, *, seed: int, positions_xy):
    payload = {
        "seed": int(seed),
        "objects": [{"x": float(x), "y": float(y)} for (x, y) in positions_xy],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def main():
    random.seed(SEED)

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    # Get spawn root group
    spawn_root = supervisor.getFromDef("SPAWN_ROOT")
    if spawn_root is None:
        print("ERROR: DEF SPAWN_ROOT Group not found in the world.")
        return
    spawn_children = spawn_root.getField("children")

    # Spawn once at beginning
    spawned_positions = []
    spawned_nodes = []

    xmin = -ARENA_X / 2.0 + WALL_MARGIN
    xmax = +ARENA_X / 2.0 - WALL_MARGIN
    ymin = -ARENA_Y / 2.0 + WALL_MARGIN
    ymax = +ARENA_Y / 2.0 - WALL_MARGIN

    drone_xy = (0.0, 0.0)  # you said drone starts at center

    def sample_position(max_tries=2000):
        for _ in range(max_tries):
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)

            # Keep away from drone spawn area
            if dist2((x, y), drone_xy) < DRONE_EXCLUSION_RADIUS ** 2:
                continue

            # Keep away from other bottles
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

        # Build node string. We set a DEF so we can debug, and a translation.
        # z is handled by the PROTO default (MedicineBottle default translation has z offset),
        # but we still set z=0 to place on floor reference.
        node_string = (
            f'DEF BOTTLE_{i:03d} {BOTTLE_PROTO_NODE} {{ '
            f'  translation {x:.4f} {y:.4f} 0 '
            f'}}'
        )

        # Import into SPAWN_ROOT children
        spawn_children.importMFNodeFromString(-1, node_string)

        # Get the inserted node back (last child)
        new_node = spawn_children.getMFNode(spawn_children.getCount() - 1)
        spawned_nodes.append(new_node)

    print(f"Spawned {len(spawned_nodes)} bottles. Seed={SEED}")

    if EXPORT_GT_JSON:
        gt_path = _shared_gt_json_path()
        _write_gt_json(gt_path, seed=SEED, positions_xy=spawned_positions)
        print(f"GT JSON exported: {gt_path}")

    # Run for EPISODE_SECONDS, then cleanup
    start_time = supervisor.getTime()
    while supervisor.step(timestep) != -1:
        t = supervisor.getTime()
        if (t - start_time) >= TOTAL_SECONDS:
            break

    # Cleanup: remove all children under SPAWN_ROOT
    # Remove from end to start to avoid index shifting surprises.
    for idx in range(spawn_children.getCount() - 1, -1, -1):
        node = spawn_children.getMFNode(idx)
        node.remove()

    print("Cleanup complete. Quitting simulation.")
    supervisor.simulationQuit(0)  # available in Supervisor API
    # simulationQuit is part of the Supervisor API set. :contentReference[oaicite:3]{index=3}


if __name__ == "__main__":
    main()
