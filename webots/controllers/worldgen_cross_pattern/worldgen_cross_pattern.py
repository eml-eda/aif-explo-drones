from __future__ import annotations

from controller import Supervisor


# --- Settings ---
SPAWN_AFTER_SECONDS = 20.0
EPISODE_SECONDS = 180.0  # total run duration (optional, matches your earlier setup)

# Arena size (meters) - match your RectangleArena floorSize (x, y)
ARENA_X = 5.5
ARENA_Y = 6.5

# Keep bottles away from walls
WALL_MARGIN = 0.35

# Spacing between bottles along the lines
SPACING_M = 0.25

# PROTO name declared as IMPORTABLE EXTERNPROTO in the world header
BOTTLE_PROTO_NODE = "BeerBottle"


def frange_inclusive(start: float, stop: float, step: float):
    """Floating range including stop if it lands on the grid."""
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        yield start + i * step


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    # Find spawn root
    spawn_root = supervisor.getFromDef("SPAWN_ROOT")
    if spawn_root is None:
        print("ERROR: DEF SPAWN_ROOT Group not found in the world.")
        return
    spawn_children = spawn_root.getField("children")

    spawned = False
    start_time = supervisor.getTime()

    # Compute line extents inside arena bounds
    xmin = -ARENA_X / 2.0 + WALL_MARGIN
    xmax = +ARENA_X / 2.0 - WALL_MARGIN
    ymin = -ARENA_Y / 2.0 + WALL_MARGIN
    ymax = +ARENA_Y / 2.0 - WALL_MARGIN

    def spawn_plus_pattern():
        # Build a set of positions so we don't duplicate the center (0,0)
        positions = set()

        # Line along X axis (y = 0)
        for x in frange_inclusive(xmin, xmax, SPACING_M):
            # round to avoid float hash weirdness
            positions.add((round(x, 4), 0.0))

        # Line along Y axis (x = 0)
        for y in frange_inclusive(ymin, ymax, SPACING_M):
            positions.add((0.0, round(y, 4)))

        # Spawn all positions
        for idx, (x, y) in enumerate(sorted(positions)):
            node_string = (
                f'DEF FOV_BOTTLE_{idx:03d} {BOTTLE_PROTO_NODE} {{ '
                f'  translation {x:.4f} {y:.4f} 0 '
                f'}}'
            )
            spawn_children.importMFNodeFromString(-1, node_string)

        print(f"Spawned {len(positions)} BeerBottle objects in + pattern "
              f"(spacing={SPACING_M}m, margin={WALL_MARGIN}m).")

    # Main loop
    while supervisor.step(timestep) != -1:
        t = supervisor.getTime()
        elapsed = t - start_time

        # Spawn after 10 seconds (only once)
        if (not spawned) and (elapsed >= SPAWN_AFTER_SECONDS):
            spawn_plus_pattern()
            spawned = True

        # Optional: end run and cleanup at EPISODE_SECONDS
        if elapsed >= EPISODE_SECONDS:
            break

    # Optional cleanup (remove everything under SPAWN_ROOT)
    for idx in range(spawn_children.getCount() - 1, -1, -1):
        node = spawn_children.getMFNode(idx)
        node.remove()

    print("Cleanup complete. Quitting simulation.")
    supervisor.simulationQuit(0)


if __name__ == "__main__":
    main()
