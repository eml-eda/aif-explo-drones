# webots/controllers/worldgen_fov_measure_supervisor/worldgen_fov_measure_supervisor.py

from __future__ import annotations

import math
import os
from controller import Supervisor

# -----------------------------------------------------------------------------
# Purpose
# -----------------------------------------------------------------------------
# Minimal supervisor for FoV measurement.
# It clears DEF SPAWN_ROOT, places the Crazyflie, and spawns one tabletop object
# using the same dynamic geometry style as the table scenario supervisors.
# It does not terminate the run; use the paired fov_measure_controller.py to fly
# manually and print FoV-size estimates.

# ---- Drone placement ----
DRONE_DEF = os.environ.get("FOV_MEASURE_DRONE_DEF", "CRAZYFLIE")
DRONE_SPAWN_X = float(os.environ.get("FOV_MEASURE_DRONE_X", "0.0"))
DRONE_SPAWN_Y = float(os.environ.get("FOV_MEASURE_DRONE_Y", "-2.0"))
DRONE_SPAWN_YAW = float(os.environ.get("FOV_MEASURE_DRONE_YAW", "0.0"))

# ---- Table placement ----
TABLE_DEF = os.environ.get("FOV_MEASURE_TABLE_DEF", "FOV_MEASURE_TABLE")
TABLE_X = float(os.environ.get("FOV_MEASURE_TABLE_X", "0.0"))
TABLE_Y = float(os.environ.get("FOV_MEASURE_TABLE_Y", "0.0"))
TABLE_YAW = float(os.environ.get("FOV_MEASURE_TABLE_YAW", "0.0"))

# Same default dimensions as the table dataset/table scenario stack.
DESK_LENGTH_M = float(os.environ.get("FOV_MEASURE_TABLE_LENGTH_M", "1.2"))
DESK_WIDTH_M = float(os.environ.get("FOV_MEASURE_TABLE_WIDTH_M", "0.7"))
DESK_HEIGHT_M = float(os.environ.get("FOV_MEASURE_TABLE_HEIGHT_M", "0.72"))
DESK_TOP_THICKNESS_M = float(os.environ.get("FOV_MEASURE_TABLE_TOP_THICKNESS_M", "0.08"))
DESK_TABLETOP_TEXTURE_URL = os.environ.get(
    "FOV_MEASURE_TABLE_TEXTURE_URL",
    "world_resources/wooden_tabletop.jpeg",
)

# Recognition color is not needed for measurement, but keeping it makes the
# table visible to Recognition/segmentation if you choose to inspect that later.
TABLE_RECOGNITION_COLOR = (1.0, 0.15, 0.85)


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


def _tabletop_node_string() -> str:
    r, g, b = TABLE_RECOGNITION_COLOR
    tabletop_center_z = DESK_HEIGHT_M - DESK_TOP_THICKNESS_M / 2.0

    return f'''
DEF {TABLE_DEF} Solid {{
  translation {TABLE_X:.6f} {TABLE_Y:.6f} {tabletop_center_z:.6f}
  rotation 0 0 1 {TABLE_YAW:.6f}
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


def _table_legs_node_string() -> str:
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
DEF {TABLE_DEF}_LEGS Pose {{
  translation {TABLE_X:.6f} {TABLE_Y:.6f} {leg_center_z:.6f}
  rotation 0 0 1 {TABLE_YAW:.6f}
  children [
{legs}
  ]
}}
'''.strip()


if __name__ == "__main__":
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    spawn_root = supervisor.getFromDef("SPAWN_ROOT")
    if spawn_root is None:
        print("ERROR: DEF SPAWN_ROOT Group not found in the world.", flush=True)
        supervisor.simulationQuit(1)
        raise SystemExit(1)

    spawn_children = spawn_root.getField("children")
    _cleanup_spawned_nodes(spawn_children)

    drone_node = supervisor.getFromDef(DRONE_DEF)
    _place_drone(
        supervisor,
        drone_node,
        x=DRONE_SPAWN_X,
        y=DRONE_SPAWN_Y,
        yaw=DRONE_SPAWN_YAW,
    )

    spawn_children.importMFNodeFromString(-1, _tabletop_node_string())
    spawn_children.importMFNodeFromString(-1, _table_legs_node_string())

    print("[FOV MEASURE SUPERVISOR] spawned measurement scene", flush=True)
    print(
        f"  drone=({DRONE_SPAWN_X:.3f}, {DRONE_SPAWN_Y:.3f}) yaw={DRONE_SPAWN_YAW:.3f}",
        flush=True,
    )
    print(
        f"  table center=({TABLE_X:.3f}, {TABLE_Y:.3f}) yaw={TABLE_YAW:.3f} "
        f"size={DESK_LENGTH_M:.3f}x{DESK_WIDTH_M:.3f} top_z={DESK_HEIGHT_M:.3f}",
        flush=True,
    )
    print("  supervisor will keep running; stop/reload the world manually when done", flush=True)

    while supervisor.step(timestep) != -1:
        pass
