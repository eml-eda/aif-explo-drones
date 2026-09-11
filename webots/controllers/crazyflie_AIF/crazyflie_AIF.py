# controllers/crazyflie_AIF/crazyflie_AIF.py

import os
import sys
import uuid
from datetime import datetime

from controller import Keyboard
from controller import Robot

from devices import CrazyflieDevices
from pid.pid_controller import pid_velocity_fixed_height_controller
from policies.wall_following import WallFollowingPolicy
from policies.pseudo_random import PseudoRandomPolicy
from policies.spiral import SpiralPolicy
from policies.rotate_and_measure import RotateAndMeasurePolicy
from policies.aif_binary import AIFBinaryPolicy
from policies.aif_zoi import AIFZOIPolicy
from utils.capture_util import save_capture
from utils.logging_AIF import ExperimentLogger
from vision.perception_manager import PerceptionManager, PerceptionSchedule
from utils.run_visualizations import generate_run_visualizations

from env.env_manager import GridSpec, GridEnvManager
from env.surface_projection import (
    effective_table_projection_cells as compute_effective_table_projection_cells,
    make_fov_spec,
    merge_visual_coverages_by_surface,
    scaled_fov_spec_for_surface,
    surface_height_payload,
)
from task.claim_manager import ClaimManager
from runtime.controller_pipeline import (
    PolicyPerceptionEvidence,
    TargetDistanceTracker,
    TaskStateSnapshot,
    build_policy_observation,
    build_progress_payload,
    detection_event_rc_to_cxcy,
    mapping_rc_to_cxcy,
    rc_to_cxcy,
)
from vision.perception_projection import (
    PerceptionProjectionConfig,
    project_perception_frame,
)

CONTROLLERS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CONTROLLERS_DIR not in sys.path:
    sys.path.insert(0, CONTROLLERS_DIR)

from common.shared_state import (
    SharedStateStore,
    atomic_write_json,
    read_json,
    wait_for_json,
    wait_for_supervisor_session,
)

from config import AIF_POLICY as DEFAULT_POLICY
import config as aif_config

# Central, validated detector selection. User-facing choices live in config.py.
DETECTOR_SELECTION = aif_config.DETECTOR_SELECTION


# ---- Run toggles ----
ENABLE_CSV_LOGGING = False

ENABLE_OCCUPANCY_HEATMAP = True
ENABLE_TRAJECTORY_PLOT = True
TRACK_AIF_TARGET_DISTANCE = False

ENABLE_LOGGING = ENABLE_CSV_LOGGING or ENABLE_OCCUPANCY_HEATMAP or ENABLE_TRAJECTORY_PLOT

ENABLE_CAPTURES = False
LOG_EVERY_N_STEPS = 20
CAPTURE_EVERY_N_STEPS = 20

TRAJECTORY_PLOTS_SUBDIR = "trajectory plots"

# ---- Mode toggles ----
EVAL_MODE = (aif_config.RUN_MODE == "evaluation")
EVAL_STATE_FILENAME = "eval_state.json"

# ---- Experiment identity / report aggregation ----
SPAWN_CONDITIONS = "spawn15"
ENVIRONMENT = aif_config.ENVIRONMENT

# ---- Flight preference ----
FLYING_ATTITUDE = 1.5

# ---- Spatial parameters ----
ARENA_WIDTH_M = 5.5
ARENA_HEIGHT_M = 6.5
CELL_SIZE_M = 0.5
GRID_COLS = int(ARENA_WIDTH_M / CELL_SIZE_M)
GRID_ROWS = int(ARENA_HEIGHT_M / CELL_SIZE_M)
MIN_X = -ARENA_WIDTH_M / 2.0
MIN_Y = -ARENA_HEIGHT_M / 2.0
MAX_X = ARENA_WIDTH_M / 2.0
MAX_Y = ARENA_HEIGHT_M / 2.0

# ---- Camera FoV footprint on floor (meters) at 1.5m altitude ----
# Measured from the Webots camera FoV setup. Forward/backward asymmetry comes
# from the downward camera offset relative to the drone center.
FOV_WIDTH_M = 1.394
FOV_FORWARD_M = 0.553
FOV_BACKWARD_M = 0.493

# ---- Height-aware FoV setup ----
# The base FoV above is the floor-plane projection at FLYING_ATTITUDE.
# For raised surfaces, footprint dimensions scale with camera-to-surface distance.
ENABLE_HEIGHT_AWARE_FOV = True
FLOOR_HEIGHT_M = 0.0
TABLE_HEIGHT_M = 0.72
DESK_CLASS_NAME = "desk"
DESK_CLASS_ID = 2
SURFACE_DETECTION_CONF_THRESHOLD = 0.30

# ---- Table-height shadow projection ----
# Current-frame geometry correction for detected desk/table bboxes. Every
# deterministic image-space probe of a desk bbox is projected onto both the
# table plane and the floor plane. Table-plane cells remain persistent,
# perception-derived table cells. A floor-plane counterpart is retained only
# when the paired cell centres satisfy the tolerance-aware forward-direction
# tests below. This excludes the rear FoV strip from the transient correction
# while preserving forward tabletop-occlusion protection. Transient cells are
# not persisted and do not create positive table evidence.
ENABLE_TABLE_SHADOW_PROJECTION = True

# Allows for 0.5 m grid-cell quantisation and approximate yaw values when
# comparing paired table/floor cell centres along the camera forward axis.
TABLE_SHADOW_FORWARD_TOLERANCE_M = 0.10

# ---- FoV → cell thresholds ----
COVERAGE_UPDATE_THRESHOLD = 0.05

# ---- Single debug config ----
DEBUG = {
    "every_n": 100,
    "fov": False,
    "cmd": False,
    "policy_grid": False,
    "policy_detections": False,
}

# ---- Snapshot / debugging ----
ENABLE_SNAPSHOTS = False
SNAPSHOT_EVERY_N_STEPS = 200
SNAPSHOT_ON_STUCK = True
STUCK_REPLAN_LIMIT = 100

# ---- Detection mode ----
GT_CONFIDENCE = 1.0
GT_JSON_FILENAME = "spawned_objects.json"

# ---- Perception sampling / throttling ----
# "none"   -> process detector every control step
# "frames" -> process once every DETECTION_EVERY_N_FRAMES control steps
# "time"   -> process once every DETECTION_EVERY_S simulation seconds
DETECTION_THROTTLE_MODE = "time"   # "none", "frames", "time"

DETECTION_EVERY_N_FRAMES = 20
DETECTION_EVERY_S = 0.625

# ---- Shared init / task / progress files ----
EXPERIMENT_INIT_FILENAME = "experiment_init.json"
TASK_STATE_FILENAME = "task_state.json"
CONTROLLER_PROGRESS_FILENAME = "controller_progress.json"

# ---- Controller claim criterion ----
CLAIM_CONFIDENCE_THRESHOLD = 0.45

# ---- Takeoff warm-up ----
AUTO_START_AUTONOMY = True
AUTONOMY_START_DELAY_S = 15.0

# ---- Startup file wait ----
STARTUP_WAIT_TIMEOUT_S = 10.0
STARTUP_WAIT_POLL_S = 0.05

# Classes that can provide target-belief evidence and create controller claims.
# Desk observations are preserved separately in per-frame detection lists, but they are not
# treated as target evidence until a table-aware policy consumes them explicitly.
TARGET_OBJECT_CLASS_NAMES = {"target_bottle", "target_can"}
TARGET_OBJECT_CLASS_IDS = {0, 1}
DESK_OBJECT_CLASS_NAMES = {DESK_CLASS_NAME}
DESK_OBJECT_CLASS_IDS = {DESK_CLASS_ID}

AIF_TARGET_DISTANCE_POLICIES = {"aif_binary", "aif_zoi"}




SHARED_STATE = SharedStateStore.from_controller_file(__file__)

def load_gt_objects(json_path: str):
    data = read_json(json_path)
    return list(data.get("objects", []))


def make_run_dir(base_log_dir: str, prefix: str = "log_run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_log_dir, f"{prefix}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def load_policy(name, debug=None, seed=None):
    policy_id = aif_config.normalize_policy_id(name)
    registry = {
        "wall_following": WallFollowingPolicy,
        "aif_binary": AIFBinaryPolicy,
        "aif_zoi": AIFZOIPolicy,
        "pseudo_random": PseudoRandomPolicy,
        "spiral": SpiralPolicy,
        "rotate_and_measure": RotateAndMeasurePolicy,
    }
    policy_cls = registry[policy_id]
    if policy_id == "pseudo_random":
        return policy_cls(seed=seed, debug=debug)
    return policy_cls(debug=debug)

def _resolve_run_dir(base_log_dir: str, policy_name: str, eval_state_path: str) -> str:
    """
    Single-run mode:
      logs/log_run_<policy>_<ts>

    Evaluation mode:
      <eval_dir>/runs/log_run_<policy>_runXX_seedYY_<ts>

    Falls back to single-run layout if eval_state is missing or malformed.
    """
    if not EVAL_MODE:
        return make_run_dir(base_log_dir, prefix=f"log_run_{policy_name}")

    eval_state = read_json(eval_state_path, {})
    eval_dir = eval_state.get("eval_dir", None)
    run_index = int(eval_state.get("current_run_index", 0))
    current_seed = eval_state.get("current_seed", "unknown")

    if not eval_dir:
        return make_run_dir(base_log_dir, prefix=f"log_run_{policy_name}")

    runs_dir = os.path.join(eval_dir, "runs")
    prefix = f"log_run_{policy_name}_run{run_index:02d}_seed{current_seed}"
    return make_run_dir(runs_dir, prefix=prefix)


def _detector_limitations_payload() -> dict:
    mode = str(DETECTION_THROTTLE_MODE).lower()

    payload = {
        "mode": mode,
        "frames_interval": None,
        "time_interval_s": None,
    }

    if mode == "frames":
        payload["frames_interval"] = int(DETECTION_EVERY_N_FRAMES)
    elif mode == "time":
        payload["time_interval_s"] = float(DETECTION_EVERY_S)

    return payload

def _detector_model_metadata_payload() -> dict:
    return DETECTOR_SELECTION.metadata()


def _experiment_metadata_payload() -> dict:
    return {
        **_detector_model_metadata_payload(),
        "claim_confidence_threshold": float(CLAIM_CONFIDENCE_THRESHOLD),
        "spawn_conditions": str(SPAWN_CONDITIONS),
        "environment": str(ENVIRONMENT),
        "run_mode": str(aif_config.RUN_MODE),
        "target_setup": str(ENVIRONMENT),
    }


def _supports_aif_target_distance(policy_name: str) -> bool:
    return str(policy_name).lower().strip() in AIF_TARGET_DISTANCE_POLICIES


def _tracks_aif_target_distance(policy_name: str) -> bool:
    return bool(TRACK_AIF_TARGET_DISTANCE and _supports_aif_target_distance(policy_name))



def _read_manual_inputs(
    keyboard,
    *,
    autonomous_mode: bool,
    task_running: bool,
    time_now_s: float,
):
    """Consume queued keyboard input and return manual motion set-points.

    The controller applies these first; an autonomous policy may overwrite the
    planar/yaw commands later in the same step.
    """
    forward_desired = 0.0
    sideways_desired = 0.0
    yaw_desired = 0.0
    height_diff_desired = 0.0
    autonomy_started_at = None

    key = keyboard.getKey()
    while key > 0:
        if key == Keyboard.UP:
            forward_desired += 0.5
        elif key == Keyboard.DOWN:
            forward_desired -= 0.5
        elif key == Keyboard.RIGHT:
            sideways_desired -= 0.5
        elif key == Keyboard.LEFT:
            sideways_desired += 0.5
        elif key == ord("Q"):
            yaw_desired = +1
        elif key == ord("E"):
            yaw_desired = -1
        elif key == ord("W"):
            height_diff_desired = 0.1
        elif key == ord("S"):
            height_diff_desired = -0.1
        elif key == ord("A"):
            if not autonomous_mode and task_running:
                autonomous_mode = True
                autonomy_started_at = float(time_now_s)
                print("Autonomous mode: ON", flush=True)
        elif key == ord("D"):
            if autonomous_mode:
                autonomous_mode = False
                print("Autonomous mode: OFF", flush=True)
        key = keyboard.getKey()

    return (
        forward_desired,
        sideways_desired,
        yaw_desired,
        height_diff_desired,
        autonomous_mode,
        autonomy_started_at,
    )

def main() -> None:
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    devices = CrazyflieDevices(robot, timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    policy_name = aif_config.normalize_policy_id(os.environ.get("AIF_POLICY", DEFAULT_POLICY))

    experiment_init_path = SHARED_STATE.path(EXPERIMENT_INIT_FILENAME)
    gt_path = SHARED_STATE.path(GT_JSON_FILENAME)
    task_state_path = SHARED_STATE.path(TASK_STATE_FILENAME)
    controller_progress_path = SHARED_STATE.path(CONTROLLER_PROGRESS_FILENAME)
    eval_state_path = SHARED_STATE.path(EVAL_STATE_FILENAME)

    controller_start_token = uuid.uuid4().hex
    wait_for_supervisor_session(
        SHARED_STATE,
        controller_token=controller_start_token,
        timeout_s=STARTUP_WAIT_TIMEOUT_S,
        poll_s=STARTUP_WAIT_POLL_S,
    )
    print("[IPC] startup session established", flush=True)

    experiment_init = wait_for_json(
        experiment_init_path,
        validator=lambda d: isinstance(d.get("drone_spawn", None), dict) and ("x" in d["drone_spawn"]) and ("y" in d["drone_spawn"]),
        timeout_s=STARTUP_WAIT_TIMEOUT_S,
        poll_s=STARTUP_WAIT_POLL_S,
    )

    if experiment_init is None:
        print(f"[WARN] experiment init file not available at startup: {experiment_init_path}", flush=True)
        experiment_init = {
            "seed": -1,
            "drone_spawn": {"x": 0.0, "y": 0.0, "yaw": 0.0, "cx": None, "cy": None},
        }

    aif_config.validate_runtime_selection(
        aif_config.EXPERIMENT_SELECTION,
        experiment_init,
    )

    episode_seed = int(experiment_init.get("seed", -1))
    policy = load_policy(policy_name, debug=DEBUG, seed=episode_seed)

    drone_spawn = experiment_init.get("drone_spawn", {}) or {}
    drone_spawn_x = float(drone_spawn.get("x", 0.0))
    drone_spawn_y = float(drone_spawn.get("y", 0.0))
    drone_spawn_yaw = float(drone_spawn.get("yaw", 0.0))
    drone_spawn_cx = drone_spawn.get("cx", None)
    drone_spawn_cy = drone_spawn.get("cy", None)

    print(
        f"[INIT] drone spawn x={drone_spawn_x:.3f} y={drone_spawn_y:.3f} "
        f"yaw={drone_spawn_yaw:.3f} cell={(drone_spawn_cx, drone_spawn_cy)}",
        flush=True,
    )

    grid_spec = GridSpec(
        min_x=MIN_X,
        max_x=MAX_X,
        min_y=MIN_Y,
        max_y=MAX_Y,
        cell_size=CELL_SIZE_M,
        rows=GRID_ROWS,
        cols=GRID_COLS,
    )

    floor_fov_spec = make_fov_spec(
        width_m=FOV_WIDTH_M,
        forward_m=FOV_FORWARD_M,
        backward_m=FOV_BACKWARD_M,
    )
    table_fov_spec = scaled_fov_spec_for_surface(
        TABLE_HEIGHT_M,
        camera_height_m=FLYING_ATTITUDE,
        reference_height_m=FLYING_ATTITUDE,
        base_width_m=FOV_WIDTH_M,
        base_forward_m=FOV_FORWARD_M,
        base_backward_m=FOV_BACKWARD_M,
    )

    env_mgr_floor = GridEnvManager(grid_spec, floor_fov_spec)
    env_mgr_table = GridEnvManager(grid_spec, table_fov_spec)

    claim_manager = ClaimManager(
        confidence_threshold=CLAIM_CONFIDENCE_THRESHOLD,
        target_class_names=TARGET_OBJECT_CLASS_NAMES,
        target_class_ids=TARGET_OBJECT_CLASS_IDS,
    )
    controller_claim_object_keys = set()  # (cx, cy, class_name)
    controller_claim_cells = set()

    task_snapshot = TaskStateSnapshot()

    visited_cells = set()
    persistent_table_cells = set()  # discovered persistent table-height cells, stored as (cx, cy)
    persistent_table_confidence = {}  # persistent max desk confidence by discovered table cell (cx, cy)

    autonomy_start_time = None
    final_snapshot_written = False

    gt_objects = None
    if DETECTOR_SELECTION.is_ground_truth:
        gt_json = wait_for_json(
            gt_path,
            validator=lambda d: isinstance(d.get("objects", None), list),
            timeout_s=STARTUP_WAIT_TIMEOUT_S,
            poll_s=STARTUP_WAIT_POLL_S,
        )
        if gt_json is None:
            raise FileNotFoundError(f"Required GT file not found or invalid: {gt_path}")

        gt_objects = load_gt_objects(gt_path)
        print(f"[GT] Loaded {len(gt_objects)} objects from {gt_path}")
        print("[GT] Example objects:", gt_objects[:3])

    environment_spec = aif_config.EXPERIMENT_SELECTION.environment_spec
    perception_class_names = [
        *environment_spec.target_classes,
        *environment_spec.context_classes,
    ]
    perception_class_name_to_id = {
        class_name: class_id
        for class_id, class_name in enumerate(perception_class_names)
    }

    perception_schedule = PerceptionSchedule(
        mode=DETECTION_THROTTLE_MODE,
        frames_interval=DETECTION_EVERY_N_FRAMES,
        time_interval_s=DETECTION_EVERY_S,
    )
    perception_manager = PerceptionManager(
        camera=camera,
        grid_spec=grid_spec,
        detector_selection=DETECTOR_SELECTION,
        gt_objects=gt_objects,
        class_name_to_id=perception_class_name_to_id,
        timestep_ms=timestep,
        schedule=perception_schedule,
        gt_confidence=GT_CONFIDENCE,
    )
    perception_projection_config = PerceptionProjectionConfig(
        target_class_names=frozenset(TARGET_OBJECT_CLASS_NAMES),
        target_class_ids=frozenset(TARGET_OBJECT_CLASS_IDS),
        desk_class_names=frozenset(DESK_OBJECT_CLASS_NAMES),
        desk_class_ids=frozenset(DESK_OBJECT_CLASS_IDS),
        surface_detection_conf_threshold=float(SURFACE_DETECTION_CONF_THRESHOLD),
        enable_table_shadow_projection=bool(ENABLE_TABLE_SHADOW_PROJECTION),
        table_shadow_forward_tolerance_m=float(TABLE_SHADOW_FORWARD_TOLERANCE_M),
        min_x=float(MIN_X),
        min_y=float(MIN_Y),
        cell_size_m=float(CELL_SIZE_M),
    )

    print("\n[FoV setup]")
    print(
        f"  cell_size={CELL_SIZE_M}m grid={GRID_ROWS}x{GRID_COLS} "
        f"x∈[{MIN_X:.2f},{MAX_X:.2f}] y∈[{MIN_Y:.2f},{MAX_Y:.2f}]"
    )
    print(
        f"  floor footprint width={FOV_WIDTH_M:.3f}m "
        f"forward={FOV_FORWARD_M:.3f}m backward={FOV_BACKWARD_M:.3f}m"
    )
    print(
        f"  table footprint height={TABLE_HEIGHT_M:.3f}m "
        f"width={(table_fov_spec.half_left + table_fov_spec.half_right):.3f}m "
        f"forward={table_fov_spec.forward:.3f}m backward={table_fov_spec.backward:.3f}m"
    )

    base_log_dir = os.path.join(os.path.dirname(__file__), "logs")
    run_dir = _resolve_run_dir(base_log_dir, policy_name, eval_state_path)
    print("Run output dir:", run_dir)

    claim_log_path = os.path.join(run_dir, f"claim_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    claim_log_file = open(claim_log_path, "w", encoding="utf-8")
    claim_log_file.write("time_s,x_m,y_m,current_cx,current_cy,claim_cx,claim_cy,class_name,confidence\n")

    logger = ExperimentLogger(
        enabled=ENABLE_LOGGING,
        log_dir=run_dir,
        grid_spec=grid_spec,
        min_time_s=AUTONOMY_START_DELAY_S,
    )

    pid_controller = pid_velocity_fixed_height_controller()

    height_desired = FLYING_ATTITUDE
    autonomous_mode = False
    step_count = 0
    occupancy_dt_accum = 0.0

    target_distance_tracker = TargetDistanceTracker()

    def write_progress(last_time_s: float) -> None:
        progress_payload = build_progress_payload(
            policy_name=policy_name,
            run_dir=run_dir,
            autonomy_start_time_s=autonomy_start_time,
            last_controller_time_s=last_time_s,
            visited_cells=visited_cells,
            claimed_found_cells=controller_claim_cells,
            claimed_found_objects=controller_claim_object_keys,
            detector_limitations=_detector_limitations_payload(),
            experiment_metadata=_experiment_metadata_payload(),
            target_distance_stats=target_distance_tracker.payload(
                enabled=TRACK_AIF_TARGET_DISTANCE,
                policy_supported=_supports_aif_target_distance(policy_name),
            ),
            drone_spawn={
                "x": drone_spawn_x,
                "y": drone_spawn_y,
                "yaw": drone_spawn_yaw,
                "cx": drone_spawn_cx,
                "cy": drone_spawn_cy,
            },
            surface_height_map=surface_height_payload(
                persistent_table_cells,
                persistent_table_confidence,
                enabled=ENABLE_HEIGHT_AWARE_FOV,
                floor_height_m=FLOOR_HEIGHT_M,
                table_height_m=TABLE_HEIGHT_M,
            ),
        )
        atomic_write_json(controller_progress_path, progress_payload)

    print("\n====== AIF Controller ======\n")
    print("ENVIRONMENT =", ENVIRONMENT)
    print("RUN_MODE =", aif_config.RUN_MODE)
    print("AIF_POLICY =", policy_name)
    print("AIF_DETECTOR =", DETECTOR_SELECTION.detector_id)
    print("Controls: Arrow keys for planar motion, Q/E yaw, W/S altitude")
    print("Press A to enable autonomous mode, D to disable\n")

    write_progress(last_time_s=0.0)

    try:
        while robot.step(timestep) != -1:
            step_count += 1

            # ---- Sensors + supervisor task state ----
            sensor_state = devices.read()
            camera_capture = sensor_state["camera"]
            dt = sensor_state["dt"]
            time_now = float(sensor_state["time"])

            occupancy_dt_accum += float(dt)

            task_snapshot = TaskStateSnapshot.from_payload(
                read_json(task_state_path, {})
            )

            # ---- Manual / autonomy mode inputs ----
            if AUTO_START_AUTONOMY and (not autonomous_mode):
                if time_now >= AUTONOMY_START_DELAY_S and task_snapshot.status == "running":
                    autonomous_mode = True
                    autonomy_start_time = float(time_now)
                    print(f"Autonomous mode: ON (auto at t={time_now:.1f}s)", flush=True)

            (
                forward_desired,
                sideways_desired,
                yaw_desired,
                height_diff_desired,
                autonomous_mode,
                manual_autonomy_start_time,
            ) = _read_manual_inputs(
                keyboard,
                autonomous_mode=autonomous_mode,
                task_running=(task_snapshot.status == "running"),
                time_now_s=time_now,
            )
            if manual_autonomy_start_time is not None:
                autonomy_start_time = manual_autonomy_start_time

            height_desired = max(0.1, height_desired + height_diff_desired * dt)

            if ENABLE_CAPTURES and (step_count % CAPTURE_EVERY_N_STEPS == 0):
                save_capture(
                    camera_capture["image"],
                    width=camera_capture["width"],
                    height=camera_capture["height"],
                    timestamp=camera_capture["time"],
                    output_dir=os.path.join(run_dir, "captures"),
                )

            # ---- Geometry + visual FoV ----
            x_pos = sensor_state["gps"]["x"]
            y_pos = sensor_state["gps"]["y"]
            yaw = sensor_state["imu"]["yaw"]

            floor_visual_coverages = env_mgr_floor.compute_cell_coverages(x_pos, y_pos, yaw)
            table_visual_coverages = env_mgr_table.compute_cell_coverages(x_pos, y_pos, yaw)

            # Reset every controller step. Populated only from desk detections
            # processed in this frame, then used below to select table-height
            # coverage for their paired floor-plane probe cells.
            transient_table_shadow_cells = set()

            effective_table_projection_cells = (
                compute_effective_table_projection_cells(
                    persistent_table_cells_cxcy=persistent_table_cells,
                    transient_table_shadow_cells_cxcy=transient_table_shadow_cells,
                    enable_table_shadow_projection=ENABLE_TABLE_SHADOW_PROJECTION,
                )
                if ENABLE_HEIGHT_AWARE_FOV
                else set(persistent_table_cells)
            )
            visual_coverages = (
                merge_visual_coverages_by_surface(
                    floor_coverages_rc=floor_visual_coverages,
                    table_coverages_rc=table_visual_coverages,
                    table_projection_cells_cxcy=effective_table_projection_cells,
                )
                if ENABLE_HEIGHT_AWARE_FOV
                else dict(floor_visual_coverages)
            )
            visible_mask = env_mgr_floor.coverage_to_visible_mask(
                visual_coverages,
                threshold=COVERAGE_UPDATE_THRESHOLD,
            )
            current_cell_rc = grid_spec.rowcol_from_xy(x_pos, y_pos)
            current_cell_cxcy = rc_to_cxcy(current_cell_rc)

            if current_cell_cxcy is not None:
                visited_cells.add(current_cell_cxcy)

            # ---- Perception + image-to-grid projection ----
            perception_frame = project_perception_frame(
                perception_manager=perception_manager,
                detector_selection=DETECTOR_SELECTION,
                config=perception_projection_config,
                step_count=step_count,
                time_now_s=time_now,
                camera_capture=camera_capture,
                drone_x=x_pos,
                drone_y=y_pos,
                drone_yaw=yaw,
                env_mgr_floor=env_mgr_floor,
                env_mgr_table=env_mgr_table,
                floor_visual_coverages=floor_visual_coverages,
                persistent_table_cells=persistent_table_cells,
                persistent_table_confidence=persistent_table_confidence,
            )
            process_detection_this_step = perception_frame.processed
            raw_detector_output = perception_frame.raw_output
            transient_table_shadow_cells = perception_frame.transient_table_shadow_cells

            # Desk detections from this frame can change which cells must use the
            # table-height visual footprint. Recompute only after perception.
            if ENABLE_HEIGHT_AWARE_FOV and process_detection_this_step:
                effective_table_projection_cells = compute_effective_table_projection_cells(
                    persistent_table_cells_cxcy=persistent_table_cells,
                    transient_table_shadow_cells_cxcy=transient_table_shadow_cells,
                    enable_table_shadow_projection=ENABLE_TABLE_SHADOW_PROJECTION,
                )
                visual_coverages = merge_visual_coverages_by_surface(
                    floor_coverages_rc=floor_visual_coverages,
                    table_coverages_rc=table_visual_coverages,
                    table_projection_cells_cxcy=effective_table_projection_cells,
                )
                visible_mask = env_mgr_floor.coverage_to_visible_mask(
                    visual_coverages,
                    threshold=COVERAGE_UPDATE_THRESHOLD,
                )

            visual_coverages_cxcy = mapping_rc_to_cxcy(visual_coverages)
            visible_mask_cxcy = mapping_rc_to_cxcy(visible_mask)
            det_conf_cxcy = mapping_rc_to_cxcy(perception_frame.target_confidence_rc)
            det_meta_cxcy = mapping_rc_to_cxcy(perception_frame.target_metadata_rc)
            det_all_events_cxcy = [
                detection_event_rc_to_cxcy(event)
                for event in perception_frame.all_events_rc
            ]
            det_target_events_cxcy = [
                detection_event_rc_to_cxcy(event)
                for event in perception_frame.target_events_rc
            ]
            table_conf_cxcy = mapping_rc_to_cxcy(perception_frame.table_confidence_rc)

            # Policies must consume perception evidence only on processed detector
            # frames. Reusing an empty observation on skipped control steps would
            # incorrectly apply repeated no-detection evidence.
            perception_evidence = PolicyPerceptionEvidence.for_step(
                perception_processed=process_detection_this_step,
                visible_mask=visible_mask_cxcy,
                det_confidence=det_conf_cxcy,
                det_confidence_multiclass=det_conf_cxcy,
                det_metadata_multiclass=det_meta_cxcy,
                detections_all=det_all_events_cxcy,
                target_detections=det_target_events_cxcy,
                table_confidence=table_conf_cxcy,
                table_visible_mask=visible_mask_cxcy,
                table_coverages=visual_coverages_cxcy,
            )

            surface_height_map_payload = surface_height_payload(
                persistent_table_cells,
                persistent_table_confidence,
                enabled=ENABLE_HEIGHT_AWARE_FOV,
                floor_height_m=FLOOR_HEIGHT_M,
                table_height_m=TABLE_HEIGHT_M,
            )

            best_obj = None
            if det_conf_cxcy:
                best_cell, best_conf = max(det_conf_cxcy.items(), key=lambda kv: float(kv[1]))
                best_meta = det_meta_cxcy.get(best_cell, {})
                best_class_name = str(best_meta.get("class_name", best_meta.get("label", "unknown")))
                best_class_id = int(best_meta.get("class_id", -1))
                best_obj = {
                    "cell": best_cell,
                    "confidence": float(best_conf),
                    "class_name": best_class_name,
                    "class_id": best_class_id,
                }

            # ---- Controller claims + task progress ----
            # Only processed perception frames can create claims. Skipped control
            # steps carry no new detector evidence.
            new_controller_claims = claim_manager.update(
                det_target_events_cxcy,
                perception_processed=process_detection_this_step,
            )

            for claim in new_controller_claims:
                print(
                    f"[CONTROLLER CLAIM] cell={(claim.cx, claim.cy)} class={claim.class_name} "
                    f"conf={claim.confidence:.2f}",
                    flush=True,
                )

                if claim_log_file is not None:
                    cur_cx, cur_cy = current_cell_cxcy if current_cell_cxcy is not None else (-1, -1)
                    claim_log_file.write(
                        f"{time_now:.6f},{x_pos:.6f},{y_pos:.6f},"
                        f"{int(cur_cx)},{int(cur_cy)},"
                        f"{claim.cx},{claim.cy},"
                        f"{claim.class_name},{claim.confidence:.6f}\n"
                    )
                    claim_log_file.flush()

            controller_claim_object_keys = claim_manager.claimed_object_keys
            controller_claim_cells = claim_manager.claimed_cells

            if DEBUG.get("fov", False) and (step_count % int(DEBUG.get("every_n", 50)) == 0):
                top_rc = sorted(visual_coverages.items(), key=lambda kv: kv[1], reverse=True)[:12]
                num_vis = sum(1 for v in visible_mask.values() if v)

                print("\n[FOV DEBUG]")
                print(
                    f"  step={step_count}  pos=({x_pos:.2f},{y_pos:.2f}) "
                    f"yaw={yaw:.3f} rad ({yaw * 180.0 / 3.14159:.1f} deg)"
                )
                print(f"  current_cell(cx,cy)={current_cell_cxcy}")
                print(
                    f"  visual_coverages: {len(visual_coverages)} cells intersect FoV; {num_vis} >= {COVERAGE_UPDATE_THRESHOLD:.2f}; "
                    f"table_cells={len(persistent_table_cells)} effective_table_projection_cells={len(effective_table_projection_cells)}"
                )
                for (r, c), frac in top_rc:
                    print(f"    cell=(cx={c},cy={r}) coverage={frac*100:5.1f}% visible={visible_mask[(r,c)]}")

            if ENABLE_LOGGING and (step_count % LOG_EVERY_N_STEPS == 0):
                logger.log_occupancy(
                    time_s=sensor_state["time"],
                    x_m=x_pos,
                    y_m=y_pos,
                    dt_s=occupancy_dt_accum,
                )
                occupancy_dt_accum = 0.0

            if (step_count % LOG_EVERY_N_STEPS == 0) or (best_obj is not None and best_obj["cell"] in controller_claim_cells):
                write_progress(last_time_s=time_now)

            # ---- Policy observation ----
            policy_observation = build_policy_observation(
                sensor_state=sensor_state,
                step=step_count,
                pose={"x": x_pos, "y": y_pos, "yaw": yaw},
                grid_spec=grid_spec,
                detector_is_ground_truth=DETECTOR_SELECTION.is_ground_truth,
                detector_detections=(
                    None if raw_detector_output is None else raw_detector_output.detections
                ),
                current_cell=current_cell_cxcy,
                visual_coverages=visual_coverages_cxcy,
                perception_evidence=perception_evidence,
                coverage_threshold=COVERAGE_UPDATE_THRESHOLD,
                persistent_table_confidence=persistent_table_confidence,
                perception_processed=process_detection_this_step,
                surface_height_map=surface_height_map_payload,
                persistent_table_cells=persistent_table_cells,
                effective_table_projection_cells=effective_table_projection_cells,
                claimed_found_cells=controller_claim_cells,
                task_state=task_snapshot,
                claim_confidence_threshold=CLAIM_CONFIDENCE_THRESHOLD,
                snapshots_enabled=ENABLE_SNAPSHOTS,
                snapshots_dir=os.path.join(run_dir, "snapshots") if run_dir else None,
                snapshots_every_n=SNAPSHOT_EVERY_N_STEPS,
                snapshots_on_stuck=SNAPSHOT_ON_STUCK,
                stuck_replan_limit=STUCK_REPLAN_LIMIT,
                floor_fov_spec=floor_fov_spec,
                table_fov_spec=table_fov_spec,
                floor_height_m=FLOOR_HEIGHT_M,
                table_height_m=TABLE_HEIGHT_M,
            )

            if task_snapshot.status != "running":
                if autonomous_mode:
                    autonomous_mode = False
                    print(f"Autonomous mode: OFF ({task_snapshot.status})", flush=True)

                if (not final_snapshot_written) and hasattr(policy, "save_snapshot_from_controller"):
                    try:
                        policy.save_snapshot_from_controller(
                            policy_observation,
                            reason=str(task_snapshot.reason or task_snapshot.status),
                        )
                        final_snapshot_written = True
                    except Exception as e:
                        print(f"[SNAPSHOT] final snapshot failed: {e}", flush=True)

            # ---- Policy action ----
            if autonomous_mode and task_snapshot.status == "running":
                policy_action = policy.compute_action(policy_observation)
                forward_desired = policy_action.get("forward", 0.0)
                sideways_desired = policy_action.get("sideways", 0.0)
                yaw_desired = policy_action.get("yaw_rate", 0.0)

                if (
                    _tracks_aif_target_distance(policy_name)
                    and target_distance_tracker.record_policy_action(policy_action)
                ):
                    write_progress(last_time_s=time_now)

            # ---- Navigation / PID / motors ----
            if DEBUG.get("cmd", False) and (step_count % int(DEBUG.get("every_n", 50)) == 0):
                print(
                    f"[cmd] step={step_count} auto={autonomous_mode} "
                    f"fwd={forward_desired:.2f} side={sideways_desired:.2f} yaw={yaw_desired:.2f} "
                    f"h_des={height_desired:.2f} z={sensor_state['gps']['z']:.2f}"
                )

            motor_power = pid_controller.pid(
                dt,
                forward_desired,
                sideways_desired,
                yaw_desired,
                height_desired,
                sensor_state["imu"]["roll"],
                sensor_state["imu"]["pitch"],
                sensor_state["gyro"]["yaw_rate"],
                sensor_state["gps"]["z"],
                sensor_state["velocity"]["vx"],
                sensor_state["velocity"]["vy"],
            )

            devices.set_motor_speeds(motor_power)

            # ---- Per-step output logging ----
            if ENABLE_LOGGING and (step_count % LOG_EVERY_N_STEPS == 0):
                det_count = len(perception_frame.all_events_rc)

                logger.log_step(
                    time=sensor_state["time"],
                    forward=forward_desired,
                    sideways=sideways_desired,
                    yaw_rate=yaw_desired,
                    altitude=sensor_state["gps"]["z"],
                    detections=det_count,
                )

    finally:
        try:
            write_progress(last_time_s=float(time_now) if "time_now" in locals() else 0.0)
        except Exception:
            pass

        logger_paths = logger.close()

        try:
            if claim_log_file is not None:
                claim_log_file.close()
        except Exception:
            pass

        if ENABLE_OCCUPANCY_HEATMAP or ENABLE_TRAJECTORY_PLOT:
            try:
                visual_outputs = generate_run_visualizations(
                    run_dir=run_dir,
                    make_heatmap=ENABLE_OCCUPANCY_HEATMAP,
                    make_trajectory=ENABLE_TRAJECTORY_PLOT,
                    occupancy_summary_path=logger_paths.get("occupancy_summary"),
                    occupancy_log_path=logger_paths.get("occupancy_log"),
                    gt_json_path=gt_path if os.path.exists(gt_path) else None,
                    plots_subdir=TRAJECTORY_PLOTS_SUBDIR,
                )

                for name, path in visual_outputs.items():
                    print(f"[VIS] wrote {name}: {path}", flush=True)

            except Exception as e:
                print(f"[VIS] failed to generate run visualizations: {e}", flush=True)

if __name__ == "__main__":
    main()
