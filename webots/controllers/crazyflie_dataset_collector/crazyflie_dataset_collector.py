# controllers/crazyflie_dataset_collector/crazyflie_dataset_collector.py

from __future__ import annotations

import json
import os
import time
from datetime import datetime

from controller import Keyboard
from controller import Robot

from devices import CrazyflieDevices
from pid.pid_controller import pid_velocity_fixed_height_controller

from policies.wall_following import WallFollowingPolicy
from policies.pseudo_random import PseudoRandomPolicy
from policies.spiral import SpiralPolicy
from policies.rotate_and_measure import RotateAndMeasurePolicy

from env.env_manager import GridSpec, FoVSpec, GridEnvManager
from vision.dataset_detector import DatasetDetector

# ---------- Dataset collection config ----------
DATASET_NAME = "mixed_bottle_can_dataset_raw"
DATASET_POLICY = os.environ.get("DATASET_POLICY", "pseudo_random")
CAPTURE_EVERY_N_STEPS = 400
MAX_SAMPLES = 800               # e.g. 2000, or None for unlimited until timeout
SAVE_EMPTY_LABELS = True
SAVE_SAMPLE_METADATA = True
SAVE_SEGMENTATION_DEBUG = False  # optional QA only

# ---------- Flight / timing ----------
FLYING_ATTITUDE = 1.5
AUTO_START_AUTONOMY = True
AUTONOMY_START_DELAY_S = 15.0
STARTUP_WAIT_TIMEOUT_S = 10.0
STARTUP_WAIT_POLL_S = 0.05

# ---------- Arena / grid ----------
ARENA_WIDTH_M = 5.5
ARENA_HEIGHT_M = 6.5
CELL_SIZE_M = 0.5
GRID_COLS = int(ARENA_WIDTH_M / CELL_SIZE_M)
GRID_ROWS = int(ARENA_HEIGHT_M / CELL_SIZE_M)
MIN_X = -ARENA_WIDTH_M / 2.0
MIN_Y = -ARENA_HEIGHT_M / 2.0
MAX_X = ARENA_WIDTH_M / 2.0
MAX_Y = ARENA_HEIGHT_M / 2.0

# ---------- Camera FoV footprint on floor ----------
FOV_WIDTH_M = 1.40
FOV_FORWARD_M = 0.55
FOV_BACKWARD_M = 0.45
COVERAGE_UPDATE_THRESHOLD = 0.05

# ---------- Shared files ----------
EXPERIMENT_INIT_FILENAME = "experiment_init.json"
DATASET_OBJECTS_FILENAME = "dataset_objects.json"
DATASET_STATE_FILENAME = "dataset_state.json"

# ---------- Class mapping ----------
MODEL_TO_CLASS = {
    "target_bottle": 0,
    "target_can": 1,
}


def rc_to_cxcy(rc):
    if rc is None:
        return None
    r, c = rc
    return (c, r)


def dict_rc_to_cxcy(d):
    return {(c, r): v for (r, c), v in d.items()}


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


def _wait_for_json(path: str, validator, timeout_s: float, poll_s: float):
    t0 = time.time()
    last_data = None
    while (time.time() - t0) < timeout_s:
        data = _read_json(path, None)
        if isinstance(data, dict):
            last_data = data
            try:
                if validator(data):
                    return data
            except Exception:
                pass
        time.sleep(poll_s)
    return last_data


def make_dataset_run_dir(base_dir: str, *, seed: int, policy_name: str) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_seed{seed:04d}_{policy_name}_{ts}")
    images_dir = os.path.join(run_dir, "images")
    labels_dir = os.path.join(run_dir, "labels")
    meta_dir = os.path.join(run_dir, "metadata")
    seg_dir = os.path.join(run_dir, "segmentation_debug")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    if SAVE_SEGMENTATION_DEBUG:
        os.makedirs(seg_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "meta_dir": meta_dir,
        "seg_dir": seg_dir,
    }


def load_policy(name, debug=None, seed=None):
    if name == "pseudo_random":
        return PseudoRandomPolicy(seed=seed, debug=debug)
    if name == "spiral":
        return SpiralPolicy(debug=debug)
    if name == "wall_following":
        return WallFollowingPolicy(debug=debug)
    if name == "rotate_and_measure":
        return RotateAndMeasurePolicy(debug=debug)
    raise ValueError(f"Unknown policy '{name}'.")


if __name__ == "__main__":
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    devices = CrazyflieDevices(robot, timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    experiment_init_path = _shared_path(EXPERIMENT_INIT_FILENAME)
    dataset_objects_path = _shared_path(DATASET_OBJECTS_FILENAME)
    dataset_state_path = _shared_path(DATASET_STATE_FILENAME)

    experiment_init = _wait_for_json(
        experiment_init_path,
        validator=lambda d: isinstance(d.get("drone_spawn", None), dict),
        timeout_s=STARTUP_WAIT_TIMEOUT_S,
        poll_s=STARTUP_WAIT_POLL_S,
    )
    if experiment_init is None:
        raise FileNotFoundError(f"Missing experiment init file: {experiment_init_path}")

    dataset_objects = _wait_for_json(
        dataset_objects_path,
        validator=lambda d: isinstance(d.get("objects", None), list),
        timeout_s=STARTUP_WAIT_TIMEOUT_S,
        poll_s=STARTUP_WAIT_POLL_S,
    )
    if dataset_objects is None:
        raise FileNotFoundError(f"Missing dataset objects file: {dataset_objects_path}")

    seed = int(experiment_init.get("seed", -1))
    policy = load_policy(DATASET_POLICY, debug=None, seed=seed)

    drone_spawn = experiment_init.get("drone_spawn", {}) or {}
    drone_spawn_x = float(drone_spawn.get("x", 0.0))
    drone_spawn_y = float(drone_spawn.get("y", 0.0))
    drone_spawn_yaw = float(drone_spawn.get("yaw", 0.0))

    print(
        f"[INIT] seed={seed} drone_spawn=({drone_spawn_x:.3f}, {drone_spawn_y:.3f}, yaw={drone_spawn_yaw:.3f})",
        flush=True,
    )

    # Grid/FoV config kept for policy compatibility
    grid_spec = GridSpec(
        min_x=MIN_X,
        max_x=MAX_X,
        min_y=MIN_Y,
        max_y=MAX_Y,
        cell_size=CELL_SIZE_M,
        rows=GRID_ROWS,
        cols=GRID_COLS,
    )

    fov_spec = FoVSpec(
        half_left=FOV_WIDTH_M / 2.0,
        half_right=FOV_WIDTH_M / 2.0,
        forward=FOV_FORWARD_M,
        backward=FOV_BACKWARD_M,
    )

    env_mgr = GridEnvManager(grid_spec, fov_spec)

    detector = DatasetDetector(
        camera=camera,
        model_to_class=MODEL_TO_CLASS,
        min_box_pixels=4,
        enable_segmentation_debug=SAVE_SEGMENTATION_DEBUG,
    )
    detector.enable(timestep)

    base_dataset_dir = os.path.join(os.path.dirname(__file__), "datasets", DATASET_NAME)
    paths = make_dataset_run_dir(base_dataset_dir, seed=seed, policy_name=DATASET_POLICY)
    run_dir = paths["run_dir"]

    print(f"[DATASET] run_dir={run_dir}", flush=True)

    manifest = {
        "dataset_name": DATASET_NAME,
        "seed": seed,
        "policy": DATASET_POLICY,
        "capture_every_n_steps": CAPTURE_EVERY_N_STEPS,
        "max_samples": MAX_SAMPLES,
        "save_empty_labels": SAVE_EMPTY_LABELS,
        "save_sample_metadata": SAVE_SAMPLE_METADATA,
        "save_segmentation_debug": SAVE_SEGMENTATION_DEBUG,
        "class_mapping": MODEL_TO_CLASS,
        "camera": {
            "width": int(camera.getWidth()),
            "height": int(camera.getHeight()),
            "fov": float(camera.getFov()),
        },
        "objects_manifest_path": dataset_objects_path,
        "sample_count": 0,
    }
    _atomic_write_json(os.path.join(run_dir, "run_manifest.json"), manifest)

    pid_controller = pid_velocity_fixed_height_controller()

    height_desired = FLYING_ATTITUDE
    autonomous_mode = False
    autonomy_start_time = None
    step_count = 0
    sample_count = 0

    print("\n====== Dataset Collection Controller ======\n")
    print("DATASET_POLICY =", DATASET_POLICY)
    print("Controls: Arrow keys planar, Q/E yaw, W/S altitude")
    print("Press A to enable autonomous mode, D to disable\n")

    while robot.step(timestep) != -1:
        step_count += 1
        observation = devices.read()
        dt = observation["dt"]
        time_now = float(observation["time"])

        dataset_state = _read_json(
            dataset_state_path,
            {"status": "running", "reason": None},
        )
        supervisor_status = str(dataset_state.get("status", "running"))

        if AUTO_START_AUTONOMY and (not autonomous_mode):
            if time_now >= AUTONOMY_START_DELAY_S and supervisor_status == "running":
                autonomous_mode = True
                autonomy_start_time = float(time_now)
                print(f"Autonomous mode: ON (auto at t={time_now:.1f}s)", flush=True)

        forward_desired = 0.0
        sideways_desired = 0.0
        yaw_desired = 0.0
        height_diff_desired = 0.0

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
                yaw_desired = +1.0
            elif key == ord("E"):
                yaw_desired = -1.0
            elif key == ord("W"):
                height_diff_desired = 0.1
            elif key == ord("S"):
                height_diff_desired = -0.1
            elif key == ord("A"):
                if not autonomous_mode and supervisor_status == "running":
                    autonomous_mode = True
                    autonomy_start_time = float(time_now)
                    print("Autonomous mode: ON", flush=True)
            elif key == ord("D"):
                if autonomous_mode:
                    autonomous_mode = False
                    print("Autonomous mode: OFF", flush=True)
            key = keyboard.getKey()

        height_desired = max(0.1, height_desired + height_diff_desired * dt)

        x_pos = observation["gps"]["x"]
        y_pos = observation["gps"]["y"]
        yaw = observation["imu"]["yaw"]

        coverages = env_mgr.compute_cell_coverages(x_pos, y_pos, yaw)
        visible_mask = env_mgr.coverage_to_visible_mask(
            coverages,
            threshold=COVERAGE_UPDATE_THRESHOLD,
        )
        current_cell_rc = grid_spec.rowcol_from_xy(x_pos, y_pos)
        current_cell_cxcy = rc_to_cxcy(current_cell_rc)

        obs_for_policy = {
            **observation,
            "step": step_count,
            "pose": {"x": x_pos, "y": y_pos, "yaw": yaw},
            "grid_spec": {
                "min_x": grid_spec.min_x,
                "max_x": grid_spec.max_x,
                "min_y": grid_spec.min_y,
                "max_y": grid_spec.max_y,
                "cell_size": grid_spec.cell_size,
                "rows": grid_spec.rows,
                "cols": grid_spec.cols,
            },
            "grid": {
                "current_cell": current_cell_cxcy,
                "coverages": dict_rc_to_cxcy(coverages),
                "visible_mask": dict_rc_to_cxcy(visible_mask),
                "coverage_threshold": COVERAGE_UPDATE_THRESHOLD,
            },
            "fov_spec": {
                "half_left": fov_spec.half_left,
                "half_right": fov_spec.half_right,
                "forward": fov_spec.forward,
                "backward": fov_spec.backward,
            },
        }

        if autonomous_mode and supervisor_status == "running":
            policy_action = policy.compute_action(obs_for_policy)
            forward_desired = float(policy_action.get("forward", 0.0))
            sideways_desired = float(policy_action.get("sideways", 0.0))
            yaw_desired = float(policy_action.get("yaw_rate", 0.0))

        motor_power = pid_controller.pid(
            dt,
            forward_desired,
            sideways_desired,
            yaw_desired,
            height_desired,
            observation["imu"]["roll"],
            observation["imu"]["pitch"],
            observation["gyro"]["yaw_rate"],
            observation["gps"]["z"],
            observation["velocity"]["vx"],
            observation["velocity"]["vy"],
        )
        devices.set_motor_speeds(motor_power)

        should_capture = (
            autonomous_mode
            and supervisor_status == "running"
            and (step_count % CAPTURE_EVERY_N_STEPS == 0)
            and (MAX_SAMPLES is None or sample_count < MAX_SAMPLES)
        )

        if should_capture:
            annotations = detector.get_annotations()

            if annotations or SAVE_EMPTY_LABELS:
                base_name = f"seed{seed:04d}_step{step_count:07d}_sample{sample_count:06d}"
                image_path = os.path.join(paths["images_dir"], base_name + ".png")
                label_path = os.path.join(paths["labels_dir"], base_name + ".txt")
                meta_path = os.path.join(paths["meta_dir"], base_name + ".json")

                camera.saveImage(image_path, 100)
                detector.save_yolo_txt(label_path, annotations)

                if SAVE_SEGMENTATION_DEBUG:
                    seg_path = os.path.join(paths["seg_dir"], base_name + "_seg.png")
                    detector.save_segmentation_debug(seg_path, 100)

                if SAVE_SAMPLE_METADATA:
                    sample_meta = {
                        "seed": seed,
                        "policy": DATASET_POLICY,
                        "step": int(step_count),
                        "time_s": float(time_now),
                        "autonomy_start_time_s": autonomy_start_time,
                        "drone_pose": {
                            "x": float(x_pos),
                            "y": float(y_pos),
                            "z": float(observation["gps"]["z"]),
                            "yaw": float(yaw),
                            "roll": float(observation["imu"]["roll"]),
                            "pitch": float(observation["imu"]["pitch"]),
                        },
                        "current_cell": current_cell_cxcy,
                        "image_filename": os.path.basename(image_path),
                        "label_filename": os.path.basename(label_path),
                        "num_annotations": int(len(annotations)),
                        "annotations": annotations,
                    }
                    _atomic_write_json(meta_path, sample_meta)

                sample_count += 1
                manifest["sample_count"] = int(sample_count)
                _atomic_write_json(os.path.join(run_dir, "run_manifest.json"), manifest)

                print(
                    f"[CAPTURE] sample={sample_count:05d} step={step_count} anns={len(annotations)} file={os.path.basename(image_path)}",
                    flush=True,
                )

        if supervisor_status != "running":
            print(f"[END] supervisor_status={supervisor_status}", flush=True)
            break