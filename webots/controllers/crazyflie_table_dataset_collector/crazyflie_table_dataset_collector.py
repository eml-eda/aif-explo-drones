# controllers/crazyflie_table_dataset_collector/crazyflie_table_dataset_collector.py

from __future__ import annotations

import traceback
import json
import os
import sys
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

HERE = os.path.dirname(__file__)
SHARED_DIR = os.path.abspath(os.path.join(HERE, "..", "shared"))
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

from table_dataset_classes import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, MODEL_TO_CLASS_ID, yolo_yaml_text

# ---------- Dataset collection config ----------
DATASET_POLICY = os.environ.get("TABLE_DATASET_POLICY", os.environ.get("DATASET_POLICY", "pseudo_random"))
CAPTURE_EVERY_N_STEPS = int(os.environ.get("TABLE_DATASET_CAPTURE_EVERY_N_STEPS", "300"))
MAX_SAMPLES_PER_RUN = int(os.environ.get("TABLE_DATASET_SAMPLES_PER_RUN", "400"))
SAVE_EMPTY_LABELS = os.environ.get("TABLE_DATASET_SAVE_EMPTY_LABELS", "1") != "0"
SAVE_SEGMENTATION_DEBUG = 1
SAVE_MANIFEST = 0
SAVE_METADATA = 0

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
EXPERIMENT_INIT_FILENAME = "table_dataset_experiment_init.json"
DATASET_OBJECTS_FILENAME = "table_dataset_objects.json"
TASK_STATE_FILENAME = "table_dataset_task_state.json"
COLLECTION_STATE_FILENAME = "table_dataset_collection_state.json"


def rc_to_cxcy(rc):
    if rc is None:
        return None
    r, c = rc
    return (c, r)


def dict_rc_to_cxcy(d):
    return {(c, r): v for (r, c), v in d.items()}


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


def _append_jsonl(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


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


def _write_target_objects_yaml(dataset_dir: str) -> None:
    yaml_path = os.path.join(dataset_dir, "target_objects.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yolo_yaml_text(dataset_dir))


def _object_records_by_id(dataset_objects: dict) -> dict[int, dict]:
    records = {}
    for obj in dataset_objects.get("objects", []):
        obj_id = obj.get("id")
        if obj_id is not None:
            records[int(obj_id)] = obj
    return records


def make_dataset_dirs(base_dir: str, collection_name: str) -> dict:
    dataset_dir = os.path.join(base_dir, collection_name)
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    seg_dir = os.path.join(dataset_dir, "segmentation_debug")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    if SAVE_SEGMENTATION_DEBUG:
        os.makedirs(seg_dir, exist_ok=True)
    return {
        "dataset_dir": dataset_dir,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "seg_dir": seg_dir,
        "manifest_path": os.path.join(dataset_dir, "manifest.jsonl"),
        "metadata_path": os.path.join(dataset_dir, "metadata.json"),
    }


def update_dataset_metadata(paths: dict, *, experiment_init: dict, dataset_objects: dict, sample_count: int) -> None:
    existing = _read_json(paths["metadata_path"], {})
    run_key = f"seed{int(experiment_init.get('seed', -1)):04d}"
    runs = dict(existing.get("runs", {}))
    runs[run_key] = {
        "seed": int(experiment_init.get("seed", -1)),
        "run_index": int(experiment_init.get("run_index", -1)),
        "samples_collected": int(sample_count),
        "objects_manifest": dataset_objects.get("metadata", {}),
    }
    payload = {
        "dataset_type": "table_desk_yolo_detection",
        "collection_name": experiment_init.get("collection_name"),
        "collection_start_time": experiment_init.get("collection_start_time"),
        "policy": experiment_init.get("policy_name", DATASET_POLICY),
        "samples_per_run": int(experiment_init.get("samples_per_run", MAX_SAMPLES_PER_RUN)),
        "class_id_to_name": CLASS_ID_TO_NAME,
        "class_name_to_id": CLASS_NAME_TO_ID,
        "model_to_class_id": MODEL_TO_CLASS_ID,
        "camera": existing.get("camera", {}),
        "runs": runs,
    }
    _atomic_write_json(paths["metadata_path"], payload)


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
    task_state_path = _shared_path(TASK_STATE_FILENAME)
    collection_state_path = _shared_path(COLLECTION_STATE_FILENAME)

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
    run_index = int(experiment_init.get("run_index", 0))
    collection_name = str(
        experiment_init.get("collection_name")
        or f"collection_table_{DATASET_POLICY}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    samples_per_run = int(experiment_init.get("samples_per_run", MAX_SAMPLES_PER_RUN))
    policy = load_policy(DATASET_POLICY, debug=None, seed=seed)

    drone_spawn = experiment_init.get("drone_spawn", {}) or {}
    drone_spawn_x = float(drone_spawn.get("x", 0.0))
    drone_spawn_y = float(drone_spawn.get("y", 0.0))
    drone_spawn_yaw = float(drone_spawn.get("yaw", 0.0))

    print(
        f"[TABLE COLLECTOR INIT] seed={seed} run_index={run_index} "
        f"collection={collection_name} drone=({drone_spawn_x:.3f},{drone_spawn_y:.3f})",
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
    fov_spec = FoVSpec(
        half_left=FOV_WIDTH_M / 2.0,
        half_right=FOV_WIDTH_M / 2.0,
        forward=FOV_FORWARD_M,
        backward=FOV_BACKWARD_M,
    )
    env_mgr = GridEnvManager(grid_spec, fov_spec)

    detector = DatasetDetector(
        camera=camera,
        model_to_class=MODEL_TO_CLASS_ID,
        class_id_to_name=CLASS_ID_TO_NAME,
        object_records_by_id=_object_records_by_id(dataset_objects),
        min_box_pixels=2,
        enable_segmentation_debug=SAVE_SEGMENTATION_DEBUG,
    )
    detector.enable(timestep)

    base_dataset_dir = os.path.join(HERE, "datasets")
    paths = make_dataset_dirs(base_dataset_dir, collection_name)
    _write_target_objects_yaml(paths["dataset_dir"])

    if SAVE_METADATA:
        metadata = _read_json(paths["metadata_path"], {})
        metadata["camera"] = {
            "width": int(camera.getWidth()),
            "height": int(camera.getHeight()),
            "fov": float(camera.getFov()),
        }
        _atomic_write_json(paths["metadata_path"], metadata)
        update_dataset_metadata(paths, experiment_init=experiment_init, dataset_objects=dataset_objects, sample_count=0)

    pid_controller = pid_velocity_fixed_height_controller()
    height_desired = FLYING_ATTITUDE
    autonomous_mode = False
    autonomy_start_time = None
    step_count = 0
    sample_count = 0
    completion_written = False

    print("\n====== Table Dataset Collection Controller ======\n")
    print("DATASET_POLICY =", DATASET_POLICY)
    print("Controls: Arrow keys planar, Q/E yaw, W/S altitude")
    print("Press A to enable autonomous mode, D to disable\n")

    while robot.step(timestep) != -1:
        step_count += 1
        observation = devices.read()
        dt = observation["dt"]
        time_now = float(observation["time"])

        task_state = _read_json(task_state_path, {"status": "running", "reason": None})
        supervisor_status = str(task_state.get("status", "running"))

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
        visible_mask = env_mgr.coverage_to_visible_mask(coverages, threshold=COVERAGE_UPDATE_THRESHOLD)
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
            and sample_count < samples_per_run
        )

        if should_capture:
            try:
                annotations = detector.get_annotations()
            except Exception as exc:
                print("[TABLE DETECTOR ERROR] detector.get_annotations() failed:", repr(exc), flush=True)
                traceback.print_exc()
                annotations = []
            if annotations or SAVE_EMPTY_LABELS:
                entry_id = f"seed{seed:04d}_frame{step_count:07d}"
                base_name = f"{entry_id}_{experiment_init.get('collection_start_time', 'nostart')}"
                image_path = os.path.join(paths["images_dir"], base_name + ".png")
                label_path = os.path.join(paths["labels_dir"], base_name + ".txt")

                camera.saveImage(image_path, 100)
                detector.save_yolo_txt(label_path, annotations)

                if SAVE_SEGMENTATION_DEBUG:
                    detector.save_segmentation_debug(os.path.join(paths["seg_dir"], base_name + "_seg.png"), 100)

                contains_desk = any(int(a["class_id"]) == CLASS_NAME_TO_ID["desk"] for a in annotations)
                contains_target = any(int(a["class_id"]) in (CLASS_NAME_TO_ID["target_bottle"], CLASS_NAME_TO_ID["target_can"]) for a in annotations)
                visible_empty_desk = any(a.get("is_empty_desk", False) for a in annotations)
                if SAVE_MANIFEST:
                    manifest_entry = {
                        "entry_id": entry_id,
                        "seed": int(seed),
                        "run_index": int(run_index),
                        "frame_index": int(step_count),
                        "time_s": float(time_now),
                        "image_filename": os.path.basename(image_path),
                        "label_filename": os.path.basename(label_path),
                        "policy": DATASET_POLICY,
                        "drone_pose": {
                            "x": float(x_pos),
                            "y": float(y_pos),
                            "z": float(observation["gps"]["z"]),
                            "yaw": float(yaw),
                            "roll": float(observation["imu"]["roll"]),
                            "pitch": float(observation["imu"]["pitch"]),
                        },
                        "current_cell": current_cell_cxcy,
                        "num_annotations": int(len(annotations)),
                        "recognized_class_names": sorted({a["class_name"] for a in annotations}),
                        "recognized_object_ids": [a["recognition_id"] for a in annotations if a.get("recognition_id") is not None],
                        "contains_desk": bool(contains_desk),
                        "contains_target": bool(contains_target),
                        "visible_empty_desk": bool(visible_empty_desk),
                        "annotations": annotations,
                    }
                    _append_jsonl(paths["manifest_path"], manifest_entry)

                sample_count += 1
                if SAVE_METADATA:
                    update_dataset_metadata(paths, experiment_init=experiment_init, dataset_objects=dataset_objects, sample_count=sample_count)

                print(
                    f"[TABLE CAPTURE] seed={seed} sample={sample_count:05d}/{samples_per_run} "
                    f"step={step_count} anns={len(annotations)} file={os.path.basename(image_path)}",
                    flush=True,
                )

        if (not completion_written) and sample_count >= samples_per_run:
            payload = {
                "seed": int(seed),
                "run_index": int(run_index),
                "status": "complete",
                "num_examples_collected": int(sample_count),
                "dataset_folder": paths["dataset_dir"],
                "manifest_path": paths["manifest_path"],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            _atomic_write_json(collection_state_path, payload)
            completion_written = True
            print(f"[TABLE COMPLETE] wrote {collection_state_path}", flush=True)

        if supervisor_status != "running":
            print(f"[TABLE END] supervisor_status={supervisor_status}", flush=True)
            break
