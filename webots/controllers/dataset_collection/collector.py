from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


CONTROLLERS_DIR = Path(__file__).resolve().parents[1]
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from common.shared_state import SharedStateStore, atomic_write_json, read_json

from .class_config import get_dataset_class_config, normalize_collection_environment
from .collection_config import get_collection_config
from .segmentation_annotator import SegmentationAnnotator


AIF_CONTROLLER_DIR = CONTROLLERS_DIR / "crazyflie_AIF"

# Dataset controllers are thin Webots entry points. Flight control, grid
# geometry and naive policies are imported from the canonical crazyflie_AIF
# controller tree so the collector directories no longer need private copies.
if str(AIF_CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(AIF_CONTROLLER_DIR))

SHARED_STATE = SharedStateStore(CONTROLLERS_DIR / "shared")


FLYING_ATTITUDE = 1.5
AUTO_START_AUTONOMY = True
AUTONOMY_START_DELAY_S = 15.0
STARTUP_WAIT_TIMEOUT_S = 10.0
STARTUP_WAIT_POLL_S = 0.05

ARENA_WIDTH_M = 5.5
ARENA_HEIGHT_M = 6.5
CELL_SIZE_M = 0.5
GRID_COLS = int(ARENA_WIDTH_M / CELL_SIZE_M)
GRID_ROWS = int(ARENA_HEIGHT_M / CELL_SIZE_M)
MIN_X = -ARENA_WIDTH_M / 2.0
MIN_Y = -ARENA_HEIGHT_M / 2.0
MAX_X = ARENA_WIDTH_M / 2.0
MAX_Y = ARENA_HEIGHT_M / 2.0

FOV_WIDTH_M = 1.40
FOV_FORWARD_M = 0.55
FOV_BACKWARD_M = 0.45
COVERAGE_UPDATE_THRESHOLD = 0.05


def rc_to_cxcy(rc):
    if rc is None:
        return None
    row, col = rc
    return (col, row)


def dict_rc_to_cxcy(values):
    return {(col, row): value for (row, col), value in values.items()}


def _object_records_by_id(dataset_objects: dict, class_config) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for obj in dataset_objects.get("objects", []):
        if not isinstance(obj, dict) or not class_config.is_labelable_record(obj):
            continue
        object_id = obj.get("id")
        if object_id is not None:
            records[int(object_id)] = obj
    return records


def _make_dataset_dirs(base_dir: str, collection_name: str, *, save_segmentation_debug: bool) -> dict:
    dataset_dir = os.path.join(base_dir, collection_name)
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    segmentation_dir = os.path.join(dataset_dir, "segmentation_debug")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    if save_segmentation_debug:
        os.makedirs(segmentation_dir, exist_ok=True)
    return {
        "dataset_dir": dataset_dir,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "segmentation_dir": segmentation_dir,
        "manifest_path": os.path.join(dataset_dir, "manifest.jsonl"),
        "metadata_path": os.path.join(dataset_dir, "metadata.json"),
        "yaml_path": os.path.join(dataset_dir, "target_objects.yaml"),
    }


def _write_dataset_yaml(path: str, dataset_dir: str, class_config) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(class_config.yolo_yaml_text(dataset_dir))


def _append_jsonl(path: str | os.PathLike[str], payload: Any) -> None:
    """Append one JSON object to a dataset-local JSONL manifest.

    Dataset manifests are ordinary output artifacts, not controller/supervisor
    IPC, so this deliberately remains local to the collection implementation
    instead of being added to ``SharedStateStore``.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _update_dataset_metadata(
    paths: dict,
    *,
    environment: str,
    policy_name: str,
    collection_config,
    class_config,
    experiment_init: dict,
    dataset_objects: dict,
    sample_count: int,
    camera_metadata: dict,
) -> None:
    existing = read_json(paths["metadata_path"], {})
    run_key = f"seed{int(experiment_init.get('seed', -1)):04d}_run{int(experiment_init.get('run_index', 0)):03d}"
    runs = dict(existing.get("runs", {}))
    runs[run_key] = {
        "seed": int(experiment_init.get("seed", -1)),
        "run_index": int(experiment_init.get("run_index", 0)),
        "samples_collected": int(sample_count),
        "objects_manifest": dataset_objects.get("metadata", {}),
    }
    payload = {
        "dataset_type": collection_config.dataset_type,
        "environment": environment,
        "annotation_source": "recognition_segmentation_mask",
        "scene_variant": experiment_init.get("scene_variant"),
        "collection_name": experiment_init.get("collection_name"),
        "collection_start_time": experiment_init.get("collection_start_time"),
        "policy": policy_name,
        "capture_every_n_steps": int(collection_config.capture_every_n_steps),
        "samples_per_run": int(experiment_init.get("samples_per_run", collection_config.samples_per_run)),
        "class_id_to_name": class_config.class_id_to_name,
        "class_name_to_id": class_config.class_name_to_id,
        "model_to_class_id": class_config.model_to_class_id,
        "camera": camera_metadata,
        "runs": runs,
    }
    atomic_write_json(paths["metadata_path"], payload)


def _load_policy(name: str, *, seed: int):
    from policies.pseudo_random import PseudoRandomPolicy
    from policies.rotate_and_measure import RotateAndMeasurePolicy
    from policies.spiral import SpiralPolicy
    from policies.wall_following import WallFollowingPolicy

    if name == "pseudo_random":
        return PseudoRandomPolicy(seed=seed, debug=None)
    if name == "spiral":
        return SpiralPolicy(debug=None)
    if name == "wall_following":
        return WallFollowingPolicy(debug=None)
    if name == "rotate_and_measure":
        return RotateAndMeasurePolicy(debug=None)
    raise ValueError(f"Unknown dataset collection policy '{name}'.")


def _resolve_output_root(collection_config, controller_dir: Path) -> str:
    override = collection_config.output_root_override()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    # Preserve the historical convention that each Webots controller owns its
    # dataset output directory, while sharing the implementation itself.
    return str(controller_dir / "datasets")


def _validate_supervisor_environment(experiment_init: dict, expected_environment: str) -> None:
    declared = experiment_init.get("environment")
    if declared is None:
        # Legacy easy dataset supervisor files did not carry environment metadata.
        return
    actual = normalize_collection_environment(str(declared))
    if actual != expected_environment:
        raise RuntimeError(
            "Dataset collection environment mismatch:\n"
            f"  collector: {expected_environment}\n"
            f"  supervisor: {actual}"
        )


def run_collection(environment: str, *, controller_dir: str | os.PathLike[str]) -> None:
    environment = normalize_collection_environment(environment)
    collection_config = get_collection_config(environment)
    class_config = get_dataset_class_config(environment)
    controller_dir = Path(controller_dir).resolve()

    from controller import Keyboard, Robot
    from devices import CrazyflieDevices
    from env.env_manager import FoVSpec, GridEnvManager, GridSpec
    from pid.pid_controller import pid_velocity_fixed_height_controller

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    devices = CrazyflieDevices(robot, timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    experiment_init_filename = collection_config.experiment_init_filename
    dataset_objects_filename = collection_config.dataset_objects_filename
    status_filename = collection_config.status_filename
    collection_state_filename = collection_config.collection_state_filename

    experiment_init = SHARED_STATE.wait(
        experiment_init_filename,
        validator=lambda data: isinstance(data.get("drone_spawn"), dict),
        timeout_s=STARTUP_WAIT_TIMEOUT_S,
        poll_s=STARTUP_WAIT_POLL_S,
    )
    if experiment_init is None:
        raise FileNotFoundError(
            f"Missing experiment init file: {SHARED_STATE.path(experiment_init_filename)}"
        )
    _validate_supervisor_environment(experiment_init, environment)

    dataset_objects = SHARED_STATE.wait(
        dataset_objects_filename,
        validator=lambda data: isinstance(data.get("objects"), list),
        timeout_s=STARTUP_WAIT_TIMEOUT_S,
        poll_s=STARTUP_WAIT_POLL_S,
    )
    if dataset_objects is None:
        raise FileNotFoundError(
            f"Missing dataset objects file: {SHARED_STATE.path(dataset_objects_filename)}"
        )

    seed = int(experiment_init.get("seed", -1))
    run_index = int(experiment_init.get("run_index", 0))
    policy_name = str(experiment_init.get("policy_name") or collection_config.policy)
    samples_per_run = int(experiment_init.get("samples_per_run", collection_config.samples_per_run))
    collection_start_time = str(
        experiment_init.get("collection_start_time")
        or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    collection_name = str(
        experiment_init.get("collection_name")
        or f"{collection_config.collection_prefix}_{policy_name}_{collection_start_time}"
    )

    policy = _load_policy(policy_name, seed=seed)

    drone_spawn = experiment_init.get("drone_spawn", {}) or {}
    print(
        f"[DATASET COLLECTOR] environment={environment} seed={seed} run_index={run_index} "
        f"policy={policy_name} collection={collection_name} "
        f"drone=({float(drone_spawn.get('x', 0.0)):.3f},{float(drone_spawn.get('y', 0.0)):.3f})",
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

    annotator = SegmentationAnnotator(
        camera=camera,
        model_to_class=class_config.model_to_class_id,
        class_id_to_name=class_config.class_id_to_name,
        object_records_by_id=_object_records_by_id(dataset_objects, class_config),
        min_box_pixels=collection_config.min_box_pixels,
        enable_segmentation_debug=collection_config.save_segmentation_debug,
        desk_class_name=class_config.desk_class_name or "desk",
    )
    # Segmentation is mandatory. If the camera does not expose it, collection
    # stops here instead of silently producing empty/incorrect labels.
    annotator.enable(timestep)

    paths = _make_dataset_dirs(
        _resolve_output_root(collection_config, controller_dir),
        collection_name,
        save_segmentation_debug=collection_config.save_segmentation_debug,
    )
    _write_dataset_yaml(paths["yaml_path"], paths["dataset_dir"], class_config)

    camera_metadata = {
        "width": int(camera.getWidth()),
        "height": int(camera.getHeight()),
        "fov": float(camera.getFov()),
    }
    if collection_config.save_metadata:
        _update_dataset_metadata(
            paths,
            environment=environment,
            policy_name=policy_name,
            collection_config=collection_config,
            class_config=class_config,
            experiment_init=experiment_init,
            dataset_objects=dataset_objects,
            sample_count=0,
            camera_metadata=camera_metadata,
        )

    pid_controller = pid_velocity_fixed_height_controller()
    height_desired = FLYING_ATTITUDE
    autonomous_mode = False
    autonomy_start_time = None
    step_count = 0
    sample_count = 0
    completion_written = False

    print("\n====== Dataset Collection Controller ======")
    print("ENVIRONMENT =", environment)
    print("DATASET_POLICY =", policy_name)
    print("ANNOTATION_SOURCE = recognition_segmentation_mask")
    print("CAPTURE_EVERY_N_STEPS =", collection_config.capture_every_n_steps)
    print("SAMPLES_PER_RUN =", samples_per_run)
    print("DATASET_DIR =", paths["dataset_dir"], "\n")

    while robot.step(timestep) != -1:
        step_count += 1
        observation = devices.read()
        dt = observation["dt"]
        time_now = float(observation["time"])

        status_payload = SHARED_STATE.read(status_filename, {"status": "running", "reason": None})
        supervisor_status = str(status_payload.get("status", "running"))

        if AUTO_START_AUTONOMY and not autonomous_mode:
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
            elif key == ord("A") and not autonomous_mode and supervisor_status == "running":
                autonomous_mode = True
                autonomy_start_time = float(time_now)
                print("Autonomous mode: ON", flush=True)
            elif key == ord("D") and autonomous_mode:
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
        current_cell_cxcy = rc_to_cxcy(grid_spec.rowcol_from_xy(x_pos, y_pos))

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
            and step_count % collection_config.capture_every_n_steps == 0
            and sample_count < samples_per_run
        )

        if should_capture:
            # Do not catch segmentation failures here. A broken segmentation
            # pipeline must abort collection rather than create empty labels.
            annotations = annotator.get_annotations()

            if annotations or collection_config.save_empty_labels:
                entry_id = f"seed{seed:04d}_frame{step_count:07d}"
                base_name = f"{entry_id}_{collection_start_time}"
                image_path = os.path.join(paths["images_dir"], base_name + ".png")
                label_path = os.path.join(paths["labels_dir"], base_name + ".txt")

                camera.saveImage(image_path, 100)
                annotator.save_yolo_txt(label_path, annotations)

                if collection_config.save_segmentation_debug:
                    annotator.save_segmentation_debug(
                        os.path.join(paths["segmentation_dir"], base_name + "_seg.png"),
                        100,
                    )

                if collection_config.save_manifest:
                    manifest_entry: dict[str, Any] = {
                        "entry_id": entry_id,
                        "environment": environment,
                        "scene_variant": experiment_init.get("scene_variant"),
                        "seed": seed,
                        "run_index": run_index,
                        "frame_index": step_count,
                        "time_s": time_now,
                        "image_filename": os.path.basename(image_path),
                        "label_filename": os.path.basename(label_path),
                        "policy": policy_name,
                        "annotation_source": "recognition_segmentation_mask",
                        "drone_pose": {
                            "x": float(x_pos),
                            "y": float(y_pos),
                            "z": float(observation["gps"]["z"]),
                            "yaw": float(yaw),
                            "roll": float(observation["imu"]["roll"]),
                            "pitch": float(observation["imu"]["pitch"]),
                        },
                        "current_cell": current_cell_cxcy,
                        "num_annotations": len(annotations),
                        "recognized_class_names": sorted({a["class_name"] for a in annotations}),
                        "annotations": annotations,
                    }
                    _append_jsonl(paths["manifest_path"], manifest_entry)

                sample_count += 1
                if collection_config.save_metadata:
                    _update_dataset_metadata(
                        paths,
                        environment=environment,
                        policy_name=policy_name,
                        collection_config=collection_config,
                        class_config=class_config,
                        experiment_init={
                            **experiment_init,
                            "collection_name": collection_name,
                            "collection_start_time": collection_start_time,
                        },
                        dataset_objects=dataset_objects,
                        sample_count=sample_count,
                        camera_metadata=camera_metadata,
                    )

                print(
                    f"[DATASET CAPTURE] env={environment} seed={seed} "
                    f"sample={sample_count:05d}/{samples_per_run} step={step_count} "
                    f"anns={len(annotations)} file={os.path.basename(image_path)}",
                    flush=True,
                )

        if (
            collection_state_filename is not None
            and not completion_written
            and sample_count >= samples_per_run
        ):
            SHARED_STATE.write(
                collection_state_filename,
                {
                    "environment": environment,
                    "seed": seed,
                    "run_index": run_index,
                    "status": "complete",
                    "num_examples_collected": sample_count,
                    "dataset_folder": paths["dataset_dir"],
                    "manifest_path": paths["manifest_path"],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )
            completion_written = True
            print(
                f"[DATASET COMPLETE] wrote {SHARED_STATE.path(collection_state_filename)}",
                flush=True,
            )

        if supervisor_status != "running":
            print(
                f"[DATASET END] environment={environment} supervisor_status={supervisor_status}",
                flush=True,
            )
            break
