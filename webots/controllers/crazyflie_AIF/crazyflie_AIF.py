# crazyflie_AIF.py

import os
from datetime import datetime

from controller import Keyboard
from controller import Robot

from devices import CrazyflieDevices
from pid.pid_controller import pid_velocity_fixed_height_controller
from policies.wall_following import WallFollowingPolicy
from utils.capture_util import save_capture
from utils.logging_AIF import ExperimentLogger
from vision.detector import Detector

from env.env_manager import GridSpec, FoVSpec, GridEnvManager


# ---- Run toggles ----
ENABLE_LOGGING = True
ENABLE_CAPTURES = True
LOG_EVERY_N_STEPS = 10
CAPTURE_EVERY_N_STEPS = 20

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
FOV_WIDTH_M = 1.35
FOV_FORWARD_M = 0.55
FOV_BACKWARD_M = 0.45

# ---- FoV → cell thresholds ----
COVERAGE_UPDATE_THRESHOLD = 0.30

# ---- Single debug config (controls controller + policy prints) ----
DEBUG = {
    "every_n": 50,

    # Controller
    "fov": True,          # prints [FOV DEBUG]
    "cmd": True,          # prints [cmd]

    # Policy
    "policy_grid": True,       # prints [POLICY GRID]
    "policy_detections": False # print top-1 detection only when non-empty
}


def rc_to_cxcy(rc):
    if rc is None:
        return None
    r, c = rc
    return (c, r)


def dict_rc_to_cxcy(d):
    return {(c, r): v for (r, c), v in d.items()}


def make_run_dir(base_log_dir: str, prefix: str = "log_run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_log_dir, f"{prefix}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def load_policy(name, debug=None):
    registry = {
        "wall_following": WallFollowingPolicy,
    }
    if name not in registry:
        raise ValueError(f"Unknown policy '{name}'. Available: {', '.join(sorted(registry))}")
    return registry[name](debug=debug)


if __name__ == "__main__":
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    devices = CrazyflieDevices(robot, timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)

    policy_name = os.environ.get("AIF_POLICY", "wall_following")
    policy = load_policy(policy_name, debug=DEBUG)
    detector = Detector()

    # ---- Grid spec ----
    grid_spec = GridSpec(
        min_x=MIN_X,
        max_x=MAX_X,
        min_y=MIN_Y,
        max_y=MAX_Y,
        cell_size=CELL_SIZE_M,
        rows=GRID_ROWS,
        cols=GRID_COLS,
    )

    # ---- FoV spec ----
    fov_spec = FoVSpec(
        half_left=FOV_WIDTH_M / 2.0,
        half_right=FOV_WIDTH_M / 2.0,
        forward=FOV_FORWARD_M,
        backward=FOV_BACKWARD_M,
    )

    env_mgr = GridEnvManager(grid_spec, fov_spec)

    print("\n[FoV setup]")
    print(f"  cell_size={CELL_SIZE_M}m grid={GRID_ROWS}x{GRID_COLS} "
          f"x∈[{MIN_X:.2f},{MAX_X:.2f}] y∈[{MIN_Y:.2f},{MAX_Y:.2f}]")
    print(f"  footprint width={FOV_WIDTH_M:.2f}m forward={FOV_FORWARD_M:.2f}m backward={FOV_BACKWARD_M:.2f}m")

    base_log_dir = os.path.join(os.path.dirname(__file__), "logs")

    run_dir = None
    if ENABLE_LOGGING or ENABLE_CAPTURES:
        run_dir = make_run_dir(base_log_dir, prefix=f"log_run_{policy_name}")
        print("Run output dir:", run_dir)

    logger = ExperimentLogger(
        enabled=ENABLE_LOGGING,
        log_dir=run_dir,
        grid_spec=grid_spec,
    )

    pid_controller = pid_velocity_fixed_height_controller()

    height_desired = FLYING_ATTITUDE
    autonomous_mode = False
    step_count = 0

    print("\n====== AIF Controller ======\n")
    print("AIF_POLICY =", policy_name)
    print("Controls: Arrow keys for planar motion, Q/E yaw, W/S altitude")
    print("Press A to enable autonomous mode, D to disable\n")

    try:
        while robot.step(timestep) != -1:
            step_count += 1
            observation = devices.read()
            dt = observation["dt"]

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
                    yaw_desired = +1
                elif key == ord("E"):
                    yaw_desired = -1
                elif key == ord("W"):
                    height_diff_desired = 0.1
                elif key == ord("S"):
                    height_diff_desired = -0.1
                elif key == ord("A"):
                    if not autonomous_mode:
                        autonomous_mode = True
                        print("Autonomous mode: ON")
                elif key == ord("D"):
                    if autonomous_mode:
                        autonomous_mode = False
                        print("Autonomous mode: OFF")
                key = keyboard.getKey()

            height_desired = max(0.1, height_desired + height_diff_desired * dt)

            detector_output = detector.process(
                observation["camera"]["image"],
                width=observation["camera"]["width"],
                height=observation["camera"]["height"],
                timestamp=observation["camera"]["time"],
            )

            if ENABLE_CAPTURES and (step_count % CAPTURE_EVERY_N_STEPS == 0):
                save_capture(
                    observation["camera"]["image"],
                    width=observation["camera"]["width"],
                    height=observation["camera"]["height"],
                    timestamp=observation["camera"]["time"],
                    output_dir=os.path.join(run_dir, "captures"),
                )

            x_pos = observation["gps"]["x"]
            y_pos = observation["gps"]["y"]
            yaw = observation["imu"]["yaw"]

            # ---- Detection -> grid confidence map (row,col internally) ----
            det_conf_rc = {}
            img_w = observation["camera"]["width"]
            img_h = observation["camera"]["height"]

            for det in detector_output.detections:
                bbox = det.get("bbox_xyxy")
                conf = float(det.get("confidence", 0.0))
                cell_rc = env_mgr.detection_cell_from_bbox(
                    bbox_xyxy=bbox,
                    img_w=img_w,
                    img_h=img_h,
                    drone_x=x_pos,
                    drone_y=y_pos,
                    drone_yaw=yaw,
                )
                if cell_rc is None:
                    continue
                prev = det_conf_rc.get(cell_rc, 0.0)
                if conf > prev:
                    det_conf_rc[cell_rc] = conf

            det_conf_cxcy = dict_rc_to_cxcy(det_conf_rc)

            coverages = env_mgr.compute_cell_coverages(x_pos, y_pos, yaw)
            visible_mask = env_mgr.coverage_to_visible_mask(
                coverages, threshold=COVERAGE_UPDATE_THRESHOLD
            )

            current_cell_rc = grid_spec.rowcol_from_xy(x_pos, y_pos)
            current_cell_cxcy = rc_to_cxcy(current_cell_rc)

            coverages_cxcy = dict_rc_to_cxcy(coverages)
            visible_mask_cxcy = dict_rc_to_cxcy(visible_mask)

            # ---- Controller debug: only FoV geometry sanity ----
            if DEBUG.get("fov", False) and (step_count % int(DEBUG.get("every_n", 50)) == 0):
                top_rc = sorted(coverages.items(), key=lambda kv: kv[1], reverse=True)[:12]
                num_vis = sum(1 for v in visible_mask.values() if v)

                print("\n[FOV DEBUG]")
                print(f"  step={step_count}  pos=({x_pos:.2f},{y_pos:.2f}) yaw={yaw:.3f} rad ({yaw * 180.0 / 3.14159:.1f} deg)")
                print(f"  current_cell(cx,cy)={current_cell_cxcy}")
                print(f"  coverages: {len(coverages)} cells intersect FoV; {num_vis} >= {COVERAGE_UPDATE_THRESHOLD:.2f}")

                for (r, c), frac in top_rc:
                    print(f"    cell=(cx={c},cy={r}) coverage={frac*100:5.1f}% visible={visible_mask[(r,c)]}")

            # ---- Logging ----
            if ENABLE_LOGGING and (step_count % LOG_EVERY_N_STEPS == 0):
                logger.log_occupancy(
                    time_s=observation["time"],
                    x_m=x_pos,
                    y_m=y_pos,
                    dt_s=dt,
                )

            # ---- Build coherent policy observation ----
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
                "detections": detector_output.detections,
                "grid": {
                    "current_cell": current_cell_cxcy,
                    "coverages": coverages_cxcy,
                    "visible_mask": visible_mask_cxcy,
                    "coverage_threshold": COVERAGE_UPDATE_THRESHOLD,
                    "det_confidence": det_conf_cxcy,
                },
            }

            if autonomous_mode:
                policy_action = policy.compute_action(obs_for_policy)
                forward_desired = policy_action.get("forward", 0.0)
                sideways_desired = policy_action.get("sideways", 0.0)
                yaw_desired = policy_action.get("yaw_rate", 0.0)

            # ---- Controller debug: only actuation sanity ----
            if DEBUG.get("cmd", False) and (step_count % int(DEBUG.get("every_n", 50)) == 0):
                print(f"[cmd] step={step_count} auto={autonomous_mode} "
                      f"fwd={forward_desired:.2f} side={sideways_desired:.2f} yaw={yaw_desired:.2f} "
                      f"h_des={height_desired:.2f} z={observation['gps']['z']:.2f}")

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

            if ENABLE_LOGGING and (step_count % LOG_EVERY_N_STEPS == 0):
                logger.log_step(
                    time=observation["time"],
                    forward=forward_desired,
                    sideways=sideways_desired,
                    yaw_rate=yaw_desired,
                    altitude=observation["gps"]["z"],
                    detections=len(detector_output.detections),
                )

    finally:
        logger.close()
