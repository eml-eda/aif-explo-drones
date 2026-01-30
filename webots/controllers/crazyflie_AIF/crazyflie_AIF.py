import os
from datetime import datetime

from controller import Keyboard
from controller import Robot

from devices import CrazyflieDevices
from pid.pid_controller import pid_velocity_fixed_height_controller
from policies.wall_following import WallFollowingPolicy
from utils.logging import ExperimentLogger
from vision.detector import Detector


FLYING_ATTITUDE = 1.0

ARENA_WIDTH_M = 5.5
ARENA_HEIGHT_M = 6.5
CELL_SIZE_M = 0.5
GRID_COLS = int(ARENA_WIDTH_M / CELL_SIZE_M)
GRID_ROWS = int(ARENA_HEIGHT_M / CELL_SIZE_M)
MIN_X = -ARENA_WIDTH_M / 2.0
MIN_Y = -ARENA_HEIGHT_M / 2.0
MAX_X = ARENA_WIDTH_M / 2.0
MAX_Y = ARENA_HEIGHT_M / 2.0


def _cell_from_position(x, y):
    if x < MIN_X or x >= MAX_X or y < MIN_Y or y >= MAX_Y:
        return None
    col = int((x - MIN_X) / CELL_SIZE_M)
    row = int((y - MIN_Y) / CELL_SIZE_M)
    if col < 0 or col >= GRID_COLS or row < 0 or row >= GRID_ROWS:
        return None
    return row, col


def load_policy(name):
    registry = {
        "wall_following": WallFollowingPolicy,
    }
    if name not in registry:
        raise ValueError(f"Unknown policy '{name}'. Available: {', '.join(sorted(registry))}")
    return registry[name]()


if __name__ == "__main__":
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    devices = CrazyflieDevices(robot, timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)

    policy_name = os.environ.get("AIF_POLICY", "wall_following")
    policy = load_policy(policy_name)
    detector = Detector()
    logger = ExperimentLogger(enabled=False)

    occupancy_time_s = [[0.0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    current_cell = None

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    occupancy_log_path = os.path.join(log_dir, f"occupancy_log_{timestamp}.csv")
    occupancy_summary_path = os.path.join(log_dir, f"occupancy_summary_{timestamp}.csv")
    occupancy_log = open(occupancy_log_path, "w", encoding="utf-8")
    occupancy_log.write(
        "time_s,x_m,y_m,row,col,dt_s,inside\n"
    )

    pid_controller = pid_velocity_fixed_height_controller()

    height_desired = FLYING_ATTITUDE
    autonomous_mode = True

    print("\n====== AIF Controller ======\n")
    print("AIF_POLICY =", policy_name)
    print("Controls: Arrow keys for planar motion, Q/E yaw, W/S altitude")
    print("Press A to enable autonomous mode, D to disable\n")

    while robot.step(timestep) != -1:
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

        x_pos = observation["gps"]["x"]
        y_pos = observation["gps"]["y"]
        cell = _cell_from_position(x_pos, y_pos)
        inside = cell is not None
        current_cell = cell
        if inside:
            row, col = cell
            occupancy_time_s[row][col] += dt

        if inside:
            row, col = cell
            occupancy_log.write(
                f"{observation['time']:.6f},{x_pos:.4f},{y_pos:.4f},{row},{col},{dt:.6f},1\n"
            )
        else:
            occupancy_log.write(
                f"{observation['time']:.6f},{x_pos:.4f},{y_pos:.4f},-1,-1,{dt:.6f},0\n"
            )

        if autonomous_mode:
            policy_action = policy.compute_action(
                {
                    **observation,
                    "detections": detector_output.detections,
                }
            )
            forward_desired = policy_action.get("forward", 0.0)
            sideways_desired = policy_action.get("sideways", 0.0)
            yaw_desired = policy_action.get("yaw_rate", 0.0)

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

        logger.log_step(
            time=observation["time"],
            forward=forward_desired,
            sideways=sideways_desired,
            yaw_rate=yaw_desired,
            altitude=observation["gps"]["z"],
            detections=len(detector_output.detections),
        )

    logger.close()

    occupancy_log.close()
    with open(occupancy_summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(
            "arena_width_m,arena_height_m,cell_size_m,min_x,max_x,min_y,max_y,rows,cols\n"
        )
        summary_file.write(
            f"{ARENA_WIDTH_M},{ARENA_HEIGHT_M},{CELL_SIZE_M},{MIN_X},{MAX_X},{MIN_Y},{MAX_Y},{GRID_ROWS},{GRID_COLS}\n"
        )
        summary_file.write("row,col,time_s\n")
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                summary_file.write(f"{row},{col},{occupancy_time_s[row][col]:.6f}\n")
