import os

from controller import Keyboard
from controller import Robot

from devices import CrazyflieDevices
from pid.pid_controller import pid_velocity_fixed_height_controller
from policies.wall_following import WallFollowingPolicy
from utils.logging import ExperimentLogger
from vision.detector import Detector


FLYING_ATTITUDE = 1.0


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
