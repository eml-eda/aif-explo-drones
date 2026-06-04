# webots/controllers/fov_measure_controller/fov_measure_controller.py

from __future__ import annotations

import math
import os
import sys

from controller import Keyboard, Robot

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROLLERS_DIR = os.path.dirname(HERE)
AIF_CONTROLLER_DIR = os.path.join(CONTROLLERS_DIR, "crazyflie_AIF")
if AIF_CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, AIF_CONTROLLER_DIR)

from devices import CrazyflieDevices
from pid.pid_controller import pid_velocity_fixed_height_controller

# -----------------------------------------------------------------------------
# Purpose
# -----------------------------------------------------------------------------
# Manual controller for measuring the effective down-camera FoV footprint.
# Use with worldgen_fov_measure_supervisor.py.
#
# The controller prints every PRINT_EVERY_S seconds:
#   - drone pose / altitude
#   - Webots camera horizontal + derived vertical FoV
#   - theoretical nadir projection size at floor and table height
#   - empirical footprint scaling from your current manually-calibrated floor FoV
#
# This file does not write reports and does not interact with task/progress JSONs.

# ---- Manual flight / altitude ----
FLYING_ATTITUDE = float(os.environ.get("FOV_MEASURE_ALTITUDE_M", "1.5"))
HEIGHT_STEP_RATE_M_PER_S = float(os.environ.get("FOV_MEASURE_HEIGHT_STEP_RATE_M_PER_S", "0.25"))
PLANAR_SPEED_MPS = float(os.environ.get("FOV_MEASURE_PLANAR_SPEED_MPS", "0.5"))
YAW_RATE_RAD_S = float(os.environ.get("FOV_MEASURE_YAW_RATE_RAD_S", "1.0"))

# If GPS z is drone/body height but the camera optical center is offset, set this.
# For the current Crazyflie setup, leaving this at 0 keeps behavior consistent
# with the existing empirical FoV calibration.
CAMERA_Z_OFFSET_M = float(os.environ.get("FOV_MEASURE_CAMERA_Z_OFFSET_M", "0.0"))

# ---- Surface heights ----
FLOOR_HEIGHT_M = float(os.environ.get("FOV_MEASURE_FLOOR_HEIGHT_M", "0.0"))
TABLE_HEIGHT_M = float(os.environ.get("FOV_MEASURE_TABLE_HEIGHT_M", "0.72"))

# ---- Existing empirical floor-level footprint at REFERENCE_CAMERA_DISTANCE_M ----
# These defaults match the values currently used in crazyflie_AIF.py.
REFERENCE_CAMERA_DISTANCE_M = float(os.environ.get("FOV_MEASURE_REFERENCE_DISTANCE_M", "1.5"))
BASE_FOV_WIDTH_M = float(os.environ.get("FOV_MEASURE_BASE_WIDTH_M", "1.40"))
BASE_FOV_FORWARD_M = float(os.environ.get("FOV_MEASURE_BASE_FORWARD_M", "0.55"))
BASE_FOV_BACKWARD_M = float(os.environ.get("FOV_MEASURE_BASE_BACKWARD_M", "0.45"))

PRINT_EVERY_S = float(os.environ.get("FOV_MEASURE_PRINT_EVERY_S", "10.0"))


def _vertical_fov_from_horizontal(horizontal_fov_rad: float, width_px: int, height_px: int) -> float:
    if width_px <= 0 or height_px <= 0:
        return float("nan")
    return 2.0 * math.atan(math.tan(0.5 * horizontal_fov_rad) * (float(height_px) / float(width_px)))


def _theoretical_nadir_rect(distance_m: float, horizontal_fov_rad: float, vertical_fov_rad: float) -> dict:
    d = max(1e-6, float(distance_m))
    return {
        "width_m": 2.0 * d * math.tan(0.5 * horizontal_fov_rad),
        "height_m": 2.0 * d * math.tan(0.5 * vertical_fov_rad),
    }


def _empirical_scaled_footprint(distance_m: float) -> dict:
    scale = max(0.0, float(distance_m)) / max(1e-6, REFERENCE_CAMERA_DISTANCE_M)
    return {
        "scale": scale,
        "width_m": BASE_FOV_WIDTH_M * scale,
        "forward_m": BASE_FOV_FORWARD_M * scale,
        "backward_m": BASE_FOV_BACKWARD_M * scale,
        "total_depth_m": (BASE_FOV_FORWARD_M + BASE_FOV_BACKWARD_M) * scale,
    }


def _print_fov_report(*, camera, obs: dict, height_desired: float) -> None:
    cam_w = int(camera.getWidth())
    cam_h = int(camera.getHeight())
    h_fov = float(camera.getFov())
    v_fov = _vertical_fov_from_horizontal(h_fov, cam_w, cam_h)

    gps = obs["gps"]
    imu = obs["imu"]
    camera_z = float(gps["z"]) + CAMERA_Z_OFFSET_M

    floor_dist = max(1e-6, camera_z - FLOOR_HEIGHT_M)
    table_dist = max(1e-6, camera_z - TABLE_HEIGHT_M)

    floor_theory = _theoretical_nadir_rect(floor_dist, h_fov, v_fov)
    table_theory = _theoretical_nadir_rect(table_dist, h_fov, v_fov)
    floor_emp = _empirical_scaled_footprint(floor_dist)
    table_emp = _empirical_scaled_footprint(table_dist)

    print("\n[FOV MEASURE]", flush=True)
    print(
        f"  t={float(obs['time']):.2f}s pose x={float(gps['x']):.3f} y={float(gps['y']):.3f} "
        f"z={float(gps['z']):.3f} yaw={float(imu['yaw']):.3f} desired_z={height_desired:.3f}",
        flush=True,
    )
    print(
        f"  camera: {cam_w}x{cam_h} horizontal_fov={math.degrees(h_fov):.3f}deg "
        f"derived_vertical_fov={math.degrees(v_fov):.3f}deg camera_z={camera_z:.3f}m",
        flush=True,
    )
    print(
        f"  floor plane z={FLOOR_HEIGHT_M:.3f}m distance={floor_dist:.3f}m | "
        f"theoretical_nadir width={floor_theory['width_m']:.3f} depth={floor_theory['height_m']:.3f} | "
        f"empirical_scaled width={floor_emp['width_m']:.3f} forward={floor_emp['forward_m']:.3f} "
        f"backward={floor_emp['backward_m']:.3f} depth={floor_emp['total_depth_m']:.3f}",
        flush=True,
    )
    print(
        f"  table plane z={TABLE_HEIGHT_M:.3f}m distance={table_dist:.3f}m | "
        f"theoretical_nadir width={table_theory['width_m']:.3f} depth={table_theory['height_m']:.3f} | "
        f"empirical_scaled width={table_emp['width_m']:.3f} forward={table_emp['forward_m']:.3f} "
        f"backward={table_emp['backward_m']:.3f} depth={table_emp['total_depth_m']:.3f}",
        flush=True,
    )
    print(
        f"  table/floor distance scale at current altitude: {table_dist / floor_dist:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    devices = CrazyflieDevices(robot, timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    pid_controller = pid_velocity_fixed_height_controller()
    height_desired = float(FLYING_ATTITUDE)
    next_print_time = 0.0

    print("\n====== FoV Measurement Controller ======\n", flush=True)
    print("Manual controls:", flush=True)
    print("  Arrow keys: planar motion", flush=True)
    print("  Q/E: yaw left/right", flush=True)
    print("  W/S: raise/lower target altitude", flush=True)
    print(f"Initial target altitude: {height_desired:.3f} m", flush=True)
    print(f"Print interval: {PRINT_EVERY_S:.1f} s", flush=True)

    while robot.step(timestep) != -1:
        obs = devices.read()
        dt = float(obs["dt"])
        time_now = float(obs["time"])

        forward_desired = 0.0
        sideways_desired = 0.0
        yaw_desired = 0.0
        height_rate = 0.0

        key = keyboard.getKey()
        while key > 0:
            if key == Keyboard.UP:
                forward_desired += PLANAR_SPEED_MPS
            elif key == Keyboard.DOWN:
                forward_desired -= PLANAR_SPEED_MPS
            elif key == Keyboard.RIGHT:
                sideways_desired -= PLANAR_SPEED_MPS
            elif key == Keyboard.LEFT:
                sideways_desired += PLANAR_SPEED_MPS
            elif key == ord("Q"):
                yaw_desired = +YAW_RATE_RAD_S
            elif key == ord("E"):
                yaw_desired = -YAW_RATE_RAD_S
            elif key == ord("W"):
                height_rate = +HEIGHT_STEP_RATE_M_PER_S
            elif key == ord("S"):
                height_rate = -HEIGHT_STEP_RATE_M_PER_S
            key = keyboard.getKey()

        height_desired = max(0.10, height_desired + height_rate * dt)

        if time_now >= next_print_time:
            _print_fov_report(camera=camera, obs=obs, height_desired=height_desired)
            next_print_time = time_now + PRINT_EVERY_S

        motor_power = pid_controller.pid(
            dt,
            forward_desired,
            sideways_desired,
            yaw_desired,
            height_desired,
            obs["imu"]["roll"],
            obs["imu"]["pitch"],
            obs["gyro"]["yaw_rate"],
            obs["gps"]["z"],
            obs["velocity"]["vx"],
            obs["velocity"]["vy"],
        )
        devices.set_motor_speeds(motor_power)
