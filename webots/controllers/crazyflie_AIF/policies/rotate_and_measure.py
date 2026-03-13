# policies/rotate_and_measure.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class _RotateMeasureSM:
    # Phase A: scan 360°, sample every 45°
    n_samples: int = 8
    sample_step_rad: float = math.pi / 4.0  # 45°
    yaw_eps_rad: float = 0.12
    k_yaw: float = 1.2
    max_yaw_rate: float = 0.50

    # Phase B: fly toward best direction up to 2m
    max_forward_speed: float = 0.30
    max_travel_m: float = 2.0
    obstacle_threshold_m: float = 1.0  # safety stop condition

    # internal state
    phase: str = "SCAN"  # "SCAN" or "MOVE"
    scan_start_yaw: float = 0.0
    scan_idx: int = 0
    scan_dists: List[float] = field(default_factory=list)

    move_target_yaw: float = 0.0
    move_start_xy: Optional[Tuple[float, float]] = None

    def reset_scan(self, yaw: float) -> None:
        self.phase = "SCAN"
        self.scan_start_yaw = float(yaw)
        self.scan_idx = 0
        self.scan_dists = [0.0 for _ in range(self.n_samples)]
        self.move_start_xy = None

    def step(
        self,
        *,
        yaw: float,
        front_m: float,
        pose_xy: Optional[Tuple[float, float]],
    ) -> Dict[str, float]:
        if self.phase == "SCAN":
            target = wrap_to_pi(self.scan_start_yaw + self.scan_idx * self.sample_step_rad)
            err = wrap_to_pi(target - yaw)

            if abs(err) > self.yaw_eps_rad:
                yaw_rate = clamp(self.k_yaw * err, -self.max_yaw_rate, self.max_yaw_rate)
                return {"forward": 0.0, "sideways": 0.0, "yaw_rate": yaw_rate}

            # aligned: record measurement and advance
            self.scan_dists[self.scan_idx] = float(front_m)
            self.scan_idx += 1

            if self.scan_idx >= self.n_samples:
                # choose best direction
                best_i = max(range(self.n_samples), key=lambda i: self.scan_dists[i])
                self.move_target_yaw = wrap_to_pi(self.scan_start_yaw + best_i * self.sample_step_rad)
                self.phase = "MOVE"
                self.move_start_xy = pose_xy
            return {"forward": 0.0, "sideways": 0.0, "yaw_rate": 0.0}

        # MOVE phase
        # 1) align yaw
        err = wrap_to_pi(self.move_target_yaw - yaw)
        if abs(err) > self.yaw_eps_rad:
            yaw_rate = clamp(self.k_yaw * err, -self.max_yaw_rate, self.max_yaw_rate)
            return {"forward": 0.0, "sideways": 0.0, "yaw_rate": yaw_rate}

        # 2) move forward until 2m or obstacle
        if front_m < self.obstacle_threshold_m:
            self.reset_scan(yaw)
            return {"forward": 0.0, "sideways": 0.0, "yaw_rate": 0.0}

        if self.move_start_xy is not None and pose_xy is not None:
            dx = pose_xy[0] - self.move_start_xy[0]
            dy = pose_xy[1] - self.move_start_xy[1]
            dist = math.hypot(dx, dy)
            if dist >= self.max_travel_m:
                self.reset_scan(yaw)
                return {"forward": 0.0, "sideways": 0.0, "yaw_rate": 0.0}

        return {"forward": self.max_forward_speed, "sideways": 0.0, "yaw_rate": 0.0}


class RotateAndMeasurePolicy:
    """
    Rotate-and-measure (paper Fig.2-D):
      - phase 1: 360° spin, read front distance every 45°
      - phase 2: fly toward the most obstacle-free direction up to 2m
    """
    def __init__(
        self,
        max_forward_speed: float = 0.30,
        max_yaw_rate: float = 0.50,
        obstacle_threshold_m: float = 1.0,
        max_travel_m: float = 2.0,
        debug: Optional[dict] = None,
    ):
        self.debug = debug or {}
        self.every_n = int(self.debug.get("every_n", 50))
        self.sm = _RotateMeasureSM(
            max_forward_speed=float(max_forward_speed),
            max_yaw_rate=float(max_yaw_rate),
            obstacle_threshold_m=float(obstacle_threshold_m),
            max_travel_m=float(max_travel_m),
        )
        self._initialized = False

    def compute_action(self, observation: dict) -> Dict[str, float]:
        ranges = observation["ranges"]
        yaw = float(observation["imu"]["yaw"])
        step = observation.get("step", None)

        front = float(ranges.get("front", 999.0))
        pose = observation.get("pose", None)
        pose_xy = None
        if pose is not None and "x" in pose and "y" in pose:
            pose_xy = (float(pose["x"]), float(pose["y"]))

        if not self._initialized:
            self.sm.reset_scan(yaw)
            self._initialized = True

        cmd = self.sm.step(yaw=yaw, front_m=front, pose_xy=pose_xy)

        if self.debug.get("policy_grid", False) and step is not None and (step % self.every_n == 0):
            if self.sm.phase == "SCAN":
                print(f"[RM] phase=SCAN idx={self.sm.scan_idx}/{self.sm.n_samples} front={front:.2f}", flush=True)
            else:
                print(f"[RM] phase=MOVE target_yaw={self.sm.move_target_yaw:.3f} front={front:.2f}", flush=True)

        return cmd