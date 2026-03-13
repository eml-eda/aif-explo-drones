# policies/pseudo_random.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class _PseudoRandomSM:
    # Paper: obstacle if ToF < 1m; turn random angle always > ±90°
    obstacle_threshold_m: float = 1.0
    max_forward_speed: float = 0.30
    max_yaw_rate: float = 0.50

    yaw_eps_rad: float = 0.12
    k_yaw: float = 1.2

    # internal
    has_turn_target: bool = False
    target_yaw: float = 0.0

    def step(self, *, front_m: float, yaw: float) -> Dict[str, float]:
        # If currently turning, keep turning until yaw aligned
        if self.has_turn_target:
            err = wrap_to_pi(self.target_yaw - yaw)
            if abs(err) <= self.yaw_eps_rad:
                self.has_turn_target = False
                return {"forward": self.max_forward_speed, "sideways": 0.0, "yaw_rate": 0.0}
            yaw_rate = clamp(self.k_yaw * err, -self.max_yaw_rate, self.max_yaw_rate)
            return {"forward": 0.0, "sideways": 0.0, "yaw_rate": yaw_rate}

        # Not turning: go straight unless obstacle
        if front_m < self.obstacle_threshold_m:
            # Random turn magnitude in (90°, 180°], random sign
            mag = random.uniform(math.pi / 2.0, math.pi)
            sign = random.choice([-1.0, +1.0])
            self.target_yaw = wrap_to_pi(yaw + sign * mag)
            self.has_turn_target = True
            # start turning immediately
            err = wrap_to_pi(self.target_yaw - yaw)
            yaw_rate = clamp(self.k_yaw * err, -self.max_yaw_rate, self.max_yaw_rate)
            return {"forward": 0.0, "sideways": 0.0, "yaw_rate": yaw_rate}

        return {"forward": self.max_forward_speed, "sideways": 0.0, "yaw_rate": 0.0}


class PseudoRandomPolicy:
    """
    Pseudo-random (paper Fig.2-A):
      - fly straight
      - if front ToF < 1m: turn by a random angle whose magnitude is > 90°
    """
    def __init__(
        self,
        obstacle_threshold_m: float = 1.0,
        max_forward_speed: float = 0.30,
        max_yaw_rate: float = 0.50,
        seed: Optional[int] = None,
        debug: Optional[dict] = None,
    ):
        if seed is not None:
            random.seed(int(seed))
        self.debug = debug or {}
        self.every_n = int(self.debug.get("every_n", 50))
        self.sm = _PseudoRandomSM(
            obstacle_threshold_m=float(obstacle_threshold_m),
            max_forward_speed=float(max_forward_speed),
            max_yaw_rate=float(max_yaw_rate),
        )

    def compute_action(self, observation: dict) -> Dict[str, float]:
        ranges = observation["ranges"]
        yaw = float(observation["imu"]["yaw"])
        step = observation.get("step", None)

        front = float(ranges.get("front", 999.0))
        cmd = self.sm.step(front_m=front, yaw=yaw)

        if self.debug.get("policy_grid", False) and step is not None and (step % self.every_n == 0):
            print(f"[PR] front={front:.2f} turning={self.sm.has_turn_target} target_yaw={self.sm.target_yaw:.3f}", flush=True)

        return cmd