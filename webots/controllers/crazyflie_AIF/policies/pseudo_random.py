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
    rng: random.Random

    # Paper: obstacle if ToF < 1m; turn random angle always > ±90°
    obstacle_threshold_m: float = 1.0
    max_forward_speed: float = 0.50
    max_yaw_rate: float = 0.50

    yaw_eps_rad: float = 0.12
    k_yaw: float = 1.2

    # internal
    has_turn_target: bool = False
    target_yaw: float = 0.0

    def step(self, *, front_m: float, yaw: float) -> Dict[str, float]:
        if self.has_turn_target:
            err = wrap_to_pi(self.target_yaw - yaw)
            if abs(err) <= self.yaw_eps_rad:
                self.has_turn_target = False
                return {"forward": self.max_forward_speed, "sideways": 0.0, "yaw_rate": 0.0}
            yaw_rate = clamp(self.k_yaw * err, -self.max_yaw_rate, self.max_yaw_rate)
            return {"forward": 0.0, "sideways": 0.0, "yaw_rate": yaw_rate}

        if front_m < self.obstacle_threshold_m:
            mag = self.rng.uniform(math.pi / 2.0, math.pi)
            sign = self.rng.choice([-1.0, +1.0])
            self.target_yaw = wrap_to_pi(yaw + sign * mag)
            self.has_turn_target = True
            err = wrap_to_pi(self.target_yaw - yaw)
            yaw_rate = clamp(self.k_yaw * err, -self.max_yaw_rate, self.max_yaw_rate)
            return {"forward": 0.0, "sideways": 0.0, "yaw_rate": yaw_rate}

        return {"forward": self.max_forward_speed, "sideways": 0.0, "yaw_rate": 0.0}


class PseudoRandomPolicy:
    """
    Pseudo-random baseline:
      - fly straight
      - if front ToF < 1 m, turn by a random angle whose magnitude is > 90°

    The policy owns a private RNG so an episode seed reproduces the same random
    turn sequence independently of random-number use elsewhere in the controller.
    """

    def __init__(
        self,
        obstacle_threshold_m: float = 1.0,
        max_forward_speed: float = 0.30,
        max_yaw_rate: float = 0.50,
        seed: Optional[int] = None,
        debug: Optional[dict] = None,
    ):
        self.seed = None if seed is None else int(seed)
        self.rng = random.Random(self.seed)
        self.debug = debug or {}
        self.every_n = int(self.debug.get("every_n", 50))
        self.sm = _PseudoRandomSM(
            rng=self.rng,
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
            print(
                f"[PR] front={front:.2f} turning={self.sm.has_turn_target} "
                f"target_yaw={self.sm.target_yaw:.3f}",
                flush=True,
            )

        return cmd
