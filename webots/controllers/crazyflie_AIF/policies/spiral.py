# policies/spiral.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from policies.wall_following import WallFollowing


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class _LapTracker:
    lap_start_xy: Optional[Tuple[float, float]] = None
    lap_start_time: float = 0.0

    def reset(self, xy: Tuple[float, float], t: float) -> None:
        self.lap_start_xy = xy
        self.lap_start_time = float(t)

    def lap_completed(self, xy: Tuple[float, float], t: float, *, min_lap_s: float, radius_m: float) -> bool:
        if self.lap_start_xy is None:
            self.reset(xy, t)
            return False
        if (float(t) - self.lap_start_time) < float(min_lap_s):
            return False
        dx = xy[0] - self.lap_start_xy[0]
        dy = xy[1] - self.lap_start_xy[1]
        return (dx * dx + dy * dy) <= (radius_m * radius_m)


class SpiralPolicy:
    """
    Spiral (paper Fig.2-C):
      - perform perimeter-like laps (wall following) with increasing distance to wall
      - when max distance reached, reverse and decrease distance
      - loop
    Notes:
      - We approximate "lap complete" by: return near lap_start after min_lap_s.
    """
    def __init__(
        self,
        start_distance_m: float = 0.5,
        step_distance_m: float = 0.5,
        max_distance_m: float = 2.0,
        max_forward_speed: float = 0.30,
        max_turn_rate: float = 0.50,
        min_lap_s: float = 18.0,
        lap_radius_m: float = 0.35,
        debug: Optional[dict] = None,
    ):
        self.debug = debug or {}
        self.every_n = int(self.debug.get("every_n", 50))

        self.start_distance_m = float(start_distance_m)
        self.step_distance_m = float(step_distance_m)
        self.max_distance_m = float(max_distance_m)

        self.min_lap_s = float(min_lap_s)
        self.lap_radius_m = float(lap_radius_m)

        self._direction = +1  # +1 increasing, -1 decreasing
        self._ref = self.start_distance_m

        self._lap = _LapTracker()

        self._wf = WallFollowing(
            angle_value_buffer=0.01,
            reference_distance_from_wall=self._ref,
            max_forward_speed=float(max_forward_speed),
            max_turn_rate=float(max_turn_rate),
            init_state=WallFollowing.StateWallFollowing.FORWARD,
        )

    def _update_reference_distance(self) -> None:
        nxt = self._ref + self._direction * self.step_distance_m
        if nxt > self.max_distance_m:
            self._direction = -1
            nxt = self.max_distance_m
        elif nxt < self.start_distance_m:
            self._direction = +1
            nxt = self.start_distance_m
        self._ref = nxt
        self._wf.reference_distance_from_wall = self._ref

    def compute_action(self, observation: dict) -> Dict[str, float]:
        ranges = observation["ranges"]
        yaw = float(observation["imu"]["yaw"])
        time_now = float(observation["time"])
        step = observation.get("step", None)

        pose = observation.get("pose", None)
        xy = None
        if pose is not None and "x" in pose and "y" in pose:
            xy = (float(pose["x"]), float(pose["y"]))

        # lap completion check (requires pose)
        if xy is not None:
            if self._lap.lap_start_xy is None:
                self._lap.reset(xy, time_now)
            elif self._lap.lap_completed(xy, time_now, min_lap_s=self.min_lap_s, radius_m=self.lap_radius_m):
                self._update_reference_distance()
                self._lap.reset(xy, time_now)
                if self.debug.get("policy_grid", False):
                    print(f"[SPIRAL] new reference_distance_from_wall={self._ref:.2f} dir={self._direction:+d}", flush=True)

        direction = WallFollowing.WallFollowingDirection.LEFT
        side_range_value = float(ranges.get("right", 999.0))

        cmd_vx, cmd_vy, cmd_wz, state_wf = self._wf.wall_follower(
            float(ranges.get("front", 999.0)),
            side_range_value,
            yaw,
            direction,
            time_now,
        )

        if self.debug.get("policy_grid", False) and step is not None and (step % self.every_n == 0):
            print(f"[SPIRAL] ref={self._ref:.2f} state={state_wf}", flush=True)

        return {
            "forward": float(cmd_vx),
            "sideways": float(cmd_vy),
            "yaw_rate": float(cmd_wz),
            "state": state_wf,
        }