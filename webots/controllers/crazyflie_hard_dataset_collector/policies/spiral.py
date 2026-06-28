# policies/spiral.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from policies.wall_following import WallFollowing


@dataclass
class _LapTracker:
    lap_start_xy: Optional[Tuple[float, float]] = None
    lap_start_time: float = 0.0
    armed: bool = False

    def clear(self) -> None:
        self.lap_start_xy = None
        self.lap_start_time = 0.0
        self.armed = False

    def reset(self, xy: Tuple[float, float], t: float) -> None:
        self.lap_start_xy = (float(xy[0]), float(xy[1]))
        self.lap_start_time = float(t)
        self.armed = True

    def lap_completed(
        self,
        xy: Tuple[float, float],
        t: float,
        *,
        min_lap_s: float,
        radius_m: float,
    ) -> bool:
        if (not self.armed) or (self.lap_start_xy is None):
            return False

        if (float(t) - self.lap_start_time) < float(min_lap_s):
            return False

        dx = float(xy[0]) - float(self.lap_start_xy[0])
        dy = float(xy[1]) - float(self.lap_start_xy[1])
        return (dx * dx + dy * dy) <= (radius_m * radius_m)


class SpiralPolicy:
    """
    Spiral exploration:
      - perform perimeter-like laps using wall following
      - first lap at 0.5 m from the wall
      - each completed lap changes the reference distance by 1.0 m
      - maximum reference distance is 2.5 m
      - once max is reached, reverse direction and decrease again

    Important:
      - a lap remains armed across corner states
      - after a reference-distance change, the next lap is NOT armed immediately
      - it is armed only once the drone has settled onto the new wall offset
    """

    def __init__(
        self,
        start_distance_m: float = 0.5,
        step_distance_m: float = 1.0,
        max_distance_m: float = 2.5,
        max_forward_speed: float = 0.50,
        max_turn_rate: float = 0.50,
        min_lap_s: float = 6.0,
        lap_radius_m: float = 0.35,
        arm_tolerance_m: float = 0.20,
        debug: Optional[dict] = None,
    ):
        self.debug = debug or {}
        self.every_n = int(self.debug.get("every_n", 50))

        self.start_distance_m = float(start_distance_m)
        self.step_distance_m = float(step_distance_m)
        self.max_distance_m = float(max_distance_m)

        self.min_lap_s = float(min_lap_s)
        self.lap_radius_m = float(lap_radius_m)
        self.arm_tolerance_m = float(arm_tolerance_m)

        self._direction = +1  # +1 increasing, -1 decreasing
        self._ref = self.start_distance_m

        self._lap = _LapTracker()
        self._was_forward_along_wall = False
        self._pending_rearm = False

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

        self._ref = float(nxt)
        self._wf.reference_distance_from_wall = self._ref

    def compute_action(self, observation: dict) -> Dict[str, float]:
        ranges = observation["ranges"]
        yaw = float(observation["imu"]["yaw"])
        time_now = float(observation["time"])
        step = observation.get("step", None)

        pose = observation.get("pose", None)
        xy: Optional[Tuple[float, float]] = None
        if pose is not None and "x" in pose and "y" in pose:
            xy = (float(pose["x"]), float(pose["y"]))

        direction = WallFollowing.WallFollowingDirection.LEFT
        front_range_value = float(ranges.get("front", 999.0))
        side_range_value = float(ranges.get("right", 999.0))
        left_range_value = float(ranges.get("left", 999.0))

        cmd_vx, cmd_vy, cmd_wz, state_wf = self._wf.wall_follower(
            front_range_value,
            side_range_value,
            yaw,
            direction,
            time_now,
        )

        in_forward_along_wall = (state_wf == WallFollowing.StateWallFollowing.FORWARD_ALONG_WALL)

        if xy is not None:
            # Initial arm: first time we enter steady wall-following
            if (
                in_forward_along_wall
                and (not self._was_forward_along_wall)
                and (not self._lap.armed)
                and (not self._pending_rearm)
            ):
                self._lap.reset(xy, time_now)
                if self.debug.get("policy_grid", False):
                    print(
                        f"[SPIRAL] lap armed at xy=({xy[0]:.2f},{xy[1]:.2f}) "
                        f"t={time_now:.2f} ref={self._ref:.2f}",
                        flush=True,
                    )

            # Rearm after ref change: wait until we are again in FORWARD_ALONG_WALL
            # and actually close to the new desired wall distance.
            if (
                self._pending_rearm
                and in_forward_along_wall
                and abs(side_range_value - self._ref) <= self.arm_tolerance_m
            ):
                self._lap.reset(xy, time_now)
                self._pending_rearm = False
                if self.debug.get("policy_grid", False):
                    print(
                        f"[SPIRAL] lap re-armed at xy=({xy[0]:.2f},{xy[1]:.2f}) "
                        f"t={time_now:.2f} ref={self._ref:.2f} side={side_range_value:.2f}",
                        flush=True,
                    )

            # Completion check for currently armed lap
            if self._lap.armed and self._lap.lap_completed(
                xy,
                time_now,
                min_lap_s=self.min_lap_s,
                radius_m=self.lap_radius_m,
            ):
                old_ref = self._ref
                self._update_reference_distance()

                # Clear old lap and wait to re-arm on the new stable offset path
                self._lap.clear()
                self._pending_rearm = True

                if self.debug.get("policy_grid", False):
                    print(
                        f"[SPIRAL] completed lap at ref={old_ref:.2f} -> new ref={self._ref:.2f} "
                        f"dir={self._direction:+d}",
                        flush=True,
                    )
                    print(
                        f"[SPIRAL] front={front_range_value:.2f} "
                        f"right={side_range_value:.2f} "
                        f"left={left_range_value:.2f} "
                        f"state={state_wf}",
                        flush=True,
                    )

        self._was_forward_along_wall = in_forward_along_wall

        if self.debug.get("policy_grid", False) and step is not None and (step % self.every_n == 0):
            print(
                f"[SPIRAL] ref={self._ref:.2f} "
                f"front={front_range_value:.2f} "
                f"right={side_range_value:.2f} "
                f"left={left_range_value:.2f} "
                f"state={state_wf} "
                f"lap_armed={self._lap.armed} "
                f"pending_rearm={self._pending_rearm}",
                flush=True,
            )

        return {
            "forward": float(cmd_vx),
            "sideways": float(cmd_vy),
            "yaw_rate": float(cmd_wz),
            "state": state_wf,
        }