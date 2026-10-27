# policies/planner_time.py
"""Flight time of a planner move, derived from the navigator's control law.

The navigator (utils/navigator.py) turns first, then translates:

    turn:  e_yaw' = -clamp(k_yaw * e_yaw, +-max_yaw_rate)   until |e_yaw| <= yaw_eps
    move:  e_x'   = -clamp(k_pos * e_x, +-max_forward)       (body frame, each axis
           e_y'   = -clamp(k_pos * e_y, +-max_sideways)       clamped independently)
                                                              until hypot(e_x, e_y) <= dist_eps

Each equation is a proportional controller saturated at a limit, whose solution
is linear at the limit while |e| > limit / gain and exponential after. The time
to a candidate is the exact arrival time of these equations, plus one detector
period for the observation the planner is valuing. Every number comes from the
navigator settings and the detector rate; the low-level velocity tracking is
assumed ideal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np

from utils.navigator import wrap_to_pi


def saturated_p_error(e0: float, gain: float, limit: float, t: float) -> float:
    """|e(t)| for e' = -clamp(gain * e, +-limit) from |e(0)| = |e0|."""
    a = abs(float(e0))
    if t <= 0.0 or a == 0.0:
        return a
    knee = limit / gain
    if a > knee:
        t_linear = (a - knee) / limit
        if t <= t_linear:
            return a - limit * t
        return knee * math.exp(-gain * (t - t_linear))
    return a * math.exp(-gain * t)


def saturated_p_time_to(e0: float, gain: float, limit: float, tol: float) -> float:
    """First time with |e| <= tol for e' = -clamp(gain * e, +-limit)."""
    a = abs(float(e0))
    if a <= tol:
        return 0.0
    knee = limit / gain
    if a > knee:
        if tol >= knee:
            return (a - tol) / limit
        return (a - knee) / limit + math.log(knee / tol) / gain
    return math.log(a / tol) / gain


# Vectorized versions (same formulas, element by element) for the vectorized planner.


def saturated_p_error_batch(e0: np.ndarray, gain: float, limit: float, t: np.ndarray) -> np.ndarray:
    """saturated_p_error for arrays of e0 and t."""
    a = np.abs(np.asarray(e0, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64)
    knee = limit / gain
    t_linear = (a - knee) / limit
    linear = a - limit * t
    with np.errstate(over="ignore"):
        after_knee = knee * np.exp(-gain * (t - t_linear))
        below_knee = a * np.exp(-gain * t)
    out = np.where(a > knee, np.where(t <= t_linear, linear, after_knee), below_knee)
    return np.where((t <= 0.0) | (a == 0.0), a, out)


def saturated_p_time_to_batch(e0: np.ndarray, gain: float, limit: float, tol: float) -> np.ndarray:
    """saturated_p_time_to for an array of e0."""
    a = np.abs(np.asarray(e0, dtype=np.float64))
    knee = limit / gain
    with np.errstate(divide="ignore", invalid="ignore"):
        above = ((a - tol) / limit if tol >= knee
                 else (a - knee) / limit + math.log(knee / tol) / gain)
        below = np.log(a / tol) / gain
    return np.where(a <= tol, 0.0, np.where(a > knee, above, below))


def hypot_le_batch(x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """hypot(x, y) <= eps as math.hypot decides it. np.hypot can round differently in
    the last digit, so values within a few ulps of eps are re-checked with math.hypot."""
    h = np.hypot(x, y)
    out = h <= eps
    close = np.flatnonzero(np.abs(h - eps) <= 8.0 * np.spacing(eps))
    for i in close:
        out[i] = math.hypot(float(x[i]), float(y[i])) <= eps
    return out


@dataclass(frozen=True)
class NavigatorTimeModel:
    max_forward: float
    max_sideways: float
    max_yaw_rate: float
    k_pos: float
    k_yaw: float
    dist_eps_m: float
    yaw_eps_rad: float
    detector_period_s: float

    @classmethod
    def from_navigator(cls, navigator_kwargs: Mapping[str, float], detector_period_s: float) -> "NavigatorTimeModel":
        return cls(
            max_forward=float(navigator_kwargs["max_forward"]),
            max_sideways=float(navigator_kwargs["max_sideways"]),
            max_yaw_rate=float(navigator_kwargs["max_yaw_rate"]),
            k_pos=float(navigator_kwargs["k_pos"]),
            k_yaw=float(navigator_kwargs["k_yaw"]),
            dist_eps_m=float(navigator_kwargs["dist_eps_m"]),
            yaw_eps_rad=float(navigator_kwargs["yaw_eps_rad"]),
            detector_period_s=float(detector_period_s),
        )

    def turn_time(self, yaw: float, target_yaw: float) -> Tuple[float, float]:
        """(time, heading at the end of the turn)."""
        err = wrap_to_pi(float(target_yaw) - float(yaw))
        if abs(err) <= self.yaw_eps_rad:
            return 0.0, float(yaw)
        t = saturated_p_time_to(err, self.k_yaw, self.max_yaw_rate, self.yaw_eps_rad)
        # The turn stops as soon as the error enters the tolerance.
        return t, float(target_yaw) - math.copysign(self.yaw_eps_rad, err)

    def move_time(self, dx_body: float, dy_body: float) -> float:
        """Time until hypot(e_x, e_y) <= dist_eps, each body axis saturated independently."""
        eps = self.dist_eps_m
        if math.hypot(dx_body, dy_body) <= eps:
            return 0.0

        def dist(t: float) -> float:
            return math.hypot(
                saturated_p_error(dx_body, self.k_pos, self.max_forward, t),
                saturated_p_error(dy_body, self.k_pos, self.max_sideways, t))

        # Both |e| decrease monotonically, so the distance does too. Each axis
        # within eps / sqrt(2) bounds the arrival time from above.
        lo = 0.0
        hi = max(
            saturated_p_time_to(dx_body, self.k_pos, self.max_forward, eps / math.sqrt(2.0)),
            saturated_p_time_to(dy_body, self.k_pos, self.max_sideways, eps / math.sqrt(2.0)))
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if dist(mid) <= eps:
                hi = mid
            else:
                lo = mid
        return hi

    def time_to(self, pose: Mapping[str, float], target_xy: Tuple[float, float], target_yaw: float) -> Tuple[float, float, float]:
        """(T, T_turn, T_move): navigator arrival time plus one detector period."""
        t_turn, heading = self.turn_time(float(pose["yaw"]), target_yaw)
        dx = float(target_xy[0]) - float(pose["x"])
        dy = float(target_xy[1]) - float(pose["y"])
        c, s = math.cos(heading), math.sin(heading)
        t_move = self.move_time(c * dx + s * dy, -s * dx + c * dy)
        return t_turn + t_move + self.detector_period_s, t_turn, t_move

    def time_to_batch(
        self, pose: Mapping[str, float], target_x: np.ndarray, target_y: np.ndarray, target_yaw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """time_to for N targets at once: (T, T_turn, T_move), each (N,)."""
        yaw = float(pose["yaw"])
        target_yaw = np.asarray(target_yaw, dtype=np.float64)
        # Turn (turn_time): wrap_to_pi, then the saturated P time to the tolerance.
        err = np.remainder(target_yaw - yaw + math.pi, 2.0 * math.pi) - math.pi
        turning = np.abs(err) > self.yaw_eps_rad
        t_turn = np.where(turning, saturated_p_time_to_batch(err, self.k_yaw, self.max_yaw_rate, self.yaw_eps_rad), 0.0)
        heading = np.where(turning, target_yaw - np.copysign(self.yaw_eps_rad, err), yaw)
        # Move (move_time) in the body frame at the end of the turn.
        dx = np.asarray(target_x, dtype=np.float64) - float(pose["x"])
        dy = np.asarray(target_y, dtype=np.float64) - float(pose["y"])
        c, s = np.cos(heading), np.sin(heading)
        ex, ey = c * dx + s * dy, -s * dx + c * dy
        eps = self.dist_eps_m
        moving = ~hypot_le_batch(ex, ey, eps)
        lo = np.zeros_like(ex)
        hi = np.maximum(
            saturated_p_time_to_batch(ex, self.k_pos, self.max_forward, eps / math.sqrt(2.0)),
            saturated_p_time_to_batch(ey, self.k_pos, self.max_sideways, eps / math.sqrt(2.0)))
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            arrived = hypot_le_batch(saturated_p_error_batch(ex, self.k_pos, self.max_forward, mid),
                                     saturated_p_error_batch(ey, self.k_pos, self.max_sideways, mid), eps)
            hi = np.where(arrived, mid, hi)
            lo = np.where(arrived, lo, mid)
        t_move = np.where(moving, hi, 0.0)
        return t_turn + t_move + self.detector_period_s, t_turn, t_move

    def as_dict(self) -> dict:
        return {
            "max_forward": self.max_forward,
            "max_sideways": self.max_sideways,
            "max_yaw_rate": self.max_yaw_rate,
            "k_pos": self.k_pos,
            "k_yaw": self.k_yaw,
            "dist_eps_m": self.dist_eps_m,
            "yaw_eps_rad": self.yaw_eps_rad,
            "detector_period_s": self.detector_period_s,
        }
