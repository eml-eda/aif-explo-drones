from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parents[1]
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

from env.env_manager import GridSpec
from policies.aif_zoi_v2 import NAVIGATOR_KWARGS
from policies.planner_time import NavigatorTimeModel, saturated_p_error, saturated_p_time_to
from utils.navigator import Navigator

GRID = GridSpec(min_x=-2.75, max_x=2.75, min_y=-3.25, max_y=3.25, cell_size=0.5, rows=13, cols=11)
PERIOD = 0.625
MODEL = NavigatorTimeModel.from_navigator(NAVIGATOR_KWARGS, PERIOD)
DT = 0.001


def fly(pose, cell, target_yaw, kwargs=NAVIGATOR_KWARGS, t_max=60.0):
    """Integrate the real Navigator with ideal velocity tracking until it reports arrival."""
    nav = Navigator(GRID, **kwargs)
    nav.set_target(cell, target_yaw)
    x, y, yaw = pose["x"], pose["y"], pose["yaw"]
    t = 0.0
    while t < t_max:
        p = {"x": x, "y": y, "yaw": yaw}
        if nav.arrived(p, {cell: 1.0}):
            return t
        cmd = nav.compute_command(p)
        c, s = math.cos(yaw), math.sin(yaw)
        x += (c * cmd["forward"] - s * cmd["sideways"]) * DT
        y += (s * cmd["forward"] + c * cmd["sideways"]) * DT
        yaw += cmd["yaw_rate"] * DT
        t += DT
    raise AssertionError("navigator did not arrive")


def cell_xy(cell):
    return (GRID.min_x + (cell[0] + 0.5) * GRID.cell_size, GRID.min_y + (cell[1] + 0.5) * GRID.cell_size)


class SaturatedControllerTests(unittest.TestCase):
    def test_error_is_linear_then_exponential(self):
        # gain 1.2, limit 0.5: knee at 0.4167 m.
        self.assertAlmostEqual(saturated_p_error(1.0, 1.2, 0.5, 1.0), 0.5)
        t_lin = (1.0 - 0.5 / 1.2) / 0.5
        self.assertAlmostEqual(saturated_p_error(1.0, 1.2, 0.5, t_lin + 1.0), (0.5 / 1.2) * math.exp(-1.2))

    def test_time_to_inverts_the_error(self):
        for e0, tol in ((3.0, 0.2), (0.3, 0.2), (0.5, 0.45), (0.1, 0.2)):
            t = saturated_p_time_to(e0, 1.2, 0.5, tol)
            self.assertAlmostEqual(saturated_p_error(e0, 1.2, 0.5, t), min(abs(e0), tol), places=12)

    def test_no_saturation_case(self):
        self.assertAlmostEqual(saturated_p_time_to(0.3, 1.2, 0.5, 0.2), math.log(0.3 / 0.2) / 1.2)


class NavigatorTimeModelTests(unittest.TestCase):
    def test_matches_the_real_navigator(self):
        cases = [
            ({"x": 0.0, "y": 0.0, "yaw": 0.0}, (5, 6), 0.0),            # stay
            ({"x": 0.0, "y": 0.0, "yaw": 0.0}, (5, 6), math.pi),        # pure half turn
            ({"x": 0.0, "y": 0.0, "yaw": 0.0}, (8, 6), 0.0),            # 1.5 m ahead
            ({"x": 0.0, "y": 0.0, "yaw": 0.0}, (5, 10), 0.0),           # 2 m sideways
            ({"x": 0.0, "y": 0.0, "yaw": 0.0}, (9, 10), math.pi / 4),   # diagonal, with a turn
            ({"x": 0.1, "y": -0.3, "yaw": 2.5}, (1, 2), -2.356),        # turn across +-pi
            ({"x": -1.0, "y": 1.2, "yaw": 0.3}, (5, 6), 1.571),
        ]
        for pose, cell, yaw in cases:
            with self.subTest(pose=pose, cell=cell, yaw=yaw):
                T, t_turn, t_move = MODEL.time_to(pose, cell_xy(cell), yaw)
                self.assertAlmostEqual(T - PERIOD, t_turn + t_move)
                self.assertAlmostEqual(t_turn + t_move, fly(pose, cell, yaw), delta=3 * DT)

    def test_stay_costs_one_detector_period(self):
        T, t_turn, t_move = MODEL.time_to({"x": 0.05, "y": -0.1, "yaw": 0.0}, cell_xy((5, 6)), 0.05)
        self.assertEqual((t_turn, t_move), (0.0, 0.0))
        self.assertEqual(T, PERIOD)

    def test_diagonal_is_set_by_the_slower_axis(self):
        # Each body axis saturates on its own: 2 m diagonally takes about as long as 2 m straight.
        _, _, straight = MODEL.time_to({"x": 0.0, "y": 0.0, "yaw": 0.0}, (2.0, 0.0), 0.0)
        _, _, diagonal = MODEL.time_to({"x": 0.0, "y": 0.0, "yaw": 0.0}, (2.0, 2.0), 0.0)
        self.assertLess(diagonal - straight, 0.5)

    def test_reads_only_the_navigator_settings(self):
        slow = dict(NAVIGATOR_KWARGS, max_yaw_rate=0.3)
        model = NavigatorTimeModel.from_navigator(slow, PERIOD)
        pose, cell = {"x": 0.0, "y": 0.0, "yaw": 0.0}, (5, 6)
        _, t_slow, _ = model.time_to(pose, cell_xy(cell), math.pi)
        _, t_fast, _ = MODEL.time_to(pose, cell_xy(cell), math.pi)
        self.assertGreater(t_slow, t_fast)
        self.assertAlmostEqual(t_slow, fly(pose, cell, math.pi, kwargs=slow), delta=3 * DT)


if __name__ == "__main__":
    unittest.main()
