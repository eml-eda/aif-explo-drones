from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CONTROLLER_DIR = Path(__file__).resolve().parents[1]
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

from configuration.aif_zoi_v2_params import AIFZOIV2Params
from env.env_manager import FoVSpec, GridEnvManager, GridSpec
from policies.aif_zoi_v2 import AIFZOIV2Policy
from policies.zoi_v2_model import (
    CELL_FLOOR,
    CELL_TABLE,
    CELL_UNOBSERVED,
    O_BOTTLE,
    O_CAN,
    O_NOTHING,
    bayes_update,
    expected_information_gain,
    likelihood_matrix,
    object_visibility,
    prior_vector,
    reweight,
    table_probability_given_target,
)

# Most tests hand the table map to the policy through grid["table_cells"], as the
# controller does, so they use the "touched" rule. The centre rule has its own tests.
PARAMS = AIFZOIV2Params(table_map_rule="touched")
CENTRE_PARAMS = AIFZOIV2Params(table_map_rule="centre")
NO_TABLE_PARAMS = AIFZOIV2Params(lambda_floor=6.0 / 143.0, rho=0.0, table_context_enabled=False)
PI_UNOBSERVED = table_probability_given_target(
    CELL_UNOBSERVED, lambda_table=PARAMS.lambda_table, lambda_floor=PARAMS.lambda_floor, rho=PARAMS.rho)

# Controller geometry: 5.5 m x 6.5 m arena, 0.5 m cells -> 11 cols x 13 rows.
GRID_SPEC = {
    "min_x": -2.75, "max_x": 2.75, "min_y": -3.25, "max_y": 3.25,
    "cell_size": 0.5, "rows": 13, "cols": 11,
}
FLOOR_FOV = {"half_left": 0.697, "half_right": 0.697, "forward": 0.553, "backward": 0.493}
# Table-height footprint: smaller than half the cell spacing in every direction,
# so with the drone over a cell centre only that cell's centre is in view.
TABLE_FOV = {"half_left": 0.36, "half_right": 0.36, "forward": 0.29, "backward": 0.26}
IMG_W, IMG_H = 320, 240


def model_A(v: float, params: AIFZOIV2Params = PARAMS) -> np.ndarray:
    return likelihood_matrix(
        v, recall=params.recall, class_accuracy=params.class_accuracy, false_alarm=params.false_alarm)


def model_prior(cell_type: int, params: AIFZOIV2Params = PARAMS) -> np.ndarray:
    return prior_vector(
        cell_type, lambda_table=params.lambda_table, lambda_floor=params.lambda_floor, rho=params.rho)


def mixed_v(v_table: float, v_floor: float) -> float:
    return PI_UNOBSERVED * v_table + (1.0 - PI_UNOBSERVED) * v_floor


def box(cell, class_name, confidence):
    return {"cell": tuple(cell), "class_name": class_name, "confidence": float(confidence)}


def desk_box(bbox_xyxy, confidence=0.9):
    return {"class_name": "desk", "class_id": 2, "confidence": float(confidence),
            "bbox_xyxy": [float(v) for v in bbox_xyxy]}


def make_obs(
    *,
    planes=None,
    events=(),
    table_cells=(),
    processed=True,
    current_cell=(5, 6),
    time_s=0.0,
    with_plane_keys=True,
):
    """planes maps cell -> (v_table, v_floor). Merged coverage follows the controller:
    table-plane coverage for table cells, floor-plane coverage otherwise."""
    planes = dict(planes or {})
    table_cells = set(table_cells)
    merged = {c: (vt if c in table_cells else vf) for c, (vt, vf) in planes.items()}
    visible = {c: (cov >= 0.05) for c, cov in merged.items()}
    cx, cy = current_cell
    grid = {
        "current_cell": current_cell,
        "coverages": merged,
        "visible_mask": visible if processed else {},
        "coverage_threshold": 0.05,
        "target_detections": list(events) if processed else [],
        "detections_all": list(events) if processed else [],
        "perception_processed_this_step": bool(processed),
        "table_cells": sorted(table_cells),
    }
    if with_plane_keys:
        grid["table_plane_coverages"] = {c: vt for c, (vt, _) in planes.items()}
        grid["floor_plane_coverages"] = {c: vf for c, (_, vf) in planes.items()}
    return {
        "step": 1,
        "time": float(time_s),
        "pose": {"x": -2.75 + 0.5 * cx + 0.25, "y": -3.25 + 0.5 * cy + 0.25, "yaw": 0.0},
        "grid_spec": dict(GRID_SPEC),
        "fov_spec": dict(FLOOR_FOV),
        "fov_spec_by_surface": {"floor": dict(FLOOR_FOV), "table": dict(TABLE_FOV)},
        "grid": grid,
        "task": {},
        "camera": {"width": IMG_W, "height": IMG_H},
    }


class AIFZOIV2PolicyBeliefTests(unittest.TestCase):
    def setUp(self):
        self.policy = AIFZOIV2Policy(params=PARAMS)

    def q(self, cell):
        return self.policy.q_s[cell[1], cell[0], :].copy()

    def ctype(self, cell):
        return int(self.policy.cell_type[cell[1], cell[0]])

    def test_initial_belief_is_unobserved_prior(self):
        self.policy.compute_action(make_obs(processed=False))
        np.testing.assert_allclose(self.policy.q_s[0, 0], model_prior(CELL_UNOBSERVED))
        self.assertTrue(np.all(self.policy.cell_type == CELL_UNOBSERVED))

    def test_unobserved_cell_uses_mixed_visibility(self):
        cell = (3, 4)  # visible, but its centre is not seen at table height
        self.policy.compute_action(make_obs(planes={cell: (0.1, 0.8)}))
        self.assertEqual(self.ctype(cell), CELL_UNOBSERVED)
        expected = bayes_update(model_prior(CELL_UNOBSERVED), model_A(mixed_v(0.1, 0.8)), O_NOTHING)
        np.testing.assert_allclose(self.q(cell), expected, atol=1e-12)

    def test_floor_cell_uses_floor_plane_visibility(self):
        cell = (5, 6)  # drone over the cell: centre seen at table height
        self.policy.compute_action(make_obs(planes={cell: (0.9, 1.0)}, current_cell=cell))
        self.assertEqual(self.ctype(cell), CELL_FLOOR)
        expected = bayes_update(model_prior(CELL_FLOOR), model_A(1.0), O_NOTHING)
        np.testing.assert_allclose(self.q(cell), expected, atol=1e-12)

    def test_table_cell_uses_table_plane_visibility(self):
        cell = (3, 4)
        self.policy.compute_action(make_obs(
            planes={cell: (0.3, 0.9)}, table_cells=[cell], events=[box(cell, "target_can", 0.8)]))
        self.assertEqual(self.ctype(cell), CELL_TABLE)
        expected = bayes_update(model_prior(CELL_TABLE), model_A(0.3), O_CAN)
        np.testing.assert_allclose(self.q(cell), expected, atol=1e-12)

    def test_no_table_environment_uses_floor_plane_visibility(self):
        policy = AIFZOIV2Policy(params=NO_TABLE_PARAMS)
        cell = (3, 4)
        policy.compute_action(make_obs(planes={cell: (0.1, 0.7)}))
        expected = bayes_update(model_prior(CELL_FLOOR, NO_TABLE_PARAMS),
                                model_A(0.7, NO_TABLE_PARAMS), O_NOTHING)
        np.testing.assert_allclose(policy.q_s[4, 3], expected, atol=1e-12)
        self.assertTrue(np.all(policy.cell_type == CELL_FLOOR))

    def test_missing_plane_keys_fall_back_to_merged_coverage(self):
        cell = (3, 4)
        self.policy.compute_action(make_obs(planes={cell: (0.1, 0.6)}, with_plane_keys=False))
        expected = bayes_update(model_prior(CELL_UNOBSERVED), model_A(0.6), O_NOTHING)
        np.testing.assert_allclose(self.q(cell), expected, atol=1e-12)

    def test_confidence_does_not_enter_the_likelihood(self):
        cell = (5, 6)
        low = AIFZOIV2Policy(params=PARAMS)
        high = AIFZOIV2Policy(params=PARAMS)
        low.compute_action(make_obs(planes={cell: (0.6, 0.8)}, events=[box(cell, "target_bottle", 0.5)]))
        high.compute_action(make_obs(planes={cell: (0.6, 0.8)}, events=[box(cell, "target_bottle", 0.99)]))
        np.testing.assert_allclose(low.q_s, high.q_s, atol=0.0)
        expected = bayes_update(model_prior(CELL_FLOOR), model_A(0.8), O_BOTTLE)
        np.testing.assert_allclose(low.q_s[6, 5], expected, atol=1e-12)

    def test_several_boxes_in_one_cell_give_one_observation(self):
        cell = (5, 6)
        events = [box(cell, "target_bottle", 0.6), box(cell, "target_can", 0.9)]
        self.policy.compute_action(make_obs(planes={cell: (1.0, 1.0)}, events=events))
        expected = bayes_update(model_prior(CELL_FLOOR), model_A(1.0), O_CAN)
        np.testing.assert_allclose(self.q(cell), expected, atol=1e-12)

    def test_non_visible_cells_do_not_change(self):
        hidden = (8, 10)
        self.policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0), hidden: (0.0, 0.01)}))
        np.testing.assert_allclose(self.q(hidden), model_prior(CELL_UNOBSERVED))
        self.assertEqual(self.ctype(hidden), CELL_UNOBSERVED)

    def test_throttled_step_changes_nothing(self):
        cell = (3, 4)
        self.policy.compute_action(make_obs(planes={cell: (0.2, 1.0)}))
        before_q = self.policy.q_s.copy()
        before_types = self.policy.cell_type.copy()
        # Drone now over the cell, but the frame was not processed: no typing,
        # no update.
        self.policy.compute_action(make_obs(
            planes={cell: (1.0, 1.0)}, events=[box(cell, "target_can", 0.9)],
            processed=False, current_cell=cell, time_s=0.1))
        np.testing.assert_array_equal(self.policy.q_s, before_q)
        np.testing.assert_array_equal(self.policy.cell_type, before_types)

    def test_box_outside_visible_mask_is_dropped(self):
        self.policy.compute_action(make_obs(
            planes={(5, 6): (1.0, 1.0)}, events=[box((9, 1), "target_can", 0.9)]))
        np.testing.assert_allclose(self.q((9, 1)), model_prior(CELL_UNOBSERVED))
        self.assertEqual(self.policy._dropped_target_events, 1)

    def test_floor_to_table_keeps_evidence(self):
        cell = (3, 4)
        # Drone over the cell: typed floor, observed with floor-plane visibility.
        self.policy.compute_action(make_obs(
            planes={cell: (0.9, 1.0)}, events=[box(cell, "target_bottle", 0.7)], current_cell=cell))
        self.assertEqual(self.ctype(cell), CELL_FLOOR)

        # The desk is detected: the cell joins the persistent table map.
        self.policy.compute_action(make_obs(
            planes={cell: (0.4, 0.8)}, table_cells=[cell], time_s=2.0))
        self.assertEqual(self.ctype(cell), CELL_TABLE)

        expected = model_prior(CELL_TABLE)
        expected = bayes_update(expected, model_A(1.0), O_BOTTLE)
        expected = bayes_update(expected, model_A(0.4), O_NOTHING)
        np.testing.assert_allclose(self.q(cell), expected, atol=1e-12)

    def test_table_cell_seen_before_first_frame_is_typed_table(self):
        cell = (2, 2)
        self.policy.compute_action(make_obs(table_cells=[cell], processed=False))
        self.assertEqual(self.ctype(cell), CELL_TABLE)
        np.testing.assert_allclose(self.q(cell), model_prior(CELL_TABLE), atol=1e-12)

    def test_table_cells_never_revert_to_floor(self):
        cell = (2, 2)
        self.policy.compute_action(make_obs(planes={cell: (1.0, 1.0)}, table_cells=[cell], current_cell=cell))
        self.policy.compute_action(make_obs(
            planes={cell: (1.0, 1.0)}, table_cells=[cell], current_cell=cell, time_s=1.0))
        self.assertEqual(self.ctype(cell), CELL_TABLE)

    def test_without_table_context_every_cell_is_floor(self):
        policy = AIFZOIV2Policy(params=NO_TABLE_PARAMS)
        policy.compute_action(make_obs(table_cells=[(2, 2)], processed=False))
        self.assertTrue(np.all(policy.cell_type == CELL_FLOOR))
        expected = np.array([1.0 - 6.0 / 143.0, 3.0 / 143.0, 3.0 / 143.0])
        np.testing.assert_allclose(policy.q_s.reshape(-1, 3), np.tile(expected, (143, 1)), atol=1e-15)

    def test_reweight_is_identity_without_type_change(self):
        q = bayes_update(model_prior(CELL_FLOOR), model_A(1.0), O_BOTTLE)
        np.testing.assert_allclose(reweight(q, model_prior(CELL_FLOOR), model_prior(CELL_FLOOR)), q)


class AIFZOIV2FloorTypingTests(unittest.TestCase):
    def setUp(self):
        self.policy = AIFZOIV2Policy(params=PARAMS)

    def test_cell_with_centre_in_table_footprint_becomes_floor(self):
        self.policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0)}, current_cell=(5, 6)))
        self.assertEqual(int(self.policy.cell_type[6, 5]), CELL_FLOOR)

    def test_visible_cell_with_centre_outside_table_footprint_stays_unobserved(self):
        # Fully covered at floor height, partly at table height, centre not in view.
        neighbour = (6, 6)
        self.policy.compute_action(make_obs(
            planes={(5, 6): (1.0, 1.0), neighbour: (0.2, 1.0)}, current_cell=(5, 6)))
        self.assertEqual(int(self.policy.cell_type[6, 6]), CELL_UNOBSERVED)
        # Only the cell under the drone has its centre in the table footprint.
        self.assertEqual(int(np.sum(self.policy.cell_type == CELL_FLOOR)), 1)

    def test_footprint_test_follows_yaw(self):
        # Forward reach at table height is 0.29 m; a cell 0.5 m ahead is outside.
        pose = {"x": -2.75 + 0.5 * 5 + 0.25, "y": -3.25 + 0.5 * 6 + 0.25, "yaw": 0.0}
        self.policy.compute_action(make_obs(processed=False))
        self.assertTrue(self.policy._centre_in_table_footprint((5, 6), pose))
        self.assertFalse(self.policy._centre_in_table_footprint((6, 6), pose))
        shifted = dict(pose, x=pose["x"] + 0.3)  # 0.2 m behind the next cell centre
        self.assertTrue(self.policy._centre_in_table_footprint((6, 6), shifted))
        turned = dict(shifted, yaw=np.pi)       # now 0.2 m behind the drone (backward 0.26)
        self.assertTrue(self.policy._centre_in_table_footprint((6, 6), turned))
        sideways = dict(pose, y=pose["y"] + 0.3, yaw=np.pi / 2)
        self.assertTrue(self.policy._centre_in_table_footprint((5, 7), sideways))


class AIFZOIV2CentreTableMapTests(unittest.TestCase):
    """Centre rule: a cell is a table cell when its centre, at table height, is in a desk box."""

    def setUp(self):
        self.policy = AIFZOIV2Policy(params=CENTRE_PARAMS)

    def test_table_plane_pixel_inverts_the_controller_projection(self):
        self.policy.compute_action(make_obs(processed=False))
        env = GridEnvManager(GridSpec(**GRID_SPEC), FoVSpec(**TABLE_FOV))
        pose = {"x": 0.1, "y": -0.2, "yaw": 0.7}
        for cell in ((5, 6), (6, 6), (5, 7), (4, 6)):
            u, v = self.policy._table_plane_pixel(cell, pose, IMG_W, IMG_H)
            if not (0.0 <= u <= IMG_W and 0.0 <= v <= IMG_H):
                continue
            rc = env.detection_cell_from_bbox(
                [u - 0.01, v - 0.01, u + 0.01, v + 0.01], IMG_W, IMG_H, pose["x"], pose["y"], pose["yaw"])
            self.assertEqual(rc, (cell[1], cell[0]))

    def test_desk_box_over_a_cell_centre_makes_it_a_table_cell(self):
        # Drone over (5, 6): its centre is at the image centre (u 160, v 126.5).
        self.policy.compute_action(make_obs(
            planes={(5, 6): (1.0, 1.0)}, events=[desk_box([120, 90, 200, 160])]))
        self.assertEqual(int(self.policy.cell_type[6, 5]), CELL_TABLE)
        self.assertEqual(self.policy._table_map, {(5, 6)})

    def test_desk_box_missing_the_centre_does_not_type_table(self):
        self.policy.compute_action(make_obs(
            planes={(5, 6): (1.0, 1.0)}, events=[desk_box([0, 0, 100, 100])]))
        self.assertEqual(self.policy._table_map, set())
        # Centre seen at table height with no desk box on it: floor.
        self.assertEqual(int(self.policy.cell_type[6, 5]), CELL_FLOOR)

    def test_controller_table_map_is_ignored(self):
        self.policy.compute_action(make_obs(
            planes={(5, 6): (1.0, 1.0)}, table_cells=[(5, 6), (2, 2)]))
        self.assertEqual(self.policy._table_map, set())
        self.assertEqual(int(self.policy.cell_type[2, 2]), CELL_UNOBSERVED)

    def test_table_map_persists_and_ignores_throttled_frames(self):
        self.policy.compute_action(make_obs(
            planes={(5, 6): (1.0, 1.0)}, events=[desk_box([120, 90, 200, 160])]))
        self.policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0)}, time_s=1.0))
        self.assertEqual(self.policy._table_map, {(5, 6)})
        policy = AIFZOIV2Policy(params=CENTRE_PARAMS)
        policy.compute_action(make_obs(events=[desk_box([0, 0, IMG_W, IMG_H])], processed=False))
        self.assertEqual(policy._table_map, set())


class AIFZOIV2PolicyPlanningTests(unittest.TestCase):
    def setUp(self):
        self.policy = AIFZOIV2Policy(params=PARAMS)
        self.policy.compute_action(make_obs(
            planes={(5, 6): (1.0, 1.0), (3, 4): (0.3, 0.9), (4, 4): (0.4, 0.9)},
            events=[box((3, 4), "target_bottle", 0.8)],
            table_cells=[(4, 4)]))
        self.table_cells = {(4, 4)}
        # Cell types now cover all three kinds.
        self.assertEqual(int(self.policy.cell_type[6, 5]), CELL_FLOOR)
        self.assertEqual(int(self.policy.cell_type[4, 3]), CELL_UNOBSERVED)
        self.assertEqual(int(self.policy.cell_type[4, 4]), CELL_TABLE)

    def expected_v(self, rc, floor_cov, table_cov):
        r, c = rc
        return object_visibility(
            int(self.policy.cell_type[r, c]),
            float(table_cov.get(rc, 0.0)),
            float(floor_cov.get(rc, 0.0)),
            lambda_table=PARAMS.lambda_table, lambda_floor=PARAMS.lambda_floor, rho=PARAMS.rho)

    def test_information_gain_uses_the_update_likelihood(self):
        for target_cell, yaw_idx in (((3, 4), 0), ((4, 4), 2), ((5, 6), 3), ((0, 0), 7)):
            comps = self.policy._efe_components(
                target_cell=target_cell, target_yaw_index=yaw_idx, table_cells=self.table_cells)
            merged = self.policy._merged_candidate_coverages_rc(
                target_cell=target_cell, target_yaw_index=yaw_idx, table_cells=self.table_cells)
            floor_cov, table_cov = self.policy._cached_candidate_coverages_rc(
                target_cell=target_cell, target_yaw_index=yaw_idx)
            expected = sum(
                expected_information_gain(
                    self.policy.q_s[r, c, :], model_A(self.expected_v((r, c), floor_cov, table_cov)))
                for (r, c), w in merged.items()
                if w >= 0.05
            )
            self.assertAlmostEqual(comps["IG"], expected, places=12)

    def test_center_utility_uses_the_update_likelihood(self):
        for target_cell in ((3, 4), (4, 4)):
            comps = self.policy._efe_components(
                target_cell=target_cell, target_yaw_index=0, table_cells=self.table_cells)
            floor_cov, table_cov = self.policy._cached_candidate_coverages_rc(
                target_cell=target_cell, target_yaw_index=0)
            rc = (target_cell[1], target_cell[0])
            q = self.policy.q_s[rc[0], rc[1], :]
            p_o = model_A(self.expected_v(rc, floor_cov, table_cov)) @ q
            need = min(q[1], q[2])
            self.assertAlmostEqual(
                comps["utility"], 2.0 * need * (p_o[O_BOTTLE] + p_o[O_CAN]), places=12)

    def test_compute_action_returns_a_target(self):
        action = self.policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0)}, time_s=5.0))
        self.assertIn("forward", action)
        self.assertTrue(self.policy.navigator.has_target())



class AIFZOIV2RatePlannerTests(unittest.TestCase):
    """planner_objective="rate": rank by (beta*IG + utility) / T."""

    def make(self, objective, ig_by_candidate, terms="info_and_class", utility_by_candidate=None):
        policy = AIFZOIV2Policy(params=AIFZOIV2Params(
            table_map_rule="touched", planner_objective=objective, planner_terms=terms))
        policy.compute_action(make_obs(processed=False))
        utilities = utility_by_candidate or {}

        def fake(*, target_cell, target_yaw_index, table_cells=None):
            key = (tuple(target_cell), int(target_yaw_index))
            ig, u = ig_by_candidate.get(key, 0.0), utilities.get(key, 0.0)
            return {"risk": -u, "utility": u, "IG": ig, "table_bonus": 0.0, "center_classification_need": 0.0,
                    "center_p_present": 0.0, "center_class_confidence": 0.0}

        policy._efe_components = fake
        return policy

    POSE = {"x": -2.75 + 0.5 * 5 + 0.25, "y": -3.25 + 0.5 * 6 + 0.25, "yaw": 0.0}  # centre of (5, 6)

    def test_near_view_wins_when_it_gives_more_information_per_second(self):
        # Near: next cell ahead, IG 0.5. Far: 3 m away and a half turn, IG 1.2.
        igs = {((6, 6), 0): 0.5, ((5, 0), 4): 1.2}
        dist = self.make("distance_cost", igs)._pick_next_action(current_cell=(5, 6), pose=self.POSE)
        rate = self.make("rate", igs)._pick_next_action(current_cell=(5, 6), pose=self.POSE)
        self.assertEqual(dist[0], (5, 0))
        self.assertEqual(rate[0], (6, 6))

    def test_far_view_still_wins_with_enough_information(self):
        igs = {((6, 6), 0): 0.5, ((5, 0), 4): 20.0}
        rate = self.make("rate", igs)._pick_next_action(current_cell=(5, 6), pose=self.POSE)
        self.assertEqual(rate[0], (5, 0))

    def test_ties_go_to_the_shortest_flight(self):
        # Nothing left to learn anywhere: stay, with the current heading.
        best, _ = self.make("rate", {})._pick_next_action(current_cell=(5, 6), pose=self.POSE, return_scores=True)
        self.assertEqual(best, ((5, 6), 0.0))

    def test_info_only_ignores_the_classification_bonus(self):
        # Next cell ahead: IG 0.5. Cell two ahead: IG 0.2 but a large classification bonus.
        igs = {((6, 6), 0): 0.5, ((7, 6), 0): 0.2}
        bonus = {((7, 6), 0): 3.0}
        with_bonus = self.make("rate", igs, utility_by_candidate=bonus)
        info_only = self.make("rate", igs, terms="info_only", utility_by_candidate=bonus)
        self.assertEqual(with_bonus._pick_next_action(current_cell=(5, 6), pose=self.POSE)[0], (7, 6))
        best, scores = info_only._pick_next_action(current_cell=(5, 6), pose=self.POSE, return_scores=True)
        self.assertEqual(best[0], (6, 6))
        rec = next(r for r in scores if r["cell"] == [7, 6] and r["yaw"] == 0.0)
        self.assertAlmostEqual(rec["rate"], 0.2 / rec["T"])
        self.assertEqual(info_only.parameters()["planning"]["terms"], "info_only")

    def test_info_only_with_distance_cost(self):
        igs = {((6, 6), 0): 0.5, ((7, 6), 0): 0.2}
        policy = self.make("distance_cost", igs, terms="info_only", utility_by_candidate={((7, 6), 0): 3.0})
        best, scores = policy._pick_next_action(current_cell=(5, 6), pose=self.POSE, return_scores=True)
        self.assertEqual(best[0], (6, 6))
        rec = next(r for r in scores if r["cell"] == [7, 6] and r["yaw"] == 0.0)
        self.assertAlmostEqual(rec["G"], -0.2 + 0.04 * 1.0)

    def test_scores_record_the_flight_time(self):
        policy = self.make("rate", {((6, 6), 0): 0.5})
        _, scores = policy._pick_next_action(current_cell=(5, 6), pose=self.POSE, return_scores=True)
        rec = next(r for r in scores if r["cell"] == [6, 6] and r["yaw"] == 0.0)
        T, _, _ = policy.time_model.time_to(self.POSE, policy._cell_center_xy(6, 6), 0.0)
        self.assertAlmostEqual(rec["T"], T)
        self.assertAlmostEqual(rec["rate"], 0.5 / T)
        self.assertEqual(policy.parameters()["planning"]["objective"], "rate")
        self.assertEqual(policy.parameters()["planning"]["time_model"]["detector_period_s"], 0.625)

class AIFZOIV2PolicyDiagnosticsTests(unittest.TestCase):
    def test_parameters_record_model_values(self):
        params = AIFZOIV2Policy(params=AIFZOIV2Params(recall=0.7)).parameters()
        self.assertEqual(params["policy"], "aif_zoi_v2")
        self.assertEqual(params["model"]["recall"], 0.7)
        for key in ("class_accuracy", "false_alarm", "lambda_table", "lambda_floor", "rho"):
            self.assertIn(key, params["model"])
        self.assertIn("visibility", params["observation_model"])
        self.assertIn("floor_typing", params["observation_model"])
        json.dumps(params)

    def test_snapshot_round_trip(self):
        policy = AIFZOIV2Policy(params=PARAMS)
        policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0), (4, 4): (0.5, 1.0)}, table_cells=[(4, 4)]))
        with tempfile.TemporaryDirectory() as tmp:
            path = policy._save_snapshot_npz(
                snap_dir=tmp, step=1, pose={"x": 0.0, "y": 0.0, "yaw": 0.0},
                current_cell=(5, 6), found_cells=set(), table_cells={(4, 4)},
                visible_mask={(5, 6): True}, coverages={(5, 6): 1.0},
                chosen=((5, 6), 0.0), scored_actions=[], reason="test")
            with np.load(path, allow_pickle=True) as data:
                self.assertEqual(data["q_s"].shape, (13, 11, 3))
                self.assertEqual(int(data["cell_type"][4, 4]), CELL_TABLE)
                self.assertEqual(int(data["cell_type"][6, 5]), CELL_FLOOR)
                self.assertAlmostEqual(float(data["prior_present"][4, 4]), PARAMS.lambda_table, places=6)
                meta = json.loads(str(data["meta_json"]))
        self.assertEqual(meta["state_names"], ["empty", "target_bottle", "target_can"])
        self.assertEqual(meta["params"]["model"]["recall"], PARAMS.recall)


if __name__ == "__main__":
    unittest.main()
