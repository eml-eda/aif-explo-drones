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
from policies.aif_zoi_v2 import RHO_THETA_GRID_SIZE, AIFZOIV2Policy
from policies.aif_zoi_v2_vectorized import AIFZOIV2VectorizedPolicy
from policies.zoi_v2_model import (
    O_BOTTLE,
    O_CAN,
    O_NOTHING,
    evidence_coefficients,
    expected_information_gain,
    feasible_grid,
    floor_only_grid,
    grid_weights,
    joint_likelihood,
    joint_obs_index,
    joint_prior_terms,
    log_evidence,
    marginal_beliefs_2d,
    table_probability,
)

PARAMS = AIFZOIV2Params.calibrated("hard", "mobilenet_ssd")
NO_TABLE_PARAMS = AIFZOIV2Params.calibrated("hard", "mobilenet_ssd", table_context_enabled=False)

# Controller geometry: 5.5 m x 6.5 m arena, 0.5 m cells -> 11 cols x 13 rows.
GRID_SPEC = {
    "min_x": -2.75, "max_x": 2.75, "min_y": -3.25, "max_y": 3.25,
    "cell_size": 0.5, "rows": 13, "cols": 11,
}
FLOOR_FOV = {"half_left": 0.697, "half_right": 0.697, "forward": 0.553, "backward": 0.493}
TABLE_FOV = {"half_left": 0.36, "half_right": 0.36, "forward": 0.29, "backward": 0.26}
IMG_W, IMG_H = 320, 240


def L(w_table, w_floor, params=PARAMS):
    return joint_likelihood(
        w_table, w_floor, desk_in_view=params.table_context_enabled and w_table >= 0.05,
        recall=params.recall, class_accuracy=params.class_accuracy, false_alarm=params.false_alarm,
        desk_recall=params.desk_recall, desk_false_alarm=params.desk_false_alarm)


def box(cell, class_name, confidence):
    return {"cell": tuple(cell), "class_name": class_name, "confidence": float(confidence)}


def desk_event(covered_cells, confidence=0.9):
    return {"class_name": "desk", "class_id": 2, "confidence": float(confidence),
            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "covered_cells": [list(c) for c in covered_cells]}


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
    """planes maps cell -> (w_table, w_floor). Merged coverage follows the controller:
    table-plane coverage for its table-map cells, floor-plane coverage otherwise."""
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
        "target_detections": [e for e in events if e["class_name"] != "desk"] if processed else [],
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


def reference_beliefs(ell, params=PARAMS):
    """Full recomputation: exact average over the whole (rho, theta) grid."""
    if params.table_context_enabled:
        terms = joint_prior_terms(p_on_table=params.p_target_on_table)
        rho, theta = feasible_grid(p_on_table=params.p_target_on_table,
                                   n_rho=RHO_THETA_GRID_SIZE, n_theta=RHO_THETA_GRID_SIZE)
    else:
        terms = joint_prior_terms(p_on_table=0.0)
        rho, theta = floor_only_grid(n_theta=RHO_THETA_GRID_SIZE)
    ell = np.asarray(ell).reshape(-1, 6)
    coef = evidence_coefficients(ell, *terms)
    log_ev = log_evidence(coef, rho, theta)
    w = grid_weights(log_ev)
    return marginal_beliefs_2d(ell, coef, terms, rho, theta, w).reshape(13, 11, 6), log_ev, w, rho, theta


def record(observations):
    """ell for cells observed once each: {cell: (w_table, w_floor, desk_covered, target_obs)}."""
    ell = np.ones((13, 11, 6))
    for (cx, cy), (wt, wf, covered, o) in observations.items():
        row = L(wt, wf)[joint_obs_index(covered, o)]
        ell[cy, cx] = row / row.max()
    return ell


class BeliefTests(unittest.TestCase):
    def setUp(self):
        self.policy = AIFZOIV2Policy(params=PARAMS)

    def qj(self, cell):
        return self.policy.q_joint[cell[1], cell[0], :].copy()

    def test_starts_from_the_uniform_prior_over_rho_and_theta(self):
        self.policy.compute_action(make_obs(processed=False))
        q, _, w, rho, theta = reference_beliefs(np.ones((13, 11, 6)))
        np.testing.assert_allclose(self.policy.q_joint, q, atol=1e-12)
        self.assertAlmostEqual(self.policy.rho_estimate()[0], float(rho.mean()))
        self.assertAlmostEqual(self.policy.theta_estimate()[0], float(theta.mean()))
        # An unseen cell's P(target) is the prior mean of theta and P(table) that of rho.
        self.assertAlmostEqual(float(self.policy.q_s[0, 0, 1:].sum()), float(theta.mean()), places=6)
        self.assertAlmostEqual(table_probability(self.qj((0, 0))), float(rho.mean()), places=6)

    def test_beliefs_are_the_exact_average_over_rho_and_theta(self):
        planes = {(5, 6): (1.0, 1.0), (3, 4): (0.3, 0.9), (4, 4): (0.6, 0.9)}
        self.policy.compute_action(make_obs(
            planes=planes, events=[desk_event([(4, 4)]), box((4, 4), "target_bottle", 0.8)]))
        ell = record({(5, 6): (1.0, 1.0, False, O_NOTHING), (3, 4): (0.3, 0.9, False, O_NOTHING),
                      (4, 4): (0.6, 0.9, True, O_BOTTLE)})
        np.testing.assert_allclose(self.policy._ell, ell, atol=1e-15)
        q, log_ev, w, rho, theta = reference_beliefs(ell)
        # The policy skips grid points with weight < 1e-12 of the largest: differences < 1e-8.
        np.testing.assert_allclose(self.policy.q_joint, q, atol=1e-8)
        np.testing.assert_allclose(self.policy.q_s, q[:, :, 0:3] + q[:, :, 3:6], atol=1e-8)
        self.assertAlmostEqual(self.policy.rho_estimate()[0], float(w @ rho), places=12)

    def test_incremental_evidence_matches_a_full_recomputation(self):
        cells = [(5, 6), (6, 6), (7, 6), (4, 6), (5, 7), (5, 5)]
        for i in range(8):
            c = cells[i % len(cells)]
            planes = {c: (1.0, 1.0), (c[0], c[1] + 1): (0.4, 0.9), (4, 4): (0.6, 0.9)}
            events = [desk_event([(4, 4)])] if i % 2 else [box((4, 4), "target_can", 0.7)]
            self.policy.compute_action(make_obs(planes=planes, events=events, current_cell=c, time_s=0.1 * i))
        q, log_ev, _, _, _ = reference_beliefs(self.policy._ell)
        np.testing.assert_allclose(self.policy._log_ev, log_ev, atol=1e-9)
        np.testing.assert_allclose(self.policy.q_joint, q, atol=1e-8)

    def test_desk_cover_raises_table_probability_and_empty_view_lowers_it(self):
        start = table_probability(self._prior_cell())
        self.policy.compute_action(make_obs(
            planes={(3, 4): (0.3, 0.9), (5, 6): (1.0, 1.0)}, events=[desk_event([(3, 4)])]))
        self.assertGreater(table_probability(self.qj((3, 4))), start)
        self.assertLess(table_probability(self.qj((5, 6))), 0.2 * start)

    def _prior_cell(self):
        policy = AIFZOIV2Policy(params=PARAMS)
        policy.compute_action(make_obs(processed=False))
        return policy.q_joint[0, 0]

    def test_cover_below_gate_carries_no_desk_evidence(self):
        self.policy.compute_action(make_obs(planes={(3, 4): (0.02, 0.9)}, events=[desk_event([(3, 4)])]))
        np.testing.assert_allclose(self.policy._ell, record({(3, 4): (0.02, 0.9, False, O_NOTHING)}), atol=1e-15)

    def test_target_box_raises_target_probability(self):
        quiet = AIFZOIV2Policy(params=PARAMS)
        quiet.compute_action(make_obs(planes={(5, 6): (0.6, 0.8)}))
        self.policy.compute_action(make_obs(planes={(5, 6): (0.6, 0.8)}, events=[box((5, 6), "target_can", 0.7)]))
        q, q0 = self.qj((5, 6)), quiet.q_joint[6, 5]
        self.assertGreater(q[2] + q[5], 0.5)
        self.assertLess(q0[2] + q0[5], 0.1)
        self.assertGreater(q[2] + q[5], 10 * (q[1] + q[4]))  # the class follows the box

    def test_confidence_does_not_enter_the_likelihood(self):
        low, high = AIFZOIV2Policy(params=PARAMS), AIFZOIV2Policy(params=PARAMS)
        low.compute_action(make_obs(planes={(5, 6): (0.6, 0.8)}, events=[box((5, 6), "target_bottle", 0.5)]))
        high.compute_action(make_obs(planes={(5, 6): (0.6, 0.8)}, events=[box((5, 6), "target_bottle", 0.99)]))
        np.testing.assert_array_equal(low.q_joint, high.q_joint)

    def test_several_boxes_in_one_cell_give_one_observation(self):
        events = [box((5, 6), "target_bottle", 0.6), box((5, 6), "target_can", 0.9)]
        self.policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0)}, events=events))
        np.testing.assert_allclose(self.policy._ell, record({(5, 6): (1.0, 1.0, False, O_CAN)}), atol=1e-15)

    def test_non_visible_cells_keep_the_shared_unseen_belief(self):
        self.policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0), (8, 10): (0.0, 0.01)}))
        np.testing.assert_array_equal(self.qj((8, 10)), self.qj((0, 0)))
        np.testing.assert_array_equal(self.policy._ell[10, 8], np.ones(6))

    def test_throttled_step_changes_nothing(self):
        self.policy.compute_action(make_obs(planes={(3, 4): (0.2, 1.0)}))
        before = self.policy.q_joint.copy()
        self.policy.compute_action(make_obs(
            planes={(3, 4): (1.0, 1.0)}, events=[desk_event([(3, 4)])], processed=False, time_s=0.1))
        np.testing.assert_array_equal(self.policy.q_joint, before)

    def test_box_outside_visible_mask_is_dropped(self):
        self.policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0)}, events=[box((9, 1), "target_can", 0.9)]))
        np.testing.assert_array_equal(self.policy._ell[1, 9], np.ones(6))
        self.assertEqual(self.policy._dropped_target_events, 1)

    def test_controller_table_map_does_not_enter_the_belief(self):
        planes = {(5, 6): (1.0, 1.0)}
        other = AIFZOIV2Policy(params=PARAMS)
        self.policy.compute_action(make_obs(planes=planes, table_cells=[(3, 4), (2, 2)]))
        other.compute_action(make_obs(planes=planes))
        np.testing.assert_array_equal(self.policy.q_joint, other.q_joint)

    def test_missing_plane_keys_fall_back_to_merged_coverage(self):
        self.policy.compute_action(make_obs(planes={(3, 4): (0.1, 0.6)}, with_plane_keys=False))
        np.testing.assert_allclose(self.policy._ell, record({(3, 4): (0.6, 0.6, False, O_NOTHING)}), atol=1e-15)

    def test_empty_views_lower_theta_and_floor_views_lower_rho(self):
        self.policy.compute_action(make_obs(processed=False))
        rho0, theta0 = self.policy.rho_estimate()[0], self.policy.theta_estimate()[0]
        for i, cell in enumerate([(5, 6), (6, 6), (7, 6), (4, 6), (5, 7), (5, 5)]):
            self.policy.compute_action(make_obs(planes={cell: (1.0, 1.0)}, current_cell=cell, time_s=0.1 * (i + 1)))
        self.assertLess(self.policy.rho_estimate()[0], rho0)
        self.assertLess(self.policy.theta_estimate()[0], theta0)
        diag = self.policy.diagnostics()
        self.assertAlmostEqual(diag["theta_mean"], round(self.policy.theta_estimate()[0], 5))
        for key in ("rho_mean", "rho_sd", "theta_sd"):
            self.assertIn(key, diag)


class NoTableEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.policy = AIFZOIV2Policy(params=NO_TABLE_PARAMS)

    def test_runs_without_desk_values(self):
        params = AIFZOIV2Params(recall=PARAMS.recall, class_accuracy=PARAMS.class_accuracy,
                                false_alarm=PARAMS.false_alarm, table_context_enabled=False)
        loop, vec = AIFZOIV2Policy(params=params), AIFZOIV2VectorizedPolicy(params=params)
        for policy in (loop, vec):
            policy.compute_action(make_obs(
                planes={(3, 4): (0.1, 0.7), (5, 6): (1.0, 1.0)}, events=[desk_event([(3, 4)])]))
        # Same beliefs as with (unused) desk values, and the two planners agree.
        self.policy.compute_action(make_obs(
            planes={(3, 4): (0.1, 0.7), (5, 6): (1.0, 1.0)}, events=[desk_event([(3, 4)])]))
        np.testing.assert_array_equal(loop.q_joint, self.policy.q_joint)
        pose = {"x": 0.1, "y": 0.2, "yaw": 0.3}
        self.assertEqual(loop._pick_next_action(current_cell=(5, 6), pose=pose),
                         vec._pick_next_action(current_cell=(5, 6), pose=pose))

    def test_rho_is_zero_and_theta_is_learned(self):
        self.policy.compute_action(make_obs(processed=False))
        self.assertEqual(self.policy.rho_estimate(), (0.0, 0.0))
        self.assertAlmostEqual(self.policy.theta_estimate()[0], 0.5)
        np.testing.assert_array_equal(self.policy.q_joint[:, :, 0:3], 0.0)

    def test_desk_boxes_carry_no_evidence_and_views_use_the_floor_plane(self):
        self.policy.compute_action(make_obs(
            planes={(3, 4): (0.1, 0.7)}, events=[desk_event([(3, 4)]), box((3, 4), "target_can", 0.8)]))
        row = L(0.1, 0.7, NO_TABLE_PARAMS)[joint_obs_index(False, O_CAN)]
        np.testing.assert_allclose(self.policy._ell[4, 3], row / row.max(), atol=1e-15)
        q, _, _, _, _ = reference_beliefs(self.policy._ell, NO_TABLE_PARAMS)
        np.testing.assert_allclose(self.policy.q_joint, q, atol=1e-8)
        np.testing.assert_array_equal(self.policy.table_probability_map(), 0.0)


class PlannerTests(unittest.TestCase):
    POSE = {"x": -2.75 + 0.5 * 5 + 0.25, "y": -3.25 + 0.5 * 6 + 0.25, "yaw": 0.0}  # centre of (5, 6)

    def setUp(self):
        self.policy = AIFZOIV2Policy(params=PARAMS)
        self.policy.compute_action(make_obs(
            planes={(5, 6): (1.0, 1.0), (3, 4): (0.3, 0.9), (4, 4): (0.4, 0.9)},
            events=[box((3, 4), "target_bottle", 0.8), desk_event([(4, 4)])],
            table_cells=[(4, 4)]))
        self.table_cells = {(4, 4)}

    def with_fake_information(self, ig_by_candidate):
        policy = AIFZOIV2Policy(params=PARAMS)
        policy.compute_action(make_obs(processed=False))

        def fake(*, target_cell, target_yaw_index, table_cells=None):
            return ig_by_candidate.get((tuple(target_cell), int(target_yaw_index)), 0.0)

        policy._candidate_information_gain = fake
        return policy

    def test_information_gain_uses_the_update_likelihood(self):
        for target_cell, yaw_idx in (((3, 4), 0), ((4, 4), 2), ((5, 6), 3), ((0, 0), 7)):
            ig = self.policy._candidate_information_gain(
                target_cell=target_cell, target_yaw_index=yaw_idx, table_cells=self.table_cells)
            merged = self.policy._merged_candidate_coverages_rc(
                target_cell=target_cell, target_yaw_index=yaw_idx, table_cells=self.table_cells)
            floor_cov, table_cov = self.policy._cached_candidate_coverages_rc(
                target_cell=target_cell, target_yaw_index=yaw_idx)
            expected = sum(
                expected_information_gain(
                    self.policy.q_joint[r, c, :], L(table_cov.get((r, c), 0.0), floor_cov.get((r, c), 0.0)))
                for (r, c), w in merged.items() if w >= 0.05)
            self.assertAlmostEqual(ig, expected, places=12)

    def test_near_view_wins_when_it_gives_more_information_per_second(self):
        # Near: next cell ahead, IG 0.5. Far: 3 m away and a half turn, IG 1.2.
        policy = self.with_fake_information({((6, 6), 0): 0.5, ((5, 0), 4): 1.2})
        self.assertEqual(policy._pick_next_action(current_cell=(5, 6), pose=self.POSE)[0], (6, 6))

    def test_far_view_still_wins_with_enough_information(self):
        policy = self.with_fake_information({((6, 6), 0): 0.5, ((5, 0), 4): 20.0})
        self.assertEqual(policy._pick_next_action(current_cell=(5, 6), pose=self.POSE)[0], (5, 0))

    def test_ties_go_to_the_shortest_flight(self):
        # Nothing left to learn anywhere: stay, with the current heading.
        best = self.with_fake_information({})._pick_next_action(current_cell=(5, 6), pose=self.POSE)
        self.assertEqual(best, ((5, 6), 0.0))

    def test_scores_record_the_flight_time(self):
        policy = self.with_fake_information({((6, 6), 0): 0.5})
        _, scores = policy._pick_next_action(current_cell=(5, 6), pose=self.POSE, return_scores=True)
        rec = next(r for r in scores if r["cell"] == [6, 6] and r["yaw"] == 0.0)
        T, _, _ = policy.time_model.time_to(self.POSE, policy._cell_center_xy(6, 6), 0.0)
        self.assertAlmostEqual(rec["T"], T)
        self.assertAlmostEqual(rec["rate"], 0.5 / T)
        self.assertAlmostEqual(rec["G"], -0.5 / T)
        self.assertEqual(len(scores), 143 * 8)

    def test_compute_action_returns_a_target(self):
        action = self.policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0)}, time_s=5.0))
        self.assertIn("forward", action)
        self.assertTrue(self.policy.navigator.has_target())


class VectorizedPlannerTests(unittest.TestCase):
    """The vectorized planner computes the same numbers as the loop planner."""

    def fly_both(self, params=PARAMS, n_frames=12, seed=3):
        rng = np.random.default_rng(seed)
        loop, vec = AIFZOIV2Policy(params=params), AIFZOIV2VectorizedPolicy(params=params)
        for i in range(n_frames):
            c = (int(rng.integers(11)), int(rng.integers(13)))
            planes = {(c[0] + dx, c[1] + dy): (float(rng.random()), float(rng.random()))
                      for dx in (-1, 0, 1) for dy in (-1, 0, 1) if 0 <= c[0] + dx < 11 and 0 <= c[1] + dy < 13}
            events = [desk_event([c])] if i % 3 == 0 else []
            if i % 4 == 1:
                events.append(box(c, "target_can", 0.8))
            obs = make_obs(planes=planes, events=events, current_cell=c, table_cells=[c], time_s=0.1 * i)
            loop.compute_action(obs)
            vec.compute_action(obs)
        return loop, vec

    def test_information_gains_flight_times_and_decisions_are_identical(self):
        for params in (PARAMS, AIFZOIV2Params.calibrated("hard", "yolo_nano"), NO_TABLE_PARAMS):
            loop, vec = self.fly_both(params)
            np.testing.assert_array_equal(loop.q_joint, vec.q_joint)
            for pose, table_cells in (({"x": 0.1, "y": 0.2, "yaw": 0.3}, {(4, 4), (5, 5)}),
                                      ({"x": -2.25, "y": 1.75, "yaw": -2.0}, set()), (None, {(0, 0)})):
                best_l, sl = loop._pick_next_action(current_cell=(5, 6), table_cells=table_cells, pose=pose,
                                                    return_scores=True)
                best_v, sv = vec._pick_next_action(current_cell=(5, 6), table_cells=table_cells, pose=pose,
                                                   return_scores=True)
                self.assertEqual(best_l, best_v)
                for key in ("cell", "yaw", "IG", "T", "T_turn", "T_move", "rate", "G"):
                    self.assertEqual([r[key] for r in sl], [r[key] for r in sv], key)

    def test_ties_go_to_the_shortest_flight_then_the_first_candidate(self):
        policy = AIFZOIV2VectorizedPolicy(params=PARAMS)
        policy.compute_action(make_obs(processed=False))
        policy._candidate_information_gains = lambda table_cells: np.zeros(len(policy._cand_cells))
        pose = {"x": -2.75 + 0.5 * 5 + 0.25, "y": -3.25 + 0.5 * 6 + 0.25, "yaw": 0.0}
        self.assertEqual(policy._pick_next_action(current_cell=(5, 6), pose=pose), ((5, 6), 0.0))

    def test_parameters_and_controller_selection(self):
        params = AIFZOIV2Params.calibrated(planner_impl="vectorized")
        self.assertEqual(AIFZOIV2VectorizedPolicy(params=params).parameters()["planning"]["implementation"],
                         "vectorized")
        self.assertEqual(AIFZOIV2Policy(params=PARAMS).parameters()["planning"]["implementation"], "loop")

    def test_replans_and_planner_time_are_logged(self):
        policy = AIFZOIV2VectorizedPolicy(params=PARAMS)
        action = policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0)}, time_s=5.0))
        diag = policy.diagnostics()
        self.assertEqual(diag["replans"], 1)
        self.assertGreater(diag["planner_ms_last"], 0.0)
        self.assertAlmostEqual(action["target_selection"]["planner_ms"], policy._planner_ms_last)


class DiagnosticsTests(unittest.TestCase):
    def test_parameters_record_the_model(self):
        policy = AIFZOIV2Policy(params=AIFZOIV2Params.calibrated(recall=0.7))
        params = policy.parameters()
        self.assertEqual(params["policy"], "aif_zoi_v2")
        self.assertEqual(params["model"]["recall"], 0.7)
        self.assertEqual(params["model"]["calibration"], "hard/mobilenet_ssd")
        self.assertEqual(params["observation_model"]["rho_theta_grid_size"], 32)
        self.assertEqual(params["observation_model"]["rho_theta_grid_points"], 513)  # before the first frame
        self.assertEqual(params["planning"]["time_model"]["detector_period_s"], 0.625)
        json.dumps(params)

    def test_grid_size_sets_the_rho_theta_grid(self):
        policy = AIFZOIV2Policy(params=AIFZOIV2Params.calibrated(grid_size=64))
        policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0)}))
        rho, _ = feasible_grid(p_on_table=0.9, n_rho=64, n_theta=64)
        self.assertEqual(policy._rho_grid.size, rho.size)
        self.assertEqual(policy.parameters()["observation_model"]["rho_theta_grid_points"], rho.size)

    def test_diagnostics_are_empty_before_the_first_frame(self):
        self.assertEqual(AIFZOIV2Policy(params=PARAMS).diagnostics(), {})

    def test_snapshot_round_trip(self):
        policy = AIFZOIV2Policy(params=PARAMS)
        policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0), (4, 4): (0.5, 1.0)},
                                       events=[desk_event([(4, 4)])], table_cells=[(4, 4)]))
        with tempfile.TemporaryDirectory() as tmp:
            path = policy._save_snapshot_npz(
                snap_dir=tmp, step=1, pose={"x": 0.0, "y": 0.0, "yaw": 0.0},
                current_cell=(5, 6), found_cells=set(), table_cells={(4, 4)},
                visible_mask={(5, 6): True}, coverages={(5, 6): 1.0},
                chosen=((5, 6), 0.0), scored_actions=[], reason="test")
            with np.load(path, allow_pickle=True) as data:
                self.assertEqual(data["q_s"].shape, (13, 11, 3))
                self.assertEqual(data["q_joint"].shape, (13, 11, 6))
                self.assertAlmostEqual(float(data["p_table"][4, 4]),
                                       float(policy.table_probability_map()[4, 4]), places=6)
                self.assertEqual(int(data["table_map"][4, 4]), 1)
                meta = json.loads(str(data["meta_json"]))
        self.assertEqual(meta["state_names"], ["empty", "target_bottle", "target_can"])
        self.assertEqual(meta["params"]["model"]["recall"], PARAMS.recall)
        self.assertIn("theta_mean", meta["rho_theta"])


if __name__ == "__main__":
    unittest.main()
