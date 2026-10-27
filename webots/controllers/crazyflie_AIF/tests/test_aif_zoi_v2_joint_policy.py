from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CONTROLLER_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for path in (CONTROLLER_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configuration.aif_zoi_v2_params import resolve_aif_zoi_v2_params
from configuration.registry import ENVIRONMENTS
from policies.aif_zoi_v2_joint import AIFZOIV2JointPolicy
from policies.zoi_v2_model import (
    CELL_UNOBSERVED,
    O_BOTTLE,
    O_CAN,
    O_NOTHING,
    bayes_update,
    content_marginal,
    expected_information_gain,
    joint_likelihood,
    joint_obs_index,
    joint_prior,
    evidence_coefficients,
    feasible_grid,
    grid_weights,
    joint_prior_base_and_slope,
    joint_prior_terms,
    log_evidence,
    marginal_beliefs,
    marginal_beliefs_2d,
    predictive,
    rho_grid,
    rho_posterior,
    rho_support,
    table_probability,
)
from test_aif_zoi_v2_policy import box, make_obs

PARAMS = resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={"AIF_ZOI_V2_SURFACE_MODEL": "joint"})
PRIOR = joint_prior(lambda_table=PARAMS.lambda_table, lambda_floor=PARAMS.lambda_floor, rho=PARAMS.rho)


def L(w_table, w_floor):
    return joint_likelihood(
        w_table, w_floor, desk_in_view=w_table >= 0.05,
        recall=PARAMS.recall, class_accuracy=PARAMS.class_accuracy, false_alarm=PARAMS.false_alarm,
        desk_recall=PARAMS.desk_recall, desk_false_alarm=PARAMS.desk_false_alarm)


def desk_event(covered_cells, confidence=0.9):
    return {"class_name": "desk", "class_id": 2, "confidence": float(confidence),
            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "covered_cells": [list(c) for c in covered_cells]}


class JointPolicyBeliefTests(unittest.TestCase):
    def setUp(self):
        self.policy = AIFZOIV2JointPolicy(params=PARAMS)

    def qj(self, cell):
        return self.policy.q_joint[cell[1], cell[0], :].copy()

    def test_initial_belief_is_joint_prior(self):
        self.policy.compute_action(make_obs(processed=False))
        np.testing.assert_allclose(self.policy.q_joint[0, 0], PRIOR)
        np.testing.assert_allclose(self.policy.q_s[0, 0], content_marginal(PRIOR))
        self.assertAlmostEqual(float(self.policy.q_s[0, 0, 1:].sum()), 6 / 143)

    def test_desk_cover_updates_table_probability(self):
        cell = (3, 4)
        self.policy.compute_action(make_obs(planes={cell: (0.3, 0.9)}, events=[desk_event([cell])]))
        expected = bayes_update(PRIOR, L(0.3, 0.9), joint_obs_index(True, O_NOTHING))
        np.testing.assert_allclose(self.qj(cell), expected, atol=1e-12)
        self.assertGreater(table_probability(self.qj(cell)), PARAMS.rho)
        np.testing.assert_allclose(self.policy.q_s[4, 3], content_marginal(expected), atol=1e-12)

    def test_good_view_without_desk_lowers_table_probability(self):
        cell = (5, 6)
        self.policy.compute_action(make_obs(planes={cell: (1.0, 1.0)}))
        expected = bayes_update(PRIOR, L(1.0, 1.0), joint_obs_index(False, O_NOTHING))
        np.testing.assert_allclose(self.qj(cell), expected, atol=1e-12)
        self.assertLess(table_probability(self.qj(cell)), 0.01)

    def test_cover_below_gate_carries_no_desk_evidence(self):
        cell = (3, 4)
        self.policy.compute_action(make_obs(planes={cell: (0.02, 0.9)}, events=[desk_event([cell])]))
        expected = bayes_update(PRIOR, L(0.02, 0.9), joint_obs_index(False, O_NOTHING))
        np.testing.assert_allclose(self.qj(cell), expected, atol=1e-12)

    def test_target_box_uses_joint_likelihood(self):
        cell = (5, 6)
        self.policy.compute_action(make_obs(
            planes={cell: (0.6, 0.8)}, events=[box(cell, "target_can", 0.7), desk_event([cell])]))
        expected = bayes_update(PRIOR, L(0.6, 0.8), joint_obs_index(True, O_CAN))
        np.testing.assert_allclose(self.qj(cell), expected, atol=1e-12)

    def test_controller_table_map_does_not_enter_the_belief(self):
        planes = {(5, 6): (1.0, 1.0), (3, 4): (0.3, 0.9)}
        other = AIFZOIV2JointPolicy(params=PARAMS)
        self.policy.compute_action(make_obs(planes=planes, table_cells=[(3, 4), (2, 2)]))
        other.compute_action(make_obs(planes=planes))
        # The visible mask is the controller's (it differs with its table map), so
        # compare a cell whose gate does not depend on it.
        np.testing.assert_allclose(self.qj((5, 6)), other.q_joint[6, 5], atol=0.0)
        np.testing.assert_allclose(self.qj((2, 2)), PRIOR)
        self.assertTrue(np.all(self.policy.cell_type == CELL_UNOBSERVED))

    def test_throttled_step_changes_nothing(self):
        cell = (3, 4)
        self.policy.compute_action(make_obs(planes={cell: (0.2, 1.0)}))
        before = self.policy.q_joint.copy()
        self.policy.compute_action(make_obs(
            planes={cell: (1.0, 1.0)}, events=[desk_event([cell])], processed=False, time_s=0.1))
        np.testing.assert_array_equal(self.policy.q_joint, before)

    def test_q_s_stays_the_content_marginal(self):
        cells = {(5, 6): (1.0, 1.0), (3, 4): (0.3, 0.9), (4, 4): (0.6, 0.9)}
        self.policy.compute_action(make_obs(
            planes=cells, events=[desk_event([(4, 4), (3, 4)]), box((4, 4), "target_bottle", 0.8)]))
        q_s = self.policy.q_joint[:, :, 0:3] + self.policy.q_joint[:, :, 3:6]
        np.testing.assert_allclose(self.policy.q_s, q_s, atol=1e-12)


class JointPolicyPlanningTests(unittest.TestCase):
    def setUp(self):
        self.policy = AIFZOIV2JointPolicy(params=PARAMS)
        self.policy.compute_action(make_obs(
            planes={(5, 6): (1.0, 1.0), (3, 4): (0.3, 0.9), (4, 4): (0.4, 0.9)},
            events=[box((3, 4), "target_bottle", 0.8), desk_event([(4, 4)])],
            table_cells=[(4, 4)]))
        self.table_cells = {(4, 4)}

    def test_information_gain_uses_the_joint_likelihood(self):
        for target_cell, yaw_idx in (((3, 4), 0), ((4, 4), 2), ((5, 6), 3), ((0, 0), 7)):
            comps = self.policy._efe_components(
                target_cell=target_cell, target_yaw_index=yaw_idx, table_cells=self.table_cells)
            merged = self.policy._merged_candidate_coverages_rc(
                target_cell=target_cell, target_yaw_index=yaw_idx, table_cells=self.table_cells)
            floor_cov, table_cov = self.policy._cached_candidate_coverages_rc(
                target_cell=target_cell, target_yaw_index=yaw_idx)
            expected = sum(
                expected_information_gain(
                    self.policy.q_joint[r, c, :], L(table_cov.get((r, c), 0.0), floor_cov.get((r, c), 0.0)))
                for (r, c), w in merged.items()
                if w >= 0.05
            )
            self.assertAlmostEqual(comps["IG"], expected, places=12)

    def test_center_utility_marginalises_the_desk_outcome(self):
        for target_cell in ((3, 4), (4, 4)):
            comps = self.policy._efe_components(
                target_cell=target_cell, target_yaw_index=0, table_cells=self.table_cells)
            floor_cov, table_cov = self.policy._cached_candidate_coverages_rc(
                target_cell=target_cell, target_yaw_index=0)
            rc = (target_cell[1], target_cell[0])
            qj = self.policy.q_joint[rc[0], rc[1], :]
            p_o = predictive(qj, L(table_cov.get(rc, 0.0), floor_cov.get(rc, 0.0)))
            p_class = p_o[O_BOTTLE] + p_o[O_CAN] + p_o[3 + O_BOTTLE] + p_o[3 + O_CAN]
            q = content_marginal(qj)
            need = min(q[1], q[2])
            self.assertAlmostEqual(comps["utility"], 2.0 * need * p_class, places=12)

    def test_compute_action_returns_a_target(self):
        action = self.policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0)}, time_s=5.0))
        self.assertIn("forward", action)
        self.assertTrue(self.policy.navigator.has_target())

    def test_parameters_record_the_joint_model(self):
        params = self.policy.parameters()
        self.assertEqual(params["model"]["surface_model"], "joint")
        self.assertAlmostEqual(params["model"]["desk_recall"], 182 / 191)
        self.assertAlmostEqual(params["model"]["desk_false_alarm"], 189 / 1490)
        self.assertIn("desk_evidence", params["observation_model"])


LEARNED = resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={
    "AIF_ZOI_V2_SURFACE_MODEL": "joint", "AIF_ZOI_V2_RHO_MODEL": "learned"})


class LearnedRhoPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = AIFZOIV2JointPolicy(params=LEARNED)
        self.base, self.d = joint_prior_base_and_slope(n_targets=6, p_on_table=0.9, n_cells=143)
        self.grid = rho_grid(*rho_support(n_targets=6, p_on_table=0.9, n_cells=143), 256)

    def test_starts_from_the_uniform_rho_prior(self):
        self.policy.compute_action(make_obs(processed=False))
        mean, _ = self.policy.rho_estimate()
        self.assertAlmostEqual(mean, float(self.grid.mean()))
        self.assertAlmostEqual(float(self.policy.q_s[0, 0, 1:].sum()), 6 / 143)
        self.assertAlmostEqual(table_probability(self.policy.q_joint[0, 0]), mean)

    def test_beliefs_are_the_exact_average_over_rho(self):
        planes = {(5, 6): (1.0, 1.0), (3, 4): (0.3, 0.9), (4, 4): (0.6, 0.9)}
        self.policy.compute_action(make_obs(
            planes=planes, events=[desk_event([(4, 4)]), box((4, 4), "target_bottle", 0.8)]))
        ell = np.ones((143, 6))
        for (cx, cy), o in (((5, 6), joint_obs_index(False, O_NOTHING)), ((3, 4), joint_obs_index(False, O_NOTHING)),
                            ((4, 4), joint_obs_index(True, O_BOTTLE))):
            row = L(*planes[(cx, cy)])[o]
            ell[cy * 11 + cx] = row / row.max()
        w = rho_posterior(ell, self.base, self.d, self.grid)
        q = marginal_beliefs(ell, self.base, self.d, self.grid, w).reshape(13, 11, 6)
        np.testing.assert_allclose(self.policy.q_joint, q, atol=1e-12)
        np.testing.assert_allclose(self.policy.q_s, q[:, :, 0:3] + q[:, :, 3:6], atol=1e-12)

    def test_floor_views_lower_rho_and_are_logged(self):
        self.policy.compute_action(make_obs(processed=False))
        start, _ = self.policy.rho_estimate()
        for i, cell in enumerate([(5, 6), (6, 6), (7, 6), (4, 6), (5, 7), (5, 5), (6, 7), (4, 5)]):
            self.policy.compute_action(make_obs(planes={cell: (1.0, 1.0)}, current_cell=cell, time_s=0.1 * (i + 1)))
        mean, sd = self.policy.rho_estimate()
        self.assertLess(mean, start)
        diag = self.policy.diagnostics()
        self.assertAlmostEqual(diag["rho_mean"], round(mean, 5))
        self.assertIn("rho_sd", diag)
        self.assertEqual(self.policy.parameters()["observation_model"]["rho_grid_size"], 256)

    def test_fixed_mode_logs_nothing(self):
        policy = AIFZOIV2JointPolicy(params=PARAMS)
        policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0)}))
        self.assertEqual(policy.diagnostics(), {})


LEARNED_THETA = resolve_aif_zoi_v2_params(ENVIRONMENTS["hard"], environ={
    "AIF_ZOI_V2_SURFACE_MODEL": "joint", "AIF_ZOI_V2_RHO_MODEL": "learned", "AIF_ZOI_V2_TARGET_MODEL": "learned"})


class LearnedThetaPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = AIFZOIV2JointPolicy(params=LEARNED_THETA)
        self.terms = joint_prior_terms(p_on_table=0.9)
        self.rho, self.theta = feasible_grid(p_on_table=0.9, n_rho=128, n_theta=128)

    def fly(self, n_frames=6):
        cells = [(5, 6), (6, 6), (7, 6), (4, 6), (5, 7), (5, 5)]
        for i in range(n_frames):
            c = cells[i % len(cells)]
            planes = {c: (1.0, 1.0), (c[0], c[1] + 1): (0.4, 0.9)}
            events = [desk_event([(4, 4)])] if i == 2 else []
            if i == 3:
                events.append(box((4, 6), "target_bottle", 0.9))
            if i == 2:
                planes[(4, 4)] = (0.6, 0.9)
            self.policy.compute_action(make_obs(planes=planes, events=events, current_cell=c, time_s=0.1 * (i + 1)))

    def test_starts_from_the_uniform_prior_over_rho_and_theta(self):
        self.policy.compute_action(make_obs(processed=False))
        theta, _ = self.policy.theta_estimate()
        self.assertAlmostEqual(theta, float(self.theta.mean()))
        # With no evidence an unseen cell's P(target) is the prior mean of theta (~1/3).
        self.assertAlmostEqual(float(self.policy.q_s[0, 0, 1:].sum()), theta, places=6)

    def test_incremental_evidence_and_shared_unseen_belief_match_a_full_recomputation(self):
        self.fly()
        ell = self.policy._ell.reshape(-1, 6)
        coef = evidence_coefficients(ell, *self.terms)
        np.testing.assert_allclose(self.policy._log_ev, log_evidence(coef, self.rho, self.theta), atol=1e-9)
        w = grid_weights(log_evidence(coef, self.rho, self.theta))
        q = marginal_beliefs_2d(ell, coef, self.terms, self.rho, self.theta, w).reshape(13, 11, 6)
        # The policy skips grid points with weight < 1e-12 of the largest: differences < 1e-8.
        np.testing.assert_allclose(self.policy.q_joint, q, atol=1e-8)
        np.testing.assert_allclose(self.policy.q_s, q[:, :, 0:3] + q[:, :, 3:6], atol=1e-8)

    def test_empty_views_lower_theta_and_are_logged(self):
        self.policy.compute_action(make_obs(processed=False))
        start, _ = self.policy.theta_estimate()
        self.fly()
        theta, _ = self.policy.theta_estimate()
        self.assertLess(theta, start)
        diag = self.policy.diagnostics()
        for key in ("rho_mean", "rho_sd", "theta_mean", "theta_sd"):
            self.assertIn(key, diag)
        obs_model = self.policy.parameters()["observation_model"]
        self.assertEqual(obs_model["rho_theta_grid_size"], 128)
        self.assertIn("learned", obs_model["targets"])

    def test_known_target_density_keeps_the_learned_rho_path(self):
        policy = AIFZOIV2JointPolicy(params=LEARNED)
        self.assertFalse(policy.theta_learned)
        policy.compute_action(make_obs(planes={(5, 6): (1.0, 1.0)}))
        self.assertNotIn("theta_mean", policy.diagnostics())


if __name__ == "__main__":
    unittest.main()
