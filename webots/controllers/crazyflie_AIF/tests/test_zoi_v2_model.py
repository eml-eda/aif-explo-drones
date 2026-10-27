from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CONTROLLER_DIR = Path(__file__).resolve().parents[1]
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

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
    prior_present,
    prior_vector,
    reweight,
    table_probability_given_target,
)
from policies.zoi_v2_model import (
    D_COVERED,
    D_NONE,
    N_JOINT,
    SURF_FLOOR,
    SURF_TABLE,
    content_marginal,
    desk_likelihood,
    joint_index,
    joint_likelihood,
    joint_obs_index,
    joint_prior,
    table_probability,
)
from policies.zoi_v2_model import (
    evidence_coefficients,
    feasible_grid,
    grid_weights,
    joint_prior_terms,
    log_evidence,
    marginal_beliefs_2d,
    joint_prior_base_and_slope,
    marginal_beliefs,
    rho_grid,
    rho_posterior,
    rho_support,
)

R, A_CLS, F = 0.84, 0.98, 0.01
PRIOR_KW = {"lambda_table": 0.54, "lambda_floor": 0.0045, "rho": 0.07}


def A(w: float) -> np.ndarray:
    return likelihood_matrix(w, recall=R, class_accuracy=A_CLS, false_alarm=F)


class LikelihoodTests(unittest.TestCase):
    def test_columns_sum_to_one(self):
        for w in (0.0, 0.3, 1.0):
            np.testing.assert_allclose(A(w).sum(axis=0), np.ones(3), atol=1e-15)

    def test_entries_match_model_table(self):
        w = 0.5
        p = w * R
        expected = np.array([
            [1 - F, 1 - p, 1 - p],
            [F / 2, p * A_CLS, p * (1 - A_CLS)],
            [F / 2, p * (1 - A_CLS), p * A_CLS],
        ])
        np.testing.assert_allclose(A(w), expected, atol=1e-15)

    def test_detection_probability_is_floored_at_false_alarm(self):
        # w * r = 0.0042 < f, so p = f.
        A_low = A(0.005)
        self.assertAlmostEqual(A_low[O_NOTHING, 1], 1 - F)
        self.assertAlmostEqual(A_low[O_BOTTLE, 1], F * A_CLS)

    def test_no_zero_entries_with_valid_parameters(self):
        for w in (0.0, 0.05, 1.0):
            self.assertTrue(np.all(A(w) > 0.0))


class PriorTests(unittest.TestCase):
    def test_prior_present_by_type(self):
        self.assertAlmostEqual(prior_present(CELL_TABLE, **PRIOR_KW), 0.54)
        self.assertAlmostEqual(prior_present(CELL_FLOOR, **PRIOR_KW), 0.0045)
        self.assertAlmostEqual(
            prior_present(CELL_UNOBSERVED, **PRIOR_KW), 0.07 * 0.54 + 0.93 * 0.0045)

    def test_prior_vector_is_a_distribution_with_even_class_split(self):
        configs = (PRIOR_KW, {"lambda_table": 0.54, "lambda_floor": 6.0 / 143.0, "rho": 0.0})
        for kw in configs:
            for cell_type in (CELL_UNOBSERVED, CELL_FLOOR, CELL_TABLE):
                q = prior_vector(cell_type, **kw)
                self.assertAlmostEqual(float(q.sum()), 1.0)
                self.assertTrue(np.all((q > 0.0) & (q < 1.0)))
                self.assertAlmostEqual(q[1], q[2])

    def test_unknown_cell_type_raises(self):
        with self.assertRaises(ValueError):
            prior_present(7, **PRIOR_KW)


class UpdateTests(unittest.TestCase):
    def test_hand_computed_posterior(self):
        # Table prior [0.46, 0.27, 0.27], full coverage, "bottle" box:
        # unnormalized [0.46*0.005, 0.27*0.84*0.98, 0.27*0.84*0.02] = sum 0.2291.
        q = bayes_update(prior_vector(CELL_TABLE, **PRIOR_KW), A(1.0), O_BOTTLE)
        np.testing.assert_allclose(
            q, [0.01003928415539066, 0.9701615015277171, 0.01979921431689219], atol=1e-12)

    def test_updates_are_order_invariant(self):
        q0 = prior_vector(CELL_UNOBSERVED, **PRIOR_KW)
        seq = [(0.3, O_NOTHING), (1.0, O_CAN), (0.7, O_BOTTLE), (0.1, O_NOTHING)]
        q1 = q0
        for w, o in seq:
            q1 = bayes_update(q1, A(w), o)
        q2 = q0
        for w, o in reversed(seq):
            q2 = bayes_update(q2, A(w), o)
        np.testing.assert_allclose(q1, q2, atol=1e-12)

    def test_reweight_preserves_evidence(self):
        seq = [(0.4, O_NOTHING), (0.9, O_BOTTLE), (0.2, O_NOTHING), (1.0, O_CAN)]
        for old_type, new_type in (
            (CELL_UNOBSERVED, CELL_FLOOR),
            (CELL_UNOBSERVED, CELL_TABLE),
            (CELL_FLOOR, CELL_TABLE),
        ):
            p_old = prior_vector(old_type, **PRIOR_KW)
            p_new = prior_vector(new_type, **PRIOR_KW)
            q_old = p_old
            q_ref = p_new
            for w, o in seq:
                q_old = bayes_update(q_old, A(w), o)
                q_ref = bayes_update(q_ref, A(w), o)
            np.testing.assert_allclose(reweight(q_old, p_old, p_new), q_ref, atol=1e-12)

    def test_negative_evidence_lowers_target_probability(self):
        q0 = prior_vector(CELL_TABLE, **PRIOR_KW)
        q1 = bayes_update(q0, A(1.0), O_NOTHING)
        self.assertLess(q1[1] + q1[2], q0[1] + q0[2])


class VisibilityTests(unittest.TestCase):
    def test_table_probability_by_cell_type(self):
        self.assertEqual(table_probability_given_target(CELL_TABLE, **PRIOR_KW), 1.0)
        self.assertEqual(table_probability_given_target(CELL_FLOOR, **PRIOR_KW), 0.0)
        expected = 0.07 * 0.54 / (0.07 * 0.54 + 0.93 * 0.0045)
        self.assertAlmostEqual(table_probability_given_target(CELL_UNOBSERVED, **PRIOR_KW), expected)
        self.assertAlmostEqual(expected, 0.9003, places=4)

    def test_no_table_environment_gives_zero_table_probability(self):
        kw = {"lambda_table": 0.54, "lambda_floor": 6.0 / 143.0, "rho": 0.0}
        for cell_type in (CELL_UNOBSERVED, CELL_FLOOR):
            self.assertEqual(table_probability_given_target(cell_type, **kw), 0.0)
            self.assertEqual(object_visibility(cell_type, 0.2, 0.9, **kw), 0.9)

    def test_visibility_mixes_planes_by_cell_type(self):
        v_table, v_floor = 0.2, 0.9
        self.assertEqual(object_visibility(CELL_TABLE, v_table, v_floor, **PRIOR_KW), v_table)
        self.assertEqual(object_visibility(CELL_FLOOR, v_table, v_floor, **PRIOR_KW), v_floor)
        pi = table_probability_given_target(CELL_UNOBSERVED, **PRIOR_KW)
        self.assertAlmostEqual(
            object_visibility(CELL_UNOBSERVED, v_table, v_floor, **PRIOR_KW),
            pi * v_table + (1 - pi) * v_floor)

    def test_mixed_visibility_equals_mixed_surface_likelihoods(self):
        # P(o | target) is linear in v when v * r >= f on both planes.
        v_table, v_floor = 0.2, 0.9
        pi = table_probability_given_target(CELL_UNOBSERVED, **PRIOR_KW)
        v = object_visibility(CELL_UNOBSERVED, v_table, v_floor, **PRIOR_KW)
        mixed = pi * A(v_table) + (1 - pi) * A(v_floor)
        np.testing.assert_allclose(A(v)[:, 1:], mixed[:, 1:], atol=1e-15)


class InformationGainTests(unittest.TestCase):
    def test_non_negative_and_non_decreasing_in_coverage(self):
        q = prior_vector(CELL_TABLE, **PRIOR_KW)
        gains = [expected_information_gain(q, A(w)) for w in np.linspace(0.0, 1.0, 21)]
        self.assertTrue(all(g >= 0.0 for g in gains))
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(gains, gains[1:])))

    def test_certain_belief_has_no_gain(self):
        q = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(expected_information_gain(q, A(1.0)), 0.0, places=9)


# Joint model: centre-on-desk priors and illustrative desk numbers.
JOINT_PRIOR_KW = {"lambda_table": 5.4 / 10.08, "lambda_floor": 0.6 / 132.92, "rho": 10.08 / 143}
DESK_KW = {"desk_recall": 0.95, "desk_false_alarm": 0.13}
GATE = 0.05
OBS_KW = {"recall": R, "class_accuracy": A_CLS, "false_alarm": 0.001}


def L(w_table, w_floor, **desk):
    return joint_likelihood(w_table, w_floor, desk_in_view=w_table >= GATE, **OBS_KW, **(desk or DESK_KW))


class JointModelTests(unittest.TestCase):
    def setUp(self):
        self.q0 = joint_prior(**JOINT_PRIOR_KW)

    def test_prior_marginals(self):
        self.assertAlmostEqual(float(self.q0.sum()), 1.0)
        self.assertAlmostEqual(table_probability(self.q0), JOINT_PRIOR_KW["rho"])
        # Same unobserved prior as the typed model: N / M.
        np.testing.assert_allclose(content_marginal(self.q0), prior_vector(CELL_UNOBSERVED, **JOINT_PRIOR_KW))
        self.assertAlmostEqual(1.0 - content_marginal(self.q0)[0], 6 / 143)

    def test_indexing(self):
        self.assertEqual(joint_index(SURF_TABLE, O_CAN), 2)
        self.assertEqual(joint_index(SURF_FLOOR, 0), 3)
        self.assertEqual(joint_obs_index(False, O_NOTHING), 0)
        self.assertEqual(joint_obs_index(True, O_BOTTLE), 4)
        self.assertEqual((D_NONE, D_COVERED), (0, 1))

    def test_likelihood_columns_sum_to_one(self):
        for wt, wf in ((0.0, 0.0), (0.2, 0.9), (1.0, 1.0)):
            lik = L(wt, wf)
            self.assertEqual(lik.shape, (N_JOINT, N_JOINT))
            np.testing.assert_allclose(lik.sum(axis=0), np.ones(N_JOINT), atol=1e-15)
            np.testing.assert_allclose(
                desk_likelihood(wt >= GATE, **DESK_KW).sum(axis=0), np.ones(N_JOINT), atol=1e-15)

    def test_desk_cover_raises_table_probability(self):
        # Compare covered and not covered under the same target observation: the
        # odds of table differ by (d / g) * ((1 - g) / (1 - d)) whenever the desk is in view.
        lik = L(1.0, 1.0)
        q = bayes_update(self.q0, lik, joint_obs_index(True, O_NOTHING))
        q_none = bayes_update(self.q0, lik, joint_obs_index(False, O_NOTHING))
        odds = lambda x: table_probability(x) / (1.0 - table_probability(x))
        d, g = DESK_KW["desk_recall"], DESK_KW["desk_false_alarm"]
        self.assertAlmostEqual(odds(q) / odds(q_none), (d / g) * ((1 - g) / (1 - d)))
        self.assertGreater(table_probability(q), JOINT_PRIOR_KW["rho"])
        q2 = bayes_update(q, lik, joint_obs_index(True, O_NOTHING))
        self.assertGreater(table_probability(q2), table_probability(q))

    def test_full_view_without_desk_lowers_table_probability(self):
        q = bayes_update(self.q0, L(1.0, 1.0), joint_obs_index(False, O_NOTHING))
        self.assertLess(table_probability(q), 0.01)

    def test_desk_out_of_view_carries_no_desk_information(self):
        out = desk_likelihood(False, **DESK_KW)
        np.testing.assert_allclose(out[D_NONE], np.ones(N_JOINT))
        np.testing.assert_allclose(out[D_COVERED], np.zeros(N_JOINT))
        # Desk evidence is the same at any table-plane coverage above the gate.
        np.testing.assert_allclose(L(0.1, 0.9)[3:].sum(axis=0), L(0.9, 0.9)[3:].sum(axis=0))

    def test_glancing_view_changes_little(self):
        q = bayes_update(self.q0, L(0.02, 0.3), joint_obs_index(False, O_NOTHING))
        self.assertAlmostEqual(table_probability(q), JOINT_PRIOR_KW["rho"], delta=0.005)

    def test_target_box_raises_table_probability(self):
        # Uninformative desk evidence (d = g) and equal visibility on both planes.
        flat = {"desk_recall": 0.5, "desk_false_alarm": 0.5}
        q = bayes_update(self.q0, L(0.8, 0.8, **flat), joint_obs_index(False, O_BOTTLE))
        # Close to P(table | target) = rho*lambda_T / (rho*lambda_T + (1-rho)*lambda_F) = 0.9,
        # a little lower because a false alarm on an empty cell can also explain the box.
        self.assertGreater(table_probability(q), 0.8)

    def test_uninformative_desk_matches_typed_unobserved_update(self):
        # With d = g the desk observation carries no information, and the content
        # marginal equals the typed model's update of an unobserved cell.
        flat = {"desk_recall": 0.6, "desk_false_alarm": 0.6}
        for wt, wf, o in ((0.3, 0.9, O_NOTHING), (0.8, 1.0, O_CAN), (1.0, 0.5, O_BOTTLE)):
            for covered in (False, True):
                q = bayes_update(self.q0, L(wt, wf, **flat), joint_obs_index(covered, o))
                typed = bayes_update(
                    prior_vector(CELL_UNOBSERVED, **JOINT_PRIOR_KW),
                    likelihood_matrix(object_visibility(CELL_UNOBSERVED, wt, wf, **JOINT_PRIOR_KW), **OBS_KW),
                    o)
                np.testing.assert_allclose(content_marginal(q), typed, atol=1e-12)

    def test_information_gain_counts_desk_evidence(self):
        no_desk = {"desk_recall": 0.5, "desk_false_alarm": 0.5}
        self.assertGreater(
            expected_information_gain(self.q0, L(1.0, 1.0)),
            expected_information_gain(self.q0, L(1.0, 1.0, **no_desk)))


SCEN = {"n_targets": 6, "p_on_table": 0.9, "n_cells": 143}


class LearnedRhoTests(unittest.TestCase):
    def setUp(self):
        self.base, self.d = joint_prior_base_and_slope(**SCEN)
        self.lo, self.hi = rho_support(**SCEN)
        self.grid = rho_grid(self.lo, self.hi, 256)

    def lik(self, covered, target_obs, wt=1.0, wf=1.0):
        return L(wt, wf)[joint_obs_index(covered, target_obs)]

    def test_linear_prior_matches_the_joint_prior(self):
        rho = 10.08 / 143
        lam_t, lam_f = 5.4 / (143 * rho), 0.6 / (143 * (1 - rho))
        np.testing.assert_allclose(self.base + rho * self.d,
                                   joint_prior(lambda_table=lam_t, lambda_floor=lam_f, rho=rho), atol=1e-15)

    def test_prior_is_positive_inside_the_support(self):
        self.assertAlmostEqual(self.lo, 5.4 / 143)
        self.assertAlmostEqual(self.hi, 1 - 0.6 / 143)
        prior = self.base[None, :] + self.grid[:, None] * self.d[None, :]
        self.assertTrue(np.all(prior > 0))

    def test_no_evidence_keeps_the_uniform_prior(self):
        ell = np.ones((143, 6))
        w = rho_posterior(ell, self.base, self.d, self.grid)
        np.testing.assert_allclose(w, np.full(256, 1 / 256))
        q = marginal_beliefs(ell, self.base, self.d, self.grid, w)
        np.testing.assert_allclose(q.sum(axis=1), np.ones(143))
        self.assertAlmostEqual(1 - (q[0, 0] + q[0, 3]), 6 / 143)  # target mass does not depend on rho

    def test_floor_evidence_lowers_rho_and_desk_evidence_raises_it(self):
        mean = lambda w: float(w @ self.grid)
        floor = np.ones((143, 6)); floor[:100] = self.lik(False, O_NOTHING)
        desk = np.ones((143, 6)); desk[:100] = self.lik(True, O_NOTHING)
        self.assertLess(mean(rho_posterior(floor, self.base, self.d, self.grid)), 0.06)
        self.assertGreater(mean(rho_posterior(desk, self.base, self.d, self.grid)), 0.5)

    def test_concentrated_posterior_reduces_to_the_fixed_update(self):
        ell = np.ones((143, 6)); ell[7] = self.lik(True, O_BOTTLE, 0.6, 0.9)
        w = np.zeros(256); w[40] = 1.0
        q = marginal_beliefs(ell, self.base, self.d, self.grid, w)
        rho = self.grid[40]
        expected = bayes_update(self.base + rho * self.d, L(0.6, 0.9), joint_obs_index(True, O_BOTTLE))
        np.testing.assert_allclose(q[7], expected, atol=1e-12)

    def test_grid_resolution_does_not_matter(self):
        ell = np.ones((143, 6))
        ell[:60] = self.lik(False, O_NOTHING)
        ell[60:66] = self.lik(True, O_NOTHING, 0.4, 0.9)
        ell[66] = self.lik(True, O_CAN)
        out = []
        for size in (256, 1024):
            g = rho_grid(self.lo, self.hi, size)
            w = rho_posterior(ell, self.base, self.d, g)
            out.append((float(w @ g), marginal_beliefs(ell, self.base, self.d, g, w)))
        self.assertAlmostEqual(out[0][0], out[1][0], places=4)
        np.testing.assert_allclose(out[0][1], out[1][1], atol=1e-3)



class LearnedRhoThetaTests(unittest.TestCase):
    def setUp(self):
        self.terms = joint_prior_terms(p_on_table=0.9)
        self.rho, self.theta = feasible_grid(p_on_table=0.9, n_rho=128, n_theta=128)

    def lik(self, covered, target_obs, wt=1.0, wf=1.0):
        return L(wt, wf)[joint_obs_index(covered, target_obs)]

    def posterior(self, ell, rho=None, theta=None):
        rho = self.rho if rho is None else rho
        theta = self.theta if theta is None else theta
        coef = evidence_coefficients(ell, *self.terms)
        w = grid_weights(log_evidence(coef, rho, theta))
        return w, marginal_beliefs_2d(ell, coef, self.terms, rho, theta, w)

    def test_prior_terms_match_the_learned_rho_prior_at_known_theta(self):
        base, d_rho, d_theta = self.terms
        b1, d1 = joint_prior_base_and_slope(n_targets=6, p_on_table=0.9, n_cells=143)
        for rho in (0.05, 0.3):
            np.testing.assert_allclose(base + rho * d_rho + (6 / 143) * d_theta, b1 + rho * d1, atol=1e-15)

    def test_prior_is_positive_on_the_feasible_grid(self):
        base, d_rho, d_theta = self.terms
        prior = base[None, :] + self.rho[:, None] * d_rho[None, :] + self.theta[:, None] * d_theta[None, :]
        self.assertTrue(np.all(prior > 0))
        np.testing.assert_allclose(prior.sum(axis=1), 1.0)

    def test_reduces_to_learned_rho_when_theta_is_fixed(self):
        ell = np.ones((143, 6)); ell[:30] = self.lik(False, O_NOTHING); ell[30:34] = self.lik(True, O_NOTHING, 0.4, 0.9)
        ell[34] = self.lik(True, O_CAN)
        lo, hi = rho_support(**SCEN); g = rho_grid(lo, hi, 256)
        w1 = rho_posterior(ell, *joint_prior_base_and_slope(**SCEN), g)
        q1 = marginal_beliefs(ell, *joint_prior_base_and_slope(**SCEN), g, w1)
        w2, q2 = self.posterior(ell, rho=g, theta=np.full_like(g, 6 / 143))
        np.testing.assert_allclose(w2, w1, atol=1e-12)
        np.testing.assert_allclose(q2, q1, atol=1e-12)

    def test_three_sums_equal_the_explicit_average(self):
        ell = np.ones((5, 6)); ell[0] = self.lik(False, O_NOTHING); ell[1] = self.lik(True, O_BOTTLE)
        rho, theta = feasible_grid(p_on_table=0.9, n_rho=16, n_theta=16)
        w, q = self.posterior(ell, rho, theta)
        base, d_rho, d_theta = self.terms
        prior = base[None, :] + rho[:, None] * d_rho[None, :] + theta[:, None] * d_theta[None, :]
        post = ell[:, None, :] * prior[None, :, :]
        post /= post.sum(axis=2, keepdims=True)
        np.testing.assert_allclose(q, np.einsum("g,cgs->cs", w, post), atol=1e-12)
        np.testing.assert_allclose(q.sum(axis=1), np.ones(5))

    def test_empty_cells_lower_theta_and_target_boxes_raise_it(self):
        mean_theta = lambda ell: float(self.posterior(ell)[0] @ self.theta)
        start = mean_theta(np.ones((143, 6)))
        empty = np.ones((143, 6)); empty[:100] = self.lik(False, O_NOTHING)
        found = empty.copy(); found[100:106] = self.lik(True, O_BOTTLE)
        self.assertLess(mean_theta(empty), 0.05)
        self.assertLess(mean_theta(empty), start)
        self.assertGreater(mean_theta(found), mean_theta(empty))

    def test_unseen_cells_contribute_nothing_to_the_evidence(self):
        coef = evidence_coefficients(np.ones((10, 6)), *self.terms)
        np.testing.assert_allclose(log_evidence(coef, self.rho, self.theta), 0.0, atol=1e-12)

    def test_grid_resolution_does_not_matter(self):
        ell = np.ones((143, 6)); ell[:80] = self.lik(False, O_NOTHING); ell[80:86] = self.lik(True, O_NOTHING, 0.5, 0.9)
        ell[86:90] = self.lik(True, O_BOTTLE)
        out = []
        for n in (128, 256):
            rho, theta = feasible_grid(p_on_table=0.9, n_rho=n, n_theta=n)
            w, q = self.posterior(ell, rho, theta)
            out.append((float(w @ rho), float(w @ theta), q))
        self.assertAlmostEqual(out[0][0], out[1][0], delta=2e-3)
        self.assertAlmostEqual(out[0][1], out[1][1], delta=2e-3)
        # 128 vs 256 points per side: beliefs within 0.5 percentage points (256 vs 512: identical).
        np.testing.assert_allclose(out[0][2], out[1][2], atol=5e-3)


if __name__ == "__main__":
    unittest.main()
