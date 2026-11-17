from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CONTROLLER_DIR = Path(__file__).resolve().parents[1]
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

from policies.zoi_model import (
    D_COVERED,
    D_NONE,
    N_JOINT,
    O_BOTTLE,
    O_CAN,
    O_NOTHING,
    SURF_FLOOR,
    SURF_TABLE,
    content_marginal,
    desk_likelihood,
    evidence_coefficients,
    expected_information_gain,
    expected_information_gain_batch,
    feasible_grid,
    floor_only_grid,
    grid_weights,
    joint_index,
    joint_likelihood,
    joint_likelihood_batch,
    joint_obs_index,
    joint_prior_terms,
    likelihood_matrix,
    log_evidence,
    marginal_beliefs_2d,
    predictive,
    table_probability,
)

R, A_CLS, F = 0.84, 0.98, 0.01
DESK_KW = {"desk_recall": 0.95, "desk_false_alarm": 0.13}
GATE = 0.05
OBS_KW = {"recall": R, "class_accuracy": A_CLS, "false_alarm": 0.001}
PI = 0.9
TERMS = joint_prior_terms(p_on_table=PI)


def A(w: float) -> np.ndarray:
    return likelihood_matrix(w, recall=R, class_accuracy=A_CLS, false_alarm=F)


def L(w_table, w_floor, **desk):
    return joint_likelihood(w_table, w_floor, desk_in_view=w_table >= GATE, **OBS_KW, **(desk or DESK_KW))


def prior_at(rho, theta, terms=TERMS):
    base, d_rho, d_theta = terms
    return base + rho * d_rho + theta * d_theta


def bayes(q, lik, o):
    post = lik[o] * q
    return post / post.sum()


# 3 desks of 1.2 m x 0.7 m on 0.5 m cells, 6 targets, 143 cells.
Q0 = prior_at(10.08 / 143, 6 / 143)


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


class InformationGainTests(unittest.TestCase):
    def test_non_negative_and_non_decreasing_in_coverage(self):
        q = np.array([0.5, 0.25, 0.25])
        gains = [expected_information_gain(q, A(w)) for w in np.linspace(0.0, 1.0, 21)]
        self.assertTrue(all(g >= 0.0 for g in gains))
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(gains, gains[1:])))

    def test_certain_belief_has_no_gain(self):
        q = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(expected_information_gain(q, A(1.0)), 0.0, places=9)

    def test_information_gain_counts_desk_evidence(self):
        no_desk = {"desk_recall": 0.5, "desk_false_alarm": 0.5}
        self.assertGreater(expected_information_gain(Q0, L(1.0, 1.0)),
                           expected_information_gain(Q0, L(1.0, 1.0, **no_desk)))


class JointModelTests(unittest.TestCase):
    def test_prior_marginals(self):
        self.assertAlmostEqual(float(Q0.sum()), 1.0)
        self.assertAlmostEqual(table_probability(Q0), 10.08 / 143)
        self.assertAlmostEqual(1.0 - content_marginal(Q0)[0], 6 / 143)
        # pi of the targets on desks, split evenly between bottle and can.
        self.assertAlmostEqual(Q0[1] + Q0[2], PI * 6 / 143)
        self.assertAlmostEqual(Q0[1], Q0[2])

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
        # Covered vs not covered under the same target observation: the odds of
        # table differ by (d / g) * ((1 - g) / (1 - d)) whenever the desk is in view.
        lik = L(1.0, 1.0)
        q = bayes(Q0, lik, joint_obs_index(True, O_NOTHING))
        q_none = bayes(Q0, lik, joint_obs_index(False, O_NOTHING))
        odds = lambda x: table_probability(x) / (1.0 - table_probability(x))
        d, g = DESK_KW["desk_recall"], DESK_KW["desk_false_alarm"]
        self.assertAlmostEqual(odds(q) / odds(q_none), (d / g) * ((1 - g) / (1 - d)))
        self.assertGreater(table_probability(q), table_probability(Q0))

    def test_full_view_without_desk_lowers_table_probability(self):
        q = bayes(Q0, L(1.0, 1.0), joint_obs_index(False, O_NOTHING))
        self.assertLess(table_probability(q), 0.01)

    def test_desk_out_of_view_carries_no_desk_information(self):
        out = desk_likelihood(False, **DESK_KW)
        np.testing.assert_allclose(out[D_NONE], np.ones(N_JOINT))
        np.testing.assert_allclose(out[D_COVERED], np.zeros(N_JOINT))
        # Desk evidence is the same at any table-plane coverage above the gate.
        np.testing.assert_allclose(L(0.1, 0.9)[3:].sum(axis=0), L(0.9, 0.9)[3:].sum(axis=0))

    def test_glancing_view_changes_little(self):
        q = bayes(Q0, L(0.02, 0.3), joint_obs_index(False, O_NOTHING))
        self.assertAlmostEqual(table_probability(q), table_probability(Q0), delta=0.005)

    def test_target_box_raises_table_probability(self):
        # Uninformative desk evidence (d = g) and equal visibility on both planes:
        # close to P(table | target) = pi, a little lower because a false alarm on
        # an empty cell can also explain the box.
        flat = {"desk_recall": 0.5, "desk_false_alarm": 0.5}
        q = bayes(Q0, L(0.8, 0.8, **flat), joint_obs_index(False, O_BOTTLE))
        self.assertGreater(table_probability(q), 0.8)

    def test_target_likelihood_uses_each_surface_plane(self):
        lik = L(0.2, 0.9)
        np.testing.assert_allclose(lik[0:3, 0:3] + lik[3:6, 0:3], likelihood_matrix(0.2, **OBS_KW))
        np.testing.assert_allclose(lik[0:3, 3:6] + lik[3:6, 3:6], likelihood_matrix(0.9, **OBS_KW))
        p_o = predictive(Q0, lik)
        self.assertAlmostEqual(float(p_o.sum()), 1.0)


class BatchTests(unittest.TestCase):
    """The batched functions of the vectorized planner give the scalar results exactly."""

    def test_joint_likelihood_and_information_gain(self):
        rng = np.random.default_rng(0)
        n = 500
        w_t, w_f = rng.random(n), rng.random(n)
        w_t[:50] = 0.0
        in_view = w_t >= GATE
        lik = joint_likelihood_batch(w_t, w_f, desk_in_view=in_view, **OBS_KW, **DESK_KW)
        q = rng.dirichlet(np.ones(6), n)
        q[:10] = Q0
        ig = expected_information_gain_batch(q, lik)
        for i in range(n):
            ref = joint_likelihood(w_t[i], w_f[i], desk_in_view=bool(in_view[i]), **OBS_KW, **DESK_KW)
            np.testing.assert_array_equal(lik[i], ref)
            self.assertEqual(ig[i], expected_information_gain(q[i], ref))


class LearnedPriorTests(unittest.TestCase):
    def setUp(self):
        self.rho, self.theta = feasible_grid(p_on_table=PI, n_rho=128, n_theta=128)

    def lik(self, covered, target_obs, wt=1.0, wf=1.0):
        return L(wt, wf)[joint_obs_index(covered, target_obs)]

    def posterior(self, ell, rho=None, theta=None, terms=TERMS):
        rho = self.rho if rho is None else rho
        theta = self.theta if theta is None else theta
        coef = evidence_coefficients(ell, *terms)
        w = grid_weights(log_evidence(coef, rho, theta))
        return w, marginal_beliefs_2d(ell, coef, terms, rho, theta, w)

    def test_prior_is_positive_on_the_feasible_grid(self):
        prior = np.array([prior_at(r, t) for r, t in zip(self.rho, self.theta)])
        self.assertTrue(np.all(prior > 0))
        np.testing.assert_allclose(prior.sum(axis=1), 1.0)
        # Feasible region: theta * pi < rho < 1 - theta * (1 - pi).
        self.assertTrue(np.all(self.rho > self.theta * PI))
        self.assertTrue(np.all(self.rho < 1.0 - self.theta * (1.0 - PI)))

    def test_concentrated_posterior_is_the_bayes_update_at_that_point(self):
        ell = np.ones((3, 6))
        ell[0] = self.lik(False, O_NOTHING)
        ell[1] = self.lik(True, O_BOTTLE, 0.4, 0.9)
        rho, theta = np.array([10.08 / 143]), np.array([6 / 143])
        w, q = self.posterior(ell, rho, theta)
        np.testing.assert_allclose(q[0], bayes(Q0, L(1.0, 1.0), joint_obs_index(False, O_NOTHING)), atol=1e-12)
        np.testing.assert_allclose(q[1], bayes(Q0, L(0.4, 0.9), joint_obs_index(True, O_BOTTLE)), atol=1e-12)
        np.testing.assert_allclose(q[2], Q0, atol=1e-12)

    def test_three_sums_equal_the_explicit_average(self):
        ell = np.ones((5, 6)); ell[0] = self.lik(False, O_NOTHING); ell[1] = self.lik(True, O_BOTTLE)
        rho, theta = feasible_grid(p_on_table=PI, n_rho=16, n_theta=16)
        w, q = self.posterior(ell, rho, theta)
        prior = np.array([prior_at(r, t) for r, t in zip(rho, theta)])
        post = ell[:, None, :] * prior[None, :, :]
        post /= post.sum(axis=2, keepdims=True)
        np.testing.assert_allclose(q, np.einsum("g,cgs->cs", w, post), atol=1e-12)
        np.testing.assert_allclose(q.sum(axis=1), np.ones(5))

    def test_evidence_is_the_product_of_the_cells_normalisers(self):
        ell = np.ones((4, 6)); ell[0] = self.lik(False, O_NOTHING); ell[1] = self.lik(True, O_CAN, 0.5, 0.9)
        coef = evidence_coefficients(ell, *TERMS)
        for g in (0, 100, 5000):
            z = [float(e @ prior_at(self.rho[g], self.theta[g])) for e in ell]
            self.assertAlmostEqual(float(log_evidence(coef, self.rho[g:g + 1], self.theta[g:g + 1])[0]),
                                   float(np.log(z).sum()), places=12)

    def test_empty_cells_lower_theta_and_target_boxes_raise_it(self):
        mean_theta = lambda ell: float(self.posterior(ell)[0] @ self.theta)
        start = mean_theta(np.ones((143, 6)))
        empty = np.ones((143, 6)); empty[:100] = self.lik(False, O_NOTHING)
        found = empty.copy(); found[100:106] = self.lik(True, O_BOTTLE)
        self.assertLess(mean_theta(empty), 0.05)
        self.assertLess(mean_theta(empty), start)
        self.assertGreater(mean_theta(found), mean_theta(empty))

    def test_floor_views_lower_rho_and_desk_views_raise_it(self):
        mean_rho = lambda ell: float(self.posterior(ell)[0] @ self.rho)
        start = mean_rho(np.ones((143, 6)))
        floor = np.ones((143, 6)); floor[:60] = self.lik(False, O_NOTHING)
        desks = floor.copy(); desks[60:75] = self.lik(True, O_NOTHING)
        self.assertLess(mean_rho(floor), start)
        self.assertGreater(mean_rho(desks), mean_rho(floor))

    def test_unseen_cells_contribute_nothing_to_the_evidence(self):
        coef = evidence_coefficients(np.ones((10, 6)), *TERMS)
        np.testing.assert_allclose(log_evidence(coef, self.rho, self.theta), 0.0, atol=1e-12)

    def test_grid_resolution_does_not_matter(self):
        ell = np.ones((143, 6)); ell[:80] = self.lik(False, O_NOTHING); ell[80:86] = self.lik(True, O_NOTHING, 0.5, 0.9)
        ell[86:90] = self.lik(True, O_BOTTLE)
        out = []
        for n in (128, 256):
            rho, theta = feasible_grid(p_on_table=PI, n_rho=n, n_theta=n)
            w, q = self.posterior(ell, rho, theta)
            out.append((float(w @ rho), float(w @ theta), q))
        self.assertAlmostEqual(out[0][0], out[1][0], delta=2e-3)
        self.assertAlmostEqual(out[0][1], out[1][1], delta=2e-3)
        # 128 vs 256 points per side: beliefs within 0.5 percentage points.
        np.testing.assert_allclose(out[0][2], out[1][2], atol=5e-3)

    def test_floor_only_grid_puts_no_mass_on_tables(self):
        terms = joint_prior_terms(p_on_table=0.0)
        rho, theta = floor_only_grid(n_theta=128)
        np.testing.assert_array_equal(rho, 0.0)
        prior = np.array([prior_at(r, t, terms) for r, t in zip(rho, theta)])
        np.testing.assert_array_equal(prior[:, 0:3], 0.0)
        self.assertTrue(np.all(prior[:, 3:] > 0))
        ell = np.ones((143, 6)); ell[:100] = self.lik(False, O_NOTHING)
        w, q = self.posterior(ell, rho, theta, terms)
        np.testing.assert_array_equal(q[:, 0:3], 0.0)
        self.assertLess(float(w @ theta), 0.05)


if __name__ == "__main__":
    unittest.main()
