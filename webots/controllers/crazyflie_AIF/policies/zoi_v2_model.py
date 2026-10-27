# policies/zoi_v2_model.py
"""Per-cell probabilistic model used by aif_zoi_v2.

Pure numpy, no geometry or controller state, so the same functions serve the
belief update and the planner's expected information gain.

State per cell: the surface and the content, 6 states
{table, floor} x {empty, bottle, can}, index surface * 3 + content. A table cell
is a cell whose centre is on a desk top.

Observations per visible cell per processed frame, independent given the state:

- target observation: nothing, "bottle" box or "can" box. With visibility v,
  recall r, class accuracy a, false alarm f and p = max(f, v * r):

                        empty     bottle      can
        nothing         1 - f     1 - p       1 - p
        "bottle" box    f / 2     p * a       p * (1 - a)
        "can" box       f / 2     p * (1 - a) p * a

  v is the probability that a target present in the cell is inside the image:
  the cell's coverage by the camera footprint at table height (w_T) in the table
  states and at floor height (w_F) in the floor states. A target is a small
  object, in view with probability w.

- desk observation: a desk box covers the cell or not. With the desk in view
  (w_T >= update gate), P(covered | table) = d (desk recall) and
  P(covered | floor) = g (desk false alarm); otherwise it carries no
  information. A desk is an extended object: if any part of a desk cell is in
  view at table height, the desk is in the image, so its evidence does not
  scale with w_T (the offline calibration shows a flat covered rate above the
  gate).

The joint likelihood has 6 outcomes, index desk * 3 + target.

Prior: the share of desk cells rho and the fraction of cells holding a target
theta are unknown and learned in flight (last section); only the share of
targets on desks pi is given.
"""

from __future__ import annotations

import numpy as np

S_EMPTY = 0
S_BOTTLE = 1
S_CAN = 2
STATE_NAMES = ["empty", "target_bottle", "target_can"]

O_NOTHING = 0
O_BOTTLE = 1
O_CAN = 2
OBS_NAMES = ["nothing", "target_bottle", "target_can"]


def likelihood_matrix(visibility: float, *, recall: float, class_accuracy: float, false_alarm: float) -> np.ndarray:
    """A[o, s] = P(observation o | state s) for a cell with the given visibility."""
    v = float(np.clip(visibility, 0.0, 1.0))
    r = float(recall)
    a = float(class_accuracy)
    f = float(false_alarm)
    p = max(f, v * r)

    A = np.empty((3, 3), dtype=np.float64)
    A[O_NOTHING, :] = [1.0 - f, 1.0 - p, 1.0 - p]
    A[O_BOTTLE, :] = [0.5 * f, p * a, p * (1.0 - a)]
    A[O_CAN, :] = [0.5 * f, p * (1.0 - a), p * a]
    return A


def predictive(q: np.ndarray, A: np.ndarray) -> np.ndarray:
    """P(o) = sum_s A[o, s] q(s)."""
    return A @ np.asarray(q, dtype=np.float64)


def expected_information_gain(q: np.ndarray, A: np.ndarray) -> float:
    """Expected KL(posterior || prior) over the predicted observation."""
    q = np.asarray(q, dtype=np.float64)
    z = float(q.sum())
    if z <= 0.0:
        return 0.0
    q = q / z
    p_o = A @ q
    ig = 0.0
    for o in range(A.shape[0]):
        po = float(p_o[o])
        if po <= 1e-12:
            continue
        post = A[o, :] * q / po
        ig += po * float(np.sum(post * (np.log(post + 1e-12) - np.log(q + 1e-12))))
    return float(ig)


# ---------------------------------------------------------------------------
# Joint surface model
# ---------------------------------------------------------------------------

SURF_TABLE = 0
SURF_FLOOR = 1
SURFACE_NAMES = ["table", "floor"]
N_JOINT = 6
JOINT_STATE_NAMES = [f"{surface}_{state}" for surface in SURFACE_NAMES for state in STATE_NAMES]

D_NONE = 0
D_COVERED = 1


def joint_index(surface: int, content: int) -> int:
    return int(surface) * 3 + int(content)


def joint_obs_index(desk_covered: bool, target_obs: int) -> int:
    return (D_COVERED if desk_covered else D_NONE) * 3 + int(target_obs)


def desk_likelihood(desk_in_view: bool, *, desk_recall: float, desk_false_alarm: float) -> np.ndarray:
    """D[o_d, s] = P(desk observation | joint state), o_d in (none, covered).

    With the desk out of view the observation is always "none", for every state.
    """
    if desk_in_view:
        covered = np.array([float(desk_recall)] * 3 + [float(desk_false_alarm)] * 3)
    else:
        covered = np.zeros(N_JOINT)
    return np.vstack([1.0 - covered, covered])


def joint_target_likelihood(
    w_table: float, w_floor: float, *, recall: float, class_accuracy: float, false_alarm: float
) -> np.ndarray:
    """T[o_t, s] = P(target observation | joint state): table-plane visibility for
    the table states, floor-plane visibility for the floor states."""
    kw = {"recall": recall, "class_accuracy": class_accuracy, "false_alarm": false_alarm}
    return np.hstack([likelihood_matrix(w_table, **kw), likelihood_matrix(w_floor, **kw)])


def joint_likelihood(
    w_table: float,
    w_floor: float,
    *,
    desk_in_view: bool,
    recall: float,
    class_accuracy: float,
    false_alarm: float,
    desk_recall: float,
    desk_false_alarm: float,
) -> np.ndarray:
    """L[desk * 3 + target, s] = P(desk obs | s) * P(target obs | s)."""
    D = desk_likelihood(desk_in_view, desk_recall=desk_recall, desk_false_alarm=desk_false_alarm)
    T = joint_target_likelihood(
        w_table, w_floor, recall=recall, class_accuracy=class_accuracy, false_alarm=false_alarm)
    return (D[:, None, :] * T[None, :, :]).reshape(N_JOINT, N_JOINT)


# Batched versions for the vectorized planner (policies/aif_zoi_v2_vectorized.py):
# the same formulas on N cells at once, element by element.


def likelihood_matrix_batch(
    visibility: np.ndarray, *, recall: float, class_accuracy: float, false_alarm: float
) -> np.ndarray:
    """(N, 3, 3): likelihood_matrix for each of N visibilities."""
    v = np.clip(np.asarray(visibility, dtype=np.float64), 0.0, 1.0)
    r, a, f = float(recall), float(class_accuracy), float(false_alarm)
    p = np.maximum(f, v * r)
    A = np.empty(v.shape + (3, 3), dtype=np.float64)
    A[..., O_NOTHING, 0] = 1.0 - f
    A[..., O_NOTHING, 1] = 1.0 - p
    A[..., O_NOTHING, 2] = 1.0 - p
    A[..., O_BOTTLE, 0] = 0.5 * f
    A[..., O_BOTTLE, 1] = p * a
    A[..., O_BOTTLE, 2] = p * (1.0 - a)
    A[..., O_CAN, 0] = 0.5 * f
    A[..., O_CAN, 1] = p * (1.0 - a)
    A[..., O_CAN, 2] = p * a
    return A


def joint_likelihood_batch(
    w_table: np.ndarray,
    w_floor: np.ndarray,
    *,
    desk_in_view: np.ndarray,
    recall: float,
    class_accuracy: float,
    false_alarm: float,
    desk_recall: float,
    desk_false_alarm: float,
) -> np.ndarray:
    """(N, 6, 6): joint_likelihood for each of N cells."""
    kw = {"recall": recall, "class_accuracy": class_accuracy, "false_alarm": false_alarm}
    w_table = np.asarray(w_table, dtype=np.float64)
    n = w_table.shape[0]
    T = np.empty((n, 3, N_JOINT), dtype=np.float64)
    T[:, :, 0:3] = likelihood_matrix_batch(w_table, **kw)
    T[:, :, 3:6] = likelihood_matrix_batch(w_floor, **kw)
    in_view = np.asarray(desk_in_view, dtype=bool)
    covered = np.zeros((n, N_JOINT), dtype=np.float64)
    covered[in_view] = np.array([float(desk_recall)] * 3 + [float(desk_false_alarm)] * 3)
    D = np.stack([1.0 - covered, covered], axis=1)                  # (N, 2, 6)
    return (D[:, :, None, :] * T[:, None, :, :]).reshape(n, N_JOINT, N_JOINT)


def expected_information_gain_batch(q: np.ndarray, A: np.ndarray) -> np.ndarray:
    """(N,): expected_information_gain for N beliefs q (N, S) and likelihoods A (N, O, S)."""
    q = np.asarray(q, dtype=np.float64)
    z = q.sum(axis=1)
    ok_z = z > 0.0
    q = q / np.where(ok_z, z, 1.0)[:, None]
    p_o = np.matmul(A, q[:, :, None])[:, :, 0]  # same kernel as A @ q in expected_information_gain
    log_q = np.log(q + 1e-12)
    ig = np.zeros(q.shape[0], dtype=np.float64)
    for o in range(A.shape[1]):
        po = p_o[:, o]
        ok = po > 1e-12
        post = A[:, o, :] * q / np.where(ok, po, 1.0)[:, None]
        ig += np.where(ok, po * np.sum(post * (np.log(post + 1e-12) - log_q), axis=1), 0.0)
    return np.where(ok_z, ig, 0.0)


def content_marginal(q_joint: np.ndarray) -> np.ndarray:
    """P(empty), P(bottle), P(can) summed over the surface."""
    q = np.asarray(q_joint, dtype=np.float64)
    return q[0:3] + q[3:6]


def table_probability(q_joint: np.ndarray) -> float:
    """P(table) for one cell."""
    return float(np.sum(np.asarray(q_joint, dtype=np.float64)[0:3]))


# ---------------------------------------------------------------------------
# Prior with rho and theta learned in flight
# ---------------------------------------------------------------------------
#
# rho is the share of desk cells and theta the fraction of cells holding a
# target (N / M if the number of targets N were known). The prior of a cell is
# linear in each:
#     P(table, target) = theta * pi            (split evenly bottle / can)
#     P(floor, target) = theta * (1 - pi)
#     P(table, empty)  = rho - theta * pi
#     P(floor, empty)  = 1 - rho - theta * (1 - pi)
# i.e. prior = base + rho * d_rho + theta * d_theta. All entries are positive on
# the feasible region theta * pi < rho < 1 - theta * (1 - pi); the prior over
# (rho, theta) is uniform on it. Cells are independent given (rho, theta). Each
# cell keeps ell_j, the product of the likelihood rows it has received, and its
# evidence is linear too:
#     Z_j = a_j + rho * b_j + theta * c_j,  (a, b, c) = ell_j . (base, d_rho, d_theta)
# The posterior over (rho, theta) is proportional to the product of the Z_j, and
# each cell's belief is the exact average of its posterior over it, which
# reduces to three sums per cell:
#     q_j(s) = ell_j(s) * (base_s * S0 + d_rho_s * S1 + d_theta_s * S2),
#     S0 = sum_g w_g / Z_j(g),  S1 = sum_g w_g rho_g / Z_j(g),  S2 = sum_g w_g theta_g / Z_j(g).
# Without desks in the environment, rho = 0 and pi = 0 and only theta is learned.


def joint_prior_terms(*, p_on_table: float) -> tuple:
    """(base, d_rho, d_theta) with prior(rho, theta) = base + rho * d_rho + theta * d_theta."""
    pi = float(p_on_table)
    base = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    d_rho = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    d_theta = np.array([-pi, 0.5 * pi, 0.5 * pi, -(1.0 - pi), 0.5 * (1.0 - pi), 0.5 * (1.0 - pi)])
    return base, d_rho, d_theta


def feasible_grid(*, p_on_table: float, n_rho: int, n_theta: int) -> tuple:
    """(rho, theta) of the grid points (midpoints of equal bins of (0, 1) x (0, 1))
    where every prior entry is positive. The grid sizes are a numerical resolution."""
    pi = float(p_on_table)
    rho = (np.arange(int(n_rho)) + 0.5) / int(n_rho)
    theta = (np.arange(int(n_theta)) + 0.5) / int(n_theta)
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    ok = (rr > tt * pi) & (rr < 1.0 - tt * (1.0 - pi))
    return rr[ok], tt[ok]


def floor_only_grid(*, n_theta: int) -> tuple:
    """(rho, theta) grid for an environment without desks: rho = 0, theta on the
    midpoints of equal bins of (0, 1). Use with joint_prior_terms(p_on_table=0)."""
    theta = (np.arange(int(n_theta)) + 0.5) / int(n_theta)
    return np.zeros_like(theta), theta


def evidence_coefficients(ell: np.ndarray, base: np.ndarray, d_rho: np.ndarray, d_theta: np.ndarray) -> np.ndarray:
    """(cells, 3): Z_j(rho, theta) = a + rho * b + theta * c for each cell's record."""
    ell = np.asarray(ell, dtype=np.float64).reshape(-1, N_JOINT)
    return np.stack([ell @ base, ell @ d_rho, ell @ d_theta], axis=1)


def log_evidence(coef: np.ndarray, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Sum over the given cells of log Z_j at every grid point."""
    coef = np.asarray(coef, dtype=np.float64).reshape(-1, 3)
    if coef.shape[0] == 0:
        return np.zeros_like(rho)
    z = coef[:, 0:1] + coef[:, 1:2] * rho[None, :] + coef[:, 2:3] * theta[None, :]
    return np.log(z).sum(axis=0)


def grid_weights(log_ev: np.ndarray) -> np.ndarray:
    """Posterior weights of the grid points from a uniform prior."""
    w = np.exp(log_ev - log_ev.max())
    return w / w.sum()


def marginal_beliefs_2d(
    ell: np.ndarray,
    coef: np.ndarray,
    terms: tuple,
    rho: np.ndarray,
    theta: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Each cell's belief averaged over p(rho, theta | data). ell: (cells, 6) -> (cells, 6)."""
    base, d_rho, d_theta = terms
    ell = np.asarray(ell, dtype=np.float64).reshape(-1, N_JOINT)
    coef = np.asarray(coef, dtype=np.float64).reshape(-1, 3)
    inv_z = 1.0 / (coef[:, 0:1] + coef[:, 1:2] * rho[None, :] + coef[:, 2:3] * theta[None, :])
    wz = inv_z * weights[None, :]
    s0, s1, s2 = wz.sum(axis=1), wz @ rho, wz @ theta
    return ell * (s0[:, None] * base[None, :] + s1[:, None] * d_rho[None, :] + s2[:, None] * d_theta[None, :])
