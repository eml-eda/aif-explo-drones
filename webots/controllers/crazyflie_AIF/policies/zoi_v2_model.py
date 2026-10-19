# policies/zoi_v2_model.py
"""Per-cell probabilistic model used by aif_zoi_v2.

Pure numpy, no geometry or controller state, so the same functions serve the
belief update and the planner's expected information gain.

States per cell:       empty, bottle, can.
Observations per cell: nothing, "bottle" box, "can" box (one per cell per frame).

Prior: P(target in cell) depends on what is known about the cell surface

    table cell        lambda_table
    floor cell        lambda_floor
    not yet observed  rho * lambda_table + (1 - rho) * lambda_floor

and the target mass is split evenly between bottle and can.

Likelihood, with visibility v, recall r, class accuracy a, false alarm f and
p = max(f, v * r):

                    empty     bottle      can
    nothing         1 - f     1 - p       1 - p
    "bottle" box    f / 2     p * a       p * (1 - a)
    "can" box       f / 2     p * (1 - a) p * a

Visibility v is the probability that a target present in the cell is inside
the camera image. With the target uniformly distributed within its cell, v at
one surface is the cell's area coverage by the camera footprint at that
surface's height. The surface is mixed with the probability that a present
target is on a table, pi_c:

    v = pi_c * v_table + (1 - pi_c) * v_floor

    table cell        pi_c = 1
    floor cell        pi_c = 0
    not yet observed  pi_c = rho * lambda_table / (rho * lambda_table + (1 - rho) * lambda_floor)

P(o | target) is linear in v, so mixing v is the same as mixing the per-surface
likelihoods (exactly whenever v_table * r and v_floor * r are both >= f).

Joint surface model (roadmap 3.1). Instead of a hard cell type, the surface is
part of the state: 6 states {table, floor} x {empty, bottle, can}, index
surface * 3 + content. A table cell is a cell whose centre is on a desk top.

    prior   P(table) = rho; content | table ~ (1 - lambda_table, lambda_table/2, lambda_table/2),
            content | floor ~ (1 - lambda_floor, lambda_floor/2, lambda_floor/2)

Each visible cell gets two observations per processed frame, independent given
the state: a desk observation (a desk box covers the cell or not) and the
target observation above. With w_T and w_F the cell's coverage at table and at
floor height, desk recall d and desk false alarm g:

    desk in view (w_T >= update gate):
        P(desk box covers the cell | table) = d
        P(desk box covers the cell | floor) = g
    otherwise the desk observation carries no information.

A desk is an extended object: if any part of a desk cell is in view at table
height, the desk is in the image, so its evidence does not scale with w_T (the
offline calibration shows a flat covered rate above the gate). A target is a
small object, in view with probability w. The target likelihood uses v = w_T in
the table states and v = w_F in the floor states. The joint likelihood has 6
outcomes, index desk * 3 + target.
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

# Cell surface types used to select the prior.
CELL_UNOBSERVED = 0
CELL_FLOOR = 1
CELL_TABLE = 2
CELL_TYPE_NAMES = ["unobserved", "floor", "table"]


def prior_present(cell_type: int, *, lambda_table: float, lambda_floor: float, rho: float) -> float:
    """P(target in cell) for a cell of the given surface type."""
    cell_type = int(cell_type)
    if cell_type == CELL_TABLE:
        return float(lambda_table)
    if cell_type == CELL_FLOOR:
        return float(lambda_floor)
    if cell_type == CELL_UNOBSERVED:
        return float(rho * lambda_table + (1.0 - rho) * lambda_floor)
    raise ValueError(f"Unknown cell type {cell_type}")


def prior_vector(cell_type: int, *, lambda_table: float, lambda_floor: float, rho: float) -> np.ndarray:
    """Prior over (empty, bottle, can) for a cell of the given surface type."""
    lam = prior_present(cell_type, lambda_table=lambda_table, lambda_floor=lambda_floor, rho=rho)
    return np.array([1.0 - lam, 0.5 * lam, 0.5 * lam], dtype=np.float64)


def table_probability_given_target(
    cell_type: int, *, lambda_table: float, lambda_floor: float, rho: float
) -> float:
    """pi_c = P(target on a table | target present in the cell, cell type)."""
    cell_type = int(cell_type)
    if cell_type == CELL_TABLE:
        return 1.0
    if cell_type == CELL_FLOOR:
        return 0.0
    if cell_type == CELL_UNOBSERVED:
        on_table = rho * lambda_table
        total = on_table + (1.0 - rho) * lambda_floor
        return float(on_table / total) if total > 0.0 else 0.0
    raise ValueError(f"Unknown cell type {cell_type}")


def object_visibility(
    cell_type: int,
    v_table: float,
    v_floor: float,
    *,
    lambda_table: float,
    lambda_floor: float,
    rho: float,
) -> float:
    """v = P(target in image | target present), mixing table and floor planes."""
    pi_c = table_probability_given_target(
        cell_type, lambda_table=lambda_table, lambda_floor=lambda_floor, rho=rho)
    return float(pi_c * float(v_table) + (1.0 - pi_c) * float(v_floor))


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


def bayes_update(q: np.ndarray, A: np.ndarray, obs_index: int) -> np.ndarray:
    """q_post proportional to A[o, :] * q. Returns q unchanged if the evidence is 0."""
    q = np.asarray(q, dtype=np.float64)
    post = A[int(obs_index), :] * q
    z = float(post.sum())
    if z <= 0.0:
        return q.copy()
    return post / z


def reweight(q: np.ndarray, prior_old: np.ndarray, prior_new: np.ndarray) -> np.ndarray:
    """Swap the prior under a posterior while keeping its evidence.

    If q is proportional to L * prior_old, then q * prior_new / prior_old is
    proportional to L * prior_new.
    """
    q = np.asarray(q, dtype=np.float64)
    prior_old = np.asarray(prior_old, dtype=np.float64)
    prior_new = np.asarray(prior_new, dtype=np.float64)
    ratio = np.divide(prior_new, prior_old, out=np.zeros_like(prior_new), where=prior_old > 0.0)
    post = q * ratio
    z = float(post.sum())
    if z <= 0.0:
        return q.copy()
    return post / z


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


def joint_prior(*, lambda_table: float, lambda_floor: float, rho: float) -> np.ndarray:
    """Prior over the 6 joint states."""
    table = float(rho) * np.array([1.0 - lambda_table, 0.5 * lambda_table, 0.5 * lambda_table])
    floor = (1.0 - float(rho)) * np.array([1.0 - lambda_floor, 0.5 * lambda_floor, 0.5 * lambda_floor])
    return np.concatenate([table, floor]).astype(np.float64)


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


def content_marginal(q_joint: np.ndarray) -> np.ndarray:
    """P(empty), P(bottle), P(can) summed over the surface."""
    q = np.asarray(q_joint, dtype=np.float64)
    return q[0:3] + q[3:6]


def table_probability(q_joint: np.ndarray) -> float:
    """P(table) for one cell."""
    return float(np.sum(np.asarray(q_joint, dtype=np.float64)[0:3]))
