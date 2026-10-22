# policies/aif_zoi_v2_joint.py

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from policies.aif_zoi_v2 import DESK_CLASS_NAMES, AIFZOIV2Policy, Cell
from policies.zoi_v2_model import (
    JOINT_STATE_NAMES,
    O_BOTTLE,
    O_CAN,
    bayes_update,
    content_marginal,
    expected_information_gain,
    joint_likelihood,
    joint_obs_index,
    joint_prior,
    joint_prior_base_and_slope,
    marginal_beliefs,
    predictive,
    rho_grid,
    rho_posterior,
    rho_support,
)

# Resolution of the grid over rho in learned mode (numerical, not a model choice;
# tests check that results do not depend on it).
RHO_GRID_SIZE = 256


class AIFZOIV2JointPolicy(AIFZOIV2Policy):
    """aif_zoi_v2 with the cell surface in the belief state (roadmap 3.1).

    Each cell holds a belief over {table, floor} x {empty, bottle, can}
    (policies/zoi_v2_model.py, joint section) instead of a hard cell type from a
    table map. Every processed frame, each visible cell gets a desk observation
    (a desk box covers it or not, informative when its table-plane coverage
    passes the update gate) and the usual target observation, and the belief is
    updated by Bayes with the joint likelihood. There is no table map, no floor
    rule and no prior reweighting. q_s holds the content marginal, so the
    planner, payloads and snapshots of the base policy work unchanged; the
    planner's information gain and classification term use the joint likelihood.

    The controller's table map (grid["table_cells"]) is still passed to the
    planner, only to predict which cells the controller's visible mask will
    update, as in the base policy.

    With params.rho_model == "learned", the share of desk cells rho is not given:
    it has a uniform prior over its feasible range and is inferred every frame
    from all the evidence (zoi_v2_model.py, learned-rho section). Each cell keeps
    the product of the likelihood rows it has received (ell), and q_joint is the
    exact average of its posterior over p(rho | data). The prior then needs only
    the number of targets and the share on tables.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_joint: Optional[np.ndarray] = None
        self.rho_learned = self.params.rho_model == "learned" and self.table_context_enabled
        self._ell: Optional[np.ndarray] = None
        self._rho_base = self._rho_slope = self._rho_grid = self._rho_weights = None

    def parameters(self) -> dict:
        out = super().parameters()
        out["observation_model"] = {
            "surface": "joint belief over {table, floor} x {empty, bottle, can}",
            "visibility": "table-plane coverage for table states, floor-plane coverage for floor states",
            "desk_evidence": "desk box covers the cell; informative when table-plane coverage >= update gate",
            "rho": "learned in flight, uniform prior on its feasible range" if self.rho_learned else "fixed",
        }
        if self.rho_learned:
            out["observation_model"]["rho_grid_size"] = RHO_GRID_SIZE
        return out

    def diagnostics(self) -> dict:
        """Per-frame values for the perception log."""
        if not self.rho_learned or self._rho_weights is None:
            return {}
        mean, sd = self.rho_estimate()
        return {"rho_mean": round(mean, 5), "rho_sd": round(sd, 5)}

    def rho_estimate(self) -> tuple:
        """(mean, sd) of p(rho | data)."""
        w, g = self._rho_weights, self._rho_grid
        mean = float(w @ g)
        return mean, float(np.sqrt(max(0.0, w @ (g - mean) ** 2)))

    # ------------------------------------------------------------------
    # Belief
    # ------------------------------------------------------------------

    def _ensure_initialized(self, obs: dict) -> None:
        if self.grid_spec is not None:
            return
        super()._ensure_initialized(obs)
        rows, cols = self.grid_spec.rows, self.grid_spec.cols
        prior = joint_prior(
            lambda_table=self.params.lambda_table,
            lambda_floor=self.params.lambda_floor,
            rho=self.params.rho if self.table_context_enabled else 0.0,
        )
        self.q_joint = np.tile(prior.reshape(1, 1, -1), (rows, cols, 1))
        self.q_s = np.tile(content_marginal(prior).reshape(1, 1, 3), (rows, cols, 1))
        if self.rho_learned:
            scen = {"n_targets": self.params.n_targets, "p_on_table": self.params.p_target_on_table,
                    "n_cells": rows * cols}
            self._rho_base, self._rho_slope = joint_prior_base_and_slope(**scen)
            self._rho_grid = rho_grid(*rho_support(**scen), RHO_GRID_SIZE)
            self._ell = np.ones((rows, cols, 6))
            self._refresh_learned_beliefs()

    def _refresh_learned_beliefs(self) -> None:
        """Recompute p(rho | data) and every cell's belief from ell."""
        rows, cols = self._ell.shape[0], self._ell.shape[1]
        ell = self._ell.reshape(-1, 6)
        self._rho_weights = rho_posterior(ell, self._rho_base, self._rho_slope, self._rho_grid)
        q = marginal_beliefs(ell, self._rho_base, self._rho_slope, self._rho_grid, self._rho_weights)
        self.q_joint = q.reshape(rows, cols, 6)
        self.q_s = (self.q_joint[:, :, 0:3] + self.q_joint[:, :, 3:6]).copy()

    def _update_cell_types(self, *, table_cells, pose) -> None:
        # The surface is part of the belief; cells keep no hard type.
        return None

    def _desk_in_view(self, w_table: float) -> bool:
        return w_table > 0.0 and w_table >= self._coverage_threshold

    def _joint_likelihood(self, w_table: float, w_floor: float) -> np.ndarray:
        p = self.params
        return joint_likelihood(
            w_table,
            w_floor,
            desk_in_view=self._desk_in_view(w_table) and self.table_context_enabled,
            recall=p.recall,
            class_accuracy=p.class_accuracy,
            false_alarm=p.false_alarm,
            desk_recall=p.desk_recall,
            desk_false_alarm=p.desk_false_alarm,
        )

    def _desk_covered_cells(self, grid: dict) -> set[Cell]:
        """Cells covered by this frame's desk boxes (the controller's projection)."""
        out: set[Cell] = set()
        for event in grid.get("detections_all", []) or []:
            if self._event_class_name(event) not in DESK_CLASS_NAMES:
                continue
            for cell in event.get("covered_cells", []) or []:
                parsed = self._parse_cell(cell)
                if parsed is not None:
                    out.add(parsed)
        return out

    def _update_beliefs_from_observation(
        self,
        *,
        grid: dict,
        visible_mask: Dict[Cell, bool],
        coverages: Dict[Cell, float],
    ) -> None:
        """One joint Bayesian update per visible cell: desk and target observation."""
        assert self.q_joint is not None and self.q_s is not None
        rows, cols = self.q_s.shape[0], self.q_s.shape[1]

        target_events_by_cell = self._target_events_by_cell(grid)
        desk_cells = self._desk_covered_cells(grid)

        floor_plane = grid.get("floor_plane_coverages", None)
        table_plane = grid.get("table_plane_coverages", None)
        if floor_plane is None or table_plane is None:
            floor_plane = table_plane = coverages
        else:
            floor_plane = self._normalize_float_cell_map(floor_plane)
            table_plane = self._normalize_float_cell_map(table_plane)

        visible_cells = set()
        for (cx, cy), vis in visible_mask.items():
            if not vis or not (0 <= int(cx) < cols and 0 <= int(cy) < rows):
                continue
            cell = (int(cx), int(cy))
            visible_cells.add(cell)
            w_table = float(table_plane.get(cell, 0.0))
            w_floor = float(floor_plane.get(cell, 0.0))
            covered = cell in desk_cells and self._desk_in_view(w_table) and self.table_context_enabled
            obs_index = joint_obs_index(covered, self._cell_observation(target_events_by_cell.get(cell, [])))
            if self.rho_learned:
                # Accumulate the evidence; beliefs are refreshed once per frame below.
                row = self._joint_likelihood(w_table, w_floor)[obs_index]
                ell = self._ell[cell[1], cell[0], :] * row
                self._ell[cell[1], cell[0], :] = ell / ell.max()
                continue
            q = bayes_update(self.q_joint[cell[1], cell[0], :], self._joint_likelihood(w_table, w_floor), obs_index)
            self.q_joint[cell[1], cell[0], :] = q
            self.q_s[cell[1], cell[0], :] = content_marginal(q)

        if self.rho_learned and visible_cells:
            self._refresh_learned_beliefs()

        for cell, events in target_events_by_cell.items():
            if cell not in visible_cells:
                self._dropped_target_events += len(events)

    def table_probability_map(self) -> np.ndarray:
        assert self.q_joint is not None
        return self.q_joint[:, :, 0:3].sum(axis=2)

    # ------------------------------------------------------------------
    # Planner hooks
    # ------------------------------------------------------------------

    def _cell_information_gain(self, cell: Cell, w_table: float, w_floor: float) -> float:
        assert self.q_joint is not None
        q = self.q_joint[int(cell[1]), int(cell[0]), :]
        return expected_information_gain(q, self._joint_likelihood(w_table, w_floor))

    def _center_utility(self, cell: Cell, w_table: float, w_floor: float, *, is_table: bool) -> float:
        assert self.q_joint is not None
        q = self.q_joint[int(cell[1]), int(cell[0]), :]
        q = q / max(float(q.sum()), 1e-300)
        p_o = predictive(q, self._joint_likelihood(w_table, w_floor))
        # P(a class box), over both desk outcomes.
        p_class_obs = float(p_o[O_BOTTLE] + p_o[O_CAN] + p_o[3 + O_BOTTLE] + p_o[3 + O_CAN])
        need = self._classification_need(content_marginal(q))
        return float(self.classification_preference_strength * need * p_class_obs)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _save_snapshot_npz(self, **kwargs) -> str:
        path = super()._save_snapshot_npz(**kwargs)
        if self.q_joint is not None:
            np.savez_compressed(
                path[:-4] + "_joint.npz",
                q_joint=self.q_joint.astype(np.float32),
                p_table=self.table_probability_map().astype(np.float32),
                joint_state_names=np.array(JOINT_STATE_NAMES, dtype=object),
            )
        return path

