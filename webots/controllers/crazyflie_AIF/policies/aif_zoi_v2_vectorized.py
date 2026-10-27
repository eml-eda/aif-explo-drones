# policies/aif_zoi_v2_vectorized.py
"""aif_zoi_v2 with the planner computed on arrays instead of a Python loop.

Same model, belief and score as AIFZOIV2Policy (policies/aif_zoi_v2.py): every
(cell, yaw) candidate is ranked by its expected information gain over the cells
the controller would update from there, divided by the flight time, with ties to
the shorter flight. Only the computation changes:

- the candidates' view geometry (which cells each candidate sees, with their
  floor- and table-plane coverage) is flattened once into arrays, one entry per
  (candidate, seen cell), kept in the order the loop planner visits them;
- each replan builds the likelihoods of all entries and their information gains
  in a few array operations, sums them per candidate in the loop's order, and
  computes every candidate's flight time at once (NavigatorTimeModel.time_to_batch).

Every step uses the same floating-point operations in the same order as the loop
planner, so information gains, flight times and decisions are identical (tests
and tools check it; numpy kernels could in principle round differently on other
hardware, so the decisions are what matter).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from policies.aif_zoi_v2 import AIFZOIV2Policy, Cell
from policies.zoi_v2_model import expected_information_gain_batch, joint_likelihood_batch


class AIFZOIV2VectorizedPolicy(AIFZOIV2Policy):
    """AIFZOIV2Policy with the vectorized planner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Candidates k = cell_index * n_yaw + yaw_index, in the loop planner's order.
        self._cand_cells: Optional[list] = None
        self._cand_x = self._cand_y = self._cand_yaw = None
        # One entry per (candidate, seen cell): candidate, position within the
        # candidate's list, flat cell index (cy * cols + cx), plane coverages.
        self._ent_cand = self._ent_pos = self._ent_cell = None
        self._ent_w_floor = self._ent_w_table = None
        self._max_entries = 0

    def parameters(self) -> dict:
        out = super().parameters()
        out["planning"]["implementation"] = "vectorized"
        return out

    def _ensure_initialized(self, obs: dict) -> None:
        if self.grid_spec is not None:
            return
        super()._ensure_initialized(obs)
        self._build_candidate_arrays()

    def _build_candidate_arrays(self) -> None:
        assert self.grid_spec is not None
        cols = self.grid_spec.cols
        cells, xs, ys, yaws = [], [], [], []
        ent_cand, ent_pos, ent_cell, ent_wf, ent_wt = [], [], [], [], []
        for cell in self._all_cells():
            tx, ty = self._cell_center_xy(cell[0], cell[1])
            for yaw_idx, yaw in enumerate(self.yaw_options):
                k = len(cells)
                cells.append(cell)
                xs.append(tx)
                ys.append(ty)
                yaws.append(yaw)
                floor_cov, table_cov = self._cached_candidate_coverages_rc(
                    target_cell=cell, target_yaw_index=yaw_idx)
                # Same iteration order as _merged_candidate_coverages_rc.
                all_rc = set(floor_cov.keys()) | set(table_cov.keys())
                for pos, (r, c) in enumerate(all_rc):
                    ent_cand.append(k)
                    ent_pos.append(pos)
                    ent_cell.append(int(r) * cols + int(c))
                    ent_wf.append(float(floor_cov.get((int(r), int(c)), 0.0)))
                    ent_wt.append(float(table_cov.get((int(r), int(c)), 0.0)))
        self._cand_cells = cells
        self._cand_x, self._cand_y, self._cand_yaw = np.array(xs), np.array(ys), np.array(yaws)
        self._ent_cand = np.array(ent_cand, dtype=np.int64)
        self._ent_pos = np.array(ent_pos, dtype=np.int64)
        self._ent_cell = np.array(ent_cell, dtype=np.int64)
        self._ent_w_floor = np.array(ent_wf)
        self._ent_w_table = np.array(ent_wt)
        self._max_entries = int(self._ent_pos.max()) + 1 if self._ent_pos.size else 0

    def _candidate_information_gains(self, table_cells: set[Cell]) -> np.ndarray:
        """(K,): every candidate's expected information gain, as _candidate_information_gain."""
        assert self.grid_spec is not None and self.q_joint is not None
        cols = self.grid_spec.cols
        rows = self.grid_spec.rows
        is_table = np.zeros(rows * cols, dtype=bool)
        for cx, cy in table_cells:
            if 0 <= int(cx) < cols and 0 <= int(cy) < rows:
                is_table[int(cy) * cols + int(cx)] = True
        # Coverage the controller's visible mask would use: table plane on its table-map cells.
        merged = np.where(is_table[self._ent_cell], self._ent_w_table, self._ent_w_floor)
        keep = (merged > 0.0) & (merged >= self._coverage_threshold)
        idx = np.flatnonzero(keep)

        w_t, w_f = self._ent_w_table[idx], self._ent_w_floor[idx]
        p = self.params
        lik = joint_likelihood_batch(
            w_t, w_f,
            desk_in_view=(w_t > 0.0) & (w_t >= self._coverage_threshold) & self.table_context_enabled,
            recall=p.recall, class_accuracy=p.class_accuracy, false_alarm=p.false_alarm,
            desk_recall=p.desk_recall, desk_false_alarm=p.desk_false_alarm)
        ig = expected_information_gain_batch(self.q_joint.reshape(-1, 6)[self._ent_cell[idx]], lik)

        # Sum per candidate in the loop's order: position by position, from 0.0.
        padded = np.zeros((len(self._cand_cells), self._max_entries))
        padded[self._ent_cand[idx], self._ent_pos[idx]] = ig
        total = np.zeros(len(self._cand_cells))
        for j in range(self._max_entries):
            total = total + padded[:, j]
        return total

    def _pick_next_action(
        self,
        *,
        current_cell: Cell,
        table_cells: Optional[set[Cell]] = None,
        return_scores: bool = False,
        pose: Optional[dict] = None,
    ):
        """Best (cell, yaw) by IG / T; ties go to the shorter flight, then to the first candidate."""
        assert self.grid_spec is not None
        assert self.q_joint is not None
        table_cells = table_cells or set()

        cx0, cy0 = current_cell
        x0, y0 = self._cell_center_xy(cx0, cy0)
        if pose is None:
            pose = {"x": x0, "y": y0, "yaw": 0.0}

        ig = self._candidate_information_gains(table_cells)
        T, t_turn, t_move = self.time_model.time_to_batch(pose, self._cand_x, self._cand_y, self._cand_yaw)
        rate = ig / T

        best_rate = rate.max()
        tied = np.flatnonzero(rate == best_rate)
        k = int(tied[np.argmin(T[tied])])  # argmin returns the first of equal T
        best = (self._cand_cells[k], float(self._cand_yaw[k]))

        if not return_scores:
            return best
        scores = []
        for i, cell in enumerate(self._cand_cells):
            r = float(rate[i])
            scores.append({
                "cell": [int(cell[0]), int(cell[1])],
                "yaw": float(self._cand_yaw[i]),
                "IG": float(ig[i]),
                "distance": float(math.hypot(self._cand_x[i] - x0, self._cand_y[i] - y0)),
                "T": float(T[i]),
                "T_turn": float(t_turn[i]),
                "T_move": float(t_move[i]),
                "rate": r,
                # Minimised by utils/snapshot_analyzer.py.
                "G": -r,
            })
        return best, scores
