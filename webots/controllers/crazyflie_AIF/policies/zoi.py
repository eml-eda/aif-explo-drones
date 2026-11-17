# policies/zoi.py

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from configuration.zoi_params import DEFAULT_GRID_SIZE, ZOIParams
from env.env_manager import FoVSpec, GridEnvManager, GridSpec
from policies.planner_time import NavigatorTimeModel
from policies.zoi_model import (
    JOINT_STATE_NAMES,
    O_BOTTLE,
    O_CAN,
    O_NOTHING,
    OBS_NAMES,
    S_BOTTLE,
    S_CAN,
    S_EMPTY,
    STATE_NAMES,
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
)
from utils.navigator import Navigator

# Navigator settings.
NAVIGATOR_KWARGS = {
    "max_forward": 0.50,
    "max_sideways": 0.50,
    "max_yaw_rate": 0.60,
    "k_pos": 1.2,
    "k_yaw": 1.5,
    "dist_eps_m": 0.2,
    "yaw_eps_rad": 0.12,
    "min_cell_coverage": 0.60,
}

# Default grid over (rho, theta): points per side over (0, 1), feasible points
# only. A numerical resolution, not a model choice (configuration/zoi_params.py).
# Set by params.grid_size.
RHO_THETA_GRID_SIZE = DEFAULT_GRID_SIZE

Cell = Tuple[int, int]  # policy/report convention: (cx, cy)

TARGET_CLASS_NAMES = {"target_bottle", "target_can"}
DESK_CLASS_NAMES = {"desk"}
CLASS_NAME_TO_OBS = {"target_bottle": O_BOTTLE, "target_can": O_CAN}


class ZOIPolicy:
    """Active-inference search with a calibrated observation model and no weights.

    Belief (policies/zoi_model.py): each cell holds a belief over
    {table, floor} x {empty, bottle, can}. Every processed frame, each visible
    cell gets a desk observation (a desk box covers it or not, informative when
    its table-plane coverage passes the update gate) and a target observation
    (nothing, "bottle" box, "can" box), with the joint likelihood: table-plane
    coverage for the table states, floor-plane coverage for the floor states.
    Detector confidence never enters the likelihood.

    Prior: the share of desk cells rho and the fraction of cells holding a
    target theta are unknown, with a uniform prior over their feasible region,
    and inferred every frame from all the evidence. Each cell keeps the product
    of the likelihood rows it has received (ell), and its belief is the exact
    average of its posterior over p(rho, theta | data). Only the share of targets
    on desks pi is given. Without desks in the environment rho = 0, pi = 0.

    Planner: every (cell, yaw) candidate is ranked by its expected information
    gain over the cells the controller would update from there, divided by the
    flight time to it (policies/planner_time.py, from the navigator's control
    law) plus one detector period. Ties go to the shorter flight. The
    controller's table map (grid["table_cells"]) is used only to predict which
    cells the controller's visible mask will update.

    q_s holds the content marginal of the belief, for logs and snapshots.
    """

    def __init__(
        self,
        debug: Optional[dict] = None,
        *,
        params: Optional[ZOIParams] = None,
        yaw_options: Optional[List[float]] = None,
        replan_cooldown_s: float = 0.4,
        classified_confidence_threshold: float = 0.80,
        # Detector period of the controller (DETECTION_EVERY_S).
        detector_period_s: float = 0.625,
    ):
        self.debug = debug or {}
        self.every_n = int(self.debug.get("every_n", 50))

        self.params = params if params is not None else ZOIParams.calibrated()
        self.table_context_enabled = bool(self.params.table_context_enabled)

        self.replan_cooldown_s = float(replan_cooldown_s)
        # Flight time of a move, derived from the navigator's control law.
        self.time_model = NavigatorTimeModel.from_navigator(NAVIGATOR_KWARGS, detector_period_s)
        # Only for the classified-cells payload of snapshots.
        self.classified_confidence_threshold = float(
            np.clip(classified_confidence_threshold, 0.0, 1.0))

        if yaw_options is None:
            yaw_options = [k * (math.pi / 4.0) for k in range(8)]
        self.yaw_options = [float(y) for y in yaw_options]

        self.grid_spec: Optional[GridSpec] = None
        self.fov_spec: Optional[FoVSpec] = None
        self.floor_fov_spec: Optional[FoVSpec] = None
        self.table_fov_spec: Optional[FoVSpec] = None
        self.env_mgr: Optional[GridEnvManager] = None
        self.env_mgr_floor: Optional[GridEnvManager] = None
        self.env_mgr_table: Optional[GridEnvManager] = None
        self.navigator: Optional[Navigator] = None

        # q_joint[cy, cx, :] over the 6 joint states; q_s[cy, cx, :] its content
        # marginal Cat(empty, bottle, can).
        self.q_joint: Optional[np.ndarray] = None
        self.q_s: Optional[np.ndarray] = None

        # Learned prior: per-cell evidence records ell, prior terms (base, d_rho,
        # d_theta), grid points, per-cell evidence coefficients (a, b, c), the
        # running sum of log Z over cells and the posterior weights of the points.
        self._ell: Optional[np.ndarray] = None
        self._terms = self._rho_grid = self._theta_grid = None
        self._coef = self._log_ev = self._rho_weights = None

        # Coverage threshold the controller uses for its visible mask. Planning
        # only predicts observations for cells that would actually be updated.
        self._coverage_threshold: float = 0.0

        # Target boxes projected onto cells outside the visible mask. They are
        # not observations of a visible cell, so they are dropped and counted.
        self._dropped_target_events: int = 0

        self._snap_dir: Optional[str] = None
        self._stay_replan_streak: int = 0
        self._last_replan_time_s: float = -1e9
        self._last_chosen: Optional[Tuple[Cell, float]] = None
        self._last_scored_actions: Optional[list] = None

        # Planner cost: number of replans and wall-clock time spent choosing them.
        self._n_replans: int = 0
        self._planner_s_total: float = 0.0
        self._planner_ms_last: float = 0.0

        # Floor and table plane coverages of every candidate (cell, yaw), cached.
        self._candidate_coverage_cache: Dict[Tuple[int, int,
                                                   int], Dict[str, Dict[Tuple[int, int], float]]] = {}

    def parameters(self) -> dict:
        """Every value the policy uses, for the run logs."""
        return {
            "policy": "zoi",
            "model": self.params.as_dict(),
            "observation_model": {
                "surface": "joint belief over {table, floor} x {empty, bottle, can}",
                "visibility": "table-plane coverage for table states, floor-plane coverage for floor states",
                "desk_evidence": "desk box covers the cell; informative when table-plane coverage >= update gate",
                "prior": ("rho and theta learned in flight, uniform prior on the feasible region"
                          if self.table_context_enabled else "theta learned in flight, rho = 0 (no desks)"),
                "rho_theta_grid_size": int(self.params.grid_size),
                "rho_theta_grid_points": self._grid_points(),
            },
            "planning": {
                "score": "expected information gain / flight time",
                "implementation": "loop",
                "yaw_options": [float(y) for y in self.yaw_options],
                "replan_cooldown_s": float(self.replan_cooldown_s),
                "classified_confidence_threshold": float(self.classified_confidence_threshold),
                "navigator": dict(NAVIGATOR_KWARGS),
                "time_model": self.time_model.as_dict(),
            },
        }

    def _grid_points(self) -> int:
        """Number of (rho, theta) grid points, also before the first frame."""
        if self._rho_grid is not None:
            return int(self._rho_grid.size)
        if not self.table_context_enabled:
            return int(self.params.grid_size)
        rho, _ = feasible_grid(p_on_table=self.params.p_target_on_table,
                               n_rho=self.params.grid_size, n_theta=self.params.grid_size)
        return int(rho.size)

    def diagnostics(self) -> dict:
        """Per-frame values for the perception log: rho and theta estimates, and the
        planner's cost so far (replans, total and last wall-clock time)."""
        if self._rho_weights is None:
            return {}
        mean, sd = self.rho_estimate()
        t_mean, t_sd = self.theta_estimate()
        return {"rho_mean": round(mean, 5), "rho_sd": round(sd, 5),
                "theta_mean": round(t_mean, 5), "theta_sd": round(t_sd, 5),
                "replans": self._n_replans, "planner_s_total": round(self._planner_s_total, 4),
                "planner_ms_last": round(self._planner_ms_last, 2)}

    @staticmethod
    def _mean_sd(w: np.ndarray, g: np.ndarray) -> tuple:
        mean = float(w @ g)
        return mean, float(np.sqrt(max(0.0, w @ (g - mean) ** 2)))

    def rho_estimate(self) -> tuple:
        """(mean, sd) of p(rho | data), the share of desk cells."""
        return self._mean_sd(self._rho_weights, self._rho_grid)

    def theta_estimate(self) -> tuple:
        """(mean, sd) of p(theta | data), the fraction of cells holding a target."""
        return self._mean_sd(self._rho_weights, self._theta_grid)

    def table_probability_map(self) -> np.ndarray:
        """P(table) per cell, [cy, cx]."""
        assert self.q_joint is not None
        return self.q_joint[:, :, 0:3].sum(axis=2)

    # ---------------------------------------------------------------------
    # Public controller API
    # ---------------------------------------------------------------------

    def compute_action(self, observation: dict) -> Dict[str, object]:
        self._ensure_initialized(observation)

        step = observation.get("step", None)
        pose = observation["pose"]
        grid = observation["grid"]
        task = observation.get("task", {}) or {}

        snap_cfg = (task.get("snapshot", {}) or {})
        snap_enabled = bool(snap_cfg.get("enabled", False))
        snap_dir = snap_cfg.get("dir", None)
        snap_every_n = int(snap_cfg.get("every_n", 0) or 0)
        snap_on_stuck = bool(snap_cfg.get("on_stuck", False))
        stuck_limit = int(snap_cfg.get("stuck_replan_limit", 0) or 0)

        if snap_enabled and (self._snap_dir is None):
            self._snap_dir = self._maybe_init_snapshot_dir(snap_dir)

        dbg_policy = bool(self.debug.get("policy_grid", False))
        dbg_every = int(self.debug.get("every_n", 50))
        step_ok = (step is not None) and (step % dbg_every == 0)

        current_cell: Optional[Cell] = self._parse_cell(
            grid.get("current_cell", None))
        visible_mask: Dict[Cell, bool] = self._normalize_bool_cell_map(
            grid.get("visible_mask", {}) or {})
        coverages: Dict[Cell, float] = self._normalize_float_cell_map(
            grid.get("coverages", {}) or {})
        perception_step = bool(
            grid.get("perception_processed_this_step", False))
        # The controller's table map: only to predict its visible mask when planning.
        table_cells = self._table_cells_from_grid(grid)
        if grid.get("coverage_threshold", None) is not None:
            self._coverage_threshold = float(grid["coverage_threshold"])

        # Belief updates only on processed perception steps, because the
        # controller sends empty visible/evidence fields on throttled frames.
        if perception_step:
            self._update_beliefs_from_observation(
                grid=grid,
                visible_mask=visible_mask,
                coverages=coverages,
            )

        assert self.navigator is not None

        scored_actions = None
        target_selection = None

        time_now = float(observation.get("time", 0.0))
        has_target = self.navigator.has_target()
        arrived = self.navigator.arrived(
            pose, coverages) if has_target else False
        cooldown_elapsed = (
            time_now - self._last_replan_time_s) >= self.replan_cooldown_s
        should_replan = (not has_target) or (arrived and cooldown_elapsed)

        if should_replan and current_cell is not None:
            t_plan = time.perf_counter()
            (next_cell, next_yaw), scored_actions = self._pick_next_action(
                current_cell=current_cell,
                table_cells=table_cells,
                return_scores=True,
                pose=pose,
            )
            self._planner_ms_last = 1000.0 * (time.perf_counter() - t_plan)
            self._planner_s_total += self._planner_ms_last / 1000.0
            self._n_replans += 1

            target_distance_m = self._cell_distance_m(current_cell, next_cell)
            target_selection = {
                "step": None if step is None else int(step),
                "time_s": float(time_now),
                "current_cell": [int(current_cell[0]), int(current_cell[1])],
                "target_cell": [int(next_cell[0]), int(next_cell[1])],
                "target_yaw": float(next_yaw),
                "distance_m": float(target_distance_m),
                "target_p_present": float(self._belief_present(next_cell[0], next_cell[1])),
                "target_classification_confidence": float(self._classification_confidence(next_cell[0], next_cell[1])),
                "target_classification_need": float(self._classification_need(self.q_s[next_cell[1], next_cell[0], :])),
                "target_p_table": float(self.table_probability_map()[next_cell[1], next_cell[0]]),
                "target_is_table": bool(next_cell in table_cells),
                "planner_ms": float(self._planner_ms_last),
            }

            self.navigator.set_target(next_cell, next_yaw)
            self._last_replan_time_s = time_now
            self._last_chosen = (next_cell, next_yaw)
            self._last_scored_actions = scored_actions

            if next_cell == current_cell:
                self._stay_replan_streak += 1
            else:
                self._stay_replan_streak = 0

            if (
                snap_enabled and snap_on_stuck and self._snap_dir and
                stuck_limit > 0 and self._stay_replan_streak >= stuck_limit and
                scored_actions is not None
            ):
                self._save_snapshot_npz(
                    snap_dir=self._snap_dir,
                    step=int(step or 0),
                    pose=pose,
                    current_cell=current_cell,
                    found_cells=self._found_cells_from_task(task),
                    table_cells=table_cells,
                    visible_mask=visible_mask,
                    coverages=coverages,
                    chosen=(next_cell, next_yaw),
                    scored_actions=scored_actions,
                    reason=f"stuck{self._stay_replan_streak}",
                )
                self._stay_replan_streak = 0

            if dbg_policy and step_ok:
                print(
                    f"[ZOI] picked target cell={next_cell} yaw={next_yaw:.3f} "
                    f"p_present={self._belief_present(next_cell[0], next_cell[1]):.2f} "
                    f"class_conf={self._classification_confidence(next_cell[0], next_cell[1]):.2f} "
                    f"p_table={self.table_probability_map()[next_cell[1], next_cell[0]]:.2f}",
                    flush=True,
                )

        if snap_enabled and self._snap_dir and snap_every_n > 0 and step is not None and (step % snap_every_n == 0):
            if current_cell is not None:
                (best_cell, best_yaw), scored = self._pick_next_action(
                    current_cell=current_cell,
                    table_cells=table_cells,
                    return_scores=True,
                    pose=pose,
                )
                self._save_snapshot_npz(
                    snap_dir=self._snap_dir,
                    step=int(step),
                    pose=pose,
                    current_cell=current_cell,
                    found_cells=self._found_cells_from_task(task),
                    table_cells=table_cells,
                    visible_mask=visible_mask,
                    coverages=coverages,
                    chosen=(best_cell, best_yaw),
                    scored_actions=scored,
                    reason="periodic",
                )

        if dbg_policy and step_ok:
            self._print_debug_grid(
                step, grid, visible_mask, coverages, table_cells)

        cmd = self.navigator.compute_command(pose)

        return {
            "forward": float(cmd["forward"]),
            "sideways": float(cmd["sideways"]),
            "yaw_rate": float(cmd["yaw_rate"]),
            "target_selection": target_selection,
        }

    def save_snapshot_from_controller(self, observation: dict, *, reason: str = "final") -> Optional[str]:
        """Compatible final/timeout snapshot hook used by the controller."""
        try:
            self._ensure_initialized(observation)
        except Exception:
            return None

        task = observation.get("task", {}) or {}
        snap_cfg = (task.get("snapshot", {}) or {})
        snap_enabled = bool(snap_cfg.get("enabled", False))
        snap_dir = snap_cfg.get("dir", None)

        if not snap_enabled or not snap_dir:
            return None

        if self._snap_dir is None:
            self._snap_dir = self._maybe_init_snapshot_dir(snap_dir)
        if not self._snap_dir:
            return None

        step = int(observation.get("step", 0) or 0)
        pose = observation.get("pose", {}) or {}
        grid = observation.get("grid", {}) or {}

        current_cell = self._parse_cell(grid.get("current_cell", None))
        table_cells = self._table_cells_from_grid(grid)
        visible_mask = self._normalize_bool_cell_map(
            grid.get("visible_mask", {}) or {})
        coverages = self._normalize_float_cell_map(
            grid.get("coverages", {}) or {})

        scored_actions = []
        chosen = self._last_chosen or (
            current_cell, float(pose.get("yaw", 0.0)))

        if current_cell is not None:
            try:
                (best_cell, best_yaw), scored_actions = self._pick_next_action(
                    current_cell=current_cell,
                    table_cells=table_cells,
                    return_scores=True,
                    pose={"x": float(pose.get("x", 0.0)), "y": float(pose.get("y", 0.0)),
                          "yaw": float(pose.get("yaw", 0.0))} if pose else None,
                )
                chosen = (best_cell, best_yaw)
            except Exception:
                pass

        return self._save_snapshot_npz(
            snap_dir=self._snap_dir,
            step=step,
            pose={"x": float(pose.get("x", 0.0)), "y": float(
                pose.get("y", 0.0)), "yaw": float(pose.get("yaw", 0.0))},
            current_cell=current_cell,
            found_cells=self._found_cells_from_task(task),
            table_cells=table_cells,
            visible_mask=visible_mask,
            coverages=coverages,
            chosen=chosen,
            scored_actions=scored_actions,
            reason=reason,
        )

    # ---------------------------------------------------------------------
    # Initialization and geometry cache
    # ---------------------------------------------------------------------

    def _ensure_initialized(self, obs: dict) -> None:
        if self.grid_spec is not None:
            return

        gs = obs["grid_spec"]
        fs = obs["fov_spec"]
        fov_by_surface = obs.get("fov_spec_by_surface", {}) or {}

        self.grid_spec = GridSpec(
            min_x=float(gs["min_x"]),
            max_x=float(gs["max_x"]),
            min_y=float(gs["min_y"]),
            max_y=float(gs["max_y"]),
            cell_size=float(gs["cell_size"]),
            rows=int(gs["rows"]),
            cols=int(gs["cols"]),
        )
        self.fov_spec = self._fov_spec_from_dict(fs)
        self.floor_fov_spec = self._fov_spec_from_dict(
            fov_by_surface.get("floor", fs))
        self.table_fov_spec = self._fov_spec_from_dict(
            fov_by_surface.get("table", fs))

        self.env_mgr_floor = GridEnvManager(
            self.grid_spec, self.floor_fov_spec)
        self.env_mgr_table = GridEnvManager(
            self.grid_spec, self.table_fov_spec)
        self.env_mgr = self.env_mgr_floor

        self.navigator = Navigator(self.grid_spec, **NAVIGATOR_KWARGS)
        self._precompute_candidate_coverages()

        # Learned prior: uniform over the feasible (rho, theta) grid, no evidence yet.
        rows, cols = self.grid_spec.rows, self.grid_spec.cols
        if self.table_context_enabled:
            pi = self.params.p_target_on_table
            self._rho_grid, self._theta_grid = feasible_grid(
                p_on_table=pi, n_rho=self.params.grid_size, n_theta=self.params.grid_size)
        else:
            pi = 0.0
            self._rho_grid, self._theta_grid = floor_only_grid(n_theta=self.params.grid_size)
        self._terms = joint_prior_terms(p_on_table=pi)
        self._ell = np.ones((rows, cols, 6))
        self._coef = evidence_coefficients(self._ell, *self._terms)
        self._log_ev = np.zeros_like(self._rho_grid)  # unseen cells have Z = 1
        self._refresh_beliefs()

    @staticmethod
    def _fov_spec_from_dict(fs: dict) -> FoVSpec:
        return FoVSpec(
            half_left=float(fs["half_left"]),
            half_right=float(fs["half_right"]),
            forward=float(fs["forward"]),
            backward=float(fs["backward"]),
        )

    def _precompute_candidate_coverages(self) -> None:
        assert self.grid_spec is not None
        assert self.env_mgr_floor is not None
        assert self.env_mgr_table is not None

        self._candidate_coverage_cache.clear()
        for cy in range(self.grid_spec.rows):
            for cx in range(self.grid_spec.cols):
                target_xy = self._cell_center_xy(cx, cy)
                for yaw_idx, yaw in enumerate(self.yaw_options):
                    key = (int(cx), int(cy), int(yaw_idx))
                    self._candidate_coverage_cache[key] = {
                        "floor": dict(self.env_mgr_floor.compute_cell_coverages(target_xy[0], target_xy[1], float(yaw))),
                        "table": dict(self.env_mgr_table.compute_cell_coverages(target_xy[0], target_xy[1], float(yaw))),
                    }

    def _cached_candidate_coverages_rc(
        self,
        *,
        target_cell: Cell,
        target_yaw_index: int,
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        assert self.env_mgr_floor is not None
        assert self.env_mgr_table is not None

        cx, cy = int(target_cell[0]), int(target_cell[1])
        key = (cx, cy, int(target_yaw_index))
        cached = self._candidate_coverage_cache.get(key, None)
        if cached is not None:
            return cached["floor"], cached["table"]

        target_xy = self._cell_center_xy(cx, cy)
        yaw = float(self.yaw_options[int(target_yaw_index)])
        floor_cov = dict(self.env_mgr_floor.compute_cell_coverages(
            target_xy[0], target_xy[1], yaw))
        table_cov = dict(self.env_mgr_table.compute_cell_coverages(
            target_xy[0], target_xy[1], yaw))
        self._candidate_coverage_cache[key] = {
            "floor": floor_cov, "table": table_cov}
        return floor_cov, table_cov

    def _merged_candidate_coverages_rc(
        self,
        *,
        target_cell: Cell,
        target_yaw_index: int,
        table_cells: Optional[set[Cell]] = None,
    ) -> Dict[Tuple[int, int], float]:
        table_cells = table_cells or set()
        floor_cov, table_cov = self._cached_candidate_coverages_rc(
            target_cell=target_cell,
            target_yaw_index=int(target_yaw_index),
        )

        out: Dict[Tuple[int, int], float] = {}
        all_rc = set(floor_cov.keys()) | set(table_cov.keys())
        for (r, c) in all_rc:
            rc = (int(r), int(c))
            cxcy = (int(c), int(r))
            val = float(table_cov.get(rc, 0.0)) if cxcy in table_cells else float(
                floor_cov.get(rc, 0.0))
            if val > 0.0:
                out[rc] = val
        return out

    # ---------------------------------------------------------------------
    # Belief (see policies/zoi_model.py)
    # ---------------------------------------------------------------------

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
        """Add each visible cell's desk and target observation to its evidence
        record, then refresh p(rho, theta | data) and every belief once."""
        assert self._ell is not None
        rows, cols = self._ell.shape[0], self._ell.shape[1]

        target_events_by_cell = self._target_events_by_cell(grid)
        desk_cells = self._desk_covered_cells(grid)

        # Coverage of each cell by the camera footprint at floor and at table
        # height. Older payloads without them fall back to the merged coverage.
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
            row = self._joint_likelihood(w_table, w_floor)[obs_index]
            ell = self._ell[cell[1], cell[0], :] * row
            self._ell[cell[1], cell[0], :] = ell / ell.max()

        if visible_cells:
            self._update_evidence(sorted(visible_cells))
            self._refresh_beliefs()

        for cell, events in target_events_by_cell.items():
            if cell not in visible_cells:
                self._dropped_target_events += len(events)

    def _update_evidence(self, cells: list) -> None:
        """Swap the updated cells' terms in the running sum of log Z over cells."""
        cols = self._ell.shape[1]
        idx = np.array([cy * cols + cx for cx, cy in cells], dtype=int)
        new = evidence_coefficients(self._ell.reshape(-1, 6)[idx], *self._terms)
        self._log_ev += (log_evidence(new, self._rho_grid, self._theta_grid)
                         - log_evidence(self._coef[idx], self._rho_grid, self._theta_grid))
        self._coef[idx] = new

    def _refresh_beliefs(self) -> None:
        """p(rho, theta | data) from the running log evidence; each cell's belief is the
        exact average over it. Unseen cells (ell = 1) share one belief."""
        rows, cols = self._ell.shape[0], self._ell.shape[1]
        self._rho_weights = grid_weights(self._log_ev)
        ell = self._ell.reshape(-1, 6)
        seen = np.flatnonzero(np.any(ell != 1.0, axis=1))
        unseen_q = marginal_beliefs_2d(np.ones((1, 6)), evidence_coefficients(np.ones((1, 6)), *self._terms),
                                       self._terms, self._rho_grid, self._theta_grid, self._rho_weights)[0]
        q = np.tile(unseen_q, (rows * cols, 1))
        if seen.size:
            # Average only over grid points that carry weight: after a few frames the
            # posterior sits on a few hundred of the ~8 k points. The dropped mass is
            # below 1e-12 of the largest weight per point, so beliefs change by < 1e-8.
            w = self._rho_weights
            active = np.flatnonzero(w > 1e-12 * w.max())
            w_act = w[active] / w[active].sum()
            q[seen] = marginal_beliefs_2d(ell[seen], self._coef[seen], self._terms,
                                          self._rho_grid[active], self._theta_grid[active], w_act)
        self.q_joint = q.reshape(rows, cols, 6)
        self.q_s = (self.q_joint[:, :, 0:3] + self.q_joint[:, :, 3:6]).copy()

    def _cell_observation(self, events: List[dict]) -> int:
        """Reduce a cell's target boxes in one frame to a single observation.

        The model allows at most one observation per cell per frame. When
        several boxes land in the same cell, the class of the highest-confidence
        box is used. Confidence only selects the box; it is not evidence.
        """
        best_obs = O_NOTHING
        best_conf = -math.inf
        for event in events:
            obs = CLASS_NAME_TO_OBS.get(self._event_class_name(event), None)
            if obs is None:
                continue
            conf = float(event.get("confidence", 0.0))
            if conf > best_conf:
                best_conf = conf
                best_obs = obs
        return int(best_obs)

    @staticmethod
    def _normalize(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        z = float(q.sum())
        if z <= 0.0:
            return np.ones_like(q) / float(q.size)
        return q / z

    # ---------------------------------------------------------------------
    # Planner: expected information gain per second of flight
    # ---------------------------------------------------------------------

    def _all_cells(self) -> List[Cell]:
        assert self.grid_spec is not None
        return [(cx, cy) for cy in range(self.grid_spec.rows) for cx in range(self.grid_spec.cols)]

    def _cell_center_xy(self, cx: int, cy: int) -> Tuple[float, float]:
        assert self.grid_spec is not None
        x0 = self.grid_spec.min_x + int(cx) * self.grid_spec.cell_size
        y0 = self.grid_spec.min_y + int(cy) * self.grid_spec.cell_size
        cs = self.grid_spec.cell_size
        return (x0 + 0.5 * cs, y0 + 0.5 * cs)

    def _cell_distance_m(self, a: Cell, b: Cell) -> float:
        ax, ay = self._cell_center_xy(int(a[0]), int(a[1]))
        bx, by = self._cell_center_xy(int(b[0]), int(b[1]))
        return float(math.hypot(bx - ax, by - ay))

    def _pick_next_action(
        self,
        *,
        current_cell: Cell,
        table_cells: Optional[set[Cell]] = None,
        return_scores: bool = False,
        pose: Optional[dict] = None,
    ):
        """Best (cell, yaw) by IG / T, T the navigator's flight time plus one
        detector period; ties go to the shorter flight."""
        assert self.grid_spec is not None
        assert self.q_joint is not None

        candidates = self._all_cells()
        table_cells = table_cells or set()
        scores = [] if return_scores else None

        best = (candidates[0], self.yaw_options[0])
        best_key = (-float("inf"), -float("inf"))

        cx0, cy0 = current_cell
        x0, y0 = self._cell_center_xy(cx0, cy0)
        if pose is None:
            pose = {"x": x0, "y": y0, "yaw": 0.0}

        for cell in candidates:
            tx, ty = self._cell_center_xy(cell[0], cell[1])
            for yaw_idx, yaw in enumerate(self.yaw_options):
                IG = self._candidate_information_gain(
                    target_cell=cell, target_yaw_index=yaw_idx, table_cells=table_cells)
                T, t_turn, t_move = self.time_model.time_to(pose, (tx, ty), yaw)
                rate = float(IG / T)

                if return_scores:
                    scores.append({
                        "cell": [int(cell[0]), int(cell[1])],
                        "yaw": float(yaw),
                        "IG": float(IG),
                        "distance": float(math.hypot(tx - x0, ty - y0)),
                        "T": float(T),
                        "T_turn": float(t_turn),
                        "T_move": float(t_move),
                        "rate": rate,
                        # Minimised by utils/snapshot_analyzer.py.
                        "G": -rate,
                    })

                key = (rate, -T)
                if key > best_key:
                    best_key = key
                    best = (cell, yaw)

        return (best, scores) if return_scores else best

    def _candidate_information_gain(
        self,
        *,
        target_cell: Cell,
        target_yaw_index: int,
        table_cells: Optional[set[Cell]] = None,
    ) -> float:
        """Expected information gain of one view from a candidate (cell, yaw), summed
        over the cells the controller would update: coverage at or above its
        visible-mask threshold, with the table-plane coverage on its table-map cells."""
        table_cells = table_cells or set()
        cov_rc = self._merged_candidate_coverages_rc(
            target_cell=target_cell,
            target_yaw_index=int(target_yaw_index),
            table_cells=table_cells,
        )
        floor_cov_rc, table_cov_rc = self._cached_candidate_coverages_rc(
            target_cell=target_cell,
            target_yaw_index=int(target_yaw_index),
        )

        IG = 0.0
        for (r, c), frac in cov_rc.items():
            if float(frac) < self._coverage_threshold:
                continue
            IG += self._cell_information_gain(
                (int(c), int(r)),
                float(table_cov_rc.get((int(r), int(c)), 0.0)),
                float(floor_cov_rc.get((int(r), int(c)), 0.0)),
            )
        return float(IG)

    def _cell_information_gain(self, cell: Cell, w_table: float, w_floor: float) -> float:
        """Expected information gain of one view of a cell with these plane coverages,
        with the same joint likelihood as the update."""
        assert self.q_joint is not None
        q = self.q_joint[int(cell[1]), int(cell[0]), :]
        return expected_information_gain(q, self._joint_likelihood(w_table, w_floor))

    @staticmethod
    def _present_prob(q: np.ndarray) -> float:
        return float(np.sum(q[1:]))

    def _classification_need(self, q: np.ndarray) -> float:
        q = self._normalize(q)
        p_present = self._present_prob(q)
        if p_present <= 1e-12:
            return 0.0
        class_conf = max(float(q[S_BOTTLE]), float(
            q[S_CAN])) / max(p_present, 1e-12)
        return float(p_present * (1.0 - class_conf))

    def _classification_confidence_from_q(self, q: np.ndarray) -> float:
        q = self._normalize(q)
        p_present = self._present_prob(q)
        if p_present <= 1e-12:
            return 0.0
        return float(max(float(q[S_BOTTLE]), float(q[S_CAN])) / max(p_present, 1e-12))

    def _belief_present(self, cx: int, cy: int) -> float:
        if self.q_s is None:
            return 0.0
        rows, cols = self.q_s.shape[0], self.q_s.shape[1]
        if not (0 <= int(cx) < cols and 0 <= int(cy) < rows):
            return 0.0
        return self._present_prob(self.q_s[int(cy), int(cx), :])

    def _classification_confidence(self, cx: int, cy: int) -> float:
        if self.q_s is None:
            return 0.0
        rows, cols = self.q_s.shape[0], self.q_s.shape[1]
        if not (0 <= int(cx) < cols and 0 <= int(cy) < rows):
            return 0.0
        return self._classification_confidence_from_q(self.q_s[int(cy), int(cx), :])

    # ---------------------------------------------------------------------
    # Observation/event parsing
    # ---------------------------------------------------------------------

    def _target_events_by_cell(self, grid: dict) -> Dict[Cell, List[dict]]:
        by_cell: Dict[Cell, List[dict]] = {}
        events = list(grid.get("target_detections", []) or [])

        # Backstop for future/older payloads where all detections are present but
        # target_detections is missing.
        if not events:
            for event in grid.get("detections_all", []) or []:
                if self._is_target_event(event):
                    events.append(event)

        for event in events:
            if not self._is_target_event(event):
                continue
            cell = self._cell_from_event(event)
            if cell is None:
                continue
            by_cell.setdefault(cell, []).append(event)

        return by_cell


    def _cell_from_event(self, event: dict) -> Optional[Cell]:
        if not isinstance(event, dict):
            return None
        cell = event.get("cell", None)
        if cell is None and "cx" in event and "cy" in event:
            cell = (event.get("cx"), event.get("cy"))
        return self._parse_cell(cell)

    def _event_class_name(self, event: dict) -> str:
        if not isinstance(event, dict):
            return ""
        name = event.get("class_name", event.get("label", ""))
        if name:
            return str(name)
        class_id = event.get("class_id", None)
        try:
            cid = int(class_id)
        except Exception:
            return ""
        if cid == 0:
            return "target_bottle"
        if cid == 1:
            return "target_can"
        if cid == 2:
            return "desk"
        return ""


    def _is_target_event(self, event: dict) -> bool:
        if not isinstance(event, dict):
            return False
        return self._event_class_name(event) in TARGET_CLASS_NAMES


    @staticmethod
    def _parse_cell(value) -> Optional[Cell]:
        if isinstance(value, dict):
            if "cell" in value:
                value = value.get("cell")
            elif "cx" in value and "cy" in value:
                value = (value.get("cx"), value.get("cy"))
        if isinstance(value, str) and "," in value:
            parts = value.split(",", 1)
            try:
                return (int(parts[0]), int(parts[1]))
            except Exception:
                return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return (int(value[0]), int(value[1]))
            except Exception:
                return None
        return None

    @classmethod
    def _normalize_float_cell_map(cls, raw: dict) -> Dict[Cell, float]:
        out: Dict[Cell, float] = {}
        for k, v in (raw or {}).items():
            cell = cls._parse_cell(k)
            if cell is not None:
                out[cell] = float(v)
        return out

    @classmethod
    def _normalize_bool_cell_map(cls, raw: dict) -> Dict[Cell, bool]:
        out: Dict[Cell, bool] = {}
        for k, v in (raw or {}).items():
            cell = cls._parse_cell(k)
            if cell is not None:
                out[cell] = bool(v)
        return out

    @classmethod
    def _table_cells_from_grid(cls, grid: dict) -> set[Cell]:
        out: set[Cell] = set()

        for cell in grid.get("table_cells", []) or []:
            parsed = cls._parse_cell(cell)
            if parsed is not None:
                out.add(parsed)

        surface_map = grid.get("surface_height_map", {}) or {}
        for cell in surface_map.get("table_cells", []) or []:
            parsed = cls._parse_cell(cell)
            if parsed is not None:
                out.add(parsed)

        # Desk detections may also carry covered_cells directly. This is not a
        # substitute for the controller's persistent table_cells map; it only
        # helps if the policy is run on an observation payload before that map is
        # populated.
        for event in grid.get("detections_all", []) or []:
            if not isinstance(event, dict):
                continue
            name = str(event.get("class_name", event.get("label", "")))
            try:
                cid = int(event.get("class_id", -1))
            except Exception:
                cid = -1
            if name not in DESK_CLASS_NAMES and cid != 2:
                continue
            for cell in event.get("covered_cells", []) or []:
                parsed = cls._parse_cell(cell)
                if parsed is not None:
                    out.add(parsed)
            parsed = cls._cell_from_event_static(event)
            if parsed is not None:
                out.add(parsed)

        return out

    @classmethod
    def _cell_from_event_static(cls, event: dict) -> Optional[Cell]:
        if not isinstance(event, dict):
            return None
        return cls._parse_cell(event.get("cell", None))

    @staticmethod
    def _found_cells_from_task(task: dict) -> set[Cell]:
        out: set[Cell] = set()
        for cell in task.get("found_cells", []) or []:
            if isinstance(cell, (list, tuple)) and len(cell) == 2:
                try:
                    out.add((int(cell[0]), int(cell[1])))
                except Exception:
                    pass
        return out

    # ---------------------------------------------------------------------
    # Debug/snapshot utilities
    # ---------------------------------------------------------------------

    def _top_beliefs(self, k: int = 5) -> List[Tuple[int, int, float, float, float]]:
        if self.q_s is None:
            return []
        p_present = self.q_s[:, :, 1:].sum(axis=2)
        flat = p_present.reshape(-1)
        if flat.size == 0:
            return []
        k = int(max(1, min(k, flat.size)))
        idxs = np.argpartition(flat, -k)[-k:]
        idxs = idxs[np.argsort(flat[idxs])[::-1]]
        cols = p_present.shape[1]
        out = []
        for idx in idxs:
            cy = int(idx // cols)
            cx = int(idx % cols)
            q = self.q_s[cy, cx, :]
            out.append((cx, cy, float(p_present[cy, cx]), float(
                self._classification_confidence_from_q(q)), float(self._classification_need(q))))
        return out

    def _print_debug_grid(
        self,
        step,
        grid: dict,
        visible_mask: Dict[Cell, bool],
        coverages: Dict[Cell, float],
        table_cells: set[Cell],
    ) -> None:
        cc = grid.get("current_cell")
        num_vis = sum(1 for v in visible_mask.values() if v)
        print(
            f"[ZOI GRID] step={step} cc={cc} "
            f"cover={len(coverages)} visible={num_vis} table_cells={len(table_cells)} "
            f"dropped_target_events={self._dropped_target_events} {self.diagnostics()}",
            flush=True,
        )

        p_table = self.table_probability_map() if self.q_joint is not None else None
        top_cov = sorted(coverages.items(),
                         key=lambda kv: kv[1], reverse=True)[:9]
        for (cx, cy), frac in top_cov:
            q = self.q_s[cy, cx, :] if self.q_s is not None else np.zeros(3)
            pt = float(p_table[cy, cx]) if p_table is not None else float("nan")
            print(
                f"  cell=({cx},{cy}) cov={frac*100:5.1f}% vis={visible_mask.get((cx, cy), None)} "
                f"p_table={pt:.2f} p={self._present_prob(q):.2f} "
                f"q=[empty={q[S_EMPTY]:.2f}, bot={q[S_BOTTLE]:.2f}, can={q[S_CAN]:.2f}]",
                flush=True,
            )

        top_bel = self._top_beliefs(k=5)
        if top_bel:
            print(
                "  top p_present: " + ", ".join(
                    [f"({cx},{cy}) p={p:.2f} cls={cls:.2f} need={need:.2f}" for (
                        cx, cy, p, cls, need) in top_bel]
                ),
                flush=True,
            )

        if self.navigator is not None and self.navigator.has_target():
            print(
                f"  target={self.navigator.target_cell} target_yaw={self.navigator.target_yaw:.3f}", flush=True)

    def _maybe_init_snapshot_dir(self, snap_dir: Optional[str]) -> Optional[str]:
        if not snap_dir:
            return None
        os.makedirs(snap_dir, exist_ok=True)
        return snap_dir

    def _classified_cells_payload(self) -> List[dict]:
        if self.q_s is None:
            return []
        rows, cols = self.q_s.shape[0], self.q_s.shape[1]
        out = []
        for cy in range(rows):
            for cx in range(cols):
                q = self.q_s[cy, cx, :]
                p_present = self._present_prob(q)
                if p_present <= 1e-9:
                    continue
                conf = self._classification_confidence_from_q(q)
                if conf < self.classified_confidence_threshold:
                    continue
                cls = "target_bottle" if q[S_BOTTLE] >= q[S_CAN] else "target_can"
                out.append({
                    "cell": [int(cx), int(cy)],
                    "class_name": cls,
                    "p_present": float(p_present),
                    "class_confidence": float(conf),
                    "q": [float(x) for x in q],
                })
        return out

    def _save_snapshot_npz(
        self,
        *,
        snap_dir: str,
        step: int,
        pose: dict,
        current_cell: Optional[Cell],
        found_cells: set[Cell],
        table_cells: Optional[set[Cell]],
        visible_mask: Dict[Cell, bool],
        coverages: Dict[Cell, float],
        chosen: Tuple[Optional[Cell], float],
        scored_actions: list,
        reason: str,
    ) -> str:
        if self.q_s is None or self.q_joint is None:
            raise RuntimeError("beliefs not initialised; cannot save snapshot")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"snapshot_step{int(step):06d}_{reason}_{ts}.npz"
        path = os.path.join(snap_dir, fname)

        q_present = self.q_s[:, :, 1:].sum(axis=2)
        q_class_conf = np.zeros_like(q_present)
        q_need = np.zeros_like(q_present)
        for cy in range(self.q_s.shape[0]):
            for cx in range(self.q_s.shape[1]):
                q = self.q_s[cy, cx, :]
                q_class_conf[cy,
                             cx] = self._classification_confidence_from_q(q)
                q_need[cy, cx] = self._classification_need(q)

        chosen_cell = chosen[0]
        meta = {
            "step": int(step),
            "reason": str(reason),
            "state_names": STATE_NAMES,
            "joint_state_names": JOINT_STATE_NAMES,
            "obs_names": OBS_NAMES,
            "pose": {"x": float(pose["x"]), "y": float(pose["y"]), "yaw": float(pose["yaw"])},
            "current_cell": list(current_cell) if current_cell is not None else None,
            "chosen_cell": [int(chosen_cell[0]), int(chosen_cell[1])] if chosen_cell is not None else None,
            "chosen_yaw": float(chosen[1]),
            "found_cells_debug_only": sorted([[int(c[0]), int(c[1])] for c in found_cells]),
            "table_cells": sorted([[int(c[0]), int(c[1])] for c in (table_cells or set())]),
            "classified_confidence_threshold": float(self.classified_confidence_threshold),
            "classified_cells": self._classified_cells_payload(),
            "visible_mask": {f"{k[0]},{k[1]}": bool(v) for k, v in visible_mask.items()},
            "coverages": {f"{k[0]},{k[1]}": float(v) for k, v in coverages.items()},
            "coverage_threshold": float(self._coverage_threshold),
            "dropped_target_events": int(self._dropped_target_events),
            "rho_theta": self.diagnostics(),
            "params": self.parameters(),
            "scored_actions": scored_actions,
        }

        np.savez_compressed(
            path,
            q_s=self.q_s.astype(np.float32),
            q_joint=self.q_joint.astype(np.float32),
            p_table=self.table_probability_map().astype(np.float32),
            q_present=q_present.astype(np.float32),
            q_class_confidence=q_class_conf.astype(np.float32),
            q_classification_need=q_need.astype(np.float32),
            table_map=self._table_map_array(
                table_cells or set()).astype(np.uint8),
            meta_json=np.array(json.dumps(meta), dtype=object),
        )
        print(f"[SNAPSHOT] wrote {path}", flush=True)
        return path

    def _table_map_array(self, table_cells: set[Cell]) -> np.ndarray:
        assert self.grid_spec is not None
        arr = np.zeros(
            (self.grid_spec.rows, self.grid_spec.cols), dtype=np.uint8)
        for cx, cy in table_cells:
            if 0 <= int(cx) < self.grid_spec.cols and 0 <= int(cy) < self.grid_spec.rows:
                arr[int(cy), int(cx)] = 1
        return arr
