# policies/aif_saccade_table.py

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from env.env_manager import FoVSpec, GridEnvManager, GridSpec
from utils.navigator import Navigator, clamp

Cell = Tuple[int, int]  # policy/report convention: (cx, cy)


# Per-cell latent target states.
S_EMPTY = 0
S_UNKNOWN = 1
S_BOTTLE = 2
S_CAN = 3
STATE_NAMES = ["empty", "unknown_target", "target_bottle", "target_can"]

# Per-cell target observation categories. There is deliberately no invisible /
# out-of-vision observation; cells outside the controller visible mask are not
# updated at all.
O_NO_TARGET = 0
O_GENERIC_TARGET = 1
O_BOTTLE = 2
O_CAN = 3
OBS_NAMES = ["no_target", "generic_target", "target_bottle", "target_can"]

TARGET_CLASS_NAMES = {"target_bottle", "target_can"}
DESK_CLASS_NAMES = {"desk"}


class AIFSaccadeTablePolicy:
    """Table-aware active-inference-style saccade policy.

    This policy keeps the controller-facing structure of AIFSaccadePolicy but
    replaces the old binary target belief with a categorical belief over
    {empty, unknown target, bottle, can}. It also maintains a table-belief map
    q_table from per-frame desk/table observations and uses q_table to condition
    the target prior. Target/table no-detection evidence is coverage-dependent
    inside the likelihood; positive detections use detector confidence directly.
    The persistent perception-derived table-cell map remains available for
    height-aware planning/FoV selection.

    The policy intentionally does not contain the old "exclude a detected target
    after N frames" mechanism. Already-classified cells become less attractive
    because the classification utility and expected information gain for those
    cells become small as the posterior class confidence becomes high.
    """

    def __init__(
        self,
        debug: Optional[dict] = None,
        *,
        # Contextual target priors. These are per-cell priors, not the global
        # probability that a target is on some table. Keep them tunable because
        # they depend on table size and number of spawned objects.
        prior_target_floor: float = 1.0 / 125.0,
        prior_target_table: float = 5.0 / 18.0,
        prior_unknown_fraction: float = 0.70,
        # Explicit table-belief model. The persistent grid["table_cells"] map is
        # still used as a certain height/context map for floor/table FoV selection,
        # but q_table is updated from per-frame desk/table observations and is used
        # to condition the target prior.
        table_belief_prior: float = 18.0 / 143.0,
        table_p_fa: float = 0.04,
        table_no_detection_sensitivity: float = 0.50,
        # Target observation model.
        p_fa: float = 0.04,
        no_detection_sensitivity: float = 0.50,
        class_report_prob: float = 1.00,
        class_accuracy: float = 0.92,
        unknown_class_report_prob: float = 0.25,
        peripheral_class_weight: float = 0.0,
        classification_coverage_threshold: float = 0.55,
        # Action scoring.
        beta_info: float = 1.0,
        classification_preference_strength: float = 2.0,
        generic_target_preference_strength: float = 0.0,
        table_exploration_weight: float = 0.0,
        distance_cost_epsilon: float = 0.04,
        planning_p_hit: float = 0.80,
        min_coverage_for_ig: float = 0.05,
        # Replanning/nav.
        yaw_options: Optional[List[float]] = None,
        replan_cooldown_s: float = 0.4,
        classified_confidence_threshold: float = 0.80,
    ):
        self.debug = debug or {}
        self.every_n = int(self.debug.get("every_n", 50))

        self.prior_target_floor = float(np.clip(prior_target_floor, 1e-6, 1.0 - 1e-6))
        self.prior_target_table = float(np.clip(prior_target_table, 1e-6, 1.0 - 1e-6))
        self.prior_unknown_fraction = float(np.clip(prior_unknown_fraction, 0.0, 1.0))
        self.table_belief_prior = float(np.clip(table_belief_prior, 1e-6, 1.0 - 1e-6))
        self.table_p_fa = float(np.clip(table_p_fa, 1e-6, 1.0 - 1e-6))
        self.table_no_detection_sensitivity = float(np.clip(table_no_detection_sensitivity, 1e-6, 1.0 - 1e-6))

        self.p_fa = float(np.clip(p_fa, 1e-6, 1.0 - 1e-6))
        self.no_detection_sensitivity = float(np.clip(no_detection_sensitivity, 1e-6, 1.0 - 1e-6))
        self.class_report_prob = float(np.clip(class_report_prob, 0.0, 1.0))
        self.class_accuracy = float(np.clip(class_accuracy, 0.5, 1.0))
        self.unknown_class_report_prob = float(np.clip(unknown_class_report_prob, 0.0, 1.0))
        self.peripheral_class_weight = float(np.clip(peripheral_class_weight, 0.0, 1.0))
        self.classification_coverage_threshold = float(np.clip(classification_coverage_threshold, 0.0, 1.0))

        self.beta_info = float(beta_info)
        self.classification_preference_strength = float(classification_preference_strength)
        self.generic_target_preference_strength = float(generic_target_preference_strength)
        self.table_exploration_weight = float(table_exploration_weight)
        self.distance_cost_epsilon = float(distance_cost_epsilon)
        self.planning_p_hit = float(np.clip(planning_p_hit, 1e-6, 1.0 - 1e-6))
        self.min_coverage_for_ig = float(np.clip(min_coverage_for_ig, 0.0, 1.0))

        self.replan_cooldown_s = float(replan_cooldown_s)
        self.classified_confidence_threshold = float(np.clip(classified_confidence_threshold, 0.0, 1.0))

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

        # q_s[cy, cx, :] is Cat(empty, unknown_target, bottle, can).
        self.q_s: Optional[np.ndarray] = None

        # q_table[cy, cx] is the policy's belief that the cell is table-height.
        # It is distinct from grid["table_cells"], which remains the persistent
        # certain context map used for height-aware FoV projection/planning.
        self.q_table: Optional[np.ndarray] = None
        self._applied_table_context_prob: Optional[np.ndarray] = None
        self._seeded_persistent_table_cells: set[Cell] = set()

        self._snap_dir: Optional[str] = None
        self._stay_replan_streak: int = 0
        self._last_replan_time_s: float = -1e9
        self._last_chosen: Optional[Tuple[Cell, float]] = None
        self._last_scored_actions: Optional[list] = None

        # Candidate FoV coverage cache, exactly mirroring the old policy idea:
        # cache floor/table projections for each candidate cell and yaw. Runtime
        # table_cells only select which surface-specific coverage to use per cell.
        self._candidate_coverage_cache: Dict[Tuple[int, int, int], Dict[str, Dict[Tuple[int, int], float]]] = {}

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

        current_cell: Optional[Cell] = self._parse_cell(grid.get("current_cell", None))
        visible_mask: Dict[Cell, bool] = self._normalize_bool_cell_map(grid.get("visible_mask", {}) or {})
        coverages: Dict[Cell, float] = self._normalize_float_cell_map(grid.get("coverages", {}) or {})
        table_cells = self._table_cells_from_grid(grid)
        perception_step = bool(grid.get("perception_processed_this_step", False))

        # Reconstruct table belief from any persistent table-confidence map already
        # accumulated by the controller before the policy first runs, then update
        # q_table from this processed perception frame. q_table then conditions
        # the target prior before target observations are assimilated.
        self._seed_table_beliefs_from_persistent_confidence(grid)
        if perception_step:
            self._update_table_beliefs_from_observation(grid=grid)
        self._apply_table_context_from_beliefs()

        # Only target-belief updates on processed perception steps because the
        # controller sends empty visible/evidence fields on throttled frames.
        if perception_step:
            self._update_beliefs_from_observation(
                grid=grid,
                current_cell=current_cell,
                visible_mask=visible_mask,
                coverages=coverages,
            )

        assert self.navigator is not None

        scored_actions = None
        target_selection = None

        time_now = float(observation.get("time", 0.0))
        has_target = self.navigator.has_target()
        arrived = self.navigator.arrived(pose, coverages) if has_target else False
        cooldown_elapsed = (time_now - self._last_replan_time_s) >= self.replan_cooldown_s
        should_replan = (not has_target) or (arrived and cooldown_elapsed)

        if should_replan and current_cell is not None:
            (next_cell, next_yaw), scored_actions = self._pick_next_action(
                current_cell=current_cell,
                table_cells=table_cells,
                return_scores=True,
            )

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
                "target_is_table": bool(next_cell in table_cells),
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
                    f"[AIF-TABLE] picked target cell={next_cell} yaw={next_yaw:.3f} "
                    f"p_present={self._belief_present(next_cell[0], next_cell[1]):.2f} "
                    f"class_conf={self._classification_confidence(next_cell[0], next_cell[1]):.2f} "
                    f"table={next_cell in table_cells}",
                    flush=True,
                )

        if snap_enabled and self._snap_dir and snap_every_n > 0 and step is not None and (step % snap_every_n == 0):
            if current_cell is not None:
                (best_cell, best_yaw), scored = self._pick_next_action(
                    current_cell=current_cell,
                    table_cells=table_cells,
                    return_scores=True,
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
            self._print_debug_grid(step, grid, visible_mask, coverages, table_cells)

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
        visible_mask = self._normalize_bool_cell_map(grid.get("visible_mask", {}) or {})
        coverages = self._normalize_float_cell_map(grid.get("coverages", {}) or {})

        scored_actions = []
        chosen = self._last_chosen or (current_cell, float(pose.get("yaw", 0.0)))

        if current_cell is not None:
            try:
                (best_cell, best_yaw), scored_actions = self._pick_next_action(
                    current_cell=current_cell,
                    table_cells=table_cells,
                    return_scores=True,
                )
                chosen = (best_cell, best_yaw)
            except Exception:
                pass

        return self._save_snapshot_npz(
            snap_dir=self._snap_dir,
            step=step,
            pose={"x": float(pose.get("x", 0.0)), "y": float(pose.get("y", 0.0)), "yaw": float(pose.get("yaw", 0.0))},
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
        self.floor_fov_spec = self._fov_spec_from_dict(fov_by_surface.get("floor", fs))
        self.table_fov_spec = self._fov_spec_from_dict(fov_by_surface.get("table", fs))

        self.env_mgr_floor = GridEnvManager(self.grid_spec, self.floor_fov_spec)
        self.env_mgr_table = GridEnvManager(self.grid_spec, self.table_fov_spec)
        self.env_mgr = self.env_mgr_floor

        self.navigator = Navigator(
            self.grid_spec,
            max_forward=0.50,
            max_sideways=0.50,
            max_yaw_rate=0.60,
            k_pos=1.2,
            k_yaw=1.5,
            dist_eps_m=0.2,
            yaw_eps_rad=0.12,
            min_cell_coverage=0.60,
        )

        self.q_s = np.zeros((self.grid_spec.rows, self.grid_spec.cols, 4), dtype=np.float64)
        initial_table_prob = self.table_belief_prior
        initial_target_prior = self._context_prior(table_prob=initial_table_prob)
        self.q_s[:, :, :] = initial_target_prior.reshape(1, 1, 4)

        self.q_table = np.full(
            (self.grid_spec.rows, self.grid_spec.cols),
            fill_value=initial_table_prob,
            dtype=np.float64,
        )
        self._applied_table_context_prob = np.full(
            (self.grid_spec.rows, self.grid_spec.cols),
            fill_value=initial_table_prob,
            dtype=np.float64,
        )

        self._precompute_candidate_coverages()

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
        floor_cov = dict(self.env_mgr_floor.compute_cell_coverages(target_xy[0], target_xy[1], yaw))
        table_cov = dict(self.env_mgr_table.compute_cell_coverages(target_xy[0], target_xy[1], yaw))
        self._candidate_coverage_cache[key] = {"floor": floor_cov, "table": table_cov}
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
            val = float(table_cov.get(rc, 0.0)) if cxcy in table_cells else float(floor_cov.get(rc, 0.0))
            if val > 0.0:
                out[rc] = val
        return out

    # ---------------------------------------------------------------------
    # Belief model
    # ---------------------------------------------------------------------

    def _context_prior(self, *, table_prob: float = 0.0) -> np.ndarray:
        """Target prior conditioned on table belief.

        table_prob is q_T(i) = P(table at cell i). The target-present prior is
        the mixture

            P(target_i) = q_T(i) P(target_i | table)
                        + (1-q_T(i)) P(target_i | floor).

        The split of target-present mass into unknown/bottle/can follows the
        same prior_unknown_fraction convention used by the previous policy.
        """
        q_t = float(np.clip(table_prob, 0.0, 1.0))
        p_present = (1.0 - q_t) * self.prior_target_floor + q_t * self.prior_target_table
        p_present = float(np.clip(p_present, 1e-6, 1.0 - 1e-6))

        p_unknown = p_present * self.prior_unknown_fraction
        p_known_each = p_present * (1.0 - self.prior_unknown_fraction) * 0.5
        prior = np.array([1.0 - p_present, p_unknown, p_known_each, p_known_each], dtype=np.float64)
        return self._normalize(prior)

    def _seed_table_beliefs_from_persistent_confidence(self, grid: dict) -> None:
        """Seed q_table from controller-persistent table confidence once per cell.

        The controller keeps grid["table_cells"] as the certain height map for
        planning/FoV selection. For the probabilistic table belief, it also sends
        the maximum desk-detection confidence that originally mapped each table
        cell. This method prevents the policy from losing table context when the
        controller discovered a desk before autonomous policy execution began.
        """
        if self.q_table is None:
            return

        raw = grid.get("table_confidence_persistent", None)
        if raw is None:
            raw = (grid.get("surface_height_map", {}) or {}).get("table_confidence_by_cell", {}) or {}

        conf_map = self._normalize_float_cell_map(raw or {})
        if not conf_map:
            return

        # If this same processed frame already carries fresh positive table
        # evidence for a cell, let the normal Bayesian table update consume it
        # instead of seeding and updating from the same bbox twice.
        current_positive = set(self._normalize_float_cell_map(grid.get("table_confidence", {}) or {}).keys())

        rows, cols = self.q_table.shape
        for (cx, cy), conf in conf_map.items():
            cx, cy = int(cx), int(cy)
            if not (0 <= cx < cols and 0 <= cy < rows):
                continue
            cell = (cx, cy)
            if cell in self._seeded_persistent_table_cells or cell in current_positive:
                continue
            self.q_table[cy, cx] = max(float(self.q_table[cy, cx]), float(np.clip(conf, 0.0, 1.0)))
            self._seeded_persistent_table_cells.add(cell)

    def _update_table_beliefs_from_observation(self, *, grid: dict) -> None:
        """Update q_table from per-frame table observations.

        Positive evidence is grid["table_confidence"], produced by desk bbox
        probing in the controller. Every cell hit by the same desk bbox receives
        that bbox confidence. Negative evidence is available for cells in
        grid["table_visible_mask"] that did not receive table_confidence in the
        current processed perception frame.
        """
        if self.q_table is None:
            return

        table_conf = self._normalize_float_cell_map(grid.get("table_confidence", {}) or {})
        table_visible = self._normalize_bool_cell_map(grid.get("table_visible_mask", {}) or {})
        table_cov = self._normalize_float_cell_map(grid.get("table_coverages", {}) or {})

        positive_cells = set(table_conf.keys())

        # Positive table observations.
        for cell, conf in table_conf.items():
            coverage = float(table_cov.get(cell, 1.0))
            self._update_table_cell_with_observation(
                cell=cell,
                table_detected=True,
                confidence=float(conf),
                coverage=coverage,
            )

        # Negative table observations for visible cells that did not receive a
        # desk/table bbox projection in this processed perception frame.
        for cell, vis in table_visible.items():
            if not vis or cell in positive_cells:
                continue
            coverage = float(table_cov.get(cell, 0.0))
            if coverage <= 0.0:
                continue
            self._update_table_cell_with_observation(
                cell=cell,
                table_detected=False,
                confidence=0.0,
                coverage=coverage,
            )

    def _update_table_cell_with_observation(
        self,
        *,
        cell: Cell,
        table_detected: bool,
        confidence: float,
        coverage: float,
    ) -> None:
        """Binary Bayesian update for q_table(cell)."""
        assert self.q_table is not None
        cx, cy = int(cell[0]), int(cell[1])
        rows, cols = self.q_table.shape
        if not (0 <= cx < cols and 0 <= cy < rows):
            return

        q_prior_table = float(self.q_table[cy, cx])

        # Positive table observations use the desk detector confidence directly.
        # For no-table observations, coverage weakens only the negative evidence:
        # with low coverage, absence of a desk bbox is weak evidence against table
        # presence. False-alarm probability remains coverage-independent because
        # the main practical failure mode is cell/bbox misassignment, not random
        # hallucination proportional to visible area.
        coverage = float(np.clip(coverage, 0.0, 1.0))
        p_fa = self.table_p_fa

        if table_detected:
            q_prior_table = max(q_prior_table, self.table_belief_prior)
            p_hit = float(confidence)
        else:
            p_hit = max(p_fa, coverage * self.table_no_detection_sensitivity)

        p_hit = float(np.clip(p_hit, 1e-6, 1.0 - 1e-6))

        prior = np.array([1.0 - q_prior_table, q_prior_table], dtype=np.float64)
        likelihood = (
            np.array([p_fa, p_hit], dtype=np.float64)
            if table_detected
            else np.array([1.0 - p_fa, 1.0 - p_hit], dtype=np.float64)
        )

        post_unnorm = likelihood * prior
        z = float(post_unnorm.sum())
        if z <= 1e-12:
            return
        full_post = post_unnorm / z
        self.q_table[cy, cx] = float(np.clip(float(full_post[1]), 0.0, 1.0))

    def _apply_table_context_from_beliefs(self) -> None:
        """Condition target beliefs on q_table without overwriting evidence.

        If the context prior changes from P_old(S) to P_new(S), and the current
        target posterior is q(S) ∝ L_target(S) P_old(S), then the corresponding
        posterior under the new context prior is

            q_new(S) ∝ q(S) · P_new(S) / P_old(S).

        This is the principled replacement for the previous one-shot blend:
        q_table itself determines the strength of the table prior effect.
        """
        assert self.q_s is not None
        if self.q_table is None or self._applied_table_context_prob is None:
            return

        rows, cols = self.q_table.shape
        for cy in range(rows):
            for cx in range(cols):
                old_t = float(self._applied_table_context_prob[cy, cx])
                new_t = float(self.q_table[cy, cx])
                if abs(new_t - old_t) <= 1e-6:
                    continue

                prior_old = self._context_prior(table_prob=old_t)
                prior_new = self._context_prior(table_prob=new_t)
                q = self.q_s[cy, cx, :].copy()
                adjusted = q * (prior_new / np.maximum(prior_old, 1e-12))
                self.q_s[cy, cx, :] = self._normalize(adjusted)
                self._applied_table_context_prob[cy, cx] = new_t

    def _update_beliefs_from_observation(
        self,
        *,
        grid: dict,
        current_cell: Optional[Cell],
        visible_mask: Dict[Cell, bool],
        coverages: Dict[Cell, float],
    ) -> None:
        assert self.q_s is not None
        rows, cols = self.q_s.shape[0], self.q_s.shape[1]

        target_events_by_cell = self._target_events_by_cell(grid)
        fixation_cell = self._current_fixation_cell(current_cell)

        for (cx, cy), vis in visible_mask.items():
            if not vis:
                continue
            if not (0 <= int(cx) < cols and 0 <= int(cy) < rows):
                continue

            coverage = float(coverages.get((cx, cy), 0.0))
            coverage = clamp(coverage, 0.0, 1.0)
            if coverage <= 0.0:
                continue

            events = target_events_by_cell.get((int(cx), int(cy)), [])
            is_fixation = self._is_fixation_evidence_cell(
                cell=(int(cx), int(cy)),
                fixation_cell=fixation_cell,
                coverage=coverage,
            )

            if not events:
                self._update_cell_with_observation(
                    cell=(int(cx), int(cy)),
                    obs_index=O_NO_TARGET,
                    confidence=0.0,
                    coverage=coverage,
                    use_center_likelihood=is_fixation,
                )
                continue

            # Multiple detector events in a cell are handled sequentially rather
            # than collapsed to one max-confidence class. With the coverage-dependent
            # likelihood, coverage is only relevant for no-detection evidence, so
            # positive detections use detector confidence directly.
            per_event_coverage = coverage / float(max(1, len(events)))
            for event in sorted(events, key=lambda e: float(e.get("confidence", 0.0)), reverse=True):
                class_name = self._event_class_name(event)
                conf = float(np.clip(float(event.get("confidence", 0.0)), 0.0, 1.0))

                if class_name == "target_bottle":
                    class_obs = O_BOTTLE
                elif class_name == "target_can":
                    class_obs = O_CAN
                else:
                    class_obs = O_GENERIC_TARGET

                if is_fixation:
                    obs_index = class_obs
                    eff_cov = per_event_coverage
                    use_center = True
                else:
                    # Peripheral detections are primarily used to attract a
                    # fixation. The class label is deliberately weakened unless
                    # this cell is the current fixation / high-coverage cell.
                    obs_index = O_GENERIC_TARGET
                    eff_cov = per_event_coverage
                    use_center = False

                self._update_cell_with_observation(
                    cell=(int(cx), int(cy)),
                    obs_index=obs_index,
                    confidence=conf,
                    coverage=eff_cov,
                    use_center_likelihood=use_center,
                )

                if (not is_fixation) and class_obs in (O_BOTTLE, O_CAN) and self.peripheral_class_weight > 0.0:
                    self._update_cell_with_observation(
                        cell=(int(cx), int(cy)),
                        obs_index=class_obs,
                        confidence=conf,
                        coverage=eff_cov * self.peripheral_class_weight,
                        use_center_likelihood=True,
                    )

    def _update_cell_with_observation(
        self,
        *,
        cell: Cell,
        obs_index: int,
        confidence: float,
        coverage: float,
        use_center_likelihood: bool,
    ) -> None:
        assert self.q_s is not None
        cx, cy = int(cell[0]), int(cell[1])

        # Coverage is now part of the observation likelihood rather than an
        # ex-post interpolation between prior and posterior. Positive target
        # detections use the detector confidence directly: once a bbox exists, it
        # is already positive evidence. For no-detection observations, coverage
        # weakens only the negative evidence by lowering the assumed probability
        # that the detector would have fired if a target had actually been present.
        # p_fa remains fixed because false detections in this setup are mainly
        # caused by bbox/cell misassignment near boundaries, not by a false-alarm
        # rate proportional to visible cell area.
        coverage = float(np.clip(coverage, 0.0, 1.0))

        if int(obs_index) == O_NO_TARGET:
            p_hit = max(self.p_fa, coverage * self.no_detection_sensitivity)
        else:
            p_hit = float(confidence)

        p_hit = float(np.clip(p_hit, 1e-6, 1.0 - 1e-6))

        A = self._A_center(p_hit) if use_center_likelihood else self._A_visible(p_hit)
        prior = self.q_s[cy, cx, :].copy()
        likelihood = A[int(obs_index), :]
        post_unnorm = likelihood * prior
        z = float(post_unnorm.sum())
        if z <= 1e-12:
            return

        full_post = post_unnorm / z
        self.q_s[cy, cx, :] = self._normalize(full_post)

    def _A_visible(self, p_hit: float) -> np.ndarray:
        """Peripheral/no-fixation likelihood: target evidence is generic."""
        p_hit = float(np.clip(p_hit, 1e-6, 1.0 - 1e-6))
        p_fa = self.p_fa

        A = np.zeros((4, 4), dtype=np.float64)

        # EMPTY
        A[O_NO_TARGET, S_EMPTY] = 1.0 - p_fa
        A[O_GENERIC_TARGET, S_EMPTY] = p_fa

        # UNKNOWN/BOTTLE/CAN all appear as generic object evidence away from
        # fixation. This is what creates the "detected object -> go fixate it"
        # behavior without prematurely classifying every peripheral detection.
        for s in (S_UNKNOWN, S_BOTTLE, S_CAN):
            A[O_NO_TARGET, s] = 1.0 - p_hit
            A[O_GENERIC_TARGET, s] = p_hit

        return self._normalize_likelihood_columns(A)

    def _A_center(self, p_hit: float) -> np.ndarray:
        """Fixation likelihood: class-specific bottle/can observations matter."""
        p_hit = float(np.clip(p_hit, 1e-6, 1.0 - 1e-6))
        p_fa = self.p_fa

        A = np.zeros((4, 4), dtype=np.float64)

        # False positives from empty cells: mostly generic, very rarely a class.
        A[O_NO_TARGET, S_EMPTY] = 1.0 - p_fa
        A[O_GENERIC_TARGET, S_EMPTY] = p_fa * 0.90
        A[O_BOTTLE, S_EMPTY] = p_fa * 0.05
        A[O_CAN, S_EMPTY] = p_fa * 0.05

        # Unknown target at fixation: often generic, sometimes class-resolved.
        unk_class = self.unknown_class_report_prob
        A[O_NO_TARGET, S_UNKNOWN] = 1.0 - p_hit
        A[O_GENERIC_TARGET, S_UNKNOWN] = p_hit * (1.0 - unk_class)
        A[O_BOTTLE, S_UNKNOWN] = p_hit * unk_class * 0.5
        A[O_CAN, S_UNKNOWN] = p_hit * unk_class * 0.5

        # Known class states: high probability of the correct class, small
        # probability of generic target or class confusion.
        class_report = self.class_report_prob
        acc = self.class_accuracy
        generic_part = p_hit * (1.0 - class_report)
        class_part = p_hit * class_report

        A[O_NO_TARGET, S_BOTTLE] = 1.0 - p_hit
        A[O_GENERIC_TARGET, S_BOTTLE] = generic_part
        A[O_BOTTLE, S_BOTTLE] = class_part * acc
        A[O_CAN, S_BOTTLE] = class_part * (1.0 - acc)

        A[O_NO_TARGET, S_CAN] = 1.0 - p_hit
        A[O_GENERIC_TARGET, S_CAN] = generic_part
        A[O_BOTTLE, S_CAN] = class_part * (1.0 - acc)
        A[O_CAN, S_CAN] = class_part * acc

        return self._normalize_likelihood_columns(A)

    @staticmethod
    def _normalize_likelihood_columns(A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=np.float64)
        col_sums = A.sum(axis=0, keepdims=True)
        col_sums = np.maximum(col_sums, 1e-12)
        return A / col_sums

    @staticmethod
    def _normalize(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        q = np.maximum(q, 1e-12)
        z = float(q.sum())
        if z <= 1e-12:
            return np.ones_like(q) / float(q.size)
        return q / z

    # ---------------------------------------------------------------------
    # Action selection / EFE-style scoring
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
    ):
        assert self.grid_spec is not None
        assert self.q_s is not None

        candidates = self._all_cells()
        table_cells = table_cells or set()
        scores = [] if return_scores else None

        best_G = float("inf")
        best = (candidates[0], self.yaw_options[0])

        cx0, cy0 = current_cell
        x0, y0 = self._cell_center_xy(cx0, cy0)

        for cell in candidates:
            tx, ty = self._cell_center_xy(cell[0], cell[1])
            dist = math.hypot(tx - x0, ty - y0)
            distance_cost = self.distance_cost_epsilon * dist

            for yaw_idx, yaw in enumerate(self.yaw_options):
                comps = self._efe_components(
                    target_cell=cell,
                    target_yaw_index=yaw_idx,
                    table_cells=table_cells,
                )
                risk = comps["risk"]
                IG = comps["IG"]
                G = float(risk - self.beta_info * IG + distance_cost)

                if return_scores:
                    scores.append({
                        "cell": [int(cell[0]), int(cell[1])],
                        "yaw": float(yaw),
                        "risk": float(risk),
                        "utility": float(comps["utility"]),
                        "IG": float(IG),
                        "table_bonus": float(comps["table_bonus"]),
                        "center_classification_need": float(comps["center_classification_need"]),
                        "center_p_present": float(comps["center_p_present"]),
                        "center_class_confidence": float(comps["center_class_confidence"]),
                        "distance": float(dist),
                        "distance_cost": float(distance_cost),
                        "G": float(G),
                    })

                if G < best_G:
                    best_G = G
                    best = (cell, yaw)

        return (best, scores) if return_scores else best

    def _efe_components(
        self,
        *,
        target_cell: Cell,
        target_yaw_index: int,
        table_cells: Optional[set[Cell]] = None,
    ) -> Dict[str, float]:
        assert self.q_s is not None
        table_cells = table_cells or set()

        cov_rc = self._merged_candidate_coverages_rc(
            target_cell=target_cell,
            target_yaw_index=int(target_yaw_index),
            table_cells=table_cells,
        )

        A_visible = self._A_visible(self.planning_p_hit)
        A_center = self._A_center(self.planning_p_hit)

        IG = 0.0
        table_bonus = 0.0

        for (r, c), frac in cov_rc.items():
            frac = float(frac)
            if frac <= self.min_coverage_for_ig:
                continue

            cy = int(r)
            cx = int(c)
            cell = (cx, cy)
            q = self.q_s[cy, cx, :]
            A = A_center if cell == target_cell else A_visible
            IG += frac * self._expected_information_gain(q, A)

            if cell in table_cells:
                # Extra value for inspecting perceived table cells that still
                # have target/class uncertainty. This is deliberately grounded
                # in the controller's table_cells only; no neighbors or GT table
                # geometry are injected.
                table_bonus += frac * self._present_prob(q) * self._classification_need(q)

        cx_t, cy_t = int(target_cell[0]), int(target_cell[1])
        q_t = self.q_s[cy_t, cx_t, :]
        center_utility = self._center_classification_utility(q_t, is_table=(target_cell in table_cells))
        utility = center_utility + self.table_exploration_weight * table_bonus

        # In the old code this term was named risk and minimized. Here risk is
        # negative utility; the final score is still G = risk - beta_info*IG + d.
        risk = -float(utility)

        return {
            "risk": float(risk),
            "utility": float(utility),
            "IG": float(IG),
            "table_bonus": float(table_bonus),
            "center_classification_need": float(self._classification_need(q_t)),
            "center_p_present": float(self._present_prob(q_t)),
            "center_class_confidence": float(self._classification_confidence_from_q(q_t)),
        }

    def _expected_information_gain(self, q: np.ndarray, A: np.ndarray) -> float:
        q = self._normalize(q)
        p_o = A @ q
        ig = 0.0
        for o in range(A.shape[0]):
            po = float(p_o[o])
            if po <= 1e-12:
                continue
            post = A[o, :] * q
            z = float(post.sum())
            if z <= 1e-12:
                continue
            post = post / z
            kl = float(np.sum(post * (np.log(post + 1e-12) - np.log(q + 1e-12))))
            ig += po * kl
        return float(ig)

    def _center_classification_utility(self, q: np.ndarray, *, is_table: bool) -> float:
        q = self._normalize(q)
        A = self._A_center(self.planning_p_hit)
        p_o = A @ q

        p_class_obs = float(p_o[O_BOTTLE] + p_o[O_CAN])
        p_generic_obs = float(p_o[O_GENERIC_TARGET])
        need = self._classification_need(q)
        unknown_mass = float(q[S_UNKNOWN])

        utility = 0.0
        utility += self.classification_preference_strength * need * p_class_obs
        utility += self.generic_target_preference_strength * unknown_mass * p_generic_obs

        return float(utility)

    @staticmethod
    def _present_prob(q: np.ndarray) -> float:
        return float(np.sum(q[1:]))

    def _classification_need(self, q: np.ndarray) -> float:
        q = self._normalize(q)
        p_present = self._present_prob(q)
        if p_present <= 1e-12:
            return 0.0
        class_conf = max(float(q[S_BOTTLE]), float(q[S_CAN])) / max(p_present, 1e-12)
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

    def _current_fixation_cell(self, current_cell: Optional[Cell]) -> Optional[Cell]:
        if self.navigator is not None and self.navigator.has_target():
            target = getattr(self.navigator, "target_cell", None)
            parsed = self._parse_cell(target)
            if parsed is not None:
                return parsed
        return current_cell

    def _is_fixation_evidence_cell(
        self,
        *,
        cell: Cell,
        fixation_cell: Optional[Cell],
        coverage: float,
    ) -> bool:
        if fixation_cell is not None and (int(cell[0]), int(cell[1])) == (int(fixation_cell[0]), int(fixation_cell[1])):
            return True
        return float(coverage) >= self.classification_coverage_threshold

    def _target_events_by_cell(self, grid: dict) -> Dict[Cell, List[dict]]:
        by_cell: Dict[Cell, List[dict]] = {}
        events = list(grid.get("target_detections", []) or [])

        # Backstop for future/older payloads where all detections are present but
        # target_detections is missing.
        if not events:
            for event in grid.get("detections_all", []) or []:
                if self._is_target_event(event):
                    events.append(event)

        # Backwards-compatible fallback to det_confidence_multiclass/map if an
        # event list is not available. This loses class labels, but keeps the
        # policy usable on old logs/controllers.
        if not events:
            det_conf = grid.get("det_confidence_multiclass", None) or grid.get("det_confidence", {}) or {}
            meta = grid.get("det_metadata_multiclass", {}) or {}
            for raw_cell, conf in det_conf.items():
                cell = self._parse_cell(raw_cell)
                if cell is None:
                    continue
                m = meta.get(raw_cell, None)
                if m is None and isinstance(raw_cell, tuple):
                    m = meta.get((int(raw_cell[0]), int(raw_cell[1])), None)
                class_name = self._event_class_name(m or {})
                events.append({"cell": cell, "confidence": float(conf), "class_name": class_name, "_target_generic": True})

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
        if bool(event.get("_target_generic", False)):
            return True
        class_name = self._event_class_name(event)
        if class_name in TARGET_CLASS_NAMES:
            return True
        class_id = event.get("class_id", None)
        try:
            return int(class_id) in {0, 1}
        except Exception:
            return False

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
            out.append((cx, cy, float(p_present[cy, cx]), float(self._classification_confidence_from_q(q)), float(self._classification_need(q))))
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
            f"[AIF-TABLE GRID] step={step} cc={cc} "
            f"cover={len(coverages)} visible={num_vis} table_cells={len(table_cells)}",
            flush=True,
        )

        top_cov = sorted(coverages.items(), key=lambda kv: kv[1], reverse=True)[:9]
        for (cx, cy), frac in top_cov:
            q = self.q_s[cy, cx, :] if self.q_s is not None else np.zeros(4)
            print(
                f"  cell=({cx},{cy}) cov={frac*100:5.1f}% vis={visible_mask.get((cx, cy), None)} "
                f"table={(cx, cy) in table_cells} p={self._present_prob(q):.2f} "
                f"q=[empty={q[S_EMPTY]:.2f}, unk={q[S_UNKNOWN]:.2f}, "
                f"bot={q[S_BOTTLE]:.2f}, can={q[S_CAN]:.2f}]",
                flush=True,
            )

        top_bel = self._top_beliefs(k=5)
        if top_bel:
            print(
                "  top p_present: " + ", ".join(
                    [f"({cx},{cy}) p={p:.2f} cls={cls:.2f} need={need:.2f}" for (cx, cy, p, cls, need) in top_bel]
                ),
                flush=True,
            )

        if self.navigator is not None and self.navigator.has_target():
            print(f"  target={self.navigator.target_cell} target_yaw={self.navigator.target_yaw:.3f}", flush=True)

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

    def _q_table_cells_payload(self, threshold: float = 0.05) -> List[dict]:
        if self.q_table is None:
            return []
        rows, cols = self.q_table.shape
        out = []
        for cy in range(rows):
            for cx in range(cols):
                q = float(self.q_table[cy, cx])
                if q < float(threshold):
                    continue
                out.append({"cell": [int(cx), int(cy)], "p_table": q})
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
        if self.q_s is None:
            raise RuntimeError("q_s is None; cannot save snapshot")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"snapshot_step{int(step):06d}_{reason}_{ts}.npz"
        path = os.path.join(snap_dir, fname)

        q_present = self.q_s[:, :, 1:].sum(axis=2)
        q_class_conf = np.zeros_like(q_present)
        q_need = np.zeros_like(q_present)
        for cy in range(self.q_s.shape[0]):
            for cx in range(self.q_s.shape[1]):
                q = self.q_s[cy, cx, :]
                q_class_conf[cy, cx] = self._classification_confidence_from_q(q)
                q_need[cy, cx] = self._classification_need(q)

        chosen_cell = chosen[0]
        meta = {
            "step": int(step),
            "reason": str(reason),
            "state_names": STATE_NAMES,
            "obs_names": OBS_NAMES,
            "pose": {"x": float(pose["x"]), "y": float(pose["y"]), "yaw": float(pose["yaw"])},
            "current_cell": list(current_cell) if current_cell is not None else None,
            "chosen_cell": [int(chosen_cell[0]), int(chosen_cell[1])] if chosen_cell is not None else None,
            "chosen_yaw": float(chosen[1]),
            "found_cells_debug_only": sorted([[int(c[0]), int(c[1])] for c in found_cells]),
            "table_cells": sorted([[int(c[0]), int(c[1])] for c in (table_cells or set())]),
            "q_table_cells": self._q_table_cells_payload(),
            "classified_confidence_threshold": float(self.classified_confidence_threshold),
            "classified_cells": self._classified_cells_payload(),
            "visible_mask": {f"{k[0]},{k[1]}": bool(v) for k, v in visible_mask.items()},
            "coverages": {f"{k[0]},{k[1]}": float(v) for k, v in coverages.items()},
            "params": {
                "prior_target_floor": float(self.prior_target_floor),
                "prior_target_table": float(self.prior_target_table),
                "table_belief_prior": float(self.table_belief_prior),
                "table_p_fa": float(self.table_p_fa),
                "table_no_detection_sensitivity": float(self.table_no_detection_sensitivity),
                "p_fa": float(self.p_fa),
                "no_detection_sensitivity": float(self.no_detection_sensitivity),
                "beta_info": float(self.beta_info),
                "classification_preference_strength": float(self.classification_preference_strength),
                "table_exploration_weight": float(self.table_exploration_weight),
                "distance_cost_epsilon": float(self.distance_cost_epsilon),
                "planning_p_hit": float(self.planning_p_hit),
            },
            "scored_actions": scored_actions,
        }

        np.savez_compressed(
            path,
            q_s=self.q_s.astype(np.float32),
            q_table=(self.q_table if self.q_table is not None else np.zeros_like(q_present)).astype(np.float32),
            q_present=q_present.astype(np.float32),
            q_class_confidence=q_class_conf.astype(np.float32),
            q_classification_need=q_need.astype(np.float32),
            table_map=self._table_map_array(table_cells or set()).astype(np.uint8),
            meta_json=np.array(json.dumps(meta), dtype=object),
        )
        print(f"[SNAPSHOT] wrote {path}", flush=True)
        return path

    def _table_map_array(self, table_cells: set[Cell]) -> np.ndarray:
        assert self.grid_spec is not None
        arr = np.zeros((self.grid_spec.rows, self.grid_spec.cols), dtype=np.uint8)
        for cx, cy in table_cells:
            if 0 <= int(cx) < self.grid_spec.cols and 0 <= int(cy) < self.grid_spec.rows:
                arr[int(cy), int(cx)] = 1
        return arr