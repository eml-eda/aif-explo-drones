# policies/aif_zoi_v2.py

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from configuration.aif_zoi_v2_params import AIFZOIV2Params
from env.env_manager import FoVSpec, GridEnvManager, GridSpec
from env.surface_projection import event_bbox_xyxy
from policies.planner_time import NavigatorTimeModel
from policies.zoi_v2_model import (
    CELL_FLOOR,
    CELL_TABLE,
    CELL_TYPE_NAMES,
    CELL_UNOBSERVED,
    O_BOTTLE,
    O_CAN,
    O_NOTHING,
    OBS_NAMES,
    S_BOTTLE,
    S_CAN,
    S_EMPTY,
    STATE_NAMES,
    bayes_update,
    expected_information_gain,
    likelihood_matrix,
    object_visibility,
    predictive,
    prior_present,
    prior_vector,
    reweight,
)
from utils.navigator import Navigator

# Navigator settings, unchanged from aif_zoi.
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

Cell = Tuple[int, int]  # policy/report convention: (cx, cy)

TARGET_CLASS_NAMES = {"target_bottle", "target_can"}
DESK_CLASS_NAMES = {"desk"}
CLASS_NAME_TO_OBS = {"target_bottle": O_BOTTLE, "target_can": O_CAN}


class AIFZOIV2Policy:
    """Table-aware active-inference saccade policy with a derived/measured model.

    Same planner and controller-facing structure as AIFZOIPolicy. The belief
    model is replaced by the one in policies/zoi_v2_model.py:

    - per-cell belief over {empty, bottle, can};
    - prior from the cell surface type (unobserved, floor, table), with the
      evidence kept by q * P_new / P_old when a cell's type becomes known;
    - one observation per visible cell per processed frame (nothing, "bottle"
      box, "can" box) with a single likelihood A(v), where v is the probability
      that a target present in the cell is inside the camera image: the cell's
      coverage at table and at floor height, mixed by the probability that the
      target is on a table given the cell type;
    - a cell is typed table when its centre, projected at table height, falls
      inside a desk box (the policy's own table map; the "touched" rule uses the
      controller's map instead, to reproduce older campaigns);
    - a cell is typed floor only when its centre has been seen at table height
      without the cell entering the table map;
    - detector confidence never enters the likelihood;
    - the planner's information gain and classification utility use the same
      A(w) as the belief update.
    """

    def __init__(
        self,
        debug: Optional[dict] = None,
        *,
        params: Optional[AIFZOIV2Params] = None,
        # Action scoring.
        beta_info: float = 1.0,
        classification_preference_strength: float = 2.0,
        table_exploration_weight: float = 0.0,
        distance_cost_epsilon: float = 0.04,
        # Replanning/nav.
        yaw_options: Optional[List[float]] = None,
        replan_cooldown_s: float = 0.4,
        classified_confidence_threshold: float = 0.80,
        # Detector period of the controller (DETECTION_EVERY_S), for the "rate" planner.
        detector_period_s: float = 0.625,
    ):
        self.debug = debug or {}
        self.every_n = int(self.debug.get("every_n", 50))

        self.params = params if params is not None else AIFZOIV2Params()
        self.table_context_enabled = bool(self.params.table_context_enabled)

        self.beta_info = float(beta_info)
        self.classification_preference_strength = float(
            classification_preference_strength)
        self.table_exploration_weight = float(table_exploration_weight)
        self.distance_cost_epsilon = float(distance_cost_epsilon)

        self.replan_cooldown_s = float(replan_cooldown_s)
        # Flight time of a move, derived from the navigator's control law.
        self.time_model = NavigatorTimeModel.from_navigator(NAVIGATOR_KWARGS, detector_period_s)
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

        # q_s[cy, cx, :] is Cat(empty, bottle, can).
        self.q_s: Optional[np.ndarray] = None

        # cell_type[cy, cx] is CELL_UNOBSERVED / CELL_FLOOR / CELL_TABLE, and
        # _applied_prior[cy, cx, :] is the prior that q_s currently encodes.
        self.cell_type: Optional[np.ndarray] = None
        self._applied_prior: Optional[np.ndarray] = None

        # Table map for the "centre" rule: cells whose centre, projected at
        # table height, fell inside a desk box on some processed frame.
        self._table_map: set[Cell] = set()

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

        # Candidate FoV coverage cache, exactly mirroring the old policy idea:
        # cache floor/table projections for each candidate cell and yaw. Runtime
        # table_cells only select which surface-specific coverage to use per cell.
        self._candidate_coverage_cache: Dict[Tuple[int, int,
                                                   int], Dict[str, Dict[Tuple[int, int], float]]] = {}

    def parameters(self) -> dict:
        """Every value the policy uses, for the run logs."""
        return {
            "policy": "aif_zoi_v2",
            "model": self.params.as_dict(),
            "observation_model": {
                "visibility": "type-mixed plane coverage",
                "floor_typing": "cell centre in table-plane footprint",
                "table_map": (
                    "cell centre in desk box at table height"
                    if self.params.table_map_rule == "centre"
                    else "controller map (cells hit by desk-box probes)"
                ),
            },
            "planning": {
                "beta_info": float(self.beta_info),
                "classification_preference_strength": float(self.classification_preference_strength),
                "table_exploration_weight": float(self.table_exploration_weight),
                "distance_cost_epsilon": float(self.distance_cost_epsilon),
                "yaw_options": [float(y) for y in self.yaw_options],
                "replan_cooldown_s": float(self.replan_cooldown_s),
                "classified_confidence_threshold": float(self.classified_confidence_threshold),
                "navigator": dict(NAVIGATOR_KWARGS),
                "objective": self.params.planner_objective,
                "terms": self.params.planner_terms,
                "time_model": self.time_model.as_dict() if self.params.planner_objective == "rate" else None,
            },
        }

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
        table_cells = self._table_cells(observation, update=perception_step)
        if grid.get("coverage_threshold", None) is not None:
            self._coverage_threshold = float(grid["coverage_threshold"])

        # Cell types first (table cells from the table map, floor cells from
        # this frame's table-height view), so the prior is
        # current before this frame's target observations are assimilated.
        # Floor typing only happens on processed frames, when the table map
        # reflects this frame's desk detections.
        if self.table_context_enabled:
            self._update_cell_types(
                table_cells=table_cells,
                pose=pose if perception_step else None,
            )

        # Only target-belief updates on processed perception steps because the
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
            (next_cell, next_yaw), scored_actions = self._pick_next_action(
                current_cell=current_cell,
                table_cells=table_cells,
                return_scores=True,
                pose=pose,
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
                    f"[AIF-ZOI-V2] picked target cell={next_cell} yaw={next_yaw:.3f} "
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
        table_cells = self._table_cells(observation, update=False)
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

        # Without table context every cell is floor from the start.
        initial_type = CELL_UNOBSERVED if self.table_context_enabled else CELL_FLOOR
        rows, cols = self.grid_spec.rows, self.grid_spec.cols
        self.cell_type = np.full((rows, cols), initial_type, dtype=np.int8)
        initial_prior = self._prior_for_type(initial_type)
        self.q_s = np.tile(initial_prior.reshape(1, 1, 3), (rows, cols, 1))
        self._applied_prior = self.q_s.copy()

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
    # Belief model (see policies/zoi_v2_model.py)
    # ---------------------------------------------------------------------

    def _prior_for_type(self, cell_type: int) -> np.ndarray:
        return prior_vector(
            cell_type,
            lambda_table=self.params.lambda_table,
            lambda_floor=self.params.lambda_floor,
            rho=self.params.rho,
        )

    def _likelihood(self, visibility: float) -> np.ndarray:
        return likelihood_matrix(
            visibility,
            recall=self.params.recall,
            class_accuracy=self.params.class_accuracy,
            false_alarm=self.params.false_alarm,
        )

    def _visibility(self, cell: Cell, v_table: float, v_floor: float) -> float:
        """P(target in image | target present) for a cell, given its type.

        Used by both the belief update and planning.
        """
        assert self.cell_type is not None
        return object_visibility(
            int(self.cell_type[int(cell[1]), int(cell[0])]),
            v_table,
            v_floor,
            lambda_table=self.params.lambda_table,
            lambda_floor=self.params.lambda_floor,
            rho=self.params.rho,
        )

    def _table_cells(self, observation: dict, *, update: bool) -> set[Cell]:
        """Cells typed table, under the configured table-map rule.

        "centre": the policy's own map; with update=True this frame's desk
        boxes are added first. "touched": the controller's map, where every cell
        hit by a desk-box probe is a table cell.
        """
        if self.params.table_map_rule == "touched":
            return self._table_cells_from_grid(observation.get("grid", {}) or {})
        if update:
            self._add_desk_boxes_to_table_map(observation)
        return set(self._table_map)

    def _add_desk_boxes_to_table_map(self, observation: dict) -> None:
        """Add the cells whose centre, at table height, is inside a desk box.

        The desk top is at table height, so a cell centre lies on a detected desk
        when its table-plane image position falls inside the desk box.
        """
        camera = observation.get("camera", {}) or {}
        img_w = float(camera.get("width", 0) or 0)
        img_h = float(camera.get("height", 0) or 0)
        if img_w <= 0.0 or img_h <= 0.0:
            return
        boxes = []
        for event in (observation.get("grid", {}) or {}).get("detections_all", []) or []:
            if self._event_class_name(event) not in DESK_CLASS_NAMES:
                continue
            bbox = event_bbox_xyxy(event)
            if bbox is not None:
                boxes.append(bbox)
        if not boxes:
            return

        assert self.grid_spec is not None
        pose = observation["pose"]
        for cy in range(self.grid_spec.rows):
            for cx in range(self.grid_spec.cols):
                if (cx, cy) in self._table_map:
                    continue
                u, v = self._table_plane_pixel((cx, cy), pose, img_w, img_h)
                if any(x1 <= u <= x2 and y1 <= v <= y2 for (x1, y1, x2, y2) in boxes):
                    self._table_map.add((cx, cy))

    def _table_plane_pixel(self, cell: Cell, pose: dict, img_w: float, img_h: float) -> Tuple[float, float]:
        """Image position of the cell centre at table height.

        Inverse of GridEnvManager.detection_cell_from_bbox: the footprint maps
        linearly to the image, left edge (+half_left) at u = 0 and front edge
        (+forward) at v = 0.
        """
        assert self.table_fov_spec is not None
        x, y = self._cell_center_xy(int(cell[0]), int(cell[1]))
        dx, dy = x - float(pose["x"]), y - float(pose["y"])
        yaw = float(pose["yaw"])
        c, s = math.cos(yaw), math.sin(yaw)
        xb = c * dx + s * dy
        yb = -s * dx + c * dy
        fov = self.table_fov_spec
        un = (fov.half_left - yb) / (fov.half_left + fov.half_right)
        vn = (fov.forward - xb) / (fov.forward + fov.backward)
        return un * img_w, vn * img_h

    def _centre_in_table_footprint(self, cell: Cell, pose: dict) -> bool:
        """Whether the cell centre is inside the camera footprint at table height."""
        assert self.table_fov_spec is not None
        x, y = self._cell_center_xy(int(cell[0]), int(cell[1]))
        dx, dy = x - float(pose["x"]), y - float(pose["y"])
        yaw = float(pose["yaw"])
        c, s = math.cos(yaw), math.sin(yaw)
        # World -> body frame (inverse of GridEnvManager._fov_polygon_world).
        xb = c * dx + s * dy
        yb = -s * dx + c * dy
        fov = self.table_fov_spec
        return (-fov.backward <= xb <= fov.forward) and (-fov.half_right <= yb <= fov.half_left)

    def _update_cell_types(
        self,
        *,
        table_cells: set[Cell],
        pose: Optional[dict],
    ) -> None:
        """Type cells as table/floor and move their prior without losing evidence.

        Table: the cell is in the controller's persistent table map (monotone,
        so a table cell never reverts). Floor: on a processed frame (pose given),
        the cell centre is inside the camera footprint at table height and the
        cell is not in the table map, so a desk covering the cell centre would
        have been in view. A floor cell can later become a table cell.
        """
        assert self.cell_type is not None
        rows, cols = self.cell_type.shape

        new_types: Dict[Cell, int] = {}
        for (cx, cy) in table_cells:
            if 0 <= int(cx) < cols and 0 <= int(cy) < rows:
                if int(self.cell_type[int(cy), int(cx)]) != CELL_TABLE:
                    new_types[(int(cx), int(cy))] = CELL_TABLE

        if pose is not None:
            for cy in range(rows):
                for cx in range(cols):
                    if (cx, cy) in table_cells or int(self.cell_type[cy, cx]) != CELL_UNOBSERVED:
                        continue
                    if self._centre_in_table_footprint((cx, cy), pose):
                        new_types[(cx, cy)] = CELL_FLOOR

        for (cx, cy), cell_type in new_types.items():
            self._set_cell_type((cx, cy), cell_type)

    def _set_cell_type(self, cell: Cell, cell_type: int) -> None:
        assert self.q_s is not None and self.cell_type is not None
        assert self._applied_prior is not None
        cx, cy = int(cell[0]), int(cell[1])
        prior_new = self._prior_for_type(cell_type)
        self.q_s[cy, cx, :] = reweight(
            self.q_s[cy, cx, :], self._applied_prior[cy, cx, :], prior_new)
        self._applied_prior[cy, cx, :] = prior_new
        self.cell_type[cy, cx] = int(cell_type)

    def _update_beliefs_from_observation(
        self,
        *,
        grid: dict,
        visible_mask: Dict[Cell, bool],
        coverages: Dict[Cell, float],
    ) -> None:
        """One Bayesian update per visible cell with that cell's observation."""
        assert self.q_s is not None
        rows, cols = self.q_s.shape[0], self.q_s.shape[1]

        target_events_by_cell = self._target_events_by_cell(grid)

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
            if not vis:
                continue
            if not (0 <= int(cx) < cols and 0 <= int(cy) < rows):
                continue
            cell = (int(cx), int(cy))
            visible_cells.add(cell)

            visibility = self._visibility(
                cell,
                float(table_plane.get(cell, 0.0)),
                float(floor_plane.get(cell, 0.0)),
            )
            obs_index = self._cell_observation(
                target_events_by_cell.get(cell, []))
            self.q_s[cell[1], cell[0], :] = bayes_update(
                self.q_s[cell[1], cell[0], :],
                self._likelihood(visibility),
                obs_index,
            )

        for cell, events in target_events_by_cell.items():
            if cell not in visible_cells:
                self._dropped_target_events += len(events)

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
        pose: Optional[dict] = None,
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

        # "rate": rank by (beta*IG + utility) / T, T the navigator's flight time plus
        # one detector period; ties go to the shorter flight.
        use_rate = self.params.planner_objective == "rate"
        # "info_only": no classification bonus; the ranking uses the information gain alone.
        info_only = self.params.planner_terms == "info_only"
        if use_rate and pose is None:
            pose = {"x": x0, "y": y0, "yaw": 0.0}
        best_key = (-float("inf"), -float("inf"))

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
                risk = 0.0 if info_only else comps["risk"]
                goal = 0.0 if info_only else comps["utility"]
                IG = comps["IG"]
                G = float(risk - self.beta_info * IG + distance_cost)
                if use_rate:
                    T, t_turn, t_move = self.time_model.time_to(pose, (tx, ty), yaw)
                    rate = float((self.beta_info * IG + goal) / T)

                if return_scores:
                    record = {
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
                    }
                    if use_rate:
                        record.update({"T": float(T), "T_turn": float(t_turn), "T_move": float(t_move),
                                       "rate": rate})
                    scores.append(record)

                if use_rate:
                    key = (rate, -T)
                    if key > best_key:
                        best_key = key
                        best = (cell, yaw)
                elif G < best_G:
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
        floor_cov_rc, table_cov_rc = self._cached_candidate_coverages_rc(
            target_cell=target_cell,
            target_yaw_index=int(target_yaw_index),
        )

        IG = 0.0
        table_bonus = 0.0

        for (r, c), frac in cov_rc.items():
            frac = float(frac)
            # Same cells the controller would update: coverage at or above its
            # visible-mask threshold.
            if frac < self._coverage_threshold:
                continue

            cy = int(r)
            cx = int(c)
            cell = (cx, cy)
            q = self.q_s[cy, cx, :]
            IG += self._cell_information_gain(
                cell,
                float(table_cov_rc.get((cy, cx), 0.0)),
                float(floor_cov_rc.get((cy, cx), 0.0)),
            )

            if cell in table_cells:
                # Extra value for inspecting perceived table cells that still
                # have target/class uncertainty. This is deliberately grounded
                # in the controller's table_cells only; no neighbors or GT table
                # geometry are injected.
                table_bonus += frac * \
                    self._present_prob(q) * self._classification_need(q)

        cx_t, cy_t = int(target_cell[0]), int(target_cell[1])
        q_t = self.q_s[cy_t, cx_t, :]
        center_utility = self._center_utility(
            (cx_t, cy_t),
            float(table_cov_rc.get((cy_t, cx_t), 0.0)),
            float(floor_cov_rc.get((cy_t, cx_t), 0.0)),
            is_table=(target_cell in table_cells),
        )
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

    def _cell_information_gain(self, cell: Cell, w_table: float, w_floor: float) -> float:
        """Expected information gain of one view of a cell with these plane coverages.

        Visibility from the candidate pose enters through A(v), the same
        likelihood as the update, so the information gain is not weighted by
        coverage again.
        """
        q = self.q_s[int(cell[1]), int(cell[0]), :]
        visibility = self._visibility(cell, w_table, w_floor)
        return expected_information_gain(q, self._likelihood(visibility))

    def _center_utility(self, cell: Cell, w_table: float, w_floor: float, *, is_table: bool) -> float:
        """Classification utility of the destination cell."""
        q = self.q_s[int(cell[1]), int(cell[0]), :]
        visibility = self._visibility(cell, w_table, w_floor)
        return self._center_classification_utility(q, visibility=visibility, is_table=is_table)

    def _center_classification_utility(self, q: np.ndarray, *, visibility: float, is_table: bool) -> float:
        q = self._normalize(q)
        p_o = predictive(q, self._likelihood(visibility))

        p_class_obs = float(p_o[O_BOTTLE] + p_o[O_CAN])
        need = self._classification_need(q)

        return float(self.classification_preference_strength * need * p_class_obs)


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
            f"[AIF-ZOI-V2 GRID] step={step} cc={cc} "
            f"cover={len(coverages)} visible={num_vis} table_cells={len(table_cells)} "
            f"dropped_target_events={self._dropped_target_events}",
            flush=True,
        )

        top_cov = sorted(coverages.items(),
                         key=lambda kv: kv[1], reverse=True)[:9]
        for (cx, cy), frac in top_cov:
            q = self.q_s[cy, cx, :] if self.q_s is not None else np.zeros(3)
            ctype = CELL_TYPE_NAMES[int(self.cell_type[cy, cx])] if self.cell_type is not None else "?"
            print(
                f"  cell=({cx},{cy}) cov={frac*100:5.1f}% vis={visible_mask.get((cx, cy), None)} "
                f"type={ctype} p={self._present_prob(q):.2f} "
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

    def _prior_present_map(self) -> np.ndarray:
        assert self.cell_type is not None
        out = np.zeros(self.cell_type.shape, dtype=np.float64)
        for cy in range(out.shape[0]):
            for cx in range(out.shape[1]):
                out[cy, cx] = prior_present(
                    int(self.cell_type[cy, cx]),
                    lambda_table=self.params.lambda_table,
                    lambda_floor=self.params.lambda_floor,
                    rho=self.params.rho,
                )
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
                q_class_conf[cy,
                             cx] = self._classification_confidence_from_q(q)
                q_need[cy, cx] = self._classification_need(q)

        chosen_cell = chosen[0]
        meta = {
            "step": int(step),
            "reason": str(reason),
            "state_names": STATE_NAMES,
            "obs_names": OBS_NAMES,
            "cell_type_names": CELL_TYPE_NAMES,
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
            "params": self.parameters(),
            "scored_actions": scored_actions,
        }

        np.savez_compressed(
            path,
            q_s=self.q_s.astype(np.float32),
            cell_type=(self.cell_type if self.cell_type is not None else np.zeros_like(
                q_present)).astype(np.uint8),
            prior_present=self._prior_present_map().astype(np.float32),
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
