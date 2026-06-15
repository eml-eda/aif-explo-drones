# policies/aif_saccade.py

from __future__ import annotations

import math
from typing import Dict, Tuple, Optional, List
import os
import json
from datetime import datetime

import numpy as np

from env.env_manager import GridSpec, FoVSpec, GridEnvManager
from utils.navigator import Navigator, clamp

Cell = Tuple[int, int]  # (cx, cy)


class AIFSaccadePolicy:
    def __init__(
        self,
        debug: Optional[dict] = None,
        *,
        prior_present: float = 0.05,
        p_fa: float = 0.05,
        conf_floor: float = 0.25,
        beta_info: float = 1.0,
        distance_cost_epsilon: float = 0.04,
        prefer_detect_strength: float = 0.95,
        yaw_options: Optional[List[float]] = None,
        planning_p_hit: float = 0.85,
        replan_cooldown_s: float = 0.4,
        target_exclusion_detection_frames: int = 2,
    ):
        self.debug = debug or {}
        self.every_n = int(self.debug.get("every_n", 50))

        self.prior_present = float(prior_present)

        self.p_fa = float(p_fa)
        self.conf_floor = float(conf_floor)

        self.beta_info = float(beta_info)
        self.distance_cost_epsilon = float(distance_cost_epsilon)
        self.planning_p_hit = float(planning_p_hit)
        self.replan_cooldown_s = float(replan_cooldown_s)
        self.target_exclusion_detection_frames = int(max(1, target_exclusion_detection_frames))

        p_det = float(prefer_detect_strength)
        self.C = np.array([1.0 - p_det, p_det], dtype=np.float64)

        self.grid_spec: Optional[GridSpec] = None
        self.fov_spec: Optional[FoVSpec] = None
        self.floor_fov_spec: Optional[FoVSpec] = None
        self.table_fov_spec: Optional[FoVSpec] = None
        self.env_mgr: Optional[GridEnvManager] = None
        self.env_mgr_floor: Optional[GridEnvManager] = None
        self.env_mgr_table: Optional[GridEnvManager] = None
        self.navigator: Optional[Navigator] = None

        self.q_s: Optional[np.ndarray] = None  # (rows, cols, 2)

        if yaw_options is None:
            yaw_options = [k * (math.pi / 4.0) for k in range(8)]
        self.yaw_options = [float(y) for y in yaw_options]

        self._snap_dir: Optional[str] = None
        self._stay_replan_streak: int = 0
        self._last_replan_time_s: float = -1e9

        # Internal exclusion memory. This deliberately does not use supervisor-
        # validated cells. A cell becomes non-selectable only after the policy
        # itself has received target detections for that cell in several processed
        # perception frames. This is less volatile than excluding cells after one
        # immediate controller claim.
        self._target_detection_counts: Dict[Cell, int] = {}
        self._target_detection_last_step: Dict[Cell, int] = {}

        # Candidate FoV coverage cache. The geometry for "stand at cell C
        # with yaw Y" is fixed after initialization, so recomputing the floor
        # and table coverages for every candidate on every replan is wasted work.
        # The cache stores the exact outputs of GridEnvManager.compute_cell_coverages
        # for every candidate cell and yaw option, separately for floor/table
        # projections. Runtime table_cells only decide which cached map supplies
        # each covered cell.
        self._candidate_coverage_cache: Dict[Tuple[int, int, int], Dict[str, Dict[Tuple[int, int], float]]] = {}

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

        # Keep immediate controller claims for debug/snapshots only. Planning
        # exclusion uses the policy's internal target-detection counter below.
        claimed_cells_list = task.get("found_cells", []) or []
        claimed_cells = {(int(cx), int(cy)) for (cx, cy) in claimed_cells_list}

        dbg_policy = bool(self.debug.get("policy_grid", False))
        dbg_every = int(self.debug.get("every_n", 50))
        step_ok = (step is not None) and (step % dbg_every == 0)

        current_cell: Optional[Cell] = grid.get("current_cell")
        visible_mask: Dict[Cell, bool] = grid.get("visible_mask", {})
        det_conf: Dict[Cell, float] = grid.get("det_confidence", {})
        coverages: Dict[Cell, float] = grid.get("coverages", {})
        table_cells = self._table_cells_from_grid(grid)
        self._update_target_detection_counts(grid, step=step)
        excluded_cells = self._excluded_cells_from_target_detection_counts()

        self._update_beliefs(
            visible_mask=visible_mask,
            det_conf=det_conf,
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

        if should_replan:
            if current_cell is not None:
                (next_cell, next_yaw), scored_actions = self._pick_next_action(
                    current_cell=current_cell,
                    excluded_cells=excluded_cells,
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
                }

                self.navigator.set_target(next_cell, next_yaw)
                self._last_replan_time_s = time_now

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
                        found_cells=claimed_cells,
                        table_cells=table_cells,
                        excluded_cells=excluded_cells,
                        det_conf=det_conf,
                        visible_mask=visible_mask,
                        coverages=coverages,
                        chosen=(next_cell, next_yaw),
                        scored_actions=scored_actions,
                        reason=f"stuck{self._stay_replan_streak}",
                    )
                    self._stay_replan_streak = 0

                if dbg_policy and step_ok:
                    print(f"[AIF] picked target cell={next_cell} yaw={next_yaw:.3f}", flush=True)

        if snap_enabled and self._snap_dir and snap_every_n > 0 and step is not None and (step % snap_every_n == 0):
            if current_cell is not None:
                (best_cell, best_yaw), scored = self._pick_next_action(
                    current_cell=current_cell,
                    excluded_cells=excluded_cells,
                    table_cells=table_cells,
                    return_scores=True,
                )
                self._save_snapshot_npz(
                    snap_dir=self._snap_dir,
                    step=int(step),
                    pose=pose,
                    current_cell=current_cell,
                    found_cells=claimed_cells,
                    table_cells=table_cells,
                    excluded_cells=excluded_cells,
                    det_conf=det_conf,
                    visible_mask=visible_mask,
                    coverages=coverages,
                    chosen=(best_cell, best_yaw),
                    scored_actions=scored,
                    reason="periodic",
                )

        if dbg_policy and step_ok:
            cc = grid.get("current_cell")
            thr = grid.get("coverage_threshold")
            num_vis = sum(1 for v in visible_mask.values() if v)
            num_det = len(det_conf)

            print(
                f"[POLICY GRID] step={step} cc={cc} thr={thr} "
                f"cells: cover={len(coverages)} mask={len(visible_mask)} det={num_det} "
                f"visible={num_vis} excluded={len(excluded_cells)}",
                flush=True
            )

            top_cov = sorted(coverages.items(), key=lambda kv: kv[1], reverse=True)[:9]
            for (cx, cy), frac in top_cov:
                v = visible_mask.get((cx, cy), None)
                d = det_conf.get((cx, cy), 0.0)
                bp = self._belief_present(cx, cy)
                print(f"  cell=({cx},{cy}) cov={frac*100:5.1f}% vis={v} det={d:.2f} q(present)={bp:.2f}", flush=True)

            top_bel = self._top_beliefs(k=5)
            if top_bel:
                print("  top q(present): " + ", ".join([f"({cx},{cy})={p:.2f}" for (cx, cy, p) in top_bel]), flush=True)

            if self.navigator is not None and self.navigator.has_target():
                print(f"  target={self.navigator.target_cell} target_yaw={self.navigator.target_yaw:.3f}", flush=True)

        cmd = self.navigator.compute_command(pose)

        if self.debug.get("policy_detections", False) and step is not None and (step % self.every_n == 0):
            if det_conf:
                top = max(det_conf.items(), key=lambda kv: kv[1])
                print(f"[detections] top_cell={top[0]} conf={top[1]:.3f}", flush=True)

        return {
            "forward": float(cmd["forward"]),
            "sideways": float(cmd["sideways"]),
            "yaw_rate": float(cmd["yaw_rate"]),
            "target_selection": target_selection,
        }

    def save_snapshot_from_controller(self, observation: dict, *, reason: str = "final") -> Optional[str]:
        """
        Called by the controller at 'done'/'timeout' to guarantee a final snapshot exists.
        Returns snapshot path if written, else None.
        """
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

        current_cell = grid.get("current_cell", None)
        if not (isinstance(current_cell, (list, tuple)) and len(current_cell) == 2):
            current_cell = current_cell if isinstance(current_cell, tuple) else None

        current_cell = (int(current_cell[0]), int(current_cell[1])) if current_cell is not None else None

        found_cells_list = task.get("found_cells", []) or []
        found_cells = {(int(cx), int(cy)) for (cx, cy) in found_cells_list}
        excluded_cells = self._excluded_cells_from_target_detection_counts()

        det_conf = grid.get("det_confidence", {}) or {}
        visible_mask = grid.get("visible_mask", {}) or {}
        coverages = grid.get("coverages", {}) or {}
        table_cells = self._table_cells_from_grid(grid)

        scored_actions = []
        chosen = (current_cell, float(pose.get("yaw", 0.0)))

        if current_cell is not None:
            (best_cell, best_yaw), scored_actions = self._pick_next_action(
                current_cell=current_cell,
                excluded_cells=excluded_cells,
                table_cells=table_cells,
                return_scores=True,
            )
            chosen = (best_cell, best_yaw)

        return self._save_snapshot_npz(
            snap_dir=self._snap_dir,
            step=step,
            pose={"x": float(pose.get("x", 0.0)), "y": float(pose.get("y", 0.0)), "yaw": float(pose.get("yaw", 0.0))},
            current_cell=current_cell,
            found_cells=found_cells,
            table_cells=table_cells,
            excluded_cells=excluded_cells,
            det_conf=det_conf,
            visible_mask=visible_mask,
            coverages=coverages,
            chosen=chosen,
            scored_actions=scored_actions,
            reason=reason,
        )

    # ---------------- init ----------------

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

        p1 = float(np.clip(self.prior_present, 1e-6, 1.0 - 1e-6))
        p0 = 1.0 - p1
        self.q_s = np.zeros((self.grid_spec.rows, self.grid_spec.cols, 2), dtype=np.float64)
        self.q_s[:, :, 0] = p0
        self.q_s[:, :, 1] = p1

        self._precompute_candidate_coverages()

    @staticmethod
    def _fov_spec_from_dict(fs: dict) -> FoVSpec:
        return FoVSpec(
            half_left=float(fs["half_left"]),
            half_right=float(fs["half_right"]),
            forward=float(fs["forward"]),
            backward=float(fs["backward"]),
        )

    @staticmethod
    def _table_cells_from_grid(grid: dict) -> set[Cell]:
        """Extract discovered table-height cells from the controller observation.

        The controller sends table cells as (cx, cy) pairs under grid["table_cells"]
        and also inside grid["surface_height_map"]. Accept both forms so old
        snapshots/debug payloads remain readable.
        """
        out: set[Cell] = set()

        raw_cells = grid.get("table_cells", []) or []
        for cell in raw_cells:
            if isinstance(cell, (list, tuple)) and len(cell) == 2:
                out.add((int(cell[0]), int(cell[1])))

        surface_map = grid.get("surface_height_map", {}) or {}
        for cell in surface_map.get("table_cells", []) or []:
            if isinstance(cell, (list, tuple)) and len(cell) == 2:
                out.add((int(cell[0]), int(cell[1])))

        return out

    @staticmethod
    def _cell_from_event_or_key(value) -> Optional[Cell]:
        if isinstance(value, dict):
            cell = value.get("cell", None)
            if cell is None and "cx" in value and "cy" in value:
                cell = (value.get("cx"), value.get("cy"))
        else:
            cell = value

        if isinstance(cell, (list, tuple)) and len(cell) == 2:
            try:
                return (int(cell[0]), int(cell[1]))
            except Exception:
                return None
        return None

    @staticmethod
    def _looks_like_target_event(event: dict) -> bool:
        if not isinstance(event, dict):
            return False
        class_name = str(event.get("class_name", event.get("label", "")))
        if class_name in {"target_bottle", "target_can"}:
            return True
        class_id = event.get("class_id", None)
        try:
            return int(class_id) in {0, 1}
        except Exception:
            return False

    def _target_detection_cells_from_grid(self, grid: dict) -> set[Cell]:
        cells: set[Cell] = set()

        for event in grid.get("target_detections", []) or []:
            if not self._looks_like_target_event(event):
                continue
            cell = self._cell_from_event_or_key(event)
            if cell is not None:
                cells.add(cell)

        # Backwards-compatible fallback for older controller payloads where only
        # the target-only confidence map is available.
        if not cells:
            det_conf = grid.get("det_confidence", {}) or {}
            for cell in det_conf.keys():
                parsed = self._cell_from_event_or_key(cell)
                if parsed is not None:
                    cells.add(parsed)

        return cells

    def _update_target_detection_counts(self, grid: dict, *, step) -> None:
        if not bool(grid.get("perception_processed_this_step", False)):
            return

        step_i = int(step or 0)
        for cell in self._target_detection_cells_from_grid(grid):
            # Count at most once per processed perception frame per cell, even if
            # two detections/classes land in that same cell during this frame.
            if self._target_detection_last_step.get(cell, None) == step_i:
                continue
            self._target_detection_counts[cell] = int(self._target_detection_counts.get(cell, 0)) + 1
            self._target_detection_last_step[cell] = step_i

    def _excluded_cells_from_target_detection_counts(self) -> set[Cell]:
        threshold = int(max(1, self.target_exclusion_detection_frames))
        return {
            (int(cx), int(cy))
            for (cx, cy), count in self._target_detection_counts.items()
            if int(count) >= threshold
        }


    def _precompute_candidate_coverages(self) -> None:
        """Cache floor/table FoV coverages for every candidate cell/yaw pair.

        This preserves the original planning semantics: each cached entry is
        produced by the same GridEnvManager.compute_cell_coverages call that
        _efe_components previously executed inside every scoring loop. The only
        difference is timing: the geometry is computed once after initialization
        instead of repeatedly at every replan.
        """
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
                        "floor": dict(
                            self.env_mgr_floor.compute_cell_coverages(
                                target_xy[0], target_xy[1], float(yaw)
                            )
                        ),
                        "table": dict(
                            self.env_mgr_table.compute_cell_coverages(
                                target_xy[0], target_xy[1], float(yaw)
                            )
                        ),
                    }

    def _cached_candidate_coverages_rc(
        self,
        *,
        target_cell: Cell,
        target_yaw_index: int,
    ) -> tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        """Return cached (floor_cov, table_cov) for a candidate cell/yaw.

        Falls back to direct computation if a caller ever supplies a yaw index
        outside the precomputed table, although normal policy planning always
        uses self.yaw_options and therefore always hits the cache.
        """
        assert self.env_mgr_floor is not None
        assert self.env_mgr_table is not None

        cx, cy = int(target_cell[0]), int(target_cell[1])
        key = (cx, cy, int(target_yaw_index))
        cached = self._candidate_coverage_cache.get(key, None)
        if cached is not None:
            return cached["floor"], cached["table"]

        # Defensive fallback; should not happen in the normal action loop.
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
        """Merge cached floor/table future coverages per covered cell.

        Cells currently known as table-height use the cached table-plane
        coverage; all other cells use the cached floor-plane coverage. This is
        equivalent to the previous implementation, which recomputed floor_cov
        and table_cov for the candidate cell/yaw immediately before this merge.
        """
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
            if cxcy in table_cells:
                val = float(table_cov.get(rc, 0.0))
            else:
                val = float(floor_cov.get(rc, 0.0))
            if val > 0.0:
                out[rc] = val
        return out


    # ---------------- belief update ----------------

    def _update_beliefs(
        self,
        *,
        visible_mask: Dict[Cell, bool],
        det_conf: Dict[Cell, float],
        coverages: Dict[Cell, float],
    ) -> None:
        """
        Coverage-weighted posterior interpolation update.

        For each visible cell:
          1) compute the full Bayesian posterior exactly as before
          2) interpolate between prior and posterior with weight = coverage

        q_new = (1 - w) * q_prior + w * q_full

        where:
          - w = coverage fraction in [0, 1]
          - w = 0   => no update
          - w = 1   => full update
        """
        assert self.q_s is not None
        rows, cols = self.q_s.shape[0], self.q_s.shape[1]

        for (cx, cy), vis in visible_mask.items():
            if not vis:
                continue
            if not (0 <= cx < cols and 0 <= cy < rows):
                continue

            coverage = float(coverages.get((cx, cy), 0.0))
            coverage = clamp(coverage, 0.0, 1.0)
            if coverage <= 0.0:
                continue

            conf = float(det_conf.get((cx, cy), 0.0))
            conf = clamp(conf, 0.0, 1.0)

            p_hit = clamp(max(conf, self.conf_floor), 0.0, 1.0)
            p_fa = self.p_fa

            # Observation model exactly as before:
            # - positive observation if conf > 0
            # - negative observation otherwise
            A0 = np.array([1.0 - p_fa, 1.0 - p_hit], dtype=np.float64)
            A1 = np.array([p_fa, p_hit], dtype=np.float64)
            likelihood = A1 if conf > 0.0 else A0

            prior = self.q_s[cy, cx, :].copy()

            # Full posterior under the original observation model
            post_unnorm = likelihood * prior
            z = float(post_unnorm.sum())
            if z <= 1e-12:
                continue
            full_post = post_unnorm / z

            # Coverage-weighted interpolation between prior and full posterior
            new_post = (1.0 - coverage) * prior + coverage * full_post

            # Numerical safeguard; convex combination should already sum to 1
            z2 = float(new_post.sum())
            if z2 <= 1e-12:
                continue
            self.q_s[cy, cx, :] = new_post / z2

    def _belief_present(self, cx: int, cy: int) -> float:
        if self.q_s is None:
            return 0.0
        rows, cols = self.q_s.shape[0], self.q_s.shape[1]
        if not (0 <= cx < cols and 0 <= cy < rows):
            return 0.0
        return float(self.q_s[cy, cx, 1])

    def _top_beliefs(self, k: int = 5):
        if self.q_s is None:
            return []
        q1 = self.q_s[:, :, 1]
        flat = q1.reshape(-1)
        if flat.size == 0:
            return []
        k = int(max(1, min(k, flat.size)))
        idxs = np.argpartition(flat, -k)[-k:]
        idxs = idxs[np.argsort(flat[idxs])[::-1]]
        cols = q1.shape[1]
        out = []
        for idx in idxs:
            cy = int(idx // cols)
            cx = int(idx % cols)
            out.append((cx, cy, float(q1[cy, cx])))
        return out

    # ---------------- action selection ----------------

    def _all_cells(self) -> List[Cell]:
        assert self.grid_spec is not None
        return [
            (cx, cy)
            for cy in range(self.grid_spec.rows)
            for cx in range(self.grid_spec.cols)
        ]

    def _cell_center_xy(self, cx: int, cy: int) -> Tuple[float, float]:
        assert self.grid_spec is not None
        x0 = self.grid_spec.min_x + cx * self.grid_spec.cell_size
        y0 = self.grid_spec.min_y + cy * self.grid_spec.cell_size
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
        excluded_cells: Optional[set] = None,
        table_cells: Optional[set[Cell]] = None,
        return_scores: bool = False,
    ):
        assert self.grid_spec is not None
        assert self.q_s is not None

        candidates = self._all_cells()

        excluded_cells = excluded_cells or set()
        table_cells = table_cells or set()
        candidates = [c for c in candidates if c not in excluded_cells]
        if not candidates:
            candidates = self._all_cells()

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
                risk, IG = self._efe_components(
                    target_cell=cell,
                    target_yaw_index=yaw_idx,
                    table_cells=table_cells,
                )
                G = float(risk - self.beta_info * IG + distance_cost)

                if return_scores:
                    scores.append({
                        "cell": [int(cell[0]), int(cell[1])],
                        "yaw": float(yaw),
                        "risk": float(risk),
                        "IG": float(IG),
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
    ) -> Tuple[float, float]:
        """
        Returns (risk, IG) used by EFE:
          G = risk - beta_info * IG + distance_cost_epsilon * dist
        """
        assert self.env_mgr_floor is not None
        assert self.env_mgr_table is not None
        assert self.q_s is not None

        cov_rc = self._merged_candidate_coverages_rc(
            target_cell=target_cell,
            target_yaw_index=int(target_yaw_index),
            table_cells=table_cells or set(),
        )

        p_hit = self.planning_p_hit
        p_fa = self.p_fa

        A = np.array(
            [
                [1.0 - p_fa, 1.0 - p_hit],
                [p_fa,       p_hit],
            ],
            dtype=np.float64,
        )

        IG = 0.0
        for (r, c), frac in cov_rc.items():
            frac = float(frac)
            if frac <= 0.05:
                continue

            cy = int(r)
            cx = int(c)
            prior = self.q_s[cy, cx, :]

            p_o = A @ prior
            ig_cell = 0.0

            for o in (0, 1):
                po = float(p_o[o])
                if po <= 1e-12:
                    continue
                post = A[o, :] * prior
                z = float(post.sum())
                if z <= 1e-12:
                    continue
                post = post / z
                kl = float(np.sum(post * (np.log(post + 1e-12) - np.log(prior + 1e-12))))
                ig_cell += po * kl

            IG += frac * ig_cell

        cx_t, cy_t = target_cell
        prior_t = self.q_s[cy_t, cx_t, :]
        p_o_t = A @ prior_t
        risk = float(np.sum(p_o_t * (np.log(p_o_t + 1e-12) - np.log(self.C + 1e-12))))

        return float(risk), float(IG)

    # ---------------- snapshot IO ----------------

    def _maybe_init_snapshot_dir(self, snap_dir: Optional[str]) -> Optional[str]:
        if not snap_dir:
            return None
        os.makedirs(snap_dir, exist_ok=True)
        return snap_dir

    def _save_snapshot_npz(
        self,
        *,
        snap_dir: str,
        step: int,
        pose: dict,
        current_cell: Optional[Cell],
        found_cells: set,
        table_cells: Optional[set[Cell]],
        excluded_cells: Optional[set[Cell]],
        det_conf: Dict[Cell, float],
        visible_mask: Dict[Cell, bool],
        coverages: Dict[Cell, float],
        chosen: Tuple[Cell, float],
        scored_actions: list,
        reason: str,
    ) -> str:
        if self.q_s is None:
            raise RuntimeError("q_s is None; cannot save snapshot")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"snapshot_step{int(step):06d}_{reason}_{ts}.npz"
        path = os.path.join(snap_dir, fname)

        meta = {
            "step": int(step),
            "reason": str(reason),
            "pose": {"x": float(pose["x"]), "y": float(pose["y"]), "yaw": float(pose["yaw"])},
            "current_cell": list(current_cell) if current_cell is not None else None,
            "chosen_cell": [int(chosen[0][0]), int(chosen[0][1])] if chosen[0] is not None else None,
            "chosen_yaw": float(chosen[1]),
            "found_cells": sorted([[int(c[0]), int(c[1])] for c in found_cells]),
            "table_cells": sorted([[int(c[0]), int(c[1])] for c in (table_cells or set())]),
            "excluded_cells": sorted([[int(c[0]), int(c[1])] for c in (excluded_cells or set())]),
            "target_detection_counts": {
                f"{int(k[0])},{int(k[1])}": int(v)
                for k, v in sorted(self._target_detection_counts.items())
            },
            "target_exclusion_detection_frames": int(self.target_exclusion_detection_frames),
            "det_conf": {f"{k[0]},{k[1]}": float(v) for k, v in det_conf.items()},
            "visible_mask": {f"{k[0]},{k[1]}": bool(v) for k, v in visible_mask.items()},
            "coverages": {f"{k[0]},{k[1]}": float(v) for k, v in coverages.items()},
            "scored_actions": scored_actions,
        }

        np.savez_compressed(
            path,
            q_s=self.q_s.astype(np.float32),
            meta_json=np.array(json.dumps(meta), dtype=object),
        )
        print(f"[SNAPSHOT] wrote {path}", flush=True)
        return path