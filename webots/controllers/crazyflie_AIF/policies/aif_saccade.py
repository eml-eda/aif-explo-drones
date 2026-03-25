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
        conf_floor: float = 0.10,
        beta_info: float = 1.0,
        move_cost: float = 0.02,
        prefer_detect_strength: float = 0.95,
        yaw_options: Optional[List[float]] = None,
        planning_p_hit: float = 0.85,
    ):
        self.debug = debug or {}
        self.every_n = int(self.debug.get("every_n", 50))

        self.prior_present = float(prior_present)

        self.p_fa = float(p_fa)
        self.conf_floor = float(conf_floor)

        self.beta_info = float(beta_info)
        self.move_cost = float(move_cost)
        self.planning_p_hit = float(planning_p_hit)

        p_det = float(prefer_detect_strength)
        self.C = np.array([1.0 - p_det, p_det], dtype=np.float64)

        self.grid_spec: Optional[GridSpec] = None
        self.fov_spec: Optional[FoVSpec] = None
        self.env_mgr: Optional[GridEnvManager] = None
        self.navigator: Optional[Navigator] = None

        self.q_s: Optional[np.ndarray] = None  # (rows, cols, 2)

        if yaw_options is None:
            yaw_options = [k * (math.pi / 4.0) for k in range(8)]
        self.yaw_options = [float(y) for y in yaw_options]

        self._snap_dir: Optional[str] = None
        self._stay_replan_streak: int = 0

    def compute_action(self, observation: dict) -> Dict[str, float]:
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

        found_cells_list = task.get("found_cells", []) or []
        found_cells = {(int(cx), int(cy)) for (cx, cy) in found_cells_list}

        dbg_policy = bool(self.debug.get("policy_grid", False))
        dbg_every = int(self.debug.get("every_n", 50))
        step_ok = (step is not None) and (step % dbg_every == 0)

        current_cell: Optional[Cell] = grid.get("current_cell")
        visible_mask: Dict[Cell, bool] = grid.get("visible_mask", {})
        det_conf: Dict[Cell, float] = grid.get("det_confidence", {})
        coverages: Dict[Cell, float] = grid.get("coverages", {})

        self._update_beliefs(visible_mask=visible_mask, det_conf=det_conf)

        assert self.navigator is not None

        scored_actions = None
        if (not self.navigator.has_target()) or self.navigator.arrived(pose, coverages):
            if current_cell is not None:
                (next_cell, next_yaw), scored_actions = self._pick_next_action(
                    current_cell=current_cell,
                    found_cells=found_cells,
                    return_scores=True,
                )
                self.navigator.set_target(next_cell, next_yaw)

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
                        found_cells=found_cells,
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
                    found_cells=found_cells,
                    return_scores=True,
                )
                self._save_snapshot_npz(
                    snap_dir=self._snap_dir,
                    step=int(step),
                    pose=pose,
                    current_cell=current_cell,
                    found_cells=found_cells,
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
                f"cells: cover={len(coverages)} mask={len(visible_mask)} det={num_det} visible={num_vis}",
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
        }

    # ---------- NEW: controller-triggered snapshot ----------
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

        det_conf = grid.get("det_confidence", {}) or {}
        visible_mask = grid.get("visible_mask", {}) or {}
        coverages = grid.get("coverages", {}) or {}

        # If current_cell missing we still save; but action scoring requires a cell.
        scored_actions = []
        chosen = (current_cell, float(pose.get("yaw", 0.0)))

        if current_cell is not None:
            (best_cell, best_yaw), scored_actions = self._pick_next_action(
                current_cell=current_cell,
                found_cells=found_cells,
                return_scores=True,
            )
            chosen = (best_cell, best_yaw)

        return self._save_snapshot_npz(
            snap_dir=self._snap_dir,
            step=step,
            pose={"x": float(pose.get("x", 0.0)), "y": float(pose.get("y", 0.0)), "yaw": float(pose.get("yaw", 0.0))},
            current_cell=current_cell,
            found_cells=found_cells,
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

        self.grid_spec = GridSpec(
            min_x=float(gs["min_x"]),
            max_x=float(gs["max_x"]),
            min_y=float(gs["min_y"]),
            max_y=float(gs["max_y"]),
            cell_size=float(gs["cell_size"]),
            rows=int(gs["rows"]),
            cols=int(gs["cols"]),
        )
        self.fov_spec = FoVSpec(
            half_left=float(fs["half_left"]),
            half_right=float(fs["half_right"]),
            forward=float(fs["forward"]),
            backward=float(fs["backward"]),
        )

        self.env_mgr = GridEnvManager(self.grid_spec, self.fov_spec)

        self.navigator = Navigator(
            self.grid_spec,
            max_forward=0.25,
            max_sideways=0.25,
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

    # ---------------- belief update ----------------

    def _update_beliefs(self, *, visible_mask: Dict[Cell, bool], det_conf: Dict[Cell, float]) -> None:
        assert self.q_s is not None
        rows, cols = self.q_s.shape[0], self.q_s.shape[1]

        for (cx, cy), vis in visible_mask.items():
            if not vis:
                continue
            if not (0 <= cx < cols and 0 <= cy < rows):
                continue

            conf = float(det_conf.get((cx, cy), 0.0))
            conf = clamp(conf, 0.0, 1.0)

            p_hit = clamp(max(conf, self.conf_floor), 0.0, 1.0)
            p_fa = self.p_fa

            A0 = np.array([1.0 - p_fa, 1.0 - p_hit], dtype=np.float64)
            A1 = np.array([p_fa, p_hit], dtype=np.float64)
            A = (A1 if conf > 0.0 else A0)

            prior = self.q_s[cy, cx, :].copy()
            post_unnorm = A * prior
            z = float(post_unnorm.sum())
            if z <= 1e-12:
                continue
            self.q_s[cy, cx, :] = post_unnorm / z

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

    def _adjacent_cells(self, current_cell: Cell) -> List[Cell]:
        assert self.grid_spec is not None
        cx, cy = current_cell
        nbrs: List[Cell] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < self.grid_spec.cols and 0 <= ny < self.grid_spec.rows:
                    nbrs.append((nx, ny))
        return nbrs

    def _cell_center_xy(self, cx: int, cy: int) -> Tuple[float, float]:
        assert self.grid_spec is not None
        x0 = self.grid_spec.min_x + cx * self.grid_spec.cell_size
        y0 = self.grid_spec.min_y + cy * self.grid_spec.cell_size
        cs = self.grid_spec.cell_size
        return (x0 + 0.5 * cs, y0 + 0.5 * cs)

    def _pick_next_action(
        self,
        *,
        current_cell: Cell,
        found_cells: Optional[set] = None,
        return_scores: bool = False,
    ):
        assert self.grid_spec is not None
        assert self.q_s is not None

        candidates = self._adjacent_cells(current_cell)

        found_cells = found_cells or set()
        candidates = [c for c in candidates if c not in found_cells]
        if not candidates:
            candidates = self._adjacent_cells(current_cell)

        scores = [] if return_scores else None

        best_G = float("inf")
        best = (candidates[0], self.yaw_options[0])

        cx0, cy0 = current_cell
        x0, y0 = self._cell_center_xy(cx0, cy0)

        for cell in candidates:
            tx, ty = self._cell_center_xy(cell[0], cell[1])
            dist = math.hypot(tx - x0, ty - y0)
            base_cost = self.move_cost * dist

            for yaw in self.yaw_options:
                risk, IG = self._efe_components(target_cell=cell, target_xy=(tx, ty), target_yaw=yaw)
                G = float(risk - self.beta_info * IG + base_cost)

                if return_scores:
                    scores.append({
                        "cell": [int(cell[0]), int(cell[1])],
                        "yaw": float(yaw),
                        "risk": float(risk),
                        "IG": float(IG),
                        "base_cost": float(base_cost),
                        "G": float(G),
                    })

                if G < best_G:
                    best_G = G
                    best = (cell, yaw)

        return (best, scores) if return_scores else best

    def _efe_components(self, *, target_cell: Cell, target_xy: Tuple[float, float], target_yaw: float) -> Tuple[float, float]:
        """
        Returns (risk, IG) used by EFE:
          G = risk - beta_info * IG + move_cost*dist
        """
        assert self.env_mgr is not None
        assert self.q_s is not None

        x, y = target_xy
        cov_rc = self.env_mgr.compute_cell_coverages(x, y, target_yaw)

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

            p_o = A @ prior  # (2,)
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