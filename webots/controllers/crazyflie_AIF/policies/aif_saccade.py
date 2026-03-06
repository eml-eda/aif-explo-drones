# policies/aif_saccade.py

from __future__ import annotations

import math
from typing import Dict, Tuple, Optional, List

import numpy as np

from env.env_manager import GridSpec, FoVSpec, GridEnvManager
from utils.navigator import Navigator, clamp

Cell = Tuple[int, int]  # (cx, cy)


class AIFSaccadePolicy:
    """
    Active Inference saccade over grid cells:
      - belief update every step
      - pick action (adjacent cell + yaw) only when navigator arrived
      - action scored by 1-step Expected Free Energy:
            G = risk - beta * info_gain + move_cost

    Hidden state per cell: s ∈ {absent, present}
    Outcome per cell:      o ∈ {no_detect, detect}

    Likelihood update uses detection confidence when visible:
      P(o=detect|s=present) = p_hit  (derived from conf, floored)
      P(o=detect|s=absent)  = p_fa   (small constant)
    """

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
        
        # prior on "object present" per cell
        self.prior_present = float(prior_present)

        # likelihood params (updates)
        self.p_fa = float(p_fa)
        self.conf_floor = float(conf_floor)

        # EFE weights
        self.beta_info = float(beta_info)
        self.move_cost = float(move_cost)
        self.planning_p_hit = float(planning_p_hit)

        # preference C over outcomes at the *target* cell (no_detect=0, detect=1)
        p_det = float(prefer_detect_strength)
        self.C = np.array([1.0 - p_det, p_det], dtype=np.float64)

        # initialized on first call
        self.grid_spec: Optional[GridSpec] = None
        self.fov_spec: Optional[FoVSpec] = None
        self.env_mgr: Optional[GridEnvManager] = None
        self.navigator: Optional[Navigator] = None

        # belief map q_s[row, col, s]
        self.q_s: Optional[np.ndarray] = None  # shape (rows, cols, 2)

        if yaw_options is None:
            # 8 headings
            yaw_options = [k * (math.pi / 4.0) for k in range(8)]
        self.yaw_options = [float(y) for y in yaw_options]

    def compute_action(self, observation: dict) -> Dict[str, float]:
        self._ensure_initialized(observation)

        step = observation.get("step", None)
        pose = observation["pose"]
        grid = observation["grid"]

        dbg_policy = bool(self.debug.get("policy_grid", False))
        dbg_every = int(self.debug.get("every_n", 50))
        step_ok = (step is not None) and (step % dbg_every == 0)

        current_cell: Optional[Cell] = grid.get("current_cell")
        visible_mask: Dict[Cell, bool] = grid.get("visible_mask", {})
        det_conf: Dict[Cell, float] = grid.get("det_confidence", {})
        coverages: Dict[Cell, float] = grid.get("coverages", {})

        # (1) belief update every controller step
        self._update_beliefs(visible_mask=visible_mask, det_conf=det_conf)

        # (2) pick new target only on arrival (or first time)
        assert self.navigator is not None
        if (not self.navigator.has_target()) or self.navigator.arrived(pose, coverages):
            if current_cell is not None:
                next_cell, next_yaw = self._pick_next_action(current_cell=current_cell)
                self.navigator.set_target(next_cell, next_yaw)

                if dbg_policy and step_ok:
                    print(f"[AIF] picked target cell={next_cell} yaw={next_yaw:.3f}", flush=True)

        # ALWAYS print debug every_n steps, not only on replans
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

        # (3) navigator -> continuous commands
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
            max_forward=0.12,
            max_sideways=0.12,
            max_yaw_rate=0.35,
            k_pos=1.2,
            k_yaw=1.5,
            dist_eps_m=0.2,
            yaw_eps_rad=0.12,
            min_cell_coverage=0.60,
        )

        # biased prior toward "absent" to enable saccade-like tracking once evidence appears
        p1 = float(np.clip(self.prior_present, 1e-6, 1.0 - 1e-6))
        p0 = 1.0 - p1
        self.q_s = np.zeros((self.grid_spec.rows, self.grid_spec.cols, 2), dtype=np.float64)
        self.q_s[:, :, 0] = p0  # absent
        self.q_s[:, :, 1] = p1  # present

    # ---------------- belief update ----------------

    def _update_beliefs(self, *, visible_mask: Dict[Cell, bool], det_conf: Dict[Cell, float]) -> None:
        """
        Bayesian update for each visible cell (cx,cy).

        Observed outcome:
          o=1 (detect) iff det_conf(cell) > 0, else o=0 (no_detect)

        A[o,s] with:
          p_hit = max(conf, conf_floor)
          p_fa  = p_fa
        """
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

            # o ∈ {0=no_detect, 1=detect}, s ∈ {0=absent, 1=present}
            # A[o,s]
            A0 = np.array([1.0 - p_fa, 1.0 - p_hit], dtype=np.float64)  # o=0
            A1 = np.array([p_fa,       p_hit],       dtype=np.float64)  # o=1
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
        """
        Candidate next cells. Includes the current cell to allow "stay + change yaw"
        as an action.
        """
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

    def _pick_next_action(self, *, current_cell: Cell) -> Tuple[Cell, float]:
        assert self.grid_spec is not None
        assert self.q_s is not None

        candidates = self._adjacent_cells(current_cell)
        if not candidates:
            return current_cell, 0.0

        best_G = float("inf")
        best = (candidates[0], self.yaw_options[0])

        cx0, cy0 = current_cell
        x0, y0 = self._cell_center_xy(cx0, cy0)

        for cell in candidates:
            tx, ty = self._cell_center_xy(cell[0], cell[1])
            dist = math.hypot(tx - x0, ty - y0)
            base_cost = self.move_cost * dist

            for yaw in self.yaw_options:
                G = self._expected_free_energy(target_cell=cell, target_xy=(tx, ty), target_yaw=yaw)
                G = G + base_cost
                if G < best_G:
                    best_G = G
                    best = (cell, yaw)

        return best

    def _expected_free_energy(self, *, target_cell: Cell, target_xy: Tuple[float, float], target_yaw: float) -> float:
        """
        1-step EFE:
          - predict visible cells from hypothetical pose at (target_xy, target_yaw)
            using the SAME geometry as runtime (GridEnvManager)
          - epistemic value: expected information gain over predicted visible cells
          - risk term: KL(predicted outcomes at target_cell || preference C)
        """
        assert self.env_mgr is not None
        assert self.q_s is not None

        x, y = target_xy

        # predict coverages using your shared env manager (returns (row,col))
        cov_rc = self.env_mgr.compute_cell_coverages(x, y, target_yaw)

        # Epistemic term (expected IG)
        IG = 0.0
        p_hit = self.planning_p_hit
        p_fa = self.p_fa

        # fixed planning likelihood A[o,s]
        # A[o,s] with o in {0,1}, s in {0,1}
        A = np.array(
            [
                [1.0 - p_fa, 1.0 - p_hit],
                [p_fa,       p_hit],
            ],
            dtype=np.float64,
        )

        for (r, c), frac in cov_rc.items():
            frac = float(frac)
            # consistently with mask visibility
            if frac <= 0.05:
                continue

            cy = int(r)
            cx = int(c)
            prior = self.q_s[cy, cx, :]

            # p(o) = A @ q(s)
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

        # Risk term at the target cell only
        cx_t, cy_t = target_cell
        prior_t = self.q_s[cy_t, cx_t, :]
        p_o_t = A @ prior_t

        risk = float(np.sum(p_o_t * (np.log(p_o_t + 1e-12) - np.log(self.C + 1e-12))))

        return float(risk - self.beta_info * IG)