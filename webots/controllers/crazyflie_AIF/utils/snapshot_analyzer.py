# utils/snapshot_analyzer.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


Cell = Tuple[int, int]


@dataclass
class SnapshotData:
    path: str
    q_s: np.ndarray  # (rows, cols, 2) float32
    meta: Dict[str, Any]


@dataclass
class ActionRankRow:
    cell: Cell
    yaw: float
    G: float
    risk: float
    IG: float
    base_cost: float


@dataclass
class AnalysisSummary:
    # high-level
    step: int
    reason: str
    pose: Dict[str, float]
    current_cell: Optional[Cell]
    chosen_cell: Cell
    chosen_yaw: float
    found_cells: List[Cell]

    # belief stats
    rows: int
    cols: int
    prior_present_mean: float
    q_present_mean: float
    q_present_max: float
    q_present_argmax: Cell
    q_present_topk: List[Tuple[Cell, float]]

    # action stats
    n_actions: int
    top_actions: List[ActionRankRow]
    chosen_rank: int  # 1 = best
    chosen_is_stay: bool

    # decomposition insights
    G_min: float
    G_max: float
    IG_min: float
    IG_max: float
    risk_min: float
    risk_max: float
    base_min: float
    base_max: float

    # “stuck” hypothesis indicators
    ig_is_exhausted: bool
    risk_is_flat: bool
    base_cost_dominates: bool

    # environment signals recorded in meta (if present)
    det_conf_cells: List[Tuple[Cell, float]]
    visible_cells: List[Cell]


class SnapshotAnalyzer:
    """
    Loads and analyzes an AIF snapshot saved by AIFSaccadePolicy.

    Snapshot format expected (np.savez_compressed):
      - q_s: (rows, cols, 2) float32
      - meta_json: JSON string with keys like:
            step, reason, pose, current_cell, chosen_cell, chosen_yaw,
            found_cells, det_conf, visible_mask, coverages, scored_actions
    """

    def __init__(self, *, eps: float = 1e-12):
        self.eps = float(eps)

    # ---------------------------
    # IO
    # ---------------------------

    def load(self, path: str) -> SnapshotData:
        d = np.load(path, allow_pickle=True)
        if "q_s" not in d or "meta_json" not in d:
            raise ValueError(f"Snapshot '{path}' missing required arrays: q_s/meta_json")

        q_s = d["q_s"]
        meta_raw = d["meta_json"].item()
        if not isinstance(meta_raw, str):
            meta_raw = str(meta_raw)

        meta = json.loads(meta_raw)
        return SnapshotData(path=path, q_s=q_s, meta=meta)

    # ---------------------------
    # Main analysis
    # ---------------------------

    def analyze(
        self,
        snap: SnapshotData,
        *,
        top_k_actions: int = 12,
        top_k_cells: int = 10,
        ig_exhausted_thresh: float = 1e-3,
        risk_flat_thresh: float = 1e-3,
        base_dom_ratio: float = 2.0,
    ) -> AnalysisSummary:
        q_s = snap.q_s
        meta = snap.meta

        rows, cols = int(q_s.shape[0]), int(q_s.shape[1])
        if q_s.ndim != 3 or q_s.shape[2] != 2:
            raise ValueError(f"q_s has wrong shape: {q_s.shape}, expected (rows, cols, 2)")

        q_present = q_s[:, :, 1].astype(np.float64)  # present prob map
        q_absent = q_s[:, :, 0].astype(np.float64)

        # Some sanity: probabilities should sum to 1-ish
        # (Don’t fail hard; just compute a quick metric.)
        sum_err = np.abs((q_present + q_absent) - 1.0)
        # Not returned directly, but you can add if you want.
        _sum_err_mean = float(sum_err.mean())

        # Belief stats
        q_present_mean = float(q_present.mean())
        q_present_max = float(q_present.max())
        argmax_flat = int(np.argmax(q_present))
        q_present_argmax = (argmax_flat % cols, argmax_flat // cols)  # (cx, cy)

        top_cells = self._topk_cells(q_present, k=top_k_cells)

        # Extract meta fields safely
        step = int(meta.get("step", -1))
        reason = str(meta.get("reason", ""))
        pose = meta.get("pose", {}) or {}
        pose_out = {
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
        }

        current_cell = self._as_cell(meta.get("current_cell", None))
        chosen_cell = self._as_cell(meta.get("chosen_cell", None), required=True)
        chosen_yaw = float(meta.get("chosen_yaw", 0.0))

        found_cells = [self._as_cell(c, required=True) for c in (meta.get("found_cells", []) or [])]

        # Parse signals
        det_conf_cells = self._parse_cell_float_map(meta.get("det_conf", {}) or {})
        visible_cells = self._parse_cell_bool_map(meta.get("visible_mask", {}) or {})

        # Action scoring table
        scored = meta.get("scored_actions", None)
        if not isinstance(scored, list):
            scored = []

        action_rows = self._parse_actions(scored)
        action_rows_sorted = sorted(action_rows, key=lambda r: r.G)

        n_actions = len(action_rows_sorted)
        top_actions = action_rows_sorted[: max(1, int(top_k_actions))]

        chosen_rank = self._find_chosen_rank(action_rows_sorted, chosen_cell, chosen_yaw)
        chosen_is_stay = (current_cell is not None) and (chosen_cell == current_cell)

        # Decomposition ranges
        G_min, G_max = self._minmax([r.G for r in action_rows_sorted])
        IG_min, IG_max = self._minmax([r.IG for r in action_rows_sorted])
        risk_min, risk_max = self._minmax([r.risk for r in action_rows_sorted])
        base_min, base_max = self._minmax([r.base_cost for r in action_rows_sorted])

        # Heuristic “why stuck” flags
        ig_is_exhausted = (IG_max <= float(ig_exhausted_thresh))
        risk_is_flat = ((risk_max - risk_min) <= float(risk_flat_thresh))

        # base_cost dominance: if variation in base_cost dwarfs variation in (risk - beta*IG)
        # We don’t have beta here, but we can approximate dominance by comparing spreads:
        # base_spread vs (risk_spread + IG_spread) scaled.
        base_spread = base_max - base_min
        signal_spread = (risk_max - risk_min) + (IG_max - IG_min)
        base_cost_dominates = (signal_spread > 0.0) and (base_spread / (signal_spread + self.eps) >= float(base_dom_ratio))

        # Prior present mean is not stored explicitly; we can approximate it from low-confidence regions.
        # A simple proxy: median present probability.
        prior_present_mean = float(np.median(q_present))

        return AnalysisSummary(
            step=step,
            reason=reason,
            pose=pose_out,
            current_cell=current_cell,
            chosen_cell=chosen_cell,
            chosen_yaw=chosen_yaw,
            found_cells=found_cells,
            rows=rows,
            cols=cols,
            prior_present_mean=prior_present_mean,
            q_present_mean=q_present_mean,
            q_present_max=q_present_max,
            q_present_argmax=q_present_argmax,
            q_present_topk=top_cells,
            n_actions=n_actions,
            top_actions=top_actions,
            chosen_rank=chosen_rank,
            chosen_is_stay=chosen_is_stay,
            G_min=G_min,
            G_max=G_max,
            IG_min=IG_min,
            IG_max=IG_max,
            risk_min=risk_min,
            risk_max=risk_max,
            base_min=base_min,
            base_max=base_max,
            ig_is_exhausted=ig_is_exhausted,
            risk_is_flat=risk_is_flat,
            base_cost_dominates=base_cost_dominates,
            det_conf_cells=det_conf_cells,
            visible_cells=visible_cells,
        )

    # ---------------------------
    # Reporting / exporting
    # ---------------------------

    def format_report(self, summary: AnalysisSummary, *, max_actions: int = 12) -> str:
        lines: List[str] = []
        lines.append(f"Snapshot analysis")
        lines.append(f"- step={summary.step} reason='{summary.reason}'")
        lines.append(f"- pose: x={summary.pose['x']:.2f} y={summary.pose['y']:.2f} yaw={summary.pose['yaw']:.3f} rad")

        lines.append(f"- current_cell={summary.current_cell} chosen_cell={summary.chosen_cell} chosen_yaw={summary.chosen_yaw:.3f}")
        lines.append(f"- chosen_rank={summary.chosen_rank}/{max(1, summary.n_actions)} chosen_is_stay={summary.chosen_is_stay}")

        lines.append("")
        lines.append("Belief map (q(present))")
        lines.append(f"- mean={summary.q_present_mean:.4f} max={summary.q_present_max:.4f} argmax={summary.q_present_argmax}")
        lines.append(f"- prior_proxy(median)={summary.prior_present_mean:.4f}")
        lines.append(f"- top cells:")
        for (cell, p) in summary.q_present_topk:
            lines.append(f"  - cell={cell} q={p:.4f}")

        lines.append("")
        lines.append("Action scoring ranges")
        lines.append(f"- G   min={summary.G_min:.6f} max={summary.G_max:.6f}")
        lines.append(f"- IG  min={summary.IG_min:.6f} max={summary.IG_max:.6f}")
        lines.append(f"- risk min={summary.risk_min:.6f} max={summary.risk_max:.6f}")
        lines.append(f"- base min={summary.base_min:.6f} max={summary.base_max:.6f}")

        lines.append("")
        lines.append("Stuck-hypothesis indicators")
        lines.append(f"- IG exhausted? {summary.ig_is_exhausted} (IG_max={summary.IG_max:.6f})")
        lines.append(f"- risk flat?    {summary.risk_is_flat} (risk_spread={summary.risk_max - summary.risk_min:.6f})")
        lines.append(f"- base dominates? {summary.base_cost_dominates}")

        lines.append("")
        lines.append(f"Top {min(max_actions, len(summary.top_actions))} actions by G (best first):")
        for i, r in enumerate(summary.top_actions[:max_actions], start=1):
            stay_flag = ""
            if summary.current_cell is not None and r.cell == summary.current_cell:
                stay_flag = " [STAY]"
            chosen_flag = ""
            if r.cell == summary.chosen_cell and abs(r.yaw - summary.chosen_yaw) < 1e-6:
                chosen_flag = " [CHOSEN]"
            lines.append(
                f"{i:02d}) cell={r.cell} yaw={r.yaw:.3f} "
                f"G={r.G:.6f} risk={r.risk:.6f} IG={r.IG:.6f} base={r.base_cost:.6f}"
                f"{stay_flag}{chosen_flag}"
            )

        lines.append("")
        lines.append(f"Found cells: {len(summary.found_cells)}")
        if summary.found_cells:
            lines.append("  " + ", ".join([str(c) for c in summary.found_cells[:30]]) + (" ..." if len(summary.found_cells) > 30 else ""))

        lines.append(f"Detections in meta: {len(summary.det_conf_cells)}")
        if summary.det_conf_cells:
            top_det = sorted(summary.det_conf_cells, key=lambda x: x[1], reverse=True)[:10]
            lines.append("  top det_conf: " + ", ".join([f"{c}:{v:.2f}" for c, v in top_det]))

        lines.append(f"Visible cells in meta: {len(summary.visible_cells)}")

        return "\n".join(lines)

    def export_summary_json(self, summary: AnalysisSummary, out_path: str) -> None:
        payload = {
            "step": summary.step,
            "reason": summary.reason,
            "pose": summary.pose,
            "current_cell": list(summary.current_cell) if summary.current_cell is not None else None,
            "chosen_cell": list(summary.chosen_cell),
            "chosen_yaw": summary.chosen_yaw,
            "chosen_rank": summary.chosen_rank,
            "chosen_is_stay": summary.chosen_is_stay,
            "grid": {"rows": summary.rows, "cols": summary.cols},
            "belief": {
                "prior_proxy_median": summary.prior_present_mean,
                "q_present_mean": summary.q_present_mean,
                "q_present_max": summary.q_present_max,
                "q_present_argmax": list(summary.q_present_argmax),
                "q_present_topk": [{"cell": list(c), "q": p} for (c, p) in summary.q_present_topk],
            },
            "ranges": {
                "G": {"min": summary.G_min, "max": summary.G_max},
                "IG": {"min": summary.IG_min, "max": summary.IG_max},
                "risk": {"min": summary.risk_min, "max": summary.risk_max},
                "base_cost": {"min": summary.base_min, "max": summary.base_max},
            },
            "stuck_indicators": {
                "ig_is_exhausted": summary.ig_is_exhausted,
                "risk_is_flat": summary.risk_is_flat,
                "base_cost_dominates": summary.base_cost_dominates,
            },
            "found_cells": [list(c) for c in summary.found_cells],
            "det_conf_cells": [{"cell": list(c), "conf": v} for (c, v) in summary.det_conf_cells],
            "visible_cells": [list(c) for c in summary.visible_cells],
            "top_actions": [
                {
                    "cell": list(r.cell),
                    "yaw": r.yaw,
                    "G": r.G,
                    "risk": r.risk,
                    "IG": r.IG,
                    "base_cost": r.base_cost,
                }
                for r in summary.top_actions
            ],
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ---------------------------
    # Internals
    # ---------------------------

    @staticmethod
    def _as_cell(x: Any, *, required: bool = False) -> Optional[Cell]:
        if x is None:
            if required:
                raise ValueError("Required cell field is missing")
            return None
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return (int(x[0]), int(x[1]))
        raise ValueError(f"Invalid cell encoding: {x}")

    @staticmethod
    def _parse_cell_float_map(d: Dict[str, Any]) -> List[Tuple[Cell, float]]:
        out: List[Tuple[Cell, float]] = []
        for k, v in d.items():
            try:
                parts = str(k).split(",")
                cx, cy = int(parts[0]), int(parts[1])
                out.append(((cx, cy), float(v)))
            except Exception:
                continue
        return out

    @staticmethod
    def _parse_cell_bool_map(d: Dict[str, Any]) -> List[Cell]:
        out: List[Cell] = []
        for k, v in d.items():
            try:
                if bool(v):
                    parts = str(k).split(",")
                    cx, cy = int(parts[0]), int(parts[1])
                    out.append((cx, cy))
            except Exception:
                continue
        return out

    @staticmethod
    def _parse_actions(scored_actions: List[dict]) -> List[ActionRankRow]:
        out: List[ActionRankRow] = []
        for r in scored_actions:
            try:
                cell_raw = r.get("cell", None)
                if not (isinstance(cell_raw, (list, tuple)) and len(cell_raw) == 2):
                    continue
                cell = (int(cell_raw[0]), int(cell_raw[1]))
                out.append(
                    ActionRankRow(
                        cell=cell,
                        yaw=float(r.get("yaw", 0.0)),
                        G=float(r.get("G", 0.0)),
                        risk=float(r.get("risk", 0.0)),
                        IG=float(r.get("IG", 0.0)),
                        base_cost=float(r.get("base_cost", 0.0)),
                    )
                )
            except Exception:
                continue
        return out

    @staticmethod
    def _minmax(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        return float(min(xs)), float(max(xs))

    @staticmethod
    def _find_chosen_rank(rows: List[ActionRankRow], chosen_cell: Cell, chosen_yaw: float) -> int:
        # Exact yaw match is fragile due to float repr; use a small tolerance.
        tol = 1e-6
        for i, r in enumerate(rows, start=1):
            if r.cell == chosen_cell and abs(r.yaw - chosen_yaw) <= tol:
                return i
        # If not found, rank unknown; return -1
        return -1

    @staticmethod
    def _topk_cells(q_present: np.ndarray, *, k: int) -> List[Tuple[Cell, float]]:
        rows, cols = q_present.shape
        flat = q_present.reshape(-1)
        if flat.size == 0:
            return []
        k = int(max(1, min(k, flat.size)))
        idxs = np.argpartition(flat, -k)[-k:]
        idxs = idxs[np.argsort(flat[idxs])[::-1]]
        out: List[Tuple[Cell, float]] = []
        for idx in idxs:
            cy = int(idx // cols)
            cx = int(idx % cols)
            out.append(((cx, cy), float(q_present[cy, cx])))
        return out


def analyze_snapshot_file(
    snapshot_path: str,
    *,
    top_k_actions: int = 12,
    top_k_cells: int = 10,
    save_json: bool = False,
    json_out_path: Optional[str] = None,
) -> str:
    """
    Convenience function: loads a snapshot, analyzes it, returns a human-readable report.
    Optionally writes a JSON summary next to the snapshot.
    """
    analyzer = SnapshotAnalyzer()
    snap = analyzer.load(snapshot_path)
    summary = analyzer.analyze(
        snap,
        top_k_actions=top_k_actions,
        top_k_cells=top_k_cells,
    )
    report = analyzer.format_report(summary, max_actions=top_k_actions)

    if save_json:
        if json_out_path is None:
            base, _ = os.path.splitext(snapshot_path)
            json_out_path = base + ".summary.json"
        analyzer.export_summary_json(summary, json_out_path)

    return report