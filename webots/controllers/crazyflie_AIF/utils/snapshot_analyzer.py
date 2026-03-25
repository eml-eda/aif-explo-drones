# utils/snapshot_analyzer.py

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


Cell = Tuple[int, int]


@dataclass
class SnapshotData:
    path: str
    q_s: np.ndarray
    meta: Dict[str, Any]


@dataclass
class ActionRankRow:
    cell: Cell
    yaw: float
    G: float
    risk: float
    IG: float
    base_cost: float


class SnapshotAnalyzer:
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
    # Analysis helpers
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

    @staticmethod
    def _extract_policy_name_from_snapshot_path(snapshot_path: str) -> str:
        """
        Example snapshot path:
          .../logs/log_run_aif_saccade_20260324_212017/snapshots/snapshot_step....

        Returns:
          aif_saccade
        """
        p = Path(snapshot_path)
        parts = p.parts

        run_dir_name = None
        for part in parts:
            if part.startswith("log_run_"):
                run_dir_name = part
                break

        if run_dir_name is None:
            return "unknown_policy"

        match = re.match(r"log_run_(.+)_\d{8}_\d{6}$", run_dir_name)
        if not match:
            return "unknown_policy"

        return match.group(1)

    @staticmethod
    def _default_belief_heatmap_output_path(snapshot_path: str) -> str:
        policy_name = SnapshotAnalyzer._extract_policy_name_from_snapshot_path(snapshot_path)
        out_dir = Path("webots/visuals/belief_heatmaps")
        out_dir.mkdir(parents=True, exist_ok=True)

        snap_name = Path(snapshot_path).stem
        out_name = f"{policy_name}__{snap_name}.png"
        return str(out_dir / out_name)

    # ---------------------------
    # Text report
    # ---------------------------

    def report(self, snap: SnapshotData, *, top_k_actions: int = 12, top_k_cells: int = 10) -> str:
        q_s = snap.q_s
        meta = snap.meta

        if q_s.ndim != 3 or q_s.shape[2] != 2:
            raise ValueError(f"q_s has wrong shape: {q_s.shape}, expected (rows, cols, 2)")

        rows, cols = int(q_s.shape[0]), int(q_s.shape[1])
        q_present = q_s[:, :, 1].astype(np.float64)

        step = int(meta.get("step", -1))
        reason = str(meta.get("reason", ""))
        pose = meta.get("pose", {}) or {}
        current_cell = self._as_cell(meta.get("current_cell", None))
        chosen_cell = self._as_cell(meta.get("chosen_cell", None), required=True)
        chosen_yaw = float(meta.get("chosen_yaw", 0.0))

        q_mean = float(q_present.mean())
        q_max = float(q_present.max())
        argmax_flat = int(np.argmax(q_present))
        argmax = (argmax_flat % cols, argmax_flat // cols)

        scored_actions = meta.get("scored_actions", []) or []
        rows_actions = sorted(self._parse_actions(scored_actions), key=lambda r: r.G)

        chosen_rank = -1
        for i, r in enumerate(rows_actions, start=1):
            if r.cell == chosen_cell and abs(r.yaw - chosen_yaw) <= 1e-6:
                chosen_rank = i
                break

        top_cells = self._topk_cells(q_present, k=top_k_cells)

        lines: List[str] = []
        lines.append("Snapshot analysis")
        lines.append(f"- file: {snap.path}")
        lines.append(f"- step={step} reason='{reason}'")
        lines.append(
            f"- pose: x={float(pose.get('x', 0.0)):.2f} "
            f"y={float(pose.get('y', 0.0)):.2f} "
            f"yaw={float(pose.get('yaw', 0.0)):.3f}"
        )
        lines.append(
            f"- current_cell={current_cell} chosen_cell={chosen_cell} "
            f"chosen_yaw={chosen_yaw:.3f} rank={chosen_rank}"
        )

        lines.append("")
        lines.append("Belief q(present)")
        lines.append(f"- grid: rows={rows} cols={cols}")
        lines.append(f"- mean={q_mean:.6f} max={q_max:.6f} argmax={argmax}")
        lines.append("- top cells:")
        for cell, p in top_cells:
            lines.append(f"  - {cell}: {p:.6f}")

        if rows_actions:
            lines.append("")
            lines.append(f"Top {min(top_k_actions, len(rows_actions))} actions by G:")
            for i, r in enumerate(rows_actions[:top_k_actions], start=1):
                flags = []
                if current_cell is not None and r.cell == current_cell:
                    flags.append("STAY")
                if r.cell == chosen_cell and abs(r.yaw - chosen_yaw) <= 1e-6:
                    flags.append("CHOSEN")
                flag_txt = f" [{' '.join(flags)}]" if flags else ""
                lines.append(
                    f"{i:02d}) cell={r.cell} yaw={r.yaw:.3f} "
                    f"G={r.G:.6f} risk={r.risk:.6f} IG={r.IG:.6f} base={r.base_cost:.6f}{flag_txt}"
                )

        return "\n".join(lines)

    # ---------------------------
    # Belief heatmap rendering
    # ---------------------------

    def render_belief_heatmap(
        self,
        snap: SnapshotData,
        *,
        out_path: Optional[str] = None,
        annotate: bool = True,
        annotate_decimals: int = 2,
        figsize_scale: float = 0.55,
        show: bool = True,
    ) -> str:
        """
        Saves a heatmap PNG for q(present). Annotates each cell with its value.

        Alignment:
        - integer ticks are cell centers
        - half-integers are cell boundaries
        """
        q_s = snap.q_s
        if q_s.ndim != 3 or q_s.shape[2] != 2:
            raise ValueError(f"q_s has wrong shape: {q_s.shape}, expected (rows, cols, 2)")

        q_present = q_s[:, :, 1].astype(np.float64)
        rows, cols = q_present.shape

        if out_path is None:
            out_path = self._default_belief_heatmap_output_path(snap.path)

        out_path_obj = Path(out_path)
        out_path_obj.parent.mkdir(parents=True, exist_ok=True)

        fig_w = max(6.0, cols * float(figsize_scale))
        fig_h = max(5.0, rows * float(figsize_scale))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(
            q_present,
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
            extent=(-0.5, cols - 0.5, -0.5, rows - 0.5),
            interpolation="nearest",
            cmap="viridis",
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("q(present)")

        policy_name = self._extract_policy_name_from_snapshot_path(snap.path)
        ax.set_title(f"Belief heatmap: q(present) per cell — {policy_name}")
        ax.set_xlabel("cx (col)")
        ax.set_ylabel("cy (row)")

        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))

        ax.set_xticks([x - 0.5 for x in range(cols + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(rows + 1)], minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)

        if annotate:
            fmt = f"{{:.{int(annotate_decimals)}f}}"
            for cy in range(rows):
                for cx in range(cols):
                    val = float(q_present[cy, cx])
                    text_color = "white" if val < 0.55 else "black"
                    ax.text(
                        cx,
                        cy,
                        fmt.format(val),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                    )

        fig.tight_layout()
        fig.savefig(out_path, dpi=180)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return str(out_path_obj)


def analyze_snapshot_file(
    snapshot_path: str,
    *,
    top_k_actions: int = 12,
    top_k_cells: int = 10,
    render_heatmap: bool = True,
    show_heatmap: bool = True,
) -> None:
    analyzer = SnapshotAnalyzer()
    snap = analyzer.load(snapshot_path)
    print(analyzer.report(snap, top_k_actions=top_k_actions, top_k_cells=top_k_cells))

    if render_heatmap:
        out_png = analyzer.render_belief_heatmap(snap, show=show_heatmap)
        print(f"[HEATMAP] wrote {out_png}")