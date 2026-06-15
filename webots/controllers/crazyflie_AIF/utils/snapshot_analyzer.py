# utils/snapshot_analyzer.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Paste the snapshot path here, then run this file directly.
# -----------------------------------------------------------------------------
SNAPSHOT_PATH = "webots/controllers/crazyflie_AIF/logs/log_run_aif_saccade_table_20260611_144219/snapshots/snapshot_step003423_done_20260611_144402.npz"

TOP_K_ACTIONS = 12
TOP_K_CELLS = 10

# Target belief heatmap: q(present)
SHOW_TARGET_HEATMAP = True
SAVE_TARGET_HEATMAP = True

# Table belief heatmap: q(table). This is only rendered when q_table exists
# in the snapshot, i.e. for table-enriched policy runs.
SHOW_TABLE_HEATMAP = True
SAVE_TABLE_HEATMAP = True

ANNOTATE_HEATMAPS = True


Cell = Tuple[int, int]


@dataclass
class SnapshotData:
    path: str
    q_s: np.ndarray
    meta: Dict[str, Any]
    q_table: Optional[np.ndarray] = None
    q_present_saved: Optional[np.ndarray] = None
    q_class_confidence_saved: Optional[np.ndarray] = None
    q_classification_need_saved: Optional[np.ndarray] = None
    table_map: Optional[np.ndarray] = None


@dataclass
class ActionRankRow:
    cell: Cell
    yaw: float
    G: float
    risk: float
    IG: float
    distance_cost: float
    utility: Optional[float] = None
    table_bonus: Optional[float] = None
    center_p_present: Optional[float] = None
    center_class_confidence: Optional[float] = None
    center_classification_need: Optional[float] = None


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

        q_s = np.asarray(d["q_s"])
        meta_raw = d["meta_json"].item()
        if not isinstance(meta_raw, str):
            meta_raw = str(meta_raw)

        meta = json.loads(meta_raw)

        def optional_array(name: str) -> Optional[np.ndarray]:
            return np.asarray(d[name]) if name in d.files else None

        return SnapshotData(
            path=path,
            q_s=q_s,
            meta=meta,
            q_table=optional_array("q_table"),
            q_present_saved=optional_array("q_present"),
            q_class_confidence_saved=optional_array("q_class_confidence"),
            q_classification_need_saved=optional_array("q_classification_need"),
            table_map=optional_array("table_map"),
        )

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
    def _float_or_none(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    @classmethod
    def _parse_actions(cls, scored_actions: List[dict]) -> List[ActionRankRow]:
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
                        distance_cost=float(r.get("distance_cost", r.get("base_cost", 0.0))),
                        utility=cls._float_or_none(r.get("utility", None)),
                        table_bonus=cls._float_or_none(r.get("table_bonus", None)),
                        center_p_present=cls._float_or_none(r.get("center_p_present", None)),
                        center_class_confidence=cls._float_or_none(r.get("center_class_confidence", None)),
                        center_classification_need=cls._float_or_none(r.get("center_classification_need", None)),
                    )
                )
            except Exception:
                continue
        return out

    @staticmethod
    def _topk_cells(values: np.ndarray, *, k: int) -> List[Tuple[Cell, float]]:
        rows, cols = values.shape
        flat = values.reshape(-1)
        if flat.size == 0:
            return []
        k = int(max(1, min(k, flat.size)))
        idxs = np.argpartition(flat, -k)[-k:]
        idxs = idxs[np.argsort(flat[idxs])[::-1]]
        out: List[Tuple[Cell, float]] = []
        for idx in idxs:
            cy = int(idx // cols)
            cx = int(idx % cols)
            out.append(((cx, cy), float(values[cy, cx])))
        return out

    @staticmethod
    def _extract_policy_name_from_snapshot_path(snapshot_path: str) -> str:
        p = Path(snapshot_path)
        run_dir_name = None
        for part in p.parts:
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
    def _default_heatmap_output_path(snapshot_path: str, suffix: str) -> str:
        policy_name = SnapshotAnalyzer._extract_policy_name_from_snapshot_path(snapshot_path)
        out_dir = Path("webots/visuals/belief_heatmaps")
        out_dir.mkdir(parents=True, exist_ok=True)

        snap_name = Path(snapshot_path).stem
        out_name = f"{policy_name}__{snap_name}__{suffix}.png"
        return str(out_dir / out_name)

    @staticmethod
    def _q_present_from_q_s(q_s: np.ndarray) -> np.ndarray:
        if q_s.ndim != 3:
            raise ValueError(f"q_s has wrong shape: {q_s.shape}, expected 3D array")
        if q_s.shape[2] == 2:
            return q_s[:, :, 1].astype(np.float64)
        if q_s.shape[2] >= 4:
            return q_s[:, :, 1:].sum(axis=2).astype(np.float64)
        raise ValueError(f"q_s has wrong state dimension: {q_s.shape}, expected 2 or >=4 states")

    @staticmethod
    def _q_state_or_none(q_s: np.ndarray, state_idx: int) -> Optional[np.ndarray]:
        if q_s.ndim == 3 and q_s.shape[2] > state_idx:
            return q_s[:, :, state_idx].astype(np.float64)
        return None

    @staticmethod
    def _classification_confidence_and_need(q_s: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if q_s.ndim != 3 or q_s.shape[2] < 4:
            return None, None

        q_present = q_s[:, :, 1:].sum(axis=2).astype(np.float64)
        q_bottle = q_s[:, :, 2].astype(np.float64)
        q_can = q_s[:, :, 3].astype(np.float64)
        max_class = np.maximum(q_bottle, q_can)

        conf = np.zeros_like(q_present)
        mask = q_present > 1e-12
        conf[mask] = max_class[mask] / q_present[mask]
        need = q_present * (1.0 - conf)
        return conf, need

    @staticmethod
    def _shape_matches(a: Optional[np.ndarray], shape: Tuple[int, int]) -> bool:
        return isinstance(a, np.ndarray) and tuple(a.shape[:2]) == tuple(shape)

    # ---------------------------
    # Text report
    # ---------------------------

    def report(self, snap: SnapshotData, *, top_k_actions: int = 12, top_k_cells: int = 10) -> str:
        q_s = snap.q_s
        meta = snap.meta

        q_present = self._q_present_from_q_s(q_s)
        rows, cols = int(q_present.shape[0]), int(q_present.shape[1])

        q_unknown = self._q_state_or_none(q_s, 1)
        q_bottle = self._q_state_or_none(q_s, 2)
        q_can = self._q_state_or_none(q_s, 3)
        q_class_conf, q_need = self._classification_confidence_and_need(q_s)

        if snap.q_class_confidence_saved is not None and self._shape_matches(snap.q_class_confidence_saved, (rows, cols)):
            q_class_conf = snap.q_class_confidence_saved.astype(np.float64)
        if snap.q_classification_need_saved is not None and self._shape_matches(snap.q_classification_need_saved, (rows, cols)):
            q_need = snap.q_classification_need_saved.astype(np.float64)

        q_table = None
        if snap.q_table is not None and self._shape_matches(snap.q_table, (rows, cols)):
            q_table = snap.q_table.astype(np.float64)

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
        lines.append(f"- grid: rows={rows} cols={cols} q_s_shape={tuple(q_s.shape)}")
        lines.append(f"- mean={q_mean:.6f} max={q_max:.6f} argmax={argmax}")
        lines.append("- top cells:")
        for cell, p in top_cells:
            cx, cy = cell
            extras = []
            if q_unknown is not None:
                extras.append(f"unk={float(q_unknown[cy, cx]):.6f}")
            if q_bottle is not None and q_can is not None:
                extras.append(f"bot={float(q_bottle[cy, cx]):.6f}")
                extras.append(f"can={float(q_can[cy, cx]):.6f}")
            if q_class_conf is not None:
                extras.append(f"cls_conf={float(q_class_conf[cy, cx]):.6f}")
            if q_need is not None:
                extras.append(f"need={float(q_need[cy, cx]):.6f}")
            if q_table is not None:
                extras.append(f"q_table={float(q_table[cy, cx]):.6f}")
            suffix = "  " + " ".join(extras) if extras else ""
            lines.append(f"  - {cell}: present={p:.6f}{suffix}")

        if q_table is not None:
            q_table_mean = float(q_table.mean())
            q_table_max = float(q_table.max())
            q_table_argmax_flat = int(np.argmax(q_table))
            q_table_argmax = (q_table_argmax_flat % cols, q_table_argmax_flat // cols)

            lines.append("")
            lines.append("Table belief q(table)")
            lines.append(f"- mean={q_table_mean:.6f} max={q_table_max:.6f} argmax={q_table_argmax}")
            lines.append("- top table cells:")
            for cell, p in self._topk_cells(q_table, k=top_k_cells):
                cx, cy = cell
                extras = [f"present={float(q_present[cy, cx]):.6f}"]
                if q_unknown is not None:
                    extras.append(f"unk={float(q_unknown[cy, cx]):.6f}")
                if q_bottle is not None and q_can is not None:
                    extras.append(f"bot={float(q_bottle[cy, cx]):.6f}")
                    extras.append(f"can={float(q_can[cy, cx]):.6f}")
                lines.append(f"  - {cell}: q_table={p:.6f}  {' '.join(extras)}")
        else:
            lines.append("")
            lines.append("Table belief q(table)")
            lines.append("- not present in this snapshot")

        params = meta.get("params", {}) or {}
        if params:
            lines.append("")
            lines.append("Snapshot parameters")
            for key in sorted(params.keys()):
                try:
                    lines.append(f"- {key}: {float(params[key]):.6f}")
                except Exception:
                    lines.append(f"- {key}: {params[key]}")

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

                extra = ""
                if r.utility is not None:
                    extra += f" utility={r.utility:.6f}"
                if r.table_bonus is not None:
                    extra += f" table_bonus={r.table_bonus:.6f}"
                if r.center_p_present is not None:
                    extra += f" center_p={r.center_p_present:.6f}"
                if r.center_class_confidence is not None:
                    extra += f" center_cls={r.center_class_confidence:.6f}"
                if r.center_classification_need is not None:
                    extra += f" center_need={r.center_classification_need:.6f}"

                lines.append(
                    f"{i:02d}) cell={r.cell} yaw={r.yaw:.3f} "
                    f"G={r.G:.6f} risk={r.risk:.6f} IG={r.IG:.6f} "
                    f"distance_cost={r.distance_cost:.6f}{extra}{flag_txt}"
                )

        return "\n".join(lines)

    # ---------------------------
    # Heatmap rendering
    # ---------------------------

    def render_heatmap(
        self,
        values: np.ndarray,
        *,
        snapshot_path: str,
        title: str,
        colorbar_label: str,
        suffix: str,
        out_path: Optional[str] = None,
        annotate: bool = True,
        annotate_decimals: int = 2,
        figsize_scale: float = 0.55,
        show: bool = True,
        save: bool = True,
    ) -> Optional[str]:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Heatmap values must be 2D, got shape {values.shape}")

        rows, cols = values.shape

        if out_path is None:
            out_path = self._default_heatmap_output_path(snapshot_path, suffix=suffix)

        out_path_obj = Path(out_path)
        if save:
            out_path_obj.parent.mkdir(parents=True, exist_ok=True)

        fig_w = max(6.0, cols * float(figsize_scale))
        fig_h = max(5.0, rows * float(figsize_scale))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(
            values,
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
            extent=(-0.5, cols - 0.5, -0.5, rows - 0.5),
            interpolation="nearest",
            cmap="viridis",
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)

        policy_name = self._extract_policy_name_from_snapshot_path(snapshot_path)
        ax.set_title(f"{title} — {policy_name}")
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
                    val = float(values[cy, cx])
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

        saved_path: Optional[str] = None
        if save:
            fig.savefig(out_path, dpi=180)
            saved_path = str(out_path_obj)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return saved_path

    def render_belief_heatmap(
        self,
        snap: SnapshotData,
        *,
        out_path: Optional[str] = None,
        annotate: bool = True,
        annotate_decimals: int = 2,
        figsize_scale: float = 0.55,
        show: bool = True,
        save: bool = True,
    ) -> Optional[str]:
        q_present = self._q_present_from_q_s(snap.q_s)
        return self.render_heatmap(
            q_present,
            snapshot_path=snap.path,
            title="Belief heatmap: q(present) per cell",
            colorbar_label="q(present)",
            suffix="q_present",
            out_path=out_path,
            annotate=annotate,
            annotate_decimals=annotate_decimals,
            figsize_scale=figsize_scale,
            show=show,
            save=save,
        )

    def render_table_belief_heatmap(
        self,
        snap: SnapshotData,
        *,
        out_path: Optional[str] = None,
        annotate: bool = True,
        annotate_decimals: int = 2,
        figsize_scale: float = 0.55,
        show: bool = True,
        save: bool = True,
    ) -> Optional[str]:
        q_present = self._q_present_from_q_s(snap.q_s)
        if snap.q_table is None or not self._shape_matches(snap.q_table, q_present.shape):
            return None
        return self.render_heatmap(
            snap.q_table.astype(np.float64),
            snapshot_path=snap.path,
            title="Table belief heatmap: q(table) per cell",
            colorbar_label="q(table)",
            suffix="q_table",
            out_path=out_path,
            annotate=annotate,
            annotate_decimals=annotate_decimals,
            figsize_scale=figsize_scale,
            show=show,
            save=save,
        )


def analyze_snapshot_file(
    snapshot_path: str,
    *,
    top_k_actions: int = 12,
    top_k_cells: int = 10,
    show_target_heatmap: bool = True,
    save_target_heatmap: bool = True,
    show_table_heatmap: bool = True,
    save_table_heatmap: bool = True,
    annotate_heatmaps: bool = True,
) -> None:
    analyzer = SnapshotAnalyzer()
    snap = analyzer.load(snapshot_path)
    print(analyzer.report(snap, top_k_actions=top_k_actions, top_k_cells=top_k_cells))

    if show_target_heatmap or save_target_heatmap:
        out_png = analyzer.render_belief_heatmap(
            snap,
            show=show_target_heatmap,
            save=save_target_heatmap,
            annotate=annotate_heatmaps,
        )
        if out_png is not None:
            print(f"[TARGET HEATMAP] wrote {out_png}")
        else:
            print("[TARGET HEATMAP] displayed without saving")

    if show_table_heatmap or save_table_heatmap:
        q_present = analyzer._q_present_from_q_s(snap.q_s)
        has_valid_q_table = snap.q_table is not None and analyzer._shape_matches(snap.q_table, q_present.shape)

        if not has_valid_q_table:
            if snap.q_table is None:
                print("[TABLE HEATMAP] skipped: snapshot has no q_table array")
            else:
                print(f"[TABLE HEATMAP] skipped: q_table shape {snap.q_table.shape} does not match q_present shape {q_present.shape}")
        else:
            table_png = analyzer.render_table_belief_heatmap(
                snap,
                show=show_table_heatmap,
                save=save_table_heatmap,
                annotate=annotate_heatmaps,
            )
            if table_png is not None:
                print(f"[TABLE HEATMAP] wrote {table_png}")
            else:
                print("[TABLE HEATMAP] displayed without saving")


def main() -> None:
    analyze_snapshot_file(
        SNAPSHOT_PATH,
        top_k_actions=TOP_K_ACTIONS,
        top_k_cells=TOP_K_CELLS,
        show_target_heatmap=SHOW_TARGET_HEATMAP,
        save_target_heatmap=SAVE_TARGET_HEATMAP,
        show_table_heatmap=SHOW_TABLE_HEATMAP,
        save_table_heatmap=SAVE_TABLE_HEATMAP,
        annotate_heatmaps=ANNOTATE_HEATMAPS,
    )


if __name__ == "__main__":
    main()