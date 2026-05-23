# webots/controllers/worldgen_eval_supervisor/worldgen_eval_supervisor.py

from __future__ import annotations

import colorsys
import json
import math
import os
import random
import statistics
import sys
from datetime import datetime
from typing import Dict, List, Tuple

from controller import Supervisor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROLLERS_DIR = os.path.dirname(CURRENT_DIR)

if CONTROLLERS_DIR not in sys.path:
    sys.path.insert(0, CONTROLLERS_DIR)

from crazyflie_AIF.config import AIF_POLICY as DEFAULT_POLICY

# ---- Evaluation settings ----
EVAL_NUM_RUNS = 100

FIXED_10_EVAL_SEEDS = [28, 31, 34, 37, 40, 43, 46, 49, 52, 55]

WARMUP_SECONDS = 15.0
EPISODE_SECONDS = 180.0
TOTAL_SECONDS = WARMUP_SECONDS + EPISODE_SECONDS
FINALIZATION_GRACE_SECONDS = 1.0

# ---- Object setup ----
N_BOTTLES = 3
N_CANS = 3

OBJECT_SPECS = [
    {
        "count": N_BOTTLES,
        "proto": "BeerBottle",
        "class_id": 0,
        "class_name": "target_bottle",
        "model": "target_bottle",
        "def_prefix": "BOTTLE",
        "bbox_radius_m": 0.035,
        "bbox_height_m": 0.245,
    },
    {
        "count": N_CANS,
        "proto": "Can",
        "class_id": 1,
        "class_name": "target_can",
        "model": "target_can",
        "def_prefix": "CAN",
        "bbox_radius_m": 0.018,
        "bbox_height_m": 0.125,
    },
]

# Recognition-aware object spawning. Keep this enabled: YOLO/MobileNet and
# analytic GT still work with these wrapped objects. Camera-recognition GT needs it.
USE_RECOGNITION_WRAPPER_SOLIDS = True

# Arena
ARENA_X = 5.5
ARENA_Y = 6.5
CELL_SIZE_M = 0.5

# Keep objects away from walls and drone start
WALL_MARGIN = 0.35
DRONE_EXCLUSION_RADIUS = 0.75

# ---- In-cell spawn restriction ----
SPAWN_CENTER_DISC_RADIUS_M = 0.15

# Random drone spawn bounds
DRONE_SPAWN_X_MIN = -2.5
DRONE_SPAWN_X_MAX = 2.5
DRONE_SPAWN_Y_MIN = -3.0
DRONE_SPAWN_Y_MAX = 3.0
DRONE_SPAWN_YAW = 0.0

# Avoid overlap between spawned objects
MIN_SEPARATION = 0.70

# ---- Drone placement ----
DRONE_DEF = "CRAZYFLIE"

# ---- Shared files ----
SPAWNED_OBJECTS_FILENAME = "spawned_objects.json"
EXPERIMENT_INIT_FILENAME = "experiment_init.json"
TASK_STATE_FILENAME = "task_state.json"
CONTROLLER_PROGRESS_FILENAME = "controller_progress.json"
EVAL_STATE_FILENAME = "eval_state.json"

# ---- Reports ----
REPORT_SUBDIR = "run_report"
EVAL_REPORT_SUBDIR = "eval_report"
INCLUDE_RUNS_IN_EVAL_REPORT = False
INCLUDE_RUN_METRICS_IN_EVAL_REPORT = True

Cell = Tuple[int, int]


def dist2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _shared_dir() -> str:
    here = os.path.dirname(__file__)
    shared_dir = os.path.abspath(os.path.join(here, "..", "shared"))
    os.makedirs(shared_dir, exist_ok=True)
    return shared_dir


def _shared_path(filename: str) -> str:
    return os.path.join(_shared_dir(), filename)


def _controller_logs_dir() -> str:
    here = os.path.dirname(__file__)
    logs_dir = os.path.abspath(os.path.join(here, "..", "crazyflie_AIF", "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _atomic_write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _node_id(node):
    if hasattr(node, "getId"):
        return int(node.getId())
    if hasattr(node, "get_id"):
        return int(node.get_id())
    return None


def _instance_color(index: int) -> tuple[float, float, float]:
    """Stable, distinct-ish recognition color for each spawned object."""
    hue = (index * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (max(0.1, r), max(0.1, g), max(0.1, b))


def _recognition_wrapper_node_string(
    *,
    def_name: str,
    proto: str,
    model: str,
    x: float,
    y: float,
    z: float,
    color_rgb: tuple[float, float, float],
    bbox_radius_m: float,
    bbox_height_m: float,
) -> str:
    r, g, b = color_rgb
    return f'''DEF {def_name} Solid {{
  translation {x:.6f} {y:.6f} {z:.6f}
  model "{model}"
  recognitionColors [
    {r:.6f} {g:.6f} {b:.6f}
  ]
  children [
    {proto} {{
    }}
  ]
  boundingObject Cylinder {{
    height {bbox_height_m:.6f}
    radius {bbox_radius_m:.6f}
    subdivision 16
  }}
}}
'''.strip()


def _plain_proto_node_string(*, def_name: str, proto: str, x: float, y: float, z: float) -> str:
    return f'DEF {def_name} {proto} {{ translation {x:.6f} {y:.6f} {z:.6f} }}'

def _configured_eval_seeds() -> list[int]:
    """
    Default behavior:
      EVAL_NUM_RUNS = 10 -> use the historical fixed 10 seeds.

    Larger evaluations:
      EVAL_NUM_RUNS = x > 10 -> use seeds 1..x.

    This preserves comparability with all previous 10-run experiments while
    making larger evaluations easy to configure.
    """
    n = int(EVAL_NUM_RUNS)

    if n == len(FIXED_10_EVAL_SEEDS):
        return list(FIXED_10_EVAL_SEEDS)

    if n > len(FIXED_10_EVAL_SEEDS):
        return list(range(1, n + 1))

    raise ValueError(
        f"EVAL_NUM_RUNS={n} is below the fixed baseline size "
        f"{len(FIXED_10_EVAL_SEEDS)}. Use 10 or any value above 10."
    )

def _normalize_detector_limitations(payload: dict | None) -> dict:
    payload = payload or {}
    mode = str(payload.get("mode", "unknown")).lower()

    out = {
        "mode": mode,
        "frames_interval": None,
        "time_interval_s": None,
    }

    if mode == "frames":
        val = payload.get("frames_interval", None)
        out["frames_interval"] = None if val is None else int(val)
    elif mode == "time":
        val = payload.get("time_interval_s", None)
        out["time_interval_s"] = None if val is None else float(val)
    elif mode == "none":
        pass

    return out

def _normalize_experiment_metadata(payload: dict | None) -> dict:
    payload = payload or {}

    found_det_threshold = payload.get("found_det_threshold", None)
    found_streak_steps = payload.get("found_streak_steps", None)

    return {
        "detector_used": str(payload.get("detector_used", "unknown")),
        "detector_model_family": None if payload.get("detector_model_family", None) is None else str(payload.get("detector_model_family")),
        "detector_model_size": None if payload.get("detector_model_size", None) is None else str(payload.get("detector_model_size")),
        "detector_model_variant": None if payload.get("detector_model_variant", None) is None else str(payload.get("detector_model_variant")),
        "detector_model_path": None if payload.get("detector_model_path", None) is None else str(payload.get("detector_model_path")),
        "detector_backend": str(payload.get("detector_backend", "unknown")),
        "use_ground_truth_detections": bool(payload.get("use_ground_truth_detections", False)),
        "found_det_threshold": None if found_det_threshold is None else float(found_det_threshold),
        "found_streak_steps": None if found_streak_steps is None else int(found_streak_steps),
        "spawn_conditions": str(payload.get("spawn_conditions", "unknown")),
        "target_setup": str(payload.get("target_setup", "unknown")),
    }


def _normalize_target_distance_stats(payload: dict | None) -> dict:
    payload = payload or {}

    count = int(payload.get("count", 0) or 0)
    sum_m = float(payload.get("sum_m", 0.0) or 0.0)

    average_m = payload.get("average_m", None)
    min_m = payload.get("min_m", None)
    max_m = payload.get("max_m", None)

    return {
        "enabled": bool(payload.get("enabled", False)),
        "policy_supported": bool(payload.get("policy_supported", False)),
        "tracked": bool(payload.get("tracked", False)),
        "count": count,
        "sum_m": sum_m,
        "average_m": None if average_m is None else float(average_m),
        "min_m": None if min_m is None else float(min_m),
        "max_m": None if max_m is None else float(max_m),
    }

def _cxcy_from_xy(x: float, y: float) -> tuple[int, int] | None:
    min_x = -ARENA_X / 2.0
    min_y = -ARENA_Y / 2.0
    max_x = ARENA_X / 2.0
    max_y = ARENA_Y / 2.0

    if x < min_x or x >= max_x or y < min_y or y >= max_y:
        return None

    cx = int((x - min_x) / CELL_SIZE_M)
    cy = int((y - min_y) / CELL_SIZE_M)
    return (cx, cy)


def _cell_center_xy(cx: int, cy: int) -> tuple[float, float]:
    min_x = -ARENA_X / 2.0
    min_y = -ARENA_Y / 2.0
    x = min_x + (cx + 0.5) * CELL_SIZE_M
    y = min_y + (cy + 0.5) * CELL_SIZE_M
    return (x, y)


def _uniform_point_in_disc(center_x: float, center_y: float, radius: float) -> tuple[float, float]:
    r = radius * math.sqrt(random.random())
    theta = random.uniform(0.0, 2.0 * math.pi)
    x = center_x + r * math.cos(theta)
    y = center_y + r * math.sin(theta)
    return (x, y)


def _eligible_spawn_cells(radius: float) -> list[tuple[int, int]]:
    cols = int(ARENA_X / CELL_SIZE_M)
    rows = int(ARENA_Y / CELL_SIZE_M)

    xmin = -ARENA_X / 2.0 + WALL_MARGIN
    xmax = +ARENA_X / 2.0 - WALL_MARGIN
    ymin = -ARENA_Y / 2.0 + WALL_MARGIN
    ymax = +ARENA_Y / 2.0 - WALL_MARGIN

    cells = []
    for cy in range(rows):
        for cx in range(cols):
            ccx, ccy = _cell_center_xy(cx, cy)

            if (ccx - radius) < xmin:
                continue
            if (ccx + radius) > xmax:
                continue
            if (ccy - radius) < ymin:
                continue
            if (ccy + radius) > ymax:
                continue

            cells.append((cx, cy))
    return cells


def _write_spawned_objects_json(path: str, *, seed: int, objects: list[dict]) -> None:
    payload = {
        "seed": int(seed),
        "objects": objects,
    }
    _atomic_write_json(path, payload)


def _write_experiment_init_json(path: str, *, seed: int, drone_spawn_xy, drone_spawn_yaw: float):
    x, y = drone_spawn_xy
    cxcy = _cxcy_from_xy(float(x), float(y))
    payload = {
        "seed": int(seed),
        "drone_spawn": {
            "x": float(x),
            "y": float(y),
            "yaw": float(drone_spawn_yaw),
            "cx": int(cxcy[0]) if cxcy is not None else None,
            "cy": int(cxcy[1]) if cxcy is not None else None,
        },
    }
    _atomic_write_json(path, payload)


def _write_task_state(
    path: str,
    *,
    status: str,
    reason: str | None,
    seed: int,
    total_objects: int,
    remaining_objects: int,
    validated_found_cells: set,
    timeout_s: float,
    warmup_s: float,
    episode_s: float,
    now_time_s: float,
    report_written: bool,
) -> None:
    payload = {
        "status": str(status),
        "reason": reason,
        "seed": int(seed),
        "total_objects": int(total_objects),
        "remaining_objects": int(remaining_objects),
        "validated_found_cells": sorted([[int(c[0]), int(c[1])] for c in validated_found_cells]),
        "timeout_s": float(timeout_s),
        "warmup_s": float(warmup_s),
        "episode_s": float(episode_s),
        "now_time_s": float(now_time_s),
        "report_written": bool(report_written),
    }
    _atomic_write_json(path, payload)


def _empty_classification_stats() -> dict:
    return {
        "unique_claims_evaluated": 0,
        "correct_class_claims": 0,
        "wrong_class_claims": 0,
        "empty_cell_claims": 0,
        "duplicate_claims_ignored": 0,
        "object_cell_claims": 0,
        "object_cell_class_accuracy": 0.0,
        "overall_claim_precision": 0.0,
        "confusion_matrix": {},
    }


def _finalize_classification_stats(stats: dict) -> dict:
    object_cell_claims = int(stats["correct_class_claims"]) + int(stats["wrong_class_claims"])
    unique_claims = int(stats["unique_claims_evaluated"])
    correct_claims = int(stats["correct_class_claims"])

    stats["object_cell_claims"] = object_cell_claims
    stats["object_cell_class_accuracy"] = (
        float(correct_claims) / float(object_cell_claims) if object_cell_claims > 0 else 0.0
    )
    stats["overall_claim_precision"] = (
        float(correct_claims) / float(unique_claims) if unique_claims > 0 else 0.0
    )
    return stats


def _update_confusion_matrix(confusion_matrix: dict, true_name: str, pred_name: str) -> None:
    if true_name not in confusion_matrix:
        confusion_matrix[true_name] = {}
    confusion_matrix[true_name][pred_name] = int(confusion_matrix[true_name].get(pred_name, 0)) + 1


def _claim_key(claim: dict) -> tuple:
    pred_name = claim.get("class_name", None)
    if pred_name is None:
        pred_name = claim.get("label", None)
    return (
        int(claim.get("cx")),
        int(claim.get("cy")),
        str(pred_name) if pred_name is not None else "unknown",
    )


def _extract_claims_from_progress(progress: dict) -> list[dict]:
    claims = []

    raw_objects = progress.get("claimed_found_objects", [])
    if isinstance(raw_objects, list):
        for obj in raw_objects:
            if not isinstance(obj, dict):
                continue
            if "cx" not in obj or "cy" not in obj:
                continue

            class_name = obj.get("class_name", None)
            if class_name is None:
                class_name = obj.get("label", None)

            claims.append(
                {
                    "cx": int(obj["cx"]),
                    "cy": int(obj["cy"]),
                    "class_id": None if obj.get("class_id", None) is None else int(obj["class_id"]),
                    "class_name": None if class_name is None else str(class_name),
                    "label": None if obj.get("label", None) is None else str(obj.get("label")),
                    "confidence": None if obj.get("confidence", None) is None else float(obj["confidence"]),
                }
            )

    # Fallback for older controller payloads
    if not claims:
        raw_cells = progress.get("claimed_found_cells", [])
        if isinstance(raw_cells, list):
            for cell in raw_cells:
                if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                    continue
                claims.append(
                    {
                        "cx": int(cell[0]),
                        "cy": int(cell[1]),
                        "class_id": None,
                        "class_name": None,
                        "label": None,
                        "confidence": None,
                    }
                )

    return claims


def _evaluate_new_claims(
    *,
    claims: list[dict],
    seen_claim_keys: set,
    true_objects_by_cell: dict,
    validated_found_cells: set,
    classification_stats: dict,
) -> set:
    new_validated_found_cells = set(validated_found_cells)

    for claim in claims:
        key = _claim_key(claim)
        if key in seen_claim_keys:
            classification_stats["duplicate_claims_ignored"] = int(classification_stats["duplicate_claims_ignored"]) + 1
            continue

        seen_claim_keys.add(key)
        classification_stats["unique_claims_evaluated"] = int(classification_stats["unique_claims_evaluated"]) + 1

        cell = (int(claim["cx"]), int(claim["cy"]))
        pred_class_id = claim.get("class_id", None)
        pred_class_name = claim.get("class_name", None)
        if pred_class_name is None:
            pred_class_name = claim.get("label", None)

        gt_obj = true_objects_by_cell.get(cell, None)
        if gt_obj is None:
            classification_stats["empty_cell_claims"] = int(classification_stats["empty_cell_claims"]) + 1
            continue

        gt_class_id = int(gt_obj["class_id"])
        gt_class_name = str(gt_obj["class_name"])

        is_correct = False

        if pred_class_name is not None:
            is_correct = (str(pred_class_name) == gt_class_name)
        elif pred_class_id is not None:
            is_correct = (int(pred_class_id) == gt_class_id)

        if is_correct:
            classification_stats["correct_class_claims"] = int(classification_stats["correct_class_claims"]) + 1
            _update_confusion_matrix(
                classification_stats["confusion_matrix"],
                gt_class_name,
                str(pred_class_name or gt_class_name),
            )
            new_validated_found_cells.add(cell)
        else:
            classification_stats["wrong_class_claims"] = int(classification_stats["wrong_class_claims"]) + 1
            _update_confusion_matrix(
                classification_stats["confusion_matrix"],
                gt_class_name,
                str(pred_class_name or (f"class_{pred_class_id}" if pred_class_id is not None else "unknown")),
            )

    return new_validated_found_cells


def _write_run_report(
    *,
    run_dir: str,
    policy_name: str,
    gt_seed: int,
    total_objects: int,
    validated_found_cells: set,
    remaining_objects: int,
    autonomy_start_time_s: float | None,
    now_time_s: float,
    timeout_s: float,
    visited_cells: set,
    grid_rows: int,
    grid_cols: int,
    reason: str,
    classification_stats: dict,
    detector_limitations: dict,
    experiment_metadata: dict,
    target_distance_stats: dict,
) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    report_dir = os.path.join(run_dir, REPORT_SUBDIR)
    os.makedirs(report_dir, exist_ok=True)

    found_count = int(len(validated_found_cells))
    elapsed = (
        float(now_time_s - autonomy_start_time_s)
        if autonomy_start_time_s is not None
        else float(now_time_s)
    )

    coverage_cells = int(len(visited_cells))
    total_cells = int(grid_rows * grid_cols)
    coverage_ratio = float(coverage_cells) / float(total_cells) if total_cells > 0 else 0.0

    cls_stats = _finalize_classification_stats(dict(classification_stats))

    payload = {
        "policy": str(policy_name),
        "gt_seed": int(gt_seed),
        "reason": str(reason),
        "detector_used": str(experiment_metadata.get("detector_used", "unknown")),
        "detector_model_family": experiment_metadata.get("detector_model_family", None),
        "detector_model_size": experiment_metadata.get("detector_model_size", None),
        "detector_model_variant": experiment_metadata.get("detector_model_variant", None),
        "detector_model_path": experiment_metadata.get("detector_model_path", None),
        "spawn_conditions": str(experiment_metadata.get("spawn_conditions", "unknown")),
        "target_setup": str(experiment_metadata.get("target_setup", "unknown")),
        "found_det_threshold": experiment_metadata.get("found_det_threshold", None),
        "found_streak_steps": experiment_metadata.get("found_streak_steps", None),
        "target_distance_stats": dict(target_distance_stats),
        "autonomy_start_time_s": autonomy_start_time_s,
        "now_time_s": float(now_time_s),
        "elapsed_s": float(elapsed),
        "timeout_s": float(timeout_s),
        "total_objects": int(total_objects),
        "found_count": int(found_count),
        "remaining_objects": int(remaining_objects),
        "found_cells": sorted([[int(c[0]), int(c[1])] for c in validated_found_cells]),
        "coverage": {
            "visited_cells": int(coverage_cells),
            "total_cells": int(total_cells),
            "coverage_ratio": float(coverage_ratio),
        },
        "classification": cls_stats,
        "detector_limitations": dict(detector_limitations),
    }

    out_path = os.path.join(run_dir, REPORT_SUBDIR, "run_report.json")
    _atomic_write_json(out_path, payload)
    print(f"[RUN REPORT] wrote {out_path}", flush=True)
    return payload


def _write_eval_report(eval_dir: str, policy_name: str, seeds: list[int], runs: list[dict]) -> str:
    report_dir = os.path.join(eval_dir, EVAL_REPORT_SUBDIR)
    os.makedirs(report_dir, exist_ok=True)

    n = len(runs)

    def _std(vals) -> float:
        return float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0

    if n == 0:
        payload = {
            "policy": policy_name,
            "num_runs": 0,
            "seeds": seeds,
        }

        if INCLUDE_RUN_METRICS_IN_EVAL_REPORT:
            payload["run_metrics"] = []

        if INCLUDE_RUNS_IN_EVAL_REPORT:
            payload["runs"] = []
    else:
        elapsed_vals = [float(r["elapsed_s"]) for r in runs]
        found_vals = [int(r["found_count"]) for r in runs]
        remaining_vals = [int(r.get("remaining_objects", 0)) for r in runs]
        total_object_vals = [int(f + rem) for f, rem in zip(found_vals, remaining_vals)]
        coverage_vals = [float(r["coverage_ratio"]) for r in runs]
        success_vals = [1 if str(r["reason"]) == "done" else 0 for r in runs]

        total_found = int(sum(found_vals))
        total_target_objects = int(sum(total_object_vals))
        detection_rate_percent = (
            100.0 * float(total_found) / float(total_target_objects)
            if total_target_objects > 0
            else 0.0
        )

        total_unique_claims = sum(int(r["classification"]["unique_claims_evaluated"]) for r in runs)
        total_correct = sum(int(r["classification"]["correct_class_claims"]) for r in runs)
        total_wrong = sum(int(r["classification"]["wrong_class_claims"]) for r in runs)
        total_empty = sum(int(r["classification"]["empty_cell_claims"]) for r in runs)
        total_duplicates = sum(int(r["classification"]["duplicate_claims_ignored"]) for r in runs)
        total_object_cell_claims = sum(int(r["classification"]["object_cell_claims"]) for r in runs)

        target_stats = [
            _normalize_target_distance_stats(r.get("target_distance_stats", {}))
            for r in runs
        ]

        target_distance_total_count = sum(int(s["count"]) for s in target_stats)
        target_distance_total_sum_m = sum(float(s["sum_m"]) for s in target_stats)

        target_distance_average_m = (
            float(target_distance_total_sum_m) / float(target_distance_total_count)
            if target_distance_total_count > 0
            else None
        )

        target_distance_mins = [
            float(s["min_m"])
            for s in target_stats
            if s.get("min_m", None) is not None
        ]

        target_distance_maxs = [
            float(s["max_m"])
            for s in target_stats
            if s.get("max_m", None) is not None
        ]

        target_distance_summary = {
            "enabled": any(bool(s["enabled"]) for s in target_stats),
            "policy_supported": any(bool(s["policy_supported"]) for s in target_stats),
            "tracked": any(bool(s["tracked"]) for s in target_stats),
            "total_target_selections": int(target_distance_total_count),
            "total_target_distance_m": float(target_distance_total_sum_m),
            "average_target_distance_m": target_distance_average_m,
            "min_target_distance_m": min(target_distance_mins) if target_distance_mins else None,
            "max_target_distance_m": max(target_distance_maxs) if target_distance_maxs else None,
        }

        run_metrics = []
        for r, total_objects_for_run in zip(runs, total_object_vals):
            cls = r.get("classification", {}) or {}
            td_stats = _normalize_target_distance_stats(r.get("target_distance_stats", {}))

            found_count = int(r.get("found_count", 0))
            run_detection_rate_percent = (
                100.0 * float(found_count) / float(total_objects_for_run)
                if total_objects_for_run > 0
                else 0.0
            )

            run_metrics.append(
                {
                    "run_index": int(r.get("run_index", len(run_metrics))),
                    "seed": int(r.get("seed", -1)),
                    "reason": str(r.get("reason", "unknown")),
                    "elapsed_s": float(r.get("elapsed_s", 0.0)),
                    "found_count": found_count,
                    "total_objects": int(total_objects_for_run),
                    "detection_rate_percent": float(run_detection_rate_percent),
                    "coverage_ratio": float(r.get("coverage_ratio", 0.0)),
                    "target_distance_average_m": td_stats.get("average_m", None),
                    "overall_claim_precision": float(cls.get("overall_claim_precision", 0.0)),
                    "object_cell_class_accuracy": float(cls.get("object_cell_class_accuracy", 0.0)),
                }
            )

        detector_limitations = dict(runs[0].get("detector_limitations", {})) if runs else {}

        first_run = runs[0] if runs else {}

        detector_used = str(first_run.get("detector_used", "unknown"))
        detector_model_family = first_run.get("detector_model_family", None)
        detector_model_size = first_run.get("detector_model_size", None)
        detector_model_variant = first_run.get("detector_model_variant", None)
        detector_model_path = first_run.get("detector_model_path", None)
        spawn_conditions = str(first_run.get("spawn_conditions", "unknown"))
        target_setup = str(first_run.get("target_setup", "unknown"))
        found_det_threshold = first_run.get("found_det_threshold", None)
        found_streak_steps = first_run.get("found_streak_steps", None)

        payload = {
            "policy": policy_name,
            "num_runs": n,
            "seeds": seeds,
            "detector_used": detector_used,
            "detector_model_family": detector_model_family,
            "detector_model_size": detector_model_size,
            "detector_model_variant": detector_model_variant,
            "detector_model_path": detector_model_path,
            "spawn_conditions": spawn_conditions,
            "target_setup": target_setup,
            "found_det_threshold": found_det_threshold,
            "found_streak_steps": found_streak_steps,
            "detector_limitations": detector_limitations,
            "target_distance_summary": target_distance_summary,
            "summary": {
                "success_rate": float(sum(success_vals) / n),
                "average_simulation_time_s": float(sum(elapsed_vals) / n),
                "std_simulation_time_s": _std(elapsed_vals),
                "average_found_count": float(sum(found_vals) / n),
                "std_found_count": _std(found_vals),
                "detection_rate_percent": float(detection_rate_percent),
                "average_coverage_ratio": float(sum(coverage_vals) / n),
                "std_coverage_ratio": _std(coverage_vals),
                "min_simulation_time_s": float(min(elapsed_vals)),
                "max_simulation_time_s": float(max(elapsed_vals)),
                "min_found_count": int(min(found_vals)),
                "max_found_count": int(max(found_vals)),
                "min_coverage_ratio": float(min(coverage_vals)),
                "max_coverage_ratio": float(max(coverage_vals)),
            },
            "classification_summary": {
                "total_unique_claims_evaluated": int(total_unique_claims),
                "total_correct_class_claims": int(total_correct),
                "total_wrong_class_claims": int(total_wrong),
                "total_empty_cell_claims": int(total_empty),
                "total_duplicate_claims_ignored": int(total_duplicates),
                "total_object_cell_claims": int(total_object_cell_claims),
                "global_overall_claim_precision": (
                    float(total_correct) / float(total_unique_claims) if total_unique_claims > 0 else 0.0
                ),
                "global_object_cell_class_accuracy": (
                    float(total_correct) / float(total_object_cell_claims) if total_object_cell_claims > 0 else 0.0
                ),
            },
        }

        if INCLUDE_RUN_METRICS_IN_EVAL_REPORT:
            payload["run_metrics"] = run_metrics

        if INCLUDE_RUNS_IN_EVAL_REPORT:
            payload["runs"] = runs

    out_path = os.path.join(report_dir, "eval_report.json")
    _atomic_write_json(out_path, payload)
    print(f"[EVAL REPORT] wrote {out_path}", flush=True)
    return out_path


def _init_or_resume_eval_state(eval_state_path: str, policy_name: str, seeds: list[int]) -> dict:
    eval_state = _read_json(eval_state_path, None)

    if isinstance(eval_state, dict) and not bool(eval_state.get("completed", False)):
        return eval_state

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(_controller_logs_dir(), f"eval_{policy_name}_{ts}")

    eval_state = {
        "policy_name": policy_name,
        "seeds": list(seeds),
        "total_runs": len(seeds),
        "current_run_index": 0,
        "current_seed": int(seeds[0]),
        "eval_dir": eval_dir,
        "aggregate_runs": [],
        "completed": False,
    }
    _atomic_write_json(eval_state_path, eval_state)
    return eval_state


def _sample_drone_spawn_xy():
    x = random.uniform(DRONE_SPAWN_X_MIN, DRONE_SPAWN_X_MAX)
    y = random.uniform(DRONE_SPAWN_Y_MIN, DRONE_SPAWN_Y_MAX)
    return (x, y)


def _place_drone(supervisor: Supervisor, drone_node, *, x: float, y: float, yaw: float) -> None:
    if drone_node is None:
        raise RuntimeError(
            f"Drone DEF '{DRONE_DEF}' not found. Set DRONE_DEF to the drone DEF used in your world."
        )

    translation_field = drone_node.getField("translation")
    rotation_field = drone_node.getField("rotation")

    current_translation = translation_field.getSFVec3f()
    z = float(current_translation[2])

    translation_field.setSFVec3f([float(x), float(y), z])
    rotation_field.setSFRotation([0.0, 0.0, 1.0, float(yaw)])

    drone_node.resetPhysics()
    supervisor.simulationResetPhysics()


def _cleanup_spawned_nodes(spawn_children) -> None:
    for idx in range(spawn_children.getCount() - 1, -1, -1):
        node = spawn_children.getMFNode(idx)
        node.remove()


def main():
    if not (0.0 < SPAWN_CENTER_DISC_RADIUS_M < (0.5 * CELL_SIZE_M)):
        raise ValueError(
            f"SPAWN_CENTER_DISC_RADIUS_M must be in (0, {0.5 * CELL_SIZE_M:.3f}), "
            f"got {SPAWN_CENTER_DISC_RADIUS_M}"
        )

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    policy_name = os.environ.get("AIF_POLICY", DEFAULT_POLICY)

    spawned_objects_path = _shared_path(SPAWNED_OBJECTS_FILENAME)
    experiment_init_path = _shared_path(EXPERIMENT_INIT_FILENAME)
    task_state_path = _shared_path(TASK_STATE_FILENAME)
    controller_progress_path = _shared_path(CONTROLLER_PROGRESS_FILENAME)
    eval_state_path = _shared_path(EVAL_STATE_FILENAME)

    _safe_remove(spawned_objects_path)
    _safe_remove(experiment_init_path)
    _safe_remove(task_state_path)
    _safe_remove(controller_progress_path)

    print("[EVAL] shared state cleared", flush=True)

    configured_seeds = _configured_eval_seeds()
    eval_state = _init_or_resume_eval_state(eval_state_path, policy_name, configured_seeds)

    seeds = [int(s) for s in eval_state["seeds"]]
    total_runs = int(eval_state["total_runs"])
    run_index = int(eval_state["current_run_index"])
    current_seed = int(eval_state["current_seed"])
    eval_dir = str(eval_state["eval_dir"])
    aggregate_runs = list(eval_state.get("aggregate_runs", []))

    random.seed(current_seed)

    _write_task_state(
        task_state_path,
        status="running",
        reason=None,
        seed=current_seed,
        total_objects=0,
        remaining_objects=0,
        validated_found_cells=set(),
        timeout_s=TOTAL_SECONDS,
        warmup_s=WARMUP_SECONDS,
        episode_s=EPISODE_SECONDS,
        now_time_s=0.0,
        report_written=False,
    )

    _atomic_write_json(
        controller_progress_path,
        {
            "policy_name": policy_name,
            "run_dir": None,
            "autonomy_start_time_s": None,
            "last_controller_time_s": 0.0,
            "visited_cells": [],
            "claimed_found_cells": [],
            "claimed_found_objects": [],
            "detector_limitations": {
                "mode": "unknown",
                "frames_interval": None,
                "time_interval_s": None,
            },
            "drone_spawn": None,
        },
    )

    print(
        f"[EVAL] initialized run {run_index + 1}/{total_runs} "
        f"policy={policy_name} seed={current_seed}",
        flush=True,
    )

    spawn_root = supervisor.getFromDef("SPAWN_ROOT")
    if spawn_root is None:
        print("ERROR: DEF SPAWN_ROOT Group not found in the world.")
        return
    spawn_children = spawn_root.getField("children")

    drone_node = supervisor.getFromDef(DRONE_DEF)
    drone_xy = _sample_drone_spawn_xy()
    _place_drone(
        supervisor,
        drone_node,
        x=drone_xy[0],
        y=drone_xy[1],
        yaw=DRONE_SPAWN_YAW,
    )

    eligible_cells = _eligible_spawn_cells(SPAWN_CENTER_DISC_RADIUS_M)
    if not eligible_cells:
        raise RuntimeError(
            "No eligible spawn cells available under current wall margin and center-disc radius."
        )

    print(
        f"[SPAWN] using center-disc spawn restriction with radius={SPAWN_CENTER_DISC_RADIUS_M:.3f} m "
        f"over {len(eligible_cells)} eligible cells",
        flush=True,
    )

    spawned_positions: List[Tuple[float, float]] = []
    spawned_objects: List[dict] = []

    def sample_position(max_tries=2000):
        for _ in range(max_tries):
            cx, cy = random.choice(eligible_cells)
            center_x, center_y = _cell_center_xy(cx, cy)
            x, y = _uniform_point_in_disc(center_x, center_y, SPAWN_CENTER_DISC_RADIUS_M)

            sampled_cell = _cxcy_from_xy(x, y)
            if sampled_cell != (cx, cy):
                continue

            if dist2((x, y), drone_xy) < DRONE_EXCLUSION_RADIUS ** 2:
                continue

            ok = True
            for (px, py) in spawned_positions:
                if dist2((x, y), (px, py)) < MIN_SEPARATION ** 2:
                    ok = False
                    break
            if not ok:
                continue

            return (x, y)

        return None

    global_index = 0
    for spec in OBJECT_SPECS:
        for local_i in range(int(spec["count"])):
            pos = sample_position()
            if pos is None:
                print(
                    f"WARNING: Could not place {spec['class_name']} #{local_i} under constraints.",
                    flush=True,
                )
                break

            x, y = pos
            spawned_positions.append((x, y))

            def_name = f"{spec['def_prefix']}_{global_index:03d}"
            z = float(spec.get("z", 0.0))
            color_rgb = _instance_color(global_index)

            if USE_RECOGNITION_WRAPPER_SOLIDS:
                node_string = _recognition_wrapper_node_string(
                    def_name=def_name,
                    proto=str(spec["proto"]),
                    model=str(spec["model"]),
                    x=float(x),
                    y=float(y),
                    z=z,
                    color_rgb=color_rgb,
                    bbox_radius_m=float(spec["bbox_radius_m"]),
                    bbox_height_m=float(spec["bbox_height_m"]),
                )
            else:
                node_string = _plain_proto_node_string(
                    def_name=def_name,
                    proto=str(spec["proto"]),
                    x=float(x),
                    y=float(y),
                    z=z,
                )

            spawn_children.importMFNodeFromString(-1, node_string)
            new_node = spawn_children.getMFNode(spawn_children.getCount() - 1)

            cxcy = _cxcy_from_xy(x, y)
            spawned_objects.append(
                {
                    "def": def_name,
                    "id": _node_id(new_node),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "class_id": int(spec["class_id"]),
                    "class_name": str(spec["class_name"]),
                    "model": str(spec["model"]),
                    "recognition_color_rgb": [
                        float(color_rgb[0]),
                        float(color_rgb[1]),
                        float(color_rgb[2]),
                    ],
                    "cx": None if cxcy is None else int(cxcy[0]),
                    "cy": None if cxcy is None else int(cxcy[1]),
                }
            )
            global_index += 1

    print(
        f"[EVAL] run {run_index + 1}/{total_runs} "
        f"policy={policy_name} seed={current_seed} "
        f"drone_spawn=({drone_xy[0]:.3f},{drone_xy[1]:.3f}) "
        f"spawned={len(spawned_objects)}",
        flush=True,
    )

    _write_experiment_init_json(
        experiment_init_path,
        seed=current_seed,
        drone_spawn_xy=drone_xy,
        drone_spawn_yaw=DRONE_SPAWN_YAW,
    )
    print(f"Experiment init JSON exported: {experiment_init_path}", flush=True)

    _write_spawned_objects_json(
        spawned_objects_path,
        seed=current_seed,
        objects=spawned_objects,
    )
    print(f"GT JSON exported: {spawned_objects_path}", flush=True)

    true_objects_by_cell: Dict[Cell, dict] = {}
    for obj in spawned_objects:
        if obj.get("cx", None) is None or obj.get("cy", None) is None:
            continue
        true_objects_by_cell[(int(obj["cx"]), int(obj["cy"]))] = obj

    total_objects = len(true_objects_by_cell)
    validated_found_cells = set()
    classification_stats = _empty_classification_stats()
    seen_claim_keys = set()

    report_written = False
    final_reason = None
    final_time_s = None

    _write_task_state(
        task_state_path,
        status="running",
        reason=None,
        seed=current_seed,
        total_objects=total_objects,
        remaining_objects=total_objects,
        validated_found_cells=validated_found_cells,
        timeout_s=TOTAL_SECONDS,
        warmup_s=WARMUP_SECONDS,
        episode_s=EPISODE_SECONDS,
        now_time_s=0.0,
        report_written=False,
    )

    spawn_cxcy = _cxcy_from_xy(drone_xy[0], drone_xy[1])

    _atomic_write_json(
        controller_progress_path,
        {
            "policy_name": None,
            "run_dir": None,
            "autonomy_start_time_s": None,
            "last_controller_time_s": 0.0,
            "visited_cells": [],
            "claimed_found_cells": [],
            "claimed_found_objects": [],
            "detector_limitations": {
                "mode": "unknown",
                "frames_interval": None,
                "time_interval_s": None,
            },
            "drone_spawn": {
                "x": float(drone_xy[0]),
                "y": float(drone_xy[1]),
                "yaw": float(DRONE_SPAWN_YAW),
                "cx": spawn_cxcy[0] if spawn_cxcy is not None else None,
                "cy": spawn_cxcy[1] if spawn_cxcy is not None else None,
            },
        },
    )

    start_time = supervisor.getTime()

    while supervisor.step(timestep) != -1:
        t = supervisor.getTime()
        elapsed_total = float(t - start_time)

        controller_progress = _read_json(
            controller_progress_path,
            {
                "policy_name": None,
                "run_dir": None,
                "autonomy_start_time_s": None,
                "last_controller_time_s": 0.0,
                "visited_cells": [],
                "claimed_found_cells": [],
                "claimed_found_objects": [],
                "detector_limitations": {
                    "mode": "unknown",
                    "frames_interval": None,
                    "time_interval_s": None,
                },
            },
        )

        detector_limitations = _normalize_detector_limitations(
            controller_progress.get("detector_limitations", {})
        )

        experiment_metadata = _normalize_experiment_metadata(
            controller_progress.get("experiment_metadata", {})
        )

        target_distance_stats = _normalize_target_distance_stats(
            controller_progress.get("target_distance_stats", {})
        )

        claims = _extract_claims_from_progress(controller_progress)
        validated_found_cells = _evaluate_new_claims(
            claims=claims,
            seen_claim_keys=seen_claim_keys,
            true_objects_by_cell=true_objects_by_cell,
            validated_found_cells=validated_found_cells,
            classification_stats=classification_stats,
        )

        remaining_objects = max(0, int(total_objects) - int(len(validated_found_cells)))

        if len(validated_found_cells) >= total_objects and total_objects > 0:
            final_reason = "done"
            final_time_s = float(t)
        elif elapsed_total >= TOTAL_SECONDS:
            final_reason = "timeout"
            final_time_s = float(t)

        if final_reason is None:
            _write_task_state(
                task_state_path,
                status="running",
                reason=None,
                seed=current_seed,
                total_objects=total_objects,
                remaining_objects=remaining_objects,
                validated_found_cells=validated_found_cells,
                timeout_s=TOTAL_SECONDS,
                warmup_s=WARMUP_SECONDS,
                episode_s=EPISODE_SECONDS,
                now_time_s=float(t),
                report_written=report_written,
            )
            continue

        if not report_written:
            _write_task_state(
                task_state_path,
                status=final_reason,
                reason=final_reason,
                seed=current_seed,
                total_objects=total_objects,
                remaining_objects=remaining_objects,
                validated_found_cells=validated_found_cells,
                timeout_s=TOTAL_SECONDS,
                warmup_s=WARMUP_SECONDS,
                episode_s=EPISODE_SECONDS,
                now_time_s=float(final_time_s),
                report_written=False,
            )

            run_dir = controller_progress.get("run_dir", None)
            autonomy_start_time_s = controller_progress.get("autonomy_start_time_s", None)
            visited_cells = {
                (int(cx), int(cy))
                for (cx, cy) in controller_progress.get("visited_cells", [])
            }

            if run_dir:
                run_payload = _write_run_report(
                    run_dir=run_dir,
                    policy_name=policy_name,
                    gt_seed=current_seed,
                    total_objects=total_objects,
                    validated_found_cells=validated_found_cells,
                    remaining_objects=remaining_objects,
                    autonomy_start_time_s=autonomy_start_time_s,
                    now_time_s=float(final_time_s),
                    timeout_s=TOTAL_SECONDS,
                    visited_cells=visited_cells,
                    grid_rows=int(ARENA_Y / CELL_SIZE_M),
                    grid_cols=int(ARENA_X / CELL_SIZE_M),
                    reason=final_reason,
                    classification_stats=classification_stats,
                    detector_limitations=detector_limitations,
                    experiment_metadata=experiment_metadata,
                    target_distance_stats=target_distance_stats,
                )
            else:
                print("[RUN REPORT] run_dir not available yet, skipping file write", flush=True)
                coverage_cells = int(len(visited_cells))
                total_cells = int(int(ARENA_Y / CELL_SIZE_M) * int(ARENA_X / CELL_SIZE_M))
                coverage_ratio = float(coverage_cells) / float(total_cells) if total_cells > 0 else 0.0
                cls_stats = _finalize_classification_stats(dict(classification_stats))
                run_payload = {
                    "policy": policy_name,
                    "gt_seed": current_seed,
                    "reason": final_reason,
                    "detector_used": str(experiment_metadata.get("detector_used", "unknown")),
                    "detector_model_family": experiment_metadata.get("detector_model_family", None),
                    "detector_model_size": experiment_metadata.get("detector_model_size", None),
                    "detector_model_variant": experiment_metadata.get("detector_model_variant", None),
                    "detector_model_path": experiment_metadata.get("detector_model_path", None),
                    "spawn_conditions": str(experiment_metadata.get("spawn_conditions", "unknown")),
                    "target_setup": str(experiment_metadata.get("target_setup", "unknown")),
                    "found_det_threshold": experiment_metadata.get("found_det_threshold", None),
                    "found_streak_steps": experiment_metadata.get("found_streak_steps", None),
                    "target_distance_stats": dict(target_distance_stats),
                    "autonomy_start_time_s": autonomy_start_time_s,
                    "now_time_s": float(final_time_s),
                    "elapsed_s": float(final_time_s - autonomy_start_time_s) if autonomy_start_time_s is not None else float(final_time_s),
                    "timeout_s": TOTAL_SECONDS,
                    "total_objects": total_objects,
                    "found_count": len(validated_found_cells),
                    "remaining_objects": remaining_objects,
                    "found_cells": sorted([[int(c[0]), int(c[1])] for c in validated_found_cells]),
                    "coverage": {
                        "visited_cells": coverage_cells,
                        "total_cells": total_cells,
                        "coverage_ratio": coverage_ratio,
                    },
                    "classification": cls_stats,
                    "detector_limitations": dict(detector_limitations),    
                }

            aggregate_runs.append(
                {
                    "run_index": run_index,
                    "seed": current_seed,
                    "policy": policy_name,
                    "reason": str(run_payload["reason"]),
                    "detector_used": str(run_payload.get("detector_used", "unknown")),
                    "detector_model_family": run_payload.get("detector_model_family", None),
                    "detector_model_size": run_payload.get("detector_model_size", None),
                    "detector_model_variant": run_payload.get("detector_model_variant", None),
                    "detector_model_path": run_payload.get("detector_model_path", None),
                    "spawn_conditions": str(run_payload.get("spawn_conditions", "unknown")),
                    "target_setup": str(run_payload.get("target_setup", "unknown")),
                    "found_det_threshold": run_payload.get("found_det_threshold", None),
                    "found_streak_steps": run_payload.get("found_streak_steps", None),
                    "target_distance_stats": dict(run_payload.get("target_distance_stats", {})),
                    "elapsed_s": float(run_payload["elapsed_s"]),
                    "found_count": int(run_payload["found_count"]),
                    "remaining_objects": int(run_payload["remaining_objects"]),
                    "coverage_ratio": float(run_payload["coverage"]["coverage_ratio"]),
                    "classification": dict(run_payload["classification"]),
                    "detector_limitations": dict(run_payload["detector_limitations"]),
                    "run_dir": run_dir,
                }
            )

            report_written = True
            is_last_run = (run_index + 1) >= total_runs

            if is_last_run:
                _write_eval_report(
                    eval_dir=eval_dir,
                    policy_name=policy_name,
                    seeds=seeds,
                    runs=aggregate_runs,
                )

                eval_state = {
                    "policy_name": policy_name,
                    "seeds": seeds,
                    "total_runs": total_runs,
                    "current_run_index": run_index,
                    "current_seed": current_seed,
                    "eval_dir": eval_dir,
                    "aggregate_runs": aggregate_runs,
                    "completed": True,
                }
                _atomic_write_json(eval_state_path, eval_state)

                _write_task_state(
                    task_state_path,
                    status=final_reason,
                    reason=final_reason,
                    seed=current_seed,
                    total_objects=total_objects,
                    remaining_objects=remaining_objects,
                    validated_found_cells=validated_found_cells,
                    timeout_s=TOTAL_SECONDS,
                    warmup_s=WARMUP_SECONDS,
                    episode_s=EPISODE_SECONDS,
                    now_time_s=float(final_time_s),
                    report_written=True,
                )

                print(f"[EVAL] completed all {total_runs} runs", flush=True)

                grace_start = supervisor.getTime()
                while supervisor.step(timestep) != -1:
                    if (supervisor.getTime() - grace_start) >= FINALIZATION_GRACE_SECONDS:
                        break

                _cleanup_spawned_nodes(spawn_children)
                print("Cleanup complete. Quitting simulation.", flush=True)
                supervisor.simulationQuit(0)
                return

            else:
                next_run_index = run_index + 1
                next_seed = int(seeds[next_run_index])

                eval_state = {
                    "policy_name": policy_name,
                    "seeds": seeds,
                    "total_runs": total_runs,
                    "current_run_index": next_run_index,
                    "current_seed": next_seed,
                    "eval_dir": eval_dir,
                    "aggregate_runs": aggregate_runs,
                    "completed": False,
                }
                _atomic_write_json(eval_state_path, eval_state)

                _write_task_state(
                    task_state_path,
                    status=final_reason,
                    reason=final_reason,
                    seed=current_seed,
                    total_objects=total_objects,
                    remaining_objects=remaining_objects,
                    validated_found_cells=validated_found_cells,
                    timeout_s=TOTAL_SECONDS,
                    warmup_s=WARMUP_SECONDS,
                    episode_s=EPISODE_SECONDS,
                    now_time_s=float(final_time_s),
                    report_written=True,
                )

                print(
                    f"[EVAL] finished run {run_index + 1}/{total_runs}; "
                    f"reloading world for next seed={next_seed}",
                    flush=True,
                )

                grace_start = supervisor.getTime()
                while supervisor.step(timestep) != -1:
                    if (supervisor.getTime() - grace_start) >= FINALIZATION_GRACE_SECONDS:
                        break

                _cleanup_spawned_nodes(spawn_children)
                supervisor.worldReload()
                return


if __name__ == "__main__":
    main()