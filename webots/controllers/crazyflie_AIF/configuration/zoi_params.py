from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Mapping, Optional

from .registry import EnvironmentSpec

# Model parameters of the ZOI policy. They come from two places only:
#
# - the offline calibration of the detector in use: target recall r, class
#   accuracy a, false alarm f, desk recall d and desk false alarm g, counted on
#   one calibration image set and turned into probabilities by the rule of
#   succession;
# - the scenario description: the share of targets on desks, pi.
#
# The share of desk cells (rho) and the target density (theta) are learned in
# flight (policies/zoi_model.py). Every value can be overridden through the
# environment variables in ENV_VARS.


def rule_of_succession(successes: float, trials: float) -> float:
    """Posterior mean of a rate under a uniform prior: (k + 1) / (n + 2).

    It is also the posterior predictive probability that the next trial
    succeeds, which is what the likelihood needs.
    """
    return (float(successes) + 1.0) / (float(trials) + 2.0)


@dataclass(frozen=True)
class DetectorCalibration:
    """Counts from one detector on the calibration set (tools/analysis/calibrate_desk.py)."""

    targets: int              # labelled bottles and cans
    targets_detected: int     # matched by a target box
    class_correct: int        # matched boxes with the right class
    false_boxes: int          # target boxes matching no labelled target
    empty_cell_frames: int    # empty cells passing the update gate, summed over images
    # Desk counts; None in environments without desks (the desk values are then unused).
    table_cells: Optional[int] = None          # cells whose centre is on a desk top, table plane in view
    table_cells_covered: Optional[int] = None  # ... covered by a desk box
    floor_cells: Optional[int] = None          # floor cells, table plane in view
    floor_cells_covered: Optional[int] = None  # ... covered by a desk box

    def rates(self) -> dict:
        out = {
            "recall": rule_of_succession(self.targets_detected, self.targets),
            "class_accuracy": rule_of_succession(self.class_correct, self.targets_detected),
            "false_alarm": rule_of_succession(self.false_boxes, self.empty_cell_frames),
            "desk_recall": None,
            "desk_false_alarm": None,
        }
        if self.table_cells is not None:
            out["desk_recall"] = rule_of_succession(self.table_cells_covered, self.table_cells)
            out["desk_false_alarm"] = rule_of_succession(self.floor_cells_covered, self.floor_cells)
        return out


# ---- Detector calibration (offline), per (environment, detector) checkpoint ----
# One set per environment: dataset world, seed 101 (no training or eval run uses it),
# 400 images, confidence 0.45, NMS IoU 0.40; a target box matches a label when its
# centre is inside the label box (as in flight, where a box goes to the cell of its
# centre). Built by tools/analysis/build_calibration.sh.
# Hard: dataset world, seed 101 (no eval run uses it), 400 images. Images:
# detection_model_training/datasets/hard_calib_seed101; drone poses and desk
# positions from the identical re-collection collection_hard_pseudo_random_20261019_114053.
# MobileNet predictions: hard_calib_seed101/eval/mobilenet_ssd/val_eval/predictions.json;
# YOLO nano: tools/analysis/yolo_calib_predictions.py (CPU), saved in
# hard_calib_seed101/eval/yolo_nano/deployed/predictions.json.
DETECTOR_CALIBRATIONS = {
    ("hard", "mobilenet_ssd"): DetectorCalibration(
        targets=167, targets_detected=140, class_correct=140, false_boxes=2, empty_cell_frames=3491,
        table_cells=189, table_cells_covered=181, floor_cells=1488, floor_cells_covered=188),
    ("hard", "yolo_nano"): DetectorCalibration(
        targets=167, targets_detected=167, class_correct=167, false_boxes=6, empty_cell_frames=3491,
        table_cells=189, table_cells_covered=182, floor_cells=1488, floor_cells_covered=198),
    # Medium: detection_model_training/datasets/medium_calib_seed101; poses and desks from the
    # identical re-collection collection_medium_pseudo_random_20261102_171055.
    ("medium", "mobilenet_ssd"): DetectorCalibration(
        targets=187, targets_detected=182, class_correct=182, false_boxes=0, empty_cell_frames=3415,
        table_cells=238, table_cells_covered=227, floor_cells=1414, floor_cells_covered=252),
    ("medium", "yolo_nano"): DetectorCalibration(
        targets=187, targets_detected=187, class_correct=187, false_boxes=3, empty_cell_frames=3415,
        table_cells=238, table_cells_covered=229, floor_cells=1414, floor_cells_covered=260),
    # Easy (no desks: r, a, f only): detection_model_training/datasets/easy_calib_seed101,
    # collection_easy_pseudo_random_20261102_164341 (EASY_DATASET_SEED=101).
    ("easy", "mobilenet_ssd"): DetectorCalibration(
        targets=333, targets_detected=301, class_correct=301, false_boxes=23, empty_cell_frames=3346),
    ("easy", "yolo_nano"): DetectorCalibration(
        targets=333, targets_detected=321, class_correct=321, false_boxes=17, empty_cell_frames=3346),
}
DEFAULT_ENVIRONMENT = "hard"
DEFAULT_DETECTOR = "mobilenet_ssd"

# ---- Scenario ----
# Share of targets on desks, pi, as given to the policy. The default 0.5 tells
# the policy nothing about where targets are more likely. The medium and hard
# worlds place 90% of the targets on desks (MEDIUM_EVAL_ / HARD_EVAL_P_TARGET_ON_TABLE).
DEFAULT_P_ON_TABLE = 0.5

ENV_VARS = {
    "recall": "ZOI_RECALL",
    "class_accuracy": "ZOI_CLASS_ACCURACY",
    "false_alarm": "ZOI_FALSE_ALARM",
    "desk_recall": "ZOI_DESK_RECALL",
    "desk_false_alarm": "ZOI_DESK_FALSE_ALARM",
    "p_target_on_table": "ZOI_P_ON_TABLE",
}
TARGET_KEYS = ("recall", "class_accuracy", "false_alarm")
DESK_KEYS = ("desk_recall", "desk_false_alarm")
CALIBRATED_KEYS = (*TARGET_KEYS, *DESK_KEYS)

# How the planner is computed, with the same decisions: "vectorized" (default,
# policies/zoi_vectorized.py) or "loop", the plain Python reference (policies/zoi.py).
PLANNER_IMPLS = ("loop", "vectorized")
DEFAULT_PLANNER_IMPL = "vectorized"
PLANNER_IMPL_ENV_VAR = "ZOI_PLANNER_IMPL"

# Points per side of the (rho, theta) grid over (0, 1) x (0, 1); feasible points
# only are kept (~half). A numerical resolution, not a model parameter: 128, 64,
# 32 and 16 gave the same success rate (600 runs each); 32 (about 512 points, ~8 kB)
# fits the GAP8 memory.
DEFAULT_GRID_SIZE = 32
GRID_SIZE_ENV_VAR = "ZOI_GRID_SIZE"


@dataclass(frozen=True)
class ZOIParams:
    """Model parameters of the ZOI policy.

    recall, class_accuracy and false_alarm define the target likelihood;
    desk_recall and desk_false_alarm the probabilities that a desk box covers a
    table / floor cell whose table plane is in view (None, and unused, in
    environments without desks). p_target_on_table is the share of targets on
    desks (unused without desks). calibration names the (environment, detector)
    whose counts gave the detector values.
    """

    recall: float
    class_accuracy: float
    false_alarm: float
    desk_recall: Optional[float] = None
    desk_false_alarm: Optional[float] = None
    p_target_on_table: float = DEFAULT_P_ON_TABLE
    table_context_enabled: bool = True
    calibration: str = f"{DEFAULT_ENVIRONMENT}/{DEFAULT_DETECTOR}"
    planner_impl: str = DEFAULT_PLANNER_IMPL
    grid_size: int = DEFAULT_GRID_SIZE

    def __post_init__(self) -> None:
        if int(self.grid_size) < 8:
            raise ValueError(f"ZOI grid_size must be >= 8, got {self.grid_size}")
        if self.planner_impl not in PLANNER_IMPLS:
            raise ValueError(f"ZOI planner_impl must be one of {PLANNER_IMPLS}, got {self.planner_impl!r}")
        # Strict open intervals: an exact 0 or 1 puts an exact zero in the
        # likelihood or prior, and a state driven to probability 0 never recovers.
        desk_given = [getattr(self, name) is not None for name in DESK_KEYS]
        if self.table_context_enabled and not all(desk_given):
            raise ValueError("ZOI needs desk_recall and desk_false_alarm in an environment with desks")
        if any(desk_given) and not all(desk_given):
            raise ValueError("ZOI desk_recall and desk_false_alarm must be given together")
        names = (*TARGET_KEYS, "p_target_on_table", *(DESK_KEYS if all(desk_given) else ()))
        for name in names:
            value = float(getattr(self, name))
            if not (0.0 < value < 1.0):
                raise ValueError(f"ZOI {name} must be in (0, 1), got {value}")
        if all(desk_given) and not float(self.desk_recall) > float(self.desk_false_alarm):
            raise ValueError(
                "ZOI desk_recall must exceed desk_false_alarm, "
                f"got {self.desk_recall} <= {self.desk_false_alarm}")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def calibrated(cls, environment: str = DEFAULT_ENVIRONMENT, detector: str = DEFAULT_DETECTOR,
                   **overrides) -> "ZOIParams":
        """Values from the calibration of (environment, detector), with overrides."""
        values = {**DETECTOR_CALIBRATIONS[(environment, detector)].rates(),
                  "calibration": f"{environment}/{detector}", **overrides}
        return cls(**values)


def _env_float(environ: Mapping[str, str], name: str) -> Optional[float]:
    raw = environ.get(name, None)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def resolve_zoi_params(
    environment_spec: EnvironmentSpec,
    environ: Optional[Mapping[str, str]] = None,
    detector_id: Optional[str] = None,
) -> ZOIParams:
    """Parameters for an environment and detector (default: MobileNet SSD).

    The detector values come from the calibration of that environment's checkpoint
    of the detector; a pair without a calibration needs the values given through
    the environment variables (the three target values, plus the two desk values
    in an environment with desks).
    """
    environ = os.environ if environ is None else environ
    accepted = (*ENV_VARS.values(), PLANNER_IMPL_ENV_VAR, GRID_SIZE_ENV_VAR)
    unknown = sorted(k for k in environ if k.startswith("ZOI_") and k not in accepted)
    if unknown:
        raise ValueError(
            f"unknown ZOI variables {unknown} (accepted: {sorted(accepted)})")

    environment = environment_spec.environment_id
    detector = DEFAULT_DETECTOR if detector_id is None else str(detector_id)
    given = {key: _env_float(environ, var) for key, var in ENV_VARS.items()}
    given = {key: value for key, value in given.items() if value is not None}

    has_desks = "desk" in environment_spec.context_classes
    needed = CALIBRATED_KEYS if has_desks else TARGET_KEYS
    calibration = DETECTOR_CALIBRATIONS.get((environment, detector), None)
    if calibration is not None:
        values = {**calibration.rates(), "calibration": f"{environment}/{detector}"}
    elif all(key in given for key in needed):
        values = {"calibration": "environment variables"}
    else:
        missing = [ENV_VARS[k] for k in needed if k not in given]
        raise ValueError(
            f"ZOI has no detector calibration for {environment}/{detector} "
            f"(calibrated: {sorted('/'.join(k) for k in DETECTOR_CALIBRATIONS)}); "
            f"calibrate it (tools/analysis/calibrate_desk.py) or set {missing}")
    values.update(given)
    planner_impl = str(environ.get(PLANNER_IMPL_ENV_VAR, "") or DEFAULT_PLANNER_IMPL).strip().lower()
    grid_size = _env_float(environ, GRID_SIZE_ENV_VAR)
    if grid_size is not None and grid_size != int(grid_size):
        raise ValueError(f"{GRID_SIZE_ENV_VAR} must be an integer, got {grid_size}")
    if not has_desks:
        # Without desks the desk values are unused; keep them only if given together.
        for key in DESK_KEYS:
            values.setdefault(key, None)
        if values["desk_recall"] is None or values["desk_false_alarm"] is None:
            values["desk_recall"] = values["desk_false_alarm"] = None
    return ZOIParams(**values, table_context_enabled=has_desks,
                          planner_impl=planner_impl,
                          grid_size=DEFAULT_GRID_SIZE if grid_size is None else int(grid_size))
