from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Mapping, Optional

from .registry import EnvironmentSpec


# Two surface models. "typed" (default) is the 73/100 model
# (eval_aif_zoi_v2_mbnetv2_hard_20261014_160738): hard cell types from the
# controller's table map, with the values of that campaign, restored on
# 2026-10-19 after the centre-rule experiment. "joint" (roadmap 3.1) puts the
# surface in the belief state; its defaults are all computed below from one
# calibration set and the scenario description. Any value can be overridden
# through the environment variable named in ENV_VARS.

# ---- Detector calibration (offline) ----
# One calibration set: hard dataset world, MobileNet SSD, seed 101 (no eval run
# uses it), 400 images, confidence 0.45, NMS IoU 0.40, matching at IoU 0.5.
# Images: detection_model_training/datasets/hard_calib_seed101; drone poses and
# desk positions from the identical re-collection
# collection_hard_pseudo_random_20261019_114053. Counts from
# tools/v2_analysis/calibrate_desk.py.
CALIB_IMAGES = 400
CALIB_TARGETS = 167            # labelled bottles and cans
CALIB_TARGETS_DETECTED = 140   # matched by a target box
CALIB_CLASS_CORRECT = 140      # matched boxes with the right class
CALIB_FALSE_BOXES = 2          # target boxes matching no labelled target
CALIB_EMPTY_CELL_FRAMES = 3491  # empty cells passing the update gate, summed over images
# Desk evidence on cells whose table-plane coverage passes the update gate.
CALIB_TABLE_CELLS = 189        # cells whose centre is on a desk top
CALIB_TABLE_CELLS_COVERED = 181
CALIB_FLOOR_CELLS = 1488
CALIB_FLOOR_CELLS_COVERED = 188


def rule_of_succession(successes: float, trials: float) -> float:
    """Posterior mean of a rate under a uniform prior: (k + 1) / (n + 2).

    It is also the posterior predictive probability that the next trial
    succeeds, which is what the likelihood needs.
    """
    return (float(successes) + 1.0) / (float(trials) + 2.0)


SUCCESSION_RECALL = rule_of_succession(CALIB_TARGETS_DETECTED, CALIB_TARGETS)            # 0.834
SUCCESSION_CLASS_ACCURACY = rule_of_succession(CALIB_CLASS_CORRECT, CALIB_TARGETS_DETECTED)  # 0.993
SUCCESSION_FALSE_ALARM = rule_of_succession(CALIB_FALSE_BOXES, CALIB_EMPTY_CELL_FRAMES)  # 0.00086
DEFAULT_DESK_RECALL = rule_of_succession(CALIB_TABLE_CELLS_COVERED, CALIB_TABLE_CELLS)   # 0.953
DEFAULT_DESK_FALSE_ALARM = rule_of_succession(CALIB_FLOOR_CELLS_COVERED, CALIB_FLOOR_CELLS)  # 0.127

# 73/100 values: 140/167; rule-of-three bound 1 - 3/140; (2+1)/400 per image over
# ~6 cells per image, rounded.
DEFAULT_RECALL = 0.84             # AIF_ZOI_V2_RECALL
DEFAULT_CLASS_ACCURACY = 0.98     # AIF_ZOI_V2_CLASS_ACCURACY
DEFAULT_FALSE_ALARM = 0.001       # AIF_ZOI_V2_FALSE_ALARM

# ---- Scenario (hard and medium worlds) ----
SCENARIO_TARGETS = 6
SCENARIO_P_ON_TABLE = 0.9
SCENARIO_TABLES = 3
DESK_LENGTH_M = 1.2
DESK_WIDTH_M = 0.7
CELL_SIZE_M = 0.5
GRID_CELLS = 11 * 13

# Table cells per desk under each table-map rule. "centre": a cell is a table
# cell when its centre falls inside a desk box. The box is axis-aligned in the
# image, so for a desk at a uniformly random angle relative to the camera its
# mean area is L*W + (L^2 + W^2) / pi, and the expected number of cell centres
# inside it is that area over the cell area. "touched": the controller's map
# (every cell a desk-box probe hits), with the cells-whose-centre-is-on-the-desk
# count L*W / cell area. The centre rule lost most runs (3/15 against 11/15 on
# seeds 1-15, 2026-10-16) and is kept only as an opt-in switch.
TABLE_MAP_RULES = ("centre", "touched")
DEFAULT_TABLE_MAP_RULE = "touched"


def table_cells_per_desk(rule: str) -> float:
    area = DESK_LENGTH_M * DESK_WIDTH_M
    if rule == "centre":
        area += (DESK_LENGTH_M ** 2 + DESK_WIDTH_M ** 2) / math.pi
    elif rule != "touched":
        raise ValueError(f"unknown table map rule {rule!r}")
    return area / (CELL_SIZE_M ** 2)


def scenario_priors(rule: str) -> dict:
    """lambda_T = N*pi / C, lambda_F = N*(1 - pi) / (M - C), rho = C / M."""
    c = SCENARIO_TABLES * table_cells_per_desk(rule)
    return {
        "lambda_table": SCENARIO_TARGETS * SCENARIO_P_ON_TABLE / c,
        "lambda_floor": SCENARIO_TARGETS * (1.0 - SCENARIO_P_ON_TABLE) / (GRID_CELLS - c),
        "rho": c / GRID_CELLS,
    }


# Environments without tables: all six targets are on the floor, spread over the
# 11 x 13 grid, and no cell can become a table cell.
DEFAULT_LAMBDA_FLOOR_NO_TABLES = SCENARIO_TARGETS / GRID_CELLS
DEFAULT_RHO_NO_TABLES = 0.0

# 73/100 values: 6*0.9/10, 0.6/133 and 10/143 with C = 3 desks x 3.4 cells,
# rounded. scenario_priors("touched") gives 0.536, 0.00451, 0.0705.
DEFAULT_LAMBDA_TABLE = 0.54       # AIF_ZOI_V2_LAMBDA_TABLE
DEFAULT_LAMBDA_FLOOR = 0.0045     # AIF_ZOI_V2_LAMBDA_FLOOR
DEFAULT_RHO = 0.07                # AIF_ZOI_V2_RHO


def default_priors(rule: str) -> dict:
    """Default priors for a table-map rule: the 73/100 values for "touched"."""
    if rule == "touched":
        return {"lambda_table": DEFAULT_LAMBDA_TABLE, "lambda_floor": DEFAULT_LAMBDA_FLOOR, "rho": DEFAULT_RHO}
    return scenario_priors(rule)

SURFACE_MODELS = ("typed", "joint")
DEFAULT_SURFACE_MODEL = "typed"

# Planner ranking. "distance_cost" (default, as aif_zoi): G = -beta*IG - kappa*U + eps*d.
# "rate": (beta*IG + kappa*U) / T, with T the navigator's arrival time plus one
# detector period (policies/planner_time.py), so moves cost seconds, not a weight.
PLANNER_OBJECTIVES = ("distance_cost", "rate")
DEFAULT_PLANNER_OBJECTIVE = "distance_cost"

# Planner terms. "info_and_class" (default, as aif_zoi): beta*IG plus the
# classification bonus kappa*U_class. "info_only": IG alone. The belief keeps
# bottle and can as separate states, so IG already values resolving the class,
# and with "rate" the score IG / T has no weight at all.
PLANNER_TERMS = ("info_and_class", "info_only")
DEFAULT_PLANNER_TERMS = "info_and_class"

ENV_VARS = {
    "recall": "AIF_ZOI_V2_RECALL",
    "class_accuracy": "AIF_ZOI_V2_CLASS_ACCURACY",
    "false_alarm": "AIF_ZOI_V2_FALSE_ALARM",
    "lambda_table": "AIF_ZOI_V2_LAMBDA_TABLE",
    "lambda_floor": "AIF_ZOI_V2_LAMBDA_FLOOR",
    "rho": "AIF_ZOI_V2_RHO",
    "desk_recall": "AIF_ZOI_V2_DESK_RECALL",
    "desk_false_alarm": "AIF_ZOI_V2_DESK_FALSE_ALARM",
}
TABLE_MAP_RULE_ENV_VAR = "AIF_ZOI_V2_TABLE_MAP"
SURFACE_MODEL_ENV_VAR = "AIF_ZOI_V2_SURFACE_MODEL"
PLANNER_OBJECTIVE_ENV_VAR = "AIF_ZOI_V2_PLANNER"
PLANNER_TERMS_ENV_VAR = "AIF_ZOI_V2_PLANNER_TERMS"


@dataclass(frozen=True)
class AIFZOIV2Params:
    """Model parameters of aif_zoi_v2.

    recall, class_accuracy and false_alarm define the per-cell observation
    likelihood. lambda_table, lambda_floor and rho define the per-cell target
    prior for table, floor and not-yet-observed cells. With surface_model
    "typed", table_map_rule says which cells a desk box makes table cells, and
    the priors must be computed for the same rule. With "joint", the surface is
    part of the belief, and desk_recall / desk_false_alarm are the probabilities
    that a desk box covers a table / floor cell whose table plane is in view.
    """

    recall: float = DEFAULT_RECALL
    class_accuracy: float = DEFAULT_CLASS_ACCURACY
    false_alarm: float = DEFAULT_FALSE_ALARM
    lambda_table: float = DEFAULT_LAMBDA_TABLE
    lambda_floor: float = DEFAULT_LAMBDA_FLOOR
    rho: float = DEFAULT_RHO
    table_map_rule: str = DEFAULT_TABLE_MAP_RULE
    surface_model: str = DEFAULT_SURFACE_MODEL
    planner_objective: str = DEFAULT_PLANNER_OBJECTIVE
    planner_terms: str = DEFAULT_PLANNER_TERMS
    desk_recall: float = DEFAULT_DESK_RECALL
    desk_false_alarm: float = DEFAULT_DESK_FALSE_ALARM
    table_context_enabled: bool = True

    def __post_init__(self) -> None:
        # Strict open intervals: an exact 0 or 1 puts an exact zero in the
        # likelihood or prior, and a state driven to probability 0 never recovers.
        for name in ("recall", "class_accuracy", "false_alarm", "lambda_table", "lambda_floor",
                     "desk_recall", "desk_false_alarm"):
            value = float(getattr(self, name))
            if not (0.0 < value < 1.0):
                raise ValueError(f"aif_zoi_v2 {name} must be in (0, 1), got {value}")
        if not (0.0 <= float(self.rho) <= 1.0):
            raise ValueError(f"aif_zoi_v2 rho must be in [0, 1], got {self.rho}")
        if self.table_map_rule not in TABLE_MAP_RULES:
            raise ValueError(
                f"aif_zoi_v2 table_map_rule must be one of {TABLE_MAP_RULES}, got {self.table_map_rule!r}")
        if self.surface_model not in SURFACE_MODELS:
            raise ValueError(
                f"aif_zoi_v2 surface_model must be one of {SURFACE_MODELS}, got {self.surface_model!r}")
        if self.planner_objective not in PLANNER_OBJECTIVES:
            raise ValueError(
                f"aif_zoi_v2 planner_objective must be one of {PLANNER_OBJECTIVES}, got {self.planner_objective!r}")
        if self.planner_terms not in PLANNER_TERMS:
            raise ValueError(
                f"aif_zoi_v2 planner_terms must be one of {PLANNER_TERMS}, got {self.planner_terms!r}")
        if not float(self.desk_recall) > float(self.desk_false_alarm):
            raise ValueError(
                "aif_zoi_v2 desk_recall must exceed desk_false_alarm, "
                f"got {self.desk_recall} <= {self.desk_false_alarm}")

    def as_dict(self) -> dict:
        return asdict(self)


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name, None)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def resolve_aif_zoi_v2_params(
    environment_spec: EnvironmentSpec,
    environ: Optional[Mapping[str, str]] = None,
) -> AIFZOIV2Params:
    environ = os.environ if environ is None else environ
    has_tables = "desk" in environment_spec.context_classes

    rule = str(environ.get(TABLE_MAP_RULE_ENV_VAR, "") or DEFAULT_TABLE_MAP_RULE).strip().lower()
    if rule not in TABLE_MAP_RULES:
        raise ValueError(f"{TABLE_MAP_RULE_ENV_VAR} must be one of {TABLE_MAP_RULES}, got {rule!r}")
    surface_model = str(environ.get(SURFACE_MODEL_ENV_VAR, "") or DEFAULT_SURFACE_MODEL).strip().lower()
    if surface_model not in SURFACE_MODELS:
        raise ValueError(f"{SURFACE_MODEL_ENV_VAR} must be one of {SURFACE_MODELS}, got {surface_model!r}")
    planner = str(environ.get(PLANNER_OBJECTIVE_ENV_VAR, "") or DEFAULT_PLANNER_OBJECTIVE).strip().lower()
    if planner not in PLANNER_OBJECTIVES:
        raise ValueError(f"{PLANNER_OBJECTIVE_ENV_VAR} must be one of {PLANNER_OBJECTIVES}, got {planner!r}")
    terms = str(environ.get(PLANNER_TERMS_ENV_VAR, "") or DEFAULT_PLANNER_TERMS).strip().lower()
    if terms not in PLANNER_TERMS:
        raise ValueError(f"{PLANNER_TERMS_ENV_VAR} must be one of {PLANNER_TERMS}, got {terms!r}")

    if surface_model == "joint":
        # Everything from the calibration counts and the scenario: a table cell is a
        # cell whose centre is on a desk top (C = 3 desks x L*W / cell area).
        detector = (SUCCESSION_RECALL, SUCCESSION_CLASS_ACCURACY, SUCCESSION_FALSE_ALARM)
        priors = scenario_priors("touched")
    else:
        detector = (DEFAULT_RECALL, DEFAULT_CLASS_ACCURACY, DEFAULT_FALSE_ALARM)
        priors = default_priors(rule)

    defaults = {
        "recall": detector[0],
        "class_accuracy": detector[1],
        "false_alarm": detector[2],
        "lambda_table": priors["lambda_table"],
        "lambda_floor": priors["lambda_floor"] if has_tables else DEFAULT_LAMBDA_FLOOR_NO_TABLES,
        "rho": priors["rho"] if has_tables else DEFAULT_RHO_NO_TABLES,
        "desk_recall": DEFAULT_DESK_RECALL,
        "desk_false_alarm": DEFAULT_DESK_FALSE_ALARM,
    }
    values = {
        key: _env_float(environ, ENV_VARS[key], default)
        for key, default in defaults.items()
    }
    return AIFZOIV2Params(
        **values, table_map_rule=rule, surface_model=surface_model, planner_objective=planner,
        planner_terms=terms, table_context_enabled=has_tables)
