#!/usr/bin/env python3
"""Reference trace for the C port of the ZOI policy (embedded/zoi).

Replays one run's perception log (logged pose and detector boxes, rounded as
logged) through the existing Python code, frame by frame, and writes what the
C code must reproduce:

- frame geometry: floor/table/merged coverage and visible mask per cell
  (env/env_manager.py, env/surface_projection.py), projected target events,
  desk-covered cells, shadow cells and the persistent table map
  (vision/perception_projection.py), new claims (task/claim_manager.py);
- belief: q_joint of every cell and the (rho, theta) posterior summary
  (policies/zoi_vectorized.py, through compute_action);
- planner: the decision from the current pose at every frame (the planner is
  side-effect free, so every frame is a test point), and every candidate's
  IG and T every --cand-every frames.

The reference is this replay, not the flight: the log is rounded and a few
desk boxes are not logged, so rho/theta differ slightly from the logged ones
(printed at the end as a sanity check).

Text format, one record per line, floats as exact hex (float.hex(), read in C
with strtof):

    ZOITRACE 1
    ENV <environment> <detector>
    SOURCE <run directory, relative to the repo>
    PARAMS <r> <a> <f> <d> <g> <pi> <grid_size> <table_context 0/1>
    FRAME <i> <t> <x> <y> <yaw> <n_boxes>
    BOX <class_id> <conf> <x1> <y1> <x2> <y2>              (n_boxes lines)
    COV <cx> <cy> <floor> <table> <merged> <visible>        (every cell in view)
    TEVT <class_id> <conf> <cx> <cy> <surface 0 floor / 1 table>
    DESKCOV <n> <cx> <cy> ...                               (this frame's desk boxes)
    SHADOW <n> <cx> <cy> ...
    TMAP <n> <cx> <cy> ...                                  (persistent, after the frame)
    CLAIM <cx> <cy> <class_id>
    RT <rho_mean> <rho_sd> <theta_mean> <theta_sd>
    Q <cx> <cy> <q0> ... <q5>                               (every cell)
    REPLANS <n>                                             (plans the flight made after this frame)
    PLAN <cur_cx> <cur_cy> <cx> <cy> <yaw_index> <rate> <T>
    CAND <k> <IG> <T>                                       (every --cand-every frames)
    END

Usage (from the repo root):
    python tools/c_port/export_trace.py <run_dir or perception_log.jsonl> -o trace.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CTRL = os.path.join(REPO, "webots", "controllers", "crazyflie_AIF")
sys.path.insert(0, CTRL)
sys.path.insert(0, os.path.dirname(CTRL))

from configuration.zoi_params import ZOIParams, resolve_zoi_params  # noqa: E402
from configuration.registry import get_environment_spec  # noqa: E402
from env.env_manager import FoVSpec, GridEnvManager, GridSpec  # noqa: E402
from env.surface_projection import (  # noqa: E402
    effective_table_projection_cells,
    merge_visual_coverages_by_surface,
)
from policies.zoi_vectorized import ZOIVectorizedPolicy  # noqa: E402
from runtime.controller_pipeline import (  # noqa: E402
    PolicyPerceptionEvidence,
    TaskStateSnapshot,
    build_policy_observation,
    detection_event_rc_to_cxcy,
    mapping_rc_to_cxcy,
)
from task.claim_manager import ClaimManager  # noqa: E402
from vision.perception_projection import (  # noqa: E402
    PerceptionProjectionConfig,
    project_perception_frame,
)

# Controller constants, as in crazyflie_AIF.py (not importable outside Webots).
TARGET_NAMES, TARGET_IDS = {"target_bottle", "target_can"}, {0, 1}
DESK_NAMES, DESK_IDS = {"desk"}, {2}
SURFACE_DETECTION_CONF_THRESHOLD = 0.30
TABLE_SHADOW_FORWARD_TOLERANCE_M = 0.10
CLAIM_CONFIDENCE_THRESHOLD = 0.45
FLOOR_HEIGHT_M = 0.0
DETECTOR_ID = {"mbnetv2": "mobilenet_ssd", "mobilenet_ssd": "mobilenet_ssd",
               "yolonano": "yolo_nano", "yolo_nano": "yolo_nano"}


def h(x) -> str:
    return float(x).hex()


class _Result:
    def __init__(self, events):
        self.processed = True
        self.raw_output = None
        self.events = events


class _StubPerception:
    """process_if_due returns the logged boxes of the current frame."""

    def __init__(self):
        self.events = []

    def process_if_due(self, **_):
        return _Result(self.events)


def _cells_line(tag, cells_cxcy) -> str:
    cells = sorted((int(c[0]), int(c[1])) for c in cells_cxcy)
    return f"{tag} {len(cells)}" + "".join(f" {cx} {cy}" for cx, cy in cells)


def load_log(path: str):
    if os.path.isdir(path):
        path = os.path.join(path, "perception_log.jsonl")
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    header = next(r for r in records if r.get("type") == "header")
    frames = [r for r in records if r.get("type") == "frame"]
    return header, frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="run directory or its perception_log.jsonl")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--cand-every", type=int, default=10, help="dump all candidates every N frames (0: never)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0: all)")
    ap.add_argument("--autonomous-only", action="store_true", default=True)
    args = ap.parse_args()

    header, frames = load_log(args.log)
    run_dir = args.log if os.path.isdir(args.log) else os.path.dirname(os.path.abspath(args.log))
    if args.autonomous_only:
        frames = [fr for fr in frames if fr.get("autonomous", True)]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    environment = str(header["environment"])
    detector = DETECTOR_ID[str(header["detector"])]
    img_w, img_h = (int(v) for v in header["image_size"])
    g = header["grid"]
    rows, cols, cs = int(g["rows"]), int(g["cols"]), float(g["cell_size_m"])
    min_x, min_y = float(g["min_x"]), float(g["min_y"])
    grid_spec = GridSpec(min_x=min_x, max_x=min_x + cols * cs, min_y=min_y, max_y=min_y + rows * cs,
                         cell_size=cs, rows=rows, cols=cols)
    fov = header["fov_spec"]
    floor_fov = FoVSpec(**{k: float(fov["floor"][k]) for k in ("half_left", "half_right", "forward", "backward")})
    table_fov = FoVSpec(**{k: float(fov["table"][k]) for k in ("half_left", "half_right", "forward", "backward")})
    env_floor, env_table = GridEnvManager(grid_spec, floor_fov), GridEnvManager(grid_spec, table_fov)
    coverage_threshold = float(header["coverage_threshold"])
    detection_every_s = float(header["detection_every_s"])
    table_height_m = float(header["table_height_m"])

    params = resolve_zoi_params(get_environment_spec(environment), environ={}, detector_id=detector)
    # Use the run's own parameters when it recorded them (e.g. a pi sweep or another grid size).
    run_params = os.path.join(run_dir, "policy_parameters.json")
    if os.path.isfile(run_params):
        with open(run_params) as f:
            model = json.load(f).get("model", {})
        fields = set(ZOIParams.__dataclass_fields__)
        params = ZOIParams(**{k: v for k, v in model.items() if k in fields})
    policy = ZOIVectorizedPolicy(params=params, detector_period_s=detection_every_s)

    proj_cfg = PerceptionProjectionConfig(
        target_class_names=frozenset(TARGET_NAMES), target_class_ids=frozenset(TARGET_IDS),
        desk_class_names=frozenset(DESK_NAMES), desk_class_ids=frozenset(DESK_IDS),
        surface_detection_conf_threshold=SURFACE_DETECTION_CONF_THRESHOLD,
        enable_table_shadow_projection=True,
        table_shadow_forward_tolerance_m=TABLE_SHADOW_FORWARD_TOLERANCE_M,
        min_x=min_x, min_y=min_y, cell_size_m=cs)
    stub = _StubPerception()
    claims = ClaimManager(confidence_threshold=CLAIM_CONFIDENCE_THRESHOLD,
                          target_class_names=TARGET_NAMES, target_class_ids=TARGET_IDS)
    persistent_table_cells: set = set()
    persistent_table_conf: dict = {}

    nan = float("nan")
    out = open(args.output, "w")
    w = out.write
    w("ZOITRACE 1\n")
    w(f"ENV {environment} {detector}\n")
    w(f"SOURCE {os.path.relpath(os.path.abspath(run_dir), REPO)}\n")
    d = params.desk_recall if params.desk_recall is not None else nan
    gg = params.desk_false_alarm if params.desk_false_alarm is not None else nan
    w(f"PARAMS {h(params.recall)} {h(params.class_accuracy)} {h(params.false_alarm)} {h(d)} {h(gg)} "
      f"{h(params.p_target_on_table)} {int(params.grid_size)} {int(params.table_context_enabled)}\n")

    rt_diff = []
    for i, fr in enumerate(frames):
        pose = fr["pose"]
        x, y, yaw = float(pose["x"]), float(pose["y"]), float(pose["yaw"])
        t = float(fr["t"])
        boxes = [
            {"class_name": e["class_name"], "class_id": int(e["class_id"]),
             "confidence": float(e["confidence"]), "bbox_xyxy": [float(v) for v in e["bbox_xyxy"]]}
            for e in fr.get("detections", []) or [] if e.get("bbox_xyxy") is not None
        ]
        w(f"FRAME {i} {h(t)} {h(x)} {h(y)} {h(yaw)} {len(boxes)}\n")
        for b in boxes:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            w(f"BOX {b['class_id']} {h(b['confidence'])} {h(x1)} {h(y1)} {h(x2)} {h(y2)}\n")

        # ---- controller geometry (crazyflie_AIF.py main loop, processed frame) ----
        floor_cov = env_floor.compute_cell_coverages(x, y, yaw)
        table_cov = env_table.compute_cell_coverages(x, y, yaw)
        stub.events = boxes
        frame = project_perception_frame(
            perception_manager=stub, config=proj_cfg,
            step_count=int(fr.get("step", i)), time_now_s=t,
            camera_capture={"image": None, "width": img_w, "height": img_h, "time": t},
            drone_x=x, drone_y=y, drone_yaw=yaw, env_mgr_floor=env_floor, env_mgr_table=env_table,
            floor_visual_coverages=floor_cov, persistent_table_cells=persistent_table_cells,
            persistent_table_confidence=persistent_table_conf)
        effective = effective_table_projection_cells(
            persistent_table_cells_cxcy=persistent_table_cells,
            transient_table_shadow_cells_cxcy=frame.transient_table_shadow_cells,
            enable_table_shadow_projection=True)
        merged = merge_visual_coverages_by_surface(
            floor_coverages_rc=floor_cov, table_coverages_rc=table_cov, table_projection_cells_cxcy=effective)
        visible = GridEnvManager.coverage_to_visible_mask(merged, threshold=coverage_threshold)
        current_cell = grid_spec.rowcol_from_xy(x, y)
        current_cxcy = None if current_cell is None else (int(current_cell[1]), int(current_cell[0]))

        for rc in sorted(set(floor_cov) | set(table_cov), key=lambda rc: (rc[1], rc[0])):
            r, c = rc
            w(f"COV {c} {r} {h(floor_cov.get(rc, 0.0))} {h(table_cov.get(rc, 0.0))} "
              f"{h(merged.get(rc, 0.0))} {int(bool(visible.get(rc, False)))}\n")

        all_events = [detection_event_rc_to_cxcy(e) for e in frame.all_events_rc]
        target_events = [detection_event_rc_to_cxcy(e) for e in frame.target_events_rc]
        for e in target_events:
            surface = 1 if e.get("surface_assumption") == "table" else 0
            w(f"TEVT {int(e['class_id'])} {h(e['confidence'])} {int(e['cell'][0])} {int(e['cell'][1])} {surface}\n")
        desk_cov = set()
        for e in all_events:
            if e.get("class_name") in DESK_NAMES:
                desk_cov.update((int(c[0]), int(c[1])) for c in e.get("covered_cells", []) or [])
        w(_cells_line("DESKCOV", desk_cov) + "\n")
        w(_cells_line("SHADOW", frame.transient_table_shadow_cells) + "\n")
        w(_cells_line("TMAP", persistent_table_cells) + "\n")
        for claim in claims.update(target_events, perception_processed=True):
            w(f"CLAIM {claim.cx} {claim.cy} {claim.class_id}\n")

        # ---- policy (belief update through the real compute_action path) ----
        evidence = PolicyPerceptionEvidence.for_step(
            perception_processed=True,
            visible_mask=mapping_rc_to_cxcy(visible),
            det_confidence=mapping_rc_to_cxcy(frame.target_confidence_rc),
            det_confidence_multiclass=mapping_rc_to_cxcy(frame.target_confidence_rc),
            det_metadata_multiclass=mapping_rc_to_cxcy(frame.target_metadata_rc),
            detections_all=all_events, target_detections=target_events,
            table_confidence=mapping_rc_to_cxcy(frame.table_confidence_rc),
            table_visible_mask=mapping_rc_to_cxcy(visible),
            table_coverages=mapping_rc_to_cxcy(merged))
        obs = build_policy_observation(
            sensor_state={}, step=int(fr.get("step", i)), pose={"x": x, "y": y, "yaw": yaw},
            grid_spec=grid_spec, detector_detections=[],
            current_cell=current_cxcy, visual_coverages=mapping_rc_to_cxcy(merged),
            perception_evidence=evidence, coverage_threshold=coverage_threshold,
            persistent_table_confidence=persistent_table_conf, perception_processed=True,
            surface_height_map={"table_cells": sorted([list(c) for c in persistent_table_cells])},
            persistent_table_cells=persistent_table_cells, effective_table_projection_cells=effective,
            claimed_found_cells=set(), task_state=TaskStateSnapshot(),
            claim_confidence_threshold=CLAIM_CONFIDENCE_THRESHOLD, snapshots_enabled=False,
            snapshots_dir=None, snapshots_every_n=0, snapshots_on_stuck=False, stuck_replan_limit=0,
            floor_fov_spec=floor_fov, table_fov_spec=table_fov, floor_height_m=FLOOR_HEIGHT_M,
            table_height_m=table_height_m,
            floor_plane_coverages=mapping_rc_to_cxcy(floor_cov),
            table_plane_coverages=mapping_rc_to_cxcy(table_cov))
        obs["grid_spec"]["rows"], obs["grid_spec"]["cols"] = rows, cols
        obs["fov_spec_by_surface"] = {"floor": dict(fov["floor"]), "table": dict(fov["table"])}
        obs["fov_spec"] = dict(fov["floor"])
        obs["time"] = t
        policy.compute_action(obs)

        rm, rs = policy.rho_estimate()
        tm, ts = policy.theta_estimate()
        w(f"RT {h(rm)} {h(rs)} {h(tm)} {h(ts)}\n")
        # The controller logs the policy diagnostics before this frame's update,
        # so the logged value of the next frame is the one to compare with.
        nxt = (frames[i + 1].get("policy") or {}) if i + 1 < len(frames) else {}
        if "rho_mean" in nxt:
            rt_diff.append((abs(rm - nxt["rho_mean"]), abs(tm - nxt["theta_mean"])))
        q = policy.q_joint
        for cy in range(rows):
            for cx in range(cols):
                w(f"Q {cx} {cy} " + " ".join(h(v) for v in q[cy, cx, :]) + "\n")

        # How many times the flight replanned between this frame and the next one
        # (the controller logs the replan counter before each frame's update).
        if i + 1 < len(frames):
            r0 = (fr.get("policy") or {}).get("replans")
            r1 = (frames[i + 1].get("policy") or {}).get("replans")
            n_replans = int(r1) - int(r0) if r0 is not None and r1 is not None else -1
        else:
            n_replans = 0
        w(f"REPLANS {n_replans}\n")

        # ---- planner from the current pose (side-effect free) ----
        if current_cxcy is not None:
            table_cells = policy._table_cells_from_grid(obs["grid"])
            (best_cell, best_yaw), scores = policy._pick_next_action(
                current_cell=current_cxcy, table_cells=table_cells, return_scores=True,
                pose={"x": x, "y": y, "yaw": yaw})
            yaw_idx = min(range(len(policy.yaw_options)), key=lambda k: abs(policy.yaw_options[k] - best_yaw))
            k_best = (best_cell[1] * cols + best_cell[0]) * len(policy.yaw_options) + yaw_idx
            sb = scores[k_best]
            w(f"PLAN {current_cxcy[0]} {current_cxcy[1]} {best_cell[0]} {best_cell[1]} {yaw_idx} "
              f"{h(sb['rate'])} {h(sb['T'])}\n")
            if args.cand_every > 0 and i % args.cand_every == 0:
                for k, s in enumerate(scores):
                    w(f"CAND {k} {h(s['IG'])} {h(s['T'])}\n")
    w("END\n")
    out.close()

    print(f"{args.output}: {len(frames)} frames, {environment}/{detector}, grid {params.grid_size}")
    if rt_diff:
        print("replay vs logged rho/theta mean: max |d rho| %.4f, max |d theta| %.4f, final %.4f / %.4f" % (
            max(a for a, _ in rt_diff), max(b for _, b in rt_diff), rt_diff[-1][0], rt_diff[-1][1]))


if __name__ == "__main__":
    main()
