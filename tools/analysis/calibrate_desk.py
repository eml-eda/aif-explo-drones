"""Offline detector calibration for the ZOI policy from one calibration image set.

Usage:
  python tools/analysis/calibrate_desk.py \
      --collection <raw collection dir>  (manifest.jsonl with drone poses, metadata.json with objects)
      --prepared   <prepared dataset dir> (manifest.jsonl: entry_id -> prepared image)
      --predictions <MobileNet predictions.json on that prepared set>

Counts, all with the deployment settings (confidence 0.45, NMS IoU 0.40):

  r, a      target recall and class accuracy. A target box matches a label when its centre
            is inside the label box (in flight a box is assigned to the cell of its centre;
            overlap never matters); one label per box, the best-overlapping one. On the hard
            and medium sets this gives the same counts as IoU >= 0.5; on the easy set the
            labels are drawn differently and IoU >= 0.5 rejects most correct boxes.
  f         false alarm per empty cell in view: target boxes matching no label over
            the empty cells passing the update gate (floor coverage >= 0.05), counted
            image by image from the drone pose
  r_table   a table cell in view is covered by a desk box
  f_table   a floor cell in view is covered by a desk box

A table cell is a cell whose centre lies on a desk top. "Covered" uses the
controller's own projection (sample_bbox_projected_cells_rc on the table plane).
A desk is an extended object: if any part of a desk cell is in view at table
height, the desk is in the image. So the desk evidence is binary, counted on the
cells whose table-plane coverage passes the same update gate (w_T >= 0.05):

  r_table = P(covered | table cell, w_T >= gate)
  f_table = P(covered | floor cell, w_T >= gate)

The covered rate binned by w_T is printed as a check of this form (a target, a
small object, is instead in view with probability w). Values use the rule of
succession (k + 1) / (n + 2). No flight data is used.
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "webots", "controllers", "crazyflie_AIF"))

from env.env_manager import GridEnvManager, GridSpec  # noqa: E402
from env.surface_projection import (  # noqa: E402
    make_fov_spec, sample_bbox_projected_cells_rc, scaled_fov_spec_for_surface)

# Same values as crazyflie_AIF.py.
GRID = GridSpec(min_x=-2.75, max_x=2.75, min_y=-3.25, max_y=3.25, cell_size=0.5, rows=13, cols=11)
FLYING_ALTITUDE_M = 1.5
TABLE_HEIGHT_M = 0.72
FOV_WIDTH_M, FOV_FORWARD_M, FOV_BACKWARD_M = 1.394, 0.553, 0.493
COVERAGE_UPDATE_THRESHOLD = 0.05
CONFIDENCE = 0.45
NMS_IOU = 0.40
IMG_W, IMG_H = 320, 240
TARGETS = {"target_bottle", "target_can"}
W_BINS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0001)


def succession(k, n):
    return (k + 1.0) / (n + 2.0)


def beta_interval(k, n):
    try:
        from scipy.stats import beta
    except ImportError:
        return (float("nan"), float("nan"))
    d = beta(k + 1.0, max(n - k, 0.0) + 1.0)
    return (float(d.ppf(0.025)), float(d.ppf(0.975)))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def deployed_boxes(detections):
    """Confidence threshold, then per-class greedy NMS, as in flight."""
    keep = []
    for d in sorted((d for d in detections if d["score"] >= CONFIDENCE), key=lambda d: -d["score"]):
        if all(k["class_name"] != d["class_name"] or iou(k["bbox_xyxy"], d["bbox_xyxy"]) <= NMS_IOU for k in keep):
            keep.append(d)
    return keep


def in_rect(x, y, rect):
    c, s = math.cos(-rect["yaw"]), math.sin(-rect["yaw"])
    dx, dy = x - rect["x"], y - rect["y"]
    bx, by = c * dx - s * dy, s * dx + c * dy
    return abs(bx) <= rect["length"] / 2.0 and abs(by) <= rect["width"] / 2.0


def cell_center(rc):
    r, c = rc
    return GRID.min_x + (c + 0.5) * GRID.cell_size, GRID.min_y + (r + 0.5) * GRID.cell_size


def frame_key(entry_id):
    """seed0101_frame0000600_<collection timestamp> -> seed0101_frame0000600.

    Matches frames across re-collections of the same seed (identical images,
    different collection timestamps).
    """
    return "_".join(str(entry_id).split("_")[:2])


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()

    raw = {frame_key(e["entry_id"]): e for e in read_jsonl(os.path.join(args.collection, "manifest.jsonl"))}
    prep = {os.path.basename(e["prepared_image_filename"]): frame_key(e["entry_id"])
            for e in read_jsonl(os.path.join(args.prepared, "manifest.jsonl"))}
    with open(os.path.join(args.collection, "metadata.json"), encoding="utf-8") as f:
        runs = json.load(f)["runs"]
    objects_by_seed = {int(run["seed"]): run.get("objects", []) for run in runs.values()}
    with open(args.predictions, encoding="utf-8") as f:
        predictions = json.load(f)

    floor_env = GridEnvManager(GRID, make_fov_spec(
        width_m=FOV_WIDTH_M, forward_m=FOV_FORWARD_M, backward_m=FOV_BACKWARD_M))
    table_env = GridEnvManager(GRID, scaled_fov_spec_for_surface(
        TABLE_HEIGHT_M, camera_height_m=FLYING_ALTITUDE_M, reference_height_m=FLYING_ALTITUDE_M,
        base_width_m=FOV_WIDTH_M, base_forward_m=FOV_FORWARD_M, base_backward_m=FOV_BACKWARD_M))

    n_targets = n_detected = n_class_ok = n_false = 0
    empty_cell_frames = 0
    desk = {"table": [0, 0], "floor": [0, 0]}  # covered, cells with w_T >= gate
    bins = {s: [[0, 0] for _ in W_BINS[:-1]] for s in desk}  # [covered, cells] per w_T bin
    missing_pose = 0
    for p in predictions:
        entry = raw.get(prep.get(os.path.basename(p["image_path"]), ""), None)
        if entry is None:
            missing_pose += 1
            continue
        pose = entry["drone_pose"]
        objects = objects_by_seed.get(int(entry["seed"]), [])
        desks = [o["tabletop_rect"] for o in objects if o.get("class_name") == "desk"]
        target_cells = {GRID.rowcol_from_xy(o["x"], o["y"]) for o in objects if o.get("class_name") in TARGETS}
        boxes = deployed_boxes(p["detections"])

        # Target detector: pixel matching against the labels.
        gts = [g for g in p["targets"] if g["class_name"] in TARGETS]
        n_targets += len(gts)
        used = set()
        for d in (b for b in boxes if b["class_name"] in TARGETS):
            cx = 0.5 * (d["bbox_xyxy"][0] + d["bbox_xyxy"][2])
            cy = 0.5 * (d["bbox_xyxy"][1] + d["bbox_xyxy"][3])
            best, best_i = -1.0, None
            for i, g in enumerate(gts):
                x1, y1, x2, y2 = g["bbox_xyxy"]
                if i in used or not (x1 <= cx <= x2 and y1 <= cy <= y2):
                    continue
                if iou(d["bbox_xyxy"], g["bbox_xyxy"]) > best:
                    best, best_i = iou(d["bbox_xyxy"], g["bbox_xyxy"]), i
            if best_i is None:
                n_false += 1
            else:
                used.add(best_i)
                n_detected += 1
                n_class_ok += d["class_name"] == gts[best_i]["class_name"]

        # Cells in view, from the pose.
        floor_cov = floor_env.compute_cell_coverages(pose["x"], pose["y"], pose["yaw"])
        table_cov = table_env.compute_cell_coverages(pose["x"], pose["y"], pose["yaw"])
        visible = [rc for rc, v in floor_cov.items() if v >= COVERAGE_UPDATE_THRESHOLD]
        empty_cell_frames += sum(1 for rc in visible if rc not in target_cells)

        covered = set()
        for d in (b for b in boxes if b["class_name"] == "desk"):
            covered |= sample_bbox_projected_cells_rc(
                table_env, bbox_xyxy=d["bbox_xyxy"], img_w=IMG_W, img_h=IMG_H,
                drone_x=pose["x"], drone_y=pose["y"], drone_yaw=pose["yaw"])
        for rc in visible:
            w = float(table_cov.get(rc, 0.0))
            surface = "table" if any(in_rect(*cell_center(rc), r) for r in desks) else "floor"
            hit = rc in covered
            if w >= COVERAGE_UPDATE_THRESHOLD:
                desk[surface][0] += hit
                desk[surface][1] += 1
            for i, (lo, hi) in enumerate(zip(W_BINS[:-1], W_BINS[1:])):
                if lo <= w < hi:
                    bins[surface][i][0] += hit
                    bins[surface][i][1] += 1

    print(f"images {len(predictions)}, without pose {missing_pose}")
    print("\ntarget detector (k / n -> (k+1)/(n+2), 95% interval)")
    for name, k, n in (("r", n_detected, n_targets), ("a", n_class_ok, n_detected),
                       ("f", n_false, empty_cell_frames)):
        lo, hi = beta_interval(k, n)
        print(f"  {name:8s} {k:5d} / {n:8.1f} -> {succession(k, n):.6f}   [{lo:.6f}, {hi:.6f}]")
    print(f"  (empty cells in view per image: {empty_cell_frames / max(1, len(predictions) - missing_pose):.2f})")

    print(f"\ndesk evidence, cells with table-plane coverage >= {COVERAGE_UPDATE_THRESHOLD}")
    for name, surface in (("r_table", "table"), ("f_table", "floor")):
        k, n = desk[surface]
        lo, hi = beta_interval(k, n)
        print(f"  {name:8s} {k:5d} / {n:8.1f} -> {succession(k, n):.6f}   [{lo:.6f}, {hi:.6f}]")

    print("\ncovered rate by w_T bin (check: flat above the gate for an extended object)")
    print("  w_T bin      | table: covered/cells | floor: covered/cells")
    for i, (lo, hi) in enumerate(zip(W_BINS[:-1], W_BINS[1:])):
        t, fl = bins["table"][i], bins["floor"][i]
        rate = lambda b: f"{b[0]:4d}/{b[1]:5d} = {b[0] / b[1]:.3f}" if b[1] else f"{b[0]:4d}/{b[1]:5d}"
        print(f"  [{lo:.2f}, {min(hi, 1.0):.2f}) | {rate(t):>20s} | {rate(fl):>20s}")


if __name__ == "__main__":
    main()
