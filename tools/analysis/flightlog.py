"""Shared helpers for reading perception_log.jsonl flight logs (AIF_PERCEPTION_LOG=1).

A campaign directory is webots/controllers/crazyflie_AIF/logs/<eval_...>/. Each run
folder under runs/ holds perception_log.jsonl: one header record (grid, camera
footprints, spawned scene) and one record per processed detector frame (pose,
visible cells with coverages, table map, detections). Cells are (cx, cy).
"""
import glob
import json
import math
import os

TARGETS = {"target_bottle", "target_can"}


def load_runs(campaign_dir, autonomous_only=True):
    """Yield (run_dir, header, frames, run_report) for every completed run."""
    for run_dir in sorted(glob.glob(os.path.join(campaign_dir, "runs", "*"))):
        log_path = os.path.join(run_dir, "perception_log.jsonl")
        reports = glob.glob(os.path.join(run_dir, "run_report", "*.json"))
        if not os.path.exists(log_path) or not reports:
            continue  # aborted or still running
        with open(log_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        frames = [fr for fr in lines[1:] if fr["autonomous"] or not autonomous_only]
        with open(reports[0], encoding="utf-8") as f:
            report = json.load(f)
        yield run_dir, lines[0], frames, report


def gt_targets(header):
    return [o for o in header["spawned_objects"]["objects"] if o["class_name"] in TARGETS]


def object_plane_spec(header, obj):
    """Camera footprint on the plane where the object sits (table objects have z=0.72)."""
    return header["fov_spec"]["table" if obj["z"] > 0.3 else "floor"]


def to_image(obj_xy, pose, spec, img_w=320, img_h=240):
    """World point -> (u, v, un, vn) with the controller's linear pixel/footprint mapping."""
    dx, dy = obj_xy[0] - pose["x"], obj_xy[1] - pose["y"]
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    xb, yb = c * dx + s * dy, -s * dx + c * dy
    un = (spec["half_left"] - yb) / (spec["half_left"] + spec["half_right"])
    vn = (spec["forward"] - xb) / (spec["forward"] + spec["backward"])
    return un * img_w, vn * img_h, un, vn


def in_image(un, vn):
    return 0.0 <= un <= 1.0 and 0.0 <= vn <= 1.0


def box_centre(det):
    b = det["bbox_xyxy"]
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def matching_box(dets, u, v, radius_px=30.0):
    """Nearest target box whose centre is within radius_px of (u, v), or None."""
    best, best_d = None, radius_px
    for d in dets:
        if d["class_name"] not in TARGETS:
            continue
        cu, cv = box_centre(d)
        dist = math.hypot(cu - u, cv - v)
        if dist <= best_d:
            best, best_d = d, dist
    return best
