"""YOLO predictions on the calibration set, in the format calibrate_desk.py reads.

Runs an environment's YOLO checkpoint with the in-flight settings (confidence
0.45, NMS IoU 0.40, 320 px, as vision/detector_yolo_context.py) on the same
images and labels as the MobileNet predictions file, so both detectors are
calibrated on one image set.

Usage:
  python tools/analysis/yolo_calib_predictions.py \
      --reference <MobileNet predictions.json (image paths and labels)> \
      --weights detection_model_training/checkpoints/hard/yolo_nano.pt \
      --out <predictions.json>
"""
import argparse
import json
import os

from PIL import Image
from ultralytics import YOLO

SCHEMAS = (["target_bottle", "target_can", "desk"],  # medium, hard
           ["target_bottle", "target_can"])          # easy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.40)
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()

    with open(args.reference, encoding="utf-8") as f:
        reference = json.load(f)
    model = YOLO(args.weights)
    names = [str(model.names[k]) for k in sorted(model.names)]
    assert names in SCHEMAS, names

    out = []
    for p in reference:
        img = Image.open(p["image_path"]).convert("RGB")
        res = model.predict(source=img, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            device="cpu", verbose=False)[0]
        w, h = img.size
        dets = []
        for box in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
            y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append({"class_name": names[int(box.cls[0])], "score": float(box.conf[0]),
                         "bbox_xyxy": [x1, y1, x2, y2]})
        out.append({"image_path": p["image_path"], "targets": p["targets"], "detections": dets})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{len(out)} images -> {args.out}")


if __name__ == "__main__":
    main()
