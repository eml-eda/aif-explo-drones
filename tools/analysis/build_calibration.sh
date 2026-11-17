#!/usr/bin/env bash
# Build the ZOI detector calibration for one environment from a seed-101 collection:
# prepare the set (everything in val), MobileNet and YOLO nano predictions, counts.
#
#   tools/analysis/build_calibration.sh <easy|medium|hard> <raw collection dir> [prepared dir]
#
# prepared dir defaults to detection_model_training/datasets/<env>_calib_seed101; if it exists it is
# reused (its images must be the collection's: checked on every frame key). Offline, CPU only.
# Writes <prepared>/eval/calibration_<detector>.txt with the counts for zoi_params.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV=$1; COLL=$(realpath "$2"); PREP=$(realpath -m "${3:-detection_model_training/datasets/${ENV}_calib_seed101}")
PY=.venv/bin/python
CK=$(realpath detection_model_training/checkpoints/$ENV)  # absolute: the eval scripts resolve relative paths elsewhere

if [ ! -d "$PREP" ]; then
    echo "== prepare $PREP"
    $PY detection_model_training/tools/prepare_dataset.py "$COLL" "$PREP" --environment "$ENV"
    # Calibration only: every image in val.
    for kind in images labels; do
        for split in train test; do
            find "$PREP/$kind/$split" -maxdepth 1 -type f -exec mv -t "$PREP/$kind/val" {} +
        done
    done
    sed -i "s|^path: .*|path: $PREP|" "$PREP/target_objects.yaml"
fi
echo "== check: prepared images belong to the collection"
$PY - "$COLL" "$PREP" <<'PYEOF'
import json, os, sys
coll, prep = sys.argv[1], sys.argv[2]
key = lambda e: "_".join(str(e).split("_")[:2])
raw = {key(json.loads(l)["entry_id"]) for l in open(os.path.join(coll, "manifest.jsonl")) if l.strip()}
p = [json.loads(l) for l in open(os.path.join(prep, "manifest.jsonl")) if l.strip()]
missing = [e["entry_id"] for e in p if key(e["entry_id"]) not in raw]
n_val = len(os.listdir(os.path.join(prep, "images", "val")))
print(f"   prepared entries {len(p)}, in val {n_val}, without a collection frame {len(missing)}")
# Same pixels: the collection gives the poses and desks, the prepared set the images.
raw_img = {key(json.loads(l)["entry_id"]): json.loads(l)["image_filename"]
           for l in open(os.path.join(coll, "manifest.jsonl")) if l.strip()}
diff = 0
for e in p:
    if key(e["entry_id"]) not in raw_img:
        continue
    a = open(os.path.join(prep, "images", "val", os.path.basename(e["prepared_image_filename"])), "rb").read()
    b = open(os.path.join(coll, "images", raw_img[key(e["entry_id"])]), "rb").read()
    diff += a != b
print(f"   images differing from the collection: {diff}")
sys.exit(1 if missing or diff else 0)
PYEOF

MBN_PRED=$PREP/eval/mobilenet_ssd/val_eval/predictions.json
if [ ! -f "$MBN_PRED" ]; then
    echo "== MobileNet predictions"; mkdir -p "$PREP/eval"
    $PY detection_model_training/eval_mobilenetv2_ssd_sigmoid.py --env "$ENV" --data "$PREP/target_objects.yaml" \
        --weights "$CK/mobilenet_ssd.pt" --split val --eval-project "$PREP/eval/mobilenet_ssd" --device cpu --no-plots \
        > "$PREP/eval/mobilenet_ssd_eval_log.txt" 2>&1
fi
YOLO_PRED=$PREP/eval/yolo_nano/deployed/predictions.json
if [ ! -f "$YOLO_PRED" ]; then
    echo "== YOLO nano predictions"
    $PY tools/analysis/yolo_calib_predictions.py --reference "$MBN_PRED" --weights "$CK/yolo_nano.pt" --out "$YOLO_PRED"
fi
for det in mobilenet_ssd yolo_nano; do
    pred=$MBN_PRED; [ "$det" = yolo_nano ] && pred=$YOLO_PRED
    out=$PREP/eval/calibration_$det.txt
    $PY tools/analysis/calibrate_desk.py --collection "$COLL" --prepared "$PREP" --predictions "$pred" > "$out"
    echo "== $ENV / $det  ($out)"
    sed -n '/target detector/,/desk evidence/p' "$out" | sed '$d'
    grep -A2 "^desk evidence" "$out" || true
done
