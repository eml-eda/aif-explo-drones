#!/usr/bin/env bash
# gvsoc on every flight header in batch/headers (built on the host by
# tools/c_port/make_gap8_batch.sh), one flight after the other, on 8 cores.
#
#   source /gap_sdk/configs/ai_deck.sh
#   cd embedded/zoi/gap8
#   ./run_batch.sh                          # flight mode (plans as in the recorded flight), every seed
#   MODES="flight full" ./run_batch.sh      # also a plan on every frame (worst case)
#   SEEDS="1 33 50" ./run_batch.sh          # only these seeds
#   BATCH=batch/tiles_2x2 ROWS=26 COLS=22 ./run_batch.sh   # a larger room (headers from
#                                           # tools/c_port/make_tiled_room_header.py, see ../README.md)
#
# Output: batch/<mode>/seedNNN.txt, one directory per mode.
# Resumable within that directory: a seed whose output already reports
# "plans equal to Python: N / N" is skipped; a failed run is kept as seedNNN.txt.part.
# trace_data.h is swapped for each flight and restored at the end (also on Ctrl-C).
# Summary: python3 tools/c_port/summarize_gap8_batch.py (from the repo root).

set -u
cd "$(dirname "$0")" || exit 1
[ -n "${RULES_DIR:-}" ] || { echo "GAP SDK not sourced: source /gap_sdk/configs/ai_deck.sh"; exit 1; }
BATCH=${BATCH:-batch}
ROWS=${ROWS:-13}
COLS=${COLS:-11}
MODES=${MODES:-flight}
ls "$BATCH"/headers/seed*.h > /dev/null 2>&1 || { echo "no headers in $BATCH/headers: run tools/c_port/make_gap8_batch.sh on the host"; exit 1; }

# Keep the original header; if a previous batch was killed, the backup is the original.
[ -f "$BATCH/trace_data.h.orig" ] || cp -p trace_data.h "$BATCH/trace_data.h.orig"
restore() { cp -p "$BATCH/trace_data.h.orig" trace_data.h && rm -f "$BATCH/trace_data.h.orig"; echo "trace_data.h restored"; }
trap restore EXIT
trap 'exit 130' INT TERM

done_ok() { grep -qE 'plans equal to Python: ([0-9]+) / \1[[:space:]]*$' "$1" 2>/dev/null; }

headers=()
for h in "$BATCH"/headers/seed*.h; do
    tag=$(basename "$h" .h)
    if [ -n "${SEEDS:-}" ]; then
        keep=0
        for s in $SEEDS; do case "$tag" in "$(printf "seed%03d" "$s")"|"$(printf "seed%03d" "$s")"_*) keep=1 ;; esac; done
        [ $keep = 1 ] || continue
    fi
    headers+=("$h")
done

for mode in $MODES; do
    case $mode in
        flight) FL=1 ;;
        full) FL=0 ;;
        *) echo "unknown mode $mode (flight, full)"; exit 1 ;;
    esac
    dir="$BATCH/$mode"
    mkdir -p "$dir"
    total=${#headers[@]}
    i=0; ran=0; spent=0
    for h in "${headers[@]}"; do
        i=$((i + 1))
        tag=$(basename "$h" .h)
        out="$dir/$tag.txt"
        done_ok "$out" && continue
        cp "$h" trace_data.h
        t0=$(date +%s)
        make clean all run platform=gvsoc FLIGHT=$FL ROWS=$ROWS COLS=$COLS > "$out.part" 2>&1
        dt=$(($(date +%s) - t0))
        if done_ok "$out.part"; then mv "$out.part" "$out"; status="ok"; else status="FAILED (see $out.part)"; fi
        ran=$((ran + 1)); spent=$((spent + dt))
        left=$(( (total - i) * spent / ran / 60 ))
        echo "[$(date "+%F %T")] $mode $tag ($i/$total): $status, $((dt / 60)) min; about $left min left in this mode"
    done
done
echo "[$(date "+%F %T")] batch done"
