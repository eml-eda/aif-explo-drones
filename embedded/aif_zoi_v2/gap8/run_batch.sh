#!/usr/bin/env bash
# gvsoc on every flight header in batch/headers (built on the host by
# tools/v2_c/make_gap8_batch.sh), one flight after the other. 8 cores, pruning 2.
#
#   source /gap_sdk/configs/ai_deck.sh
#   cd embedded/aif_zoi_v2/gap8
#   IG_CACHE=1 ./run_batch.sh               # flight mode (plans as in the recorded flight), every seed
#   IG_CACHE=0 ./run_batch.sh               # the same without the IG cache (default 0)
#   MODES="flight full" ./run_batch.sh      # also a plan on every frame (worst case)
#   SEEDS="1 33 50" ./run_batch.sh          # only these seeds
#
# Output: batch/<mode>_igcache<0|1>/seedNNN.txt, one directory per configuration.
# Resumable within that directory: a seed whose output already reports
# "plans equal to Python: N / N" is skipped; a failed run is kept as seedNNN.txt.part.
# trace_data.h is swapped for each flight and restored at the end (also on Ctrl-C).
# Summary: python3 tools/v2_c/summarize_gap8_batch.py (from the repo root).

set -u
cd "$(dirname "$0")" || exit 1
[ -n "${RULES_DIR:-}" ] || { echo "GAP SDK not sourced: source /gap_sdk/configs/ai_deck.sh"; exit 1; }
BATCH=batch
MODES=${MODES:-flight}
IG_CACHE=${IG_CACHE:-0}
case $IG_CACHE in 0|1) ;; *) echo "IG_CACHE must be 0 or 1"; exit 1 ;; esac
ls "$BATCH"/headers/seed*.h > /dev/null 2>&1 || { echo "no headers in $BATCH/headers: run tools/v2_c/make_gap8_batch.sh on the host"; exit 1; }

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
        for s in $SEEDS; do [ "$tag" = "$(printf "seed%03d" "$s")" ] && keep=1; done
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
    dir="$BATCH/${mode}_igcache$IG_CACHE"
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
        make clean all run platform=gvsoc PRUNING=2 FLIGHT=$FL IG_CACHE=$IG_CACHE > "$out.part" 2>&1
        dt=$(($(date +%s) - t0))
        if done_ok "$out.part"; then mv "$out.part" "$out"; status="ok"; else status="FAILED (see $out.part)"; fi
        ran=$((ran + 1)); spent=$((spent + dt))
        left=$(( (total - i) * spent / ran / 60 ))
        echo "[$(date "+%F %T")] ${mode}_igcache$IG_CACHE $tag ($i/$total): $status, $((dt / 60)) min; about $left min left in this mode"
    done
done
echo "[$(date "+%F %T")] batch done"
