#!/usr/bin/env bash
# GAP8 trace headers for every flight of a campaign, for the gvsoc batch
# (embedded/zoi/gap8/run_batch.sh). For each run with a perception log:
#   1. reference trace through the Python code (export_trace.py);
#   2. C replay check: build/replay_test must PASS (beliefs, claims, every plan);
#   3. GAP8 header (make_gap8_trace.py) -> <out>/headers/seedNNN.h; an existing
#      header must be regenerated identical;
#   4. host build of gap8/main.c on that header, FLIGHT=1 and FLIGHT=0: "plans
#      equal to Python" must be N / N.
# The traces (several MB each) are deleted once the header is checked.
#
#   tools/c_port/make_gap8_batch.sh <campaign dir> [out dir]   # out dir default: embedded/zoi/gap8/batch
#   JOBS=8 tools/c_port/make_gap8_batch.sh ...                 # parallel jobs (default 8)
#
# Resumable: a seed whose <out>/prep/seedNNN.status starts with OK is skipped.
# Also in prep/: seedNNN.replay (the replay summary) and seedNNN.host_fl1.txt (host
# output of FLIGHT=1: per-frame exact IG counts, read by summarize_gap8_batch.py --files).
# Result: <out>/manifest.tsv (seed, status, frames, plans in flight, plans every frame).

set -u
cd "$(dirname "$0")/../.." || exit 1

CAMP=${1:?usage: tools/c_port/make_gap8_batch.sh <campaign dir> [out dir]}
OUT=${2:-embedded/zoi/gap8/batch}
JOBS=${JOBS:-8}
PY=${PYTHON:-.venv/bin/python}
CC=${CC:-gcc}
EMB=embedded/zoi
[ -d "$CAMP/runs" ] || { echo "no runs/ in $CAMP"; exit 1; }
mkdir -p "$OUT/headers" "$OUT/prep"
make -s -C "$EMB" || { echo "host build of the replay test failed"; exit 1; }

prep_one() {
    local run=$1 seed tag st tmp fl line pre checks=""
    seed=$(basename "$run" | sed -n 's/.*_seed\([0-9][0-9]*\)_.*/\1/p')
    [ -n "$seed" ] || return 0
    tag=$(printf "seed%03d" "$seed")
    pre="$OUT/prep/$tag"
    st="$pre.status"
    if [ -f "$st" ] && grep -q "^OK" "$st"; then return 0; fi
    if [ ! -f "$run/perception_log.jsonl" ]; then echo "SKIP no perception log" > "$st"; return 0; fi
    tmp=$(mktemp -d)
    fail() { echo "FAIL $1" > "$st"; cp "$tmp"/*.log "$OUT/prep/" 2>/dev/null; rm -rf "$tmp"; echo "$tag: FAIL $1"; }
    "$PY" tools/c_port/export_trace.py "$run" -o "$tmp/trace.txt" > "$tmp/${tag}_export.log" 2>&1 || { fail export; return 0; }
    "$EMB/build/replay_test" "$tmp/trace.txt" > "$tmp/${tag}_replay.log" 2>&1 || { fail "C replay"; return 0; }
    grep -h "later plans" "$tmp/${tag}_replay.log" | sed 's/^ *//' > "$pre.replay"
    "$PY" tools/c_port/make_gap8_trace.py "$tmp/trace.txt" -o "$tmp/trace_data.h" > /dev/null 2>&1 || { fail header; return 0; }
    if [ -f "$OUT/headers/$tag.h" ]; then
        cmp -s "$tmp/trace_data.h" "$OUT/headers/$tag.h" || { fail "header differs from headers/$tag.h"; return 0; }
    else
        cp "$tmp/trace_data.h" "$OUT/headers/$tag.h"
    fi
    cp "$EMB/gap8/main.c" "$tmp/main.c"
    for fl in 1 0; do
        "$CC" -std=gnu99 -O2 -ffp-contract=off -I"$EMB/host" -I"$EMB/include" -DNB_CORES=8 -DFLIGHT=$fl \
            -o "$tmp/host" "$tmp/main.c" "$EMB"/src/*.c -lm > "$tmp/${tag}_cc$fl.log" 2>&1 || { fail "host build"; return 0; }
        "$tmp/host" > "$tmp/host.txt"
        [ $fl = 1 ] && cp "$tmp/host.txt" "$pre.host_fl1.txt"
        line=$(grep "plans equal to Python" "$tmp/host.txt")
        echo "$line" | grep -qE 'plans equal to Python: ([0-9]+) / \1[[:space:]]*$' ||
            { fail "host plans FLIGHT=$fl: $line"; return 0; }
        checks="$checks	$(echo "$line" | sed 's/.*: //')"
    done
    local frames
    frames=$(sed -n 's/^#define TRACE_N_FRAMES \([0-9]*\).*/\1/p' "$OUT/headers/$tag.h")
    printf "OK\t%s%s\n" "$frames" "$checks" > "$st"
    rm -rf "$tmp"
    echo "$tag: OK, $frames frames"
}
export -f prep_one
export OUT PY CC EMB

find "$CAMP/runs" -mindepth 1 -maxdepth 1 -type d | sort | xargs -P "$JOBS" -I{} bash -c 'prep_one "$1"' _ {}

MAN="$OUT/manifest.tsv"
{
    printf "seed\tstatus\tframes\tplans_flight\tplans_every_frame\n"
    for st in "$OUT"/prep/seed[0-9][0-9][0-9].status; do
        printf "%s\t%s\n" "$(basename "$st" .status)" "$(cat "$st")"
    done
} > "$MAN"
n_ok=$(grep -c "	OK	" "$MAN")
n_all=$(($(wc -l < "$MAN") - 1))
echo "headers ready: $n_ok / $n_all (manifest: $MAN)"
grep -v "	OK	" "$MAN" | tail -n +2
