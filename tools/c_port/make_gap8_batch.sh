#!/usr/bin/env bash
# GAP8 trace headers for every flight of a campaign, for the gvsoc batch
# (embedded/zoi/gap8/run_batch.sh). For each run with a perception log:
#   1. reference trace through the Python code (export_trace.py);
#   2. C replay check: build/replay_test_prune2 must PASS (beliefs, claims, every plan);
#      with IG_CACHE=1 build/replay_test_prune2_igc too, with the same decisions bit
#      for bit on every plan (the flight's later plans of a frame included);
#   3. GAP8 header (make_gap8_trace.py) -> <out>/headers/seedNNN.h; the headers do not
#      depend on the configuration, so an existing one must be regenerated identical;
#   4. host build of gap8/main.c on that header, FLIGHT=1 and FLIGHT=0, PRUNING=2,
#      ZOI_IG_CACHE=$IG_CACHE: "plans equal to Python" must be N / N. With
#      IG_CACHE=1 the same two builds without the cache must give the same
#      "plan decisions hash".
# The traces (several MB each) are deleted once the header is checked.
#
#   tools/c_port/make_gap8_batch.sh <campaign dir> [out dir]   # out dir default: embedded/zoi/gap8/batch
#   IG_CACHE=1 tools/c_port/make_gap8_batch.sh ...             # checks with the IG cache (default 0)
#   JOBS=8 tools/c_port/make_gap8_batch.sh ...                 # parallel jobs (default 8)
#
# Per configuration (C = IG_CACHE), resumable: a seed whose <out>/prep/seedNNN.igcacheC.status
# starts with OK is skipped. Also in prep/: seedNNN.igcacheC.replay (the replay summaries)
# and seedNNN.igcacheC.host_fl1_cX.txt (host output of FLIGHT=1 with cache X: per-frame
# exact IG counts, read by summarize_gap8_batch.py --files).
# Result: <out>/manifest_igcacheC.tsv (seed, status, frames, plans in flight, host checks).

set -u
cd "$(dirname "$0")/../.." || exit 1

CAMP=${1:?usage: tools/c_port/make_gap8_batch.sh <campaign dir> [out dir]}
OUT=${2:-embedded/zoi/gap8/batch}
JOBS=${JOBS:-8}
PY=${PYTHON:-.venv/bin/python}
CC=${CC:-gcc}
IG_CACHE=${IG_CACHE:-0}
case $IG_CACHE in 0|1) ;; *) echo "IG_CACHE must be 0 or 1"; exit 1 ;; esac
EMB=embedded/zoi
[ -d "$CAMP/runs" ] || { echo "no runs/ in $CAMP"; exit 1; }
mkdir -p "$OUT/headers" "$OUT/prep"
make -s -C "$EMB" || { echo "host build of the replay tests failed"; exit 1; }

prep_one() {
    local run=$1 seed tag st tmp fl c line pre
    seed=$(basename "$run" | sed -n 's/.*_seed\([0-9][0-9]*\)_.*/\1/p')
    [ -n "$seed" ] || return 0
    tag=$(printf "seed%03d" "$seed")
    pre="$OUT/prep/$tag.igcache$IG_CACHE"
    st="$pre.status"
    if [ -f "$st" ] && grep -q "^OK" "$st"; then return 0; fi
    if [ ! -f "$run/perception_log.jsonl" ]; then echo "SKIP no perception log" > "$st"; return 0; fi
    tmp=$(mktemp -d)
    fail() { echo "FAIL $1" > "$st"; cp "$tmp"/*.log "$OUT/prep/" 2>/dev/null; rm -rf "$tmp"; echo "$tag: FAIL $1"; }
    "$PY" tools/c_port/export_trace.py "$run" -o "$tmp/trace.txt" > "$tmp/${tag}_export.log" 2>&1 || { fail export; return 0; }
    "$EMB/build/replay_test_prune2" "$tmp/trace.txt" -d "$tmp/dec0.txt" > "$tmp/${tag}_replay.log" 2>&1 || { fail "C replay"; return 0; }
    grep -h "later plans" "$tmp/${tag}_replay.log" | sed 's/^ */prune2: /' > "$pre.replay"
    if [ "$IG_CACHE" = 1 ]; then
        "$EMB/build/replay_test_prune2_igc" "$tmp/trace.txt" -d "$tmp/dec1.txt" > "$tmp/${tag}_replay_igc.log" 2>&1 ||
            { fail "C replay with IG cache"; return 0; }
        cmp -s "$tmp/dec0.txt" "$tmp/dec1.txt" || { fail "C replay: IG cache changes the decisions"; return 0; }
        grep -h "later plans" "$tmp/${tag}_replay_igc.log" | sed 's/^ */prune2_igc: /' >> "$pre.replay"
    fi
    "$PY" tools/c_port/make_gap8_trace.py "$tmp/trace.txt" -o "$tmp/trace_data.h" > /dev/null 2>&1 || { fail header; return 0; }
    if [ -f "$OUT/headers/$tag.h" ]; then
        cmp -s "$tmp/trace_data.h" "$OUT/headers/$tag.h" || { fail "header differs from headers/$tag.h"; return 0; }
    else
        cp "$tmp/trace_data.h" "$OUT/headers/$tag.h"
    fi
    cp "$EMB/gap8/main.c" "$tmp/main.c"
    local checks="" caches="$IG_CACHE"
    [ "$IG_CACHE" = 1 ] && caches="1 0"
    for fl in 1 0; do
        local hash1=""
        for c in $caches; do
            "$CC" -std=gnu99 -O2 -ffp-contract=off -I"$EMB/host" -I"$EMB/include" \
                -DNB_CORES=8 -DZOI_REFERENCE_KERNELS=0 -DZOI_PLANNER_PRUNING=2 -DFLIGHT=$fl -DZOI_IG_CACHE=$c \
                -o "$tmp/host" "$tmp/main.c" "$EMB"/src/*.c -lm > "$tmp/${tag}_cc$fl$c.log" 2>&1 || { fail "host build"; return 0; }
            "$tmp/host" > "$tmp/host.txt"
            [ $fl = 1 ] && cp "$tmp/host.txt" "$pre.host_fl1_c$c.txt"
            line=$(grep "plans equal to Python" "$tmp/host.txt")
            echo "$line" | grep -qE 'plans equal to Python: ([0-9]+) / \1[[:space:]]*$' ||
                { fail "host plans FLIGHT=$fl IG_CACHE=$c: $line"; return 0; }
            local hash
            hash=$(sed -n 's/^plan decisions hash: \(0x[0-9a-f]*\).*/\1/p' "$tmp/host.txt")
            if [ "$c" = "$IG_CACHE" ]; then
                checks="$checks	$(echo "$line" | sed 's/.*: //')"
                hash1=$hash
            elif [ "$hash" != "$hash1" ]; then
                fail "host FLIGHT=$fl: plan decisions hash $hash1 with IG cache, $hash without"; return 0
            fi
        done
    done
    local frames
    frames=$(sed -n 's/^#define TRACE_N_FRAMES \([0-9]*\).*/\1/p' "$OUT/headers/$tag.h")
    printf "OK\t%s%s\n" "$frames" "$checks" > "$st"
    rm -rf "$tmp"
    echo "$tag: OK, $frames frames"
}
export -f prep_one
export OUT PY CC EMB IG_CACHE

find "$CAMP/runs" -mindepth 1 -maxdepth 1 -type d | sort | xargs -P "$JOBS" -I{} bash -c 'prep_one "$1"' _ {}

MAN="$OUT/manifest_igcache$IG_CACHE.tsv"
{
    printf "seed\tstatus\tframes\tplans_flight\tplans_every_frame\n"
    for st in "$OUT"/prep/seed*.igcache$IG_CACHE.status; do
        printf "%s\t%s\n" "$(basename "$st" .igcache$IG_CACHE.status)" "$(cat "$st")"
    done
} > "$MAN"
n_ok=$(grep -c "	OK	" "$MAN")
n_all=$(($(wc -l < "$MAN") - 1))
echo "headers ready (IG_CACHE=$IG_CACHE): $n_ok / $n_all (manifest: $MAN)"
grep -v "	OK	" "$MAN" | tail -n +2
