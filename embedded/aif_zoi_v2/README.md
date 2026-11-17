# aif_zoi_v2 in C (float32)

A C99 port of the `aif_zoi_v2` policy. It also ports the per-frame geometry the controller runs before the policy, so this is everything that runs between the detector output and the velocity command. It targets the GAP8 of the Crazyflie AI-deck and also builds on a PC. Every number is `float`, with no `double` anywhere; there is no `malloc`, and all memory lives in one `zoi_state_t` (55 kB).

| Python | C |
|---|---|
| `env/env_manager.py` (footprint coverage, bbox → cell) | `src/zoi_v2_geometry.c` |
| `vision/perception_projection.py`, `env/surface_projection.py`, `task/claim_manager.py` (desk map, target projection, shadow cells, visible mask, claims) | `src/zoi_v2_perception.c` |
| `policies/zoi_v2_model.py`, belief update and learned (ρ, θ) in `policies/aif_zoi_v2.py` | `src/zoi_v2_belief.c` |
| planner (IG / T) in `policies/aif_zoi_v2*.py`, `policies/planner_time.py` | `src/zoi_v2_planner.c` |
| `utils/navigator.py`, replan rule of `compute_action` | `src/zoi_v2.c` |
| calibration counts (`configuration/aif_zoi_v2_params.py`), controller constants (`crazyflie_AIF.py`) | `src/zoi_v2_config.c` |

The API is in `include/zoi_v2.h`; the sizes are in `include/zoi_v2_config.h`.

```c
zoi_config_t cfg;
zoi_config_default(&cfg, ZOI_ENV_HARD, ZOI_DET_MOBILENET);
static zoi_state_t st;
zoi_init(&st, &cfg);
/* every detector frame */  zoi_perception_frame(&st, &pose, boxes, n_boxes);
/* every control step   */  zoi_control_step(&st, &pose, time_s, &cmd);   /* cmd.forward, .sideways, .yaw_rate */
```

## Differences from the Python code

None of these changes the model; they exist because of float32 and the memory budget.

- **`log ell` per cell instead of `ell`.** In float32 a linear `ell` underflows to 0, and a zero state never recovers. Over the 30 test flights the lowest `log ell` reached was −509. In 15 of the 30 flights some cell went below −87.3, the float32 limit. Python's float64 limit is −745.
- **`log_ev` is re-centred** to max 0 after every frame. The weights are unchanged.
- **Beliefs are computed only when the planner reads them.** This is exact: in Python they are recomputed every frame, but only the planner uses them.
- **View patterns.** The cells seen from a candidate pose, with their floor- and table-plane coverages, are the same relative to the candidate's cell for every cell (checked to 1e-12 in Python). So they are computed once per yaw at init: 88 entries instead of a 11,080-entry per-candidate table (~110 kB). Each entry's likelihood is precomputed too.

## Two kernel variants

Selected with `ZOI_V2_REFERENCE_KERNELS`. Both give the same decisions on all test flights.

| | `=1` reference | `=0` optimised |
|---|---|---|
| information gain | expected KL, as Python (42 logs per cell view) | H(O) − Σ q_s H(O\|s), with H(O\|s) precomputed per pattern entry (6 logs) |
| (ρ, θ) evidence | log Z_new − log Z_old (2 logs) | log(Z_new / Z_old) (1 log) |
| flight-time bisection | 50 steps, `hypotf` | 24 steps (the float32 limit), squared distances |

## Planner pruning (`ZOI_V2_PLANNER_PRUNING`)

The planner scores every (cell, yaw) candidate by IG / T. Most of them cannot win, and an optimistic score is enough to prove it:

- **T_min**, a lower bound on the flight time T: the exact turn time, plus (d − 0.2 m) at the navigator's top speed √(0.5² + 0.5²) ≈ 0.71 m/s, plus one detector period. Since T ≥ T_min, the real rate is at most IG / T_min.
- **IG_max**, an upper bound on IG: the sum of the belief entropies of the cells the candidate would update (a view can't reveal more about a cell than its remaining uncertainty).

| value | search | exact IG computed | exact T computed |
|---|---|---|---|
| 0 | every candidate (default) | 1144 | 1144 |
| 1 | IG for every candidate, sorted by IG / T_min; exact T only if IG / T_min ≥ best rate so far | 1144 | 9.2 on average, max 38 |
| 2 | sorted by IG_max / T_min; stop as soon as it is below the best exact rate; exact IG, then T only if still able to win | 241 on average (flight means 96–426), max 958 | 9.2 on average, max 38 |

Averages over all plans of the 30 test flights. Both levels take **exactly** the decisions of the exhaustive search: `make test` compares them plan by plan. The search stops only on a strictly lower bound (possible ties are still examined), and both bounds carry a 1e-4 margin against float32 rounding. Candidates are examined in steps of 8 (one per cluster core), with the best rate updated between steps.

There is no planning radius to choose: the cut-off comes from the beliefs and the navigator's speed, so it is small when there is information nearby and grows only when the drone has to go far.

## IG reuse between plans (`ZOI_V2_IG_CACHE`)

**Why.** On GAP8 the worst detector period is the one with two planner calls. After a plan the drone flies to the target, and on arrival it plans again. The replan cooldown (0.4 s) is shorter than the detector period (0.625 s), so two plans can fall in the same period. This happened in 2,591 of the 21,533 periods of the 100 A50M flights (12%). In A50M seed 33, frame 9, the two plans took 242 ms together, which with MobileNet (410 ms) comes to 652 ms, over the 625 ms budget.

**Idea.** The exact IG of a candidate depends only on the beliefs `q` and on the table map the planner sees. It does not depend on the pose. Between two plans with no detector frame in between, `q` is the same, so the second plan can reuse the IGs of the first. Only the flight times T, which do depend on the pose, are recomputed, and those are the cheap part. The decisions stay **bit-identical**: same candidate, same rate, same T.

| value | behaviour |
|---|---|
| 0 | every plan starts from scratch (default, keeps the existing measurements reproducible) |
| 1 | exact IGs are kept between plans while they are still valid |

**What is cached.** Nothing new per candidate. The cache is `cand_ig` plus bit 0 of `cand_eval` ("exact IG known"), which survives between plans. Bit 2 marks the IGs computed by the current plan, and `n_ig_evals` counts it, so `n_ig_evals` is the work done. With pruning 2 the per-cell entropies `prune_H` are kept too. The state gains a validity flag, a counter and the table map the cached IGs were computed with: `sizeof(zoi_state_t)` goes from 64,468 to 64,620 B with pruning, and from 56,428 to 56,580 B without. With cache 0 the state is unchanged.

**When cached IGs are dropped.**
- **When `q` is recomputed:** `zoi_refresh_beliefs` clears the whole cache. `q` is written only there, so this covers every path. A frame with no visible cell leaves `q` unchanged, so the cache stays valid, which is correct.
- **When the planner's table map changes.** `zoi_plan` uses `table_map` plus, for the first plan after a frame only, that frame's desk cells, so two plans can see different maps. `is_table[j]` enters a candidate's IG only through the coverage gate (`merged > 0 && merged >= threshold`, on the table or the floor coverage); the likelihood is the pattern entry's own. So a candidate's IG changes only if it sees a changed cell through a view whose gate differs between the two planes. Only those candidates are dropped: for each changed cell, at most 88 pattern entries are checked. The others read the same `q` and the same gates, so they sum the same terms in the same order. Dropping everything would also be correct, but it would lose the savings exactly when the frame's desk cells differ from the persistent map.

**Pruning.** Levels 0 and 1 skip the IGs they already have. Level 2 skips the entropy kernel when `q` is unchanged. For a candidate whose IG is cached it sorts with the Level 1 key, IG / T_min (with the same 1e-4 margin), instead of the entropy bound. This is still exact: both keys bound the rate from above, and the search stops only on a key strictly below the best exact rate. So a skipped candidate cannot win or tie, the tie rule in `better()` does not depend on the order, and the IG and T values of the candidates examined are the same exact values. Only the number of T computations can change, not the result.

**Verification** (host, `make test`, 10 flights: hard π = 0.5 with MobileNet (5 seeds, including seed 33), hard with YOLO nano, medium and easy with both detectors):
- All 12 variants pass against Python: reference or optimised kernels × pruning 0 / 1 / 2 × cache 0 / 1.
- For each kernel variant, the 6 pruning × cache builds take the same decisions on every plan, with rate and T identical bit for bit. This includes the later plans of a frame.
- Stress test, `MIN_PLANS=3`: three plans on every frame, poses offset within the arrival tolerances, and a synthetic set of desk cells on every other plan, so the table map changes between consecutive plans. Up to 607 cached IGs are dropped in one plan, and the decisions stay identical.
- With the selective drop disabled, the same stress test fails: 101 plans differ on seed 33.
- `tools/v2_c/make_gap8_batch.sh` with `IG_CACHE=1`, 100 A50M flights: 100 / 100 OK. The replay passes with and without the cache with identical decisions, `gap8/main.c` on the host gives N / N plans equal to Python in both modes, and the decision hash is the same with the cache on and off.

Exact IG computations of the later plan of a period (the flight's sequences, from the arrival pose):

| | cache 0 mean | cache 0 max | cache 1 mean | cache 1 max |
|---|---|---|---|---|
| pruning 2, 100 A50M flights (2,591 plans) | 97.8 | 750 | 6.2 | 43 |
| pruning 2, 10 test flights (223 plans) | 91.6 | 621 | 5.4 | 20 |
| pruning 0 or 1, 10 test flights | 1144 | 1144 | 0 | 0 |

In the recorded flights the table map never differs between the two plans of a period: the frame's desk cells are already in the persistent map. So the IGs still computed with the cache are those of candidates that the first plan had pruned but that the second one, from another pose, has to examine. The exact T computations do not change (about 8 per plan). The time on GAP8 comes from gvsoc (`gap8/run_batch.sh`, below).

## Checking against Python

1. **Reference traces.** `tools/v2_c/export_trace.py` replays a flight's `perception_log.jsonl` (logged poses and boxes) through the existing Python code. It writes the inputs and every intermediate result to a text trace: coverage, events, desk map, shadow cells, claims, ρ/θ, all beliefs, and the planner decision at every frame.
   - The replay reproduces the ρ̂/θ̂ logged in flight to 5 decimals, so the Python side is faithful.
2. **Host test.** `host/replay_test.c` feeds the same poses and boxes to the C code and compares everything.

```bash
# from the repo root
python tools/v2_c/export_trace.py <run dir with perception_log.jsonl> -o /tmp/trace.txt
cd embedded/aif_zoi_v2
make                                  # build/replay_test_{ref,refprune1,refprune2,opt,prune1,prune2}[_igc]
./build/replay_test_ref /tmp/trace.txt   # -v lists every difference
./build/replay_test_opt /tmp/trace.txt
make test TRACES="/tmp/trace.txt ..."  # every variant; per kernel, all pruning levels and IG cache on/off
                                       # must take the same decisions, rate and T bit for bit, on every plan
make test TRACES="..." MIN_PLANS=3     # stress test: 3 plans per frame, table map changing between them
```

Plan sequences. After each frame the replay makes as many plans as the flight did (`REPLANS` in the trace). The first plan is made from the frame's pose, with that frame's desk cells in the planner's table map. This is where Python took the decision stored in the trace, and the first plan is checked against it. The later plans follow as in flight: they happen at a later control step, after `zoi_control_step` has cleared `frame_desk_valid`, from the arrival pose on the previous plan's target (cell centre, target yaw). The trace has no Python decision for them, so they are compared between builds only.

Results on 30 flights (seeds 2–4 of hard rooms A, B and C with MobileNet, hard A and B with YOLO nano, medium and easy with both detectors, and the hard π = 0.5 sweep), both variants:

| check | result |
|---|---|
| coverage (floor, table, merged) | max error 4.8e-7 |
| visible mask, target events, desk map, shadow cells, claims | identical |
| beliefs q (all cells, all frames) | max error 3.6e-6 |
| ρ, θ mean and sd | max error 4.5e-7 |
| planner decisions | 5,470 / 5,470 identical |

## GAP8

- `make gap8-check [PRUNING=0|1|2] [IG_CACHE=0|1]` compiles the library with the PULP toolchain (`-march=rv32imcxgap8`). It reports the size (about 21 kB of code; 24–26 kB with pruning 2, plus about 0.7 kB with the IG cache) and confirms that no double-precision routine is called. Only the single-precision soft-float routines appear (`__addsf3`, `__mulsf3`, ...), because GAP8 has no FPU.
- `make gap8-host [PRUNING=0|1|2] [FLIGHT=0|1] [IG_CACHE=0|1]` runs `gap8/main.c` on the PC with `host/pmsis.h`, a stand-in for the SDK. The 8 cores run one after the other, so the work split is exercised too. It must print `plans equal to Python: 232 / 232`.
- `gap8/` is the gvsoc program. It runs the recorded flight in `gap8/trace_data.h` (hard room A, MobileNet, seed 1, 232 frames; regenerate it with `tools/v2_c/make_gap8_trace.py`) and prints the cycles of each stage. See **[GVSOC_GUIDE.md](GVSOC_GUIDE.md)**.
- `gap8/main.c` with `FLIGHT=1` makes the plans of the recorded flight. The first plan of a frame comes from the frame's pose and is checked against Python. Since 2026-11-10 a later plan of the same frame (the second plan of a period) is made as in flight, whatever `IG_CACHE`: with `frame_desk_valid = 0`, from the arrival pose on the previous target (cell centre, target yaw). The same sequence runs in `host/replay_test.c`. Earlier `FLIGHT=1` outputs (seed 1, seed 33) made it from the frame's pose with the frame's desk cells, so they differ on the frames with two plans and remain only as a historical reference. "plans equal to Python" still checks only the first plan, because the trace has no Python decision for the second. The last line, `plan decisions hash`, covers every plan, so two builds can be compared (IG cache on and off).
- Batch over a whole campaign: `IG_CACHE=0|1 tools/v2_c/make_gap8_batch.sh <campaign dir>` builds and checks the headers on the host. Then, in the GAP SDK, `IG_CACHE=0|1 gap8/run_batch.sh` runs gvsoc on every header, one directory per configuration (`batch/flight_igcache0`, `batch/flight_igcache1`, `batch/full_*`). The summary is `python3 tools/v2_c/summarize_gap8_batch.py [--igcache 0|1] [--mode flight|full]`.
