# ZOI in C (float32)

A C99 port of the `zoi` policy for the GAP8 processor of the Crazyflie AI-deck. It also ports the per-frame geometry that the controller runs before the policy. It therefore covers everything between the detector output and the velocity command. It builds on a PC too. Every number is a `float`, and there is no `double` anywhere. There is no `malloc`: all memory lives in one `zoi_state_t` of about 55 kB.

| Python | C |
|---|---|
| `env/env_manager.py` (footprint coverage, bbox → cell) | `src/zoi_geometry.c` |
| `vision/perception_projection.py`, `env/surface_projection.py`, `task/claim_manager.py` (desk map, target projection, shadow cells, visible mask, claims) | `src/zoi_perception.c` |
| `policies/zoi_model.py`, belief update and learned (ρ, θ) in `policies/zoi.py` | `src/zoi_belief.c` |
| planner (IG / T) in `policies/zoi*.py`, `policies/planner_time.py` | `src/zoi_planner.c` |
| `utils/navigator.py`, replan rule of `compute_action` | `src/zoi.c` |
| calibration counts (`configuration/zoi_params.py`), controller constants (`crazyflie_AIF.py`) | `src/zoi_config.c` |

The API is in `include/zoi.h`; the sizes are in `include/zoi_config.h`.

```c
zoi_config_t cfg;
zoi_config_default(&cfg, ZOI_ENV_HARD, ZOI_DET_MOBILENET);
static zoi_state_t st;
zoi_init(&st, &cfg);
/* every detector frame */  zoi_perception_frame(&st, &pose, boxes, n_boxes);
/* every control step   */  zoi_control_step(&st, &pose, time_s, &cmd);   /* cmd.forward, .sideways, .yaw_rate */
```

## Differences from the Python code

None of these changes the model. They exist because of float32 and the memory budget.

- **`log ell` per cell instead of `ell`.** In float32 a linear `ell` underflows to 0, and a zero state never recovers. In the test flights `log ell` reached −509, below the float32 limit of −87.3.
- **`log_ev` is re-centred** to a maximum of 0 after every frame. The weights are unchanged.
- **Beliefs are computed only when the planner reads them.** This is exact, because only the planner uses them.
- **View patterns.** The cells seen from a candidate pose, with their floor- and table-plane coverages, are the same relative to the candidate's cell for every cell. They are therefore computed once per yaw at start-up: 88 entries instead of an 11,080-entry per-candidate table (about 110 kB). Each entry's likelihood is precomputed too.

## Build options

All options are compile-time macros. Every combination takes the same decisions as the Python code.

| macro | values |
|---|---|
| `ZOI_REFERENCE_KERNELS` | `1`: the same formulas as Python; `0`: optimised kernels (see below) |
| `ZOI_PLANNER_PRUNING` | `0`: exhaustive search; `1`, `2`: exact pruning (see below) |
| `ZOI_IG_CACHE` | `0`: every plan starts from scratch; `1`: reuse information gains between plans |

### Kernels

| | reference | optimised |
|---|---|---|
| information gain | expected KL, as in Python (42 logs per cell view) | H(O) − Σ q_s H(O\|s), with H(O\|s) precomputed per pattern entry (6 logs) |
| (ρ, θ) evidence | log Z_new − log Z_old (2 logs) | log(Z_new / Z_old) (1 log) |
| flight-time bisection | 50 steps, `hypotf` | 24 steps (the float32 limit), squared distances |

### Planner pruning

The planner scores each (cell, yaw) candidate by IG / T. Most candidates cannot win, and an optimistic score is enough to prove it:

- **T_min** is a lower bound on the flight time T. It is the exact turn time, plus (d − 0.2 m) at the navigator's top speed √(0.5² + 0.5²) ≈ 0.71 m/s, plus one detector period.
- **IG_max** is an upper bound on IG: the sum of the belief entropies of the cells the candidate would update.

| level | search | exact IGs per plan | exact Ts per plan |
|---|---|---|---|
| 0 | every candidate | 1144 | 1144 |
| 1 | all IGs, candidates sorted by IG / T_min; T computed only while IG / T_min ≥ the best rate so far | 1144 | 9.2 mean, 38 max |
| 2 | candidates sorted by IG_max / T_min; the search stops as soon as the bound falls below the best exact rate | 241 mean, 958 max | 9.2 mean, 38 max |

These are statistics over all plans of the test flights. The search stops only on a strictly lower bound, so ties are still examined. Both bounds carry a 1e-4 margin against float32 rounding. The result is the exhaustive search's decision. Candidates are examined in groups of 8, one per cluster core. There is no planning radius: the cut-off follows from the beliefs and the navigator's speed.

### Reusing information gains between plans

The replan cooldown (0.4 s) is shorter than the detector period (0.625 s), so two plans can fall in the same detector period. This happens in about one period in eight in the recorded flights. A candidate's exact IG depends only on the beliefs `q` and on the table map the planner sees, not on the pose. Between two plans with no detector frame in between, the second plan therefore reuses the first plan's IGs and recomputes only the flight times.

The cached IGs are dropped in two cases:
- **when `q` is recomputed**, in `zoi_refresh_beliefs`, the only place that writes `q`;
- **when the planner's table map changes** between the two plans. Then only the candidates whose view of a changed cell crosses a coverage gate differently are dropped.

The decisions, rates and flight times stay bit-identical. With pruning level 2, the second plan of a period computes 6 exact IGs on average instead of 98.

## Checking against Python

1. **Reference traces.** `tools/c_port/export_trace.py` replays a flight's `perception_log.jsonl` (logged poses and boxes) through the Python code. It writes the inputs and every intermediate result to a text trace: coverage, events, desk map, shadow cells, claims, ρ/θ, all beliefs, and the planner decision at every frame.
2. **Host test.** `host/replay_test.c` feeds the same poses and boxes to the C code and compares everything. After each frame it makes as many plans as the flight did. The first plan is checked against Python. The later plans are compared between builds, which must agree bit for bit.

```bash
# from the repo root; the flight must have been run with AIF_PERCEPTION_LOG=1
python tools/c_port/export_trace.py <run dir with perception_log.jsonl> -o /tmp/trace.txt
cd embedded/zoi
make                                    # build/replay_test_<variant>[_igc], every kernel × pruning × cache variant
./build/replay_test_ref /tmp/trace.txt  # -v lists every difference
make test TRACES="/tmp/trace.txt ..."   # every variant on every trace
make test TRACES="..." MIN_PLANS=3      # stress test: 3 plans per frame, table map changing between them
```

Results on 31 flights (easy, medium and hard rooms, both detectors), reference and optimised kernels. All 12 builds take the same decisions on every plan.

| check | result |
|---|---|
| coverage (floor, table, merged) | max error 4.8e-7 |
| visible mask, target events, desk map, shadow cells, claims | identical |
| beliefs q (all cells, all frames) | max error 3.6e-6 |
| ρ, θ mean and sd | max error 4.5e-7 |
| planner decisions | 5,702 / 5,702 identical |

## GAP8

- `make gap8-check [PRUNING=0|1|2] [IG_CACHE=0|1]` compiles the library with the PULP toolchain (`-march=rv32imcxgap8`). It reports the code size: about 21 kB, 24–26 kB with pruning level 2, and about 0.7 kB more with the IG cache. It also confirms that no double-precision routine is called. Only single-precision soft-float routines appear, because GAP8 has no FPU.
- `make gap8-host [PRUNING=0|1|2] [FLIGHT=0|1] [IG_CACHE=0|1]` runs `gap8/main.c` on the PC with `host/pmsis.h`, a stand-in for the SDK. The 8 cores run one after the other, so the work split is exercised too. It must print `plans equal to Python: 232 / 232`.
- `gap8/` is the gvsoc program. It replays the recorded flight in `gap8/trace_data.h` (hard environment, MobileNetV2-SSD, seed 1, 232 frames) and prints the cycles of each stage. Regenerate the trace with `tools/c_port/make_gap8_trace.py`.

```bash
source <gap_sdk>/configs/ai_deck.sh
cd embedded/zoi/gap8
make clean all run platform=gvsoc                          # 8 cores, optimised kernels
make clean all run platform=gvsoc NB_CORES=1 REFERENCE=1   # 1 core, reference kernels
make clean all run platform=gvsoc PRUNING=2 IG_CACHE=1 FLIGHT=1
```

Options of `gap8/Makefile`:
- `NB_CORES` (1–8);
- `REFERENCE`, `PRUNING`, `IG_CACHE`;
- `FLIGHT`: `0` makes a plan on every frame (the worst case), `1` makes the plans of the recorded flight;
- `MAX_FRAMES`, for shorter runs;
- `CL_FREQ_MHZ`, used only to print milliseconds.

**A whole campaign:**
1. On the host, `tools/c_port/make_gap8_batch.sh <campaign dir>` builds and checks one header per flight.
2. In the GAP SDK, `gap8/run_batch.sh` runs gvsoc on each one. The outputs go to `gap8/batch/`.
3. `python3 tools/c_port/summarize_gap8_batch.py` summarises them.
