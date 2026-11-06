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

## Checking against Python

1. **Reference traces.** `tools/v2_c/export_trace.py` replays a flight's `perception_log.jsonl` (logged poses and boxes) through the existing Python code. It writes the inputs and every intermediate result to a text trace: coverage, events, desk map, shadow cells, claims, ρ/θ, all beliefs, and the planner decision at every frame.
   - The replay reproduces the ρ̂/θ̂ logged in flight to 5 decimals, so the Python side is faithful.
2. **Host test.** `host/replay_test.c` feeds the same poses and boxes to the C code and compares everything.

```bash
# from the repo root
python tools/v2_c/export_trace.py <run dir with perception_log.jsonl> -o /tmp/trace.txt
cd embedded/aif_zoi_v2
make                                  # build/replay_test_ref and build/replay_test_opt
./build/replay_test_ref /tmp/trace.txt   # -v lists every difference
./build/replay_test_opt /tmp/trace.txt
```

Results on 30 flights (seeds 2–4 of hard rooms A, B and C with MobileNet, hard A and B with YOLO nano, medium and easy with both detectors, and the hard π = 0.5 sweep), both variants:

| check | result |
|---|---|
| coverage (floor, table, merged) | max error 4.8e-7 |
| visible mask, target events, desk map, shadow cells, claims | identical |
| beliefs q (all cells, all frames) | max error 3.6e-6 |
| ρ, θ mean and sd | max error 4.5e-7 |
| planner decisions | 5,470 / 5,470 identical |

## GAP8

- `make gap8-check` compiles the library with the PULP toolchain (`-march=rv32imcxgap8`). It reports the size (about 21 kB of code) and confirms that no double-precision routine is called. Only the single-precision soft-float routines appear (`__addsf3`, `__mulsf3`, ...), because GAP8 has no FPU.
- `make gap8-host` runs `gap8/main.c` on the PC with `host/pmsis.h`, a stand-in for the SDK. The 8 cores run one after the other, so the work split is exercised too. It must print `plans equal to Python: 232 / 232`.
- `gap8/` is the gvsoc program. It runs the recorded flight in `gap8/trace_data.h` (hard room A, MobileNet, seed 1, 232 frames; regenerate it with `tools/v2_c/make_gap8_trace.py`) and prints the cycles of each stage. See **[GVSOC_GUIDE.md](GVSOC_GUIDE.md)**.
