# ZOI in C (float32)

A C99 port of the `zoi` policy for the GAP8 processor of the Crazyflie AI-deck. It also ports the per-frame geometry that the controller runs before the policy. It therefore covers everything between the detector output and the velocity command. It builds on a PC too. Every number is a `float`, and there is no `double` anywhere. There is no `malloc`: all memory lives in one `zoi_state_t` of about 65 kB.

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

None of these changes the model. They exist because of float32 and the memory and time budgets of GAP8.

- **`log ell` per cell instead of `ell`.** In float32 a linear `ell` underflows to 0, and a zero state never recovers. In the test flights `log ell` reached −509, below the float32 limit of −87.3.
- **`log_ev` is re-centred** to a maximum of 0 after every frame. The weights are unchanged.
- **Beliefs are computed only when the planner reads them.** This is exact, because only the planner uses them.
- **View patterns.** The cells seen from a candidate pose, with their floor- and table-plane coverages, are the same relative to the candidate's cell for every cell. They are therefore computed once per yaw at start-up: 88 entries instead of an 11,080-entry per-candidate table (about 110 kB). Each entry's likelihood is precomputed too.
- **Cheaper formulas for the same quantities.**
  - The information gain is computed as H(O) − Σ q_s H(O|s), with H(O|s) precomputed per pattern entry: 6 logs per cell view instead of the 42 of the expected KL.
  - The (ρ, θ) evidence update takes one log per cell, log(Z_new / Z_old).
  - The flight-time bisection stops after 24 steps, the float32 resolution, and compares squared distances.
- **Pruned planner search** and **information gains reused between plans**, described below. They never change a decision.

## Planner search

The planner scores each (cell, yaw) candidate by IG / T. Most candidates cannot win, and an optimistic score is enough to prove it:

- **T_min** is a lower bound on the flight time T. It is the exact turn time, plus (d − 0.2 m) at the navigator's top speed √(0.5² + 0.5²) ≈ 0.71 m/s, plus one detector period.
- **IG_max** is an upper bound on IG: the sum of the belief entropies of the cells the candidate would update.

The candidates are sorted by IG_max / T_min. The search computes the exact IG of each in turn, and its exact T only while IG / T_min can still beat the best rate so far. It stops as soon as the bound falls below the best exact rate. On the first plan after each frame of the test flights, it computed 241 exact IGs on average (958 at most) and 9.2 exact flight times (38 at most), out of 1144 candidates.

The search stops only on a strictly lower bound, so ties are still examined. Both bounds carry a 1e-4 margin against float32 rounding. The result is the exhaustive search's decision. Candidates are examined in groups of 8, one per cluster core. There is no planning radius: the cut-off follows from the beliefs and the navigator's speed.

### Reusing information gains between plans

The replan cooldown (0.4 s) is shorter than the detector period (0.625 s), so two plans can fall in the same detector period. This happens in about one period in eight in the recorded flights. A candidate's exact IG depends only on the beliefs `q` and on the table map the planner sees, not on the pose. Between two plans with no detector frame in between, the second plan therefore reuses the first plan's IGs and recomputes only the flight times.

The cached IGs are dropped in two cases:
- **when `q` is recomputed**, in `zoi_refresh_beliefs`, the only place that writes `q`;
- **when the planner's table map changes** between the two plans. Then only the candidates whose view of a changed cell crosses a coverage gate differently are dropped.

A reused IG is the value a fresh computation would give, so the decisions, rates and flight times are bit-identical to planning from scratch. The second plan of a period computes 6 exact IGs on average, against 98 without the reuse.

## Checking against Python

1. **Reference traces.** `tools/c_port/export_trace.py` replays a flight's `perception_log.jsonl` (logged poses and boxes) through the Python code. It writes the inputs and every intermediate result to a text trace: coverage, events, desk map, shadow cells, claims, ρ/θ, all beliefs, and the planner decision at every frame.
2. **Host test.** `host/replay_test.c` feeds the same poses and boxes to the C code and compares everything. After each frame it makes as many plans as the flight did. The first plan is checked against Python; the trace has no Python decision for the later ones.

```bash
# from the repo root; the flight must have been run with AIF_PERCEPTION_LOG=1
python tools/c_port/export_trace.py <run dir with perception_log.jsonl> -o /tmp/trace.txt
cd embedded/zoi
make                                    # build/replay_test
./build/replay_test /tmp/trace.txt      # -v lists every difference
make test TRACES="/tmp/trace.txt ..."   # every trace
make test TRACES="..." MIN_PLANS=3      # stress test: 3 plans per frame, table map changing between them
```

Results on 31 flights (easy, medium and hard rooms, both detectors):

| check | result |
|---|---|
| coverage (floor, table, merged) | max error 4.8e-7 |
| visible mask, target events, desk map, shadow cells, claims | identical |
| beliefs q (all cells, all frames) | max error 3.6e-6 |
| ρ, θ mean and sd | max error 4.5e-7 |
| planner decisions | 5,702 / 5,702 identical |

## GAP8

We deployed the port on a real Crazyflie with an AI-deck. Without the hardware, gvsoc, the GAP8 simulator of the GAP SDK, runs the same program on recorded flights and counts the cycles.

- `make gap8-check` compiles the library with the PULP toolchain (`-march=rv32imcxgap8`). It reports the code size, about 26 kB, and confirms that no double-precision routine is called. Only single-precision soft-float routines appear, because GAP8 has no FPU.
- `make gap8-host [FLIGHT=0|1]` runs `gap8/main.c` on the PC with `host/pmsis.h`, a stand-in for the SDK. The 8 cores run one after the other, so the work split is exercised too. It must print `plans equal to Python: 232 / 232`.
- `gap8/` is the gvsoc program. It replays the recorded flight in `gap8/trace_data.h` (hard environment, MobileNetV2-SSD, seed 1, 232 frames) and prints the cycles of each stage. Regenerate the trace with `tools/c_port/make_gap8_trace.py`.

```bash
source <gap_sdk>/configs/ai_deck.sh
cd embedded/zoi/gap8
make clean all run platform=gvsoc               # 8 cores, a plan on every frame
make clean all run platform=gvsoc NB_CORES=1    # 1 core
make clean all run platform=gvsoc FLIGHT=1      # the plans of the recorded flight
```

Options of `gap8/Makefile`:
- `NB_CORES` (1–8);
- `FLIGHT`: `0` makes a plan on every frame (the worst case), `1` makes the plans of the recorded flight;
- `MAX_FRAMES`, for shorter runs;
- `CL_FREQ_MHZ`, used only to print milliseconds;
- `ROWS`, `COLS`: the map size (see below).

**A whole campaign:**
1. On the host, `tools/c_port/make_gap8_batch.sh <campaign dir>` builds and checks one header per flight.
2. In the GAP SDK, `gap8/run_batch.sh` runs gvsoc on each one. The outputs go to `gap8/batch/flight/`.
3. `python3 tools/c_port/summarize_gap8_batch.py` summarises them and lists the failed runs.

**Larger rooms.** The map size is a compile-time option (`ZOI_ROWS`, `ZOI_COLS`; `ROWS=`, `COLS=` in the Makefiles). `tools/c_port/make_tiled_room_header.py --tiles TRxTC` lays TR × TC copies of the Webots room side by side. It replays the recorded flights of TR·TC seeds one after the other, one per tile, and keeps the state between them. There is no Python decision for these rooms, so the output reads `plans equal to Python: 0 / 0`. `tools/c_port/scaling_gap8.py` fits the cost against the number of cells.

```bash
B=embedded/zoi/gap8/batch
for t in 1x2 1x3 2x2; do python3 tools/c_port/make_tiled_room_header.py --tiles $t --group $B/headers -o $B/tiles_$t/headers; done
cd embedded/zoi/gap8      # GAP SDK sourced; seeds 1..24 at every size
BATCH=batch/tiles_1x2 COLS=22 SEEDS="$(seq -s ' ' 1 2 23)" ./run_batch.sh
BATCH=batch/tiles_1x3 COLS=33 SEEDS="$(seq -s ' ' 1 3 22)" ./run_batch.sh
BATCH=batch/tiles_2x2 ROWS=26 COLS=22 SEEDS="1 5 9 13 17 21" ./run_batch.sh
cd ../../.. && python3 tools/c_port/scaling_gap8.py --seeds-upto 24 $B/flight $B/tiles_{1x2,1x3,2x2}/flight
```

| map (cells) | 13 × 11 (143) | 13 × 22 (286) | 13 × 33 (429) | 26 × 22 (572) |
|---|---|---|---|---|
| `zoi_state_t` | 64.6 kB | 98.5 kB | 132.4 kB | 166.3 kB |
