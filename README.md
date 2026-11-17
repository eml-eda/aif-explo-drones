# Active Inference for Nano-drone Object Search

This repository contains the Webots simulation, navigation policies, perception
pipeline, dataset-generation tools, detector training code, evaluation
infrastructure and embedded C implementation developed for autonomous indoor
object search with a nano-drone.

The task is to find bottles and cans in indoor environments of increasing
difficulty. A perception-aware Active Inference policy, ZOI, is compared with
four lightweight navigation baselines and across several perception backends,
under a common simulation and evaluation pipeline.

## Overview

Experiments run in Webots with a simulated Crazyflie equipped with a
downward-facing camera.

Three canonical environments are provided:

- `easy`: bottles and cans on the floor.
- `medium`: bottles and cans may also be on desks; desks are detected as contextual objects.
- `hard`: the medium setting with additional non-target distractor objects and more challenging visual conditions.

Bottles and cans are always the targets. Desks provide context in the medium
and hard environments but are not search targets.

## Navigation policies

| Policy ID | Description |
| --- | --- |
| `wall_following` | Boundary-following exploration |
| `spiral` | Spiral coverage trajectory |
| `pseudo_random` | Lightweight randomized exploration |
| `rotate_and_measure` | Rotation-and-translation coverage baseline |
| `zoi` | Active Inference with a calibrated joint surface/target belief, learned priors and an information-per-second planner |

The four baselines do not use the detector output to choose where to fly.

### ZOI

`zoi` is an Active Inference search with a calibrated observation model and no
hand-set weights. The policy is in
`webots/controllers/crazyflie_AIF/policies/zoi.py`, and the model is in
`policies/zoi_model.py`.

- **Belief.** Each grid cell holds a belief over {table, floor} × {empty, bottle, can}.
  Every processed frame gives each visible cell two observations:
  - a desk observation: whether a desk box covers the cell. It is informative when the cell's table-plane coverage passes the update gate.
  - a target observation: nothing, a bottle box or a can box. Its detection probability is `max(f, w·r)`, where `w` is the cell's coverage at table height in the table states and at floor height in the floor states.

  Detector confidence does not enter the likelihood.
- **Prior.** The share of desk cells `ρ` and the fraction of cells holding a
  target `θ` are unknown. They start from a uniform prior on their feasible
  region and are inferred in flight from all the evidence. Each cell's belief is
  the exact average over `p(ρ, θ | data)`. Only the share of targets on desks,
  `π`, is given. Its default is 0.5, which tells the policy nothing about where
  targets are more likely. Without desks (easy), `ρ = 0` and only `θ` is learned.
- **Planner.** Each (cell, yaw) candidate is ranked by its expected information
  gain divided by `T`. `T` is the navigator's exact arrival time under its own
  control law, plus one detector period (`policies/planner_time.py`).
  - The default implementation computes the planner on arrays (`policies/zoi_vectorized.py`).
  - `ZOI_PLANNER_IMPL=loop` selects the plain Python reference in `policies/zoi.py`. It gives identical information gains, flight times and decisions, about 50 times more slowly.
- **Grid over (`ρ`, `θ`).** `ZOI_GRID_SIZE` points per side over (0, 1), with
  only the feasible points kept. The default is 32, about 512 points (8 kB), a
  size chosen for the GAP8. Grids of 128, 64, 32 and 16 points per side gave the
  same success rate.

The detector values come from an offline calibration of each environment's
checkpoint of each detector. Each environment has one calibration image set: the
dataset world, seed 101, 400 images, never used for training or evaluation. It is
built by `tools/analysis/build_calibration.sh`. A target box counts as a
detection when its centre is inside the label box, as in flight. Counts become
probabilities by the rule of succession `(k + 1) / (n + 2)`:

| Parameter | Variable | Hard MBN | Hard YOLO-n | Medium MBN | Medium YOLO-n | Easy MBN | Easy YOLO-n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target recall `r` | `ZOI_RECALL` | 0.834 | 0.994 | 0.968 | 0.995 | 0.901 | 0.961 |
| class accuracy `a` | `ZOI_CLASS_ACCURACY` | 0.993 | 0.994 | 0.995 | 0.995 | 0.997 | 0.997 |
| false alarm per empty cell `f` | `ZOI_FALSE_ALARM` | 0.00086 | 0.0020 | 0.00029 | 0.0012 | 0.0072 | 0.0054 |
| desk recall | `ZOI_DESK_RECALL` | 0.953 | 0.958 | 0.950 | 0.958 | — | — |
| desk false alarm | `ZOI_DESK_FALSE_ALARM` | 0.127 | 0.134 | 0.179 | 0.184 | — | — |

The share of targets on desks given to the policy, `π`, is set with
`ZOI_P_ON_TABLE` (default 0.5). Easy has no desks, so only `r`, `a` and `f` are
used. Any other (environment, detector) pair needs the values set through the
variables. Any other `ZOI_*` variable is rejected at startup.

Each run writes the values it used to `policy_parameters.json` in its run
directory. With `AIF_PERCEPTION_LOG=1`, each run also writes a per-frame
`perception_log.jsonl`. It includes the `ρ` and `θ` estimates
(`policy.rho_mean`, `rho_sd`, `theta_mean`, `theta_sd`) and the planner's
wall-clock time.

### Embedded implementation

`embedded/zoi/` is a float32 C99 port of the ZOI policy and of the per-frame
geometry that runs before it, for the GAP8 processor of the Crazyflie AI-deck.
It takes the same decisions as the Python code on replayed flights. Its
[README](embedded/zoi/README.md) covers the build, the checks against Python, and
the cycle counts with gvsoc.

## Perception backends

Four detector IDs are supported:

| Detector ID | Backend |
| --- | --- |
| `ground_truth` | Webots ground-truth perception |
| `yolo_nano` | YOLO11-nano |
| `yolo_small` | YOLO11-small |
| `mobilenet_ssd` | Custom MobileNetV2-SSD detector |

The learned detectors use environment-specific checkpoints.

Final checkpoints are stored under:

```text
detection_model_training/checkpoints/
├── easy/
│   ├── mobilenet_ssd.pt
│   ├── yolo_nano.pt
│   └── yolo_small.pt
├── medium/
│   ├── mobilenet_ssd.pt
│   ├── yolo_nano.pt
│   └── yolo_small.pt
└── hard/
    ├── mobilenet_ssd.pt
    ├── yolo_nano.pt
    └── yolo_small.pt
```

The checkpoint files are versioned with Git LFS. Their SHA-256 hashes are
provided in `detection_model_training/checkpoints/CHECKSUMS.sha256`.

## Requirements

The simulation targets **Webots R2025a**.

Python dependencies are listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Git LFS is required for the learned detector checkpoints:

```bash
git lfs install
git lfs pull
```

CUDA can be used for learned-detector inference. CPU inference is selected with
`AIF_DETECTOR_DEVICE=cpu`.

## Running evaluations

The canonical evaluation worlds are:

```text
webots/worlds/AIF_Drone_World_easy.wbt
webots/worlds/AIF_Drone_World_medium.wbt
webots/worlds/AIF_Drone_World_hard.wbt
```

The controller is configured through environment variables.

Easy environment, spiral baseline, ground-truth perception:

```bash
AIF_ENVIRONMENT=easy \
AIF_RUN_MODE=evaluation \
AIF_POLICY=spiral \
AIF_DETECTOR=ground_truth \
webots webots/worlds/AIF_Drone_World_easy.wbt
```

Medium environment, ZOI, YOLO11-nano:

```bash
AIF_ENVIRONMENT=medium \
AIF_RUN_MODE=evaluation \
AIF_POLICY=zoi \
AIF_DETECTOR=yolo_nano \
webots webots/worlds/AIF_Drone_World_medium.wbt
```

Hard environment, ZOI, MobileNetV2-SSD:

```bash
AIF_ENVIRONMENT=hard \
AIF_RUN_MODE=evaluation \
AIF_POLICY=zoi \
AIF_DETECTOR=mobilenet_ssd \
webots webots/worlds/AIF_Drone_World_hard.wbt
```

Campaigns run faster without rendering. Simulated time is what counts, so the
results are the same:

```bash
webots --no-rendering --mode=fast --minimize --batch --stdout --stderr <world>
```

Only one simulation should run at a time: the controller and the supervisor
exchange state through files in `webots/controllers/shared/`.

The model registry automatically selects the released checkpoint for the
requested environment and detector. A checkpoint can be overridden explicitly
with:

```bash
AIF_MODEL_PATH=/path/to/checkpoint.pt
```

The main detector runtime options are:

```text
AIF_DETECTOR_CONFIDENCE
AIF_DETECTOR_IOU
AIF_DETECTOR_IMAGE_SIZE
AIF_DETECTOR_DEVICE
```

Their defaults are defined in `webots/controllers/crazyflie_AIF/config.py`.

### Scenario variables

An evaluation campaign has 100 runs by default, with seeds 1 to 100 and 180 s
per run. In the medium and hard environments, these variables change the
scenario (prefix `MEDIUM_EVAL_` or `HARD_EVAL_`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `*_NUM_RUNS` | 100 | runs per campaign: more than 10 uses seeds 1..N, exactly 10 a fixed seed set, fewer is an error |
| `*_N_TABLES` | 3 | desks in the room |
| `*_N_BOTTLES`, `*_N_CANS` | 3, 3 | targets |
| `*_P_TARGET_ON_TABLE` | 0.9 | share of targets placed on desks |
| `*_FORCE_ONE_EMPTY_TABLE_NUM_RUNS` | 5 | the first N runs have one desk with no target |

The hard environment also has `HARD_EVAL_EPISODE_SECONDS` and the distractor
counts (`HARD_EVAL_N_FLOWER_POTS`, `HARD_EVAL_N_CEREAL_BOXES`,
`HARD_EVAL_N_PLATES`, `HARD_EVAL_N_SOCCER_BALLS`).

### Evaluation summaries

Every completed evaluation writes its detailed report and a compact metrics
summary under that campaign's `eval_report/` directory:

```text
eval_<policy>_<detector>_<environment>_<timestamp>/eval_report/eval_report.json
eval_<policy>_<detector>_<environment>_<timestamp>/eval_report/global_results.json
```

The compact result records environment, policy, detector, run count, success
rate, target recovery, time to success, and visited-cell ratio. Time to success
is calculated from successful runs only; target recovery and visited-cell
statistics use all runs. Reprocessing the same campaign replaces its entry by
`campaign_id` rather than adding a duplicate in that campaign's summary file.

## Reproducing the paper

Every campaign of the paper has 100 runs (seeds 1–100) and uses the default
ZOI settings: 32 × 32 (`ρ`, `θ`) grid, vectorized planner, `π` given as 0.5.
The detectors are `mobilenet_ssd` and `yolo_nano`, plus `yolo_small` for the
baselines. Each campaign is one command:

```bash
env AIF_RUN_MODE=evaluation AIF_ENVIRONMENT=<environment> AIF_POLICY=<policy> AIF_DETECTOR=<detector> \
  <room variables> \
  webots --no-rendering --mode=fast --minimize --batch --stdout --stderr \
  webots/worlds/AIF_Drone_World_<environment>.wbt
```

| Room | Environment | Room variables |
| --- | --- | --- |
| Easy | `easy` | — |
| Medium | `medium` | — |
| Hard A: 3 desks, 6 targets | `hard` | — |
| Hard B: 4 desks, 8 targets | `hard` | `HARD_EVAL_N_TABLES=4 HARD_EVAL_N_BOTTLES=4 HARD_EVAL_N_CANS=4` |
| Hard C: 2 desks, 4 targets | `hard` | `HARD_EVAL_N_TABLES=2 HARD_EVAL_N_BOTTLES=2 HARD_EVAL_N_CANS=2` |

- **Baselines:** the four baselines run in hard room A with each of the three learned detectors.
- **Sensitivity to `π`:** hard room A with `HARD_EVAL_P_TARGET_ON_TABLE=0.7`. ZOI is given `ZOI_P_ON_TABLE` = 0.9, 0.8, 0.7, 0.6 and 0.5.

Add `AIF_PERCEPTION_LOG=1` to record the per-frame log. The log is needed for
the belief-map figures and for exporting flights to the C port.

## Webots worlds

Eight canonical worlds are included.

Evaluation:

```text
AIF_Drone_World_easy.wbt
AIF_Drone_World_medium.wbt
AIF_Drone_World_hard.wbt
```

Dataset collection:

```text
AIF_Drone_World_dataset_easy.wbt
AIF_Drone_World_dataset_medium.wbt
AIF_Drone_World_dataset_hard.wbt
AIF_Drone_World_dataset_hard_a.wbt
AIF_Drone_World_dataset_hard_b.wbt
```

The hard A and B worlds are the two collection domains used to construct the
final hard detector dataset.

## Dataset collection

Dataset collection uses Webots Recognition Segmentation to generate object
annotations from the simulator rather than predictions from a learned detector.

The shared collection implementation is located in:

```text
webots/controllers/dataset_collection/
```

with environment-specific controller entry points for easy, medium, and hard
collection.

The custom Webots assets used by the collection pipeline include:

```text
webots/protos/CrazyFlieDownCam.proto
webots/protos/DatasetBeerBottle.proto
webots/protos/DatasetCan.proto
```

The camera operates at `320x240`, matching the detector pipeline.

### Prepared datasets

Prepared detector datasets are intentionally not stored directly in the Git
repository. They are distributed separately as release assets.

After downloading them, the expected local layout is:

```text
detection_model_training/datasets/
├── easy/
├── medium/
└── hard/
```

Each prepared dataset contains the standard train, validation, and test image
and label splits together with its dataset YAML.

The easy dataset contains bottle and can classes.

The medium and hard datasets contain bottle, can, and desk classes. Distractor
objects in the hard environment are intentionally left unlabeled.

### Hard A+B dataset

The released hard detector dataset is built from two collection domains with
different floor appearance and distractor identities while preserving the same
underlying search-task semantics.

Each collection domain contains six runs of 200 samples:

```text
Hard A: 1200 samples
Hard B: 1200 samples
Total:  2400 samples
```

The prepared split is:

```text
train: 1600
val:    400
test:   400
```

Runs with seeds 1–4 contribute to training, seed 5 to validation, and seed 6
to testing.

## Detector training and evaluation

Generic detector scripts are located directly under
`detection_model_training/`.

### YOLO11

Train YOLO11-nano on the medium environment:

```bash
python detection_model_training/train_yolo.py \
  --env medium \
  --model-size nano
```

Train YOLO11-small by replacing `nano` with `small`.

Offline evaluation uses:

```bash
python detection_model_training/eval_yolo.py \
  --env medium \
  --model-size nano
```

The same scripts support `easy`, `medium`, and `hard`.

### MobileNetV2-SSD

Training:

```bash
python detection_model_training/train_mobilenetv2_ssd_sigmoid.py \
  --env medium
```

Evaluation:

```bash
python detection_model_training/eval_mobilenetv2_ssd_sigmoid.py \
  --env medium
```

The detector scripts expect the prepared datasets under
`detection_model_training/datasets/`.

Training runs, evaluation outputs, temporary dataset YAML files, and historical
development workspaces are intentionally excluded from version control. Only
the final released checkpoints are stored in the repository.

## Repository structure

```text
.
├── detection_model_training/
│   ├── checkpoints/
│   ├── tools/
│   ├── train_yolo.py
│   ├── eval_yolo.py
│   ├── train_mobilenetv2_ssd_sigmoid.py
│   └── eval_mobilenetv2_ssd_sigmoid.py
│
├── embedded/
│   └── zoi/              C port of the ZOI policy (host and GAP8)
│
├── tools/
│   ├── analysis/         detector calibration and campaign analysis
│   └── c_port/           flight export and checks for the C port
│
├── webots/
│   ├── controllers/
│   │   ├── common/
│   │   ├── crazyflie_AIF/
│   │   ├── dataset_collection/
│   │   ├── crazyflie_easy_dataset_collector/
│   │   ├── crazyflie_medium_dataset_collector/
│   │   ├── crazyflie_hard_dataset_collector/
│   │   └── worldgen_*_supervisor/
│   ├── protos/
│   └── worlds/
│
├── requirements.txt
└── README.md
```

The main runtime controller is:

```text
webots/controllers/crazyflie_AIF/crazyflie_AIF.py
```

Its supporting implementation is divided into configuration, runtime,
perception, task-management, environment, navigation, and policy modules.

Runtime communication between controllers and supervisors uses small JSON state
files under:

```text
webots/controllers/shared/
```

These files are generated during execution and are intentionally excluded from
version control.

## Tests

The unit-test suite covers configuration, canonical naming, perception,
projection, target claiming, dataset preparation and collection support,
shared runtime state, surface projection, controller pipeline behavior, the ZOI
model and policy, and the planner's flight-time model.

Run the complete suite from the repository root with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover \
  -s webots/controllers/crazyflie_AIF/tests \
  -p 'test_*.py'
```

The C port has its own replay test against Python (`embedded/zoi/README.md`).

## Reproducibility notes

The evaluation supervisors use deterministic run seeds and explicitly record
experiment initialization and runtime state.

For campaigns with more than ten runs, seeds are assigned sequentially from 1
upward. Scene generation is seeded independently for each run. The same
environment, seed and scenario variables therefore give the same initial scene,
whatever the detector or policy, and a given policy and detector give the same
run.

Because ZOI is perception-driven, different detector outputs can change the
belief state and the subsequent trajectory. Closed-loop task performance should
therefore not be expected to vary monotonically with offline detector metrics.

## Generated and local-only files

The repository intentionally excludes:

```text
prepared detector datasets
raw dataset-collection output
detector training runs
offline evaluation output
temporary runtime JSON state
generated heatmaps and visualizations
analysis notebooks and figures
historical detector-development workspaces
```

This keeps the repository focused on the simulation, reusable source code,
released detector checkpoints, and reproducible training/evaluation tooling.
