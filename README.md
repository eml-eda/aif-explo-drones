# Active Inference for Nano-drone Object Search

This repository contains the Webots simulation, navigation policies, perception
pipeline, dataset-generation tools, detector training code, and evaluation
infrastructure developed for autonomous indoor object search with a Nano-drone.

The project studies whether perception-aware Active Inference policies can make
more informative search decisions than lightweight navigation baselines. The
task is to locate bottles and cans in indoor environments of increasing
difficulty while comparing multiple navigation policies and perception
backends under a common simulation and evaluation pipeline.

## Overview

Experiments are implemented in Webots and use a simulated Crazyflie equipped
with a downward-facing camera.

Three canonical environments are provided:

- `easy` — bottles and cans distributed on the floor.
- `medium` — bottles and cans may appear on desks; desks are also detected as contextual objects.
- `hard` — extends the medium setting with additional non-target distractor objects and more challenging visual conditions.

The search task always considers bottles and cans as targets. Desks provide
context in the medium and hard environments but are not search targets.

## Navigation policies

The repository contains four lightweight baseline policies and two Active
Inference policies:

| Policy ID | Description |
| --- | --- |
| `wall_following` | Boundary-following exploration |
| `spiral` | Spiral coverage trajectory |
| `pseudo_random` | Lightweight randomized exploration |
| `rotate_and_measure` | Rotation-and-translation coverage baseline |
| `aif_binary` | Active Inference using a binary target-presence belief |
| `aif_zoi` | Active Inference using target class and zone-of-interest context |

`aif_binary` maintains a binary belief over target absence or presence for each
grid cell.

`aif_zoi` extends the belief model with bottle, can, and unknown-target states,
together with a separate belief over desk occupancy. This allows the policy to
use contextual information when selecting informative observations.

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

CUDA can be used for learned-detector inference, but CPU inference can also be
selected through the runtime detector device setting.

## Running evaluations

The canonical evaluation worlds are:

```text
webots/worlds/AIF_Drone_World_easy.wbt
webots/worlds/AIF_Drone_World_medium.wbt
webots/worlds/AIF_Drone_World_hard.wbt
```

The controller is configured through environment variables.

Easy environment, AIF Binary, ground-truth perception:

```bash
AIF_ENVIRONMENT=easy \
AIF_RUN_MODE=evaluation \
AIF_POLICY=aif_binary \
AIF_DETECTOR=ground_truth \
webots webots/worlds/AIF_Drone_World_easy.wbt
```

Medium environment, AIF ZOI, YOLO11-nano:

```bash
AIF_ENVIRONMENT=medium \
AIF_RUN_MODE=evaluation \
AIF_POLICY=aif_zoi \
AIF_DETECTOR=yolo_nano \
webots webots/worlds/AIF_Drone_World_medium.wbt
```

Hard environment, AIF ZOI, MobileNetV2-SSD:

```bash
AIF_ENVIRONMENT=hard \
AIF_RUN_MODE=evaluation \
AIF_POLICY=aif_zoi \
AIF_DETECTOR=mobilenet_ssd \
webots webots/worlds/AIF_Drone_World_hard.wbt
```

The model registry automatically selects the released checkpoint associated
with the requested environment and detector.

A checkpoint can be overridden explicitly with:

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

The complete 15-campaign Easy-environment matrix can be run sequentially with:

```bash
scripts/run_easy_experiments.sh --list
scripts/run_easy_experiments.sh
```

Queue progress is saved under the ignored experiment logs. Rerunning the script
after an interruption resumes the queued campaign. Use `--status` to inspect
the next job, `--start-at N` to select a one-based queue position, or `--reset`
to return the queue to campaign 1.

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
shared runtime state, surface projection, and controller pipeline behavior.

Run the complete suite from the repository root with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover \
  -s webots/controllers/crazyflie_AIF/tests \
  -p 'test_*.py'
```

## Reproducibility notes

The evaluation supervisors use deterministic run seeds and explicitly record
experiment initialization and runtime state.

For standard evaluation campaigns with more than ten episodes, run seeds are
assigned sequentially from 1 upward. Scene generation is seeded independently
for each episode so detector and policy configurations can be evaluated from
the same initial scene definitions when the same environment and seed are used.

Because the Active Inference policies are perception-driven, different detector
outputs can change the belief state and subsequent trajectory. Consequently,
closed-loop task performance should not be expected to vary monotonically with
offline detector metrics.

## Generated and local-only files

The repository intentionally excludes:

```text
prepared detector datasets
raw dataset-collection output
detector training runs
offline evaluation output
temporary runtime JSON state
generated heatmaps and visualizations
thesis-specific analysis notebooks and figures
historical detector-development workspaces
```

This keeps the repository focused on the simulation, reusable source code,
released detector checkpoints, and reproducible training/evaluation tooling.
