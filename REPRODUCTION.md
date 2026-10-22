# H-JEPA Server Reproduction

Updated: 2026-10-23. This is an independent reproduction of
[kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA), not the authors' repository.
The authors' attribution and MIT license are preserved.

## Which Code Is Published

The model, training entry, data code, configurations, planner, and environments
match upstream commit `97054ecaff83e13839195ec4b92d74c08a0127c5`. On 2026-10-23,
all 169 files in the A100 training source manifest were checked against that
commit byte-for-byte. This is the source used by the latest Cube and Push-T
single-GPU multi-seed experiments, not the earlier DDP implementation.

The previous GitHub revision `6a5679e` had periodic manual garbage collection,
early seeding inside `main_hjepa.py`, and a manual-clipping compatibility helper.
Those changes are no longer in the core entry, matching the latest server source.
Python automatic garbage collection remains enabled; gradient clipping remains
the upstream value of 1.0. This update does not change a running server experiment.

The exact server wrappers are included, without edits, under:

- `reproduction/cube_seeds_20261020/train_entry.py` and `postprocess.py`.
- `reproduction/pusht_seeds_20261020/train_entry.py` and `postprocess.py`.

Their SHA256 values are in `reproduction/SHA256SUMS`. The training wrappers seed
before model construction and record the initial model hash. The Push-T wrapper
also logs scalar memory/GC counters every 50 optimizer steps and asserts the
single-GPU update budget. Neither wrapper calls `gc.collect()` or `empty_cache()`.
They use the original model, losses, queues, optimizer, and dataset code.

The wrappers, rather than a bare `python main_hjepa.py`, reproduce the latest
initialization procedure. They are not identical to the unwrapped upstream entry.
Older results and changes remain documented in
[the historical note](docs/reproduction_20261019.md) and Git history.

## Environment and Data

Use Python 3.11.13 and the repository dependencies. For NJU:

```bash
git clone https://github.com/beck83879-bit/H-JEPA.git
cd H-JEPA
conda create -n hjepa python=3.11.13
conda activate hjepa
python -m pip install -e '.[train,env]' \
  --index-url https://mirrors.nju.edu.cn/pypi/web/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip check
export HJEPA_HOME="$HOME/.hjepa"
```

The CUDA index needs network access or cached wheels. This is not a historical
environment lockfile: lower-bounded dependencies can resolve differently. The
existing A100 environment also had an external checkpoint-saving compatibility
patch; it is not part of these repository files. Exact environment reconstruction
requires the archived environment/setup records in addition to this code.

Follow the original README for downloads, Cube train/validation splitting,
Push-T orientation preparation, and task checksums. Simulation data files are:

| Task | Files under `HJEPA_HOME` |
| --- | --- |
| Push-T | `pusht_expert_train.h5`, `pusht_expert_val.h5` |
| Cube | `cube_single_expert_train.h5`, `cube_single_expert_val.h5` |
| FourRoom | `fourroom_tp35_d1.h5`, `fourroom_tp35_d1_val.h5`, `fourroom_tp35_d1_probing.h5`, `fourroom_tp35_d1_probing_val.h5` |
| AntMaze | `visual_antmaze_medium_explore_stitch_train.h5`, `visual_antmaze_medium_stitch_val.h5`, `visual_antmaze_medium_probing_train.h5`, `visual_antmaze_medium_probing_eval.h5` |

Place official fixed tasks in the task-specific directories under
`h_jepa/assets/eval_trajs/`. Data, weights, credentials, generated videos, and
large experiment logs are not included in this Git repository.

## Latest Single-GPU Run Procedure

Run this from the repository root in the configured environment. Select an
available physical GPU and a new output tag; the example uses GPU0 and seed43.
The training command is long-running and occupies that GPU.

```bash
export HJEPA_REPO="$PWD"
export PYTHONPATH="$HJEPA_REPO:$HJEPA_REPO/h_jepa${PYTHONPATH:+:$PYTHONPATH}"
export HJEPA_TRAIN_SEED=43
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HJEPA_RUN_TAG="manual_$(date +%Y%m%d_%H%M%S)"
export HJEPA_INIT_RECORD="$HJEPA_HOME/reproduction/$HJEPA_RUN_TAG/initialization.json"
mkdir -p "$(dirname "$HJEPA_INIT_RECORD")"
cd "$HJEPA_REPO/h_jepa"

# Push-T L2: 9 epochs, batch128, BF16, 119628 optimizer updates.
export HJEPA_RUN_SUBDIR="pusht/$HJEPA_RUN_TAG/pusht_hjepa_l2/seed$HJEPA_TRAIN_SEED"
python "$HJEPA_REPO/reproduction/pusht_seeds_20261020/train_entry.py" \
  --config-name pusht_hjepa_l2 seed="$HJEPA_TRAIN_SEED" \
  subdir="$HJEPA_RUN_SUBDIR" trainer.devices=1 loader.batch_size=128 \
  planning_eval.enabled=false final_probing_decoding_eval.enabled=false

# Run only after training exits successfully.
export HJEPA_RUN_DIR="$HJEPA_HOME/ckpts/$HJEPA_RUN_SUBDIR"
python "$HJEPA_REPO/reproduction/pusht_seeds_20261020/postprocess.py" \
  inspect "$HJEPA_RUN_DIR" --train-seed "$HJEPA_TRAIN_SEED"
python "$HJEPA_REPO/reproduction/pusht_seeds_20261020/postprocess.py" \
  evaluate "$HJEPA_RUN_DIR" --train-seed "$HJEPA_TRAIN_SEED"
python "$HJEPA_REPO/reproduction/pusht_seeds_20261020/postprocess.py" \
  probe "$HJEPA_RUN_DIR" --train-seed "$HJEPA_TRAIN_SEED"
```

For Cube, use a fresh tag and initialization record, replace the wrapper directory
with `cube_seeds_20261020`, the config with `cube_hjepa_l3`, and the run subdirectory
with `cube/$HJEPA_RUN_TAG/cube_hjepa_l3/seed$HJEPA_TRAIN_SEED`.
Cube uses 8 epochs and 83912 optimizer updates. Both tasks retain the upstream
12 DataLoader workers and prefetch factor 8. Training-time planning/probe callbacks
are disabled because the same stages run separately after training; they must
not be omitted. The probe budget is 25 epochs.

The evaluation wrapper checks task SHA256, budget, horizon, candidate counts,
noise, and per-task metrics. It uses planner seed42, matmul `highest`, CUDA matmul
TF32=false, cuDNN TF32=true, and cuDNN benchmark=false. This is our unified
standalone evaluation setting, not a claim to know every authors' runtime flag.
It refuses to overwrite an existing evaluation/probe directory. `check-eval`
validates the evaluation configuration without running rollouts.

| Setting | Cube L3 | Push-T L2 |
| --- | --- | --- |
| Fixed tasks | 50 | 50 |
| Goal offset / action budget | 20 / 30 | 75 / 90 |
| Level horizons | [2, 2, 1] | [2, 8] |
| Candidate counts | [25, 800, 800] | [50, 400] |
| Optimization steps per level | 90 | 90 |
| Action noise per level | [0, 0.01, 0.01] | [0, 0.01] |

## Latest Results and Limits

| Task | Training seed | Planner seed | Successes | Success rate |
| --- | --- | --- | --- | --- |
| Cube L3 | 43 | 42 | 26/50 | 52% |
| Cube L3 | 44 | 42 | 28/50 | 56% |
| Push-T L2 | 43 | 42 | 18/50 | 36% |
| Push-T L2 | 44 | 42 | 26/50 | 52% |
| Push-T L2 | 45 | 42 | 18/50 | 36% |
| Push-T L2 | 46 | Not evaluated | Training OOM | Not a 0% result |

Cube's two new seeds average 54%, versus the paper's 60.0 +/- 5.0% mean +/- SE.
Push-T's three completed seeds average 41.3 +/- 5.3% mean +/- SE, versus
45.3 +/- 1.3% in the paper. Push-T seed46 failed with CUDA OOM, so this average
conditions on completed runs. Retaining upstream automatic GC does not eliminate
the observed long-run OOM risk. Do not silently reduce the batch or inject manual
GC and then label the experiment as the same run.

These are repeated evaluations of the same 50 fixed tasks. Training seeds differ
from the paper's 42/43/44 set, and planner seed42 is held fixed here rather than
matched to every training seed. This is not the full paper protocol or a controlled
GC ablation. The historical Cube run labeled seed42 used a different initialization
procedure and should not be merged into an unexplained three-seed average.

The previously published 32% single-GPU and 28% DDP Push-T runs used different
source/runtime variants. They are not results of these latest single-GPU wrappers.
