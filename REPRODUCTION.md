# Local H-JEPA Reproduction

Updated: 2026-10-19.

This repository preserves the authors' code, attribution, and MIT license from
[kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA), based on commit
`97054ecaff83e13839195ec4b92d74c08a0127c5`.
It is an independent reproduction copy, not the authors' official repository.

## Changes Included Here

The local changes in `h_jepa/main_hjepa.py` are:

- Call `pl.seed_everything(cfg.seed, workers=True)` before constructing the model.
  The upstream Manager seeds later; a command-line seed alone therefore does not
  establish the same initialization as the authors' historical training run.
- Collect unreachable Python objects every 25 training/validation batches and at
  the end of each training epoch. This addresses accumulated reference cycles
  without reducing the configured batch size, model depth, or training budget.
- Transfer gradient-clipping settings to the fields consumed by
  stable-pretraining 0.1.7 in manual optimization, and disable Lightning's
  incompatible automatic clipping field. The intended clipping value is retained.

Training and evaluation YAML files, model architecture, losses, planner, and
environment success criteria are unchanged from the upstream base in this copy.
These compatibility changes have not been shown to reproduce identical numerical
training trajectories across dependency versions or GPUs.

## Code Provenance

This branch contains the local checkout and its compatibility changes. The A100
experiments summarized below used a separately archived server source snapshot.
That snapshot also included runtime monitoring, DDP-related synchronization, and
contiguous latent-queue storage changes. Its world size was one, so DDP branches
were inactive. It did not use this checkout's manual-clipping compatibility helper.

Consequently, the results below are experiment records, not a claim that this
exact branch produced those checkpoints. The archived server source and full
experiment artifacts are not bundled in this repository. Do not infer exact
historical reproducibility from matching config names alone.

## Setup and Data

Use Python 3.11.13 and the dependencies in `pyproject.toml`. For the NJU environment,
use the NJU PyPI mirror, retaining the upstream CUDA wheel index:

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
mkdir -p "$HJEPA_HOME"
```

The extra CUDA index requires network access or an available wheel cache. Resolve
dependency-check failures before training; do not silently replace the pinned
PyTorch version. Some upstream dependencies have lower bounds only, so this install
command is not a complete lockfile for historical experiments.

Follow README sections 3.1 and 3.2 for downloads, Cube splitting, Push-T orientation
preparation, and official task checksums. Required simulation training files are:

| Task | Files under `HJEPA_HOME` |
| --- | --- |
| Push-T | `pusht_expert_train.h5`, `pusht_expert_val.h5` |
| Cube | `cube_single_expert_train.h5`, `cube_single_expert_val.h5` |
| FourRoom | `fourroom_tp35_d1.h5`, `fourroom_tp35_d1_val.h5`, `fourroom_tp35_d1_probing.h5`, `fourroom_tp35_d1_probing_val.h5` |
| AntMaze | `visual_antmaze_medium_explore_stitch_train.h5`, `visual_antmaze_medium_stitch_val.h5`, `visual_antmaze_medium_probing_train.h5`, `visual_antmaze_medium_probing_eval.h5` |

Official fixed evaluation tasks belong under `h_jepa/assets/eval_trajs/`, with
the task-specific subdirectories documented in the upstream README. Datasets,
checkpoints, downloaded tasks, credentials, and local run outputs are not uploaded.

## Training and Evaluation

Run from `h_jepa/`. Assign an available GPU for each command:

```bash
cd h_jepa
CUDA_VISIBLE_DEVICES=0 python main_hjepa.py --config-name fourroom_hjepa_l3 seed=42
CUDA_VISIBLE_DEVICES=0 python main_hjepa.py --config-name cube_hjepa_l3 seed=42
CUDA_VISIBLE_DEVICES=0 python main_hjepa.py --config-name pusht_hjepa_l2 seed=42
CUDA_VISIBLE_DEVICES=0 python main_hjepa.py --config-name ant_hjepa_l3 seed=42
```

These commands run sequentially when pasted together. Training performs final
planning evaluation automatically. For a separate Cube evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config-name cube_l3 seed=42 \
  policy="$HJEPA_HOME/ckpts/cube/cube_hjepa_l3/seed42/cube_hjepa_l3_object.ckpt" \
  output.dir=eval_cube_l3_seed42_review
```

Use a new evaluation directory when changing planner settings: completed tasks are
skipped on rerun. The paper reports model seeds 42, 43, and 44 with matching planner
seeds. One trained seed evaluated on 50 tasks is not the full three-seed protocol.
The complete upstream simulation matrix contains 84 training runs; the four
commands above cover selected hierarchical configurations only.

## Recorded Results and Limits

A100 experiments used training seed 42, base planner seed 42, and the 50 official
fixed tasks. Official-checkpoint controls used the same evaluation environment.

| Task | Our A100 checkpoint | Official checkpoint control | Paper mean +/- SE |
| --- | --- | --- | --- |
| FourRoom L3 | 98% (49/50) | Not measured | 96.0 +/- 1.1% |
| Cube L3 | 44% (22/50) | 58% (29/50) | 60.0 +/- 5.0% |
| Push-T L2 | 32% (16/50) | 46% (23/50) | 45.3 +/- 1.3% |

The controls narrow the Cube/Push-T gap to differences in the learned checkpoint,
including weights, BatchNorm statistics, and latent-action queues. They do not
isolate one faulty component or prove a particular training change caused the gap.
Matching tensor shapes, action-normalization values, and optimizer update counts
does not establish identical training histories. AntMaze is omitted because this
record does not establish its final result.

Observed failures include Cube never lifting the object sufficiently, and Push-T
failing the joint agent/block position criterion despite a small angular error.
Cube success checks 3D object position within 4 cm. Push-T requires joint position
error below 20 pixels and block angle error below 20 degrees. These observations
are not proof that orientation probing or the memory-management changes caused
the performance difference.

Full multi-seed reproduction and controlled diagnosis of the learned dynamics,
normalization statistics, and model-specific action queues remain outstanding.
