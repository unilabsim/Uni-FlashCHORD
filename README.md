# Uni-FlashCHORD

**Reference implementation:** [NVIDIA FlashCHORD](https://github.com/nvidia-isaac/video_to_data/tree/main/robotic_grounding/flash_chord) · **Language:** English | [简体中文](README_zh.md)

![Uni-FlashCHORD tissue-box demo](docs/media/tissue_demo.gif)

Uni-FlashCHORD is an independent reproduction of FlashCHORD's dexterous-hand reference retargeting and tracking workflow, built on **UniLab + UniSim Newton**. It loads ManoSharpaData sequences into a floating dual-Sharpa-hand scene, trains reference-tracking policies with FlashSAC, and provides physics replay and checkpoint visualization.

**[Installation](#installation) → [Example data](#example-data) → [Replay](#replay) → [Training](#training) → [Training results](docs/training_results.md) → [Evaluation](#evaluation)**

## Features

- Read parquet references directly and resolve object, support-surface, and collision assets at runtime, including CoACD decomposition.
- Run 4096 Newton environments with 20 Hz control and 100 Hz physics.
- Train with asynchronous Torch FlashSAC, n-step replay, a VOC curriculum, and checkpoint resume.
- Replay in `kinematic`, `voc0`, or `voc1` mode.
- Visualize checkpoints in Viser with deterministic actions and automatic reference-loop resets.

## Installation

Requirements:

- Linux and an NVIDIA CUDA environment supported by Newton;
- Python 3.11–3.13;
- [uv](https://docs.astral.sh/uv/) and Git LFS;
- A 48 GiB-class GPU is recommended for the default training recipe.

```bash
git clone https://github.com/unilabsim/Uni-FlashCHORD.git
cd Uni-FlashCHORD
git lfs pull
uv sync
```

Git LFS downloads the bundled parquet, Sharpa meshes, and other assets. uv installs the released UniLab 1.5.3 and UniSim 2.2.4 dependencies; no additional repository clone is required.

Viser defaults to `http://localhost:8080`. Forward this port over SSH when running remotely. Closing a browser tab does not stop the server: use **Ctrl+C** in the terminal before starting another viewer.

## Example data

The repository includes the complete `tissue_box_simple` reference sequence and its assets. From the repository root:

```bash
PARQUET="$PWD/data/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave"
```

The replay, training, and evaluation commands below reuse this variable. To use another sequence, point it to the directory containing the target parquet:

```bash
PARQUET="/absolute/path/to/robot_name=sharpa_wave"
```

Uni-FlashCHORD consumes **ManoSharpaData parquet files and their URDF, mesh, and support-surface assets**. Asset conversion and CoACD caching happen automatically at runtime; dataset-preprocessing scripts do not need to be copied into this repository.

If you only have raw video, motion capture, or reconstruction output, first use the external `video_to_data/robotic_grounding` pipeline to produce parquet and assets. Data production belongs to that pipeline; Uni-FlashCHORD owns training, replay, and evaluation.

## Replay

```bash
uv run python scripts/replay.py "task.parquet='$PARQUET'" mode=voc1
```

Open `http://localhost:8080` and start playback. Available modes:

- `kinematic`: show recorded states frame by frame without physics integration;
- `voc0`: enable collisions without object VOC wrenches or articulation drive;
- `voc1`: enable the full VOC setup.

Viser provides play, restart, progress, and speed controls. Add `record.output=/tmp/replay.npz` to record public states and controls.

## CUDA MPS setup

The recommended training path uses CUDA MPS to share the GPU between the collector and learner. On a single-GPU Linux host, start the daemon and import its environment before training:

```bash
uv run uni-cumps start --gpus 0 --name chord-mps --daemon
eval "$(uv run uni-cumps env --name chord-mps --shell posix)"
```

The daemon is owned and managed by the user. Stop it after training:

```bash
uv run uni-cumps stop --name chord-mps
```

## Training

```bash
uv run python -m chord_unilab.scripts.train_flashsac \
  task=sharpa_chord/newton \
  "env.scene.reference_path='$PARQUET'" \
  training.no_play=true \
  training.cuda_process_sharing=mps
```

This is the complete **approximately 250M-environment-step** recipe. It defaults to 4096 parallel environments and trains on one reference sequence per run. Change only `PARQUET` to switch sequences; no dataset-specific YAML or Hydra search path is needed.

Checkpoints are saved every 2500 iterations:

```text
logs/flash_sac/SharpaChordFlashSAC/<run>/model_<iteration>.pt
```

## Evaluation

```bash
CHECKPOINT="/absolute/path/to/model.pt"

uv run python scripts/evaluate.py \
  --parquet "$PARQUET" \
  --checkpoint "$CHECKPOINT"
```

Evaluation shows one environment. The policy starts at reference frame 0 with VOC=0 and no settling, then resets automatically and loops at the end of the reference.

Green translucent meshes show the reference hands and object at the current reference frame; the normally rendered meshes show the actual policy simulation. The overlay is read-only and does not affect physics or actions.

- Use Viser to pause or single-step through behavior.
- Use `--port` to change the port.
- Press **Ctrl+C** to exit. A missing checkpoint raises an error; the viewer never falls back to zero actions.

This entry point performs deterministic policy visualization. It does not start training or compute aggregate metrics such as MPPE or ADD-AUC.

## Configuration

| Content | Location |
| --- | --- |
| Scene, observations, actions, rewards, resets, and terminations | `src/chord_unilab/conf/flashsac/task/sharpa_chord/base.yaml` |
| Newton, 4096 environments, and FlashSAC recipe | `src/chord_unilab/conf/flashsac/task/sharpa_chord/newton.yaml` |
| Training, logging, and Viser defaults | `src/chord_unilab/conf/flashsac/config.yaml` |
| Data format and resampling semantics | `src/chord_unilab/data/README.md` |
| Asset resolution, conversion, and caching | `src/chord_unilab/assets/README.md` |

Append `key=value` overrides to the training command, for example:

```bash
algo.seed=123 training.wandb_name=tissue_seed_123
```

## Development

```bash
make check      # Ruff, formatting, mypy, and pyright
make test       # Complete tests; requires Git LFS assets
make test-fast  # Skip real-Newton integration tests
```

Real-Newton tests use the bundled tissue-box data by default. Set `CHORD_TISSUE_PARQUET` to override it.

## License

Apache License 2.0. See `THIRD_PARTY_NOTICES` for third-party code provenance.
