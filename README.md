# TinyVLA

**SmolVLA on LIBERO with supervised and reinforcement-learning post-training**

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![LeRobot](https://img.shields.io/badge/LeRobot-0.6.1-1f6feb.svg)](https://github.com/huggingface/lerobot)
[![Benchmark](https://img.shields.io/badge/Benchmark-LIBERO-2ea44f.svg)](https://github.com/Lifelong-Robot-Learning/LIBERO)

TinyVLA is a lightweight engineering project for experiments with embodied vision-language-action (VLA) models. It uses LeRobot, SmolVLA, and LIBERO to provide a single workflow for base-policy inference, supervised fine-tuning (SFT), reinforcement-learning post-training, closed-loop evaluation, and rollout video recording.

The project provides two reinforcement-learning integrations:

- **DSRL**: SAC selects structured initial noise in SmolVLA's Flow Matching space. The frozen SmolVLA converts the selected noise into an action chunk.
- **EXPO**: an edit policy generates action residuals from the observation and a base action chunk. A Critic selects between base and edited candidates.

TinyVLA brings together the configurations and scripts needed to run these workflows and compare the two RL interfaces in LIBERO.

## Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Prepare Data and Models](#prepare-data-and-models)
- [SFT](#sft)
- [DSRL](#dsrl)
- [EXPO](#expo)
- [Training Runtime](#training-runtime)
- [Evaluation and Videos](#evaluation-and-videos)
- [Results](#results)
- [References](#references)
- [License](#license)

## Project Structure

```text
TinyVLA/
├── src/tinyvla/
│   ├── libero.py                  Shared LIBERO and SmolVLA initialization
│   ├── dsrl/                      DSRL SAC, Replay Buffer, and rollout
│   └── expo/                      EXPO edit policy, Critic, and rollout
├── scripts/
│   ├── train_smolvla_sft.sh       SFT training
│   ├── eval_smolvla_sft.sh        SFT evaluation
│   ├── train_smolvla_dsrl.py      DSRL training
│   ├── eval_smolvla_dsrl.py       DSRL evaluation
│   ├── train_smolvla_expo.py      EXPO training
│   ├── eval_smolvla_expo.py       EXPO evaluation
│   ├── record_rollout.py          Unified rollout video entry point
│   └── tools/                     Environment, data, and showcase tools
├── configs/
│   ├── sft/                       Stable SFT configurations
│   ├── dsrl/                      Stable DSRL configurations
│   ├── expo/                      Stable EXPO configurations
│   └── tasks/                     Task configurations
├── models/                        Local model weights
├── data/                          Local datasets
├── outputs/runs/                  Training, evaluation, and video artifacts
├── upstream.lock                  Pinned dependency and model revisions
├── LICENSE                        Apache-2.0 license
└── README.md
```

## Installation

Create the project environment and install versions compatible with the entries in `upstream.lock`:

```bash
conda create -n tinyvla python=3.12 -y
conda activate tinyvla

pip install --upgrade pip
pip install -e .
```

For headless LIBERO rendering, set the MuJoCo backend after activating the environment:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

If Hugging Face is not reachable from your network, set an accessible mirror before downloading:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Prepare Data and Models

Install the Hugging Face CLI and download the base model, the LIBERO fine-tuned model, and the dataset. The revisions below match `upstream.lock`:

```bash
pip install --upgrade huggingface_hub

hf download lerobot/smolvla_base \
  --repo-type model \
  --revision c83c3163b8ca9b7e67c509fffd9121e66cb96205 \
  --local-dir models/upstream/smolvla_base

hf download lerobot/smolvla_libero \
  --repo-type model \
  --revision 31d453f7edd78c839a8bbc39744a292686daf0de \
  --local-dir models/sft/libero_40tasks

hf download lerobot/libero \
  --repo-type dataset \
  --revision a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --local-dir data/libero
```

Before the first rollout, check the installed packages, CUDA, and runtime information:

```bash
python scripts/tools/check_env.py
```

Prepare the LIBERO scene assets and run a short environment test:

```bash
python scripts/tools/smoke_libero.py \
  --suite libero_goal \
  --task-id 0 \
  --steps 5 \
  --output-dir outputs/runs/sft/smoke \
  --assets-dir assets/libero
```

## SFT

The default SFT configuration is [`configs/sft/train.yaml`](configs/sft/train.yaml). Its output is written to `outputs/runs/sft/libero_40tasks/train`.

Start training with the environment that contains `lerobot-train`:

```bash
scripts/train_smolvla_sft.sh configs/sft/train.yaml
```

Evaluate a trained policy on one LIBERO suite:

```bash
scripts/eval_smolvla_sft.sh models/sft/libero_40tasks sft_baseline
```

## DSRL

DSRL freezes SmolVLA and trains only SAC. SAC outputs a 32-dimensional structured noise vector, which is expanded along the action-chunk time dimension before being passed to SmolVLA. The stable training and evaluation configurations are:

- [`configs/dsrl/train.yaml`](configs/dsrl/train.yaml)
- [`configs/dsrl/eval.yaml`](configs/dsrl/eval.yaml)

Start training:

```bash
python scripts/train_smolvla_dsrl.py \
  --config configs/dsrl/train.yaml
```

Evaluate the resulting learner:

```bash
python scripts/eval_smolvla_dsrl.py \
  --config configs/dsrl/eval.yaml
```

## EXPO

EXPO freezes SmolVLA and applies an edit policy to the beginning of each base action chunk. The edit policy generates element-wise action residuals; the training loop optimizes the edit policy, Critic, and temperature, while candidate actions are selected by the Critic.

The stable configurations are:

- [`configs/expo/train.yaml`](configs/expo/train.yaml)
- [`configs/expo/eval.yaml`](configs/expo/eval.yaml)

Start training:

```bash
python scripts/train_smolvla_expo.py \
  --config configs/expo/train.yaml
```

Evaluation requires the learner checkpoint produced by training:

```bash
python scripts/eval_smolvla_expo.py \
  --config configs/expo/eval.yaml \
  --learner-checkpoint \
  outputs/runs/expo/libero_spatial/task_04/baseline/seed_3000/train/checkpoint_latest.pt
```

## Training Runtime

The following figures are practical references for the default configurations. Actual usage depends on the number of cameras, candidate count, storage speed, and whether multiple seeds share one GPU.

| Method | Training scale                 | RTX 3090         | RTX 4090 Plus      |
| ------ | ------------------------------ | ---------------- | ------------------ |
| SFT    | 25k frozen-vision steps        | ~8 GB, 7–8 h    | ~8 GB, 4.5–6 h    |
| DSRL   | 10k transitions / 190k updates | ~2–4 GB, 6–7 h | ~2–4 GB, 3.5–5 h |
| EXPO   | 6k transitions / 20k updates   | ~4–8 GB, 2–4 h | ~4–8 GB, 1–2.5 h |

End-to-end SFT, which also updates the visual encoder, needs more memory than the frozen-vision configuration. Reserve approximately 16–24 GB on an RTX 3090 and 24–32 GB on an RTX 4090 Plus. For a new setup, run a short warm-up first to measure the actual peak memory and throughput before starting a long job.

## Evaluation and Videos

### Unified Rollout Entry Point

[`scripts/record_rollout.py`](scripts/record_rollout.py) supports four methods: `base`, `sft`, `dsrl`, and `expo`. Each run writes a front-view video, a first-frame PNG, and JSON metadata.

Base model:

```bash
python scripts/record_rollout.py \
  --method base \
  --checkpoint models/upstream/smolvla_base \
  --suite libero_spatial \
  --task-id 0 \
  --output outputs/runs/showcase/base/task_00/episode.mp4
```

SFT model:

```bash
python scripts/record_rollout.py \
  --method sft \
  --checkpoint models/sft/libero_40tasks \
  --suite libero_spatial \
  --task-id 0 \
  --output outputs/runs/showcase/sft/task_00/episode.mp4
```

DSRL and EXPO require their evaluation configuration and learner checkpoint:

```bash
python scripts/record_rollout.py \
  --method dsrl \
  --checkpoint models/sft/libero_40tasks \
  --config configs/dsrl/eval.yaml \
  --actor-checkpoint outputs/runs/dsrl/libero_spatial/task_04/baseline/seed_1000/train/checkpoint_latest.pt \
  --suite libero_spatial --task-id 4 \
  --output outputs/runs/showcase/dsrl/task_04/episode.mp4
```

```bash
python scripts/record_rollout.py \
  --method expo \
  --checkpoint models/sft/libero_40tasks \
  --config configs/expo/eval.yaml \
  --learner-checkpoint outputs/runs/expo/libero_spatial/task_04/baseline/seed_3000/train/checkpoint_latest.pt \
  --suite libero_spatial --task-id 4 \
  --output outputs/runs/showcase/expo/task_04/episode.mp4
```

The default video resolution is `1280x720`; use `--width`, `--height`, and `--fps` to change it.

### Output Files

Training and evaluation outputs are organized under `outputs/runs/<method>/`. Common files include:

- `training.json`: training progress, update metrics, and final state;
- `evaluation.json`: success rate, episode statistics, and candidate diagnostics;
- `checkpoint_latest.pt`: the latest resumable training state;
- `videos/*.mp4`, `*.png`, and `*.json`: rollout videos, first frames, and metadata.

Models, datasets, checkpoints, videos, and run logs are stored as local experiment artifacts and ignored by `.gitignore`.

## Results

The following representative results summarize the implemented workflows. The methods use different tasks, numbers of initial states, and evaluation protocols, so the values should be read together with their evaluation setup.

| Method | Evaluation Setup                                                    | Success rate |
| ------ | ------------------------------------------------------------------- | -----------: |
| SFT    | Four LIBERO suites, 40 tasks, 400 episodes                          |       68.25% |
| DSRL   | One task post-training, 50 fixed initial states, best checkpoint    |          92% |
| EXPO   | One task post training, 20 episodes, best Critic-warm-up checkpoint |         100% |

## References

[1] [Hugging Face, LeRobot, GitHub repository.](https://github.com/huggingface/lerobot)

[2] [M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, S. Alibert, M. Cord, T. Wolf, and R. Cadene, SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics, arXiv preprint arXiv:2506.01844, 2025.](https://arxiv.org/abs/2506.01844)

[3] [Hugging Face, SmolVLA Base, Hugging Face model repository, accessed Aug. 23, 2026.](https://huggingface.co/lerobot/smolvla_base)

[4] [Hugging Face, LIBERO Dataset, Hugging Face dataset repository, accessed Aug. 23, 2026.](https://huggingface.co/datasets/lerobot/libero)

[5] [Lifelong Robot Learning, LIBERO: Benchmark for lifelong robot learning, GitHub repository.](https://github.com/Lifelong-Robot-Learning/LIBERO)

[6] [A. Wagenmaker, M. Nakamoto, Y. Zhang, S. Park, W. Yagoub, A. Nagabandi, A. Gupta, and S. Levine, DSRL: Steering Your Diffusion Policy with Latent Space Reinforcement Learning, arXiv preprint arXiv:2506.15799, 2025.](https://arxiv.org/abs/2506.15799)

[7] [Nakamotoo, DSRL for $\pi_0$, GitHub repository.](https://github.com/nakamotoo/dsrl_pi0)

[8] [P. Dong, K.-H. Hung, T. Gao, D. Sadigh, and C. Finn, EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models, arXiv preprint arXiv:2605.25477, 2026.](https://arxiv.org/abs/2605.25477)

[9] [P. D. Perry, EXPO-FT, GitHub repository.](https://github.com/pd-perry/expo-ft)

## License

This project is licensed under the [Apache License 2.0](LICENSE). Third-party models, datasets, and simulation assets remain subject to their respective licenses and terms of use.
