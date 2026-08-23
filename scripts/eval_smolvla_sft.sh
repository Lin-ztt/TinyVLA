#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 POLICY_PATH [RUN_NAME]" >&2
  exit 2
fi

POLICY_PATH=$1
RUN_NAME=${2:-sft_eval}
CONFIG="${CONFIG:-configs/sft/eval.yaml}"
SUITE="${SUITE:-libero_goal}"
ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-}}"
if [[ -z "$ENV_PREFIX" ]]; then
  echo "Set ENV_PREFIX to the LZT_torch environment directory." >&2
  exit 2
fi
EVAL_BIN="${LEROBOT_EVAL:-$ENV_PREFIX/bin/lerobot-eval}"
[[ -x "$EVAL_BIN" ]] || { echo "lerobot-eval not found: $EVAL_BIN" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "SFT eval config not found: $CONFIG" >&2; exit 1; }
[[ -e "$POLICY_PATH" ]] || { echo "Policy path not found: $POLICY_PATH" >&2; exit 1; }

OUTPUT_DIR="outputs/runs/sft/$RUN_NAME/$SUITE"
mkdir -p outputs/runs/logs
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PWD/outputs/runs/sft/smoke/libero_config}" \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
  "$EVAL_BIN" \
  --config_path="$CONFIG" \
  --env.task="$SUITE" \
  --policy.path="$POLICY_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="${RUN_NAME}_${SUITE}"
