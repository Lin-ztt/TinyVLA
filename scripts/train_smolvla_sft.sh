#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-configs/sft/train.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi
ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-}}"
if [[ -z "$ENV_PREFIX" ]]; then
  echo "Set ENV_PREFIX to the LZT_torch environment directory." >&2
  exit 2
fi
TRAIN_BIN="${LEROBOT_TRAIN:-$ENV_PREFIX/bin/lerobot-train}"
[[ -x "$TRAIN_BIN" ]] || { echo "lerobot-train not found: $TRAIN_BIN" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "SFT config not found: $CONFIG" >&2; exit 1; }

HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$TRAIN_BIN" --config_path="$CONFIG" "$@"
