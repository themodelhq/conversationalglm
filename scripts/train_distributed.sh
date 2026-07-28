#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-config/train_sft.yaml}
GPUS=${GPUS:-1}
NODES=${NODES:-1}
EXTRA=()
if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then EXTRA+=(--deepspeed "$DEEPSPEED_CONFIG"); fi
if [[ "${FSDP:-0}" == "1" ]]; then EXTRA+=(--fsdp); fi
python -m training.launch --config "$CONFIG" --gpus "$GPUS" --nodes "$NODES" "${EXTRA[@]}"
