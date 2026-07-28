#!/usr/bin/env bash
set -euo pipefail
: "${GLM_MODEL_PATH:?Set GLM_MODEL_PATH to a trained model directory}"
exec python -m uvicorn backend.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --workers "${WORKERS:-1}"
