#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:?usage: export_model.sh MODEL_DIR OUTPUT_DIR}
OUTPUT=${2:?usage: export_model.sh MODEL_DIR OUTPUT_DIR}
mkdir -p "$OUTPUT"
python -m export.pytorch --model "$MODEL" --output "$OUTPUT/pytorch"
python -m export.onnx --model "$MODEL" --output "$OUTPUT/model.onnx"
python -m export.quantize --input "$OUTPUT/model.onnx" --output "$OUTPUT/model.int8.onnx"
python -m export.validate --model "$MODEL" --onnx "$OUTPUT/model.onnx"
