#!/usr/bin/env bash

# interaction model runs on gpus 0 and 1
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0,1}"
export CUDA_VISIBLE_DEVICES="$GPUS"

"$(dirname "$0")/.venv/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size 2 \
  --served-model-name interaction \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.93 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 4}' \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 147456}'
