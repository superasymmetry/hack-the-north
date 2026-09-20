#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-2,3}"
export CUDA_VISIBLE_DEVICES="$GPUS"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8001}"

vllm serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 2007040}'
