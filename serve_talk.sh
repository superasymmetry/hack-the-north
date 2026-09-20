#!/usr/bin/env bash
# The talking model: serve_interaction.sh's weights quantized to FP8 at load, for
# spoken replies only (interaction_model.py --talk-url). Gates, plans and
# grounding stay on serve_interaction.sh at full precision -- they become actions.
#
# At batch 1, decoding is bound by reading the weights once per token, and FP8
# halves that, so expect up to ~2x tokens/s over bf16 on the same GPUs (UNMEASURED
# here). L40S (Ada) runs FP8 natively; on Ampere vLLM falls back to weight-only.
#
# Needs 2 GPUs of its own: ~17 GB of weights each at TP=2. The main job's 4 are
# already full, so run it from serve_talk.sbatch.
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8002}"
GPUS="${GPUS:-0,1}"
export CUDA_VISIBLE_DEVICES="$GPUS"

# --mm-processor-kwargs must match serve_interaction.sh, so a reply sees the frame
# at the same resolution the plan does. --host: the caller may be on another node.
"$(dirname "$0")/.venv/bin/vllm" serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 2 \
  --quantization fp8 \
  --served-model-name talk \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 147456}'
