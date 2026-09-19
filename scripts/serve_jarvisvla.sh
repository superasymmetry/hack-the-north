#!/usr/bin/env bash
# Serve JarvisVLA-Qwen2-VL-7B for scripts/jarvisvla.sh. Run this ON THE GPU BOX.
#
#   ./scripts/serve_jarvisvla.sh                                  # one replica, GPU 0, port 8000
#   PORT=8001 GPUS=1 ./scripts/serve_jarvisvla.sh                 # a second one alongside it
#   MODEL=/path/to/local/checkpoint ./scripts/serve_jarvisvla.sh  # a checkpoint you trained
#
# Run it under tmux or sbatch: started from an interactive ssh shell, it dies with the shell.
# First run downloads ~17 GB of weights. Startup takes a few minutes -- wait for
# "Application startup complete" before pointing the agent at it.
set -euo pipefail

MODEL="${MODEL:-CraftJarvis/JarvisVLA-Qwen2-VL-7B}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0}"

CUDA_VISIBLE_DEVICES="$GPUS" vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name jarvisvla \
  --max-model-len 8448 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 5}' &

# --limit-mm-per-prompt is the one flag you cannot drop: vLLM allows ONE image per request by
# default, and the agent sends three (the current frame plus two of history). Without it every
# call 400s once the history fills up -- fine for the first two steps, then dead.
#
# --host 0.0.0.0 matters for the same practical reason: bound to localhost the SSH tunnel has
# nothing to connect to.

cat <<MSG

on the laptop, tunnel to this node's IP (NOT its short hostname -- the login node cannot
resolve those) and point the agent at it:

  ssh -NC -L ${PORT}:$(hostname -I | awk '{print $1}'):${PORT} szeng26@ssh.ccv.brown.edu
  export JARVISVLA_URL=http://127.0.0.1:${PORT}/v1
  ./scripts/jarvisvla.sh

MSG
wait
