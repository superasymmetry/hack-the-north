#!/usr/bin/env bash
# Both Cloudflare tunnels + the interaction model. Start the two vLLM servers
# yourself (serve_interaction.sh, serve_vlm_simple.sh) before running this.
#
# cloudflared runs HERE, on the GPU node -- it dials out to Cloudflare's edge
# (7844/tcp), so nothing needs to be open inbound on Oscar or on your laptop.
# Your laptop is just a client of the resulting https/wss hostnames.
#
# TWO tunnels, because they carry traffic in opposite directions:
#   uplink  ($PORT)       laptop -> here: finalized transcripts and agent-view
#                         frames, interleaved; here -> laptop: the spoken reply.
#   goals   ($GOAL_PORT)  here -> laptop: plan entries for the Minecraft policy.
#                         The laptop sends nothing on it.
# Both authenticate with the SAME token in the X-Agent-Token header -- the
# client has no setting for a second one.
#
# interaction_model.py is always given the model names vLLM actually serves.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-9000}"
GOAL_PORT="${GOAL_PORT:-9001}"
# Two vLLM servers: the real-time interaction model (serve_interaction.sh,
# GPUs 0,1) and the background model (serve_vlm_simple.sh, GPUs 2,3).
INTERACT_VLLM_PORT="${INTERACT_VLLM_PORT:-8000}"
BG_VLLM_PORT="${BG_VLLM_PORT:-8001}"
LOG="${LOG:-cloudflared.log}"
GOAL_LOG="${GOAL_LOG:-cloudflared.goals.log}"

[ -s .agent_token ] || { echo "missing .agent_token" >&2; exit 1; }
TOKEN=$(cat .agent_token)

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null' EXIT

served_model() {  # $1 = port -> prints the model id vLLM serves there, fails if it isn't up
  curl -sf "http://localhost:$1/v1/models" |
    python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])'
}

MODEL_ID=$(served_model "$INTERACT_VLLM_PORT") || { echo "no vLLM on :$INTERACT_VLLM_PORT -- start serve_interaction.sh first" >&2; exit 1; }
BG_MODEL_ID=$(served_model "$BG_VLLM_PORT") || { echo "no vLLM on :$BG_VLLM_PORT -- start serve_vlm_simple.sh first" >&2; exit 1; }
echo "vLLM ready: interaction=$MODEL_ID  background=$BG_MODEL_ID"

# Optional third server, for spoken replies only: the FP8 copy from serve_talk.sh,
# usually on another node (serve_talk.sbatch). Unset, replies come from the
# interaction model as before.
TALK_URL="${TALK_URL:-}"
TALK_ARGS=()
if [ -n "$TALK_URL" ]; then
  TALK_MODEL_ID=$(curl -sf "$TALK_URL/models" |
    python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])') ||
    { echo "no vLLM at $TALK_URL -- is serve_talk.sbatch running?" >&2; exit 1; }
  TALK_ARGS=(--talk-url "$TALK_URL" --talk-model "$TALK_MODEL_ID")
  echo "vLLM ready: talk=$TALK_MODEL_ID at $TALK_URL"
fi

# A quick tunnel: no Cloudflare account needed, but a NEW random hostname every
# launch -- so every Slurm restart breaks your laptop's config, twice over now.
# For stable hostnames you need an account plus a domain you control:
#   cloudflared tunnel login
#   cloudflared tunnel create mc-agent
#   cloudflared tunnel route dns mc-agent mc.yourdomain.com
#   cloudflared tunnel run --url http://127.0.0.1:$PORT mc-agent
: > "$LOG"; : > "$GOAL_LOG"
~/bin/cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate >>"$LOG" 2>&1 &
PIDS+=($!)
~/bin/cloudflared tunnel --url "http://127.0.0.1:$GOAL_PORT" --no-autoupdate >>"$GOAL_LOG" 2>&1 &
PIDS+=($!)

wait_for_url() {  # $1 = log file -> prints the hostname it finds
  local url
  for _ in $(seq 1 30); do
    url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$1" | head -1)
    [ -n "$url" ] && { echo "$url"; return 0; }
    sleep 1
  done
  return 1
}

URL=$(wait_for_url "$LOG") || { echo "uplink tunnel failed; see $LOG" >&2; exit 1; }
GOAL_URL=$(wait_for_url "$GOAL_LOG") || { echo "goal tunnel failed; see $GOAL_LOG" >&2; exit 1; }

echo "$URL" > tunnel_url.txt
echo "$GOAL_URL" > goal_tunnel_url.txt
cat <<MSG

  tunnels up -- on your laptop:

AGENT_WS_URL=${URL/https:/wss:} AGENT_TOKEN=$TOKEN AGENT_GOAL_WS_URL=${GOAL_URL/https:/wss:} ./scripts/local_client.sh --check

  or write them into .env (run from the repo root):

touch .env && grep -vE '^(AGENT_WS_URL|AGENT_TOKEN|AGENT_GOAL_WS_URL)=' .env > .env.tmp; printf 'AGENT_WS_URL=%s\nAGENT_TOKEN=%s\nAGENT_GOAL_WS_URL=%s\n' '${URL/https:/wss:}' '$TOKEN' '${GOAL_URL/https:/wss:}' >> .env.tmp && mv .env.tmp .env

MSG

# Foreground: Ctrl-C here tears down the tunnels.
./.venv/bin/python interaction_model.py --listen "$PORT" --goal-port "$GOAL_PORT" \
  --url "http://localhost:$INTERACT_VLLM_PORT/v1" --model "$MODEL_ID" \
  --background-url "http://localhost:$BG_VLLM_PORT/v1" --background-model "$BG_MODEL_ID" \
  --token "$TOKEN" \
  "${TALK_ARGS[@]}"
