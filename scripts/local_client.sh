#!/usr/bin/env bash
# Capture side: microphone -> transcript -> the agent on the GPU box, over one WebSocket.
# AGENT_WS_URL is the Cloudflare tunnel in front of it; AGENT_TOKEN authenticates.
# See docs/local-client.md.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m mcagents.local_client "$@"
