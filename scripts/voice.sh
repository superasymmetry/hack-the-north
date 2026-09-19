#!/usr/bin/env bash
# Talk to the LLM. VOICE_LLM_URL points at the chat server (see docs/voice.md).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m mcagents.voice "$@"
