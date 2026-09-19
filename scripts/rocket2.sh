#!/usr/bin/env bash
# Run ROCKET-2. MC_PORT=9000 attaches to a scripts/mc_server.sh holder.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m mcagents.cli.rocket2 "$@"
