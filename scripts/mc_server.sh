#!/usr/bin/env bash
# Hold one Minecraft open for repeated runs to attach to. Leave running.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m mcagents.cli.mc_server "$@"
