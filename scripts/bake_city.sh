#!/usr/bin/env bash
# Build the city into a world zip once, so runs load it as terrain (~20s off every reset).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m mcagents.cli.bake_city "$@"
