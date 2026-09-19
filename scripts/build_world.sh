#!/usr/bin/env bash
# Generate and cache a world once, so later runs load it (~13s a reset).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m mcagents.cli.build_world "$@"
