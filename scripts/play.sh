#!/usr/bin/env bash
# Play the simulator yourself.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m mcagents.cli.play "$@"
