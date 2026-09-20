#!/usr/bin/env bash
# Track a pinching index finger and publish its screen position for Rocket2.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

GAZE_ENV="$REPO_ROOT/.gaze-env"
if [[ ! -x "$GAZE_ENV/bin/python" ]]; then
    echo "No hand-tracking environment at $GAZE_ENV -- create it with:" >&2
    echo "    bash scripts/setup/install_gaze.sh" >&2
    exit 1
fi

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "$GAZE_ENV/bin/python" \
    -m mcagents.cli.hand "$@"