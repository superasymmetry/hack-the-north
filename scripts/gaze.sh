#!/usr/bin/env bash
# Track the eyes and publish where they are looking, for `rocket2.sh --gaze` to read.
#
# Its own virtualenv, deliberately: the tracker needs numpy 2, opencv 5 and mediapipe, and
# the environment that runs Minecraft has numpy 1.26 and opencv 4.8. They must not meet.
# Create it once with scripts/setup/install_gaze.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

GAZE_ENV="$REPO_ROOT/.gaze-env"
if [[ ! -x "$GAZE_ENV/bin/python" ]]; then
    echo "No gaze environment at $GAZE_ENV -- create it with:" >&2
    echo "    bash scripts/setup/install_gaze.sh" >&2
    exit 1
fi

# The repo itself, so the tracker can import mcagents.gaze -- which is stdlib only, and the
# single file these two environments share.
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "$GAZE_ENV/bin/python" \
    -m mcagents.cli.gaze "$@"
