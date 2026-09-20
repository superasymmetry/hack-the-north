#!/usr/bin/env bash
# Create .gaze-env and install the eye tracker into it. Run once.
#
# Why a separate environment rather than `pip install gazefollower` into .conda-env: the
# tracker's dependencies are numpy 2, opencv-python 5, opencv-contrib-python and mediapipe.
# The simulator environment runs numpy 1.26 and opencv 4.8, and mcagents/gui.py documents
# how delicate the OpenCV situation already is there. Installing one on top of the other
# breaks the game to gain an eye tracker. They talk through /tmp instead -- see
# mcagents/gaze.py.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GAZE_ENV="$REPO_ROOT/.gaze-env"

if [[ ! -x "$GAZE_ENV/bin/python" ]]; then
    echo "Creating $GAZE_ENV"
    python3 -m venv --copies "$GAZE_ENV"
fi

"$GAZE_ENV/bin/pip" install --upgrade pip
"$GAZE_ENV/bin/pip" install gazefollower

echo
echo "Installed. Check it can see you, and calibrate:"
echo "    ./scripts/gaze.sh --check"
echo
echo "Note: GazeFollower is CC BY-NC-SA 4.0 (non-commercial). See docs/gaze.md."
