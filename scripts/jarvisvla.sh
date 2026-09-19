#!/usr/bin/env bash
# Run JarvisVLA. MC_PORT=9000 attaches to a scripts/mc_server.sh holder.
#
# The policy is NOT local: it is a 7B VLM on the GPU box (scripts/serve_jarvisvla.sh) reached
# over an SSH tunnel. JARVISVLA_URL defaults to the near end of the usual tunnel
# (`./scripts/connect.sh <node ip>`, i.e. `ssh -NC -L 8000:<node ip>:8000 <user>@…`), so it is
# right whenever that tunnel is up on the standard port. See docs/jarvisvla.md.
#
# A rollout is one round trip per env step and the frames are most of the bytes, so the
# tunnel -- not the GPU -- sets the step rate. JARVISVLA_JPEG_QUALITY and JARVISVLA_HISTORY
# are the two knobs that move it; docs/jarvisvla.md has the measured table.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

export JARVISVLA_URL="${JARVISVLA_URL:-http://127.0.0.1:8000/v1}"
python -m mcagents.cli.jarvisvla "$@"
