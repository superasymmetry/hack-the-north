#!/usr/bin/env bash
# Tunnel to the vLLM server on an Oscar compute node. Run this ON THE LAPTOP.
#
#   ./scripts/connect.sh 172.20.237.6        # node IP from `hostname -I` on the GPU box
#   NODE_IP=172.20.237.6 ./scripts/connect.sh
#   COMPRESS=0 ./scripts/connect.sh 172.20.237.6   # no -C, for a fast link
#
# -C is on because of what goes up this tunnel: base64-encoded JPEG, which is a JPEG (already
# incompressible) inflated 33% by base64 (entirely compressible). gzip takes a 130 KB request
# back to 97 KB -- 75%, i.e. it recovers the base64 overhead and nothing else. That is ~25% off
# the step time on a link where the bytes dominate, and JarvisVLA's do. On a fast link the
# compression CPU is the larger cost, hence COMPRESS=0.
#
# The IP, not the short hostname: `-L 8000:gpu4104:8000` fails with "Name or service not
# known" because the login node cannot resolve Oscar's compute-node names, and the failed
# channel shows up on this end as "Connection reset by peer". The node changes every
# allocation, so re-read `hostname -I` each time -- serve_jarvisvla.sh prints the command.
#
# Printing nothing after "Success. Logging you in..." is the tunnel working, not hanging.
# Leave it running and use a second terminal.
set -euo pipefail

NODE_IP="${1:-${NODE_IP:-}}"
PORT="${PORT:-8000}"
USER_HOST="${USER_HOST:-szeng26@ssh.ccv.brown.edu}"
COMPRESS="${COMPRESS:-1}"
[[ "$COMPRESS" == "0" ]] && COMPRESS_FLAG=() || COMPRESS_FLAG=(-C)

if [[ -z "$NODE_IP" ]]; then
  echo "usage: $0 <node-ip>   (run 'hostname -I' on the GPU box; use the first 172.20.x address)" >&2
  exit 1
fi

echo "tunnelling 127.0.0.1:${PORT} -> ${NODE_IP}:${PORT} via ${USER_HOST}" >&2
exec ssh -N "${COMPRESS_FLAG[@]}" -L "${PORT}:${NODE_IP}:${PORT}" "$USER_HOST"
