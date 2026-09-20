"""Laptop side: stream transcribed speech to the interaction model on Oscar.

Reads typed lines today; swap `lines()` for a streaming ASR source (faster-whisper,
Parakeet) that yields partials and finals, and nothing else here changes.

    pip install websockets
    python local_client.py --url wss://<host>.trycloudflare.com --token <token>
"""

import argparse
import json
import sys
from websockets.sync.client import connect


def lines():
    """Stand-in for streaming ASR: yields (text, is_final)."""
    for line in sys.stdin:
        if line.strip():
            yield line.strip(), True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", required=True)
    args = ap.parse_args()

    # One long-lived connection: a TLS handshake per utterance would cost more
    # than the model's whole turn.
    with connect(args.url, additional_headers={"X-Agent-Token": args.token}) as ws:
        print(f"connected to {args.url}", file=sys.stderr)
        for text, final in lines():
            ws.send(json.dumps({"text": text, "final": final}))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
