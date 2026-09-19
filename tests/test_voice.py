"""The voice module's LLM leg, against a stub server. No microphone, no vLLM, no GPU.

    python tests/test_voice.py

What is worth testing here is the half that is easy to get silently wrong: that the client
actually *streams* (a reply that arrives in one lump has thrown away the latency the fast
ASR bought, and looks identical from the outside), and that history stays bounded. The
microphone half is Moonshine's, and needs a person to talk into it.
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, ".")

import contextlib
import io

from mcagents.voice import LiveLine, Responder, VoiceConfig

TOKENS = ["I ", "see ", "an ", "oak ", "log."]
GAP = 0.05                                   # seconds the stub waits between tokens


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1, so the reply is chunked the way vLLM's is. Under HTTP/1.0 the client cannot
    # see a chunk boundary and buffers the whole body -- which makes a streaming client look
    # exactly like a non-streaming one, and this test pass for the wrong reason.
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def handle_one_request(self):
        # A pooled client dropping a keep-alive connection is normal, and socketserver
        # prints a traceback for it. Nothing here outlives the test; absorb it.
        try:
            super().handle_one_request()
        except ConnectionResetError:
            self.close_connection = True

    def do_GET(self):
        body = json.dumps({"data": [{"id": "stub-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(request)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for token in TOKENS:
            self._chunk(f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n")
            time.sleep(GAP)
        self._chunk("data: [DONE]\n\n")
        self._chunk("")

    def _chunk(self, text: str) -> None:
        body = text.encode()
        self.wfile.write(f"{len(body):X}\r\n".encode() + body + b"\r\n")
        self.wfile.flush()


def serve():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def test_streams_incrementally(url):
    """Tokens must arrive as they are produced, not all at the end."""
    responder = Responder(VoiceConfig(llm_url=url))
    assert responder.model == "stub-model", responder.model      # discovered, not guessed

    started = time.time()
    arrivals = [(token, time.time() - started) for token in responder.ask("hello")]

    assert [t for t, _ in arrivals] == TOKENS, arrivals
    first = arrivals[0][1]
    last = arrivals[-1][1]
    # The first token must beat the last by most of the stub's own cadence. Buffered, they
    # would land within a millisecond of each other.
    assert last - first > 3 * GAP, f"not streaming: first {first:.3f}s, last {last:.3f}s"
    print(f"  streams: first token {first * 1000:.0f} ms, last {last * 1000:.0f} ms")


def test_history_is_bounded(url):
    config = VoiceConfig(llm_url=url, history_turns=2)
    responder = Responder(config)
    for index in range(5):
        assert "".join(responder.ask(f"turn {index}")) == "".join(TOKENS)

    assert len(responder.history) == 4, responder.history        # 2 turns = 4 messages
    assert responder.history[0]["content"] == "turn 3", responder.history[0]
    print(f"  history: {len(responder.history)} messages after 5 turns")


def test_sends_system_and_history(url, server):
    responder = Responder(VoiceConfig(llm_url=url))
    "".join(responder.ask("first"))
    "".join(responder.ask("second"))

    sent = server.requests[-1]
    assert sent["stream"] is True, sent
    roles = [message["role"] for message in sent["messages"]]
    assert roles == ["system", "user", "assistant", "user"], roles
    assert sent["messages"][-1]["content"] == "second"
    print(f"  request: {roles}")


def screen(fn) -> str:
    """What fn leaves on a terminal, applying \\r-overwrite the way one would."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn()
    text, cursor = "", 0
    for character in buffer.getvalue():
        if character == "\r":
            cursor = 0
        else:
            text = text[:cursor] + character + text[cursor + 1:]
            cursor += 1
    return text


def test_live_line_leaves_no_debris():
    """A shorter revision must paint over the longer one it replaces, not sit inside it."""
    live = LiveLine()
    assert screen(lambda: live.show("mine the die")).strip() == "... mine the die"
    assert screen(lambda: live.show("mine the diamond ore")).strip() == "... mine the diamond ore"
    # The failure this guards: '... hi' followed by the tail of the previous phrase.
    assert screen(lambda: live.show("hi")).strip() == "... hi"
    assert screen(live.clear).strip() == ""
    print("  live line: revises in place, shrinks without debris")


def test_live_line_yields_the_terminal():
    """Partials arrive on Moonshine's thread; they must not redraw through a reply."""
    live = LiveLine()
    screen(lambda: live.show("mine the dia"))   # captured, not on this terminal

    def reply():
        with live.pause():
            live.show("SHOULD NOT APPEAR")
            print("  llm: an oak log.")

    assert screen(reply) == "  llm: an oak log.\n", screen(reply)
    print("  live line: held back while the reply prints")


def main():
    server, url = serve()
    print(f"stub server on {url}")
    test_live_line_leaves_no_debris()
    test_live_line_yields_the_terminal()
    test_streams_incrementally(url)
    test_history_is_bounded(url)
    server.requests.clear()
    test_sends_system_and_history(url, server)
    print("ok")


if __name__ == "__main__":
    main()
