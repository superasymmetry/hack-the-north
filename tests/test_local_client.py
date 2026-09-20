"""The socket half of local_client, against a stub server. No microphone, no tunnel, no GPU.

    python tests/test_local_client.py

What is worth testing here is everything that only shows up when the network misbehaves,
which is exactly what a demo will not do until it matters:

* a final survives the socket dying underneath it and arrives on the next connection --
  the Slurm job going away mid-session is expected, not exceptional;
* a wrong token *stops*, rather than retrying forever against a server that will refuse
  identically every time and look like a network fault while doing it;
* partials are superseded rather than queued, so backpressure delivers the newest text
  instead of a backlog of stale text;
* frames go up as `{"kind": "frame", "image": ...}`, only while something is publishing
  them, and never twice -- an agent that has stopped must go quiet rather than narrate a
  session that is over.

The microphone half is Moonshine's and needs a person to talk into it.
"""
import asyncio
import base64
import contextlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, ".")

from websockets.asyncio.server import serve

from mcagents import frames
from mcagents.goals import GoalSpool, StatusChannel
from mcagents.local_client import (ClientConfig, Cue, Frame, FrameCadence, MicSource, Outbox, Speaker,
                                   TokenRejected, Utterance, describe_goal, frame_pump,
                                   describe_status, format_reply, parse_goals, parse_reply,
                                   parse_turn_command, probe, run_goal_link, run_link,
                                   describe_selection, shared_mute, starts_a_task)

TOKEN = "correct-horse"


class Stub:
    """A server that checks the token the way the real one does, and records what arrives.

    `drop_first` closes the first connection without a close frame, which is what a process
    disappearing under Slurm actually looks like from out here -- not a polite goodbye.
    """

    def __init__(self, token=TOKEN, drop_first=False, sends=()):
        self.token = token
        self.drop_first = drop_first
        self.sends = list(sends)    # pushed down to the client as soon as it connects
        self.messages = []          # every JSON payload received, in order
        self.connections = 0

    async def handle(self, socket):
        self.connections += 1
        if socket.request.headers.get("X-Agent-Token") != self.token:
            await socket.close(1008, "bad token")
            return
        if self.drop_first and self.connections == 1:
            socket.transport.abort()        # no close frame: the process simply vanished
            return
        for payload in self.sends:
            await socket.send(payload if isinstance(payload, str) else json.dumps(payload))
        try:
            async for raw in socket:
                self.messages.append(json.loads(raw))
        except Exception:
            pass


def config_for(port, **overrides):
    config = ClientConfig(url=f"ws://127.0.0.1:{port}", token=TOKEN,
                          backoff_initial=0.05, backoff_max=0.2, open_timeout=5.0)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


async def until(predicate, timeout=10.0):
    """Poll rather than sleep a fixed amount: a fixed sleep is either slow or flaky."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


async def with_link(config, outbox, body, sent=None):
    """Run the link as a task for the duration of `body`, then cancel it."""
    task = asyncio.create_task(
        run_link(config, outbox, sent or (lambda utterance: None), lambda message: None))
    try:
        return await body()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def test_partials_and_finals_reach_the_server():
    stub = Stub()
    async with serve(stub.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        outbox = Outbox(asyncio.get_running_loop())

        async def body():
            outbox.put(Utterance("mine the dia", final=False))
            assert await until(lambda: len(stub.messages) == 1)
            outbox.put(Utterance("mine the diamond ore", final=True))
            assert await until(lambda: len(stub.messages) == 2)

        await with_link(config_for(port), outbox, body)

    assert stub.messages[0] == {"text": "mine the dia", "final": False}, stub.messages
    assert stub.messages[1] == {"text": "mine the diamond ore", "final": True}, stub.messages
    print("  wire: partials go as final=false, finals as final=true")


async def test_one_connection_for_many_utterances():
    """A connection per utterance would cost a TLS handshake per transcript delta."""
    stub = Stub()
    async with serve(stub.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        outbox = Outbox(asyncio.get_running_loop())

        async def body():
            for index in range(6):
                outbox.put(Utterance(f"line {index}", final=True))
                assert await until(lambda i=index: len(stub.messages) == i + 1)

        await with_link(config_for(port), outbox, body)

    assert stub.connections == 1, f"opened {stub.connections} connections, not 1"
    print("  wire: six utterances over one connection")


async def test_a_final_survives_the_socket_dying():
    stub = Stub(drop_first=True)
    async with serve(stub.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        outbox = Outbox(asyncio.get_running_loop())

        async def body():
            # Queued while the first connection is being torn down underneath it.
            outbox.put(Utterance("chop the oak log", final=True))
            assert await until(lambda: stub.messages), "final was lost across the reconnect"
            assert await until(lambda: stub.connections >= 2)

        await with_link(config_for(port), outbox, body)

    assert stub.messages[0] == {"text": "chop the oak log", "final": True}, stub.messages
    print(f"  reconnect: final delivered on connection {stub.connections}, not dropped")


async def test_a_wrong_token_stops_instead_of_retrying():
    stub = Stub(token="something-else")
    async with serve(stub.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        outbox = Outbox(asyncio.get_running_loop())
        raised = None
        try:
            await asyncio.wait_for(
                run_link(config_for(port), outbox, lambda u: None, lambda m: None),
                timeout=10)
        except TokenRejected as exc:
            raised = exc
        except asyncio.TimeoutError:
            raised = None

    assert raised is not None, "a 1008 close was retried instead of surfacing"
    assert "1008" in str(raised), raised
    # One refusal, not a retry loop -- that is the whole point of the distinction.
    assert stub.connections == 1, f"retried a refused token {stub.connections} times"
    print(f"  auth: {raised}; stopped after one attempt")


async def test_outbox_supersedes_partials_and_keeps_finals():
    loop = asyncio.get_running_loop()
    outbox = Outbox(loop, max_final_age=30.0)
    outbox._put(Utterance("mine", final=False))
    outbox._put(Utterance("mine the", final=False))
    outbox._put(Utterance("mine the diamond", final=False))
    outbox._put(Utterance("mine the diamond ore", final=True))

    first = await outbox.get()
    assert first.final and first.text == "mine the diamond ore", first
    second = await outbox.get()
    # Only the newest partial survived; the two older ones were never worth sending.
    assert not second.final and second.text == "mine the diamond", second
    print("  outbox: finals first, and only the newest partial")


async def test_outbox_gives_up_on_stale_finals():
    """Held across a reconnect, yes -- but a command that lands a minute late is wrong."""
    outbox = Outbox(asyncio.get_running_loop(), max_final_age=0.2)
    outbox._put(Utterance("build a house", final=True,
                          queued_at=time.monotonic() - 5.0))
    outbox._put(Utterance("dig down", final=True))
    got = await outbox.get()
    assert got.text == "dig down", got
    assert outbox.expired == 1, outbox.expired
    print("  outbox: a 5s-old final was dropped, the fresh one sent")


class FakeLine:
    """Just the fields of a moonshine TranscriptLine that the timing depends on."""

    def __init__(self, text, start_time, duration, latency=141):
        self.text = text
        self.start_time = start_time
        self.duration = duration
        self.last_transcription_latency_ms = latency


async def test_end_of_speech_comes_from_the_audio_clock():
    """The number the whole budget is about, measured without needing a microphone.

    Timing from the callback instead would measure how fast we reacted to being told the
    phrase was over -- always small, always flattering, and not what anyone is asking.
    """
    outbox = Outbox(asyncio.get_running_loop())
    source = MicSource(ClientConfig(join_window=0), outbox, live=None)
    source.epoch = time.monotonic() - 3.0        # the stream opened 3s ago
    # The phrase ran from 1.0s to 2.5s on that stream, so speech ended 0.5s ago.
    source._final(FakeLine("mine the diamond ore", start_time=1.0, duration=1.5))

    got = await outbox.get()
    assert got.final and got.asr_ms == 141, got
    lag = (time.monotonic() - got.speech_end) * 1000
    assert 400 < lag < 700, f"end-of-speech placed at {lag:.0f} ms ago, expected ~500"
    print(f"  timing: end-of-speech from start_time+duration, {lag:.0f} ms ago")


async def test_an_implausible_anchor_is_not_believed():
    """The sound card's clock is not the system's. A drifted anchor must report nothing
    rather than a confident wrong number."""
    outbox = Outbox(asyncio.get_running_loop())
    source = MicSource(ClientConfig(join_window=0), outbox, live=None)

    source.epoch = time.monotonic() - 1.0
    source._final(FakeLine("from the future", start_time=50.0, duration=1.0))
    assert (await outbox.get()).speech_end is None, "believed an end-of-speech in the future"

    source.epoch = time.monotonic() - 600.0
    source._final(FakeLine("ancient history", start_time=1.0, duration=1.0))
    assert (await outbox.get()).speech_end is None, "believed a 10-minute-old anchor"
    print("  timing: a drifted anchor reports asr only, not a wrong end->sent")


async def test_a_phrase_split_at_a_pause_goes_out_as_one():
    """Moonshine ends a line at a short hesitation. The server must get one command, not a
    first half to act on and a second half to be muted over."""
    outbox = Outbox(asyncio.get_running_loop())
    source = MicSource(ClientConfig(join_window=0.2), outbox, live=None)
    source.epoch = time.monotonic() - 5.0

    source._final(FakeLine("Go forward into the red,", start_time=1.0, duration=1.5))
    await asyncio.sleep(0.1)
    source._started()                            # speech back inside the window
    await asyncio.sleep(0.4)                     # ...and a long second line: still holding
    assert outbox.empty(), "sent the first half while the second was still being said"
    source._partial("brick")
    partial = await outbox.get()
    assert partial.text == "Go forward into the red, brick" and not partial.final, partial
    source._final(FakeLine("brick building.", start_time=3.0, duration=1.0))
    got = await outbox.get()
    assert got.final and got.text == "Go forward into the red, brick building.", got
    lag = time.monotonic() - got.speech_end
    assert 1.5 < lag < 2.4, f"end-of-speech should be the second line's, not the first's: {lag}"

    # speech heard again before Moonshine has noticed: held past the window, then joined
    for _ in range(50):
        source._level(0.001)                     # a quiet room sets the floor
    ago = lambda seconds: time.monotonic() - source.epoch - seconds
    source._final(FakeLine("Walk to the", start_time=ago(2.0), duration=1.0))  # ended 1 s ago
    source._level(0.05)                          # ...and speech is back now
    await asyncio.sleep(0.45)                    # window long closed
    assert outbox.empty(), "the level said speech was back, and it went out anyway"
    source._started()
    source._final(FakeLine("oak tree.", start_time=ago(0.5), duration=0.4))
    got = await outbox.get()
    assert got.text == "Walk to the oak tree.", got

    # trailing breath right at the end of a line is not a new line
    source._level(0.05)
    source._final(FakeLine("jump", start_time=ago(0.05), duration=0.05))
    started = time.monotonic()
    got = await outbox.get()
    assert got.text == "jump" and time.monotonic() - started < 0.5, got

    # a phrase with nothing after it goes out once the window closes
    started = time.monotonic()
    source._final(FakeLine("stop", start_time=ago(0.3), duration=0.3))
    got = await outbox.get()
    waited = time.monotonic() - started
    assert got.text == "stop" and 0.15 < waited < 0.5, (got, waited)
    # and an empty line does not hold anything up or send anything
    source._final(FakeLine("", start_time=ago(0.2), duration=0.2))
    await asyncio.sleep(0.3)
    assert outbox.empty()
    source.close()
    print(f"  speech: a phrase split at a pause is sent joined; a lone one after {waited:.2f}s")


# ---------------------------------------------------------------- frames

JPEG = b"\xff\xd8\xff\xe0" + b"pretend this is a game frame" * 8


@contextlib.contextmanager
def published(jpeg=JPEG, age=0.0):
    """A frame file as the agent process would leave it, optionally already `age` seconds old."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "frame.jpg")
        frames.write(jpeg, path)
        if age:
            stamp = time.time() - age
            os.utime(path, (stamp, stamp))
        yield path


def test_a_stale_frame_is_not_read():
    """The file outlives the process: a killed agent leaves a valid JPEG lying there forever,
    and sending it would narrate a session that ended."""
    with published(age=0.0) as path:
        assert frames.read_latest(path, max_age=10.0)[0] == JPEG
    with published(age=60.0) as path:
        assert frames.read_latest(path, max_age=10.0) is None, "sent a minute-old frame"
    assert frames.read_latest("/nonexistent/frame.jpg", max_age=10.0) is None
    print("  frames: a stale file, and a missing one, both read as nothing to send")


def test_a_frame_is_written_whole_or_not_at_all():
    """os.replace, not write-in-place: a reader must never catch a half-written JPEG."""
    with published() as path:
        for _ in range(20):
            frames.write(JPEG + b"x" * 4096, path)
            with open(path, "rb") as handle:
                assert handle.read() in (JPEG, JPEG + b"x" * 4096), "torn frame"
        assert not [name for name in os.listdir(os.path.dirname(path))
                    if name.endswith(".tmp")], "left a temp file behind"
    print("  frames: republished atomically, and no temp files left over")


def test_an_implausible_file_is_not_read():
    """MCAGENTS_FRAME_PATH is a well-known path and a typo can point it at anything. A
    reader that trusts it would load the whole file and then base64 it, 33% larger."""
    with published() as path:
        with open(path, "wb") as handle:
            handle.write(b"\0" * (frames.MAX_BYTES + 1))
        assert frames.read_latest(path, max_age=10.0) is None, "read a 4 MB+ 'frame'"
        with open(path, "wb") as handle:
            pass
        assert frames.read_latest(path, max_age=10.0) is None, "read an empty frame"
    print("  frames: an oversized or empty file is refused before it is read")


async def test_an_unreadable_channel_does_not_end_the_session():
    """The pump shares a wait() with the link, so returning would cancel the link and stop
    the microphone. A side channel does not get to end the session."""
    outbox = Outbox(asyncio.get_running_loop())
    reports = []
    config = config_for(0, frame_path="/dev/null", frame_interval=0.02)

    def explode(path, max_age):
        raise RuntimeError("something nobody predicted")

    original, frames.read_latest = frames.read_latest, explode
    module = sys.modules["mcagents.local_client"]
    module.read_latest = explode
    try:
        pump = asyncio.create_task(frame_pump(config, outbox, reports.append))
        await asyncio.sleep(0.2)
        assert not pump.done(), f"the pump ended the session: {pump.exception()}"
        assert len(reports) == 1 and "carrying on without it" in reports[0], reports
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)
    finally:
        frames.read_latest = original
        module.read_latest = original
    print("  frames: an unexpected read failure is said once and the session carries on")


def test_the_publisher_is_rate_limited():
    """It runs inside a 20-33 Hz step loop for a reader that wants one frame every few
    seconds, so publishing every step would be ~30x the work for a frame thrown away."""
    frame = np.full((224, 224, 3), 120, np.uint8)
    with published() as path:
        pub = frames.FramePublisher(interval=10.0, path=path)
        assert pub.offer(frame), "the first frame was withheld"
        assert not any(pub.offer(frame) for _ in range(50)), "published inside the interval"
        assert pub.published == 1
        assert not frames.FramePublisher(interval=0.0, path=path).offer(frame), \
            "interval 0 is documented as off, and published anyway"
    print("  frames: one publish per interval, and interval 0 is off")


def test_a_broken_channel_does_not_take_the_rollout_down():
    """A full disk or a read-only /tmp is a broken side channel, not a broken rollout. It is
    reported once and then the publisher stops trying -- a failure per step at 30 Hz would
    bury the run's own output."""
    said = []
    pub = frames.FramePublisher(interval=0.001, path="/proc/nowhere/frame.jpg",
                                report=said.append)
    frame = np.full((224, 224, 3), 120, np.uint8)
    assert pub.offer(frame) is False and not pub.enabled, "kept trying after a failure"
    assert pub.offer(frame) is False
    assert len(said) == 1 and "giving up" in said[0], said
    print("  frames: an unwritable path is reported once, then the channel switches off")


async def test_outbox_keeps_only_the_newest_frame():
    """One slot, newest wins -- a frame is worth nothing once a newer one exists."""
    outbox = Outbox(asyncio.get_running_loop())
    outbox._put(Frame(jpeg=b"old"))
    outbox._put(Frame(jpeg=b"new"))
    assert (await outbox.get()).jpeg == b"new"
    assert outbox.empty(), "a superseded frame was queued rather than replaced"
    print("  outbox: an unsent frame is replaced by the next one, not queued behind it")


async def test_a_final_goes_out_ahead_of_a_frame():
    """The command is the latency budget; the picture can wait a few hundred microseconds."""
    outbox = Outbox(asyncio.get_running_loop())
    outbox._put(Frame(jpeg=JPEG))
    outbox._put(Utterance("mine the diamond ore", final=True))
    first = await outbox.get()
    assert isinstance(first, Utterance), "a frame went ahead of a final"
    assert isinstance(await outbox.get(), Frame)
    print("  outbox: finals go ahead of a pending frame, which follows it")


async def test_a_spoken_final_carries_what_is_being_pointed_at():
    """"mine *that*" is two halves, and this is the half that used to be missing.

    The coordinate already went up on the frames, but a frame is up to `frame_interval` old
    and arrives as its own message, so the model had to work out for itself which picture the
    words belonged to. Stamped onto the final, the phrase and the point are one message.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "frame.jpg")
        frames.write(JPEG, path)
        frames.write_selection({"point": [0.51, 0.42], "box": [0.42, 0.31, 0.6, 0.78],
                                "locked": True, "held": True, "age": 1.4}, path)
        config = ClientConfig(frame_path=path)
        outbox = Outbox(asyncio.get_running_loop(),
                        selection_of=lambda: frames.read_selection(config.frame_path,
                                                                   config.frame_max_age))

        outbox._put(Utterance("mine that", final=True))
        final = await outbox.get()
        assert final.payload() == {"text": "mine that", "final": True,
                                   "selection": {"point": [0.51, 0.42],
                                                 "box": [0.42, 0.31, 0.6, 0.78],
                                                 "locked": True, "held": True,
                                                 "age": 1.4}}, final.payload()
        assert "pointing at 0.51, 0.42" in describe_selection(final.selection)

        # A partial is not stamped: the server acts on finals, and a partial three times a
        # second would re-read the channel for an answer nobody looks at.
        outbox._put(Utterance("mine th", final=False))
        assert (await outbox.get()).payload() == {"text": "mine th", "final": False}

        # And pointing at nothing is not an error. The words still go, without a coordinate,
        # because plenty of what gets said names its own target.
        frames.write_selection(None, path)
        outbox._put(Utterance("come back here", final=True))
        assert (await outbox.get()).payload() == {"text": "come back here", "final": True}
        assert describe_selection(None) == ""
    print("  wire: a final carries the point that was being pointed at as it ended")


async def test_a_frame_does_not_hold_up_ctrl_d():
    """--text drains what was *said* and exits. Frames keep arriving for as long as the agent
    runs, so draining them too would mean never exiting."""
    outbox = Outbox(asyncio.get_running_loop())
    outbox._put(Frame(jpeg=JPEG))
    assert not outbox.text_pending() and not outbox.empty()
    outbox._put(Utterance("chop the oak log", final=True))
    assert outbox.text_pending()
    print("  outbox: ctrl-d drains speech, not the frame channel")


async def test_frames_reach_the_server_as_base64_jpeg():
    stub = Stub()
    with published() as path:
        async with serve(stub.handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            outbox = Outbox(asyncio.get_running_loop())
            config = config_for(port, frame_path=path, frame_interval=0.05)
            pump = asyncio.create_task(frame_pump(config, outbox, lambda message: None))

            async def body():
                assert await until(lambda: len(stub.messages) == 1), "no frame arrived"

            try:
                await with_link(config, outbox, body)
            finally:
                pump.cancel()

    assert stub.messages[0] == {"kind": "frame",
                                "image": base64.b64encode(JPEG).decode()}, stub.messages
    print("  wire: a published frame arrives as {kind: frame, image: <base64 jpeg>}")


async def test_the_same_frame_is_not_sent_twice():
    """A paused or wedged agent republishes nothing; the client must go quiet rather than
    repeat the last picture down the tunnel every three seconds."""
    outbox = Outbox(asyncio.get_running_loop())
    reports = []
    with published() as path:
        config = config_for(0, frame_path=path, frame_interval=0.05)
        pump = asyncio.create_task(frame_pump(config, outbox, reports.append))
        try:
            assert await until(lambda: outbox._frame is not None)
            await outbox.get()
            await asyncio.sleep(0.3)                  # several polls, same file
            assert outbox._frame is None, "resent a frame the agent had not republished"

            frames.write(JPEG + b"moved", path)       # the agent steps again
            assert await until(lambda: outbox._frame is not None), "missed a new frame"
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)

    assert len(reports) == 1 and "is live" in reports[0], reports
    print("  frames: only republished frames are sent, and going live is reported once")


async def test_no_agent_running_sends_nothing():
    """The common case: the voice client is up and ROCKET-2 is not. It must be silent, not
    an error and not a retry storm."""
    outbox = Outbox(asyncio.get_running_loop())
    reports = []
    config = config_for(0, frame_path="/nonexistent/frame.jpg", frame_interval=0.05)
    pump = asyncio.create_task(frame_pump(config, outbox, reports.append))
    await asyncio.sleep(0.3)
    pump.cancel()
    await asyncio.gather(pump, return_exceptions=True)
    assert outbox.empty() and reports == [], (reports, outbox._frame)
    print("  frames: with nothing publishing, nothing is sent and nothing is said")


# ---------------------------------------------------------------- coming back

def test_a_reply_is_text_and_an_optional_reason():
    """`{"text"}` is a reply, `{"text", "reason"}` a goal ending; nothing else is either."""
    for message, expected in (
            ('{"text": "mining now"}', ("mining now", None)),
            (b'{"text": "mining now"}', ("mining now", None)),
            ('{"text": "I\'m here.", "reason": "arrived"}', ("I'm here.", "arrived")),
            ('{"text": "Stopping.", "reason": "lost"}', ("Stopping.", "lost")),
            ('{"text": "hm", "reason": 3}', ("hm", "3")),          # print whatever arrives
            ('{"text": "mining now", "point": "tree"}', ("mining now", None)),
            ("not json at all", None),
            ('"just a string"', None),
            ('{"said": "mining now"}', None),
            ('{"text": "   "}', None),
            ('{"text": 5}', None),
            ('{"kind": "frame", "image": "..."}', None),
            ('{"point": "tree", "interaction": "Mine"}', None),
            ("[1, 2]", None),
    ):
        assert parse_reply(message) == expected, (message, parse_reply(message))
    print("  reply: JSON with a string text is read, with its reason; anything else ignored")


def test_a_reply_is_labelled_by_why_it_was_said():
    noon = time.mktime((2026, 9, 13, 12, 3, 41, 0, 0, -1))
    assert format_reply("Heading out.", None, now=noon) == "[12:03:41] agent: Heading out."
    assert format_reply("I'm here.", "arrived", now=noon) == \
        "[12:03:41] agent (done): I'm here."
    assert format_reply("Stopping.", "lost", now=noon) == "[12:03:41] agent (lost): Stopping."
    assert format_reply("Hm.", "timeout", now=noon) == "[12:03:41] agent (timeout): Hm."
    assert format_reply("I'm here.", "arrived", color=True, now=noon) == \
        "\033[32m[12:03:41] agent (done): I'm here.\033[0m"
    assert "\033" not in format_reply("Stopping.", "lost", color=True, now=noon)
    print("  reply: timestamped, labelled done/lost/<reason>, done lines green on a TTY")


class FakeSpeaker(Speaker):
    """The real threads and mute logic; synthesis and playback replaced by timed sleeps.
    Synthesis is deliberately slow, since that gap before each sentence is where an
    is-it-talking poll would have reopened the mic."""

    def __init__(self, mute, synth=0.1, play=0.15):
        super().__init__("fake", mute=mute, report=lambda message: None)
        self.synth, self.play, self.played = synth, play, []

    def _split(self, text):
        return [part.strip() + "." for part in text.split(".") if part.strip()]

    def _synthesize(self, text):
        time.sleep(self.synth)
        return text.encode()

    def _play(self, wav):
        time.sleep(self.play)
        self.played.append(wav.decode())


async def test_the_mic_stays_muted_until_the_reply_has_played():
    muted = []
    speaker = FakeSpeaker(mute=muted.append)
    speaker.TAIL = 0.05
    speaker.start()
    started = time.monotonic()
    speaker.say("Heading to the tree. Then mining it.")
    assert muted == [True]
    await asyncio.sleep(0.2)
    speaker.say("Found it.")                # a second reply while the first plays
    assert await until(lambda: muted[-1] is False, timeout=5), muted
    elapsed = time.monotonic() - started
    # three sentences at 0.15 s of playback each, plus the tail, is the floor
    assert elapsed >= 0.5, f"unmuted after {elapsed:.2f}s, before everything played"
    assert muted.count(False) == 1, muted
    assert speaker.played == ["Heading to the tree.", "Then mining it.", "Found it."]
    speaker.close()
    print(f"  speech: mic muted across three sentences and reopened once, {elapsed:.2f}s later")


def test_only_a_new_goal_plays_the_cue():
    assert starts_a_task({"point": "the oak tree", "interaction": "Mine"})
    assert not starts_a_task({"cancel": True})
    assert not starts_a_task({"turn": 90})
    assert not starts_a_task({"point": [0.5, 0.5], "keep_memory": True})
    print("  cue: a new goal plays it; a cancel, a turn and a keep_memory resend do not")


def test_the_mic_stays_muted_while_either_is_playing():
    muted = []
    switch = shared_mute(muted.append)
    speaker, cue = switch("speaker"), switch("cue")
    cue(True)
    speaker(True)
    speaker(False)                          # the reply ended; the cue has not
    assert muted[-1] is True, muted
    cue(False)
    assert muted[-1] is False, muted
    print("  cue: the speaker finishing does not reopen the mic under a playing cue")


def test_a_cue_that_cannot_play_says_so_and_unmutes():
    muted, said = [], []
    cue = Cue("/nonexistent.mp3", mute=muted.append, report=said.append)
    cue.TAIL = 0
    with contextlib.suppress(RuntimeError):
        cue.check()
    cue.play()
    cue.play()
    assert until_sync(lambda: muted and muted[-1] is False), muted
    assert len(said) <= 1, said
    cue.close()
    print("  cue: a missing file or player unmutes the mic and complains at most once")


def until_sync(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_goals_are_read_in_every_shape_a_server_might_send():
    one = {"point": "tree", "interaction": "Mine", "stop": {"item": "log", "count": 3}}
    assert parse_goals(json.dumps(one)) == [one]
    assert parse_goals(json.dumps([one, one])) == [one, one]
    assert parse_goals(json.dumps({"goals": [one]})) == [one]
    assert parse_goals(json.dumps({"plan": [one]})) == [one]
    assert parse_goals(json.dumps({"kind": "goal", **one})) == [one], "kept the wrapper key"
    assert parse_goals(json.dumps({"instruction": "Chop the oak log."})) == \
        [{"instruction": "Chop the oak log."}]
    assert parse_goals('{"cancel": true}') == [{"cancel": True}], "the halt is a goal message"
    assert describe_goal({"cancel": True}) == "cancel"
    print("  goal: one, a list, or a wrapped list -- all read, in order")


def test_a_message_that_is_not_a_goal_does_not_become_one():
    """A stray {"status": "ok"} turned into a goal sends the agent at nothing, silently."""
    for junk in ('{"status": "ok"}', '"hello"', "not json", "[]", '{"goals": []}',
                 '[1, 2, 3]', "null"):
        assert parse_goals(junk) == [], (junk, parse_goals(junk))
    print("  goal: an object with no goal key in it is not treated as a goal")


def test_a_turn_is_its_own_message_and_is_checked():
    """`{"turn": 90}` used to vanish: no goal key, so not a goal, so dropped without a word."""
    for degrees in (90, -90, 180, -180, 1, 45.5):
        assert parse_turn_command(json.dumps({"turn": degrees})) == {"turn": degrees}
    assert parse_turn_command(json.dumps({"kind": "turn", "turn": 90})) == {"turn": 90}
    assert parse_goals('{"turn": 90}') == [], "a turn is not a plan entry"
    assert describe_goal({"turn": -90}) == "turn -90 deg (left)"
    for bad in ('{"turn": 0}', '{"turn": 270}', '{"turn": -181}', '{"turn": "90"}',
                '{"turn": true}', '{"turn": null}', '{"turn": 90, "interaction": "Approach"}',
                '[{"turn": 90}]', '{"goals": [{"turn": 90}, {"cancel": true}]}'):
        try:
            parse_turn_command(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    # ...and a turn mixed with goal keys never runs as the goal, either.
    assert parse_goals('{"turn": 90, "interaction": "Approach", "point": "tree"}') == []
    for other in ('{"cancel": true}', '{"point": "tree"}', '{"status": "ok"}', "not json"):
        assert parse_turn_command(other) is None, other
    print("  goal: a turn is recognised, range-checked, and never half-run as a goal")


async def test_turns_queue_and_bad_turns_are_refused_out_loud():
    turns = [{"turn": 90}, {"turn": 0}, {"turn": -90}, {"turn": 270},
             {"turn": 90, "interaction": "Approach"}, {"turn": 180}]
    stub = Stub(sends=[json.dumps(turn) for turn in turns])
    lines = []
    with tempfile.TemporaryDirectory() as goals, tempfile.TemporaryDirectory() as out:
        spool, statuses = GoalSpool(goals), GoalSpool(out)
        async with serve(stub.handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            config = config_for(port, goal_url=f"ws://127.0.0.1:{port}")
            task = asyncio.create_task(run_goal_link(config, spool, lines.append, statuses=statuses))
            try:
                assert await until(lambda: len(stub.messages) == 3), stub.messages
                assert await until(lambda: spool.pending() == 3), spool.pending()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert [spool.take() for _ in range(3)] == [{"turn": 90}, {"turn": -90}, {"turn": 180}]
    rejected = [(m["event"], m["reason"], m["turn"]) for m in stub.messages]
    assert rejected == [("rejected", "invalid", 0), ("rejected", "invalid", 270),
                        ("rejected", "invalid", 90)], rejected
    refusals = [line for line in lines if line.startswith("turn: rejected")]
    assert len(refusals) == 3 and "0 < |turn| <= 180" in refusals[0], lines
    assert "also carries interaction" in refusals[2], refusals
    assert "goal: turn +90 deg (right)" in lines, lines
    assert describe_status({"event": "ended", "reason": "turned", "turn": 180, "steps": 12}) == \
        "status: turn ended (turned) turn=180 steps=12"
    print("  wire: turns queue for ROCKET-2; 0, 270 and a turn with goal keys are logged and "
          "reported rejected")


async def test_the_conversation_comes_back_up_the_uplink():
    """Sending and receiving are concurrent: the answer must not wait for the next thing
    the person says."""
    stub = Stub(sends=['{"text": "heading to the tree now"}'])
    heard = []
    async with serve(stub.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        outbox = Outbox(asyncio.get_running_loop())
        config = config_for(port)
        task = asyncio.create_task(run_link(config, outbox, lambda m: None,
                                            lambda m: None, heard.append))
        try:
            assert await until(lambda: heard), "no reply arrived"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert parse_reply(heard[0]) == ("heading to the tree now", None), heard
    print("  wire: the model's words come back up the same socket, without being asked")


async def test_a_bad_reply_does_not_take_the_microphone_down():
    """A handler that raises must not drop the socket -- that would stop the session over a
    message nobody anticipated."""
    stub = Stub(sends=['{"text": "one"}', '{"text": "two"}'])
    seen = []

    def explode(raw):
        seen.append(raw)
        raise RuntimeError("a handler nobody debugged")

    async with serve(stub.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        outbox = Outbox(asyncio.get_running_loop())
        task = asyncio.create_task(run_link(config_for(port), outbox, lambda m: None,
                                            lambda m: None, explode))
        try:
            assert await until(lambda: len(seen) == 2), f"stopped reading after {len(seen)}"
            outbox.put(Utterance("still here", final=True))
            assert await until(lambda: stub.messages), "the uplink died with the handler"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    print("  wire: a reply that cannot be handled is reported, and the link carries on")


async def test_goals_arrive_on_the_second_tunnel_and_queue_for_rocket2():
    """The whole downlink, end to end: server -> goal socket -> spool -> the other process.

    Goals sent in one message stay one entry: a plan's second goal must not preempt its
    first, which is what a separate entry arriving mid-goal now does."""
    goal = {"point": "the oak tree", "interaction": "Mine", "stop": {"item": "log"}}
    stub = Stub(sends=[json.dumps(goal), json.dumps([goal, goal]), json.dumps({"cancel": True})])
    with tempfile.TemporaryDirectory() as directory:
        spool = GoalSpool(directory)
        async with serve(stub.handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            config = config_for(port, goal_url=f"ws://127.0.0.1:{port}")
            task = asyncio.create_task(run_goal_link(config, spool, lambda m: None))
            try:
                assert await until(lambda: spool.pending() == 3), \
                    f"only {spool.pending()} of 3 messages queued"
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert [spool.take() for _ in range(3)] == [goal, {"goals": [goal, goal]},
                                                    {"cancel": True}]
    print("  wire: goals off the second tunnel queue for the ROCKET-2 process, one entry "
          "per message, in order")


async def test_goal_statuses_go_up_the_goal_tunnel():
    """What became of each goal is sent upstream, in order, and a status caught by a dead
    socket is not lost."""
    stub = Stub()
    with tempfile.TemporaryDirectory() as goals, tempfile.TemporaryDirectory() as out:
        channel = StatusChannel(out)
        channel.emit("started", None, [0.1, 0.2, 0.3, 0.4], interaction="Approach")
        channel.emit("ended", "arrived", [0.0, 0.1, 0.7, 0.9], interaction="Approach", steps=88)
        sent = []
        async with serve(stub.handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            config = config_for(port, goal_url=f"ws://127.0.0.1:{port}")
            task = asyncio.create_task(run_goal_link(
                config, GoalSpool(goals), lambda m: None, statuses=channel.spool,
                on_status=sent.append))
            try:
                assert await until(lambda: len(stub.messages) == 2), stub.messages
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    events = [(m["kind"], m["event"], m["reason"]) for m in stub.messages]
    assert events == [("goal.status", "started", None), ("goal.status", "ended", "arrived")], events
    assert stub.messages[1]["box"] == [0.0, 0.1, 0.7, 0.9] and "t" in stub.messages[1]
    assert len(sent) == 2
    print("  wire: goal.status messages go up the goal tunnel, in order")


def test_frames_speed_up_while_an_approach_runs():
    config = ClientConfig(frame_interval=3.0, approach_frame_interval=0.5)
    cadence = FrameCadence(config)
    assert cadence.interval == 3.0
    cadence.observe({"event": "started", "interaction": "Mine"})
    assert cadence.interval == 3.0, "only an Approach is judged by eye"
    cadence.observe({"event": "started", "interaction": "Approach"})
    assert cadence.interval == 0.5
    cadence.observe({"event": "updated", "interaction": "Approach"})
    assert cadence.interval == 0.5
    cadence.observe({"event": "ended", "reason": "arrived"})
    assert cadence.interval == 3.0
    print("  frames: ~1 fps while an Approach runs, back to the slow cadence when it ends")


async def test_the_goal_tunnel_stops_on_a_wrong_token_like_the_uplink():
    """Both directions get the same refusal handling, because they got it from one place."""
    stub = Stub(token="something-else")
    with tempfile.TemporaryDirectory() as directory:
        async with serve(stub.handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            config = config_for(port, goal_url=f"ws://127.0.0.1:{port}")
            try:
                await asyncio.wait_for(
                    run_goal_link(config, GoalSpool(directory), lambda m: None), timeout=10)
            except TokenRejected:
                assert stub.connections == 1, f"retried a refusal {stub.connections} times"
                print("  auth: the goal tunnel stops on 1008 too, after one attempt")
                return
    raise AssertionError("a wrong token on the goal tunnel did not stop the client")


# ---------------------------------------------------------------- the goal spool

def test_the_spool_is_a_queue_not_a_slot():
    """The opposite of the frame channel, deliberately: a goal is a command, and the second
    one does not replace the first."""
    with tempfile.TemporaryDirectory() as directory:
        spool = GoalSpool(directory)
        for name in ("first", "second", "third"):
            spool.submit({"point": name, "interaction": "Mine"})
        assert spool.pending() == 3
        assert [spool.take()["point"] for _ in range(3)] == ["first", "second", "third"]
        assert spool.take() is None
    print("  spool: goals queue and come back oldest first, none overwritten")


def test_a_goal_that_waited_too_long_is_dropped():
    """The world has moved and the person has said something else since."""
    with tempfile.TemporaryDirectory() as directory:
        spool = GoalSpool(directory, max_age=30.0)
        path = spool.submit({"point": "tree", "interaction": "Mine"})
        stamp = time.time() - 120
        os.utime(path, (stamp, stamp))
        assert spool.take() is None and spool.dropped == 1
        assert spool.pending() == 0, "a dropped goal was left in the queue"
    print("  spool: a two-minute-old goal is dropped and counted, not run late")


def test_the_spool_survives_junk_without_stopping():
    """A malformed or oversized file is one bad goal, not a reason to stop listening."""
    with tempfile.TemporaryDirectory() as directory:
        spool = GoalSpool(directory)
        pathlib.Path(directory, "00000000000000000001-0.json").write_text("{not json")
        pathlib.Path(directory, "00000000000000000002-0.json").write_bytes(
            b"[" + b"0," * 40000 + b"0]")
        good = spool.submit({"point": "tree", "interaction": "Mine"})
        reasons = []
        spool.on_drop = lambda reason, name: reasons.append(reason)
        assert spool.take() == {"point": "tree", "interaction": "Mine"}, "junk hid a good goal"
        assert spool.pending() == 0 and not os.path.exists(good)
        assert reasons == ["malformed", "oversized"], f"dropped silently: {reasons}"
    print("  spool: malformed and oversized entries are discarded, the good one still runs")


def test_clearing_the_spool_reports_what_it_discarded():
    """--listen clears on the way in: goals said to an earlier session are about a world
    that no longer exists."""
    with tempfile.TemporaryDirectory() as directory:
        spool = GoalSpool(directory)
        spool.submit({"point": "tree"})
        spool.submit({"point": "rock"})
        assert spool.clear() == 2 and spool.pending() == 0
        assert spool.clear() == 0
    print("  spool: clearing says how many goals it threw away")


async def probed(config):
    """--check's exit code and what it printed, without it landing in the test output."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = await probe(config)
    return code, buffer.getvalue()


async def test_probe_reports_a_working_link():
    """And enqueues nothing: the probe is final=false, which the server is to ignore."""
    stub = Stub()
    async with serve(stub.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        code, out = await probed(config_for(port))

    assert code == 0, out
    assert "ok --" in out, out
    assert stub.messages == [{"text": "connection check", "final": False}], stub.messages
    print("  probe: reports a good link, and enqueues nothing doing it")


async def test_probe_blames_the_token_not_the_tunnel():
    """The failure the protocol actually specifies: upgrade accepted, then closed 1008.

    It surfaces out of send(), not connect(), which is easy to leave uncaught -- and the
    whole value of the probe is that this must not read as a network problem.
    """
    stub = Stub(token="something-else")
    async with serve(stub.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        code, out = await probed(config_for(port))

    assert code == 2, f"exit {code}, expected 2 (the token)\n{out}"
    assert "1008" in out and "AGENT_TOKEN" in out, out
    assert "tunnel and the server are both fine" in out, out
    print("  probe: a 1008 after the upgrade is reported as the token, exit 2")


async def test_probe_blames_the_tunnel_not_the_token():
    """Nothing listening at all must not read as an auth problem."""
    code, out = await probed(ClientConfig(url="ws://127.0.0.1:1", token="tok",
                                          open_timeout=5.0))
    assert code == 1, f"exit {code}, expected 1 (the link)\n{out}"
    assert "AGENT_TOKEN" not in out, out
    print("  probe: a dead endpoint is reported as the link, exit 1")


def test_text_mode_end_to_end():
    """The whole CLI, as the user runs it first: typed lines in, JSON on the wire out."""
    stub = Stub()
    received = []

    async def serve_once():
        async with serve(stub.handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "mcagents.local_client", "--text", "--no-speak", "--no-frames",
                "--url", f"ws://127.0.0.1:{port}", "--token", TOKEN,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
            process.stdin.write(b"mine the diamond ore\nchop the oak log\n")
            await process.stdin.drain()
            ok = await until(lambda: len(stub.messages) == 2, timeout=30)
            process.stdin.close()          # ctrl-d: it should drain and exit by itself
            try:
                out = await asyncio.wait_for(process.stdout.read(), timeout=15)
                code = await asyncio.wait_for(process.wait(), timeout=15)
            except asyncio.TimeoutError:
                out, code = b"", "timed out -- did not exit on EOF"
                process.terminate()
                await process.wait()
            received.append((ok, out.decode(), code))

    asyncio.run(serve_once())
    ok, out, code = received[0]
    assert ok, f"typed lines never reached the server; client said:\n{out}"
    assert code == 0, f"exit was {code}, output:\n{out}"
    assert stub.messages == [{"text": "mine the diamond ore", "final": True},
                             {"text": "chop the oak log", "final": True}], stub.messages
    assert "sent: mine the diamond ore" in out, out
    print("  --text: two typed lines sent as finals, echoed locally, clean exit on ctrl-d")


def test_text_mode_prints_replies_and_goal_endings():
    """The whole CLI again, with the server talking: a reply, a goal ending, and junk."""
    stub = Stub(sends=[{"text": "hello"},
                       {"text": "I'm here at the red building.", "reason": "arrived"},
                       "not json"])
    received = []

    async def serve_once():
        async with serve(stub.handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "mcagents.local_client", "--text", "--no-speak",
                "--no-frames", "--url", f"ws://127.0.0.1:{port}", "--token", TOKEN,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
            process.stdin.write(b"go to the red building\n")
            await process.stdin.drain()
            await until(lambda: stub.messages, timeout=30)
            await asyncio.sleep(0.5)
            process.stdin.close()
            out = await asyncio.wait_for(process.stdout.read(), timeout=15)
            await asyncio.wait_for(process.wait(), timeout=15)
            received.append(out.decode())

    asyncio.run(serve_once())
    out = received[0]
    assert re.search(r"^\[\d\d:\d\d:\d\d\] agent: hello$", out, re.M), out
    assert re.search(r"^\[\d\d:\d\d:\d\d\] agent \(done\): I'm here at the red building\.$",
                     out, re.M), out
    assert "not json" not in out and "\033" not in out, out    # piped: no colour
    print("  --text: replies and goal endings printed with their labels, junk ignored")


def test_placeholders_are_refused():
    config = ClientConfig()
    try:
        config.check()
    except SystemExit as exc:
        assert "placeholder" in str(exc), exc
        print("  config: the shipped placeholder host/token are refused with instructions")
        return
    raise AssertionError("the placeholder token was accepted")


def test_empty_credentials_are_refused():
    """`AGENT_TOKEN=` in .env is exported empty, so the dataclass default never applies.
    Caught here it is one line; caught at the server it is a 1008 that looks like a wrong
    token, and the tunnel gets blamed for it."""
    for url, token, expected in (
            ("wss://real.trycloudflare.com", "", "AGENT_TOKEN"),
            ("", "sometoken", "AGENT_WS_URL"),
            ("", "", "AGENT_WS_URL and AGENT_TOKEN")):
        try:
            ClientConfig(url=url, token=token).check()
        except SystemExit as exc:
            assert expected in str(exc), (expected, str(exc))
            continue
        raise AssertionError(f"empty {expected} was accepted")
    print("  config: an empty host or token is named and refused, not sent to the server")


def main():
    test_placeholders_are_refused()
    test_empty_credentials_are_refused()
    test_a_stale_frame_is_not_read()
    test_a_frame_is_written_whole_or_not_at_all()
    test_an_implausible_file_is_not_read()
    test_the_publisher_is_rate_limited()
    test_a_broken_channel_does_not_take_the_rollout_down()
    asyncio.run(test_outbox_supersedes_partials_and_keeps_finals())
    asyncio.run(test_outbox_gives_up_on_stale_finals())
    asyncio.run(test_partials_and_finals_reach_the_server())
    asyncio.run(test_one_connection_for_many_utterances())
    asyncio.run(test_a_final_survives_the_socket_dying())
    asyncio.run(test_a_wrong_token_stops_instead_of_retrying())
    asyncio.run(test_end_of_speech_comes_from_the_audio_clock())
    asyncio.run(test_an_implausible_anchor_is_not_believed())
    asyncio.run(test_a_phrase_split_at_a_pause_goes_out_as_one())
    asyncio.run(test_outbox_keeps_only_the_newest_frame())
    asyncio.run(test_a_final_goes_out_ahead_of_a_frame())
    asyncio.run(test_a_spoken_final_carries_what_is_being_pointed_at())
    asyncio.run(test_a_frame_does_not_hold_up_ctrl_d())
    asyncio.run(test_frames_reach_the_server_as_base64_jpeg())
    asyncio.run(test_the_same_frame_is_not_sent_twice())
    asyncio.run(test_no_agent_running_sends_nothing())
    asyncio.run(test_an_unreadable_channel_does_not_end_the_session())
    test_a_reply_is_text_and_an_optional_reason()
    test_a_reply_is_labelled_by_why_it_was_said()
    asyncio.run(test_the_mic_stays_muted_until_the_reply_has_played())
    test_only_a_new_goal_plays_the_cue()
    test_the_mic_stays_muted_while_either_is_playing()
    test_a_cue_that_cannot_play_says_so_and_unmutes()
    test_goals_are_read_in_every_shape_a_server_might_send()
    test_a_message_that_is_not_a_goal_does_not_become_one()
    test_a_turn_is_its_own_message_and_is_checked()
    asyncio.run(test_turns_queue_and_bad_turns_are_refused_out_loud())
    test_the_spool_is_a_queue_not_a_slot()
    test_a_goal_that_waited_too_long_is_dropped()
    test_the_spool_survives_junk_without_stopping()
    test_clearing_the_spool_reports_what_it_discarded()
    asyncio.run(test_the_conversation_comes_back_up_the_uplink())
    asyncio.run(test_a_bad_reply_does_not_take_the_microphone_down())
    asyncio.run(test_goals_arrive_on_the_second_tunnel_and_queue_for_rocket2())
    asyncio.run(test_goal_statuses_go_up_the_goal_tunnel())
    test_frames_speed_up_while_an_approach_runs()
    asyncio.run(test_the_goal_tunnel_stops_on_a_wrong_token_like_the_uplink())
    asyncio.run(test_probe_reports_a_working_link())
    asyncio.run(test_probe_blames_the_token_not_the_tunnel())
    asyncio.run(test_probe_blames_the_tunnel_not_the_token())
    test_text_mode_end_to_end()
    test_text_mode_prints_replies_and_goal_endings()
    print("ok")


if __name__ == "__main__":
    main()
