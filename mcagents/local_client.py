"""The capture side: microphone and game view -> one long-lived socket to the GPU box.

The interaction model runs remotely, behind a Cloudflare tunnel. This laptop's job is to
turn speech into text, pick up the frame ROCKET-2 is publishing next door, and get both
across the wire fast enough that the reply still feels like a reply.

    python -m mcagents.local_client                    # talk
    python -m mcagents.local_client --text             # type instead; no mic, no ASR
    python -m mcagents.local_client --check            # one attempt at the tunnel, a verdict
    python -m mcagents.local_client --no-frames        # words only, no game view
    python -m mcagents.local_client --no-speak         # read replies, do not say them
    python -m mcagents.local_client --list-devices     # which input is actually live
    AGENT_WS_URL=wss://host AGENT_TOKEN=... python -m mcagents.local_client

Either spelling works -- `python mcagents/local_client.py`, or `python local_client.py` from
inside the package -- and `--url`/`--token` stand in for the environment when the tunnel
host is a one-off rather than the one in .env.

The recognition is [voice.py](voice.py)'s `Listener` -- Moonshine v2, CPU-only -- rather than
a second ASR stack of its own. That is not only reuse. It is the latency budget: measured on
this laptop (Core Ultra 9 285H, CPU int8, beam 1), faster-whisper finalises a phrase in

    base.en   494 ms      tiny.en   283 ms

and those numbers are *flat* across 1s, 2s and 5s utterances, because Whisper's encoder
zero-pads every clip to 30 seconds. It is a floor, not a curve, and it does not come down
with threads or a smaller beam. Against a 100-150 ms finalisation budget inside a ~500 ms
speech-to-audio-back target, base.en spends the entire budget and then the rest of it too.
Moonshine's streaming encoder instead runs *while the person is still talking* and finalises
in ~148 ms at 7.84% WER, so what is left when they stop is the tail, not the whole job. The
reasoning is docs/voice.md; the measurement above is this machine.

Endpointing is Moonshine's, from the audio, which is why there is no webrtcvad here and no
silence timeout to tune. A separate VAD would be a second opinion about where the phrase
ended, arriving *after* the recogniser already had one.

## The wire

One connection for the whole session, because a TLS handshake per transcript delta costs
more than the model's entire turn. Two kinds of message go up it:

    {"text": "mine the diamond ore", "final": true}
    {"kind": "frame", "image": "<base64 jpeg>"}

...and either can carry what is being pointed at, which is the other half of "mine *that*":

    {"text": "mine that", "final": true,
     "selection": {"point": [0.51, 0.42], "box": [...], "locked": true, "held": true,
                   "age": 1.4}}

The point comes from whoever has the pixels -- `--hand` or `--gaze` in the ROCKET-2 process --
through the same sidecar the frames come through. On the utterance as well as on the frame,
because a frame is up to `frame_interval` old and arrives as its own message: stamped onto the
final, the phrase and the coordinate of *that* are one thing to reason about. See `Outbox`.

Partials go as `final: false` -- the server ignores them today, and they are what
endpointing and barge-in will hang off tomorrow -- but partials and finals are not worth the
same thing, which is the one idea in `Outbox`: a partial is worthless a moment later and
gets dropped, a final is the thing the person actually said and is held across a reconnect.

Frames are the second half of "mine *that*". A sentence with no picture asks the model to
resolve a pointing word against a world it cannot see, so every few seconds the agent's own
224x224 view goes up alongside the words. The pixels come from the process that has the
sim -- ROCKET-2 publishes them, `frames.py` is the seam, and this file only ever moves
bytes it did not decode. A frame is worthless once a newer one exists, so it takes the
partial's guarantees rather than the final's: one slot, newest wins, dropped on a reconnect.
Nothing is sent at all when no agent is running.

The remote runs under Slurm and the job can die mid-session, so a dropped socket is expected
operating procedure, not an error: reconnect with exponential backoff and keep the
microphone open throughout. A *rejected* socket is the opposite -- a wrong token closes with
1008, and retrying that just fails identically every few seconds while looking like a
network problem. So 1008 stops the client and says why.

## What it says back

The model's reply is printed with the time and a label -- `agent:` for an answer,
`agent (done):` when it speaks up on its own because a goal ended -- and spoken with
Moonshine's own on-device TTS, the same package as the ASR, so there is nothing new to
install. It plays through PipeWire (`pw-cat`) rather than PortAudio, for the reason in
`Speaker`. Piper `lessac-medium` by default:
on this laptop, with the sim running, it synthesised a 2.8 s sentence in ~400 ms where
Kokoro took 1.75 s. `say()` splits on sentences, so the first one plays while the rest is
still being made.

The microphone is muted while it talks, from the moment the reply arrives until playback and
a short echo tail are over. On laptop speakers the alternative is the client hearing its own
voice, transcribing it, and sending the model its answer back as the next command. The cost
is no barge-in; with headphones on, `--no-mute` gives it back.

## What it prints

Each final, as it goes out, with the lag beside it:

    sent: mine the diamond ore        (asr 141 ms . end->sent 156 ms)

`asr` is Moonshine's own reported transcription cost. `end->sent` is the number the budget
is actually about: from the end of the speech to the moment the bytes left this machine.
End-of-speech comes from the phrase's own place on the audio clock (`start_time + duration`)
anchored to when the stream opened -- not from when the callback happened to fire, which
would quietly measure delivery instead and always look good. `--timing` prints the raw
components if that anchoring ever needs checking.
"""
import argparse
import asyncio
import base64
import collections
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

# `python -m mcagents.local_client` starts with the repo root on sys.path; running the file
# as a plain script starts with *this directory* there instead, and then `mcagents` is not
# importable at all and the traceback is about voice.py rather than about how it was
# invoked. The two spellings should not have different outcomes, so the missing entry goes
# back -- appended, so a real installation of the package still wins.
if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcagents.frames import (DEFAULT_PATH as DEFAULT_FRAME_PATH, read_latest,
                             read_selection)
from mcagents.goals import (DEFAULT_DIR as DEFAULT_GOAL_DIR, DEFAULT_STATUS_DIR, TURN_KEY,
                            GoalSpool, parse_turn)
from mcagents.voice import (LiveLine, Listener, VoiceConfig, input_devices, input_level,
                            pipewire_default_input, PROBE_SECONDS, PROBE_WARMUP)

#: What the file ships with. Both have to be replaced before anything can connect, and
#: connecting to the literal placeholder produces a DNS failure that reads like a tunnel
#: problem -- so it is checked for by name instead.
PLACEHOLDER_URL = "wss://REPLACE-WITH-YOUR-TUNNEL.trycloudflare.com"
PLACEHOLDER_TOKEN = "REPLACE-WITH-YOUR-TOKEN"

#: The clip played when the agent takes on a new task.
DEFAULT_CUE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "assets", "let-me-do-it-for-you.mp3")


@dataclass
class ClientConfig:
    url: str = PLACEHOLDER_URL
    token: str = PLACEHOLDER_TOKEN

    #: ~300 ms between partials. This is the streaming update rate, not phrase length --
    #: where a phrase *ends* is decided inside Moonshine from the audio.
    update_interval: float = 0.3
    #: TINY rather than voice.py's SMALL: this laptop is also running Minecraft and ROCKET-2,
    #: and under that load SMALL finalised in 500-900 ms and overflowed the audio input.
    #: TINY costs accuracy (12.01% WER against 7.84%) to get the CPU back.
    arch: str = "TINY_STREAMING"
    device: Optional[str] = None
    audio_log: bool = False
    partial: bool = True
    #: How long a finished phrase waits for the next one before going out. Moonshine ends a
    #: line at a pause as short as 0.25 s, so "go forward into the red / brick building"
    #: arrives as two finals -- and the server acts on the first half, replies, and the reply
    #: mutes the mic over the second. The wait is cut short by nothing and extended by the
    #: mic hearing speech again (see MicSource), so it only has to cover the pause itself:
    #: the final already lands ~0.3-0.6 s after speech stops, and 0.4 on top of that joins
    #: pauses up to about a second. 0 sends each line as it comes.
    join_window: float = 0.4
    #: Seconds of VAD average that end a phrase. Moonshine's 0.5 is a floor under every
    #: end->sent; shorter answers sooner and can split a sentence at a thinking pause.
    vad_window: Optional[float] = 0.3
    #: None keeps Moonshine's 0.5.
    vad_threshold: Optional[float] = None

    #: Say the model's replies out loud, not only print them.
    speak: bool = True
    #: Any id from moonshine_voice's `list_tts_voices("en_us")`. Piper voices are the cheap
    #: ones; `kokoro_*` sound better and cost ~4x the CPU.
    tts_voice: str = "piper_en_US-lessac-medium"
    #: Stop listening while speaking, so the mic does not transcribe the speakers. Turn off
    #: with headphones, to be able to talk over a reply.
    mute_while_speaking: bool = True
    #: Played when a new goal arrives -- the agent saying it is on it. Empty plays nothing.
    cue: str = ""

    #: A quick tunnel closes an idle connection, and the first thing anyone notices is a
    #: dropped utterance after a quiet minute. Ping through the gaps in the conversation.
    ping_interval: float = 20.0
    ping_timeout: float = 20.0
    open_timeout: float = 10.0

    backoff_initial: float = 0.5
    #: Slurm requeues take a while and there is no point hammering; a minute between tries
    #: is still far below the human threshold for "I will just restart it".
    backoff_max: float = 30.0

    #: How long a final may sit unsent while the socket is down. It is held rather than
    #: dropped, because losing what someone just said means they have to say it again --
    #: but the agent is driving Minecraft in real time, and a command that lands forty
    #: seconds late is not the command anyone wanted.
    max_final_age: float = 30.0
    max_finals: int = 32

    #: Forward the game frames ROCKET-2 publishes, so the model sees what it is being
    #: told about. Nothing is sent when no agent is running, so this costs nothing to
    #: leave on. See mcagents/frames.py for the other half.
    frames: bool = True
    #: Seconds between frames on the wire. Three is a compromise: the model is being given
    #: context for a spoken instruction, not driven by vision -- JarvisVLA's own control
    #: loop calls its server once per *environment step* and is a different budget entirely.
    frame_interval: float = 3.0
    #: Seconds between looks at the frame channel while an Approach is running, when the
    #: planner is watching for arrival rather than waiting for a sentence. The agent
    #: publishes once a second and a frame is never sent twice, so polling at half that
    #: puts every published frame on the wire -- ~1 fps. 0 keeps `frame_interval` throughout.
    approach_frame_interval: float = 0.5
    frame_path: str = DEFAULT_FRAME_PATH
    #: Older than this and the published frame is a picture of a session that has ended:
    #: the file outlives the process that wrote it, so a crashed or ctrl-c'd agent leaves a
    #: perfectly valid JPEG lying there forever. A few missed publishes of headroom.
    frame_max_age: float = 10.0

    #: The second tunnel: the goals ROCKET-2 runs, as plan-entry JSON. Empty means the
    #: server has only the conversation tunnel up, which is a working session with no
    #: agent on the other end of it -- so this is optional rather than checked.
    goal_url: str = ""
    #: Where received goals queue for the other process. See mcagents/goals.py.
    goal_dir: str = DEFAULT_GOAL_DIR
    #: Where the other process leaves goal statuses for this one to send up the goal tunnel.
    status_dir: str = DEFAULT_STATUS_DIR
    #: Show what the model says back. There is no reason to turn this off except a test.
    conversation: bool = True

    keyterms: bool = True
    timing: bool = False

    @classmethod
    def from_env(cls) -> "ClientConfig":
        return cls(
            url=os.environ.get("AGENT_WS_URL", cls.url).rstrip("/"),
            token=os.environ.get("AGENT_TOKEN", cls.token),
            update_interval=float(os.environ.get("AGENT_UPDATE_INTERVAL",
                                                 cls.update_interval)),
            arch=os.environ.get("VOICE_ARCH", cls.arch),
            device=os.environ.get("VOICE_DEVICE") or None,
            vad_window=float(os.environ.get("VOICE_VAD_WINDOW", cls.vad_window)),
            join_window=float(os.environ.get("AGENT_JOIN_WINDOW", cls.join_window)),
            vad_threshold=(float(os.environ["VOICE_VAD_THRESHOLD"])
                           if os.environ.get("VOICE_VAD_THRESHOLD") else None),
            audio_log=os.environ.get("VOICE_AUDIO_LOG", "0") != "0",
            frames=os.environ.get("AGENT_FRAMES", "1") != "0",
            frame_interval=float(os.environ.get("AGENT_FRAME_INTERVAL",
                                                cls.frame_interval)),
            approach_frame_interval=float(os.environ.get("AGENT_APPROACH_FRAME_INTERVAL",
                                                         cls.approach_frame_interval)),
            goal_url=os.environ.get("AGENT_GOAL_WS_URL", cls.goal_url).rstrip("/"),
            speak=os.environ.get("AGENT_SPEAK", "1") != "0",
            tts_voice=os.environ.get("VOICE_TTS_VOICE") or cls.tts_voice,
            mute_while_speaking=os.environ.get("VOICE_MUTE_WHILE_SPEAKING", "1") != "0",
            cue=os.environ.get("AGENT_CUE", DEFAULT_CUE),
        )

    def voice_config(self) -> VoiceConfig:
        """The ASR half, as voice.py wants it. The LLM half of VoiceConfig goes unused:
        the model lives behind the socket now, not behind an HTTP call from here."""
        config = VoiceConfig.from_env()
        config.arch = self.arch
        config.update_interval = self.update_interval
        config.device = self.device
        config.audio_log = self.audio_log
        if self.vad_window is not None:
            config.vad_window = self.vad_window
        if self.vad_threshold is not None:
            config.vad_threshold = self.vad_threshold
        if not self.keyterms:
            config.keyterms = []
        return config

    def check(self) -> None:
        """Configured at all? Not just "is this the shipped string".

        Empty counts as unconfigured, and is its own failure rather than a subset of the
        placeholder one: `AGENT_TOKEN=` in .env is exported as the empty string, so
        `os.environ.get` finds the key and never reaches the default here. That used to
        reach the server and come back a 1008 -- indistinguishable from a genuinely wrong
        token, one network round trip later.
        """
        if not self.url or not self.token:
            empty = [name for name, value in (("AGENT_WS_URL", self.url),
                                              ("AGENT_TOKEN", self.token)) if not value]
            raise SystemExit(
                f"{' and '.join(empty)} {'is' if len(empty) == 1 else 'are'} empty -- "
                f"set {'it' if len(empty) == 1 else 'them'}, or pass --url and --token.\n"
                "  (an empty value in .env still counts as set, so the default never "
                "applies.)")
        if PLACEHOLDER_TOKEN in self.token or "REPLACE-WITH" in self.url:
            raise SystemExit(
                "the tunnel host and token are still the placeholders.\n"
                "  export AGENT_WS_URL=wss://your-tunnel.trycloudflare.com\n"
                "  export AGENT_TOKEN=your-token\n"
                "or pass --url and --token.")


class TokenRejected(RuntimeError):
    """The server refused the credentials. Retrying re-fails identically -- stop instead."""


@dataclass
class Utterance:
    text: str
    final: bool
    queued_at: float = field(default_factory=time.monotonic)
    #: Monotonic wall-clock of the end of the speech itself, when it can be established.
    speech_end: Optional[float] = None
    #: Moonshine's own reported cost for the final transcription pass.
    asr_ms: Optional[float] = None
    #: Raw timing components, kept only for --timing.
    debug: Optional[dict] = None
    #: What was being pointed at as this phrase ended -- the same dict the frames carry.
    #: Frames already carry it, but a frame is up to `frame_interval` old and the model has
    #: to *correlate* it with the words; stamped onto the utterance instead, "mine that"
    #: arrives with the coordinate of `that` in the same message. See `Outbox.selection_of`.
    selection: Optional[dict] = None

    def payload(self) -> dict:
        body = {"text": self.text, "final": self.final}
        if self.selection is not None:
            body["selection"] = self.selection
        return body


@dataclass
class Frame:
    """One game frame from the sim, on its way to the model.

    `final` is False, and not as a formality: it is what makes a frame behave like a partial
    everywhere the link already reasons about messages. An unsent frame is replaced by a
    newer one rather than queued behind it, and one caught in flight when the socket dies is
    dropped instead of being carried to the next connection -- both because a picture of the
    game three seconds ago is not worth the bytes, and both already written.

    The JPEG is carried as bytes and base64-encoded only in `payload()`, i.e. once, at the
    moment it is actually going out. A frame that gets superseded in the outbox never pays
    for its own encoding.
    """
    jpeg: bytes
    #: The publisher's mtime, so the reader can tell a new frame from the same one again.
    mtime: float = 0.0
    #: What was highlighted in *this* frame -- the centre of the red box and the box itself,
    #: as fractions of it -- or None when nothing was. Published alongside the JPEG by
    #: whoever had the pixels, because coordinates only mean something against a picture.
    #: It is what lets the model answer "mine that" with an interaction and a sentence
    #: instead of having to work out which tree "that" was.
    selection: Optional[dict] = None
    queued_at: float = field(default_factory=time.monotonic)
    final: bool = False

    def payload(self) -> dict:
        body = {"kind": "frame", "image": base64.b64encode(self.jpeg).decode("ascii")}
        if self.selection is not None:
            body["selection"] = self.selection
        return body


#: What travels through the Outbox and out of the socket. The two share `final`,
#: `queued_at` and `payload()`, which is the whole interface the link needs.
Message = Union[Utterance, Frame]

#: The keys a plan entry can carry -- `mcagents.cli.plan` and `Agent.set_goal` between them.
#: An object with none of these is not a goal, however goal-shaped it looks.
GOAL_KEYS = frozenset({"point", "instruction", "interaction", "stop",
                       "normalized", "keep_memory", "cancel"})

#: How a reply's `reason` is labelled on screen. A reason not listed here is shown as itself,
#: so one the server adds later still says *why* rather than disappearing.
REASON_LABELS = {"arrived": "done"}


def decode(raw) -> Any:
    """A frame off the socket as JSON, or as the string it plainly is."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def parse_reply(raw) -> Optional[Tuple[str, Optional[str]]]:
    """`(text, reason)` from a conversation message, or None if it is not one.

    The conversation tunnel carries `{"text": ...}`, plus `"reason"` when the agent speaks
    unprompted because a goal ended. Anything else -- not JSON, no string `text`, blank
    text -- is ignored rather than shown: those words are also spoken, and nobody wants a
    status object read aloud brace by brace.
    """
    message = decode(raw)
    if not isinstance(message, dict):
        return None
    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    reason = message.get("reason")
    return text.strip(), (None if reason is None or reason == "" else str(reason))


def format_reply(text: str, reason: Optional[str], color: bool = False,
                 now: Optional[float] = None) -> str:
    """`[12:04:02] agent (done): I'm here.` -- local wall clock, and why, when there is a why.

    A goal that finished is the one line worth spotting in a scrolling terminal, so on a
    TTY it is green.
    """
    label = "agent" if reason is None else f"agent ({REASON_LABELS.get(reason, reason)})"
    line = f"[{time.strftime('%H:%M:%S', time.localtime(now))}] {label}: {text}"
    if color and reason == "arrived":
        line = f"\033[32m{line}\033[0m"
    return line


def parse_goals(raw) -> List[Dict[str, Any]]:
    """Every plan entry in one message, in order. An empty list if it holds none.

    A server may reasonably send one goal, a list of them, or a list wrapped in an object,
    so all three are read. What is *not* accepted is an object with no goal key in it: a
    stray `{"status": "ok"}` turned into a goal would send the agent at nothing, and
    silently, which is the worst of both.
    """
    message = decode(raw)
    if isinstance(message, dict):
        for key in ("goals", "plan", "entries"):
            if isinstance(message.get(key), list):
                message = message[key]
                break
    entries = message if isinstance(message, list) else [message]
    goals = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry = {key: value for key, value in entry.items() if key != "kind"}
        # A turn is not a plan entry, however many goal keys ride along with it -- see
        # `parse_turn_command`, which refuses the whole message instead.
        if set(entry) & GOAL_KEYS and TURN_KEY not in entry:
            goals.append(entry)
    return goals


def parse_turn_command(raw) -> Optional[Dict[str, Any]]:
    """`{"turn": degrees}` if this message is an in-place turn, None if it is not one at all.

    A message with a `turn` key is a turn, and a turn travels alone. Raises ValueError for one
    that cannot be run -- out of range, not a number, carrying goal keys too, or inside a list
    of goals -- so the caller can say so rather than let it vanish the way a key nobody knew
    used to.
    """
    message = decode(raw)
    if isinstance(message, dict):
        message = {key: value for key, value in message.items() if key != "kind"}
        if TURN_KEY in message:
            return {TURN_KEY: parse_turn(message)}
        for key in ("goals", "plan", "entries"):
            if isinstance(message.get(key), list):
                message = message[key]
                break
    if isinstance(message, list) and any(isinstance(entry, dict) and TURN_KEY in entry
                                         for entry in message):
        raise ValueError("a turn is sent on its own, not inside a list of goals")
    return None


def describe_selection(selection: Optional[dict]) -> str:
    """` [pointing at 0.51, 0.42]` for the terminal, or nothing when nothing was pointed at.

    Worth a few characters on the line that is already printed for every final: whether the
    coordinate rode along with the words is the one thing that decides if "mine that" can be
    answered at all, and without it the failure is a silent one on the far side of a tunnel.
    """
    if not isinstance(selection, dict):
        return ""
    point = selection.get("point")
    if not (isinstance(point, (list, tuple)) and len(point) == 2):
        return " [pointing]"
    held = " held" if selection.get("held") else ""
    return f" [pointing at {point[0]:.2f}, {point[1]:.2f}{held}]"


def describe_goal(entry: Dict[str, Any]) -> str:
    """One line naming a goal, for the terminal the person is watching."""
    if entry.get("cancel") is True:
        return "cancel"
    if TURN_KEY in entry:
        degrees = entry[TURN_KEY]
        return f"turn {degrees:+g} deg ({'right' if degrees > 0 else 'left'})"
    what = entry.get("point", entry.get("instruction", "?"))
    if isinstance(what, (list, tuple)) and len(what) == 2:
        what = f"({what[0]}, {what[1]})"
    return f"{entry.get('interaction', 'Approach')} {what!r} stop={entry.get('stop')!r}"


class Outbox:
    """The hand-off from the producer threads to the socket's event loop.

    Three kinds of message with different guarantees, which is the whole reason this is not
    just an asyncio.Queue. A partial is worth nothing once a newer one exists, so an unsent
    partial is *replaced* rather than queued behind -- under any backpressure at all the
    right partial to send is the current one. A frame is the same argument in pixels: there
    is one slot, and the newest occupant wins. A final is the utterance itself, so finals
    queue, survive a reconnect, and are only given up on when they are too old to still be
    what the speaker meant.

    The order out is finals, then the frame, then the partial. Finals first because they are
    the command and can only ever belong to an earlier or the same phrase. The frame ahead
    of the partial because the server ignores partials today and does not ignore frames, and
    because a frame arrives every few seconds while a partial regenerates three times a
    second -- so this cannot starve partials, where the other order could sit on a frame for
    the length of a long sentence.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, max_final_age: float = 30.0,
                 max_finals: int = 32, selection_of=None):
        #: Called for every final, to stamp it with what was being pointed at as it ended.
        #: Here rather than at each of the two places a final is made (the microphone and
        #: --text) because this is already the one place that knows a final from a partial --
        #: and it runs at *queue* time, so the coordinate is the one from when the phrase
        #: ended and not the one from whenever the socket got around to sending it.
        self.selection_of = selection_of
        self._loop = loop
        self._finals: "collections.deque[Utterance]" = collections.deque(maxlen=max_finals)
        self._partial: Optional[Utterance] = None
        self._frame: Optional[Frame] = None
        self._ready = asyncio.Event()
        self.max_final_age = max_final_age
        self.expired = 0                       # finals given up on, for the closing summary

    def put(self, message: Message) -> None:
        """Callable from any thread: Moonshine's worker, or the stdin reader."""
        self._loop.call_soon_threadsafe(self._put, message)

    def _put(self, message: Message) -> None:
        if isinstance(message, Frame):
            self._frame = message
        elif message.final:
            if message.selection is None and self.selection_of is not None:
                message.selection = self.selection_of()
            self._finals.append(message)
        else:
            self._partial = message
        self._ready.set()

    async def get(self) -> Message:
        while True:
            await self._ready.wait()
            now = time.monotonic()
            while self._finals:
                utterance = self._finals.popleft()
                if now - utterance.queued_at > self.max_final_age:
                    self.expired += 1
                    continue
                self._settle()
                return utterance
            if self._frame is not None:
                frame, self._frame = self._frame, None
                self._settle()
                return frame
            if self._partial is not None:
                utterance, self._partial = self._partial, None
                self._settle()
                return utterance
            self._ready.clear()

    def _settle(self) -> None:
        if self.empty():
            self._ready.clear()

    def empty(self) -> bool:
        return not self._finals and self._partial is None and self._frame is None

    def text_pending(self) -> bool:
        """Whether anything *said* is still waiting. What ctrl-d has to drain.

        Frames are excluded deliberately: they arrive on their own clock for as long as the
        agent is running, so waiting for the outbox to be empty of everything would be
        waiting for the agent to stop, and `--text` would never exit.
        """
        return bool(self._finals) or self._partial is not None


class _PeerGone(Exception):
    """The socket died while the outbox was empty -- noticed by watching, not by sending."""

    def __init__(self, code, reason):
        self.code = code
        super().__init__(f"closed {code}" + (f" ({reason})" if reason else ""))


async def next_to_send(outbox: "Outbox", closed: asyncio.Task) -> Optional[Message]:
    """The next message, or None if the connection died first.

    Waiting only on the outbox is the version of this that looks correct and silently is
    not: with nothing to say, a dropped socket goes unnoticed until the person next speaks,
    which turns every reconnect into a lost utterance and moves the reconnect delay to the
    worst possible moment. So the socket is watched at the same time.

    If both finish together the message still wins -- it is already out of the outbox, and
    `inflight` will carry it to the next connection rather than drop it on the floor.
    """
    getter = asyncio.ensure_future(outbox.get())
    await asyncio.wait({getter, closed}, return_when=asyncio.FIRST_COMPLETED)
    if getter.done():
        return getter.result()
    getter.cancel()
    await asyncio.gather(getter, return_exceptions=True)
    return None


def describe_close(exc: BaseException) -> str:
    if isinstance(exc, ConnectionClosed):
        return f"closed {exc.code}" + (f" ({exc.reason})" if exc.reason else "")
    return f"{type(exc).__name__}: {exc}"


async def hold_open(config: ClientConfig, url: str, label: str, body, report,
                    on_disconnect=None) -> None:
    """One connection to `url`, held open; and when it dies, the next one. Forever.

    The loop body is a whole connection's life, so backoff resets by simply being reassigned
    at the top of a successful one -- there is no separate "are we healthy yet" state to get
    out of step with the socket.

    `body(socket)` is what the connection is *for*, and returning from it means the socket
    is gone: the send loop returns when the outbox and the socket both stop answering, the
    receive loop when the iterator ends. Either way it becomes a `_PeerGone` carrying the
    close code, so the two links classify a disconnect identically.

    Both directions run through here because both need the same three things and get them
    wrong in the same three ways: reconnect with jittered backoff, distinguish a *dropped*
    socket (the Slurm job went away -- normal) from a *refused* one (the token is wrong --
    retrying re-fails identically forever while looking like a network fault), and say which
    happened. `on_disconnect` is the hook for whatever state the caller carries across
    connections, which today is only the uplink's `inflight`.
    """
    delay = config.backoff_initial
    while True:
        try:
            async with connect(url,
                               additional_headers={"X-Agent-Token": config.token},
                               open_timeout=config.open_timeout,
                               ping_interval=config.ping_interval,
                               ping_timeout=config.ping_timeout) as socket:
                delay = config.backoff_initial
                report(f"connected to {url}{label}")
                await body(socket)
                raise _PeerGone(socket.close_code, socket.close_reason)

        except InvalidStatus as exc:
            # Refused during the HTTP upgrade. 401/403 is the token; anything else is the
            # tunnel itself -- a 502 is the usual shape of "the Slurm job is not there".
            status = exc.response.status_code
            if status in (401, 403):
                raise TokenRejected(f"HTTP {status} on the upgrade request") from exc
            reason = f"HTTP {status} on the upgrade request"

        except _PeerGone as exc:
            if exc.code == 1008:
                raise TokenRejected("the server closed with 1008 (policy violation)") from exc
            reason = str(exc)

        except ConnectionClosed as exc:
            if exc.code == 1008:
                raise TokenRejected("the server closed with 1008 (policy violation)") from exc
            reason = describe_close(exc)

        except (OSError, InvalidHandshake, asyncio.TimeoutError) as exc:
            reason = describe_close(exc)

        if on_disconnect is not None:
            on_disconnect()

        # Jittered, so that a client and a restarting server do not fall into lockstep.
        wait = delay * (1 + random.random() * 0.25)
        report(f"disconnected{label} -- {reason}; retrying in {wait:.1f}s")
        await asyncio.sleep(wait)
        delay = min(delay * 2, config.backoff_max)


async def receive_loop(socket, handle) -> None:
    """Hand every inbound message to `handle`, until the socket ends.

    `handle` is not allowed to take the link down with it. A reply in a shape nobody
    anticipated is a bad message, not a dead connection, and dropping the socket over one
    would also drop the microphone -- so it is reported and the next message is read.
    """
    async for raw in socket:
        try:
            handle(raw)
        except Exception as exc:
            print(f"  (could not handle a reply: {type(exc).__name__}: {exc})", flush=True)


async def run_link(config: ClientConfig, outbox: Outbox, on_sent, report,
                   on_reply=None) -> None:
    """The uplink: everything in the outbox goes out here, and replies come back.

    `inflight` is the part that is easy to leave out and impossible to notice until the
    demo. Taking a final out of the outbox and *then* failing to send it loses the utterance
    at precisely the moment the whole reconnect design exists to survive, so it is held
    outside the connection's scope and retried on the next one. A partial or a frame in the
    same position is simply dropped: by the time there is a socket again it is long
    superseded.

    A successful `send()` means the frame reached the kernel, not that the server processed
    it. Surviving that last gap needs an ack in the protocol, and the protocol has none.

    Sending and receiving are concurrent, and have to be. A receive loop that took its turn
    between sends would only notice the model's answer when the person next spoke, which is
    the wrong way round: the answer is what they are waiting for.
    """
    inflight: Optional[Message] = None

    async def body(socket) -> None:
        nonlocal inflight
        closed = asyncio.ensure_future(socket.wait_closed())
        receiver = (asyncio.ensure_future(receive_loop(socket, on_reply))
                    if on_reply is not None else None)
        try:
            while True:
                if inflight is None:
                    inflight = await next_to_send(outbox, closed)
                    if inflight is None:
                        return
                await socket.send(json.dumps(inflight.payload()))
                sent, inflight = inflight, None
                on_sent(sent)
        finally:
            closed.cancel()
            if receiver is not None:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

    def on_disconnect() -> None:
        nonlocal inflight
        if inflight is not None and not inflight.final:
            inflight = None                    # neither a partial nor a frame survives one
        if inflight is not None and (time.monotonic() - inflight.queued_at
                                     > outbox.max_final_age):
            outbox.expired += 1                # ...and neither does a final this old
            inflight = None

    await hold_open(config, config.url, "", body, report, on_disconnect)


class FrameCadence:
    """How often `frame_pump` looks for a frame, which the goal statuses can change.

    An Approach is the one goal whose end the planner judges by eye, so while one runs the
    frames go up at `approach_frame_interval`; anything else, or nothing, is back to the
    conversational `frame_interval`. `changed` wakes the pump so a switch takes effect now
    rather than after the rest of a three-second sleep.
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.approaching = False
        self.changed: Optional[asyncio.Event] = None

    @property
    def interval(self) -> float:
        fast = self.config.approach_frame_interval
        return fast if self.approaching and fast > 0 else self.config.frame_interval

    def observe(self, status: Dict[str, Any]) -> None:
        if status.get("event") == "started":
            approaching = str(status.get("interaction", "")).lower() == "approach"
        elif status.get("event") == "ended":
            approaching = False
        else:
            return
        if approaching != self.approaching:
            self.approaching = approaching
            if self.changed is not None:
                self.changed.set()


def describe_status(status: Dict[str, Any]) -> str:
    """One line for a goal status, for the terminal."""
    box = status.get("box")
    width = f" width {box[2] - box[0]:.2f}" if isinstance(box, list) and len(box) == 4 else ""
    extra = "".join(f" {key}={status[key]}" for key in ("turn", "steps", "seconds", "after")
                    if key in status)
    reason = f" ({status['reason']})" if status.get("reason") else ""
    what = status.get("interaction") or ("turn" if "turn" in status else "")
    return (f"status: {what} {status.get('event')}{reason}"
            f"{width}{extra}")


async def run_goal_link(config: ClientConfig, spool: GoalSpool, report, on_goal=None,
                        statuses: Optional[GoalSpool] = None, on_status=None) -> None:
    """The second tunnel: goals for ROCKET-2 down, and what became of them back up.

    A separate connection rather than a second message kind on the uplink, because that is
    how the server is built -- one tunnel carries the conversation with the person, one
    carries the JSON the agent runs. Keeping them apart end to end means a restart of the
    planner behind one does not interrupt the other.

    Nothing is interpreted on the way down beyond "is this a goal-shaped object", and "is
    this a turn that can be run" -- a `{"turn": degrees}` out of range is refused here, logged
    and reported `rejected` / `invalid`, rather than dropped. This
    process has no sim, no policy and no opinion about whether a target exists; it is a
    courier, and `mcagents.cli.rocket2` is what turns an entry into a mask and an action.
    What it does keep is the *message*: several goals sent together are one sequence and are
    queued as one `{"goals": [...]}` entry, so the agent can tell a plan from a correction --
    the second goal of a plan must not preempt the first.

    On the way up go the `goal.status` messages the agent leaves in `statuses`. The server
    reads and discards them today, which is exactly what makes sending them free. A status
    taken and then caught by a dying socket is held and sent on the next connection.
    """
    def handle(raw) -> None:
        try:
            turn = parse_turn_command(raw)
        except ValueError as invalid:
            text = raw if isinstance(raw, str) else repr(raw)
            report(f"turn: rejected {text[:120]} -- {invalid}")
            if statuses is not None:
                message = decode(raw)
                statuses.submit({"kind": "goal.status", "event": "rejected", "reason": "invalid",
                                 "box": None, "t": round(time.time(), 3),
                                 "turn": message.get(TURN_KEY) if isinstance(message, dict) else None,
                                 "detail": str(invalid)})
            return
        goals = [turn] if turn is not None else parse_goals(raw)
        if not goals:
            return
        spool.submit(goals[0] if len(goals) == 1 else {"goals": goals})
        for entry in goals:
            report(f"goal: {describe_goal(entry)}")
            if on_goal is not None:
                on_goal(entry)

    inflight: Optional[Dict[str, Any]] = None

    async def send_statuses(socket) -> None:
        nonlocal inflight
        while True:
            if inflight is None:
                inflight = statuses.take()
            if inflight is None:
                await asyncio.sleep(0.05)
                continue
            await socket.send(json.dumps(inflight))
            sent, inflight = inflight, None
            if on_status is not None:
                on_status(sent)

    async def body(socket) -> None:
        if statuses is None:
            await receive_loop(socket, handle)
            return
        tasks = [asyncio.ensure_future(receive_loop(socket, handle)),
                 asyncio.ensure_future(send_statuses(socket))]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception() is not None and not isinstance(task.exception(),
                                                                   ConnectionClosed):
                    raise task.exception()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    await hold_open(config, config.goal_url, " (goals)", body, report)


async def probe(config: ClientConfig) -> int:
    """One attempt, no retry, and a verdict. `--text` is for *using* the link; this is for
    finding out why there is not one.

    The reconnect loop is right for a session and wrong for a question: with the tunnel down
    it retries politely forever and never says so. Here every stage is timed and named, so a
    failure lands on one of them -- and the three that get confused for each other (the
    tunnel is absent, the job behind it is absent, the token is wrong) look nothing alike.

    Only a partial is sent. The server enqueues on `final: true`, so this exercises the
    whole path -- DNS, TLS, the upgrade, the header, a frame -- without putting words in the
    agent's mouth.
    """
    parts = urlsplit(config.url)
    host = parts.hostname or ""
    port = parts.port or (443 if parts.scheme == "wss" else 80)
    print(f"  url      {config.url}")
    print(f"  token    {len(config.token)} chars"
          f"{' -- has surrounding whitespace' if config.token != config.token.strip() else ''}")

    clock = time.perf_counter
    if parts.scheme not in ("ws", "wss"):
        print(f"\n  the URL must start with ws:// or wss://, not {parts.scheme!r}://")
        return 1

    started = clock()
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port)
        addresses = sorted({info[4][0] for info in infos})
        print(f"  dns      {host} -> {', '.join(addresses)}"
              f"{'':<2}({(clock() - started) * 1000:.0f} ms)")
    except OSError as exc:
        print(f"  dns      cannot resolve {host} ({exc.strerror or exc})\n"
              f"\n  The tunnel host is wrong, or the tunnel is not running. A quick tunnel\n"
              f"  prints its hostname when `cloudflared` starts and gets a new one every\n"
              f"  restart -- check that against AGENT_WS_URL in .env.")
        return 1

    started = clock()
    try:
        async with connect(config.url,
                           additional_headers={"X-Agent-Token": config.token},
                           open_timeout=config.open_timeout,
                           ping_interval=None) as socket:
            print(f"  upgrade  101, connected{'':<12}({(clock() - started) * 1000:.0f} ms)")

            started = clock()
            probe = {"text": "connection check", "final": False}
            await socket.send(json.dumps(probe))
            print(f"  send     {json.dumps(probe)}{'':<2}({(clock() - started) * 1000:.0f} ms)")

            # A server that dislikes the token may accept the upgrade and close immediately
            # after, so give it a beat to say so before calling this a success.
            try:
                await asyncio.wait_for(socket.wait_closed(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            else:
                if socket.close_code == 1008:
                    raise TokenRejected("the server closed with 1008 after accepting")
                print(f"  closed   the server closed with {socket.close_code} "
                      f"({socket.close_reason or 'no reason given'})")
                return 1

    except InvalidStatus as exc:
        status = exc.response.status_code
        print(f"  upgrade  HTTP {status}\n")
        if status in (401, 403):
            print("  The token. AGENT_TOKEN in .env does not match the server's.")
            return 2
        if status in (502, 503, 504):
            print("  The tunnel is up but nothing is answering behind it -- the Slurm job\n"
                  "  is not running, or is not listening on the port cloudflared points at.")
        else:
            print("  The tunnel answered, but not with a WebSocket upgrade. Check that\n"
                  "  AGENT_WS_URL points at the agent server and not at something else.")
        return 1

    except TokenRejected as exc:
        print(f"\n  rejected: {exc}\n"
              "  The token. AGENT_TOKEN in .env does not match the server's.")
        return 2

    except ConnectionClosed as exc:
        # The shape the protocol actually specifies: the upgrade is accepted and the close
        # arrives a moment later, so this surfaces out of send() rather than connect().
        # Reaching here at all means the tunnel and the server are both fine.
        print(f"  closed   {describe_close(exc)}\n")
        if exc.code == 1008:
            print("  The token. The upgrade was accepted and then refused with 1008, so\n"
                  "  the tunnel and the server are both fine -- AGENT_TOKEN in .env does\n"
                  "  not match the server's.")
            return 2
        print("  The server accepted the connection and then closed it. That is the agent\n"
              "  server's own doing, not the tunnel's -- check its logs.")
        return 1

    except (OSError, InvalidHandshake, asyncio.TimeoutError) as exc:
        print(f"  upgrade  failed -- {describe_close(exc)}\n"
              "  DNS resolved, so the hostname is real, but the connection did not open.\n"
              "  Usually the tunnel process has stopped since that hostname was issued.")
        return 1

    print("\n  ok -- tunnel, token and protocol all good. Nothing was enqueued: the probe\n"
          "  was final=false, which the server accepts and ignores.")
    return 0


async def frame_pump(config: ClientConfig, outbox: Outbox, report,
                     cadence: Optional[FrameCadence] = None) -> None:
    """Every `frame_interval`, the frame ROCKET-2 last published -- if there is a new one.

    The sim is in the other process, so this is a poll rather than a subscription, and that
    is the right shape for the data: the channel holds exactly one frame and a reader that
    misses three of them has missed nothing, because it wanted the newest either way. See
    [frames.py](frames.py) for why that is a file.

    Two things it deliberately does not send. The *same* frame twice -- mtime says whether
    the publisher has moved since we last looked, and a paused or wedged agent should go
    quiet rather than repeat itself down the tunnel. And an *old* frame: `read_latest`
    refuses anything past `frame_max_age`, so an agent that died leaves the client sending
    nothing rather than narrating a session that is over.

    Going live and going quiet are both reported, once each. This is a side channel that is
    supposed to be invisible when it works, and the failure it actually has -- no agent
    running, so nothing is going -- is otherwise completely silent.

    The read is a few KB off tmpfs and stays on the event loop; a thread would cost more in
    hand-offs than the read does in blocking.

    Nothing in here is allowed to end the session. This coroutine sits in the same `wait()`
    as the link, so *returning* would cancel the link and exit the client -- which would
    mean the microphone stopped because a side channel had an opinion. So it never returns:
    an unexpected failure is reported once and then it keeps looping, quietly, and the
    person can go on talking.
    """
    last_mtime = 0.0
    live = False
    complained = False
    while True:
        try:
            latest = read_latest(config.frame_path, config.frame_max_age)
        except Exception as exc:                 # read_latest already swallows OSError
            latest = None
            if not complained:
                complained = True
                report(f"frames: cannot read {config.frame_path}, carrying on without it "
                       f"({type(exc).__name__}: {exc})")
        if latest is None:
            if live:
                report(f"frames: nothing published for {config.frame_max_age:.0f}s -- "
                       f"stopped sending")
                live = False
        else:
            jpeg, mtime = latest
            if mtime > last_mtime:
                last_mtime = mtime
                # Read after the frame, never before: the publisher writes the JPEG and then
                # the sidecar, so reading in that order cannot pair new coordinates with an
                # old picture. The worst case is the reverse -- a selection one frame stale,
                # which is a box slightly behind rather than a box on the wrong object.
                selection = read_selection(config.frame_path, config.frame_max_age)
                outbox.put(Frame(jpeg=jpeg, mtime=mtime, selection=selection))
                if not live:
                    report(f"frames: {config.frame_path} is live, sending one every "
                           f"{config.frame_interval:g}s ({len(jpeg) / 1024:.1f} KB)")
                    live = True
        if cadence is None:
            await asyncio.sleep(config.frame_interval)
            continue
        if cadence.changed is None:
            cadence.changed = asyncio.Event()
        cadence.changed.clear()
        try:
            await asyncio.wait_for(cadence.changed.wait(), timeout=cadence.interval)
        except asyncio.TimeoutError:
            pass


class MicSource:
    """Moonshine's Listener, wired to an Outbox, with the clocks lined up.

    The only subtlety is where end-of-speech comes from. `TranscriptLine` carries
    `start_time` and `duration` on the audio stream's own clock, so anchoring that clock to
    a monotonic reading taken as the stream opens gives the real instant the person stopped
    talking. Timing from the callback instead would measure how promptly we reacted to being
    told, which is a number that always looks good and means nothing.

    The anchor can only drift (the sound card's clock is not the system's), so the result is
    sanity-checked before it is believed rather than trusted outright.

    Finished lines are held for `join_window` before they go out, and joined to the next line
    if speech comes back first -- see ClientConfig.join_window. "Comes back" has two
    witnesses, because the obvious one is slow: Moonshine reports a new line ~0.6-0.9 s
    after speech resumes, by which point a short hold has already sent the first half. The
    mic's own level says it within one block. So when the window closes, the held words
    still wait if either
      * a new line has started -- then for it to finish, up to LONGEST_WAIT; or
      * the mic has heard speech since the held line ended -- then for Moonshine to catch
        up and start the line, up to HEARD_WAIT past the last loud block, and never more
        than HEARD_CAP in all, so a noisy room costs a moment rather than a stall.
    Loud means well above a noise floor that follows the room down fast and up slowly, so
    a fan or the game's steady hum raises the floor rather than holding commands back.
    """

    #: Held words go out after this long regardless. Moonshine's own longest segment is 15 s.
    LONGEST_WAIT = 15.0
    #: Measured gap from speech resuming to Moonshine starting the line, plus a margin.
    HEARD_WAIT = 1.0
    HEARD_CAP = 2.5
    #: A block is speech above this multiple of the floor, and above an absolute minimum
    #: (list_devices calls a mic "live" at a peak of 0.002).
    LOUD_OVER_FLOOR = 3.0
    LOUD_MIN = 0.004
    #: Trailing breath and room echo belong to the line that just ended, not a new one.
    TAIL_MARGIN = 0.15
    #: How often a held phrase re-checks whether it can go.
    POLL = 0.05

    #: Beyond this the anchor is not credible -- report the ASR cost alone rather than a
    #: confident wrong number.
    PLAUSIBLE = 30.0

    def __init__(self, config: ClientConfig, outbox: Outbox, live: Optional[LiveLine]):
        self.config = config
        self.outbox = outbox
        self.live = live
        self.epoch: Optional[float] = None
        self.listener: Optional[Listener] = None
        #: Words finished but not yet sent, and the last line's timing to send them with.
        self._held: List[str] = []
        self._held_timing: dict = {}
        self._held_at = 0.0                # when the most recent held line finished
        self._line_open = False            # a line has started since then
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._floor: Optional[float] = None
        self._heard_at = 0.0               # monotonic time of the last loud block

    def _level(self, rms: float) -> None:
        """PortAudio's callback thread: two float updates and nothing else."""
        floor = self._floor
        if floor is None or rms < floor:
            self._floor = rms
        else:
            self._floor = floor + (rms - floor) * 0.002
        if rms > max(self.LOUD_MIN, (floor or 0.0) * self.LOUD_OVER_FLOOR):
            self._heard_at = time.monotonic()

    def _partial(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            text = " ".join(self._held + [text])
        if self.live is not None:
            self.live.show(text)
        self.outbox.put(Utterance(text=text, final=False))

    def _started(self) -> None:
        """Speech is back. Held words now wait for this line, not for the window."""
        with self._lock:
            if self._held:
                self._line_open = True

    def _final(self, line) -> None:
        text = line.text.strip()
        speech_end = None
        if self.epoch is not None:
            end = self.epoch + float(line.start_time) + float(line.duration)
            if 0.0 <= (time.monotonic() - end) <= self.PLAUSIBLE:
                speech_end = end
        with self._lock:
            self._line_open = False
            if text:
                self._held.append(text)
                self._held_at = time.monotonic()
                self._held_timing = dict(
                    speech_end=speech_end, asr_ms=line.last_transcription_latency_ms,
                    debug=({"start_time": line.start_time, "duration": line.duration,
                            "since_epoch": time.monotonic() - (self.epoch or time.monotonic())}
                           if self.config.timing else None))
            if not self._held:
                return                     # an empty line with nothing waiting on it
            if self.config.join_window > 0:
                self._arm(self.config.join_window)
                return
        self._flush()

    def _arm(self, seconds: float) -> None:
        """(Re)start the send clock. Caller holds the lock."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(seconds, self._due)
        self._timer.daemon = True
        self._timer.start()

    def _still_speaking(self, now: float) -> bool:
        """Whether held words should keep waiting. Caller holds the lock."""
        if self._line_open:
            return now - self._held_at < self.LONGEST_WAIT
        ended = self._held_timing.get("speech_end")
        heard_again = ended is not None and self._heard_at > ended + self.TAIL_MARGIN
        return (heard_again and now - self._heard_at < self.HEARD_WAIT
                and now - self._held_at < self.HEARD_CAP)

    def _due(self) -> None:
        with self._lock:
            if self._held and self._still_speaking(time.monotonic()):
                self._arm(self.POLL)
                return
        self._flush()

    def _flush(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            words, self._held = self._held, []
            self._line_open = False
            timing = self._held_timing
        if words:
            self.outbox.put(Utterance(text=" ".join(words), final=True, **timing))

    def start(self) -> "MicSource":
        listener = Listener(self.config.voice_config(),
                            on_partial=self._partial,
                            on_final=self._final,
                            on_start=self._started,
                            on_level=self._level,
                            report=lambda message: print(message, flush=True))
        listener.load()
        # Anchor as close to the stream opening as it is possible to get from out here.
        self.epoch = time.monotonic()
        listener.start()
        self.listener = listener
        return self

    def close(self) -> None:
        if self.listener is not None:
            self.listener.close()
            self.listener = None
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class Speaker:
    """The model's replies, out loud, with the microphone held off while they play.

    Moonshine synthesises; PipeWire plays. Not Moonshine's own `say()`, because that plays
    through PortAudio, and the PortAudio in this conda env is built against bare ALSA: it
    cannot see PipeWire at all, and its "default output" is whichever raw device enumerates
    first -- on this laptop, NVIDIA HDMI 0. The speech played perfectly, into a port with
    nothing plugged in. `pw-cat` plays to PipeWire's default sink instead, which is the one
    the volume keys and the headphone jack already move.

    Raw PCM with the format spelled out, not a WAV. Piped a WAV, pw-cat 1.0.5 exits after
    about a third of the sentence -- exit 0, no error, and the rest simply never plays.

    Two threads, so the next sentence is synthesised while the current one plays, and one
    count of sentences not yet *heard*. The mic reopens when that count reaches zero and
    stays there for the echo tail -- which is also exact, because `pw-cat` only exits once
    its audio has drained out of the speaker.
    """

    #: After the last sample leaves, the room is still ringing and the mic still hears it.
    TAIL = 0.25

    def __init__(self, voice: str, mute=None, report=print):
        self.voice = voice
        self._mute = mute
        self._report = report
        self._tts = None
        self._player = ""
        self._texts: "queue.Queue[Optional[str]]" = queue.Queue()
        #: (16-bit little-endian mono PCM, sample rate) per sentence.
        self._audio: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._pending = 0                  # sentences queued and not yet finished playing
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._playing: Optional[subprocess.Popen] = None
        self._threads: List[threading.Thread] = []

    def load(self) -> "Speaker":
        """Blocks, and the first run downloads the voice -- so it says so, once."""
        from moonshine_voice.download import list_tts_voices
        from moonshine_voice.tts import TextToSpeech

        if shutil.which("pw-cat"):
            self._player = "pw-cat"
        elif shutil.which("aplay"):        # ALSA's default, which PipeWire usually owns
            self._player = "aplay"
        else:
            raise RuntimeError("neither pw-cat nor aplay is installed")

        # Asked up front: the progress callback also fires while checking cached files.
        try:
            cached = self.voice in list_tts_voices("en_us")["present"]
        except Exception:
            cached = True                  # not knowing is no reason to announce anything
        if not cached:
            self._report(f"speech: downloading the {self.voice} voice...")
        self._tts = (TextToSpeech().language("en_us").voice(self.voice)
                     .on_progress(lambda fraction, name: None).load())
        return self.start()

    def start(self) -> "Speaker":
        for target in (self._synth_worker, self._play_worker):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def say(self, text: str) -> None:
        """Returns at once. Muting happens here, before synthesis, not when sound starts."""
        sentences = self._split(text)
        if not sentences:
            return
        if self._mute is not None:
            self._mute(True)
        with self._lock:
            self._pending += len(sentences)
        for sentence in sentences:
            self._texts.put(sentence)

    def _split(self, text: str) -> List[str]:
        from moonshine_voice.tts import split_say_utterances
        return split_say_utterances(text, "en_us")

    def _synthesize(self, text: str) -> tuple:
        """One sentence as 16-bit mono PCM, and its rate."""
        samples, rate = self._tts.synthesize(text)
        pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
               * 32767).astype("<i2").tobytes()
        return pcm, rate

    def _play(self, audio: tuple) -> None:
        pcm, rate = audio
        if self._player == "pw-cat":
            command = ["pw-cat", "--playback", "--format", "s16", "--rate", str(rate),
                       "--channels", "1", "--media-role", "Communication", "-"]
        else:
            command = ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(rate),
                       "-c", "1", "-"]
        self._playing = subprocess.Popen(command, stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            _, err = self._playing.communicate(pcm)
        finally:
            code, self._playing = self._playing.returncode, None
        if code and not self._stopping.is_set():
            raise RuntimeError(f"{self._player} exited {code}: "
                               f"{err.decode(errors='replace').strip()[:200]}")

    def _synth_worker(self) -> None:
        while not self._stopping.is_set():
            text = self._texts.get()
            if text is None:
                break
            try:
                self._audio.put(self._synthesize(text))
            except Exception as exc:
                self._report(f"speech: could not synthesise ({type(exc).__name__}: {exc})")
                self._finished()

    def _play_worker(self) -> None:
        complained = False
        while not self._stopping.is_set():
            audio = self._audio.get()
            if audio is None:
                break
            try:
                self._play(audio)
            except Exception as exc:
                if not complained:         # once: a broken player breaks every sentence
                    complained = True
                    self._report(f"speech: could not play ({type(exc).__name__}: {exc})")
            self._finished()

    def _finished(self) -> None:
        with self._lock:
            self._pending -= 1
            quiet = self._pending == 0
        if not quiet or self._mute is None:
            return
        time.sleep(self.TAIL)              # on the play thread, which has nothing else to do
        with self._lock:
            if self._pending:              # another reply arrived during the tail
                return
        self._mute(False)

    def close(self) -> None:
        self._stopping.set()
        playing = self._playing
        if playing is not None and playing.poll() is None:
            playing.kill()
        self._texts.put(None)
        self._audio.put(None)
        for thread in self._threads:
            thread.join(timeout=2.0)
        if self._tts is not None:
            self._tts.close()
            self._tts = None


def starts_a_task(entry: Dict[str, Any]) -> bool:
    """Whether a goal off the tunnel is the agent taking on something new.

    Not a cancel, not an in-place turn, and not a `keep_memory` resend -- a planner refreshes
    its running goal every few seconds that way, and a cue on each would never stop playing.
    """
    return (entry.get("cancel") is not True and TURN_KEY not in entry
            and entry.get("keep_memory") is not True)


class Cue:
    """One sound file, played through PipeWire the moment a task starts.

    `pw-play` rather than `Speaker`'s queue: the file is an mp3, which pw-play decodes itself,
    and the cue is meant to be immediate rather than wait behind a reply being synthesised.
    A cue that is still playing is not restarted -- a plan of three goals is one task.

    The mic is held muted while it plays, for `Speaker`'s reason. `mute` here is its own
    switch; `shared_mute` is what combines it with the speaker's.
    """

    TAIL = Speaker.TAIL

    def __init__(self, path: str, mute=None, report=print):
        self.path = path
        self._mute = mute
        self._report = report
        self._playing: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._complained = False

    def check(self) -> "Cue":
        if not os.path.isfile(self.path):
            raise RuntimeError(f"no such file: {self.path}")
        if not shutil.which("pw-play"):
            raise RuntimeError("pw-play is not installed")
        return self

    def play(self) -> None:
        """Returns at once."""
        with self._lock:
            if self._playing is not None and self._playing.poll() is None:
                return
            if self._mute is not None:
                self._mute(True)
            try:
                self._playing = subprocess.Popen(
                    ["pw-play", "--media-role", "Notification", self.path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except OSError as exc:
                self._playing = None
                self._failed(f"{type(exc).__name__}: {exc}")
                return
            playing = self._playing
        threading.Thread(target=self._wait, args=(playing,), daemon=True).start()

    def _wait(self, playing: subprocess.Popen) -> None:
        _, err = playing.communicate()
        if playing.returncode and playing.returncode > 0:
            self._failed(f"pw-play exited {playing.returncode}: "
                         f"{err.decode(errors='replace').strip()[:200]}")
            return
        time.sleep(self.TAIL)
        with self._lock:
            if self._playing is not playing:   # closed, or replaced during the tail
                return
            self._playing = None
        if self._mute is not None:
            self._mute(False)

    def _failed(self, why: str) -> None:
        with self._lock:
            self._playing = None
        if self._mute is not None:
            self._mute(False)
        if not self._complained:              # once: a broken player breaks every cue
            self._complained = True
            self._report(f"cue: could not play ({why})")

    def close(self) -> None:
        with self._lock:
            playing, self._playing = self._playing, None
        if playing is not None and playing.poll() is None:
            playing.kill()


def shared_mute(mute):
    """Several things that each want the mic off, and one mic.

    Each caller gets its own switch, and the mic is muted while any switch is on. Without
    this the speaker finishing a reply would reopen the mic in the middle of a cue.
    """
    held: Dict[str, bool] = {}
    lock = threading.Lock()

    def switch(name: str):
        def set_muted(muted: bool) -> None:
            with lock:
                held[name] = muted
                mute(any(held.values()))
        return set_muted
    return switch


def stdin_lines(outbox: Outbox, stop: threading.Event, on_eof) -> None:
    """--text: every typed line is one final. Blocking reads, so it lives on a thread.

    This exists to take the entire audio stack out of the picture: if typing a line does not
    reach the agent, the problem is the token, the tunnel or the protocol, and no amount of
    staring at microphone levels will show it.
    """
    for line in sys.stdin:
        if stop.is_set():
            return
        text = line.strip()
        if text:
            outbox.put(Utterance(text=text, final=True, speech_end=time.monotonic()))
    on_eof()


async def drain_then_stop(outbox: Outbox, eof: asyncio.Event) -> None:
    """ctrl-d means finished, not abandoned: send what is queued, then come back."""
    await eof.wait()
    while outbox.text_pending():
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.15)          # the last send is in flight, not in the outbox


def make_reporter(live: Optional[LiveLine]):
    """Printing has to go around the rewriting partial line, or it lands on top of it."""
    def report(message: str) -> None:
        if live is None:
            print(message, flush=True)
        else:
            with live.pause():
                print(message, flush=True)
    return report


def make_on_sent(config: ClientConfig, live: Optional[LiveLine], stats: dict):
    """The line printed as each message leaves. Frames are counted, not narrated.

    A frame every three seconds would be three lines a minute of nothing happening, on a
    display whose whole job is to show a sentence being recognised. The transition into and
    out of sending is reported by `frame_pump`, the total at exit, and every individual
    frame only under `--timing`.
    """
    def on_sent(message: Message) -> None:
        if isinstance(message, Frame):
            stats["frames"] = stats.get("frames", 0) + 1
            if config.timing:
                age = (time.time() - message.mtime) * 1000
                emit(f"  frame: {len(message.jpeg) / 1024:.1f} KB   "
                     f"(published->sent {age:.0f} ms)")
            return
        utterance = message
        if not utterance.final:
            return
        now = time.monotonic()
        parts = []
        if utterance.asr_ms:
            parts.append(f"asr {utterance.asr_ms:.0f} ms")
        if utterance.speech_end is not None:
            parts.append(f"end->sent {(now - utterance.speech_end) * 1000:.0f} ms")
        else:
            parts.append(f"queued->sent {(now - utterance.queued_at) * 1000:.0f} ms")
        if utterance.debug:
            parts.append(" ".join(f"{k}={v:.3f}" for k, v in utterance.debug.items()))
        emit(f"  sent: {utterance.text}{describe_selection(utterance.selection)}   "
             f"({' . '.join(parts)})")

    def emit(line: str) -> None:
        if live is None:
            print(line, flush=True)
        else:
            with live.pause():
                print(line, flush=True)

    return on_sent


async def run(config: ClientConfig, use_text: bool) -> int:
    loop = asyncio.get_running_loop()
    live = LiveLine() if (config.partial and not use_text) else None
    outbox = Outbox(loop, config.max_final_age, config.max_finals,
                    selection_of=(lambda: read_selection(config.frame_path,
                                                         config.frame_max_age))
                    if config.frames else None)
    report = make_reporter(live)
    stats = {"frames": 0, "replies": 0, "goals": 0}
    on_sent = make_on_sent(config, live, stats)

    speaker: Optional[Speaker] = None
    cue: Optional[Cue] = None
    color = sys.stdout.isatty()

    def on_reply(raw) -> None:
        """What the model says back -- a reply, or a goal ending -- printed and spoken."""
        reply = parse_reply(raw)
        if reply is None:
            return
        text, reason = reply
        stats["replies"] += 1
        # In --text the terminal is echoing whatever is half-typed on the current line, and
        # a reply printed onto the end of it reads as part of the command.
        report(("\n" if use_text else "") + format_reply(text, reason, color))
        if speaker is not None:
            speaker.say(text)

    spool = GoalSpool(config.goal_dir)
    statuses = GoalSpool(config.status_dir, max_pending=256)
    cadence = FrameCadence(config)

    def on_goal(entry) -> None:
        stats["goals"] += 1
        if cue is not None and starts_a_task(entry):
            cue.play()

    def on_status(status) -> None:
        cadence.observe(status)
        if status.get("event") in ("ended", "rejected", "updated"):
            report(describe_status(status))

    source = None
    stop = threading.Event()
    eof = asyncio.Event()
    if use_text:
        threading.Thread(target=stdin_lines, daemon=True, args=(
            outbox, stop, lambda: loop.call_soon_threadsafe(eof.set))).start()
        print("Type a line and press enter; ctrl-d to stop.\n", flush=True)
    else:
        # Loading before the greeting, for voice.py's reason: the first run downloads
        # weights and opening the device is where the audio stack has its say, and a prompt
        # printed before either finishes is a prompt that gets buried.
        source = MicSource(config, outbox, live).start()

    switch = lambda name: None
    if source is not None and config.mute_while_speaking:
        switch = shared_mute(lambda muted: source.listener and source.listener.mute(muted))
    if config.cue:
        try:
            cue = Cue(config.cue, switch("cue"), report).check()
        except Exception as exc:
            report(f"cue: {exc} -- new tasks will start silently")
    if config.speak and config.conversation:
        try:
            speaker = Speaker(config.tts_voice, switch("speaker"), report).load()
        except Exception as exc:
            # A voice that will not load is a reason to stay quiet, not to lose the mic.
            report(f"speech: cannot load {config.tts_voice} ({type(exc).__name__}: {exc}) "
                   f"-- replies will only be printed")

    if not use_text:
        print(f"Listening with {config.arch}. Speak; ctrl-c to stop.\n", flush=True)

    if not config.goal_url:
        report("goals: AGENT_GOAL_WS_URL is not set -- the model can talk back, but "
               "nothing will drive ROCKET-2")

    link = asyncio.ensure_future(
        run_link(config, outbox, on_sent, report,
                 on_reply if config.conversation else None))
    # The pump never returns either, so it only ever leaves this list by being cancelled
    # alongside the link -- it is here to be cancelled, not to be waited on.
    waiting = [link]
    if config.frames:
        waiting.append(asyncio.ensure_future(frame_pump(config, outbox, report, cadence)))
    if config.goal_url:
        waiting.append(asyncio.ensure_future(
            run_goal_link(config, spool, report, on_goal, statuses, on_status)))
    if use_text:
        waiting.append(asyncio.ensure_future(drain_then_stop(outbox, eof)))
    try:
        # run_link never returns on its own, so whichever of these finishes first is the
        # reason we are stopping: a refused token, or stdin running out.
        done, pending = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            task.result()                       # re-raises TokenRejected
    except TokenRejected as exc:
        if live is not None:
            live.clear()
        print(f"\nrejected by the server: {exc}\n"
              f"  X-Agent-Token was {'unset' if not config.token else 'sent and refused'}. "
              f"Check AGENT_TOKEN against the server's, and that AGENT_WS_URL\n"
              f"  ({config.url}) is the tunnel in front of *that* server.\n"
              f"  Not retrying: a refused token refuses identically every time.",
              file=sys.stderr)
        return 2
    except asyncio.CancelledError:
        raise
    finally:
        stop.set()
        if speaker is not None:
            speaker.close()
        if cue is not None:
            cue.close()
        if source is not None:
            source.close()
        if live is not None:
            live.clear()
        if outbox.expired:
            print(f"({outbox.expired} final(s) went stale while disconnected and were "
                  f"dropped)", file=sys.stderr)
        summary = [f"{stats[key]} {key}" for key in ("frames", "replies", "goals")
                   if stats[key]]
        if summary:
            print(f"({', '.join(summary)})", file=sys.stderr)
    return 0


def list_devices() -> None:
    wanted = pipewire_default_input()
    print(f"PipeWire's default source is {wanted or 'unknown'}. Levels are peak amplitude "
          f"over {PROBE_SECONDS - PROBE_WARMUP:.1f}s of whatever the room is doing -- "
          f"speak, and they should jump:\n")
    for index, name in input_devices():
        level = input_level(index)
        mark = "*" if wanted and wanted in name else " "
        verdict = "live" if level > 0.002 else "silent -- nothing plugged in?"
        print(f" {mark} [{index}] {name:<34} peak {level:.4f}  {verdict}")
    print("\nPick one with --device N (or VOICE_DEVICE).")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", action="store_true",
                        help="read typed lines from stdin; no microphone, no ASR")
    parser.add_argument("--check", action="store_true",
                        help="connect once, report what happened, exit; no mic, no retry")
    parser.add_argument("--url", help="wss://... (default: $AGENT_WS_URL)")
    parser.add_argument("--token", help="X-Agent-Token (default: $AGENT_TOKEN)")
    parser.add_argument("--arch", help="TINY_STREAMING | SMALL_STREAMING | MEDIUM_STREAMING")
    parser.add_argument("--device", help="input device index or name substring")
    parser.add_argument("--list-devices", action="store_true",
                        help="show every input with its measured level, then exit")
    parser.add_argument("--update-interval", type=float,
                        help="seconds between partials (default 0.3)")
    parser.add_argument("--join-window", type=float,
                        help="seconds a finished phrase waits to be joined by the next "
                             "(default 0.4; 0 sends each at once)")
    parser.add_argument("--vad-window", type=float,
                        help="seconds of silence that end a phrase (default 0.3)")
    parser.add_argument("--vad-threshold", type=float,
                        help="VAD speech probability threshold (Moonshine default 0.5)")
    parser.add_argument("--no-partial", dest="partial", action="store_false",
                        help="do not show or send in-progress text")
    parser.add_argument("--no-keyterms", dest="keyterms", action="store_false",
                        help="do not bias the decoder towards Minecraft vocabulary")
    parser.add_argument("--no-frames", dest="frames", action="store_false",
                        help="do not forward the game frames ROCKET-2 publishes")
    parser.add_argument("--frame-interval", type=float,
                        help="seconds between frames on the wire (default 3)")
    parser.add_argument("--frame-path",
                        help=f"where the agent publishes (default {DEFAULT_FRAME_PATH})")
    parser.add_argument("--goal-url",
                        help="wss://... for the goal tunnel (default: $AGENT_GOAL_WS_URL)")
    parser.add_argument("--goal-dir",
                        help=f"where goals queue for ROCKET-2 (default {DEFAULT_GOAL_DIR})")
    parser.add_argument("--status-dir",
                        help=f"where ROCKET-2 leaves goal statuses to send upstream "
                             f"(default {DEFAULT_STATUS_DIR})")
    parser.add_argument("--no-speak", dest="speak", action="store_false",
                        help="print the model's replies without saying them")
    parser.add_argument("--tts-voice",
                        help="Moonshine TTS voice id (default piper_en_US-lessac-medium)")
    parser.add_argument("--no-mute", dest="mute", action="store_false",
                        help="keep listening while a reply plays (for headphones)")
    parser.add_argument("--no-cue", dest="cue", action="store_false",
                        help="do not play the let-me-do-it-for-you clip when a task starts")
    parser.add_argument("--no-conversation", dest="conversation", action="store_false",
                        help="do not print what the model says back")
    parser.add_argument("--timing", action="store_true",
                        help="print the raw components behind end->sent")
    parser.add_argument("--audio-log", action="store_true",
                        help="let PortAudio's probe chatter through (normally hidden)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        list_devices()
        return 0

    config = ClientConfig.from_env()
    if args.url:
        config.url = args.url.rstrip("/")
    if args.token:
        config.token = args.token
    if args.arch:
        config.arch = args.arch
    if args.device:
        config.device = args.device
    if args.update_interval:
        config.update_interval = args.update_interval
    if args.join_window is not None:
        config.join_window = args.join_window
    if args.vad_window is not None:
        config.vad_window = args.vad_window
    if args.vad_threshold is not None:
        config.vad_threshold = args.vad_threshold
    if args.frame_interval:
        config.frame_interval = args.frame_interval
    if args.frame_path:
        config.frame_path = args.frame_path
    if args.goal_url:
        config.goal_url = args.goal_url.rstrip("/")
    if args.goal_dir:
        config.goal_dir = args.goal_dir
    if args.status_dir:
        config.status_dir = args.status_dir
    config.conversation = args.conversation
    config.speak = config.speak and args.speak
    config.mute_while_speaking = config.mute_while_speaking and args.mute
    if args.tts_voice:
        config.tts_voice = args.tts_voice
    if not args.cue:
        config.cue = ""
    config.frames = config.frames and args.frames
    config.partial = args.partial
    config.keyterms = args.keyterms
    config.timing = args.timing
    config.audio_log = config.audio_log or args.audio_log
    config.check()

    try:
        if args.check:
            return asyncio.run(probe(config))
        return asyncio.run(run(config, args.text))
    except KeyboardInterrupt:
        print("stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
