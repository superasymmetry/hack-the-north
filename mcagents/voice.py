"""Speech in, streamed reply out: Moonshine v2 on the CPU, the LLM on the GPU box.

The third goal channel, next to ROCKET-2's point and JarvisVLA's sentence: a person
talking. The sentence controller already takes English, so the only missing piece is
turning a microphone into that English fast enough that a reply feels like a reply.

    python -m mcagents.voice                  # talk; watch the model answer
    python -m mcagents.voice --no-llm         # transcription only, no server needed
    python -m mcagents.voice --say "..."      # no microphone, just the LLM leg

Words appear on a rewriting line as they are recognised, revised in place while you are
still speaking -- "mine the die" becomes "mine the diamond ore" -- and settle into a `you:`
line when the phrase ends. That is Moonshine's streaming encoder made visible, and it is
also the honest picture of what the LLM is about to be given. `--no-partial` turns it off;
`Listener(on_partial=...)` is the same stream for anything that is not a terminal.

Nothing here imports minestudio, cv2 or torch, so it runs on a laptop with no Minecraft
and no GPU -- which is what makes it testable on its own while a rollout holds the GPU.

To drive something else with it, poll a Listener from that thing's own loop:

    listener = Listener().start()             # load() first if the boot order matters
    while agent.busy:
        agent.step()
        said = listener.poll()                # None until a phrase is *finished*
        if said:
            agent.set_goal(said, ...)

`poll()` never blocks, which is the property that lets it sit inside ROCKET-2's ~33 FPS
loop without costing frames. See "Running next to ROCKET-2" below.

Why Moonshine and not Whisper. Whisper's encoder is fixed at 30 s and zero-pads whatever
you give it, so a two-second phrase costs the same compute as a half-minute one -- a
constant floor you cannot design around, and it lands squarely on the one number a
conversation is judged by. Moonshine v2 processes the audio's actual length and caches
encoder and decoder state *while the person is still talking*, so most of the work is
already done when they stop (arXiv:2602.12241). It is also small: the wheel is 20 MB, the
runtime deps are numpy and sounddevice, and SMALL_STREAMING is 123M parameters at ~18% of
one core. Nothing touches CUDA.

Three things go wrong quietly:

1. **No PortAudio.** `sounddevice` is a binding, not a library; without libportaudio the
   import raises before any model loads. `conda install -c conda-forge portaudio`.
2. **A model download inside the run.** `load()` blocks and the first call fetches weights.
   Call it *before* a 30s Minecraft boot, not during -- same reason JarvisVLAClient probes
   the server before the sim starts.
3. **Both models on one vLLM.** JarvisVLA calls the server once per environment step,
   serially, and a chat completion is hundreds of tokens of decode against its eight. Share
   a server and the rollout stutters whenever anyone speaks. Serve the chat model
   separately and point VOICE_LLM_URL at it.

Running next to ROCKET-2: the ASR is CPU-only and the policy is GPU-only, so they do not
compete for the 8 GB the sim is also rendering into. What they do share is the CPU, and
Moonshine's own audio and worker threads are outside the GIL for most of their work.
"""
import argparse
import contextlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence

import numpy as np
import requests

#: One ONNX Runtime thread per model, not one per core. The multi-threaded default spins every
#: core while it waits for the next 300 ms update: measured here, TINY_STREAMING burned ~75
#: CPU-seconds on a 5 s clip, against ~1 s single-threaded -- and finalised *faster* and more
#: steadily (~75 ms against 150-500 ms), because it was no longer fighting the sim, ROCKET-2
#: and its own PortAudio callback for cores. That fight is what "input overflow" was.
#: Read when the native library creates a session, so it has to be set before load();
#: setdefault, so MOONSHINE_ORT_SINGLE_THREAD=0 in the environment still opts back out.
os.environ.setdefault("MOONSHINE_ORT_SINGLE_THREAD", "1")

#: Bias the decoder towards the vocabulary the game is about. Streaming models take these
#: as a prior over the next token, which is most of the difference between "mine the
#: diamond ore" and "mine the diamond or". Swap them per scene with Listener.keyterms().
MINECRAFT_KEYTERMS = [
    "Minecraft", "cobblestone", "oak log", "birch log", "spruce", "sapling", "furnace",
    "crafting table", "diamond ore", "iron ore", "coal ore", "redstone", "obsidian",
    "pickaxe", "shovel", "sword", "torch", "chest", "creeper", "zombie", "skeleton",
    "enderman", "pig", "cow", "sheep", "villager", "inventory", "hotbar", "biome",
]

DEFAULT_SYSTEM = (
    "You are the voice of a Minecraft agent. The user is speaking to you out loud, so "
    "their words arrive through speech recognition and may be slightly wrong -- read "
    "through obvious mishearings. Answer in one or two short sentences."
)


@dataclass
class VoiceConfig:
    #: SMALL_STREAMING is the sweet spot: 7.84% WER at ~148 ms, against TINY's 12.01% and
    #: MEDIUM's 29%-of-a-core. The names come from moonshine_voice.ModelArch.
    arch: str = "SMALL_STREAMING"
    language: str = "en"
    #: Seconds between streaming updates. Lower means partial text lands sooner and the
    #: model runs more often; it does not change when a *phrase* ends, which the library
    #: decides from the audio.
    update_interval: float = 0.5
    #: None lets PortAudio pick the default input; an int or a substring of the name picks
    #: one out of `python -m sounddevice`.
    device: Optional[str] = None
    keyterms: List[str] = field(default_factory=lambda: list(MINECRAFT_KEYTERMS))
    #: Let PortAudio's probe chatter through. Off because opening the input prints a dozen
    #: lines of ALSA complaint about a sample rate the library then handles by itself, and
    #: they land *after* the "listening" prompt, which makes a working mic look broken.
    audio_log: bool = False
    #: How long speech has to be gone before a phrase is over -- Moonshine averages its VAD
    #: over this window, so it is a floor on end-of-speech -> final. None keeps the library's
    #: 0.5 s. Shorter ends phrases sooner, and also splits one at a thinking pause.
    vad_window: Optional[float] = None
    #: VAD probability that counts as speech; None keeps the library's 0.5. Higher ends a
    #: phrase sooner on a trailing breath, and starts clipping quiet words.
    vad_threshold: Optional[float] = None

    #: Deliberately *not* JARVISVLA_URL: sharing one server with a once-per-step policy is
    #: failure mode 3 above. Left unset it still points at the usual port, so a single-server
    #: setup works out of the box and only pays for it under load.
    llm_url: str = "http://127.0.0.1:8000/v1"
    llm_model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 200
    timeout: float = 60.0
    system: str = DEFAULT_SYSTEM
    #: Turns kept as context. Two is enough for "and then what?" without growing prefill.
    history_turns: int = 4

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        return cls(
            arch=os.environ.get("VOICE_ARCH", cls.arch),
            language=os.environ.get("VOICE_LANGUAGE", cls.language),
            update_interval=float(os.environ.get("VOICE_UPDATE_INTERVAL", cls.update_interval)),
            device=os.environ.get("VOICE_DEVICE") or None,
            audio_log=os.environ.get("VOICE_AUDIO_LOG", "0") != "0",
            vad_window=float(os.environ["VOICE_VAD_WINDOW"])
            if os.environ.get("VOICE_VAD_WINDOW") else None,
            vad_threshold=float(os.environ["VOICE_VAD_THRESHOLD"])
            if os.environ.get("VOICE_VAD_THRESHOLD") else None,
            llm_url=os.environ.get("VOICE_LLM_URL", cls.llm_url).rstrip("/"),
            llm_model=os.environ.get("VOICE_LLM_MODEL") or None,
            temperature=float(os.environ.get("VOICE_TEMPERATURE", cls.temperature)),
            max_tokens=int(os.environ.get("VOICE_MAX_TOKENS", cls.max_tokens)),
            timeout=float(os.environ.get("VOICE_TIMEOUT", cls.timeout)),
        )


@contextlib.contextmanager
def _quiet_audio(quiet: bool = True):
    """Swallow PortAudio's chatter, which is written from C and never touches sys.stderr.

    Opening the input makes ALSA object at length that the card will not do 16 kHz -- true,
    and handled: Moonshine falls back to 48 kHz and resamples. Only the file descriptor can
    be moved, because the writes come from inside the C library.

    Exceptions are unaffected, and the pipeline's own errors arrive through on_error rather
    than fd 2, so this hides explanation, not failure. VOICE_AUDIO_LOG=1 keeps it.
    """
    if not quiet:
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def _as_device(value):
    """VOICE_DEVICE is a string, but sounddevice reads a bare string as a *name* substring."""
    if value is None or isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.lstrip("-").isdigit() else text or None


def pipewire_default_input() -> Optional[str]:
    """The ALSA name -- "hw:1,6" -- of the source PipeWire is actually using, or None.

    This is the fix for the failure that costs the most time here: conda-forge's PortAudio
    is built with only the ALSA host API, so it cannot see PulseAudio or PipeWire at all.
    Its "default input" is therefore just the first ALSA capture device on the lowest card,
    which on a laptop is the analog headset jack -- silent with nothing plugged into it --
    while the built-in microphone array sits on a later device that PipeWire has already
    selected and PortAudio knows nothing about.

    So ask PipeWire directly and translate. Absent wpctl, the caller falls back to listening
    to each device in turn.
    """
    try:
        output = subprocess.run(["wpctl", "inspect", "@DEFAULT_AUDIO_SOURCE@"],
                                capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = dict(re.findall(r'^\s*\*?\s*(alsa\.(?:card|device))\s*=\s*"([^"]*)"',
                            output, re.MULTILINE))
    card, device = found.get("alsa.card"), found.get("alsa.device")
    return f"hw:{card},{device}" if card and device else None


#: How long to listen to a device before judging it, and how much of that to throw away.
#: A digital microphone array takes about half a second to come up, and measured over a
#: shorter window than this one it reads as dead: 0.3s of the live mic on this laptop peaks
#: at 0.0004, against 0.037 over a full second. Getting this wrong turns the probe into a
#: confident, wrong answer, which is worse than not having it.
PROBE_SECONDS = 1.2
PROBE_WARMUP = 0.5


def input_level(index: int, seconds: float = PROBE_SECONDS) -> float:
    """Peak amplitude on one input. Room noise is enough to tell a live mic from a dead jack.

    Measured on this laptop with nobody speaking: the empty headset jack peaks around
    0.0006, the microphone array around 0.04 -- near enough two orders of magnitude.
    """
    import numpy as np
    import sounddevice as sd

    try:
        with _quiet_audio():
            data = sd.rec(int(seconds * 48000), samplerate=48000, channels=1,
                          device=index, dtype="float32")
            sd.wait()
        return float(np.abs(data[int(PROBE_WARMUP * 48000):]).max())
    except Exception:
        return 0.0                               # a device that will not open is not a candidate


def input_devices() -> List[tuple]:
    """(index, name) for everything that can capture."""
    import sounddevice as sd
    return [(index, device["name"]) for index, device in enumerate(sd.query_devices())
            if device["max_input_channels"] > 0]


def choose_input(explicit=None, report=None) -> Optional[object]:
    """Which device to open: what was asked for, what PipeWire uses, or whatever has signal.

    Returns None to mean "let PortAudio decide", which is right when there is only one
    candidate and wrong often enough otherwise that it is the last resort, not the first.
    """
    say = report or (lambda message: None)
    if explicit is not None:
        return explicit

    devices = input_devices()
    if len(devices) < 2:
        return None                              # nothing to get wrong

    wanted = pipewire_default_input()             # e.g. "hw:1,6"
    if wanted:
        for index, name in devices:
            if wanted in name:
                say(f"input: [{index}] {name} -- PipeWire's default source")
                return index
        say(f"input: PipeWire uses {wanted}, which PortAudio does not list; listening instead")

    # Nothing authoritative to go on, so listen to each in turn and take the one that is
    # plainly alive. "Plainly" is the point: a close call means guessing, and a wrong guess
    # here is silent, so leave it to PortAudio and say so.
    levels = sorted(((input_level(index), index, name) for index, name in devices),
                    reverse=True)
    best, runner_up = levels[0], levels[1]
    if best[0] > 0.002 and best[0] > 20 * runner_up[0]:
        say(f"input: [{best[1]}] {best[2]} -- the only one with signal "
            f"(peak {best[0]:.4f} against {runner_up[0]:.4f})")
        return best[1]

    say("input: PortAudio's default, which may be the wrong one -- "
        "`--list-devices` measures them, `--device N` picks one")
    return None


class _LevelQueue(queue.Queue):
    """MicTranscriber's audio queue, reporting each block's loudness as it goes in."""

    def __init__(self, on_level):
        super().__init__()
        self._on_level = on_level

    def put(self, item, block=True, timeout=None):
        if isinstance(item, tuple) and item and len(item[0]):
            try:
                samples = item[0]
                self._on_level(float(np.sqrt(np.mean(samples * samples))))
            except Exception:
                pass                             # never at the cost of the audio itself
        super().put(item, block, timeout)


class Listener:
    """The microphone side: finished phrases onto a queue, nothing blocking.

    Moonshine runs its own audio callback and worker thread, so this owns no thread of its
    own -- it is the queue between their callbacks and whatever loop wants the words. A
    phrase is handed over when the library marks the line complete, which it decides from
    the audio; there is no silence timeout here to tune.

        with Listener() as listener:
            while True:
                said = listener.poll(timeout=0.1)
    """

    def __init__(self, config: Optional[VoiceConfig] = None,
                 on_partial=None, report=None, on_final=None, on_start=None,
                 on_level=None):
        self.config = config or VoiceConfig.from_env()
        #: Where device selection explains itself. Silent by default, because a library
        #: should not print; the CLI passes one in.
        self.report = report or (lambda message: None)
        self.lines: "queue.Queue[str]" = queue.Queue()
        self.latency_ms: Optional[float] = None      # of the last finished phrase
        self._on_partial = on_partial
        #: The whole TranscriptLine, for callers that need more than its text --
        #: `start_time`/`duration` place the phrase on the audio's own clock, which is the
        #: only way to time anything against end-of-speech rather than against delivery.
        #: Called on Moonshine's worker thread, before the text reaches poll().
        self._on_final = on_final
        #: A new line has begun -- the earliest sign that speech resumed after a phrase ended.
        self._on_start = on_start
        #: RMS of each captured block, on PortAudio's callback thread -- keep it trivial.
        self._on_level = on_level
        self._mic = None

    def load(self) -> "Listener":
        """Fetch and open the model. Blocks, and the first call downloads -- see failure 2."""
        from moonshine_voice import ModelArch
        from moonshine_voice.mic_transcriber import MicTranscriber

        mic = (MicTranscriber()
               .language(self.config.language)
               .model_arch(getattr(ModelArch, self.config.arch))
               .update_interval(self.config.update_interval)
               .on_line(self._line)
               .on_error(lambda exc: print(f"[voice] {exc}", file=sys.stderr)))
        device = choose_input(_as_device(self.config.device), report=self.report)
        if device is not None:
            mic.device(device)
        if self._on_partial:
            mic.on_text(self._on_partial)
        if self._on_level and isinstance(getattr(mic, "_audio_queue", None), queue.Queue):
            # Moonshine has no level callback, but every captured block passes through this
            # one queue on its way from the audio callback to the worker, so the queue is
            # where to listen. A private attribute: if a later version renames it, the
            # isinstance check fails and levels are simply not reported.
            mic._audio_queue = _LevelQueue(self._on_level)
        if self._on_start:
            from moonshine_voice.transcriber import LineStarted
            on_start = self._on_start
            mic.add_listener(lambda event: on_start() if isinstance(event, LineStarted) else None)
        options = {name: str(value) for name, value in (
            ("vad_window_duration", self.config.vad_window),
            ("vad_threshold", self.config.vad_threshold)) if value is not None}
        if options:
            mic.options(options)

        try:
            with _quiet_audio(not self.config.audio_log):
                mic.load()
        except OSError as exc:                       # PortAudio is missing, or no input
            raise SystemExit(f"cannot open audio ({exc}).\n"
                             "conda install -c conda-forge portaudio, and check "
                             "`python -m sounddevice` lists an input.") from exc
        if self.config.keyterms:
            mic.set_keyterms(self.config.keyterms)   # streaming architectures only
        self._mic = mic
        return self

    def start(self) -> "Listener":
        if self._mic is None:
            self.load()
        with _quiet_audio(not self.config.audio_log):
            self._mic.start()
        return self

    def _line(self, line) -> None:
        text = line.text.strip()
        if text:
            self.latency_ms = line.last_transcription_latency_ms
            if self._on_final is not None:
                self._on_final(line)             # first: the caller may be timing this
            self.lines.put(text)

    def poll(self, timeout: float = 0.0) -> Optional[str]:
        """The next finished phrase, or None. Default is a pure non-blocking peek."""
        try:
            return self.lines.get(timeout=timeout) if timeout else self.lines.get_nowait()
        except queue.Empty:
            return None

    def keyterms(self, terms: Sequence[str]) -> None:
        """Re-bias the decoder mid-run -- for whatever the agent is currently looking at."""
        if self._mic is not None:
            self._mic.set_keyterms(list(terms))

    def mute(self, muted: bool = True) -> None:
        """Stop feeding audio without tearing the model down -- e.g. while the agent talks."""
        if self._mic is not None:
            self._mic.mute(muted)

    def close(self) -> None:
        if self._mic is not None:
            self._mic.close()
            self._mic = None

    def __enter__(self) -> "Listener":
        return self if self._mic is not None else self.start()

    def __exit__(self, *exc) -> None:
        self.close()


class Responder:
    """The vLLM side: one streamed chat completion per phrase, with a little history.

    Streamed because the whole point is that the first word comes back before the last one
    is generated; a 200-token reply that arrives all at once has thrown away most of what
    the fast ASR bought. One pooled connection, for the reason documented in
    JarvisVLAClient: through `ssh -L` a fresh TCP connection is a new channel negotiated
    with the login node before any bytes move.
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig.from_env()
        self._session = requests.Session()
        self.history: List[dict] = []
        self.model = self.config.llm_model or self._discover()

    def _discover(self) -> str:
        url = f"{self.config.llm_url}/models"
        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()["data"][0]["id"]
        except Exception as exc:
            raise SystemExit(
                f"cannot reach a vLLM server at {url} ({exc}).\n"
                "Start one, tunnel to it (scripts/connect.sh), and export VOICE_LLM_URL.\n"
                "Or run transcription on its own: python -m mcagents.voice --no-llm"
            ) from exc

    def ask(self, text: str) -> Iterator[str]:
        """Stream the reply to one utterance, token by token, and keep it in history."""
        messages = ([{"role": "system", "content": self.config.system}]
                    + self.history + [{"role": "user", "content": text}])
        response = self._session.post(
            f"{self.config.llm_url}/chat/completions",
            json={"model": self.model, "messages": messages, "stream": True,
                  "temperature": self.config.temperature,
                  "max_tokens": self.config.max_tokens},
            timeout=self.config.timeout, stream=True)
        response.raise_for_status()

        reply = []
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            payload = raw[len("data: "):]
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0]["delta"].get("content")
            if delta:
                reply.append(delta)
                yield delta

        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": "".join(reply)}]
        del self.history[:-2 * self.config.history_turns or None]


class LiveLine:
    """One rewritable line at the bottom of the terminal, for text that is still changing.

    Moonshine revises a phrase as it hears more of it -- "mine the die" becomes "mine the
    diamond ore" -- so in-progress text has to overwrite itself rather than scroll. Two
    things that look like details and are not:

    * A revision can be *shorter* than what it replaces, so the tail of the old line has to
      be painted over. `\r` alone leaves the difference on screen as debris.
    * It arrives on Moonshine's worker thread while the main loop is printing replies to the
      same terminal. `pause()` holds the line back rather than interleaving with them.
    """

    def __init__(self, prefix: str = "  ... "):
        self.prefix = prefix
        self._painted = 0                        # characters currently on screen
        self._paused = False
        self._lock = threading.Lock()

    def show(self, text: str) -> None:
        with self._lock:
            if self._paused:
                return
            room = shutil.get_terminal_size((100, 24)).columns - len(self.prefix) - 1
            if len(text) > room:                 # keep the newest words, not the oldest
                text = "\u2026" + text[-(room - 1):]
            body = self.prefix + text
            print("\r" + body + " " * max(0, self._painted - len(body)), end="", flush=True)
            self._painted = len(body)

    def clear(self) -> None:
        with self._lock:
            if self._painted:
                print("\r" + " " * self._painted + "\r", end="", flush=True)
                self._painted = 0

    @contextlib.contextmanager
    def pause(self):
        """Keep the line off the screen while something else writes to the terminal."""
        self.clear()
        with self._lock:
            self._paused = True
        try:
            yield
        finally:
            with self._lock:
                self._paused = False


class _Silent:
    """Stand-in for a Listener when there is no microphone, so --say shares answer()."""
    latency_ms = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-llm", dest="llm", action="store_false",
                        help="transcribe only; do not contact a server")
    parser.add_argument("--say", metavar="TEXT",
                        help="skip the microphone and send one line to the LLM")
    parser.add_argument("--arch", help="TINY_STREAMING | SMALL_STREAMING | MEDIUM_STREAMING")
    parser.add_argument("--device", help="input device index or name substring")
    parser.add_argument("--list-devices", action="store_true",
                        help="show every input with its measured level, then exit")
    parser.add_argument("--no-partial", dest="partial", action="store_false",
                        help="do not show in-progress text, only finished phrases")
    parser.add_argument("--audio-log", action="store_true",
                        help="let PortAudio's probe chatter through (it is normally hidden)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    config = VoiceConfig.from_env()

    if args.list_devices:   # needs no server and no model
        wanted = pipewire_default_input()
        print(f"PipeWire's default source is {wanted or 'unknown'}. Levels are peak "
              f"amplitude over {PROBE_SECONDS - PROBE_WARMUP:.1f}s of whatever the room "
              f"is doing -- speak, and they should jump:\n")
        for index, name in input_devices():
            level = input_level(index)
            mark = "*" if wanted and wanted in name else " "
            verdict = "live" if level > 0.002 else "silent -- nothing plugged in?"
            print(f" {mark} [{index}] {name:<34} peak {level:.4f}  {verdict}")
        print("\nPick one with --device N (or VOICE_DEVICE).")
        return

    if args.arch:
        config.arch = args.arch
    if args.device:
        config.device = args.device
    if args.audio_log:
        config.audio_log = True

    responder = Responder(config) if args.llm else None
    if responder:
        print(f"Serving {responder.model} at {config.llm_url}.")

    live = LiveLine()

    def answer(text: str) -> None:
        # Held for the whole reply: the microphone is still live, and a phrase revised
        # mid-stream would otherwise redraw itself straight through the model's answer.
        with live.pause():
            asr = f"   ({listener.latency_ms:.0f} ms)" if listener.latency_ms else ""
            print(f"  you: {text}{asr}")
            if not responder:
                return
            started = time.time()
            first = None
            print("  llm: ", end="", flush=True)
            for token in responder.ask(text):
                first = first if first is not None else time.time() - started
                print(token, end="", flush=True)
            print(f"\n       ({first * 1000:.0f} ms to first token)\n" if first else "")

    if args.say:                                     # no microphone, just the LLM leg
        listener = _Silent()                         # nothing has been heard, so no latency
        return answer(args.say)

    # Load *and* start before the greeting, so the prompt is the last line on the screen.
    # The first run downloads the model, and opening the device is where the audio stack has
    # its say; a prompt printed before either has finished is a prompt that gets buried.
    listener = Listener(config, on_partial=live.show if args.partial else None,
                        report=lambda message: print(message, flush=True)).start()
    print(f"Listening with {config.arch}. Speak; ctrl-c to stop.\n", flush=True)

    with listener:
        try:
            while True:
                said = listener.poll(timeout=0.2)
                if said:
                    answer(said)
        except KeyboardInterrupt:
            live.clear()
            print("stopped.")


if __name__ == "__main__":
    main()
