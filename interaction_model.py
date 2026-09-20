from __future__ import annotations

import argparse
import base64
import io
import json
import math
import queue
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from openai import OpenAI

# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #

# Event kinds. Everything the model ever sees is one of these.
SPEECH_FINAL = "speech.final"        # user said something (typed, for now)
VISION_FRAME = "vision.frame"        # a screenshot
GAME_STATE = "game.state"            # health, hunger, biome, time, nearby mobs
AGENT_SAID = "agent.said"            # what actually reached the user
GOAL_SENT = "goal.sent"              # a plan entry shipped down tunnel 2
CLOCK = "clock"                      # elapsed-time marker; no longer emitted
BG_DISPATCHED = "background.dispatched"
BG_RESULT = "background.result"
TASK_DONE = "task.done"              # the standing instruction was carried out
LATENCY = "latency"                  # per-stage timings, for the budget ledger
ARRIVAL_CHECK = "arrival.check"      # one per new frame while a goal runs, pass or fail
GOAL_STOPPED = "goal.stopped"        # a stop was sent, retried, confirmed or given up on


@dataclass
class Event:
    kind: str
    t: float                          # seconds since session start
    text: str = ""
    image: Optional[str] = None       # base64 jpeg, only for VISION_FRAME
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        if d["image"]:                # don't dump megabytes into the log
            d["image"] = f"<{len(d['image'])} b64 chars>"
        return d


class Timeline:
    """Append-only, monotonically timestamped log. The single source of truth.

    What the model reads, what ships to the background model, and -- as JSONL --
    the replay harness, latency ledger, and future SFT data.
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self._events: list[Event] = []
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._fh = log_path.open("a") if log_path else None

    def now(self) -> float:
        return time.monotonic() - self._t0

    def append(self, kind: str, text: str = "", image: Optional[str] = None, **meta) -> Event:
        ev = Event(kind, self.now(), text, image, meta)
        with self._lock:
            self._events.append(ev)
        if self._fh:
            self._fh.write(json.dumps(ev.to_json()) + "\n")
            self._fh.flush()
        return ev

    def snapshot(self) -> list[Event]:
        with self._lock:
            return list(self._events)


@dataclass
class Task:
    """What the agent is working on right now, and nothing else.

    The Timeline stays the append-only record of everything; this is the narrow
    working set the model actually reads. Scoped to a single instruction on
    purpose. An unbounded history renders as a chat transcript, and an instruct
    model continues a chat transcript whether or not anyone is there -- one
    logged session produced 44 consecutive self-replies against zero user input,
    each one a plausible continuation of the last.

    `instruction is None` is the idle state, and it means nothing runs at all --
    no gate call, no speak call. It is reached two ways: nothing has been asked
    yet, or what was asked is finished. Both should look identical to the model,
    which is why finishing *removes* the instruction rather than annotating it.
    """

    instruction: Optional[str] = None
    #: What the agent has done since this instruction arrived. Evidence for the
    #: completion gate, and the only memory that survives inside a task.
    actions: list[str] = field(default_factory=list)
    started_at: float = 0.0

    @property
    def idle(self) -> bool:
        return self.instruction is None

    def assign(self, text: str, now: float) -> None:
        """A new instruction supersedes the old one, done or not -- the person
        asking for something else is the clearest possible signal about what
        matters now."""
        self.instruction = text
        self.actions = []
        self.started_at = now

    def clear(self) -> None:
        self.instruction = None
        self.actions = []


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are playing Minecraft together with the user. You see their game \
screen and hear them talk. You are the one who thinks about what to do next in the game -- \
read the screen, track the goal, decide the next move -- while they handle the controls.

Talk like a friend on voice chat: one or two short spoken sentences, plain words, no markdown \
and no lists, because everything you say is read aloud verbatim. Name the specific thing you \
can see and what to do about it. Never generic encouragement -- "keep going" and "nice work" \
are worse than silence.

Only name items, blocks and mobs that actually appear in the [game] line or on screen. If \
they are holding a stone sword, do not tell them to raise a shield. If you are not sure what \
they have, say what to do in terms of what you can see.

[game] lines are the live game state -- read them before the pixels, they are ~100x cheaper \
and never ambiguous. [Ns silence] is real elapsed time. [background result: ...] is planning \
you asked for earlier; work it into the conversation when it fits."""

# The speak/no-speak decision deliberately lives OUTSIDE the system prompt. See `Gate`.
#
# MEASURED, on Qwen2.5-VL-7B over 10 labelled Minecraft timeline states (separation =
# min P(speak | should speak) - max P(speak | should stay quiet); positive means some
# threshold works):
#
#     "is what I see worth mentioning"  (salience)  +0.200   <- usable
#     "is now the right moment"         (timing)    -0.272   <- unusable
#
# The model can judge whether something on screen is worth a remark. It cannot judge
# conversational timing -- mid-sentence, already-answered, and reminder-due all score
# indistinguishably. Nine prompt variants were tried; none fixed the timing half, because
# the article's model gets that from training on streams where silence is a valid
# continuation and Qwen never saw one.
#
# So this gate is only trustworthy for the salience question. Timing is structural and
# belongs in rules: new user input, a due reminder, a landed background result.
GATE_SPEAK = """Decide whether to speak on this tick of a Minecraft session.

Examples:
- user just asked "where's iron?" and you have not answered -> yes
- a creeper is next to them and they seem not to have noticed -> yes
- health dropped to 4 in a cave -> yes
- a reminder they asked for is due this second -> yes
- they are halfway through a sentence -> no
- you answered 2 seconds ago and nothing changed -> no
- they are quietly mining and nothing new happened -> no
- you would only be saying "keep going" or "nice work" -> no

Given the session above, should you speak on this tick?"""

# Asked once per fresh utterance, about the utterance: a question about *what was said*,
# the salience kind (+0.200), not the timing kind (-0.272). Text only -- whether a line
# wants an answer is in the words, and dropping the frame is most of the prefill cost.
# UNMEASURED: label real finals from session.jsonl before trusting `--reply-threshold`.
GATE_REPLY = """Decide whether the user's latest line is addressed to you -- a question, a request, or an instruction you should respond to or act on.

Examples:
- "where's iron?" -> yes
- "go chop that tree" -> yes
- "what should we do next" -> yes
- "okay" acknowledging what you just said -> no
- "mm", "oh", "wait", "but," -> no
- they are thinking aloud or talking to someone else -> no
- a fragment of a sentence they have not finished -> no

Is the user's latest line addressed to you?"""

# Asked only of a line GATE_REPLY already let through. Without it every addressed
# line went to `plan`, whose schema requires a target and an interaction, so a
# question had to become a goal: "what do you see in this building" came back as
# Use 'the door' (session.jsonl, t=141), copied from the two replies before it,
# and it replaced the goal that was running. A question gets an answer and
# leaves the running goal alone. Text only, like GATE_REPLY. UNMEASURED; see
# `--question-threshold`. Measure it with `calibrate_question_gate.py` -- label
# lines from session.jsonl, then score and sweep the threshold against them.
GATE_QUESTION = """Decide whether the user's latest line only asks for information -- what you see, where something is, what something is -- rather than asking you to do or change anything in the game.

Answer no to anything that asks you to move, go somewhere, or act, however it is \
phrased -- naming a place you can see is still asking you to go there, not a \
question about it.

Examples:
- "what do you see in this building" -> yes
- "is there a chest in here?" -> yes
- "where are we" -> yes
- "how many logs do we have" -> yes
- "tell me if that's a village" -> yes
- "describe the building in front of you" -> yes
- "open the door" -> no
- "now exit this building" -> no
- "can you go to that tree" -> no
- "what should we do next" -> no
- "go to the white house in front of you" -> no
- "head over to that tower" -> no
- "walk to the water" -> no
- "get to the building on your left" -> no

Does the user's latest line only ask for information?"""

ANSWER_PROMPT = ("Answer the user's latest question from the screenshot and the [game] line, "
                 "in one or two short spoken sentences. Name what you actually see. If you "
                 "cannot tell from the screen, say so. Do not describe an action you are "
                 "taking and do not suggest one.")

# Spoken the moment an instruction is heard, while the planner is still working out
# the goal on its own thread. It cannot know the plan, so it only repeats back what
# was asked -- a target or route of its own could disagree with the goal that goes
# down tunnel 2 a few seconds later.
ACK_PROMPT = ("Acknowledge the user's latest line in one short first-person spoken sentence "
              "saying what you are about to do -- \"Okay, heading to the brick building\", "
              "\"On it, chopping that oak tree\". Name the target in their words; do not add a "
              "direction, route or target of your own, because the plan is still being worked "
              "out. If they asked something, answer it in one short sentence instead. Nothing "
              "else: no praise, no suggestions.")

REPLY_SYSTEM = "You are on voice chat with a friend who is playing Minecraft."
RECENT_LINES = 6

# Contrasting the two outcomes explicitly is what makes this one work; asking "does this
# need background work?" saturated at ~0.99 on everything, because to a chat model every
# request sounds like it deserves effort. Separation +0.069 -- correct ordering on all five
# labelled cases, but on a compressed scale, hence the low default threshold.
# The gate that makes idle reachable. Without it an instruction is permanent -- the
# agent keeps finding things to say about a job that ended ten minutes ago.
#
# UNMEASURED. The other two gates carry a separation number because they were run
# against labelled timeline states; this one has not been. It is closer in kind to
# the salience question (+0.200, usable) than to the timing one (-0.272, not) --
# "is the wood gathered" is a fact about the screen, not a judgement about
# conversational rhythm -- but that is an argument, not a measurement. Both failure
# modes are visible: too low and the agent goes mute mid-task, too high and it
# never stops. Measure before trusting `--done-threshold`.
GATE_DONE = """The user asked for something and you have been working on it.

Examples:
- they asked for wood and [done so far] shows the wood is gathered -> yes
- they asked where iron is and you have already told them -> yes
- they asked you to watch for creepers and the cave is behind you -> yes
- they asked for a shelter and only two walls are up -> no
- they asked for wood and nothing has been done yet -> no
- you answered part of what they asked and the rest is still open -> no

Given the session above, is the user's instruction finished?"""

GATE_DELEGATE = ("Two options for this request. NOW: you can answer it from what is on screen "
                 "in one or two sentences, or it needs a reaction this second. LATER: it needs "
                 "a multi-step plan spanning many minutes of gameplay -- a tech-tree route, a "
                 "full base design, a raid or dimension trip -- that would take too long to "
                 "think through while they wait. Is this a LATER request?")

def recent_dialogue(events: list[Event], n: int = RECENT_LINES) -> str:
    """The last `n` spoken lines, both sides, as plain quoted text.

    Text inside a user turn, never assistant turns -- see `render` for why.
    """
    who = {SPEECH_FINAL: "user", AGENT_SAID: "you"}
    said = [f"{who[ev.kind]}: {ev.text}" for ev in events if ev.kind in who]
    return "\n".join(said[-n:])


def render_reply(events: list[Event]) -> list[dict]:
    """Text-only context for GATE_REPLY: the recent conversation and nothing else."""
    return [{"role": "system", "content": REPLY_SYSTEM},
            {"role": "user", "content": "[recent conversation]\n" + recent_dialogue(events)}]


def latest_background(events: list[Event], since: float) -> str:
    """The newest background result that landed at or after `since`, else "".

    Results used to be logged and never rendered, so every delegation was
    computed and thrown away. Only the newest: an older one answered a question
    the newer one has already moved past.
    """
    for ev in reversed(events):
        if ev.t < since:
            break
        if ev.kind == BG_RESULT and not ev.text.startswith("(delegation failed"):
            return ev.text
    return ""


def render(task: Task, frame: Optional[str], game: str = "", recent: str = "",
           background: str = "") -> list[dict]:
    """The model's entire world: this screen, this instruction, what has been done.

    One user turn, and no assistant turns at all. That absence is the design.
    Rendering AGENT_SAID as an `assistant` message made the context a
    conversation whose most recent line was always the model's own, so the gate
    reliably found a conversation worth continuing and the loop fed itself. Here
    what the agent has done appears as a line of text *inside* the user turn:
    evidence about the world, not a turn inviting the next one.

    Only the newest frame is sent. Older frames described a screen that no longer
    exists, and the instruction plus the current screen is what deciding the next
    move actually needs.

    Callers must not reach this while `task.idle` -- there is no instruction to
    render, and main() skips the tick rather than asking the model about nothing.
    """
    parts: list[dict] = []
    if frame:
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
    if game:
        parts.append({"type": "text", "text": f"[game] {game}"})
    if recent:
        parts.append({"type": "text", "text": f"[recent conversation]\n{recent}"})
    if background:
        parts.append({"type": "text", "text": f"[background result: {background}]"})
    parts.append({"type": "text", "text": f"[user] {task.instruction}"})
    if task.actions:
        parts.append({"type": "text",
                      "text": "[done so far] " + "; ".join(task.actions)})
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": parts}]


def context_dump(events: list[Event]) -> str:
    """Flat text of the session, for the background model.

    The arrow in the diagram is labelled *context*, not *query*: the background
    model gets the conversation, not a distilled question.
    """
    role = {SPEECH_FINAL: "user: ", AGENT_SAID: "you: ", VISION_FRAME: "(screenshot)",
            BG_RESULT: "earlier background result: "}
    return "\n".join(f"[{ev.t:6.1f}s] {role[ev.kind]}{ev.text}"
                     for ev in events if ev.kind in role)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def stdin_lines() -> queue.Queue:
    """Typed stdin on a reader thread.

    Stands in for streaming ASR: swap this for a partial-transcript source and
    the loop is unchanged.
    """
    q: queue.Queue[str] = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            if line.strip():
                q.put(line.strip())

    threading.Thread(target=read, daemon=True).start()
    return q


# --------------------------------------------------------------------------- #
# Tunnels
# --------------------------------------------------------------------------- #
#
# Two sockets, both authenticated by the same `X-Agent-Token` header on the HTTP
# upgrade, both held open for the life of the session:
#
#   tunnel 1 (Uplink)   client -> us: finalized utterances AND agent-view frames,
#                       interleaved. us -> client: the spoken reply, and nothing else.
#   tunnel 2 (GoalLink) us -> client: plan entries for the Minecraft policy. The
#                       client sends nothing; we read and discard to keep it open.
#
# CLOSE CODES. 1008 is the client's "credentials refused" signal and stops it
# permanently -- it will not retry. So 1008 is reachable from exactly one place
# in this file: a token mismatch on the upgrade. Every other failure (a bad
# frame, a dead vLLM, this process restarting under Slurm) must surface as an
# ordinary drop, which the client retries with jittered backoff. Concretely:
# handlers swallow their own exceptions, and nothing here calls close() with a
# code of its own choosing.

#: Utterance delivery is at-least-once with no acks, so a socket that dies
#: mid-send gets that utterance again on the next connection. Re-assigning the
#: same instruction wipes `task.actions` and restarts work that was half done,
#: so identical finals inside this window are dropped.
DEDUPE_WINDOW_S = 60.0

#: The agent's full-resolution frame. We are shown 224x224 -- a different aspect
#: ratio, not just a smaller one -- so a pixel pair read off the image we saw
#: lands somewhere else entirely, and fails silently by mining the wrong block.
#: Nothing in this file ever emits pixels; see `validate_entry`.
FULL_W, FULL_H = 640, 360

#: What the person is physically pointing at, as the laptop's hand tracker had
#: it when the message left. `locked` is the whole point: the pinch was held long
#: enough to be a choice, not the cursor drifting across something. `held` flips
#: true once they let go and `age` counts the seconds since, so a pinch released
#: a while ago is no longer about what they are saying now.
SELECTION_MAX_AGE = 5.0

#: A selection goal carries no arrival rule of its own unless the instruction
#: named a count, and a goal with no stop never ends on the client. A bound, not
#: an arrival detector -- the same UNCALIBRATED caveat as `approach_step_cap`.
SELECTION_STEPS = 300

INTERACTIONS = {"Hunt", "Mine", "Use", "Interact", "Craft", "Switch", "Approach", "None"}
GOAL_KEYS = ("point", "instruction", "interaction", "stop", "normalized", "keep_memory")

#: Egocentric turns. "Turn right" / "turn around" name no target, so they cannot
#: go through `locate`, and INTERACTIONS has no rotation. The planner picks
#: interaction "Turn" with target left/right/around; it goes out as
#: {"turn": <yaw degrees>}, positive = clockwise (right).
#:
#: UNCONFIRMED on the client: it drops goals with unknown keys, so a turn does
#: nothing until the client applies `turn` as a direct camera yaw (not through
#: the policy or the targeter). Checked in order: "behind/around" before "left".
TURN_YAW = (("around", 180), ("behind", 180), ("back", 180),
            ("left", -90), ("right", 90))
PLAN_INTERACTIONS = INTERACTIONS | {"Turn"}


def _fractions(v, n: int) -> bool:
    """n numbers, each a 0..1 fraction of the frame. Pixels are never accepted
    anywhere in this file; see FULL_W."""
    return isinstance(v, (list, tuple)) and len(v) == n and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0 for x in v)


def selection_of(msg: dict) -> Optional[dict]:
    """The deliberate selection on an uplink message, or None.

    Either message kind may carry one. An unlocked selection is discarded: it is
    where the hand happens to be, not a choice. See SELECTION_MAX_AGE.
    """
    s = msg.get("selection")
    if not isinstance(s, dict) or not s.get("locked"):
        return None
    age = s.get("age")
    if s.get("held") and isinstance(age, (int, float)) and not isinstance(age, bool) \
            and age > SELECTION_MAX_AGE:
        return None                              # pinched, released, long over
    out = {k: [float(x) for x in s[k]]
           for k, n in (("point", 2), ("box", 4)) if _fractions(s.get(k), n)}
    return out or None


class _Link:
    """Shared plumbing: token check, client set, broadcast, serve-in-a-thread."""

    name = "link"

    def __init__(self, token: str = "") -> None:
        self.token = token
        self._lock = threading.Lock()
        self._clients: set = set()
        self._server = None

    # -- lifecycle ---------------------------------------------------------- #

    def serve(self, port: int, host: str = "127.0.0.1"):
        from websockets.sync.server import serve as ws_serve
        self._server = ws_serve(self._handler, host, port)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"[{self.name}] listening on ws://{host}:{port}", file=sys.stderr)
        return self

    def _handler(self, ws) -> None:
        if self.token and ws.request.headers.get("X-Agent-Token") != self.token:
            ws.close(1008, "bad token")          # the only 1008 in this file
            return
        peer = ws.remote_address[0] if ws.remote_address else "?"
        print(f"[{self.name}] client connected from {peer}", file=sys.stderr)
        with self._lock:
            self._clients.add(ws)
        try:
            for raw in ws:                        # holds the socket open
                try:
                    self.on_message(raw)
                except Exception as e:            # a bad message is not a bad socket
                    print(f"[{self.name}] dropped a message: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[{self.name}] connection ended: {e}", file=sys.stderr)
        finally:
            with self._lock:
                self._clients.discard(ws)
            print(f"[{self.name}] client disconnected", file=sys.stderr)

    def close(self, code: int = 1012, reason: str = "restarting") -> None:
        """Stop listening and hang up on everyone.

        1012 (service restart), not 1008: this process dying and coming back
        under Slurm is a transient drop the client is built to ride out, and
        1008 would retire the client permanently instead.
        """
        if code == 1008:
            raise ValueError("1008 stops the client for good; use it only for a bad token")
        with self._lock:
            clients, self._clients = list(self._clients), set()
        for ws in clients:
            try:
                ws.close(code, reason)
            except Exception:
                pass
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    def on_message(self, raw) -> None:
        """Default: read and discard. Keeps a receive-only socket alive."""

    @property
    def connected(self) -> int:
        with self._lock:
            return len(self._clients)

    def _broadcast(self, payload: str) -> int:
        with self._lock:
            clients = list(self._clients)
        sent = 0
        for ws in clients:
            try:
                ws.send(payload)
                sent += 1
            except Exception as e:
                print(f"[{self.name}] send failed: {e}", file=sys.stderr)
                with self._lock:
                    self._clients.discard(ws)
        return sent


class Uplink(_Link):
    """Tunnel 1. Utterances and frames up; the conversational reply down.

    Same contract as `stdin_lines` on the text side -- `finals()` yields
    finalized user utterances -- plus a single-slot frame buffer.

    Frames arrive every ~3s whether or not anyone is speaking, so they are NOT
    one-per-turn and must not be queued: an older frame describes a screen that
    no longer exists, which is exactly what `render` refuses to send. Newest wins,
    everything else is discarded on arrival.
    """

    name = "uplink"

    def __init__(self, token: str = "", dedupe_window: float = DEDUPE_WINDOW_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(token)
        self.dedupe_window = dedupe_window
        self._clock = clock
        self._q: queue.Queue[str] = queue.Queue()
        self._frame: Optional[str] = None
        self._frame_t = 0.0
        self._seen: dict[str, float] = {}
        self._selection: Optional[dict] = None    # newest pointing, from either kind
        self._selection_t = 0.0
        self._final_selection: Optional[dict] = None
        self.stats = {"frames": 0, "finals": 0, "partials": 0, "dupes": 0, "ignored": 0}

    # -- message handling (pure; the replay test drives this directly) ------- #

    def on_message(self, raw) -> None:
        """One uplink message. Branches on `kind` BEFORE touching anything else.

        Text messages carry no `kind`, so reaching for msg["text"] first would
        throw on every frame -- three times a minute, forever.
        """
        msg = raw
        if isinstance(msg, (bytes, bytearray)):
            msg = msg.decode("utf-8", "replace")
        if isinstance(msg, str):
            try:
                msg = json.loads(msg)
            except json.JSONDecodeError:
                msg = {"text": msg}               # bare prose is an utterance
        if not isinstance(msg, dict):
            self.stats["ignored"] += 1
            return

        # Pointing rides on frames and on utterances alike, so it is read before
        # the branch: a pinch seen on the frame 1s ago is still what "that" means.
        if (sel := selection_of(msg)) is not None:
            with self._lock:
                self._selection, self._selection_t = sel, self._clock()

        kind = msg.get("kind")
        if kind == "frame":
            self._put_frame(msg.get("image"))
            return
        if kind is not None:                       # some future message kind
            self.stats["ignored"] += 1
            return
        self._put_text(msg)

    def _put_frame(self, image) -> None:
        if not isinstance(image, str) or not image.strip():
            self.stats["ignored"] += 1
            return
        image = image.strip()
        if image.startswith("data:"):              # contract says no prefix; tolerate one
            image = image.split(",", 1)[-1]
        with self._lock:
            self._frame, self._frame_t = image, self._clock()
        self.stats["frames"] += 1

    def _put_text(self, msg: dict) -> None:
        text = str(msg.get("text") or "").strip()
        if not text:
            self.stats["ignored"] += 1
            return
        if not msg.get("final", True):
            self.stats["partials"] += 1            # what barge-in will hang off
            return
        if self._seen_recently(text):
            self.stats["dupes"] += 1
            return
        self.stats["finals"] += 1
        sel = selection_of(msg) or self._recent_selection()
        print(f"[{self.name}] heard: {text!r}" + (f" pointing at {sel}" if sel else ""),
              file=sys.stderr)
        self._q.put((text, sel))

    def _recent_selection(self) -> Optional[dict]:
        """The pointing from a message just before this one, if it is still warm.

        The laptop may put the selection on the frame rather than on the
        utterance, and both describe the same hand at nearly the same moment.
        """
        with self._lock:
            if self._selection is None \
                    or self._clock() - self._selection_t > SELECTION_MAX_AGE:
                return None
            return self._selection

    def _seen_recently(self, text: str) -> bool:
        now = self._clock()
        with self._lock:
            for k, t in list(self._seen.items()):
                if now - t > self.dedupe_window:
                    del self._seen[k]
            dup = text in self._seen
            self._seen[text] = now
        return dup

    # -- readers ------------------------------------------------------------ #

    def finals(self) -> list[str]:
        items = drain(self._q)
        if items:
            # main() acts on the last line only, so its selection is the one that
            # matters; snapshotted here so a message landing mid-tick can't swap it.
            self._final_selection = items[-1][1]
        return [text for text, _ in items]

    @property
    def selection(self) -> Optional[dict]:
        """What they were pointing at when they said the last line `finals`
        returned, or None. See `selection_of` and THE ONE RULE in `plan_entry`."""
        return self._final_selection

    def frame(self, max_age: Optional[float] = None) -> Optional[str]:
        with self._lock:
            if self._frame is None:
                return None
            if max_age is not None and self._clock() - self._frame_t > max_age:
                return None
            return self._frame

    # -- downlink ----------------------------------------------------------- #

    def say(self, text: str, reason: Optional[str] = None) -> int:
        """The reply for the person. Goal keys are silently ignored on this
        socket, so nothing but prose is ever sent here.

        `reason` ("arrived", "lost") marks a line the agent says on its own when
        a goal ends, so the laptop can tell it from an answer. Omitted otherwise,
        which keeps an ordinary reply exactly {"text": ...}."""
        text = (text or "").strip()
        if not text:
            return 0                               # sending nothing is valid
        msg = {"text": text} if reason is None else {"text": text, "reason": reason}
        return self._broadcast(json.dumps(msg))


def _validate_stop(stop):
    """(ok, normalized_stop). An unrecognised stop condition means the client
    rejects the whole goal, so we reject it here instead of shipping a no-op."""
    def _count(v):
        return isinstance(v, int) and not isinstance(v, bool) and v > 0

    if stop is None:
        return True, None
    if _count(stop):
        return True, stop                          # a bare step budget
    if not isinstance(stop, dict):
        return False, None
    keys = set(stop)
    if keys == {"steps"} and _count(stop["steps"]):
        return True, {"steps": stop["steps"]}
    if keys <= {"item", "count", "mode"} and {"item", "count"} <= keys:
        if isinstance(stop["item"], str) and stop["item"].strip() and _count(stop["count"]):
            out = {"item": stop["item"].strip(), "count": stop["count"]}
            if "mode" in stop:
                if stop["mode"] not in ("total", "delta"):
                    return False, None
                out["mode"] = stop["mode"]
            return True, out
        return False, None
    if keys <= {"stat", "match", "count"} and {"stat", "count"} <= keys:
        if isinstance(stop["stat"], str) and stop["stat"].strip() and _count(stop["count"]):
            out = {"stat": stop["stat"].strip(), "count": stop["count"]}
            if "match" in stop:
                if not isinstance(stop["match"], str) or not stop["match"].strip():
                    return False, None
                out["match"] = stop["match"].strip()
            return True, out
        return False, None
    return False, None


def validate_entry(entry) -> Optional[dict]:
    """A plan entry the client will actually run, or None.

    Rejecting here rather than at the client is the point: a malformed goal is
    dropped silently on the far side, and a goal with 224-space pixels is worse
    than dropped -- it runs, against the wrong block. So numeric points are
    accepted ONLY as normalized 0..1 fractions; anything else must be a
    description for the local vision targeter.

    `use_selection` is ours, never wire: it says the person pinched the target
    themselves, and strips the goal of every way of naming one. It is kept on the
    way out so that validating an already-validated entry -- which is exactly
    what `GoalLink.send` does to everything -- says the same thing twice;
    `GoalLink.send` drops it on the wire. See `plan_entry`.
    """
    if not isinstance(entry, dict):
        return None
    e = {k: v for k, v in entry.items() if k != "kind"}     # "kind" is stripped
    picked = bool(e.pop("use_selection", False))
    if "turn" in e:                                         # see TURN_YAW
        yaw = e["turn"]
        if set(e) != {"turn"} or not isinstance(yaw, (int, float)) \
                or isinstance(yaw, bool) or not (0 < abs(yaw) <= 180):
            return None
        return {"turn": int(round(yaw))}
    if not picked and not any(k in e for k in GOAL_KEYS):
        return None                                         # not a goal at all

    out: dict = {}
    if picked:
        # THE ONE RULE. Omitting `point` is what makes the client run against the
        # object the person pinched. A point of either kind -- fractions or a
        # description -- overrides that pick and discards their pointing, and
        # `instruction` is re-resolved by the same targeter, so neither goes out.
        e.pop("point", None)
        e.pop("instruction", None)
    point = e.get("point")
    if isinstance(point, str) and point.strip():
        out["point"] = point.strip()
    elif isinstance(point, (list, tuple)) and len(point) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in point):
        if not e.get("normalized"):
            return None                                     # the coordinate trap
        u, v = float(point[0]), float(point[1])
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None
        out["point"], out["normalized"] = [u, v], True
    elif point is not None:
        return None

    instruction = str(e.get("instruction") or "").strip()
    if "point" not in out:
        if not (picked or instruction):
            return None                                     # point or instruction, required
        if instruction:
            out["instruction"] = instruction
    elif instruction:
        out["instruction"] = instruction

    action = e.get("interaction", "Approach")
    if action not in INTERACTIONS:
        return None
    out["interaction"] = action

    ok, stop = _validate_stop(e.get("stop"))
    if not ok:
        return None
    out["stop"] = stop

    if "keep_memory" in e:
        out["keep_memory"] = bool(e["keep_memory"])
    if isinstance(e.get("id"), str) and e["id"].strip():
        out["id"] = e["id"].strip()                         # echoed back in goal.status
    if picked:
        out["use_selection"] = True                         # ours; stripped before sending
    return out


class GoalLink(_Link):
    """Tunnel 2. Plan entries down; nothing ever comes up.

    Goals are NOT buffered for a future connection. There is no dedupe on the
    client, so a goal replayed onto a reconnect runs a second time, minutes late,
    against a world that has moved on. If nobody is listening the goal is dropped
    and the fact is logged.
    """

    name = "goals"

    def __init__(self, token: str = "", stop_goal: Optional[dict] = None) -> None:
        super().__init__(token)
        self.stop_goal = dict(stop_goal or STOP_GOAL)
        self.stats = {"sent": 0, "rejected": 0, "undeliverable": 0}
        self.last_status: Optional[dict] = None
        self._n = 0

    def _next_id(self) -> str:
        """Every goal goes out with an id, so `goal.status` coming back can be
        matched to what caused it rather than to whatever ran most recently."""
        with self._lock:
            self._n += 1
            return f"g{self._n}"

    def on_message(self, raw) -> None:
        """`goal.status` from the client: started, ended or rejected.

        Logged, not acted on: arrival and halting are still judged from frames by
        `GoalMonitor`. "rejected" is the one to watch -- it is what a goal that
        expected a selection the client no longer has looks like.
        """
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict) or msg.get("kind") != "goal.status":
            return
        self.last_status = msg
        event = str(msg.get("event") or "?")
        self.stats[f"status.{event}"] = self.stats.get(f"status.{event}", 0) + 1
        print(f"[goals] status {event} id={msg.get('id')} "
              f"reason={msg.get('reason')!r}", file=sys.stderr)

    def send(self, entries) -> int:
        """Validate and ship. Returns the number of entries actually sent."""
        batch = entries if isinstance(entries, list) else [entries]
        valid, rejected = [], 0
        for raw in batch:
            if (v := validate_entry(raw)) is not None:
                valid.append(v)
            else:
                rejected += 1
                print(f"[goals] rejected {raw!r}", file=sys.stderr)
        self.stats["rejected"] += rejected
        if not valid:
            return 0
        for v in valid:
            v.setdefault("id", self._next_id())
        if not self.connected:
            self.stats["undeliverable"] += len(valid)
            print(f"[goals] no client attached; dropped {len(valid)} goal(s)", file=sys.stderr)
            return 0
        # `use_selection` says what to leave out; it is not itself part of the
        # wire format, and the client drops goals carrying keys it does not know.
        wire = [{k: v for k, v in g.items() if k != "use_selection"} for g in valid]
        # A bare object for one, a list for several -- both are accepted framings,
        # and a list of one would run identically anyway.
        if self._broadcast(json.dumps(wire[0] if len(wire) == 1 else wire)):
            self.stats["sent"] += len(valid)
            return len(valid)
        return 0

    def stop(self) -> int:
        """Halt whatever the policy is running. Returns the number delivered --
        0 means it reached nobody, and the caller must not assume it halted.

        What actually halts the client is unconfirmed; `--stop-goal` swaps the
        payload without a code change once the client owner says what works."""
        return self.send(dict(self.stop_goal))


# --------------------------------------------------------------------------- #
# Turning an instruction into a plan entry
# --------------------------------------------------------------------------- #

#: Constrained decoding, not free-form JSON. The model picks a target and an
#: interaction; WHERE the target is gets asked separately, by `locate`.
#:
#: This schema used to carry optional `u`/`v` fractions. Under a JSON schema the
#: model filled them in almost every time, with coarse round guesses -- "go to
#: the building in white" came back as [0.6, 0.2] and drove at the building in
#: front instead. Qwen2.5-VL grounds well, but only when asked the way it was
#: trained: a bare "Locate X" prompt answered with pixel `bbox_2d`.
#:
#: `reasoning` is first on purpose: guided decoding emits properties in schema
#: order, so it is the only place the model can think before it commits to a
#: step. With one `target` and no thinking, "go to the other side of the
#: building" came back as target 'the other side of the red brick building'
#: (session.jsonl line 94708) -- a place, not a thing -- and the agent stood
#: still for 120s. `steps` lets a relation be broken into things it can see.
#:
#: No `say`: the planner runs in a background thread now, and the spoken reply
#: comes from `acknowledge` while it works, so a `say` decoded here would arrive
#: seconds after the person already heard one.
MAX_STEPS = 4
GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 240},   # ~50 tokens, ~2.5s at 20 tok/s
        "steps": {
            "type": "array", "minItems": 1, "maxItems": MAX_STEPS,
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "interaction": {"type": "string", "enum": sorted(PLAN_INTERACTIONS)},
                },
                "required": ["target", "interaction"],
                "additionalProperties": False,
            },
        },
        "item": {"type": "string"},
        "count": {"type": "integer", "minimum": 1},
        "done_when": {"type": "string"},
        "done_say": {"type": "string"},
    },
    "required": ["reasoning", "steps", "done_when", "done_say"],
    "additionalProperties": False,
}

GOAL_PROMPT = """Turn the user's instruction into a short plan for the Minecraft policy.

reasoning: think before you plan, in English, in at most two short sentences: what on screen the
instruction is about, where it is relative to you, and which steps get you there. Never
read aloud.
steps: one to four steps, run in order. Only the first runs now; after it you are asked
again with a new screenshot, so later steps are a best guess. Each step has:
  target: something you can see on screen right now, as a short noun phrase -- "the oak
  tree", "the nearest cow", "the right corner of the brick building". Never a whole sentence
  and never a place described relative to something -- "the other side of the building",
  "behind the tree", "past the river" are not targets. Break those into steps that walk to
  things you can see, or turn.
  interaction: Mine for blocks, Hunt for mobs you kill, Use or Interact for containers and
  doors, Craft, Switch to change held item, Approach to just walk there, None to stand
  still, Turn to rotate in place without walking -- then target is exactly "left", "right"
  or "around" ("turn to what's behind you" -> Turn, around).
Examples:
- "chop that oak tree" -> Mine "the oak tree"
- "go to the other side of the brick building" -> Approach "the right corner of the brick
  building", then Approach "the far corner of the brick building"
- "what's behind us, go there" -> Turn "around", then Approach whatever you expect to see
item and count: give both only if the instruction names a countable result ("three logs" ->
item oak_log, count 3). Omit both otherwise.
done_when: what the screen looks like the moment the instruction is carried out, as one
short statement you could check against a screenshot -- "the brick building fills the view
up close", "the cow is gone". It is implied by the instruction; spell it out.
done_say: what you tell the user, read aloud, at the moment done_when becomes true. One
short first-person sentence naming the same target -- "I'm here at the red building", "Got
the three oak logs". Nothing else.

Answer with the goal only."""

#: A goal with no stop condition never ends on the client, and an Approach whose
#: target is a description gets re-resolved against every new frame -- arrive at
#: one brick building and the targeter finds the next. The client can stop on an
#: item count or a stat but not on "arrived", so for everything it cannot express
#: we check arrival against each new frame ourselves and send this.
#:
#: UNCONFIRMED that it halts anything. In session.jsonl the view kept changing for
#: ~45s after it was sent (red building, t=68 -> ~113), and again after t=39 in the
#: next session. It carries no point, so "stand still" may be handed to the
#: targeter as a description. `GoalMonitor` therefore watches for motion after
#: every stop and resends; override the payload with `--stop-goal`.
STOP_GOAL = {"instruction": "stand still", "interaction": "None", "stop": {"steps": 1}}

# A fact about the screen, not about conversational timing -- the same kind of
# question as the salience gate (+0.200, usable), not the timing one (-0.272).
# UNMEASURED all the same; see `--arrive-threshold`.
#
# Asked over the bare image only (`image_only`), never over `render(...)`: that
# context carries "[done so far] Heading straight towards the red building", and
# with it the gate said yes (0.679) on the first frame the agent moved at all.
# The logged yes values are all logistic of multiples of 0.125 -- 0.6225, 0.6792,
# 0.7311 -- so a 0.6 threshold stops on the weakest lean. For Approach goals it
# is only a veto on the box tracker, not the trigger.
GATE_ARRIVED = """Look only at the current screenshot.

Is this true right now: {done_when}?"""


#: Qwen2.5-VL's grounding prompt, verbatim in shape. No system prompt, no
#: schema, no dialogue around it -- every one of those moves the call away from
#: what it was trained on. Absent targets come back as prose ("There are
#: none."), which `parse_bbox` reads as None.
LOCATE_PROMPT = "Locate {target} in the image and output its bbox coordinates in JSON format."
PRESENT_PROMPT = "Is there {target} visible in this image?"

#: Interactions that act on something in view. Craft/Switch/None have nothing
#: to point at, so they skip the locate call entirely.
LOCATABLE = {"Hunt", "Mine", "Use", "Interact", "Approach"}

#: A target naming a place relative to something else, not a thing to box.
#: Checked before `locate`, because the presence gate only says "not there" and
#: the description then goes to a targeter that cannot act on it either.
_RELATION_RE = re.compile(r"\b(side|behind|past|between|through|beyond|opposite|across|"
                          r"around|inside|outside|(?:front|back|rear) of)\b")
MAX_TARGET_WORDS = 8

PLAN_FEEDBACK = ("[feedback] \"{target}\" is a place described relative to something else, "
                 "not a thing you can see. Plan again: every step's target must be something "
                 "visible on screen, or left/right/around for a Turn.")
PLAN_REPLAN = ("[replan] You are partway through the user's instruction. {why} Steps you had "
               "left: {pending}. Plan again from the current screenshot, starting with what "
               "you can see now.")
#: The planner still chooses the interaction and what to say; what it must not
#: do is treat "that" as something to describe. It is told the object is already
#: chosen so its `target` and `done_say` name the right thing ("Chopping that
#: oak") -- neither of which is sent. See THE ONE RULE in `validate_entry`.
SELECTION_NOTE = ("[pointing] The user is physically pointing at one object on screen{where}. "
                  "\"That\", \"it\" and \"there\" mean that object: make it the target of the "
                  "first step. Do not pick a different one.")
CANT_PLAN_SAY = "I can't see a way to do that from here."


def relational_target(target: str) -> bool:
    """True for a target that is a relation ("the other side of X"), not a thing."""
    t = target.lower()
    return bool(_RELATION_RE.search(t)) or len(t.split()) > MAX_TARGET_WORDS


def plan_steps(d) -> list[dict]:
    """The usable steps of a planner reply, in order, at most MAX_STEPS."""
    steps = d.get("steps") if isinstance(d, dict) else None
    out = []
    for s in steps if isinstance(steps, list) else []:
        if not isinstance(s, dict):
            continue
        target = str(s.get("target") or "").strip()
        if s.get("interaction") == "Turn" and turn_yaw(target) is None:
            continue                               # "Turn the corner of X" cannot be sent
        if target and s.get("interaction") in PLAN_INTERACTIONS:
            out.append({"target": target, "interaction": s["interaction"]})
    return out[:MAX_STEPS]


def step_text(step: dict) -> str:
    """"Approach the oak tree" -- for logs and the replan prompt."""
    return f"{step.get('interaction', '?')} {step.get('target', '')}".strip()

_BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,'
                      r'\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]')


def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = 3136, max_pixels: int = 147456) -> tuple[int, int]:
    """(h, w) of the image the model actually sees -- Qwen2.5-VL's own resize.

    `bbox_2d` is in pixels of THIS image, not of the frame we sent, so it is the
    only correct denominator. Defaults match `serve_interaction.sh`; a 224x224
    frame passes through unchanged (64 visual tokens, measured).
    """
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def parse_bboxes(text: str, width: int, height: int) -> list[list[float]]:
    """Grounding reply -> every box as [x1, y1, x2, y2] fractions, in reply order.

    Clamped to the image; a box with no area is dropped. All of them, because
    tracking has to pick the one that is still the SAME building, and that is
    not always the largest.
    """
    out = []
    for m in _BBOX_RE.finditer(text or ""):
        try:
            x1, y1, x2, y2 = (float(g) for g in m.groups())
        except ValueError:
            continue
        x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
        y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
        if (x2 - x1) * (y2 - y1) > 0:
            out.append([x1 / width, y1 / height, x2 / width, y2 / height])
    return out


def box_area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a: list[float], b: list[float]) -> float:
    inter = box_area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def box_side(b: Optional[list[float]]) -> str:
    """Which way to turn to look at `b` again: "left", "right" or "ahead".

    The only thing that survives a lost lock, and what the replan prompt needs
    to say "turn left" instead of guessing.
    """
    if b is None:
        return ""
    cx, _ = box_center(b)
    return "left" if cx < 0.4 else "right" if cx > 0.6 else "ahead"


def parse_bbox(text: str, width: int, height: int) -> Optional[list[float]]:
    """Grounding reply -> the chosen box as [x1, y1, x2, y2] fractions, or None.

    Several matches pick the largest: for "go to the building" the biggest box
    is the nearest one, which is the one a person means. That is right for the
    FIRST pick only; after that `Tracker` follows the instance, not the size.
    """
    return max(parse_bboxes(text, width, height), key=box_area, default=None)


def approach_step_cap(bbox: Optional[list[float]], scale: float,
                      floor: int, ceiling: int) -> Optional[int]:
    """A step budget for an Approach, so it ends even if every stop is lost.

    Bigger box = closer = fewer steps. A description goal with no box gets the
    ceiling. `scale <= 0` disables the cap. UNCALIBRATED: what a client "step"
    is (tick? action?) is unknown -- log it against when motion actually stops
    and tune `--approach-steps-*`. It is a bound, not an arrival detector.
    """
    if scale <= 0:
        return None
    if bbox is None:
        return int(ceiling)
    width = max(bbox[2] - bbox[0], 0.05)
    return int(min(max(round(scale / width), floor), ceiling))


def turn_yaw(target: str) -> Optional[int]:
    """"left" / "right" / "around" (and "behind", "back") -> yaw degrees, or None."""
    words = set(re.findall(r"[a-z]+", target.lower()))
    return next((yaw for word, yaw in TURN_YAW if word in words), None)


def plan_entry(d: dict, point: Optional[list[float]] = None,
               steps: Optional[int] = None, selection: bool = False) -> Optional[dict]:
    """Model JSON (+ a located point) -> a validated plan entry, or None.

    The pixel path does not exist here on purpose. `point` is a `normalized`
    [u, v] from `locate`, which means the same thing in 224x224 and in 640x360
    as long as the client resizes rather than crops; without one, the target
    description goes out for the client's own targeter.

    `steps` caps an Approach that has no other stop (see `approach_step_cap`);
    an item/count stop always wins.

    `selection` means the person pinched the target on the laptop. The entry then
    names no target at all -- see THE ONE RULE in `validate_entry` -- so `target`
    stays ours, for the log, the tracker and what gets said, never for the wire.
    Everything else about the entry is decided exactly as it always was.
    """
    if not isinstance(d, dict):
        return None
    target = str(d.get("target") or "").strip()
    if not target:
        return None
    if d.get("interaction") == "Turn":
        yaw = turn_yaw(target)
        return None if yaw is None else validate_entry({"turn": yaw})
    entry: dict = {"interaction": d.get("interaction", "Approach"), "stop": None}
    if selection:
        entry["use_selection"] = True          # never sent; stripped on validation
    elif point is not None:
        entry["point"] = [float(point[0]), float(point[1])]
        entry["normalized"] = True
        entry["instruction"] = target          # kept as the targeter's fallback
    else:
        entry["point"] = target
    item, count = d.get("item"), d.get("count")
    if isinstance(item, str) and item.strip() and isinstance(count, int) \
            and not isinstance(count, bool) and count > 0:
        entry["stop"] = {"item": item.strip(), "count": count}
    elif selection or (steps and entry["interaction"] == "Approach"):
        # A selection goal always gets a budget: the instruction named no count,
        # and a goal with no stop at all never ends on the client.
        entry["stop"] = {"steps": int(steps or SELECTION_STEPS)}
    return validate_entry(entry)


@dataclass
class Plan:
    """What `InteractionModel.plan` decided. `entry` is the first step only."""
    entry: Optional[dict] = None
    done_when: str = ""
    say: str = ""
    bbox: Optional[list[float]] = None
    done_say: str = ""
    reasoning: str = ""
    pending: list[dict] = field(default_factory=list)   # steps after `entry`, a best guess
    target: str = ""                  # what the first step is about, in words. A
                                      # selection entry carries no target, and the
                                      # monitor still needs one to track and to speak.


@dataclass
class TrackerConfig:
    arrive_width: float = 0.65     # box width (fraction of frame) that counts as there
    arrive_hits: int = 2           # consecutive frames that must agree
    lock_iou: float = 0.3          # IoU with the locked box that keeps the lock
    max_jump: float = 0.2          # else: center may move at most this far (frame fractions)
    shrink_ratio: float = 0.6      # a match smaller than this x the last box is not it
    lock_misses: int = 4           # consecutive misses before the lock counts as lost
    edge: float = 0.02             # a box this close to both side edges spans the view


class Tracker:
    """Follows ONE instance across frames and says when it is close enough.

    Replaces "is this true: we're up close to the red building?" for Approach
    goals. That question cannot tell which red building, and it said yes on the
    first frame the agent moved. Width is used, not area: the red building's box
    was already clipped at the top when the goal was sent (y1 = 0), so height and
    area saturate long before arrival.

    Pure -- no model calls -- so thresholds can be replayed offline against
    logged `arrival.check` boxes.
    """

    def __init__(self, box: Optional[list[float]], cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.box = box                       # the locked instance; None = pick on first sight
        self.width_prev: Optional[float] = None
        self.hits = 0
        self.misses = 0

    def update(self, boxes: list[list[float]]) -> dict:
        """One frame's grounding boxes -> a decision dict, logged verbatim.

        state is "tracking", "arrived" or "lost"."""
        cfg, info = self.cfg, {"n_boxes": len(boxes)}
        match, why = None, "no_box"
        if boxes and self.box is None:
            match, why = max(boxes, key=box_area), "first_sight"
        elif boxes:
            iou, best = max(((box_iou(b, self.box), b) for b in boxes), key=lambda p: p[0])
            cx, cy = box_center(self.box)
            jump, near = min((math.dist(box_center(b), (cx, cy)), b) for b in boxes)
            info.update(iou=round(iou, 3), jump=round(jump, 3))
            if iou >= cfg.lock_iou:
                match, why = best, "iou"
            elif jump <= cfg.max_jump:
                match, why = near, "near"
            else:
                why = "jumped"
            if match is not None and box_area(match) < cfg.shrink_ratio * box_area(self.box):
                match, why = None, "shrank"
            # The jump/shrink rules tell instances apart; with one box there is
            # nothing to tell apart. Every logged "lost" but one was a single box
            # of the right building rejected as jumped/shrank while the camera
            # walked or turned (white house, yellow bus, red building).
            if match is None and len(boxes) == 1:
                info["rejected"] = why
                match, why = boxes[0], "only_box"
        info["match"] = why

        if match is None:
            self.misses += 1
            self.hits = 0
            info.update(misses=self.misses, hits=0)
            info["state"] = "lost" if self.misses >= cfg.lock_misses else "tracking"
            return info

        self.misses = 0
        self.box = match
        width = match[2] - match[0]
        spans = match[0] <= cfg.edge and match[2] >= 1 - cfg.edge
        # One frame interval ahead: a check lands ~3-4s after the frame it looks
        # at, ~15 blocks at walking speed, so stop on the frame BEFORE crossing.
        growth = None if self.width_prev is None else width - self.width_prev
        predicted = growth is not None and growth > 0 and width + growth >= cfg.arrive_width
        self.width_prev = width
        close = width >= cfg.arrive_width or spans or predicted
        self.hits = self.hits + 1 if close else 0
        info.update(box=[round(v, 4) for v in match], width=round(width, 4),
                    growth=None if growth is None else round(growth, 4),
                    spans=spans, predicted=predicted, hits=self.hits, misses=0)
        info["state"] = "arrived" if self.hits >= cfg.arrive_hits else "tracking"
        return info


def frame_thumb(frame: str, size: int = 32) -> Optional[bytes]:
    """A tiny grayscale copy for motion checks. None if it will not decode."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(frame)))
        return img.convert("L").resize((size, size)).tobytes()
    except Exception:
        return None


def frame_motion(prev: Optional[bytes], cur: Optional[bytes]) -> Optional[float]:
    """Mean absolute pixel difference (0..255) between two thumbnails.

    The only evidence the policy halted: nothing comes back up tunnel 2. A
    still screen scores near 0; walking scores far higher. Threshold is
    `--still-motion`, UNCALIBRATED -- read it off logged vision.frame events."""
    if prev is None or cur is None or len(prev) != len(cur):
        return None
    return sum(abs(a - b) for a, b in zip(prev, cur)) / len(cur)


def image_only(frame: str) -> list[dict]:
    """Messages holding the frame and nothing else -- no system prompt, no dialogue."""
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}]}]


def drain(q: queue.Queue) -> list[str]:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


class ScreenCapture:
    """Keyframes, not every rendered frame.

    Vision tokens are where a 300ms loop dies, so we sample sparsely and
    downscale hard. `--frame-dir` replays a directory of images instead, which is
    how you test this on a headless node.
    """

    def __init__(self, interval: float, size: int, frame_dir: Optional[Path] = None) -> None:
        self.interval, self.size = interval, size
        self._last = -1e9
        self._sct = None
        self._i = 0
        self._files = sorted(
            (p for p in frame_dir.iterdir()
             if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        ) if frame_dir else []
        if not self._files:
            try:
                import mss
                self._sct = mss.mss()
            except Exception:
                print("[capture] mss unavailable; running without vision "
                      "(use --frame-dir to replay images)", file=sys.stderr)

    def due(self, now: float) -> bool:
        return (self._sct is not None or bool(self._files)) and now - self._last >= self.interval

    def grab(self, now: float) -> Optional[str]:
        from PIL import Image
        self._last = now
        if self._files:
            img = Image.open(self._files[self._i % len(self._files)])
            self._i += 1
        elif self._sct is not None:
            shot = self._sct.grab(self._sct.monitors[1])
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        else:
            return None
        img = img.convert("RGB")
        img.thumbnail((self.size, self.size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Background model (outside the real-time box)
# --------------------------------------------------------------------------- #

BG_SYSTEM = ("You do the slow thinking for a real-time voice agent. You are given the full "
             "session so far and a task. Reason as much as you need, then answer in at most "
             "three sentences that can be read aloud verbatim.")


class BackgroundModel:
    """Fire-and-forget delegation to a second, slower vLLM instance.

    A separate server on purpose: a long deliberation must not contend for the
    interaction model's decode slots, or the real-time boundary the whole design
    protects is gone.
    """

    def __init__(self, client: OpenAI, model: str, timeline: Timeline) -> None:
        self.client, self.model, self.timeline = client, model, timeline
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bg")
        self.inflight = 0
        self._lock = threading.Lock()

    def dispatch(self, task: str, context: str) -> None:
        self.timeline.append(BG_DISPATCHED, task)
        with self._lock:
            self.inflight += 1
        self.pool.submit(self._run, task, context)

    def _run(self, task: str, context: str) -> None:
        t0 = time.monotonic()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": BG_SYSTEM},
                          {"role": "user",
                           "content": f"Session so far:\n{context}\n\nTask: {task}"}],
                max_tokens=512,
                temperature=0.3,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:                       # never kill the loop
            text = f"(delegation failed: {e})"
        self.timeline.append(BG_RESULT, text, task=task,
                             latency_s=round(time.monotonic() - t0, 2))
        with self._lock:
            self.inflight -= 1


# --------------------------------------------------------------------------- #
# Interaction model
# --------------------------------------------------------------------------- #

#: Where a streamed reply is cut into sentences: after . ! or ? and whitespace, so
#: "1.5 blocks" stays whole and "Okay. Heading there" is two.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class SentenceStream:
    """Tokens in, whole sentences out, as soon as each one closes.

    The laptop starts speaking a message as soon as it arrives, so a reply sent a
    sentence at a time is heard after its first sentence decodes instead of its last.
    """

    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit, self.buf = emit, ""
        self.first_t: Optional[float] = None       # monotonic time of the first emit

    def __call__(self, delta: str) -> None:
        self.buf += delta
        *done, self.buf = _SENTENCE_END.split(self.buf)
        for sentence in done:
            self._emit(sentence)

    def flush(self) -> None:
        self._emit(self.buf)
        self.buf = ""

    def _emit(self, sentence: str) -> None:
        if sentence.strip():
            if self.first_t is None:
                self.first_t = time.monotonic()
            self.emit(sentence.strip())


class InteractionModel:
    """One micro-turn.

    `gate` decides whether this tick produces anything at all; `speak` streams the
    utterance. On a silent tick only the gate call runs -- one token over a prefix
    the server has already cached -- so silence stays nearly free.

    Spoken replies (`speak`, so `answer` and `acknowledge`) can go to a separate
    `talk_client`: an FP8 copy of the model, faster to decode. Gates, plans and
    grounding -- everything that becomes an action -- stay on `client` at full
    precision. Without one, both are the same server.
    """

    def __init__(self, client: OpenAI, model: str, max_tokens: int,
                 min_pixels: int = 3136, max_pixels: int = 147456,
                 present_threshold: float = 0.8,
                 approach_steps: tuple[float, int, int] = (0.0, 0, 0),
                 talk_client: Optional[OpenAI] = None,
                 talk_model: Optional[str] = None) -> None:
        self.client, self.model = client, model
        self.talk_client, self.talk_model = talk_client or client, talk_model or model
        self.max_tokens = max_tokens
        self.present_threshold = present_threshold
        self.approach_steps = approach_steps        # (scale, floor, ceiling); see approach_step_cap
        # Must match the server's --mm-processor-kwargs, or `locate` divides
        # by the wrong image size and every point lands off target.
        self.min_pixels, self.max_pixels = min_pixels, max_pixels

    def gate(self, messages: list[dict], question: str) -> float:
        """P(yes) for `question` -- a yes/no decision read as a probability.

        One token, `structured_outputs.choice` constrained to yes/no, with logprobs. The
        article's model needs none of this: silence is a normal output of every
        200ms micro-turn, learned in the weights. Qwen was never trained on a
        stream where silence is a valid continuation, so we bolt it on outside.

        A scalar beats a JSON header three ways: format compliance cannot fail,
        it costs ~1 token over a cached prefix, and the threshold is a runtime
        knob. It is also the right shape to swap for a trained head later --
        same call site, the probability just comes from a LoRA instead.
        """
        # Copied, not appended to: the caller renders once per tick and hands the
        # same list to every gate, so mutating it would leak one gate's question
        # into the next one's prompt.
        messages = messages + [
            {"role": "user", "content": f"[gate] {question} Answer yes or no."}]
        choice = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=1, temperature=0.0,
            logprobs=True, top_logprobs=5, extra_body={"structured_outputs": {"choice": ["yes", "no"]}},
        ).choices[0]

        if choice.logprobs and choice.logprobs.content:
            # Summed, not assigned: "Yes"/"yes"/" yes" all normalize to one key, and a
            # plain dict comprehension would keep the last (least likely) variant.
            p: dict[str, float] = {}
            for t in choice.logprobs.content[0].top_logprobs:
                key = t.token.strip().lower()
                p[key] = p.get(key, 0.0) + math.exp(t.logprob)
            yes, no = p.get("yes"), p.get("no")
            if yes is not None and no is not None and yes + no > 0:
                return yes / (yes + no)              # renormalize over the two options
            if yes is not None:
                return yes
        return float((choice.message.content or "").strip().lower().startswith("y"))

    def speak(self, messages: list[dict], on_token: Callable[[str], None]) -> dict:
        t0 = time.monotonic()
        stream = self.talk_client.chat.completions.create(
            model=self.talk_model, messages=messages,
            max_tokens=self.max_tokens, temperature=0.6, stream=True,
        )
        said, ttft = "", None
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                ttft = ttft if ttft is not None else time.monotonic() - t0
                said += delta
                on_token(delta)
        finally:
            stream.close()   # abort in flight; also where barge-in will hook in
        return {"said": said.strip(), "ttft_s": ttft, "total_s": time.monotonic() - t0}

    def answer(self, messages: list[dict],
               on_sentence: Callable[[str], None] = lambda _: None) -> dict:
        """A spoken answer to a question, with no goal attached.

        `speak` over the tick's render plus ANSWER_PROMPT. Not `plan`: its schema
        forces an action. See `talk` for `on_sentence` and the result.
        """
        return self.talk(messages, f"[answer] {ANSWER_PROMPT}", on_sentence)

    def acknowledge(self, messages: list[dict],
                    on_sentence: Callable[[str], None] = lambda _: None) -> dict:
        """What to say the moment an instruction is heard, while `plan` runs elsewhere."""
        return self.talk(messages, f"[reply] {ACK_PROMPT}", on_sentence)

    def talk(self, messages: list[dict], prompt: str,
             on_sentence: Callable[[str], None]) -> dict:
        """`speak` with `prompt` appended, handing each sentence to `on_sentence` as
        soon as it closes. Returns `speak`'s result plus `first_s`, the time to the
        first sentence -- when the person starts hearing it."""
        t0 = time.monotonic()
        sentences = SentenceStream(on_sentence)
        try:
            result = self.speak(messages + [{"role": "user", "content": prompt}], sentences)
        finally:
            sentences.flush()        # a reply cut off mid-sentence is still sent
        result["first_s"] = None if sentences.first_t is None else sentences.first_t - t0
        return result

    def delegation_task(self, messages: list[dict]) -> str:
        """One line describing what the background model should go and do."""
        messages = messages + [
            {"role": "user", "content":
             "[gate] In one sentence, state the task to hand to a background "
             "researcher. Write only the task."}]
        return (self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=60, temperature=0.2
        ).choices[0].message.content or "").strip()

    def locate(self, frame: str, target: str) -> Optional[list[float]]:
        """Where `target` is in `frame`, as an [x1, y1, x2, y2] fraction box, or None.

        Its own call on purpose -- see LOCATE_PROMPT. Not guided: forcing a
        schema is exactly what degraded the old u/v fractions.

        Asked "is it there?" first. Grounding alone boxes the nearest lookalike
        for an absent target -- "the white cow" came back as the white building
        -- and a point overrides the description on the client, so a wrong box
        is worse than none. MEASURED on one synthetic Minecraft frame against
        the 32B: present targets >=0.963, absent <=0.500. One frame is not a
        calibration; see `--present-threshold`.
        """
        if self.gate(image_only(frame),
                     PRESENT_PROMPT.format(target=target)) < self.present_threshold:
            return None
        return max(self.locate_all(frame, target), key=box_area, default=None)

    def locate_all(self, frame: str, target: str) -> list[list[float]]:
        """Every box grounding returns for `target`, no presence gate.

        For tracking an instance already chosen by `locate`: presence was
        settled then, and a lookalike box is filtered by the lock, not here.
        """
        from PIL import Image
        image = image_only(frame)[0]["content"][0]
        width, height = Image.open(io.BytesIO(base64.b64decode(frame))).size
        seen_h, seen_w = smart_resize(height, width, min_pixels=self.min_pixels,
                                      max_pixels=self.max_pixels)
        text = self.client.chat.completions.create(
            model=self.model, max_tokens=200, temperature=0.0,
            messages=[{"role": "user", "content": [
                image, {"type": "text", "text": LOCATE_PROMPT.format(target=target)}]}],
        ).choices[0].message.content or ""
        return parse_bboxes(text, seen_w, seen_h)

    def _decode_plan(self, messages: list[dict]) -> Optional[dict]:
        raw = (self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=450, temperature=0.2,
            extra_body={"structured_outputs": {"json": GOAL_SCHEMA}},
        ).choices[0].message.content or "").strip()
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[goals] unparseable plan: {raw!r}", file=sys.stderr)
            return None
        return d if isinstance(d, dict) else None

    def plan(self, messages: list[dict], frame: Optional[str] = None,
             note: str = "", selection: Optional[dict] = None) -> Plan:
        """The instruction as a `Plan`: the first step as a tunnel-2 entry, plus
        what is left and how to tell it is done.

        Runs on a planner thread, off the main loop; the reply the person hears
        comes from `acknowledge` meanwhile. The entry is None if the goal is
        unusable, and `say` is set only to admit that (CANT_PLAN_SAY). `bbox` is
        what `locate` found, for the log. `done_say`
        is spoken when the monitor sees done_when happen -- decoded now, with the
        same target in hand, so arrival costs no extra call.

        Only the first step is sent. `pending` is the rest, and it is never sent
        as-is: when a step ends, main() plans again from the new screen with
        `note` (PLAN_REPLAN), because the corner you walk to shows a different
        view than the one the later steps were guessed from.

        A step whose target is a relation ("the other side of X") gets one retry
        with PLAN_FEEDBACK; if that fails too, nothing is sent and `say` admits
        it, rather than shipping a goal the policy stands still on.

        Costs a constrained-decode over the same cached prefix the gates use
        (reasoning makes it longer), plus one grounding call on the frame.

        `done_when` rides alongside rather than inside the entry: it is ours to
        check, and the client drops goals carrying keys it does not know.

        `selection` is what the person is pointing at (`selection_of`). It costs
        the grounding call rather than adding one -- the instance is already
        picked, and the entry deliberately carries no point for it to fill.
        """
        if selection is not None:
            box = selection.get("box")
            note = (note + "\n" if note else "") + SELECTION_NOTE.format(
                where=f" inside the box {[round(v, 2) for v in box]} (x1 y1 x2 y2, "
                      f"fractions of the frame)" if box else "")
        messages = messages + ([{"role": "user", "content": note}] if note else []) \
            + [{"role": "user", "content": f"[goal] {GOAL_PROMPT}"}]
        d = self._decode_plan(messages)
        steps = plan_steps(d)
        bad = next((s["target"] for s in steps
                    if s["interaction"] in LOCATABLE and relational_target(s["target"])), None)
        if bad is not None:
            print(f"[goals] relational target {bad!r}; replanning "
                  f"(reasoning: {(d or {}).get('reasoning')!r})", file=sys.stderr)
            d = self._decode_plan(messages + [
                {"role": "user", "content": PLAN_FEEDBACK.format(target=bad)}])
            steps = plan_steps(d)
            bad = next((s["target"] for s in steps if s["interaction"] in LOCATABLE
                        and relational_target(s["target"])), None)
            if bad is not None:
                print(f"[goals] still relational {bad!r}; not sending", file=sys.stderr)
                return Plan(say=CANT_PLAN_SAY,
                            reasoning=str((d or {}).get("reasoning") or ""))
        if d is None or not steps:
            return Plan()

        reasoning = str(d.get("reasoning") or "").strip()
        done_when = str(d.get("done_when") or "").strip()
        done_say = str(d.get("done_say") or "").strip()
        step, pending = dict(steps[0]), steps[1:]
        if not pending:                            # a count belongs to the whole instruction
            step.update({k: d[k] for k in ("item", "count") if k in d})
        print(f"[goals] reasoning: {reasoning!r} steps: "
              f"{'; '.join(step_text(s) for s in steps)}", file=sys.stderr)
        if step["interaction"] == "Turn" and not pending:
            done_when = done_say = ""              # nothing to check; done once sent

        bbox, point = None, None
        target = step["target"]
        if selection is not None:
            # No `locate`: they chose the instance themselves, and the box they
            # chose is a better seed for the tracker than grounding a description
            # we are deliberately not sending.
            bbox = selection.get("box")
        elif frame and step["interaction"] in LOCATABLE:
            try:
                bbox = self.locate(frame, target)
            except Exception as e:                 # a failed locate still sends the text
                print(f"[goals] locate failed: {e}", file=sys.stderr)
            if bbox is not None:
                point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            print(f"[goals] locate {target!r} -> {bbox}", file=sys.stderr)
        cap = approach_step_cap(bbox, *self.approach_steps) \
            if step["interaction"] == "Approach" \
            else (SELECTION_STEPS if selection is not None else None)
        return Plan(plan_entry(step, point, cap, selection is not None), done_when, "",
                    bbox, done_say, reasoning, pending, target)


# --------------------------------------------------------------------------- #
# Watching a running goal: arrival, then an actual halt
# --------------------------------------------------------------------------- #

@dataclass
class MonitorConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    arrive_threshold: float = 0.6       # yes/no gate, non-Approach goals only
    veto_threshold: float = 0.5         # image-only veto on a tracker "arrived"; 0 = off
    still_motion: float = 4.0           # frame_motion at or below this = halted
    stop_retries: int = 2               # resends after the first stop
    refresh_point: bool = False         # resend the tracked point each frame (client-dependent)
    # Backstop for a goal nothing else ends. Use 'the door' went out with no box
    # and stop=None, its done_when never passed (p~0.4), and it ran for 8000+ s.
    goal_timeout: float = 120.0         # seconds; 0 = off
    # "Go to the other side of the building" went out as a description the policy
    # could not act on; motion sat at ~0.25 for all 120s while the tracker kept
    # "tracking" the building (session.jsonl line 94714 on). A running goal on a
    # still screen is stuck, or -- for a step with more after it -- finished.
    stall_frames: int = 3               # consecutive still frames; 0 = off


@dataclass
class GoalRun:
    instruction: str
    entry: dict
    target: str
    done_when: str
    sent_t: float
    tracker: Optional[Tracker] = None
    done_say: str = ""                  # spoken on arrival; see `done_line`
    phase: str = "running"              # running -> stopping -> gone
    gate_hits: int = 0
    stop_reason: str = ""
    stop_t: float = 0.0
    stop_attempts: int = 0
    frames_since_stop: int = 0
    delivered: int = 0
    pending: list[dict] = field(default_factory=list)   # planned steps after this one
    frames_seen: int = 0
    still_frames: int = 0


class GoalMonitor:
    """The one goal the policy is running, from send until it is seen to halt.

    Separate from `Task` on purpose. `Task` going idle means the conversation
    is done; it must not mean we stop watching the game, because the log shows a
    stop that did nothing (view still changing ~45s later) and the old loop,
    idle, never looked again. So a stop moves the run to "stopping", and it
    leaves only when a frame shows the screen still, or after `stop_retries`
    resends -- every step logged as `goal.stopped`.

    Driven once per NEW frame from main(), before the idle short-circuit.
    """

    def __init__(self, goals: GoalLink, interaction: InteractionModel,
                 timeline: Timeline, cfg: MonitorConfig) -> None:
        self.goals, self.interaction, self.timeline, self.cfg = goals, interaction, timeline, cfg
        self.run: Optional[GoalRun] = None

    @property
    def phase(self) -> str:
        return "idle" if self.run is None else self.run.phase

    @property
    def running(self) -> bool:
        return self.run is not None and self.run.phase == "running"

    def started(self, instruction: str, entry: dict, done_when: str,
                bbox: Optional[list[float]], done_say: str = "",
                pending: Optional[list[dict]] = None, target: str = "") -> None:
        """A goal was just delivered. Whatever ran before is assumed preempted --
        UNCONFIRMED client behavior, so it is logged rather than trusted.

        `pending` is the rest of the plan. With some, the end of this step is
        reported as "stepped" instead of "arrived", so main() plans the next."""
        if self.run is not None:
            self.timeline.append(GOAL_STOPPED, self.run.instruction, event="replaced",
                                 reason="new_goal", phase=self.run.phase)
        pending = list(pending or [])
        if "turn" in entry and not pending:
            # A turn has no arrival to watch, and must not be followed by a
            # "superseded" stop that could cancel it mid-rotation.
            self.run = None
            return
        # A selection entry names no target on the wire, so the planner's own
        # words for it come in alongside; everything else reads it off the entry.
        target = target or entry.get("instruction") or entry.get("point")
        target = target if isinstance(target, str) else ""
        tracker = Tracker(bbox, self.cfg.tracker) \
            if entry.get("interaction") == "Approach" and target else None
        self.run = GoalRun(instruction, entry, target, done_when,
                           self.timeline.now(), tracker, done_say, pending=pending)

    def stop(self, reason: str) -> None:
        run = self.run
        if run is None or run.phase == "stopping":
            return
        run.phase, run.stop_reason, run.stop_t = "stopping", reason, self.timeline.now()
        self._send_stop()

    def _send_stop(self) -> None:
        run = self.run
        run.stop_attempts += 1
        run.frames_since_stop = 0
        run.delivered = self.goals.stop()
        self.timeline.append(GOAL_STOPPED, run.instruction,
                             event="sent" if run.stop_attempts == 1 else "resent",
                             reason=run.stop_reason, attempt=run.stop_attempts,
                             delivered=run.delivered)
        if not run.delivered:
            print(f"[goals] stop NOT delivered (attempt {run.stop_attempts})", file=sys.stderr)

    def on_frame(self, frame: str, motion: Optional[float]) -> Optional[dict]:
        """Returns the arrival-check record when one ran, else None."""
        if self.run is None:
            return None
        if self.run.phase == "stopping":
            self._check_halted(motion)
            return None
        return self._check_arrival(frame, motion)

    def _check_halted(self, motion: Optional[float]) -> None:
        run, cfg = self.run, self.cfg
        run.frames_since_stop += 1
        can_retry = run.stop_attempts <= cfg.stop_retries
        if not run.delivered:
            if can_retry:
                self._send_stop()
            else:
                self._finish("failed", motion)
            return
        # The first frame after a stop still spans time before it landed; judge
        # the second, whose diff lies entirely after the stop.
        if run.frames_since_stop < 2:
            return
        if motion is not None and motion <= cfg.still_motion:
            self._finish("confirmed", motion)
        elif can_retry:
            self._send_stop()
        else:
            self._finish("failed", motion)

    def _finish(self, event: str, motion: Optional[float]) -> None:
        run = self.run
        self.timeline.append(GOAL_STOPPED, run.instruction, event=event,
                             reason=run.stop_reason, attempts=run.stop_attempts,
                             motion=None if motion is None else round(motion, 2),
                             after_s=round(self.timeline.now() - run.stop_t, 1))
        print(f"[goals] stop {event} ({run.stop_reason}, {run.stop_attempts} sent)",
              file=sys.stderr)
        self.run = None

    def _check_arrival(self, frame: str, motion: Optional[float] = None) -> Optional[dict]:
        run, cfg = self.run, self.cfg
        age = self.timeline.now() - run.sent_t
        run.frames_seen += 1
        still = motion is not None and motion <= cfg.still_motion
        run.still_frames = run.still_frames + 1 if still else 0
        if cfg.goal_timeout > 0 and age >= cfg.goal_timeout:
            # Checked before the nothing-to-check return: those goals need it most.
            info = {"interaction": run.entry.get("interaction"), "target": run.target,
                    "goal_age_s": round(age, 1), "state": "timeout"}
            self.timeline.append(ARRIVAL_CHECK, run.instruction, **info)
            print(f"[goals] arrival check: {info}", file=sys.stderr)
            self.stop("timeout")
            info["instruction"] = run.instruction
            return info
        if "turn" in run.entry:
            # Only reached with steps after it. Done once the screen stops
            # turning -- never stopped, which could cut the rotation short. The
            # first frame straddles the send, so judge from the second.
            if run.frames_seen < 2 or (not still and run.frames_seen < 4):
                return None
            info = {"interaction": "Turn", "target": run.target, "turn": run.entry["turn"],
                    "goal_age_s": round(age, 1), "state": "stepped",
                    "pending": run.pending, "instruction": run.instruction}
            self.timeline.append(ARRIVAL_CHECK, run.instruction,
                                 **{k: v for k, v in info.items() if k != "instruction"})
            print(f"[goals] arrival check: {info}", file=sys.stderr)
            self.run = None
            return info
        # Approach only, or any step with more after it: a Mine on a single
        # block can stand still for a while and still be working.
        if cfg.stall_frames > 0 and run.still_frames >= cfg.stall_frames \
                and (run.entry.get("interaction") == "Approach" or run.pending):
            info = {"interaction": run.entry.get("interaction"), "target": run.target,
                    "goal_age_s": round(age, 1), "state": "stalled",
                    "still_frames": run.still_frames, "motion": motion,
                    "pending": run.pending}
            self.timeline.append(ARRIVAL_CHECK, run.instruction, **info)
            print(f"[goals] arrival check: {info}", file=sys.stderr)
            self.stop("stalled")
            info["instruction"] = run.instruction
            return info
        if run.tracker is None and not run.done_when:
            return None                               # nothing we can check
        t0 = time.monotonic()
        info: dict = {"interaction": run.entry.get("interaction"), "target": run.target,
                      "goal_age_s": round(age, 1)}
        try:
            if run.tracker is not None:
                info.update(run.tracker.update(self.interaction.locate_all(frame, run.target)))
                if info["state"] == "lost":
                    # The last box we did match: where to look for it again. A
                    # target never seen at all (no_box on the first frame) has no
                    # side, and the replan then just works from this screen.
                    info["last_side"] = box_side(run.tracker.box)
                if info["state"] == "arrived" and cfg.veto_threshold > 0:
                    # done_when is the whole instruction's; an in-between step
                    # only has to reach its own target.
                    claim = (not run.pending and run.done_when) \
                        or f"we are right up close to {run.target}"
                    p = self.interaction.gate(image_only(frame),
                                              GATE_ARRIVED.format(done_when=claim))
                    info["p_veto"] = round(p, 4)
                    if p < cfg.veto_threshold:
                        info["state"] = "vetoed"
                        run.tracker.hits = cfg.tracker.arrive_hits - 1   # re-ask next close frame
            else:
                p = self.interaction.gate(image_only(frame),
                                          GATE_ARRIVED.format(done_when=run.done_when))
                run.gate_hits = run.gate_hits + 1 if p >= cfg.arrive_threshold else 0
                info.update(p_arrived=round(p, 4), hits=run.gate_hits,
                            state="arrived" if run.gate_hits >= cfg.tracker.arrive_hits
                            else "checking")
        except Exception as e:
            info.update(state="error", error=str(e))
            print(f"\n[goals] arrival check failed: {e}", file=sys.stderr)
        info["check_s"] = round(time.monotonic() - t0, 3)
        if info["state"] == "arrived" and run.tracker is not None and run.pending:
            info.update(state="stepped", pending=run.pending)
        self.timeline.append(ARRIVAL_CHECK, run.instruction, **info)
        print(f"[goals] arrival check: {info}", file=sys.stderr)

        if info["state"] in ("arrived", "lost", "stepped"):
            self.stop(info["state"])
        elif (cfg.refresh_point and run.tracker is not None and run.tracker.box is not None
              # Never on a selection goal: a goal that named no target must not
              # start naming one, or the refresh discards what they pinched.
              and "point" in run.entry
              and info.get("match") in ("iou", "near", "first_sight")):
            cx, cy = box_center(run.tracker.box)
            refreshed = dict(run.entry, point=[cx, cy], normalized=True, keep_memory=True)
            if self.goals.send(refreshed):
                self.timeline.append(GOAL_SENT, json.dumps(refreshed), refresh=True)
        info["instruction"] = run.instruction
        if info["state"] == "arrived":
            info["say"] = done_line(run)
        return info


def done_line(run: GoalRun) -> str:
    """What to say when a goal is seen to be done: the planner's `done_say`,
    else a template, so arrival is never silent."""
    if run.done_say:
        return run.done_say
    if not run.target:
        return "Done."
    if run.entry.get("interaction") == "Approach":
        return f"I'm here at {run.target}."
    return f"Done with {run.target}."


def step_note(check: dict) -> str:
    """A `[done so far]` line for a step that ended without ending the instruction."""
    target, action = check.get("target") or "it", str(check.get("interaction") or "").lower()
    if check["state"] == "stalled":
        return f"tried to {action} {target} but stopped moving"
    if check["state"] == "lost":
        side = check.get("last_side")
        return f"lost sight of {target}" + (f", last seen to the {side}" if side in
                                            ("left", "right") else "")
    if "turn" in check:
        return {90: "turned right", -90: "turned left"}.get(check["turn"], "turned around")
    return f"reached {target}" if action == "approach" else f"finished {action} {target}"


def replan_why(state: str, check: Optional[dict]) -> str:
    """The `{why}` of PLAN_REPLAN: why the planner is being asked again.

    A lost lock gets the side it was last seen on, so the next plan can turn
    towards it instead of guessing from a screen it no longer appears on.
    """
    if state == "stalled":
        return "The last step stopped moving you; try something different."
    if state == "lost":
        target = (check or {}).get("target") or "the target"
        side = (check or {}).get("last_side")
        if side in ("left", "right"):
            return (f"You lost sight of {target}; it was last seen towards the {side}. "
                    f"Turn {side} first, then approach it again.")
        return (f"You lost sight of {target} and it is not on screen now. Turn to look "
                f"for it, or approach something you can see that leads to it.")
    return "The last step is done."


def save_frame(directory: Optional[Path], t: float, frame: str) -> Optional[str]:
    """Keep the pixels the log throws away, so arrival can be labelled later."""
    if directory is None:
        return None
    try:
        path = directory / f"{t:09.2f}.jpg"
        path.write_bytes(base64.b64decode(frame))
        return str(path)
    except Exception as e:
        print(f"[frames] save failed: {e}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--background-url", default=None, help="second vLLM instance; defaults to --url")
    ap.add_argument("--background-model", default=None)
    ap.add_argument("--talk-url", default=None,
                    help="vLLM instance for spoken replies only (serve_talk.sh, FP8); "
                         "defaults to --url. Gates, plans and grounding never use it")
    ap.add_argument("--talk-model", default=None, help="defaults to --model")
    ap.add_argument("--tick", type=float, default=0.3, help="micro-turn length (s)")
    ap.add_argument("--frame-dir", type=Path, default=None, help="replay images instead of grabbing the screen")
    ap.add_argument("--frame-interval", type=float, default=1.0)
    ap.add_argument("--frame-size", type=int, default=384)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--min-pixels", type=int, default=3136,
                    help="the server's mm-processor min_pixels; locate needs it to "
                         "turn bbox pixels into fractions")
    ap.add_argument("--max-pixels", type=int, default=147456,
                    help="the server's mm-processor max_pixels; see --min-pixels")
    ap.add_argument("--present-threshold", type=float, default=0.8,
                    help="P(target is on screen) below which no point is sent and the "
                         "client's targeter gets the description -- see locate")
    ap.add_argument("--reply-threshold", type=float, default=0.2,
                    help="P(the latest utterance needs a reply) above which it is answered "
                         "and becomes the instruction -- see GATE_REPLY, unmeasured")
    ap.add_argument("--question-threshold", type=float, default=0.5,
                    help="P(an addressed line only asks for information) above which it "
                         "is answered without planning a goal -- see GATE_QUESTION, unmeasured")
    ap.add_argument("--delegate-threshold", type=float, default=0.07,
                    help="low because the LATER/NOW gate orders correctly but on a "
                         "compressed scale -- see GATE_DELEGATE")
    ap.add_argument("--done-threshold", type=float, default=0.6,
                    help="P(the instruction is finished) above which it is dropped "
                         "and the agent goes idle -- see GATE_DONE, unmeasured")
    ap.add_argument("--arrive-threshold", type=float, default=0.6,
                    help="P(done_when is true on screen), image only, for NON-Approach "
                         "goals; --arrive-hits frames in a row stop the goal -- unmeasured")
    # Approach goals: box tracking. All UNCALIBRATED -- tune from arrival.check events.
    ap.add_argument("--arrive-width", type=float, default=0.65,
                    help="tracked box width (fraction of frame) that counts as arrived")
    ap.add_argument("--arrive-hits", type=int, default=2,
                    help="consecutive frames that must say arrived before stopping")
    ap.add_argument("--lock-iou", type=float, default=0.3,
                    help="IoU with the locked box that keeps the same instance")
    ap.add_argument("--max-jump", type=float, default=0.2,
                    help="else, max center shift (frame fractions) still the same instance")
    ap.add_argument("--shrink-ratio", type=float, default=0.6,
                    help="a match smaller than this x the last box is treated as a different one")
    ap.add_argument("--lock-misses", type=int, default=4,
                    help="consecutive frames without the instance before stopping as lost")
    ap.add_argument("--arrive-veto-threshold", type=float, default=0.5,
                    help="image-only P(arrived) below which a tracker 'arrived' is ignored; 0 = off")
    ap.add_argument("--refresh-point", action="store_true",
                    help="resend the tracked point with keep_memory each frame -- only once "
                         "the client owner confirms a resend does not restart the policy badly")
    ap.add_argument("--approach-steps-scale", type=float, default=200.0,
                    help="Approach step cap = scale / box width, clamped; 0 disables")
    ap.add_argument("--approach-steps-floor", type=int, default=100)
    ap.add_argument("--approach-steps-ceiling", type=int, default=1200)
    # Halting.
    ap.add_argument("--stop-goal", default=None,
                    help="JSON plan entry sent to halt the policy (default: STOP_GOAL)")
    ap.add_argument("--still-motion", type=float, default=4.0,
                    help="frame_motion (0..255) at or below which a stop counts as halted")
    ap.add_argument("--stop-retries", type=int, default=2,
                    help="stop resends when the screen is still moving or delivery failed")
    ap.add_argument("--goal-timeout", type=float, default=120.0,
                    help="stop any goal still running after this many seconds; 0 = off")
    ap.add_argument("--stall-frames", type=int, default=3,
                    help="consecutive frames at or below --still-motion that count a running "
                         "Approach (or any step with more after it) as stuck; 0 = off")
    ap.add_argument("--max-replans", type=int, default=6,
                    help="times one instruction may be planned again after a step ends "
                         "or stalls, before the agent gives up")
    ap.add_argument("--frame-log-dir", default="session_frames",
                    help="save every received frame as JPEG here ('' = off)")
    ap.add_argument("--listen", type=int, default=None,
                    help="accept transcripts on this WebSocket port instead of stdin")
    ap.add_argument("--goal-port", type=int, default=None,
                    help="second WebSocket port: plan entries out to the Minecraft "
                         "policy, receive-only from the client's side")
    ap.add_argument("--token", default=None,
                    help="shared secret required in the X-Agent-Token header on BOTH "
                         "tunnels -- the client has no setting for a second one")
    ap.add_argument("--log", default="session.jsonl")
    return ap.parse_args()


@dataclass
class PlanJob:
    """A `plan` call running on the planner thread, and what it was for.

    Only the newest job is kept: a newer instruction (or replan) replaces it, and
    the old result is dropped when it lands. Its goal is sent only if `instruction`
    is still the task by then.
    """
    instruction: str
    continuing: Optional[str]          # None for a new instruction, else "stepped"/"stalled"
    future: Future
    t0: float = field(default_factory=time.monotonic)


def _round(x: Optional[float], n: int = 3) -> Optional[float]:
    return None if x is None else round(x, n)


def main() -> None:
    args = parse_args()
    timeline = Timeline(Path(args.log) if args.log else None)
    client = OpenAI(base_url=args.url, api_key="EMPTY")

    stop_goal = None
    if args.stop_goal:
        stop_goal = validate_entry(json.loads(args.stop_goal))
        if stop_goal is None:
            raise SystemExit(f"--stop-goal is not a valid plan entry: {args.stop_goal}")
    frame_dir = None
    if args.frame_log_dir:
        frame_dir = Path(args.frame_log_dir) / time.strftime("%Y%m%d-%H%M%S")
        frame_dir.mkdir(parents=True, exist_ok=True)

    interaction = InteractionModel(
        client, args.model, args.max_tokens, args.min_pixels, args.max_pixels,
        args.present_threshold,
        (args.approach_steps_scale, args.approach_steps_floor, args.approach_steps_ceiling),
        OpenAI(base_url=args.talk_url, api_key="EMPTY") if args.talk_url else None,
        args.talk_model)
    background = BackgroundModel(
        OpenAI(base_url=args.background_url or args.url, api_key="EMPTY"),
        args.background_model or args.model, timeline)
    token = args.token or ""
    uplink = Uplink(token).serve(args.listen) if args.listen else None
    goals = GoalLink(token, stop_goal).serve(args.goal_port) if args.goal_port else None
    monitor = GoalMonitor(goals, interaction, timeline, MonitorConfig(
        tracker=TrackerConfig(args.arrive_width, args.arrive_hits, args.lock_iou,
                              args.max_jump, args.shrink_ratio, args.lock_misses),
        arrive_threshold=args.arrive_threshold, veto_threshold=args.arrive_veto_threshold,
        still_motion=args.still_motion, stop_retries=args.stop_retries,
        refresh_point=args.refresh_point, goal_timeout=args.goal_timeout,
        stall_frames=args.stall_frames)) if goals is not None else None
    stdin_q = None if uplink else stdin_lines()
    # Frames come up tunnel 1 when there is one; grabbing this node's own screen
    # is only the headless-replay path.
    capture = ScreenCapture(args.frame_interval, args.frame_size, args.frame_dir) \
        if uplink is None else None

    # Two gate calls per utterance run side by side; plans run off the main loop
    # entirely. Two planner workers so a new instruction does not queue behind the
    # stale plan it just made obsolete.
    gate_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gate")
    planner = ThreadPoolExecutor(max_workers=2, thread_name_prefix="plan")

    def say_now(text: str, reason: Optional[str] = None) -> None:
        """One sentence (or line) to the person, now."""
        print(text)
        if uplink is not None:
            uplink.say(text, reason=reason)

    print(f"[interaction] tick={args.tick}s model={args.model} "
          f"talk={args.talk_model or args.model}@{args.talk_url or args.url} "
          f"reply_threshold={args.reply_threshold} done_threshold={args.done_threshold}"
          f" uplink={args.listen or 'stdin'} goals={args.goal_port or 'off'}"
          f"  (idle until instructed, ^C to quit)")

    task = Task()
    planned: Optional[str] = None     # instruction already sent down tunnel 2
    plan_job: Optional[PlanJob] = None   # the plan being worked out on the planner thread
    replan: Optional[str] = None      # set when a step ended: plan the next one this tick
    replans = 0                       # replans spent on the current instruction
    frame: Optional[str] = None       # newest screenshot, or None without vision
    thumb: Optional[bytes] = None     # its 32x32 gray copy, for motion
    game = ""                         # newest [game] line; nothing produces one yet

    # main thread: never blocks on anything slow
    while True:
        tick_start = time.monotonic()
        now = timeline.now()

        # --- 1. append everything that arrived since the last tick ---------- #
        lines = uplink.finals() if uplink else drain(stdin_q)
        # What they were pointing at as they said the last line, if anything. Only
        # the last line becomes the instruction, and only it is planned against.
        selection = uplink.selection if uplink is not None else None
        for i, line in enumerate(lines):
            timeline.append(SPEECH_FINAL, line, **(
                {"selection": selection} if selection and i == len(lines) - 1 else {}))
        # Does what they just said want an answer? Only a line that does becomes
        # the instruction -- "Okay" must not supersede the real one.
        # The question gate is asked at the same time rather than after: both read the
        # same text-only prefix, and on a line that turns out not to be addressed its
        # answer is simply unused -- one wasted token against a serial round trip.
        p_reply, reply_s, addressed = None, None, False
        question_f: Optional[Future] = None
        if lines:
            reply_t0 = time.monotonic()
            context = render_reply(timeline.snapshot())
            reply_f = gate_pool.submit(interaction.gate, context, GATE_REPLY)
            question_f = gate_pool.submit(interaction.gate, context, GATE_QUESTION)
            try:
                p_reply = reply_f.result()
                addressed = p_reply >= args.reply_threshold
                print(f"[interaction] p_reply={p_reply:.3f} addressed={addressed} "
                      f"line={lines[-1]!r}", file=sys.stderr)
            except Exception as e:
                print(f"\n[interaction] reply gate failed: {e}", file=sys.stderr)
            reply_s = round(time.monotonic() - reply_t0, 3)
        # A question is answered in 2b and never becomes the instruction: planning
        # it sends a goal, and that goal replaces whatever was running.
        question: Optional[str] = None
        p_question = None
        if addressed:
            try:
                p_question = question_f.result()
                print(f"[interaction] p_question={p_question:.3f}", file=sys.stderr)
                if p_question >= args.question_threshold:
                    question, addressed = lines[-1], False
            except Exception as e:
                print(f"\n[interaction] question gate failed: {e}", file=sys.stderr)
        if addressed:
            # The last line wins. Two utterances inside one 300ms tick is the
            # ASR splitting a sentence, not two instructions.
            # The running goal is NOT dropped here: it is still running on the
            # client. It is replaced if the new plan sends a goal, else stopped
            # after planning (see 3b) -- the white-building goal at session.jsonl
            # line 17438 ran on unchecked because this used to just forget it.
            task.assign(lines[-1], now)
            replan, replans = None, 0

        new_frame, check = None, None
        if uplink is not None:
            # One slot, newest wins. Frames land every ~3s regardless of speech,
            # so this is not one-per-turn and older ones are never wanted.
            if (img := uplink.frame()) is not None and img != frame:
                new_frame = img
        elif capture.due(now):
            new_frame = capture.grab(now)
        if new_frame:
            new_thumb = frame_thumb(new_frame)
            motion = frame_motion(thumb, new_thumb)
            timeline.append(VISION_FRAME, image=new_frame,
                            motion=None if motion is None else round(motion, 2),
                            path=save_frame(frame_dir, timeline.now(), new_frame))
            frame, thumb = new_frame, new_thumb

            # --- 1b. watch the running goal: arrived? lost? actually halted? - #
            # Before the idle check on purpose: a stop that did not take has to
            # be noticed and resent after the conversation has gone quiet.
            # Spoken only if the goal still belongs to the current instruction --
            # "I'm here at the red building" after they asked for something else
            # would answer a question nobody is asking any more.
            check = monitor.on_frame(frame, motion) if monitor is not None else None
            # "lost" replans too: the target leaving the view is a reason to look
            # for it again, not to end the instruction. Out of replans it becomes
            # a "timeout" below, which is where giving up is spoken.
            if check and check["state"] in ("stepped", "stalled", "lost") \
                    and task.instruction == check["instruction"]:
                # Not the end of the instruction: plan the next step in 3 from
                # this screen. `[done so far]` is what the replan reasons from.
                task.actions.append(step_note(check))
                # Bounded for "stepped" too: every replan can bring new pending steps.
                if replans < args.max_replans:
                    replan = check["state"]
                    print(f"[goals] {check['state']} -- replanning", file=sys.stderr)
                else:
                    check = dict(check, state="timeout")     # out of replans: give up
            if check and check["state"] in ("arrived", "timeout") \
                    and task.instruction == check["instruction"]:
                target = check["target"] or "it"
                line = {"arrived": check.get("say", ""),
                        "timeout": f"I couldn't get {target} done, so I'm stopping.",
                        }[check["state"]]
                print(line)
                if uplink is not None:
                    uplink.say(line, reason=check["state"])
                timeline.append(AGENT_SAID, line, reason=check["state"])
                if check["state"] == "arrived":
                    timeline.append(TASK_DONE, task.instruction, reason="arrived",
                                    **{k: check[k] for k in ("width", "p_arrived", "p_veto")
                                       if k in check})
                print(f"[goals] {check['state']} -- stopping, idle", file=sys.stderr)
                task.clear()
                planned, replan = None, None

        # --- 2b. a question: answer it, touch nothing else ----------------- #
        # Before the idle check, so "what do you see" works with no task too.
        # Task, `planned` and the monitor are left alone: the running goal keeps
        # running and keeps being checked.
        if question:
            answer_t0 = time.monotonic()
            talk: dict = {}
            try:
                events = timeline.snapshot()
                # Each sentence goes out as soon as it closes; see SentenceStream.
                talk = interaction.answer(render(
                    Task(instruction=question), frame, game, recent_dialogue(events),
                    latest_background(events, task.started_at)), say_now)
            except Exception as e:
                print(f"\n[interaction] answer failed: {e}", file=sys.stderr)
            answered = talk.get("said", "")
            if answered:
                timeline.append(AGENT_SAID, answered, reason="answer")
            timeline.append(LATENCY, "answer", spoke=bool(answered),
                            p_reply=round(p_reply, 4), reply_s=reply_s,
                            p_question=round(p_question, 4),
                            answer_s=round(time.monotonic() - answer_t0, 3),
                            ttft_s=_round(talk.get("ttft_s")),
                            first_s=_round(talk.get("first_s")),
                            goal_phase=monitor.phase if monitor else None)

        # --- 2. no instruction, no tick ------------------------------------- #
        # Not a cheap tick -- no tick. The gate was measured at -0.272 separation
        # on conversational timing, so asking it "should I speak?" against an
        # empty room is asking the one question it cannot answer, 3x a second.
        # Whether to start talking is structural and belongs here, in a rule.
        if task.idle:
            time.sleep(args.tick)
            continue

        # --- 3. an addressed line: plan in the background, reply now -------- #
        # SIMA 2 style: the agent speaks only when spoken to, in the first person,
        # about what it is going to do. The plan -- reasoning, steps, grounding --
        # takes seconds, so it goes to the planner thread and the reply is spoken
        # while it works; its goal goes down tunnel 2 in 3a, on whichever tick it
        # lands. The reply is ACK_PROMPT's, decoded without the plan, so it only
        # repeats back what was asked: it cannot describe a move the plan did not
        # make, only leave one unmentioned. No unprompted remarks, no free chat.
        events = timeline.snapshot()
        messages = render(task, frame, game, recent_dialogue(events),
                          latest_background(events, task.started_at))
        said, ack = "", {}
        continuing, replan = replan, None
        if addressed or continuing:
            note = ""
            if continuing:
                # A replan is a new goal for the same instruction, on purpose.
                planned, replans = None, replans + 1
                pending = check.get("pending", []) if check else []
                note = PLAN_REPLAN.format(
                    why=replan_why(continuing, check),
                    pending="; ".join(step_text(s) for s in pending) or "none")
            # Submitted before the reply is decoded, so the two run side by side.
            # A replan plans nothing from the pointing: the step it is replacing
            # already ended, and the pinch that started it is long released.
            plan_job = PlanJob(task.instruction, continuing,
                               planner.submit(interaction.plan, messages, frame, note,
                                              selection if addressed else None))
        if addressed:
            # A replan says nothing here: the user already heard what we are doing,
            # and a step is not news. Sentences go out as they close (tunnel 1 only;
            # goals go elsewhere).
            try:
                ack = interaction.acknowledge(messages, say_now)
            except Exception as e:
                print(f"\n[interaction] reply failed: {e}", file=sys.stderr)
            said = ack.get("said", "")
        if said:
            # AGENT_SAID records what actually reached the user. Once TTS is
            # wired up this must be truncated to the audible prefix on barge-in,
            # or every later turn is built on a lie about what was heard.
            timeline.append(AGENT_SAID, said)
            task.actions.append(said)

        # --- 3a. a plan landed: send its goal ------------------------------- #
        # Checked every tick, so a plan that lands while nobody is talking still
        # goes out within one tick of it. A plan for an instruction that has since
        # been replaced or finished is dropped: its goal answers a request, and a
        # screen, that no longer exist.
        plan_s, turned, landed = None, False, None
        if plan_job is not None and plan_job.future.done():
            job, plan_job = plan_job, None
            plan_s = round(time.monotonic() - job.t0, 3)
            try:
                plan = job.future.result()
            except Exception as e:
                plan = Plan()
                print(f"\n[goals] plan failed: {e}", file=sys.stderr)
            if job.instruction != task.instruction:
                print(f"[goals] dropped the plan for {job.instruction!r}: superseded",
                      file=sys.stderr)
            else:
                landed, entry, sent = job, plan.entry, False
                # Once per plan. There is no dedupe on the client, so a goal sent
                # twice runs twice; `planned` stops the resend, and it survives
                # reconnects because we never replay.
                if goals is not None and entry and planned != task.instruction \
                        and goals.send(entry):
                    planned, sent = task.instruction, True
                    turned = "turn" in entry and not plan.pending
                    monitor.started(task.instruction, entry, plan.done_when, plan.bbox,
                                    plan.done_say, plan.pending, plan.target)
                    timeline.append(GOAL_SENT, json.dumps(entry), done_when=plan.done_when,
                                    bbox=plan.bbox, done_say=plan.done_say,
                                    reasoning=plan.reasoning, pending=plan.pending,
                                    replan=job.continuing)
                # The person has already heard "on it", so a plan that came to
                # nothing is said out loud and the instruction dropped -- otherwise
                # the agent promised a move and silently never made it.
                if (job.continuing and not sent) or (not job.continuing and entry is None):
                    line = "I couldn't find a way to finish that, so I'm stopping." \
                        if job.continuing else (plan.say or CANT_PLAN_SAY)
                    say_now(line, reason="lost" if job.continuing else None)
                    timeline.append(AGENT_SAID, line, reason="replan_failed"
                                    if job.continuing else "plan_failed")
                    if job.continuing and monitor is not None and monitor.running:
                        monitor.stop("replan_failed")
                    task.clear()
                    planned = None
            # --- 3b. superseded without a replacement goal: stop the old one - #
            if monitor is not None and monitor.running \
                    and monitor.run.instruction != task.instruction:
                monitor.stop("superseded")

        # (Arrival is checked in 1b, by GoalMonitor, once per new frame.)

        # --- 4. if it needs real work, delegate to the background thread ---- #
        # Only checked on ticks with new user input: delegation is rare, and this
        # keeps the steady-state cost of a working tick at one gate call.
        p_delegate = None
        if addressed and background.inflight == 0:
            try:
                p_delegate = interaction.gate(messages, GATE_DELEGATE)
                if p_delegate >= args.delegate_threshold:
                    events = timeline.snapshot()
                    background.dispatch(interaction.delegation_task(messages),
                                        context_dump(events))
            except Exception as e:
                print(f"\n[interaction] delegate check failed: {e}", file=sys.stderr)

        # --- 5. finished? then the instruction goes away -------------------- #
        # Only after acting, which now means once the plan has landed (3a): asked
        # on the reply's tick, it would judge "Heading to the door" against a
        # screen where nothing has been sent yet. Re-rendered so the gate sees the
        # reply appended above. A replan's landing does not ask either -- that
        # step's end is the monitor's to report.
        # A goal the monitor is watching is ended by the monitor, not here: this
        # runs on the very tick the goal is sent, and in session.jsonl it cleared
        # goals <1s after sending (lines 15008->15010, 15027->15029).
        p_done = None
        watched = monitor is not None and monitor.running \
            and monitor.run.instruction == task.instruction \
            and (monitor.run.tracker is not None or bool(monitor.run.done_when))
        if turned:
            timeline.append(TASK_DONE, task.instruction, reason="turned")
            print(f"[interaction] done: {task.instruction!r} (turn sent) -- idle",
                  file=sys.stderr)
            task.clear()
            planned = None
        elif landed is not None and not landed.continuing and not task.idle \
                and not watched:
            try:
                p_done = interaction.gate(render(task, frame, game), GATE_DONE)
                if p_done >= args.done_threshold:
                    timeline.append(TASK_DONE, task.instruction,
                                    actions=len(task.actions))
                    print(f"[interaction] done: {task.instruction!r} -- idle",
                          file=sys.stderr)
                    if monitor is not None and monitor.running:
                        monitor.stop("done_gate")   # idle here means idle in the game too
                    task.clear()
                    planned = None
            except Exception as e:
                print(f"\n[interaction] done check failed: {e}", file=sys.stderr)

        timeline.append(
            LATENCY, "tick", spoke=bool(said),
            p_reply=None if p_reply is None else round(p_reply, 4), reply_s=reply_s,
            p_delegate=None if p_delegate is None else round(p_delegate, 4),
            p_done=None if p_done is None else round(p_done, 4),
            # plan_s is on the tick the plan LANDED (submit to result), not the
            # tick it was asked for; ack_* are the reply's, time to first sentence
            # being the one the person feels.
            plan_s=plan_s, ack_ttft_s=_round(ack.get("ttft_s")),
            ack_first_s=_round(ack.get("first_s")), ack_s=_round(ack.get("total_s")),
            n_actions=len(task.actions), bg_inflight=background.inflight,
            fresh_input=bool(lines), goal_phase=monitor.phase if monitor else None,
        )

        # --- 6. else: continue ---------------------------------------------- #
        if (slack := args.tick - (time.monotonic() - tick_start)) > 0:
            time.sleep(slack)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interaction] bye")
