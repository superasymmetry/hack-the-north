"""The goals coming back down, from the process with the socket to the process with the sim.

The mirror of [frames.py](frames.py), and deliberately not the same shape. A frame is
worthless once a newer one exists, so that channel is one slot and last writer wins. A goal
is a *command* -- somebody said it and is waiting to see it happen -- so this one is a queue:
goals arrive in order, leave in order, and none is overwritten by the next.

    # in the client process, as the server answers
    spool.submit({"point": "the oak tree", "interaction": "Mine", "stop": {"item": "log"}})

    # in the rocket2 process, between goals
    entry = spool.take()                        # the oldest, or None

The entry is a **plan entry** -- the same JSON `--plan` files hold and `mcagents.cli.plan`
documents. That is the whole reason this is a thin channel rather than a protocol: the goal
vocabulary already existed, a planner was always meant to emit it, and the server is just a
planner that happens to be listening to a person.

One file per goal, so the queue needs no locking and no reader-writer agreement beyond
`os.replace` -- each goal appears whole or not at all, and the order is in the names. A
crash between reading a goal and deleting it can redeliver that goal once; the alternative
is losing it, and for "chop the tree" the second is worse than the first.

Staleness is the same argument `Outbox` makes about a final. A goal that has sat in the
spool for a minute is not the goal anyone meant any more -- the world has moved, and the
person has probably said something else since -- so it is dropped on the way out and
counted, rather than executed late.

The same queue runs the other way too. `StatusChannel` is how the rocket2 process says what
became of each goal -- started, ended and why, rejected and why -- to the client, which sends
it up the goal tunnel. A status is small and ordered like a goal, so it is the same shape of
channel, pointed at its own directory.
"""
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Union

#: Where goals queue up. Both processes are on this laptop, so this is a runtime path.
DEFAULT_DIR = os.environ.get("MCAGENTS_GOAL_DIR", "/tmp/mcagents-goals")

#: Where goal statuses queue for the client to send upstream.
DEFAULT_STATUS_DIR = os.environ.get("MCAGENTS_STATUS_DIR", "/tmp/mcagents-status")

#: How long a goal may wait to be picked up. Longer than a frame's ten seconds, because a
#: goal is worth waiting for -- ROCKET-2 may be finishing the previous one -- and shorter
#: than the time it takes for "mine the oak log" to stop meaning anything.
DEFAULT_MAX_AGE = 60.0

#: A goal that cannot plausibly be one. Same reasoning as frames.MAX_BYTES: this is a
#: well-known directory and a plan entry is a few hundred bytes.
MAX_BYTES = 64 * 1024

#: The key of an in-place turn, `{"turn": 90}`. Not a plan entry: a turn has no target, so
#: nothing about it goes near the targeter or the policy.
TURN_KEY = "turn"

#: The largest turn either way, in degrees. Anything past it is the same turn the other way.
MAX_TURN = 180


def parse_turn(entry: Dict[str, Any]) -> Union[int, float]:
    """The yaw of a turn command, in degrees: positive right, negative left.

    Checked in both processes -- the client, so a bad one is refused where the person can see
    it, and the agent, because the spool is a well-known directory. A turn travels alone: one
    that also carries a goal key is refused whole rather than half-run, since whichever half
    were dropped, the result is not what was said. Raises ValueError saying what is wrong.
    """
    extra = sorted(set(entry) - {TURN_KEY, "kind"})
    if extra:
        raise ValueError(f"a turn is sent on its own, but this one also carries "
                         f"{', '.join(extra)}")
    degrees = entry.get(TURN_KEY)
    if (isinstance(degrees, bool) or not isinstance(degrees, (int, float))
            or not math.isfinite(degrees)):
        raise ValueError(f"turn must be a number of degrees, got {degrees!r}")
    if not 0 < abs(degrees) <= MAX_TURN:
        raise ValueError(f"turn must be within 0 < |turn| <= {MAX_TURN}, got {degrees}")
    return degrees


def _plain(value: Any) -> Any:
    """numpy scalars and arrays as the JSON they obviously are; anything else still raises."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class GoalSpool:
    """The queue itself. Safe to construct in both processes; neither owns it.

    `submit` and `take` are the two halves and never run in the same process in practice,
    but nothing here assumes that -- the tests drive both ends, and so does `--listen
    --self-test`.
    """

    def __init__(self, directory: str = DEFAULT_DIR, max_age: float = DEFAULT_MAX_AGE,
                 max_pending: Optional[int] = None, on_drop=None):
        self.directory = directory
        self.max_age = max_age
        #: Oldest entries are discarded past this many. None for no limit -- right for goals,
        #: which someone is reading; wrong for statuses, which nobody may be.
        self.max_pending = max_pending
        #: Called as on_drop(reason, name) for every entry `take` discards, so a dropped goal
        #: is a log line and not a silence. Reasons: stale, oversized, malformed, not_an_object.
        self.on_drop = on_drop
        self.dropped = 0                  # goals that went stale waiting, for the summary

    # ------------------------------------------------------------------ writing

    def submit(self, entry: Dict[str, Any]) -> str:
        """Queue one goal. Returns the path it landed at.

        The name is `<nanoseconds>-<pid>.json`: nanoseconds so a lexicographic sort is a
        chronological one, the pid so two writers cannot collide on the same tick.
        """
        os.makedirs(self.directory, exist_ok=True)
        name = f"{time.time_ns():020d}-{os.getpid()}.json"
        path = os.path.join(self.directory, name)
        temporary = f"{path}.tmp"
        payload = json.dumps(entry, default=_plain).encode()
        try:
            with open(temporary, "wb") as handle:
                handle.write(payload)
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        if self.max_pending is not None:
            names = self._names()
            for name in names[:max(0, len(names) - self.max_pending)]:
                try:
                    os.unlink(os.path.join(self.directory, name))
                except OSError:
                    pass
        return path

    # ------------------------------------------------------------------ reading

    def _names(self) -> List[str]:
        try:
            return sorted(name for name in os.listdir(self.directory)
                          if name.endswith(".json"))
        except OSError:
            return []                     # no directory yet: nobody has submitted anything

    def pending(self) -> int:
        return len(self._names())

    def take(self) -> Optional[Dict[str, Any]]:
        """The oldest goal still worth running, removed from the queue. None if there is none.

        Everything that can go wrong with a file here ends the same way -- the goal is
        dropped and the next one is tried -- because a malformed or vanished entry is not a
        reason to stop listening. A goal that loses a race to another reader simply is not
        there any more, which is indistinguishable from never having existed.
        """
        for name in self._names():
            path = os.path.join(self.directory, name)
            try:
                age = time.time() - os.path.getmtime(path)
                if age > self.max_age:
                    os.unlink(path)
                    self._drop("stale", name)
                    continue
                with open(path, "rb") as handle:
                    raw = handle.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    os.unlink(path)
                    self._drop("oversized", name)
                    continue
                entry = json.loads(raw)
                os.unlink(path)
            except FileNotFoundError:
                continue                  # another reader took it: it was never ours
            except (OSError, ValueError):
                try:
                    os.unlink(path)
                except OSError:
                    pass
                self._drop("malformed", name)
                continue
            if isinstance(entry, dict):
                return entry
            self._drop("not_an_object", name)
        return None

    def _drop(self, reason: str, name: str) -> None:
        self.dropped += 1
        if self.on_drop is not None:
            self.on_drop(reason, name)

    def wait(self, timeout: float, poll: float = 0.05) -> Optional[Dict[str, Any]]:
        """`take`, but blocking for up to `timeout` seconds. None if nothing arrived.

        A timeout rather than a block, because the caller has something else to do while it
        waits: the sim has to keep stepping or the world freezes, the frame channel goes
        stale, and the client stops sending the pictures this loop runs on.
        """
        deadline = time.monotonic() + timeout
        while True:
            entry = self.take()
            if entry is not None:
                return entry
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(poll, max(0.0, deadline - time.monotonic())))

    def clear(self) -> int:
        """Empty the queue. Returns how many goals were discarded."""
        removed = 0
        for name in self._names():
            try:
                os.unlink(os.path.join(self.directory, name))
                removed += 1
            except OSError:
                pass
        return removed


class StatusChannel:
    """What became of each goal, from the rocket2 process to the client's goal tunnel.

    Every message is `{"kind": "goal.status", "event", "reason", "box", "t", ...}`:

        event   started | updated | ended | rejected
        reason  why -- arrived, steps, max_steps, cancelled, replaced, lost, terminated,
                item, stat, turned for `ended`; keep_memory for `updated`; idle, no_target,
                not_found, targeter_error, invalid, stale, stale_followup, replaced,
                cancelled for `rejected`
        box     [x0, y0, x1, y1] of the target as fractions of the frame, or null
        t       this laptop's wall clock, epoch seconds

    Writing one must never be what stops a goal, so a failure is reported once and the
    channel goes quiet.
    """

    def __init__(self, directory: str = DEFAULT_STATUS_DIR, report=None):
        self.spool = GoalSpool(directory, max_age=DEFAULT_MAX_AGE, max_pending=256)
        self.report = report or (lambda message: print(message, flush=True))
        self.enabled = True

    def emit(self, event: str, reason: Optional[str] = None,
             box: Optional[List[float]] = None, **extra: Any) -> Dict[str, Any]:
        message = {"kind": "goal.status", "event": event, "reason": reason,
                   "box": [round(float(v), 4) for v in box] if box is not None else None,
                   "t": round(time.time(), 3), **extra}
        if self.enabled:
            try:
                self.spool.submit(message)
            except OSError as exc:
                self.enabled = False
                self.report(f"[status] cannot write to {self.spool.directory}, giving up on it "
                            f"({type(exc).__name__}: {exc})")
            except (TypeError, ValueError) as exc:
                self.report(f"[status] could not encode a {event} status, skipped it "
                            f"({type(exc).__name__}: {exc})")
        return message

    def clear(self) -> int:
        return self.spool.clear()
