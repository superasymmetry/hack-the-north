"""Run ROCKET-2 against a live Minecraft, one pointed goal at a time.

    bash scripts/rocket2.sh                     # launches its own Minecraft
    MC_PORT=9000 bash scripts/rocket2.sh        # attaches to a scripts/mc_server.sh holder
    python -m mcagents.cli.rocket2 --plan my_plan.json --no-city

    # or say what you want in words, without writing a plan file at all:
    bash scripts/rocket2.sh --goal "the low white building across the street"
    bash scripts/rocket2.sh --goal "the yellow storefront on the corner" \
                            --interaction Approach --stop 400

--goal is the whole pipeline in one flag: the description goes to the targeter, which
returns a point; SAM-2 turns that point into the mask ROCKET-2 actually takes as its goal.
Because a *description* is the thing OWLv2 is worst at -- it matches bare nouns against
natural-image box priors -- --goal selects the remote VLM targeter unless you have asked
for one yourself with --targeter or $MCAGENTS_TARGETER. Repeat --goal for a sequence.

Each plan entry is the three fields ROCKET-2 takes:

    point       [x, y] in the 640x360 frame, [u, v] in 0..1 with "normalized": true, a
                description of the target ("tree", "cow") for the targeter to find in the
                current frame, or "click" to point at it yourself.
    interaction Hunt | Mine | Use | Interact | Craft | Switch | Approach
    stop        see mcagents.agents.base

The run opens a live window of what the policy is looking at -- the goal mask, ROCKET-2's own
guess at where the target is now, and its visibility estimate -- and writes an mp4 of the raw
game view under logs/rocket2. --no-preview (or ROCKET2_PREVIEW=0) drops the window, and it
drops itself when there is no display to open it on; --no-record drops the mp4. Between them
a headless run still leaves something to watch afterwards.

With --city the world is the one from mcagents.minecraft.city -- streets, buildings, a park
and a fountain plaza -- because a pointing agent needs somewhere to point. It costs ~20s of
every reset and cannot be cached.
"""
import argparse
import collections
import json
import math
import os
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

# Before minestudio, deliberately: this pulls in mcagents.gui, which has to open its first
# OpenCV window while PyAV is still out of the process. See mcagents/gui.py.
from mcagents.agents.rocket2 import Rocket2Agent, Rocket2Config

from mcagents import gui
from mcagents.cli.plan import Plan, load_plan
from mcagents.goals import (DEFAULT_DIR as DEFAULT_GOAL_DIR, TURN_KEY, GoalSpool, StatusChannel,
                            parse_turn)
from mcagents.minecraft.session import EnvConfig, Session
from mcagents.perception.detection import Targeter, load_targeter

#: What a supervising planner would emit -- a word, not pixels, because resolving the word is
#: the part worth demonstrating. The run spawns at a crossroads facing a brick tower across a
#: plaza, with more of the city down each street, so "building" is in frame however the spawn
#: is oriented; "tree" is not, which is why this is not a gathering goal.
DEFAULT_PLAN: Plan = [
    {"point": "building", "interaction": "Approach", "stop": {"steps": 300}},
]

#: What --goal defaults to when --stop is not given. Long enough to cross a street and
#: arrive, short enough that a goal aimed at nothing does not eat the whole run.
DEFAULT_GOAL_STOP: Any = {"steps": 300}

#: Steps per second while --listen waits for a goal. Unpaced, the idle loop renders and
#: previews as fast as the machine allows for as long as the process lives, which on a
#: laptop is sustained full load with nobody talking. 5 Hz keeps the published frame well
#: inside the client's 1s/3s cadence; the world runs at a quarter of Minecraft's 20 Hz
#: while nothing is asked of it, which nothing is looking closely enough to notice.
DEFAULT_IDLE_HZ = 5.0

#: Most degrees of yaw per step of an in-place turn. The sim would take 180 in one step (it
#: turns exactly what it is told), but the frames going up would show a jump cut rather than
#: a turn. At 15, 90 degrees is 6 steps and 180 is 12.
DEFAULT_TURN_STEP = 15.0


def parse_stop(spec: Optional[str]) -> Any:
    """`--stop` as the shared vocabulary in `mcagents.agents.base` writes it.

    A bare integer is a step budget, and anything else has to be the JSON object form --
    `'{"item": "oak_log", "count": 3}'`. Parsed here rather than in the agent so a typo is a
    usage error before Minecraft boots, not 30s later.
    """
    if spec is None:
        return DEFAULT_GOAL_STOP
    text = spec.strip()
    if text.lstrip("-").isdigit():
        return int(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise SystemExit(f"--stop {spec!r} is neither a step count nor JSON, e.g. "
                         f"--stop 300 or --stop '{{\"item\": \"oak_log\", \"count\": 3}}'")
    if not isinstance(parsed, (int, dict)):
        raise SystemExit(f"--stop {spec!r} must be a number or a JSON object")
    return parsed


def inline_plan(goals: Sequence[str], interaction: str, stop: Optional[str]) -> Plan:
    """The plan `--goal` describes: one entry per description, in the order they were given.

    `point` carries the description verbatim -- PointResolver already treats any string as
    something for the targeter to find, so a typed description and one read out of a plan
    file take the identical path from here on.
    """
    return [{"point": text, "interaction": interaction, "stop": parse_stop(stop)}
            for text in goals]


def targeter_backend(args: argparse.Namespace) -> Optional[str]:
    """Which targeter to build: what was asked for, else "vlm" when --goal was used.

    Returning None leaves `load_targeter` to read $MCAGENTS_TARGETER and fall back to OWLv2,
    which stays the default for plan files and bare-noun goals.
    """
    if args.targeter:
        return args.targeter
    if args.goal and not os.environ.get("MCAGENTS_TARGETER"):
        return "vlm"
    return None


class PointResolver:
    """Turns a plan entry's `point` into pixel coordinates.

    "click" asks a human; any other string is a target description for the targeter to find
    in the current frame; anything else is already [x, y]. Only the middle case lets a
    machine write a whole plan entry, which is the point of it.

    The targeter is built on first use, and only if a plan actually describes a target in
    words: OWLv2 is another ~600 MB on a card already holding ROCKET-2 and SAM-2.
    """

    def __init__(self, backend: Optional[str] = None):
        self.backend = backend
        self._targeter: Optional[Targeter] = None

    @property
    def targeter(self) -> Targeter:
        if self._targeter is None:
            self._targeter = load_targeter(self.backend)
            print(f"[plan] targeting with {type(self._targeter).__name__}", flush=True)
        return self._targeter

    def resolve(self, agent: Rocket2Agent, spec: Any) -> Tuple[float, float]:
        """The point to aim this goal at. Raises LookupError if there is nowhere to aim."""
        if spec == "click":
            print("[plan] click the target in the window (ESC to skip this goal)", flush=True)
            point = gui.pick_point(agent.frame)
            if point is None:
                raise LookupError("no point selected")
            return point
        if isinstance(spec, str):
            point = self.targeter.locate(agent.frame, spec)
            if point is None:
                raise LookupError(
                    f"nothing in view matches {spec!r} -- lower MCAGENTS_OWL_THRESHOLD (OWLv2) "
                    f"or reword the query (VLM). Check a still frame with: "
                    f"python -m mcagents.cli.locate <image> {spec!r}")
            print(f"[plan] {spec!r} -> ({point[0]:.0f}, {point[1]:.0f})", flush=True)
            return point
        return tuple(spec)


#: What `set_goal` will actually take. A goal arriving from a server may carry more than
#: that -- a JarvisVLA-style `instruction`, a `kind`, whatever the planner found useful --
#: and passing it straight through is a TypeError two hours into a demo.
SET_GOAL_KEYS = ("interaction", "stop", "normalized", "keep_memory")

#: Keys the loop reads itself and never hands to `set_goal`, so they are not "unknown".
LOOP_KEYS = ("id", "cancel")

#: The message that halts. See `is_halt`.
CANCEL_GOAL: Dict[str, Any] = {"cancel": True}

#: Seconds after a goal ends during which a `keep_memory: true` resend of it is refused
#: rather than restarting it. A planner refreshing its goal every ~3 s has one resend in
#: flight when the goal ends, and running it would undo the very stop that just happened.
DEFAULT_FOLLOWUP_WINDOW = 5.0


def is_halt(entry: Dict[str, Any]) -> bool:
    """Whether `entry` means "stop now": cancel the running goal and press nothing after.

    `{"cancel": true}` is the explicit form. `{"interaction": "None", ...}` with no
    coordinate point means the same -- it is the shape a stop has always been sent in, and
    run as a goal it handed its `instruction` ("stand still") to the targeter as something
    to walk toward. A "None" goal *with* [x, y] is still an ordinary goal.
    """
    if entry.get("cancel") is True:
        return True
    if "interaction" not in entry:
        return False
    interaction = entry["interaction"]
    unnamed = interaction is None or (isinstance(interaction, str) and interaction.lower() == "none")
    return unnamed and not isinstance(entry.get("point"), (list, tuple))


def is_turn(entry: Dict[str, Any]) -> bool:
    """Whether `entry` is an in-place turn, `{"turn": degrees}`, valid or not."""
    return TURN_KEY in entry


def turn_pieces(degrees: float, most: float = DEFAULT_TURN_STEP) -> List[float]:
    """`degrees` as equal per-step yaws of at most `most` each, adding up to exactly it."""
    if most <= 0:
        return [float(degrees)]
    steps = max(1, math.ceil(abs(degrees) / most - 1e-9))
    return [degrees / steps] * steps


def expand(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One spool entry as the goals in it: a `{"goals": [...]}` batch is a sequence."""
    goals = entry.get("goals")
    if isinstance(goals, list):
        return [goal for goal in goals if isinstance(goal, dict)]
    return [entry]


def _interaction(entry: Dict[str, Any]) -> str:
    return str(entry.get("interaction") or "Approach").lower()


class GoalLoop:
    """Run goals, and listen for the next one *while* each runs.

    Checking for new goals only between them was the original design, and it is why a stop
    sent 68 s into a session took effect at 113 s: it sat in the queue until the Approach in
    front of it ran out its step ceiling. Now the spool is read before every step, and what
    arrives decides what happens to the goal in progress:

        a halt (`is_halt`)             cancel now -- "cancelled"; no-op steps follow
        an invalid turn                refused -- the goal in progress carries on
        `keep_memory: true`, same      merge -- keep the lock, the policy state and the
          interaction (and same id)      step count; take the new stop
        anything else                  preempt -- "replaced"; the new goal runs next

    Every outcome goes out on `status`, when there is one.
    """

    def __init__(self, agent: Rocket2Agent, resolver: "PointResolver",
                 spool: Optional[GoalSpool] = None, status: Optional[StatusChannel] = None,
                 followup_window: float = DEFAULT_FOLLOWUP_WINDOW,
                 turn_step: float = DEFAULT_TURN_STEP):
        self.agent = agent
        self.resolver = resolver
        self.spool = spool
        self.status = status
        self.followup_window = followup_window
        self.turn_step = turn_step
        self.pending: Deque[Dict[str, Any]] = collections.deque()
        self.running: Optional[Dict[str, Any]] = None
        #: (interaction, id, reason, monotonic time) of the last goal to end.
        self.last_end: Optional[Tuple[str, Any, str, float]] = None

    # ------------------------------------------------------------------ plumbing

    def emit(self, event: str, reason: Optional[str], entry: Optional[Dict[str, Any]] = None,
             box=None, **extra: Any) -> None:
        if self.status is None:
            return
        if entry is not None and not is_turn(entry):
            extra.setdefault("interaction", entry.get("interaction", "Approach")
                             if not is_halt(entry) else "None")
            if "id" in entry:
                extra.setdefault("id", entry["id"])
        self.status.emit(event, reason, box, **extra)

    def next(self) -> Optional[Dict[str, Any]]:
        """The next goal to run: what a preemption queued, else the spool's oldest."""
        while not self.pending and self.spool is not None:
            entry = self.spool.take()
            if entry is None:
                return None
            self.pending.extend(expand(entry))
        return self.pending.popleft() if self.pending else None

    def _drop_pending(self, reason: str, label: str) -> None:
        for entry in self.pending:
            print(f"[{label}] dropped a queued goal -- {reason}", flush=True)
            self.emit("rejected", reason, entry)
        self.pending.clear()

    def _same_goal(self, entry: Dict[str, Any], interaction: str, goal_id: Any) -> bool:
        if _interaction(entry) != interaction.lower():
            return False
        return "id" not in entry or goal_id is None or entry["id"] == goal_id

    # ------------------------------------------------------------------ one goal

    def run(self, entry: Dict[str, Any], label: str) -> bool:
        """Point, set, and step until it ends. False if the episode is over."""
        agent = self.agent
        if is_turn(entry):
            return self.turn(entry, label)
        if is_halt(entry):
            print(f"[{label}] cancel -- nothing is running; staying put", flush=True)
            self.emit("rejected", "idle", entry)
            return True

        if entry.get("keep_memory") is True and self.last_end is not None:
            interaction, goal_id, reason, ended = self.last_end
            if (time.monotonic() - ended < self.followup_window
                    and self._same_goal(entry, interaction, goal_id)):
                print(f"[{label}] skipped -- a keep_memory resend of a goal that ended "
                      f"{time.monotonic() - ended:.1f}s ago ({reason}); not restarting it",
                      flush=True)
                self.emit("rejected", "stale_followup", entry, after=reason)
                return True

        entry = dict(entry)
        spec = entry.pop("point", None)
        if spec is None:
            # ROCKET-2's goal channel is a point, not a sentence. A planner that sent an
            # instruction still said what it wants looked at, so hand the words to the
            # targeter rather than refusing a goal that is nearly right.
            spec = entry.pop("instruction", None)
            if spec is not None:
                print(f"[{label}] no point given; targeting the instruction {spec!r}", flush=True)
        entry.pop("instruction", None)
        if spec is None:
            print(f"[{label}] skipped -- the goal names no point and no instruction", flush=True)
            self.emit("rejected", "no_target", entry)
            return True

        try:
            point = self.resolver.resolve(agent, spec)
        except LookupError as unpointable:
            # One goal with no target in frame is not a reason to tear the world down: the next
            # goal may well have one, and the window and the mp4 keep going.
            print(f"[{label}] skipped -- {unpointable}", flush=True)
            self.emit("rejected", "not_found", entry)
            return True
        except Exception as broken:
            # A VLM that is down raises a requests error, not a LookupError, and that used
            # to end the whole rocket2 process from inside a listener meant to outlive it.
            print(f"[{label}] skipped -- the targeter failed: {type(broken).__name__}: {broken}",
                  flush=True)
            self.emit("rejected", "targeter_error", entry)
            return True

        fields = {key: entry[key] for key in SET_GOAL_KEYS if key in entry}
        ignored = sorted(set(entry) - set(fields) - set(LOOP_KEYS))
        if ignored:
            print(f"[{label}] ignoring unknown goal key(s): {', '.join(ignored)}", flush=True)

        print(f"\n[{label}] {fields.get('interaction', 'Approach')} at {point}", flush=True)
        try:
            agent.set_goal(point=point, **fields)
        except (ValueError, TypeError) as unusable:
            print(f"[{label}] skipped -- {unusable}", flush=True)
            self.emit("rejected", "invalid", entry, detail=str(unusable))
            return True

        goal = agent.goal
        budget = goal.stop.get("steps") if isinstance(goal.stop, dict) else None
        if budget is not None and int(budget) > agent.max_steps:
            print(f"[{label}] note: a step budget of {budget} is capped by max_steps="
                  f"{agent.max_steps}", flush=True)
        self.running = entry
        self.emit("started", None, entry, box=agent.goal_box(), locked=goal.lock is not None,
                  stop=goal.stop if not callable(goal.stop) else repr(goal.stop))

        while agent.busy:
            if self.spool is not None and self._poll(label):
                break
            agent.step()
            if goal.steps % 50 == 0:
                lock = goal.lock
                held = "" if lock is None else (
                    f"  lock {'ok' if lock.fresh else f'unconfirmed {lock.stale} steps'}"
                    f"  box {[round(v, 2) for v in (lock.normalized_box() or ())]}")
                print(f"  step {goal.steps:>3}  visibility {agent.visibility:.2f}{held}", flush=True)

        result = agent.result
        print(f"[{label}] {result}", flush=True)
        self.running = None
        self.last_end = (goal.interaction, entry.get("id"), result.reason, time.monotonic())
        self.emit("ended", result.reason, entry, box=agent.goal_box(), steps=result.steps,
                  seconds=round(result.seconds, 2))
        if agent.terminated:
            print(f"[{label}] episode ended (death or reset) -- stopping here.")
            return False
        return True

    def turn(self, entry: Dict[str, Any], label: str) -> bool:
        """Turn in place by `entry["turn"]` degrees, then stand still. False if the episode is over.

        Camera only: no movement, no attack or use, and nothing asked of the policy or the
        targeter. By the time this runs whatever was going on has already been cancelled
        (`_poll` treats a turn like any other new goal), and it is not resumed afterwards --
        `listen` goes back to no-op steps. The policy's memory is cleared with it, since the
        view that memory was built on is now somewhere else.

        Not interruptible: 180 degrees is 12 steps, well under a second, and a cancel or goal
        that lands meanwhile is read the moment it finishes.
        """
        agent = self.agent
        try:
            degrees = parse_turn(entry)
        except ValueError as invalid:
            print(f"[{label}] skipped -- {invalid}", flush=True)
            self.emit("rejected", "invalid", entry, turn=entry.get(TURN_KEY), detail=str(invalid))
            return True

        pieces = turn_pieces(degrees, self.turn_step)
        print(f"\n[{label}] turn {degrees:+g} deg ({'right' if degrees > 0 else 'left'}) "
              f"over {len(pieces)} step{'s' * (len(pieces) != 1)}", flush=True)
        if hasattr(agent, "clear_memory"):
            agent.clear_memory()
        started = time.monotonic()
        steps = 0
        for piece in pieces:
            alive = agent.turn_step(piece)
            steps += 1
            if not alive:
                break
        seconds = time.monotonic() - started
        reason = "terminated" if agent.terminated else "turned"
        print(f"[{label}] {reason} {degrees:+g} deg in {steps} steps ({seconds:.2f}s)", flush=True)
        self.last_end = ("turn", None, reason, time.monotonic())
        self.emit("ended", reason, entry, turn=degrees, steps=steps, seconds=round(seconds, 2))
        if agent.terminated:
            print(f"[{label}] episode ended (death or reset) -- stopping here.")
            return False
        return True

    def _poll(self, label: str) -> bool:
        """Read what arrived since the last step. True if the running goal was ended by it."""
        agent = self.agent
        while True:
            incoming = self.spool.take()
            if incoming is None:
                return False
            batch = expand(incoming)
            if not batch:
                continue
            if len(batch) == 1 and is_turn(batch[0]):
                try:
                    parse_turn(batch[0])
                except ValueError as invalid:
                    print(f"[{label}] refused a turn, the goal carries on -- {invalid}", flush=True)
                    self.emit("rejected", "invalid", batch[0], turn=batch[0].get(TURN_KEY),
                              detail=str(invalid))
                    continue
            if len(batch) == 1 and is_halt(batch[0]):
                print(f"[{label}] cancel received at step {agent.goal.steps} -- halting", flush=True)
                self._drop_pending("cancelled", label)
                agent.cancel("cancelled")
                return True
            if (len(batch) == 1 and batch[0].get("keep_memory") is True
                    and self._same_goal(batch[0], agent.goal.interaction,
                                        (self.running or {}).get("id"))):
                self._merge(batch[0], label)
                continue
            print(f"[{label}] a new goal arrived at step {agent.goal.steps} -- replacing this one",
                  flush=True)
            self._drop_pending("replaced", label)
            self.pending.extend(batch)
            agent.cancel("replaced")
            return True

    def _merge(self, entry: Dict[str, Any], label: str) -> None:
        agent = self.agent
        point = entry.get("point")
        coordinates = point if isinstance(point, (list, tuple)) and len(point) == 2 else None
        try:
            outcome = agent.update_goal(stop=entry.get("stop"), point=coordinates,
                                        normalized=bool(entry.get("normalized", False)),
                                        has_stop="stop" in entry)
        except (ValueError, TypeError) as unusable:
            print(f"[{label}] follow-up refused, the goal carries on -- {unusable}", flush=True)
            self.emit("rejected", "invalid", entry, detail=str(unusable))
            return
        self.emit("updated", "keep_memory", entry, box=agent.goal_box(),
                  steps=agent.goal.steps, agrees=outcome["agrees"])


def run_goal(agent: Rocket2Agent, resolver: "PointResolver", entry: Dict[str, Any],
             label: str) -> bool:
    """Point, set, and step until it ends. False if the episode is over.

    Factored out of the plan loop because a goal read off a socket and a goal read out of a
    file have to behave identically -- and the moment they do not, the one that has never
    been run by hand is the one that is broken. A plan has no spool, so nothing preempts it.
    """
    return GoalLoop(agent, resolver).run(entry, label)


def listen(agent: Rocket2Agent, resolver: "PointResolver", spool: GoalSpool,
           idle_hz: float = DEFAULT_IDLE_HZ, status: Optional[StatusChannel] = None,
           followup_window: float = DEFAULT_FOLLOWUP_WINDOW,
           turn_step: float = DEFAULT_TURN_STEP) -> None:
    """Take goals from the spool for as long as the episode lasts.

    The world keeps stepping between goals, which is the whole reason this is a loop and
    not a blocking read: a frozen sim stops publishing frames within seconds, and the
    planner on the other end of the tunnel is then choosing what to do next from a picture
    of a world that has stopped. Idling costs a no-op action per step and buys a live view
    -- at `idle_hz`, not flat out, because a wait has no reason to be faster than the frames
    anyone is reading. 0 lifts the cap. The no-op is also what "halted" means: a goal that
    ends for any reason -- arrived, out of steps, cancelled -- is followed by no-ops and
    nothing else until the next goal.

    Goals queued before this process existed are discarded on the way in. They were said to
    a different session, about a world that no longer exists -- the spawn has moved, the
    inventory is empty again -- and running them would be the most confusing possible start.
    """
    stale = spool.clear()
    if status is not None:
        status.clear()
    loop = GoalLoop(agent, resolver, spool, status, followup_window, turn_step)

    def dropped(reason: str, name: str) -> None:
        print(f"[listen] dropped goal {name} from the spool -- {reason}", flush=True)
        loop.emit("rejected", reason)

    spool.on_drop = dropped
    print(f"[listen] waiting for goals in {spool.directory}"
          f"{f' ({stale} from an earlier session discarded)' if stale else ''}\n"
          f"[listen] talk to it with: ./scripts/local_client.sh", flush=True)

    period = 1.0 / idle_hz if idle_hz > 0 else 0.0
    goals = 0
    while True:
        entry = loop.next()
        if entry is None:
            started = time.monotonic()
            if not agent.idle_step():
                print("[listen] episode ended (death or reset) -- stopping here.")
                return
            time.sleep(max(0.0, period - (time.monotonic() - started)))
            continue
        goals += 1
        if not loop.run(entry, f"listen {goals}"):
            return


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", help="JSON file of goals; default is the one in this module")
    parser.add_argument("--goal", action="append", metavar="DESCRIPTION",
                        help="describe the target in words instead of writing a plan file; "
                             "repeat for a sequence of goals")
    parser.add_argument("--interaction", default="Approach",
                        help="the interaction --goal entries use (default Approach)")
    parser.add_argument("--stop", help="stop condition for --goal entries: a step count "
                                       "(300) or the JSON form ('{\"item\": \"oak_log\", "
                                       "\"count\": 3}'); default 300 steps")
    parser.add_argument("--listen", action="store_true",
                        help="after any plan, take goals from the voice client's spool and "
                             "keep going until the episode ends")
    parser.add_argument("--idle-hz", type=float,
                        default=float(os.environ.get("MCAGENTS_IDLE_HZ", DEFAULT_IDLE_HZ)),
                        help=f"steps per second while --listen waits for a goal "
                             f"(default {DEFAULT_IDLE_HZ:g}, 0 = uncapped)")
    parser.add_argument("--goal-dir",
                        help=f"where --listen reads goals (default {DEFAULT_GOAL_DIR})")
    parser.add_argument("--status-dir",
                        help="where --listen writes goal statuses for the client to send "
                             "upstream (default $MCAGENTS_STATUS_DIR or /tmp/mcagents-status)")
    parser.add_argument("--followup-window", type=float, default=DEFAULT_FOLLOWUP_WINDOW,
                        help="seconds after a goal ends during which a keep_memory resend of "
                             f"it is refused (default {DEFAULT_FOLLOWUP_WINDOW:g})")
    parser.add_argument("--turn-step", type=float,
                        default=float(os.environ.get("MCAGENTS_TURN_STEP", DEFAULT_TURN_STEP)),
                        help=f"most degrees of yaw per step of an in-place turn "
                             f"(default {DEFAULT_TURN_STEP:g}, 0 = the whole turn in one step)")
    parser.add_argument("--mc-port", type=int, help="attach to a held instance on this port")
    parser.add_argument("--city", action="store_true", default=None, help="build the city (~20s a reset)")
    parser.add_argument("--no-city", dest="city", action="store_false")
    parser.add_argument("--targeter", choices=["owl", "vlm"], help="how word goals become points")
    parser.add_argument("--cfg", type=float, help="classifier-free guidance strength (0 = fastest)")
    parser.add_argument("--max-steps", type=int, help="hard ceiling per goal")
    parser.add_argument("--record", dest="record", action="store_true", default=None,
                        help="write an mp4 of the run under logs/ (default, except with "
                             "--listen: the recorder holds every frame in RAM until exit)")
    parser.add_argument("--no-record", dest="record", action="store_false",
                        help="do not write an mp4 of the run under logs/")
    parser.add_argument("--no-preview", dest="preview", action="store_false", default=None,
                        help="do not open a live window of what the policy is looking at")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.goal and args.plan:
        raise SystemExit("--goal and --plan both say what to do; pass one or the other")
    # --listen with nothing else asked for waits, rather than approaching a building first:
    # the default plan is a demo of pointing, and here the person is about to say what they
    # want. Given explicitly, a plan still runs and the listener picks up after it.
    plan = (inline_plan(args.goal, args.interaction, args.stop) if args.goal
            else load_plan(args.plan, [] if args.listen else DEFAULT_PLAN))

    env = EnvConfig.from_env(city_default=True)
    if args.mc_port is not None:
        env.mc_port = args.mc_port
    if args.city is not None:
        env.city = args.city

    config = Rocket2Config.from_env()
    if args.cfg is not None:
        config.cfg_coef = args.cfg
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.preview is not None:
        config.preview = args.preview

    resolver = PointResolver(targeter_backend(args))

    from minestudio.simulator.callbacks import PrevActionCallback, RecordCallback

    # 224x224 observations and a PrevActionCallback are both hard requirements of the
    # checkpoint; Rocket2Agent refuses a sim without them.
    callbacks = [PrevActionCallback()]
    # MineStudio's RecordCallback keeps every frame in a list and encodes only on close:
    # ~0.7 MB a frame at 640x360. A plan is bounded by max_steps, so that is fine; --listen
    # has no end, and at 20 steps a second fills 30 GB of RAM in about half an hour and
    # takes the machine down with it. So a listening run records only when asked to.
    record = args.record if args.record is not None else not args.listen
    if record and args.listen:
        print("[rocket2] --record with --listen: frames accumulate in RAM until exit "
              "(~0.7 MB each). Keep the session short.", flush=True)
    if record:
        # The raw game view, the way upstream's own evaluations are inspected -- not the
        # preview overlay, which is drawn after the frame has left the sim. RecordCallback
        # writes the file in before_close, and Session closes the sim however the run ends.
        callbacks.append(RecordCallback(record_path="logs/rocket2", fps=20, frame_type="pov"))

    with Session(env) as session:
        sim = session.open(callbacks=callbacks, action_type="env", obs_size=(224, 224))
        agent = Rocket2Agent(sim, config)

        # The agent only draws its preview after a step, so without this there is nothing on
        # screen until a goal is actually running -- and resolving the first point can take
        # a while (OWLv2 is ~600 MB to load) or fail outright, which used to mean a run that
        # showed the world exactly never.
        if config.preview:
            gui.show(agent.overlay(), "ROCKET-2")

        # run() would do a goal in one call; stepping by hand is what a planner does, so it
        # can watch `visibility` fall and re-point instead of burning the budget.
        for number, entry in enumerate(plan, 1):
            if not run_goal(agent, resolver, entry, f"plan {number}/{len(plan)}"):
                break
        else:
            if args.listen:
                status = (StatusChannel(args.status_dir) if args.status_dir
                          else StatusChannel())
                listen(agent, resolver, GoalSpool(args.goal_dir or DEFAULT_GOAL_DIR),
                       idle_hz=args.idle_hz, status=status,
                       followup_window=args.followup_window, turn_step=args.turn_step)


if __name__ == "__main__":
    main()
