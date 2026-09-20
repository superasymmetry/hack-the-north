"""What the two controllers have in common: one goal at a time, a stop condition, a result.

A controller here is a small state machine. You give it a goal (a point and an interaction
for ROCKET-2, a sentence for JarvisVLA) plus a condition that ends it, then either block on
`run()` or drive `step()` yourself and interleave your own planning:

    agent.set_goal(...)
    while agent.busy:
        agent.step()
    print(agent.result)

The stop vocabulary is shared, so a planner that can drive one controller can drive the
other:

    None                                     run until max_steps
    200 / {"steps": 200}                     a step budget
    {"item": "log", "count": 3}              three more logs than at the start
    {"item": "log", "count": 3, "mode": "total"}   three logs in total
    {"stat": "kill_entity", "match": "sheep", "count": 1}
    {"arrive": {"width": 0.6}}               the locked target's box is 60% of the frame wide
    {"arrive": {"distance": 3}}              the agent is within 3 blocks of the target
    {"steps": 800, "arrive": {"width": 0.6}} whichever comes first
    callable(agent) -> bool                  anything else

`arrive` is the only key checked against what the agent *sees*, so only a controller that
tracks its target can take it (`supports_arrive`). It combines with any of the others as an
OR; `"arrive": true` uses the controller's default width and `"arrive": false` opts a goal
out of a default the controller would otherwise add.
"""
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from mcagents.minecraft.inventory import count_item, inventory_counts, stat_counts

#: A stop condition as a caller writes it. See the module docstring.
StopSpec = Any

#: A compiled stop condition: returns the reason the goal ended, or None to keep going.
StopFn = Callable[["Agent"], Optional[str]]

STOP_KEYS = {"steps", "item", "count", "mode", "stat", "match", "arrive"}

#: The keys an `arrive` object may carry: `width` and `height` are fractions of the frame the
#: target's box must reach, `distance` is blocks between the agent and the target's world
#: position. A controller that can answer `distance` should prefer it -- see `Agent.arrived`.
ARRIVE_KEYS = {"width", "height", "distance"}

#: Reasons a goal can end. These mean the caller got what they asked for; the rest --
#: max_steps, terminated, cancelled, replaced, lost -- mean it was cut short.
SUCCESS_REASONS = frozenset({"item", "stat", "steps", "predicate", "arrived"})


@dataclass
class Goal:
    """One goal and the bookkeeping needed to tell what changed while it ran."""
    stop: StopSpec = None
    start_inventory: Counter = field(default_factory=Counter)
    start_stats: Dict[str, Counter] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    steps: int = 0

    def describe(self) -> str:
        return f"stop={self.stop!r}"


@dataclass
class GoalResult:
    """Why a goal ended, and what the world did while it ran."""
    reason: str
    steps: int
    seconds: float
    gained: Dict[str, int]

    @property
    def success(self) -> bool:
        """True unless the goal ran out of budget or the episode ended under it."""
        return self.reason in SUCCESS_REASONS

    @property
    def gained_text(self) -> str:
        return ", ".join(f"{k}+{v}" for k, v in sorted(self.gained.items())) or "nothing"

    def __str__(self) -> str:
        return (f"{self.reason} after {self.steps} steps ({self.seconds:.1f}s), "
                f"gained {self.gained_text}")


def parse_arrive(value: Any, default_width: float) -> Optional[Dict[str, float]]:
    """An `arrive` value as {"width": w, "height": h, "distance": blocks}, or None when off.

    `true` is the default width, a bare number is a width, and an object names its own
    thresholds. Width and height are fractions of the frame in (0, 1]; distance is a positive
    number of blocks.
    """
    if value is None or value is False:
        return None
    if value is True:
        return {"width": float(default_width)}
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = {"width": float(value)}
    if not isinstance(value, dict) or not value:
        raise ValueError(f"arrive must be true, a width fraction or an object with "
                         f"width/height -- got {value!r}")
    unknown = set(value) - ARRIVE_KEYS
    if unknown:
        raise ValueError(f"unknown arrive keys {sorted(unknown)}; expected width/height")
    parsed = {}
    for key, threshold in value.items():
        threshold = float(threshold)
        if key == "distance":
            if threshold <= 0:
                raise ValueError(f"arrive distance must be a positive number of blocks, "
                                 f"got {threshold}")
        elif not 0.0 < threshold <= 1.0:
            raise ValueError(f"arrive {key} must be a fraction of the frame in (0, 1], "
                             f"got {threshold}")
        parsed[key] = threshold
    return parsed


def uses_arrive(spec: StopSpec) -> bool:
    """True if `spec` asks for an arrival check -- which needs a controller that tracks."""
    return isinstance(spec, dict) and spec.get("arrive") not in (None, False)


def compile_stop(spec: StopSpec, default_arrive_width: float = 0.6) -> StopFn:
    """Turn a `stop` spec into a predicate returning a reason string, or None to continue."""
    if spec is None:
        return lambda agent: None

    if callable(spec):
        return lambda agent: "predicate" if spec(agent) else None

    if isinstance(spec, (int, np.integer)):
        spec = {"steps": int(spec)}

    if not isinstance(spec, dict):
        raise TypeError(f"stop must be None, an int, a dict or a callable -- got {type(spec)}")

    unknown = set(spec) - STOP_KEYS
    if unknown:
        raise ValueError(f"unknown stop keys {sorted(unknown)}")

    if "arrive" in spec:
        arrive = parse_arrive(spec["arrive"], default_arrive_width)
        rest = {key: value for key, value in spec.items() if key != "arrive"}
        others = compile_stop(rest, default_arrive_width) if rest else (lambda agent: None)
        if arrive is None:
            return others

        def arrived_or_other(agent: "Agent") -> Optional[str]:
            return "arrived" if agent.arrived(arrive) else others(agent)

        return arrived_or_other

    if "steps" in spec:
        budget = int(spec["steps"])
        return lambda agent: "steps" if agent.goal.steps >= budget else None

    if "item" in spec:
        item, count = str(spec["item"]), int(spec.get("count", 1))
        mode = spec.get("mode", "gain")
        if mode not in ("gain", "total"):
            raise ValueError(f"stop mode must be 'gain' or 'total', got {mode!r}")

        def item_reached(agent: "Agent") -> Optional[str]:
            have = count_item(inventory_counts(agent.info), item)
            if mode == "gain":
                have -= count_item(agent.goal.start_inventory, item)
            return "item" if have >= count else None

        return item_reached

    if "stat" in spec:
        stat = str(spec["stat"])
        match, count = str(spec.get("match", "")), int(spec.get("count", 1))

        def stat_reached(agent: "Agent") -> Optional[str]:
            start = agent.goal.start_stats.setdefault(stat, stat_counts(agent.info, stat))
            now = stat_counts(agent.info, stat)
            gained = sum(value - start[key] for key, value in now.items() if match in key)
            return "stat" if gained >= count else None

        return stat_reached

    raise ValueError(f"stop dict needs one of steps/item/stat -- got {sorted(spec)}")


class Agent:
    """Base for the controllers: runs one goal at a time against a MinecraftSim.

    Subclasses supply `_action()` (ask the policy what to press) and usually a `set_goal()`
    of their own that builds the goal object this drives. Everything else -- the step loop,
    the stop conditions, the result -- is here.
    """

    #: Printed in front of this controller's log lines.
    log_tag = "agent"

    #: The Goal subclass `_begin()` instantiates when a subclass does not build one itself.
    goal_class = Goal

    #: The GoalResult subclass `_finish()` produces.
    result_class = GoalResult

    #: Whether this controller tracks its target well enough to answer `arrived()`. A stop
    #: with `arrive` is refused by one that does not, rather than silently never firing.
    supports_arrive = False

    #: What `sim.step()` is given to mean "read the keyboard instead". MineStudio's
    #: PlayCallback branches on a str or None action and passes anything else straight
    #: through to the env, which is how an agent and a person share one sim.
    HUMAN_ACTION = "human"

    def __init__(self, sim, max_steps: int = 600, verbose: bool = True):
        self.sim = sim
        self.max_steps = max_steps
        self.verbose = verbose

        self.goal: Optional[Goal] = None
        self.result: Optional[GoalResult] = None
        self.terminated = False
        self.reward = 0.0
        self.obs: Optional[Dict[str, Any]] = getattr(sim, "obs", None)
        self.info: Dict[str, Any] = getattr(sim, "info", None) or {}
        self._stop: StopFn = compile_stop(None)

    # ------------------------------------------------------------------ observation

    @property
    def frame(self) -> np.ndarray:
        """The current game frame at render resolution (RGB) -- what `point` indexes."""
        self._sync()
        return self.info["pov"]

    def _sync(self) -> None:
        """Pick up obs/info from the sim if we have not stepped it ourselves yet."""
        if not self.info:
            self.obs, self.info = getattr(self.sim, "obs", None), getattr(self.sim, "info", None) or {}
        if not self.info or "pov" not in self.info:
            raise RuntimeError("no observation yet -- call sim.reset() before driving the agent")

    # ------------------------------------------------------------------ the loop

    @property
    def busy(self) -> bool:
        """True while a goal is running and no stop condition has fired."""
        return self.goal is not None and self.result is None

    def step(self) -> Optional[GoalResult]:
        """Advance the environment one frame; returns the result on the step that ends the goal."""
        if not self.busy:
            raise RuntimeError("no goal is running -- call set_goal() first")

        self._sync()
        action = self._action()
        self.obs, reward, terminated, truncated, self.info = self.sim.step(action)
        self.reward = float(reward)
        self.terminated = bool(terminated or truncated)
        self.goal.steps += 1
        self._after_step()

        reason = self._stop_reason()
        if reason is not None:
            self.result = self._finish(reason)
        return self.result

    def idle_step(self) -> bool:
        """Step the world with no goal, and say whether it is still running.

        A sim with nothing to do is usually just closed, which is why this did not exist --
        but a run that *waits* for goals has to keep the world moving while it waits. A
        frozen sim is not a neutral state: `info` stops changing, the frame channel goes
        stale within seconds (see mcagents/frames.py), the voice client stops sending
        pictures, and whatever is choosing the next goal is left planning against a still
        image of a world that has since moved on.

        `_after_step` runs, so previews and the frame publisher carry on exactly as they do
        under a goal -- the only difference is that nothing is pressing any keys.
        """
        if self.busy:
            raise RuntimeError("a goal is running -- call step(), not idle_step()")
        return self._step_without_goal(self.sim.noop_action())

    def human_step(self) -> bool:
        """Step the world with *a person* at the controls, and say whether it is still running.

        The other half of `idle_step()`. Both mean "no goal is running", and both exist for the
        same reason -- a sim that stops stepping stops publishing frames, and whoever is
        choosing the next goal is left planning against a still image. The difference is only
        who presses the keys: `idle_step()` presses nothing, this hands the frame to
        MineStudio's PlayCallback, which reads the keyboard and mouse.

        That makes the hand-off between person and policy a property of what is passed to
        `sim.step()` and nothing else: a dict is the agent driving, `HUMAN_ACTION` is the
        person. No mode flag, no callback to reconfigure, and nothing to get out of sync --
        whoever stepped last was in control.

        `_after_step` runs either way, so the preview, the frame publisher and the gaze
        selector carry on exactly as they do under a goal.
        """
        if self.busy:
            raise RuntimeError("a goal is running -- call step(), not human_step()")
        if not self.plays:
            raise RuntimeError(
                "human_step() needs a sim with a PlayCallback -- it is what reads the "
                "keyboard. Build the sim with MinecraftSim(callbacks=[..., PlayCallback()]), "
                "or use idle_step() to step the world with nothing pressed."
            )
        return self._step_without_goal(self.HUMAN_ACTION)

    @property
    def plays(self) -> bool:
        """Whether this sim has the callback that turns `HUMAN_ACTION` into keypresses.

        Matched through the whole class hierarchy rather than on the exact type: what is
        actually in the sim is a *subclass* -- mcagents.minecraft.play.PlayWindow -- and an
        exact name check quietly answers "no", which shows up only as the keyboard doing
        nothing, with nothing logged to say why. By name rather than by import so the base
        agent stays free of minestudio, the way `_check_sim` already is.
        """
        return any(any(ancestor.__name__ == "PlayCallback"
                       for ancestor in type(callback).__mro__)
                   for callback in getattr(self.sim, "callbacks", []))

    def turn_step(self, yaw: float, pitch: float = 0.0) -> bool:
        """Step the world with the camera turned by [pitch, yaw] degrees and nothing pressed.

        The one action that goes around the policy. An in-place turn has no target, so there
        is nothing to condition a controller on, and what it has to do is known exactly.
        MineRL's convention: positive yaw turns right, positive pitch looks down. On
        MineStudio's env format a yaw of N turns exactly N degrees, measured from 1 to 180 in
        a single step, so nothing here clamps; a caller spreads a turn over steps to be seen
        turning, not to be obeyed. Env-format sims only, since the agent format's camera is
        quantized bins.
        """
        if self.busy:
            raise RuntimeError("a goal is running -- cancel() it before turning")
        if getattr(self.sim, "action_type", "env") != "env":
            raise RuntimeError("turning in place needs an env-format sim; the agent format's "
                               "camera is quantized bins")
        action = dict(self.sim.noop_action())
        action["camera"] = np.array([pitch, yaw], dtype=np.float32)
        return self._step_without_goal(action)

    def _step_without_goal(self, action: Any) -> bool:
        self._sync()
        self.obs, _, terminated, truncated, self.info = self.sim.step(action)
        self.terminated = bool(terminated or truncated)
        self._after_step()
        return not self.terminated

    def drain(self) -> GoalResult:
        """Step until the current goal ends. Blocking; returns why it ended."""
        if self.goal is None:
            raise RuntimeError("no goal set -- call set_goal() first")
        while self.busy:
            self.step()
        return self.result

    def cancel(self, reason: str = "cancelled") -> Optional[GoalResult]:
        """End the running goal now, without stepping. None if nothing was running.

        Nothing is pressed on the way out: the goal simply stops being busy, so whoever
        drives the loop goes back to `idle_step()` -- a no-op action -- on the very next step.
        """
        if not self.busy:
            return None
        self.result = self._finish(reason)
        return self.result

    def arrived(self, thresholds: Dict[str, float]) -> bool:
        """Whether the target is within `thresholds`. Controllers that track override."""
        return False

    def _action(self) -> Any:
        """The action to pass to `sim.step()`, in whatever format this sim expects."""
        raise NotImplementedError

    def _after_step(self) -> None:
        """Hook for subclasses: previews, logging, anything that runs after every step."""

    # ------------------------------------------------------------------ goals

    def _begin(self, goal: Goal) -> Goal:
        """Adopt `goal` as the current one and compile its stop condition.

        The stop condition is compiled *before* anything is adopted, so a goal that is
        rejected leaves the agent exactly as it was. Compiling after assignment reads
        naturally and is wrong: `compile_stop` raises on a bad spec, and by then `goal` is
        set and `result` is not, which is the definition of `busy` -- so the agent is stuck
        running a goal it never accepted, against the *previous* goal's stop function. A
        person typing that spec sees a traceback and retries; a planner sending it over a
        socket wedges the run.
        """
        self._sync()
        stop = self._compile_stop(goal.stop)
        goal.start_inventory = inventory_counts(self.info)
        goal.started_at = time.time()
        self.goal = goal
        self.result = None
        self.reward = 0.0
        self._stop = stop
        self._log(goal.describe())
        return goal

    def _compile_stop(self, spec: StopSpec) -> StopFn:
        if uses_arrive(spec) and not self.supports_arrive:
            raise ValueError(f"{type(self).__name__} does not track its target, so it cannot "
                             f"take an arrive stop")
        return compile_stop(spec)

    def replace_stop(self, spec: StopSpec) -> None:
        """Swap the running goal's stop condition, keeping its step count and baselines.

        What a follow-up to the same goal wants: a new budget measured from when the goal
        *started*, not a restart. Compiled first, so a bad spec leaves the old one running.
        """
        if not self.busy:
            raise RuntimeError("no goal is running")
        stop = self._compile_stop(spec)
        self.goal.stop = spec
        self._stop = stop

    def _stop_reason(self) -> Optional[str]:
        if self.terminated:
            return "terminated"
        if self.goal.steps >= self.max_steps:
            return "max_steps"
        return self._stop(self)

    def _finish(self, reason: str) -> GoalResult:
        result = self.result_class(
            reason=reason,
            steps=self.goal.steps,
            seconds=time.time() - self.goal.started_at,
            gained=dict(inventory_counts(self.info) - self.goal.start_inventory),
            **self._result_extras(),
        )
        self._log(str(result))
        return result

    def _result_extras(self) -> Dict[str, Any]:
        """Extra keyword arguments for `result_class`, for subclasses with richer results."""
        return {}

    # ------------------------------------------------------------------ reporting

    def status(self) -> Dict[str, Any]:
        """A JSON-able snapshot for a supervising planner."""
        return {
            "busy": self.busy,
            "steps": self.goal.steps if self.goal else 0,
            "inventory": dict(inventory_counts(self.info)) if self.info else {},
            "result": str(self.result) if self.result else None,
        }

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[{self.log_tag}] {message}", flush=True)
