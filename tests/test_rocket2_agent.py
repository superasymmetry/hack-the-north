"""Check the Rocket2Agent wiring without booting Minecraft.

A stub sim standing in for MinecraftSim exercises everything that is easy to get wrong: the
stop conditions, the inventory delta, the action-format conversion, the guards on obs_size
and PrevActionCallback, and both the CFG and non-CFG forward paths. It loads the real
ROCKET-2 and SAM-2 weights, so it takes ~30s and needs the GPU, but it needs no game, no
world and no X display.

    conda activate ./.conda-env && python tests/test_rocket2_agent.py
"""
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcagents.agents.rocket2 import Rocket2Agent, Rocket2Config
from mcagents.cli.rocket2 import GoalLoop, PointResolver, is_halt, run_goal, turn_pieces
from mcagents.goals import GoalSpool, StatusChannel

BINARY_KEYS = ["attack", "use", "inventory", "forward", "back", "left", "right", "sneak",
               "sprint", "jump", "drop"] + [f"hotbar.{i}" for i in range(1, 10)]


class StubSim:
    """Mimics MinecraftSim's obs/info contract closely enough to drive the agent.

    A log lands in the inventory every 25 steps, so the item and stat stop conditions have
    something to fire on at a predictable step count.
    """

    obs_size = (224, 224)
    action_type = "env"

    def __init__(self):
        from minestudio.simulator.callbacks import PrevActionCallback

        self.callbacks = [PrevActionCallback()]
        self.steps = 0
        self.logs = 0
        #: Called with the step number after each step -- how a test lands a goal mid-goal.
        self.on_step = None
        #: Whether the trunk is drawn; a test removes it to lose the lock.
        self.trunk = True
        self.actions = []
        self.obs, self.info = self._frame()

    def _frame(self):
        pov = np.zeros((360, 640, 3), np.uint8)
        if self.trunk:
            pov[120:260, 260:400] = (110, 70, 40)      # a "tree trunk" to point at
        info = {
            "pov": pov,
            "inventory": {i: {"type": "none", "quantity": 0} for i in range(36)},
            "mine_block": {"minecraft.mine_block:minecraft.oak_log": np.array(float(self.logs))},
        }
        if self.logs:
            info["inventory"][0] = {"type": "oak_log", "quantity": self.logs}
        obs = {
            "image": np.zeros((224, 224, 3), np.uint8),
            "env_prev_action": {**{k: np.array(0) for k in BINARY_KEYS},
                                "camera": np.zeros(2, np.float32)},
        }
        return obs, info

    def noop_action(self) -> Dict[str, Any]:
        return {**{k: np.array(0) for k in BINARY_KEYS}, "camera": np.zeros(2, np.float32)}

    def agent_action_to_env_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        assert set(action) == {"buttons", "camera"}, action
        return {**{k: np.array(0) for k in BINARY_KEYS}, "camera": np.zeros(2, np.float32)}

    def step(self, action):
        assert "camera" in action and "buttons" not in action, "expected an env-format action"
        self.actions.append(action)
        self.steps += 1
        if self.steps % 25 == 0:
            self.logs += 1
        self.obs, self.info = self._frame()
        if self.on_step is not None:
            self.on_step(self.steps)
        return self.obs, 0.0, False, False, self.info


def check_stop_conditions(agent: Rocket2Agent) -> None:
    # The LLM-ish loose name "log" against the game's "oak_log".
    result = agent.run(point=[330, 190], interaction="Mine", stop={"item": "log", "count": 2})
    assert result.reason == "item" and result.steps == 50, result
    assert result.gained == {"oak_log": 2}, result.gained
    assert result.success

    result = agent.run(point=[0.5, 0.5], interaction="Approach", stop={"steps": 10}, normalized=True)
    assert result.reason == "steps" and result.steps == 10, result

    result = agent.run(point=[330, 190], interaction="Mine",
                       stop={"stat": "mine_block", "match": "log", "count": 1})
    assert result.reason == "stat", result

    result = agent.run(point=[330, 190], interaction="Hunt", stop={"item": "diamond", "count": 1})
    assert result.reason == "max_steps" and result.steps == 200 and not result.success, result


def check_streaming_api(agent: Rocket2Agent) -> None:
    agent.set_goal(point=[330, 190], interaction="Use", stop=lambda a: a.goal.steps == 7)
    while agent.busy:
        agent.step()
    assert agent.result.reason == "predicate" and agent.result.steps == 7, agent.result

    status = agent.status()
    assert status["busy"] is False and status["interaction"] == "Use", status
    assert 0.0 <= status["visibility"] <= 1.0, status
    assert status["predicted_point"] is not None


def check_guardrails(agent: Rocket2Agent) -> None:
    bad_goals = [
        (dict(point=[9999, 10], interaction="Mine"), "outside"),
        (dict(point=[330, 190], interaction="Nope"), "unknown interaction"),
        (dict(point=[330, 190], interaction="Mine", stop={"nope": 1}), "unknown stop keys"),
    ]
    for goal, expected in bad_goals:
        try:
            agent.run(**goal)
        except (ValueError, TypeError) as error:
            assert expected in str(error), (expected, error)
        else:
            raise AssertionError(f"expected a failure for {goal}")

    class WrongObsSize(StubSim):
        obs_size = (128, 128)

    class NoPrevAction(StubSim):
        def __init__(self):
            super().__init__()
            self.callbacks = []

    # A rejected goal must leave the agent usable. Compiling the stop condition after
    # adopting the goal left `busy` true forever against the previous goal's stop function
    # -- survivable when a person retypes it, fatal when a planner sends it down a socket.
    assert not agent.busy, "a rejected goal left the agent stuck running it"
    agent.run(point=[330, 190], interaction="Mine", stop=3)
    assert agent.result.steps == 3, agent.result

    for sim_class, expected in [(WrongObsSize, "224x224"), (NoPrevAction, "PrevActionCallback")]:
        try:
            Rocket2Agent(sim_class())
        except ValueError as error:
            assert expected in str(error), (expected, error)
        else:
            raise AssertionError(f"expected {sim_class.__name__} to be rejected")


def check_idle_stepping(agent: Rocket2Agent) -> None:
    """--listen keeps the world running between goals. A frozen sim stops publishing frames
    within seconds, and the planner is then choosing from a picture of a stopped world."""
    before = agent.sim.steps
    assert agent.idle_step() is True
    assert agent.sim.steps == before + 1, "idle_step did not step the world"
    assert agent.goal is None or not agent.busy

    agent.set_goal(point=[330, 190], interaction="Mine", stop=5)
    try:
        agent.idle_step()
    except RuntimeError as error:
        assert "call step()" in str(error), error
    else:
        raise AssertionError("idle_step ran while a goal was busy")
    agent.drain()


def check_goals_off_the_wire(agent: Rocket2Agent) -> None:
    """A goal that arrived as JSON has to behave exactly like one out of a plan file --
    including the keys ROCKET-2's set_goal has never heard of."""
    resolver = PointResolver()
    with tempfile.TemporaryDirectory() as directory:
        spool = GoalSpool(directory)
        spool.submit({"kind": "goal", "point": [330, 190], "interaction": "Mine",
                      "stop": 5, "rationale": "the model explained itself"})
        entry = spool.take()
        assert run_goal(agent, resolver, entry, "test") is True
        assert agent.result.steps == 5, agent.result

    # No point, only a sentence: the words go to the targeter rather than being refused.
    # A recording stub stands in for it, because the real one is ~600 MB on a card already
    # holding ROCKET-2 and SAM-2 -- what is being checked here is the routing.
    class Recorder:
        asked = []

        def resolve(self, agent, spec):
            self.asked.append(spec)
            raise LookupError("nothing in view matches that")

    recorder = Recorder()
    assert run_goal(agent, recorder, {"instruction": "Chop the oak log."}, "test") is True
    assert recorder.asked == ["Chop the oak log."], recorder.asked

    # A goal that names nothing at all is skipped without ever reaching the targeter.
    assert run_goal(agent, recorder, {"interaction": "Mine", "stop": 5}, "test") is True
    assert len(recorder.asked) == 1, "a goal with no target still went to the targeter"


def check_arrival_and_lock(agent: Rocket2Agent) -> None:
    """Approach tracks the pointed instance, ends when it is close, and ends when it is gone."""
    # The trunk is 140 of 640 px wide: 0.22 of the frame. Approach's default arrival (0.6)
    # does not fire, so a step budget still ends it -- and the lock holds on a still frame.
    result = agent.run(point=[330, 190], interaction="Approach", stop={"steps": 12})
    assert result.reason == "steps" and result.steps == 12, result
    assert agent.goal.stop == {"steps": 12, "arrive": {"width": 0.6}}, agent.goal.stop
    assert agent.goal.lock is not None and agent.goal.lock.updates >= 3, "lock never confirmed"
    box = agent.goal_box()
    assert abs((box[2] - box[0]) - 140 / 640) < 0.02, box

    # Already as wide as asked: arrived before a single action is taken.
    before = agent.sim.steps
    result = agent.run(point=[330, 190], interaction="Approach", stop={"arrive": {"width": 0.2}})
    assert result.reason == "arrived" and result.steps == 0, result
    assert agent.sim.steps == before, "an arrived goal still pressed something"

    # "arrive": false opts out; item stops are left alone.
    agent.set_goal(point=[330, 190], interaction="Approach", stop={"steps": 2, "arrive": False})
    assert agent.goal.stop == {"steps": 2, "arrive": False}
    agent.drain()
    agent.set_goal(point=[330, 190], interaction="Mine", stop={"item": "log", "count": 1})
    assert agent.goal.lock is None and agent.goal.stop == {"item": "log", "count": 1}
    agent.drain()

    # The target vanishes: the lock goes unconfirmed, and past the grace the goal is lost --
    # it does not wander off after whatever is left in frame.
    base = agent.sim.steps

    def vanish(step):
        if step - base == 5:
            agent.sim.trunk = False
    agent.sim.on_step = vanish
    try:
        result = agent.run(point=[330, 190], interaction="Approach", stop={"steps": 150})
    finally:
        agent.sim.on_step, agent.sim.trunk = None, True
        agent.idle_step()                          # re-render: the last frame had no trunk
    grace = agent.config.lock_grace
    # The last confirmed update is at or before step 5; the goal ends grace steps after it.
    assert result.reason == "lost" and grace < result.steps <= 5 + grace + 1, result
    assert not result.success

    for bad, expected in [({"arrive": {"width": 1.5}}, "fraction"),
                          ({"arrive": {"depth": 3}}, "unknown arrive keys")]:
        try:
            agent.set_goal(point=[330, 190], interaction="Approach", stop=bad)
        except ValueError as error:
            assert expected in str(error), (expected, error)
        else:
            raise AssertionError(f"accepted {bad}")
    assert not agent.busy


def check_goals_arriving_mid_goal(agent: Rocket2Agent) -> None:
    """The spool is read before every step: a cancel halts, a keep_memory resend merges, and
    anything else replaces the goal in progress."""
    with tempfile.TemporaryDirectory() as goals, tempfile.TemporaryDirectory() as out:
        spool, status = GoalSpool(goals), StatusChannel(out)
        loop = GoalLoop(agent, PointResolver(), spool, status)

        def at(step, entry):
            base = agent.sim.steps                 # the sim counts every step it has taken

            def hook(n):
                if n - base == step:
                    spool.submit(entry)
            return hook

        def statuses():
            found = []
            while (message := status.spool.take()) is not None:
                found.append((message["event"], message["reason"]))
            return found

        # A cancel lands after step 7; step 8 never happens, and the next action is a no-op.
        agent.sim.on_step = at(7, {"cancel": True})
        assert loop.run({"point": [330, 190], "interaction": "Approach", "stop": 300}, "t")
        assert agent.result.reason == "cancelled" and agent.result.steps == 7, agent.result
        agent.sim.on_step = None
        agent.idle_step()
        noop = agent.sim.noop_action()
        last = agent.sim.actions[-1]
        assert all(np.array_equal(last[key], noop[key]) for key in noop), last
        assert statuses() == [("started", None), ("ended", "cancelled")]

        # The server's own stop payload means the same, and never reaches the targeter.
        server_stop = {"instruction": "stand still", "interaction": "None", "stop": {"steps": 1}}
        assert is_halt(server_stop) and not is_halt({"point": [3, 4], "interaction": "None"})
        agent.sim.on_step = at(3, server_stop)
        loop.run({"point": [330, 190], "interaction": "Approach", "stop": 300}, "t")
        assert agent.result.reason == "cancelled" and agent.result.steps == 3, agent.result
        agent.sim.on_step = None
        statuses()

        # A resend with keep_memory merges: the same lock object, no memory reset, the step
        # count carries on, and the new budget is measured from the goal's start.
        cleared = []
        agent.clear_memory = lambda original=agent.clear_memory: (cleared.append(1), original())
        agent.sim.on_step = at(6, {"point": [0.52, 0.5], "normalized": True, "keep_memory": True,
                                   "interaction": "Approach", "stop": {"steps": 20}})
        loop.run({"point": [330, 190], "interaction": "Approach", "stop": {"steps": 300}}, "t")
        del agent.clear_memory
        assert len(cleared) == 1, f"memory cleared {len(cleared)} times -- the merge restarted it"
        assert agent.result.reason == "steps" and agent.result.steps == 20, agent.result
        assert statuses() == [("started", None), ("updated", "keep_memory"), ("ended", "steps")]

        # ...and a resend that lands just after that goal ended is not allowed to restart it.
        assert loop.run({"point": [0.52, 0.5], "normalized": True, "keep_memory": True,
                         "interaction": "Approach", "stop": {"steps": 20}}, "t")
        assert statuses() == [("rejected", "stale_followup")]

        # A different goal replaces the running one and runs next.
        agent.sim.on_step = at(4, {"point": [330, 190], "interaction": "Mine", "stop": 3})
        loop.run({"point": [330, 190], "interaction": "Approach", "stop": 300}, "t")
        agent.sim.on_step = None
        assert agent.result.reason == "replaced" and agent.result.steps == 4, agent.result
        replacement = loop.next()
        assert replacement["interaction"] == "Mine", replacement
        loop.run(replacement, "t")
        assert agent.result.reason == "steps" and agent.result.steps == 3, agent.result
        assert statuses() == [("started", None), ("ended", "replaced"),
                              ("started", None), ("ended", "steps")]

        # A cancel with nothing running says so, and nothing moves.
        before = agent.sim.steps
        loop.run({"cancel": True}, "t")
        assert agent.sim.steps == before and statuses() == [("rejected", "idle")]


def check_turns_in_place(agent: Rocket2Agent) -> None:
    """`{"turn": deg}` rotates the camera by exactly that and presses nothing else -- no policy,
    no targeter -- preempting whatever ran and then standing still."""
    assert turn_pieces(180, 15) == [15.0] * 12 and turn_pieces(-90, 15) == [-15.0] * 6
    assert len(turn_pieces(100, 15)) == 7 and abs(sum(turn_pieces(100, 15)) - 100) < 1e-9
    assert turn_pieces(90, 0) == [90.0]

    class NoTargeting(PointResolver):
        def resolve(self, agent, spec):
            assert isinstance(spec, (list, tuple)), f"{spec!r} reached the targeter"
            return super().resolve(agent, spec)

    def only_camera(actions, degrees):
        noop = agent.sim.noop_action()
        for action in actions:
            for key in noop:
                if key != "camera":
                    assert np.array_equal(action[key], noop[key]), (key, action[key])
        camera = np.array([np.asarray(a["camera"], np.float64) for a in actions])
        assert abs(camera[:, 1].sum() - degrees) < 1e-3, camera
        assert np.all(camera[:, 0] == 0), "a turn changed pitch"

    with tempfile.TemporaryDirectory() as goals, tempfile.TemporaryDirectory() as out:
        spool, status = GoalSpool(goals), StatusChannel(out)
        loop = GoalLoop(agent, NoTargeting(), spool, status, turn_step=15)

        def statuses():
            found = []
            while (message := status.spool.take()) is not None:
                found.append(message)
            return found

        policy = agent.policy.get_action
        agent.policy.get_action = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("a turn asked the policy"))
        try:
            for degrees, steps in ((90, 6), (-90, 6), (180, 12)):
                before = len(agent.sim.actions)
                assert loop.run({"turn": degrees}, "t") is True
                taken = agent.sim.actions[before:]
                assert len(taken) == steps, (degrees, len(taken))
                only_camera(taken, degrees)
                ended = statuses()
                assert [(m["event"], m["reason"], m["turn"], m["steps"]) for m in ended] == \
                    [("ended", "turned", degrees, steps)], ended
                assert "interaction" not in ended[0], "a turn reported itself as an Approach"
                assert not agent.busy

            # Refused, with nothing stepped and nothing preempted.
            for bad in ({"turn": 0}, {"turn": 270}, {"turn": 90, "interaction": "Approach"}):
                before = agent.sim.steps
                assert loop.run(bad, "t") is True and agent.sim.steps == before, bad
                assert [(m["event"], m["reason"]) for m in statuses()] == [("rejected", "invalid")]
        finally:
            agent.policy.get_action = policy

        # Mid-Approach: the Approach ends as replaced, the turn runs next and clears the
        # policy's memory, and afterwards nothing but no-ops.
        base = agent.sim.steps

        def turn_at_5(n):
            if n - base == 5:
                spool.submit({"turn": 90})
        agent.sim.on_step = turn_at_5
        cleared = []
        agent.clear_memory = lambda original=agent.clear_memory: (cleared.append(1), original())
        try:
            loop.run({"point": [330, 190], "interaction": "Approach", "stop": 300}, "t")
            assert agent.result.reason == "replaced" and agent.result.steps == 5, agent.result
            agent.sim.on_step = None
            before = len(agent.sim.actions)
            cleared.clear()
            assert loop.run(loop.next(), "t")
            only_camera(agent.sim.actions[before:], 90)
            assert cleared, "the turn kept the preempted goal's memory"
        finally:
            del agent.clear_memory
            agent.sim.on_step = None
        assert loop.next() is None, "the preempted goal came back"
        agent.idle_step()
        noop, last = agent.sim.noop_action(), agent.sim.actions[-1]
        assert all(np.array_equal(last[key], noop[key]) for key in noop), last
        assert [(m["event"], m["reason"]) for m in statuses()] == \
            [("started", None), ("ended", "replaced"), ("ended", "turned")]

        # An invalid turn arriving mid-goal does not cost the goal in progress.
        base = agent.sim.steps
        agent.sim.on_step = lambda n: spool.submit({"turn": 0}) if n - base == 2 else None
        try:
            loop.run({"point": [330, 190], "interaction": "Mine", "stop": 6}, "t")
        finally:
            agent.sim.on_step = None
        assert agent.result.reason == "steps" and agent.result.steps == 6, agent.result
        assert [(m["event"], m["reason"]) for m in statuses()] == \
            [("started", None), ("rejected", "invalid"), ("ended", "steps")]


def check_cfg_path(masker) -> None:
    """Classifier-free guidance takes a different forward path; it must still produce actions."""
    agent = Rocket2Agent(StubSim(), Rocket2Config(cfg_coef=1.0, max_steps=20),
                         masker=masker, verbose=False)
    result = agent.run(point=[330, 190], interaction="Mine", stop=5)
    assert result.reason == "steps" and result.steps == 5, result


def main() -> int:
    agent = Rocket2Agent(StubSim(), Rocket2Config(cfg_coef=0.0, max_steps=200, preview=False))
    check_stop_conditions(agent)
    check_streaming_api(agent)
    check_guardrails(agent)
    check_idle_stepping(agent)
    check_goals_off_the_wire(agent)
    check_arrival_and_lock(agent)
    check_goals_arriving_mid_goal(agent)
    check_turns_in_place(agent)
    check_cfg_path(agent.masker)
    print("\nALL API TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
