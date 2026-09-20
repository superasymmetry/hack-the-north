"""Offline tests for arrival tracking and halting -- no vLLM, no sockets.

    ./.venv/bin/python test_arrival.py          (also runs under pytest)
"""

import base64
import io

from interaction_model import (ARRIVAL_CHECK, GOAL_STOPPED, GoalMonitor, MonitorConfig,
                               Timeline, Tracker, TrackerConfig, approach_step_cap,
                               frame_motion, frame_thumb, parse_bbox, parse_bboxes,
                               plan_entry)

# The red-building box at goal time in session.jsonl line 24057.
RED = [0.2589, 0.0, 0.6830, 0.625]


def test_parse_bboxes_keeps_every_match():
    two = '[{"bbox_2d": [0, 0, 10, 10]}, {"bbox_2d": [20, 20, 120, 200]}]'
    assert parse_bboxes(two, 224, 224) == [[0, 0, 10 / 224, 10 / 224],
                                           [20 / 224, 20 / 224, 120 / 224, 200 / 224]]
    assert parse_bbox(two, 224, 224) == [20 / 224, 20 / 224, 120 / 224, 200 / 224]
    assert parse_bboxes("There are none.", 224, 224) == []


def test_step_cap_and_plan_entry():
    assert approach_step_cap(None, 0, 100, 1200) is None               # disabled
    assert approach_step_cap(None, 200, 100, 1200) == 1200             # no box: ceiling
    assert approach_step_cap(RED, 200, 100, 1200) == round(200 / (RED[2] - RED[0]))
    assert approach_step_cap([0, 0, 1, 1], 50, 100, 1200) == 100       # floor

    d = {"target": "the red building", "interaction": "Approach"}
    assert plan_entry(d, [0.47, 0.31], steps=473)["stop"] == {"steps": 473}
    mine = {"target": "oak", "interaction": "Mine", "item": "oak_log", "count": 3}
    assert plan_entry(mine, [0.5, 0.5], steps=99)["stop"] == {"item": "oak_log", "count": 3}
    assert plan_entry({"target": "cow", "interaction": "Hunt"}, steps=99)["stop"] is None


def grow(box, dx):
    return [max(0.0, box[0] - dx), box[1], min(1.0, box[2] + dx), box[3]]


def test_tracker_does_not_arrive_on_the_first_moving_frame():
    """The old gate stopped here (p=0.679). Width 0.42 is not arrived."""
    tr = Tracker(RED, TrackerConfig())
    assert tr.update([grow(RED, 0.01)])["state"] == "tracking"


def test_tracker_arrives_after_sustained_growth_and_predicts_one_frame_early():
    tr = Tracker(RED, TrackerConfig(arrive_width=0.65, arrive_hits=2))
    b = RED
    states = []
    for _ in range(6):
        b = grow(b, 0.04)
        states.append(tr.update([b])["state"])
    assert "arrived" in states
    first = states.index("arrived")
    # Reached 2 frames after the first "close" frame, before width passes 0.65 twice.
    assert first <= 3, states


def test_tracker_follows_the_same_instance_not_the_biggest():
    tr = Tracker(RED, TrackerConfig())
    other = [0.7, 0.0, 1.0, 0.9]                 # a bigger red building, off to the right
    info = tr.update([other, grow(RED, 0.02)])
    assert info["match"] == "iou" and tr.box == grow(RED, 0.02)


def test_tracker_loses_lock_on_jump_or_shrink():
    tr = Tracker(RED, TrackerConfig(lock_misses=2))
    far = [0.85, 0.3, 0.95, 0.4]
    assert tr.update([far])["match"] == "jumped"
    assert tr.update([])["state"] == "lost"

    tr = Tracker(RED, TrackerConfig(lock_misses=1))
    small = [0.40, 0.2, 0.50, 0.3]               # near the center but a fraction of the area
    info = tr.update([small])
    assert info["match"] == "shrank" and info["state"] == "lost"


def jpeg(shade: int) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (shade, shade, shade)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_frame_motion():
    a, b = frame_thumb(jpeg(10)), frame_thumb(jpeg(200))
    assert frame_motion(a, a) == 0
    assert frame_motion(a, b) > 100
    assert frame_motion(None, a) is None


class FakeGoals:
    def __init__(self, deliver=1):
        self.deliver, self.stops, self.sent = deliver, 0, []

    def stop(self):
        self.stops += 1
        return self.deliver

    def send(self, entry):
        self.sent.append(entry)
        return 1


class FakeModel:
    def __init__(self, boxes_seq, p=0.9):
        self.boxes_seq, self.p = list(boxes_seq), p

    def locate_all(self, frame, target):
        return self.boxes_seq.pop(0) if self.boxes_seq else []

    def gate(self, messages, question):
        assert len(messages) == 1, "arrival gates must be image-only"
        return self.p


ENTRY = {"point": [0.47, 0.31], "normalized": True, "instruction": "the red building",
         "interaction": "Approach", "stop": {"steps": 473}}


def events(tl, kind):
    return [e.meta for e in tl.snapshot() if e.kind == kind]


def test_monitor_arrives_stops_and_confirms_halt():
    tl, goals = Timeline(), FakeGoals()
    b1, b2 = grow(RED, 0.12), grow(RED, 0.2)
    mon = GoalMonitor(goals, FakeModel([[b1], [b2]]), tl, MonitorConfig())
    mon.started("go to the red building", ENTRY, "we're up close", RED)

    assert mon.on_frame("f1", 30.0)["state"] == "tracking"
    assert mon.on_frame("f2", 30.0)["state"] == "arrived"
    assert goals.stops == 1 and mon.phase == "stopping"

    mon.on_frame("f3", 25.0)                     # straddles the stop: not judged
    assert goals.stops == 1 and mon.phase == "stopping"
    mon.on_frame("f4", 1.0)                      # still
    assert mon.phase == "idle"
    assert [m["event"] for m in events(tl, GOAL_STOPPED)] == ["sent", "confirmed"]
    assert len(events(tl, ARRIVAL_CHECK)) == 2


def test_monitor_resends_when_still_moving_then_gives_up():
    tl, goals = Timeline(), FakeGoals()
    mon = GoalMonitor(goals, FakeModel([]), tl, MonitorConfig(stop_retries=1))
    mon.started("go", ENTRY, "", RED)
    mon.stop("superseded")
    for f in ("a", "b", "c", "d"):
        mon.on_frame(f, 40.0)
    assert goals.stops == 2 and mon.phase == "idle"
    assert [m["event"] for m in events(tl, GOAL_STOPPED)] == ["sent", "resent", "failed"]


def test_monitor_retries_undelivered_stop_immediately():
    tl, goals = Timeline(), FakeGoals(deliver=0)
    mon = GoalMonitor(goals, FakeModel([]), tl, MonitorConfig(stop_retries=2))
    mon.started("go", ENTRY, "", RED)
    mon.stop("arrived")
    mon.on_frame("a", None)
    mon.on_frame("b", None)
    mon.on_frame("c", None)
    assert goals.stops == 3 and mon.phase == "idle"
    assert events(tl, GOAL_STOPPED)[-1]["event"] == "failed"


def test_veto_holds_the_stop():
    tl, goals = Timeline(), FakeGoals()
    big = [0.0, 0.0, 1.0, 0.9]
    mon = GoalMonitor(goals, FakeModel([[big], [big], [big]], p=0.2), tl, MonitorConfig())
    mon.started("go", ENTRY, "up close", big)
    mon.on_frame("a", 20.0)
    assert mon.on_frame("b", 20.0)["state"] == "vetoed"
    assert goals.stops == 0 and mon.running


def test_arrival_carries_the_done_line():
    tl, goals = Timeline(), FakeGoals()
    big = [0.0, 0.0, 1.0, 0.9]
    mon = GoalMonitor(goals, FakeModel([[big], [big]]), tl, MonitorConfig())
    mon.started("go", ENTRY, "up close", big, "I'm here at the red building.")
    assert "say" not in mon.on_frame("a", 20.0)                  # tracking: nothing to say
    assert mon.on_frame("b", 20.0)["say"] == "I'm here at the red building."

    mon = GoalMonitor(goals, FakeModel([[big], [big]]), tl, MonitorConfig())
    mon.started("go", ENTRY, "up close", big)                     # planner gave no line
    mon.on_frame("a", 20.0)
    assert mon.on_frame("b", 20.0)["say"] == "I'm here at the red building."

    mine = dict(ENTRY, interaction="Mine", instruction="the glass pane", stop=None)
    mon = GoalMonitor(goals, FakeModel([], p=0.9), tl, MonitorConfig())
    mon.started("mine it", mine, "the pane is broken", None)
    mon.on_frame("a", 5.0)
    assert mon.on_frame("b", 5.0)["say"] == "Done with the glass pane."


def test_non_approach_uses_image_only_gate_with_hits():
    tl, goals = Timeline(), FakeGoals()
    mine = dict(ENTRY, interaction="Mine", stop=None)
    mon = GoalMonitor(goals, FakeModel([], p=0.7), tl, MonitorConfig())
    mon.started("mine the pane", mine, "the pane is broken", None)
    assert mon.run.tracker is None
    assert mon.on_frame("a", 5.0)["state"] == "checking"
    assert mon.on_frame("b", 5.0)["state"] == "arrived"
    assert goals.stops == 1



def test_turn_instructions_become_yaw_goals():
    from interaction_model import turn_yaw, validate_entry
    turn = lambda t: plan_entry({"target": t, "interaction": "Turn"})
    assert turn("right") == {"turn": 90}
    assert turn("left") == {"turn": -90}
    assert turn("around") == {"turn": 180}
    assert turn_yaw("what's behind you, on the left") == 180
    assert turn("the oak tree") is None
    assert validate_entry({"turn": 0}) is None
    assert validate_entry({"turn": 270}) is None
    assert validate_entry({"turn": 90, "interaction": "Approach"}) is None


def test_turn_replaces_the_run_without_a_stop():
    tl, goals = Timeline(), FakeGoals()
    mon = GoalMonitor(goals, FakeModel([]), tl, MonitorConfig())
    mon.started("go to the red building", ENTRY, "", grow(RED, 0.12))
    mon.started("turn right", {"turn": 90}, "", None)
    assert mon.run is None and not mon.running and goals.stops == 0
    assert events(tl, GOAL_STOPPED)[-1]["event"] == "replaced"


def test_relational_targets_are_caught():
    from interaction_model import plan_steps, relational_target
    assert relational_target("the other side of the red brick building")
    assert relational_target("behind the oak tree")
    assert relational_target("the front of the house")
    assert not relational_target("the right corner of the brick building")
    assert not relational_target("the far corner of the brick building")
    assert not relational_target("the nearest cow")
    d = {"steps": [{"target": "the oak tree", "interaction": "Mine"},
                   {"target": "", "interaction": "Mine"},
                   {"target": "x", "interaction": "Dance"}, "junk",
                   {"target": "the corner of the building", "interaction": "Turn"}]}
    assert plan_steps(d) == [{"target": "the oak tree", "interaction": "Mine"}]
    assert plan_steps(None) == [] and plan_steps({"steps": "no"}) == []


class FakeChat:
    """Stands in for the OpenAI client: returns queued replies, records prompts."""
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []
        self.chat = self.completions = self

    def create(self, **kw):
        import json as _json
        from types import SimpleNamespace as NS
        self.prompts.append(kw["messages"])
        reply = self.replies.pop(0)
        text = reply if isinstance(reply, str) else _json.dumps(reply)
        return NS(choices=[NS(message=NS(content=text))])


def _plan_reply(*steps, **extra):
    return dict({"reasoning": "r", "say": "On it.", "done_when": "there",
                 "done_say": "Here.", "steps": [{"target": t, "interaction": i}
                                                 for i, t in steps]}, **extra)


def test_plan_retries_a_relational_target_once():
    from interaction_model import PLAN_FEEDBACK, InteractionModel
    chat = FakeChat([_plan_reply(("Approach", "the other side of the building")),
                     _plan_reply(("Approach", "the right corner of the building"),
                                 ("Approach", "the far corner of the building"))])
    plan = InteractionModel(chat, "m", 50).plan([{"role": "user", "content": "go"}])
    assert len(chat.prompts) == 2
    assert chat.prompts[1][-1]["content"].startswith(PLAN_FEEDBACK[:12])
    assert plan.entry["point"] == "the right corner of the building"
    assert plan.pending == [{"target": "the far corner of the building",
                             "interaction": "Approach"}]
    assert plan.say == "" and plan.reasoning == "r"      # the reply is acknowledge's now


def test_plan_gives_up_instead_of_sending_a_relation():
    from interaction_model import CANT_PLAN_SAY, InteractionModel
    bad = _plan_reply(("Approach", "behind the building"))
    plan = InteractionModel(FakeChat([bad, bad]), "m", 50).plan([])
    assert plan.entry is None and plan.say == CANT_PLAN_SAY


def test_plan_count_only_on_a_single_step():
    from interaction_model import InteractionModel
    one = _plan_reply(("Mine", "the oak tree"), item="oak_log", count=3)
    assert InteractionModel(FakeChat([one]), "m", 50).plan([]).entry["stop"] == \
        {"item": "oak_log", "count": 3}
    two = _plan_reply(("Approach", "the oak tree"), ("Mine", "the oak tree"),
                      item="oak_log", count=3)
    plan = InteractionModel(FakeChat([two]), "m", 50).plan([])
    assert plan.entry["stop"] is None and len(plan.pending) == 1


def test_plan_sends_no_target_when_they_are_pointing():
    """The pinched object stays the target only if we name no other one."""
    from interaction_model import InteractionModel, SELECTION_STEPS
    reply = _plan_reply(("Mine", "the oak tree"))
    sel = {"point": [0.51, 0.42], "box": [0.42, 0.31, 0.60, 0.78]}
    chat = FakeChat([reply])
    plan = InteractionModel(chat, "m", 50).plan([], frame="ignored", selection=sel)
    assert plan.entry == {"interaction": "Mine", "stop": {"steps": SELECTION_STEPS},
                          "use_selection": True}          # stripped by GoalLink.send
    assert plan.target == "the oak tree"          # ours: the log, the tracker, the line
    assert plan.bbox == sel["box"]                # their box seeds the tracker
    assert len(chat.prompts) == 1                 # and `locate` was never asked


def test_still_approach_stalls_after_n_frames():
    tl, goals = Timeline(), FakeGoals()
    desc = {"point": "the other side of the red brick building", "interaction": "Approach",
            "stop": {"steps": 1200}}
    box = [0.0, 0.3036, 0.5045, 1.0]                     # session.jsonl line 94714
    mon = GoalMonitor(goals, FakeModel([[box]] * 5), tl, MonitorConfig(stall_frames=3))
    mon.started("go to the other side", desc, "", None)
    assert mon.on_frame("a", 0.29)["state"] == "tracking"
    assert mon.on_frame("b", 0.25)["state"] == "tracking"
    check = mon.on_frame("c", 0.26)
    assert check["state"] == "stalled" and goals.stops == 1 and mon.phase == "stopping"

    mon = GoalMonitor(goals, FakeModel([[box]] * 5), tl, MonitorConfig(stall_frames=3))
    mon.started("go", desc, "", None)
    for f, m in (("a", 0.2), ("b", 0.2), ("c", 30.0), ("d", 0.2)):   # motion resets it
        assert mon.on_frame(f, m)["state"] == "tracking"


def test_intermediate_step_reports_stepped_not_arrived():
    tl, goals = Timeline(), FakeGoals()
    big = [0.0, 0.0, 1.0, 0.9]
    rest = [{"target": "the far corner", "interaction": "Approach"}]
    # An intermediate step's veto asks about its own target, not done_when.
    model = FakeModel([[big], [big]])
    asked = []
    model.gate = lambda msgs, q: asked.append(q) or 0.9
    mon = GoalMonitor(goals, model, tl, MonitorConfig())
    mon.started("go to the other side", ENTRY, "we are past the building", big, "", rest)
    mon.on_frame("a", 20.0)
    check = mon.on_frame("b", 20.0)
    assert check["state"] == "stepped" and check["pending"] == rest
    assert "past the building" not in asked[-1] and goals.stops == 1


def test_turn_with_more_steps_waits_for_the_screen_to_settle():
    tl, goals = Timeline(), FakeGoals()
    rest = [{"target": "the cow", "interaction": "Approach"}]
    mon = GoalMonitor(goals, FakeModel([]), tl, MonitorConfig())
    mon.started("go behind us", {"turn": 180}, "", None, "", rest)
    assert mon.running
    assert mon.on_frame("a", 1.0) is None                # straddles the send
    assert mon.on_frame("b", 40.0) is None               # still turning
    check = mon.on_frame("c", 1.0)
    assert check["state"] == "stepped" and check["turn"] == 180
    assert mon.run is None and goals.stops == 0


def test_step_notes():
    from interaction_model import step_note
    assert step_note({"state": "stepped", "turn": -90}) == "turned left"
    assert step_note({"state": "stepped", "interaction": "Approach",
                      "target": "the corner"}) == "reached the corner"
    assert step_note({"state": "stalled", "interaction": "Approach",
                      "target": "x"}) == "tried to approach x but stopped moving"


def test_sentence_stream_sends_each_sentence_as_it_closes():
    from interaction_model import SentenceStream
    out = []
    s = SentenceStream(out.append)
    for tok in ["Okay", ".", " Heading to", " the 1.5", " block. ", "Almost"]:
        s(tok)
    assert out == ["Okay.", "Heading to the 1.5 block."]    # "1.5" is not a sentence end
    s.flush()
    assert out[-1] == "Almost"


def test_talk_streams_sentences_from_the_talk_server():
    from types import SimpleNamespace as NS
    from interaction_model import InteractionModel
    chunks = ["On it", ", heading", " there. ", "Almost."]

    class Stream:
        def __iter__(self):
            return iter(NS(choices=[NS(delta=NS(content=c))]) for c in chunks)

        def close(self):
            pass

    calls, heard = [], []
    talk = NS(chat=NS(completions=NS(create=lambda **kw: calls.append(kw) or Stream())))
    model = InteractionModel(object(), "big", 50, talk_client=talk, talk_model="fp8")
    result = model.acknowledge([{"role": "user", "content": "go"}], heard.append)
    assert heard == ["On it, heading there.", "Almost."]
    assert result["said"] == "On it, heading there. Almost."
    assert calls[0]["model"] == "fp8" and result["first_s"] is not None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
