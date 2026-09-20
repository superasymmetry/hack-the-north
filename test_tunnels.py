"""Replay + live-socket tests for the two tunnels.

    ./.venv/bin/python test_tunnels.py          (also runs under pytest)

The replay half feeds `uplink_transcript.jsonl` -- a recorded capture session of
partials, finals, frames and wire junk -- through the handler and asserts it
never raises. That is the regression that matters: before frames existed every
handler reached straight for msg["text"], and one frame every three seconds
would have taken the loop down within a minute of the client shipping.
"""

import base64
import json
import threading
import time
from pathlib import Path

from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed

from interaction_model import (SELECTION_STEPS, GoalLink, Uplink, parse_bbox, plan_entry,
                               selection_of, smart_resize, validate_entry)

TRANSCRIPT = Path(__file__).parent / "uplink_transcript.jsonl"
TOKEN = "test-token"


def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --------------------------------------------------------------------------- #
# Replay: the recorded transcript through the handler
# --------------------------------------------------------------------------- #

def test_recorded_transcript_replays_without_raising():
    up = Uplink()
    for line in TRANSCRIPT.read_text().splitlines():
        if line.strip():
            up.on_message(line)                     # must never raise

    assert up.finals() == [
        "mine the oak tree",
        "how much wood do we have",
        "now go find a cow",
        "just plain prose, not json",
        "{not json at all",
    ]
    s = up.stats
    assert s["partials"] == 3                       # accepted and dropped
    assert s["dupes"] == 1                          # the at-least-once redelivery
    assert s["frames"] == 7
    assert s["ignored"] == 4                        # empty text, imageless frame,
                                                    # telemetry, a bare JSON list

    # Newest frame only, and it is decodable JPEG the model can be handed.
    frame = up.frame()
    assert frame is not None
    raw = base64.b64decode(frame)
    assert raw[:3] == b"\xff\xd8\xff"
    rows = [json.loads(l) for l in TRANSCRIPT.read_text().splitlines()
            if l.startswith("{") and '"frame"' in l and "image" in l]
    assert frame == rows[-1]["image"]


def test_frames_never_touch_the_text_path():
    """The failure this whole branch exists to prevent."""
    up = Uplink()
    for _ in range(50):
        up.on_message(json.dumps({"kind": "frame", "image": "Zm9v"}))
    assert up.finals() == []
    assert up.frame() == "Zm9v"


def test_dedupe_expires():
    now = [0.0]
    up = Uplink(dedupe_window=10.0, clock=lambda: now[0])
    up.on_message('{"text": "mine the oak tree", "final": true}')
    up.on_message('{"text": "mine the oak tree", "final": true}')
    assert up.finals() == ["mine the oak tree"]
    now[0] = 11.0
    up.on_message('{"text": "mine the oak tree", "final": true}')
    assert up.finals() == ["mine the oak tree"]      # asking again later is real


def test_stale_frames_are_refusable():
    now = [0.0]
    up = Uplink(clock=lambda: now[0])
    up.on_message({"kind": "frame", "image": "Zm9v"})
    now[0] = 30.0
    assert up.frame(max_age=10.0) is None
    assert up.frame() == "Zm9v"


# --------------------------------------------------------------------------- #
# Goal validation
# --------------------------------------------------------------------------- #

def test_valid_goals_survive():
    assert validate_entry(
        {"kind": "goal", "point": "the oak tree", "interaction": "Mine",
         "stop": {"item": "oak_log", "count": 3}}
    ) == {"point": "the oak tree", "interaction": "Mine",
          "stop": {"item": "oak_log", "count": 3}}

    for stop in (None, 200, {"steps": 200},
                 {"item": "oak_log", "count": 3, "mode": "total"},
                 {"stat": "mine_block", "match": "log", "count": 1}):
        assert validate_entry({"point": "a tree", "stop": stop}) is not None, stop

    assert validate_entry({"point": "a tree"})["interaction"] == "Approach"
    assert validate_entry({"instruction": "go to the water"}) == {
        "instruction": "go to the water", "interaction": "Approach", "stop": None}
    assert validate_entry({"point": [0.5, 0.25], "normalized": True})["point"] == [0.5, 0.25]


def test_the_coordinate_trap_is_unreachable():
    """224-space pixels would run against a 640x360 frame and fail silently."""
    assert validate_entry({"point": [112, 112], "interaction": "Mine"}) is None
    assert validate_entry({"point": [112, 112], "normalized": False}) is None
    assert validate_entry({"point": [1.4, 0.2], "normalized": True}) is None


def test_bad_goals_are_rejected():
    for bad in ({"status": "ok"},                       # not a goal at all
                {},
                "nope",
                {"point": "a tree", "stop": {"item": "oak_log"}},        # no count
                {"point": "a tree", "stop": {"item": "oak_log", "count": 0}},
                {"point": "a tree", "stop": {"until": "dawn"}},
                {"point": "a tree", "stop": {"item": "x", "count": 1, "mode": "some"}},
                {"point": "a tree", "interaction": "Yeet"},
                {"instruction": "   "}):
        assert validate_entry(bad) is None, bad


def test_plan_entry_never_emits_pixels():
    e = plan_entry({"target": "the oak tree", "interaction": "Mine",
                    "item": "oak_log", "count": 3}, point=[0.5, 0.4])
    assert e["normalized"] is True and e["point"] == [0.5, 0.4]
    assert e["instruction"] == "the oak tree"
    assert e["stop"] == {"item": "oak_log", "count": 3}

    # u/v in the model's JSON are no longer a way in -- only `locate` supplies points.
    e = plan_entry({"target": "the nearest cow", "interaction": "Hunt", "u": 0.6, "v": 0.2})
    assert e == {"point": "the nearest cow", "interaction": "Hunt", "stop": None}

    assert plan_entry({"target": "", "interaction": "Mine"}) is None
    assert plan_entry("garbage") is None


# --------------------------------------------------------------------------- #
# Pointing
# --------------------------------------------------------------------------- #

LOCKED = {"point": [0.51, 0.42], "box": [0.42, 0.31, 0.60, 0.78],
          "locked": True, "held": True, "age": 1.4}


def test_only_a_deliberate_pinch_counts_as_pointing():
    assert selection_of({"text": "mine that", "selection": LOCKED}) == {
        "point": [0.51, 0.42], "box": [0.42, 0.31, 0.60, 0.78]}
    for ignored in (dict(LOCKED, locked=False),          # the hand passing over it
                    dict(LOCKED, age=60.0),              # released a minute ago
                    {"locked": True},                    # nothing to point at
                    {"locked": True, "point": [326, 151]},   # pixels, not fractions
                    "over there", None):
        assert selection_of({"selection": ignored}) is None, ignored
    assert selection_of({"text": "mine that"}) is None
    # Still pinching: `age` only ages a selection once they have let go.
    assert selection_of({"selection": dict(LOCKED, held=False, age=60.0)}) is not None


def test_pointing_arrives_on_a_frame_too():
    clock = [0.0]
    up = Uplink(clock=lambda: clock[0])
    up.on_message(json.dumps({"kind": "frame", "image": "abc", "selection": LOCKED}))
    up.on_message(json.dumps({"text": "mine that", "final": True}))
    assert up.finals() == ["mine that"]
    assert up.selection["box"] == [0.42, 0.31, 0.60, 0.78]
    assert up.stats["frames"] == 1               # still just a frame, not an utterance

    # The line main() acts on is the last one, and so is the selection it reads.
    up.on_message(json.dumps({"text": "no, that one", "final": True}))
    up.on_message(json.dumps({"text": "actually, keep going", "final": True,
                              "selection": None}))
    assert up.finals() == ["no, that one", "actually, keep going"]
    assert up.selection is not None              # the frame's pinch is still warm

    clock[0] += 60
    up.on_message(json.dumps({"text": "and now this", "final": True}))
    assert up.finals() == ["and now this"]
    assert up.selection is None                  # the pinch went stale in between


def test_a_selection_goal_names_no_target():
    """THE ONE RULE: a point of either kind would override what they pinched."""
    e = plan_entry({"target": "the oak tree", "interaction": "Mine"}, selection=True)
    assert e == {"interaction": "Mine", "stop": {"steps": SELECTION_STEPS},
                 "use_selection": True}
    assert "point" not in e and "instruction" not in e

    # A point handed in alongside is dropped, not sent -- and validating twice,
    # which every send does, must not turn the goal back into a bad one.
    once = validate_entry({"use_selection": True, "point": [0.5, 0.4], "normalized": True,
                           "instruction": "the oak tree", "interaction": "Mine",
                           "stop": {"item": "oak_log", "count": 3}})
    assert once == {"interaction": "Mine", "stop": {"item": "oak_log", "count": 3},
                    "use_selection": True}
    assert validate_entry(once) == once

    # Without one, nothing changes: the description still goes out to the targeter.
    assert plan_entry({"target": "the nearest tree", "interaction": "Mine"}) == {
        "point": "the nearest tree", "interaction": "Mine", "stop": None}
    # A goal with no target and no selection is still rejected.
    assert validate_entry({"interaction": "Mine", "stop": 300}) is None


def test_goal_status_is_read_not_ignored():
    goals = GoalLink()
    goals.on_message(json.dumps({"kind": "goal.status", "event": "rejected",
                                 "reason": "no selection", "id": "g1"}))
    goals.on_message("{not json")                # must not raise
    goals.on_message(json.dumps({"kind": "frame"}))
    assert goals.last_status["event"] == "rejected"
    assert goals.stats["status.rejected"] == 1


def test_smart_resize_matches_what_the_server_sees():
    assert smart_resize(224, 224) == (224, 224)        # 64 visual tokens, measured live
    h, w = smart_resize(360, 640)                      # over max_pixels: shrunk, x28
    assert h * w <= 147456 and h % 28 == 0 and w % 28 == 0


def test_parse_bbox_reads_the_trained_grounding_reply():
    # Verbatim reply from Qwen2.5-VL-32B on a 224x224 frame.
    reply = '```json\n[\n\t{"bbox_2d": [158, 90, 176, 134], "label": "white building"}\n]\n```'
    assert parse_bbox(reply, 224, 224) == [158 / 224, 90 / 224, 176 / 224, 134 / 224]

    two = '[{"bbox_2d": [0, 0, 10, 10]}, {"bbox_2d": [20, 20, 120, 200]}]'
    assert parse_bbox(two, 224, 224) == [20 / 224, 20 / 224, 120 / 224, 200 / 224]  # largest

    assert parse_bbox('{"bbox_2d": [-5, 10, 300, 50]}', 224, 224) == [0.0, 10 / 224, 1.0, 50 / 224]
    assert parse_bbox("There are none.", 224, 224) is None                  # absent target
    assert parse_bbox('{"bbox_2d": [50, 50, 50, 90]}', 224, 224) is None    # no area
    assert parse_bbox("", 224, 224) is None


# --------------------------------------------------------------------------- #
# Live sockets
# --------------------------------------------------------------------------- #

def test_tunnel_one_round_trip():
    up = Uplink(TOKEN).serve(port := _free_port())
    try:
        with connect(f"ws://127.0.0.1:{port}",
                     additional_headers={"X-Agent-Token": TOKEN}) as ws:
            ws.send(json.dumps({"text": "half a sen", "final": False}))
            ws.send(json.dumps({"kind": "frame", "image": "Zm9v"}))
            ws.send(json.dumps({"text": "mine the oak tree", "final": True}))
            for _ in range(100):
                if up.stats["finals"]:
                    break
                time.sleep(0.01)
            assert up.finals() == ["mine the oak tree"]
            assert up.frame() == "Zm9v"

            assert up.say("heading for the oak on your left") == 1
            assert json.loads(ws.recv(timeout=2)) == {
                "text": "heading for the oak on your left"}

            # Still open, still ours.
            ws.send(json.dumps({"text": "and grab three logs", "final": True}))
            for _ in range(100):
                if (got := up.finals()):
                    break
                time.sleep(0.01)
            assert got == ["and grab three logs"]
    finally:
        up.close()


def test_tunnel_two_is_send_only_and_stays_open():
    goals = GoalLink(TOKEN).serve(port := _free_port())
    try:
        with connect(f"ws://127.0.0.1:{port}",
                     additional_headers={"X-Agent-Token": TOKEN}) as ws:
            for _ in range(100):
                if goals.connected:
                    break
                time.sleep(0.01)

            assert goals.send({"kind": "goal", "point": "the oak tree",
                               "interaction": "Mine",
                               "stop": {"item": "oak_log", "count": 3}}) == 1
            sent = json.loads(ws.recv(timeout=2))
            assert sent.pop("id")                     # every goal is identified
            assert sent == {"point": "the oak tree", "interaction": "Mine",
                            "stop": {"item": "oak_log", "count": 3}}

            assert goals.send([{"point": "the cow", "interaction": "Hunt"},
                               {"point": [0.5, 0.5], "normalized": True}]) == 2
            batch = json.loads(ws.recv(timeout=2))
            assert isinstance(batch, list) and len(batch) == 2

            # A pointing goal: no target of any kind, and no internal keys either.
            assert goals.send(plan_entry({"target": "the oak tree", "interaction": "Mine"},
                                         selection=True)) == 1
            pointed = json.loads(ws.recv(timeout=2))
            assert pointed.pop("id")
            assert pointed == {"interaction": "Mine", "stop": {"steps": SELECTION_STEPS}}

            # Rejected goals reach nobody, and the socket survives them.
            assert goals.send({"status": "ok"}) == 0
            assert goals.send({"point": [112, 112]}) == 0

            # The client sends nothing on this tunnel; if it ever does, we ignore
            # it rather than dying.
            ws.send("ignore me")
            assert goals.send({"point": "the water", "interaction": "Approach"}) == 1
            assert json.loads(ws.recv(timeout=2))["point"] == "the water"
    finally:
        goals.close()


def test_undelivered_goals_are_dropped_not_queued():
    """Replaying a goal onto a reconnect would run it a second time."""
    goals = GoalLink(TOKEN).serve(port := _free_port())
    try:
        assert goals.send({"point": "the oak tree", "interaction": "Mine"}) == 0
        with connect(f"ws://127.0.0.1:{port}",
                     additional_headers={"X-Agent-Token": TOKEN}) as ws:
            for _ in range(100):
                if goals.connected:
                    break
                time.sleep(0.01)
            try:
                ws.recv(timeout=0.5)
                assert False, "a goal was replayed onto a fresh connection"
            except TimeoutError:
                pass
    finally:
        goals.close()


def test_1008_is_only_for_a_bad_token():
    for link in (Uplink(TOKEN), GoalLink(TOKEN)):
        link.serve(port := _free_port())
        try:
            try:
                with connect(f"ws://127.0.0.1:{port}",
                             additional_headers={"X-Agent-Token": "wrong"}) as ws:
                    ws.recv(timeout=2)
                assert False, "a bad token was accepted"
            except ConnectionClosed as e:
                assert e.rcvd.code == 1008

            # A restart is not an auth failure: the client must reconnect.
            with connect(f"ws://127.0.0.1:{port}",
                         additional_headers={"X-Agent-Token": TOKEN}) as ws:
                for _ in range(100):
                    if link.connected:
                        break
                    time.sleep(0.01)
                threading.Thread(target=link.close, daemon=True).start()
                try:
                    ws.recv(timeout=2)
                except ConnectionClosed as e:
                    assert e.rcvd.code != 1008
                    assert e.rcvd.code == 1012
        finally:
            link.close()

    try:
        Uplink(TOKEN).close(1008)
        assert False, "close(1008) should be refused"
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
