"""Check gaze selection without a webcam, a GPU, or Minecraft.

The three pieces that decide whether a red box lands on the right object are the channel
(is this sample fresh enough to believe?), the screen mapping (where on the frame is that
person looking?), and the state machine (hover, commit, hold, let go). None of them needs
SAM-2 to test -- a stub masker that answers from a known scene is a better oracle than the
real one, because a wrong answer is then unambiguously this code's fault.

    python tests/test_gaze_selection.py
"""
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcagents import gaze
from mcagents.perception.selection import (GazeSelector, ScreenMap, Selection, SelectorConfig)

#: The window geometry probed on this laptop at MCAGENTS_GUI_SCALE=4 -- a real rect, so the
#: arithmetic is checked against the thing it will actually be handed.
RECT = (182, 188, 2560, 1350)
FRAME = (360, 640)

#: Three objects at known places in a 640x360 frame: (x0, y0, x1, y1).
SCENE = {
    "trunk": (260, 120, 400, 260),
    "house": (60, 40, 200, 110),
    "water": (0, 300, 640, 360),        # big, low: the "ground" case
}


class SceneMasker:
    """Answers with whichever scene rectangle contains the point, and counts the work."""

    def __init__(self):
        self.embeds = 0
        self.decodes = 0

    def embed(self, frame):
        self.embeds += 1
        return ("embedding", self.embeds)

    def mask_at(self, embedding, point) -> np.ndarray:
        self.decodes += 1
        mask = np.zeros(FRAME, bool)
        for x0, y0, x1, y1 in SCENE.values():
            if x0 <= point[0] < x1 and y0 <= point[1] < y1:
                mask[y0:y1, x0:x1] = True
                break
        else:
            mask[int(point[1]), int(point[0])] = True     # a speck: "sky"
        return mask


class Eyes:
    """A gaze source a test can put wherever it likes, in screen pixels."""

    def __init__(self):
        self.at: Optional[Tuple[float, float]] = None

    def look(self, frame_point) -> "Eyes":
        """Aim at a point given in *frame* pixels, mapped back out to the screen."""
        left, top, width, height = RECT
        self.at = (left + frame_point[0] / FRAME[1] * width,
                   top + frame_point[1] / FRAME[0] * height)
        return self

    def away(self) -> "Eyes":
        self.at = None
        return self

    def sample(self):
        if self.at is None:
            return None
        return gaze.GazeSample(x=self.at[0], y=self.at[1], t=time.time())


def settle(selector: GazeSelector, frame, steps: int = 4):
    """Step the selector until its background encoder has produced something to decode with."""
    for _ in range(steps):
        selector.update(frame)
        for _ in range(200):
            if selector.worker.latest() is not None:
                break
            time.sleep(0.001)
    return selector.selection


def check_channel() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "gaze.json")
        assert gaze.read_latest(path) is None, "a path nobody has written to is not a sample"

        gaze.write(gaze.GazeSample(x=1730.5, y=921.25), path)
        sample = gaze.read_latest(path)
        assert sample is not None and (sample.x, sample.y) == (1730.5, 921.25), sample

        # Stale: a box left sitting where somebody stopped looking reads as a confident
        # mistake, so the bar is one fixation rather than one timeout.
        assert gaze.read_latest(path, max_age=0.0) is None

        # Running but blind -- a blink, a turned head -- is "nothing to point at", the same
        # answer as no tracker at all, because there is nothing different to do about it.
        gaze.write(gaze.GazeSample(x=1.0, y=2.0, valid=False), path)
        assert gaze.read_latest(path) is None

        # Anything unreadable is also just "no sample": this is a well-known path.
        Path(path).write_text("{not json")
        assert gaze.read_latest(path) is None
        Path(path).write_bytes(b"x" * (gaze.MAX_BYTES + 1))
        assert gaze.read_latest(path) is None

        publisher = gaze.GazePublisher(path, report=lambda message: None)
        assert publisher.offer(5.0, 6.0) is True
        assert gaze.read_latest(path).x == 5.0
        broken = gaze.GazePublisher("/proc/nowhere/gaze.json", report=lambda message: None)
        assert broken.offer(1.0, 1.0) is False and broken.enabled is False


def check_screen_map() -> None:
    screen = ScreenMap(lambda: RECT)
    left, top, width, height = RECT

    # Dead centre of the window is dead centre of the frame.
    centre = gaze.GazeSample(x=left + width / 2, y=top + height / 2)
    x, y = screen.to_frame(centre, FRAME)
    assert abs(x - 320) < 0.01 and abs(y - 180) < 0.01, (x, y)

    # The two axes scale independently: 640x360 is stretched into 2560x1350, which is not
    # the same aspect ratio, and assuming one factor would put every point off vertically.
    assert abs(2560 / 640 - 4.0) < 1e-9 and abs(1350 / 360 - 3.75) < 1e-9
    x, y = screen.to_frame(gaze.GazeSample(x=left + 4.0 * 100, y=top + 3.75 * 50), FRAME)
    assert abs(x - 100) < 0.01 and abs(y - 50) < 0.01, (x, y)

    # Outside the view is None, not a clamp: looking at the info panel or another window is
    # "not pointing at anything", which is different from pointing at the frame's edge.
    for off in [(left - 1, top + 10), (left + 10, top - 1),
                (left + width, top + 10), (left + 10, top + height)]:
        assert screen.to_frame(gaze.GazeSample(x=off[0], y=off[1]), FRAME) is None, off

    assert ScreenMap(lambda: None).to_frame(centre, FRAME) is None, "no window, no point"


def check_hover_and_lock() -> None:
    frame = np.zeros((*FRAME, 3), np.uint8)
    eyes, masker = Eyes(), SceneMasker()
    config = SelectorConfig(lock_ms=50, deadzone=0.0, smoothing=1.0)
    selector = GazeSelector(masker, ScreenMap(lambda: RECT), eyes, config,
                            report=lambda message: None)
    try:
        # Nothing to point at until somebody looks.
        eyes.away()
        assert settle(selector, frame) is None

        # Looking at the trunk boxes the trunk -- immediately, hovering, not yet committed.
        eyes.look((330, 190))
        selection = settle(selector, frame)
        assert selection is not None, "looking straight at an object selected nothing"
        assert selection.box == SCENE["trunk"], selection.box
        assert selection.locked is False, "committed before it had been looked at"

        # ...and commits once it has been there for lock_ms.
        time.sleep(config.lock_ms / 1000.0)
        selector.update(frame)
        assert selector.selection.locked is True, "never committed"
        assert selector.selection.box == SCENE["trunk"]

        # The lock is sticky: a look that lands just outside the box, which is what ~100 px
        # of tracker bias looks like, must not drop it.
        x0, y0, x1, y1 = SCENE["trunk"]
        eyes.look((x1 + 10, y1 + 10))
        selector.update(frame)
        assert selector.selection.locked and selector.selection.box == SCENE["trunk"], \
            "the lock let go for a look just outside it"

        # Eyes crossing sky do not drop a committed lock: every webcam tracker drops
        # samples, and a lock that evaporates on the first one is not a lock.
        eyes.look((620, 20))
        selector.update(frame)
        assert selector.selection is not None and selector.selection.locked, \
            "a glance at the sky dropped the lock"

        # ...but looking at nothing for longer than the grace does let it go.
        time.sleep(config.release_ms / 1000.0)
        selector.update(frame)
        assert selector.selection is None, "the lock never let go"

        # Looking properly at something else switches immediately -- no grace, because that
        # is not a dropout, it is a decision -- and the newcomer earns its own lock.
        eyes.look((330, 190))
        selector.update(frame)
        time.sleep(config.lock_ms / 1000.0)
        selector.update(frame)
        assert selector.selection.locked and selector.selection.box == SCENE["trunk"]
        eyes.look((130, 75))
        selector.update(frame)
        assert selector.selection.box == SCENE["house"], selector.selection.box
        assert selector.selection.locked is False, "the new object inherited the old lock"

        # An uncommitted hover needs no grace at all: looking away clears it at once.
        eyes.away()
        selector.update(frame)
        assert selector.selection is None, "the box outlived the gaze"
    finally:
        selector.close()


def check_still_gaze_locks() -> None:
    """The steadiest possible gaze must lock, and it is the case most easily broken.

    With a deadzone suppressing re-decodes, a gaze that never moves produces no further
    decodes at all -- so a state machine that only advances inside a decode leaves the one
    person who is looking perfectly steadily as the one person who never gets a lock.
    """
    frame = np.zeros((*FRAME, 3), np.uint8)
    eyes, masker = Eyes(), SceneMasker()
    config = SelectorConfig(lock_ms=50, deadzone=40.0)      # a wide deadzone, on purpose
    selector = GazeSelector(masker, ScreenMap(lambda: RECT), eyes, config,
                            report=lambda message: None)
    try:
        eyes.look((330, 190))                               # and never move again
        assert settle(selector, frame) is not None
        decodes = masker.decodes
        deadline = time.monotonic() + 2.0
        while selector.selection is not None and not selector.selection.locked:
            assert time.monotonic() < deadline, "a perfectly still gaze never locked"
            selector.update(frame)
            time.sleep(0.01)
        assert selector.selection.box == SCENE["trunk"], selector.selection.box
        # The deadzone still does its job: a motionless gaze re-decodes only when there is a
        # genuinely new frame underneath it, never several times against the same one.
        assert masker.decodes <= masker.embeds + 1, \
            f"{masker.decodes} decodes against {masker.embeds} embeddings"
    finally:
        selector.close()


def check_refusals() -> None:
    """The two ways a point prompt is right and useless: a speck, and the whole world."""
    frame = np.zeros((*FRAME, 3), np.uint8)
    eyes, masker = Eyes(), SceneMasker()
    selector = GazeSelector(masker, ScreenMap(lambda: RECT), eyes,
                            SelectorConfig(lock_ms=50, deadzone=0.0, smoothing=1.0),
                            report=lambda message: None)
    try:
        # Sky: SAM-2 always answers, so without a floor on the area the box spends the
        # session confidently around patches of nothing.
        eyes.look((620, 20))
        assert settle(selector, frame) is None, "a speck of sky was selected"

        # Ground/water: 640x60 is 17% of the frame, under the ceiling, so it is selectable...
        eyes.look((320, 330))
        selector.update(frame)
        assert selector.selection is not None and selector.selection.box == SCENE["water"]

        # ...but raise the bar and the same mask is refused as "that is not an object".
        selector.config.max_area_fraction = 0.05
        eyes.away()
        selector.update(frame)
        eyes.look((320, 330))
        selector.update(frame)
        assert selector.selection is None, "the ground was offered as a target"
    finally:
        selector.close()


def check_stale_view() -> None:
    """A view that has moved on invalidates the embedding: hold the box dimmed, do not decode.

    Measured by comparing frames rather than by adding up camera degrees, which miss walking
    altogether and which MineStudio overreports badly while the mouse is captured.
    """
    frame = np.zeros((*FRAME, 3), np.uint8)
    for x0, y0, x1, y1 in SCENE.values():
        frame[y0:y1, x0:x1] = 90                    # something for a thumbnail to notice
    eyes, masker = Eyes(), SceneMasker()
    selector = GazeSelector(masker, ScreenMap(lambda: RECT), eyes,
                            SelectorConfig(lock_ms=0, deadzone=0.0, smoothing=1.0,
                                           max_change=6.0, release_ms=10_000),
                            report=lambda message: None)
    try:
        eyes.look((330, 190))
        assert settle(selector, frame) is not None
        assert selector.change < 1.0, f"a still view looks like motion ({selector.change})"
        decodes = masker.decodes

        # The world moves under a frozen embedding -- a step forward, a turn, either way the
        # pixels are no longer the ones that were encoded.
        selector.worker.close()
        moved = np.roll(frame, 80, axis=1)
        selector.update(moved)
        assert selector.change > 6.0, f"a moved view looks still ({selector.change})"
        assert masker.decodes == decodes, "decoded against an embedding of a different view"
        assert selector.selection is not None and selector.selection.fresh is False, \
            "a box from a stale view was still presented as current"

        # Back to the encoded view, and it is trusted again.
        selector.update(frame)
        assert masker.decodes > decodes, "never resumed once the view came back"
        assert selector.selection.fresh is True
    finally:
        selector.close()


def check_payload() -> None:
    """What the remote model is told: the centre of the box, normalized."""
    mask = np.zeros(FRAME, bool)
    mask[120:260, 260:400] = True
    selection = Selection(point=(330.0, 190.0), box=(260.0, 120.0, 400.0, 260.0),
                          mask=mask, frame=np.zeros((*FRAME, 3), np.uint8), locked=True)
    assert selection.centre == (330.0, 190.0), selection.centre
    payload = selection.payload()
    assert payload["point"] == [round(330 / 640, 4), round(190 / 360, 4)], payload
    assert payload["box"] == [round(260 / 640, 4), round(120 / 360, 4),
                              round(400 / 640, 4), round(260 / 360, 4)], payload
    assert payload["locked"] is True
    assert selection.contains((410, 190)) and not selection.contains((460, 190))


def main() -> int:
    for check in (check_channel, check_screen_map, check_hover_and_lock, check_still_gaze_locks, check_refusals,
                  check_stale_view, check_payload):
        check()
        print(f"  {check.__name__} ok")
    print("\nALL GAZE SELECTION TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
