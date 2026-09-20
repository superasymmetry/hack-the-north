"""Pinch at a Minecraft object and run ROCKET-2 toward it.

`TRACKER` is tracker.py's pinch test, but it runs under `.gaze-env`'s interpreter: MediaPipe
wants numpy 2 and opencv 5, and importing those here wedges the simulator's numpy 1.26 and
opencv 4.8. It reports `x y pinched fist swipe` per camera frame, normalized and already
mirrored.

Six gestures, and the split between them is what the plumbing is shaped around. A pinch is a
*pose held at a place*: it names a point, and the point is the goal. The rest carry no point,
so they are commands -- the four swipes turn the camera the way the hand went, and a closed
fist cancels whatever ROCKET-2 is doing and leaves it standing still.
"""
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from mcagents import gui  # first: see the import-order rule in gui.py
import cv2

from mcagents.agents.rocket2 import Rocket2Agent, Rocket2Config
from mcagents.minecraft.session import EnvConfig, Session
from mcagents.perception import identify

GAZE_PYTHON = Path(__file__).resolve().parents[2] / ".gaze-env" / "bin" / "python"
CUE = Path(__file__).resolve().parents[2] / "assets" / "let-me-do-it-for-you.mp3"
DWELL = 0.25       # seconds a pinch must hold: it flickers open while the hand moves
THRESHOLD = 0.05   # normalized thumb-to-index distance that counts as pinched
TIMEOUT = 60.0     # seconds without a hand before giving up
SWIPE = 0.18       # fraction of frame width the hand must cross to count as a swipe
# Taller bar for the vertical one, and not fussiness: x and y are both fractions of a frame
# that is wider than it is tall, so the same physical movement of your hand scores about 1.5x
# higher in y. Equal numbers here would mean an up-swipe needed noticeably less arm than a
# left-swipe, and vertical twitch while swiping sideways would win the dominant axis.
SWIPE_Y = 0.26     # same, as a fraction of frame height
WINDOW = 0.45      # seconds it has to cross it in
QUIET = 0.6        # seconds swipes are ignored after one fires, so the arm can come back
HAND = 9           # the middle-finger knuckle: the steadiest landmark on a moving hand
#: Per swipe: degrees of [yaw, pitch] per step, and steps. Spread over steps so it reads as a
#: turn rather than the view teleporting. MineRL's convention -- positive yaw right, positive
#: pitch *down* -- so up-swipe carries a negative pitch. Less pitch than yaw on purpose:
#: 45 degrees across is a look around, 45 up is the sky.
TURNS = {"left": (-5.0, 0.0, 9), "right": (5.0, 0.0, 9),
         "up": (0.0, -5.0, 6), "down": (0.0, 5.0, 6)}

TRACKER = f"""
import math, time
from collections import deque
import cv2, mediapipe as mp

api = mp.solutions.hands
# tip, pip. A finger is curled when its tip hangs *below* its middle joint -- image y grows
# downward -- and all four curled is a fist.
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]
capture = cv2.VideoCapture(0)
trail = deque()  # (when, x, y), trimmed to the last {WINDOW}s: a swipe is motion, not a pose
was_fist, quiet = False, 0.0
with api.Hands(max_num_hands=1, min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as hands:
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        found = hands.process(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
        if not found.multi_hand_landmarks:
            # Deliberately NOT clearing the trail. MediaPipe drops the hand for a frame or
            # two exactly when it is moving fastest -- so clearing here threw away the
            # evidence for precisely the swipes that were most clearly swipes. Stale samples
            # cannot cause a phantom one anyway: the window below ages them out first.
            continue
        points = found.multi_hand_landmarks[0].landmark
        index, thumb = points[api.HandLandmark.INDEX_FINGER_TIP], points[api.HandLandmark.THUMB_TIP]
        pinched = math.hypot(index.x - thumb.x, index.y - thumb.y) < {THRESHOLD}
        fist = all(points[tip].y > points[joint].y for tip, joint in FINGERS)
        now = time.monotonic()
        hand = points[{HAND}]
        trail.append((now, hand.x, hand.y))
        while trail and now - trail[0][0] > {WINDOW}:
            trail.popleft()
        swipe = "none"
        # Not while making a fist: a fist travelling across the frame is one gesture, not two.
        if trail and not fist and now >= quiet:
            dx, dy = hand.x - trail[0][1], hand.y - trail[0][2]
            # One axis wins outright. A real swipe is never purely horizontal or vertical,
            # and scoring them separately would let one diagonal fire both.
            across = abs(dx) / {SWIPE} >= abs(dy) / {SWIPE_Y}
            if across and abs(dx) > {SWIPE}:
                swipe = "right" if dx > 0 else "left"
            elif not across and abs(dy) > {SWIPE_Y}:
                swipe = "down" if dy > 0 else "up"   # image y grows downward
            if swipe != "none":
                # Both, and for different reasons: clearing stops the same swipe firing on
                # every following frame, and the quiet period stops the arm *coming back*
                # from firing the opposite one and undoing the turn.
                trail.clear()
                quiet = now + {QUIET}
        # Rising edge only. The pose lasts as long as the hand holds it; the command is one.
        print(index.x, index.y, int(pinched), int(fist and not was_fist), swipe, flush=True)
        was_fist = fist
capture.release()
"""


class Cue:
    """The clip that sounds the moment a pinch commits.

    `pw-play` and not a library, for local_client.Cue's reason -- it decodes the mp3 itself
    and starts at once -- but its own small copy rather than an import: local_client drags in
    websockets, numpy and the ASR stack, and this process has an import order to keep (see
    gui.py). No mic here, so none of that Cue's muting comes with it.

    Fire-and-forget, and a clip still playing is not restarted: a second pinch while the line
    is still talking is the same voice interrupting itself.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._playing: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._complained = False

    def play(self) -> None:
        """Returns at once."""
        with self._lock:
            if self._playing is not None and self._playing.poll() is None:
                return
            try:
                self._playing = subprocess.Popen(
                    ["pw-play", "--media-role", "Notification", str(self.path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as exc:
                self._playing = None
                if not self._complained:  # once: a missing player misses every cue
                    self._complained = True
                    print(f"cue: could not play ({type(exc).__name__}: {exc})", flush=True)

    def close(self) -> None:
        with self._lock:
            playing, self._playing = self._playing, None
        if playing is not None and playing.poll() is None:
            playing.kill()


cue = Cue(CUE)


class Hands:
    """The tracker subprocess, read in the background.

    Two channels, because the gestures are two different kinds of thing. `sample` is the
    held pose -- one slot, last writer wins, since a stale fingertip lags the hand and a
    pinch that was true a frame ago is true enough. `take()` drains queued *edges*, which
    cannot be overwritten that way: a swipe is true for a single frame, and the loops below
    poll slower than the camera runs, so last-writer-wins would swallow most of them.

    Owned by `main` and outliving any one goal, so a fist can be seen while ROCKET-2 is
    driving -- which is the only time cancelling it means anything.
    """

    def __init__(self) -> None:
        python = str(GAZE_PYTHON) if GAZE_PYTHON.exists() else sys.executable
        self.tracker = subprocess.Popen([python, "-c", TRACKER], stdout=subprocess.PIPE,
                                        text=True, bufsize=1)
        self.sample: Optional[Tuple[float, float, bool]] = None
        self.events: list = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.tracker.stdout:
            x, y, pinched, fist, swipe = line.split()
            self.sample = (float(x), float(y), pinched == "1")
            if fist == "1":
                self.events.append("fist")
            if swipe != "none":
                self.events.append(swipe)  # "left" | "right" | "up" | "down"

    def take(self) -> Optional[str]:
        """The oldest gesture still waiting: a key of `TURNS`, "fist", or None."""
        return self.events.pop(0) if self.events else None

    def close(self) -> None:
        self.tracker.terminate()


def pinch_point(hands: Hands, frame, label: Optional[Callable[[Tuple[int, int]], str]] = None,
                window="ROCKET-2") -> Optional[Tuple[str, object]]:
    """What the hand asked for on `frame` (RGB): ("point", (x, y)) or ("turn", direction).

    None if it timed out or was cancelled. A fist here is drained and ignored on purpose:
    nothing is running to stop, and leaving it queued would cancel the *next* goal instead.

    Deliberately the title the agent previews under: cv2 reuses a window by title, so the
    frozen frame stays up through the second of segmenting and the first forward pass,
    instead of the window vanishing and reopening.

    :param label: what to write beside the dot for a point -- how the interaction this pinch
        would run gets shown *before* the pinch commits to it. Called once a frame on a
        frozen frame, so it has to stay cheap; a voxel cast is (see `mcagents.perception.
        identify`), a model would not be.
    """
    rows, cols = frame.shape[:2]
    background = gui.to_bgr(frame)
    # A pinch left over from the last task must open before it can point again.
    ready, held, last = False, 0.0, time.monotonic()
    deadline = last + TIMEOUT
    while (now := time.monotonic()) < deadline:
        elapsed, last = now - last, now
        gesture = hands.take()
        if gesture in TURNS:
            return "turn", gesture
        sample, point = hands.sample, None
        if sample is None:
            held = 0.0
        else:
            x, y, pinched = sample
            point = (min(max(int(x * cols), 0), cols - 1),
                     min(max(int(y * rows), 0), rows - 1))
            ready = ready or not pinched
            held = held + elapsed if pinched and ready else 0.0
            if held >= DWELL:
                cue.play()
                return "point", point
        if not gui.GUI_READY:
            time.sleep(0.02)
            continue
        preview = background.copy()
        if point is not None:
            cv2.circle(preview, point, 5, (0, 0, 255) if held else (0, 255, 0),
                       -1 if held else 1)
            if label is not None:
                # Twice, thick and dark under thin and light: Minecraft is as likely to
                # put white sky behind this text as dark stone.
                at = (point[0] + 10, point[1] - 10)
                for color, weight in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                    cv2.putText(preview, label(point), at, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, weight, cv2.LINE_AA)
        gui.show(preview, window)
        if cv2.waitKey(20) == 27:  # ESC
            return None
    print("no pinch seen -- is the camera visible to you?", flush=True)
    return None


def main() -> None:
    from minestudio.simulator.callbacks import PrevActionCallback, SummonMobsCallback

    # Ahead of PrevActionCallback, and that is not a preference: SummonMobsCallback's
    # after_reset re-wraps a fresh obs from execute_cmd, dropping the env_prev_action
    # key a PrevActionCallback in front of it had just added -- which Rocket2 requires.
    # Down the -x/+z diagonal, because that is where you are looking: CityCallback lands
    # the player at yaw 45. The ranges are world-axis offsets, not facing-relative, so a
    # box centred on the player puts most of the herd behind you.
    mobs = [{"name": "cow", "number": 10, "range_x": [-10, -2], "range_z": [2, 10]}]
    callbacks = [SummonMobsCallback(mobs), PrevActionCallback()]

    with Session(EnvConfig.from_env(city_default=True)) as session:
        sim = session.open(callbacks=callbacks, action_type="env",
                           obs_size=(224, 224))
        config = Rocket2Config.from_env()
        # Never end as "lost": the lock goes stale whenever the view swings hard, but
        # the target is still where it was -- let the stop conditions decide instead.
        config.lock_grace = 10 ** 9
        agent = Rocket2Agent(sim, config)
        hands = Hands()
        # Back to the preview after each task, so the next pinch starts from the view
        # ROCKET-2 left off at; ESC or the no-hand timeout ends the run.
        try:
            while True:
                # Frozen for the length of the pinch, so the same cast answers for every frame
                # of the preview and for the goal that comes out of it.
                frame, info = agent.info["pov"], agent.info
                decide = lambda at: identify.choose(sim, info, at, frame.shape, config.fov)
                asked = pinch_point(hands, frame, label=lambda at: str(decide(at)))
                if asked is None:
                    break
                kind, value = asked
                if kind == "turn":
                    yaw, pitch, steps = TURNS[value]
                    for _ in range(steps):
                        agent.turn_step(yaw, pitch)
                    continue
                choice = decide(value)
                print(f"pinched {choice.block or 'nothing within reach'} -> {choice}", flush=True)
                agent.set_goal(value, interaction=choice.interaction, stop=choice.stop)
                # Stepped here rather than by `agent.run`, which blocks: this is the only
                # place a fist can be noticed *while* the policy is driving. Cancelling
                # presses nothing on the way out, so the player simply stands still.
                while agent.busy:
                    agent.step()
                    if hands.take() == "fist":  # swipes during a goal are dropped, not queued
                        agent.cancel("fist")
                # An `arrive` of distance alone cannot fire without an anchor, so for the goals
                # that use one say which ending this was: got there, or never had a target
                # position to measure against.
                print(agent.result, agent.status()["range"] or "never anchored", flush=True)
        finally:
            cue.close()
            hands.close()


if __name__ == "__main__":
    main()
