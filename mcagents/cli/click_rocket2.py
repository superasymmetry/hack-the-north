"""Pinch at a Minecraft object and run ROCKET-2 toward it.

`TRACKER` is tracker.py's pinch test, but it runs under `.gaze-env`'s interpreter: MediaPipe
wants numpy 2 and opencv 5, and importing those here wedges the simulator's numpy 1.26 and
opencv 4.8. It reports `x y pinched` per camera frame, normalized and already mirrored.
"""
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

from mcagents import gui  # first: see the import-order rule in gui.py
import cv2

from mcagents.agents.rocket2 import Rocket2Agent, Rocket2Config
from mcagents.minecraft.session import EnvConfig, Session

GAZE_PYTHON = Path(__file__).resolve().parents[2] / ".gaze-env" / "bin" / "python"
DWELL = 0.25       # seconds a pinch must hold: it flickers open while the hand moves
THRESHOLD = 0.05   # normalized thumb-to-index distance that counts as pinched
TIMEOUT = 60.0     # seconds without a hand before giving up

TRACKER = f"""
import math
import cv2, mediapipe as mp

api = mp.solutions.hands
capture = cv2.VideoCapture(0)
with api.Hands(max_num_hands=1, min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as hands:
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        found = hands.process(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
        if not found.multi_hand_landmarks:
            continue
        points = found.multi_hand_landmarks[0].landmark
        index, thumb = points[api.HandLandmark.INDEX_FINGER_TIP], points[api.HandLandmark.THUMB_TIP]
        pinched = math.hypot(index.x - thumb.x, index.y - thumb.y) < {THRESHOLD}
        print(index.x, index.y, int(pinched), flush=True)
capture.release()
"""


def pinch_point(frame, window="ROCKET-2") -> Optional[Tuple[int, int]]:
    """Where a held pinch points on `frame` (RGB). None if it timed out or was cancelled.

    Deliberately the title the agent previews under: cv2 reuses a window by title, so the
    frozen frame stays up through the second of segmenting and the first forward pass,
    instead of the window vanishing and reopening.
    """
    rows, cols = frame.shape[:2]
    background = gui.to_bgr(frame)
    python = str(GAZE_PYTHON) if GAZE_PYTHON.exists() else sys.executable
    tracker = subprocess.Popen([python, "-c", TRACKER], stdout=subprocess.PIPE,
                               text=True, bufsize=1)
    latest: list = [None]  # one slot, last writer wins: a stale fingertip lags the hand

    def read() -> None:
        for line in tracker.stdout:
            x, y, pinched = line.split()
            latest[0] = (float(x), float(y), pinched == "1")

    threading.Thread(target=read, daemon=True).start()
    # A pinch left over from the last task must open before it can point again.
    ready, held, last = False, 0.0, time.monotonic()
    deadline = last + TIMEOUT
    try:
        while (now := time.monotonic()) < deadline:
            elapsed, last = now - last, now
            sample, point = latest[0], None
            if sample is None:
                held = 0.0
            else:
                x, y, pinched = sample
                point = (min(max(int(x * cols), 0), cols - 1),
                         min(max(int(y * rows), 0), rows - 1))
                ready = ready or not pinched
                held = held + elapsed if pinched and ready else 0.0
                if held >= DWELL:
                    return point
            if not gui.GUI_READY:
                time.sleep(0.02)
                continue
            preview = background.copy()
            if point is not None:
                cv2.circle(preview, point, 5, (0, 0, 255) if held else (0, 255, 0),
                           -1 if held else 1)
            gui.show(preview, window)
            if cv2.waitKey(20) == 27:  # ESC
                return None
        print("no pinch seen -- is the camera visible to you?", flush=True)
        return None
    finally:
        tracker.terminate()


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
        # Back to the preview after each task, so the next pinch starts from the view
        # ROCKET-2 left off at; ESC or the no-hand timeout ends the run.
        while (point := pinch_point(agent.info["pov"])) is not None:
            result = agent.run(point=point, interaction="Approach", stop={"steps": 200, "arrive": {"distance": 8}})
            # An `arrive` of distance alone cannot fire without an anchor, so say which
            # ending this was: got there, or never had a target position to measure
            # against.
            print(result, agent.status()["range"] or "never anchored", flush=True)


if __name__ == "__main__":
    main()
