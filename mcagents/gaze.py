"""Where the eyes are, from the process with the webcam to the process with the game.

The third channel of the same shape as [frames.py](frames.py) and [goals.py](goals.py), and
the same argument decides its shape: a gaze sample is worthless the moment a newer one
exists, so this is one slot, last writer wins, and nobody blocks on anybody. The tracker
lives in its own process and its own virtualenv -- it wants mediapipe and an `opencv-python`
that must not be allowed near the process running Minecraft (see docs/gaze.md) -- so a file
on a tmpfs is also the cheapest thing that crosses that boundary.

    # in the gaze process, as fast as the camera produces samples
    gaze.write(GazeSample(x=1730.0, y=921.0))          # screen pixels

    # in the game process, once per step
    sample = gaze.read_latest()                        # or None

What differs from a frame is `max_age`, and it differs by a factor of thirty. A stale frame
is a slightly old picture; a stale *gaze* is a red box sitting on an object the person
stopped looking at half a second ago, which reads as the system having locked on and being
wrong rather than as the system having nothing to say. So the bar is `DEFAULT_MAX_AGE`, about
the length of one fixation: if the tracker blinks out, loses the face, or its process dies,
the selection goes away within a few frames instead of lying there.

Coordinates are **screen pixels**, the raw thing the tracker measures, and they stay that way
until something that knows where the game window is turns them into frame pixels -- which is
`ScreenMap` in [perception/selection.py](perception/selection.py). Nothing here needs to know
a window exists, so moving the window mid-session costs nothing.
"""
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: Where the sample lives. Both processes are on this laptop, so this is a runtime path.
DEFAULT_PATH = os.environ.get("MCAGENTS_GAZE_PATH", "/tmp/mcagents-gaze.json")

#: How old a sample may be and still be believed. See the module docstring -- this is a
#: fixation, not a network timeout.
DEFAULT_MAX_AGE = 0.3

#: A sample that cannot plausibly be one. Same reasoning as frames.MAX_BYTES: this is a
#: well-known path that a typo in $MCAGENTS_GAZE_PATH can point anywhere.
MAX_BYTES = 4096


@dataclass
class GazeSample:
    """One look, in screen pixels, with the origin at the top-left of the display."""
    x: float
    y: float
    #: Wall clock when the tracker produced it. Carried for diagnostics only; staleness is
    #: decided from the file's mtime, so a sample cannot outlive the process that wrote it
    #: by lying about when it was taken.
    t: float = 0.0
    #: False when the tracker is running but cannot see the eyes -- a blink, a turned head.
    #: Distinct from "no sample at all", which is the tracker not running.
    valid: bool = True
    #: Optional hand landmarks in screen pixels, in MediaPipe's 21-point order.
    landmarks: Optional[List[Tuple[float, float]]] = None

    def payload(self) -> dict:
        payload = {"x": float(self.x), "y": float(self.y),
                   "t": float(self.t or time.time()), "valid": bool(self.valid)}
        if self.landmarks is not None:
            payload["landmarks"] = [[float(x), float(y)] for x, y in self.landmarks]
        return payload


def write(sample: GazeSample, path: str = DEFAULT_PATH) -> None:
    """Replace the published sample, atomically.

    `os.replace` for the same reason frames.py uses it: a reader either sees the whole
    previous sample or the whole new one, never half a JSON object. The temporary carries
    the pid so two trackers cannot land on each other's partial file.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temporary, "w") as handle:
            json.dump(sample.payload(), handle)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_latest(path: str = DEFAULT_PATH,
                max_age: float = DEFAULT_MAX_AGE,
                allow_invalid: bool = False) -> Optional[GazeSample]:
    """The published sample, or None if there is not a fresh and usable one.

    None covers "no tracker has ever run", "the tracker stopped", "the tracker is wedged"
    and "the tracker cannot see your eyes" -- deliberately, because from here they are one
    fact: there is nowhere to point right now, so nothing should be highlighted.

    Everything is decided from the open handle rather than the path, so a write landing
    mid-read cannot leave the mtime describing one sample and the bytes another.
    """
    try:
        with open(path, "rb") as handle:
            status = os.fstat(handle.fileno())
            if not status.st_size or status.st_size > MAX_BYTES:
                return None
            if time.time() - status.st_mtime > max_age:
                return None
            payload = json.loads(handle.read(MAX_BYTES))
        sample = GazeSample(x=float(payload["x"]), y=float(payload["y"]),
                            t=float(payload.get("t", status.st_mtime)),
                            valid=bool(payload.get("valid", True)),
                            landmarks=[(float(point[0]), float(point[1]))
                                       for point in payload.get("landmarks", [])]
                            or None)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return sample if sample.valid or allow_invalid else None


class GazePublisher:
    """The writing half, for the tracker's loop. Never the reason a session ends.

    A tracker that cannot write its sample is a broken side-channel, not a broken session --
    the game is still playable, it just has no gaze in it -- so the first failure is reported
    and the publisher turns itself off, exactly as `frames.FramePublisher` does.
    """

    def __init__(self, path: str = DEFAULT_PATH, report=None):
        self.path = path
        self.report = report or (lambda message: print(message, flush=True))
        self.enabled = True
        self.published = 0

    def offer(self, x: float, y: float, valid: bool = True,
              landmarks: Optional[Sequence[Tuple[float, float]]] = None) -> bool:
        """Publish one sample. Returns whether it was written.

        Unrate-limited, unlike the frame publisher: a sample is ~60 bytes onto a tmpfs, and
        the whole point is to be as current as the camera allows.
        """
        if not self.enabled:
            return False
        try:
            write(GazeSample(x=x, y=y, t=time.time(), valid=valid,
                         landmarks=list(landmarks) if landmarks is not None else None),
                self.path)
        except Exception as exc:
            self.enabled = False
            self.report(f"gaze: publishing to {self.path} failed, giving up on it "
                        f"({type(exc).__name__}: {exc})")
            return False
        self.published += 1
        return True
