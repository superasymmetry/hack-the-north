"""The latest game frame, from whoever holds the sim to whoever holds the socket.

The two halves of a voice session are two processes on this laptop. ROCKET-2 has the sim and
therefore the pixels; [local_client.py](local_client.py) has the one WebSocket to the model
on the GPU box and no idea what the game looks like. This module is the seam: the agent
publishes its own view, the client picks up whatever is there and sends it.

    # in the agent process, once per step
    publisher = FramePublisher(interval=1.0)
    publisher.offer(self.obs["image"])          # RGB uint8, (224, 224, 3)

    # in the client process, every few seconds
    latest = read_latest()                      # (jpeg_bytes, mtime), or None

A file, rather than a socket or a queue, because of what the data is. A frame is worthless
the moment a newer one exists, so the channel wants exactly one slot, last writer wins, no
backlog and no readers to block on -- which is a file, and is not any queue. It also means
neither process has to exist for the other to start: the client polls a path that may never
appear, the agent writes to one nobody is reading, and a restart of either is not an event.

`os.replace` is what makes that safe. It is atomic within a filesystem, so a reader either
sees the whole previous JPEG or the whole new one and never a half-written frame; writing in
place would hand out torn files a few times a minute and they would decode as garbage rather
than as an error.

Staleness is the reader's job and it is not optional. The file outlives the process that
wrote it -- a Slurm kill, a crash, a ctrl-c all leave the last frame sitting there looking
perfectly valid -- so `read_latest` refuses anything older than `max_age` instead of sending
the model a picture of a Minecraft session that ended twenty minutes ago.
"""
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: Where the frame lives. Both processes are on this laptop, so this is a runtime path and
#: not a project file: /tmp is a tmpfs here, which means the write never reaches a disk.
DEFAULT_PATH = os.environ.get("MCAGENTS_FRAME_PATH", "/tmp/mcagents-frame.jpg")

#: A frame that cannot plausibly be one. The channel is a well-known path that a typo in
#: MCAGENTS_FRAME_PATH can point anywhere, and a reader that trusts it would happily load a
#: multi-gigabyte file and then base64 it, which is 33% worse. Even a full-resolution q95
#: JPEG is well under a megabyte; the frames this actually carries are ~12 KB.
MAX_BYTES = 4 * 1024 * 1024

#: JPEG quality. Measured over 44 frames of a real run (logs/jarvisvla/episode_1.mp4,
#: downsized to the 224x224 the policy sees): 10.3-13.7 KB, median 11.5 KB. That is ~15 KB
#: of base64 on the wire, or 5 KB/s at one frame every three seconds -- and an encode far
#: below one 20 Hz step, which is what lets this sit in the step loop at all.
DEFAULT_QUALITY = 80


def write(jpeg: bytes, path: str = DEFAULT_PATH) -> None:
    """Replace the published frame with `jpeg`, atomically.

    The temporary name carries the pid so two publishers -- a leftover run, a second
    terminal -- cannot land on each other's partial file. They will still fight over the
    final path, but the loser's frame is simply superseded, which is the intended semantics.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temporary, "wb") as handle:
            handle.write(jpeg)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def encode(frame: np.ndarray, quality: int = DEFAULT_QUALITY) -> bytes:
    """An RGB frame from the sim as JPEG bytes.

    cv2 is imported here rather than at module scope so that the reading half -- the voice
    client, which only ever moves bytes -- does not pull OpenCV in behind it. That is also
    the import-order rule in [gui.py](gui.py): the fewer processes that load cv2 without
    needing it, the fewer that can trip over it.

    MineRL hands back vertically flipped *views* of its frame buffer (negative strides), and
    cv2 will not take one, so the frame is materialized before it is encoded.
    """
    import cv2

    image = np.ascontiguousarray(frame)
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"cv2 could not encode a {image.shape} {image.dtype} frame as JPEG")
    return buffer.tobytes()


def read_latest(path: str = DEFAULT_PATH,
                max_age: float = 10.0) -> Optional[Tuple[bytes, float]]:
    """The published frame and its mtime, or None if there is not a fresh one.

    None covers all three of "no agent has ever run", "the agent stopped" and "the agent is
    alive but wedged", deliberately: from here they are the same fact -- there is nothing
    current to send -- and the caller has nothing different to do about them.

    Everything is decided from the open *handle* rather than from the path, so a publish
    landing mid-read cannot leave the mtime describing one frame and the bytes another --
    `os.replace` swaps the directory entry and leaves this handle on the file it opened.

    Size is checked before the read, not after. This path is a well-known one that a typo
    can point at anything, and "read it, then find out it was 5 GB" is not a check.
    """
    try:
        with open(path, "rb") as handle:
            status = os.fstat(handle.fileno())
            if not status.st_size or status.st_size > MAX_BYTES:
                return None
            if time.time() - status.st_mtime > max_age:
                return None
            return handle.read(MAX_BYTES), status.st_mtime
    except OSError:
        return None


def selection_path(path: str = DEFAULT_PATH) -> str:
    """Where the selection that goes with `path` lives."""
    return f"{path}.json"


def write_selection(selection: Optional[Dict[str, Any]], path: str = DEFAULT_PATH) -> None:
    """Publish what was highlighted in the frame at `path` -- or that nothing was.

    A sidecar rather than a second channel, and written by whoever writes the frame, because
    the two only mean anything together: "the box is at (0.51, 0.42)" is a statement *about a
    picture*, and pairing it with a different picture is worse than sending neither. Writing
    `None` is meaningful and not a no-op -- it is how a selection that has gone away stops
    being attached to every frame after it.
    """
    body = json.dumps({"selection": selection, "t": round(time.time(), 3)}).encode()
    write(body, selection_path(path))


def read_selection(path: str = DEFAULT_PATH,
                   max_age: float = 10.0) -> Optional[Dict[str, Any]]:
    """What was highlighted in the published frame, or None if nothing was.

    `max_age` matches the frame's own, since a selection older than the frame it describes
    describes a frame nobody is sending any more.
    """
    try:
        with open(selection_path(path), "rb") as handle:
            status = os.fstat(handle.fileno())
            if not status.st_size or status.st_size > MAX_BYTES:
                return None
            if time.time() - status.st_mtime > max_age:
                return None
            body = json.loads(handle.read(MAX_BYTES))
    except (OSError, ValueError):
        return None
    selection = body.get("selection") if isinstance(body, dict) else None
    return selection if isinstance(selection, dict) else None


class FramePublisher:
    """Rate-limited publishing of the agent's view, safe to call from the step loop.

    Rate-limited because the loop runs at ~20-33 Hz and nothing downstream wants frames that
    fast: the client sends one every few seconds, so encoding every step would be ~30x the
    work for a frame that is thrown away. The default interval is *shorter* than the
    client's on purpose -- the two poll independently, so publishing at the same cadence the
    client reads at would make the frame it picks up as much as two intervals old. At 1s
    against a 3s client, what goes on the wire is never more than about a second stale.

    It also refuses to be the reason a run dies. A full disk, a read-only /tmp or an
    unencodable frame is a broken side-channel, not a broken rollout, so the first failure
    is reported and the publisher turns itself off.
    """

    def __init__(self, interval: float = 1.0, path: str = DEFAULT_PATH,
                 quality: int = DEFAULT_QUALITY, report=None):
        self.interval = interval
        self.path = path
        self.quality = quality
        self.report = report or (lambda message: print(message, flush=True))
        self.enabled = interval > 0
        self.published = 0
        self._last = 0.0

    def offer(self, frame: Optional[np.ndarray],
              selection: Optional[Dict[str, Any]] = None) -> bool:
        """Publish `frame`, and what was highlighted in it, if one is due.

        The two go out together or not at all, so the coordinates the model is given always
        describe the picture it is looking at. See `write_selection`.
        """
        if not self.enabled or frame is None:
            return False
        now = time.monotonic()
        if now - self._last < self.interval:
            return False
        try:
            write(encode(frame, self.quality), self.path)
            write_selection(selection, self.path)
        except Exception as exc:                       # never take the rollout down with it
            self.enabled = False
            self.report(f"frames: publishing to {self.path} failed, giving up on it "
                        f"({type(exc).__name__}: {exc})")
            return False
        self._last = now
        self.published += 1
        return True
