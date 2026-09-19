"""OpenCV windows, and the one import-order rule that makes them work.

PyAV -- pulled in by minestudio -- ships its own copy of libxcb in av.libs/. Once that is
loaded alongside the one Qt is using, OpenCV's *first* cv2.imshow() never returns: it blocks
forever creating its window, with no error and nothing on screen. Every imshow after a
successful one is fine, so the fix is to make the first one happen before PyAV is in the
process, which is what importing this module does.

    import mcagents.gui          # first, above anything that pulls in minestudio
    from minestudio.simulator import MinecraftSim

`GUI_READY` says whether that worked. When it did not -- no DISPLAY, a headless OpenCV
build, or minestudio was already imported -- `pick_point()` falls back to writing the frame
out and asking on the terminal, and previews turn themselves off.
"""
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _prewarm() -> bool:
    """Open and close a throwaway window while that is still possible; say whether it worked."""
    if not os.environ.get("DISPLAY"):
        return False
    if "av" in sys.modules:
        print("mcagents.gui: no preview windows -- PyAV was imported first (import mcagents.gui "
              "above minestudio to get them back).", file=sys.stderr, flush=True)
        return False
    try:
        cv2.namedWindow("_mcagents_warmup", cv2.WINDOW_NORMAL)
        cv2.waitKey(1)
        cv2.destroyWindow("_mcagents_warmup")
        cv2.waitKey(1)
    except cv2.error:  # headless build, unusable DISPLAY
        return False
    return True


#: Whether cv2 windows work in this process.
GUI_READY = _prewarm()


def to_bgr(frame: np.ndarray) -> np.ndarray:
    """An RGB frame from the sim as something cv2 will accept.

    MineRL hands back a vertically flipped *view* of its frame buffer -- an array with a
    negative stride -- and neither cv2 nor torch will take one, so everything that leaves
    the sim has to be materialized first.
    """
    return cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)


def show(image: np.ndarray, window: str) -> bool:
    """Display a BGR image; returns False once windows are known not to work."""
    if not GUI_READY:
        return False
    try:
        cv2.imshow(window, image)
        cv2.waitKey(1)
        return True
    except cv2.error:  # not worth killing a run over
        return False


def pick_point(frame: np.ndarray, window: str = "click the target") -> Optional[Tuple[int, int]]:
    """Ask a human for a point on `frame` (RGB). Returns [x, y], or None if they declined.

    A stand-in for the pointing model, so a ROCKET-2 subgoal can be demoed before a targeter
    is wired in. ESC in the window (or a blank line in the terminal fallback) means "skip".
    """
    if not GUI_READY:
        return _pick_point_on_stdin(frame)

    clicked: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    cv2.imshow(window, to_bgr(frame))
    cv2.setMouseCallback(window, on_mouse)
    while not clicked:
        if cv2.waitKey(20) == 27:  # ESC
            cv2.destroyWindow(window)
            return None
    cv2.destroyWindow(window)
    return clicked[0]


def _pick_point_on_stdin(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """The no-window fallback: write the frame somewhere it can be looked at, ask on stdin."""
    path = Path(tempfile.gettempdir()) / "mcagents_point.png"
    cv2.imwrite(str(path), to_bgr(frame))
    height, width = frame.shape[:2]
    print(f"No window available. Open {path} ({width}x{height}) and give the target as "
          f"'x y' -- blank to skip this goal.", flush=True)
    try:
        reply = input("point> ").strip()
    except EOFError:
        raise RuntimeError(
            f"pick_point() has neither a window nor a terminal to ask on. Put explicit "
            f"coordinates in the plan instead of \"click\" -- the frame is at {path}."
        ) from None
    if not reply:
        return None
    x, y = (int(float(value)) for value in reply.replace(",", " ").split()[:2])
    return (x, y)
