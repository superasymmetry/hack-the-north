"""The shared shape of "find `text` in this frame": a box, a point, and who produces them.

Two backends implement it -- OWLv2 locally (`owlv2.py`) and a remote VLM (`vlm.py`) -- and
callers pick between them with `load_targeter()` so nothing downstream knows which is running.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


@dataclass
class Detection:
    """One match, in the frame's own pixel coordinates."""
    score: float
    box: Box                                     # x0, y0, x1, y1

    @property
    def point(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def clipped(self, width: float, height: float) -> "Detection":
        """The same detection with the box trimmed to the frame."""
        x0, y0, x1, y1 = self.box
        return Detection(self.score, (max(0.0, min(x0, x1)), max(0.0, min(y0, y1)),
                                      min(width, max(x0, x1)), min(height, max(y0, y1))))


class Targeter(Protocol):
    """text + frame -> where to point. Implemented by OwlV2Targeter and VLMTargeter."""

    def detect(self, frame: np.ndarray, text: str) -> List[Detection]:
        ...

    def locate(self, frame: np.ndarray, text: str, prefer: str = "score") -> Optional[Tuple[float, float]]:
        ...


def best(found: Sequence[Detection], prefer: str = "score") -> Optional[Tuple[float, float]]:
    """The point to hand a controller, or None if nothing matched.

    `prefer="score"` trusts the detector's own ranking; "largest" takes the biggest box,
    which is a decent stand-in for "nearest" when several of the same thing are in view.
    """
    if not found:
        return None
    if prefer == "largest":
        found = sorted(found, key=lambda d: d.area, reverse=True)
    elif prefer != "score":
        raise ValueError(f"prefer must be 'score' or 'largest', not {prefer!r}")
    return found[0].point


def annotate(frame: np.ndarray, found: Sequence[Detection]) -> np.ndarray:
    """Draw the boxes and their centres on an RGB frame, for checking a query by eye."""
    import cv2

    canvas = np.ascontiguousarray(frame).copy()
    for index, detection in enumerate(found):
        x0, y0, x1, y1 = (int(v) for v in detection.box)
        colour = (0, 255, 0) if index == 0 else (0, 160, 255)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
        cv2.circle(canvas, (int(detection.point[0]), int(detection.point[1])), 4, colour, -1)
        cv2.putText(canvas, f"{detection.score:.2f}", (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return canvas


def load_targeter(backend: Optional[str] = None) -> Targeter:
    """Build the targeter named by `backend` (or $MCAGENTS_TARGETER, default "owl").

    Imported lazily: OWLv2 is ~600 MB of weights on a card that is already holding ROCKET-2
    and SAM-2, and the VLM backend needs no local model at all.
    """
    backend = (backend or os.environ.get("MCAGENTS_TARGETER", "owl")).lower()
    if backend in ("owl", "owlv2"):
        from mcagents.perception.owlv2 import OwlV2Targeter
        return OwlV2Targeter()
    if backend == "vlm":
        from mcagents.perception.vlm import VLMTargeter
        return VLMTargeter()
    raise ValueError(f"unknown targeter backend {backend!r} -- expected 'owl' or 'vlm'")
