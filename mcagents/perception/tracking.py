"""Keep hold of *which* object a goal pointed at, frame to frame.

ROCKET-2 is given one mask on one frame and does its own cross-view alignment from there,
which is what makes it fast -- and what lets it drift. Walk up to a red building and the
goal view (a small red building, far away) no longer looks like the view (a wall), while
the *next* red building down the street looks exactly like it. Nothing downstream of the
policy knew the difference, so it kept walking.

`InstanceLock` is that missing check, and deliberately a dumb one. It never asks what the
target *is* -- a description would re-resolve to the best match, which is the bug -- only
where the same pixels went:

1. Between updates the box is carried by the camera's own rotation. The action says how
   many degrees the view turned, the field of view says how many pixels that is, so a
   building that slid 90 px left because the agent turned right is predicted there rather
   than looked for.
2. Every `every` steps SAM-2 re-segments at the predicted mask's most interior point, and
   the result is accepted only if it overlaps the prediction (mask IoU), has not changed
   size implausibly, and is still roughly the colour it was. An identical building
   elsewhere in the frame never overlaps. The colour check is there because a point prompt
   always returns *something*: with the target gone, SAM-2 happily segments a blob of
   whatever is behind it at the same place and size.
3. A lock that goes `grace` steps without an accepted update is lost. The caller halts;
   nothing here ever picks a new target.

SAM-2 tiny is ~116 ms a call fp32 through the processor on the laptop GPU, and ~35 ms with
bf16 and preprocessing on the card -- hence `every` rather than every step.
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]


def mask_box(mask: np.ndarray) -> Optional[Box]:
    """The tight [x0, y0, x1, y1) box around a boolean mask, in pixels. None if it is empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def mean_lab(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The mean colour under `mask` in CIE Lab (L 0..100), where distance tracks what eyes see."""
    lab = cv2.cvtColor(np.ascontiguousarray(frame).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    return lab[mask].mean(axis=0) if mask.any() else np.zeros(3, np.float32)


def pixels_per_degree(height: int, fov_degrees: float) -> float:
    """How far the image moves per degree of camera rotation, near the centre.

    Minecraft's FOV setting is the *vertical* field of view (70 by default, which is what
    MineRL leaves it at), so the focal length comes from the frame's height.
    """
    focal = (height / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    return focal * math.pi / 180.0


@dataclass
class LockConfig:
    #: Steps between SAM-2 re-segmentations. Between them the box moves with the camera.
    every: int = 3
    #: Steps without an accepted update before the lock counts as lost.
    grace: int = 20
    #: Mask IoU against the camera-shifted previous mask needed to accept a re-segmentation.
    min_iou: float = 0.3
    #: How much the mask may grow or shrink in one update and still be the same object.
    max_area_ratio: float = 3.0
    #: Largest CIE Lab distance between the candidate's mean colour and the target's. Wide
    #: enough for a face in shade against the same face lit; far narrower than a brown trunk
    #: against the black behind it (~40).
    max_color_distance: float = 30.0
    #: Vertical field of view of the game camera, degrees.
    fov: float = 70.0


class InstanceLock:
    """One target, followed by overlap and continuity. See the module docstring."""

    def __init__(self, masker, frame: np.ndarray, mask: np.ndarray,
                 config: Optional[LockConfig] = None):
        self.masker = masker
        self.config = config or LockConfig()
        self.height, self.width = frame.shape[:2]
        self.ppd = pixels_per_degree(self.height, self.config.fov)
        self.mask = mask.astype(bool)
        self.box: Optional[Box] = mask_box(self.mask)
        self.color = mean_lab(frame, self.mask)
        #: Pixels the view has moved since `mask` was last accepted.
        self._shift = np.zeros(2, np.float64)
        self._steps = 0
        #: Steps since the last accepted update; 0 right after one.
        self.stale = 0
        self.updates = 0
        self.rejections = 0

    # ------------------------------------------------------------------ state

    @property
    def lost(self) -> bool:
        return self.stale > self.config.grace

    @property
    def fresh(self) -> bool:
        """Whether the box is recent enough to act on -- an arrival, say."""
        return self.box is not None and self.stale <= self.config.every

    def current_box(self) -> Optional[Box]:
        """The last accepted box carried by the camera since, clipped to the frame."""
        if self.box is None:
            return None
        dx, dy = self._shift
        x0, y0, x1, y1 = self.box
        x0, x1 = max(0.0, x0 + dx), min(float(self.width), x1 + dx)
        y0, y1 = max(0.0, y0 + dy), min(float(self.height), y1 + dy)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def normalized_box(self) -> Optional[Box]:
        box = self.current_box()
        if box is None:
            return None
        x0, y0, x1, y1 = box
        return x0 / self.width, y0 / self.height, x1 / self.width, y1 / self.height

    def contains(self, point: Sequence[float], margin: float = 0.25) -> bool:
        """Whether `point` (pixels) falls inside the current box grown by `margin` of its size."""
        box = self.current_box()
        if box is None:
            return False
        x0, y0, x1, y1 = box
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        return bool(x0 - mx <= point[0] <= x1 + mx and y0 - my <= point[1] <= y1 + my)

    # ------------------------------------------------------------------ updating

    def advance(self, frame: np.ndarray, camera: Sequence[float] = (0.0, 0.0)) -> bool:
        """One step happened: the camera turned by `camera` = [pitch, yaw] degrees.

        MineRL's convention, which is Minecraft's: positive pitch looks down, positive yaw
        turns right, so the scene moves up and left. Returns whether this step produced an
        accepted re-segmentation.
        """
        pitch, yaw = float(camera[0]), float(camera[1])
        self._shift += (-yaw * self.ppd, -pitch * self.ppd)
        self._steps += 1
        self.stale += 1
        if self._steps % max(1, self.config.every):
            return False
        return self._resegment(frame)

    def _predicted_mask(self) -> np.ndarray:
        dx, dy = self._shift
        moved = cv2.warpAffine(self.mask.astype(np.uint8), np.float32([[1, 0, dx], [0, 1, dy]]),
                               (self.width, self.height), flags=cv2.INTER_NEAREST,
                               borderValue=0)
        return moved.astype(bool)

    def _resegment(self, frame: np.ndarray) -> bool:
        predicted = self._predicted_mask()
        if not predicted.any():
            self.rejections += 1                      # carried out of the frame entirely
            return False

        # The most interior point, not the box centre: an L-shaped building's centre can
        # be sky, and SAM-2 would segment the sky with total confidence.
        inside = cv2.distanceTransform(np.pad(predicted.astype(np.uint8), 1), cv2.DIST_L2, 3)
        _, _, _, (sx, sy) = cv2.minMaxLoc(inside[1:-1, 1:-1])
        segment = getattr(self.masker, "mask_fast", self.masker.mask)
        candidate = np.asarray(segment(frame, (float(sx), float(sy)))).astype(bool)

        area, expected = int(candidate.sum()), int(predicted.sum())
        union = int((candidate | predicted).sum())
        iou = (candidate & predicted).sum() / union if union else 0.0
        ratio = area / expected if expected else math.inf
        limit = self.config.max_area_ratio
        if area == 0 or iou < self.config.min_iou or not (1.0 / limit <= ratio <= limit):
            self.rejections += 1
            return False
        color = mean_lab(frame, candidate)
        if float(np.linalg.norm(color - self.color)) > self.config.max_color_distance:
            self.rejections += 1
            return False

        # A running reference, so light changing over a long approach is followed rather
        # than eventually refused; slow enough that a few bad updates cannot walk it away.
        self.color = 0.8 * self.color + 0.2 * color
        self.mask = candidate
        self.box = mask_box(candidate)
        self._shift[:] = 0.0
        self.stale = 0
        self.updates += 1
        return True
