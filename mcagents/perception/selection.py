"""Pointing at things with your eyes: gaze -> a segmented object -> a red box.

ROCKET-2 takes a *mask* as its goal, and the person driving already knows which object they
mean, because they are looking at it. This turns the second fact into the first, fast enough
that the box feels attached to your eyes rather than chasing them.

Fast enough is the whole problem. One SAM-2 call is ~113 ms, which is four game frames -- a
box that lags a third of a second behind your eyes reads as broken. But almost all of that is
the image encoder, and the image is the thing that is barely changing:

    EmbedWorker   runs the encoder off the game loop, ~90 ms, continuously
    GazeSelector  decodes at the current gaze point, ~4 ms, once per frame

So the loop pays 4 ms and the encoder never blocks it. What the encoder's latency does buy
is *staleness*: the mask is always "the object at that point, as of up to ~90 ms ago". While
you are looking around to choose a target that is invisible. While the view is moving it is
not -- the object under your gaze now is not the object that was there then -- so a decode
is skipped once the view has changed too much since the embedding was taken, and the
previous box is held and dimmed instead of being confidently wrong.

"Changed too much" is measured by *comparing the frames*, on a 64x36 thumbnail, and not by
adding up camera degrees. Degrees were the obvious choice and are wrong twice over. They
miss walking entirely -- stride forward with WASD and the view transforms without the camera
rotating a degree -- and MineStudio's own numbers are not trustworthy while the mouse is
captured: measured here, an untouched mouse reported 133 degrees over 40 idle steps, in
bursts, because pointer warps arrive as enormous motion deltas. A thumbnail difference has
neither problem, costs ~0.1 ms, and measures the thing actually being asked about.

The alternative -- carrying the box along by camera motion the way [tracking.py](tracking.py)
does for goals -- would make the drawn box disagree with the mask underneath it, and that
mask is what ROCKET-2 is handed.

The state machine is "instant hover, sticky lock":

    hover    the box follows your eyes object to object, thin and red
    lock     after `lock_ms` on one object it commits: solid, thicker
    hold     and stays there while you look anywhere inside it plus a margin
    release  gaze leaves it, goes stale, or leaves the window

The stickiness is not cosmetic. A good webcam tracker has excellent *precision* (~10 px of
jitter) and mediocre *accuracy* (~100 px of stable bias), so without hysteresis the box
flickers between a tree and the house behind it; with it, the bias has to be overcome
deliberately before the selection moves.
"""
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from mcagents import gaze
from mcagents.perception.masking import Embedding
from mcagents.perception.tracking import Box, mask_box

#: (x, y, width, height) of the game view in screen pixels.
Rect = Tuple[int, int, int, int]


#: Size the staleness comparison is done at. Small enough to be free, large enough that a
#: step forward or a few degrees of turn move it well clear of the noise floor.
THUMBNAIL = (64, 36)


def thumbnail(frame: np.ndarray) -> np.ndarray:
    """A frame reduced to the only question asked of it: has the view changed since?"""
    small = cv2.resize(frame, THUMBNAIL, interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = small.mean(axis=2)
    return small.astype(np.float32)


def difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two thumbnails, 0-255."""
    return float(np.abs(a - b).mean())


class GazeSource(Protocol):
    """Anywhere a look can come from: a tracker, a mouse, a test."""

    def sample(self) -> Optional[gaze.GazeSample]:
        ...


class ChannelSource:
    """The real thing: whatever `mcagents.cli.gaze` last published."""

    def __init__(self, path: str = gaze.DEFAULT_PATH, max_age: float = gaze.DEFAULT_MAX_AGE):
        self.path = path
        self.max_age = max_age

    def sample(self) -> Optional[gaze.GazeSample]:
        return gaze.read_latest(self.path, self.max_age, allow_invalid=True)


class PointerSource:
    """The mouse, for demoing and testing without calibrating a tracker.

    Goes through the identical screen-to-frame mapping a real sample does, so what it
    exercises is the real path. Needs the mouse released from camera capture (`C`).
    """

    def __init__(self, window):
        self.window = window

    def sample(self) -> Optional[gaze.GazeSample]:
        pointer = getattr(self.window, "pointer", None)
        if pointer is None:
            return None
        return gaze.GazeSample(x=pointer[0], y=pointer[1], t=time.time())


class ScreenMap:
    """Screen pixels -> frame pixels, through wherever the game is being shown.

    The window rect is read live on every sample rather than cached, so moving or resizing
    the window mid-session needs no notification and cannot go stale -- and a window that has
    gone away simply stops producing points.
    """

    def __init__(self, rect_of: Callable[[], Optional[Rect]]):
        self.rect_of = rect_of

    def to_frame(self, sample: gaze.GazeSample,
                 shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """`sample` as [x, y] in a `shape` = (height, width) frame, or None if it missed it.

        None means the eyes are off the game view -- the info panel, another window, the
        wall behind the laptop -- which is a different thing from looking at nothing in the
        game, and the caller treats it as "no selection" rather than clamping to an edge.
        """
        rect = self.rect_of()
        if rect is None:
            return None
        left, top, width, height = rect
        if width <= 0 or height <= 0:
            return None
        u = (float(sample.x) - left) / width
        v = (float(sample.y) - top) / height
        if not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
            return None
        rows, columns = shape
        # Scaled independently on each axis: MineStudio stretches 640x360 to fill the window
        # rather than letterboxing it, so the two ratios are not the same number.
        return u * columns, v * rows


class EmbedWorker:
    """Runs SAM-2's image encoder on a thread, so the game loop never waits ~90 ms for it.

    One slot in each direction, newest wins, for the reason [frames.py](../frames.py) gives:
    a frame is worthless once a newer one exists, and a queue here would only guarantee that
    what comes out is old. The frame is copied on the way in because the sim hands back views
    of a buffer it is about to overwrite.
    """

    def __init__(self, masker, report=None):
        self.masker = masker
        self.report = report or (lambda message: print(message, flush=True))
        self.embeds = 0
        self._pending: Optional[np.ndarray] = None
        self._latest: Optional[Tuple[np.ndarray, Embedding, np.ndarray]] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="sam-embed", daemon=True)
        self._thread.start()

    def offer(self, frame: Optional[np.ndarray]) -> None:
        """Hand the encoder a frame to work on next, replacing any it has not started."""
        if frame is None or self._closed:
            return
        with self._lock:
            self._pending = np.ascontiguousarray(frame)
        self._wake.set()

    def latest(self) -> Optional[Tuple[np.ndarray, Embedding, np.ndarray]]:
        """The newest (frame, embedding, thumbnail), or None until the first one lands.

        The thumbnail is computed once, here, rather than on every update: it is the
        reference half of the comparison, and it does not change until the next embedding.
        """
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while not self._closed:
            self._wake.wait(0.1)
            self._wake.clear()
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            try:
                embedding = self.masker.embed(frame)
            except Exception as broken:
                # A dead encoder means no selection, not a dead game.
                self._closed = True
                self.report(f"[gaze] the SAM-2 encoder failed, selection is off "
                            f"({type(broken).__name__}: {broken})")
                return
            with self._lock:
                self._latest = (frame, embedding, thumbnail(frame))
                self.embeds += 1

    def close(self) -> None:
        self._closed = True
        self._wake.set()


@dataclass
class Selection:
    """One object, picked out by where somebody was looking."""
    #: Where the gaze landed, in frame pixels.
    point: Tuple[float, float]
    #: The object's tight box in frame pixels.
    box: Box
    #: Its mask, at frame resolution.
    mask: np.ndarray
    #: The frame the mask was computed on -- which is *not* necessarily the current one, and
    #: has to travel with the mask, because it is the cross-view image ROCKET-2 is
    #: conditioned on and the two only mean anything together.
    frame: np.ndarray
    #: Whether it has been looked at long enough to commit to.
    locked: bool = False
    #: False while the view has turned too far since the embedding for the mask to be
    #: trusted -- drawn dimmed, and refused as a goal.
    fresh: bool = True
    t: float = field(default_factory=time.monotonic)

    @property
    def centre(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def normalized_box(self) -> Box:
        """The box as fractions of the frame -- the form `goal.status` already speaks."""
        rows, columns = self.mask.shape[:2]
        x0, y0, x1, y1 = self.box
        return x0 / columns, y0 / rows, x1 / columns, y1 / rows

    def normalized_centre(self) -> Tuple[float, float]:
        rows, columns = self.mask.shape[:2]
        x, y = self.centre
        return x / columns, y / rows

    def contains(self, point: Sequence[float], margin: float = 0.25) -> bool:
        """Whether `point` falls inside the box grown by `margin` of its size."""
        x0, y0, x1, y1 = self.box
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        return bool(x0 - mx <= point[0] <= x1 + mx and y0 - my <= point[1] <= y1 + my)

    def payload(self) -> dict:
        """What goes up the wire to the remote model: the centre of the box, and the box."""
        return {"point": [round(v, 4) for v in self.normalized_centre()],
                "box": [round(v, 4) for v in self.normalized_box()],
                "locked": bool(self.locked)}


@dataclass
class SelectorConfig:
    #: Milliseconds of looking at one object before the box commits to it. Short enough to
    #: feel immediate, long enough that a saccade crossing an object does not claim it.
    lock_ms: float = 150.0
    #: Screen pixels the smoothed gaze must move before anything is re-decoded. Below this
    #: the eyes are fixating and the answer cannot have changed.
    deadzone: float = 8.0
    #: Weight of each new sample in the smoothed point. 1.0 is no smoothing.
    smoothing: float = 0.45
    #: A mask smaller than this is nothing worth selecting -- sky, a gap between blocks, the
    #: sliver between two leaves. A point prompt always returns *something*, so without a
    #: floor the box spends its time confidently around patches of sky. Measured against a
    #: real spawn view: Minecraft's own crosshair, dead centre, segments to ~35 px, while a
    #: building a street away is ~320 px, so the floor sits between them and nearer the
    #: crosshair -- refusing a distant target is the more annoying of the two mistakes.
    min_area: int = 150
    #: A mask larger than this fraction of the frame is the ground or the sky rather than an
    #: object, and equally not what anyone meant.
    max_area_fraction: float = 0.7
    #: Mask IoU for "still the same object" when deciding whether a hover has settled.
    same_iou: float = 0.5
    #: How far outside the locked box the gaze may stray before the lock stops being held,
    #: as a fraction of the box's own size.
    hold_margin: float = 0.25
    #: Milliseconds a committed lock survives with nothing supporting it -- a blink, a
    #: glance at the sky, the tracker losing the face for a frame. Looking properly at
    #: another object still switches immediately; this only covers looking at *nothing*,
    #: which is the shape every tracker dropout has.
    release_ms: float = 250.0
    #: How far the view may change after an embedding before its masks stop being believed,
    #: as the mean absolute difference between 64x36 thumbnails, 0-255. Measured on a live
    #: session: a still view sits near 0, and any real movement is well above this.
    max_change: float = 6.0

    @classmethod
    def from_env(cls) -> "SelectorConfig":
        """The knobs worth reaching for mid-session, as $MCAGENTS_GAZE_*.

        The area bounds are the two that actually get touched: what counts as "an object"
        depends on how far away you play and how big the things you point at are, and the
        symptom of both being wrong is the same silent one -- no box.
        """
        return cls(
            lock_ms=float(os.environ.get("MCAGENTS_GAZE_LOCK_MS", cls.lock_ms)),
            deadzone=float(os.environ.get("MCAGENTS_GAZE_DEADZONE", cls.deadzone)),
            smoothing=float(os.environ.get("MCAGENTS_GAZE_SMOOTHING", cls.smoothing)),
            min_area=int(os.environ.get("MCAGENTS_GAZE_MIN_AREA", cls.min_area)),
            max_area_fraction=float(os.environ.get("MCAGENTS_GAZE_MAX_AREA",
                                                   cls.max_area_fraction)),
            same_iou=float(os.environ.get("MCAGENTS_GAZE_SAME_IOU", cls.same_iou)),
            hold_margin=float(os.environ.get("MCAGENTS_GAZE_HOLD_MARGIN", cls.hold_margin)),
            release_ms=float(os.environ.get("MCAGENTS_GAZE_RELEASE_MS", cls.release_ms)),
            max_change=float(os.environ.get("MCAGENTS_GAZE_MAX_CHANGE", cls.max_change)),
        )


#: BGR-free: the painter draws on the RGB frame MineStudio is about to blit.
HOVER_COLOUR = (255, 60, 60)
LOCK_COLOUR = (255, 0, 0)
STALE_COLOUR = (150, 70, 70)
HAND_COLOUR = (40, 220, 80)
HAND_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                    (5, 9), (9, 13), (13, 17))


class GazeSelector:
    """Gaze in, a segmented object out, and a red box on the screen. See the module docstring.

        selector.update(frame, camera)      # once per step, from the agent's _after_step
        selector.selection                  # what is boxed right now, or None
        window.paint(selector.draw)         # the overlay
    """

    def __init__(self, masker, screen: ScreenMap, source: GazeSource,
                 config: Optional[SelectorConfig] = None, report=None):
        self.config = config or SelectorConfig()
        self.screen = screen
        self.source = source
        self.report = report or (lambda message: print(message, flush=True))
        self.worker = EmbedWorker(masker, report=self.report)

        self.selection: Optional[Selection] = None
        #: Smoothed gaze in screen pixels, and the frame point it last mapped to.
        self._smoothed: Optional[Tuple[float, float]] = None
        self._decoded_at: Optional[Tuple[float, float]] = None
        #: When the object currently hovered was first seen, for the lock delay.
        self._since: Optional[float] = None
        #: The current view, reduced for comparison against the embedding's own.
        self._thumb: Optional[np.ndarray] = None
        #: Which embedding the standing selection was decoded against.
        self._embeds_seen = -1
        #: (height, width) of the last frame seen, for scaling the deadzone.
        self._shape: Tuple[int, int] = (360, 640)
        #: When the gaze stopped supporting a committed lock, for `release_ms`.
        self._unsupported: Optional[float] = None
        self.decodes = 0
        self.decode_ms = 0.0
        #: Why there is nothing selected right now. "No box on screen" is the one failure
        #: this has, and every cause of it -- no tracker, eyes off the window, the encoder
        #: not warm, a mask refused as sky or as the whole ground -- looks identical from
        #: the outside. Reported when it changes, so it is one line rather than a stream.
        self.why = "starting up"
        self._said: Optional[str] = None
        self._hand_landmarks = None

    # ------------------------------------------------------------------ the loop

    def update(self, frame: Optional[np.ndarray]) -> None:
        """One step happened, and this is what the view looks like now."""
        if frame is None:
            return
        self._shape = frame.shape[:2]
        self._thumb = thumbnail(frame)
        self.worker.offer(frame)

        sample = self.source.sample()
        if sample is None:
            self._hand_landmarks = None
            self._note("no fresh sample -- is ./scripts/gaze.sh running and calibrated?")
            return self._relax()

        self._hand_landmarks = sample.landmarks
        if not sample.valid:
            self._note("hand is open -- pinch to select")
            return self._relax()

        self._smooth(sample)
        point = self.screen.to_frame(
            gaze.GazeSample(x=self._smoothed[0], y=self._smoothed[1]), self._shape)
        if point is None:
            # The eyes are off the game view -- the info panel, another window, the wall.
            self._note("looking outside the game view")
            return self._relax()

        if self._should_decode(point):
            self._decode(point)
        # Always, decode or not. Committing is a function of *time spent looking*, and
        # hanging it off a decode meant the steadiest possible gaze -- one that never moved
        # far enough to be worth re-decoding -- was the one case that could never commit.
        self._promote()

    @property
    def change(self) -> float:
        """How far the view has moved since the frame the current embedding was built from."""
        latest = self.worker.latest()
        if latest is None or self._thumb is None:
            return 0.0
        return difference(self._thumb, latest[2])

    def _smooth(self, sample: gaze.GazeSample) -> None:
        weight = self.config.smoothing
        if self._smoothed is None:
            self._smoothed = (sample.x, sample.y)
            return
        x, y = self._smoothed
        self._smoothed = (x + weight * (sample.x - x), y + weight * (sample.y - y))

    def _should_decode(self, point: Tuple[float, float]) -> bool:
        """Whether this sample is worth ~4 ms of GPU, and whether the answer could be trusted."""
        if self.worker.latest() is None:
            self._note("the SAM-2 encoder has not produced a frame yet")
            return False
        if self.change > self.config.max_change:
            # The view has turned since the frame the embedding was built from, so whatever
            # is under this point *now* is not what was there then. Hold what we had, dimmed,
            # rather than box the wrong object with total confidence.
            self._note(f"the view has moved on since the last encode "
                       f"({self.change:.1f} > {self.config.max_change:g})")
            self._go_stale()
            return False
        if self._decoded_at is None or self.selection is None or not self.selection.fresh:
            return True
        if self.worker.embeds != self._embeds_seen:
            # The world has moved on under a still gaze -- a mob walked past, the light
            # changed, the agent was driving. The box should follow the pixels, not just the
            # eyes, so a new embedding is reason enough on its own.
            return True
        # The deadzone is quoted in screen pixels and compared in frame pixels: the same eye
        # movement is worth fewer frame pixels the larger the view is drawn.
        rect = self.screen.rect_of()
        scale = self._shape[1] / rect[2] if rect and rect[2] else 1.0
        moved = float(np.hypot(point[0] - self._decoded_at[0], point[1] - self._decoded_at[1]))
        return moved >= self.config.deadzone * scale

    def _decode(self, point: Tuple[float, float]) -> None:
        latest = self.worker.latest()
        if latest is None:
            return
        frame, embedding, _reference = latest
        started = time.perf_counter()
        try:
            mask = self.worker.masker.mask_at(embedding, point)
        except Exception as broken:
            self.report(f"[gaze] a decode failed, selection is off "
                        f"({type(broken).__name__}: {broken})")
            self.worker.close()
            return self._release()
        self.decode_ms = (time.perf_counter() - started) * 1000.0
        self.decodes += 1
        self._decoded_at = point
        self._embeds_seen = self.worker.embeds
        self._adopt(point, mask, frame)

    # ------------------------------------------------------------------ the state machine

    def _adopt(self, point: Tuple[float, float], mask: np.ndarray, frame: np.ndarray) -> None:
        """Turn a freshly decoded mask into hover, a lock, or nothing at all."""
        held = self.selection
        area, size = int(mask.sum()), mask.size
        box = mask_box(mask) if self._plausible(mask) else None
        if box is None:
            if area < self.config.min_area:
                self._note(f"what you are looking at is {area}px, under the "
                           f"{self.config.min_area}px floor -- sky, or a gap "
                           f"($MCAGENTS_GAZE_MIN_AREA lowers it)")
            else:
                self._note(f"what you are looking at is {100 * area / size:.0f}% of the "
                           f"frame -- ground or sky rather than an object "
                           f"($MCAGENTS_GAZE_MAX_AREA raises the ceiling)")
            # Sky, or a sliver between two leaves, or the whole ground: a point prompt always
            # answers, so "there is nothing here worth selecting" has to be decided here.
            # A committed lock rides it out -- eyes crossing sky on the way back to the tree
            # is exactly what `release_ms` is for.
            return self._relax()

        # A lock holds while you look anywhere inside it, even where SAM-2 would now answer
        # with the leaf rather than the whole tree. Letting go takes looking elsewhere.
        if held is not None and held.locked and held.contains(point, self.config.hold_margin):
            self._unsupported = None
            self.why, self._said = "", None
            self.selection = Selection(point=point, box=box, mask=mask, frame=frame,
                                       locked=True, fresh=True)
            return

        # Anything else is a new object, including one arriving while a lock is held: looking
        # properly at something else switches now, and the newcomer earns its own lock from
        # scratch. `_since` is when the object currently being looked at was first seen, so
        # it survives re-decoding the same object and resets the moment it becomes a
        # different one.
        self._unsupported = None
        self.why, self._said = "", None
        if held is None or held.locked or not self._same(held.mask, mask):
            self._since = time.monotonic()
        self.selection = Selection(point=point, box=box, mask=mask, frame=frame,
                                   locked=False, fresh=True)

    def _promote(self) -> None:
        """A hover that has stayed on one object for `lock_ms` becomes a lock."""
        held = self.selection
        if held is None or held.locked or self._since is None:
            return
        if (time.monotonic() - self._since) * 1000.0 >= self.config.lock_ms:
            self.selection = replace(held, locked=True)

    def _plausible(self, mask: np.ndarray) -> bool:
        area = int(mask.sum())
        if area < self.config.min_area:
            return False
        return area <= self.config.max_area_fraction * mask.size

    def _same(self, before: np.ndarray, after: np.ndarray) -> bool:
        union = int((before | after).sum())
        return bool(union and (before & after).sum() / union >= self.config.same_iou)

    def _go_stale(self) -> None:
        if self.selection is not None and self.selection.fresh:
            self.selection = replace(self.selection, fresh=False)

    def _note(self, why: str) -> None:
        self.why = why
        if why != self._said:
            self._said = why
            self.report(f"[gaze] nothing selected: {why}")

    def _relax(self) -> None:
        """Nothing is supporting the selection this frame. Give a committed one a moment.

        The difference between this and `_release` is the difference between a tracker
        blinking and a person looking away. Every webcam tracker drops samples, and a lock
        that evaporates on the first dropped one is a lock nobody can rely on -- but one that
        never lets go is worse, so the grace is `release_ms` and then it is gone.
        """
        selection = self.selection
        if selection is None or not selection.locked:
            return self._release()
        if self._unsupported is None:
            self._unsupported = time.monotonic()
        elif (time.monotonic() - self._unsupported) * 1000.0 >= self.config.release_ms:
            self._release()

    def _release(self) -> None:
        self.selection = None
        self._since = None
        self._decoded_at = None
        self._unsupported = None

    # ------------------------------------------------------------------ the overlay

    def draw(self, image: np.ndarray) -> None:
        """Paint the box onto the display-resolution frame. Registered with `PlayWindow.paint`.

        The frame arrives already upscaled to the window, so the box is scaled to match and
        drawn crisp rather than as a magnified 640-wide rectangle.
        """
        selection = self.selection
        self._draw_hand(image)
        if selection is None:
            return
        rows, columns = selection.mask.shape[:2]
        height, width = image.shape[:2]
        sx, sy = width / columns, height / rows
        x0, y0, x1, y1 = selection.box
        corners = (int(x0 * sx), int(y0 * sy)), (int(x1 * sx) - 1, int(y1 * sy) - 1)

        if not selection.fresh:
            colour, thickness = STALE_COLOUR, 2
        elif selection.locked:
            colour, thickness = LOCK_COLOUR, 4
        else:
            colour, thickness = HOVER_COLOUR, 2
        cv2.rectangle(image, corners[0], corners[1], colour, thickness)

    def _draw_hand(self, image: np.ndarray) -> None:
        """Draw the tracked hand in the game viewport's display coordinates."""
        landmarks = self._hand_landmarks
        rect = self.screen.rect_of()
        if not landmarks or rect is None or len(landmarks) < 21:
            return
        left, top, width, height = rect
        image_height, image_width = image.shape[:2]

        def point(index):
            x, y = landmarks[index]
            return (int((x - left) * image_width / width),
                    int((y - top) * image_height / height))

        for first, second in HAND_CONNECTIONS:
            cv2.line(image, point(first), point(second), HAND_COLOUR, 2)
        for index in range(21):
            cv2.circle(image, point(index), 4, HAND_COLOUR, -1)

    def close(self) -> None:
        self.worker.close()
