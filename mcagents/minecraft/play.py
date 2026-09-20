"""MineStudio's play window, as something an agent can share with a person.

`PlayCallback` renders the game in a pyglet window and reads the keyboard and mouse. What
makes it useful here is a detail of how it decides whether to: `before_step` only reads the
keyboard when the action it is handed is a string or None, and passes **anything else
straight through to the env**. So no mode switch is needed to share a sim -- what you pass
to `sim.step()` *is* the switch:

    agent.human_step()      # sim.step("human")  -> PlayCallback reads WASD and the mouse
    agent.step()            # sim.step({...})    -> ROCKET-2 drives, keys ignored

This wraps that with the two things the rest of the project needs from the window:

`paint()`   registers an overlay. MineStudio calls every `extra_draw_call` from inside
            `_update_image`, *after* the frame has been upscaled from 640x360 to the window
            and *before* it is blitted -- so an overlay drawn there is crisp at display
            resolution rather than a magnified 640-wide rectangle. Painters are handed the
            display-resolution image and draw on it in place.

`pov_rect()` says where that image is on screen, which is what turns a gaze coordinate into
            a frame coordinate (mcagents/perception/selection.py). pyglet blits the pov at
            the top of the window with the info panel underneath, and reports window
            position and size in the same physical pixels an eye tracker reports gaze in --
            verified on this machine at 3200x2000 with window.scale == 2.0, where a cv2
            window would have reported logical pixels instead.

How big that rectangle is decides how well gaze selection works: the error is an angle, so
it costs half as many *frame* pixels on a window twice as wide. `$MCAGENTS_GUI_SCALE` sets
it (see scripts/setup/apply_patches.py and docs/gaze.md); 4 gives a 2560x1350 view.
"""
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from minestudio.simulator.callbacks import PlayCallback

#: What the window's info panel calls it when something other than the keyboard is driving.
#: PlayCallback shows `switch` as "Role:", and during a demo who holds the controls is the
#: one thing worth being able to read off the screen.
AGENT_ROLE = "rocket2"

#: An overlay: given the display-resolution RGB frame, draw on it in place.
Painter = Callable[[np.ndarray], None]


class PlayWindow(PlayCallback):
    """`PlayCallback` with a place to hang overlays and a way to find itself on screen."""

    def __init__(self, painters: Optional[List[Painter]] = None):
        self.painters: List[Painter] = list(painters or [])
        #: Where the mouse last was, in screen pixels -- None until it moves, and stale the
        #: whole time the mouse is captured. See `_track_pointer`.
        self.pointer: Optional[Tuple[float, float]] = None
        # agent_generator stays None: PlayCallback's own agent is the "press L to hand over
        # to a policy" demo, and the policy here arrives as goals instead.
        super().__init__(agent_generator=None, extra_draw_call=[self._paint])
        self._track_pointer()

    def _track_pointer(self) -> None:
        """Also record where the mouse is, not just how far it moved.

        MineStudio only wants the delta -- that is what turns into camera degrees -- so it
        throws the position away. The position is what a pointing device needs, and it is
        what stands in for an eye tracker before one is calibrated: it goes through the
        identical screen-to-frame mapping a gaze sample does, so the fallback exercises that
        mapping rather than quietly bypassing it.

        Only meaningful while the mouse is *not* captured (press `C`). Captured, it is locked
        to the window centre and driving the camera, which is also why it cannot be the
        pointing device during normal play, and why eyes are worth having.
        """
        gui = self.gui
        moved = gui._on_mouse_motion

        def on_mouse_motion(x, y, dx, dy):
            if not gui.capture_mouse:
                self.pointer = self._to_screen(x, y)
            moved(x, y, dx, dy)

        gui.window.on_mouse_motion = on_mouse_motion

    def _to_screen(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """A pyglet window coordinate as a screen one: origin to the top-left, y downward."""
        rect = self.pov_rect()
        if rect is None:
            return None
        constants = self.gui.constants
        height = constants.INFO_HEIGHT + constants.FRAME_HEIGHT
        return rect[0] + float(x), rect[1] + (height - float(y))

    def paint(self, painter: Painter) -> Painter:
        """Register an overlay, drawn on every frame in the order registered."""
        self.painters.append(painter)
        return painter

    def _paint(self, info: dict, **kwargs: Any) -> dict:
        image = info.get("pov")
        if image is None:
            return info
        for painter in self.painters:
            # An overlay is decoration. One that raises must not be what stops the game --
            # and a traceback per frame at 25 Hz buries every other line in the terminal.
            try:
                painter(image)
            except Exception as broken:
                self._painter_failed(painter, broken)
        return info

    def _painter_failed(self, painter: Painter, broken: Exception) -> None:
        self.painters.remove(painter)
        print(f"[play] an overlay raised {type(broken).__name__}: {broken} -- dropped it, "
              f"the game carries on", flush=True)

    # ------------------------------------------------------------------ geometry

    @property
    def pov_size(self) -> Tuple[int, int]:
        """(width, height) the 640x360 frame is displayed at. Painters draw in this space.

        Read off the GUI's own constants rather than the callback's copy: that is the object
        `_update_image` resizes against, so it is the one that cannot disagree with reality.
        """
        constants = self.gui.constants
        return constants.WINDOW_WIDTH, constants.FRAME_HEIGHT

    def pov_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """(x, y, width, height) of the game view in screen pixels, or None if there is no window.

        The pov occupies the top of the window: pyglet's y runs upward, and `_update_image`
        blits the texture at y=INFO_HEIGHT, which puts the info panel below it.
        """
        window = getattr(self.gui, "window", None)
        if window is None:
            return None
        try:
            x, y = window.get_location()
        except Exception:      # the window is gone, or the platform will not say where it is
            return None
        width, height = self.pov_size
        return int(x), int(y), int(width), int(height)

    # ------------------------------------------------------------------ who is driving

    def before_step(self, sim, action: Any) -> dict:
        """Label the info panel with whoever is about to drive, then behave exactly as upstream.

        A dict action means the policy is stepping; upstream leaves `switch` alone in that
        case, so the panel would go on claiming "human" for the whole of a goal. Setting it
        here is safe because `super()` only reads `switch` on the string/None path, and a
        later `human_step()` sets it straight back.
        """
        if action is not None and not isinstance(action, str):
            self.switch = AGENT_ROLE
        return super().before_step(sim, action)
