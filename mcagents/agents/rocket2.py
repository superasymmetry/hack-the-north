"""ROCKET-2 as a three-field controller: point, interaction, stop.

Text-conditioned policies cannot say *which* object you mean. ROCKET-2 (arXiv:2503.02505)
replaces the text channel with a pointed goal -- a segmentation mask over one frame plus an
interaction type -- and scores ~0.94 on the Minecraft Interaction benchmark where a text
channel scores ~0.19. That maps onto three fields, which is exactly what an LLM planner can
emit as JSON:

    agent.run(point=[420, 190], interaction="Mine", stop={"item": "oak_log", "count": 3})

    point       [x, y] in the current frame (info['pov'], render_size, default 640x360).
                SAM-2 turns it into the mask. normalized=True takes [0..1] fractions instead,
                which is usually easier for a model to produce.
    interaction Hunt | Mine | Use | Interact | Craft | Switch | Approach (or None).
    stop        see mcagents.agents.base for the shared vocabulary.

The mask is captured ONCE per goal and then held fixed: ROCKET-2 tracks the target itself
across viewpoint changes (the "cross-view goal alignment" of the title, and why it runs 3-6x
faster than ROCKET-1, which re-segmented every 3-10 steps). Re-pointing means calling
set_goal() again, i.e. one SAM-2 call per goal rather than one per step.

What ROCKET-2 does *not* do is promise the thing it is walking toward is still the thing that
was pointed at. So an Approach goal -- or any goal with an `arrive` stop -- also carries an
`InstanceLock` (mcagents/perception/tracking.py): the pointed instance, followed by overlap
and camera motion, re-segmented every few steps. It is what `arrive` measures, and when it is
lost for longer than a short grace the goal ends as "lost" rather than following the policy
onto a lookalike. Approach with no `arrive` of its own gets one added.

Arrival itself is a world measurement wherever it can be. The pointed pixel is cast through
the voxel grid around the player (mcagents/minecraft/ranging.py) to the block it lands on,
and from then on the goal knows its target's address: `distance` is `|player_pos - anchor|`
horizontally, which turning cannot change and walking past cannot fake. The grid only
reaches ~7 blocks, so the cast is retried each step until the target comes inside it, and a
target that never does falls back to `width` -- the fraction of the frame its box fills.

Measured on an RTX 5060 laptop (8GB), model forward only:

    cfg_coef = 0   33 FPS   950 MB VRAM     <- default; comfortably above Minecraft's 20 Hz
    cfg_coef = 1   16 FPS   950 MB VRAM     <- two forward passes per step, below 20 Hz

See docs/rocket2.md for the wiring and mcagents/cli/rocket2.py for a runnable example.
"""
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

# Before anything that pulls in minestudio (the vendored policy below does), or the first
# OpenCV window this process opens will hang. See mcagents.gui.
from mcagents import gui
from mcagents.agents.base import Agent, Goal, GoalResult, StopSpec, uses_arrive
from mcagents.frames import FramePublisher
from mcagents.minecraft import ranging
from mcagents.perception.masking import Masker, PointMasker
from mcagents.perception.tracking import InstanceLock, LockConfig, mask_box
from mcagents.vendor.rocket2 import CFGWrapper, CrossViewRocket

#: Interaction ids, copied verbatim from ROCKET-2's own demo (launch.py:SEGMENT_MAPPING).
#: Two things look like typos but are not: "Use" and "Interact" share id 3 upstream, and id 1
#: is unused.
INTERACTIONS: Dict[str, int] = {
    "Hunt": 0,
    "Mine": 2,
    "Use": 3,
    "Interact": 3,
    "Craft": 4,
    "Switch": 5,
    "Approach": 6,
    "None": -1,
}

#: ROCKET-2's ViT input size. A sim built at any other obs_size cannot drive it.
OBS_SIZE = (224, 224)


def interaction_id(name: Optional[str]) -> int:
    """Map an interaction name to ROCKET-2's embedding id, case-insensitively."""
    for known, value in INTERACTIONS.items():
        if known.lower() == str(name if name is not None else "None").lower():
            return value
    raise ValueError(f"unknown interaction {name!r}; expected one of {sorted(INTERACTIONS)}")


def camera_degrees(action: Any) -> Tuple[float, float]:
    """[pitch, yaw] in degrees out of an env-format action; (0, 0) if it has none to give.

    Only the env format carries degrees -- the agent format is quantized bins -- so a sim
    driven with agent actions tracks on re-segmentation alone.
    """
    try:
        camera = np.asarray(action["camera"], dtype=np.float64).reshape(-1)
        return float(camera[0]), float(camera[1])
    except (KeyError, TypeError, IndexError, ValueError):
        return 0.0, 0.0


@dataclass
class Rocket2Config:
    #: The 1x checkpoint (22w steps). "1.5x" (phython96/ROCKET-2-1.5x-17w) is the wider
    #: model: better, but ~1.5x the compute, which an 8 GB laptop GPU cannot afford at 20 Hz.
    checkpoint: str = "phython96/ROCKET-2-1x-22w"
    device: str = "cuda"
    #: Classifier-free guidance strength. 0 runs a single forward pass; anything > 0 runs the
    #: conditional *and* an unconditional pass and extrapolates between them, buying goal
    #: adherence at the cost of dropping below Minecraft's 20 Hz. ROCKET-2's own demo
    #: defaults to 1.0, but it is not driving a live game loop.
    cfg_coef: float = 0.0
    #: Hard ceiling per goal, so a stop condition that never fires cannot hang a run.
    max_steps: int = 600
    #: Open a window showing the mask, ROCKET-2's own target guess and its visibility
    #: estimate. Needs a DISPLAY; degrades to a no-op without one.
    preview: bool = False
    #: Seconds between publishing the agent's own view for the voice client to forward to
    #: the remote model -- see mcagents/frames.py. Shorter than the client's send interval
    #: on purpose, so the frame it picks up is fresh. 0 disables publishing entirely.
    frame_interval: float = 1.0
    #: The arrival Approach gets when its stop names none: the locked target's box is this
    #: fraction of the frame wide. 0.6 is about 7 blocks from a 10-wide building and about
    #: 2 from a 3-wide tree, i.e. "in front of it" without walking into it. Only a fallback
    #: now -- see `arrive_distance`, which is preferred wherever the target has an address.
    arrive_width: float = 0.6
    #: Blocks from the target's world position that count as arrived, when the sim carries
    #: voxels and the target could be anchored to a block (mcagents/minecraft/ranging.py).
    #: Measured horizontally, so it means "standing next to it" whatever the target's height.
    #: 0 falls back to `arrive_width` everywhere.
    arrive_distance: float = 3.0
    #: Steps between re-segmentations of the locked target (see tracking.py for the cost).
    track_every: int = 3
    #: Steps the lock may go unconfirmed before the goal ends as "lost". ~1.5 s at ~14 Hz.
    lock_grace: int = 20
    #: Vertical field of view of the game camera, for turning camera degrees into pixels.
    fov: float = 70.0
    #: Keep following the gaze while a goal is running. Off by default: ROCKET-2's forward
    #: is ~30 ms of every step and SAM-2's encoder is another ~90 ms, and the two together
    #: drag the loop under the rate a person can play at. You select, then you speak.
    gaze_during_goals: bool = False
    #: How long a committed selection stays usable after nothing is supporting it any more --
    #: the hand opened, the eyes left. Without it a spoken pointed task cannot work at all:
    #: the selector drops a lock 250 ms after the pinch is released (`release_ms`), and the
    #: round trip from "mine that" to a goal coming back down the tunnel is the ASR final,
    #: the client's join window, the wire, the remote model and the spool poll -- seconds,
    #: not milliseconds. So "that" has to outlive the pointing by about as long as it takes
    #: to say it and be answered. 0 turns the hold off and makes a goal need a live pinch.
    selection_hold_ms: float = 5000.0

    @classmethod
    def from_env(cls) -> "Rocket2Config":
        return cls(
            checkpoint=os.environ.get("ROCKET2_CKPT", cls.checkpoint),
            device=os.environ.get("MCAGENTS_DEVICE", cls.device),
            cfg_coef=float(os.environ.get("ROCKET2_CFG", cls.cfg_coef)),
            max_steps=int(os.environ.get("ROCKET2_MAX_STEPS", cls.max_steps)),
            preview=os.environ.get("ROCKET2_PREVIEW", "1") != "0",
            frame_interval=float(os.environ.get("MCAGENTS_FRAME_INTERVAL",
                                                cls.frame_interval)),
            arrive_width=float(os.environ.get("ROCKET2_ARRIVE_WIDTH", cls.arrive_width)),
            arrive_distance=float(os.environ.get("ROCKET2_ARRIVE_DISTANCE",
                                                 cls.arrive_distance)),
            track_every=int(os.environ.get("ROCKET2_TRACK_EVERY", cls.track_every)),
            lock_grace=int(os.environ.get("ROCKET2_LOCK_GRACE", cls.lock_grace)),
            fov=float(os.environ.get("MCAGENTS_FOV", cls.fov)),
            gaze_during_goals=os.environ.get("MCAGENTS_GAZE_DURING_GOALS", "0") != "0",
            selection_hold_ms=float(os.environ.get("MCAGENTS_SELECTION_HOLD_MS",
                                                   cls.selection_hold_ms)),
        )


@dataclass
class Subgoal(Goal):
    """One (point, interaction, stop) triple, plus the frame and mask it was pinned to."""
    point: Tuple[float, float] = (0.0, 0.0)
    interaction: str = "Approach"
    obj_id: int = -1
    mask: Optional[np.ndarray] = None             # bool, at pov resolution
    frame: Optional[np.ndarray] = None            # RGB, at pov resolution
    #: The pointed instance, followed frame to frame. None for goals that do not lock.
    lock: Optional[InstanceLock] = None
    #: The target's world position as [x, y, z], once a cast has found one. Stays None for a
    #: target that has never been inside the voxel grid, and never changes once it is set:
    #: that is the whole point of it. See mcagents/minecraft/ranging.py.
    anchor: Optional[np.ndarray] = None

    def describe(self) -> str:
        anchor = (f" at ({self.anchor[0]:.1f},{self.anchor[1]:.1f},{self.anchor[2]:.1f})"
                  if self.anchor is not None else "")
        return (f"{self.interaction}@({self.point[0]:.0f},{self.point[1]:.0f}) "
                f"stop={self.stop!r} mask={int(self.mask.sum()) if self.mask is not None else 0}px"
                f"{' locked' if self.lock is not None else ''}{anchor}")


@dataclass
class Rocket2Result(GoalResult):
    #: ROCKET-2's own P(target is in view) at the end -- the cheapest signal there is for
    #: "this goal has gone off the rails, re-point me".
    visibility: float = 0.0

    def __str__(self) -> str:
        return f"{super().__str__()}, visibility {self.visibility:.2f}"


class Rocket2Agent(Agent):
    """Drives a MinecraftSim with ROCKET-2, one pointed goal at a time.

    The sim must be built with obs_size=(224, 224) and a PrevActionCallback (the checkpoint
    was trained with use_prev_action=True and reads obs['env_prev_action']); both are checked
    at construction rather than left to fail as a shape error 200 steps in.

        result = agent.run(point=[420, 190], interaction="Mine", stop={"item": "log", "count": 3})

        agent.set_goal(point=[420, 190], interaction="Mine", stop=200)
        while agent.busy:                 # interleave with your planner
            agent.step()
        result = agent.result
    """

    log_tag = "rocket2"
    result_class = Rocket2Result
    supports_arrive = True

    def __init__(self, sim, config: Optional[Rocket2Config] = None,
                 masker: Optional[Masker] = None, verbose: bool = True, selector=None):
        self.config = config or Rocket2Config()
        self._check_sim(sim)
        super().__init__(sim, max_steps=self.config.max_steps, verbose=verbose)

        model = CrossViewRocket.from_pretrained(self.config.checkpoint).to(self.config.device).eval()
        # CFGWrapper always pays for two forward passes, even at k=0 where the second one
        # cancels out of the logit arithmetic entirely -- so at k=0 talk to the policy
        # directly. Both expose the same get_action/initial_state pair.
        self.model = model
        self.policy = CFGWrapper(model, k=self.config.cfg_coef) if self.config.cfg_coef > 0 else model
        self.masker: Masker = masker if masker is not None else PointMasker()

        self.state = self.policy.initial_state()
        self.visibility = 0.0
        self.predicted_point: Optional[Tuple[float, float]] = None
        #: [pitch, yaw] degrees of the last action sent, which is how the lock follows the view.
        self._camera = (0.0, 0.0)
        self.preview = self.config.preview
        # The voice client is a separate process with the socket and no pixels; this is the
        # only place in the system that has both the sim and a frame worth looking at.
        self.publisher = FramePublisher(self.config.frame_interval,
                                        report=lambda message: self._log(message))
        #: Where the person is looking, if anything is watching -- a
        #: mcagents.perception.selection.GazeSelector, built by whoever owns the window,
        #: since that is what knows where on screen the game is. None when nothing is.
        self.selector = selector
        #: The last committed selection and when it was last seen. See `held_selection`.
        self._held: Optional[Tuple[Any, float]] = None

    @staticmethod
    def _check_sim(sim) -> None:
        if tuple(sim.obs_size) != OBS_SIZE:
            raise ValueError(
                f"ROCKET-2 expects {OBS_SIZE[0]}x{OBS_SIZE[1]} observations, this sim has "
                f"obs_size={sim.obs_size}. Build it with MinecraftSim(obs_size={OBS_SIZE}, ...)."
            )
        if not any(type(cb).__name__ == "PrevActionCallback" for cb in sim.callbacks):
            raise ValueError(
                "The ROCKET-2 checkpoint was trained with use_prev_action=True and reads "
                "obs['env_prev_action'], which only exists if the sim has a PrevActionCallback."
            )

    # ------------------------------------------------------------------ goals

    def set_goal(self, point: Sequence[float], interaction: str, stop: StopSpec = None,
                 normalized: bool = False, keep_memory: bool = False,
                 mask: Optional[np.ndarray] = None,
                 frame: Optional[np.ndarray] = None) -> Subgoal:
        """Pin a new goal: segment `point` on the current frame, under `interaction`, until `stop`.

        :param normalized: interpret `point` as [0..1] fractions of width/height instead of
            pixels -- what you want when an LLM or a VLM produced the coordinates.
        :param keep_memory: by default the recurrent state is cleared, because it holds the
            previous target and carrying it into a new goal makes the agent hesitate
            (ROCKET-2's demo exposes the same thing as its "Clear Memory" button).
        :param mask: a segmentation somebody already has, instead of running SAM-2 again.
        :param frame: the frame that `mask` was computed on. Required with it, and not the
            same thing as the current frame: `Subgoal.frame` is the cross-view image the
            policy is conditioned on, and a mask means nothing except against the pixels it
            was drawn from.

        `mask` is how a gaze selection becomes a goal. The object under your eyes has already
        been segmented -- that is what the red box *is* -- so re-running SAM-2 on the same
        point would cost ~113 ms to arrive at the same answer, and might not: a frame or two
        has passed, and the second call could land on the leaf rather than the tree. Passing
        the mask makes the goal start instantly and makes it, exactly, the thing that was
        highlighted when the person said so.

        An Approach goal, or any goal whose stop has `arrive`, locks the instance under the
        point (see the module docstring). If that instance already fills the arrival width
        the goal ends here as "arrived", before a single action is taken.
        """
        self._sync()
        if (mask is None) != (frame is None):
            raise ValueError("mask and frame go together: a mask means nothing except "
                             "against the frame it was computed on")
        given = mask is not None
        if frame is None:
            frame = np.ascontiguousarray(self.info["pov"])
        x, y = self.to_pixels(point, normalized)

        stop = self.with_default_arrival(stop, interaction)
        obj_id = interaction_id(interaction)
        self._compile_stop(stop)                      # refuse a bad goal before SAM-2 runs
        if given:
            mask = np.ascontiguousarray(mask).astype(bool)
            if mask.shape != frame.shape[:2]:
                raise ValueError(f"the mask is {mask.shape} but its frame is "
                                 f"{frame.shape[:2]}; they must be the same size")
        else:
            mask = self.masker.mask(frame, (x, y))
        if not mask.any():
            whose = "the given mask is empty" if given else "the masker returned an empty mask"
            raise ValueError(f"{whose} for point ({x:.0f}, {y:.0f})")

        lock = None
        if self.locks(interaction, stop):
            lock = InstanceLock(self.masker, frame, mask, LockConfig(
                every=self.config.track_every, grace=self.config.lock_grace, fov=self.config.fov))

        if not keep_memory:
            self.clear_memory()
        goal = self._begin(Subgoal(
            stop=stop,
            point=(x, y),
            interaction=str(interaction),
            obj_id=obj_id,
            mask=mask,
            frame=frame.copy(),
            lock=lock,
        ))
        self._camera = (0.0, 0.0)
        self._take_anchor()
        if lock is not None and uses_arrive(stop) and self._stop(self) == "arrived":
            self.result = self._finish("arrived")
        return goal

    def to_pixels(self, point: Sequence[float], normalized: bool = False) -> Tuple[float, float]:
        """`point` in pov pixels. Normalized points scale by the frame -- no crop, no letterbox,
        matching how MineStudio squashes the same view to the 224x224 the server is shown."""
        self._sync()
        height, width = self.info["pov"].shape[:2]
        x, y = float(point[0]), float(point[1])
        if normalized:
            x, y = x * width, y * height
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"point ({x:.0f}, {y:.0f}) is outside the {width}x{height} frame")
        return x, y

    def with_default_arrival(self, stop: StopSpec, interaction: str) -> StopSpec:
        """Approach runs until it gets there, not until the budget runs out.

        Added only where the caller said nothing about arriving: a stop of None, a step count,
        or a dict with no `arrive` key. `"arrive": false` keeps it off; item and stat stops
        and callables are left exactly as they were.
        """
        if str(interaction).lower() != "approach" or max(self.config.arrive_width,
                                                         self.config.arrive_distance) <= 0:
            return stop
        default = {"width": self.config.arrive_width}
        if self.config.arrive_distance > 0:
            default["distance"] = self.config.arrive_distance
        if stop is None:
            return {"arrive": default}
        if isinstance(stop, (int, np.integer)):
            return {"steps": int(stop), "arrive": default}
        if isinstance(stop, dict) and "arrive" not in stop and not ({"item", "stat"} & set(stop)):
            return {**stop, "arrive": default}
        return stop

    @staticmethod
    def locks(interaction: str, stop: StopSpec) -> bool:
        """Which goals carry an instance lock: Approach, and anything that asks to arrive.

        Mine and Hunt do not, on purpose. Their target is *supposed* to vanish -- a log
        broken, a cow killed -- and a lock would call that lost.
        """
        return str(interaction).lower() == "approach" or uses_arrive(stop)

    def update_goal(self, stop: StopSpec = None, point: Optional[Sequence[float]] = None,
                    normalized: bool = False, has_stop: bool = False) -> Dict[str, Any]:
        """A follow-up to the running goal: same lock, same policy state, same step count.

        What a planner resending the goal with `keep_memory: true` means. The stop, if given,
        replaces the old one but is still measured from when the goal started -- so a resend
        every few seconds cannot keep renewing a step budget. The point is *not* re-segmented
        and never retargets the lock: it is only compared against it, and the answer is
        returned for the caller to report.
        """
        goal: Subgoal = self.goal
        if not self.busy:
            raise RuntimeError("no goal is running")
        if has_stop:
            self.replace_stop(self.with_default_arrival(stop, goal.interaction))
        agrees = None
        if point is not None:
            pixels = self.to_pixels(point, normalized)
            agrees = goal.lock.contains(pixels) if goal.lock is not None else None
        self._log(f"follow-up to {goal.interaction} at step {goal.steps}: kept lock and memory"
                  f"{f', stop={goal.stop!r}' if has_stop else ''}"
                  f"{'' if agrees is None else ('; its point is on the locked target' if agrees else '; its point is OFF the locked target -- ignored')}")
        return {"agrees": agrees}

    def arrived(self, thresholds: Dict[str, float]) -> bool:
        """Whether the target is within `thresholds` -- by world distance where it can be.

        The two tests are not peers. Distance is measured between two absolute positions and
        means what it says however the agent is facing; width is an apparent size, and grows
        both when the agent gets close and when it merely turns until the target crops the
        frame. So where the target has an anchor the distance is the whole answer, and width
        only stands in for a target that has never been inside the voxel grid.
        """
        range_now = self.range_to_goal()
        if range_now is not None and "distance" in thresholds:
            return range_now <= thresholds["distance"]
        lock = self.goal.lock if self.goal is not None else None
        if lock is None or not lock.fresh:
            return False
        box = lock.normalized_box()
        if box is None:
            return False
        x0, y0, x1, y1 = box
        return any(extent >= thresholds[key]
                   for key, extent in (("width", x1 - x0), ("height", y1 - y0))
                   if key in thresholds)

    def range_to_goal(self) -> Optional[float]:
        """Blocks between the agent and the goal's anchor, or None while it has none."""
        goal: Optional[Subgoal] = self.goal
        if goal is None or goal.anchor is None:
            return None
        return ranging.horizontal(ranging.position(self.info), goal.anchor)

    def _take_anchor(self) -> Optional[np.ndarray]:
        """Try to pin the running goal to a block, once. Cheap, and a no-op after it lands.

        Retried while it comes back None, because "None" almost always means the target is
        further away than the voxel grid reaches -- which walking toward it fixes. On the
        lock's own cadence rather than every step: the march is a few thousand array lookups
        and the agent covers ~0.2 blocks a step, so retrying oftener buys nothing.
        """
        goal: Optional[Subgoal] = self.goal
        if goal is None or goal.anchor is not None or self.config.arrive_distance <= 0:
            return None
        if goal.steps % max(1, self.config.track_every):
            return None
        box = goal.lock.current_box() if goal.lock is not None else None
        point = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) if box else goal.point
        frame = self.info["pov"]
        goal.anchor = ranging.cast(
            ranging.voxels(self.sim, self.info),
            ranging.view_ray(self.info, point, frame.shape, self.config.fov))
        if goal.anchor is not None:
            self._log(f"anchored the target at ({goal.anchor[0]:.1f}, {goal.anchor[1]:.1f}, "
                      f"{goal.anchor[2]:.1f}), {self.range_to_goal():.1f} blocks away")
        return goal.anchor

    def goal_box(self) -> Optional[Tuple[float, float, float, float]]:
        """The target's box as fractions of the frame: the lock's if there is one, else the
        goal mask's own box on the frame it was pinned to."""
        goal: Optional[Subgoal] = self.goal
        if goal is None:
            return None
        if goal.lock is not None:
            return goal.lock.normalized_box()
        box = mask_box(goal.mask) if goal.mask is not None else None
        if box is None:
            return None
        height, width = goal.mask.shape[:2]
        return box[0] / width, box[1] / height, box[2] / width, box[3] / height

    def run(self, point: Optional[Sequence[float]] = None, interaction: str = "Approach",
            stop: StopSpec = None, normalized: bool = False,
            keep_memory: bool = False) -> Rocket2Result:
        """Set a goal (if given) and step the sim until it ends. Blocking; returns why."""
        if point is not None:
            self.set_goal(point, interaction, stop, normalized=normalized, keep_memory=keep_memory)
        return self.drain()

    def clear_memory(self) -> None:
        """Reset the recurrent state. Also the fix for an agent that has got itself stuck."""
        self.state = self.policy.initial_state()

    # ------------------------------------------------------------------ the loop

    def _action(self) -> Any:
        action, self.state = self.policy.get_action(self._model_input(), self.state, input_shape="*")
        self._read_latents()
        if self.sim.action_type == "env":
            action = self.sim.agent_action_to_env_action(action)
        self._camera = camera_degrees(action)
        return action

    def _stop_reason(self) -> Optional[str]:
        reason = super()._stop_reason()
        if reason is None and self.goal.lock is not None and self.goal.lock.lost:
            return "lost"
        return reason

    def _model_input(self) -> Dict[str, Any]:
        """ROCKET-2's observation: the agent view plus the pinned cross view.

        Both images are squashed to 224x224 without preserving aspect ratio and the mask is
        resized with INTER_LINEAR, matching the upstream demo's preprocessing exactly -- the
        policy is sensitive to how its goal was rendered at training time.
        """
        goal: Subgoal = self.goal
        return {
            "image": self.obs["image"],
            "env_prev_action": self.obs["env_prev_action"],
            "cross_view": {
                "cross_view_image": cv2.resize(goal.frame, OBS_SIZE, interpolation=cv2.INTER_LINEAR),
                "cross_view_obj_id": torch.tensor(goal.obj_id),
                "cross_view_obj_mask": torch.tensor(
                    cv2.resize(goal.mask.astype(np.uint8), OBS_SIZE, interpolation=cv2.INTER_LINEAR),
                    dtype=torch.uint8),
            },
        }

    def _read_latents(self) -> None:
        """Pull the visibility estimate and predicted target point out of the last forward pass."""
        latents = getattr(self.policy, "cache_latents", {})
        if "exist" in latents:
            self.visibility = float(torch.sigmoid(latents["exist"]).reshape(()).item())
        if "point" in latents:
            # ROCKET-2 predicts the centroid as normalized (y, x) -- the order the upstream
            # demo unpacks it in when drawing its crosshair.
            py, px = latents["point"].detach().cpu().numpy().reshape(2).tolist()
            height, width = self.info["pov"].shape[:2]
            self.predicted_point = (px * width, py * height)

    def _after_step(self) -> None:
        # obs["image"] rather than info["pov"]: it is the 224x224 the policy is actually
        # looking at, so what the remote model is shown is what drove the last action --
        # info["pov"] is the same view at 640x360 if it ever needs the detail more than it
        # needs the correspondence.
        goal: Optional[Subgoal] = self.goal
        if self.busy and goal.lock is not None:
            goal.lock.advance(np.ascontiguousarray(self.info["pov"]), self._camera)
        if self.busy:
            self._take_anchor()
        self._follow_gaze()
        self.publisher.offer(self.obs["image"] if self.obs is not None else None,
                             self.selection_payload())
        if self.preview:
            self.preview = gui.show(self.overlay(), "ROCKET-2")

    def _follow_gaze(self) -> None:
        """Move the selection along with the frame, when anything is watching the eyes."""
        if self.selector is None or (self.busy and not self.config.gaze_during_goals):
            return
        self.selector.update(np.ascontiguousarray(self.info["pov"]))
        # Remembered here rather than in `_after_step` so the age means "since the pointing
        # was last seen". `_follow_gaze` is skipped while a goal runs, which leaves the
        # selector's own selection frozen -- and a frozen selection re-remembered every step
        # would report an age of zero for as long as the goal lasted.
        selection = self.selection
        if selection is not None and selection.locked and selection.fresh:
            self._held = (selection, time.monotonic())

    @property
    def selection(self):
        """The object the person is looking at, or None. See perception/selection.py."""
        return self.selector.selection if self.selector is not None else None

    def held_selection(self):
        """The last committed selection, for `selection_hold_ms` after the pointing stopped.

        What makes "mine that" work when *that* was pointed at a second and a half ago: see
        `Rocket2Config.selection_hold_ms`. Expired, it is forgotten rather than kept around
        to be refused on every later look.
        """
        if self._held is None:
            return None
        selection, seen = self._held
        if (time.monotonic() - seen) * 1000.0 > self.config.selection_hold_ms:
            self._held = None
            return None
        return selection

    def held_age(self) -> Optional[float]:
        """Seconds since the held selection was last supported, or None if there is none."""
        if self.held_selection() is None:
            return None
        return time.monotonic() - self._held[1]

    def selection_payload(self) -> Optional[Dict[str, Any]]:
        """What the remote model is told about the pointing, alongside the frame.

        `held` and `age` are the difference between "they are pointing at this right now" and
        "they were pointing at this a moment ago", which is a distinction the model has to be
        able to make: the second is still worth resolving "that" against, and the first is the
        only one it should trust for anything it is about to say is on screen.
        """
        live = self.selection
        if live is not None:
            return {**live.payload(), "held": False, "age": 0.0}
        held = self.held_selection()
        if held is None:
            return None
        return {**held.payload(), "held": True, "age": round(self.held_age() or 0.0, 2)}

    def set_goal_from(self, selection, interaction: str, stop: StopSpec = None,
                      keep_memory: bool = False) -> Subgoal:
        """Run a goal against the object that is currently highlighted.

        The point of the whole gaze path: the mask is already in hand, so this costs no SAM-2
        call and the goal is aimed at exactly what the red box was around -- not at whatever
        a second segmentation of a slightly later frame would have found there.
        """
        return self.set_goal(point=selection.point, interaction=interaction, stop=stop,
                             keep_memory=keep_memory, mask=selection.mask,
                             frame=selection.frame)

    def _result_extras(self) -> Dict[str, Any]:
        return {"visibility": self.visibility}

    # ------------------------------------------------------------------ reporting

    def status(self) -> Dict[str, Any]:
        goal: Optional[Subgoal] = self.goal
        range_now = self.range_to_goal()
        return {
            **super().status(),
            "interaction": goal.interaction if goal else None,
            "point": list(goal.point) if goal else None,
            "visibility": round(self.visibility, 3),
            "box": [round(v, 3) for v in self.goal_box()] if self.goal_box() else None,
            "anchor": ([round(float(v), 1) for v in goal.anchor]
                       if goal is not None and goal.anchor is not None else None),
            "range": None if range_now is None else round(range_now, 1),
            "predicted_point": [round(v, 1) for v in self.predicted_point] if self.predicted_point else None,
        }

    def overlay(self) -> np.ndarray:
        """The current frame (BGR) with the goal mask, the point, and ROCKET-2's own guess."""
        self._sync()
        image = gui.to_bgr(self.info["pov"])
        goal: Optional[Subgoal] = self.goal
        if goal is not None and goal.mask is not None:
            contours, _ = cv2.findContours(goal.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 200, 255), 2)
            cv2.circle(image, (int(goal.point[0]), int(goal.point[1])), 4, (0, 200, 255), -1)
            cv2.putText(image, f"{goal.interaction} | step {goal.steps}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        box = goal.lock.current_box() if goal is not None and goal.lock is not None else None
        if box is not None:
            colour = (0, 220, 0) if goal.lock.fresh else (0, 0, 255)
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]) - 1, int(box[3]) - 1),
                          colour, 2)
        if self.predicted_point is not None:
            px, py = int(self.predicted_point[0]), int(self.predicted_point[1])
            cv2.line(image, (px - 10, py), (px + 10, py), (255, 255, 255), 1)
            cv2.line(image, (px, py - 10), (px, py + 10), (255, 255, 255), 1)
        cv2.rectangle(image, (10, 34), (260, 44), (90, 90, 90), -1)
        cv2.rectangle(image, (10, 34), (10 + int(250 * self.visibility), 44), (255, 255, 255), -1)
        return image
