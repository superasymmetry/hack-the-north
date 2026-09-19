"""JarvisVLA-Qwen2-VL-7B as a plain-text controller: an English sentence in, keyboard out.

The other end of the trade from ROCKET-2. ROCKET-2 takes a point, which fixes "which
object?" but needs something to do the pointing and cannot drive a GUI. JarvisVLA
(arXiv:2503.16365) is a 7B vision-language model post-trained to emit VPT actions directly,
so the goal channel is a sentence -- "Chop down the oak log." -- and inventory and
crafting-table GUIs are in scope because it was trained on them.

What it costs is that the policy *is* a 7B VLM: it will not run next to Minecraft on a
laptop GPU. So it runs on the GPU box under vLLM (scripts/serve_jarvisvla.sh) and this module
is the client.

    agent = JarvisVLAAgent(sim)            # sim needs the 21-bin camera config, see _check_sim
    result = agent.run("Chop down the oak log.", stop={"item": "log", "count": 3})

Two failure modes are silent -- a still or drunk agent, never an exception:

1. The camera config. JarvisVLA's tokens are 21-way; MinecraftSim's default CameraConfig is
   11-way. `_check_sim()` refuses to start rather than let that through. `sim_kwargs()` below
   is the configuration that satisfies it.
2. `skip_special_tokens` must be off. The action *is* a run of special tokens, and vLLM
   strips those from the response by default -- which leaves an empty string, which decodes
   to the null action: an agent that stands perfectly still while the server returns 200 OK.

See docs/jarvisvla.md for the wiring.
"""
import base64
import itertools
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests

# Before anything that pulls in minestudio: the first OpenCV window this process opens has
# to open while PyAV is still out of it, or it blocks forever. See mcagents.gui.
from mcagents import gui
from mcagents.agents.base import Agent, Goal, GoalResult, StopSpec
from mcagents.agents.jarvisvla_actions import (N_CAMERA_BINS, decode_actions, describe_action,
                                               null_action_text)


@dataclass
class JarvisVLAConfig:
    #: Comma-separated in the environment: serve_jarvisvla.sh can bring up one replica per
    #: GPU and the agent round-robins over them. One call per *step*, so this is a hot path.
    urls: List[str] = field(default_factory=lambda: ["http://127.0.0.1:8000/v1"])
    #: Left None, the served id is read from /v1/models -- which is what keeps a
    #: `--served-model-name` on the server side from breaking the client.
    model: Optional[str] = None
    timeout: float = 60.0
    #: Upstream's rollout defaults (scripts/evaluate/rollout-*.sh). Greedy (0.0) is
    #: noticeably worse in the way it is for VPT-style policies: it gets stuck holding a key.
    temperature: float = 0.6
    #: Frames of (observation, action) history the model conditions on.
    history_num: int = 2
    #: How many actions from one response to play before asking again. The model is trained
    #: to emit one per turn, so >1 only trades control frequency for round trips.
    chunk_len: int = 1
    max_steps: int = 600
    #: Predict on a worker thread instead of stopping the game to wait for the server. The
    #: game then runs at its own rate and the policy re-decides whenever a reply lands, so
    #: each decision is held for several ticks and acts on a frame that is one call old.
    #: That is a real behavioural change, which is why it is off by default -- but it is the
    #: difference between a demo that stutters and one that runs at Minecraft's own speed.
    async_predict: bool = False
    #: Ticks per second to hold the game to. Minecraft's own rate is 20; the sim will run at
    #: ~38 unpaced, which looks fast-forwarded. 0 removes the cap (for benchmarking).
    #: Irrelevant unless async_predict is on -- a synchronous rollout never reaches it.
    fps: float = 20.0
    #: Upstream's processor saves frames through PIL at the default quality, so quality-75
    #: artifacts are what the model was evaluated against -- not a bandwidth compromise.
    #: It is still the largest byte knob there is: q75 is 43 KB a frame, q50 29 KB, q40
    #: 25 KB, and over a home uplink the bytes are most of a step. See docs/jarvisvla.md.
    jpeg_quality: int = 75
    #: A live window of what the model is looking at, with the action it chose drawn on top.
    #: Turns itself off when there is no display to open it on -- see mcagents.gui.
    preview: bool = True

    @classmethod
    def from_env(cls) -> "JarvisVLAConfig":
        urls = [u.strip().rstrip("/") for u in
                os.environ.get("JARVISVLA_URL", "http://127.0.0.1:8000/v1").split(",") if u.strip()]
        return cls(
            urls=urls,
            model=os.environ.get("JARVISVLA_MODEL") or None,
            timeout=float(os.environ.get("JARVISVLA_TIMEOUT", cls.timeout)),
            temperature=float(os.environ.get("JARVISVLA_TEMPERATURE", cls.temperature)),
            history_num=int(os.environ.get("JARVISVLA_HISTORY", cls.history_num)),
            chunk_len=int(os.environ.get("JARVISVLA_CHUNK", cls.chunk_len)),
            max_steps=int(os.environ.get("JARVISVLA_MAX_STEPS", cls.max_steps)),
            jpeg_quality=int(os.environ.get("JARVISVLA_JPEG_QUALITY", cls.jpeg_quality)),
            async_predict=os.environ.get("JARVISVLA_ASYNC", "0") != "0",
            fps=float(os.environ.get("JARVISVLA_FPS", cls.fps)),
            preview=os.environ.get("JARVISVLA_PREVIEW", "1") != "0",
        )


@dataclass
class Task(Goal):
    """One instruction and the condition that ends it."""
    instruction: str = ""
    calls: int = 0
    wait: float = 0.0                            # seconds spent waiting on the server

    def describe(self) -> str:
        return f"{self.instruction!r} stop={self.stop!r}"


@dataclass
class TaskResult(GoalResult):
    calls: int = 0
    wait: float = 0.0

    def __str__(self) -> str:
        rate = f", {self.steps / self.seconds:.1f} steps/s" if self.seconds else ""
        per_call = f" ({self.wait / self.calls * 1000:.0f} ms/call)" if self.calls else ""
        held = (f", {self.steps / self.calls:.1f} ticks per decision"
                if self.calls and self.steps / self.calls > 1.05 else "")
        return (f"{self.reason} after {self.steps} steps ({self.seconds:.1f}s{rate}), "
                f"gained {self.gained_text}, {self.wait:.1f}s of that waiting on the "
                f"model{per_call}{held}")


def sim_kwargs() -> Dict[str, Any]:
    """The MinecraftSim arguments this checkpoint needs, as a dict to splat into the constructor.

    action_type="agent" because the model emits VPT (buttons, camera) indices; 640x360 for
    both obs and render to match upstream's evaluation configs (the model reads info['pov'],
    which is render_size); and the 21-bin camera quantization its action tokens are written
    against.
    """
    from minestudio.simulator.entry import CameraConfig

    return {
        "action_type": "agent",
        "obs_size": (640, 360),
        "render_size": (640, 360),
        "camera_config": CameraConfig(camera_binsize=1, camera_maxval=10, camera_mu=20,
                                      camera_quantization_scheme="mu_law"),
    }


def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = 4 * 28 * 28, max_pixels: int = 1024 * 28 * 28) -> Tuple[int, int]:
    """The size Qwen2-VL's processor will resize an (height, width) image to.

    Reproduced from JarvisVLA's processor_wrapper because the server runs it and the client
    wants the answer: a 640x360 frame becomes 644x364, and sending exactly what the model
    will see keeps one resample out of the loop.
    """
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = ((height * width) / max_pixels) ** 0.5
        h = max(factor, int(height / beta // factor) * factor)
        w = max(factor, int(width / beta // factor) * factor)
    elif h * w < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        h = -(-int(height * beta) // factor) * factor
        w = -(-int(width * beta) // factor) * factor
    return h, w


def encode_frame(frame: np.ndarray, jpeg_quality: int = 75) -> str:
    """An RGB frame -> the data URL to put in a chat message."""
    frame = np.ascontiguousarray(frame)          # cv2 refuses MineRL's negative strides
    height, width = frame.shape[:2]
    new_h, new_w = smart_resize(height, width)
    if (new_h, new_w) != (height, width):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("cv2 failed to encode the frame as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


#: What the message builders accept for an image: a raw frame, or the data URL a frame has
#: already been encoded to. The second is what lets a frame be sent three times for one
#: encode -- see JarvisVLAClient.encode.
Image = Union[np.ndarray, str]


def _label(image: np.ndarray, text: str, y: int) -> None:
    """White caption over a black outline -- plain white is unreadable against Minecraft's sky."""
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness,
                    cv2.LINE_AA)


class JarvisVLAClient:
    """The vLLM side: model discovery, message building, one chat completion per call.

    Separate from the agent so a single frame can be checked without a Minecraft:

        python -m mcagents.cli.jarvisvla --frame logs/frame.png "Chop down the oak log."
    """

    def __init__(self, config: Optional[JarvisVLAConfig] = None):
        self.config = config or JarvisVLAConfig.from_env()
        if not self.config.urls:
            raise ValueError("no server URLs -- set JARVISVLA_URL")
        self._next_url = itertools.cycle(self.config.urls)
        # One pooled connection for the whole rollout. Without it every step opens a fresh
        # TCP connection, and through `ssh -L` that is a new channel negotiated with the
        # login node before a single byte of the frame moves: ~110 ms per step, measured.
        self._session = requests.Session()
        self.model = self.config.model or self.discover_model()

    def discover_model(self) -> str:
        """Ask the server what it is serving, and fail loudly with the fix if it is not there.

        Doubles as the reachability probe worth running *before* a 30s Minecraft boot: an
        unreachable tunnel should cost a second, not a reset.
        """
        url = f"{self.config.urls[0]}/models"
        try:
            response = self._session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()["data"][0]["id"]
        except Exception as exc:
            raise SystemExit(
                f"cannot reach a vLLM server at {url} ({exc}).\n"
                f"Start one on the GPU box with scripts/serve_jarvisvla.sh, tunnel to it, and\n"
                f"export JARVISVLA_URL. See docs/jarvisvla.md."
            ) from exc

    def complete(self, messages: List[Dict[str, Any]]) -> str:
        """One chat completion. Returns the raw reply, action tokens and all."""
        response = self._session.post(
            f"{next(self._next_url)}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "top_p": 0.99,
                "top_k": -1,
                "max_tokens": 1024,
                # Without this the reply comes back empty and every step decodes to the null
                # action -- see the module docstring.
                "skip_special_tokens": False,
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def encode(self, frame: Image) -> str:
        """A frame -> the data URL to put in a message. An encoded one passes through.

        With history_num=2 each frame is sent three times -- once as the observation, then
        twice more as history -- and encoding it once per send is three JPEG passes for one
        image. The agent encodes each frame here once and keeps the string.
        """
        return frame if isinstance(frame, str) else encode_frame(frame, self.config.jpeg_quality)

    def build_messages(self, instruction: str, frame: Image,
                       history: List[Tuple[Image, str]]) -> List[Dict[str, Any]]:
        """The conversation upstream's VLLM_AGENT.forward builds, verbatim in shape.

        The instruction rides on the *first* message only; every later turn is just
        "observation:" plus a frame, with the model's own previous replies as the assistant
        turns. Before the first reply exists, history is padded with the current frame and
        the null action, which is upstream's cold start.
        """
        if self.config.history_num and not history:
            history = [(frame, null_action_text())] * self.config.history_num

        messages: List[Dict[str, Any]] = []
        for index, (past_frame, action_text) in enumerate(history):
            prompt = "\nobservation: "
            if index == 0:
                prompt = instruction + prompt
            messages.append(self._user_message(prompt, past_frame))
            messages.append({"role": "assistant",
                             "content": [{"type": "text", "text": f"{action_text}\n"}]})

        prompt = "\nobservation: "
        if not self.config.history_num:
            prompt = instruction + prompt
        messages.append(self._user_message(prompt, frame))
        return messages

    def _user_message(self, text: str, frame: Optional[Image] = None) -> Dict[str, Any]:
        """Text first, then the image -- the order JarvisVLA's create_message_vllm produces."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": f"{text}\n"}]
        if frame is not None:
            content.append({"type": "image_url", "image_url": {"url": self.encode(frame)}})
        return {"role": "user", "content": content}


class _Predictor:
    """Runs the model on a worker thread, newest frame wins.

    The game and the server run at very different rates -- ~38 ticks/s against ~5 calls/s --
    and a synchronous loop resolves that by making the game wait. This resolves it the other
    way: `submit` hands over the current frame and returns immediately, `take` hands back the
    most recent reply. A frame submitted while a call is in flight replaces any frame still
    queued, so the model always thinks about the freshest observation rather than working
    through a backlog it can never catch up with.
    """

    def __init__(self, predict):
        self._predict = predict
        self._frame = None                       # the next frame to think about, if any
        self._result = None                      # the newest actions decoded from a reply
        self._error: Optional[BaseException] = None
        self._stop = threading.Event()
        self._ready = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="jarvisvla-predict", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        with self._ready:
            self._frame = frame
            self._ready.notify_all()

    def take(self, timeout: float) -> Optional[List[Dict[str, np.ndarray]]]:
        """The newest prediction. Blocks only before the first one has ever arrived."""
        with self._ready:
            if self._result is None:
                self._ready.wait_for(
                    lambda: self._result is not None or self._error or self._stop.is_set(),
                    timeout)
            if self._error:
                raise self._error
            return self._result

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._ready:
                self._ready.wait_for(lambda: self._frame is not None or self._stop.is_set(), 0.2)
                if self._stop.is_set():
                    return
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            try:
                result = self._predict(frame)
            except BaseException as exc:        # surfaced on the next take(), not swallowed
                with self._ready:
                    self._error = exc
                    self._ready.notify_all()
                return
            with self._ready:
                self._result = result
                self._ready.notify_all()

    def close(self) -> None:
        with self._ready:
            self._stop.set()
            self._ready.notify_all()
        self._thread.join(timeout=2.0)


class JarvisVLAAgent(Agent):
    """Drives a MinecraftSim from an English instruction, one server call per step."""

    log_tag = "jarvisvla"
    result_class = TaskResult

    def __init__(self, sim, config: Optional[JarvisVLAConfig] = None,
                 client: Optional[JarvisVLAClient] = None, verbose: bool = True):
        self.config = config or JarvisVLAConfig.from_env()
        self._check_sim(sim)
        super().__init__(sim, max_steps=self.config.max_steps, verbose=verbose)

        self.client = client or JarvisVLAClient(self.config)
        self.last_action_text = ""
        #: (encoded frame, action-token text) for the last `history_num` steps, oldest first.
        self._history: List[Tuple[str, str]] = []
        self._pending: List[Dict[str, np.ndarray]] = []
        #: Live only while an async goal is running -- see _Predictor.
        self._predictor: Optional[_Predictor] = None
        self._last_tick = time.time()
        #: Cleared for the rest of the run the first time gui.show() reports no window.
        self.preview = self.config.preview

        extra = f" (+{len(self.config.urls) - 1} more)" if len(self.config.urls) > 1 else ""
        self._log(f"{self.client.model} at {self.config.urls[0]}{extra}")

    @staticmethod
    def _check_sim(sim) -> None:
        """Refuse a sim whose action space is not the one this checkpoint speaks."""
        bins = getattr(getattr(sim, "action_mapper", None), "n_camera_bins", None)
        if bins != N_CAMERA_BINS:
            raise ValueError(
                f"this sim quantizes the camera into {bins} bins, but JarvisVLA's action "
                f"tokens are {N_CAMERA_BINS}-way. Build the sim with "
                f"**mcagents.agents.jarvisvla.sim_kwargs()."
            )
        if getattr(sim, "action_type", None) != "agent":
            raise ValueError(
                f"sim.action_type is {getattr(sim, 'action_type', None)!r}; JarvisVLA emits "
                f"VPT (buttons, camera) actions, so it needs 'agent'."
            )

    # ------------------------------------------------------------------ goals

    def set_goal(self, instruction: str, stop: StopSpec = None) -> Task:
        """Start a new instruction, clearing the frame/action history it conditions on."""
        self._stop_predicting()
        self._history = []
        self._pending = []
        self.last_action_text = ""
        goal = self._begin(Task(stop=stop, instruction=instruction))
        self._last_tick = time.time()
        if self.config.async_predict:
            self._predictor = _Predictor(self._predict)
        return goal

    def _stop_predicting(self) -> None:
        """Join the worker, so nothing is still writing to a goal that has finished."""
        if self._predictor is not None:
            self._predictor.close()
            self._predictor = None

    def close(self) -> None:
        self._stop_predicting()

    def _finish(self, reason: str):
        # Before super(), which reads goal.calls and goal.wait into the result.
        self._stop_predicting()
        return super()._finish(reason)

    def run(self, instruction: str, stop: StopSpec = None) -> TaskResult:
        """Set a task and step it to completion."""
        self.set_goal(instruction, stop)
        return self.drain()

    # ------------------------------------------------------------------ the loop

    def _action(self) -> Dict[str, np.ndarray]:
        if self._predictor is not None:
            return self._copy(self._async_action())
        if not self._pending:
            self._pending = self._predict()
        return self._copy(self._pending.pop(0))

    @staticmethod
    def _copy(action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """MinecraftSim.step pops 'buttons'/'camera' out of the dict it is given, so what goes
        in has to be a throwaway -- doubly so in async, where one decision is stepped many times."""
        return {key: np.asarray(value).copy() for key, value in action.items()}

    def _async_action(self) -> Dict[str, np.ndarray]:
        """Hand the worker the current frame; keep playing the newest reply it has returned."""
        self._predictor.submit(self.frame)
        actions = self._predictor.take(self.config.timeout)
        if actions is None:
            raise RuntimeError(
                f"no reply from {self.config.urls[0]} within {self.config.timeout}s of the "
                f"first frame. The server is reachable (the model id was read from it) but "
                f"not answering -- check it is not still loading weights."
            )
        return actions[0]

    def _predict(self, frame: Optional[np.ndarray] = None) -> List[Dict[str, np.ndarray]]:
        """One server call: a frame (plus history) -> up to chunk_len actions.

        Called on the main thread synchronously, or on the worker with the frame it was
        handed. Everything it touches -- the history, the call counters, last_action_text --
        belongs to whichever of the two is running, never both: the worker is started in
        set_goal and joined in _finish.
        """
        image = self.client.encode(self.frame if frame is None else frame)
        messages = self.client.build_messages(self.goal.instruction, image, self._history)

        started = time.time()
        text = self.client.complete(messages)
        self.goal.calls += 1
        self.goal.wait += time.time() - started
        self.last_action_text = text

        if self.config.history_num:
            self._history = (self._history + [(image, text)])[-self.config.history_num:]

        return decode_actions(text)[:max(1, self.config.chunk_len)]

    def _after_step(self) -> None:
        # Unpaced, the sim runs at ~38 ticks/s and an async rollout looks fast-forwarded.
        # A synchronous one never gets near the cap, so this costs it nothing.
        if self.config.fps > 0:
            time.sleep(max(0.0, self._last_tick + 1.0 / self.config.fps - time.time()))
            self._last_tick = time.time()
        if self.preview:
            self.preview = gui.show(self.overlay(), "JarvisVLA")

    def _result_extras(self) -> Dict[str, Any]:
        return {"calls": self.goal.calls, "wait": self.goal.wait}

    # ------------------------------------------------------------------ reporting

    @property
    def last_action(self) -> Optional[Dict[str, np.ndarray]]:
        """The action decoded from the model's most recent reply, if there has been one."""
        return decode_actions(self.last_action_text)[0] if self.last_action_text else None

    def describe_last_action(self) -> str:
        """What the model is currently pressing -- the signal that tells a stalled run from a lost one."""
        action = self.last_action
        return describe_action(action, self.sim) if action is not None else "-"

    def overlay(self) -> np.ndarray:
        """The current frame (BGR) captioned with the instruction and what it is pressing.

        The same two facts the log lines carry, at every step instead of every 25th: a run
        that is stuck holding one key looks nothing like one that is hunting for a tree.
        """
        image = gui.to_bgr(self.frame)
        goal: Optional[Task] = self.goal
        if goal is not None:
            _label(image, goal.instruction, 24)
            _label(image, f"step {goal.steps}  {self.describe_last_action()}", 46)
        return image

    def status(self) -> Dict[str, Any]:
        return {
            **super().status(),
            "instruction": self.goal.instruction if self.goal else None,
            "action": self.describe_last_action() if self.last_action_text else None,
        }
