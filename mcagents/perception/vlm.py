"""The remote alternative to OWLv2: a Qwen2.5-VL on the GPU box, same detect/locate contract.

OWLv2 matches a caption against boxes proposed from natural-image priors, and Minecraft is
off that distribution. A 32B/72B VLM has actually seen Minecraft screenshots and takes
phrases -- "the nearest tree", "the cow facing away" -- rather than bare nouns. It costs a
network round trip (~0.3-1.5 s on a LAN), which is affordable because pointing happens once
per goal, not once per step: ROCKET-2 holds the goal mask fixed and tracks the target itself.

    export MCAGENTS_TARGETER=vlm
    export MCAGENTS_VLM_URL=http://10.0.0.7:8000/v1
    python -m mcagents.cli.locate frame.png "the nearest tree"

The one thing that can silently ruin this is coordinates: Qwen2.5-VL emits absolute pixels
in the image *as its processor resized it*, not as you sent it -- hence `smart_resize`.
"""
import base64
import itertools
import json
import os
import re
from dataclasses import dataclass, field
from math import ceil, floor, sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mcagents.perception.detection import Box, Detection, best

#: Qwen2.5-VL's own grounding phrasing. Asking for JSON with a bbox_2d key is what it was
#: tuned to answer; inventing a format here measurably degrades it.
PROMPT = (
    "Locate {text} in this Minecraft screenshot. Output every match as JSON: "
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}]. '
    "Order them best-first. Choose points that lie on the object itself, not on background "
    "visible through it. If there is no match, output []."
)


def _urls_from_env(name: str, default: str) -> List[str]:
    """Comma-separated server URLs, so a multi-replica server needs no load balancer."""
    return [u.strip().rstrip("/") for u in os.environ.get(name, default).split(",") if u.strip()]


@dataclass
class VLMConfig:
    urls: List[str] = field(default_factory=lambda: ["http://127.0.0.1:8000/v1"])
    model: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    timeout: float = 60.0
    #: A 640x360 frame is small and a distant sheep is a handful of pixels; upscaling before
    #: sending buys the model more visual tokens to spend on it. 2x costs ~4x the prefill,
    #: which is nothing next to the round trip.
    upscale: float = 2.0
    #: Must match the server's --mm-processor-kwargs, or `smart_resize` below computes a
    #: different size than the server did and every point lands off by a constant factor.
    min_pixels: int = 3136
    max_pixels: int = 2560 * 28 * 28
    #: Measured over an SSH tunnel: a 1280x720 frame is ~2.2 MB as base64 PNG and ~375 KB as
    #: JPEG, and the round trip went 4.6s -> 1.3s for the identical predicted point.
    jpeg_quality: int = 92

    @classmethod
    def from_env(cls) -> "VLMConfig":
        return cls(
            urls=_urls_from_env("MCAGENTS_VLM_URL", "http://127.0.0.1:8000/v1"),
            model=os.environ.get("MCAGENTS_VLM_MODEL", cls.model),
            timeout=float(os.environ.get("MCAGENTS_VLM_TIMEOUT", cls.timeout)),
            upscale=float(os.environ.get("MCAGENTS_VLM_UPSCALE", cls.upscale)),
        )


def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = 3136, max_pixels: int = 2560 * 28 * 28) -> Tuple[int, int]:
    """The size Qwen's processor will resize an (height, width) image to.

    Reproduced from the Qwen2.5-VL processor rather than imported, because the server runs it
    and the client needs the answer: the model's output coordinates live in *this* space.
    """
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = sqrt((height * width) / max_pixels)
        h = max(factor, floor(height / beta / factor) * factor)
        w = max(factor, floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = sqrt(min_pixels / (height * width))
        h = ceil(height * beta / factor) * factor
        w = ceil(width * beta / factor) * factor
    return h, w


def parse_boxes(text: str) -> List[Box]:
    """Pull boxes out of whatever the model actually said.

    It is asked for a bare JSON array and usually gives one, but ```json fences and a
    sentence of preamble are both common. Anything unparseable is dropped rather than raised
    on: a malformed box is the same outcome as no detection, which callers already handle.
    """
    block = re.search(r"\[.*\]", text, re.S)
    if not block:
        return []
    try:
        items = json.loads(block.group(0))
    except json.JSONDecodeError:
        return []

    boxes: List[Box] = []
    for item in items if isinstance(items, list) else []:
        box = item.get("bbox_2d") or item.get("bbox") or item.get("box") if isinstance(item, dict) else item
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                boxes.append(tuple(float(v) for v in box))
            except (TypeError, ValueError):
                continue
    return boxes


class VLMTargeter:
    """A remote Qwen2.5-VL as a point source, interchangeable with OwlV2Targeter."""

    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig.from_env()
        if not self.config.urls:
            raise ValueError("no server URLs -- set MCAGENTS_VLM_URL")
        self._next_url = itertools.cycle(self.config.urls)

    @property
    def urls(self) -> List[str]:
        return self.config.urls

    @property
    def model(self) -> str:
        return self.config.model

    def detect(self, frame: np.ndarray, text: str) -> List[Detection]:
        """Every box matching `text` in `frame` (H, W, 3 RGB), in frame coordinates."""
        out_h, out_w = frame.shape[:2]
        data_url, sent_h, sent_w = self._encode(frame)
        content = self._ask(data_url, PROMPT.format(text=text))

        # The coordinates are absolute pixels in the image *after* the processor's resize, so
        # they must be divided by that size and not by what we sent. For a 640x360 frame at
        # 2x upscale the two differ by under 1% -- exactly the kind of error that reads as
        # the model being slightly sloppy instead of as a bug.
        proc_h, proc_w = smart_resize(sent_h, sent_w, min_pixels=self.config.min_pixels,
                                      max_pixels=self.config.max_pixels)
        sx, sy = out_w / proc_w, out_h / proc_h

        found = []
        for x0, y0, x1, y1 in parse_boxes(content):
            # No calibrated confidence here, unlike OWLv2: the model gives an ordered list and
            # nothing else, so scores descend by rank and prefer="score" means "its first pick".
            box = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
            found.append(Detection(1.0 / (len(found) + 1), box).clipped(out_w, out_h))
        return [d for d in found if d.area > 0]

    def locate(self, frame: np.ndarray, text: str,
               prefer: str = "score") -> Optional[Tuple[float, float]]:
        """The point to hand a controller, or None if nothing matched.

        With a VLM you can often skip `prefer` and just say what you mean --
        locate(frame, "the nearest tree").
        """
        return best(self.detect(frame, text), prefer)

    # ------------------------------------------------------------------ transport

    def _encode(self, frame: np.ndarray) -> Tuple[str, int, int]:
        """Frame -> data URL, plus the (height, width) the server's processor will see."""
        import cv2

        frame = np.ascontiguousarray(frame)       # cv2 refuses MineRL's negative strides
        height, width = frame.shape[:2]
        if self.config.upscale != 1.0:
            width, height = int(width * self.config.upscale), int(height * self.config.upscale)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        if not ok:
            raise RuntimeError("cv2 failed to encode the frame as JPEG")
        return ("data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode(),
                height, width)

    def _ask(self, data_url: str, prompt: str) -> str:
        import requests

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]}],
            # Greedy: the same frame and query must give the same point, or a failed goal
            # cannot be reproduced.
            "temperature": 0.0,
            "max_tokens": 512,
        }
        response = requests.post(f"{next(self._next_url)}/chat/completions", json=payload,
                                 timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
