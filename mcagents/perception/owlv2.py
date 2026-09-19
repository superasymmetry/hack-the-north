"""Open-vocabulary detection with OWLv2, so a plan can say "tree" instead of a pixel.

OWLv2 (arXiv:2306.09683) scores every box it proposes against a text query rather than a
fixed label set, so "cow", "oak log" and "iron ore block" all work untrained. It is chosen
for the same reason SAM-2 is: transformers is already in this env, so there is nothing to
install and nothing to serve. The base checkpoint is ~150M params (~600 MB fp32), which fits
alongside ROCKET-2's 950 MB and SAM-2's 455 MB, and it runs once per goal, not once per step.

Minecraft is off OWLv2's training distribution, so scores run low and the default threshold
is deliberately loose. To eyeball a query before trusting it:

    python -m mcagents.cli.locate frame.png "tree"
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from mcagents.perception.detection import Detection, best

#: OWLv2 was trained on captions, not bare nouns, and scores noticeably higher on a phrase.
QUERY_TEMPLATE = "a photo of a {}"


@dataclass
class OwlV2Config:
    model_id: str = "google/owlv2-base-patch16-ensemble"
    device: str = "cuda"
    #: 0.1 is the transformers default, about right for natural images. Minecraft blocks and
    #: mobs score lower, so treat it as a starting point and tune it on your own frames.
    threshold: float = 0.1

    @classmethod
    def from_env(cls) -> "OwlV2Config":
        return cls(
            model_id=os.environ.get("MCAGENTS_OWL_MODEL", cls.model_id),
            device=os.environ.get("MCAGENTS_DEVICE", cls.device),
            threshold=float(os.environ.get("MCAGENTS_OWL_THRESHOLD", cls.threshold)),
        )


class OwlV2Targeter:
    """OWLv2 as a point source. Load once and reuse: the constructor pulls ~600 MB onto the GPU."""

    def __init__(self, config: Optional[OwlV2Config] = None):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.config = config or OwlV2Config.from_env()
        self.processor = Owlv2Processor.from_pretrained(self.config.model_id)
        self.model = (Owlv2ForObjectDetection.from_pretrained(self.config.model_id)
                      .to(self.config.device).eval())

    def detect(self, frame: np.ndarray, text: str) -> List[Detection]:
        """Every box matching `text` in `frame` (H, W, 3 RGB), best score first."""
        frame = np.ascontiguousarray(frame)      # torch refuses MineRL's negative strides
        height, width = frame.shape[:2]

        inputs = self.processor(images=frame, text=[[QUERY_TEMPLATE.format(text)]],
                                return_tensors="pt").to(self.config.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # The processor pads the frame to a square (bottom and right) before resizing to
        # 960x960, so its boxes are normalized against that square, not against the 640x360
        # frame. Passing the original size as target_sizes would stretch every y coordinate
        # by 640/360; passing the square's side in original pixels undoes resize and padding
        # in one step and lands the boxes back in frame coordinates.
        side = max(height, width)
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.config.threshold, target_sizes=[(side, side)]
        )[0]

        found = [Detection(float(score), tuple(float(v) for v in box)).clipped(width, height)
                 for score, box in zip(results["scores"].tolist(), results["boxes"].tolist())]
        # The padding sat outside the real image, so a box can legitimately overhang it: clip
        # rather than drop, since a half-visible tree at the frame edge is still a valid target.
        found = [d for d in found if d.area > 0]
        found.sort(key=lambda d: d.score, reverse=True)
        return found

    def locate(self, frame: np.ndarray, text: str,
               prefer: str = "score") -> Optional[Tuple[float, float]]:
        """The point to hand a controller, or None if nothing matched.

        The box centre is not guaranteed to land *on* the object -- a tree's centre can fall
        between branches, and SAM-2 will then happily segment the sky behind it. Rocket2Agent
        raises on an empty mask, so that failure is loud, but it is the reason to check hit
        rate on real frames before trusting this end to end.
        """
        return best(self.detect(frame, text), prefer)
