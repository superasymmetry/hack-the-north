"""Point -> mask. ROCKET-2 wants a segmentation of the target, not a click.

SAM-2 comes from transformers rather than Meta's `sam2` package or the realtime fork
MineStudio's PlaySegmentCallback wants: transformers is already in this env, and the
video/tracking predictors those provide would be wasted here. ROCKET-2 does its own
tracking, so all that is ever needed is one mask on one frame per goal.
"""
import os
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np
import torch


class Masker(Protocol):
    """point + frame -> a boolean mask at frame resolution."""

    def mask(self, frame: np.ndarray, point: Sequence[float]) -> np.ndarray:
        ...


@dataclass
class SamConfig:
    #: Tiny is the right pick: the mask only has to be right once per goal, and the bigger
    #: variants cost VRAM that Minecraft's renderer and ROCKET-2 want.
    model_id: str = "facebook/sam2.1-hiera-tiny"
    device: str = "cuda"

    @classmethod
    def from_env(cls) -> "SamConfig":
        return cls(
            model_id=os.environ.get("MCAGENTS_SAM_MODEL", cls.model_id),
            device=os.environ.get("MCAGENTS_DEVICE", cls.device),
        )


class PointMasker:
    """SAM-2: segments whatever object contains the point. ~455 MB, loaded once."""

    def __init__(self, config: Optional[SamConfig] = None):
        from transformers import Sam2Model, Sam2Processor

        self.config = config or SamConfig.from_env()
        self.processor = Sam2Processor.from_pretrained(self.config.model_id)
        self.model = Sam2Model.from_pretrained(self.config.model_id).to(self.config.device).eval()

    def mask(self, frame: np.ndarray, point: Sequence[float]) -> np.ndarray:
        """Segment the object at `point` ([x, y] pixels) in `frame` (H, W, 3 RGB)."""
        frame = np.ascontiguousarray(frame)      # torch refuses MineRL's negative strides
        inputs = self.processor(
            images=frame,
            input_points=[[[[float(point[0]), float(point[1])]]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        ).to(self.config.device)
        with torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])
        return masks[0][0].squeeze().cpu().numpy().astype(bool)

    def mask_fast(self, frame: np.ndarray, point: Sequence[float]) -> np.ndarray:
        """`mask`, for tracking: bf16, with the frame preprocessed on the card.

        ~35 ms against ~116 ms on the laptop GPU, and the same mask on the frames it was
        checked against (IoU 1.0). The goal mask ROCKET-2 is conditioned on still comes from
        `mask`, so the policy's input is exactly what it was.
        """
        image = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).to(self.config.device)
        inputs = self.processor(
            images=image,
            input_points=[[[[float(point[0]), float(point[1])]]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        ).to(self.config.device)
        with torch.inference_mode(), torch.autocast(self.config.device.split(":")[0],
                                                    dtype=torch.bfloat16):
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(outputs.pred_masks.float(),
                                                  inputs["original_sizes"])
        return masks[0][0].squeeze().cpu().numpy().astype(bool)


class DiskMasker:
    """Degraded fallback: a filled circle around the point, for when SAM-2 is unavailable.

    ROCKET-2 was trained on real object masks, so a disk is a worse goal specification than a
    segmentation -- fine for a smoke test, not for a demo.
    """

    def __init__(self, radius: int = 40):
        self.radius = radius

    def mask(self, frame: np.ndarray, point: Sequence[float]) -> np.ndarray:
        canvas = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(canvas, (int(point[0]), int(point[1])), self.radius, 1, -1)
        return canvas.astype(bool)
