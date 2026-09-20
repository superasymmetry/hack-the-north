"""Point -> mask. ROCKET-2 wants a segmentation of the target, not a click.

SAM-2 comes from transformers rather than Meta's `sam2` package or the realtime fork
MineStudio's PlaySegmentCallback wants: transformers is already in this env, and the
video/tracking predictors those provide would be wasted here. ROCKET-2 does its own
tracking, so all that is ever needed is one mask on one frame per goal.

Following a *gaze* is the exception: there the point moves many times a second over a frame
that is barely changing, and one mask call per sample is nowhere near fast enough. So the
call is also available split in two -- `embed()` runs the image encoder, `mask_at()` runs
only the prompt decoder against a cached embedding. Measured on this laptop's GPU at 640x360,
fp32:

    mask()                 113 ms      encode + decode, one point
    embed()                 84 ms      the encoder, which is nearly all of it
    mask_at()                4 ms      per point, against an embedding already computed

That is what makes a box that follows your eyes possible at all. See
mcagents/perception/selection.py, which owns the policy of when to re-encode.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence, Tuple

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


@dataclass
class Embedding:
    """Everything SAM-2 learned about one frame -- all `mask_at` needs except the point.

    `shape` and `target` are kept because the processor's own point rescaling happens on the
    way in with the image, and a decode that skips the image has to redo it: the encoder sees
    a square `target` (1024x1024), so a point in a 640x360 frame is scaled by width and height
    *independently*. Getting that wrong does not fail, it quietly segments somewhere else.
    """
    features: List[Any]
    sizes: Any
    #: (height, width) of the frame this was computed from.
    shape: Tuple[int, int]
    #: (height, width) the encoder actually saw.
    target: Tuple[int, int]
    #: `time.monotonic()` when it was computed, so a caller can tell how stale it is.
    t: float = field(default_factory=time.monotonic)

    @property
    def age(self) -> float:
        return time.monotonic() - self.t

    def to_encoder(self, point: Sequence[float]) -> Tuple[float, float]:
        """`point` in frame pixels, as the coordinates the encoder's grid is indexed by."""
        height, width = self.shape
        rows, columns = self.target
        return float(point[0]) * columns / width, float(point[1]) * rows / height


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

    # ------------------------------------------------------------------ the split path

    def embed(self, frame: np.ndarray) -> Embedding:
        """Run the image encoder over `frame` and keep the result. ~84 ms.

        The expensive half of `mask()`, done once so that any number of points can be asked
        about the same frame for ~4 ms each.
        """
        frame = np.ascontiguousarray(frame)
        inputs = self.processor(images=frame, return_tensors="pt").to(self.config.device)
        with torch.inference_mode():
            features = self.model.get_image_embeddings(inputs["pixel_values"])
        return Embedding(
            features=features,
            sizes=inputs["original_sizes"],
            shape=frame.shape[:2],
            target=tuple(inputs["pixel_values"].shape[-2:]),
        )

    def mask_at(self, embedding: Embedding, point: Sequence[float]) -> np.ndarray:
        """Segment the object at `point` ([x, y] pixels) in the frame `embedding` came from. ~4 ms.

        Identical to `mask()` on the same frame and point -- the only thing skipped is
        recomputing what the image looks like, which is the thing that did not change.
        """
        x, y = embedding.to_encoder(point)
        points = torch.tensor([[[[x, y]]]], dtype=torch.float32, device=self.config.device)
        labels = torch.tensor([[[1]]], dtype=torch.int32, device=self.config.device)
        with torch.inference_mode():
            outputs = self.model(image_embeddings=embedding.features, input_points=points,
                                 input_labels=labels, multimask_output=False)
        masks = self.processor.post_process_masks(outputs.pred_masks.float(), embedding.sizes)
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
