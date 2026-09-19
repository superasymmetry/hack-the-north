"""Turning a frame into somewhere to point: detection (text -> box) and masking (point -> mask)."""
from mcagents.perception.detection import Detection, Targeter, load_targeter

__all__ = ["Detection", "Targeter", "load_targeter"]
