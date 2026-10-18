"""Image-only matching methods. No GT/masks are accepted by this interface."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2

@dataclass
class Prediction:
    method: str
    points0: np.ndarray = field(default_factory=lambda: np.empty((0,2), dtype=np.float32))
    points1: np.ndarray = field(default_factory=lambda: np.empty((0,2), dtype=np.float32))
    confidence: float | None = None
    reason: str = ''
    model_type: str = 'homography'
    # phase only estimates translation; it should not be silently compared as unconstrained homography
    precomputed_H: np.ndarray | None = None
    n_keypoints0: int | None = None
    n_keypoints1: int | None = None
    keypoints0: np.ndarray = field(default_factory=lambda: np.empty((0,2),dtype=np.float32))
    keypoints1: np.ndarray = field(default_factory=lambda: np.empty((0,2),dtype=np.float32))
