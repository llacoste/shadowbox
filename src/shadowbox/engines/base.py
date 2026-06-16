"""Slicing engine Protocol.

An engine turns a preprocessed image into N binary masks, one per laser-cut
layer, ordered back-to-front. mask[i] is True where material remains on
layer i.
"""

from __future__ import annotations

from typing import ClassVar, Protocol

import numpy as np


class SlicingEngine(Protocol):
    name: ClassVar[str]

    def slice(
        self,
        image: np.ndarray,
        n_layers: int,
        *,
        threshold_mode: str = "otsu",
        invert_layers: tuple[int, ...] = (),
        smoothing: int = 2,
    ) -> list[np.ndarray]:
        """Return N binary masks, back-to-front.

        - `image` is HxW grayscale (uint8) or HxWxC (will be converted).
        - `threshold_mode` is engine-specific. Engines that don't support a
          mode should raise ValueError on anything but their default.
        - `invert_layers` is a tuple of 1-based layer indices whose mask
          should be flipped before return.
        - `smoothing` ∈ {0..3} hints at how aggressively the engine should
          clean up tiny artifacts (specks, holes). Engines may ignore it.
        """
        ...
