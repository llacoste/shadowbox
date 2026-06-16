"""Luminance-band slicer.

Converts the image to grayscale, partitions the brightness range into N bands,
and emits a binary mask per band. mask[i] = True where the source grayscale is
in band i or darker — so the back layer (i=0) is fullest and the front layer
(i=N-1) holds only the darkest features. This matches how laser-cut shadow
boxes physically stack: back is mostly solid, front carries the deepest cuts.

Three threshold modes pick where the band boundaries fall:
- "otsu": multi-level Otsu. Finds breakpoints that minimize within-class
  variance. Best general-purpose default — adapts to the image's histogram.
- "equal": uniform spacing across [min, max]. Predictable; good for synthetic
  inputs and depth-style outputs.
- "kmeans": 1-D k-means on luminance values. Good for posterized inputs with
  natural clusters that Otsu sometimes splits.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening
from skimage.filters import threshold_multiotsu

# Structuring-element radii per smoothing level. Opening (erode→dilate) drops
# tiny True specks; closing (dilate→erode) fills small False holes. Together
# they take a photograph's noisy threshold output from "hundreds of floating
# islands" to "a handful of cuttable shapes".
_MORPHOLOGY_BY_LEVEL: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (1, 1),
    2: (2, 2),
    3: (4, 3),
}


class LuminanceEngine:
    name: ClassVar[str] = "luminance"

    def slice(
        self,
        image: np.ndarray,
        n_layers: int,
        *,
        threshold_mode: str = "otsu",
        invert_layers: tuple[int, ...] = (),
        smoothing: int = 2,
    ) -> list[np.ndarray]:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        gray = _to_grayscale(image)
        thresholds = _thresholds(gray, n_layers, threshold_mode)
        # Layer ordering convention: index 0 = BACK (most material), index N-1
        # = FRONT (least material). `gray <= t` keeps dark pixels — so we sort
        # thresholds descending: the back layer's threshold is the loosest
        # (keeps everything except the brightest highlights), the front
        # layer's is the strictest (keeps only the darkest features).
        thresholds = sorted(thresholds, reverse=True)
        masks = [(gray <= t) for t in thresholds]

        opening_r, closing_r = _MORPHOLOGY_BY_LEVEL.get(smoothing, _MORPHOLOGY_BY_LEVEL[2])
        if opening_r or closing_r:
            masks = [_cleanup(m, opening_r, closing_r) for m in masks]

        invert_set = {i - 1 for i in invert_layers}
        return [np.logical_not(m) if i in invert_set else m for i, m in enumerate(masks)]


def _cleanup(mask: np.ndarray, opening_r: int, closing_r: int) -> np.ndarray:
    out = mask
    if opening_r > 0:
        out = binary_opening(out, structure=_disk(opening_r))
    if closing_r > 0:
        out = binary_closing(out, structure=_disk(closing_r))
    return out


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Coerce to HxW uint8 grayscale via ITU-R BT.601 luminance."""
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    pil = Image.fromarray(image).convert("L")
    return np.array(pil, dtype=np.uint8)


def _thresholds(gray: np.ndarray, n_layers: int, mode: str) -> list[int]:
    """Return N break points in [0, 255], sorted low-to-high."""
    if n_layers == 1:
        # Single layer: one mask covering everything dark enough to be material.
        # Use the midpoint; the user gets a silhouette.
        return [int(np.median(gray))]

    if mode == "equal":
        # n_layers bands → n_layers-1 interior breaks plus the top edge.
        # We need n_layers thresholds (one per layer), spaced through the range.
        lo, hi = int(gray.min()), int(gray.max())
        if hi <= lo:
            return [128] * n_layers
        step = (hi - lo) / n_layers
        return [round(lo + step * (i + 1)) for i in range(n_layers)]

    if mode == "otsu":
        # threshold_multiotsu wants N-1 classes for N thresholds, but we want
        # N thresholds total (one per layer). Use n_layers classes which gives
        # n_layers-1 interior breaks, and append the top of the range.
        if n_layers >= 2:
            try:
                breaks = threshold_multiotsu(gray, classes=n_layers).astype(int).tolist()
            except ValueError:
                # Falls through to equal-spacing on degenerate (flat) images.
                return _thresholds(gray, n_layers, "equal")
            return [*breaks, 255]
        return _thresholds(gray, n_layers, "equal")

    if mode == "kmeans":
        # 1-D k-means on luminance via scipy (already a scikit-image transitive dep).
        # Cheaper than pulling sklearn for one call.
        from scipy.cluster.vq import kmeans2

        flat = gray.reshape(-1).astype(np.float32)
        seeds = np.linspace(flat.min(), flat.max(), n_layers).astype(np.float32)
        centers, _ = kmeans2(flat, seeds, minit="matrix", seed=0)
        centers = sorted(int(c) for c in centers)
        midpoints = [round((centers[i] + centers[i + 1]) / 2) for i in range(len(centers) - 1)]
        return [*midpoints, 255]

    raise ValueError(f"Unknown threshold_mode {mode!r}. Expected: equal|otsu|kmeans")
