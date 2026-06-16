"""Binary mask → SVG path strings via potrace.

The mask comes from an engine as a boolean ndarray. Output paths are in
image-pixel coordinates; the composer scales them to mm.
"""

from __future__ import annotations

import numpy as np
import potrace  # the `potracer` PyPI package installs as `potrace`


def vectorize(
    mask: np.ndarray,
    *,
    min_feature_pixels: int = 0,
) -> list[str]:
    """Trace `mask` into a list of SVG `d`-attribute path strings.

    `min_feature_pixels` is mapped to potrace's `turdsize` parameter: any
    connected region with fewer pixels than this is suppressed before the
    curve fit, which keeps laser-uncuttable specks out of the output.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)

    height, width = mask.shape

    # We trace the INVERTED mask: potrace's foreground/background convention
    # gives cleaner shape outlines that way for our material-as-True semantics.
    # In particular, shapes that touch the canvas edge get traced as the shape
    # outline alone, instead of "shape merged with the canvas frame".
    bitmap = potrace.Bitmap(~mask)
    path = bitmap.trace(
        turdsize=max(0, min_feature_pixels),
        # Defaults below match potrace's "Crisp" / smooth-but-faithful preset.
        # We expose only min_feature_size as a knob in v1; the rest are sane
        # for laser output and noisier on real photos isn't a win for shadow
        # boxes anyway.
        alphamax=1.0,
        opticurve=True,
        opttolerance=0.2,
    )

    paths: list[str] = []
    for curve in path:
        if _spans_canvas(curve, width, height):
            # potrace emits the canvas perimeter as a path whenever a
            # foreground region touches the image edge. Our composer draws
            # the registration frame independently, so this duplicates would
            # confuse downstream software.
            continue
        segments: list[str] = []
        start = curve.start_point
        segments.append(f"M{start.x:.3f} {start.y:.3f}")
        for segment in curve.segments:
            if segment.is_corner:
                c = segment.c
                end = segment.end_point
                segments.append(f"L{c.x:.3f} {c.y:.3f}L{end.x:.3f} {end.y:.3f}")
            else:
                c1, c2, end = segment.c1, segment.c2, segment.end_point
                segments.append(f"C{c1.x:.3f} {c1.y:.3f} {c2.x:.3f} {c2.y:.3f} {end.x:.3f} {end.y:.3f}")
        segments.append("Z")
        paths.append("".join(segments))
    return paths


def _spans_canvas(curve, width: int, height: int, tol: float = 0.5) -> bool:
    """True if `curve` traces (approximately) the full canvas perimeter."""
    xs: list[float] = [curve.start_point.x]
    ys: list[float] = [curve.start_point.y]
    for seg in curve.segments:
        xs.append(seg.end_point.x)
        ys.append(seg.end_point.y)
    return min(xs) <= tol and min(ys) <= tol and max(xs) >= width - tol and max(ys) >= height - tol


def mm_to_pixels(mm: float, canvas_px: int, canvas_mm: float) -> float:
    """Convert a millimeter value to image-pixel units given the canvas scale."""
    if canvas_mm <= 0:
        raise ValueError("canvas_mm must be > 0")
    return mm * canvas_px / canvas_mm
