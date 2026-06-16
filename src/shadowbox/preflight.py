"""Post-vectorization warnings about cuttability.

Run after vectorization, before download. Surfaces issues the user can act on:
features the laser can't cut, layers that are wasted material, islands that
will fall out of the workpiece when cut, and aspect-ratio mismatches that
would distort the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Severity = Literal["info", "warn", "error"]


@dataclass(frozen=True)
class Warning:
    layer: int  # 1-based
    code: str
    severity: Severity
    message: str


def analyze(
    layers: list[list[str]],
    *,
    canvas_px: tuple[int, int],
    canvas_mm: tuple[float, float],
    min_feature_mm: float,
    image_aspect: float,
) -> list[Warning]:
    """Examine each layer's path set and return a flat list of warnings."""
    warnings: list[Warning] = []
    width_px, _height_px = canvas_px
    width_mm, height_mm = canvas_mm
    px_per_mm = width_px / width_mm
    min_feature_px = min_feature_mm * px_per_mm
    output_aspect = width_mm / height_mm

    # Aspect mismatch is a per-image concern, not per-layer.
    if abs(output_aspect - image_aspect) / max(image_aspect, 1e-6) > 0.05:
        warnings.append(
            Warning(
                layer=0,
                code="aspect_mismatch",
                severity="warn",
                message=(
                    f"Image aspect {image_aspect:.3f} differs from output aspect "
                    f"{output_aspect:.3f} by >5%. The output will be visibly stretched."
                ),
            )
        )

    for idx, paths in enumerate(layers, start=1):
        if not paths:
            warnings.append(
                Warning(
                    layer=idx,
                    code="solid_layer",
                    severity="warn",
                    message=(
                        "Layer has no interior cuts — it'll be a solid sheet. "
                        "Consider reducing layer count or switching threshold mode."
                    ),
                )
            )
            continue

        bboxes = [_bbox(p) for p in paths]
        tiny = [b for b in bboxes if b is not None and max(b[2], b[3]) < min_feature_px]
        if tiny:
            warnings.append(
                Warning(
                    layer=idx,
                    code="tiny_features",
                    severity="warn",
                    message=(
                        f"{len(tiny)} features smaller than {min_feature_mm:.2f}mm. "
                        f"The laser may not cut them reliably — try raising --min-feature-mm."
                    ),
                )
            )

        islands = _detect_floating_islands(bboxes)
        if islands:
            warnings.append(
                Warning(
                    layer=idx,
                    code="floating_islands",
                    severity="info",
                    message=(
                        f"{islands} closed region(s) appear to be fully enclosed by another "
                        f"on this layer. They will drop free when cut — that's usually fine for "
                        f"a shadow box, but worth knowing."
                    ),
                )
            )

    return warnings


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _bbox(path_d: str) -> tuple[float, float, float, float] | None:
    """Coarse axis-aligned bbox from path 'd' string — extracts every numeric pair.

    Bezier control points pad the bbox slightly, but for the cuttability check
    that's fine: we want to be conservative about what counts as "tiny".
    """
    nums = [float(n) for n in _NUM.findall(path_d)]
    if len(nums) < 2:
        return None
    xs = nums[0::2]
    ys = nums[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return (x0, y0, x1 - x0, y1 - y0)


def _detect_floating_islands(bboxes: list[tuple[float, float, float, float] | None]) -> int:
    """Count bboxes fully contained inside another bbox on the same layer.

    bbox-containment overcounts (a path can be inside another's bbox without
    being inside the path itself), so we mark this `info` severity. A real
    polygon-in-polygon check would use Shapely on point-sampled paths.
    """
    boxes = [b for b in bboxes if b is not None]
    contained = 0
    for i, inner in enumerate(boxes):
        ix, iy, iw, ih = inner
        for j, outer in enumerate(boxes):
            if i == j:
                continue
            ox, oy, ow, oh = outer
            if ow < iw or oh < ih:
                continue
            if ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih:
                contained += 1
                break
    return contained
