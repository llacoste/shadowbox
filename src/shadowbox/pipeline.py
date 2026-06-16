"""End-to-end image → layered SVGs orchestrator.

Single entry point used by both the CLI and the web app.

Stages:
1. preprocess  — optional bg removal + alpha-aware crop.
2. engine.slice — N binary masks back→front.
3. kerf-compensate masks (morphology in mask space — sub-pixel kerf is dropped
   with a warning).
4. vectorize    — masks → SVG path strings.
5. compose      — paths → per-layer SVG strings.
6. preflight    — analyze the result and attach warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from shadowbox import compose, engines, preflight, preprocess, vectorize
from shadowbox.project import ProjectSettings


@dataclass(frozen=True)
class SvgLayer:
    index: int  # 1-based
    total: int
    svg: str
    paths: list[str]


@dataclass(frozen=True)
class PipelineResult:
    layers: list[SvgLayer]
    warnings: list[preflight.Warning]
    canvas_px: tuple[int, int]
    canvas_mm: tuple[float, float]
    image_sha256: str = ""
    notes: list[str] = field(default_factory=list)


def process(
    image_bytes: bytes | BinaryIO,
    settings: ProjectSettings,
) -> PipelineResult:
    """Run the full pipeline."""
    raw = _read_bytes(image_bytes)
    image_sha = _sha256(raw)

    pil = Image.open(BytesIO(raw))
    pil.load()
    cleaned = preprocess.preprocess(
        pil,
        remove_bg=settings.remove_bg,
        auto_crop=True,
        smoothing=settings.smoothing,
    )
    if cleaned.mode not in ("L", "RGB"):
        cleaned = cleaned.convert("RGB") if cleaned.mode in ("RGBA", "LA", "P") else cleaned.convert("L")

    width_px, height_px = cleaned.size
    arr = np.array(cleaned)

    # Resolve output mm dimensions. In fit_aspect mode (default) the user's
    # width_mm/height_mm define a bounding box; the actual output preserves the
    # image's aspect ratio inside that box. Stretch mode honors the literal
    # dimensions even if it distorts.
    output_mm = _resolve_output_dims(
        image_px=(width_px, height_px),
        box_mm=(settings.width_mm, settings.height_mm),
        fit_aspect=settings.fit_aspect,
    )

    notes: list[str] = []
    if settings.fit_aspect and output_mm != (settings.width_mm, settings.height_mm):
        notes.append(
            f"Aspect-fit: output sized to {output_mm[0]:.1f}x{output_mm[1]:.1f}mm "
            f"to preserve the image's {width_px}x{height_px} aspect inside the "
            f"{settings.width_mm:.0f}x{settings.height_mm:.0f}mm bounding box."
        )

    engine = engines.get(settings.engine)
    masks = engine.slice(
        arr,
        settings.layers,
        threshold_mode=settings.threshold_mode,
        invert_layers=tuple(settings.invert_layers),
        smoothing=settings.smoothing,
    )

    kerf_radius_px = _kerf_radius_px(settings.kerf_mm, width_px, output_mm[0])
    if kerf_radius_px > 0:
        masks = [_apply_kerf(m, kerf_radius_px) for m in masks]
    elif settings.kerf_mm > 0:
        notes.append(
            f"Kerf {settings.kerf_mm}mm rounds to sub-pixel at this resolution; "
            f"applied no compensation. Render at a higher resolution to enforce."
        )

    min_feature_pixels = max(
        0,
        round(_mm_to_px(settings.min_feature_mm, width_px, output_mm[0])),
    )
    per_layer_paths = [vectorize.vectorize(m, min_feature_pixels=min_feature_pixels) for m in masks]

    layers: list[SvgLayer] = []
    for i, paths in enumerate(per_layer_paths, start=1):
        svg = compose.compose_layer(
            paths,
            canvas_px=(width_px, height_px),
            canvas_mm=output_mm,
            layer_index=i,
            total_layers=settings.layers,
            frame=settings.frame,
            engrave_number=settings.engrave_numbers,
            margin_mm=settings.margin_mm,
        )
        layers.append(SvgLayer(index=i, total=settings.layers, svg=svg, paths=paths))

    image_aspect = width_px / height_px
    warnings = preflight.analyze(
        per_layer_paths,
        canvas_px=(width_px, height_px),
        canvas_mm=output_mm,
        min_feature_mm=settings.min_feature_mm,
        image_aspect=image_aspect,
    )

    return PipelineResult(
        layers=layers,
        warnings=warnings,
        canvas_px=(width_px, height_px),
        canvas_mm=output_mm,
        image_sha256=image_sha,
        notes=notes,
    )


def _resolve_output_dims(
    *,
    image_px: tuple[int, int],
    box_mm: tuple[float, float],
    fit_aspect: bool,
) -> tuple[float, float]:
    if not fit_aspect:
        return box_mm
    w_px, h_px = image_px
    w_mm, h_mm = box_mm
    if w_px <= 0 or h_px <= 0:
        return box_mm
    image_aspect = w_px / h_px
    box_aspect = w_mm / h_mm
    if image_aspect > box_aspect:
        # Image is wider than the box → width-limited; shrink height.
        return w_mm, w_mm / image_aspect
    # Image is taller or equal → height-limited; shrink width.
    return h_mm * image_aspect, h_mm


def _read_bytes(source: bytes | BinaryIO) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    return source.read()


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _mm_to_px(mm: float, canvas_px: int, canvas_mm: float) -> float:
    if canvas_mm <= 0:
        return 0.0
    return mm * canvas_px / canvas_mm


def _kerf_radius_px(kerf_mm: float, canvas_px: int, canvas_mm: float) -> int:
    """Half the kerf width, in pixels — the structuring-element radius for morphology."""
    half = _mm_to_px(kerf_mm, canvas_px, canvas_mm) / 2.0
    return max(0, round(half))


def _apply_kerf(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Grow the kept-material region outward by `radius_px`.

    Cuts run along the boundary between material and air; the laser carves a
    `kerf_mm`-wide trough centered on that line. To end up with pieces at
    nominal dimensions, the material region must be `kerf/2` larger before
    cutting. Dilating the mask achieves that — interior holes shrink by the
    same amount, which is the right behavior for a shadow-box layer.
    """
    if radius_px <= 0:
        return mask
    structure = _disk(radius_px)
    return binary_dilation(mask, structure=structure)


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius
