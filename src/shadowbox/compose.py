"""Assemble a per-layer SVG from cut paths.

Output convention (matches LightBurn / xTool / Glowforge defaults):
- `#FF0000` (red) — interior cuts.
- `#0000FF` (blue) — outer registration frame. Distinct color so the laser
  software can order it to cut last (otherwise the workpiece falls free of
  the bed before its interior cuts complete).
- `#000000` (black) — engraved layer number.

SVG dimensions are in millimeters. The `viewBox` stays in pixel space so the
path coordinates emitted by the vectorizer don't need re-scaling.
"""

from __future__ import annotations

from collections.abc import Iterable

import svgwrite

CUT_COLOR = "#FF0000"
FRAME_COLOR = "#0000FF"
ENGRAVE_COLOR = "#000000"

# Stroke widths in mm. Laser software treats stroke-width as visual only, but
# some importers reject zero-width strokes, so we use a vanishingly small
# nonzero value.
CUT_STROKE_MM = 0.01


def compose_layer(
    paths: Iterable[str],
    *,
    canvas_px: tuple[int, int],
    canvas_mm: tuple[float, float],
    layer_index: int,
    total_layers: int,
    frame: bool = True,
    engrave_number: bool = True,
    margin_mm: float = 0.0,
) -> str:
    """Return an SVG string for one layer.

    `paths` is an iterable of SVG path `d` strings in pixel coordinates.
    `canvas_px` is the (width, height) in pixels (matches the source image).
    `canvas_mm` is the physical (width, height) in millimeters.
    `layer_index` is 1-based; the engraved label reads f"{layer_index}/{total_layers}".
    `margin_mm` insets the cuts (and frame) inward from the canvas edge by
    this many millimeters — gives the subject breathing room.
    """
    width_px, height_px = canvas_px
    width_mm, height_mm = canvas_mm
    if width_px <= 0 or height_px <= 0 or width_mm <= 0 or height_mm <= 0:
        raise ValueError(f"canvas dims must be positive; got px={canvas_px} mm={canvas_mm}")

    dwg = svgwrite.Drawing(
        size=(f"{width_mm:.4f}mm", f"{height_mm:.4f}mm"),
        viewBox=f"0 0 {width_px} {height_px}",
    )
    px_per_mm_x = width_px / width_mm
    px_per_mm_y = height_px / height_mm
    cut_stroke_units = CUT_STROKE_MM * px_per_mm_x

    margin_px_x = max(0.0, margin_mm) * px_per_mm_x
    margin_px_y = max(0.0, margin_mm) * px_per_mm_y
    inset_w = max(1.0, width_px - 2 * margin_px_x)
    inset_h = max(1.0, height_px - 2 * margin_px_y)
    scale_x = inset_w / width_px
    scale_y = inset_h / height_px

    cuts_attrs: dict[str, object] = {
        "id": "cuts",
        "stroke": CUT_COLOR,
        "fill": "none",
        "stroke_width": f"{cut_stroke_units:.4f}",
    }
    if margin_px_x > 0 or margin_px_y > 0:
        cuts_attrs["transform"] = f"translate({margin_px_x} {margin_px_y}) scale({scale_x} {scale_y})"
    cuts = dwg.g(**cuts_attrs)
    for d in paths:
        cuts.add(dwg.path(d=d))
    dwg.add(cuts)

    if frame:
        frame_group = dwg.g(
            id="frame",
            stroke=FRAME_COLOR,
            fill="none",
            stroke_width=f"{cut_stroke_units:.4f}",
        )
        frame_group.add(dwg.rect(insert=(margin_px_x, margin_px_y), size=(inset_w, inset_h)))
        dwg.add(frame_group)

    if engrave_number:
        # ~5mm character height, inside the frame's bottom-right corner.
        label_height_mm = 5.0
        label_margin_mm = 2.0 + max(0.0, margin_mm)
        font_size_units = label_height_mm * px_per_mm_x
        margin_units = label_margin_mm * px_per_mm_x
        label = dwg.g(id="layer-number", fill=ENGRAVE_COLOR, stroke="none")
        label.add(
            dwg.text(
                f"{layer_index}/{total_layers}",
                insert=(width_px - margin_units, height_px - margin_units),
                text_anchor="end",
                font_family="Arial, Helvetica, sans-serif",
                font_size=f"{font_size_units:.2f}",
            )
        )
        dwg.add(label)

    return dwg.tostring()
